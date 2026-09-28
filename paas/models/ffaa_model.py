"""FFAA MLLM+MIDS component for PAAS v4 -- Qwen3.5-4B (IN-PROCESS vLLM) + from-scratch MIDS 4-class head.

SINGLE-VENV design (transformers 5.13 + vLLM): the whole ensemble runs in ONE process on ONE venv.
The Qwen3.5-4B MLLM is built IN-PROCESS with vLLM's ``LLM`` (no server, no HTTP), and the MIDS 4-class
head (CLIP-ViT-L/14-336 + T5-base) loads in the same interpreter -- its tf4.37-trained weights are
remapped to the tf5 CLIPVisionModel layout on load (paas.compat.remap_clip_state_dict), so no
retraining is needed. This removes the two-process HTTP split that v4_inf needed only because its CLIP
stack was pinned to tf4.37.2.

FFAA answers use the 3-PASS CONDITIONAL protocol (base -> parse -> real/fake-conditioned) with
enable_thinking=False (Qwen3.5's empty <think></think> matches the training targets -> clean FFAA JSON).
A batch of N frames is a SINGLE ``llm.generate([...N...])`` call -- vLLM continuous-batches internally,
so no thread pool / concurrent HTTP is needed. Uniform ``score_frames`` output preserved:
  {"fake": float|None, "analysis": "real"/"fake"|None, "match": float|None,
   "forgery_type": str|None, "answer": str|None, "error": str|None}.

IMPORTANT: vLLM must claim CUDA before anything else in the process, so this model builds the LLM
FIRST (before the MIDS/CLIP tensors touch the GPU), and the pipeline constructs FFAA before the other
detectors.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
from PIL import Image

CONDITION = "This is a _ human face. What evidence do you have?"


def _NM(n_answers: int):
    return {3: (1, 1), 2: (0, 1), 1: (0, 0)}.get(n_answers, (0, 0))


class FFAAModel:
    name = "ffaa"

    def __init__(self, cfg, device: str = "cuda:0", cache: Optional[dict] = None):
        # ---- 1) build the IN-PROCESS vLLM MLLM FIRST (must claim CUDA before torch) ----
        import transformers
        from vllm import LLM, SamplingParams
        self.proc = transformers.AutoProcessor.from_pretrained(cfg.qwen_dir)
        self.llm = LLM(model=cfg.qwen_dir, dtype="bfloat16",
                       gpu_memory_utilization=getattr(cfg, "qwen_gpu_mem", 0.45),
                       max_model_len=getattr(cfg, "qwen_max_model_len", 4096),
                       limit_mm_per_prompt={"image": 1},
                       mm_processor_kwargs={"size": {"shortest_edge": 65536,
                                                     "longest_edge": getattr(cfg, "qwen_max_pixels", 451584)}})
        self.sp = SamplingParams(temperature=0, max_tokens=getattr(cfg, "max_new_tokens", 512))

        # ---- 2) now the MIDS 4-class head (from-scratch), same process/venv ----
        import torch
        from transformers import CLIPProcessor, T5Tokenizer
        from mids.mids_arch import MIDS
        from mids.selector import make_decision, make_decision_batch
        from utils.file_utils import mask_result, decode_response
        from paas.compat import remap_clip_state_dict, T5_MAX_LEN

        self.t = torch
        self.F = torch.nn.functional
        self.cfg = cfg
        self._make_decision = make_decision
        self._make_decision_batch = make_decision_batch
        self._mask_result = mask_result
        self._decode = decode_response
        self.cache = cache
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._t5_max = T5_MAX_LEN

        self.base_prompt = (open(cfg.prompt_file).readline().strip()
                            if getattr(cfg, "prompt_file", None) else cfg.prompt)
        self.p_fake = CONDITION.replace("_", "fake")
        self.p_real = CONDITION.replace("_", "real")

        self.t5_tokenizer = T5Tokenizer.from_pretrained(cfg.t5_path, use_fast=False, legacy=False)
        self.clip_processor = CLIPProcessor.from_pretrained(cfg.clip_path)
        self.mids = MIDS(768, image_model_path=cfg.clip_path, text_model_path=cfg.t5_path)
        sd = self.mids.state_dict()
        ft = remap_clip_state_dict(torch.load(cfg.mids_path, map_location="cpu"))
        sd.update(ft); self.mids.load_state_dict(sd)
        self.mids = self.mids.to(self.device).eval()

        self._whole_frame = bool(getattr(cfg, "whole_frame", False))
        self._tmpl_cache = {}   # fixed chat templates (base / real / fake-conditioned)

    # ------------------------------------------------------------------ MLLM (in-process)
    def _tmpl(self, q: str) -> str:
        t = self._tmpl_cache.get(q)
        if t is None:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            t = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
            self._tmpl_cache[q] = t
        return t

    def _gen(self, prompts: List[str], imgs: List[Image.Image]) -> List[str]:
        """prompts[i] (text) + imgs[i] -> answer text[i]. The whole list is ONE llm.generate call;
        vLLM continuous-batches all N internally -- no HTTP, no thread pool."""
        reqs = [{"prompt": self._tmpl(q), "multi_modal_data": {"image": im}}
                for q, im in zip(prompts, imgs)]
        outs = self.llm.generate(reqs, self.sp, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

    def score_frames(self, rgb_list: List[np.ndarray], batch_size: int = 64,
                     keys: Optional[List[str]] = None) -> List[dict]:
        pil = [Image.fromarray(r).convert("RGB") for r in rgb_list]
        if self.cache is None:
            return self._score_live(pil, batch_size)
        out: List[Optional[dict]] = [None] * len(pil)
        hit_idx, hit_imgs, hit_recs, miss_idx, miss_imgs = [], [], [], [], []
        for i, img in enumerate(pil):
            key = keys[i] if keys else None
            rec = self.cache.get(os.path.abspath(key)) if key else None
            if rec:
                hit_idx.append(i); hit_imgs.append(img); hit_recs.append(rec)
            else:
                miss_idx.append(i); miss_imgs.append(img)
        if hit_imgs:
            for i, r in zip(hit_idx, self._score_cached(hit_imgs, hit_recs, batch_size)):
                out[i] = r
        if miss_imgs:
            for i, r in zip(miss_idx, self._score_live(miss_imgs, batch_size)):
                out[i] = r
        return out

    def _score_live(self, pil: List[Image.Image], batch_size: int) -> List[dict]:
        """3-pass conditional generation via the in-process vLLM, then MIDS-score each frame's 3 answers."""
        out: List[dict] = []
        for i in range(0, len(pil), batch_size):
            chunk = pil[i:i + batch_size]
            try:
                a1 = self._gen([self.base_prompt] * len(chunk), chunk)
                p2, p3 = [], []
                for t in a1:
                    rj, _ = self._decode(t)
                    if len(rj) != 5 or rj.get("Analysis result", "").lower() == "real":
                        p2.append(self.p_fake); p3.append(self.p_real)
                    else:
                        p2.append(self.p_real); p3.append(self.p_fake)
                a2 = self._gen(p2, chunk); a3 = self._gen(p3, chunk)
            except Exception as e:
                out.extend({"fake": None, "analysis": None, "match": None, "forgery_type": None,
                            "answer": None, "error": f"generate: {e}"} for _ in chunk)
                continue
            for img, x1, x2, x3 in zip(chunk, a1, a2, a3):
                out.append(self._score_one(img, [x1, x2, x3]))
        return out

    def _pixels(self, imgs):
        """MIDS-head pixels. whole_frame=True letterboxes the WHOLE frame (matching a head
        trained that way); otherwise the legacy CLIPProcessor resize+center-crop is used."""
        if not self._whole_frame:
            return self.clip_processor(images=imgs, return_tensors="pt")["pixel_values"]
        from paas.preprocess import letterbox_to_tensor
        return self.t.stack([letterbox_to_tensor(np.asarray(im.convert("RGB")), 336) for im in imgs])

    def _score_cached(self, imgs: List[Image.Image], recs: List[list], batch_size: int) -> List[dict]:
        out: List[Optional[dict]] = [None] * len(imgs)
        buckets = defaultdict(list)
        for i, rec in enumerate(recs):
            buckets[len(rec)].append(i)
        for nans, idxs in buckets.items():
            N, M = _NM(nans)
            for s in range(0, len(idxs), batch_size):
                grp = idxs[s:s + batch_size]
                bimgs = [imgs[k] for k in grp]
                contents = [a["content"] for k in grp for a in recs[k]]
                results = [(a.get("result") or "fake").lower() for k in grp for a in recs[k]]
                try:
                    with self.t.inference_mode():
                        pix = self._pixels(bimgs).to(self.device)
                        ans_ids = self.t5_tokenizer(contents, return_tensors="pt", padding="longest",
                                                    max_length=self._t5_max, truncation=True)
                        ans_ids = {k: v.to(self.device) for k, v in ans_ids.items()}
                        logits = self.mids(ans_ids, pix, None, len(bimgs), N, M)["logits"]
                        scores = self.F.softmax(logits, dim=2)
                        bidx, preds, matches, forgeries = self._make_decision_batch(results, scores, chunk_size=nans)
                    for j, k in enumerate(grp):
                        analysis = "real" if int(preds[j]) == 0 else "fake"
                        res_k = [(a.get("result") or "fake").lower() for a in recs[k]]
                        difficulty = "easy" if len(set(res_k)) == 1 else "hard"  # 3-answer verdict agreement
                        out[k] = {"fake": float(forgeries[j]), "analysis": analysis, "match": float(matches[j]),
                                  "forgery_type": "real" if analysis == "real" else None,
                                  "answer": recs[k][int(bidx[j])]["content"],
                                  "difficulty": difficulty, "error": None}
                except Exception as e:
                    for k in grp:
                        out[k] = {"fake": None, "analysis": None, "match": None, "forgery_type": None,
                                  "answer": None, "error": f"cache-score: {e}"}
        return out

    def _score_one(self, img, answers: List[str]) -> dict:
        try:
            answers_result, processed = [], []
            for a in answers:
                masked, res = self._mask_result(a)
                processed.append(masked); answers_result.append(res)
            N, M = _NM(len(answers))
            with self.t.inference_mode():
                pix = self._pixels([img]).to(self.device)
                ans_ids = self.t5_tokenizer(processed, return_tensors="pt", padding="longest",
                                            max_length=self._t5_max, truncation=True)
                ans_ids = {k: v.to(self.device) for k, v in ans_ids.items()}
                logits = self.mids(ans_ids, pix, None, 1, N, M)["logits"]
                scores = self.F.softmax(logits, dim=2).squeeze(0)
                best_idx, pred, match, forgery = self._make_decision(answers_result, scores)
            analysis = "real" if int(pred) == 0 else "fake"
            ftype = None
            try:
                bj, _ = self._decode(answers[best_idx])
                ftype = (bj.get("Forgery type") or None) if analysis == "fake" else "real"
            except Exception:
                pass
            difficulty = "easy" if len(set(answers_result)) == 1 else "hard"  # 3-answer verdict agreement
            return {"fake": float(forgery), "analysis": analysis, "match": float(match),
                    "forgery_type": ftype, "answer": answers[best_idx],
                    "difficulty": difficulty, "error": None}
        except Exception as e:
            return {"fake": None, "analysis": None, "match": None,
                    "forgery_type": None, "answer": None, "error": f"score: {e}"}
