#!/usr/bin/env python
"""Generate the MIDS training/eval json with the (finetuned) Qwen3-VL MLLM via vLLM.

Faithful port of FFAA's make_mids_dataset_from_folder_batch.py 3-pass protocol:
  pass 1: base prompt ("Is it real or fake? Why?")
  pass 2/3: "This is a {fake|real} human face. What evidence do you have?" -- the order depends
            on pass-1's parsed 'Analysis result' (opposite first), exactly like FFAA.
Record format (identical to FFAA):
  {"id", "image", "cls_label" (0 real / 1 fake), "answers": [{content, result, label=2*cls+claim} x3]}
Reals whose pass-1 'Image description' reports quality low/poor are dropped (FFAA quality gate).
Bad-format answers -> image recorded in <out>.errors.json (no archiving), sample skipped.

Sharding: --which-part i --n-divided N processes every N-th image; shards are merged by the
caller. --resume reloads an existing shard and skips its images (safe re-run).

Run in the PROJECT VENV (vLLM + transformers>=5):
  CUDA_VISIBLE_DEVICES=g ./venv/bin/python qwen/gen_mids_vllm.py --model runs/qwen_merged \
      --images-json /datasets/.../mids_first_half.json --out temp_qwen/mids_qwen_train_g.json \
      --which-part g --n-divided 4
"""
import argparse
import json
import os
import sys

from PIL import Image

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffaa")  # v4: utils/ + playground/ live under ffaa/
sys.path.insert(0, _ROOT)
from get_label import get_label_all                       # noqa: E402
from utils.file_utils import decode_response              # noqa: E402

CONDITION = "This is a _ human face. What evidence do you have?"


def recover(args, build_record):
    """ONE-BY-ONE recovery of format-error images from a previous batch run.

    Loads --recover (the .errors.json), reprocesses each 'format' entry individually
    (batch of 1, never batched): attempt 0 greedy, further attempts sampled
    (temperature 0.8) so a deterministic bad format can escape. Recovered records are
    appended to --out; the errors file is rewritten with only the still-failing and
    non-format entries.
    """
    import transformers
    from vllm import LLM, SamplingParams

    errs = json.load(open(args.recover))
    todo = [e for e in errs if e.get("error") == "format"]
    keep = [e for e in errs if e.get("error") != "format"]
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {r["image"] for r in results}
    todo = [e for e in todo if os.path.abspath(e["image"]) not in done]
    print(f"[recover] {len(todo):,} format-error images to retry one-by-one "
          f"({len(keep):,} non-format entries kept)")
    if not todo:
        json.dump(keep, open(args.recover, "w")); return

    prompt_base = open(os.path.join(_ROOT, "playground", "prompts.txt")).readline().strip()
    proc = transformers.AutoProcessor.from_pretrained(args.model)

    def chat(q):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)  # Qwen3.5 thinking off (ignored by Qwen3-VL)

    tmpl_base = chat(prompt_base)
    tmpl_fake = chat(CONDITION.replace("_", "fake"))
    tmpl_real = chat(CONDITION.replace("_", "real"))

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              max_model_len=4096, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"size": {"shortest_edge": 65536, "longest_edge": args.max_pixels}})

    def gen1(text, img, sp):
        out = llm.generate([{"prompt": text, "multi_modal_data": {"image": img}}], sp)
        return out[0].outputs[0].text.strip()

    still = []
    n_rec = 0
    for k, e in enumerate(todo):
        p = e["image"]
        lab = get_label_all(p)
        if lab < 0:
            still.append(e); continue
        cls_label = 0 if lab == 0 else 1
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            still.append({"image": p, "error": "unreadable"}); continue
        recovered = False
        for attempt in range(max(1, args.retries)):
            sp = (SamplingParams(temperature=0, max_tokens=args.max_tokens) if attempt == 0 else
                  SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.max_tokens,
                                 seed=1000 + attempt))
            a1 = gen1(tmpl_base, img, sp)
            rj, _ = decode_response(a1)
            if len(rj) != 5 or rj.get("Analysis result", "").lower() == "real":
                a2 = gen1(tmpl_fake, img, sp); a3 = gen1(tmpl_real, img, sp)
            else:
                a2 = gen1(tmpl_real, img, sp); a3 = gen1(tmpl_fake, img, sp)
            status, payload = build_record(p, cls_label, (a1, a2, a3))
            if status == "ok":
                results.append(payload); n_rec += 1; recovered = True
                break
            if payload == "lowq_real":                 # deliberate drop, not a retryable error
                still.append({"image": p, "error": "lowq_real"}); recovered = True
                break
        if not recovered:
            print(f"Error Format (unrecovered after {max(1, args.retries)} attempts): {p}", flush=True)
            still.append({"image": p, "error": "format"})
        if (k + 1) % 50 == 0 or k + 1 == len(todo):
            json.dump(results, open(args.out, "w"))
            print(f"[recover] {k + 1:,}/{len(todo):,} | recovered {n_rec:,}", flush=True)

    json.dump(results, open(args.out, "w"))
    json.dump(keep + still, open(args.recover, "w"))
    print(f"[recover] DONE: recovered {n_rec:,}/{len(todo):,}; "
          f"{len([s for s in still if s.get('error') == 'format']):,} still unparseable")


def load_image_list(args):
    """Unique (path, cls_label) list, sharded. cls_label: 0 real, 1 fake; UNKNOWN dropped."""
    paths = []
    if args.images_json:
        seen = set()
        for r in json.load(open(args.images_json)):
            p = r.get("image")
            if p and p not in seen:
                seen.add(p); paths.append(p)
    else:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for dp, _, files in os.walk(args.input_dir):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in exts:
                    paths.append(os.path.join(dp, fn))
    paths.sort()
    items = []
    for i, p in enumerate(paths):
        if i % args.n_divided != args.which_part:
            continue
        lab = get_label_all(p)
        if lab < 0:
            continue
        items.append((p, 0 if lab == 0 else 1))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images-json", default=None)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which-part", type=int, default=0)
    ap.add_argument("--n-divided", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-pixels", type=int, default=451584)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recover", default=None,
                    help="errors json from a previous run: reprocess its FORMAT-error images "
                         "ONE-BY-ONE (batch of 1) with sampling retries; recovered records are "
                         "appended to --out and the errors file is rewritten")
    ap.add_argument("--retries", type=int, default=5,
                    help="recovery attempts per image (attempt 0 greedy, then sampled)")
    args = ap.parse_args()
    assert args.images_json or args.input_dir or args.recover

    import transformers
    from vllm import LLM, SamplingParams

    prompt_base = open(os.path.join(_ROOT, "playground", "prompts.txt")).readline().strip()
    proc = transformers.AutoProcessor.from_pretrained(args.model)

    def chat(q):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)  # Qwen3.5 thinking off (ignored by Qwen3-VL)

    tmpl_base = chat(prompt_base)
    tmpl_fake = chat(CONDITION.replace("_", "fake"))
    tmpl_real = chat(CONDITION.replace("_", "real"))

    def build_record(p, cls_label, answers3):
        """FFAA record from 3 raw answers -> ("ok", record) | ("err", reason)."""
        contents, res, claims = [], [], []
        for a in answers3:
            rj, _ = decode_response(a)
            if all(k in rj for k in ("Analysis result", "Image description", "Forgery reasoning")):
                r = rj["Analysis result"].lower()
                contents.append("Image description: %s\nForgery reasoning: %s"
                                % (rj["Image description"], rj["Forgery reasoning"]))
                res.append(r); claims.append(0 if r == "real" else 1)
            else:
                return "err", "format"
        if cls_label == 0:                              # FFAA quality gate on reals
            rj, _ = decode_response(answers3[0])
            q = ""
            for part in rj.get("Image description", "").split(","):
                part = part.strip()
                if "-" in part:
                    k, v = part.split("-", 1)
                    if k.strip() == "quality":
                        q = v.strip().lower()
            # if q in ("low", "poor"):
            if q in ("poor"): #zzzzzz
                return "err", "lowq_real"
        return "ok", {
            "id": os.path.basename(p),
            "image": os.path.abspath(p),
            "cls_label": cls_label,
            "answers": [{"content": contents[k], "result": res[k],
                         "label": 2 * cls_label + claims[k]} for k in range(3)],
        }

    if args.recover:
        recover(args, build_record)
        return

    items = load_image_list(args)
    done, results, errors = set(), [], []
    if os.path.exists(args.out):                        # resume
        results = json.load(open(args.out))
        done = {r["image"] for r in results}
        print(f"[resume] {len(done):,} images already in {args.out}")
    items = [(p, c) for p, c in items if p not in done]
    if args.limit:
        items = items[:args.limit]
    print(f"[shard {args.which_part}/{args.n_divided}] {len(items):,} images to generate")
    if not items:
        json.dump(results, open(args.out, "w")); return

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              max_model_len=4096, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"size": {"shortest_edge": 65536, "longest_edge": args.max_pixels}})
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    def gen(prompt_texts, images):
        reqs = [{"prompt": t, "multi_modal_data": {"image": im}} for t, im in zip(prompt_texts, images)]
        outs = llm.generate(reqs, sp)
        return [o.outputs[0].text.strip() for o in outs]

    n_saved = 0
    for s in range(0, len(items), args.batch_size):
        chunk = items[s:s + args.batch_size]
        imgs, ok = [], []
        for p, c in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB")); ok.append((p, c))
            except Exception:
                errors.append({"image": p, "error": "unreadable"})
        if not imgs:
            continue
        a1 = gen([tmpl_base] * len(imgs), imgs)
        # branch pass 2/3 on pass-1 verdict (FFAA logic)
        p2, p3 = [], []
        for t in a1:
            rj, _ = decode_response(t)
            if len(rj) != 5 or rj.get("Analysis result", "").lower() == "real":
                p2.append(tmpl_fake); p3.append(tmpl_real)
            else:
                p2.append(tmpl_real); p3.append(tmpl_fake)
        a2 = gen(p2, imgs)
        a3 = gen(p3, imgs)

        for (p, cls_label), answers3 in zip(ok, zip(a1, a2, a3)):
            status, payload = build_record(p, cls_label, answers3)
            if status == "ok":
                results.append(payload)
            else:
                if payload == "format":
                    print(f"Error Format: {p}", flush=True)
                errors.append({"image": p, "error": payload})
        if len(results) - n_saved >= args.save_every:
            n_saved = len(results)
            json.dump(results, open(args.out, "w"))
            json.dump(errors, open(args.out + ".errors.json", "w"))
        print(f"[shard {args.which_part}] {min(s + args.batch_size, len(items)):,}/{len(items):,} "
              f"| kept {len(results):,} | errs {len(errors):,}", flush=True)

    json.dump(results, open(args.out, "w"))
    json.dump(errors, open(args.out + ".errors.json", "w"))
    print(f"[shard {args.which_part}] DONE: kept {len(results):,}, errors {len(errors):,} -> {args.out}")


if __name__ == "__main__":
    main()
