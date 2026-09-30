#!/usr/bin/env python
"""Functional evaluation of the (merged) MLLM on eFFAA_ext_eval: the metrics that matter
for the FFAA pipeline, rather than LM loss.

  * parse rate       : % of answers decode_response can parse (5-field format)
  * verdict accuracy : % where 'Analysis result' matches the ground-truth answer's verdict
                       (per-class real/fake breakdown included)

  CUDA_VISIBLE_DEVICES=0 ./venv/bin/python qwen/eval_effaa.py --model runs/qwen_merged \
      --eval /datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json --limit 5000
"""
import argparse
import json
import os
import re
import sys

from PIL import Image

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffaa")  # v4: utils/ + playground/ live under ffaa/
sys.path.insert(0, _ROOT)
from utils.file_utils import decode_response          # noqa: E402


def gt_verdict(answer_text):
    m = re.search(r"Analysis result:\s*(\w+)", answer_text)
    return m.group(1).lower() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--image-root", default="/datasets/newout")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-pixels", type=int, default=451584)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--which-part", type=int, default=0, help="shard index (multi-GPU gate)")
    ap.add_argument("--n-divided", type=int, default=1, help="number of shards")
    ap.add_argument("--json-out", default=None, help="write counters json (for shard merging)")
    ap.add_argument("--compliance", action="store_true",
                    help="also measure conditioned-answer compliance (the FFAA pass-2/3 prompts): "
                         "does 'This is a {real|fake} human face...' yield a matching verdict?")
    args = ap.parse_args()

    import transformers
    from vllm import LLM, SamplingParams

    data = json.load(open(args.eval))
    items = []
    for r in data:
        conv = r.get("conversations") or []
        if len(conv) < 2 or "image" not in r:
            continue
        gt = gt_verdict(conv[1]["value"])
        if gt in ("real", "fake"):
            q = conv[0]["value"].replace("<image>", "").strip()
            items.append((os.path.join(args.image_root, r["image"]), q, gt))
    if args.limit and args.limit < len(items):
        stride = len(items) / args.limit
        items = [items[int(i * stride)] for i in range(args.limit)]
    items = items[args.which_part::args.n_divided]          # shard for multi-GPU gating
    print(f"eval items (shard {args.which_part}/{args.n_divided}): {len(items):,}")

    proc = transformers.AutoProcessor.from_pretrained(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              max_model_len=4096, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"size": {"shortest_edge": 65536, "longest_edge": args.max_pixels}})
    sp = SamplingParams(temperature=0, max_tokens=512)

    n = ok_fmt = ok_verdict = 0
    per = {"real": [0, 0], "fake": [0, 0]}   # gt -> [correct, total]
    for s in range(0, len(items), args.batch_size):
        chunk = items[s:s + args.batch_size]
        reqs, metas = [], []
        for p, q, gt in chunk:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            reqs.append({"prompt": proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                            enable_thinking=False),
                         "multi_modal_data": {"image": img}})
            metas.append(gt)
        outs = llm.generate(reqs, sp)
        for gt, o in zip(metas, outs):
            n += 1
            rj, _ = decode_response(o.outputs[0].text.strip())
            if all(k in rj for k in ("Analysis result", "Image description", "Forgery reasoning")):
                ok_fmt += 1
                pred = rj["Analysis result"].lower()
                per[gt][1] += 1
                if pred == gt:
                    ok_verdict += 1; per[gt][0] += 1
        print(f"{min(s + args.batch_size, len(items)):,}/{len(items):,} | "
              f"parse {ok_fmt / max(n, 1) * 100:.2f}% | verdict {ok_verdict / max(ok_fmt, 1) * 100:.2f}%",
              flush=True)

    comply = {"real": [0, 0], "fake": [0, 0]}
    if args.compliance:
        cond = "This is a _ human face. What evidence do you have?"
        for target in ("real", "fake"):
            q = cond.replace("_", target)
            msg = proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for s in range(0, len(items), args.batch_size):
                chunk = items[s:s + args.batch_size]
                reqs = []
                for p, _, _ in chunk:
                    try:
                        reqs.append({"prompt": msg, "multi_modal_data": {"image": Image.open(p).convert("RGB")}})
                    except Exception:
                        continue
                for o in llm.generate(reqs, sp):
                    rj, _ = decode_response(o.outputs[0].text.strip())
                    if "Analysis result" in rj:
                        comply[target][1] += 1
                        if rj["Analysis result"].lower() == target:
                            comply[target][0] += 1
    print("\n==== eFFAA_ext_eval functional metrics ====")
    print(f"n={n:,}  parse-rate={ok_fmt / max(n, 1) * 100:.2f}%  "
          f"verdict-acc={ok_verdict / max(ok_fmt, 1) * 100:.2f}%")
    for k, (c, t) in per.items():
        print(f"  {k}: {c}/{t} = {c / max(t, 1) * 100:.2f}%")
    if args.compliance:
        for k, (c, t) in comply.items():
            print(f"  comply[{k}]: {c}/{t} = {c / max(t, 1) * 100:.2f}%")
    if args.json_out:
        json.dump({"n": n, "ok_fmt": ok_fmt, "ok_verdict": ok_verdict, "per": per,
                   "comply": comply}, open(args.json_out, "w"))


if __name__ == "__main__":
    main()
