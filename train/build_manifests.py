#!/usr/bin/env python3
"""Build detector trainset manifests from an image DIRECTORY (or an existing manifest).

Truth comes from ``get_label_all`` on the PATH -- no MLLM, no GPU. MAKEUP folds into PAD and
UNKNOWN is dropped, matching mids_plus/scripts/build_mids9.py (which produced the deployed A2_9c)
and config/ensemble9.json ("label = 3*true + claim ; true/claim in {0 real,1 pad,2 deepfake} ;
pad includes makeup").

  --mode images : [{"image", "cls_label"}]                      -> GSD / SeLop (they derive the
                  label from the path themselves; they only need the image list)
  --mode mids9  : [{"image", "cls_label", "answers":[3 x {content, result, label}]}]
                  the 9-class set. The three claim anchors are FIXED TEXT TEMPLATES (the 9-class
                  branch is MLLM-free) read from config/ensemble9.json so they never drift from
                  what the deployed ensemble feeds at inference.

  python train/build_manifests.py --images <dir|json> --mode mids9 --out runs/data/mids9_train.json
"""
import argparse, json, os, sys, collections

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "qwen"))
from get_label import get_label_all, REAL, PAD, MAKEUP, UNKNOWN  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
CLAIM_IDX = {"real": 0, "pad": 1, "deepfake": 2}


def claim_templates():
    """[(claim_index, claim_name, content)] from the deployed ensemble9 config (single source of truth)."""
    cfg = json.load(open(os.path.join(_ROOT, "config", "ensemble9.json")))
    texts, order = cfg["texts"], cfg["texts_claim_order"]
    return [(CLAIM_IDX[name], name, txt) for txt, name in zip(texts, order)]


def iter_images(src):
    if os.path.isdir(src):
        for dp, _, files in os.walk(src):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in EXTS:
                    yield os.path.join(dp, fn)
    else:
        for r in json.load(open(src)):
            p = r.get("image")
            if p:
                yield p


def true3(path):
    """REAL 0 / PAD 1 / DEEPFAKE 2 ; MAKEUP -> PAD ; UNKNOWN -> None (drop)."""
    l = get_label_all(path)
    if l == MAKEUP:
        l = PAD
    return None if l == UNKNOWN else l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="image DIRECTORY (walked) or an existing .json manifest")
    ap.add_argument("--mode", choices=("images", "mids9"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tmpl = claim_templates() if args.mode == "mids9" else None
    out, cnt, seen = [], collections.Counter(), set()
    for p in iter_images(args.images):
        if p in seen:
            continue
        seen.add(p)
        t = true3(p)
        if t is None:
            cnt["dropped_unknown"] += 1
            continue
        cnt[("real", "pad", "deepfake")[t]] += 1
        rec = {"image": p, "cls_label": 0 if t == REAL else 1}
        if tmpl:
            rec["answers"] = [{"content": txt, "result": name, "label": 3 * t + claim}
                              for (claim, name, txt) in tmpl]
        out.append(rec)
        if args.limit and len(out) >= args.limit:
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    real = cnt["real"]; fake = cnt["pad"] + cnt["deepfake"]
    print(f"[manifest:{args.mode}] {args.out}: {len(out):,} records "
          f"(real {real:,} / pad {cnt['pad']:,} / deepfake {cnt['deepfake']:,}; "
          f"dropped_unknown {cnt['dropped_unknown']:,})")
    if out:
        print(f"  binary balance: real {real/len(out)*100:.1f}% fake {fake/len(out)*100:.1f}% "
              f"({fake/max(real,1):.2f}:1)")
    if tmpl:
        hist = collections.Counter(a["label"] for r in out for a in r["answers"])
        print("  9-class label histogram (0..8):", {k: hist.get(k, 0) for k in range(9)})


if __name__ == "__main__":
    main()
