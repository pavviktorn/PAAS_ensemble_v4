#!/usr/bin/env python3
"""Assemble a runnable deployment from a finished run_finetuning.sh output dir.

Collects each branch's BEST checkpoint into <out>/deploy/ and writes (a) an ensemble9 member config
pointing at the retrained A2_9c and (b) a PAAS experiment config whose per-detector weight paths
point at the retrained heads. Nothing under weights/ or config/ is overwritten -- promote by hand
after you have seen the numbers.

  python train/make_deploy_config.py --run runs/finetune_YYYYmmdd_HHMMSS [--base config/experiments/paas4_qwen.json]
"""
import argparse, glob, json, os, shutil, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def newest(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_finetuning output dir (runs/finetune_*)")
    ap.add_argument("--base", default=os.path.join(_ROOT, "config", "experiments", "paas4_qwen.json"))
    ap.add_argument("--qwen", default=None,
                    help="merged MLLM dir THIS run used (run_finetuning.sh passes $MERGED). REQUIRED for "
                         "correctness when the MLLM was retrained: the MIDS head is trained on this "
                         "model's answers, so serving any other MLLM is train/serve skew.")
    args = ap.parse_args()
    if args.qwen and not os.path.isdir(args.qwen):
        sys.exit(f"--qwen dir does not exist: {args.qwen}")
    run = os.path.abspath(args.run)
    dep = os.path.join(run, "deploy"); os.makedirs(dep, exist_ok=True)

    # ---- locate each branch's best checkpoint (None = branch not trained in this run) ----
    found = {
        "mids":  newest(os.path.join(run, "mids", "**", "best.pth")),
        "ninec": newest(os.path.join(run, "ninec", "**", "best.pt")) or newest(os.path.join(run, "ninec", "best.pt")),
        "gsd":   newest(os.path.join(run, "gsd", "best*.pt")),
        "selop": newest(os.path.join(run, "selop", "best*.pt")),
    }
    for k, v in found.items():
        print(f"  {k:6s}: {v or 'NOT FOUND (branch skipped or failed)'}")

    cfg = json.load(open(args.base))
    names = {"mids": "ffaa_mids.pth", "ninec": "A2_9c.pt", "gsd": "gsd.pt", "selop": "selop.pt"}
    for k, src in found.items():
        if src:
            shutil.copy2(src, os.path.join(dep, names[k]))

    # ---- ensemble9 member config -> retrained A2 (absolute paths; keeps the canonical texts) ----
    if found["ninec"]:
        e9 = json.load(open(os.path.join(_ROOT, "config", "ensemble9.json")))
        e9["base_models"] = {"image": os.path.join(_ROOT, "base_models", "clip-vit-large-patch14-336"),
                             "text": os.path.join(_ROOT, "base_models", "t5-base")}
        e9["members"] = [{"name": "A2_9c", "checkpoint": os.path.join(dep, names["ninec"])}]
        e9["_about"] = f"retrained A2_9c from {run}"
        e9_path = os.path.join(dep, "ensemble9.json")
        json.dump(e9, open(e9_path, "w"), indent=2)
        cfg.setdefault("ensemble9", {})["config_path"] = e9_path

    # ---- point the experiment config at the retrained heads ----
    # The MLLM MUST be pinned explicitly: FFAACfg.qwen_dir defaults to env.QWEN_DIR
    # (weights/qwen35_4b_merged, the previously deployed model). If the MLLM was retrained this run
    # it lives in runs/qwen_merged, and leaving the default would serve the MIDS head answers from a
    # DIFFERENT MLLM than it was trained on -- train/serve skew that also corrupts the threshold fit.
    if args.qwen:
        cfg.setdefault("ffaa", {})["qwen_dir"] = os.path.abspath(args.qwen)
        print(f"  MLLM   : {os.path.abspath(args.qwen)} (pinned)")
    else:
        print("  MLLM   : !! NOT PINNED -- config falls back to env.QWEN_DIR "
              "(weights/qwen35_4b_merged). Pass --qwen if the MLLM was retrained.")
    if found["mids"]:
        cfg.setdefault("ffaa", {})["mids_path"] = os.path.join(dep, names["mids"])
        # heads retrained by run_finetuning.sh use WHOLE-FRAME pixels; serving them with the legacy
        # CLIP center-crop would reintroduce exactly the train/serve skew we removed.
        cfg["ffaa"]["whole_frame"] = True
    if found["gsd"]:
        cfg.setdefault("gsd", {})["ckpt"] = os.path.join(dep, names["gsd"])
    if found["selop"]:
        cfg.setdefault("selop", {})["ckpt"] = os.path.join(dep, names["selop"])
    cfg["name"] = "paas4_qwen_retrained"
    cfg["_note"] = (f"RETRAINED heads from {run}; MLLM={cfg.get('ffaa',{}).get('qwen_dir','<env default>')}. "
                    f"THRESHOLD IS STALE until train/fit_threshold.py re-fits it on this build -- the "
                    f"inherited value was fitted on the old weights.")
    out_cfg = os.path.join(dep, "paas_retrained.json")
    json.dump(cfg, open(out_cfg, "w"), indent=2)
    print(f"\ndeploy dir : {dep}\nconfig     : {out_cfg}")
    print("NEXT: score a labeled set with this config, then fit the threshold:\n"
          f"  python scripts/run_dataset.py --config {out_cfg} --input-dir <testset dir> --out-dir {dep}/eval\n"
          f"  python train/fit_threshold.py --results {dep}/eval/results_paas.txt --config {out_cfg}")


if __name__ == "__main__":
    main()
