#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Preflight: parse the PRODUCTION configs through their real parsers and fail fast.

Why this exists: a config key the parser rejects (e.g. MidsPlusConfig raises
"Unknown config keys: ['select_metric']") kills a branch the moment it starts -- potentially after
the MLLM chain has already burned a day. Smoke mode cannot be relied on to catch it, because smoke
trains with its own tiny configs; only the configs the REAL run will use are authoritative.

  python train/validate_configs.py --ninec train/configs/mids_a2_9c.yaml \
      --gsd train/configs/gsd_default.json --selop train/configs/selop_config.json
"""
import argparse, json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (_ROOT, os.path.join(_ROOT, "ensemble9")):
    if _d not in sys.path:
        sys.path.insert(0, _d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ninec"), ap.add_argument("--gsd"), ap.add_argument("--selop")
    args = ap.parse_args()
    errs = []

    if args.ninec:
        try:
            from mids9lib.config import MidsPlusConfig
            c = MidsPlusConfig.from_yaml(args.ninec)
            print(f"  9c    {os.path.basename(args.ninec)}: num_classes={c.num_classes} "
                  f"artifact={c.artifact_enabled} clip_adapt={c.clip_adapt} "
                  f"select_metric={getattr(c, 'select_metric', 'auc')}")
            if c.num_classes != 9:
                errs.append(f"9c num_classes={c.num_classes}, but the deployed ensemble does "
                            f"logits.view(n,3,3) and REQUIRES 9")
        except Exception as e:
            errs.append(f"9c config {args.ninec}: {type(e).__name__}: {e}")

    if args.gsd:
        try:
            from gsd.config import GSDConfig
            d = json.load(open(args.gsd))
            # A faithful config parsed as GSDConfig validates "successfully" while every
            # faithful-only field is dropped -- the preflight would green-light a run that then
            # builds a completely different model. Detect the variant and parse it as itself, and
            # report the mismatch here rather than leaving it to the trainer, which only refuses
            # AFTER other expensive branches of the preflight have run.
            FAITHFUL_ONLY = {"lambda_max", "a_init", "b_init", "suppress_cls",
                             "basis_from_patch_tokens", "infer_protocol"}
            hit = sorted(FAITHFUL_ONLY & set(d))
            if hit:
                from gsd.faithful import GSDFaithfulConfig
                c = GSDFaithfulConfig.from_dict(d).validate()
                print(f"  gsd   {os.path.basename(args.gsd)}: PAPER-FAITHFUL variant "
                      f"(fields {hit}) | num_classes={c.num_classes} trainable={c.trainable} "
                      f"k={c.k} lambda_max={c.lambda_max} a0={c.a_init} b0={c.b_init} "
                      f"protocol={c.infer_protocol} whole_frame={getattr(c, 'whole_frame', False)}")
                if os.environ.get("GSD_FAITHFUL", "0") != "1":
                    errs.append(f"gsd config {args.gsd} is the PAPER-FAITHFUL variant (fields {hit}) "
                                f"but GSD_FAITHFUL is not 1. run_finetuning.sh would then invoke "
                                f"train_gsd.py WITHOUT --faithful, which builds the DEPLOYED "
                                f"GSDDetector and silently discards every field above. Set "
                                f"GSD_FAITHFUL=1, or point GSD_CFG at a deployed config.")
                if c.infer_protocol == "reference" and c.anchor_data and c.val_data \
                        and os.path.abspath(c.anchor_data) == os.path.abspath(c.val_data):
                    errs.append(f"gsd config {args.gsd}: infer_protocol=reference with "
                                f"anchor_data == val_data, so validation scores against a basis "
                                f"fitted on the val images -- the val metric is self-fitted.")
            else:
                c = GSDConfig.from_dict(d)
                print(f"  gsd   {os.path.basename(args.gsd)}: num_classes={c.num_classes} "
                      f"trainable={c.trainable} whole_frame={getattr(c, 'whole_frame', False)}")
                # THE INVERSE MISMATCH. The check above catches a faithful config launched without
                # the flag; this catches the flag launched with a DEPLOYED config, which parses
                # cleanly as GSDFaithfulConfig with every faithful field at its default -- and those
                # defaults are wrong here in three ways at once: infer_protocol='batch' (selected
                # under a protocol paas/models/gsd_model.py never serves), anchor_data=null (the
                # reference basis falls back to val_data, so the val metric is self-fitted), and
                # anchor_limit=0 (an unbounded frozen CLIP pass -- the deployed run spent 50,384
                # forwards on exactly this). train_gsd.py refuses the run; the preflight says so
                # first, before the rest of the chain has burned any time.
                if os.environ.get("GSD_FAITHFUL", "0") == "1":
                    errs.append(
                        f"gsd config {args.gsd} is a DEPLOYED config but GSD_FAITHFUL=1, so "
                        f"run_finetuning.sh would pass --faithful. Every faithful field would take "
                        f"its default: infer_protocol=batch (validated under a protocol the server "
                        f"never uses), anchor_data={d.get('anchor_data')!r} (falls back to val_data "
                        f"-> self-fitted val metric), anchor_limit={d.get('anchor_limit')} "
                        f"(0 = every image in the reference set). Point GSD_CFG at "
                        f"train/configs/gsd_faithful.json, or unset GSD_FAITHFUL.")
        except Exception as e:
            errs.append(f"gsd config {args.gsd}: {type(e).__name__}: {e}")

    if args.selop:
        try:
            d = json.load(open(args.selop))
            missing = [k for k in ("clip_path", "num_classes", "rank", "n_intervene", "batch_size",
                                   "epochs", "lr", "image_size", "select_metric") if k not in d]
            if missing:
                errs.append(f"selop config {args.selop}: missing keys {missing}")
            print(f"  selop {os.path.basename(args.selop)}: num_classes={d.get('num_classes')} "
                  f"init_from={d.get('init_from')} whole_frame={d.get('whole_frame')}")
        except Exception as e:
            errs.append(f"selop config {args.selop}: {type(e).__name__}: {e}")

    if errs:
        print("\n[preflight] CONFIG ERRORS -- fix before launching:")
        for e in errs:
            print(f"  * {e}")
        sys.exit(1)
    print("  [preflight] all configs parse")


if __name__ == "__main__":
    main()
