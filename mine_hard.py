"""Hard-negative mining: run the deployed v4 ensemble over a dir of DEEPFAKE images and copy the
ones the model mis-classifies as REAL (fused decision == "real") into an output dir, preserving the
folder tree. Batched, GPU 0, resumable (skips paths already in the progress file).

  CUDA_VISIBLE_DEVICES=0 python mine_hard.py [--limit N]
"""
import argparse, json, os, shutil, sys, time
import numpy as np
from PIL import Image

ROOT = "/datasets/work/vLLM/temp/PAAS_ensemble_v4"
sys.path.insert(0, ROOT)
IN = "/datasets/work/vLLM/data_processing/down_gs/new_df_faces"
OUT = "/datasets/work/vLLM/data_processing/down_gs/new_df_faces_hard"
SCRATCH = "/tmp/claude-1001/-datasets-work-vLLM-temp/78771557-5bd2-4eaf-81fb-4801ebb47561/scratchpad"
PROGRESS = os.path.join(SCRATCH, "mine_hard_progress.txt")   # every processed rel-path (resume)
RESULTS = os.path.join(SCRATCH, "mine_hard_realmiss.txt")    # mis-classified-as-real rel-paths
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=7200.0, help="wall-clock cap; stop even if unfinished")
    ap.add_argument("--score-max", type=float, default=0.4, help="copy images with fused forgery_score < this")
    ap.add_argument("--tag", default="s04", help="tag for progress/hits/scores files (fresh scan per criterion)")
    args = ap.parse_args()
    progress = os.path.join(SCRATCH, f"mine_hard_progress_{args.tag}.txt")
    results  = os.path.join(SCRATCH, f"mine_hard_hits_{args.tag}.txt")
    scores   = os.path.join(SCRATCH, f"mine_hard_scores_{args.tag}.txt")

    os.chdir(ROOT)
    from paas import env; env.setup(device="cuda:0")
    from paas.config import PaasConfig
    from paas.pipeline import PaasPipeline
    cfg = PaasConfig.from_dict(json.load(open("config/experiments/paas4_qwen.json"))).validate()
    cfg.ffaa.qwen_gpu_mem = float(os.environ.get("QWEN_GPU_MEM", "0.45"))   # GPU 0 dedicated -> bigger KV cache
    print(f"[mine] building ensemble: {cfg.fusion.components} | qwen_gpu_mem={cfg.ffaa.qwen_gpu_mem}", flush=True)
    pipe = PaasPipeline(cfg)

    all_imgs = []
    for root, _, files in os.walk(IN):
        for f in files:
            if f.lower().endswith(EXTS):
                all_imgs.append(os.path.relpath(os.path.join(root, f), IN))
    all_imgs.sort(reverse=True)   # start from the VERY LAST end of the tree, going backwards
    done = set()
    if os.path.exists(progress):
        done = {l.strip() for l in open(progress) if l.strip()}
    todo = [r for r in all_imgs if r not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[mine] total={len(all_imgs):,} done={len(done):,} todo={len(todo):,}", flush=True)

    os.makedirs(SCRATCH, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    pf = open(progress, "a"); rf = open(results, "a"); sf = open(scores, "a")
    print(f"[mine] criterion: forgery_score < {args.score_max} | scores logged -> {scores}", flush=True)
    nmiss = nerr = 0; t0 = time.time()
    B = args.batch
    for i in range(0, len(todo), B):
        if time.time() - t0 >= args.max_seconds:
            print(f"[mine] TIME LIMIT ({args.max_seconds/3600:.1f} h) reached -> stopping at {i:,}/{len(todo):,}", flush=True)
            break
        chunk = todo[i:i + B]
        rgbs, keys, valid = [], [], []
        for rel in chunk:
            try:
                rgbs.append(np.array(Image.open(os.path.join(IN, rel)).convert("RGB")))
                keys.append(os.path.join(IN, rel)); valid.append(rel)
            except Exception:
                pf.write(rel + "\n")   # unreadable -> mark done, skip
        if rgbs:
            cb = min(B, 64)   # CLIP detectors (4 models) OOM at large batches; keep them modest
            res = pipe.predict_frames(rgbs, keys=keys, ens_batch_size=cb,
                                      ffaa_batch_size=B, gsd_batch_size=cb, selop_batch_size=cb)
            for rel, r in zip(valid, res):
                fs = r.get("forgery_score")
                if r.get("decision") == "error" or fs is None:
                    nerr += 1
                else:
                    sf.write(f"{rel}\t{fs:.4f}\n")          # log EVERY score -> future re-thresholding is free
                    if fs < args.score_max:                 # low forgery_score = hard fake (looks real)
                        dst = os.path.join(OUT, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(os.path.join(IN, rel), dst)
                        rf.write(rel + "\n"); nmiss += 1
                pf.write(rel + "\n")
        pf.flush(); rf.flush(); sf.flush()
        proc = min(i + B, len(todo))
        if (i // B) % 5 == 0 or proc == len(todo):
            el = time.time() - t0; rate = proc / max(el, 1e-6)
            eta = (len(todo) - proc) / max(rate, 1e-6) / 60
            print(f"[mine] {proc:,}/{len(todo):,} | hits(<{args.score_max})={nmiss:,} "
                  f"({nmiss/max(proc,1)*100:.2f}%) | err={nerr} | {rate:.1f} img/s | ETA {eta:.0f} min",
                  flush=True)
    print(f"[mine] DONE processed={len(todo):,} hits(score<{args.score_max})={nmiss:,} err={nerr}", flush=True)


if __name__ == "__main__":
    main()
