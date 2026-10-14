#!/usr/bin/env python3
"""Fit the fusion decision threshold from a scored results file, and (optionally) write it into a
PAAS experiment config.

The deployed threshold is only valid for the weights it was fitted on -- after retraining the
detectors it MUST be re-fitted or the fused accuracy degrades even if every head improved.

Input: the unified results line format written by scripts/run_dataset.py, e.g.
    OK truth=fake pred=fake fake=0.9997 match=0.9997 /path/to/frame.jpg

  python train/fit_threshold.py --results runs/.../results_paas.txt \
      --floor 0.90 --config runs/.../paas_retrained.json
"""
import argparse, bisect, json, re, sys

KEY = re.compile(r"truth=(\w+)\b.*?\bfake=([0-9.]+|-+)\b")


def load(path):
    ys, ss = [], []
    for ln in open(path):
        if ln.startswith("#"):
            continue
        m = KEY.search(ln)
        if not m or m.group(2).strip("-") == "":
            continue
        ys.append(1 if m.group(1) == "fake" else 0)
        ss.append(float(m.group(2)))
    return ys, ss


def _metrics(reals, fakes, tau):
    """decision: fake iff score >= tau."""
    rr = sum(1 for s in reals if s < tau) / len(reals)
    fr = sum(1 for s in fakes if s >= tau) / len(fakes)
    return rr, fr


def thr_at_floor(ys, ss, floor):
    """SMALLEST threshold whose real-recall >= floor; returns (tau, real_recall, fake_recall).

    Ties matter: evaluator scores are rounded to 4 decimals, so many reals share a value. The
    decision is `fake iff score >= tau`, so a tau EQUAL to a tied group excludes that whole group
    from real-recall. Picking reals[round(floor*n)] therefore silently undershoots the floor (a 90%
    request returned 50% real-recall on a tied distribution). Instead: walk the distinct real scores
    and take the first whose cumulative count meets the floor, then place tau strictly ABOVE that
    group. The returned tau is what the caller writes, and the reported recalls are measured with
    that exact value -- so the floor it claims is the floor it delivers.
    """
    reals = sorted(s for y, s in zip(ys, ss) if y == 0)
    fakes = sorted(s for y, s in zip(ys, ss) if y == 1)
    if not reals or not fakes:
        return None
    n = len(reals)
    allv = sorted(set(reals) | set(fakes))
    tau = None
    for c in sorted(set(reals)):
        if bisect.bisect_right(reals, c) / n >= floor:      # reals <= c would all be recalled
            nxt = allv[bisect.bisect_right(allv, c)] if bisect.bisect_right(allv, c) < len(allv) else None
            tau = (c + nxt) / 2.0 if nxt is not None else c + 1e-4   # strictly above the tied group
            break
    if tau is None:                                          # floor unreachable below the max real
        tau = reals[-1] + 1e-4
    rr, fr = _metrics(reals, fakes, tau)
    return tau, rr, fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--floors", default="0.90,0.95,0.98,0.99")
    ap.add_argument("--floor", type=float, default=0.90, help="which floor to write into --config")
    ap.add_argument("--config", default=None, help="PAAS experiment json to update in place")
    args = ap.parse_args()

    ys, ss = load(args.results)
    if not ys:
        print(f"no scored rows in {args.results}", file=sys.stderr); sys.exit(1)
    nr = sum(1 for y in ys if y == 0)
    try:
        from sklearn.metrics import roc_auc_score
        auc = f"{roc_auc_score(ys, ss):.4f}"
    except Exception:
        auc = "n/a"
    print(f"n={len(ys):,} (real={nr:,} fake={len(ys)-nr:,})  AUC={auc}")

    chosen = None
    # ALWAYS include the floor we are about to write: a DEPLOY_FLOOR not present in --floors
    # would otherwise leave `chosen` None and silently write no threshold at all.
    floors = sorted({float(x) for x in args.floors.split(",")} | {args.floor})
    for f in floors:
        r = thr_at_floor(ys, ss, f)
        if not r:
            continue
        tau, rr, fr = r
        star = " <-- selected" if abs(f - args.floor) < 1e-9 else ""
        print(f"  real-floor {f*100:.0f}%: tau={tau:.4f}  real-recall={rr*100:.2f}%  "
              f"fake-recall={fr*100:.2f}%{star}")
        if star:
            chosen = (tau, rr, fr, f)

    if args.config and chosen:
        tau, rr, fr, f = chosen
        # Write a value that STILL meets the floor: rounding can land back on a tied score group and
        # silently undo the guarantee, so verify the rounded value and fall back to full precision.
        reals = sorted(s for y, s in zip(ys, ss) if y == 0)
        fakes = sorted(s for y, s in zip(ys, ss) if y == 1)
        tau_w = round(tau, 6)
        rr_w, fr_w = _metrics(reals, fakes, tau_w)
        if rr_w < f:
            tau_w, rr_w, fr_w = tau, rr, fr
            print(f"  (kept full precision {tau_w!r}: rounding would have dropped real-recall below the floor)")
        rr, fr, tau = rr_w, fr_w, tau_w
        cfg = json.load(open(args.config))
        old = cfg.get("decision", {}).get("threshold")
        cfg.setdefault("decision", {})["threshold"] = tau_w
        cfg["_note"] = (f"threshold re-fitted on {args.results} at {f*100:.0f}% real-recall floor: "
                        f"tau={tau:.4f} (real {rr*100:.2f}% / fake {fr*100:.2f}%, AUC {auc}). "
                        f"previous={old}")
        json.dump(cfg, open(args.config, "w"), indent=2)
        print(f"\nwrote threshold {tau:.4f} -> {args.config} (was {old})")


if __name__ == "__main__":
    main()
