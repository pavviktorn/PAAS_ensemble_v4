#!/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
"""Train the GSD detector. Runs on the ONE project venv (transformers>=5 + vLLM); see train/_bootstrap.py.

  $VENV_PY train/train_gsd.py --config configs/default.json \
      --set train_data=/path/with/real_fake val_data=/path/val output_dir=runs/gsd batch_size=64
"""
import argparse
import json
import os
import sys

import _bootstrap  # noqa: F401  (project-root sys.path)
from gsd.cpu_limit import limit_cpu
limit_cpu(verbose=False)            # cap CPU threads BEFORE torch/OMP/MKL initialise (refined from cfg below)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gsd.config import GSDConfig, CLASS_NAMES
from gsd.model import GSDDetector, _pool
from gsd.faithful import GSDFaithfulConfig, GSDFaithfulDetector
from gsd.data import FaceDataset
from gsd.engine import train_one_epoch, evaluate, build_scheduler
from gsd.householder import semantic_basis
from gsd.runlog import setup_logging

_BOOL = {"true": True, "false": False, "1": True, "0": False}


def apply_overrides(cfg: GSDConfig, sets):
    d = cfg.to_dict()
    for kv in sets or []:
        k, v = kv.split("=", 1)
        if k not in d:
            raise SystemExit(f"unknown config key: {k}")
        cur = d[k]
        if isinstance(cur, bool) or v.lower() in _BOOL and isinstance(cur, bool):
            d[k] = _BOOL.get(v.lower(), bool(v))
        elif isinstance(cur, int) and not isinstance(cur, bool):
            d[k] = int(v)
        elif isinstance(cur, float):
            d[k] = float(v)
        else:
            d[k] = v
    return type(cfg).from_dict(d)


class _GuidePooler(nn.Module):
    """Frozen CLIP -> pooled guide vector (B, D). Wrapped in DataParallel so guide extraction for the
    anchor runs across all training GPUs (no GSD hooks here -- those live on the trainable stream)."""

    def __init__(self, frozen, pool):
        super().__init__()
        self.frozen, self.pool = frozen, pool

    def forward(self, x):
        return _pool(self.frozen(x).last_hidden_state, self.pool)


class _LayerTokenPooler(nn.Module):
    """Frozen CLIP -> per-layer NON-CLS token matrices, for the paper-faithful reference bases.

    Returns a stacked tensor (B, L_sel, N, D) so DataParallel can gather it. Only the selected GSD
    layers are kept: keeping all 24 would be ~6x the memory for no use."""

    def __init__(self, frozen, layer_ids):
        super().__init__()
        self.frozen, self.layer_ids = frozen, list(layer_ids)

    def forward(self, x):
        hs = self.frozen(x, output_hidden_states=True).hidden_states
        return torch.stack([hs[i + 1][:, 1:, :] for i in self.layer_ids], dim=1)


@torch.no_grad()
def build_reference_bases(core, cfg, device, source, limit, gpu_ids):
    """Paper Sec. 4.3 reference-based inference, per layer: collect the frozen NON-CLS tokens of a
    reference set at each injected layer and take the top-k RIGHT singular vectors of the stacked
    (R*N, D) token matrix -- the same estimator as eq. 12, over the reference set instead of a batch.

    Memory note: R*N tokens at D=1024 is large (5,000 refs x 576 tokens = 2.9M rows), so the Gram
    matrix G = M^T M is ACCUMULATED incrementally per batch and eigendecomposed once at the end.
    That is mathematically identical to decomposing the full matrix and needs O(D^2) memory, not
    O(R*N*D)."""
    from gsd.batch_svd import top_k_right_singular       # local import: keeps module import cheap
    ds = FaceDataset(source, cfg.image_size, train=False, cfg=cfg, limit=limit)
    ngpu = max(1, len(gpu_ids)) if device.startswith("cuda") else 1
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size * ngpu, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
    core.frozen.eval()
    ids = list(core.gsd_layer_ids)
    pooler = _LayerTokenPooler(core.frozen, ids)
    if ngpu > 1:
        pooler = nn.DataParallel(pooler, device_ids=gpu_ids)
    D = core.dim
    grams = {i: torch.zeros(D, D, dtype=torch.float64, device=device) for i in ids}
    for x, _y, _p in dl:
        x = x.to(device, non_blocking=True)
        toks = pooler(x)                                  # (B, L_sel, N, D)
        for j, i in enumerate(ids):
            M = toks[:, j].reshape(-1, D).double()         # (B*N, D)
            grams[i] += M.t() @ M
    bases = {}
    for i in ids:
        evals, evecs = torch.linalg.eigh(grams[i])
        k = int(max(1, min(cfg.k, D)))
        bases[i] = evecs[:, -k:].flip(dims=(1,)).float().cpu().contiguous()
    return bases, len(ds), ngpu


@torch.no_grad()
def build_anchor_U(core, cfg, device, source, limit, gpu_ids):
    """Build a fixed semantic basis U from a reference set (frozen CLIP guides -> Householder QR).
    Multi-GPU: guides are extracted with DataParallel across `gpu_ids`. Strided sampling keeps a
    block-ordered testset class-balanced. Kept in fp32 for a numerically stable QR."""
    ds = FaceDataset(source, cfg.image_size, train=False, cfg=cfg, limit=limit)
    ngpu = max(1, len(gpu_ids)) if device.startswith("cuda") else 1
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size * ngpu, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
    core.frozen.eval()
    pooler = _GuidePooler(core.frozen, cfg.guide_pool)
    if ngpu > 1:
        pooler = nn.DataParallel(pooler, device_ids=gpu_ids)
    guides = []
    for x, _y, _p in dl:
        x = x.to(device, non_blocking=True)
        guides.append(pooler(x).float().cpu())
    U = semantic_basis(torch.cat(guides, dim=0), cfg.k, cfg.qr_method)   # (D, K)
    return U, len(ds), ngpu


def main():
    ap = argparse.ArgumentParser(description="Train GSD detector")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "gsd_default.json"))
    ap.add_argument("--set", nargs="*", default=[], help="cfg overrides key=value")
    ap.add_argument("--limit", type=int, default=0, help="cap #train images (debug)")
    ap.add_argument("--faithful", action="store_true",
                    help="train the PAPER-FAITHFUL GSD (gsd/faithful.py): per-layer patch-token "
                         "Batch-SVD, top-k right singular vectors, adaptive layer-aware lambda. "
                         "The default path (gsd/model.py) is left untouched -- it is the deployed "
                         "detector and a member of the running like-for-like comparison.")
    ap.add_argument("--init-from", dest="init_from", default=None,
                    help="warm-start head+backbone weights from a checkpoint (fresh optimizer; "
                         "use when changing trainable scope, e.g. head -> lastN)")
    ap.add_argument("--allow-protocol-mismatch", action="store_true",
                    help="permit a faithful run whose infer_protocol is not 'reference'. The "
                         "checkpoint is then SELECTED under a protocol the server does not use; "
                         "only for deliberate study of the paper's batch/per-sample protocols.")
    args = ap.parse_args()

    _CfgCls = GSDFaithfulConfig if args.faithful else GSDConfig
    # A config carrying faithful-only fields must NOT be silently built as the deployed model.
    _FAITHFUL_ONLY = {"lambda_max", "a_init", "b_init", "suppress_cls",
                      "basis_from_patch_tokens", "infer_protocol"}
    if not args.faithful and os.path.isfile(args.config):
        try:
            _raw = json.load(open(args.config))
        except Exception:
            _raw = {}
        _hit = sorted(_FAITHFUL_ONLY & set(_raw))
        if _hit:
            raise SystemExit(
                f"[gsd] REFUSING: {args.config} carries paper-faithful field(s) {_hit} but --faithful "
                f"was not passed. Without the flag this builds GSDConfig + GSDDetector (the DEPLOYED "
                f"detector) and DISCARDS every faithful field, producing a different model with no "
                f"error and a plausible log. Pass --faithful, or point --config at a deployed config.")
    cfg = _CfgCls.from_file(args.config) if os.path.isfile(args.config) else _CfgCls()
    cfg = apply_overrides(cfg, args.set)
    if args.faithful:
        # ---- the INVERSE of the guard above. --faithful with a DEPLOYED config parses fine (every
        # ---- faithful-only field takes its dataclass default) and builds a real faithful model, so
        # ---- refusing outright would block a legitimate "faithful mechanism, default hyperparams"
        # ---- run. What must not pass silently are the three DEFAULTS that are wrong for this
        # ---- codebase, each of which produced a measured defect in an earlier run:
        _raw = {}
        if os.path.isfile(args.config):
            try:
                _raw = json.load(open(args.config))
            except Exception:
                _raw = {}
        _declared = sorted(_FAITHFUL_ONLY & set(_raw))
        if not _declared:
            print(f"[gsd] WARNING: --faithful with {os.path.basename(args.config)}, which declares "
                  f"NO faithful-only field. Every one takes its default: lambda_max={cfg.lambda_max} "
                  f"a_init={cfg.a_init} b_init={cfg.b_init} suppress_cls={cfg.suppress_cls} "
                  f"basis_from_patch_tokens={cfg.basis_from_patch_tokens} "
                  f"infer_protocol={cfg.infer_protocol}. If you meant the paper-faithful ARM, use "
                  f"train/configs/gsd_faithful.json.", flush=True)
        _anchor_src = cfg.anchor_data or cfg.val_data
        if cfg.anchor_limit == 0 and _anchor_src:
            raise SystemExit(
                f"[gsd] REFUSING: --faithful with anchor_limit=0 builds the reference basis from "
                f"EVERY image in {os.path.basename(str(_anchor_src))}. That is an unbounded frozen "
                f"CLIP pass before training starts -- the deployed GSD run did exactly this and "
                f"spent 50,384 forwards on it (runs/retrain/gsd/train.log). The paper uses a "
                f"reference SET, not the whole corpus. Set anchor_limit (gsd_faithful.json uses "
                f"5000), or anchor_data=\"\" to skip the reference basis entirely.")
        if cfg.infer_protocol == "reference" and _anchor_src and cfg.val_data \
                and os.path.abspath(_anchor_src) == os.path.abspath(cfg.val_data):
            raise SystemExit(
                f"[gsd] REFUSING: infer_protocol=reference with the reference set EQUAL to val_data "
                f"({os.path.basename(str(cfg.val_data))}). Validation would then score against a "
                f"basis fitted on the val images, so the metric selecting best.pt is self-fitted -- "
                f"the defect recorded for the deployed GSD in "
                f"EVAL_SPACE/runs/gsd_regression_explained.json. Point anchor_data at a third file.")
        if cfg.infer_protocol != "reference" and not args.allow_protocol_mismatch:
            raise SystemExit(
                f"[gsd] REFUSING: infer_protocol={cfg.infer_protocol} but paas/models/gsd_model.py "
                f"FORCES reference-based inference when it loads a paper_faithful checkpoint. best.pt "
                f"would be selected under {cfg.infer_protocol} inference and then served under "
                f"reference inference -- the metric that chose the checkpoint would describe "
                f"behaviour that is never scored or served. Set infer_protocol=reference, or pass "
                f"--allow-protocol-mismatch to study the paper's other protocols deliberately.")
    if not cfg.train_data:
        raise SystemExit("set train_data=... (a real/fake tree or a json list)")
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "train.log")
    setup_logging(log_path)                                  # tee stdout/stderr to train.log
    print(f"[gsd] logging to {log_path}", flush=True)
    gpu_ids = [int(g) for g in str(cfg.gpus).split(",") if g.strip() != ""]
    use_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    device = f"cuda:{gpu_ids[0]}" if use_cuda else "cpu"
    print(f"[gsd] GPUs={gpu_ids if use_cuda else 'cpu'} | visible={torch.cuda.device_count()} | "
          f"primary={device}", flush=True)
    limit_cpu(cfg.cpu_fraction)                              # apply configured CPU cap (env + torch threads)
    torch.manual_seed(cfg.seed)
    cfg.save(os.path.join(cfg.output_dir, "config.json"))

    if args.faithful:
        print(f"[gsd] building PAPER-FAITHFUL model (K={cfg.k}, gsd_layers={cfg.n_gsd_layers}, "
              f"per-layer patch-token Batch-SVD + top-k right singular vectors, "
              f"lambda_max={cfg.lambda_max} a0={cfg.a_init} b0={cfg.b_init}, "
              f"suppress_cls={cfg.suppress_cls}, trainable={cfg.trainable}) ...", flush=True)
        model = GSDFaithfulDetector(cfg).to(device)
    else:
        print(f"[gsd] building model (K={cfg.k}, gsd_layers={cfg.n_gsd_layers}, guide={cfg.guide_pool}, "
              f"qr={cfg.qr_method}, trainable={cfg.trainable}) ...", flush=True)
        model = GSDDetector(cfg).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[gsd] trainable params: {n_tr/1e6:.1f}M | GSD layers: {model.gsd_layer_ids}", flush=True)

    if args.init_from:                                       # warm-start (weights only; fresh optimizer)
        payload = torch.load(args.init_from, map_location="cpu")
        mb, ub = model.trainable.load_state_dict(payload["trainable"], strict=False)
        prev = payload.get("config", {})
        if args.faithful and payload.get("suppress") is not None:
            try:
                model.suppress.load_state_dict(payload["suppress"])
                print(f"[gsd] warm-start restored learned suppression params a_l/b_l "
                      f"({len(payload['suppress'])} tensors)", flush=True)
            except Exception as e:
                print(f"[gsd] WARNING: could not restore suppression params ({e}); a_l/b_l "
                      f"fall back to init a={cfg.a_init} b={cfg.b_init}", flush=True)
        elif args.faithful:
            print("[gsd] WARNING: --init-from checkpoint carries no 'suppress' state; the learned "
                  "adaptive coefficients are NOT restored and reset to their init.", flush=True)
        try:
            model.head.load_state_dict(payload["head"])
            head_msg = "head loaded"
        except Exception as e:                               # e.g. num_classes changed -> keep fresh head
            head_msg = f"head re-initialised ({e})"
        print(f"[gsd] warm-start from {args.init_from} (prev trainable={prev.get('trainable')}, "
              f"val_{cfg.select_metric}={payload.get('val_'+cfg.select_metric, 'n/a')}) | "
              f"backbone: {len(mb)} missing/{len(ub)} unexpected | {head_msg}", flush=True)

    tr = FaceDataset(cfg.train_data, cfg.image_size, train=True, cfg=cfg, limit=args.limit)
    tl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    drop_last=True, pin_memory=True)        # drop_last keeps batch>=2 for GSD
    counts = tr.class_counts()
    names = CLASS_NAMES if cfg.num_classes == 3 else ("real", "fake")
    print(f"[gsd] train={len(tr)} imgs (dropped {tr.n_dropped} unknown) | "
          f"classes: {dict(zip(names, counts))}", flush=True)

    # inverse-frequency class weights (mean-normalised) -> handles real/pad/deepfake imbalance
    class_weights = None
    if cfg.class_weight:
        tot = sum(counts)
        inv = [tot / (cfg.num_classes * c) if c else 0.0 for c in counts]
        class_weights = torch.tensor(inv, dtype=torch.float32)
        print(f"[gsd] class weights: {dict(zip(names, [round(w,3) for w in inv]))}", flush=True)

    vl = None
    if cfg.val_data:
        va = FaceDataset(cfg.val_data, cfg.image_size, train=False, cfg=cfg, limit=cfg.eval_limit)
        vl = DataLoader(va, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)
        print(f"[gsd] val={len(va)} imgs | classes: {dict(zip(names, va.class_counts()))}", flush=True)

    # ---- build the fixed semantic anchor (from the testset by default) and embed it in checkpoints ----
    anchor_src = cfg.anchor_data or cfg.val_data
    anchor_U = None
    ref_bases = None
    if args.faithful and anchor_src:
        # Reference-based inference (paper Sec. 4.3): a reusable PER-LAYER basis. Note this is the
        # paper's own third protocol, so it is faithful -- but building it from val_data means the
        # val metric is partly self-fitted, exactly as recorded for the deployed GSD in
        # runs/gsd_regression_explained.json. Point anchor_data at a THIRD file to avoid that.
        ref_bases, n_ref, r_ngpu = build_reference_bases(model, cfg, device, anchor_src,
                                                        cfg.anchor_limit, gpu_ids)
        model.set_reference_bases(ref_bases)
        _o = max((rb.t() @ rb - torch.eye(rb.shape[1])).abs().max().item() for rb in ref_bases.values())
        print(f"[gsd] reference bases: {len(ref_bases)} layers x {tuple(next(iter(ref_bases.values())).shape)} "
              f"from {n_ref} refs in {os.path.basename(anchor_src)} (on {r_ngpu} GPU(s), "
              f"max|VᵀV-I|={_o:.1e}) -> embedded in checkpoints", flush=True)
        if cfg.infer_protocol == "reference":
            if anchor_src == cfg.val_data:
                print("[gsd] WARNING: infer_protocol=reference AND the reference set is val_data, so "
                      "validation scores against a basis built from the val images -- the val metric "
                      "is partly self-fitted. Point anchor_data at a third file.", flush=True)
            else:
                print(f"[gsd] validation uses the REFERENCE protocol with a basis from "
                      f"{os.path.basename(anchor_src)} (not the val split), so selection, scoring and "
                      f"serving are the same model and the val metric is not self-fitted.", flush=True)
        else:
            print(f"[gsd] NOTE: infer_protocol={cfg.infer_protocol} -- the reference basis is embedded "
                  f"in the checkpoint but NOT used at validation, so the metric that selects best.pt "
                  f"is not the behaviour paas/models/gsd_model.py serves (it switches to reference "
                  f"inference). Set infer_protocol=reference to make them identical.", flush=True)
    elif anchor_src:
        anchor_U, n_anchor, a_ngpu = build_anchor_U(model, cfg, device, anchor_src, cfg.anchor_limit, gpu_ids)
        model.set_fixed_U(anchor_U.to(device))
        ortho = (anchor_U.t() @ anchor_U - torch.eye(cfg.k)).abs().max().item()
        print(f"[gsd] anchor U{tuple(anchor_U.shape)} from {n_anchor} refs in {os.path.basename(anchor_src)} "
              f"(on {a_ngpu} GPU(s), max|UᵀU-I|={ortho:.1e}) -> embedded in checkpoints", flush=True)

    # optimizer is built from the (unwrapped) core params; DataParallel reduces grads back into them
    core = model
    opt = torch.optim.AdamW(core.param_groups(), weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(tl)
    sched = build_scheduler(opt, cfg, total_steps)
    if sched is not None:
        print(f"[gsd] lr scheduler: {cfg.lr_scheduler} | warmup={cfg.warmup_steps} "
              f"min_lr_ratio={cfg.min_lr_ratio} | total_steps={total_steps} "
              f"(peak lr={cfg.lr:.1e}/head {cfg.head_lr:.1e})", flush=True)
    if use_cuda and len(gpu_ids) > 1:
        # Warm up lazy torch.linalg on EACH device in the main thread first: the per-batch U is built
        # with linalg.qr/svd inside the forward, which DataParallel runs in parallel replica threads;
        # torch's first-call lazy init isn't thread-safe ("lazy wrapper should be called at most once").
        for gid in gpu_ids:
            w = torch.randn(8, 4, device=f"cuda:{gid}")
            torch.linalg.qr(w, mode="reduced")
            torch.linalg.svd(w, full_matrices=False)
        torch.cuda.synchronize()
        model = nn.DataParallel(core, device_ids=gpu_ids)
        per = cfg.batch_size // len(gpu_ids)
        _basis_note = ("semantic basis estimated over the WHOLE batch via summed per-shard Grams "
                       "(exact; paper eq. 12)" if args.faithful else
                       "GSD U estimated per-GPU sub-batch")
        print(f"[gsd] DataParallel across {gpu_ids} (batch {cfg.batch_size} -> ~{per}/GPU; "
              f"{_basis_note})", flush=True)

    def payload(extra=None):
        p = {"config": cfg.to_dict(), "head": core.head.state_dict(),
             "trainable": core.trainable.state_dict()}
        if anchor_U is not None:
            p["anchor_U"] = anchor_U.cpu()
        if args.faithful:
            # UNCONDITIONAL: the variant marker and the learned suppression parameters identify the
            # model and must not depend on whether a reference set happened to be built. Without
            # this, a faithful run with no val_data/anchor_data wrote a checkpoint that
            # paas/models/gsd_model.py reads as the DEPLOYED GSD, dropping eq. 9-10 silently.
            p["gsd_variant"] = "paper_faithful"
            p["suppress"] = {k: v.cpu() for k, v in core.suppress.state_dict().items()}
            p["lambda_report"] = core.lambda_report()
            if ref_bases is not None:
                p["ref_bases"] = {int(k): v.cpu() for k, v in ref_bases.items()}
        if extra:
            p.update(extra)
        return p

    state = {"best": -1.0}

    def run_eval(gstep, ep, tag="step"):
        """Evaluate on the primary GPU and save best.pt when select_metric improves."""
        ev = evaluate(core, vl, device, cfg)
        rec = " ".join(f"{n}={r:.3f}" for n, r in zip(names, ev["recall"]))
        msg = (f"[gsd] eval @ ep{ep} gstep{gstep} | bin_auc={ev['bin_auc']:.4f} acc={ev['acc']:.4f} "
               f"bal_acc={ev['bal_acc']:.4f} [{rec}] (n={ev['n']})")
        metric = ev[cfg.select_metric]
        if metric == metric and metric >= state["best"]:     # NaN-safe; select_metric (default bin_auc)
            state["best"] = metric
            torch.save(payload({f"val_{cfg.select_metric}": metric, "global_step": gstep}),
                       os.path.join(cfg.output_dir, "best.pt"))
            msg += f"  <- saved best.pt ({cfg.select_metric}={metric:.4f})"
        print(msg, flush=True)
        return ev

    gstep = 0
    for ep in range(cfg.epochs):
        st = train_one_epoch(model, tl, opt, device, cfg, ep, class_weights=class_weights,
                             global_step=gstep, eval_every=cfg.eval_every if vl else 0, eval_cb=run_eval,
                             scheduler=sched)
        gstep = st["global_step"]
        print(f"[gsd] epoch {ep} done | train_loss={st['loss']:.4f} | gstep={gstep}", flush=True)
        if vl:
            run_eval(gstep, ep, tag="epoch")                # end-of-epoch eval + best save
            torch.save(payload(), os.path.join(cfg.output_dir, f"{ep}.pt"))
    torch.save(payload(), os.path.join(cfg.output_dir, "last.pt"))
    print(f"[gsd] done. checkpoints -> {cfg.output_dir} | best {cfg.select_metric}={state['best']:.4f}", flush=True)


if __name__ == "__main__":
    main()
