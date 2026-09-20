"""Training / evaluation loops for the GSD detector (CrossEntropy, AdamW).

Default head is 3-class (real/pad/deepfake). Metrics report overall accuracy, per-class recall,
balanced accuracy, and the operational real-vs-fake AUC (P(fake)=1-P(real))."""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def build_scheduler(optimizer, cfg, total_steps: int):
    """Per-step LR schedule (LambdaLR scales every param group by the same factor of its peak lr).
    'cosine': linear warmup -> cosine decay to min_lr_ratio. 'linear': warmup -> linear decay."""
    if cfg.lr_scheduler == "none":
        return None
    warmup = max(0, int(cfg.warmup_steps))
    floor = float(cfg.min_lr_ratio)

    def lr_lambda(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(1.0, max(0.0, prog))
        if cfg.lr_scheduler == "cosine":
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
        return floor + (1 - floor) * (1 - prog)             # linear

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _amp_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(name, torch.float32)


def _core(model):
    """Underlying GSDDetector, unwrapping nn.DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def roc_auc(labels: List[int], scores: List[float]) -> float:
    """Dependency-free AUC (rank statistic / Mann-Whitney U)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                       # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _set_train_mode(core):
    core.train()                      # restore the PARENT flag too (children alone left it False
                                      # after evaluate()'s model.eval())
    core.trainable.train(); core.head.train(); core.frozen.eval()
    core._infer_anchor = False        # training always uses the per-batch basis (no test leakage)


def train_one_epoch(model, loader, optimizer, device, cfg, epoch: int,
                    class_weights: Optional[torch.Tensor] = None,
                    global_step: int = 0, eval_every: int = 0, eval_cb=None,
                    scheduler=None) -> Dict[str, float]:
    core = _core(model)
    _set_train_mode(core)
    w = class_weights.to(device) if class_weights is not None else None
    crit = nn.CrossEntropyLoss(weight=w)
    adt = _amp_dtype(cfg.amp_dtype)
    use_amp = adt in (torch.bfloat16, torch.float16) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=(adt == torch.float16))
    t0, running, seen = time.time(), 0.0, 0
    # Per-step train logging (loss/acc/lr). TRAIN_LOG_EVERY overrides cfg.log_every so all four
    # detectors share one cadence knob. Reported values are WINDOW means since the previous line,
    # while `running/seen` stays epoch-cumulative for the returned epoch summary.
    log_every = int(os.environ.get("TRAIN_LOG_EVERY", 0)) or cfg.log_every
    w_loss = w_correct = w_seen = w_n = 0
    for step, (x, y, _paths) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        # The paper-faithful GSD estimates its semantic basis over the WHOLE batch (eq. 12). Under
        # DataParallel each replica only sees batch/n_gpus, so the basis must be built here, before
        # the forward is scattered. Guarded by an attribute so the deployed detector is untouched.
        _c = _core(model)
        if getattr(_c, "needs_prepare_step", False):
            _c.prepare_step(x, gpu_ids=(model.device_ids if isinstance(model, nn.DataParallel) else None))
        with torch.autocast(device_type="cuda", dtype=adt, enabled=use_amp):
            logits = model(x)
            loss = crit(logits, y)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_((p for g in optimizer.param_groups for p in g["params"]), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_((p for g in optimizer.param_groups for p in g["params"]), cfg.grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += loss.item() * x.size(0); seen += x.size(0)
        global_step += 1
        with torch.no_grad():
            w_correct += int((logits.argmax(dim=1) == y).sum())
        w_seen += x.size(0)
        w_loss += loss.item() * x.size(0)
        w_n += 1
        if log_every and step % log_every == 0:
            rate = seen / max(time.time() - t0, 1e-9)
            lr = optimizer.param_groups[-1]["lr"]           # backbone group lr (last); head is [0]
            print(f"[gsd] ep {epoch} step {step} gstep {global_step}/{len(loader)*cfg.epochs} "
                  f"loss {w_loss/max(w_seen,1):.4f} acc {w_correct/max(w_seen,1):.4f} "
                  f"lr {lr:.2e} {rate:.1f} img/s", flush=True)
            w_loss = w_correct = w_seen = w_n = 0
        if eval_every and eval_cb and global_step % eval_every == 0:   # mid-epoch evaluation + best save
            eval_cb(global_step, epoch)
            _set_train_mode(core)                                      # restore after eval set .eval()
    return {"loss": running / max(seen, 1), "global_step": global_step}


@torch.no_grad()
def evaluate(model, loader, device, cfg) -> Dict:
    """3-class metrics + operational real-vs-fake AUC. P(fake) = 1 - softmax[real]."""
    model.eval()
    _c = _core(model)
    _prev_anchor = getattr(_c, "_infer_anchor", False)
    _c._infer_anchor = True           # validate the way we serve: fixed anchor, batch-independent
    adt = _amp_dtype(cfg.amp_dtype)
    use_amp = adt in (torch.bfloat16, torch.float16) and device.startswith("cuda")
    C = cfg.num_classes
    y_true: List[int] = []; y_pred: List[int] = []; fake_score: List[float] = []
    for x, y, _paths in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=adt, enabled=use_amp):
            prob = torch.softmax(model(x).float(), dim=1)
        y_pred.extend(prob.argmax(1).cpu().tolist())
        fake_score.extend((1.0 - prob[:, 0]).cpu().tolist())     # class 0 == real
        y_true.extend(y.int().tolist())
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(n, 1)
    # per-class recall + balanced accuracy
    recall, counts = [], []
    for c in range(C):
        tot = sum(t == c for t in y_true)
        hit = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        counts.append(tot)
        recall.append(hit / tot if tot else float("nan"))
    valid = [r for r in recall if r == r]
    bal_acc = sum(valid) / len(valid) if valid else float("nan")
    # operational real(0) vs fake(>0) AUC
    bin_lab = [0 if t == 0 else 1 for t in y_true]
    bin_auc = roc_auc(bin_lab, fake_score)
    _c._infer_anchor = _prev_anchor        # leave the model exactly as we found it
    return {"bin_auc": bin_auc, "acc": acc, "bal_acc": bal_acc, "recall": recall,
            "counts": counts, "n": n}
