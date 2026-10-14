"""MIDS++ trainer.

Plain PyTorch + (optional) torchrun DDP + AMP -- no DeepSpeed dependency, which suits the
4x96GB box this is built for while staying portable.  Launch single-GPU::

    mids-pp-train --config configs/mids_pp.yaml --set train_data_path=... val_data_path=...

or multi-GPU::

    torchrun --nproc_per_node=4 -m mids_plus.train --config configs/mids_pp.yaml --set ...

Only the learned tensors are checkpointed (see ``checkpoint.save_checkpoint``).  Validation tracks
ACC/AUC/AP via the unchanged FFAA match-score selector; the best-ACC checkpoint is kept.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

import _bootstrap  # noqa: F401
from mids9lib.checkpoint import save_checkpoint
from mids9lib.config import MidsPlusConfig
from mids9lib.data import MidsAnswersDataset, build_image_transform, collate_fn
from mids9lib.distributed import (all_gather_lists, cleanup_distributed, is_distributed,
                          is_main, rank0_print, setup_distributed)
from mids9lib.losses import MidsPlusLoss
from mids9lib.metrics import binary_metrics
from mids9lib.model import build_model
from mids9lib.selector import make_decision_batch, make_decision9_batch
from mids9lib.tokenize import build_tokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **_):
        return it


def _steplog(msg: str) -> None:
    """Emit ONE clean line even while a tqdm bar is live.

    A bare print() is written at the cursor, which sits mid-bar -- the line comes out glued to the
    bar text ("epoch 1/1: 10%|...[9c] ep 1 step 2 ..."), so `grep '^\\[9c\\]'` silently misses a
    third of the steps. tqdm.write() clears the bar, writes the line, and redraws. Falls back to
    print() when tqdm is the no-op stub defined above.
    """
    if not is_main():
        return
    w = getattr(tqdm, "write", None)
    if callable(w):
        w(msg)
    else:
        print(msg, flush=True)


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "S_residual" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _tokenize(tokenizer, texts: List[str], device):
    enc = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, max_length=512)
    return enc.to(device)


@torch.no_grad()
def evaluate_split(model, core, tokenizer, loader, device, cfg, amp_ctx) -> dict:
    model.eval()
    y_true, y_pred, y_score = [], [], []
    s = cfg.samples_per_image
    for batch in loader:
        images = batch["image"].to(device)
        inputs = _tokenize(tokenizer, batch["texts"], device)
        b = images.size(0)
        with amp_ctx():
            out = model(inputs, images, None, b, s - 1, 0)
        scores = F.softmax(out["logits"].float(), dim=2)[:, :s, :]
        if scores.size(2) == 9:
            _, preds, _, _sel = make_decision9_batch(batch["answers_result"], scores, chunk_size=s)
            # SCORE THE WAY PRODUCTION DOES. ensemble.py computes
            #     Pt = softmax.view(n_answers, 3, 3).sum(dim=2);  fake = mean_a (1 - Pt[:, 0])
            # i.e. the true-class marginal averaged over ALL answers -- NOT the winning-answer
            # selector score make_decision9_batch returns. The two are not monotonically related, so
            # selecting a checkpoint by AUC of the selector score optimises the wrong ranking.
            Pt = scores.view(scores.size(0), s, 3, 3).sum(dim=3)        # (b, s, 3 true)
            forgery = (1.0 - Pt[:, :, 0]).mean(dim=1).detach().cpu().tolist()
        else:
            _, preds, _, forgery = make_decision_batch(batch["answers_result"], scores, chunk_size=s)
        y_true.extend(batch["cls_label"].tolist())
        y_pred.extend(preds)
        y_score.extend(forgery)
    y_true = all_gather_lists(y_true)
    y_pred = all_gather_lists(y_pred)
    y_score = all_gather_lists(y_score)
    return binary_metrics(y_true, y_pred, y_score)


def train(cfg: MidsPlusConfig, resume: str | None = None) -> None:
    rank, world, local_rank = setup_distributed()
    _seed_everything(cfg.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(payload["state"], strict=False)
        rank0_print(f"Resumed trainable weights from {resume}")
    model.to(device)
    rank0_print(f"Trainable parameters: {model.num_trainable_parameters()/1e6:.2f}M")
    if cfg.clip_adapt == "svd":
        rank0_print(f"SVD-adapted CLIP linears: {len(model.svd_report.replaced)}")

    core = model
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    tokenizer = build_tokenizer(cfg)
    transform_train = build_image_transform(cfg, "train")
    transform_val = build_image_transform(cfg, "val")
    train_ds = MidsAnswersDataset(cfg.train_data_path, cfg, "train", transform_train)
    val_ds = MidsAnswersDataset(cfg.val_data_path, cfg, "val", transform_val) if cfg.val_data_path else None

    train_sampler = DistributedSampler(train_ds) if is_distributed() else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
        num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=True, pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
        val_loader = DataLoader(
            val_ds, batch_size=cfg.val_batch_size, sampler=val_sampler, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
        )

    criterion = MidsPlusLoss(cfg)
    optimizer = AdamW(_param_groups(core, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    total_steps = cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _lr_lambda(step, warmup_steps, total_steps)
    )

    use_cuda = torch.cuda.is_available()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp_dtype)
    use_scaler = use_cuda and cfg.amp_dtype == "fp16"
    try:  # torch >= 2.3
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def amp_ctx():
        if use_cuda and amp_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        import contextlib

        return contextlib.nullcontext()

    if is_main():
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.dump(os.path.join(cfg.output_dir, "config.yaml"))

    s = cfg.samples_per_image
    best_acc = -1.0
    global_step = 0
    # Per-step train logging: one plain line every TRAIN_LOG_EVERY steps (tqdm writes \r, so its
    # postfix is not greppable in a stage log). Values are WINDOW means since the previous line --
    # not epoch-cumulative -- so a stalled or diverging run is visible immediately. Rank 0 only,
    # therefore computed on rank 0's shard (a sample of the global batch, which is what you want
    # for a progress signal; the authoritative numbers come from the end-of-epoch validation).
    log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    w_loss = w_correct = w_total = w_n = 0
    for epoch in range(cfg.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running = 0.0
        last_parts = {}
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", disable=not is_main()):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            cls_labels = batch["cls_label"].to(device, non_blocking=True)
            inputs = _tokenize(tokenizer, batch["texts"], device)
            b = images.size(0)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                out = model(inputs, images, None, b, s - 1, 0, cls_labels)
                loss_out = criterion(out, labels, cls_labels, core)
            loss = loss_out.total
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], cfg.grad_clip)
                optimizer.step()
            scheduler.step()
            global_step += 1
            running += float(loss.detach())
            last_parts = {k: float(v) for k, v in loss_out.parts.items()}

            # train acc on the SAME quantity the CE optimises: flattened 9-class logits vs labels,
            # honouring the -1 ignore_index that losses.py uses.
            with torch.no_grad():
                _lg = out["logits"].reshape(-1, cfg.num_classes)
                _lb = labels.reshape(-1).long().to(_lg.device)
                _m = _lb != -1
                if bool(_m.any()):
                    w_correct += int((_lg[_m].argmax(dim=1) == _lb[_m]).sum())
                    w_total += int(_m.sum())
            w_loss += float(loss.detach())
            w_n += 1
            if log_every and global_step % log_every == 0:
                _steplog(f"[9c] ep {epoch+1}/{cfg.epochs} step {global_step}/{total_steps} "
                         f"loss {w_loss/max(w_n,1):.4f} acc {w_correct/max(w_total,1):.4f} "
                         f"lr {scheduler.get_last_lr()[0]:.2e}")
                w_loss = w_correct = w_total = w_n = 0

            if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                rank0_print(f"[step {global_step}] loss={float(loss.detach()):.4f} "
                            f"parts={ {k: round(v, 3) for k, v in last_parts.items()} }")
                if is_main():
                    save_checkpoint(core, cfg, os.path.join(cfg.output_dir, "last.pt"))

        running /= max(1, len(train_loader))
        rank0_print(f"[epoch {epoch+1}] train_loss={running:.4f} parts={last_parts} lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loader is not None:
            metrics = evaluate_split(model, core, tokenizer, val_loader, device, cfg, amp_ctx)
            rank0_print(f"[epoch {epoch+1}] val acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} ap={metrics['ap']:.4f}")
            # Select by AUC, not ACC: production ranks by the mean marginal fake-score
            # (ensemble.py: 1 - P(true=real)), so AUC is the metric that actually transfers.
            # ACC selection is what hid the axon1 tail in Exp 17/18.
            _sel = getattr(cfg, "select_metric", "auc")
            _score = metrics.get(_sel, metrics["auc"])
            if is_main() and _score >= best_acc:
                best_acc = _score
                save_checkpoint(core, cfg, os.path.join(cfg.output_dir, "best.pt"))
                rank0_print(f"  saved best.pt ({_sel}={best_acc:.4f})")

        if is_main():
            save_checkpoint(core, cfg, os.path.join(cfg.output_dir, "last.pt"))

    rank0_print(f"Done. best val acc={best_acc:.4f}")
    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MIDS++")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = MidsPlusConfig.from_yaml(args.config).apply_overrides(args.set)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
