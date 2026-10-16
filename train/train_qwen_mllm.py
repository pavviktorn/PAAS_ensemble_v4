#!/usr/bin/env python
"""LoRA-finetune Qwen3-VL-Instruct on the eFFAA (LLaVA-format) forgery-analysis data.

Runs in the PROJECT VENV (transformers>=5, peft, accelerate) via torchrun:

  ./venv/bin/torchrun --nproc_per_node=4 qwen/train_qwen_lora.py \
      --model base_models/Qwen3-VL-4B-Instruct \
      --train /datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json \
      --eval /datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json \
      --image-root /datasets/newout --out runs/qwen_lora

LoRA mirrors the FFAA recipe (r=32, alpha=48, dropout 0.05, lr 1e-4 cosine, 3% warmup) and is
applied to the LANGUAGE decoder only (vision tower + merger frozen).
"""
import argparse
import os
import re
import sys

import torch
import transformers
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401
from qwen.data_effaa import EffaaCollator, EffaaDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--image-root", default="/datasets/newout")
    ap.add_argument("--out", default="runs/qwen_lora")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=48)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--max-pixels", type=int, default=451584)   # ~576 visual tokens cap
    ap.add_argument("--limit", type=int, default=0, help="cap #train records (smoke test)")
    ap.add_argument("--eval-limit", type=int, default=2000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--resume-adapter", default=None,
                    help="continue training from an existing LoRA adapter dir (fresh optimizer)")
    ap.add_argument("--train-merger", action="store_true",
                    help="FULLY fine-tune the vision->language merger(s) (frozen by default)")
    ap.add_argument("--merger-lr", type=float, default=2e-5,
                    help="independent LR for the fully-fine-tuned merger group (FFAA mm_projector_lr "
                         "style: full-FT pretrained weights need a gentler LR than zero-init LoRA)")
    ap.add_argument("--balance-real", type=float, default=0.0,
                    help=">0: oversample REAL records by this factor (e.g. 2.44 to balance 29/71)")
    args = ap.parse_args()

    processor = transformers.AutoProcessor.from_pretrained(args.model)
    if hasattr(processor, "image_processor") and args.max_pixels:
        # Qwen3-VL (tf>=5): the visual-token budget is size.longest_edge in PIXEL AREA
        # (patch16 x merge2 -> 1024 px/token; 451584 px ~= 441 tokens max per image)
        processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": args.max_pixels}
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.config.use_cache = False

    # FFAA-style split: vision tower FROZEN, LLM decoder via LoRA, and (with --train-merger)
    # the vision->language merger/projector FULLY fine-tuned (modules_to_save, no LoRA on it).
    # Merger targets are DISCOVERED from the model so this works across architectures:
    #   Qwen3-VL  -> visual.merger + visual.deepstack_merger_list.{0,1,2}
    #   Qwen3.5   -> visual.merger only (no deepstack)
    save_merger = None
    if args.train_merger:
        mnames = {n for n, _ in model.named_modules()}
        save_merger = []
        if any(n.endswith("visual.merger") for n in mnames):
            save_merger.append("visual.merger")
        ds = sorted({int(m.group(1)) for n in mnames
                     for m in [re.search(r"visual\.deepstack_merger_list\.(\d+)$", n)] if m})
        save_merger += [f"visual.deepstack_merger_list.{i}" for i in ds]
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"[merger full-FT] modules_to_save = {save_merger}")
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        # Covers standard transformer layers (Qwen3-VL: self_attn q/k/v/o) AND hybrid
        # linear-attention layers (Qwen3.5: linear_attn in_proj_qkv/z + out_proj -- the LARGE
        # token-mixing matrices; in_proj_a/b are tiny 32-dim delta-net params, left frozen).
        # mlp adapted on every layer. Non-existent alternatives simply don't match per arch.
        target_modules=(r".*language_model\.layers\.\d+\.("
                        r"self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
                        r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|"
                        r"mlp\.(gate_proj|up_proj|down_proj))"),
        modules_to_save=save_merger,
    )
    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"resumed adapter from {args.resume_adapter}")
    else:
        model = get_peft_model(model, lora)
    if int(os.environ.get("RANK", 0)) == 0:
        model.print_trainable_parameters()
        mt = [n for n, p in model.named_parameters() if p.requires_grad and "visual" in n]
        print(f"trainable visual (merger) param tensors: {len(mt)}")

    train_ds = EffaaDataset(args.train, args.image_root, limit=args.limit,
                            balance_real=args.balance_real)
    eval_ds = EffaaDataset(args.eval, args.image_root, limit=args.eval_limit) if args.eval else None
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"train records: {len(train_ds):,} | eval records: {len(eval_ds) if eval_ds else 0:,}")

    targs = transformers.TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        report_to=[],
        ddp_find_unused_parameters=False,
    )
    # two optimizer groups (FFAA-style): LoRA @ --lr, fully-fine-tuned merger @ --merger-lr.
    optimizer = None
    if args.train_merger:
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
        merger_params = [p for n, p in model.named_parameters()
                         if p.requires_grad and "lora_" not in n]
        optimizer = torch.optim.AdamW(
            [{"params": lora_params, "lr": args.lr},
             {"params": merger_params, "lr": args.merger_lr}],
            weight_decay=targs.weight_decay, betas=(0.9, 0.999), eps=1e-8)
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"optimizer groups: LoRA {sum(p.numel() for p in lora_params)/1e6:.1f}M @ {args.lr} | "
                  f"merger(full-FT) {sum(p.numel() for p in merger_params)/1e6:.1f}M @ {args.merger_lr}")
    trainer = transformers.Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=EffaaCollator(processor, max_len=args.max_len),
        optimizers=(optimizer, None),          # None -> HF builds the cosine schedule over both groups
    )
    trainer.train(resume_from_checkpoint=any(
        d.startswith("checkpoint-") for d in os.listdir(args.out)) if os.path.isdir(args.out) else None)
    trainer.save_model(os.path.join(args.out, "adapter_final"))
    if int(os.environ.get("RANK", 0)) == 0:
        processor.save_pretrained(os.path.join(args.out, "adapter_final"))
        print("saved adapter ->", os.path.join(args.out, "adapter_final"))


if __name__ == "__main__":
    main()
