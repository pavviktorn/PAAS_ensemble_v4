#!/usr/bin/env python
"""Merge a trained LoRA adapter into the Qwen3-VL base model -> a standalone merged model dir.

  ./venv/bin/python qwen/merge_lora.py --base base_models/Qwen3-VL-4B-Instruct \
      --adapter runs/qwen_lora/adapter_final --out runs/qwen_merged
"""
import argparse

import torch
import transformers
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = transformers.AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    processor = transformers.AutoProcessor.from_pretrained(args.base)
    processor.save_pretrained(args.out)
    print("merged model ->", args.out)


if __name__ == "__main__":
    main()
