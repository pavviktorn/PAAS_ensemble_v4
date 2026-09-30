"""eFFAA (LLaVA-format) conversations -> Qwen3-VL chat-template training samples.

Each record: {"id", "image" (relative to --image-root), "conversations":
[{"from": "human", "value": "<image>\n<question>"}, {"from": "gpt", "value": "<answer>"}]}.

The collator tokenizes prompt+answer with the processor's chat template and masks the prompt
tokens to -100 so only the assistant answer is learned (single-turn; multi-turn records use
only the first human/gpt pair, which is all eFFAA contains).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset


class EffaaDataset(Dataset):
    def __init__(self, json_path: str, image_root: str, limit: int = 0,
                 balance_real: float = 0.0):
        import re
        with open(json_path) as fh:
            data = json.load(fh)
        self.items = []
        reals = []
        for r in data:
            conv = r.get("conversations") or []
            if len(conv) < 2 or "image" not in r:
                continue
            q = conv[0]["value"].replace("<image>", "").strip()
            a = conv[1]["value"].strip()
            item = (os.path.join(image_root, r["image"]), q, a)
            self.items.append(item)
            if balance_real > 0:
                m = re.search(r"Analysis result:\s*(\w+)", a)
                if m and m.group(1).lower() == "real":
                    reals.append(item)
        if balance_real > 0 and reals:
            # oversample REAL records by (balance_real - 1) extra copies (fractional tail included)
            extra = balance_real - 1.0
            whole, frac = int(extra), extra - int(extra)
            add = reals * whole + reals[:int(len(reals) * frac)]
            self.items.extend(add)
            print(f"[data] balance_real={balance_real}: +{len(add):,} oversampled real records "
                  f"-> {len(self.items):,} total")
        if limit and limit < len(self.items):
            stride = len(self.items) / limit          # strided subset keeps the class mix
            self.items = [self.items[int(i * stride)] for i in range(limit)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@dataclass
class EffaaCollator:
    processor: object
    max_len: int = 1536

    def __call__(self, batch):
        texts, prompts, images = [], [], []
        for path, q, a in batch:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (336, 336))
            images.append(img)
            user = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            full = user + [{"role": "assistant", "content": [{"type": "text", "text": a}]}]
            prompts.append(self.processor.apply_chat_template(
                user, tokenize=False, add_generation_prompt=True))
            texts.append(self.processor.apply_chat_template(
                full, tokenize=False, add_generation_prompt=False))

        enc = self.processor(text=texts, images=images, padding=True,
                             truncation=True, max_length=self.max_len, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        # mask the prompt part: everything before the assistant answer
        for b, p in enumerate(prompts):
            plen = len(self.processor.tokenizer(p, add_special_tokens=False)["input_ids"])
            # image placeholder tokens expand inside the processor; recompute prompt length on
            # the encoded sequence by matching the assistant header instead when lengths differ.
            ids = enc["input_ids"][b]
            n = int(enc["attention_mask"][b].sum())
            # find the LAST assistant header token sequence in the encoded ids
            hdr = self.processor.tokenizer("<|im_start|>assistant\n",
                                           add_special_tokens=False)["input_ids"]
            pos = -1
            for s in range(n - len(hdr), -1, -1):
                if ids[s:s + len(hdr)].tolist() == hdr:
                    pos = s + len(hdr)
                    break
            labels[b, :pos if pos > 0 else min(plen, n)] = -100
        enc["labels"] = labels
        return enc
