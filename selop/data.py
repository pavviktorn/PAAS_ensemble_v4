"""MIDS dataset for SeLop.

The ground-truth label is derived from the image PATH via `get_label_all`
(REAL=0, PAD=1, DEEPFAKE=2, MAKEUP=3, UNKNOWN=-1) — not from the json's
`cls_label`/`answers` fields. For SeLop's binary task the 4-class label is
mapped to {real=0, fake=1} (PAD/DEEPFAKE/MAKEUP -> fake); UNKNOWN (-1) samples
are dropped. Set `num_classes=4` to keep the raw 4-class label instead.

To avoid re-parsing the 1.5 GB train json in every DDP process, a slim
`<json>.selop_idx.<mode>.tsv` cache (path<TAB>label per line) is built once.
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .get_label import get_label_all, REAL, PAD, DEEPFAKE, MAKEUP, UNKNOWN


# ---- UNREADABLE-IMAGE GUARD ---------------------------------------------------------------
# A missing/corrupt file used to be swapped for a BLACK frame with no log line. That is invisible
# and it corrupts results rather than stopping them: when the testset's deepfake folder was
# briefly moved aside, 9,725 of 30,197 eval images (32%) became black squares, and GSD still
# reported "anchor U(1024,16) from 30197 refs" and a full val split -- an anchor built one-third
# from a constant image, embedded into every checkpoint, with nothing in the log to show it.
#
# Policy: tolerate genuinely rare corruption (one bad JPEG must not kill a 6-hour run) but ABORT
# on anything systematic. Every substitution is logged (first 10 with the path), and the job dies
# once the count exceeds IMG_MISS_MAX or the rate exceeds IMG_MISS_RATE.
# NOTE: counters are PER DataLoader WORKER PROCESS, so with num_workers=W the effective absolute
# cap is IMG_MISS_MAX * W; the RATE check is what catches systematic breakage quickly.
_MISS = {"n": 0, "seen": 0}
_MISS_MAX = int(os.environ.get("IMG_MISS_MAX", "100"))     # absolute cap per worker
_MISS_RATE = float(os.environ.get("IMG_MISS_RATE", "0.01"))  # 1% of reads
_MISS_MIN_SEEN = int(os.environ.get("IMG_MISS_MIN_SEEN", "200"))


def _open_rgb_guarded(path, size, tag):
    """Open an image as RGB. Substitutes a black frame ONLY for rare failures, and only after
    logging; raises once the failures look systematic."""
    _MISS["seen"] += 1
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        _MISS["n"] += 1
        n, seen = _MISS["n"], _MISS["seen"]
        if n <= 10:
            print(f"[{tag}] UNREADABLE IMAGE #{n}: {path} ({type(exc).__name__})", flush=True)
        rate = n / max(seen, 1)
        if n > _MISS_MAX or (seen >= _MISS_MIN_SEEN and rate > _MISS_RATE):
            raise RuntimeError(
                f"[{tag}] too many unreadable images: {n} of {seen} reads ({rate*100:.2f}%). "
                f"Last: {path}. This is a DATA problem, not a transient one -- a black-frame "
                f"substitution at this rate silently corrupts anchors, val metrics and checkpoint "
                f"selection. Check the manifest paths still exist. "
                f"Override with IMG_MISS_MAX / IMG_MISS_RATE if this is genuinely expected."
            ) from exc
        return Image.new("RGB", (size, size))

# 3-class scheme (matches the GSD project): real / pad / deepfake.
# MAKEUP is folded into PAD; UNKNOWN images are dropped.
CLASS_NAMES_3 = ("real", "pad", "deepfake")
CLASS_NAMES_2 = ("real", "fake")


def derive_label(path, num_classes):
    """Path -> training label via get_label_all. Returns None to drop the sample."""
    lab = get_label_all(path)
    if lab == UNKNOWN:
        return None
    if num_classes == 2:
        return 0 if lab == REAL else 1                 # real vs fake
    if num_classes == 3:
        if lab == REAL:
            return 0                                   # real
        if lab == DEEPFAKE:
            return 2                                   # deepfake
        return 1                                       # PAD or MAKEUP -> pad
    raise ValueError(f"num_classes must be 2 or 3, got {num_classes}")


def class_names(num_classes):
    return CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_2

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(image_size=336, train=True, whole_frame=False):
    """whole_frame=True letterboxes the WHOLE frame for BOTH train and eval (no crop, no squash).

    The legacy path was also train/serve inconsistent: RandomResizedCrop while training but a
    Resize-squash at eval. Whole-frame uses one geometry everywhere."""
    norm = transforms.Normalize(CLIP_MEAN, CLIP_STD)
    if whole_frame:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from paas.preprocess import LetterboxSquare
        ops = [LetterboxSquare(image_size)]
        if train:
            ops.append(transforms.RandomHorizontalFlip(0.5))
        return transforms.Compose(ops + [transforms.ToTensor(), norm])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), ratio=(0.9, 1.0 / 0.9),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        norm,
    ])


def build_index(json_path, num_classes=2, cache=True, log=print):
    """Parse a MIDS json into a list[(path, label)] using `get_label_all`;
    cache to a slim .tsv keyed by num_classes."""
    tsv = f"{json_path}.selop_idx.{num_classes}c.tsv"
    if cache and os.path.exists(tsv) and os.path.getmtime(tsv) >= os.path.getmtime(json_path):
        samples = []
        with open(tsv) as f:
            for line in f:
                p, lab = line.rstrip("\n").rsplit("\t", 1)
                samples.append((p, int(lab)))
        log(f"[data] loaded cached index {tsv} ({len(samples)} samples)")
        return samples

    log(f"[data] parsing {json_path} ...")
    with open(json_path) as f:
        data = json.load(f)
    samples = []
    dropped = 0
    for rec in data:
        img = rec.get("image")
        if img is None:
            continue
        lab = derive_label(img, num_classes)
        if lab is None:
            dropped += 1
            continue
        samples.append((img, lab))
    log(f"[data] {len(samples)} samples kept, {dropped} dropped (UNKNOWN)")
    if cache:
        tmp = tsv + ".tmp"
        with open(tmp, "w") as f:
            for p, lab in samples:
                f.write(f"{p}\t{lab}\n")
        os.replace(tmp, tsv)
        log(f"[data] wrote index cache {tsv}")
    return samples


class MidsBinaryDataset(Dataset):
    def __init__(self, samples, transform, image_size=336, return_index=False):
        self.samples = samples
        self.transform = transform
        self.image_size = image_size
        self.return_index = return_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = _open_rgb_guarded(path, self.image_size, "selop")
        x = self.transform(img)
        if self.return_index:
            return x, label, idx
        return x, label
