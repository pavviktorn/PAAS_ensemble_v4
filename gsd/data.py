"""Dataset + augmentation for GSD training/eval.

Ground truth = a `real`/`fake` path component (label 0/1), matching the rest of the PAAS project,
OR a MIDS-style JSON list of {"image": path, "cls_label": 0|1}. Preprocessing matches CLIP
(resize->center-crop->normalize); training adds horizontal flip + Gaussian blur + JPEG recompression
(the robustness augmentations used in the GSD paper).
"""
from __future__ import annotations

import io
import json
import os
import random
from collections import Counter
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

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

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _label_from_path(path: str) -> Optional[int]:
    """Coarse real/fake from a path component (fallback only)."""
    parts = path.lower().split(os.sep)
    if "fake" in parts:
        return 1
    if "real" in parts:
        return 0
    return None


def resolve_label(path: str, num_classes: int, source: str = "get_label_all",
                  cls_label: Optional[int] = None) -> Optional[int]:
    """Map an image to a training label.

    3-class: REAL->0, PAD->1, DEEPFAKE->2, MAKEUP->PAD(1); UNKNOWN -> None (dropped).
    2-class: REAL->0, every spoof (pad/deepfake/makeup) -> 1.
    `source` selects the truth: 'get_label_all' (authoritative), 'cls_label' (json field, binary
    only), or 'path' (coarse fallback).
    """
    if source == "cls_label" and cls_label is not None:
        fine = REAL if int(cls_label) == 0 else PAD       # binary field: no pad/deepfake split
    elif source == "path":
        b = _label_from_path(path)
        fine = REAL if b == 0 else (PAD if b == 1 else UNKNOWN)
    else:
        fine = get_label_all(path)
    if fine == UNKNOWN:
        return None
    if fine == MAKEUP:                                    # makeup attack folded into PAD
        fine = PAD
    if num_classes == 2:
        return 0 if fine == REAL else 1
    return int(fine)                                      # REAL/PAD/DEEPFAKE already 0/1/2


class _RandomJPEG:
    """Re-encode the PIL image as JPEG at a random quality (compression artifacts)."""

    def __init__(self, p: float, min_q: int = 40, max_q: int = 95):
        self.p, self.min_q, self.max_q = p, min_q, max_q

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=random.randint(self.min_q, self.max_q))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_transform(image_size: int, train: bool, cfg=None) -> T.Compose:
    ops: List = []
    if train:
        if cfg is None or cfg.aug_hflip:
            ops.append(T.RandomHorizontalFlip())
        if cfg is not None and cfg.aug_jpeg_p > 0:
            ops.append(_RandomJPEG(cfg.aug_jpeg_p, cfg.aug_jpeg_min_q))
        if cfg is not None and cfg.aug_blur_p > 0:
            ops.append(T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=cfg.aug_blur_p))
    if getattr(cfg, "whole_frame", False):
        # WHOLE FRAME: pad to square (nothing discarded) -> resize. Center-crop would throw away the
        # borders where PAD evidence (screen edges, bezels, hands, moire) lives.
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from paas.preprocess import LetterboxSquare
        ops += [LetterboxSquare(image_size)]
    else:
        ops += [T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size)]
    ops += [
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ]
    return T.Compose(ops)


class FaceDataset(Dataset):
    """`source` is a MIDS-style JSON list of {"image", "cls_label"} or a directory tree.

    Each image's label is resolved by `resolve_label` (default: get_label_all -> real/pad/deepfake);
    images that resolve to UNKNOWN are dropped. Returns (tensor, long_label, path).
    """

    def __init__(self, source: str, image_size: int, train: bool, cfg=None, limit: int = 0):
        self.tf = build_transform(image_size, train, cfg)
        self.image_size = image_size
        self.num_classes = getattr(cfg, "num_classes", 2)
        src_kind = getattr(cfg, "label_source", "get_label_all")
        self.items: List[Tuple[str, int]] = []
        n_drop = 0
        if source.endswith(".json"):
            for r in json.load(open(source)):
                lab = resolve_label(r["image"], self.num_classes, src_kind, r.get("cls_label"))
                if lab is not None:
                    self.items.append((r["image"], lab))
                else:
                    n_drop += 1
        else:
            for dp, _, files in os.walk(source):
                for fn in sorted(files):
                    if os.path.splitext(fn)[1].lower() in IMG_EXT:
                        p = os.path.join(dp, fn)
                        lab = resolve_label(p, self.num_classes, src_kind)
                        if lab is not None:
                            self.items.append((p, lab))
                        else:
                            n_drop += 1
        if limit and limit < len(self.items):
            if train:
                self.items = self.items[:limit]                 # debug cap; train loader shuffles anyway
            else:                                               # strided subset preserves class mix even
                stride = len(self.items) / limit                # if the json is block-ordered by class
                self.items = [self.items[int(i * stride)] for i in range(limit)]
        if not self.items:
            raise ValueError(f"no labelled images under {source}")
        self.n_dropped = n_drop

    def class_counts(self) -> List[int]:
        c = Counter(lab for _, lab in self.items)
        return [c.get(i, 0) for i in range(self.num_classes)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        img = _open_rgb_guarded(path, self.image_size, "gsd")
        return self.tf(img), torch.tensor(int(label), dtype=torch.long), path
