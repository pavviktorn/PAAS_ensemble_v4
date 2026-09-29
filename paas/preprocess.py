"""Canonical WHOLE-FRAME preprocessing, shared by every detector for BOTH training and inference.

Rationale: the face is never cropped. Cropping (CLIP's resize+center-crop, or RandomResizedCrop)
discards the image border -- which for presentation attacks is exactly where the evidence lives
(screen borders, moire, hands, bezels). It also created train/serve skew, because the 9-class
ensemble already letterboxes at inference (ensemble9/mids9lib/transform.py) while its training used
CLIP's center-crop, and SeLop trained on RandomResizedCrop but served Resize-squash.

Whole-frame = zero-pad to a square (aspect preserved, nothing discarded) -> resize -> CLIP-normalize.
This matches the deployed 9-class inference pipeline exactly.

Checkpoints trained this way record `whole_frame: true` in their saved config so the inference
wrappers can pick the matching transform and legacy checkpoints keep their original behaviour.
"""
from __future__ import annotations

import numpy as np
import torch

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_MEAN = torch.tensor(CLIP_MEAN).view(3, 1, 1)
_STD = torch.tensor(CLIP_STD).view(3, 1, 1)


def letterbox_rgb(rgb: np.ndarray, size: int = 336) -> np.ndarray:
    """RGB uint8 HxWx3 -> square uint8 size x size x 3, aspect preserved, zero-padded (no crop)."""
    import cv2
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=rgb.dtype)
    top, left = (s - h) // 2, (s - w) // 2
    canvas[top:top + h, left:left + w] = rgb
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def letterbox_to_tensor(rgb: np.ndarray, size: int = 336) -> torch.Tensor:
    """RGB uint8 HxWx3 -> CLIP-normalized float tensor (3, size, size). Whole frame, no crop."""
    img = letterbox_rgb(rgb, size)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


class LetterboxSquare:
    """torchvision-Compose-compatible PIL -> PIL whole-frame square (pad, then resize)."""

    def __init__(self, size: int = 336):
        self.size = size

    def __call__(self, img):
        from PIL import Image
        rgb = np.asarray(img.convert("RGB"))
        return Image.fromarray(letterbox_rgb(rgb, self.size))

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size})"
