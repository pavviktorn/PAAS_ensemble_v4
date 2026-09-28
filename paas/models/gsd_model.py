"""Wrapper around the GSD detector (Geometric Semantic Decoupling, arXiv:2603.09242).

Frozen + trainable CLIP ViT-L/14-336; the trained checkpoint embeds a fixed semantic basis U
(built from the testset at train time) so single-image scoring works with no reference batch.
Exposes the uniform ``score_frames`` used by the PAAS pipeline: per frame, the 3-class
(real/pad/deepfake) fake-score ``1 - P(real)``.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # v3 project root


class GSDModel:
    name = "gsd"

    def __init__(self, ckpt_path: str, clip_path: str, device: str = "cuda:0",
                 amp_dtype: str = "bf16"):
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        import torch
        from gsd.config import GSDConfig, CLASS_NAMES
        from gsd.model import GSDDetector
        from gsd.data import build_transform

        self.torch = torch
        self.device = device
        payload = torch.load(ckpt_path, map_location="cpu")
        # Which GSD is in this checkpoint? train_gsd.py --faithful stamps gsd_variant, so the two
        # arms can be served side by side without the caller having to know which is which.
        self.variant = payload.get("gsd_variant", "deployed")
        from paas.compat import remap_clip_state_dict
        if self.variant == "paper_faithful":
            from gsd.faithful import GSDFaithfulConfig, GSDFaithfulDetector
            cfg = GSDFaithfulConfig.from_dict(payload["config"])
            cfg.clip_path = clip_path
            self.cfg = cfg
            model = GSDFaithfulDetector(cfg)
            model.trainable.load_state_dict(remap_clip_state_dict(payload["trainable"]))
            model.head.load_state_dict(payload["head"])
            # a_l / b_l are model parameters and MUST be restored, or the served lambda reverts to its
            # init and the detector silently becomes a different model than the one that was selected
            if payload.get("suppress") is not None:
                model.suppress.load_state_dict(payload["suppress"])
            else:
                print("[gsd] WARNING: faithful checkpoint carries no 'suppress' state; lambda "
                      "parameters fall back to their INIT (a=%.3g, b=%.3g). The served model is then "
                      "not the trained one." % (cfg.a_init, cfg.b_init))
            rb = payload.get("ref_bases")
            if rb:
                model.set_reference_bases({int(k): v.to(device) for k, v in rb.items()})
                # Serving REQUIRES the reference protocol: under Batch-SVD a frame's score depends on
                # which other frames share its request, so the same image scores differently at
                # different batch compositions and a calibrated threshold does not transfer. If the
                # checkpoint was SELECTED under a different protocol, say so -- the val metric that
                # chose it then describes different behaviour from what is served here.
                trained_proto = getattr(cfg, "infer_protocol", "batch")
                if trained_proto != "reference":
                    print(f"[gsd] NOTE: checkpoint was validated with infer_protocol="
                          f"{trained_proto!r} but serving forces 'reference' for per-frame "
                          f"determinism. The metric that selected best.pt is therefore not this "
                          f"model's behaviour. Retrain with infer_protocol=reference to align them.")
                model.cfg.infer_protocol = "reference"
            else:
                print("[gsd] WARNING: faithful checkpoint has no ref_bases; falling back to "
                      "Batch-SVD inference, so a frame's score depends on its batch")
            self.model = model.to(device).eval()
            self.tfm = build_transform(cfg.image_size, train=False, cfg=cfg)
            self.names = list(CLASS_NAMES) if cfg.num_classes == 3 else ["real", "fake"]
            self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16,
                        "fp32": torch.float32}.get(amp_dtype, torch.bfloat16)
            print(f"[gsd] loaded PAPER-FAITHFUL GSD (k={cfg.k}, layers={model.gsd_layer_ids}, "
                  f"protocol={model.cfg.infer_protocol}, lambda_max={cfg.lambda_max})")
            return
        cfg = GSDConfig.from_dict(payload["config"])
        cfg.clip_path = clip_path                      # use the v3 shared CLIP backbone, not the train-time path
        self.cfg = cfg
        model = GSDDetector(cfg)
        model.trainable.load_state_dict(remap_clip_state_dict(payload["trainable"]))
        model.head.load_state_dict(payload["head"])
        if payload.get("anchor_U") is not None:
            model.set_fixed_U(payload["anchor_U"].to(device))
            model.use_fixed_anchor(True)   # deterministic: a frame's score must not depend
                                           # on which other frames share its batch
        elif getattr(model, "_fixed_U", None) is None:
            print("[gsd] WARNING: checkpoint has no embedded anchor U; single-image scores are undefined")
        self.model = model.to(device).eval()
        self.tfm = build_transform(cfg.image_size, train=False, cfg=cfg)
        self.names = list(CLASS_NAMES) if cfg.num_classes == 3 else ["real", "fake"]
        self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16,
                    "fp32": torch.float32}.get(amp_dtype, torch.bfloat16)

    def score_frames(self, rgb_list: List[np.ndarray], batch_size: int = 64) -> List[dict]:
        """rgb_list: HxWx3 uint8 arrays. Returns one dict per frame (input order):
        {"fake": float|None, "type_probs": [r,p,d]|None, "pred": str|None, "error": str|None}."""
        torch = self.torch
        from PIL import Image
        out: List[Optional[dict]] = [None] * len(rgb_list)
        with torch.no_grad():
            for i in range(0, len(rgb_list), batch_size):
                chunk = rgb_list[i:i + batch_size]
                try:
                    x = torch.stack([self.tfm(Image.fromarray(r)) for r in chunk]).to(self.device)
                    with torch.autocast(device_type="cuda", dtype=self.amp,
                                        enabled=(self.device.startswith("cuda") and self.amp != torch.float32)):
                        logits = self.model(x)
                    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                    for j, pr in enumerate(probs):
                        cls = int(pr.argmax())
                        tp = [float(pr[0]), float(pr[1]), float(pr[2])] if len(pr) == 3 else None
                        out[i + j] = {"fake": float(1.0 - pr[0]), "type_probs": tp,
                                      "pred": self.names[cls], "error": None}
                except Exception as exc:                    # a bad frame in the chunk fails only that chunk
                    for j in range(len(chunk)):
                        out[i + j] = {"fake": None, "type_probs": None, "pred": None, "error": repr(exc)}
        return out
