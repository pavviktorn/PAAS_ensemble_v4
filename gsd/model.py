"""Dual-stream GSD detector (GSD paper, arXiv 2603.09242).

  * Frozen stream  : CLIP ViT-L/14 -> global guide features g_i (semantic anchor source).
  * Trainable stream: a CLIP ViT-L/14 copy whose FINAL N encoder layers have GSD injected -- each
                      such layer's patch tokens F_l are projected to F_l(I - U U^T), removing the
                      per-batch semantic subspace U estimated from the frozen guides.
  * Head           : pool the trainable final features -> LayerNorm -> Linear -> num_classes logits
                      (CrossEntropy; default 3-class real/pad/deepfake).

U is built per mini-batch from the FROZEN guides (so it is detached and recomputed every step; no
running statistics). GSD therefore needs batch_size >= 2 (a single image has no batch centroid);
for single-image inference, set a fixed anchor with `set_fixed_anchor()`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from .config import GSDConfig
from .householder import semantic_basis
from .projection import desemanticize


def _pool(hidden: torch.Tensor, how: str) -> torch.Tensor:
    """hidden: (B, 1+N, D). 'gap' -> mean over patch tokens (exclude CLS); 'cls' -> token 0."""
    return hidden[:, 0, :] if how == "cls" else hidden[:, 1:, :].mean(dim=1)


def _gsd_hook(module, inputs, output):
    """Forward hook: de-semanticize a layer's patch tokens using the basis stashed on the layer as
    `module._gsd_U`. Reading U from the *module argument* (not a captured `self`) keeps this correct
    under nn.DataParallel, where each replica sets `_gsd_U` on its own layer copy."""
    U = getattr(module, "_gsd_U", None)
    if U is None or U.numel() == 0:
        return output
    # tf4.37 CLIPEncoderLayer returned a TUPLE; tf5.13 returns a plain TENSOR. Handle both.
    is_tuple = isinstance(output, tuple)
    hs = output[0] if is_tuple else output              # (B, 1+N, D)
    cls, patches = hs[:, :1, :], hs[:, 1:, :]
    patches = desemanticize(patches, U)                 # F'(I - U U^T) on patch tokens
    new_hs = torch.cat([cls, patches], dim=1)
    return ((new_hs,) + tuple(output[1:])) if is_tuple else new_hs


class GSDDetector(nn.Module):
    def __init__(self, cfg: GSDConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.frozen = CLIPVisionModel.from_pretrained(cfg.clip_path)
        self.trainable = CLIPVisionModel.from_pretrained(cfg.clip_path)
        self.dim = self.frozen.config.hidden_size
        self.n_layers = self.frozen.config.num_hidden_layers
        self.gsd_layer_ids = list(range(self.n_layers - cfg.n_gsd_layers, self.n_layers))

        # frozen semantic extractor: never trains
        self.frozen.eval()
        for p in self.frozen.parameters():
            p.requires_grad = False

        self._set_trainable_scope()

        self.head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Dropout(0.1),
            nn.Linear(self.dim, cfg.num_classes),
        )

        # fixed anchor for single-image / batch<2 inference (per-batch U is undefined there)
        self._fixed_U: Optional[torch.Tensor] = None
        # Use the fixed anchor ONLY when explicitly asked (inference / validation). This must NOT
        # be derived from self.training: gsd/engine._set_train_mode restores the CHILD modules
        # after validation but not the parent flag, so a mode-derived rule silently switched
        # optimization onto the testset-derived anchor -> train/serve drift AND test leakage.
        self._infer_anchor = False
        self._register_gsd_hooks()

    # ------------------------------------------------------------------ setup
    def _set_trainable_scope(self) -> None:
        scope = self.cfg.trainable
        for p in self.trainable.parameters():
            p.requires_grad = (scope == "full")
        if scope == "lastN":
            layers = self.trainable.encoder.layers
            for p in layers[-self.cfg.n_gsd_layers:].parameters():
                p.requires_grad = True
            for p in self.trainable.post_layernorm.parameters():
                p.requires_grad = True
        # scope == "head": trainable backbone fully frozen (only self.head trains)

    def _register_gsd_hooks(self) -> None:
        layers = self.trainable.encoder.layers
        for idx in self.gsd_layer_ids:
            layers[idx]._gsd_U = None                        # plain attr on the layer module (DP-replicated)
            layers[idx].register_forward_hook(_gsd_hook)

    # ------------------------------------------------------------------ anchor / U
    @torch.no_grad()
    def _compute_U(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run the frozen stream and build the semantic basis U (global, or per GSD layer)."""
        cfg = self.cfg
        if cfg.per_layer_guide:
            out = self.frozen(pixel_values, output_hidden_states=True)
            hs = out.hidden_states                          # tuple len L+1; hs[l+1] = output of layer l
            return {idx: semantic_basis(_pool(hs[idx + 1], cfg.guide_pool), cfg.k, cfg.qr_method)
                    for idx in self.gsd_layer_ids}
        out = self.frozen(pixel_values)
        guide = _pool(out.last_hidden_state, cfg.guide_pool)   # (B, D)
        U = semantic_basis(guide, cfg.k, cfg.qr_method)        # (D, K)
        return {-1: U}                                          # -1 = shared U for all GSD layers

    @torch.no_grad()
    def set_fixed_anchor(self, guides: torch.Tensor) -> None:
        """Freeze a semantic basis from a reference set of guide vectors (B_ref, D) for single-image
        / small-batch inference, where a per-batch centroid is undefined."""
        self._fixed_U = semantic_basis(guides, self.cfg.k, self.cfg.qr_method)

    def use_fixed_anchor(self, on: bool = True) -> None:
        """Inference/validation: score every image against the embedded anchor, so a frame's
        score never depends on which other frames share its batch. Training leaves this False
        and keeps the paper's per-batch basis."""
        self._infer_anchor = bool(on)

    @torch.no_grad()
    def set_fixed_U(self, U: torch.Tensor) -> None:
        """Install a precomputed semantic basis U (D, K) directly (e.g. loaded from anchor.pt)."""
        self._fixed_U = U

    # ------------------------------------------------------------------ forward
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # AT INFERENCE (eval mode) always use the FIXED anchor when one is embedded: a batch-derived
        # basis makes a frame's score depend on whichever other frames share its batch, so the same
        # image scores differently across request sizes/compositions -- and the threshold calibrated
        # at one batch composition would not transfer. Training (self.training) keeps the paper's
        # per-batch estimation, so the recipe is unchanged.
        if self._fixed_U is not None and self._infer_anchor:
            U_map = {-1: self._fixed_U.to(pixel_values.device)}
        elif pixel_values.shape[0] >= 2:
            U_map = self._compute_U(pixel_values)           # {-1: U} shared, or {idx: U} per layer
        elif self._fixed_U is not None:
            U_map = {-1: self._fixed_U.to(pixel_values.device)}
        else:
            U_map = {}                                      # too small to estimate -> no projection
        # stash U on each GSD layer (registered submodule path -> remapped per DP replica)
        layers = self.trainable.encoder.layers
        for idx in self.gsd_layer_ids:
            layers[idx]._gsd_U = U_map.get(idx, U_map.get(-1)) if U_map else None
        out = self.trainable(pixel_values)                  # hooks de-semanticize the last N layers
        pooled = _pool(out.last_hidden_state, self.cfg.head_pool)
        return self.head(pooled)                            # (B, num_classes) logits

    # convenience for optimizers
    def param_groups(self):
        head = list(self.head.parameters())
        bb = [p for p in self.trainable.parameters() if p.requires_grad]
        groups = [{"params": head, "lr": self.cfg.head_lr}]
        if bb:
            groups.append({"params": bb, "lr": self.cfg.lr})
        return groups
