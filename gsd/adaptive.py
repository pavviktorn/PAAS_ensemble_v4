"""Layer-aware, sample-adaptive semantic suppression for GSD (arXiv 2603.09242, eq. 9-10).

gsd/projection.py implements the lambda = 1 special case:  f' = f - U U^T f  (full removal).
The paper does not do that. It defines a coefficient that varies per SAMPLE and per LAYER:

    semantic occupancy (eq. 9)      r[i,l] = || f[i,l] V V^T ||_F^2  /  || f[i,l] ||_F^2
    adaptive suppression (eq. 10)   f_hat[i,l] = f[i,l] - lambda[i,l] * f_parallel[i,l]
                                    lambda[i,l] = lambda_max * sigmoid(a_l * r[i,l] + b_l)

a_l and b_l are LEARNABLE scalars per injected layer (paper defaults: a=2, b=1, lambda_max=1).
Samples whose representation sits mostly inside the semantic subspace get suppressed harder; each
layer calibrates its own sensitivity.

Why this matters rather than being a tuning detail: the paper's Table 2 sweeps lambda over
{0, 0.2, 0.4, 0.6, 0.8} and its analysis derives "the existence of a non-trivial optimal suppression
strength that balances semantic interference removal with the preservation of discriminative
forensic cues". lambda = 1 sits outside that swept range and discards whatever forensic signal lies
inside the removed subspace. Note lambda_max = 1 does NOT make lambda 1: sigmoid() is strictly
below 1, so the coefficient is strictly inside (0, lambda_max).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def semantic_occupancy(feats: torch.Tensor, V: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Paper eq. 9. feats: (B, T, D) trainable-branch tokens; V: (D, K) frozen basis (detached).

    Returns r: (B,) in [0, 1] -- the fraction of representation ENERGY inside span(V).
    Computed via the projection coefficients: ||f V V^T||_F = ||f V||_F because V is orthonormal,
    which avoids materialising the (D, D) projector.
    """
    coeff = feats @ V                                     # (B, T, K) coordinates in the subspace
    # REDUCE IN FP32. r drives lambda, and a bf16 sum over T*K elements loses enough precision to
    # move the suppression coefficient. The matmul above stays in the activation dtype (it is the
    # expensive part); only the reductions are upcast, which is nearly free.
    num = coeff.float().pow(2).flatten(1).sum(dim=1)      # ||f_parallel||_F^2  per sample
    den = feats.float().pow(2).flatten(1).sum(dim=1)      # ||f||_F^2           per sample
    return num / (den + eps)


class AdaptiveSuppression(nn.Module):
    """Holds the learnable (a_l, b_l) for ONE injected layer and applies eq. 10.

    One module per GSD layer, so `a` and `b` are genuinely layer-specific parameters that receive
    gradients -- the "layer-aware" half of the paper's claim. The basis V is always detached: it
    comes from the frozen anchor and must not be trained through.
    """

    def __init__(self, a_init: float = 2.0, b_init: float = 1.0, lambda_max: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))
        self.lambda_max = float(lambda_max)
        # diagnostics only; never read by the forward path
        self.register_buffer("_last_lambda_mean", torch.zeros(()), persistent=False)
        self.register_buffer("_last_r_mean", torch.zeros(()), persistent=False)

    def lambda_for(self, r: torch.Tensor) -> torch.Tensor:
        """r: (B,) occupancy -> lambda: (B,) in (0, lambda_max)."""
        return self.lambda_max * torch.sigmoid(self.a * r + self.b)

    def forward(self, feats: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, D) trainable tokens; V: (D, K) frozen basis. Returns the same shape."""
        if V is None or V.numel() == 0:
            return feats
        # V arrives in FP32 from gsd/batch_svd.py and is cast HERE, at the matmul, rather than being
        # stored down-cast -- see the precision note in top_k_right_singular.
        Vd = V.detach().to(feats.dtype)
        r = semantic_occupancy(feats, Vd)                          # (B,)  eq. 9, fp32 reductions
        lam = self.lambda_for(r.to(self.a.dtype)).to(feats.dtype)  # (B,)  eq. 10
        coeff = feats @ Vd                                         # (B, T, K)
        par = coeff @ Vd.t()                                       # (B, T, D) = f_parallel
        with torch.no_grad():
            self._last_lambda_mean.fill_(float(lam.mean()))
            self._last_r_mean.fill_(float(r.mean()))
        return feats - lam.view(-1, 1, 1) * par                    # eq. 10
