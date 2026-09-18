"""Paper-faithful semantic-subspace estimation for GSD (arXiv 2603.09242, Sec. 4.2-4.3).

This module exists because gsd/householder.py is NOT what the paper specifies. The paper's text
contains zero occurrences of "QR" or "Householder"; it specifies SVD of the frozen branch's
NON-CLS PATCH TOKENS, PER LAYER, keeping the TOP-K RIGHT singular vectors:

  Single-SVD (Sec. 4.2), for sample i at layer l:
      P_bar[i,l] in R^(N x d)  = non-CLS tokens of the FROZEN anchor at layer l
      P_bar[i,l] = U S V^T ;  basis V^(k)[i,l] in R^(d x k) = top-k RIGHT singular vectors

  Batch-SVD (Sec. 4.3, eq. 12) -- the paper's own efficiency approximation:
      M_bar[l] = [P_bar[1,l] ; ... ; P_bar[B,l]] in R^(BN x d)      <- concatenated along TOKENS
      M_bar[l] = U^(b) S^(b) (V^(b))^T ;  shared basis V^(b,k)[l] in R^(d x k)

Two differences from gsd/householder.py that change what is being estimated, not just how:
  * the paper decomposes a (BN x d) PATCH-TOKEN matrix; householder.py decomposes a (d x B) matrix
    of image-level GAP vectors. The first finds directions of variation across patch tokens (spatial
    semantic structure, which the paper explicitly prefers over the CLS token); the second finds
    directions of variation across whichever images share a batch.
  * top-k right singular vectors are ordered by captured variance. The first k columns of a QR
    factor are ordered by nothing at all -- they span the first k input columns, so shuffling the
    batch changes the subspace.

Implementation note: we need only the top k right singular vectors, so we eigendecompose the
d x d Gram matrix G = M^T M instead of factorising the (BN x d) matrix. eigh on a symmetric PSD
matrix is exact and DETERMINISTIC (no randomised range finder), and d=1024 makes it trivial next to
the BN x d GEMM that forms G. Right singular vectors of M are exactly the eigenvectors of M^T M,
ordered by eigenvalue = squared singular value.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def top_k_right_singular(M: torch.Tensor, k: int, center: bool = False,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """M: (n, d) token matrix. Returns V_k: (d, k), the top-k right singular vectors of M, in FP32.

    center=False matches the paper: eq. 12 decomposes the token matrix itself, with no centring.
    (householder.py centres by the batch mean, which is a PCA-style choice the paper does not make.)

    TWO PRECISION HAZARDS, both measured, both handled here:
      1. This runs inside the trainer's bf16 autocast context (gsd/engine.py:94). Autocast re-casts
         MATMUL OUTPUTS regardless of input dtype, so `M.float()` alone still produced a bf16 Gram
         and a basis orthonormality error of 4.7e-04 instead of ~4e-07. The whole computation is
         therefore wrapped in an autocast-DISABLED region -- upcasting the input is not sufficient.
      2. Returning `V.to(M.dtype)` cast the basis back down to bf16, costing ~5e-04 of orthonormality
         on its own however accurately it was computed, which stops the projector V V^T from being
         idempotent. The basis is returned in FP32; callers cast at the point of the matmul.

    fp32 (not fp64) for the eigendecomposition is a measured cost choice: on a (128*576, 1024) batch
    fp64 eigh is 66.4 ms against 33.0 ms, i.e. 2.1 h vs 1.3 h of pure decomposition over a 3-epoch
    4-layer run, for a top-64 projector difference of 7.1e-07. The reference-basis path in
    train_gsd.py keeps fp64 because it ACCUMULATES a Gram over thousands of batches, where summation
    error genuinely compounds, and it runs once.
    """
    if M.dim() != 2:
        raise ValueError(f"M must be (n, d); got {tuple(M.shape)}")
    n, d = M.shape
    k = int(max(1, min(k, d, n)))
    dev = "cuda" if M.is_cuda else "cpu"
    with torch.autocast(device_type=dev, enabled=False):
        X = M.float()
        if center:
            X = X - X.mean(dim=0, keepdim=True)
        G = X.t() @ X                      # (d, d) symmetric PSD; eigvecs == right singular vectors
        if G.dtype != torch.float32:        # belt and braces: autocast must not have touched this
            raise RuntimeError(f"Gram came out {G.dtype}, expected float32 -- autocast leaked in")
        # eigh returns ascending eigenvalues; the top-k are the LAST k, reversed to descending
        evals, evecs = torch.linalg.eigh(G.to(dtype))
        V = evecs[:, -k:].flip(dims=(1,))
    return V.float().contiguous()


@torch.no_grad()
def batch_svd_basis(frozen_hidden: torch.Tensor, k: int, drop_cls: bool = True) -> torch.Tensor:
    """Paper eq. 12 for ONE layer.

    frozen_hidden: (B, 1+N, D) hidden states of the FROZEN anchor at this layer.
    Returns V: (D, k) shared semantic basis for the whole mini-batch at this layer.
    """
    if frozen_hidden.dim() != 3:
        raise ValueError(f"frozen_hidden must be (B, 1+N, D); got {tuple(frozen_hidden.shape)}")
    P = frozen_hidden[:, 1:, :] if drop_cls else frozen_hidden      # (B, N, D) non-CLS tokens
    B, N, D = P.shape
    M = P.reshape(B * N, D)                                          # (BN, D)  <- eq. 12
    return top_k_right_singular(M, k)


@torch.no_grad()
def single_svd_basis(frozen_hidden_one: torch.Tensor, k: int, drop_cls: bool = True) -> torch.Tensor:
    """Paper Sec. 4.2 (Single-SVD): a per-SAMPLE basis, used by the per-sample inference protocol.

    frozen_hidden_one: (1+N, D) for one image. Returns V: (D, k).
    """
    P = frozen_hidden_one[1:, :] if drop_cls else frozen_hidden_one
    return top_k_right_singular(P, k)
