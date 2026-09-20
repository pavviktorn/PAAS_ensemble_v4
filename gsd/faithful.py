"""Paper-faithful GSD detector (arXiv 2603.09242) -- an ADDITIONAL arm, not a replacement.

gsd/model.py stays exactly as it is: it is the deployed detector and it is a member of the running
like-for-like comparison, so changing it mid-experiment would destroy that comparison. This module
is the paper's actual method, for the first time, as a separate model:

    per-layer  : the semantic basis is estimated at EACH injected layer from that layer's frozen
                 hidden states, not once from the final layer and shared (model.py:113 returns {-1: U})
    patch-token: eq. 12 decomposes M_bar[l] in R^(BN x d), the frozen NON-CLS tokens concatenated
                 across the batch, not a (d x B) matrix of image-level GAP vectors
    top-k SVD  : top-k RIGHT singular vectors, ordered by captured variance, not the first k columns
                 of a QR factor (which are ordered by nothing -- see gsd/batch_svd.py)
    adaptive   : f_hat = f - lambda[i,l] * f_parallel with lambda = lambda_max * sigmoid(a_l r + b_l),
                 learnable per layer, driven by per-sample semantic occupancy -- not fixed lambda = 1

Paper hyperparameters (Sec. 5.1): last four layers, a=2, b=1, lambda_max=1, k=64, CLIP ViT-L/14,
AdamW, backbone lr 1e-6, batch 128, blur+JPEG augmentation, full end-to-end fine-tuning.

Inference protocols (Sec. 4.3). All three are supported because they are part of the method:
    batch      : Batch-SVD on the incoming mini-batch (paper default)
    per-sample : Single-SVD per image, so a score never depends on batch composition
    reference  : a reusable basis precomputed from a reference set (embedded in the checkpoint)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from .adaptive import AdaptiveSuppression
from .batch_svd import batch_svd_basis, single_svd_basis
from .config import GSDConfig


@dataclass
class GSDFaithfulConfig(GSDConfig):
    """GSDConfig plus the fields the paper's method needs. Inherits every training field."""
    # ---- eq. 10: layer-aware, sample-adaptive suppression ----
    lambda_max: float = 1.0            # paper Sec. 5.1
    a_init: float = 2.0                # paper Sec. 5.1
    b_init: float = 1.0                # paper Sec. 5.1
    # ---- eq. 9/10 scope: the paper's f[i,l] is the FULL (N+1) x d token matrix, CLS included ----
    suppress_cls: bool = True
    # ---- basis estimation ----
    basis_from_patch_tokens: bool = True   # paper: non-CLS tokens of the frozen branch
    infer_protocol: str = "batch"          # 'batch' | 'per_sample' | 'reference'

    def validate(self) -> "GSDFaithfulConfig":
        super().validate()
        if self.lambda_max < 0:
            raise ValueError("lambda_max must be >= 0")
        if self.infer_protocol not in ("batch", "per_sample", "reference"):
            raise ValueError(f"infer_protocol must be batch|per_sample|reference, got {self.infer_protocol}")
        return self


def _pool(hidden: torch.Tensor, how: str) -> torch.Tensor:
    return hidden[:, 0, :] if how == "cls" else hidden[:, 1:, :].mean(dim=1)


def _gsd_faithful_hook(module, inputs, output):
    """Apply eq. 9-10 to this layer's tokens, using the basis and the suppression module stashed on
    the layer. Reading both off `module` (not a captured self) keeps it correct under DataParallel."""
    V = getattr(module, "_gsd_V", None)
    # _gsd_sup is a ONE-ELEMENT LIST, not the module itself. Assigning an nn.Module to an attribute
    # of another nn.Module REGISTERS it as a submodule, which had two consequences: the suppression
    # parameters leaked into trainable.state_dict() (so strict loading failed), and param_groups()
    # picked them up via self.trainable.parameters() as well as via self.suppress -- putting a_l/b_l
    # in TWO optimizer groups and updating them twice per step at different learning rates.
    # Wrapping in a list keeps the per-replica stash (needed under DataParallel) without registering.
    sup_box = getattr(module, "_gsd_sup", None)
    sup = sup_box[0] if sup_box else None
    if V is None or sup is None or V.numel() == 0:
        return output
    is_tuple = isinstance(output, tuple)
    hs = output[0] if is_tuple else output                     # (B, 1+N, D)
    if getattr(module, "_gsd_suppress_cls", True):
        new_hs = sup(hs, V)                                    # paper: f[i,l] includes CLS
    else:
        cls, patches = hs[:, :1, :], hs[:, 1:, :]
        new_hs = torch.cat([cls, sup(patches, V)], dim=1)
    return ((new_hs,) + tuple(output[1:])) if is_tuple else new_hs


class _FrozenGramCollector(nn.Module):
    """Frozen CLIP -> per-layer Gram matrices M^T M for THIS shard, as a stacked (L, D, D) tensor.

    The point of returning Grams rather than a basis: a Gram is ADDITIVE, so
        M^T M  =  sum_over_shards  M_s^T M_s
    exactly. Wrapping this in DataParallel and summing the gathered per-shard Grams therefore
    reproduces the EXACT batch-128 Batch-SVD of paper eq. 12, with no approximation, while keeping
    the frozen pass spread across all GPUs.

    Without this, DataParallel gives each replica its own batch-32 basis. Measured on real CLIP
    features (runs/gsd_dp_subspace_deviation_REAL.json): at k=64 a batch-32 basis agrees with the
    batch-128 basis at only 0.836 mean principal-angle cosine, and two replicas agree with each other
    at 0.695. That is inside the range the paper validates (its ideal per-sample Single-SVD agrees at
    just 0.487) but it is NOT the batch-128 estimator the paper specifies, and the fix is cheap.
    """

    def __init__(self, frozen, layer_ids, drop_cls: bool = True):
        super().__init__()
        self.frozen, self.layer_ids, self.drop_cls = frozen, list(layer_ids), drop_cls

    @torch.no_grad()
    def forward(self, x):
        hs = self.frozen(x, output_hidden_states=True).hidden_states
        gs = []
        dev = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=dev, enabled=False):    # never let autocast touch the Gram
            for i in self.layer_ids:
                P = hs[i + 1][:, 1:, :] if self.drop_cls else hs[i + 1]
                B, N, D = P.shape
                M = P.reshape(B * N, D).float()
                gs.append(M.t() @ M)
        return torch.stack(gs, dim=0)                            # (L, D, D) fp32


class _PersistentGramFleet:
    """One frozen CLIP copy PER DEVICE, built once, plus targeted forward hooks on only the injected
    layers. Replaces nn.DataParallel(_FrozenGramCollector) in prepare_step().

    WHY. The first version wrapped the collector in nn.DataParallel, which calls replicate() on EVERY
    forward: it rebuilds several hundred module objects in Python and re-broadcasts ~1.2 GB of frozen
    CLIP weights to each device, twice per step once the main model's own DataParallel is counted. The
    measured result was 14.2 img/s against the deployed arm's 267.3 -- 19x slower, with all four GPUs
    at 0-7% and one CPU core pinned -- and it got WORSE over the run (4.8 -> 9.0 s/step), consistent
    with growing object churn. Two hypotheses were tested and REJECTED first: transformers hook
    leakage (hooks stabilise at 48 and per-call time is flat) and Python hook cost
    (output_hidden_states=True vs 4 targeted hooks is only 1.13x, and the Grams are bit-identical).

    Also drops output_hidden_states=True in favour of capturing only the 4 layers actually used, which
    is exact -- measured maxdiff 0.000e+00 against the hidden_states path -- and avoids materialising
    all 25 hidden-state tensors.
    """

    def __init__(self, frozen, layer_ids, drop_cls, gpu_ids):
        import copy
        self.layer_ids = list(layer_ids)
        self.drop_cls = bool(drop_cls)
        self.gpu_ids = list(gpu_ids)
        self.cap = {}                     # (device, layer_idx) -> tokens; keyed by device so the
        self.reps = {}                    # per-device threads never write the same key
        for g in self.gpu_ids:
            r = copy.deepcopy(frozen).to(f"cuda:{g}").eval()
            for prm in r.parameters():
                prm.requires_grad_(False)
            for i in self.layer_ids:
                r.encoder.layers[i].register_forward_hook(self._mk(i))
            self.reps[g] = r

    def _mk(self, i):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            self.cap[(t.device, i)] = t
        return hook

    def _shard_gram(self, g, xs):
        dev = f"cuda:{g}"
        self.reps[g](xs)
        out = []
        with torch.autocast(device_type="cuda", enabled=False):
            for i in self.layer_ids:
                P = self.cap.pop((torch.device(dev), i))
                P = P[:, 1:, :] if self.drop_cls else P
                B, N, D = P.shape
                M = P.reshape(B * N, D).float()
                out.append(M.t() @ M)
        return torch.stack(out, dim=0)

    @torch.no_grad()
    def batch_gram(self, x):
        """Exact batch-wide per-layer Gram, summed over device shards. Grams add, so this equals the
        single-pass whole-batch Gram (verified to 2.3e-07 relative in
        EVAL_SPACE/runs/gsd_dp_subspace_deviation_REAL.json)."""
        import threading
        n = x.shape[0]
        if len(self.gpu_ids) == 1 or n < len(self.gpu_ids):
            return self._shard_gram(self.gpu_ids[0], x.to(f"cuda:{self.gpu_ids[0]}"))
        chunks = [c for c in torch.chunk(x, len(self.gpu_ids), dim=0) if c.shape[0] > 0]
        res, errs = [None] * len(chunks), [None] * len(chunks)

        def run(j, g, xs):
            try:
                # BLOCKING copy, and pinned to this device for the whole shard. The first version used
                # non_blocking=True: a cuda:0 -> cuda:g copy launched from a worker THREAD is not
                # ordered against device g's default stream, so the frozen forward could read memory
                # the copy had not finished writing. It survived a single-shot idle test (the copy won
                # the race every time) and then corrupted the bases under real multi-threaded load --
                # reproducibly, across two runs, with reference-protocol deepfake recall pinned at
                # exactly 0.000 while the same weights scored 0.9846 under the batch protocol (which
                # goes through _compute_bases and never touches this class).
                with torch.cuda.device(g):
                    xg = xs.to(f"cuda:{g}")          # blocking: ordered before anything below
                    torch.cuda.current_stream().synchronize()
                    res[j] = self._shard_gram(g, xg)
            except BaseException as exc:      # a silent thread failure would mean a partial Gram
                errs[j] = exc

        ts = [threading.Thread(target=run, args=(j, self.gpu_ids[j], chunks[j]))
              for j in range(len(chunks))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for e in errs:
            if e is not None:
                raise RuntimeError(f"frozen Gram shard failed: {type(e).__name__}: {e}") from e
        prim = f"cuda:{self.gpu_ids[0]}"
        return torch.stack([r.to(prim) for r in res], dim=0).sum(dim=0)


class GSDFaithfulDetector(nn.Module):
    def __init__(self, cfg: GSDFaithfulConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.frozen = CLIPVisionModel.from_pretrained(cfg.clip_path)
        self.trainable = CLIPVisionModel.from_pretrained(cfg.clip_path)
        self.dim = self.frozen.config.hidden_size
        self.n_layers = self.frozen.config.num_hidden_layers
        self.gsd_layer_ids = list(range(self.n_layers - cfg.n_gsd_layers, self.n_layers))

        self.frozen.eval()
        for p in self.frozen.parameters():
            p.requires_grad = False
        self._set_trainable_scope()

        # ONE suppression module per injected layer -> a_l, b_l are genuinely layer-specific
        self.suppress = nn.ModuleDict({
            str(idx): AdaptiveSuppression(cfg.a_init, cfg.b_init, cfg.lambda_max)
            for idx in self.gsd_layer_ids})

        self.head = nn.Sequential(nn.LayerNorm(self.dim), nn.Dropout(0.1),
                                  nn.Linear(self.dim, cfg.num_classes))
        self._ref_bases: Optional[dict] = None      # {layer_idx: (D, K)} reference-based inference
        self._step_bases: Optional[dict] = None     # exact global bases for the current train step
        self._collector = None                      # lazily-built DataParallel frozen Gram collector
        self._fleet = None                          # persistent per-device frozen Gram replicas
        self._collector_ids: Optional[list] = None
        self.needs_prepare_step = True              # engine.py calls prepare_step() before forward
        self._register_hooks()

    # ------------------------------------------------------------------ setup
    def _set_trainable_scope(self) -> None:
        scope = self.cfg.trainable
        for p in self.trainable.parameters():
            p.requires_grad = (scope == "full")
        if scope == "lastN":
            for p in self.trainable.encoder.layers[-self.cfg.n_gsd_layers:].parameters():
                p.requires_grad = True
            for p in self.trainable.post_layernorm.parameters():
                p.requires_grad = True

    def _register_hooks(self) -> None:
        layers = self.trainable.encoder.layers
        for idx in self.gsd_layer_ids:
            layers[idx]._gsd_V = None
            layers[idx]._gsd_sup = None        # set per forward to [module]; see _gsd_faithful_hook
            layers[idx]._gsd_suppress_cls = self.cfg.suppress_cls
            layers[idx].register_forward_hook(_gsd_faithful_hook)

    # ------------------------------------------------------------------ bases
    @torch.no_grad()
    def _compute_bases(self, pixel_values: torch.Tensor) -> dict:
        """eq. 12 per injected layer, from the FROZEN branch's hidden states at THAT layer."""
        out = self.frozen(pixel_values, output_hidden_states=True)
        hs = out.hidden_states                        # len L+1; hs[l+1] == output of layer l
        drop_cls = self.cfg.basis_from_patch_tokens
        return {idx: batch_svd_basis(hs[idx + 1], self.cfg.k, drop_cls=drop_cls)
                for idx in self.gsd_layer_ids}

    @torch.no_grad()
    def _compute_bases_per_sample(self, pixel_values: torch.Tensor) -> list:
        """Single-SVD (Sec. 4.2): one basis set per image, so a score cannot depend on its batch."""
        out = self.frozen(pixel_values, output_hidden_states=True)
        hs = out.hidden_states
        drop_cls = self.cfg.basis_from_patch_tokens
        return [{idx: single_svd_basis(hs[idx + 1][b], self.cfg.k, drop_cls=drop_cls)
                 for idx in self.gsd_layer_ids} for b in range(pixel_values.shape[0])]

    @torch.no_grad()
    def build_reference_bases(self, pixel_values: torch.Tensor) -> dict:
        """Reference-based inference (Sec. 4.3): a reusable per-layer basis from a reference set."""
        self._ref_bases = self._compute_bases(pixel_values)
        return self._ref_bases

    @torch.no_grad()
    def set_reference_bases(self, bases: dict) -> None:
        self._ref_bases = {int(k): v for k, v in bases.items()}

    def _stash(self, bases: Optional[dict]) -> None:
        layers = self.trainable.encoder.layers
        for idx in self.gsd_layer_ids:
            layers[idx]._gsd_V = None if not bases else bases.get(idx)
            layers[idx]._gsd_sup = [self.suppress[str(idx)]]      # list => not registered as a submodule

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def prepare_step(self, pixel_values: torch.Tensor, gpu_ids: Optional[list] = None) -> None:
        """Compute the EXACT batch-wide per-layer bases for this step and stash them.

        Called from the training loop BEFORE the (possibly DataParallel) forward, so the basis is
        estimated over the WHOLE batch rather than per replica. The frozen pass is itself spread over
        the GPUs, and the per-shard Grams are summed -- which is exact, because Grams add.
        """
        # DIAGNOSTIC SWITCH, default unchanged. GSD_GRAM_IMPL=single forces the whole-batch
        # SINGLE-DEVICE collector, which is ground-truth by construction, while leaving the outer
        # DataParallel and everything else identical. It exists to locate a reproducible training
        # difference between two implementations whose Grams agree to 4e-07 (see
        # EVAL_SPACE/runs/faithful_fleet_unexplained.json). Slower by design; diagnostic only.
        import os as _os
        if _os.environ.get("GSD_GRAM_IMPL", "").lower() == "single":
            if self._collector is None:
                self._collector = _FrozenGramCollector(self.frozen, self.gsd_layer_ids,
                                                       self.cfg.basis_from_patch_tokens)
            if not getattr(self, "_warned_single", False):
                print("[gsd] GSD_GRAM_IMPL=single -- whole-batch Gram on the PRIMARY GPU only "
                      "(diagnostic; slower)", flush=True)
                self._warned_single = True
            G = self._collector(pixel_values)
        elif _os.environ.get("GSD_GRAM_IMPL", "").lower() == "fleet" and gpu_ids and len(gpu_ids) > 1:
            # ---- OPT-IN ONLY. THIS PATH IS KNOWN BROKEN. ------------------------------------------
            # _PersistentGramFleet is 14x faster and its Gram matches the single-device ground truth to
            # 4e-07 on 1, 2 and 4 GPUs, idle and under contention, blocking and async copies, train and
            # eval mode. It nevertheless trains a DIFFERENT AND MUCH WORSE model, reproducibly across
            # three runs: val bin_auc 0.8196 / 0.8254 / 0.8237 at gstep 2000 with deepfake recall
            # EXACTLY 0.000 in all nine evaluations, against 0.9880 / deepfake 0.839 for the
            # DataParallel collector below and 0.9888 / 0.860 for the single-device path. Located by a
            # pre-registered test (EVAL_SPACE/runs/gram_impl_discriminating_test.json); root cause
            # unknown after eleven falsified hypotheses. Numerical equivalence of the Gram is evidently
            # NOT sufficient for training equivalence here. Do not make this the default again without
            # a training-level A/B, not a Gram comparison.
            if self._fleet is None or self._fleet.gpu_ids != list(gpu_ids):
                self._fleet = _PersistentGramFleet(self.frozen, self.gsd_layer_ids,
                                                   self.cfg.basis_from_patch_tokens, list(gpu_ids))
            print("[gsd] WARNING: GSD_GRAM_IMPL=fleet is a KNOWN-BROKEN diagnostic path "
                  "(see gsd/faithful.py). Training results from it are not valid.", flush=True)
            G = self._fleet.batch_gram(pixel_values)
        elif gpu_ids and len(gpu_ids) > 1:
            # DEFAULT, restored: the DataParallel collector. Slow (14.2 img/s measured) but it agrees
            # with the single-device ground-truth path at the training level, which the fleet does not.
            if self._collector is None or self._collector_ids != list(gpu_ids):
                coll = _FrozenGramCollector(self.frozen, self.gsd_layer_ids,
                                            self.cfg.basis_from_patch_tokens)
                self._collector = nn.DataParallel(coll, device_ids=list(gpu_ids))
                self._collector_ids = list(gpu_ids)
            G = self._collector(pixel_values)                   # (n_shards*L, D, D), gathered
            L = len(self.gsd_layer_ids)
            if G.shape[0] % L != 0:
                raise RuntimeError(f"gathered Gram has {G.shape[0]} rows, not a multiple of L={L}")
            G = G.view(-1, L, G.shape[-2], G.shape[-1]).sum(dim=0)    # EXACT batch-wide Gram
        else:
            # cached too: the single-GPU path rebuilt this wrapper on every step
            if self._collector is None:
                self._collector = _FrozenGramCollector(self.frozen, self.gsd_layer_ids,
                                                       self.cfg.basis_from_patch_tokens)
            G = self._collector(pixel_values)                   # (L, D, D) already whole-batch
        k = int(max(1, min(self.cfg.k, G.shape[-1])))
        bases = {}
        dev = "cuda" if pixel_values.is_cuda else "cpu"
        with torch.autocast(device_type=dev, enabled=False):
            for j, idx in enumerate(self.gsd_layer_ids):
                evals, evecs = torch.linalg.eigh(G[j].float())
                bases[idx] = evecs[:, -k:].flip(dims=(1,)).float().contiguous()
        self._step_bases = bases
        self._step_seq = int(getattr(self, "_step_seq", 0)) + 1

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        proto = self.cfg.infer_protocol
        if self.training:
            # _step_bases is PER-STEP state owned by the caller: gsd/engine.py calls prepare_step()
            # before every training forward. Two ways that contract can break, both handled here
            # rather than left to produce a plausible-looking run:
            if self._step_bases is not None:
                seq = int(getattr(self, "_step_seq", 0))
                if seq == int(getattr(self, "_used_seq", -1)):
                    raise RuntimeError(
                        f"[gsd] STALE BASIS: this training forward would reuse the semantic basis "
                        f"from step {seq}, because prepare_step() was not called again. The basis is "
                        f"estimated from THIS batch's frozen tokens (paper eq. 12); reusing the "
                        f"previous batch's is a different estimator and nothing else would show it. "
                        f"Call prepare_step(batch) before every training forward -- gsd/engine.py "
                        f"does. NOTE: under DataParallel each replica gets a COPY of __dict__, so a "
                        f"replica cannot observe the master's _used_seq and this check cannot fire; "
                        f"there the per-iteration prepare_step() in the engine is the only guarantee.")
                self._used_seq = seq
                bases = {k_: v.to(pixel_values.device) for k_, v in self._step_bases.items()}
            elif self.needs_prepare_step:
                raise RuntimeError(
                    "[gsd] prepare_step() was never called, so there is no batch-wide semantic "
                    "basis for this step. Falling back to a per-replica estimate would silently "
                    "train a DIFFERENT estimator (batch/n_gpus tokens instead of the whole batch, "
                    "measured basis cosine 0.84-0.92 rather than 1.0000). Call "
                    "prepare_step(batch, gpu_ids=...) before each training forward, or set "
                    "needs_prepare_step=False to accept the per-replica estimator deliberately.")
            else:
                if not getattr(self, "_warned_no_prepare", False):
                    print("[gsd] NOTE: needs_prepare_step=False, so the semantic basis is "
                          "estimated PER REPLICA (batch/n_gpus), not over the whole batch. That is "
                          "not the paper's batch-level estimator.", flush=True)
                    self._warned_no_prepare = True
                bases = self._compute_bases(pixel_values) if pixel_values.shape[0] >= 1 else None
        elif proto == "reference" and self._ref_bases is not None:
            bases = {k: v.to(pixel_values.device) for k, v in self._ref_bases.items()}
        elif proto == "per_sample":
            per = self._compute_bases_per_sample(pixel_values)
            outs = []
            for b in range(pixel_values.shape[0]):
                self._stash(per[b])
                o = self.trainable(pixel_values[b:b + 1])
                outs.append(self.head(_pool(o.last_hidden_state, self.cfg.head_pool)))
            return torch.cat(outs, dim=0)
        else:
            bases = self._compute_bases(pixel_values)
        self._stash(bases)
        out = self.trainable(pixel_values)
        return self.head(_pool(out.last_hidden_state, self.cfg.head_pool))

    # ------------------------------------------------------------------ optim
    def param_groups(self):
        head = list(self.head.parameters())
        lam = list(self.suppress.parameters())          # a_l, b_l -- follow the head's lr, not 1e-6
        bb = [p for p in self.trainable.parameters() if p.requires_grad]
        # GUARD: a parameter in two groups is updated twice per step with two different lrs. This
        # happened for real when the suppression modules were stashed onto the CLIP layers directly.
        _lam_ids = {id(q) for q in lam}
        overlap = [q for q in bb if id(q) in _lam_ids]
        if overlap:
            raise RuntimeError(
                f"{len(overlap)} suppression parameter(s) are also reachable through "
                f"self.trainable -- they would be optimized twice per step. The per-layer stash must "
                f"not register them as submodules (see _gsd_faithful_hook).")
        groups = [{"params": head + lam, "lr": self.cfg.head_lr}]
        if bb:
            groups.append({"params": bb, "lr": self.cfg.lr})
        return groups

    def lambda_report(self) -> dict:
        """Per-layer mean lambda and occupancy from the last forward -- diagnostics for the log."""
        return {int(k): {"lambda_mean": float(m._last_lambda_mean),
                         "r_mean": float(m._last_r_mean),
                         "a": float(m.a.detach()), "b": float(m.b.detach())}
                for k, m in self.suppress.items()}
