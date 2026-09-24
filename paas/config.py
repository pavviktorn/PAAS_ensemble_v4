"""PAAS_ensemble_v3 experiment configuration.

A single ``PaasConfig`` describes ONE experiment: which detectors to run and how to fuse their
per-frame fake-scores into a decision. v3 fuses up to FIVE component detectors:

    ffaa   - LLaVA-Mistral-7B MLLM + MIDS       (Exp 9)
    A1_9c  - 9-class SVD ensemble member        (Exp 7-8)   } read from the 9-class ensemble's
    A2_9c  - 9-class SVD+GenD ensemble member    (Exp 7-8)   } per-model output (A3_9c also available)
    gsd    - Geometric Semantic Decoupling       (Exp 10)
    selop  - SeLop / LROR low-rank orthogonal     (Exp 11)

The recommended combination (docs/COMBINATION_FINDINGS_axon1.md, Exp 13) is the PLAIN MEAN of
{ffaa, A1_9c, A2_9c, gsd, selop} -- AUC 0.9998, fake-recall 99.97% at a 90% real-recall floor.
Which detectors load is derived from ``fusion.components``, so changing the combination is a config
change, not a code change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from . import env

# every component name the fusion understands (which model produces each is handled in the pipeline)
ENS_MEMBERS = ("A1_9c", "A2_9c", "A3_9c")
ALL_COMPONENTS = ("ffaa",) + ENS_MEMBERS + ("gsd", "selop")


@dataclass
class FFAACfg:
    # v4 FFAA = Qwen3.5-4B MLLM (IN-PROCESS vLLM, single venv/tf5) + FROM-SCRATCH MIDS 4-class head.
    # Everything runs in ONE process on the same venv: vLLM builds the LLM in-process (no server, no
    # HTTP), and the MIDS/CLIP stack loads in the same interpreter via the tf5 weight remap.
    enabled: bool = True
    qwen_dir: str = env.QWEN_DIR                      # merged Qwen3.5-4B model dir (loaded in-process)
    qwen_gpu_mem: float = 0.45                        # vLLM GPU frac; rest is left for MIDS/A2/GSD/SeLop
    qwen_max_model_len: int = 4096
    qwen_max_pixels: int = 451584                     # mm_processor longest_edge
    mids_path: str = env.MIDS_PATH
    whole_frame: bool = False        # True: letterbox the WHOLE frame for the MIDS head, matching
                                     # a head trained that way. Legacy heads used CLIP center-crop.
    clip_path: str = env.BASE_CLIP
    t5_path: str = env.BASE_T5
    prompt_file: Optional[str] = env.PROMPT_FILE     # first line = FFAA base prompt
    prompt: str = "The image is a human face image. Is it real or fake? Why?"
    max_new_tokens: int = 512
    generate_num: int = 3                            # 3-pass conditional protocol (N=1,M=1)
    cache_path: Optional[str] = None


@dataclass
class Ensemble9Cfg:
    enabled: bool = True
    config_path: str = env.ENSEMBLE9_CONFIG


@dataclass
class GSDCfg:
    enabled: bool = True
    ckpt: str = env.GSD_CKPT
    clip_path: str = env.BASE_CLIP
    amp_dtype: str = "bf16"


@dataclass
class SeLopCfg:
    enabled: bool = True
    ckpt: str = env.SELOP_CKPT
    clip_path: str = env.BASE_CLIP
    amp_dtype: str = "bf16"


@dataclass
class FusionCfg:
    # method: "mean" | "weighted"  (over the component fake-scores listed below)
    method: str = "mean"
    # v4 best combination (deployable plain-mean): axonlabs_data_1 AUC 0.9998, FR99 99.74.
    components: List[str] = field(
        default_factory=lambda: ["ffaa", "A2_9c", "gsd", "selop"])
    # for method=="weighted": one weight per component (same order); normalised internally.
    weights: Optional[List[float]] = None


@dataclass
class DecisionCfg:
    # v4 mean{ffaa(qwen),A2_9c,gsd,selop} operating thresholds on axonlabs_data_1 (613,415 frames):
    #   real>=0.90 -> tau 0.2192 (fake-recall 99.96%)   real>=0.95 -> 0.2688 (99.93%)
    #   real>=0.98 -> tau 0.3893 (fake-recall 99.82%)   real>=0.99 -> 0.4634 (99.74%)
    # Default = 90% real-recall floor (deployment-robust; matches v3's default floor choice).
    threshold: float = 0.2192
    real_ambiguous_match_min: float = 0.9  # decision==real & match<this -> "ambiguous"
    treat_likely_fake_as_ambiguous: bool = True


@dataclass
class PaasConfig:
    name: str = "paas4_qwen_mean"
    device: str = "cuda:0"
    ffaa: FFAACfg = field(default_factory=FFAACfg)
    ensemble9: Ensemble9Cfg = field(default_factory=Ensemble9Cfg)
    gsd: GSDCfg = field(default_factory=GSDCfg)
    selop: SeLopCfg = field(default_factory=SeLopCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)

    # ---- which models must load, derived from the requested components ----
    def needs(self) -> dict:
        c = set(self.fusion.components)
        return {
            "ffaa": "ffaa" in c,
            "ens": bool(c & set(ENS_MEMBERS)),
            "gsd": "gsd" in c,
            "selop": "selop" in c,
        }

    # ---- validation ----
    def validate(self) -> "PaasConfig":
        comps = self.fusion.components
        if not comps:
            raise ValueError("fusion.components is empty")
        bad = [c for c in comps if c not in ALL_COMPONENTS]
        if bad:
            raise ValueError(f"unknown fusion components {bad}; allowed: {ALL_COMPONENTS}")
        if self.fusion.method not in ("mean", "weighted"):
            raise ValueError(f"fusion.method must be 'mean' or 'weighted', got {self.fusion.method!r}")
        if self.fusion.method == "weighted":
            w = self.fusion.weights
            if not w or len(w) != len(comps):
                raise ValueError("fusion.method=='weighted' needs weights of len(components)")
        need = self.needs()
        if need["ffaa"] and not self.ffaa.enabled:
            raise ValueError("components include 'ffaa' but ffaa.enabled=False")
        if need["ens"] and not self.ensemble9.enabled:
            raise ValueError("components include a 9-class member but ensemble9.enabled=False")
        if need["gsd"] and not self.gsd.enabled:
            raise ValueError("components include 'gsd' but gsd.enabled=False")
        if need["selop"] and not self.selop.enabled:
            raise ValueError("components include 'selop' but selop.enabled=False")
        return self

    # ---- (de)serialise ----
    @classmethod
    def from_dict(cls, d: dict) -> "PaasConfig":
        d = dict(d)
        sub = {
            "ffaa": (FFAACfg, d.pop("ffaa", {})),
            "ensemble9": (Ensemble9Cfg, d.pop("ensemble9", {})),
            "gsd": (GSDCfg, d.pop("gsd", {})),
            "selop": (SeLopCfg, d.pop("selop", {})),
            "fusion": (FusionCfg, d.pop("fusion", {})),
            "decision": (DecisionCfg, d.pop("decision", {})),
        }
        kw = {k: klass(**(vals or {})) for k, (klass, vals) in sub.items()}
        top = {"name", "device"}                      # drop annotations/unknowns (e.g. "_note", "_about")
        d = {k: v for k, v in d.items() if k in top}
        return cls(**d, **kw).validate()

    @classmethod
    def from_file(cls, path: str) -> "PaasConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
