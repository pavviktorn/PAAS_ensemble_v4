"""Environment + path setup for PAAS_ensemble_v2.

Everything in this project is designed to run on the GLOBAL interpreter
the ONE project venv (transformers>=5 + vLLM): /datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
Import this module (and call :func:`setup`) before importing any FFAA or 9-class code so that
(a) the vendored ``ffaa/`` and ``ensemble9/`` trees are importable, (b) the CUDA device is
chosen *before* the FFAA modules pin ``CUDA_VISIBLE_DEVICES``, and (c) HF stays offline/quiet.
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFAA_DIR = os.path.join(PROJECT_ROOT, "ffaa")
ENSEMBLE9_DIR = os.path.join(PROJECT_ROOT, "ensemble9")

# Bundled assets (self-contained; see README "Layout").
BASE_CLIP = os.path.join(PROJECT_ROOT, "base_models", "clip-vit-large-patch14-336")
BASE_T5 = os.path.join(PROJECT_ROOT, "base_models", "t5-base")
# v4 FFAA = Qwen3.5-4B (merged, vLLM) + FROM-SCRATCH MIDS 4-class head (not axon0 warm-start).
QWEN_DIR = os.path.join(PROJECT_ROOT, "weights", "qwen35_4b_merged")
MIDS_PATH = os.path.join(PROJECT_ROOT, "weights", "ffaa_qwen35_mids", "best.pth")
PROMPT_FILE = os.path.join(PROJECT_ROOT, "ffaa", "playground", "prompts.txt")
ENSEMBLE9_CONFIG = os.path.join(PROJECT_ROOT, "config", "ensemble9.json")

# GSD (Exp 10) and SeLop/LROR (Exp 11) - the two CLIP detectors added in v3. Both reuse the shared
# BASE_CLIP backbone above (the CLIP vision weights are identical across all members).
GSD_CKPT = os.path.join(PROJECT_ROOT, "weights", "gsd", "best_lastN_ep0_auc0.9930.pt")
SELOP_CKPT = os.path.join(PROJECT_ROOT, "weights", "selop", "best.pt")


def setup(device: str | None = None, quiet: bool = True) -> None:
    """Make the vendored trees importable and pin the CUDA device.

    ``device`` like ``"cuda:0"`` / ``"cuda:2"`` / ``"cpu"``. When a cuda index is given we set
    ``CUDA_VISIBLE_DEVICES`` to that physical index and the process then sees it as ``cuda:0`` --
    this is what lets FFAA's ``models.py`` (which calls ``setdefault('CUDA_VISIBLE_DEVICES','0')``)
    land on the device we want, and is the basis for the multi-GPU file-sharding scripts.
    """
    if device and device.startswith("cuda:"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", device.split(":", 1)[1])
    if quiet:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for d in (FFAA_DIR, ENSEMBLE9_DIR, PROJECT_ROOT):   # PROJECT_ROOT -> `import gsd` / `import selop`
        if d not in sys.path:
            sys.path.insert(0, d)


def visible_device() -> str:
    """The device string to hand torch *after* CUDA_VISIBLE_DEVICES has been pinned (always cuda:0
    when a single physical GPU was selected)."""
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# ---- ONE-VENV GUARD ----------------------------------------------------------------
# Every stage (training AND inference) runs on the single project venv:
#   /datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python
# The tf5 CLIP ports (flattened .vision_model, tensor-returning CLIPEncoderLayer) and the in-process
# vLLM both require transformers>=5, so silently running on the legacy transformers==4.37 interpreter
# would either crash deep in a model load or, worse, train something that cannot be served.
def _require_project_venv():
    import sys
    try:
        import transformers
        major = int(transformers.__version__.split(".")[0])
    except Exception as e:                      # transformers missing entirely -> wrong interpreter
        raise SystemExit(f"[venv] cannot import transformers ({e}).\n"
                         f"[venv] run everything with: /datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python")
    if major < 5:
        raise SystemExit(
            f"[venv] transformers {transformers.__version__} at {sys.executable} -- this project needs >=5.\n"
            f"[venv] run everything (train AND inference) with: /datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python")


# ---- ISOLATION GUARD ---------------------------------------------------------------
# The venv must be SELF-CONTAINED: no ~/.local, no /usr/local. ~/.local carries a complete
# legacy stack (transformers 4.37.2, peft 0.7.1, tokenizers 0.15.2, accelerate 0.21.0) that
# is inert only because the venv sorts earlier on sys.path -- so an isolation slip would
# silently serve on tf4. Enforced by pyvenv.cfg include-system-site-packages=false plus
# PYTHONNOUSERSITE=1. Mirrors train/_bootstrap.py so BOTH deploy and train paths check.
_VENV_PREFIX = "/datasets/work/vLLM/temp/PAAS_qwen3vl/venv"


def _require_isolated_site():
    import sys
    stray = [p for p in sys.path
             if ("site-packages" in p or "dist-packages" in p) and not p.startswith(_VENV_PREFIX)]
    if stray:
        raise SystemExit(
            "[venv] NON-VENV package paths on sys.path -- the environment is not isolated:\n"
            + "".join(f"        {p}\n" for p in stray)
            + f"[venv] expected only {_VENV_PREFIX}/lib*/python3.12/site-packages.\n"
              "[venv] fix: PYTHONNOUSERSITE=1 and include-system-site-packages=false in pyvenv.cfg.")


_require_project_venv()
_require_isolated_site()
