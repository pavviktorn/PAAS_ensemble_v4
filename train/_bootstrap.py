"""Path bootstrap for the training drivers: makes the v4 vendored detector libs importable
(gsd/selop at root, mids/utils under ffaa/, mids9lib under ensemble9/) when a driver is run
directly, e.g. `venv/bin/python train/train_gsd.py ...`. Import this FIRST in every driver."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in (ROOT, os.path.join(ROOT, "ffaa"), os.path.join(ROOT, "ensemble9")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

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
# The venv must be SELF-CONTAINED: no ~/.local, no /usr/local. This is not cosmetic --
# ~/.local carries a complete legacy stack (transformers 4.37.2, peft 0.7.1, tokenizers
# 0.15.2, accelerate 0.21.0) that is inert only because the venv sorts earlier on sys.path.
# Anything that perturbs that order (removing a venv package, a stray PYTHONPATH, a venv
# rebuilt with --system-site-packages) would silently load tf4 and train a model that
# cannot be served. Enforced by pyvenv.cfg include-system-site-packages=false plus
# PYTHONNOUSERSITE=1; this check fails fast if either is ever undone.
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
