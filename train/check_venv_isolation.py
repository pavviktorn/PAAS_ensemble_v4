#!/usr/bin/env python3
"""Prove the venv is SELF-CONTAINED before a multi-day run.

Two independent checks:
  1. sys.path carries no site-packages outside the venv (no ~/.local, no /usr/local).
  2. every third-party module the project imports actually RESOLVES under the venv prefix,
     and the accuracy-critical ones are at the exact validated versions.

Why this matters here: ~/.local holds a complete legacy stack -- transformers 4.37.2, peft 0.7.1,
tokenizers 0.15.2, accelerate 0.21.0 -- which was inert only because the venv sorted earlier on
sys.path. Loading it would not crash loudly; it would train a head that cannot be served.

  venv/bin/python train/check_venv_isolation.py        # exits 1 on any violation
"""
import importlib
import sys

VENV = "/datasets/work/vLLM/temp/PAAS_qwen3vl/venv"

# every third-party module imported anywhere in the project (AST-scanned, plus the runtime-only
# ones that import-scanning cannot see: sentencepiece via T5Tokenizer, multipart via FastAPI)
MODULES = [
    "torch", "torchvision", "transformers", "vllm", "peft", "deepspeed", "accelerate",
    "numpy", "cv2", "PIL", "albumentations", "sklearn", "scipy", "yaml", "tqdm",
    "safetensors", "tokenizers", "huggingface_hub", "sentencepiece", "einops", "fla",
    "onnxruntime", "fastapi", "uvicorn", "pydantic", "starlette", "werkzeug", "multipart",
    # flash_attn: train_qwen_mllm.py passes attn_implementation="flash_attention_2" and
    # transformers hard-raises when it is missing (no sdpa fallback). It was absent from the first
    # isolated venv and killed Step 2a ten seconds in, AFTER 10h of detector training -- exactly
    # the kind of late failure this probe exists to move to the front of the run.
    "flash_attn",
]

# versions the pipeline was validated against (Exp 17-21). A silent drift here is an accuracy risk,
# not just a packaging one.
PINS = {
    "torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.13.1", "vllm": "0.21.0",
    "numpy": "1.26.4", "peft": "0.19.1", "deepspeed": "0.14.5", "accelerate": "1.14.0",
    "tokenizers": "0.22.2", "safetensors": "0.8.0", "albumentations": "2.0.8",
    "opencv-python-headless": "4.13.0.92", "sentencepiece": "0.2.1", "scikit-learn": "1.8.0",
}


def main() -> int:
    bad = 0

    print("== 1. sys.path ==")
    stray = [p for p in sys.path
             if ("site-packages" in p or "dist-packages" in p) and not p.startswith(VENV)]
    for p in stray:
        print(f"  STRAY {p}")
    if stray:
        bad += len(stray)
    else:
        print("  ok  only venv site-packages on sys.path")

    print("== 2. module resolution ==")
    for m in MODULES:
        try:
            mod = importlib.import_module(m)
        except Exception as e:                       # noqa: BLE001 - report, do not abort
            print(f"  FAIL {m:20} import error: {type(e).__name__}: {e}")
            bad += 1
            continue
        f = getattr(mod, "__file__", "") or ""
        if not f.startswith(VENV):
            print(f"  FAIL {m:20} resolves OUTSIDE the venv: {f}")
            bad += 1
    if not bad:
        print(f"  ok  all {len(MODULES)} modules resolve under {VENV}")

    # A bare `import X` is not proof the package works: vllm loads its submodules lazily, so
    # `import vllm` succeeded while `from vllm import LLM` died on a missing pycountry (a runtime
    # import that no package metadata declares). These are the real entry points the pipeline uses.
    print("== 2b. deep imports (lazy submodules the bare import does not touch) ==")
    DEEP = [
        ("vllm", "from vllm import LLM, SamplingParams"),
        ("transformers", "from transformers import CLIPVisionModel, T5Tokenizer, AutoProcessor"),
        ("peft", "from peft import LoraConfig, get_peft_model"),
        ("torchvision", "from torchvision import transforms"),
        ("albumentations", "import albumentations as A; A.ImageCompression"),
        ("sklearn", "from sklearn.metrics import roc_auc_score, average_precision_score"),
        # the exact predicate transformers uses to decide whether FA2 can be enabled -- a bare
        # `import flash_attn` can succeed while this still returns False (e.g. a build compiled
        # against a different torch), which is what the MLLM trainer actually depends on.
        ("flash_attn(FA2)",
         "from transformers.utils import is_flash_attn_2_available as f; assert f(), 'FA2 unavailable'"),
    ]
    for name, stmt in DEEP:
        try:
            exec(compile(stmt, "<deep>", "exec"), {})
        except Exception as e:                       # noqa: BLE001
            print(f"  FAIL {name:20} {stmt!r} -> {type(e).__name__}: {e}")
            bad += 1
    if not bad:
        print(f"  ok  all {len(DEEP)} deep imports work")

    print("== 3. validated versions ==")
    import importlib.metadata as md
    for name, want in sorted(PINS.items()):
        try:
            got = md.version(name)
        except Exception:
            got = "MISSING"
        if got.split("+")[0] != want:
            print(f"  FAIL {name:26} want {want:12} got {got}")
            bad += 1
    print(f"  {'ok  all pins match' if bad == 0 else 'see failures above'}")

    print(f"\n{'ISOLATION OK' if bad == 0 else f'ISOLATION VIOLATIONS: {bad}'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
