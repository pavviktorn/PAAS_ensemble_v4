#!/usr/bin/env bash
# Launch PAAS_ensemble_v4: ONE process, ONE venv (transformers 5.13 + vLLM).
# The FastAPI ensemble app builds the Qwen3.5-4B MLLM IN-PROCESS with vLLM (no separate server, no
# HTTP model backend) and loads the MIDS/A2/GSD/SeLop CLIP stack in the SAME interpreter via the tf5
# weight remap. A single uvicorn worker owns the GPU.
#
#   bash run_server.sh                 # ensemble API on :$APP_PORT (default 3000), GPU 0
#   APP_PORT=8080 bash run_server.sh   # pick a port
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_server.sh   # deploy across GPUs (needs qwen tensor-parallel cfg)
set -euo pipefail
cd "$(dirname "$0")"

VENV_PY="${VENV_PY:-/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python}"   # tf5 + vLLM (the ONE venv)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
APP_HOST="${APP_HOST:-0.0.0.0}"; APP_PORT="${APP_PORT:-3000}"
export PAAS_CONFIG="${PAAS_CONFIG:-config/experiments/paas4_qwen.json}"

echo "[run_server] single-venv ensemble app on $APP_HOST:$APP_PORT (GPU $CUDA_VISIBLE_DEVICES) ..."
echo "[run_server] Qwen3.5-4B vLLM is built IN-PROCESS at startup -- give it ~1-2 min to warm up."
exec "$VENV_PY" -m uvicorn app_fastapi_json_v5_safe_jsonfmt:app \
    --host "$APP_HOST" --port "$APP_PORT" --workers 1
