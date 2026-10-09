#!/usr/bin/env bash
# stop_server.sh -- stop ONLY the process started by run_server.sh and free its GPU memory.
# run_server.sh is now SINGLE-process (one venv): the FastAPI ensemble app (uvicorn) with the
# Qwen3.5-4B vLLM built IN-PROCESS, which spawns a vLLM EngineCore subprocess.
# Does NOT touch any training / data-gen / scoring jobs (use stop_all.sh for those).
#
#   bash stop_server.sh        # graceful (SIGTERM) then force (SIGKILL) leftovers
#   bash stop_server.sh -9     # straight to SIGKILL
set -uo pipefail

FORCE=0; [ "${1:-}" = "-9" ] && FORCE=1

# only run_server.sh's processes (bracketed first char => pkill never matches itself)
PATTERNS=(
  "[u]vicorn app_fastapi_json_v5_safe_jsonfmt"   # FastAPI ensemble app (single-process launcher)
  "[E]ngineCore"                                 # in-process vLLM engine-core subprocess
)

echo "[stop_server] GPU before:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

matched=0
for p in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$p" 2>/dev/null || true)
    [ -z "$pids" ] && continue
    matched=1
    echo "[stop_server] $p -> $(echo "$pids" | tr '\n' ' ')"
    [ "$FORCE" = "1" ] && kill -9 $pids 2>/dev/null || kill $pids 2>/dev/null || true
done

if [ "$FORCE" != "1" ] && [ "$matched" = "1" ]; then
    echo "[stop_server] waiting up to 15s for graceful exit ..."
    for _ in $(seq 1 15); do
        still=0; for p in "${PATTERNS[@]}"; do pgrep -f "$p" >/dev/null 2>&1 && still=1; done
        [ "$still" = "0" ] && break; sleep 1
    done
    for p in "${PATTERNS[@]}"; do
        pids=$(pgrep -f "$p" 2>/dev/null || true)
        [ -n "$pids" ] && { echo "[stop_server] force-kill $p -> $pids"; kill -9 $pids 2>/dev/null || true; }
    done
fi

sleep 3
echo "[stop_server] remaining:"
left=0; for p in "${PATTERNS[@]}"; do pgrep -f "$p" >/dev/null 2>&1 && { echo "  STILL RUNNING: $p"; left=1; }; done
[ "$left" = "0" ] && echo "  (none)"
echo "[stop_server] GPU after:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
