#!/usr/bin/env bash
# stop_all.sh -- stop ALL PAAS / FFAA / vLLM / training processes and free the GPUs.
#
#   bash stop_all.sh          # graceful (SIGTERM) then force (SIGKILL) the leftovers
#   bash stop_all.sh -9       # go straight to SIGKILL
#
# Safe to run anytime: it only targets this project's process patterns (uses the [x] bracket trick
# so the pkill command never matches itself), and reports GPU memory before/after.
set -uo pipefail

FORCE=0; [ "${1:-}" = "-9" ] && FORCE=1

# process patterns to stop (bracketed first char => never self-matches)
PATTERNS=(
  "[v]llm.entrypoints.openai"        # vLLM OpenAI server
  "[E]ngineCore"                     # vLLM engine-core subprocess
  "VLLM::[E]ngineCore"
  "[u]vicorn app_fastapi"            # the FastAPI ensemble app
  "app_fastapi_json_v5_safe_jsonfmt" # app module
  "[t]rain_qwen_lora.py"             # MLLM LoRA finetune / distillation
  "[t]rain_mids_new.py"              # MIDS head training (deepspeed)
  "[d]eepspeed.*train_mids"
  "[g]en_mids_vllm.py"               # Step-2 MIDS data generation
  "[g]en_score_axon1.py"             # axon1 gen/score
  "[s]core_frames_mids.py"           # MIDS scoring
  "[m]ulti_head_rescore.py"
  "[b]ench_speed.py"
  "[r]un_finetuning.sh"              # pipeline driver
  "[s]moke_v4.py"
)

echo "[stop_all] GPU before:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

matched=0
for p in "${PATTERNS[@]}"; do
    pids=$(pgrep -f "$p" 2>/dev/null || true)
    [ -z "$pids" ] && continue
    matched=1
    echo "[stop_all] $p -> $(echo "$pids" | tr '\n' ' ')"
    if [ "$FORCE" = "1" ]; then
        kill -9 $pids 2>/dev/null || true
    else
        kill $pids 2>/dev/null || true      # SIGTERM (let vLLM shut down cleanly)
    fi
done

if [ "$FORCE" != "1" ] && [ "$matched" = "1" ]; then
    echo "[stop_all] waiting up to 15s for graceful exit ..."
    for _ in $(seq 1 15); do
        still=0
        for p in "${PATTERNS[@]}"; do pgrep -f "$p" >/dev/null 2>&1 && still=1; done
        [ "$still" = "0" ] && break
        sleep 1
    done
    # force-kill any survivors
    for p in "${PATTERNS[@]}"; do
        pids=$(pgrep -f "$p" 2>/dev/null || true)
        [ -n "$pids" ] && { echo "[stop_all] force-kill $p -> $pids"; kill -9 $pids 2>/dev/null || true; }
    done
fi

sleep 3
echo "[stop_all] remaining project procs:"
left=0
for p in "${PATTERNS[@]}"; do pgrep -f "$p" >/dev/null 2>&1 && { echo "  STILL RUNNING: $p"; left=1; }; done
[ "$left" = "0" ] && echo "  (none)"
echo "[stop_all] GPU after:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
