#!/usr/bin/env bash
# Emit a compact status block for the active run_finetuning.sh run, once per interval.
# Used as a Monitor command so each block arrives as one notification.
#   bash status_report.sh [interval_seconds]      (default 1800 = 30 min)
cd "$(dirname "$0")" || exit 1
INT="${1:-1800}"
while true; do
  RUN=$(cat .last_run 2>/dev/null)
  [ -z "$RUN" ] && { echo "[status] no .last_run"; sleep "$INT"; continue; }
  DRV="${RUN}_driver.log"
  # current stage = last ">>> name" line without a matching "<<< name"
  STAGE=$(grep -oE ">>> [a-z_0-9]+" "$DRV" 2>/dev/null | tail -1 | awk '{print $2}')
  DONE=$(grep -cE "<<< .* OK" "$DRV" 2>/dev/null)
  FAIL=$(grep -cE "<<< .* FAILED" "$DRV" 2>/dev/null)
  ALIVE=$(pgrep -fc "run_finetuning.sh" 2>/dev/null || echo 0)
  echo "── $(date '+%m-%d %H:%M')  stage=${STAGE:-?}  done=$DONE failed=$FAIL  driver_alive=$ALIVE"
  # elapsed since the stage started
  if [ -n "$STAGE" ] && [ -f "$RUN/$STAGE.log" ]; then
    SZ=$(stat -c %s "$RUN/$STAGE.log" 2>/dev/null)
    AGE=$(( $(date +%s) - $(stat -c %Y "$RUN/$STAGE.log" 2>/dev/null || date +%s) ))
    echo "   log=$RUN/$STAGE.log ${SZ}B  last_write=${AGE}s ago"
    # last meaningful progress line (tqdm writes with \r -- take the final segment)
    tr '\r' '\n' < "$RUN/$STAGE.log" 2>/dev/null | grep -vE "^\s*$" | tail -2 | sed 's/^/   /' | cut -c1-190
  fi
  # GPUs 1,2,3 only (GPU 0 is the user's app server)
  echo "   gpu $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader \
        | awk -F, '$1!=0{printf "  %s:%s/%s", $1, $2, $3}')"
  sleep "$INT"
done
