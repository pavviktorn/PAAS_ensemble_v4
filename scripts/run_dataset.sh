#!/usr/bin/env bash
# Multi-GPU wrapper around scripts/run_dataset.py.
#
# Scoring is the slowest stage of the pipeline: run_dataset.py loads the whole ensemble (vLLM MLLM
# + 4 detector heads) on ONE GPU and reached ~44 frames/min, so a full testset pass took many hours
# while the other three GPUs idled. This shards the MEDIA FILES across GPUs (every N-th labelled
# file, never splitting a video mid-file) and merges the per-shard result files at the end.
#
#   bash scripts/run_dataset.sh --config <cfg> --input-dir <dir> --out-dir <out> [--frame-stride 10]
#   GPUS=0,1,2,3 bash scripts/run_dataset.sh ...          # default: all four
#   GPUS=1,2,3   bash scripts/run_dataset.sh ...          # leave GPU 0 for the app server
#
# Any extra flags are passed straight through to run_dataset.py.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin
PY="$VENV/python"
export PYTHONNOUSERSITE=1          # the venv is self-contained; never fall through to ~/.local

GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -ra G <<< "$GPUS"; N="${#G[@]}"

# pull --out-dir out of the args (we need it to place shard dirs + merge), pass the rest through
OUT=""; ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out-dir) OUT="$2"; shift 2 ;;
    --out-dir=*) OUT="${1#*=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ -z "$OUT" ] && { echo "usage: $0 --config C --input-dir D --out-dir O [...]"; exit 2; }
mkdir -p "$OUT"

echo "[run_dataset] $N shard(s) on GPU(s) $GPUS -> $OUT"
pids=()
for i in "${!G[@]}"; do
  sd="$OUT/shard_$i"; mkdir -p "$sd"
  CUDA_VISIBLE_DEVICES="${G[$i]}" "$PY" scripts/run_dataset.py \
      "${ARGS[@]}" --out-dir "$sd" --which-part "$i" --n-divided "$N" \
      > "$OUT/shard_$i.log" 2>&1 &
  pids+=($!)
  echo "  shard $i -> gpu ${G[$i]} (pid ${pids[$i]}, log $OUT/shard_$i.log)"
done

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "  shard $i FAILED (rc=$?)"; tail -5 "$OUT/shard_$i.log"; rc=1; }
done

# ---- merge: concatenate the shard files, keeping ONE header ----
# Each shard writes the same three result files; the fused one (results_paas.txt) is what
# train/fit_threshold.py reads.
for name in results_paas.txt results_ffaa.txt results_ensemble.txt; do
  first=1; out="$OUT/$name"; : > "$out"
  for i in "${!G[@]}"; do
    f="$OUT/shard_$i/$name"; [ -f "$f" ] || continue
    if [ "$first" = 1 ]; then cat "$f" >> "$out"; first=0
    else grep -v '^#' "$f" >> "$out"; fi
  done
  [ -s "$out" ] && echo "  merged $name: $(grep -vc '^#' "$out") rows"
done

# report the label mix -- a threshold fitted on one class only is invalid (fit_threshold refuses,
# but say so here too, because that is exactly how a calibration run silently wastes hours)
if [ -s "$OUT/results_paas.txt" ]; then
  echo "  label mix:"; grep -vE '^#' "$OUT/results_paas.txt" | awk '{print $2}' | sort | uniq -c | sed 's/^/    /'
fi
exit $rc
