#!/usr/bin/env bash
# run_finetuning.sh -- PAAS_ensemble_v4 END-TO-END FFAA + ensemble finetuning, ONE venv (tf5 + vLLM).
#
#   STEP 0  build detector manifests from MIDS_IMAGES (a DIRECTORY, walked recursively; truth from
#           get_label_all, MAKEUP->PAD, UNKNOWN dropped). No GPU, minutes.
#             images_train.json  -> GSD / SeLop (they derive labels from the PATH)
#             mids9_{train,val}.json -> 9-class A2 set: 3 FIXED claim anchors, label = 3*true + claim
#   STEP 1  MLLM-FREE detectors (no dependency on the MLLM, so they run first and survive a gen
#           failure): 9-class A2 (SVD+GenD), GSD (anchor=testset), SeLop
#   STEP 2  Qwen3.5-4B MLLM: base LoRA (r=32/alpha=48) -> distill continuation -> merge  [RUN_QWEN]
#   GATE    functional eval of the merged MLLM (parse/verdict/comply); aborts before the big gen
#   STEP 3  generate the MIDS answer data with the 4B over MIDS_IMAGES -> $WORK/mids_qwen_*.json
#   STEP 4  MIDS 4-class head (the ONLY branch that needs the MLLM answers)
#   STEP 5  assemble a deploy dir + RE-FIT the fusion threshold (the inherited one is only valid for
#           the weights it was fitted on) [RUN_DEPLOY]
#
# ALL detectors train FROM SCRATCH. Trainset for every branch = MIDS_IMAGES only.
# All temp/result files (manifests, gen data, merge, heads, logs, deploy) live under this project.
#
#   bash run_finetuning.sh                                   # full chain, all 4 GPUs
#   RUN_QWEN=0 RUN_GEN=0 RUN_MIDS=0 bash run_finetuning.sh    # only the MLLM-free detectors
#   RUN_9C=0 RUN_GSD=0 RUN_SELOP=0 bash run_finetuning.sh     # only the MLLM chain + MIDS-4c
#   SMOKE=1 SMOKE_GPU=1 bash run_finetuning.sh                # tiny end-to-end validation
set -uo pipefail
cd "$(dirname "$0")"
PROOT="$PWD"
OUT_ROOT="${OUT_ROOT:-runs/finetune_$(date +%Y%m%d_%H%M%S)}"   # defined early: WORK/MANI default under it
VENV=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin
PY="$VENV/python"; TORCHRUN="$PY -m torch.distributed.run"
# SELF-CONTAINED VENV: never fall through to ~/.local (which still holds a full legacy stack --
# transformers 4.37.2, peft 0.7.1, tokenizers 0.15.2 -- that would silently replace the tf5 one).
# Pairs with include-system-site-packages=false in pyvenv.cfg; train/_bootstrap.py hard-fails if
# either is undone. Exported so every torchrun rank and vLLM subprocess inherits it.
export PYTHONNOUSERSITE=1

# ===================== WHICH STAGES (1=run, 0=skip) =====================
_QWEN_SET="${RUN_QWEN+x}"; _NINEC_SET="${NINEC_CFG+x}"   # did the caller choose RUN_QWEN explicitly?
RUN_MANIFEST="${RUN_MANIFEST:-1}" # Step 0: build image/9-class manifests from MIDS_IMAGES
RUN_9C="${RUN_9C:-1}"        # Step 1: 9-class A2 (SVD+GenD, 9 classes) -- MLLM-free
RUN_GSD="${RUN_GSD:-1}"      # Step 1: GSD dual-stream (DataParallel)   -- MLLM-free
GSD_FAITHFUL="${GSD_FAITHFUL:-0}"  # 1 = train the paper-faithful GSD arm (needs GSD_CFG=
                                   # train/configs/gsd_faithful.json). Default 0 keeps the deployed
                                   # detector, which is what the shipped ensemble uses.
# Normalised because it is interpolated UNQUOTED into run_config.json and gsd_provenance.json:
# GSD_FAITHFUL=yes would otherwise emit invalid JSON that nothing reads until much later.
case "$GSD_FAITHFUL" in
  1|true|yes|on) GSD_FAITHFUL=1 ;;
  0|false|no|off|"") GSD_FAITHFUL=0 ;;
  *) echo "[run_finetuning] GSD_FAITHFUL='$GSD_FAITHFUL' is not a boolean (use 0 or 1)"; exit 2 ;;
esac
RUN_SELOP="${RUN_SELOP:-1}"  # Step 1: SeLop per-layer LROR (DDP)       -- MLLM-free
RUN_QWEN="${RUN_QWEN:-1}"    # Step 2a: Qwen3.5-4B base LoRA (0=REUSE the deployed 4B merged)
RUN_DISTILL="${RUN_DISTILL:-1}" # Step 2b: conditioning-distillation continuation (only if RUN_QWEN=1)
RUN_GATE="${RUN_GATE:-1}"    # Gate:   functional eval of the merged MLLM
RUN_GEN="${RUN_GEN:-1}"      # Step 3: generate MIDS answer data with the merged MLLM
RUN_MIDS="${RUN_MIDS:-1}"    # Step 4: MIDS 4-class head (deepspeed) -- needs the MLLM answers
RUN_DEPLOY="${RUN_DEPLOY:-1}"  # Step 5: assemble deploy dir + re-fit the fusion threshold

# ===================== PER-STAGE / PER-DETECTOR GPUs (comma list from {0,1,2,3}) =====================
GPU_QWEN="${GPU_QWEN:-0,1,2,3}"          # MLLM steps: LoRA, Gate + Gen sharding
GPU_MIDS="${GPU_MIDS:-0,1,2,3}"
GPU_9C="${GPU_9C:-0,1,2,3}"
GPU_GSD="${GPU_GSD:-0,1,2,3}"
GPU_SELOP="${GPU_SELOP:-0,1,2,3}"

# ===================== DATASETS =====================
# Qwen MLLM (Step 2) -- eFFAA conversation json + image root:
EFFAA_TRAIN="${EFFAA_TRAIN:-/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json}"
EFFAA_EVAL="${EFFAA_EVAL:-/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/datasets/newout}"
# TRAINSET for ALL detectors = every image under MIDS_IMAGES (a DIRECTORY, walked recursively;
# truth via get_label_all, UNKNOWN dropped). A .json manifest is also accepted.
MIDS_IMAGES="${MIDS_IMAGES:-/datasets/work/vLLM/data/no_delete_mids_train}"
MIDS_EVAL_IMAGES="${MIDS_EVAL_IMAGES:-/datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json}"
TESTSET_DIR="${TESTSET_DIR:-/datasets/work/vLLM/temp/testset}"   # media dir for the Step-5 threshold fit
WORK="${WORK:-$OUT_ROOT/mids_data}"                            # Step 3 (gen) output dir -- PER RUN.
# gen_mids_vllm.py resumes unconditionally (skips images already in the shard), which is what makes a
# crashed 1-day gen recoverable -- but a SHARED dir would let a retrained MLLM silently reuse the old
# model's answers. Per-run by default; override WORK to deliberately resume/reuse (guarded below).
MANI="${MANI:-$PROOT/runs/manifests}"                            # Step 0 manifests
GEN_TRAIN="$WORK/mids_qwen_train.json"; GEN_TEST="$WORK/mids_qwen_testset.json"
IMG_TRAIN="$MANI/images_train.json"                              # GSD / SeLop trainset
M9_TRAIN="$MANI/mids9_train.json"; M9_VAL="$MANI/mids9_val.json" # 9-class A2 train/val
TEST_DATA="${TEST_DATA:-$MIDS_EVAL_IMAGES}"

# ===================== MODELS =====================
QWEN_BASE="${QWEN_BASE:-/datasets/work/vLLM/temp/PAAS_qwen3vl/base_models/Qwen3.5-4B}"
# MLLM = Qwen3.5-4B. RUN_QWEN=1 -> Step-2 merge output; RUN_QWEN=0 -> reuse the deployed 4B merged.
if [ "$RUN_QWEN" = "1" ]; then MERGED="${MERGED:-$OUT_ROOT/qwen_merged}"; else MERGED="${MERGED:-$PROOT/weights/qwen35_4b_merged}"; fi
CLIP="${CLIP:-$PROOT/base_models/clip-vit-large-patch14-336}"
T5="${T5:-$PROOT/base_models/t5-base}"

# ===================== BEST-KNOWN HYPERPARAMS =====================
# trained Qwen3.5-4B recipe (r=32, alpha=48, 3 ep, bs8 x grad-accum4, lr 1e-4, merger full-FT @ 2e-5)
QWEN_EPOCHS="${QWEN_EPOCHS:-3}"; QWEN_BS="${QWEN_BS:-8}"; QWEN_ACC="${QWEN_ACC:-4}"; QWEN_LR="${QWEN_LR:-1e-4}"
QWEN_R="${QWEN_R:-32}"; QWEN_A="${QWEN_A:-48}"; QWEN_MLR="${QWEN_MLR:-2e-5}"
# real-oversample factor -- DATASET-DEPENDENT: 'auto' = n_fake/n_real measured from EFFAA_TRAIN; 0=off
QWEN_BAL="${QWEN_BAL:-auto}"
QWEN_COND="${QWEN_COND:-/datasets/newout/vqa_info_2+13+4+3_fmt/temp_qwen/effaa_cond_mix.json}"
QWEN_DEPOCHS="${QWEN_DEPOCHS:-1}"; QWEN_DLR="${QWEN_DLR:-2e-5}"; QWEN_DMLR="${QWEN_DMLR:-1e-5}"
GATE_N="${GATE_N:-5000}"; GEN_BS="${GEN_BS:-256}"; RETRIES="${RETRIES:-5}"
GEN_MIN_COV="${GEN_MIN_COV:-0.95}"   # min generated/expected ratio before any branch trains
GATE_MIN_PARSE="${GATE_MIN_PARSE:-99}"; GATE_MIN_REAL="${GATE_MIN_REAL:-45}"
GATE_MIN_CR="${GATE_MIN_CR:-60}"; GATE_MIN_CF="${GATE_MIN_CF:-80}"
# fake-class verdict accuracy was previously computed but never gated (a model that fails on
# fakes could pass). Observed on the deployed 4B: 99.97.
GATE_MIN_FAKE="${GATE_MIN_FAKE:-80}"
# conditioned answers that fail to parse are EXCLUDED from the compliance denominator, so
# compliance can read ~100% while most conditioned outputs are unusable -> gate the parse-rate.
GATE_MIN_CPARSE="${GATE_MIN_CPARSE:-90}"
# MIDS 4-class head: FROM SCRATCH (init="") + AUC selection -- the Exp-17/18/20 lesson.
MIDS_INIT="${MIDS_INIT:-}"; MIDS_EPOCHS="${MIDS_EPOCHS:-5}"; MIDS_LR="${MIDS_LR:-1e-5}"; MIDS_BS="${MIDS_BS:-24}"; MIDS_SELECT="${MIDS_SELECT:-auc}"
MIDS_WD="${MIDS_WD:-0.01}"   # what the proven Exp-18 head actually trained at (see train_mids4c.py)
# ONE seed for ALL branches, so a re-run is a controlled variation rather than four independent
# lotteries. Defaults differed per trainer (9c 0, GSD 42, SeLop 42) and MIDS-4c was NOT SEEDED AT
# ALL (it drives its own loop, so TrainingArguments.seed was inert until set_seed was added).
SEED="${SEED:-0}"
# 9-class A2: num_classes 9 + artifact OFF -- matches the DEPLOYED A2_svdgend_9c.pt config exactly.
NINEC_CFG="${NINEC_CFG:-train/configs/mids_a2_9c.yaml}"
GSD_CFG="${GSD_CFG:-train/configs/gsd_default.json}"
SELOP_CFG="${SELOP_CFG:-train/configs/selop_config.json}"; SELOP_INIT="${SELOP_INIT:-}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"    # 1 = deploy even if a stage failed (mixes new+old weights)
DEPLOY_FLOOR="${DEPLOY_FLOOR:-0.90}"   # real-recall floor the deployed threshold is fitted at

mkdir -p "$OUT_ROOT" "$WORK" "$MANI"

# ---- REPRODUCIBILITY RECORD -------------------------------------------------------------
# One seed drives all four detector branches; write it, and every knob that shapes the result,
# next to the weights. A seed you cannot find later is not reproducible. Copied into deploy/ at
# Step 5 so a promoted build carries its own provenance.
# CAVEAT (stated, not hidden): identical seed + identical data reproduces the RECIPE, not
# bit-identical weights -- cuDNN autotune, atomics, NCCL reduction order and dataloader worker
# interleaving are all non-deterministic on multi-GPU. Expect metrics to match closely, not exactly.
export PYTHONHASHSEED="$SEED"
cat > "$OUT_ROOT/run_config.json" <<EOF
{
  "seed": $SEED,
  "started": "$(date -Is)",
  "out_root": "$OUT_ROOT",
  "trainset": "$MIDS_IMAGES",
  "eval_set": "$MIDS_EVAL_IMAGES",
  "testset_dir": "$TESTSET_DIR",
  "mllm": "$MERGED",
  "gen_work": "$WORK",
  "manifests": "$MANI",
  "stages": {"manifest": $RUN_MANIFEST, "ninec": $RUN_9C, "gsd": $RUN_GSD, "selop": $RUN_SELOP,
             "qwen": $RUN_QWEN, "distill": $RUN_DISTILL, "gate": $RUN_GATE, "gen": $RUN_GEN,
             "mids": $RUN_MIDS, "deploy": $RUN_DEPLOY},
  "gpus": {"qwen": "$GPU_QWEN", "mids": "$GPU_MIDS", "9c": "$GPU_9C", "gsd": "$GPU_GSD", "selop": "$GPU_SELOP"},
  "mids4c": {"init": "$MIDS_INIT", "epochs": $MIDS_EPOCHS, "lr": "$MIDS_LR", "bs": $MIDS_BS,
             "select": "$MIDS_SELECT", "weight_decay": "$MIDS_WD"},
  "qwen": {"epochs": $QWEN_EPOCHS, "bs": $QWEN_BS, "accum": $QWEN_ACC, "lr": "$QWEN_LR",
           "r": $QWEN_R, "alpha": $QWEN_A, "merger_lr": "$QWEN_MLR", "balance_real": "$QWEN_BAL"},
  "configs": {"ninec": "$NINEC_CFG", "gsd": "$GSD_CFG", "selop": "$SELOP_CFG"},
  "gsd_variant": {"faithful": $GSD_FAITHFUL,
                  "_note": "1 = train/train_gsd.py --faithful (gsd/faithful.py, arXiv 2603.09242); 0 = the deployed GSDDetector. The RESOLVED reference set is in gsd_provenance.json, written when the GSD stage launches -- it cannot be resolved here because SMOKE=1 rewrites the data paths further down."},
  "deploy_floor": "$DEPLOY_FLOOR",
  "gen_min_coverage": "$GEN_MIN_COV"
}
EOF
echo "[run_finetuning] SEED=$SEED -> $OUT_ROOT/run_config.json"

# ===================== SMOKE (tiny end-to-end validation) =====================
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  SG="${SMOKE_GPU:-0}"
  echo "[run_finetuning] SMOKE -- tiny chain on GPU $SG."
  GPU_QWEN=$SG; GPU_MIDS=$SG; GPU_9C=$SG; GPU_GSD=$SG; GPU_SELOP=$SG
  MIDS_IMAGES="${SMOKE_IMAGES:-$PROOT/train/configs/_smoke_train.json}"; MIDS_EVAL_IMAGES="$PROOT/train/configs/_smoke_val.json"
  TEST_DATA="$MIDS_EVAL_IMAGES"
  QWEN_EPOCHS=1; QWEN_BS=2; QWEN_ACC=1; QWEN_R=8; QWEN_A=16; MIDS_EPOCHS=1; QWEN_BAL=0
  GATE_N=8; GATE_MIN_PARSE=0; GATE_MIN_REAL=0; GATE_MIN_CR=0; GATE_MIN_CF=0; GATE_MIN_FAKE=0; GATE_MIN_CPARSE=0
  # only substitute the tiny 9c config if the caller did NOT pick one -- otherwise you can
  # never smoke the PRODUCTION config, which is exactly where config faults live.
  [ -z "$_NINEC_SET" ] && NINEC_CFG="train/configs/smoke9.yaml"
  RUN_DEPLOY=0
  # Smoke manifests are PER-RUN. They used to be written to the shared $PROOT/runs/manifests, so a
  # smoke run silently overwrote the production manifests (1,366,146 records -> 40); a later real
  # run with RUN_MANIFEST=0 would then have trained every branch on 40 images without complaint.
  # MANI is re-derived here because IMG_TRAIN/M9_* are computed from it further up.
  MANI="$OUT_ROOT/manifests"
  IMG_TRAIN="$MANI/images_train.json"
  M9_TRAIN="$MANI/mids9_train.json"; M9_VAL="$MANI/mids9_val.json"
  mkdir -p "$MANI"
  # Reuse the deployed 4B unless the caller explicitly set RUN_QWEN: a toy 8-record MLLM cannot
  # emit parseable FFAA JSON, which stalls gen in one-by-one error recovery.
  [ -z "$_QWEN_SET" ] && { RUN_QWEN=0; MERGED="$PROOT/weights/qwen35_4b_merged"; }   # calibration needs the full testset; smoke skips it
fi

# ===================== helpers =====================
ngpu(){ awk -F, '{print NF}' <<<"$1"; }
relids(){ seq -s, 0 $(( $(ngpu "$1") - 1 )); }
first(){ cut -d, -f1 <<<"$1"; }
compute_balance_real(){ # $1 = eFFAA json -> n_fake/n_real (so oversampled reals ~ #fakes). 0 if no reals.
  "$PY" - "$1" <<'PY'
import json, re, sys
d = json.load(open(sys.argv[1])); nr = nf = 0
for r in d:
    conv = r.get("conversations") or []
    if len(conv) < 2 or "image" not in r: continue
    m = re.search(r"Analysis result:\s*(\w+)", conv[1]["value"])
    if m and m.group(1).lower() == "real": nr += 1
    else: nf += 1
print(f"{nf/nr:.4f}" if nr else "0.0")
PY
}
PORT=29500
declare -A STATUS SECS
run_step(){ local name="$1"; shift
  local log="$OUT_ROOT/${name}.log" t0=$SECONDS rc
  echo "================================================================"
  echo "[run_finetuning] >>> $name | $(date '+%H:%M:%S') -> $log"
  ( "$@" ) > "$log" 2>&1; rc=$?
  SECS[$name]=$((SECONDS-t0)); STATUS[$name]=$rc; PORT=$((PORT+1))
  [ $rc -eq 0 ] && echo "[run_finetuning] <<< $name OK (${SECS[$name]}s)" \
                || { echo "[run_finetuning] <<< $name FAILED rc=$rc (${SECS[$name]}s):"; tail -12 "$log"; }
  return $rc
}
# ---- PREFLIGHT: parse the PRODUCTION configs before any GPU work ----
# A config key a parser rejects would otherwise kill a branch mid-run, possibly after the MLLM chain
# has already burned a day. Smoke mode uses its own tiny configs, so only this checks the real ones.
_pf=()
[ "$RUN_9C" = "1" ]    && _pf+=(--ninec "$NINEC_CFG")
[ "$RUN_GSD" = "1" ]   && _pf+=(--gsd "$GSD_CFG")
[ "$RUN_SELOP" = "1" ] && _pf+=(--selop "$SELOP_CFG")
if [ ${#_pf[@]} -gt 0 ]; then
  "$PY" train/validate_configs.py "${_pf[@]}" || {
    echo "[run_finetuning] preflight failed -> abort (nothing has run yet)."; exit 1; }
fi

IFS=',' read -ra QG <<< "$GPU_QWEN"; NQ="${#QG[@]}"
echo "[run_finetuning] out=$OUT_ROOT | trainset=$MIDS_IMAGES | MLLM=$MERGED"

# ===================== STEP 0: build detector manifests (no GPU) =====================
if [ "$RUN_MANIFEST" = "1" ]; then
  run_step manifests bash -c "
    set -e
    '$PY' train/build_manifests.py --images '$MIDS_IMAGES'      --mode images --out '$IMG_TRAIN'
    '$PY' train/build_manifests.py --images '$MIDS_IMAGES'      --mode mids9  --out '$M9_TRAIN'
    '$PY' train/build_manifests.py --images '$MIDS_EVAL_IMAGES' --mode mids9  --out '$M9_VAL'"
  [ "${STATUS[manifests]:-1}" -ne 0 ] && { echo "[run_finetuning] manifest build failed -> abort."; exit 1; }
  grep -hE "^\[manifest|  binary balance|  9-class" "$OUT_ROOT/manifests.log" 2>/dev/null | sed 's/^/    /'
else
  echo "[run_finetuning] Step 0 skipped; using manifests in $MANI"
fi

# ===================== STEP 1: MLLM-FREE detectors (9c / GSD / SeLop) =====================
# The 9-class set uses FIXED claim anchors and GSD/SeLop label from the PATH, so none of these
# depend on the MLLM -- they run first and are unaffected by a Step-2/3 failure.
if [ "$RUN_9C" = "1" ]; then
  N=$(ngpu "$GPU_9C")
  run_step ninec env CUDA_VISIBLE_DEVICES="$GPU_9C" \
    $TORCHRUN --nproc_per_node="$N" --master_port="$PORT" train/train_ensemble9.py --config "$NINEC_CFG" \
      --set seed="$SEED" train_data_path="$M9_TRAIN" val_data_path="$M9_VAL" \
            image_model_path="$CLIP" text_model_path="$T5" output_dir="$OUT_ROOT/ninec"
fi
if [ "$RUN_GSD" = "1" ]; then
  # SMOKE batch_size=8: the production batch (>= 40) exceeds the 40-image smoke set, and with
  # drop_last that yielded ZERO training steps -- smoke reported "gsd OK" for three epochs while
  # train_loss=0.0000 and gstep=0, i.e. the optimisation loop was never executed. Model build,
  # anchor and eval were covered; the training step was not.
  REL=$(relids "$GPU_GSD"); SM=(); [ "$SMOKE" = 1 ] && SM=(eval_limit=24 batch_size=8 epochs=1)
  # GSD_FAITHFUL=1 trains the PAPER-FAITHFUL arm (gsd/faithful.py) instead of the deployed detector.
  # Previously this stage passed no --faithful, so pointing GSD_CFG at gsd_faithful.json silently
  # built GSDConfig + GSDDetector and discarded every faithful field. train_gsd.py now REFUSES that
  # combination outright; this flag is the supported way to ask for it.
  FAITH=(); [ "${GSD_FAITHFUL:-0}" = "1" ] && FAITH=(--faithful)
  # The reference set must NOT be the val split for the faithful arm: it validates under the
  # REFERENCE protocol, so a val-derived basis makes the val metric partly self-fitted -- the exact
  # defect recorded for the deployed GSD in EVAL_SPACE/runs/gsd_regression_explained.json. The
  # deployed arm keeps its historical behaviour (anchor from the eval set) so its row still
  # represents what ships.
  if [ "${GSD_FAITHFUL:-0}" = "1" ]; then GSD_ANCHOR="${GSD_ANCHOR:-$IMG_TRAIN}";
  else GSD_ANCHOR="${GSD_ANCHOR:-$MIDS_EVAL_IMAGES}"; fi
  # Provenance for the one stage whose behaviour is decided by an ENV VAR rather than by its config
  # file: which arm was built, and which file the semantic basis came from. Without this, a
  # checkpoint plus a run_config.json cannot tell you whether the reference basis was the val split
  # -- the difference between a self-fitted val metric and a clean one.
  cat > "$OUT_ROOT/gsd_provenance.json" <<GSDPROV
{
  "faithful": $GSD_FAITHFUL,
  "config": "$GSD_CFG",
  "resolved_anchor_data": "$GSD_ANCHOR",
  "val_data": "$MIDS_EVAL_IMAGES",
  "anchor_is_val_data": $([ "$GSD_ANCHOR" = "$MIDS_EVAL_IMAGES" ] && echo true || echo false),
  "train_data": "$IMG_TRAIN",
  "gpus": "$GPU_GSD",
  "_note": "anchor_is_val_data=true means the inference basis was fitted on the same images the val metric is computed on, so that metric is partly self-fitted -- and so is the best.pt it selects."
}
GSDPROV
  run_step gsd env CUDA_VISIBLE_DEVICES="$GPU_GSD" \
    "$PY" train/train_gsd.py "${FAITH[@]}" --config "$GSD_CFG" \
      --set seed="$SEED" gpus="$REL" train_data="$IMG_TRAIN" val_data="$MIDS_EVAL_IMAGES" \
            anchor_data="$GSD_ANCHOR" clip_path="$CLIP" output_dir="$OUT_ROOT/gsd" "${SM[@]}"
fi
if [ "$RUN_SELOP" = "1" ]; then
  N=$(ngpu "$GPU_SELOP"); INIT=(); [ -n "$SELOP_INIT" ] && INIT=(--init_from "$SELOP_INIT")
  # --batch_size 8: config batch_size is 128 vs a 16-image smoke slice -> drop_last gave zero
  # training steps (same silent gap as GSD above). --max_steps 2 so at least one step is taken.
  SM=(); [ "$SMOKE" = 1 ] && SM=(--max_steps 2 --limit_train 16 --limit_val 8 --batch_size 8)
  run_step selop env CUDA_VISIBLE_DEVICES="$GPU_SELOP" \
    $TORCHRUN --nproc_per_node="$N" --master_port="$PORT" train/train_selop.py --config "$SELOP_CFG" \
      --train_data "$IMG_TRAIN" --val_data "$MIDS_EVAL_IMAGES" --out_dir "$OUT_ROOT/selop" --seed "$SEED" "${INIT[@]}" "${SM[@]}"
fi

# ===================== STEP 2: Qwen3.5-4B base LoRA -> distill continuation -> merge =====================
if [ "$RUN_QWEN" = "1" ]; then
  if [ "$QWEN_BAL" = "auto" ]; then
    QWEN_BAL="$(compute_balance_real "$EFFAA_TRAIN" 2>/dev/null | tail -1)"
    echo "[run_finetuning] balance-real auto -> $QWEN_BAL (n_fake/n_real from $(basename "$EFFAA_TRAIN"))"
  fi
  LORA_OUT="$OUT_ROOT/qwen_lora"; MERGER=(--train-merger --merger-lr "$QWEN_MLR"); SM=()
  [ "$SMOKE" = "1" ] && { MERGER=(); SM=(--limit 8 --eval-limit 4); }
  run_step qwen_lora env CUDA_VISIBLE_DEVICES="$GPU_QWEN" \
    $TORCHRUN --nproc_per_node="$NQ" --master_port="$PORT" train/train_qwen_mllm.py \
      --model "$QWEN_BASE" --train "$EFFAA_TRAIN" --eval "$EFFAA_EVAL" --image-root "$IMAGE_ROOT" \
      --out "$LORA_OUT" --epochs "$QWEN_EPOCHS" --bs "$QWEN_BS" --grad-accum "$QWEN_ACC" --lr "$QWEN_LR" \
      --lora-r "$QWEN_R" --lora-alpha "$QWEN_A" --balance-real "$QWEN_BAL" "${MERGER[@]}" "${SM[@]}" \
      || { echo "[run_finetuning] Step 2a failed -> abort."; exit 1; }
  ADAPTER="$LORA_OUT/adapter_final"
  if [ "$RUN_DISTILL" = "1" ]; then
    DIST_OUT="$OUT_ROOT/qwen_lora_distill"; DMERGER=(--train-merger --merger-lr "$QWEN_DMLR")
    DCOND="$QWEN_COND"; DSM=(); [ "$SMOKE" = "1" ] && { DMERGER=(); DSM=(--limit 8 --eval-limit 4); DCOND="$EFFAA_TRAIN"; }
    run_step qwen_distill env CUDA_VISIBLE_DEVICES="$GPU_QWEN" \
      $TORCHRUN --nproc_per_node="$NQ" --master_port="$PORT" train/train_qwen_mllm.py \
        --model "$QWEN_BASE" --resume-adapter "$LORA_OUT/adapter_final" \
        --train "$DCOND" --eval "$EFFAA_EVAL" --image-root "$IMAGE_ROOT" \
        --out "$DIST_OUT" --epochs "$QWEN_DEPOCHS" --bs "$QWEN_BS" --grad-accum "$QWEN_ACC" --lr "$QWEN_DLR" \
        --lora-r "$QWEN_R" --lora-alpha "$QWEN_A" "${DMERGER[@]}" "${DSM[@]}" \
      && ADAPTER="$DIST_OUT/adapter_final"
    if [ "${STATUS[qwen_distill]:-1}" -ne 0 ] && [ "$ALLOW_PARTIAL" != "1" ]; then
      echo "[run_finetuning] distillation FAILED -- refusing to merge the un-distilled base adapter"
      echo "    (the deployed 4B recipe includes the conditioning distillation). ALLOW_PARTIAL=1 to override."
      exit 1
    fi
  fi
  run_step qwen_merge env CUDA_VISIBLE_DEVICES="$(first "$GPU_QWEN")" \
    "$PY" qwen/merge_lora.py --base "$QWEN_BASE" --adapter "$ADAPTER" --out "$MERGED"
  [ "${STATUS[qwen_merge]:-1}" -ne 0 ] && { echo "[run_finetuning] Step 2 merge failed -> abort."; exit 1; }
else
  echo "[run_finetuning] Step 2 skipped; using merged MLLM at $MERGED"
fi

# ===================== GATE: functional eval of the merged MLLM =====================
if [ "$RUN_GATE" = "1" ]; then
  echo "================================================================"
  echo "[run_finetuning] >>> gate | eval $MERGED on $EFFAA_EVAL ($NQ shard(s))"
  GDIR="$OUT_ROOT/gate"; mkdir -p "$GDIR"; gp=()
  for i in "${!QG[@]}"; do
    CUDA_VISIBLE_DEVICES="${QG[$i]}" VLLM_LOGGING_LEVEL=WARNING \
      "$PY" qwen/eval_effaa.py --model "$MERGED" --eval "$EFFAA_EVAL" --image-root "$IMAGE_ROOT" \
        --limit "$GATE_N" --which-part "$i" --n-divided "$NQ" --compliance \
        --json-out "$GDIR/gate_$i.json" > "$GDIR/gate_$i.log" 2>&1 & gp+=("$!")
  done
  gfail=0; for i in "${!gp[@]}"; do wait "${gp[$i]}" || { echo "  gate shard $i FAILED ($GDIR/gate_$i.log)"; gfail=1; }; done
  [ "$gfail" = "0" ] || { echo "[run_finetuning] gate shards failed -> abort."; exit 1; }
  "$PY" - "$GDIR" "$NQ" "$GATE_MIN_PARSE" "$GATE_MIN_REAL" "$GATE_MIN_CR" "$GATE_MIN_CF" "$GATE_MIN_FAKE" "$GATE_MIN_CPARSE" <<'PY'
import json, sys
gdir, n = sys.argv[1], int(sys.argv[2]); mp, mr, mcr, mcf, mfk, mcp = (float(x) for x in sys.argv[3:9])
N=fmt=ok=0; per={"real":[0,0],"fake":[0,0]}; comp={"real":[0,0],"fake":[0,0]}
for i in range(n):
    d=json.load(open(f"{gdir}/gate_{i}.json")); N+=d["n"]; fmt+=d["ok_fmt"]; ok+=d["ok_verdict"]
    for k in per:
        per[k][0]+=d["per"][k][0]; per[k][1]+=d["per"][k][1]
        c=d.get("comply",{}).get(k,[0,0]); comp[k][0]+=c[0]; comp[k][1]+=c[1]
parse=fmt/max(N,1)*100; real=per["real"][0]/max(per["real"][1],1)*100
fake=per["fake"][0]/max(per["fake"][1],1)*100
# compliance % counts only PARSABLE conditioned outputs -- report their parse-rate too, else a
# model that fails to parse most conditioned answers can still show ~100% compliance.
cpr=comp["real"][1]/max(N,1)*100; cpf=comp["fake"][1]/max(N,1)*100
cr=comp["real"][0]/max(comp["real"][1],1)*100; cf=comp["fake"][0]/max(comp["fake"][1],1)*100
print(f"==== GATE: n={N:,} parse={parse:.2f}% verdict={ok/max(fmt,1)*100:.2f}% "
      f"real={real:.2f}% fake={fake:.2f}% | comply real={cr:.2f}% fake={cf:.2f}% "
      f"(conditioned parse-rate real={cpr:.1f}% fake={cpf:.1f}%) | "
      f"need parse>={mp} real>={mr} fake>={mfk} cr>={mcr} cf>={mcf} cparse>={mcp}")
if parse<mp or real<mr or fake<mfk or cr<mcr or cf<mcf or cpr<mcp or cpf<mcp:
    print("==== GATE FAILED -- chain aborted (RUN_GATE=0 to bypass) ===="); sys.exit(1)
print("==== GATE PASSED ====")
PY
  [ $? -eq 0 ] || exit 1
fi

# ---- guard: $WORK must not hold answers generated by a DIFFERENT MLLM ----
# Reusing them would train/evaluate the MIDS head on another model's answer distribution.
if [ "$RUN_GEN" = "1" ] || [ "$RUN_MIDS" = "1" ]; then
  mkdir -p "$WORK"; STAMP="$WORK/.mllm"
  FP="$(readlink -f "$MERGED" 2>/dev/null || echo "$MERGED")|$(stat -c %Y "$MERGED" 2>/dev/null || echo 0)"
  if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" != "$FP" ]; then
    echo "[run_finetuning] ERROR: $WORK holds MIDS answers from a DIFFERENT MLLM."
    echo "    existing: $(cat "$STAMP")"
    echo "    current : $FP"
    echo "    Reusing them would train the MIDS head on another model's answers (train/serve skew)."
    echo "    Fix: use a fresh WORK (default \$OUT_ROOT/mids_data), or rm -rf $WORK to regenerate."
    exit 1
  fi
  echo "$FP" > "$STAMP"
fi

# ===================== STEP 3: generate MIDS answer data (vLLM, sharded) =====================
if [ "$RUN_GEN" = "1" ]; then
  for TASK in train testset; do
    [ "$TASK" = train ] && { SRC="$MIDS_IMAGES"; OB="$WORK/mids_qwen_train"; } || { SRC="$MIDS_EVAL_IMAGES"; OB="$WORK/mids_qwen_testset"; }
    # a directory is walked recursively (labels via get_label_all); a .json is read as a manifest
    if [ -d "$SRC" ]; then SRCFLAG=(--input-dir "$SRC"); else SRCFLAG=(--images-json "$SRC"); fi
    echo "================================================================"
    echo "[run_finetuning] >>> gen($TASK) | $SRC -> $OB.json ($NQ vLLM shard(s))"
    gp=(); for i in "${!QG[@]}"; do
      CUDA_VISIBLE_DEVICES="${QG[$i]}" VLLM_LOGGING_LEVEL=WARNING \
        "$PY" qwen/gen_mids_vllm.py --model "$MERGED" "${SRCFLAG[@]}" --out "${OB}_$i.json" \
          --which-part "$i" --n-divided "$NQ" --batch-size "$GEN_BS" > "$WORK/gen_${TASK}_$i.log" 2>&1 & gp+=("$!")
    done
    gfail=0; for i in "${!gp[@]}"; do wait "${gp[$i]}" || { echo "  gen shard $i FAILED"; gfail=1; }; done
    [ "$gfail" = 0 ] || { echo "[run_finetuning] gen($TASK) failed -> abort."; exit 1; }
    rp=(); for i in "${!QG[@]}"; do
      if [ -s "${OB}_$i.json.errors.json" ]; then
        CUDA_VISIBLE_DEVICES="${QG[$i]}" VLLM_LOGGING_LEVEL=WARNING \
          "$PY" qwen/gen_mids_vllm.py --model "$MERGED" --recover "${OB}_$i.json.errors.json" \
            --out "${OB}_$i.json" --retries "$RETRIES" >> "$WORK/gen_${TASK}_$i.log" 2>&1 & rp+=("$!")
      fi
    done
    _rfail=0
    for j in "${!rp[@]}"; do wait "${rp[$j]}" || { echo "  recovery shard $j FAILED"; _rfail=$((_rfail+1)); }; done
    [ "$_rfail" -gt 0 ] && echo "  WARNING: $_rfail recovery shard(s) failed -> coverage check below is authoritative"
    [ "$TASK" = train ] && EXP_SRC="$IMG_TRAIN" || EXP_SRC="$MIDS_EVAL_IMAGES"
    "$PY" - "$OB" "$NQ" "$EXP_SRC" "$GEN_MIN_COV" <<'PY' || { echo "[run_finetuning] gen($TASK) coverage check failed -> abort."; exit 1; }
import json, os, sys
from collections import Counter
base, n = sys.argv[1], int(sys.argv[2]); allr, seen = [], set()
for i in range(n):
    for r in json.load(open(f"{base}_{i}.json")):
        if r["image"] not in seen: seen.add(r["image"]); allr.append(r)
json.dump(allr, open(f"{base}.json", "w"))
print(f"  merged -> {base}.json: {len(allr):,} records | cls:", dict(Counter(r['cls_label'] for r in allr)))
# COVERAGE: silently training on a fraction of the intended data is a real risk -- format
# errors and failed recovery shards both shrink the set without failing anything.
exp_src, min_cov = sys.argv[3], float(sys.argv[4])
if exp_src and os.path.exists(exp_src):
    expected = len(json.load(open(exp_src)))
    cov = len(allr) / max(expected, 1)
    print(f"  coverage: {len(allr):,}/{expected:,} = {cov*100:.2f}% (min {min_cov*100:.0f}%)")
    if cov < min_cov:
        print("  COVERAGE TOO LOW -- refusing to train on a truncated set"); sys.exit(1)
PY
  done
else
  echo "[run_finetuning] Step 3 skipped; MIDS-4c data = $GEN_TRAIN / $GEN_TEST"
fi

# ===================== STEP 4: MIDS 4-class head (needs the MLLM answers) =====================
if [ "$RUN_MIDS" = "1" ]; then
  if [ ! -f "$GEN_TRAIN" ]; then
    echo "[run_finetuning] MIDS-4c requested but $GEN_TRAIN is missing (run Step 3 first)."
    echo "    Recording this as a FAILURE: a silent skip would let deploy keep the OLD MIDS head."
    STATUS[mids]=1; SECS[mids]=0
  else
    N=$(ngpu "$GPU_MIDS"); SM=(); [ "$SMOKE" = 1 ] && SM=(--max_steps 1 --per_device_val_batch_size 4)
    run_step mids env CUDA_VISIBLE_DEVICES="$GPU_MIDS" \
      $TORCHRUN --nproc_per_node="$N" --master_port="$PORT" train/train_mids4c.py \
        --image_model_path "$CLIP" --text_model_path "$T5" --init_model_path "$MIDS_INIT" \
        --data_path "$GEN_TRAIN" --val_data_path "$GEN_TEST" --output_dir "$OUT_ROOT/mids" \
        --num_train_epochs "$MIDS_EPOCHS" --learning_rate "$MIDS_LR" \
        --per_device_train_batch_size "$MIDS_BS" --select_metric "$MIDS_SELECT" \
        --weight_decay "$MIDS_WD" --seed "$SEED" "${SM[@]}"
  fi
fi

# ===================== STEP 5: deploy dir + RE-FIT the fusion threshold =====================
# The inherited threshold was fitted on the OLD weights; after retraining it must be re-fitted or
# the fused accuracy degrades even if every head improved.
if [ "$RUN_DEPLOY" = "1" ]; then
  _failed=""
  for k in ninec gsd selop mids qwen_lora qwen_distill qwen_merge; do
    [ -n "${STATUS[$k]+x}" ] && [ "${STATUS[$k]}" -ne 0 ] && _failed="$_failed $k"
  done
  if [ -n "$_failed" ] && [ "$ALLOW_PARTIAL" != "1" ]; then
    echo "[run_finetuning] SKIPPING deploy: failed stage(s):$_failed"
    echo "    A partial deploy silently mixes retrained heads with the previously deployed ones."
    echo "    Re-run the failed stage, or set ALLOW_PARTIAL=1 to build anyway."
    RUN_DEPLOY=0
  fi
fi
if [ "$RUN_DEPLOY" = "1" ]; then
  run_step deploy "$PY" train/make_deploy_config.py --run "$OUT_ROOT" --qwen "$MERGED"
  DCFG="$OUT_ROOT/deploy/paas_retrained.json"
  if [ "${STATUS[deploy]:-1}" -eq 0 ] && [ -f "$DCFG" ] && [ -d "$TESTSET_DIR" ]; then
    run_step calibrate bash -c "
      set -e
      GPUS='$GPU_MIDS' bash scripts/run_dataset.sh --config '$DCFG' \
        --input-dir '$TESTSET_DIR' --out-dir '$OUT_ROOT/deploy/eval' --frame-stride 10
      '$PY' train/fit_threshold.py --results '$OUT_ROOT/deploy/eval/results_paas.txt' \
        --floor '$DEPLOY_FLOOR' --config '$DCFG'"
    cp -f "$OUT_ROOT/run_config.json" "$OUT_ROOT/deploy/run_config.json" 2>/dev/null || true
    grep -hE "real-floor|AUC=|wrote threshold" "$OUT_ROOT/calibrate.log" 2>/dev/null | sed 's/^/    /'
  else
    echo "[run_finetuning] calibrate skipped (deploy failed or TESTSET_DIR missing: $TESTSET_DIR)"
  fi
fi

# ===================== summary =====================
echo "================================================================"
echo "[run_finetuning] SUMMARY (out=$OUT_ROOT)"
fail=0
for k in manifests ninec gsd selop qwen_lora qwen_distill qwen_merge mids deploy calibrate; do
  [ -z "${STATUS[$k]+x}" ] && { printf "  %-12s: skipped\n" "$k"; continue; }
  if [ "${STATUS[$k]}" -eq 0 ]; then printf "  %-12s: OK    %5ss\n" "$k" "${SECS[$k]}"
  else printf "  %-12s: FAIL(%s) %5ss\n" "$k" "${STATUS[$k]}" "${SECS[$k]}"; fail=1; fi
done
echo "[run_finetuning] MLLM: $MERGED"
echo "[run_finetuning] deploy build: $OUT_ROOT/deploy (promote to weights/ + config/ once the numbers look right)"
exit $fail
