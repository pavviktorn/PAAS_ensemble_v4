# Training — PAAS_ensemble_v4 (single venv: transformers 5.13 + vLLM)

All five detectors train on the SAME venv as inference:
`VENV=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin`

Each driver self-bootstraps sys.path (see `_bootstrap.py`) so it finds the vendored libs
(`gsd/`, `selop/`, `ensemble9/mids9lib/`, `ffaa/mids`, `ffaa/utils`). Run from the project root.
Configs live in `train/configs/`. Deploy across GPUs 0-3; the commands below show single-GPU.

## MIDS 4-class head (deepspeed)
```
$VENV/python -m torch.distributed.run --nproc_per_node=1 train/train_mids4c.py \
  --image_model_path base_models/clip-vit-large-patch14-336 --text_model_path base_models/t5-base \
  --init_model_path "" \                     # "" = from-scratch head (project-best; avoids the axon warm-start real-FP wall)
  --data_path <train.json> --val_data_path <val.json> \
  --output_dir runs/mids --num_train_epochs 5 --per_device_train_batch_size 24
```

## 9-class ensemble (A1 svd / A2 svd+gend / A3 mids++) — plain PyTorch DDP+AMP
```
$VENV/torchrun --nproc_per_node=4 train/train_ensemble9.py --config train/configs/mids_pp.yaml \
  --set train_data_path=<train.json> val_data_path=<val.json> \
        image_model_path=base_models/clip-vit-large-patch14-336 text_model_path=base_models/t5-base
# A1: mids_svd_only.yaml   A2: mids_pp.yaml   A3: mids_original.yaml (mids++)
```

## GSD (dual-stream, DataParallel)
```
$VENV/python train/train_gsd.py --config train/configs/gsd_default.json \
  --set gpus=0,1,2,3 train_data=<train.json> val_data=<val.json>
```

## SeLop (per-layer LROR, DDP)
```
$VENV/torchrun --nproc_per_node=4 train/train_selop.py --config train/configs/selop_config.json \
  --out_dir runs/selop
```

## Qwen3.5 MLLM (LoRA via transformers.Trainer + peft) — the FFAA MLLM
```
$VENV/torchrun --nproc_per_node=4 train/train_qwen_mllm.py \
  --model <base_qwen_dir> \
  --train /datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json \
  --eval  /datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json \
  --image-root /datasets/newout --out runs/qwen_lora --train-merger --balance-real 2.44
# then merge the adapter into a standalone model for in-process vLLM inference.
```

## Smoke tests (verified 2026-08-05, all pass on tf5)
Tiny fixtures `train/configs/_smoke_{train,val}.json` (40/24 items). Append to any command:
- MIDS/GSD/SeLop/9c: add `--max_steps 1` / `--limit` / small data; 9c: `--config train/configs/smoke.yaml` (stub encoders, CPU).
- Qwen: `--limit 8 --epochs 1 --bs 2`.
