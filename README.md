# PAAS_ensemble_v4_inf

Standalone face real/fake **ensemble** inference service. v4 replaces the v3 LLaVA-Mistral-7B FFAA
with the project-best **Qwen3.5-4B FFAA** and uses the empirically-selected best deployable
combination.

## Best combination (selected by an axonlabs_data_1 fusion sweep)

**`mean( ffaa , A2_9c , gsd , selop )`** — plain mean of four per-frame fake-scores:

| detector | what it is |
|----------|------------|
| `ffaa`   | **Qwen3.5-4B** MLLM (vLLM, 3-pass conditional, `enable_thinking=False`) + **from-scratch** MIDS 4-class head |
| `A2_9c`  | 9-class SVD+GenD ensemble member (MLLM-free, CLIP+T5) |
| `gsd`    | Geometric Semantic Decoupling (CLIP) |
| `selop`  | SeLop / LROR low-rank orthogonal (CLIP) |

axonlabs_data_1 (613,415 frames, plain mean): **AUC 0.9998**, fake-recall **99.74 %** at a 99 %
real-recall floor. This beats the Qwen3.5-4B FFAA alone (AUC 0.9993 / FR99 99.27) — GSD & SeLop
(full-image CLIP, a different modality) catch the few frames the MLLM misses. Rank-mean scores
slightly higher but needs whole-dataset ranks, so it is **not** deployable; plain mean is used.
(The 9-class A1/A3 add nothing on top and A3 collapses at high real-floors, so only A2 is kept.)

### Operating thresholds (mean, on axonlabs_data_1)
| real-recall floor | tau | fake-recall |
|---|---|---|
| 90 % | 0.2192 (default) | 99.96 % |
| 95 % | 0.2688 | 99.93 % |
| 98 % | 0.3893 | 99.82 % |
| 99 % | 0.4634 | 99.74 % |

## Why the from-scratch MIDS head
The axon0 (LLaVA-era) MIDS **warm-start** imported CLIP calibration that over-fired "fake" on hard/
blurry real identities (R_13, R_15). Training the head **from scratch** on Qwen3.5-4B's own answers
removed it: axon1 AUC 0.9919→0.9993, FR99 0.7464→0.9927; testset R_13 real-recall 89.7→98.8.

## Environment (IMPORTANT)
Qwen3.5-4B is a hybrid linear-attention model needing **transformers 5.13 + vLLM 0.21**, so the WHOLE
ensemble runs on that **venv** (NOT global python3.12). The MIDS/GSD/SeLop components are compatible
with that stack (verified). `run_server.sh` uses `PAAS_PY` (default: the PAAS_qwen3vl venv).

## Run
```bash
bash run_server.sh                 # v4 best combo, GPU 0, :8000
PAAS_CONFIG=config/experiments/ffaa_only.json bash run_server.sh   # Qwen FFAA alone
python inference.py --config config/experiments/paas4_qwen.json --dir <images>   # offline
```
All four detectors load onto ONE GPU (vLLM KV fraction `ffaa.gpu_mem=0.5` leaves room for the three
CLIP heads). vLLM is constructed FIRST so it owns CUDA init.

## Layout
```
weights/qwen35_4b_merged/     merged Qwen3.5-4B MLLM (symlink)
weights/ffaa_qwen35_mids/     from-scratch MIDS 4-class head (best.pth)
weights/ensemble9/            A2 (+A1/A3) 9-class members
weights/gsd/ , weights/selop/ CLIP detector heads
base_models/                  clip-vit-large-patch14-336 , t5-base (symlink)
paas/                         fusion framework (models/, fusion.py, pipeline.py, decision.py)
config/experiments/paas4_qwen.json   the best-combination config
app_fastapi_json_v5_safe_jsonfmt.py  FastAPI service
```
