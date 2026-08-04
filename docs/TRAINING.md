# Training

Train the two Genesis operators — **super-resolution** and **outpainting** — from scratch. The
launcher defaults already match the paper's main config; the full hyperparameter table (lr, betas,
EMA, curriculum boundaries, precision, …) is in [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) §2.

## Setup
```bash
uv sync && source .venv/bin/activate
```
Training reads Git-10M (🤗 `lcybuaa/Git-10M`) plus the quad-split metadata + hierarchy index from
the Google Drive release — all auto-downloaded on first use (see the top-level `README.md` for the
rclone `gdrive` remote). DINOv3-sat weights (SR) auto-pull the same way.

> **Global batch = 256** in the paper = `batch_size 32 × accumulate_grad_batches 4 × devices 2`.
> On a single GPU that is 128 — pass `--accumulate_grad_batches 8` to keep 256.

## Super-resolution
```bash
cd src/superresolution
chmod +x run_sr.sh
# Defaults = paper main: DINO + LPIPS 0.5, 800k steps, JiT-B/16.
# MODEL_SIZE=H16 for the H/16 run; DEVICES=1 for a single GPU (default 2).
MODEL_SIZE=H16 DEVICES=1 OUTPUT_DIR=./runs/main_h16 WANDB_MODE=online ./run_sr.sh

# Resume from a Lightning checkpoint:
RESUME_CKPT=./runs/main_h16/last.ckpt \
  MODEL_SIZE=H16 DEVICES=1 OUTPUT_DIR=./runs/main_h16 WANDB_MODE=online ./run_sr.sh
```

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_SIZE` | `B16` | `B16` or `H16` |
| `MAX_STEPS` | `800000` | paper total; curriculum boundaries auto-computed at 20/40/60 % |
| `NO_DINO` | `0` | `1` = pixel-only conditioning |
| `LPIPS_WEIGHT` | `0.5` | per-quadrant LPIPS weight (paper main; `0.0` = `*_nolpips` ablation) |
| `DEVICES` | `2` | GPU count or `"0,1"` |
| `WANDB_MODE` | `disabled` | `online` to log to W&B |
| `RESUME_CKPT` | _(none)_ | Lightning ckpt to resume |

Full ablation matrix, architecture, and CLI flags → [`../src/superresolution/README.md`](../src/superresolution/README.md).

## Outpainting
```bash
cd src/outpainting
chmod +x run_op.sh
# Defaults = paper main: no-DINO + no-time-weight, quadrant curriculum, 800k steps, JiT-B/16.
MODEL_SIZE=H16 DEVICES=1 OUTPUT_DIR=./runs/main_h16 WANDB_MODE=online ./run_op.sh
```

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_SIZE` | `B16` | `B16` or `H16` |
| `MAX_STEPS` | `800000` | paper total; curriculum boundaries auto-computed at 20/40/60 % |
| `DEVICES` | `2` | GPU count or `"0,1"` |
| `WANDB_MODE` | `disabled` | `online` to log to W&B |
| `RESUME_CKPT` | _(none)_ | Lightning ckpt to resume |

Curriculum stage map, architecture, and CLI flags → [`../src/outpainting/README.md`](../src/outpainting/README.md).

## Reproducing checkpoints
The released main checkpoints are 800k steps (SR ablations 500k). Curriculum boundaries fall at
20/40/60 % of `MAX_STEPS`. See [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) for the exact per-operator
settings and [`EVALUATION.md`](EVALUATION.md) to evaluate a trained checkpoint against the paper.
