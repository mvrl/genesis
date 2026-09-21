# Training

Train the two Genesis operators — **super-resolution** and **outpainting** — from scratch. The
launcher defaults already match the paper's main config; the full hyperparameter table (lr, betas,
EMA, curriculum boundaries, precision, …) is in [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md) §2.

## Setup
```bash
uv sync && source .venv/bin/activate
```
Training reads Git-10M (🤗 `lcybuaa/Git-10M`) plus the quad-split metadata + hierarchy index from
the Hugging Face release (🤗 [`MVRL/genesis`](https://huggingface.co/MVRL/genesis)) — all
auto-downloaded on first use, no authentication needed. DINOv3-sat weights (SR) auto-download the
same way.

> **Global batch = 256** in the paper = `batch_size 32 × accumulate_grad_batches 4 × devices 2`.
> On a single GPU that is 128 — pass `--accumulate_grad_batches 8` to keep 256.

## Super-resolution
```bash
cd src/superresolution
chmod +x run_sr.sh
# Defaults = paper main: DINO + LPIPS 0.5, 800k steps, JiT-B/16.
# MODEL_SIZE=H16 for the H/16 run; DEVICES=1 for a single GPU (default 2 — on one
# GPU add --accumulate_grad_batches 8 to keep the paper's effective batch of 256).
MODEL_SIZE=H16 DEVICES=1 OUTPUT_DIR=./runs/main_h16 WANDB_MODE=online ./run_sr.sh --accumulate_grad_batches 8

# Resume from a Lightning checkpoint:
RESUME_CKPT=./runs/main_h16/last.ckpt \
  MODEL_SIZE=H16 DEVICES=1 OUTPUT_DIR=./runs/main_h16 WANDB_MODE=online ./run_sr.sh --accumulate_grad_batches 8
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
MODEL_SIZE=H16 DEVICES=1 OUTPUT_DIR=./runs/main_h16 WANDB_MODE=online ./run_op.sh --accumulate_grad_batches 8
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

## Auxiliary side-channel SR (non-RGB targets)

The SR operator can also super-resolve a **non-RGB raster** (e.g. a DSM) while conditioning on
a **co-registered auxiliary raster** at a different resolution (e.g. high-resolution optical).
This is off by default and does not change any behaviour above. See
[`SIDE_CHANNEL.md`](SIDE_CHANNEL.md) for the three fusion mechanisms (`early` / `crossattn` /
`late`), the generic `target`/`parent`/`side` batch protocol, the wall-clock-budgeted training
loop, and the terrain-aware metrics.
