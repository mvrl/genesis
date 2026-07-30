# Genesis

A Generative Engine for Hierarchical Satellite Image Synthesis.

```
src/
├── superresolution/   # Parent-conditional 256 → 512 SR (ZL → ZL+1)
├── outpainting/       # Mask-based 256×256 image completion
├── pyramid/           # Full-pyramid generation engine (multi-scale tile completion)
├── eval/              # Per-sample + pyramid evaluation (reproduces the paper tables)
├── dataprep/          # One-shot scripts to build the shared meta + hierarchy
└── util/              # Shared building blocks (RoPE, RMSNorm, sin-cos pos embed)
```

| Task | Directory | Launcher | README |
|------|-----------|----------|--------|
| Super-resolution | [`src/superresolution/`](src/superresolution/) | `run_sr.sh` | [`src/superresolution/README.md`](src/superresolution/README.md) |
| Outpainting | [`src/outpainting/`](src/outpainting/) | `run_op.sh` | [`src/outpainting/README.md`](src/outpainting/README.md) |
| Evaluation | [`src/eval/`](src/eval/) | `run_eval_sr.sh` / `run_eval_op.sh` | [`src/eval/README.md`](src/eval/README.md) + [`docs/METRICS.md`](docs/METRICS.md) |

Interactive demos under [`demos/`](demos/) auto-fetch checkpoints from
Google Drive and run inference end-to-end — see
[`demos/README.md`](demos/README.md).

📖 **Reproducibility docs** live in [`docs/`](docs/): every training and evaluation
hyperparameter (paper vs. code, per experiment) is tabulated in
[`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md); metric definitions and backends in
[`docs/METRICS.md`](docs/METRICS.md).

---

## Installation

Uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync                       # creates .venv and installs everything (pinned via uv.lock)
source .venv/bin/activate     # or prefix each command with `uv run`
```

`uv sync` installs the exact locked versions (Python 3.13, torch 2.8.0 + CUDA
12.8). For CPU-only, drop the `pytorch-cu128` index in `pyproject.toml` first.
Checkpoints, datasets, and weights still download on first use (see below) —
they are **not** part of the environment.

---

## Data & weights — all on Hugging Face or Google Drive

Every external artifact is hosted on the Hugging Face Hub or the public Google
Drive release folder (`1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA`) and is fetched
automatically on first use — nothing lives on a private cluster path.

| Artifact | Hosted on | How it's obtained |
|----------|-----------|-------------------|
| Git-10M tiles (training/eval images) | 🤗 `lcybuaa/Git-10M` | `datasets.load_dataset(...)` (HF cache; override with `$HF_DATASETS_CACHE`) |
| `git10m_quad_meta.json` (splits) | Google Drive `shared/` | auto-pull to `$GENESIS_CKPT_BASE/shared/` (rclone `gdrive:`); override with `$GENESIS_GIT10M_META` |
| `hierarchy_index_quad.pkl` (SR parent→child) | Google Drive `shared/` | auto-pull to `$GENESIS_CKPT_BASE/shared/`; override with `$GENESIS_HINDEX` |
| dense500 benchmark | 🤗 `dcher95/dense500` | auto-download via `pyramid.dataset.GlobalDense` (override with `$GENESIS_DENSE500_REPO`) |
| Checkpoints — main **and** ablation | Google Drive release | auto-pull to `$GENESIS_CKPT_BASE` (`--exp main_b16 / main_h16 / abl_*`) |
| DINOv3-sat encoder weights | Google Drive release | auto-pull; override with `$GENESIS_DINO_WEIGHTS` (`NO_DINO=1` to ablate) |

`git10m_quad_meta.json` carries the `splits` key (seeded-shuffled, ZL-balanced
index lists for `train / val_random / val_spatial / random_test / spatial_test`).
Set `GENESIS_CKPT_BASE` to choose where Google-Drive files cache (default
`<repo>/logs/ckpts/`); auto-pull needs an rclone Google Drive remote named
`gdrive`. See **Pretrained checkpoints** below for the rclone setup.

---

## Quick start — Super-resolution

```bash
cd src/superresolution
chmod +x run_sr.sh

# Defaults already match the paper main config (DINO + LPIPS 0.5, 800k steps, JiT-B/16).
# MODEL_SIZE=H16 for the H/16 run; DEVICES=1 for a single GPU (default 2).
MODEL_SIZE=H16 DEVICES=1 OUTPUT_DIR=./runs/main_h16 WANDB_MODE=online ./run_sr.sh

# Resume:
RESUME_CKPT=./runs/main_h16/last.ckpt \
  MODEL_SIZE=H16 DEVICES=1 OUTPUT_DIR=./runs/main_h16 WANDB_MODE=online ./run_sr.sh
```

> **Global batch = 256** in the paper (`batch_size × accumulate_grad_batches × num_gpus`
> = 32 × 4 × 2). On one GPU that's 128 — pass `--accumulate_grad_batches 8` to keep 256.

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_SIZE` | `B16` | `B16` or `H16` |
| `MAX_STEPS` | `800000` | paper total; curriculum boundaries auto-computed at 20/40/60 % |
| `NO_DINO` | `0` | `1` = pixel-only conditioning |
| `LPIPS_WEIGHT` | `0.5` | per-quadrant LPIPS weight (paper main; `0.0` = *_nolpips ablation) |
| `DEVICES` | `2` | GPU count or `"0,1"` |
| `WANDB_MODE` | `disabled` | `online` to log to W&B |
| `RESUME_CKPT` | _(none)_ | Lightning ckpt to resume |

Full ablation matrix, architecture, full CLI flag list →
[`src/superresolution/README.md`](src/superresolution/README.md).

---

## Quick start — Outpainting

```bash
cd src/outpainting
chmod +x run_op.sh

# Defaults already match the paper main config (no-DINO + no-time-weight, quadrant
# curriculum, 800k steps, JiT-B/16). MODEL_SIZE=H16 for H/16; DEVICES=1 for one GPU.
MODEL_SIZE=H16 DEVICES=1 OUTPUT_DIR=./runs/main_h16 WANDB_MODE=online ./run_op.sh
```

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_SIZE` | `B16` | `B16` or `H16` |
| `MAX_STEPS` | `800000` | paper total; curriculum boundaries auto-computed at 20/40/60 % |
| `DEVICES` | `2` | GPU count or `"0,1"` |
| `WANDB_MODE` | `disabled` | `online` to log to W&B |
| `RESUME_CKPT` | _(none)_ | Lightning ckpt to resume |

Curriculum stage map, architecture, full CLI flag list →
[`src/outpainting/README.md`](src/outpainting/README.md).

---

## Quick start — Evaluation

Evaluation lives in [`src/eval/`](src/eval/) and reproduces the **Genesis (B/16 and H/16)**
numbers in **every paper table** — SR (Table 2), outpainting (Table 3), dense500 pyramid
completion (Table 4), the SR ablation (Table 5), and mean FID (Table 6). Metrics are
PSNR / SSIM / LPIPS / DISTS / B-diff (OP) / FID for the tile tasks, plus CLIP-I / DINOv3-sat /
Caption and LR-PSNR / xLR-PSNR / RAPSD for the pyramid — all library-backed; see
[`docs/METRICS.md`](docs/METRICS.md). Checkpoints and the dense500 benchmark
auto-download on first use.

```bash
cd src/eval

# SR (Table 2): score a checkpoint on a split (writes samples.csv + images), then FID
EXP=main_h16 MODEL=JiT-H/16 SPLIT=random_test OUT_DIR=results/sr_h16_random ./run_eval_sr.sh
python compute_fid_kid.py --root results --pattern sr_h16_random --partition parent_zl

# Outpainting (Table 3): one mask regime at a time
REGIME=quad1 EXP=main_h16 MODEL=JiT-H/16 SPLIT=random_test ./run_eval_op.sh

# pivot samples.csv into the paper layout
python aggregate.py --task sr --metric lpips --pivot exp --columns split,zl_bucket
```

The full per-table recipe (including dense500 generation + scoring for Table 4) is in
**[`src/eval/README.md`](src/eval/README.md)**. Comparison baselines are not included — this
repo evaluates our checkpoints only.

---

## Quick start — Demo

The flagship demo is the **full-pyramid generator**: drop sparse seed tiles on a map and Genesis
completes the entire multi-scale quadtree (the paper's *multi-scale tile completion*).

```bash
export GENESIS_CKPT_BASE=/path/to/ckpt_cache   # where Google-Drive ckpts land (rclone `gdrive` remote)
python demos/app_pyramid_unified.py            # opens a Gradio app; prints a local + shareable URL
```

Click the map to drop seeds → **Generate**; the main B/16 + H/16 checkpoints auto-pull from Google
Drive on first use. The other apps (per-tile SR, single-tile outpainting, deep 6-level pyramid,
no-model interpolation baselines) are documented in [`demos/README.md`](demos/README.md).

---

## Pretrained checkpoints

The **main** 800k-step ckpts are mirrored to a public Google Drive folder:

<https://drive.google.com/drive/folders/1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA>

```
main/                                # 800k-step main runs, end-of-stage-3
├── superresolution/main_{B16,H16}/sr-full-tile-stage3-step0800000.ckpt        # DINO + LPIPS=0.5
└── outpainting/main_{B16,H16}/op-new-stage3-step0800000.ckpt                  # no DINO, no TW, quadrant curriculum
dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth                                    # DINO encoder weights
shared/git10m_quad_meta.json                                                   # eval metadata
shared/hierarchy_index_quad.pkl                                                # SR parent→child lookup
```

Main is what the Gradio demos default to: SR = DINO + LPIPS(0.5), OP =
no-DINO + no-TW (see [`src/outpainting/README.md`](src/outpainting/README.md)).

The folder is shared as "Anyone with the link can view", so configuring
any Google account with rclone works (`rclone config` → `n` (new
remote) → `drive` → accept defaults → `n` for advanced → use the OAuth
flow). The folder ID is passed via `--drive-root-folder-id`, so the
remote itself does not need to be pinned to this folder.

Set `GENESIS_CKPT_BASE` first (add to `~/.bashrc` to persist):

```bash
export GENESIS_CKPT_BASE=/path/to/checkpoints/project_genesis/gdrive_copy
```

Then pull with rclone (requires a Google Drive remote named `gdrive` — `type = drive`):

```bash
GDRIVE_FOLDER_ID=1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA

# pull everything (~57 GB) — what the demos use
rclone copy gdrive: "$GENESIS_CKPT_BASE" \
  --drive-root-folder-id "$GDRIVE_FOLDER_ID" \
  --progress --transfers 4

# or just one variant
rclone copy gdrive:main/superresolution/main_B16 \
  "$GENESIS_CKPT_BASE/main/superresolution/main_B16" \
  --drive-root-folder-id "$GDRIVE_FOLDER_ID" --progress
```

The Gradio demos auto-pull these on first **Generate** — manual `rclone copy`
is only needed if you want to stage them ahead of time.

**Ablations.** The SR Phase-1 ablation ckpts
(`abl_{nodino_nolpips,dino_nolpips,nodino_lpips,dino_lpips}`) are on
Google Drive under `ablations/` in the same release folder. OP is
ablation-free (the locked OP config is no-DINO + no-time-weight under
the quadrant mask curriculum), so this only affects SR.
