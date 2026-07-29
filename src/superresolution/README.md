# Super-Resolution — Full-Tile ZL → ZL+1

Generates a 512×512 stitched mosaic (4 children at ZL+1) from a 256×256 parent
tile at ZL. Single-step ×2; iterate the model for higher scales.

```bash
cd src/superresolution
source .venv/bin/activate
chmod +x run_sr.sh
```

All paths (DINO weights, `git10m_quad_meta.json`, `hierarchy_index_quad.pkl`,
HF cache) are hardcoded at the top of `run_sr.sh` — edit there if your
layout differs.

---

## Architecture

```
parent 256×256 ──► parent_context_embedder (Conv2d patch=16) ──┐
                                                                │
[--use_dino] ──► DINO-ViT-L (frozen) ──► parent_proj ───────────┤  K, V
                                                                │
noisy child 512×512 ──► BottleneckPatchEmbed (patch=32) ────────┤  Q
                       + sincos pos_embed                       │
                       JiTBlock × depth                          │
                         self-attn (RoPE) ◄─────────────────────┘
                         sr_cross_attn   (child Q ↔ parent K,V)
                         adaLN(c = t_emb + zoom_emb)
                       FinalLayer(p=32) → unpatchify
                                ↓
                       512×512 clean child mosaic
```

- **Pixel parent** (Conv2d patch=16 → 256 tokens) is always active.
- **DINO** (`--use_dino`, default on): adds 256 frozen ViT-L tokens
  → 512 context tokens after concat.
- **Zoom-level conditioning** (always on): target tile zoom (= parent ZL+1) is
  embedded with a sinusoidal MLP and added to the AdaLN vector
  `c = t_emb + zoom_emb`. `zoom=0` is a learned "unknown" sentinel.

---

## Training

**Data** — `lcybuaa/Git-10M`, `zoom_diff=1` only. Built on the fly by
`HFParentChildDataset` from the pre-shuffled `splits` lists in
`git10m_quad_meta.json` plus the parent→child lookup in
`hierarchy_index_quad.pkl`.

| Split | Pair type | Quantity |
|-------|-----------|----------|
| `train`        | complete + partial | 362,572 + 530,021 |
| `val_random`   | complete only | 5,000 |
| `val_spatial`  | complete only | 5,000 (50% urban) |

Partial-quad training is supported: the engine masks the velocity loss to
known children only via `children_valid`. LPIPS (when enabled) crops each
valid child quadrant from the 512×512 prediction and pairs it with the
matching child GT — no zero-filled composites pass through AlexNet.

**Curriculum** — auto-computed boundaries at 20 / 40 / 60 % of `--max_steps`,
rounded to the nearest 5 k. End-step columns below give the exact step at
which each stage *finishes* for the two canonical `MAX_STEPS` settings
(500 k for Phase-1 ablations, 800 k for the main run):

| Stage | Sampling | End step (500k) | End step (800k) |
|-------|----------|---------------:|---------------:|
| 0 | 100 % easy (parent ZL ≤ `--zl_easy_max`, default 15) | 100,000 | 160,000 |
| 1 | 50 % easy + 50 % hard | 200,000 | 320,000 |
| 2 | uniform ZL, all pairs | 300,000 | 480,000 |
| 3 | uniform ZL, complete quads only | 500,000 | 800,000 |

`ZLCurriculumSampler` reads `shared_stage` per epoch; `on_train_batch_start`
updates it from `global_step` every step (transitions accurate to ~1k steps).

**Loss** — masked velocity MSE  +  optional LPIPS:
`total = Σ(sq · mask)/Σ(mask)  +  lpips_weight · Σ(t · lpips)/Σ(t)`

**Checkpoints** — two callbacks: top-3 by `val/val_random_loss` (+`last.ckpt`)
and `StepCheckpointCallback` saving 10 uniform steps + each stage end (label
`uniform`/`stage{0..3}`). Resume with `RESUME_CKPT=...`.

---

## Experiment plan

All runs from this directory; the only difference between them is `NO_DINO` /
`LPIPS_WEIGHT` / `OUTPUT_DIR`.

```bash
# Phase 1 — 2×2 ablation (B16, 500k steps each, run 4 jobs in parallel)
MODEL_SIZE=B16 MAX_STEPS=500000 \
  NO_DINO=1                  OUTPUT_DIR=./runs/abl_nodino_nolpips ./run_sr.sh   # A
MODEL_SIZE=B16 MAX_STEPS=500000 \
                             OUTPUT_DIR=./runs/abl_dino_nolpips    ./run_sr.sh   # B
MODEL_SIZE=B16 MAX_STEPS=500000 \
  NO_DINO=1   LPIPS_WEIGHT=0.5 OUTPUT_DIR=./runs/abl_nodino_lpips  ./run_sr.sh   # C
MODEL_SIZE=B16 MAX_STEPS=500000 \
              LPIPS_WEIGHT=0.5 OUTPUT_DIR=./runs/abl_dino_lpips    ./run_sr.sh   # D
```

| Run | DINO | LPIPS |
|-----|------|-------|
| A | ✗ | ✗ |
| B | ✓ | ✗ |
| C | ✗ | ✓ |
| D | ✓ | ✓ |

Carry the winner forward to the main run (800 k steps × {B16, H16}). To
resume: add `RESUME_CKPT=./runs/.../last.ckpt` and re-run.

---

## Configuration

Env vars (consumed by `run_sr.sh`):

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_SIZE` | `B16` | `B16` or `H16` |
| `MAX_STEPS` | `800000` | paper total; curriculum boundaries auto-computed at 20/40/60 % |
| `NO_DINO` | `0` | `1` = pixel-only conditioning |
| `LPIPS_WEIGHT` | `0.5` | per-quadrant LPIPS weight (paper main; `0.0` = *_nolpips ablation) |
| `DEVICES` | `2` | GPU count or `"0,1"` |
| `WANDB_MODE` | `disabled` | `online` to log to W&B |
| `WANDB_RUN_NAME` | _(none)_ | Optional W&B run name |
| `NUM_WORKERS` | `16` | DataLoader workers |
| `OUTPUT_DIR` | `./output_sr_new` | checkpoint output dir |
| `RESUME_CKPT` | _(none)_ | Lightning ckpt to resume |

Extra `main.py` flags forwarded after `./run_sr.sh`:

| Flag | Default | Purpose |
|------|---------|---------|
| `--batch_size` | `32` | Per-device batch size |
| `--lr` | `5e-5` | AdamW LR |
| `--accumulate_grad_batches` | `4` | Grad accumulation |
| `--gradient_clip_val` | `1.0` | Norm clip |
| `--precision` | `bf16-mixed` | Trainer precision |
| `--zl_easy_max` | `15` | Parent-ZL cutoff for "easy" bucket |
| `--cfg` | `1.0` | Inference CFG scale |
| `--num_sampling_steps` | `50` | Euler steps |
| `--bicubic_init` / `--bicubic_init_t_start` | _(off)_ / `0.5` | SDEdit warm start (inference only) |

---

## Files

| File | Description |
|------|-------------|
| `model_sr.py` | `SR_JiT` — ViT backbone + `FinalLayer(p=32)` + `unpatchify` |
| `engine.py` | `DenoiserSR` (masked velocity loss, per-tile crop LPIPS, `--bicubic_init` SDEdit), `LitDenoiserSR` (`ema_context()` for swap-in EMA at val/gen), `ZLCurriculumSampler`, `SuperResDataModule` |
| `dataloader.py` | `HFParentChildDataset(require_complete_quad=...)` — partial quads in train, full quads in eval |
| `main.py` | CLI; auto-computes curriculum boundaries |
| `utils.py` | `compute_stage_boundaries()`, `StepCheckpointCallback` |
| `probe_dataset.py` | Visualize parent + child mosaic |
| `run_sr.sh` | Launch script |

**Data prep** (one-time, `src/dataprep/build_quad_splits_hf.py`):
`git10m_quad_meta.json` (per-tile metadata + seeded-shuffled, ZL-balanced
`splits`) and `hierarchy_index_quad.pkl` (parent→child lookup).
Both mirrored in the Google Drive release folder under `shared/`:
<https://drive.google.com/drive/folders/1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA>
