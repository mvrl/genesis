# Outpainting — Mask-Based Image Completion

Generates the masked region of a 256×256 satellite image conditioned on the
known region. SD-inpainting-style conditioning (`zt ‖ mask ‖ masked_image`,
7-channel). Target-tile zoom (GSD) is added to the AdaLN vector.

```bash
cd src/outpainting
source .venv/bin/activate
chmod +x run_op.sh
```

All paths (`git10m_quad_meta.json`, HF cache) are hardcoded at the top of
`run_op.sh` — edit there if your layout differs.

---

## Architecture

```
image 256×256, mask (1=hole, 0=known)
    masked_image = image · (1 - mask)
    noisy_hole   = t · image + (1-t) · noise
    zt           = masked_image + noisy_hole · mask        (noise only in hole)
    inpaint_cond = concat(zt, mask, masked_image)          (7 channels)

zt ──► x_embedder ─────────────────────────────────────► Q tokens
inpaint_cond ──► context_embedder (zero-init) ─────────► K, V (op_context)

JiTBlock × depth
    self-attn (RoPE)
    op_cross_attn       (Q ↔ op_context, masked to known patches only)
    adaLN(c = t_emb + zoom_emb)
FinalLayer → unpatchify → 256×256 output
result = output · mask + masked_image · (1 - mask)        (known region frozen)
```

- **`inpaint_cond`** — always present, zero-init `context_embedder`.
- **Zoom-level conditioning** (always on) — `index_to_meta['zoom']` of the
  current tile, sinusoidal-MLP embedded, added to AdaLN. `zoom=0` = "unknown
  GSD" sentinel (learned via `cond_drop_prob` dropout).

---

## Training

**Mask curriculum** — fully quadrant-aligned, monotonically harder by
known-count. Stage boundaries auto-computed at 20 / 40 / 60 % of
`--max_steps`, rounded to the nearest 5 k. Stage 3 consolidates on the
full eval distribution so the final ckpt has recently seen every regime.
End-step columns below give the exact step at which each stage
*finishes* for the main run (`MAX_STEPS=800000`):

| Stage | Regime sampled per step | Known | Hole fraction | End step (800k run) |
|-------|------------------------|-------|---------------|--------------------:|
| 0 | `quad3` | 3 of 4 | 25 % | 160,000 |
| 1 | 50/50 `quad2_adj` / `quad2_diag` | 2 of 4 | 50 % | 320,000 |
| 2 | `quad1` | 1 of 4 | 75 % | 480,000 |
| 3 | uniform mix of `quad1` / `quad2_adj` / `quad2_diag` / `quad3` | 1–3 | 25–75 % | 800,000 |

The schedule matches the eval-time regime distribution (see
[`src/eval/README.md`](../eval/README.md)) and aligns training with the
4-quadrant pyramid the model is architecturally built around.

**Loss** — per-sample masked velocity MSE (time weighting is disabled):

```
loss_per[b] = Σ((v_pred − v_tgt)² · mask) / Σ(mask)
loss        = mean(loss_per)
```

Non-finite loss → DDP-safe zero (`Σ 0·p.sum()` over trainable params).

**Checkpoints** — top-3 by `val/val_random_loss` + `last.ckpt`, plus
`StepCheckpointCallback` saving 10 uniform steps + each stage end (label
`uniform` / `stage{0..3}`). Resume with `RESUME_CKPT=...`.

**Validation splits** monitored during training:

| Split | Samples | Purpose |
|-------|---------|---------|
| `val_random`  | 24,969 | IID training-time monitor |
| `val_spatial` | 24,958 (50 % urban) | OOD generalization gap |

---

## Main run

OP training is locked at the quadrant mask curriculum above (all task
config lives in `run_op.sh`). The only user knobs are model size and
the usual run-management vars:

```bash
MODEL_SIZE=B16 MAX_STEPS=800000 OUTPUT_DIR=./runs/main_b16 ./run_op.sh
```

Swap `MODEL_SIZE=H16` for the H16 variant. Resume with
`RESUME_CKPT=./runs/.../last.ckpt`.

---

## Configuration

Env vars (consumed by `run_op.sh`):

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_SIZE` | `B16` | `B16` or `H16` |
| `MAX_STEPS` | `800000` | paper total; curriculum boundaries auto-computed at 20/40/60 % |
| `DEVICES` | `2` | GPU count or `"0,1"` |
| `WANDB_MODE` | `disabled` | `online` to enable W&B |
| `WANDB_RUN_NAME` | _(none)_ | Optional W&B run name |
| `NUM_WORKERS` | `16` | DataLoader workers |
| `OUTPUT_DIR` | `./output_op_new` | checkpoint output dir |
| `RESUME_CKPT` | _(none)_ | Lightning ckpt to resume |

Extra `main.py` flags forwarded after `./run_op.sh`:

| Flag | Default | Purpose |
|------|---------|---------|
| `--batch_size` | `32` | Per-device batch size |
| `--lr` | `1e-5` | AdamW LR (betas=(0.9, 0.95)) |
| `--accumulate_grad_batches` | `4` | Grad accumulation |
| `--gradient_clip_val` | `0.5` | Norm clip |
| `--precision` | `bf16-mixed` | Trainer precision |
| `--cfg` | `1.0` | Inference CFG scale |
| `--num_sampling_steps` | `50` | Euler steps |

---

## Files

| File | Description |
|------|-------------|
| `model_op.py` | `JiT_Outpainting` — ViT backbone with `op_cross_attn` + `parent_cross_attn` |
| `engine.py` | `DenoiserOutpainting`, `LitDenoiserOutpainting`, `OutpaintingDataModule` |
| `dataloader.py` | `Git10MOutpaintingDataset` — on-the-fly mask gen with curriculum |
| `main.py` | CLI; auto-computes curriculum boundaries |
| `utils.py` | `compute_stage_boundaries()`, `StepCheckpointCallback` |
| `run_op.sh` | Launch script |

**Data prep** — shared with SR (`src/dataprep/build_quad_splits_hf.py`).
`git10m_quad_meta.json` carries seeded-shuffled, ZL-balanced `splits` lists
keyed by HF index. Mirrored in the Hugging Face release under `shared/`:
<https://huggingface.co/MVRL/genesis>
