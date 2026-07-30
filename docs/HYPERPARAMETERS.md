# Genesis — Experiments & Hyperparameters

A single reproducibility reference for **every training and evaluation experiment**.
Each value is cross-checked against the **paper** (SIGSPATIAL'26; Tables 7/8, §5.1–5.5,
Appendix A) and the **code** (defaults in `src/…` and the `logs/eval_table*.sh` runners).
Where the paper abstracts a value (e.g. "batch size 256"), the concrete code decomposition
is given. Numbers here are the *paper-faithful defaults* — running the referenced scripts
with no overrides reproduces the paper.

> Model note: the backbone key in code is `JiT-B/16` or `JiT-H/16` for both operators; the
> per-operator **patch size** is **32 (SR)** and **16 (OP)** (paper §5.1).

---

## 1. Data splits — paper Table 7

Complete quads = parents whose four `z+1` children all exist. Training additionally uses
partial quads (1–3 children) for the masked losses.

| Split | Complete quads | Tiles |
|---|---|---|
| Train | 362,572 | 6,568,890 |
| Test (spatial) | 10,000 | 49,825 |
| Test (random) | 10,000 | 49,826 |
| Val (spatial) | 5,000 | 24,958 |
| Val (random) | 5,000 | 24,969 |

Split index lists live in `git10m_quad_meta.json` (`splits` key). Tiles come from
🤗 `lcybuaa/Git-10M`. **Source:** paper §A.1; `src/dataprep/`.

---

## 2. Training hyperparameters — paper Table 8

Global batch **256** = `batch_size 32 × accumulate_grad_batches 4 × devices 2`.
Curriculum boundaries at **20/40/60%** of the step budget → **160k / 320k / 480k** of 800k
(rounded to 5k).

| Hyperparameter | SR | OP | Code (default) |
|---|---|---|---|
| Backbone / patch size | JiT-B or JiT-H / **32** | JiT-B or JiT-H / **16** | `--model JiT-{B,H}/16` |
| Optimizer | AdamW | AdamW | — |
| (β₁, β₂) | (0.9, 0.999) | (0.9, 0.95) | SR: AdamW default · OP: explicit |
| Learning rate | 5×10⁻⁵ | 1×10⁻⁵ | `--lr` (SR `5e-5`, OP `1e-5`) |
| Batch size (global) | 256 | 256 | `32 × 4 × 2` |
| ‑ per-GPU batch | 32 | 32 | `--batch_size 32` |
| ‑ grad accumulation | 4 | 4 | `--accumulate_grad_batches 4` |
| ‑ GPUs | 2 | 2 | `--devices 2` |
| Gradient clip | 1.0 | 0.5 | `--gradient_clip_val` |
| Precision | bf16-mixed | bf16-mixed | `--precision bf16-mixed` |
| Total steps | 800k | 800k | `--max_steps 800000` |
| EMA decays | 0.9999, 0.9996 | 0.9999, 0.9996 | `--ema_decay1/2` |
| Noise schedule (P_mean, P_std) | (−0.8, 0.8) | (−0.8, 0.8) | `--P_mean -0.8 --P_std 0.8` (logit-normal) |
| Noise scale | 1.0 | 1.0 | `--noise_scale 1.0` |
| Curriculum boundaries | 20/40/60% | 20/40/60% | auto-computed from `max_steps` |
| Conditioning dropout (pₐᵣₒₚ) | 0.1 | 0.1 | (paper §5.2) |
| DINOv3-sat conditioning | **on** (frozen ViT-L) | **off** | SR `NO_DINO=0` · OP `NO_DINO=1` |
| Aux per-quadrant LPIPS λ_perc | **0.5** | — | SR `--lpips_weight 0.5` |
| Time-weighting | — | **off** | OP `--no-enable_time_weight` |

**Source:** paper Table 8 + §5.1–5.2; code `src/superresolution/main.py`,
`src/outpainting/main.py`, launchers `src/superresolution/run_sr.sh`,
`src/outpainting/run_op.sh`.

### 2.1 Curriculum stages (paper §5.1)

| Stage (step %) | SR (over parent zoom; coarse = parent ≤ z15) | OP (over mask) |
|---|---|---|
| i (0–20%) | coarse-only | quad3 — 3 known quadrants, 25% hole |
| ii (20–40%) | 50/50 coarse/fine | quad2 — 2 known (adj or diag), 50% |
| iii (40–60%) | uniform over all zoom levels | quad1 — 1 known, 75% |
| iv (60–100%) | complete quads only | uniform mixture of all regimes |

---

## 3. Inference sampler (per-operator eval) — paper §5.1–5.2

| Setting | Value | Note |
|---|---|---|
| Solver | Euler | flow-matching ODE |
| Sampling steps | **50** | `--num_sampling_steps 50` |
| Guidance scale (CFG) | **1.0** | `--cfg 1.0` |
| SR bicubic warm-start (τ) | **context-dependent** | see per-table rows below |

**Bicubic warm-start (SDEdit) — where each τ applies:**
- **Table 2 (main SR eval): OFF** — sampled from pure noise (the Table-2 launcher passes no
  `--bicubic_init`; our reproduction matched the paper this way).
- **Table 5 (SR ablation): ON, τ = 0.5** — the "bicubic-init SR setting" (paper §6.1, §5.2).
- **Table 4 (pyramid SR pass): ON, τ = 0.4** — MultiDiffusion warm-start over the mosaic (§5.4).

---

## 4. Table 2 — Super-resolution eval

Reproduces per child-zoom group × split. **Run:** `src/eval/run_eval_sr.sh` (per cell) /
`logs/eval_table2_sr.sh` (full table).

| Setting | Value | Flag |
|---|---|---|
| Checkpoint | main, 800k, stage 3 | `--exp main_b16 / main_h16 --stage 3` |
| Model | JiT-B/16, JiT-H/16 | `--model` |
| Split | random_test **and** spatial_test | `--split` |
| Scale | ×2 (parent→child) | `--scale 1` |
| CFG / steps | 1.0 / 50 | `--cfg 1.0 --num_sampling_steps 50` |
| Bicubic-init | **off** | (no `--bicubic_init`) |
| DINOv3-sat | on | (default) |
| Samples | **full split**, per child-zoom group | `--max_samples 0 --child_zl_group {z12-15,z16-17,z18}` |
| LPIPS max size | 1024 | `--lpips_max_size 1024` |
| Seed | 42 | `--seed 42` |
| Metrics | PSNR, SSIM, LPIPS, DISTS, **FID** | — |

Child-zoom groups & paper N (per split): `z12–15` = 2470 / 3146, `z16–17` = 2003 / 2087,
`z18` = 4096 / 4096 (random / spatial). *Note:* the paper caps `z18` at 4096; running the
full split (default) uses all such tiles (~4.7–5.5k) — numbers are effectively identical.
FID needs the full set (small-N FID is positively biased). **Source:** paper Table 2, §5.5.

---

## 5. Table 3 — Outpainting eval

**Run:** `src/eval/run_eval_op.sh` (per cell) / `logs/eval_table3_op.sh` (full table).

| Setting | Value | Flag |
|---|---|---|
| Checkpoint | main, 800k, stage 3 | `--exp main_b16 / main_h16 --stage 3` |
| Model | JiT-B/16, JiT-H/16 | `--model` |
| Split | random_test **and** spatial_test | `--split` |
| Mask regimes | quad3, quad2_adj, quad2_diag, quad1 | `--regime` |
| CFG / steps | 1.0 / 50 | `--cfg 1.0 --num_sampling_steps 50` |
| DINOv3-sat | **off** (OP is no-DINO) | `--no_dino` |
| Samples | **full split** (~50k/regime) | `--max_samples 0` |
| LPIPS max size | 0 (full-res, 256²) | `--lpips_max_size 0` |
| Seed | 42 | `--seed 42` |
| Metrics | LPIPS, DISTS, B-diff, **FID** | hole-aware (synthesized region only) |

B-diff = mean abs pixel difference in a 4-px ring inside the hole boundary. **Source:**
paper Table 3, §5.5.

---

## 6. Table 5 — SR ablation (DINOv3 × LPIPS)

**Run:** `src/eval/run_eval_sr_ablation.sh` (per cell) / `logs/eval_table5_ablation.sh` (all 4).

| Setting | Value | Flag |
|---|---|---|
| Checkpoints | ablation, **500k**, stage 3 | `--exp abl_{nodino_nolpips,dino_nolpips,nodino_lpips,dino_lpips}` |
| Model | JiT-B/16 | `--model JiT-B/16` |
| Split | val_spatial | `--split val_spatial` |
| Samples | 5000 | `--max_samples 5000` |
| Bicubic-init | **on, τ = 0.5** | `--bicubic_init --bicubic_init_t_start 0.5` |
| DINOv3-sat | on for `dino_*`, off for `nodino_*` | `--no_dino` for `nodino_*` |
| CFG / steps | 1.0 / 50 | `--cfg 1.0 --num_sampling_steps 50` |
| Metrics | PSNR, SSIM, LPIPS, DISTS, FID | — |

The four cells toggle {DINOv3 on/off} × {LPIPS loss on/off}. Ablations are the one experiment
trained to **500k** (not 800k). **Source:** paper Table 5, §6.1.

---

## 7. Table 4 — dense500 multi-scale pyramid completion

Benchmark **dense500**: 500 sites, depth-4 quadtree (z11→z14), 85 tiles/site (🤗 `dcher95/dense500`).
**Run:** `python -m pyramid.run_pyramid_eval` (generate) → `eval_pyramid.py` / `eval_ext.py` /
`caption_agree.py` (score) / `logs/eval_table4_dense500.sh` (all).

### 7.1 Seed protocols (paper §5.5)
| Protocol | Seeds | Tests |
|---|---|---|
| A (hardest) | 1 seed, level drawn ∝ tile count | full-pyramid from a single anchor |
| B | 3 independent seeds, one per non-root level | multi-scale reconciliation |
| C (easiest) | 4 spatially independent (2 leaf + 1 D-1 + 1 D-2) | dense leaf anchors under coarse guidance |

### 7.2 Generation config (campaign `worker.sh`)
| Setting | Value | Flag |
|---|---|---|
| Checkpoints | SR + OP main, 800k | `--arch JiT-{B,H}/16` |
| Sampling steps | **50** | `--steps 50` |
| CFG | 2.5 | `--cfg 2.5` |
| Unified SR overlap | 32 px | `--uni_overlap 32` |
| Unified SR warm-start τ | 0.40 | `--uni_t 0.40` |
| Coverage | 100 sites × 5 windows × 3 protocols = **1500 / arch** | `--sites 0:100 --window 1..5 --protocol A/B/C` |

### 7.3 Pyramid engine (paper §5.4)
Offset `r=1` (working level = penultimate); SR = MultiDiffusion over the stitched mosaic,
warm-started at τ=0.4 (256² windows, 32-px overlap, plateau blending); max-coverage OP fills
2×2 quadrant blocks most-known-first; seam repair on internal leaf boundaries worst-first
(discontinuity **threshold 14**, up to **64** boundaries, 64-px masked band, soft-edge blend);
reconcile = bottom-up downsample only where seam-repair changed children.

### 7.4 Metrics (paper §5.5, A.3–A.5)
| Group | Metrics | Backend |
|---|---|---|
| Image quality | PSNR, SSIM, LPIPS, FID | torchmetrics + torch-fidelity |
| Semantic | CLIP-I, DINOv3-sat, Caption | open_clip (ViT-B-32) · DINOv3-sat ViT-L · Qwen2.5-VL-3B-Instruct + Sentence-BERT (all-MiniLM-L6-v2) |
| Consistency | LR-PSNRbox, xLR-PSNR, RAPSD | numpy (box downsample / cross-scale / radial PSD) |

Caption uses `--sites_per_cell 10`. **Source:** paper Table 4, §5.4–5.5, A.3–A.5.

---

## 8. Table 6 — mean FID summary

Mean FID over all cells: **6 SR cells** (2 splits × 3 zoom groups) and **8 OP cells**
(2 splits × 4 regimes), per model. Derived from Tables 2 and 3 — no separate run. Paper:
SR Genesis-H 12.64 / Genesis-B 14.43; OP Genesis-H 9.12. **Source:** paper Table 6.

---

## 9. Checkpoints (Google Drive release `1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA`)

| Experiment | Path under release | Steps |
|---|---|---|
| SR main | `main/superresolution/main_{B16,H16}/sr-full-tile-stage3-step0800000.ckpt` | 800k (DINO + LPIPS 0.5) |
| OP main | `main/outpainting/main_{B16,H16}/op-new-stage3-step0800000.ckpt` | 800k (no-DINO, no-TW) |
| SR ablation | `ablations/abl_{nodino_nolpips,dino_nolpips,nodino_lpips,dino_lpips}/sr-full-tile-stage3-step0500000.ckpt` | 500k |
| DINOv3-sat weights | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | — |
| Eval metadata | `shared/git10m_quad_meta.json`, `shared/hierarchy_index_quad.pkl` | — |

Auto-pulled to `$GENESIS_CKPT_BASE` (default `<repo>/logs/ckpts`) via an rclone `gdrive:` remote.

---

## 10. Quick reproduction map

| Paper table | What | Script (defaults reproduce it) |
|---|---|---|
| Table 2 | SR (PSNR/SSIM/LPIPS/DISTS/FID) | `logs/eval_table2_sr.sh` → `src/eval/run_eval_sr.sh` |
| Table 3 | OP (LPIPS/DISTS/B-diff/FID) | `logs/eval_table3_op.sh` → `src/eval/run_eval_op.sh` |
| Table 4 | dense500 pyramid | `logs/eval_table4_dense500.sh` → `pyramid.run_pyramid_eval` + `eval_{pyramid,ext}.py` + `caption_agree.py` |
| Table 5 | SR ablation | `logs/eval_table5_ablation.sh` → `src/eval/run_eval_sr_ablation.sh` |
| Table 6 | mean FID | emitted by the Table 2 & Table 3 runs |
| Table 8 | training | `src/superresolution/run_sr.sh`, `src/outpainting/run_op.sh` |
