# Genesis — Experiments & Hyperparameters

A single reproducibility reference for **every training and evaluation experiment**.
Each value is cross-checked against the **paper** (SIGSPATIAL'26) and the **code**
(defaults in `src/…`; copy-paste reproduction loops in [`EVALUATION.md`](EVALUATION.md)).
Where the paper abstracts a value (e.g. "batch size 256"), the concrete code decomposition
is given. Numbers here are the *paper-faithful defaults* — running the referenced scripts
with no overrides reproduces the paper.

> Model note: the backbone key in code is `JiT-B/16` or `JiT-H/16` for both operators; the
> per-operator **patch size** is **32 (SR)** and **16 (OP)** (per the paper's training setup).

---

## 1. Data splits

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
🤗 `lcybuaa/Git-10M`. **Source:** paper (data-splits appendix); `src/dataprep/`.

---

## 2. Training hyperparameters

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
| Conditioning dropout (pₐᵣₒₚ) | 0.1 | 0.1 | (paper, operator setup) |
| DINOv3-sat conditioning | **on** (frozen ViT-L) | **off** | SR `NO_DINO=0` · OP `NO_DINO=1` |
| Aux per-quadrant LPIPS λ_perc | **0.5** | — | SR `--lpips_weight 0.5` |
| Time-weighting | — | **off** | OP `--no-enable_time_weight` |

**Source:** paper (training hyperparameters + operator setup); code `src/superresolution/main.py`,
`src/outpainting/main.py`, launchers `src/superresolution/run_sr.sh`,
`src/outpainting/run_op.sh`.

### 2.1 Curriculum stages

| Stage (step %) | SR (over parent zoom; coarse = parent ≤ z15) | OP (over mask) |
|---|---|---|
| i (0–20%) | coarse-only | quad3 — 3 known quadrants, 25% hole |
| ii (20–40%) | 50/50 coarse/fine | quad2 — 2 known (adj or diag), 50% |
| iii (40–60%) | uniform over all zoom levels | quad1 — 1 known, 75% |
| iv (60–100%) | complete quads only | uniform mixture of all regimes |

---

## 3. Inference sampler (per-operator eval)

| Setting | Value | Note |
|---|---|---|
| Solver | Euler | flow-matching ODE |
| Sampling steps | **50** | `--num_sampling_steps 50` |
| Guidance scale (CFG) | **1.0** | `--cfg 1.0` |
| SR bicubic warm-start (τ) | **context-dependent** | see per-table rows below |

**Bicubic warm-start (SDEdit) — where each τ applies:**
- **Main SR eval: OFF** — sampled from pure noise (the main-SR launcher passes no
  `--bicubic_init`; our reproduction matched the paper this way).
- **SR ablation: ON, τ = 0.5** — the paper's "bicubic-init SR setting".
- **Pyramid SR pass (dense500): ON, τ = 0.4** — MultiDiffusion warm-start over the mosaic.

---

## 4. Evaluation sample counts (N per cell)

Realized N for **every evaluation cell** across the results tables (all except the mean-FID
summary, which is derived from the SR + OP FIDs). **FID is the only count-sensitive metric** —
these counts are exactly what make FID reproduce the paper; PSNR / SSIM / LPIPS / DISTS / B-diff
and the dense500 semantic / consistency metrics are count-independent averages.

| Experiment | Split | Sub-group | N per cell |
|---|---|---|---|
| **Super-resolution** | random_test | child z12-15 | **2,470** |
| | random_test | child z16-17 | **2,003** |
| | random_test | child z18 | **4,096** (cap) |
| | spatial_test | child z12-15 | **3,146** |
| | spatial_test | child z16-17 | **2,087** |
| | spatial_test | child z18 | **4,096** (cap) |
| **Outpainting** | random_test & spatial_test | quad3 / quad2_adj / quad2_diag / quad1 | **8,192** each |
| **dense500** | 500-site benchmark | protocol A / B / C | **500 sites** each |
| **SR ablation** | val_spatial | 4 DINOv3×LPIPS cells | **5,000** each |

**Totals** — SR: **8,569** (random) / **9,329** (spatial) · OP: **32,768** per split (4 × 8,192) ·
dense500: **1,500** pyramids/arch (500 sites × 3 protocols) · ablation: **20,000** (4 × 5,000).

**SR is a per-group cap:** 4,096 is a *cap*, not the realized N — z12-15 / z16-17 fall under it
(run in full), z18 is capped to 4,096. **Ready-to-use table headings** — SR:
`z12-15 (N=2470 random / 3146 spatial)`, `z16-17 (N=2003 random / 2087 spatial)`,
`z18 (N=4096 for both)`; OP: `(N=8192)`.

---

## 5. Super-resolution eval

Reproduces per child-zoom group × split. **Run:** `src/eval/run_eval_sr.sh` (per cell); full-table loop in [`EVALUATION.md`](EVALUATION.md).

| Setting | Value | Flag |
|---|---|---|
| Checkpoint | main, 800k, stage 3 | `--exp main_b16 / main_h16 --stage 3` |
| Model | JiT-B/16, JiT-H/16 | `--model` |
| Split | random_test **and** spatial_test | `--split` |
| Scale | ×2 (parent→child) | `--scale 1` |
| CFG / steps | 1.0 / 50 | `--cfg 1.0 --num_sampling_steps 50` |
| Bicubic-init | **off** | (no `--bicubic_init`) |
| DINOv3-sat | on | (default) |
| Samples | **4096 per child-zoom group** (cap) | `--max_samples 4096 --child_zl_group {z12-15,z16-17,z18}` |
| LPIPS max size | 1024 | `--lpips_max_size 1024` |
| Seed | 42 | `--seed 42` |
| Metrics | PSNR, SSIM, LPIPS, DISTS, **FID** | — |

Realized N per group × split is in the consolidated **Evaluation sample counts** table (§4);
`--max_samples 4096` is a per-group cap (z12-15 / z16-17 run in full, z18 → 4096). FID is the only
count-sensitive metric, so this cap is what makes FID match the paper. **Source:** paper
(super-resolution results + evaluation protocol).

---

## 6. Outpainting eval

**Run:** `src/eval/run_eval_op.sh` (per cell); full-table loop in [`EVALUATION.md`](EVALUATION.md).

| Setting | Value | Flag |
|---|---|---|
| Checkpoint | main, 800k, stage 3 | `--exp main_b16 / main_h16 --stage 3` |
| Model | JiT-B/16, JiT-H/16 | `--model` |
| Split | random_test **and** spatial_test | `--split` |
| Mask regimes | quad3, quad2_adj, quad2_diag, quad1 | `--regime` |
| CFG / steps | 1.0 / 50 | `--cfg 1.0 --num_sampling_steps 50` |
| DINOv3-sat | **off** (OP is no-DINO) | `--no_dino` |
| Samples | **8192 per split/regime** | `--max_samples 8192` |
| LPIPS max size | 0 (full-res, 256²) | `--lpips_max_size 0` |
| Seed | 42 | `--seed 42` |
| Metrics | LPIPS, DISTS, B-diff, **FID** | hole-aware (synthesized region only) |

B-diff = mean abs pixel difference in a 4-px ring inside the hole boundary.

Realized N per cell = 8192 per split × regime (see the consolidated **Evaluation sample counts**
table, §4). As in SR, FID is the only count-sensitive metric — 8192 is what makes FID match the
paper (full ~50k gives a lower, less-biased FID). **Source:** paper (outpainting results; the
8192 cap is what the OP eval launcher used to produce the paper table). In this repo it is the
default `MAX_SAMPLES` in `src/eval/run_eval_op.sh`.

---

## 7. SR ablation (DINOv3 × LPIPS)

**Run:** `src/eval/run_eval_sr_ablation.sh` (per cell); all four cells in [`EVALUATION.md`](EVALUATION.md).

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
trained to **500k** (not 800k). **Source:** paper (SR ablation).

---

## 8. dense500 multi-scale pyramid completion

Benchmark **dense500**: 500 sites, depth-4 quadtree (z11→z14), 85 tiles/site (🤗 `dcher95/dense500`).
**Run:** `python -m pyramid.run_pyramid_eval` (generate) → `eval_pyramid.py` / `eval_ext.py` /
`caption_agree.py` (score); full loop in [`EVALUATION.md`](EVALUATION.md).

### 8.1 Seed protocols
| Protocol | Seeds | Tests |
|---|---|---|
| A (hardest) | 1 seed, level drawn ∝ tile count | full-pyramid from a single anchor |
| B | 3 independent seeds, one per non-root level | multi-scale reconciliation |
| C (easiest) | 4 spatially independent (2 leaf + 1 D-1 + 1 D-2) | dense leaf anchors under coarse guidance |

### 8.2 Generation config (dense500 pyramid campaign)

These are the paper's dense500 generation settings; in this repo they are the defaults of
`src/pyramid/run_pyramid_eval.py` (driven by `src/eval/run_all_dense500.sh`).

| Setting | Value | Flag |
|---|---|---|
| Checkpoints | SR + OP main, 800k | `--arch JiT-{B,H}/16` |
| Sampling steps | **50** | `--steps 50` |
| CFG | 2.5 | `--cfg 2.5` |
| Unified SR overlap | 32 px | `--uni_overlap 32` |
| Unified SR warm-start τ | 0.40 | `--uni_t 0.40` |
| Coverage | 100 sites × 5 windows × 3 protocols = **1500 / arch** | `--sites 0:100 --window 1..5 --protocol A/B/C` |

### 8.3 Pyramid engine
Offset `r=1` (working level = penultimate); SR = MultiDiffusion over the stitched mosaic,
warm-started at τ=0.4 (256² windows, 32-px overlap, plateau blending); max-coverage OP fills
2×2 quadrant blocks most-known-first; seam repair on internal leaf boundaries worst-first
(discontinuity **threshold 14**, up to **64** boundaries, 64-px masked band, soft-edge blend);
reconcile = bottom-up downsample only where seam-repair changed children.

### 8.4 Metrics
| Group | Metrics | Backend |
|---|---|---|
| Image quality | PSNR, SSIM, LPIPS, FID | torchmetrics + torch-fidelity |
| Semantic | CLIP-I, DINOv3-sat, Caption | open_clip (ViT-B-32) · DINOv3-sat ViT-L · Qwen2.5-VL-3B-Instruct + Sentence-BERT (all-MiniLM-L6-v2) |
| Consistency | LR-PSNRbox, xLR-PSNR, RAPSD | numpy (box downsample / cross-scale / radial PSD) |

Caption uses `--sites_per_cell 10`. Full metric definitions (with formulas) are in
[`METRICS_DENSE500.md`](METRICS_DENSE500.md). **Source:** paper (dense500 pyramid results + metric-details appendix).

---

## 9. Mean-FID summary

Mean FID over all cells: **6 SR cells** (2 splits × 3 zoom groups) and **8 OP cells**
(2 splits × 4 regimes), per model. Derived from the SR and OP results — no separate run.
Paper: SR Genesis-H 12.64 / Genesis-B 14.43; OP Genesis-H 9.12. **Source:** paper (mean-FID summary).

---

## 10. Checkpoints (Hugging Face release [`MVRL/genesis`](https://huggingface.co/MVRL/genesis))

| Experiment | Path under release | Steps |
|---|---|---|
| SR main | `main/superresolution/main_{B16,H16}/sr-full-tile-stage3-step0800000.ckpt` | 800k (DINO + LPIPS 0.5) |
| OP main | `main/outpainting/main_{B16,H16}/op-new-stage3-step0800000.ckpt` | 800k (no-DINO, no-TW) |
| SR ablation | `ablations/abl_{nodino_nolpips,dino_nolpips,nodino_lpips,dino_lpips}/sr-full-tile-stage3-step0500000.ckpt` | 500k |
| DINOv3-sat weights | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | — |
| Eval metadata | `shared/git10m_quad_meta.json`, `shared/hierarchy_index_quad.pkl` | — |

Auto-downloaded to `$GENESIS_CKPT_BASE` (default `<repo>/logs/ckpts`) via `huggingface_hub` — no authentication needed.

---

## 11. Quick reproduction map

Copy-paste commands for every table: [`EVALUATION.md`](EVALUATION.md).

| Experiment | What | Script (defaults reproduce it) |
|---|---|---|
| Super-resolution | PSNR/SSIM/LPIPS/DISTS/FID | `src/eval/run_eval_sr.sh` (per zoom-group) |
| Outpainting | LPIPS/DISTS/B-diff/FID | `src/eval/run_eval_op.sh` (per regime) |
| dense500 pyramid | image quality + semantic + consistency | `pyramid.run_pyramid_eval` + `eval_{pyramid,ext}.py` + `caption_agree.py` |
| SR ablation | DINOv3 × LPIPS | `src/eval/run_eval_sr_ablation.sh` |
| Mean FID | mean over SR + OP cells | emitted by the SR & OP runs |
| Training | hyperparameters | `src/superresolution/run_sr.sh`, `src/outpainting/run_op.sh` |
