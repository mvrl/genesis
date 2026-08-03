# Genesis Evaluation

Reproduces the **Genesis (JiT-B/16 and JiT-H/16) numbers** in the paper:

| Experiment | What it reports | Section |
|---|---|---|
| Super-resolution | PSNR/SSIM/LPIPS/DISTS/FID per child-zoom group, 2 splits | [SR](#super-resolution) |
| Outpainting | LPIPS/DISTS/B-diff/FID per mask regime, 2 splits | [OP](#outpainting) |
| dense500 pyramid | image quality + semantic + consistency | [Pyramid](#dense500-pyramid-completion) |
| SR ablation | DINOv3 × LPIPS on `val_spatial` | [Ablation](#sr-ablation) |
| Mean FID | mean across all SR / OP cells | [Mean FID](#mean-fid) |

> Comparison **baselines are not included** — this repo evaluates our checkpoints only.
> Metric definitions and per-metric implementation notes are in [`docs/METRICS.md`](../../docs/METRICS.md).
> Every training/eval hyperparameter (paper vs. code, per experiment) is tabulated in
> [`docs/HYPERPARAMETERS.md`](../../docs/HYPERPARAMETERS.md).

## Setup

```bash
uv sync && source .venv/bin/activate      # from the repo root (installs pinned deps)
```

- **Checkpoints** auto-download on first use from the public Google Drive release
  (folder `1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA`) via an rclone `gdrive:` remote into
  `$GENESIS_CKPT_BASE` (default `<repo>/logs/ckpts/`). `--exp main_b16 | main_h16` and the
  `abl_*` ablation cells all resolve automatically. DINOv3-sat weights resolve the same way
  (or set `$GENESIS_DINO_WEIGHTS`).
- **Git-10M** eval needs the quad-split metadata + hierarchy index + HF cache, configured by
  `$GENESIS_GIT10M_META` / `$GENESIS_HINDEX` / `$HF_DATASETS_CACHE` (repo-relative `data/`
  fallbacks). Set them to where you keep the files, or pass
  `--meta_file_path/--hindex_pkl/--hf_data_cache`.
- **dense500** auto-downloads from the HF dataset `dcher95/dense500` on first use (no manual step).
- Outputs default to `src/eval/results/` (override with `$GENESIS_RESULTS_ROOT` or `--out_dir`).

Run everything from `src/eval/`:

```bash
cd src/eval
```

---

## Super-resolution

`eval_sr.py` (via `run_eval_sr.sh`) scores a checkpoint on a Git-10M split per child-zoom group,
writing per-sample `samples.csv` **and** `predictions/`+`targets/` PNGs, then FID over those PNGs.
The paper caps each group at **4096 samples** (`run_eval_sr.sh` default `MAX_SAMPLES=4096`; z12-15 /
z16-17 fall under it, z18 → 4096). Loop the three groups for each model × split:

```bash
# JiT-H/16, random_test  (repeat with SPLIT=spatial_test, and MODEL=JiT-B/16 EXP=main_b16)
for G in z12-15 z16-17 z18; do
  EXP=main_h16 MODEL=JiT-H/16 SPLIT=random_test OUT_DIR=results/sr_h16_random_$G \
    ./run_eval_sr.sh --child_zl_group $G
  python compute_fid_kid.py --root results --pattern sr_h16_random_$G
done
```

Aggregate the reconstruction/perceptual columns into the Table-2 layout:

```bash
python aggregate.py --task sr --metric lpips --pivot exp --columns split,zl_bucket
# swap --metric psnr|ssim|dists to fill the other columns
```

`--scale {1,2,3,4}` selects ×2/×4/×8/×16 (SR uses single-step ×2 = `--scale 1`, the default).

---

## Outpainting

`eval_op.py` (via `run_eval_op.sh`) scores one mask regime at a time (hole-aware metrics + B-diff).
Loop the four regimes for each model × split:

```bash
for R in quad1 quad2_adj quad2_diag quad3; do
  REGIME=$R EXP=main_h16 MODEL=JiT-H/16 SPLIT=random_test \
    OUT_DIR=results/op_h16_${R}_random ./run_eval_op.sh
  python compute_fid_kid.py --root results --pattern op_h16_${R}_random
done
# repeat for EXP=main_b16 MODEL=JiT-B/16, and SPLIT=spatial_test
```

```bash
python aggregate.py --task op --metric lpips --pivot exp --columns split,regime
# swap --metric dists|boundary_diff for the other columns
```

---

## dense500 pyramid completion

> Metric definitions (image quality + semantic + consistency) are in
> [`docs/METRICS_DENSE500.md`](../../docs/METRICS_DENSE500.md).

**1. Generate** the full-pyramid completions for each seed protocol (A/B/C) × zoom window (1–5),
100 sites each, for both backbones. Settings match the paper's engine configuration
(offset r=1, MultiDiffusion overlap 32, warm-start τ=0.4):

```bash
for P in A B C; do for W in 1 2 3 4 5; do
  PYTHONPATH=../.. python -m pyramid.run_pyramid_eval \
    --protocol $P --window $W --sites 0:100 \
    --arch JiT-H/16 --steps 50 --cfg 2.5 --uni_overlap 32 --uni_t 0.40 \
    --out_root outputs/dense500_eval_h16
done; done
# repeat with --arch JiT-B/16 --out_root outputs/dense500_eval_b16
```

**2. Score** (image quality + FID, then semantic + consistency, then caption). Run per gen root:

```bash
PYTHONPATH=../.. python eval_pyramid.py  --gen_root outputs/dense500_eval_h16 --device cuda
PYTHONPATH=../.. python eval_ext.py      --gen_root outputs/dense500_eval_h16 --families consistency,semantic --device cuda
PYTHONPATH=../.. python caption_agree.py --gen_root outputs/dense500_eval_h16 --sites_per_cell 10 --device cuda
# repeat for outputs/dense500_eval_b16
```

The Table-4 numbers are written per gen root as JSON:
`<gen_root>/_metrics/aggregate.json` (PSNR/SSIM/LPIPS/FID),
`<gen_root>/_metrics_ext/aggregate_ext_*.json` (CLIP-I/DINOv3-sat, LR-PSNR_box/xLR-PSNR/RAPSD),
and `<gen_root>/_metrics_ext/aggregate_caption.json` (Caption), each stratified
overall / by protocol / by window / by level.

Seed protocols A/B/C are defined deterministically in [`pyramid.dataset`](../pyramid/dataset.py)
(seeded RNG), so a given `(protocol, window, site)` reproduces the same seeds without any extra
files.

---

## SR ablation

The four ablation checkpoints (DINOv3 × LPIPS) evaluated on `val_spatial` under the bicubic-init
SR setting (500k ablation ckpts). `run_eval_sr_ablation.sh` sets the whole Table-5 config from
`CELL` (bicubic-init on, `val_spatial`, 5000 samples, JiT-B/16, DINO off for the `nodino_*` cells):

```bash
for C in nodino_nolpips dino_nolpips nodino_lpips dino_lpips; do
  CELL=$C OUT_DIR=results/abl_$C ./run_eval_sr_ablation.sh
  python compute_fid_kid.py --root results --pattern abl_$C
done
python aggregate.py --task sr --metric lpips --pivot exp --columns split
```

---

## Mean FID

Mean of the per-cell FID values: the 6 SR cells (2 splits × 3 child-zoom groups) and the 8 OP
cells (2 splits × 4 regimes), read from the `fid_kid.json` files produced above. Compute the SR
and OP means from those values (e.g. with `aggregate.py --metric fid` once FID is merged into the
per-sample tables, or directly from the `fid_kid.json` outputs).

---

## Files

`common.py` (checkpoint/DINO resolution, model loading, `compute_metrics`, `SampleWriter`) ·
`eval_sr.py` · `eval_op.py` · `eval_pyramid.py` · `eval_ext.py` + `metrics_ext.py` ·
`caption_agree.py` · `compute_fid_kid.py` · `aggregate.py` · wrappers `run_eval_sr.sh` (SR) /
`run_eval_sr_ablation.sh` (SR ablation) / `run_eval_op.sh` (OP).
