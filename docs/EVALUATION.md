# Evaluation

End-to-end: one command per table reproduces the **Genesis (JiT-B/16 & JiT-H/16)** numbers in the
paper — super-resolution, outpainting, dense500 pyramid completion, the SR ablation, and the
mean-FID summary.

Every script runs **all cells** of its table with the **paper-exact defaults** (sample counts,
sampler steps, guidance) — nothing to configure. Per-cell N and every hyperparameter are in
[`HYPERPARAMETERS.md`](HYPERPARAMETERS.md); metric definitions in [`METRICS.md`](METRICS.md) (tile)
and [`METRICS_DENSE500.md`](METRICS_DENSE500.md) (pyramid).

## 1. Install
```bash
uv sync && source .venv/bin/activate        # from the repo root; installs pinned deps (torch cu128)
```

## 2. Download checkpoints (Google Drive)
Checkpoints, the DINOv3-sat encoder, and the split metadata live in a public Google Drive release
(folder `1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA`). Set up an rclone remote named `gdrive` once:
```bash
rclone config          # n(ew) -> name: gdrive -> storage: drive -> accept defaults -> OAuth in browser
```
Then pre-download everything (main + ablation ckpts, DINOv3, split meta) into `$GENESIS_CKPT_BASE`
(default `<repo>/logs/ckpts`):
```bash
bash src/eval/download_ckpts.sh                    # or: GENESIS_CKPT_BASE=/my/cache bash src/eval/download_ckpts.sh
```
> Pre-downloading is optional — the eval scripts auto-pull each file on first use — but it avoids
> parallel jobs racing on the same download. If you cache to a custom path, `export
> GENESIS_CKPT_BASE=/that/path` before step 4. The **Git-10M** and **dense500** datasets download
> automatically from the Hugging Face Hub (no manual step).

## 3. (optional) point at custom cache locations
```bash
export GENESIS_CKPT_BASE=/path/to/ckpt_cache       # where the Drive files live (default <repo>/logs/ckpts)
export HF_DATASETS_CACHE=/path/to/hf_cache         # optional; else HF default cache
```

## 4. Run — one command per table
| Paper table | Command |
|---|---|
| Super-resolution | `bash src/eval/run_all_sr.sh` |
| Outpainting | `bash src/eval/run_all_op.sh` |
| SR ablation | `bash src/eval/run_all_ablation.sh` |
| dense500 pyramid | `bash src/eval/run_all_dense500.sh` |

Mean FID needs no run — it is the mean of the per-cell FIDs produced by the SR and OP scripts.

Each script is a GPU-heavy sweep (all archs × splits × sub-groups), so run one per GPU. They are
**idempotent**: a cell whose output already exists is skipped, so a re-run resumes.

## 5. Where results land
- SR / OP / ablation -> `src/eval/results/<cell>/` — per-sample `samples.csv`, `predictions/` +
  `targets/` PNGs, `fid_kid.json`; each script prints an aggregated table at the end.
- dense500 -> `src/eval/outputs/dense500_{B16,H16}/_metrics*/` — `aggregate.json` (PSNR/SSIM/LPIPS/FID),
  `aggregate_ext_consistency-semantic.json` (CLIP-I / DINOv3-sat / LR-PSNR / xLR-PSNR / RAPSD),
  `aggregate_caption.json` (Caption), stratified by seed protocol.

## Running a single cell (optional)
The `run_all_*.sh` drivers just loop the per-cell wrappers with paper defaults; to score one cell:
```bash
cd src/eval
EXP=main_h16 MODEL=JiT-H/16 SPLIT=random_test ./run_eval_sr.sh --child_zl_group z18
CELL=dino_lpips ./run_eval_sr_ablation.sh
```
See [`../src/eval/README.md`](../src/eval/README.md) for per-cell knobs and metric internals.
