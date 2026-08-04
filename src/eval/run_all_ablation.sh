#!/usr/bin/env bash
# SR ablation results — the four DINOv3 × LPIPS cells via run_eval_sr_ablation.sh (val_spatial,
# bicubic-init, JiT-B/16, 5000 samples, 500k ablation ckpts). Writes results/abl_* + FID.
#   bash src/eval/run_all_ablation.sh
set -uo pipefail
cd "$(dirname "$0")"
for C in nodino_nolpips dino_nolpips nodino_lpips dino_lpips; do
  CELL="$C" OUT_DIR="results/abl_$C" ./run_eval_sr_ablation.sh
  python compute_fid_kid.py --root results --pattern "abl_$C"
done
python aggregate.py --task sr --metric lpips --pivot exp --columns split
echo "[run_all_ablation] done — per-cell results in src/eval/results/abl_*"
