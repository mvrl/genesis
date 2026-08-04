#!/usr/bin/env bash
# Super-resolution results — evaluate ALL cells (both archs × random/spatial × 3 child-zoom
# groups) with the paper-exact defaults from run_eval_sr.sh (×2, cfg 1.0, 50 steps, DINO on,
# no bicubic-init, 4096-per-group cap). Writes per-cell samples.csv + PNGs + FID under results/.
#   bash src/eval/run_all_sr.sh
set -uo pipefail
cd "$(dirname "$0")"
for EXP in main_b16 main_h16; do
  MODEL=$([ "$EXP" = main_h16 ] && echo "JiT-H/16" || echo "JiT-B/16")
  for SPLIT in random_test spatial_test; do
    for G in z12-15 z16-17 z18; do
      EXP="$EXP" MODEL="$MODEL" SPLIT="$SPLIT" OUT_DIR="results/sr_${EXP}_${SPLIT}_${G}" \
        ./run_eval_sr.sh --child_zl_group "$G"
      python compute_fid_kid.py --root results --pattern "sr_${EXP}_${SPLIT}_${G}"
    done
  done
done
python aggregate.py --task sr --metric lpips --pivot exp --columns split,zl_bucket
echo "[run_all_sr] done — per-cell results in src/eval/results/sr_*"
