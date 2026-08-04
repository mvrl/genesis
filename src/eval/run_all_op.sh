#!/usr/bin/env bash
# Outpainting results — evaluate ALL cells (both archs × random/spatial × 4 mask regimes)
# with the paper-exact defaults from run_eval_op.sh (cfg 1.0, 50 steps, no-DINO, 8192/regime).
# Writes per-cell samples.csv + PNGs + FID under results/.
#   bash src/eval/run_all_op.sh
set -uo pipefail
cd "$(dirname "$0")"
for EXP in main_b16 main_h16; do
  MODEL=$([ "$EXP" = main_h16 ] && echo "JiT-H/16" || echo "JiT-B/16")
  for SPLIT in random_test spatial_test; do
    for R in quad3 quad2_adj quad2_diag quad1; do
      REGIME="$R" EXP="$EXP" MODEL="$MODEL" SPLIT="$SPLIT" OUT_DIR="results/op_${EXP}_${SPLIT}_${R}" \
        ./run_eval_op.sh
      python compute_fid_kid.py --root results --pattern "op_${EXP}_${SPLIT}_${R}"
    done
  done
done
python aggregate.py --task op --metric lpips --pivot exp --columns split,regime
echo "[run_all_op] done — per-cell results in src/eval/results/op_*"
