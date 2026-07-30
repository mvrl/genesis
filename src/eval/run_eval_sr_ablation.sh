#!/usr/bin/env bash
# =============================================================================
# run_eval_sr_ablation.sh — SR ablation eval (paper Table 5): DINOv3 × LPIPS,
# on val_spatial under the bicubic-init setting, JiT-B/16, 5000 samples, 500k ckpts.
#
# Thin wrapper over run_eval_sr.sh (same eval engine) with the Table-5 config:
# picks the ablation checkpoint from CELL, sets bicubic-init on, DINO off for the
# nodino_* cells, val_spatial, and 5000 samples. Override any of it via env vars.
#
# Usage:
#   CELL=dino_lpips ./run_eval_sr_ablation.sh
#   #   CELL ∈ { nodino_nolpips, dino_nolpips, nodino_lpips, dino_lpips }
#   MAX_SAMPLES=200 CELL=dino_lpips ./run_eval_sr_ablation.sh      # smoke test
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

CELL="${CELL:-dino_lpips}"
case "$CELL" in
  nodino_nolpips|dino_nolpips|nodino_lpips|dino_lpips) ;;
  *) echo "ERROR: CELL must be one of {nodino_nolpips,dino_nolpips,nodino_lpips,dino_lpips} (got '$CELL')" >&2; exit 2 ;;
esac
ND=0; case "$CELL" in nodino*) ND=1 ;; esac   # nodino_* ckpts were trained without DINO

# Delegate to the shared engine with the Table-5 configuration. Other env vars
# (OUT_DIR, DEVICE, BATCH_SIZE, HF cache, …) are inherited and pass through.
EXP="abl_${CELL}" \
MODEL="${MODEL:-JiT-B/16}" \
STAGE="${STAGE:-3}" \
SPLIT="${SPLIT:-val_spatial}" \
BICUBIC_INIT="${BICUBIC_INIT:-1}" \
NO_DINO="$ND" \
MAX_SAMPLES="${MAX_SAMPLES:-5000}" \
exec "$ROOT/run_eval_sr.sh" "$@"
