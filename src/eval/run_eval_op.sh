#!/usr/bin/env bash
# =============================================================================
# run_eval_op.sh — evaluate a Genesis outpainting ckpt on a Git-10M split.
#
# Evaluates a stage-3 OP ckpt on the Genesis-aligned 1-of-4-quadrants
# family of regimes. REGIME controls how many quadrants are known:
#   quad1 (1 known, 75% hole — default),
#   quad2_adj / quad2_diag (2 known, 50% hole), quad3 (3 known, 25% hole).
#
# Usage:
#   ./run_eval_op.sh                                  # defaults: stage 3, regime=quad1, random_test
#   REGIME=quad3 ./run_eval_op.sh                     # 25%-hole eval
#   EXP=main_h16 ./run_eval_op.sh                     # H16 main run
#   SPLIT=val_spatial ./run_eval_op.sh                # spatial val split
#   ./run_eval_op.sh --max_samples 32                 # smoke test
#   CKPT=/abs/path/to/x.ckpt ./run_eval_op.sh         # bypass --exp / --stage
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Hardcoded paths ──────────────────────────────────────────────────────────
META_FILE="${GENESIS_GIT10M_META:-}"   # empty → auto-pull from Google Drive
HF_CACHE="${HF_DATASETS_CACHE:-}"      # empty → HF default cache

# ── Configurable ─────────────────────────────────────────────────────────────
EXP="${EXP:-main_b16}"
STAGE="${STAGE:-3}"
MODEL="${MODEL:-JiT-B/16}"
CKPT="${CKPT:-}"

SPLIT="${SPLIT:-random_test}"
REGIME="${REGIME:-quad1}"         # quad1 / quad2_adj / quad2_diag / quad3

CFG="${CFG:-1.0}"
STEPS="${STEPS:-50}"

MAX_SAMPLES="${MAX_SAMPLES:-8192}"          # paper Table 3: 8192 per split/regime (0 = full split)
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# Device: GPU is strongly recommended (CPU works but is far slower). If DEVICE is
# unset we auto-detect CUDA and fall back to CPU; override with DEVICE=cuda/cpu.
if [[ -z "${DEVICE:-}" ]]; then
  if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
    echo "[info] No CUDA GPU detected — running on CPU (much slower). Set DEVICE=cuda to force." >&2
  fi
fi
SEED="${SEED:-42}"

METRICS="${METRICS:-psnr,ssim,lpips,dists,boundary_diff}"
LPIPS_MAX_SIZE="${LPIPS_MAX_SIZE:-0}"
DINO_WEIGHTS="${DINO_WEIGHTS:-}"
NO_DINO="${NO_DINO:-1}"                    # main OP ckpts are no-DINO; set 0 for any DINO ckpt

OUT_DIR="${OUT_DIR:-}"

# ── Build args ───────────────────────────────────────────────────────────────
ARGS=(
  --split "$SPLIT"
  --regime "$REGIME"
  --model "$MODEL"
  --cfg "$CFG"
  --num_sampling_steps "$STEPS"
  --max_samples "$MAX_SAMPLES"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  --metrics "$METRICS"
  --lpips_max_size "$LPIPS_MAX_SIZE"
)

# Only override if set; otherwise eval_op.py auto-pulls the Git-10M meta from the
# Google Drive release and HF fetches lcybuaa/Git-10M into its default cache.
[[ -n "$META_FILE" ]]    && ARGS+=(--meta_file_path "$META_FILE")
[[ -n "$HF_CACHE" ]]     && ARGS+=(--hf_data_cache "$HF_CACHE")
[[ -n "$CKPT" ]]         && ARGS+=(--ckpt  "$CKPT")
[[ -z "$CKPT" ]]         && ARGS+=(--exp   "$EXP" --stage "$STAGE")
[[ -n "$DINO_WEIGHTS" ]] && ARGS+=(--dino_weights "$DINO_WEIGHTS")
[[ "$NO_DINO" == "1" ]]  && ARGS+=(--no_dino)
[[ -n "$OUT_DIR" ]]      && ARGS+=(--out_dir "$OUT_DIR")

echo "============================================================"
echo "  OP eval"
echo "============================================================"
echo "  EXP/STAGE : ${CKPT:-$EXP / stage $STAGE}"
echo "  SPLIT     : $SPLIT"
echo "  REGIME    : $REGIME"
echo "  CFG       : $CFG"
echo "  STEPS     : $STEPS"
echo "  METRICS   : $METRICS"
echo "  DEVICE    : $DEVICE"
echo "============================================================"

cd "$ROOT"
python eval_op.py "${ARGS[@]}" "$@"
