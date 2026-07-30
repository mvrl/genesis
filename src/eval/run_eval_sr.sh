#!/usr/bin/env bash
# =============================================================================
# run_eval_sr.sh — Super-resolution eval for the Genesis MAIN models (paper Table 2).
#
# Defaults reproduce a Table-2 cell: main_b16, random_test, ×2, cfg 1.0, 50 steps,
# DINO on, NO bicubic-init, full split. Override via env vars or extra `--…` flags.
# For the SR ablation (paper Table 5) use run_eval_sr_ablation.sh instead.
#
# Usage:
#   ./run_eval_sr.sh                                              # Table 2: main_b16, random_test
#   EXP=main_h16 MODEL=JiT-H/16 SPLIT=spatial_test ./run_eval_sr.sh
#   EXP=main_b16 ./run_eval_sr.sh --child_zl_group z18            # one child-zoom group
#   ./run_eval_sr.sh --max_samples 32                            # smoke test
#   CKPT=/abs/path/to/x.ckpt ./run_eval_sr.sh                    # bypass --exp / --stage
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Hardcoded paths (edit if your layout differs) ────────────────────────────
META_FILE="${GENESIS_GIT10M_META:-}"   # empty → auto-pull from Google Drive
HINDEX_PKL="${GENESIS_HINDEX:-}"       # empty → auto-pull from Google Drive
HF_CACHE="${HF_DATASETS_CACHE:-}"      # empty → HF default cache

# ── Configurable ─────────────────────────────────────────────────────────────
EXP="${EXP:-main_b16}"            # main_b16 / main_h16 (paper Table 2) or abl_* (Table 5)
STAGE="${STAGE:-3}"
MODEL="${MODEL:-JiT-B/16}"
CKPT="${CKPT:-}"                  # if set, --ckpt wins over --exp + --stage

SPLIT="${SPLIT:-random_test}"     # paper Table 2 test splits: random_test / spatial_test
SCALE="${SCALE:-1}"               # 1=×2, 2=×4, 3=×8, 4=×16

CFG="${CFG:-1.0}"
STEPS="${STEPS:-50}"
BICUBIC_INIT="${BICUBIC_INIT:-0}"           # 1 → enable SDEdit warm start
BICUBIC_INIT_T_START="${BICUBIC_INIT_T_START:-0.5}"

MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

METRICS="${METRICS:-psnr,ssim,lpips,dists}"
LPIPS_MAX_SIZE="${LPIPS_MAX_SIZE:-1024}"
DINO_WEIGHTS="${DINO_WEIGHTS:-}"            # empty → auto-resolve
NO_DINO="${NO_DINO:-0}"                     # 1 → disable DINO (for abl_nodino_* ckpts)

OUT_DIR="${OUT_DIR:-}"

# ── Build args ───────────────────────────────────────────────────────────────
ARGS=(
  --split "$SPLIT"
  --scale "$SCALE"
  --model "$MODEL"
  --cfg "$CFG"
  --num_sampling_steps "$STEPS"
  --bicubic_init_t_start "$BICUBIC_INIT_T_START"
  --max_samples "$MAX_SAMPLES"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --device "$DEVICE"
  --seed "$SEED"
  --metrics "$METRICS"
  --lpips_max_size "$LPIPS_MAX_SIZE"
)

# Git-10M meta / hierarchy / HF cache: only override if set; otherwise eval_sr.py
# auto-pulls the meta + hierarchy index from the Google Drive release and lets HF
# fetch lcybuaa/Git-10M into its default cache.
[[ -n "$META_FILE" ]]     && ARGS+=(--meta_file_path "$META_FILE")
[[ -n "$HINDEX_PKL" ]]    && ARGS+=(--hindex_pkl "$HINDEX_PKL")
[[ -n "$HF_CACHE" ]]      && ARGS+=(--hf_data_cache "$HF_CACHE")
[[ -n "$CKPT" ]]          && ARGS+=(--ckpt  "$CKPT")
[[ -z "$CKPT" ]]          && ARGS+=(--exp   "$EXP" --stage "$STAGE")
[[ -n "$DINO_WEIGHTS" ]]  && ARGS+=(--dino_weights "$DINO_WEIGHTS")
[[ "$NO_DINO" == "1" ]]   && ARGS+=(--no_dino)
[[ "$BICUBIC_INIT" == "1" ]] && ARGS+=(--bicubic_init)
[[ -n "$OUT_DIR" ]]       && ARGS+=(--out_dir "$OUT_DIR")

echo "============================================================"
echo "  SR eval"
echo "============================================================"
echo "  EXP/STAGE  : ${CKPT:-$EXP / stage $STAGE}"
echo "  SPLIT      : $SPLIT"
echo "  SCALE      : $SCALE   ($((2**SCALE))×)"
echo "  CFG        : $CFG"
echo "  STEPS      : $STEPS"
echo "  METRICS    : $METRICS"
echo "  DEVICE     : $DEVICE"
echo "============================================================"

cd "$ROOT"
python eval_sr.py "${ARGS[@]}" "$@"
