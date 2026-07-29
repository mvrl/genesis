#!/usr/bin/env bash
# =============================================================================
# run_op.sh — Launch script for outpainting (mask-based image completion)
#
# Usage:
#   cd <repo>/src/outpainting
#   source .venv/bin/activate   # after: uv sync (repo root)
#   ./run_op.sh                          # default run
#   ./run_op.sh --lr 5e-5                # pass extra args to main.py
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── TGI paths (hardcoded) ─────────────────────────────────────────────────────
DINO_WEIGHTS="${GENESIS_DINO_WEIGHTS:-}"
META_FILE="${GENESIS_GIT10M_META:-}"   # empty → main.py uses $GENESIS_CKPT_BASE/shared
HF_CACHE="${HF_DATASETS_CACHE:-}"      # empty → HF default cache

# ── Configurable knobs (override via env vars) ────────────────────────────────
# MODEL_SIZE   : B16 | H16
# MAX_STEPS    : total training steps (curriculum boundaries auto-computed)
# OUTPUT_DIR   : checkpoint + log output directory
# DEVICES      : number of GPUs (int) or comma-separated IDs e.g. "0,1"
# WANDB_MODE   : disabled | online
# NUM_WORKERS  : dataloader workers
# RESUME_CKPT  : Lightning checkpoint to resume full training state from
#
# OP config is locked at no-DINO + no-time-weight, quadrant-only mask
# curriculum; NO_DINO and NO_TIME_WEIGHT default to 1 and are not intended
# for user override.
MODEL_SIZE="${MODEL_SIZE:-B16}"
MAX_STEPS="${MAX_STEPS:-800000}"
NO_DINO="${NO_DINO:-1}"
NO_TIME_WEIGHT="${NO_TIME_WEIGHT:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output_op_new}"
DEVICES="${DEVICES:-2}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
NUM_WORKERS="${NUM_WORKERS:-16}"
RESUME_CKPT="${RESUME_CKPT:-}"

if [[ "$MODEL_SIZE" == "B16" ]]; then
    MODEL_NAME="JiT-B/16"
else
    MODEL_NAME="JiT-H/16"
fi

# ── Sanity checks (only for explicitly-set overrides; empty = resolved by main.py) ──
[[ -z "$META_FILE" || -e "$META_FILE" ]] || { echo "ERROR: path not found: $META_FILE" >&2; exit 1; }

# Point HF at a custom cache only if one was provided (else HF uses its default).
[[ -n "$HF_CACHE" ]] && { export HF_DATASETS_CACHE="$HF_CACHE"; export HF_HOME="$HF_CACHE"; }

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(
    --model            "$MODEL_NAME"
    --output_dir       "$OUTPUT_DIR"
    --devices          "$DEVICES"
    --num_workers      "$NUM_WORKERS"
    --max_steps        "$MAX_STEPS"
    --wandb_mode       "$WANDB_MODE"
)
# Git-10M meta / HF cache: only override if set; otherwise main.py uses the
# Google-Drive-cached location ($GENESIS_CKPT_BASE/shared) and HF's cache.
[[ -n "$META_FILE" ]] && ARGS+=(--meta_file_path "$META_FILE")
[[ -n "$HF_CACHE" ]]  && ARGS+=(--hf_data_cache "$HF_CACHE")
[[ -n "$WANDB_RUN_NAME" ]] && ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
[[ -n "$RESUME_CKPT" ]] && ARGS+=(--resume_ckpt "$RESUME_CKPT")
[[ "$NO_TIME_WEIGHT" == "1" ]] && ARGS+=(--no-enable_time_weight)
if [[ "$NO_DINO" == "1" ]]; then
    ARGS+=(--dino_weights "")
elif [[ -f "$DINO_WEIGHTS" ]]; then
    ARGS+=(--dino_weights "$DINO_WEIGHTS")
else
    echo "WARNING: DINO weights not found ($DINO_WEIGHTS), disabling DINO." >&2
    ARGS+=(--dino_weights "")
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo "======================================================="
echo "  OP-New Experiment"
echo "======================================================="
echo "  MODEL        : $MODEL_NAME"
echo "  MAX_STEPS    : $MAX_STEPS"
echo "  CURRICULUM   : quadrant-aligned, easy→hard known-count (boundaries auto-computed)"
echo "  RESUME_CKPT  : ${RESUME_CKPT:-<none>}"
echo "  OUTPUT_DIR   : $OUTPUT_DIR"
echo "  DEVICES      : $DEVICES"
echo "  WANDB_MODE   : $WANDB_MODE"
echo "  WANDB_NAME   : ${WANDB_RUN_NAME:-<auto>}"
echo "======================================================="

cd "$ROOT"
python main.py "${ARGS[@]}" "$@"
