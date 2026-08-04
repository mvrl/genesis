#!/usr/bin/env bash
# =============================================================================
# run_sr.sh — Launch script for superresolution (full-tile ZL→ZL+1 SR)
#
# Usage:
#   cd <repo>/src/superresolution
#   source .venv/bin/activate   # after: uv sync (repo root)
#   ./run_sr.sh                          # default run
#   ./run_sr.sh --batch_size 8           # pass extra args to main.py
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── TGI paths (hardcoded) ─────────────────────────────────────────────────────
DINO_WEIGHTS="${GENESIS_DINO_WEIGHTS:-}"
META_FILE="${GENESIS_GIT10M_META:-}"   # empty → main.py uses $GENESIS_CKPT_BASE/shared
HINDEX_PKL="${GENESIS_HINDEX:-}"       # empty → main.py uses $GENESIS_CKPT_BASE/shared
HF_CACHE="${HF_DATASETS_CACHE:-}"      # empty → HF default cache

# ── Configurable knobs (override via env vars) ────────────────────────────────
# MODEL_SIZE     : B16 | H16
# MAX_STEPS      : total training steps (curriculum boundaries auto-computed)
# NO_DINO        : 1 = pixel-only conditioning (disable DINO additional signal)
# LPIPS_WEIGHT   : perceptual loss weight (0.0 = disabled)
# OUTPUT_DIR     : checkpoint + log output directory
# DEVICES        : number of GPUs (int) or comma-separated IDs e.g. "0,1"
# WANDB_MODE     : disabled | online
# NUM_WORKERS    : dataloader workers
# RESUME_CKPT    : Lightning checkpoint to resume full training state from
MODEL_SIZE="${MODEL_SIZE:-B16}"
MAX_STEPS="${MAX_STEPS:-800000}"
NO_DINO="${NO_DINO:-0}"
LPIPS_WEIGHT="${LPIPS_WEIGHT:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output_sr_new}"
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
for f in "$META_FILE" "$HINDEX_PKL"; do
    [[ -z "$f" || -e "$f" ]] || { echo "ERROR: path not found: $f" >&2; exit 1; }
done
# DINO weights only required when DINO is enabled; empty ⇒ set $GENESIS_DINO_WEIGHTS.
if [[ "$NO_DINO" != "1" && -n "$DINO_WEIGHTS" && ! -e "$DINO_WEIGHTS" ]]; then
    echo "ERROR: DINO weights not found: $DINO_WEIGHTS" >&2; exit 1
fi

# Point HF at a custom cache only if one was provided (else HF uses its default).
[[ -n "$HF_CACHE" ]] && { export HF_DATASETS_CACHE="$HF_CACHE"; export HF_HOME="$HF_CACHE"; }

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(
    --model          "$MODEL_NAME"
    --output_dir     "$OUTPUT_DIR"
    --devices        "$DEVICES"
    --num_workers    "$NUM_WORKERS"
    --max_steps      "$MAX_STEPS"
    --lpips_weight   "$LPIPS_WEIGHT"
    --wandb_mode     "$WANDB_MODE"
)
# Git-10M meta / hierarchy / HF cache: only override if set; otherwise main.py
# uses the Google-Drive-cached location ($GENESIS_CKPT_BASE/shared) and HF's cache.
[[ -n "$META_FILE" ]]  && ARGS+=(--meta_file_path "$META_FILE")
[[ -n "$HINDEX_PKL" ]] && ARGS+=(--hindex_pkl "$HINDEX_PKL")
[[ -n "$HF_CACHE" ]]   && ARGS+=(--hf_data_cache "$HF_CACHE")
[[ -n "$WANDB_RUN_NAME" ]] && ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
[[ "$NO_DINO"       != "1" ]] && ARGS+=(--dino_weights "$DINO_WEIGHTS") || ARGS+=(--no-use_dino)
[[ -n "$RESUME_CKPT"      ]] && ARGS+=(--resume_ckpt "$RESUME_CKPT")

# ── Launch ────────────────────────────────────────────────────────────────────
echo "======================================================="
echo "  SR-New Experiment"
echo "======================================================="
echo "  MODEL        : $MODEL_NAME"
echo "  DINO         : $([ "$NO_DINO" == "1" ] && echo "disabled (pixel only)" || echo "enabled (pixel + DINO)")"
echo "  LPIPS_WEIGHT : $LPIPS_WEIGHT"
echo "  MAX_STEPS    : $MAX_STEPS"
echo "  CURRICULUM   : enabled (boundaries auto-computed)"
echo "  RESUME_CKPT  : ${RESUME_CKPT:-<none>}"
echo "  OUTPUT_DIR   : $OUTPUT_DIR"
echo "  DEVICES      : $DEVICES"
echo "  WANDB_MODE   : $WANDB_MODE"
echo "  WANDB_NAME   : ${WANDB_RUN_NAME:-<auto>}"
echo "======================================================="

cd "$ROOT"
python main.py "${ARGS[@]}" "$@"
