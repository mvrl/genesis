#!/usr/bin/env bash
# sync_ckpts.sh — download the Genesis demo checkpoints from the Hugging Face release
# (https://huggingface.co/MVRL/genesis) to GENESIS_CKPT_BASE.
#
# Usage:
#   export GENESIS_CKPT_BASE=/path/to/ckpt_cache   # optional; default <repo>/logs/ckpts
#   bash demos/sync_ckpts.sh
#
# The release repo is public — no authentication or extra tooling needed.
# Grabs the main SR/OP checkpoints (B/16 + H/16) and the DINOv3-sat encoder;
# the eval metadata and ablation checkpoints are only needed for the paper
# evaluation (use src/eval/download_ckpts.sh for the full release).
# Override the source repo with: export GENESIS_HF_REPO=<user/repo>

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CB="${GENESIS_CKPT_BASE:-$REPO/logs/ckpts}"
HF_REPO="${GENESIS_HF_REPO:-MVRL/genesis}"

echo "Destination : $CB"
echo "Source      : https://huggingface.co/$HF_REPO"
echo

mkdir -p "$CB"
python - "$HF_REPO" "$CB" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    local_dir=dest,
    allow_patterns=["main/**", "*.pth"],
)
PY

echo
echo "Done. Contents of $CB:"
ls "$CB"
