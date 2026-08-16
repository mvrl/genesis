#!/usr/bin/env bash
# Download all Genesis checkpoints + DINOv3-sat weights + split metadata from the public
# Hugging Face release (https://huggingface.co/MVRL/genesis) into $GENESIS_CKPT_BASE
# (default <repo>/logs/ckpts). Optional: the eval wrappers auto-download each file on first
# use, but pre-staging avoids parallel jobs racing on a download. No authentication needed.
#   bash src/eval/download_ckpts.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CB="${GENESIS_CKPT_BASE:-$REPO/logs/ckpts}"
HF_REPO="${GENESIS_HF_REPO:-MVRL/genesis}"
mkdir -p "$CB"
echo "Downloading Genesis release ($HF_REPO) -> $CB (~70 GB)"
python - "$HF_REPO" "$CB" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    local_dir=dest,
    allow_patterns=["main/**", "ablations/**", "shared/**", "*.pth"],
)
PY
echo "Done. If you used a custom path, export GENESIS_CKPT_BASE=$CB before running the eval scripts."
