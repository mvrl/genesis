#!/usr/bin/env bash
# Download all Genesis checkpoints + DINOv3-sat weights + split metadata from the public Google
# Drive release into $GENESIS_CKPT_BASE (default <repo>/logs/ckpts). Optional: the eval wrappers
# auto-pull each file on first use, but pre-staging avoids parallel jobs racing on a download.
# Requires an rclone remote named `gdrive` (see the top-level README, "Data & weights").
#   bash src/eval/download_ckpts.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CB="${GENESIS_CKPT_BASE:-$REPO/logs/ckpts}"
GID=1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA
DINO=dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
command -v rclone >/dev/null || { echo "ERROR: rclone not on PATH — install it and configure a 'gdrive' remote (README → Data & weights)."; exit 1; }
rclone listremotes 2>/dev/null | grep -q '^gdrive:' || { echo "ERROR: rclone remote 'gdrive:' not configured — run 'rclone config' (new remote 'gdrive', type 'drive')."; exit 1; }
mkdir -p "$CB"
echo "Downloading Genesis release -> $CB (~tens of GB)"
rclone copy gdrive:main      "$CB/main"      --drive-root-folder-id "$GID" --transfers 4 --progress
rclone copy gdrive:ablations "$CB/ablations" --drive-root-folder-id "$GID" --transfers 4 --progress
rclone copy gdrive:shared    "$CB/shared"    --drive-root-folder-id "$GID" --progress
rclone copy "gdrive:$DINO"    "$CB"          --drive-root-folder-id "$GID" --progress
echo "Done. If you used a custom path, export GENESIS_CKPT_BASE=$CB before running the eval scripts."
