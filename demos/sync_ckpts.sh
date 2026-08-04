#!/usr/bin/env bash
# sync_ckpts.sh — download Genesis checkpoints from Google Drive to GENESIS_CKPT_BASE
#
# Usage:
#   export GENESIS_CKPT_BASE=/path/to/ckpt_cache   # where the Google-Drive checkpoints land
#   bash demos/sync_ckpts.sh
#
# Requirements: rclone configured with a Google Drive remote (type = drive)
#   Default remote name is "gdrive"; run `rclone listremotes` to verify.
#   Override with: export GENESIS_GDRIVE_REMOTE=<your-remote-name>
#
# The release folder is public ("Anyone with the link can view"); any
# authenticated Google account works.
#   Folder URL: https://drive.google.com/drive/folders/1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA
#   Override the folder ID with: export GENESIS_GDRIVE_FOLDER_ID=<id>

set -euo pipefail

RCLONE_REMOTE="${GENESIS_GDRIVE_REMOTE:-gdrive}"
GDRIVE_FOLDER_ID="${GENESIS_GDRIVE_FOLDER_ID:-1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA}"

if [[ -z "${GENESIS_CKPT_BASE:-}" ]]; then
    echo "ERROR: GENESIS_CKPT_BASE is not set."
    echo "       Add this to your ~/.bashrc and re-source it:"
    echo "       export GENESIS_CKPT_BASE=/path/to/checkpoints/project_genesis/gdrive_copy"
    exit 1
fi

if ! command -v rclone &>/dev/null; then
    echo "ERROR: rclone is not on PATH."
    echo "       Install rclone and configure a Google Drive remote named '${RCLONE_REMOTE}'."
    echo "       See: https://rclone.org/drive/"
    exit 1
fi

REMOTE_TYPE="$(rclone config show "${RCLONE_REMOTE}:" 2>/dev/null | awk '/^type/{print $3}')"
if [[ "${REMOTE_TYPE}" != "drive" ]]; then
    echo "ERROR: rclone remote '${RCLONE_REMOTE}' is not a Google Drive remote (type = ${REMOTE_TYPE:-not found})."
    echo "       This script only supports Google Drive (type = drive)."
    echo "       Run: rclone config  — and create a remote of type 'drive' named '${RCLONE_REMOTE}'."
    exit 1
fi

echo "Destination : $GENESIS_CKPT_BASE"
echo "Source      : ${RCLONE_REMOTE}:  (folder id ${GDRIVE_FOLDER_ID})"
echo

mkdir -p "$GENESIS_CKPT_BASE"

rclone_args=(
    copy "${RCLONE_REMOTE}:" "$GENESIS_CKPT_BASE"
    --drive-root-folder-id "$GDRIVE_FOLDER_ID"
    --progress
    --transfers 4
    --checkers 8
)
rclone "${rclone_args[@]}"

echo
echo "Done. Contents of $GENESIS_CKPT_BASE:"
ls "$GENESIS_CKPT_BASE"
