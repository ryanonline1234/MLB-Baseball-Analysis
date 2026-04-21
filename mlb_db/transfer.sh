#!/usr/bin/env bash
# transfer.sh — sync raw_video to DGX Spark for GPU processing,
#               then pull processed keypoints/ back to local machine.
#
# Usage:
#   ./transfer.sh <DGX_IP>
#   DGX_USER=ubuntu REMOTE_DIR=~/mlb_db ./transfer.sh 192.168.1.100
#
# Environment overrides:
#   DGX_USER    SSH username (default: ubuntu)
#   REMOTE_DIR  Remote path for the mlb_db directory (default: ~/mlb_db)

set -euo pipefail

# ─── Args ─────────────────────────────────────────────────────────────────────

DGX_IP="${1:-}"
if [[ -z "$DGX_IP" ]]; then
    echo "Usage: ./transfer.sh <DGX_IP>"
    echo "       DGX_USER=ubuntu REMOTE_DIR=~/mlb_db ./transfer.sh 192.168.1.100"
    exit 1
fi

DGX_USER="${DGX_USER:-ubuntu}"
REMOTE_DIR="${REMOTE_DIR:-~/mlb_db}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DGX="${DGX_USER}@${DGX_IP}"

# ─── Header ───────────────────────────────────────────────────────────────────

echo "======================================================"
echo "  MLB Pitcher DB — DGX Transfer"
echo "  Remote : ${DGX}:${REMOTE_DIR}"
echo "  Local  : ${SCRIPT_DIR}"
echo "======================================================"
echo ""

# ─── Step 1: Ensure remote directories exist ──────────────────────────────────

echo "[setup] Creating remote directories..."
ssh "${DGX}" "mkdir -p ${REMOTE_DIR}/raw_video ${REMOTE_DIR}/keypoints"

# ─── Step 2: Push raw_video/ → DGX ────────────────────────────────────────────

echo ""
echo "[1/2] Pushing raw_video/ → DGX (skipping unchanged files)..."
rsync \
    --archive \
    --checksum \
    --human-readable \
    --progress \
    --exclude '*.DS_Store' \
    --exclude '__pycache__' \
    "${SCRIPT_DIR}/raw_video/" \
    "${DGX}:${REMOTE_DIR}/raw_video/"

echo ""
echo "      raw_video/ sync complete."

# ─── Step 3: Pull keypoints/ ← DGX ───────────────────────────────────────────

echo ""
echo "[2/2] Pulling keypoints/ ← DGX (skipping unchanged files)..."
rsync \
    --archive \
    --checksum \
    --human-readable \
    --progress \
    "${DGX}:${REMOTE_DIR}/keypoints/" \
    "${SCRIPT_DIR}/keypoints/"

echo ""
echo "      keypoints/ sync complete."

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "======================================================"
echo "  Transfer complete."
echo ""
echo "  Next steps:"
echo "    On DGX: cd ${REMOTE_DIR} && python run.py add-pitcher --backend mmpose"
echo "    Locally: python run.py add-pitcher  (aggregates pulled keypoints)"
echo "======================================================"
