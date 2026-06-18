#!/bin/bash
# Sync z32 checkpoints from ikaros to daedalus
REMOTE="daedalus"
LOCAL_DIR="models/z32_embodied"
REMOTE_DIR="/home/ikaros/feel_checkpoints/z32_embodied"

while true; do
    if ls ${LOCAL_DIR}/*.pt 2>/dev/null | head -1 > /dev/null; then
        rsync -avz --progress ${LOCAL_DIR}/*.pt ${REMOTE}:${REMOTE_DIR}/ 2>&1 | tail -5
        echo "[$(date +%H:%M:%S)] Synced checkpoints to daedalus"
    else
        echo "[$(date +%H:%M:%S)] No checkpoints yet..."
    fi
    sleep 60
done
