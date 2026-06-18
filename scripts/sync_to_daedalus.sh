#!/bin/bash
# Sync z30 checkpoints from ikaros to daedalus
# Run on ikaros: ./scripts/sync_to_daedalus.sh [--loop]

REMOTE="daedalus@192.168.0.37"
REMOTE_DIR="~/z30_checkpoints"
INTERVAL=120  # seconds between syncs

sync_once() {
    echo "[$(date +%H:%M:%S)] Syncing checkpoints to daedalus..."
    sshpass -p "daedalus" ssh -o StrictHostKeyChecking=no $REMOTE "mkdir -p $REMOTE_DIR" 2>/dev/null
    sshpass -p "daedalus" rsync -avz --progress \
        models/z30_overnight/*.pt \
        $REMOTE:$REMOTE_DIR/ 2>/dev/null
    echo "[$(date +%H:%M:%S)] Sync complete."
}

if [ "$1" == "--loop" ]; then
    echo "Starting continuous sync (every ${INTERVAL}s)..."
    while true; do
        sync_once
        sleep $INTERVAL
    done
else
    sync_once
fi
