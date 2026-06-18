#!/bin/bash
# z31 Checkpoint Sync to Daedalus
# Syncs checkpoints from ikaros training to daedalus for validation

REMOTE="daedalus@192.168.0.37"
REMOTE_DIR="~/z31_checkpoints"
LOCAL_DIR="models/z31_embodied"
INTERVAL=120  # Sync every 2 minutes

sync_once() {
    echo "[$(date '+%H:%M:%S')] Syncing checkpoints to daedalus..."

    # Ensure remote directory exists
    sshpass -p "daedalus" ssh -o StrictHostKeyChecking=no $REMOTE "mkdir -p $REMOTE_DIR"

    # Sync checkpoints
    sshpass -p "daedalus" rsync -avz --progress \
        --include='step_*.pt' \
        --exclude='*' \
        $LOCAL_DIR/ $REMOTE:$REMOTE_DIR/

    echo "[$(date '+%H:%M:%S')] Sync complete"
}

# Also sync the validator script
sync_validator() {
    echo "[$(date '+%H:%M:%S')] Syncing validator script..."
    sshpass -p "daedalus" scp scripts/z31_daedalus_validator.py $REMOTE:~/z31_validator.py
    echo "[$(date '+%H:%M:%S')] Validator synced"
}

start_validator() {
    echo "[$(date '+%H:%M:%S')] Starting validator on daedalus..."
    sshpass -p "daedalus" ssh -o StrictHostKeyChecking=no $REMOTE \
        "cd ~ && source venvs/torch-rocm/bin/activate && \
         nohup python z31_validator.py --checkpoint-dir ~/z31_checkpoints --watch --interval 60 \
         > z31_validator.log 2>&1 &"
    echo "[$(date '+%H:%M:%S')] Validator started"
}

case "$1" in
    --loop)
        echo "Starting continuous sync (every ${INTERVAL}s)..."
        sync_validator
        while true; do
            sync_once
            sleep $INTERVAL
        done
        ;;
    --validator)
        sync_validator
        ;;
    --start-validator)
        start_validator
        ;;
    --full-setup)
        sync_validator
        start_validator
        sync_once
        ;;
    *)
        sync_once
        ;;
esac
