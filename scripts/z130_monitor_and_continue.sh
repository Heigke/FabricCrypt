#!/bin/bash
# z130 - Monitor Phase 1 and continue with full embodiment pipeline
# NO COMPROMISES - WORK HARD

set -e

# Configuration
DAEDALUS_HOST="192.168.0.37"
DAEDALUS_USER="daedalus"
DAEDALUS_PASS="daedalus"
REMOTE_DIR="/home/daedalus/AMD_gfx1151_energy"
RESULTS_DIR="${REMOTE_DIR}/results/z125_night_train/phase1"
TARGET_PPL=15.0  # Target PPL to consider Phase 1 "good enough"
MIN_STEP=10000   # Minimum steps before considering early stop
CHECK_INTERVAL=300  # Check every 5 minutes

echo "============================================================"
echo "FEEL-SLM Full Embodiment Pipeline Monitor"
echo "Started: $(date)"
echo "============================================================"

ssh_cmd() {
    sshpass -p "$DAEDALUS_PASS" ssh -o StrictHostKeyChecking=no "$DAEDALUS_USER@$DAEDALUS_HOST" "$@"
}

scp_cmd() {
    sshpass -p "$DAEDALUS_PASS" scp -o StrictHostKeyChecking=no "$@"
}

get_latest_checkpoint() {
    ssh_cmd "ls -t ${RESULTS_DIR}/*.pt 2>/dev/null | head -1"
}

get_checkpoint_info() {
    local ckpt=$1
    ssh_cmd "source /home/daedalus/venvs/torch-rocm/bin/activate && python3 -c \"
import torch
import math
c = torch.load('$ckpt', map_location='cpu', weights_only=False)
step = c.get('step', 0)
loss = c.get('metrics', {}).get('train_loss', 999)
ppl = math.exp(min(loss, 10))
print(f'{step},{ppl:.2f}')
\""
}

is_training_running() {
    ssh_cmd "pgrep -f 'z121_real_training' > /dev/null 2>&1 && echo 'yes' || echo 'no'"
}

echo ""
echo "Monitoring Phase 1 training on daedalus..."
echo "Target PPL: ${TARGET_PPL}, Min steps: ${MIN_STEP}"
echo ""

PHASE1_DONE=false
BEST_CHECKPOINT=""
BEST_PPL=9999

while true; do
    # Check if training is running
    running=$(is_training_running)

    # Get latest checkpoint
    latest_ckpt=$(get_latest_checkpoint)

    if [ -n "$latest_ckpt" ]; then
        info=$(get_checkpoint_info "$latest_ckpt")
        step=$(echo $info | cut -d',' -f1)
        ppl=$(echo $info | cut -d',' -f2)

        echo "[$(date '+%H:%M:%S')] Step: $step, PPL: $ppl, Running: $running"

        # Track best
        if (( $(echo "$ppl < $BEST_PPL" | bc -l) )); then
            BEST_PPL=$ppl
            BEST_CHECKPOINT=$latest_ckpt
        fi

        # Check if we should proceed
        if [ "$running" = "no" ]; then
            echo "Training completed!"
            PHASE1_DONE=true
            break
        fi

        # Check if PPL is good enough and we have enough steps
        if (( step >= MIN_STEP )) && (( $(echo "$ppl <= $TARGET_PPL" | bc -l) )); then
            echo "PPL target reached at step $step!"
            PHASE1_DONE=true
            break
        fi
    fi

    sleep $CHECK_INTERVAL
done

if [ "$PHASE1_DONE" = true ]; then
    echo ""
    echo "============================================================"
    echo "Phase 1 Complete! Best checkpoint: $BEST_CHECKPOINT (PPL: $BEST_PPL)"
    echo "============================================================"

    # Copy best checkpoint locally
    LOCAL_CKPT="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z130_embodiment/phase1_best.pt"
    mkdir -p "$(dirname $LOCAL_CKPT)"

    echo "Copying checkpoint to local..."
    scp_cmd "$DAEDALUS_USER@$DAEDALUS_HOST:$BEST_CHECKPOINT" "$LOCAL_CKPT"

    echo ""
    echo "Starting Phase 2-3 and Reporter training on daedalus..."

    # Run the embodiment pipeline on daedalus
    ssh_cmd "cd ${REMOTE_DIR} && source /home/daedalus/venvs/torch-rocm/bin/activate && \
        nohup bash -c 'HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z129_embodiment_final.py \
            --phase1-checkpoint $BEST_CHECKPOINT \
            --output-dir results/z130_embodiment \
            --phase2-epochs 5 \
            --phase3-epochs 3 \
            --reporter-epochs 20 \
            2>&1' > logs/z130_embodiment.log 2>&1 &"

    echo "Embodiment pipeline started on daedalus!"
    echo "Monitor with: sshpass -p daedalus ssh daedalus@192.168.0.37 'tail -f ${REMOTE_DIR}/logs/z130_embodiment.log'"
fi

echo ""
echo "Done: $(date)"
