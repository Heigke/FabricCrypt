#!/bin/bash
# FEEL z27 Dual-Machine Orchestrator
#
# z1 (ikaros): Main GRPO training with STE hard skip
# z2 (daedalus): Continuous validation, same W&B run
#
# Usage: ./z27_dual_machine.sh

set -e

# Configuration
DAEDALUS_HOST="daedalus@192.168.0.37"
DAEDALUS_PASS="daedalus"
IKAROS_IP=$(hostname -I | awk '{print $1}')

PROJECT_DIR="/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy"
CHECKPOINT_DIR="$PROJECT_DIR/models/grpo_z27"
LOG_DIR="$PROJECT_DIR/logs"

# Create dirs
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

echo "============================================================"
echo "FEEL z27: DUAL-MACHINE TRAINING"
echo "============================================================"
echo ""
echo "z1 (ikaros @ $IKAROS_IP): Training with STE hard skip"
echo "z2 (daedalus @ 192.168.0.37): Continuous validation"
echo ""
echo "Both machines write to SAME W&B run for unified metrics."
echo "============================================================"
echo ""

# Step 1: Start training on z1 and capture W&B run ID
echo "[1/3] Starting training on z1 (ikaros)..."

# Start training in background, capturing output
HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 "$PROJECT_DIR/scripts/z27_ste_trainer.py" \
    --epochs 3 \
    --max-prompts 500 \
    --num-samples 4 \
    --max-tokens 128 \
    --target-power 75 \
    --power-band 15 \
    --gate-lr 1e-4 \
    --gate-reg 0.1 \
    --start-threshold 0.55 \
    --end-threshold 0.35 \
    --val-every 100 \
    --val-samples 128 \
    --log-every 10 \
    --disturbance periodic \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$LOG_DIR/z27_training.log" &

TRAIN_PID=$!
echo "[1/3] Training started (PID: $TRAIN_PID)"

# Wait for W&B to initialize and extract run ID
echo "[2/3] Waiting for W&B run ID..."
sleep 30

WANDB_RUN_ID=""
for i in {1..10}; do
    WANDB_RUN_ID=$(grep -oP 'runs/\K[a-z0-9]+' "$LOG_DIR/z27_training.log" | head -1)
    if [ -n "$WANDB_RUN_ID" ]; then
        break
    fi
    sleep 5
done

if [ -z "$WANDB_RUN_ID" ]; then
    echo "ERROR: Could not extract W&B run ID"
    exit 1
fi

echo "[2/3] W&B Run ID: $WANDB_RUN_ID"
echo ""

# Step 3: Start validator on z2 (daedalus)
echo "[3/3] Starting validator on z2 (daedalus)..."

# Copy validator script to daedalus
sshpass -p "$DAEDALUS_PASS" scp -o StrictHostKeyChecking=no \
    "$PROJECT_DIR/scripts/z27_remote_validator.py" \
    "$DAEDALUS_HOST:/tmp/"

# Start validator on daedalus
sshpass -p "$DAEDALUS_PASS" ssh -o StrictHostKeyChecking=no "$DAEDALUS_HOST" \
    "source venvs/torch-rocm/bin/activate && \
     nohup python3 /tmp/z27_remote_validator.py \
         --run-id $WANDB_RUN_ID \
         --checkpoint-host ikaros@$IKAROS_IP \
         --checkpoint-dir $CHECKPOINT_DIR \
         --password '$DAEDALUS_PASS' \
         --val-interval 120 \
         --val-samples 512 \
         > /tmp/z27_validator.log 2>&1 &"

echo "[3/3] Validator started on daedalus"
echo ""

echo "============================================================"
echo "DUAL-MACHINE TRAINING RUNNING"
echo "============================================================"
echo ""
echo "W&B Run: https://wandb.ai/bergvall-eric/feel-z27-ste/runs/$WANDB_RUN_ID"
echo ""
echo "Logs:"
echo "  z1 training: $LOG_DIR/z27_training.log"
echo "  z2 validator: ssh daedalus 'tail -f /tmp/z27_validator.log'"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/z27_training.log"
echo ""
echo "Stop:"
echo "  kill $TRAIN_PID"
echo "  ssh daedalus 'pkill -f z27_remote_validator'"
echo ""
echo "============================================================"

# Wait for training to complete
wait $TRAIN_PID
echo "Training complete!"

# Stop validator on daedalus
sshpass -p "$DAEDALUS_PASS" ssh "$DAEDALUS_HOST" "pkill -f z27_remote_validator" || true

echo "Done!"
