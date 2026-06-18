#!/bin/bash
# Continuous validation for z42 - checks for new checkpoints and validates them

CHECKPOINT_DIR="models/z43_embodied"
RESULTS_DIR="results"
PYTHON="/home/ikaros/venvs/torch-rocm/bin/python"
VALIDATOR="scripts/z42_embodied_validator.py"
VALIDATED_FILE=".validated_checkpoints"

mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR"
touch "$VALIDATED_FILE"

echo "========================================"
echo "Z42 CONTINUOUS VALIDATION"
echo "========================================"
echo "Watching: $CHECKPOINT_DIR"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Check for new checkpoints
    for ckpt in "$CHECKPOINT_DIR"/step_*.pt; do
        if [ -f "$ckpt" ]; then
            ckpt_name=$(basename "$ckpt")

            # Skip if already validated
            if grep -q "$ckpt_name" "$VALIDATED_FILE" 2>/dev/null; then
                continue
            fi

            echo "[$(date '+%H:%M:%S')] Found new checkpoint: $ckpt_name"

            # Extract step number
            step=$(echo "$ckpt_name" | grep -oP 'step_\K\d+')
            output_file="$RESULTS_DIR/z42_validation_step${step}.json"

            echo "[$(date '+%H:%M:%S')] Running validation..."
            $PYTHON -u "$VALIDATOR" --checkpoint "$ckpt" --output "$output_file" 2>&1 | tee -a z42_continuous_validation.log

            # Mark as validated
            echo "$ckpt_name" >> "$VALIDATED_FILE"
            echo "[$(date '+%H:%M:%S')] Validation complete: $output_file"
            echo ""
        fi
    done

    # Also check daedalus for new checkpoints and sync them
    if sshpass -p "daedalus" ssh daedalus@192.168.0.37 "ls ~/AMD_gfx1151_energy/models/z42_embodied/*.pt 2>/dev/null" | head -1 > /dev/null 2>&1; then
        for remote_ckpt in $(sshpass -p "daedalus" ssh daedalus@192.168.0.37 "ls ~/AMD_gfx1151_energy/models/z42_embodied/*.pt 2>/dev/null" 2>/dev/null); do
            ckpt_name=$(basename "$remote_ckpt")
            local_ckpt="$CHECKPOINT_DIR/$ckpt_name"

            if [ ! -f "$local_ckpt" ]; then
                echo "[$(date '+%H:%M:%S')] Syncing checkpoint from daedalus: $ckpt_name"
                sshpass -p "daedalus" scp daedalus@192.168.0.37:"$remote_ckpt" "$local_ckpt"
            fi
        done
    fi

    sleep 60  # Check every minute
done
