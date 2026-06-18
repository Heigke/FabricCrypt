#!/bin/bash
# Overnight experiment runner — z2310, z2311, z2313
# Safe: FPGA Ethernet + hwmon only. No /dev/mem, no MSR.
set -e

cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=11.0.0
VENV=venv/bin/python

echo "=== OVERNIGHT EXPERIMENT SUITE ==="
echo "Started: $(date)"
echo "Experiments: z2310 (Mackey-Glass), z2311 (IPC), z2313 (Surrogate Controls)"
echo ""

# Wait for cool temps before starting
wait_cool() {
    local temp=$(cat /sys/class/thermal/thermal_zone0/temp)
    temp=$((temp / 1000))
    while [ "$temp" -gt 50 ]; do
        echo "  [COOL] APU=${temp}C, waiting..."
        sleep 10
        temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        temp=$((temp / 1000))
    done
    echo "  [COOL] APU=${temp}C — ready"
}

echo "=== z2310: Mackey-Glass Prediction ==="
echo "Started: $(date)"
wait_cool
$VENV scripts/z2310_mackey_glass.py 2>&1 | tee results/z2310_log.txt
echo "z2310 finished: $(date)"
echo ""

echo "=== z2311: Information Processing Capacity ==="
echo "Started: $(date)"
wait_cool
$VENV scripts/z2311_ipc_capacity.py 2>&1 | tee results/z2311_log.txt
echo "z2311 finished: $(date)"
echo ""

echo "=== z2313: Surrogate Data Controls ==="
echo "Started: $(date)"
wait_cool
$VENV scripts/z2313_surrogate_controls.py 2>&1 | tee results/z2313_log.txt
echo "z2313 finished: $(date)"
echo ""

echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "Finished: $(date)"
echo "Results in: results/z2310_*.json, results/z2311_*.json, results/z2313_*.json"
