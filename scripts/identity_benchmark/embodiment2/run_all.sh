#!/bin/bash
# Sequential runner for D/E/F. Sequential because heavy CPU; thermal-safe.
set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
PY=$ROOT/venv/bin/python
LOG=$ROOT/logs/embodiment2/run_all.log
mkdir -p $(dirname "$LOG")
export PYTHONUNBUFFERED=1
export HSA_OVERRIDE_GFX_VERSION=11.0.0

log(){ echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
cool(){
    for i in $(seq 1 60); do
        T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 99000)
        if [ "$T" -le 55000 ]; then return 0; fi
        sleep 5
    done
}

log "=== embodiment2 run_all START ==="

log "--- D1: deeper envelope ---"
cool
$PY $ROOT/scripts/identity_benchmark/embodiment2/d1_deeper_envelope.py --seeds 4 2>&1 | tee -a "$LOG"

log "--- D2: axes ablation ---"
cool
$PY $ROOT/scripts/identity_benchmark/embodiment2/d2_axes_ablation.py --seeds 3 2>&1 | tee -a "$LOG"

log "--- D3: live envelope ---"
cool
$PY $ROOT/scripts/identity_benchmark/embodiment2/d3_live_envelope.py --seeds 3 2>&1 | tee -a "$LOG"

log "--- E1: scaleup (N=128,512 only; skip 2048 unless thermal headroom) ---"
cool
$PY $ROOT/scripts/identity_benchmark/embodiment2/e1_scaleup.py --seeds 2 --Ns 128,512 2>&1 | tee -a "$LOG"

log "--- E2: more tasks ---"
cool
$PY $ROOT/scripts/identity_benchmark/embodiment2/e2_tasks.py --seeds 3 2>&1 | tee -a "$LOG"

log "--- F: advantage hunt ---"
cool
$PY $ROOT/scripts/identity_benchmark/embodiment2/f_advantage.py --seeds 4 2>&1 | tee -a "$LOG"

log "=== embodiment2 run_all DONE ==="
