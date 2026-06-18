#!/bin/bash
# Overnight experiment runner — z2235, z2236, z2237
# Runs sequentially until 0900 or all complete
set -e
cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTHONUNBUFFERED=1
VENV=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/bin/python

echo "=== OVERNIGHT RUNNER STARTED: $(date) ==="
echo "Will run z2235, z2236, z2237 sequentially"
echo ""

# z2235: XOR + NARMA benchmarks
echo ">>> Starting z2235_xor_narma_mac.py at $(date)"
$VENV scripts/z2235_xor_narma_mac.py 2>&1 | tee results/z2235_stdout.txt
echo ">>> z2235 completed at $(date)"
echo ""

# z2236: GPU neuromorphic
echo ">>> Starting z2236_gpu_neuromorphic.py at $(date)"
$VENV scripts/z2236_gpu_neuromorphic.py 2>&1 | tee results/z2236_stdout.txt
echo ">>> z2236 completed at $(date)"
echo ""

# z2237: Convergence
echo ">>> Starting z2237_convergence.py at $(date)"
$VENV scripts/z2237_convergence.py 2>&1 | tee results/z2237_stdout.txt
echo ">>> z2237 completed at $(date)"
echo ""

echo "=== ALL OVERNIGHT EXPERIMENTS COMPLETE: $(date) ==="

# Summarize results
echo ""
echo "=== RESULTS SUMMARY ==="
for f in results/z2235_xor_narma_mac.json results/z2236_gpu_neuromorphic.json results/z2237_convergence.json; do
    if [ -f "$f" ]; then
        echo "--- $(basename $f) ---"
        python3 -c "import json; d=json.load(open('$f')); print(d.get('summary','no summary'))"
    fi
done
