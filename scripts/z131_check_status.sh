#!/bin/bash
# Quick status check for embodiment pipeline on daedalus

sshpass -p "daedalus" ssh -o StrictHostKeyChecking=no "daedalus@192.168.0.37" "
echo '============================================================'
echo 'FEEL-SLM Embodiment Pipeline Status'
echo '============================================================'

# Check if running
if pgrep -f 'z129_embodiment' > /dev/null; then
    echo 'Status: RUNNING'
    ps aux | grep 'z129.*python' | grep -v grep | head -1 | awk '{print \"  CPU%:\", \$3, \"MEM%:\", \$4, \"Runtime:\", \$10}'
else
    echo 'Status: NOT RUNNING'
fi

echo ''
echo 'Results:'
for phase in phase2 phase3 reporter final; do
    count=\$(ls /home/daedalus/AMD_gfx1151_energy/results/z130_embodiment/\$phase/*.pt 2>/dev/null | wc -l)
    if [ \"\$count\" -gt 0 ]; then
        latest=\$(ls -t /home/daedalus/AMD_gfx1151_energy/results/z130_embodiment/\$phase/*.pt 2>/dev/null | head -1)
        echo \"  \$phase: \$count checkpoints (latest: \$(basename \$latest))\"
    else
        echo \"  \$phase: no checkpoints yet\"
    fi
done

echo ''
echo 'Log tail:'
tail -5 /home/daedalus/AMD_gfx1151_energy/logs/z130_embodiment.log 2>/dev/null || echo '  No log'
"
