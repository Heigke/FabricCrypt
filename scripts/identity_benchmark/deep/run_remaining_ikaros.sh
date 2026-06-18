#!/bin/bash
set -e
cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
PY=venv/bin/python
RES=results/IDENTITY_BENCHMARK_2026-05-30/deep/ikaros
LOG=logs/identity_deep
mkdir -p $LOG
while [ ! -f $RES/A_power.json ]; do sleep 5; done
echo "$(date) A done, cool 60s" >> $LOG/remaining.log
sleep 60
echo "$(date) E" >> $LOG/remaining.log
$PY -u scripts/identity_benchmark/deep/E_cpu_per_core.py --cores 16 --repeats 4 --out $RES/E_cpu.json >> $LOG/E_ikaros.log 2>&1
sleep 30
echo "$(date) B" >> $LOG/remaining.log
$PY -u scripts/identity_benchmark/deep/B_thermal_tc.py --cycles 6 --heat_s 25 --cool_s 50 --out $RES/B_thermal.json >> $LOG/B_ikaros.log 2>&1
sleep 30
echo "$(date) D" >> $LOG/remaining.log
HSA_OVERRIDE_GFX_VERSION=11.0.0 $PY -u scripts/identity_benchmark/deep/D_vmin_sweep.py --reps 60 --out $RES/D_vmin.json >> $LOG/D_ikaros.log 2>&1
echo "$(date) ALL DONE" >> $LOG/remaining.log
