#!/bin/bash
set -u
NODE=$1; ARGS=$2; PY=$3; shift 3
ENV_VARS=("$@")
OUT=results/sweep_d2
mkdir -p $OUT/_logs
LOG=$OUT/_logs/${NODE}_$(date +%H%M).log
echo "[$(date)] D2 node=$NODE START" > $LOG
TOTAL=$(grep -c . $ARGS); DONE=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  CELL=$(echo "$line" | grep -oP '(?<=--cell_id )\S+')
  DONE=$((DONE+1))
  if [ -e "$OUT/cell_${CELL}_mnist/summary.json" ]; then
    echo "[$(date)] SKIP $CELL ($DONE/$TOTAL)" >> $LOG
    continue
  fi
  # APU thermal check (only meaningful on ikaros/daedalus; ZGX has different path)
  if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
    T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{print int($1/1000)}')
    while [ "$T" -gt 85 ]; do
      echo "[$(date)] THERMAL WAIT APU=$T°C" >> $LOG
      sleep 15
      T=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{print int($1/1000)}')
    done
  fi
  echo "[$(date)] BEGIN $CELL ($DONE/$TOTAL)" >> $LOG
  env "${ENV_VARS[@]}" $PY scripts/z280_d2_corrective_cell.py $line --out_dir $OUT \
    --surrogate results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz \
    >> $LOG 2>&1
done < $ARGS
echo "[$(date)] D2 node=$NODE DONE" >> $LOG
