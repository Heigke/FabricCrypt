#!/bin/bash
# Usage: run_sweep_v2.sh <node_name> <args_file> <python> [env_vars...]
# env_vars are passed as KEY=VAL strings, applied via env command.
set -u
NODE=$1
ARGS_FILE=$2
PY=$3
shift 3
ENV_VARS=("$@")
OUT=results/sweep_v2
mkdir -p $OUT/_logs
LOG=$OUT/_logs/${NODE}_$(date +%H%M).log
echo "[$(date)] node=$NODE START env=${ENV_VARS[*]}" > $LOG
TOTAL=$(grep -c . $ARGS_FILE)
DONE=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    CELL=$(echo "$line" | grep -oP '(?<=--cell_id )\S+')
    DONE=$((DONE+1))
    if [ -e "$OUT/cell_$CELL/summary.json" ]; then
        echo "[$(date)] SKIP $CELL ($DONE/$TOTAL)" >> $LOG
        continue
    fi
    echo "[$(date)] BEGIN $CELL ($DONE/$TOTAL)" >> $LOG
    env "${ENV_VARS[@]}" $PY scripts/z272_sweep_cell_gpu.py $line --out_dir $OUT \
      --surrogate results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz \
      >> $LOG 2>&1
done < $ARGS_FILE
echo "[$(date)] node=$NODE DONE ($DONE/$TOTAL processed)" >> $LOG
