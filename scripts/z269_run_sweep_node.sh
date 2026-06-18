#!/bin/bash
# Usage: run_sweep_node.sh <node_name> <args_file> <python> <env_prefix>
set -u
NODE=$1
ARGS_FILE=$2
PY=$3
ENVP=$4
OUT=results/sweep_v1
mkdir -p $OUT/_logs
LOG=$OUT/_logs/${NODE}_$(date +%H%M).log
echo "[$(date)] node=$NODE starting" > $LOG
while IFS= read -r line; do
    [ -z "$line" ] && continue
    CELL=$(echo "$line" | grep -oP '(?<=--cell_id )\S+')
    if [ -e "$OUT/cell_$CELL/summary.json" ]; then
        echo "[$(date)] SKIP $CELL" >> $LOG
        continue
    fi
    echo "[$(date)] BEGIN $CELL" >> $LOG
    $ENVP $PY scripts/z269_sweep_cell.py $line --out_dir $OUT >> $LOG 2>&1
    echo "[$(date)] END $CELL" >> $LOG
done < $ARGS_FILE
echo "[$(date)] node=$NODE DONE" >> $LOG
