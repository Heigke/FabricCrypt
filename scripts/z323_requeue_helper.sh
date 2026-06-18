#!/usr/bin/env bash
# Background helper: when a z323/z323b job fails because it landed on zgx (no
# nsram/ dir, my host gate fires rc=2), move it back to pending so daedalus or
# ikaros can pick it up. Max 5 retries per job.
set -u
QDIR=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/research_plan/job_queue
START=$(date +%s)
WALL=$((3*3600 + 600))
declare -A RETRIES

while true; do
  NOW=$(date +%s); ELAPSED=$((NOW-START))
  if [ $ELAPSED -gt $WALL ]; then
    echo "[requeue] wall reached, exiting" >&2
    break
  fi
  for f in $QDIR/failed/z323_k1s_*.json $QDIR/failed/z323b_k1s_*.json; do
    [ -e "$f" ] || continue
    jid=$(basename "$f" .json)
    n=${RETRIES[$jid]:-0}
    if [ "$n" -ge 5 ]; then
      continue
    fi
    err=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('error','')+'\n'+(d.get('log_tail','')[:1500]))" 2>/dev/null)
    if echo "$err" | grep -qE "rc=2|missing nsram|Permission denied: '/home/ikaros'"; then
      RETRIES[$jid]=$((n+1))
      mv "$f" "$QDIR/pending/${jid}.json" 2>/dev/null && \
        echo "[requeue] $jid moved back to pending (retry ${RETRIES[$jid]})" >&2
    fi
  done
  # Exit when all 5 partials exist
  np=$(ls /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z323_v5_extended/partial_*.json 2>/dev/null | wc -l)
  if [ "$np" -ge 5 ]; then
    echo "[requeue] 5 partials present, exiting" >&2
    break
  fi
  sleep 25
done
