#!/bin/bash
# Collect daedalus robust signature ONLY (no ikaros impact).
set -u
ROOT=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
HOST="${DAEDALUS_HOST:-daedalus.local}"
USER="${DAEDALUS_USER:-daedalus}"
PASS="${DAEDALUS_PASS:-daedalus}"
RPY="/home/daedalus/venvs/torch-rocm/bin/python"
LOCAL_OUT="$ROOT/results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/signatures/daedalus_v2d.json"
REMOTE_OUT="/tmp/daedalus_v2d.json"
REMOTE_PY="/tmp/robust_signature.py"
N="${1:-60}"

mkdir -p $(dirname "$LOCAL_OUT")
sshpass -p "$PASS" scp -o StrictHostKeyChecking=no \
    "$ROOT/scripts/identity_benchmark/embodiment3/robust_signature.py" \
    "$USER@$HOST:$REMOTE_PY"
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$USER@$HOST" \
    "$RPY $REMOTE_PY --N $N --interval 0.5 --label daedalus --out $REMOTE_OUT"
sshpass -p "$PASS" scp -o StrictHostKeyChecking=no \
    "$USER@$HOST:$REMOTE_OUT" "$LOCAL_OUT"
echo "DONE: $LOCAL_OUT"
ls -la "$LOCAL_OUT"
