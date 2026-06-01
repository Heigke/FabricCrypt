#!/usr/bin/env bash
# 01_collect_signature.sh — collect a per-die 290-dim signature on this machine.
# Outputs data/<hostname>_sig_v2.npz + meta JSON.
#
# Usage:
#   ./scripts/01_collect_signature.sh                 # 10 reps, host=hostname
#   ./scripts/01_collect_signature.sh --reps 20
#   ./scripts/01_collect_signature.sh --host alice    # explicit label
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || true

REPS=10
HOST="$(cat /etc/hostname 2>/dev/null || hostname)"
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --reps)  REPS="$2"; shift 2;;
        --host)  HOST="$2"; shift 2;;
        *) EXTRA+=("$1"); shift;;
    esac
done

mkdir -p data
echo "[collect] host=$HOST reps=$REPS"
PYTHONPATH="." python -m src.signature.signature_v2 \
    --reps "$REPS" --host "$HOST" --out_dir data "${EXTRA[@]}"
echo "[collect] DONE -> data/${HOST}_sig_v2.npz"
