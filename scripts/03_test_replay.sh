#!/usr/bin/env bash
# 03_test_replay.sh — run the 7-attack spoof suite.
#
# Requires a trained T3 classifier (data/<host>_t3_best.pt).
# Train it once with:
#   PYTHONPATH=. python -m src.protocol.train --n_train 400 \
#       --peer_npz data/<peer>_paired_sigs.npz
#
# Usage:
#   ./scripts/03_test_replay.sh
#   ./scripts/03_test_replay.sh --peer_npz data/peer_paired_sigs.npz
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || true

PYTHONPATH="." python -m src.protocol.attacks \
    --n_eval 200 --out_dir data "$@"
