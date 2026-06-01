#!/usr/bin/env bash
# 02_classify.sh — leave-one-out cross-chassis classification.
# Usage:
#   ./scripts/02_classify.sh data/hostA_sig_v2.npz data/hostB_sig_v2.npz
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || true

if [ $# -lt 2 ]; then
    echo "Usage: $0 <sig1.npz> <sig2.npz> [<sig3.npz> ...]"
    exit 2
fi
PYTHONPATH="." python -m src.analysis.compare_chassis \
    "$@" --out_json data/loo_classification.json
