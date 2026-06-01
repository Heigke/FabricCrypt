#!/usr/bin/env bash
# 04_demo.sh — launch the interactive demo server on http://127.0.0.1:8770
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate 2>/dev/null || true

PYTHONPATH="." python -m src.demo.server --host 127.0.0.1 --port 8770 "$@"
