#!/usr/bin/env bash
# Reproduction script: bring up the demo on the current machine.
#
# Prerequisites:
#   - AMD Strix Halo (gfx1151) machine, same kernel/governor as peer
#   - venv at ./venv with fastapi, uvicorn, torch installed
#   - cpufreq governor pinned to 'performance' (see below)
#   - Phase 14B sig dumps for own + peer:
#       results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/<host>_sigs.npz
#
# Usage:
#   ./scripts/demo_embodied_ai/reproduce.sh ikaros 8770
#   ./scripts/demo_embodied_ai/reproduce.sh daedalus 8770
set -euo pipefail

HOSTN="${1:-ikaros}"
PORT="${2:-8770}"

case "$HOSTN" in
  ikaros)   PEER="daedalus" ;;
  daedalus) PEER="ikaros" ;;
  *) echo "host must be 'ikaros' or 'daedalus'"; exit 2 ;;
esac

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

OWN="results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/${HOSTN}_sigs.npz"
PEER_SIGS="results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/${PEER}_sigs.npz"

if [[ ! -f "$OWN" ]]; then
  echo "Missing own sigs: $OWN"
  echo "Generate via:  venv/bin/python scripts/identity_benchmark/embodiment14b/train_and_eval.py --also_dump_sigs"
  exit 3
fi
if [[ ! -f "$PEER_SIGS" ]]; then
  echo "WARN: missing peer sigs ($PEER_SIGS) — transplant demo will be disabled."
fi

# Governor confound control (optional but recommended)
if command -v cpupower >/dev/null 2>&1; then
  sudo cpupower frequency-set -g performance >/dev/null 2>&1 || true
fi

export HSA_OVERRIDE_GFX_VERSION=11.0.0
exec venv/bin/python scripts/demo_embodied_ai/demo_server.py \
  --host-name "$HOSTN" --port "$PORT" \
  --own-sigs "$OWN" --peer-sigs "$PEER_SIGS"
