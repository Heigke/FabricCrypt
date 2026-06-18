#!/bin/bash
# F9 — time stability. Snapshot now, wait 60+ min, snapshot again.
# Compare modal values.
set -u
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd "$(dirname "$0")"
OUT=../../../../results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F9
mkdir -p "$OUT"
BIN=../divergent_matmul
HOST=$(hostname)
TAG="${1:-t0}"  # t0 or t1
"$BIN" 64 4096 32 "$OUT/${HOST}_${TAG}.bin" 2> "$OUT/${HOST}_${TAG}.log"
echo "[F9] snapshot $TAG done"
