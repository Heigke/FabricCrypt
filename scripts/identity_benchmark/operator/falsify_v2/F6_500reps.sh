#!/bin/bash
# F6 — 500-rep capture for bootstrap CI.
set -u
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd "$(dirname "$0")"
OUT=../../../../results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F6
mkdir -p "$OUT"

BIN=../divergent_matmul
if [ ! -x "$BIN" ]; then
  echo "[F6] building divergent_matmul"
  hipcc --offload-arch=gfx1151 -O3 -ffast-math \
    -o "$BIN" ../divergent_matmul.hip || exit 1
fi

HOST=$(hostname)
echo "[F6] host=$HOST running 500 reps"
"$BIN" 64 4096 500 "$OUT/${HOST}_500.bin" 2> "$OUT/${HOST}.log"
echo "[F6] done"
