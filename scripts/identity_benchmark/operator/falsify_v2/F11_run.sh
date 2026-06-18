#!/bin/bash
set -u
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd "$(dirname "$0")"
OUT=../../../../results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F11
mkdir -p "$OUT"
BIN="$OUT/f11_bin"
hipcc --offload-arch=gfx1151 -O3 -ffast-math -o "$BIN" F11_adversarial.hip 2> "$OUT/build.log" || { echo "[F11] build fail"; exit 1; }
HOST=$(hostname)
"$BIN" 64 4096 32 "$OUT/${HOST}_f11.bin" 2> "$OUT/${HOST}.log"
echo "[F11] done host=$HOST"
