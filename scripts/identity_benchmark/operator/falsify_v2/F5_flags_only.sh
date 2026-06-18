#!/bin/bash
# F5 v2 — use the ORIGINAL divergent_matmul.hip kernel verbatim and only
# vary the compile flags (no source-level changes). The original kernel
# is known to be runtime-stable on gfx1151. We can isolate ONLY the
# build-flag axis (fast-math vs strict, finite vs non-finite, fp-contract).
# We cannot disable the FMA __fmaf_rn intrinsic this way (it's a builtin),
# nor disable atomics (also a builtin), but we can measure whether the
# operator-substrate signal survives strict math.
set -u
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd "$(dirname "$0")"
OUT=../../../../results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F5flags
mkdir -p "$OUT"

declare -A VARIANTS=(
  [F5f_strict_default]="-O3 -fno-fast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=on -fdenormal-fp-math=ieee"
  [F5g_ffp_off]="-O3 -ffast-math -ffp-contract=off"
  [F5h_no_unsafe]="-O3 -fno-unsafe-math-optimizations -ffp-contract=fast"
  [F5i_ffast_math_only]="-O3 -ffast-math"
)
HOST=$(hostname)
SRC=../divergent_matmul.hip
echo "[F5v2] host=$HOST"
for name in "${!VARIANTS[@]}"; do
  flags="${VARIANTS[$name]}"
  bin="$OUT/${name}_bin"
  binout="$OUT/${name}_${HOST}.bin"
  echo "[F5v2] building $name : $flags"
  if hipcc --offload-arch=gfx1151 $flags -o "$bin" "$SRC" 2> "$OUT/${name}_build.log"; then
    HSA_OVERRIDE_GFX_VERSION=11.0.0 "$bin" 64 4096 32 "$binout" 2> "$OUT/${name}_${HOST}.log"
    size=$(stat -c%s "$binout" 2>/dev/null)
    echo "[F5v2]   ran $name : size=$size"
  else
    echo "[F5v2]   BUILD FAIL $name (see $OUT/${name}_build.log)"
  fi
done
echo "[F5v2] done"
