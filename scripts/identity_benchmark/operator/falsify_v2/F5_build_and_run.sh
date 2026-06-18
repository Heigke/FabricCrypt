#!/bin/bash
# F5 — build 5 variants, run each locally on ikaros (and via SCP+ssh on daedalus).
# Each variant isolates one source of non-determinism.
set -u
export HSA_OVERRIDE_GFX_VERSION=11.0.0
cd "$(dirname "$0")"
mkdir -p ../../../../results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F5
OUT=../../../../results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F5

# Variants
# F5a: atomics OFF only        -> -DF5_DISABLE_ATOMICS -ffast-math
# F5b: FMA OFF only            -> -DF5_DISABLE_FMA -ffast-math
# F5c: denormals strict only   -> -ffast-math but with daz_opt_off (handled by builtin bitcode default off)
#      We approximate by using -fno-fast-math but enabling reassoc... simplest: -ffp-contract=on -fno-fast-math
# F5d: -ffast-math OFF only    -> default (no extra flag), keep atomics+fma enabled
# F5e: all OFF (= F1 baseline) -> -DF5_DISABLE_ATOMICS -DF5_DISABLE_FMA -fno-fast-math

declare -A VARIANTS=(
  [F5a_no_atomics]="-DF5_DISABLE_ATOMICS -O3 -ffast-math"
  [F5b_no_fma]="-DF5_DISABLE_FMA -O3 -ffast-math"
  [F5c_strict_denorm]="-O3 -fno-fast-math -fno-finite-math-only -ffp-contract=on"
  [F5d_no_fast_math]="-O3 -fno-fast-math"
  [F5e_all_off]="-DF5_DISABLE_ATOMICS -DF5_DISABLE_FMA -O3 -fno-fast-math -fno-finite-math-only -ffp-contract=on"
)

HOST=$(hostname)
echo "[F5] host=$HOST"
for name in "${!VARIANTS[@]}"; do
  flags="${VARIANTS[$name]}"
  bin="$OUT/${name}_bin"
  binout="$OUT/${name}_${HOST}.bin"
  echo "[F5] building $name with: $flags"
  if hipcc --offload-arch=gfx1151 $flags -o "$bin" F5_flag_isolation.hip 2> "$OUT/${name}_build.log"; then
    if "$bin" 64 4096 32 "$binout" 2> "$OUT/${name}_${HOST}.log"; then
      echo "[F5]   ran $name ok"
    else
      echo "[F5]   RUN FAIL $name"
    fi
  else
    echo "[F5]   BUILD FAIL $name -- see $OUT/${name}_build.log"
  fi
done
echo "[F5] done"
