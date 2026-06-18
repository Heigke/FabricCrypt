#!/bin/bash
# Build the GPU-FPGA persistent bridge
# Requirements: ROCm HIP compiler, gfx1100 target (gfx1151 uses HSA override)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "[BUILD] Compiling GPU-FPGA bridge (HIP + host UDP)..."
echo "[BUILD] Target: gfx1100 (gfx1151 with HSA_OVERRIDE_GFX_VERSION=11.0.0)"

# Compile HIP program (includes both GPU kernel and host bridge)
hipcc --offload-arch=gfx1100 -O2 \
    -o scripts/gpu_fpga_bridge \
    scripts/gpu_fpga_bridge.hip \
    -lpthread

echo "[BUILD] Success: scripts/gpu_fpga_bridge"
echo "[BUILD] Run with: HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/gpu_fpga_bridge --help"

ls -la scripts/gpu_fpga_bridge
