#!/usr/bin/env python3
"""
z2336_trap_ring_probe.py — Trap Handler + Ring Buffer Exploration (Våning 2)
=============================================================================
Comprehensive probing of everything between HIP and firmware on gfx1151 (RDNA4).

PART A: Trap Handler Probing (from inside HIP kernels)
  A1. s_getreg sweep — all 32 HWREG IDs, full 32-bit reads
  A2. Known HWREG decode — STATUS, MODE, TRAPSTS, GPR_ALLOC, LDS_ALLOC, etc.
  A3. s_memtime / s_memrealtime — alternative clock sources
  A4. EXEC mask probing — initial mask, save-exec patterns
  A5. HW_REG_SHADER_CYCLES — gfx11+ cycle counter, variance analysis
  A6. s_sendmsg probing — safe message types only

PART B: Ring Buffer / CP Probing (from host, debugfs/sysfs)
  B1. Ring buffer state from debugfs
  B2. Wave status scanning
  B3. GPU firmware version / ucode info

PART C: Reservoir Feature Test
  C1. SHADER_CYCLES as reservoir feature — if it works, mini wave4 benchmark

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 ./venv/bin/python scripts/z2336_trap_ring_probe.py
"""

import os, sys, time, json, struct, glob, subprocess
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2336_trap_ring_probe.json'

TEMP_PAUSE = 75
TEMP_RESUME = 50
TEMP_ABORT = 85

# ======================================================================
# Thermal Safety
# ======================================================================
def get_temp():
    try: return int(open('/sys/class/thermal/thermal_zone0/temp').read()) // 1000
    except: return 0

def check_abort():
    t = get_temp()
    if t >= TEMP_ABORT:
        print(f"\n  [ABORT] Temperature {t}C >= {TEMP_ABORT}C! Saving and exiting.", flush=True)
        return True
    return False

def wait_cool(label="", target=None):
    if target is None: target = TEMP_RESUME
    t = get_temp()
    if t <= target: return t
    print(f"  [TEMP] {label} {t}C -> {target}C...", end="", flush=True)
    t0 = time.time()
    while t > target and time.time() - t0 < 180:
        time.sleep(3); t = get_temp()
        print(f" {t}", end="", flush=True)
    print(f" OK ({time.time()-t0:.0f}s)")
    return t

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

results = {
    'experiment': 'z2336_trap_ring_probe',
    'description': 'Trap handler + ring buffer exploration on gfx1151 RDNA4',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'parts': {}
}

def save_results():
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)

def save_intermediate(part_name, data, txt_suffix=None):
    """Save intermediate results to both JSON and optional txt file."""
    results['parts'][part_name] = data
    save_results()
    if txt_suffix:
        txt_file = RESULTS / f'z2336_{txt_suffix}.txt'
        with open(txt_file, 'w') as f:
            f.write(f"=== {part_name} ===\n")
            f.write(json.dumps(data, indent=2, cls=NpEncoder))
            f.write('\n')
        print(f"  [SAVED] {txt_file}", flush=True)


# ======================================================================
# PART A: HIP Kernel Probes
# ======================================================================
print("=" * 70)
print("z2336: Trap Handler + Ring Buffer Exploration — gfx1151 RDNA4")
print("=" * 70)

import torch
from torch.utils.cpp_extension import load_inline

# ------ A1 + A2 + A3 + A4 + A5: Comprehensive HWREG + Clock Probes ------

PROBE_SRC = r'''
#include <torch/extension.h>

#define HWREG(id, offset, size) ((id) | ((offset) << 6) | (((size)-1) << 11))

// Known HWREG IDs for RDNA3/4 (gfx11)
#define HW_REG_MODE         0
#define HW_REG_STATUS       1
#define HW_REG_TRAPSTS      2
#define HW_REG_HW_ID        3
#define HW_REG_GPR_ALLOC    4
#define HW_REG_LDS_ALLOC    5
#define HW_REG_IB_STS       6
#define HW_REG_SH_MEM_BASES 14
#define HW_REG_TBA_LO       15
#define HW_REG_TBA_HI       16
#define HW_REG_TMA_LO       17
#define HW_REG_TMA_HI       18
#define HW_REG_FLAT_SCR_LO  20
#define HW_REG_FLAT_SCR_HI  21
#define HW_REG_HW_ID1       23
#define HW_REG_HW_ID2       24
#define HW_REG_SHADER_CYCLES 29

// =====================================================================
// A1: Full HWREG sweep — read all 32 register IDs (full 32 bits each)
// Output: [32] uint32 values, one per register ID
// =====================================================================
__global__ void kernel_hwreg_sweep(unsigned int* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;  // only one thread reads

    // We must use compile-time constants for s_getreg, so unroll manually
    out[0]  = __builtin_amdgcn_s_getreg(HWREG(0,  0, 32));
    out[1]  = __builtin_amdgcn_s_getreg(HWREG(1,  0, 32));
    out[2]  = __builtin_amdgcn_s_getreg(HWREG(2,  0, 32));
    out[3]  = __builtin_amdgcn_s_getreg(HWREG(3,  0, 32));
    out[4]  = __builtin_amdgcn_s_getreg(HWREG(4,  0, 32));
    out[5]  = __builtin_amdgcn_s_getreg(HWREG(5,  0, 32));
    out[6]  = __builtin_amdgcn_s_getreg(HWREG(6,  0, 32));
    out[7]  = __builtin_amdgcn_s_getreg(HWREG(7,  0, 32));
    out[8]  = __builtin_amdgcn_s_getreg(HWREG(8,  0, 32));
    out[9]  = __builtin_amdgcn_s_getreg(HWREG(9,  0, 32));
    out[10] = __builtin_amdgcn_s_getreg(HWREG(10, 0, 32));
    out[11] = __builtin_amdgcn_s_getreg(HWREG(11, 0, 32));
    out[12] = __builtin_amdgcn_s_getreg(HWREG(12, 0, 32));
    out[13] = __builtin_amdgcn_s_getreg(HWREG(13, 0, 32));
    out[14] = __builtin_amdgcn_s_getreg(HWREG(14, 0, 32));
    out[15] = __builtin_amdgcn_s_getreg(HWREG(15, 0, 32));
    out[16] = __builtin_amdgcn_s_getreg(HWREG(16, 0, 32));
    out[17] = __builtin_amdgcn_s_getreg(HWREG(17, 0, 32));
    out[18] = __builtin_amdgcn_s_getreg(HWREG(18, 0, 32));
    out[19] = __builtin_amdgcn_s_getreg(HWREG(19, 0, 32));
    out[20] = __builtin_amdgcn_s_getreg(HWREG(20, 0, 32));
    out[21] = __builtin_amdgcn_s_getreg(HWREG(21, 0, 32));
    out[22] = __builtin_amdgcn_s_getreg(HWREG(22, 0, 32));
    out[23] = __builtin_amdgcn_s_getreg(HWREG(23, 0, 32));
    out[24] = __builtin_amdgcn_s_getreg(HWREG(24, 0, 32));
    out[25] = __builtin_amdgcn_s_getreg(HWREG(25, 0, 32));
    out[26] = __builtin_amdgcn_s_getreg(HWREG(26, 0, 32));
    out[27] = __builtin_amdgcn_s_getreg(HWREG(27, 0, 32));
    out[28] = __builtin_amdgcn_s_getreg(HWREG(28, 0, 32));
    out[29] = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));
    out[30] = __builtin_amdgcn_s_getreg(HWREG(30, 0, 32));
    out[31] = __builtin_amdgcn_s_getreg(HWREG(31, 0, 32));
}

torch::Tensor probe_hwreg_sweep() {
    auto out = torch::zeros({32}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_hwreg_sweep<<<1, 1>>>((unsigned int*)out.data_ptr<int>());
    return out;
}

// =====================================================================
// A1b: Multi-wave HWREG sweep — run on MANY waves to see per-CU variation
// Output: [n_waves, 34] — 32 HWREG vals + wgp + simd per wave
// =====================================================================
__global__ void kernel_hwreg_multiwave(unsigned int* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    // Only lane 0 of each wavefront reads (all lanes see same SGPR values)
    if (threadIdx.x % 32 != 0) return;
    int wave_id = tid / 32;

    int base = wave_id * 34;
    out[base + 0]  = __builtin_amdgcn_s_getreg(HWREG(0,  0, 32));
    out[base + 1]  = __builtin_amdgcn_s_getreg(HWREG(1,  0, 32));
    out[base + 2]  = __builtin_amdgcn_s_getreg(HWREG(2,  0, 32));
    out[base + 3]  = __builtin_amdgcn_s_getreg(HWREG(3,  0, 32));
    out[base + 4]  = __builtin_amdgcn_s_getreg(HWREG(4,  0, 32));
    out[base + 5]  = __builtin_amdgcn_s_getreg(HWREG(5,  0, 32));
    out[base + 6]  = __builtin_amdgcn_s_getreg(HWREG(6,  0, 32));
    out[base + 7]  = __builtin_amdgcn_s_getreg(HWREG(7,  0, 32));
    out[base + 8]  = __builtin_amdgcn_s_getreg(HWREG(8,  0, 32));
    out[base + 9]  = __builtin_amdgcn_s_getreg(HWREG(9,  0, 32));
    out[base + 10] = __builtin_amdgcn_s_getreg(HWREG(10, 0, 32));
    out[base + 11] = __builtin_amdgcn_s_getreg(HWREG(11, 0, 32));
    out[base + 12] = __builtin_amdgcn_s_getreg(HWREG(12, 0, 32));
    out[base + 13] = __builtin_amdgcn_s_getreg(HWREG(13, 0, 32));
    out[base + 14] = __builtin_amdgcn_s_getreg(HWREG(14, 0, 32));
    out[base + 15] = __builtin_amdgcn_s_getreg(HWREG(15, 0, 32));
    out[base + 16] = __builtin_amdgcn_s_getreg(HWREG(16, 0, 32));
    out[base + 17] = __builtin_amdgcn_s_getreg(HWREG(17, 0, 32));
    out[base + 18] = __builtin_amdgcn_s_getreg(HWREG(18, 0, 32));
    out[base + 19] = __builtin_amdgcn_s_getreg(HWREG(19, 0, 32));
    out[base + 20] = __builtin_amdgcn_s_getreg(HWREG(20, 0, 32));
    out[base + 21] = __builtin_amdgcn_s_getreg(HWREG(21, 0, 32));
    out[base + 22] = __builtin_amdgcn_s_getreg(HWREG(22, 0, 32));
    out[base + 23] = __builtin_amdgcn_s_getreg(HWREG(23, 0, 32));
    out[base + 24] = __builtin_amdgcn_s_getreg(HWREG(24, 0, 32));
    out[base + 25] = __builtin_amdgcn_s_getreg(HWREG(25, 0, 32));
    out[base + 26] = __builtin_amdgcn_s_getreg(HWREG(26, 0, 32));
    out[base + 27] = __builtin_amdgcn_s_getreg(HWREG(27, 0, 32));
    out[base + 28] = __builtin_amdgcn_s_getreg(HWREG(28, 0, 32));
    out[base + 29] = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));
    out[base + 30] = __builtin_amdgcn_s_getreg(HWREG(30, 0, 32));
    out[base + 31] = __builtin_amdgcn_s_getreg(HWREG(31, 0, 32));
    // HW_ID1 and HW_ID2 decoded
    out[base + 32] = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    out[base + 33] = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID2, 0, 32));
}

torch::Tensor probe_hwreg_multiwave(int n_waves) {
    int n = n_waves * 32;  // threads
    auto out = torch::zeros({n_waves * 34}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_hwreg_multiwave<<<n_waves, 32>>>((unsigned int*)out.data_ptr<int>(), n);
    return out.reshape({n_waves, 34});
}


// =====================================================================
// A2: Detailed HWREG bitfield decode — known registers with sub-fields
// Output: [N_FIELDS] — decoded sub-fields
// =====================================================================
__global__ void kernel_hwreg_decode(unsigned int* out) {
    if (threadIdx.x != 0) return;

    // MODE register (id=0) sub-fields
    out[0] = __builtin_amdgcn_s_getreg(HWREG(0, 0, 4));   // FP_ROUND  [3:0]
    out[1] = __builtin_amdgcn_s_getreg(HWREG(0, 4, 4));   // FP_DENORM [7:4]
    out[2] = __builtin_amdgcn_s_getreg(HWREG(0, 8, 1));   // DX10_CLAMP [8]
    out[3] = __builtin_amdgcn_s_getreg(HWREG(0, 9, 1));   // IEEE [9]
    out[4] = __builtin_amdgcn_s_getreg(HWREG(0, 10, 1));  // LOD_CLAMPED [10]
    out[5] = __builtin_amdgcn_s_getreg(HWREG(0, 11, 1));  // DEBUG [11]
    out[6] = __builtin_amdgcn_s_getreg(HWREG(0, 12, 3));  // EXCP_EN [14:12]
    out[7] = __builtin_amdgcn_s_getreg(HWREG(0, 15, 9));  // MODE[23:15] reserved?
    out[8] = __builtin_amdgcn_s_getreg(HWREG(0, 24, 4));  // FP16_OVFL [27:24]
    out[9] = __builtin_amdgcn_s_getreg(HWREG(0, 29, 1));  // DISABLE_PERF [29]

    // STATUS register (id=1) sub-fields
    out[10] = __builtin_amdgcn_s_getreg(HWREG(1, 0, 1));   // SCC [0]
    out[11] = __builtin_amdgcn_s_getreg(HWREG(1, 1, 1));   // SPI_PRIO [2:1] bit 1
    out[12] = __builtin_amdgcn_s_getreg(HWREG(1, 1, 2));   // SPI_PRIO [2:1] both bits
    out[13] = __builtin_amdgcn_s_getreg(HWREG(1, 3, 1));   // USER_PRIO [4:3] bit
    out[14] = __builtin_amdgcn_s_getreg(HWREG(1, 5, 1));   // PRIV [5]
    out[15] = __builtin_amdgcn_s_getreg(HWREG(1, 6, 1));   // TRAP_EN [6]
    out[16] = __builtin_amdgcn_s_getreg(HWREG(1, 7, 1));   // TTRACE_EN [7]
    out[17] = __builtin_amdgcn_s_getreg(HWREG(1, 8, 1));   // EXPORT_RDY [8]
    out[18] = __builtin_amdgcn_s_getreg(HWREG(1, 9, 1));   // EXECZ [9]
    out[19] = __builtin_amdgcn_s_getreg(HWREG(1, 10, 1));  // VCCZ [10]
    out[20] = __builtin_amdgcn_s_getreg(HWREG(1, 11, 1));  // IN_TG [11]
    out[21] = __builtin_amdgcn_s_getreg(HWREG(1, 12, 1));  // IN_BARRIER [12]
    out[22] = __builtin_amdgcn_s_getreg(HWREG(1, 13, 1));  // HALT [13]
    out[23] = __builtin_amdgcn_s_getreg(HWREG(1, 14, 1));  // TRAP [14]
    out[24] = __builtin_amdgcn_s_getreg(HWREG(1, 15, 1));  // TTRACE_CU_EN [15]
    out[25] = __builtin_amdgcn_s_getreg(HWREG(1, 16, 1));  // VALID [16]
    out[26] = __builtin_amdgcn_s_getreg(HWREG(1, 17, 1));  // ECC_ERR [17]
    out[27] = __builtin_amdgcn_s_getreg(HWREG(1, 18, 1));  // SKIP_EXPORT [18]
    out[28] = __builtin_amdgcn_s_getreg(HWREG(1, 19, 1));  // PERF_EN [19]
    out[29] = __builtin_amdgcn_s_getreg(HWREG(1, 20, 1));  // COND_DBG_USER [20]
    out[30] = __builtin_amdgcn_s_getreg(HWREG(1, 21, 1));  // COND_DBG_SYS [21]
    out[31] = __builtin_amdgcn_s_getreg(HWREG(1, 22, 1));  // ALLOW_REPLAY [22]

    // TRAPSTS register (id=2) — exception flags
    out[32] = __builtin_amdgcn_s_getreg(HWREG(2, 0, 1));   // EXCP_INVALID [0]
    out[33] = __builtin_amdgcn_s_getreg(HWREG(2, 1, 1));   // EXCP_INPUT_DENORM [1]
    out[34] = __builtin_amdgcn_s_getreg(HWREG(2, 2, 1));   // EXCP_DIV0 [2]
    out[35] = __builtin_amdgcn_s_getreg(HWREG(2, 3, 1));   // EXCP_OVERFLOW [3]
    out[36] = __builtin_amdgcn_s_getreg(HWREG(2, 4, 1));   // EXCP_UNDERFLOW [4]
    out[37] = __builtin_amdgcn_s_getreg(HWREG(2, 5, 1));   // EXCP_INEXACT [5]
    out[38] = __builtin_amdgcn_s_getreg(HWREG(2, 6, 1));   // EXCP_INT_DIV0 [6]
    out[39] = __builtin_amdgcn_s_getreg(HWREG(2, 7, 1));   // EXCP_ADDR_WATCH0 [7]
    out[40] = __builtin_amdgcn_s_getreg(HWREG(2, 8, 1));   // EXCP_MEM_VIOL [8]
    out[41] = __builtin_amdgcn_s_getreg(HWREG(2, 10, 1));  // SAVE_CTX [10]
    out[42] = __builtin_amdgcn_s_getreg(HWREG(2, 11, 3));  // ILLEGAL_INST [13:11]
    out[43] = __builtin_amdgcn_s_getreg(HWREG(2, 14, 1));  // EXCP_HI [14]
    out[44] = __builtin_amdgcn_s_getreg(HWREG(2, 23, 1));  // XNACK_ERR [23]
    out[45] = __builtin_amdgcn_s_getreg(HWREG(2, 28, 1));  // HOST_TRAP [28]
    out[46] = __builtin_amdgcn_s_getreg(HWREG(2, 29, 1));  // WAVE_START [29]
    out[47] = __builtin_amdgcn_s_getreg(HWREG(2, 30, 1));  // WAVE_END [30]
    out[48] = __builtin_amdgcn_s_getreg(HWREG(2, 31, 1));  // TRAP_AFTER_INST [31]

    // GPR_ALLOC (id=4)
    out[49] = __builtin_amdgcn_s_getreg(HWREG(4, 0, 32));  // full
    // LDS_ALLOC (id=5)
    out[50] = __builtin_amdgcn_s_getreg(HWREG(5, 0, 32));  // full
    // IB_STS (id=6)
    out[51] = __builtin_amdgcn_s_getreg(HWREG(6, 0, 32));  // full

    // HW_ID1 (id=23) — detailed decode
    out[52] = __builtin_amdgcn_s_getreg(HWREG(23, 0, 4));   // WAVE_ID [3:0]
    out[53] = __builtin_amdgcn_s_getreg(HWREG(23, 4, 2));   // SIMD_ID [5:4]
    out[54] = __builtin_amdgcn_s_getreg(HWREG(23, 8, 4));   // WGP_ID [11:8] (was CU_ID in older ISAs)
    out[55] = __builtin_amdgcn_s_getreg(HWREG(23, 12, 1));  // SA_ID [12]
    out[56] = __builtin_amdgcn_s_getreg(HWREG(23, 13, 3));  // SE_ID [15:13]
    out[57] = __builtin_amdgcn_s_getreg(HWREG(23, 16, 4));  // QUEUE_ID [19:16] — which HW queue
    out[58] = __builtin_amdgcn_s_getreg(HWREG(23, 20, 4));  // STATE_ID [23:20] — VMID
    out[59] = __builtin_amdgcn_s_getreg(HWREG(23, 24, 8));  // reserved/undocumented [31:24]

    // HW_ID2 (id=24) — detailed decode
    out[60] = __builtin_amdgcn_s_getreg(HWREG(24, 0, 4));   // PIPE_ID [3:0]
    out[61] = __builtin_amdgcn_s_getreg(HWREG(24, 4, 4));   // QUEUE_ID2 [7:4]
    out[62] = __builtin_amdgcn_s_getreg(HWREG(24, 8, 4));   // ME_ID [11:8]
    out[63] = __builtin_amdgcn_s_getreg(HWREG(24, 0, 32));  // full

    // SHADER_CYCLES (id=29) — gfx11+
    out[64] = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));
}

torch::Tensor probe_hwreg_decode() {
    auto out = torch::zeros({65}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_hwreg_decode<<<1, 1>>>((unsigned int*)out.data_ptr<int>());
    return out;
}


// =====================================================================
// A3: Clock source comparison — wall_clock64, clock64, SHADER_CYCLES
// Note: s_memtime and s_memrealtime are NOT available on RDNA3/4 (gfx11+)
// Output per wave: [wall0, wall1, clk0, clk1, shader_cyc0, shader_cyc1, wgp, simd]
// Total: [n_waves, 8]
// =====================================================================
__global__ void kernel_clock_compare(int64_t* out, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    if (threadIdx.x % 32 != 0) return;
    int wave_id = tid / 32;

    // Warmup
    float x = 1.0f;
    for (int i = 0; i < 100; i++) x = fmaf(x, 0.9999f, 0.0001f);

    // Read all clocks BEFORE work
    long long wall0 = wall_clock64();
    long long clk0  = (long long)clock64();
    unsigned sc0    = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));

    // Do measured work
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = sinf(x) * cosf(x + 0.1f) + expf(-x * x);
        x = x * 1.0001f + 0.0001f;
    }

    // Read all clocks AFTER work
    long long wall1 = wall_clock64();
    long long clk1  = (long long)clock64();
    unsigned sc1    = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));

    unsigned hw1 = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    int wgp  = (hw1 >> 8) & 0xF;
    int simd = (hw1 >> 4) & 0x3;

    int base = wave_id * 8;
    out[base + 0] = wall0;
    out[base + 1] = wall1;
    out[base + 2] = clk0;
    out[base + 3] = clk1;
    out[base + 4] = (long long)sc0;
    out[base + 5] = (long long)sc1;
    out[base + 6] = (long long)wgp;
    out[base + 7] = (long long)simd;

    // Anti-optimization: write x somewhere
    if (wave_id == 0 && x < -1e30f) out[0] = __float_as_int(x);
}

torch::Tensor probe_clock_compare(int n_waves, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n_waves * 8}, torch::device(torch::kCUDA).dtype(torch::kLong));
    kernel_clock_compare<<<n_waves, 32>>>(out.data_ptr<int64_t>(), n_iters, n);
    return out.reshape({n_waves, 8});
}


// =====================================================================
// A4: EXEC mask probing — read initial EXEC, test save-exec patterns
// Output per wave: [exec_lo, exec_hi, wgp, simd, lane_count]
// =====================================================================
__global__ void kernel_exec_probe(int64_t* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    if (threadIdx.x % 32 != 0) return;
    int wave_id = tid / 32;

    // Use __ballot to see which lanes are active (reads EXEC implicitly)
    unsigned long long ballot = __ballot(1);  // all active lanes vote 1

    unsigned hw1 = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    int wgp  = (hw1 >> 8) & 0xF;
    int simd = (hw1 >> 4) & 0x3;

    // Count active lanes
    int active = __popcll(ballot);

    int base = wave_id * 5;
    out[base + 0] = (int64_t)(ballot & 0xFFFFFFFF);
    out[base + 1] = (int64_t)(ballot >> 32);
    out[base + 2] = (int64_t)wgp;
    out[base + 3] = (int64_t)simd;
    out[base + 4] = (int64_t)active;
}

torch::Tensor probe_exec_mask(int n_waves) {
    int n = n_waves * 32;
    auto out = torch::zeros({n_waves * 5}, torch::device(torch::kCUDA).dtype(torch::kLong));
    kernel_exec_probe<<<n_waves, 32>>>(out.data_ptr<int64_t>(), n);
    return out.reshape({n_waves, 5});
}


// =====================================================================
// A5: SHADER_CYCLES variance — multiple reads from many CUs
// Output: [n_waves, 4] — [shader_cyc_before, shader_cyc_after, wgp, simd]
// Runs 1000 iterations, measures shader cycle delta per CU
// =====================================================================
__global__ void kernel_shader_cycles_variance(unsigned int* out, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    if (threadIdx.x % 32 != 0) return;
    int wave_id = tid / 32;

    unsigned sc0 = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));

    float x = 1.0f;
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = fmaf(x, 0.9999f, 0.0001f);
        x = sinf(x);
    }

    unsigned sc1 = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));

    unsigned hw1 = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    int wgp  = (hw1 >> 8) & 0xF;
    int simd = (hw1 >> 4) & 0x3;

    int base = wave_id * 4;
    out[base + 0] = sc0;
    out[base + 1] = sc1;
    out[base + 2] = (unsigned)wgp;
    out[base + 3] = (unsigned)simd;

    // Anti-opt
    if (x < -1e30f) out[0] = __float_as_uint(x);
}

torch::Tensor probe_shader_cycles_variance(int n_waves, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n_waves * 4}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_shader_cycles_variance<<<n_waves, 32>>>((unsigned int*)out.data_ptr<int>(), n_iters, n);
    return out.reshape({n_waves, 4});
}


// =====================================================================
// A6: s_sendmsg probing — SAFE messages only (MSG_INTERRUPT, MSG_HALT_COND)
// We test if s_sendmsg triggers exceptions or silently works
// Output: [msg_id, status_before, status_after, trapsts_before, trapsts_after]
// Only test msg 1 (INTERRUPT) and 15 (SYSMSG) — others are graphics-specific
// =====================================================================
__global__ void kernel_sendmsg_probe(unsigned int* out) {
    if (threadIdx.x != 0) return;

    // Read status/trapsts before
    out[0] = __builtin_amdgcn_s_getreg(HWREG(1, 0, 32));  // STATUS before
    out[1] = __builtin_amdgcn_s_getreg(HWREG(2, 0, 32));  // TRAPSTS before

    // Try MSG_INTERRUPT (1) — should be safe, signals CP
    __builtin_amdgcn_s_sendmsg(1, 0);

    out[2] = __builtin_amdgcn_s_getreg(HWREG(1, 0, 32));  // STATUS after MSG_INTERRUPT
    out[3] = __builtin_amdgcn_s_getreg(HWREG(2, 0, 32));  // TRAPSTS after MSG_INTERRUPT

    // Don't try MSG_SAVEWAVE(4) or MSG_HALT_COND(7) or SYSMSG(15) — too risky for first probe
    // Just record what MSG_INTERRUPT did
    out[4] = 1;  // msg type tested
}

torch::Tensor probe_sendmsg() {
    auto out = torch::zeros({5}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_sendmsg_probe<<<1, 32>>>((unsigned int*)out.data_ptr<int>());
    return out;
}


// =====================================================================
// C1: SHADER_CYCLES as reservoir feature — mini wave4 with shader timing
// Each dispatch: N waves read shader_cycles delta after input-dependent work
// Output: [n_waves, 3] — [shader_delta, wall_delta, wgp]
// =====================================================================
__global__ void kernel_reservoir_shader(
    float* out,
    float input_val,
    int n_iters_base,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    if (threadIdx.x % 32 != 0) return;
    int wave_id = tid / 32;

    unsigned hw1 = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    int wgp = (hw1 >> 8) & 0xF;

    // Input-modulated work — different CUs get different amounts
    int iters = n_iters_base + (int)(50.0f * fabsf(input_val) * (float)(wgp + 1) / 16.0f);

    unsigned sc0 = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));
    long long w0 = wall_clock64();

    float x = input_val + (float)(wgp + 1) * 0.01f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        x = sinf(x) * cosf(x + 0.1f) + expf(-x * x);
        x = x * 1.0001f + 0.0001f;
    }

    unsigned sc1 = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));
    long long w1 = wall_clock64();

    int base = wave_id * 3;
    out[base + 0] = (float)(sc1 - sc0);
    out[base + 1] = (float)(w1 - w0) * 10.0f;
    out[base + 2] = (float)wgp;
}

torch::Tensor probe_reservoir_shader(float input_val, int n_waves, int n_iters_base) {
    int n = n_waves * 32;
    auto out = torch::zeros({n_waves * 3}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_reservoir_shader<<<n_waves, 32>>>(
        out.data_ptr<float>(), input_val, n_iters_base, n);
    return out.reshape({n_waves, 3});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_hwreg_sweep", &probe_hwreg_sweep);
    m.def("probe_hwreg_multiwave", &probe_hwreg_multiwave);
    m.def("probe_hwreg_decode", &probe_hwreg_decode);
    m.def("probe_clock_compare", &probe_clock_compare);
    m.def("probe_exec_mask", &probe_exec_mask);
    m.def("probe_shader_cycles_variance", &probe_shader_cycles_variance);
    m.def("probe_sendmsg", &probe_sendmsg);
    m.def("probe_reservoir_shader", &probe_reservoir_shader);
}
'''

print("\n[A0] Compiling HIP probe kernels...", flush=True)
try:
    probe_mod = load_inline(
        name='z2336_trap_probes',
        cpp_sources='',
        cuda_sources=PROBE_SRC,
        extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        verbose=False,
    )
    print("  [OK] Compilation succeeded", flush=True)
    results['compilation'] = 'success'
except Exception as e:
    print(f"  [FAIL] Compilation failed: {e}", flush=True)
    results['compilation'] = {'error': str(e)}
    save_results()
    sys.exit(1)

torch.cuda.synchronize()

# ======================================================================
# PART A1: HWREG Sweep (single wave)
# ======================================================================
print("\n" + "=" * 70)
print("[A1] HWREG Sweep — all 32 register IDs from single wave")
print("=" * 70, flush=True)

HWREG_NAMES = {
    0: 'HW_REG_MODE', 1: 'HW_REG_STATUS', 2: 'HW_REG_TRAPSTS',
    3: 'HW_REG_HW_ID', 4: 'HW_REG_GPR_ALLOC', 5: 'HW_REG_LDS_ALLOC',
    6: 'HW_REG_IB_STS', 7: 'UNKNOWN_7', 8: 'UNKNOWN_8', 9: 'UNKNOWN_9',
    10: 'UNKNOWN_10', 11: 'UNKNOWN_11', 12: 'UNKNOWN_12', 13: 'UNKNOWN_13',
    14: 'HW_REG_SH_MEM_BASES', 15: 'HW_REG_TBA_LO', 16: 'HW_REG_TBA_HI',
    17: 'HW_REG_TMA_LO', 18: 'HW_REG_TMA_HI', 19: 'UNKNOWN_19',
    20: 'HW_REG_FLAT_SCR_LO', 21: 'HW_REG_FLAT_SCR_HI', 22: 'UNKNOWN_22',
    23: 'HW_REG_HW_ID1', 24: 'HW_REG_HW_ID2', 25: 'UNKNOWN_25',
    26: 'UNKNOWN_26', 27: 'UNKNOWN_27', 28: 'UNKNOWN_28',
    29: 'HW_REG_SHADER_CYCLES', 30: 'UNKNOWN_30', 31: 'UNKNOWN_31',
}

try:
    wait_cool("A1")
    raw = probe_mod.probe_hwreg_sweep()
    torch.cuda.synchronize()
    vals = raw.cpu().numpy().astype(np.uint32)

    a1_data = {'registers': {}}
    for i in range(32):
        v = int(vals[i])
        name = HWREG_NAMES.get(i, f'UNKNOWN_{i}')
        a1_data['registers'][f'{i:2d}_{name}'] = {
            'dec': v, 'hex': f'0x{v:08X}', 'bin': f'{v:032b}'
        }
        nz = " *** NON-ZERO" if v != 0 else ""
        print(f"  HWREG[{i:2d}] {name:25s} = 0x{v:08X} ({v:10d}){nz}", flush=True)

    # Count non-zero
    nonzero = sum(1 for i in range(32) if vals[i] != 0)
    a1_data['nonzero_count'] = nonzero
    a1_data['status'] = 'success'
    print(f"\n  Non-zero registers: {nonzero}/32", flush=True)

    save_intermediate('A1_hwreg_sweep', a1_data, 'A1_hwreg_sweep')

except Exception as e:
    print(f"  [ERROR] A1 failed: {e}", flush=True)
    save_intermediate('A1_hwreg_sweep', {'status': 'error', 'error': str(e)}, 'A1_hwreg_sweep')


# ======================================================================
# PART A1b: Multi-wave HWREG Sweep (per-CU variation)
# ======================================================================
print("\n" + "=" * 70)
print("[A1b] Multi-wave HWREG Sweep — 64 waves, per-CU variation")
print("=" * 70, flush=True)

try:
    if check_abort(): raise SystemExit("thermal abort")
    wait_cool("A1b")

    N_WAVES = 64
    raw = probe_mod.probe_hwreg_multiwave(N_WAVES)
    torch.cuda.synchronize()
    data = raw.cpu().numpy().astype(np.uint32)  # [N_WAVES, 34]

    a1b_data = {'n_waves': N_WAVES, 'per_register_variation': {}}

    # For each HWREG, check if values vary across waves
    for reg_id in range(32):
        col = data[:, reg_id]
        unique_vals = np.unique(col)
        name = HWREG_NAMES.get(reg_id, f'UNKNOWN_{reg_id}')
        varies = len(unique_vals) > 1
        entry = {
            'name': name,
            'unique_count': int(len(unique_vals)),
            'varies_across_waves': varies,
            'min': int(col.min()), 'max': int(col.max()),
            'unique_values_hex': [f'0x{v:08X}' for v in unique_vals[:10]],
        }
        a1b_data['per_register_variation'][str(reg_id)] = entry
        if varies or col[0] != 0:
            print(f"  HWREG[{reg_id:2d}] {name:25s}: {len(unique_vals)} unique vals, "
                  f"range 0x{int(col.min()):08X}-0x{int(col.max()):08X}"
                  f"{' *** VARIES' if varies else ''}", flush=True)

    # Decode HW_ID1 per wave to show CU distribution
    hw_id1_col = data[:, 23]
    wgps = (hw_id1_col >> 8) & 0xF
    simds = (hw_id1_col >> 4) & 0x3
    ses = (hw_id1_col >> 13) & 0x7
    sas = (hw_id1_col >> 12) & 0x1
    a1b_data['hw_topology'] = {
        'unique_wgps': sorted([int(x) for x in np.unique(wgps)]),
        'unique_simds': sorted([int(x) for x in np.unique(simds)]),
        'unique_ses': sorted([int(x) for x in np.unique(ses)]),
        'unique_sas': sorted([int(x) for x in np.unique(sas)]),
        'n_unique_wgps': int(len(np.unique(wgps))),
    }
    print(f"\n  HW topology: WGPs={sorted(np.unique(wgps).tolist())}, "
          f"SIMDs={sorted(np.unique(simds).tolist())}, "
          f"SEs={sorted(np.unique(ses).tolist())}, "
          f"SAs={sorted(np.unique(sas).tolist())}", flush=True)

    a1b_data['status'] = 'success'
    save_intermediate('A1b_hwreg_multiwave', a1b_data, 'A1b_hwreg_multiwave')

except SystemExit:
    save_results(); sys.exit(1)
except Exception as e:
    print(f"  [ERROR] A1b failed: {e}", flush=True)
    save_intermediate('A1b_hwreg_multiwave', {'status': 'error', 'error': str(e)}, 'A1b_hwreg_multiwave')


# ======================================================================
# PART A2: Detailed HWREG Bitfield Decode
# ======================================================================
print("\n" + "=" * 70)
print("[A2] HWREG Bitfield Decode — STATUS, MODE, TRAPSTS sub-fields")
print("=" * 70, flush=True)

try:
    if check_abort(): raise SystemExit("thermal abort")
    wait_cool("A2")

    raw = probe_mod.probe_hwreg_decode()
    torch.cuda.synchronize()
    vals = raw.cpu().numpy().astype(np.uint32)

    a2_data = {}

    # MODE register decode
    mode_fields = [
        (0, 'FP_ROUND'), (1, 'FP_DENORM'), (2, 'DX10_CLAMP'), (3, 'IEEE'),
        (4, 'LOD_CLAMPED'), (5, 'DEBUG'), (6, 'EXCP_EN'), (7, 'MODE_RSVD'),
        (8, 'FP16_OVFL'), (9, 'DISABLE_PERF'),
    ]
    print("  MODE register:", flush=True)
    a2_data['MODE'] = {}
    for idx, name in mode_fields:
        v = int(vals[idx])
        a2_data['MODE'][name] = v
        if v != 0:
            print(f"    {name:20s} = {v} (0x{v:X})", flush=True)
        else:
            print(f"    {name:20s} = 0", flush=True)

    # STATUS register decode
    status_fields = [
        (10, 'SCC'), (11, 'SPI_PRIO_b1'), (12, 'SPI_PRIO'), (13, 'USER_PRIO'),
        (14, 'PRIV'), (15, 'TRAP_EN'), (16, 'TTRACE_EN'), (17, 'EXPORT_RDY'),
        (18, 'EXECZ'), (19, 'VCCZ'), (20, 'IN_TG'), (21, 'IN_BARRIER'),
        (22, 'HALT'), (23, 'TRAP'), (24, 'TTRACE_CU_EN'), (25, 'VALID'),
        (26, 'ECC_ERR'), (27, 'SKIP_EXPORT'), (28, 'PERF_EN'), (29, 'COND_DBG_USER'),
        (30, 'COND_DBG_SYS'), (31, 'ALLOW_REPLAY'),
    ]
    print("\n  STATUS register:", flush=True)
    a2_data['STATUS'] = {}
    for idx, name in status_fields:
        v = int(vals[idx])
        a2_data['STATUS'][name] = v
        marker = " ***" if v != 0 and name not in ('SCC', 'SPI_PRIO', 'VALID', 'IN_TG') else ""
        print(f"    {name:20s} = {v}{marker}", flush=True)

    # TRAPSTS register decode
    trapsts_fields = [
        (32, 'EXCP_INVALID'), (33, 'EXCP_INPUT_DENORM'), (34, 'EXCP_DIV0'),
        (35, 'EXCP_OVERFLOW'), (36, 'EXCP_UNDERFLOW'), (37, 'EXCP_INEXACT'),
        (38, 'EXCP_INT_DIV0'), (39, 'EXCP_ADDR_WATCH0'), (40, 'EXCP_MEM_VIOL'),
        (41, 'SAVE_CTX'), (42, 'ILLEGAL_INST'), (43, 'EXCP_HI'),
        (44, 'XNACK_ERR'), (45, 'HOST_TRAP'), (46, 'WAVE_START'),
        (47, 'WAVE_END'), (48, 'TRAP_AFTER_INST'),
    ]
    print("\n  TRAPSTS register:", flush=True)
    a2_data['TRAPSTS'] = {}
    for idx, name in trapsts_fields:
        v = int(vals[idx])
        a2_data['TRAPSTS'][name] = v
        marker = " *** ACTIVE" if v != 0 else ""
        print(f"    {name:20s} = {v}{marker}", flush=True)

    # GPR_ALLOC, LDS_ALLOC, IB_STS
    print(f"\n  GPR_ALLOC  = 0x{int(vals[49]):08X}", flush=True)
    print(f"  LDS_ALLOC  = 0x{int(vals[50]):08X}", flush=True)
    print(f"  IB_STS     = 0x{int(vals[51]):08X}", flush=True)
    a2_data['GPR_ALLOC'] = {'full': int(vals[49]), 'hex': f'0x{int(vals[49]):08X}'}
    a2_data['LDS_ALLOC'] = {'full': int(vals[50]), 'hex': f'0x{int(vals[50]):08X}'}
    a2_data['IB_STS']    = {'full': int(vals[51]), 'hex': f'0x{int(vals[51]):08X}'}

    # GPR_ALLOC decode (RDNA3/4): VGPR_BASE[5:0], VGPR_SIZE[13:8], SGPR_BASE[21:16], SGPR_SIZE[27:24]
    gpr = int(vals[49])
    vgpr_base = gpr & 0x3F
    vgpr_size = (gpr >> 8) & 0x3F
    sgpr_base = (gpr >> 16) & 0x3F
    sgpr_size = (gpr >> 24) & 0xF
    print(f"    VGPR: base={vgpr_base}, size_field={vgpr_size} (={vgpr_size * 8} regs)")
    print(f"    SGPR: base={sgpr_base}, size_field={sgpr_size} (={sgpr_size * 16} regs)")
    a2_data['GPR_ALLOC']['vgpr_base'] = vgpr_base
    a2_data['GPR_ALLOC']['vgpr_size_field'] = vgpr_size
    a2_data['GPR_ALLOC']['vgpr_count'] = vgpr_size * 8
    a2_data['GPR_ALLOC']['sgpr_base'] = sgpr_base
    a2_data['GPR_ALLOC']['sgpr_size_field'] = sgpr_size
    a2_data['GPR_ALLOC']['sgpr_count'] = sgpr_size * 16

    # HW_ID1 detailed decode
    print("\n  HW_ID1 decode:", flush=True)
    hw_id1_fields = [
        (52, 'WAVE_ID'), (53, 'SIMD_ID'), (54, 'WGP_ID'), (55, 'SA_ID'),
        (56, 'SE_ID'), (57, 'QUEUE_ID'), (58, 'STATE_ID_VMID'), (59, 'RSVD_31_24'),
    ]
    a2_data['HW_ID1'] = {}
    for idx, name in hw_id1_fields:
        v = int(vals[idx])
        a2_data['HW_ID1'][name] = v
        print(f"    {name:20s} = {v}", flush=True)

    # HW_ID2 detailed decode
    print("\n  HW_ID2 decode:", flush=True)
    hw_id2_fields = [
        (60, 'PIPE_ID'), (61, 'QUEUE_ID2'), (62, 'ME_ID'),
    ]
    a2_data['HW_ID2'] = {}
    for idx, name in hw_id2_fields:
        v = int(vals[idx])
        a2_data['HW_ID2'][name] = v
        print(f"    {name:20s} = {v}", flush=True)
    a2_data['HW_ID2']['full'] = int(vals[63])
    a2_data['HW_ID2']['full_hex'] = f'0x{int(vals[63]):08X}'

    # SHADER_CYCLES
    sc = int(vals[64])
    a2_data['SHADER_CYCLES'] = {'value': sc, 'hex': f'0x{sc:08X}', 'works': sc != 0}
    print(f"\n  SHADER_CYCLES = {sc} (0x{sc:08X}) {'*** WORKS!' if sc != 0 else '(zero — may not work on gfx1151)'}", flush=True)

    a2_data['status'] = 'success'
    save_intermediate('A2_hwreg_decode', a2_data, 'A2_hwreg_decode')

except SystemExit:
    save_results(); sys.exit(1)
except Exception as e:
    print(f"  [ERROR] A2 failed: {e}", flush=True)
    save_intermediate('A2_hwreg_decode', {'status': 'error', 'error': str(e)}, 'A2_hwreg_decode')


# ======================================================================
# PART A3: Clock Source Comparison
# ======================================================================
print("\n" + "=" * 70)
print("[A3] Clock Source Comparison — wall_clock64, clock64, s_memtime, s_memrealtime, SHADER_CYCLES")
print("=" * 70, flush=True)

try:
    if check_abort(): raise SystemExit("thermal abort")
    wait_cool("A3")

    N_WAVES = 64
    N_ITERS = 500

    # Run 5 rounds, aggregate
    all_rounds = []
    for r in range(5):
        raw = probe_mod.probe_clock_compare(N_WAVES, N_ITERS)
        torch.cuda.synchronize()
        data = raw.cpu().numpy()  # [N_WAVES, 12]
        all_rounds.append(data)
        time.sleep(0.1)

    a3_data = {
        'n_waves': N_WAVES, 'n_iters': N_ITERS, 'n_rounds': 5, 'clocks': {},
        'note': 's_memtime and s_memrealtime are NOT available on RDNA3/4 (gfx11+)',
    }

    clock_names = ['wall_clock64', 'clock64', 'SHADER_CYCLES']
    clock_cols = [(0, 1), (2, 3), (4, 5)]

    for name, (c0, c1) in zip(clock_names, clock_cols):
        # Compute deltas across all rounds
        all_deltas = []
        for data in all_rounds:
            deltas = data[:, c1] - data[:, c0]
            all_deltas.append(deltas)
        all_deltas = np.concatenate(all_deltas)

        # Basic stats
        works = np.any(all_deltas != 0)
        entry = {
            'works': bool(works),
            'mean_delta': float(np.mean(all_deltas)),
            'std_delta': float(np.std(all_deltas)),
            'min_delta': float(np.min(all_deltas)),
            'max_delta': float(np.max(all_deltas)),
            'cv': float(np.std(all_deltas) / max(np.mean(all_deltas), 1e-30)),
            'nonzero_frac': float(np.mean(all_deltas != 0)),
        }

        # Per-WGP analysis (only for first round)
        wgps = all_rounds[0][:, 6].astype(int)
        deltas0 = all_rounds[0][:, c1] - all_rounds[0][:, c0]
        wgp_means = {}
        for w in np.unique(wgps):
            mask = wgps == w
            if mask.sum() > 0:
                wgp_means[int(w)] = float(np.mean(deltas0[mask]))
        entry['per_wgp_means'] = wgp_means

        a3_data['clocks'][name] = entry

        status = "WORKS" if works else "ZERO/BROKEN"
        print(f"  {name:20s}: {status}  mean={entry['mean_delta']:.1f}  "
              f"std={entry['std_delta']:.1f}  CV={entry['cv']:.4f}  "
              f"nonzero={entry['nonzero_frac']:.2f}", flush=True)

    # Cross-clock correlation
    print("\n  Clock-to-clock correlation (round 0):", flush=True)
    a3_data['correlations'] = {}
    deltas_r0 = {}
    for name, (c0, c1) in zip(clock_names, clock_cols):
        deltas_r0[name] = all_rounds[0][:, c1] - all_rounds[0][:, c0]

    from itertools import combinations
    for n1, n2 in combinations(clock_names, 2):
        d1, d2 = deltas_r0[n1], deltas_r0[n2]
        if np.std(d1) > 0 and np.std(d2) > 0:
            corr = float(np.corrcoef(d1, d2)[0, 1])
        else:
            corr = 0.0
        a3_data['correlations'][f'{n1}_vs_{n2}'] = corr
        if abs(corr) > 0.01:
            print(f"    {n1} vs {n2}: r={corr:.4f}", flush=True)

    a3_data['status'] = 'success'
    save_intermediate('A3_clock_compare', a3_data, 'A3_clock_compare')

except SystemExit:
    save_results(); sys.exit(1)
except Exception as e:
    print(f"  [ERROR] A3 failed: {e}", flush=True)
    save_intermediate('A3_clock_compare', {'status': 'error', 'error': str(e)}, 'A3_clock_compare')


# ======================================================================
# PART A4: EXEC Mask Probing
# ======================================================================
print("\n" + "=" * 70)
print("[A4] EXEC Mask Probing — initial mask, active lane count")
print("=" * 70, flush=True)

try:
    if check_abort(): raise SystemExit("thermal abort")
    wait_cool("A4")

    N_WAVES = 64
    raw = probe_mod.probe_exec_mask(N_WAVES)
    torch.cuda.synchronize()
    data = raw.cpu().numpy()  # [N_WAVES, 5]

    a4_data = {'n_waves': N_WAVES}

    # Decode
    exec_lo = data[:, 0].astype(np.uint64)
    exec_hi = data[:, 1].astype(np.uint64)
    exec_full = exec_lo | (exec_hi << 32)
    wgps = data[:, 2].astype(int)
    simds = data[:, 3].astype(int)
    active_lanes = data[:, 4].astype(int)

    a4_data['active_lanes'] = {
        'min': int(active_lanes.min()),
        'max': int(active_lanes.max()),
        'mean': float(active_lanes.mean()),
        'unique': sorted([int(x) for x in np.unique(active_lanes)]),
    }
    a4_data['exec_masks'] = {
        'unique_lo': sorted([int(x) for x in np.unique(exec_lo)])[:10],
        'unique_hi': sorted([int(x) for x in np.unique(exec_hi)])[:10],
        'n_unique_masks': int(len(np.unique(exec_full))),
    }

    print(f"  Active lanes: min={active_lanes.min()}, max={active_lanes.max()}, "
          f"mean={active_lanes.mean():.1f}", flush=True)
    print(f"  Unique EXEC masks: {len(np.unique(exec_full))}", flush=True)
    print(f"  Sample EXEC_LO values: {[f'0x{int(v):08X}' for v in np.unique(exec_lo)[:5]]}", flush=True)
    print(f"  WGPs seen: {sorted(np.unique(wgps).tolist())}", flush=True)

    a4_data['status'] = 'success'
    save_intermediate('A4_exec_mask', a4_data, 'A4_exec_mask')

except SystemExit:
    save_results(); sys.exit(1)
except Exception as e:
    print(f"  [ERROR] A4 failed: {e}", flush=True)
    save_intermediate('A4_exec_mask', {'status': 'error', 'error': str(e)}, 'A4_exec_mask')


# ======================================================================
# PART A5: SHADER_CYCLES Variance Analysis
# ======================================================================
print("\n" + "=" * 70)
print("[A5] SHADER_CYCLES Variance — per-CU timing variation")
print("=" * 70, flush=True)

try:
    if check_abort(): raise SystemExit("thermal abort")
    wait_cool("A5")

    N_WAVES = 128
    N_ITERS = 500
    N_ROUNDS = 10

    all_deltas = []
    all_wgps = []

    for r in range(N_ROUNDS):
        raw = probe_mod.probe_shader_cycles_variance(N_WAVES, N_ITERS)
        torch.cuda.synchronize()
        data = raw.cpu().numpy().astype(np.uint32)  # [N_WAVES, 4]
        deltas = data[:, 1].astype(np.int64) - data[:, 0].astype(np.int64)
        # Handle wraparound
        deltas[deltas < 0] += (1 << 32)
        all_deltas.append(deltas)
        all_wgps.append(data[:, 2])
        time.sleep(0.05)

    all_deltas = np.concatenate(all_deltas)
    all_wgps = np.concatenate(all_wgps)

    a5_data = {
        'n_waves': N_WAVES, 'n_iters': N_ITERS, 'n_rounds': N_ROUNDS,
        'shader_cycles_works': bool(np.any(all_deltas != 0)),
    }

    if np.any(all_deltas != 0):
        a5_data['global_stats'] = {
            'mean': float(np.mean(all_deltas)),
            'std': float(np.std(all_deltas)),
            'cv': float(np.std(all_deltas) / max(np.mean(all_deltas), 1)),
            'min': int(np.min(all_deltas)),
            'max': int(np.max(all_deltas)),
        }

        # Per-WGP analysis
        a5_data['per_wgp'] = {}
        for w in sorted(np.unique(all_wgps)):
            mask = all_wgps == w
            d = all_deltas[mask]
            a5_data['per_wgp'][int(w)] = {
                'mean': float(np.mean(d)),
                'std': float(np.std(d)),
                'n_samples': int(mask.sum()),
            }

        # Cross-round stability
        round_means = [float(np.mean(all_deltas[i*N_WAVES:(i+1)*N_WAVES])) for i in range(N_ROUNDS)]
        a5_data['round_means'] = round_means
        a5_data['round_std'] = float(np.std(round_means))

        print(f"  SHADER_CYCLES WORKS on gfx1151!", flush=True)
        print(f"  Global: mean={np.mean(all_deltas):.1f}, std={np.std(all_deltas):.1f}, "
              f"CV={np.std(all_deltas)/max(np.mean(all_deltas),1):.4f}", flush=True)
        print(f"  Per-WGP means:", flush=True)
        for w in sorted(a5_data['per_wgp'].keys()):
            e = a5_data['per_wgp'][w]
            print(f"    WGP {w}: mean={e['mean']:.1f} std={e['std']:.1f} (n={e['n_samples']})", flush=True)
        print(f"  Cross-round stability: std of means = {np.std(round_means):.1f}", flush=True)
    else:
        print(f"  SHADER_CYCLES returns zero — not functional on gfx1151", flush=True)

    a5_data['status'] = 'success'
    save_intermediate('A5_shader_cycles', a5_data, 'A5_shader_cycles')

except SystemExit:
    save_results(); sys.exit(1)
except Exception as e:
    print(f"  [ERROR] A5 failed: {e}", flush=True)
    save_intermediate('A5_shader_cycles', {'status': 'error', 'error': str(e)}, 'A5_shader_cycles')


# ======================================================================
# PART A6: s_sendmsg Probing
# ======================================================================
print("\n" + "=" * 70)
print("[A6] s_sendmsg Probing — MSG_INTERRUPT (safe)")
print("=" * 70, flush=True)

try:
    if check_abort(): raise SystemExit("thermal abort")
    wait_cool("A6")

    raw = probe_mod.probe_sendmsg()
    torch.cuda.synchronize()
    vals = raw.cpu().numpy().astype(np.uint32)

    a6_data = {
        'STATUS_before': int(vals[0]), 'STATUS_before_hex': f'0x{int(vals[0]):08X}',
        'TRAPSTS_before': int(vals[1]), 'TRAPSTS_before_hex': f'0x{int(vals[1]):08X}',
        'STATUS_after_MSG_INTERRUPT': int(vals[2]), 'STATUS_after_hex': f'0x{int(vals[2]):08X}',
        'TRAPSTS_after_MSG_INTERRUPT': int(vals[3]), 'TRAPSTS_after_hex': f'0x{int(vals[3]):08X}',
        'msg_type_tested': int(vals[4]),
        'status_changed': int(vals[0]) != int(vals[2]),
        'trapsts_changed': int(vals[1]) != int(vals[3]),
    }

    print(f"  Before MSG_INTERRUPT:", flush=True)
    print(f"    STATUS  = 0x{int(vals[0]):08X}", flush=True)
    print(f"    TRAPSTS = 0x{int(vals[1]):08X}", flush=True)
    print(f"  After MSG_INTERRUPT:", flush=True)
    print(f"    STATUS  = 0x{int(vals[2]):08X} {'*** CHANGED' if a6_data['status_changed'] else '(same)'}", flush=True)
    print(f"    TRAPSTS = 0x{int(vals[3]):08X} {'*** CHANGED' if a6_data['trapsts_changed'] else '(same)'}", flush=True)

    a6_data['status'] = 'success'
    save_intermediate('A6_sendmsg', a6_data, 'A6_sendmsg')

except SystemExit:
    save_results(); sys.exit(1)
except Exception as e:
    print(f"  [ERROR] A6 failed: {e}", flush=True)
    save_intermediate('A6_sendmsg', {'status': 'error', 'error': str(e)}, 'A6_sendmsg')


# ======================================================================
# PART B: Ring Buffer / CP Probing (from host)
# ======================================================================
print("\n" + "=" * 70)
print("[B1] Ring Buffer / Debugfs Probing")
print("=" * 70, flush=True)

b_data = {'debugfs': {}, 'sysfs': {}, 'firmware': {}}

try:
    # B1: Check debugfs ring buffers
    debugfs_base = Path('/sys/kernel/debug/dri')
    cards = sorted(debugfs_base.iterdir()) if debugfs_base.exists() else []

    for card in cards:
        card_name = card.name
        if not card_name.isdigit():
            continue

        # List ring files
        ring_files = sorted(card.glob('amdgpu_ring_*'))
        wave_file = card / 'amdgpu_wave'
        fence_file = card / 'amdgpu_fence_info'
        gpu_reset = card / 'amdgpu_gpu_recover'

        b_data['debugfs'][card_name] = {
            'ring_files': [f.name for f in ring_files],
            'has_wave_status': wave_file.exists(),
            'has_fence_info': fence_file.exists(),
        }

        print(f"\n  Card {card_name}:", flush=True)
        print(f"    Ring files: {[f.name for f in ring_files]}", flush=True)
        print(f"    Wave status: {'YES' if wave_file.exists() else 'no'}", flush=True)
        print(f"    Fence info: {'YES' if fence_file.exists() else 'no'}", flush=True)

        # Try reading ring buffers (needs root usually)
        for rf in ring_files[:3]:
            try:
                content = rf.read_text()[:2000]
                lines = content.strip().split('\n')
                b_data['debugfs'][card_name][rf.name] = {
                    'readable': True,
                    'n_lines': len(lines),
                    'first_10_lines': lines[:10],
                }
                print(f"    {rf.name}: {len(lines)} lines (readable!)", flush=True)
                for line in lines[:5]:
                    print(f"      {line}", flush=True)
            except PermissionError:
                b_data['debugfs'][card_name][rf.name] = {'readable': False, 'error': 'permission denied'}
                print(f"    {rf.name}: permission denied (need root)", flush=True)
            except Exception as e:
                b_data['debugfs'][card_name][rf.name] = {'readable': False, 'error': str(e)}

        # Try reading wave status
        if wave_file.exists():
            try:
                wave_content = wave_file.read_text()[:3000]
                lines = wave_content.strip().split('\n')
                b_data['debugfs'][card_name]['wave_status'] = {
                    'readable': True, 'n_lines': len(lines),
                    'content': lines[:20],
                }
                print(f"    Wave status: {len(lines)} lines", flush=True)
                for line in lines[:10]:
                    print(f"      {line}", flush=True)
            except PermissionError:
                b_data['debugfs'][card_name]['wave_status'] = {'readable': False, 'error': 'permission denied'}
                print(f"    Wave status: permission denied", flush=True)
            except Exception as e:
                b_data['debugfs'][card_name]['wave_status'] = {'readable': False, 'error': str(e)}

        # Try reading fence info
        if fence_file.exists():
            try:
                fence_content = fence_file.read_text()[:3000]
                lines = fence_content.strip().split('\n')
                b_data['debugfs'][card_name]['fence_info'] = {
                    'readable': True, 'n_lines': len(lines),
                    'content': lines[:20],
                }
                print(f"    Fence info: {len(lines)} lines", flush=True)
                for line in lines[:10]:
                    print(f"      {line}", flush=True)
            except PermissionError:
                b_data['debugfs'][card_name]['fence_info'] = {'readable': False, 'error': 'permission denied'}
            except Exception as e:
                b_data['debugfs'][card_name]['fence_info'] = {'readable': False, 'error': str(e)}

    if not cards:
        print("  debugfs not accessible (need root or mount)", flush=True)
        b_data['debugfs']['error'] = 'not accessible'

except Exception as e:
    print(f"  [ERROR] B1 debugfs failed: {e}", flush=True)
    b_data['debugfs']['error'] = str(e)

# B2: Sysfs GPU info
print("\n  Sysfs GPU info:", flush=True)
try:
    drm_base = Path('/sys/class/drm/card0/device')
    sysfs_files = [
        'gpu_busy_percent', 'mem_busy_percent', 'current_link_speed',
        'current_link_width', 'vbios_version', 'unique_id',
        'pp_cur_state', 'pp_force_state', 'pp_table',
        'pp_dpm_sclk', 'pp_dpm_mclk', 'pp_od_clk_voltage',
    ]
    for fname in sysfs_files:
        f = drm_base / fname
        if f.exists():
            try:
                content = f.read_text().strip()[:500]
                b_data['sysfs'][fname] = content
                print(f"    {fname}: {content[:80]}", flush=True)
            except Exception as e:
                b_data['sysfs'][fname] = f'error: {e}'
        else:
            b_data['sysfs'][fname] = 'not found'
except Exception as e:
    b_data['sysfs']['error'] = str(e)

# B3: Firmware / ucode info
print("\n  Firmware versions:", flush=True)
try:
    fw_base = Path('/sys/class/drm/card0/device')
    for fw_name in ['fw_version', 'gpu_metrics']:
        f = fw_base / fw_name
        if f.exists():
            try:
                if 'metrics' in fw_name:
                    data = f.read_bytes()
                    b_data['firmware'][fw_name] = {
                        'size': len(data),
                        'header_hex': data[:32].hex(),
                    }
                    # Parse gpu_metrics header
                    if len(data) >= 4:
                        struct_size = struct.unpack_from('<H', data, 0)[0]
                        format_rev = data[2] if len(data) > 2 else 0
                        content_rev = data[3] if len(data) > 3 else 0
                        b_data['firmware'][fw_name]['struct_size'] = struct_size
                        b_data['firmware'][fw_name]['format_rev'] = format_rev
                        b_data['firmware'][fw_name]['content_rev'] = content_rev
                        print(f"    {fw_name}: size={len(data)}, struct_size={struct_size}, "
                              f"format_rev={format_rev}, content_rev={content_rev}", flush=True)
                else:
                    content = f.read_text().strip()
                    b_data['firmware'][fw_name] = content
                    print(f"    {fw_name}: {content}", flush=True)
            except Exception as e:
                b_data['firmware'][fw_name] = f'error: {e}'

    # Check for IP discovery table info
    ip_disc = fw_base / 'ip_discovery'
    if ip_disc.exists():
        try:
            content = ip_disc.read_text().strip()[:500]
            b_data['firmware']['ip_discovery'] = content
        except: pass

    # Read ucode version from sysfs
    for ucode in ['gfx_fw_version', 'sdma_fw_version', 'mc_fw_version',
                   'smc_fw_version', 'rlc_fw_version', 'ce_fw_version',
                   'pfp_fw_version', 'me_fw_version', 'mec_fw_version',
                   'mec2_fw_version', 'sos_fw_version', 'asd_fw_version',
                   'ta_fw_version', 'vcn_fw_version']:
        f = fw_base / ucode
        if f.exists():
            try:
                v = f.read_text().strip()
                b_data['firmware'][ucode] = v
                print(f"    {ucode}: {v}", flush=True)
            except: pass

except Exception as e:
    b_data['firmware']['error'] = str(e)

# Also check amdgpu driver info
print("\n  AMDGPU driver info:", flush=True)
try:
    result = subprocess.run(['lspci', '-nn', '-k', '-d', '1002:'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        b_data['lspci'] = result.stdout.strip()
        for line in result.stdout.strip().split('\n')[:6]:
            print(f"    {line}", flush=True)
except Exception as e:
    b_data['lspci'] = f'error: {e}'

save_intermediate('B_ring_buffer_cp', b_data, 'B_ring_buffer_cp')


# ======================================================================
# PART C1: SHADER_CYCLES as Reservoir Feature
# ======================================================================
print("\n" + "=" * 70)
print("[C1] SHADER_CYCLES as Reservoir Feature — mini wave4 benchmark")
print("=" * 70, flush=True)

try:
    if check_abort(): raise SystemExit("thermal abort")
    wait_cool("C1")

    # Check if SHADER_CYCLES works
    sc_works = results.get('parts', {}).get('A5_shader_cycles', {}).get('shader_cycles_works', False)
    print(f"  SHADER_CYCLES functional: {sc_works}", flush=True)

    N_WAVES = 64
    N_STEPS = 600
    WARMUP = 100
    N_ITERS_BASE = 200
    SEED = 42
    rng = np.random.RandomState(SEED)

    # Generate 4-class waveform targets
    t = np.arange(N_STEPS) / 20.0
    waves = {
        0: np.sin(t),
        1: np.sign(np.sin(t * 2)),
        2: (t % 1.0) * 2 - 1,
        3: np.sin(t) * np.sin(t * 3),
    }
    labels = np.zeros(N_STEPS, dtype=int)
    segment_len = N_STEPS // 4
    for c in range(4):
        labels[c * segment_len:(c + 1) * segment_len] = c
    inputs = np.zeros(N_STEPS)
    for i in range(N_STEPS):
        inputs[i] = waves[labels[i]][i]

    print(f"  Collecting {N_STEPS} steps with {N_WAVES} waves...", flush=True)

    all_features = []
    for step in range(N_STEPS):
        if step % 100 == 0:
            temp = get_temp()
            if temp > TEMP_PAUSE:
                wait_cool(f"C1 step {step}")
            if check_abort():
                raise SystemExit("thermal abort")

        inp = float(inputs[step])
        raw = probe_mod.probe_reservoir_shader(inp, N_WAVES, N_ITERS_BASE)
        torch.cuda.synchronize()
        data = raw.cpu().numpy()  # [N_WAVES, 3] — shader_delta, wall_delta, wgp

        # Features: shader_delta and wall_delta per WGP (aggregate)
        shader_deltas = data[:, 0]
        wall_deltas = data[:, 1]
        wgps = data[:, 2].astype(int)

        # Build feature vector: per-WGP shader and wall deltas
        unique_wgps = sorted(np.unique(wgps))
        n_wgps = len(unique_wgps)
        feat = np.zeros(n_wgps * 2)
        for j, w in enumerate(unique_wgps):
            mask = wgps == w
            feat[j] = np.mean(shader_deltas[mask])
            feat[j + n_wgps] = np.mean(wall_deltas[mask])

        all_features.append(feat)

    features = np.array(all_features)  # [N_STEPS, n_features]
    print(f"  Feature matrix: {features.shape}", flush=True)

    # Split train/test (after warmup)
    X = features[WARMUP:]
    y = labels[WARMUP:]
    split = len(X) // 2
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Ridge regression classifier
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train_s, y_train)
    train_acc = float(clf.score(X_train_s, y_train))
    test_acc = float(clf.score(X_test_s, y_test))

    print(f"  Wave4 classification: train={train_acc:.3f}, test={test_acc:.3f}", flush=True)

    # Separate shader-only vs wall-only
    n_wgps_half = features.shape[1] // 2
    X_shader = features[WARMUP:, :n_wgps_half]
    X_wall = features[WARMUP:, n_wgps_half:]

    shader_train = scaler.fit_transform(X_shader[:split])
    shader_test = scaler.transform(X_shader[split:])
    clf_s = RidgeClassifier(alpha=1.0)
    clf_s.fit(shader_train, y_train)
    shader_acc = float(clf_s.score(shader_test, y_test))

    wall_train = scaler.fit_transform(X_wall[:split])
    wall_test = scaler.transform(X_wall[split:])
    clf_w = RidgeClassifier(alpha=1.0)
    clf_w.fit(wall_train, y_train)
    wall_acc = float(clf_w.score(wall_test, y_test))

    print(f"  Shader-cycles-only: {shader_acc:.3f}", flush=True)
    print(f"  Wall-clock-only:    {wall_acc:.3f}", flush=True)

    # Memory capacity (linear regression on delayed copies)
    from sklearn.linear_model import Ridge

    mc_total_shader = 0.0
    mc_total_wall = 0.0
    mc_total_combined = 0.0
    mc_details = {}
    for delay in range(1, 11):
        if delay >= len(X_train):
            break
        y_mc = inputs[WARMUP + delay:WARMUP + delay + split]
        if len(y_mc) != len(X_train):
            y_mc = y_mc[:len(X_train)]
            if len(y_mc) < len(X_train):
                continue

        y_mc_test = inputs[WARMUP + split + delay:WARMUP + split + delay + len(X_test)]
        if len(y_mc_test) != len(X_test):
            y_mc_test = y_mc_test[:len(X_test)]
            if len(y_mc_test) < len(X_test):
                continue

        # Combined
        reg = Ridge(alpha=1.0)
        Xt = scaler.fit_transform(X_train)
        reg.fit(Xt, y_mc)
        Xte = scaler.transform(X_test)
        pred = reg.predict(Xte)
        corr = np.corrcoef(pred, y_mc_test)[0, 1] if np.std(pred) > 0 and np.std(y_mc_test) > 0 else 0.0
        r2 = max(0, corr ** 2)
        mc_total_combined += r2

        # Shader only
        reg_s = Ridge(alpha=1.0)
        sst = scaler.fit_transform(X_shader[:split])
        reg_s.fit(sst, y_mc)
        sste = scaler.transform(X_shader[split:])
        pred_s = reg_s.predict(sste)
        corr_s = np.corrcoef(pred_s, y_mc_test)[0, 1] if np.std(pred_s) > 0 and np.std(y_mc_test) > 0 else 0.0
        mc_total_shader += max(0, corr_s ** 2)

        # Wall only
        reg_w = Ridge(alpha=1.0)
        wst = scaler.fit_transform(X_wall[:split])
        reg_w.fit(wst, y_mc)
        wste = scaler.transform(X_wall[split:])
        pred_w = reg_w.predict(wste)
        corr_w = np.corrcoef(pred_w, y_mc_test)[0, 1] if np.std(pred_w) > 0 and np.std(y_mc_test) > 0 else 0.0
        mc_total_wall += max(0, corr_w ** 2)

        mc_details[delay] = {
            'combined_r2': float(r2),
            'shader_r2': float(max(0, corr_s**2)),
            'wall_r2': float(max(0, corr_w**2)),
        }

    print(f"  Memory Capacity (delays 1-10):", flush=True)
    print(f"    Combined:     MC = {mc_total_combined:.3f}", flush=True)
    print(f"    Shader-only:  MC = {mc_total_shader:.3f}", flush=True)
    print(f"    Wall-only:    MC = {mc_total_wall:.3f}", flush=True)

    c1_data = {
        'n_waves': N_WAVES, 'n_steps': N_STEPS, 'warmup': WARMUP,
        'shader_cycles_as_feature': sc_works,
        'wave4_classification': {
            'combined': test_acc,
            'shader_only': shader_acc,
            'wall_only': wall_acc,
            'train_combined': train_acc,
        },
        'memory_capacity': {
            'combined': mc_total_combined,
            'shader_only': mc_total_shader,
            'wall_only': mc_total_wall,
            'per_delay': mc_details,
        },
        'feature_stats': {
            'n_features': int(features.shape[1]),
            'n_wgps': n_wgps_half,
            'feature_mean': float(np.mean(features)),
            'feature_std': float(np.std(features)),
        },
        'status': 'success',
    }

    save_intermediate('C1_reservoir_shader_cycles', c1_data, 'C1_reservoir')

except SystemExit:
    save_results(); sys.exit(1)
except Exception as e:
    print(f"  [ERROR] C1 failed: {e}", flush=True)
    save_intermediate('C1_reservoir_shader_cycles', {'status': 'error', 'error': str(e)}, 'C1_reservoir')


# ======================================================================
# Final Summary
# ======================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70, flush=True)

results['timestamp_end'] = time.strftime('%Y-%m-%d %H:%M:%S')

# Compute summary
summary = {}
for part_name, part_data in results.get('parts', {}).items():
    if isinstance(part_data, dict):
        summary[part_name] = part_data.get('status', 'unknown')

results['summary'] = summary

n_success = sum(1 for v in summary.values() if v == 'success')
n_total = len(summary)
print(f"\n  Parts completed: {n_success}/{n_total}", flush=True)

for name, status in summary.items():
    marker = "OK" if status == 'success' else "FAIL"
    print(f"    [{marker}] {name}", flush=True)

# Key findings
findings = []
parts = results.get('parts', {})

# SHADER_CYCLES
if parts.get('A5_shader_cycles', {}).get('shader_cycles_works'):
    findings.append("SHADER_CYCLES (HWREG[29]) is FUNCTIONAL on gfx1151")
else:
    findings.append("SHADER_CYCLES (HWREG[29]) is NOT functional on gfx1151")

# Clock sources
a3 = parts.get('A3_clock_compare', {}).get('clocks', {})
for name, info in a3.items():
    if isinstance(info, dict) and info.get('works'):
        findings.append(f"{name}: works, CV={info.get('cv', 0):.4f}")

# Register count
a1 = parts.get('A1_hwreg_sweep', {})
if 'nonzero_count' in a1:
    findings.append(f"{a1['nonzero_count']}/32 HWREG IDs return non-zero values")

# Reservoir performance
c1 = parts.get('C1_reservoir_shader_cycles', {})
if c1.get('status') == 'success':
    wave4 = c1.get('wave4_classification', {})
    mc = c1.get('memory_capacity', {})
    findings.append(f"Reservoir wave4: combined={wave4.get('combined', 0):.3f}, "
                    f"shader_only={wave4.get('shader_only', 0):.3f}")
    findings.append(f"Reservoir MC: combined={mc.get('combined', 0):.3f}, "
                    f"shader_only={mc.get('shader_only', 0):.3f}")

results['key_findings'] = findings
print("\n  Key findings:", flush=True)
for f in findings:
    print(f"    - {f}", flush=True)

save_results()
print(f"\n  Final results: {SAVE_FILE}", flush=True)
print("  DONE.", flush=True)
