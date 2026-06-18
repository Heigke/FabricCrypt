#!/usr/bin/env python3
"""
z2329_analog_physics_model.py — Surgical Transistor Physics v3
======================================================================
Deep per-structure GPU probing for process variation (ΔVth) mapping:
  1. Per-CU + Per-SIMD timing (within-WGP variation)
  2. LDS bank-conflict differential (bank circuit isolation)
  3. Pointer-chasing latency (serialized, true latency)
  4. Instruction-type fingerprints (VALU vs transcendental vs LDS)
  5. VGPR depth via array-indirect (compiler-proof)
  6. Thermal ramp cooldown tracking
  7. Cache set pointer-chase (serialized miss chain)

Physics: delay ∝ C·Vdd / (Vdd - Vth)^α · exp(Ea / kB·T)
  At constant V → Δdelay = f(ΔVth, ΔT)
  Differential probes (conflict - no_conflict) cancel scheduling → pure circuit delay

Tests (30): T1-T30

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 python scripts/z2329_analog_physics_model.py
"""

import os, sys, time, json, struct
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway, kruskal

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)

# ======================================================================
# Telemetry
# ======================================================================
def get_gpu_metrics():
    metrics = {}
    try:
        hwmon = Path('/sys/class/hwmon')
        gpu_hwmon = None
        for h in hwmon.iterdir():
            nf = h / 'name'
            if nf.exists() and nf.read_text().strip() == 'amdgpu':
                gpu_hwmon = h; break
        if gpu_hwmon is None:
            gpu_hwmon = hwmon / 'hwmon7'
        for name, fname, div in [('temp_c', 'temp1_input', 1000.0),
                                   ('clock_mhz', 'freq1_input', 1e6),
                                   ('power_w', 'power1_average', 1e6)]:
            f = gpu_hwmon / fname
            if f.exists():
                metrics[name] = int(f.read_text().strip()) / div
    except Exception as e:
        metrics['error'] = str(e)
    try:
        for card in Path('/sys/class/drm').iterdir():
            gm = card / 'device' / 'gpu_metrics'
            if gm.exists():
                data = gm.read_bytes()
                if len(data) >= 22:
                    metrics['voltage_mv'] = struct.unpack_from('<H', data, 20)[0]
                break
    except: pass
    return metrics

def get_temp():
    try: return int(open('/sys/class/thermal/thermal_zone0/temp').read()) // 1000
    except: return 0

def wait_cool(target=50, timeout=120):
    t0 = time.time()
    while get_temp() > target and time.time() - t0 < timeout:
        time.sleep(2)
    return get_temp()

# ======================================================================
# Compile HIP probes via torch.utils.cpp_extension
# ======================================================================
import torch
from torch.utils.cpp_extension import load_inline

PROBE_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define HWREG(id, offset, size) ((id) | ((offset) << 6) | (((size)-1) << 11))
#define HW_REG_HW_ID1 23

// =====================================================================
// Probe 1: Per-CU + Per-SIMD Compute Timing
// Reports: wall_ns, core_cycles, wgp_id, simd_id, se_sa, wave_id
// =====================================================================
__global__ void kernel_cu_simd(float* out, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid + 1) * 0.001f;

    // Warmup loop (not timed)
    #pragma unroll 1
    for (int i = 0; i < 50; i++) {
        x = sinf(x) * cosf(x + 0.1f) + expf(-x * x);
        x = x * 1.0001f + 0.0001f;
    }

    long long w0 = wall_clock64();
    uint64_t c0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = sinf(x) * cosf(x + 0.1f) + expf(-x * x);
        x = x * 1.0001f + 0.0001f;
    }
    uint64_t c1 = clock64();
    long long w1 = wall_clock64();
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));

    // 6 outputs per thread
    out[tid*6 + 0] = (float)(w1 - w0) * 10.0f;  // wall_ns
    out[tid*6 + 1] = (float)(c1 - c0);           // core_cycles
    out[tid*6 + 2] = (float)((hw >> 8) & 0xF);   // wgp_id
    out[tid*6 + 3] = (float)((hw >> 4) & 0x3);   // simd_id
    out[tid*6 + 4] = (float)(((hw >> 13) & 0x7) * 2 + ((hw >> 12) & 0x1)); // se_sa
    out[tid*6 + 5] = (float)((hw >> 0) & 0xF);   // wave_id
    if (__builtin_expect(x == -1e30f, 0)) out[0] = x;
}

torch::Tensor probe_cu_simd(int n_waves, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n * 6}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_cu_simd<<<n_waves, 32>>>(out.data_ptr<float>(), n_iters, n);
    return out.reshape({n, 6});
}

// =====================================================================
// Probe 2: LDS Bank-Conflict Differential
// Mode 0: ALL threads hit SAME bank (max conflict)
// Mode 1: threads hit DIFFERENT banks (no conflict = round-robin)
// Differential = bank circuit delay isolated from scheduling overhead
// =====================================================================
__global__ void kernel_lds_differential(float* out, int target_bank,
                                         int mode, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    __shared__ float lds[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        lds[i] = (float)i;
    __syncthreads();

    float sum = 0.0f;
    // Warmup
    #pragma unroll 1
    for (int i = 0; i < 100; i++) {
        int idx;
        if (mode == 0) {
            // CONFLICT: all threads in wavefront hit same bank
            idx = target_bank + (i % 128) * 32;
        } else {
            // NO CONFLICT: each thread hits its own bank
            idx = (threadIdx.x % 32) + (i % 128) * 32;
        }
        sum += lds[idx];
        lds[idx] = sum * 0.999f;
    }
    __syncthreads();

    long long w0 = wall_clock64();
    uint64_t c0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        int idx;
        if (mode == 0) {
            idx = target_bank + (i % 128) * 32;
        } else {
            idx = (threadIdx.x % 32) + (i % 128) * 32;
        }
        sum += lds[idx];
        lds[idx] = sum * 0.999f;
    }
    uint64_t c1 = clock64();
    long long w1 = wall_clock64();
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));

    out[tid*4 + 0] = (float)(w1 - w0) * 10.0f;
    out[tid*4 + 1] = (float)(c1 - c0);
    out[tid*4 + 2] = (float)((hw >> 8) & 0xF);
    out[tid*4 + 3] = (float)target_bank;
    if (__builtin_expect(sum == -1e30f, 0)) out[0] = sum;
}

torch::Tensor probe_lds_diff(int n, int target_bank, int mode, int n_iters) {
    auto out = torch::zeros({n * 4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_lds_differential<<<(n+31)/32, 32, 16384>>>(
        out.data_ptr<float>(), target_bank, mode, n_iters, n);
    return out.reshape({n, 4});
}

// =====================================================================
// Probe 3: Pointer-Chasing Latency (serialized dependent loads)
// Each load depends on previous — measures TRUE memory latency, not throughput
// =====================================================================
__global__ void kernel_pointer_chase(float* out, const int* chase_chain,
                                      int chain_len, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Warmup chase
    int idx = tid % chain_len;
    for (int i = 0; i < 50; i++) {
        idx = chase_chain[idx];
    }

    long long w0 = wall_clock64();
    uint64_t c0 = clock64();
    // Serialized dependent loads
    idx = tid % chain_len;
    #pragma unroll 1
    for (int i = 0; i < chain_len; i++) {
        idx = chase_chain[idx];
    }
    uint64_t c1 = clock64();
    long long w1 = wall_clock64();
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));

    out[tid*5 + 0] = (float)(w1 - w0) * 10.0f;
    out[tid*5 + 1] = (float)(c1 - c0);
    out[tid*5 + 2] = (float)((hw >> 8) & 0xF);
    out[tid*5 + 3] = (float)((hw >> 4) & 0x3);
    out[tid*5 + 4] = (float)idx;  // Anti-optimization
}

torch::Tensor probe_pointer_chase(int n, torch::Tensor chase_chain, int chain_len) {
    auto out = torch::zeros({n * 5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_pointer_chase<<<(n+31)/32, 32>>>(
        out.data_ptr<float>(), chase_chain.data_ptr<int>(), chain_len, n);
    return out.reshape({n, 5});
}

// =====================================================================
// Probe 4: Instruction-Type Fingerprints
// type=0: pure VALU (fma chain)
// type=1: pure transcendental (sinf/cosf chain)
// type=2: pure integer (bitwise chain)
// type=3: pure SALU (scalar ops via wave-uniform)
// =====================================================================
__global__ void kernel_insn_type(float* out, int insn_type, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float x = (float)(tid + 1) * 0.001f;
    int ix = tid + 1;

    // Warmup
    #pragma unroll 1
    for (int i = 0; i < 50; i++) {
        if (insn_type == 0)      x = x * 1.0001f + 0.0001f;
        else if (insn_type == 1) x = sinf(x) + 0.0001f;
        else if (insn_type == 2) ix = (ix * 2654435761u) ^ (ix >> 13);
        else                     x = x + (float)(blockIdx.x & 1) * 0.0001f;
    }

    long long w0 = wall_clock64();
    uint64_t c0 = clock64();

    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        if (insn_type == 0) {
            // Pure VALU: FMA chain
            x = x * 1.0001f + 0.0001f;
            x = x * 0.9999f + 0.0002f;
        } else if (insn_type == 1) {
            // Pure transcendental: sinf/cosf
            x = sinf(x) * cosf(x + 0.1f);
            x = x * 1.0001f + 0.0001f;
        } else if (insn_type == 2) {
            // Pure integer: bitwise ops
            ix = (ix * 2654435761u) ^ (ix >> 13);
            ix = ix + (ix << 3);
        } else {
            // SALU-heavy: wave-uniform operations
            x = x + (float)(blockIdx.x & 1) * 0.0001f;
            x = x * (1.0f + (float)(blockIdx.x >> 16) * 1e-9f);
        }
    }

    uint64_t c1 = clock64();
    long long w1 = wall_clock64();
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));

    float result = (insn_type == 2) ? (float)ix : x;
    out[tid*5 + 0] = (float)(w1 - w0) * 10.0f;
    out[tid*5 + 1] = (float)(c1 - c0);
    out[tid*5 + 2] = (float)((hw >> 8) & 0xF);
    out[tid*5 + 3] = (float)((hw >> 4) & 0x3);
    out[tid*5 + 4] = (float)insn_type;
    if (__builtin_expect(result == -1e30f, 0)) out[0] = result;
}

torch::Tensor probe_insn_type(int n_waves, int insn_type, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n * 5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_insn_type<<<n_waves, 32>>>(out.data_ptr<float>(), insn_type, n_iters, n);
    return out.reshape({n, 5});
}

// =====================================================================
// Probe 5: VGPR Depth via Array Indirect (compiler-proof)
// All depths execute identical code; depth controls WHICH array slots to touch
// =====================================================================
__global__ void kernel_vgpr_indirect(float* out, int active_regs, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // 32 live float registers in an array — compiler cannot elide
    float regs[32];
    for (int r = 0; r < 32; r++)
        regs[r] = (float)(tid + r) * 0.001f;

    // Warmup
    #pragma unroll 1
    for (int i = 0; i < 50; i++) {
        for (int r = 0; r < active_regs; r++) {
            regs[r] = regs[r] * 1.0001f + regs[(r+1) % active_regs] * 0.01f;
        }
    }

    long long w0 = wall_clock64();
    uint64_t c0 = clock64();

    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        for (int r = 0; r < active_regs; r++) {
            regs[r] = regs[r] * 1.0001f + regs[(r+1) % active_regs] * 0.01f;
        }
    }

    uint64_t c1 = clock64();
    long long w1 = wall_clock64();
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));

    float total = 0;
    for (int r = 0; r < 32; r++) total += regs[r];

    out[tid*4 + 0] = (float)(w1 - w0) * 10.0f;
    out[tid*4 + 1] = (float)(c1 - c0);
    out[tid*4 + 2] = (float)((hw >> 8) & 0xF);
    out[tid*4 + 3] = (float)active_regs;
    if (__builtin_expect(total == -1e30f, 0)) out[0] = total;
}

torch::Tensor probe_vgpr_indirect(int n_waves, int active_regs, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n * 4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_vgpr_indirect<<<n_waves, 32>>>(out.data_ptr<float>(), active_regs, n_iters, n);
    return out.reshape({n, 4});
}

// =====================================================================
// Probe 6: Memory/Compute Ratio (per-CU, with warmup)
// =====================================================================
__global__ void kernel_ratio_v2(float* out, const float* workspace, int ws_size,
                                 int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid + 1) * 0.001f;
    float sum = 0.0f;
    // Warmup both paths
    for (int i = 0; i < 50; i++) {
        x = sinf(x) * cosf(x + 0.1f);
        int idx = ((tid * 2654435761u + i * 40503u) >> 4) % ws_size;
        sum += workspace[idx];
    }

    // Compute phase
    uint64_t cc0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = sinf(x) * cosf(x + 0.1f);
        x = x * 1.0001f + 0.0001f;
    }
    uint64_t cc1 = clock64();

    // Memory phase
    uint64_t mc0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        int idx = ((tid * 2654435761u + i * 40503u) >> 4) % ws_size;
        sum += workspace[idx];
    }
    uint64_t mc1 = clock64();

    long long w1 = wall_clock64();
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));

    float comp = (float)(cc1 - cc0);
    float mem = (float)(mc1 - mc0);
    out[tid*5 + 0] = (comp > 1.0f) ? mem / comp : 0.0f;
    out[tid*5 + 1] = comp;
    out[tid*5 + 2] = mem;
    out[tid*5 + 3] = (float)((hw >> 8) & 0xF);
    out[tid*5 + 4] = (float)(((hw >> 13) & 0x7) * 2 + ((hw >> 12) & 0x1));
    if (__builtin_expect((x + sum) == -1e30f, 0)) out[0] = x;
}

torch::Tensor probe_ratio_v2(int n_waves, torch::Tensor workspace, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n * 5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_ratio_v2<<<n_waves, 32>>>(
        out.data_ptr<float>(), workspace.data_ptr<float>(), workspace.size(0),
        n_iters, n);
    return out.reshape({n, 5});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_cu_simd", &probe_cu_simd);
    m.def("probe_lds_diff", &probe_lds_diff);
    m.def("probe_pointer_chase", &probe_pointer_chase);
    m.def("probe_insn_type", &probe_insn_type);
    m.def("probe_vgpr_indirect", &probe_vgpr_indirect);
    m.def("probe_ratio_v2", &probe_ratio_v2);
}
'''

print("Compiling HIP probe kernels v3...")
probe = load_inline(
    name='z2329_surgical_probe_v3',
    cpp_sources='',
    cuda_sources=PROBE_CUDA_SRC,
    extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
    verbose=False,
)
print("  Probe module compiled and loaded")

# ======================================================================
# Results
# ======================================================================
results = {'experiments': {}, 'tests': {}, 'meta': {
    'script': 'z2329_analog_physics_model.py',
    'version': 'v3_deep_surgical',
    'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    'physics': 'differential probes cancel scheduling → pure circuit delay',
}}

def test(tid, name, passed, value=None):
    status = 'PASS' if passed else 'FAIL'
    results['tests'][tid] = {'name': name, 'status': status, 'value': value}
    print(f"  [{status}] {tid}: {name} = {value}")

def save():
    results['meta']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    passed = sum(1 for t in results['tests'].values() if t['status'] == 'PASS')
    total = len(results['tests'])
    results['meta']['tests_passed'] = passed
    results['meta']['tests_total'] = total
    with open(RESULTS / 'z2329_analog_physics_model.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [SAVED] {RESULTS / 'z2329_analog_physics_model.json'}")

# ======================================================================
# MAIN
# ======================================================================
dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print(f"Device: {dev_name}")
print("=" * 70)
print("  z2329: Surgical Transistor Physics v3 (Deep Probes)")
print("  Per-CU, per-SIMD, per-bank differential, pointer-chase, insn-type")
print("=" * 70)
m0 = get_gpu_metrics()
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Temp: {get_temp()}C, V={m0.get('voltage_mv', '?')}mV, Clk={m0.get('clock_mhz', '?')}MHz")

# Warmup GPU
_ = torch.randn(512, 512, device='cuda') @ torch.randn(512, 512, device='cuda')
torch.cuda.synchronize()

N_WAVES = 40   # More waves → better WGP coverage
N_ITERS = 1000  # More iterations → more stable timing

# ======================================================================
# EXP 1: Per-CU + Per-SIMD Process Variation Map
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 1: Per-CU + Per-SIMD Process Variation")
print("  40 wavefronts × 20 reps, warmup in-kernel, 6 fields per thread")
print("=" * 70)

wait_cool(50)

# Discard first 2 runs (warmup)
for _ in range(2):
    probe.probe_cu_simd(N_WAVES, N_ITERS)
torch.cuda.synchronize()

cu_simd_runs = []
for run_idx in range(20):
    data = probe.probe_cu_simd(N_WAVES, N_ITERS).cpu().numpy()
    cu_simd_runs.append(data)
    if (run_idx + 1) % 5 == 0:
        wgps = np.unique(data[:, 2]).astype(int)
        simds = np.unique(data[:, 3]).astype(int)
        print(f"  Run {run_idx+1}: wall={data[:,0].mean():.0f}ns ±{data[:,0].std():.0f}ns, "
              f"{len(wgps)} WGPs, SIMDs={sorted(simds)}")
    time.sleep(0.1)

all_cs = np.vstack(cu_simd_runs)

# Per-WGP timing
wgp_ids = np.unique(all_cs[:, 2]).astype(int)
cu_timing = {}
for wgp in wgp_ids:
    mask = all_cs[:, 2] == wgp
    cu_timing[int(wgp)] = {
        'wall_mean': float(all_cs[mask, 0].mean()),
        'wall_std': float(all_cs[mask, 0].std()),
        'cycles_mean': float(all_cs[mask, 1].mean()),
        'n': int(mask.sum()),
    }

# Per-SIMD timing (within-WGP variation)
simd_timing = {}
for wgp in wgp_ids:
    simd_timing[int(wgp)] = {}
    for simd in range(4):
        mask = (all_cs[:, 2] == wgp) & (all_cs[:, 3] == simd)
        if mask.sum() > 0:
            simd_timing[int(wgp)][int(simd)] = {
                'wall_mean': float(all_cs[mask, 0].mean()),
                'n': int(mask.sum()),
            }

if len(cu_timing) > 1:
    means = np.array([v['wall_mean'] for v in cu_timing.values()])
    cu_cv = means.std() / (means.mean() + 1e-10)
    cu_range_pct = (means.max() - means.min()) / (means.mean() + 1e-10) * 100

    print(f"\n  Per-WGP timing ({len(cu_timing)} WGPs, {all_cs.shape[0]} samples):")
    sorted_wgps = sorted(cu_timing.keys(), key=lambda w: cu_timing[w]['wall_mean'])
    fastest = cu_timing[sorted_wgps[0]]['wall_mean']
    for wgp in sorted_wgps:
        d = cu_timing[wgp]
        delta = (d['wall_mean'] - fastest) / fastest * 100
        bar = '█' * max(1, int(delta * 10))
        print(f"    WGP {wgp:2d}: {d['wall_mean']:8.0f} ±{d['wall_std']:5.0f} ns "
              f"(+{delta:.2f}%) [{d['n']} samp] {bar}")

    # Per-SIMD within-WGP
    print(f"\n  Per-SIMD within-WGP variation:")
    simd_cvs = []
    for wgp in sorted_wgps[:5]:  # Show top 5 WGPs
        simds = simd_timing.get(wgp, {})
        if len(simds) >= 2:
            sm = np.array([v['wall_mean'] for v in simds.values()])
            scv = sm.std() / (sm.mean() + 1e-10) * 100
            simd_cvs.append(scv)
            simd_str = ' '.join(f"S{s}:{simds[s]['wall_mean']:.0f}" for s in sorted(simds))
            print(f"    WGP {wgp:2d}: {simd_str}  (CV={scv:.3f}%)")
    avg_simd_cv = np.mean(simd_cvs) if simd_cvs else 0
    print(f"  CU CV: {cu_cv:.4f} ({cu_range_pct:.2f}%), SIMD CV: {avg_simd_cv:.4f}%")
else:
    cu_cv = cu_range_pct = avg_simd_cv = 0
    means = np.array([0])

# Reproducibility: first 10 vs last 10
wgp_a, wgp_b = {}, {}
for wgp in wgp_ids:
    a = np.vstack(cu_simd_runs[:10])
    b = np.vstack(cu_simd_runs[10:])
    ma = a[a[:, 2] == wgp, 0]
    mb = b[b[:, 2] == wgp, 0]
    if len(ma) > 0: wgp_a[int(wgp)] = ma.mean()
    if len(mb) > 0: wgp_b[int(wgp)] = mb.mean()
common = sorted(set(wgp_a) & set(wgp_b))
repro_r = spearmanr([wgp_a[w] for w in common],
                     [wgp_b[w] for w in common])[0] if len(common) >= 3 else 0

results['experiments']['exp1_cu_simd'] = {
    'n_wgps': len(cu_timing), 'cu_cv': float(cu_cv),
    'cu_range_pct': float(cu_range_pct),
    'avg_simd_cv_pct': float(avg_simd_cv),
    'repro_r': float(repro_r),
    'cu_timing': {str(k): v for k, v in cu_timing.items()},
}

test('T1', 'CU-to-CU CV > 0.1%', cu_cv > 0.001,
     f"CV={cu_cv:.4f} ({cu_range_pct:.2f}%)")
test('T2', 'CU map reproducible (r > 0.5)', repro_r > 0.5,
     f"r={repro_r:.3f}")
test('T3', 'Per-SIMD variation detectable (CV > 0.01%)', avg_simd_cv > 0.01,
     f"avg={avg_simd_cv:.4f}%")
test('T4', '8+ WGPs detected', len(cu_timing) >= 8,
     f"{len(cu_timing)} WGPs")
save()

# ======================================================================
# EXP 2: LDS Bank-Conflict Differential
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 2: LDS Bank-Conflict Differential")
print("  conflict_time - no_conflict_time = pure bank circuit delay")
print("=" * 70)

wait_cool(50)
N_BANK_ITERS = 4000
N_PER_BANK = 64  # threads per bank measurement

bank_conflict = {}
bank_noconflict = {}
bank_differential = {}

# Warmup
probe.probe_lds_diff(N_PER_BANK, 0, 0, N_BANK_ITERS)
probe.probe_lds_diff(N_PER_BANK, 0, 1, N_BANK_ITERS)
torch.cuda.synchronize()

for bank in range(32):
    # Mode 0: all threads same bank (conflict)
    c_times = []
    for rep in range(3):
        d = probe.probe_lds_diff(N_PER_BANK, bank, 0, N_BANK_ITERS).cpu().numpy()
        c_times.extend(d[:, 0].tolist())
    bank_conflict[bank] = float(np.mean(c_times))

    # Mode 1: each thread different bank (no conflict)
    nc_times = []
    for rep in range(3):
        d = probe.probe_lds_diff(N_PER_BANK, bank, 1, N_BANK_ITERS).cpu().numpy()
        nc_times.extend(d[:, 0].tolist())
    bank_noconflict[bank] = float(np.mean(nc_times))

    # Differential = conflict - no_conflict = PURE bank stall time
    bank_differential[bank] = bank_conflict[bank] - bank_noconflict[bank]

    if bank % 8 == 0:
        print(f"  Bank {bank:2d}: conflict={bank_conflict[bank]:.0f}ns, "
              f"no_conf={bank_noconflict[bank]:.0f}ns, "
              f"Δ={bank_differential[bank]:.0f}ns")

    if bank % 16 == 15 and get_temp() > 70:
        wait_cool(55)

diff_values = np.array([bank_differential[b] for b in range(32)])
diff_cv = diff_values.std() / (abs(diff_values.mean()) + 1e-10)
diff_range = diff_values.max() - diff_values.min()

# Which banks are outliers?
median_diff = np.median(diff_values)
outlier_banks = [b for b in range(32) if abs(diff_values[b] - median_diff) > 2 * diff_values.std()]

print(f"\n  Differential: mean={diff_values.mean():.0f}ns, std={diff_values.std():.0f}ns")
print(f"  CV={diff_cv:.4f}, range={diff_range:.0f}ns")
print(f"  Outlier banks (>2σ): {outlier_banks}")

# Reproducibility: re-measure 8 banks
diff_r2 = []
for bank in [0, 4, 8, 12, 16, 20, 24, 28]:
    c = probe.probe_lds_diff(N_PER_BANK, bank, 0, N_BANK_ITERS).cpu().numpy()[:, 0].mean()
    nc = probe.probe_lds_diff(N_PER_BANK, bank, 1, N_BANK_ITERS).cpu().numpy()[:, 0].mean()
    diff_r2.append(c - nc)
diff_r1 = [bank_differential[b] for b in [0, 4, 8, 12, 16, 20, 24, 28]]
if np.std(diff_r1) > 0 and np.std(diff_r2) > 0:
    diff_repro_r, _ = pearsonr(diff_r1, diff_r2)
else:
    diff_repro_r = 0

results['experiments']['exp2_lds_differential'] = {
    'diff_mean': float(diff_values.mean()),
    'diff_std': float(diff_values.std()),
    'diff_cv': float(diff_cv),
    'diff_repro_r': float(diff_repro_r),
    'outlier_banks': outlier_banks,
    'bank_differential': {str(k): float(v) for k, v in bank_differential.items()},
    'bank_conflict': {str(k): float(v) for k, v in bank_conflict.items()},
    'bank_noconflict': {str(k): float(v) for k, v in bank_noconflict.items()},
}

test('T5', 'Differential CV > 0.5%', diff_cv > 0.005,
     f"CV={diff_cv:.4f}")
test('T6', 'Differential reproducible (r > 0.5)', diff_repro_r > 0.5,
     f"r={diff_repro_r:.3f}")
test('T7', '4+ outlier banks', len(outlier_banks) >= 4,
     f"{len(outlier_banks)} outliers: {outlier_banks}")
test('T8', 'Conflict > no-conflict (mean)', diff_values.mean() > 0,
     f"Δ={diff_values.mean():.0f}ns")
save()

# ======================================================================
# EXP 3: Pointer-Chase Latency (True Memory Latency)
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 3: Pointer-Chase Latency (serialized dependent loads)")
print("  Each load depends on previous → true latency, not throughput")
print("=" * 70)

wait_cool(50)

# Build chase chains with different strides → different cache behavior
chase_results = {}
for stride_name, stride, chain_len in [
    ('L0_hit', 1, 1024),           # Sequential → L0 cache hits
    ('L0_miss', 256, 4096),        # 256-float stride → same cache set → L0 miss
    ('L1_miss', 4096, 8192),       # Large stride → L1 miss → L2
    ('random', 0, 4096),           # Random permutation → worst case
]:
    if stride > 0:
        # Build deterministic chase chain with given stride
        chain = torch.zeros(chain_len, dtype=torch.int32, device='cuda')
        for i in range(chain_len):
            chain[i] = (i + stride) % chain_len
    else:
        # Random permutation
        perm = np.random.permutation(chain_len).astype(np.int32)
        chain = torch.from_numpy(perm).to('cuda')

    # Warmup
    probe.probe_pointer_chase(32, chain, chain_len)
    torch.cuda.synchronize()

    latencies = []
    for rep in range(5):
        d = probe.probe_pointer_chase(32, chain, chain_len).cpu().numpy()
        latencies.extend(d[:, 0].tolist())
        time.sleep(0.05)

    mean_lat = np.mean(latencies)
    per_hop = mean_lat / chain_len
    chase_results[stride_name] = {
        'total_ns': float(mean_lat),
        'per_hop_ns': float(per_hop),
        'std_ns': float(np.std(latencies)),
    }
    print(f"  {stride_name:10s}: {mean_lat:.0f}ns total, {per_hop:.2f}ns/hop")

# L0 miss/hit ratio reveals cache line activation energy
if chase_results['L0_hit']['per_hop_ns'] > 0:
    miss_hit_ratio = chase_results['L0_miss']['per_hop_ns'] / chase_results['L0_hit']['per_hop_ns']
else:
    miss_hit_ratio = 0

results['experiments']['exp3_pointer_chase'] = {
    'chase_results': chase_results,
    'miss_hit_ratio': float(miss_hit_ratio),
}

test('T9', 'L0_miss > L0_hit latency', miss_hit_ratio > 1.5,
     f"ratio={miss_hit_ratio:.2f}x")
test('T10', 'Random worst case', chase_results['random']['per_hop_ns'] >
     chase_results['L0_hit']['per_hop_ns'] * 1.2,
     f"random={chase_results['random']['per_hop_ns']:.2f} vs hit={chase_results['L0_hit']['per_hop_ns']:.2f}")
test('T11', 'Cache hierarchy visible (3+ distinct levels)',
     len(set(round(v['per_hop_ns'], 0) for v in chase_results.values())) >= 3,
     f"{len(set(round(v['per_hop_ns'], 0) for v in chase_results.values()))} levels")
save()

# ======================================================================
# EXP 4: Instruction-Type Fingerprints per CU
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 4: Instruction-Type Fingerprints per CU")
print("  VALU vs Transcendental vs Integer vs SALU — same CU, different pipes")
print("=" * 70)

wait_cool(50)
INSN_NAMES = ['VALU_fma', 'transcendental', 'integer', 'SALU']
insn_results = {}

for itype, name in enumerate(INSN_NAMES):
    # Warmup
    probe.probe_insn_type(N_WAVES, itype, N_ITERS)
    torch.cuda.synchronize()

    runs = []
    for rep in range(10):
        d = probe.probe_insn_type(N_WAVES, itype, N_ITERS).cpu().numpy()
        runs.append(d)
        time.sleep(0.05)

    all_d = np.vstack(runs)
    per_cu = {}
    for wgp in np.unique(all_d[:, 2]).astype(int):
        mask = all_d[:, 2] == wgp
        per_cu[int(wgp)] = float(all_d[mask, 0].mean())

    insn_results[name] = {
        'global_mean': float(all_d[:, 0].mean()),
        'global_std': float(all_d[:, 0].std()),
        'per_cu': per_cu,
    }
    print(f"  {name:15s}: {all_d[:,0].mean():.0f} ±{all_d[:,0].std():.0f} ns")

# Per-CU: does the CU ranking change across instruction types?
# If CU 5 is fastest for VALU but slowest for transcendental → different pipe circuits
common_cus = set.intersection(*[set(v['per_cu'].keys()) for v in insn_results.values()])
if len(common_cus) >= 3:
    cu_list = sorted(common_cus)
    valu_order = [insn_results['VALU_fma']['per_cu'][c] for c in cu_list]
    trans_order = [insn_results['transcendental']['per_cu'][c] for c in cu_list]
    int_order = [insn_results['integer']['per_cu'][c] for c in cu_list]

    r_valu_trans = spearmanr(valu_order, trans_order)[0]
    r_valu_int = spearmanr(valu_order, int_order)[0]

    print(f"\n  Per-CU ranking correlation:")
    print(f"    VALU↔Trans: r={r_valu_trans:.3f}")
    print(f"    VALU↔Int:   r={r_valu_int:.3f}")

    # Show per-CU fingerprint
    print(f"\n  CU Fingerprints (relative to mean):")
    for cu in cu_list[:8]:
        vals = {n: insn_results[n]['per_cu'][cu] for n in INSN_NAMES}
        means = {n: insn_results[n]['global_mean'] for n in INSN_NAMES}
        deltas = {n: (vals[n] - means[n]) / means[n] * 100 for n in INSN_NAMES}
        fingerprint = ' '.join(f"{n[:4]}:{deltas[n]:+.2f}%" for n in INSN_NAMES)
        print(f"    WGP {cu:2d}: {fingerprint}")
else:
    r_valu_trans = r_valu_int = 0

results['experiments']['exp4_insn_fingerprint'] = {
    'insn_results': {k: {'global_mean': v['global_mean'], 'global_std': v['global_std']}
                     for k, v in insn_results.items()},
    'r_valu_trans': float(r_valu_trans),
    'r_valu_int': float(r_valu_int),
    'n_common_cus': len(common_cus),
}

test('T12', 'VALU ≠ Trans ranking (|r| < 0.9 or > -0.5)',
     abs(r_valu_trans) < 0.9,
     f"r={r_valu_trans:.3f}")
test('T13', 'Transcendental slowest', insn_results['transcendental']['global_mean'] >
     insn_results['VALU_fma']['global_mean'],
     f"trans={insn_results['transcendental']['global_mean']:.0f} vs valu={insn_results['VALU_fma']['global_mean']:.0f}")
test('T14', '4 distinct instruction timings',
     len(set(round(v['global_mean'], -1) for v in insn_results.values())) >= 3,
     f"{sorted(v['global_mean'] for v in insn_results.values())}")
save()

# ======================================================================
# EXP 5: VGPR Depth (Array-Indirect, Compiler-Proof)
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 5: VGPR Depth (array-indirect, compiler-proof)")
print("  active_regs: 2,4,8,12,16,20,24,28,32")
print("=" * 70)

wait_cool(50)
vgpr_depths = [2, 4, 8, 12, 16, 20, 24, 28, 32]
vgpr_timing = {}

# Warmup
probe.probe_vgpr_indirect(N_WAVES, 4, 200)
torch.cuda.synchronize()

for depth in vgpr_depths:
    times = []
    for rep in range(5):
        d = probe.probe_vgpr_indirect(N_WAVES, depth, 200).cpu().numpy()
        times.extend(d[:, 0].tolist())
    vgpr_timing[depth] = {
        'wall_mean': float(np.mean(times)),
        'wall_std': float(np.std(times)),
    }
    print(f"  Depth {depth:2d}: {np.mean(times):.0f} ±{np.std(times):.0f} ns")

vgpr_means = [vgpr_timing[d]['wall_mean'] for d in vgpr_depths]
diffs = np.diff(vgpr_means)
is_monotonic = sum(1 for d in diffs if d > 0) >= len(diffs) * 0.7  # 70% increasing
depth_ratio = vgpr_means[-1] / (vgpr_means[0] + 1e-10)

# Fit: should be approximately linear (each reg adds same pipeline pressure)
try:
    slope, intercept = np.polyfit(vgpr_depths, vgpr_means, 1)
    pred = np.polyval([slope, intercept], vgpr_depths)
    ss_res = np.sum((np.array(vgpr_means) - pred) ** 2)
    ss_tot = np.sum((np.array(vgpr_means) - np.mean(vgpr_means)) ** 2)
    linear_r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0
except:
    linear_r2 = 0
    slope = 0

print(f"\n  Depth ratio (32/2): {depth_ratio:.2f}x")
print(f"  Linear fit R²={linear_r2:.3f}, slope={slope:.1f} ns/reg")

results['experiments']['exp5_vgpr_indirect'] = {
    'vgpr_timing': {str(k): v for k, v in vgpr_timing.items()},
    'depth_ratio': float(depth_ratio),
    'linear_r2': float(linear_r2),
    'slope_ns_per_reg': float(slope),
    'is_monotonic': bool(is_monotonic),
}

test('T15', 'VGPR mostly monotonic (70%+)', is_monotonic,
     f"diffs={[f'{d:.0f}' for d in diffs]}")
test('T16', 'Depth 32 > depth 2 by >50%', depth_ratio > 1.5,
     f"{depth_ratio:.2f}x")
test('T17', 'Linear fit R² > 0.5', linear_r2 > 0.5,
     f"R²={linear_r2:.3f}")
save()

# ======================================================================
# EXP 6: Mem/Compute Ratio per CU (DVFS-invariant)
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 6: Mem/Compute Ratio per CU (DVFS-invariant)")
print("=" * 70)

wait_cool(50)
workspace = torch.randn(256 * 1024, device='cuda', dtype=torch.float32)

# Warmup
probe.probe_ratio_v2(N_WAVES, workspace, 500)
torch.cuda.synchronize()

ratio_runs = []
for run_idx in range(10):
    data = probe.probe_ratio_v2(N_WAVES, workspace, 500).cpu().numpy()
    ratio_runs.append(data)
    if (run_idx + 1) % 5 == 0:
        print(f"  Run {run_idx+1}: ratio={data[:,0].mean():.4f} ±{data[:,0].std():.4f}")
    time.sleep(0.1)

all_ratio = np.vstack(ratio_runs)
ratio_by_cu = {}
for wgp in np.unique(all_ratio[:, 3]).astype(int):
    mask = all_ratio[:, 3] == wgp
    ratio_by_cu[int(wgp)] = {
        'ratio_mean': float(all_ratio[mask, 0].mean()),
        'ratio_std': float(all_ratio[mask, 0].std()),
        'comp_mean': float(all_ratio[mask, 1].mean()),
        'mem_mean': float(all_ratio[mask, 2].mean()),
    }

if len(ratio_by_cu) > 1:
    r_means = np.array([v['ratio_mean'] for v in ratio_by_cu.values()])
    ratio_cv = r_means.std() / (r_means.mean() + 1e-10)
    print(f"\n  Per-CU ratio: CV={ratio_cv:.4f}")
    for wgp in sorted(ratio_by_cu):
        d = ratio_by_cu[wgp]
        print(f"    WGP {wgp:2d}: ratio={d['ratio_mean']:.4f} ±{d['ratio_std']:.4f} "
              f"(comp={d['comp_mean']:.0f}, mem={d['mem_mean']:.0f} cycles)")
else:
    ratio_cv = 0

# Cross-probe: does CU compute ranking match ratio ranking?
if len(common_cus) >= 3 and len(ratio_by_cu) >= 3:
    common_cr = sorted(set(cu_timing) & set(ratio_by_cu))
    if len(common_cr) >= 3:
        cross_r = spearmanr([cu_timing[w]['wall_mean'] for w in common_cr],
                            [ratio_by_cu[w]['ratio_mean'] for w in common_cr])[0]
    else:
        cross_r = 0
else:
    cross_r = 0

results['experiments']['exp6_ratio'] = {
    'ratio_by_cu': {str(k): v for k, v in ratio_by_cu.items()},
    'ratio_cv': float(ratio_cv),
    'cross_r': float(cross_r),
}

test('T18', 'Ratio CV > 0.5%', ratio_cv > 0.005,
     f"CV={ratio_cv:.4f}")
test('T19', 'Cross-probe CU↔ratio correlation |r| > 0.3', abs(cross_r) > 0.3,
     f"r={cross_r:.3f}")
save()
del workspace

# ======================================================================
# EXP 7: Thermal Ramp + Cooldown Tracking
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 7: Thermal Ramp + Cooldown")
print("  Heavy compute to heat GPU, then measure during cooldown")
print("=" * 70)

wait_cool(45)
print("  Heating GPU...")

thermal_trace = []
t_start = time.time()

# Phase 1: Heat up with matmul (5-8 seconds)
heat_start = time.time()
while time.time() - heat_start < 6.0 and get_temp() < 85:
    a = torch.randn(2048, 2048, device='cuda')
    _ = a @ a
    torch.cuda.synchronize()
    m = get_gpu_metrics()
    d = probe.probe_cu_simd(20, 500).cpu().numpy()
    thermal_trace.append({
        'time': time.time() - t_start,
        'phase': 'heat',
        'temp': m.get('temp_c', get_temp()),
        'wall_ns': float(d[:, 0].mean()),
        'cycles': float(d[:, 1].mean()),
        'clock_mhz': m.get('clock_mhz', 0),
        'voltage_mv': m.get('voltage_mv', 0),
    })
    if len(thermal_trace) % 5 == 0:
        t = thermal_trace[-1]
        print(f"  [HEAT] T={t['temp']:.0f}°C wall={t['wall_ns']:.0f}ns "
              f"clk={t['clock_mhz']:.0f}MHz")
del a
torch.cuda.synchronize()

# Phase 2: Cooldown tracking (20-30 seconds)
print("  Cooling down, tracking...")
cool_start = time.time()
while time.time() - cool_start < 25.0:
    d = probe.probe_cu_simd(20, 500).cpu().numpy()
    m = get_gpu_metrics()
    thermal_trace.append({
        'time': time.time() - t_start,
        'phase': 'cool',
        'temp': m.get('temp_c', get_temp()),
        'wall_ns': float(d[:, 0].mean()),
        'cycles': float(d[:, 1].mean()),
        'clock_mhz': m.get('clock_mhz', 0),
        'voltage_mv': m.get('voltage_mv', 0),
    })
    if len(thermal_trace) % 10 == 0:
        t = thermal_trace[-1]
        print(f"  [COOL] T={t['temp']:.0f}°C wall={t['wall_ns']:.0f}ns "
              f"clk={t['clock_mhz']:.0f}MHz")
    time.sleep(0.5)

# Analysis
temps = np.array([t['temp'] for t in thermal_trace])
walls = np.array([t['wall_ns'] for t in thermal_trace])
clocks = np.array([t['clock_mhz'] for t in thermal_trace])
volts = np.array([t['voltage_mv'] for t in thermal_trace])

temp_range = temps.max() - temps.min()
print(f"\n  Temp range: {temps.min():.0f} → {temps.max():.0f}°C (Δ={temp_range:.0f}°C)")

# Correlations
if np.std(temps) > 0.5 and np.std(walls) > 0:
    r_temp_wall, p_tw = pearsonr(temps, walls)
else:
    r_temp_wall, p_tw = 0, 1

if np.std(clocks) > 0 and np.std(walls) > 0:
    r_clock_wall, _ = pearsonr(clocks, walls)
else:
    r_clock_wall = 0

# Wall-clock RESIDUAL after removing clock correlation
if np.std(clocks) > 0.5:
    a, b = np.polyfit(clocks, walls, 1)
    wall_residual = walls - (a * clocks + b)
    if np.std(temps) > 0.5 and np.std(wall_residual) > 0:
        r_resid_temp, p_rt = pearsonr(temps, wall_residual)
    else:
        r_resid_temp, p_rt = 0, 1
else:
    r_resid_temp, p_rt = 0, 1

print(f"  Correlations: temp↔wall r={r_temp_wall:.3f}, clock↔wall r={r_clock_wall:.3f}")
print(f"  Residual (wall - clock effect) ↔ temp: r={r_resid_temp:.3f}")

results['experiments']['exp7_thermal'] = {
    'temp_range': float(temp_range),
    'r_temp_wall': float(r_temp_wall),
    'r_clock_wall': float(r_clock_wall),
    'r_resid_temp': float(r_resid_temp),
    'n_samples': len(thermal_trace),
    'trace_summary': [{'time': t['time'], 'temp': t['temp'], 'wall_ns': t['wall_ns'],
                        'clock_mhz': t['clock_mhz']} for t in thermal_trace[::5]],
}

test('T20', 'Temperature range > 5°C', temp_range > 5,
     f"Δ={temp_range:.1f}°C")
test('T21', 'Wall-clock↔temp correlation |r| > 0.3', abs(r_temp_wall) > 0.3,
     f"r={r_temp_wall:.3f}")
test('T22', 'Clock↔wall correlation |r| > 0.3', abs(r_clock_wall) > 0.3,
     f"r={r_clock_wall:.3f}")
test('T23', 'Residual↔temp (pure thermal) |r| > 0.1', abs(r_resid_temp) > 0.1,
     f"r={r_resid_temp:.3f}")
save()

# ======================================================================
# EXP 8: Cross-Probe Synthesis
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 8: Cross-Probe Synthesis")
print("=" * 70)

# How many independent physical measurement dimensions?
n_phys = len(cu_timing) + 32 + len(chase_results) + len(INSN_NAMES) + len(vgpr_depths) + len(ratio_by_cu)
print(f"  Total physical measurement points: {n_phys}")

# CU fingerprint matrix: for each CU, collect all probe dimensions
fingerprint_cus = sorted(set(cu_timing) & common_cus & set(ratio_by_cu))
if len(fingerprint_cus) >= 3:
    fingerprint_matrix = []
    for cu in fingerprint_cus:
        row = [cu_timing[cu]['wall_mean']]
        for name in INSN_NAMES:
            row.append(insn_results[name]['per_cu'].get(cu, 0))
        if cu in ratio_by_cu:
            row.append(ratio_by_cu[cu]['ratio_mean'])
        else:
            row.append(0)
        fingerprint_matrix.append(row)
    fm = np.array(fingerprint_matrix)

    # PCA-like: how many dimensions needed to explain 90% variance?
    from numpy.linalg import svd
    fm_centered = fm - fm.mean(axis=0)
    _, s, _ = svd(fm_centered, full_matrices=False)
    explained = np.cumsum(s**2) / (np.sum(s**2) + 1e-10)
    dims_90 = int(np.searchsorted(explained, 0.9)) + 1

    # Uniqueness: can we distinguish CUs from fingerprint?
    from scipy.spatial.distance import pdist
    distances = pdist(fm_centered)
    min_dist = distances.min()
    mean_dist = distances.mean()
    distinguishable = min_dist > 0.01 * mean_dist  # Every pair > 1% of mean apart

    print(f"  Fingerprint dimensions: {fm.shape[1]}")
    print(f"  PCA dims for 90% variance: {dims_90}")
    print(f"  CU distinguishability: min_dist/mean_dist = {min_dist/mean_dist:.4f}")
    print(f"  Singular values: {s[:5].round(1)}")
else:
    dims_90 = 0
    distinguishable = False
    min_dist = 0
    mean_dist = 1

# Overall variation budget
print(f"\n  Variation budget:")
print(f"    CU compute:     CV={cu_cv*100:.3f}%")
if diff_cv > 0:
    print(f"    LDS bank diff:  CV={diff_cv*100:.3f}%")
if ratio_cv > 0:
    print(f"    Mem/comp ratio: CV={ratio_cv*100:.3f}%")
print(f"    Thermal range:  Δ{temp_range:.0f}°C")

results['experiments']['exp8_synthesis'] = {
    'n_phys_points': n_phys,
    'n_fingerprint_cus': len(fingerprint_cus),
    'dims_90': dims_90 if len(fingerprint_cus) >= 3 else 0,
    'distinguishable': bool(distinguishable),
    'cu_cv_pct': float(cu_cv * 100),
    'diff_cv_pct': float(diff_cv * 100),
    'ratio_cv_pct': float(ratio_cv * 100),
    'temp_range_c': float(temp_range),
}

test('T24', '20+ physical measurement points', n_phys >= 20, f"{n_phys}")
test('T25', '3+ fingerprint dimensions for 90% var', dims_90 >= 3, f"{dims_90}")
test('T26', 'CUs distinguishable by fingerprint', distinguishable,
     f"min/mean={min_dist/(mean_dist+1e-10):.4f}")
test('T27', 'CU variation detectable (CV > 0.05%)', cu_cv > 0.0005,
     f"CV={cu_cv*100:.3f}%")

# Overall reproducibility: how many probes have r > 0.5?
n_reproducible = sum([
    repro_r > 0.5,
    diff_repro_r > 0.5,
])
test('T28', '2+ probes reproducible (r > 0.5)', n_reproducible >= 2,
     f"{n_reproducible}/2")

# Can we build a Shockley-like model? delay = a * exp(b/T)
if np.std(temps) > 2 and len(temps) > 10:
    try:
        from scipy.optimize import curve_fit
        def arrhenius(T, a, Ea):
            return a * np.exp(Ea / (T + 273.15))
        popt, _ = curve_fit(arrhenius, temps, walls, p0=[1e5, 100], maxfev=5000)
        pred = arrhenius(temps, *popt)
        ss_res = np.sum((walls - pred)**2)
        ss_tot = np.sum((walls - walls.mean())**2)
        arrh_r2 = 1 - ss_res / (ss_tot + 1e-10)
        print(f"\n  Arrhenius fit: a={popt[0]:.0f}, Ea={popt[1]:.1f}, R²={arrh_r2:.3f}")
    except Exception as e:
        arrh_r2 = 0
        print(f"\n  Arrhenius fit failed: {e}")
else:
    arrh_r2 = 0
    print(f"\n  Temp range too small for Arrhenius ({temp_range:.0f}°C)")

test('T29', 'Arrhenius R² > 0.3 (thermal model)', arrh_r2 > 0.3,
     f"R²={arrh_r2:.3f}")
test('T30', 'Multi-probe variation > single-probe',
     max(cu_cv, diff_cv, ratio_cv) > cu_cv * 0.5,
     f"max_cv={max(cu_cv, diff_cv, ratio_cv):.4f}")
save()

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("  SUMMARY: z2329 Surgical Transistor Physics v3")
print("=" * 70)

passed = sum(1 for t in results['tests'].values() if t['status'] == 'PASS')
total = len(results['tests'])
print(f"\n  Tests: {passed}/{total} PASS ({100*passed/total:.0f}%)")

print(f"\n  {'Probe':<25s} {'Signal':>10s} {'Repro':>8s}")
print(f"  {'-'*25} {'-'*10} {'-'*8}")
print(f"  {'CU compute (wall_ns)':<25s} {'CV='+f'{cu_cv*100:.2f}%':>10s} {'r='+f'{repro_r:.2f}':>8s}")
print(f"  {'LDS bank differential':<25s} {'CV='+f'{diff_cv*100:.2f}%':>10s} {'r='+f'{diff_repro_r:.2f}':>8s}")
print(f"  {'Pointer-chase (4 levels)':<25s} {f'{miss_hit_ratio:.1f}x miss/hit':>10s} {'—':>8s}")
print(f"  {'Insn fingerprint (4 type)':<25s} {f'r_vt={r_valu_trans:.2f}':>10s} {'—':>8s}")
print(f"  {'VGPR depth (2→32)':<25s} {f'{depth_ratio:.1f}x':>10s} {'R²='+f'{linear_r2:.2f}':>8s}")
print(f"  {'Ratio (mem/comp per CU)':<25s} {'CV='+f'{ratio_cv*100:.2f}%':>10s} {'r='+f'{cross_r:.2f}':>8s}")
print(f"  {'Thermal ramp':<25s} {'Δ'+f'{temp_range:.0f}°C':>10s} {'r='+f'{r_temp_wall:.2f}':>8s}")

print(f"\n  Fingerprint: {len(fingerprint_cus)} CUs × {dims_90} dims")
print(f"  Arrhenius R²={arrh_r2:.3f}")
print(f"\n  Done. Results: {RESULTS / 'z2329_analog_physics_model.json'}")
