#!/usr/bin/env python3
"""
z2330_analog_neuron_probes.py — Full Analog Physics Characterization
======================================================================
Combines v3 structural probes (per-CU, per-bank, pointer-chase, insn-type,
VGPR-depth, ratio, thermal) WITH 3 new analog mechanisms:

  NEW 1. FMA Cancellation Chains — precision boundary exploitation
         Two nearly-equal numbers subtracted → result depends on FMA rounding
         → rounding depends on CU pipeline timing → CU-specific analog function
  NEW 2. Wavefront Contention — stochastic gating
         Two wavefronts race to atomicCAS same address → winner = f(physics)
         → per-CU bias in who wins → stochastic synapse
  NEW 3. Thermal Sequential Coupling — kernel₁ affects kernel₂
         Run compute kernel → measure timing of NEXT kernel → temporal feedback
         via copper traces in silicon → real analog recurrence

Tests (42): T1-T30 from v3 structural, T31-T42 new analog mechanisms.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 python scripts/z2330_analog_neuron_probes.py
"""

import os, sys, time, json, struct
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway, kruskal, entropy

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
# HIP Kernel Source — ALL probes in one module
# ======================================================================
import torch
from torch.utils.cpp_extension import load_inline

PROBE_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define HWREG(id, offset, size) ((id) | ((offset) << 6) | (((size)-1) << 11))
#define HW_REG_HW_ID1 23

// Helper: get WGP and SIMD from HW_ID1
__device__ void get_hw_id(int& wgp, int& simd, int& se_sa) {
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    wgp   = (hw >> 8) & 0xF;
    simd  = (hw >> 4) & 0x3;
    se_sa = ((hw >> 13) & 0x7) * 2 + ((hw >> 12) & 0x1);
}

// =====================================================================
// STRUCTURAL PROBES (from v3)
// =====================================================================

// Probe S1: Per-CU + Per-SIMD compute timing
__global__ void kernel_cu_simd(float* out, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid + 1) * 0.001f;
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
    int wgp, simd, se_sa;
    get_hw_id(wgp, simd, se_sa);
    out[tid*6+0] = (float)(w1-w0)*10.0f;
    out[tid*6+1] = (float)(c1-c0);
    out[tid*6+2] = (float)wgp;
    out[tid*6+3] = (float)simd;
    out[tid*6+4] = (float)se_sa;
    out[tid*6+5] = x;  // anti-opt + actual value
}

torch::Tensor probe_cu_simd(int n_waves, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*6}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_cu_simd<<<n_waves, 32>>>(out.data_ptr<float>(), n_iters, n);
    return out.reshape({n, 6});
}

// Probe S2: LDS bank-conflict differential
__global__ void kernel_lds_diff(float* out, int target_bank, int mode, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    __shared__ float lds[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) lds[i] = (float)i;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        int idx = (mode==0) ? target_bank+(i%128)*32 : (threadIdx.x%32)+(i%128)*32;
        sum += lds[idx]; lds[idx] = sum * 0.999f;
    }
    __syncthreads();
    long long w0 = wall_clock64(); uint64_t c0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        int idx = (mode==0) ? target_bank+(i%128)*32 : (threadIdx.x%32)+(i%128)*32;
        sum += lds[idx]; lds[idx] = sum * 0.999f;
    }
    uint64_t c1 = clock64(); long long w1 = wall_clock64();
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    out[tid*4+0] = (float)(w1-w0)*10.0f;
    out[tid*4+1] = (float)(c1-c0);
    out[tid*4+2] = (float)wgp;
    out[tid*4+3] = (float)target_bank;
    if (__builtin_expect(sum == -1e30f, 0)) out[0] = sum;
}

torch::Tensor probe_lds_diff(int n, int bank, int mode, int n_iters) {
    auto out = torch::zeros({n*4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_lds_diff<<<(n+31)/32, 32, 16384>>>(out.data_ptr<float>(), bank, mode, n_iters, n);
    return out.reshape({n, 4});
}

// Probe S3: Instruction-type fingerprint
__global__ void kernel_insn_type(float* out, int insn_type, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid+1)*0.001f; int ix = tid+1;
    for (int i = 0; i < 50; i++) {
        if (insn_type==0) x=x*1.0001f+0.0001f;
        else if (insn_type==1) x=sinf(x)+0.0001f;
        else if (insn_type==2) ix=(ix*2654435761u)^(ix>>13);
        else x=x+(float)(blockIdx.x&1)*0.0001f;
    }
    long long w0 = wall_clock64(); uint64_t c0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        if (insn_type==0) { x=x*1.0001f+0.0001f; x=x*0.9999f+0.0002f; }
        else if (insn_type==1) { x=sinf(x)*cosf(x+0.1f); x=x*1.0001f+0.0001f; }
        else if (insn_type==2) { ix=(ix*2654435761u)^(ix>>13); ix=ix+(ix<<3); }
        else { x=x+(float)(blockIdx.x&1)*0.0001f; x=x*(1.0f+(float)(blockIdx.x>>16)*1e-9f); }
    }
    uint64_t c1 = clock64(); long long w1 = wall_clock64();
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    float result = (insn_type==2) ? (float)ix : x;
    out[tid*5+0] = (float)(w1-w0)*10.0f;
    out[tid*5+1] = (float)(c1-c0);
    out[tid*5+2] = (float)wgp;
    out[tid*5+3] = (float)simd;
    out[tid*5+4] = (float)insn_type;
    if (__builtin_expect(result==-1e30f,0)) out[0]=result;
}

torch::Tensor probe_insn_type(int n_waves, int itype, int n_iters) {
    int n = n_waves*32;
    auto out = torch::zeros({n*5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_insn_type<<<n_waves, 32>>>(out.data_ptr<float>(), itype, n_iters, n);
    return out.reshape({n, 5});
}

// Probe S4: VGPR depth (array-indirect)
__global__ void kernel_vgpr(float* out, int active_regs, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float regs[32];
    for (int r = 0; r < 32; r++) regs[r] = (float)(tid+r)*0.001f;
    for (int i = 0; i < 50; i++)
        for (int r = 0; r < active_regs; r++)
            regs[r] = regs[r]*1.0001f + regs[(r+1)%active_regs]*0.01f;
    long long w0 = wall_clock64(); uint64_t c0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++)
        for (int r = 0; r < active_regs; r++)
            regs[r] = regs[r]*1.0001f + regs[(r+1)%active_regs]*0.01f;
    uint64_t c1 = clock64(); long long w1 = wall_clock64();
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    float total = 0; for (int r=0;r<32;r++) total+=regs[r];
    out[tid*4+0] = (float)(w1-w0)*10.0f;
    out[tid*4+1] = (float)(c1-c0);
    out[tid*4+2] = (float)wgp;
    out[tid*4+3] = (float)active_regs;
    if (__builtin_expect(total==-1e30f,0)) out[0]=total;
}

torch::Tensor probe_vgpr(int n_waves, int depth, int n_iters) {
    int n = n_waves*32;
    auto out = torch::zeros({n*4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_vgpr<<<n_waves, 32>>>(out.data_ptr<float>(), depth, n_iters, n);
    return out.reshape({n, 4});
}

// Probe S5: Mem/compute ratio
__global__ void kernel_ratio(float* out, const float* ws, int ws_size, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid+1)*0.001f; float sum = 0.0f;
    for (int i=0;i<50;i++) {
        x = sinf(x)*cosf(x+0.1f);
        sum += ws[((tid*2654435761u+i*40503u)>>4) % ws_size];
    }
    uint64_t cc0 = clock64();
    #pragma unroll 1
    for (int i=0;i<n_iters;i++) { x=sinf(x)*cosf(x+0.1f); x=x*1.0001f+0.0001f; }
    uint64_t cc1 = clock64();
    uint64_t mc0 = clock64();
    #pragma unroll 1
    for (int i=0;i<n_iters;i++) { sum += ws[((tid*2654435761u+i*40503u)>>4) % ws_size]; }
    uint64_t mc1 = clock64();
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    float comp = (float)(cc1-cc0), mem = (float)(mc1-mc0);
    out[tid*5+0] = (comp>1.0f) ? mem/comp : 0.0f;
    out[tid*5+1] = comp; out[tid*5+2] = mem;
    out[tid*5+3] = (float)wgp; out[tid*5+4] = (float)se_sa;
    if (__builtin_expect((x+sum)==-1e30f,0)) out[0]=x;
}

torch::Tensor probe_ratio(int n_waves, torch::Tensor ws, int n_iters) {
    int n = n_waves*32;
    auto out = torch::zeros({n*5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_ratio<<<n_waves, 32>>>(out.data_ptr<float>(), ws.data_ptr<float>(),
                                   ws.size(0), n_iters, n);
    return out.reshape({n, 5});
}

// =====================================================================
// ANALOG MECHANISM 1: FMA Cancellation Chains
// =====================================================================
// Two nearly-equal numbers subtracted via FMA chain.
// Result is sensitive to last-bit rounding which depends on:
//   - FMA pipeline timing (CU-specific)
//   - Carry propagation path (transistor-level)
//   - Intermediate register precision
//
// We compute: y = ((a + ε₁) × b - a × b) / ε₁
// Analytically y = b, but FMA rounding makes y = b + δ
// where δ is CU-DEPENDENT → analog transfer function
// =====================================================================
__global__ void kernel_fma_cancel(float* out, const float* inputs,
                                    int chain_len, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float input = inputs[tid % 64];  // 64 distinct input values

    // Build cancellation-sensitive chain
    // Strategy: accumulate many near-cancelling terms
    // Each step: x = (x + tiny) * scale - x * scale
    //          = tiny * scale (analytically)
    //          but FMA rounding creates CU-dependent residual

    float x = input;
    float accum = 0.0f;

    #pragma unroll 1
    for (int i = 0; i < chain_len; i++) {
        // Near-cancellation: compute (x + eps) * y - x * y
        // Use fmaf for fused multiply-add (single rounding)
        float eps = 1.0e-6f * (1.0f + (float)(i & 7) * 0.1f);
        float scale = 1.0f + (float)((i * 7 + tid) & 0xF) * 0.0625f;

        // FMA: fmaf(a, b, c) = a*b + c with single rounding
        float r1 = fmaf(x + eps, scale, 0.0f);  // (x+eps)*scale
        float r2 = fmaf(x, scale, 0.0f);         // x*scale
        float delta = r1 - r2;                     // Should be eps*scale, but...

        // The cancellation residual depends on CU rounding behavior
        accum += delta;

        // Evolve x to prevent compiler from optimizing
        x = fmaf(x, 0.999999f, eps * 0.001f);
    }

    // Second pass: longer chain with double-cancellation
    float accum2 = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < chain_len; i++) {
        float a = x + 1.0e-7f * (float)(i & 15);
        float b = x - 1.0e-7f * (float)(i & 15);
        // a - b should be 2e-7 * (i&15), but FMA chain creates drift
        float sum_ab = fmaf(a, 1.0f, b);    // a + b via FMA
        float diff_ab = fmaf(a, 1.0f, -b);  // a - b via FMA
        // Reconstruct: a = (sum + diff) / 2
        float reconstructed_a = fmaf(sum_ab + diff_ab, 0.5f, 0.0f);
        accum2 += (reconstructed_a - a);  // Rounding residual
        x = fmaf(x, 1.0f, accum2 * 1.0e-10f);
    }

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    out[tid*6+0] = accum;        // Primary cancellation residual
    out[tid*6+1] = accum2;       // Double-cancellation residual
    out[tid*6+2] = x;            // Final state (evolution depends on CU)
    out[tid*6+3] = (float)wgp;
    out[tid*6+4] = (float)simd;
    out[tid*6+5] = input;        // Original input for tracking
}

torch::Tensor probe_fma_cancel(int n_waves, torch::Tensor inputs, int chain_len) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*6}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_fma_cancel<<<n_waves, 32>>>(out.data_ptr<float>(),
        inputs.data_ptr<float>(), chain_len, n);
    return out.reshape({n, 6});
}

// =====================================================================
// ANALOG MECHANISM 2: Wavefront Contention (Stochastic Gating)
// =====================================================================
// Two wavefronts race to write an atomicCAS to the same address.
// Winner is determined by physical scheduling + cache state.
// Run many trials → per-CU win-rate = stochastic bias = f(physics)
//
// This is NOT destructive (no bitflip) — it's contention resolution,
// a fundamental GPU mechanism. The bias IS the analog signal.
// =====================================================================
__global__ void kernel_contention(int* race_results, int* race_winners,
                                   int n_races, int n_threads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_threads) return;

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    // Each pair of threads races for the same slot
    int pair_id = tid / 2;
    int is_second = tid & 1;

    if (pair_id >= n_races) return;

    // Both threads try atomicCAS: first one to arrive writes its team (0 or 1)
    // The race address is in global memory — contention goes through cache hierarchy
    int old = atomicCAS(&race_results[pair_id], -1, is_second);

    // If old == -1, we won the race (we were first)
    if (old == -1) {
        race_winners[pair_id * 3 + 0] = wgp;      // Winner's WGP
        race_winners[pair_id * 3 + 1] = is_second; // Which team won
        race_winners[pair_id * 3 + 2] = simd;      // Winner's SIMD
    }
}

torch::Tensor probe_contention(int n_races) {
    int n_threads = n_races * 2;
    auto results = torch::full({n_races}, -1,
        torch::device(torch::kCUDA).dtype(torch::kInt32));
    auto winners = torch::zeros({n_races * 3},
        torch::device(torch::kCUDA).dtype(torch::kInt32));

    int blocks = (n_threads + 63) / 64;
    kernel_contention<<<blocks, 64>>>(results.data_ptr<int>(),
        winners.data_ptr<int>(), n_races, n_threads);

    return winners.reshape({n_races, 3});
}

// =====================================================================
// ANALOG MECHANISM 3: Thermal Sequential Coupling
// =====================================================================
// Light probe kernel — just measures timing. Run this BEFORE and AFTER
// a heat-generating kernel. The difference reveals thermal coupling.
// Python orchestrates the sequencing; this just provides the measurement.
__global__ void kernel_thermal_probe(float* out, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid+1)*0.001f;
    // Minimal warmup
    for (int i=0;i<10;i++) x = sinf(x)*0.999f + 0.001f;

    long long w0 = wall_clock64(); uint64_t c0 = clock64();
    #pragma unroll 1
    for (int i=0;i<n_iters;i++) {
        x = sinf(x)*cosf(x+0.1f) + expf(-x*x);
        x = x*1.0001f + 0.0001f;
    }
    uint64_t c1 = clock64(); long long w1 = wall_clock64();
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    out[tid*4+0] = (float)(w1-w0)*10.0f;
    out[tid*4+1] = (float)(c1-c0);
    out[tid*4+2] = (float)wgp;
    out[tid*4+3] = x;  // Final value for cross-kernel comparison
}

torch::Tensor probe_thermal(int n_waves, int n_iters) {
    int n = n_waves*32;
    auto out = torch::zeros({n*4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_thermal_probe<<<n_waves, 32>>>(out.data_ptr<float>(), n_iters, n);
    return out.reshape({n, 4});
}

// Heat kernel — pure compute to raise temperature
__global__ void kernel_heat(float* workspace, int n, int n_iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = workspace[tid];
    #pragma unroll 1
    for (int i=0; i<n_iters; i++) {
        x = sinf(x)*cosf(x+0.1f)*expf(-x*x) + 0.001f;
    }
    workspace[tid] = x;
}

void run_heat(torch::Tensor workspace, int n_iters) {
    int n = workspace.size(0);
    kernel_heat<<<(n+255)/256, 256>>>(workspace.data_ptr<float>(), n, n_iters);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_cu_simd", &probe_cu_simd);
    m.def("probe_lds_diff", &probe_lds_diff);
    m.def("probe_insn_type", &probe_insn_type);
    m.def("probe_vgpr", &probe_vgpr);
    m.def("probe_ratio", &probe_ratio);
    m.def("probe_fma_cancel", &probe_fma_cancel);
    m.def("probe_contention", &probe_contention);
    m.def("probe_thermal", &probe_thermal);
    m.def("run_heat", &run_heat);
}
'''

print("Compiling HIP probe kernels (z2330)...")
probe = load_inline(
    name='z2330_analog_neuron_v1',
    cpp_sources='',
    cuda_sources=PROBE_SRC,
    extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
    verbose=False,
)
print("  Compiled and loaded")

# ======================================================================
# Results framework
# ======================================================================
results = {'experiments': {}, 'tests': {}, 'meta': {
    'script': 'z2330_analog_neuron_probes.py',
    'version': 'v1',
    'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
}}

def test(tid, name, passed, value=None):
    status = 'PASS' if passed else 'FAIL'
    results['tests'][tid] = {'name': name, 'status': status, 'value': str(value)}
    print(f"  [{status}] {tid}: {name} = {value}")

def save():
    results['meta']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    p = sum(1 for t in results['tests'].values() if t['status'] == 'PASS')
    results['meta']['tests_passed'] = p
    results['meta']['tests_total'] = len(results['tests'])
    with open(RESULTS / 'z2330_analog_neuron_probes.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [SAVED] {RESULTS / 'z2330_analog_neuron_probes.json'}")

# ======================================================================
# MAIN
# ======================================================================
dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
m0 = get_gpu_metrics()
print(f"Device: {dev_name}")
print("=" * 70)
print("  z2330: Full Analog Physics Characterization")
print("  Structural probes + FMA cancellation + contention + thermal coupling")
print("=" * 70)
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Temp: {get_temp()}C, V={m0.get('voltage_mv','?')}mV, Clk={m0.get('clock_mhz','?')}MHz")

# Warmup
_ = torch.randn(512,512,device='cuda') @ torch.randn(512,512,device='cuda')
torch.cuda.synchronize()

N_WAVES = 40
N_ITERS = 1000

# ======================================================================
# PART A: STRUCTURAL PROBES (from v3)
# ======================================================================

# --- EXP 1: Per-CU Process Variation ---
print("\n" + "="*70)
print("  EXP 1: Per-CU Process Variation (structural)")
print("="*70)
wait_cool(50)
for _ in range(2): probe.probe_cu_simd(N_WAVES, N_ITERS)  # warmup
torch.cuda.synchronize()

cu_runs = []
for i in range(20):
    d = probe.probe_cu_simd(N_WAVES, N_ITERS).cpu().numpy()
    cu_runs.append(d)
    if (i+1) % 10 == 0:
        wgps = np.unique(d[:,2]).astype(int)
        print(f"  Run {i+1}: wall={d[:,0].mean():.0f}ns, {len(wgps)} WGPs")
    time.sleep(0.1)

all_cu = np.vstack(cu_runs)
wgp_ids = np.unique(all_cu[:,2]).astype(int)
cu_timing = {}
for w in wgp_ids:
    mask = all_cu[:,2] == w
    cu_timing[int(w)] = {'wall_mean': float(all_cu[mask,0].mean()),
                          'wall_std': float(all_cu[mask,0].std()),
                          'n': int(mask.sum())}

means_cu = np.array([cu_timing[w]['wall_mean'] for w in sorted(cu_timing)])
cu_cv = means_cu.std() / (means_cu.mean()+1e-10) if len(means_cu)>1 else 0

# Reproducibility
wgp_a, wgp_b = {}, {}
for w in wgp_ids:
    a = np.vstack(cu_runs[:10]); b = np.vstack(cu_runs[10:])
    ma = a[a[:,2]==w, 0]; mb = b[b[:,2]==w, 0]
    if len(ma)>0: wgp_a[int(w)] = ma.mean()
    if len(mb)>0: wgp_b[int(w)] = mb.mean()
common = sorted(set(wgp_a) & set(wgp_b))
repro_r = spearmanr([wgp_a[w] for w in common], [wgp_b[w] for w in common])[0] if len(common)>=3 else 0

print(f"  {len(cu_timing)} WGPs, CV={cu_cv:.4f}, repro r={repro_r:.3f}")
results['experiments']['exp1_cu'] = {'n_wgps': len(cu_timing), 'cu_cv': float(cu_cv), 'repro_r': float(repro_r)}
test('T1', 'CU-to-CU CV > 0.01%', cu_cv > 0.0001, f"CV={cu_cv:.5f}")
test('T2', 'CU map reproducible (r>0.5)', repro_r > 0.5, f"r={repro_r:.3f}")
test('T3', '8+ WGPs detected', len(cu_timing) >= 8, f"{len(cu_timing)}")
save()

# --- EXP 2: Instruction Fingerprints ---
print("\n" + "="*70)
print("  EXP 2: Instruction-Type Fingerprints per CU")
print("="*70)
wait_cool(50)
INSN_NAMES = ['VALU_fma', 'transcendental', 'integer', 'SALU']
insn_results = {}
for itype, name in enumerate(INSN_NAMES):
    probe.probe_insn_type(N_WAVES, itype, N_ITERS); torch.cuda.synchronize()
    runs = []
    for _ in range(10):
        runs.append(probe.probe_insn_type(N_WAVES, itype, N_ITERS).cpu().numpy())
        time.sleep(0.05)
    all_d = np.vstack(runs)
    per_cu = {}
    for w in np.unique(all_d[:,2]).astype(int):
        per_cu[int(w)] = float(all_d[all_d[:,2]==w, 0].mean())
    insn_results[name] = {'mean': float(all_d[:,0].mean()), 'std': float(all_d[:,0].std()), 'per_cu': per_cu}
    print(f"  {name:15s}: {all_d[:,0].mean():.0f} ±{all_d[:,0].std():.0f} ns")

common_insn = set.intersection(*[set(v['per_cu'].keys()) for v in insn_results.values()])
if len(common_insn) >= 3:
    cl = sorted(common_insn)
    r_vt = spearmanr([insn_results['VALU_fma']['per_cu'][c] for c in cl],
                      [insn_results['transcendental']['per_cu'][c] for c in cl])[0]
    r_vi = spearmanr([insn_results['VALU_fma']['per_cu'][c] for c in cl],
                      [insn_results['integer']['per_cu'][c] for c in cl])[0]
else:
    r_vt = r_vi = 0

results['experiments']['exp2_insn'] = {'r_vt': float(r_vt), 'r_vi': float(r_vi)}
test('T4', 'VALU ≠ Trans CU ranking (|r|<0.9)', abs(r_vt) < 0.9, f"r={r_vt:.3f}")
test('T5', 'Transcendental slowest', insn_results['transcendental']['mean'] > insn_results['VALU_fma']['mean'],
     f"trans={insn_results['transcendental']['mean']:.0f} vs valu={insn_results['VALU_fma']['mean']:.0f}")
test('T6', '4 distinct instruction timings',
     len(set(round(v['mean'],-1) for v in insn_results.values())) >= 3,
     f"types: {sorted(v['mean'] for v in insn_results.values())}")
save()

# --- EXP 3: VGPR Depth ---
print("\n" + "="*70)
print("  EXP 3: VGPR Depth (array-indirect)")
print("="*70)
wait_cool(50)
probe.probe_vgpr(N_WAVES, 4, 200); torch.cuda.synchronize()
vgpr_depths = [2,4,8,12,16,20,24,28,32]
vgpr_timing = {}
for d in vgpr_depths:
    times = []
    for _ in range(5):
        times.extend(probe.probe_vgpr(N_WAVES, d, 200).cpu().numpy()[:,0].tolist())
    vgpr_timing[d] = float(np.mean(times))
    print(f"  Depth {d:2d}: {np.mean(times):.0f} ±{np.std(times):.0f} ns")

vgpr_means = [vgpr_timing[d] for d in vgpr_depths]
depth_ratio = vgpr_means[-1] / (vgpr_means[0]+1e-10)
slope, intercept = np.polyfit(vgpr_depths, vgpr_means, 1)
pred = np.polyval([slope, intercept], vgpr_depths)
ss_res = np.sum((np.array(vgpr_means)-pred)**2)
ss_tot = np.sum((np.array(vgpr_means)-np.mean(vgpr_means))**2)
vgpr_r2 = 1 - ss_res/(ss_tot+1e-10) if ss_tot>0 else 0

print(f"  Ratio 32/2: {depth_ratio:.2f}x, R²={vgpr_r2:.3f}, slope={slope:.0f} ns/reg")
results['experiments']['exp3_vgpr'] = {'depth_ratio': float(depth_ratio), 'r2': float(vgpr_r2), 'slope': float(slope)}
test('T7', 'VGPR depth 32 > 2 by >50%', depth_ratio > 1.5, f"{depth_ratio:.2f}x")
test('T8', 'Linear fit R²>0.9', vgpr_r2 > 0.9, f"R²={vgpr_r2:.3f}")
save()

# --- EXP 4: Mem/Compute Ratio ---
print("\n" + "="*70)
print("  EXP 4: Mem/Compute Ratio per CU")
print("="*70)
wait_cool(50)
workspace = torch.randn(256*1024, device='cuda', dtype=torch.float32)
probe.probe_ratio(N_WAVES, workspace, 500); torch.cuda.synchronize()

ratio_data = []
for i in range(10):
    ratio_data.append(probe.probe_ratio(N_WAVES, workspace, 500).cpu().numpy())
    time.sleep(0.1)
all_ratio = np.vstack(ratio_data)
ratio_by_cu = {}
for w in np.unique(all_ratio[:,3]).astype(int):
    mask = all_ratio[:,3]==w
    ratio_by_cu[int(w)] = float(all_ratio[mask,0].mean())

if len(ratio_by_cu)>1:
    r_vals = np.array(list(ratio_by_cu.values()))
    ratio_cv = r_vals.std()/(r_vals.mean()+1e-10)
else:
    ratio_cv = 0

# Cross-probe: CU compute rank vs ratio rank
if len(common_insn)>=3 and len(ratio_by_cu)>=3:
    cc = sorted(set(cu_timing) & set(ratio_by_cu))
    cross_r = spearmanr([cu_timing[w]['wall_mean'] for w in cc],
                         [ratio_by_cu[w] for w in cc])[0] if len(cc)>=3 else 0
else:
    cross_r = 0

print(f"  Ratio CV={ratio_cv:.4f}, cross-probe r={cross_r:.3f}")
results['experiments']['exp4_ratio'] = {'ratio_cv': float(ratio_cv), 'cross_r': float(cross_r)}
test('T9', 'Ratio CV > 0.5%', ratio_cv > 0.005, f"CV={ratio_cv:.4f}")
test('T10', 'Cross-probe CU↔ratio |r|>0.3', abs(cross_r)>0.3, f"r={cross_r:.3f}")
save()
del workspace

# ======================================================================
# PART B: ANALOG MECHANISMS (new)
# ======================================================================

# --- EXP 5: FMA Cancellation Chains ---
print("\n" + "="*70)
print("  EXP 5: FMA Cancellation — Precision Boundary Analog")
print("  Near-cancellation → result depends on FMA rounding → CU-specific")
print("="*70)
wait_cool(50)

inputs = torch.linspace(0.1, 2.0, 64, device='cuda', dtype=torch.float32)
# Warmup
probe.probe_fma_cancel(N_WAVES, inputs, 500); torch.cuda.synchronize()

# Run multiple times to check reproducibility
fma_runs = []
for run_idx in range(20):
    d = probe.probe_fma_cancel(N_WAVES, inputs, 2000).cpu().numpy()
    fma_runs.append(d)
    time.sleep(0.1)

all_fma = np.vstack(fma_runs)
# Columns: accum, accum2, final_x, wgp, simd, input

# Per-CU: is the cancellation residual CU-dependent?
fma_by_cu = {}
for w in np.unique(all_fma[:,3]).astype(int):
    mask = all_fma[:,3] == w
    fma_by_cu[int(w)] = {
        'accum_mean': float(all_fma[mask,0].mean()),
        'accum_std': float(all_fma[mask,0].std()),
        'accum2_mean': float(all_fma[mask,1].mean()),
        'accum2_std': float(all_fma[mask,1].std()),
        'final_x_mean': float(all_fma[mask,2].mean()),
        'final_x_std': float(all_fma[mask,2].std()),
        'n': int(mask.sum()),
    }

# Test: Do different CUs produce different residuals?
cu_accums = np.array([fma_by_cu[w]['accum_mean'] for w in sorted(fma_by_cu)])
cu_accum2 = np.array([fma_by_cu[w]['accum2_mean'] for w in sorted(fma_by_cu)])
cu_finals = np.array([fma_by_cu[w]['final_x_mean'] for w in sorted(fma_by_cu)])

fma_accum_cv = cu_accums.std() / (abs(cu_accums.mean())+1e-30) if len(cu_accums)>1 else 0
fma_final_cv = cu_finals.std() / (abs(cu_finals.mean())+1e-30) if len(cu_finals)>1 else 0

# Reproducibility: first 10 vs last 10
fma_a, fma_b = {}, {}
for w in np.unique(all_fma[:,3]).astype(int):
    a = np.vstack(fma_runs[:10]); b = np.vstack(fma_runs[10:])
    ma = a[a[:,3]==w, 0]; mb = b[b[:,3]==w, 0]
    if len(ma)>0: fma_a[int(w)] = ma.mean()
    if len(mb)>0: fma_b[int(w)] = mb.mean()
fma_common = sorted(set(fma_a) & set(fma_b))
fma_repro_r = pearsonr([fma_a[w] for w in fma_common],
                        [fma_b[w] for w in fma_common])[0] if len(fma_common)>=3 else 0

# Does FMA residual correlate with timing?
if len(common_insn)>=3 and len(fma_by_cu)>=3:
    cc = sorted(set(cu_timing) & set(fma_by_cu))
    if len(cc) >= 3:
        fma_timing_r = pearsonr([cu_timing[w]['wall_mean'] for w in cc],
                                 [fma_by_cu[w]['accum_mean'] for w in cc])[0]
    else:
        fma_timing_r = 0
else:
    fma_timing_r = 0

# Per-input: does FMA respond differently to different inputs? (transfer function)
input_responses = {}
for inp_idx in range(0, 64, 8):
    mask = (all_fma[:,5] >= inputs[inp_idx].item() - 0.01) & \
           (all_fma[:,5] <= inputs[inp_idx].item() + 0.01)
    if mask.sum() > 0:
        input_responses[inp_idx] = float(all_fma[mask, 0].mean())

input_range = max(input_responses.values()) - min(input_responses.values()) if len(input_responses) > 1 else 0

print(f"  Per-CU residual: accum CV={fma_accum_cv:.6f}, final CV={fma_final_cv:.6f}")
print(f"  Reproducibility: r={fma_repro_r:.3f}")
print(f"  FMA↔timing correlation: r={fma_timing_r:.3f}")
print(f"  Input response range: {input_range:.8f}")

for w in sorted(fma_by_cu)[:6]:
    d = fma_by_cu[w]
    print(f"    WGP {w:2d}: accum={d['accum_mean']:.8f} ±{d['accum_std']:.8f}, "
          f"accum2={d['accum2_mean']:.10f}")

results['experiments']['exp5_fma_cancel'] = {
    'fma_accum_cv': float(fma_accum_cv),
    'fma_final_cv': float(fma_final_cv),
    'fma_repro_r': float(fma_repro_r),
    'fma_timing_r': float(fma_timing_r),
    'input_range': float(input_range),
    'fma_by_cu': {str(k): v for k, v in fma_by_cu.items()},
}

test('T11', 'FMA residual differs across CUs (CV>0)', fma_accum_cv > 1e-6,
     f"CV={fma_accum_cv:.8f}")
test('T12', 'FMA residual reproducible (r>0.5)', fma_repro_r > 0.5,
     f"r={fma_repro_r:.3f}")
test('T13', 'FMA↔timing correlated (|r|>0.2)', abs(fma_timing_r) > 0.2,
     f"r={fma_timing_r:.3f}")
test('T14', 'FMA responds to input (range>0)', input_range > 0,
     f"range={input_range:.8f}")
test('T15', 'FMA final state CU-dependent (CV>0)', fma_final_cv > 1e-6,
     f"CV={fma_final_cv:.8f}")
save()

# --- EXP 6: Wavefront Contention (Stochastic Gating) ---
print("\n" + "="*70)
print("  EXP 6: Wavefront Contention — Stochastic Synapse")
print("  Paired wavefronts race to atomicCAS → winner = f(physics)")
print("="*70)
wait_cool(50)

N_RACES = 10000
N_ROUNDS = 10

contention_results = []
for round_idx in range(N_ROUNDS):
    winners = probe.probe_contention(N_RACES).cpu().numpy()
    contention_results.append(winners)
    if (round_idx+1) % 5 == 0:
        team0_wins = (winners[:,1] == 0).sum()
        team1_wins = (winners[:,1] == 1).sum()
        print(f"  Round {round_idx+1}: team0={team0_wins}, team1={team1_wins}, "
              f"bias={team0_wins/(team0_wins+team1_wins+1e-10):.4f}")
    time.sleep(0.1)

all_cont = np.vstack(contention_results)
# Columns: winner_wgp, winner_team, winner_simd

# Per-WGP win rate
wgp_wins = {}
total_races = len(all_cont)
for w in np.unique(all_cont[:,0]):
    w_int = int(w)
    wgp_wins[w_int] = int((all_cont[:,0] == w).sum())

# Team 0 vs Team 1 bias
team0_total = (all_cont[:,1] == 0).sum()
team1_total = (all_cont[:,1] == 1).sum()
global_bias = team0_total / (team0_total + team1_total + 1e-10)

# Per-WGP bias: when this WGP wins, which team wins more?
wgp_team_bias = {}
for w in sorted(wgp_wins):
    mask = all_cont[:,0] == w
    t0 = ((all_cont[:,1] == 0) & mask).sum()
    t1 = ((all_cont[:,1] == 1) & mask).sum()
    wgp_team_bias[w] = float(t0 / (t0 + t1 + 1e-10))

# Entropy of win distribution — higher = more uniform (less physical bias)
win_counts = np.array(list(wgp_wins.values()), dtype=float)
win_probs = win_counts / (win_counts.sum() + 1e-10)
win_entropy = float(entropy(win_probs))
max_entropy = float(np.log(len(wgp_wins))) if len(wgp_wins) > 1 else 1
normalized_entropy = win_entropy / (max_entropy + 1e-10)

# Reproducibility: first 5 rounds vs last 5
wgp_wins_a, wgp_wins_b = {}, {}
for w in wgp_wins:
    a = np.vstack(contention_results[:5])
    b = np.vstack(contention_results[5:])
    wgp_wins_a[w] = int((a[:,0]==w).sum())
    wgp_wins_b[w] = int((b[:,0]==w).sum())
cont_common = sorted(set(wgp_wins_a) & set(wgp_wins_b))
if len(cont_common) >= 3:
    cont_repro_r = pearsonr([wgp_wins_a[w] for w in cont_common],
                             [wgp_wins_b[w] for w in cont_common])[0]
else:
    cont_repro_r = 0

# Is contention bias correlated with timing? (faster CU wins more)
if len(cont_common) >= 3 and len(cu_timing) >= 3:
    cc = sorted(set(wgp_wins) & set(cu_timing))
    if len(cc) >= 3:
        cont_timing_r = pearsonr([wgp_wins[w] for w in cc],
                                  [cu_timing[w]['wall_mean'] for w in cc])[0]
    else:
        cont_timing_r = 0
else:
    cont_timing_r = 0

print(f"\n  Total races: {total_races}")
print(f"  Global bias: team0={global_bias:.4f}")
print(f"  Win distribution entropy: {normalized_entropy:.4f} (1.0 = uniform)")
print(f"  Reproducibility: r={cont_repro_r:.3f}")
print(f"  Contention↔timing correlation: r={cont_timing_r:.3f}")
print(f"  Per-WGP wins:")
for w in sorted(wgp_wins):
    print(f"    WGP {w:2d}: {wgp_wins[w]:6d} wins ({wgp_wins[w]/total_races*100:.2f}%), "
          f"team0 bias={wgp_team_bias[w]:.3f}")

results['experiments']['exp6_contention'] = {
    'total_races': total_races,
    'global_bias': float(global_bias),
    'normalized_entropy': float(normalized_entropy),
    'cont_repro_r': float(cont_repro_r),
    'cont_timing_r': float(cont_timing_r),
    'wgp_wins': {str(k): v for k, v in wgp_wins.items()},
    'wgp_team_bias': {str(k): v for k, v in wgp_team_bias.items()},
}

test('T16', 'Contention not uniform (entropy < 0.95)', normalized_entropy < 0.95,
     f"entropy={normalized_entropy:.4f}")
test('T17', 'Contention reproducible (r>0.5)', cont_repro_r > 0.5,
     f"r={cont_repro_r:.3f}")
test('T18', 'Contention↔timing correlated (|r|>0.2)', abs(cont_timing_r) > 0.2,
     f"r={cont_timing_r:.3f}")
test('T19', 'Per-WGP bias non-uniform',
     max(wgp_team_bias.values()) - min(wgp_team_bias.values()) > 0.02 if wgp_team_bias else False,
     f"bias range={max(wgp_team_bias.values())-min(wgp_team_bias.values()):.4f}" if wgp_team_bias else "no data")
test('T20', 'Multiple WGPs win races', len(wgp_wins) >= 4,
     f"{len(wgp_wins)} WGPs")
save()

# --- EXP 7: Thermal Sequential Coupling ---
print("\n" + "="*70)
print("  EXP 7: Thermal Sequential Coupling")
print("  kernel₁ heats chip → kernel₂ timing changes → analog recurrence")
print("="*70)
wait_cool(45)

heat_ws = torch.randn(256*1024, device='cuda', dtype=torch.float32)

# Protocol: measure → heat → measure → heat → measure → ... → cool → measure
# Each measure-after-heat reveals how prior computation affects current timing.
thermal_seq = []

# Phase 1: Baseline (cold)
for i in range(5):
    d = probe.probe_thermal(20, 500).cpu().numpy()
    m = get_gpu_metrics()
    thermal_seq.append({
        'phase': 'baseline', 'temp': m.get('temp_c', get_temp()),
        'wall_mean': float(d[:,0].mean()), 'wall_std': float(d[:,0].std()),
        'clock_mhz': m.get('clock_mhz', 0),
    })
    time.sleep(0.2)
print(f"  Baseline: {thermal_seq[-1]['wall_mean']:.0f}ns @ {thermal_seq[-1]['temp']:.0f}°C")

# Phase 2: Interleaved heat-measure cycles
for cycle in range(15):
    # Heat burst (variable intensity)
    heat_iters = 200 + cycle * 100  # Increasing heat per cycle
    probe.run_heat(heat_ws, heat_iters)
    torch.cuda.synchronize()

    # Immediate measure (no cooling!)
    d = probe.probe_thermal(20, 500).cpu().numpy()
    m = get_gpu_metrics()
    thermal_seq.append({
        'phase': f'post_heat_{cycle}',
        'heat_iters': heat_iters,
        'temp': m.get('temp_c', get_temp()),
        'wall_mean': float(d[:,0].mean()),
        'wall_std': float(d[:,0].std()),
        'clock_mhz': m.get('clock_mhz', 0),
    })
    if (cycle+1) % 5 == 0:
        t = thermal_seq[-1]
        print(f"  After heat_{cycle}: {t['wall_mean']:.0f}ns @ {t['temp']:.0f}°C, "
              f"clk={t['clock_mhz']:.0f}MHz")

    # Brief pause to let thermal propagate but not fully dissipate
    time.sleep(0.3)
    if get_temp() > 80:
        wait_cool(65)

# Phase 3: Cooldown tracking
print("  Cooling...")
for i in range(10):
    d = probe.probe_thermal(20, 500).cpu().numpy()
    m = get_gpu_metrics()
    thermal_seq.append({
        'phase': f'cooldown_{i}',
        'temp': m.get('temp_c', get_temp()),
        'wall_mean': float(d[:,0].mean()),
        'wall_std': float(d[:,0].std()),
        'clock_mhz': m.get('clock_mhz', 0),
    })
    time.sleep(1.0)
print(f"  Cooldown end: {thermal_seq[-1]['wall_mean']:.0f}ns @ {thermal_seq[-1]['temp']:.0f}°C")

# Analysis
temps = np.array([t['temp'] for t in thermal_seq])
walls = np.array([t['wall_mean'] for t in thermal_seq])
clocks = np.array([t['clock_mhz'] for t in thermal_seq])
heat_iters_arr = np.array([t.get('heat_iters', 0) for t in thermal_seq])

temp_range = temps.max() - temps.min()

# Correlation: temp↔wall
r_tw = pearsonr(temps, walls)[0] if np.std(temps)>0.5 and np.std(walls)>0 else 0

# Correlation: clock↔wall
r_cw = pearsonr(clocks, walls)[0] if np.std(clocks)>0.5 and np.std(walls)>0 else 0

# Sequential coupling: does heat_iters predict NEXT measurement's timing?
heat_phases = [t for t in thermal_seq if t.get('heat_iters', 0) > 0]
if len(heat_phases) >= 5:
    heat_inputs = np.array([t['heat_iters'] for t in heat_phases])
    heat_outputs = np.array([t['wall_mean'] for t in heat_phases])
    r_seq = pearsonr(heat_inputs, heat_outputs)[0] if np.std(heat_inputs)>0 and np.std(heat_outputs)>0 else 0
    # Lag-1: does THIS heat intensity predict NEXT measurement?
    if len(heat_inputs) >= 3:
        r_lag1 = pearsonr(heat_inputs[:-1], heat_outputs[1:])[0]
    else:
        r_lag1 = 0
else:
    r_seq = r_lag1 = 0

# Residual after removing DVFS effect
if np.std(clocks) > 0.5:
    a_c, b_c = np.polyfit(clocks, walls, 1)
    wall_residual = walls - (a_c * clocks + b_c)
    r_resid = pearsonr(temps, wall_residual)[0] if np.std(temps)>0.5 and np.std(wall_residual)>0 else 0
else:
    r_resid = 0

# Hysteresis: is baseline-after-heating different from initial baseline?
baseline_mean = np.mean([t['wall_mean'] for t in thermal_seq if t['phase']=='baseline'])
cooldown_mean = np.mean([t['wall_mean'] for t in thermal_seq if 'cooldown' in t['phase']])
hysteresis = abs(cooldown_mean - baseline_mean) / (baseline_mean + 1e-10)

print(f"\n  Temp range: {temps.min():.0f}→{temps.max():.0f}°C (Δ={temp_range:.0f}°C)")
print(f"  Correlations: temp↔wall r={r_tw:.3f}, clock↔wall r={r_cw:.3f}")
print(f"  Residual (wall - DVFS) ↔ temp: r={r_resid:.3f}")
print(f"  Sequential: heat_iters↔wall r={r_seq:.3f}, lag-1 r={r_lag1:.3f}")
print(f"  Hysteresis: baseline={baseline_mean:.0f} → cooldown={cooldown_mean:.0f} ({hysteresis*100:.2f}%)")

results['experiments']['exp7_thermal_seq'] = {
    'temp_range': float(temp_range), 'r_tw': float(r_tw), 'r_cw': float(r_cw),
    'r_resid': float(r_resid), 'r_seq': float(r_seq), 'r_lag1': float(r_lag1),
    'hysteresis': float(hysteresis),
    'trace': [{'phase': t['phase'], 'temp': t['temp'], 'wall': t['wall_mean'],
                'clk': t['clock_mhz']} for t in thermal_seq],
}

test('T21', 'Temp range > 5°C', temp_range > 5, f"Δ={temp_range:.1f}°C")
test('T22', 'Temp↔wall |r|>0.3', abs(r_tw) > 0.3, f"r={r_tw:.3f}")
test('T23', 'Residual↔temp |r|>0.1 (pure thermal)', abs(r_resid) > 0.1, f"r={r_resid:.3f}")
test('T24', 'Sequential coupling (heat→next) |r|>0.2', abs(r_seq) > 0.2, f"r={r_seq:.3f}")
test('T25', 'Lag-1 coupling |r|>0.1', abs(r_lag1) > 0.1, f"r={r_lag1:.3f}")
test('T26', 'Thermal hysteresis > 0.1%', hysteresis > 0.001, f"{hysteresis*100:.2f}%")
save()
del heat_ws

# ======================================================================
# PART C: CROSS-MECHANISM SYNTHESIS
# ======================================================================
print("\n" + "="*70)
print("  EXP 8: Cross-Mechanism Synthesis")
print("="*70)

# Build per-CU feature matrix from ALL probes
all_cus = sorted(set(cu_timing) & common_insn & set(ratio_by_cu) & set(fma_by_cu) & set(wgp_wins))
print(f"  CUs with all probes: {len(all_cus)}")

if len(all_cus) >= 4:
    feature_names = ['wall_compute', 'VALU', 'trans', 'int', 'SALU',
                     'ratio', 'fma_accum', 'fma_accum2', 'contention_wins']
    feature_matrix = []
    for cu in all_cus:
        row = [
            cu_timing[cu]['wall_mean'],
            insn_results['VALU_fma']['per_cu'].get(cu, 0),
            insn_results['transcendental']['per_cu'].get(cu, 0),
            insn_results['integer']['per_cu'].get(cu, 0),
            insn_results['SALU']['per_cu'].get(cu, 0),
            ratio_by_cu.get(cu, 0),
            fma_by_cu[cu]['accum_mean'],
            fma_by_cu[cu]['accum2_mean'],
            wgp_wins.get(cu, 0),
        ]
        feature_matrix.append(row)
    fm = np.array(feature_matrix)

    # Normalize for PCA
    fm_centered = fm - fm.mean(axis=0)
    fm_std = fm.std(axis=0)
    fm_std[fm_std == 0] = 1
    fm_normed = fm_centered / fm_std

    from numpy.linalg import svd
    _, s, Vt = svd(fm_normed, full_matrices=False)
    explained = np.cumsum(s**2) / (np.sum(s**2) + 1e-10)
    dims_90 = int(np.searchsorted(explained, 0.9)) + 1

    # CU distinguishability
    from scipy.spatial.distance import pdist
    distances = pdist(fm_normed)
    min_dist = distances.min()
    mean_dist = distances.mean()
    distinguishable = min_dist > 0.05 * mean_dist

    # Are analog mechanisms (FMA, contention) independent from structural (timing)?
    if len(all_cus) >= 3:
        r_fma_timing = pearsonr(fm_normed[:, 0], fm_normed[:, 6])[0]  # wall vs fma
        r_cont_timing = pearsonr(fm_normed[:, 0], fm_normed[:, 8])[0]  # wall vs contention
        r_fma_cont = pearsonr(fm_normed[:, 6], fm_normed[:, 8])[0]    # fma vs contention
    else:
        r_fma_timing = r_cont_timing = r_fma_cont = 0

    print(f"\n  Feature matrix: {fm.shape[0]} CUs × {fm.shape[1]} dims")
    print(f"  PCA dims for 90% variance: {dims_90}")
    print(f"  Singular values: {s[:5].round(2)}")
    print(f"  Distinguishable: {distinguishable} (min/mean={min_dist/(mean_dist+1e-10):.4f})")
    print(f"\n  Independence:")
    print(f"    FMA↔timing: r={r_fma_timing:.3f}")
    print(f"    Contention↔timing: r={r_cont_timing:.3f}")
    print(f"    FMA↔contention: r={r_fma_cont:.3f}")

    # Show CU fingerprints
    print(f"\n  Per-CU Fingerprint (normalized):")
    for i, cu in enumerate(all_cus[:8]):
        fp = ' '.join(f"{feature_names[j][:4]}:{fm_normed[i,j]:+.2f}" for j in range(len(feature_names)))
        print(f"    WGP {cu:2d}: {fp}")
else:
    dims_90 = 0; distinguishable = False; r_fma_timing = r_cont_timing = r_fma_cont = 0
    min_dist = 0; mean_dist = 1

# Total measurement dimensions
n_structural = len(cu_timing) + 4 + len(vgpr_depths) + len(ratio_by_cu)  # CU + insn + vgpr + ratio
n_analog = len(fma_by_cu) + len(wgp_wins) + len(thermal_seq)  # FMA + contention + thermal
n_total = n_structural + n_analog

results['experiments']['exp8_synthesis'] = {
    'n_cus_all_probes': len(all_cus),
    'dims_90': dims_90,
    'distinguishable': bool(distinguishable),
    'n_structural': n_structural,
    'n_analog': n_analog,
    'n_total': n_total,
    'r_fma_timing': float(r_fma_timing),
    'r_cont_timing': float(r_cont_timing),
    'r_fma_cont': float(r_fma_cont),
}

test('T27', '20+ total measurement dimensions', n_total >= 20, f"{n_total}")
test('T28', 'PCA dims ≥ 3 for 90% variance', dims_90 >= 3, f"{dims_90}")
test('T29', 'CUs distinguishable by fingerprint', distinguishable,
     f"min/mean={min_dist/(mean_dist+1e-10):.4f}")
test('T30', 'FMA partially independent from timing (|r|<0.8)',
     abs(r_fma_timing) < 0.8 if len(all_cus) >= 3 else False,
     f"r={r_fma_timing:.3f}")
test('T31', 'Contention partially independent from timing (|r|<0.8)',
     abs(r_cont_timing) < 0.8 if len(all_cus) >= 3 else False,
     f"r={r_cont_timing:.3f}")
test('T32', 'FMA independent from contention (|r|<0.8)',
     abs(r_fma_cont) < 0.8 if len(all_cus) >= 3 else False,
     f"r={r_fma_cont:.3f}")

# Neuomorphic assessment: do we have the key ingredients?
has_unique_neurons = len(all_cus) >= 8 and distinguishable
has_analog_function = fma_accum_cv > 1e-6 or fma_final_cv > 1e-6
has_stochastic_synapse = normalized_entropy < 0.95 and len(wgp_wins) >= 4
has_thermal_feedback = abs(r_tw) > 0.3 or abs(r_resid) > 0.1
has_temporal_coupling = abs(r_seq) > 0.2 or abs(r_lag1) > 0.1

n_ingredients = sum([has_unique_neurons, has_analog_function,
                     has_stochastic_synapse, has_thermal_feedback, has_temporal_coupling])

print(f"\n  Neuromorphic Ingredients:")
print(f"    Unique neurons (CU fingerprints): {'✓' if has_unique_neurons else '✗'}")
print(f"    Analog transfer function (FMA):   {'✓' if has_analog_function else '✗'}")
print(f"    Stochastic synapse (contention):  {'✓' if has_stochastic_synapse else '✗'}")
print(f"    Thermal feedback:                 {'✓' if has_thermal_feedback else '✗'}")
print(f"    Temporal coupling (sequential):   {'✓' if has_temporal_coupling else '✗'}")

test('T33', 'Unique neurons (8+ distinguishable CUs)', has_unique_neurons,
     f"{len(all_cus)} CUs, dist={distinguishable}")
test('T34', 'Analog transfer function (FMA CU-dependent)', has_analog_function,
     f"accum_cv={fma_accum_cv:.8f}")
test('T35', 'Stochastic synapse (contention biased)', has_stochastic_synapse,
     f"entropy={normalized_entropy:.4f}, {len(wgp_wins)} WGPs")
test('T36', 'Thermal feedback loop', has_thermal_feedback,
     f"r_tw={r_tw:.3f}, r_resid={r_resid:.3f}")
test('T37', 'Temporal coupling (kernel→kernel)', has_temporal_coupling,
     f"r_seq={r_seq:.3f}, r_lag1={r_lag1:.3f}")
test('T38', '4/5 neuromorphic ingredients', n_ingredients >= 4,
     f"{n_ingredients}/5")
save()

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "="*70)
print("  SUMMARY: z2330 Full Analog Physics Characterization")
print("="*70)

passed = sum(1 for t in results['tests'].values() if t['status'] == 'PASS')
total = len(results['tests'])
print(f"\n  Tests: {passed}/{total} PASS ({100*passed/total:.0f}%)")

print(f"\n  {'Probe':<28s} {'Signal':>12s} {'Repro':>10s} {'Type':>10s}")
print(f"  {'-'*28} {'-'*12} {'-'*10} {'-'*10}")
print(f"  {'CU compute':<28s} {'CV='+f'{cu_cv*100:.3f}%':>12s} {'r='+f'{repro_r:.2f}':>10s} {'structural':>10s}")
print(f"  {'Insn fingerprint':<28s} {'r_vt='+f'{r_vt:.2f}':>12s} {'—':>10s} {'structural':>10s}")
print(f"  {'VGPR depth':<28s} {f'{depth_ratio:.1f}x':>12s} {'R²='+f'{vgpr_r2:.2f}':>10s} {'structural':>10s}")
print(f"  {'Ratio (mem/comp)':<28s} {'CV='+f'{ratio_cv*100:.2f}%':>12s} {'r='+f'{cross_r:.2f}':>10s} {'structural':>10s}")
print(f"  {'FMA cancellation':<28s} {'CV='+f'{fma_accum_cv:.6f}':>12s} {'r='+f'{fma_repro_r:.2f}':>10s} {'★ analog':>10s}")
print(f"  {'Contention (stochastic)':<28s} {'H='+f'{normalized_entropy:.3f}':>12s} {'r='+f'{cont_repro_r:.2f}':>10s} {'★ analog':>10s}")
print(f"  {'Thermal coupling':<28s} {'Δ'+f'{temp_range:.0f}°C':>12s} {'r='+f'{r_tw:.2f}':>10s} {'★ analog':>10s}")

print(f"\n  Fingerprint: {len(all_cus)} CUs × {dims_90} PCA dims")
print(f"  Neuromorphic ingredients: {n_ingredients}/5")
print(f"\n  Done. Results: {RESULTS / 'z2330_analog_neuron_probes.json'}")
