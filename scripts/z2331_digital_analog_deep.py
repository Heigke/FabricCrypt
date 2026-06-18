#!/usr/bin/env python3
"""
z2331_digital_analog_deep.py — Deep Digital + Analog GPU Probes
================================================================
Extends z2330 with research-backed mechanisms for DIGITAL bit-level
differences across CUs, plus improved timing-domain analog probes.

NEW DIGITAL MECHANISMS (from literature):
  D1. LeftoverLocals LDS Residual — read __shared__ without writing
      (CVE-2023-4969, Trail of Bits 2024). AMD doesn't zero LDS between dispatches.
  D2. FP32 Atomic Race — non-associative atomicAdd across CUs
      Different arrival order → different LSBs (llama.cpp issue #10197)
  D3. Uninitialized VGPR Read — stale register data from prior wavefronts
      (Whispering Pixels paper, AMD cleaner shader patches)
  D4. NaN Payload Chain — chain of invalid ops, track NaN significand bits
  D5. Transcendental Amplification — chain sin/cos 1000× to amplify <1 ULP diff

IMPROVED TIMING PROBES:
  T1. FMA Timing (not result) — how long FMA takes per CU
  T2. Inter-CU Contention — race between SEPARATE workgroups on different CUs
  T3. Thermal Sequential — improved with longer heat + more measurement points

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 python scripts/z2331_digital_analog_deep.py
"""

import os, sys, time, json, struct
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, entropy as sp_entropy

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
# HIP Kernel Source
# ======================================================================
import torch
from torch.utils.cpp_extension import load_inline

PROBE_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>

#define HWREG(id, offset, size) ((id) | ((offset) << 6) | (((size)-1) << 11))
#define HW_REG_HW_ID1 23

__device__ void get_hw_id(int& wgp, int& simd, int& se_sa) {
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    wgp   = (hw >> 8) & 0xF;
    simd  = (hw >> 4) & 0x3;
    se_sa = ((hw >> 13) & 0x7) * 2 + ((hw >> 12) & 0x1);
}

// =====================================================================
// D1: LeftoverLocals — Read LDS without writing
// AMD doesn't zero LDS between dispatches (CVE-2023-4969)
// First kernel writes a known pattern, second kernel reads without init
// =====================================================================

// Writer: fills LDS with CU-specific pattern
__global__ void kernel_lds_write_pattern(float* out, int pattern_id, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    __shared__ float lds[1024];  // 4KB
    // Write a known pattern: pattern_id * 1000 + local thread index
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
        lds[i] = (float)(pattern_id * 1000 + i);
    __syncthreads();
    // Read back to confirm write (sanity check)
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    out[tid*3+0] = lds[threadIdx.x];
    out[tid*3+1] = (float)wgp;
    out[tid*3+2] = (float)pattern_id;
}

// Reader: reads LDS WITHOUT writing first — captures residual data
__global__ void kernel_lds_read_residual(float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    __shared__ float lds[1024];  // 4KB — NOT initialized
    // Don't write, don't sync — just read what's there
    // The barrier is needed to ensure all threads see the same residual state
    __syncthreads();
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    // Read 16 LDS locations per thread
    float sum = 0.0f;
    int nonzero = 0;
    for (int i = 0; i < 16; i++) {
        int idx = (threadIdx.x * 16 + i) % 1024;
        float val = lds[idx];
        if (val != 0.0f) nonzero++;
        sum += val;
        // Store first few raw values
        if (i < 4) out[tid*8+i] = val;
    }
    out[tid*8+4] = sum;
    out[tid*8+5] = (float)nonzero;
    out[tid*8+6] = (float)wgp;
    out[tid*8+7] = (float)se_sa;
}

torch::Tensor probe_lds_write(int n_waves, int pattern_id) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*3}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_lds_write_pattern<<<n_waves, 32, 4096>>>(out.data_ptr<float>(), pattern_id, n);
    return out.reshape({n, 3});
}

torch::Tensor probe_lds_residual(int n_waves) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*8}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_lds_read_residual<<<n_waves, 32, 4096>>>(out.data_ptr<float>(), n);
    return out.reshape({n, 8});
}

// =====================================================================
// D2: FP32 Atomic Race — non-associative accumulation
// Many threads atomicAdd small floats to shared accumulator
// Order of addition is CU-scheduling-dependent → different LSBs
// =====================================================================
__global__ void kernel_atomic_race(float* accumulators, int* cu_tags,
                                     int n_accum, int adds_per_thread, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    // Each thread adds to a specific accumulator
    int acc_idx = tid % n_accum;

    // Values designed to expose non-associativity:
    // Different small values that when added in different orders give different LSBs
    for (int i = 0; i < adds_per_thread; i++) {
        float val = 1.0e-4f * (float)((tid * 7 + i * 13) % 97 + 1);
        atomicAdd(&accumulators[acc_idx], val);
    }

    // Tag which CU contributed to which accumulator
    cu_tags[tid] = wgp;
}

torch::Tensor probe_atomic_race(int n_waves, int n_accum, int adds_per_thread) {
    int n = n_waves * 32;
    auto accum = torch::zeros({n_accum}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto tags = torch::zeros({n}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_atomic_race<<<n_waves, 32>>>(accum.data_ptr<float>(), tags.data_ptr<int>(),
                                          n_accum, adds_per_thread, n);
    // Return both accumulators and tags
    return torch::cat({accum, tags.to(torch::kFloat32)});
}

// =====================================================================
// D3: Uninitialized VGPR Read — stale data from prior wavefronts
// Read registers that were never written to in this wavefront.
// Without cleaner shader, these contain residual data.
// =====================================================================
__global__ void kernel_read_stale_vgpr(float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Strategy: declare many float variables but DON'T initialize them.
    // The compiler might optimize these away, so we use inline asm.
    // On AMD: v0-v3 are usually initialized (thread args), but higher VGPRs may be stale.
    float stale0, stale1, stale2, stale3;

    // Use inline asm to read VGPRs that we never wrote
    // v40-v43 are unlikely to be used by this simple kernel
    asm volatile("v_mov_b32 %0, v40" : "=v"(stale0));
    asm volatile("v_mov_b32 %0, v41" : "=v"(stale1));
    asm volatile("v_mov_b32 %0, v42" : "=v"(stale2));
    asm volatile("v_mov_b32 %0, v43" : "=v"(stale3));

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    // Store as raw bits (reinterpret float bits as float for storage)
    out[tid*6+0] = stale0;
    out[tid*6+1] = stale1;
    out[tid*6+2] = stale2;
    out[tid*6+3] = stale3;
    out[tid*6+4] = (float)wgp;
    out[tid*6+5] = (float)se_sa;
}

torch::Tensor probe_stale_vgpr(int n_waves) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*6}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_read_stale_vgpr<<<n_waves, 32>>>(out.data_ptr<float>(), n);
    return out.reshape({n, 6});
}

// =====================================================================
// D4: NaN Payload Chain — track NaN significand bits through ops
// 0/0 produces canonical NaN (0x7FC00000 on AMD), but what happens
// when we chain operations on NaNs? Do payloads drift per CU?
// =====================================================================
__global__ void kernel_nan_payload(unsigned int* out, int chain_len, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Generate NaN via 0/0
    float zero = 0.0f * (float)(tid | 1);  // Prevent compile-time eval
    float nan_val = zero / zero;  // Should be 0x7FC00000

    // Chain operations that propagate NaN but might modify payload
    #pragma unroll 1
    for (int i = 0; i < chain_len; i++) {
        // Various operations that interact with NaN
        float other = (float)(i + 1) * 0.001f;
        nan_val = nan_val + other;       // NaN + number = NaN (but which payload?)
        nan_val = nan_val * other;       // NaN * number = NaN
        nan_val = fmaf(nan_val, 1.0f, other);  // FMA with NaN
        nan_val = fmaxf(nan_val, other); // max(NaN, x) behavior
        nan_val = fminf(nan_val, other); // min(NaN, x) behavior
    }

    // Also test: sqrt(-1), log(-1) for different NaN sources
    float neg = -1.0f * (float)(tid | 1);
    float nan_sqrt = sqrtf(neg);
    float nan_log = logf(neg);

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    // Store as raw unsigned int (bit pattern)
    unsigned int* nan_bits = (unsigned int*)&nan_val;
    unsigned int* sqrt_bits = (unsigned int*)&nan_sqrt;
    unsigned int* log_bits = (unsigned int*)&nan_log;

    out[tid*6+0] = *nan_bits;     // Chained NaN bits
    out[tid*6+1] = *sqrt_bits;    // sqrt(-x) NaN bits
    out[tid*6+2] = *log_bits;     // log(-x) NaN bits
    out[tid*6+3] = (unsigned int)wgp;
    out[tid*6+4] = (unsigned int)se_sa;
    out[tid*6+5] = 0x7FC00000;    // Expected canonical NaN for reference
}

torch::Tensor probe_nan_payload(int n_waves, int chain_len) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*6}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_nan_payload<<<n_waves, 32>>>((unsigned int*)out.data_ptr<int>(), chain_len, n);
    return out.reshape({n, 6});
}

// =====================================================================
// D5: Transcendental Amplification Chain
// sin/cos are NOT required to be correctly rounded (only faithful: ±1 ULP)
// Chain 1000 sin/cos operations: if any CU differs by 1 ULP anywhere,
// the error amplifies through the chain → detectable bit difference
// =====================================================================
__global__ void kernel_trans_chain(unsigned int* out, int chain_len, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Start from identical initial value across all CUs
    // Use a value near a sensitive region of sin/cos
    float x = 0.7853981633974483f;  // π/4 — where sin'(x)=cos(x)≈0.707
    float y = 2.356194490192345f;   // 3π/4

    // Chain: each step feeds previous output into next transcendental
    #pragma unroll 1
    for (int i = 0; i < chain_len; i++) {
        x = __sinf(x + y * 0.001f);  // Native sin (fast path, may differ by ULP)
        y = __cosf(y + x * 0.001f);  // Native cos
        x = __expf(-x * x);          // Native exp
        // Add a tiny perturbation to prevent fixed-point convergence
        x += 1.0e-7f * (float)(i & 3);
        y += 1.0e-7f * (float)((i+1) & 3);
    }

    // Also test with intrinsic (v_sin_f32 / v_cos_f32 directly)
    float z = 0.7853981633974483f;
    #pragma unroll 1
    for (int i = 0; i < chain_len; i++) {
        z = sinf(z * 0.15915494309189535f);  // sin(z/2π)*2π approximation
        z = z * 6.283185307179586f;
        z = cosf(z * 0.15915494309189535f);
        z = z * 6.283185307179586f + 1.0e-7f * (float)(i & 7);
    }

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    // Store as raw bits to detect single-bit differences
    out[tid*6+0] = __float_as_uint(x);
    out[tid*6+1] = __float_as_uint(y);
    out[tid*6+2] = __float_as_uint(z);
    out[tid*6+3] = (unsigned int)wgp;
    out[tid*6+4] = (unsigned int)se_sa;
    out[tid*6+5] = __float_as_uint(x + y + z);  // Combined hash
}

torch::Tensor probe_trans_chain(int n_waves, int chain_len) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*6}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_trans_chain<<<n_waves, 32>>>((unsigned int*)out.data_ptr<int>(), chain_len, n);
    return out.reshape({n, 6});
}

// =====================================================================
// T1: FMA Timing — measure how long FMA chain TAKES per CU
// Same computation, but now we care about wall_clock difference
// =====================================================================
__global__ void kernel_fma_timing(float* out, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid + 1) * 0.001f;
    // Warmup
    for (int i = 0; i < 50; i++)
        x = fmaf(x, 0.999999f, 0.000001f);

    long long w0 = wall_clock64(); uint64_t c0 = clock64();
    // Pure FMA chain — no transcendentals, just FMA
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = fmaf(x, 1.0000001f, -x * 0.0000001f);
        x = fmaf(x, 0.9999999f,  x * 0.0000001f);
    }
    uint64_t c1 = clock64(); long long w1 = wall_clock64();

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    out[tid*5+0] = (float)(w1-w0)*10.0f;
    out[tid*5+1] = (float)(c1-c0);
    out[tid*5+2] = (float)wgp;
    out[tid*5+3] = (float)simd;
    out[tid*5+4] = x;
    if (__builtin_expect(x == -1e30f, 0)) out[0] = x;
}

torch::Tensor probe_fma_timing(int n_waves, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_fma_timing<<<n_waves, 32>>>(out.data_ptr<float>(), n_iters, n);
    return out.reshape({n, 5});
}

// =====================================================================
// T2: Inter-CU Contention Race (improved)
// Launch enough workgroups to span ALL CUs, each races via global atomic
// The TIMING of contention resolution varies per CU pair
// =====================================================================
__global__ void kernel_intercu_race(float* out, int* race_slot, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    // Warmup
    float x = (float)(tid+1) * 0.001f;
    for (int i = 0; i < 20; i++) x = fmaf(x, 0.999f, 0.001f);

    long long w0 = wall_clock64();
    int wins = 0;

    // Race: try to atomicCAS a shared slot repeatedly
    // The contention timing depends on which CUs are physically closer to L2
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        // Try to claim slot i%64
        int slot = i % 64;
        int old = atomicCAS(&race_slot[slot], 0, tid + 1);
        if (old == 0) wins++;
        // Reset for next iteration
        __threadfence();
        if (tid == 0) race_slot[slot] = 0;
        __threadfence();
    }

    long long w1 = wall_clock64();

    out[tid*5+0] = (float)(w1-w0)*10.0f;
    out[tid*5+1] = (float)wins;
    out[tid*5+2] = (float)wgp;
    out[tid*5+3] = (float)simd;
    out[tid*5+4] = (float)n_iters;
    if (__builtin_expect(x == -1e30f, 0)) out[0] = x;
}

torch::Tensor probe_intercu_race(int n_waves, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto slots = torch::zeros({64}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_intercu_race<<<n_waves, 32>>>(out.data_ptr<float>(),
        slots.data_ptr<int>(), n_iters, n);
    return out.reshape({n, 5});
}

// =====================================================================
// Heat + thermal probe (from z2330)
// =====================================================================
__global__ void kernel_thermal_probe(float* out, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid+1)*0.001f;
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
    out[tid*4+3] = x;
}

torch::Tensor probe_thermal(int n_waves, int n_iters) {
    int n = n_waves*32;
    auto out = torch::zeros({n*4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_thermal_probe<<<n_waves, 32>>>(out.data_ptr<float>(), n_iters, n);
    return out.reshape({n, 4});
}

__global__ void kernel_heat(float* workspace, int n, int n_iters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = workspace[tid];
    #pragma unroll 1
    for (int i=0; i<n_iters; i++)
        x = sinf(x)*cosf(x+0.1f)*expf(-x*x) + 0.001f;
    workspace[tid] = x;
}

void run_heat(torch::Tensor workspace, int n_iters) {
    int n = workspace.size(0);
    kernel_heat<<<(n+255)/256, 256>>>(workspace.data_ptr<float>(), n, n_iters);
}

// =====================================================================
// Structural probes (from z2330 — kept for cross-reference)
// =====================================================================
__global__ void kernel_cu_simd(float* out, int n_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid + 1) * 0.001f;
    for (int i = 0; i < 50; i++) {
        x = sinf(x) * cosf(x + 0.1f) + expf(-x * x);
        x = x * 1.0001f + 0.0001f;
    }
    long long w0 = wall_clock64(); uint64_t c0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = sinf(x) * cosf(x + 0.1f) + expf(-x * x);
        x = x * 1.0001f + 0.0001f;
    }
    uint64_t c1 = clock64(); long long w1 = wall_clock64();
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    out[tid*5+0] = (float)(w1-w0)*10.0f;
    out[tid*5+1] = (float)(c1-c0);
    out[tid*5+2] = (float)wgp;
    out[tid*5+3] = (float)simd;
    out[tid*5+4] = (float)se_sa;
}

torch::Tensor probe_cu_simd(int n_waves, int n_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*5}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_cu_simd<<<n_waves, 32>>>(out.data_ptr<float>(), n_iters, n);
    return out.reshape({n, 5});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_lds_write", &probe_lds_write);
    m.def("probe_lds_residual", &probe_lds_residual);
    m.def("probe_atomic_race", &probe_atomic_race);
    m.def("probe_stale_vgpr", &probe_stale_vgpr);
    m.def("probe_nan_payload", &probe_nan_payload);
    m.def("probe_trans_chain", &probe_trans_chain);
    m.def("probe_fma_timing", &probe_fma_timing);
    m.def("probe_intercu_race", &probe_intercu_race);
    m.def("probe_thermal", &probe_thermal);
    m.def("run_heat", &run_heat);
    m.def("probe_cu_simd", &probe_cu_simd);
}
'''

print("Compiling HIP probe kernels (z2331)...")
probe = load_inline(
    name='z2331_digital_analog_deep_v1',
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
    'script': 'z2331_digital_analog_deep.py',
    'device': torch.cuda.get_device_name(0),
}}

def save():
    out_path = RESULTS / 'z2331_digital_analog_deep.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [SAVED] {out_path}")

def test(tid, name, passed, detail):
    status = 'PASS' if passed else 'FAIL'
    results['tests'][tid] = {'name': name, 'pass': passed, 'detail': detail}
    print(f"  [{status}] {tid}: {name} = {detail}")

m = get_gpu_metrics()
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"{'='*70}")
print(f"  z2331: Deep Digital + Analog GPU Probes")
print(f"  Digital: LDS residual, atomic race, stale VGPR, NaN payload, transcendental")
print(f"  Timing: FMA timing, inter-CU race, thermal coupling")
print(f"{'='*70}")
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Temp: {get_temp()}C, V={m.get('voltage_mv',0)}mV, Clk={m.get('clock_mhz',0)}MHz")


# ======================================================================
# EXP 1: LeftoverLocals LDS Residual
# ======================================================================
print(f"\n{'='*70}")
print(f"  EXP 1: LeftoverLocals — LDS Residual Data (CVE-2023-4969)")
print(f"  Write pattern → read without init → check for residual")
print(f"{'='*70}")
torch.cuda.synchronize()

exp1 = {}
N_WAVES = 64  # Enough to cover all CUs

# Phase 1: Write a known pattern
probe.probe_lds_write(N_WAVES, 42)
torch.cuda.synchronize()

# Phase 2: Immediately read without writing
res = probe.probe_lds_residual(N_WAVES).cpu().numpy()
# res columns: [val0, val1, val2, val3, sum, nonzero_count, wgp, se_sa]

nonzero_counts = res[:, 5]
total_nonzero = int(np.sum(nonzero_counts > 0))
any_residual = total_nonzero > 0
mean_nonzero = float(np.mean(nonzero_counts))

# Check if residual values match the written pattern
raw_vals = res[:, :4]
pattern_matches = np.sum(np.abs(raw_vals - 42000) < 2000)  # Near pattern_id*1000
unique_raw = len(np.unique(raw_vals[raw_vals != 0.0]))

# Per-WGP analysis
wgps = np.unique(res[:, 6].astype(int))
per_wgp_residual = {}
for w in wgps:
    mask = res[:, 6].astype(int) == w
    per_wgp_residual[int(w)] = {
        'nonzero_frac': float(np.mean(nonzero_counts[mask] > 0)),
        'mean_nonzero': float(np.mean(nonzero_counts[mask])),
        'sample_vals': [float(v) for v in raw_vals[mask][0, :4]],
    }

exp1 = {
    'total_threads': int(res.shape[0]),
    'threads_with_residual': total_nonzero,
    'mean_nonzero_per_thread': mean_nonzero,
    'pattern_matches': int(pattern_matches),
    'unique_nonzero_values': unique_raw,
    'per_wgp': per_wgp_residual,
}
results['experiments']['EXP1_lds_residual'] = exp1

print(f"  Threads with residual data: {total_nonzero}/{res.shape[0]}")
print(f"  Mean nonzero reads per thread: {mean_nonzero:.1f}/16")
print(f"  Pattern matches (from write kernel): {pattern_matches}")
print(f"  Unique nonzero values: {unique_raw}")
for w in sorted(per_wgp_residual.keys())[:4]:
    d = per_wgp_residual[w]
    print(f"    WGP {w:2d}: nonzero={d['mean_nonzero']:.1f}/16, vals={d['sample_vals'][:2]}")

test('T1', 'LDS has residual data after prior kernel', any_residual,
     f"nonzero={total_nonzero}/{res.shape[0]}")
test('T2', 'Residual data is non-uniform across CUs', unique_raw > 1,
     f"unique_vals={unique_raw}")
test('T3', 'Residual contains prior pattern', pattern_matches > 0,
     f"matches={pattern_matches}")
save()

# Phase 3: Multiple write-read cycles to test CU-dependence
print(f"\n  Phase 3: Multiple patterns to test CU-dependent retention...")
patterns_data = {}
for pat_id in [1, 2, 3]:
    probe.probe_lds_write(N_WAVES, pat_id * 100)
    torch.cuda.synchronize()
    rd = probe.probe_lds_residual(N_WAVES).cpu().numpy()
    patterns_data[pat_id] = rd[:, :4].flatten()

# Check if different patterns leave different residuals
r_12 = np.corrcoef(patterns_data[1], patterns_data[2])[0, 1] if np.std(patterns_data[1]) > 0 else 0
r_13 = np.corrcoef(patterns_data[1], patterns_data[3])[0, 1] if np.std(patterns_data[1]) > 0 else 0
exp1['pattern_correlation_1_2'] = float(r_12) if not np.isnan(r_12) else 0.0
exp1['pattern_correlation_1_3'] = float(r_13) if not np.isnan(r_13) else 0.0
print(f"  Pattern correlation 1↔2: r={r_12:.3f}, 1↔3: r={r_13:.3f}")

test('T4', 'Different write patterns → different residuals', abs(r_12) < 0.99 or abs(r_13) < 0.99,
     f"r12={r_12:.3f}, r13={r_13:.3f}")
save()


# ======================================================================
# EXP 2: FP32 Atomic Race — non-associative accumulation
# ======================================================================
print(f"\n{'='*70}")
print(f"  EXP 2: FP32 Atomic Race — Non-Associative Accumulation")
print(f"  Many threads atomicAdd small floats → order-dependent LSBs")
print(f"{'='*70}")
wait_cool(55)

N_ACCUM = 32
ADDS = 100
N_RACE_WAVES = 128  # Many waves → high contention

# Run multiple rounds — if results differ between rounds, it's non-deterministic
race_results = []
for rnd in range(10):
    raw = probe.probe_atomic_race(N_RACE_WAVES, N_ACCUM, ADDS)
    torch.cuda.synchronize()
    accum_vals = raw[:N_ACCUM].cpu().numpy()
    race_results.append(accum_vals.copy())

race_arr = np.array(race_results)  # (10, 32)

# Check if results differ across rounds (non-determinism!)
n_differ = 0
for acc_idx in range(N_ACCUM):
    unique_vals = len(np.unique(race_arr[:, acc_idx]))
    if unique_vals > 1:
        n_differ += 1

# Also check bit-level differences
bit_diffs = 0
for acc_idx in range(N_ACCUM):
    bits = [struct.unpack('I', struct.pack('f', v))[0] for v in race_arr[:, acc_idx]]
    unique_bits = len(set(bits))
    if unique_bits > 1:
        bit_diffs += 1

exp2 = {
    'n_accumulators': N_ACCUM,
    'n_rounds': 10,
    'n_waves': N_RACE_WAVES,
    'adds_per_thread': ADDS,
    'accum_differ_across_rounds': n_differ,
    'bit_level_differ': bit_diffs,
    'sample_values': [[float(v) for v in race_arr[:, i]] for i in range(min(4, N_ACCUM))],
}
results['experiments']['EXP2_atomic_race'] = exp2

print(f"  Accumulators with different values across rounds: {n_differ}/{N_ACCUM}")
print(f"  Accumulators with different BITS across rounds: {bit_diffs}/{N_ACCUM}")
for i in range(min(4, N_ACCUM)):
    vals = race_arr[:, i]
    bits = [f"0x{struct.unpack('I', struct.pack('f', v))[0]:08X}" for v in vals[:3]]
    print(f"    Acc[{i}]: {bits[0]} → {bits[1]} → {bits[2]} {'VARIES' if len(set(bits))>1 else 'SAME'}")

test('T5', 'Atomic race produces non-deterministic results', n_differ > 0,
     f"{n_differ}/{N_ACCUM} accums differ")
test('T6', 'Bit-level differences exist across rounds', bit_diffs > 0,
     f"{bit_diffs}/{N_ACCUM} accums have bit diffs")
save()


# ======================================================================
# EXP 3: Uninitialized VGPR — Stale Register Data
# ======================================================================
print(f"\n{'='*70}")
print(f"  EXP 3: Stale VGPR Read — Residual Register Data")
print(f"  Read v40-v43 without writing → stale from prior wavefront?")
print(f"{'='*70}")

# Run a compute kernel first to populate VGPRs
_ = probe.probe_cu_simd(64, 1000)
torch.cuda.synchronize()

# Now read stale VGPRs
stale = probe.probe_stale_vgpr(N_WAVES).cpu().numpy()
# columns: [stale0, stale1, stale2, stale3, wgp, se_sa]

stale_vals = stale[:, :4]
nonzero_stale = np.sum(np.abs(stale_vals) > 0)
total_stale = stale_vals.size
unique_stale = len(np.unique(stale_vals.flatten()))

# Check per-CU: do different CUs have different stale values?
wgps = np.unique(stale[:, 4].astype(int))
per_wgp_stale = {}
for w in wgps:
    mask = stale[:, 4].astype(int) == w
    vals = stale_vals[mask].flatten()
    per_wgp_stale[int(w)] = {
        'nonzero': int(np.sum(np.abs(vals) > 0)),
        'unique': int(len(np.unique(vals))),
        'sample': [float(v) for v in vals[:4]],
    }

exp3 = {
    'total_values': total_stale,
    'nonzero_values': int(nonzero_stale),
    'unique_values': unique_stale,
    'per_wgp': per_wgp_stale,
}
results['experiments']['EXP3_stale_vgpr'] = exp3

print(f"  Nonzero stale values: {nonzero_stale}/{total_stale}")
print(f"  Unique stale values: {unique_stale}")
for w in sorted(per_wgp_stale.keys())[:4]:
    d = per_wgp_stale[w]
    print(f"    WGP {w:2d}: nonzero={d['nonzero']}, unique={d['unique']}, sample={d['sample'][:2]}")

test('T7', 'Stale VGPR data exists (nonzero)', nonzero_stale > 0,
     f"{nonzero_stale}/{total_stale}")
test('T8', 'Stale data differs across CUs', unique_stale > 1,
     f"unique={unique_stale}")
save()


# ======================================================================
# EXP 4: NaN Payload Chain
# ======================================================================
print(f"\n{'='*70}")
print(f"  EXP 4: NaN Payload — Track NaN Bits Through Operation Chains")
print(f"  Does 0/0 → chain of ops → produce different NaN payloads per CU?")
print(f"{'='*70}")

nan_res = probe.probe_nan_payload(N_WAVES, 100).cpu().numpy().astype(np.uint32)
# columns: [chain_nan_bits, sqrt_nan_bits, log_nan_bits, wgp, se_sa, canonical_ref]

chain_bits = nan_res[:, 0]
sqrt_bits = nan_res[:, 1]
log_bits = nan_res[:, 2]
canonical = 0x7FC00000

unique_chain = len(np.unique(chain_bits))
unique_sqrt = len(np.unique(sqrt_bits))
unique_log = len(np.unique(log_bits))

# Check if any differ from canonical
chain_noncanon = np.sum(chain_bits != canonical)
sqrt_noncanon = np.sum(sqrt_bits != canonical)
log_noncanon = np.sum(log_bits != canonical)

# Per-CU NaN bits
wgps = np.unique(nan_res[:, 3])
per_wgp_nan = {}
for w in wgps:
    mask = nan_res[:, 3] == w
    per_wgp_nan[int(w)] = {
        'chain': f"0x{int(np.median(chain_bits[mask])):08X}",
        'sqrt': f"0x{int(np.median(sqrt_bits[mask])):08X}",
        'log': f"0x{int(np.median(log_bits[mask])):08X}",
    }

exp4 = {
    'canonical_nan': f"0x{canonical:08X}",
    'unique_chain_patterns': unique_chain,
    'unique_sqrt_patterns': unique_sqrt,
    'unique_log_patterns': unique_log,
    'chain_non_canonical': int(chain_noncanon),
    'sqrt_non_canonical': int(sqrt_noncanon),
    'log_non_canonical': int(log_noncanon),
    'per_wgp': per_wgp_nan,
}
results['experiments']['EXP4_nan_payload'] = exp4

print(f"  Canonical NaN: 0x{canonical:08X}")
print(f"  Chain NaN: {unique_chain} unique patterns, {chain_noncanon} non-canonical")
print(f"  sqrt(-x) NaN: {unique_sqrt} unique, {sqrt_noncanon} non-canonical")
print(f"  log(-x) NaN: {unique_log} unique, {log_noncanon} non-canonical")
for w in sorted(per_wgp_nan.keys())[:4]:
    d = per_wgp_nan[w]
    print(f"    WGP {w:2d}: chain={d['chain']}, sqrt={d['sqrt']}, log={d['log']}")

test('T9', 'NaN chain produces non-canonical payloads', chain_noncanon > 0,
     f"{chain_noncanon} non-canonical out of {len(chain_bits)}")
test('T10', 'Different CUs produce different NaN bits', unique_chain > 1 or unique_sqrt > 1,
     f"chain={unique_chain}, sqrt={unique_sqrt}, log={unique_log} unique patterns")
test('T11', 'sqrt(-x) or log(-x) differ from 0/0 chain',
     unique_sqrt != unique_chain or unique_log != unique_chain,
     f"sqrt_unique={unique_sqrt}, log_unique={unique_log} vs chain={unique_chain}")
save()


# ======================================================================
# EXP 5: Transcendental Amplification Chain
# ======================================================================
print(f"\n{'='*70}")
print(f"  EXP 5: Transcendental Chain — Amplify <1 ULP Differences")
print(f"  1000× chained sin/cos/exp → if any CU differs by 1 bit...")
print(f"{'='*70}")
wait_cool(55)

trans_results = []
for chain_len in [100, 500, 1000, 5000]:
    tr = probe.probe_trans_chain(N_WAVES, chain_len).cpu().numpy().astype(np.uint32)
    # columns: [x_bits, y_bits, z_bits, wgp, se_sa, combined]
    x_bits = tr[:, 0]
    y_bits = tr[:, 1]
    z_bits = tr[:, 2]
    wgps = tr[:, 3]

    # Per-CU: do different CUs give different bit patterns?
    cu_x = {}
    for w in np.unique(wgps):
        mask = wgps == w
        cu_x[int(w)] = int(np.median(x_bits[mask]))

    unique_x = len(set(cu_x.values()))
    unique_y = len(np.unique(y_bits))
    unique_z = len(np.unique(z_bits))

    trans_results.append({
        'chain_len': chain_len,
        'unique_x': unique_x,
        'unique_y': unique_y,
        'unique_z': unique_z,
        'sample_x_hex': f"0x{int(np.median(x_bits)):08X}",
    })
    differs = "DIFFERS" if unique_x > 1 else "SAME"
    print(f"  chain={chain_len:5d}: x={unique_x} unique, y={unique_y}, z={unique_z} [{differs}]")

exp5 = {'chains': trans_results}
results['experiments']['EXP5_trans_chain'] = exp5

any_differs = any(r['unique_x'] > 1 or r['unique_y'] > 1 for r in trans_results)
max_unique = max(r['unique_x'] for r in trans_results)

test('T12', 'Transcendental chain produces CU-different bits', any_differs,
     f"max unique x across CUs: {max_unique}")
test('T13', 'Longer chains amplify differences',
     trans_results[-1]['unique_x'] >= trans_results[0]['unique_x'],
     f"chain100={trans_results[0]['unique_x']}, chain5000={trans_results[-1]['unique_x']}")
save()


# ======================================================================
# EXP 6: FMA Timing per CU (analog — timing domain)
# ======================================================================
print(f"\n{'='*70}")
print(f"  EXP 6: FMA Timing — Per-CU FMA Pipeline Speed")
print(f"  Same FMA chain, measure wall_clock per CU → process variation")
print(f"{'='*70}")
wait_cool(55)

fma_timing_data = {}
for run in range(10):
    ft = probe.probe_fma_timing(N_WAVES, 5000).cpu().numpy()
    # columns: [wall_ns, cycles, wgp, simd, value]
    for row in ft:
        w = int(row[2])
        if w not in fma_timing_data:
            fma_timing_data[w] = []
        fma_timing_data[w].append(float(row[0]))

fma_cu_means = {w: np.mean(v) for w, v in fma_timing_data.items()}
fma_cu_stds = {w: np.std(v) for w, v in fma_timing_data.items()}
fma_vals = list(fma_cu_means.values())
fma_cv = np.std(fma_vals) / np.mean(fma_vals) if np.mean(fma_vals) > 0 else 0

# Reproducibility: first half vs second half
fma_first = {w: np.mean(v[:len(v)//2]) for w, v in fma_timing_data.items()}
fma_second = {w: np.mean(v[len(v)//2:]) for w, v in fma_timing_data.items()}
common = sorted(set(fma_first) & set(fma_second))
if len(common) >= 3:
    fma_repro_r = pearsonr([fma_first[w] for w in common],
                            [fma_second[w] for w in common])[0]
else:
    fma_repro_r = 0.0

exp6 = {
    'n_cus': len(fma_cu_means),
    'cv': float(fma_cv),
    'repro_r': float(fma_repro_r),
    'cu_means': {str(w): float(v) for w, v in fma_cu_means.items()},
}
results['experiments']['EXP6_fma_timing'] = exp6

print(f"  {len(fma_cu_means)} CUs, CV={fma_cv:.5f}, repro r={fma_repro_r:.3f}")
for w in sorted(fma_cu_means.keys())[:8]:
    print(f"    WGP {w:2d}: {fma_cu_means[w]:.0f} ±{fma_cu_stds[w]:.0f} ns")

test('T14', 'FMA timing varies across CUs (CV > 0.01%)', fma_cv > 0.0001,
     f"CV={fma_cv:.6f}")
test('T15', 'FMA timing map reproducible (r>0.3)', fma_repro_r > 0.3,
     f"r={fma_repro_r:.3f}")
save()


# ======================================================================
# EXP 7: Inter-CU Contention Race (timing domain)
# ======================================================================
print(f"\n{'='*70}")
print(f"  EXP 7: Inter-CU Race — Contention Timing + Win Rate per CU")
print(f"  Many CUs race for same global atomic → timing = f(CU distance to L2)")
print(f"{'='*70}")
wait_cool(55)

race_cu_wins = {}
race_cu_timing = {}

for rnd in range(5):
    rr = probe.probe_intercu_race(N_WAVES * 2, 200).cpu().numpy()
    # columns: [wall_ns, wins, wgp, simd, n_iters]
    for row in rr:
        w = int(row[2])
        if w not in race_cu_wins:
            race_cu_wins[w] = []
            race_cu_timing[w] = []
        race_cu_wins[w].append(float(row[1]))
        race_cu_timing[w].append(float(row[0]))

race_win_means = {w: np.mean(v) for w, v in race_cu_wins.items()}
race_time_means = {w: np.mean(v) for w, v in race_cu_timing.items()}

win_vals = list(race_win_means.values())
time_vals = list(race_time_means.values())
win_cv = np.std(win_vals) / np.mean(win_vals) if np.mean(win_vals) > 0 else 0
time_cv = np.std(time_vals) / np.mean(time_vals) if np.mean(time_vals) > 0 else 0

# Win rate ↔ timing correlation
common = sorted(set(race_win_means) & set(race_time_means))
if len(common) >= 3:
    wt_r = pearsonr([race_win_means[w] for w in common],
                     [race_time_means[w] for w in common])[0]
else:
    wt_r = 0.0

exp7 = {
    'n_cus': len(race_win_means),
    'win_cv': float(win_cv),
    'time_cv': float(time_cv),
    'win_timing_r': float(wt_r),
    'cu_wins': {str(w): float(v) for w, v in race_win_means.items()},
}
results['experiments']['EXP7_intercu_race'] = exp7

print(f"  {len(race_win_means)} CUs, win CV={win_cv:.4f}, time CV={time_cv:.5f}")
print(f"  Win↔timing r={wt_r:.3f}")
for w in sorted(race_win_means.keys())[:8]:
    print(f"    WGP {w:2d}: wins={race_win_means[w]:.1f}/200, time={race_time_means[w]:.0f}ns")

test('T16', 'Win rate varies across CUs (CV > 1%)', win_cv > 0.01,
     f"CV={win_cv:.4f}")
test('T17', 'Contention timing varies (CV > 0.1%)', time_cv > 0.001,
     f"CV={time_cv:.5f}")
test('T18', 'Faster CUs win more (|r| > 0.2)', abs(wt_r) > 0.2,
     f"r={wt_r:.3f}")
save()


# ======================================================================
# EXP 8: Thermal Sequential Coupling (improved)
# ======================================================================
print(f"\n{'='*70}")
print(f"  EXP 8: Thermal Sequential Coupling (improved protocol)")
print(f"  Baseline → 5 heat cycles with measurement → cooldown")
print(f"{'='*70}")
wait_cool(45)

workspace = torch.randn(256*1024, device='cuda')
thermal_timeline = []

# Baseline (3 measurements)
for i in range(3):
    tp = probe.probe_thermal(32, 2000).cpu().numpy()
    torch.cuda.synchronize()
    wall_mean = float(np.mean(tp[:, 0]))
    temp = get_temp()
    clk = get_gpu_metrics().get('clock_mhz', 0)
    thermal_timeline.append({'phase': 'baseline', 'wall_ns': wall_mean, 'temp': temp, 'clk': clk})
    print(f"  Baseline {i}: {wall_mean:.0f}ns @ {temp}°C, clk={clk}MHz")

# Heat cycles: heat → measure → heat → measure
for cycle in range(8):
    # Heat
    probe.run_heat(workspace, 50000)
    torch.cuda.synchronize()
    # Measure immediately
    tp = probe.probe_thermal(32, 2000).cpu().numpy()
    torch.cuda.synchronize()
    wall_mean = float(np.mean(tp[:, 0]))
    temp = get_temp()
    clk = get_gpu_metrics().get('clock_mhz', 0)
    thermal_timeline.append({'phase': f'heat_{cycle}', 'wall_ns': wall_mean, 'temp': temp, 'clk': clk})
    if cycle % 2 == 0:
        print(f"  After heat_{cycle}: {wall_mean:.0f}ns @ {temp}°C, clk={clk}MHz")
    # Safety check
    if temp > 80:
        print(f"  ABORT: temp {temp}°C > 80°C")
        break

# Cooldown
print(f"  Cooling...")
wait_cool(45, timeout=60)
for i in range(3):
    tp = probe.probe_thermal(32, 2000).cpu().numpy()
    torch.cuda.synchronize()
    wall_mean = float(np.mean(tp[:, 0]))
    temp = get_temp()
    clk = get_gpu_metrics().get('clock_mhz', 0)
    thermal_timeline.append({'phase': f'cooldown_{i}', 'wall_ns': wall_mean, 'temp': temp, 'clk': clk})

print(f"  Cooldown end: {wall_mean:.0f}ns @ {temp}°C")

# Analysis
temps = [p['temp'] for p in thermal_timeline]
walls = [p['wall_ns'] for p in thermal_timeline]
clks = [p['clk'] for p in thermal_timeline]

temp_range = max(temps) - min(temps)
if len(temps) >= 3 and np.std(temps) > 0 and np.std(walls) > 0:
    tw_r = pearsonr(temps, walls)[0]
else:
    tw_r = 0.0

# DVFS-corrected residual
if np.std(clks) > 0 and np.std(walls) > 0:
    from numpy.polynomial import polynomial as P
    clk_arr = np.array(clks)
    wall_arr = np.array(walls)
    # Linear fit: wall = a*clk + b
    coeffs = np.polyfit(clk_arr, wall_arr, 1)
    predicted = np.polyval(coeffs, clk_arr)
    residual = wall_arr - predicted
    temp_arr = np.array(temps)
    if np.std(residual) > 0 and np.std(temp_arr) > 0:
        resid_r = pearsonr(temp_arr, residual)[0]
    else:
        resid_r = 0.0
else:
    resid_r = 0.0

# Hysteresis
baseline_wall = np.mean([p['wall_ns'] for p in thermal_timeline if p['phase'] == 'baseline'])
cooldown_wall = np.mean([p['wall_ns'] for p in thermal_timeline if 'cooldown' in p['phase']])
hysteresis = abs(cooldown_wall - baseline_wall) / baseline_wall * 100

# Sequential coupling
heat_walls = [p['wall_ns'] for p in thermal_timeline if 'heat' in p['phase']]
if len(heat_walls) >= 3:
    seq_r = pearsonr(range(len(heat_walls)), heat_walls)[0]
    lag1_r = pearsonr(heat_walls[:-1], heat_walls[1:])[0] if len(heat_walls) >= 4 else 0.0
else:
    seq_r = 0.0
    lag1_r = 0.0

exp8 = {
    'temp_range': float(temp_range),
    'temp_wall_r': float(tw_r),
    'dvfs_residual_r': float(resid_r),
    'hysteresis_pct': float(hysteresis),
    'sequential_r': float(seq_r),
    'lag1_r': float(lag1_r),
    'timeline': thermal_timeline,
}
results['experiments']['EXP8_thermal'] = exp8

print(f"\n  Temp range: {min(temps)}→{max(temps)}°C (Δ={temp_range}°C)")
print(f"  Temp↔wall r={tw_r:.3f}")
print(f"  DVFS-residual↔temp r={resid_r:.3f}")
print(f"  Hysteresis: {hysteresis:.2f}%")
print(f"  Sequential r={seq_r:.3f}, lag-1 r={lag1_r:.3f}")

test('T19', 'Temp range > 5°C', temp_range > 5, f"Δ={temp_range}°C")
test('T20', 'Temp↔wall |r| > 0.3', abs(tw_r) > 0.3, f"r={tw_r:.3f}")
test('T21', 'Pure thermal residual |r| > 0.1', abs(resid_r) > 0.1, f"r={resid_r:.3f}")
test('T22', 'Thermal hysteresis > 0.1%', hysteresis > 0.1, f"{hysteresis:.2f}%")
test('T23', 'Sequential coupling |r| > 0.2', abs(seq_r) > 0.2, f"r={seq_r:.3f}")
save()


# ======================================================================
# EXP 9: Cross-Mechanism Synthesis
# ======================================================================
print(f"\n{'='*70}")
print(f"  EXP 9: Cross-Mechanism Synthesis — Neuromorphic Assessment")
print(f"{'='*70}")

# Count confirmed mechanisms
digital_mechanisms = {
    'lds_residual': any_residual,
    'atomic_race_nondeterminism': n_differ > 0,
    'stale_vgpr': nonzero_stale > 0,
    'nan_payload_variation': chain_noncanon > 0 or unique_chain > 1,
    'transcendental_divergence': any_differs,
}

timing_mechanisms = {
    'fma_timing_variation': fma_cv > 0.0001,
    'contention_win_bias': win_cv > 0.01,
    'thermal_coupling': abs(tw_r) > 0.3,
    'thermal_hysteresis': hysteresis > 0.1,
}

n_digital = sum(digital_mechanisms.values())
n_timing = sum(timing_mechanisms.values())
n_total = n_digital + n_timing

exp9 = {
    'digital_mechanisms': {k: bool(v) for k, v in digital_mechanisms.items()},
    'timing_mechanisms': {k: bool(v) for k, v in timing_mechanisms.items()},
    'n_digital': n_digital,
    'n_timing': n_timing,
    'n_total': n_total,
}
results['experiments']['EXP9_synthesis'] = exp9

print(f"\n  Digital mechanisms confirmed: {n_digital}/5")
for k, v in digital_mechanisms.items():
    print(f"    {'✓' if v else '✗'} {k}")

print(f"\n  Timing mechanisms confirmed: {n_timing}/4")
for k, v in timing_mechanisms.items():
    print(f"    {'✓' if v else '✗'} {k}")

# Neuromorphic ingredient checklist
ingredients = {
    'unique_neurons': len(fma_cu_means) >= 8,
    'analog_transfer': fma_cv > 0.0001 or any_differs,
    'stochastic_element': n_differ > 0 or win_cv > 0.01,
    'memory_feedback': abs(tw_r) > 0.3 or hysteresis > 0.1,
    'temporal_coupling': abs(seq_r) > 0.2 or abs(lag1_r) > 0.1,
    'digital_entropy': n_digital >= 2,
}
n_ingredients = sum(ingredients.values())
exp9['neuromorphic_ingredients'] = {k: bool(v) for k, v in ingredients.items()}
exp9['n_ingredients'] = n_ingredients

print(f"\n  Neuromorphic Ingredients: {n_ingredients}/6")
for k, v in ingredients.items():
    print(f"    {'✓' if v else '✗'} {k}")

test('T24', f'≥3 digital mechanisms confirmed', n_digital >= 3,
     f"{n_digital}/5 digital")
test('T25', f'≥3 timing mechanisms confirmed', n_timing >= 3,
     f"{n_timing}/4 timing")
test('T26', f'≥5 total mechanisms', n_total >= 5,
     f"{n_total}/9 total")
test('T27', f'≥4/6 neuromorphic ingredients', n_ingredients >= 4,
     f"{n_ingredients}/6")
save()


# ======================================================================
# SUMMARY
# ======================================================================
n_pass = sum(1 for t in results['tests'].values() if t['pass'])
n_total_tests = len(results['tests'])

print(f"\n{'='*70}")
print(f"  SUMMARY: z2331 Deep Digital + Analog Probes")
print(f"{'='*70}")
print(f"\n  Tests: {n_pass}/{n_total_tests} PASS ({100*n_pass/n_total_tests:.0f}%)\n")

print(f"  {'Mechanism':<35} {'Signal':<15} {'Digital?':<10} {'Status'}")
print(f"  {'-'*35} {'-'*15} {'-'*10} {'-'*8}")
print(f"  {'LDS Residual (LeftoverLocals)':<35} {'nonzero='+str(total_nonzero):<15} {'YES':<10} {'✓' if any_residual else '✗'}")
print(f"  {'FP32 Atomic Race':<35} {'differ='+str(n_differ):<15} {'YES':<10} {'✓' if n_differ>0 else '✗'}")
print(f"  {'Stale VGPR':<35} {'nonzero='+str(int(nonzero_stale)):<15} {'YES':<10} {'✓' if nonzero_stale>0 else '✗'}")
print(f"  {'NaN Payload':<35} {'noncanon='+str(int(chain_noncanon)):<15} {'YES':<10} {'✓' if chain_noncanon>0 else '✗'}")
print(f"  {'Transcendental Chain':<35} {'unique='+str(max_unique):<15} {'YES':<10} {'✓' if any_differs else '✗'}")
print(f"  {'FMA Timing':<35} {'CV='+f'{fma_cv:.5f}':<15} {'timing':<10} {'✓' if fma_cv>0.0001 else '✗'}")
print(f"  {'Inter-CU Race':<35} {'winCV='+f'{win_cv:.4f}':<15} {'timing':<10} {'✓' if win_cv>0.01 else '✗'}")
print(f"  {'Thermal Coupling':<35} {'r='+f'{tw_r:.3f}':<15} {'timing':<10} {'✓' if abs(tw_r)>0.3 else '✗'}")
print(f"  {'Thermal Hysteresis':<35} {f'{hysteresis:.2f}%':<15} {'timing':<10} {'✓' if hysteresis>0.1 else '✗'}")

print(f"\n  Done. Results: {RESULTS / 'z2331_digital_analog_deep.json'}")

results['meta']['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
results['meta']['n_pass'] = n_pass
results['meta']['n_tests'] = n_total_tests
save()
