#!/usr/bin/env python3
"""
z2333_cross_substrate.py — GPU Physics × FPGA Neuron Cross-Substrate Reservoir
===============================================================================
Combines z2332 GPU physics reservoir (atomic race, contention, stale VGPR,
FMA timing) with cached z2305 FPGA 128-neuron states for cross-substrate
reservoir computing.

Key insight: same input sequence (seed=42) → GPU and FPGA states are driven
by same signal → cross-substrate products capture JOINT dynamics.

Conditions (5):
  1) GPU_ONLY:  z2332 physics populations (atomic + timing + digital)
  2) FPGA_ONLY: z2305 cached 128-neuron membrane voltages + temporal features
  3) CROSS:     GPU + FPGA features concatenated (independent readout)
  4) FUSION:    GPU × FPGA cross-products (joint encoding)
  5) CONTROL:   Shuffled GPU states (destroys temporal structure)

Benchmarks: Wave4, XOR(τ=1,3,5), MC(d=1..20), NARMA-5

Tests (18):
  T1:  GPU_ONLY wave4 > 70% (reproduce z2332)
  T2:  FPGA_ONLY wave4 > 80% (reproduce z2305 L3)
  T3:  CROSS wave4 > GPU_ONLY (FPGA adds value)
  T4:  CROSS wave4 > FPGA_ONLY (GPU adds value)
  T5:  FUSION wave4 > FPGA_ONLY (joint > single substrate)
  T6:  GPU_ONLY MC > 3.0 (GPU physics has memory)
  T7:  FPGA_ONLY MC > 8.0 (128 neurons with temporal)
  T8:  CROSS MC > max(GPU, FPGA) (cross-substrate synergy)
  T9:  FUSION MC > FPGA_ONLY (joint encoding helps)
  T10: GPU_ONLY XOR1 > 80%
  T11: CROSS XOR1 > max(GPU, FPGA)
  T12: CROSS XOR5 > 70%
  T13: FUSION NARMA5 < FPGA_ONLY (better regression)
  T14: CONTROL MC < 1.0 (shuffled = no memory)
  T15: CROSS > CONTROL on ≥3/5 benchmarks
  T16: FUSION n_features > 500 (rich joint space)
  T17: GPU_ONLY processing speed > 100 steps/s
  T18: At least one condition achieves wave4 > 90%

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 ./venv/bin/python scripts/z2333_cross_substrate.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2333_cross_substrate.json'

# Match z2305 parameters
N_STEPS = 1500
WARMUP = 300
SEED = 42
SAMPLE_HZ = 50

TEMP_PAUSE = 70.0
TEMP_RESUME = 50.0


# =================================================================
# Thermal
# =================================================================
def get_max_temp():
    temps = []
    for path in ['/sys/class/thermal/thermal_zone0/temp',
                 '/sys/class/hwmon/hwmon7/temp1_input']:
        try:
            with open(path, 'r') as f:
                temps.append(float(f.read().strip()) / 1000.0)
        except Exception:
            pass
    return max(temps) if temps else 0.0


def wait_cool(label="", target=None):
    if target is None:
        target = TEMP_RESUME
    temp = get_max_temp()
    if temp <= target:
        return temp
    print(f"  [TEMP] {label} {temp:.0f}C -> {target:.0f}C...", end="", flush=True)
    t0 = time.time()
    while temp > target and (time.time() - t0) < 180:
        time.sleep(5)
        temp = get_max_temp()
        print(f" {temp:.0f}", end="", flush=True)
    print(f" OK ({time.time()-t0:.0f}s)")
    return temp


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


# =================================================================
# GPU Physics Reservoir — reuse z2332's compiled kernels + sampler
# =================================================================
# Import z2332's infrastructure
sys.path.insert(0, str(BASE / 'scripts'))

N_ACCUM = 64
N_WAVES = 32
BASE_ADDS = 50
BASE_FMA = 2000
BASE_RACE = 100


def compile_and_load_gpu():
    """Compile z2332-style GPU kernels via load_inline with proper pybind."""
    import torch
    from torch.utils.cpp_extension import load_inline

    RESERVOIR_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define HWREG(id, offset, size) ((id) | ((offset) << 6) | (((size)-1) << 11))
#define HW_REG_HW_ID1 23

__device__ void get_hw_id(int& wgp, int& simd, int& se_sa) {
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    wgp   = (hw >> 8) & 0xF;
    simd  = (hw >> 4) & 0x3;
    se_sa = ((hw >> 13) & 0x7) * 2 + ((hw >> 12) & 0x1);
}

__global__ void kernel_atomic_reservoir(
    float* accumulators, float input_val, int base_adds, int n_accum, int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float u = input_val;
    int n_adds = base_adds + (int)(base_adds * 0.5f * u);
    if (n_adds < 1) n_adds = 1;
    int acc_idx = tid % n_accum;
    float base_val = 1.0e-4f * (1.0f + u * 0.5f);
    for (int i = 0; i < n_adds; i++) {
        float val = base_val * (float)((tid * 7 + i * 13 + (int)(u * 100.0f)) % 97 + 1);
        atomicAdd(&accumulators[acc_idx], val);
    }
}

__global__ void kernel_contention_reservoir(
    float* out, int* race_slots, float input_val, int base_iters, int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    float x = (float)(tid+1) * 0.001f;
    int work = (int)(20.0f * (1.0f + input_val));
    if (work < 1) work = 1;
    for (int i = 0; i < work; i++) x = fmaf(x, 0.999f, 0.001f);
    long long w0 = wall_clock64();
    int wins = 0;
    #pragma unroll 1
    for (int i = 0; i < base_iters; i++) {
        int slot = i % 64;
        int old = atomicCAS(&race_slots[slot], 0, tid + 1);
        if (old == 0) wins++;
        __threadfence();
        if (tid == 0) race_slots[slot] = 0;
        __threadfence();
    }
    long long w1 = wall_clock64();
    out[tid*4+0] = (float)(w1-w0)*10.0f;
    out[tid*4+1] = (float)wins;
    out[tid*4+2] = (float)wgp;
    out[tid*4+3] = (float)work;
    if (__builtin_expect(x == -1e30f, 0)) out[0] = x;
}

__global__ void kernel_stale_writer(float* workspace, float input_val, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = input_val * (float)(tid + 1);
    for (int i = 0; i < 50; i++) x = fmaf(x, 1.0001f, input_val * 0.001f);
    workspace[tid] = x;
}

__global__ void kernel_stale_reader(float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float s0, s1, s2, s3;
    asm volatile("v_mov_b32 %0, v40" : "=v"(s0));
    asm volatile("v_mov_b32 %0, v41" : "=v"(s1));
    asm volatile("v_mov_b32 %0, v42" : "=v"(s2));
    asm volatile("v_mov_b32 %0, v43" : "=v"(s3));
    __shared__ float lds[256];
    float lds_val = lds[threadIdx.x % 256];
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    out[tid*7+0] = s0; out[tid*7+1] = s1; out[tid*7+2] = s2; out[tid*7+3] = s3;
    out[tid*7+4] = lds_val; out[tid*7+5] = (float)wgp; out[tid*7+6] = (float)se_sa;
}

__global__ void kernel_fma_reservoir(float* out, float input_val, int base_iters, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float x = (float)(tid + 1) * 0.001f + input_val * 0.01f;
    for (int i = 0; i < 20; i++) x = fmaf(x, 0.999999f, 0.000001f);
    int n_iters = base_iters + (int)(base_iters * 0.3f * input_val);
    if (n_iters < 100) n_iters = 100;
    long long w0 = wall_clock64(); uint64_t c0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = fmaf(x, 1.0000001f, -x * 0.0000001f);
        x = fmaf(x, 0.9999999f,  x * 0.0000001f);
    }
    uint64_t c1 = clock64(); long long w1 = wall_clock64();
    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);
    out[tid*4+0] = (float)(w1-w0)*10.0f;
    out[tid*4+1] = (float)(c1-c0);
    out[tid*4+2] = (float)wgp;
    out[tid*4+3] = x;
    if (__builtin_expect(x == -1e30f, 0)) out[0] = x;
}

void run_atomic_reservoir(torch::Tensor accum, float input_val, int base_adds, int n_waves) {
    int n = n_waves * 32;
    kernel_atomic_reservoir<<<n_waves, 32>>>(accum.data_ptr<float>(), input_val, base_adds, accum.size(0), n);
}
torch::Tensor run_contention_reservoir(int n_waves, float input_val, int base_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto slots = torch::zeros({64}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_contention_reservoir<<<n_waves, 32>>>(out.data_ptr<float>(), slots.data_ptr<int>(), input_val, base_iters, n);
    return out.reshape({n, 4});
}
void run_stale_writer(torch::Tensor ws, float input_val) {
    int n = ws.size(0);
    kernel_stale_writer<<<(n+255)/256, 256>>>(ws.data_ptr<float>(), input_val, n);
}
torch::Tensor run_stale_reader(int n_waves) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*7}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_stale_reader<<<n_waves, 32>>>(out.data_ptr<float>(), n);
    return out.reshape({n, 7});
}
torch::Tensor run_fma_reservoir(int n_waves, float input_val, int base_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    kernel_fma_reservoir<<<n_waves, 32>>>(out.data_ptr<float>(), input_val, base_iters, n);
    return out.reshape({n, 4});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_atomic_reservoir", &run_atomic_reservoir);
    m.def("run_contention_reservoir", &run_contention_reservoir);
    m.def("run_stale_writer", &run_stale_writer);
    m.def("run_stale_reader", &run_stale_reader);
    m.def("run_fma_reservoir", &run_fma_reservoir);
}
'''

    print("Compiling HIP reservoir kernels (z2333)...", flush=True)
    rsv = load_inline(
        name='z2333_cross_substrate_v1',
        cpp_sources='',
        cuda_sources=RESERVOIR_SRC,
        extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
        verbose=False,
    )
    print("  Compiled and loaded", flush=True)
    return rsv


def sample_gpu_state(rsv, input_val, torch):
    """Sample one GPU physics state driven by input_val."""
    u = float(input_val)

    # Pop A: Atomic Race
    accum = torch.zeros(N_ACCUM, device='cuda')
    rsv.run_atomic_reservoir(accum, u, BASE_ADDS, N_WAVES)
    torch.cuda.synchronize()
    atomic_feats = accum.cpu().numpy()

    # Pop B: Contention
    ct = rsv.run_contention_reservoir(N_WAVES, u, BASE_RACE)
    torch.cuda.synchronize()
    ct_np = ct.cpu().numpy()
    cont_feats = np.zeros(32)
    wgps = np.unique(ct_np[:, 2].astype(int))
    for w in wgps:
        mask = ct_np[:, 2].astype(int) == w
        if w < 16:
            cont_feats[w] = np.mean(ct_np[mask, 0])
            cont_feats[16 + w] = np.mean(ct_np[mask, 1])

    # Pop C: Stale data
    ws = torch.randn(4096, device='cuda')
    rsv.run_stale_writer(ws, u)
    torch.cuda.synchronize()
    stale = rsv.run_stale_reader(N_WAVES)
    torch.cuda.synchronize()
    stale_np = stale.cpu().numpy()
    stale_feats = np.zeros(32)
    wgps_s = np.unique(stale_np[:, 5].astype(int))
    for w in wgps_s:
        mask = stale_np[:, 5].astype(int) == w
        vals = stale_np[mask, :4].flatten()
        if w < 16:
            stale_feats[w] = np.mean(vals)
            stale_feats[16 + w] = np.std(vals)

    # Pop D: FMA Timing
    ft = rsv.run_fma_reservoir(N_WAVES, u, BASE_FMA)
    torch.cuda.synchronize()
    ft_np = ft.cpu().numpy()
    fma_feats = np.zeros(16)
    wgps_f = np.unique(ft_np[:, 2].astype(int))
    for w in wgps_f:
        mask = ft_np[:, 2].astype(int) == w
        if w < 16:
            fma_feats[w] = np.mean(ft_np[mask, 0])

    return np.concatenate([atomic_feats, cont_feats, stale_feats, fma_feats])  # 64+32+32+16 = 144


def run_gpu_reservoir(u_raw, rsv, torch):
    """Run GPU physics reservoir on full input sequence."""
    total = len(u_raw)
    # Get feature size from first sample
    f0 = sample_gpu_state(rsv, float(u_raw[0]), torch)
    n_feat = len(f0)
    states = np.zeros((total, n_feat), dtype=np.float32)
    states[0] = f0

    t0 = time.time()
    for t in range(1, total):
        if t % 50 == 0:
            temp = get_max_temp()
            if temp > TEMP_PAUSE:
                wait_cool(f"step {t}/{total}", TEMP_RESUME)
        if t % 200 == 0:
            elapsed = time.time() - t0
            rate = t / elapsed
            print(f"    Step {t}/{total} ({rate:.0f} steps/s, temp={get_max_temp():.0f}°C)", flush=True)
        states[t] = sample_gpu_state(rsv, float(u_raw[t]), torch)
    elapsed = time.time() - t0
    print(f"  GPU reservoir: {total} steps in {elapsed:.1f}s ({total/elapsed:.0f} steps/s)", flush=True)
    return states


# =================================================================
# Feature builders
# =================================================================
def build_temporal_features(states, n_select=24, seed=42):
    """Temporal products for any state matrix."""
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, delta]

    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(n_ch, size=min(n_select, n_ch), replace=False))
    sq = states[:, qi]

    for tau in [1, 2, 3, 5, 8, 12, 20]:
        shifted = np.zeros_like(sq)
        shifted[tau:] = sq[:-tau]
        feats.append(sq * shifted)

    feats.append(np.square(sq))
    feats.append((sq > np.median(sq, axis=0)).astype(float))

    return np.hstack(feats)


def build_cross_features(gpu_states, fpga_states, n_select=16, seed=43):
    """Cross-substrate product features: GPU × FPGA."""
    rng = np.random.default_rng(seed)
    g_idx = np.sort(rng.choice(gpu_states.shape[1], size=min(n_select, gpu_states.shape[1]), replace=False))
    f_idx = np.sort(rng.choice(fpga_states.shape[1], size=min(n_select, fpga_states.shape[1]), replace=False))

    gs = gpu_states[:, g_idx]
    fs = fpga_states[:, f_idx]

    # Normalize both to similar scale
    gs_n = (gs - gs.mean(0)) / (gs.std(0) + 1e-8)
    fs_n = (fs - fs.mean(0)) / (fs.std(0) + 1e-8)

    cross_products = []
    # Direct cross-products: each GPU × each FPGA channel
    for i in range(min(n_select, 8)):
        for j in range(min(n_select, 8)):
            cross_products.append((gs_n[:, i] * fs_n[:, j]).reshape(-1, 1))

    # Temporal cross-products: GPU(t) × FPGA(t-τ) and vice versa
    for tau in [1, 2, 3, 5]:
        gs_shift = np.zeros_like(gs_n)
        gs_shift[tau:] = gs_n[:-tau]
        fs_shift = np.zeros_like(fs_n)
        fs_shift[tau:] = fs_n[:-tau]
        for i in range(min(n_select, 4)):
            cross_products.append((gs_n[:, i] * fs_shift[:, i]).reshape(-1, 1))
            cross_products.append((fs_n[:, i] * gs_shift[:, i]).reshape(-1, 1))

    return np.hstack(cross_products)


# =================================================================
# Benchmarks (same as z2305/z2332)
# =================================================================
def ridge_solve(X_tr, y_tr, X_te, y_te, task='regression'):
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best = -1e10 if task == 'regression' else 0.0
    for alpha in alphas:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
            pred = X_te @ w
            if task == 'regression':
                ss_res = np.sum((y_te - pred) ** 2)
                ss_tot = np.sum((y_te - y_te.mean()) ** 2)
                score = max(0, 1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            else:
                score = np.mean((pred > 0.5).astype(float) == y_te)
            if score > best:
                best = score
        except Exception:
            pass
    return max(best, 0.0)


def full_benchmark(X, u_raw, warmup, label=""):
    n = len(X)
    n_tr = int(0.7 * n)

    # Remove constant/near-constant features
    stds = X.std(axis=0)
    keep = stds > 1e-8
    X = X[:, keep]
    if X.shape[1] == 0:
        print(f"    {label}: ALL features constant!", flush=True)
        return {'wave4': 0.25, 'xor': {'tau1': 0.5, 'tau3': 0.5, 'tau5': 0.5},
                'mc_total': 0.0, 'narma5': 999.0, 'n_features': 0}

    # Wave4 classification
    quartiles = np.percentile(u_raw[warmup:warmup+n], [25, 50, 75])
    u_seg = u_raw[warmup:warmup+n]
    labels = np.zeros(n)
    labels[u_seg > quartiles[2]] = 3
    labels[(u_seg > quartiles[1]) & (u_seg <= quartiles[2])] = 2
    labels[(u_seg > quartiles[0]) & (u_seg <= quartiles[1])] = 1

    scores_m = np.zeros((n - n_tr, 4))
    for c in range(4):
        y = (labels == c).astype(float)
        for alpha in [1.0, 10.0, 100.0]:
            try:
                I_a = np.eye(X[:n_tr].shape[1])
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I_a, X[:n_tr].T @ y[:n_tr])
                scores_m[:, c] = X[n_tr:] @ w
                break
            except:
                pass
    wave4 = float(np.mean(np.argmax(scores_m, axis=1) == labels[n_tr:]))

    # MC
    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 21):
        target = u_raw[warmup-d:len(u_raw)-d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    # XOR
    xor = {}
    for tau in [1, 3, 5]:
        u_a = (u_raw[warmup:] > 0).astype(float)
        u_b = (u_raw[warmup-tau:len(u_raw)-tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor[f'tau{tau}'] = acc

    # NARMA-5
    T = len(u_raw)
    u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
    y_nar = np.zeros(T)
    for t in range(5, T):
        y_nar[t] = 0.3*y_nar[t-1] + 0.05*y_nar[t-1]*np.sum(y_nar[t-5:t]) + 1.5*u_n[t-1]*u_n[t-5] + 0.1
        y_nar[t] = np.tanh(y_nar[t])
    target = y_nar[warmup:]
    nn = min(n, len(target))
    best_nrmse = 999.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        try:
            I_a = np.eye(X[:n_tr].shape[1])
            w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I_a, X[:n_tr].T @ target[:n_tr])
            pred = X[n_tr:nn] @ w
            gt = target[n_tr:nn]
            nrmse = np.sqrt(np.mean((gt-pred)**2)) / (np.std(gt)+1e-10)
            if nrmse < best_nrmse:
                best_nrmse = nrmse
        except Exception:
            pass

    print(f"      {label}: Wave4={wave4:.3f} XOR1={xor['tau1']:.3f} XOR5={xor['tau5']:.3f} MC={mc_total:.3f} NARMA5={best_nrmse:.3f} ({X.shape[1]} feat)", flush=True)
    return {'wave4': wave4, 'xor': xor, 'mc_total': mc_total,
            'mc_per_delay': mc_per_d, 'narma5': best_nrmse, 'n_features': int(X.shape[1])}


# =================================================================
# Main
# =================================================================
def main():
    print("=" * 70)
    print("  z2333: GPU Physics × FPGA Neuron Cross-Substrate Reservoir")
    print("  Atomic Race + Contention + FMA Timing + Stale VGPR × 128 LIF Neurons")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Temp: {get_max_temp():.0f}°C")

    results = {'experiments': {}, 'tests': {}}

    # -----------------------------------------------------------
    # Load cached FPGA states
    # -----------------------------------------------------------
    fpga_file = RESULTS / 'z2305_fpga_states.npy'
    if not fpga_file.exists():
        print(f"  ERROR: {fpga_file} not found! Run z2305 first.")
        sys.exit(1)

    fpga_raw = np.load(fpga_file)  # (1800, 128)
    print(f"\n  FPGA states: {fpga_raw.shape} from z2305 cache")
    total_steps = fpga_raw.shape[0]  # 1800

    # Generate SAME input sequence as z2305
    rng = np.random.default_rng(SEED)
    u_raw = rng.uniform(-1, 1, total_steps)

    # -----------------------------------------------------------
    # Run GPU physics reservoir on same input
    # -----------------------------------------------------------
    import torch
    print(f"  Device: {torch.cuda.get_device_name(0)}")

    wait_cool("pre-GPU", TEMP_RESUME)
    rsv = compile_and_load_gpu()

    print(f"\n  Running GPU physics reservoir ({total_steps} steps)...")
    t0 = time.time()
    gpu_raw = run_gpu_reservoir(u_raw, rsv, torch)  # (1800, 144)
    gpu_time = time.time() - t0
    print(f"  GPU raw: {gpu_raw.shape}")

    # Save GPU states
    np.save(RESULTS / 'z2333_gpu_states.npy', gpu_raw)

    # -----------------------------------------------------------
    # Build feature sets for each condition
    # -----------------------------------------------------------
    # Use data after warmup for benchmarks
    fpga_states = fpga_raw[WARMUP:]   # (1500, 128)
    gpu_states = gpu_raw[WARMUP:]     # (1500, 192)

    print(f"\n  Building features (warmup={WARMUP}, eval={len(fpga_states)} steps)...")

    # 1) GPU_ONLY: GPU raw + temporal
    gpu_temp = build_temporal_features(gpu_states, n_select=24, seed=42)
    print(f"    GPU_ONLY: {gpu_temp.shape[1]} features")

    # 2) FPGA_ONLY: FPGA raw + temporal
    fpga_temp = build_temporal_features(fpga_states, n_select=24, seed=42)
    print(f"    FPGA_ONLY: {fpga_temp.shape[1]} features")

    # 3) CROSS: concatenate GPU + FPGA temporal features
    cross_feats = np.hstack([gpu_temp, fpga_temp])
    print(f"    CROSS: {cross_feats.shape[1]} features")

    # 4) FUSION: CROSS + cross-substrate products
    xprod = build_cross_features(gpu_states, fpga_states, n_select=16, seed=43)
    fusion_feats = np.hstack([gpu_temp, fpga_temp, xprod])
    print(f"    FUSION: {fusion_feats.shape[1]} features ({xprod.shape[1]} cross-products)")

    # 5) CONTROL: shuffle GPU states temporally
    gpu_shuffled = gpu_states.copy()
    rng_ctrl = np.random.default_rng(999)
    for c in range(gpu_shuffled.shape[1]):
        rng_ctrl.shuffle(gpu_shuffled[:, c])
    ctrl_temp = build_temporal_features(gpu_shuffled, n_select=24, seed=42)
    print(f"    CONTROL: {ctrl_temp.shape[1]} features (shuffled)")

    # -----------------------------------------------------------
    # Run benchmarks
    # -----------------------------------------------------------
    conditions = [
        ('GPU_ONLY', gpu_temp),
        ('FPGA_ONLY', fpga_temp),
        ('CROSS', cross_feats),
        ('FUSION', fusion_feats),
        ('CONTROL', ctrl_temp),
    ]

    for name, X in conditions:
        print(f"\n{'='*70}")
        print(f"  CONDITION: {name}")
        print(f"{'='*70}")
        wait_cool(name, TEMP_RESUME)
        bm = full_benchmark(X, u_raw, WARMUP, name)
        bm['time_s'] = gpu_time if name == 'GPU_ONLY' else 0.0
        results['experiments'][name] = bm
        save_results(results)

    # -----------------------------------------------------------
    # Tests
    # -----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  TESTS")
    print(f"{'='*70}")

    E = results['experiments']
    tests = {}

    def test(tid, name, cond, detail):
        p = "PASS" if cond else "FAIL"
        print(f"  [{p}] {tid}: {name} = {detail}", flush=True)
        tests[tid] = {'name': name, 'pass': bool(cond), 'detail': detail}

    test('T1', 'GPU_ONLY wave4 > 70%', E['GPU_ONLY']['wave4'] > 0.70,
         f"GPU={E['GPU_ONLY']['wave4']:.3f}")
    test('T2', 'FPGA_ONLY wave4 > 80%', E['FPGA_ONLY']['wave4'] > 0.80,
         f"FPGA={E['FPGA_ONLY']['wave4']:.3f}")
    test('T3', 'CROSS wave4 > GPU_ONLY', E['CROSS']['wave4'] > E['GPU_ONLY']['wave4'],
         f"CROSS={E['CROSS']['wave4']:.3f} vs GPU={E['GPU_ONLY']['wave4']:.3f}")
    test('T4', 'CROSS wave4 > FPGA_ONLY', E['CROSS']['wave4'] > E['FPGA_ONLY']['wave4'],
         f"CROSS={E['CROSS']['wave4']:.3f} vs FPGA={E['FPGA_ONLY']['wave4']:.3f}")
    test('T5', 'FUSION wave4 > FPGA_ONLY', E['FUSION']['wave4'] > E['FPGA_ONLY']['wave4'],
         f"FUSION={E['FUSION']['wave4']:.3f} vs FPGA={E['FPGA_ONLY']['wave4']:.3f}")

    test('T6', 'GPU_ONLY MC > 3.0', E['GPU_ONLY']['mc_total'] > 3.0,
         f"MC={E['GPU_ONLY']['mc_total']:.3f}")
    test('T7', 'FPGA_ONLY MC > 8.0', E['FPGA_ONLY']['mc_total'] > 8.0,
         f"MC={E['FPGA_ONLY']['mc_total']:.3f}")
    max_single_mc = max(E['GPU_ONLY']['mc_total'], E['FPGA_ONLY']['mc_total'])
    test('T8', 'CROSS MC > max(GPU,FPGA)', E['CROSS']['mc_total'] > max_single_mc,
         f"CROSS={E['CROSS']['mc_total']:.3f} vs max={max_single_mc:.3f}")
    test('T9', 'FUSION MC > FPGA_ONLY', E['FUSION']['mc_total'] > E['FPGA_ONLY']['mc_total'],
         f"FUSION={E['FUSION']['mc_total']:.3f} vs FPGA={E['FPGA_ONLY']['mc_total']:.3f}")

    test('T10', 'GPU_ONLY XOR1 > 80%', E['GPU_ONLY']['xor']['tau1'] > 0.80,
         f"XOR1={E['GPU_ONLY']['xor']['tau1']:.3f}")
    max_single_xor = max(E['GPU_ONLY']['xor']['tau1'], E['FPGA_ONLY']['xor']['tau1'])
    test('T11', 'CROSS XOR1 > max(GPU,FPGA)', E['CROSS']['xor']['tau1'] > max_single_xor,
         f"CROSS={E['CROSS']['xor']['tau1']:.3f} vs max={max_single_xor:.3f}")
    test('T12', 'CROSS XOR5 > 70%', E['CROSS']['xor']['tau5'] > 0.70,
         f"XOR5={E['CROSS']['xor']['tau5']:.3f}")

    test('T13', 'FUSION NARMA5 < FPGA_ONLY', E['FUSION']['narma5'] < E['FPGA_ONLY']['narma5'],
         f"FUSION={E['FUSION']['narma5']:.3f} vs FPGA={E['FPGA_ONLY']['narma5']:.3f}")

    test('T14', 'CONTROL MC < 1.0', E['CONTROL']['mc_total'] < 1.0,
         f"MC={E['CONTROL']['mc_total']:.3f}")

    ctrl_benchmarks = [
        E['CROSS']['wave4'] > E['CONTROL']['wave4'],
        E['CROSS']['xor']['tau1'] > E['CONTROL']['xor']['tau1'],
        E['CROSS']['xor']['tau5'] > E['CONTROL']['xor']['tau5'],
        E['CROSS']['mc_total'] > E['CONTROL']['mc_total'],
        E['CROSS']['narma5'] < E['CONTROL']['narma5'],
    ]
    test('T15', 'CROSS > CONTROL on ≥3/5', sum(ctrl_benchmarks) >= 3,
         f"wins={sum(ctrl_benchmarks)}/5")

    test('T16', 'FUSION n_features > 500', E['FUSION']['n_features'] > 500,
         f"n={E['FUSION']['n_features']}")

    test('T17', 'GPU speed > 100 steps/s', total_steps/gpu_time > 100,
         f"{total_steps/gpu_time:.0f} steps/s")

    any_90 = any(E[c]['wave4'] > 0.90 for c in ['GPU_ONLY', 'FPGA_ONLY', 'CROSS', 'FUSION'])
    test('T18', 'At least one wave4 > 90%', any_90,
         f"best={max(E[c]['wave4'] for c in ['GPU_ONLY', 'FPGA_ONLY', 'CROSS', 'FUSION']):.3f}")

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    results['summary'] = {
        'total_pass': n_pass,
        'total_tests': len(tests),
        'pass_rate': n_pass / len(tests),
    }
    save_results(results)

    # -----------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SUMMARY: z2333 Cross-Substrate Reservoir")
    print(f"{'='*70}")
    print(f"\n  Tests: {n_pass}/{len(tests)} PASS ({100*n_pass/len(tests):.0f}%)")
    print(f"\n  {'Condition':<15} {'Wave4':>8} {'XOR-1':>8} {'XOR-5':>8} {'MC':>8} {'NARMA5':>8} {'Feat':>6}")
    print(f"  {'-'*12:<15} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*6:>6}")
    for name in ['GPU_ONLY', 'FPGA_ONLY', 'CROSS', 'FUSION', 'CONTROL']:
        e = E[name]
        print(f"  {name:<15} {e['wave4']:>8.3f} {e['xor']['tau1']:>8.3f} {e['xor']['tau5']:>8.3f} {e['mc_total']:>8.3f} {e['narma5']:>8.3f} {e['n_features']:>6}")

    print(f"\n  Done. Results: {SAVE_FILE}")


if __name__ == '__main__':
    main()
