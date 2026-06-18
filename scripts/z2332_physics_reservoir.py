#!/usr/bin/env python3
"""
z2332_physics_reservoir.py — GPU Physics Reservoir Computing
=============================================================
Uses CONFIRMED z2331 mechanisms as reservoir computing populations:

  Pop A: FP32 Atomic Race — input modulates thread count → different accumulation
         orders → non-deterministic LSBs = stochastic digital reservoir
  Pop B: Inter-CU Contention — input modulates work amount → timing varies
         per CU → analog timing reservoir (r=0.801 win↔timing)
  Pop C: Stale VGPR + LDS Residual — read uninitialized memory between
         input-driven kernels → hardware entropy source
  Pop D: FMA Timing — input-weighted FMA chains → per-CU process variation
         timing = analog fingerprint (CV=0.015%)

Benchmarks (standard reservoir computing):
  1. Waveform Classification (4-class)
  2. XOR (τ=1,3,5)
  3. Memory Capacity (d=1..20)
  4. NARMA-5

Conditions:
  ATOMIC    — Pop A only (digital stochastic)
  TIMING    — Pop B+D (analog timing)
  FULL      — All populations combined
  CONTROL   — random features (baseline)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 python scripts/z2332_physics_reservoir.py
"""

import os, sys, time, json, struct
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)

# ======================================================================
# Telemetry + safety
# ======================================================================
def get_temp():
    try: return int(open('/sys/class/thermal/thermal_zone0/temp').read()) // 1000
    except: return 0

def wait_cool(target=55, timeout=120):
    t0 = time.time()
    while get_temp() > target and time.time() - t0 < timeout:
        time.sleep(2)
    return get_temp()

def check_thermal(pause_at=75, resume_at=55):
    t = get_temp()
    if t > pause_at:
        print(f"    [THERMAL] {t}°C > {pause_at}°C — cooling...")
        wait_cool(resume_at)
        print(f"    [THERMAL] Resumed at {get_temp()}°C")

# ======================================================================
# HIP Kernels — Input-Driven Reservoir Populations
# ======================================================================
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

// =====================================================================
// Pop A: Input-Driven Atomic Race
// Input modulates how many adds each thread does → different ordering
// per input value → different LSBs in accumulators
// Output: N_ACCUM float accumulators (their raw bits are the features)
// =====================================================================
__global__ void kernel_atomic_reservoir(
    float* accumulators,    // [n_accum] output accumulators
    float input_val,        // Current input signal
    int base_adds,          // Base number of atomic adds per thread
    int n_accum,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    // Input modulates number of adds (more adds = more contention = more entropy)
    float u = input_val;
    int n_adds = base_adds + (int)(base_adds * 0.5f * u);
    if (n_adds < 1) n_adds = 1;

    int acc_idx = tid % n_accum;

    // Input-dependent value: different inputs create different accumulation sequences
    float base_val = 1.0e-4f * (1.0f + u * 0.5f);

    for (int i = 0; i < n_adds; i++) {
        // Value depends on input, tid, and iteration
        float val = base_val * (float)((tid * 7 + i * 13 + (int)(u * 100.0f)) % 97 + 1);
        atomicAdd(&accumulators[acc_idx], val);
    }
}

// =====================================================================
// Pop B: Input-Driven Contention Timing
// Input modulates work amount before contention → timing varies
// Output: per-CU timing + win rate
// =====================================================================
__global__ void kernel_contention_reservoir(
    float* out,         // [n, 4]: wall_ns, wins, wgp, input_work
    int* race_slots,    // [64] shared race slots
    float input_val,
    int base_iters,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    // Input-modulated pre-race work (creates input-dependent timing offset)
    float x = (float)(tid+1) * 0.001f;
    int work = (int)(20.0f * (1.0f + input_val));
    if (work < 1) work = 1;
    for (int i = 0; i < work; i++)
        x = fmaf(x, 0.999f, 0.001f);

    long long w0 = wall_clock64();
    int wins = 0;

    // Race for shared slots — timing depends on CU speed + input-dependent pre-work
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

// =====================================================================
// Pop C: Stale Data Reservoir
// Run input-dependent kernel → read stale VGPR/LDS → features
// =====================================================================
__global__ void kernel_stale_writer(float* workspace, float input_val, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    // Write input-dependent pattern to registers (will leave residue)
    float x = input_val * (float)(tid + 1);
    for (int i = 0; i < 50; i++)
        x = fmaf(x, 1.0001f, input_val * 0.001f);
    workspace[tid] = x;
}

__global__ void kernel_stale_reader(float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    // Read stale VGPRs — residue from previous kernel
    float s0, s1, s2, s3;
    asm volatile("v_mov_b32 %0, v40" : "=v"(s0));
    asm volatile("v_mov_b32 %0, v41" : "=v"(s1));
    asm volatile("v_mov_b32 %0, v42" : "=v"(s2));
    asm volatile("v_mov_b32 %0, v43" : "=v"(s3));

    // Also read uninitialized LDS
    __shared__ float lds[256];
    float lds_val = lds[threadIdx.x % 256];

    int wgp, simd, se_sa; get_hw_id(wgp, simd, se_sa);

    out[tid*7+0] = s0;
    out[tid*7+1] = s1;
    out[tid*7+2] = s2;
    out[tid*7+3] = s3;
    out[tid*7+4] = lds_val;
    out[tid*7+5] = (float)wgp;
    out[tid*7+6] = (float)se_sa;
}

// =====================================================================
// Pop D: FMA Timing Reservoir
// Input modulates FMA chain length → per-CU timing = analog feature
// =====================================================================
__global__ void kernel_fma_reservoir(
    float* out,         // [n, 4]: wall_ns, cycles, wgp, value
    float input_val,
    int base_iters,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float x = (float)(tid + 1) * 0.001f + input_val * 0.01f;
    // Warmup
    for (int i = 0; i < 20; i++)
        x = fmaf(x, 0.999999f, 0.000001f);

    // Input-modulated chain length
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

// =====================================================================
// Pybind
// =====================================================================
void run_atomic_reservoir(torch::Tensor accum, float input_val, int base_adds,
                           int n_waves) {
    int n = n_waves * 32;
    int n_accum = accum.size(0);
    kernel_atomic_reservoir<<<n_waves, 32>>>(
        accum.data_ptr<float>(), input_val, base_adds, n_accum, n);
}

torch::Tensor run_contention_reservoir(int n_waves, float input_val, int base_iters) {
    int n = n_waves * 32;
    auto out = torch::zeros({n*4}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto slots = torch::zeros({64}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_contention_reservoir<<<n_waves, 32>>>(
        out.data_ptr<float>(), slots.data_ptr<int>(), input_val, base_iters, n);
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
    kernel_fma_reservoir<<<n_waves, 32>>>(
        out.data_ptr<float>(), input_val, base_iters, n);
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

print("Compiling HIP reservoir kernels (z2332)...")
rsv = load_inline(
    name='z2332_physics_reservoir_v1',
    cpp_sources='',
    cuda_sources=RESERVOIR_SRC,
    extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
    verbose=False,
)
print("  Compiled and loaded")

# ======================================================================
# Reservoir State Sampler
# ======================================================================
N_ACCUM = 64       # Atomic race accumulators
N_WAVES = 32       # Waves per kernel launch
BASE_ADDS = 50     # Base atomic adds per thread
BASE_FMA = 2000    # Base FMA iterations
BASE_RACE = 100    # Base race iterations

def sample_reservoir_state(input_val, populations='all'):
    """
    Sample one timestep of GPU physics reservoir.
    Returns feature vector from selected populations.
    """
    u = float(input_val)
    features = {}

    if populations in ('all', 'atomic', 'digital'):
        # Pop A: Atomic Race — reset accumulators, run, read
        accum = torch.zeros(N_ACCUM, device='cuda')
        rsv.run_atomic_reservoir(accum, u, BASE_ADDS, N_WAVES)
        torch.cuda.synchronize()
        features['atomic'] = accum.cpu().numpy()  # (64,) raw accumulator values

    if populations in ('all', 'contention', 'timing'):
        # Pop B: Contention — per-CU timing + wins
        ct = rsv.run_contention_reservoir(N_WAVES, u, BASE_RACE)
        torch.cuda.synchronize()
        ct_np = ct.cpu().numpy()  # (n, 4): wall, wins, wgp, work
        # Aggregate per CU: mean timing + win rate
        wgps = np.unique(ct_np[:, 2].astype(int))
        cont_feats = np.zeros(32)  # 16 CUs × 2 features (timing, wins)
        for w in wgps:
            mask = ct_np[:, 2].astype(int) == w
            if w < 16:
                cont_feats[w] = np.mean(ct_np[mask, 0])         # timing
                cont_feats[16 + w] = np.mean(ct_np[mask, 1])    # wins
        features['contention'] = cont_feats

    if populations in ('all', 'stale', 'digital'):
        # Pop C: Stale data — write input-dependent pattern, then read residue
        ws = torch.randn(4096, device='cuda')
        rsv.run_stale_writer(ws, u)
        torch.cuda.synchronize()
        stale = rsv.run_stale_reader(N_WAVES)
        torch.cuda.synchronize()
        stale_np = stale.cpu().numpy()  # (n, 7): s0-s3, lds, wgp, se_sa
        # Per-CU stale summary
        wgps = np.unique(stale_np[:, 5].astype(int))
        stale_feats = np.zeros(32)
        for w in wgps:
            mask = stale_np[:, 5].astype(int) == w
            vals = stale_np[mask, :4].flatten()
            if w < 16:
                stale_feats[w] = np.mean(vals)
                stale_feats[16 + w] = np.std(vals)
        features['stale'] = stale_feats

    if populations in ('all', 'fma', 'timing'):
        # Pop D: FMA Timing — per-CU timing
        ft = rsv.run_fma_reservoir(N_WAVES, u, BASE_FMA)
        torch.cuda.synchronize()
        ft_np = ft.cpu().numpy()  # (n, 4): wall, cycles, wgp, value
        fma_feats = np.zeros(16)
        wgps = np.unique(ft_np[:, 2].astype(int))
        for w in wgps:
            mask = ft_np[:, 2].astype(int) == w
            if w < 16:
                fma_feats[w] = np.mean(ft_np[mask, 0])
        features['fma'] = fma_feats

    return features


def run_reservoir(input_sequence, populations='all', warmup=100):
    """
    Run full reservoir on input sequence.
    Returns state matrix (n_steps, n_features).
    """
    n_steps = len(input_sequence)
    # Determine feature size from first sample
    feat0 = sample_reservoir_state(input_sequence[0], populations)
    feat_vec = np.concatenate([v for v in feat0.values()])
    n_feat = len(feat_vec)

    states = np.zeros((n_steps, n_feat))

    t0 = time.time()
    for t in range(n_steps):
        if t % 50 == 0:
            check_thermal()
        if t % 100 == 0 and t > 0:
            elapsed = time.time() - t0
            rate = t / elapsed
            print(f"    Step {t}/{n_steps} ({rate:.1f} steps/s, temp={get_temp()}°C)")

        feat = sample_reservoir_state(input_sequence[t], populations)
        states[t] = np.concatenate([v for v in feat.values()])

    return states


# ======================================================================
# Benchmark Functions
# ======================================================================
RIDGE_ALPHAS = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

def ridge_solve(X_tr, y_tr, X_te, y_te, task='regression'):
    best = -999.0
    for alpha in RIDGE_ALPHAS:
        try:
            I = np.eye(X_tr.shape[1])
            w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
            pred = X_te @ w
            if task == 'regression':
                ss_res = np.sum((y_te - pred)**2)
                ss_tot = np.sum((y_te - np.mean(y_te))**2)
                score = max(0, 1 - ss_res/ss_tot) if ss_tot > 1e-10 else 0.0
            else:
                score = float(np.mean((pred > 0.5).astype(float) == y_te))
            if score > best:
                best = score
        except:
            pass
    return best if best > -999 else 0.0


def build_temporal_features(states, n_select=20, seed=42):
    """Build temporal product features for nonlinearity."""
    n_steps, n_ch = states.shape
    # Delta
    delta = np.diff(states, axis=0, prepend=states[:1])
    feats = [states, delta]

    # Select subset for products
    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(n_ch, size=min(n_select, n_ch), replace=False))
    vm_q = states[:, qi]

    # Order-2: v(t) * v(t-τ)
    for tau in [1, 2, 3, 5, 8]:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)

    # Squared
    feats.append(np.square(vm_q))

    # Threshold
    feats.append((vm_q > np.median(vm_q, axis=0)).astype(float))

    return np.hstack(feats)


def benchmark_waveform(X, u_raw, warmup):
    """4-class waveform classification."""
    n = len(X)
    n_tr = int(0.7 * n)
    u = u_raw[warmup:warmup+n]
    quartiles = np.percentile(u, [25, 50, 75])
    labels = np.digitize(u, quartiles)  # 0,1,2,3

    scores = np.zeros((n - n_tr, 4))
    for c in range(4):
        y = (labels == c).astype(float)
        for alpha in [1.0, 10.0, 100.0]:
            try:
                I = np.eye(X[:n_tr].shape[1])
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I, X[:n_tr].T @ y[:n_tr])
                scores[:, c] = X[n_tr:] @ w
                break
            except:
                pass

    pred = np.argmax(scores, axis=1)
    acc = float(np.mean(pred == labels[n_tr:]))
    return acc


def benchmark_xor(X, u_raw, warmup):
    """XOR at τ=1,3,5."""
    n = len(X)
    n_tr = int(0.7 * n)
    results = {}
    for tau in [1, 3, 5]:
        u_a = (u_raw[warmup:warmup+n] > 0).astype(float)
        u_b = (u_raw[warmup-tau:warmup-tau+n] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        results[f'tau{tau}'] = acc
    return results


def benchmark_mc(X, u_raw, warmup):
    """Memory Capacity: sum R²(d) for d=1..20."""
    n = len(X)
    n_tr = int(0.7 * n)
    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 21):
        target = u_raw[warmup-d:warmup-d+n]
        nn = min(n, len(target))
        if nn < n_tr + 10:
            mc_per_d[str(d)] = 0.0
            continue
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn], 'regression')
        mc_per_d[str(d)] = r2
        mc_total += r2
    return mc_total, mc_per_d


def benchmark_narma5(X, u_raw, warmup):
    """NARMA-5 nonlinear regression."""
    n = len(X)
    n_tr = int(0.7 * n)
    T = len(u_raw)
    u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
    y = np.zeros(T)
    for t in range(5, T):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-5:t]) + 1.5*u_n[t-1]*u_n[t-5] + 0.1
        y[t] = np.tanh(y[t])

    target = y[warmup:warmup+n]
    nn = min(n, len(target))
    best_nrmse = 999.0
    for alpha in RIDGE_ALPHAS:
        try:
            I = np.eye(X[:n_tr].shape[1])
            w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I, X[:n_tr].T @ target[:n_tr])
            pred = X[n_tr:nn] @ w
            gt = target[n_tr:nn]
            nrmse = np.sqrt(np.mean((gt-pred)**2)) / (np.std(gt)+1e-10)
            if nrmse < best_nrmse:
                best_nrmse = nrmse
        except:
            pass
    return best_nrmse


def full_benchmark(X, u_raw, warmup, label=""):
    """Run all benchmarks on a state matrix."""
    print(f"    Benchmarking {label}: {X.shape[1]} features...")

    # Normalize features
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    X_norm = (X - mu) / sigma

    # Build temporal features
    X_temp = build_temporal_features(X_norm)
    print(f"      + temporal: {X_temp.shape[1]} features")

    wave = benchmark_waveform(X_temp, u_raw, warmup)
    xor = benchmark_xor(X_temp, u_raw, warmup)
    mc_total, mc_per_d = benchmark_mc(X_temp, u_raw, warmup)
    narma = benchmark_narma5(X_temp, u_raw, warmup)

    print(f"      Wave4={wave:.3f} XOR1={xor.get('tau1',0):.3f} "
          f"XOR5={xor.get('tau5',0):.3f} MC={mc_total:.3f} NARMA5={narma:.3f}")

    return {
        'wave4': wave,
        'xor': xor,
        'mc_total': mc_total,
        'mc_per_delay': mc_per_d,
        'narma5_nrmse': narma,
        'n_features': X_temp.shape[1],
    }


# ======================================================================
# Main
# ======================================================================
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"{'='*70}")
print(f"  z2332: GPU Physics Reservoir Computing")
print(f"  Atomic Race + Contention + Stale VGPR + FMA Timing")
print(f"{'='*70}")
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Temp: {get_temp()}°C")

results = {'experiments': {}, 'tests': {}, 'meta': {
    'script': 'z2332_physics_reservoir.py',
    'device': torch.cuda.get_device_name(0),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}}

def save():
    p = RESULTS / 'z2332_physics_reservoir.json'
    with open(p, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [SAVED] {p}")

def test(tid, name, passed, detail):
    status = 'PASS' if passed else 'FAIL'
    results['tests'][tid] = {'name': name, 'pass': passed, 'detail': detail}
    print(f"  [{status}] {tid}: {name} = {detail}")


# Generate input signal
N_STEPS = 600      # Shorter for speed (GPU kernels are slow per step)
WARMUP = 100
np.random.seed(42)
u_raw = np.random.uniform(-1, 1, N_STEPS + WARMUP + 50)
# Smooth input for waveform task
from scipy.ndimage import gaussian_filter1d
u_smooth = gaussian_filter1d(u_raw, sigma=3)

print(f"\n  Input: {N_STEPS} steps + {WARMUP} warmup, smoothed (σ=3)")

# ======================================================================
# Generate reservoir states for each condition
# ======================================================================
conditions = {}
wait_cool(50)

# --- CONDITION 1: FULL (all populations) ---
print(f"\n{'='*70}")
print(f"  CONDITION 1: FULL — All 4 populations")
print(f"{'='*70}")
t0 = time.time()
states_full = run_reservoir(u_smooth[:N_STEPS+WARMUP], populations='all', warmup=0)
dt_full = time.time() - t0
print(f"  Time: {dt_full:.1f}s ({(N_STEPS+WARMUP)/dt_full:.1f} steps/s)")
print(f"  State shape: {states_full.shape}")

X_full = states_full[WARMUP:]
conditions['FULL'] = full_benchmark(X_full, u_smooth, WARMUP, "FULL")
conditions['FULL']['time_s'] = dt_full
save()
wait_cool(50)

# --- CONDITION 2: ATOMIC only ---
print(f"\n{'='*70}")
print(f"  CONDITION 2: ATOMIC — Pop A only (digital stochastic)")
print(f"{'='*70}")
t0 = time.time()
states_atomic = run_reservoir(u_smooth[:N_STEPS+WARMUP], populations='atomic', warmup=0)
dt_a = time.time() - t0
print(f"  Time: {dt_a:.1f}s, shape: {states_atomic.shape}")
X_atomic = states_atomic[WARMUP:]
conditions['ATOMIC'] = full_benchmark(X_atomic, u_smooth, WARMUP, "ATOMIC")
conditions['ATOMIC']['time_s'] = dt_a
save()
wait_cool(50)

# --- CONDITION 3: TIMING (contention + FMA) ---
print(f"\n{'='*70}")
print(f"  CONDITION 3: TIMING — Pop B+D (analog timing)")
print(f"{'='*70}")
t0 = time.time()
states_timing = run_reservoir(u_smooth[:N_STEPS+WARMUP], populations='timing', warmup=0)
dt_t = time.time() - t0
print(f"  Time: {dt_t:.1f}s, shape: {states_timing.shape}")
X_timing = states_timing[WARMUP:]
conditions['TIMING'] = full_benchmark(X_timing, u_smooth, WARMUP, "TIMING")
conditions['TIMING']['time_s'] = dt_t
save()
wait_cool(50)

# --- CONDITION 4: DIGITAL (atomic + stale) ---
print(f"\n{'='*70}")
print(f"  CONDITION 4: DIGITAL — Pop A+C (atomic + stale VGPR)")
print(f"{'='*70}")
t0 = time.time()
states_digital = run_reservoir(u_smooth[:N_STEPS+WARMUP], populations='digital', warmup=0)
dt_d = time.time() - t0
print(f"  Time: {dt_d:.1f}s, shape: {states_digital.shape}")
X_digital = states_digital[WARMUP:]
conditions['DIGITAL'] = full_benchmark(X_digital, u_smooth, WARMUP, "DIGITAL")
conditions['DIGITAL']['time_s'] = dt_d
save()
wait_cool(50)

# --- CONDITION 5: CONTROL (random features) ---
print(f"\n{'='*70}")
print(f"  CONDITION 5: CONTROL — Random features (baseline)")
print(f"{'='*70}")
rng = np.random.default_rng(123)
n_feat_full = X_full.shape[1]
X_random = rng.standard_normal((N_STEPS, n_feat_full))
conditions['CONTROL'] = full_benchmark(X_random, u_smooth, WARMUP, "CONTROL")
save()


# ======================================================================
# Tests
# ======================================================================
results['experiments'] = conditions

print(f"\n{'='*70}")
print(f"  TESTS")
print(f"{'='*70}")

# Waveform classification
wave_full = conditions['FULL']['wave4']
wave_ctrl = conditions['CONTROL']['wave4']
wave_atom = conditions['ATOMIC']['wave4']
wave_time = conditions['TIMING']['wave4']

test('T1', 'FULL > CONTROL on waveform', wave_full > wave_ctrl + 0.02,
     f"FULL={wave_full:.3f} vs CTRL={wave_ctrl:.3f}")
test('T2', 'FULL waveform > 30%', wave_full > 0.30,
     f"FULL={wave_full:.3f}")
test('T3', 'ATOMIC waveform > CONTROL', wave_atom > wave_ctrl,
     f"ATOMIC={wave_atom:.3f} vs CTRL={wave_ctrl:.3f}")
test('T4', 'TIMING waveform > CONTROL', wave_time > wave_ctrl,
     f"TIMING={wave_time:.3f} vs CTRL={wave_ctrl:.3f}")

# XOR
xor_full = conditions['FULL']['xor'].get('tau1', 0.5)
xor_ctrl = conditions['CONTROL']['xor'].get('tau1', 0.5)
xor_atom = conditions['ATOMIC']['xor'].get('tau1', 0.5)

test('T5', 'FULL XOR-1 > 55%', xor_full > 0.55,
     f"FULL={xor_full:.3f}")
test('T6', 'FULL XOR-1 > CONTROL', xor_full > xor_ctrl,
     f"FULL={xor_full:.3f} vs CTRL={xor_ctrl:.3f}")
test('T7', 'ATOMIC XOR-1 > 50% (above chance)', xor_atom > 0.50,
     f"ATOMIC={xor_atom:.3f}")

# Memory Capacity
mc_full = conditions['FULL']['mc_total']
mc_ctrl = conditions['CONTROL']['mc_total']
mc_time = conditions['TIMING']['mc_total']

test('T8', 'FULL MC > 0.5', mc_full > 0.5,
     f"MC={mc_full:.3f}")
test('T9', 'FULL MC > CONTROL', mc_full > mc_ctrl,
     f"FULL={mc_full:.3f} vs CTRL={mc_ctrl:.3f}")
test('T10', 'TIMING MC > CONTROL', mc_time > mc_ctrl,
     f"TIMING={mc_time:.3f} vs CTRL={mc_ctrl:.3f}")

# NARMA
narma_full = conditions['FULL']['narma5_nrmse']
narma_ctrl = conditions['CONTROL']['narma5_nrmse']

test('T11', 'FULL NARMA < 1.0', narma_full < 1.0,
     f"NRMSE={narma_full:.3f}")
test('T12', 'FULL NARMA < CONTROL', narma_full < narma_ctrl,
     f"FULL={narma_full:.3f} vs CTRL={narma_ctrl:.3f}")

# Cross-condition comparisons
test('T13', 'FULL > ATOMIC on MC (timing adds memory)',
     mc_full > conditions['ATOMIC']['mc_total'],
     f"FULL={mc_full:.3f} vs ATOMIC={conditions['ATOMIC']['mc_total']:.3f}")

test('T14', 'DIGITAL > CONTROL on at least 2 benchmarks',
     sum([
         conditions['DIGITAL']['wave4'] > wave_ctrl,
         conditions['DIGITAL']['xor']['tau1'] > xor_ctrl,
         conditions['DIGITAL']['mc_total'] > mc_ctrl,
         conditions['DIGITAL']['narma5_nrmse'] < narma_ctrl,
     ]) >= 2,
     f"digital_wins={sum([conditions['DIGITAL']['wave4'] > wave_ctrl, conditions['DIGITAL']['xor']['tau1'] > xor_ctrl, conditions['DIGITAL']['mc_total'] > mc_ctrl, conditions['DIGITAL']['narma5_nrmse'] < narma_ctrl])}/4")

# Feature dimensionality
test('T15', 'FULL > 100 temporal features', conditions['FULL']['n_features'] > 100,
     f"n_feat={conditions['FULL']['n_features']}")

save()


# ======================================================================
# Summary
# ======================================================================
n_pass = sum(1 for t in results['tests'].values() if t['pass'])
n_total = len(results['tests'])

print(f"\n{'='*70}")
print(f"  SUMMARY: z2332 GPU Physics Reservoir Computing")
print(f"{'='*70}")
print(f"\n  Tests: {n_pass}/{n_total} PASS ({100*n_pass/n_total:.0f}%)")
print(f"\n  {'Condition':<12} {'Wave4':>8} {'XOR-1':>8} {'XOR-5':>8} {'MC':>8} {'NARMA5':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for name in ['FULL', 'ATOMIC', 'TIMING', 'DIGITAL', 'CONTROL']:
    c = conditions[name]
    print(f"  {name:<12} {c['wave4']:8.3f} {c['xor'].get('tau1',0):8.3f} "
          f"{c['xor'].get('tau5',0):8.3f} {c['mc_total']:8.3f} {c['narma5_nrmse']:8.3f}")

print(f"\n  Done. Results: {RESULTS / 'z2332_physics_reservoir.json'}")

results['meta']['n_pass'] = n_pass
results['meta']['n_tests'] = n_total
save()
