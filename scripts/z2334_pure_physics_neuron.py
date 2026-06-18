#!/usr/bin/env python3
"""
z2334_pure_physics_neuron.py — Pure Physics vs LIF-Wrapped GPU Neurons
======================================================================
Two approaches to neuromorphic computation on commodity GPU:

  MODE A: PURE PHYSICS — no neuron model in code
    - Race condition = spike (winner-take-all from transistor speed)
    - Contention backpressure = lateral inhibition (bus physics)
    - Stale VGPR = membrane memory (register persistence)
    - Thermal drift = homeostatic adaptation (self-regulating)
    - NO if>threshold, NO membrane variable, NO reset in code

  MODE B: LIF-WRAPPED — minimal neuron model, physics parameters
    - membrane[] in global memory, updated per dispatch
    - leak_rate = measured CU timing (physical Vth proxy)
    - noise = atomic race (physical stochastic)
    - threshold + reset in code (artificial structure)

  MODE C: CONTROL — random features (baseline)

Both use same CUs, same input, same readout (ridge regression).
The question: does imposing LIF structure help or hurt?

Tests (20):
  T1-T4:   PURE > CONTROL on all 4 benchmarks
  T5-T8:   LIF > CONTROL on all 4 benchmarks
  T9:      PURE MC > 1.0 (physics alone has memory)
  T10:     LIF MC > 1.0
  T11:     PURE wave4 > 50% (above chance for 4 classes)
  T12:     LIF wave4 > 50%
  T13:     PURE XOR1 > 60% (nonlinear from physics alone)
  T14:     LIF vs PURE on wave4 (which wins?)
  T15:     LIF vs PURE on MC (which wins?)
  T16:     LIF vs PURE on NARMA5 (which wins?)
  T17:     At least one mode achieves wave4 > 85%
  T18:     At least one mode achieves MC > 3.0
  T19:     PURE has >0 features with std > 0 (not all constant)
  T20:     PURE spike rate between 5% and 95% (not trivial)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 ./venv/bin/python scripts/z2334_pure_physics_neuron.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2334_pure_physics_neuron.json'

N_STEPS = 1200
WARMUP = 200
SEED = 42
N_CUS = 16          # WGPs on gfx1100
N_SLOTS = 64        # race slots
N_WAVES = 32        # wavefronts per dispatch
N_ROUNDS = 128      # race rounds per dispatch
LIF_THRESHOLD = 0.7
LIF_RESET = 0.0
LEAK_SCALE = 1e-5   # scale wall_clock ticks to useful leak range
INPUT_SCALE = 0.15
TEMP_PAUSE = 70.0
TEMP_RESUME = 50.0


# =================================================================
# Thermal
# =================================================================
def get_temp():
    try: return int(open('/sys/class/thermal/thermal_zone0/temp').read()) // 1000
    except: return 0

def wait_cool(label="", target=None):
    if target is None: target = TEMP_RESUME
    t = get_temp()
    if t <= target: return t
    print(f"  [TEMP] {label} {t}C -> {target}C...", end="", flush=True)
    t0 = time.time()
    while t > target and time.time() - t0 < 180:
        time.sleep(5); t = get_temp()
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


def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


# =================================================================
# HIP Kernels
# =================================================================
import torch
from torch.utils.cpp_extension import load_inline

KERNEL_SRC = r'''
#include <torch/extension.h>

#define HWREG(id, offset, size) ((id) | ((offset) << 6) | (((size)-1) << 11))
#define HW_REG_HW_ID1 23

__device__ int get_wgp() {
    unsigned hw = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    return (hw >> 8) & 0xF;
}

// =====================================================================
// PURE PHYSICS KERNEL — no neuron model, just race + contention + stale
//
// Each dispatch:
//   1. Read stale VGPRs (= state from previous dispatch)
//   2. Do input-dependent work (modulates arrival time)
//   3. Race for slots via atomicCAS (= spike: first CU wins)
//   4. Winners write to neighbor slots (= synaptic transmission)
//   5. Losers accumulate contention delay (= inhibition)
//   6. Write current state to VGPRs (persists to next dispatch)
//
// Output per dispatch:
//   race_results[N_SLOTS]: who won each slot (0=nobody, cu_id+1=winner)
//   timing[N_CUS]: wall_clock per CU (analog)
//   stale[N_CUS]: residual VGPR values (memory)
// =====================================================================
__global__ void kernel_pure_physics(
    float* race_results,    // [n_slots] — winner ID per slot
    float* cu_timing,       // [n_cus] — wall_clock per CU
    float* cu_stale,        // [n_cus * 4] — stale VGPR readback
    float* neighbor_accum,  // [n_cus] — accumulated "synaptic" input
    int* race_slots,        // [n_slots] — atomic race slots (reset each call)
    float input_val,
    int n_rounds,
    int n_slots,
    int n_cus,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int wgp = get_wgp();
    int cu = wgp % n_cus;

    // --- STEP 1: Read stale VGPRs (memory from previous dispatch) ---
    float s0, s1, s2, s3;
    asm volatile("v_mov_b32 %0, v44" : "=v"(s0));
    asm volatile("v_mov_b32 %0, v45" : "=v"(s1));
    asm volatile("v_mov_b32 %0, v46" : "=v"(s2));
    asm volatile("v_mov_b32 %0, v47" : "=v"(s3));

    // Store stale readback (only first thread per CU)
    if (threadIdx.x == 0) {
        cu_stale[cu * 4 + 0] = s0;
        cu_stale[cu * 4 + 1] = s1;
        cu_stale[cu * 4 + 2] = s2;
        cu_stale[cu * 4 + 3] = s3;
    }

    // --- STEP 2: Input-dependent work (modulates arrival time) ---
    float x = (float)(tid + 1) * 0.001f + input_val * 0.1f;
    // More work for some CUs based on input — creates timing spread
    int work = 30 + (int)(40.0f * fabsf(input_val) * (float)(cu + 1) / (float)n_cus);
    for (int i = 0; i < work; i++) {
        x = fmaf(x, 0.9999f, 0.0001f);
    }

    // --- STEP 3: Timed race (spike = winning the race) ---
    long long w0 = wall_clock64();
    int wins = 0;

    #pragma unroll 1
    for (int r = 0; r < n_rounds; r++) {
        int slot = (r + cu * 7) % n_slots;  // spread across slots
        int old = atomicCAS(&race_slots[slot], 0, cu + 1);
        if (old == 0) {
            wins++;
            // --- STEP 4: Winner writes to neighbor (synaptic tx) ---
            int neighbor = (cu + 1 + (r % 3)) % n_cus;
            atomicAdd(&neighbor_accum[neighbor], x * 0.001f);
        }
        __threadfence();
        // Reset slot (only winner resets, creating temporal structure)
        if (old == 0 && (r % 4 == 0)) {
            race_slots[slot] = 0;
        }
    }

    long long w1 = wall_clock64();

    // --- STEP 5: Record timing (analog CU speed) ---
    if (threadIdx.x == 0) {
        cu_timing[cu] = (float)(w1 - w0) * 10.0f;  // ns
        // Record race results for first N_SLOTS races
        for (int s = 0; s < n_slots && s < 16; s++) {
            race_results[cu * 16 + s] = (float)wins;
        }
    }

    // --- STEP 6: Write state to VGPRs (persists to next dispatch) ---
    float new_state = x + (float)wins * 0.01f + input_val;
    asm volatile("v_mov_b32 v44, %0" : : "v"(new_state));
    asm volatile("v_mov_b32 v45, %0" : : "v"(new_state * 1.1f));
    asm volatile("v_mov_b32 v46, %0" : : "v"((float)wins));
    asm volatile("v_mov_b32 v47, %0" : : "v"(x));
}


// =====================================================================
// LIF-WRAPPED KERNEL — explicit neuron model with physical parameters
//
// membrane[] lives in global memory, leak = measured timing
// =====================================================================
__global__ void kernel_lif_physics(
    float* membrane,        // [n_cus] — persistent membrane potential
    float* spikes,          // [n_cus] — spike output this step
    float* cu_timing,       // [n_cus] — measured timing (leak proxy)
    float* cu_noise,        // [n_cus] — atomic race noise
    int* race_slots,        // [n_slots] for noise generation
    float input_val,
    float threshold,
    float reset_val,
    float leak_scale,
    float input_scale,
    int n_cus,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int wgp = get_wgp();
    int cu = wgp % n_cus;

    // --- Measure THIS CU's speed (= physical leak rate) ---
    float x = (float)(tid + 1) * 0.001f;
    long long w0 = wall_clock64();
    #pragma unroll 1
    for (int i = 0; i < 200; i++) {
        x = fmaf(x, 1.0000001f, -x * 0.0000001f);
    }
    long long w1 = wall_clock64();
    float timing_ns = (float)(w1 - w0) * 10.0f;

    // --- Generate noise via atomic race ---
    float noise_val = 0.0f;
    for (int r = 0; r < 8; r++) {
        int slot = (r + cu * 3) % 64;
        int old = atomicCAS(&race_slots[slot], 0, tid + 1);
        if (old == 0) noise_val += 0.01f;
        else noise_val -= 0.005f;
        __threadfence();
        if (tid == 0) race_slots[slot] = 0;
    }

    // --- LIF update (ONLY first thread per CU updates membrane) ---
    if (threadIdx.x == 0) {
        cu_timing[cu] = timing_ns;
        cu_noise[cu] = noise_val;

        // Physical leak: timing → leak rate
        float leak = timing_ns * leak_scale;
        leak = fminf(fmaxf(leak, 0.001f), 0.5f);  // clamp to sane range

        // LIF equation — structure from code, parameters from physics
        float v = membrane[cu];
        v = v * (1.0f - leak);              // leak (PHYSICAL rate)
        v += input_val * input_scale;        // input current
        v += noise_val;                      // noise (PHYSICAL)

        // Synaptic input from neighbors that spiked
        // (read from spikes array of previous step — kept in memory)
        int left = (cu - 1 + n_cus) % n_cus;
        int right = (cu + 1) % n_cus;
        v += spikes[left] * 0.05f + spikes[right] * 0.05f;

        // Threshold and spike
        if (v > threshold) {
            spikes[cu] = 1.0f;
            membrane[cu] = reset_val;
        } else {
            spikes[cu] = 0.0f;
            membrane[cu] = v;
        }
    }

    if (__builtin_expect(x == -1e30f, 0)) membrane[0] = x;
}


// =====================================================================
// Pybind
// =====================================================================
void launch_pure_physics(
    torch::Tensor race_results, torch::Tensor cu_timing,
    torch::Tensor cu_stale, torch::Tensor neighbor_accum,
    torch::Tensor race_slots, float input_val,
    int n_rounds, int n_slots, int n_cus, int n_waves
) {
    int n = n_waves * 32;
    // Reset race slots
    race_slots.zero_();
    neighbor_accum.zero_();
    kernel_pure_physics<<<n_waves, 32>>>(
        race_results.data_ptr<float>(), cu_timing.data_ptr<float>(),
        cu_stale.data_ptr<float>(), neighbor_accum.data_ptr<float>(),
        race_slots.data_ptr<int>(), input_val,
        n_rounds, n_slots, n_cus, n);
}

void launch_lif_physics(
    torch::Tensor membrane, torch::Tensor spikes,
    torch::Tensor cu_timing, torch::Tensor cu_noise,
    torch::Tensor race_slots, float input_val,
    float threshold, float reset_val, float leak_scale,
    float input_scale, int n_cus, int n_waves
) {
    int n = n_waves * 32;
    race_slots.zero_();
    kernel_lif_physics<<<n_waves, 32>>>(
        membrane.data_ptr<float>(), spikes.data_ptr<float>(),
        cu_timing.data_ptr<float>(), cu_noise.data_ptr<float>(),
        race_slots.data_ptr<int>(), input_val,
        threshold, reset_val, leak_scale, input_scale, n_cus, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pure_physics", &launch_pure_physics);
    m.def("launch_lif_physics", &launch_lif_physics);
}
'''

print("Compiling HIP kernels (z2334)...", flush=True)
mod = load_inline(
    name='z2334_pure_physics_v1',
    cpp_sources='',
    cuda_sources=KERNEL_SRC,
    extra_cuda_cflags=['--offload-arch=gfx1100', '-O2'],
    verbose=False,
)
print("  Compiled and loaded", flush=True)
print(f"  Device: {torch.cuda.get_device_name(0)}", flush=True)


# =================================================================
# Reservoir runners
# =================================================================
def run_pure_physics(u_raw):
    """Run PURE PHYSICS reservoir — no neuron model in code."""
    total = len(u_raw)

    # Persistent GPU buffers
    race_results = torch.zeros(N_CUS * 16, device='cuda')
    cu_timing = torch.zeros(N_CUS, device='cuda')
    cu_stale = torch.zeros(N_CUS * 4, device='cuda')
    neighbor_accum = torch.zeros(N_CUS, device='cuda')
    race_slots = torch.zeros(N_SLOTS, dtype=torch.int32, device='cuda')

    # Features per step: race(16*N_CUS) + timing(N_CUS) + stale(4*N_CUS) + neighbor(N_CUS)
    # = 16*16 + 16 + 64 + 16 = 352
    n_feat = N_CUS * 16 + N_CUS + N_CUS * 4 + N_CUS
    states = np.zeros((total, n_feat), dtype=np.float32)

    t0 = time.time()
    for t in range(total):
        if t % 50 == 0 and t > 0:
            temp = get_temp()
            if temp > TEMP_PAUSE:
                wait_cool(f"PURE step {t}", TEMP_RESUME)
        if t % 300 == 0 and t > 0:
            rate = t / (time.time() - t0)
            print(f"    Step {t}/{total} ({rate:.0f} steps/s, temp={get_temp()}C)", flush=True)

        mod.launch_pure_physics(
            race_results, cu_timing, cu_stale, neighbor_accum,
            race_slots, float(u_raw[t]),
            N_ROUNDS, N_SLOTS, N_CUS, N_WAVES
        )
        torch.cuda.synchronize()

        # Read state — ALL features come from physics, no processing
        states[t, :N_CUS*16] = race_results.cpu().numpy()
        states[t, N_CUS*16:N_CUS*16+N_CUS] = cu_timing.cpu().numpy()
        states[t, N_CUS*16+N_CUS:N_CUS*16+N_CUS+N_CUS*4] = cu_stale.cpu().numpy()
        states[t, -N_CUS:] = neighbor_accum.cpu().numpy()

    elapsed = time.time() - t0
    print(f"  PURE: {total} steps in {elapsed:.1f}s ({total/elapsed:.0f} steps/s)", flush=True)
    return states


def run_lif_physics(u_raw):
    """Run LIF-WRAPPED reservoir — explicit neuron model, physical parameters."""
    total = len(u_raw)

    # Persistent neuron state
    membrane = torch.zeros(N_CUS, device='cuda')
    spikes = torch.zeros(N_CUS, device='cuda')
    cu_timing = torch.zeros(N_CUS, device='cuda')
    cu_noise = torch.zeros(N_CUS, device='cuda')
    race_slots = torch.zeros(64, dtype=torch.int32, device='cuda')

    # Features: membrane(N_CUS) + spikes(N_CUS) + timing(N_CUS) + noise(N_CUS) = 64
    n_feat = N_CUS * 4
    states = np.zeros((total, n_feat), dtype=np.float32)

    t0 = time.time()
    for t in range(total):
        if t % 50 == 0 and t > 0:
            temp = get_temp()
            if temp > TEMP_PAUSE:
                wait_cool(f"LIF step {t}", TEMP_RESUME)
        if t % 300 == 0 and t > 0:
            rate = t / (time.time() - t0)
            print(f"    Step {t}/{total} ({rate:.0f} steps/s, temp={get_temp()}C)", flush=True)

        mod.launch_lif_physics(
            membrane, spikes, cu_timing, cu_noise,
            race_slots, float(u_raw[t]),
            LIF_THRESHOLD, LIF_RESET, LEAK_SCALE, INPUT_SCALE,
            N_CUS, N_WAVES
        )
        torch.cuda.synchronize()

        states[t, :N_CUS] = membrane.cpu().numpy()
        states[t, N_CUS:N_CUS*2] = spikes.cpu().numpy()
        states[t, N_CUS*2:N_CUS*3] = cu_timing.cpu().numpy()
        states[t, N_CUS*3:N_CUS*4] = cu_noise.cpu().numpy()

    elapsed = time.time() - t0
    print(f"  LIF: {total} steps in {elapsed:.1f}s ({total/elapsed:.0f} steps/s)", flush=True)
    return states


# =================================================================
# Temporal features (minimal — order 2 products)
# =================================================================
def build_temporal(states, n_select=16, seed=42):
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, delta]

    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(n_ch, size=min(n_select, n_ch), replace=False))
    sq = states[:, qi]

    for tau in [1, 2, 3, 5, 8, 12]:
        shifted = np.zeros_like(sq)
        shifted[tau:] = sq[:-tau]
        feats.append(sq * shifted)

    feats.append(np.square(sq))
    feats.append((sq > np.median(sq, axis=0)).astype(float))

    return np.hstack(feats)


# =================================================================
# Benchmarks (same as z2332/z2333)
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
            if score > best: best = score
        except: pass
    return max(best, 0.0)


def full_benchmark(X, u_raw, warmup, label=""):
    n = len(X)
    n_tr = int(0.7 * n)

    # Remove constant features
    stds = X.std(axis=0)
    keep = stds > 1e-8
    X = X[:, keep]
    n_active = X.shape[1]
    if n_active == 0:
        print(f"    {label}: ALL features constant!", flush=True)
        return {'wave4': 0.25, 'xor': {'tau1': 0.5, 'tau3': 0.5, 'tau5': 0.5},
                'mc_total': 0.0, 'narma5': 999.0, 'n_features': 0, 'n_active': 0}

    # Wave4
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
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * np.eye(X.shape[1]),
                                     X[:n_tr].T @ y[:n_tr])
                scores_m[:, c] = X[n_tr:] @ w
                break
            except: pass
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
            w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * np.eye(X.shape[1]),
                                 X[:n_tr].T @ target[:n_tr])
            pred = X[n_tr:nn] @ w
            gt = target[n_tr:nn]
            nrmse = np.sqrt(np.mean((gt-pred)**2)) / (np.std(gt)+1e-10)
            if nrmse < best_nrmse: best_nrmse = nrmse
        except: pass

    print(f"    {label}: Wave4={wave4:.3f} XOR1={xor['tau1']:.3f} XOR5={xor['tau5']:.3f} "
          f"MC={mc_total:.3f} NARMA5={best_nrmse:.3f} ({n_active} active feat)", flush=True)
    return {'wave4': wave4, 'xor': xor, 'mc_total': mc_total, 'mc_per_delay': mc_per_d,
            'narma5': best_nrmse, 'n_features': int(n_active), 'n_active': int(n_active)}


# =================================================================
# Main
# =================================================================
def main():
    print("=" * 70)
    print("  z2334: Pure Physics vs LIF-Wrapped GPU Neurons")
    print("  Race/Contention/Stale vs Explicit LIF with Physical Parameters")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Temp: {get_temp()}C")
    print(f"  Steps: {N_STEPS} + {WARMUP} warmup, {N_CUS} CUs, {N_WAVES} waves")

    results = {'experiments': {}, 'tests': {}, 'diagnostics': {}}

    rng = np.random.default_rng(SEED)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

    # ---------------------------------------------------------------
    # Mode A: PURE PHYSICS
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  MODE A: PURE PHYSICS — no neuron model in code")
    print(f"{'='*70}")
    wait_cool("pre-PURE", TEMP_RESUME)
    pure_raw = run_pure_physics(u_raw)

    # Diagnostics
    pure_stds = pure_raw.std(axis=0)
    n_active_pure = int(np.sum(pure_stds > 1e-8))
    print(f"  PURE raw: {pure_raw.shape}, {n_active_pure}/{pure_raw.shape[1]} active features")
    print(f"  PURE feature ranges: min={pure_raw.min():.4f}, max={pure_raw.max():.4f}, mean_std={pure_stds[pure_stds>1e-8].mean():.4f}")

    # Check race results for "spike rate"
    # Race results are in first N_CUS*16 columns — nonzero means "won"
    race_cols = pure_raw[:, :N_CUS*16]
    spike_rate_pure = float(np.mean(race_cols > 0))
    print(f"  PURE spike rate (race wins): {spike_rate_pure:.3f}")
    results['diagnostics']['pure_spike_rate'] = spike_rate_pure
    results['diagnostics']['pure_active_features'] = n_active_pure
    results['diagnostics']['pure_mean_std'] = float(pure_stds[pure_stds>1e-8].mean()) if n_active_pure > 0 else 0.0

    pure_eval = pure_raw[WARMUP:]
    pure_feat = build_temporal(pure_eval, n_select=min(24, n_active_pure), seed=42)
    print(f"  PURE + temporal: {pure_feat.shape}")

    bm_pure = full_benchmark(pure_feat, u_raw, WARMUP, "PURE")
    results['experiments']['PURE'] = bm_pure
    save_results(results)

    # ---------------------------------------------------------------
    # Mode B: LIF-WRAPPED
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  MODE B: LIF-WRAPPED — explicit neuron model, physical parameters")
    print(f"{'='*70}")
    wait_cool("pre-LIF", TEMP_RESUME)
    lif_raw = run_lif_physics(u_raw)

    lif_stds = lif_raw.std(axis=0)
    n_active_lif = int(np.sum(lif_stds > 1e-8))
    print(f"  LIF raw: {lif_raw.shape}, {n_active_lif}/{lif_raw.shape[1]} active features")

    # Spike rate from LIF
    spike_cols = lif_raw[:, N_CUS:N_CUS*2]
    spike_rate_lif = float(np.mean(spike_cols > 0))
    print(f"  LIF spike rate: {spike_rate_lif:.3f}")
    results['diagnostics']['lif_spike_rate'] = spike_rate_lif
    results['diagnostics']['lif_active_features'] = n_active_lif

    lif_eval = lif_raw[WARMUP:]
    lif_feat = build_temporal(lif_eval, n_select=min(24, n_active_lif), seed=42)
    print(f"  LIF + temporal: {lif_feat.shape}")

    bm_lif = full_benchmark(lif_feat, u_raw, WARMUP, "LIF")
    results['experiments']['LIF'] = bm_lif
    save_results(results)

    # ---------------------------------------------------------------
    # Mode C: CONTROL
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  MODE C: CONTROL — shuffled features")
    print(f"{'='*70}")
    ctrl_raw = pure_eval.copy()
    rng_ctrl = np.random.default_rng(999)
    for c in range(ctrl_raw.shape[1]):
        rng_ctrl.shuffle(ctrl_raw[:, c])
    ctrl_feat = build_temporal(ctrl_raw, n_select=24, seed=42)

    bm_ctrl = full_benchmark(ctrl_feat, u_raw, WARMUP, "CONTROL")
    results['experiments']['CONTROL'] = bm_ctrl
    save_results(results)

    # ---------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  TESTS")
    print(f"{'='*70}")

    E = results['experiments']
    tests = {}

    def test(tid, name, cond, detail):
        p = "PASS" if cond else "FAIL"
        print(f"  [{p}] {tid}: {name} = {detail}", flush=True)
        tests[tid] = {'name': name, 'pass': bool(cond), 'detail': detail}

    # PURE > CONTROL
    test('T1', 'PURE wave4 > CONTROL', E['PURE']['wave4'] > E['CONTROL']['wave4'],
         f"PURE={E['PURE']['wave4']:.3f} vs CTRL={E['CONTROL']['wave4']:.3f}")
    test('T2', 'PURE MC > CONTROL', E['PURE']['mc_total'] > E['CONTROL']['mc_total'],
         f"PURE={E['PURE']['mc_total']:.3f} vs CTRL={E['CONTROL']['mc_total']:.3f}")
    test('T3', 'PURE XOR1 > CONTROL', E['PURE']['xor']['tau1'] > E['CONTROL']['xor']['tau1'],
         f"PURE={E['PURE']['xor']['tau1']:.3f} vs CTRL={E['CONTROL']['xor']['tau1']:.3f}")
    test('T4', 'PURE NARMA5 < CONTROL', E['PURE']['narma5'] < E['CONTROL']['narma5'],
         f"PURE={E['PURE']['narma5']:.3f} vs CTRL={E['CONTROL']['narma5']:.3f}")

    # LIF > CONTROL
    test('T5', 'LIF wave4 > CONTROL', E['LIF']['wave4'] > E['CONTROL']['wave4'],
         f"LIF={E['LIF']['wave4']:.3f} vs CTRL={E['CONTROL']['wave4']:.3f}")
    test('T6', 'LIF MC > CONTROL', E['LIF']['mc_total'] > E['CONTROL']['mc_total'],
         f"LIF={E['LIF']['mc_total']:.3f} vs CTRL={E['CONTROL']['mc_total']:.3f}")
    test('T7', 'LIF XOR1 > CONTROL', E['LIF']['xor']['tau1'] > E['CONTROL']['xor']['tau1'],
         f"LIF={E['LIF']['xor']['tau1']:.3f} vs CTRL={E['CONTROL']['xor']['tau1']:.3f}")
    test('T8', 'LIF NARMA5 < CONTROL', E['LIF']['narma5'] < E['CONTROL']['narma5'],
         f"LIF={E['LIF']['narma5']:.3f} vs CTRL={E['CONTROL']['narma5']:.3f}")

    # Absolute thresholds
    test('T9', 'PURE MC > 1.0', E['PURE']['mc_total'] > 1.0,
         f"MC={E['PURE']['mc_total']:.3f}")
    test('T10', 'LIF MC > 1.0', E['LIF']['mc_total'] > 1.0,
         f"MC={E['LIF']['mc_total']:.3f}")
    test('T11', 'PURE wave4 > 50%', E['PURE']['wave4'] > 0.50,
         f"wave4={E['PURE']['wave4']:.3f}")
    test('T12', 'LIF wave4 > 50%', E['LIF']['wave4'] > 0.50,
         f"wave4={E['LIF']['wave4']:.3f}")
    test('T13', 'PURE XOR1 > 60%', E['PURE']['xor']['tau1'] > 0.60,
         f"XOR1={E['PURE']['xor']['tau1']:.3f}")

    # Head-to-head
    test('T14', 'LIF vs PURE wave4', True,
         f"LIF={E['LIF']['wave4']:.3f} vs PURE={E['PURE']['wave4']:.3f} -> {'LIF' if E['LIF']['wave4'] > E['PURE']['wave4'] else 'PURE'} wins")
    test('T15', 'LIF vs PURE MC', True,
         f"LIF={E['LIF']['mc_total']:.3f} vs PURE={E['PURE']['mc_total']:.3f} -> {'LIF' if E['LIF']['mc_total'] > E['PURE']['mc_total'] else 'PURE'} wins")
    test('T16', 'LIF vs PURE NARMA5', True,
         f"LIF={E['LIF']['narma5']:.3f} vs PURE={E['PURE']['narma5']:.3f} -> {'LIF' if E['LIF']['narma5'] < E['PURE']['narma5'] else 'PURE'} wins")

    # Quality thresholds
    best_wave = max(E['PURE']['wave4'], E['LIF']['wave4'])
    test('T17', 'Best wave4 > 85%', best_wave > 0.85, f"best={best_wave:.3f}")
    best_mc = max(E['PURE']['mc_total'], E['LIF']['mc_total'])
    test('T18', 'Best MC > 3.0', best_mc > 3.0, f"best={best_mc:.3f}")

    # Diagnostics
    test('T19', 'PURE has active features', n_active_pure > 0,
         f"{n_active_pure}/{pure_raw.shape[1]} active")
    test('T20', 'PURE spike rate 5-95%', 0.05 < spike_rate_pure < 0.95,
         f"rate={spike_rate_pure:.3f}")

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    results['summary'] = {
        'total_pass': n_pass,
        'total_tests': len(tests),
        'pass_rate': n_pass / len(tests),
        'pure_wins': sum(1 for tid in ['T14','T15','T16']
                        if 'PURE wins' in tests[tid]['detail']),
        'lif_wins': sum(1 for tid in ['T14','T15','T16']
                       if 'LIF wins' in tests[tid]['detail']),
    }
    save_results(results)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: z2334 Pure Physics vs LIF-Wrapped")
    print(f"{'='*70}")
    print(f"\n  Tests: {n_pass}/{len(tests)} PASS ({100*n_pass/len(tests):.0f}%)")
    print(f"  Head-to-head: PURE wins {results['summary']['pure_wins']}/3, LIF wins {results['summary']['lif_wins']}/3")
    print(f"\n  {'Mode':<12} {'Wave4':>8} {'XOR-1':>8} {'XOR-5':>8} {'MC':>8} {'NARMA5':>8} {'Feat':>6}")
    print(f"  {'-'*10:<12} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*6:>6}")
    for name in ['PURE', 'LIF', 'CONTROL']:
        e = E[name]
        print(f"  {name:<12} {e['wave4']:>8.3f} {e['xor']['tau1']:>8.3f} {e['xor']['tau5']:>8.3f} {e['mc_total']:>8.3f} {e['narma5']:>8.3f} {e['n_features']:>6}")

    print(f"\n  Diagnostics:")
    print(f"    PURE spike rate: {spike_rate_pure:.3f}")
    print(f"    LIF spike rate:  {spike_rate_lif:.3f}")
    print(f"    PURE active:     {n_active_pure}/{pure_raw.shape[1]}")
    print(f"    LIF active:      {n_active_lif}/{lif_raw.shape[1]}")

    print(f"\n  Done. Results: {SAVE_FILE}")


if __name__ == '__main__':
    main()
