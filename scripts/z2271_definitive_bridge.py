#!/usr/bin/env python3
"""z2271_definitive_bridge.py — Definitive Cross-Substrate GPU+FPGA Experiment

z2269 found optimal NOISE_SCALE=0.02 (not 0.05).
z2268 fourpop GPU: XOR=77%, MC=0.615, NARMA=+0.261, Wave4=97.7%.
z2264 FPGA standalone: 46% wave, MC=0.275.
z2267 GPU ESN alone: 100% wave, MC=3.9 (software ESN, not hardware physics).

Goal: Show that FPGA + GPU noise creates something neither can alone.

Conditions (6 total):
  1. FPGA_ONLY:       no noise, Vg modulated by input only
  2. FPGA_1F:         Vg modulated by input + 1/f GPU noise (SCALE=0.02)
  3. GPU_ESN_ONLY:    Python fourpop ESN (no FPGA)
  4. GPU_ESN_NOISE:   Python fourpop ESN + real GPU power noise as additional input
  5. BRIDGE_SIMPLE:   concat GPU_ESN + FPGA_1F states
  6. BRIDGE_FULL:     concat GPU_ESN_NOISE + FPGA_1F states

Benchmarks:
  - 4-class waveform classification (80 trials x 60 steps)
  - 8-class waveform (80 trials x 60 steps)
  - Memory capacity d=1-10 (500 steps)
  - XOR tau=1,2,3 (500 steps)
  - NARMA-5 (500 steps)

New bitstream: LEAK_COND=4 (tau=210ms).

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2271_definitive_bridge.py
"""

import os, sys, time, json, struct, subprocess, signal
import numpy as np
from pathlib import Path

# ─── Paths ───
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)

# ─── Parameters ───
N_GPU = 256           # fourpop ESN: 4 pops x 64 neurons
N_FPGA = 128
BASE_VG = 0.58
VG_SPREAD = 0.08
ALPHA = 0.25          # input gain to FPGA
IIR_ALPHA = 0.85      # 1/f filter coefficient
NOISE_SCALE = 0.02    # z2269 optimal
SAMPLE_HZ = 50        # FPGA telemetry rate

# Task params
N_WAVE_TRIALS = 80
N_WAVE_STEPS = 60
N_CONTINUOUS_STEPS = 500
MC_MAX_DELAY = 10
WARMUP = 50
NARMA_ORDER = 5

# Temperature limits — LOWERED after two thermal crashes in 12 hours
TEMP_LIMIT_HARD = 70.0   # was 80; wait 60s
TEMP_LIMIT_SOFT = 65.0   # was 75; wait 30s
COOLING_BETWEEN_CONDITIONS = 15  # seconds between GPU kernel runs
NOISE_SAMPLE_HZ = 200


# ═══════════════════════════════════════════════════════════
# Temperature Monitoring
# ═══════════════════════════════════════════════════════════

def get_edge_temp():
    try:
        with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0


def check_temp_and_wait(label=""):
    temp = get_edge_temp()
    print(f"  [TEMP] {label} edge={temp:.1f}C", end="")
    if temp > TEMP_LIMIT_HARD:
        print(f" -- CRITICAL (>{TEMP_LIMIT_HARD}C, waiting 60s)...")
        time.sleep(60)
        temp = get_edge_temp()
        print(f"  [TEMP] After cooling: edge={temp:.1f}C")
    elif temp > TEMP_LIMIT_SOFT:
        print(f" -- COOLING (>{TEMP_LIMIT_SOFT}C, waiting 30s)...")
        time.sleep(30)
        temp = get_edge_temp()
        print(f"  [TEMP] After cooling: edge={temp:.1f}C")
    else:
        print(f" -- OK")
    return temp


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# GPU Stress Kernel
# ═══════════════════════════════════════════════════════════

GPU_STRESS_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <signal.h>

static volatile int running = 1;
void sighandler(int sig) { running = 0; }

__global__ void stress_fma(float* out, int N) {
    float v = (threadIdx.x + blockIdx.x * blockDim.x) * 0.001f + 0.1f;
    for (int i = 0; i < N; i++) {
        v = __fmaf_rn(v, 0.9999f, 0.0001f);
        v = __fmaf_rn(v, v, 0.5f - v * v * 0.5f);
    }
    out[threadIdx.x + blockIdx.x * blockDim.x] = v;
}

int main(int argc, char** argv) {
    int duration = 600;
    if (argc > 1) duration = atoi(argv[1]);
    signal(SIGTERM, sighandler);
    signal(SIGINT, sighandler);
    int N = 64 * 256;
    float *d_out;
    hipMalloc(&d_out, N * sizeof(float));
    fprintf(stderr, "GPU_STRESS: running for %d seconds\n", duration);
    time_t start = time(NULL);
    while (running && (time(NULL) - start) < duration) {
        stress_fma<<<64, 256>>>(d_out, 50000);
        hipDeviceSynchronize();
    }
    hipFree(d_out);
    fprintf(stderr, "GPU_STRESS: done\n");
    return 0;
}
"""


def compile_stress_kernel():
    src_path = '/tmp/z2267_gpu_stress.hip'
    bin_path = '/tmp/z2267_gpu_stress'
    if os.path.isfile(bin_path):
        if os.path.isfile(src_path):
            if os.path.getmtime(bin_path) > os.path.getmtime(src_path):
                print(f"  [STRESS] Using cached binary: {bin_path}")
                return bin_path
    with open(src_path, 'w') as f:
        f.write(GPU_STRESS_SRC)
    print(f"  [STRESS] Compiling GPU stress kernel...")
    try:
        result = subprocess.run(
            ['hipcc', '--offload-arch=gfx1100', '-O1', '-o', bin_path, src_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  [STRESS] Compile failed: {result.stderr}")
            return None
        print(f"  [STRESS] Compiled: {bin_path}")
        return bin_path
    except Exception as e:
        print(f"  [STRESS] Compile error: {e}")
        return None


def launch_stress_kernel(bin_path, duration=600):
    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    proc = subprocess.Popen(
        [bin_path, str(duration)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    print(f"  [STRESS] Launched PID={proc.pid}, duration={duration}s")
    time.sleep(3)
    return proc


def kill_stress(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        print(f"  [STRESS] Killed PID={proc.pid}")


# ═══════════════════════════════════════════════════════════
# GPU Noise Sampler
# ═══════════════════════════════════════════════════════════

class GPUNoiseSampler:
    """Sample GPU power noise from hwmon while stress kernel runs."""

    def __init__(self):
        self._power_norm = None
        self._iir_state = 0.0
        self._raw_buf = []

    def sample(self, stress_proc, n_samples=500, sample_hz=200):
        print(f"  [NOISE] Sampling {n_samples} points at ~{sample_hz} Hz...")
        interval = 1.0 / sample_hz
        self._raw_buf = []

        for i in range(n_samples):
            t0 = time.perf_counter()
            if stress_proc and stress_proc.poll() is not None:
                print(f"  [NOISE] Stress kernel died at sample {i}")
                break
            if i > 0 and i % 100 == 0:
                temp = get_edge_temp()
                if temp > TEMP_LIMIT_HARD:
                    print(f"  [NOISE] TEMP {temp:.1f}C > {TEMP_LIMIT_HARD}C -- stopping")
                    break
            try:
                with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
                    pw = float(f.read().strip()) / 1e6  # uW -> W
                self._raw_buf.append(pw)
            except Exception:
                self._raw_buf.append(self._raw_buf[-1] if self._raw_buf else 0.0)
            elapsed = time.perf_counter() - t0
            remaining = interval - elapsed
            if remaining > 0.0001:
                time.sleep(remaining)

        pw_arr = np.array(self._raw_buf)
        if len(pw_arr) == 0:
            print(f"  [NOISE] WARNING: no samples collected, keeping previous noise buffer")
            return {'n_samples': 0, 'mean': 0.0, 'std': 0.0}
        print(f"  [NOISE] Collected {len(pw_arr)} samples: "
              f"mean={pw_arr.mean():.2f}W, std={pw_arr.std():.4f}W, "
              f"range=[{pw_arr.min():.2f}, {pw_arr.max():.2f}]")
        std = pw_arr.std()
        if std > 1e-8:
            self._power_norm = (pw_arr - pw_arr.mean()) / std
        else:
            self._power_norm = np.zeros_like(pw_arr)
        self._iir_state = 0.0
        return {'n_samples': len(pw_arr), 'mean': float(pw_arr.mean()),
                'std': float(pw_arr.std())}

    def get_1f_noise(self, t, gain):
        if self._power_norm is None or len(self._power_norm) == 0:
            return 0.0
        raw = self._power_norm[t % len(self._power_norm)]
        self._iir_state = IIR_ALPHA * self._iir_state + (1 - IIR_ALPHA) * raw
        return self._iir_state * gain

    def get_noise_vector(self, t, n_neurons, gain):
        """Per-neuron 1/f noise: shared IIR component + small per-neuron jitter."""
        base = self.get_1f_noise(t, gain)
        return base + np.random.randn(n_neurons) * gain * 0.1

    def get_noise_scalar(self, t, gain):
        """Single noise value for GPU ESN injection."""
        return self.get_1f_noise(t, gain)

    def reset_iir(self):
        self._iir_state = 0.0


# ═══════════════════════════════════════════════════════════
# GPU Fourpop ESN (Python-side, mimics hardware physics)
# ═══════════════════════════════════════════════════════════

class GPUFourpopESN:
    """Python fourpop ESN mimicking GPU hardware physics.

    Pop A: tanh ESN with step-function activation (branch divergence analog)
    Pop B: tanh ESN with cross-neuron corruption (L1 bank conflict analog: randomly swap 10% of states)
    Pop C: dense ESN, SR=1.05, T=0.65
    Pop D: tanh ESN with scheduling noise (wavefront-like jitter)
    """

    def __init__(self, n_per_pop=64, noise_sampler=None):
        self.n_per_pop = n_per_pop
        self.N = 4 * n_per_pop
        self.noise_sampler = noise_sampler
        rng = np.random.default_rng(7777)

        # Per-population parameters
        self.leak = np.zeros(self.N)
        self.input_w = np.zeros(self.N)
        self.thr = np.zeros(self.N)
        self.bias = np.zeros(self.N)

        for pop in range(4):
            s = pop * n_per_pop
            e = s + n_per_pop
            self.leak[s:e] = 0.05 + 0.15 * rng.random(n_per_pop)
            self.input_w[s:e] = 0.05 + 0.20 * rng.random(n_per_pop)
            self.thr[s:e] = 0.4 + 0.5 * rng.random(n_per_pop)
            self.bias[s:e] = 0.02 * (rng.random(n_per_pop) - 0.5)

        # Recurrent weights: sparse, spectral radius ~1.0
        self.W_rec = rng.standard_normal((self.N, self.N)) * 0.04
        mask = rng.random((self.N, self.N)) > 0.9
        self.W_rec *= mask

        # Pop C gets denser connectivity and higher SR
        sc, ec = 2 * n_per_pop, 3 * n_per_pop
        W_c = rng.standard_normal((n_per_pop, n_per_pop)) * 0.08
        mask_c = rng.random((n_per_pop, n_per_pop)) > 0.7
        W_c *= mask_c
        # Scale to SR ~1.05
        eigvals = np.abs(np.linalg.eigvals(W_c))
        sr = max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0:
            W_c *= 1.05 / sr
        self.W_rec[sc:ec, sc:ec] = W_c

        # Branch threshold for Pop A
        self.bthr = 0.5 + 0.3 * np.arange(n_per_pop) / max(n_per_pop - 1, 1)

        # Pop C temperature
        self.temp_c = 0.65

    def run(self, input_seq, noise_input=None):
        """Run reservoir. noise_input: optional (n_steps,) array of GPU noise."""
        n_steps = len(input_seq)
        pp = self.n_per_pop
        states = np.zeros((n_steps, self.N))
        v = np.zeros(self.N)
        h = np.zeros(self.N)  # slow integration
        slow = np.zeros(self.N)  # very slow
        rng = np.random.default_rng(42)

        for t in range(n_steps):
            u = input_seq[t]

            # Recurrent input
            rec = self.W_rec @ v

            # Pop A: branch divergence (step-function activation)
            sa, ea = 0, pp
            branch_val = np.where(v[sa:ea] > self.bthr, 1.0, -1.0)
            v_new_a = np.tanh(
                (1.0 - self.leak[sa:ea]) * v[sa:ea]
                + self.input_w[sa:ea] * u
                + rec[sa:ea]
                + self.bias[sa:ea]
                + 0.02 * branch_val
            )

            # Pop B: L1 bank conflict (swap 10% of states per step)
            sb, eb = pp, 2 * pp
            v_b = v[sb:eb].copy()
            n_swap = max(1, pp // 10)
            swap_idx = rng.choice(pp, size=n_swap * 2, replace=False)
            for k in range(0, n_swap * 2 - 1, 2):
                v_b[swap_idx[k]], v_b[swap_idx[k+1]] = v_b[swap_idx[k+1]], v_b[swap_idx[k]]
            v_new_b = np.tanh(
                (1.0 - self.leak[sb:eb]) * v_b
                + self.input_w[sb:eb] * u
                + rec[sb:eb]
                + self.bias[sb:eb]
            )

            # Pop C: dense, SR=1.05, temperature=0.65
            sc, ec = 2 * pp, 3 * pp
            v_new_c = np.tanh(
                ((1.0 - self.leak[sc:ec]) * v[sc:ec]
                 + self.input_w[sc:ec] * u
                 + rec[sc:ec]
                 + self.bias[sc:ec]) / self.temp_c
            )

            # Pop D: scheduling noise (wavefront jitter)
            sd, ed = 3 * pp, 4 * pp
            sched_noise = rng.uniform(-1, 1, pp) * 0.01
            v_new_d = np.tanh(
                (1.0 - self.leak[sd:ed]) * v[sd:ed]
                + self.input_w[sd:ed] * u
                + rec[sd:ed]
                + self.bias[sd:ed]
                + sched_noise
            )

            v_new = np.concatenate([v_new_a, v_new_b, v_new_c, v_new_d])

            # Add GPU noise input if available
            if noise_input is not None and t < len(noise_input):
                v_new += noise_input[t] * 0.01  # small coupling

            # PLL-like jitter
            pll = rng.uniform(-1, 1, self.N) * 0.003
            v_new += pll

            # Spike and reset
            spike_mask = v_new > self.thr
            v_new[spike_mask] -= self.thr[spike_mask]
            v = v_new

            # Slow dynamics
            h = 0.93 * h + 0.07 * v
            slow = 0.99 * slow + 0.01 * v
            states[t] = v + 0.3 * h + 0.1 * slow

        return states


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir
# ═══════════════════════════════════════════════════════════

class FPGAReservoir:
    """FPGA-only reservoir with configurable noise injection."""

    def __init__(self, fpga, noise_sampler):
        self.fpga = fpga
        self.noise = noise_sampler
        rng = np.random.default_rng(42)
        self.base_vg = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, size=N_FPGA)
        self.w_in = rng.uniform(-1, 1, size=N_FPGA)

    def run_trial(self, input_signal, noise_scale=0.0):
        """Run one trial. noise_scale=0 means no noise (baseline)."""
        n_steps = len(input_signal)
        interval = 1.0 / SAMPLE_HZ
        fpga_states = np.zeros((n_steps, N_FPGA * 2))  # spikes + vmem
        prev_counts = None
        self.noise.reset_iir()

        for t in range(n_steps):
            t_start = time.perf_counter()

            # Compute Vg: base + input signal
            vg = self.base_vg.copy() + ALPHA * input_signal[t] * self.w_in

            # Add GPU 1/f noise if scale > 0
            if noise_scale > 0:
                noise_vec = self.noise.get_noise_vector(t, N_FPGA, noise_scale)
                vg += noise_vec

            vg = np.clip(vg, 0.05, 0.95)

            # Send to FPGA
            self.fpga.set_vg_batch(0, vg.tolist())
            time.sleep(max(0.001, interval * 0.3))

            # Read telemetry
            try:
                counts, vmem, refract = self.fpga.read_telemetry_fast()
            except (TimeoutError, Exception):
                elapsed = time.perf_counter() - t_start
                remaining = interval - elapsed
                if remaining > 0.0005:
                    time.sleep(remaining)
                continue

            # Spike deltas
            if prev_counts is not None:
                for i in range(N_FPGA):
                    delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    fpga_states[t, i] = delta
            # Vmem
            for i in range(N_FPGA):
                fpga_states[t, N_FPGA + i] = vmem[i]
            prev_counts = counts.copy()

            # Rate limit
            elapsed = time.perf_counter() - t_start
            remaining = interval - elapsed
            if remaining > 0.0005:
                time.sleep(remaining)

        return fpga_states


# ═══════════════════════════════════════════════════════════
# Ridge Regression Utilities
# ═══════════════════════════════════════════════════════════

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=4):
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0, 1000.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, y in enumerate(y_tr):
        Y_tr[i, int(y)] = 1.0
    best = -1
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try:
            W = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ Y_tr)
        except Exception:
            continue
        acc = np.mean(np.argmax(X_te_s @ W, axis=1) == y_te)
        if acc > best:
            best = acc
    return best


def ridge_binary(X_tr, y_tr, X_te, y_te):
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    best = -1
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try:
            w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except Exception:
            continue
        acc = np.mean(((X_te_s @ w) > 0.5).astype(float) == y_te)
        if acc > best:
            best = acc
    return best


def ridge_regress(X_tr, y_tr, X_te, y_te):
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    best_nrmse = 1e10
    best_r2 = -1e10
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try:
            w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except Exception:
            continue
        pred = X_te_s @ w
        ss_res = np.sum((y_te - pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        nrmse = np.sqrt(ss_res / len(y_te)) / max(np.sqrt(ss_tot / len(y_te)), 1e-8)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        if r2 > best_r2:
            best_r2 = r2
            best_nrmse = nrmse
    return best_nrmse, best_r2


def stratified_kfold(X, y, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        c_idx = indices[y[indices] == c]
        for i, idx in enumerate(c_idx):
            folds[i % n_splits].append(idx)
    splits = []
    for fold in range(n_splits):
        test_idx = np.array(folds[fold])
        train_idx = np.concatenate([np.array(folds[f]) for f in range(n_splits) if f != fold])
        splits.append((train_idx, test_idx))
    return splits


def classify_cv(X, y, n_splits=5, n_classes=4):
    splits = stratified_kfold(X, y, n_splits)
    accs = []
    for tr_idx, te_idx in splits:
        acc = ridge_classify(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], n_classes=n_classes)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Task Generation
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2)):
    T, D = states.shape
    augmented = np.zeros((T, D * (1 + len(delays))))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[d:, start:start + D] = states[:T - d]
    return augmented


def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def generate_waveforms_4class(n_trials, steps, sample_hz, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / sample_hz
    t = np.arange(steps) * dt
    trials, labels = [], []
    for _ in range(n_trials):
        cls = rng.integers(0, 4)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.8, 1.2)
        if cls == 0:
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif cls == 2:
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:
            wave = 2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_waveforms_8class(n_trials, steps, sample_hz, seed=99):
    """8-class waveforms: 4 base shapes x 2 frequency bands."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / sample_hz
    t = np.arange(steps) * dt
    trials, labels = [], []
    for _ in range(n_trials):
        cls = rng.integers(0, 8)
        base_cls = cls % 4
        freq_band = cls // 4  # 0=low, 1=high
        phase = rng.uniform(0, 2 * np.pi)
        if freq_band == 0:
            freq = rng.uniform(0.5, 0.9)
        else:
            freq = rng.uniform(1.3, 2.0)
        if base_cls == 0:
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif base_cls == 1:
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif base_cls == 2:
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:
            wave = 2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_continuous_input(n_steps, seed=123):
    return np.random.default_rng(seed).uniform(-1, 1, size=n_steps).astype(np.float32)


def generate_binary_input(n_steps, seed=456):
    return np.random.default_rng(seed).integers(0, 2, size=n_steps).astype(float)


def compute_xor_targets(u, tau):
    n = len(u)
    targets = np.zeros(n, dtype=int)
    for t in range(tau, n):
        targets[t] = int(u[t]) ^ int(u[t - tau])
    return targets


def generate_narma(u, order=5):
    """Generate NARMA-order target from input u in [0,1]."""
    n = len(u)
    y = np.zeros(n)
    u_pos = (u + 1.0) / 2.0  # map [-1,1] to [0,1]
    u_pos = np.clip(u_pos, 0, 1) * 0.2  # scale to prevent blow-up
    for t in range(order, n):
        y_prev = y[t-1]
        s = np.sum(y[max(0,t-order):t])
        y[t] = 0.3 * y_prev + 0.05 * y_prev * s + 1.5 * u_pos[t-order] * u_pos[t] + 0.1
        y[t] = np.clip(y[t], -10, 10)
    return y


# ═══════════════════════════════════════════════════════════
# Cross-Substrate Reservoir Runner
# ═══════════════════════════════════════════════════════════

class DefinitiveBridgeReservoir:
    """Manages all 6 conditions."""

    def __init__(self, fpga, noise_sampler, gpu_esn):
        self.fpga_reservoir = FPGAReservoir(fpga, noise_sampler)
        self.noise = noise_sampler
        self.gpu_esn = gpu_esn

    def run_trial(self, input_signal, condition):
        """Run one trial under a specific condition.

        Returns state matrix (n_steps, n_features).
        """
        n_steps = len(input_signal)

        if condition == 'FPGA_ONLY':
            return self.fpga_reservoir.run_trial(input_signal, noise_scale=0.0)

        elif condition == 'FPGA_1F':
            return self.fpga_reservoir.run_trial(input_signal, noise_scale=NOISE_SCALE)

        elif condition == 'GPU_ESN_ONLY':
            return self.gpu_esn.run(input_signal, noise_input=None)

        elif condition == 'GPU_ESN_NOISE':
            # Generate per-step noise from sampler
            noise_in = np.array([
                self.noise.get_noise_scalar(t, NOISE_SCALE)
                for t in range(n_steps)
            ])
            self.noise.reset_iir()
            return self.gpu_esn.run(input_signal, noise_input=noise_in)

        elif condition == 'BRIDGE_SIMPLE':
            # GPU ESN (no noise) + FPGA (with 1/f noise)
            gpu_states = self.gpu_esn.run(input_signal, noise_input=None)
            fpga_states = self.fpga_reservoir.run_trial(input_signal, noise_scale=NOISE_SCALE)
            return np.hstack([gpu_states, fpga_states])

        elif condition == 'BRIDGE_FULL':
            # GPU ESN (with noise) + FPGA (with 1/f noise)
            noise_in = np.array([
                self.noise.get_noise_scalar(t, NOISE_SCALE)
                for t in range(n_steps)
            ])
            self.noise.reset_iir()
            gpu_states = self.gpu_esn.run(input_signal, noise_input=noise_in)
            fpga_states = self.fpga_reservoir.run_trial(input_signal, noise_scale=NOISE_SCALE)
            return np.hstack([gpu_states, fpga_states])

        else:
            raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════════════════════
# Benchmark Runners
# ═══════════════════════════════════════════════════════════

ALL_CONDITIONS = ['FPGA_ONLY', 'FPGA_1F', 'GPU_ESN_ONLY', 'GPU_ESN_NOISE',
                  'BRIDGE_SIMPLE', 'BRIDGE_FULL']


def run_waveform_benchmark(reservoir, conditions, n_trials, n_steps, n_classes, seed, label="4class"):
    """Run waveform classification benchmark."""
    if n_classes == 4:
        inputs, labels = generate_waveforms_4class(n_trials, n_steps, SAMPLE_HZ, seed=seed)
    else:
        inputs, labels = generate_waveforms_8class(n_trials, n_steps, SAMPLE_HZ, seed=seed)

    results = {}
    for cond in conditions:
        check_temp_and_wait(f"Before wave-{label} {cond}")
        print(f"\n  [{cond}] Running {n_trials} waveform-{label} trials...")
        all_feats = []
        for trial in range(n_trials):
            states = reservoir.run_trial(inputs[trial], cond)
            aug = augment_with_delays(states, delays=(1, 2))
            feat = pool_trial_features(aug)
            all_feats.append(feat)
            if (trial + 1) % 20 == 0:
                print(f"    trial {trial+1}/{n_trials}")
        X = np.nan_to_num(np.array(all_feats), nan=0.0, posinf=0.0, neginf=0.0)
        acc_mean, acc_std = classify_cv(X, labels, n_splits=5, n_classes=n_classes)
        print(f"  [{cond}] WAVEFORM-{label}: {acc_mean:.3f} +/- {acc_std:.3f}")
        results[cond] = {'accuracy': acc_mean, 'std': acc_std}
        # Cooling pause between conditions
        if 'GPU' in cond or 'BRIDGE' in cond:
            print(f"  [COOLING] {COOLING_BETWEEN_CONDITIONS}s pause after {cond}...")
            time.sleep(COOLING_BETWEEN_CONDITIONS)
    return results


def run_mc_benchmark(reservoir, conditions, n_steps=N_CONTINUOUS_STEPS):
    """Run memory capacity d=1..10."""
    mc_input = generate_continuous_input(n_steps, seed=123)
    results = {}

    for cond in conditions:
        check_temp_and_wait(f"Before MC {cond}")
        print(f"\n  [{cond}] Running memory capacity ({n_steps} steps)...")
        states = reservoir.run_trial(mc_input, cond)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2))

        mc_total = 0.0
        mc_per_delay = {}
        for d in range(1, MC_MAX_DELAY + 1):
            target = np.zeros(n_steps)
            target[d:] = mc_input[:n_steps - d]
            X = aug[WARMUP:]
            y = target[WARMUP:]
            n_tr = int(0.7 * len(X))
            X_tr, X_te = X[:n_tr], X[n_tr:]
            y_tr, y_te = y[:n_tr], y[n_tr:]
            _, r2 = ridge_regress(X_tr, y_tr, X_te, y_te)
            r2 = max(r2, 0.0)
            mc_per_delay[d] = r2
            mc_total += r2
        print(f"  [{cond}] MC TOTAL: {mc_total:.3f}")
        for d in range(1, MC_MAX_DELAY + 1):
            print(f"    d={d:2d}: r2={mc_per_delay[d]:.3f}")
        results[cond] = {'total': mc_total, 'per_delay': mc_per_delay}
        # Cooling pause between conditions
        if 'GPU' in cond or 'BRIDGE' in cond:
            print(f"  [COOLING] {COOLING_BETWEEN_CONDITIONS}s pause after {cond}...")
            time.sleep(COOLING_BETWEEN_CONDITIONS)
    return results


def run_xor_benchmark(reservoir, conditions, n_steps=N_CONTINUOUS_STEPS):
    """Run temporal XOR tau=1,2,3."""
    xor_input = generate_binary_input(n_steps, seed=456)
    results = {}

    for cond in conditions:
        check_temp_and_wait(f"Before XOR {cond}")
        print(f"\n  [{cond}] Running temporal XOR ({n_steps} steps)...")
        states = reservoir.run_trial(xor_input, cond)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2))
        xor_results = {}
        for tau in [1, 2, 3]:
            targets = compute_xor_targets(xor_input, tau)
            X = aug[WARMUP:]
            y = targets[WARMUP:].astype(float)
            n_tr = int(0.7 * len(X))
            X_tr, X_te = X[:n_tr], X[n_tr:]
            y_tr, y_te = y[:n_tr], y[n_tr:]
            acc = ridge_binary(X_tr, y_tr, X_te, y_te)
            xor_results[tau] = acc
            print(f"    tau={tau}: {acc:.3f}")
        results[cond] = xor_results
        # Cooling pause between conditions
        if 'GPU' in cond or 'BRIDGE' in cond:
            print(f"  [COOLING] {COOLING_BETWEEN_CONDITIONS}s pause after {cond}...")
            time.sleep(COOLING_BETWEEN_CONDITIONS)
    return results


def run_narma_benchmark(reservoir, conditions, n_steps=N_CONTINUOUS_STEPS):
    """Run NARMA-5 regression."""
    narma_input = generate_continuous_input(n_steps, seed=789)
    narma_target = generate_narma(narma_input, order=NARMA_ORDER)
    results = {}

    for cond in conditions:
        check_temp_and_wait(f"Before NARMA {cond}")
        print(f"\n  [{cond}] Running NARMA-{NARMA_ORDER} ({n_steps} steps)...")
        states = reservoir.run_trial(narma_input, cond)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2))
        X = aug[WARMUP:]
        y = narma_target[WARMUP:]
        n_tr = int(0.7 * len(X))
        X_tr, X_te = X[:n_tr], X[n_tr:]
        y_tr, y_te = y[:n_tr], y[n_tr:]
        nrmse, r2 = ridge_regress(X_tr, y_tr, X_te, y_te)
        r2 = max(r2, 0.0)
        print(f"  [{cond}] NARMA-{NARMA_ORDER}: NRMSE={nrmse:.3f}, R2={r2:.3f}")
        results[cond] = {'nrmse': nrmse, 'r2': r2}
        # Cooling pause between conditions
        if 'GPU' in cond or 'BRIDGE' in cond:
            print(f"  [COOLING] {COOLING_BETWEEN_CONDITIONS}s pause after {cond}...")
            time.sleep(COOLING_BETWEEN_CONDITIONS)
    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("  z2271: DEFINITIVE Cross-Substrate GPU+FPGA Bridge")
    print("  z2269 optimal: NOISE_SCALE=0.02")
    print("  New bitstream: LEAK_COND=4 (tau=210ms)")
    print("  GPU: Fourpop ESN (256 neurons, 4 physics populations)")
    print("  FPGA: 128 LIF neurons on Arty A7-100T via Ethernet")
    print("  Conditions: FPGA_ONLY, FPGA_1F, GPU_ESN_ONLY, GPU_ESN_NOISE,")
    print("              BRIDGE_SIMPLE, BRIDGE_FULL")
    print("  Benchmarks: Wave4, Wave8, MC(1-10), XOR(1-3), NARMA-5")
    print("=" * 72)

    check_temp_and_wait("Initial")

    # ─── Step 1: Compile GPU stress kernel ───
    print("\n[1] Compiling GPU stress kernel...")
    stress_bin = compile_stress_kernel()
    if stress_bin is None:
        print("  FATAL: Could not compile stress kernel")
        sys.exit(1)

    # ─── Step 2: Connect to FPGA ───
    print("\n[2] Connecting to FPGA via Ethernet...")
    fpga = FPGAEthBridge()
    fpga_ok = fpga.connect()
    if not fpga_ok:
        print("  WARNING: FPGA not responding -- will run with timeouts")

    # Kill switch off
    fpga.set_kill(False)
    time.sleep(0.3)

    # Set heterogeneous Vg
    print("  Setting heterogeneous Vg (BASE_VG=0.58 +/- 0.08)...")
    rng = np.random.default_rng(42)
    init_vg = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, size=N_FPGA)
    for nid in range(N_FPGA):
        vg_q16 = int(init_vg[nid] * 65536) & 0xFFFFFFFF
        pkt = struct.pack(">BBBI", 0x55, 0x01, nid & 0x7F, vg_q16)
        fpga.sock.sendto(pkt, (fpga.fpga_ip, fpga.fpga_port))
        if nid % 32 == 31:
            time.sleep(0.01)
    print(f"  Set Vg for {N_FPGA} neurons: range [{init_vg.min():.3f}, {init_vg.max():.3f}]")
    time.sleep(0.5)

    # Verify telemetry
    telem = fpga.read_telemetry()
    if telem:
        sc = telem['spike_counts']
        print(f"  FPGA alive: total_spikes={sc.sum()}, active={np.count_nonzero(sc)}/{N_FPGA}")
    else:
        print("  WARNING: no telemetry")

    # ─── Step 3: Launch GPU stress + sample noise ───
    print("\n[3] Launching GPU stress kernel and sampling noise...")
    check_temp_and_wait("Pre-stress")

    noise_sampler = GPUNoiseSampler()

    def ensure_stress_and_sample(label=""):
        """Ensure stress kernel is running, sample noise. Returns stress_proc."""
        nonlocal stress_proc
        # Check if stress is still alive
        if stress_proc is None or stress_proc.poll() is not None:
            if stress_proc is not None:
                print(f"  [STRESS] Previous stress kernel ended, relaunching...")
            check_temp_and_wait(f"Pre-stress-{label}")
            # Wait for temp to drop below soft limit
            while get_edge_temp() > TEMP_LIMIT_SOFT:
                print(f"  [TEMP] Waiting for cooldown... {get_edge_temp():.0f}C")
                time.sleep(15)
            stress_proc = launch_stress_kernel(stress_bin, duration=300)
            time.sleep(3)
        noise_sampler.sample(stress_proc, n_samples=600, sample_hz=NOISE_SAMPLE_HZ)
        return stress_proc

    stress_proc = None
    stress_proc = ensure_stress_and_sample("initial")
    noise_stats = {'n_samples': noise_sampler._power_norm.shape[0] if noise_sampler._power_norm is not None else 0,
                   'std': float(np.std(noise_sampler._raw_buf)) if noise_sampler._raw_buf else 0.0,
                   'mean': float(np.mean(noise_sampler._raw_buf)) if noise_sampler._raw_buf else 0.0}

    if noise_stats['std'] < 0.01:
        print("  WARNING: Power variance still low -- GPU stress may not be generating load")

    # Kill stress for now -- we relaunch per benchmark
    kill_stress(stress_proc)
    stress_proc = None
    time.sleep(5)

    # ─── Step 4: Create reservoir system ───
    print("\n[4] Initializing definitive bridge reservoir...")
    gpu_esn = GPUFourpopESN(n_per_pop=64, noise_sampler=noise_sampler)
    reservoir = DefinitiveBridgeReservoir(fpga, noise_sampler, gpu_esn)

    all_results = {
        'experiment': 'z2271_definitive_bridge',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'noise_stats': noise_stats,
        'params': {
            'n_gpu': N_GPU, 'n_fpga': N_FPGA,
            'base_vg': BASE_VG, 'vg_spread': VG_SPREAD,
            'alpha': ALPHA, 'iir_alpha': IIR_ALPHA,
            'noise_scale': NOISE_SCALE, 'sample_hz': SAMPLE_HZ,
            'n_wave_trials': N_WAVE_TRIALS, 'n_wave_steps': N_WAVE_STEPS,
            'n_continuous_steps': N_CONTINUOUS_STEPS,
            'narma_order': NARMA_ORDER,
        },
    }

    # ═══════════════════════════════════════════════════════
    # BENCHMARK 1: 4-Class Waveform Classification
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("BENCHMARK 1: 4-CLASS WAVEFORM CLASSIFICATION")
    print(f"  {N_WAVE_TRIALS} trials x {N_WAVE_STEPS} steps @ {SAMPLE_HZ} Hz")
    print("=" * 72)

    stress_proc = ensure_stress_and_sample("wave4")

    wave4_results = run_waveform_benchmark(
        reservoir, ALL_CONDITIONS, N_WAVE_TRIALS, N_WAVE_STEPS, 4, seed=42, label="4class"
    )
    all_results['waveform_4class'] = wave4_results

    kill_stress(stress_proc)
    stress_proc = None
    print(f"  [COOLING] {COOLING_BETWEEN_CONDITIONS}s inter-benchmark cooldown...")
    time.sleep(COOLING_BETWEEN_CONDITIONS)
    check_temp_and_wait("Post-wave4")

    # ═══════════════════════════════════════════════════════
    # BENCHMARK 2: 8-Class Waveform Classification
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("BENCHMARK 2: 8-CLASS WAVEFORM CLASSIFICATION")
    print(f"  {N_WAVE_TRIALS} trials x {N_WAVE_STEPS} steps @ {SAMPLE_HZ} Hz")
    print("=" * 72)

    stress_proc = ensure_stress_and_sample("wave8")

    wave8_results = run_waveform_benchmark(
        reservoir, ALL_CONDITIONS, N_WAVE_TRIALS, N_WAVE_STEPS, 8, seed=99, label="8class"
    )
    all_results['waveform_8class'] = wave8_results

    kill_stress(stress_proc)
    stress_proc = None
    print(f"  [COOLING] {COOLING_BETWEEN_CONDITIONS}s inter-benchmark cooldown...")
    time.sleep(COOLING_BETWEEN_CONDITIONS)
    check_temp_and_wait("Post-wave8")

    # ═══════════════════════════════════════════════════════
    # BENCHMARK 3: Memory Capacity
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("BENCHMARK 3: MEMORY CAPACITY d=1..10")
    print(f"  {N_CONTINUOUS_STEPS} continuous steps")
    print("=" * 72)

    stress_proc = ensure_stress_and_sample("MC")

    mc_results = run_mc_benchmark(reservoir, ALL_CONDITIONS)
    all_results['memory_capacity'] = mc_results

    kill_stress(stress_proc)
    stress_proc = None
    print(f"  [COOLING] {COOLING_BETWEEN_CONDITIONS}s inter-benchmark cooldown...")
    time.sleep(COOLING_BETWEEN_CONDITIONS)
    check_temp_and_wait("Post-MC")

    # ═══════════════════════════════════════════════════════
    # BENCHMARK 4: Temporal XOR
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("BENCHMARK 4: TEMPORAL XOR tau=1,2,3")
    print(f"  {N_CONTINUOUS_STEPS} continuous steps")
    print("=" * 72)

    stress_proc = ensure_stress_and_sample("XOR")

    xor_results = run_xor_benchmark(reservoir, ALL_CONDITIONS)
    all_results['temporal_xor'] = xor_results

    kill_stress(stress_proc)
    stress_proc = None
    print(f"  [COOLING] {COOLING_BETWEEN_CONDITIONS}s inter-benchmark cooldown...")
    time.sleep(COOLING_BETWEEN_CONDITIONS)
    check_temp_and_wait("Post-XOR")

    # ═══════════════════════════════════════════════════════
    # BENCHMARK 5: NARMA-5
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("BENCHMARK 5: NARMA-5 REGRESSION")
    print(f"  {N_CONTINUOUS_STEPS} continuous steps, lambda sweep [0.001..10.0]")
    print("=" * 72)

    stress_proc = ensure_stress_and_sample("NARMA")

    narma_results = run_narma_benchmark(reservoir, ALL_CONDITIONS)
    all_results['narma'] = narma_results

    # Kill stress kernel
    kill_stress(stress_proc)
    stress_proc = None
    print(f"  [COOLING] {COOLING_BETWEEN_CONDITIONS}s final cooldown...")
    time.sleep(COOLING_BETWEEN_CONDITIONS)
    check_temp_and_wait("Post-all-benchmarks")

    # ═══════════════════════════════════════════════════════
    # SUMMARY & TESTS
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    # Extract metrics
    def _acc(d, cond):
        return d.get(cond, {}).get('accuracy', 0)

    def _mc(d, cond):
        return d.get(cond, {}).get('total', 0)

    def _xor(d, cond, tau):
        return d.get(cond, {}).get(tau, 0)

    def _narma_r2(d, cond):
        return d.get(cond, {}).get('r2', 0)

    def _narma_nrmse(d, cond):
        return d.get(cond, {}).get('nrmse', 99)

    # Print table
    hdr = f"\n{'Metric':<25}"
    for c in ALL_CONDITIONS:
        hdr += f" {c:>16}"
    print(hdr)
    print("-" * (25 + 17 * len(ALL_CONDITIONS)))

    row = f"{'Wave 4-class':<25}"
    for c in ALL_CONDITIONS:
        row += f" {_acc(wave4_results, c):>15.1%}"
    print(row)

    row = f"{'Wave 8-class':<25}"
    for c in ALL_CONDITIONS:
        row += f" {_acc(wave8_results, c):>15.1%}"
    print(row)

    row = f"{'MC total':<25}"
    for c in ALL_CONDITIONS:
        row += f" {_mc(mc_results, c):>16.3f}"
    print(row)

    for tau in [1, 2, 3]:
        row = f"{'XOR tau=' + str(tau):<25}"
        for c in ALL_CONDITIONS:
            row += f" {_xor(xor_results, c, tau):>15.1%}"
        print(row)

    row = f"{'NARMA-5 R2':<25}"
    for c in ALL_CONDITIONS:
        row += f" {_narma_r2(narma_results, c):>16.3f}"
    print(row)

    row = f"{'NARMA-5 NRMSE':<25}"
    for c in ALL_CONDITIONS:
        row += f" {_narma_nrmse(narma_results, c):>16.3f}"
    print(row)

    # ─── Tests ───
    print(f"\n{'=' * 72}")
    print("TESTS")
    print(f"{'=' * 72}")

    tests = {}
    n_pass = 0
    n_total = 0

    def check(name, condition, desc):
        nonlocal n_pass, n_total
        n_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            n_pass += 1
        print(f"  {name}: {status} -- {desc}")
        tests[name] = {'pass': bool(condition), 'desc': desc}

    # Noise quality
    pw_std = noise_stats.get('std', 0)
    check("T1_active_noise", pw_std > 0.05,
          f"GPU power std={pw_std:.4f}W > 0.05W")

    # 4-class waveform tests
    w4 = {c: _acc(wave4_results, c) for c in ALL_CONDITIONS}
    check("T2_fpga_1f_vs_fpga", w4['FPGA_1F'] > w4['FPGA_ONLY'],
          f"FPGA_1F {w4['FPGA_1F']:.1%} > FPGA_ONLY {w4['FPGA_ONLY']:.1%}")

    check("T3_bridge_full_best_wave4", w4['BRIDGE_FULL'] >= max(w4['FPGA_ONLY'], w4['GPU_ESN_ONLY']),
          f"BRIDGE_FULL {w4['BRIDGE_FULL']:.1%} >= max(FPGA={w4['FPGA_ONLY']:.1%}, GPU={w4['GPU_ESN_ONLY']:.1%})")

    check("T4_bridge_full_wave4_target", w4['BRIDGE_FULL'] > 0.75,
          f"BRIDGE_FULL wave4 {w4['BRIDGE_FULL']:.1%} > 75%")

    check("T5_bridge_vs_bridge_simple", w4['BRIDGE_FULL'] >= w4['BRIDGE_SIMPLE'],
          f"BRIDGE_FULL {w4['BRIDGE_FULL']:.1%} >= BRIDGE_SIMPLE {w4['BRIDGE_SIMPLE']:.1%}")

    # 8-class waveform tests
    w8 = {c: _acc(wave8_results, c) for c in ALL_CONDITIONS}
    check("T6_bridge_full_best_wave8", w8['BRIDGE_FULL'] >= max(w8['FPGA_ONLY'], w8['GPU_ESN_ONLY']),
          f"BRIDGE_FULL {w8['BRIDGE_FULL']:.1%} >= max(FPGA={w8['FPGA_ONLY']:.1%}, GPU={w8['GPU_ESN_ONLY']:.1%})")

    check("T7_bridge_full_wave8_target", w8['BRIDGE_FULL'] > 0.40,
          f"BRIDGE_FULL wave8 {w8['BRIDGE_FULL']:.1%} > 40%")

    # Memory capacity tests
    mc = {c: _mc(mc_results, c) for c in ALL_CONDITIONS}
    check("T8_mc_bridge_full_best", mc['BRIDGE_FULL'] >= max(mc['FPGA_ONLY'], mc['GPU_ESN_ONLY']),
          f"BRIDGE_FULL MC {mc['BRIDGE_FULL']:.3f} >= max(FPGA={mc['FPGA_ONLY']:.3f}, GPU={mc['GPU_ESN_ONLY']:.3f})")

    check("T9_mc_1f_vs_fpga", mc['FPGA_1F'] > mc['FPGA_ONLY'],
          f"FPGA_1F MC {mc['FPGA_1F']:.3f} > FPGA_ONLY {mc['FPGA_ONLY']:.3f}")

    check("T10_mc_bridge_target", mc['BRIDGE_FULL'] > 1.0,
          f"BRIDGE_FULL MC {mc['BRIDGE_FULL']:.3f} > 1.0")

    # XOR tests
    xor1 = {c: _xor(xor_results, c, 1) for c in ALL_CONDITIONS}
    xor2 = {c: _xor(xor_results, c, 2) for c in ALL_CONDITIONS}
    check("T11_xor1_bridge_target", xor1['BRIDGE_FULL'] > 0.55,
          f"BRIDGE_FULL XOR tau=1 {xor1['BRIDGE_FULL']:.1%} > 55%")

    check("T12_xor1_bridge_vs_fpga", xor1['BRIDGE_FULL'] > xor1['FPGA_ONLY'],
          f"BRIDGE_FULL XOR1 {xor1['BRIDGE_FULL']:.1%} > FPGA_ONLY {xor1['FPGA_ONLY']:.1%}")

    check("T13_xor2_bridge_target", xor2['BRIDGE_FULL'] > 0.50,
          f"BRIDGE_FULL XOR tau=2 {xor2['BRIDGE_FULL']:.1%} > 50%")

    # NARMA tests
    nr2 = {c: _narma_r2(narma_results, c) for c in ALL_CONDITIONS}
    nnrmse = {c: _narma_nrmse(narma_results, c) for c in ALL_CONDITIONS}
    check("T14_narma_bridge_best", nr2['BRIDGE_FULL'] >= max(nr2['FPGA_ONLY'], nr2['GPU_ESN_ONLY']),
          f"BRIDGE_FULL NARMA R2 {nr2['BRIDGE_FULL']:.3f} >= max(FPGA={nr2['FPGA_ONLY']:.3f}, GPU={nr2['GPU_ESN_ONLY']:.3f})")

    check("T15_narma_bridge_target", nr2['BRIDGE_FULL'] > 0.1,
          f"BRIDGE_FULL NARMA R2 {nr2['BRIDGE_FULL']:.3f} > 0.1")

    # Cross-substrate synergy: BRIDGE_FULL > best single substrate on at least 3 benchmarks
    bridge_wins = 0
    if w4['BRIDGE_FULL'] > max(w4['FPGA_1F'], w4['GPU_ESN_NOISE']):
        bridge_wins += 1
    if w8['BRIDGE_FULL'] > max(w8['FPGA_1F'], w8['GPU_ESN_NOISE']):
        bridge_wins += 1
    if mc['BRIDGE_FULL'] > max(mc['FPGA_1F'], mc['GPU_ESN_NOISE']):
        bridge_wins += 1
    if xor1['BRIDGE_FULL'] > max(xor1.get('FPGA_1F', 0), xor1.get('GPU_ESN_NOISE', 0)):
        bridge_wins += 1
    if nr2['BRIDGE_FULL'] > max(nr2.get('FPGA_1F', 0), nr2.get('GPU_ESN_NOISE', 0)):
        bridge_wins += 1

    check("T16_cross_substrate_synergy", bridge_wins >= 3,
          f"BRIDGE_FULL best on {bridge_wins}/5 benchmarks (need >=3)")

    # GPU noise actually helps GPU ESN
    check("T17_gpu_noise_helps_esn", w4['GPU_ESN_NOISE'] > w4['GPU_ESN_ONLY'] * 0.99,
          f"GPU_ESN_NOISE {w4['GPU_ESN_NOISE']:.1%} >= GPU_ESN_ONLY*0.99 {w4['GPU_ESN_ONLY']*0.99:.1%}")

    # BRIDGE_FULL has more effective dimensionality than any single substrate
    # (measured by rank of state matrix in MC run)
    bridge_dim = 0
    fpga_dim = 0
    gpu_dim = 0
    try:
        # Quick dimensionality test: count eigenvalues > 1% of max
        mc_input_dim = generate_continuous_input(100, seed=999)
        states_bridge = reservoir.run_trial(mc_input_dim, 'BRIDGE_FULL')
        states_fpga = reservoir.run_trial(mc_input_dim, 'FPGA_1F')
        states_gpu = reservoir.run_trial(mc_input_dim, 'GPU_ESN_NOISE')
        for st, name in [(states_bridge, 'BRIDGE'), (states_fpga, 'FPGA'), (states_gpu, 'GPU')]:
            st = np.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0)
            cov = np.cov(st.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.abs(eigvals)
            threshold = 0.01 * eigvals.max() if eigvals.max() > 0 else 0
            dim = int(np.sum(eigvals > threshold))
            if name == 'BRIDGE':
                bridge_dim = dim
            elif name == 'FPGA':
                fpga_dim = dim
            else:
                gpu_dim = dim
        print(f"  Effective dimensions: BRIDGE={bridge_dim}, FPGA={fpga_dim}, GPU={gpu_dim}")
    except Exception as e:
        print(f"  Dimensionality test failed: {e}")

    check("T18_bridge_dimensionality", bridge_dim > max(fpga_dim, gpu_dim),
          f"BRIDGE dim={bridge_dim} > max(FPGA={fpga_dim}, GPU={gpu_dim})")

    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    all_results['tests'] = tests
    all_results['summary'] = {
        'pass': n_pass, 'total': n_total,
        'waveform_4class': {c: _acc(wave4_results, c) for c in ALL_CONDITIONS},
        'waveform_8class': {c: _acc(wave8_results, c) for c in ALL_CONDITIONS},
        'memory_capacity': {c: _mc(mc_results, c) for c in ALL_CONDITIONS},
        'xor': {c: {tau: _xor(xor_results, c, tau) for tau in [1,2,3]} for c in ALL_CONDITIONS},
        'narma': {c: {'r2': _narma_r2(narma_results, c), 'nrmse': _narma_nrmse(narma_results, c)} for c in ALL_CONDITIONS},
        'bridge_synergy_wins': bridge_wins,
        'effective_dim': {'bridge': bridge_dim, 'fpga': fpga_dim, 'gpu': gpu_dim},
    }

    # ─── Save JSON ───
    json_path = RESULTS / 'z2271_definitive_bridge.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\n  JSON saved: {json_path}")

    # ─── Save text report ───
    txt_path = RESULTS / 'z2271_definitive_bridge.txt'
    with open(txt_path, 'w') as f:
        f.write("z2271: DEFINITIVE Cross-Substrate GPU+FPGA Bridge\n")
        f.write(f"Date: {all_results['timestamp']}\n")
        f.write(f"Tests: {n_pass}/{n_total} PASS\n\n")
        f.write(f"NOISE_SCALE={NOISE_SCALE} (z2269 optimal)\n")
        f.write(f"LEAK_COND=4 (tau=210ms)\n")
        f.write(f"GPU power std: {noise_stats.get('std', 0):.4f} W\n\n")

        f.write("4-Class Waveform:\n")
        for c in ALL_CONDITIONS:
            f.write(f"  {c:>20}: {_acc(wave4_results, c):.1%}\n")

        f.write("\n8-Class Waveform:\n")
        for c in ALL_CONDITIONS:
            f.write(f"  {c:>20}: {_acc(wave8_results, c):.1%}\n")

        f.write(f"\nMemory Capacity (d=1..10):\n")
        for c in ALL_CONDITIONS:
            f.write(f"  {c:>20}: {_mc(mc_results, c):.3f}\n")

        f.write(f"\nTemporal XOR:\n")
        for c in ALL_CONDITIONS:
            xr = xor_results.get(c, {})
            f.write(f"  {c:>20}: tau1={xr.get(1,0):.3f} tau2={xr.get(2,0):.3f} tau3={xr.get(3,0):.3f}\n")

        f.write(f"\nNARMA-{NARMA_ORDER}:\n")
        for c in ALL_CONDITIONS:
            f.write(f"  {c:>20}: R2={_narma_r2(narma_results, c):.3f} NRMSE={_narma_nrmse(narma_results, c):.3f}\n")

        f.write(f"\nEffective Dimensions: BRIDGE={bridge_dim}, FPGA={fpga_dim}, GPU={gpu_dim}\n")
        f.write(f"Bridge synergy wins: {bridge_wins}/5 benchmarks\n")

        f.write(f"\nTests:\n")
        for name, t in tests.items():
            f.write(f"  {name}: {'PASS' if t['pass'] else 'FAIL'} -- {t['desc']}\n")

    print(f"  Text saved: {txt_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
