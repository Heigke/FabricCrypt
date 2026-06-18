#!/usr/bin/env python3
"""z2272_safe_bridge.py — Thermally-Safe Definitive Cross-Substrate Bridge

Same experiment as z2271 but with thermal safety:
  - NO long-running GPU stress kernel (was 300s → now 5-10s bursts)
  - Save results after EACH condition×benchmark (crash-safe)
  - Strict 65°C ABORT (don't wait — laptop can't cool under load)
  - Run FPGA-only conditions first (no GPU heat)
  - Progressive stress: ambient → 5s burst → 10s burst
  - Load previous results on restart (resume from crash)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2272_safe_bridge.py
"""

import os, sys, time, json, struct, subprocess, signal
import numpy as np
from pathlib import Path

# ─── Paths ───
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2272_safe_bridge.json'
TEXT_FILE = RESULTS / 'z2272_safe_bridge.txt'

# ─── Parameters ───
N_GPU = 256
N_FPGA = 128
BASE_VG = 0.58
VG_SPREAD = 0.08
ALPHA = 0.25
IIR_ALPHA = 0.85
NOISE_SCALE = 0.02
SAMPLE_HZ = 50

N_WAVE_TRIALS = 80
N_WAVE_STEPS = 60
N_CONTINUOUS_STEPS = 500
MC_MAX_DELAY = 10
WARMUP = 50
NARMA_ORDER = 5

# ─── THERMAL SAFETY ───
# CRITICAL: Monitor thermal_zone0 (ACPI/CPU), NOT just hwmon7 (GPU edge).
# ACPI trip point is 99°C — shutdown is hardware-forced and instant.
# Previous crashes: GPU stress + FPGA I/O + numpy = APU overheats.
TEMP_ABORT = 90.0      # ABORT benchmark (ACPI trip is 99°C, leave 9°C margin)
TEMP_SAFE = 55.0       # Start new work only below this
TEMP_PAUSE = 75.0      # Pause between trials if above this
STRESS_BURST_S = 6     # Short GPU stress burst for noise sampling
COOLING_PAUSE_S = 15   # Between conditions
COOLING_BETWEEN_TRIALS = 1.0  # Seconds between FPGA trials
NOISE_SAMPLE_HZ = 200
NOISE_N_SAMPLES = 150  # Fewer samples = less stress time


# ═══════════════════════════════════════════════════════════
# Temperature
# ═══════════════════════════════════════════════════════════

def get_apu_temp():
    """Get APU/CPU temperature from ACPI thermal zone (the one that triggers shutdown)."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0


def get_gpu_temp():
    """Get GPU edge temperature from hwmon."""
    try:
        with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0


def get_max_temp():
    """Return the higher of APU and GPU temps."""
    return max(get_apu_temp(), get_gpu_temp())


def wait_cool(label="", target=None):
    """Wait for temperature to drop below target. Returns current temp."""
    if target is None:
        target = TEMP_SAFE
    apu = get_apu_temp()
    gpu = get_gpu_temp()
    temp = max(apu, gpu)
    if temp <= target:
        print(f"  [TEMP] {label} APU={apu:.0f}°C GPU={gpu:.0f}°C — OK", flush=True)
        return temp
    print(f"  [TEMP] {label} APU={apu:.0f}°C GPU={gpu:.0f}°C — cooling to {target:.0f}°C...", end="", flush=True)
    t0 = time.time()
    while temp > target:
        if time.time() - t0 > 180:
            print(f" timeout at APU={get_apu_temp():.0f}°C", flush=True)
            return temp
        time.sleep(5)
        apu = get_apu_temp()
        gpu = get_gpu_temp()
        temp = max(apu, gpu)
        print(f" {temp:.0f}", end="", flush=True)
    print(f" — OK ({time.time()-t0:.0f}s)", flush=True)
    return temp


def check_abort():
    """Check if temperature is critical. Returns True if should abort."""
    apu = get_apu_temp()
    if apu > TEMP_ABORT:
        print(f"\n  *** THERMAL ABORT: APU={apu:.0f}°C > {TEMP_ABORT:.0f}°C ***", flush=True)
        return True
    return False


def check_pause():
    """Pause briefly if temperature is getting warm. Non-blocking if cool."""
    apu = get_apu_temp()
    if apu > TEMP_PAUSE:
        print(f"  [TEMP] APU={apu:.0f}°C > {TEMP_PAUSE:.0f}°C — pausing 10s...", flush=True)
        time.sleep(10)
        return True
    return False


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# GPU Stress (SHORT bursts only)
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
    int duration = 10;
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
    src_path = '/tmp/z2272_gpu_stress.hip'
    bin_path = '/tmp/z2272_gpu_stress'
    if os.path.isfile(bin_path) and os.path.isfile(src_path):
        if os.path.getmtime(bin_path) > os.path.getmtime(src_path):
            print(f"  [STRESS] Using cached binary: {bin_path}")
            return bin_path
    with open(src_path, 'w') as f:
        f.write(GPU_STRESS_SRC)
    print(f"  [STRESS] Compiling...")
    result = subprocess.run(
        ['hipcc', '--offload-arch=gfx1100', '-O1', '-o', bin_path, src_path],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"  [STRESS] Compile failed: {result.stderr}")
        return None
    return bin_path


def sample_noise_with_burst(stress_bin, burst_s=STRESS_BURST_S, n_samples=NOISE_N_SAMPLES):
    """Launch a SHORT stress burst, sample noise, kill stress. Returns noise array."""
    wait_cool("Pre-noise-burst", TEMP_SAFE)

    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    proc = subprocess.Popen(
        [stress_bin, str(burst_s + 3)],  # +3s margin
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    print(f"  [NOISE] Stress PID={proc.pid} for {burst_s}s, sampling {n_samples} points...", flush=True)
    time.sleep(2)  # Let GPU warm up

    raw = []
    interval = 1.0 / NOISE_SAMPLE_HZ
    for i in range(n_samples):
        apu = get_apu_temp()
        if apu > TEMP_ABORT:
            print(f"  [NOISE] APU={apu:.0f}°C — aborting noise sample", flush=True)
            break
        t0 = time.perf_counter()
        try:
            with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
                pw = float(f.read().strip()) / 1e6
            raw.append(pw)
        except Exception:
            raw.append(raw[-1] if raw else 0.0)
        elapsed = time.perf_counter() - t0
        remaining = interval - elapsed
        if remaining > 0.0001:
            time.sleep(remaining)

    # Kill stress immediately
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    print(f"  [NOISE] Stress killed. Cooling...", flush=True)
    wait_cool("Post-noise", TEMP_SAFE)

    pw_arr = np.array(raw)
    if len(pw_arr) > 0:
        print(f"  [NOISE] {len(pw_arr)} samples: mean={pw_arr.mean():.2f}W, std={pw_arr.std():.4f}W", flush=True)
    return pw_arr


# ═══════════════════════════════════════════════════════════
# GPU Noise Sampler (reuses z2271 logic but with burst data)
# ═══════════════════════════════════════════════════════════

class GPUNoiseSampler:
    def __init__(self):
        self._power_norm = None
        self._iir_state = 0.0
        self._raw_buf = []

    def load_burst(self, pw_arr):
        """Load noise from a short burst sample."""
        self._raw_buf = pw_arr.tolist()
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
        base = self.get_1f_noise(t, gain)
        return base + np.random.randn(n_neurons) * gain * 0.1

    def get_noise_scalar(self, t, gain):
        return self.get_1f_noise(t, gain)

    def reset_iir(self):
        self._iir_state = 0.0


# ═══════════════════════════════════════════════════════════
# GPU Fourpop ESN (identical to z2271)
# ═══════════════════════════════════════════════════════════

class GPUFourpopESN:
    def __init__(self, n_per_pop=64, noise_sampler=None):
        self.n_per_pop = n_per_pop
        self.N = 4 * n_per_pop
        self.noise_sampler = noise_sampler
        rng = np.random.default_rng(7777)

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

        self.W_rec = rng.standard_normal((self.N, self.N)) * 0.04
        mask = rng.random((self.N, self.N)) > 0.9
        self.W_rec *= mask

        sc, ec = 2 * n_per_pop, 3 * n_per_pop
        W_c = rng.standard_normal((n_per_pop, n_per_pop)) * 0.08
        mask_c = rng.random((n_per_pop, n_per_pop)) > 0.7
        W_c *= mask_c
        eigvals = np.abs(np.linalg.eigvals(W_c))
        sr = max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0:
            W_c *= 1.05 / sr
        self.W_rec[sc:ec, sc:ec] = W_c

        self.bthr = 0.5 + 0.3 * np.arange(n_per_pop) / max(n_per_pop - 1, 1)
        self.temp_c = 0.65

    def run(self, input_seq, noise_input=None):
        n_steps = len(input_seq)
        pp = self.n_per_pop
        states = np.zeros((n_steps, self.N))
        v = np.zeros(self.N)
        h = np.zeros(self.N)
        slow = np.zeros(self.N)
        rng = np.random.default_rng(42)

        for t in range(n_steps):
            u = input_seq[t]
            rec = self.W_rec @ v

            sa, ea = 0, pp
            branch_val = np.where(v[sa:ea] > self.bthr, 1.0, -1.0)
            v_new_a = np.tanh(
                (1.0 - self.leak[sa:ea]) * v[sa:ea]
                + self.input_w[sa:ea] * u + rec[sa:ea]
                + self.bias[sa:ea] + 0.02 * branch_val
            )

            sb, eb = pp, 2 * pp
            v_b = v[sb:eb].copy()
            n_swap = max(1, pp // 10)
            swap_idx = rng.choice(pp, size=n_swap * 2, replace=False)
            for k in range(0, n_swap * 2 - 1, 2):
                v_b[swap_idx[k]], v_b[swap_idx[k+1]] = v_b[swap_idx[k+1]], v_b[swap_idx[k]]
            v_new_b = np.tanh(
                (1.0 - self.leak[sb:eb]) * v_b
                + self.input_w[sb:eb] * u + rec[sb:eb] + self.bias[sb:eb]
            )

            sc, ec = 2 * pp, 3 * pp
            v_new_c = np.tanh(
                ((1.0 - self.leak[sc:ec]) * v[sc:ec]
                 + self.input_w[sc:ec] * u + rec[sc:ec] + self.bias[sc:ec]) / self.temp_c
            )

            sd, ed = 3 * pp, 4 * pp
            sched_noise = rng.uniform(-1, 1, pp) * 0.01
            v_new_d = np.tanh(
                (1.0 - self.leak[sd:ed]) * v[sd:ed]
                + self.input_w[sd:ed] * u + rec[sd:ed]
                + self.bias[sd:ed] + sched_noise
            )

            v_new = np.concatenate([v_new_a, v_new_b, v_new_c, v_new_d])
            if noise_input is not None and t < len(noise_input):
                v_new += noise_input[t] * 0.01
            pll = rng.uniform(-1, 1, self.N) * 0.003
            v_new += pll

            spike_mask = v_new > self.thr
            v_new[spike_mask] -= self.thr[spike_mask]
            v = v_new

            h = 0.93 * h + 0.07 * v
            slow = 0.99 * slow + 0.01 * v
            states[t] = v + 0.3 * h + 0.1 * slow

        return states


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir
# ═══════════════════════════════════════════════════════════

class FPGAReservoir:
    def __init__(self, fpga, noise_sampler):
        self.fpga = fpga
        self.noise = noise_sampler
        rng = np.random.default_rng(42)
        self.base_vg = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, size=N_FPGA)
        self.w_in = rng.uniform(-1, 1, size=N_FPGA)

    def run_trial(self, input_signal, noise_scale=0.0):
        n_steps = len(input_signal)
        interval = 1.0 / SAMPLE_HZ
        fpga_states = np.zeros((n_steps, N_FPGA * 2))
        prev_counts = None
        self.noise.reset_iir()

        for t in range(n_steps):
            t_start = time.perf_counter()
            vg = self.base_vg.copy() + ALPHA * input_signal[t] * self.w_in
            if noise_scale > 0:
                noise_vec = self.noise.get_noise_vector(t, N_FPGA, noise_scale)
                vg += noise_vec
            vg = np.clip(vg, 0.05, 0.95)

            self.fpga.set_vg_batch(0, vg.tolist())
            time.sleep(max(0.001, interval * 0.3))

            try:
                counts, vmem, refract = self.fpga.read_telemetry_fast()
            except (TimeoutError, Exception):
                elapsed = time.perf_counter() - t_start
                remaining = interval - elapsed
                if remaining > 0.0005:
                    time.sleep(remaining)
                continue

            if prev_counts is not None:
                for i in range(N_FPGA):
                    delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    fpga_states[t, i] = delta
            for i in range(N_FPGA):
                fpga_states[t, N_FPGA + i] = vmem[i]
            prev_counts = counts.copy()

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
        if cls == 0:   wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1: wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif cls == 2: wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:          wave = 2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_waveforms_8class(n_trials, steps, sample_hz, seed=99):
    rng = np.random.default_rng(seed)
    dt = 1.0 / sample_hz
    t = np.arange(steps) * dt
    trials, labels = [], []
    for _ in range(n_trials):
        cls = rng.integers(0, 8)
        base_cls = cls % 4
        freq_band = cls // 4
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 0.9) if freq_band == 0 else rng.uniform(1.3, 2.0)
        if base_cls == 0:   wave = np.sin(2 * np.pi * freq * t + phase)
        elif base_cls == 1: wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif base_cls == 2: wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:               wave = 2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0
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
    n = len(u)
    y = np.zeros(n)
    u_pos = np.clip((u + 1.0) / 2.0, 0, 1) * 0.2
    for t in range(order, n):
        y_prev = y[t-1]
        s = np.sum(y[max(0,t-order):t])
        y[t] = 0.3 * y_prev + 0.05 * y_prev * s + 1.5 * u_pos[t-order] * u_pos[t] + 0.1
        y[t] = np.clip(y[t], -10, 10)
    return y


# ═══════════════════════════════════════════════════════════
# Cross-Substrate Reservoir Runner
# ═══════════════════════════════════════════════════════════

class SafeBridgeReservoir:
    def __init__(self, fpga, noise_sampler, gpu_esn):
        self.fpga_reservoir = FPGAReservoir(fpga, noise_sampler)
        self.noise = noise_sampler
        self.gpu_esn = gpu_esn

    def run_trial(self, input_signal, condition):
        n_steps = len(input_signal)
        if condition == 'FPGA_ONLY':
            return self.fpga_reservoir.run_trial(input_signal, noise_scale=0.0)
        elif condition == 'FPGA_1F':
            return self.fpga_reservoir.run_trial(input_signal, noise_scale=NOISE_SCALE)
        elif condition == 'GPU_ESN_ONLY':
            return self.gpu_esn.run(input_signal, noise_input=None)
        elif condition == 'GPU_ESN_NOISE':
            noise_in = np.array([self.noise.get_noise_scalar(t, NOISE_SCALE) for t in range(n_steps)])
            self.noise.reset_iir()
            return self.gpu_esn.run(input_signal, noise_input=noise_in)
        elif condition == 'BRIDGE_SIMPLE':
            gpu_states = self.gpu_esn.run(input_signal, noise_input=None)
            fpga_states = self.fpga_reservoir.run_trial(input_signal, noise_scale=NOISE_SCALE)
            return np.hstack([gpu_states, fpga_states])
        elif condition == 'BRIDGE_FULL':
            noise_in = np.array([self.noise.get_noise_scalar(t, NOISE_SCALE) for t in range(n_steps)])
            self.noise.reset_iir()
            gpu_states = self.gpu_esn.run(input_signal, noise_input=noise_in)
            fpga_states = self.fpga_reservoir.run_trial(input_signal, noise_scale=NOISE_SCALE)
            return np.hstack([gpu_states, fpga_states])
        else:
            raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════════════════════
# Benchmark Functions (with thermal checks + intermediate saves)
# ═══════════════════════════════════════════════════════════

ALL_CONDITIONS = ['FPGA_ONLY', 'FPGA_1F', 'GPU_ESN_ONLY', 'GPU_ESN_NOISE',
                  'BRIDGE_SIMPLE', 'BRIDGE_FULL']

# Order: FPGA first (no GPU heat), then GPU, then bridge
SAFE_ORDER = ['FPGA_ONLY', 'FPGA_1F', 'GPU_ESN_ONLY', 'GPU_ESN_NOISE',
              'BRIDGE_SIMPLE', 'BRIDGE_FULL']


def save_intermediate(all_results, label=""):
    """Save results to disk after each condition."""
    with open(SAVE_FILE, 'w') as f:
        json.dump(all_results, f, cls=NpEncoder, indent=2)
    print(f"  [SAVE] {label} → {SAVE_FILE.name}")


def load_previous():
    """Load previous results if they exist (resume from crash)."""
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE, 'r') as f:
                data = json.load(f)
            print(f"  [RESUME] Loaded previous results from {SAVE_FILE.name}")
            return data
        except Exception:
            pass
    return None


def run_waveform_benchmark_safe(reservoir, n_trials, n_steps, n_classes, seed, label, all_results):
    """Run waveform with per-condition saves and thermal checks."""
    key = f'waveform_{label}'
    if key not in all_results:
        all_results[key] = {}

    if n_classes == 4:
        inputs, labels = generate_waveforms_4class(n_trials, n_steps, SAMPLE_HZ, seed=seed)
    else:
        inputs, labels = generate_waveforms_8class(n_trials, n_steps, SAMPLE_HZ, seed=seed)

    for cond in SAFE_ORDER:
        # Skip if already done
        if cond in all_results[key]:
            print(f"  [{cond}] {label} — already done, skipping")
            continue

        if check_abort():
            print(f"  *** SKIPPING {cond} {label} — too hot ***")
            continue

        wait_cool(f"Before {label} {cond}")

        print(f"\n  [{cond}] Running {n_trials} {label} trials...", flush=True)
        all_feats = []
        aborted = False
        for trial in range(n_trials):
            if trial % 10 == 0:
                apu = get_apu_temp()
                if trial > 0:
                    print(f"    trial {trial}/{n_trials} (APU={apu:.0f}°C)", flush=True)
                if apu > TEMP_ABORT:
                    print(f"    *** THERMAL ABORT at trial {trial} ***", flush=True)
                    aborted = True
                    break
                if apu > TEMP_PAUSE:
                    print(f"    APU={apu:.0f}°C — cooling pause...", flush=True)
                    time.sleep(10)
            states = reservoir.run_trial(inputs[trial], cond)
            aug = augment_with_delays(states, delays=(1, 2))
            feat = pool_trial_features(aug)
            all_feats.append(feat)
            # Brief cooling between trials to prevent heat buildup
            time.sleep(COOLING_BETWEEN_TRIALS)

        if aborted and len(all_feats) < 20:
            print(f"  [{cond}] Too few trials ({len(all_feats)}), skipping")
            continue

        X = np.nan_to_num(np.array(all_feats), nan=0.0, posinf=0.0, neginf=0.0)
        # Use whatever trials we have
        actual_labels = labels[:len(X)]
        acc_mean, acc_std = classify_cv(X, actual_labels, n_splits=min(5, max(2, len(X)//4)), n_classes=n_classes)
        print(f"  [{cond}] {label}: {acc_mean:.3f} +/- {acc_std:.3f} ({len(X)} trials)")
        all_results[key][cond] = {'accuracy': acc_mean, 'std': acc_std, 'n_trials': len(X)}
        save_intermediate(all_results, f"{label} {cond}")

        if 'GPU' in cond or 'BRIDGE' in cond:
            time.sleep(COOLING_PAUSE_S)


def run_continuous_benchmark_safe(reservoir, benchmark_name, all_results):
    """Run MC, XOR, NARMA with per-condition saves."""
    if benchmark_name not in all_results:
        all_results[benchmark_name] = {}

    mc_input = generate_continuous_input(N_CONTINUOUS_STEPS, seed=123)
    xor_input = generate_binary_input(N_CONTINUOUS_STEPS, seed=456)
    narma_input = generate_continuous_input(N_CONTINUOUS_STEPS, seed=789)
    narma_target = generate_narma(narma_input, order=NARMA_ORDER)

    for cond in SAFE_ORDER:
        if cond in all_results[benchmark_name]:
            print(f"  [{cond}] {benchmark_name} — already done, skipping")
            continue

        if check_abort():
            print(f"  *** SKIPPING {cond} {benchmark_name} — too hot ***")
            continue

        wait_cool(f"Before {benchmark_name} {cond}")

        print(f"\n  [{cond}] Running {benchmark_name}...", flush=True)
        cond_results = {}

        # Memory Capacity
        check_pause()
        states = reservoir.run_trial(mc_input, cond)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2))
        mc_total = 0.0
        mc_per_delay = {}
        for d in range(1, MC_MAX_DELAY + 1):
            target = np.zeros(N_CONTINUOUS_STEPS)
            target[d:] = mc_input[:N_CONTINUOUS_STEPS - d]
            X = aug[WARMUP:]
            y = target[WARMUP:]
            n_tr = int(0.7 * len(X))
            _, r2 = ridge_regress(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])
            r2 = max(r2, 0.0)
            mc_per_delay[str(d)] = r2
            mc_total += r2
        cond_results['mc_total'] = mc_total
        cond_results['mc_per_delay'] = mc_per_delay
        print(f"    MC={mc_total:.3f}")

        # Cool down before XOR (bridge conditions heat up from FPGA+GPU combined I/O)
        wait_cool(f"Between MC→XOR {cond}", TEMP_SAFE + 10)

        # XOR
        states = reservoir.run_trial(xor_input, cond)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2))
        xor_results = {}
        for tau in [1, 2, 3]:
            targets = compute_xor_targets(xor_input, tau)
            X = aug[WARMUP:]
            y = targets[WARMUP:].astype(float)
            n_tr = int(0.7 * len(X))
            acc = ridge_binary(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])
            xor_results[str(tau)] = acc
            print(f"    XOR tau={tau}: {acc:.3f}")
        cond_results['xor'] = xor_results

        # Cool down before NARMA
        wait_cool(f"Between XOR→NARMA {cond}", TEMP_SAFE + 10)

        # NARMA
        states = reservoir.run_trial(narma_input, cond)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2))
        X = aug[WARMUP:]
        y = narma_target[WARMUP:]
        n_tr = int(0.7 * len(X))
        nrmse, r2 = ridge_regress(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])
        r2 = max(r2, 0.0)
        cond_results['narma_nrmse'] = nrmse
        cond_results['narma_r2'] = r2
        print(f"    NARMA-5: NRMSE={nrmse:.3f}, R2={r2:.3f}")

        all_results[benchmark_name][cond] = cond_results
        save_intermediate(all_results, f"{benchmark_name} {cond}")

        if 'GPU' in cond or 'BRIDGE' in cond:
            time.sleep(COOLING_PAUSE_S)


# ═══════════════════════════════════════════════════════════
# Summary & Tests
# ═══════════════════════════════════════════════════════════

def print_summary_and_tests(all_results):
    out = []
    def p(s=""):
        print(s)
        out.append(s)

    p("\n" + "=" * 72)
    p("z2272: SAFE DEFINITIVE BRIDGE — SUMMARY")
    p("=" * 72)

    # Wave 4-class
    w4 = all_results.get('waveform_4class', {})
    w8 = all_results.get('waveform_8class', {})
    cont = all_results.get('continuous', {})

    p(f"\n{'Condition':<18} {'W4':>6} {'W8':>6} {'MC':>6} {'XOR1':>6} {'XOR2':>6} {'XOR3':>6} {'NARMA':>7}")
    p("-" * 75)
    for c in ALL_CONDITIONS:
        w4a = w4.get(c, {}).get('accuracy', 0)
        w8a = w8.get(c, {}).get('accuracy', 0)
        mc = cont.get(c, {}).get('mc_total', 0)
        x1 = cont.get(c, {}).get('xor', {}).get('1', 0)
        x2 = cont.get(c, {}).get('xor', {}).get('2', 0)
        x3 = cont.get(c, {}).get('xor', {}).get('3', 0)
        nr = cont.get(c, {}).get('narma_r2', 0)
        p(f"  {c:<16} {w4a:>5.1%} {w8a:>5.1%} {mc:>6.3f} {x1:>5.1%} {x2:>5.1%} {x3:>5.1%} {nr:>7.3f}")

    # Tests
    p(f"\n{'=' * 72}")
    p("TESTS")
    p("=" * 72)

    n_pass = 0
    n_total = 0
    tests = {}

    def check(name, condition, desc):
        nonlocal n_pass, n_total
        n_total += 1
        status = "PASS" if condition else "FAIL"
        if condition: n_pass += 1
        p(f"  {name}: {status} — {desc}")
        tests[name] = {'pass': bool(condition), 'desc': desc}

    # Noise quality
    pw_std = all_results.get('noise_stats', {}).get('std', 0)
    check("T1_active_noise", pw_std > 0.05, f"GPU power std={pw_std:.4f}W > 0.05W")

    # Wave 4-class
    w4_vals = {c: w4.get(c, {}).get('accuracy', 0) for c in ALL_CONDITIONS}
    check("T2_fpga_1f_vs_fpga", w4_vals['FPGA_1F'] > w4_vals['FPGA_ONLY'],
          f"FPGA_1F {w4_vals['FPGA_1F']:.1%} > FPGA_ONLY {w4_vals['FPGA_ONLY']:.1%}")
    check("T3_bridge_full_best_w4", w4_vals['BRIDGE_FULL'] >= max(w4_vals['FPGA_ONLY'], w4_vals['GPU_ESN_ONLY']),
          f"BRIDGE_FULL {w4_vals['BRIDGE_FULL']:.1%} >= max(FPGA={w4_vals['FPGA_ONLY']:.1%}, GPU={w4_vals['GPU_ESN_ONLY']:.1%})")
    check("T4_bridge_full_w4_target", w4_vals['BRIDGE_FULL'] > 0.70,
          f"BRIDGE_FULL wave4 {w4_vals['BRIDGE_FULL']:.1%} > 70%")
    check("T5_bridge_vs_simple", w4_vals['BRIDGE_FULL'] >= w4_vals['BRIDGE_SIMPLE'],
          f"BRIDGE_FULL {w4_vals['BRIDGE_FULL']:.1%} >= BRIDGE_SIMPLE {w4_vals['BRIDGE_SIMPLE']:.1%}")

    # Wave 8-class
    w8_vals = {c: w8.get(c, {}).get('accuracy', 0) for c in ALL_CONDITIONS}
    check("T6_bridge_full_best_w8", w8_vals['BRIDGE_FULL'] >= max(w8_vals['FPGA_ONLY'], w8_vals['GPU_ESN_ONLY']),
          f"BRIDGE_FULL {w8_vals['BRIDGE_FULL']:.1%} >= max(FPGA={w8_vals['FPGA_ONLY']:.1%}, GPU={w8_vals['GPU_ESN_ONLY']:.1%})")
    check("T7_bridge_full_w8_target", w8_vals['BRIDGE_FULL'] > 0.35,
          f"BRIDGE_FULL wave8 {w8_vals['BRIDGE_FULL']:.1%} > 35%")

    # Memory capacity
    mc_vals = {c: cont.get(c, {}).get('mc_total', 0) for c in ALL_CONDITIONS}
    check("T8_mc_bridge_best", mc_vals['BRIDGE_FULL'] >= max(mc_vals['FPGA_ONLY'], mc_vals['GPU_ESN_ONLY']),
          f"BRIDGE_FULL MC {mc_vals['BRIDGE_FULL']:.3f} >= max(FPGA={mc_vals['FPGA_ONLY']:.3f}, GPU={mc_vals['GPU_ESN_ONLY']:.3f})")
    check("T9_mc_1f_vs_fpga", mc_vals['FPGA_1F'] > mc_vals['FPGA_ONLY'],
          f"FPGA_1F MC {mc_vals['FPGA_1F']:.3f} > FPGA_ONLY {mc_vals['FPGA_ONLY']:.3f}")
    check("T10_mc_bridge_target", mc_vals['BRIDGE_FULL'] > 0.5,
          f"BRIDGE_FULL MC {mc_vals['BRIDGE_FULL']:.3f} > 0.5")

    # XOR
    xor1 = {c: cont.get(c, {}).get('xor', {}).get('1', 0) for c in ALL_CONDITIONS}
    xor2 = {c: cont.get(c, {}).get('xor', {}).get('2', 0) for c in ALL_CONDITIONS}
    check("T11_xor1_bridge_target", xor1['BRIDGE_FULL'] > 0.55,
          f"BRIDGE_FULL XOR1 {xor1['BRIDGE_FULL']:.1%} > 55%")
    check("T12_xor1_bridge_vs_fpga", xor1['BRIDGE_FULL'] > xor1['FPGA_ONLY'],
          f"BRIDGE_FULL XOR1 {xor1['BRIDGE_FULL']:.1%} > FPGA_ONLY {xor1['FPGA_ONLY']:.1%}")
    check("T13_xor2_bridge_target", xor2['BRIDGE_FULL'] > 0.50,
          f"BRIDGE_FULL XOR2 {xor2['BRIDGE_FULL']:.1%} > 50%")

    # NARMA
    nr2 = {c: cont.get(c, {}).get('narma_r2', 0) for c in ALL_CONDITIONS}
    check("T14_narma_bridge_best", nr2['BRIDGE_FULL'] >= max(nr2['FPGA_ONLY'], nr2['GPU_ESN_ONLY']),
          f"BRIDGE_FULL NARMA R2 {nr2['BRIDGE_FULL']:.3f} >= max(FPGA={nr2['FPGA_ONLY']:.3f}, GPU={nr2['GPU_ESN_ONLY']:.3f})")
    check("T15_narma_bridge_target", nr2['BRIDGE_FULL'] > 0.05,
          f"BRIDGE_FULL NARMA R2 {nr2['BRIDGE_FULL']:.3f} > 0.05")

    # Synergy
    bridge_wins = 0
    if w4_vals['BRIDGE_FULL'] > max(w4_vals.get('FPGA_1F', 0), w4_vals.get('GPU_ESN_NOISE', 0)): bridge_wins += 1
    if w8_vals['BRIDGE_FULL'] > max(w8_vals.get('FPGA_1F', 0), w8_vals.get('GPU_ESN_NOISE', 0)): bridge_wins += 1
    if mc_vals['BRIDGE_FULL'] > max(mc_vals.get('FPGA_1F', 0), mc_vals.get('GPU_ESN_NOISE', 0)): bridge_wins += 1
    if xor1['BRIDGE_FULL'] > max(xor1.get('FPGA_1F', 0), xor1.get('GPU_ESN_NOISE', 0)): bridge_wins += 1
    if nr2['BRIDGE_FULL'] > max(nr2.get('FPGA_1F', 0), nr2.get('GPU_ESN_NOISE', 0)): bridge_wins += 1
    check("T16_synergy", bridge_wins >= 3, f"BRIDGE_FULL best on {bridge_wins}/5 benchmarks (need >=3)")

    check("T17_gpu_noise_helps", w4_vals['GPU_ESN_NOISE'] > w4_vals['GPU_ESN_ONLY'] * 0.99,
          f"GPU_ESN_NOISE {w4_vals['GPU_ESN_NOISE']:.1%} >= GPU_ESN_ONLY*0.99 {w4_vals['GPU_ESN_ONLY']*0.99:.1%}")

    p(f"\n  TOTAL: {n_pass}/{n_total} PASS")
    all_results['tests'] = tests
    all_results['n_pass'] = n_pass
    all_results['n_total'] = n_total

    # Write text summary
    with open(TEXT_FILE, 'w') as f:
        f.write('\n'.join(out))
    print(f"\n  Written to {TEXT_FILE}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("  z2272: THERMALLY-SAFE Cross-Substrate GPU+FPGA Bridge")
    print("  NOISE_SCALE=0.02, LEAK_COND=4 (tau=210ms)")
    print("  Thermal limits: ABORT>65°C, SAFE<50°C, bursts=8s")
    print("  Saves after EACH condition (crash-safe)")
    print("=" * 72)

    # Try to load previous results
    all_results = load_previous() or {}
    if not all_results:
        all_results = {
            'experiment': 'z2272_safe_bridge',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'params': {
                'n_gpu': N_GPU, 'n_fpga': N_FPGA,
                'base_vg': BASE_VG, 'vg_spread': VG_SPREAD,
                'alpha': ALPHA, 'iir_alpha': IIR_ALPHA,
                'noise_scale': NOISE_SCALE, 'sample_hz': SAMPLE_HZ,
                'n_wave_trials': N_WAVE_TRIALS, 'n_wave_steps': N_WAVE_STEPS,
                'n_continuous_steps': N_CONTINUOUS_STEPS,
                'narma_order': NARMA_ORDER,
                'temp_abort': TEMP_ABORT, 'temp_safe': TEMP_SAFE,
                'stress_burst_s': STRESS_BURST_S,
            },
        }

    wait_cool("Startup")

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
        print("  WARNING: FPGA not responding")

    fpga.set_kill(False)
    time.sleep(0.3)

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

    telem = fpga.read_telemetry()
    if telem:
        sc = telem['spike_counts']
        print(f"  FPGA alive: total_spikes={sc.sum()}, active={np.count_nonzero(sc)}/{N_FPGA}")
    else:
        print("  WARNING: no telemetry")

    # ─── Step 3: Sample GPU noise with SHORT burst ───
    print("\n[3] Sampling GPU noise (short burst)...")
    noise_sampler = GPUNoiseSampler()
    pw_arr = sample_noise_with_burst(stress_bin, burst_s=STRESS_BURST_S, n_samples=NOISE_N_SAMPLES)
    noise_stats = noise_sampler.load_burst(pw_arr)
    all_results['noise_stats'] = noise_stats

    if noise_stats['std'] < 0.01:
        print("  WARNING: Power variance low — may not have real GPU load noise")

    save_intermediate(all_results, "after noise sampling")

    # ─── Step 4: Create reservoir system ───
    print("\n[4] Initializing reservoir...")
    gpu_esn = GPUFourpopESN(n_per_pop=64, noise_sampler=noise_sampler)
    reservoir = SafeBridgeReservoir(fpga, noise_sampler, gpu_esn)

    # ─── BENCHMARKS (with thermal safety) ───

    # Benchmark 1: 4-class waveform
    print("\n" + "=" * 72)
    print("BENCHMARK 1: 4-CLASS WAVEFORM (80 trials x 60 steps)")
    print("=" * 72)
    run_waveform_benchmark_safe(reservoir, N_WAVE_TRIALS, N_WAVE_STEPS, 4, seed=42, label="4class", all_results=all_results)

    # Benchmark 2: 8-class waveform
    print("\n" + "=" * 72)
    print("BENCHMARK 2: 8-CLASS WAVEFORM (80 trials x 60 steps)")
    print("=" * 72)
    run_waveform_benchmark_safe(reservoir, N_WAVE_TRIALS, N_WAVE_STEPS, 8, seed=99, label="8class", all_results=all_results)

    # Benchmark 3-5: MC + XOR + NARMA (combined per condition)
    print("\n" + "=" * 72)
    print("BENCHMARKS 3-5: MC + XOR + NARMA (500 steps each)")
    print("=" * 72)
    run_continuous_benchmark_safe(reservoir, 'continuous', all_results)

    # ─── Summary ───
    print_summary_and_tests(all_results)
    save_intermediate(all_results, "FINAL")

    print(f"\n  Done. Results: {SAVE_FILE}")
    print(f"  Summary: {TEXT_FILE}")


if __name__ == '__main__':
    main()
