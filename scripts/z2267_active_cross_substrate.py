#!/usr/bin/env python3
"""z2267_active_cross_substrate.py — Active GPU Workload + FPGA Reservoir

z2265 failed because GPU firmware noise sources (hwmon power, PM table thermal,
clock frequency) returned near-zero variance when idle. This experiment launches
a background GPU stress kernel to generate real 1/f power dynamics during sampling.

Architecture:
  GPU stress kernel (HIP FMA loops) → power/thermal fluctuations
  hwmon/PM table sampling → IIR 1/f filter → Vg modulation → FPGA 128 neurons
  GPU twopop ESN → combined readout with FPGA states

Conditions:
  FPGA_ONLY:        no noise, input-only Vg
  FPGA_ACTIVE_1F:   1/f filtered active GPU noise → Vg
  FPGA_ACTIVE_WHITE: white noise control → Vg
  GPU_ONLY:         twopop ESN states only (no FPGA)
  COMBINED:         GPU twopop + FPGA spike states concatenated

Tasks: 4-class waveform, MC d=1-10, XOR tau=1-3

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z2267_active_cross_substrate.py
"""

import os, sys, time, json, struct, subprocess, signal, tempfile
import numpy as np
from pathlib import Path

# ─── Paths ───
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)

# ─── Parameters ───
N_GPU = 128
N_FPGA = 128
BASE_VG = 0.58
VG_SPREAD = 0.08
ALPHA = 0.25
IIR_ALPHA = 0.85
NOISE_SCALE = 0.05        # maps noise to ±0.05 Vg range
GPU_TO_FPGA_GAIN = 0.10
SAMPLE_HZ = 50            # FPGA telemetry rate (conservative for serial noise sampling)
STRESS_DURATION = 60       # seconds of GPU stress per condition
TEMP_LIMIT_HARD = 80.0     # kill stress above this
TEMP_LIMIT_SOFT = 75.0     # wait 30s above this
NOISE_SAMPLE_HZ = 200      # how fast to sample hwmon during stress

# Task params
N_WAVE_TRIALS = 80
N_WAVE_STEPS = 60
N_CONTINUOUS_STEPS = 1200
MC_MAX_DELAY = 10
WARMUP = 50

# ─── Temperature Monitoring ───

def get_edge_temp():
    """Read GPU edge temperature from hwmon."""
    try:
        with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
            return float(f.read().strip()) / 1000.0  # mC → °C
    except Exception:
        pass
    try:
        out = subprocess.check_output(['sensors'], text=True, timeout=5)
        for line in out.splitlines():
            if 'edge:' in line:
                val = line.split('+')[1].split('°')[0]
                return float(val)
    except Exception:
        pass
    return 0.0


def check_temp_and_wait(label="", hard_limit=TEMP_LIMIT_HARD, soft_limit=TEMP_LIMIT_SOFT):
    """Check temperature, wait if needed. Returns temp."""
    temp = get_edge_temp()
    print(f"  [TEMP] {label} edge={temp:.1f}°C", end="")
    if temp > hard_limit:
        print(f" — CRITICAL (>{hard_limit}°C, waiting 60s)...")
        time.sleep(60)
        temp = get_edge_temp()
        print(f"  [TEMP] After cooling: edge={temp:.1f}°C")
    elif temp > soft_limit:
        print(f" — COOLING (>{soft_limit}°C, waiting 30s)...")
        time.sleep(30)
        temp = get_edge_temp()
        print(f"  [TEMP] After cooling: edge={temp:.1f}°C")
    else:
        print(f" — OK")
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
    int duration = 60;
    if (argc > 1) duration = atoi(argv[1]);

    signal(SIGTERM, sighandler);
    signal(SIGINT, sighandler);

    int N = 64 * 256;  // threads
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
    """Compile GPU stress kernel. Returns path to binary or None."""
    src_path = '/tmp/z2267_gpu_stress.hip'
    bin_path = '/tmp/z2267_gpu_stress'

    # Check if already compiled
    if os.path.isfile(bin_path):
        # Recompile if source changed
        if os.path.isfile(src_path):
            src_mtime = os.path.getmtime(src_path)
            bin_mtime = os.path.getmtime(bin_path)
            if bin_mtime > src_mtime:
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


def launch_stress_kernel(bin_path, duration=60):
    """Launch GPU stress in background. Returns subprocess.Popen."""
    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    proc = subprocess.Popen(
        [bin_path, str(duration)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print(f"  [STRESS] Launched PID={proc.pid}, duration={duration}s")
    time.sleep(2)  # let GPU ramp up
    return proc


def kill_stress(proc):
    """Kill stress kernel safely."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        print(f"  [STRESS] Killed PID={proc.pid}")


# ═══════════════════════════════════════════════════════════
# Active GPU Noise Sampler
# ═══════════════════════════════════════════════════════════

class ActiveGPUNoiseSampler:
    """Sample GPU firmware noise WHILE a stress kernel is running.

    Sources:
      - power: /sys/class/hwmon/hwmon7/power1_average (~11W ± 1.5W under load)
      - thermal: /sys/kernel/ryzen_smu_drv/pm_table offset 0x004C (float32 hotspot)
      - clock: /sys/class/hwmon/hwmon7/freq1_input (GPU clock Hz)
    """

    def __init__(self):
        self._iir_state = 0.0
        self._power_buf = []
        self._thermal_buf = []
        self._clock_buf = []
        self._power_norm = None
        self._thermal_norm = None
        self._clock_norm = None

    def sample_during_stress(self, stress_proc, n_samples=500, sample_hz=200):
        """Collect noise samples while GPU stress is running."""
        print(f"  [NOISE] Sampling {n_samples} points at ~{sample_hz} Hz while GPU active...")
        interval = 1.0 / sample_hz
        self._power_buf = []
        self._thermal_buf = []
        self._clock_buf = []

        for i in range(n_samples):
            t0 = time.perf_counter()

            # Check stress is still alive
            if stress_proc and stress_proc.poll() is not None:
                print(f"  [NOISE] Stress kernel died at sample {i}")
                break

            # Safety: check temp every 50 samples
            if i > 0 and i % 50 == 0:
                temp = get_edge_temp()
                if temp > TEMP_LIMIT_HARD:
                    print(f"  [NOISE] TEMP {temp:.1f}°C > {TEMP_LIMIT_HARD}°C — stopping sampling")
                    break

            # Power
            try:
                with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
                    pw = float(f.read().strip()) / 1e6  # uW → W
                self._power_buf.append(pw)
            except Exception:
                self._power_buf.append(self._power_buf[-1] if self._power_buf else 0.0)

            # Thermal (PM table hotspot)
            try:
                with open('/sys/kernel/ryzen_smu_drv/pm_table', 'rb') as f:
                    f.seek(0x004C)
                    raw = f.read(4)
                    if len(raw) == 4:
                        th = struct.unpack('<f', raw)[0]
                        self._thermal_buf.append(th)
                    else:
                        self._thermal_buf.append(self._thermal_buf[-1] if self._thermal_buf else 0.0)
            except Exception:
                self._thermal_buf.append(self._thermal_buf[-1] if self._thermal_buf else 0.0)

            # Clock frequency
            try:
                with open('/sys/class/hwmon/hwmon7/freq1_input', 'r') as f:
                    ck = float(f.read().strip()) / 1e6  # Hz → MHz
                self._clock_buf.append(ck)
            except Exception:
                self._clock_buf.append(self._clock_buf[-1] if self._clock_buf else 0.0)

            # Rate limit
            elapsed = time.perf_counter() - t0
            remaining = interval - elapsed
            if remaining > 0.0001:
                time.sleep(remaining)

        n = len(self._power_buf)
        print(f"  [NOISE] Collected {n} samples")

        # Report raw stats
        pw_arr = np.array(self._power_buf)
        th_arr = np.array(self._thermal_buf)
        ck_arr = np.array(self._clock_buf)
        print(f"    power:   mean={pw_arr.mean():.2f}W, std={pw_arr.std():.4f}W, "
              f"range=[{pw_arr.min():.2f}, {pw_arr.max():.2f}]")
        print(f"    thermal: mean={th_arr.mean():.2f}°C, std={th_arr.std():.4f}°C, "
              f"range=[{th_arr.min():.2f}, {th_arr.max():.2f}]")
        print(f"    clock:   mean={ck_arr.mean():.1f}MHz, std={ck_arr.std():.4f}MHz, "
              f"range=[{ck_arr.min():.1f}, {ck_arr.max():.1f}]")

        # Normalize to zero-mean unit-variance
        def _norm(arr):
            std = arr.std()
            if std > 1e-8:
                return (arr - arr.mean()) / std
            else:
                return np.zeros_like(arr)

        self._power_norm = _norm(pw_arr)
        self._thermal_norm = _norm(th_arr)
        self._clock_norm = _norm(ck_arr)
        self._iir_state = 0.0

        return {
            'n_samples': n,
            'power_mean': float(pw_arr.mean()),
            'power_std': float(pw_arr.std()),
            'thermal_mean': float(th_arr.mean()),
            'thermal_std': float(th_arr.std()),
            'clock_mean': float(ck_arr.mean()),
            'clock_std': float(ck_arr.std()),
        }

    def get_1f_noise(self, t, gain=NOISE_SCALE):
        """IIR-filtered power noise (1/f character)."""
        if self._power_norm is None or len(self._power_norm) == 0:
            return 0.0
        raw = self._power_norm[t % len(self._power_norm)]
        self._iir_state = IIR_ALPHA * self._iir_state + (1 - IIR_ALPHA) * raw
        return self._iir_state * gain

    def get_white_noise(self, gain=NOISE_SCALE):
        """White noise control (no temporal correlation)."""
        return np.random.randn() * gain

    def get_full_noise(self, t, gain=NOISE_SCALE):
        """Multi-source: IIR power + thermal + clock jitter."""
        if self._power_norm is None or len(self._power_norm) == 0:
            return np.random.randn() * gain * 0.1
        n = len(self._power_norm)
        pw = self._power_norm[t % n]
        th = self._thermal_norm[t % n] if self._thermal_norm is not None and len(self._thermal_norm) > 0 else 0.0
        ck = self._clock_norm[t % n] if self._clock_norm is not None and len(self._clock_norm) > 0 else 0.0
        self._iir_state = IIR_ALPHA * self._iir_state + (1 - IIR_ALPHA) * pw
        combined = 0.5 * self._iir_state + 0.3 * th + 0.2 * ck
        return combined * gain

    def get_noise_vector(self, t, n_neurons, mode='1f', gain=NOISE_SCALE):
        """Per-neuron noise vector for a given mode."""
        if mode == '1f':
            base = self.get_1f_noise(t, gain)
            return base + np.random.randn(n_neurons) * gain * 0.1
        elif mode == 'white':
            return np.random.randn(n_neurons) * gain
        elif mode == 'full':
            base = self.get_full_noise(t, gain)
            return base + np.random.randn(n_neurons) * gain * 0.1
        else:
            return np.zeros(n_neurons)

    def reset_iir(self):
        self._iir_state = 0.0


# ═══════════════════════════════════════════════════════════
# GPU Two-Population ESN (Python fallback)
# ═══════════════════════════════════════════════════════════

class GPUTwopopESN:
    """Python ESN that mimics GPU two-population physics reservoir."""

    def __init__(self, noise_sampler=None):
        rng = np.random.default_rng(7777)
        self.N = N_GPU
        self.leak = 0.05 + 0.15 * rng.random(self.N)
        self.input_w = 0.05 + 0.20 * rng.random(self.N)
        self.thr = 0.4 + 0.5 * rng.random(self.N)
        self.bias = 0.02 * (rng.random(self.N) - 0.5)
        self.bthr = 0.5 + 0.3 * np.arange(self.N) / (self.N - 1)
        self.W_rec = rng.standard_normal((self.N, self.N)) * 0.04
        mask = rng.random((self.N, self.N)) > 0.9
        self.W_rec *= mask
        self.is_pop_a = np.arange(self.N) < (self.N // 2)
        self.noise_sampler = noise_sampler

    def run(self, input_seq):
        """Run reservoir. Returns (n_steps, N_GPU) state matrix."""
        n_steps = len(input_seq)
        states = np.zeros((n_steps, self.N))
        v = np.zeros(self.N)
        h = np.zeros(self.N)
        slow = np.zeros(self.N)

        for t in range(n_steps):
            u = input_seq[t]
            branch_val = np.where(v > self.bthr, 1.0, -1.0)

            # L1 bank conflict jitter for Pop B — use real HW noise if available
            l1_signal = np.zeros(self.N)
            if self.noise_sampler is not None and self.noise_sampler._power_norm is not None:
                n_b = np.sum(~self.is_pop_a)
                l1_signal[~self.is_pop_a] = self.noise_sampler.get_noise_vector(
                    t, n_b, mode='1f', gain=0.3
                )
            else:
                l1_signal[~self.is_pop_a] = np.random.randn(np.sum(~self.is_pop_a)) * 0.3

            rec = self.W_rec @ v
            pll = np.random.uniform(-1, 1, self.N) * 0.005

            v_new = np.tanh(
                (1.0 - self.leak) * v
                + self.input_w * u
                + rec
                + self.bias
                + 0.02 * branch_val
                + 0.005 * l1_signal
                + pll
            )

            spike_mask = v_new > self.thr
            v_new[spike_mask] -= self.thr[spike_mask]
            v = v_new

            h = 0.93 * h + 0.07 * v
            slow = 0.99 * slow + 0.01 * v
            states[t] = v + 0.3 * h + 0.1 * slow

        return states


# ═══════════════════════════════════════════════════════════
# Ridge Regression Utilities
# ═══════════════════════════════════════════════════════════

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(np.concatenate([y_tr, y_te])))
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
    alphas = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
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


def classify_cv(X, y, n_splits=5, n_classes=None):
    splits = stratified_kfold(X, y, n_splits)
    accs = []
    for tr_idx, te_idx in splits:
        acc = ridge_classify(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], n_classes=n_classes)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Task Generation
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2, 3)):
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


def generate_binary_input(n_steps, seed=42):
    return np.random.default_rng(seed).integers(0, 2, size=n_steps).astype(float)


def generate_continuous_input(n_steps, seed=42):
    return np.random.default_rng(seed).uniform(-1, 1, size=n_steps).astype(np.float32)


def compute_xor_targets(u, tau):
    n = len(u)
    targets = np.zeros(n, dtype=int)
    for t in range(tau, n):
        targets[t] = int(u[t]) ^ int(u[t - tau])
    return targets


# ═══════════════════════════════════════════════════════════
# Active Cross-Substrate Reservoir
# ═══════════════════════════════════════════════════════════

class ActiveCrossSubstrateReservoir:
    """Cross-substrate reservoir with active GPU noise sampling.

    Unlike z2265 which sampled idle GPU noise (near-zero variance),
    this launches a stress kernel to generate real power/thermal dynamics.
    """

    def __init__(self, fpga, noise_sampler, gpu_esn):
        self.fpga = fpga
        self.noise = noise_sampler
        self.gpu = gpu_esn

        rng = np.random.default_rng(42)
        self.base_vg = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, size=N_FPGA)
        self.w_in = rng.uniform(-1, 1, size=N_FPGA)
        self.gpu_to_fpga = rng.standard_normal((N_GPU, N_FPGA)) * GPU_TO_FPGA_GAIN / np.sqrt(N_GPU)

    def run_trial(self, input_signal, mode='combined'):
        """Run one trial through reservoir.

        Modes:
          fpga_only:        input → Vg, no noise
          fpga_active_1f:   input + IIR 1/f GPU noise → Vg
          fpga_active_white: input + white noise → Vg
          gpu_only:         GPU ESN only
          combined:         GPU ESN → Vg coupling + FPGA, concatenated readout
        """
        n_steps = len(input_signal)
        interval = 1.0 / SAMPLE_HZ

        # GPU ESN for gpu_only and combined
        gpu_states = None
        if mode in ('gpu_only', 'combined'):
            gpu_states = self.gpu.run(input_signal)

        if mode == 'gpu_only':
            return gpu_states

        # FPGA path
        fpga_states = np.zeros((n_steps, N_FPGA * 2))
        prev_counts = None

        # Determine noise mode
        if mode == 'fpga_active_1f':
            noise_mode = '1f'
        elif mode == 'fpga_active_white':
            noise_mode = 'white'
        elif mode == 'combined':
            noise_mode = '1f'  # combined uses 1/f noise + GPU coupling
        else:
            noise_mode = None

        self.noise.reset_iir()

        for t in range(n_steps):
            t_start = time.perf_counter()

            # Compute Vg: base + input
            vg = self.base_vg.copy() + ALPHA * input_signal[t] * self.w_in

            # GPU ESN coupling (combined mode)
            if mode == 'combined' and gpu_states is not None:
                gpu_coupling = gpu_states[t] @ self.gpu_to_fpga
                vg += gpu_coupling

            # GPU firmware noise injection
            if noise_mode is not None:
                noise_vec = self.noise.get_noise_vector(t, N_FPGA, noise_mode, NOISE_SCALE)
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

            if n_steps > 100 and (t + 1) % (n_steps // 5) == 0:
                rate = 1.0 / max(time.perf_counter() - t_start, 1e-6)
                print(f"      step {t+1}/{n_steps} ({rate:.0f} Hz)")

        if mode == 'combined':
            return np.hstack([gpu_states, fpga_states])
        else:
            return fpga_states


# ═══════════════════════════════════════════════════════════
# Benchmark Runners
# ═══════════════════════════════════════════════════════════

def run_waveform_benchmark(reservoir, modes, n_trials=N_WAVE_TRIALS, n_steps=N_WAVE_STEPS):
    inputs, labels = generate_waveforms_4class(n_trials, n_steps, SAMPLE_HZ, seed=42)
    results = {}
    for mode in modes:
        check_temp_and_wait(f"Before wave {mode}")
        print(f"\n  [{mode}] Running {n_trials} waveform trials...")
        all_feats = []
        for trial in range(n_trials):
            states = reservoir.run_trial(inputs[trial], mode=mode)
            aug = augment_with_delays(states, delays=(1, 2))
            feat = pool_trial_features(aug)
            all_feats.append(feat)
            if (trial + 1) % 20 == 0:
                print(f"    trial {trial+1}/{n_trials}")
        X = np.array(all_feats)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        acc_mean, acc_std = classify_cv(X, labels, n_splits=5, n_classes=4)
        print(f"  [{mode}] WAVEFORM: {acc_mean:.3f} +/- {acc_std:.3f}")
        results[mode] = {'accuracy': acc_mean, 'std': acc_std}
    return results


def run_mc_benchmark(reservoir, modes, n_steps=N_CONTINUOUS_STEPS):
    mc_input = generate_continuous_input(n_steps, seed=123)
    results = {}
    for mode in modes:
        check_temp_and_wait(f"Before MC {mode}")
        print(f"\n  [{mode}] Running memory capacity ({n_steps} steps)...")
        states = reservoir.run_trial(mc_input, mode=mode)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2, 3))
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
        print(f"  [{mode}] MC TOTAL: {mc_total:.3f}")
        for d in range(1, MC_MAX_DELAY + 1):
            print(f"    d={d:2d}: r²={mc_per_delay[d]:.3f}")
        results[mode] = {'total': mc_total, 'per_delay': mc_per_delay}
    return results


def run_xor_benchmark(reservoir, modes, n_steps=N_CONTINUOUS_STEPS):
    xor_input = generate_binary_input(n_steps, seed=456)
    results = {}
    for mode in modes:
        check_temp_and_wait(f"Before XOR {mode}")
        print(f"\n  [{mode}] Running temporal XOR ({n_steps} steps)...")
        states = reservoir.run_trial(xor_input, mode=mode)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2, 3))
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
        results[mode] = xor_results
    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("  z2267: Active GPU Workload + FPGA Cross-Substrate Reservoir")
    print("  KEY FIX: GPU stress kernel drives real power/thermal dynamics")
    print("  GPU: stress FMA → hwmon noise sampling → IIR 1/f → FPGA Vg")
    print("  FPGA: 128 LIF neurons on Arty A7-100T via Ethernet")
    print("  Conditions: FPGA_ONLY, FPGA_ACTIVE_1F, FPGA_ACTIVE_WHITE,")
    print("              GPU_ONLY, COMBINED")
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
        print("  WARNING: FPGA not responding — will run with timeouts")

    # Kill switch off
    fpga.set_kill(False)
    time.sleep(0.3)

    # Set heterogeneous Vg
    print("  Setting heterogeneous Vg (BASE_VG=0.58 ± 0.08)...")
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

    stress_proc = launch_stress_kernel(stress_bin, duration=STRESS_DURATION)
    time.sleep(3)  # let power ramp

    noise_sampler = ActiveGPUNoiseSampler()
    noise_stats = noise_sampler.sample_during_stress(
        stress_proc, n_samples=800, sample_hz=NOISE_SAMPLE_HZ
    )

    # Check if noise was actually collected with variance
    if noise_stats['power_std'] < 0.01:
        print("  WARNING: Power variance still low — GPU stress may not be generating load")
        print("  Continuing anyway with available noise...")

    # Let stress continue a bit more, then kill it to cool down
    check_temp_and_wait("Post-sampling")

    # Kill stress for now — we'll relaunch per condition if needed
    kill_stress(stress_proc)
    stress_proc = None
    time.sleep(5)
    check_temp_and_wait("Post-stress-kill")

    # ─── Step 4: Create reservoir ───
    print("\n[4] Initializing active cross-substrate reservoir...")
    gpu_esn = GPUTwopopESN(noise_sampler=noise_sampler)
    reservoir = ActiveCrossSubstrateReservoir(fpga, noise_sampler, gpu_esn)

    modes = ['fpga_only', 'fpga_active_1f', 'fpga_active_white', 'gpu_only', 'combined']

    all_results = {
        'experiment': 'z2267_active_cross_substrate',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'noise_stats': noise_stats,
        'params': {
            'n_gpu': N_GPU, 'n_fpga': N_FPGA,
            'base_vg': BASE_VG, 'vg_spread': VG_SPREAD,
            'alpha': ALPHA, 'iir_alpha': IIR_ALPHA,
            'noise_scale': NOISE_SCALE,
            'gpu_to_fpga_gain': GPU_TO_FPGA_GAIN,
            'sample_hz': SAMPLE_HZ,
            'stress_duration': STRESS_DURATION,
            'n_wave_trials': N_WAVE_TRIALS,
            'n_wave_steps': N_WAVE_STEPS,
            'n_continuous_steps': N_CONTINUOUS_STEPS,
        },
    }

    # ═══════════════════════════════════════════════════════
    # BENCHMARK 1: Waveform Classification
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("BENCHMARK 1: 4-CLASS WAVEFORM CLASSIFICATION")
    print(f"  {N_WAVE_TRIALS} trials × {N_WAVE_STEPS} steps @ {SAMPLE_HZ} Hz")
    print("=" * 72)

    # Re-sample noise with fresh stress for waveform benchmark
    print("  [RE-STRESS] Launching fresh GPU stress for waveform benchmark...")
    check_temp_and_wait("Pre-wave-stress")
    stress_proc = launch_stress_kernel(stress_bin, duration=STRESS_DURATION + 30)
    time.sleep(2)
    fresh_stats = noise_sampler.sample_during_stress(stress_proc, n_samples=600, sample_hz=NOISE_SAMPLE_HZ)
    all_results['noise_stats_wave'] = fresh_stats

    wave_results = run_waveform_benchmark(reservoir, modes)
    all_results['waveform'] = wave_results

    kill_stress(stress_proc)
    stress_proc = None
    time.sleep(5)
    check_temp_and_wait("Post-wave")

    # ═══════════════════════════════════════════════════════
    # BENCHMARK 2: Memory Capacity
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("BENCHMARK 2: MEMORY CAPACITY d=1..10")
    print(f"  {N_CONTINUOUS_STEPS} continuous steps")
    print("=" * 72)

    print("  [RE-STRESS] Launching fresh GPU stress for MC benchmark...")
    check_temp_and_wait("Pre-MC-stress")
    stress_proc = launch_stress_kernel(stress_bin, duration=STRESS_DURATION + 30)
    time.sleep(2)
    fresh_stats = noise_sampler.sample_during_stress(stress_proc, n_samples=600, sample_hz=NOISE_SAMPLE_HZ)
    all_results['noise_stats_mc'] = fresh_stats

    mc_results = run_mc_benchmark(reservoir, modes)
    all_results['memory_capacity'] = mc_results

    kill_stress(stress_proc)
    stress_proc = None
    time.sleep(5)
    check_temp_and_wait("Post-MC")

    # ═══════════════════════════════════════════════════════
    # BENCHMARK 3: Temporal XOR
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("BENCHMARK 3: TEMPORAL XOR tau=1,2,3")
    print(f"  {N_CONTINUOUS_STEPS} continuous steps")
    print("=" * 72)

    print("  [RE-STRESS] Launching fresh GPU stress for XOR benchmark...")
    check_temp_and_wait("Pre-XOR-stress")
    stress_proc = launch_stress_kernel(stress_bin, duration=STRESS_DURATION + 30)
    time.sleep(2)
    fresh_stats = noise_sampler.sample_during_stress(stress_proc, n_samples=600, sample_hz=NOISE_SAMPLE_HZ)
    all_results['noise_stats_xor'] = fresh_stats

    xor_results = run_xor_benchmark(reservoir, modes)
    all_results['temporal_xor'] = xor_results

    kill_stress(stress_proc)
    stress_proc = None
    time.sleep(5)
    check_temp_and_wait("Post-XOR")

    # ═══════════════════════════════════════════════════════
    # SUMMARY & TESTS
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    # Extract metrics
    def _get(d, mode, key, default=0):
        return d.get(mode, {}).get(key, default)

    wave_vals = {m: _get(wave_results, m, 'accuracy') for m in modes}
    mc_vals = {m: _get(mc_results, m, 'total') for m in modes}
    xor_vals = {m: xor_results.get(m, {}) for m in modes}

    hdr = f"\n{'Metric':<25}"
    for m in modes:
        hdr += f" {m:>18}"
    print(hdr)
    print("-" * (25 + 19 * len(modes)))

    row = f"{'Waveform 4-class':<25}"
    for m in modes:
        row += f" {wave_vals[m]:>17.1%}"
    print(row)

    row = f"{'Memory Capacity':<25}"
    for m in modes:
        row += f" {mc_vals[m]:>18.3f}"
    print(row)

    for tau in [1, 2, 3]:
        row = f"{'XOR tau=' + str(tau):<25}"
        for m in modes:
            v = xor_vals[m].get(tau, 0)
            row += f" {v:>17.1%}"
        print(row)

    # ─── Tests ───
    print(f"\n{'─' * 72}")
    print("TESTS")
    print(f"{'─' * 72}")

    tests = {}
    n_pass = 0
    n_total = 0

    def check(name, condition, desc):
        nonlocal n_pass, n_total
        n_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            n_pass += 1
        print(f"  {name}: {status} — {desc}")
        tests[name] = {'pass': bool(condition), 'desc': desc}

    w_fpga = wave_vals.get('fpga_only', 0)
    w_1f = wave_vals.get('fpga_active_1f', 0)
    w_white = wave_vals.get('fpga_active_white', 0)
    w_gpu = wave_vals.get('gpu_only', 0)
    w_comb = wave_vals.get('combined', 0)

    mc_fpga = mc_vals.get('fpga_only', 0)
    mc_1f = mc_vals.get('fpga_active_1f', 0)
    mc_white = mc_vals.get('fpga_active_white', 0)
    mc_gpu = mc_vals.get('gpu_only', 0)
    mc_comb = mc_vals.get('combined', 0)

    xor1_fpga = xor_vals.get('fpga_only', {}).get(1, 0)
    xor1_1f = xor_vals.get('fpga_active_1f', {}).get(1, 0)
    xor1_white = xor_vals.get('fpga_active_white', {}).get(1, 0)
    xor1_gpu = xor_vals.get('gpu_only', {}).get(1, 0)
    xor1_comb = xor_vals.get('combined', {}).get(1, 0)
    xor2_comb = xor_vals.get('combined', {}).get(2, 0)

    # T1: Active 1/f noise has real variance (the key fix)
    pw_std = noise_stats.get('power_std', 0)
    check("T1_active_noise", pw_std > 0.05,
          f"active GPU power std={pw_std:.4f}W > 0.05W (idle was ~0)")

    # T2: Active 1/f beats FPGA alone (waveform)
    check("T2_1f_vs_fpga_wave", w_1f > w_fpga,
          f"1/f {w_1f:.1%} > FPGA {w_fpga:.1%}")

    # T3: Active 1/f beats white noise (waveform) — 1/f temporal structure helps
    check("T3_1f_vs_white_wave", w_1f > w_white,
          f"1/f {w_1f:.1%} > white {w_white:.1%}")

    # T4: Combined best waveform
    check("T4_combined_best_wave", w_comb >= max(w_fpga, w_1f, w_gpu),
          f"combined {w_comb:.1%} >= max(fpga={w_fpga:.1%}, 1f={w_1f:.1%}, gpu={w_gpu:.1%})")

    # T5: Combined waveform > 70% (lower threshold than z2265's 85% — first run)
    check("T5_wave_target", w_comb > 0.70,
          f"combined waveform {w_comb:.1%} > 70%")

    # T6: MC combined > 1.5
    check("T6_mc_target", mc_comb > 1.5,
          f"combined MC {mc_comb:.3f} > 1.5")

    # T7: MC 1/f > FPGA alone (noise helps memory)
    check("T7_mc_1f_vs_fpga", mc_1f > mc_fpga,
          f"1/f MC {mc_1f:.3f} > FPGA {mc_fpga:.3f}")

    # T8: MC 1/f > white (temporal structure aids memory)
    check("T8_mc_1f_vs_white", mc_1f > mc_white,
          f"1/f MC {mc_1f:.3f} > white {mc_white:.3f}")

    # T9: XOR tau=1 combined > 55%
    check("T9_xor1_target", xor1_comb > 0.55,
          f"combined XOR tau=1 {xor1_comb:.1%} > 55%")

    # T10: XOR tau=2 combined > 50%
    check("T10_xor2_target", xor2_comb > 0.50,
          f"combined XOR tau=2 {xor2_comb:.1%} > 50%")

    # T11: Combined > FPGA (XOR synergy)
    check("T11_xor1_synergy", xor1_comb > xor1_fpga,
          f"combined XOR1 {xor1_comb:.1%} > FPGA {xor1_fpga:.1%}")

    # T12: 1/f noise helps XOR
    check("T12_xor1_1f_vs_fpga", xor1_1f > xor1_fpga,
          f"1/f XOR1 {xor1_1f:.1%} > FPGA {xor1_fpga:.1%}")

    # T13: Thermal noise has variance (PM table readable)
    th_std = noise_stats.get('thermal_std', 0)
    check("T13_thermal_variance", th_std > 0.01,
          f"thermal std={th_std:.4f}°C > 0.01")

    # T14: Clock noise has variance
    ck_std = noise_stats.get('clock_std', 0)
    check("T14_clock_variance", ck_std > 0.01,
          f"clock std={ck_std:.4f}MHz > 0.01")

    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    all_results['tests'] = tests
    all_results['summary'] = {
        'pass': n_pass, 'total': n_total,
        'waveform': wave_vals, 'mc': mc_vals,
        'xor': {m: xor_vals.get(m, {}) for m in modes},
    }

    # ─── Save JSON ───
    json_path = RESULTS / 'z2267_active_cross_substrate.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\n  JSON saved: {json_path}")

    # ─── Save text report ───
    txt_path = RESULTS / 'z2267_active_cross_substrate.txt'
    with open(txt_path, 'w') as f:
        f.write("z2267: Active GPU Workload + FPGA Cross-Substrate Reservoir\n")
        f.write(f"Date: {all_results['timestamp']}\n")
        f.write(f"Tests: {n_pass}/{n_total} PASS\n\n")

        f.write("KEY FIX: GPU stress kernel generates real power/thermal dynamics\n")
        f.write(f"  Power std: {noise_stats.get('power_std', 0):.4f} W (was ~0 idle)\n")
        f.write(f"  Thermal std: {noise_stats.get('thermal_std', 0):.4f} °C\n")
        f.write(f"  Clock std: {noise_stats.get('clock_std', 0):.4f} MHz\n\n")

        f.write("Waveform 4-class:\n")
        for m in modes:
            f.write(f"  {m:>20}: {wave_vals[m]:.1%}\n")
        f.write(f"\nMemory Capacity (d=1..10):\n")
        for m in modes:
            f.write(f"  {m:>20}: {mc_vals[m]:.3f}\n")
        f.write(f"\nTemporal XOR:\n")
        for m in modes:
            xr = xor_vals.get(m, {})
            f.write(f"  {m:>20}: tau1={xr.get(1,0):.3f} tau2={xr.get(2,0):.3f} tau3={xr.get(3,0):.3f}\n")

        f.write(f"\nTests:\n")
        for name, t in tests.items():
            f.write(f"  {name}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']}\n")

    print(f"  Text saved: {txt_path}")

    # Cleanup
    if stress_proc and stress_proc.poll() is None:
        kill_stress(stress_proc)
    fpga.close()

    print(f"\n  DONE: {n_pass}/{n_total} PASS")
    return n_pass, n_total


if __name__ == '__main__':
    main()
