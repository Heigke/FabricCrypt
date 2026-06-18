#!/usr/bin/env python3
"""z2269_noise_scale_sweep.py — Find optimal NOISE_SCALE for GPU→FPGA injection

z2267 showed NOISE_SCALE=0.05 HURTS classification by -25pp. The noise overwhelms
the input signal. This experiment sweeps NOISE_SCALE from 0.001 to 0.05 to find the
sweet spot where 1/f noise helps without destroying the signal.

New bitstream has LEAK_COND=4 (τ=210ms vs 49ms), should improve memory retention.

Conditions:
  NO_NOISE:  baseline (input-only Vg, no GPU noise)
  SCALE_X:   IIR 1/f GPU noise at scale X → Vg modulation

Tasks: 4-class waveform (60 trials), MC d=1-5

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2269_noise_scale_sweep.py
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
N_FPGA = 128
BASE_VG = 0.58
VG_SPREAD = 0.08
ALPHA = 0.25           # input gain
IIR_ALPHA = 0.85       # 1/f filter coefficient
SAMPLE_HZ = 50         # FPGA telemetry rate

# Noise scales to sweep
NOISE_SCALES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

# Task params
N_WAVE_TRIALS = 60
N_WAVE_STEPS = 60
N_MC_STEPS = 800
MC_MAX_DELAY = 5
WARMUP = 40

# Temperature limits
TEMP_LIMIT_HARD = 80.0
TEMP_LIMIT_SOFT = 75.0


# ─── Temperature Monitoring ───

def get_edge_temp():
    try:
        with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0


def check_temp_and_wait(label=""):
    temp = get_edge_temp()
    print(f"  [TEMP] {label} edge={temp:.1f}°C", end="")
    if temp > TEMP_LIMIT_HARD:
        print(f" — CRITICAL (>{TEMP_LIMIT_HARD}°C, waiting 60s)...")
        time.sleep(60)
        temp = get_edge_temp()
        print(f"  [TEMP] After cooling: edge={temp:.1f}°C")
    elif temp > TEMP_LIMIT_SOFT:
        print(f" — COOLING (>{TEMP_LIMIT_SOFT}°C, waiting 30s)...")
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


# ─── GPU Stress Kernel ───

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
    time.sleep(3)  # let GPU ramp up
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


# ─── GPU Noise Sampler ───

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
                    print(f"  [NOISE] TEMP {temp:.1f}°C > {TEMP_LIMIT_HARD}°C — stopping")
                    break
            try:
                with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
                    pw = float(f.read().strip()) / 1e6  # uW → W
                self._raw_buf.append(pw)
            except Exception:
                self._raw_buf.append(self._raw_buf[-1] if self._raw_buf else 0.0)
            elapsed = time.perf_counter() - t0
            remaining = interval - elapsed
            if remaining > 0.0001:
                time.sleep(remaining)

        pw_arr = np.array(self._raw_buf)
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

    def reset_iir(self):
        self._iir_state = 0.0


# ─── Ridge Regression ───

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


def ridge_regress(X_tr, y_tr, X_te, y_te):
    alphas = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
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
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        if r2 > best_r2:
            best_r2 = r2
    return max(best_r2, 0.0)


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


# ─── Feature Extraction & Task Generation ───

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


def generate_continuous_input(n_steps, seed=123):
    return np.random.default_rng(seed).uniform(-1, 1, size=n_steps).astype(np.float32)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Runner
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

    def run_continuous(self, input_signal, noise_scale=0.0):
        """Run continuous input (for MC). Same as run_trial but longer."""
        return self.run_trial(input_signal, noise_scale)


# ═══════════════════════════════════════════════════════════
# Benchmarks
# ═══════════════════════════════════════════════════════════

def run_waveform_sweep(reservoir, noise_scales, n_trials=N_WAVE_TRIALS, n_steps=N_WAVE_STEPS):
    """Run waveform classification for each noise scale."""
    inputs, labels = generate_waveforms_4class(n_trials, n_steps, SAMPLE_HZ, seed=42)
    results = {}

    # NO_NOISE baseline
    print(f"\n  [NO_NOISE] Running {n_trials} waveform trials (baseline)...")
    check_temp_and_wait("Before NO_NOISE")
    all_feats = []
    for trial in range(n_trials):
        states = reservoir.run_trial(inputs[trial], noise_scale=0.0)
        aug = augment_with_delays(states, delays=(1, 2))
        feat = pool_trial_features(aug)
        all_feats.append(feat)
        if (trial + 1) % 15 == 0:
            print(f"    trial {trial+1}/{n_trials}")
    X = np.nan_to_num(np.array(all_feats), nan=0.0, posinf=0.0, neginf=0.0)
    acc_mean, acc_std = classify_cv(X, labels, n_splits=5, n_classes=4)
    print(f"  [NO_NOISE] WAVEFORM: {acc_mean:.3f} +/- {acc_std:.3f}")
    results['NO_NOISE'] = {'accuracy': acc_mean, 'std': acc_std, 'scale': 0.0}

    # Sweep noise scales
    for ns in noise_scales:
        check_temp_and_wait(f"Before SCALE={ns}")
        print(f"\n  [SCALE={ns}] Running {n_trials} waveform trials...")
        all_feats = []
        for trial in range(n_trials):
            states = reservoir.run_trial(inputs[trial], noise_scale=ns)
            aug = augment_with_delays(states, delays=(1, 2))
            feat = pool_trial_features(aug)
            all_feats.append(feat)
            if (trial + 1) % 15 == 0:
                print(f"    trial {trial+1}/{n_trials}")
        X = np.nan_to_num(np.array(all_feats), nan=0.0, posinf=0.0, neginf=0.0)
        acc_mean, acc_std = classify_cv(X, labels, n_splits=5, n_classes=4)
        print(f"  [SCALE={ns}] WAVEFORM: {acc_mean:.3f} +/- {acc_std:.3f}")
        results[f'SCALE_{ns}'] = {'accuracy': acc_mean, 'std': acc_std, 'scale': ns}

    return results


def run_mc_sweep(reservoir, noise_scales, n_steps=N_MC_STEPS):
    """Run memory capacity for each noise scale."""
    mc_input = generate_continuous_input(n_steps, seed=123)
    results = {}

    conditions = [('NO_NOISE', 0.0)] + [(f'SCALE_{ns}', ns) for ns in noise_scales]

    for label, ns in conditions:
        check_temp_and_wait(f"Before MC {label}")
        print(f"\n  [{label}] Running memory capacity ({n_steps} steps)...")
        states = reservoir.run_continuous(mc_input, noise_scale=ns)
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
            r2 = ridge_regress(X_tr, y_tr, X_te, y_te)
            mc_per_delay[d] = r2
            mc_total += r2
        print(f"  [{label}] MC TOTAL: {mc_total:.3f}")
        for d in range(1, MC_MAX_DELAY + 1):
            print(f"    d={d}: r²={mc_per_delay[d]:.3f}")
        results[label] = {'total': mc_total, 'per_delay': mc_per_delay, 'scale': ns}

    return results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("  z2269: Noise Scale Sweep — Find Optimal GPU→FPGA Injection Level")
    print("  z2267 showed NOISE_SCALE=0.05 hurts by -25pp. Sweeping 0.001-0.05")
    print("  New bitstream: LEAK_COND=4 (τ=210ms), should retain more memory")
    print("  FPGA: 128 LIF neurons on Arty A7-100T via Ethernet")
    print(f"  Scales: {NOISE_SCALES}")
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
        fpga.set_vg(nid, init_vg[nid])
        if nid % 32 == 31:
            time.sleep(0.01)
    print(f"  Set Vg for {N_FPGA} neurons: range [{init_vg.min():.3f}, {init_vg.max():.3f}]")
    time.sleep(0.5)

    # Verify telemetry
    telem = fpga.read_telemetry()
    if telem is not None:
        sc = telem['spike_counts']
        vm = telem['vmem']
        active = np.sum(sc > 0)
        print(f"  Telemetry OK: {active}/{N_FPGA} neurons active, "
              f"spike range [{sc.min()}, {sc.max()}], vmem range [{vm.min():.3f}, {vm.max():.3f}]")
    else:
        print("  WARNING: No telemetry received!")

    # ─── Step 3: Launch GPU stress and sample noise ───
    print("\n[3] Launching GPU stress and sampling noise...")
    stress_proc = launch_stress_kernel(stress_bin, duration=600)  # long enough for full sweep
    check_temp_and_wait("After stress launch")

    noise_sampler = GPUNoiseSampler()
    noise_stats = noise_sampler.sample(stress_proc, n_samples=1000, sample_hz=200)
    print(f"  Noise stats: {noise_stats}")

    # ─── Step 4: Create reservoir ───
    reservoir = FPGAReservoir(fpga, noise_sampler)

    # ─── Step 5: Waveform classification sweep ───
    print("\n" + "=" * 72)
    print("  [4] WAVEFORM CLASSIFICATION SWEEP")
    print("=" * 72)
    wave_results = run_waveform_sweep(reservoir, NOISE_SCALES)

    # ─── Step 6: Memory capacity sweep ───
    print("\n" + "=" * 72)
    print("  [5] MEMORY CAPACITY SWEEP")
    print("=" * 72)
    mc_results = run_mc_sweep(reservoir, NOISE_SCALES)

    # ─── Step 7: Kill stress and cool down ───
    print("\n[6] Killing GPU stress...")
    kill_stress(stress_proc)
    temp = get_edge_temp()
    if temp > 70.0:
        print(f"  Cooling down from {temp:.1f}°C (waiting 10s)...")
        time.sleep(10)

    # ─── Step 8: Analysis ───
    print("\n" + "=" * 72)
    print("  [7] ANALYSIS")
    print("=" * 72)

    baseline_wave = wave_results['NO_NOISE']['accuracy']
    baseline_mc = mc_results['NO_NOISE']['total']

    print(f"\n  Baseline (NO_NOISE): waveform={baseline_wave:.3f}, MC={baseline_mc:.3f}")
    print(f"\n  {'Scale':<10} {'Wave Acc':<12} {'Δ Wave':<12} {'MC Total':<12} {'Δ MC':<12}")
    print(f"  {'-'*58}")

    best_wave_scale = 0.0
    best_wave_acc = baseline_wave
    best_mc_scale = 0.0
    best_mc_total = baseline_mc

    for ns in NOISE_SCALES:
        key = f'SCALE_{ns}'
        w_acc = wave_results[key]['accuracy']
        w_delta = w_acc - baseline_wave
        m_total = mc_results[key]['total']
        m_delta = m_total - baseline_mc
        marker = ""
        if w_acc > best_wave_acc:
            best_wave_acc = w_acc
            best_wave_scale = ns
            marker = " ← BEST WAVE"
        if m_total > best_mc_total:
            best_mc_total = m_total
            best_mc_scale = ns
            if "BEST" not in marker:
                marker += " ← BEST MC"
            else:
                marker += " + MC"
        print(f"  {ns:<10.3f} {w_acc:<12.3f} {w_delta:+<12.3f} {m_total:<12.3f} {m_delta:+<12.3f}{marker}")

    # Find sweet spot (best combined rank)
    all_scales = [0.0] + NOISE_SCALES
    wave_accs = [baseline_wave] + [wave_results[f'SCALE_{ns}']['accuracy'] for ns in NOISE_SCALES]
    mc_totals = [baseline_mc] + [mc_results[f'SCALE_{ns}']['total'] for ns in NOISE_SCALES]

    # Rank by wave acc and MC total
    wave_ranks = np.argsort(-np.array(wave_accs))  # descending
    mc_ranks = np.argsort(-np.array(mc_totals))
    combined_rank = np.zeros(len(all_scales))
    for rank, idx in enumerate(wave_ranks):
        combined_rank[idx] += rank
    for rank, idx in enumerate(mc_ranks):
        combined_rank[idx] += rank
    best_combined_idx = np.argmin(combined_rank)
    optimal_scale = all_scales[best_combined_idx]

    print(f"\n  OPTIMAL NOISE_SCALE = {optimal_scale:.3f}")
    print(f"    Best waveform scale: {best_wave_scale:.3f} ({best_wave_acc:.3f})")
    print(f"    Best MC scale:       {best_mc_scale:.3f} ({best_mc_total:.3f})")
    print(f"    Best combined:       {optimal_scale:.3f}")

    # Determine if noise helps at all
    noise_helps_wave = best_wave_acc > baseline_wave + 0.01
    noise_helps_mc = best_mc_total > baseline_mc + 0.05
    print(f"\n  Noise helps waveform?  {'YES' if noise_helps_wave else 'NO'} "
          f"(best {best_wave_acc:.3f} vs baseline {baseline_wave:.3f})")
    print(f"  Noise helps MC?        {'YES' if noise_helps_mc else 'NO'} "
          f"(best {best_mc_total:.3f} vs baseline {baseline_mc:.3f})")

    # ─── Save results ───
    all_results = {
        'experiment': 'z2269_noise_scale_sweep',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'params': {
            'N_FPGA': N_FPGA, 'BASE_VG': BASE_VG, 'VG_SPREAD': VG_SPREAD,
            'ALPHA': ALPHA, 'IIR_ALPHA': IIR_ALPHA, 'SAMPLE_HZ': SAMPLE_HZ,
            'N_WAVE_TRIALS': N_WAVE_TRIALS, 'N_WAVE_STEPS': N_WAVE_STEPS,
            'N_MC_STEPS': N_MC_STEPS, 'MC_MAX_DELAY': MC_MAX_DELAY,
            'NOISE_SCALES': NOISE_SCALES,
        },
        'noise_stats': noise_stats,
        'waveform_results': wave_results,
        'mc_results': mc_results,
        'analysis': {
            'baseline_wave': baseline_wave,
            'baseline_mc': baseline_mc,
            'best_wave_scale': best_wave_scale,
            'best_wave_acc': best_wave_acc,
            'best_mc_scale': best_mc_scale,
            'best_mc_total': best_mc_total,
            'optimal_scale': optimal_scale,
            'noise_helps_wave': noise_helps_wave,
            'noise_helps_mc': noise_helps_mc,
        },
    }

    # Save JSON
    json_path = RESULTS / 'z2269_noise_scale_sweep.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved JSON: {json_path}")

    # Save human-readable text
    txt_path = RESULTS / 'z2269_noise_scale_sweep.txt'
    with open(txt_path, 'w') as f:
        f.write("=" * 72 + "\n")
        f.write("z2269: Noise Scale Sweep — Optimal GPU→FPGA Injection Level\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 72 + "\n\n")

        f.write(f"New bitstream: LEAK_COND=4 (τ=210ms)\n")
        f.write(f"GPU noise: hwmon power1_average, IIR alpha={IIR_ALPHA}\n")
        f.write(f"Noise stats: mean={noise_stats['mean']:.2f}W, std={noise_stats['std']:.4f}W\n\n")

        f.write("WAVEFORM CLASSIFICATION (4-class, 60 trials, 5-fold CV)\n")
        f.write(f"{'Condition':<16} {'Accuracy':<12} {'Std':<10} {'Δ vs Baseline':<14}\n")
        f.write("-" * 52 + "\n")
        f.write(f"{'NO_NOISE':<16} {baseline_wave:<12.3f} {wave_results['NO_NOISE']['std']:<10.3f} {'---':<14}\n")
        for ns in NOISE_SCALES:
            key = f'SCALE_{ns}'
            w = wave_results[key]
            delta = w['accuracy'] - baseline_wave
            f.write(f"{'SCALE=' + str(ns):<16} {w['accuracy']:<12.3f} {w['std']:<10.3f} {delta:+.3f}\n")

        f.write(f"\nMEMORY CAPACITY (d=1-{MC_MAX_DELAY}, {N_MC_STEPS} steps)\n")
        f.write(f"{'Condition':<16} {'MC Total':<12} {'Δ vs Baseline':<14}\n")
        f.write("-" * 42 + "\n")
        f.write(f"{'NO_NOISE':<16} {baseline_mc:<12.3f} {'---':<14}\n")
        for ns in NOISE_SCALES:
            key = f'SCALE_{ns}'
            m = mc_results[key]
            delta = m['total'] - baseline_mc
            f.write(f"{'SCALE=' + str(ns):<16} {m['total']:<12.3f} {delta:+.3f}\n")

        f.write(f"\nANALYSIS\n")
        f.write(f"  Optimal NOISE_SCALE = {optimal_scale:.3f}\n")
        f.write(f"  Best waveform: scale={best_wave_scale:.3f}, acc={best_wave_acc:.3f}\n")
        f.write(f"  Best MC:       scale={best_mc_scale:.3f}, total={best_mc_total:.3f}\n")
        f.write(f"  Noise helps waveform: {'YES' if noise_helps_wave else 'NO'}\n")
        f.write(f"  Noise helps MC:       {'YES' if noise_helps_mc else 'NO'}\n")

    print(f"  Saved TXT: {txt_path}")
    print("\n  DONE.")

    fpga.close()


if __name__ == '__main__':
    main()
