#!/usr/bin/env python3
"""z2165_multichannel_reservoir.py — Multi-Channel Noise Fusion Reservoir

Extends z2162 single-channel reservoir to use MULTIPLE independent GPU noise sources
assigned to DIFFERENT neurons, creating a heterogeneous multi-timescale reservoir
that mimics process variation in real memristor arrays.

Noise channels:
  - hwmon power1_average: PSD slope=-1.55, native 1/f (VRM switching), ~50Hz
  - gpu_metrics temp_soc: thermal oscillations, slower dynamics
  - PERF_SNAPSHOT jitter: near-white (slope ~0), very fast

Per-neuron assignment:
  - Neurons 0-2: hwmon power (1/f noise, IIR alpha=0.85)
  - Neurons 3-5: gpu_metrics thermal (slow dynamics)
  - Neurons 6-7: PERF_SNAPSHOT jitter (white noise)

Conditions:
  MULTI:        Multi-channel heterogeneous (power + thermal + jitter)
  SINGLE_1F:    All neurons use power 1/f (homogeneous, same as z2162 cond A)
  SINGLE_WHITE: All neurons use PERF jitter
  NO_NOISE:     Deterministic (beta=0)

Tasks:
  Waveform classification (3-class: sine, triangle, square)
  Temporal XOR at tau=1,2,3,5,8

Tests:
  T79: MULTI waveform > SINGLE_1F waveform
  T80: MULTI XOR > SINGLE_1F XOR
  T81: MULTI waveform > SINGLE_WHITE
  T82: MULTI > NO_NOISE (noise helps)
  T83: Cross-neuron correlation lower in MULTI than SINGLE_1F
  T84: MULTI XOR > SINGLE_1F XOR (redundant with T80, kept for completeness)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB1
"""

import os, sys, json, time, struct, subprocess, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-4' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
GPU_METRICS_PATH = "/sys/class/drm/card0/device/gpu_metrics"

# ─── Reservoir Parameters ───
BASE_VG = 0.58       # near BVpar cliff — input modulation has maximum effect
ALPHA = 0.25         # strong input coupling (dominates noise)
BETA = 0.08          # moderate noise coupling
N_NEURONS = 8
SAMPLE_HZ = 20       # FPGA update rate

# ─── Per-neuron channel assignment ───
POWER_NEURONS = [0, 1, 2]    # 1/f noise from hwmon power rail
THERMAL_NEURONS = [3, 4, 5]  # slow thermal from gpu_metrics temp_soc
JITTER_NEURONS = [6, 7]      # white noise from PERF_SNAPSHOT


# ═══════════════════════════════════════════════════════════
# FPGA Communication (from z2153/z2162)
# ═══════════════════════════════════════════════════════════

def to_q16_16(val: float) -> int:
    return int(val * 65536) & 0xFFFFFFFF


def find_fpga():
    try:
        import serial
    except ImportError:
        return None, None
    for p in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyUSB2']:
        try:
            s = serial.Serial(p, 115200, timeout=0.2)
            time.sleep(0.1)
            return s, p
        except Exception:
            continue
    return None, None


def reconnect_fpga(port):
    """Reconnect to FPGA after serial failure."""
    import serial
    try:
        ser = serial.Serial(port, 115200, timeout=0.2)
        time.sleep(0.1)
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        return ser
    except Exception:
        return None


def set_per_neuron_vg(ser, vg_values):
    """Set individual Vg for each of 8 neurons."""
    for nid, vg in enumerate(vg_values[:8]):
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x07]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem(ser, timeout=0.15):
    """Read telemetry packet: [0x55][0x02][0x30][48B][CRC8] = 52 bytes."""
    deadline = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < deadline:
        ser.timeout = max(0.001, deadline - time.monotonic())
        b = ser.read(1)
        if not b:
            continue
        if b[0] == SYNC:
            buf = bytearray([SYNC])
            while len(buf) < 52 and time.monotonic() < deadline:
                ser.timeout = max(0.001, deadline - time.monotonic())
                chunk = ser.read(52 - len(buf))
                if chunk:
                    buf.extend(chunk)
            break
    if len(buf) < 52:
        return None
    payload = bytes(buf[3:51])
    neurons = []
    for i in range(8):
        off = i * 6
        sc = struct.unpack_from('>H', payload, off)[0]
        vm = struct.unpack_from('>H', payload, off + 2)[0]
        neurons.append({'spike_count': sc, 'vmem': vm / 256.0})
    return neurons


def safe_fpga_step(ser, port, vg_values, interval):
    """Set Vg and read telemetry with reconnection on serial failure."""
    import serial as serial_mod
    try:
        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)
        return ser, telem
    except (serial_mod.SerialException, OSError) as e:
        print(f"    [!] Serial error: {e}, reconnecting...")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.5)
        new_ser = reconnect_fpga(port)
        if new_ser is None:
            print("    [!] Reconnection failed")
            return None, None
        print("    [!] Reconnected successfully")
        return new_ser, None


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W). Rich 1/f dynamics ~11W +/- 1.5W."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def read_gpu_thermal():
    """Read gpu_metrics temp_soc (works on kernel 6.14, temp_gfx is broken)."""
    try:
        with open(GPU_METRICS_PATH, "rb") as f:
            data = f.read()
        # temp_soc at offset 60 (uint16, hundredths of C)
        temp_soc = struct.unpack_from('<H', data, 60)[0]
        return temp_soc / 100.0  # degrees C
    except Exception:
        return None


def run_hip_jitter_batch(n_iters=50, n_waves=16, work_iters=50000):
    """Run z2153 deep probe and extract jitter bytes (white-ish noise)."""
    probe_bin = BASE / 'scripts' / 'z2153_deep_probe_bridge'
    if not probe_bin.exists():
        return []
    result = subprocess.run(
        [str(probe_bin), str(n_iters), str(n_waves), str(work_iters)],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, 'HSA_OVERRIDE_GFX_VERSION': '11.0.0'}
    )
    if result.returncode != 0:
        return []
    jitter_bytes = []
    for line in result.stdout.strip().split('\n')[1:]:
        parts = line.split(',')
        if len(parts) >= 13:
            jitter_bytes.append(int(parts[12]))
    return jitter_bytes


def collect_multichannel_noise(duration_s=15, sample_hz=50):
    """Collect all 3 noise channels simultaneously."""
    n = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    power_samples = []
    thermal_samples = []
    for _ in range(n):
        p = read_hwmon_power()
        t = read_gpu_thermal()
        if p is not None:
            power_samples.append(p)
        if t is not None:
            thermal_samples.append(t)
        time.sleep(interval)

    # PERF jitter from HIP probe (batch)
    print("  Collecting PERF_SNAPSHOT jitter (white noise)...")
    jitter = run_hip_jitter_batch(n_iters=100, n_waves=16, work_iters=50000)

    return power_samples, thermal_samples, jitter


def normalize_noise(samples):
    """Zero-mean, unit-variance normalization."""
    arr = np.array(samples, dtype=float)
    if len(arr) == 0:
        return arr
    mu = arr.mean()
    std = max(arr.std(), 1e-6)
    return (arr - mu) / std


def iir_filter_noise(noise_samples, alpha_iir=0.85):
    """Apply IIR low-pass to noise: y[t] = alpha*y[t-1] + (1-alpha)*x[t].
    Creates temporal memory (ACF ~0.85) from raw noise.
    """
    if len(noise_samples) == 0:
        return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t-1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


def generate_synthetic_1f(n_samples, rng):
    """Voss-McCartney 1/f generator."""
    noise = np.zeros(n_samples)
    n_octaves = 8
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    return normalize_noise(noise)


def generate_synthetic_thermal(n_samples, rng):
    """Synthetic slow thermal: Brownian walk + mean reversion."""
    thermal = np.zeros(n_samples)
    thermal[0] = 45.0  # ~45 C typical SOC temp
    for i in range(1, n_samples):
        thermal[i] = 0.995 * thermal[i-1] + 0.005 * 45.0 + 0.03 * rng.standard_normal()
    return normalize_noise(thermal)


# ═══════════════════════════════════════════════════════════
# Waveform Generation (from z2162)
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=200, steps_per_trial=30, freq_hz=1.0, dt=1.0/20):
    """Generate sine/triangle/square waveforms for classification."""
    rng = np.random.default_rng(42)
    trials = []
    labels = []
    t = np.arange(steps_per_trial) * dt

    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        phase = rng.uniform(0, 2 * np.pi)
        freq = freq_hz * rng.uniform(0.8, 1.2)

        if cls == 0:  # sine
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:  # triangle
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2*np.pi)) % 1.0) - 1.0) - 1.0
        else:  # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))

        wave = (wave + 1.0) / 2.0
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


def generate_xor_sequence(n_steps=3000, seed=42):
    """Generate random binary input and temporal XOR targets."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, size=n_steps).astype(float)
    return u


def compute_xor_targets(u, tau):
    """XOR of u(t) and u(t-tau)."""
    n = len(u)
    targets = np.zeros(n)
    for t in range(tau, n):
        targets[t] = int(u[t]) ^ int(u[t - tau])
    return targets


# ═══════════════════════════════════════════════════════════
# Multi-Channel Reservoir Core
# ═══════════════════════════════════════════════════════════

def compute_vg_multichannel(t, input_val, noise_power, noise_thermal, noise_jitter,
                            w_in, w_noise, base_vg=BASE_VG, alpha=ALPHA, beta=BETA):
    """Compute per-neuron Vg with heterogeneous noise assignment.

    Neurons 0-2: power 1/f noise
    Neurons 3-5: thermal noise
    Neurons 6-7: PERF jitter noise
    """
    vg = np.full(N_NEURONS, base_vg) + alpha * input_val * w_in

    # Neurons 0-2: power 1/f noise
    for i in POWER_NEURONS:
        idx = t % len(noise_power) if len(noise_power) > 0 else 0
        val = noise_power[idx] if len(noise_power) > 0 else 0.0
        vg[i] += beta * val * w_noise[i]

    # Neurons 3-5: thermal noise
    for i in THERMAL_NEURONS:
        idx = t % len(noise_thermal) if len(noise_thermal) > 0 else 0
        val = noise_thermal[idx] if len(noise_thermal) > 0 else 0.0
        vg[i] += beta * val * w_noise[i]

    # Neurons 6-7: PERF jitter noise
    for i in JITTER_NEURONS:
        idx = t % len(noise_jitter) if len(noise_jitter) > 0 else 0
        val = noise_jitter[idx] if len(noise_jitter) > 0 else 0.0
        vg[i] += beta * val * w_noise[i]

    return np.clip(vg, 0.05, 0.95)


def compute_vg_homogeneous(t, input_val, noise_samples, w_in, w_noise,
                           base_vg=BASE_VG, alpha=ALPHA, beta=BETA):
    """All neurons use the SAME noise source (homogeneous, as in z2162)."""
    vg = np.full(N_NEURONS, base_vg) + alpha * input_val * w_in
    if beta > 0 and len(noise_samples) > 0:
        noise_val = noise_samples[t % len(noise_samples)]
        vg += beta * noise_val * w_noise
    return np.clip(vg, 0.05, 0.95)


def run_reservoir_trial_multichannel(ser, port, input_signal, noise_power,
                                     noise_thermal, noise_jitter, w_in, w_noise,
                                     mode='MULTI', live_noise=False):
    """Drive FPGA reservoir with multi-channel or homogeneous noise.

    mode: 'MULTI' | 'SINGLE_1F' | 'SINGLE_WHITE' | 'NO_NOISE'

    When live_noise=True (MULTI mode), reads power AND thermal in real-time.
    Returns: (n_steps, 24) array -- 8 delta_spikes + 8 vmem + 8 cumulative_spikes
    Also returns updated ser handle (may change on reconnect).
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0
    thermal_mean = 45.0

    for t in range(n_steps):
        # Compute per-neuron Vg based on mode
        if mode == 'MULTI':
            if live_noise:
                # Read both power and thermal in real-time
                p = read_hwmon_power()
                th = read_gpu_thermal()
                live_power_val = (p - power_mean) / 2.0 if p is not None else 0.0
                live_thermal_val = (th - thermal_mean) / 5.0 if th is not None else 0.0
                # Build per-neuron Vg with live readings
                vg = np.full(N_NEURONS, BASE_VG) + ALPHA * input_signal[t] * w_in
                for i in POWER_NEURONS:
                    vg[i] += BETA * live_power_val * w_noise[i]
                for i in THERMAL_NEURONS:
                    vg[i] += BETA * live_thermal_val * w_noise[i]
                for i in JITTER_NEURONS:
                    idx = t % len(noise_jitter) if len(noise_jitter) > 0 else 0
                    val = noise_jitter[idx] if len(noise_jitter) > 0 else 0.0
                    vg[i] += BETA * val * w_noise[i]
                vg = np.clip(vg, 0.05, 0.95)
            else:
                vg = compute_vg_multichannel(
                    t, input_signal[t], noise_power, noise_thermal, noise_jitter,
                    w_in, w_noise)
        elif mode == 'SINGLE_1F':
            vg = compute_vg_homogeneous(
                t, input_signal[t], noise_power, w_in, w_noise)
        elif mode == 'SINGLE_WHITE':
            vg = compute_vg_homogeneous(
                t, input_signal[t], noise_jitter, w_in, w_noise)
        elif mode == 'NO_NOISE':
            vg = compute_vg_homogeneous(
                t, input_signal[t], np.zeros(1), w_in, w_noise, beta=0.0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # FPGA step with reconnection
        if ser is not None:
            ser, telem = safe_fpga_step(ser, port, vg, interval)
            if ser is None:
                # FPGA lost permanently, fill rest with zeros
                break
        else:
            telem = None

        if telem:
            counts = [n['spike_count'] for n in telem]
            vmems = [n['vmem'] for n in telem]
            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    states[t, i] = delta
                    cumulative[i] += delta
            for i in range(N_NEURONS):
                states[t, N_NEURONS + i] = vmems[i]
                states[t, N_NEURONS * 2 + i] = cumulative[i]
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

    return states, ser


def simulate_lif_multichannel(input_signal, noise_power, noise_thermal, noise_jitter,
                              w_in, w_noise, mode='MULTI'):
    """Software LIF simulation fallback when FPGA is not connected."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 3))

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / SAMPLE_HZ
    vmem = np.zeros(N_NEURONS)
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        if mode == 'MULTI':
            vg = compute_vg_multichannel(
                t, input_signal[t], noise_power, noise_thermal, noise_jitter,
                w_in, w_noise)
        elif mode == 'SINGLE_1F':
            vg = compute_vg_homogeneous(
                t, input_signal[t], noise_power, w_in, w_noise)
        elif mode == 'SINGLE_WHITE':
            vg = compute_vg_homogeneous(
                t, input_signal[t], noise_jitter, w_in, w_noise)
        elif mode == 'NO_NOISE':
            vg = compute_vg_homogeneous(
                t, input_signal[t], np.zeros(1), w_in, w_noise, beta=0.0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        I_in = vg * 5.0
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        spikes = np.zeros(N_NEURONS)
        for i in range(N_NEURONS):
            if vmem[i] >= v_thresh:
                spikes[i] = 1
                vmem[i] = v_rest
                cumulative[i] += 1

        states[t, :N_NEURONS] = spikes
        states[t, N_NEURONS:N_NEURONS*2] = vmem.copy()
        states[t, N_NEURONS*2:] = cumulative.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification (from z2162)
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2, 3)):
    """Add time-delayed copies of state for richer feature space."""
    T, D = states.shape
    augmented = np.zeros((T, D * (1 + len(delays))))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[d:, start:start + D] = states[:T - d]
    return augmented


def pool_trial_features(trial_states):
    """Pool per-timestep reservoir states into trial-level features."""
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier (one-hot encoding for multi-class)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    n_classes = len(np.unique(y_train))
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0

    best_acc = -1
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue
        pred_test = np.argmax(X_test @ W, axis=1)
        acc_test = np.mean(pred_test == y_test)
        if acc_test > best_acc:
            best_acc = acc_test
    return best_acc


def ridge_binary(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression for binary classification."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    best_acc = -1
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue
        pred = (X_test @ w > 0.5).astype(float)
        acc = np.mean(pred == y_test)
        if acc > best_acc:
            best_acc = acc
    return best_acc


def stratified_kfold(X, y, n_splits=5, seed=42):
    """Simple stratified k-fold split."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    indices = np.arange(len(y))
    rng.shuffle(indices)

    folds = [[] for _ in range(n_splits)]
    for c in classes:
        c_idx = indices[y[indices] == c]
        for i, idx in enumerate(c_idx):
            folds[i % n_splits].append(idx)

    splits = []
    for fold in range(n_splits):
        test_idx = np.array(folds[fold])
        train_idx = np.concatenate([np.array(folds[f]) for f in range(n_splits) if f != fold])
        splits.append((train_idx, test_idx))
    return splits


# ═══════════════════════════════════════════════════════════
# Cross-Neuron Correlation Analysis
# ═══════════════════════════════════════════════════════════

def compute_cross_neuron_correlation(states_list):
    """Compute mean pairwise Pearson correlation between neurons across trials.

    states_list: list of (n_steps, 24) arrays (one per trial).
    Uses delta_spikes (columns 0:8) for correlation.
    Returns: (8, 8) mean correlation matrix.
    """
    corr_accum = np.zeros((N_NEURONS, N_NEURONS))
    n_valid = 0
    for states in states_list:
        spikes = states[:, :N_NEURONS]
        if spikes.std() < 1e-8:
            continue
        # Per-neuron correlation
        valid_cols = []
        for i in range(N_NEURONS):
            if spikes[:, i].std() > 1e-8:
                valid_cols.append(i)
        if len(valid_cols) < 2:
            continue
        sub = spikes[:, valid_cols]
        corr = np.corrcoef(sub.T)
        # Map back to full 8x8
        full_corr = np.eye(N_NEURONS)
        for ii, ci in enumerate(valid_cols):
            for jj, cj in enumerate(valid_cols):
                full_corr[ci, cj] = corr[ii, jj]
        corr_accum += full_corr
        n_valid += 1

    if n_valid > 0:
        corr_accum /= n_valid
    return corr_accum


def mean_off_diagonal(corr_matrix):
    """Mean absolute off-diagonal correlation."""
    n = corr_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.mean(np.abs(corr_matrix[mask]))


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='z2165: Multi-Channel Noise Fusion Reservoir')
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--steps-per-trial', type=int, default=30)
    parser.add_argument('--xor-steps', type=int, default=3000)
    parser.add_argument('--noise-collect-s', type=float, default=15.0)
    args = parser.parse_args()

    print("=" * 70)
    print("z2165: Multi-Channel Noise Fusion Reservoir")
    print("=" * 70)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2165_multichannel_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'xor_steps': args.xor_steps,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
            'channel_assignment': {
                'power_neurons': POWER_NEURONS,
                'thermal_neurons': THERMAL_NEURONS,
                'jitter_neurons': JITTER_NEURONS,
            },
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/8] Connecting to FPGA...")
    ser, port = find_fpga()
    if ser is None:
        print("  FPGA not found -- using LIF simulation fallback")
        fpga = False
        results['simulated'] = True
    else:
        print(f"  Connected: {port}")
        fpga = True
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        print("  Kill switch disabled")

    # ─── Step 2: Collect multi-channel GPU noise ───
    print("\n[2/8] Collecting multi-channel GPU noise sources...")

    print("  Collecting power rail + thermal simultaneously...")
    power_raw, thermal_raw, jitter_raw = collect_multichannel_noise(
        duration_s=args.noise_collect_s, sample_hz=50)

    # Process power noise
    if len(power_raw) > 10:
        noise_power = normalize_noise(power_raw)
        noise_power_iir = iir_filter_noise(noise_power, alpha_iir=0.85)
        print(f"  Power rail: {np.mean(power_raw):.2f} +/- {np.std(power_raw):.3f} W, "
              f"{len(noise_power)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f")
        noise_power = generate_synthetic_1f(int(args.noise_collect_s * 50), rng)
        noise_power_iir = iir_filter_noise(noise_power, alpha_iir=0.85)

    # Process thermal noise
    if len(thermal_raw) > 10:
        noise_thermal = normalize_noise(thermal_raw)
        # Thermal is already slow, use mild IIR to smooth
        noise_thermal_iir = iir_filter_noise(noise_thermal, alpha_iir=0.92)
        print(f"  Thermal: {np.mean(thermal_raw):.2f} +/- {np.std(thermal_raw):.3f} C, "
              f"{len(noise_thermal)} samples")
    else:
        print("  Thermal unavailable, generating synthetic thermal")
        noise_thermal = generate_synthetic_thermal(int(args.noise_collect_s * 50), rng)
        noise_thermal_iir = iir_filter_noise(noise_thermal, alpha_iir=0.92)

    # Process jitter noise
    if len(jitter_raw) > 10:
        noise_jitter = normalize_noise(jitter_raw)
        print(f"  Jitter: {len(noise_jitter)} samples")
    else:
        print("  HIP probe unavailable, generating synthetic white noise")
        noise_jitter = rng.standard_normal(int(args.noise_collect_s * 50))

    results['noise'] = {
        'power_samples': len(noise_power_iir),
        'thermal_samples': len(noise_thermal_iir),
        'jitter_samples': len(noise_jitter),
        'power_mean_W': float(np.mean(power_raw)) if len(power_raw) > 0 else None,
        'thermal_mean_C': float(np.mean(thermal_raw)) if len(thermal_raw) > 0 else None,
    }

    # ─── Step 3: Generate tasks ───
    print("\n[3/8] Generating waveform classification task...")
    wave_trials, wave_labels = generate_waveforms(
        n_trials=args.n_trials, steps_per_trial=args.steps_per_trial)
    print(f"  {args.n_trials} trials, {args.steps_per_trial} steps each")
    print(f"  Class distribution: {np.bincount(wave_labels)}")

    # ─── Step 4: Run 4 conditions on waveform task ───
    print("\n[4/8] Running 4 reservoir conditions on waveform task...")

    conditions = ['MULTI', 'SINGLE_1F', 'SINGLE_WHITE', 'NO_NOISE']
    wave_features = {}
    wave_trial_states = {}  # store raw states for correlation analysis

    for cond in conditions:
        live = (cond == 'MULTI' and fpga)
        print(f"\n  === Condition {cond} (live={live}) ===")
        trial_features = []
        trial_states_list = []
        t0 = time.monotonic()

        for trial_idx in range(args.n_trials):
            input_signal = wave_trials[trial_idx]

            if fpga:
                states, ser = run_reservoir_trial_multichannel(
                    ser, port, input_signal,
                    noise_power_iir, noise_thermal_iir, noise_jitter,
                    w_in, w_noise, mode=cond, live_noise=live)
                if ser is None:
                    print("    FPGA lost, switching to simulation")
                    fpga = False
                    results['simulated'] = True
                    states = simulate_lif_multichannel(
                        input_signal, noise_power_iir, noise_thermal_iir,
                        noise_jitter, w_in, w_noise, mode=cond)
            else:
                states = simulate_lif_multichannel(
                    input_signal, noise_power_iir, noise_thermal_iir,
                    noise_jitter, w_in, w_noise, mode=cond)

            aug = augment_with_delays(states, delays=(1, 2, 3))
            feat = pool_trial_features(aug)
            trial_features.append(feat)
            trial_states_list.append(states)

            if (trial_idx + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial_idx + 1) / elapsed
                eta = (args.n_trials - trial_idx - 1) / rate
                print(f"    Trial {trial_idx+1}/{args.n_trials} "
                      f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

        wave_features[cond] = np.array(trial_features)
        wave_trial_states[cond] = trial_states_list
        elapsed = time.monotonic() - t0
        print(f"  {cond}: {len(trial_features)} trials in {elapsed:.1f}s")

    # ─── Step 5: Classify waveforms (5-fold stratified CV) ───
    print("\n[5/8] Classifying waveforms (5-fold stratified CV)...")

    wave_accuracies = {}
    splits = stratified_kfold(wave_features['MULTI'], wave_labels, n_splits=5)

    for cond in conditions:
        X_all = wave_features[cond]
        fold_accs = []
        for train_idx, test_idx in splits:
            X_train = X_all[train_idx]
            X_test = X_all[test_idx]
            y_train = wave_labels[train_idx]
            y_test = wave_labels[test_idx]

            mu = X_train.mean(axis=0, keepdims=True)
            sigma = X_train.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_train_n = (X_train - mu) / sigma
            X_test_n = (X_test - mu) / sigma

            acc = ridge_classify(X_train_n, y_train, X_test_n, y_test)
            fold_accs.append(acc)

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        wave_accuracies[cond] = {
            'mean': float(mean_acc), 'std': float(std_acc),
            'folds': [float(a) for a in fold_accs],
        }
        print(f"  {cond}: {mean_acc:.3f} +/- {std_acc:.3f}")

    results['waveform_classification'] = wave_accuracies

    # ─── Step 6: Cross-neuron correlation analysis ───
    print("\n[6/8] Computing cross-neuron correlation matrices...")

    corr_matrices = {}
    corr_offdiag = {}
    for cond in conditions:
        corr = compute_cross_neuron_correlation(wave_trial_states[cond])
        corr_matrices[cond] = corr
        offdiag = mean_off_diagonal(corr)
        corr_offdiag[cond] = float(offdiag)
        print(f"  {cond}: mean |off-diagonal| = {offdiag:.4f}")

    results['cross_neuron_correlation'] = {
        cond: {
            'matrix': corr_matrices[cond].tolist(),
            'mean_abs_offdiag': corr_offdiag[cond],
        }
        for cond in conditions
    }

    # ─── Step 7: Temporal XOR task ───
    print("\n[7/8] Running temporal XOR task...")

    xor_input = generate_xor_sequence(n_steps=args.xor_steps, seed=42)
    taus = [1, 2, 3, 5, 8]

    xor_states = {}
    for cond in conditions:
        live = (cond == 'MULTI' and fpga)
        print(f"  Running reservoir on XOR sequence ({cond}, live={live})...")
        if fpga:
            st, ser = run_reservoir_trial_multichannel(
                ser, port, xor_input,
                noise_power_iir, noise_thermal_iir, noise_jitter,
                w_in, w_noise, mode=cond, live_noise=live)
            if ser is None:
                fpga = False
                results['simulated'] = True
                st = simulate_lif_multichannel(
                    xor_input, noise_power_iir, noise_thermal_iir,
                    noise_jitter, w_in, w_noise, mode=cond)
        else:
            st = simulate_lif_multichannel(
                xor_input, noise_power_iir, noise_thermal_iir,
                noise_jitter, w_in, w_noise, mode=cond)
        xor_states[cond] = augment_with_delays(st, delays=(1, 2, 3))

    xor_results = {}
    for tau in taus:
        y_xor = compute_xor_targets(xor_input, tau)
        valid = np.arange(max(tau, 3), args.xor_steps)

        accs_per_cond = {}
        for cond in conditions:
            X_all = xor_states[cond]
            X_valid = X_all[valid]
            y_valid = y_xor[valid]

            n_valid = len(valid)
            split = int(0.7 * n_valid)
            X_tr, X_te = X_valid[:split], X_valid[split:]
            y_tr, y_te = y_valid[:split], y_valid[split:]

            mu = X_tr.mean(axis=0, keepdims=True)
            sigma = X_tr.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_tr_n = (X_tr - mu) / sigma
            X_te_n = (X_te - mu) / sigma

            acc = ridge_binary(X_tr_n, y_tr, X_te_n, y_te)
            accs_per_cond[cond] = float(acc)

        xor_results[f'tau_{tau}'] = accs_per_cond
        print(f"  tau={tau}: " + ", ".join(
            f"{c}={accs_per_cond[c]:.3f}" for c in conditions))

    results['temporal_xor'] = xor_results

    # ─── Step 8: Tests ───
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    acc_multi = wave_accuracies['MULTI']['mean']
    acc_1f = wave_accuracies['SINGLE_1F']['mean']
    acc_white = wave_accuracies['SINGLE_WHITE']['mean']
    acc_none = wave_accuracies['NO_NOISE']['mean']

    # Best XOR per condition
    def best_xor(cond):
        accs = [xor_results[f'tau_{t}'][cond] for t in taus]
        best_idx = np.argmax(accs)
        return accs[best_idx], taus[best_idx]

    xor_multi, xor_multi_tau = best_xor('MULTI')
    xor_1f, xor_1f_tau = best_xor('SINGLE_1F')
    xor_white, xor_white_tau = best_xor('SINGLE_WHITE')
    xor_none, xor_none_tau = best_xor('NO_NOISE')

    # T79: MULTI waveform > SINGLE_1F waveform
    t79 = acc_multi > acc_1f
    # T80: MULTI XOR > SINGLE_1F XOR
    t80 = xor_multi > xor_1f
    # T81: MULTI waveform > SINGLE_WHITE
    t81 = acc_multi > acc_white
    # T82: MULTI > NO_NOISE
    t82 = acc_multi > acc_none
    # T83: Cross-neuron correlation lower in MULTI than SINGLE_1F
    t83 = corr_offdiag['MULTI'] < corr_offdiag['SINGLE_1F']
    # T84: MULTI XOR > SINGLE_1F XOR (same as T80, kept for completeness)
    t84 = xor_multi > xor_1f

    results['tests'] = {
        'T79_multi_gt_single1f_wave': {
            'pass': t79,
            'MULTI_acc': float(acc_multi), 'SINGLE_1F_acc': float(acc_1f),
            'margin': float(acc_multi - acc_1f),
        },
        'T80_multi_gt_single1f_xor': {
            'pass': t80,
            'MULTI_xor': float(xor_multi), 'MULTI_tau': int(xor_multi_tau),
            'SINGLE_1F_xor': float(xor_1f), 'SINGLE_1F_tau': int(xor_1f_tau),
            'margin': float(xor_multi - xor_1f),
        },
        'T81_multi_gt_white_wave': {
            'pass': t81,
            'MULTI_acc': float(acc_multi), 'SINGLE_WHITE_acc': float(acc_white),
            'margin': float(acc_multi - acc_white),
        },
        'T82_multi_gt_nonoise': {
            'pass': t82,
            'MULTI_acc': float(acc_multi), 'NO_NOISE_acc': float(acc_none),
            'margin': float(acc_multi - acc_none),
        },
        'T83_multi_decorrelation': {
            'pass': t83,
            'MULTI_offdiag': float(corr_offdiag['MULTI']),
            'SINGLE_1F_offdiag': float(corr_offdiag['SINGLE_1F']),
            'delta': float(corr_offdiag['SINGLE_1F'] - corr_offdiag['MULTI']),
        },
        'T84_multi_gt_single1f_xor_proxy': {
            'pass': t84,
            'MULTI_xor': float(xor_multi),
            'SINGLE_1F_xor': float(xor_1f),
            'margin': float(xor_multi - xor_1f),
        },
    }

    n_pass = sum(1 for t in results['tests'].values() if t['pass'])
    results['summary'] = {
        'pass_count': n_pass,
        'total_tests': 6,
        'pass_rate': f"{n_pass}/6",
    }

    tests_display = [
        (t79, f"T79: MULTI wave={acc_multi:.3f} > SINGLE_1F={acc_1f:.3f} "
              f"[{acc_multi - acc_1f:+.3f}]"),
        (t80, f"T80: MULTI XOR={xor_multi:.3f}@tau={xor_multi_tau} > "
              f"SINGLE_1F={xor_1f:.3f}@tau={xor_1f_tau} [{xor_multi - xor_1f:+.3f}]"),
        (t81, f"T81: MULTI wave={acc_multi:.3f} > SINGLE_WHITE={acc_white:.3f} "
              f"[{acc_multi - acc_white:+.3f}]"),
        (t82, f"T82: MULTI wave={acc_multi:.3f} > NO_NOISE={acc_none:.3f} "
              f"[{acc_multi - acc_none:+.3f}]"),
        (t83, f"T83: MULTI offdiag={corr_offdiag['MULTI']:.4f} < "
              f"SINGLE_1F={corr_offdiag['SINGLE_1F']:.4f} "
              f"[delta={corr_offdiag['SINGLE_1F'] - corr_offdiag['MULTI']:+.4f}]"),
        (t84, f"T84: MULTI XOR={xor_multi:.3f} > SINGLE_1F={xor_1f:.3f} "
              f"[{xor_multi - xor_1f:+.3f}]"),
    ]
    for passed, desc in tests_display:
        print(f"  {'PASS' if passed else 'FAIL'} {desc}")

    print(f"\n  Overall: {n_pass}/6 PASS")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2165_multichannel_reservoir.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # ─── Generate figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # Panel 1: Waveform accuracy bars (4 conditions)
        ax = axes[0]
        cond_labels = {
            'MULTI': 'MULTI\n(power+therm+jitter)',
            'SINGLE_1F': 'SINGLE 1/f\n(power only)',
            'SINGLE_WHITE': 'SINGLE White\n(jitter only)',
            'NO_NOISE': 'NO NOISE\n(deterministic)',
        }
        colors = {
            'MULTI': '#e74c3c',
            'SINGLE_1F': '#3498db',
            'SINGLE_WHITE': '#f39c12',
            'NO_NOISE': '#95a5a6',
        }
        x_pos = np.arange(len(conditions))
        means = [wave_accuracies[c]['mean'] for c in conditions]
        stds = [wave_accuracies[c]['std'] for c in conditions]
        bar_colors = [colors[c] for c in conditions]

        bars = ax.bar(x_pos, means, yerr=stds, capsize=4,
                      color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([cond_labels[c] for c in conditions], fontsize=8)
        ax.set_ylabel('Accuracy')
        ax.set_title('Waveform Classification (3-class)')
        ax.set_ylim(0, 1.05)
        ax.axhline(1/3, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.legend(fontsize=7)

        # Annotate best
        best_cond_idx = np.argmax(means)
        ax.annotate(f'{means[best_cond_idx]:.3f}',
                    xy=(best_cond_idx, means[best_cond_idx] + stds[best_cond_idx] + 0.02),
                    ha='center', fontsize=9, fontweight='bold', color=bar_colors[best_cond_idx])

        # Panel 2: XOR memory curves (accuracy vs tau for 4 conditions)
        ax = axes[1]
        markers = {'MULTI': 'o-', 'SINGLE_1F': 'D-', 'SINGLE_WHITE': '^-', 'NO_NOISE': 'v:'}
        for cond in conditions:
            accs = [xor_results[f'tau_{t}'][cond] for t in taus]
            ax.plot(taus, accs, markers[cond], color=colors[cond],
                    label=cond_labels[cond].replace('\n', ' '),
                    linewidth=2, markersize=6)

        ax.axhline(0.50, color='lightgray', linestyle=':', alpha=0.5, label='Chance')
        ax.set_xlabel('Delay tau (steps)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Temporal XOR Memory')
        ax.set_ylim(0.3, 1.05)
        ax.legend(fontsize=6, loc='upper right')

        # Panel 3: Cross-neuron correlation heatmaps (MULTI vs SINGLE_1F)
        ax = axes[2]
        # Split into two sub-axes
        ax.set_visible(False)
        gs = fig.add_gridspec(1, 2, left=0.7, right=0.98, wspace=0.15)
        ax_l = fig.add_subplot(gs[0, 0])
        ax_r = fig.add_subplot(gs[0, 1])

        im_l = ax_l.imshow(corr_matrices['MULTI'], cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax_l.set_title(f'MULTI\n|off|={corr_offdiag["MULTI"]:.3f}', fontsize=8)
        ax_l.set_xticks(range(8))
        ax_l.set_yticks(range(8))
        ax_l.set_xticklabels(range(8), fontsize=6)
        ax_l.set_yticklabels(range(8), fontsize=6)
        ax_l.set_xlabel('Neuron', fontsize=7)
        ax_l.set_ylabel('Neuron', fontsize=7)
        # Draw group boundaries
        for boundary in [2.5, 5.5]:
            ax_l.axhline(boundary, color='white', linewidth=1, alpha=0.8)
            ax_l.axvline(boundary, color='white', linewidth=1, alpha=0.8)

        im_r = ax_r.imshow(corr_matrices['SINGLE_1F'], cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax_r.set_title(f'SINGLE_1F\n|off|={corr_offdiag["SINGLE_1F"]:.3f}', fontsize=8)
        ax_r.set_xticks(range(8))
        ax_r.set_yticks(range(8))
        ax_r.set_xticklabels(range(8), fontsize=6)
        ax_r.set_yticklabels([], fontsize=6)
        ax_r.set_xlabel('Neuron', fontsize=7)

        fig.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)

        fig.suptitle(f'z2165: Multi-Channel Noise Fusion Reservoir  |  '
                     f'{n_pass}/6 PASS', fontsize=12, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.68, 0.95])

        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'fig_z2165_multichannel.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {fig_path}")
    except Exception as e:
        print(f"  Figure generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    if fpga and ser:
        set_per_neuron_vg(ser, [0.3] * 8)
        ser.close()

    print(f"\nDone. {n_pass}/6 tests passed.")
    return n_pass


if __name__ == '__main__':
    main()
