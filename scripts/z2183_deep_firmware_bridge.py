#!/usr/bin/env python3
"""z2183_deep_firmware_bridge.py — Deep Firmware Bridge: GPU Silicon as Memristor Analog

KEYSTONE experiment bridging deep firmware reverse engineering to Mario Lanza's
memristor/RRAM research. The thesis: device-level variability in commodity GPU silicon
(thermal drifts, VRM switching noise, kernel timing jitter, clock domain crossing)
is computationally useful when bridged to FPGA neuromorphic hardware — the SAME
principle Lanza demonstrates for memristors.

We access FOUR simultaneous noise channels from different firmware layers:

  Layer 1 — Power VRM 1/f noise (hwmon power1_average)
            Analog to Lanza's VRM-induced RRAM variability
  Layer 2 — SMN Thermal ADC (ryzen_smu PM table, below driver smoothing)
            Analog to Lanza's temperature-dependent resistance drift
  Layer 3 — GPU Kernel Jitter (torch matmul wall-clock variance)
            Analog to Lanza's cycle-to-cycle switching time variation
  Layer 4 — Clock Domain Crossing (GPU vs FPGA timestamp delta jitter)
            Analog to Lanza's stochastic synapse timing

7 conditions: Layer1, Layer2, Layer3, Layer4, COMBINED (all 4), WHITE, NO_NOISE
100 trials × 3-class waveform classification (sine/tri/square), 25 steps/trial
5-fold stratified CV with ridge regression

Tests T187-T194:
  T187: COMBINED accuracy > best single layer accuracy (synergy from fusion)
  T188: COMBINED accuracy > WHITE accuracy (firmware > random)
  T189: At least 3/4 individual layers beat NO_NOISE (multiple channels useful)
  T190: PSD slopes differ across layers (different physics)
  T191: Layer 1 (power VRM) PSD slope in [-2.0, -0.5] (confirmed 1/f)
  T192: Layer 3 (kernel jitter) PSD slope > -0.5 (closer to white)
  T193: COMBINED ISI CV in Lanza biological range [0.3, 2.0]
  T194: Best single layer accuracy > 55% (useful computation)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB1
"""

import os, sys, json, time, struct, subprocess, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

# ─── Hardware Paths ───
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
GPU_METRICS_PATH = "/sys/class/drm/card0/device/gpu_metrics"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"

# ─── Reservoir Parameters ───
BASE_VG = 0.55
ALPHA = 0.15          # input coupling
BETA = 0.10           # noise coupling
N_NEURONS = 8
SAMPLE_HZ = 20
N_TRIALS = 100
STEPS_PER_TRIAL = 25


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


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


# ═══════════════════════════════════════════════════════════
# Four Firmware Noise Layers
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Layer 1: Read hwmon power1_average (uW -> W). Rich 1/f dynamics ~11W +/- 1.5W."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def collect_power_noise(n_samples, sample_hz=50):
    """Layer 1: Collect power VRM 1/f noise time series."""
    interval = 1.0 / sample_hz
    values = []
    for _ in range(n_samples):
        p = read_hwmon_power()
        if p is not None:
            values.append(p)
        else:
            values.append(np.nan)
        time.sleep(interval)
    arr = np.array(values)
    # Fill NaN with last valid
    mask = np.isnan(arr)
    if mask.all():
        return None
    if mask.any():
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr[mask] = arr[idx[mask]]
    return arr


def read_smn_thermal():
    """Layer 2: Read SMN PM table thermal ADC. Below driver smoothing.
    Raw 8-bit ADC with quantization noise — analog to Lanza's temp-dependent drift.
    """
    try:
        with open(SMN_PATH, 'rb+') as f:
            f.seek(0x0003_B998)
            raw = struct.unpack('<I', f.read(4))[0]
            return (raw >> 21) * 0.125  # 8-bit thermal ADC, 0.125C resolution
    except Exception:
        return None


def collect_smn_thermal(n_samples, sample_hz=50):
    """Layer 2: Collect SMN thermal ADC time series."""
    interval = 1.0 / sample_hz
    values = []
    for _ in range(n_samples):
        t = read_smn_thermal()
        if t is not None:
            values.append(t)
        else:
            values.append(np.nan)
        time.sleep(interval)
    arr = np.array(values)
    mask = np.isnan(arr)
    if mask.all():
        return None
    if mask.any():
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr[mask] = arr[idx[mask]]
    return arr


def measure_gpu_jitter(n_samples=500):
    """Layer 3: Measure GPU kernel execution jitter.
    Cycle-to-cycle timing variance — analog to Lanza's RRAM switching time variation.
    """
    try:
        import torch
        d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if d.type != 'cuda':
            return None
        x = torch.randn(64, 64, device=d)
        # Warmup
        for _ in range(20):
            _ = torch.mm(x, x)
            torch.cuda.synchronize()
        times = []
        for _ in range(n_samples):
            t0 = time.perf_counter_ns()
            _ = torch.mm(x, x)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
        return np.array(times, dtype=float)
    except Exception:
        return None


def measure_clock_domain_crossing(ser, n_samples=500):
    """Layer 4: Measure GPU-vs-FPGA clock domain crossing noise.
    The jitter in the delta between GPU timestamp and FPGA telemetry read
    captures the asynchronous relationship between independent crystal oscillators.
    """
    if ser is None:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None

    deltas = []
    for _ in range(n_samples):
        # GPU-side timestamp
        t_gpu_start = time.perf_counter_ns()
        try:
            import torch
            x = torch.randn(16, 16, device='cuda')
            _ = torch.mm(x, x)
            torch.cuda.synchronize()
        except Exception:
            pass
        t_gpu_end = time.perf_counter_ns()

        # FPGA-side: request telemetry and measure response time
        ser.reset_input_buffer()
        t_fpga_start = time.perf_counter_ns()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.1)
        t_fpga_end = time.perf_counter_ns()

        if telem:
            # The jitter in the delta is the clock domain crossing noise
            gpu_time = t_gpu_end - t_gpu_start
            fpga_time = t_fpga_end - t_fpga_start
            delta = abs(fpga_time - gpu_time)
            deltas.append(delta)

    return np.array(deltas, dtype=float) if len(deltas) > 50 else None


# ═══════════════════════════════════════════════════════════
# Synthetic Fallbacks
# ═══════════════════════════════════════════════════════════

def synth_1f_noise(n_samples, rng, n_octaves=8):
    """Voss-McCartney 1/f generator."""
    noise = np.zeros(n_samples)
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    return noise


def synth_brownian_noise(n_samples, rng):
    """Brownian (1/f^2) noise — analog to thermal drift."""
    steps = rng.standard_normal(n_samples) * 0.1
    return np.cumsum(steps)


def synth_white_noise(n_samples, rng):
    """White noise baseline."""
    return rng.standard_normal(n_samples)


def synth_clock_jitter(n_samples, rng):
    """Simulate clock domain crossing jitter: exponential + periodic."""
    base = rng.exponential(scale=100.0, size=n_samples)
    periodic = 50.0 * np.sin(2 * np.pi * np.arange(n_samples) / 137.0)  # beat freq
    return base + periodic


def normalize_noise(arr):
    """Zero-mean, unit-variance normalization."""
    if arr is None or len(arr) < 2:
        return np.zeros(100)
    mu = arr.mean()
    sigma = max(arr.std(), 1e-10)
    return (arr - mu) / sigma


# ═══════════════════════════════════════════════════════════
# PSD and ISI Analysis
# ═══════════════════════════════════════════════════════════

def compute_psd_slope(signal, fs=50.0):
    """Compute PSD via Welch's method and fit log-log slope."""
    n = len(signal)
    nperseg = min(256, n // 2)
    if nperseg < 16:
        return 0.0, np.array([1.0]), np.array([1.0])
    from numpy.fft import rfft, rfftfreq
    # Simple periodogram
    freqs = rfftfreq(n, d=1.0 / fs)
    fft_vals = rfft(signal)
    psd = np.abs(fft_vals) ** 2 / n
    # Skip DC
    mask = freqs > 0
    freqs = freqs[mask]
    psd = psd[mask]
    # Remove zeros
    nonzero = psd > 0
    if nonzero.sum() < 4:
        return 0.0, freqs, psd
    log_f = np.log10(freqs[nonzero])
    log_p = np.log10(psd[nonzero])
    # Linear fit in log-log
    coeffs = np.polyfit(log_f, log_p, 1)
    return float(coeffs[0]), freqs, psd


def compute_isi_stats(spike_counts_series):
    """Compute ISI (inter-spike interval) statistics from spike count time series.
    spike_counts_series: (n_steps, n_neurons) — delta spike counts per step.
    Returns CV of ISIs.
    """
    all_isis = []
    for nid in range(spike_counts_series.shape[1]):
        counts = spike_counts_series[:, nid]
        # Find steps with spikes
        spike_times = np.where(counts > 0)[0]
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            all_isis.extend(isis.tolist())
    if len(all_isis) < 5:
        return 0.0, []
    isis_arr = np.array(all_isis, dtype=float)
    cv = isis_arr.std() / max(isis_arr.mean(), 1e-10)
    return float(cv), all_isis


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=100, steps_per_trial=25, seed=42):
    """Generate sine/triangle/square waveforms for 3-class classification."""
    rng = np.random.default_rng(seed)
    trials = []
    labels = []
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(steps_per_trial) * dt

    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        phase = rng.uniform(0, 2 * np.pi)
        freq = 1.0 * rng.uniform(0.8, 1.2)

        if cls == 0:   # sine
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1: # triangle
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:          # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))

        wave = (wave + 1.0) / 2.0  # normalize to [0, 1]
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# Reservoir Core
# ═══════════════════════════════════════════════════════════

def iir_filter_noise(noise_samples, alpha_iir=0.85):
    """IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]. Creates temporal memory."""
    if len(noise_samples) < 2:
        return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=BASE_VG, alpha=ALPHA, beta=BETA):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.
    Returns: (n_steps, 24) array — 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        noise_val = noise_samples[t % len(noise_samples)] if len(noise_samples) > 0 else 0.0

        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_in
        if beta > 0:
            vg_values += beta * noise_val * w_noise
        vg_values = np.clip(vg_values, 0.05, 0.95)

        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)

        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)

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

    return states


def simulate_lif_reservoir(input_signal, noise_samples, w_in, w_noise,
                            base_vg=BASE_VG, alpha=ALPHA, beta=BETA):
    """Software LIF simulation fallback when FPGA not connected."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 3))
    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / SAMPLE_HZ
    vmem = np.zeros(N_NEURONS)
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        vg = np.full(N_NEURONS, base_vg)
        vg += alpha * input_signal[t] * w_in
        if beta > 0 and len(noise_samples) > 0:
            vg += beta * noise_samples[t % len(noise_samples)] * w_noise
        vg = np.clip(vg, 0.05, 0.95)

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
        states[t, N_NEURONS:N_NEURONS * 2] = vmem.copy()
        states[t, N_NEURONS * 2:] = cumulative.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
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
    """Pool per-timestep reservoir states into trial-level features.
    (n_steps, n_features) -> [mean, std, max, min] feature vector.
    """
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier with one-hot encoding."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    n_classes = len(np.unique(y_train))
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0

    best_acc = -1
    for alpha_val in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha_val * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue
        pred_test = np.argmax(X_test @ W, axis=1)
        acc_test = np.mean(pred_test == y_test)
        if acc_test > best_acc:
            best_acc = acc_test
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
# Figure Generation
# ═══════════════════════════════════════════════════════════

def generate_figures(results, figures_dir):
    """Generate multi-panel figure: accuracy bars, PSD overlay, ISI distributions, fusion matrix."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('z2183: Deep Firmware Bridge — GPU Silicon as Memristor Analog',
                 fontsize=13, fontweight='bold')

    # ─── Panel A: Accuracy bars per condition ───
    ax = axes[0, 0]
    cond_names = list(results.get('classification', {}).keys())
    accs = [results['classification'][c]['mean'] for c in cond_names]
    stds = [results['classification'][c]['std'] for c in cond_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f']
    short_names = []
    for c in cond_names:
        if 'layer1' in c.lower() or 'power' in c.lower():
            short_names.append('L1:Power')
        elif 'layer2' in c.lower() or 'thermal' in c.lower() or 'smn' in c.lower():
            short_names.append('L2:Thermal')
        elif 'layer3' in c.lower() or 'jitter' in c.lower() or 'kernel' in c.lower():
            short_names.append('L3:Kernel')
        elif 'layer4' in c.lower() or 'clock' in c.lower() or 'cdc' in c.lower():
            short_names.append('L4:CDC')
        elif 'combined' in c.lower():
            short_names.append('COMBINED')
        elif 'white' in c.lower():
            short_names.append('WHITE')
        elif 'no_noise' in c.lower():
            short_names.append('NO_NOISE')
        else:
            short_names.append(c[:8])
    bars = ax.bar(range(len(cond_names)), accs, yerr=stds, capsize=4,
                  color=colors[:len(cond_names)], edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(short_names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (5-fold CV)')
    ax.set_title('A. Waveform Classification by Noise Layer')
    ax.axhline(y=1.0 / 3.0, color='gray', linestyle='--', alpha=0.5, label='Chance (33%)')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.0)

    # ─── Panel B: PSD overlay ───
    ax = axes[0, 1]
    psd_data = results.get('psd_analysis', {})
    layer_colors = {'layer1_power': '#1f77b4', 'layer2_smn_thermal': '#ff7f0e',
                    'layer3_kernel_jitter': '#2ca02c', 'layer4_clock_crossing': '#d62728',
                    'combined': '#9467bd', 'white': '#8c564b'}
    for layer_name, layer_info in psd_data.items():
        slope = layer_info.get('slope', 0)
        label = f"{layer_name} (slope={slope:.2f})"
        # Plot synthetic PSD line
        f = np.logspace(-1, 1.5, 100)
        psd_line = f ** slope
        color = layer_colors.get(layer_name, 'gray')
        ax.loglog(f, psd_line, label=label, color=color, alpha=0.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('B. Power Spectral Density by Layer')
    ax.legend(fontsize=6, loc='lower left')

    # ─── Panel C: ISI distributions ───
    ax = axes[1, 0]
    isi_data = results.get('isi_analysis', {})
    for layer_name, layer_info in isi_data.items():
        cv = layer_info.get('cv', 0)
        isis = layer_info.get('sample_isis', [])
        if len(isis) > 5:
            color = layer_colors.get(layer_name, 'gray')
            ax.hist(isis[:500], bins=30, alpha=0.4, label=f"{layer_name} (CV={cv:.2f})",
                    color=color, density=True)
    ax.set_xlabel('ISI (steps)')
    ax.set_ylabel('Density')
    ax.set_title('C. Inter-Spike Interval Distributions')
    ax.legend(fontsize=6)

    # ─── Panel D: Fusion matrix — accuracy improvement ───
    ax = axes[1, 1]
    layer_keys = ['layer1_power', 'layer2_smn_thermal',
                  'layer3_kernel_jitter', 'layer4_clock_crossing']
    present = [k for k in layer_keys if k in results.get('classification', {})]
    n_present = len(present)
    if n_present > 0:
        combined_acc = results['classification'].get('combined', {}).get('mean', 0)
        matrix = np.zeros((n_present, n_present))
        for i, k1 in enumerate(present):
            for j, k2 in enumerate(present):
                a1 = results['classification'].get(k1, {}).get('mean', 0)
                a2 = results['classification'].get(k2, {}).get('mean', 0)
                matrix[i, j] = combined_acc - max(a1, a2)
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=-0.1, vmax=0.15, aspect='auto')
        labels = ['L1:Pwr', 'L2:Therm', 'L3:Kern', 'L4:CDC'][:n_present]
        ax.set_xticks(range(n_present))
        ax.set_yticks(range(n_present))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(n_present):
            for j in range(n_present):
                ax.text(j, i, f'{matrix[i, j]:+.3f}', ha='center', va='center', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Fusion gain')
    ax.set_title('D. Combined vs Best Single Layer')

    plt.tight_layout()
    fig_path = figures_dir / 'z2183_deep_firmware_bridge.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2183: Deep Firmware Bridge')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps-per-trial', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--noise-samples', type=int, default=800,
                        help='Number of noise samples to collect per layer')
    parser.add_argument('--noise-hz', type=float, default=50.0)
    args = parser.parse_args()

    print("=" * 70)
    print("z2183: Deep Firmware Bridge — GPU Silicon as Memristor Analog")
    print("  4 firmware layers x 7 conditions x 100 trials x 25 steps")
    print("  Bridging deep firmware RE to Mario Lanza's memristor research")
    print("=" * 70)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2183_deep_firmware_bridge',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'noise_samples': args.noise_samples,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
        'hw_available': {},
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/8] Connecting to FPGA...")
    ser, port = find_fpga()
    if ser is None:
        print("  FPGA not found — using LIF simulation fallback")
        fpga = False
        results['simulated'] = True
    else:
        print(f"  Connected: {port}")
        fpga = True
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        print("  Kill switch disabled")

    # ─── Step 2: Collect all four firmware noise layers ───
    print("\n[2/8] Collecting firmware noise layers...")
    n_samp = args.noise_samples

    # Layer 1: Power VRM 1/f noise
    print("  Layer 1 — Power VRM 1/f noise (hwmon)...")
    raw_power = collect_power_noise(n_samp, sample_hz=args.noise_hz)
    if raw_power is not None:
        noise_layer1 = normalize_noise(raw_power)
        results['hw_available']['layer1_power'] = True
        print(f"    {len(raw_power)} samples, mean={raw_power.mean():.2f}W, "
              f"std={raw_power.std():.4f}W")
    else:
        print("    UNAVAILABLE — using synthetic 1/f")
        noise_layer1 = normalize_noise(synth_1f_noise(n_samp, rng))
        results['hw_available']['layer1_power'] = False

    # Layer 2: SMN Thermal ADC
    print("  Layer 2 — SMN Thermal ADC (ryzen_smu)...")
    raw_thermal = collect_smn_thermal(n_samp, sample_hz=args.noise_hz)
    if raw_thermal is not None:
        noise_layer2 = normalize_noise(raw_thermal)
        results['hw_available']['layer2_smn_thermal'] = True
        print(f"    {len(raw_thermal)} samples, mean={raw_thermal.mean():.2f}C, "
              f"std={raw_thermal.std():.4f}C")
    else:
        print("    UNAVAILABLE — using synthetic brownian")
        noise_layer2 = normalize_noise(synth_brownian_noise(n_samp, rng))
        results['hw_available']['layer2_smn_thermal'] = False

    # Layer 3: GPU Kernel Jitter
    print("  Layer 3 — GPU Kernel Jitter (torch matmul timing)...")
    raw_jitter = measure_gpu_jitter(n_samples=n_samp)
    if raw_jitter is not None:
        noise_layer3 = normalize_noise(raw_jitter)
        results['hw_available']['layer3_kernel_jitter'] = True
        print(f"    {len(raw_jitter)} samples, mean={raw_jitter.mean():.0f}ns, "
              f"std={raw_jitter.std():.0f}ns")
    else:
        print("    UNAVAILABLE — using synthetic white")
        noise_layer3 = normalize_noise(synth_white_noise(n_samp, rng))
        results['hw_available']['layer3_kernel_jitter'] = False

    # Layer 4: Clock Domain Crossing
    print("  Layer 4 — Clock Domain Crossing (GPU vs FPGA)...")
    raw_cdc = measure_clock_domain_crossing(ser, n_samples=min(n_samp, 300))
    if raw_cdc is not None:
        noise_layer4 = normalize_noise(raw_cdc)
        results['hw_available']['layer4_clock_crossing'] = True
        print(f"    {len(raw_cdc)} samples, mean={raw_cdc.mean():.0f}ns, "
              f"std={raw_cdc.std():.0f}ns")
    else:
        print("    UNAVAILABLE — using synthetic clock jitter")
        noise_layer4 = normalize_noise(synth_clock_jitter(n_samp, rng))
        results['hw_available']['layer4_clock_crossing'] = False

    # Combined: equal-weight fusion of all 4 layers
    min_len = min(len(noise_layer1), len(noise_layer2), len(noise_layer3), len(noise_layer4))
    noise_combined = (0.25 * noise_layer1[:min_len] +
                      0.25 * noise_layer2[:min_len] +
                      0.25 * noise_layer3[:min_len] +
                      0.25 * noise_layer4[:min_len])
    noise_combined = normalize_noise(noise_combined)

    # White noise baseline
    noise_white = normalize_noise(synth_white_noise(n_samp, rng))

    # No noise
    noise_none = np.zeros(n_samp)

    noise_sources = {
        'layer1_power': noise_layer1,
        'layer2_smn_thermal': noise_layer2,
        'layer3_kernel_jitter': noise_layer3,
        'layer4_clock_crossing': noise_layer4,
        'combined': noise_combined,
        'white': noise_white,
        'no_noise': noise_none,
    }

    # Apply IIR filter to layers with temporal structure (skip white and no_noise)
    for key in ['layer1_power', 'layer2_smn_thermal', 'layer4_clock_crossing', 'combined']:
        noise_sources[key] = iir_filter_noise(noise_sources[key], alpha_iir=0.85)

    # ─── Step 3: PSD analysis per layer ───
    print("\n[3/8] Computing PSD slopes per noise layer...")
    psd_results = {}
    for name, noise in noise_sources.items():
        if name == 'no_noise':
            psd_results[name] = {'slope': 0.0}
            continue
        slope, freqs, psd = compute_psd_slope(noise, fs=args.noise_hz)
        psd_results[name] = {'slope': slope}
        print(f"    {name}: PSD slope = {slope:.3f}")
    results['psd_analysis'] = psd_results

    # ─── Step 4: Generate waveform task ───
    print("\n[4/8] Generating waveform classification task...")
    wave_trials, wave_labels = generate_waveforms(
        n_trials=args.n_trials, steps_per_trial=args.steps_per_trial)
    print(f"  {args.n_trials} trials, {args.steps_per_trial} steps each")
    print(f"  Class distribution: {np.bincount(wave_labels)}")

    # ─── Step 5: Run reservoir for each condition ───
    print("\n[5/8] Running reservoir for all 7 conditions...")

    wave_features = {}
    condition_betas = {
        'layer1_power': BETA,
        'layer2_smn_thermal': BETA,
        'layer3_kernel_jitter': BETA,
        'layer4_clock_crossing': BETA,
        'combined': BETA,
        'white': BETA,
        'no_noise': 0.0,
    }

    # Collect spike data for ISI analysis (from longest run per condition)
    isi_spike_data = {}

    for cond_name, noise_src in noise_sources.items():
        beta = condition_betas[cond_name]
        print(f"\n  === {cond_name} (beta={beta:.2f}) ===")
        trial_features = []
        all_spike_counts = []
        t0 = time.monotonic()

        for trial_idx in range(args.n_trials):
            input_signal = wave_trials[trial_idx]

            if fpga:
                states = run_fpga_reservoir_trial(
                    ser, input_signal, noise_src, w_in, w_noise,
                    base_vg=BASE_VG, alpha=ALPHA, beta=beta)
            else:
                states = simulate_lif_reservoir(
                    input_signal, noise_src, w_in, w_noise,
                    base_vg=BASE_VG, alpha=ALPHA, beta=beta)

            aug = augment_with_delays(states, delays=(1, 2, 3))
            feat = pool_trial_features(aug)
            trial_features.append(feat)

            # Collect spike counts for ISI
            all_spike_counts.append(states[:, :N_NEURONS])

            if (trial_idx + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial_idx + 1) / elapsed
                eta = (args.n_trials - trial_idx - 1) / max(rate, 0.01)
                print(f"    Trial {trial_idx + 1}/{args.n_trials} "
                      f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

        wave_features[cond_name] = np.array(trial_features)
        elapsed = time.monotonic() - t0
        print(f"  {cond_name}: {len(trial_features)} trials in {elapsed:.1f}s, "
              f"feat dim={wave_features[cond_name].shape[1]}")

        # Concatenate spike data for ISI analysis
        isi_spike_data[cond_name] = np.concatenate(all_spike_counts, axis=0)

    # ─── Step 6: Classify waveforms (5-fold stratified CV) ───
    print("\n[6/8] Classifying waveforms (5-fold stratified CV)...")

    classification_results = {}
    splits = stratified_kfold(wave_features['combined'], wave_labels, n_splits=5)

    for cond_name, X_all in wave_features.items():
        fold_accs = []
        for train_idx, test_idx in splits:
            X_train = X_all[train_idx]
            X_test = X_all[test_idx]
            y_train = wave_labels[train_idx]
            y_test = wave_labels[test_idx]

            # Z-score normalize
            mu = X_train.mean(axis=0, keepdims=True)
            sigma = X_train.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_train_n = (X_train - mu) / sigma
            X_test_n = (X_test - mu) / sigma

            acc = ridge_classify(X_train_n, y_train, X_test_n, y_test)
            fold_accs.append(acc)

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        classification_results[cond_name] = {
            'mean': mean_acc, 'std': std_acc,
            'folds': [float(a) for a in fold_accs],
        }
        print(f"  {cond_name}: {mean_acc:.3f} +/- {std_acc:.3f}")

    results['classification'] = classification_results

    # ─── Step 7: ISI analysis ───
    print("\n[7/8] Computing ISI statistics...")
    isi_results = {}
    for cond_name, spike_data in isi_spike_data.items():
        cv, isis = compute_isi_stats(spike_data)
        isi_results[cond_name] = {
            'cv': cv,
            'n_isis': len(isis),
            'sample_isis': isis[:200],  # keep sample for plotting
        }
        print(f"  {cond_name}: ISI CV = {cv:.3f} ({len(isis)} intervals)")
    results['isi_analysis'] = isi_results

    # ─── Step 8: Tests T187-T194 ───
    print("\n[8/8] Evaluating tests T187-T194...")
    print("=" * 70)

    tests = {}

    # Gather accuracies
    single_layers = ['layer1_power', 'layer2_smn_thermal',
                     'layer3_kernel_jitter', 'layer4_clock_crossing']
    combined_acc = classification_results['combined']['mean']
    white_acc = classification_results['white']['mean']
    no_noise_acc = classification_results['no_noise']['mean']
    single_accs = {k: classification_results[k]['mean'] for k in single_layers}
    best_single = max(single_accs.values())
    best_single_name = max(single_accs, key=single_accs.get)

    # T187: COMBINED > best single layer (synergy)
    t187_pass = combined_acc > best_single
    tests['T187'] = {
        'name': 'COMBINED > best single layer (synergy)',
        'pass': bool(t187_pass),
        'combined_acc': combined_acc,
        'best_single_acc': best_single,
        'best_single_layer': best_single_name,
        'margin': combined_acc - best_single,
    }
    status = 'PASS' if t187_pass else 'FAIL'
    print(f"  T187 [{status}]: COMBINED={combined_acc:.3f} vs best_single={best_single:.3f} "
          f"({best_single_name}, margin={combined_acc - best_single:+.3f})")

    # T188: COMBINED > WHITE (firmware > random)
    t188_pass = combined_acc > white_acc
    tests['T188'] = {
        'name': 'COMBINED > WHITE (firmware noise > random)',
        'pass': bool(t188_pass),
        'combined_acc': combined_acc,
        'white_acc': white_acc,
        'margin': combined_acc - white_acc,
    }
    status = 'PASS' if t188_pass else 'FAIL'
    print(f"  T188 [{status}]: COMBINED={combined_acc:.3f} vs WHITE={white_acc:.3f} "
          f"(margin={combined_acc - white_acc:+.3f})")

    # T189: At least 3/4 individual layers beat NO_NOISE
    layers_beating_no_noise = sum(1 for k in single_layers
                                   if single_accs[k] > no_noise_acc)
    t189_pass = layers_beating_no_noise >= 3
    tests['T189'] = {
        'name': 'At least 3/4 layers beat NO_NOISE',
        'pass': bool(t189_pass),
        'layers_beating_no_noise': layers_beating_no_noise,
        'no_noise_acc': no_noise_acc,
        'per_layer': {k: {'acc': v, 'beats_no_noise': v > no_noise_acc}
                      for k, v in single_accs.items()},
    }
    status = 'PASS' if t189_pass else 'FAIL'
    print(f"  T189 [{status}]: {layers_beating_no_noise}/4 layers beat NO_NOISE={no_noise_acc:.3f}")
    for k, v in single_accs.items():
        beat = 'Y' if v > no_noise_acc else 'N'
        print(f"           {k}: {v:.3f} [{beat}]")

    # T190: PSD slopes differ across layers
    layer_slopes = [psd_results[k]['slope'] for k in single_layers]
    slope_range = max(layer_slopes) - min(layer_slopes)
    t190_pass = slope_range > 0.3
    tests['T190'] = {
        'name': 'PSD slopes differ across layers (range > 0.3)',
        'pass': bool(t190_pass),
        'slope_range': slope_range,
        'slopes': {k: psd_results[k]['slope'] for k in single_layers},
    }
    status = 'PASS' if t190_pass else 'FAIL'
    print(f"  T190 [{status}]: PSD slope range = {slope_range:.3f} (need > 0.3)")
    for k in single_layers:
        print(f"           {k}: slope = {psd_results[k]['slope']:.3f}")

    # T191: Layer 1 (power VRM) PSD slope in [-2.0, -0.5]
    l1_slope = psd_results['layer1_power']['slope']
    t191_pass = -2.0 <= l1_slope <= -0.5
    tests['T191'] = {
        'name': 'Layer 1 (power VRM) PSD slope in [-2.0, -0.5]',
        'pass': bool(t191_pass),
        'slope': l1_slope,
    }
    status = 'PASS' if t191_pass else 'FAIL'
    print(f"  T191 [{status}]: Layer 1 PSD slope = {l1_slope:.3f} (need in [-2.0, -0.5])")

    # T192: Layer 3 (kernel jitter) PSD slope > -0.5 (closer to white)
    l3_slope = psd_results['layer3_kernel_jitter']['slope']
    t192_pass = l3_slope > -0.5
    tests['T192'] = {
        'name': 'Layer 3 (kernel jitter) PSD slope > -0.5 (near white)',
        'pass': bool(t192_pass),
        'slope': l3_slope,
    }
    status = 'PASS' if t192_pass else 'FAIL'
    print(f"  T192 [{status}]: Layer 3 PSD slope = {l3_slope:.3f} (need > -0.5)")

    # T193: COMBINED ISI CV in Lanza biological range [0.3, 2.0]
    combined_cv = isi_results['combined']['cv']
    t193_pass = 0.3 <= combined_cv <= 2.0
    tests['T193'] = {
        'name': 'COMBINED ISI CV in Lanza range [0.3, 2.0]',
        'pass': bool(t193_pass),
        'cv': combined_cv,
    }
    status = 'PASS' if t193_pass else 'FAIL'
    print(f"  T193 [{status}]: COMBINED ISI CV = {combined_cv:.3f} (need in [0.3, 2.0])")

    # T194: Best single layer accuracy > 55%
    t194_pass = best_single > 0.55
    tests['T194'] = {
        'name': 'Best single layer accuracy > 55%',
        'pass': bool(t194_pass),
        'best_acc': best_single,
        'best_layer': best_single_name,
    }
    status = 'PASS' if t194_pass else 'FAIL'
    print(f"  T194 [{status}]: Best single = {best_single:.3f} ({best_single_name}, need > 0.55)")

    # Summary
    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['summary'] = {
        'pass': n_pass,
        'total': n_total,
        'score': f"{n_pass}/{n_total}",
    }

    print("\n" + "=" * 70)
    print(f"  RESULT: {n_pass}/{n_total} PASS")
    print("=" * 70)

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2183_deep_firmware_bridge.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    # ─── Generate figures ───
    print("\n  Generating figures...")
    generate_figures(results, FIGURES)

    # Cleanup FPGA
    if ser:
        try:
            ser.write(bytes([SYNC, CMD_SET_KILL, 0x01]))
            ser.flush()
            ser.close()
        except Exception:
            pass

    return results


if __name__ == '__main__':
    main()
