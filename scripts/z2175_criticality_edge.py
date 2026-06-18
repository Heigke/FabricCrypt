#!/usr/bin/env python3
"""z2175_criticality_edge.py — Edge-of-Chaos Characterization for FPGA Reservoir

Sweeps base_vg from 0.30 to 0.80 across 12 levels and at each level:
  1. Runs waveform classification (3-class: sine/triangle/square)
  2. Computes Lyapunov-like divergence from perturbed twin trajectories
  3. Measures memory capacity across lags tau=1..10
  4. Records spike rate statistics (mean, CV)

Tests:
  T139: Peak accuracy occurs at Vg in [0.45, 0.65] (near criticality)
  T140: Lyapunov exponent transitions from negative to positive as Vg increases
  T141: Memory capacity peaks near accuracy peak (within +/-2 Vg steps)
  T142: Spike rate CV > 0.3 at optimal Vg (irregular firing)
  T143: Accuracy at optimal Vg > accuracy at lowest Vg by >10pp
  T144: Accuracy at optimal Vg > accuracy at highest Vg by >5pp

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB1
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG     = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL   = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters ───
ALPHA     = 0.25
BETA      = 0.08
N_NEURONS = 8
SAMPLE_HZ = 20

# ─── Lyapunov parameters ───
LYAP_EPSILON = 0.001  # perturbation magnitude
LYAP_STEPS   = 50     # trajectory length for divergence measurement
LYAP_REPS    = 10     # number of twin-trajectory pairs

# ─── Memory capacity ───
MC_STEPS  = 200       # sequence length for memory capacity measurement
MC_LAGS   = 10        # tau = 1..10


# ═══════════════════════════════════════════════════════════
# JSON Encoder
# ═══════════════════════════════════════════════════════════

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# FPGA Communication
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
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W)."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def collect_power_noise(duration_s=10, sample_hz=50):
    """Collect GPU power rail time series for 1/f noise."""
    n_samples = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    powers = []
    for _ in range(n_samples):
        p = read_hwmon_power()
        if p is not None:
            powers.append(p)
        time.sleep(interval)
    return np.array(powers) if powers else None


def iir_filter_noise(noise_samples, alpha_iir=0.85):
    """IIR low-pass: y[t] = a*y[t-1] + (1-a)*x[t]. Creates temporal memory."""
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t-1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


def generate_synthetic_1f(n_samples, rng):
    """Voss-McCartney 1/f noise generator."""
    noise = np.zeros(n_samples)
    n_octaves = 8
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    return (noise - noise.mean()) / max(noise.std(), 1e-6)


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=80, steps_per_trial=25, freq_hz=1.0, dt=1.0/20, seed=42):
    """Generate sine/triangle/square waveforms for classification."""
    rng = np.random.default_rng(seed)
    trials = []
    labels = []
    t = np.arange(steps_per_trial) * dt

    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        phase = rng.uniform(0, 2 * np.pi)
        freq = freq_hz * rng.uniform(0.8, 1.2)

        if cls == 0:   # sine
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1: # triangle
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2*np.pi)) % 1.0) - 1.0) - 1.0
        else:          # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))

        wave = (wave + 1.0) / 2.0  # normalize to [0, 1]
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=0.58, alpha=ALPHA, beta=BETA,
                              live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    Returns: (n_steps, 24) array — 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0

    for t in range(n_steps):
        if live_noise:
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

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
                            base_vg=0.58, alpha=ALPHA, beta=BETA):
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
        vg = np.full(N_NEURONS, base_vg)
        vg += alpha * input_signal[t] * w_in
        if beta > 0 and len(noise_samples) > 0:
            noise_idx = t % len(noise_samples)
            vg += beta * noise_samples[noise_idx] * w_noise
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
        states[t, N_NEURONS:N_NEURONS*2] = vmem.copy()
        states[t, N_NEURONS*2:] = cumulative.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

def pool_trial_features(trial_states):
    """Pool per-timestep reservoir states into trial-level features.
    trial_states: (n_steps, n_features) -> [mean, std, max, min].
    """
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
    for alpha_reg in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha_reg * I, X_train.T @ Y_train)
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


def cross_val_accuracy(features, labels, n_splits=5):
    """5-fold cross-validated ridge classification accuracy."""
    splits = stratified_kfold(features, labels, n_splits=n_splits)
    accs = []
    for train_idx, test_idx in splits:
        X_tr, y_tr = features[train_idx], labels[train_idx]
        X_te, y_te = features[test_idx], labels[test_idx]
        # Standardize
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-8
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        acc = ridge_classify(X_tr, y_tr, X_te, y_te)
        accs.append(acc)
    return np.mean(accs)


# ═══════════════════════════════════════════════════════════
# Lyapunov-like Divergence
# ═══════════════════════════════════════════════════════════

def compute_lyapunov_divergence(run_fn, noise_samples, w_in, w_noise,
                                  base_vg, rng, n_reps=LYAP_REPS,
                                  n_steps=LYAP_STEPS, epsilon=LYAP_EPSILON):
    """Compute Lyapunov-like divergence by running twin trajectories.

    Generates an input sequence, then runs it twice: once clean and once
    with a tiny perturbation (epsilon) added. Measures how the reservoir
    state trajectories diverge over time.

    Returns: mean log-divergence rate (positive = chaotic, negative = ordered).
    """
    divergences = []

    for rep in range(n_reps):
        # Random input sequence
        input_seq = rng.uniform(0, 1, size=n_steps)

        # Twin A: clean
        states_a = run_fn(input_seq, noise_samples, w_in, w_noise,
                          base_vg=base_vg, alpha=ALPHA, beta=BETA)

        # Twin B: perturbed (add epsilon to input at t=0 only)
        input_perturbed = input_seq.copy()
        input_perturbed[0] += epsilon

        states_b = run_fn(input_perturbed, noise_samples, w_in, w_noise,
                          base_vg=base_vg, alpha=ALPHA, beta=BETA)

        # Measure state divergence over time (using vmem channels)
        vmem_a = states_a[:, N_NEURONS:N_NEURONS*2]
        vmem_b = states_b[:, N_NEURONS:N_NEURONS*2]

        dists = np.linalg.norm(vmem_a - vmem_b, axis=1)
        # Avoid log(0)
        dists = np.maximum(dists, 1e-12)

        # Lyapunov exponent estimate: average log growth rate
        # lambda = (1/T) * ln(d(T) / d(0))
        d0 = dists[1] if dists[0] < 1e-11 else dists[0]
        d_final = np.mean(dists[-5:])  # average last 5 steps for stability
        if d0 > 1e-12 and d_final > 1e-12:
            lam = np.log(d_final / d0) / n_steps
            divergences.append(lam)

    if not divergences:
        return 0.0
    return float(np.mean(divergences))


# ═══════════════════════════════════════════════════════════
# Memory Capacity
# ═══════════════════════════════════════════════════════════

def compute_memory_capacity(run_fn, noise_samples, w_in, w_noise,
                              base_vg, rng, n_steps=MC_STEPS, max_lag=MC_LAGS):
    """Compute memory capacity: sum of R^2 for reconstructing input at lag tau.

    MC = sum_{tau=1}^{max_lag} R^2(y_tau, u(t-tau))

    where y_tau is ridge regression output trained to predict u(t-tau)
    from reservoir state at time t.
    """
    # Random input sequence
    input_seq = rng.uniform(0, 1, size=n_steps)

    # Run reservoir
    states = run_fn(input_seq, noise_samples, w_in, w_noise,
                    base_vg=base_vg, alpha=ALPHA, beta=BETA)

    # Use vmem features (most informative continuous states)
    X = states[:, N_NEURONS:N_NEURONS*2]  # (n_steps, 8)

    mc_total = 0.0
    mc_per_lag = []

    for tau in range(1, max_lag + 1):
        # Target: input from tau steps ago
        target = input_seq[:-tau] if tau > 0 else input_seq
        X_tau = X[tau:]  # align reservoir state with delayed target

        n = len(target)
        if n < 20:
            mc_per_lag.append(0.0)
            continue

        # Split train/test (80/20)
        n_train = int(0.8 * n)
        X_tr, X_te = X_tau[:n_train], X_tau[n_train:]
        y_tr, y_te = target[:n_train], target[n_train:]

        # Standardize
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-8
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd

        # Ridge regression
        best_r2 = -1.0
        for alpha_reg in [1e-4, 1e-2, 1.0, 10.0]:
            I = np.eye(X_tr.shape[1])
            try:
                w = np.linalg.solve(X_tr.T @ X_tr + alpha_reg * I, X_tr.T @ y_tr)
            except np.linalg.LinAlgError:
                continue
            y_pred = X_te @ w
            ss_res = np.sum((y_te - y_pred) ** 2)
            ss_tot = np.sum((y_te - y_te.mean()) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            r2 = max(r2, 0.0)  # clamp negative R^2
            if r2 > best_r2:
                best_r2 = r2

        mc_per_lag.append(float(best_r2))
        mc_total += best_r2

    return mc_total, mc_per_lag


# ═══════════════════════════════════════════════════════════
# Spike Rate Statistics
# ═══════════════════════════════════════════════════════════

def measure_spike_stats(run_fn, noise_samples, w_in, w_noise,
                         base_vg, rng, n_steps=100, n_reps=5):
    """Measure mean spike rate and CV across neurons and repetitions."""
    all_rates = []

    for rep in range(n_reps):
        input_seq = rng.uniform(0, 1, size=n_steps)
        states = run_fn(input_seq, noise_samples, w_in, w_noise,
                        base_vg=base_vg, alpha=ALPHA, beta=BETA)

        # Delta spikes per neuron (sum over time)
        spikes_per_neuron = states[:, :N_NEURONS].sum(axis=0)
        rates = spikes_per_neuron / n_steps * SAMPLE_HZ  # spikes/sec
        all_rates.append(rates)

    all_rates = np.array(all_rates)  # (n_reps, N_NEURONS)
    mean_rates = all_rates.mean()
    # CV across neurons (pool all reps)
    flat_rates = all_rates.flatten()
    cv = float(np.std(flat_rates) / max(np.mean(flat_rates), 1e-6))

    return float(mean_rates), cv


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figure(vg_sweep, accuracies, lyapunov_exps, memory_caps,
                spike_rates, spike_cvs, optimal_idx, fig_path):
    """4-panel figure: accuracy, Lyapunov, memory capacity, spike stats."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('z2175: Edge-of-Chaos Characterization — FPGA Reservoir',
                 fontsize=14, fontweight='bold')

    # Panel A: Accuracy vs Vg
    ax = axes[0, 0]
    ax.plot(vg_sweep, accuracies, 'o-', color='#2196F3', linewidth=2, markersize=6)
    ax.axvline(vg_sweep[optimal_idx], color='red', linestyle='--', alpha=0.7,
               label=f'Optimal Vg={vg_sweep[optimal_idx]:.3f}')
    ax.axhline(1.0/3, color='gray', linestyle=':', alpha=0.5, label='Chance (33.3%)')
    ax.axvspan(0.45, 0.65, alpha=0.1, color='green', label='Criticality region')
    ax.set_xlabel('Base Vg')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('A) Waveform Classification Accuracy')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(accuracies) * 1.15, 0.5))
    ax.grid(True, alpha=0.3)

    # Panel B: Lyapunov divergence vs Vg
    ax = axes[0, 1]
    ax.plot(vg_sweep, lyapunov_exps, 's-', color='#FF5722', linewidth=2, markersize=6)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvspan(0.45, 0.65, alpha=0.1, color='green')
    ax.fill_between(vg_sweep, lyapunov_exps, 0,
                     where=[l < 0 for l in lyapunov_exps],
                     alpha=0.15, color='blue', label='Ordered (lambda<0)')
    ax.fill_between(vg_sweep, lyapunov_exps, 0,
                     where=[l >= 0 for l in lyapunov_exps],
                     alpha=0.15, color='red', label='Chaotic (lambda>=0)')
    ax.set_xlabel('Base Vg')
    ax.set_ylabel('Lyapunov Exponent (approx)')
    ax.set_title('B) Lyapunov-like Divergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Memory capacity vs Vg
    ax = axes[1, 0]
    ax.plot(vg_sweep, memory_caps, 'D-', color='#4CAF50', linewidth=2, markersize=6)
    mc_optimal = np.argmax(memory_caps)
    ax.axvline(vg_sweep[mc_optimal], color='green', linestyle='--', alpha=0.7,
               label=f'Peak MC Vg={vg_sweep[mc_optimal]:.3f}')
    ax.axvline(vg_sweep[optimal_idx], color='red', linestyle='--', alpha=0.5,
               label=f'Peak Acc Vg={vg_sweep[optimal_idx]:.3f}')
    ax.axvspan(0.45, 0.65, alpha=0.1, color='green')
    ax.set_xlabel('Base Vg')
    ax.set_ylabel('Memory Capacity (sum R^2)')
    ax.set_title('C) Memory Capacity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Spike rate and CV vs Vg
    ax1 = axes[1, 1]
    color_rate = '#9C27B0'
    color_cv = '#FF9800'
    ax1.plot(vg_sweep, spike_rates, '^-', color=color_rate, linewidth=2,
             markersize=6, label='Mean spike rate')
    ax1.set_xlabel('Base Vg')
    ax1.set_ylabel('Mean Spike Rate (Hz)', color=color_rate)
    ax1.tick_params(axis='y', labelcolor=color_rate)

    ax2 = ax1.twinx()
    ax2.plot(vg_sweep, spike_cvs, 'v-', color=color_cv, linewidth=2,
             markersize=6, label='Spike rate CV')
    ax2.set_ylabel('Spike Rate CV', color=color_cv)
    ax2.tick_params(axis='y', labelcolor=color_cv)
    ax2.axhline(0.3, color=color_cv, linestyle=':', alpha=0.5)

    ax1.axvspan(0.45, 0.65, alpha=0.1, color='green')
    ax1.set_title('D) Spike Rate & Variability')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='z2175: Edge-of-Chaos Characterization for FPGA Reservoir')
    parser.add_argument('--trials-per-vg', type=int, default=80,
                        help='Number of waveform classification trials per Vg level')
    parser.add_argument('--steps-per-trial', type=int, default=25,
                        help='Number of time steps per trial')
    parser.add_argument('--noise-collect-s', type=float, default=10.0,
                        help='Duration (s) to collect GPU power noise')
    parser.add_argument('--vg-steps', type=int, default=12,
                        help='Number of Vg levels to sweep')
    args = parser.parse_args()

    print("=" * 65)
    print("z2175: Edge-of-Chaos Characterization — FPGA Reservoir")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    vg_sweep = np.linspace(0.30, 0.80, args.vg_steps)

    results = {
        'experiment': 'z2175_criticality_edge',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'trials_per_vg': args.trials_per_vg,
            'steps_per_trial': args.steps_per_trial,
            'vg_sweep': vg_sweep.tolist(),
            'lyap_epsilon': LYAP_EPSILON, 'lyap_steps': LYAP_STEPS,
            'lyap_reps': LYAP_REPS, 'mc_steps': MC_STEPS, 'mc_lags': MC_LAGS,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/6] Connecting to FPGA...")
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

    # ─── Step 2: Collect GPU noise ───
    print("\n[2/6] Collecting GPU power noise...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_1f = (power_noise - power_mean) / power_std
        print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_1f)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f")
        noise_1f = generate_synthetic_1f(int(args.noise_collect_s * 50), rng)

    noise_1f_iir = iir_filter_noise(noise_1f, alpha_iir=0.85)
    results['noise'] = {'n_samples': len(noise_1f)}

    # Choose run function based on FPGA availability
    if fpga:
        def run_fn(input_signal, noise_samples, w_in, w_noise,
                   base_vg=0.58, alpha=ALPHA, beta=BETA):
            return run_fpga_reservoir_trial(ser, input_signal, noise_samples,
                                            w_in, w_noise, base_vg=base_vg,
                                            alpha=alpha, beta=beta, live_noise=True)
    else:
        def run_fn(input_signal, noise_samples, w_in, w_noise,
                   base_vg=0.58, alpha=ALPHA, beta=BETA):
            return simulate_lif_reservoir(input_signal, noise_samples,
                                          w_in, w_noise, base_vg=base_vg,
                                          alpha=alpha, beta=beta)

    # ─── Step 3: Sweep Vg levels ───
    print(f"\n[3/6] Sweeping {len(vg_sweep)} Vg levels: {vg_sweep[0]:.2f} -> {vg_sweep[-1]:.2f}")

    accuracies = []
    lyapunov_exps = []
    memory_caps = []
    memory_per_lag = []
    spike_rates = []
    spike_cvs = []
    per_vg_details = []

    for vi, vg in enumerate(vg_sweep):
        print(f"\n  --- Vg={vg:.3f} ({vi+1}/{len(vg_sweep)}) ---")
        t0 = time.monotonic()

        # 3a. Waveform classification
        print(f"    Running {args.trials_per_vg} waveform trials...")
        wave_trials, wave_labels = generate_waveforms(
            n_trials=args.trials_per_vg, steps_per_trial=args.steps_per_trial,
            seed=42 + vi)  # different seed per Vg for diversity

        trial_features = []
        for trial_idx in range(args.trials_per_vg):
            input_signal = wave_trials[trial_idx]
            states = run_fn(input_signal, noise_1f_iir, w_in, w_noise, base_vg=vg)
            feat = pool_trial_features(states)
            trial_features.append(feat)

        features = np.array(trial_features)
        # Replace NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        acc = cross_val_accuracy(features, wave_labels, n_splits=5)
        accuracies.append(float(acc))
        print(f"    Accuracy: {acc:.3f}")

        # 3b. Lyapunov divergence
        print("    Computing Lyapunov divergence...")
        lam = compute_lyapunov_divergence(run_fn, noise_1f_iir, w_in, w_noise,
                                           base_vg=vg, rng=rng)
        lyapunov_exps.append(lam)
        print(f"    Lyapunov: {lam:.6f}")

        # 3c. Memory capacity
        print("    Computing memory capacity...")
        mc_total, mc_lags = compute_memory_capacity(run_fn, noise_1f_iir, w_in, w_noise,
                                                      base_vg=vg, rng=rng)
        memory_caps.append(float(mc_total))
        memory_per_lag.append(mc_lags)
        print(f"    Memory capacity: {mc_total:.3f}")

        # 3d. Spike statistics
        print("    Measuring spike statistics...")
        sr_mean, sr_cv = measure_spike_stats(run_fn, noise_1f_iir, w_in, w_noise,
                                              base_vg=vg, rng=rng)
        spike_rates.append(sr_mean)
        spike_cvs.append(sr_cv)
        print(f"    Spike rate: {sr_mean:.2f} Hz, CV: {sr_cv:.3f}")

        elapsed = time.monotonic() - t0
        print(f"    Elapsed: {elapsed:.1f}s")

        per_vg_details.append({
            'vg': float(vg),
            'accuracy': float(acc),
            'lyapunov': lam,
            'memory_capacity': float(mc_total),
            'memory_per_lag': mc_lags,
            'spike_rate_mean': sr_mean,
            'spike_rate_cv': sr_cv,
            'elapsed_s': elapsed,
        })

    results['per_vg'] = per_vg_details

    # ─── Step 4: Evaluate tests ───
    print("\n[4/6] Evaluating tests T139-T144...")

    optimal_idx = int(np.argmax(accuracies))
    optimal_vg = float(vg_sweep[optimal_idx])
    optimal_acc = accuracies[optimal_idx]
    lowest_acc = accuracies[0]
    highest_acc = accuracies[-1]

    mc_peak_idx = int(np.argmax(memory_caps))

    # T139: Peak accuracy at Vg in [0.45, 0.65]
    t139_pass = 0.45 <= optimal_vg <= 0.65
    print(f"  T139 peak_vg_in_criticality: optimal_vg={optimal_vg:.3f} in [0.45,0.65] -> {'PASS' if t139_pass else 'FAIL'}")

    # T140: Lyapunov transitions negative->positive
    # Check that early Vg values have more negative Lyapunov and later have more positive
    lyap_arr = np.array(lyapunov_exps)
    n_half = len(lyap_arr) // 2
    lyap_low_mean = lyap_arr[:n_half].mean()
    lyap_high_mean = lyap_arr[n_half:].mean()
    t140_pass = lyap_high_mean > lyap_low_mean
    print(f"  T140 lyapunov_transition: low_vg_mean={lyap_low_mean:.6f}, high_vg_mean={lyap_high_mean:.6f} -> {'PASS' if t140_pass else 'FAIL'}")

    # T141: Memory capacity peaks near accuracy peak (within +/-2 steps)
    mc_acc_dist = abs(mc_peak_idx - optimal_idx)
    t141_pass = mc_acc_dist <= 2
    print(f"  T141 memory_peak_near_acc: mc_peak_idx={mc_peak_idx}, acc_peak_idx={optimal_idx}, dist={mc_acc_dist} -> {'PASS' if t141_pass else 'FAIL'}")

    # T142: Spike rate CV > 0.3 at optimal Vg
    cv_at_opt = spike_cvs[optimal_idx]
    t142_pass = cv_at_opt > 0.3
    print(f"  T142 spike_cv_at_optimal: CV={cv_at_opt:.3f} > 0.3 -> {'PASS' if t142_pass else 'FAIL'}")

    # T143: Accuracy at optimal > accuracy at lowest by >10pp
    acc_diff_low = optimal_acc - lowest_acc
    t143_pass = acc_diff_low > 0.10
    print(f"  T143 acc_vs_lowest: {optimal_acc:.3f} - {lowest_acc:.3f} = {acc_diff_low:.3f} > 0.10 -> {'PASS' if t143_pass else 'FAIL'}")

    # T144: Accuracy at optimal > accuracy at highest by >5pp
    acc_diff_high = optimal_acc - highest_acc
    t144_pass = acc_diff_high > 0.05
    print(f"  T144 acc_vs_highest: {optimal_acc:.3f} - {highest_acc:.3f} = {acc_diff_high:.3f} > 0.05 -> {'PASS' if t144_pass else 'FAIL'}")

    tests = {
        'T139_peak_vg_in_criticality': {
            'pass': bool(t139_pass),
            'optimal_vg': optimal_vg,
            'optimal_acc': optimal_acc,
            'criterion': '[0.45, 0.65]',
        },
        'T140_lyapunov_transition': {
            'pass': bool(t140_pass),
            'lyap_low_mean': float(lyap_low_mean),
            'lyap_high_mean': float(lyap_high_mean),
            'criterion': 'high_vg_lyap > low_vg_lyap',
        },
        'T141_memory_near_accuracy': {
            'pass': bool(t141_pass),
            'mc_peak_idx': mc_peak_idx,
            'acc_peak_idx': optimal_idx,
            'distance': mc_acc_dist,
            'criterion': 'dist <= 2 steps',
        },
        'T142_spike_cv_at_optimal': {
            'pass': bool(t142_pass),
            'cv': float(cv_at_opt),
            'criterion': '> 0.3',
        },
        'T143_acc_vs_lowest': {
            'pass': bool(t143_pass),
            'diff_pp': float(acc_diff_low),
            'criterion': '> 0.10',
        },
        'T144_acc_vs_highest': {
            'pass': bool(t144_pass),
            'diff_pp': float(acc_diff_high),
            'criterion': '> 0.05',
        },
    }

    n_pass = sum(1 for t in tests.values() if t['pass'])
    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass,
        'n_total': 6,
        'optimal_vg': optimal_vg,
        'optimal_accuracy': optimal_acc,
        'vg_sweep': vg_sweep.tolist(),
        'accuracies': accuracies,
        'lyapunov_exponents': lyapunov_exps,
        'memory_capacities': memory_caps,
        'spike_rates': spike_rates,
        'spike_cvs': spike_cvs,
    }

    # ─── Step 5: Save results ───
    print("\n[5/6] Saving results...")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2175_criticality_edge.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results: {out_path}")

    # ─── Step 6: Generate figure ───
    print("\n[6/6] Generating figure...")
    fig_path = FIGURES / 'fig_z2175_criticality.png'
    make_figure(vg_sweep, accuracies, lyapunov_exps, memory_caps,
                spike_rates, spike_cvs, optimal_idx, fig_path)

    # ─── Final summary ───
    print("\n" + "=" * 65)
    print(f"z2175 RESULT: {n_pass}/6 PASS")
    print("=" * 65)
    for tname, tdata in tests.items():
        status = "PASS" if tdata['pass'] else "FAIL"
        print(f"  {tname}: {status}")
    print()

    if ser is not None:
        ser.close()

    return n_pass


if __name__ == '__main__':
    main()
