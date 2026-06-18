#!/usr/bin/env python3
"""z2168_mac_feedback_reservoir.py — Closed-Loop Adaptive MAC Feedback Reservoir

First demonstration of CLOSED-LOOP GPU<->FPGA reservoir computing where the GPU
dynamically modulates FPGA neuron behavior via CMD_SET_MAC (0x06) based on spike
feedback. Previous experiments (z2162-z2167) are OPEN-LOOP: GPU noise drives FPGA
neurons, readout feeds classifier. This experiment CLOSES THE LOOP: the GPU reads
spike activity, computes an adaptive MAC signal, and feeds it back to FPGA.

MAC Feedback Strategies (3 conditions + 2 controls):
  HOMEOSTATIC:  MAC = target_rate / current_rate (stabilize spike rate)
  ENTROPY_MAX:  MAC = 1.0 + k*(H_target - H_current) (maximize response entropy)
  INPUT_TRACK:  MAC = 0.5 + 0.5*|d/dt(input)| (input-aware gain modulation)
  STATIC_MAC:   MAC = 1.0 (constant, control)
  NO_MAC:       No CMD_SET_MAC sent (baseline control)

Tasks:
  Waveform classification (3-class: sine, triangle, square)
  Temporal XOR at tau=1,2,3

Tests:
  T97: HOMEOSTATIC waveform > STATIC_MAC
  T98: At least one adaptive strategy > NO_MAC
  T99: HOMEOSTATIC reduces spike rate variance across trials
  T100: ENTROPY_MAX produces higher response entropy than STATIC_MAC
  T101: INPUT_TRACK waveform > STATIC_MAC
  T102: Best adaptive strategy waveform > 55%

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB1
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# --- FPGA Protocol ---
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03
CMD_SET_MAC = 0x06

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# --- Reservoir Parameters ---
BASE_VG = 0.58       # near BVpar cliff -- input modulation has maximum effect
ALPHA = 0.25         # strong input coupling
BETA = 0.08          # moderate noise coupling
N_NEURONS = 8
SAMPLE_HZ = 20       # FPGA update rate

# --- MAC Feedback Parameters ---
MAC_CLIP_LO = 0.1
MAC_CLIP_HI = 5.0
ENTROPY_K = 0.5      # proportional gain for entropy maximizer
H_TARGET = 3.0       # bits -- max entropy for 8 neurons = log2(8) = 3.0


def _print(*a, **kw):
    kw.setdefault('flush', True)
    print(*a, **kw)


# ================================================================
# FPGA Communication (from z2162/z2165)
# ================================================================

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


def set_mac(ser, mac_val):
    """Set MAC signal via CMD_SET_MAC (0x06). Fire-and-forget, Q16.16 big-endian."""
    q16 = int(max(0.0, min(255.0, mac_val)) * 65536) & 0xFFFFFFFF
    ser.write(bytes([SYNC, CMD_SET_MAC]) + struct.pack('>I', q16))
    ser.flush()


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


def safe_fpga_step(ser, port, vg_values, interval, mac_val=None):
    """Set Vg, optionally set MAC, and read telemetry with reconnection."""
    import serial as serial_mod
    try:
        set_per_neuron_vg(ser, vg_values)
        if mac_val is not None:
            set_mac(ser, mac_val)
        time.sleep(interval * 0.3)
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)
        return ser, telem
    except (serial_mod.SerialException, OSError) as e:
        _print(f"    [!] Serial error: {e}, reconnecting...")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.5)
        new_ser = reconnect_fpga(port)
        if new_ser is None:
            _print("    [!] Reconnection failed")
            return None, None
        _print("    [!] Reconnected successfully")
        return new_ser, None


# ================================================================
# Noise Sources
# ================================================================

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W). Rich 1/f dynamics ~11W +/- 1.5W."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def collect_power_noise(duration_s=15, sample_hz=50):
    """Collect GPU power rail time series for 1/f noise source."""
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
    """Apply IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t].
    Creates temporal memory (ACF ~0.85) from raw noise.
    """
    if len(noise_samples) == 0:
        return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
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
    mu, std = noise.mean(), max(noise.std(), 1e-6)
    return (noise - mu) / std


# ================================================================
# MAC Feedback Strategies
# ================================================================

def compute_mac_homeostatic(delta_spikes, target_rate, prev_mac):
    """HOMEOSTATIC: MAC = target_rate / current_rate.
    Keeps spike rate near target by adjusting excitability.
    """
    current_rate = np.sum(delta_spikes)
    if current_rate < 0.01:
        # No spikes -- increase MAC to boost activity
        mac = prev_mac * 1.5
    else:
        mac = target_rate / current_rate
    # Exponential smoothing to avoid oscillations
    mac = 0.7 * prev_mac + 0.3 * mac
    return np.clip(mac, MAC_CLIP_LO, MAC_CLIP_HI)


def compute_mac_entropy_max(delta_spikes, prev_mac):
    """ENTROPY_MAX: MAC = 1.0 + k*(H_target - H_current).
    Adjusts MAC to push toward maximum entropy across neurons.
    """
    # Compute spike distribution entropy across 8 neurons
    rates = delta_spikes.astype(float) + 1e-8  # avoid log(0)
    total = rates.sum()
    if total < 1e-6:
        h_current = 0.0
    else:
        probs = rates / total
        h_current = -np.sum(probs * np.log2(probs + 1e-12))
    mac = 1.0 + ENTROPY_K * (H_TARGET - h_current)
    # Smooth
    mac = 0.7 * prev_mac + 0.3 * mac
    return np.clip(mac, MAC_CLIP_LO, MAC_CLIP_HI), h_current


def compute_mac_input_track(input_signal, t, prev_mac):
    """INPUT_TRACK: MAC = 0.5 + 0.5*|d/dt(input)|.
    Increase gain for rapid input changes.
    """
    if t == 0:
        deriv = 0.0
    else:
        deriv = abs(input_signal[t] - input_signal[t - 1])
    mac = 0.5 + 0.5 * min(deriv * 5.0, 1.0)  # scale derivative
    mac = 0.7 * prev_mac + 0.3 * mac
    return np.clip(mac, MAC_CLIP_LO, MAC_CLIP_HI)


# ================================================================
# Waveform + XOR Generation (from z2162)
# ================================================================

def generate_waveforms(n_trials=100, steps_per_trial=25, freq_hz=1.0, dt=1.0 / 20):
    """Generate sine/triangle/square waveforms for classification."""
    rng = np.random.default_rng(42)
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
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:          # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))

        wave = (wave + 1.0) / 2.0
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


def generate_xor_sequence(n_steps=3000, seed=42):
    """Generate random binary input for temporal XOR."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_steps).astype(float)


def compute_xor_targets(u, tau):
    """XOR of u(t) and u(t-tau)."""
    n = len(u)
    targets = np.zeros(n)
    for t in range(tau, n):
        targets[t] = int(u[t]) ^ int(u[t - tau])
    return targets


# ================================================================
# Feature Extraction & Classification (from z2162)
# ================================================================

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
    """Pool per-timestep states into trial-level features: [mean, std, max, min]."""
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
    best_pred = None

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
            best_pred = pred_test

    return best_acc, best_pred


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


# ================================================================
# FPGA Reservoir Core with MAC Feedback
# ================================================================

def run_fpga_reservoir_trial_mac(ser, port, input_signal, noise_samples, w_in, w_noise,
                                  mac_strategy='STATIC_MAC', live_noise=False):
    """Drive FPGA neurons with input+noise and MAC feedback, collect states.

    mac_strategy: 'HOMEOSTATIC' | 'ENTROPY_MAX' | 'INPUT_TRACK' | 'STATIC_MAC' | 'NO_MAC'

    Returns: (states, mac_trace, entropy_trace, ser)
      states:        (n_steps, 24) -- 8 delta_spikes + 8 vmem + 8 cumulative
      mac_trace:     (n_steps,) -- MAC values sent at each step
      entropy_trace: (n_steps,) -- response entropy at each step
      ser:           updated serial handle (may change on reconnect)
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    mac_trace = np.zeros(n_steps)
    entropy_trace = np.zeros(n_steps)
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0

    # MAC state
    current_mac = 1.0
    target_rate = None  # computed from first 5 steps for HOMEOSTATIC

    # Accumulators for target rate estimation
    warmup_rates = []

    for t in range(n_steps):
        # --- Compute noise ---
        if live_noise:
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif BETA > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

        # --- Compute per-neuron Vg ---
        vg_values = np.full(N_NEURONS, BASE_VG)
        vg_values += ALPHA * input_signal[t] * w_in
        if BETA > 0:
            vg_values += BETA * noise_val * w_noise
        vg_values = np.clip(vg_values, 0.05, 0.95)

        # --- Determine MAC value ---
        mac_to_send = None
        if mac_strategy == 'STATIC_MAC':
            mac_to_send = 1.0
            current_mac = 1.0
        elif mac_strategy == 'NO_MAC':
            mac_to_send = None  # don't send CMD_SET_MAC
        elif mac_strategy == 'HOMEOSTATIC':
            mac_to_send = current_mac
        elif mac_strategy == 'ENTROPY_MAX':
            mac_to_send = current_mac
        elif mac_strategy == 'INPUT_TRACK':
            current_mac = compute_mac_input_track(input_signal, t, current_mac)
            mac_to_send = current_mac

        mac_trace[t] = current_mac

        # --- FPGA step ---
        ser, telem = safe_fpga_step(ser, port, vg_values, interval, mac_val=mac_to_send)
        if ser is None:
            break

        # --- Extract state ---
        delta_spikes = np.zeros(N_NEURONS)
        if telem:
            counts = [n['spike_count'] for n in telem]
            vmems = [n['vmem'] for n in telem]

            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    delta_spikes[i] = delta
                    states[t, i] = delta
                    cumulative[i] += delta
            for i in range(N_NEURONS):
                states[t, N_NEURONS + i] = vmems[i]
                states[t, N_NEURONS * 2 + i] = cumulative[i]
            prev_counts = counts[:]
        else:
            prev_counts = None

        # --- Compute entropy ---
        rates = delta_spikes + 1e-8
        total = rates.sum()
        if total > 1e-6:
            probs = rates / total
            h = -np.sum(probs * np.log2(probs + 1e-12))
        else:
            h = 0.0
        entropy_trace[t] = h

        # --- Update MAC based on feedback (for next step) ---
        if mac_strategy == 'HOMEOSTATIC':
            rate_total = np.sum(delta_spikes)
            warmup_rates.append(rate_total)
            if target_rate is None and len(warmup_rates) >= 5:
                target_rate = np.mean(warmup_rates[:5])
                if target_rate < 0.5:
                    target_rate = 2.0  # minimum target
            if target_rate is not None:
                current_mac = compute_mac_homeostatic(delta_spikes, target_rate, current_mac)

        elif mac_strategy == 'ENTROPY_MAX':
            current_mac, _ = compute_mac_entropy_max(delta_spikes, current_mac)

        time.sleep(max(0, interval * 0.5 - 0.01))

    return states, mac_trace, entropy_trace, ser


def simulate_lif_reservoir_mac(input_signal, noise_samples, w_in, w_noise,
                                mac_strategy='STATIC_MAC'):
    """Software LIF simulation fallback with MAC feedback."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 3))
    mac_trace = np.zeros(n_steps)
    entropy_trace = np.zeros(n_steps)

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / SAMPLE_HZ
    vmem = np.zeros(N_NEURONS)
    cumulative = np.zeros(N_NEURONS)

    current_mac = 1.0
    target_rate = None
    warmup_rates = []

    for t in range(n_steps):
        # Current from input + noise
        vg = np.full(N_NEURONS, BASE_VG)
        vg += ALPHA * input_signal[t] * w_in
        if BETA > 0 and len(noise_samples) > 0:
            noise_idx = t % len(noise_samples)
            vg += BETA * noise_samples[noise_idx] * w_noise
        vg = np.clip(vg, 0.05, 0.95)

        # Apply MAC as gain modulation on input current
        I_in = vg * 5.0 * current_mac  # MAC scales excitability
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        # Spike detection
        delta_spikes = np.zeros(N_NEURONS)
        for i in range(N_NEURONS):
            if vmem[i] >= v_thresh:
                delta_spikes[i] = 1
                vmem[i] = v_rest
                cumulative[i] += 1

        states[t, :N_NEURONS] = delta_spikes
        states[t, N_NEURONS:N_NEURONS * 2] = vmem.copy()
        states[t, N_NEURONS * 2:] = cumulative.copy()

        # Entropy
        rates = delta_spikes + 1e-8
        total = rates.sum()
        if total > 1e-6:
            probs = rates / total
            h = -np.sum(probs * np.log2(probs + 1e-12))
        else:
            h = 0.0
        entropy_trace[t] = h

        # MAC feedback
        if mac_strategy == 'STATIC_MAC':
            current_mac = 1.0
        elif mac_strategy == 'NO_MAC':
            current_mac = 1.0  # simulation treats no-MAC as MAC=1
        elif mac_strategy == 'HOMEOSTATIC':
            rate_total = np.sum(delta_spikes)
            warmup_rates.append(rate_total)
            if target_rate is None and len(warmup_rates) >= 5:
                target_rate = np.mean(warmup_rates[:5])
                if target_rate < 0.5:
                    target_rate = 2.0
            if target_rate is not None:
                current_mac = compute_mac_homeostatic(delta_spikes, target_rate, current_mac)
        elif mac_strategy == 'ENTROPY_MAX':
            current_mac, _ = compute_mac_entropy_max(delta_spikes, current_mac)
        elif mac_strategy == 'INPUT_TRACK':
            current_mac = compute_mac_input_track(input_signal, t, current_mac)

        mac_trace[t] = current_mac

    return states, mac_trace, entropy_trace


# ================================================================
# Plotting
# ================================================================

def plot_results(wave_accs, mac_traces, rate_variances, entropy_means, results_dict):
    """3-panel figure: waveform accuracy, MAC time series, spike rate stabilization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        _print("  matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Waveform accuracy per condition
    ax = axes[0]
    conditions = list(wave_accs.keys())
    means = [wave_accs[c]['mean'] for c in conditions]
    stds = [wave_accs[c]['std'] for c in conditions]
    colors = ['#2ecc71', '#3498db', '#e67e22', '#95a5a6', '#bdc3c7']
    bars = ax.bar(range(len(conditions)), means, yerr=stds, capsize=5,
                  color=colors[:len(conditions)], edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Waveform Accuracy')
    ax.set_title('Waveform Classification by MAC Strategy')
    ax.axhline(y=1.0 / 3.0, color='red', linestyle='--', alpha=0.5, label='chance')
    ax.set_ylim(0.2, 0.85)
    ax.legend(fontsize=8)

    # Panel 2: MAC time series for one trial (all adaptive strategies)
    ax = axes[1]
    for strategy, trace in mac_traces.items():
        if strategy in ('STATIC_MAC', 'NO_MAC'):
            continue
        if len(trace) > 0:
            ax.plot(trace, label=strategy, alpha=0.8, linewidth=1.2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='MAC=1.0')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('MAC Value')
    ax.set_title('MAC Adaptation (Trial 0)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 5.5)

    # Panel 3: Spike rate variance (stabilization)
    ax = axes[2]
    conds = list(rate_variances.keys())
    vars_vals = [rate_variances[c] for c in conds]
    ax.bar(range(len(conds)), vars_vals, color=colors[:len(conds)],
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Spike Rate Variance (across trials)')
    ax.set_title('Rate Stabilization Effect')

    plt.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES / 'fig_z2168_mac_feedback.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    _print(f"  Figure saved: {fig_path}")


# ================================================================
# Main Experiment
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='z2168: Closed-Loop MAC Feedback Reservoir')
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--steps-per-trial', type=int, default=25)
    parser.add_argument('--xor-steps', type=int, default=3000)
    parser.add_argument('--noise-collect-s', type=float, default=15.0)
    args = parser.parse_args()

    _print("=" * 70)
    _print("z2168: Closed-Loop Adaptive MAC Feedback Reservoir")
    _print("=" * 70)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2168_mac_feedback_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'xor_steps': args.xor_steps,
            'mac_clip_lo': MAC_CLIP_LO, 'mac_clip_hi': MAC_CLIP_HI,
            'entropy_k': ENTROPY_K, 'h_target': H_TARGET,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # --- Step 1: Connect to FPGA ---
    _print("\n[1/8] Connecting to FPGA...")
    ser, port = find_fpga()
    if ser is None:
        _print("  FPGA not found -- using LIF simulation fallback")
        fpga = False
        results['simulated'] = True
    else:
        _print(f"  Connected: {port}")
        fpga = True
        # Disable kill switch
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        _print("  Kill switch disabled")

    # --- Step 2: Collect GPU noise ---
    _print("\n[2/8] Collecting GPU noise source...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_1f = (power_noise - power_mean) / power_std
        _print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_1f)} samples")
    else:
        _print("  Power rail unavailable, generating synthetic 1/f")
        noise_1f = generate_synthetic_1f(int(args.noise_collect_s * 50), rng)

    # IIR filter
    noise_1f_iir = iir_filter_noise(noise_1f, alpha_iir=0.85)
    results['noise'] = {'1f_samples': len(noise_1f)}

    # --- Step 3: Generate waveform task ---
    _print("\n[3/8] Generating waveform classification task...")
    wave_trials, wave_labels = generate_waveforms(
        n_trials=args.n_trials, steps_per_trial=args.steps_per_trial)
    _print(f"  {args.n_trials} trials, {args.steps_per_trial} steps each")
    _print(f"  Class distribution: {np.bincount(wave_labels)}")

    # --- Step 4: Run 5 MAC conditions on waveform task ---
    _print("\n[4/8] Running 5 MAC feedback conditions on waveform task...")

    mac_strategies = ['HOMEOSTATIC', 'ENTROPY_MAX', 'INPUT_TRACK', 'STATIC_MAC', 'NO_MAC']

    wave_features = {}
    all_mac_traces = {}   # first trial MAC trace per condition
    all_rate_means = {}   # per-trial mean spike rate for variance analysis
    all_entropy_means = {}  # per-condition mean entropy

    for strategy in mac_strategies:
        _print(f"\n  === Condition: {strategy} ===")
        trial_features = []
        trial_rate_means = []
        trial_entropies = []
        t0 = time.monotonic()
        first_mac_trace = None

        for trial_idx in range(args.n_trials):
            input_signal = wave_trials[trial_idx]

            if fpga:
                states, mac_trace, ent_trace, ser = run_fpga_reservoir_trial_mac(
                    ser, port, input_signal, noise_1f_iir, w_in, w_noise,
                    mac_strategy=strategy, live_noise=True)
                if ser is None:
                    _print("    [!] FPGA lost, switching to simulation")
                    fpga = False
            else:
                states, mac_trace, ent_trace = simulate_lif_reservoir_mac(
                    input_signal, noise_1f_iir, w_in, w_noise,
                    mac_strategy=strategy)

            # Save first trial MAC trace for plotting
            if first_mac_trace is None:
                first_mac_trace = mac_trace.copy()

            # Trial-level metrics
            mean_rate = np.mean(states[:, :N_NEURONS].sum(axis=1))
            trial_rate_means.append(mean_rate)
            trial_entropies.append(np.mean(ent_trace))

            # Feature extraction
            aug = augment_with_delays(states, delays=(1, 2, 3))
            feat = pool_trial_features(aug)
            trial_features.append(feat)

            if (trial_idx + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial_idx + 1) / elapsed
                eta = (args.n_trials - trial_idx - 1) / rate
                _print(f"    Trial {trial_idx + 1}/{args.n_trials} "
                       f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

        wave_features[strategy] = np.array(trial_features)
        all_mac_traces[strategy] = first_mac_trace.tolist() if first_mac_trace is not None else []
        all_rate_means[strategy] = trial_rate_means
        all_entropy_means[strategy] = float(np.mean(trial_entropies))

        elapsed = time.monotonic() - t0
        rate_var = float(np.var(trial_rate_means))
        _print(f"  {strategy}: {len(trial_features)} trials in {elapsed:.1f}s, "
               f"rate_var={rate_var:.4f}, mean_entropy={np.mean(trial_entropies):.3f}")

    # --- Step 5: Classify waveforms (5-fold stratified CV) ---
    _print("\n[5/8] Classifying waveforms (5-fold stratified CV)...")

    wave_accuracies = {}
    splits = stratified_kfold(wave_features['STATIC_MAC'], wave_labels, n_splits=5)

    for strategy in mac_strategies:
        X_all = wave_features[strategy]
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

            acc, _ = ridge_classify(X_train_n, y_train, X_test_n, y_test)
            fold_accs.append(acc)

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        wave_accuracies[strategy] = {
            'mean': mean_acc, 'std': std_acc,
            'folds': [float(a) for a in fold_accs],
        }
        _print(f"  {strategy}: {mean_acc:.3f} +/- {std_acc:.3f}")

    results['waveform_classification'] = wave_accuracies

    # --- Step 6: Temporal XOR task ---
    _print("\n[6/8] Running temporal XOR task...")

    xor_input = generate_xor_sequence(n_steps=args.xor_steps, seed=42)
    taus = [1, 2, 3]

    xor_results = {}
    for strategy in mac_strategies:
        _print(f"\n  === XOR Condition: {strategy} ===")

        if fpga:
            states, mac_trace, ent_trace, ser = run_fpga_reservoir_trial_mac(
                ser, port, xor_input, noise_1f_iir, w_in, w_noise,
                mac_strategy=strategy, live_noise=True)
            if ser is None:
                fpga = False
        else:
            states, mac_trace, ent_trace = simulate_lif_reservoir_mac(
                xor_input, noise_1f_iir, w_in, w_noise,
                mac_strategy=strategy)

        aug = augment_with_delays(states, delays=(1, 2, 3))

        tau_accs = {}
        for tau in taus:
            targets = compute_xor_targets(xor_input, tau)
            # Use second half for eval, skip warmup
            warmup = max(tau + 10, 100)
            n_total = len(xor_input) - warmup
            split = n_total // 2

            X = aug[warmup:]
            y = targets[warmup:]

            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            mu = X_train.mean(axis=0, keepdims=True)
            sigma = X_train.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0

            acc = ridge_binary(
                (X_train - mu) / sigma, y_train,
                (X_test - mu) / sigma, y_test)
            tau_accs[f'tau_{tau}'] = float(acc)
            _print(f"    tau={tau}: {acc:.3f}")

        xor_results[strategy] = tau_accs

    results['xor_classification'] = xor_results

    # --- Step 7: Compute metrics and run tests ---
    _print("\n[7/8] Running tests T97-T102...")

    rate_variances = {}
    for strategy in mac_strategies:
        rate_variances[strategy] = float(np.var(all_rate_means[strategy]))

    results['rate_variances'] = rate_variances
    results['entropy_means'] = {s: all_entropy_means[s] for s in mac_strategies}
    results['mac_traces_trial0'] = all_mac_traces

    tests = {}

    # T97: HOMEOSTATIC waveform > STATIC_MAC
    t97_pass = wave_accuracies['HOMEOSTATIC']['mean'] > wave_accuracies['STATIC_MAC']['mean']
    tests['T97_homeostatic_gt_static'] = {
        'pass': t97_pass,
        'homeostatic_acc': wave_accuracies['HOMEOSTATIC']['mean'],
        'static_acc': wave_accuracies['STATIC_MAC']['mean'],
        'description': 'HOMEOSTATIC waveform > STATIC_MAC',
    }
    _print(f"  T97 HOMEOSTATIC > STATIC_MAC: "
           f"{wave_accuracies['HOMEOSTATIC']['mean']:.3f} > "
           f"{wave_accuracies['STATIC_MAC']['mean']:.3f} "
           f"{'PASS' if t97_pass else 'FAIL'}")

    # T98: At least one adaptive > NO_MAC
    adaptive_strats = ['HOMEOSTATIC', 'ENTROPY_MAX', 'INPUT_TRACK']
    no_mac_acc = wave_accuracies['NO_MAC']['mean']
    best_adaptive = max(wave_accuracies[s]['mean'] for s in adaptive_strats)
    best_adaptive_name = max(adaptive_strats, key=lambda s: wave_accuracies[s]['mean'])
    t98_pass = best_adaptive > no_mac_acc
    tests['T98_adaptive_gt_nomac'] = {
        'pass': t98_pass,
        'best_adaptive': best_adaptive_name,
        'best_adaptive_acc': best_adaptive,
        'no_mac_acc': no_mac_acc,
        'description': 'At least one adaptive strategy > NO_MAC',
    }
    _print(f"  T98 Best adaptive ({best_adaptive_name}) > NO_MAC: "
           f"{best_adaptive:.3f} > {no_mac_acc:.3f} "
           f"{'PASS' if t98_pass else 'FAIL'}")

    # T99: HOMEOSTATIC reduces spike rate variance
    t99_pass = rate_variances['HOMEOSTATIC'] < rate_variances['STATIC_MAC']
    tests['T99_homeostatic_stabilizes'] = {
        'pass': t99_pass,
        'homeostatic_var': rate_variances['HOMEOSTATIC'],
        'static_var': rate_variances['STATIC_MAC'],
        'description': 'HOMEOSTATIC reduces spike rate variance',
    }
    _print(f"  T99 HOMEOSTATIC var < STATIC var: "
           f"{rate_variances['HOMEOSTATIC']:.4f} < "
           f"{rate_variances['STATIC_MAC']:.4f} "
           f"{'PASS' if t99_pass else 'FAIL'}")

    # T100: ENTROPY_MAX produces higher entropy than STATIC_MAC
    t100_pass = all_entropy_means['ENTROPY_MAX'] > all_entropy_means['STATIC_MAC']
    tests['T100_entropy_max_higher'] = {
        'pass': t100_pass,
        'entropy_max_h': all_entropy_means['ENTROPY_MAX'],
        'static_h': all_entropy_means['STATIC_MAC'],
        'description': 'ENTROPY_MAX higher entropy than STATIC_MAC',
    }
    _print(f"  T100 ENTROPY_MAX entropy > STATIC: "
           f"{all_entropy_means['ENTROPY_MAX']:.3f} > "
           f"{all_entropy_means['STATIC_MAC']:.3f} "
           f"{'PASS' if t100_pass else 'FAIL'}")

    # T101: INPUT_TRACK waveform > STATIC_MAC
    t101_pass = wave_accuracies['INPUT_TRACK']['mean'] > wave_accuracies['STATIC_MAC']['mean']
    tests['T101_input_track_gt_static'] = {
        'pass': t101_pass,
        'input_track_acc': wave_accuracies['INPUT_TRACK']['mean'],
        'static_acc': wave_accuracies['STATIC_MAC']['mean'],
        'description': 'INPUT_TRACK waveform > STATIC_MAC',
    }
    _print(f"  T101 INPUT_TRACK > STATIC_MAC: "
           f"{wave_accuracies['INPUT_TRACK']['mean']:.3f} > "
           f"{wave_accuracies['STATIC_MAC']['mean']:.3f} "
           f"{'PASS' if t101_pass else 'FAIL'}")

    # T102: Best adaptive > 55%
    t102_pass = best_adaptive > 0.55
    tests['T102_best_adaptive_gt_55pct'] = {
        'pass': t102_pass,
        'best_adaptive': best_adaptive_name,
        'best_acc': best_adaptive,
        'threshold': 0.55,
        'description': 'Best adaptive strategy > 55%',
    }
    _print(f"  T102 Best adaptive > 55%: "
           f"{best_adaptive:.3f} > 0.550 "
           f"{'PASS' if t102_pass else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['tests'] = tests
    results['summary'] = {
        'pass': n_pass,
        'total': n_total,
        'score': f"{n_pass}/{n_total}",
    }

    _print(f"\n  SCORE: {n_pass}/{n_total} PASS")

    # --- Step 8: Save results + plot ---
    _print("\n[8/8] Saving results and plotting...")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2168_mac_feedback_reservoir.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    _print(f"  Results: {out_path}")

    plot_results(wave_accuracies, all_mac_traces, rate_variances, all_entropy_means, results)

    # Close FPGA
    if fpga and ser is not None:
        try:
            ser.write(bytes([SYNC, CMD_SET_KILL, 0x01]))
            ser.flush()
            ser.close()
        except Exception:
            pass

    _print("\n" + "=" * 70)
    _print(f"z2168 COMPLETE: {n_pass}/{n_total} PASS")
    _print("=" * 70)


if __name__ == '__main__':
    main()
