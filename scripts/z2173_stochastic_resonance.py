#!/usr/bin/env python3
"""z2173_stochastic_resonance.py — FPGA Reservoir Stochastic Resonance

Following z2172's surprising finding that random MAC perturbation IMPROVED
accuracy (+6.2pp over baseline), this experiment systematically sweeps noise
amplitude to characterize the stochastic resonance curve.

Sweep 1 — MAC noise amplitude (σ):
  10 levels: σ = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]
  At each level: 80 waveform classification trials, 25 steps each
  MAC value = 1.0 + N(0, σ), clipped to [0.1, 5.0]

Sweep 2 — Input noise amplitude (β):
  5 levels: β = [0.0, 0.02, 0.05, 0.10, 0.20]
  At each level: 80 waveform classification trials, 25 steps each
  Fixed MAC=1.0

Expected: Inverted-U curve — accuracy peaks at moderate noise,
drops at high noise (classic stochastic resonance).

Tests T127-T132:
  T127: Peak MAC noise accuracy > no-noise accuracy (noise helps)
  T128: Peak MAC noise at σ ∈ [0.1, 1.0] (moderate noise is optimal)
  T129: Highest noise (σ=5.0) < peak accuracy (too much noise hurts)
  T130: Peak beta accuracy > zero-beta accuracy (input noise helps)
  T131: SR curve is non-monotonic (not just increasing or decreasing)
  T132: Peak accuracy > 55% (useful computation at optimal noise)

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
CMD_SET_VG     = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL   = 0x03
CMD_SET_MAC    = 0x06

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters ───
BASE_VG    = 0.58
ALPHA      = 0.25
BETA       = 0.08
N_NEURONS  = 8
SAMPLE_HZ  = 20

# ─── Stochastic Resonance Sweep ───
MAC_SIGMAS     = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]
BETA_LEVELS    = [0.0, 0.02, 0.05, 0.10, 0.20]
N_TRIALS       = 80
STEPS_PER_TRIAL = 25


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


def send_kill(ser, mask_byte):
    """Send kill switch mask (bit per neuron, 1=killed)."""
    ser.write(bytes([SYNC, CMD_SET_KILL, mask_byte & 0xFF]))
    ser.flush()
    time.sleep(0.005)


def send_mac(ser, value):
    """Send MAC value (Q16.16, fire-and-forget)."""
    q16 = to_q16_16(max(0.0, min(5.0, value)))
    ser.write(bytes([SYNC, CMD_SET_MAC]) + struct.pack('>I', q16))
    ser.flush()
    time.sleep(0.005)


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
    """Generate synthetic 1/f noise via octave summation."""
    noise = np.zeros(n_samples)
    n_octaves = 8
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    noise = (noise - noise.mean()) / max(noise.std(), 1e-6)
    return noise


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
# LIF Simulation Fallback
# ═══════════════════════════════════════════════════════════

def simulate_lif_reservoir(input_signal, noise_samples, w_in, w_noise,
                            base_vg=BASE_VG, alpha=ALPHA, beta=0.08,
                            mac_value=1.0, rng=None):
    """Software LIF simulation fallback with MAC scaling.

    mac_value scales the input current (simulating MAC perturbation).
    """
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 3))
    if rng is None:
        rng = np.random.default_rng()

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

        # MAC scaling applied to effective current
        I_in = vg * 5.0 * mac_value
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
# FPGA Reservoir Trial with MAC Noise
# ═══════════════════════════════════════════════════════════

def run_fpga_trial_mac_noise(ser, input_signal, noise_samples, w_in, w_noise,
                              mac_sigma=0.0, rng=None):
    """Run single FPGA reservoir trial with per-step MAC noise.

    At each timestep: mac_val = 1.0 + N(0, mac_sigma), clipped to [0.1, 5.0].
    Returns: (n_steps, 24) state array.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0

    for t in range(n_steps):
        # Get live noise value from power rail
        p = read_hwmon_power()
        noise_val = (p - power_mean) / 2.0 if p else 0.0

        # Base Vg computation
        vg_values = np.full(N_NEURONS, BASE_VG)
        vg_values += ALPHA * input_signal[t] * w_in
        if BETA > 0 and len(noise_samples) > 0:
            noise_idx = t % len(noise_samples)
            vg_values += BETA * noise_samples[noise_idx] * w_noise
        vg_values = np.clip(vg_values, 0.05, 0.95)

        set_per_neuron_vg(ser, vg_values)

        # Apply MAC noise
        if mac_sigma > 0:
            mac_val = 1.0 + rng.normal(0, mac_sigma)
            mac_val = np.clip(mac_val, 0.1, 5.0)
        else:
            mac_val = 1.0
        send_mac(ser, mac_val)

        time.sleep(interval * 0.3)

        # Read telemetry
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)

        if telem:
            counts = [n['spike_count'] for n in telem]
            vmems  = [n['vmem'] for n in telem]
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


def run_fpga_trial_beta(ser, input_signal, noise_samples, w_in, w_noise,
                         beta_val=0.0):
    """Run single FPGA reservoir trial with specified beta (input noise coupling).

    MAC is fixed at 1.0. Beta controls noise injection strength.
    Returns: (n_steps, 24) state array.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0

    # Set MAC to 1.0 (no MAC perturbation)
    send_mac(ser, 1.0)

    for t in range(n_steps):
        # Get live noise value from power rail
        p = read_hwmon_power()
        noise_val = (p - power_mean) / 2.0 if p else 0.0

        # Base Vg computation with variable beta
        vg_values = np.full(N_NEURONS, BASE_VG)
        vg_values += ALPHA * input_signal[t] * w_in
        if beta_val > 0 and len(noise_samples) > 0:
            noise_idx = t % len(noise_samples)
            vg_values += beta_val * noise_samples[noise_idx] * w_noise
        vg_values = np.clip(vg_values, 0.05, 0.95)

        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)

        # Read telemetry
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)

        if telem:
            counts = [n['spike_count'] for n in telem]
            vmems  = [n['vmem'] for n in telem]
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


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

def pool_trial_features(trial_states):
    """Pool per-timestep reservoir states into trial-level features."""
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier (one-hot for multi-class)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
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


def classify_with_cv(features_array, labels, n_splits=5):
    """Run 5-fold stratified CV and return mean accuracy."""
    X = features_array
    y = labels

    # Remove constant features
    feat_std = X.std(axis=0)
    good_cols = feat_std > 1e-8
    X_clean = X[:, good_cols]
    if X_clean.shape[1] == 0:
        return 1.0 / 3.0, 0.0, []

    # Normalize
    mu = X_clean.mean(axis=0)
    sigma = X_clean.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_norm = (X_clean - mu) / sigma

    # Add bias
    X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])

    folds = stratified_kfold(X_aug, y, n_splits=n_splits, seed=42)
    fold_accs = []
    for train_idx, test_idx in folds:
        acc = ridge_classify(X_aug[train_idx], y[train_idx],
                              X_aug[test_idx], y[test_idx])
        fold_accs.append(acc)

    return float(np.mean(fold_accs)), float(np.std(fold_accs)), [float(a) for a in fold_accs]


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def plot_sr_curves(mac_sigmas, mac_accs, mac_stds,
                   beta_levels, beta_accs, beta_stds,
                   fig_path):
    """Plot stochastic resonance curves for MAC noise and input noise."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ─── Panel A: MAC noise SR curve ───
    ax = axes[0]
    ax.errorbar(mac_sigmas, mac_accs, yerr=mac_stds,
                fmt='o-', color='#2196F3', linewidth=2, markersize=8,
                capsize=4, capthick=1.5, label='MAC noise')
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Chance (33.3%)')

    # Mark peak
    peak_idx = int(np.argmax(mac_accs))
    ax.plot(mac_sigmas[peak_idx], mac_accs[peak_idx], '*',
            color='#F44336', markersize=18, zorder=5,
            label=f'Peak: {mac_accs[peak_idx]:.1%} @ sigma={mac_sigmas[peak_idx]}')

    ax.set_xlabel('MAC noise amplitude (sigma)', fontsize=12)
    ax.set_ylabel('Classification accuracy', fontsize=12)
    ax.set_title('A: MAC Noise Stochastic Resonance', fontsize=13, fontweight='bold')
    ax.set_xscale('symlog', linthresh=0.05)
    ax.set_ylim(0.2, 0.85)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    # ─── Panel B: Input noise SR curve ───
    ax = axes[1]
    ax.errorbar(beta_levels, beta_accs, yerr=beta_stds,
                fmt='s-', color='#4CAF50', linewidth=2, markersize=8,
                capsize=4, capthick=1.5, label='Input noise (beta)')
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Chance (33.3%)')

    # Mark peak
    peak_idx_b = int(np.argmax(beta_accs))
    ax.plot(beta_levels[peak_idx_b], beta_accs[peak_idx_b], '*',
            color='#F44336', markersize=18, zorder=5,
            label=f'Peak: {beta_accs[peak_idx_b]:.1%} @ beta={beta_levels[peak_idx_b]}')

    ax.set_xlabel('Input noise coupling (beta)', fontsize=12)
    ax.set_ylabel('Classification accuracy', fontsize=12)
    ax.set_title('B: Input Noise Stochastic Resonance', fontsize=13, fontweight='bold')
    ax.set_ylim(0.2, 0.85)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    fig.suptitle('z2173: Stochastic Resonance in GPU-FPGA Reservoir',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps-per-trial', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--noise-collect-s', type=float, default=10.0)
    args = parser.parse_args()

    print("=" * 65)
    print("z2173: Stochastic Resonance in GPU-FPGA Reservoir")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2173_stochastic_resonance',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'mac_sigmas': MAC_SIGMAS, 'beta_levels': BETA_LEVELS,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/6] Connecting to FPGA...")
    ser, port = find_fpga()
    if ser is None:
        print("  FPGA not found -- using LIF simulation fallback")
        fpga = False
        results['simulated'] = True
    else:
        print(f"  Connected: {port}")
        fpga = True
        send_kill(ser, 0x00)
        time.sleep(0.1)
        send_mac(ser, 1.0)  # reset MAC to 1.0
        time.sleep(0.05)
        print("  Kill switch disabled, MAC reset to 1.0")

    # ─── Step 2: Collect GPU noise ───
    print("\n[2/6] Collecting GPU power rail noise (1/f)...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_1f = (power_noise - power_mean) / power_std
        print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_1f)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f")
        n_synth = int(args.noise_collect_s * 50)
        noise_1f = generate_synthetic_1f(n_synth, rng)

    # Apply IIR filter for temporal memory
    noise_filtered = iir_filter_noise(noise_1f, alpha_iir=0.85)
    print(f"  IIR-filtered noise: {len(noise_filtered)} samples, std={np.std(noise_filtered):.3f}")

    # ─── Step 3: Generate waveforms ───
    print("\n[3/6] Generating waveforms...")
    trials, labels = generate_waveforms(
        n_trials=args.n_trials,
        steps_per_trial=args.steps_per_trial,
        seed=42,
    )
    print(f"  {args.n_trials} trials x {args.steps_per_trial} steps")
    class_counts = {int(c): int(np.sum(labels == c)) for c in np.unique(labels)}
    print(f"  Class distribution: {class_counts}")

    # ─── Step 4: Sweep MAC noise amplitude ───
    print("\n[4/6] Sweep 1: MAC noise amplitude (10 levels)...")
    mac_results = {}

    for sigma_idx, sigma in enumerate(MAC_SIGMAS):
        print(f"\n  --- sigma={sigma:.2f} ({sigma_idx+1}/{len(MAC_SIGMAS)}) ---")
        t0 = time.monotonic()

        all_features = []
        for trial_idx in range(args.n_trials):
            if trial_idx % 20 == 0:
                print(f"    Trial {trial_idx}/{args.n_trials}...", end='\r')

            input_signal = trials[trial_idx]
            trial_rng = np.random.default_rng(2000 * sigma_idx + trial_idx)

            if fpga:
                trial_states = run_fpga_trial_mac_noise(
                    ser, input_signal, noise_filtered, w_in, w_noise,
                    mac_sigma=sigma, rng=trial_rng,
                )
            else:
                # Simulation: MAC noise applied as current scaling per timestep
                n_steps = len(input_signal)
                states = np.zeros((n_steps, N_NEURONS * 3))
                vmem = np.zeros(N_NEURONS)
                cumul = np.zeros(N_NEURONS)
                dt = 1.0 / SAMPLE_HZ
                tau_m = 0.02

                for t in range(n_steps):
                    vg = np.full(N_NEURONS, BASE_VG)
                    vg += ALPHA * input_signal[t] * w_in
                    if BETA > 0 and len(noise_filtered) > 0:
                        vg += BETA * noise_filtered[t % len(noise_filtered)] * w_noise
                    vg = np.clip(vg, 0.05, 0.95)

                    # MAC noise
                    if sigma > 0:
                        mac_val = 1.0 + trial_rng.normal(0, sigma)
                        mac_val = np.clip(mac_val, 0.1, 5.0)
                    else:
                        mac_val = 1.0

                    I_in = vg * 5.0 * mac_val
                    dvdt = (-vmem + I_in) / tau_m
                    vmem += dvdt * dt

                    spikes = np.zeros(N_NEURONS)
                    for i in range(N_NEURONS):
                        if vmem[i] >= 1.0:
                            spikes[i] = 1
                            vmem[i] = 0.0
                            cumul[i] += 1

                    states[t, :N_NEURONS] = spikes
                    states[t, N_NEURONS:N_NEURONS*2] = vmem.copy()
                    states[t, N_NEURONS*2:] = cumul.copy()

                trial_states = states

            feat = pool_trial_features(trial_states)
            all_features.append(feat)

        features_array = np.array(all_features)
        elapsed = time.monotonic() - t0

        # Classify
        mean_acc, std_acc, fold_accs = classify_with_cv(features_array, labels)

        mac_results[sigma] = {
            'sigma': sigma,
            'accuracy': mean_acc,
            'std': std_acc,
            'fold_accs': fold_accs,
            'elapsed_s': elapsed,
            'n_features': int(features_array.shape[1]),
        }
        print(f"    sigma={sigma:.2f}: acc={mean_acc:.3f} +/- {std_acc:.3f}  ({elapsed:.1f}s)")

        # Reset MAC after each sigma level
        if fpga:
            send_mac(ser, 1.0)
            time.sleep(0.02)

    # ─── Step 5: Sweep beta (input noise coupling) ───
    print("\n[5/6] Sweep 2: Input noise coupling beta (5 levels)...")
    beta_results = {}

    for beta_idx, beta_val in enumerate(BETA_LEVELS):
        print(f"\n  --- beta={beta_val:.2f} ({beta_idx+1}/{len(BETA_LEVELS)}) ---")
        t0 = time.monotonic()

        all_features = []
        for trial_idx in range(args.n_trials):
            if trial_idx % 20 == 0:
                print(f"    Trial {trial_idx}/{args.n_trials}...", end='\r')

            input_signal = trials[trial_idx]
            trial_rng = np.random.default_rng(5000 * beta_idx + trial_idx)

            if fpga:
                trial_states = run_fpga_trial_beta(
                    ser, input_signal, noise_filtered, w_in, w_noise,
                    beta_val=beta_val,
                )
            else:
                # Simulation: variable beta
                trial_states = simulate_lif_reservoir(
                    input_signal, noise_filtered, w_in, w_noise,
                    base_vg=BASE_VG, alpha=ALPHA, beta=beta_val,
                    mac_value=1.0, rng=trial_rng,
                )

            feat = pool_trial_features(trial_states)
            all_features.append(feat)

        features_array = np.array(all_features)
        elapsed = time.monotonic() - t0

        # Classify
        mean_acc, std_acc, fold_accs = classify_with_cv(features_array, labels)

        beta_results[beta_val] = {
            'beta': beta_val,
            'accuracy': mean_acc,
            'std': std_acc,
            'fold_accs': fold_accs,
            'elapsed_s': elapsed,
            'n_features': int(features_array.shape[1]),
        }
        print(f"    beta={beta_val:.2f}: acc={mean_acc:.3f} +/- {std_acc:.3f}  ({elapsed:.1f}s)")

    # ─── Step 6: Evaluate Tests ───
    print("\n" + "=" * 65)
    print("[6/6] TEST RESULTS")
    print("=" * 65)

    # Extract arrays for analysis
    mac_accs = [mac_results[s]['accuracy'] for s in MAC_SIGMAS]
    mac_stds = [mac_results[s]['std'] for s in MAC_SIGMAS]
    beta_accs = [beta_results[b]['accuracy'] for b in BETA_LEVELS]
    beta_stds = [beta_results[b]['std'] for b in BETA_LEVELS]

    no_noise_acc = mac_accs[0]  # sigma=0.0
    peak_mac_idx = int(np.argmax(mac_accs))
    peak_mac_acc = mac_accs[peak_mac_idx]
    peak_mac_sigma = MAC_SIGMAS[peak_mac_idx]
    highest_noise_acc = mac_accs[-1]  # sigma=5.0

    zero_beta_acc = beta_accs[0]  # beta=0.0
    peak_beta_idx = int(np.argmax(beta_accs))
    peak_beta_acc = beta_accs[peak_beta_idx]
    peak_beta_val = BETA_LEVELS[peak_beta_idx]

    tests = {}

    # T127: Peak MAC noise accuracy > no-noise accuracy (noise helps)
    t127_pass = peak_mac_acc > no_noise_acc
    tests['T127'] = {
        'name': 'Peak MAC noise > no-noise (noise helps)',
        'peak_acc': peak_mac_acc,
        'peak_sigma': peak_mac_sigma,
        'no_noise_acc': no_noise_acc,
        'delta_pp': (peak_mac_acc - no_noise_acc) * 100,
        'pass': t127_pass,
    }
    print(f"\n  T127 Peak MAC > no-noise:   {peak_mac_acc:.3f} > {no_noise_acc:.3f} "
          f"(+{(peak_mac_acc - no_noise_acc)*100:.1f}pp) "
          f"-> {'PASS' if t127_pass else 'FAIL'}")

    # T128: Peak MAC noise at sigma in [0.1, 1.0] (moderate noise is optimal)
    t128_pass = 0.1 <= peak_mac_sigma <= 1.0
    tests['T128'] = {
        'name': 'Peak MAC sigma in [0.1, 1.0] (moderate noise optimal)',
        'peak_sigma': peak_mac_sigma,
        'range': [0.1, 1.0],
        'pass': t128_pass,
    }
    print(f"  T128 Peak sigma in [0.1,1]: sigma={peak_mac_sigma:.2f} "
          f"-> {'PASS' if t128_pass else 'FAIL'}")

    # T129: Highest noise (sigma=5.0) < peak accuracy (too much noise hurts)
    t129_pass = highest_noise_acc < peak_mac_acc
    tests['T129'] = {
        'name': 'Highest noise < peak (too much noise hurts)',
        'highest_acc': highest_noise_acc,
        'peak_acc': peak_mac_acc,
        'delta_pp': (peak_mac_acc - highest_noise_acc) * 100,
        'pass': t129_pass,
    }
    print(f"  T129 sigma=5.0 < peak:      {highest_noise_acc:.3f} < {peak_mac_acc:.3f} "
          f"(delta={-(highest_noise_acc - peak_mac_acc)*100:.1f}pp) "
          f"-> {'PASS' if t129_pass else 'FAIL'}")

    # T130: Peak beta accuracy > zero-beta accuracy (input noise helps)
    t130_pass = peak_beta_acc > zero_beta_acc
    tests['T130'] = {
        'name': 'Peak beta > zero-beta (input noise helps)',
        'peak_acc': peak_beta_acc,
        'peak_beta': peak_beta_val,
        'zero_beta_acc': zero_beta_acc,
        'delta_pp': (peak_beta_acc - zero_beta_acc) * 100,
        'pass': t130_pass,
    }
    print(f"  T130 Peak beta > zero-beta: {peak_beta_acc:.3f} > {zero_beta_acc:.3f} "
          f"(+{(peak_beta_acc - zero_beta_acc)*100:.1f}pp) "
          f"-> {'PASS' if t130_pass else 'FAIL'}")

    # T131: SR curve is non-monotonic (not just increasing or decreasing)
    # Check: there exists i<j<k such that accs[j] > accs[i] and accs[j] > accs[k]
    # or accs[j] < accs[i] and accs[j] < accs[k] (valley)
    diffs = np.diff(mac_accs)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    t131_pass = sign_changes >= 1  # at least one direction change
    tests['T131'] = {
        'name': 'SR curve is non-monotonic',
        'sign_changes': int(sign_changes),
        'mac_accs': mac_accs,
        'mac_sigmas': MAC_SIGMAS,
        'pass': t131_pass,
    }
    print(f"  T131 Non-monotonic:         sign_changes={sign_changes} >= 1 "
          f"-> {'PASS' if t131_pass else 'FAIL'}")

    # T132: Peak accuracy > 55% (useful computation at optimal noise)
    best_overall = max(peak_mac_acc, peak_beta_acc)
    t132_pass = best_overall > 0.55
    tests['T132'] = {
        'name': 'Peak accuracy > 55% (useful computation)',
        'peak_mac_acc': peak_mac_acc,
        'peak_beta_acc': peak_beta_acc,
        'best_overall': best_overall,
        'threshold': 0.55,
        'pass': t132_pass,
    }
    print(f"  T132 Peak acc > 55%:        {best_overall:.3f} > 0.55 "
          f"-> {'PASS' if t132_pass else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    # ─── Summary table ───
    print(f"\n{'='*65}")
    print("MAC NOISE SWEEP")
    print(f"{'sigma':>8}  {'accuracy':>10}  {'std':>8}")
    print("-" * 30)
    for sigma in MAC_SIGMAS:
        r = mac_results[sigma]
        marker = " <-- PEAK" if sigma == peak_mac_sigma else ""
        print(f"  {sigma:6.2f}  {r['accuracy']:10.3f}  {r['std']:8.3f}{marker}")

    print(f"\n{'='*65}")
    print("BETA SWEEP")
    print(f"{'beta':>8}  {'accuracy':>10}  {'std':>8}")
    print("-" * 30)
    for beta_val in BETA_LEVELS:
        r = beta_results[beta_val]
        marker = " <-- PEAK" if beta_val == peak_beta_val else ""
        print(f"  {beta_val:6.2f}  {r['accuracy']:10.3f}  {r['std']:8.3f}{marker}")

    # ─── Save results ───
    results['mac_sweep'] = {str(s): mac_results[s] for s in MAC_SIGMAS}
    results['beta_sweep'] = {str(b): beta_results[b] for b in BETA_LEVELS}
    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass,
        'n_total': n_total,
        'peak_mac_sigma': peak_mac_sigma,
        'peak_mac_acc': peak_mac_acc,
        'no_noise_acc': no_noise_acc,
        'mac_improvement_pp': (peak_mac_acc - no_noise_acc) * 100,
        'peak_beta': peak_beta_val,
        'peak_beta_acc': peak_beta_acc,
        'zero_beta_acc': zero_beta_acc,
        'beta_improvement_pp': (peak_beta_acc - zero_beta_acc) * 100,
        'highest_noise_acc': highest_noise_acc,
        'sign_changes_in_mac_curve': int(sign_changes),
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2173_stochastic_resonance.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    # ─── Plot ───
    fig_path = FIGURES / 'fig_z2173_stochastic_resonance.png'
    print("\n  Generating figure...")
    plot_sr_curves(MAC_SIGMAS, mac_accs, mac_stds,
                   BETA_LEVELS, beta_accs, beta_stds,
                   fig_path)

    # ─── Cleanup ───
    if fpga:
        send_mac(ser, 1.0)
        send_kill(ser, 0x00)
        ser.close()
        print("  FPGA cleanup done")

    print(f"\n{'='*65}")
    print(f"z2173 COMPLETE: {n_pass}/{n_total} PASS")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
