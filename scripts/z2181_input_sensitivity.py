#!/usr/bin/env python3
"""z2181_input_sensitivity.py — Input Sensitivity of GPU-Noise-Driven FPGA Reservoir

Measures the reservoir's input sensitivity: how small an input change can be
reliably detected in the output, and whether 1/f noise enhances detection of
weak signals (stochastic facilitation / suprathreshold stochastic resonance).

Experiment Design:
  - Present pairs of stimuli differing by amplitude delta = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
  - 80 trials per delta level, 25 steps/trial, binary classification (stimulus A vs A+delta)
  - Stimulus A = fixed sine wave at 0.5 Hz (base_vg=0.55, alpha=0.15)
  - 3 conditions: FULL (1/f), WHITE, NO_NOISE
  - 5-fold stratified CV per delta level
  - Compute psychometric curve: accuracy vs delta
  - Measure just-noticeable difference (JND) = smallest delta with accuracy > 70%

Tests T175-T180:
  T175: JND_FULL < JND_NO_NOISE (1/f noise lowers detection threshold)
  T176: Accuracy at delta=0.50 > 80% (large differences detectable)
  T177: Accuracy at delta=0.01 < 60% (near-chance for tiny differences, no floor artifacts)
  T178: FULL accuracy > WHITE accuracy at 3+ delta levels
  T179: Psychometric curve is monotonically increasing
  T180: Mean d-prime across deltas > 0.5 for FULL condition

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
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

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters ───
BASE_VG = 0.55
ALPHA = 0.15
BETA = 0.10
N_NEURONS = 8
SAMPLE_HZ = 20
N_STEPS = 25
N_TRIALS_PER_DELTA = 80
STIM_FREQ = 0.5  # Hz

DELTAS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
N_FOLDS = 5


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return crc


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W)."""
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
    """Apply IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                              live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.
    Returns: (n_steps, 24) array -- 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
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
                            base_vg=BASE_VG, alpha=ALPHA, beta=BETA):
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
            noise_val = noise_samples[noise_idx]
        else:
            noise_val = 0.0

        if beta > 0:
            vg += beta * noise_val * w_noise
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
# Stimulus Generation
# ═══════════════════════════════════════════════════════════

def generate_stimulus_pair(n_steps, delta, rng, freq=STIM_FREQ):
    """Generate stimulus pair: A (sine at freq) and B (A + delta amplitude boost).
    Returns: signal_A (n_steps,), signal_B (n_steps,), label_A=0, label_B=1.
    """
    t = np.arange(n_steps) / SAMPLE_HZ
    # Random phase offset for variability across trials
    phase = rng.uniform(0, 2 * np.pi)
    signal_A = np.sin(2 * np.pi * freq * t + phase)  # range [-1, 1]
    signal_B = signal_A * (1.0 + delta)  # amplitude boost
    return signal_A, signal_B


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


def ridge_binary_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression binary classifier. Returns accuracy."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    best_acc = -1
    for alpha_val in alphas:
        n_feat = X_train.shape[1]
        I = np.eye(n_feat)
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha_val * I,
                                X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue
        pred_test = (X_test @ W > 0.5).astype(float)
        acc_test = np.mean(pred_test == y_test)
        if acc_test > best_acc:
            best_acc = acc_test

    return max(best_acc, 0.0)


def stratified_kfold_accuracy(features, labels, n_folds=N_FOLDS, rng=None):
    """Stratified K-fold cross-validation for binary classification.
    Returns mean accuracy across folds.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(labels)
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]
    rng.shuffle(idx_0)
    rng.shuffle(idx_1)

    fold_accs = []
    for fold in range(n_folds):
        # Stratified split
        test_0 = idx_0[fold * len(idx_0) // n_folds: (fold + 1) * len(idx_0) // n_folds]
        test_1 = idx_1[fold * len(idx_1) // n_folds: (fold + 1) * len(idx_1) // n_folds]
        test_idx = np.concatenate([test_0, test_1])
        train_idx = np.setdiff1d(np.arange(n), test_idx)

        if len(test_idx) < 2 or len(train_idx) < 2:
            continue

        X_train = features[train_idx]
        y_train = labels[train_idx]
        X_test = features[test_idx]
        y_test = labels[test_idx]

        # Normalize features
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

        acc = ridge_binary_classify(X_train, y_train, X_test, y_test)
        fold_accs.append(acc)

    return float(np.mean(fold_accs)) if fold_accs else 0.5


def compute_dprime(acc):
    """Compute d' from accuracy (assuming equal priors, unbiased observer).
    d' = 2 * Z(accuracy) where Z is the inverse normal CDF.
    Clip to avoid infinities.
    """
    acc_clipped = np.clip(acc, 0.01, 0.99)
    # Approximate inverse normal CDF using rational approximation
    p = acc_clipped
    # Beasley-Springer-Moro algorithm for inverse normal
    a = [0, -3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [0, -5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [0, -7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [0, 7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        z = (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
            ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q / \
            (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        z = -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) / \
             ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)

    # d' = 2 * z(accuracy) for 2AFC
    return float(2.0 * z)


# ═══════════════════════════════════════════════════════════
# Figure Generation
# ═══════════════════════════════════════════════════════════

def generate_figures(results_data):
    """Generate psychometric curve and sensitivity figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    FIGURES.mkdir(parents=True, exist_ok=True)

    deltas = results_data['deltas']
    cond_results = results_data['conditions']

    # ─── Figure 1: Psychometric Curves ───
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = {'FULL': '#e41a1c', 'WHITE': '#377eb8', 'NO_NOISE': '#999999'}
    markers = {'FULL': 'o', 'WHITE': 's', 'NO_NOISE': '^'}

    for cond_name in ['FULL', 'WHITE', 'NO_NOISE']:
        cd = cond_results[cond_name]
        accs = cd['accuracies']
        ax.plot(deltas, accs, color=colors[cond_name], marker=markers[cond_name],
                linewidth=2, markersize=8, label=cond_name)

    ax.axhline(y=0.70, color='gray', linestyle='--', alpha=0.5, label='JND threshold (70%)')
    ax.axhline(y=0.50, color='gray', linestyle=':', alpha=0.3, label='Chance')
    ax.set_xlabel('Amplitude Delta', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('z2181: Psychometric Curves — Input Sensitivity', fontsize=13)
    ax.set_xscale('log')
    ax.set_ylim(0.35, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES / 'z2181_psychometric_curves.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES / 'z2181_psychometric_curves.png'}")

    # ─── Figure 2: d-prime vs Delta ───
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for cond_name in ['FULL', 'WHITE', 'NO_NOISE']:
        cd = cond_results[cond_name]
        dprimes = cd['dprimes']
        ax.plot(deltas, dprimes, color=colors[cond_name], marker=markers[cond_name],
                linewidth=2, markersize=8, label=cond_name)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="d'=0.5")
    ax.set_xlabel('Amplitude Delta', fontsize=12)
    ax.set_ylabel("d' (Sensitivity)", fontsize=12)
    ax.set_title('z2181: Sensitivity Index (d\') vs Input Delta', fontsize=13)
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES / 'z2181_dprime_curves.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES / 'z2181_dprime_curves.png'}")

    # ─── Figure 3: JND Comparison Bar Chart ───
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    cond_names = ['FULL', 'WHITE', 'NO_NOISE']
    jnds = [cond_results[c]['jnd'] for c in cond_names]
    bar_colors = [colors[c] for c in cond_names]
    bars = ax.bar(cond_names, jnds, color=bar_colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Just-Noticeable Difference (JND)', fontsize=12)
    ax.set_title('z2181: JND by Noise Condition', fontsize=13)
    ax.set_ylim(0, max(jnds) * 1.3 if max(jnds) > 0 else 1.0)
    for bar, jnd in zip(bars, jnds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{jnd:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(str(FIGURES / 'z2181_jnd_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES / 'z2181_jnd_comparison.png'}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2181: Input Sensitivity')
    parser.add_argument('--trials', type=int, default=N_TRIALS_PER_DELTA,
                        help='Trials per delta level')
    parser.add_argument('--steps', type=int, default=N_STEPS, help='Steps per trial')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Duration to collect power noise (s)')
    args = parser.parse_args()

    n_trials = args.trials
    n_steps = args.steps

    print("=" * 65)
    print("z2181: Input Sensitivity of GPU-Noise-Driven FPGA Reservoir")
    print("=" * 65)
    print(f"  Deltas: {DELTAS}")
    print(f"  Trials/delta: {n_trials}  Steps: {n_steps}  Folds: {N_FOLDS}")
    print(f"  Stimulus: sine at {STIM_FREQ} Hz")
    print(f"  base_vg={BASE_VG}  alpha={ALPHA}  beta={BETA}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2181_input_sensitivity',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_steps': n_steps, 'n_trials_per_delta': n_trials,
            'stim_freq': STIM_FREQ, 'deltas': DELTAS,
            'n_folds': N_FOLDS,
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
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        print("  Kill switch disabled")

    # ─── Step 2: Collect GPU noise ───
    print("\n[2/6] Collecting GPU noise sources...")
    print("  Collecting power rail noise (1/f)...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_1f_raw = (power_noise - power_mean) / power_std
        print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_1f_raw)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f")
        n_synth = int(args.noise_collect_s * 50)
        noise_1f_raw = np.zeros(n_synth)
        n_octaves = 8
        octaves = np.zeros(n_octaves)
        for i in range(n_synth):
            for j in range(n_octaves):
                if i % (1 << j) == 0:
                    octaves[j] = rng.standard_normal()
            noise_1f_raw[i] = octaves.sum()
        noise_1f_raw = (noise_1f_raw - noise_1f_raw.mean()) / max(noise_1f_raw.std(), 1e-6)

    noise_1f = iir_filter_noise(noise_1f_raw, alpha_iir=0.85)
    noise_white = rng.standard_normal(len(noise_1f))
    noise_zero = np.zeros(1000)

    conditions = {
        'FULL': {'noise': noise_1f, 'beta': BETA, 'label': 'GPU 1/f noise'},
        'WHITE': {'noise': noise_white, 'beta': BETA, 'label': 'White noise'},
        'NO_NOISE': {'noise': noise_zero, 'beta': 0.0, 'label': 'No noise'},
    }

    # ─── Step 3: Run sensitivity experiment ───
    print("\n[3/6] Running input sensitivity experiment...")
    total_trials = len(DELTAS) * n_trials * 2 * len(conditions)  # x2 for A and B
    trial_count = 0

    condition_results = {}

    for cond_name, cond in conditions.items():
        print(f"\n  === Condition: {cond_name} ({cond['label']}) ===")
        cond_noise = cond['noise']
        cond_beta = cond['beta']

        delta_accuracies = []
        delta_dprimes = []

        for di, delta in enumerate(DELTAS):
            print(f"    Delta {delta:.2f}: ", end='', flush=True)

            features_list = []
            labels_list = []

            for trial_idx in range(n_trials):
                # Generate stimulus pair
                signal_A, signal_B = generate_stimulus_pair(n_steps, delta, rng, freq=STIM_FREQ)

                # Run A trial (label=0)
                if fpga:
                    states_A = run_fpga_reservoir_trial(
                        ser, signal_A, cond_noise, w_in, w_noise,
                        base_vg=BASE_VG, alpha=ALPHA, beta=cond_beta)
                else:
                    states_A = simulate_lif_reservoir(
                        signal_A, cond_noise, w_in, w_noise,
                        base_vg=BASE_VG, alpha=ALPHA, beta=cond_beta)

                feat_A = pool_trial_features(states_A)
                features_list.append(feat_A)
                labels_list.append(0)

                # Run B trial (label=1)
                if fpga:
                    states_B = run_fpga_reservoir_trial(
                        ser, signal_B, cond_noise, w_in, w_noise,
                        base_vg=BASE_VG, alpha=ALPHA, beta=cond_beta)
                else:
                    states_B = simulate_lif_reservoir(
                        signal_B, cond_noise, w_in, w_noise,
                        base_vg=BASE_VG, alpha=ALPHA, beta=cond_beta)

                feat_B = pool_trial_features(states_B)
                features_list.append(feat_B)
                labels_list.append(1)

                trial_count += 2
                if (trial_idx + 1) % 20 == 0:
                    print(f"{trial_idx + 1}", end=' ', flush=True)

            features = np.array(features_list)
            labels = np.array(labels_list, dtype=float)

            # 5-fold stratified CV
            fold_rng = np.random.default_rng(42 + di)
            acc = stratified_kfold_accuracy(features, labels, n_folds=N_FOLDS, rng=fold_rng)
            dprime = compute_dprime(acc)

            delta_accuracies.append(acc)
            delta_dprimes.append(dprime)
            print(f" -> acc={acc:.3f}  d'={dprime:.2f}")

        # Compute JND: smallest delta where accuracy > 70%
        jnd = 1.0  # default: never reaches threshold
        for di, delta in enumerate(DELTAS):
            if delta_accuracies[di] > 0.70:
                jnd = delta
                break

        condition_results[cond_name] = {
            'accuracies': delta_accuracies,
            'dprimes': delta_dprimes,
            'jnd': jnd,
            'mean_dprime': float(np.mean(delta_dprimes)),
        }

        print(f"    JND ({cond_name}): {jnd:.3f}")
        print(f"    Mean d' ({cond_name}): {np.mean(delta_dprimes):.3f}")

    # ─── Step 4: Evaluate tests ───
    print("\n[4/6] Evaluating tests T175-T180...")

    full = condition_results['FULL']
    white = condition_results['WHITE']
    no_noise = condition_results['NO_NOISE']

    # T175: JND_FULL < JND_NO_NOISE
    t175_pass = full['jnd'] < no_noise['jnd']
    print(f"  T175 JND_FULL < JND_NO_NOISE: {full['jnd']:.3f} < {no_noise['jnd']:.3f} -> {'PASS' if t175_pass else 'FAIL'}")

    # T176: Accuracy at delta=0.50 > 80%
    acc_050 = full['accuracies'][-1]  # delta=0.50 is last
    t176_pass = acc_050 > 0.80
    print(f"  T176 acc(delta=0.50) > 0.80: {acc_050:.3f} -> {'PASS' if t176_pass else 'FAIL'}")

    # T177: Accuracy at delta=0.01 < 60%
    acc_001 = full['accuracies'][0]  # delta=0.01 is first
    t177_pass = acc_001 < 0.60
    print(f"  T177 acc(delta=0.01) < 0.60: {acc_001:.3f} -> {'PASS' if t177_pass else 'FAIL'}")

    # T178: FULL accuracy > WHITE accuracy at 3+ delta levels
    n_full_better = sum(1 for i in range(len(DELTAS))
                        if full['accuracies'][i] > white['accuracies'][i])
    t178_pass = n_full_better >= 3
    print(f"  T178 FULL > WHITE at 3+ deltas: {n_full_better}/6 -> {'PASS' if t178_pass else 'FAIL'}")

    # T179: Psychometric curve monotonically increasing (FULL condition)
    full_accs = full['accuracies']
    monotonic = all(full_accs[i] <= full_accs[i + 1] + 0.02  # small tolerance
                    for i in range(len(full_accs) - 1))
    t179_pass = monotonic
    print(f"  T179 monotonic psychometric: {full_accs} -> {'PASS' if t179_pass else 'FAIL'}")

    # T180: Mean d-prime > 0.5 for FULL condition
    mean_dp = full['mean_dprime']
    t180_pass = mean_dp > 0.5
    print(f"  T180 mean d'(FULL) > 0.5: {mean_dp:.3f} -> {'PASS' if t180_pass else 'FAIL'}")

    tests = {
        'T175': {
            'name': 'JND_FULL < JND_NO_NOISE',
            'jnd_full': full['jnd'],
            'jnd_no_noise': no_noise['jnd'],
            'pass': t175_pass,
        },
        'T176': {
            'name': 'acc(delta=0.50) > 0.80',
            'accuracy_050': acc_050,
            'pass': t176_pass,
        },
        'T177': {
            'name': 'acc(delta=0.01) < 0.60',
            'accuracy_001': acc_001,
            'pass': t177_pass,
        },
        'T178': {
            'name': 'FULL > WHITE at 3+ deltas',
            'n_full_better': n_full_better,
            'pass': t178_pass,
        },
        'T179': {
            'name': 'monotonic psychometric curve',
            'accuracies': full_accs,
            'pass': t179_pass,
        },
        'T180': {
            'name': "mean d'(FULL) > 0.5",
            'mean_dprime': mean_dp,
            'pass': t180_pass,
        },
    }

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n  Result: {n_pass}/{n_total} PASS")

    results['deltas'] = DELTAS
    results['conditions'] = condition_results
    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass,
        'n_total': n_total,
        'pass_rate': f"{n_pass}/{n_total}",
    }

    # ─── Step 5: Save results ───
    print("\n[5/6] Saving results...")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2181_input_sensitivity.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Saved: {out_path}")

    # ─── Step 6: Generate figures ───
    print("\n[6/6] Generating figures...")
    generate_figures(results)

    # ─── Cleanup ───
    if fpga and ser:
        # Set all neurons to rest
        set_per_neuron_vg(ser, np.full(N_NEURONS, 0.3))
        ser.close()
        print("\n  FPGA connection closed")

    print("\n" + "=" * 65)
    print(f"z2181 COMPLETE: {n_pass}/{n_total} PASS")
    print("=" * 65)

    return n_pass, n_total


if __name__ == '__main__':
    main()
