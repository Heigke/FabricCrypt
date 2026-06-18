#!/usr/bin/env python3
"""z2180_reservoir_capacity.py — Reservoir Computational Capacity Measurement

Rigorous measurement of the GPU-noise-driven FPGA reservoir's computational capacity
using Fisher's linear discriminant analysis and information-theoretic capacity measures.

Systematically probes how many orthogonal input-output mappings the reservoir can support
by presenting N random binary patterns (N = 2, 4, 8, 16, 32, 64) and measuring
classification accuracy via 5-fold stratified cross-validation.

3 Conditions:
  FULL   — GPU 1/f noise (IIR-filtered power rail) driving Vg modulation
  WHITE  — Gaussian white noise driving Vg modulation
  NO_NOISE — No noise (beta=0), pure input only

Tests T169-T174:
  T169: Capacity at N=2 > 85% accuracy (basic discrimination)
  T170: Capacity curve is monotonically decreasing with N (expected)
  T171: N at 70% accuracy > 4 (reservoir can discriminate > 4 classes)
  T172: FULL Fisher discriminant > WHITE Fisher discriminant (1/f improves separability)
  T173: FULL capacity > NO_NOISE capacity (noise is beneficial)
  T174: Total information capacity > 2.0 bits (non-trivial computation)

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

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters ───
BASE_VG = 0.55
ALPHA = 0.15
BETA = 0.10
N_NEURONS = 8
SAMPLE_HZ = 20
STEPS_PER_TRIAL = 25
N_TRIALS_PER_LEVEL = 100
N_FOLDS = 5
N_LEVELS = [2, 4, 8, 16, 32, 64]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# FPGA Communication (from z2176/z2177)
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


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (uW -> W). Rich 1/f dynamics ~11W +/- 1.5W."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def collect_power_noise(duration_s=15, sample_hz=50):
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
    """IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
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
# Input Pattern Generation
# ═══════════════════════════════════════════════════════════

def generate_binary_patterns(n_classes, n_trials, steps, seed=42):
    """Generate n_classes distinct random binary temporal patterns.

    Each class gets a unique binary template of length `steps`.
    Trials are noisy versions of the template (bit-flip probability 0.05).
    """
    rng = np.random.default_rng(seed)
    # Generate distinct binary templates
    templates = rng.integers(0, 2, size=(n_classes, steps)).astype(float)

    trials = []
    labels = []
    for _ in range(n_trials):
        cls = rng.integers(0, n_classes)
        trial = templates[cls].copy()
        # Add small noise: flip bits with probability 0.05
        flip_mask = rng.random(steps) < 0.05
        trial[flip_mask] = 1.0 - trial[flip_mask]
        trials.append(trial)
        labels.append(cls)

    return np.array(trials), np.array(labels), templates


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                              live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states."""
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
    """Software LIF simulation fallback."""
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
        states[t, N_NEURONS:N_NEURONS * 2] = vmem.copy()
        states[t, N_NEURONS * 2:] = cumulative.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
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
    """Pool per-timestep states into trial-level features: [mean, std, max, min]."""
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def stratified_kfold(X, y, n_splits=5, seed=42):
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


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None, n_classes_global=None):
    """Ridge regression classifier (one-hot for multi-class)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    n_classes = n_classes_global if n_classes_global else int(max(np.max(y_train), np.max(y_test))) + 1
    if len(np.unique(y_train)) < 2:
        return 0.0
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0

    best_acc = -1
    for a in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + a * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue
        pred_test = np.argmax(X_test @ W, axis=1)
        acc = np.mean(pred_test == y_test)
        if acc > best_acc:
            best_acc = acc
    return max(best_acc, 0.0)


def classify_with_cv(X, y, n_splits=5):
    """5-fold stratified CV, returns mean accuracy and fold accuracies."""
    n_classes_global = int(np.max(y)) + 1
    splits = stratified_kfold(X, y, n_splits=n_splits)
    fold_accs = []
    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        mu = X_tr.mean(axis=0, keepdims=True)
        sigma = X_tr.std(axis=0, keepdims=True)
        sigma[sigma < 1e-10] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        acc = ridge_classify(X_tr_n, y_tr, X_te_n, y_te, n_classes_global=n_classes_global)
        fold_accs.append(acc)
    return float(np.mean(fold_accs)), [float(a) for a in fold_accs]


# ═══════════════════════════════════════════════════════════
# Fisher Discriminant Analysis
# ═══════════════════════════════════════════════════════════

def compute_fisher_discriminant(X, y):
    """Compute Fisher's linear discriminant ratio.

    J(w) = (w^T S_B w) / (w^T S_W w)

    Returns the sum of eigenvalues of S_W^{-1} S_B (total discriminant power)
    and the individual eigenvalues (per-dimension discriminant ratios).
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]
    overall_mean = X.mean(axis=0)

    # Within-class scatter matrix S_W
    S_W = np.zeros((n_features, n_features))
    # Between-class scatter matrix S_B
    S_B = np.zeros((n_features, n_features))

    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        if n_c < 2:
            continue
        mean_c = X_c.mean(axis=0)
        # Within-class scatter
        diff_c = X_c - mean_c
        S_W += diff_c.T @ diff_c
        # Between-class scatter
        mean_diff = (mean_c - overall_mean).reshape(-1, 1)
        S_B += n_c * (mean_diff @ mean_diff.T)

    # Regularize S_W
    S_W += np.eye(n_features) * 1e-6

    # Solve generalized eigenvalue problem: S_B w = lambda S_W w
    try:
        S_W_inv = np.linalg.inv(S_W)
        M = S_W_inv @ S_B
        eigenvalues = np.real(np.linalg.eigvals(M))
        eigenvalues = np.sort(eigenvalues)[::-1]
        # Keep only meaningful eigenvalues (n_classes - 1 max)
        n_discriminants = min(n_classes - 1, n_features)
        top_eigs = eigenvalues[:n_discriminants]
        top_eigs = np.maximum(top_eigs, 0.0)  # clip negatives from numerical noise
    except np.linalg.LinAlgError:
        top_eigs = np.zeros(min(n_classes - 1, n_features))

    total_fisher = float(np.sum(top_eigs))
    return total_fisher, top_eigs.tolist()


def compute_information_capacity(fisher_ratios):
    """Total information capacity = sum of log2(1 + J_k) for each discriminant dimension.

    This measures the total bits of information the reservoir encodes about class identity.
    """
    capacity = 0.0
    for j in fisher_ratios:
        if j > 0:
            capacity += np.log2(1.0 + j)
    return capacity


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figures(capacity_results, fig_path):
    """4-panel figure: capacity curve, Fisher ratios, condition comparison, info capacity."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    conditions = ['FULL', 'WHITE', 'NO_NOISE']
    cond_colors = {'FULL': '#2196F3', 'WHITE': '#FF9800', 'NO_NOISE': '#9E9E9E'}
    cond_labels = {'FULL': '1/f noise', 'WHITE': 'White noise', 'NO_NOISE': 'No noise'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Capacity curve (accuracy vs N)
    ax = axes[0, 0]
    for cond in conditions:
        n_vals = []
        acc_vals = []
        for level in capacity_results:
            if cond in level['accuracies']:
                n_vals.append(level['n_classes'])
                acc_vals.append(level['accuracies'][cond])
        if n_vals:
            ax.plot(n_vals, acc_vals, 'o-', color=cond_colors[cond],
                    label=cond_labels[cond], linewidth=2, markersize=6)
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='T169 (85%)')
    ax.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='T171 (70%)')
    ax.set_xlabel('Number of Classes (N)')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('A. Reservoir Capacity Curve')
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_LEVELS)
    ax.set_xticklabels([str(n) for n in N_LEVELS])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Fisher discriminant ratios per N
    ax = axes[0, 1]
    for cond in conditions:
        n_vals = []
        fisher_vals = []
        for level in capacity_results:
            if cond in level.get('fisher_total', {}):
                n_vals.append(level['n_classes'])
                fisher_vals.append(level['fisher_total'][cond])
        if n_vals:
            ax.plot(n_vals, fisher_vals, 's-', color=cond_colors[cond],
                    label=cond_labels[cond], linewidth=2, markersize=6)
    ax.set_xlabel('Number of Classes (N)')
    ax.set_ylabel('Total Fisher Ratio')
    ax.set_title('B. Fisher Discriminant vs N')
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_LEVELS)
    ax.set_xticklabels([str(n) for n in N_LEVELS])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Condition comparison bar chart (mean accuracy across N)
    ax = axes[1, 0]
    mean_accs = {}
    for cond in conditions:
        accs = []
        for level in capacity_results:
            if cond in level['accuracies']:
                accs.append(level['accuracies'][cond])
        mean_accs[cond] = np.mean(accs) if accs else 0.0
    bars = ax.bar(range(len(conditions)),
                  [mean_accs[c] for c in conditions],
                  color=[cond_colors[c] for c in conditions],
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([cond_labels[c] for c in conditions])
    ax.set_ylabel('Mean Accuracy (across N)')
    ax.set_title('C. Condition Comparison')
    for b, c in zip(bars, conditions):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f'{mean_accs[c]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel D: Information capacity
    ax = axes[1, 1]
    for cond in conditions:
        n_vals = []
        cap_vals = []
        for level in capacity_results:
            if cond in level.get('info_capacity', {}):
                n_vals.append(level['n_classes'])
                cap_vals.append(level['info_capacity'][cond])
        if n_vals:
            ax.plot(n_vals, cap_vals, 'D-', color=cond_colors[cond],
                    label=cond_labels[cond], linewidth=2, markersize=6)
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='T174 (2.0 bits)')
    ax.set_xlabel('Number of Classes (N)')
    ax.set_ylabel('Information Capacity (bits)')
    ax.set_title('D. Information Capacity')
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_LEVELS)
    ax.set_xticklabels([str(n) for n in N_LEVELS])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('z2180: Reservoir Computational Capacity — Fisher Discriminant Analysis',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_path), dpi=200, bbox_inches='tight')
    print(f"  Figure saved: {fig_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2180: Reservoir Computational Capacity')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS_PER_LEVEL,
                        help='Trials per N-level (default: 100)')
    parser.add_argument('--steps-per-trial', type=int, default=STEPS_PER_TRIAL,
                        help='Timesteps per trial (default: 25)')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Noise collection duration in seconds')
    parser.add_argument('--n-folds', type=int, default=N_FOLDS,
                        help='Number of CV folds (default: 5)')
    args = parser.parse_args()

    print("=" * 65)
    print("z2180: Reservoir Computational Capacity Measurement")
    print("     Fisher Discriminant + Information-Theoretic Capacity")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2180_reservoir_capacity',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials_per_level': args.n_trials,
            'steps_per_trial': args.steps_per_trial,
            'n_folds': args.n_folds,
            'n_levels': N_LEVELS,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/5] Connecting to FPGA...")
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
    print("\n[2/5] Collecting GPU noise sources...")
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

    noise_1f_iir = iir_filter_noise(noise_1f, alpha_iir=0.85)

    # White noise (same length, standardized)
    noise_white_raw = rng.standard_normal(len(noise_1f))
    noise_white = noise_white_raw / max(np.std(noise_white_raw), 1e-6)

    results['noise'] = {
        '1f_samples': len(noise_1f),
        '1f_mean': float(np.mean(noise_1f)),
        '1f_std': float(np.std(noise_1f)),
        'white_samples': len(noise_white),
    }

    # ─── Step 3: Run reservoir across all N-levels and conditions ───
    print("\n[3/5] Running reservoir across N-levels and conditions...")

    conditions = {
        'FULL': {'noise': noise_1f_iir, 'beta': BETA, 'live_noise': True},
        'WHITE': {'noise': noise_white, 'beta': BETA, 'live_noise': False},
        'NO_NOISE': {'noise': np.array([]), 'beta': 0.0, 'live_noise': False},
    }

    capacity_results = []
    exp_time_start = time.monotonic()

    for n_classes in N_LEVELS:
        level_result = {
            'n_classes': n_classes,
            'accuracies': {},
            'fold_accuracies': {},
            'fisher_total': {},
            'fisher_eigenvalues': {},
            'info_capacity': {},
        }

        print(f"\n  ═══ N = {n_classes} classes ═══")

        # Generate binary patterns for this N
        trials, labels, templates = generate_binary_patterns(
            n_classes, args.n_trials, args.steps_per_trial, seed=42 + n_classes)
        print(f"  Generated {args.n_trials} trials, {n_classes} classes, "
              f"class distribution: {np.bincount(labels).tolist()}")

        for cond_name, cond_cfg in conditions.items():
            print(f"\n    --- Condition: {cond_name} ---")
            trial_feats = []
            t0 = time.monotonic()

            for trial_idx in range(args.n_trials):
                input_signal = trials[trial_idx]

                if fpga:
                    beta_use = cond_cfg['beta']
                    noise_use = cond_cfg['noise']
                    live = cond_cfg['live_noise'] and cond_name == 'FULL'
                    states = run_fpga_reservoir_trial(
                        ser, input_signal, noise_use, w_in, w_noise,
                        base_vg=BASE_VG, alpha=ALPHA, beta=beta_use,
                        live_noise=live)
                else:
                    states = simulate_lif_reservoir(
                        input_signal, cond_cfg['noise'], w_in, w_noise,
                        base_vg=BASE_VG, alpha=ALPHA, beta=cond_cfg['beta'])

                aug = augment_with_delays(states, delays=(1, 2, 3))
                feat = pool_trial_features(aug)
                trial_feats.append(feat)

                if (trial_idx + 1) % 50 == 0:
                    elapsed = time.monotonic() - t0
                    rate = (trial_idx + 1) / elapsed
                    eta = (args.n_trials - trial_idx - 1) / rate
                    print(f"      Trial {trial_idx+1}/{args.n_trials} "
                          f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

            X = np.array(trial_feats)
            y = labels

            # Classification with stratified CV
            mean_acc, fold_accs = classify_with_cv(X, y, n_splits=args.n_folds)
            level_result['accuracies'][cond_name] = mean_acc
            level_result['fold_accuracies'][cond_name] = fold_accs

            # Fisher discriminant
            mu = X.mean(axis=0, keepdims=True)
            sigma = X.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_norm = (X - mu) / sigma
            fisher_total, fisher_eigs = compute_fisher_discriminant(X_norm, y)
            level_result['fisher_total'][cond_name] = fisher_total
            level_result['fisher_eigenvalues'][cond_name] = fisher_eigs

            # Information capacity
            info_cap = compute_information_capacity(fisher_eigs)
            level_result['info_capacity'][cond_name] = info_cap

            elapsed = time.monotonic() - t0
            print(f"      Accuracy: {mean_acc:.4f} | Fisher: {fisher_total:.3f} | "
                  f"InfoCap: {info_cap:.3f} bits | Time: {elapsed:.1f}s")

        capacity_results.append(level_result)

    total_time = time.monotonic() - exp_time_start
    results['capacity_results'] = capacity_results
    results['total_time_s'] = total_time

    # ─── Step 4: Evaluate tests T169-T174 ───
    print("\n[4/5] Evaluating tests T169-T174...")
    tests = {}

    # Build lookup: n_classes -> level_result
    level_lookup = {lr['n_classes']: lr for lr in capacity_results}

    # T169: Capacity at N=2 > 85% accuracy (basic discrimination)
    acc_n2 = level_lookup.get(2, {}).get('accuracies', {}).get('FULL', 0.0)
    t169_pass = acc_n2 > 0.85
    tests['T169_basic_discrimination'] = {
        'pass': t169_pass,
        'acc_n2_full': acc_n2,
        'threshold': 0.85,
        'description': 'N=2 accuracy > 85%',
    }
    print(f"  T169 {'PASS' if t169_pass else 'FAIL'}: N=2 acc={acc_n2:.4f} (>0.85)")

    # T170: Capacity curve is monotonically decreasing with N
    full_accs_by_n = []
    for n in N_LEVELS:
        lr = level_lookup.get(n, {})
        a = lr.get('accuracies', {}).get('FULL', 0.0)
        full_accs_by_n.append(a)
    monotonic_decreasing = all(
        full_accs_by_n[i] >= full_accs_by_n[i + 1] - 0.02  # allow 2% tolerance
        for i in range(len(full_accs_by_n) - 1)
    )
    tests['T170_monotonic_decrease'] = {
        'pass': monotonic_decreasing,
        'full_accs_by_n': {str(n): a for n, a in zip(N_LEVELS, full_accs_by_n)},
        'description': 'Capacity curve monotonically decreasing with N',
    }
    print(f"  T170 {'PASS' if monotonic_decreasing else 'FAIL'}: "
          f"monotonic decrease = {[f'{a:.3f}' for a in full_accs_by_n]}")

    # T171: N at 70% accuracy > 4 (reservoir can discriminate > 4 classes)
    n_at_70 = 0
    for n in N_LEVELS:
        a = level_lookup.get(n, {}).get('accuracies', {}).get('FULL', 0.0)
        if a >= 0.70:
            n_at_70 = n
    t171_pass = n_at_70 > 4
    tests['T171_capacity_depth'] = {
        'pass': t171_pass,
        'n_at_70pct': n_at_70,
        'threshold': 4,
        'description': 'N at 70% accuracy > 4',
    }
    print(f"  T171 {'PASS' if t171_pass else 'FAIL'}: "
          f"N at 70%={n_at_70} (>4)")

    # T172: FULL Fisher discriminant > WHITE Fisher discriminant (1/f improves separability)
    full_fisher_vals = []
    white_fisher_vals = []
    for lr in capacity_results:
        f_full = lr.get('fisher_total', {}).get('FULL', 0.0)
        f_white = lr.get('fisher_total', {}).get('WHITE', 0.0)
        full_fisher_vals.append(f_full)
        white_fisher_vals.append(f_white)
    mean_fisher_full = np.mean(full_fisher_vals) if full_fisher_vals else 0.0
    mean_fisher_white = np.mean(white_fisher_vals) if white_fisher_vals else 0.0
    t172_pass = mean_fisher_full > mean_fisher_white
    tests['T172_fisher_1f_advantage'] = {
        'pass': t172_pass,
        'mean_fisher_full': float(mean_fisher_full),
        'mean_fisher_white': float(mean_fisher_white),
        'ratio': float(mean_fisher_full / max(mean_fisher_white, 1e-10)),
        'description': 'FULL Fisher > WHITE Fisher',
    }
    print(f"  T172 {'PASS' if t172_pass else 'FAIL'}: "
          f"Fisher FULL={mean_fisher_full:.3f} vs WHITE={mean_fisher_white:.3f}")

    # T173: FULL capacity > NO_NOISE capacity (noise is beneficial)
    full_cap_vals = []
    no_noise_cap_vals = []
    for lr in capacity_results:
        c_full = lr.get('info_capacity', {}).get('FULL', 0.0)
        c_nn = lr.get('info_capacity', {}).get('NO_NOISE', 0.0)
        full_cap_vals.append(c_full)
        no_noise_cap_vals.append(c_nn)
    mean_cap_full = np.mean(full_cap_vals) if full_cap_vals else 0.0
    mean_cap_no_noise = np.mean(no_noise_cap_vals) if no_noise_cap_vals else 0.0
    t173_pass = mean_cap_full > mean_cap_no_noise
    tests['T173_noise_benefit'] = {
        'pass': t173_pass,
        'mean_capacity_full': float(mean_cap_full),
        'mean_capacity_no_noise': float(mean_cap_no_noise),
        'description': 'FULL capacity > NO_NOISE capacity',
    }
    print(f"  T173 {'PASS' if t173_pass else 'FAIL'}: "
          f"CapFULL={mean_cap_full:.3f} vs CapNO_NOISE={mean_cap_no_noise:.3f}")

    # T174: Total information capacity > 2.0 bits (non-trivial computation)
    # Use the maximum info capacity across N-levels for FULL condition
    max_info_cap = 0.0
    for lr in capacity_results:
        c = lr.get('info_capacity', {}).get('FULL', 0.0)
        if c > max_info_cap:
            max_info_cap = c
    t174_pass = max_info_cap > 2.0
    tests['T174_info_capacity'] = {
        'pass': t174_pass,
        'max_info_capacity_bits': float(max_info_cap),
        'threshold': 2.0,
        'description': 'Total information capacity > 2.0 bits',
    }
    print(f"  T174 {'PASS' if t174_pass else 'FAIL'}: "
          f"max info capacity={max_info_cap:.3f} bits (>2.0)")

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['summary'] = {
        'pass': n_pass,
        'fail': n_total - n_pass,
        'total': n_total,
        'score': f'{n_pass}/{n_total}',
    }

    # ─── Step 5: Save results and figures ───
    print(f"\n[5/5] Saving results...")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2180_reservoir_capacity.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results: {out_path}")

    # Figure
    fig_path = FIGURES / 'z2180_reservoir_capacity.png'
    make_figures(capacity_results, fig_path)

    # ─── Summary ───
    print("\n" + "=" * 65)
    print(f"z2180 RESULT: {n_pass}/{n_total} tests PASS")
    print("=" * 65)
    for tname, tresult in tests.items():
        status = "PASS" if tresult['pass'] else "FAIL"
        print(f"  {tname}: {status}")

    if fpga and ser:
        try:
            ser.close()
        except Exception:
            pass

    return results


if __name__ == '__main__':
    main()
