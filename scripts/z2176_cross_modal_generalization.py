#!/usr/bin/env python3
"""z2176_cross_modal_generalization.py — Cross-Modal Generalization via FPGA Reservoir

Tests whether an 8-neuron LIF reservoir on Arty A7 FPGA, driven by GPU 1/f noise,
generalizes across different input modalities. Can features learned from one task
transfer to another?

4 Tasks (3-class each, 100 trials x 25 steps):
  Task 1: Waveform classification   (sine / triangle / square)
  Task 2: Frequency discrimination  (low 0.5Hz / mid 1.0Hz / high 2.0Hz)
  Task 3: Amplitude detection       (low 0.2 / mid 0.5 / high 0.8)
  Task 4: Temporal pattern           (constant / ramping / oscillating)

For each task pair (A, B):
  - Within-task accuracy:  5-fold CV on same task
  - Cross-task accuracy:   Train on A features, test on B
  - Transfer ratio:        cross_acc / within_acc

Tests T145-T150:
  T145: Within-task accuracy > 45% for all 4 tasks
  T146: Cross-task transfer ratio > 0.3 for at least 2 task pairs
  T147: Waveform->Frequency transfer > Waveform->Amplitude (shape encodes freq)
  T148: Best within-task accuracy > 55%
  T149: Cross-task matrix NOT symmetric (transfer is directional)
  T150: Mean transfer ratio > 0.25

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
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
N_NEURONS = 8
SAMPLE_HZ = 20


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


# ═══════════════════════════════════════════════════════════
# Input Generators — 4 Modalities
# ═══════════════════════════════════════════════════════════

def generate_waveform_trials(n_trials, steps, dt=1.0/20, seed=42):
    """Task 1: sine/triangle/square (3-class)."""
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    t = np.arange(steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        phase = rng.uniform(0, 2 * np.pi)
        freq = 1.0 * rng.uniform(0.8, 1.2)
        if cls == 0:
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        trials.append((wave + 1.0) / 2.0)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_frequency_trials(n_trials, steps, dt=1.0/20, seed=43):
    """Task 2: low(0.5Hz) / mid(1.0Hz) / high(2.0Hz) sine waves (3-class)."""
    rng = np.random.default_rng(seed)
    freqs = [0.5, 1.0, 2.0]
    trials, labels = [], []
    t = np.arange(steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        freq = freqs[cls] * rng.uniform(0.9, 1.1)
        phase = rng.uniform(0, 2 * np.pi)
        wave = np.sin(2 * np.pi * freq * t + phase)
        trials.append((wave + 1.0) / 2.0)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_amplitude_trials(n_trials, steps, dt=1.0/20, seed=44):
    """Task 3: low(0.2) / mid(0.5) / high(0.8) amplitude sines (3-class)."""
    rng = np.random.default_rng(seed)
    amps = [0.2, 0.5, 0.8]
    trials, labels = [], []
    t = np.arange(steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        amp = amps[cls] * rng.uniform(0.9, 1.1)
        phase = rng.uniform(0, 2 * np.pi)
        freq = 1.0 * rng.uniform(0.9, 1.1)
        wave = amp * np.sin(2 * np.pi * freq * t + phase)
        trials.append((wave + 1.0) / 2.0)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_temporal_pattern_trials(n_trials, steps, dt=1.0/20, seed=45):
    """Task 4: constant / ramping / oscillating envelope (3-class)."""
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    t = np.arange(steps) * dt
    t_norm = t / t[-1]  # normalize to [0, 1]
    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        freq = 1.0 * rng.uniform(0.9, 1.1)
        phase = rng.uniform(0, 2 * np.pi)
        carrier = np.sin(2 * np.pi * freq * t + phase)
        if cls == 0:  # constant envelope
            envelope = np.ones(steps) * 0.5
        elif cls == 1:  # ramping envelope
            envelope = t_norm * 0.8 + 0.1
        else:  # oscillating envelope
            env_freq = rng.uniform(0.3, 0.6)
            envelope = 0.4 * np.sin(2 * np.pi * env_freq * t) + 0.5
        wave = envelope * carrier
        trials.append((wave + 1.0) / 2.0)
        labels.append(cls)
    return np.array(trials), np.array(labels)


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


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier (one-hot for multi-class)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    n_classes = len(np.unique(y_train))
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
    return best_acc


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


def classify_within_task(X, y, n_splits=5):
    """5-fold stratified CV, returns mean accuracy."""
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
        acc = ridge_classify(X_tr_n, y_tr, X_te_n, y_te)
        fold_accs.append(acc)
    return float(np.mean(fold_accs)), [float(a) for a in fold_accs]


def classify_cross_task(X_train, y_train, X_test, y_test):
    """Train on all of task A, test on all of task B. Returns accuracy."""
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-10] = 1.0
    X_tr_n = (X_train - mu) / sigma
    X_te_n = (X_test - mu) / sigma
    acc = ridge_classify(X_tr_n, y_train, X_te_n, y_test)
    return float(acc)


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figure(task_names, within_accs, transfer_matrix, transfer_ratios, fig_path):
    """3-panel figure: within-task bars, transfer heatmap, ratio distribution."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    short_names = ['Waveform', 'Frequency', 'Amplitude', 'Temporal']

    # Panel A: Within-task accuracy bars
    ax = axes[0]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = ax.bar(range(4), within_accs, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.45, color='red', linestyle='--', alpha=0.7, label='T145 threshold (45%)')
    ax.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5, label='Chance (33.3%)')
    ax.set_xticks(range(4))
    ax.set_xticklabels(short_names, rotation=15, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('A. Within-Task Accuracy')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    for b, v in zip(bars, within_accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel B: Cross-task transfer matrix (4x4 heatmap)
    ax = axes[1]
    im = ax.imshow(transfer_matrix, cmap='YlOrRd', vmin=0, vmax=1.0, aspect='equal')
    ax.set_xticks(range(4))
    ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Test Task')
    ax.set_ylabel('Train Task')
    ax.set_title('B. Cross-Task Accuracy')
    for i in range(4):
        for j in range(4):
            color = 'white' if transfer_matrix[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{transfer_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=9, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel C: Transfer ratio distribution
    ax = axes[2]
    ratios_flat = []
    ratio_labels = []
    for i in range(4):
        for j in range(4):
            if i != j:
                ratios_flat.append(transfer_ratios[i, j])
                ratio_labels.append(f'{short_names[i][:3]}->{short_names[j][:3]}')
    sorted_idx = np.argsort(ratios_flat)[::-1]
    ratios_sorted = [ratios_flat[k] for k in sorted_idx]
    labels_sorted = [ratio_labels[k] for k in sorted_idx]
    bar_colors = ['#4CAF50' if r > 0.3 else '#FF5722' for r in ratios_sorted]
    ax.barh(range(len(ratios_sorted)), ratios_sorted, color=bar_colors,
            edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='T146 threshold (0.3)')
    ax.axvline(x=0.25, color='orange', linestyle=':', alpha=0.7, label='T150 threshold (0.25)')
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=7)
    ax.set_xlabel('Transfer Ratio')
    ax.set_title('C. Transfer Ratios (cross/within)')
    ax.legend(fontsize=7, loc='lower right')
    ax.invert_yaxis()

    plt.suptitle('z2176: Cross-Modal Generalization — FPGA Reservoir', fontsize=13, y=1.02)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_path), dpi=200, bbox_inches='tight')
    print(f"  Figure saved: {fig_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2176: Cross-Modal Generalization')
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--steps-per-trial', type=int, default=25)
    parser.add_argument('--noise-collect-s', type=float, default=15.0)
    args = parser.parse_args()

    print("=" * 65)
    print("z2176: Cross-Modal Generalization via FPGA Reservoir")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    task_names = ['waveform', 'frequency', 'amplitude', 'temporal_pattern']

    results = {
        'experiment': 'z2176_cross_modal_generalization',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
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
    print("\n[2/6] Collecting GPU noise sources...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_1f = (power_noise - power_mean) / power_std
        print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_1f)} samples")
    else:
        print("  Power rail unavailable, generating synthetic 1/f")
        n_synth = int(args.noise_collect_s * 50)
        noise_1f = np.zeros(n_synth)
        n_octaves = 8
        octaves = np.zeros(n_octaves)
        for i in range(n_synth):
            for j in range(n_octaves):
                if i % (1 << j) == 0:
                    octaves[j] = rng.standard_normal()
            noise_1f[i] = octaves.sum()
        noise_1f = (noise_1f - noise_1f.mean()) / max(noise_1f.std(), 1e-6)

    noise_1f_iir = iir_filter_noise(noise_1f, alpha_iir=0.85)
    results['noise'] = {'1f_samples': len(noise_1f)}

    # ─── Step 3: Generate all 4 task inputs ───
    print("\n[3/6] Generating task inputs...")
    task_generators = {
        'waveform': generate_waveform_trials,
        'frequency': generate_frequency_trials,
        'amplitude': generate_amplitude_trials,
        'temporal_pattern': generate_temporal_pattern_trials,
    }
    task_data = {}
    for tname, gen_fn in task_generators.items():
        trials, labels = gen_fn(args.n_trials, args.steps_per_trial)
        task_data[tname] = (trials, labels)
        print(f"  {tname}: {trials.shape}, classes={np.bincount(labels)}")

    # ─── Step 4: Run reservoir on all tasks ───
    print("\n[4/6] Running reservoir on all 4 tasks...")
    task_features = {}

    for tname in task_names:
        trials, labels = task_data[tname]
        print(f"\n  === Task: {tname} ===")
        trial_feats = []
        t0 = time.monotonic()

        for trial_idx in range(args.n_trials):
            input_signal = trials[trial_idx]

            if fpga:
                states = run_fpga_reservoir_trial(
                    ser, input_signal, noise_1f_iir, w_in, w_noise,
                    base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                    live_noise=True)
            else:
                states = simulate_lif_reservoir(
                    input_signal, noise_1f_iir, w_in, w_noise,
                    base_vg=BASE_VG, alpha=ALPHA, beta=BETA)

            aug = augment_with_delays(states, delays=(1, 2, 3))
            feat = pool_trial_features(aug)
            trial_feats.append(feat)

            if (trial_idx + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial_idx + 1) / elapsed
                eta = (args.n_trials - trial_idx - 1) / rate
                print(f"    Trial {trial_idx+1}/{args.n_trials} "
                      f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

        task_features[tname] = np.array(trial_feats)
        elapsed = time.monotonic() - t0
        print(f"  {tname}: {len(trial_feats)} trials in {elapsed:.1f}s, "
              f"features shape={task_features[tname].shape}")

    # ─── Step 5: Within-task and cross-task classification ───
    print("\n[5/6] Computing within-task and cross-task accuracies...")

    # Within-task accuracies (5-fold CV)
    within_results = {}
    for tname in task_names:
        X = task_features[tname]
        _, y = task_data[tname]
        mean_acc, fold_accs = classify_within_task(X, y, n_splits=5)
        within_results[tname] = {'mean': mean_acc, 'folds': fold_accs}
        print(f"  Within {tname}: {mean_acc:.3f} (folds: {fold_accs})")

    # Cross-task transfer matrix (4x4)
    n_tasks = len(task_names)
    cross_matrix = np.zeros((n_tasks, n_tasks))
    transfer_ratio_matrix = np.zeros((n_tasks, n_tasks))

    cross_details = {}
    for i, train_task in enumerate(task_names):
        X_train = task_features[train_task]
        _, y_train = task_data[train_task]
        for j, test_task in enumerate(task_names):
            X_test = task_features[test_task]
            _, y_test = task_data[test_task]
            if i == j:
                cross_matrix[i, j] = within_results[train_task]['mean']
                transfer_ratio_matrix[i, j] = 1.0
            else:
                cross_acc = classify_cross_task(X_train, y_train, X_test, y_test)
                cross_matrix[i, j] = cross_acc
                within_acc = within_results[train_task]['mean']
                ratio = cross_acc / within_acc if within_acc > 0 else 0.0
                transfer_ratio_matrix[i, j] = ratio
                key = f"{train_task}->{test_task}"
                cross_details[key] = {
                    'cross_acc': cross_acc,
                    'train_within_acc': within_acc,
                    'transfer_ratio': ratio,
                }
                print(f"  {key}: cross_acc={cross_acc:.3f}, "
                      f"ratio={ratio:.3f}")

    results['within_task'] = within_results
    results['cross_task_matrix'] = cross_matrix.tolist()
    results['transfer_ratio_matrix'] = transfer_ratio_matrix.tolist()
    results['cross_task_details'] = cross_details

    # ─── Step 6: Evaluate Tests T145-T150 ───
    print("\n[6/6] Evaluating tests T145-T150...")
    tests = {}

    within_accs = [within_results[t]['mean'] for t in task_names]

    # T145: Within-task accuracy > 45% for all 4 tasks
    t145_pass = all(a > 0.45 for a in within_accs)
    tests['T145_within_task_above_45pct'] = {
        'pass': t145_pass,
        'within_accs': {t: within_results[t]['mean'] for t in task_names},
        'threshold': 0.45,
    }
    print(f"  T145 within-task >45%: {'PASS' if t145_pass else 'FAIL'} "
          f"(min={min(within_accs):.3f})")

    # T146: Cross-task transfer ratio > 0.3 for at least 2 task pairs
    off_diag_ratios = []
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i != j:
                off_diag_ratios.append(transfer_ratio_matrix[i, j])
    n_above_03 = sum(1 for r in off_diag_ratios if r > 0.3)
    t146_pass = n_above_03 >= 2
    tests['T146_transfer_ratio_above_03'] = {
        'pass': t146_pass,
        'n_pairs_above_03': n_above_03,
        'threshold_n_pairs': 2,
    }
    print(f"  T146 transfer ratio >0.3 for >=2 pairs: {'PASS' if t146_pass else 'FAIL'} "
          f"(n={n_above_03})")

    # T147: Waveform->Frequency transfer > Waveform->Amplitude
    wf_idx = task_names.index('waveform')
    freq_idx = task_names.index('frequency')
    amp_idx = task_names.index('amplitude')
    wf_to_freq = cross_matrix[wf_idx, freq_idx]
    wf_to_amp = cross_matrix[wf_idx, amp_idx]
    t147_pass = wf_to_freq > wf_to_amp
    tests['T147_waveform_freq_gt_amp'] = {
        'pass': t147_pass,
        'waveform_to_frequency': float(wf_to_freq),
        'waveform_to_amplitude': float(wf_to_amp),
    }
    print(f"  T147 Waveform->Freq > Waveform->Amp: {'PASS' if t147_pass else 'FAIL'} "
          f"({wf_to_freq:.3f} vs {wf_to_amp:.3f})")

    # T148: Best within-task accuracy > 55%
    best_within = max(within_accs)
    t148_pass = best_within > 0.55
    tests['T148_best_within_above_55pct'] = {
        'pass': t148_pass,
        'best_within': float(best_within),
        'best_task': task_names[np.argmax(within_accs)],
        'threshold': 0.55,
    }
    print(f"  T148 best within >55%: {'PASS' if t148_pass else 'FAIL'} "
          f"({best_within:.3f}, task={task_names[np.argmax(within_accs)]})")

    # T149: Cross-task matrix NOT symmetric (transfer is directional)
    asymmetries = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            diff = abs(cross_matrix[i, j] - cross_matrix[j, i])
            asymmetries.append(diff)
    max_asymmetry = max(asymmetries) if asymmetries else 0
    t149_pass = max_asymmetry > 0.02  # at least 2% difference somewhere
    tests['T149_asymmetric_transfer'] = {
        'pass': t149_pass,
        'max_asymmetry': float(max_asymmetry),
        'all_asymmetries': [float(a) for a in asymmetries],
        'threshold': 0.02,
    }
    print(f"  T149 asymmetric transfer: {'PASS' if t149_pass else 'FAIL'} "
          f"(max asymmetry={max_asymmetry:.3f})")

    # T150: Mean transfer ratio > 0.25
    mean_transfer = np.mean(off_diag_ratios)
    t150_pass = mean_transfer > 0.25
    tests['T150_mean_transfer_above_025'] = {
        'pass': t150_pass,
        'mean_transfer_ratio': float(mean_transfer),
        'threshold': 0.25,
    }
    print(f"  T150 mean transfer >0.25: {'PASS' if t150_pass else 'FAIL'} "
          f"({mean_transfer:.3f})")

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['summary'] = {
        'pass': n_pass,
        'total': n_total,
        'score': f"{n_pass}/{n_total}",
    }

    print(f"\n{'=' * 65}")
    print(f"RESULT: {n_pass}/{n_total} tests PASS")
    print(f"{'=' * 65}")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2176_cross_modal_generalization.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved: {out_path}")

    # ─── Generate figure ───
    print("\nGenerating figure...")
    fig_path = FIGURES / 'fig_z2176_generalization.png'
    make_figure(task_names, within_accs, cross_matrix, transfer_ratio_matrix, fig_path)

    # ─── Cleanup ───
    if fpga and ser:
        ser.close()
        print("FPGA connection closed")

    return results


if __name__ == '__main__':
    main()
