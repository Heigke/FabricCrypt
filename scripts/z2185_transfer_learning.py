#!/usr/bin/env python3
"""z2185_transfer_learning.py — Transfer Learning in GPU-Noise-Driven FPGA Reservoir

Tests whether a reservoir trained on GPU noise from one TASK can transfer to a
DIFFERENT task — proving the noise creates generalizable computational structure,
not task-specific overfitting.

Protocol:
  1. Train ridge classifier on Task A (3-class waveform: sine/triangle/square)
  2. Apply same readout weights to Task B (frequency discrimination: 1Hz/2Hz/4Hz)
  3. Compare transfer accuracy against fresh training and random weights

4 conditions:
  FULL      — GPU power 1/f noise -> FPGA neurons
  WHITE     — White noise -> FPGA neurons
  NO_NOISE  — Deterministic FPGA
  PINK_IIR  — Software IIR pink noise -> FPGA neurons

Tests T201-T206:
  T201: Transfer accuracy (A->B, FULL) > chance (33.3%)
  T202: Transfer accuracy (A->B, FULL) > Transfer accuracy (A->B, WHITE)
  T203: Transfer accuracy (A->B, FULL) > Transfer accuracy (A->B, NO_NOISE)
  T204: Transfer efficiency = Transfer_acc / Fresh_acc > 0.50 for FULL
  T205: Transfer accuracy (A->C, FULL) > chance (33.3%)
  T206: Mean transfer efficiency (FULL) > Mean transfer efficiency (WHITE)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters (defaults) ───
BASE_VG = 0.55
ALPHA = 0.15
BETA = 0.10
N_NEURONS = 8
SAMPLE_HZ = 20


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
# FPGA Communication (from z2177)
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


class VossMcCartneyFilter:
    """10-octave Voss-McCartney 1/f noise generator."""

    def __init__(self, n_octaves=10):
        self.n_octaves = n_octaves
        self.values = np.zeros(n_octaves)
        self.counters = np.zeros(n_octaves, dtype=int)
        self.step = 0

    def process(self, white_sample):
        """Process one white noise sample, return 1/f filtered value."""
        x = (white_sample / 127.5) - 1.0
        for k in range(self.n_octaves):
            period = 1 << k
            if self.step % period == 0:
                self.values[k] = x
        self.step += 1
        total = np.sum(self.values) / self.n_octaves
        return np.clip(total, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core (from z2177)
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                              sample_hz=SAMPLE_HZ, live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    Returns: (n_steps, 24) array — 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / sample_hz
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
# Feature Building (from z2177: 130 features)
# ═══════════════════════════════════════════════════════════

def build_features(trial_states, delays=(1, 2, 3)):
    """Build 130-dim feature vector from trial reservoir states.

    8 neurons x (delta_spikes + vmem) = 16 channels
    x (1 + 3 delays) = 64 time-delayed features
    + mean pool (16) + max pool (16) + std pool (16) + min pool (16)
    Total: 64 + 16 + 16 + 16 + 16 = 128... we use first 24 channels (including cumulative)
    to get to ~130.

    Actually: 8*(delta + vmem) = 16 per step, x (1+3 delays) = 64
    Then pool: mean(24) + max(24) + std(24) + min(24) = 96
    Delay features at last step: 24 * 4 = 96... let's do augment + pool.
    """
    # Use delta_spikes (8) + vmem (8) = 16 channels for delay features
    n_steps = trial_states.shape[0]
    spk_vmem = trial_states[:, :N_NEURONS * 2]  # (T, 16)

    # Time-delayed copies
    D = spk_vmem.shape[1]
    aug = np.zeros((n_steps, D * (1 + len(delays))))
    aug[:, :D] = spk_vmem
    for i, d in enumerate(delays):
        start = D * (i + 1)
        aug[d:, start:start + D] = spk_vmem[:n_steps - d]

    # Pool over time
    feat = np.concatenate([
        aug.mean(axis=0),   # 64
        aug.max(axis=0),    # 64
        # Plus summary stats of full state
        trial_states.mean(axis=0)[:2],  # 2 (overall mean delta, vmem)
    ])
    return feat  # 64 + 64 + 2 = 130


# ═══════════════════════════════════════════════════════════
# Task Generation
# ═══════════════════════════════════════════════════════════

def generate_task_a(n_trials, n_steps, sample_hz=SAMPLE_HZ, seed=100):
    """Task A: Waveform classification — sine(0), triangle(1), square(2)."""
    rng = np.random.default_rng(seed)
    inputs = np.zeros((n_trials, n_steps))
    labels = np.zeros(n_trials, dtype=int)
    amplitude = 0.3
    t_axis = np.arange(n_steps) / sample_hz

    for trial in range(n_trials):
        cls = trial % 3
        freq = rng.uniform(0.8, 1.5)
        phase = rng.uniform(0, 2 * np.pi)
        labels[trial] = cls

        if cls == 0:  # sine
            inputs[trial] = amplitude * np.sin(2 * np.pi * freq * t_axis + phase)
        elif cls == 1:  # triangle
            from scipy.signal import sawtooth
            inputs[trial] = amplitude * sawtooth(2 * np.pi * freq * t_axis + phase, width=0.5)
        else:  # square
            inputs[trial] = amplitude * np.sign(np.sin(2 * np.pi * freq * t_axis + phase))

    # Shuffle
    perm = rng.permutation(n_trials)
    return inputs[perm], labels[perm]


def generate_task_b(n_trials, n_steps, sample_hz=SAMPLE_HZ, seed=200):
    """Task B: Frequency discrimination — 1Hz(0), 2Hz(1), 4Hz(2) sine waves."""
    rng = np.random.default_rng(seed)
    inputs = np.zeros((n_trials, n_steps))
    labels = np.zeros(n_trials, dtype=int)
    amplitude = 0.3
    t_axis = np.arange(n_steps) / sample_hz
    freqs = [1.0, 2.0, 4.0]

    for trial in range(n_trials):
        cls = trial % 3
        freq = freqs[cls]
        phase = rng.uniform(0, 2 * np.pi)
        labels[trial] = cls
        inputs[trial] = amplitude * np.sin(2 * np.pi * freq * t_axis + phase)

    perm = rng.permutation(n_trials)
    return inputs[perm], labels[perm]


def generate_task_c(n_trials, n_steps, sample_hz=SAMPLE_HZ, seed=300):
    """Task C: Amplitude discrimination — 0.15(0), 0.30(1), 0.60(2)."""
    rng = np.random.default_rng(seed)
    inputs = np.zeros((n_trials, n_steps))
    labels = np.zeros(n_trials, dtype=int)
    amplitudes = [0.15, 0.30, 0.60]
    t_axis = np.arange(n_steps) / sample_hz

    for trial in range(n_trials):
        cls = trial % 3
        amp = amplitudes[cls]
        freq = rng.uniform(1.0, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        labels[trial] = cls
        inputs[trial] = amp * np.sin(2 * np.pi * freq * t_axis + phase)

    perm = rng.permutation(n_trials)
    return inputs[perm], labels[perm]


# ═══════════════════════════════════════════════════════════
# Ridge Multiclass Classifier
# ═══════════════════════════════════════════════════════════

def ridge_multiclass_fit(X_train, y_train, n_classes=3, alphas=None):
    """Fit ridge regression for multiclass (one-vs-all). Returns weight matrix."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    n_features = X_train.shape[1]

    # One-hot encode labels
    Y_onehot = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_onehot[i, int(y)] = 1.0

    best_W = None
    best_gcv = np.inf
    I = np.eye(n_features)

    for alpha in alphas:
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_onehot)
        except np.linalg.LinAlgError:
            continue
        resid = Y_onehot - X_train @ W
        hat_trace = np.trace(X_train @ np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T))
        dof = max(1, len(y_train) - hat_trace)
        gcv = np.sum(resid ** 2) / dof
        if gcv < best_gcv:
            best_gcv = gcv
            best_W = W

    if best_W is None:
        best_W = np.zeros((n_features, n_classes))
    return best_W


def ridge_multiclass_predict(X, W):
    """Predict class from weight matrix."""
    scores = X @ W
    return np.argmax(scores, axis=1)


def ridge_multiclass_accuracy(X, y, W):
    """Compute accuracy using weight matrix."""
    preds = ridge_multiclass_predict(X, W)
    return float(np.mean(preds == y))


def ridge_multiclass_cv(X, y, n_classes=3, n_folds=5, seed=42):
    """5-fold CV ridge multiclass, returns mean accuracy."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    fold_size = len(y) // n_folds
    accs = []
    for fold in range(n_folds):
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        W = ridge_multiclass_fit(X[train_idx], y[train_idx], n_classes=n_classes)
        acc = ridge_multiclass_accuracy(X[test_idx], y[test_idx], W)
        accs.append(acc)
    return float(np.mean(accs))


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2185: Transfer Learning Benchmark')
    parser.add_argument('--n-trials', type=int, default=200, help='Trials per task')
    parser.add_argument('--steps-per-trial', type=int, default=25, help='Steps per trial')
    parser.add_argument('--sample-hz', type=int, default=20, help='Sample rate (Hz)')
    parser.add_argument('--base-vg', type=float, default=0.55, help='Base gate voltage')
    parser.add_argument('--alpha', type=float, default=0.15, help='Input coupling strength')
    parser.add_argument('--beta', type=float, default=0.10, help='Noise coupling strength')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Duration to collect power noise (s)')
    args = parser.parse_args()

    n_trials = args.n_trials
    n_steps = args.steps_per_trial
    base_vg = args.base_vg
    alpha = args.alpha
    beta = args.beta

    # Update globals for build_features compatibility
    global SAMPLE_HZ
    SAMPLE_HZ = args.sample_hz

    print("=" * 65)
    print("z2185: Transfer Learning in GPU-Noise-Driven FPGA Reservoir")
    print("=" * 65)
    print(f"  Trials: {n_trials}  Steps: {n_steps}  Sample Hz: {args.sample_hz}")
    print(f"  Base Vg: {base_vg}  Alpha: {alpha}  Beta: {beta}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2185_transfer_learning',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': base_vg, 'alpha': alpha, 'beta': beta,
            'n_neurons': N_NEURONS, 'sample_hz': args.sample_hz,
            'n_trials': n_trials, 'n_steps': n_steps,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/7] Connecting to FPGA...")
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
    print("\n[2/7] Collecting GPU noise sources...")
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

    # PINK_IIR: Voss-McCartney filter applied to white noise bytes
    print("  Generating Voss-McCartney pink noise...")
    vm_filter = VossMcCartneyFilter(n_octaves=10)
    white_bytes = rng.integers(0, 256, size=len(noise_1f))
    noise_pink = np.array([vm_filter.process(wb) for wb in white_bytes])
    noise_pink = noise_pink / max(np.std(noise_pink), 1e-6)
    print(f"  Pink noise: {len(noise_pink)} samples, std={np.std(noise_pink):.3f}")

    conditions = {
        'FULL':     {'noise': noise_1f,    'beta': beta, 'label': 'GPU 1/f noise',     'live': True},
        'WHITE':    {'noise': noise_white,  'beta': beta, 'label': 'White noise',       'live': False},
        'NO_NOISE': {'noise': noise_zero,   'beta': 0.0,  'label': 'No noise',          'live': False},
        'PINK_IIR': {'noise': noise_pink,   'beta': beta, 'label': 'Voss-McCartney 1/f','live': False},
    }

    # ─── Step 3: Generate task data ───
    print("\n[3/7] Generating task data...")
    task_a_inputs, task_a_labels = generate_task_a(n_trials, n_steps, args.sample_hz, seed=100)
    task_b_inputs, task_b_labels = generate_task_b(n_trials, n_steps, args.sample_hz, seed=200)
    task_c_inputs, task_c_labels = generate_task_c(n_trials, n_steps, args.sample_hz, seed=300)
    print(f"  Task A (waveform):  {n_trials} trials, classes {np.bincount(task_a_labels)}")
    print(f"  Task B (frequency): {n_trials} trials, classes {np.bincount(task_b_labels)}")
    print(f"  Task C (amplitude): {n_trials} trials, classes {np.bincount(task_c_labels)}")

    tasks = {
        'A': {'inputs': task_a_inputs, 'labels': task_a_labels, 'name': 'Waveform'},
        'B': {'inputs': task_b_inputs, 'labels': task_b_labels, 'name': 'Frequency'},
        'C': {'inputs': task_c_inputs, 'labels': task_c_labels, 'name': 'Amplitude'},
    }

    # ─── Step 4: Run reservoir for all conditions x tasks ───
    print("\n[4/7] Running reservoir across conditions and tasks...")
    all_features = {}  # {cond_name: {task_name: (X, y)}}

    for cond_name, cond in conditions.items():
        print(f"\n  === Condition: {cond_name} ({cond['label']}) ===")
        cond_noise = cond['noise']
        cond_beta = cond['beta']
        cond_live = cond['live']
        all_features[cond_name] = {}

        for task_name, task in tasks.items():
            print(f"    Task {task_name} ({task['name']})...")
            task_inputs = task['inputs']
            task_labels = task['labels']
            features_list = []

            for trial in range(n_trials):
                inp = task_inputs[trial]
                if fpga:
                    st = run_fpga_reservoir_trial(
                        ser, inp, cond_noise, w_in, w_noise,
                        base_vg=base_vg, alpha=alpha, beta=cond_beta,
                        sample_hz=args.sample_hz,
                        live_noise=(cond_live and fpga))
                else:
                    st = simulate_lif_reservoir(
                        inp, cond_noise, w_in, w_noise,
                        base_vg=base_vg, alpha=alpha, beta=cond_beta)
                features_list.append(build_features(st))

                if (trial + 1) % 50 == 0:
                    print(f"      trial {trial + 1}/{n_trials}")

            X = np.array(features_list)
            y = task_labels.copy()
            all_features[cond_name][task_name] = (X, y)
            print(f"      Features: {X.shape}, labels: {np.bincount(y)}")

    # ─── Step 5: Compute fresh and transfer accuracies ───
    print("\n[5/7] Computing fresh and transfer accuracies...")
    condition_results = {}

    for cond_name in conditions:
        print(f"\n  === {cond_name} ===")
        X_a, y_a = all_features[cond_name]['A']
        X_b, y_b = all_features[cond_name]['B']
        X_c, y_c = all_features[cond_name]['C']

        # Fresh accuracy (5-fold CV on each task)
        fresh_a = ridge_multiclass_cv(X_a, y_a, n_classes=3)
        fresh_b = ridge_multiclass_cv(X_b, y_b, n_classes=3)
        fresh_c = ridge_multiclass_cv(X_c, y_c, n_classes=3)
        print(f"    Fresh A: {fresh_a:.4f}  Fresh B: {fresh_b:.4f}  Fresh C: {fresh_c:.4f}")

        # Transfer: Train on A, test on B and C
        W_a = ridge_multiclass_fit(X_a, y_a, n_classes=3)
        transfer_ab = ridge_multiclass_accuracy(X_b, y_b, W_a)
        transfer_ac = ridge_multiclass_accuracy(X_c, y_c, W_a)
        print(f"    Transfer A->B: {transfer_ab:.4f}  Transfer A->C: {transfer_ac:.4f}")

        # Random readout baseline
        rng_rand = np.random.default_rng(999)
        W_rand = rng_rand.standard_normal(W_a.shape) * 0.01
        random_b = ridge_multiclass_accuracy(X_b, y_b, W_rand)
        random_c = ridge_multiclass_accuracy(X_c, y_c, W_rand)
        print(f"    Random  A->B: {random_b:.4f}  Random  A->C: {random_c:.4f}")

        # Transfer efficiency
        eff_ab = transfer_ab / fresh_b if fresh_b > 0 else 0.0
        eff_ac = transfer_ac / fresh_c if fresh_c > 0 else 0.0
        mean_eff = (eff_ab + eff_ac) / 2.0
        print(f"    Efficiency A->B: {eff_ab:.4f}  A->C: {eff_ac:.4f}  Mean: {mean_eff:.4f}")

        condition_results[cond_name] = {
            'fresh_a': fresh_a, 'fresh_b': fresh_b, 'fresh_c': fresh_c,
            'transfer_ab': transfer_ab, 'transfer_ac': transfer_ac,
            'random_b': random_b, 'random_c': random_c,
            'efficiency_ab': eff_ab, 'efficiency_ac': eff_ac,
            'mean_efficiency': mean_eff,
        }

    results['condition_results'] = condition_results

    # ─── Step 6: Evaluate tests T201-T206 ───
    print(f"\n{'=' * 65}")
    print("[6/7] Evaluating tests T201-T206...")
    print(f"{'=' * 65}")

    full = condition_results['FULL']
    white = condition_results['WHITE']
    no_noise = condition_results['NO_NOISE']

    tests = {}

    # T201: Transfer accuracy (A->B, FULL) > chance (33.3%)
    t201_pass = full['transfer_ab'] > 1.0 / 3.0
    tests['T201'] = {
        'name': 'Transfer A->B (FULL) > chance (33.3%)',
        'transfer_ab_full': full['transfer_ab'],
        'threshold': 1.0 / 3.0,
        'pass': t201_pass,
    }
    print(f"  T201: Transfer A->B (FULL)={full['transfer_ab']:.4f} > 0.333 -> "
          f"{'PASS' if t201_pass else 'FAIL'}")

    # T202: Transfer accuracy (A->B, FULL) > Transfer accuracy (A->B, WHITE)
    t202_pass = full['transfer_ab'] > white['transfer_ab']
    tests['T202'] = {
        'name': 'Transfer A->B (FULL) > Transfer A->B (WHITE)',
        'transfer_ab_full': full['transfer_ab'],
        'transfer_ab_white': white['transfer_ab'],
        'pass': t202_pass,
    }
    print(f"  T202: Transfer A->B FULL={full['transfer_ab']:.4f} > WHITE={white['transfer_ab']:.4f} -> "
          f"{'PASS' if t202_pass else 'FAIL'}")

    # T203: Transfer accuracy (A->B, FULL) > Transfer accuracy (A->B, NO_NOISE)
    t203_pass = full['transfer_ab'] > no_noise['transfer_ab']
    tests['T203'] = {
        'name': 'Transfer A->B (FULL) > Transfer A->B (NO_NOISE)',
        'transfer_ab_full': full['transfer_ab'],
        'transfer_ab_no_noise': no_noise['transfer_ab'],
        'pass': t203_pass,
    }
    print(f"  T203: Transfer A->B FULL={full['transfer_ab']:.4f} > NO_NOISE={no_noise['transfer_ab']:.4f} -> "
          f"{'PASS' if t203_pass else 'FAIL'}")

    # T204: Transfer efficiency = Transfer_acc / Fresh_acc > 0.50 for FULL
    t204_pass = full['efficiency_ab'] > 0.50
    tests['T204'] = {
        'name': 'Transfer efficiency A->B (FULL) > 0.50',
        'efficiency_ab_full': full['efficiency_ab'],
        'threshold': 0.50,
        'pass': t204_pass,
    }
    print(f"  T204: Efficiency A->B (FULL)={full['efficiency_ab']:.4f} > 0.50 -> "
          f"{'PASS' if t204_pass else 'FAIL'}")

    # T205: Transfer accuracy (A->C, FULL) > chance (33.3%)
    t205_pass = full['transfer_ac'] > 1.0 / 3.0
    tests['T205'] = {
        'name': 'Transfer A->C (FULL) > chance (33.3%)',
        'transfer_ac_full': full['transfer_ac'],
        'threshold': 1.0 / 3.0,
        'pass': t205_pass,
    }
    print(f"  T205: Transfer A->C (FULL)={full['transfer_ac']:.4f} > 0.333 -> "
          f"{'PASS' if t205_pass else 'FAIL'}")

    # T206: Mean transfer efficiency (FULL) > Mean transfer efficiency (WHITE)
    t206_pass = full['mean_efficiency'] > white['mean_efficiency']
    tests['T206'] = {
        'name': 'Mean transfer efficiency (FULL) > (WHITE)',
        'mean_eff_full': full['mean_efficiency'],
        'mean_eff_white': white['mean_efficiency'],
        'pass': t206_pass,
    }
    print(f"  T206: Mean eff FULL={full['mean_efficiency']:.4f} > WHITE={white['mean_efficiency']:.4f} -> "
          f"{'PASS' if t206_pass else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/6 PASS")
    results['tests'] = tests
    results['n_pass'] = n_pass
    results['n_total'] = 6

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2185_transfer_learning.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    # ─── Step 7: Generate figure ───
    print("\n[7/7] Generating figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('z2185: Transfer Learning — GPU-Noise-Driven FPGA Reservoir',
                      fontsize=14, fontweight='bold')

        cond_names = ['FULL', 'WHITE', 'NO_NOISE', 'PINK_IIR']
        colors = {'FULL': '#e74c3c', 'WHITE': '#3498db', 'NO_NOISE': '#95a5a6', 'PINK_IIR': '#9b59b6'}
        bar_labels = {'FULL': 'GPU 1/f', 'WHITE': 'White', 'NO_NOISE': 'No noise', 'PINK_IIR': 'Pink IIR'}

        # Panel 1: Fresh vs Transfer accuracy for Task B
        ax = axes[0, 0]
        x = np.arange(len(cond_names))
        width = 0.35
        fresh_vals = [condition_results[c]['fresh_b'] for c in cond_names]
        transfer_vals = [condition_results[c]['transfer_ab'] for c in cond_names]
        bars1 = ax.bar(x - width / 2, fresh_vals, width, label='Fresh (Task B)', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width / 2, transfer_vals, width, label='Transfer (A->B)', color='#e67e22', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([bar_labels[c] for c in cond_names], fontsize=9)
        ax.set_title('Task B: Fresh vs Transfer Accuracy')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=1.0 / 3.0, color='gray', linestyle='--', alpha=0.5, label='Chance (33.3%)')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        for bar, v in zip(bars1, fresh_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        for bar, v in zip(bars2, transfer_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)

        # Panel 2: Fresh vs Transfer accuracy for Task C
        ax = axes[0, 1]
        fresh_vals_c = [condition_results[c]['fresh_c'] for c in cond_names]
        transfer_vals_c = [condition_results[c]['transfer_ac'] for c in cond_names]
        bars1 = ax.bar(x - width / 2, fresh_vals_c, width, label='Fresh (Task C)', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width / 2, transfer_vals_c, width, label='Transfer (A->C)', color='#e67e22', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([bar_labels[c] for c in cond_names], fontsize=9)
        ax.set_title('Task C: Fresh vs Transfer Accuracy')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=1.0 / 3.0, color='gray', linestyle='--', alpha=0.5, label='Chance (33.3%)')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        for bar, v in zip(bars1, fresh_vals_c):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        for bar, v in zip(bars2, transfer_vals_c):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)

        # Panel 3: Transfer efficiency comparison
        ax = axes[1, 0]
        eff_ab_vals = [condition_results[c]['efficiency_ab'] for c in cond_names]
        eff_ac_vals = [condition_results[c]['efficiency_ac'] for c in cond_names]
        bars1 = ax.bar(x - width / 2, eff_ab_vals, width, label='A->B Efficiency', color='#e67e22', alpha=0.8)
        bars2 = ax.bar(x + width / 2, eff_ac_vals, width, label='A->C Efficiency', color='#8e44ad', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([bar_labels[c] for c in cond_names], fontsize=9)
        ax.set_title('Transfer Efficiency (transfer_acc / fresh_acc)')
        ax.set_ylabel('Efficiency')
        ax.axhline(y=0.50, color='green', linestyle=':', alpha=0.5, label='T204 threshold (0.50)')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.2)
        for bar, v in zip(bars1, eff_ab_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        for bar, v in zip(bars2, eff_ac_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)

        # Panel 4: Test results summary
        ax = axes[1, 1]
        test_names = list(tests.keys())
        test_pass = [1 if tests[t]['pass'] else 0 for t in test_names]
        bar_colors = ['#2ecc71' if p else '#e74c3c' for p in test_pass]
        bars = ax.barh(test_names, test_pass, color=bar_colors, alpha=0.8)
        ax.set_xlim(-0.1, 1.5)
        ax.set_title(f'Test Results: {n_pass}/6 PASS')
        ax.set_xlabel('PASS (1) / FAIL (0)')
        for i, (t_name, passed) in enumerate(zip(test_names, test_pass)):
            short_desc = tests[t_name]['name'][:50]
            ax.text(0.05, i, f"  {short_desc}", va='center', fontsize=7, color='black')

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2185_transfer_learning.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {fig_path}")

    except ImportError as e:
        print(f"  Matplotlib not available: {e}")
    except Exception as e:
        print(f"  Figure generation failed: {e}")

    if fpga and ser:
        ser.close()
        print("  FPGA serial closed")

    print(f"\n{'=' * 65}")
    print(f"z2185 COMPLETE: {n_pass}/6 PASS")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
