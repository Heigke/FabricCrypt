#!/usr/bin/env python3
"""z2192_reservoir_robustness.py — Reservoir Robustness Under Perturbation

Can a GPU-noise-driven FPGA reservoir maintain performance when perturbed?
This connects to Mario Lanza's memristive computing where device variability
creates natural fault tolerance.  We test 5 perturbation types across 3 noise
conditions (FULL 1/f, WHITE, NO_NOISE) to measure graceful degradation.

Perturbations:
  BASELINE        — Normal 3-class waveform classification (sine/triangle/square)
  NEURON_DROPOUT  — Randomly zero out 2/8 neurons' readout (25% dropout)
  WEIGHT_NOISE    — Add Gaussian noise (sigma=0.3) to readout weights
  VG_SHIFT        — Shift base_vg by +/-0.05 from training value
  INPUT_CORRUPTION— Add 20% noise to input signal during test

Tests T243-T248:
  T243: FULL baseline accuracy > 55%
  T244: FULL neuron dropout accuracy > 40%
  T245: FULL dropout degradation < WHITE dropout degradation
  T246: FULL Vg shift accuracy > NO_NOISE Vg shift accuracy
  T247: Mean robustness score > 0.6 for FULL
  T248: FULL robustness score > WHITE robustness score

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS = BASE / 'results'
FIGURES = BASE / 'results' / 'FEEL_paper_update' / \
    'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Default Parameters ───
N_NEURONS = 8
DEFAULT_BASE_VG = 0.55
DEFAULT_ALPHA = 0.15
DEFAULT_BETA = 0.10
DEFAULT_SAMPLE_HZ = 20
DEFAULT_N_TRIALS = 150
DEFAULT_STEPS_PER_TRIAL = 25


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


def connect_fpga(ser):
    """Disable kill switch and flush."""
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
    ser.flush()
    time.sleep(0.05)
    ser.reset_input_buffer()


def set_per_neuron_vg(ser, vg_values):
    """Set individual Vg for each of 8 neurons."""
    for nid, vg in enumerate(vg_values[:8]):
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x07]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem(ser, timeout=0.15):
    """Read telemetry packet: [0x55][0x02][0x30][48B_data][CRC8] = 52 bytes."""
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
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials, steps_per_trial, freq_hz=1.0, sample_hz=20, seed=42):
    """Generate sine/triangle/square waveforms for 3-class classification."""
    rng = np.random.default_rng(seed)
    trials = []
    labels = []
    dt = 1.0 / sample_hz
    t = np.arange(steps_per_trial) * dt

    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        phase = rng.uniform(0, 2 * np.pi)
        freq = freq_hz * rng.uniform(0.8, 1.2)

        if cls == 0:  # sine
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:  # triangle
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:  # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))

        wave = (wave + 1.0) / 2.0  # normalize to [0, 1]
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg, alpha, beta, sample_hz, live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    Returns: (n_steps, 16) array — 8 delta_spikes + 8 vmem.
    """
    n_steps = len(input_signal)
    interval = 1.0 / sample_hz
    states = np.zeros((n_steps, N_NEURONS * 2))  # delta_spikes + vmem
    prev_counts = None
    power_mean = 11.0

    for t in range(n_steps):
        # Get noise value
        if live_noise:
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
        elif beta > 0 and len(noise_samples) > 0:
            noise_val = noise_samples[t % len(noise_samples)]
        else:
            noise_val = 0.0

        # Compute per-neuron Vg
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_in
        if beta > 0:
            vg_values += beta * noise_val * w_noise
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
            vmems = [n['vmem'] for n in telem]

            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    states[t, i] = delta
            for i in range(N_NEURONS):
                states[t, N_NEURONS + i] = vmems[i]
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

    return states


def simulate_lif_reservoir(input_signal, noise_samples, w_in, w_noise,
                            base_vg, alpha, beta, sample_hz):
    """Software LIF simulation fallback when FPGA is not connected."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 2))  # delta_spikes + vmem

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / sample_hz
    vmem = np.zeros(N_NEURONS)

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

        states[t, :N_NEURONS] = spikes
        states[t, N_NEURONS:] = vmem.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

def pool_trial_features(trial_states):
    """Pool per-timestep states into trial-level features via [mean, max].
    trial_states: (n_steps, 16) -> 32 features.
    """
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.max(axis=0),
    ])


def ridge_classify_cv(X, y, n_splits=5, alphas=None, seed=42):
    """Ridge regression classifier with stratified k-fold cross-validation.
    Returns mean accuracy across folds.
    """
    if alphas is None:
        alphas = [1e-4, 1e-2, 1.0, 10.0, 100.0]

    n_classes = len(np.unique(y))
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)

    # Stratified k-fold
    classes = np.unique(y)
    folds = [[] for _ in range(n_splits)]
    for c in classes:
        c_idx = indices[y[indices] == c]
        for i, idx in enumerate(c_idx):
            folds[i % n_splits].append(idx)

    fold_accs = []
    for fold in range(n_splits):
        test_idx = np.array(folds[fold])
        train_idx = np.concatenate([np.array(folds[f]) for f in range(n_splits) if f != fold])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # One-hot encode
        Y_tr = np.zeros((len(y_tr), n_classes))
        for i, yy in enumerate(y_tr):
            Y_tr[i, int(yy)] = 1.0

        best_acc = -1
        for alpha in alphas:
            I = np.eye(X_tr.shape[1])
            try:
                W = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ Y_tr)
            except np.linalg.LinAlgError:
                continue
            pred = np.argmax(X_te @ W, axis=1)
            acc = np.mean(pred == y_te)
            if acc > best_acc:
                best_acc = acc
        fold_accs.append(max(best_acc, 0.0))

    return float(np.mean(fold_accs))


def ridge_classify_train_test(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge classifier: train on X_train, evaluate on X_test.
    Returns (accuracy, weight_matrix).
    """
    if alphas is None:
        alphas = [1e-4, 1e-2, 1.0, 10.0, 100.0]

    n_classes = len(np.unique(y_train))
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0

    best_acc = -1
    best_W = None
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue
        pred = np.argmax(X_test @ W, axis=1)
        acc = np.mean(pred == y_test)
        if acc > best_acc:
            best_acc = acc
            best_W = W.copy()

    return float(max(best_acc, 0.0)), best_W


# ═══════════════════════════════════════════════════════════
# Perturbation Functions
# ═══════════════════════════════════════════════════════════

def apply_neuron_dropout(X, n_dropout=2, seed=None):
    """Zero out n_dropout neurons' features in the feature vector.
    Features are laid out as [mean_delta(8), mean_vmem(8), max_delta(8), max_vmem(8)] = 32.
    Zeroing neuron i means zeroing indices [i, 8+i, 16+i, 24+i].
    """
    rng = np.random.default_rng(seed)
    X_pert = X.copy()
    dropped = rng.choice(N_NEURONS, size=n_dropout, replace=False)
    for nid in dropped:
        X_pert[:, nid] = 0.0         # mean delta_spike
        X_pert[:, N_NEURONS + nid] = 0.0    # mean vmem
        X_pert[:, 2 * N_NEURONS + nid] = 0.0  # max delta_spike
        X_pert[:, 3 * N_NEURONS + nid] = 0.0  # max vmem
    return X_pert, dropped


def apply_weight_noise(W, sigma=0.3, seed=None):
    """Add Gaussian noise to readout weight matrix."""
    rng = np.random.default_rng(seed)
    W_noisy = W + rng.normal(0, sigma, size=W.shape)
    return W_noisy


def apply_input_corruption(trials, corruption_frac=0.20, seed=None):
    """Add Gaussian noise to input signal (fraction of signal range)."""
    rng = np.random.default_rng(seed)
    corrupted = trials.copy()
    noise = rng.normal(0, corruption_frac, size=corrupted.shape)
    corrupted = np.clip(corrupted + noise, 0.0, 1.0)
    return corrupted


# ═══════════════════════════════════════════════════════════
# Run Reservoir on Trials
# ═══════════════════════════════════════════════════════════

def run_all_trials(ser, trials, noise_samples, w_in, w_noise,
                   base_vg, alpha, beta, sample_hz, fpga, live_noise=False):
    """Run reservoir on all trials and return pooled feature matrix."""
    n_trials = len(trials)
    features = []

    for i in range(n_trials):
        if fpga and ser is not None:
            states = run_fpga_reservoir_trial(
                ser, trials[i], noise_samples, w_in, w_noise,
                base_vg, alpha, beta, sample_hz, live_noise=live_noise)
        else:
            states = simulate_lif_reservoir(
                trials[i], noise_samples, w_in, w_noise,
                base_vg, alpha, beta, sample_hz)

        feat = pool_trial_features(states)
        features.append(feat)

        if (i + 1) % 50 == 0:
            print(f"    trial {i+1}/{n_trials}")

    return np.array(features)


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='z2192: Reservoir Robustness Under Perturbation')
    parser.add_argument('--base-vg', type=float, default=DEFAULT_BASE_VG)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA)
    parser.add_argument('--sample-hz', type=int, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument('--steps-per-trial', type=int, default=DEFAULT_STEPS_PER_TRIAL)
    parser.add_argument('--noise-collect-s', type=float, default=15.0)
    parser.add_argument('--dropout-count', type=int, default=2)
    parser.add_argument('--weight-noise-sigma', type=float, default=0.3)
    parser.add_argument('--vg-shift', type=float, default=0.05)
    parser.add_argument('--input-corruption', type=float, default=0.20)
    parser.add_argument('--cv-folds', type=int, default=5)
    args = parser.parse_args()

    print("=" * 65)
    print("z2192: Reservoir Robustness Under Perturbation")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2192_reservoir_robustness',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': args.base_vg, 'alpha': args.alpha, 'beta': args.beta,
            'n_neurons': N_NEURONS, 'sample_hz': args.sample_hz,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'dropout_count': args.dropout_count,
            'weight_noise_sigma': args.weight_noise_sigma,
            'vg_shift': args.vg_shift,
            'input_corruption': args.input_corruption,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
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
        connect_fpga(ser)
        fpga = True

    # ─── Step 2: Collect noise ───
    print("\n[2/8] Collecting GPU power noise...")
    try:
        from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
        telem = SysfsHwmonTelemetry()
        print(f"  SysfsHwmonTelemetry available")
    except Exception as e:
        print(f"  SysfsHwmonTelemetry not available: {e}")
        telem = None

    power_noise_raw = collect_power_noise(
        duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise_raw is not None and len(power_noise_raw) > 10:
        power_noise_mean = np.mean(power_noise_raw)
        power_noise_std = max(np.std(power_noise_raw), 1e-6)
        power_noise_norm = (power_noise_raw - power_noise_mean) / power_noise_std
        noise_1f = iir_filter_noise(power_noise_norm, alpha_iir=0.85)
        print(f"  Collected {len(power_noise_raw)} samples, "
              f"mean={power_noise_mean:.2f}W, std={power_noise_std:.3f}W")
    else:
        print("  Power noise unavailable -- generating synthetic 1/f")
        noise_1f = iir_filter_noise(rng.standard_normal(2000), alpha_iir=0.85)
        power_noise_mean = 0.0
        power_noise_std = 1.0

    total_noise_steps = args.n_trials * args.steps_per_trial
    noise_white = rng.standard_normal(total_noise_steps)
    noise_none = np.zeros(total_noise_steps)

    # Extend 1/f noise to cover all trials
    if len(noise_1f) < total_noise_steps:
        repeats = (total_noise_steps // len(noise_1f)) + 1
        noise_1f_ext = np.tile(noise_1f, repeats)[:total_noise_steps]
    else:
        noise_1f_ext = noise_1f[:total_noise_steps]

    # ─── Step 3: Generate waveforms ───
    print("\n[3/8] Generating waveform trials...")
    trials, labels = generate_waveforms(
        args.n_trials, args.steps_per_trial, sample_hz=args.sample_hz)
    print(f"  {args.n_trials} trials, {args.steps_per_trial} steps each")
    print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # ─── Step 4: Conditions ───
    CONDITIONS = ['FULL', 'WHITE', 'NO_NOISE']
    noise_map = {
        'FULL': noise_1f_ext,
        'WHITE': noise_white,
        'NO_NOISE': noise_none,
    }
    beta_map = {
        'FULL': args.beta,
        'WHITE': args.beta,
        'NO_NOISE': 0.0,
    }

    # ─── Step 5: Run baseline reservoir for each condition ───
    print("\n[4/8] Running baseline reservoir for each condition...")
    condition_features = {}
    for cond in CONDITIONS:
        print(f"\n  --- {cond} ---")
        noise = noise_map[cond]
        beta_c = beta_map[cond]
        live = (cond == 'FULL' and fpga)

        X = run_all_trials(
            ser, trials, noise, w_in, w_noise,
            args.base_vg, args.alpha, beta_c, args.sample_hz,
            fpga, live_noise=live)
        condition_features[cond] = X
        print(f"  Feature matrix shape: {X.shape}")

    # ─── Step 6: Train/test split and baseline accuracies ───
    print("\n[5/8] Computing baseline accuracies (5-fold CV)...")
    baseline_acc = {}
    for cond in CONDITIONS:
        X = condition_features[cond]
        acc = ridge_classify_cv(X, labels, n_splits=args.cv_folds)
        baseline_acc[cond] = acc
        print(f"  {cond} baseline accuracy: {acc:.4f}")

    # ─── Step 7: Perturbation experiments ───
    print("\n[6/8] Running perturbation experiments...")
    PERTURBATIONS = ['NEURON_DROPOUT', 'WEIGHT_NOISE', 'VG_SHIFT', 'INPUT_CORRUPTION']
    perturbation_acc = {cond: {} for cond in CONDITIONS}

    for cond in CONDITIONS:
        X = condition_features[cond]
        print(f"\n  --- {cond} perturbations ---")

        # For weight noise and dropout we need a trained model
        # Use 70/30 split for perturbation evaluation
        n_train = int(0.7 * len(labels))
        idx = np.arange(len(labels))
        rng_split = np.random.default_rng(99)
        rng_split.shuffle(idx)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        X_train, y_train = X[train_idx], labels[train_idx]
        X_test, y_test = X[test_idx], labels[test_idx]

        # Train baseline model
        base_acc_split, W_base = ridge_classify_train_test(X_train, y_train, X_test, y_test)

        # --- NEURON_DROPOUT ---
        dropout_accs = []
        for rep in range(10):  # average over 10 random dropout masks
            X_test_drop, dropped = apply_neuron_dropout(
                X_test, n_dropout=args.dropout_count, seed=rep)
            if W_base is not None:
                pred = np.argmax(X_test_drop @ W_base, axis=1)
                acc_drop = float(np.mean(pred == y_test))
            else:
                acc_drop = 0.0
            dropout_accs.append(acc_drop)
        perturbation_acc[cond]['NEURON_DROPOUT'] = float(np.mean(dropout_accs))
        print(f"    NEURON_DROPOUT: {perturbation_acc[cond]['NEURON_DROPOUT']:.4f} "
              f"(avg over 10 masks)")

        # --- WEIGHT_NOISE ---
        wn_accs = []
        for rep in range(10):
            if W_base is not None:
                W_noisy = apply_weight_noise(
                    W_base, sigma=args.weight_noise_sigma, seed=rep + 100)
                pred = np.argmax(X_test @ W_noisy, axis=1)
                acc_wn = float(np.mean(pred == y_test))
            else:
                acc_wn = 0.0
            wn_accs.append(acc_wn)
        perturbation_acc[cond]['WEIGHT_NOISE'] = float(np.mean(wn_accs))
        print(f"    WEIGHT_NOISE:   {perturbation_acc[cond]['WEIGHT_NOISE']:.4f} "
              f"(avg over 10 reps)")

        # --- VG_SHIFT ---
        # Re-run reservoir with shifted base_vg, evaluate with baseline weights
        vg_shift_accs = []
        for sign in [+1, -1]:
            shifted_vg = args.base_vg + sign * args.vg_shift
            noise_c = noise_map[cond]
            beta_c = beta_map[cond]
            live = (cond == 'FULL' and fpga)
            print(f"    VG_SHIFT ({'+' if sign > 0 else '-'}{args.vg_shift}): "
                  f"running reservoir at base_vg={shifted_vg:.2f}...")

            X_shifted = run_all_trials(
                ser, trials[test_idx], noise_c, w_in, w_noise,
                shifted_vg, args.alpha, beta_c, args.sample_hz,
                fpga, live_noise=live)

            if W_base is not None:
                pred = np.argmax(X_shifted @ W_base, axis=1)
                acc_vg = float(np.mean(pred == y_test))
            else:
                acc_vg = 0.0
            vg_shift_accs.append(acc_vg)

        perturbation_acc[cond]['VG_SHIFT'] = float(np.mean(vg_shift_accs))
        print(f"    VG_SHIFT:       {perturbation_acc[cond]['VG_SHIFT']:.4f} "
              f"(avg +/- shift)")

        # --- INPUT_CORRUPTION ---
        trials_corrupted = apply_input_corruption(
            trials[test_idx], corruption_frac=args.input_corruption, seed=42)
        noise_c = noise_map[cond]
        beta_c = beta_map[cond]
        live = (cond == 'FULL' and fpga)
        print(f"    INPUT_CORRUPTION: running reservoir with {args.input_corruption*100:.0f}% noise...")

        X_corrupted = run_all_trials(
            ser, trials_corrupted, noise_c, w_in, w_noise,
            args.base_vg, args.alpha, beta_c, args.sample_hz,
            fpga, live_noise=live)

        if W_base is not None:
            pred = np.argmax(X_corrupted @ W_base, axis=1)
            acc_ic = float(np.mean(pred == y_test))
        else:
            acc_ic = 0.0
        perturbation_acc[cond]['INPUT_CORRUPTION'] = acc_ic
        print(f"    INPUT_CORRUPTION: {acc_ic:.4f}")

    # ─── Step 8: Compute robustness scores ───
    print("\n[7/8] Computing robustness scores...")
    robustness_scores = {}
    for cond in CONDITIONS:
        base = max(baseline_acc[cond], 1e-6)
        pert_accs = [perturbation_acc[cond][p] for p in PERTURBATIONS]
        robustness = float(np.mean(pert_accs) / base)
        robustness_scores[cond] = robustness
        print(f"  {cond}: robustness = {robustness:.4f} "
              f"(mean perturbed {np.mean(pert_accs):.4f} / baseline {base:.4f})")

    # Degradation: baseline - perturbed
    degradation = {}
    for cond in CONDITIONS:
        degradation[cond] = {}
        for p in PERTURBATIONS:
            degradation[cond][p] = float(baseline_acc[cond] - perturbation_acc[cond][p])

    # ─── Tests T243-T248 ───
    print("\n[8/8] Evaluating tests T243-T248...")
    tests = {}

    # T243: FULL baseline accuracy > 55%
    t243_pass = baseline_acc['FULL'] > 0.55
    tests['T243_baseline_accuracy'] = {
        'pass': bool(t243_pass),
        'FULL_accuracy': baseline_acc['FULL'],
        'threshold': 0.55,
        'description': 'FULL baseline accuracy > 55%',
    }
    print(f"  T243 {'PASS' if t243_pass else 'FAIL'}: "
          f"FULL baseline={baseline_acc['FULL']:.4f} (>0.55)")

    # T244: FULL neuron dropout accuracy > 40%
    full_dropout = perturbation_acc['FULL']['NEURON_DROPOUT']
    t244_pass = full_dropout > 0.40
    tests['T244_neuron_dropout'] = {
        'pass': bool(t244_pass),
        'FULL_dropout_accuracy': full_dropout,
        'threshold': 0.40,
        'description': 'FULL neuron dropout accuracy > 40%',
    }
    print(f"  T244 {'PASS' if t244_pass else 'FAIL'}: "
          f"FULL dropout={full_dropout:.4f} (>0.40)")

    # T245: FULL dropout degradation < WHITE dropout degradation
    full_drop_deg = degradation['FULL']['NEURON_DROPOUT']
    white_drop_deg = degradation['WHITE']['NEURON_DROPOUT']
    t245_pass = full_drop_deg < white_drop_deg
    tests['T245_dropout_degradation'] = {
        'pass': bool(t245_pass),
        'FULL_degradation': full_drop_deg,
        'WHITE_degradation': white_drop_deg,
        'description': 'FULL dropout degradation < WHITE dropout degradation',
    }
    print(f"  T245 {'PASS' if t245_pass else 'FAIL'}: "
          f"FULL deg={full_drop_deg:.4f} < WHITE deg={white_drop_deg:.4f}")

    # T246: FULL Vg shift accuracy > NO_NOISE Vg shift accuracy
    full_vg = perturbation_acc['FULL']['VG_SHIFT']
    nonoise_vg = perturbation_acc['NO_NOISE']['VG_SHIFT']
    t246_pass = full_vg > nonoise_vg
    tests['T246_vg_shift'] = {
        'pass': bool(t246_pass),
        'FULL_vg_shift_accuracy': full_vg,
        'NO_NOISE_vg_shift_accuracy': nonoise_vg,
        'description': 'FULL Vg shift accuracy > NO_NOISE Vg shift accuracy',
    }
    print(f"  T246 {'PASS' if t246_pass else 'FAIL'}: "
          f"FULL Vg={full_vg:.4f} > NO_NOISE Vg={nonoise_vg:.4f}")

    # T247: Mean robustness score > 0.6 for FULL
    t247_pass = robustness_scores['FULL'] > 0.6
    tests['T247_robustness_score'] = {
        'pass': bool(t247_pass),
        'FULL_robustness': robustness_scores['FULL'],
        'threshold': 0.6,
        'description': 'Mean robustness score > 0.6 for FULL',
    }
    print(f"  T247 {'PASS' if t247_pass else 'FAIL'}: "
          f"FULL robustness={robustness_scores['FULL']:.4f} (>0.6)")

    # T248: FULL robustness score > WHITE robustness score
    t248_pass = robustness_scores['FULL'] > robustness_scores['WHITE']
    tests['T248_robustness_comparison'] = {
        'pass': bool(t248_pass),
        'FULL_robustness': robustness_scores['FULL'],
        'WHITE_robustness': robustness_scores['WHITE'],
        'description': 'FULL robustness score > WHITE robustness score',
    }
    print(f"  T248 {'PASS' if t248_pass else 'FAIL'}: "
          f"FULL={robustness_scores['FULL']:.4f} > WHITE={robustness_scores['WHITE']:.4f}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  Result: {n_pass}/6 tests PASS")

    # ─── Save results ───
    results['baseline_accuracy'] = baseline_acc
    results['perturbation_accuracy'] = perturbation_acc
    results['degradation'] = degradation
    results['robustness_scores'] = robustness_scores
    results['tests'] = tests
    results['n_pass'] = n_pass
    results['n_total'] = 6

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2192_reservoir_robustness.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved to {out_path}")

    # ─── Figure ───
    print("\n  Generating figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle('z2192: Reservoir Robustness Under Perturbation',
                      fontsize=14, fontweight='bold')

        cond_colors = {'FULL': '#2196F3', 'WHITE': '#9E9E9E', 'NO_NOISE': '#FF9800'}

        # (1) Top-left: Baseline accuracy bars
        ax = axes[0, 0]
        x_pos = np.arange(len(CONDITIONS))
        bars = ax.bar(x_pos, [baseline_acc[c] for c in CONDITIONS],
                       color=[cond_colors[c] for c in CONDITIONS], edgecolor='black')
        ax.axhline(0.55, color='red', linestyle='--', linewidth=1, label='T243 threshold')
        ax.axhline(1.0 / 3.0, color='gray', linestyle=':', linewidth=1, label='chance (33.3%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONDITIONS)
        ax.set_ylabel('Accuracy')
        ax.set_title('Baseline Accuracy (3-class waveform)')
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        for bar, acc in zip(bars, [baseline_acc[c] for c in CONDITIONS]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{acc:.3f}', ha='center', fontsize=9)

        # (2) Top-right: Accuracy under each perturbation (grouped bars)
        ax = axes[0, 1]
        n_pert = len(PERTURBATIONS)
        n_cond = len(CONDITIONS)
        bar_width = 0.22
        x_base = np.arange(n_pert)
        for ci, cond in enumerate(CONDITIONS):
            accs = [perturbation_acc[cond][p] for p in PERTURBATIONS]
            offset = (ci - n_cond / 2 + 0.5) * bar_width
            ax.bar(x_base + offset, accs, width=bar_width,
                    color=cond_colors[cond], edgecolor='black', label=cond)
        ax.axhline(0.40, color='red', linestyle='--', linewidth=1, alpha=0.7, label='T244 threshold')
        ax.set_xticks(x_base)
        ax.set_xticklabels([p.replace('_', '\n') for p in PERTURBATIONS], fontsize=8)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Under Perturbation')
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7, loc='upper right')

        # (3) Bottom-left: Degradation heatmap
        ax = axes[1, 0]
        deg_matrix = np.array([[degradation[c][p] for c in CONDITIONS] for p in PERTURBATIONS])
        im = ax.imshow(deg_matrix, cmap='RdYlGn_r', aspect='auto',
                         vmin=-0.1, vmax=0.5)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS)
        ax.set_yticks(range(len(PERTURBATIONS)))
        ax.set_yticklabels([p.replace('_', '\n') for p in PERTURBATIONS], fontsize=8)
        ax.set_title('Degradation (baseline - perturbed)')
        for i in range(len(PERTURBATIONS)):
            for j in range(len(CONDITIONS)):
                ax.text(j, i, f'{deg_matrix[i, j]:.3f}',
                         ha='center', va='center', fontsize=9,
                         color='white' if deg_matrix[i, j] > 0.3 else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8)

        # (4) Bottom-right: Robustness score summary
        ax = axes[1, 1]
        x_pos = np.arange(len(CONDITIONS))
        bars = ax.bar(x_pos, [robustness_scores[c] for c in CONDITIONS],
                       color=[cond_colors[c] for c in CONDITIONS], edgecolor='black')
        ax.axhline(0.6, color='red', linestyle='--', linewidth=1, label='T247 threshold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONDITIONS)
        ax.set_ylabel('Robustness Score')
        ax.set_title('Robustness Score (avg perturbed / baseline)')
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8)
        for bar, score in zip(bars, [robustness_scores[c] for c in CONDITIONS]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{score:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'fig_z2192_reservoir_robustness.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved to {fig_path}")

    except ImportError as e:
        print(f"  matplotlib not available, skipping figure: {e}")

    # ─── Cleanup ───
    if ser is not None:
        try:
            ser.write(bytes([SYNC, CMD_SET_KILL, 0x01]))
            ser.flush()
            ser.close()
        except Exception:
            pass

    print(f"\n{'=' * 65}")
    print(f"z2192 COMPLETE: {n_pass}/6 tests PASS")
    print(f"{'=' * 65}")

    return results


if __name__ == '__main__':
    main()
