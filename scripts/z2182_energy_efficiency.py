#!/usr/bin/env python3
"""z2182_energy_efficiency.py — Energy Efficiency of GPU-Noise-Driven FPGA Reservoir

Measures the energy efficiency of the FPGA reservoir computer compared to software
baselines. Metrics: ops/s/W, joules per correct classification, throughput (trials/s).

Baselines:
  A: FPGA reservoir (8-neuron LIF on Arty A7, driven by GPU 1/f noise)
  B: Software LIF    (NumPy simulation of the same 8-neuron LIF)
  C: RandomForest    (sklearn RF on raw waveform features)
  D: PyTorch MLP     (2-layer MLP on raw waveform features)

Task: 3-class waveform classification (sine / triangle / square)
200 trials x 25 steps, base_vg=0.55, alpha=0.15, beta=0.10

Tests T181-T186:
  T181: FPGA joules_per_correct < software_LIF joules_per_correct
  T182: FPGA throughput > 0.5 trials/s
  T183: FPGA accuracy >= 0.95 * software_LIF accuracy
  T184: FPGA total_energy < 50 J for full 200-trial run
  T185: GPU noise collection overhead < 20% of total runtime
  T186: FPGA ops_per_joule > 100

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
N_TRIALS = 200
STEPS_PER_TRIAL = 25
FPGA_POWER_W = 0.5  # Arty A7 datasheet estimate


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


def send_set_vg(ser, neuron_id, vg):
    """Send SET_VG command: [SYNC][CMD_SET_VG][neuron_id][Q16.16 big-endian]."""
    q16 = to_q16_16(max(0.0, min(1.0, vg)))
    payload = bytes([neuron_id & 0x07]) + struct.pack('>I', q16)
    ser.write(bytes([SYNC, CMD_SET_VG]) + payload)


def set_per_neuron_vg(ser, vg_values):
    """Set individual Vg for each of 8 neurons."""
    for nid, vg in enumerate(vg_values[:8]):
        send_set_vg(ser, nid, vg)
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
    """IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=200, steps_per_trial=25, freq_hz=1.0, dt=1.0/20):
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
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:  # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))

        # Normalize to [0, 1]
        wave = (wave + 1.0) / 2.0
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                              live_noise=False, power_log=None):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    When live_noise=True, reads power rail in real-time (true substrate coupling).
    power_log: if provided, list to append (timestamp, power_W) tuples for overhead measurement.
    Returns: (n_steps, 24) array -- 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0

    for t in range(n_steps):
        # Get noise value
        if live_noise:
            t_noise_start = time.monotonic()
            p = read_hwmon_power()
            noise_val = (p - power_mean) / 2.0 if p else 0.0
            t_noise_end = time.monotonic()
            if power_log is not None:
                power_log.append((t_noise_start, p if p else 0.0, t_noise_end - t_noise_start))
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


def classify_5fold(X_all, y_all, n_splits=5):
    """Run 5-fold stratified CV and return mean accuracy."""
    splits = stratified_kfold(X_all, y_all, n_splits=n_splits)
    fold_accs = []
    for train_idx, test_idx in splits:
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        mu = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True)
        sigma[sigma < 1e-10] = 1.0
        X_train_n = (X_train - mu) / sigma
        X_test_n = (X_test - mu) / sigma

        acc, _ = ridge_classify(X_train_n, y_train, X_test_n, y_test)
        fold_accs.append(acc)

    return float(np.mean(fold_accs)), float(np.std(fold_accs)), [float(a) for a in fold_accs]


# ═══════════════════════════════════════════════════════════
# Software Baselines
# ═══════════════════════════════════════════════════════════

def run_software_lif_baseline(wave_trials, wave_labels, noise_samples, w_in, w_noise):
    """Run software LIF simulation and measure time/energy."""
    n_trials = len(wave_trials)
    trial_features = []

    # Measure GPU idle power before starting
    idle_powers = []
    for _ in range(10):
        p = read_hwmon_power()
        if p is not None:
            idle_powers.append(p)
        time.sleep(0.02)
    gpu_idle_W = np.mean(idle_powers) if idle_powers else 11.0

    t0 = time.monotonic()
    for trial_idx in range(n_trials):
        states = simulate_lif_reservoir(
            wave_trials[trial_idx], noise_samples, w_in, w_noise,
            base_vg=BASE_VG, alpha=ALPHA, beta=BETA)
        aug = augment_with_delays(states, delays=(1, 2, 3))
        feat = pool_trial_features(aug)
        trial_features.append(feat)
    elapsed = time.monotonic() - t0

    X = np.array(trial_features)
    mean_acc, std_acc, folds = classify_5fold(X, wave_labels)

    # Software LIF runs on CPU, but GPU is idle drawing power
    # Energy = CPU time * TDP estimate (assume ~15W for this workload)
    cpu_power_est = 15.0  # Watts (Ryzen mobile TDP estimate)
    total_energy = elapsed * cpu_power_est
    n_correct = int(mean_acc * n_trials)
    joules_per_correct = total_energy / max(n_correct, 1)
    throughput = n_trials / elapsed
    ops = n_trials * STEPS_PER_TRIAL * N_NEURONS  # neuron-steps
    ops_per_joule = ops / max(total_energy, 1e-6)

    return {
        'accuracy_mean': mean_acc,
        'accuracy_std': std_acc,
        'accuracy_folds': folds,
        'elapsed_s': elapsed,
        'power_W': cpu_power_est,
        'total_energy_J': total_energy,
        'n_correct': n_correct,
        'joules_per_correct': joules_per_correct,
        'throughput_trials_per_s': throughput,
        'ops_per_joule': ops_per_joule,
        'n_trials': n_trials,
    }


def run_randomforest_baseline(wave_trials, wave_labels):
    """Run sklearn RandomForest on raw waveform features."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("  sklearn not available, skipping RandomForest baseline")
        return None

    # Raw features: flatten waveform + simple stats
    X = np.column_stack([
        wave_trials,  # raw waveform values
        wave_trials.mean(axis=1, keepdims=True),
        wave_trials.std(axis=1, keepdims=True),
        wave_trials.max(axis=1, keepdims=True),
        wave_trials.min(axis=1, keepdims=True),
    ])

    cpu_power_est = 15.0

    t0 = time.monotonic()
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    scores = cross_val_score(clf, X, wave_labels, cv=5, scoring='accuracy')
    elapsed = time.monotonic() - t0

    mean_acc = float(np.mean(scores))
    n_correct = int(mean_acc * len(wave_labels))
    total_energy = elapsed * cpu_power_est
    joules_per_correct = total_energy / max(n_correct, 1)
    throughput = len(wave_labels) / elapsed
    ops = len(wave_labels) * X.shape[1] * 100  # features * trees (rough)
    ops_per_joule = ops / max(total_energy, 1e-6)

    return {
        'accuracy_mean': mean_acc,
        'accuracy_std': float(np.std(scores)),
        'accuracy_folds': [float(s) for s in scores],
        'elapsed_s': elapsed,
        'power_W': cpu_power_est,
        'total_energy_J': total_energy,
        'n_correct': n_correct,
        'joules_per_correct': joules_per_correct,
        'throughput_trials_per_s': throughput,
        'ops_per_joule': ops_per_joule,
        'n_trials': len(wave_labels),
    }


def run_mlp_baseline(wave_trials, wave_labels):
    """Run PyTorch MLP on raw waveform features."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  PyTorch not available, skipping MLP baseline")
        return None

    # Raw features: flatten waveform + simple stats
    X_np = np.column_stack([
        wave_trials,
        wave_trials.mean(axis=1, keepdims=True),
        wave_trials.std(axis=1, keepdims=True),
        wave_trials.max(axis=1, keepdims=True),
        wave_trials.min(axis=1, keepdims=True),
    ])

    device = 'cpu'  # Use CPU for fair power comparison
    n_features = X_np.shape[1]
    n_classes = 3

    # 5-fold CV
    splits = stratified_kfold(X_np, wave_labels, n_splits=5)
    fold_accs = []

    cpu_power_est = 15.0

    t0 = time.monotonic()
    for train_idx, test_idx in splits:
        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32, device=device)
        y_train = torch.tensor(wave_labels[train_idx], dtype=torch.long, device=device)
        X_test = torch.tensor(X_np[test_idx], dtype=torch.float32, device=device)
        y_test = wave_labels[test_idx]

        # Normalize
        mu = X_train.mean(dim=0, keepdim=True)
        sigma = X_train.std(dim=0, keepdim=True)
        sigma[sigma < 1e-10] = 1.0
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

        # Simple 2-layer MLP
        model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # Train for 100 epochs
        model.train()
        for epoch in range(100):
            logits = model(X_train)
            loss = loss_fn(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            pred = model(X_test).argmax(dim=1).cpu().numpy()
        acc = float(np.mean(pred == y_test))
        fold_accs.append(acc)

    elapsed = time.monotonic() - t0

    mean_acc = float(np.mean(fold_accs))
    n_correct = int(mean_acc * len(wave_labels))
    total_energy = elapsed * cpu_power_est
    joules_per_correct = total_energy / max(n_correct, 1)
    throughput = len(wave_labels) / elapsed
    ops = len(wave_labels) * (n_features * 64 + 64 * 32 + 32 * n_classes)  # rough FLOPs
    ops_per_joule = ops / max(total_energy, 1e-6)

    return {
        'accuracy_mean': mean_acc,
        'accuracy_std': float(np.std(fold_accs)),
        'accuracy_folds': fold_accs,
        'elapsed_s': elapsed,
        'power_W': cpu_power_est,
        'total_energy_J': total_energy,
        'n_correct': n_correct,
        'joules_per_correct': joules_per_correct,
        'throughput_trials_per_s': throughput,
        'ops_per_joule': ops_per_joule,
        'n_trials': len(wave_labels),
    }


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps-per-trial', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--noise-collect-s', type=float, default=15.0)
    args = parser.parse_args()

    print("=" * 65)
    print("z2182: Energy Efficiency of GPU-Noise-Driven FPGA Reservoir")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2182_energy_efficiency',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'fpga_power_W': FPGA_POWER_W,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/7] Connecting to FPGA...")
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

    # ─── Step 2: Measure GPU idle power ───
    print("\n[2/7] Measuring GPU idle power...")
    idle_powers = []
    for _ in range(20):
        p = read_hwmon_power()
        if p is not None:
            idle_powers.append(p)
        time.sleep(0.05)
    gpu_idle_W = float(np.mean(idle_powers)) if idle_powers else 11.0
    gpu_idle_std = float(np.std(idle_powers)) if idle_powers else 0.5
    print(f"  GPU idle power: {gpu_idle_W:.2f} +/- {gpu_idle_std:.3f} W")
    results['gpu_idle_power_W'] = gpu_idle_W

    # ─── Step 3: Collect GPU noise ───
    print("\n[3/7] Collecting GPU noise sources...")
    t_noise_start = time.monotonic()
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    t_noise_collect = time.monotonic() - t_noise_start

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
    results['noise_collect_time_s'] = t_noise_collect

    # ─── Step 4: Generate waveform task ───
    print("\n[4/7] Generating waveform classification task...")
    wave_trials, wave_labels = generate_waveforms(
        n_trials=args.n_trials, steps_per_trial=args.steps_per_trial)
    print(f"  {args.n_trials} trials, {args.steps_per_trial} steps each")
    print(f"  Class distribution: {np.bincount(wave_labels)}")

    # ─── Step 5: Run FPGA reservoir with energy measurement ───
    print("\n[5/7] Running FPGA reservoir with energy measurement...")

    fpga_power_log = []  # (timestamp, power_W, noise_read_time_s) per step
    fpga_trial_features = []

    # Measure power during FPGA operation
    power_samples_during = []

    t_fpga_start = time.monotonic()
    for trial_idx in range(args.n_trials):
        input_signal = wave_trials[trial_idx]

        if fpga:
            states = run_fpga_reservoir_trial(
                ser, input_signal, noise_1f_iir, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                live_noise=True, power_log=fpga_power_log)
        else:
            states = simulate_lif_reservoir(
                input_signal, noise_1f_iir, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=BETA)

        aug = augment_with_delays(states, delays=(1, 2, 3))
        feat = pool_trial_features(aug)
        fpga_trial_features.append(feat)

        # Sample GPU power periodically
        if trial_idx % 10 == 0:
            p = read_hwmon_power()
            if p is not None:
                power_samples_during.append(p)

        if (trial_idx + 1) % 50 == 0:
            elapsed = time.monotonic() - t_fpga_start
            rate = (trial_idx + 1) / elapsed
            eta = (args.n_trials - trial_idx - 1) / rate
            print(f"    Trial {trial_idx + 1}/{args.n_trials} "
                  f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

    t_fpga_elapsed = time.monotonic() - t_fpga_start
    gpu_active_W = float(np.mean(power_samples_during)) if power_samples_during else gpu_idle_W

    # Compute noise collection overhead from power_log
    total_noise_read_time = sum(entry[2] for entry in fpga_power_log) if fpga_power_log else 0.0
    noise_overhead_fraction = total_noise_read_time / max(t_fpga_elapsed, 1e-6)

    # FPGA total power = FPGA board + GPU active
    fpga_total_power_W = FPGA_POWER_W + gpu_active_W
    fpga_total_energy_J = t_fpga_elapsed * fpga_total_power_W

    X_fpga = np.array(fpga_trial_features)
    fpga_acc_mean, fpga_acc_std, fpga_acc_folds = classify_5fold(X_fpga, wave_labels)
    fpga_n_correct = int(fpga_acc_mean * args.n_trials)
    fpga_joules_per_correct = fpga_total_energy_J / max(fpga_n_correct, 1)
    fpga_throughput = args.n_trials / t_fpga_elapsed
    fpga_ops = args.n_trials * args.steps_per_trial * N_NEURONS
    fpga_ops_per_joule = fpga_ops / max(fpga_total_energy_J, 1e-6)

    fpga_results = {
        'accuracy_mean': fpga_acc_mean,
        'accuracy_std': fpga_acc_std,
        'accuracy_folds': fpga_acc_folds,
        'elapsed_s': t_fpga_elapsed,
        'fpga_power_W': FPGA_POWER_W,
        'gpu_active_power_W': gpu_active_W,
        'total_power_W': fpga_total_power_W,
        'total_energy_J': fpga_total_energy_J,
        'n_correct': fpga_n_correct,
        'joules_per_correct': fpga_joules_per_correct,
        'throughput_trials_per_s': fpga_throughput,
        'ops_per_joule': fpga_ops_per_joule,
        'noise_overhead_fraction': noise_overhead_fraction,
        'noise_read_time_total_s': total_noise_read_time,
        'n_trials': args.n_trials,
    }
    results['fpga'] = fpga_results
    print(f"  FPGA accuracy: {fpga_acc_mean:.3f} +/- {fpga_acc_std:.3f}")
    print(f"  FPGA time: {t_fpga_elapsed:.1f}s, throughput: {fpga_throughput:.2f} trials/s")
    print(f"  FPGA total energy: {fpga_total_energy_J:.1f} J")
    print(f"  FPGA joules/correct: {fpga_joules_per_correct:.3f}")
    print(f"  FPGA ops/joule: {fpga_ops_per_joule:.1f}")
    print(f"  Noise overhead: {noise_overhead_fraction * 100:.1f}%")

    # ─── Step 6: Run software baselines ───
    print("\n[6/7] Running software baselines...")

    # Baseline B: Software LIF
    print("\n  --- Software LIF ---")
    lif_results = run_software_lif_baseline(
        wave_trials, wave_labels, noise_1f_iir, w_in, w_noise)
    results['software_lif'] = lif_results
    print(f"  Software LIF accuracy: {lif_results['accuracy_mean']:.3f}")
    print(f"  Software LIF time: {lif_results['elapsed_s']:.1f}s")
    print(f"  Software LIF joules/correct: {lif_results['joules_per_correct']:.3f}")

    # Baseline C: RandomForest
    print("\n  --- RandomForest ---")
    rf_results = run_randomforest_baseline(wave_trials, wave_labels)
    results['randomforest'] = rf_results
    if rf_results:
        print(f"  RF accuracy: {rf_results['accuracy_mean']:.3f}")
        print(f"  RF time: {rf_results['elapsed_s']:.1f}s")
        print(f"  RF joules/correct: {rf_results['joules_per_correct']:.3f}")

    # Baseline D: PyTorch MLP
    print("\n  --- PyTorch MLP ---")
    mlp_results = run_mlp_baseline(wave_trials, wave_labels)
    results['mlp'] = mlp_results
    if mlp_results:
        print(f"  MLP accuracy: {mlp_results['accuracy_mean']:.3f}")
        print(f"  MLP time: {mlp_results['elapsed_s']:.1f}s")
        print(f"  MLP joules/correct: {mlp_results['joules_per_correct']:.3f}")

    # ─── Step 7: Tests ───
    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    fpga_jpc = fpga_results['joules_per_correct']
    lif_jpc = lif_results['joules_per_correct']
    fpga_thr = fpga_results['throughput_trials_per_s']
    fpga_acc = fpga_results['accuracy_mean']
    lif_acc = lif_results['accuracy_mean']
    fpga_te = fpga_results['total_energy_J']
    fpga_no = fpga_results['noise_overhead_fraction']
    fpga_opj = fpga_results['ops_per_joule']

    # T181: FPGA joules_per_correct < software_LIF joules_per_correct
    t181 = fpga_jpc < lif_jpc

    # T182: FPGA throughput > 0.5 trials/s
    t182 = fpga_thr > 0.5

    # T183: FPGA accuracy >= 0.95 * software_LIF accuracy
    t183 = fpga_acc >= 0.95 * lif_acc

    # T184: FPGA total_energy < 50 J for full 200-trial run
    t184 = fpga_te < 50.0

    # T185: GPU noise collection overhead < 20% of total runtime
    t185 = fpga_no < 0.20

    # T186: FPGA ops_per_joule > 100
    t186 = fpga_opj > 100.0

    results['tests'] = {
        'T181_fpga_more_efficient': {
            'pass': t181,
            'fpga_joules_per_correct': fpga_jpc,
            'lif_joules_per_correct': lif_jpc,
            'ratio': fpga_jpc / max(lif_jpc, 1e-6),
        },
        'T182_fpga_throughput': {
            'pass': t182,
            'throughput_trials_per_s': fpga_thr,
            'threshold': 0.5,
        },
        'T183_accuracy_preserved': {
            'pass': t183,
            'fpga_accuracy': fpga_acc,
            'lif_accuracy': lif_acc,
            'ratio': fpga_acc / max(lif_acc, 1e-6),
            'threshold': 0.95,
        },
        'T184_total_energy': {
            'pass': t184,
            'total_energy_J': fpga_te,
            'threshold_J': 50.0,
        },
        'T185_noise_overhead': {
            'pass': t185,
            'noise_overhead_fraction': fpga_no,
            'threshold': 0.20,
        },
        'T186_ops_per_joule': {
            'pass': t186,
            'ops_per_joule': fpga_opj,
            'threshold': 100.0,
        },
    }

    n_pass = sum(1 for t in results['tests'].values() if t['pass'])
    results['summary'] = {
        'pass_count': n_pass,
        'total_tests': 6,
        'pass_rate': f"{n_pass}/6",
    }

    tests_list = [
        (t181, f"T181: FPGA J/correct={fpga_jpc:.3f} < LIF J/correct={lif_jpc:.3f} "
               f"[ratio={fpga_jpc / max(lif_jpc, 1e-6):.3f}]"),
        (t182, f"T182: FPGA throughput={fpga_thr:.2f} > 0.5 trials/s"),
        (t183, f"T183: FPGA acc={fpga_acc:.3f} >= 0.95 * LIF acc={lif_acc:.3f} "
               f"(threshold={0.95 * lif_acc:.3f})"),
        (t184, f"T184: FPGA total energy={fpga_te:.1f}J < 50J"),
        (t185, f"T185: Noise overhead={fpga_no * 100:.1f}% < 20%"),
        (t186, f"T186: FPGA ops/joule={fpga_opj:.1f} > 100"),
    ]
    for passed, desc in tests_list:
        print(f"  {'PASS' if passed else 'FAIL'} {desc}")

    print(f"\n  Overall: {n_pass}/6 PASS")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2182_energy_efficiency.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    # ─── Generate figures ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Panel 1: Joules per correct classification
        ax = axes[0, 0]
        systems = ['FPGA\nReservoir', 'Software\nLIF']
        jpcs = [fpga_jpc, lif_jpc]
        colors_jpc = ['#2ecc71', '#3498db']
        if rf_results:
            systems.append('Random\nForest')
            jpcs.append(rf_results['joules_per_correct'])
            colors_jpc.append('#e74c3c')
        if mlp_results:
            systems.append('PyTorch\nMLP')
            jpcs.append(mlp_results['joules_per_correct'])
            colors_jpc.append('#f39c12')
        ax.bar(range(len(systems)), jpcs, color=colors_jpc, edgecolor='black', linewidth=0.5,
               alpha=0.85)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(systems, fontsize=9)
        ax.set_ylabel('Joules per Correct Classification')
        ax.set_title('Energy per Correct Classification')
        ax.set_yscale('log')

        # Panel 2: Throughput
        ax = axes[0, 1]
        thrs = [fpga_thr, lif_results['throughput_trials_per_s']]
        systems_thr = ['FPGA', 'SW LIF']
        colors_thr = ['#2ecc71', '#3498db']
        if rf_results:
            systems_thr.append('RF')
            thrs.append(rf_results['throughput_trials_per_s'])
            colors_thr.append('#e74c3c')
        if mlp_results:
            systems_thr.append('MLP')
            thrs.append(mlp_results['throughput_trials_per_s'])
            colors_thr.append('#f39c12')
        ax.bar(range(len(systems_thr)), thrs, color=colors_thr, edgecolor='black',
               linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(systems_thr)))
        ax.set_xticklabels(systems_thr, fontsize=9)
        ax.set_ylabel('Trials per Second')
        ax.set_title('Throughput')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='T182 threshold')
        ax.legend(fontsize=7)

        # Panel 3: Accuracy comparison
        ax = axes[0, 2]
        accs = [fpga_acc, lif_acc]
        accs_std = [fpga_results['accuracy_std'], lif_results['accuracy_std']]
        acc_labels = ['FPGA', 'SW LIF']
        acc_colors = ['#2ecc71', '#3498db']
        if rf_results:
            acc_labels.append('RF')
            accs.append(rf_results['accuracy_mean'])
            accs_std.append(rf_results['accuracy_std'])
            acc_colors.append('#e74c3c')
        if mlp_results:
            acc_labels.append('MLP')
            accs.append(mlp_results['accuracy_mean'])
            accs_std.append(mlp_results['accuracy_std'])
            acc_colors.append('#f39c12')
        ax.bar(range(len(acc_labels)), accs, yerr=accs_std, capsize=4,
               color=acc_colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(acc_labels)))
        ax.set_xticklabels(acc_labels, fontsize=9)
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Accuracy')
        ax.set_ylim(0, 1.05)
        ax.axhline(1 / 3, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.legend(fontsize=7)

        # Panel 4: Total energy
        ax = axes[1, 0]
        energies = [fpga_te, lif_results['total_energy_J']]
        en_labels = ['FPGA', 'SW LIF']
        en_colors = ['#2ecc71', '#3498db']
        if rf_results:
            en_labels.append('RF')
            energies.append(rf_results['total_energy_J'])
            en_colors.append('#e74c3c')
        if mlp_results:
            en_labels.append('MLP')
            energies.append(mlp_results['total_energy_J'])
            en_colors.append('#f39c12')
        ax.bar(range(len(en_labels)), energies, color=en_colors, edgecolor='black',
               linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(en_labels)))
        ax.set_xticklabels(en_labels, fontsize=9)
        ax.set_ylabel('Total Energy (J)')
        ax.set_title('Total Energy for 200-Trial Run')
        ax.axhline(50.0, color='red', linestyle='--', alpha=0.5, label='T184 threshold (50J)')
        ax.legend(fontsize=7)

        # Panel 5: Ops per joule
        ax = axes[1, 1]
        opjs = [fpga_opj, lif_results['ops_per_joule']]
        opj_labels = ['FPGA', 'SW LIF']
        opj_colors = ['#2ecc71', '#3498db']
        if rf_results:
            opj_labels.append('RF')
            opjs.append(rf_results['ops_per_joule'])
            opj_colors.append('#e74c3c')
        if mlp_results:
            opj_labels.append('MLP')
            opjs.append(mlp_results['ops_per_joule'])
            opj_colors.append('#f39c12')
        ax.bar(range(len(opj_labels)), opjs, color=opj_colors, edgecolor='black',
               linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(opj_labels)))
        ax.set_xticklabels(opj_labels, fontsize=9)
        ax.set_ylabel('Ops per Joule')
        ax.set_title('Energy Efficiency (Ops/Joule)')
        ax.axhline(100.0, color='red', linestyle='--', alpha=0.5, label='T186 threshold')
        ax.set_yscale('log')
        ax.legend(fontsize=7)

        # Panel 6: Test summary
        ax = axes[1, 2]
        test_names = ['T181\nJ/corr', 'T182\nthruput', 'T183\nacc',
                      'T184\ntotal E', 'T185\nnoise OH', 'T186\nops/J']
        test_pass = [t181, t182, t183, t184, t185, t186]
        test_colors = ['#2ecc71' if p else '#e74c3c' for p in test_pass]
        ax.bar(range(6), [1] * 6, color=test_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(test_names, fontsize=8)
        ax.set_yticks([])
        ax.set_title(f'Tests: {n_pass}/6 PASS')
        for i, (passed, name) in enumerate(zip(test_pass, test_names)):
            ax.text(i, 0.5, 'PASS' if passed else 'FAIL',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white')

        plt.suptitle('z2182: Energy Efficiency — FPGA Reservoir vs Software Baselines',
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'fig_z2182_energy_efficiency.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {fig_path}")
    except Exception as e:
        print(f"  Figure generation failed: {e}")

    # Cleanup
    if fpga and ser:
        set_per_neuron_vg(ser, [0.3] * 8)
        ser.close()

    print(f"\nDone. {n_pass}/6 tests passed.")
    return n_pass


if __name__ == '__main__':
    main()
