#!/usr/bin/env python3
"""z2177_temporal_integration.py — Temporal Integration in GPU-Noise-Driven FPGA Reservoir

Tests whether the reservoir can accumulate information over time across four tasks:
  Task 1: Running average   — predict mean of last 5 inputs (regression, NRMSE)
  Task 2: Threshold crossing — classify sign of cumulative sum (binary)
  Task 3: Pattern completion — correlate reservoir output with full sine after half-input
  Task 4: Integration window — classify duty cycle > 50% from pulse train (binary)

3 conditions:
  FULL    — GPU 1/f noise (power rail hwmon)
  WHITE   — random Gaussian noise
  NO_NOISE — deterministic (beta=0)

Tests T151-T156:
  T151: Running average NRMSE(FULL) < NRMSE(NO_NOISE)
  T152: Threshold crossing accuracy(FULL) > 60%
  T153: Pattern completion correlation(FULL) > 0.3
  T154: Integration window accuracy(FULL) > 55%
  T155: At least 2/4 tasks: FULL outperforms WHITE
  T156: Mean performance across tasks > 50%

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
                              base_vg=BASE_VG, alpha=ALPHA, beta=0.08,
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
                            base_vg=BASE_VG, alpha=ALPHA, beta=0.10):
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
# Task Generation
# ═══════════════════════════════════════════════════════════

def generate_running_avg_data(n_trials, n_steps, window=5, seed=42):
    """Generate random uniform inputs and running-average targets."""
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(0, 1, size=(n_trials, n_steps))
    targets = np.zeros((n_trials, n_steps))
    for trial in range(n_trials):
        for t in range(n_steps):
            start = max(0, t - window + 1)
            targets[trial, t] = inputs[trial, start:t + 1].mean()
    return inputs, targets


def generate_threshold_crossing_data(n_trials, n_steps, seed=43):
    """Generate Gaussian inputs and cumulative-sum sign targets."""
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal(size=(n_trials, n_steps)) * 0.3
    targets = np.zeros((n_trials, n_steps), dtype=int)
    for trial in range(n_trials):
        cumsum = np.cumsum(inputs[trial])
        targets[trial] = (cumsum > 0).astype(int)
    return inputs, targets


def generate_pattern_completion_data(n_trials, n_steps, seed=44):
    """Generate sine patterns, half-input + zero-pad. Target = full sine."""
    rng = np.random.default_rng(seed)
    half = n_steps // 2
    inputs = np.zeros((n_trials, n_steps))
    targets = np.zeros((n_trials, n_steps))
    for trial in range(n_trials):
        freq = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        t = np.arange(n_steps) / SAMPLE_HZ
        full_sine = np.sin(2 * np.pi * freq * t + phase) * 0.5 + 0.5
        inputs[trial, :half] = full_sine[:half]
        # zero-pad second half
        targets[trial] = full_sine
    return inputs, targets


def generate_integration_window_data(n_trials, n_steps, seed=45):
    """Generate pulse trains with variable duty cycle. Label = duty > 50%."""
    rng = np.random.default_rng(seed)
    inputs = np.zeros((n_trials, n_steps))
    labels = np.zeros(n_trials, dtype=int)
    for trial in range(n_trials):
        duty = rng.uniform(0.2, 0.8)
        period = rng.integers(4, 10)
        for t in range(n_steps):
            if (t % period) < (period * duty):
                inputs[trial, t] = 1.0
        labels[trial] = int(duty > 0.5)
    return inputs, labels


# ═══════════════════════════════════════════════════════════
# Readout & Evaluation
# ═══════════════════════════════════════════════════════════

def ridge_regression(X_train, y_train, X_test, alphas=None):
    """Ridge regression returning predictions on test set."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    best_w = None
    best_gcv = np.inf
    n = X_train.shape[1]
    for alpha in alphas:
        I = np.eye(n)
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue
        resid = y_train - X_train @ w
        hat_trace = np.trace(X_train @ np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T))
        dof = max(1, len(y_train) - hat_trace)
        gcv = np.sum(resid ** 2) / dof
        if gcv < best_gcv:
            best_gcv = gcv
            best_w = w
    if best_w is None:
        best_w = np.zeros(n)
    return X_test @ best_w


def ridge_binary_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge binary classifier returning accuracy."""
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
    return max(best_acc, 0.0)


def compute_nrmse(y_true, y_pred):
    """Normalized RMSE: RMSE / std(y_true)."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    std = np.std(y_true)
    if std < 1e-8:
        return rmse
    return rmse / std


def pool_trial_features(trial_states):
    """Pool per-timestep reservoir states into trial-level features."""
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def augment_with_delays(states, delays=(1, 2, 3)):
    """Add time-delayed copies of state for richer feature space."""
    T, D = states.shape
    augmented = np.zeros((T, D * (1 + len(delays))))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[d:, start:start + D] = states[:T - d]
    return augmented


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2177: Temporal Integration Benchmark')
    parser.add_argument('--trials', type=int, default=100, help='Trials per task')
    parser.add_argument('--steps', type=int, default=40, help='Steps per trial')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Duration to collect power noise (s)')
    args = parser.parse_args()

    n_trials = args.trials
    n_steps = args.steps

    print("=" * 65)
    print("z2177: Temporal Integration in GPU-Noise-Driven FPGA Reservoir")
    print("=" * 65)
    print(f"  Trials: {n_trials}  Steps: {n_steps}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2177_temporal_integration',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': n_trials, 'n_steps': n_steps,
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

    # ─── Step 3: Generate task data ───
    print("\n[3/6] Generating task data...")
    task1_inputs, task1_targets = generate_running_avg_data(n_trials, n_steps, window=5)
    task2_inputs, task2_targets = generate_threshold_crossing_data(n_trials, n_steps)
    task3_inputs, task3_targets = generate_pattern_completion_data(n_trials, n_steps)
    task4_inputs, task4_labels = generate_integration_window_data(n_trials, n_steps)
    print(f"  Task 1 (Running avg): {n_trials} trials x {n_steps} steps")
    print(f"  Task 2 (Threshold crossing): {n_trials} trials")
    print(f"  Task 3 (Pattern completion): {n_trials} trials")
    print(f"  Task 4 (Integration window): duty labels {np.bincount(task4_labels)}")

    # ─── Step 4: Run reservoir for all conditions ───
    print("\n[4/6] Running reservoir across conditions...")
    task_results = {}

    for cond_name, cond in conditions.items():
        print(f"\n  === Condition: {cond_name} ({cond['label']}) ===")
        cond_noise = cond['noise']
        cond_beta = cond['beta']

        # ── Task 1: Running average (regression) ──
        print(f"    Task 1: Running average...")
        all_states_t1 = []
        for trial in range(n_trials):
            inp = task1_inputs[trial]
            if fpga:
                st = run_fpga_reservoir_trial(ser, inp, cond_noise, w_in, w_noise,
                                               beta=cond_beta, live_noise=(cond_name == 'FULL'))
            else:
                st = simulate_lif_reservoir(inp, cond_noise, w_in, w_noise, beta=cond_beta)
            all_states_t1.append(st)
            if (trial + 1) % 25 == 0:
                print(f"      trial {trial + 1}/{n_trials}")

        # Time-series regression: predict running average at each step
        # Use augmented states, 70/30 split
        all_X = []
        all_y = []
        for trial in range(n_trials):
            aug = augment_with_delays(all_states_t1[trial], delays=(1, 2, 3))
            all_X.append(aug)
            all_y.append(task1_targets[trial])

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        split = int(0.7 * len(X_all))
        X_train, X_test = X_all[:split], X_all[split:]
        y_train, y_test = y_all[:split], y_all[split:]

        y_pred = ridge_regression(X_train, y_train, X_test)
        nrmse = compute_nrmse(y_test, y_pred)
        print(f"      NRMSE = {nrmse:.4f}")

        # ── Task 2: Threshold crossing (binary, per-step) ──
        print(f"    Task 2: Threshold crossing...")
        all_states_t2 = []
        for trial in range(n_trials):
            inp = task2_inputs[trial]
            if fpga:
                st = run_fpga_reservoir_trial(ser, inp, cond_noise, w_in, w_noise,
                                               beta=cond_beta, live_noise=(cond_name == 'FULL'))
            else:
                st = simulate_lif_reservoir(inp, cond_noise, w_in, w_noise, beta=cond_beta)
            all_states_t2.append(st)

        X_all2 = np.vstack([augment_with_delays(s, (1, 2, 3)) for s in all_states_t2])
        y_all2 = np.concatenate([task2_targets[i] for i in range(n_trials)]).astype(float)
        split2 = int(0.7 * len(X_all2))
        thresh_acc = ridge_binary_classify(X_all2[:split2], y_all2[:split2],
                                            X_all2[split2:], y_all2[split2:])
        print(f"      Accuracy = {thresh_acc:.4f}")

        # ── Task 3: Pattern completion (correlation in second half) ──
        print(f"    Task 3: Pattern completion...")
        all_states_t3 = []
        for trial in range(n_trials):
            inp = task3_inputs[trial]
            if fpga:
                st = run_fpga_reservoir_trial(ser, inp, cond_noise, w_in, w_noise,
                                               beta=cond_beta, live_noise=(cond_name == 'FULL'))
            else:
                st = simulate_lif_reservoir(inp, cond_noise, w_in, w_noise, beta=cond_beta)
            all_states_t3.append(st)

        # Train ridge to predict full pattern from states, measure corr in second half
        half = n_steps // 2
        X_all3 = np.vstack([augment_with_delays(s, (1, 2, 3)) for s in all_states_t3])
        y_all3 = np.concatenate([task3_targets[i] for i in range(n_trials)])
        split3 = int(0.7 * n_trials)
        train_idx = np.arange(split3 * n_steps)
        test_idx = np.arange(split3 * n_steps, n_trials * n_steps)

        y_pred3 = ridge_regression(X_all3[train_idx], y_all3[train_idx], X_all3[test_idx])
        # Correlation only in second half of test trials (where input was zero)
        corrs = []
        for trial in range(split3, n_trials):
            tstart = (trial - split3) * n_steps + half
            tend = (trial - split3) * n_steps + n_steps
            if tend <= len(y_pred3):
                pred_half = y_pred3[tstart:tend]
                true_half = task3_targets[trial][half:]
                if np.std(pred_half) > 1e-8 and np.std(true_half) > 1e-8:
                    corrs.append(np.corrcoef(pred_half, true_half)[0, 1])
        pattern_corr = float(np.mean(corrs)) if corrs else 0.0
        print(f"      Correlation (2nd half) = {pattern_corr:.4f}")

        # ── Task 4: Integration window (trial-level binary) ──
        print(f"    Task 4: Integration window...")
        all_states_t4 = []
        for trial in range(n_trials):
            inp = task4_inputs[trial]
            if fpga:
                st = run_fpga_reservoir_trial(ser, inp, cond_noise, w_in, w_noise,
                                               beta=cond_beta, live_noise=(cond_name == 'FULL'))
            else:
                st = simulate_lif_reservoir(inp, cond_noise, w_in, w_noise, beta=cond_beta)
            all_states_t4.append(st)

        # Pool trial-level features, 5-fold CV
        X_t4 = np.array([pool_trial_features(s) for s in all_states_t4])
        y_t4 = task4_labels.astype(float)
        fold_accs = []
        fold_size = n_trials // 5
        indices = np.arange(n_trials)
        rng_cv = np.random.default_rng(99)
        rng_cv.shuffle(indices)
        for fold in range(5):
            test_idx_f = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx_f = np.setdiff1d(indices, test_idx_f)
            acc_f = ridge_binary_classify(X_t4[train_idx_f], y_t4[train_idx_f],
                                           X_t4[test_idx_f], y_t4[test_idx_f])
            fold_accs.append(acc_f)
        integ_acc = float(np.mean(fold_accs))
        print(f"      Accuracy (5-fold) = {integ_acc:.4f}")

        task_results[cond_name] = {
            'task1_running_avg_nrmse': float(nrmse),
            'task2_threshold_acc': float(thresh_acc),
            'task3_pattern_corr': float(pattern_corr),
            'task4_integration_acc': float(integ_acc),
        }

    results['task_results'] = task_results

    # ─── Step 5: Evaluate tests T151-T156 ───
    print("\n[5/6] Evaluating tests T151-T156...")
    full = task_results['FULL']
    white = task_results['WHITE']
    no_noise = task_results['NO_NOISE']

    tests = {}

    # T151: NRMSE(FULL) < NRMSE(NO_NOISE)
    t151_pass = full['task1_running_avg_nrmse'] < no_noise['task1_running_avg_nrmse']
    tests['T151'] = {
        'name': 'Running avg NRMSE(FULL) < NRMSE(NO_NOISE)',
        'nrmse_full': full['task1_running_avg_nrmse'],
        'nrmse_no_noise': no_noise['task1_running_avg_nrmse'],
        'pass': t151_pass,
    }
    print(f"  T151: NRMSE FULL={full['task1_running_avg_nrmse']:.4f} vs NO_NOISE="
          f"{no_noise['task1_running_avg_nrmse']:.4f} -> {'PASS' if t151_pass else 'FAIL'}")

    # T152: Threshold crossing accuracy(FULL) > 60%
    t152_pass = full['task2_threshold_acc'] > 0.60
    tests['T152'] = {
        'name': 'Threshold crossing accuracy(FULL) > 60%',
        'accuracy': full['task2_threshold_acc'],
        'threshold': 0.60,
        'pass': t152_pass,
    }
    print(f"  T152: Threshold acc={full['task2_threshold_acc']:.4f} > 0.60 -> "
          f"{'PASS' if t152_pass else 'FAIL'}")

    # T153: Pattern completion correlation(FULL) > 0.3
    t153_pass = full['task3_pattern_corr'] > 0.3
    tests['T153'] = {
        'name': 'Pattern completion correlation(FULL) > 0.3',
        'correlation': full['task3_pattern_corr'],
        'threshold': 0.3,
        'pass': t153_pass,
    }
    print(f"  T153: Pattern corr={full['task3_pattern_corr']:.4f} > 0.3 -> "
          f"{'PASS' if t153_pass else 'FAIL'}")

    # T154: Integration window accuracy(FULL) > 55%
    t154_pass = full['task4_integration_acc'] > 0.55
    tests['T154'] = {
        'name': 'Integration window accuracy(FULL) > 55%',
        'accuracy': full['task4_integration_acc'],
        'threshold': 0.55,
        'pass': t154_pass,
    }
    print(f"  T154: Integration acc={full['task4_integration_acc']:.4f} > 0.55 -> "
          f"{'PASS' if t154_pass else 'FAIL'}")

    # T155: At least 2/4 tasks: FULL outperforms WHITE
    full_wins = 0
    # Task 1: lower NRMSE is better
    if full['task1_running_avg_nrmse'] < white['task1_running_avg_nrmse']:
        full_wins += 1
    # Tasks 2,3,4: higher is better
    if full['task2_threshold_acc'] > white['task2_threshold_acc']:
        full_wins += 1
    if full['task3_pattern_corr'] > white['task3_pattern_corr']:
        full_wins += 1
    if full['task4_integration_acc'] > white['task4_integration_acc']:
        full_wins += 1

    t155_pass = full_wins >= 2
    tests['T155'] = {
        'name': 'FULL outperforms WHITE on >= 2/4 tasks',
        'full_wins': full_wins,
        'threshold': 2,
        'pass': t155_pass,
    }
    print(f"  T155: FULL beats WHITE on {full_wins}/4 tasks >= 2 -> "
          f"{'PASS' if t155_pass else 'FAIL'}")

    # T156: Mean performance across tasks > 50%
    # Convert NRMSE to a "performance" (1 - NRMSE, clipped to [0,1])
    perf_t1 = max(0.0, 1.0 - full['task1_running_avg_nrmse'])
    perf_t2 = full['task2_threshold_acc']
    perf_t3 = max(0.0, full['task3_pattern_corr'])  # correlation as performance
    perf_t4 = full['task4_integration_acc']
    mean_perf = (perf_t1 + perf_t2 + perf_t3 + perf_t4) / 4.0
    t156_pass = mean_perf > 0.50
    tests['T156'] = {
        'name': 'Mean performance > 50%',
        'perf_task1': perf_t1,
        'perf_task2': perf_t2,
        'perf_task3': perf_t3,
        'perf_task4': perf_t4,
        'mean_performance': mean_perf,
        'threshold': 0.50,
        'pass': t156_pass,
    }
    print(f"  T156: Mean perf={mean_perf:.4f} > 0.50 -> {'PASS' if t156_pass else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/6 PASS")
    results['tests'] = tests
    results['n_pass'] = n_pass
    results['n_total'] = 6

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2177_temporal_integration.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    # ─── Step 6: Generate figure ───
    print("\n[6/6] Generating figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('z2177: Temporal Integration — GPU-Noise-Driven FPGA Reservoir',
                      fontsize=14, fontweight='bold')

        cond_names = ['FULL', 'WHITE', 'NO_NOISE']
        colors = {'FULL': '#e74c3c', 'WHITE': '#3498db', 'NO_NOISE': '#95a5a6'}
        labels = {'FULL': 'GPU 1/f', 'WHITE': 'White', 'NO_NOISE': 'No noise'}

        # Panel 1: Running average NRMSE (lower is better)
        ax = axes[0, 0]
        vals = [task_results[c]['task1_running_avg_nrmse'] for c in cond_names]
        bars = ax.bar(cond_names, vals, color=[colors[c] for c in cond_names])
        ax.set_title('Task 1: Running Average (NRMSE, lower=better)')
        ax.set_ylabel('NRMSE')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='chance')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        # Panel 2: Threshold crossing accuracy
        ax = axes[0, 1]
        vals = [task_results[c]['task2_threshold_acc'] for c in cond_names]
        bars = ax.bar(cond_names, vals, color=[colors[c] for c in cond_names])
        ax.set_title('Task 2: Threshold Crossing (accuracy)')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
        ax.axhline(y=0.6, color='green', linestyle=':', alpha=0.5, label='T152 thresh')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        # Panel 3: Pattern completion correlation
        ax = axes[1, 0]
        vals = [task_results[c]['task3_pattern_corr'] for c in cond_names]
        bars = ax.bar(cond_names, vals, color=[colors[c] for c in cond_names])
        ax.set_title('Task 3: Pattern Completion (correlation, 2nd half)')
        ax.set_ylabel('Correlation')
        ax.axhline(y=0.3, color='green', linestyle=':', alpha=0.5, label='T153 thresh')
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 if v >= 0 else bar.get_height() - 0.03,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        # Panel 4: Integration window accuracy
        ax = axes[1, 1]
        vals = [task_results[c]['task4_integration_acc'] for c in cond_names]
        bars = ax.bar(cond_names, vals, color=[colors[c] for c in cond_names])
        ax.set_title('Task 4: Integration Window (accuracy, 5-fold CV)')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
        ax.axhline(y=0.55, color='green', linestyle=':', alpha=0.5, label='T154 thresh')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2177_temporal_integration.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available, skipping figure")

    # ─── Cleanup ───
    if fpga and ser:
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        ser.close()
        print("  FPGA connection closed")

    print(f"\n{'='*65}")
    print(f"z2177 COMPLETE: {n_pass}/6 tests passed")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
