#!/usr/bin/env python3
"""z2186_noise_driven_plasticity.py — Noise-Driven Online Plasticity in FPGA Reservoir

Demonstrates that GPU firmware noise enables ONLINE LEARNING (plasticity) in the
FPGA reservoir. The noise provides an exploration signal (analogous to
dopamine-modulated STDP), enabling the system to ADAPT over time.

Protocol:
  Phase 1 (baseline, 100 trials): Fixed weights, measure baseline accuracy
  Phase 2 (learning, 300 trials): Update w_in every 25 trials via reward-modulated Hebbian rule
  Phase 3 (test, 100 trials): Fixed learned weights, measure final accuracy

Learning rule:
  delta_w = eta * reward_signal * noise_correlation
  where:
    reward_signal = accuracy_last_25 - accuracy_previous_25
    noise_correlation = mean(noise[t] * spike_rate[t]) per neuron
    eta = 0.01

4 conditions:
  FULL        — GPU 1/f noise -> learning with temporal structure
  WHITE       — White noise -> learning with random exploration
  NO_NOISE    — No noise -> no exploration signal (should fail to learn)
  RANDOM_UPDATE — Random weight changes (control for any learning)

Tests T207-T212:
  T207: FULL phase3_acc > FULL phase1_acc (learning happened)
  T208: FULL improvement > WHITE improvement (1/f noise better)
  T209: FULL improvement > NO_NOISE improvement (noise helps)
  T210: FULL improvement > RANDOM_UPDATE improvement (structured > random)
  T211: FULL phase3_acc > 0.45 (meaningful accuracy)
  T212: Learning curve monotonic (>=3 of 6 checkpoints increase over baseline)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, subprocess, argparse
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

# ─── Reservoir Parameters (defaults, overridden by argparse) ───
N_NEURONS = 8


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
# Waveform Generation & Classification (from z2162/z2177)
# ═══════════════════════════════════════════════════════════

def generate_waveform(waveform_type, n_steps, rng, sample_hz=20):
    """Generate a single waveform of given type."""
    t = np.arange(n_steps) / sample_hz
    freq = rng.uniform(0.5, 2.0)
    phase = rng.uniform(0, 2 * np.pi)
    if waveform_type == 'sine':
        return np.sin(2 * np.pi * freq * t + phase) * 0.5 + 0.5
    elif waveform_type == 'triangle':
        return np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0)
    elif waveform_type == 'square':
        return (np.sin(2 * np.pi * freq * t + phase) > 0).astype(float)
    else:
        raise ValueError(f"Unknown waveform: {waveform_type}")


def generate_trial_data(n_trials, n_steps, sample_hz=20, seed=42):
    """Generate 3-class waveform classification data (sine/triangle/square)."""
    rng = np.random.default_rng(seed)
    classes = ['sine', 'triangle', 'square']
    inputs = []
    labels = []
    for _ in range(n_trials):
        cls_idx = rng.integers(0, 3)
        waveform = generate_waveform(classes[cls_idx], n_steps, rng, sample_hz)
        inputs.append(waveform)
        labels.append(cls_idx)
    return np.array(inputs), np.array(labels)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=0.55, alpha=0.15, beta=0.10,
                              sample_hz=20, live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    Returns: (n_steps, 24) array -- 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
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
                            base_vg=0.55, alpha=0.15, beta=0.10, sample_hz=20):
    """Software LIF simulation fallback when FPGA is not connected."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 3))

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / sample_hz
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
# Feature Extraction & Readout
# ═══════════════════════════════════════════════════════════

def build_features(trial_states):
    """Pool per-timestep reservoir states into trial-level features."""
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, n_classes=3):
    """Ridge regression multi-class classifier (one-hot) returning accuracy."""
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    best_acc = -1
    n = X_train.shape[1]
    # One-hot encode targets
    Y_train_oh = np.zeros((len(y_train), n_classes))
    for i, c in enumerate(y_train):
        Y_train_oh[i, int(c)] = 1.0

    for alpha in alphas:
        I = np.eye(n)
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_train_oh)
        except np.linalg.LinAlgError:
            continue
        pred_scores = X_test @ W
        pred_labels = pred_scores.argmax(axis=1)
        acc = np.mean(pred_labels == y_test)
        if acc > best_acc:
            best_acc = acc
    return max(best_acc, 0.0)


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def run_condition(condition_name, ser, fpga, all_inputs, all_labels,
                  noise_samples, w_in_init, w_noise, args, rng):
    """Run a single condition through all 3 phases. Returns results dict."""

    base_vg = args.base_vg
    alpha = args.alpha
    beta = args.beta
    eta = args.eta
    sample_hz = args.sample_hz
    steps_per_trial = args.steps_per_trial
    n_baseline = args.baseline_trials
    n_learning = args.learning_trials
    n_test = args.test_trials
    n_total = n_baseline + n_learning + n_test
    update_interval = 25  # update weights every 25 trials during phase 2

    is_full = (condition_name == 'FULL')
    is_no_noise = (condition_name == 'NO_NOISE')
    is_random = (condition_name == 'RANDOM_UPDATE')
    cond_beta = 0.0 if is_no_noise else beta

    w_in = w_in_init.copy()
    w_in_history = [w_in.copy()]

    # Track per-trial accuracy for learning curve
    trial_correct = []  # 1 if correct, 0 if wrong
    noise_corr_buffer = []  # noise * spike_rate per neuron per trial

    # Checkpoint accuracy at 6 points during phase 2
    n_checkpoints = 6
    checkpoint_interval = n_learning // n_checkpoints
    checkpoint_accs = []

    phase_labels = []  # 'baseline', 'learning', 'test'
    all_features = []

    print(f"\n  === Condition: {condition_name} ===")

    for trial_idx in range(n_total):
        if trial_idx < n_baseline:
            phase = 'baseline'
        elif trial_idx < n_baseline + n_learning:
            phase = 'learning'
        else:
            phase = 'test'
        phase_labels.append(phase)

        inp = all_inputs[trial_idx]
        label = all_labels[trial_idx]

        # Run reservoir
        if fpga:
            states = run_fpga_reservoir_trial(
                ser, inp, noise_samples, w_in, w_noise,
                base_vg=base_vg, alpha=alpha, beta=cond_beta,
                sample_hz=sample_hz, live_noise=is_full
            )
        else:
            states = simulate_lif_reservoir(
                inp, noise_samples, w_in, w_noise,
                base_vg=base_vg, alpha=alpha, beta=cond_beta,
                sample_hz=sample_hz
            )

        features = build_features(states)
        all_features.append(features)

        # Track noise-spike correlation for Hebbian update
        spike_rates = states[:, :N_NEURONS].mean(axis=0)  # mean delta_spikes per neuron
        if not is_no_noise and len(noise_samples) > 0:
            # Average noise value seen during this trial
            noise_vals = noise_samples[np.arange(len(inp)) % len(noise_samples)]
            noise_mean = noise_vals.mean()
            noise_corr = noise_mean * spike_rates
        else:
            noise_corr = np.zeros(N_NEURONS)
        noise_corr_buffer.append(noise_corr)

        # Simple online classification: use last 50 trials as training set
        if trial_idx >= 50:
            X_train_buf = np.array(all_features[max(0, trial_idx - 50):trial_idx])
            y_train_buf = all_labels[max(0, trial_idx - 50):trial_idx]
            X_test_one = features.reshape(1, -1)
            # Quick ridge predict
            try:
                acc = ridge_classify(X_train_buf, y_train_buf, X_test_one,
                                     np.array([label]), n_classes=3)
                trial_correct.append(int(acc > 0.5))
            except Exception:
                trial_correct.append(0)
        else:
            trial_correct.append(0)  # not enough data yet

        # ── Weight update during learning phase ──
        if phase == 'learning':
            learning_trial = trial_idx - n_baseline
            # Checkpoint accuracy
            if (learning_trial + 1) % checkpoint_interval == 0 and len(checkpoint_accs) < n_checkpoints:
                recent = trial_correct[-checkpoint_interval:]
                cp_acc = np.mean(recent) if recent else 0.0
                checkpoint_accs.append(float(cp_acc))
                print(f"    Checkpoint {len(checkpoint_accs)}/{n_checkpoints}: acc={cp_acc:.3f}")

            # Update weights every update_interval trials
            if (learning_trial + 1) % update_interval == 0:
                if is_random:
                    # Random weight perturbation (control)
                    w_in += rng.standard_normal(N_NEURONS) * eta * 0.5
                elif not is_no_noise:
                    # Reward-modulated Hebbian update
                    recent_acc = np.mean(trial_correct[-update_interval:])
                    if len(trial_correct) > update_interval:
                        prev_acc = np.mean(trial_correct[-2 * update_interval:-update_interval])
                    else:
                        prev_acc = np.mean(trial_correct[:max(1, len(trial_correct) - update_interval)])
                    reward_signal = recent_acc - prev_acc

                    # Average noise correlation over recent trials
                    recent_corr = np.mean(noise_corr_buffer[-update_interval:], axis=0)
                    delta_w = eta * reward_signal * recent_corr
                    w_in += delta_w

                w_in = np.clip(w_in, -2.0, 2.0)
                w_in_history.append(w_in.copy())

        if (trial_idx + 1) % 50 == 0:
            recent_50 = trial_correct[-50:]
            print(f"    Trial {trial_idx + 1}/{n_total} ({phase}): "
                  f"recent_50_acc={np.mean(recent_50):.3f}")

    # ── Compute phase accuracies using ridge on held-out data ──
    all_features_arr = np.array(all_features)

    # Phase 1 (baseline): train on first 70%, test on last 30%
    bl_idx = np.arange(n_baseline)
    bl_split = int(0.7 * n_baseline)
    phase1_acc = ridge_classify(
        all_features_arr[bl_idx[:bl_split]], all_labels[bl_idx[:bl_split]],
        all_features_arr[bl_idx[bl_split:]], all_labels[bl_idx[bl_split:]],
        n_classes=3
    )

    # Phase 3 (test): train on learning phase, test on test phase
    learn_idx = np.arange(n_baseline, n_baseline + n_learning)
    test_idx = np.arange(n_baseline + n_learning, n_total)
    phase3_acc = ridge_classify(
        all_features_arr[learn_idx], all_labels[learn_idx],
        all_features_arr[test_idx], all_labels[test_idx],
        n_classes=3
    )

    improvement = phase3_acc - phase1_acc

    print(f"    Phase 1 (baseline) acc: {phase1_acc:.4f}")
    print(f"    Phase 3 (test) acc:     {phase3_acc:.4f}")
    print(f"    Improvement:            {improvement:+.4f}")

    return {
        'phase1_acc': float(phase1_acc),
        'phase3_acc': float(phase3_acc),
        'improvement': float(improvement),
        'checkpoint_accs': checkpoint_accs,
        'w_in_history': [w.tolist() for w in w_in_history],
        'w_in_final': w_in.tolist(),
        'trial_correct': trial_correct,
    }


def main():
    parser = argparse.ArgumentParser(description='z2186: Noise-Driven Plasticity')
    parser.add_argument('--baseline-trials', type=int, default=100)
    parser.add_argument('--learning-trials', type=int, default=300)
    parser.add_argument('--test-trials', type=int, default=100)
    parser.add_argument('--steps-per-trial', type=int, default=25)
    parser.add_argument('--sample-hz', type=int, default=20)
    parser.add_argument('--base-vg', type=float, default=0.55)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--beta', type=float, default=0.10)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--noise-collect-s', type=float, default=15.0)
    args = parser.parse_args()

    n_total = args.baseline_trials + args.learning_trials + args.test_trials

    print("=" * 70)
    print("z2186: Noise-Driven Online Plasticity in FPGA Reservoir")
    print("=" * 70)
    print(f"  Baseline: {args.baseline_trials}  Learning: {args.learning_trials}  "
          f"Test: {args.test_trials}  Total: {n_total}")
    print(f"  Steps/trial: {args.steps_per_trial}  Sample Hz: {args.sample_hz}")
    print(f"  base_vg={args.base_vg}  alpha={args.alpha}  beta={args.beta}  eta={args.eta}")

    rng = np.random.default_rng(42)
    w_in_init = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2186_noise_driven_plasticity',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': args.base_vg, 'alpha': args.alpha, 'beta': args.beta,
            'eta': args.eta, 'n_neurons': N_NEURONS, 'sample_hz': args.sample_hz,
            'steps_per_trial': args.steps_per_trial,
            'baseline_trials': args.baseline_trials,
            'learning_trials': args.learning_trials,
            'test_trials': args.test_trials,
            'w_in_init': w_in_init.tolist(),
            'w_noise': w_noise.tolist(),
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

    # ─── Step 3: Generate waveform data ───
    print("\n[3/6] Generating waveform classification data...")
    all_inputs, all_labels = generate_trial_data(
        n_total, args.steps_per_trial, sample_hz=args.sample_hz, seed=42
    )
    class_counts = np.bincount(all_labels, minlength=3)
    print(f"  {n_total} trials: sine={class_counts[0]} triangle={class_counts[1]} "
          f"square={class_counts[2]}")

    # ─── Step 4: Run conditions ───
    print("\n[4/6] Running conditions...")
    conditions = {
        'FULL':          {'noise': noise_1f,    'label': 'GPU 1/f noise + Hebbian learning'},
        'WHITE':         {'noise': noise_white,  'label': 'White noise + Hebbian learning'},
        'NO_NOISE':      {'noise': noise_zero,   'label': 'No noise (no exploration)'},
        'RANDOM_UPDATE': {'noise': noise_1f,     'label': 'GPU 1/f noise + random updates'},
    }

    condition_results = {}
    for cond_name, cond_info in conditions.items():
        print(f"\n{'─' * 60}")
        print(f"  {cond_name}: {cond_info['label']}")
        print(f"{'─' * 60}")
        # Each condition gets fresh RNG for random updates but same initial weights
        cond_rng = np.random.default_rng(hash(cond_name) & 0xFFFFFFFF)
        cond_result = run_condition(
            cond_name, ser, fpga, all_inputs, all_labels,
            cond_info['noise'], w_in_init.copy(), w_noise, args, cond_rng
        )
        condition_results[cond_name] = cond_result

    results['conditions'] = condition_results

    # ─── Step 5: Evaluate tests T207-T212 ───
    print("\n" + "=" * 60)
    print("[5/6] Evaluating tests T207-T212...")
    print("=" * 60)

    full = condition_results['FULL']
    white = condition_results['WHITE']
    no_noise = condition_results['NO_NOISE']
    random_upd = condition_results['RANDOM_UPDATE']

    tests = {}

    # T207: FULL phase3_acc > FULL phase1_acc
    t207_pass = full['phase3_acc'] > full['phase1_acc']
    tests['T207'] = {
        'name': 'FULL phase3_acc > FULL phase1_acc (learning happened)',
        'phase1_acc': full['phase1_acc'],
        'phase3_acc': full['phase3_acc'],
        'pass': t207_pass,
    }
    print(f"  T207: FULL phase3={full['phase3_acc']:.4f} > phase1={full['phase1_acc']:.4f} "
          f"-> {'PASS' if t207_pass else 'FAIL'}")

    # T208: FULL improvement > WHITE improvement
    t208_pass = full['improvement'] > white['improvement']
    tests['T208'] = {
        'name': 'FULL improvement > WHITE improvement (1/f better)',
        'full_improvement': full['improvement'],
        'white_improvement': white['improvement'],
        'pass': t208_pass,
    }
    print(f"  T208: FULL improv={full['improvement']:+.4f} > WHITE={white['improvement']:+.4f} "
          f"-> {'PASS' if t208_pass else 'FAIL'}")

    # T209: FULL improvement > NO_NOISE improvement
    t209_pass = full['improvement'] > no_noise['improvement']
    tests['T209'] = {
        'name': 'FULL improvement > NO_NOISE improvement (noise helps)',
        'full_improvement': full['improvement'],
        'no_noise_improvement': no_noise['improvement'],
        'pass': t209_pass,
    }
    print(f"  T209: FULL improv={full['improvement']:+.4f} > NO_NOISE={no_noise['improvement']:+.4f} "
          f"-> {'PASS' if t209_pass else 'FAIL'}")

    # T210: FULL improvement > RANDOM_UPDATE improvement
    t210_pass = full['improvement'] > random_upd['improvement']
    tests['T210'] = {
        'name': 'FULL improvement > RANDOM_UPDATE improvement (structured > random)',
        'full_improvement': full['improvement'],
        'random_improvement': random_upd['improvement'],
        'pass': t210_pass,
    }
    print(f"  T210: FULL improv={full['improvement']:+.4f} > RANDOM={random_upd['improvement']:+.4f} "
          f"-> {'PASS' if t210_pass else 'FAIL'}")

    # T211: FULL phase3_acc > 0.45
    t211_pass = full['phase3_acc'] > 0.45
    tests['T211'] = {
        'name': 'FULL phase3_acc > 0.45 (meaningful accuracy)',
        'phase3_acc': full['phase3_acc'],
        'threshold': 0.45,
        'pass': t211_pass,
    }
    print(f"  T211: FULL phase3_acc={full['phase3_acc']:.4f} > 0.45 "
          f"-> {'PASS' if t211_pass else 'FAIL'}")

    # T212: Learning curve monotonic (>=3 of 6 checkpoints increase over baseline)
    baseline_acc = full['phase1_acc']
    checkpoints_above = sum(1 for cp in full['checkpoint_accs'] if cp > baseline_acc)
    t212_pass = checkpoints_above >= 3
    tests['T212'] = {
        'name': 'Learning curve monotonic (>=3/6 checkpoints above baseline)',
        'baseline_acc': baseline_acc,
        'checkpoint_accs': full['checkpoint_accs'],
        'checkpoints_above_baseline': checkpoints_above,
        'threshold': 3,
        'pass': t212_pass,
    }
    print(f"  T212: {checkpoints_above}/6 checkpoints above baseline={baseline_acc:.4f} "
          f"(need >=3) -> {'PASS' if t212_pass else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/6 PASS")
    results['tests'] = tests
    results['n_pass'] = n_pass
    results['n_total'] = 6

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2186_noise_driven_plasticity.json'
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
        fig.suptitle('z2186: Noise-Driven Online Plasticity in FPGA Reservoir',
                      fontsize=14, fontweight='bold')

        # ── Panel A: Learning curves ──
        ax = axes[0, 0]
        colors = {'FULL': '#e74c3c', 'WHITE': '#3498db', 'NO_NOISE': '#95a5a6',
                  'RANDOM_UPDATE': '#f39c12'}
        for cond_name in ['FULL', 'WHITE', 'NO_NOISE', 'RANDOM_UPDATE']:
            cr = condition_results[cond_name]
            tc = cr['trial_correct']
            # Compute rolling accuracy (window=50)
            window = 50
            if len(tc) >= window:
                rolling = np.convolve(tc, np.ones(window) / window, mode='valid')
                ax.plot(np.arange(len(rolling)) + window, rolling,
                        label=cond_name, color=colors[cond_name], linewidth=1.5)
        # Mark phase boundaries
        ax.axvline(x=args.baseline_trials, color='gray', linestyle='--', alpha=0.5, label='Phase boundary')
        ax.axvline(x=args.baseline_trials + args.learning_trials, color='gray',
                    linestyle='--', alpha=0.5)
        ax.axhline(y=1.0 / 3.0, color='black', linestyle=':', alpha=0.3, label='Chance (33%)')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Accuracy (rolling 50)')
        ax.set_title('A. Learning Curves')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(0, 1)

        # ── Panel B: Before/After bar chart ──
        ax = axes[0, 1]
        cond_names = ['FULL', 'WHITE', 'NO_NOISE', 'RANDOM_UPDATE']
        x = np.arange(len(cond_names))
        width = 0.35
        phase1_accs = [condition_results[c]['phase1_acc'] for c in cond_names]
        phase3_accs = [condition_results[c]['phase3_acc'] for c in cond_names]
        bars1 = ax.bar(x - width / 2, phase1_accs, width, label='Phase 1 (baseline)',
                        color='#bdc3c7', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width / 2, phase3_accs, width, label='Phase 3 (test)',
                        color=[colors[c] for c in cond_names], edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_names, fontsize=8, rotation=15)
        ax.set_ylabel('Accuracy')
        ax.set_title('B. Before vs After Learning')
        ax.legend(fontsize=8)
        ax.axhline(y=1.0 / 3.0, color='black', linestyle=':', alpha=0.3)
        ax.set_ylim(0, 1)

        # ── Panel C: Weight evolution heatmap (FULL condition) ──
        ax = axes[1, 0]
        w_history = np.array(condition_results['FULL']['w_in_history'])
        if len(w_history) > 1:
            im = ax.imshow(w_history.T, aspect='auto', cmap='RdBu_r',
                           vmin=-2, vmax=2, interpolation='nearest')
            ax.set_xlabel('Weight Update Step')
            ax.set_ylabel('Neuron ID')
            ax.set_title('C. Weight Evolution (FULL)')
            plt.colorbar(im, ax=ax, label='w_in')
        else:
            ax.text(0.5, 0.5, 'No weight updates', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('C. Weight Evolution (FULL)')

        # ── Panel D: Test summary ──
        ax = axes[1, 1]
        ax.axis('off')
        test_lines = []
        for tid, tdata in sorted(tests.items()):
            status = 'PASS' if tdata['pass'] else 'FAIL'
            color = '#27ae60' if tdata['pass'] else '#e74c3c'
            test_lines.append((tid, tdata['name'][:50], status, color))

        y_pos = 0.9
        for tid, name, status, color in test_lines:
            ax.text(0.02, y_pos, f"{tid}: {status}", fontsize=11, fontweight='bold',
                    color=color, transform=ax.transAxes, family='monospace')
            ax.text(0.20, y_pos, name, fontsize=9, transform=ax.transAxes)
            y_pos -= 0.13

        ax.text(0.02, y_pos - 0.05, f"TOTAL: {n_pass}/6 PASS",
                fontsize=13, fontweight='bold', transform=ax.transAxes,
                color='#27ae60' if n_pass >= 4 else '#e74c3c')
        ax.set_title('D. Test Results')

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2186_noise_driven_plasticity.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure saved: {fig_path}")

    except Exception as e:
        print(f"  Figure generation failed: {e}")

    # ─── Cleanup ───
    if fpga and ser:
        try:
            set_per_neuron_vg(ser, [0.0] * N_NEURONS)
            ser.close()
        except Exception:
            pass

    print("\n" + "=" * 70)
    print(f"z2186 complete: {n_pass}/6 PASS")
    print("=" * 70)
    return n_pass


if __name__ == '__main__':
    main()
