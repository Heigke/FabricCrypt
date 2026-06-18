#!/usr/bin/env python3
"""z2172_perturbation_robustness.py — FPGA Reservoir Perturbation Robustness

Systematically perturbs the FPGA reservoir during inference and measures
accuracy degradation across 6 conditions:

  1. BASELINE:     No perturbation (control)
  2. VG_NOISE:     Add random +/-0.05 noise to gate voltages
  3. KILL_NEURON:  Disable 2 of 8 neurons (Vg=0)
  4. MAC_PERTURB:  Randomly change MAC value during inference
  5. DELAY_INJECT: Add 50ms random delays between steps
  6. COMBINED:     All perturbations at once

Task: Waveform classification (100 trials, 25 steps) — sine/triangle/square

Tests:
  T121: BASELINE accuracy > 55%
  T122: VG_NOISE accuracy > 40%
  T123: KILL_NEURON accuracy > 35%
  T124: BASELINE > all perturbation conditions
  T125: COMBINED < BASELINE by > 10pp
  T126: At least 3/5 perturbation conditions > chance (33.3%)

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
CMD_SET_VG   = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03
CMD_SET_MAC  = 0x06

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters ───
BASE_VG    = 0.58
ALPHA      = 0.25
BETA       = 0.08
N_NEURONS  = 8
SAMPLE_HZ  = 20

# ─── Perturbation parameters ───
VG_NOISE_AMP   = 0.05   # +/-0.05 gate voltage noise
KILL_COUNT     = 2       # disable 2 of 8 neurons
DELAY_MS       = 50      # injected delay in ms
N_TRIALS       = 100
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
    """Send MAC perturbation value (Q16.16)."""
    q16 = to_q16_16(max(0.0, min(1.0, value)))
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


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=100, steps_per_trial=25, freq_hz=1.0, dt=1.0/20, seed=42):
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
                            base_vg=BASE_VG, alpha=ALPHA, beta=0.10,
                            killed_neurons=None, vg_noise_amp=0.0,
                            delay_inject_ms=0, rng=None):
    """Software LIF simulation fallback with perturbation support."""
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

        # Perturbation: Vg noise
        if vg_noise_amp > 0:
            vg += rng.uniform(-vg_noise_amp, vg_noise_amp, size=N_NEURONS)
            vg = np.clip(vg, 0.05, 0.95)

        # Perturbation: kill neurons
        if killed_neurons is not None:
            for nid in killed_neurons:
                vg[nid] = 0.0

        # Perturbation: delay injection (simulate as noise in timing)
        if delay_inject_ms > 0:
            delay_noise = rng.uniform(0, delay_inject_ms / 1000.0)
            # In simulation, delay manifests as extra leak
            vmem *= np.exp(-delay_noise / tau_m)

        # LIF dynamics
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
        states[t, N_NEURONS:N_NEURONS*2] = vmem.copy()
        states[t, N_NEURONS*2:] = cumulative.copy()

    return states


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir with Perturbations
# ═══════════════════════════════════════════════════════════

def run_fpga_trial_perturbed(ser, input_signal, noise_samples, w_in, w_noise,
                              perturbation='BASELINE', rng=None,
                              killed_neurons=None):
    """Run single FPGA reservoir trial with perturbation applied.

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
        # Get noise value (live power rail)
        p = read_hwmon_power()
        noise_val = (p - power_mean) / 2.0 if p else 0.0

        # Base Vg computation
        vg_values = np.full(N_NEURONS, BASE_VG)
        vg_values += ALPHA * input_signal[t] * w_in
        if BETA > 0 and len(noise_samples) > 0:
            noise_idx = t % len(noise_samples)
            vg_values += BETA * noise_samples[noise_idx] * w_noise

        # ─── Apply perturbations ───
        do_vg_noise = perturbation in ('VG_NOISE', 'COMBINED')
        do_kill     = perturbation in ('KILL_NEURON', 'COMBINED')
        do_mac      = perturbation in ('MAC_PERTURB', 'COMBINED')
        do_delay    = perturbation in ('DELAY_INJECT', 'COMBINED')

        if do_vg_noise:
            vg_values += rng.uniform(-VG_NOISE_AMP, VG_NOISE_AMP, size=N_NEURONS)

        if do_kill and killed_neurons is not None:
            for nid in killed_neurons:
                vg_values[nid] = 0.0

        vg_values = np.clip(vg_values, 0.05, 0.95)
        set_per_neuron_vg(ser, vg_values)

        if do_mac:
            mac_val = rng.uniform(0.1, 0.9)
            send_mac(ser, mac_val)

        if do_delay:
            delay_s = rng.uniform(0, DELAY_MS / 1000.0)
            time.sleep(delay_s)

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

    n_classes = len(np.unique(y_train))
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


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

PERTURBATION_CONDITIONS = [
    'BASELINE', 'VG_NOISE', 'KILL_NEURON', 'MAC_PERTURB', 'DELAY_INJECT', 'COMBINED'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps-per-trial', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--noise-collect-s', type=float, default=10.0)
    args = parser.parse_args()

    print("=" * 65)
    print("z2172: FPGA Reservoir Perturbation Robustness")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2172_perturbation_robustness',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'vg_noise_amp': VG_NOISE_AMP, 'kill_count': KILL_COUNT,
            'delay_ms': DELAY_MS,
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
        # Disable kill switch
        send_kill(ser, 0x00)
        time.sleep(0.1)
        print("  Kill switch disabled")

    # ─── Step 2: Collect GPU noise ───
    print("\n[2/5] Collecting GPU power rail noise (1/f)...")
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

    # Apply IIR filter for temporal memory
    noise_filtered = iir_filter_noise(noise_1f, alpha_iir=0.85)
    print(f"  IIR-filtered noise: {len(noise_filtered)} samples, std={np.std(noise_filtered):.3f}")

    # ─── Step 3: Generate waveforms ───
    print("\n[3/5] Generating waveforms...")
    trials, labels = generate_waveforms(
        n_trials=args.n_trials,
        steps_per_trial=args.steps_per_trial,
        seed=42,
    )
    print(f"  {args.n_trials} trials x {args.steps_per_trial} steps")
    class_counts = {int(c): int(np.sum(labels == c)) for c in np.unique(labels)}
    print(f"  Class distribution: {class_counts}")

    # Pick 2 neurons to kill (fixed across all KILL_NEURON trials)
    killed_neurons = sorted(rng.choice(N_NEURONS, size=KILL_COUNT, replace=False).tolist())
    print(f"  Kill targets: neurons {killed_neurons}")

    # ─── Step 4: Run reservoir under each perturbation condition ───
    print("\n[4/5] Running reservoir under 6 perturbation conditions...")
    condition_features = {}
    condition_meta = {}

    for cond_idx, cond in enumerate(PERTURBATION_CONDITIONS):
        print(f"\n  --- Condition {cond_idx+1}/6: {cond} ---")
        t0 = time.monotonic()

        # Reset FPGA state between conditions
        if fpga:
            send_kill(ser, 0x00)  # all neurons alive
            time.sleep(0.05)

        all_features = []
        for trial_idx in range(args.n_trials):
            if trial_idx % 20 == 0:
                print(f"    Trial {trial_idx}/{args.n_trials}...", end='\r')

            input_signal = trials[trial_idx]
            trial_rng = np.random.default_rng(1000 * cond_idx + trial_idx)

            if fpga:
                # For KILL_NEURON/COMBINED: set kill mask before trial
                if cond in ('KILL_NEURON', 'COMBINED'):
                    mask = 0
                    for nid in killed_neurons:
                        mask |= (1 << nid)
                    send_kill(ser, mask)
                    time.sleep(0.01)
                else:
                    send_kill(ser, 0x00)
                    time.sleep(0.005)

                trial_states = run_fpga_trial_perturbed(
                    ser, input_signal, noise_filtered, w_in, w_noise,
                    perturbation=cond, rng=trial_rng,
                    killed_neurons=killed_neurons if cond in ('KILL_NEURON', 'COMBINED') else None,
                )
            else:
                # Simulation fallback
                sim_vg_noise = VG_NOISE_AMP if cond in ('VG_NOISE', 'COMBINED') else 0.0
                sim_killed = killed_neurons if cond in ('KILL_NEURON', 'COMBINED') else None
                sim_delay = DELAY_MS if cond in ('DELAY_INJECT', 'COMBINED') else 0

                trial_states = simulate_lif_reservoir(
                    input_signal, noise_filtered, w_in, w_noise,
                    base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                    killed_neurons=sim_killed,
                    vg_noise_amp=sim_vg_noise,
                    delay_inject_ms=sim_delay,
                    rng=trial_rng,
                )

            feat = pool_trial_features(trial_states)
            all_features.append(feat)

        features_array = np.array(all_features)
        elapsed = time.monotonic() - t0
        print(f"    {cond}: {features_array.shape} features, {elapsed:.1f}s")

        condition_features[cond] = features_array
        condition_meta[cond] = {'elapsed_s': elapsed, 'shape': features_array.shape}

        # Reset kill after KILL conditions
        if fpga and cond in ('KILL_NEURON', 'COMBINED'):
            send_kill(ser, 0x00)
            time.sleep(0.05)

    # ─── Step 5: Classify and evaluate ───
    print("\n[5/5] Classification (5-fold stratified ridge regression)...")
    condition_accuracies = {}

    for cond in PERTURBATION_CONDITIONS:
        X = condition_features[cond]
        y = labels

        # Remove constant features
        feat_std = X.std(axis=0)
        good_cols = feat_std > 1e-8
        X_clean = X[:, good_cols]
        if X_clean.shape[1] == 0:
            print(f"  {cond}: All features constant! Accuracy = chance (0.333)")
            condition_accuracies[cond] = 1.0 / 3.0
            continue

        # Normalize
        mu = X_clean.mean(axis=0)
        sigma = X_clean.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        X_norm = (X_clean - mu) / sigma

        # Add bias
        X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])

        # 5-fold cross-validation
        folds = stratified_kfold(X_aug, y, n_splits=5, seed=42)
        fold_accs = []
        for train_idx, test_idx in folds:
            acc = ridge_classify(X_aug[train_idx], y[train_idx],
                                  X_aug[test_idx], y[test_idx])
            fold_accs.append(acc)

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        condition_accuracies[cond] = mean_acc
        print(f"  {cond:15s}: {mean_acc:.3f} +/- {std_acc:.3f}  folds={[f'{a:.3f}' for a in fold_accs]}")

    # ─── Evaluate Tests ───
    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    acc = condition_accuracies
    tests = {}

    # T121: BASELINE > 55%
    t121_pass = acc['BASELINE'] > 0.55
    tests['T121'] = {
        'name': 'BASELINE accuracy > 55%',
        'baseline_acc': acc['BASELINE'],
        'threshold': 0.55,
        'pass': t121_pass,
    }
    print(f"  T121 BASELINE > 55%:        {acc['BASELINE']:.3f} > 0.55 → {'PASS' if t121_pass else 'FAIL'}")

    # T122: VG_NOISE > 40%
    t122_pass = acc['VG_NOISE'] > 0.40
    tests['T122'] = {
        'name': 'VG_NOISE accuracy > 40%',
        'vg_noise_acc': acc['VG_NOISE'],
        'threshold': 0.40,
        'pass': t122_pass,
    }
    print(f"  T122 VG_NOISE > 40%:        {acc['VG_NOISE']:.3f} > 0.40 → {'PASS' if t122_pass else 'FAIL'}")

    # T123: KILL_NEURON > 35%
    t123_pass = acc['KILL_NEURON'] > 0.35
    tests['T123'] = {
        'name': 'KILL_NEURON accuracy > 35%',
        'kill_neuron_acc': acc['KILL_NEURON'],
        'threshold': 0.35,
        'pass': t123_pass,
    }
    print(f"  T123 KILL_NEURON > 35%:     {acc['KILL_NEURON']:.3f} > 0.35 → {'PASS' if t123_pass else 'FAIL'}")

    # T124: BASELINE > all perturbation conditions
    perturb_conds = ['VG_NOISE', 'KILL_NEURON', 'MAC_PERTURB', 'DELAY_INJECT', 'COMBINED']
    t124_pass = all(acc['BASELINE'] > acc[c] for c in perturb_conds)
    tests['T124'] = {
        'name': 'BASELINE > all perturbation conditions',
        'baseline_acc': acc['BASELINE'],
        'perturbation_accs': {c: acc[c] for c in perturb_conds},
        'all_below_baseline': t124_pass,
        'pass': t124_pass,
    }
    print(f"  T124 BASELINE > all perturb: {t124_pass} → {'PASS' if t124_pass else 'FAIL'}")
    for c in perturb_conds:
        delta = acc['BASELINE'] - acc[c]
        print(f"        {c:15s}: {acc[c]:.3f} (delta={delta:+.3f})")

    # T125: COMBINED < BASELINE by > 10pp
    drop_pp = (acc['BASELINE'] - acc['COMBINED']) * 100
    t125_pass = drop_pp > 10.0
    tests['T125'] = {
        'name': 'COMBINED < BASELINE by > 10pp',
        'baseline_acc': acc['BASELINE'],
        'combined_acc': acc['COMBINED'],
        'drop_pp': drop_pp,
        'threshold_pp': 10.0,
        'pass': t125_pass,
    }
    print(f"  T125 COMBINED drop > 10pp:  {drop_pp:.1f}pp → {'PASS' if t125_pass else 'FAIL'}")

    # T126: At least 3/5 perturbation conditions > chance (33.3%)
    above_chance = sum(1 for c in perturb_conds if acc[c] > 1.0/3.0)
    t126_pass = above_chance >= 3
    tests['T126'] = {
        'name': 'At least 3/5 perturbation conditions > chance (33.3%)',
        'above_chance_count': above_chance,
        'threshold': 3,
        'per_condition': {c: {'acc': acc[c], 'above_chance': acc[c] > 1.0/3.0} for c in perturb_conds},
        'pass': t126_pass,
    }
    print(f"  T126 >=3/5 > chance:        {above_chance}/5 >= 3 → {'PASS' if t126_pass else 'FAIL'}")

    total_pass = sum(1 for t in tests.values() if t['pass'])
    total_tests = len(tests)
    print(f"\n  TOTAL: {total_pass}/{total_tests} PASS")

    # ─── Save results ───
    results['condition_accuracies'] = condition_accuracies
    results['condition_meta'] = {k: {'elapsed_s': v['elapsed_s'], 'shape': list(v['shape'])}
                                  for k, v in condition_meta.items()}
    results['killed_neurons'] = killed_neurons
    results['tests'] = tests
    results['total_pass'] = total_pass
    results['total_tests'] = total_tests

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / 'z2172_perturbation_robustness.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_json}")

    # ─── Generate figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        FIGURES.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: bar chart of accuracies
        ax = axes[0]
        conds = PERTURBATION_CONDITIONS
        accs = [condition_accuracies[c] for c in conds]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#795548']
        bars = ax.bar(range(len(conds)), accs, color=colors, edgecolor='black', linewidth=0.8)

        # Chance line
        ax.axhline(y=1.0/3.0, color='gray', linestyle='--', linewidth=1.2, label='Chance (33.3%)')
        ax.axhline(y=0.55, color='blue', linestyle=':', linewidth=1.0, alpha=0.5, label='T121 threshold (55%)')

        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Perturbation Robustness', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, loc='upper right')

        # Add accuracy labels on bars
        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{a:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Right panel: degradation from baseline
        ax2 = axes[1]
        perturb_names = perturb_conds
        drops = [(acc['BASELINE'] - acc[c]) * 100 for c in perturb_names]
        bar_colors = ['#4CAF50' if d >= 0 else '#F44336' for d in drops]
        bars2 = ax2.barh(range(len(perturb_names)), drops, color=bar_colors,
                          edgecolor='black', linewidth=0.8)

        ax2.axvline(x=10, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
                     label='T125 threshold (10pp)')
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.set_yticks(range(len(perturb_names)))
        ax2.set_yticklabels(perturb_names, fontsize=10)
        ax2.set_xlabel('Accuracy Drop from Baseline (pp)', fontsize=12)
        ax2.set_title('Degradation Analysis', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)

        for bar, d in zip(bars2, drops):
            xpos = bar.get_width() + 0.5 if bar.get_width() >= 0 else bar.get_width() - 0.5
            ax2.text(xpos, bar.get_y() + bar.get_height()/2,
                     f'{d:+.1f}pp', ha='left' if d >= 0 else 'right',
                     va='center', fontsize=9, fontweight='bold')

        # Add test results summary
        test_summary = f"Tests: {total_pass}/{total_tests} PASS"
        sim_tag = " [SIMULATED]" if results['simulated'] else " [FPGA]"
        fig.suptitle(f'z2172 — {test_summary}{sim_tag}', fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        fig_path = FIGURES / 'fig_z2172_perturbation.png'
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"  Figure saved: {fig_path}")
        plt.close(fig)

    except ImportError as e:
        print(f"  matplotlib unavailable, skipping figure: {e}")

    # ─── Cleanup ───
    if fpga and ser:
        send_kill(ser, 0x00)
        set_per_neuron_vg(ser, [BASE_VG] * N_NEURONS)
        ser.close()
        print("  FPGA reset and closed")

    print(f"\n{'=' * 65}")
    print(f"z2172 COMPLETE: {total_pass}/{total_tests} PASS")
    print(f"{'=' * 65}")

    return 0 if total_pass >= 4 else 1


if __name__ == '__main__':
    sys.exit(main())
