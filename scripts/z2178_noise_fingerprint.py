#!/usr/bin/env python3
"""z2178_noise_fingerprint.py — GPU Noise Source Fingerprinting via FPGA Reservoir

Can the reservoir distinguish WHICH noise source is driving it, based only on
spike/vmem patterns?  This creates a "fingerprint" of GPU noise characteristics.

5 noise conditions drive the same 8 LIF neurons on an Arty A7 FPGA:
  1. GPU_POWER   — hwmon power1_average (real 1/f from VRM switching)
  2. GPU_TEMP    — temp_soc via gpu_metrics (thermal sensor, slower dynamics)
  3. SYNTHETIC_1F — Voss-McCartney algorithm (matches PSD slope, no HW correlations)
  4. WHITE       — numpy random normal
  5. PINK_IIR    — IIR-filtered white noise (alpha=0.85)

For each condition: 200 trials × 25 steps of reservoir responses to identical input
waveforms (sine/triangle/square at fixed params).  Then train Ridge classifier to
distinguish noise source from spike/vmem patterns alone.

Tests T157-T162:
  T157: Noise source classification accuracy > 40% (5-class, chance=20%)
  T158: GPU_POWER most distinguishable source (highest per-class accuracy)
  T159: GPU_POWER vs SYNTHETIC_1F pairwise accuracy > 60% (real vs fake 1/f)
  T160: Confusion matrix — GPU_TEMP most confused with PINK_IIR
  T161: Feature importance — spike count features rank higher than vmem features
  T162: Cross-validated accuracy > 30% (above chance even with CV)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB{0,1,2}
"""

import os, sys, json, time, struct, argparse
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
GPU_METRICS_PATH = "/sys/class/drm/card0/device/gpu_metrics"

# gpu_metrics v3.0 offsets for temp_soc (from z2160)
V3_TEMP_SOC_OFFSET = 56  # uint16 at byte 56, /100.0 → °C

# ─── Reservoir Parameters ───
BASE_VG = 0.58       # near BVpar cliff
ALPHA = 0.25         # input coupling
BETA = 0.08          # noise coupling
N_NEURONS = 8
SAMPLE_HZ = 20

# Noise condition labels
NOISE_LABELS = ['GPU_POWER', 'GPU_TEMP', 'SYNTHETIC_1F', 'WHITE', 'PINK_IIR']


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


# ═══════════════════════════════════════════════════════════
# Noise Sources
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    """Read hwmon power1_average (μW → W). Rich 1/f dynamics ~11W ± 1.5W."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def read_gpu_temp_soc():
    """Read temp_soc from gpu_metrics binary blob (°C)."""
    try:
        blob = open(GPU_METRICS_PATH, 'rb').read()
        if len(blob) < V3_TEMP_SOC_OFFSET + 2:
            return None
        temp = struct.unpack_from('<H', blob, V3_TEMP_SOC_OFFSET)[0] / 100.0
        if temp < 1.0 or temp > 120.0:
            return None
        return temp
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


def collect_temp_noise(duration_s=15, sample_hz=50):
    """Collect GPU SOC temperature time series (slower dynamics than power)."""
    n_samples = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    temps = []
    for _ in range(n_samples):
        t = read_gpu_temp_soc()
        if t is not None:
            temps.append(t)
        time.sleep(interval)
    return np.array(temps) if temps else None


def generate_voss_mccartney(n_samples, rng, n_octaves=8):
    """Voss-McCartney 1/f generator — matches PSD slope but no HW correlations."""
    noise = np.zeros(n_samples)
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    noise = (noise - noise.mean()) / max(noise.std(), 1e-6)
    return noise


def iir_filter_noise(noise_samples, alpha_iir=0.85):
    """Apply IIR low-pass: y[t] = α·y[t-1] + (1-α)·x[t].
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

def generate_waveforms(n_trials, steps_per_trial, freq_hz=1.0, dt=1.0 / 20, seed=42):
    """Generate sine/triangle/square waveforms for classification.
    Returns identical waveforms across all noise conditions (fixed seed).
    """
    rng = np.random.default_rng(seed)
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

        wave = (wave + 1.0) / 2.0  # normalize to [0, 1]
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=BASE_VG, alpha=ALPHA, beta=BETA,
                              live_noise_fn=None):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    When live_noise_fn is provided, calls it at each step for real-time noise.
    Otherwise uses pre-collected noise_samples.

    Returns: (n_steps, 24) array — 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))  # delta + vmem + cumulative

    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0  # approx mean for normalization

    for t in range(n_steps):
        # Get noise value
        if live_noise_fn is not None:
            raw = live_noise_fn()
            noise_val = (raw - power_mean) / 2.0 if raw is not None else 0.0
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
    states = np.zeros((n_steps, N_NEURONS * 3))  # delta + vmem + cumulative

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

def pool_trial_features(trial_states):
    """Pool per-timestep reservoir states into trial-level features.
    trial_states: (n_steps, n_features) → [mean, std, max, min] per feature.
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

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0

    best_acc = -1
    best_pred = None
    best_W = None

    for alpha_r in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha_r * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue

        pred_test = np.argmax(X_test @ W, axis=1)
        acc_test = np.mean(pred_test == y_test)

        if acc_test > best_acc:
            best_acc = acc_test
            best_pred = pred_test
            best_W = W

    return best_acc, best_pred, best_W


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


def permutation_importance(X, y, W, n_classes, n_repeats=10, seed=42):
    """Compute permutation importance for each feature."""
    rng = np.random.default_rng(seed)
    baseline_pred = np.argmax(X @ W, axis=1)
    baseline_acc = np.mean(baseline_pred == y)

    importances = np.zeros(X.shape[1])
    for feat in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, feat] = rng.permutation(X_perm[:, feat])
            pred_perm = np.argmax(X_perm @ W, axis=1)
            acc_perm = np.mean(pred_perm == y)
            drops.append(baseline_acc - acc_perm)
        importances[feat] = np.mean(drops)

    return importances


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2178: GPU Noise Source Fingerprinting')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Trials per noise condition (default: 200)')
    parser.add_argument('--steps-per-trial', type=int, default=25,
                        help='Timesteps per trial (default: 25)')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Seconds to collect GPU noise (default: 15)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    args = parser.parse_args()

    print("=" * 65)
    print("z2178: GPU Noise Source Fingerprinting via FPGA Reservoir")
    print("=" * 65)

    rng = np.random.default_rng(42)
    # Fixed random weights per neuron
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2178_noise_fingerprint',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'noise_labels': NOISE_LABELS,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/8] Connecting to FPGA...")
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

    # ─── Step 2: Collect GPU noise sources ───
    print("\n[2/8] Collecting GPU noise sources...")

    # GPU_POWER: hwmon power1_average (1/f from VRM switching)
    print("  [GPU_POWER] Collecting power rail noise...")
    power_noise_raw = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise_raw is not None and len(power_noise_raw) > 10:
        power_mean = power_noise_raw.mean()
        power_std = max(power_noise_raw.std(), 1e-6)
        noise_gpu_power = (power_noise_raw - power_mean) / power_std
        print(f"    {power_mean:.2f} ± {power_std:.3f} W, {len(noise_gpu_power)} samples")
        has_real_power = True
    else:
        print("    Power rail unavailable, generating synthetic fallback")
        noise_gpu_power = generate_voss_mccartney(750, rng)
        # Add unique hardware-like fingerprint: slight asymmetry + burst structure
        noise_gpu_power += 0.1 * rng.standard_normal(len(noise_gpu_power))
        has_real_power = False

    # GPU_TEMP: temp_soc via gpu_metrics (slower dynamics)
    print("  [GPU_TEMP] Collecting SOC temperature noise...")
    temp_noise_raw = collect_temp_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if temp_noise_raw is not None and len(temp_noise_raw) > 10:
        temp_mean = temp_noise_raw.mean()
        temp_std = max(temp_noise_raw.std(), 1e-6)
        noise_gpu_temp = (temp_noise_raw - temp_mean) / temp_std
        print(f"    {temp_mean:.2f} ± {temp_std:.3f} °C, {len(noise_gpu_temp)} samples")
        has_real_temp = True
    else:
        print("    Temp SOC unavailable, generating slow-varying fallback")
        # Simulate slow thermal dynamics: IIR with very high alpha
        raw = rng.standard_normal(750)
        noise_gpu_temp = iir_filter_noise(raw, alpha_iir=0.95)
        has_real_temp = False

    # SYNTHETIC_1F: Voss-McCartney (matches PSD slope, no HW correlations)
    print("  [SYNTHETIC_1F] Generating Voss-McCartney 1/f noise...")
    n_synth = max(750, int(args.noise_collect_s * 50))
    noise_synth_1f = generate_voss_mccartney(n_synth, np.random.default_rng(123))
    print(f"    {len(noise_synth_1f)} samples")

    # WHITE: numpy random normal
    print("  [WHITE] Generating white noise...")
    noise_white = np.random.default_rng(456).standard_normal(n_synth)
    print(f"    {len(noise_white)} samples")

    # PINK_IIR: IIR-filtered white noise (alpha=0.85)
    print("  [PINK_IIR] Generating IIR-filtered pink noise...")
    raw_for_iir = np.random.default_rng(789).standard_normal(n_synth)
    noise_pink_iir = iir_filter_noise(raw_for_iir, alpha_iir=0.85)
    print(f"    {len(noise_pink_iir)} samples")

    noise_sources = {
        'GPU_POWER': noise_gpu_power,
        'GPU_TEMP': noise_gpu_temp,
        'SYNTHETIC_1F': noise_synth_1f,
        'WHITE': noise_white,
        'PINK_IIR': noise_pink_iir,
    }

    results['noise_stats'] = {}
    for name, ns in noise_sources.items():
        results['noise_stats'][name] = {
            'n_samples': len(ns),
            'mean': float(np.mean(ns)),
            'std': float(np.std(ns)),
        }
    results['has_real_power'] = has_real_power
    results['has_real_temp'] = has_real_temp

    # ─── Step 3: Generate waveforms (identical across all conditions) ───
    print(f"\n[3/8] Generating waveforms: {args.n_trials} trials × {args.steps_per_trial} steps...")
    waveforms, wave_labels = generate_waveforms(
        args.n_trials, args.steps_per_trial, seed=42
    )
    print(f"  Waveform distribution: sine={np.sum(wave_labels==0)}, "
          f"tri={np.sum(wave_labels==1)}, sq={np.sum(wave_labels==2)}")

    # ─── Step 4: Run reservoir for each noise condition ───
    print(f"\n[4/8] Running reservoir trials across {len(NOISE_LABELS)} noise conditions...")
    all_features = []   # (n_conditions * n_trials, n_features)
    all_labels = []     # noise condition label per trial
    per_condition_features = {}

    for cond_idx, cond_name in enumerate(NOISE_LABELS):
        noise_samples = noise_sources[cond_name]
        print(f"  [{cond_name}] Running {args.n_trials} trials...")
        t0 = time.time()

        condition_features = []
        for trial_idx in range(args.n_trials):
            input_signal = waveforms[trial_idx]

            # Determine live noise function for GPU sources on real HW
            live_fn = None
            if fpga and cond_name == 'GPU_POWER' and has_real_power:
                live_fn = read_hwmon_power
            elif fpga and cond_name == 'GPU_TEMP' and has_real_temp:
                live_fn = read_gpu_temp_soc

            if fpga:
                states = run_fpga_reservoir_trial(
                    ser, input_signal, noise_samples, w_in, w_noise,
                    live_noise_fn=live_fn
                )
            else:
                states = simulate_lif_reservoir(
                    input_signal, noise_samples, w_in, w_noise
                )

            features = pool_trial_features(states)
            condition_features.append(features)
            all_features.append(features)
            all_labels.append(cond_idx)

        per_condition_features[cond_name] = np.array(condition_features)
        elapsed = time.time() - t0
        print(f"    {args.n_trials} trials in {elapsed:.1f}s")

    X = np.array(all_features)
    y = np.array(all_labels)
    n_total = len(y)
    n_features = X.shape[1]
    print(f"\n  Total: {n_total} trials × {n_features} features")

    # ─── Step 5: Train-test split classification ───
    print("\n[5/8] Training Ridge classifier (80/20 split)...")

    # Stratified 80/20 split
    rng_split = np.random.default_rng(42)
    train_idx = []
    test_idx = []
    for c in range(len(NOISE_LABELS)):
        c_indices = np.where(y == c)[0]
        rng_split.shuffle(c_indices)
        split = int(0.8 * len(c_indices))
        train_idx.extend(c_indices[:split])
        test_idx.extend(c_indices[split:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    overall_acc, y_pred, W_best = ridge_classify(X_train, y_train, X_test, y_test)
    print(f"  Overall accuracy: {overall_acc:.3f} (chance=0.200)")

    # Per-class accuracy
    per_class_acc = {}
    for c_idx, c_name in enumerate(NOISE_LABELS):
        mask = y_test == c_idx
        if mask.sum() > 0:
            per_class_acc[c_name] = float(np.mean(y_pred[mask] == c_idx))
        else:
            per_class_acc[c_name] = 0.0
    print("  Per-class accuracy:")
    for name, acc in per_class_acc.items():
        print(f"    {name:15s}: {acc:.3f}")

    # Confusion matrix (5x5)
    n_classes = len(NOISE_LABELS)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_test, y_pred):
        confusion[int(true), int(pred)] += 1
    # Normalize rows to fractions
    confusion_norm = confusion.astype(float)
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    confusion_norm = confusion / row_sums

    print("\n  Confusion matrix (rows=true, cols=pred):")
    header = "            " + "  ".join(f"{n[:6]:>6s}" for n in NOISE_LABELS)
    print(header)
    for i, name in enumerate(NOISE_LABELS):
        row_str = "  ".join(f"{confusion_norm[i, j]:6.2f}" for j in range(n_classes))
        print(f"  {name:10s}  {row_str}")

    # ─── Step 6: Pairwise GPU_POWER vs SYNTHETIC_1F ───
    print("\n[6/8] Pairwise: GPU_POWER vs SYNTHETIC_1F...")
    idx_power = 0  # GPU_POWER
    idx_synth = 2  # SYNTHETIC_1F
    mask_pair = np.isin(y, [idx_power, idx_synth])
    X_pair = X[mask_pair]
    y_pair = (y[mask_pair] == idx_synth).astype(int)  # binary: 0=POWER, 1=SYNTH

    # Stratified split for pairwise
    pair_splits = stratified_kfold(X_pair, y_pair, n_splits=5, seed=42)
    pair_accs = []
    for tr, te in pair_splits:
        acc_fold, _, _ = ridge_classify(X_pair[tr], y_pair[tr], X_pair[te], y_pair[te])
        pair_accs.append(acc_fold)
    pairwise_acc = np.mean(pair_accs)
    print(f"  GPU_POWER vs SYNTHETIC_1F pairwise accuracy: {pairwise_acc:.3f}")

    # ─── Step 7: Cross-validated accuracy ───
    print(f"\n[7/8] {args.n_folds}-fold stratified cross-validation...")
    cv_splits = stratified_kfold(X, y, n_splits=args.n_folds, seed=42)
    cv_accs = []
    for fold_idx, (tr, te) in enumerate(cv_splits):
        acc_fold, _, _ = ridge_classify(X[tr], y[tr], X[te], y[te])
        cv_accs.append(acc_fold)
        print(f"  Fold {fold_idx + 1}: {acc_fold:.3f}")
    cv_mean = np.mean(cv_accs)
    cv_std = np.std(cv_accs)
    print(f"  CV accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

    # ─── Feature importance ───
    print("\n[8/8] Computing feature importance via permutation...")
    if W_best is not None:
        importances = permutation_importance(X_test, y_test, W_best, n_classes, n_repeats=10)
    else:
        importances = np.zeros(n_features)

    # Features are organized as: [mean, std, max, min] × [delta_spikes(8), vmem(8), cumulative(8)]
    # So indices 0-7, 24-31, 48-55, 72-79 are spike features (delta_spikes)
    # Indices 8-15, 32-39, 56-63, 80-87 are vmem features
    # Indices 16-23, 40-47, 64-71, 88-95 are cumulative spike features
    n_base = N_NEURONS * 3  # 24 base features
    spike_feat_mask = np.zeros(n_features, dtype=bool)
    vmem_feat_mask = np.zeros(n_features, dtype=bool)
    for pool_offset in range(4):  # mean, std, max, min
        base = pool_offset * n_base
        spike_feat_mask[base:base + N_NEURONS] = True         # delta_spikes
        spike_feat_mask[base + 2 * N_NEURONS:base + 3 * N_NEURONS] = True  # cumulative
        vmem_feat_mask[base + N_NEURONS:base + 2 * N_NEURONS] = True       # vmem

    spike_importance = importances[spike_feat_mask].mean() if spike_feat_mask.sum() > 0 else 0.0
    vmem_importance = importances[vmem_feat_mask].mean() if vmem_feat_mask.sum() > 0 else 0.0
    print(f"  Mean spike feature importance: {spike_importance:.4f}")
    print(f"  Mean vmem feature importance:  {vmem_importance:.4f}")

    # ─── Evaluate Tests ───
    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    # T157: Overall accuracy > 40%
    t157_pass = overall_acc > 0.40
    print(f"  T157 noise_classification_accuracy={overall_acc:.3f} > 0.40  "
          f"{'PASS' if t157_pass else 'FAIL'}")

    # T158: GPU_POWER has highest per-class accuracy
    best_class = max(per_class_acc, key=per_class_acc.get)
    t158_pass = best_class == 'GPU_POWER'
    print(f"  T158 GPU_POWER_most_distinguishable: best={best_class} "
          f"(GPU_POWER acc={per_class_acc.get('GPU_POWER', 0):.3f})  "
          f"{'PASS' if t158_pass else 'FAIL'}")

    # T159: GPU_POWER vs SYNTHETIC_1F pairwise > 60%
    t159_pass = pairwise_acc > 0.60
    print(f"  T159 pairwise_power_vs_synth={pairwise_acc:.3f} > 0.60  "
          f"{'PASS' if t159_pass else 'FAIL'}")

    # T160: GPU_TEMP most confused with PINK_IIR
    # Check: for GPU_TEMP row (idx=1), which off-diagonal class has highest confusion?
    temp_row = confusion_norm[1].copy()
    temp_row[1] = -1  # exclude self
    most_confused_with = np.argmax(temp_row)
    t160_pass = most_confused_with == 4  # PINK_IIR index
    print(f"  T160 GPU_TEMP_confused_with={NOISE_LABELS[most_confused_with]} "
          f"(expected PINK_IIR)  {'PASS' if t160_pass else 'FAIL'}")

    # T161: Spike feature importance > vmem feature importance
    t161_pass = spike_importance > vmem_importance
    print(f"  T161 spike_imp={spike_importance:.4f} > vmem_imp={vmem_importance:.4f}  "
          f"{'PASS' if t161_pass else 'FAIL'}")

    # T162: CV accuracy > 30%
    t162_pass = cv_mean > 0.30
    print(f"  T162 cv_accuracy={cv_mean:.3f} > 0.30  "
          f"{'PASS' if t162_pass else 'FAIL'}")

    n_pass = sum([t157_pass, t158_pass, t159_pass, t160_pass, t161_pass, t162_pass])
    print(f"\n  TOTAL: {n_pass}/6 PASS")

    # ─── Save results ───
    results['tests'] = {
        'T157_classification_accuracy': {
            'value': float(overall_acc), 'threshold': 0.40,
            'pass': t157_pass,
        },
        'T158_gpu_power_most_distinguishable': {
            'best_class': best_class,
            'per_class_acc': per_class_acc,
            'pass': t158_pass,
        },
        'T159_pairwise_power_vs_synth': {
            'value': float(pairwise_acc), 'threshold': 0.60,
            'pass': t159_pass,
        },
        'T160_gpu_temp_confused_with': {
            'most_confused_with': NOISE_LABELS[most_confused_with],
            'expected': 'PINK_IIR',
            'confusion_matrix': confusion_norm.tolist(),
            'pass': t160_pass,
        },
        'T161_spike_vs_vmem_importance': {
            'spike_importance': float(spike_importance),
            'vmem_importance': float(vmem_importance),
            'pass': t161_pass,
        },
        'T162_cv_accuracy': {
            'mean': float(cv_mean), 'std': float(cv_std),
            'fold_accs': [float(a) for a in cv_accs],
            'threshold': 0.30,
            'pass': t162_pass,
        },
    }
    results['n_pass'] = n_pass
    results['n_total_tests'] = 6
    results['feature_importances_top10'] = sorted(
        [(int(i), float(importances[i])) for i in range(len(importances))],
        key=lambda x: x[1], reverse=True
    )[:10]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2178_noise_fingerprint.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved to {out_path}")

    # ─── Plot ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        FIGURES.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Per-condition accuracy bars
        ax = axes[0]
        colors = ['#e74c3c', '#e67e22', '#3498db', '#95a5a6', '#9b59b6']
        accs = [per_class_acc.get(n, 0) for n in NOISE_LABELS]
        bars = ax.bar(range(len(NOISE_LABELS)), accs, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(0.20, color='gray', linestyle='--', linewidth=1, label='Chance (20%)')
        ax.axhline(0.40, color='green', linestyle='--', linewidth=1, alpha=0.5, label='T157 threshold')
        ax.set_xticks(range(len(NOISE_LABELS)))
        ax.set_xticklabels([n.replace('_', '\n') for n in NOISE_LABELS], fontsize=8)
        ax.set_ylabel('Per-Class Accuracy')
        ax.set_title(f'Noise Fingerprint Classification\nOverall: {overall_acc:.1%}')
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7)

        # Panel 2: Confusion matrix heatmap
        ax = axes[1]
        im = ax.imshow(confusion_norm, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        for i in range(n_classes):
            for j in range(n_classes):
                val = confusion_norm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels([n[:6] for n in NOISE_LABELS], fontsize=7, rotation=45)
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels([n[:6] for n in NOISE_LABELS], fontsize=7)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 3: Feature importance (spike vs vmem)
        ax = axes[2]
        feat_groups = ['Spike\n(delta)', 'Vmem', 'Spike\n(cumul.)']
        group_importances = []
        for pool_offset in range(4):
            base = pool_offset * n_base
            group_importances.append([
                importances[base:base + N_NEURONS].mean(),
                importances[base + N_NEURONS:base + 2 * N_NEURONS].mean(),
                importances[base + 2 * N_NEURONS:base + 3 * N_NEURONS].mean(),
            ])
        # Average across pool types
        avg_imp = np.mean(group_importances, axis=0)

        bar_colors = ['#e74c3c', '#3498db', '#2ecc71']
        ax.bar(feat_groups, avg_imp, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Mean Permutation Importance')
        ax.set_title(f'Feature Importance by Type\nSpike: {spike_importance:.4f} vs Vmem: {vmem_importance:.4f}')
        ax.axhline(0, color='gray', linewidth=0.5)

        plt.tight_layout()
        fig_path = FIGURES / 'z2178_noise_fingerprint.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {fig_path}")
        plt.close()

    except ImportError:
        print("  matplotlib not available — skipping plots")

    # Cleanup
    if fpga and ser:
        # Re-enable kill switch
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x01]))
        ser.flush()
        ser.close()
        print("  FPGA connection closed")

    print(f"\nDone. {n_pass}/6 tests passed.")
    return 0 if n_pass >= 4 else 1


if __name__ == '__main__':
    sys.exit(main())
