#!/usr/bin/env python3
"""z2167_scaling_law.py — Reservoir Scaling Law: neurons, coupling, temporal memory

Systematically maps how FPGA reservoir performance scales with 3 key parameters:
  1. n_active_neurons: [2, 4, 6, 8] — disable inactive by setting Vg=0
  2. alpha (input coupling): [0.10, 0.20, 0.30, 0.40]
  3. iir_alpha (temporal memory): [0.50, 0.70, 0.85, 0.95]

Smart experimental design (not full factorial):
  Phase 1: Full n_neurons x alpha grid (16 configs) at IIR=0.85
  Phase 2: IIR sweep at n_neurons=8, alpha=0.25 (4 configs)
  Total: 20 configs x 60 trials x 20 steps ~ 20 minutes

Tests:
  T91: Accuracy increases with n_neurons (monotonic, >=3/4 ordered pairs)
  T92: Best alpha between 0.15-0.35
  T93: IIR alpha=0.85 or 0.95 beats alpha=0.50
  T94: 8-neuron config achieves >=55% waveform accuracy
  T95: 2-neuron config > 33% chance
  T96: Scaling sublinear — accuracy_per_neuron decreases with more neurons

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB1
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Reservoir Parameters ───
BASE_VG = 0.58       # near BVpar cliff
N_NEURONS = 8
SAMPLE_HZ = 20

# ─── Scaling Grid ───
NEURON_COUNTS = [2, 4, 6, 8]
ALPHA_VALUES = [0.10, 0.20, 0.30, 0.40]
IIR_ALPHA_VALUES = [0.50, 0.70, 0.85, 0.95]
DEFAULT_IIR_ALPHA = 0.85
DEFAULT_ALPHA = 0.25
BETA = 0.08


def _print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ═══════════════════════════════════════════════════════════
# FPGA Communication (from z2165/z2162)
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


def reconnect_fpga(port):
    """Reconnect to FPGA after serial failure."""
    import serial
    try:
        ser = serial.Serial(port, 115200, timeout=0.2)
        time.sleep(0.1)
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        return ser
    except Exception:
        return None


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


def safe_fpga_step(ser, port, vg_values, interval):
    """Set Vg and read telemetry with reconnection on serial failure."""
    import serial as serial_mod
    try:
        set_per_neuron_vg(ser, vg_values)
        time.sleep(interval * 0.3)
        ser.reset_input_buffer()
        ser.write(bytes([SYNC, CMD_READ_TELEM]))
        ser.flush()
        telem = read_telem(ser, timeout=0.15)
        return ser, telem
    except (serial_mod.SerialException, OSError) as e:
        _print(f"    [!] Serial error: {e}, reconnecting...")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.5)
        new_ser = reconnect_fpga(port)
        if new_ser is None:
            _print("    [!] Reconnection failed")
            return None, None
        _print("    [!] Reconnected successfully")
        return new_ser, None


# ═══════════════════════════════════════════════════════════
# Noise Source
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


def normalize_noise(samples):
    """Zero-mean, unit-variance normalization."""
    arr = np.array(samples, dtype=float)
    if len(arr) == 0:
        return arr
    mu = arr.mean()
    std = max(arr.std(), 1e-6)
    return (arr - mu) / std


def iir_filter_noise(noise_samples, alpha_iir=0.85):
    """Apply IIR low-pass: y[t] = alpha*y[t-1] + (1-alpha)*x[t].
    Creates temporal memory from raw noise.
    """
    if len(noise_samples) == 0:
        return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t-1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


def generate_synthetic_1f(n_samples, rng):
    """Voss-McCartney 1/f generator for simulation fallback."""
    noise = np.zeros(n_samples)
    n_octaves = 8
    octaves = np.zeros(n_octaves)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    return normalize_noise(noise)


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=60, steps_per_trial=20, freq_hz=1.0, dt=1.0/20):
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
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2*np.pi)) % 1.0) - 1.0) - 1.0
        else:  # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))

        wave = (wave + 1.0) / 2.0
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_trial(ser, port, input_signal, noise_samples, w_in, w_noise,
                   active_neurons, base_vg=BASE_VG, alpha=0.25, beta=BETA):
    """Drive FPGA with n active neurons and collect states.

    Inactive neurons get Vg=0 (won't spike).
    Returns: (ser, states) where states is (n_steps, n_active*3).
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    n_active = len(active_neurons)
    states = np.zeros((n_steps, n_active * 3))  # delta + vmem + cumulative
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        # Compute per-neuron Vg (all 8 neurons)
        vg_values = np.zeros(N_NEURONS)
        for idx, nid in enumerate(active_neurons):
            vg_values[nid] = base_vg + alpha * input_signal[t] * w_in[nid]
            if beta > 0 and len(noise_samples) > 0:
                noise_val = noise_samples[t % len(noise_samples)]
                vg_values[nid] += beta * noise_val * w_noise[nid]
            vg_values[nid] = max(0.05, min(0.95, vg_values[nid]))
        # Inactive neurons remain at Vg=0

        ser, telem = safe_fpga_step(ser, port, vg_values, interval)
        if ser is None:
            return None, states

        if telem:
            counts = [n['spike_count'] for n in telem]
            vmems = [n['vmem'] for n in telem]

            if prev_counts is not None:
                for i, nid in enumerate(active_neurons):
                    delta = (counts[nid] - prev_counts[nid]) & 0xFFFF
                    if delta > 30000:
                        delta = 0
                    states[t, i] = delta
                    cumulative[nid] += delta
            for i, nid in enumerate(active_neurons):
                states[t, n_active + i] = vmems[nid]
                states[t, n_active * 2 + i] = cumulative[nid]
            prev_counts = counts[:]

        time.sleep(max(0, interval * 0.5 - 0.01))

    return ser, states


def simulate_lif_trial(input_signal, noise_samples, w_in, w_noise,
                       active_neurons, base_vg=BASE_VG, alpha=0.25, beta=BETA):
    """Software LIF simulation fallback."""
    n_steps = len(input_signal)
    n_active = len(active_neurons)
    states = np.zeros((n_steps, n_active * 3))

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / SAMPLE_HZ
    vmem = np.zeros(N_NEURONS)
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        vg = np.zeros(N_NEURONS)
        for idx, nid in enumerate(active_neurons):
            vg[nid] = base_vg + alpha * input_signal[t] * w_in[nid]
            if beta > 0 and len(noise_samples) > 0:
                noise_val = noise_samples[t % len(noise_samples)]
                vg[nid] += beta * noise_val * w_noise[nid]
            vg[nid] = max(0.05, min(0.95, vg[nid]))

        I_in = vg * 5.0
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        for i, nid in enumerate(active_neurons):
            if vmem[nid] >= v_thresh:
                states[t, i] = 1
                vmem[nid] = v_rest
                cumulative[nid] += 1
            states[t, n_active + i] = vmem[nid]
            states[t, n_active * 2 + i] = cumulative[nid]

        # Reset inactive neuron vmem to prevent drift
        for nid in range(N_NEURONS):
            if nid not in active_neurons:
                vmem[nid] = 0.0

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


def evaluate_waveform_cv(trial_features, labels, n_splits=5):
    """5-fold stratified CV waveform accuracy."""
    X = np.array(trial_features)
    y = np.array(labels)

    # Add small noise to avoid singular matrices
    X += np.random.default_rng(99).normal(0, 1e-8, X.shape)

    splits = stratified_kfold(X, y, n_splits=n_splits)
    accs = []
    for train_idx, test_idx in splits:
        acc = ridge_classify(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        if acc >= 0:
            accs.append(acc)
    return float(np.mean(accs)) if accs else 0.333


# ═══════════════════════════════════════════════════════════
# Run Single Configuration
# ═══════════════════════════════════════════════════════════

def run_config(ser, port, n_active, alpha_val, iir_alpha, noise_raw, w_in, w_noise,
               n_trials, steps_per_trial, use_fpga):
    """Run waveform classification for one configuration.

    Returns: (accuracy, ser)
    """
    active_neurons = list(range(n_active))
    noise_filtered = iir_filter_noise(noise_raw.copy(), alpha_iir=iir_alpha)

    waveforms, labels = generate_waveforms(n_trials=n_trials,
                                           steps_per_trial=steps_per_trial)

    trial_features = []
    for trial_idx in range(n_trials):
        input_signal = waveforms[trial_idx]

        if use_fpga and ser is not None:
            ser, states = run_fpga_trial(ser, port, input_signal, noise_filtered,
                                         w_in, w_noise, active_neurons,
                                         alpha=alpha_val, beta=BETA)
            if ser is None:
                _print("    [!] FPGA lost, switching to simulation")
                use_fpga = False
                states = simulate_lif_trial(input_signal, noise_filtered,
                                            w_in, w_noise, active_neurons,
                                            alpha=alpha_val, beta=BETA)
        else:
            states = simulate_lif_trial(input_signal, noise_filtered,
                                        w_in, w_noise, active_neurons,
                                        alpha=alpha_val, beta=BETA)

        aug = augment_with_delays(states, delays=(1, 2, 3))
        feat = pool_trial_features(aug)
        trial_features.append(feat)

    accuracy = evaluate_waveform_cv(trial_features, labels, n_splits=5)
    return accuracy, ser


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_plots(results_data, fig_path):
    """Create 3-panel scaling law figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        _print("[!] matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ─── Panel 1: Accuracy vs n_neurons (at best alpha, IIR=0.85) ───
    ax = axes[0]
    neuron_accs = results_data.get('neuron_scaling', {})
    if neuron_accs:
        ns = sorted([int(k) for k in neuron_accs.keys()])
        accs = [neuron_accs[str(n)] for n in ns]
        ax.plot(ns, accs, 'o-', color='#2196F3', linewidth=2, markersize=8)
        ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Number of Active Neurons', fontsize=12)
        ax.set_ylabel('Waveform Accuracy', fontsize=12)
        ax.set_title('Scaling with Neuron Count', fontsize=13, fontweight='bold')
        ax.set_xticks(ns)
        ax.set_ylim([0.25, 0.85])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # ─── Panel 2: Accuracy vs alpha (at n=8, IIR=0.85) ───
    ax = axes[1]
    alpha_accs = results_data.get('alpha_scaling', {})
    if alpha_accs:
        alphas = sorted([float(k) for k in alpha_accs.keys()])
        accs = [alpha_accs[f"{a:.2f}"] for a in alphas]
        ax.plot(alphas, accs, 's-', color='#FF5722', linewidth=2, markersize=8)
        ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Input Coupling (alpha)', fontsize=12)
        ax.set_ylabel('Waveform Accuracy', fontsize=12)
        ax.set_title('Scaling with Input Coupling', fontsize=13, fontweight='bold')
        ax.set_ylim([0.25, 0.85])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # ─── Panel 3: Accuracy vs IIR alpha (at n=8, alpha=0.25) ───
    ax = axes[2]
    iir_accs = results_data.get('iir_scaling', {})
    if iir_accs:
        iirs = sorted([float(k) for k in iir_accs.keys()])
        accs = [iir_accs[f"{a:.2f}"] for a in iirs]
        ax.plot(iirs, accs, 'D-', color='#4CAF50', linewidth=2, markersize=8)
        ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('IIR Alpha (temporal memory)', fontsize=12)
        ax.set_ylabel('Waveform Accuracy', fontsize=12)
        ax.set_title('Scaling with Temporal Memory', fontsize=13, fontweight='bold')
        ax.set_ylim([0.25, 0.85])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('z2167: Reservoir Scaling Law — Neurons, Coupling, Memory',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    _print(f"  Figure saved: {fig_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2167: Reservoir Scaling Law')
    parser.add_argument('--n-trials', type=int, default=60,
                        help='Waveform trials per configuration (default: 60)')
    parser.add_argument('--steps-per-trial', type=int, default=20,
                        help='Time steps per trial (default: 20)')
    parser.add_argument('--noise-collect-s', type=float, default=15.0,
                        help='Seconds of noise to collect (default: 15)')
    args = parser.parse_args()

    _print("=" * 65)
    _print("z2167: Reservoir Scaling Law — Neurons, Coupling, Memory")
    _print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2167_scaling_law',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'neuron_counts': NEURON_COUNTS,
            'alpha_values': ALPHA_VALUES,
            'iir_alpha_values': IIR_ALPHA_VALUES,
            'n_trials': args.n_trials,
            'steps_per_trial': args.steps_per_trial,
        },
        'configs': [],
        'tests': {},
    }

    # ─── Connect FPGA ───
    ser, port = find_fpga()
    use_fpga = ser is not None
    if use_fpga:
        _print(f"  FPGA connected on {port}")
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
    else:
        _print("  [!] No FPGA found — running simulation fallback")

    # ─── Collect noise ───
    _print(f"\n[1/3] Collecting {args.noise_collect_s}s power noise for 1/f source...")
    noise_raw_arr = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if noise_raw_arr is None or len(noise_raw_arr) < 10:
        _print("  [!] hwmon power not available, using synthetic 1/f")
        noise_raw_arr = generate_synthetic_1f(int(args.noise_collect_s * 50), rng)
    else:
        _print(f"  Collected {len(noise_raw_arr)} samples, mean={noise_raw_arr.mean():.2f}W")
        noise_raw_arr = normalize_noise(noise_raw_arr)

    # ─── Phase 1: n_neurons x alpha grid at IIR=0.85 ───
    _print(f"\n[2/3] Phase 1: {len(NEURON_COUNTS)}x{len(ALPHA_VALUES)} = "
           f"{len(NEURON_COUNTS)*len(ALPHA_VALUES)} configs (n_neurons x alpha, IIR={DEFAULT_IIR_ALPHA})")

    config_idx = 0
    total_configs = len(NEURON_COUNTS) * len(ALPHA_VALUES) + len(IIR_ALPHA_VALUES)

    # Store results by axis for analysis
    # neuron_scaling: for each n, best accuracy across alphas
    # alpha_scaling: for each alpha, accuracy at n=8
    grid_results = {}  # (n_neurons, alpha) -> accuracy

    for n_active in NEURON_COUNTS:
        for alpha_val in ALPHA_VALUES:
            config_idx += 1
            _print(f"  Config {config_idx}/{total_configs}: "
                   f"n={n_active}, alpha={alpha_val:.2f}, iir={DEFAULT_IIR_ALPHA}")
            t0 = time.monotonic()

            acc, ser = run_config(ser, port, n_active, alpha_val, DEFAULT_IIR_ALPHA,
                                  noise_raw_arr, w_in, w_noise,
                                  args.n_trials, args.steps_per_trial, use_fpga)

            elapsed = time.monotonic() - t0
            _print(f"    accuracy={acc:.3f} ({elapsed:.1f}s)")

            grid_results[(n_active, alpha_val)] = acc
            results['configs'].append({
                'n_active': n_active,
                'alpha': alpha_val,
                'iir_alpha': DEFAULT_IIR_ALPHA,
                'accuracy': acc,
                'elapsed_s': round(elapsed, 1),
                'phase': 1,
            })

    # ─── Phase 2: IIR sweep at n=8, alpha=0.25 ───
    _print(f"\n[3/3] Phase 2: IIR sweep at n={N_NEURONS}, alpha={DEFAULT_ALPHA}")

    iir_results = {}
    for iir_a in IIR_ALPHA_VALUES:
        config_idx += 1
        _print(f"  Config {config_idx}/{total_configs}: "
               f"n={N_NEURONS}, alpha={DEFAULT_ALPHA:.2f}, iir={iir_a:.2f}")
        t0 = time.monotonic()

        acc, ser = run_config(ser, port, N_NEURONS, DEFAULT_ALPHA, iir_a,
                              noise_raw_arr, w_in, w_noise,
                              args.n_trials, args.steps_per_trial, use_fpga)

        elapsed = time.monotonic() - t0
        _print(f"    accuracy={acc:.3f} ({elapsed:.1f}s)")

        iir_results[iir_a] = acc
        results['configs'].append({
            'n_active': N_NEURONS,
            'alpha': DEFAULT_ALPHA,
            'iir_alpha': iir_a,
            'accuracy': acc,
            'elapsed_s': round(elapsed, 1),
            'phase': 2,
        })

    # ─── Build scaling curves ───
    # Neuron scaling: for each n, take BEST accuracy across all alphas
    neuron_scaling = {}
    for n_active in NEURON_COUNTS:
        best_acc = max(grid_results[(n_active, a)] for a in ALPHA_VALUES)
        neuron_scaling[str(n_active)] = round(best_acc, 4)

    # Alpha scaling: at n=8, accuracy for each alpha
    alpha_scaling = {}
    for alpha_val in ALPHA_VALUES:
        alpha_scaling[f"{alpha_val:.2f}"] = round(grid_results[(8, alpha_val)], 4)

    # IIR scaling
    iir_scaling = {}
    for iir_a in IIR_ALPHA_VALUES:
        iir_scaling[f"{iir_a:.2f}"] = round(iir_results[iir_a], 4)

    results['neuron_scaling'] = neuron_scaling
    results['alpha_scaling'] = alpha_scaling
    results['iir_scaling'] = iir_scaling

    _print("\n" + "=" * 65)
    _print("SCALING CURVES:")
    _print(f"  Neuron scaling (best across alpha): {neuron_scaling}")
    _print(f"  Alpha scaling (at n=8):             {alpha_scaling}")
    _print(f"  IIR scaling (at n=8, alpha=0.25):   {iir_scaling}")

    # ─── Tests ───
    _print("\n" + "=" * 65)
    _print("TESTS:")

    n_vals = sorted([int(k) for k in neuron_scaling.keys()])
    n_accs = [neuron_scaling[str(n)] for n in n_vals]

    # T91: Monotonic increase with neurons (>=3/4 ordered pairs)
    ordered_pairs = 0
    total_pairs = 0
    for i in range(len(n_vals)):
        for j in range(i+1, len(n_vals)):
            total_pairs += 1
            if n_accs[j] >= n_accs[i]:
                ordered_pairs += 1
    t91_pass = ordered_pairs >= 3
    results['tests']['T91_monotonic_neurons'] = {
        'pass': t91_pass,
        'ordered_pairs': ordered_pairs,
        'total_pairs': total_pairs,
        'description': f'Accuracy increases with n_neurons: {ordered_pairs}/{total_pairs} ordered'
    }
    _print(f"  T91 monotonic_neurons: {'PASS' if t91_pass else 'FAIL'} "
           f"({ordered_pairs}/{total_pairs} ordered pairs)")

    # T92: Best alpha between 0.15-0.35
    alpha_vals_sorted = sorted([float(k) for k in alpha_scaling.keys()])
    alpha_accs_sorted = [alpha_scaling[f"{a:.2f}"] for a in alpha_vals_sorted]
    best_alpha_idx = int(np.argmax(alpha_accs_sorted))
    best_alpha = alpha_vals_sorted[best_alpha_idx]
    t92_pass = 0.15 <= best_alpha <= 0.35
    results['tests']['T92_optimal_alpha_range'] = {
        'pass': t92_pass,
        'best_alpha': best_alpha,
        'best_accuracy': alpha_accs_sorted[best_alpha_idx],
        'description': f'Best alpha={best_alpha:.2f} in [0.15, 0.35]: {t92_pass}'
    }
    _print(f"  T92 optimal_alpha_range: {'PASS' if t92_pass else 'FAIL'} "
           f"(best alpha={best_alpha:.2f})")

    # T93: IIR 0.85 or 0.95 beats 0.50
    iir_050 = iir_results.get(0.50, 0)
    iir_085 = iir_results.get(0.85, 0)
    iir_095 = iir_results.get(0.95, 0)
    t93_pass = max(iir_085, iir_095) > iir_050
    results['tests']['T93_temporal_memory_helps'] = {
        'pass': t93_pass,
        'iir_050': round(iir_050, 4),
        'iir_085': round(iir_085, 4),
        'iir_095': round(iir_095, 4),
        'description': f'max(IIR_0.85={iir_085:.3f}, IIR_0.95={iir_095:.3f}) > IIR_0.50={iir_050:.3f}'
    }
    _print(f"  T93 temporal_memory: {'PASS' if t93_pass else 'FAIL'} "
           f"(0.85={iir_085:.3f}, 0.95={iir_095:.3f} vs 0.50={iir_050:.3f})")

    # T94: 8-neuron config >= 55% accuracy
    acc_8 = neuron_scaling.get('8', 0)
    t94_pass = acc_8 >= 0.55
    results['tests']['T94_8neuron_accuracy'] = {
        'pass': t94_pass,
        'accuracy_8': acc_8,
        'threshold': 0.55,
        'description': f'8-neuron accuracy={acc_8:.3f} >= 0.55'
    }
    _print(f"  T94 8neuron_accuracy: {'PASS' if t94_pass else 'FAIL'} "
           f"(acc={acc_8:.3f})")

    # T95: 2-neuron config > 33% chance
    acc_2 = neuron_scaling.get('2', 0)
    t95_pass = acc_2 > 0.333
    results['tests']['T95_2neuron_above_chance'] = {
        'pass': t95_pass,
        'accuracy_2': acc_2,
        'threshold': 0.333,
        'description': f'2-neuron accuracy={acc_2:.3f} > 0.333'
    }
    _print(f"  T95 2neuron_above_chance: {'PASS' if t95_pass else 'FAIL'} "
           f"(acc={acc_2:.3f})")

    # T96: Sublinear scaling — accuracy/neuron decreases
    acc_per_neuron = {}
    for n in n_vals:
        acc_per_neuron[n] = neuron_scaling[str(n)] / n
    apn_vals = [acc_per_neuron[n] for n in n_vals]
    # Check that acc/neuron decreases for at least 2/3 consecutive pairs
    decreasing = sum(1 for i in range(len(apn_vals)-1) if apn_vals[i+1] < apn_vals[i])
    t96_pass = decreasing >= 2
    results['tests']['T96_sublinear_scaling'] = {
        'pass': t96_pass,
        'acc_per_neuron': {str(n): round(v, 5) for n, v in acc_per_neuron.items()},
        'decreasing_pairs': decreasing,
        'description': f'Sublinear: acc/neuron decreasing for {decreasing}/3 pairs'
    }
    _print(f"  T96 sublinear_scaling: {'PASS' if t96_pass else 'FAIL'} "
           f"({decreasing}/3 consecutive pairs decreasing)")

    # ─── Summary ───
    all_tests = results['tests']
    n_pass = sum(1 for t in all_tests.values() if t['pass'])
    n_total = len(all_tests)
    results['summary'] = {
        'pass': n_pass,
        'total': n_total,
        'pass_rate': f"{n_pass}/{n_total}",
        'hw_mode': 'FPGA' if use_fpga else 'SIMULATION',
    }

    _print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2167_scaling_law.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    _print(f"\n  Results saved: {out_path}")

    # ─── Plot ───
    fig_path = FIGURES / 'fig_z2167_scaling_law.png'
    make_plots(results, fig_path)

    # ─── Cleanup ───
    if ser:
        try:
            set_per_neuron_vg(ser, [0.0] * N_NEURONS)
            ser.close()
        except Exception:
            pass

    _print("\nDone.")


if __name__ == '__main__':
    main()
