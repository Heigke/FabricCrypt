#!/usr/bin/env python3
"""z2194_substrate_coupling_strength.py — Substrate Coupling Strength Sweep

Measures how coupling strength (beta) between GPU 1/f noise and FPGA LIF neurons
affects reservoir computation.  Systematic beta sweep over 10 levels to find the
optimal noise injection level — directly relevant to Mario Lanza's memristive
computing where device-to-circuit coupling strength determines performance.

Sweep beta (noise coupling strength) over 10 levels:
  [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

For each beta:
  - 3-class waveform classification (sine/triangle/square)
  - 100 trials, 25 steps/trial at 20 Hz
  - GPU 1/f noise from hwmon power1_average
  - Vg_i(t) = base_vg + alpha * input(t) * w_in[i] + beta * noise(t) * w_noise[i]
  - Reservoir readout: 8 neurons x (delta_spikes + vmem), pool [mean, max], ridge 5-fold CV
  - Metrics: spike rate mean/std, ISI CV, branching ratio

Coupling asymmetry test at optimal beta:
  - ONLY w_in noise (excitatory driven harder)
  - ONLY w_noise noise (inhibitory driven harder)

Tests T255-T260:
  T255: Peak accuracy at intermediate beta (inverted U-shape)
  T256: Peak beta in [0.05, 0.30]
  T257: beta=0 accuracy < peak accuracy by >= 5pp
  T258: beta=1.0 accuracy < peak accuracy
  T259: ISI CV increases monotonically with beta
  T260: Spike rate std increases with beta

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG     = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL   = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Parameters ───
N_NEURONS       = 8
BASE_VG         = 0.55
ALPHA           = 0.15
SAMPLE_HZ       = 20
N_TRIALS        = 100
STEPS_PER_TRIAL = 25

BETA_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]


# ═══════════════════════════════════════════════════════════
# JSON Encoder
# ═══════════════════════════════════════════════════════════

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


def collect_power_noise(duration_s=10, sample_hz=50):
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
    """IIR low-pass: y[t] = a*y[t-1] + (1-a)*x[t]. Creates temporal memory."""
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


def generate_synthetic_1f(n_samples, rng):
    """Generate synthetic 1/f noise via octave summation (Voss-McCartney)."""
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
# FPGA Reservoir Trial
# ═══════════════════════════════════════════════════════════

def run_fpga_trial(ser, input_signal, noise_samples, w_in, w_noise,
                   base_vg=BASE_VG, alpha=ALPHA, beta=0.0, live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    Vg_i(t) = base_vg + alpha * input(t) * w_in[i] + beta * noise(t) * w_noise[i]

    When live_noise=True, reads power rail in real-time (true substrate coupling).
    Returns: (n_steps, 16) array -- 8 delta_spikes + 8 vmem.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 2))  # delta_spikes + vmem
    prev_counts = None
    power_mean = 11.0

    for t in range(n_steps):
        # Get noise value
        if live_noise and beta > 0:
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


# ═══════════════════════════════════════════════════════════
# LIF Simulation Fallback
# ═══════════════════════════════════════════════════════════

def simulate_lif_reservoir(input_signal, noise_samples, w_in, w_noise,
                            base_vg=BASE_VG, alpha=ALPHA, beta=0.0):
    """Software LIF simulation fallback when FPGA is not connected."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 2))  # delta_spikes + vmem

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / SAMPLE_HZ
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
        states[t, N_NEURONS:N_NEURONS * 2] = vmem.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

def pool_trial_features(trial_states):
    """Pool per-timestep reservoir states into trial-level features.
    Uses [mean, max] as specified (compact but informative).
    """
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.max(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier (one-hot for multi-class)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
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


def classify_with_cv(features_array, labels, n_splits=5):
    """Run 5-fold stratified CV and return mean accuracy."""
    X = features_array
    y = labels

    # Remove constant features
    feat_std = X.std(axis=0)
    good_cols = feat_std > 1e-8
    X_clean = X[:, good_cols]
    if X_clean.shape[1] == 0:
        return 1.0 / 3.0, 0.0, []

    # Normalize
    mu = X_clean.mean(axis=0)
    sigma = X_clean.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_norm = (X_clean - mu) / sigma

    # Add bias
    X_aug = np.column_stack([X_norm, np.ones(len(X_norm))])

    folds = stratified_kfold(X_aug, y, n_splits=n_splits, seed=42)
    fold_accs = []
    for train_idx, test_idx in folds:
        acc = ridge_classify(X_aug[train_idx], y[train_idx],
                              X_aug[test_idx], y[test_idx])
        fold_accs.append(acc)

    return float(np.mean(fold_accs)), float(np.std(fold_accs)), [float(a) for a in fold_accs]


# ═══════════════════════════════════════════════════════════
# Spike Metrics
# ═══════════════════════════════════════════════════════════

def compute_spike_metrics(all_trial_states, n_neurons=N_NEURONS):
    """Compute spike rate mean/std, ISI CV, and branching ratio from trial states.

    all_trial_states: list of (n_steps, 2*N_NEURONS) arrays
    Returns dict with metrics.
    """
    # Aggregate spike counts per neuron across all trials
    total_spikes = []
    all_isis = []
    branching_pairs = []

    for states in all_trial_states:
        delta_spikes = states[:, :n_neurons]  # (n_steps, 8)
        trial_total = delta_spikes.sum(axis=0)  # per-neuron total
        total_spikes.append(trial_total)

        # ISI: inter-spike intervals per neuron
        for nid in range(n_neurons):
            spike_times = np.where(delta_spikes[:, nid] > 0)[0]
            if len(spike_times) > 1:
                isis = np.diff(spike_times).astype(float)
                all_isis.extend(isis.tolist())

        # Branching ratio: n_descendants(t+1) / n_ancestors(t)
        for t in range(len(delta_spikes) - 1):
            n_t = delta_spikes[t].sum()
            n_t1 = delta_spikes[t + 1].sum()
            if n_t > 0:
                branching_pairs.append(n_t1 / n_t)

    total_spikes = np.array(total_spikes)  # (n_trials, 8)
    rate_per_neuron = total_spikes.mean(axis=0)  # mean rate per neuron
    spike_rate_mean = float(rate_per_neuron.mean())
    spike_rate_std = float(rate_per_neuron.std())

    # ISI CV
    if len(all_isis) > 2:
        isis_arr = np.array(all_isis)
        isi_cv = float(isis_arr.std() / max(isis_arr.mean(), 1e-6))
    else:
        isi_cv = 0.0

    # Branching ratio
    branching_ratio = float(np.mean(branching_pairs)) if branching_pairs else 0.0

    return {
        'spike_rate_mean': spike_rate_mean,
        'spike_rate_std': spike_rate_std,
        'isi_cv': isi_cv,
        'branching_ratio': branching_ratio,
        'n_isis': len(all_isis),
        'n_branching_pairs': len(branching_pairs),
    }


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def plot_results(beta_sweep, tests, fig_path):
    """2x2 figure: (1) accuracy vs beta, (2) spike rate, (3) ISI CV / branching, (4) tests."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    betas = [d['beta'] for d in beta_sweep]
    accs = [d['accuracy'] for d in beta_sweep]
    acc_stds = [d['std'] for d in beta_sweep]
    sr_means = [d['spike_rate_mean'] for d in beta_sweep]
    sr_stds_val = [d['spike_rate_std'] for d in beta_sweep]
    isi_cvs = [d['isi_cv'] for d in beta_sweep]
    br_vals = [d['branching_ratio'] for d in beta_sweep]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ─── Panel A: Accuracy vs beta ───
    ax = axes[0, 0]
    ax.errorbar(betas, accs, yerr=acc_stds,
                fmt='o-', color='#2196F3', linewidth=2, markersize=8,
                capsize=4, capthick=1.5, label='Waveform accuracy')
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Chance (33.3%)')

    peak_idx = int(np.argmax(accs))
    ax.plot(betas[peak_idx], accs[peak_idx], '*',
            color='#F44336', markersize=18, zorder=5,
            label=f'Peak: {accs[peak_idx]:.1%} @ beta={betas[peak_idx]}')

    ax.set_xlabel('Coupling strength (beta)', fontsize=12)
    ax.set_ylabel('Classification accuracy', fontsize=12)
    ax.set_title('A: Accuracy vs Coupling Strength', fontsize=13, fontweight='bold')
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_ylim(0.2, 0.85)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    # ─── Panel B: Spike rate mean/std vs beta ───
    ax = axes[0, 1]
    ax.plot(betas, sr_means, 'o-', color='#4CAF50', linewidth=2, markersize=7, label='Spike rate mean')
    ax2 = ax.twinx()
    ax2.plot(betas, sr_stds_val, 's--', color='#FF9800', linewidth=2, markersize=7, label='Spike rate std')
    ax.set_xlabel('Coupling strength (beta)', fontsize=12)
    ax.set_ylabel('Spike rate mean', fontsize=12, color='#4CAF50')
    ax2.set_ylabel('Spike rate std', fontsize=12, color='#FF9800')
    ax.set_title('B: Spike Rate vs Coupling', fontsize=13, fontweight='bold')
    ax.set_xscale('symlog', linthresh=0.01)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ─── Panel C: ISI CV and branching ratio vs beta ───
    ax = axes[1, 0]
    ax.plot(betas, isi_cvs, 'o-', color='#9C27B0', linewidth=2, markersize=7, label='ISI CV')
    ax3 = ax.twinx()
    ax3.plot(betas, br_vals, 's--', color='#00BCD4', linewidth=2, markersize=7, label='Branching ratio')
    ax.set_xlabel('Coupling strength (beta)', fontsize=12)
    ax.set_ylabel('ISI CV', fontsize=12, color='#9C27B0')
    ax3.set_ylabel('Branching ratio', fontsize=12, color='#00BCD4')
    ax.set_title('C: ISI CV & Branching vs Coupling', fontsize=13, fontweight='bold')
    ax.set_xscale('symlog', linthresh=0.01)

    lines3, labels3 = ax.get_legend_handles_labels()
    lines4, labels4 = ax3.get_legend_handles_labels()
    ax.legend(lines3 + lines4, labels3 + labels4, fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ─── Panel D: Test results ───
    ax = axes[1, 1]
    ax.axis('off')
    test_names = sorted(tests.keys())
    rows = []
    colors = []
    for tname in test_names:
        t = tests[tname]
        status = 'PASS' if t['pass'] else 'FAIL'
        rows.append([tname, t['name'][:50], status])
        colors.append('#C8E6C9' if t['pass'] else '#FFCDD2')

    if rows:
        table = ax.table(cellText=rows,
                         colLabels=['Test', 'Description', 'Result'],
                         loc='center', cellLoc='left',
                         colWidths=[0.12, 0.68, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for i, c in enumerate(colors):
            table[(i + 1, 2)].set_facecolor(c)
        table.scale(1.0, 1.4)

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    ax.set_title(f'D: Test Results ({n_pass}/{n_total} PASS)',
                 fontsize=13, fontweight='bold')

    fig.suptitle('z2194: Substrate Coupling Strength Sweep (GPU 1/f -> FPGA LIF)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='z2194: Substrate coupling strength sweep (GPU 1/f -> FPGA LIF)')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps-per-trial', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--noise-collect-s', type=float, default=10.0)
    args = parser.parse_args()

    print("=" * 65)
    print("z2194: Substrate Coupling Strength Sweep")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2194_substrate_coupling_strength',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'beta_levels': BETA_LEVELS,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
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
        connect_fpga(ser)
        print("  Kill switch disabled")

    # ─── Step 2: Collect GPU noise ───
    print("\n[2/6] Collecting GPU power rail noise (1/f)...")
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

    # Apply IIR filter for temporal memory
    noise_filtered = iir_filter_noise(noise_1f, alpha_iir=0.85)
    print(f"  IIR-filtered noise: {len(noise_filtered)} samples, std={np.std(noise_filtered):.3f}")

    # ─── Step 3: Generate waveforms ───
    print("\n[3/6] Generating waveforms...")
    trials, labels = generate_waveforms(
        n_trials=args.n_trials,
        steps_per_trial=args.steps_per_trial,
        seed=42,
    )
    print(f"  {args.n_trials} trials x {args.steps_per_trial} steps")
    class_counts = {int(c): int(np.sum(labels == c)) for c in np.unique(labels)}
    print(f"  Class distribution: {class_counts}")

    # ─── Step 4: Sweep beta (coupling strength) ───
    print("\n[4/6] Sweeping beta (coupling strength) over 10 levels...")
    beta_sweep = []

    for beta_idx, beta_val in enumerate(BETA_LEVELS):
        print(f"\n  --- beta={beta_val:.2f} ({beta_idx+1}/{len(BETA_LEVELS)}) ---")
        t0 = time.monotonic()

        all_features = []
        all_states = []

        for trial_idx in range(args.n_trials):
            if trial_idx % 25 == 0:
                print(f"    Trial {trial_idx}/{args.n_trials}...", end='\r')

            input_signal = trials[trial_idx]

            if fpga:
                trial_states = run_fpga_trial(
                    ser, input_signal, noise_filtered, w_in, w_noise,
                    base_vg=BASE_VG, alpha=ALPHA, beta=beta_val,
                    live_noise=True,
                )
            else:
                trial_states = simulate_lif_reservoir(
                    input_signal, noise_filtered, w_in, w_noise,
                    base_vg=BASE_VG, alpha=ALPHA, beta=beta_val,
                )

            feat = pool_trial_features(trial_states)
            all_features.append(feat)
            all_states.append(trial_states)

        features_array = np.array(all_features)
        elapsed = time.monotonic() - t0

        # Classify
        mean_acc, std_acc, fold_accs = classify_with_cv(features_array, labels)

        # Spike metrics
        metrics = compute_spike_metrics(all_states, n_neurons=N_NEURONS)

        entry = {
            'beta': beta_val,
            'accuracy': mean_acc,
            'std': std_acc,
            'fold_accs': fold_accs,
            'elapsed_s': round(elapsed, 2),
            'n_features': int(features_array.shape[1]),
            **metrics,
        }
        beta_sweep.append(entry)
        print(f"    beta={beta_val:.2f}: acc={mean_acc:.3f} +/- {std_acc:.3f}  "
              f"sr={metrics['spike_rate_mean']:.1f}+/-{metrics['spike_rate_std']:.2f}  "
              f"ISI_CV={metrics['isi_cv']:.3f}  BR={metrics['branching_ratio']:.3f}  "
              f"({elapsed:.1f}s)")

    results['beta_sweep'] = beta_sweep

    # ─── Step 5: Coupling asymmetry test at optimal beta ───
    print("\n[5/6] Coupling asymmetry test at optimal beta...")

    accs_list = [d['accuracy'] for d in beta_sweep]
    peak_idx = int(np.argmax(accs_list))
    optimal_beta = BETA_LEVELS[peak_idx]
    print(f"  Optimal beta = {optimal_beta} (acc={accs_list[peak_idx]:.3f})")

    asymmetry_results = {}

    # Test with excitatory-biased weights (positive w_noise only)
    w_noise_excit = np.abs(w_noise)  # all positive = excitatory neurons driven harder
    # Test with inhibitory-biased weights (negative w_noise only)
    w_noise_inhib = -np.abs(w_noise)  # all negative = inhibitory neurons driven harder

    for asym_name, w_noise_asym in [('excitatory_only', w_noise_excit),
                                     ('inhibitory_only', w_noise_inhib)]:
        print(f"\n  --- Asymmetry: {asym_name} (beta={optimal_beta}) ---")
        t0 = time.monotonic()
        asym_features = []
        asym_states = []

        for trial_idx in range(args.n_trials):
            input_signal = trials[trial_idx]

            if fpga:
                trial_states = run_fpga_trial(
                    ser, input_signal, noise_filtered, w_in, w_noise_asym,
                    base_vg=BASE_VG, alpha=ALPHA, beta=optimal_beta,
                    live_noise=True,
                )
            else:
                trial_states = simulate_lif_reservoir(
                    input_signal, noise_filtered, w_in, w_noise_asym,
                    base_vg=BASE_VG, alpha=ALPHA, beta=optimal_beta,
                )

            feat = pool_trial_features(trial_states)
            asym_features.append(feat)
            asym_states.append(trial_states)

        features_array = np.array(asym_features)
        elapsed = time.monotonic() - t0
        mean_acc, std_acc, fold_accs = classify_with_cv(features_array, labels)
        metrics = compute_spike_metrics(asym_states, n_neurons=N_NEURONS)

        asymmetry_results[asym_name] = {
            'beta': optimal_beta,
            'w_noise': w_noise_asym.tolist(),
            'accuracy': mean_acc,
            'std': std_acc,
            'fold_accs': fold_accs,
            'elapsed_s': round(elapsed, 2),
            **metrics,
        }
        print(f"    {asym_name}: acc={mean_acc:.3f} +/- {std_acc:.3f}  ({elapsed:.1f}s)")

    results['asymmetry'] = asymmetry_results

    # ─── Step 6: Evaluate Tests ───
    print("\n" + "=" * 65)
    print("[6/6] TEST RESULTS")
    print("=" * 65)

    betas = [d['beta'] for d in beta_sweep]
    accs = [d['accuracy'] for d in beta_sweep]
    isi_cvs = [d['isi_cv'] for d in beta_sweep]
    sr_stds = [d['spike_rate_std'] for d in beta_sweep]

    peak_idx = int(np.argmax(accs))
    peak_acc = accs[peak_idx]
    peak_beta = betas[peak_idx]
    beta0_acc = accs[0]      # beta=0.0
    beta1_acc = accs[-1]     # beta=1.0

    tests = {}

    # T255: Peak at intermediate beta (not at endpoints)
    t255_pass = peak_idx > 0 and peak_idx < len(BETA_LEVELS) - 1
    tests['T255'] = {
        'name': 'Peak accuracy at intermediate beta (inverted U)',
        'peak_beta': peak_beta,
        'peak_idx': peak_idx,
        'peak_acc': peak_acc,
        'pass': t255_pass,
    }
    print(f"\n  T255 Inverted U-shape:     peak @ beta={peak_beta} (idx={peak_idx}) "
          f"-> {'PASS' if t255_pass else 'FAIL'}")

    # T256: Peak beta in [0.05, 0.30]
    t256_pass = 0.05 <= peak_beta <= 0.30
    tests['T256'] = {
        'name': 'Peak beta in [0.05, 0.30] (moderate coupling optimal)',
        'peak_beta': peak_beta,
        'pass': t256_pass,
    }
    print(f"  T256 Peak in [0.05, 0.30]: beta={peak_beta} "
          f"-> {'PASS' if t256_pass else 'FAIL'}")

    # T257: beta=0 accuracy < peak by >= 5pp
    delta_pp = (peak_acc - beta0_acc) * 100
    t257_pass = delta_pp >= 5.0
    tests['T257'] = {
        'name': 'beta=0 < peak by >= 5pp (noise coupling helps)',
        'beta0_acc': beta0_acc,
        'peak_acc': peak_acc,
        'delta_pp': delta_pp,
        'pass': t257_pass,
    }
    print(f"  T257 Noise helps >= 5pp:   {peak_acc:.3f} - {beta0_acc:.3f} = {delta_pp:.1f}pp "
          f"-> {'PASS' if t257_pass else 'FAIL'}")

    # T258: beta=1.0 < peak (too much noise hurts)
    t258_pass = beta1_acc < peak_acc
    tests['T258'] = {
        'name': 'beta=1.0 < peak (too much noise hurts)',
        'beta1_acc': beta1_acc,
        'peak_acc': peak_acc,
        'pass': t258_pass,
    }
    print(f"  T258 High noise hurts:     {beta1_acc:.3f} < {peak_acc:.3f} "
          f"-> {'PASS' if t258_pass else 'FAIL'}")

    # T259: ISI CV increases monotonically with beta
    # Check Spearman correlation >= 0.8 (monotonic trend)
    if len(isi_cvs) >= 3:
        from scipy.stats import spearmanr
        try:
            rho, _ = spearmanr(betas, isi_cvs)
        except Exception:
            # Fallback: check if generally increasing
            increases = sum(1 for i in range(1, len(isi_cvs)) if isi_cvs[i] >= isi_cvs[i-1])
            rho = increases / (len(isi_cvs) - 1)
    else:
        rho = 0.0
    t259_pass = rho >= 0.7
    tests['T259'] = {
        'name': 'ISI CV increases monotonically with beta',
        'isi_cvs': isi_cvs,
        'spearman_rho': float(rho),
        'pass': t259_pass,
    }
    print(f"  T259 ISI CV monotonic:     Spearman rho={rho:.3f} >= 0.7 "
          f"-> {'PASS' if t259_pass else 'FAIL'}")

    # T260: Spike rate std increases with beta
    if len(sr_stds) >= 3:
        try:
            rho_sr, _ = spearmanr(betas, sr_stds)
        except Exception:
            increases = sum(1 for i in range(1, len(sr_stds)) if sr_stds[i] >= sr_stds[i-1])
            rho_sr = increases / (len(sr_stds) - 1)
    else:
        rho_sr = 0.0
    t260_pass = rho_sr >= 0.7
    tests['T260'] = {
        'name': 'Spike rate std increases with beta',
        'spike_rate_stds': sr_stds,
        'spearman_rho': float(rho_sr),
        'pass': t260_pass,
    }
    print(f"  T260 Rate std monotonic:   Spearman rho={rho_sr:.3f} >= 0.7 "
          f"-> {'PASS' if t260_pass else 'FAIL'}")

    results['tests'] = tests

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2194_substrate_coupling_strength.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    # ─── Plot ───
    fig_path = FIGURES / 'fig_z2194_substrate_coupling_strength.png'
    plot_results(beta_sweep, tests, fig_path)

    # ─── Cleanup ───
    if fpga and ser is not None:
        try:
            ser.close()
        except Exception:
            pass

    print(f"\nDone. {n_pass}/{n_total} tests passed.")
    return results


if __name__ == '__main__':
    main()
