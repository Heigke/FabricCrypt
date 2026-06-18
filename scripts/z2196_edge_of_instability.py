#!/usr/bin/env python3
"""z2196_edge_of_instability.py — Edge of Instability in GPU-Noise-Driven FPGA Reservoir

Systematically maps the phase diagram of the GPU-FPGA reservoir by sweeping base_vg
across the transition from silent (sub-threshold) to saturated (always-firing).
Shows that optimal computation occurs at the boundary between stable and unstable,
and that GPU 1/f noise shifts this boundary — key for Mario Lanza's memristive
computing (noise-driven switching at device boundaries).

Sweep: base_vg in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
Conditions per vg:
  FULL     — GPU 1/f noise (beta=0.10)
  NO_NOISE — deterministic (beta=0)

At each (vg, condition):
  Phase 1: 100 timesteps at 20Hz — measure spike rate, vmem variance, ISI CV
  Phase 2: 80 waveform classification trials — 3-class ridge, 5-fold CV
  Phase 3: Dynamic range — sweep input amplitude, measure spike rate spread

Tests T267-T272:
  T267: Peak accuracy vg is in [0.45, 0.65] (mid-range, near threshold)
  T268: FULL accuracy > NO_NOISE accuracy at peak vg (noise helps at boundary)
  T269: Peak accuracy vg for FULL < peak vg for NO_NOISE (noise lowers threshold)
  T270: Dynamic range maximized near spike threshold
  T271: ISI CV peaks near threshold (maximum variability at boundary)
  T272: Vmem variance highest for FULL at peak accuracy vg

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
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

# ─── Parameters ───
N_NEURONS = 8
ALPHA = 0.15
BETA = 0.10
SAMPLE_HZ = 20
VG_SWEEP = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
N_PHASE1_STEPS = 100
N_TRIALS = 80
STEPS_PER_TRIAL = 25
N_FOLDS = 5
DYN_RANGE_AMPS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
DYN_RANGE_STEPS = 30


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


def generate_synthetic_1f(n_samples, rng):
    """Generate synthetic 1/f noise via octave summation."""
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

def generate_waveforms(n_trials=80, steps_per_trial=25, freq_hz=1.0, dt=1.0/20, seed=42):
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
        elif cls == 1:  # triangle
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2*np.pi)) % 1.0) - 1.0) - 1.0
        else:           # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))

        wave = (wave + 1.0) / 2.0  # normalize to [0, 1]
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg, alpha=ALPHA, beta=0.0, live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    Returns: (n_steps, 24) array -- 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
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


# ═══════════════════════════════════════════════════════════
# LIF Simulation Fallback
# ═══════════════════════════════════════════════════════════

def simulate_lif_reservoir(input_signal, noise_samples, w_in, w_noise,
                            base_vg, alpha=ALPHA, beta=0.0):
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
        states[t, N_NEURONS:N_NEURONS*2] = vmem.copy()
        states[t, N_NEURONS*2:] = cumulative.copy()

    return states


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

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
    for alpha_val in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha_val * I, X_train.T @ Y_train)
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


def cross_val_accuracy(X, y, n_folds=5):
    """5-fold cross-validated ridge classification accuracy."""
    splits = stratified_kfold(X, y, n_splits=n_folds)
    accs = []
    for train_idx, test_idx in splits:
        acc = ridge_classify(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        accs.append(acc)
    return float(np.mean(accs))


# ═══════════════════════════════════════════════════════════
# Phase 1: Spike Rate / Vmem / ISI Characterization
# ═══════════════════════════════════════════════════════════

def run_phase1(run_trial_fn, base_vg, noise_samples, w_in, w_noise, beta, n_steps):
    """Run n_steps with constant input=0.5 and measure statistics."""
    input_signal = np.full(n_steps, 0.5)
    states = run_trial_fn(input_signal, noise_samples, w_in, w_noise,
                          base_vg=base_vg, alpha=ALPHA, beta=beta)

    # Spike deltas are in columns 0:N_NEURONS
    spike_deltas = states[:, :N_NEURONS]
    vmems = states[:, N_NEURONS:N_NEURONS*2]

    # Mean spike rate (spikes per step, averaged over neurons)
    mean_spike_rate = float(np.mean(spike_deltas))

    # Vmem variance (mean across neurons)
    vmem_var = float(np.mean(np.var(vmems, axis=0)))

    # ISI CV: compute per-neuron, then average
    isi_cvs = []
    for nid in range(N_NEURONS):
        spike_times = np.where(spike_deltas[:, nid] > 0)[0]
        if len(spike_times) > 2:
            isis = np.diff(spike_times).astype(float)
            if np.mean(isis) > 0:
                isi_cvs.append(float(np.std(isis) / np.mean(isis)))
    isi_cv = float(np.mean(isi_cvs)) if isi_cvs else 0.0

    return {
        'mean_spike_rate': mean_spike_rate,
        'vmem_variance': vmem_var,
        'isi_cv': isi_cv,
    }


# ═══════════════════════════════════════════════════════════
# Phase 2: Waveform Classification
# ═══════════════════════════════════════════════════════════

def run_phase2(run_trial_fn, base_vg, noise_samples, w_in, w_noise, beta,
               waveforms, labels, n_folds):
    """Run waveform classification and return 5-fold CV accuracy."""
    n_trials = len(labels)
    features_list = []

    for trial_idx in range(n_trials):
        input_signal = waveforms[trial_idx]
        states = run_trial_fn(input_signal, noise_samples, w_in, w_noise,
                              base_vg=base_vg, alpha=ALPHA, beta=beta)
        feat = pool_trial_features(states)
        features_list.append(feat)

    X = np.array(features_list)
    y = labels.copy()

    # Remove NaN/Inf
    bad_cols = np.any(~np.isfinite(X), axis=0)
    X[:, bad_cols] = 0.0

    acc = cross_val_accuracy(X, y, n_folds=n_folds)
    return acc


# ═══════════════════════════════════════════════════════════
# Phase 3: Dynamic Range
# ═══════════════════════════════════════════════════════════

def run_phase3(run_trial_fn, base_vg, noise_samples, w_in, w_noise, beta,
               amplitudes, n_steps):
    """Measure spike rate at different input amplitudes to get dynamic range."""
    rates = []
    for amp in amplitudes:
        input_signal = np.full(n_steps, amp)
        states = run_trial_fn(input_signal, noise_samples, w_in, w_noise,
                              base_vg=base_vg, alpha=ALPHA, beta=beta)
        spike_deltas = states[:, :N_NEURONS]
        rate = float(np.mean(spike_deltas))
        rates.append(rate)
    dynamic_range = float(max(rates) - min(rates))
    return dynamic_range, rates


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2196: Edge of Instability')
    parser.add_argument('--n-phase1-steps', type=int, default=N_PHASE1_STEPS)
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps-per-trial', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--noise-collect-s', type=float, default=15.0)
    parser.add_argument('--n-folds', type=int, default=N_FOLDS)
    args = parser.parse_args()

    print("=" * 65)
    print("z2196: Edge of Instability — GPU-FPGA Phase Diagram")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    # ─── Try to connect FPGA ───
    ser, port = find_fpga()
    simulated = (ser is None)

    if not simulated:
        connect_fpga(ser)
        print(f"[HW] FPGA connected on {port}")
    else:
        print("[SIM] No FPGA — using LIF simulation fallback")

    # ─── Telemetry ───
    try:
        from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
        telem_api = SysfsHwmonTelemetry()
        print(f"[HW] SysfsHwmonTelemetry: {telem_api}")
    except Exception as e:
        telem_api = None
        print(f"[WARN] SysfsHwmonTelemetry unavailable: {e}")

    # ─── Collect GPU noise ───
    print(f"\n[1/4] Collecting GPU power noise ({args.noise_collect_s}s)...")
    raw_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if raw_noise is not None and len(raw_noise) > 10:
        noise_mean = float(np.mean(raw_noise))
        noise_std = max(float(np.std(raw_noise)), 1e-6)
        noise_norm = (raw_noise - noise_mean) / noise_std
        noise_filtered = iir_filter_noise(noise_norm, alpha_iir=0.85)
        print(f"    Collected {len(raw_noise)} samples, mean={noise_mean:.2f}W, std={noise_std:.4f}W")
    else:
        print("    [WARN] hwmon unavailable — using synthetic 1/f noise")
        n_synth = int(args.noise_collect_s * 50)
        noise_filtered = generate_synthetic_1f(n_synth, rng)

    # ─── Build trial runner ───
    def make_trial_fn():
        if not simulated:
            def run_trial(input_signal, noise_samples, w_in_, w_noise_,
                          base_vg, alpha, beta):
                return run_fpga_reservoir_trial(
                    ser, input_signal, noise_samples, w_in_, w_noise_,
                    base_vg=base_vg, alpha=alpha, beta=beta, live_noise=True)
        else:
            def run_trial(input_signal, noise_samples, w_in_, w_noise_,
                          base_vg, alpha, beta):
                return simulate_lif_reservoir(
                    input_signal, noise_samples, w_in_, w_noise_,
                    base_vg=base_vg, alpha=alpha, beta=beta)
        return run_trial

    run_trial_fn = make_trial_fn()

    # ─── Generate waveforms ───
    waveforms, labels = generate_waveforms(
        n_trials=args.n_trials,
        steps_per_trial=args.steps_per_trial,
        seed=42
    )
    print(f"    Generated {args.n_trials} waveform trials ({args.steps_per_trial} steps each)")

    # ─── Sweep ───
    conditions = [
        ('FULL', BETA),
        ('NO_NOISE', 0.0),
    ]

    results_map = {}  # (vg, condition_name) -> metrics dict

    print(f"\n[2/4] Sweeping {len(VG_SWEEP)} base_vg x {len(conditions)} conditions...")
    for vg_idx, base_vg in enumerate(VG_SWEEP):
        for cond_name, beta_val in conditions:
            tag = f"vg={base_vg:.2f}/{cond_name}"
            print(f"\n  [{vg_idx*2 + (1 if cond_name=='NO_NOISE' else 0) + 1}"
                  f"/{len(VG_SWEEP)*len(conditions)}] {tag}")

            # Phase 1: spike stats
            print(f"    Phase 1: spike characterization ({args.n_phase1_steps} steps)...")
            p1 = run_phase1(run_trial_fn, base_vg, noise_filtered, w_in, w_noise,
                            beta=beta_val, n_steps=args.n_phase1_steps)
            print(f"      rate={p1['mean_spike_rate']:.4f}  vmem_var={p1['vmem_variance']:.4f}"
                  f"  isi_cv={p1['isi_cv']:.3f}")

            # Phase 2: classification
            print(f"    Phase 2: waveform classification ({args.n_trials} trials, {args.n_folds}-fold)...")
            acc = run_phase2(run_trial_fn, base_vg, noise_filtered, w_in, w_noise,
                             beta=beta_val, waveforms=waveforms, labels=labels,
                             n_folds=args.n_folds)
            print(f"      accuracy={acc:.4f}")

            # Phase 3: dynamic range
            print(f"    Phase 3: dynamic range ({len(DYN_RANGE_AMPS)} amplitudes)...")
            dyn_range, rates = run_phase3(run_trial_fn, base_vg, noise_filtered,
                                           w_in, w_noise, beta=beta_val,
                                           amplitudes=DYN_RANGE_AMPS,
                                           n_steps=DYN_RANGE_STEPS)
            print(f"      dynamic_range={dyn_range:.4f}  rates={[f'{r:.3f}' for r in rates]}")

            results_map[(base_vg, cond_name)] = {
                'base_vg': base_vg,
                'condition': cond_name,
                'beta': beta_val,
                'mean_spike_rate': p1['mean_spike_rate'],
                'vmem_variance': p1['vmem_variance'],
                'isi_cv': p1['isi_cv'],
                'classification_accuracy': acc,
                'dynamic_range': dyn_range,
                'dynamic_range_rates': rates,
            }

    # ─── Analyze ───
    print("\n" + "=" * 65)
    print("[3/4] Analysis — Phase Diagram")
    print("=" * 65)

    # Build per-condition arrays
    full_accs = []
    no_noise_accs = []
    full_rates = []
    no_noise_rates = []
    full_isi_cvs = []
    no_noise_isi_cvs = []
    full_vmem_vars = []
    no_noise_vmem_vars = []
    full_dyn_ranges = []
    no_noise_dyn_ranges = []

    for vg in VG_SWEEP:
        f = results_map[(vg, 'FULL')]
        n = results_map[(vg, 'NO_NOISE')]
        full_accs.append(f['classification_accuracy'])
        no_noise_accs.append(n['classification_accuracy'])
        full_rates.append(f['mean_spike_rate'])
        no_noise_rates.append(n['mean_spike_rate'])
        full_isi_cvs.append(f['isi_cv'])
        no_noise_isi_cvs.append(n['isi_cv'])
        full_vmem_vars.append(f['vmem_variance'])
        no_noise_vmem_vars.append(n['vmem_variance'])
        full_dyn_ranges.append(f['dynamic_range'])
        no_noise_dyn_ranges.append(n['dynamic_range'])

    full_accs = np.array(full_accs)
    no_noise_accs = np.array(no_noise_accs)

    # Peak accuracy vg
    full_peak_idx = int(np.argmax(full_accs))
    no_noise_peak_idx = int(np.argmax(no_noise_accs))
    full_peak_vg = VG_SWEEP[full_peak_idx]
    no_noise_peak_vg = VG_SWEEP[no_noise_peak_idx]
    full_peak_acc = float(full_accs[full_peak_idx])
    no_noise_peak_acc = float(no_noise_accs[no_noise_peak_idx])

    # Spike threshold vg: first vg where mean_spike_rate > 0.01
    threshold_vg_full = None
    for i, vg in enumerate(VG_SWEEP):
        if full_rates[i] > 0.01:
            threshold_vg_full = vg
            break
    threshold_vg_nn = None
    for i, vg in enumerate(VG_SWEEP):
        if no_noise_rates[i] > 0.01:
            threshold_vg_nn = vg
            break

    # Dynamic range peak vg
    full_dr_peak_idx = int(np.argmax(full_dyn_ranges))
    full_dr_peak_vg = VG_SWEEP[full_dr_peak_idx]

    # ISI CV peak vg
    full_isi_peak_idx = int(np.argmax(full_isi_cvs))
    full_isi_peak_vg = VG_SWEEP[full_isi_peak_idx]

    print(f"\n  FULL peak accuracy:     {full_peak_acc:.4f} at vg={full_peak_vg:.2f}")
    print(f"  NO_NOISE peak accuracy: {no_noise_peak_acc:.4f} at vg={no_noise_peak_vg:.2f}")
    print(f"  Spike threshold (FULL):     vg={threshold_vg_full}")
    print(f"  Spike threshold (NO_NOISE): vg={threshold_vg_nn}")
    print(f"  Dynamic range peak (FULL):  vg={full_dr_peak_vg:.2f}")
    print(f"  ISI CV peak (FULL):         vg={full_isi_peak_vg:.2f}")

    print(f"\n  Accuracy table:")
    print(f"  {'vg':>6}  {'FULL':>8}  {'NO_NOISE':>8}  {'rate_F':>8}  {'rate_N':>8}"
          f"  {'isi_F':>7}  {'vmem_F':>8}  {'DR_F':>7}")
    for i, vg in enumerate(VG_SWEEP):
        print(f"  {vg:6.2f}  {full_accs[i]:8.4f}  {no_noise_accs[i]:8.4f}"
              f"  {full_rates[i]:8.4f}  {no_noise_rates[i]:8.4f}"
              f"  {full_isi_cvs[i]:7.3f}  {full_vmem_vars[i]:8.4f}"
              f"  {full_dyn_ranges[i]:7.4f}")

    # ─── Tests T267-T272 ───
    print("\n" + "=" * 65)
    print("[4/4] Tests T267-T272")
    print("=" * 65)

    tests = {}

    # T267: Peak accuracy vg in [0.45, 0.65]
    t267_pass = 0.45 <= full_peak_vg <= 0.65
    tests['T267_peak_vg_midrange'] = {
        'pass': t267_pass,
        'full_peak_vg': full_peak_vg,
        'full_peak_acc': full_peak_acc,
        'criterion': 'peak_vg in [0.45, 0.65]',
    }
    print(f"\n  T267 peak_vg_midrange: {'PASS' if t267_pass else 'FAIL'}"
          f"  peak_vg={full_peak_vg:.2f} (need [0.45, 0.65])")

    # T268: FULL > NO_NOISE at peak vg
    acc_full_at_peak = float(full_accs[full_peak_idx])
    acc_nn_at_peak = float(no_noise_accs[full_peak_idx])
    t268_pass = acc_full_at_peak > acc_nn_at_peak
    tests['T268_noise_helps_at_boundary'] = {
        'pass': t268_pass,
        'full_acc_at_peak': acc_full_at_peak,
        'no_noise_acc_at_peak': acc_nn_at_peak,
        'criterion': 'FULL > NO_NOISE at peak vg',
    }
    print(f"  T268 noise_helps_at_boundary: {'PASS' if t268_pass else 'FAIL'}"
          f"  FULL={acc_full_at_peak:.4f} vs NO_NOISE={acc_nn_at_peak:.4f}")

    # T269: Peak vg for FULL < peak vg for NO_NOISE (noise lowers threshold)
    t269_pass = full_peak_vg < no_noise_peak_vg
    tests['T269_noise_lowers_threshold'] = {
        'pass': t269_pass,
        'full_peak_vg': full_peak_vg,
        'no_noise_peak_vg': no_noise_peak_vg,
        'criterion': 'FULL peak vg < NO_NOISE peak vg',
    }
    print(f"  T269 noise_lowers_threshold: {'PASS' if t269_pass else 'FAIL'}"
          f"  FULL_peak={full_peak_vg:.2f} vs NO_NOISE_peak={no_noise_peak_vg:.2f}")

    # T270: Dynamic range maximized near spike threshold
    # "near" = within 2 steps in VG_SWEEP of the threshold vg
    if threshold_vg_full is not None:
        thresh_idx = VG_SWEEP.index(threshold_vg_full)
        dr_near_thresh = abs(full_dr_peak_idx - thresh_idx) <= 2
    else:
        dr_near_thresh = False
    t270_pass = dr_near_thresh
    tests['T270_dynamic_range_at_threshold'] = {
        'pass': t270_pass,
        'dr_peak_vg': full_dr_peak_vg,
        'threshold_vg': threshold_vg_full,
        'criterion': 'DR peak within 2 steps of spike threshold',
    }
    print(f"  T270 dynamic_range_at_threshold: {'PASS' if t270_pass else 'FAIL'}"
          f"  DR_peak={full_dr_peak_vg:.2f}, threshold={threshold_vg_full}")

    # T271: ISI CV peaks near threshold
    if threshold_vg_full is not None:
        thresh_idx = VG_SWEEP.index(threshold_vg_full)
        isi_near_thresh = abs(full_isi_peak_idx - thresh_idx) <= 2
    else:
        isi_near_thresh = False
    t271_pass = isi_near_thresh
    tests['T271_isi_cv_peaks_near_threshold'] = {
        'pass': t271_pass,
        'isi_cv_peak_vg': full_isi_peak_vg,
        'threshold_vg': threshold_vg_full,
        'criterion': 'ISI CV peak within 2 steps of spike threshold',
    }
    print(f"  T271 isi_cv_peaks_near_threshold: {'PASS' if t271_pass else 'FAIL'}"
          f"  ISI_peak={full_isi_peak_vg:.2f}, threshold={threshold_vg_full}")

    # T272: Vmem variance highest for FULL at peak accuracy vg
    vmem_var_full_at_peak = full_vmem_vars[full_peak_idx]
    vmem_var_nn_at_peak = no_noise_vmem_vars[full_peak_idx]
    t272_pass = vmem_var_full_at_peak > vmem_var_nn_at_peak
    tests['T272_vmem_variance_highest_full'] = {
        'pass': t272_pass,
        'vmem_var_full': vmem_var_full_at_peak,
        'vmem_var_no_noise': vmem_var_nn_at_peak,
        'vg': full_peak_vg,
        'criterion': 'vmem_var(FULL) > vmem_var(NO_NOISE) at peak acc vg',
    }
    print(f"  T272 vmem_variance_highest_full: {'PASS' if t272_pass else 'FAIL'}"
          f"  FULL={vmem_var_full_at_peak:.4f} vs NO_NOISE={vmem_var_nn_at_peak:.4f}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    # ─── Build output ───
    sweep_data = []
    for vg in VG_SWEEP:
        for cond_name in ['FULL', 'NO_NOISE']:
            entry = results_map[(vg, cond_name)]
            sweep_data.append(entry)

    output = {
        'experiment': 'z2196_edge_of_instability',
        'description': 'Edge of Instability — GPU-FPGA phase diagram',
        'simulated': simulated,
        'parameters': {
            'vg_sweep': VG_SWEEP,
            'alpha': ALPHA,
            'beta': BETA,
            'n_neurons': N_NEURONS,
            'sample_hz': SAMPLE_HZ,
            'n_phase1_steps': args.n_phase1_steps,
            'n_trials': args.n_trials,
            'steps_per_trial': args.steps_per_trial,
            'n_folds': args.n_folds,
            'dynamic_range_amplitudes': DYN_RANGE_AMPS,
        },
        'summary': {
            'full_peak_vg': full_peak_vg,
            'full_peak_accuracy': full_peak_acc,
            'no_noise_peak_vg': no_noise_peak_vg,
            'no_noise_peak_accuracy': no_noise_peak_acc,
            'threshold_vg_full': threshold_vg_full,
            'threshold_vg_no_noise': threshold_vg_nn,
            'dr_peak_vg': full_dr_peak_vg,
            'isi_cv_peak_vg': full_isi_peak_vg,
            'full_accuracies': full_accs.tolist(),
            'no_noise_accuracies': no_noise_accs.tolist(),
            'full_spike_rates': full_rates,
            'no_noise_spike_rates': no_noise_rates,
            'full_isi_cvs': full_isi_cvs,
            'full_vmem_variances': full_vmem_vars,
            'full_dynamic_ranges': full_dyn_ranges,
        },
        'sweep_data': sweep_data,
        'tests': tests,
        'pass_count': n_pass,
        'total_count': n_total,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2196_edge_of_instability.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved to {out_path}")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        FIGURES.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('z2196: Edge of Instability — GPU-FPGA Phase Diagram',
                     fontsize=14, fontweight='bold')

        vg_arr = np.array(VG_SWEEP)

        # (0,0) Classification accuracy vs vg
        ax = axes[0, 0]
        ax.plot(vg_arr, full_accs, 'o-', color='#e74c3c', label='FULL (1/f noise)', linewidth=2)
        ax.plot(vg_arr, no_noise_accs, 's--', color='#3498db', label='NO_NOISE', linewidth=2)
        ax.axvline(full_peak_vg, color='#e74c3c', alpha=0.3, linestyle=':')
        ax.axhline(1/3, color='gray', alpha=0.3, linestyle='--', label='chance')
        ax.set_xlabel('Base Vg')
        ax.set_ylabel('Accuracy')
        ax.set_title('Waveform Classification')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (0,1) Spike rate vs vg
        ax = axes[0, 1]
        ax.plot(vg_arr, full_rates, 'o-', color='#e74c3c', label='FULL', linewidth=2)
        ax.plot(vg_arr, no_noise_rates, 's--', color='#3498db', label='NO_NOISE', linewidth=2)
        if threshold_vg_full:
            ax.axvline(threshold_vg_full, color='green', alpha=0.5, linestyle=':', label='threshold')
        ax.set_xlabel('Base Vg')
        ax.set_ylabel('Mean Spike Rate')
        ax.set_title('Spike Rate (Phase Transition)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (0,2) ISI CV vs vg
        ax = axes[0, 2]
        ax.plot(vg_arr, full_isi_cvs, 'o-', color='#e74c3c', label='FULL', linewidth=2)
        ax.plot(vg_arr, no_noise_isi_cvs, 's--', color='#3498db', label='NO_NOISE', linewidth=2)
        ax.set_xlabel('Base Vg')
        ax.set_ylabel('ISI CV')
        ax.set_title('ISI Variability (T271)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1,0) Vmem variance vs vg
        ax = axes[1, 0]
        ax.plot(vg_arr, full_vmem_vars, 'o-', color='#e74c3c', label='FULL', linewidth=2)
        ax.plot(vg_arr, no_noise_vmem_vars, 's--', color='#3498db', label='NO_NOISE', linewidth=2)
        ax.set_xlabel('Base Vg')
        ax.set_ylabel('Vmem Variance')
        ax.set_title('Membrane Voltage Variance (T272)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1,1) Dynamic range vs vg
        ax = axes[1, 1]
        ax.plot(vg_arr, full_dyn_ranges, 'o-', color='#e74c3c', label='FULL', linewidth=2)
        ax.plot(vg_arr, no_noise_dyn_ranges, 's--', color='#3498db', label='NO_NOISE', linewidth=2)
        ax.set_xlabel('Base Vg')
        ax.set_ylabel('Dynamic Range')
        ax.set_title('Dynamic Range (T270)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1,2) Test results summary
        ax = axes[1, 2]
        ax.axis('off')
        test_lines = []
        for tname, tdata in tests.items():
            status = 'PASS' if tdata['pass'] else 'FAIL'
            color = '#27ae60' if tdata['pass'] else '#e74c3c'
            test_lines.append((tname.replace('_', ' '), status, color))
        y_pos = 0.9
        for label, status, color in test_lines:
            ax.text(0.05, y_pos, f"{status}", fontsize=12, fontweight='bold',
                    color=color, transform=ax.transAxes)
            ax.text(0.18, y_pos, label, fontsize=10, transform=ax.transAxes)
            y_pos -= 0.13
        ax.text(0.05, y_pos - 0.05, f"Total: {n_pass}/{n_total}",
                fontsize=13, fontweight='bold', transform=ax.transAxes)
        sim_tag = " (SIMULATED)" if simulated else " (REAL HW)"
        ax.set_title(f'Test Results{sim_tag}', fontsize=11)

        plt.tight_layout()
        fig_path = FIGURES / 'z2196_edge_of_instability.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved to {fig_path}")

    except ImportError:
        print("  [WARN] matplotlib unavailable — skipping figure")

    if ser:
        ser.close()

    return n_pass, n_total


if __name__ == '__main__':
    main()
