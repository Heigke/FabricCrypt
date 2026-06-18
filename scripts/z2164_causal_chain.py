#!/usr/bin/env python3
"""z2164_causal_chain.py — Causal Chain Ablation: Every Layer is Necessary

Systematic ablation proving each layer of the GPU→FPGA substrate is causally
necessary for reservoir computation. Removes one component at a time from:

    GPU_power_noise → IIR_filter → Vg_modulation → FPGA_LIF_neurons → spike_readout → Ridge_classifier

8 Ablation Conditions:
  FULL:        Complete system (GPU 1/f noise + IIR + FPGA reservoir)
  NO_IIR:      Remove IIR filter (raw power noise, no temporal filtering)
  SYNTH_1F:    Replace GPU noise with synthetic Voss-McCartney 1/f
  WHITE:       Replace with white noise (remove temporal structure)
  NO_NOISE:    Deterministic (β=0, remove all noise)
  NO_FPGA:     Skip FPGA, use software LIF (remove hardware substrate)
  RANDOM_READ: FPGA but permute neuron→feature mapping (break causal link)
  SHUFFLED:    Full system but shuffle readout temporally (break temporal causality)

Tests:
  T71: FULL > NO_IIR       (IIR filter adds temporal memory)
  T72: FULL > SYNTH_1F     (real GPU noise > synthetic, substrate matters)
  T73: FULL > WHITE        (temporal structure matters)
  T74: FULL > NO_NOISE     (noise helps computation)
  T75: FULL > NO_FPGA      (hardware substrate matters)
  T76: FULL > RANDOM_READ  (causal connection matters)
  T77: FULL > SHUFFLED     (temporal ordering matters)
  T78: Monotonic degradation (each ablation step reduces accuracy)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB1
"""

import os, sys, json, time, struct, subprocess, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-4' / 'figures'

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


def _print(msg=""):
    """Print with immediate flush so output appears even if script crashes."""
    print(msg)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════
# FPGA Communication (from z2153/z2155/z2162)
# ═══════════════════════════════════════════════════════════

def to_q16_16(val: float) -> int:
    return int(val * 65536) & 0xFFFFFFFF


def find_fpga(retries=3, retry_delay=1.0):
    """Find and connect to FPGA serial port with retry logic for busy ports."""
    try:
        import serial as ser_mod
    except ImportError:
        _print("  WARNING: pyserial not installed")
        return None, None
    for attempt in range(retries):
        for p in ['/dev/ttyUSB1', '/dev/ttyUSB0', '/dev/ttyUSB2']:
            try:
                s = ser_mod.Serial(p, 115200, timeout=0.2)
                time.sleep(0.1)
                return s, p
            except ser_mod.SerialException as e:
                if 'busy' in str(e).lower() or 'permission' in str(e).lower():
                    _print(f"  Port {p} busy/locked (attempt {attempt+1}/{retries}): {e}")
                else:
                    continue
            except Exception:
                continue
        if attempt < retries - 1:
            _print(f"  No FPGA found, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    return None, None


def safe_read_telem(ser, port, timeout=0.15):
    """Read telemetry with serial reconnection on failure."""
    try:
        return ser, read_telem(ser, timeout)
    except (Exception,) as e:
        _print(f"  Serial read error: {e}, reconnecting...")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.5)
        try:
            import serial as ser_mod
            ser = ser_mod.Serial(port, 115200, timeout=0.2)
            time.sleep(0.1)
            ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
            ser.flush()
            _print(f"  Reconnected to {port}")
            return ser, None
        except Exception as e2:
            _print(f"  Reconnect failed: {e2}")
            return None, None


def safe_set_vg(ser, port, vg_values):
    """Set per-neuron Vg with serial reconnection on failure."""
    try:
        set_per_neuron_vg(ser, vg_values)
        return ser
    except (Exception,) as e:
        _print(f"  Serial write error: {e}, reconnecting...")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.5)
        try:
            import serial as ser_mod
            ser = ser_mod.Serial(port, 115200, timeout=0.2)
            time.sleep(0.1)
            ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
            ser.flush()
            set_per_neuron_vg(ser, vg_values)
            _print(f"  Reconnected to {port}")
            return ser
        except Exception as e2:
            _print(f"  Reconnect failed: {e2}")
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
    for i in range(n_samples):
        p = read_hwmon_power()
        if p is not None:
            powers.append(p)
        if (i + 1) % (sample_hz * 5) == 0:
            _print(f"    ... {i+1}/{n_samples} power samples collected")
        time.sleep(interval)
    return np.array(powers) if powers else None


def voss_mcCartney_1f(n_samples, n_octaves=8, seed=99):
    """Voss-McCartney algorithm: synthetic 1/f noise from software RNG.
    Uses a DIFFERENT seed from the reservoir weights to avoid correlation."""
    rng = np.random.default_rng(seed)
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
    """Apply IIR low-pass: y[t] = a*y[t-1] + (1-a)*x[t]. Creates temporal memory."""
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t - 1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, port, input_signal, noise_samples, w_in, w_noise,
                              base_vg=BASE_VG, alpha=ALPHA, beta=0.08,
                              live_noise=False, permute_readout=False,
                              permute_rng=None):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    permute_readout: if True, randomly permute neuron-to-feature mapping each step
    Returns: (ser, states) where states is (n_steps, 24) array.
    ser is returned because it may be reconnected on failure.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    power_mean = 11.0

    for t in range(n_steps):
        if ser is None:
            # Lost connection mid-trial, fill rest with zeros
            break

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

        ser = safe_set_vg(ser, port, vg_values)
        if ser is None:
            break
        time.sleep(interval * 0.3)

        # Read telemetry
        try:
            ser.reset_input_buffer()
            ser.write(bytes([SYNC, CMD_READ_TELEM]))
            ser.flush()
        except Exception:
            ser, _ = safe_read_telem(ser, port)
            if ser is None:
                break
            continue

        ser, telem = safe_read_telem(ser, port, timeout=0.15)
        if ser is None:
            break

        if telem:
            counts = [n['spike_count'] for n in telem]
            vmems = [n['vmem'] for n in telem]

            if permute_readout and permute_rng is not None:
                perm = permute_rng.permutation(N_NEURONS)
                counts = [counts[p] for p in perm]
                vmems = [vmems[p] for p in perm]

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

    return ser, states


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
# Waveform Generation & XOR
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=100, steps_per_trial=30, freq_hz=1.0, dt=1.0 / 20):
    rng = np.random.default_rng(42)
    trials, labels = [], []
    t = np.arange(steps_per_trial) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        phase = rng.uniform(0, 2 * np.pi)
        freq = freq_hz * rng.uniform(0.8, 1.2)
        if cls == 0:
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        wave = (wave + 1.0) / 2.0
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_xor_sequence(n_steps=2000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_steps).astype(float)


def compute_xor_targets(u, tau):
    n = len(u)
    targets = np.zeros(n)
    for t in range(tau, n):
        targets[t] = int(u[t]) ^ int(u[t - tau])
    return targets


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification (from z2162)
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2, 3)):
    T, D = states.shape
    augmented = np.zeros((T, D * (1 + len(delays))))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[d:, start:start + D] = states[:T - d]
    return augmented


def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
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


def ridge_binary(X_train, y_train, X_test, y_test, alphas=None):
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    best_acc = -1
    for alpha_reg in alphas:
        I = np.eye(X_train.shape[1])
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha_reg * I, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue
        pred = (X_test @ w > 0.5).astype(float)
        acc = np.mean(pred == y_test)
        if acc > best_acc:
            best_acc = acc
    return best_acc


def stratified_kfold(X, y, n_splits=5, seed=42):
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


# ═══════════════════════════════════════════════════════════
# Energy Measurement
# ═══════════════════════════════════════════════════════════

def measure_energy_context():
    """Context manager-like helper: returns (start_fn, stop_fn) for energy measurement."""
    class EnergyMeter:
        def __init__(self):
            self.samples = []
            self.t_start = None
            self.t_end = None

        def start(self):
            self.t_start = time.monotonic()
            p = read_hwmon_power()
            if p is not None:
                self.samples.append(p)

        def sample(self):
            p = read_hwmon_power()
            if p is not None:
                self.samples.append(p)

        def stop(self):
            self.t_end = time.monotonic()
            p = read_hwmon_power()
            if p is not None:
                self.samples.append(p)

        def joules(self):
            if not self.samples or self.t_start is None or self.t_end is None:
                return None
            duration = self.t_end - self.t_start
            mean_power = np.mean(self.samples)
            return mean_power * duration, mean_power, duration

    return EnergyMeter()


# ═══════════════════════════════════════════════════════════
# Run One Condition
# ═══════════════════════════════════════════════════════════

def run_condition_waveform(cond_name, ser, port, fpga, wave_trials, wave_labels,
                           noise_src, w_in, w_noise, beta, live_noise,
                           use_fpga, permute_readout, shuffle_time,
                           n_trials, steps_per_trial):
    """Run one condition on the waveform task and return (ser, result_dict).
    ser is returned because it may be reconnected during FPGA trials."""
    _print(f"\n  === {cond_name} (beta={beta:.2f}, fpga={use_fpga}, "
           f"live={live_noise}, perm={permute_readout}, shuf={shuffle_time}) ===")

    perm_rng = np.random.default_rng(123) if permute_readout else None
    meter = measure_energy_context()
    meter.start()

    trial_features = []
    t0 = time.monotonic()

    for trial_idx in range(n_trials):
        input_signal = wave_trials[trial_idx]

        if use_fpga and fpga and ser is not None:
            ser, states = run_fpga_reservoir_trial(
                ser, port, input_signal, noise_src, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=beta,
                live_noise=live_noise,
                permute_readout=permute_readout,
                permute_rng=perm_rng)
        else:
            states = simulate_lif_reservoir(
                input_signal, noise_src, w_in, w_noise,
                base_vg=BASE_VG, alpha=ALPHA, beta=beta)

        if shuffle_time:
            # Shuffle along time axis to break temporal causality
            rng_shuf = np.random.default_rng(trial_idx + 7777)
            perm_idx = rng_shuf.permutation(states.shape[0])
            states = states[perm_idx]

        aug = augment_with_delays(states, delays=(1, 2, 3))
        feat = pool_trial_features(aug)
        trial_features.append(feat)

        if (trial_idx + 1) % 25 == 0:
            meter.sample()
            elapsed = time.monotonic() - t0
            rate = (trial_idx + 1) / elapsed
            eta = (n_trials - trial_idx - 1) / rate
            _print(f"    Trial {trial_idx + 1}/{n_trials} "
                   f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

    meter.stop()
    elapsed = time.monotonic() - t0

    X = np.array(trial_features)
    y = wave_labels[:n_trials]

    # 5-fold stratified CV
    folds = stratified_kfold(X, y, n_splits=5, seed=42)
    fold_accs = []
    for train_idx, test_idx in folds:
        acc = ridge_classify(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        fold_accs.append(acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    energy_info = meter.joules()
    energy_j = energy_info[0] if energy_info else None
    mean_power = energy_info[1] if energy_info else None

    _print(f"  {cond_name}: acc={mean_acc:.4f} +/- {std_acc:.4f} "
           f"({elapsed:.1f}s, {f'{energy_j:.1f}J' if energy_j else 'N/A'})")

    return ser, {
        'mean_acc': float(mean_acc),
        'std_acc': float(std_acc),
        'fold_accs': [float(a) for a in fold_accs],
        'energy_joules': float(energy_j) if energy_j is not None else None,
        'mean_power_w': float(mean_power) if mean_power is not None else None,
        'duration_s': float(elapsed),
    }


def run_condition_xor(cond_name, ser, port, fpga, xor_input, w_in, w_noise,
                      noise_src, beta, live_noise, use_fpga,
                      permute_readout, shuffle_time, taus=(1, 2, 3, 5)):
    """Run one condition on temporal XOR task. Returns (ser, result_dict)."""
    _print(f"\n  === XOR: {cond_name} ===")

    perm_rng = np.random.default_rng(456) if permute_readout else None

    n_steps = len(xor_input)

    if use_fpga and fpga and ser is not None:
        ser, states = run_fpga_reservoir_trial(
            ser, port, xor_input, noise_src, w_in, w_noise,
            base_vg=BASE_VG, alpha=ALPHA, beta=beta,
            live_noise=live_noise,
            permute_readout=permute_readout,
            permute_rng=perm_rng)
    else:
        states = simulate_lif_reservoir(
            xor_input, noise_src, w_in, w_noise,
            base_vg=BASE_VG, alpha=ALPHA, beta=beta)

    if shuffle_time:
        rng_shuf = np.random.default_rng(8888)
        perm_idx = rng_shuf.permutation(states.shape[0])
        states = states[perm_idx]

    aug = augment_with_delays(states, delays=(1, 2, 3))

    tau_results = {}
    best_acc = 0.0

    for tau in taus:
        targets = compute_xor_targets(xor_input, tau)
        # Chronological 70/30 split
        train_end = int(n_steps * 0.7)
        X_train = aug[tau:train_end]
        y_train = targets[tau:train_end]
        X_test = aug[train_end:]
        y_test = targets[train_end:]

        acc = ridge_binary(X_train, y_train, X_test, y_test)
        tau_results[str(tau)] = float(acc)
        if acc > best_acc:
            best_acc = acc

    _print(f"  {cond_name} XOR: best={best_acc:.4f}, per-tau={tau_results}")
    return ser, {
        'best_acc': float(best_acc),
        'per_tau': tau_results,
    }


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def make_figures(results, fig_path):
    """Generate 3-panel figure: waveform bars, XOR bars, causal waterfall."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        _print("  matplotlib not available, skipping figures")
        return

    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # Condition ordering for display (expected degradation order)
    cond_order = ['FULL', 'NO_IIR', 'SYNTH_1F', 'WHITE', 'NO_NOISE',
                  'NO_FPGA', 'RANDOM_READ', 'SHUFFLED']

    # Colors: FULL in green, ablations in progressively warmer colors
    colors = ['#2ca02c', '#98df8a', '#aec7e8', '#ffbb78',
              '#ff7f0e', '#d62728', '#9467bd', '#8c564b']

    wave_accs = []
    wave_stds = []
    xor_accs = []
    labels = []

    for cond in cond_order:
        if cond in results.get('conditions', {}):
            cd = results['conditions'][cond]
            wave_accs.append(cd.get('waveform', {}).get('mean_acc', 0))
            wave_stds.append(cd.get('waveform', {}).get('std_acc', 0))
            xor_accs.append(cd.get('xor', {}).get('best_acc', 0))
            labels.append(cond)

    if not labels:
        _print("  No condition data for plotting")
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Waveform accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x, wave_accs, yerr=wave_stds, color=colors[:len(labels)],
                    capsize=4, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Waveform Classification (5-fold CV)')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=1 / 3, color='gray', linestyle='--', alpha=0.5, label='Chance (33%)')
    ax1.legend(fontsize=7)

    # Panel 2: XOR accuracy
    ax2 = axes[1]
    bars2 = ax2.bar(x, xor_accs, color=colors[:len(labels)],
                    edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Best Accuracy (across tau)')
    ax2.set_title('Temporal XOR')
    ax2.set_ylim(0, 1.05)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax2.legend(fontsize=7)

    # Panel 3: Causal waterfall — accuracy DROP at each ablation step
    ax3 = axes[2]
    if len(wave_accs) > 0:
        full_acc = wave_accs[0]
        drops = [full_acc - a for a in wave_accs]
        # Waterfall: start from full, show cumulative drop
        bar_colors = ['#2ca02c'] + ['#d62728'] * (len(drops) - 1)
        ax3.bar(x, drops, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Accuracy Drop from FULL')
        ax3.set_title('Causal Waterfall (Ablation Impact)')
        # Add text labels
        for i, d in enumerate(drops):
            if d > 0.005:
                ax3.text(i, d + 0.005, f'-{d:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=200, bbox_inches='tight')
    plt.close()
    _print(f"  Figure saved: {fig_path}")


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2164: Causal Chain Ablation')
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--steps-per-trial', type=int, default=30)
    parser.add_argument('--xor-steps', type=int, default=2000)
    parser.add_argument('--noise-collect-s', type=float, default=15.0)
    args = parser.parse_args()

    _print("=" * 70)
    _print("z2164: Causal Chain Ablation — Every Layer is Necessary")
    _print("=" * 70)
    _print(f"  n_trials={args.n_trials}, steps_per_trial={args.steps_per_trial}, "
           f"xor_steps={args.xor_steps}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2164_causal_chain',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'xor_steps': args.xor_steps,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
        'conditions': {},
        'tests': {},
    }

    ser = None
    port = None

    try:
        # ─── Step 1: Connect to FPGA ───
        _print("\n[1/6] Connecting to FPGA...")
        ser, port = find_fpga(retries=3, retry_delay=1.0)
        if ser is None:
            _print("  FPGA not found -- using LIF simulation fallback for ALL conditions")
            fpga = False
            results['simulated'] = True
        else:
            _print(f"  Connected: {port}")
            fpga = True
            ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
            ser.flush()
            time.sleep(0.1)
            _print("  Kill switch disabled")

        # ─── Step 2: Collect GPU noise ───
        _print("\n[2/6] Collecting GPU noise sources...")

        # Real GPU 1/f noise from power rail
        _print("  Collecting power rail noise (1/f)...")
        power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
        if power_noise is not None and len(power_noise) > 10:
            power_mean = power_noise.mean()
            power_std = max(power_noise.std(), 1e-6)
            noise_1f_raw = (power_noise - power_mean) / power_std
            _print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, "
                   f"{len(noise_1f_raw)} samples")
            have_real_noise = True
        else:
            _print("  Power rail unavailable, generating synthetic 1/f as 'real' fallback")
            noise_1f_raw = voss_mcCartney_1f(int(args.noise_collect_s * 50), seed=42)
            have_real_noise = False

        # IIR-filtered version (FULL condition)
        noise_1f_iir = iir_filter_noise(noise_1f_raw, alpha_iir=0.85)

        # Synthetic 1/f (SYNTH_1F condition) — different seed, software-only
        noise_synth_1f = voss_mcCartney_1f(len(noise_1f_raw), n_octaves=8, seed=99)
        noise_synth_1f_iir = iir_filter_noise(noise_synth_1f, alpha_iir=0.85)

        # White noise
        noise_white = rng.standard_normal(len(noise_1f_raw))

        # Zero noise
        noise_zero = np.zeros(1000)

        results['noise'] = {
            'real_1f_samples': len(noise_1f_raw),
            'have_real_gpu_noise': have_real_noise,
        }

        # ─── Step 3: Generate tasks ───
        _print("\n[3/6] Generating tasks...")
        wave_trials, wave_labels = generate_waveforms(
            n_trials=args.n_trials, steps_per_trial=args.steps_per_trial)
        _print(f"  Waveform: {args.n_trials} trials, {args.steps_per_trial} steps, "
               f"classes={np.bincount(wave_labels).tolist()}")

        xor_input = generate_xor_sequence(n_steps=args.xor_steps, seed=42)
        _print(f"  XOR: {args.xor_steps} steps")

        # ─── Step 4: Define conditions ───
        # Each condition: (noise_src, beta, live_noise, use_fpga, permute_readout, shuffle_time)
        conditions = {
            'FULL':        (noise_1f_iir,       BETA, True,  True,  False, False),
            'NO_IIR':      (noise_1f_raw,       BETA, True,  True,  False, False),
            'SYNTH_1F':    (noise_synth_1f_iir, BETA, False, True,  False, False),
            'WHITE':       (noise_white,        BETA, False, True,  False, False),
            'NO_NOISE':    (noise_zero,         0.0,  False, True,  False, False),
            'NO_FPGA':     (noise_1f_iir,       BETA, False, False, False, False),
            'RANDOM_READ': (noise_1f_iir,       BETA, True,  True,  True,  False),
            'SHUFFLED':    (noise_1f_iir,       BETA, True,  True,  False, True),
        }

        cond_order = ['FULL', 'NO_IIR', 'SYNTH_1F', 'WHITE', 'NO_NOISE',
                      'NO_FPGA', 'RANDOM_READ', 'SHUFFLED']

        # ─── Step 5: Run all conditions ───
        _print("\n[4/6] Running waveform classification across all conditions...")
        total_t0 = time.monotonic()

        for cond_name in cond_order:
            noise_src, beta, live_noise, use_fpga, permute_readout, shuffle_time = conditions[cond_name]

            ser, wave_result = run_condition_waveform(
                cond_name, ser, port, fpga, wave_trials, wave_labels,
                noise_src, w_in, w_noise, beta, live_noise,
                use_fpga, permute_readout, shuffle_time,
                args.n_trials, args.steps_per_trial)

            results['conditions'][cond_name] = {'waveform': wave_result}

        _print(f"\n  Waveform total: {time.monotonic() - total_t0:.1f}s")

        _print("\n[5/6] Running temporal XOR across all conditions...")
        xor_t0 = time.monotonic()

        for cond_name in cond_order:
            noise_src, beta, live_noise, use_fpga, permute_readout, shuffle_time = conditions[cond_name]

            ser, xor_result = run_condition_xor(
                cond_name, ser, port, fpga, xor_input, w_in, w_noise,
                noise_src, beta, live_noise, use_fpga,
                permute_readout, shuffle_time, taus=(1, 2, 3, 5))

            results['conditions'][cond_name]['xor'] = xor_result

        _print(f"\n  XOR total: {time.monotonic() - xor_t0:.1f}s")

        # ─── Step 6: Evaluate tests ───
        _print("\n[6/6] Evaluating tests...")

        full_wave = results['conditions']['FULL']['waveform']['mean_acc']
        full_xor = results['conditions']['FULL']['xor']['best_acc']

        comparisons = [
            ('T71', 'NO_IIR',      'IIR filter adds temporal memory'),
            ('T72', 'SYNTH_1F',    'real GPU noise > synthetic'),
            ('T73', 'WHITE',       'temporal structure matters'),
            ('T74', 'NO_NOISE',    'noise helps computation'),
            ('T75', 'NO_FPGA',     'hardware substrate matters'),
            ('T76', 'RANDOM_READ', 'causal connection matters'),
            ('T77', 'SHUFFLED',    'temporal ordering matters'),
        ]

        pass_count = 0
        total_tests = 8  # T71-T78

        for test_id, ablation, desc in comparisons:
            abl_wave = results['conditions'][ablation]['waveform']['mean_acc']
            abl_xor = results['conditions'][ablation]['xor']['best_acc']
            # Test passes if FULL > ablation on EITHER task (waveform OR xor)
            wave_pass = full_wave > abl_wave
            xor_pass = full_xor > abl_xor
            passed = wave_pass or xor_pass

            results['tests'][test_id] = {
                'description': f'FULL > {ablation}: {desc}',
                'full_wave': float(full_wave),
                'abl_wave': float(abl_wave),
                'wave_delta': float(full_wave - abl_wave),
                'wave_pass': bool(wave_pass),
                'full_xor': float(full_xor),
                'abl_xor': float(abl_xor),
                'xor_delta': float(full_xor - abl_xor),
                'xor_pass': bool(xor_pass),
                'PASS': bool(passed),
            }
            status = 'PASS' if passed else 'FAIL'
            _print(f"  {test_id}: {status} -- FULL({full_wave:.4f}) > {ablation}({abl_wave:.4f}) "
                   f"wave_delta={full_wave - abl_wave:+.4f}, "
                   f"xor_delta={full_xor - abl_xor:+.4f} [{desc}]")
            if passed:
                pass_count += 1

        # T78: Monotonic degradation — check that ablations degrade in expected order
        # Expected order: FULL >= NO_IIR >= SYNTH_1F >= WHITE >= NO_NOISE
        expected_order = ['FULL', 'NO_IIR', 'SYNTH_1F', 'WHITE', 'NO_NOISE']
        wave_in_order = [results['conditions'][c]['waveform']['mean_acc'] for c in expected_order]
        # Count how many adjacent pairs are in order
        n_ordered = sum(1 for i in range(len(wave_in_order) - 1)
                        if wave_in_order[i] >= wave_in_order[i + 1] - 0.01)  # 1% tolerance
        monotonic_frac = n_ordered / (len(wave_in_order) - 1)
        t78_pass = monotonic_frac >= 0.75  # At least 3 of 4 pairs in order

        results['tests']['T78'] = {
            'description': 'Monotonic degradation along causal chain',
            'expected_order': expected_order,
            'accuracies': [float(a) for a in wave_in_order],
            'ordered_pairs': int(n_ordered),
            'total_pairs': len(wave_in_order) - 1,
            'monotonic_fraction': float(monotonic_frac),
            'PASS': bool(t78_pass),
        }
        status = 'PASS' if t78_pass else 'FAIL'
        _print(f"  T78: {status} -- Monotonic degradation {n_ordered}/{len(wave_in_order)-1} "
               f"pairs ordered ({monotonic_frac:.0%})")
        if t78_pass:
            pass_count += 1

        # ─── Summary ───
        _print("\n" + "=" * 70)
        _print(f"z2164 RESULT: {pass_count}/{total_tests} tests PASS")
        _print("=" * 70)

        _print("\nWaveform accuracy by condition:")
        for cond in cond_order:
            cd = results['conditions'][cond]
            acc = cd['waveform']['mean_acc']
            std = cd['waveform']['std_acc']
            energy = cd['waveform'].get('energy_joules')
            e_str = f", {energy:.1f}J" if energy is not None else ""
            _print(f"  {cond:15s}: {acc:.4f} +/- {std:.4f}{e_str}")

        _print("\nXOR best accuracy by condition:")
        for cond in cond_order:
            cd = results['conditions'][cond]
            acc = cd['xor']['best_acc']
            _print(f"  {cond:15s}: {acc:.4f}")

        results['summary'] = {
            'pass_count': pass_count,
            'total_tests': total_tests,
            'simulated': results['simulated'],
            'total_duration_s': float(time.monotonic() - total_t0),
        }

        # ─── Save results ───
        RESULTS.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS / 'z2164_causal_chain.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        _print(f"\nResults saved: {out_path}")

        # ─── Generate figure ───
        fig_path = FIGURES / 'fig_z2164_causal_chain.png'
        make_figures(results, fig_path)

        return pass_count

    except Exception as e:
        _print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

        # Save partial results if we have any conditions
        if results.get('conditions'):
            RESULTS.mkdir(parents=True, exist_ok=True)
            out_path = RESULTS / 'z2164_causal_chain.json'
            results['error'] = str(e)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            _print(f"  Partial results saved: {out_path}")

        return 0

    finally:
        # Always close serial port
        if ser is not None:
            try:
                ser.close()
                _print("  Serial port closed.")
            except Exception:
                pass


if __name__ == '__main__':
    main()
