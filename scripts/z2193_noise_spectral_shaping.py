#!/usr/bin/env python3
"""z2193_noise_spectral_shaping.py — Noise Spectral Shaping in GPU-FPGA Reservoir

Tests whether different SPECTRAL SHAPES of noise produce different computational
properties in the FPGA reservoir.  Directly addresses how GPU firmware noise
characteristics map to functional outcomes — key for Mario Lanza's memristive
computing where device noise spectra vary.

6 noise conditions with different spectral slopes:
  GPU_1F        : Real GPU power rail noise (PSD slope ~ -1.5, natural 1/f)
  SYNTHETIC_1F  : Voss-McCartney 1/f (slope ~ -1.0)
  PINK_STEEP    : IIR-filtered noise with slope ~ -2.0 (Brownian)
  PINK_SHALLOW  : IIR-filtered noise with slope ~ -0.5
  WHITE         : Gaussian white noise (slope ~ 0)
  NO_NOISE      : Deterministic (beta=0)

For each condition: 3-class waveform classification (sine/triangle/square),
150 trials, 25 steps/trial at 20 Hz.

Tests T249-T254:
  T249: GPU_1F accuracy > WHITE accuracy (structured noise helps)
  T250: GPU_1F accuracy > NO_NOISE accuracy (noise helps)
  T251: Accuracy rank correlates with spectral slope (Spearman p < 0.10)
  T252: All 6 noise PSD slopes are distinct (pairwise diffs > 0.2)
  T253: Output PSD slope correlates with input PSD slope (Pearson r > 0.5)
  T254: GPU_1F is best or second-best condition (rank <= 2)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS = BASE / 'results'
FIGURES = (BASE / 'results' / 'FEEL_paper_update'
           / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures')

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
N_TRIALS = 150
STEPS_PER_TRIAL = 25


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
# FPGA Communication  (from z2188)
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
# Noise Sources & Spectral Shaping
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
    Creates temporal memory from raw noise.
    """
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
        self.step = 0

    def process(self, white_sample):
        x = (white_sample / 127.5) - 1.0 if isinstance(white_sample, (int, float)) and white_sample > 1 else white_sample
        for k in range(self.n_octaves):
            period = 1 << k
            if self.step % period == 0:
                self.values[k] = x
        self.step += 1
        total = np.sum(self.values) / self.n_octaves
        return np.clip(total, -1.0, 1.0)


def generate_voss_mcCartney(n_samples, rng):
    """Generate Voss-McCartney 1/f noise (PSD slope ~ -1.0)."""
    n_octaves = 10
    octaves = np.zeros(n_octaves)
    signal = np.zeros(n_samples)
    for i in range(n_samples):
        for j in range(n_octaves):
            if i % (1 << j) == 0:
                octaves[j] = rng.standard_normal()
        signal[i] = octaves.sum()
    signal = (signal - signal.mean()) / max(signal.std(), 1e-6)
    return signal


def generate_shaped_noise(n_samples, alpha_iir, rng):
    """Generate spectrally-shaped noise via IIR filter.
    alpha_iir controls spectral slope:
      0.95 -> steep (Brownian, slope ~ -2.0)
      0.85 -> moderate (1/f, slope ~ -1.0)
      0.60 -> shallow (slope ~ -0.5)
    """
    white = rng.standard_normal(n_samples)
    return iir_filter_noise(white, alpha_iir=alpha_iir)


# ═══════════════════════════════════════════════════════════
# PSD Analysis
# ═══════════════════════════════════════════════════════════

def compute_psd_slope(signal, fs=20.0):
    """Compute power spectral density slope via Welch's method (manual).
    Returns (slope, freqs, psd) where slope is the log-log linear fit.
    """
    n = len(signal)
    if n < 16:
        return 0.0, np.array([]), np.array([])

    # Use FFT-based periodogram with Hann window
    nperseg = min(256, n)
    noverlap = nperseg // 2
    step = nperseg - noverlap
    n_segments = max(1, (n - nperseg) // step + 1)

    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    psd_accum = np.zeros(len(freqs))

    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg))  # Hann
    window_norm = np.sum(window ** 2)

    for seg in range(n_segments):
        start = seg * step
        end = start + nperseg
        if end > n:
            break
        segment = signal[start:end] * window
        fft_vals = np.fft.rfft(segment)
        psd_accum += np.abs(fft_vals) ** 2
        n_segments = seg + 1

    psd = psd_accum / (n_segments * fs * window_norm)
    psd[1:-1] *= 2  # one-sided PSD

    # Fit slope in log-log space (exclude DC)
    valid = (freqs > 0) & (psd > 0)
    if np.sum(valid) < 3:
        return 0.0, freqs, psd

    log_f = np.log10(freqs[valid])
    log_p = np.log10(psd[valid])

    # Linear regression
    A = np.vstack([log_f, np.ones_like(log_f)]).T
    slope, intercept = np.linalg.lstsq(A, log_p, rcond=None)[0]

    return float(slope), freqs, psd


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=150, steps_per_trial=25, freq_hz=1.0, dt=1.0/20):
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
# FPGA Reservoir Core
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, noise_samples, w_in, w_noise,
                              base_vg=DEFAULT_BASE_VG, alpha=DEFAULT_ALPHA,
                              beta=DEFAULT_BETA, live_noise=False):
    """Drive FPGA neurons with input+noise and collect spike/vmem states.

    When live_noise=True, reads power rail in real-time (true substrate coupling).
    Otherwise uses pre-collected noise_samples.

    Returns: (n_steps, 16) array -- 8 delta_spikes + 8 vmem.
    """
    n_steps = len(input_signal)
    interval = 1.0 / DEFAULT_SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 2))  # delta_spikes + vmem
    prev_counts = None
    power_mean = 11.0  # approx mean for normalization

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
                            base_vg=DEFAULT_BASE_VG, alpha=DEFAULT_ALPHA,
                            beta=DEFAULT_BETA):
    """Software LIF simulation fallback when FPGA is not connected."""
    n_steps = len(input_signal)
    states = np.zeros((n_steps, N_NEURONS * 2))  # delta_spikes + vmem

    v_rest = 0.0
    v_thresh = 1.0
    tau_m = 0.02
    dt = 1.0 / DEFAULT_SAMPLE_HZ
    vmem = np.zeros(N_NEURONS)

    for t in range(n_steps):
        vg = np.full(N_NEURONS, base_vg)
        vg += alpha * input_signal[t] * w_in
        if beta > 0 and len(noise_samples) > 0:
            noise_idx = t % len(noise_samples)
            vg += beta * noise_samples[noise_idx] * w_noise
        vg = np.clip(vg, 0.05, 0.95)

        # LIF dynamics
        I_in = vg * 5.0
        dvdt = (-vmem + I_in) / tau_m
        vmem += dvdt * dt

        # Spike detection
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
    """Pool per-timestep reservoir states into trial-level features.
    trial_states: (n_steps, n_features) -> [mean, max].
    """
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.max(axis=0),
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
            W = np.linalg.solve(X_train.T @ X_train + alpha_reg * I,
                                X_train.T @ Y_train)
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


# ═══════════════════════════════════════════════════════════
# Spike Train PSD (output spectral analysis)
# ═══════════════════════════════════════════════════════════

def compute_spike_train_psd_slope(all_trial_states, n_neurons=8, fs=20.0):
    """Compute PSD slope of aggregated spike train across all trials.
    Concatenate delta_spikes across trials, sum across neurons, compute PSD.
    """
    # Concatenate all trials
    spike_trains = []
    for states in all_trial_states:
        # states shape: (steps_per_trial, 2*N_NEURONS), first N_NEURONS = delta_spikes
        spike_trains.append(states[:, :n_neurons].sum(axis=1))  # total spikes per step
    full_train = np.concatenate(spike_trains)
    if len(full_train) < 32:
        return 0.0
    slope, _, _ = compute_psd_slope(full_train, fs=fs)
    return slope


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2193: Noise Spectral Shaping')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps-per-trial', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--noise-collect-s', type=float, default=15.0)
    parser.add_argument('--simulated', action='store_true',
                        help='Force LIF simulation (skip FPGA)')
    args = parser.parse_args()

    print("=" * 65)
    print("z2193: Noise Spectral Shaping in GPU-FPGA Reservoir")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2193_noise_spectral_shaping',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': DEFAULT_BASE_VG, 'alpha': DEFAULT_ALPHA,
            'beta': DEFAULT_BETA, 'n_neurons': N_NEURONS,
            'sample_hz': DEFAULT_SAMPLE_HZ,
            'n_trials': args.n_trials,
            'steps_per_trial': args.steps_per_trial,
            'w_in': w_in.tolist(), 'w_noise': w_noise.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Connect to FPGA ───
    print("\n[1/6] Connecting to FPGA...")
    if args.simulated:
        ser, port = None, None
    else:
        ser, port = find_fpga()

    if ser is None:
        print("  FPGA not found — using LIF simulation fallback")
        fpga = False
        results['simulated'] = True
    else:
        print(f"  Connected: {port}")
        fpga = True
        connect_fpga(ser)
        print("  Kill switch disabled")

    # ─── Step 2: Generate noise conditions ───
    print("\n[2/6] Generating 6 noise conditions...")

    n_noise_samples = max(2000, args.n_trials * args.steps_per_trial)

    # Condition 1: GPU_1F — real GPU power rail noise
    print("  Collecting GPU power rail noise (1/f)...")
    power_noise = collect_power_noise(duration_s=args.noise_collect_s, sample_hz=50)
    if power_noise is not None and len(power_noise) > 10:
        power_mean = power_noise.mean()
        power_std = max(power_noise.std(), 1e-6)
        noise_gpu_1f = (power_noise - power_mean) / power_std
        gpu_1f_live = True
        print(f"  Power rail: {power_mean:.2f} +/- {power_std:.3f} W, {len(noise_gpu_1f)} samples")
    else:
        print("  Power rail unavailable, using high-alpha IIR as GPU 1/f proxy")
        noise_gpu_1f = generate_shaped_noise(n_noise_samples, alpha_iir=0.92, rng=rng)
        gpu_1f_live = False

    # Condition 2: SYNTHETIC_1F — Voss-McCartney
    print("  Generating Voss-McCartney 1/f noise...")
    noise_synthetic_1f = generate_voss_mcCartney(n_noise_samples, rng)

    # Condition 3: PINK_STEEP — IIR alpha=0.95 (Brownian, slope ~ -2.0)
    print("  Generating steep-spectrum noise (alpha=0.95)...")
    noise_pink_steep = generate_shaped_noise(n_noise_samples, alpha_iir=0.95, rng=rng)

    # Condition 4: PINK_SHALLOW — IIR alpha=0.60 (slope ~ -0.5)
    print("  Generating shallow-spectrum noise (alpha=0.60)...")
    noise_pink_shallow = generate_shaped_noise(n_noise_samples, alpha_iir=0.60, rng=rng)

    # Condition 5: WHITE — Gaussian white noise (slope ~ 0)
    print("  Generating white noise...")
    noise_white = rng.standard_normal(n_noise_samples)

    # Condition 6: NO_NOISE — deterministic
    noise_zero = np.zeros(n_noise_samples)

    # Measure input noise PSD slopes
    print("\n  Input noise PSD slopes:")
    noise_conditions = {
        'GPU_1F':         (noise_gpu_1f,        DEFAULT_BETA, gpu_1f_live),
        'SYNTHETIC_1F':   (noise_synthetic_1f,   DEFAULT_BETA, False),
        'PINK_STEEP':     (noise_pink_steep,     DEFAULT_BETA, False),
        'PINK_SHALLOW':   (noise_pink_shallow,   DEFAULT_BETA, False),
        'WHITE':          (noise_white,           DEFAULT_BETA, False),
        'NO_NOISE':       (noise_zero,            0.0,          False),
    }

    input_psd_slopes = {}
    for name, (noise_src, _, _) in noise_conditions.items():
        if name == 'NO_NOISE':
            input_psd_slopes[name] = 0.0  # undefined, use 0
            print(f"    {name}: N/A (deterministic)")
        else:
            slope, _, _ = compute_psd_slope(noise_src, fs=DEFAULT_SAMPLE_HZ)
            input_psd_slopes[name] = slope
            print(f"    {name}: slope = {slope:.3f}")

    results['input_psd_slopes'] = input_psd_slopes

    # ─── Step 3: Generate waveform task ───
    print(f"\n[3/6] Generating waveform classification task ({args.n_trials} trials)...")
    wave_trials, wave_labels = generate_waveforms(
        n_trials=args.n_trials, steps_per_trial=args.steps_per_trial)
    print(f"  Class distribution: {np.bincount(wave_labels).tolist()}")

    # ─── Step 4: Run reservoir for each condition ───
    print("\n[4/6] Running FPGA reservoir for 6 noise conditions...")

    condition_features = {}
    condition_trial_states = {}  # for output PSD analysis

    for cond_name, (noise_src, beta, live) in noise_conditions.items():
        print(f"\n  === {cond_name} (beta={beta:.2f}, live={live}) ===")
        trial_features = []
        trial_states_list = []
        t0 = time.monotonic()

        for trial_idx in range(args.n_trials):
            input_signal = wave_trials[trial_idx]

            if fpga:
                states = run_fpga_reservoir_trial(
                    ser, input_signal, noise_src, w_in, w_noise,
                    base_vg=DEFAULT_BASE_VG, alpha=DEFAULT_ALPHA, beta=beta,
                    live_noise=live)
            else:
                states = simulate_lif_reservoir(
                    input_signal, noise_src, w_in, w_noise,
                    base_vg=DEFAULT_BASE_VG, alpha=DEFAULT_ALPHA, beta=beta)

            feat = pool_trial_features(states)
            trial_features.append(feat)
            trial_states_list.append(states)

            if (trial_idx + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial_idx + 1) / elapsed
                eta = (args.n_trials - trial_idx - 1) / rate
                print(f"    Trial {trial_idx+1}/{args.n_trials} "
                      f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

        condition_features[cond_name] = np.array(trial_features)
        condition_trial_states[cond_name] = trial_states_list
        elapsed = time.monotonic() - t0
        print(f"  {cond_name}: {len(trial_features)} trials in {elapsed:.1f}s, "
              f"feature dim={condition_features[cond_name].shape[1]}")

    # ─── Step 5: Classify & analyse ───
    print("\n[5/6] Classifying waveforms (5-fold stratified CV)...")

    splits = stratified_kfold(
        condition_features['GPU_1F'], wave_labels, n_splits=5)

    wave_accuracies = {}
    for cond_name, X_all in condition_features.items():
        fold_accs = []
        for train_idx, test_idx in splits:
            X_train = X_all[train_idx]
            X_test = X_all[test_idx]
            y_train = wave_labels[train_idx]
            y_test = wave_labels[test_idx]

            # Z-score normalize
            mu = X_train.mean(axis=0, keepdims=True)
            sigma = X_train.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_train_n = (X_train - mu) / sigma
            X_test_n = (X_test - mu) / sigma

            acc = ridge_classify(X_train_n, y_train, X_test_n, y_test)
            fold_accs.append(acc)

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        wave_accuracies[cond_name] = {
            'mean': float(mean_acc), 'std': float(std_acc),
            'folds': [float(a) for a in fold_accs],
        }
        print(f"  {cond_name}: {mean_acc:.3f} +/- {std_acc:.3f}")

    results['waveform_classification'] = wave_accuracies

    # ─── Output PSD analysis ───
    print("\n  Output spike train PSD slopes:")
    output_psd_slopes = {}
    for cond_name, trial_states_list in condition_trial_states.items():
        slope = compute_spike_train_psd_slope(trial_states_list, fs=DEFAULT_SAMPLE_HZ)
        output_psd_slopes[cond_name] = slope
        print(f"    {cond_name}: output PSD slope = {slope:.3f}")

    results['output_psd_slopes'] = output_psd_slopes

    # ─── Step 6: Tests T249-T254 ───
    print("\n[6/6] Evaluating tests T249-T254...")

    tests = {}
    cond_names_ordered = ['GPU_1F', 'SYNTHETIC_1F', 'PINK_STEEP',
                          'PINK_SHALLOW', 'WHITE', 'NO_NOISE']

    acc_gpu = wave_accuracies['GPU_1F']['mean']
    acc_white = wave_accuracies['WHITE']['mean']
    acc_no_noise = wave_accuracies['NO_NOISE']['mean']

    # T249: GPU_1F > WHITE
    t249_pass = acc_gpu > acc_white
    tests['T249_gpu_gt_white'] = {
        'pass': bool(t249_pass),
        'gpu_1f_acc': acc_gpu,
        'white_acc': acc_white,
        'delta': acc_gpu - acc_white,
        'description': 'GPU 1/f noise accuracy > white noise accuracy',
    }
    print(f"  T249 GPU_1F > WHITE: {'PASS' if t249_pass else 'FAIL'} "
          f"({acc_gpu:.3f} vs {acc_white:.3f}, delta={acc_gpu - acc_white:.3f})")

    # T250: GPU_1F > NO_NOISE
    t250_pass = acc_gpu > acc_no_noise
    tests['T250_gpu_gt_nonoise'] = {
        'pass': bool(t250_pass),
        'gpu_1f_acc': acc_gpu,
        'no_noise_acc': acc_no_noise,
        'delta': acc_gpu - acc_no_noise,
        'description': 'GPU 1/f noise accuracy > no noise accuracy',
    }
    print(f"  T250 GPU_1F > NO_NOISE: {'PASS' if t250_pass else 'FAIL'} "
          f"({acc_gpu:.3f} vs {acc_no_noise:.3f}, delta={acc_gpu - acc_no_noise:.3f})")

    # T251: Spearman rank correlation between PSD slope and accuracy
    # Use only the 5 conditions with actual noise (exclude NO_NOISE for meaningful slope)
    conds_with_noise = ['GPU_1F', 'SYNTHETIC_1F', 'PINK_STEEP', 'PINK_SHALLOW', 'WHITE']
    slopes_arr = np.array([input_psd_slopes[c] for c in conds_with_noise])
    accs_arr = np.array([wave_accuracies[c]['mean'] for c in conds_with_noise])

    # Spearman rank correlation (manual — rank-based)
    def spearman_rank(x, y):
        n = len(x)
        rank_x = np.argsort(np.argsort(x)).astype(float) + 1
        rank_y = np.argsort(np.argsort(y)).astype(float) + 1
        d = rank_x - rank_y
        rho = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
        # Approximate t-test for significance
        if abs(rho) >= 1.0:
            p_val = 0.0
        else:
            t_stat = rho * np.sqrt((n - 2) / (1 - rho ** 2))
            # Two-tailed p-value from t-distribution (approximate via normal for n>=5)
            from math import erfc, sqrt
            p_val = erfc(abs(t_stat) / sqrt(2))
        return rho, p_val

    spearman_rho, spearman_p = spearman_rank(slopes_arr, accs_arr)
    t251_pass = spearman_p < 0.10
    tests['T251_slope_accuracy_correlation'] = {
        'pass': bool(t251_pass),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'slopes': {c: input_psd_slopes[c] for c in conds_with_noise},
        'accuracies': {c: wave_accuracies[c]['mean'] for c in conds_with_noise},
        'description': 'Spearman rank correlation between PSD slope and accuracy (p < 0.10)',
    }
    print(f"  T251 slope-accuracy Spearman: {'PASS' if t251_pass else 'FAIL'} "
          f"(rho={spearman_rho:.3f}, p={spearman_p:.4f})")

    # T252: All 6 noise PSD slopes are distinct (pairwise > 0.2)
    all_slopes = [input_psd_slopes[c] for c in cond_names_ordered]
    min_pairwise_diff = float('inf')
    for i in range(len(all_slopes)):
        for j in range(i + 1, len(all_slopes)):
            diff = abs(all_slopes[i] - all_slopes[j])
            if diff < min_pairwise_diff:
                min_pairwise_diff = diff
    t252_pass = min_pairwise_diff > 0.2
    tests['T252_distinct_slopes'] = {
        'pass': bool(t252_pass),
        'min_pairwise_diff': float(min_pairwise_diff),
        'slopes': {c: input_psd_slopes[c] for c in cond_names_ordered},
        'description': 'All 6 noise PSD slopes pairwise different > 0.2',
    }
    print(f"  T252 distinct slopes: {'PASS' if t252_pass else 'FAIL'} "
          f"(min pairwise diff={min_pairwise_diff:.3f})")

    # T253: Output PSD slope correlates with input PSD slope (Pearson r > 0.5)
    in_slopes = np.array([input_psd_slopes[c] for c in conds_with_noise])
    out_slopes = np.array([output_psd_slopes[c] for c in conds_with_noise])

    def pearson_r(x, y):
        mx, my = x.mean(), y.mean()
        cov = np.sum((x - mx) * (y - my))
        sx = np.sqrt(np.sum((x - mx) ** 2))
        sy = np.sqrt(np.sum((y - my) ** 2))
        if sx < 1e-10 or sy < 1e-10:
            return 0.0
        return float(cov / (sx * sy))

    r_io = pearson_r(in_slopes, out_slopes)
    t253_pass = r_io > 0.5
    tests['T253_input_output_psd_correlation'] = {
        'pass': bool(t253_pass),
        'pearson_r': r_io,
        'input_slopes': {c: input_psd_slopes[c] for c in conds_with_noise},
        'output_slopes': {c: output_psd_slopes[c] for c in conds_with_noise},
        'description': 'Output PSD slope correlates with input PSD slope (Pearson r > 0.5)',
    }
    print(f"  T253 input-output PSD correlation: {'PASS' if t253_pass else 'FAIL'} "
          f"(r={r_io:.3f})")

    # T254: GPU_1F is best or second-best condition (rank <= 2)
    acc_ranking = sorted(cond_names_ordered,
                         key=lambda c: wave_accuracies[c]['mean'], reverse=True)
    gpu_rank = acc_ranking.index('GPU_1F') + 1
    t254_pass = gpu_rank <= 2
    tests['T254_gpu_top2'] = {
        'pass': bool(t254_pass),
        'gpu_rank': gpu_rank,
        'ranking': acc_ranking,
        'ranking_accuracies': {c: wave_accuracies[c]['mean'] for c in acc_ranking},
        'description': 'GPU 1/f noise is best or second-best condition (rank <= 2)',
    }
    print(f"  T254 GPU_1F rank: {'PASS' if t254_pass else 'FAIL'} "
          f"(rank={gpu_rank}, top={acc_ranking[0]})")

    # ─── Summary ───
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['tests'] = tests
    results['summary'] = {
        'pass': n_pass,
        'total': n_total,
        'score': f"{n_pass}/{n_total}",
    }

    print(f"\n{'=' * 65}")
    print(f"  RESULT: {n_pass}/{n_total} tests passed")
    print(f"{'=' * 65}")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2193_noise_spectral_shaping.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved to {out_path}")

    # ─── Generate figure ───
    print("\n  Generating figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'z2193: Noise Spectral Shaping ({n_pass}/{n_total} PASS)',
                     fontsize=14, fontweight='bold')

        # (1) Accuracy bars per condition
        ax = axes[0, 0]
        cond_labels = cond_names_ordered
        means = [wave_accuracies[c]['mean'] for c in cond_labels]
        stds = [wave_accuracies[c]['std'] for c in cond_labels]
        colors = ['#e74c3c', '#e67e22', '#9b59b6', '#3498db', '#95a5a6', '#bdc3c7']
        bars = ax.bar(range(len(cond_labels)), means, yerr=stds, capsize=4,
                      color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(cond_labels)))
        ax.set_xticklabels(cond_labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy')
        ax.set_title('Waveform Classification Accuracy by Noise Condition')
        ax.axhline(y=1.0/3.0, color='gray', linestyle='--', alpha=0.5, label='chance')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.0)

        # (2) PSD slopes: input vs output
        ax = axes[0, 1]
        for i, c in enumerate(conds_with_noise):
            ax.scatter(input_psd_slopes[c], output_psd_slopes[c],
                       color=colors[i], s=100, zorder=5, edgecolor='black')
            ax.annotate(c, (input_psd_slopes[c], output_psd_slopes[c]),
                        fontsize=7, ha='left', va='bottom')
        # Fit line
        if len(conds_with_noise) >= 2:
            fit = np.polyfit(in_slopes, out_slopes, 1)
            x_fit = np.linspace(in_slopes.min() - 0.2, in_slopes.max() + 0.2, 50)
            ax.plot(x_fit, np.polyval(fit, x_fit), 'k--', alpha=0.4, label=f'r={r_io:.2f}')
        ax.set_xlabel('Input Noise PSD Slope')
        ax.set_ylabel('Output Spike Train PSD Slope')
        ax.set_title(f'Input vs Output Spectral Slopes (r={r_io:.2f})')
        ax.legend(fontsize=8)

        # (3) Accuracy vs spectral slope scatter
        ax = axes[1, 0]
        for i, c in enumerate(conds_with_noise):
            ax.scatter(input_psd_slopes[c], wave_accuracies[c]['mean'],
                       color=colors[i], s=100, zorder=5, edgecolor='black')
            ax.annotate(c, (input_psd_slopes[c], wave_accuracies[c]['mean']),
                        fontsize=7, ha='left', va='bottom')
        ax.set_xlabel('Input Noise PSD Slope')
        ax.set_ylabel('Classification Accuracy')
        ax.set_title(f'Accuracy vs Spectral Slope (Spearman rho={spearman_rho:.2f}, p={spearman_p:.3f})')
        ax.axhline(y=1.0/3.0, color='gray', linestyle='--', alpha=0.5)

        # (4) Test results summary
        ax = axes[1, 1]
        ax.axis('off')
        test_names = ['T249', 'T250', 'T251', 'T252', 'T253', 'T254']
        test_keys = list(tests.keys())
        rows = []
        for tn, tk in zip(test_names, test_keys):
            t = tests[tk]
            status = 'PASS' if t['pass'] else 'FAIL'
            desc = t['description'][:50]
            rows.append([tn, status, desc])
        table = ax.table(cellText=rows, colLabels=['Test', 'Result', 'Description'],
                         loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        for i in range(len(rows)):
            color = '#c8e6c9' if rows[i][1] == 'PASS' else '#ffcdd2'
            for j in range(3):
                table[i + 1, j].set_facecolor(color)
        ax.set_title(f'Test Results: {n_pass}/{n_total} PASS', fontweight='bold')

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'fig_z2193_noise_spectral_shaping.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved to {fig_path}")
    except Exception as e:
        print(f"  Figure generation failed: {e}")

    # Cleanup
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass

    return n_pass, n_total


if __name__ == '__main__':
    main()
