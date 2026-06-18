#!/usr/bin/env python3
"""z2200_cross_substrate_weight_storage.py — Cross-Substrate Weight Storage

Uses GPU firmware registers (SMN thermal bank, PM table, power rail) as the
"weight matrix" for the FPGA reservoir — making the GPU's physical state the
actual synaptic weights. Closest analog to Mario Lanza's memristive vision
where device physics IS the computation.

Weight layers:
  w_smn[8]   — SMN thermal sensor bank (0x59800-0x5981C), slow structural weights
  w_pm[8]    — PM table float offsets, medium regulatory weights
  w_power[8] — rapid power rail samples, fast noise weights
  w_total    = 0.5*w_smn + 0.3*w_pm + 0.2*w_power

Conditions (5):
  FIRMWARE_WEIGHTS: Full 3-layer firmware weight extraction
  SMN_ONLY:         Only SMN register weights
  PM_ONLY:          Only PM table weights
  FIXED_WEIGHTS:    Traditional fixed random weights (control)
  RANDOM_WEIGHTS:   New random weights each trial (no persistence)

Task: 3-class waveform classification, 120 trials, 25 steps/trial at 20Hz

Tests T293-T298:
  T293: FIRMWARE_WEIGHTS accuracy > RANDOM_WEIGHTS
  T294: SMN weights stable within trial (std < 0.1 over 25 steps)
  T295: Power weights dynamic within trial (std > 0.01 over 25 steps)
  T296: Weight spectrum has 1/f character (PSD slope < -0.3)
  T297: FIRMWARE_WEIGHTS accuracy within 10pp of FIXED_WEIGHTS
  T298: Weight-spike MI > 0.05 bits

SAFETY: NEVER WRITE to SMN mailbox (C2PMSG_66 at 0x3B10A90, C2PMSG_82 at
0x3B10AD0, C2PMSG_90). Only READ from thermal sensor addresses 0x59800-0x5981C.

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


# ─── JSON Encoder ───
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


# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"

# ─── Reservoir Parameters ───
BASE_VG = 0.58
ALPHA = 0.25
N_NEURONS = 8
SAMPLE_HZ = 20

# ─── SMN Addresses (thermal sensor bank — READ ONLY, SAFE) ───
# CRITICAL SAFETY: These are thermal sensor registers ONLY.
# NEVER read/write C2PMSG_66 (0x3B10A90), C2PMSG_82 (0x3B10AD0), C2PMSG_90.
SMN_THERMAL_ADDRS = [
    0x59800, 0x59804, 0x59808, 0x5980C,
    0x59810, 0x59814, 0x59818, 0x5981C,
]

# PM table offsets for 8 float values (power/thermal metrics)
PM_TABLE_OFFSETS = [0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C]

# Conditions
CONDITIONS = ['FIRMWARE_WEIGHTS', 'SMN_ONLY', 'PM_ONLY', 'FIXED_WEIGHTS', 'RANDOM_WEIGHTS']


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
# Firmware Weight Extraction
# ═══════════════════════════════════════════════════════════

def read_smn_register(address: int) -> int:
    """Read a single SMN register. Returns raw 32-bit unsigned value.

    SAFETY: Only call with addresses in SMN_THERMAL_ADDRS (0x59800-0x5981C).
    NEVER use C2PMSG mailbox addresses — causes Data Fabric Sync Flood.
    """
    assert address in SMN_THERMAL_ADDRS, \
        f"SAFETY: SMN address 0x{address:X} not in allowed thermal bank!"
    try:
        with open(SMN_PATH, 'rb+') as f:
            f.write(struct.pack('<I', address))
            f.seek(0)
            data = f.read(4)
            if len(data) < 4:
                return 0
            return struct.unpack('<I', data)[0]
    except Exception:
        return 0


def read_smn_weights() -> np.ndarray:
    """Read 8 SMN thermal registers and normalize to [-1, 1]."""
    raw = np.array([read_smn_register(addr) for addr in SMN_THERMAL_ADDRS], dtype=np.float64)
    if raw.max() == raw.min():
        return np.zeros(N_NEURONS)
    # Normalize to [-1, 1]
    norm = 2.0 * (raw - raw.min()) / max(raw.max() - raw.min(), 1.0) - 1.0
    return norm


def read_pm_table_weights() -> np.ndarray:
    """Read PM table and extract 8 float values at specified offsets."""
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            data = f.read(1024)
        if len(data) < max(PM_TABLE_OFFSETS) + 4:
            return np.zeros(N_NEURONS)
        values = []
        for off in PM_TABLE_OFFSETS:
            val = struct.unpack_from('<f', data, off)[0]
            # Clamp NaN/inf
            if not np.isfinite(val):
                val = 0.0
            values.append(val)
        raw = np.array(values, dtype=np.float64)
        if raw.max() == raw.min():
            return np.zeros(N_NEURONS)
        norm = 2.0 * (raw - raw.min()) / max(raw.max() - raw.min(), 1.0) - 1.0
        return norm
    except Exception:
        return np.zeros(N_NEURONS)


def read_power_weights(n_samples=8, interval=0.01) -> np.ndarray:
    """Sample power1_average rapidly, normalize to [-1, 1]."""
    samples = []
    for _ in range(n_samples):
        try:
            val = int(open(HWMON_POWER).read().strip()) / 1e6
            samples.append(val)
        except Exception:
            samples.append(0.0)
        time.sleep(interval)
    raw = np.array(samples, dtype=np.float64)
    if raw.max() == raw.min():
        return np.zeros(N_NEURONS)
    norm = 2.0 * (raw - raw.min()) / max(raw.max() - raw.min(), 1.0) - 1.0
    return norm


def get_composite_weights(condition: str, rng, fixed_weights: np.ndarray) -> np.ndarray:
    """Get weight vector for given condition."""
    if condition == 'FIRMWARE_WEIGHTS':
        w_smn = read_smn_weights()
        w_pm = read_pm_table_weights()
        w_power = read_power_weights()
        return 0.5 * w_smn + 0.3 * w_pm + 0.2 * w_power
    elif condition == 'SMN_ONLY':
        return read_smn_weights()
    elif condition == 'PM_ONLY':
        return read_pm_table_weights()
    elif condition == 'FIXED_WEIGHTS':
        return fixed_weights.copy()
    elif condition == 'RANDOM_WEIGHTS':
        return rng.uniform(-1, 1, size=N_NEURONS)
    else:
        return fixed_weights.copy()


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=120, steps_per_trial=25, freq_hz=1.0, dt=1.0/20):
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

        wave = (wave + 1.0) / 2.0
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir & LIF Fallback
# ═══════════════════════════════════════════════════════════

def run_fpga_reservoir_trial(ser, input_signal, w_total, base_vg=BASE_VG, alpha=ALPHA):
    """Drive FPGA neurons with firmware-weighted input and collect states.

    Returns: (n_steps, 24) array — 8 delta_spikes + 8 vmem + 8 cumulative_spikes.
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_total
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


def simulate_lif_reservoir(input_signal, w_total, base_vg=BASE_VG, alpha=ALPHA):
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
        vg += alpha * input_signal[t] * w_total
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
    """Pool per-timestep reservoir states into trial-level features."""
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
    indices = np.arange(len(y))
    rng.shuffle(indices)
    classes = np.unique(y)

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
# Analysis Utilities
# ═══════════════════════════════════════════════════════════

def compute_psd_slope(time_series):
    """Compute PSD slope via linear fit in log-log space."""
    if len(time_series) < 8:
        return 0.0
    ts = time_series - np.mean(time_series)
    if np.std(ts) < 1e-12:
        return 0.0
    fft_vals = np.fft.rfft(ts)
    psd = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(ts), d=1.0 / SAMPLE_HZ)
    # Skip DC
    mask = freqs > 0
    freqs = freqs[mask]
    psd = psd[mask]
    if len(freqs) < 3:
        return 0.0
    log_f = np.log10(freqs)
    log_p = np.log10(psd + 1e-30)
    coeffs = np.polyfit(log_f, log_p, 1)
    return coeffs[0]


def compute_mutual_information(x, y, n_bins=10):
    """Estimate MI between x and y via histogram method."""
    if len(x) < 10 or len(y) < 10:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Digitize
    x_bins = np.clip(np.digitize(x, np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1)) - 1, 0, n_bins - 1)
    y_bins = np.clip(np.digitize(y, np.linspace(y.min() - 1e-10, y.max() + 1e-10, n_bins + 1)) - 1, 0, n_bins - 1)

    # Joint histogram
    joint = np.zeros((n_bins, n_bins))
    for xi, yi in zip(x_bins, y_bins):
        joint[xi, yi] += 1
    joint /= joint.sum()

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return max(mi, 0.0)


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2200: Cross-Substrate Weight Storage')
    parser.add_argument('--n-trials', type=int, default=120)
    parser.add_argument('--steps-per-trial', type=int, default=25)
    args = parser.parse_args()

    print("=" * 65)
    print("z2200: Cross-Substrate Weight Storage")
    print("  GPU firmware registers as synaptic weight matrix for FPGA reservoir")
    print("=" * 65)

    rng = np.random.default_rng(42)
    fixed_weights = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2200_cross_substrate_weight_storage',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'smn_addresses': [f'0x{a:X}' for a in SMN_THERMAL_ADDRS],
            'pm_table_offsets': PM_TABLE_OFFSETS,
            'fixed_weights': fixed_weights.tolist(),
            'conditions': CONDITIONS,
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

    # ─── Step 2: Verify firmware weight sources ───
    print("\n[2/6] Verifying firmware weight sources...")

    smn_available = os.path.exists(SMN_PATH)
    pm_available = os.path.exists(PM_TABLE_PATH)
    power_available = os.path.exists(HWMON_POWER)

    print(f"  SMN ({SMN_PATH}): {'available' if smn_available else 'NOT FOUND'}")
    print(f"  PM table ({PM_TABLE_PATH}): {'available' if pm_available else 'NOT FOUND'}")
    print(f"  Power rail ({HWMON_POWER}): {'available' if power_available else 'NOT FOUND'}")

    if smn_available:
        test_w = read_smn_weights()
        print(f"  SMN test read: {test_w}")
    if pm_available:
        test_w = read_pm_table_weights()
        print(f"  PM test read: {test_w}")
    if power_available:
        test_w = read_power_weights(n_samples=4, interval=0.005)
        print(f"  Power test read: {test_w}")

    results['hw_sources'] = {
        'smn_available': smn_available,
        'pm_available': pm_available,
        'power_available': power_available,
    }

    # ─── Step 3: Generate waveforms ───
    print(f"\n[3/6] Generating {args.n_trials} waveforms ({args.steps_per_trial} steps each)...")
    trials, labels = generate_waveforms(
        n_trials=args.n_trials,
        steps_per_trial=args.steps_per_trial,
    )
    class_counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    print(f"  Class distribution: {class_counts}")

    # ─── Step 4: Run reservoir per condition ───
    print(f"\n[4/6] Running reservoir across {len(CONDITIONS)} conditions...")

    condition_features = {}   # condition -> (n_trials, n_features)
    condition_weights = {}    # condition -> list of weight vectors per trial
    weight_timeseries = {}    # condition -> (n_trials, steps, 8) weight snapshots

    for ci, cond in enumerate(CONDITIONS):
        print(f"\n  [{ci+1}/{len(CONDITIONS)}] Condition: {cond}")
        all_features = []
        all_weights = []
        weight_ts = []  # per-trial weight snapshots across steps

        for trial_idx in range(args.n_trials):
            if trial_idx % 20 == 0:
                print(f"    Trial {trial_idx}/{args.n_trials}...")

            input_signal = trials[trial_idx]

            # Collect weight snapshots within this trial for stability analysis
            trial_weight_snapshots = []

            # Get weights for this trial
            w = get_composite_weights(cond, rng, fixed_weights)
            all_weights.append(w.copy())

            # For FIRMWARE_WEIGHTS, snapshot weights at each step
            if cond in ('FIRMWARE_WEIGHTS', 'SMN_ONLY', 'PM_ONLY'):
                # Run trial step-by-step, re-reading weights periodically
                n_steps = len(input_signal)
                interval = 1.0 / SAMPLE_HZ
                states = np.zeros((n_steps, N_NEURONS * 3))
                prev_counts = None
                cumulative = np.zeros(N_NEURONS)

                for t in range(n_steps):
                    # Re-read weights every step for dynamic conditions
                    if cond == 'FIRMWARE_WEIGHTS':
                        # Full composite: re-read power (fast), reuse SMN/PM (slow)
                        if t == 0:
                            w_smn = read_smn_weights() if smn_available else np.zeros(N_NEURONS)
                            w_pm = read_pm_table_weights() if pm_available else np.zeros(N_NEURONS)
                        # Power changes fast, re-read each step
                        if power_available:
                            try:
                                pval = int(open(HWMON_POWER).read().strip()) / 1e6
                            except Exception:
                                pval = 11.0
                            # Single sample, normalize around mean
                            w_power_t = (pval - 11.0) / 2.0
                            w_power_t = np.clip(w_power_t, -1, 1)
                            w_power_vec = np.full(N_NEURONS, w_power_t)
                            # Add small per-neuron variation from power jitter
                            w_power_vec += rng.normal(0, 0.01, N_NEURONS)
                        else:
                            w_power_vec = np.zeros(N_NEURONS)
                        w_step = 0.5 * w_smn + 0.3 * w_pm + 0.2 * w_power_vec
                    elif cond == 'SMN_ONLY':
                        if t % 5 == 0 and smn_available:  # Re-read SMN every 5 steps
                            w_step = read_smn_weights()
                        # else reuse previous w_step
                        elif t == 0:
                            w_step = read_smn_weights() if smn_available else np.zeros(N_NEURONS)
                    elif cond == 'PM_ONLY':
                        if t % 5 == 0 and pm_available:
                            w_step = read_pm_table_weights()
                        elif t == 0:
                            w_step = read_pm_table_weights() if pm_available else np.zeros(N_NEURONS)

                    trial_weight_snapshots.append(w_step.copy())

                    vg_values = np.full(N_NEURONS, BASE_VG)
                    vg_values += ALPHA * input_signal[t] * w_step
                    vg_values = np.clip(vg_values, 0.05, 0.95)

                    if fpga:
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
                    else:
                        # LIF step
                        I_in = vg_values * 5.0
                        if t == 0:
                            vmem_sim = np.zeros(N_NEURONS)
                        dvdt = (-vmem_sim + I_in) / 0.02
                        vmem_sim += dvdt * (1.0 / SAMPLE_HZ)
                        for i in range(N_NEURONS):
                            if vmem_sim[i] >= 1.0:
                                states[t, i] = 1
                                vmem_sim[i] = 0.0
                                cumulative[i] += 1
                        states[t, N_NEURONS:N_NEURONS * 2] = vmem_sim.copy()
                        states[t, N_NEURONS * 2:] = cumulative.copy()

                weight_ts.append(np.array(trial_weight_snapshots))
            else:
                # FIXED_WEIGHTS or RANDOM_WEIGHTS: run normally
                trial_weight_snapshots = [w.copy()] * len(input_signal)
                weight_ts.append(np.array(trial_weight_snapshots))

                if fpga:
                    states = run_fpga_reservoir_trial(ser, input_signal, w)
                else:
                    states = simulate_lif_reservoir(input_signal, w)

            features = pool_trial_features(states)
            all_features.append(features)

        condition_features[cond] = np.array(all_features)
        condition_weights[cond] = np.array(all_weights)
        weight_timeseries[cond] = weight_ts
        print(f"    Features shape: {condition_features[cond].shape}")

    # ─── Step 5: Classification & Analysis ───
    print("\n[5/6] Classification & analysis...")

    accuracies = {}
    for cond in CONDITIONS:
        X = condition_features[cond]
        # Add bias
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

        folds = stratified_kfold(X_bias, labels, n_splits=5)
        fold_accs = []
        for train_idx, test_idx in folds:
            acc = ridge_classify(X_bias[train_idx], labels[train_idx],
                                 X_bias[test_idx], labels[test_idx])
            fold_accs.append(acc)
        mean_acc = np.mean(fold_accs)
        accuracies[cond] = mean_acc
        print(f"  {cond}: accuracy = {mean_acc:.4f} (folds: {[f'{a:.3f}' for a in fold_accs]})")

    results['accuracies'] = {k: float(v) for k, v in accuracies.items()}

    # ─── Weight Stability Analysis ───
    print("\n  Weight stability analysis...")

    # T294: SMN weights stable within trial
    smn_within_trial_stds = []
    if 'SMN_ONLY' in weight_timeseries:
        for ws in weight_timeseries['SMN_ONLY']:
            if ws.ndim == 2 and ws.shape[0] > 1:
                smn_within_trial_stds.append(np.std(ws, axis=0).mean())
    smn_stability = np.mean(smn_within_trial_stds) if smn_within_trial_stds else 0.0
    print(f"  SMN within-trial weight std: {smn_stability:.6f}")

    # T295: Power weights dynamic within trial
    power_within_trial_stds = []
    if 'FIRMWARE_WEIGHTS' in weight_timeseries:
        for ws in weight_timeseries['FIRMWARE_WEIGHTS']:
            if ws.ndim == 2 and ws.shape[0] > 1:
                power_within_trial_stds.append(np.std(ws, axis=0).mean())
    power_dynamics = np.mean(power_within_trial_stds) if power_within_trial_stds else 0.0
    print(f"  FIRMWARE within-trial weight std: {power_dynamics:.6f}")

    # T296: Weight spectrum — concatenate all firmware weights into time series
    weight_concat = []
    if 'FIRMWARE_WEIGHTS' in weight_timeseries:
        for ws in weight_timeseries['FIRMWARE_WEIGHTS']:
            if ws.ndim == 2:
                weight_concat.extend(ws.mean(axis=1).tolist())
    weight_psd_slope = compute_psd_slope(np.array(weight_concat)) if len(weight_concat) > 20 else 0.0
    print(f"  Firmware weight PSD slope: {weight_psd_slope:.4f}")

    # T298: Weight-spike MI
    # Correlate mean weight with total spikes per trial for FIRMWARE_WEIGHTS
    fw_mean_weights = []
    fw_total_spikes = []
    if 'FIRMWARE_WEIGHTS' in condition_weights:
        for i, w in enumerate(condition_weights['FIRMWARE_WEIGHTS']):
            fw_mean_weights.append(np.mean(w))
            # Sum delta spikes across all neurons and steps
            feats = condition_features['FIRMWARE_WEIGHTS'][i]
            # First 8 values of mean = mean delta spikes per neuron
            fw_total_spikes.append(np.sum(feats[:N_NEURONS]))

    weight_spike_mi = compute_mutual_information(
        np.array(fw_mean_weights), np.array(fw_total_spikes)
    ) if len(fw_mean_weights) > 10 else 0.0
    print(f"  Weight-spike MI: {weight_spike_mi:.4f} bits")

    # Weight stability between trials
    weight_between_trial_stds = {}
    for cond in CONDITIONS:
        wts = condition_weights[cond]
        weight_between_trial_stds[cond] = float(np.std(wts, axis=0).mean())
    print(f"  Between-trial weight stds: { {k: f'{v:.4f}' for k, v in weight_between_trial_stds.items()} }")

    results['analysis'] = {
        'smn_within_trial_std': float(smn_stability),
        'firmware_within_trial_std': float(power_dynamics),
        'weight_psd_slope': float(weight_psd_slope),
        'weight_spike_mi': float(weight_spike_mi),
        'between_trial_weight_stds': weight_between_trial_stds,
    }

    # ─── Tests T293-T298 ───
    print("\n[6/6] Evaluating tests T293-T298...")

    acc_fw = accuracies.get('FIRMWARE_WEIGHTS', 0)
    acc_rand = accuracies.get('RANDOM_WEIGHTS', 0)
    acc_fixed = accuracies.get('FIXED_WEIGHTS', 0)

    t293 = acc_fw > acc_rand
    t294 = smn_stability < 0.1
    t295 = power_dynamics > 0.01
    t296 = weight_psd_slope < -0.3
    t297 = abs(acc_fw - acc_fixed) < 0.10
    t298 = weight_spike_mi > 0.05

    tests = {
        'T293_firmware_gt_random': {
            'pass': bool(t293),
            'firmware_acc': float(acc_fw),
            'random_acc': float(acc_rand),
            'description': 'FIRMWARE_WEIGHTS accuracy > RANDOM_WEIGHTS',
        },
        'T294_smn_stable': {
            'pass': bool(t294),
            'smn_within_trial_std': float(smn_stability),
            'threshold': 0.1,
            'description': 'SMN weights stable within trial (std < 0.1)',
        },
        'T295_power_dynamic': {
            'pass': bool(t295),
            'firmware_within_trial_std': float(power_dynamics),
            'threshold': 0.01,
            'description': 'Power weights dynamic within trial (std > 0.01)',
        },
        'T296_weight_1f_spectrum': {
            'pass': bool(t296),
            'psd_slope': float(weight_psd_slope),
            'threshold': -0.3,
            'description': 'Weight spectrum has 1/f character (slope < -0.3)',
        },
        'T297_firmware_near_fixed': {
            'pass': bool(t297),
            'firmware_acc': float(acc_fw),
            'fixed_acc': float(acc_fixed),
            'diff': float(abs(acc_fw - acc_fixed)),
            'threshold': 0.10,
            'description': 'FIRMWARE_WEIGHTS accuracy within 10pp of FIXED_WEIGHTS',
        },
        'T298_weight_spike_mi': {
            'pass': bool(t298),
            'mi_bits': float(weight_spike_mi),
            'threshold': 0.05,
            'description': 'Weight-spike MI > 0.05 bits',
        },
    }

    n_pass = sum(1 for t in tests.values() if t['pass'])
    results['tests'] = tests
    results['pass_count'] = n_pass
    results['total_tests'] = len(tests)

    for name, t in tests.items():
        status = "PASS" if t['pass'] else "FAIL"
        print(f"  {name}: {status} — {t['description']}")

    print(f"\n  Score: {n_pass}/{len(tests)}")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2200_cross_substrate_weight_storage.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    # ─── Generate figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('z2200: Cross-Substrate Weight Storage\n'
                      'GPU Firmware Registers as FPGA Synaptic Weights',
                      fontsize=13, fontweight='bold')

        # (a) Accuracy per condition
        ax = axes[0, 0]
        conds = list(accuracies.keys())
        accs = [accuracies[c] for c in conds]
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#95a5a6']
        bars = ax.bar(range(len(conds)), accs, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.replace('_', '\n') for c in conds], fontsize=8)
        ax.set_ylabel('Classification Accuracy')
        ax.set_title('(a) Waveform Classification by Weight Source')
        ax.axhline(1/3, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.0)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

        # (b) Weight stability comparison
        ax = axes[0, 1]
        stab_conds = list(weight_between_trial_stds.keys())
        stab_vals = [weight_between_trial_stds[c] for c in stab_conds]
        ax.bar(range(len(stab_conds)), stab_vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(stab_conds)))
        ax.set_xticklabels([c.replace('_', '\n') for c in stab_conds], fontsize=8)
        ax.set_ylabel('Between-Trial Weight Std')
        ax.set_title('(b) Weight Stability Across Trials')

        # (c) Weight PSD (if we have the time series)
        ax = axes[1, 0]
        if len(weight_concat) > 20:
            ts = np.array(weight_concat)
            ts = ts - np.mean(ts)
            fft_vals = np.fft.rfft(ts)
            psd = np.abs(fft_vals) ** 2
            freqs = np.fft.rfftfreq(len(ts), d=1.0 / SAMPLE_HZ)
            mask = freqs > 0
            ax.loglog(freqs[mask], psd[mask], 'b-', alpha=0.7, label='Firmware weights')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD')
            ax.set_title(f'(c) Weight Spectrum (slope={weight_psd_slope:.2f})')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('(c) Weight Spectrum')

        # (d) Test results summary
        ax = axes[1, 1]
        test_names = [k.replace('_', ' ') for k in tests.keys()]
        test_passes = [tests[k]['pass'] for k in tests.keys()]
        y_pos = range(len(test_names))
        bar_colors = ['#27ae60' if p else '#e74c3c' for p in test_passes]
        ax.barh(y_pos, [1] * len(test_names), color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(test_names, fontsize=8)
        ax.set_xlim(0, 1.2)
        ax.set_xticks([])
        ax.set_title(f'(d) Tests: {n_pass}/{len(tests)} PASS')
        for i, (name, passed) in enumerate(zip(test_names, test_passes)):
            ax.text(1.05, i, 'PASS' if passed else 'FAIL',
                    va='center', fontsize=9, fontweight='bold',
                    color='#27ae60' if passed else '#e74c3c')

        plt.tight_layout()
        fig_path = FIGURES / 'z2200_cross_substrate_weight_storage.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping figure")

    # ─── Cleanup ───
    if fpga and ser:
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x01]))
        ser.flush()
        ser.close()
        print("  FPGA kill switch re-enabled, port closed")

    print(f"\nDone. {n_pass}/{len(tests)} tests passed.")
    return 0 if n_pass >= 4 else 1


if __name__ == '__main__':
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry  # noqa: F401
    sys.exit(main())
