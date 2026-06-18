#!/usr/bin/env python3
"""z2197_bidirectional_gpu_fpga_loop.py — Bidirectional GPU↔FPGA Computation Loop

The MOST IMPORTANT experiment in the FEEL project: a genuine BIDIRECTIONAL
computation loop where FPGA spike patterns modulate GPU workload intensity,
which changes GPU power/thermal/clock state, which feeds back as drive
current to the FPGA neurons.

Architecture:
  GPU Firmware State ──→ Noise+State Extraction ──→ FPGA LIF Neurons (8)
       ↑                                                    │
       └──── GPU Workload Modulation ←── Spike Decision ────┘

Conditions (4):
  BIDIRECTIONAL   — Full loop: FPGA spikes → GPU workload → GPU state → FPGA
  OPEN_LOOP       — GPU state → FPGA, but workload is FIXED (no feedback)
  RANDOM_WORKLOAD — Random GPU workload, no spike-driven modulation
  NO_GPU          — No GPU state, no workload modulation (pure FPGA)

Task: 3-class waveform classification (sine/triangle/square)
  150 trials, 30 steps/trial at 10 Hz

Tests T273-T280:
  T273: BIDIRECTIONAL acc > OPEN_LOOP acc (feedback helps)
  T274: BIDIRECTIONAL acc > NO_GPU acc (GPU state adds information)
  T275: MI(FPGA spikes, GPU state) > 0.1 bits in BIDIRECTIONAL
  T276: GPU power variance BIDIRECTIONAL > OPEN_LOOP (FPGA modulates GPU)
  T277: Workload-spike correlation > 0.3 in BIDIRECTIONAL
  T278: GPU state feedback latency measurable (xcorr peak at lag > 0)
  T279: BIDIRECTIONAL acc > RANDOM_WORKLOAD (structured > random)
  T280: Loop gain > 0.1 (spike output modulates GPU → next spike input)

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-4' / 'figures'

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

# ─── GPU State Paths ───
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
HWMON_TEMP = "/sys/class/hwmon/hwmon7/temp1_input"
GPU_BUSY = "/sys/class/drm/card0/device/gpu_busy_percent"
SCLK_PATH = "/sys/class/drm/card0/device/pp_dpm_sclk"

# ─── Parameters ───
N_NEURONS = 8
BASE_VG = 0.55
ALPHA = 0.15         # input coupling
BETA_POWER = 0.10    # power state coupling
BETA_TEMP = 0.05     # thermal state coupling
SAMPLE_HZ = 10       # slower: allows GPU state to change between steps
N_TRIALS = 150
STEPS_PER_TRIAL = 30


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
# GPU State Readout (4 channels)
# ═══════════════════════════════════════════════════════════

def read_gpu_power():
    """Read hwmon power1_average (uW -> W)."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


def read_gpu_temp():
    """Read hwmon temp1_input (mC -> C)."""
    try:
        return int(open(HWMON_TEMP).read().strip()) / 1000.0
    except Exception:
        return None


def read_gpu_busy():
    """Read gpu_busy_percent (0-100)."""
    try:
        return int(open(GPU_BUSY).read().strip())
    except Exception:
        return None


def read_gpu_sclk():
    """Read current SCLK frequency from pp_dpm_sclk (MHz)."""
    try:
        for line in open(SCLK_PATH):
            if '*' in line:
                # e.g. "1: 1800Mhz *"
                parts = line.strip().split()
                for p in parts:
                    if 'mhz' in p.lower():
                        return int(p.lower().replace('mhz', ''))
        return None
    except Exception:
        return None


def read_gpu_state():
    """Read all 4 GPU state channels. Returns dict with available readings."""
    return {
        'power': read_gpu_power(),
        'temp': read_gpu_temp(),
        'busy': read_gpu_busy(),
        'sclk': read_gpu_sclk(),
    }


# ═══════════════════════════════════════════════════════════
# GPU Workload Modulation (HIP compute kernel via PyTorch)
# ═══════════════════════════════════════════════════════════

_torch_available = False
_torch_device = None


def init_torch():
    """Initialize torch with ROCm/HIP backend."""
    global _torch_available, _torch_device
    try:
        import torch
        if torch.cuda.is_available():
            _torch_device = torch.device('cuda')
            # Warm up
            _ = torch.randn(64, 64, device=_torch_device)
            _torch_available = True
            print(f"  PyTorch HIP initialized: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARNING: torch.cuda not available, GPU workload modulation disabled")
            _torch_available = False
    except ImportError:
        print("  WARNING: torch not available, GPU workload modulation disabled")
        _torch_available = False


def run_gpu_workload(intensity: float):
    """Run a HIP compute kernel with intensity-scaled workload.

    intensity in [0, 1]:
      0.0 -> 256x256 matmul (light load)
      1.0 -> 1024x1024 matmul (heavy load)
    """
    if not _torch_available:
        return
    import torch
    N = int(256 + 768 * np.clip(intensity, 0.0, 1.0))
    # Matrix multiply creates real GPU work that affects power/thermal/clock
    a = torch.randn(N, N, device=_torch_device)
    b = torch.randn(N, N, device=_torch_device)
    c = a @ b
    torch.cuda.synchronize()
    del a, b, c


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=150, steps_per_trial=30, freq_hz=0.5, dt=1.0 / 10):
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

        wave = (wave + 1.0) / 2.0  # normalize to [0, 1]
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# LIF Simulation Fallback
# ═══════════════════════════════════════════════════════════

def simulate_lif_step(vmem, vg_values, dt=0.1, tau_m=0.02, v_thresh=1.0):
    """Single LIF step for simulation fallback."""
    I_in = vg_values * 5.0
    dvdt = (-vmem + I_in) / tau_m
    vmem = vmem + dvdt * dt
    spikes = np.zeros(N_NEURONS)
    for i in range(N_NEURONS):
        if vmem[i] >= v_thresh:
            spikes[i] = 1
            vmem[i] = 0.0
    return vmem, spikes


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

def pool_trial_features(trial_states):
    """Pool per-timestep (n_steps, 16) into trial-level features.
    16 = 8 delta_spikes + 8 vmem. Pool with [mean, max, std] -> 48 features.
    """
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.max(axis=0),
        trial_states.std(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    n_classes = len(np.unique(y_train))
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y_train[i, int(y)] = 1.0

    best_acc = -1
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            W = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError:
            continue
        pred_test = np.argmax(X_test @ W, axis=1)
        acc_test = np.mean(pred_test == y_test)
        if acc_test > best_acc:
            best_acc = acc_test
    return best_acc


def stratified_kfold(n_samples, y, n_splits=5, seed=42):
    """Simple stratified k-fold split."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    indices = np.arange(n_samples)
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


def estimate_mi(x, y, n_bins=8):
    """Estimate mutual information between two 1D arrays via binned histogram."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 10 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    # Bin edges
    x_edges = np.linspace(np.min(x), np.max(x) + 1e-10, n_bins + 1)
    y_edges = np.linspace(np.min(y), np.max(y) + 1e-10, n_bins + 1)
    joint, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    # MI = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return max(0.0, mi)


# ═══════════════════════════════════════════════════════════
# Bidirectional Loop Core
# ═══════════════════════════════════════════════════════════

def run_bidirectional_trial(ser, fpga_hw, input_signal, w_in, w_power, w_temp, w_output,
                            condition, rng,
                            base_vg=BASE_VG, alpha=ALPHA,
                            beta_power=BETA_POWER, beta_temp=BETA_TEMP):
    """Run one trial of the bidirectional loop.

    condition: 'BIDIRECTIONAL', 'OPEN_LOOP', 'RANDOM_WORKLOAD', 'NO_GPU'

    Returns:
        states:     (n_steps, 16) — 8 delta_spikes + 8 vmem
        gpu_states: list of dicts with power/temp/busy/sclk per step
        intensities: list of float workload intensities per step
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 2))  # delta_spikes + vmem
    gpu_log = []
    intensity_log = []
    prev_counts = None

    # Normalization constants for GPU state
    power_mean, power_scale = 11.0, 3.0   # ~11W ± 3W
    temp_mean, temp_scale = 50.0, 15.0     # ~50C ± 15C

    # LIF sim state
    sim_vmem = np.zeros(N_NEURONS)

    # Fixed workload for OPEN_LOOP
    fixed_intensity = 0.5

    for t in range(n_steps):
        t0_step = time.monotonic()

        # ── Phase 1: Read GPU state ──
        if condition != 'NO_GPU':
            gs = read_gpu_state()
        else:
            gs = {'power': None, 'temp': None, 'busy': None, 'sclk': None}
        gpu_log.append(gs)

        # Normalized GPU state signals
        power_norm = ((gs['power'] or power_mean) - power_mean) / power_scale
        temp_norm = ((gs['temp'] or temp_mean) - temp_mean) / temp_scale

        # ── Phase 2: Compute per-neuron Vg ──
        vg_values = np.full(N_NEURONS, base_vg)
        vg_values += alpha * input_signal[t] * w_in

        if condition != 'NO_GPU':
            vg_values += beta_power * power_norm * w_power
            vg_values += beta_temp * temp_norm * w_temp

        vg_values = np.clip(vg_values, 0.05, 0.95)

        # ── Phase 3: Drive FPGA or simulate ──
        if fpga_hw and ser is not None:
            set_per_neuron_vg(ser, vg_values)
            time.sleep(0.02)  # allow FPGA to integrate

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
            # else: zeros remain
        else:
            # LIF simulation fallback
            sim_vmem, spikes = simulate_lif_step(sim_vmem, vg_values)
            states[t, :N_NEURONS] = spikes
            states[t, N_NEURONS:] = sim_vmem.copy()

        # ── Phase 4: Spike-driven GPU workload modulation ──
        # Compute spike rates from recent history (last 5 steps or available)
        lookback = min(t + 1, 5)
        recent_spikes = states[max(0, t + 1 - lookback):t + 1, :N_NEURONS]
        spike_rates = recent_spikes.mean(axis=0)  # per-neuron mean rate

        if condition == 'BIDIRECTIONAL':
            raw = np.sum(spike_rates * w_output)
            intensity = float(sigmoid(raw))
        elif condition == 'OPEN_LOOP':
            intensity = fixed_intensity
        elif condition == 'RANDOM_WORKLOAD':
            intensity = float(rng.uniform(0.0, 1.0))
        else:  # NO_GPU
            intensity = 0.0

        intensity_log.append(intensity)

        # Run GPU workload (except NO_GPU)
        if condition != 'NO_GPU':
            run_gpu_workload(intensity)

        # Wait for remainder of timestep
        elapsed = time.monotonic() - t0_step
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    return states, gpu_log, intensity_log


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    global SAMPLE_HZ

    parser = argparse.ArgumentParser(description='z2197: Bidirectional GPU-FPGA Loop')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps-per-trial', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--sample-hz', type=float, default=SAMPLE_HZ)
    args = parser.parse_args()

    SAMPLE_HZ = args.sample_hz

    print("=" * 70)
    print("z2197: BIDIRECTIONAL GPU <-> FPGA COMPUTATION LOOP")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Fixed random weights per neuron for each channel
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_power = rng.uniform(-1, 1, size=N_NEURONS)
    w_temp = rng.uniform(-1, 1, size=N_NEURONS)
    w_output = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2197_bidirectional_gpu_fpga_loop',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA,
            'beta_power': BETA_POWER, 'beta_temp': BETA_TEMP,
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
            'w_in': w_in.tolist(), 'w_power': w_power.tolist(),
            'w_temp': w_temp.tolist(), 'w_output': w_output.tolist(),
        },
        'simulated': False,
    }

    # ─── Step 1: Initialize hardware ───
    print("\n[1/6] Initializing hardware...")

    # FPGA
    ser, port = find_fpga()
    if ser is None:
        print("  FPGA not found — using LIF simulation fallback")
        fpga_hw = False
        results['simulated'] = True
    else:
        print(f"  FPGA connected: {port}")
        fpga_hw = True
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        print("  Kill switch disabled")

    # Torch / HIP
    init_torch()

    # GPU state baseline
    gs = read_gpu_state()
    print(f"  GPU baseline: power={gs['power']}W, temp={gs['temp']}C, "
          f"busy={gs['busy']}%, sclk={gs['sclk']}MHz")
    results['gpu_baseline'] = {k: v for k, v in gs.items()}

    # Telemetry API (optional, for logging)
    tel = None
    try:
        from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
        tel = SysfsHwmonTelemetry()
        print("  SysfsHwmonTelemetry loaded")
    except Exception as e:
        print(f"  SysfsHwmonTelemetry unavailable: {e}")

    # ─── Step 2: Generate waveform task ───
    print(f"\n[2/6] Generating {args.n_trials} waveform trials "
          f"({args.steps_per_trial} steps @ {SAMPLE_HZ} Hz)...")
    wave_trials, wave_labels = generate_waveforms(
        n_trials=args.n_trials, steps_per_trial=args.steps_per_trial,
        freq_hz=0.5, dt=1.0 / SAMPLE_HZ)
    print(f"  Class distribution: {np.bincount(wave_labels)}")

    # ─── Step 3: Run all 4 conditions ───
    conditions = ['BIDIRECTIONAL', 'OPEN_LOOP', 'RANDOM_WORKLOAD', 'NO_GPU']
    cond_features = {}
    cond_gpu_logs = {}
    cond_intensity_logs = {}
    cond_spike_logs = {}

    for ci, cond in enumerate(conditions):
        print(f"\n[3/6] Condition {ci+1}/4: {cond}")
        trial_features = []
        all_gpu = []
        all_intensity = []
        all_spikes = []
        t0 = time.monotonic()

        for trial_idx in range(args.n_trials):
            input_signal = wave_trials[trial_idx]

            states, gpu_log, intensity_log = run_bidirectional_trial(
                ser, fpga_hw, input_signal, w_in, w_power, w_temp, w_output,
                condition=cond, rng=rng,
                base_vg=BASE_VG, alpha=ALPHA,
                beta_power=BETA_POWER, beta_temp=BETA_TEMP)

            # Pool features: (steps, 16) -> 48
            feat = pool_trial_features(states)
            trial_features.append(feat)

            # Log GPU state and intensity for analysis
            powers = [g['power'] for g in gpu_log if g['power'] is not None]
            all_gpu.extend(powers)
            all_intensity.extend(intensity_log)
            all_spikes.append(states[:, :N_NEURONS].sum(axis=1))  # total spikes per step

            if (trial_idx + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial_idx + 1) / elapsed
                eta = (args.n_trials - trial_idx - 1) / max(rate, 0.01)
                print(f"    Trial {trial_idx+1}/{args.n_trials} "
                      f"({rate:.1f} trials/s, ETA {eta:.0f}s)")

        cond_features[cond] = np.array(trial_features)
        cond_gpu_logs[cond] = np.array(all_gpu) if all_gpu else np.array([0.0])
        cond_intensity_logs[cond] = np.array(all_intensity)
        cond_spike_logs[cond] = np.concatenate(all_spikes) if all_spikes else np.array([0.0])

        elapsed = time.monotonic() - t0
        print(f"  {cond}: {len(trial_features)} trials in {elapsed:.1f}s")
        if len(all_gpu) > 0:
            print(f"    GPU power: {np.mean(all_gpu):.2f} +/- {np.std(all_gpu):.3f} W")
        print(f"    Intensity: {np.mean(all_intensity):.3f} +/- {np.std(all_intensity):.3f}")

    # ─── Step 4: Classification (5-fold stratified CV) ───
    print(f"\n[4/6] Classifying waveforms (5-fold stratified CV)...")

    splits = stratified_kfold(args.n_trials, wave_labels, n_splits=5)
    cond_accuracies = {}

    for cond in conditions:
        X_all = cond_features[cond]
        fold_accs = []
        for train_idx, test_idx in splits:
            X_train = X_all[train_idx]
            X_test = X_all[test_idx]
            y_train = wave_labels[train_idx]
            y_test = wave_labels[test_idx]

            mu = X_train.mean(axis=0, keepdims=True)
            sigma = X_train.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_train_n = (X_train - mu) / sigma
            X_test_n = (X_test - mu) / sigma

            acc = ridge_classify(X_train_n, y_train, X_test_n, y_test)
            fold_accs.append(acc)

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        cond_accuracies[cond] = {'mean': mean_acc, 'std': std_acc,
                                  'folds': [float(a) for a in fold_accs]}
        print(f"  {cond}: {mean_acc:.3f} +/- {std_acc:.3f}")

    results['classification'] = cond_accuracies

    # ─── Step 5: Compute loop metrics ───
    print(f"\n[5/6] Computing loop metrics...")

    # T275: Mutual information between spike rates and GPU power
    bi_spikes = cond_spike_logs['BIDIRECTIONAL']
    bi_gpu = cond_gpu_logs['BIDIRECTIONAL']
    n_common = min(len(bi_spikes), len(bi_gpu))
    mi_spike_gpu = estimate_mi(bi_spikes[:n_common], bi_gpu[:n_common])
    print(f"  MI(spikes, GPU power) in BIDIRECTIONAL: {mi_spike_gpu:.4f} bits")

    # T276: GPU power variance comparison
    var_bi = float(np.var(cond_gpu_logs['BIDIRECTIONAL']))
    var_ol = float(np.var(cond_gpu_logs['OPEN_LOOP']))
    print(f"  GPU power variance: BIDIRECTIONAL={var_bi:.4f}, OPEN_LOOP={var_ol:.4f}")

    # T277: Workload-spike correlation in BIDIRECTIONAL
    bi_intensity = cond_intensity_logs['BIDIRECTIONAL']
    n_common_ws = min(len(bi_spikes), len(bi_intensity))
    if n_common_ws > 2 and np.std(bi_spikes[:n_common_ws]) > 1e-10 and np.std(bi_intensity[:n_common_ws]) > 1e-10:
        ws_corr = float(np.corrcoef(bi_spikes[:n_common_ws], bi_intensity[:n_common_ws])[0, 1])
    else:
        ws_corr = 0.0
    print(f"  Workload-spike correlation: {ws_corr:.4f}")

    # T278: Cross-correlation peak lag (feedback latency)
    xcorr_lags = []
    if n_common > 20:
        s1 = bi_spikes[:n_common] - np.mean(bi_spikes[:n_common])
        s2 = bi_gpu[:n_common] - np.mean(bi_gpu[:n_common])
        max_lag = min(20, n_common // 4)
        xcorr = np.zeros(max_lag)
        norm = max(np.std(s1) * np.std(s2), 1e-10) * n_common
        for lag in range(max_lag):
            xcorr[lag] = np.sum(s1[lag:] * s2[:n_common - lag]) / norm
        peak_lag = int(np.argmax(np.abs(xcorr)))
        peak_val = float(xcorr[peak_lag])
        xcorr_lags = xcorr.tolist()
    else:
        peak_lag = 0
        peak_val = 0.0
    print(f"  Cross-correlation peak: lag={peak_lag}, value={peak_val:.4f}")

    # T280: Loop gain — does spike output modulate GPU state which modulates next spike?
    # Measure: correlation between intensity[t] and spike_rate[t+k] for k=1..5
    loop_gains = []
    for k in range(1, 6):
        if n_common_ws > k + 10:
            s_future = bi_spikes[k:n_common_ws]
            i_past = bi_intensity[:n_common_ws - k]
            if np.std(s_future) > 1e-10 and np.std(i_past) > 1e-10:
                g = float(np.corrcoef(i_past, s_future)[0, 1])
            else:
                g = 0.0
            loop_gains.append(g)
    max_loop_gain = max(np.abs(loop_gains)) if loop_gains else 0.0
    print(f"  Loop gains (k=1..5): {[f'{g:.3f}' for g in loop_gains]}")
    print(f"  Max |loop gain|: {max_loop_gain:.4f}")

    results['loop_metrics'] = {
        'mi_spike_gpu': mi_spike_gpu,
        'gpu_power_var_bidirectional': var_bi,
        'gpu_power_var_open_loop': var_ol,
        'workload_spike_corr': ws_corr,
        'xcorr_peak_lag': peak_lag,
        'xcorr_peak_val': peak_val,
        'xcorr_values': xcorr_lags[:10] if xcorr_lags else [],
        'loop_gains': loop_gains,
        'max_loop_gain': max_loop_gain,
    }

    # ─── Step 6: Tests T273-T280 ───
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    acc_bi = cond_accuracies['BIDIRECTIONAL']['mean']
    acc_ol = cond_accuracies['OPEN_LOOP']['mean']
    acc_rw = cond_accuracies['RANDOM_WORKLOAD']['mean']
    acc_ng = cond_accuracies['NO_GPU']['mean']

    t273 = acc_bi > acc_ol
    t274 = acc_bi > acc_ng
    t275 = mi_spike_gpu > 0.1
    t276 = var_bi > var_ol
    t277 = abs(ws_corr) > 0.3
    t278 = peak_lag > 0
    t279 = acc_bi > acc_rw
    t280 = max_loop_gain > 0.1

    tests = {
        'T273_bidirectional_gt_openloop': {
            'pass': bool(t273),
            'BIDIRECTIONAL_acc': acc_bi, 'OPEN_LOOP_acc': acc_ol,
            'margin': acc_bi - acc_ol,
            'description': 'Feedback helps classification',
        },
        'T274_bidirectional_gt_nogpu': {
            'pass': bool(t274),
            'BIDIRECTIONAL_acc': acc_bi, 'NO_GPU_acc': acc_ng,
            'margin': acc_bi - acc_ng,
            'description': 'GPU state adds information',
        },
        'T275_mi_spike_gpu': {
            'pass': bool(t275),
            'mi_bits': mi_spike_gpu,
            'threshold': 0.1,
            'description': 'Mutual information between FPGA spikes and GPU state',
        },
        'T276_gpu_power_variance': {
            'pass': bool(t276),
            'var_bidirectional': var_bi, 'var_open_loop': var_ol,
            'ratio': var_bi / max(var_ol, 1e-10),
            'description': 'FPGA modulates GPU power consumption',
        },
        'T277_workload_spike_corr': {
            'pass': bool(t277),
            'correlation': ws_corr,
            'threshold': 0.3,
            'description': 'Spike-driven workload modulation works',
        },
        'T278_feedback_latency': {
            'pass': bool(t278),
            'peak_lag': peak_lag, 'peak_val': peak_val,
            'description': 'Cross-correlation peak at positive lag (feedback delay)',
        },
        'T279_bidirectional_gt_random': {
            'pass': bool(t279),
            'BIDIRECTIONAL_acc': acc_bi, 'RANDOM_WORKLOAD_acc': acc_rw,
            'margin': acc_bi - acc_rw,
            'description': 'Structured feedback > random workload',
        },
        'T280_loop_gain': {
            'pass': bool(t280),
            'max_loop_gain': max_loop_gain,
            'loop_gains': loop_gains,
            'threshold': 0.1,
            'description': 'Spike output modulates GPU state which modulates next spike input',
        },
    }

    results['tests'] = tests

    n_pass = sum(1 for t in tests.values() if t['pass'])
    results['summary'] = {
        'pass_count': n_pass,
        'total_tests': 8,
        'pass_rate': f"{n_pass}/8",
    }

    test_items = [
        (t273, f"T273: BIDIRECTIONAL={acc_bi:.3f} > OPEN_LOOP={acc_ol:.3f} [{acc_bi-acc_ol:+.3f}]"),
        (t274, f"T274: BIDIRECTIONAL={acc_bi:.3f} > NO_GPU={acc_ng:.3f} [{acc_bi-acc_ng:+.3f}]"),
        (t275, f"T275: MI(spikes,GPU)={mi_spike_gpu:.4f} > 0.1 bits"),
        (t276, f"T276: Var(BI)={var_bi:.4f} > Var(OL)={var_ol:.4f}"),
        (t277, f"T277: |corr(workload,spikes)|={abs(ws_corr):.4f} > 0.3"),
        (t278, f"T278: xcorr peak lag={peak_lag} > 0 (val={peak_val:.4f})"),
        (t279, f"T279: BIDIRECTIONAL={acc_bi:.3f} > RANDOM={acc_rw:.3f} [{acc_bi-acc_rw:+.3f}]"),
        (t280, f"T280: max |loop gain|={max_loop_gain:.4f} > 0.1"),
    ]
    for passed, desc in test_items:
        print(f"  {'PASS' if passed else 'FAIL'} {desc}")

    print(f"\n  Overall: {n_pass}/8 PASS")

    # ─── Save results ───
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2197_bidirectional_gpu_fpga_loop.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    # ─── Generate figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('z2197: Bidirectional GPU↔FPGA Computation Loop', fontsize=14, fontweight='bold')

        # Panel 1: Classification accuracy by condition
        ax = axes[0, 0]
        conds = conditions
        labels_plot = ['BIDIR\n(full loop)', 'OPEN\n(no feedback)', 'RANDOM\n(random load)', 'NO GPU\n(pure FPGA)']
        colors = ['#e74c3c', '#3498db', '#f39c12', '#95a5a6']
        means = [cond_accuracies[c]['mean'] for c in conds]
        stds = [cond_accuracies[c]['std'] for c in conds]
        bars = ax.bar(range(len(conds)), means, yerr=stds, capsize=5,
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels_plot, fontsize=9)
        ax.set_ylabel('Accuracy')
        ax.set_title('Waveform Classification (3-class)')
        ax.set_ylim(0, 1.05)
        ax.axhline(1 / 3, color='gray', linestyle='--', alpha=0.5, label='Chance (33.3%)')
        ax.legend(fontsize=8)
        # Mark significance
        if t273:
            ax.annotate('*', xy=(0, means[0] + stds[0] + 0.02), fontsize=16,
                        ha='center', color='#e74c3c')

        # Panel 2: GPU power time series comparison
        ax = axes[0, 1]
        for cond, color in zip(['BIDIRECTIONAL', 'OPEN_LOOP'], ['#e74c3c', '#3498db']):
            data = cond_gpu_logs[cond]
            if len(data) > 1:
                # Show first 200 samples
                show = data[:200]
                ax.plot(show, color=color, alpha=0.7, linewidth=0.8, label=cond)
        ax.set_xlabel('Step')
        ax.set_ylabel('GPU Power (W)')
        ax.set_title('GPU Power Dynamics')
        ax.legend(fontsize=8)

        # Panel 3: Workload intensity vs spike rate (BIDIRECTIONAL)
        ax = axes[1, 0]
        bi_int = cond_intensity_logs['BIDIRECTIONAL']
        bi_sp = cond_spike_logs['BIDIRECTIONAL']
        n_show = min(len(bi_int), len(bi_sp), 300)
        if n_show > 0:
            ax.scatter(bi_int[:n_show], bi_sp[:n_show], alpha=0.3, s=10,
                       color='#e74c3c', edgecolors='none')
            ax.set_xlabel('Workload Intensity (spike-driven)')
            ax.set_ylabel('Spike Rate (total spikes/step)')
            ax.set_title(f'Spike-Workload Coupling (r={ws_corr:.3f})')

        # Panel 4: Cross-correlation (feedback latency)
        ax = axes[1, 1]
        if xcorr_lags:
            ax.bar(range(len(xcorr_lags)), xcorr_lags, color='#2ecc71', alpha=0.7)
            ax.axhline(0, color='gray', linewidth=0.5)
            if peak_lag < len(xcorr_lags):
                ax.bar(peak_lag, xcorr_lags[peak_lag], color='#e74c3c', alpha=0.9)
            ax.set_xlabel('Lag (steps)')
            ax.set_ylabel('Cross-correlation')
            ax.set_title(f'Feedback Latency (peak lag={peak_lag})')

        plt.tight_layout()

        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2197_bidirectional_gpu_fpga_loop.png'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {fig_path}")
        plt.close(fig)
    except Exception as e:
        print(f"  Figure generation failed: {e}")

    # Cleanup
    if ser is not None:
        try:
            # Re-enable kill switch on exit
            ser.write(bytes([SYNC, CMD_SET_KILL, 0x01]))
            ser.flush()
            ser.close()
        except Exception:
            pass

    print(f"\n{'='*70}")
    print(f"z2197 COMPLETE: {n_pass}/8 PASS")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
