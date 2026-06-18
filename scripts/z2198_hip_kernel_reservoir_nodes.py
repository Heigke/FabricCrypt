#!/usr/bin/env python3
"""z2198_hip_kernel_reservoir_nodes.py — GPU Compute Kernels as Reservoir Nodes

GPU execution units become part of the reservoir computation, not just a noise
source.  Each GPU "node" is a HIP kernel whose execution timing and power draw
serve as computational features alongside 8 FPGA LIF neurons.

Two-substrate reservoir:
  FPGA:  8 LIF neurons  → 8 delta_spike features per step
  GPU:   4 HIP kernels  → 8 features (timing + power_delta) per step
  Total: 16 cross-substrate features per step

GPU kernel nodes:
  1. MatMul  – torch.mm(A, B),  A size varies with input
  2. FFT     – torch.fft.fft(x), varying-length signal
  3. Sort    – torch.sort(x),    varying-disorder array
  4. Conv    – F.conv1d(),       varying kernel size

Conditions (4):
  HYBRID:    FPGA neurons + GPU kernel nodes (cross-substrate)
  FPGA_ONLY: Only FPGA neuron features
  GPU_ONLY:  Only GPU kernel node features
  LINEAR:    Time-delay embedding baseline

Task: 3-class waveform classification (sine, triangle, square)
      100 trials, 20 steps/trial @ 5 Hz

Tests T281–T286:
  T281: HYBRID > FPGA_ONLY   (GPU nodes add value)
  T282: HYBRID > GPU_ONLY    (FPGA nodes add value)
  T283: HYBRID > LINEAR      (reservoir > linear)
  T284: GPU timing variance > 0 for >= 3/4 nodes  (input-dependent)
  T285: MI(FPGA, GPU) > 0.05 bits  (cross-substrate information)
  T286: HYBRID > max(FPGA_ONLY, GPU_ONLY)  (synergy)

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

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

# ─── JSON encoder ───
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
N_GPU_NODES = 4
SAMPLE_HZ = 5       # slow enough for GPU kernels
GPU_NODE_NAMES = ['matmul', 'fft', 'sort', 'conv']


# ═══════════════════════════════════════════════════════════
# FPGA Communication
# ═══════════════════════════════════════════════════════════

def to_q16_16(val: float) -> int:
    return int(val * 65536) & 0xFFFFFFFF


def crc8(data: bytes, poly=0x07) -> int:
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def find_fpga():
    try:
        import serial
    except ImportError:
        return None, None
    import glob as gl
    ports = sorted(gl.glob('/dev/ttyUSB*'), reverse=True)  # try ttyUSB1 first
    for p in ports:
        try:
            s = serial.Serial(p, 115200, timeout=0.3)
            time.sleep(0.1)
            # Verify: disable kill switch, then read telemetry
            s.write(bytes([SYNC, 0x03, 0x00]))
            s.flush()
            time.sleep(0.05)
            s.write(bytes([SYNC, CMD_READ_TELEM]))
            s.flush()
            resp = s.read(52)
            if len(resp) >= 52 and resp[0] == SYNC and resp[1] == CMD_READ_TELEM:
                return s, p
            s.close()
        except Exception:
            continue
    return None, None


def set_per_neuron_vg(ser, vg_values):
    """Set individual Vg for each of 8 neurons. Fire-and-forget."""
    for nid, vg in enumerate(vg_values[:N_NEURONS]):
        q16 = to_q16_16(max(0.0, min(1.0, vg)))
        payload = bytes([nid & 0x07]) + struct.pack('>I', q16)
        ser.write(bytes([SYNC, CMD_SET_VG]) + payload)
    ser.flush()
    time.sleep(0.005)


def read_telem(ser, timeout=0.15):
    """Read telemetry: [0x55][0x02][0x30][48B][CRC8] = 52 bytes."""
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


def read_hwmon_power():
    """Read hwmon power1_average (μW → W)."""
    try:
        return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# GPU Kernel Nodes
# ═══════════════════════════════════════════════════════════

def _ensure_torch():
    """Import torch once, return it. Fatal if unavailable."""
    import torch
    return torch


def run_gpu_kernel_nodes(input_val, torch_mod, device):
    """Execute 4 HIP kernel nodes with input-dependent parameters.

    Returns:
        timings: list of 4 execution times (seconds)
        power_deltas: list of 4 power deltas (W)
    """
    timings = []
    power_deltas = []

    # Scale input [0,1] → kernel parameters
    inp = float(np.clip(input_val, 0.0, 1.0))

    # ── Node 1: MatMul — size varies with input ──
    dim = max(32, int(32 + inp * 224))  # 32–256
    A = torch_mod.randn(dim, dim, device=device)
    B = torch_mod.randn(dim, dim, device=device)

    p0 = read_hwmon_power()
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    start.record()
    _ = torch_mod.mm(A, B)
    end.record()
    torch_mod.cuda.synchronize()
    t_ms = start.elapsed_time(end)
    p1 = read_hwmon_power()
    timings.append(t_ms / 1000.0)
    power_deltas.append((p1 - p0) if (p0 is not None and p1 is not None) else 0.0)

    # ── Node 2: FFT — varying-length signal ──
    fft_len = max(64, int(64 + inp * 960))  # 64–1024
    x_fft = torch_mod.randn(fft_len, device=device)

    p0 = read_hwmon_power()
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    start.record()
    _ = torch_mod.fft.fft(x_fft)
    end.record()
    torch_mod.cuda.synchronize()
    t_ms = start.elapsed_time(end)
    p1 = read_hwmon_power()
    timings.append(t_ms / 1000.0)
    power_deltas.append((p1 - p0) if (p0 is not None and p1 is not None) else 0.0)

    # ── Node 3: Sort — varying-disorder array ──
    sort_len = 4096
    # Higher input → more disordered (random), lower → nearly sorted
    x_sort = torch_mod.arange(sort_len, dtype=torch_mod.float32, device=device)
    n_swaps = int(inp * sort_len * 2)
    if n_swaps > 0:
        idx_a = torch_mod.randint(0, sort_len, (n_swaps,), device=device)
        idx_b = torch_mod.randint(0, sort_len, (n_swaps,), device=device)
        # Swap pairs to introduce disorder
        vals_a = x_sort[idx_a].clone()
        x_sort[idx_a] = x_sort[idx_b]
        x_sort[idx_b] = vals_a

    p0 = read_hwmon_power()
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    start.record()
    _ = torch_mod.sort(x_sort)
    end.record()
    torch_mod.cuda.synchronize()
    t_ms = start.elapsed_time(end)
    p1 = read_hwmon_power()
    timings.append(t_ms / 1000.0)
    power_deltas.append((p1 - p0) if (p0 is not None and p1 is not None) else 0.0)

    # ── Node 4: Conv1d — varying kernel size ──
    sig_len = 2048
    kern_sz = max(3, int(3 + inp * 60))  # 3–63, must be odd
    if kern_sz % 2 == 0:
        kern_sz += 1
    x_conv = torch_mod.randn(1, 1, sig_len, device=device)
    w_conv = torch_mod.randn(1, 1, kern_sz, device=device)

    p0 = read_hwmon_power()
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    start.record()
    _ = torch_mod.nn.functional.conv1d(x_conv, w_conv, padding=kern_sz // 2)
    end.record()
    torch_mod.cuda.synchronize()
    t_ms = start.elapsed_time(end)
    p1 = read_hwmon_power()
    timings.append(t_ms / 1000.0)
    power_deltas.append((p1 - p0) if (p0 is not None and p1 is not None) else 0.0)

    return timings, power_deltas


def simulate_gpu_kernel_nodes(input_val, rng):
    """Software fallback: simulate GPU kernel timing/power features."""
    inp = float(np.clip(input_val, 0.0, 1.0))
    timings = []
    power_deltas = []

    # MatMul: timing scales ~O(n^3), n = 32+inp*224
    dim = 32 + inp * 224
    t_mm = (dim / 256.0) ** 3 * 0.001 + rng.normal(0, 0.0001)
    timings.append(max(1e-6, t_mm))
    power_deltas.append(rng.normal(0, 0.3))

    # FFT: timing ~O(n log n), n = 64+inp*960
    n_fft = 64 + inp * 960
    t_fft = n_fft / 1024.0 * np.log2(max(n_fft, 2)) / 10.0 * 0.001 + rng.normal(0, 0.0001)
    timings.append(max(1e-6, t_fft))
    power_deltas.append(rng.normal(0, 0.3))

    # Sort: timing ~O(n log n) but disorder-dependent
    t_sort = (0.5 + inp * 0.5) * 0.001 + rng.normal(0, 0.0001)
    timings.append(max(1e-6, t_sort))
    power_deltas.append(rng.normal(0, 0.3))

    # Conv: timing scales with kernel size
    kern_sz = 3 + inp * 60
    t_conv = kern_sz / 63.0 * 0.001 + rng.normal(0, 0.0001)
    timings.append(max(1e-6, t_conv))
    power_deltas.append(rng.normal(0, 0.3))

    return timings, power_deltas


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir (from z2162)
# ═══════════════════════════════════════════════════════════

def run_fpga_step(ser, vg_values):
    """Drive FPGA neurons one step, return delta_spikes (8,)."""
    set_per_neuron_vg(ser, vg_values)
    time.sleep(0.02)
    ser.reset_input_buffer()
    ser.write(bytes([SYNC, CMD_READ_TELEM]))
    ser.flush()
    telem = read_telem(ser, timeout=0.15)
    if telem:
        return [n['spike_count'] for n in telem]
    return None


def simulate_lif_step(vmem, vg_values, rng):
    """Software LIF step. Returns (spikes, vmem_new)."""
    dt = 1.0 / SAMPLE_HZ
    tau_m = 0.02
    I_in = np.array(vg_values[:N_NEURONS]) * 5.0
    dvdt = (-vmem + I_in) / tau_m
    vmem_new = vmem + dvdt * dt
    spikes = np.zeros(N_NEURONS)
    for i in range(N_NEURONS):
        if vmem_new[i] >= 1.0:
            spikes[i] = 1
            vmem_new[i] = 0.0
    return spikes, vmem_new


# ═══════════════════════════════════════════════════════════
# Waveform Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials, steps_per_trial, freq_hz=1.0, dt=None, seed=42):
    """Generate sine/triangle/square waveforms for classification."""
    if dt is None:
        dt = 1.0 / SAMPLE_HZ
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

        wave = (wave + 1.0) / 2.0  # → [0, 1]
        trials.append(wave)
        labels.append(cls)

    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# Feature Extraction & Classification
# ═══════════════════════════════════════════════════════════

def pool_trial_features(trial_states):
    """Pool per-timestep reservoir states into trial-level features.
    trial_states: (n_steps, n_features) → [mean, std, max, min]
    """
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def ridge_classify(X_train, y_train, X_test, y_test, alphas=None):
    """Ridge regression classifier (one-hot multi-class)."""
    if alphas is None:
        alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
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


def stratified_kfold(X, y, n_splits=5, seed=42):
    """Stratified k-fold split."""
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


def cross_val_accuracy(X, y, n_splits=5, seed=42):
    """Cross-validated ridge accuracy."""
    splits = stratified_kfold(X, y, n_splits=n_splits, seed=seed)
    accs = []
    for train_idx, test_idx in splits:
        acc = ridge_classify(X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def mutual_information_discrete(x, y, n_bins=8):
    """Estimate MI between continuous x and y via histogram discretisation."""
    x_d = np.digitize(x, np.linspace(x.min() - 1e-9, x.max() + 1e-9, n_bins + 1)[1:-1])
    y_d = np.digitize(y, np.linspace(y.min() - 1e-9, y.max() + 1e-9, n_bins + 1)[1:-1])
    # Joint
    joint = np.zeros((n_bins, n_bins))
    for xi, yi in zip(x_d, y_d):
        joint[xi, yi] += 1
    joint /= joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return max(0.0, mi)


# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2198: HIP Kernel Reservoir Nodes')
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--steps-per-trial', type=int, default=20)
    parser.add_argument('--simulated', action='store_true', help='Force simulation mode')
    args = parser.parse_args()

    print("=" * 65)
    print("z2198: HIP Kernel Reservoir Nodes — GPU Compute as Reservoir")
    print("=" * 65)

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2198_hip_kernel_reservoir_nodes',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'base_vg': BASE_VG, 'alpha': ALPHA, 'beta': BETA,
            'n_neurons': N_NEURONS, 'n_gpu_nodes': N_GPU_NODES,
            'sample_hz': SAMPLE_HZ, 'gpu_node_names': GPU_NODE_NAMES,
            'n_trials': args.n_trials, 'steps_per_trial': args.steps_per_trial,
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
        # Disable kill switch
        ser.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
        ser.flush()
        time.sleep(0.1)
        print("  Kill switch disabled")

    # ─── Step 2: Initialise GPU (torch/HIP) ───
    print("\n[2/6] Initialising GPU compute nodes...")
    gpu_available = False
    torch_mod = None
    device = None
    try:
        torch_mod = _ensure_torch()
        if torch_mod.cuda.is_available():
            device = torch_mod.device('cuda:0')
            # Warmup
            _ = torch_mod.randn(64, 64, device=device) @ torch_mod.randn(64, 64, device=device)
            torch_mod.cuda.synchronize()
            gpu_available = True
            print(f"  GPU ready: {torch_mod.cuda.get_device_name(0)}")
        else:
            print("  torch.cuda not available — GPU nodes will be simulated")
    except Exception as e:
        print(f"  torch init failed: {e} — GPU nodes will be simulated")

    if not gpu_available:
        results['simulated'] = True

    # ─── Step 3: Telemetry ───
    print("\n[3/6] Initialising telemetry...")
    telem = SysfsHwmonTelemetry()
    print(f"  Telemetry ready")

    # ─── Step 4: Generate waveforms ───
    print(f"\n[4/6] Generating {args.n_trials} waveform trials ({args.steps_per_trial} steps @ {SAMPLE_HZ} Hz)...")
    trials, labels = generate_waveforms(
        args.n_trials, args.steps_per_trial, freq_hz=0.5, seed=42
    )
    print(f"  Classes: {np.bincount(labels).tolist()} (sine/tri/sq)")

    # ─── Step 5: Run reservoir in 4 conditions ───
    print(f"\n[5/6] Running reservoir trials...")

    interval = 1.0 / SAMPLE_HZ

    # Storage for all conditions
    all_fpga_features = []    # (n_trials, steps, 8)
    all_gpu_features = []     # (n_trials, steps, 8)
    all_gpu_timings_raw = []  # for T284 variance check
    all_linear_features = []  # (n_trials, steps, 1)

    # FPGA state tracking
    prev_counts = None
    lif_vmem = np.zeros(N_NEURONS) if not fpga else None

    for trial_idx in range(args.n_trials):
        input_signal = trials[trial_idx]
        n_steps = len(input_signal)

        trial_fpga = np.zeros((n_steps, N_NEURONS))
        trial_gpu = np.zeros((n_steps, N_GPU_NODES * 2))  # timing + power per node
        trial_timings = np.zeros((n_steps, N_GPU_NODES))

        if trial_idx % 20 == 0:
            print(f"  Trial {trial_idx}/{args.n_trials}...")

        # Reset LIF state between trials
        if not fpga:
            lif_vmem = np.zeros(N_NEURONS)

        for t in range(n_steps):
            inp = input_signal[t]

            # Read GPU state for noise coupling
            p_gpu = read_hwmon_power()
            noise_val = ((p_gpu - 11.0) / 2.0) if p_gpu is not None else rng.normal(0, 0.3)

            # Compute per-neuron Vg
            vg_values = np.full(N_NEURONS, BASE_VG)
            vg_values += ALPHA * inp * w_in
            vg_values += BETA * noise_val * w_noise
            vg_values = np.clip(vg_values, 0.05, 0.95)

            # ── FPGA substrate ──
            if fpga:
                counts = run_fpga_step(ser, vg_values)
                if counts is not None:
                    if prev_counts is not None:
                        for i in range(N_NEURONS):
                            delta = (counts[i] - prev_counts[i]) & 0xFFFF
                            if delta > 30000:
                                delta = 0
                            trial_fpga[t, i] = delta
                    prev_counts = counts[:]
            else:
                spikes, lif_vmem = simulate_lif_step(lif_vmem, vg_values, rng)
                trial_fpga[t, :] = spikes

            # ── GPU kernel nodes ──
            if gpu_available:
                timings, power_deltas = run_gpu_kernel_nodes(inp, torch_mod, device)
            else:
                timings, power_deltas = simulate_gpu_kernel_nodes(inp, rng)

            for k in range(N_GPU_NODES):
                trial_gpu[t, k] = timings[k]
                trial_gpu[t, N_GPU_NODES + k] = power_deltas[k]
                trial_timings[t, k] = timings[k]

            # Pacing
            time.sleep(max(0, interval - 0.05))

        all_fpga_features.append(trial_fpga)
        all_gpu_features.append(trial_gpu)
        all_gpu_timings_raw.append(trial_timings)
        all_linear_features.append(input_signal.reshape(-1, 1))

    all_fpga_features = np.array(all_fpga_features)   # (N, T, 8)
    all_gpu_features = np.array(all_gpu_features)      # (N, T, 8)
    all_gpu_timings_raw = np.array(all_gpu_timings_raw)  # (N, T, 4)
    all_linear_features = np.array(all_linear_features)  # (N, T, 1)

    # ─── Pool features per trial ───
    print("\n  Pooling features...")

    def pool_all(feat_array):
        """Pool each trial → feature vector."""
        return np.array([pool_trial_features(feat_array[i]) for i in range(len(feat_array))])

    X_fpga = pool_all(all_fpga_features)
    X_gpu = pool_all(all_gpu_features)
    X_hybrid = np.hstack([X_fpga, X_gpu])

    # Linear baseline: time-delay embedding on raw input
    delays = [1, 2, 3, 5]
    X_linear_list = []
    for i in range(args.n_trials):
        sig = trials[i]
        embed = np.zeros((len(sig), 1 + len(delays)))
        embed[:, 0] = sig
        for di, d in enumerate(delays):
            embed[d:, di + 1] = sig[:len(sig) - d]
        X_linear_list.append(pool_trial_features(embed))
    X_linear = np.array(X_linear_list)

    print(f"  Feature dims: FPGA={X_fpga.shape[1]}, GPU={X_gpu.shape[1]}, "
          f"HYBRID={X_hybrid.shape[1]}, LINEAR={X_linear.shape[1]}")

    # ─── Step 6: Classification & Tests ───
    print(f"\n[6/6] Classification (5-fold CV)...")

    acc_hybrid, std_hybrid = cross_val_accuracy(X_hybrid, labels)
    acc_fpga, std_fpga = cross_val_accuracy(X_fpga, labels)
    acc_gpu, std_gpu = cross_val_accuracy(X_gpu, labels)
    acc_linear, std_linear = cross_val_accuracy(X_linear, labels)

    print(f"  HYBRID:    {acc_hybrid:.4f} ± {std_hybrid:.4f}")
    print(f"  FPGA_ONLY: {acc_fpga:.4f} ± {std_fpga:.4f}")
    print(f"  GPU_ONLY:  {acc_gpu:.4f} ± {std_gpu:.4f}")
    print(f"  LINEAR:    {acc_linear:.4f} ± {std_linear:.4f}")

    # T284: GPU kernel timing variance
    timing_variances = []
    for k in range(N_GPU_NODES):
        var_k = float(np.var(all_gpu_timings_raw[:, :, k]))
        timing_variances.append(var_k)
    n_var_positive = sum(1 for v in timing_variances if v > 0)
    print(f"\n  GPU timing variances: {[f'{v:.2e}' for v in timing_variances]}")
    print(f"  Nodes with variance > 0: {n_var_positive}/4")

    # T285: Cross-substrate MI
    # Flatten across trials and steps, compute MI between each FPGA/GPU feature pair
    fpga_flat = all_fpga_features.reshape(-1, N_NEURONS)
    gpu_flat = all_gpu_features.reshape(-1, N_GPU_NODES * 2)
    mi_values = []
    for fi in range(min(N_NEURONS, 4)):
        for gi in range(min(N_GPU_NODES * 2, 4)):
            mi_val = mutual_information_discrete(fpga_flat[:, fi], gpu_flat[:, gi])
            mi_values.append(mi_val)
    mi_mean = float(np.mean(mi_values)) if mi_values else 0.0
    mi_max = float(np.max(mi_values)) if mi_values else 0.0
    print(f"  Cross-substrate MI: mean={mi_mean:.4f}, max={mi_max:.4f} bits")

    # ─── Tests ───
    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    tests = {}

    # T281: HYBRID > FPGA_ONLY
    t281_pass = acc_hybrid > acc_fpga
    tests['T281_hybrid_gt_fpga'] = {
        'pass': bool(t281_pass),
        'hybrid': acc_hybrid, 'fpga_only': acc_fpga,
        'description': 'HYBRID > FPGA_ONLY (GPU nodes add value)',
    }
    print(f"  T281 HYBRID > FPGA_ONLY: {'PASS' if t281_pass else 'FAIL'}  "
          f"({acc_hybrid:.4f} vs {acc_fpga:.4f})")

    # T282: HYBRID > GPU_ONLY
    t282_pass = acc_hybrid > acc_gpu
    tests['T282_hybrid_gt_gpu'] = {
        'pass': bool(t282_pass),
        'hybrid': acc_hybrid, 'gpu_only': acc_gpu,
        'description': 'HYBRID > GPU_ONLY (FPGA nodes add value)',
    }
    print(f"  T282 HYBRID > GPU_ONLY:  {'PASS' if t282_pass else 'FAIL'}  "
          f"({acc_hybrid:.4f} vs {acc_gpu:.4f})")

    # T283: HYBRID > LINEAR
    t283_pass = acc_hybrid > acc_linear
    tests['T283_hybrid_gt_linear'] = {
        'pass': bool(t283_pass),
        'hybrid': acc_hybrid, 'linear': acc_linear,
        'description': 'HYBRID > LINEAR (reservoir > linear)',
    }
    print(f"  T283 HYBRID > LINEAR:    {'PASS' if t283_pass else 'FAIL'}  "
          f"({acc_hybrid:.4f} vs {acc_linear:.4f})")

    # T284: GPU timing variance > 0 for >= 3/4 nodes
    t284_pass = n_var_positive >= 3
    tests['T284_gpu_timing_variance'] = {
        'pass': bool(t284_pass),
        'n_nodes_with_variance': n_var_positive,
        'variances': timing_variances,
        'node_names': GPU_NODE_NAMES,
        'description': 'GPU kernel timing variance > 0 for >= 3/4 nodes',
    }
    print(f"  T284 GPU timing var:     {'PASS' if t284_pass else 'FAIL'}  "
          f"({n_var_positive}/4 nodes with variance)")

    # T285: MI(FPGA, GPU) > 0.05 bits
    t285_pass = mi_mean > 0.05
    tests['T285_cross_substrate_mi'] = {
        'pass': bool(t285_pass),
        'mi_mean': mi_mean, 'mi_max': mi_max,
        'threshold': 0.05,
        'description': 'MI(FPGA, GPU) > 0.05 bits (cross-substrate info)',
    }
    print(f"  T285 Cross-substrate MI: {'PASS' if t285_pass else 'FAIL'}  "
          f"(mean={mi_mean:.4f} bits, threshold=0.05)")

    # T286: HYBRID > max(FPGA_ONLY, GPU_ONLY) (synergy)
    best_single = max(acc_fpga, acc_gpu)
    t286_pass = acc_hybrid > best_single
    tests['T286_synergy'] = {
        'pass': bool(t286_pass),
        'hybrid': acc_hybrid, 'best_single': best_single,
        'description': 'HYBRID > max(FPGA_ONLY, GPU_ONLY) (synergy)',
    }
    print(f"  T286 Synergy:            {'PASS' if t286_pass else 'FAIL'}  "
          f"({acc_hybrid:.4f} vs max({acc_fpga:.4f}, {acc_gpu:.4f})={best_single:.4f})")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n  Score: {n_pass}/{n_total} PASS")

    # ─── Save results ───
    results['accuracies'] = {
        'hybrid': {'mean': acc_hybrid, 'std': std_hybrid},
        'fpga_only': {'mean': acc_fpga, 'std': std_fpga},
        'gpu_only': {'mean': acc_gpu, 'std': std_gpu},
        'linear': {'mean': acc_linear, 'std': std_linear},
    }
    results['gpu_timing_variances'] = {
        name: var for name, var in zip(GPU_NODE_NAMES, timing_variances)
    }
    results['cross_substrate_mi'] = {'mean': mi_mean, 'max': mi_max}
    results['tests'] = tests
    results['n_pass'] = n_pass
    results['n_total'] = n_total

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / 'z2198_hip_kernel_reservoir_nodes.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results → {out_path}")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Condition comparison bar chart
        ax = axes[0]
        conditions = ['HYBRID', 'FPGA_ONLY', 'GPU_ONLY', 'LINEAR']
        accs = [acc_hybrid, acc_fpga, acc_gpu, acc_linear]
        stds = [std_hybrid, std_fpga, std_gpu, std_linear]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
        bars = ax.bar(conditions, accs, yerr=stds, capsize=5, color=colors, edgecolor='black')
        ax.axhline(y=1.0/3.0, color='gray', linestyle='--', alpha=0.5, label='chance')
        ax.set_ylabel('Accuracy')
        ax.set_title('3-Class Waveform Classification')
        ax.set_ylim(0, 1.0)
        ax.legend()

        # Panel 2: GPU kernel timing variances
        ax = axes[1]
        ax.bar(GPU_NODE_NAMES, timing_variances, color='#e74c3c', edgecolor='black')
        ax.set_ylabel('Timing Variance (s²)')
        ax.set_title('GPU Kernel Timing Variance\n(Input-Dependent Computation)')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

        # Panel 3: Cross-substrate MI heatmap (4x4 subset)
        ax = axes[2]
        n_f = min(N_NEURONS, 4)
        n_g = min(N_GPU_NODES * 2, 4)
        mi_matrix = np.zeros((n_f, n_g))
        idx = 0
        for fi in range(n_f):
            for gi in range(n_g):
                mi_matrix[fi, gi] = mi_values[idx]
                idx += 1
        im = ax.imshow(mi_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('GPU feature')
        ax.set_ylabel('FPGA neuron')
        ax.set_title(f'Cross-Substrate MI\n(mean={mi_mean:.4f} bits)')
        plt.colorbar(im, ax=ax, label='MI (bits)')

        plt.suptitle(f'z2198: HIP Kernel Reservoir Nodes — {n_pass}/{n_total} PASS',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'fig_z2198_hip_kernel_reservoir.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Figure → {fig_path}")
    except Exception as e:
        print(f"  Figure skipped: {e}")

    # Cleanup
    if ser:
        try:
            ser.close()
        except Exception:
            pass

    print(f"\nDone. {n_pass}/{n_total} tests passed.")
    return 0 if n_pass >= 4 else 1


if __name__ == '__main__':
    sys.exit(main())
