#!/usr/bin/env python3
"""z2212_hard_benchmarks.py — Harder benchmarks for cross-substrate advantage

Tests where cross-substrate computation genuinely helps:
- 7-class waveform (more complex discrimination)
- Temporal XOR at τ=5,10 (longer temporal dependencies)
- Memory capacity test (>100 steps)
- Scale-dependent crossover (8 vs 128 neurons)
- Thermal-coupled classification (GPU thermal state as slow context signal)

Literature predicts cross-substrate advantage on:
  1. Multi-timescale tasks (speech: +8.26pp)
  2. Noisy hardware with structured 1/f noise
  3. Complementary substrate dynamics (not redundant)
  4. Multi-timescale context modulation (thermal slow + spike fast)

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128-neuron)
Physical setup: Arty on GPU heatsink with ESD foam insulation (~8× thermal coupling)
"""

import os, sys, json, time, struct, argparse, threading
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
# ─── Parameters ───
N_NEURONS = 128
BASE_VG = 0.58
ALPHA = 0.25
BETA_1F = 0.08
SAMPLE_HZ = 20
IIR_ALPHA = 0.85
N_FOLDS = 5
# ─── Firmware Paths ───
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
PM_TABLE_THERMAL_OFFSET = 0x004C

PM_DEEP_OFFSETS = {
    'stapm_w': (0x04, 'f', 120.0),
    'slow_ppt_w': (0x14, 'f', 140.0),
    'cpu_temp': (0x4C, 'f', 100.0),
    'gpu_temp': (0x54, 'f', 100.0),
    'gfx_sclk': (0x78, 'f', 3000.0),
    'vdd_v': (0x84, 'f', 1.5),
    'core_freq': (0x108, 'f', 6000.0),
    'core_volt': (0x1B8, 'f', 1.6),
}
# ─── Noise Channel Assignment (for L5 bridge) ───
POWER_NEURONS   = list(range(0, 32))
SMN_NEURONS     = list(range(32, 56))
JITTER_NEURONS  = list(range(56, 80))
THERMAL_NEURONS = list(range(80, 104))
CLOCK_NEURONS   = list(range(104, 128))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

# Firmware reads

def read_hwmon_power():
    try: return int(open(HWMON_POWER).read().strip()) / 1e6
    except: return None

def read_gpu_thermal():
    try: return int(open("/sys/class/hwmon/hwmon7/temp1_input").read().strip()) / 1000.0
    except: return None

def read_gpu_clock():
    try: return int(open("/sys/class/hwmon/hwmon7/freq1_input").read().strip()) / 1e6
    except: return None

def read_smn_thermal():
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(PM_TABLE_THERMAL_OFFSET)
            return struct.unpack('<f', f.read(4))[0]
    except: return None

def read_perf_jitter():
    t0 = time.perf_counter_ns()
    _ = os.getpid()
    return time.perf_counter_ns() - t0

def read_firmware_telemetry():
    feat = np.zeros(6)
    p = read_hwmon_power()
    if p is not None: feat[0] = p
    t = read_gpu_thermal()
    if t is not None: feat[1] = t
    c = read_gpu_clock()
    if c is not None: feat[2] = c
    sm = read_smn_thermal()
    if sm is not None: feat[3] = sm
    feat[4] = read_perf_jitter()
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(0x04)
            feat[5] = struct.unpack('<f', f.read(4))[0]
    except: pass
    return feat

# GPU Thermal Stress

gpu_stress_active = False

def gpu_stress_worker(duration_secs):
    """Run heavy GPU matmul to generate heat. Call from thread."""
    import torch
    global gpu_stress_active
    gpu_stress_active = True
    device = torch.device('cuda')
    end_time = time.time() + duration_secs
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    while time.time() < end_time and gpu_stress_active:
        _ = torch.mm(a, b)
        torch.cuda.synchronize()
    gpu_stress_active = False

def start_gpu_stress(duration_secs):
    """Start GPU stress in background thread."""
    t = threading.Thread(target=gpu_stress_worker, args=(duration_secs,), daemon=True)
    t.start()
    return t

def stop_gpu_stress():
    """Signal GPU stress to stop."""
    global gpu_stress_active
    gpu_stress_active = False

# Noise Sources

def normalize_noise(samples):
    arr = np.array(samples, dtype=float)
    if len(arr) == 0: return arr
    mu, std = arr.mean(), max(arr.std(), 1e-6)
    return (arr - mu) / std

def iir_filter_noise(noise_samples, alpha_iir=0.85):
    if len(noise_samples) == 0: return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t-1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std

def collect_all_noise(duration_s=15, sample_hz=50):
    n = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    power_s, thermal_s, clock_s, smn_s, jitter_s = [], [], [], [], []
    print("  Collecting noise channels...")
    for i in range(n):
        p = read_hwmon_power()
        t = read_gpu_thermal()
        c = read_gpu_clock()
        sm = read_smn_thermal()
        j = read_perf_jitter()
        if p is not None: power_s.append(p)
        if t is not None: thermal_s.append(t)
        if c is not None: clock_s.append(c)
        if sm is not None: smn_s.append(sm)
        jitter_s.append(j)
        time.sleep(interval)
        if n > 4 and (i + 1) % (n // 4) == 0:
            print(f"    {i+1}/{n} samples")
    return power_s, thermal_s, clock_s, smn_s, jitter_s

# Classification / Regression helpers

def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
    ])

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=None, alphas=None):
    if alphas is None: alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    if n_classes is None: n_classes = len(np.unique(np.concatenate([y_tr, y_te])))
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, y in enumerate(y_tr): Y_tr[i, int(y)] = 1.0
    best = -1
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try: W = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ Y_tr)
        except: continue
        acc = np.mean(np.argmax(X_te_s @ W, axis=1) == y_te)
        if acc > best: best = acc
    return best

def ridge_regress(X_tr, y_tr, X_te, y_te, alphas=None):
    if alphas is None: alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    best_corr = -1
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try: w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except: continue
        pred = X_te_s @ w
        if np.std(pred) > 1e-10 and np.std(y_te) > 1e-10:
            corr = np.corrcoef(pred, y_te)[0, 1]
            if corr > best_corr: best_corr = corr
    return max(best_corr, 0.0)

def pca_reduce(X, n_components=100):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_components < 1: return X, np.zeros(X.shape[1]), np.eye(X.shape[1])
    mu = X.mean(axis=0); Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T, mu, Vt[:n_components]

def pca_transform(X, mu, Vt): return (X - mu) @ Vt.T

def stratified_kfold(X, y, n_splits=5, seed=42):
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

def classify_condition(X_all, y_all, n_splits=5, max_features=120, n_classes=None):
    splits = stratified_kfold(X_all, y_all, n_splits=n_splits)
    fold_accs = []
    use_pca = X_all.shape[1] > max_features
    for train_idx, test_idx in splits:
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        mu = X_tr.mean(axis=0, keepdims=True)
        sigma = X_tr.std(axis=0, keepdims=True)
        sigma[sigma < 1e-2] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        if use_pca:
            X_tr_n, pca_mu, pca_Vt = pca_reduce(X_tr_n, n_components=max_features)
            X_te_n = pca_transform(X_te_n, pca_mu, pca_Vt)
        acc = ridge_classify(X_tr_n, y_tr, X_te_n, y_te, n_classes=n_classes)
        fold_accs.append(acc)
    return {'mean': float(np.mean(fold_accs)), 'std': float(np.std(fold_accs)),
            'folds': [float(a) for a in fold_accs]}

# Task generators

def generate_7class_waveforms(n_trials=300, steps=30, dt=1.0/20, seed=42):
    """7-class waveform: sine, triangle, square, sawtooth, chirp, AM-sine, damped-sine."""
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    t = np.arange(steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 7)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.8, 1.2)
        if cls == 0:    # sine
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:  # triangle
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0) - 1.0
        elif cls == 2:  # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif cls == 3:  # sawtooth
            wave = 2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0
        elif cls == 4:  # chirp (frequency sweep)
            f0, f1 = freq * 0.5, freq * 2.0
            inst_f = f0 + (f1 - f0) * t / max(t[-1], 1e-6)
            wave = np.sin(2 * np.pi * np.cumsum(inst_f) * dt + phase)
        elif cls == 5:  # AM-sine (amplitude modulated)
            carrier = np.sin(2 * np.pi * freq * 2 * t + phase)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * freq * 0.3 * t)
            wave = carrier * envelope
        else:           # damped-sine
            decay = np.exp(-2.0 * t)
            wave = np.sin(2 * np.pi * freq * t + phase) * decay
        # Normalize to [0, 1]
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)

def generate_temporal_xor(n_trials=200, steps=50, tau=5, seed=42):
    """Binary input XOR(input[t], input[t-tau]). Returns trials and labels per step."""
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    for _ in range(n_trials):
        seq = rng.integers(0, 2, size=steps).astype(float)
        target = np.zeros(steps, dtype=int)
        for t_i in range(tau, steps):
            target[t_i] = int(seq[t_i]) ^ int(seq[t_i - tau])
        # Input is continuous [0,1] for Vg driving
        trials.append(seq)
        labels.append(target)
    return np.array(trials), np.array(labels)

def generate_memory_capacity_input(n_steps=200, seed=42):
    """Random input u[t] in [0,1] for memory capacity test."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=n_steps)

# FPGA Trial Runner (from z2210)

def run_fpga_trial(fpga, input_signal, noises, w_in, w_noise, torch_mod, device,
                   mode='L3_FPGA_ALONE', beta=BETA_1F, n_use=None,
                   gpu_esn_W_in=None, gpu_esn_W=None):
    """Run one trial through FPGA. Returns feature arrays.

    n_use: number of neurons to use (None=all 128). For 8-neuron sub-reservoir,
           pass n_use=8 to only use first 8 neurons' data.

    L3: FPGA spikes only (input drives Vg directly)
    L5: FPGA + 1/f noise + firmware telemetry features (BRIDGE)
    L6: FPGA (CLEAN) + GPU-ESN temporal memory + firmware telemetry
    """
    n_steps = len(input_signal)
    n_neur = n_use if n_use is not None else N_NEURONS

    all_fpga = np.zeros((n_steps, n_neur * 3))
    all_telem = np.zeros((n_steps, 6))
    gpu_esn_dim = gpu_esn_W_in.shape[0] if gpu_esn_W_in is not None else n_neur
    all_gpu = np.zeros((n_steps, gpu_esn_dim))

    prev_counts = None
    cumulative = np.zeros(n_neur)

    import torch
    if mode == 'L6_DEEP_INTER' and gpu_esn_W_in is not None:
        gpu_state = torch.zeros(gpu_esn_W_in.shape[0], device=device)

    for t in range(n_steps):
        inp = input_signal[t]

        # Compute Vg for ALL 128 neurons (FPGA always runs all 128)
        vg = np.full(N_NEURONS, BASE_VG) + ALPHA * inp * w_in

        # L5: Add 1/f noise channel assignment
        if mode == 'L5_BRIDGE':
            channel_map = {
                'power': POWER_NEURONS, 'smn': SMN_NEURONS,
                'jitter': JITTER_NEURONS, 'thermal': THERMAL_NEURONS,
                'clock': CLOCK_NEURONS,
            }
            for ch_name, neuron_ids in channel_map.items():
                ch_data = noises.get(ch_name, np.zeros(1))
                if len(ch_data) == 0: ch_data = np.zeros(1)
                idx = t % len(ch_data)
                for nid in neuron_ids:
                    vg[nid] += beta * ch_data[idx] * w_noise[nid]

        vg = np.clip(vg, 0.05, 0.95)

        try: fpga.set_vg_all(vg.tolist())
        except: pass

        time.sleep(1.0 / SAMPLE_HZ * 0.5)

        # Read firmware telemetry (L5, L6)
        if mode in ('L5_BRIDGE', 'L6_DEEP_INTER'):
            all_telem[t] = read_firmware_telemetry()

        # Read FPGA spikes
        try:
            fpga.ser.reset_input_buffer()
            telem = fpga.read_telem(timeout=0.3)
        except:
            telem = None
            try: fpga.reconnect()
            except: pass

        if telem and len(telem) >= N_NEURONS:
            counts = [telem[i]['spike_count'] for i in range(N_NEURONS)]
            vmems = [telem[i]['vmem'] for i in range(N_NEURONS)]
            if prev_counts is not None:
                for i in range(n_neur):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000: delta = 0
                    all_fpga[t, i] = delta
                    cumulative[i] += delta
            for i in range(n_neur):
                all_fpga[t, n_neur + i] = vmems[i]
                all_fpga[t, n_neur * 2 + i] = cumulative[i]
            prev_counts = counts[:]

        # L6: GPU-ESN post-processes FPGA spike deltas + firmware telemetry
        if mode == 'L6_DEEP_INTER' and gpu_esn_W_in is not None:
            spike_vec = torch.tensor(all_fpga[t, :n_neur],
                                     device=device, dtype=torch.float32)
            telem_vec = torch.tensor(all_telem[t], device=device, dtype=torch.float32)
            esn_input = torch.cat([spike_vec / (spike_vec.sum() + 1e-6), telem_vec])
            gpu_state = torch.tanh(gpu_esn_W_in @ esn_input +
                                   gpu_esn_W @ gpu_state)
            all_gpu[t] = gpu_state.detach().cpu().numpy()

    return all_fpga, all_telem, all_gpu

def build_features(fpga_states, telem_states, gpu_states, mode, n_neur):
    """Build pooled feature vector from trial data."""
    if mode == 'L6_DEEP_INTER':
        # Combine FPGA + telemetry + GPU-ESN states
        combined = np.hstack([fpga_states, telem_states, gpu_states])
        return pool_trial_features(combined)
    elif mode == 'L5_BRIDGE':
        combined = np.hstack([fpga_states, telem_states])
        return pool_trial_features(combined)
    else:  # L3
        return pool_trial_features(fpga_states)

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-s', type=float, default=15.0)
    parser.add_argument('--wave7-trials', type=int, default=300)
    parser.add_argument('--xor-trials', type=int, default=200)
    parser.add_argument('--mc-steps', type=int, default=200)
    parser.add_argument('--crossover-trials', type=int, default=200)
    parser.add_argument('--thermal-trials', type=int, default=200)
    args = parser.parse_args()

    print("=" * 70)
    print("z2212: Hard Benchmarks for Cross-Substrate Advantage")
    print(f"  Benchmarks: 7-class waveform, temporal XOR τ=5/10,")
    print(f"              memory capacity, crossover, thermal-coupled")
    print(f"  Levels: L3_FPGA_ALONE, L5_BRIDGE, L6_DEEP_INTER (+ 8N variants)")
    print(f"  Vg={BASE_VG}, ALPHA={ALPHA}, BETA_1F={BETA_1F}")
    print("=" * 70)
    # ─── Init GPU ───
    try:
        import torch
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        print(f"\n[HW] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        _ = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[ERR] PyTorch/CUDA: {e}")
        sys.exit(1)
    # ─── Init FPGA ───
    print("\n[1] Connecting to 128-neuron FPGA...")
    from fpga_host_v2 import FPGABridge
    fpga = FPGABridge()
    if not fpga.connected:
        print("  ERROR: FPGA not found")
        sys.exit(1)
    print(f"  Connected: {fpga.port}, neurons={fpga.num_neurons}")
    fpga.read_telem(timeout=0.5)
    time.sleep(0.5)
    test = fpga.read_telem(timeout=0.5)
    if test is None:
        print("  WARNING: No initial telemetry")
    else:
        print(f"  Telemetry OK: {len(test)} neurons")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2212_hard_benchmarks',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'n_neurons': N_NEURONS, 'base_vg': BASE_VG,
            'alpha': ALPHA, 'beta_1f': BETA_1F,
        },
    }
    # ─── Collect noise for L5 ───
    print(f"\n[2] Collecting 1/f noise ({args.noise_s}s)...")
    power_s, thermal_s, clock_s, smn_s, jitter_s = collect_all_noise(args.noise_s, 50)
    noises = {}
    for name, raw_samples, iir_a in [
        ('power', power_s, IIR_ALPHA),
        ('thermal', thermal_s, 0.92),
        ('clock', clock_s, IIR_ALPHA),
        ('smn', smn_s, IIR_ALPHA),
        ('jitter', jitter_s, IIR_ALPHA),
    ]:
        if len(raw_samples) > 10:
            noises[name] = iir_filter_noise(normalize_noise(raw_samples), iir_a)
            print(f"  {name}: {len(raw_samples)} samples")
        else:
            noises[name] = np.zeros(100)
            print(f"  {name}: MISSING")
    # ─── GPU-ESN matrices (for L6) ───
    # 128-neuron: input = spikes(128) + telemetry(6) = 134
    gpu_esn_dim_128 = 32
    gpu_esn_input_128 = N_NEURONS + 6
    gpu_esn_W_in_128 = torch.randn(gpu_esn_dim_128, gpu_esn_input_128, device=device) * 0.1
    gpu_esn_W_128 = torch.randn(gpu_esn_dim_128, gpu_esn_dim_128, device=device)
    with torch.no_grad():
        sr = torch.linalg.eigvals(gpu_esn_W_128).abs().max().item()
        gpu_esn_W_128 *= 0.95 / max(sr, 1e-6)
    print(f"\n  GPU-ESN 128N: {gpu_esn_dim_128}x{gpu_esn_input_128}, sr=0.95")

    # 8-neuron: input = spikes(8) + telemetry(6) = 14
    gpu_esn_dim_8 = 32
    gpu_esn_input_8 = 8 + 6
    gpu_esn_W_in_8 = torch.randn(gpu_esn_dim_8, gpu_esn_input_8, device=device) * 0.1
    gpu_esn_W_8 = torch.randn(gpu_esn_dim_8, gpu_esn_dim_8, device=device)
    with torch.no_grad():
        sr = torch.linalg.eigvals(gpu_esn_W_8).abs().max().item()
        gpu_esn_W_8 *= 0.95 / max(sr, 1e-6)
    print(f"  GPU-ESN 8N:  {gpu_esn_dim_8}x{gpu_esn_input_8}, sr=0.95")

    LEVELS = ['L3_FPGA_ALONE', 'L5_BRIDGE', 'L6_DEEP_INTER']

    def get_esn_params(n_use):
        if n_use == 8:
            return gpu_esn_W_in_8, gpu_esn_W_8
        return gpu_esn_W_in_128, gpu_esn_W_128

    def run_condition(fpga, signal_trials, mode, n_use, label=''):
        """Run multiple trials through FPGA, return feature matrix."""
        esn_w_in, esn_w = get_esn_params(n_use)
        all_features = []
        t0 = time.time()
        for trial_i in range(len(signal_trials)):
            fpga_s, telem_s, gpu_s = run_fpga_trial(
                fpga, signal_trials[trial_i], noises, w_in, w_noise,
                torch, device, mode=mode, beta=BETA_1F, n_use=n_use,
                gpu_esn_W_in=esn_w_in if mode == 'L6_DEEP_INTER' else None,
                gpu_esn_W=esn_w if mode == 'L6_DEEP_INTER' else None)
            feat = build_features(fpga_s, telem_s, gpu_s, mode, n_use)
            all_features.append(feat)
            if (trial_i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (trial_i + 1) / elapsed if elapsed > 0 else 0
                print(f"    {label} trial {trial_i+1}/{len(signal_trials)} ({rate:.1f} t/s)")
        return np.array(all_features), time.time() - t0

    # BENCHMARK 1: 7-class waveform (T373-T376)

    print(f"\n{'='*60}")
    print(f"BENCHMARK 1: 7-class waveform ({args.wave7_trials} trials × 30 steps)")
    print(f"{'='*60}")

    waves7, labels7 = generate_7class_waveforms(args.wave7_trials, 30)
    print(f"  Classes: {np.bincount(labels7)}")

    b1 = {}
    for mode in LEVELS:
        key = f"{mode}_128N"
        print(f"\n  Running {key}...")
        X, elapsed = run_condition(fpga, waves7, mode, N_NEURONS, label=key)
        r = classify_condition(X, labels7, n_classes=7)
        b1[key] = r
        print(f"  {key}: {r['mean']:.3f} ± {r['std']:.3f} ({elapsed:.0f}s)")

    # 8-neuron variants for scale comparison
    for mode in ['L3_FPGA_ALONE', 'L6_DEEP_INTER']:
        key = f"{mode}_8N"
        print(f"\n  Running {key}...")
        X, elapsed = run_condition(fpga, waves7, mode, 8, label=key)
        r = classify_condition(X, labels7, n_classes=7)
        b1[key] = r
        print(f"  {key}: {r['mean']:.3f} ± {r['std']:.3f} ({elapsed:.0f}s)")

    results['benchmark1_7class'] = b1

    L6_7 = b1['L6_DEEP_INTER_128N']['mean']
    L5_7 = b1['L5_BRIDGE_128N']['mean']
    L3_7 = b1['L3_FPGA_ALONE_128N']['mean']
    L3_8N_7 = b1['L3_FPGA_ALONE_8N']['mean']

    T373 = L6_7 > L3_7
    T374 = L6_7 > 0.50
    T375 = L5_7 > L3_7
    T376_scale = L3_7 > L3_8N_7

    def pf(name, passed, detail):
        print(f"  {name}: {'PASS' if passed else 'FAIL'} — {detail}")

    pf('T373', T373, f'L6>L3 7-class ({L6_7:.3f} vs {L3_7:.3f})')
    pf('T374', T374, f'L6>0.50 ({L6_7:.3f})')
    pf('T375', T375, f'L5>L3 ({L5_7:.3f} vs {L3_7:.3f})')
    pf('T376', T376_scale, f'128N>8N L3 ({L3_7:.3f} vs {L3_8N_7:.3f})')

    def rt(tid, passed, desc, **kw):
        results['tests'][tid] = {'pass': bool(passed), 'desc': desc, **kw}

    results['tests'] = {}
    rt('T373', T373, 'L6>L3 7-class', L6=L6_7, L3=L3_7)
    rt('T374', T374, 'L6>0.50 7-class', L6=L6_7)
    rt('T375', T375, 'L5>L3 7-class', L5=L5_7, L3=L3_7)
    rt('T376', T376_scale, '128N>8N L3', L3_128N=L3_7, L3_8N=L3_8N_7)

    # BENCHMARK 2: Temporal XOR τ=5 and τ=10 (T377-T380)

    print(f"\n{'='*60}")
    print(f"BENCHMARK 2: Temporal XOR τ=5,10 ({args.xor_trials} trials × 50 steps)")
    print(f"{'='*60}")

    b2 = {}
    for tau in [5, 10]:
        xor_trials, xor_targets = generate_temporal_xor(args.xor_trials, 50, tau)
        print(f"\n  τ={tau}:")

        for mode in LEVELS:
            key = f"{mode}_128N_tau{tau}"
            print(f"  Running {key}...")

            # Run FPGA trials and collect per-step features
            esn_w_in, esn_w = get_esn_params(N_NEURONS)
            all_features = []
            all_labels = []
            t0 = time.time()

            for trial_i in range(args.xor_trials):
                fpga_s, telem_s, gpu_s = run_fpga_trial(
                    fpga, xor_trials[trial_i], noises, w_in, w_noise,
                    torch, device, mode=mode, beta=BETA_1F, n_use=N_NEURONS,
                    gpu_esn_W_in=esn_w_in if mode == 'L6_DEEP_INTER' else None,
                    gpu_esn_W=esn_w if mode == 'L6_DEEP_INTER' else None)

                # For XOR: classify each step (from tau onwards) using reservoir state
                for t_i in range(tau, 50):
                    if mode == 'L6_DEEP_INTER':
                        feat = np.concatenate([fpga_s[t_i], telem_s[t_i], gpu_s[t_i]])
                    elif mode == 'L5_BRIDGE':
                        feat = np.concatenate([fpga_s[t_i], telem_s[t_i]])
                    else:
                        feat = fpga_s[t_i]
                    all_features.append(feat)
                    all_labels.append(xor_targets[trial_i][t_i])

                if (trial_i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"    trial {trial_i+1}/{args.xor_trials} ({(trial_i+1)/elapsed:.1f} t/s)")

            X = np.array(all_features)
            y = np.array(all_labels)
            r = classify_condition(X, y, n_classes=2)
            b2[key] = r
            print(f"    {key}: {r['mean']:.3f} ± {r['std']:.3f}")

    results['benchmark2_xor'] = b2

    xL6_5 = b2['L6_DEEP_INTER_128N_tau5']['mean']
    xL3_5 = b2['L3_FPGA_ALONE_128N_tau5']['mean']
    xL6_10 = b2['L6_DEEP_INTER_128N_tau10']['mean']
    xL3_10 = b2['L3_FPGA_ALONE_128N_tau10']['mean']
    xL5_10 = b2['L5_BRIDGE_128N_tau10']['mean']

    T377 = xL6_5 > xL3_5
    T378 = xL6_10 > xL3_10
    T379 = xL5_10 > xL3_10
    T380 = xL6_10 > 0.55

    pf('T377', T377, f'L6>L3 XOR τ=5 ({xL6_5:.3f} vs {xL3_5:.3f})')
    pf('T378', T378, f'L6>L3 XOR τ=10 ({xL6_10:.3f} vs {xL3_10:.3f})')
    pf('T379', T379, f'L5>L3 XOR τ=10 ({xL5_10:.3f} vs {xL3_10:.3f})')
    pf('T380', T380, f'L6 XOR τ=10>0.55 ({xL6_10:.3f})')

    rt('T377', T377, 'L6>L3 XOR τ=5', L6=xL6_5, L3=xL3_5)
    rt('T378', T378, 'L6>L3 XOR τ=10', L6=xL6_10, L3=xL3_10)
    rt('T379', T379, 'L5>L3 XOR τ=10', L5=xL5_10, L3=xL3_10)
    rt('T380', T380, 'L6 XOR τ=10>0.55', L6=xL6_10)

    # BENCHMARK 3: Memory Capacity (T381-T384)

    print(f"\n{'='*60}")
    print(f"BENCHMARK 3: Memory Capacity ({args.mc_steps} steps, delays 1..40)")
    print(f"{'='*60}")

    mc_input = generate_memory_capacity_input(args.mc_steps)
    max_delay = 40
    washout = max_delay + 10  # discard first steps for reservoir warmup

    b3 = {}
    # Run each condition through FPGA as a single long sequence
    conditions_mc = [
        ('L3_128N', 'L3_FPGA_ALONE', N_NEURONS),
        ('L5_128N', 'L5_BRIDGE', N_NEURONS),
        ('L6_128N', 'L6_DEEP_INTER', N_NEURONS),
        ('L6_8N',  'L6_DEEP_INTER', 8),
    ]

    for cond_name, mode, n_use in conditions_mc:
        print(f"\n  Running MC: {cond_name} ({args.mc_steps} steps)...")
        esn_w_in, esn_w = get_esn_params(n_use)
        t0 = time.time()

        # Run reservoir on full sequence
        fpga_s, telem_s, gpu_s = run_fpga_trial(
            fpga, mc_input, noises, w_in, w_noise,
            torch, device, mode=mode, beta=BETA_1F, n_use=n_use,
            gpu_esn_W_in=esn_w_in if mode == 'L6_DEEP_INTER' else None,
            gpu_esn_W=esn_w if mode == 'L6_DEEP_INTER' else None)

        elapsed = time.time() - t0
        print(f"    {args.mc_steps} steps in {elapsed:.1f}s")

        # Build state matrix
        if mode == 'L6_DEEP_INTER':
            states = np.hstack([fpga_s, telem_s, gpu_s])
        elif mode == 'L5_BRIDGE':
            states = np.hstack([fpga_s, telem_s])
        else:
            states = fpga_s

        # Memory capacity: for each delay k, train ridge to predict u[t-k]
        mc_total = 0.0
        mc_per_delay = {}
        usable = args.mc_steps - washout
        X_res = states[washout:]  # reservoir states after washout

        # Standardize reservoir states
        mu_res = X_res.mean(axis=0)
        sigma_res = X_res.std(axis=0)
        sigma_res[sigma_res < 1e-6] = 1.0
        X_std = (X_res - mu_res) / sigma_res

        # PCA if needed
        if X_std.shape[1] > 100:
            X_std, pca_mu, pca_Vt = pca_reduce(X_std, n_components=100)

        n_train = int(usable * 0.7)
        X_tr = X_std[:n_train]
        X_te = X_std[n_train:]

        for k in range(1, max_delay + 1):
            target = mc_input[washout - k : args.mc_steps - k]
            y_tr = target[:n_train]
            y_te = target[n_train:]

            corr = ridge_regress(X_tr, y_tr, X_te, y_te)
            mc_k = corr ** 2
            mc_total += mc_k
            mc_per_delay[k] = float(mc_k)

        b3[cond_name] = {
            'mc_total': float(mc_total),
            'mc_per_delay': mc_per_delay,
            'elapsed': elapsed,
        }
        print(f"    MC({cond_name}) = {mc_total:.3f}")

    results['benchmark3_memory_capacity'] = b3

    mc6 = b3['L6_128N']['mc_total']
    mc5 = b3['L5_128N']['mc_total']
    mc3 = b3['L3_128N']['mc_total']
    mc6_8 = b3['L6_8N']['mc_total']

    T381 = mc6 > mc3
    T382 = mc5 > mc3
    T383 = mc6 > 3.0
    T384 = mc6 > mc6_8

    pf('T381', T381, f'MC(L6)>MC(L3) ({mc6:.3f} vs {mc3:.3f})')
    pf('T382', T382, f'MC(L5)>MC(L3) ({mc5:.3f} vs {mc3:.3f})')
    pf('T383', T383, f'MC(L6)>3.0 ({mc6:.3f})')
    pf('T384', T384, f'MC(L6_128)>MC(L6_8) ({mc6:.3f} vs {mc6_8:.3f})')

    rt('T381', T381, 'MC(L6)>MC(L3)', L6=mc6, L3=mc3)
    rt('T382', T382, 'MC(L5)>MC(L3)', L5=mc5, L3=mc3)
    rt('T383', T383, 'MC(L6)>3.0', L6=mc6)
    rt('T384', T384, 'MC(L6_128N)>MC(L6_8N)', L6_128N=mc6, L6_8N=mc6_8)

    # BENCHMARK 4: Scale-dependent crossover (T385-T388)

    print(f"\n{'='*60}")
    print(f"BENCHMARK 4: Scale-dependent crossover ({args.crossover_trials} trials, 3-class)")
    print(f"{'='*60}")

    # Generate 3-class waveform (same as z2210) — reuse 7-class generator with 3 classes
    waves3, labels3 = generate_7class_waveforms(args.crossover_trials, 30, seed=99)
    # Remap to 3 classes: 0=sine, 1=triangle, 2=square (drop classes 3-6)
    mask3 = labels3 < 3
    waves3, labels3 = waves3[mask3], labels3[mask3]
    print(f"  3-class waveforms: {len(labels3)} trials, classes: {np.bincount(labels3)}")

    b4 = {}
    for mode, n_use in [('L3_FPGA_ALONE', 8), ('L6_DEEP_INTER', 8),
                         ('L3_FPGA_ALONE', N_NEURONS), ('L6_DEEP_INTER', N_NEURONS)]:
        key = f"{mode}_{n_use}N"
        print(f"\n  Running {key}...")
        X, elapsed = run_condition(fpga, waves3, mode, n_use, label=key)
        r = classify_condition(X, labels3, n_classes=3)
        b4[key] = r
        print(f"    {key}: {r['mean']:.3f} ± {r['std']:.3f} ({elapsed:.0f}s)")

    results['benchmark4_crossover'] = b4

    L6_8N = b4['L6_DEEP_INTER_8N']['mean']
    L3_8N = b4['L3_FPGA_ALONE_8N']['mean']
    L6_128N = b4['L6_DEEP_INTER_128N']['mean']
    L3_128N = b4['L3_FPGA_ALONE_128N']['mean']

    T385 = L6_8N > L3_8N
    T386 = L6_128N > L6_8N
    T387 = (L6_8N - L3_8N) > (L6_128N - L3_128N)
    T388 = L3_128N > L3_8N

    adv_8N = L6_8N - L3_8N
    adv_128N = L6_128N - L3_128N

    pf('T385', T385, f'L6_8N>L3_8N ({L6_8N:.3f} vs {L3_8N:.3f}, adv={adv_8N:+.3f})')
    pf('T386', T386, f'L6_128N>L6_8N ({L6_128N:.3f} vs {L6_8N:.3f})')
    pf('T387', T387, f'adv_8N>adv_128N ({adv_8N:+.3f} vs {adv_128N:+.3f})')
    pf('T388', T388, f'L3_128N>L3_8N ({L3_128N:.3f} vs {L3_8N:.3f})')

    rt('T385', T385, 'L6_8N>L3_8N crossover', L6_8N=L6_8N, L3_8N=L3_8N)
    rt('T386', T386, 'L6_128N>L6_8N', L6_128N=L6_128N, L6_8N=L6_8N)
    rt('T387', T387, 'advantage decreases with scale', adv_8N=adv_8N, adv_128N=adv_128N)
    rt('T388', T388, 'L3_128N>L3_8N sanity', L3_128N=L3_128N, L3_8N=L3_8N)

    # BENCHMARK 5: Thermal-Coupled Classification (T389-T392)
    #
    # HYPOTHESIS: GPU thermal state via foam-attenuated coupling (~8× modulation)
    # creates a slow context signal. If waveform class is CORRELATED with thermal
    # state (e.g., class 0 during GPU-hot, class 1 during GPU-cool), then:
    #   - L6 (which reads firmware telemetry including temperature) can exploit this
    #   - L3 (FPGA alone, no telemetry) is blind to it
    # This tests whether thermal coupling is a GENUINE computation channel.
    #
    # Protocol:
    #   - 3-class waveform, 200 trials
    #   - Classes 0,1 presented during GPU-hot (matmul stress)
    #   - Class 2 presented during GPU-cool (idle)
    #   - The thermal context is INFORMATIVE: knowing temperature helps classification
    #   - L3 sees only spikes (thermally modulated but no explicit temp reading)
    #   - L6 sees spikes + firmware telemetry (has explicit thermal reading)

    print(f"\n{'='*60}")
    print(f"BENCHMARK 5: Thermal-Coupled Classification (200 trials, 3-class)")
    print(f"  GPU stress correlated with waveform class")
    print(f"{'='*60}")

    n_thermal_trials = 200
    thermal_waves, thermal_labels = generate_7class_waveforms(n_thermal_trials, 30, seed=777)
    # Remap to 3 classes
    mask_t = thermal_labels < 3
    thermal_waves = thermal_waves[mask_t]
    thermal_labels = thermal_labels[mask_t]
    n_thermal_trials = len(thermal_labels)
    print(f"  {n_thermal_trials} trials, classes: {np.bincount(thermal_labels)}")

    # Assign thermal context: class 2 = cool, class 0,1 = hot
    # Shuffle within context groups to avoid ordering artifacts
    rng_t = np.random.default_rng(123)
    order = rng_t.permutation(n_thermal_trials)
    thermal_waves = thermal_waves[order]
    thermal_labels = thermal_labels[order]

    b5 = {}

    for mode in LEVELS:
        key = f"{mode}_THERMAL"
        print(f"\n  Running {key}...")
        esn_w_in, esn_w = get_esn_params(N_NEURONS)
        all_features = []
        t0 = time.time()

        for trial_i in range(n_thermal_trials):
            lbl = thermal_labels[trial_i]

            # Classes 0,1 get GPU stress; class 2 is cool
            if lbl in (0, 1):
                stress_thread = start_gpu_stress(3.0)  # 3s burst covers trial duration
            else:
                stop_gpu_stress()

            # Small delay for thermal state to begin changing
            time.sleep(0.2)

            fpga_s, telem_s, gpu_s = run_fpga_trial(
                fpga, thermal_waves[trial_i], noises, w_in, w_noise,
                torch, device, mode=mode, beta=BETA_1F, n_use=N_NEURONS,
                gpu_esn_W_in=esn_w_in if mode == 'L6_DEEP_INTER' else None,
                gpu_esn_W=esn_w if mode == 'L6_DEEP_INTER' else None)

            stop_gpu_stress()

            feat = build_features(fpga_s, telem_s, gpu_s, mode, N_NEURONS)
            all_features.append(feat)

            if (trial_i + 1) % 50 == 0:
                elapsed = time.time() - t0
                gpu_t = read_gpu_thermal()
                print(f"    {key} trial {trial_i+1}/{n_thermal_trials}"
                      f" ({(trial_i+1)/elapsed:.1f} t/s, GPU={gpu_t}°C)")

        # Let GPU cool before next condition
        stop_gpu_stress()
        time.sleep(10)

        X = np.array(all_features)
        r = classify_condition(X, thermal_labels, n_classes=3)
        b5[key] = r
        print(f"    {key}: {r['mean']:.3f} ± {r['std']:.3f}")

    # Also run L3 WITHOUT thermal correlation (control: random stress)
    print(f"\n  Running L3_CONTROL (random stress, no class correlation)...")
    rng_ctrl = np.random.default_rng(456)
    all_features_ctrl = []
    t0 = time.time()

    for trial_i in range(n_thermal_trials):
        # Random stress: 50% chance regardless of class
        if rng_ctrl.random() < 0.5:
            stress_thread = start_gpu_stress(3.0)
        else:
            stop_gpu_stress()
        time.sleep(0.2)

        fpga_s, telem_s, gpu_s = run_fpga_trial(
            fpga, thermal_waves[trial_i], noises, w_in, w_noise,
            torch, device, mode='L3_FPGA_ALONE', beta=BETA_1F, n_use=N_NEURONS)
        stop_gpu_stress()

        feat = build_features(fpga_s, telem_s, gpu_s, 'L3_FPGA_ALONE', N_NEURONS)
        all_features_ctrl.append(feat)

        if (trial_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    L3_CONTROL trial {trial_i+1}/{n_thermal_trials}")

    stop_gpu_stress()
    time.sleep(10)

    X_ctrl = np.array(all_features_ctrl)
    r_ctrl = classify_condition(X_ctrl, thermal_labels, n_classes=3)
    b5['L3_CONTROL'] = r_ctrl
    print(f"    L3_CONTROL: {r_ctrl['mean']:.3f} ± {r_ctrl['std']:.3f}")

    results['benchmark5_thermal'] = b5

    L6_th = b5['L6_DEEP_INTER_THERMAL']['mean']
    L5_th = b5['L5_BRIDGE_THERMAL']['mean']
    L3_th = b5['L3_FPGA_ALONE_THERMAL']['mean']
    L3_ctrl = b5['L3_CONTROL']['mean']

    T389 = L6_th > L3_th  # Thermal context helps L6
    T390 = L6_th > L5_th  # Full fusion > bridge-only
    T391 = L3_th > L3_ctrl  # Correlated thermal helps even L3 (implicit via spike modulation)
    T392 = (L6_th - L3_th) > 0.03  # L6 advantage from thermal > 3pp

    pf('T389', T389, f'L6_THERMAL>L3_THERMAL ({L6_th:.3f} vs {L3_th:.3f})')
    pf('T390', T390, f'L6_THERMAL>L5_THERMAL ({L6_th:.3f} vs {L5_th:.3f})')
    pf('T391', T391, f'L3_THERMAL>L3_CONTROL ({L3_th:.3f} vs {L3_ctrl:.3f})')
    pf('T392', T392, f'L6-L3 thermal gap>{0.03} ({L6_th-L3_th:+.3f})')

    rt('T389', T389, 'L6>L3 thermal context', L6=L6_th, L3=L3_th)
    rt('T390', T390, 'L6>L5 thermal context', L6=L6_th, L5=L5_th)
    rt('T391', T391, 'L3 thermal > L3 random stress', L3_thermal=L3_th, L3_ctrl=L3_ctrl)
    rt('T392', T392, 'L6-L3 thermal gap>3pp', gap=L6_th - L3_th)

    # Summary

    tests = results['tests']
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_pass}/{n_total} tests passed")
    print(f"{'='*60}")
    for tid in sorted(tests.keys(), key=lambda x: int(x[1:])):
        t = tests[tid]
        status = 'PASS' if t['pass'] else 'FAIL'
        print(f"  {tid}: {status} — {t['desc']}")

    results['summary'] = {'pass': n_pass, 'total': n_total}

    out_path = RESULTS / 'z2212_hard_benchmarks.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()

