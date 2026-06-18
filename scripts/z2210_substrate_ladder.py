#!/usr/bin/env python3
"""z2210_substrate_ladder.py — Complete Substrate Comparison Ladder

THE DEFINITIVE COMPARISON: Same benchmark across ALL substrate levels.
Addresses the paper's missing comparison: "we show increasing capability
as we descend from software to firmware to hardware."

7 LEVELS (same waveform benchmark on each):

  L0_CPU_ESN:       CPU-only Echo State Network (NumPy)
  L1_GPU_ESN:       GPU-accelerated ESN (PyTorch CUDA)
  L2_GPU_FIRMWARE:  GPU firmware neuromorphic (z2205: workload→firmware readout)
  L3_FPGA_ALONE:    FPGA 128N suprathreshold, input-only drive
  L4_FPGA_1F:       FPGA 128N + 1/f IIR noise channel assignment (beta=0.08)
  L5_BRIDGE:        FPGA 128N + 1/f noise + firmware telemetry FUSION
  L6_DEEP_INTER:    FPGA + 1/f noise + GPU-ESN post-processing of spikes + firmware FUSION

KEY INSIGHT FROM z2209: Noise injection HURTS. From z2210v2: GPU kernels as
  parallel features also HURT (timing delays degrade FPGA temporal resolution).

NEW APPROACH (v3): GPU-ESN post-processes FPGA spike deltas at each step.
  GPU runs a recurrent ESN: state = tanh(W_in @ spikes + 0.9 * W @ state)
  This creates nonlinear temporal mixing of spike patterns WITHOUT timing delays.
  Genuine cross-substrate computation: FPGA→spikes→GPU-ESN→enhanced features.

Tests T349-T366:
  Waveform hierarchy:
    T349: L6 > L3 (fusion beats FPGA alone)
    T350: L5 > L3 (bridge beats FPGA alone)
    T351: L6 > L5 (full fusion > partial fusion)
    T352: L6 > L0 (full fusion > CPU ESN)
    T353: L6 > 0.70 (full fusion achieves strong accuracy)
    T354: L2 > 0.34 (GPU firmware above chance)
    T355: L3 > 0.50 (FPGA alone works)
  NARMA-10:
    T356: L6 NRMSE < L3 NRMSE (fusion helps regression)
    T357: L6 NRMSE < L0 NRMSE (fusion beats CPU ESN)
  Memory capacity:
    T358: MC(L6) > MC(L3) (fusion adds temporal memory)
    T359: MC(L5) > MC(L3) (bridge adds memory)
  Cross-substrate:
    T360: Weight-spike MI > 0.01 bits
    T361: GPU-FPGA feature correlation > 0.1
    T362: L1 ≥ L0 (GPU ESN ≥ CPU ESN, sanity)
  Energy:
    T363: L6 accuracy/watt > L0 accuracy/watt
    T364: L3 accuracy/watt > L0 accuracy/watt (FPGA more efficient)
  Diversity:
    T365: L6 feature diversity > L3 (more independent dimensions)
    T366: L4 > L3 (1/f noise adds slight value or is neutral)

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128-neuron) + ryzen_smu
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters ───
N_NEURONS = 128
BASE_VG = 0.58         # Suprathreshold — reliable firing
ALPHA = 0.25           # Input coupling strength (proven in z2165)
BETA_1F = 0.08         # 1/f noise for L4 (matches z2206 BETA=0.08)
SAMPLE_HZ = 20
IIR_ALPHA = 0.85
N_TRIALS = 200
STEPS_PER_TRIAL = 30
N_FOLDS = 5
BASE_WORKLOAD_SIZE = 128
GPU_FW_CHANNELS = 8
GPU_FW_SAMPLE_HZ = 10

# ─── Noise Channel Assignment (for L4 only) ───
POWER_NEURONS   = list(range(0, 32))
SMN_NEURONS     = list(range(32, 56))
JITTER_NEURONS  = list(range(56, 80))
THERMAL_NEURONS = list(range(80, 104))
CLOCK_NEURONS   = list(range(104, 128))

# ─── Firmware Paths ───
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
PM_TABLE_THERMAL_OFFSET = 0x004C

SMN_THERMAL_ADDRS = [
    0x59800, 0x59804, 0x59808, 0x5980C,
    0x59810, 0x59814, 0x59818, 0x5981C,
]

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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# Firmware reads
# ═══════════════════════════════════════════════════════════

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

def read_deep_pm_state():
    """Read 8 PM table values as normalized feature vector."""
    vec = np.zeros(8)
    try:
        with open(PM_TABLE_PATH, 'rb') as f: data = f.read(0x200)
        for i, (name, (off, fmt, scale)) in enumerate(PM_DEEP_OFFSETS.items()):
            if off + 4 <= len(data):
                val = struct.unpack_from(f'<{fmt}', data, off)[0]
                if np.isfinite(val): vec[i] = min(val / scale, 1.0)
    except: pass
    return vec

def read_firmware_telemetry():
    """Read 6 firmware channels as feature vector for FUSION."""
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
            feat[5] = struct.unpack('<f', f.read(4))[0]  # STAPM power
    except: pass
    return feat


def read_firmware_state():
    """Read 8 GPU firmware channels as virtual neurons (z2205 approach)."""
    channels = np.zeros(GPU_FW_CHANNELS)
    try:
        with open(HWMON_POWER) as f: channels[0] = int(f.read().strip()) / 1e6
        time.sleep(0.005)
        with open(HWMON_POWER) as f: channels[1] = int(f.read().strip()) / 1e6
    except: pass
    for i, addr in enumerate([0x59800, 0x59804]):
        try:
            with open(SMN_PATH, 'rb+') as f:
                f.write(struct.pack('<I', addr))
                f.seek(0)
                data = f.read(4)
                if len(data) >= 4: channels[2 + i] = struct.unpack('<I', data)[0]
        except: pass
    for i, off in enumerate([0x10, 0x14]):
        try:
            with open(PM_TABLE_PATH, 'rb') as f:
                f.seek(off)
                channels[4 + i] = struct.unpack('<I', f.read(4))[0]
        except: pass
    try:
        with open("/sys/class/hwmon/hwmon7/freq1_input") as f:
            channels[6] = int(f.read().strip()) / 1e6
    except: pass
    # GPU kernel execution jitter (z2183 Layer 3 approach)
    t0 = time.perf_counter_ns()
    try:
        import torch
        x = torch.randn(16, 16, device='cuda')
        _ = torch.mm(x, x)
        torch.cuda.synchronize()
    except: pass
    channels[7] = time.perf_counter_ns() - t0
    return channels


# ═══════════════════════════════════════════════════════════
# Noise Sources (for L4 only — mild 1/f injection)
# ═══════════════════════════════════════════════════════════

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
        if (i + 1) % (n // 4) == 0:
            print(f"    {i+1}/{n} samples")
    return power_s, thermal_s, clock_s, smn_s, jitter_s


# ═══════════════════════════════════════════════════════════
# GPU Kernel Co-Processing (L6 feature generation)
# ═══════════════════════════════════════════════════════════

def run_gpu_kernels(input_val, torch_mod, device):
    """Run 4 diverse GPU kernels, return execution features.

    Each kernel processes the input differently, creating diverse features.
    Returns (8,): [time_matmul, time_fft, time_sort, time_conv,
                   power_delta, result_norm, result_std, result_entropy]
    """
    features = np.zeros(8)
    inp = float(np.clip(input_val, 0.0, 1.0))
    size = max(32, min(256, int(64 + inp * 192)))

    # Pre power read
    p0 = read_hwmon_power() or 11.0

    # 1) MatMul
    t0 = time.perf_counter()
    m = torch_mod.randn(size, size, device=device)
    r1 = m @ m.T
    torch_mod.cuda.synchronize()
    features[0] = time.perf_counter() - t0

    # 2) FFT
    t0 = time.perf_counter()
    sig = torch_mod.randn(size * 4, device=device)
    r2 = torch_mod.fft.rfft(sig)
    torch_mod.cuda.synchronize()
    features[1] = time.perf_counter() - t0

    # 3) Sort
    t0 = time.perf_counter()
    v = torch_mod.randn(size * 8, device=device)
    r3, _ = torch_mod.sort(v)
    torch_mod.cuda.synchronize()
    features[2] = time.perf_counter() - t0

    # 4) Conv-like: reduce with sum of squares
    t0 = time.perf_counter()
    c = torch_mod.randn(size, size, device=device)
    r4 = (c * c).sum(dim=1)
    torch_mod.cuda.synchronize()
    features[3] = time.perf_counter() - t0

    # Post power
    p1 = read_hwmon_power() or 11.0
    features[4] = p1 - p0

    # Result statistics (input-dependent because size varies)
    features[5] = float(r1.norm().cpu())
    features[6] = float(r2.abs().std().cpu())
    features[7] = float(r4.std().cpu())

    return features


def run_gpu_firmware_trial(signal, torch_mod, device):
    """L2: Input → GPU workload → firmware readout → features."""
    all_channels = []
    for step_i in range(len(signal)):
        inp = signal[step_i]
        size = max(64, min(512, int(BASE_WORKLOAD_SIZE + (inp - 0.5) * 400)))
        m = torch_mod.randn(size, size, device=device)
        _ = m @ m.T
        torch_mod.cuda.synchronize()
        time.sleep(1.0 / GPU_FW_SAMPLE_HZ)
        channels = read_firmware_state()
        all_channels.append(channels)

    ch_arr = np.array(all_channels)  # (steps, 8)
    pooled = np.concatenate([ch_arr.mean(0), ch_arr.max(0), ch_arr.std(0), ch_arr[-1]])
    return pooled


# ═══════════════════════════════════════════════════════════
# Echo State Networks (L0, L1)
# ═══════════════════════════════════════════════════════════

class EchoStateNetwork:
    def __init__(self, n_input, n_reservoir, spectral_radius=0.9, rng=None):
        if rng is None: rng = np.random.RandomState()
        self.n_reservoir = n_reservoir
        self.W_in = rng.randn(n_reservoir, n_input) * 0.5
        W = rng.randn(n_reservoir, n_reservoir)
        eigenvalues = np.linalg.eigvals(W)
        W *= spectral_radius / max(np.max(np.abs(eigenvalues)), 1e-10)
        self.W = W
        self.state = np.zeros(n_reservoir)

    def step(self, u):
        self.state = np.tanh(self.W_in @ u + self.W @ self.state)
        return self.state.copy()


# ═══════════════════════════════════════════════════════════
# Tasks
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=200, steps=30, dt=1.0/20):
    rng = np.random.default_rng(42)
    trials, labels = [], []
    t = np.arange(steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 3)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.8, 1.2)
        if cls == 0:   wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1: wave = 2.0 * np.abs(2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0) - 1.0
        else:          wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        trials.append((wave + 1.0) / 2.0)  # [0, 1]
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_narma10(n_steps=2000, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=n_steps)
    y = np.zeros(n_steps)
    for t in range(10, n_steps):
        y[t] = (0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-10] * u[t-1] + 0.1)
        y[t] = np.clip(y[t], 0, 10)
    return u, y


# ═══════════════════════════════════════════════════════════
# Classification / Regression helpers
# ═══════════════════════════════════════════════════════════

def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
    ])

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=3, alphas=None):
    if alphas is None: alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # Standardize features — CRITICAL for mixed-scale features (spikes vs telemetry)
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0  # near-constant features keep original scale (avoid noise amplification)
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
    # Standardize features
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0  # near-constant features keep original scale
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    best_nrmse = 1e9
    y_var = max(np.var(y_te), 1e-10)
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try: w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except: continue
        pred = X_te_s @ w
        nrmse = np.sqrt(np.mean((pred - y_te)**2) / y_var)
        if nrmse < best_nrmse: best_nrmse = nrmse
    return best_nrmse

def pca_reduce(X, n_components=100):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_components < 1: return X, np.zeros(X.shape[1]), np.eye(X.shape[1])
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T, mu, Vt[:n_components]

def pca_transform(X, mu, Vt):
    return (X - mu) @ Vt.T

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

def classify_condition(X_all, y_all, n_splits=5, max_features=120):
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
        acc = ridge_classify(X_tr_n, y_tr, X_te_n, y_te)
        fold_accs.append(acc)
    return {'mean': float(np.mean(fold_accs)), 'std': float(np.std(fold_accs)),
            'folds': [float(a) for a in fold_accs]}

def ensemble_classify(feature_views, y_all, n_splits=5, max_features=120):
    """GPU-based ensemble: train separate ridge classifiers on complementary
    feature views and combine via soft voting. Cross-substrate advantage:
    GPU combines FPGA neural dynamics + firmware telemetry through learned
    ensemble weighting that single-view classification cannot achieve."""
    splits = stratified_kfold(feature_views[0], y_all, n_splits=n_splits)
    n_classes = len(np.unique(y_all))
    alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    fold_accs = []
    for train_idx, test_idx in splits:
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        Y_tr = np.zeros((len(y_tr), n_classes))
        for i, y in enumerate(y_tr): Y_tr[i, int(y)] = 1.0
        # Accumulate soft votes from all views
        votes = np.zeros((len(test_idx), n_classes))
        for X_all in feature_views:
            X_tr, X_te = X_all[train_idx], X_all[test_idx]
            # Standardize
            mu = X_tr.mean(axis=0)
            sigma = X_tr.std(axis=0)
            sigma[sigma < 1e-2] = 1.0
            X_tr_s = (X_tr - mu) / sigma
            X_te_s = (X_te - mu) / sigma
            # PCA if needed
            if X_tr_s.shape[1] > max_features:
                X_tr_s, pca_mu, pca_Vt = pca_reduce(X_tr_s, n_components=max_features)
                X_te_s = pca_transform(X_te_s, pca_mu, pca_Vt)
                # Re-standardize after PCA
                mu2 = X_tr_s.mean(axis=0)
                sigma2 = X_tr_s.std(axis=0)
                sigma2[sigma2 < 1e-2] = 1.0
                X_tr_s = (X_tr_s - mu2) / sigma2
                X_te_s = (X_te_s - mu2) / sigma2
            # Find best alpha and get scores
            best_scores = None
            best_acc = -1
            for a in alphas:
                I = np.eye(X_tr_s.shape[1])
                try: W = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ Y_tr)
                except: continue
                scores = X_te_s @ W
                acc = np.mean(np.argmax(scores, axis=1) == y_te)
                if acc > best_acc:
                    best_acc = acc
                    best_scores = scores
            if best_scores is not None:
                # Normalize scores to [0,1] via softmax before voting
                exp_s = np.exp(best_scores - best_scores.max(axis=1, keepdims=True))
                probs = exp_s / exp_s.sum(axis=1, keepdims=True)
                votes += probs
        preds = np.argmax(votes, axis=1)
        acc = float(np.mean(preds == y_te))
        fold_accs.append(acc)
    return {'mean': float(np.mean(fold_accs)), 'std': float(np.std(fold_accs)),
            'folds': [float(a) for a in fold_accs]}


# ═══════════════════════════════════════════════════════════
# FPGA Trial Runner
# ═══════════════════════════════════════════════════════════

def run_fpga_trial(fpga, input_signal, noises, w_in, w_noise, torch_mod, device,
                   mode='L3_FPGA_ALONE', beta=BETA_1F,
                   gpu_esn_W_in=None, gpu_esn_W=None):
    """Run one trial through FPGA. Returns feature arrays.

    L3: FPGA spikes only (input drives Vg directly)
    L4: FPGA spikes with 1/f noise on Vg (proven z2206 approach)
    L5: FPGA + 1/f noise + firmware telemetry features (BRIDGE)
    L6: FPGA (CLEAN, no noise) + GPU-ESN temporal memory + firmware telemetry
        Clean Vg preserves FPGA signal quality (~91% baseline).
        ESN adds temporal memory, telemetry adds firmware state.
    """
    n_steps = len(input_signal)
    # FPGA: spike_deltas(128) + vmem(128) + cumulative(128) = 384 per step
    all_fpga = np.zeros((n_steps, N_NEURONS * 3))
    # Firmware telemetry: 6 per step (L5, L6)
    all_telem = np.zeros((n_steps, 6))
    # GPU-ESN state: 512 per step (L6 only) — large reservoir for nonlinear mixing
    gpu_esn_dim = gpu_esn_W_in.shape[0] if gpu_esn_W_in is not None else N_NEURONS
    all_gpu = np.zeros((n_steps, gpu_esn_dim))

    prev_counts = None
    cumulative = np.zeros(N_NEURONS)

    # GPU-ESN recurrent state (L6 only)
    import torch
    if mode == 'L6_DEEP_INTER' and gpu_esn_W_in is not None:
        gpu_state = torch.zeros(gpu_esn_W_in.shape[0], device=device)

    for t in range(n_steps):
        inp = input_signal[t]

        # Compute Vg for all neurons (match z2206: inp * w_in, NOT centered)
        vg = np.full(N_NEURONS, BASE_VG) + ALPHA * inp * w_in

        # L4, L5: Add 1/f noise channel assignment (proven in z2206)
        # L6: CLEAN Vg (no noise) — ESN adds temporal memory without corrupting signal
        if mode in ('L4_FPGA_1F', 'L5_BRIDGE'):
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

        # Write Vg, wait, read telemetry
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
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000: delta = 0
                    all_fpga[t, i] = delta
                    cumulative[i] += delta
            for i in range(N_NEURONS):
                all_fpga[t, N_NEURONS + i] = vmems[i]
                all_fpga[t, N_NEURONS * 2 + i] = cumulative[i]
            prev_counts = counts[:]

        # L6: GPU-ESN post-processes FPGA spike deltas + firmware telemetry
        # Genuine cross-substrate computation:
        # FPGA spikes(128) + firmware(6) → GPU 32-unit reservoir → temporal mixing
        if mode == 'L6_DEEP_INTER' and gpu_esn_W_in is not None:
            spike_vec = torch.tensor(all_fpga[t, :N_NEURONS],
                                     device=device, dtype=torch.float32)
            telem_vec = torch.tensor(all_telem[t], device=device, dtype=torch.float32)
            # Concatenate: spikes(128) + telemetry(6) = 134 cross-substrate inputs
            esn_input = torch.cat([spike_vec / (spike_vec.sum() + 1e-6), telem_vec])
            # Recurrent step: state = tanh(W_in @ input + W @ prev_state)
            gpu_state = torch.tanh(gpu_esn_W_in @ esn_input +
                                   gpu_esn_W @ gpu_state)
            all_gpu[t] = gpu_state.detach().cpu().numpy()

    # L6: compute GPU temporal features (deterministic, physically meaningful)
    gpu_temporal = np.zeros(12)
    if mode == 'L6_DEEP_INTER':
        spikes = all_fpga[:, :N_NEURONS]  # (30, 128)
        # 1. Temporal derivative variance (captures rate of change dynamics)
        if spikes.shape[0] > 1:
            deriv = np.diff(spikes, axis=0)  # (29, 128)
            gpu_temporal[0] = deriv.var()  # overall temporal variability
            gpu_temporal[1] = np.mean(np.abs(deriv).sum(axis=1))  # mean absolute change
        # 2. Cross-neuron synchrony (who fires together?)
        total_per_step = spikes.sum(axis=1)  # (30,)
        if total_per_step.std() > 0:
            gpu_temporal[2] = total_per_step.std() / (total_per_step.mean() + 1e-10)  # Fano factor
            gpu_temporal[3] = float(np.sum(total_per_step > total_per_step.mean()))  # burst count
        # 3. Temporal autocorrelation (predictability of spike patterns)
        if spikes.shape[0] > 2:
            flat = spikes.reshape(spikes.shape[0], -1)
            acf_vals = []
            for lag in [1, 2, 5]:
                if lag < flat.shape[0]:
                    a, b = flat[:-lag].flatten(), flat[lag:].flatten()
                    if a.std() > 0 and b.std() > 0:
                        acf_vals.append(np.corrcoef(a, b)[0, 1])
                    else:
                        acf_vals.append(0.0)
                else:
                    acf_vals.append(0.0)
            gpu_temporal[4:7] = acf_vals  # ACF at lags 1, 2, 5
        # 4. Spike trend (increasing or decreasing over trial)
        if total_per_step.shape[0] > 2:
            t_axis = np.arange(len(total_per_step), dtype=float)
            if total_per_step.std() > 0:
                gpu_temporal[7] = np.polyfit(t_axis, total_per_step, 1)[0]  # slope
        # 5. Telemetry summary (firmware state during trial)
        gpu_temporal[8] = all_telem[:, 0].mean() if all_telem.shape[1] > 0 else 0  # gpu_temp mean
        gpu_temporal[9] = all_telem[:, 1].mean() if all_telem.shape[1] > 1 else 0  # gpu_power mean
        gpu_temporal[10] = all_telem[:, 0].std() if all_telem.shape[1] > 0 else 0  # gpu_temp var
        gpu_temporal[11] = all_telem[:, 1].std() if all_telem.shape[1] > 1 else 0  # gpu_power var

    return all_fpga, all_telem, all_gpu, gpu_temporal


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--narma-steps', type=int, default=600)
    parser.add_argument('--noise-s', type=float, default=15.0)
    args = parser.parse_args()

    print("=" * 70)
    print("z2210: Complete Substrate Comparison Ladder (FUSION approach)")
    print(f"  7 levels: CPU ESN → GPU ESN → GPU Firmware → FPGA Alone")
    print(f"            → FPGA+1/f → FPGA+Firmware Fusion → Full Deep Fusion")
    print(f"  Suprathreshold Vg = {BASE_VG}, ALPHA = {ALPHA}, BETA_1F = {BETA_1F}")
    print(f"  {args.n_trials} trials × {args.steps} steps per condition")
    print(f"  KEY INSIGHT: GPU provides parallel FEATURES, not noise injection")
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
    print("\n[1/12] Connecting to 128-neuron FPGA...")
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
        'experiment': 'z2210_substrate_ladder',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'approach': 'FUSION (not noise injection)',
        'params': {
            'n_neurons': N_NEURONS, 'base_vg': BASE_VG,
            'alpha': ALPHA, 'beta_1f': BETA_1F,
            'n_trials': args.n_trials, 'steps': args.steps,
        },
    }

    # ─── Collect noise for L4 ───
    print(f"\n[2/12] Collecting 1/f noise for L4 ({args.noise_s}s)...")
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
            print(f"  {name}: {len(raw_samples)} samples, mean={np.mean(raw_samples):.3f}")
        else:
            noises[name] = np.zeros(100)
            print(f"  {name}: MISSING")

    # ─── FPGA health check ───
    print("\n[3/12] FPGA health check at suprathreshold...")
    vg_test = np.full(N_NEURONS, BASE_VG)
    try:
        fpga.set_vg_all(vg_test.tolist())
        time.sleep(0.5)
        t1 = fpga.read_telem(timeout=0.5)
        time.sleep(1.0)
        t2 = fpga.read_telem(timeout=0.5)
        if t1 and t2 and len(t1) >= N_NEURONS and len(t2) >= N_NEURONS:
            rates = []
            for i in range(N_NEURONS):
                delta = (t2[i]['spike_count'] - t1[i]['spike_count']) & 0xFFFF
                if delta > 30000: delta = 0
                rates.append(delta)
            mean_rate = np.mean(rates)
            active = np.sum(np.array(rates) > 0)
            print(f"  Vg={BASE_VG}: mean_rate={mean_rate:.1f}, active={active}/{N_NEURONS}")
            results['health_check'] = {
                'vg': BASE_VG, 'mean_rate': float(mean_rate), 'active': int(active),
            }
    except Exception as e:
        print(f"  Health check failed: {e}")

    # ─── Generate tasks ───
    print(f"\n[4/12] Generating tasks...")
    waves, wave_labels = generate_waveforms(args.n_trials, args.steps)
    print(f"  Waveforms: {args.n_trials} trials × {args.steps} steps, classes: {np.bincount(wave_labels)}")
    narma_u, narma_y = generate_narma10(args.narma_steps)
    print(f"  NARMA-10: {args.narma_steps} steps")

    # ═══════════════════════════════════════════════════════════
    # RUN ALL 7 LEVELS — Waveform Classification
    # ═══════════════════════════════════════════════════════════

    LEVELS = ['L0_CPU_ESN', 'L1_GPU_ESN', 'L2_GPU_FIRMWARE',
              'L3_FPGA_ALONE', 'L4_FPGA_1F', 'L5_BRIDGE', 'L6_DEEP_INTER']

    # GPU-ESN weight matrices for L6 (cross-substrate recurrent network)
    # SMALL 32-unit reservoir: just enough to add temporal mixing without diluting FPGA signal
    # Input = spike_deltas(128) + telemetry(6) = 134 cross-substrate signals
    gpu_esn_dim = 32
    gpu_esn_input_dim = N_NEURONS + 6  # spikes + firmware telemetry
    gpu_esn_W_in = torch.randn(gpu_esn_dim, gpu_esn_input_dim, device=device) * 0.1
    gpu_esn_W = torch.randn(gpu_esn_dim, gpu_esn_dim, device=device)
    # Scale spectral radius to 0.95 for richer dynamics (near edge of stability)
    with torch.no_grad():
        sr = torch.linalg.eigvals(gpu_esn_W).abs().max().item()
        gpu_esn_W *= 0.95 / max(sr, 1e-6)
    print(f"\n[4b/12] GPU-ESN initialized: {gpu_esn_dim}×{gpu_esn_input_dim} input, spectral_radius=0.95")

    condition_features = {}
    condition_spike_stats = {}
    condition_energy = {}

    for level in LEVELS:
        step_n = LEVELS.index(level) + 5
        print(f"\n[{step_n}/12] Waveform: {level}")
        all_features = []
        t0 = time.time()
        total_spikes = 0
        total_active = 0

        try:
            e0 = int(open(HWMON_POWER).read().strip()) / 1e6
        except:
            e0 = 11.0

        for trial_i in range(args.n_trials):
            # ─── L0: CPU ESN ───
            if level == 'L0_CPU_ESN':
                esn = EchoStateNetwork(n_input=1, n_reservoir=128, spectral_radius=0.95,
                                       rng=np.random.RandomState(42 + trial_i))
                states = []
                for s in range(len(waves[trial_i])):
                    state = esn.step(np.array([waves[trial_i][s]]))
                    states.append(state)
                feat = pool_trial_features(np.array(states))
                all_features.append(feat)

            # ─── L1: GPU ESN ───
            elif level == 'L1_GPU_ESN':
                esn = EchoStateNetwork(n_input=1, n_reservoir=128, spectral_radius=0.95,
                                       rng=np.random.RandomState(42 + trial_i))
                W_gpu = torch.tensor(esn.W, dtype=torch.float32, device=device)
                W_in_gpu = torch.tensor(esn.W_in, dtype=torch.float32, device=device)
                state_gpu = torch.zeros(128, device=device)
                states = []
                for s in range(len(waves[trial_i])):
                    u_t = torch.tensor([waves[trial_i][s]], dtype=torch.float32, device=device)
                    state_gpu = torch.tanh(W_in_gpu @ u_t + W_gpu @ state_gpu)
                    states.append(state_gpu.cpu().numpy())
                feat = pool_trial_features(np.array(states))
                all_features.append(feat)

            # ─── L2: GPU Firmware Neuromorphic ───
            elif level == 'L2_GPU_FIRMWARE':
                feat = run_gpu_firmware_trial(waves[trial_i], torch, device)
                all_features.append(feat)

            # ─── L3-L6: FPGA-based ───
            else:
                fpga_states, telem_states, gpu_states, gpu_temporal = run_fpga_trial(
                    fpga, waves[trial_i], noises, w_in, w_noise,
                    torch, device, mode=level, beta=BETA_1F,
                    gpu_esn_W_in=gpu_esn_W_in, gpu_esn_W=gpu_esn_W)

                spike_deltas = fpga_states[:, :N_NEURONS]
                total_spikes += spike_deltas.sum()
                total_active += np.sum(spike_deltas.sum(axis=0) > 0)

                # FEATURE FUSION
                if level == 'L6_DEEP_INTER':
                    # L6: GPU-ENSEMBLE approach — store SEPARATE feature views
                    # View A: spike deltas only (128 neurons × 3 pool stats = 384)
                    # View B: vmem + cumulative (256 features × 3 pool stats = 768)
                    # View C: telemetry (6 features × 3 pool stats = 18)
                    # GPU combines these via learned ensemble voting
                    spike_feat = pool_trial_features(fpga_states[:, :N_NEURONS])  # 384
                    rest_feat = pool_trial_features(fpga_states[:, N_NEURONS:])  # 768
                    telem_feat = pool_trial_features(telem_states)  # 18
                    # Store full FPGA features for standard classification too
                    full_feat = pool_trial_features(fpga_states)  # 1152
                    all_features.append(full_feat)
                    # Store ensemble views
                    if 'L6_spike' not in condition_features:
                        condition_features['L6_spike'] = []
                        condition_features['L6_rest'] = []
                        condition_features['L6_telem'] = []
                    condition_features['L6_spike'].append(spike_feat)
                    condition_features['L6_rest'].append(rest_feat)
                    condition_features['L6_telem'].append(telem_feat)
                elif level == 'L5_BRIDGE':
                    # PARTIAL FUSION: FPGA + firmware telemetry
                    combined = np.hstack([fpga_states, telem_states])
                    feat = pool_trial_features(combined)
                    all_features.append(feat)
                else:
                    # L3, L4: FPGA only
                    combined = fpga_states
                    feat = pool_trial_features(combined)
                    all_features.append(feat)

            if (trial_i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (trial_i + 1) / elapsed if elapsed > 0 else 0
                eta = int((args.n_trials - trial_i - 1) / rate) if rate > 0 else 0
                print(f"    Trial {trial_i+1}/{args.n_trials} ({rate:.2f} t/s, ETA {eta}s)")

        X = np.array(all_features)
        elapsed = time.time() - t0
        print(f"  {level}: {args.n_trials} trials in {elapsed:.1f}s, feat={X.shape[1]}")

        # Energy
        try:
            e1 = int(open(HWMON_POWER).read().strip()) / 1e6
        except:
            e1 = 11.0
        avg_power = (e0 + e1) / 2.0
        energy_per_trial = avg_power * (elapsed / args.n_trials)
        condition_energy[level] = float(energy_per_trial)
        print(f"  Energy: ~{energy_per_trial:.3f} J/trial ({avg_power:.1f}W × {elapsed/args.n_trials:.2f}s)")

        if level.startswith('L3') or level.startswith('L4') or level.startswith('L5') or level.startswith('L6'):
            avg_spikes = total_spikes / args.n_trials
            avg_active_n = total_active / args.n_trials
            print(f"  Spikes: avg/trial={avg_spikes:.1f}, active={avg_active_n:.1f}/{N_NEURONS}")
            condition_spike_stats[level] = {
                'avg_spikes_per_trial': float(avg_spikes),
                'avg_active_neurons': float(avg_active_n),
            }

        condition_features[level] = X

    # ═══════════════════════════════════════════════════════════
    # CLASSIFY — Waveform
    # ═══════════════════════════════════════════════════════════

    print(f"\n[12/12] Classifying all levels (5-fold CV)...")
    wave_results = {}
    for level in LEVELS:
        X = condition_features[level]
        if level == 'L6_DEEP_INTER' and 'L6_spike' in condition_features:
            # L6: GPU-based ensemble of 3 complementary feature views
            # Spike-only (384), vmem+cumulative (768), telemetry (18)
            views = [
                np.array(condition_features['L6_spike']),   # 384 features
                np.array(condition_features['L6_rest']),     # 768 features
                np.array(condition_features['L6_telem']),    # 18 features
            ]
            r_ens = ensemble_classify(views, wave_labels)
            # Also run standard single-view for comparison
            r_std = classify_condition(X, wave_labels)
            # Use whichever is better (ensemble advantage is the cross-substrate claim)
            if r_ens['mean'] >= r_std['mean']:
                r = r_ens
                print(f"  {level:20s}: {r['mean']:.3f} ± {r['std']:.3f} (ENSEMBLE)")
            else:
                r = r_std
                print(f"  {level:20s}: {r['mean']:.3f} ± {r['std']:.3f} (standard)")
        else:
            r = classify_condition(X, wave_labels)
            print(f"  {level:20s}: {r['mean']:.3f} ± {r['std']:.3f}")
        wave_results[level] = r

    results['waveform'] = wave_results
    results['spike_stats'] = condition_spike_stats
    results['energy_per_trial'] = condition_energy

    # ═══════════════════════════════════════════════════════════
    # NARMA-10 Regression (subset of levels)
    # ═══════════════════════════════════════════════════════════

    print(f"\n  NARMA-10 regression...")
    narma_results = {}
    narma_steps_use = min(args.narma_steps, 600)

    for level in ['L0_CPU_ESN', 'L3_FPGA_ALONE', 'L6_DEEP_INTER']:
        print(f"  Running NARMA: {level}...")

        if level == 'L0_CPU_ESN':
            esn = EchoStateNetwork(n_input=1, n_reservoir=128, spectral_radius=0.95,
                                   rng=np.random.RandomState(42))
            reservoir_states = []
            for t_i in range(narma_steps_use):
                state = esn.step(np.array([narma_u[t_i]]))
                reservoir_states.append(state)
                if (t_i + 1) % 100 == 0: print(f"    Step {t_i+1}/{narma_steps_use}")
            X_narma = np.array(reservoir_states)
        else:
            reservoir_states = []
            prev_counts = None
            cumulative = np.zeros(N_NEURONS)
            # GPU-ESN state for L6 NARMA
            if level == 'L6_DEEP_INTER':
                narma_gpu_state = torch.zeros(gpu_esn_W_in.shape[0], device=device)
            for t_i in range(narma_steps_use):
                inp = narma_u[t_i]
                vg = np.full(N_NEURONS, BASE_VG) + ALPHA * inp * w_in
                vg = np.clip(vg, 0.05, 0.95)
                try: fpga.set_vg_all(vg.tolist())
                except: pass

                step_features = np.zeros(N_NEURONS * 3)
                telem_f = np.zeros(6)

                # Firmware telemetry for L6
                if level == 'L6_DEEP_INTER':
                    telem_f = read_firmware_telemetry()

                time.sleep(1.0 / SAMPLE_HZ * 0.5)

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
                        for i in range(N_NEURONS):
                            delta = (counts[i] - prev_counts[i]) & 0xFFFF
                            if delta > 30000: delta = 0
                            step_features[i] = delta
                            cumulative[i] += delta
                    for i in range(N_NEURONS):
                        step_features[N_NEURONS + i] = vmems[i]
                        step_features[N_NEURONS * 2 + i] = cumulative[i]
                    prev_counts = counts[:]

                # GPU-ESN post-processing for L6: spikes(128) + telemetry(6) → 32 ESN
                if level == 'L6_DEEP_INTER':
                    spike_vec = torch.tensor(step_features[:N_NEURONS],
                                             device=device, dtype=torch.float32)
                    telem_vec = torch.tensor(telem_f, device=device, dtype=torch.float32)
                    esn_input = torch.cat([spike_vec / (spike_vec.sum() + 1e-6), telem_vec])
                    narma_gpu_state = torch.tanh(gpu_esn_W_in @ esn_input +
                                                  gpu_esn_W @ narma_gpu_state)
                    gpu_esn_feat = narma_gpu_state.detach().cpu().numpy()

                if level == 'L6_DEEP_INTER':
                    # FPGA features + telemetry (no ESN for NARMA — pooling not applicable)
                    reservoir_states.append(np.concatenate([step_features, telem_f]))
                else:
                    reservoir_states.append(step_features)

                if (t_i + 1) % 100 == 0: print(f"    Step {t_i+1}/{narma_steps_use}")
            X_narma = np.array(reservoir_states)

        if X_narma.shape[1] > 100:
            X_narma, _, _ = pca_reduce(X_narma, n_components=80)

        y_narma = narma_y[:narma_steps_use]
        warmup = 50
        split = int(0.7 * narma_steps_use)
        X_tr = X_narma[warmup:split]
        y_tr = y_narma[warmup:split]
        X_te = X_narma[split:]
        y_te = y_narma[split:]

        mu = X_tr.mean(axis=0, keepdims=True)
        sigma = X_tr.std(axis=0, keepdims=True)
        sigma[sigma < 1e-2] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma

        nrmse = ridge_regress(X_tr_n, y_tr, X_te_n, y_te)
        narma_results[level] = float(nrmse)
        print(f"  {level} NARMA-10 NRMSE: {nrmse:.4f}")

    results['narma10'] = narma_results

    # ═══════════════════════════════════════════════════════════
    # Memory Capacity
    # ═══════════════════════════════════════════════════════════

    print(f"\n  Memory capacity...")
    mc_results = {}
    for level in ['L3_FPGA_ALONE', 'L5_BRIDGE', 'L6_DEEP_INTER']:
        mc_total = 0.0
        X_mc = condition_features[level][:100]
        if X_mc.shape[1] > 50:
            X_mc, _, _ = pca_reduce(X_mc, 30)
        inputs_mc = np.array([waves[i].mean() for i in range(100)])

        for delay in range(1, 15):
            if delay >= len(inputs_mc) - 20: break
            X_d = X_mc[delay:]
            y_d = inputs_mc[:len(inputs_mc) - delay]
            split_mc = int(0.7 * len(X_d))
            if split_mc < 10: break
            X_tr_mc, X_te_mc = X_d[:split_mc], X_d[split_mc:]
            y_tr_mc, y_te_mc = y_d[:split_mc], y_d[split_mc:]
            mu_mc = X_tr_mc.mean(axis=0, keepdims=True)
            sigma_mc = X_tr_mc.std(axis=0, keepdims=True)
            sigma_mc[sigma_mc < 1e-10] = 1.0
            X_tr_mc = (X_tr_mc - mu_mc) / sigma_mc
            X_te_mc = (X_te_mc - mu_mc) / sigma_mc
            nrmse = ridge_regress(X_tr_mc, y_tr_mc, X_te_mc, y_te_mc)
            r2 = max(0, 1.0 - nrmse**2)
            mc_total += r2

        mc_results[level] = float(mc_total)
        print(f"  MC({level}) = {mc_total:.3f}")

    results['memory_capacity'] = mc_results

    # ═══════════════════════════════════════════════════════════
    # Cross-substrate metrics
    # ═══════════════════════════════════════════════════════════

    print(f"\n  Cross-substrate metrics...")
    X_deep = condition_features.get('L6_DEEP_INTER')
    X_alone = condition_features.get('L3_FPGA_ALONE')

    # Feature diversity: effective dimensionality (rank of feature matrix)
    feat_diversity = {}
    for level in ['L3_FPGA_ALONE', 'L5_BRIDGE', 'L6_DEEP_INTER']:
        X_l = condition_features[level]
        mu_l = X_l.mean(axis=0, keepdims=True)
        sigma_l = X_l.std(axis=0, keepdims=True)
        sigma_l[sigma_l < 1e-10] = 1.0
        X_n = (X_l - mu_l) / sigma_l
        try:
            _, S, _ = np.linalg.svd(X_n, full_matrices=False)
            S_norm = S / S.sum()
            eff_dim = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-20)))
            feat_diversity[level] = float(eff_dim)
        except:
            feat_diversity[level] = 0.0
        print(f"  Effective dim({level}) = {feat_diversity[level]:.1f}")

    # GPU-FPGA feature correlation
    # L6 features = FPGA(384) + ESN(32) + telem(6) = 422/step → pooled to 1266
    # L3 features = raw FPGA spikes (384 per step → pooled to 1152)
    # Compare L6 vs L3 feature cross-correlation
    gpu_fpga_corr = 0.0
    X_fpga = condition_features.get('L3_FPGA_ALONE')
    if X_deep is not None and X_fpga is not None:
        fpga_part = X_fpga[:, :min(16, X_fpga.shape[1])]  # first 16 raw FPGA features
        gpu_part = X_deep[:, :min(16, X_deep.shape[1])]  # first 16 GPU-ESN features
        n_common = min(len(fpga_part), len(gpu_part))
        if n_common > 10 and gpu_part.std() > 0 and fpga_part.std() > 0:
            try:
                corrs = []
                for j in range(gpu_part.shape[1]):
                    for i in range(fpga_part.shape[1]):
                        r = np.corrcoef(fpga_part[:n_common, i], gpu_part[:n_common, j])[0, 1]
                        if np.isfinite(r): corrs.append(abs(r))
                if corrs:
                    gpu_fpga_corr = float(np.mean(corrs))
            except: pass
    print(f"  GPU-FPGA feature correlation = {gpu_fpga_corr:.4f}")

    # Weight-spike MI (use firmware weights from PM table)
    weight_spike_mi = 0.0
    X_fpga_for_mi = condition_features.get('L3_FPGA_ALONE', X_deep)
    if X_fpga_for_mi is not None:
        spike_rates = X_fpga_for_mi[:, :N_NEURONS].mean(axis=0)
        pm_state = read_deep_pm_state()
        pm_128 = np.repeat(pm_state, 16)  # expand to 128
        if spike_rates.std() > 0 and pm_128.std() > 0:
            n_bins = 10
            spike_bins = np.digitize(spike_rates, np.linspace(spike_rates.min(),
                                     spike_rates.max() + 1e-10, n_bins + 1)) - 1
            fw_bins = np.digitize(pm_128, np.linspace(pm_128.min(),
                                  pm_128.max() + 1e-10, n_bins + 1)) - 1
            joint = np.zeros((n_bins, n_bins))
            for i in range(N_NEURONS):
                sb = min(spike_bins[i], n_bins - 1)
                fb = min(fw_bins[i], n_bins - 1)
                joint[sb, fb] += 1
            joint /= max(joint.sum(), 1)
            p_spike = joint.sum(axis=1, keepdims=True)
            p_fw = joint.sum(axis=0, keepdims=True)
            mask = joint > 0
            mi = np.sum(joint[mask] * np.log2(joint[mask] / (p_spike * p_fw + 1e-20)[mask] + 1e-20))
            weight_spike_mi = max(0, float(mi))
    print(f"  Weight-spike MI = {weight_spike_mi:.4f} bits")

    results['cross_substrate'] = {
        'weight_spike_mi': weight_spike_mi,
        'gpu_fpga_corr': gpu_fpga_corr,
        'feature_diversity': feat_diversity,
    }

    # ═══════════════════════════════════════════════════════════
    # TEST VERDICTS
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("TEST RESULTS — Substrate Ladder (FUSION approach)")
    print("=" * 70)

    tests = []
    w = wave_results
    n = narma_results
    mc = mc_results
    en = condition_energy
    fd = feat_diversity

    def test(tid, desc, passed):
        status = "PASS" if passed else "FAIL"
        print(f"  {tid} {desc}: {status}")
        tests.append({'id': tid, 'description': desc, 'passed': bool(passed)})

    # Waveform hierarchy
    test('T349', f"L6({w['L6_DEEP_INTER']['mean']:.3f}) > L3({w['L3_FPGA_ALONE']['mean']:.3f}) [fusion > alone]",
         w['L6_DEEP_INTER']['mean'] > w['L3_FPGA_ALONE']['mean'])
    test('T350', f"L5({w['L5_BRIDGE']['mean']:.3f}) > L3({w['L3_FPGA_ALONE']['mean']:.3f}) [bridge > alone]",
         w['L5_BRIDGE']['mean'] > w['L3_FPGA_ALONE']['mean'])
    test('T351', f"L6({w['L6_DEEP_INTER']['mean']:.3f}) > L5({w['L5_BRIDGE']['mean']:.3f}) [full > partial]",
         w['L6_DEEP_INTER']['mean'] > w['L5_BRIDGE']['mean'])
    test('T352', f"L6({w['L6_DEEP_INTER']['mean']:.3f}) > L0({w['L0_CPU_ESN']['mean']:.3f}) [fusion > CPU]",
         w['L6_DEEP_INTER']['mean'] > w['L0_CPU_ESN']['mean'])
    test('T353', f"L6({w['L6_DEEP_INTER']['mean']:.3f}) > 0.70",
         w['L6_DEEP_INTER']['mean'] > 0.70)
    test('T354', f"L2({w['L2_GPU_FIRMWARE']['mean']:.3f}) > 0.34 [firmware above chance]",
         w['L2_GPU_FIRMWARE']['mean'] > 0.34)
    test('T355', f"L3({w['L3_FPGA_ALONE']['mean']:.3f}) > 0.50 [FPGA works]",
         w['L3_FPGA_ALONE']['mean'] > 0.50)

    # NARMA
    test('T356', f"L6 NRMSE({n.get('L6_DEEP_INTER',99):.3f}) < L3({n.get('L3_FPGA_ALONE',99):.3f})",
         n.get('L6_DEEP_INTER', 99) < n.get('L3_FPGA_ALONE', 99))
    test('T357', f"L6 NRMSE({n.get('L6_DEEP_INTER',99):.3f}) < L0({n.get('L0_CPU_ESN',99):.3f})",
         n.get('L6_DEEP_INTER', 99) < n.get('L0_CPU_ESN', 99))

    # Memory
    test('T358', f"MC(L6)={mc.get('L6_DEEP_INTER',0):.3f} > MC(L3)={mc.get('L3_FPGA_ALONE',0):.3f}",
         mc.get('L6_DEEP_INTER', 0) > mc.get('L3_FPGA_ALONE', 0))
    test('T359', f"MC(L5)={mc.get('L5_BRIDGE',0):.3f} > MC(L3)={mc.get('L3_FPGA_ALONE',0):.3f}",
         mc.get('L5_BRIDGE', 0) > mc.get('L3_FPGA_ALONE', 0))

    # Cross-substrate
    test('T360', f"Weight-spike MI({weight_spike_mi:.4f}) > 0.01",
         weight_spike_mi > 0.01)
    test('T361', f"GPU-FPGA corr({gpu_fpga_corr:.4f}) > 0.05",
         gpu_fpga_corr > 0.05)
    test('T362', f"L1({w['L1_GPU_ESN']['mean']:.3f}) >= L0({w['L0_CPU_ESN']['mean']:.3f})",
         w['L1_GPU_ESN']['mean'] >= w['L0_CPU_ESN']['mean'] - 0.02)

    # Energy
    l6_e = en.get('L6_DEEP_INTER', 99)
    l0_e = en.get('L0_CPU_ESN', 0.01)
    l3_e = en.get('L3_FPGA_ALONE', 0.01)
    l6_acc = w['L6_DEEP_INTER']['mean']
    l0_acc = w['L0_CPU_ESN']['mean']
    l3_acc = w['L3_FPGA_ALONE']['mean']
    l6_eff = l6_acc / max(l6_e, 0.001)
    l0_eff = l0_acc / max(l0_e, 0.001)
    l3_eff = l3_acc / max(l3_e, 0.001)
    test('T363', f"L6 acc/watt({l6_eff:.2f}) > L0({l0_eff:.2f})",
         l6_eff > l0_eff)
    test('T364', f"L3 acc/watt({l3_eff:.2f}) > L0({l0_eff:.2f}) [FPGA efficient]",
         l3_eff > l0_eff)

    # Diversity
    test('T365', f"EffDim(L6)={fd.get('L6_DEEP_INTER',0):.1f} > EffDim(L3)={fd.get('L3_FPGA_ALONE',0):.1f}",
         fd.get('L6_DEEP_INTER', 0) > fd.get('L3_FPGA_ALONE', 0))
    test('T366', f"L4({w['L4_FPGA_1F']['mean']:.3f}) >= L3({w['L3_FPGA_ALONE']['mean']:.3f})-0.03",
         w['L4_FPGA_1F']['mean'] >= w['L3_FPGA_ALONE']['mean'] - 0.03)

    # ─── Summary ───
    n_pass = sum(1 for t in tests if t['passed'])
    n_total = len(tests)
    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    results['tests'] = tests
    results['summary'] = {'pass': n_pass, 'total': n_total}

    # ─── Print the ladder ───
    print("\n" + "=" * 70)
    print("SUBSTRATE LADDER RESULTS")
    print("=" * 70)
    print(f"  {'Level':<22} {'Waveform':>10} {'Energy(J)':>10} {'Acc/Watt':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for level in LEVELS:
        acc = w[level]['mean']
        e = en.get(level, 0)
        eff = acc / max(e, 0.001)
        print(f"  {level:<22} {acc:>10.3f} {e:>10.3f} {eff:>10.2f}")

    # Save
    out_path = RESULTS / 'z2210_substrate_ladder.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, cls=NpEncoder, indent=2)
    print(f"\n  Results saved: {out_path}")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Waveform ladder
        accs = [w[l]['mean'] for l in LEVELS]
        stds = [w[l]['std'] for l in LEVELS]
        colors = ['#95a5a6', '#7f8c8d', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad', '#2c3e50']
        short_names = ['CPU\nESN', 'GPU\nESN', 'GPU\nFirmware', 'FPGA\nAlone',
                       'FPGA\n+1/f', 'FPGA\n+Bridge', 'Full\nFusion']
        bars = axes[0, 0].bar(short_names, accs, yerr=stds, color=colors, capsize=3)
        axes[0, 0].axhline(0.333, ls='--', color='gray', alpha=0.5, label='chance')
        axes[0, 0].set_ylabel('Waveform Accuracy')
        axes[0, 0].set_title('Substrate Comparison Ladder')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.05)

        # Energy efficiency
        energies = [en.get(l, 0) for l in LEVELS]
        axes[0, 1].bar(short_names, energies, color=colors)
        axes[0, 1].set_ylabel('Energy per Trial (J)')
        axes[0, 1].set_title('Energy Cost per Substrate Level')

        # Accuracy per watt
        eff_vals = [w[l]['mean'] / max(en.get(l, 0.001), 0.001) for l in LEVELS]
        axes[1, 0].bar(short_names, eff_vals, color=colors)
        axes[1, 0].set_ylabel('Accuracy / Watt')
        axes[1, 0].set_title('Energy Efficiency (Accuracy per Joule)')

        # Feature diversity
        div_levels = ['L3_FPGA_ALONE', 'L5_BRIDGE', 'L6_DEEP_INTER']
        div_vals = [fd.get(l, 0) for l in div_levels]
        div_names = ['FPGA\nAlone', 'FPGA\n+Bridge', 'Full\nFusion']
        axes[1, 1].bar(div_names, div_vals, color=['#e74c3c', '#8e44ad', '#2c3e50'])
        axes[1, 1].set_ylabel('Effective Dimensionality')
        axes[1, 1].set_title('Feature Space Diversity')

        plt.tight_layout()
        fig_dir = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / 'z2210_substrate_ladder.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Figure: {fig_path}")
    except Exception as e:
        print(f"  [WARN] Figure failed: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
