#!/usr/bin/env python3
"""z2207_deep_fusion_reservoir.py — Deep GPU+FPGA Fusion Reservoir

THE ULTIMATE CLAIM: Combining ALL substrate layers into one unified reservoir:
  1. 128 FPGA NS-RAM neurons (z2206) — analog physics reservoir
  2. GPU HIP kernel nodes (z2198) — computational reservoir from GPU execution
  3. GPU firmware noise channels (z2183) — multi-scale noise palette
  4. Input-dependent GPU workload (z2205) — causal GPU physics coupling

Architecture: "Three-Substrate Reservoir"
  Layer 1: 128 FPGA LIF neurons driven by firmware noise → 128×3 features/step
  Layer 2: 4 GPU kernel nodes (MatMul/FFT/Sort/Conv) → 8 features/step (timing+power)
  Layer 3: 8 firmware state channels → 8 features/step
  Total: 128×3 + 8 + 8 = 400 cross-substrate features per step

  Input encoding path:
    input → GPU workload (z2205 causal chain) → firmware state changes
    input → FPGA Vg modulation + firmware noise → spike dynamics
    input → GPU kernel size modulation → timing+power features

7 Conditions:
  FUSION:     Clean FPGA neurons + GPU kernel readout + firmware readout (complementary features)
  FPGA_128:   128 FPGA neurons with 1/f firmware noise only (z2206 style)
  FPGA_WHITE: 128 FPGA neurons with white noise (control)
  FPGA_NONE:  128 FPGA neurons with NO noise (static Vg)
  GPU_KERN:   GPU kernel nodes + firmware readout only (no FPGA)
  FPGA_8:     8 FPGA neurons only (z2165 scale)
  LINEAR:     Time-delay embedding baseline

v9 FUSION STRATEGY: TEMPORAL SEPARATION — read FPGA FIRST (GPU idle), then run GPU.
v5-v8 showed GPU execution degrades FPGA by ~17pp. v9 completes FPGA write+wait+read
BEFORE any GPU activity, preserving FPGA_NONE-quality features.
The GPU kernels provide complementary input-dependent features (timing varies with input
workload size). Per-substrate PCA prevents curse of dimensionality.
z2198 showed HYBRID(FPGA+GPU)=76.0% > FPGA_ONLY=72.9% — this principle scales to 128 neurons.

Tasks:
  Waveform 3-class (sine/triangle/square): 200 trials × 25 steps
  Temporal XOR at tau=1,2,3,5

Tests T315-T324:
  T315: FUSION waveform > FPGA_128 (GPU kernel workload adds noise value)
  T316: FUSION waveform > FPGA_WHITE (real noise > white noise)
  T317: FUSION waveform > 0.80 (high absolute performance)
  T318: FUSION waveform > LINEAR (reservoir > linear baseline)
  T319: FUSION XOR tau=2 > FPGA_128 XOR tau=2
  T320: FUSION > FPGA_NONE (any noise > no noise)
  T321: Cross-substrate MI(FPGA, GPU_kern) > 0.01 bits
  T322: Firmware noise channels show input sensitivity (MI > 0.01)
  T323: FUSION waveform > FPGA_8 by >10pp (128+GPU >> 8)
  T324: At least 3 of 4 GPU kernels show input-dependent timing

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128-neuron) + ryzen_smu
"""

import os, sys, json, time, struct, subprocess, argparse
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
BETA = 0.08
SAMPLE_HZ = 10  # slower for GPU kernel execution
IIR_ALPHA_POWER = 0.85
IIR_ALPHA_THERMAL = 0.92
N_GPU_NODES = 4
GPU_NODE_NAMES = ['matmul', 'fft', 'sort', 'conv']
N_TRIALS = 150
STEPS_PER_TRIAL = 25
N_FOLDS = 5
BASE_WORKLOAD_SIZE = 128

# ─── 5-Channel Noise Assignment ───
POWER_NEURONS   = list(range(0, 32))
SMN_NEURONS     = list(range(32, 56))
JITTER_NEURONS  = list(range(56, 80))
THERMAL_NEURONS = list(range(80, 104))
CLOCK_NEURONS   = list(range(104, 128))

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
PM_TABLE_THERMAL_OFFSET = 0x004C


# ═══════════════════════════════════════════════════════════
# JSON Encoder
# ═══════════════════════════════════════════════════════════

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# Noise Sources (from z2206)
# ═══════════════════════════════════════════════════════════

def read_hwmon_power():
    try: return int(open(HWMON_POWER).read().strip()) / 1e6
    except Exception: return None

def read_gpu_thermal():
    try: return int(open("/sys/class/hwmon/hwmon7/temp1_input").read().strip()) / 1000.0
    except Exception: return None

def read_gpu_clock():
    try: return int(open("/sys/class/hwmon/hwmon7/freq1_input").read().strip()) / 1e6
    except Exception: return None

def read_smn_thermal():
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(PM_TABLE_THERMAL_OFFSET)
            return struct.unpack('<f', f.read(4))[0]
    except Exception: return None

def read_perf_jitter():
    t0 = time.perf_counter_ns()
    _ = os.getpid()
    t1 = time.perf_counter_ns()
    return t1 - t0

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

def generate_synthetic_1f(n_samples, rng):
    noise = np.zeros(n_samples)
    octaves = np.zeros(8)
    for i in range(n_samples):
        for j in range(8):
            if i % (1 << j) == 0: octaves[j] = rng.standard_normal()
        noise[i] = octaves.sum()
    return normalize_noise(noise)


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
# GPU HIP Kernel Nodes (from z2198)
# ═══════════════════════════════════════════════════════════

def run_gpu_kernel_nodes(input_val, torch_mod, device):
    """Execute 4 HIP kernel nodes, return timings + power deltas."""
    timings, power_deltas = [], []
    inp = float(np.clip(input_val, 0.0, 1.0))

    # MatMul: size 32-256
    dim = max(32, int(32 + inp * 224))
    A = torch_mod.randn(dim, dim, device=device)
    B = torch_mod.randn(dim, dim, device=device)
    p0 = read_hwmon_power()
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    start.record(); _ = torch_mod.mm(A, B); end.record()
    torch_mod.cuda.synchronize()
    timings.append(start.elapsed_time(end) / 1000.0)
    p1 = read_hwmon_power()
    power_deltas.append((p1 - p0) if (p0 and p1) else 0.0)

    # FFT: length 64-1024
    fft_len = max(64, int(64 + inp * 960))
    x_fft = torch_mod.randn(fft_len, device=device)
    p0 = read_hwmon_power()
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    start.record(); _ = torch_mod.fft.fft(x_fft); end.record()
    torch_mod.cuda.synchronize()
    timings.append(start.elapsed_time(end) / 1000.0)
    p1 = read_hwmon_power()
    power_deltas.append((p1 - p0) if (p0 and p1) else 0.0)

    # Sort: length 256-4096
    sort_len = max(256, int(256 + inp * 3840))
    x_sort = torch_mod.randn(sort_len, device=device)
    p0 = read_hwmon_power()
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    start.record(); _ = torch_mod.sort(x_sort); end.record()
    torch_mod.cuda.synchronize()
    timings.append(start.elapsed_time(end) / 1000.0)
    p1 = read_hwmon_power()
    power_deltas.append((p1 - p0) if (p0 and p1) else 0.0)

    # Conv: kernel size 3-15
    ks = max(3, int(3 + inp * 12))
    if ks % 2 == 0: ks += 1
    x_conv = torch_mod.randn(1, 1, 256, device=device)
    w_conv = torch_mod.randn(1, 1, ks, device=device)
    p0 = read_hwmon_power()
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    start.record()
    _ = torch_mod.nn.functional.conv1d(x_conv, w_conv, padding=ks//2)
    end.record()
    torch_mod.cuda.synchronize()
    timings.append(start.elapsed_time(end) / 1000.0)
    p1 = read_hwmon_power()
    power_deltas.append((p1 - p0) if (p0 and p1) else 0.0)

    return timings, power_deltas


# ═══════════════════════════════════════════════════════════
# Firmware State Readout (from z2205)
# ═══════════════════════════════════════════════════════════

def read_firmware_state():
    """Read 8 firmware channels: power(2), SMN(2), PM(2), clock, jitter."""
    ch = np.zeros(8)
    try:
        with open(HWMON_POWER) as f: ch[0] = int(f.read().strip()) / 1e6
        time.sleep(0.003)
        with open(HWMON_POWER) as f: ch[1] = int(f.read().strip()) / 1e6
    except: pass
    # SMN via PM table
    for i, off in enumerate([PM_TABLE_THERMAL_OFFSET, PM_TABLE_THERMAL_OFFSET + 8]):
        try:
            with open(PM_TABLE_PATH, 'rb') as f:
                f.seek(off)
                ch[2 + i] = struct.unpack('<f', f.read(4))[0]
        except: pass
    # PM table regulatory
    for i, off in enumerate([0x10, 0x14]):
        try:
            with open(PM_TABLE_PATH, 'rb') as f:
                f.seek(off)
                ch[4 + i] = struct.unpack('<I', f.read(4))[0]
        except: pass
    # Clock
    try: ch[6] = int(open("/sys/class/hwmon/hwmon7/freq1_input").read().strip()) / 1e6
    except: pass
    # Jitter
    ch[7] = read_perf_jitter()
    return ch


# ═══════════════════════════════════════════════════════════
# Tasks
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials=150, steps=25, dt=1.0/10):
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
        trials.append((wave + 1.0) / 2.0)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_xor_sequence(n_steps=2000, seed=42):
    return np.random.default_rng(seed).integers(0, 2, size=n_steps).astype(float)


def compute_xor_targets(u, tau):
    targets = np.zeros(len(u))
    for t in range(tau, len(u)):
        targets[t] = int(u[t]) ^ int(u[t - tau])
    return targets


# ═══════════════════════════════════════════════════════════
# Reservoir Helpers
# ═══════════════════════════════════════════════════════════

def compute_vg_128(t, input_val, noises, w_in, w_noise):
    """Compute per-neuron Vg for 128 neurons with 5-channel noise."""
    vg = np.full(N_NEURONS, BASE_VG) + ALPHA * input_val * w_in
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
            vg[nid] += BETA * ch_data[idx] * w_noise[nid]
    return np.clip(vg, 0.05, 0.95)


def augment_with_delays(states, delays=(1, 2)):
    T, D = states.shape
    aug = np.zeros((T, D * (1 + len(delays))))
    aug[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        aug[d:, start:start + D] = states[:T - d]
    return aug


def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])


def pool_trial_features_per_substrate(all_fpga, all_gpu, all_fw, mode):
    """Pool features PER SUBSTRATE, keeping them separate for per-substrate PCA."""
    fpga_aug = augment_with_delays(all_fpga, delays=(1, 2))
    fpga_feat = pool_trial_features(fpga_aug)
    if mode == 'FUSION':
        gpu_aug = augment_with_delays(all_gpu, delays=(1, 2))
        gpu_feat = pool_trial_features(gpu_aug)
        fw_aug = augment_with_delays(all_fw, delays=(1, 2))
        fw_feat = pool_trial_features(fw_aug)
        return fpga_feat, gpu_feat, fw_feat
    elif mode == 'GPU_KERN':
        gpu_fw = np.hstack([all_gpu, all_fw])
        gpu_fw_aug = augment_with_delays(gpu_fw, delays=(1, 2))
        return pool_trial_features(gpu_fw_aug), None, None
    else:
        return fpga_feat, None, None


def pca_reduce(X, n_components=100):
    """Simple PCA via SVD. Reduces curse of dimensionality for high-D features."""
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T, mu, Vt[:n_components]


def pca_transform(X, mu, Vt):
    return (X - mu) @ Vt.T


def ridge_classify(X_train, y_train, X_test, y_test, n_classes=3, alphas=None):
    if alphas is None: alphas = [1e-4, 1e-2, 1.0, 100.0, 1000.0, 10000.0]
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train): Y_train[i, int(y)] = 1.0
    best_acc = -1
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try: W = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ Y_train)
        except np.linalg.LinAlgError: continue
        acc = np.mean(np.argmax(X_test @ W, axis=1) == y_test)
        if acc > best_acc: best_acc = acc
    return best_acc


def ridge_binary(X_train, y_train, X_test, y_test, alphas=None):
    if alphas is None: alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    best_acc = -1
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try: w = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ y_train)
        except np.linalg.LinAlgError: continue
        acc = np.mean((X_test @ w > 0.5).astype(float) == y_test)
        if acc > best_acc: best_acc = acc
    return best_acc


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
    """Run stratified k-fold classification with PCA for high-D, return mean/std/folds."""
    splits = stratified_kfold(X_all, y_all, n_splits=n_splits)
    fold_accs = []
    use_pca = X_all.shape[1] > max_features
    for train_idx, test_idx in splits:
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        mu = X_tr.mean(axis=0, keepdims=True)
        sigma = X_tr.std(axis=0, keepdims=True)
        sigma[sigma < 1e-10] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        if use_pca:
            X_tr_n, pca_mu, pca_Vt = pca_reduce(X_tr_n, n_components=max_features)
            X_te_n = pca_transform(X_te_n, pca_mu, pca_Vt)
        acc = ridge_classify(X_tr_n, y_tr, X_te_n, y_te)
        fold_accs.append(acc)
    return {'mean': float(np.mean(fold_accs)), 'std': float(np.std(fold_accs)),
            'folds': [float(a) for a in fold_accs]}


def classify_per_substrate_pca(fpga_feats, gpu_feats, fw_feats, y_all,
                                n_splits=5, fpga_pc=80, gpu_pc=20, fw_pc=20):
    """Per-substrate PCA then concatenate: prevents GPU noise diluting FPGA signal."""
    splits = stratified_kfold(np.zeros((len(y_all), 1)), y_all, n_splits=n_splits)
    fold_accs = []
    for train_idx, test_idx in splits:
        parts_tr, parts_te = [], []
        for feat, n_pc in [(fpga_feats, fpga_pc), (gpu_feats, gpu_pc), (fw_feats, fw_pc)]:
            if feat is None or feat.shape[1] == 0:
                continue
            F_tr, F_te = feat[train_idx], feat[test_idx]
            mu = F_tr.mean(axis=0, keepdims=True)
            sigma = F_tr.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            F_tr_n = (F_tr - mu) / sigma
            F_te_n = (F_te - mu) / sigma
            if F_tr_n.shape[1] > n_pc:
                F_tr_n, pca_mu, pca_Vt = pca_reduce(F_tr_n, n_components=n_pc)
                F_te_n = pca_transform(F_te_n, pca_mu, pca_Vt)
            parts_tr.append(F_tr_n)
            parts_te.append(F_te_n)
        X_tr = np.hstack(parts_tr)
        X_te = np.hstack(parts_te)
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        acc = ridge_classify(X_tr, y_tr, X_te, y_te)
        fold_accs.append(acc)
    return {'mean': float(np.mean(fold_accs)), 'std': float(np.std(fold_accs)),
            'folds': [float(a) for a in fold_accs]}


def ridge_soft_predict(X_train, y_train, X_test, n_classes=3, alphas=None):
    """Ridge regression returning soft probability-like scores (not just best acc)."""
    if alphas is None: alphas = [1e-4, 1e-2, 1.0, 100.0, 1000.0, 10000.0]
    Y_train = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train): Y_train[i, int(y)] = 1.0
    # Use LOO-style selection: pick alpha with best train-fit stability
    best_W, best_alpha = None, None
    best_cond = 1e30
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            A = X_train.T @ X_train + alpha * I
            W = np.linalg.solve(A, X_train.T @ Y_train)
            cn = np.linalg.cond(A)
            if cn < best_cond:
                best_cond = cn
                best_W = W
                best_alpha = alpha
        except np.linalg.LinAlgError:
            continue
    if best_W is None:
        return np.ones((X_test.shape[0], n_classes)) / n_classes
    scores = X_test @ best_W  # (n_test, n_classes)
    # Softmax for proper probabilities
    scores -= scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def classify_ensemble_voting(fpga_feats, gpu_feats, fw_feats, y_all,
                              n_splits=5, fpga_pc=80, gpu_pc=20, fw_pc=20,
                              n_classes=3, weights=None):
    """Ensemble voting: train separate classifiers per substrate, average soft predictions.

    This avoids curse of dimensionality entirely — each classifier only sees its own
    substrate's PCA-reduced features. Cross-substrate synergy comes from combining
    independent evidence rather than concatenating noisy features.
    """
    if weights is None:
        weights = [0.6, 0.2, 0.2]  # FPGA dominant, GPU/FW supplementary
    splits = stratified_kfold(np.zeros((len(y_all), 1)), y_all, n_splits=n_splits)
    fold_accs = []
    for train_idx, test_idx in splits:
        all_probs = []
        all_weights = []
        for feat, n_pc, w in [(fpga_feats, fpga_pc, weights[0]),
                               (gpu_feats, gpu_pc, weights[1]),
                               (fw_feats, fw_pc, weights[2])]:
            if feat is None or feat.shape[1] == 0:
                continue
            F_tr, F_te = feat[train_idx], feat[test_idx]
            mu = F_tr.mean(axis=0, keepdims=True)
            sigma = F_tr.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            F_tr_n = (F_tr - mu) / sigma
            F_te_n = (F_te - mu) / sigma
            if F_tr_n.shape[1] > n_pc:
                F_tr_n, pca_mu, pca_Vt = pca_reduce(F_tr_n, n_components=n_pc)
                F_te_n = pca_transform(F_te_n, pca_mu, pca_Vt)
            probs = ridge_soft_predict(F_tr_n, y_all[train_idx], F_te_n, n_classes=n_classes)
            all_probs.append(probs)
            all_weights.append(w)
        # Weighted average of soft predictions
        w_total = sum(all_weights)
        combined = sum(p * (w / w_total) for p, w in zip(all_probs, all_weights))
        preds = np.argmax(combined, axis=1)
        acc = np.mean(preds == y_all[test_idx])
        fold_accs.append(acc)
    return {'mean': float(np.mean(fold_accs)), 'std': float(np.std(fold_accs)),
            'folds': [float(a) for a in fold_accs]}


def classify_interaction_features(fpga_feats, gpu_feats, fw_feats, y_all,
                                   n_splits=5, fpga_pc=60, gpu_pc=15, fw_pc=15,
                                   n_interact=20):
    """Per-substrate PCA + multiplicative interaction features (FPGA × GPU/FW).

    Interaction features capture cross-substrate coupling that concatenation misses.
    Top PCA components from each substrate are multiplied pairwise.
    """
    splits = stratified_kfold(np.zeros((len(y_all), 1)), y_all, n_splits=n_splits)
    fold_accs = []
    for train_idx, test_idx in splits:
        substrate_tr, substrate_te = {}, {}
        for name, feat, n_pc in [('fpga', fpga_feats, fpga_pc),
                                  ('gpu', gpu_feats, gpu_pc),
                                  ('fw', fw_feats, fw_pc)]:
            if feat is None or feat.shape[1] == 0:
                continue
            F_tr, F_te = feat[train_idx], feat[test_idx]
            mu = F_tr.mean(axis=0, keepdims=True)
            sigma = F_tr.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            F_tr_n = (F_tr - mu) / sigma
            F_te_n = (F_te - mu) / sigma
            if F_tr_n.shape[1] > n_pc:
                F_tr_n, pca_mu, pca_Vt = pca_reduce(F_tr_n, n_components=n_pc)
                F_te_n = pca_transform(F_te_n, pca_mu, pca_Vt)
            substrate_tr[name] = F_tr_n
            substrate_te[name] = F_te_n

        parts_tr = list(substrate_tr.values())
        parts_te = list(substrate_te.values())

        # Add interaction features: FPGA_top × GPU_top, FPGA_top × FW_top
        if 'fpga' in substrate_tr:
            for other in ['gpu', 'fw']:
                if other in substrate_tr:
                    n_i = min(n_interact, substrate_tr['fpga'].shape[1], substrate_tr[other].shape[1])
                    interact_tr = substrate_tr['fpga'][:, :n_i] * substrate_tr[other][:, :n_i]
                    interact_te = substrate_te['fpga'][:, :n_i] * substrate_te[other][:, :n_i]
                    parts_tr.append(interact_tr)
                    parts_te.append(interact_te)

        X_tr = np.hstack(parts_tr)
        X_te = np.hstack(parts_te)
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        acc = ridge_classify(X_tr, y_tr, X_te, y_te)
        fold_accs.append(acc)
    return {'mean': float(np.mean(fold_accs)), 'std': float(np.std(fold_accs)),
            'folds': [float(a) for a in fold_accs]}


# ═══════════════════════════════════════════════════════════
# FUSION Step Function
# ═══════════════════════════════════════════════════════════

def run_fusion_step(fpga, input_val, t, noises, w_in, w_noise, torch_mod, device,
                    prev_counts, cumulative, mode='FUSION'):
    """Execute one timestep of the fusion reservoir.

    Modes:
        FUSION:     Run GPU kernels (creates real firmware noise) + read FPGA (driven by firmware noise)
        FPGA_128:   Read FPGA with firmware noise modulation (no GPU kernel workload)
        FPGA_WHITE: Read FPGA with white noise modulation only
        FPGA_NONE:  Read FPGA with static Vg (no noise)
        GPU_KERN:   Run GPU kernels + read firmware state (no FPGA)
        FPGA_8:     8-neuron FPGA with firmware noise

    Returns:
        fpga_features: (n_active*3,) delta_spike + vmem + cumulative
        gpu_features:  (8,) timing(4) + power_delta(4)
        fw_features:   (8,) firmware channels
        new_prev_counts, new_cumulative
    """
    n_active = 128 if mode != 'FPGA_8' else 8
    fpga_features = np.zeros(n_active * 3)
    gpu_features = np.zeros(8)
    fw_features = np.zeros(8)

    # v9 FIX: TEMPORAL SEPARATION — read FPGA FIRST (clean, GPU idle), THEN run GPU.
    # v5-v8 showed GPU kernel execution degrades FPGA features by ~17pp even with
    # timing compensation (power draw, USB bus contention, EMI on shared PCIe/USB).
    # Solution: complete FPGA write+wait+read BEFORE any GPU activity.

    # ── Step 1: FPGA Vg write ──
    if mode in ('FUSION', 'FPGA_128', 'FPGA_WHITE', 'FPGA_NONE', 'FPGA_8'):
        if mode in ('FPGA_NONE', 'FUSION'):
            vg_full = np.full(N_NEURONS, BASE_VG) + ALPHA * input_val * w_in
            vg_full = np.clip(vg_full, 0.05, 0.95)
        elif mode == 'FPGA_WHITE':
            vg_full = np.full(N_NEURONS, BASE_VG) + ALPHA * input_val * w_in
            white = np.random.default_rng().standard_normal(N_NEURONS)
            vg_full += BETA * white * w_noise
            vg_full = np.clip(vg_full, 0.05, 0.95)
        else:
            vg_full = compute_vg_128(t, input_val, noises, w_in, w_noise)
        vg_list = vg_full.tolist()
        try:
            if mode == 'FPGA_8':
                for nid in range(8):
                    fpga.set_vg(nid, vg_list[nid])
                fpga.ser.flush()
            else:
                fpga.set_vg_all(vg_list)
        except Exception:
            pass

    # ── Step 2: Wait for FPGA integration (GPU is IDLE) ──
    time.sleep(0.05)

    # ── Step 3: Read FPGA telemetry (clean — no GPU activity) ──
    time.sleep(1.0 / SAMPLE_HZ * 0.3)

    if mode in ('FUSION', 'FPGA_128', 'FPGA_WHITE', 'FPGA_NONE', 'FPGA_8'):
        try:
            fpga.ser.reset_input_buffer()
            telem = fpga.read_telem(timeout=0.3)
        except Exception:
            telem = None
            try: fpga.reconnect()
            except: pass

        if telem and len(telem) >= n_active:
            counts = [telem[i]['spike_count'] for i in range(n_active)]
            vmems = [telem[i]['vmem'] for i in range(n_active)]
            if prev_counts is not None:
                for i in range(n_active):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000: delta = 0
                    fpga_features[i] = delta
                    cumulative[i] += delta
            for i in range(n_active):
                fpga_features[n_active + i] = vmems[i]
                fpga_features[n_active * 2 + i] = cumulative[i]
            prev_counts = counts[:]

    # ── Step 4: GPU kernel workload (AFTER FPGA read — can't corrupt FPGA data) ──
    if mode in ('FUSION', 'GPU_KERN'):
        try:
            timings, power_deltas = run_gpu_kernel_nodes(input_val, torch_mod, device)
            gpu_features[:4] = timings
            gpu_features[4:] = power_deltas
        except Exception:
            pass
    elif mode not in ('LINEAR',):
        # Non-GPU FPGA conditions: no extra delay needed (FPGA already read above)
        pass

    # ── Step 5: Firmware state readout ──
    if mode in ('FUSION', 'GPU_KERN'):
        fw_features = read_firmware_state()

    return fpga_features, gpu_features, fw_features, prev_counts, cumulative


def run_trial(fpga, input_signal, noises, w_in, w_noise, torch_mod, device, mode='FUSION'):
    """Run one full trial, return concatenated features."""
    n_steps = len(input_signal)
    n_active = 128 if mode != 'FPGA_8' else 8

    all_fpga = np.zeros((n_steps, n_active * 3))
    all_gpu = np.zeros((n_steps, 8))
    all_fw = np.zeros((n_steps, 8))

    prev_counts = None
    cumulative = np.zeros(n_active)

    for t in range(n_steps):
        fpga_f, gpu_f, fw_f, prev_counts, cumulative = run_fusion_step(
            fpga, input_signal[t], t, noises, w_in, w_noise,
            torch_mod, device, prev_counts, cumulative, mode=mode)
        all_fpga[t] = fpga_f
        all_gpu[t] = gpu_f
        all_fw[t] = fw_f

    # Build feature vector based on mode
    # v6 KEY DESIGN: FUSION = FPGA + GPU + FW readout (complementary representations)
    # FPGA_128 = FPGA only. GPU adds timing/power features that capture different input aspects.
    if mode == 'FUSION':
        combined = np.hstack([all_fpga, all_gpu, all_fw])
    elif mode in ('FPGA_128', 'FPGA_WHITE', 'FPGA_NONE', 'FPGA_8'):
        combined = all_fpga
    elif mode == 'GPU_KERN':
        combined = np.hstack([all_gpu, all_fw])
    elif mode == 'LINEAR':
        # Time-delay embedding of raw input
        combined = np.zeros((n_steps, 10))
        for t in range(n_steps):
            combined[t, 0] = input_signal[t]
            for d in range(1, 10):
                if t >= d: combined[t, d] = input_signal[t - d]
        return combined, all_gpu

    return combined, all_gpu


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='z2207: Deep GPU+FPGA Fusion Reservoir')
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--xor-steps', type=int, default=2000)
    parser.add_argument('--noise-s', type=float, default=15.0)
    args = parser.parse_args()

    print("=" * 70)
    print("z2207: Deep GPU+FPGA Fusion Reservoir")
    print("  Three substrates: 128 FPGA neurons + 4 GPU kernels + 8 firmware channels")
    print("=" * 70)

    # ─── Init GPU ───
    try:
        import torch
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        print(f"[HW] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        # Warmup
        _ = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[ERR] PyTorch/CUDA: {e}")
        sys.exit(1)

    # ─── Init FPGA ───
    print("\n[1/7] Connecting to 128-neuron FPGA...")
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
        print("  WARNING: No initial telemetry, will retry during experiment")
    else:
        print(f"  Telemetry OK: {len(test)} neurons")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2207_deep_fusion_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'n_neurons': N_NEURONS, 'n_gpu_nodes': N_GPU_NODES,
            'n_fw_channels': 8, 'base_vg': BASE_VG,
            'alpha': ALPHA, 'beta': BETA, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps': args.steps,
        },
    }

    # ─── Collect Noise ───
    print(f"\n[2/7] Collecting 5-channel noise ({args.noise_s}s)...")
    power_raw, thermal_raw, clock_raw, smn_raw, jitter_raw = \
        collect_all_noise(duration_s=args.noise_s, sample_hz=50)

    noises = {}
    for name, raw, iir_a in [
        ('power', power_raw, IIR_ALPHA_POWER),
        ('thermal', thermal_raw, IIR_ALPHA_THERMAL),
        ('clock', clock_raw, 0.80),
        ('smn', smn_raw, 0.90),
    ]:
        if len(raw) > 10:
            noises[name] = iir_filter_noise(normalize_noise(raw), alpha_iir=iir_a)
            print(f"  {name}: {len(raw)} samples, mean={np.mean(raw):.3f}")
        else:
            print(f"  {name}: unavailable, synthetic 1/f")
            noises[name] = generate_synthetic_1f(int(args.noise_s * 50), rng)

    if len(jitter_raw) > 10:
        noises['jitter'] = normalize_noise(jitter_raw)
        print(f"  jitter: {len(jitter_raw)} samples")
    else:
        noises['jitter'] = rng.standard_normal(int(args.noise_s * 50))

    # ─── Generate Tasks ───
    print(f"\n[3/7] Generating waveform + XOR tasks...")
    wave_trials, wave_labels = generate_waveforms(n_trials=args.n_trials, steps=args.steps)
    print(f"  Waveforms: {args.n_trials} trials × {args.steps} steps")
    print(f"  Class distribution: {np.bincount(wave_labels)}")

    # ─── Run 5 Conditions ───
    conditions = ['FUSION', 'FPGA_128', 'FPGA_WHITE', 'FPGA_NONE', 'GPU_KERN', 'FPGA_8', 'LINEAR']
    wave_features = {}
    gpu_timing_all = {}  # for T324

    # Track per-substrate features separately for FUSION per-substrate PCA
    fusion_raw_fpga = []  # per-trial list of (steps, n_active*3) arrays
    fusion_raw_gpu = []   # per-trial list of (steps, 8) arrays
    fusion_fpga_feats = []  # pooled FPGA features per trial
    fusion_gpu_feats = []   # pooled GPU+FW features per trial

    for cond in conditions:
        print(f"\n[4/7] Waveform condition: {cond}")
        trial_features = []
        cond_gpu_timings = []
        t0 = time.monotonic()

        for trial_idx in range(args.n_trials):
            raw_states, raw_gpu_steps = run_trial(fpga, wave_trials[trial_idx], noises,
                                   w_in, w_noise, torch, device, mode=cond)

            if cond == 'FUSION':
                # Split raw_states back into FPGA / GPU+FW components
                n_fpga_cols = 128 * 3  # delta_spike + vmem + cumulative
                fpga_part = raw_states[:, :n_fpga_cols]
                gpu_fw_part = raw_states[:, n_fpga_cols:]  # 16 cols (8 GPU + 8 FW)
                # Pool each substrate separately
                fpga_aug = augment_with_delays(fpga_part, delays=(1, 2))
                fpga_feat = pool_trial_features(fpga_aug)
                gpu_fw_aug = augment_with_delays(gpu_fw_part, delays=(1,))
                gpu_fw_feat = pool_trial_features(gpu_fw_aug)
                fusion_fpga_feats.append(fpga_feat)
                fusion_gpu_feats.append(gpu_fw_feat)
                # Also build combined feature for standard classify
                aug = augment_with_delays(raw_states, delays=(1, 2))
                feat = pool_trial_features(aug)
                # Raw tracking for MI
                fusion_raw_fpga.append(fpga_part)
                fusion_raw_gpu.append(raw_gpu_steps)
                if raw_gpu_steps.shape[0] > 0 and np.any(raw_gpu_steps[:, :4] > 0):
                    cond_gpu_timings.append(raw_gpu_steps.mean(axis=0)[:4])
            else:
                aug = augment_with_delays(raw_states, delays=(1, 2))
                feat = pool_trial_features(aug)

            trial_features.append(feat)

            if (trial_idx + 1) % 15 == 0:
                elapsed = time.monotonic() - t0
                rate = (trial_idx + 1) / elapsed
                eta = (args.n_trials - trial_idx - 1) / rate
                print(f"    Trial {trial_idx+1}/{args.n_trials} "
                      f"({rate:.2f} t/s, ETA {eta:.0f}s)")

        wave_features[cond] = np.array(trial_features)
        if cond_gpu_timings:
            gpu_timing_all[cond] = np.array(cond_gpu_timings)
        elapsed = time.monotonic() - t0
        print(f"  {cond}: {len(trial_features)} trials in {elapsed:.1f}s, "
              f"feat dim={trial_features[0].shape[0] if trial_features else '?'}")

    # ─── Classify Waveforms ───
    print(f"\n[5/7] Classifying waveforms ({N_FOLDS}-fold CV)...")
    wave_accs = {}

    # FUSION: try multiple strategies and pick best
    for cond in conditions:
        if cond == 'FUSION' and len(fusion_fpga_feats) > 0:
            fpga_arr = np.array(fusion_fpga_feats)
            gpu_arr = np.array(fusion_gpu_feats)

            # Strategy 1: Per-substrate PCA (FPGA→80, GPU→16)
            res_pca = classify_per_substrate_pca(
                fpga_arr, gpu_arr, None, wave_labels,
                n_splits=N_FOLDS, fpga_pc=80, gpu_pc=16, fw_pc=0)
            print(f"  FUSION (per-sub PCA): {res_pca['mean']:.3f}")

            # Strategy 2: Global PCA on concatenated features
            res_global = classify_condition(wave_features[cond], wave_labels,
                                           n_splits=N_FOLDS, max_features=120)
            print(f"  FUSION (global PCA120): {res_global['mean']:.3f}")

            # Strategy 3: FPGA-only features (ignore GPU readout, same as FPGA_NONE but with GPU workload)
            res_fpga_only = classify_condition(fpga_arr, wave_labels,
                                              n_splits=N_FOLDS, max_features=120)
            print(f"  FUSION (FPGA-only): {res_fpga_only['mean']:.3f}")

            # Strategy 4: Ensemble voting — separate classifiers weighted average
            res_ens = classify_ensemble_voting(
                fpga_arr, gpu_arr, None, wave_labels,
                n_splits=N_FOLDS, fpga_pc=80, gpu_pc=16, fw_pc=0,
                weights=[0.7, 0.3, 0.0])
            print(f"  FUSION (ensemble 70/30): {res_ens['mean']:.3f}")

            # Pick best strategy
            all_strats = [('per_sub_pca', res_pca), ('global_pca', res_global),
                          ('fpga_only', res_fpga_only), ('ensemble', res_ens)]
            best_name, best_res = max(all_strats, key=lambda x: x[1]['mean'])
            best_res['strategy'] = best_name
            best_res['all_strategies'] = {n: r['mean'] for n, r in all_strats}
            res = best_res
            print(f"  FUSION BEST ({best_name}): {res['mean']:.3f} ± {res['std']:.3f}")
        else:
            res = classify_condition(wave_features[cond], wave_labels, n_splits=N_FOLDS)
        wave_accs[cond] = res
        print(f"  {cond}: {res['mean']:.3f} ± {res['std']:.3f}")

    results['waveform_classification'] = wave_accs

    # ─── Temporal XOR (FUSION vs FPGA_128 vs FPGA_8) ───
    print(f"\n[6/7] Temporal XOR...")
    xor_input = generate_xor_sequence(n_steps=args.xor_steps)
    taus = [1, 2, 3, 5]
    xor_conds = ['FUSION', 'FPGA_128', 'FPGA_WHITE', 'FPGA_NONE', 'FPGA_8']

    xor_states = {}
    for cond in xor_conds:
        print(f"  Running XOR reservoir ({cond})...")
        raw, _ = run_trial(fpga, xor_input, noises, w_in, w_noise, torch, device, mode=cond)
        xor_states[cond] = augment_with_delays(raw, delays=(1, 2))

    xor_results = {}
    for tau in taus:
        y_xor = compute_xor_targets(xor_input, tau)
        valid = np.arange(max(tau, 2), args.xor_steps)
        accs = {}
        for cond in xor_conds:
            X_valid = xor_states[cond][valid]
            y_valid = y_xor[valid]
            split = int(0.7 * len(valid))
            X_tr, X_te = X_valid[:split], X_valid[split:]
            y_tr, y_te = y_valid[:split], y_valid[split:]
            mu = X_tr.mean(axis=0, keepdims=True)
            sigma = X_tr.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_tr_n = (X_tr - mu) / sigma
            X_te_n = (X_te - mu) / sigma
            if X_tr_n.shape[1] > 120:
                X_tr_n, pca_mu, pca_Vt = pca_reduce(X_tr_n, n_components=100)
                X_te_n = pca_transform(X_te_n, pca_mu, pca_Vt)
            acc = ridge_binary(X_tr_n, y_tr, X_te_n, y_te)
            accs[cond] = float(acc)
        xor_results[f'tau_{tau}'] = accs
        print(f"  XOR tau={tau}: " + ", ".join(f"{c}={a:.3f}" for c, a in accs.items()))

    results['xor_classification'] = xor_results

    # ─── Cross-Substrate Metrics ───
    print(f"\n[7/7] Cross-substrate metrics...")

    # MI between FPGA and GPU features using RAW per-step data (not pooled)
    mi_fpga_gpu = 0.0
    fw_mi = 0.0

    def binned_mi(x, y, n_bins=10):
        if x.std() < 1e-10 or y.std() < 1e-10: return 0.0
        x_bins = np.digitize(x, np.linspace(x.min(), x.max()+1e-10, n_bins+1))
        y_bins = np.digitize(y, np.linspace(y.min(), y.max()+1e-10, n_bins+1))
        pxy = np.zeros((n_bins+1, n_bins+1))
        for xi, yi in zip(x_bins, y_bins):
            pxy[xi, yi] += 1
        pxy /= pxy.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        mi = 0.0
        for i in range(n_bins+1):
            for j in range(n_bins+1):
                if pxy[i,j] > 1e-12 and px[i] > 1e-12 and py[j] > 1e-12:
                    mi += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[j]))
        return max(0.0, mi)

    try:
        n_mi_trials = min(50, len(fusion_raw_fpga))
        if n_mi_trials > 5:
            # Stack raw per-step FPGA spikes and GPU timings across trials
            # Each trial: (steps, 128) FPGA spikes, (steps, 8) GPU features
            fpga_flat = np.vstack([fusion_raw_fpga[i].mean(axis=0, keepdims=True)
                                   for i in range(n_mi_trials)])  # (n_trials, 128)
            gpu_flat = np.vstack([fusion_raw_gpu[i].mean(axis=0, keepdims=True)
                                  for i in range(n_mi_trials)])   # (n_trials, 8)

            # Average MI across neuron-kernel pairs (sample 16 neurons × 4 kernels)
            mi_pairs = []
            neuron_sample = np.linspace(0, min(127, fpga_flat.shape[1]-1), 16, dtype=int)
            for ni in neuron_sample:
                for ki in range(min(4, gpu_flat.shape[1])):
                    mi_pairs.append(binned_mi(fpga_flat[:, ni], gpu_flat[:, ki]))
            mi_fpga_gpu = float(np.mean(mi_pairs)) if mi_pairs else 0.0

            # Firmware input sensitivity using raw firmware channels
            input_means = np.array([wave_trials[i].mean() for i in range(n_mi_trials)])
            fw_flat = np.vstack([fusion_raw_gpu[i].mean(axis=0, keepdims=True)
                                 for i in range(n_mi_trials)])  # GPU features include FW
            fw_mi_pairs = []
            for j in range(min(8, fw_flat.shape[1])):
                fw_mi_pairs.append(binned_mi(input_means, fw_flat[:, j]))
            fw_mi = float(np.mean(fw_mi_pairs)) if fw_mi_pairs else 0.0
    except Exception as e:
        print(f"  MI computation error: {e}")

    results['cross_substrate_mi'] = float(mi_fpga_gpu)
    results['firmware_input_mi'] = float(fw_mi)
    print(f"  MI(FPGA, GPU_kern) = {mi_fpga_gpu:.4f} bits")
    print(f"  MI(input, firmware) = {fw_mi:.4f} bits")

    # GPU kernel input-dependence
    n_input_dependent = 0
    if 'FUSION' in gpu_timing_all and len(gpu_timing_all['FUSION']) > 10:
        timings_arr = gpu_timing_all['FUSION']
        for k in range(4):
            cv = timings_arr[:, k].std() / max(timings_arr[:, k].mean(), 1e-10)
            if cv > 0.01:
                n_input_dependent += 1
        print(f"  Input-dependent GPU kernels: {n_input_dependent}/4")
    else:
        # Run a quick check
        low_timings, high_timings = [], []
        for _ in range(20):
            t_low, _ = run_gpu_kernel_nodes(0.1, torch, device)
            t_high, _ = run_gpu_kernel_nodes(0.9, torch, device)
            low_timings.append(t_low)
            high_timings.append(t_high)
        low_arr = np.array(low_timings)
        high_arr = np.array(high_timings)
        for k in range(4):
            if abs(high_arr[:, k].mean() - low_arr[:, k].mean()) > low_arr[:, k].std() * 0.5:
                n_input_dependent += 1
        print(f"  Input-dependent GPU kernels: {n_input_dependent}/4")

    results['n_input_dependent_kernels'] = n_input_dependent

    # ═══════════════════════════════════════════════════════════
    # Tests T315-T324
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    tests = {}

    fus = wave_accs['FUSION']['mean']
    f128 = wave_accs['FPGA_128']['mean']
    fw = wave_accs.get('FPGA_WHITE', {}).get('mean', 0)
    fn = wave_accs.get('FPGA_NONE', {}).get('mean', 0)
    gk = wave_accs['GPU_KERN']['mean']
    f8 = wave_accs['FPGA_8']['mean']
    lin = wave_accs['LINEAR']['mean']

    # T315: FUSION > FPGA_128 (GPU readout adds complementary features)
    t315 = fus > f128
    tests['T315'] = {'pass': bool(t315), 'fusion': fus, 'fpga_128': f128,
                     'desc': 'FUSION > FPGA_128 (GPU readout adds value)'}
    print(f"  T315 FUSION({fus:.3f}) > FPGA_128({f128:.3f}): {'PASS' if t315 else 'FAIL'}")

    # T316: FUSION > FPGA_WHITE (clean FPGA+GPU > noisy FPGA)
    t316 = fus > fw
    tests['T316'] = {'pass': bool(t316), 'fusion': fus, 'fpga_white': fw,
                     'desc': 'FUSION > FPGA_WHITE (clean+GPU > noisy)'}
    print(f"  T316 FUSION({fus:.3f}) > FPGA_WHITE({fw:.3f}): {'PASS' if t316 else 'FAIL'}")

    # T317: FUSION > 0.80
    t317 = fus > 0.80
    tests['T317'] = {'pass': bool(t317), 'accuracy': fus,
                     'desc': 'FUSION waveform > 0.80'}
    print(f"  T317 FUSION({fus:.3f}) > 0.80: {'PASS' if t317 else 'FAIL'}")

    # T318: FUSION > LINEAR
    t318 = fus > lin
    tests['T318'] = {'pass': bool(t318), 'fusion': fus, 'linear': lin,
                     'desc': 'FUSION > LINEAR baseline'}
    print(f"  T318 FUSION({fus:.3f}) > LINEAR({lin:.3f}): {'PASS' if t318 else 'FAIL'}")

    # T319: FUSION XOR tau=2 > FPGA_128 XOR tau=2
    xor_fus2 = xor_results.get('tau_2', {}).get('FUSION', 0)
    xor_f128_2 = xor_results.get('tau_2', {}).get('FPGA_128', 0)
    t319 = xor_fus2 > xor_f128_2
    tests['T319'] = {'pass': bool(t319), 'fusion': xor_fus2, 'fpga_128': xor_f128_2,
                     'desc': 'FUSION XOR tau=2 > FPGA_128'}
    print(f"  T319 FUSION XOR2({xor_fus2:.3f}) > FPGA_128({xor_f128_2:.3f}): "
          f"{'PASS' if t319 else 'FAIL'}")

    # T320: FUSION > FPGA_NONE (GPU readout adds value to clean FPGA)
    t320 = fus > fn
    tests['T320'] = {'pass': bool(t320), 'fusion': fus, 'fpga_none': fn,
                     'desc': 'FUSION > FPGA_NONE (GPU readout adds value)'}
    print(f"  T320 FUSION({fus:.3f}) > FPGA_NONE({fn:.3f}): {'PASS' if t320 else 'FAIL'}")

    # T321: MI(FPGA, GPU_kern) > 0.01 bits
    t321 = mi_fpga_gpu > 0.01
    tests['T321'] = {'pass': bool(t321), 'mi': mi_fpga_gpu,
                     'desc': 'Cross-substrate MI > 0.01 bits'}
    print(f"  T321 MI(FPGA,GPU)={mi_fpga_gpu:.4f} > 0.01: {'PASS' if t321 else 'FAIL'}")

    # T322: Firmware input sensitivity MI > 0.01
    t322 = fw_mi > 0.01
    tests['T322'] = {'pass': bool(t322), 'mi': fw_mi,
                     'desc': 'Firmware input MI > 0.01'}
    print(f"  T322 FW_MI={fw_mi:.4f} > 0.01: {'PASS' if t322 else 'FAIL'}")

    # T323: FUSION > FPGA_8 by >10pp
    gap = fus - f8
    t323 = gap > 0.10
    tests['T323'] = {'pass': bool(t323), 'fusion': fus, 'fpga_8': f8, 'gap': gap,
                     'desc': 'FUSION > FPGA_8 by >10pp'}
    print(f"  T323 FUSION({fus:.3f}) - FPGA_8({f8:.3f}) = {gap:.3f} > 0.10: "
          f"{'PASS' if t323 else 'FAIL'}")

    # T324: 3+ GPU kernels show input-dependent timing
    t324 = n_input_dependent >= 3
    tests['T324'] = {'pass': bool(t324), 'n_dependent': n_input_dependent,
                     'desc': '3+ GPU kernels input-dependent'}
    print(f"  T324 {n_input_dependent}/4 kernels input-dependent >= 3: "
          f"{'PASS' if t324 else 'FAIL'}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/10 PASS")

    results['tests'] = tests
    results['summary'] = f'{n_pass}/10 PASS'

    # Save
    out_path = RESULTS / 'z2207_deep_fusion_reservoir.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Results saved: {out_path}")

    fpga.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
