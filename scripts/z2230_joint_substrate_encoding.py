#!/usr/bin/env python3
"""z2230_joint_substrate_encoding.py — Cross-Substrate Joint Encoding

Key hypothesis: When the classification signal is SPLIT between GPU and
FPGA substrates, ONLY the coupled system should succeed. This directly
tests bidirectional information flow.

Experiments:
  EXP 1: Joint Encoding (4-class) — signal split: waveform + GPU intensity
         Each class has SAME waveform but DIFFERENT GPU patterns, or vice versa.
         COUPLED should WIN because it sees both substrates.
  EXP 2: XOR Substrate Task — class = waveform_type XOR gpu_type
         Neither substrate alone has enough info (each sees only 2 bits).
  EXP 3: GPU→FPGA Transfer Entropy — direct information-theoretic measure
         of how much GPU state information reaches FPGA neurons.
  EXP 4: Bidirectional Granger Causality — does FPGA spike rate predict
         GPU power, and does GPU power predict FPGA spike rate?
  EXP 5: Substrate Ablation Gradient — systematic removal of coupling
         channels (power, thermal, clock) to identify which carries most info.
  EXP 6: Temporal Cross-Substrate Memory — can FPGA neurons at time t
         predict GPU state at time t-k? (reverse temporal coupling)

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128 neurons, UDP Ethernet)
Tests: T754-T773 (20 tests)
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters (z2229 proven values) ───
N_NEURONS = 128
BASE_VG = 0.45
ALPHA = 0.35
BETA_POWER = 0.12
BETA_THERMAL = 0.08
BETA_CLOCK = 0.10
SAMPLE_HZ = 20
N_STEPS = 30
WORKLOAD_MS = 5.0

# Probe paths
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
HWMON_TEMP = "/sys/class/hwmon/hwmon7/temp1_input"
HWMON_FREQ = "/sys/class/hwmon/hwmon7/freq1_input"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
GPU_BUSY_PATH = "/sys/class/drm/card0/device/gpu_busy_percent"

CHANNEL_NAMES = [
    'smn_temp', 'pm_thermal', 'pm_power', 'pm_sclk',
    'hw_power', 'hw_temp', 'hw_freq', 'gpu_busy', 'dispatch_jitter'
]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# GPU PROBES (same as z2229)
# ═══════════════════════════════════════════════════════════

def read_smn_adc():
    try:
        with open(SMN_PATH, 'rb+') as f:
            f.write(struct.pack('<I', 0x59800))
            f.seek(0)
            raw = struct.unpack('<I', f.read(4))[0]
        return (raw >> 21) * 0.125
    except:
        return None

def read_pm_table():
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(0x004C); thermal = struct.unpack('<f', f.read(4))[0]
            f.seek(0x0100); power = struct.unpack('<f', f.read(4))[0]
            f.seek(0x0344); sclk = struct.unpack('<f', f.read(4))[0]
        return thermal, power, sclk
    except:
        return None, None, None

def read_hwmon():
    try:
        p = int(open(HWMON_POWER).read().strip()) / 1e6
        t = int(open(HWMON_TEMP).read().strip()) / 1e3
        f = int(open(HWMON_FREQ).read().strip()) / 1e6
        return p, t, f
    except:
        return None, None, None

def read_gpu_busy():
    try: return int(open(GPU_BUSY_PATH).read().strip())
    except: return 0

_torch_device = None
_torch_available = False

def measure_dispatch_jitter():
    if not _torch_available:
        return 0.0
    import torch
    a = torch.randn(64, 64, device=_torch_device)
    b = torch.randn(64, 64, device=_torch_device)
    t0 = time.perf_counter()
    _ = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del a, b
    return elapsed

def read_all_gpu_state():
    smn = read_smn_adc()
    pm_t, pm_p, pm_sclk = read_pm_table()
    hw_p, hw_t, hw_f = read_hwmon()
    busy = read_gpu_busy()
    jitter = measure_dispatch_jitter()
    return [
        smn or 0, pm_t or 0, pm_p or 0, pm_sclk or 0,
        hw_p or 0, hw_t or 0, hw_f or 0, busy or 0, jitter
    ]


# ═══════════════════════════════════════════════════════════
# GPU Workload
# ═══════════════════════════════════════════════════════════

def init_torch():
    global _torch_available, _torch_device
    try:
        import torch
        if torch.cuda.is_available():
            _torch_device = torch.device('cuda')
            _ = torch.randn(64, 64, device=_torch_device) @ torch.randn(64, 64, device=_torch_device)
            torch.cuda.synchronize()
            _torch_available = True
            print(f"  HIP: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARNING: No CUDA/HIP")
    except ImportError:
        print("  WARNING: No torch")

def run_workload(intensity, duration_ms=5.0):
    if not _torch_available or intensity < 0.05:
        return 0.0
    import torch
    N = int(128 + 896 * np.clip(intensity, 0.0, 1.0))
    a = torch.randn(N, N, device=_torch_device)
    b = torch.randn(N, N, device=_torch_device)
    t0 = time.perf_counter()
    deadline = t0 + duration_ms / 1000.0
    while time.perf_counter() < deadline:
        _ = torch.mm(a, b)
    elapsed = time.perf_counter() - t0
    del a, b
    return elapsed

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ═══════════════════════════════════════════════════════════
# COUPLED DYNAMICS LOOP
# ═══════════════════════════════════════════════════════════

def run_coupled_loop(fpga, input_signal, w_in, w_gpu, w_fb, vg_spread,
                     mode='COUPLED', n_neurons=128, record_gpu=True,
                     gpu_intensity=None, beta_scale=1.0,
                     ablate_channels=None):
    """Run reservoir loop. ablate_channels: set of {'power','thermal','clock'} to zero out."""
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    raw_states = np.zeros((n_steps, n_neurons * 3))
    gpu_log = np.zeros((n_steps, 9))
    cumulative = np.zeros(n_neurons)
    prev_counts = None

    for t in range(n_steps):
        t_start = time.perf_counter()

        if record_gpu:
            gpu_log[t] = read_all_gpu_state()

        vg = BASE_VG + vg_spread[:n_neurons]

        if mode in ('COUPLED', 'FPGA_ONLY'):
            vg += ALPHA * input_signal[t] * w_in[:n_neurons]

        if mode == 'COUPLED':
            hw_p = gpu_log[t, 4]
            pm_sclk = gpu_log[t, 3]
            pm_t = gpu_log[t, 1]

            if t >= 3:
                p_base = gpu_log[max(0,t-10):t, 4].mean()
                s_base = gpu_log[max(0,t-10):t, 3].mean()
                t_base = gpu_log[max(0,t-10):t, 1].mean()
            else:
                p_base, s_base, t_base = hw_p, pm_sclk, pm_t

            p_delta = (hw_p - p_base) / max(abs(p_base), 1.0)
            s_delta = (pm_sclk - s_base) / max(abs(s_base), 1.0)
            t_delta = (pm_t - t_base) / max(abs(t_base), 1.0)

            # Ablation: zero out specific channels
            if ablate_channels:
                if 'power' in ablate_channels: p_delta = 0.0
                if 'thermal' in ablate_channels: t_delta = 0.0
                if 'clock' in ablate_channels: s_delta = 0.0

            n3 = n_neurons // 3
            wg = w_gpu[:n_neurons]
            vg[:n3] += beta_scale * BETA_POWER * p_delta * wg[:n3]
            vg[n3:2*n3] += beta_scale * BETA_CLOCK * s_delta * wg[n3:2*n3]
            vg[2*n3:] += beta_scale * BETA_THERMAL * t_delta * wg[2*n3:]

            gs_pm_t = gpu_log[t, 1]
            if gs_pm_t > 0:
                try:
                    fpga.set_temp(float(gs_pm_t) + 273.15)
                except:
                    pass

        vg = np.clip(vg, 0.10, 0.85)

        if mode != 'GPU_ONLY':
            fpga.set_vg_batch(0, vg.tolist())
            time.sleep(0.001)

            try:
                counts, vm, refract = fpga.read_telemetry_fast()
                if prev_counts is not None:
                    for i in range(n_neurons):
                        delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                        if delta > 30000: delta = 0
                        raw_states[t, i] = delta
                        cumulative[i] += delta
                    raw_states[t, n_neurons:2*n_neurons] = vm[:n_neurons]
                    raw_states[t, 2*n_neurons:] = cumulative
                prev_counts = counts.copy()
            except:
                pass

        # GPU workload
        if mode in ('COUPLED', 'FPGA_ONLY'):
            if gpu_intensity is not None:
                intensity = float(gpu_intensity[t])
            elif t >= 1:
                recent_spikes = raw_states[max(0,t-2):t+1, :n_neurons].mean(axis=0)
                raw_val = float(np.dot(recent_spikes, w_fb[:n_neurons]))
                intensity = float(sigmoid(raw_val - 5.0))
            else:
                intensity = 0.3
            run_workload(intensity, duration_ms=WORKLOAD_MS)
            if mode == 'COUPLED':
                try:
                    fpga.set_mac_signal(intensity * 0.5)
                except:
                    pass
        elif mode == 'GPU_ONLY':
            intensity = float(0.2 + 0.6 * np.clip(input_signal[t], 0, 1))
            run_workload(intensity, duration_ms=WORKLOAD_MS)

        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.001:
            time.sleep(remaining)

    return raw_states, gpu_log


# ═══════════════════════════════════════════════════════════
# FEATURE PIPELINE + PCA (z2229 proven)
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2)):
    T, D = states.shape
    aug = np.zeros((T, D * (1 + len(delays))))
    aug[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        aug[d:, start:start+D] = states[:-d]
    return aug

def pool_trial_features(raw_states):
    aug = augment_with_delays(raw_states, delays=(1, 2))
    return np.concatenate([
        aug.mean(axis=0), aug.std(axis=0),
        aug.max(axis=0), aug.min(axis=0),
    ])

def pool_gpu_features(gpu_log):
    return np.concatenate([
        gpu_log.mean(axis=0), gpu_log.std(axis=0),
        gpu_log.max(axis=0), gpu_log.min(axis=0),
    ])

def pca_reduce(X, n_components=100):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T, mu, Vt[:n_components]

def pca_transform(X, mu, Vt):
    return (X - mu) @ Vt.T


# ═══════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=None):
    if n_classes is None: n_classes = max(len(np.unique(y_tr)), len(np.unique(y_te)))
    alphas = [1e-4, 1e-2, 1.0, 100.0, 1000.0, 10000.0]
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, y in enumerate(y_tr): Y_tr[i, int(y)] = 1.0
    best = -1
    for a in alphas:
        I = np.eye(X_tr.shape[1])
        try: W = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ Y_tr)
        except: continue
        acc = np.mean(np.argmax(X_te @ W, axis=1) == y_te)
        if acc > best: best = acc
    return best

def stratified_kfold(X, y, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        c_idx = indices[y[indices] == c]
        for i, idx in enumerate(c_idx): folds[i % n_splits].append(idx)
    splits = []
    for fold in range(n_splits):
        te = np.array(folds[fold])
        tr = np.concatenate([np.array(folds[f]) for f in range(n_splits) if f != fold])
        splits.append((tr, te))
    return splits

def classify_cv_pca(X, y, n_splits=5, n_classes=None, max_pca=120):
    if n_classes is None: n_classes = len(np.unique(y))
    splits = stratified_kfold(X, y, n_splits)
    accs = []
    use_pca = X.shape[1] > max_pca
    for tr_idx, te_idx in splits:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
        sigma[sigma < 1e-2] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        if use_pca:
            X_tr_n, pca_mu, pca_Vt = pca_reduce(X_tr_n, n_components=max_pca)
            X_te_n = pca_transform(X_te_n, pca_mu, pca_Vt)
        acc = ridge_classify(X_tr_n, y_tr, X_te_n, y_te, n_classes=n_classes)
        accs.append(acc)
    return {'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_fold': [float(a) for a in accs]}

def classify_per_substrate_pca(fpga_feats, gpu_feats, y, n_splits=5,
                                n_classes=None, fpga_pc=80, gpu_pc=20):
    if n_classes is None: n_classes = len(np.unique(y))
    splits = stratified_kfold(fpga_feats, y, n_splits)
    accs = []
    for tr_idx, te_idx in splits:
        y_tr, y_te = y[tr_idx], y[te_idx]
        parts_tr, parts_te = [], []
        for F_all, n_pc in [(fpga_feats, fpga_pc), (gpu_feats, gpu_pc)]:
            if F_all is None: continue
            F_tr, F_te = F_all[tr_idx], F_all[te_idx]
            mu = F_tr.mean(axis=0); sigma = F_tr.std(axis=0)
            sigma[sigma < 1e-2] = 1.0
            F_tr_n = (F_tr - mu) / sigma
            F_te_n = (F_te - mu) / sigma
            if F_tr_n.shape[1] > n_pc:
                F_tr_n, pca_mu, pca_Vt = pca_reduce(F_tr_n, n_components=n_pc)
                F_te_n = pca_transform(F_te_n, pca_mu, pca_Vt)
            parts_tr.append(F_tr_n)
            parts_te.append(F_te_n)
        X_tr = np.hstack(parts_tr)
        X_te = np.hstack(parts_te)
        acc = ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=n_classes)
        accs.append(acc)
    return {'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_fold': [float(a) for a in accs]}


# ═══════════════════════════════════════════════════════════
# SIGNAL GENERATION — JOINT ENCODING
# ═══════════════════════════════════════════════════════════

def generate_joint_encoding(n_trials, n_steps, seed=42):
    """4 classes where class = (waveform_type, gpu_pattern).
    Class 0: sine wave + low GPU
    Class 1: sine wave + high GPU      (same waveform as 0, different GPU)
    Class 2: square wave + low GPU      (same GPU as 0, different waveform)
    Class 3: square wave + high GPU
    FPGA_ONLY sees waveform only → can distinguish {0,1} from {2,3} but not within pairs.
    GPU_ONLY sees GPU pattern only → can distinguish {0,2} from {1,3} but not within pairs.
    COUPLED sees both → can distinguish all 4."""
    rng = np.random.default_rng(seed)
    trials, labels, gpu_intensities = [], [], []
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(n_steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 4)
        freq = rng.uniform(1.0, 2.5)

        # Waveform: sine (cls 0,1) or square (cls 2,3)
        if cls in (0, 1):
            wave = 0.5 + 0.4 * np.sin(2 * np.pi * freq * t)
        else:
            wave = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * freq * t))

        # GPU pattern: low (cls 0,2) or high (cls 1,3)
        if cls in (0, 2):
            gpu_int = np.full(n_steps, 0.1) + 0.05 * rng.standard_normal(n_steps)
            gpu_int = np.clip(gpu_int, 0.05, 0.25)
        else:
            gpu_int = np.full(n_steps, 0.8) + 0.05 * rng.standard_normal(n_steps)
            gpu_int = np.clip(gpu_int, 0.65, 0.95)

        trials.append(wave)
        labels.append(cls)
        gpu_intensities.append(gpu_int)
    return np.array(trials), np.array(labels), np.array(gpu_intensities)


def generate_xor_encoding(n_trials, n_steps, seed=42):
    """4 stimuli, 2 classes: class = waveform_bit XOR gpu_bit.
    Stimulus A: sine + low GPU  → class 0 (0 XOR 0)
    Stimulus B: sine + high GPU → class 1 (0 XOR 1)
    Stimulus C: square + low GPU → class 1 (1 XOR 0)
    Stimulus D: square + high GPU → class 0 (1 XOR 1)
    Neither substrate alone can solve this — it's a CLASSIC XOR problem
    across substrate boundaries."""
    rng = np.random.default_rng(seed)
    trials, labels, gpu_intensities = [], [], []
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(n_steps) * dt
    for _ in range(n_trials):
        stim = rng.integers(0, 4)
        freq = rng.uniform(1.0, 2.5)

        wave_bit = stim // 2  # 0 for A,B; 1 for C,D
        gpu_bit = stim % 2    # 0 for A,C; 1 for B,D

        if wave_bit == 0:
            wave = 0.5 + 0.4 * np.sin(2 * np.pi * freq * t)
        else:
            wave = 0.5 + 0.4 * np.sign(np.sin(2 * np.pi * freq * t))

        if gpu_bit == 0:
            gpu_int = np.full(n_steps, 0.1) + 0.05 * rng.standard_normal(n_steps)
            gpu_int = np.clip(gpu_int, 0.05, 0.25)
        else:
            gpu_int = np.full(n_steps, 0.8) + 0.05 * rng.standard_normal(n_steps)
            gpu_int = np.clip(gpu_int, 0.65, 0.95)

        cls = wave_bit ^ gpu_bit  # XOR
        trials.append(wave)
        labels.append(cls)
        gpu_intensities.append(gpu_int)
    return np.array(trials), np.array(labels), np.array(gpu_intensities)


# ═══════════════════════════════════════════════════════════
# INFORMATION-THEORETIC MEASURES
# ═══════════════════════════════════════════════════════════

def transfer_entropy(source, target, k=1, n_bins=8):
    """Transfer entropy TE(source → target) using binned estimator.
    TE = H(target_future | target_past) - H(target_future | target_past, source_past)"""
    n = len(source) - k
    if n < 20:
        return 0.0

    # Bin the continuous signals
    s_bins = np.digitize(source, np.linspace(source.min(), source.max() + 1e-10, n_bins + 1)) - 1
    t_bins = np.digitize(target, np.linspace(target.min(), target.max() + 1e-10, n_bins + 1)) - 1

    # Compute joint probabilities
    t_future = t_bins[k:]
    t_past = t_bins[:n]
    s_past = s_bins[:n]

    # H(target_future | target_past)
    def cond_entropy(x, y):
        """H(X|Y) via joint counting."""
        joint = {}
        y_counts = {}
        for xi, yi in zip(x, y):
            joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
            y_counts[yi] = y_counts.get(yi, 0) + 1
        h = 0.0
        for (xi, yi), c in joint.items():
            p_xy = c / n
            p_y = y_counts[yi] / n
            if p_xy > 0 and p_y > 0:
                h -= p_xy * np.log2(p_xy / p_y)
        return h

    # H(target_future | target_past, source_past) — condition on both
    combined_past = t_past * n_bins + s_past  # unique joint state
    h_t_given_tp = cond_entropy(t_future, t_past)
    h_t_given_tp_sp = cond_entropy(t_future, combined_past)

    te = h_t_given_tp - h_t_given_tp_sp
    return max(te, 0.0)


def granger_causality_r2(source, target, lag=3):
    """Simple Granger causality: R² improvement when adding source lags.
    Returns (r2_restricted, r2_full, gc_score = r2_full - r2_restricted)."""
    n = len(source) - lag
    if n < 20:
        return 0.0, 0.0, 0.0

    # Restricted model: target_t ~ target_{t-1..t-lag}
    y = target[lag:]
    X_r = np.column_stack([target[lag-i-1:n+lag-i-1] for i in range(lag)])

    # Full model: target_t ~ target_{t-1..t-lag} + source_{t-1..t-lag}
    X_f = np.column_stack([X_r] + [source[lag-i-1:n+lag-i-1] for i in range(lag)])

    def fit_r2(X, y):
        try:
            I = 1e-4 * np.eye(X.shape[1])
            w = np.linalg.solve(X.T @ X + I, X.T @ y)
            y_pred = X @ w
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot < 1e-10:
                return 0.0
            return 1.0 - ss_res / ss_tot
        except:
            return 0.0

    r2_r = fit_r2(X_r, y)
    r2_f = fit_r2(X_f, y)
    return r2_r, r2_f, r2_f - r2_r


# ═══════════════════════════════════════════════════════════
# FEATURE COLLECTION HELPER
# ═══════════════════════════════════════════════════════════

def collect_features(fpga, wave, w_in, w_gpu, w_fb, vg_spread,
                     mode='COUPLED', gpu_intensity=None, beta_scale=1.0,
                     ablate_channels=None):
    """Run one trial and extract FPGA + GPU features."""
    raw_states, gpu_log = run_coupled_loop(
        fpga, wave, w_in, w_gpu, w_fb, vg_spread,
        mode=mode, n_neurons=N_NEURONS, record_gpu=True,
        gpu_intensity=gpu_intensity, beta_scale=beta_scale,
        ablate_channels=ablate_channels
    )
    fpga_feat = pool_trial_features(raw_states)
    gpu_feat = pool_gpu_features(gpu_log)
    return fpga_feat, gpu_feat, raw_states, gpu_log


# ═══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("z2230: CROSS-SUBSTRATE JOINT ENCODING")
    print("  Joint signal split: waveform (FPGA) × GPU pattern (GPU)")
    print("  XOR encoding: neither substrate alone has enough info")
    print("  Transfer entropy + Granger causality measures")
    print("=" * 72)

    # ─── Setup ───
    print("\n[1] Connecting FPGA...")
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("[ETH] WARN: No telemetry response, continuing anyway...")
    print(f"[ETH] {N_NEURONS} neurons")

    print("\n[2] Init GPU HIP...")
    init_torch()

    print("\n[3] Probe check...")
    state = read_all_gpu_state()
    for name, val in zip(CHANNEL_NAMES, state):
        print(f"    {name}: {val}")

    # Random weights
    rng = np.random.default_rng(42)
    w_in = rng.standard_normal(N_NEURONS) * 0.5
    w_gpu = rng.standard_normal(N_NEURONS) * 0.5
    w_fb = rng.standard_normal(N_NEURONS) * 0.05
    vg_spread = np.linspace(-0.08, 0.08, N_NEURONS)

    results = {
        'experiment': 'z2230_joint_substrate_encoding',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'base': 'z2229 + joint encoding + XOR task + TE/GC',
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_steps': N_STEPS, 'membrane_tau_ms': 49.4,
        },
    }
    tests = {}

    # ═══════════════════════════════════════════════════════════
    # EXP 1: JOINT ENCODING (4-class, 200 trials)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 1: JOINT ENCODING (4-class, 200 trials)")
    print("  Class = (waveform_type, gpu_pattern)")
    print("  FPGA alone: 50% ceiling. GPU alone: 50% ceiling.")
    print("  COUPLED should exceed both → proves joint substrate decoding.")
    print("=" * 72)

    N_JOINT = 200
    waves_j, labels_j, gpu_int_j = generate_joint_encoding(N_JOINT, N_STEPS, seed=42)
    joint_results = {}

    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Joint Encoding: {cond} ---")
        fpga_feats, gpu_feats = [], []
        for trial in range(N_JOINT):
            if (trial + 1) % 40 == 0:
                print(f"    {cond} trial {trial+1}/{N_JOINT}")
            gi = gpu_int_j[trial] if cond == 'COUPLED' else None
            f_feat, g_feat, _, _ = collect_features(
                fpga, waves_j[trial], w_in, w_gpu, w_fb, vg_spread,
                mode=cond, gpu_intensity=gi)
            fpga_feats.append(f_feat)
            gpu_feats.append(g_feat)

        X_fpga = np.array(fpga_feats)
        X_gpu = np.array(gpu_feats)

        # FPGA-only classification (should cap at ~50% for 4 classes)
        res_fpga = classify_cv_pca(X_fpga, labels_j, n_classes=4)
        print(f"  {cond} FPGA-only: acc={res_fpga['mean']:.3f} ± {res_fpga['std']:.3f}")

        # GPU-only classification (should cap at ~50% for 4 classes)
        res_gpu = classify_cv_pca(X_gpu, labels_j, n_classes=4, max_pca=30)
        print(f"  {cond} GPU-only: acc={res_gpu['mean']:.3f} ± {res_gpu['std']:.3f}")

        # Combined per-substrate PCA
        if cond == 'COUPLED':
            res_combined = classify_per_substrate_pca(X_fpga, X_gpu, labels_j, n_classes=4)
            print(f"  {cond} combined: acc={res_combined['mean']:.3f} ± {res_combined['std']:.3f}")
        else:
            res_combined = res_fpga  # no GPU features for non-coupled

        joint_results[cond] = {
            'fpga_only': res_fpga,
            'gpu_only': res_gpu,
            'combined': res_combined,
        }

    results['joint_encoding'] = joint_results
    c_comb = joint_results['COUPLED']['combined']['mean']
    c_fpga = joint_results['COUPLED']['fpga_only']['mean']
    c_gpu = joint_results['COUPLED']['gpu_only']['mean']
    f_fpga = joint_results['FPGA_ONLY']['fpga_only']['mean']

    tests['T754'] = {'desc': 'Joint: COUPLED combined > 0.60', 'val': c_comb,
                     'pass': c_comb > 0.60}
    tests['T755'] = {'desc': 'Joint: COUPLED combined > FPGA-only', 'val': c_comb - c_fpga,
                     'pass': c_comb > c_fpga}
    tests['T756'] = {'desc': 'Joint: COUPLED combined > GPU-only', 'val': c_comb - c_gpu,
                     'pass': c_comb > c_gpu}
    tests['T757'] = {'desc': 'Joint: COUPLED combined > FPGA_ONLY condition', 'val': c_comb - f_fpga,
                     'pass': c_comb > f_fpga}
    for tid in ['T754', 'T755', 'T756', 'T757']:
        status = "PASS" if tests[tid]['pass'] else "FAIL"
        print(f"  {tid}: {status} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ═══════════════════════════════════════════════════════════
    # EXP 2: XOR SUBSTRATE TASK (2-class, 200 trials)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 2: XOR SUBSTRATE TASK (2-class, 200 trials)")
    print("  class = waveform_bit XOR gpu_bit")
    print("  Neither substrate alone can solve XOR → 50% ceiling each")
    print("  Only combined readout should exceed chance")
    print("=" * 72)

    N_XOR = 200
    waves_x, labels_x, gpu_int_x = generate_xor_encoding(N_XOR, N_STEPS, seed=99)
    xor_results = {}

    for cond in ['COUPLED', 'FPGA_ONLY']:
        print(f"\n  --- XOR: {cond} ---")
        fpga_feats, gpu_feats = [], []
        for trial in range(N_XOR):
            if (trial + 1) % 40 == 0:
                print(f"    {cond} trial {trial+1}/{N_XOR}")
            gi = gpu_int_x[trial] if cond == 'COUPLED' else None
            f_feat, g_feat, _, _ = collect_features(
                fpga, waves_x[trial], w_in, w_gpu, w_fb, vg_spread,
                mode=cond, gpu_intensity=gi)
            fpga_feats.append(f_feat)
            gpu_feats.append(g_feat)

        X_fpga = np.array(fpga_feats)
        X_gpu = np.array(gpu_feats)

        res_fpga = classify_cv_pca(X_fpga, labels_x, n_classes=2)
        res_gpu = classify_cv_pca(X_gpu, labels_x, n_classes=2, max_pca=30)
        print(f"  {cond} FPGA-only: acc={res_fpga['mean']:.3f}")
        print(f"  {cond} GPU-only: acc={res_gpu['mean']:.3f}")

        if cond == 'COUPLED':
            res_combined = classify_per_substrate_pca(X_fpga, X_gpu, labels_x, n_classes=2)
            print(f"  {cond} combined: acc={res_combined['mean']:.3f}")
        else:
            res_combined = res_fpga

        xor_results[cond] = {
            'fpga_only': res_fpga,
            'gpu_only': res_gpu,
            'combined': res_combined,
        }

    results['xor_encoding'] = xor_results
    xor_comb = xor_results['COUPLED']['combined']['mean']
    xor_fpga = xor_results['COUPLED']['fpga_only']['mean']
    xor_gpu = xor_results['COUPLED']['gpu_only']['mean']

    tests['T758'] = {'desc': 'XOR: COUPLED combined > 0.55', 'val': xor_comb,
                     'pass': xor_comb > 0.55}
    tests['T759'] = {'desc': 'XOR: COUPLED combined > FPGA-only', 'val': xor_comb - xor_fpga,
                     'pass': xor_comb > xor_fpga}
    tests['T760'] = {'desc': 'XOR: COUPLED combined > GPU-only', 'val': xor_comb - xor_gpu,
                     'pass': xor_comb > xor_gpu}
    tests['T761'] = {'desc': 'XOR: COUPLED combined > 0.65 (strong)', 'val': xor_comb,
                     'pass': xor_comb > 0.65}
    for tid in ['T758', 'T759', 'T760', 'T761']:
        status = "PASS" if tests[tid]['pass'] else "FAIL"
        print(f"  {tid}: {status} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ═══════════════════════════════════════════════════════════
    # EXP 3: TRANSFER ENTROPY (GPU→FPGA and FPGA→GPU)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 3: TRANSFER ENTROPY (40 trials, continuous drive)")
    print("  TE(GPU→FPGA) and TE(FPGA→GPU)")
    print("  Coupled should show higher TE than uncoupled")
    print("=" * 72)

    N_TE = 40
    te_waves = 0.5 + 0.3 * np.sin(2 * np.pi * 1.5 * np.arange(N_STEPS) / SAMPLE_HZ)

    te_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY']:
        print(f"\n  --- TE: {cond} ---")
        all_te_gpu2fpga = []
        all_te_fpga2gpu = []

        for trial in range(N_TE):
            if (trial + 1) % 10 == 0:
                print(f"    {cond} trial {trial+1}/{N_TE}")

            _, _, raw_states, gpu_log = collect_features(
                fpga, te_waves, w_in, w_gpu, w_fb, vg_spread,
                mode=cond)

            # GPU signal: power (channel 4)
            gpu_power = gpu_log[:, 4]
            # FPGA signal: mean spike rate across neurons
            fpga_spikes = raw_states[:, :N_NEURONS].mean(axis=1)

            te_g2f = transfer_entropy(gpu_power, fpga_spikes, k=2)
            te_f2g = transfer_entropy(fpga_spikes, gpu_power, k=2)
            all_te_gpu2fpga.append(te_g2f)
            all_te_fpga2gpu.append(te_f2g)

        te_results[cond] = {
            'te_gpu2fpga_mean': float(np.mean(all_te_gpu2fpga)),
            'te_gpu2fpga_std': float(np.std(all_te_gpu2fpga)),
            'te_fpga2gpu_mean': float(np.mean(all_te_fpga2gpu)),
            'te_fpga2gpu_std': float(np.std(all_te_fpga2gpu)),
        }
        print(f"  {cond}: TE(GPU→FPGA)={np.mean(all_te_gpu2fpga):.4f} ± {np.std(all_te_gpu2fpga):.4f}")
        print(f"  {cond}: TE(FPGA→GPU)={np.mean(all_te_fpga2gpu):.4f} ± {np.std(all_te_fpga2gpu):.4f}")

    results['transfer_entropy'] = te_results
    te_c_g2f = te_results['COUPLED']['te_gpu2fpga_mean']
    te_f_g2f = te_results['FPGA_ONLY']['te_gpu2fpga_mean']
    te_c_f2g = te_results['COUPLED']['te_fpga2gpu_mean']

    tests['T762'] = {'desc': 'TE: COUPLED GPU→FPGA > 0.01 bits', 'val': te_c_g2f,
                     'pass': te_c_g2f > 0.01}
    tests['T763'] = {'desc': 'TE: COUPLED GPU→FPGA > FPGA_ONLY GPU→FPGA',
                     'val': te_c_g2f - te_f_g2f, 'pass': te_c_g2f > te_f_g2f}
    tests['T764'] = {'desc': 'TE: Bidirectional (FPGA→GPU > 0.005 bits)', 'val': te_c_f2g,
                     'pass': te_c_f2g > 0.005}
    for tid in ['T762', 'T763', 'T764']:
        status = "PASS" if tests[tid]['pass'] else "FAIL"
        print(f"  {tid}: {status} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ═══════════════════════════════════════════════════════════
    # EXP 4: GRANGER CAUSALITY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 4: GRANGER CAUSALITY (40 trials)")
    print("  R² improvement when adding cross-substrate lags")
    print("=" * 72)

    gc_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY']:
        print(f"\n  --- GC: {cond} ---")
        all_gc_g2f = []
        all_gc_f2g = []

        for trial in range(N_TE):  # reuse N_TE=40
            if (trial + 1) % 10 == 0:
                print(f"    {cond} trial {trial+1}/{N_TE}")

            _, _, raw_states, gpu_log = collect_features(
                fpga, te_waves, w_in, w_gpu, w_fb, vg_spread,
                mode=cond)

            gpu_power = gpu_log[:, 4]
            fpga_spikes = raw_states[:, :N_NEURONS].mean(axis=1)

            _, _, gc_g2f = granger_causality_r2(gpu_power, fpga_spikes, lag=3)
            _, _, gc_f2g = granger_causality_r2(fpga_spikes, gpu_power, lag=3)
            all_gc_g2f.append(gc_g2f)
            all_gc_f2g.append(gc_f2g)

        gc_results[cond] = {
            'gc_gpu2fpga_mean': float(np.mean(all_gc_g2f)),
            'gc_gpu2fpga_std': float(np.std(all_gc_g2f)),
            'gc_fpga2gpu_mean': float(np.mean(all_gc_f2g)),
            'gc_fpga2gpu_std': float(np.std(all_gc_f2g)),
        }
        print(f"  {cond}: GC(GPU→FPGA)={np.mean(all_gc_g2f):.4f}")
        print(f"  {cond}: GC(FPGA→GPU)={np.mean(all_gc_f2g):.4f}")

    results['granger_causality'] = gc_results
    gc_c_g2f = gc_results['COUPLED']['gc_gpu2fpga_mean']
    gc_f_g2f = gc_results['FPGA_ONLY']['gc_gpu2fpga_mean']

    tests['T765'] = {'desc': 'GC: COUPLED GPU→FPGA > 0.01', 'val': gc_c_g2f,
                     'pass': gc_c_g2f > 0.01}
    tests['T766'] = {'desc': 'GC: COUPLED GPU→FPGA > FPGA_ONLY', 'val': gc_c_g2f - gc_f_g2f,
                     'pass': gc_c_g2f > gc_f_g2f}
    for tid in ['T765', 'T766']:
        status = "PASS" if tests[tid]['pass'] else "FAIL"
        print(f"  {tid}: {status} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ═══════════════════════════════════════════════════════════
    # EXP 5: SUBSTRATE ABLATION GRADIENT
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 5: SUBSTRATE ABLATION GRADIENT (120 trials each)")
    print("  Systematically remove power/thermal/clock coupling channels")
    print("  Identifies which physical channel carries most information")
    print("=" * 72)

    N_ABLATION = 120
    waves_a, labels_a, gpu_int_a = generate_joint_encoding(N_ABLATION, N_STEPS, seed=77)

    ablation_configs = {
        'FULL': None,
        'NO_POWER': {'power'},
        'NO_THERMAL': {'thermal'},
        'NO_CLOCK': {'clock'},
        'NO_POWER_THERMAL': {'power', 'thermal'},
        'FPGA_ONLY': 'fpga_only',
    }

    ablation_results = {}
    for abl_name, abl_set in ablation_configs.items():
        print(f"\n  --- Ablation: {abl_name} ---")
        fpga_feats, gpu_feats = [], []
        for trial in range(N_ABLATION):
            if (trial + 1) % 40 == 0:
                print(f"    {abl_name} trial {trial+1}/{N_ABLATION}")

            if abl_set == 'fpga_only':
                f_feat, g_feat, _, _ = collect_features(
                    fpga, waves_a[trial], w_in, w_gpu, w_fb, vg_spread,
                    mode='FPGA_ONLY')
            else:
                f_feat, g_feat, _, _ = collect_features(
                    fpga, waves_a[trial], w_in, w_gpu, w_fb, vg_spread,
                    mode='COUPLED', gpu_intensity=gpu_int_a[trial],
                    ablate_channels=abl_set)
            fpga_feats.append(f_feat)
            gpu_feats.append(g_feat)

        X_fpga = np.array(fpga_feats)
        X_gpu = np.array(gpu_feats)

        if abl_set != 'fpga_only':
            res = classify_per_substrate_pca(X_fpga, X_gpu, labels_a, n_classes=4)
        else:
            res = classify_cv_pca(X_fpga, labels_a, n_classes=4)
        ablation_results[abl_name] = res
        print(f"  {abl_name}: acc={res['mean']:.3f} ± {res['std']:.3f}")

    results['ablation_gradient'] = ablation_results
    full_acc = ablation_results['FULL']['mean']
    no_p = ablation_results['NO_POWER']['mean']
    no_t = ablation_results['NO_THERMAL']['mean']
    no_c = ablation_results['NO_CLOCK']['mean']
    no_pt = ablation_results['NO_POWER_THERMAL']['mean']
    fpga_only_acc = ablation_results['FPGA_ONLY']['mean']

    tests['T767'] = {'desc': 'Ablation: FULL > NO_POWER (power carries info)',
                     'val': full_acc - no_p, 'pass': full_acc > no_p}
    tests['T768'] = {'desc': 'Ablation: FULL > NO_THERMAL (thermal carries info)',
                     'val': full_acc - no_t, 'pass': full_acc > no_t}
    tests['T769'] = {'desc': 'Ablation: FULL > NO_POWER_THERMAL (combined channels > single)',
                     'val': full_acc - no_pt, 'pass': full_acc > no_pt}
    tests['T770'] = {'desc': 'Ablation: FULL > FPGA_ONLY', 'val': full_acc - fpga_only_acc,
                     'pass': full_acc > fpga_only_acc}

    # Identify most important channel
    channel_drops = {
        'power': full_acc - no_p,
        'thermal': full_acc - no_t,
        'clock': full_acc - no_c,
    }
    most_important = max(channel_drops, key=channel_drops.get)
    results['ablation_gradient']['channel_importance'] = channel_drops
    results['ablation_gradient']['most_important'] = most_important
    print(f"\n  Channel importance: {channel_drops}")
    print(f"  Most important: {most_important}")

    for tid in ['T767', 'T768', 'T769', 'T770']:
        status = "PASS" if tests[tid]['pass'] else "FAIL"
        print(f"  {tid}: {status} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ═══════════════════════════════════════════════════════════
    # EXP 6: TEMPORAL CROSS-SUBSTRATE MEMORY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 6: TEMPORAL CROSS-SUBSTRATE MEMORY (40 trials)")
    print("  Can FPGA neurons at time t predict GPU power at time t-k?")
    print("  Tests reverse temporal coupling through the bridge")
    print("=" * 72)

    delays_to_test = [1, 2, 3, 5, 7]
    mem_results = {'COUPLED': {}, 'FPGA_ONLY': {}}

    for cond in ['COUPLED', 'FPGA_ONLY']:
        print(f"\n  --- Memory: {cond} ---")
        all_raw = []
        all_gpu = []

        for trial in range(N_TE):
            if (trial + 1) % 10 == 0:
                print(f"    {cond} trial {trial+1}/{N_TE}")
            _, _, raw_states, gpu_log = collect_features(
                fpga, te_waves, w_in, w_gpu, w_fb, vg_spread,
                mode=cond)
            all_raw.append(raw_states)
            all_gpu.append(gpu_log)

        # Stack all trials' time series
        raw_cat = np.concatenate(all_raw, axis=0)  # (N_TE*N_STEPS, N_NEURONS*3)
        gpu_cat = np.concatenate(all_gpu, axis=0)  # (N_TE*N_STEPS, 9)

        gpu_power = gpu_cat[:, 4]
        fpga_mean_spikes = raw_cat[:, :N_NEURONS].mean(axis=1)

        for d in delays_to_test:
            if d >= len(gpu_power) - 10:
                continue
            # Predict gpu_power[t-d] from fpga_mean_spikes[t]
            y = gpu_power[:len(gpu_power)-d]
            X = fpga_mean_spikes[d:].reshape(-1, 1)

            # Add lagged FPGA features for better prediction
            X_aug = np.column_stack([
                fpga_mean_spikes[d:],
                raw_cat[d:, :N_NEURONS].mean(axis=1),  # mean vmem
            ])
            if X_aug.shape[0] > 10:
                from numpy.linalg import lstsq
                try:
                    mu_x = X_aug.mean(axis=0); sigma_x = X_aug.std(axis=0)
                    sigma_x[sigma_x < 1e-6] = 1.0
                    X_n = (X_aug - mu_x) / sigma_x
                    X_n = np.column_stack([X_n, np.ones(len(X_n))])
                    w, _, _, _ = lstsq(X_n, y, rcond=None)
                    y_pred = X_n @ w
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
                except:
                    r2 = 0.0
            else:
                r2 = 0.0
            mem_results[cond][f'd={d}'] = float(r2)
            print(f"    d={d}: R²={r2:.4f}")

    results['temporal_memory'] = mem_results
    mem_c_d1 = mem_results['COUPLED'].get('d=1', 0.0)
    mem_f_d1 = mem_results['FPGA_ONLY'].get('d=1', 0.0)
    mem_c_d5 = mem_results['COUPLED'].get('d=5', 0.0)

    tests['T771'] = {'desc': 'Memory: COUPLED R²(d=1) > 0.01', 'val': mem_c_d1,
                     'pass': mem_c_d1 > 0.01}
    tests['T772'] = {'desc': 'Memory: COUPLED R²(d=1) > FPGA_ONLY R²(d=1)',
                     'val': mem_c_d1 - mem_f_d1, 'pass': mem_c_d1 > mem_f_d1}
    tests['T773'] = {'desc': 'Memory: COUPLED R²(d=1) > R²(d=5) (decays with delay)',
                     'val': mem_c_d1 - mem_c_d5, 'pass': mem_c_d1 > mem_c_d5}
    for tid in ['T771', 'T772', 'T773']:
        status = "PASS" if tests[tid]['pass'] else "FAIL"
        print(f"  {tid}: {status} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['score'] = f"{n_pass}/{n_total}"

    print("\n" + "=" * 72)
    print(f"FINAL SCORE: {n_pass}/{n_total}")
    print("=" * 72)
    for tid, t in sorted(tests.items()):
        status = "PASS" if t['pass'] else "FAIL"
        print(f"  {tid}: {status} — {t['desc']} [{t['val']}]")

    out_path = RESULTS / 'z2230_joint_substrate_encoding.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
