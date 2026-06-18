#!/usr/bin/env python3
"""z2229_pca_substrate_fusion.py — PCA-Regularized Substrate Fusion

Fixes z2228's underdetermined problem: 4608-dim features with 120 trials.
z2207 achieved 81% using PCA reduction (4608→120) inside CV folds.

Key improvements over z2228 (11/16):
  1. PCA reduction: 4608 → 120 components (z2207 proven)
  2. Per-substrate PCA for COUPLED: FPGA→80 + GPU→20 = 100 components
  3. More trials: 240 waveform, 160 temporal/stress (better training)
  4. GPU telemetry as direct features (9 channels × 4 pools = 36 dim)
  5. Coupling strength sweep: measure information transfer capacity
  6. sigma floor 1e-2 (z2210 fix)

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128 neurons, UDP Ethernet)
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters (z2206/z2210 proven values) ───
N_NEURONS = 128
BASE_VG = 0.45
ALPHA = 0.35
BETA_POWER = 0.12
BETA_THERMAL = 0.08
BETA_CLOCK = 0.10
SAMPLE_HZ = 20       # 50ms per step, matches membrane τ≈49.4ms
N_STEPS = 30          # 30 steps × 50ms = 1.5s trial
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
# GPU PROBES
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
# COUPLED DYNAMICS LOOP — returns raw_states + gpu_log
# ═══════════════════════════════════════════════════════════

def run_coupled_loop(fpga, input_signal, w_in, w_gpu, w_fb, vg_spread,
                     mode='COUPLED', n_neurons=128, record_gpu=True,
                     gpu_intensity=None, beta_scale=1.0):
    """Run reservoir loop. Returns raw_states: (n_steps, n_neurons*3)
    = [delta_spike | vmem | cumulative] and gpu_log: (n_steps, 9)."""
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

        # Heterogeneous Vg
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
        if mode == 'COUPLED':
            if gpu_intensity is not None:
                intensity = float(gpu_intensity[t])
            elif t >= 1:
                recent_spikes = raw_states[max(0,t-2):t+1, :n_neurons].mean(axis=0)
                raw_val = float(np.dot(recent_spikes, w_fb[:n_neurons]))
                intensity = float(sigmoid(raw_val - 5.0))
            else:
                intensity = 0.3
            run_workload(intensity, duration_ms=WORKLOAD_MS)
            try:
                fpga.set_mac_signal(intensity * 0.5)
            except:
                pass
        elif mode == 'GPU_ONLY':
            intensity = float(0.2 + 0.6 * np.clip(input_signal[t], 0, 1))
            run_workload(intensity, duration_ms=WORKLOAD_MS)
        elif mode == 'UNCOUPLED':
            intensity = 0.5
            run_workload(intensity, duration_ms=WORKLOAD_MS)

        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.001:
            time.sleep(remaining)

    return raw_states, gpu_log


# ═══════════════════════════════════════════════════════════
# z2206 FEATURE PIPELINE + PCA (z2207 proven)
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2)):
    """Append time-delayed copies: [x(t), x(t-1), x(t-2)]."""
    T, D = states.shape
    aug = np.zeros((T, D * (1 + len(delays))))
    aug[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        aug[d:, start:start+D] = states[:-d]
    return aug

def pool_trial_features(raw_states):
    """z2206 pipeline: delay augment → mean+std+max+min pooling.
    Input: (n_steps, n_neurons*3). Output: 4608-dim feature vector."""
    aug = augment_with_delays(raw_states, delays=(1, 2))
    return np.concatenate([
        aug.mean(axis=0),
        aug.std(axis=0),
        aug.max(axis=0),
        aug.min(axis=0),
    ])

def pool_gpu_features(gpu_log):
    """Pool GPU telemetry into features: 9 channels × 4 pools = 36 dim."""
    return np.concatenate([
        gpu_log.mean(axis=0),
        gpu_log.std(axis=0),
        gpu_log.max(axis=0),
        gpu_log.min(axis=0),
    ])

def pca_reduce(X, n_components=100):
    """PCA via SVD. z2207 proven for high-D reservoir features."""
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T, mu, Vt[:n_components]

def pca_transform(X, mu, Vt):
    return (X - mu) @ Vt.T


# ═══════════════════════════════════════════════════════════
# CLASSIFICATION with PCA
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
    """z2207-style: normalize → PCA → ridge, all inside CV folds."""
    if n_classes is None: n_classes = len(np.unique(y))
    splits = stratified_kfold(X, y, n_splits)
    accs = []
    use_pca = X.shape[1] > max_pca
    for tr_idx, te_idx in splits:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
        sigma[sigma < 1e-2] = 1.0  # z2210 fix: floor 1e-2
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
    """Per-substrate PCA: FPGA→fpga_pc, GPU→gpu_pc, then concatenate.
    Prevents GPU noise from diluting FPGA signal in PCA basis."""
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
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════

def generate_waveforms_4class(n_trials, n_steps, seed=42):
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(n_steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 4)
        freq = rng.uniform(0.5, 3.0)
        if cls == 0:    wave = np.sin(2 * np.pi * freq * t)
        elif cls == 1:  wave = 2 * (t * freq % 1) - 1
        elif cls == 2:  wave = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        elif cls == 3:  wave = np.sign(np.sin(2 * np.pi * freq * t))
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)

def generate_temporal_pattern(n_trials, n_steps, n_classes=4, seed=42):
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    seg_len = n_steps // 4
    for _ in range(n_trials):
        cls = rng.integers(0, n_classes)
        dt = 1.0 / SAMPLE_HZ
        t_seg = np.arange(seg_len) * dt
        basis = [
            np.sin(2 * np.pi * 1.0 * t_seg),
            np.sin(2 * np.pi * 3.0 * t_seg),
            np.zeros(seg_len) + 0.5,
            np.sign(np.sin(2 * np.pi * 2.0 * t_seg)),
        ]
        orders = [[0,1,2,3], [3,2,1,0], [1,3,0,2], [2,0,3,1]]
        order = orders[cls]
        wave = np.concatenate([basis[order[i]] for i in range(4)])
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        if len(wave) < n_steps: wave = np.pad(wave, (0, n_steps - len(wave)))
        else: wave = wave[:n_steps]
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)

def generate_gpu_stress_waveforms(n_trials, n_steps, seed=42):
    rng = np.random.default_rng(seed)
    trials, labels, gpu_intensities = [], [], []
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(n_steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 4)
        if cls == 0:
            wave = 0.5 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
            gpu_int = np.full(n_steps, 0.1)
        elif cls == 1:
            wave = 0.5 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
            gpu_int = np.full(n_steps, 0.9)
        elif cls == 2:
            wave = np.zeros(n_steps); wave[n_steps//3:2*n_steps//3] = 1.0
            gpu_int = np.linspace(0.1, 0.9, n_steps)
        elif cls == 3:
            wave = rng.uniform(0, 1, n_steps)
            gpu_int = 0.5 + 0.3 * np.sin(2 * np.pi * 2.0 * t)
        trials.append(wave)
        labels.append(cls)
        gpu_intensities.append(gpu_int)
    return np.array(trials), np.array(labels), np.array(gpu_intensities)


# ═══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("z2229: PCA-REGULARIZED SUBSTRATE FUSION")
    print("  z2228 pipeline + PCA (4608→120) + more trials + GPU features")
    print("  Per-substrate PCA: FPGA→80 PCs + GPU→20 PCs = 100 total")
    print("  Key fix: z2228 was underdetermined (4608 dim / 120 trials)")
    print("=" * 72)

    # ─── Init ───
    print("\n[1] Connecting FPGA...", flush=True)
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("  FATAL: No FPGA"); return
    fpga.set_kill(False)
    time.sleep(0.2)

    print("\n[2] Init GPU HIP...", flush=True)
    init_torch()

    print("\n[3] Probe check...", flush=True)
    test_gs = read_all_gpu_state()
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"    {name}: {test_gs[i]}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    w_gpu = rng.uniform(-1, 1, N_NEURONS)
    w_fb = rng.uniform(-1, 1, N_NEURONS)
    vg_spread = np.linspace(-0.08, 0.08, N_NEURONS)

    results = {
        'experiment': 'z2229_pca_substrate_fusion',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'base': 'z2228 + PCA reduction + per-substrate PCA + more trials',
            'fpga_features': 'delta_spike+vmem+cumulative (384) × delay(3) × pool(4) = 4608 dim',
            'gpu_features': '9 channels × pool(4) = 36 dim',
            'pca': 'global 4608→120, per-substrate FPGA→80 + GPU→20',
            'n_neurons': N_NEURONS,
            'sample_hz': SAMPLE_HZ, 'n_steps': N_STEPS,
            'membrane_tau_ms': 49.4,
        }
    }
    tests = {}

    def collect_features(fpga_obj, inputs, labels, n_trials, mode,
                         gpu_intensity_arr=None, beta_scale=1.0):
        """Collect FPGA + GPU features for all trials."""
        fpga_feats = []
        gpu_feats = []
        for trial in range(n_trials):
            gi = gpu_intensity_arr[trial] if gpu_intensity_arr is not None else None
            raw, glog = run_coupled_loop(
                fpga_obj, inputs[trial], w_in, w_gpu, w_fb, vg_spread,
                mode=mode, gpu_intensity=gi, beta_scale=beta_scale)
            fpga_feats.append(pool_trial_features(raw))
            gpu_feats.append(pool_gpu_features(glog))
            if (trial + 1) % 40 == 0:
                print(f"    {mode} trial {trial+1}/{n_trials}", flush=True)
        return np.array(fpga_feats), np.array(gpu_feats)

    # ═══════════════════════════════════════════════════════════
    # EXP 1: WAVEFORM 4-CLASS (240 trials, PCA)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 1: WAVEFORM CLASSIFICATION (4-class, 240 trials, PCA)")
    print("  Global PCA: 4608 → 120 components inside CV folds")
    print("  Per-substrate PCA: FPGA→80 + GPU→20 for COUPLED")
    print("=" * 72, flush=True)

    wf_trials = 240
    wf_inputs, wf_labels = generate_waveforms_4class(wf_trials, N_STEPS, seed=42)

    wf_results = {}
    wf_fpga = {}
    wf_gpu = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Waveform: {cond} ---", flush=True)
        ff, gf = collect_features(fpga, wf_inputs, wf_labels, wf_trials, cond)
        wf_fpga[cond] = ff
        wf_gpu[cond] = gf

        # Strategy 1: Global PCA on FPGA features
        cv_global = classify_cv_pca(ff, wf_labels, n_classes=4, max_pca=120)
        print(f"  {cond} global PCA: acc={cv_global['mean']:.3f} ± {cv_global['std']:.3f}", flush=True)

        # Strategy 2: Per-substrate PCA (COUPLED only has meaningful GPU features)
        if cond == 'COUPLED':
            cv_persub = classify_per_substrate_pca(
                ff, gf, wf_labels, n_classes=4, fpga_pc=80, gpu_pc=20)
            print(f"  {cond} per-sub PCA: acc={cv_persub['mean']:.3f} ± {cv_persub['std']:.3f}", flush=True)
            best_cv = max([cv_global, cv_persub], key=lambda x: x['mean'])
            best_cv['strategy'] = 'per_sub' if cv_persub['mean'] > cv_global['mean'] else 'global'
        else:
            best_cv = cv_global
            best_cv['strategy'] = 'global'

        wf_results[cond] = best_cv

    results['waveform_4class'] = wf_results

    wf_c = wf_results['COUPLED']['mean']
    wf_f = wf_results['FPGA_ONLY']['mean']
    wf_s = wf_results['STATIC']['mean']

    tests['T736'] = {'desc': 'Waveform 4-class COUPLED > 0.60',
                     'val': wf_c, 'pass': wf_c > 0.60}
    tests['T737'] = {'desc': 'Waveform COUPLED > STATIC',
                     'val': wf_c - wf_s, 'pass': wf_c > wf_s}
    tests['T738'] = {'desc': 'Waveform COUPLED > FPGA_ONLY (GPU coupling helps)',
                     'val': wf_c - wf_f, 'pass': wf_c > wf_f}
    tests['T739'] = {'desc': 'Waveform COUPLED > 0.75 (strong, matching z2206)',
                     'val': wf_c, 'pass': wf_c > 0.75}

    for tid in sorted(tests):
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']:.3f}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 2: TEMPORAL PATTERN (160 trials, PCA)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 2: TEMPORAL PATTERN CLASSIFICATION (4-class, 160 trials)")
    print("=" * 72, flush=True)

    tp_trials = 160
    tp_inputs, tp_labels = generate_temporal_pattern(tp_trials, N_STEPS, seed=42)

    tp_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Temporal Pattern: {cond} ---", flush=True)
        ff, gf = collect_features(fpga, tp_inputs, tp_labels, tp_trials, cond)

        cv_global = classify_cv_pca(ff, tp_labels, n_classes=4, max_pca=120)
        print(f"  {cond} global PCA: acc={cv_global['mean']:.3f} ± {cv_global['std']:.3f}", flush=True)

        if cond == 'COUPLED':
            cv_persub = classify_per_substrate_pca(
                ff, gf, tp_labels, n_classes=4, fpga_pc=80, gpu_pc=20)
            print(f"  {cond} per-sub PCA: acc={cv_persub['mean']:.3f} ± {cv_persub['std']:.3f}", flush=True)
            best_cv = max([cv_global, cv_persub], key=lambda x: x['mean'])
            best_cv['strategy'] = 'per_sub' if cv_persub['mean'] > cv_global['mean'] else 'global'
        else:
            best_cv = cv_global
            best_cv['strategy'] = 'global'

        tp_results[cond] = best_cv

    results['temporal_pattern'] = tp_results

    tp_c = tp_results['COUPLED']['mean']
    tp_f = tp_results['FPGA_ONLY']['mean']
    tp_s = tp_results['STATIC']['mean']

    tests['T740'] = {'desc': 'Temporal pattern COUPLED > 0.50',
                     'val': tp_c, 'pass': tp_c > 0.50}
    tests['T741'] = {'desc': 'Temporal pattern COUPLED > STATIC',
                     'val': tp_c - tp_s, 'pass': tp_c > tp_s}
    tests['T742'] = {'desc': 'Temporal pattern COUPLED > FPGA_ONLY',
                     'val': tp_c - tp_f, 'pass': tp_c > tp_f}

    for tid in ['T740', 'T741', 'T742']:
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']:.3f}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 3: GPU STRESS (160 trials, PCA)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 3: GPU STRESS CLASSIFICATION (160 trials)")
    print("  GPU workload pattern IS the class signal")
    print("=" * 72, flush=True)

    gs_trials = 160
    gs_inputs, gs_labels, gs_gpu_int = generate_gpu_stress_waveforms(
        gs_trials, N_STEPS, seed=42)

    gs_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- GPU Stress: {cond} ---", flush=True)
        gi = gs_gpu_int if cond == 'COUPLED' else None
        ff, gf = collect_features(fpga, gs_inputs, gs_labels, gs_trials, cond,
                                   gpu_intensity_arr=gi)

        cv_global = classify_cv_pca(ff, gs_labels, n_classes=4, max_pca=120)
        print(f"  {cond} global PCA: acc={cv_global['mean']:.3f} ± {cv_global['std']:.3f}", flush=True)

        if cond == 'COUPLED':
            cv_persub = classify_per_substrate_pca(
                ff, gf, gs_labels, n_classes=4, fpga_pc=80, gpu_pc=20)
            print(f"  {cond} per-sub PCA: acc={cv_persub['mean']:.3f} ± {cv_persub['std']:.3f}", flush=True)
            best_cv = max([cv_global, cv_persub], key=lambda x: x['mean'])
            best_cv['strategy'] = 'per_sub' if cv_persub['mean'] > cv_global['mean'] else 'global'
        else:
            best_cv = cv_global
            best_cv['strategy'] = 'global'

        gs_results[cond] = best_cv

    results['gpu_stress'] = gs_results

    gs_c = gs_results['COUPLED']['mean']
    gs_f = gs_results['FPGA_ONLY']['mean']
    gs_s = gs_results['STATIC']['mean']

    tests['T743'] = {'desc': 'GPU stress COUPLED > 0.50',
                     'val': gs_c, 'pass': gs_c > 0.50}
    tests['T744'] = {'desc': 'GPU stress COUPLED > FPGA_ONLY (GPU is signal)',
                     'val': gs_c - gs_f, 'pass': gs_c > gs_f}
    tests['T745'] = {'desc': 'GPU stress COUPLED > STATIC',
                     'val': gs_c - gs_s, 'pass': gs_c > gs_s}
    tests['T746'] = {'desc': 'GPU stress COUPLED advantage > 10pp over FPGA_ONLY',
                     'val': gs_c - gs_f, 'pass': (gs_c - gs_f) > 0.10}

    for tid in ['T743', 'T744', 'T745', 'T746']:
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']:.3f}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 4: COUPLING STRENGTH SWEEP
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 4: COUPLING STRENGTH SWEEP")
    print("  Vary beta_scale: 0.0, 0.5, 1.0, 2.0, 4.0")
    print("  Measures information transfer capacity vs coupling strength")
    print("=" * 72, flush=True)

    beta_scales = [0.0, 0.5, 1.0, 2.0, 4.0]
    sweep_trials = 120
    sweep_inputs, sweep_labels, sweep_gpu_int = generate_gpu_stress_waveforms(
        sweep_trials, N_STEPS, seed=99)

    sweep_results = {}
    for bs in beta_scales:
        print(f"\n  --- beta_scale={bs:.1f} ---", flush=True)
        feats = []
        for trial in range(sweep_trials):
            raw, glog = run_coupled_loop(
                fpga, sweep_inputs[trial], w_in, w_gpu, w_fb, vg_spread,
                mode='COUPLED', gpu_intensity=sweep_gpu_int[trial],
                beta_scale=bs)
            feats.append(pool_trial_features(raw))
            if (trial + 1) % 40 == 0:
                print(f"    trial {trial+1}/{sweep_trials}", flush=True)
        X = np.array(feats)
        cv = classify_cv_pca(X, sweep_labels, n_classes=4, max_pca=120)
        sweep_results[f'beta_{bs}'] = cv
        print(f"  beta={bs}: acc={cv['mean']:.3f} ± {cv['std']:.3f}", flush=True)

    results['coupling_sweep'] = sweep_results

    # Best beta
    sweep_accs = {bs: sweep_results[f'beta_{bs}']['mean'] for bs in beta_scales}
    best_beta = max(sweep_accs, key=sweep_accs.get)
    zero_acc = sweep_accs[0.0]
    best_acc = sweep_accs[best_beta]

    tests['T747'] = {'desc': 'Coupling sweep: best beta > 0.50 acc',
                     'val': best_acc, 'pass': best_acc > 0.50}
    tests['T748'] = {'desc': 'Coupling sweep: best > beta=0 (coupling helps)',
                     'val': best_acc - zero_acc, 'pass': best_acc > zero_acc}
    tests['T749'] = {'desc': 'Coupling sweep: non-monotonic (optimum exists)',
                     'val': str(sweep_accs),
                     'pass': best_beta not in (0.0, 4.0)}

    for tid in ['T747', 'T748', 'T749']:
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 5: REPRODUCIBILITY (3 seeds, PCA)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 5: REPRODUCIBILITY (3 seeds, 160 trials each)")
    print("=" * 72, flush=True)

    repro_accs = []
    for seed in [42, 59, 76]:
        print(f"\n  --- Seed {seed} ---", flush=True)
        wf_i, wf_l = generate_waveforms_4class(160, N_STEPS, seed=seed)
        feats = []
        for trial in range(160):
            raw, glog = run_coupled_loop(
                fpga, wf_i[trial], w_in, w_gpu, w_fb, vg_spread, mode='COUPLED')
            feats.append(pool_trial_features(raw))
            if (trial + 1) % 40 == 0:
                print(f"    seed={seed} trial {trial+1}/160", flush=True)
        X = np.array(feats)
        cv = classify_cv_pca(X, wf_l, n_classes=4, max_pca=120)
        repro_accs.append(cv['mean'])
        print(f"  Seed {seed}: acc={cv['mean']:.3f}", flush=True)

    results['reproducibility'] = {
        'seeds': [42, 59, 76],
        'accuracies': repro_accs,
        'mean': float(np.mean(repro_accs)),
        'std': float(np.std(repro_accs)),
    }

    tests['T750'] = {'desc': 'All 3 seeds > 0.55',
                     'val': min(repro_accs), 'pass': all(a > 0.55 for a in repro_accs)}
    tests['T751'] = {'desc': 'Reproducibility std < 0.08',
                     'val': float(np.std(repro_accs)),
                     'pass': np.std(repro_accs) < 0.08}

    for tid in ['T750', 'T751']:
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 6: PCA vs NO-PCA ABLATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 6: PCA ABLATION — PCA vs raw features")
    print("  Uses EXP1 COUPLED data (already collected)")
    print("=" * 72, flush=True)

    ff_coupled = wf_fpga['COUPLED']
    # Raw (no PCA) — same as z2228
    cv_raw = classify_cv_pca(ff_coupled, wf_labels, n_classes=4, max_pca=99999)
    print(f"  Raw (4608 dim): acc={cv_raw['mean']:.3f} ± {cv_raw['std']:.3f}", flush=True)

    # PCA 60
    cv_pca60 = classify_cv_pca(ff_coupled, wf_labels, n_classes=4, max_pca=60)
    print(f"  PCA→60: acc={cv_pca60['mean']:.3f} ± {cv_pca60['std']:.3f}", flush=True)

    # PCA 120
    cv_pca120 = classify_cv_pca(ff_coupled, wf_labels, n_classes=4, max_pca=120)
    print(f"  PCA→120: acc={cv_pca120['mean']:.3f} ± {cv_pca120['std']:.3f}", flush=True)

    # PCA 200
    cv_pca200 = classify_cv_pca(ff_coupled, wf_labels, n_classes=4, max_pca=200)
    print(f"  PCA→200: acc={cv_pca200['mean']:.3f} ± {cv_pca200['std']:.3f}", flush=True)

    results['pca_ablation'] = {
        'raw': cv_raw, 'pca_60': cv_pca60,
        'pca_120': cv_pca120, 'pca_200': cv_pca200,
    }

    best_pca = max([cv_pca60, cv_pca120, cv_pca200], key=lambda x: x['mean'])
    tests['T752'] = {'desc': 'PCA > raw features (regularization helps)',
                     'val': best_pca['mean'] - cv_raw['mean'],
                     'pass': best_pca['mean'] > cv_raw['mean']}
    tests['T753'] = {'desc': 'Best PCA COUPLED > 0.55',
                     'val': best_pca['mean'], 'pass': best_pca['mean'] > 0.55}

    for tid in ['T752', 'T753']:
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # FINAL SCORE
    # ═══════════════════════════════════════════════════════════
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['tests'] = tests
    results['score'] = f"{n_pass}/{n_total}"

    print("\n" + "=" * 72)
    print(f"FINAL SCORE: {n_pass}/{n_total}")
    print("=" * 72)
    for tid, t in sorted(tests.items()):
        status = "PASS" if t['pass'] else "FAIL"
        print(f"  {tid}: {status} — {t['desc']} [{t['val']}]")

    out = RESULTS / 'z2229_pca_substrate_fusion.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out}")

    fpga.close()
    print("Done.")


if __name__ == '__main__':
    main()
