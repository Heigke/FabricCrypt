#!/usr/bin/env python3
"""z2226_temporal_benchmark.py — Trial-Level Temporal Processing Benchmark

Tests whether MC=0.556 (slow membrane τ≈49.4ms) + lateral connections + heterogeneous Vg
translate to real temporal tasks at the TRIAL level (not step level).

1. TEMPORAL PATTERN: 4-class classification by temporal ORDER of segments
2. WAVEFORM DISCRIMINATION: 6-class waveform classification (proven 81-93.5% in z2206/z2210)
3. SEQUENCE MEMORY: Reconstruct delayed signal from reservoir trajectory
4. TEMPORAL INTEGRATION: Predict running statistics of input from reservoir state
5. MC-TASK CORRELATION: Correlate MC profile with per-delay memory reconstruction

All with COUPLED/FPGA_ONLY/STATIC to confirm substrate coupling advantage.
Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128 neurons, UDP Ethernet)
Lateral connections: N±1 (w=0.125), N±2 (w=0.0625) ring topology
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters (z2225 proven values) ───
N_NEURONS = 128
BASE_VG = 0.45
ALPHA = 0.35
BETA_POWER = 0.12
BETA_THERMAL = 0.08
BETA_CLOCK = 0.10
SAMPLE_HZ = 200
WORKLOAD_MS = 1.5
N_STEPS = 300
N_TRIALS = 30

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
# GPU PROBES — Same 9 channels as z2225
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

_torch_device = None
_torch_available = False

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

def run_workload(intensity, duration_ms=1.5):
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
# COUPLED DYNAMICS LOOP — with heterogeneous Vg
# ═══════════════════════════════════════════════════════════

def run_coupled_loop(fpga, input_signal, w_in, w_gpu, w_fb, vg_spread,
                     mode='COUPLED', n_neurons=128, record_gpu=True):
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    spikes = np.zeros((n_steps, n_neurons))
    vmem_log = np.zeros((n_steps, n_neurons))
    gpu_log = np.zeros((n_steps, 9))
    intensities = np.zeros(n_steps)
    prev_counts = None

    for t in range(n_steps):
        t_start = time.perf_counter()

        if record_gpu:
            gpu_log[t] = read_all_gpu_state()

        # Heterogeneous Vg: each neuron has unique base voltage
        vg = BASE_VG + vg_spread[:n_neurons]

        if mode in ('COUPLED', 'FPGA_ONLY'):
            vg += ALPHA * input_signal[t] * w_in[:n_neurons]

        if mode == 'COUPLED':
            hw_p = gpu_log[t, 4]
            pm_sclk = gpu_log[t, 3]
            pm_t = gpu_log[t, 1]

            if t >= 5:
                p_base = gpu_log[max(0,t-20):t, 4].mean()
                s_base = gpu_log[max(0,t-20):t, 3].mean()
                t_base = gpu_log[max(0,t-20):t, 1].mean()
            else:
                p_base, s_base, t_base = hw_p, pm_sclk, pm_t

            p_delta = (hw_p - p_base) / max(abs(p_base), 1.0)
            s_delta = (pm_sclk - s_base) / max(abs(s_base), 1.0)
            t_delta = (pm_t - t_base) / max(abs(t_base), 1.0)

            n3 = n_neurons // 3
            wg = w_gpu[:n_neurons]
            vg[:n3] += BETA_POWER * p_delta * wg[:n3]
            vg[n3:2*n3] += BETA_CLOCK * s_delta * wg[n3:2*n3]
            vg[2*n3:] += BETA_THERMAL * t_delta * wg[2*n3:]

            gs_pm_t = gpu_log[t, 1]
            if gs_pm_t > 0:
                try:
                    fpga.set_temp(float(gs_pm_t) + 273.15)
                except:
                    pass

        vg = np.clip(vg, 0.10, 0.85)

        if mode != 'GPU_ONLY':
            fpga.set_vg_batch(0, vg.tolist())
            time.sleep(0.0003)

            try:
                counts, vm, refract = fpga.read_telemetry_fast()
                if prev_counts is not None:
                    for i in range(n_neurons):
                        delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                        if delta > 30000: delta = 0
                        spikes[t, i] = delta
                vmem_log[t] = vm[:n_neurons]
                prev_counts = counts.copy()
            except:
                pass

        if mode == 'COUPLED':
            if t >= 1:
                recent_spikes = spikes[max(0,t-2):t+1].mean(axis=0)
                raw = float(np.dot(recent_spikes, w_fb[:n_neurons]))
                intensity = float(sigmoid(raw - 5.0))
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
        else:
            intensity = 0.0

        intensities[t] = intensity

        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.0003:
            time.sleep(remaining)

    return spikes, vmem_log, gpu_log, intensities


# ═══════════════════════════════════════════════════════════
# TRIAL-LEVEL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_trial_features(spk, vm, mode='temporal'):
    """Extract trial-level features from full vmem/spike trajectories.

    mode='temporal': Quarter-segment mean+std (8 × N_NEURONS = 1024 dim)
    mode='rich': temporal + spike stats + cross-neuron correlations
    """
    n_steps, n_neurons = vm.shape
    n_q = n_steps // 4

    if mode == 'temporal':
        feats = []
        for q in range(4):
            seg = vm[q*n_q:(q+1)*n_q]
            feats.extend([seg.mean(axis=0), seg.std(axis=0)])
        return np.concatenate(feats)

    if mode == 'rich':
        feats = []
        # Quarter-segment mean+std for vmem (8 × 128 = 1024)
        for q in range(4):
            seg = vm[q*n_q:(q+1)*n_q]
            feats.extend([seg.mean(axis=0), seg.std(axis=0)])
        # Spike rate per quarter (4 × 128 = 512)
        for q in range(4):
            seg = spk[q*n_q:(q+1)*n_q]
            feats.append(seg.sum(axis=0))
        # Cross-neuron correlation: top-16 neurons by vmem variance
        var = vm.var(axis=0)
        top_idx = np.argsort(var)[-16:]
        vm_top = vm[:, top_idx]
        # Pairwise correlations (120 dim)
        corrs = []
        for i in range(16):
            for j in range(i + 1, 16):
                c = np.corrcoef(vm_top[:, i], vm_top[:, j])[0, 1]
                corrs.append(c if np.isfinite(c) else 0.0)
        feats.append(np.array(corrs))
        return np.concatenate(feats)

    return np.concatenate([vm.mean(axis=0), vm.std(axis=0)])


def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=None):
    if n_classes is None: n_classes = max(len(np.unique(y_tr)), len(np.unique(y_te)))
    alphas = [1e-4, 1e-2, 1.0, 100.0, 10000.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    Xts = (X_tr - mu) / sigma; Xes = (X_te - mu) / sigma
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, y in enumerate(y_tr): Y_tr[i, int(y)] = 1.0
    best = -1
    for a in alphas:
        I = np.eye(Xts.shape[1])
        try: W = np.linalg.solve(Xts.T @ Xts + a * I, Xts.T @ Y_tr)
        except: continue
        acc = np.mean(np.argmax(Xes @ W, axis=1) == y_te)
        if acc > best: best = acc
    return best

def ridge_solve(X_tr, y_tr, X_te, y_te, alpha=1.0):
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    Xts = (X_tr - mu) / sigma; Xes = (X_te - mu) / sigma
    I = np.eye(Xts.shape[1])
    try:
        w = np.linalg.solve(Xts.T @ Xts + alpha * I, Xts.T @ y_tr)
        y_pred = Xes @ w
        ss_res = np.sum((y_te - y_pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
        nrmse = np.sqrt(ss_res / len(y_te)) / max(np.std(y_te), 1e-10)
        return y_pred, float(r2), float(nrmse)
    except:
        return np.zeros_like(y_te), 0.0, 999.0

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

def classify_cv(X, y, n_splits=5, n_classes=None):
    splits = stratified_kfold(X, y, n_splits)
    accs = [ridge_classify(X[tr], y[tr], X[te], y[te], n_classes=n_classes)
            for tr, te in splits]
    return {'mean': float(np.mean(accs)), 'std': float(np.std(accs))}


# ═══════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════

def generate_temporal_pattern(n_trials, n_steps, n_classes=4, seed=42):
    """Sequences where class depends on TEMPORAL ORDER, not just statistics."""
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
        if len(wave) < n_steps:
            wave = np.pad(wave, (0, n_steps - len(wave)))
        else:
            wave = wave[:n_steps]
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)

def generate_waveforms(n_trials, n_steps, n_classes=6, seed=42):
    """6-class waveform classification — proven task from z2206/z2210.
    Classes: sine, sawtooth, triangle, square, chirp, noise-modulated-sine
    """
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(n_steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, n_classes)
        freq = rng.uniform(0.5, 3.0)  # Random frequency
        if cls == 0:  # Sine
            wave = np.sin(2 * np.pi * freq * t)
        elif cls == 1:  # Sawtooth
            wave = 2 * (t * freq % 1) - 1
        elif cls == 2:  # Triangle
            wave = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        elif cls == 3:  # Square
            wave = np.sign(np.sin(2 * np.pi * freq * t))
        elif cls == 4:  # Chirp
            f1 = rng.uniform(0.3, 1.0)
            f2 = rng.uniform(2.0, 5.0)
            inst_freq = f1 + (f2 - f1) * t / t[-1]
            wave = np.sin(2 * np.pi * np.cumsum(inst_freq) * dt)
        elif cls == 5:  # Noise-modulated sine
            wave = np.sin(2 * np.pi * freq * t) * (1.0 + 0.3 * np.cumsum(rng.normal(0, 0.1, n_steps)))
        # Normalize to [0, 1]
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)

def generate_temporal_integration(n_trials, n_steps, seed=42):
    """Random walk inputs → predict running mean/variance.
    Tests temporal integration capacity of the reservoir.
    """
    rng = np.random.default_rng(seed)
    inputs = []
    targets_mean = []
    targets_var = []
    window = 20  # 100ms integration window at 200Hz
    for _ in range(n_trials):
        u = np.cumsum(rng.normal(0, 0.1, n_steps))
        u = (u - u.min()) / max(u.max() - u.min(), 1e-6)
        inputs.append(u)
        # Running mean and variance over last 20 steps
        rm = np.zeros(n_steps)
        rv = np.zeros(n_steps)
        for t in range(n_steps):
            w = u[max(0, t-window+1):t+1]
            rm[t] = w.mean()
            rv[t] = w.var() if len(w) > 1 else 0
        targets_mean.append(rm)
        targets_var.append(rv)
    return np.array(inputs), np.array(targets_mean), np.array(targets_var)


# ═══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("z2226: TRIAL-LEVEL TEMPORAL PROCESSING BENCHMARK")
    print("  Slow membrane τ≈49.4ms + lateral connections + heterogeneous Vg")
    print("  Lateral: N±1 (w=0.125), N±2 (w=0.0625) ring topology")
    print("  Features: trial-level temporal (quarter-segment stats)")
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

    # Per-neuron Vg spread: ±0.08 around BASE_VG for input diversity
    vg_spread = np.linspace(-0.08, 0.08, N_NEURONS)

    results = {
        'experiment': 'z2226_temporal_benchmark',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'base': 'z2225 (24/30) + slow membrane + lateral connections + heterogeneous Vg',
            'vg_spread': '±0.08 linspace across 128 neurons',
            'features': 'trial-level: quarter-segment mean+std (1024 dim)',
            'lateral': 'N±1 (w=0.125), N±2 (w=0.0625) ring topology',
            'channels': CHANNEL_NAMES,
            'n_neurons': N_NEURONS,
            'sample_hz': SAMPLE_HZ, 'n_steps': N_STEPS,
            'membrane_tau_ms': 49.4,
        }
    }
    tests = {}

    # ═══════════════════════════════════════════════════════════
    # EXP 1: WAVEFORM CLASSIFICATION (6-class)
    #   Proven task from z2206 (81.0%) and z2210 L3 (93.5%)
    #   Tests fundamental reservoir computation capacity
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 1: WAVEFORM CLASSIFICATION (6-class)")
    print("  Proven task: sine, sawtooth, triangle, square, chirp, noise-sine")
    print("  Tests trial-level temporal discrimination")
    print("=" * 72, flush=True)

    wf_trials = 90  # 15 per class
    wf_inputs, wf_labels = generate_waveforms(wf_trials, N_STEPS, n_classes=6, seed=42)

    wf_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Waveform: {cond} ---", flush=True)
        feats = []
        for trial in range(wf_trials):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, wf_inputs[trial], w_in, w_gpu, w_fb, vg_spread, mode=cond)
            feats.append(extract_trial_features(spk, vm, mode='temporal'))
            if (trial + 1) % 30 == 0:
                print(f"    {cond} trial {trial+1}/{wf_trials}", flush=True)

        X = np.array(feats)
        cv = classify_cv(X, wf_labels, n_classes=6)
        wf_results[cond] = cv
        print(f"  {cond}: acc={cv['mean']:.3f} ± {cv['std']:.3f}", flush=True)

    results['waveform_6class'] = wf_results

    # T680: COUPLED waveform > 0.50 (above chance 0.167 by large margin)
    wf_c = wf_results['COUPLED']['mean']
    tests['T680'] = {'desc': 'Waveform 6-class COUPLED > 0.50',
                     'val': wf_c, 'pass': wf_c > 0.50}
    print(f"\n  T680: COUPLED = {wf_c:.3f} {'PASS' if wf_c > 0.50 else 'FAIL'}", flush=True)

    # T681: COUPLED > STATIC (reservoir advantage over noise floor)
    wf_s = wf_results['STATIC']['mean']
    tests['T681'] = {'desc': 'Waveform COUPLED > STATIC',
                     'val': wf_c - wf_s, 'pass': wf_c > wf_s}
    print(f"  T681: COUPLED({wf_c:.3f}) > STATIC({wf_s:.3f}) "
          f"{'PASS' if wf_c > wf_s else 'FAIL'}", flush=True)

    # T682: COUPLED > FPGA_ONLY (GPU coupling helps)
    wf_f = wf_results['FPGA_ONLY']['mean']
    tests['T682'] = {'desc': 'Waveform COUPLED > FPGA_ONLY (GPU coupling helps)',
                     'val': wf_c - wf_f, 'pass': wf_c > wf_f}
    print(f"  T682: COUPLED({wf_c:.3f}) > FPGA_ONLY({wf_f:.3f}) "
          f"{'PASS' if wf_c > wf_f else 'FAIL'}", flush=True)

    # T683: COUPLED > 0.70 (strong classification, matching z2206/z2210)
    tests['T683'] = {'desc': 'Waveform COUPLED > 0.70 (strong)',
                     'val': wf_c, 'pass': wf_c > 0.70}
    print(f"  T683: COUPLED = {wf_c:.3f} {'PASS' if wf_c > 0.70 else 'FAIL'}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 2: TEMPORAL PATTERN CLASSIFICATION (4-class)
    #   Same components, different temporal ORDER
    #   Tests temporal memory: statistics alone insufficient
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 2: TEMPORAL PATTERN CLASSIFICATION (4-class)")
    print("  Same frequency components, different temporal order")
    print("  Requires temporal memory — statistics identical across classes")
    print("=" * 72, flush=True)

    tp_trials = 80  # 20 per class
    tp_inputs, tp_labels = generate_temporal_pattern(tp_trials, N_STEPS, n_classes=4, seed=42)

    tp_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Temporal Pattern: {cond} ---", flush=True)
        feats = []
        for trial in range(tp_trials):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, tp_inputs[trial], w_in, w_gpu, w_fb, vg_spread, mode=cond)
            feats.append(extract_trial_features(spk, vm, mode='temporal'))
            if (trial + 1) % 20 == 0:
                print(f"    {cond} trial {trial+1}/{tp_trials}", flush=True)

        X = np.array(feats)
        cv = classify_cv(X, tp_labels, n_classes=4)
        tp_results[cond] = cv
        print(f"  {cond}: acc={cv['mean']:.3f} ± {cv['std']:.3f}", flush=True)

    results['temporal_pattern'] = tp_results

    # T684: COUPLED temporal pattern > 0.40 (above chance 0.25)
    tp_c = tp_results['COUPLED']['mean']
    tests['T684'] = {'desc': 'Temporal pattern COUPLED > 0.40',
                     'val': tp_c, 'pass': tp_c > 0.40}
    print(f"\n  T684: COUPLED = {tp_c:.3f} {'PASS' if tp_c > 0.40 else 'FAIL'}", flush=True)

    # T685: COUPLED > STATIC (temporal memory advantage)
    tp_s = tp_results['STATIC']['mean']
    tests['T685'] = {'desc': 'Temporal pattern COUPLED > STATIC',
                     'val': tp_c - tp_s, 'pass': tp_c > tp_s}
    print(f"  T685: COUPLED({tp_c:.3f}) > STATIC({tp_s:.3f}) "
          f"{'PASS' if tp_c > tp_s else 'FAIL'}", flush=True)

    # T686: COUPLED > FPGA_ONLY (GPU temporal dynamics help)
    tp_f = tp_results['FPGA_ONLY']['mean']
    tests['T686'] = {'desc': 'Temporal pattern COUPLED > FPGA_ONLY',
                     'val': tp_c - tp_f, 'pass': tp_c > tp_f}
    print(f"  T686: COUPLED({tp_c:.3f}) > FPGA_ONLY({tp_f:.3f}) "
          f"{'PASS' if tp_c > tp_f else 'FAIL'}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 3: SEQUENCE MEMORY (Delayed Regression)
    #   Predict u(t-d) from reservoir state (trial-level R² at each delay)
    #   Same as MC but with smooth signals and trial aggregation
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 3: SEQUENCE MEMORY (Delayed Regression)")
    print("  Predict u(t-d) from reservoir state for d=1..10")
    print("  Smooth sinusoidal inputs, trial-level aggregation")
    print("=" * 72, flush=True)

    mem_trials = 40
    mem_delays = [1, 2, 3, 5, 7, 10]
    mem_rng = np.random.default_rng(99)
    dt = 1.0 / SAMPLE_HZ
    t_axis = np.arange(N_STEPS) * dt
    mem_inputs = []
    for _ in range(mem_trials):
        freq = mem_rng.uniform(0.3, 2.0)
        phase = mem_rng.uniform(0, 2 * np.pi)
        wave = 0.5 + 0.4 * np.sin(2 * np.pi * freq * t_axis + phase)
        wave += 0.1 * mem_rng.normal(0, 1, N_STEPS)
        mem_inputs.append(np.clip(wave, 0, 1))
    mem_inputs = np.array(mem_inputs)

    mem_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Seq Memory: {cond} ---", flush=True)
        all_states = []

        for trial in range(mem_trials):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, mem_inputs[trial], w_in, w_gpu, w_fb, vg_spread, mode=cond)
            all_states.append(vm)  # Raw vmem trajectories
            if (trial + 1) % 10 == 0:
                print(f"    {cond} trial {trial+1}/{mem_trials}", flush=True)

        all_states = np.array(all_states)
        cond_r2 = {}

        for d in mem_delays:
            features = []
            targets = []
            for trial in range(mem_trials):
                warmup = max(d + 5, 20)
                for t in range(warmup, N_STEPS):
                    features.append(all_states[trial, t])
                    targets.append(mem_inputs[trial, t - d])
            X = np.array(features)
            y = np.array(targets)
            n = len(y)
            idx = mem_rng.permutation(n)
            split = int(0.7 * n)

            best_r2 = -999
            for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
                _, r2, _ = ridge_solve(X[idx[:split]], y[idx[:split]],
                                       X[idx[split:]], y[idx[split:]], alpha=alpha)
                if r2 > best_r2: best_r2 = r2

            cond_r2[d] = float(best_r2)
            print(f"    d={d}: R²={best_r2:.4f}", flush=True)

        mem_results[cond] = cond_r2

    results['sequence_memory'] = mem_results

    # T687: COUPLED R² at d=1 > 0.1 (short-term memory)
    sm_d1_c = mem_results['COUPLED'].get(1, 0.0)
    tests['T687'] = {'desc': 'Seq memory R² at d=1 > 0.1 (COUPLED)',
                     'val': sm_d1_c, 'pass': sm_d1_c > 0.1}
    print(f"\n  T687: Seq mem d=1 R² = {sm_d1_c:.4f} {'PASS' if sm_d1_c > 0.1 else 'FAIL'}", flush=True)

    # T688: COUPLED > STATIC at d=1
    sm_d1_s = mem_results['STATIC'].get(1, 0.0)
    tests['T688'] = {'desc': 'Seq memory COUPLED > STATIC at d=1',
                     'val': sm_d1_c - sm_d1_s, 'pass': sm_d1_c > sm_d1_s}
    print(f"  T688: COUPLED({sm_d1_c:.3f}) > STATIC({sm_d1_s:.3f}) "
          f"{'PASS' if sm_d1_c > sm_d1_s else 'FAIL'}", flush=True)

    # T689: R² decays with delay (≥4/5 pairs monotonic)
    sm_coupled = [mem_results['COUPLED'].get(d, 0.0) for d in mem_delays]
    sm_mono = sum(1 for i in range(len(mem_delays)-1) if sm_coupled[i] >= sm_coupled[i+1])
    tests['T689'] = {'desc': 'Seq memory R² decays with delay (≥4/5 pairs)',
                     'val': sm_mono, 'pass': sm_mono >= 4}
    print(f"  T689: R² monotonic {sm_mono}/{len(mem_delays)-1} "
          f"{'PASS' if sm_mono >= 4 else 'FAIL'}", flush=True)

    # T690: R²(d=10) < R²(d=1)
    sm_d10_c = mem_results['COUPLED'].get(10, 0.0)
    tests['T690'] = {'desc': 'R²(d=10) < R²(d=1)',
                     'val': f"d10={sm_d10_c:.3f} < d1={sm_d1_c:.3f}",
                     'pass': sm_d10_c < sm_d1_c}
    print(f"  T690: R²(d=10)={sm_d10_c:.4f} < R²(d=1)={sm_d1_c:.4f} "
          f"{'PASS' if sm_d10_c < sm_d1_c else 'FAIL'}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 4: TEMPORAL INTEGRATION
    #   Predict running mean of input from reservoir state
    #   Tests whether slow membrane integrates over time
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 4: TEMPORAL INTEGRATION")
    print("  Predict 20-step running mean from reservoir state")
    print("  Tests slow membrane integration capacity (τ≈49.4ms ≈ 10 steps)")
    print("=" * 72, flush=True)

    ti_trials = 40
    ti_inputs, ti_targets_mean, ti_targets_var = generate_temporal_integration(
        ti_trials, N_STEPS, seed=77)

    ti_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Temporal Integration: {cond} ---", flush=True)
        all_features = []
        all_targets_m = []

        for trial in range(ti_trials):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, ti_inputs[trial], w_in, w_gpu, w_fb, vg_spread, mode=cond)

            warmup = 25
            for t in range(warmup, N_STEPS):
                all_features.append(vm[t])
                all_targets_m.append(ti_targets_mean[trial][t])

            if (trial + 1) % 10 == 0:
                print(f"    {cond} trial {trial+1}/{ti_trials}", flush=True)

        X = np.array(all_features)
        y_m = np.array(all_targets_m)

        n = len(y_m)
        idx = rng.permutation(n)
        split = int(0.7 * n)

        best_r2 = -999
        for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
            _, r2, _ = ridge_solve(X[idx[:split]], y_m[idx[:split]],
                                   X[idx[split:]], y_m[idx[split:]], alpha=alpha)
            if r2 > best_r2: best_r2 = r2

        ti_results[cond] = {'r2_mean': float(best_r2)}
        print(f"  {cond}: R²(running_mean)={best_r2:.4f}", flush=True)

    results['temporal_integration'] = ti_results

    # T691: COUPLED running mean R² > 0.3 (meaningful integration)
    ti_c = ti_results['COUPLED']['r2_mean']
    tests['T691'] = {'desc': 'Temporal integration R²(mean) > 0.3',
                     'val': ti_c, 'pass': ti_c > 0.3}
    print(f"\n  T691: R²(mean) = {ti_c:.4f} {'PASS' if ti_c > 0.3 else 'FAIL'}", flush=True)

    # T692: COUPLED > STATIC (integration advantage)
    ti_s = ti_results['STATIC']['r2_mean']
    tests['T692'] = {'desc': 'Integration COUPLED > STATIC',
                     'val': ti_c - ti_s, 'pass': ti_c > ti_s}
    print(f"  T692: COUPLED({ti_c:.3f}) > STATIC({ti_s:.3f}) "
          f"{'PASS' if ti_c > ti_s else 'FAIL'}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 5: MC-TASK CORRELATION
    #   Do per-delay MC values (z2225) predict per-delay seq memory R²?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 5: MC-TASK CORRELATION")
    print("  Correlate z2225 MC profile with sequence memory R²")
    print("=" * 72, flush=True)

    # z2225 MC per-delay (COUPLED)
    z2225_mc = [0.424, 0.085, 0.024, 0.013, 0.005, 0.0004]  # d=1,2,3,5,7,10
    sm_for_corr = [mem_results['COUPLED'].get(d, 0.0) for d in mem_delays]

    if np.std(z2225_mc) > 1e-10 and np.std(sm_for_corr) > 1e-10:
        mc_task_corr = np.corrcoef(z2225_mc, sm_for_corr)[0, 1]
    else:
        mc_task_corr = 0.0

    results['mc_task_correlation'] = {
        'mc_delays': z2225_mc,
        'seq_memory_r2': sm_for_corr,
        'pearson_r': float(mc_task_corr),
    }

    print(f"  MC profile:     {[f'{v:.3f}' for v in z2225_mc]}")
    print(f"  Seq mem R²:     {[f'{v:.3f}' for v in sm_for_corr]}")
    print(f"  Pearson r = {mc_task_corr:.3f}")

    # T693: MC-task correlation > 0.5
    tests['T693'] = {'desc': 'MC-task Pearson r > 0.5',
                     'val': mc_task_corr, 'pass': mc_task_corr > 0.5}
    print(f"\n  T693: r = {mc_task_corr:.3f} {'PASS' if mc_task_corr > 0.5 else 'FAIL'}", flush=True)

    # T694: Both MC and seq memory decay with delay
    mc_decay = all(z2225_mc[i] >= z2225_mc[i+1] for i in range(len(z2225_mc)-1))
    sm_decay = sum(1 for i in range(len(sm_for_corr)-1) if sm_for_corr[i] >= sm_for_corr[i+1])
    both_decay = mc_decay and (sm_decay >= 3)
    tests['T694'] = {'desc': 'Both MC and seq memory decay with delay',
                     'val': f"mc_mono={mc_decay}, sm_mono={sm_decay}/5",
                     'pass': both_decay}
    print(f"  T694: MC monotonic={mc_decay}, SM mono={sm_decay}/5 "
          f"{'PASS' if both_decay else 'FAIL'}", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 6: RICH FEATURES — repeat waveform with richer features
    #   Uses spike stats + cross-neuron correlations
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 6: RICH FEATURE WAVEFORM CLASSIFICATION")
    print("  Same 6-class waveforms, richer features (spike stats + correlations)")
    print("=" * 72, flush=True)

    # Re-use waveform data but with 'rich' features
    rich_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Rich Waveform: {cond} ---", flush=True)
        feats = []
        for trial in range(wf_trials):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, wf_inputs[trial], w_in, w_gpu, w_fb, vg_spread, mode=cond)
            feats.append(extract_trial_features(spk, vm, mode='rich'))
            if (trial + 1) % 30 == 0:
                print(f"    {cond} trial {trial+1}/{wf_trials}", flush=True)

        X = np.array(feats)
        cv = classify_cv(X, wf_labels, n_classes=6)
        rich_results[cond] = cv
        print(f"  {cond}: acc={cv['mean']:.3f} ± {cv['std']:.3f}", flush=True)

    results['waveform_rich'] = rich_results

    # T695: Rich features improve over temporal-only
    rc = rich_results['COUPLED']['mean']
    tests['T695'] = {'desc': 'Rich features COUPLED > temporal-only COUPLED',
                     'val': rc - wf_c, 'pass': rc > wf_c}
    print(f"\n  T695: Rich({rc:.3f}) > Temporal({wf_c:.3f}) "
          f"{'PASS' if rc > wf_c else 'FAIL'}", flush=True)

    # T696: Rich COUPLED > Rich STATIC
    rs = rich_results['STATIC']['mean']
    tests['T696'] = {'desc': 'Rich COUPLED > Rich STATIC',
                     'val': rc - rs, 'pass': rc > rs}
    print(f"  T696: Rich COUPLED({rc:.3f}) > STATIC({rs:.3f}) "
          f"{'PASS' if rc > rs else 'FAIL'}", flush=True)

    # T697: Rich COUPLED > 0.75 (strong multimodal classification)
    tests['T697'] = {'desc': 'Rich COUPLED > 0.75',
                     'val': rc, 'pass': rc > 0.75}
    print(f"  T697: Rich COUPLED = {rc:.3f} {'PASS' if rc > 0.75 else 'FAIL'}", flush=True)

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

    out = RESULTS / 'z2226_temporal_benchmark.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out}")

    fpga.close()
    print("Done.")


if __name__ == '__main__':
    main()
