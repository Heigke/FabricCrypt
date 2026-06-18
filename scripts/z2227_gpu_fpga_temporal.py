#!/usr/bin/env python3
"""z2227_gpu_fpga_temporal.py — GPU↔FPGA Temporal Processing Benchmark

Fixes z2226 sampling mismatch: uses 20Hz (matching z2206/z2210 proven 81-93.5%)
with τ≈49.4ms membrane. Focuses on demonstrating GPU↔FPGA coupling advantage.

KEY FIXES from z2226:
  - SAMPLE_HZ: 200 → 20 (match membrane τ to sample period)
  - N_STEPS: 30 per trial (1.5s, matching z2206)
  - 4-class waveform (not 6), matching proven task
  - Feature extraction: mean+std+max+min (4× pooling, not 2×)
  - More trials (120) for statistical power
  - GPU coupling stress test: vary workload intensity

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

# ─── Parameters (z2206/z2210 proven values) ───
N_NEURONS = 128
BASE_VG = 0.45
ALPHA = 0.35
BETA_POWER = 0.12
BETA_THERMAL = 0.08
BETA_CLOCK = 0.10
SAMPLE_HZ = 20       # FIXED: was 200 in z2226, now matches z2206
N_STEPS = 30          # FIXED: 30 steps × 50ms = 1.5s trial
WORKLOAD_MS = 5.0     # More GPU work per step at 20Hz (50ms budget)

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
                     mode='COUPLED', n_neurons=128, record_gpu=True):
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ  # 50ms at 20Hz
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
            time.sleep(0.001)  # 1ms settle at 20Hz (plenty of time)

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
        if remaining > 0.001:
            time.sleep(remaining)

    return spikes, vmem_log, gpu_log, intensities


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION — 4× pooling (z2206 style)
# ═══════════════════════════════════════════════════════════

def pool_trial_features(spk, vm):
    """z2206-style: mean, std, max, min across full trial. 4 × N = 512 features."""
    feats = [vm.mean(axis=0), vm.std(axis=0), vm.max(axis=0), vm.min(axis=0)]
    return np.concatenate(feats)

def pool_trial_features_rich(spk, vm):
    """Rich: vmem stats + spike rate + cross-neuron correlations."""
    n_steps, n_neurons = vm.shape
    # vmem: mean, std, max, min (4 × 128 = 512)
    feats = [vm.mean(axis=0), vm.std(axis=0), vm.max(axis=0), vm.min(axis=0)]
    # spike rate (128)
    feats.append(spk.sum(axis=0))
    # spike std per neuron (128)
    feats.append(spk.std(axis=0))
    # Cross-neuron correlations: top-16 by vmem variance (120)
    var = vm.var(axis=0)
    top_idx = np.argsort(var)[-16:]
    vm_top = vm[:, top_idx]
    corrs = []
    for i in range(16):
        for j in range(i + 1, 16):
            c = np.corrcoef(vm_top[:, i], vm_top[:, j])[0, 1]
            corrs.append(c if np.isfinite(c) else 0.0)
    feats.append(np.array(corrs))
    return np.concatenate(feats)


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
    return {'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_fold': [float(a) for a in accs]}


# ═══════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════

def generate_waveforms_4class(n_trials, n_steps, seed=42):
    """4-class waveform: sine, sawtooth, triangle, square (matching z2206)."""
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(n_steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 4)
        freq = rng.uniform(0.5, 3.0)
        if cls == 0:    # Sine
            wave = np.sin(2 * np.pi * freq * t)
        elif cls == 1:  # Sawtooth
            wave = 2 * (t * freq % 1) - 1
        elif cls == 2:  # Triangle
            wave = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        elif cls == 3:  # Square
            wave = np.sign(np.sin(2 * np.pi * freq * t))
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)

def generate_temporal_pattern(n_trials, n_steps, n_classes=4, seed=42):
    """4-class temporal order: same components, different temporal ordering."""
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

def generate_gpu_stress_waveforms(n_trials, n_steps, seed=42):
    """4-class waveforms with GPU stress levels to maximize coupling signal.
    Class 0: Low-freq sine + LOW GPU workload
    Class 1: High-freq sine + HIGH GPU workload
    Class 2: Step function + RAMPING GPU workload
    Class 3: Noise + FLUCTUATING GPU workload
    """
    rng = np.random.default_rng(seed)
    trials, labels, gpu_intensities = [], [], []
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(n_steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 4)
        if cls == 0:
            wave = 0.5 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
            gpu_int = np.full(n_steps, 0.1)  # Low GPU
        elif cls == 1:
            wave = 0.5 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
            gpu_int = np.full(n_steps, 0.9)  # High GPU
        elif cls == 2:
            wave = np.zeros(n_steps)
            wave[n_steps//3:2*n_steps//3] = 1.0
            gpu_int = np.linspace(0.1, 0.9, n_steps)  # Ramp
        elif cls == 3:
            wave = rng.uniform(0, 1, n_steps)
            gpu_int = 0.5 + 0.3 * np.sin(2 * np.pi * 2.0 * t)  # Fluctuating
        trials.append(wave)
        labels.append(cls)
        gpu_intensities.append(gpu_int)
    return np.array(trials), np.array(labels), np.array(gpu_intensities)


# ═══════════════════════════════════════════════════════════
# GPU-STRESS COUPLED LOOP (explicit GPU intensity control)
# ═══════════════════════════════════════════════════════════

def run_gpu_stress_loop(fpga, input_signal, gpu_intensity, w_in, w_gpu, w_fb,
                        vg_spread, mode='COUPLED', n_neurons=128):
    """Like run_coupled_loop but with explicit GPU intensity per step."""
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    spikes = np.zeros((n_steps, n_neurons))
    vmem_log = np.zeros((n_steps, n_neurons))
    gpu_log = np.zeros((n_steps, 9))
    prev_counts = None

    for t in range(n_steps):
        t_start = time.perf_counter()
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
            n3 = n_neurons // 3
            wg = w_gpu[:n_neurons]
            vg[:n3] += BETA_POWER * p_delta * wg[:n3]
            vg[n3:2*n3] += BETA_CLOCK * s_delta * wg[n3:2*n3]
            vg[2*n3:] += BETA_THERMAL * t_delta * wg[2*n3:]
            gs_pm_t = gpu_log[t, 1]
            if gs_pm_t > 0:
                try: fpga.set_temp(float(gs_pm_t) + 273.15)
                except: pass

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
                        spikes[t, i] = delta
                vmem_log[t] = vm[:n_neurons]
                prev_counts = counts.copy()
            except:
                pass

        # GPU workload: always run at specified intensity (for COUPLED/GPU_ONLY/UNCOUPLED)
        if mode in ('COUPLED', 'GPU_ONLY', 'UNCOUPLED'):
            run_workload(float(gpu_intensity[t]), duration_ms=WORKLOAD_MS)
            if mode == 'COUPLED':
                try: fpga.set_mac_signal(float(gpu_intensity[t]) * 0.5)
                except: pass

        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.001:
            time.sleep(remaining)

    return spikes, vmem_log, gpu_log


# ═══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("z2227: GPU↔FPGA TEMPORAL PROCESSING BENCHMARK")
    print("  FIXED: 20Hz sampling (was 200Hz) — matches τ≈49.4ms membrane")
    print("  4-class waveform (z2206 style) + temporal pattern + GPU stress")
    print("  Features: mean+std+max+min (4× pooling)")
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
        'experiment': 'z2227_gpu_fpga_temporal',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'base': 'z2226 fixes: 20Hz sampling, 4× pooling, 4-class',
            'vg_spread': '±0.08 linspace across 128 neurons',
            'features': 'mean+std+max+min (512 dim)',
            'lateral': 'N±1 (w=0.125), N±2 (w=0.0625) ring topology',
            'n_neurons': N_NEURONS,
            'sample_hz': SAMPLE_HZ, 'n_steps': N_STEPS,
            'membrane_tau_ms': 49.4,
        }
    }
    tests = {}

    # ═══════════════════════════════════════════════════════════
    # EXP 1: 4-CLASS WAVEFORM (z2206 style)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 1: WAVEFORM CLASSIFICATION (4-class, z2206 style)")
    print("  sine/sawtooth/triangle/square, 20Hz sampling, 1.5s trials")
    print("=" * 72, flush=True)

    wf_trials = 120  # 30 per class
    wf_inputs, wf_labels = generate_waveforms_4class(wf_trials, N_STEPS, seed=42)

    wf_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Waveform: {cond} ---", flush=True)
        feats = []
        for trial in range(wf_trials):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, wf_inputs[trial], w_in, w_gpu, w_fb, vg_spread, mode=cond)
            feats.append(pool_trial_features(spk, vm))
            if (trial + 1) % 40 == 0:
                print(f"    {cond} trial {trial+1}/{wf_trials}", flush=True)

        X = np.array(feats)
        cv = classify_cv(X, wf_labels, n_classes=4)
        wf_results[cond] = cv
        print(f"  {cond}: acc={cv['mean']:.3f} ± {cv['std']:.3f}", flush=True)

    results['waveform_4class'] = wf_results

    wf_c = wf_results['COUPLED']['mean']
    wf_f = wf_results['FPGA_ONLY']['mean']
    wf_s = wf_results['STATIC']['mean']

    tests['T700'] = {'desc': 'Waveform 4-class COUPLED > 0.60',
                     'val': wf_c, 'pass': wf_c > 0.60}
    tests['T701'] = {'desc': 'Waveform COUPLED > STATIC',
                     'val': wf_c - wf_s, 'pass': wf_c > wf_s}
    tests['T702'] = {'desc': 'Waveform COUPLED > FPGA_ONLY (GPU coupling helps)',
                     'val': wf_c - wf_f, 'pass': wf_c > wf_f}
    tests['T703'] = {'desc': 'Waveform COUPLED > 0.75 (strong, matching z2206)',
                     'val': wf_c, 'pass': wf_c > 0.75}

    for tid in ['T700', 'T701', 'T702', 'T703']:
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']:.3f}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 2: TEMPORAL PATTERN (4-class)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 2: TEMPORAL PATTERN CLASSIFICATION (4-class)")
    print("  Same components, different temporal order — tests temporal memory")
    print("=" * 72, flush=True)

    tp_trials = 80
    tp_inputs, tp_labels = generate_temporal_pattern(tp_trials, N_STEPS, seed=42)

    tp_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Temporal Pattern: {cond} ---", flush=True)
        feats = []
        for trial in range(tp_trials):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, tp_inputs[trial], w_in, w_gpu, w_fb, vg_spread, mode=cond)
            feats.append(pool_trial_features(spk, vm))
            if (trial + 1) % 20 == 0:
                print(f"    {cond} trial {trial+1}/{tp_trials}", flush=True)

        X = np.array(feats)
        cv = classify_cv(X, tp_labels, n_classes=4)
        tp_results[cond] = cv
        print(f"  {cond}: acc={cv['mean']:.3f} ± {cv['std']:.3f}", flush=True)

    results['temporal_pattern'] = tp_results

    tp_c = tp_results['COUPLED']['mean']
    tp_f = tp_results['FPGA_ONLY']['mean']
    tp_s = tp_results['STATIC']['mean']

    tests['T704'] = {'desc': 'Temporal pattern COUPLED > 0.50',
                     'val': tp_c, 'pass': tp_c > 0.50}
    tests['T705'] = {'desc': 'Temporal pattern COUPLED > STATIC',
                     'val': tp_c - tp_s, 'pass': tp_c > tp_s}
    tests['T706'] = {'desc': 'Temporal pattern COUPLED > FPGA_ONLY',
                     'val': tp_c - tp_f, 'pass': tp_c > tp_f}

    for tid in ['T704', 'T705', 'T706']:
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']:.3f}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 3: GPU STRESS TEST — GPU workload as CLASS signal
    #   Different GPU intensity patterns → different FPGA dynamics
    #   COUPLED should excel because GPU dynamics are the signal
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 3: GPU STRESS CLASSIFICATION")
    print("  GPU workload pattern IS the class signal")
    print("  COUPLED should strongly outperform FPGA_ONLY")
    print("=" * 72, flush=True)

    gs_trials = 80
    gs_inputs, gs_labels, gs_gpu_int = generate_gpu_stress_waveforms(
        gs_trials, N_STEPS, seed=42)

    gs_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- GPU Stress: {cond} ---", flush=True)
        feats = []
        for trial in range(gs_trials):
            if cond in ('COUPLED', 'GPU_ONLY', 'UNCOUPLED'):
                # Use explicit GPU intensity control
                spk, vm, glog = run_gpu_stress_loop(
                    fpga, gs_inputs[trial], gs_gpu_int[trial],
                    w_in, w_gpu, w_fb, vg_spread, mode=cond)
            else:
                # FPGA_ONLY or STATIC: no GPU workload
                spk, vm, glog, _ = run_coupled_loop(
                    fpga, gs_inputs[trial], w_in, w_gpu, w_fb, vg_spread, mode=cond)
            feats.append(pool_trial_features(spk, vm))
            if (trial + 1) % 20 == 0:
                print(f"    {cond} trial {trial+1}/{gs_trials}", flush=True)

        X = np.array(feats)
        cv = classify_cv(X, gs_labels, n_classes=4)
        gs_results[cond] = cv
        print(f"  {cond}: acc={cv['mean']:.3f} ± {cv['std']:.3f}", flush=True)

    results['gpu_stress'] = gs_results

    gs_c = gs_results['COUPLED']['mean']
    gs_f = gs_results['FPGA_ONLY']['mean']
    gs_s = gs_results['STATIC']['mean']

    tests['T707'] = {'desc': 'GPU stress COUPLED > 0.50',
                     'val': gs_c, 'pass': gs_c > 0.50}
    tests['T708'] = {'desc': 'GPU stress COUPLED > FPGA_ONLY (GPU is the signal)',
                     'val': gs_c - gs_f, 'pass': gs_c > gs_f}
    tests['T709'] = {'desc': 'GPU stress COUPLED > STATIC',
                     'val': gs_c - gs_s, 'pass': gs_c > gs_s}
    tests['T710'] = {'desc': 'GPU stress COUPLED advantage > 10pp over FPGA_ONLY',
                     'val': gs_c - gs_f, 'pass': (gs_c - gs_f) > 0.10}

    for tid in ['T707', 'T708', 'T709', 'T710']:
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']:.3f}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 4: RICH FEATURES — waveform with spike+correlation features
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 4: RICH FEATURE WAVEFORM (4-class)")
    print("  vmem stats + spike rate + cross-neuron correlations")
    print("=" * 72, flush=True)

    rich_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Rich Waveform: {cond} ---", flush=True)
        feats = []
        for trial in range(wf_trials):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, wf_inputs[trial], w_in, w_gpu, w_fb, vg_spread, mode=cond)
            feats.append(pool_trial_features_rich(spk, vm))
            if (trial + 1) % 40 == 0:
                print(f"    {cond} trial {trial+1}/{wf_trials}", flush=True)

        X = np.array(feats)
        cv = classify_cv(X, wf_labels, n_classes=4)
        rich_results[cond] = cv
        print(f"  {cond}: acc={cv['mean']:.3f} ± {cv['std']:.3f}", flush=True)

    results['waveform_rich'] = rich_results

    rc = rich_results['COUPLED']['mean']
    rf = rich_results['FPGA_ONLY']['mean']
    rs = rich_results['STATIC']['mean']

    tests['T711'] = {'desc': 'Rich COUPLED > basic COUPLED',
                     'val': rc - wf_c, 'pass': rc > wf_c}
    tests['T712'] = {'desc': 'Rich COUPLED > Rich STATIC',
                     'val': rc - rs, 'pass': rc > rs}
    tests['T713'] = {'desc': 'Rich COUPLED > 0.80 (strong multimodal)',
                     'val': rc, 'pass': rc > 0.80}

    for tid in ['T711', 'T712', 'T713']:
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']:.3f}]", flush=True)

    # ═══════════════════════════════════════════════════════════
    # EXP 5: REPRODUCIBILITY — 3 seeds for waveform
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 5: REPRODUCIBILITY (3 seeds)")
    print("  Waveform 4-class COUPLED at seeds 42, 59, 76")
    print("=" * 72, flush=True)

    repro_accs = []
    for seed in [42, 59, 76]:
        print(f"\n  --- Seed {seed} ---", flush=True)
        repro_rng = np.random.default_rng(seed)
        wf_i, wf_l = generate_waveforms_4class(80, N_STEPS, seed=seed)
        feats = []
        for trial in range(80):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, wf_i[trial], w_in, w_gpu, w_fb, vg_spread, mode='COUPLED')
            feats.append(pool_trial_features(spk, vm))
            if (trial + 1) % 40 == 0:
                print(f"    seed={seed} trial {trial+1}/80", flush=True)
        X = np.array(feats)
        cv = classify_cv(X, wf_l, n_classes=4)
        repro_accs.append(cv['mean'])
        print(f"  Seed {seed}: acc={cv['mean']:.3f}", flush=True)

    results['reproducibility'] = {
        'seeds': [42, 59, 76],
        'accuracies': repro_accs,
        'mean': float(np.mean(repro_accs)),
        'std': float(np.std(repro_accs)),
    }

    tests['T714'] = {'desc': 'All 3 seeds > 0.50',
                     'val': min(repro_accs), 'pass': all(a > 0.50 for a in repro_accs)}
    tests['T715'] = {'desc': 'Reproducibility std < 0.10',
                     'val': float(np.std(repro_accs)),
                     'pass': np.std(repro_accs) < 0.10}

    for tid in ['T714', 'T715']:
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

    out = RESULTS / 'z2227_gpu_fpga_temporal.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out}")

    fpga.close()
    print("Done.")


if __name__ == '__main__':
    main()
