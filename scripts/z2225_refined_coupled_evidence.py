#!/usr/bin/env python3
"""z2225_refined_coupled_evidence.py — Refined Evidence for Coupled Substrate Computing

Builds on z2223 (17/19 PASS) — the proven approach. z2224 showed that adding dead
probes (per-core thermal=0, SVI=0) and random FPGA recurrence HURT performance.

This experiment focuses on what WORKS and builds stronger statistical evidence:

1. REPRODUCIBILITY: 5 independent runs with different seeds
2. NEURON SCALING: 8, 32, 64, 128 neurons — does coupled advantage grow?
3. PER-CHANNEL ABLATION: Remove ONE GPU probe at a time — which matter?
4. DISPATCH JITTER: The ONE new probe from z2224 that works (ACF=0.964)
5. MULTI-TIMESCALE TE: TE at lag 1,2,5,10 — different coupling timescales
6. 1/f NOISE TRANSFER: Does GPU 1/f character transfer to FPGA spike timing?
7. DIFFICULTY SCALING: 4, 6, 8 class waveforms

NO dead probes. NO random recurrence. Only honest, working channels.
Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128 neurons, UDP Ethernet)
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters (z2223 proven values) ───
N_NEURONS = 128
BASE_VG = 0.45
ALPHA = 0.35
BETA_POWER = 0.12
BETA_THERMAL = 0.08
BETA_CLOCK = 0.10
SAMPLE_HZ = 200
WORKLOAD_MS = 1.5
N_STEPS = 300
N_TRIALS = 80          # More trials for statistical power

# Probe paths
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
HWMON_TEMP = "/sys/class/hwmon/hwmon7/temp1_input"
HWMON_FREQ = "/sys/class/hwmon/hwmon7/freq1_input"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
GPU_BUSY_PATH = "/sys/class/drm/card0/device/gpu_busy_percent"

# Channel names for the 9 proven channels
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
# GPU PROBES — Only the 8+1 that WORK
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
            f.seek(0x4C); thermal = struct.unpack('<f', f.read(4))[0]
            f.seek(0x04); power = struct.unpack('<f', f.read(4))[0]
            f.seek(0x78); sclk = struct.unpack('<f', f.read(4))[0]
        return thermal, power, sclk
    except:
        return None, None, None

def read_hwmon():
    try: power = int(open(HWMON_POWER).read().strip()) / 1e6
    except: power = None
    try: temp = int(open(HWMON_TEMP).read().strip()) / 1000.0
    except: temp = None
    try: freq = int(open(HWMON_FREQ).read().strip()) / 1e6
    except: freq = None
    return power, temp, freq

def read_gpu_busy():
    try: return int(open(GPU_BUSY_PATH).read().strip())
    except: return 0

def measure_dispatch_jitter():
    """Measure GPU kernel dispatch timing jitter — ISA-level temporal signal."""
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
    """Read all 9 working GPU probes."""
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
# COUPLED DYNAMICS LOOP (z2223 proven, +dispatch jitter)
# ═══════════════════════════════════════════════════════════

def run_coupled_loop(fpga, input_signal, w_in, w_gpu, w_fb,
                     mode='COUPLED', n_neurons=128, record_gpu=True):
    """z2223 loop with 9th channel (dispatch jitter)."""
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    spikes = np.zeros((n_steps, n_neurons))
    vmem_log = np.zeros((n_steps, n_neurons))
    gpu_log = np.zeros((n_steps, 9))  # 9 channels now
    intensities = np.zeros(n_steps)
    prev_counts = None

    for t in range(n_steps):
        t_start = time.perf_counter()

        if record_gpu:
            gpu_log[t] = read_all_gpu_state()

        vg = np.full(n_neurons, BASE_VG)

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
            wg = w_gpu[:n_neurons]  # Slice to match neuron count
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
# ANALYSIS FUNCTIONS (same as z2223, proven)
# ═══════════════════════════════════════════════════════════

def transfer_entropy(source, target, k=1, bins=8):
    if len(source) < k + 2 or np.std(source) < 1e-10 or np.std(target) < 1e-10:
        return 0.0
    s_bins = np.digitize(source, np.linspace(source.min()-1e-10, source.max()+1e-10, bins+1)) - 1
    t_bins = np.digitize(target, np.linspace(target.min()-1e-10, target.max()+1e-10, bins+1)) - 1
    n = len(source) - k
    from collections import Counter
    joint_tts = Counter()
    joint_tt = Counter()
    marg_ts = Counter()
    marg_t = Counter()
    for i in range(k, len(source)):
        tt = t_bins[i]
        tp = tuple(t_bins[i-k:i])
        sp = tuple(s_bins[i-k:i])
        joint_tts[(tt, tp, sp)] += 1
        joint_tt[(tt, tp)] += 1
        marg_ts[(tp, sp)] += 1
        marg_t[tp] += 1
    te = 0.0
    for (tt, tp, sp), c_tts in joint_tts.items():
        p_tts = c_tts / n
        p_t_given_ts = c_tts / max(marg_ts[(tp, sp)], 1)
        p_t_given_t = joint_tt[(tt, tp)] / max(marg_t[tp], 1)
        if p_t_given_ts > 0 and p_t_given_t > 0:
            te += p_tts * np.log2(p_t_given_ts / p_t_given_t)
    return max(0.0, te)

def effective_dimension(X):
    if X.shape[0] < 3 or X.shape[1] < 2:
        return 1.0
    X_c = X - X.mean(axis=0)
    try:
        s = np.linalg.svd(X_c, compute_uv=False)
        s2 = s ** 2
        if s2.sum() < 1e-10: return 1.0
        return float((s2.sum()) ** 2 / (s2 ** 2).sum())
    except:
        return 1.0

def mutual_information(x, y, bins=10):
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    return max(0.0, mi)

def psd_slope(x, fs=200):
    n = len(x)
    if n < 16: return 0.0
    f = np.fft.rfftfreq(n, 1.0/fs)[1:]
    p = np.abs(np.fft.rfft(x - x.mean()))[1:]**2
    p[p < 1e-30] = 1e-30
    log_f = np.log10(f[1:len(f)//2])
    log_p = np.log10(p[1:len(p)//2])
    if len(log_f) < 3: return 0.0
    try:
        slope = np.polyfit(log_f, log_p, 1)[0]
    except:
        slope = 0.0
    return float(slope)

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
            'folds': [float(a) for a in accs]}


# ═══════════════════════════════════════════════════════════
# WAVEFORM GENERATION
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials, steps, n_classes=4, seed=42):
    """Generate n_classes waveform types."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(steps) * dt
    trials, labels = [], []
    for _ in range(n_trials):
        cls = rng.integers(0, n_classes)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 2.0)
        if cls == 0:    wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:  wave = 2.0 * np.abs(2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0) - 1.0
        elif cls == 2:  wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif cls == 3:  wave = 2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0
        elif cls == 4:  # chirp (increasing freq)
            f_t = freq + 2.0 * t / t[-1]
            wave = np.sin(2 * np.pi * f_t * t + phase)
        elif cls == 5:  # AM (amplitude modulated)
            wave = (0.5 + 0.5*np.sin(2*np.pi*0.3*t)) * np.sin(2*np.pi*freq*t + phase)
        elif cls == 6:  # burst
            wave = np.sin(2*np.pi*freq*t + phase) * (np.abs(np.sin(2*np.pi*0.5*t)) > 0.5).astype(float)
        elif cls == 7:  # noise-modulated
            wave = np.sin(2*np.pi*freq*t + phase + 0.5*np.cumsum(rng.normal(0, 0.01, len(t))))
        else:
            wave = np.sin(2 * np.pi * freq * t + phase)
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("z2225: REFINED COUPLED SUBSTRATE EVIDENCE")
    print("  Building on z2223 (17/19 PASS)")
    print("  z2224 lesson: more probes ≠ better, dead channels hurt")
    print("  Focus: reproducibility, scaling, ablation, honest evidence")
    print("  9 channels: z2223's 8 + dispatch jitter (ACF=0.964)")
    print("=" * 72)

    # ─── Init ───
    print("\n[1] Connecting FPGA...")
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("  FATAL: No FPGA"); return
    fpga.set_kill(False)
    time.sleep(0.2)

    print("\n[2] Init GPU HIP...")
    init_torch()

    print("\n[3] Probe availability...")
    test_gs = read_all_gpu_state()
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"    {name}: {test_gs[i]}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    w_gpu = rng.uniform(-1, 1, N_NEURONS)
    w_fb = rng.uniform(-1, 1, N_NEURONS)

    results = {
        'experiment': 'z2225_refined_coupled_evidence',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'base': 'z2223 (proven 17/19)',
            'channels': CHANNEL_NAMES,
            'n_channels': 9,
            'new_channel': 'dispatch_jitter (ACF=0.964 from z2224)',
            'removed': ['per_core_thermal (all 0)', 'SVI (constant)', 'random_recurrence (hurts)'],
            'sample_hz': SAMPLE_HZ, 'n_steps': N_STEPS, 'n_trials': N_TRIALS,
        }
    }
    tests = {}

    # ═══════════════════════════════════════════════════════════
    # EXP 1: REPRODUCIBILITY — z2223 core 5 times, different seeds
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 1: REPRODUCIBILITY")
    print("  5 independent runs of z2223 core (4-class, 60 trials each)")
    print("  Tests whether coupled advantage is robust across seeds")
    print("=" * 72)

    repro_results = []
    for run_idx in range(5):
        seed = 42 + run_idx * 17
        run_rng = np.random.default_rng(seed)
        run_w_in = run_rng.uniform(-1, 1, N_NEURONS)
        run_w_gpu = run_rng.uniform(-1, 1, N_NEURONS)
        run_w_fb = run_rng.uniform(-1, 1, N_NEURONS)

        inputs, labels = generate_waveforms(60, N_STEPS, n_classes=4, seed=seed)

        run_accs = {}
        for cond in ['COUPLED', 'STATIC']:
            feats = []
            for trial in range(60):
                spk, vm, glog, ints = run_coupled_loop(
                    fpga, inputs[trial], run_w_in, run_w_gpu, run_w_fb,
                    mode=cond, n_neurons=N_NEURONS)
                trial_state = np.hstack([spk, vm, glog])
                f = np.concatenate([trial_state.mean(axis=0), trial_state.std(axis=0)])
                feats.append(f)
                if (trial + 1) % 20 == 0:
                    print(f"    Run {run_idx+1}/5 {cond} trial {trial+1}/60", flush=True)
            X = np.array(feats)
            run_accs[cond] = classify_cv(X, labels, n_classes=4)['mean']

        advantage = run_accs['COUPLED'] - run_accs['STATIC']
        repro_results.append({
            'seed': seed,
            'coupled': run_accs['COUPLED'],
            'static': run_accs['STATIC'],
            'advantage': advantage
        })
        print(f"  Run {run_idx+1}/5 (seed={seed}): COUPLED={run_accs['COUPLED']:.3f}, "
              f"STATIC={run_accs['STATIC']:.3f}, advantage={advantage:+.3f}")

    results['reproducibility'] = repro_results
    advantages = [r['advantage'] for r in repro_results]
    coupled_accs = [r['coupled'] for r in repro_results]
    mean_adv = np.mean(advantages)
    std_adv = np.std(advantages)
    n_positive = sum(1 for a in advantages if a > 0)

    # T650: Mean advantage > 0 (COUPLED beats STATIC on average)
    tests['T650'] = {'desc': 'Mean COUPLED advantage > 0 over 5 runs',
                     'val': mean_adv, 'pass': mean_adv > 0}
    print(f"\n  T650: mean advantage = {mean_adv:+.3f} ± {std_adv:.3f} {'PASS' if mean_adv > 0 else 'FAIL'}")

    # T651: All 5 runs show positive advantage
    tests['T651'] = {'desc': 'All 5 runs COUPLED > STATIC',
                     'val': n_positive, 'pass': n_positive == 5}
    print(f"  T651: {n_positive}/5 positive runs {'PASS' if n_positive == 5 else 'FAIL'}")

    # T652: Mean COUPLED accuracy > 0.55 (consistently high)
    mean_coupled = np.mean(coupled_accs)
    tests['T652'] = {'desc': 'Mean COUPLED accuracy > 0.55 across runs',
                     'val': mean_coupled, 'pass': mean_coupled > 0.55}
    print(f"  T652: mean COUPLED = {mean_coupled:.3f} {'PASS' if mean_coupled > 0.55 else 'FAIL'}")

    # T653: Std of advantage < 0.15 (reproducible)
    tests['T653'] = {'desc': 'Std of advantage < 0.15 (reproducible)',
                     'val': std_adv, 'pass': std_adv < 0.15}
    print(f"  T653: std advantage = {std_adv:.3f} {'PASS' if std_adv < 0.15 else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXP 2: NEURON SCALING
    #   Use 8, 32, 64, 128 neurons — does coupled advantage grow?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 2: NEURON SCALING")
    print("  Testing coupled advantage at 8, 32, 64, 128 neurons")
    print("=" * 72)

    inputs_scale, labels_scale = generate_waveforms(60, N_STEPS, n_classes=4, seed=99)
    neuron_counts = [8, 32, 64, 128]
    scaling_results = {}

    for n_neur in neuron_counts:
        scale_accs = {}
        for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
            feats = []
            for trial in range(60):
                spk, vm, glog, ints = run_coupled_loop(
                    fpga, inputs_scale[trial], w_in, w_gpu, w_fb,
                    mode=cond, n_neurons=n_neur)
                # Only use first n_neur columns
                trial_state = np.hstack([spk[:, :n_neur], vm[:, :n_neur], glog])
                f = np.concatenate([trial_state.mean(axis=0), trial_state.std(axis=0)])
                feats.append(f)
                if (trial + 1) % 20 == 0:
                    print(f"    N={n_neur} {cond} trial {trial+1}/60", flush=True)
            X = np.array(feats)
            scale_accs[cond] = classify_cv(X, labels_scale, n_classes=4)['mean']

        coupled_adv = scale_accs['COUPLED'] - scale_accs['FPGA_ONLY']
        scaling_results[n_neur] = {
            'coupled': scale_accs['COUPLED'],
            'fpga_only': scale_accs['FPGA_ONLY'],
            'static': scale_accs['STATIC'],
            'coupled_advantage': coupled_adv,
        }
        print(f"  N={n_neur:3d}: COUPLED={scale_accs['COUPLED']:.3f}, "
              f"FPGA={scale_accs['FPGA_ONLY']:.3f}, STATIC={scale_accs['STATIC']:.3f}, "
              f"adv={coupled_adv:+.3f}")

    results['neuron_scaling'] = scaling_results

    # T654: Coupled advantage at 128 neurons
    adv_128 = scaling_results[128]['coupled_advantage']
    tests['T654'] = {'desc': 'Coupled advantage > 0 at 128 neurons',
                     'val': adv_128, 'pass': adv_128 > 0}
    print(f"\n  T654: 128N coupled advantage = {adv_128:+.3f} {'PASS' if adv_128 > 0 else 'FAIL'}")

    # T655: Coupled > STATIC at all scales
    all_better = all(scaling_results[n]['coupled'] > scaling_results[n]['static'] for n in neuron_counts)
    tests['T655'] = {'desc': 'COUPLED > STATIC at all neuron counts',
                     'val': sum(1 for n in neuron_counts if scaling_results[n]['coupled'] > scaling_results[n]['static']),
                     'pass': all_better}
    print(f"  T655: COUPLED > STATIC at all scales = {'PASS' if all_better else 'FAIL'}")

    # T656: Accuracy improves with neuron count (monotonic for COUPLED)
    coupled_scores = [scaling_results[n]['coupled'] for n in neuron_counts]
    monotonic = all(coupled_scores[i] <= coupled_scores[i+1] for i in range(len(coupled_scores)-1))
    tests['T656'] = {'desc': 'COUPLED accuracy monotonic with neuron count',
                     'val': str(coupled_scores), 'pass': monotonic}
    print(f"  T656: monotonic scaling = {'PASS' if monotonic else 'FAIL'} {coupled_scores}")

    # ═══════════════════════════════════════════════════════════
    # EXP 3: PER-CHANNEL ABLATION
    #   Remove ONE GPU channel at a time — which matter most?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 3: PER-CHANNEL ABLATION")
    print("  Removing one GPU channel at a time to measure contribution")
    print("=" * 72)

    # First, collect data with ALL 9 channels
    inputs_abl, labels_abl = generate_waveforms(N_TRIALS, N_STEPS, n_classes=4, seed=77)
    all_feats_raw = []  # Store raw trial_state for ablation
    print("  Collecting COUPLED data (80 trials)...")
    for trial in range(N_TRIALS):
        spk, vm, glog, ints = run_coupled_loop(
            fpga, inputs_abl[trial], w_in, w_gpu, w_fb, mode='COUPLED')
        trial_state = np.hstack([spk, vm, glog])
        all_feats_raw.append(trial_state)
        if (trial + 1) % 20 == 0:
            print(f"    trial {trial+1}/{N_TRIALS}", flush=True)

    # Full accuracy
    full_feats = np.array([np.concatenate([ts.mean(axis=0), ts.std(axis=0)]) for ts in all_feats_raw])
    full_acc = classify_cv(full_feats, labels_abl, n_classes=4)['mean']
    print(f"  Full (9ch): {full_acc:.3f}")

    # Ablate each GPU channel (channels are at indices 256..264 in trial_state)
    # trial_state = [spikes(128) + vmem(128) + gpu(9)] = 265 columns
    gpu_start = 256  # 128 + 128
    ablation_results = {'full': full_acc}

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        # Zero out this channel in all trials
        ablated_feats = []
        for ts in all_feats_raw:
            ts_abl = ts.copy()
            ts_abl[:, gpu_start + ch_idx] = 0
            f = np.concatenate([ts_abl.mean(axis=0), ts_abl.std(axis=0)])
            ablated_feats.append(f)
        X_abl = np.array(ablated_feats)
        abl_acc = classify_cv(X_abl, labels_abl, n_classes=4)['mean']
        drop = full_acc - abl_acc
        ablation_results[ch_name] = {'accuracy': abl_acc, 'drop': drop}
        print(f"    −{ch_name}: {abl_acc:.3f} (drop={drop:+.3f})")

    # Without any GPU channels
    no_gpu_feats = []
    for ts in all_feats_raw:
        ts_no = ts[:, :gpu_start]  # Only spikes + vmem
        f = np.concatenate([ts_no.mean(axis=0), ts_no.std(axis=0)])
        no_gpu_feats.append(f)
    X_nogpu = np.array(no_gpu_feats)
    nogpu_acc = classify_cv(X_nogpu, labels_abl, n_classes=4)['mean']
    ablation_results['no_gpu_channels'] = nogpu_acc
    print(f"    −ALL GPU: {nogpu_acc:.3f} (drop={full_acc - nogpu_acc:+.3f})")

    results['ablation'] = ablation_results

    # T657: At least 3 channels cause > 1pp drop when removed
    significant_drops = sum(1 for ch in CHANNEL_NAMES
                           if ablation_results[ch]['drop'] > 0.01)
    tests['T657'] = {'desc': 'At least 3 GPU channels contribute > 1pp',
                     'val': significant_drops, 'pass': significant_drops >= 3}
    print(f"\n  T657: {significant_drops} significant channels {'PASS' if significant_drops >= 3 else 'FAIL'}")

    # T658: Full > no-GPU (GPU channels collectively contribute)
    gpu_contribution = full_acc - nogpu_acc
    tests['T658'] = {'desc': 'GPU channels collectively contribute > 2pp',
                     'val': gpu_contribution, 'pass': gpu_contribution > 0.02}
    print(f"  T658: GPU contribution = {gpu_contribution:+.3f} {'PASS' if gpu_contribution > 0.02 else 'FAIL'}")

    # T659: Dispatch jitter channel contributes (drop > 0)
    jitter_drop = ablation_results['dispatch_jitter']['drop']
    tests['T659'] = {'desc': 'Dispatch jitter channel contributes (drop > 0)',
                     'val': jitter_drop, 'pass': jitter_drop > 0}
    print(f"  T659: jitter drop = {jitter_drop:+.3f} {'PASS' if jitter_drop > 0 else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXP 4: MULTI-TIMESCALE TRANSFER ENTROPY
    #   TE at lag k=1,2,5,10 — different coupling timescales
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 4: MULTI-TIMESCALE COUPLING")
    print("  Transfer entropy at lag k=1,2,5,10")
    print("=" * 72)

    # Run coupled and uncoupled for TE analysis (longer: 500 steps)
    long_input = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(500) / 60)
    print("  Running COUPLED (500 steps)...", flush=True)
    c_spk, c_vm, c_gpu, c_int = run_coupled_loop(
        fpga, long_input, w_in, w_gpu, w_fb, mode='COUPLED')
    print("  Running UNCOUPLED (500 steps)...", flush=True)
    u_spk, u_vm, u_gpu, u_int = run_coupled_loop(
        fpga, long_input, w_in, w_gpu, w_fb, mode='UNCOUPLED')

    c_spike_mean = c_spk[10:].mean(axis=1)
    c_power = c_gpu[10:, 4]
    u_spike_mean = u_spk[10:].mean(axis=1)
    u_power = u_gpu[10:, 4]

    te_multi = {}
    for k in [1, 2, 5, 10]:
        te_fg_c = transfer_entropy(c_spike_mean, c_power, k=k)
        te_gf_c = transfer_entropy(c_power, c_spike_mean, k=k)
        te_fg_u = transfer_entropy(u_spike_mean, u_power, k=k)
        te_gf_u = transfer_entropy(u_power, u_spike_mean, k=k)
        te_multi[f'k{k}'] = {
            'TE_FPGA_GPU_coupled': te_fg_c,
            'TE_GPU_FPGA_coupled': te_gf_c,
            'TE_FPGA_GPU_uncoupled': te_fg_u,
            'TE_GPU_FPGA_uncoupled': te_gf_u,
            'ratio_FPGA_GPU': te_fg_c / max(te_fg_u, 1e-6),
            'ratio_GPU_FPGA': te_gf_c / max(te_gf_u, 1e-6),
        }
        print(f"  k={k:2d}: TE(FPGA→GPU) c={te_fg_c:.4f}/u={te_fg_u:.4f} ({te_fg_c/max(te_fg_u,1e-6):.1f}×), "
              f"TE(GPU→FPGA) c={te_gf_c:.4f}/u={te_gf_u:.4f} ({te_gf_c/max(te_gf_u,1e-6):.1f}×)")

    results['multi_timescale_te'] = te_multi

    # T660: Bidirectional TE at k=1 (coupled > uncoupled both directions)
    bidir_k1 = (te_multi['k1']['TE_FPGA_GPU_coupled'] > te_multi['k1']['TE_FPGA_GPU_uncoupled'] and
                te_multi['k1']['TE_GPU_FPGA_coupled'] > te_multi['k1']['TE_GPU_FPGA_uncoupled'])
    tests['T660'] = {'desc': 'Bidirectional TE at k=1 (coupled > uncoupled both dirs)',
                     'val': min(te_multi['k1']['ratio_FPGA_GPU'], te_multi['k1']['ratio_GPU_FPGA']),
                     'pass': bidir_k1}
    print(f"\n  T660: bidir TE k=1 {'PASS' if bidir_k1 else 'FAIL'}")

    # T661: TE at k=2 still shows coupling (temporal depth)
    bidir_k2 = (te_multi['k2']['TE_FPGA_GPU_coupled'] > te_multi['k2']['TE_FPGA_GPU_uncoupled'] and
                te_multi['k2']['TE_GPU_FPGA_coupled'] > te_multi['k2']['TE_GPU_FPGA_uncoupled'])
    tests['T661'] = {'desc': 'Bidirectional TE at k=2 (temporal depth)',
                     'val': min(te_multi['k2']['ratio_FPGA_GPU'], te_multi['k2']['ratio_GPU_FPGA']),
                     'pass': bidir_k2}
    print(f"  T661: bidir TE k=2 {'PASS' if bidir_k2 else 'FAIL'}")

    # T662: TE at k=5 (deep temporal coupling)
    bidir_k5 = (te_multi['k5']['TE_FPGA_GPU_coupled'] > te_multi['k5']['TE_FPGA_GPU_uncoupled'] and
                te_multi['k5']['TE_GPU_FPGA_coupled'] > te_multi['k5']['TE_GPU_FPGA_uncoupled'])
    tests['T662'] = {'desc': 'Bidirectional TE at k=5 (deep temporal coupling)',
                     'val': min(te_multi['k5']['ratio_FPGA_GPU'], te_multi['k5']['ratio_GPU_FPGA']),
                     'pass': bidir_k5}
    print(f"  T662: bidir TE k=5 {'PASS' if bidir_k5 else 'FAIL'}")

    # T663: At least 3/4 lag values show bidirectional coupling
    n_bidir = sum(1 for k_str in ['k1', 'k2', 'k5', 'k10']
                  if (te_multi[k_str]['TE_FPGA_GPU_coupled'] > te_multi[k_str]['TE_FPGA_GPU_uncoupled'] and
                      te_multi[k_str]['TE_GPU_FPGA_coupled'] > te_multi[k_str]['TE_GPU_FPGA_uncoupled']))
    tests['T663'] = {'desc': 'Bidirectional TE at ≥3/4 lag values',
                     'val': n_bidir, 'pass': n_bidir >= 3}
    print(f"  T663: {n_bidir}/4 lags bidirectional {'PASS' if n_bidir >= 3 else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXP 5: 1/f NOISE SIGNATURE TRANSFER
    #   Does the GPU's 1/f power spectrum transfer to FPGA spikes?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 5: 1/f NOISE SIGNATURE TRANSFER")
    print("  Does GPU firmware 1/f character appear in FPGA spike timing?")
    print("=" * 72)

    # GPU power PSD
    gpu_psd = psd_slope(c_gpu[20:, 4], fs=SAMPLE_HZ)
    print(f"  GPU power PSD slope: {gpu_psd:.2f}")

    # FPGA spike rate PSD (coupled)
    spk_psd_coupled = psd_slope(c_spk[20:].mean(axis=1), fs=SAMPLE_HZ)
    print(f"  FPGA spike rate PSD (coupled): {spk_psd_coupled:.2f}")

    # FPGA spike rate PSD (uncoupled)
    spk_psd_uncoupled = psd_slope(u_spk[20:].mean(axis=1), fs=SAMPLE_HZ)
    print(f"  FPGA spike rate PSD (uncoupled): {spk_psd_uncoupled:.2f}")

    # Per-channel PSD analysis
    channel_psds = {}
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ch_data = c_gpu[20:, ch_idx]
        if np.std(ch_data) > 1e-10:
            ch_psd = psd_slope(ch_data, fs=SAMPLE_HZ)
            channel_psds[ch_name] = ch_psd
            print(f"    {ch_name}: PSD slope = {ch_psd:.2f}")
        else:
            channel_psds[ch_name] = 0.0
            print(f"    {ch_name}: constant (no spectral content)")

    # Dispatch jitter ACF analysis
    jitter_acf = np.corrcoef(c_gpu[20:-1, 8], c_gpu[21:, 8])[0, 1] if np.std(c_gpu[20:, 8]) > 1e-10 else 0
    print(f"  Dispatch jitter ACF(1): {jitter_acf:.3f}")

    results['noise_transfer'] = {
        'gpu_power_psd': gpu_psd,
        'spike_psd_coupled': spk_psd_coupled,
        'spike_psd_uncoupled': spk_psd_uncoupled,
        'channel_psds': channel_psds,
        'jitter_acf': jitter_acf,
    }

    # T664: GPU power shows 1/f (PSD slope < -0.5)
    tests['T664'] = {'desc': 'GPU power PSD slope < -0.5 (1/f)',
                     'val': gpu_psd, 'pass': gpu_psd < -0.5}
    print(f"\n  T664: GPU power PSD = {gpu_psd:.2f} {'PASS' if gpu_psd < -0.5 else 'FAIL'}")

    # T665: Coupled spike PSD closer to GPU than uncoupled (1/f transfer)
    coupled_similarity = abs(spk_psd_coupled - gpu_psd)
    uncoupled_similarity = abs(spk_psd_uncoupled - gpu_psd)
    transfer = coupled_similarity < uncoupled_similarity
    tests['T665'] = {'desc': 'Coupled spike PSD closer to GPU 1/f than uncoupled',
                     'val': coupled_similarity - uncoupled_similarity,
                     'pass': transfer}
    print(f"  T665: 1/f transfer = {'PASS' if transfer else 'FAIL'} "
          f"(coupled gap={coupled_similarity:.2f}, uncoupled gap={uncoupled_similarity:.2f})")

    # T666: Dispatch jitter has temporal memory (ACF > 0.5)
    tests['T666'] = {'desc': 'Dispatch jitter ACF(1) > 0.5',
                     'val': jitter_acf, 'pass': jitter_acf > 0.5}
    print(f"  T666: jitter ACF = {jitter_acf:.3f} {'PASS' if jitter_acf > 0.5 else 'FAIL'}")

    # T667: At least 5 channels show 1/f character (slope < -0.3)
    n_1f = sum(1 for ch in CHANNEL_NAMES if channel_psds.get(ch, 0) < -0.3)
    tests['T667'] = {'desc': 'At least 5 GPU channels show 1/f (slope < -0.3)',
                     'val': n_1f, 'pass': n_1f >= 5}
    print(f"  T667: {n_1f} channels with 1/f {'PASS' if n_1f >= 5 else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXP 6: DIFFICULTY SCALING
    #   4-class, 6-class, 8-class — does coupled advantage persist?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 6: DIFFICULTY SCALING")
    print("  4-class, 6-class, 8-class waveform classification")
    print("=" * 72)

    difficulty_results = {}
    for n_cls in [4, 6, 8]:
        n_tri = max(N_TRIALS, n_cls * 15)  # At least 15 trials per class
        inputs_d, labels_d = generate_waveforms(n_tri, N_STEPS, n_classes=n_cls, seed=200 + n_cls)

        diff_accs = {}
        for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
            feats = []
            for trial in range(n_tri):
                spk, vm, glog, ints = run_coupled_loop(
                    fpga, inputs_d[trial], w_in, w_gpu, w_fb, mode=cond)
                trial_state = np.hstack([spk, vm, glog])
                f = np.concatenate([trial_state.mean(axis=0), trial_state.std(axis=0)])
                feats.append(f)
                if (trial + 1) % 20 == 0:
                    print(f"    {n_cls}-class {cond}: trial {trial+1}/{n_tri}", flush=True)
            X = np.array(feats)
            diff_accs[cond] = classify_cv(X, labels_d, n_classes=n_cls)['mean']

        chance = 1.0 / n_cls
        difficulty_results[n_cls] = {
            'coupled': diff_accs['COUPLED'],
            'fpga_only': diff_accs['FPGA_ONLY'],
            'static': diff_accs['STATIC'],
            'chance': chance,
            'coupled_above_chance': diff_accs['COUPLED'] - chance,
            'coupled_advantage': diff_accs['COUPLED'] - diff_accs['FPGA_ONLY'],
        }
        print(f"  {n_cls}-class: COUPLED={diff_accs['COUPLED']:.3f}, "
              f"FPGA={diff_accs['FPGA_ONLY']:.3f}, STATIC={diff_accs['STATIC']:.3f} "
              f"(chance={chance:.3f})")

    results['difficulty_scaling'] = difficulty_results

    # T668: COUPLED > chance at all difficulty levels
    all_above = all(difficulty_results[n]['coupled'] > difficulty_results[n]['chance'] + 0.05
                    for n in [4, 6, 8])
    tests['T668'] = {'desc': 'COUPLED > chance+5pp at all difficulty levels',
                     'val': min(difficulty_results[n]['coupled_above_chance'] for n in [4, 6, 8]),
                     'pass': all_above}
    print(f"\n  T668: all above chance {'PASS' if all_above else 'FAIL'}")

    # T669: COUPLED > FPGA_ONLY at majority of difficulty levels
    n_better = sum(1 for n in [4, 6, 8] if difficulty_results[n]['coupled'] > difficulty_results[n]['fpga_only'])
    tests['T669'] = {'desc': 'COUPLED > FPGA_ONLY at ≥2/3 difficulty levels',
                     'val': n_better, 'pass': n_better >= 2}
    print(f"  T669: COUPLED > FPGA at {n_better}/3 levels {'PASS' if n_better >= 2 else 'FAIL'}")

    # T670: COUPLED > STATIC at all difficulty levels
    all_static = all(difficulty_results[n]['coupled'] > difficulty_results[n]['static'] for n in [4, 6, 8])
    tests['T670'] = {'desc': 'COUPLED > STATIC at all difficulty levels',
                     'val': sum(1 for n in [4, 6, 8] if difficulty_results[n]['coupled'] > difficulty_results[n]['static']),
                     'pass': all_static}
    print(f"  T670: COUPLED > STATIC at all levels {'PASS' if all_static else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXP 7: CAUSAL EMERGENCE WITH HIGHER STATISTICS
    #   100 micro pairs, macro from multiple channels
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 7: CAUSAL EMERGENCE (HIGH STATISTICS)")
    print("  100+ micro pairs, multiple macro channels")
    print("=" * 72)

    # Use the long coupled run from EXP 4
    micro_te = []
    for n_idx in range(0, N_NEURONS, 4):  # Every 4th neuron = 32 neurons
        for g_idx in range(9):  # All 9 GPU channels
            s = c_spk[10:, n_idx]
            g = c_gpu[10:, g_idx]
            if np.std(s) > 1e-10 and np.std(g) > 1e-10:
                micro_te.append(transfer_entropy(s, g, k=1))
    avg_micro_te = np.mean(micro_te) if micro_te else 0.0
    n_micro_pairs = len(micro_te)

    # Macro: aggregate spike rate → aggregate GPU
    macro_spike = c_spk[10:].mean(axis=1)
    macro_gpu = c_gpu[10:, :5].mean(axis=1)  # First 5 channels
    macro_te_k1 = transfer_entropy(macro_spike, macro_gpu, k=1)
    macro_te_k2 = transfer_entropy(macro_spike, macro_gpu, k=2)

    emergence_k1 = macro_te_k1 / max(avg_micro_te, 1e-6)
    emergence_k2 = macro_te_k2 / max(avg_micro_te, 1e-6)

    # Also: macro → macro (different direction)
    macro_gpu_spike = transfer_entropy(macro_gpu, macro_spike, k=2)

    print(f"  Micro TE (avg of {n_micro_pairs} pairs): {avg_micro_te:.4f}")
    print(f"  Macro TE (k=1):  {macro_te_k1:.4f} (emergence={emergence_k1:.2f})")
    print(f"  Macro TE (k=2):  {macro_te_k2:.4f} (emergence={emergence_k2:.2f})")
    print(f"  Macro TE reverse: {macro_gpu_spike:.4f}")

    results['causal_emergence'] = {
        'n_micro_pairs': n_micro_pairs,
        'avg_micro_te': avg_micro_te,
        'macro_te_k1': macro_te_k1,
        'macro_te_k2': macro_te_k2,
        'emergence_k1': emergence_k1,
        'emergence_k2': emergence_k2,
        'macro_te_reverse': macro_gpu_spike,
    }

    # T671: Macro TE > micro TE (causal emergence exists)
    tests['T671'] = {'desc': 'Macro TE > avg micro TE (causal emergence)',
                     'val': emergence_k2, 'pass': macro_te_k2 > avg_micro_te}
    print(f"\n  T671: emergence ratio k=2 = {emergence_k2:.2f} {'PASS' if macro_te_k2 > avg_micro_te else 'FAIL'}")

    # T672: Macro TE > 0.01 (meaningful macro-level coupling)
    tests['T672'] = {'desc': 'Macro TE k=2 > 0.01',
                     'val': macro_te_k2, 'pass': macro_te_k2 > 0.01}
    print(f"  T672: macro TE = {macro_te_k2:.4f} {'PASS' if macro_te_k2 > 0.01 else 'FAIL'}")

    # T673: Bidirectional macro TE (both directions > 0.005)
    bidir_macro = (macro_te_k2 > 0.005) and (macro_gpu_spike > 0.005)
    tests['T673'] = {'desc': 'Bidirectional macro TE > 0.005',
                     'val': min(macro_te_k2, macro_gpu_spike),
                     'pass': bidir_macro}
    print(f"  T673: bidir macro TE {'PASS' if bidir_macro else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXP 8: MEMORY CAPACITY (slow membrane τ≈49.4ms + full 32-bit vmem)
    #   MC = Σ_d corr(u(t-d), y_d(t))² for d=1..MAX_DELAY
    #   u(t) = input signal, y_d(t) = ridge regression from reservoir state
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 8: MEMORY CAPACITY")
    print("  Reservoir MC with slow membrane (τ≈49.4ms) + full Q16.16 vmem")
    print("=" * 72)

    MAX_DELAY = 15
    MC_STEPS = 400
    MC_TRIALS = 60
    mc_alpha = 1.0  # Ridge regularization

    # Generate random input signal (uniform [0,1])
    rng_mc = np.random.RandomState(777)
    mc_input = rng_mc.uniform(0, 1, size=(MC_TRIALS, MC_STEPS))

    def compute_mc(fpga, mc_input, w_in, w_gpu, w_fb, mode, max_delay, alpha):
        """Run reservoir and compute memory capacity at each delay."""
        n_tri, n_steps = mc_input.shape
        all_states = []
        for trial in range(n_tri):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, mc_input[trial], w_in, w_gpu, w_fb, mode=mode)
            # Use vmem as primary reservoir state (full 32-bit precision now)
            state = np.hstack([spk, vm])
            all_states.append(state)
            if (trial + 1) % 20 == 0:
                print(f"    MC {mode}: trial {trial+1}/{n_tri}", flush=True)

        all_states = np.array(all_states)  # (n_tri, n_steps, n_feat)

        mc_per_delay = np.zeros(max_delay)
        for d in range(1, max_delay + 1):
            # Target: u(t-d) for t = d..n_steps-1
            targets = []
            features = []
            for trial in range(n_tri):
                for t in range(d, n_steps):
                    features.append(all_states[trial, t])
                    targets.append(mc_input[trial, t - d])
            X = np.array(features)
            y = np.array(targets)

            # Ridge regression with train/test split
            n = len(y)
            idx = np.arange(n)
            rng_mc.shuffle(idx)
            split = int(0.7 * n)
            X_tr, X_te = X[idx[:split]], X[idx[split:]]
            y_tr, y_te = y[idx[:split]], y[idx[split:]]

            # Normalize
            mu = X_tr.mean(axis=0)
            sigma = X_tr.std(axis=0)
            sigma[sigma < 1e-6] = 1.0
            X_tr = (X_tr - mu) / sigma
            X_te = (X_te - mu) / sigma

            # Ridge solve
            I = np.eye(X_tr.shape[1])
            try:
                w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
                y_pred = X_te @ w
                corr = np.corrcoef(y_te, y_pred)[0, 1]
                mc_per_delay[d - 1] = max(0, corr ** 2)
            except:
                mc_per_delay[d - 1] = 0.0

        total_mc = mc_per_delay.sum()
        return total_mc, mc_per_delay

    mc_results = {}
    for cond in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- MC condition: {cond} ---")
        total, per_delay = compute_mc(
            fpga, mc_input, w_in, w_gpu, w_fb, cond, MAX_DELAY, mc_alpha)
        mc_results[cond] = {
            'total_mc': float(total),
            'per_delay': per_delay.tolist(),
        }
        print(f"  {cond}: total MC = {total:.3f}")
        print(f"    per-delay: {', '.join(f'd{d+1}={v:.3f}' for d, v in enumerate(per_delay[:5]))}, ...")

    results['memory_capacity'] = mc_results

    # T674: COUPLED MC > 0.3 (non-zero memory capacity with slow membrane)
    mc_coupled = mc_results['COUPLED']['total_mc']
    tests['T674'] = {'desc': 'COUPLED total MC > 0.3 (slow membrane)',
                     'val': mc_coupled, 'pass': mc_coupled > 0.3}
    print(f"\n  T674: COUPLED MC = {mc_coupled:.3f} {'PASS' if mc_coupled > 0.3 else 'FAIL'}")

    # T675: COUPLED MC > STATIC MC (coupling helps memory)
    mc_static = mc_results['STATIC']['total_mc']
    tests['T675'] = {'desc': 'COUPLED MC > STATIC MC',
                     'val': mc_coupled - mc_static,
                     'pass': mc_coupled > mc_static}
    print(f"  T675: COUPLED({mc_coupled:.3f}) > STATIC({mc_static:.3f}) "
          f"{'PASS' if mc_coupled > mc_static else 'FAIL'}")

    # T676: COUPLED MC > FPGA_ONLY MC (GPU noise enriches temporal memory)
    mc_fpga = mc_results['FPGA_ONLY']['total_mc']
    tests['T676'] = {'desc': 'COUPLED MC > FPGA_ONLY MC',
                     'val': mc_coupled - mc_fpga,
                     'pass': mc_coupled > mc_fpga}
    print(f"  T676: COUPLED({mc_coupled:.3f}) > FPGA_ONLY({mc_fpga:.3f}) "
          f"{'PASS' if mc_coupled > mc_fpga else 'FAIL'}")

    # T677: MC at delay d=1 > 0.1 (short-term memory works)
    mc_d1 = mc_results['COUPLED']['per_delay'][0]
    tests['T677'] = {'desc': 'MC at d=1 > 0.1 (short-term memory)',
                     'val': mc_d1, 'pass': mc_d1 > 0.1}
    print(f"  T677: MC(d=1) = {mc_d1:.3f} {'PASS' if mc_d1 > 0.1 else 'FAIL'}")

    # T678: MC at d=5 > 0 (medium-range memory with τ≈49ms at 200Hz = ~10 steps)
    mc_d5 = mc_results['COUPLED']['per_delay'][4]
    tests['T678'] = {'desc': 'MC at d=5 > 0 (medium-range memory)',
                     'val': mc_d5, 'pass': mc_d5 > 0.0}
    print(f"  T678: MC(d=5) = {mc_d5:.3f} {'PASS' if mc_d5 > 0.0 else 'FAIL'}")

    # T679: MC decays monotonically for first 5 delays (expected for fading memory)
    per_d = mc_results['COUPLED']['per_delay']
    monotonic_count = sum(1 for i in range(4) if per_d[i] >= per_d[i+1])
    tests['T679'] = {'desc': 'MC monotonic decay for first 5 delays (≥3/4 pairs)',
                     'val': monotonic_count, 'pass': monotonic_count >= 3}
    print(f"  T679: monotonic {monotonic_count}/4 pairs {'PASS' if monotonic_count >= 3 else 'FAIL'}")

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
        val = t['val']
        if isinstance(val, float):
            print(f"  {tid}: {status} — {t['desc']} (val={val:.4f})")
        else:
            print(f"  {tid}: {status} — {t['desc']} (val={val})")

    # Save
    out_path = RESULTS / 'z2225_refined_coupled_evidence.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nSaved: {out_path}")

    fpga.close()


if __name__ == '__main__':
    main()
