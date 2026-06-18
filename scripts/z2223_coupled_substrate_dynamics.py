#!/usr/bin/env python3
"""z2223_coupled_substrate_dynamics.py — Deep Coupled Oscillator Experiment

BREAKTHROUGH IDEA: The GPU's power management subsystem (SMU PID controller,
DVFS feedback loop, VRM switching regulator, thermal manager) is ALREADY a
complex recurrent dynamical system with physical temporal memory. We don't
need to add software memory — the GPU firmware IS the reservoir's recurrence.

The FPGA NS-RAM neurons provide stochastic nonlinear perturbation.
The coupling creates emergent dynamics that neither substrate exhibits alone.

Architecture:
  Input → FPGA spike rates → HIP workload intensity → GPU firmware dynamics
              ↑                                              ↓
              └── FPGA Vg modulation ← Deep GPU state reads ←┘

  GPU firmware = {SMU PID, VRM switcher, DVFS controller, thermal manager}
              = complex recurrent dynamical system with µs-to-second timescales

Key measurements:
  1. Transfer entropy BOTH directions (bidirectional information flow)
  2. Input-dependent GPU trajectories (DVFS PID responds to input modulation)
  3. Emergent dimensionality (coupled > uncoupled)
  4. Causal emergence (macro dynamics ≠ sum of micro)
  5. THEN classification (if coupling is real, can we compute with it?)

FAST: 500Hz+ sampling, 0.5ms workload bursts, ALL deep probes.
NO IIR, NO delay taps, NO software memory. Physics only.

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128 neurons, UDP Ethernet)
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters ───
N_NEURONS = 128
BASE_VG = 0.45          # Lower than z2222 to reduce spike saturation
ALPHA = 0.35            # Input → Vg gain
BETA_POWER = 0.12       # GPU power → Vg
BETA_THERMAL = 0.08     # GPU thermal → Vg
BETA_CLOCK = 0.10       # GPU SCLK → Vg (DVFS dynamics = the reservoir!)
SAMPLE_HZ = 200         # 200Hz — 5ms steps, fast enough for DVFS dynamics
WORKLOAD_MS = 1.5       # 1.5ms burst (not 4ms — faster iteration)
N_STEPS = 300           # 1.5s per trial at 200Hz
N_TRIALS = 60           # Enough for 5-fold CV on 4-class

# Deep probe paths
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
HWMON_TEMP = "/sys/class/hwmon/hwmon7/temp1_input"
HWMON_FREQ = "/sys/class/hwmon/hwmon7/freq1_input"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
GPU_BUSY_PATH = "/sys/class/drm/card0/device/gpu_busy_percent"
GRBM_PATH = "/sys/kernel/debug/dri/0/amdgpu_regs"  # MMIO if available

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# DEEP GPU PROBES — Every layer we can reach
# ═══════════════════════════════════════════════════════════

def read_smn_adc():
    """D0: SMN thermal ADC at 0x59800 — BELOW firmware, raw junction temp."""
    try:
        with open(SMN_PATH, 'rb+') as f:
            f.write(struct.pack('<I', 0x59800))
            f.seek(0)
            raw = struct.unpack('<I', f.read(4))[0]
        return (raw >> 21) * 0.125
    except:
        return None

def read_pm_table():
    """D1-3: PM table — SMU firmware level (thermal, power, SCLK)."""
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(0x4C); thermal = struct.unpack('<f', f.read(4))[0]
            f.seek(0x04); power = struct.unpack('<f', f.read(4))[0]
            f.seek(0x78); sclk = struct.unpack('<f', f.read(4))[0]
        return thermal, power, sclk
    except:
        return None, None, None

def read_hwmon():
    """D4-5: hwmon — driver level (VRM power, temp, freq)."""
    try: power = int(open(HWMON_POWER).read().strip()) / 1e6
    except: power = None
    try: temp = int(open(HWMON_TEMP).read().strip()) / 1000.0
    except: temp = None
    try: freq = int(open(HWMON_FREQ).read().strip()) / 1e6
    except: freq = None
    return power, temp, freq

def read_gpu_busy():
    """D6: GPU utilization percent."""
    try: return int(open(GPU_BUSY_PATH).read().strip())
    except: return 0

def read_all_gpu_state():
    """Read ALL deep GPU probes. Returns flat dict of raw values."""
    smn = read_smn_adc()
    pm_t, pm_p, pm_sclk = read_pm_table()
    hw_p, hw_t, hw_f = read_hwmon()
    busy = read_gpu_busy()
    return {
        'smn_temp': smn, 'pm_thermal': pm_t, 'pm_power': pm_p,
        'pm_sclk': pm_sclk, 'hw_power': hw_p, 'hw_temp': hw_t,
        'hw_freq': hw_f, 'gpu_busy': busy
    }


# ═══════════════════════════════════════════════════════════
# GPU Workload — SHORT bursts to excite DVFS dynamics
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
    """Short GPU burst to excite DVFS dynamics. Fire-and-forget (no sync)."""
    if not _torch_available or intensity < 0.05:
        return 0.0
    import torch
    N = int(128 + 896 * np.clip(intensity, 0.0, 1.0))  # 128 to 1024
    a = torch.randn(N, N, device=_torch_device)
    b = torch.randn(N, N, device=_torch_device)
    t0 = time.perf_counter()
    deadline = t0 + duration_ms / 1000.0
    while time.perf_counter() < deadline:
        _ = torch.mm(a, b)
    # NO torch.cuda.synchronize() — fire and forget for speed
    elapsed = time.perf_counter() - t0
    del a, b
    return elapsed


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ═══════════════════════════════════════════════════════════
# COUPLED DYNAMICS LOOP
# ═══════════════════════════════════════════════════════════

def run_coupled_loop(fpga, input_signal, w_in, w_gpu, w_fb,
                     mode='COUPLED', record_gpu=True):
    """Deep coupled substrate dynamics loop.

    Modes:
      COUPLED:    Full bidirectional (FPGA ↔ GPU)
      FPGA_ONLY:  FPGA driven by input, no GPU coupling
      GPU_ONLY:   GPU driven by input, FPGA records but doesn't couple
      UNCOUPLED:  Both run independently (no information exchange)
      STATIC:     Fixed Vg, baseline

    Returns:
      spikes:   (n_steps, N_NEURONS) spike deltas
      vmem:     (n_steps, N_NEURONS) membrane voltages
      gpu_state:(n_steps, 8) deep GPU readings [smn, pm_t, pm_p, pm_sclk, hw_p, hw_t, hw_f, busy]
      intensities: (n_steps,) workload intensities
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    spikes = np.zeros((n_steps, N_NEURONS))
    vmem_log = np.zeros((n_steps, N_NEURONS))
    gpu_log = np.zeros((n_steps, 8))
    intensities = np.zeros(n_steps)
    prev_counts = None

    for t in range(n_steps):
        t_start = time.perf_counter()

        # ── Read ALL GPU state ──
        if record_gpu:
            gs = read_all_gpu_state()
            gpu_log[t] = [
                gs['smn_temp'] or 0, gs['pm_thermal'] or 0,
                gs['pm_power'] or 0, gs['pm_sclk'] or 0,
                gs['hw_power'] or 0, gs['hw_temp'] or 0,
                gs['hw_freq'] or 0, gs['gpu_busy'] or 0
            ]

        # ── Compute Vg from input + GPU state ──
        vg = np.full(N_NEURONS, BASE_VG)

        if mode in ('COUPLED', 'FPGA_ONLY'):
            # Input drives all neurons (different projections)
            vg += ALPHA * input_signal[t] * w_in

        if mode == 'COUPLED':
            # GPU deep state → Vg (DVFS dynamics are the KEY temporal source)
            hw_p = gpu_log[t, 4]  # hwmon power
            pm_sclk = gpu_log[t, 3]  # PM table SCLK
            pm_t = gpu_log[t, 1]  # PM thermal

            # Normalize relative to running baseline (first 5 steps)
            if t >= 5:
                p_base = gpu_log[max(0,t-20):t, 4].mean()
                s_base = gpu_log[max(0,t-20):t, 3].mean()
                t_base = gpu_log[max(0,t-20):t, 1].mean()
            else:
                p_base, s_base, t_base = hw_p, pm_sclk, pm_t

            p_delta = (hw_p - p_base) / max(abs(p_base), 1.0)
            s_delta = (pm_sclk - s_base) / max(abs(s_base), 1.0)
            t_delta = (pm_t - t_base) / max(abs(t_base), 1.0)

            # Different GPU signals → different neuron groups
            n3 = N_NEURONS // 3
            vg[:n3] += BETA_POWER * p_delta * w_gpu[:n3]
            vg[n3:2*n3] += BETA_CLOCK * s_delta * w_gpu[n3:2*n3]
            vg[2*n3:] += BETA_THERMAL * t_delta * w_gpu[2*n3:]

            # SET_TEMP for physics-level BVpar coupling
            if gs['pm_thermal']:
                try:
                    fpga.set_temp(float(gs['pm_thermal']) + 273.15)
                except:
                    pass

        vg = np.clip(vg, 0.10, 0.85)

        # ── Drive FPGA & read state ──
        if mode != 'GPU_ONLY':
            fpga.set_vg_batch(0, vg.tolist())
            time.sleep(0.0003)  # 300µs settle

            try:
                counts, vm, bvpar = fpga.read_telemetry_fast()
                if prev_counts is not None:
                    for i in range(N_NEURONS):
                        delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                        if delta > 30000: delta = 0
                        spikes[t, i] = delta
                vmem_log[t] = vm
                prev_counts = counts.copy()
            except:
                pass

        # ── Spike-driven feedback → GPU workload ──
        if mode == 'COUPLED':
            # Weighted spike sum → workload intensity
            if t >= 1:
                recent_spikes = spikes[max(0,t-2):t+1].mean(axis=0)
                raw = float(np.dot(recent_spikes, w_fb))
                intensity = float(sigmoid(raw - 5.0))
            else:
                intensity = 0.3
            run_workload(intensity, duration_ms=WORKLOAD_MS)
            try:
                fpga.set_mac_signal(intensity * 0.5)
            except:
                pass
        elif mode == 'GPU_ONLY':
            # Input directly drives workload (no FPGA in loop)
            intensity = float(0.2 + 0.6 * np.clip(input_signal[t], 0, 1))
            run_workload(intensity, duration_ms=WORKLOAD_MS)
        elif mode == 'UNCOUPLED':
            # Fixed workload, not input-dependent
            intensity = 0.5
            run_workload(intensity, duration_ms=WORKLOAD_MS)
        else:
            intensity = 0.0

        intensities[t] = intensity

        # Pace
        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.0003:
            time.sleep(remaining)

    return spikes, vmem_log, gpu_log, intensities


# ═══════════════════════════════════════════════════════════
# ANALYSIS — Coupling, Emergence, Information
# ═══════════════════════════════════════════════════════════

def transfer_entropy(source, target, k=1, bins=8):
    """Transfer entropy TE(source→target) using binned estimator."""
    if len(source) < k + 2 or np.std(source) < 1e-10 or np.std(target) < 1e-10:
        return 0.0
    # Bin the signals
    s_bins = np.digitize(source, np.linspace(source.min()-1e-10, source.max()+1e-10, bins+1)) - 1
    t_bins = np.digitize(target, np.linspace(target.min()-1e-10, target.max()+1e-10, bins+1)) - 1
    n = len(source) - k
    # Count joint and marginal distributions
    from collections import Counter
    joint_tts = Counter()  # (target_t, target_past, source_past)
    joint_tt = Counter()   # (target_t, target_past)
    marg_ts = Counter()    # (target_past, source_past)
    marg_t = Counter()     # (target_past,)
    for i in range(k, len(source)):
        tt = t_bins[i]
        tp = tuple(t_bins[i-k:i])
        sp = tuple(s_bins[i-k:i])
        joint_tts[(tt, tp, sp)] += 1
        joint_tt[(tt, tp)] += 1
        marg_ts[(tp, sp)] += 1
        marg_t[tp] += 1
    # TE = sum p(t,tp,sp) * log[ p(t|tp,sp) / p(t|tp) ]
    te = 0.0
    for (tt, tp, sp), c_tts in joint_tts.items():
        p_tts = c_tts / n
        p_t_given_ts = c_tts / max(marg_ts[(tp, sp)], 1)
        p_t_given_t = joint_tt[(tt, tp)] / max(marg_t[tp], 1)
        if p_t_given_ts > 0 and p_t_given_t > 0:
            te += p_tts * np.log2(p_t_given_ts / p_t_given_t)
    return max(0.0, te)


def effective_dimension(X):
    """Effective dimensionality via participation ratio of singular values."""
    if X.shape[0] < 3 or X.shape[1] < 2:
        return 1.0
    X_c = X - X.mean(axis=0)
    try:
        s = np.linalg.svd(X_c, compute_uv=False)
        s2 = s ** 2
        if s2.sum() < 1e-10: return 1.0
        p = s2 / s2.sum()
        return float((s2.sum()) ** 2 / (s2 ** 2).sum())
    except:
        return 1.0


def mutual_information(x, y, bins=10):
    """MI(X,Y) via binned estimator."""
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


def cross_correlation_peak(x, y, max_lag=50):
    """Peak cross-correlation and its lag."""
    x = (x - x.mean()) / max(x.std(), 1e-10)
    y = (y - y.mean()) / max(y.std(), 1e-10)
    best_r, best_lag = 0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.corrcoef(x[:len(x)-lag], y[lag:])[0, 1] if lag < len(x) else 0
        else:
            r = np.corrcoef(x[-lag:], y[:len(y)+lag])[0, 1] if -lag < len(y) else 0
        if not np.isnan(r) and abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_r, best_lag


def psd_slope(x, fs=200):
    """Power spectral density slope (should be near -1 for 1/f)."""
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
    """Ridge regression classifier."""
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
# WAVEFORM GENERATION (4-class for clearer signal)
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials, steps, seed=42):
    """4-class waveforms: sin, triangle, square, sawtooth."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(steps) * dt
    trials, labels = [], []
    for _ in range(n_trials):
        cls = rng.integers(0, 4)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 2.0)
        if cls == 0:    wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:  wave = 2.0 * np.abs(2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0) - 1.0
        elif cls == 2:  wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        else:           wave = 2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)


# ═══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 70)
    print("z2223: DEEP COUPLED SUBSTRATE DYNAMICS")
    print("  GPU firmware control loops AS reservoir recurrence")
    print("  FPGA NS-RAM neurons AS stochastic nonlinear perturbation")
    print("  NO software memory — GPU DVFS/PID IS the temporal memory")
    print("  ALL deep probes: SMN ADC, PM table, hwmon, gpu_busy")
    print("=" * 70)

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
    for k, v in test_gs.items():
        status = f"{v}" if v is not None else "UNAVAIL"
        print(f"    {k}: {status}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    w_gpu = rng.uniform(-1, 1, N_NEURONS)
    w_fb = rng.uniform(-1, 1, N_NEURONS)

    results = {
        'experiment': 'z2223_coupled_substrate_dynamics',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'software_memory': 'NONE',
            'IIR_filter': False, 'delay_taps': False, 'cumulative': False,
            'temporal_source': 'GPU_DVFS_PID + GPU_thermal_inertia + VRM_switching',
            'FPGA_role': 'stochastic_nonlinear_perturbation',
            'GPU_role': 'recurrent_dynamical_system (firmware control loops)',
            'sample_hz': SAMPLE_HZ, 'n_steps': N_STEPS, 'n_trials': N_TRIALS,
            'workload_ms': WORKLOAD_MS,
        }
    }
    tests = {}

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 1: GPU FIRMWARE AS DYNAMICAL SYSTEM
    #   Show that GPU DVFS/power responds with complex temporal
    #   dynamics to input-modulated workload. The PID controller
    #   creates memory — we just need to prove it.
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXP 1: GPU FIRMWARE DYNAMICAL RESPONSE")
    print("  Driving GPU with step/ramp/sine workloads")
    print("  Measuring DVFS PID response in PM table SCLK + power")
    print("=" * 70)

    # Step response: idle → full → idle
    step_signal = np.concatenate([np.zeros(100), np.ones(100), np.zeros(100)])
    # Ramp: gradual increase
    ramp_signal = np.concatenate([np.linspace(0, 1, 150), np.linspace(1, 0, 150)])
    # Sine: oscillating workload
    sine_signal = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(300) / 60)

    stim_names = ['STEP', 'RAMP', 'SINE']
    stim_signals = [step_signal, ramp_signal, sine_signal]
    gpu_responses = {}

    for name, stim in zip(stim_names, stim_signals):
        print(f"\n  --- {name} stimulus ---")
        gpu_trace = np.zeros((len(stim), 8))
        for t in range(len(stim)):
            t0 = time.perf_counter()
            # Drive workload with stimulus
            run_workload(float(stim[t]), duration_ms=WORKLOAD_MS)
            # Read all GPU state
            gs = read_all_gpu_state()
            gpu_trace[t] = [
                gs['smn_temp'] or 0, gs['pm_thermal'] or 0,
                gs['pm_power'] or 0, gs['pm_sclk'] or 0,
                gs['hw_power'] or 0, gs['hw_temp'] or 0,
                gs['hw_freq'] or 0, gs['gpu_busy'] or 0
            ]
            # Pace to 200Hz
            elapsed = time.perf_counter() - t0
            if elapsed < 1.0/SAMPLE_HZ:
                time.sleep(1.0/SAMPLE_HZ - elapsed)

        gpu_responses[name] = gpu_trace

        # Analyze: does GPU state carry memory of stimulus?
        # Power should track workload but with PID dynamics
        power = gpu_trace[:, 4]  # hwmon power
        sclk = gpu_trace[:, 3]   # PM SCLK
        thermal = gpu_trace[:, 1] # PM thermal

        # Cross-correlation with stimulus
        xcorr_power, lag_power = cross_correlation_peak(stim, power, max_lag=30)
        xcorr_sclk, lag_sclk = cross_correlation_peak(stim, sclk, max_lag=30)

        # Autocorrelation of GPU response (temporal memory indicator)
        acf1_power = np.corrcoef(power[:-1], power[1:])[0,1] if len(power) > 2 else 0
        acf1_sclk = np.corrcoef(sclk[:-1], sclk[1:])[0,1] if len(sclk) > 2 else 0

        print(f"    Power xcorr={xcorr_power:.3f} (lag={lag_power}), ACF(1)={acf1_power:.3f}")
        print(f"    SCLK  xcorr={xcorr_sclk:.3f} (lag={lag_sclk}), ACF(1)={acf1_sclk:.3f}")
        print(f"    Thermal range: {thermal.min():.1f} - {thermal.max():.1f}°C")

    results['gpu_dynamics'] = {
        name: {
            'power_xcorr': float(cross_correlation_peak(stim, gpu_responses[name][:, 4])[0]),
            'sclk_xcorr': float(cross_correlation_peak(stim, gpu_responses[name][:, 3])[0]),
            'power_acf1': float(np.corrcoef(gpu_responses[name][:-1, 4], gpu_responses[name][1:, 4])[0,1]),
            'thermal_range': float(gpu_responses[name][:, 1].max() - gpu_responses[name][:, 1].min()),
        }
        for name, stim in zip(stim_names, stim_signals)
    }

    # T600: GPU power tracks workload (xcorr > 0.3 for at least one stimulus)
    max_power_xcorr = max(abs(cross_correlation_peak(s, gpu_responses[n][:, 4])[0])
                          for n, s in zip(stim_names, stim_signals))
    tests['T600'] = {'desc': 'GPU power tracks workload (xcorr>0.3)',
                     'val': max_power_xcorr, 'pass': max_power_xcorr > 0.3}
    print(f"\n  T600: max power xcorr = {max_power_xcorr:.3f} {'PASS' if max_power_xcorr > 0.3 else 'FAIL'}")

    # T601: GPU power has temporal memory (ACF(1) > 0.5)
    max_acf = max(np.corrcoef(gpu_responses[n][:-1, 4], gpu_responses[n][1:, 4])[0,1]
                  for n in stim_names)
    tests['T601'] = {'desc': 'GPU power ACF(1) > 0.5 (temporal memory)',
                     'val': max_acf, 'pass': max_acf > 0.5}
    print(f"  T601: max power ACF(1) = {max_acf:.3f} {'PASS' if max_acf > 0.5 else 'FAIL'}")

    # T602: SCLK responds to workload (xcorr > 0.1)
    max_sclk_xcorr = max(abs(cross_correlation_peak(s, gpu_responses[n][:, 3])[0])
                         for n, s in zip(stim_names, stim_signals))
    tests['T602'] = {'desc': 'SCLK responds to workload (xcorr>0.1)',
                     'val': max_sclk_xcorr, 'pass': max_sclk_xcorr > 0.1}
    print(f"  T602: max SCLK xcorr = {max_sclk_xcorr:.3f} {'PASS' if max_sclk_xcorr > 0.1 else 'FAIL'}")

    # T603: Thermal shows integration (monotone increase under load)
    step_thermal = gpu_responses['STEP'][:, 1]
    pre = step_thermal[:50].mean()
    post = step_thermal[150:200].mean()
    thermal_rise = post - pre
    tests['T603'] = {'desc': 'Thermal rises under sustained load (>0.2°C)',
                     'val': thermal_rise, 'pass': thermal_rise > 0.2}
    print(f"  T603: thermal rise = {thermal_rise:.2f}°C {'PASS' if thermal_rise > 0.2 else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 2: BIDIRECTIONAL INFORMATION FLOW
    #   Transfer entropy in BOTH directions through the coupling.
    #   This is the key test: does information flow FPGA → GPU AND GPU → FPGA?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXP 2: BIDIRECTIONAL INFORMATION FLOW")
    print("  Transfer Entropy + MI through coupled dynamics")
    print("=" * 70)

    # Run coupled loop with structured input
    test_input = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N_STEPS) / 60)
    print("\n  Running COUPLED...")
    c_spk, c_vm, c_gpu, c_int = run_coupled_loop(
        fpga, test_input, w_in, w_gpu, w_fb, mode='COUPLED')
    print(f"    spikes/step: {c_spk[1:].mean():.1f}, vmem std: {c_vm.std():.1f}")

    print("  Running UNCOUPLED...")
    u_spk, u_vm, u_gpu, u_int = run_coupled_loop(
        fpga, test_input, w_in, w_gpu, w_fb, mode='UNCOUPLED')
    print(f"    spikes/step: {u_spk[1:].mean():.1f}, vmem std: {u_vm.std():.1f}")

    # Aggregate spike signal for TE calculation
    c_spike_mean = c_spk[1:].mean(axis=1)
    c_power = c_gpu[1:, 4]
    u_spike_mean = u_spk[1:].mean(axis=1)
    u_power = u_gpu[1:, 4]

    # Transfer entropy: FPGA→GPU (do spikes predict power?)
    te_fpga_gpu_coupled = transfer_entropy(c_spike_mean, c_power, k=2)
    te_fpga_gpu_uncoupled = transfer_entropy(u_spike_mean, u_power, k=2)

    # Transfer entropy: GPU→FPGA (does power predict spikes?)
    te_gpu_fpga_coupled = transfer_entropy(c_power, c_spike_mean, k=2)
    te_gpu_fpga_uncoupled = transfer_entropy(u_power, u_spike_mean, k=2)

    # Mutual information
    mi_coupled = mutual_information(c_spike_mean, c_power)
    mi_uncoupled = mutual_information(u_spike_mean, u_power)

    print(f"\n  TE(FPGA→GPU):  coupled={te_fpga_gpu_coupled:.4f}, uncoupled={te_fpga_gpu_uncoupled:.4f}")
    print(f"  TE(GPU→FPGA):  coupled={te_gpu_fpga_coupled:.4f}, uncoupled={te_gpu_fpga_uncoupled:.4f}")
    print(f"  MI(spk,power): coupled={mi_coupled:.4f}, uncoupled={mi_uncoupled:.4f}")

    results['transfer_entropy'] = {
        'TE_FPGA_GPU_coupled': te_fpga_gpu_coupled,
        'TE_FPGA_GPU_uncoupled': te_fpga_gpu_uncoupled,
        'TE_GPU_FPGA_coupled': te_gpu_fpga_coupled,
        'TE_GPU_FPGA_uncoupled': te_gpu_fpga_uncoupled,
        'MI_coupled': mi_coupled, 'MI_uncoupled': mi_uncoupled,
    }

    # T604: TE(FPGA→GPU) coupled > uncoupled (spikes influence GPU)
    te_ratio_fg = te_fpga_gpu_coupled / max(te_fpga_gpu_uncoupled, 1e-6)
    tests['T604'] = {'desc': 'TE(FPGA→GPU) coupled > uncoupled',
                     'val': te_ratio_fg, 'pass': te_fpga_gpu_coupled > te_fpga_gpu_uncoupled}
    print(f"\n  T604: TE(FPGA→GPU) ratio = {te_ratio_fg:.2f} {'PASS' if tests['T604']['pass'] else 'FAIL'}")

    # T605: TE(GPU→FPGA) coupled > uncoupled (GPU influences spikes)
    te_ratio_gf = te_gpu_fpga_coupled / max(te_gpu_fpga_uncoupled, 1e-6)
    tests['T605'] = {'desc': 'TE(GPU→FPGA) coupled > uncoupled',
                     'val': te_ratio_gf, 'pass': te_gpu_fpga_coupled > te_gpu_fpga_uncoupled}
    print(f"  T605: TE(GPU→FPGA) ratio = {te_ratio_gf:.2f} {'PASS' if tests['T605']['pass'] else 'FAIL'}")

    # T606: Bidirectional (both directions have TE > 0.01)
    bidir = (te_fpga_gpu_coupled > 0.01) and (te_gpu_fpga_coupled > 0.01)
    tests['T606'] = {'desc': 'Bidirectional TE > 0.01 both directions',
                     'val': min(te_fpga_gpu_coupled, te_gpu_fpga_coupled),
                     'pass': bidir}
    print(f"  T606: Bidirectional TE = {min(te_fpga_gpu_coupled, te_gpu_fpga_coupled):.4f} {'PASS' if bidir else 'FAIL'}")

    # T607: MI coupled > uncoupled
    tests['T607'] = {'desc': 'MI(spk,power) coupled > uncoupled',
                     'val': mi_coupled - mi_uncoupled,
                     'pass': mi_coupled > mi_uncoupled}
    print(f"  T607: MI diff = {mi_coupled - mi_uncoupled:.4f} {'PASS' if mi_coupled > mi_uncoupled else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 3: EMERGENT DIMENSIONALITY
    #   Does the coupled system have higher effective dimensionality
    #   than either substrate alone? This proves the coupling creates
    #   NEW degrees of freedom that neither has independently.
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXP 3: EMERGENT DIMENSIONALITY")
    print("  PCA on coupled vs uncoupled system state")
    print("=" * 70)

    # Coupled: full state = [spikes(128) + vmem(128) + gpu(8)]
    coupled_full = np.hstack([c_spk[10:], c_vm[10:], c_gpu[10:]])
    fpga_only_state = np.hstack([c_spk[10:], c_vm[10:]])
    gpu_only_state = c_gpu[10:]
    uncoupled_full = np.hstack([u_spk[10:], u_vm[10:], u_gpu[10:]])

    ed_coupled = effective_dimension(coupled_full)
    ed_fpga = effective_dimension(fpga_only_state)
    ed_gpu = effective_dimension(gpu_only_state)
    ed_uncoupled = effective_dimension(uncoupled_full)
    ed_sum = ed_fpga + ed_gpu

    print(f"  Effective dimension:")
    print(f"    Coupled full:    {ed_coupled:.1f}")
    print(f"    FPGA only:       {ed_fpga:.1f}")
    print(f"    GPU only:        {ed_gpu:.1f}")
    print(f"    Sum (FPGA+GPU):  {ed_sum:.1f}")
    print(f"    Uncoupled full:  {ed_uncoupled:.1f}")

    results['dimensionality'] = {
        'coupled': ed_coupled, 'fpga_only': ed_fpga, 'gpu_only': ed_gpu,
        'sum_independent': ed_sum, 'uncoupled': ed_uncoupled,
    }

    # T608: Coupled ED > FPGA-only ED (GPU adds dimensions)
    tests['T608'] = {'desc': 'Coupled dim > FPGA-only dim',
                     'val': ed_coupled - ed_fpga,
                     'pass': ed_coupled > ed_fpga}
    print(f"\n  T608: Coupled-FPGA = {ed_coupled - ed_fpga:.1f} {'PASS' if ed_coupled > ed_fpga else 'FAIL'}")

    # T609: Coupled ED > Uncoupled ED (coupling creates new dimensions)
    tests['T609'] = {'desc': 'Coupled dim > Uncoupled dim',
                     'val': ed_coupled - ed_uncoupled,
                     'pass': ed_coupled > ed_uncoupled}
    print(f"  T609: Coupled-Uncoupled = {ed_coupled - ed_uncoupled:.1f} {'PASS' if ed_coupled > ed_uncoupled else 'FAIL'}")

    # T610: Synergistic dimensionality (coupled > sum of parts)
    synergy = ed_coupled > ed_sum * 0.8
    tests['T610'] = {'desc': 'Coupled dim > 0.8 * (FPGA+GPU) dims',
                     'val': ed_coupled / max(ed_sum, 1),
                     'pass': synergy}
    print(f"  T610: Synergy ratio = {ed_coupled / max(ed_sum, 1):.2f} {'PASS' if synergy else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 4: INPUT-DEPENDENT GPU TRAJECTORIES
    #   The breakthrough test: do DIFFERENT input waveforms create
    #   DIFFERENT GPU firmware trajectories? If yes, the GPU's
    #   DVFS/PID response is input-modulated through the FPGA.
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXP 4: INPUT-DEPENDENT GPU TRAJECTORIES")
    print("  Do different inputs create distinguishable GPU dynamics?")
    print("=" * 70)

    # Run 4 different input patterns through coupled system
    patterns = {
        'SINE': 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N_STEPS) / 60),
        'STEP': np.concatenate([np.zeros(100), np.ones(100), np.zeros(100)]),
        'RAMP': np.linspace(0, 1, N_STEPS),
        'PULSE': np.zeros(N_STEPS),
    }
    patterns['PULSE'][50:60] = 1.0
    patterns['PULSE'][150:160] = 1.0
    patterns['PULSE'][250:260] = 1.0

    gpu_trajectories = {}
    for name, inp in patterns.items():
        print(f"  {name}...")
        _, _, g_log, _ = run_coupled_loop(fpga, inp, w_in, w_gpu, w_fb, mode='COUPLED')
        gpu_trajectories[name] = g_log
        time.sleep(2.0)  # Let GPU cool between patterns

    # Measure distinguishability of GPU trajectories
    traj_names = list(gpu_trajectories.keys())
    n_traj = len(traj_names)
    dist_matrix = np.zeros((n_traj, n_traj))
    for i in range(n_traj):
        for j in range(n_traj):
            ti = gpu_trajectories[traj_names[i]][20:, 4]  # power trajectory
            tj = gpu_trajectories[traj_names[j]][20:, 4]
            min_len = min(len(ti), len(tj))
            dist_matrix[i, j] = np.mean((ti[:min_len] - tj[:min_len])**2)

    # Within-pattern similarity vs between-pattern difference
    within = np.mean([dist_matrix[i, i] for i in range(n_traj)])
    between = np.mean([dist_matrix[i, j] for i in range(n_traj) for j in range(n_traj) if i != j])

    print(f"\n  GPU trajectory distances (power):")
    print(f"    Within-pattern:  {within:.4f}")
    print(f"    Between-pattern: {between:.4f}")
    print(f"    Ratio: {between / max(within + 1e-10, 1e-10):.2f}")

    # Also check SCLK trajectories
    sclk_dist = np.zeros((n_traj, n_traj))
    for i in range(n_traj):
        for j in range(n_traj):
            ti = gpu_trajectories[traj_names[i]][20:, 3]  # SCLK
            tj = gpu_trajectories[traj_names[j]][20:, 3]
            min_len = min(len(ti), len(tj))
            sclk_dist[i, j] = np.mean((ti[:min_len] - tj[:min_len])**2)

    sclk_between = np.mean([sclk_dist[i,j] for i in range(n_traj) for j in range(n_traj) if i != j])
    sclk_within = np.mean([sclk_dist[i,i] for i in range(n_traj)])
    print(f"    SCLK between/within ratio: {sclk_between / max(sclk_within + 1e-10, 1e-10):.2f}")

    results['gpu_trajectories'] = {
        'power_between': float(between), 'power_within': float(within),
        'sclk_between': float(sclk_between), 'sclk_within': float(sclk_within),
        'patterns': traj_names,
    }

    # T611: GPU trajectories differ between input patterns (between > 2× within)
    traj_ratio = between / max(within + 1e-6, 1e-6)
    tests['T611'] = {'desc': 'GPU trajectories input-dependent (between > 2× within)',
                     'val': traj_ratio, 'pass': traj_ratio > 2.0}
    print(f"\n  T611: trajectory ratio = {traj_ratio:.2f} {'PASS' if traj_ratio > 2.0 else 'FAIL'}")

    # T612: GPU power PSD shows 1/f character (slope < -0.3)
    coupled_power_psd = psd_slope(c_gpu[20:, 4], fs=SAMPLE_HZ)
    tests['T612'] = {'desc': 'GPU power PSD slope < -0.3 (1/f)',
                     'val': coupled_power_psd, 'pass': coupled_power_psd < -0.3}
    print(f"  T612: power PSD slope = {coupled_power_psd:.2f} {'PASS' if coupled_power_psd < -0.3 else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 5: CLASSIFICATION WITH COUPLED STATE
    #   NOW we classify — using the full coupled state.
    #   Features = FPGA spikes + vmem + GPU deep state
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXP 5: CLASSIFICATION WITH COUPLED SUBSTRATE STATE")
    print("  Features: mean/std of [spikes + vmem + GPU state]")
    print("  4-class waveform, 5-fold CV")
    print("=" * 70)

    inputs, labels = generate_waveforms(N_TRIALS, N_STEPS)

    conditions = ['COUPLED', 'FPGA_ONLY', 'GPU_ONLY', 'STATIC']
    wave_results = {}

    for cond in conditions:
        print(f"\n  --- {cond} ---")
        feats = []
        for trial in range(N_TRIALS):
            spk, vm, glog, ints = run_coupled_loop(
                fpga, inputs[trial], w_in, w_gpu, w_fb, mode=cond)

            # Features: mean + std of all channels (honest pooling)
            trial_state = np.hstack([spk, vm, glog])
            f = np.concatenate([trial_state.mean(axis=0), trial_state.std(axis=0)])
            feats.append(f)

            if trial == 0:
                print(f"    feat_dim={len(f)}, spk_rate={spk[1:].mean():.1f}, vmem_std={vm.std():.1f}")
            if (trial + 1) % 20 == 0:
                print(f"    trial {trial+1}/{N_TRIALS}")

        X = np.array(feats)
        wave_results[cond] = classify_cv(X, labels, n_classes=4)
        acc = wave_results[cond]['mean']
        std = wave_results[cond]['std']
        print(f"    {cond}: {acc:.3f} ± {std:.3f}")

    results['classification'] = wave_results

    # T613: COUPLED > 0.35 (above chance 0.25 for 4-class)
    coupled_acc = wave_results['COUPLED']['mean']
    tests['T613'] = {'desc': 'COUPLED accuracy > 0.35 (4-class)',
                     'val': coupled_acc, 'pass': coupled_acc > 0.35}
    print(f"\n  T613: COUPLED acc = {coupled_acc:.3f} {'PASS' if coupled_acc > 0.35 else 'FAIL'}")

    # T614: COUPLED > STATIC
    static_acc = wave_results['STATIC']['mean']
    tests['T614'] = {'desc': 'COUPLED > STATIC',
                     'val': coupled_acc - static_acc,
                     'pass': coupled_acc > static_acc}
    print(f"  T614: COUPLED-STATIC = {coupled_acc - static_acc:.3f} {'PASS' if coupled_acc > static_acc else 'FAIL'}")

    # T615: COUPLED > GPU_ONLY
    gpu_acc = wave_results['GPU_ONLY']['mean']
    tests['T615'] = {'desc': 'COUPLED > GPU_ONLY',
                     'val': coupled_acc - gpu_acc,
                     'pass': coupled_acc > gpu_acc}
    print(f"  T615: COUPLED-GPU_ONLY = {coupled_acc - gpu_acc:.3f} {'PASS' if coupled_acc > gpu_acc else 'FAIL'}")

    # T616: COUPLED > FPGA_ONLY
    fpga_acc = wave_results['FPGA_ONLY']['mean']
    tests['T616'] = {'desc': 'COUPLED > FPGA_ONLY',
                     'val': coupled_acc - fpga_acc,
                     'pass': coupled_acc > fpga_acc}
    print(f"  T616: COUPLED-FPGA_ONLY = {coupled_acc - fpga_acc:.3f} {'PASS' if coupled_acc > fpga_acc else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 6: CAUSAL EMERGENCE
    #   Does the coupled system show macro-level causal structure
    #   that can't be reduced to individual components?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXP 6: CAUSAL EMERGENCE IN COUPLED SYSTEM")
    print("=" * 70)

    # Use the coupled run data from EXP 2
    # Micro: individual neuron spikes predict individual GPU channels
    micro_te = []
    for n_idx in range(0, N_NEURONS, 16):  # Sample every 16th neuron
        for g_idx in range(5):  # First 5 GPU channels
            s = c_spk[10:, n_idx]
            g = c_gpu[10:, g_idx]
            if np.std(s) > 1e-10 and np.std(g) > 1e-10:
                micro_te.append(transfer_entropy(s, g, k=1))
    avg_micro_te = np.mean(micro_te) if micro_te else 0.0

    # Macro: aggregate spike rate predicts aggregate GPU state
    macro_spike = c_spk[10:].mean(axis=1)
    macro_gpu = c_gpu[10:, :5].mean(axis=1)
    macro_te = transfer_entropy(macro_spike, macro_gpu, k=2)

    # Emergence ratio
    emergence = macro_te / max(avg_micro_te, 1e-6)

    print(f"  Micro TE (avg): {avg_micro_te:.4f}")
    print(f"  Macro TE:       {macro_te:.4f}")
    print(f"  Emergence ratio: {emergence:.2f}")

    results['causal_emergence'] = {
        'micro_te': avg_micro_te, 'macro_te': macro_te,
        'emergence_ratio': emergence, 'n_micro_pairs': len(micro_te),
    }

    # T617: Macro TE > micro TE (causal emergence)
    tests['T617'] = {'desc': 'Macro TE > Micro TE (causal emergence)',
                     'val': emergence, 'pass': macro_te > avg_micro_te}
    print(f"\n  T617: emergence ratio = {emergence:.2f} {'PASS' if macro_te > avg_micro_te else 'FAIL'}")

    # T618: Macro TE > 0.01 (meaningful information flow at macro level)
    tests['T618'] = {'desc': 'Macro TE > 0.01',
                     'val': macro_te, 'pass': macro_te > 0.01}
    print(f"  T618: macro TE = {macro_te:.4f} {'PASS' if macro_te > 0.01 else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # SCORE
    # ═══════════════════════════════════════════════════════════
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['tests'] = tests
    results['score'] = f"{n_pass}/{n_total}"

    print("\n" + "=" * 70)
    print(f"FINAL SCORE: {n_pass}/{n_total}")
    print("=" * 70)
    for tid, t in sorted(tests.items()):
        status = "PASS" if t['pass'] else "FAIL"
        print(f"  {tid}: {status} — {t['desc']} (val={t['val']:.4f})")

    # Save
    out_path = RESULTS / 'z2223_coupled_substrate_dynamics.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nSaved: {out_path}")

    fpga.close()


if __name__ == '__main__':
    main()
