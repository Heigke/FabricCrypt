#!/usr/bin/env python3
"""z2224_deep_coupled_full_stack.py — Full-Stack Deep Coupled Experiment

Extension of z2223 (17/19 PASS). Adds ALL unused deep probes:
  1. Per-core SMN thermal sensors (31 sensors at 0x598A4-0x5991C)
  2. SVI voltage rails (Block B at 0x5B000)
  3. GPU dispatch jitter (ISA-level timing proxy)
  4. gpu_metrics raw (system_clock_counter)
  5. FPGA SET_SYNAPSE for recurrence (CMD 0x04)

Full probe count: 8 base + 31 thermal + 1 SVI + 1 jitter + 1 gpu_metrics = 42 channels

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128 neurons, UDP Ethernet)
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path
from collections import Counter

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters ───
N_NEURONS = 128
BASE_VG = 0.45
ALPHA = 0.35
BETA_POWER = 0.12
BETA_THERMAL = 0.08
BETA_CLOCK = 0.10
BETA_THERMAL_ARRAY = 0.04    # Per-core thermal → Vg (spread across neurons)
BETA_SVI = 0.06              # SVI voltage → Vg
BETA_JITTER = 0.08           # Dispatch jitter → Vg
SAMPLE_HZ = 200
WORKLOAD_MS = 1.5
N_STEPS = 300                # 1.5s per trial at 200Hz
N_STEPS_LONG = 500           # Extended for deep coupling (EXP 3)
N_TRIALS = 60                # 4-class, 5-fold CV
N_TRIALS_LONG = 80           # Extended for EXP 4
N_TRIALS_CHARACTERIZE = 1000 # EXP 1 probe characterization

# Deep probe paths
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
HWMON_TEMP = "/sys/class/hwmon/hwmon7/temp1_input"
HWMON_FREQ = "/sys/class/hwmon/hwmon7/freq1_input"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
GPU_BUSY_PATH = "/sys/class/drm/card0/device/gpu_busy_percent"
GPU_METRICS_PATH = "/sys/class/drm/card0/device/gpu_metrics"

# ─── JSON encoder ───
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# DEEP GPU PROBES — BASE (from z2223)
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

def read_all_gpu_state_base():
    """Read BASE GPU probes (8 channels, same as z2223)."""
    smn = read_smn_adc()
    pm_t, pm_p, pm_sclk = read_pm_table()
    hw_p, hw_t, hw_f = read_hwmon()
    busy = read_gpu_busy()
    return [smn or 0, pm_t or 0, pm_p or 0, pm_sclk or 0,
            hw_p or 0, hw_t or 0, hw_f or 0, busy or 0]


# ═══════════════════════════════════════════════════════════
# NEW DEEP PROBES — z2224 additions
# ═══════════════════════════════════════════════════════════

_smn_thermal_available = None

def read_smn_thermal_array():
    """D7-D37: Per-core thermal sensors — 31 sensors from SMN bank A (0x598A4-0x5991C).
    Returns array of 31 temperatures in degrees C."""
    global _smn_thermal_available
    temps = np.zeros(31)
    try:
        with open(SMN_PATH, 'rb+') as f:
            base = 0x598A4
            for i in range(31):
                addr = base + i * 4
                f.seek(0)
                f.write(struct.pack('<I', addr))
                f.seek(0)
                raw = struct.unpack('<I', f.read(4))[0]
                temps[i] = (raw >> 21) * 0.125
        _smn_thermal_available = True
        return temps
    except:
        _smn_thermal_available = False
        return temps

def read_svi_voltage():
    """D38: SVI Block B GPU core voltage (actual, not requested)."""
    try:
        with open(SMN_PATH, 'rb+') as f:
            f.write(struct.pack('<I', 0x5B000))
            f.seek(0)
            raw = struct.unpack('<I', f.read(4))[0]
        vid = raw & 0xFF
        voltage = 1.55 - vid * 0.00625
        return voltage
    except:
        return None

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

def measure_dispatch_jitter(n_samples=16):
    """D39: GPU kernel dispatch timing jitter — proxy for PERF_SNAPSHOT.
    Captures silicon-level scheduling variance without inline ASM."""
    if not _torch_available:
        return np.zeros(n_samples)
    import torch
    jitters = []
    a = torch.randn(32, 32, device=_torch_device)
    for _ in range(n_samples):
        t0 = time.perf_counter()
        _ = torch.mm(a, a)
        torch.cuda.synchronize()
        jitters.append(time.perf_counter() - t0)
    del a
    return np.array(jitters)

def measure_dispatch_jitter_single():
    """Single dispatch jitter measurement (for in-loop use)."""
    if not _torch_available:
        return 0.0
    import torch
    a = torch.randn(32, 32, device=_torch_device)
    t0 = time.perf_counter()
    _ = torch.mm(a, a)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del a
    return dt

def read_gpu_metrics_raw():
    """D40: Raw gpu_metrics table — system_clock_counter at offset 0x68."""
    try:
        with open(GPU_METRICS_PATH, 'rb') as f:
            data = f.read(264)
        if len(data) >= 0x70:
            sys_clock = struct.unpack_from('<Q', data, 0x68)[0]
            return sys_clock
        return None
    except:
        return None

def read_all_deep_state():
    """Read ALL probes: 8 base + 31 thermal + 1 SVI + 1 jitter + 1 gpu_metrics = 42."""
    base = read_all_gpu_state_base()             # 8 channels
    thermal_array = read_smn_thermal_array()      # 31 channels
    svi = read_svi_voltage()                      # 1 channel
    jitter = measure_dispatch_jitter_single()     # 1 channel
    gpu_met = read_gpu_metrics_raw()              # 1 channel
    return base, thermal_array, svi or 0.0, jitter, gpu_met or 0


# ═══════════════════════════════════════════════════════════
# GPU WORKLOAD — SHORT bursts (from z2223)
# ═══════════════════════════════════════════════════════════

def run_workload(intensity, duration_ms=1.5):
    """Short GPU burst to excite DVFS dynamics. Fire-and-forget (no sync)."""
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
# FPGA RECURRENCE — SET_SYNAPSE
# ═══════════════════════════════════════════════════════════

def enable_fpga_recurrence(fpga, w_recur, connectivity=0.10):
    """Enable sparse recurrent connections via CMD 0x04 SET_SYNAPSE.
    Packet: [0x55][0x04][neuron_id(1)][syn_id(1)][weight_Q16.16(4)]
    Returns number of synapses set."""
    CMD_SET_SYNAPSE = 0x04
    n_set = 0
    for i in range(min(len(w_recur), 128)):
        for j in range(min(w_recur.shape[1] if w_recur.ndim > 1 else 4, 4)):
            w = w_recur[i, j] if w_recur.ndim > 1 else w_recur[i]
            if abs(w) > 0.001:
                w_q16 = int(w * 65536) & 0xFFFFFFFF
                pkt = struct.pack(">BBBBI", 0x55, CMD_SET_SYNAPSE,
                                  i & 0x7F, j & 0x03, w_q16)
                try:
                    fpga._send(pkt)
                    n_set += 1
                except:
                    pass
    return n_set

def disable_fpga_recurrence(fpga):
    """Zero all synapses."""
    CMD_SET_SYNAPSE = 0x04
    for i in range(128):
        for j in range(4):
            pkt = struct.pack(">BBBBI", 0x55, CMD_SET_SYNAPSE,
                              i & 0x7F, j & 0x03, 0)
            try:
                fpga._send(pkt)
            except:
                pass

def generate_sparse_recurrence(n_neurons, connectivity=0.10, strength=0.3, seed=42):
    """Generate sparse recurrence weight matrix (n_neurons x 4 synapses)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_neurons, 4))
    for i in range(n_neurons):
        for j in range(4):
            if rng.random() < connectivity:
                w[i, j] = rng.uniform(-strength, strength)
    return w


# ═══════════════════════════════════════════════════════════
# COUPLED DYNAMICS LOOP — FULL STACK
# ═══════════════════════════════════════════════════════════

def run_coupled_loop(fpga, input_signal, w_in, w_gpu, w_fb,
                     mode='COUPLED', deep=True, n_steps=None):
    """Deep coupled substrate dynamics loop with ALL probes.

    Modes:
      COUPLED:       Full bidirectional (FPGA <-> GPU), all probes
      COUPLED_DEEP:  Like COUPLED but with 31 thermal + SVI + jitter
      FPGA_ONLY:     FPGA driven by input, no GPU coupling
      GPU_ONLY:      GPU driven by input, FPGA records but doesn't couple
      UNCOUPLED:     Both run independently
      STATIC:        Fixed Vg, baseline

    Returns:
      spikes:     (n_steps, N_NEURONS)
      vmem:       (n_steps, N_NEURONS)
      gpu_base:   (n_steps, 8) base GPU readings
      gpu_thermal:(n_steps, 31) per-core thermal
      gpu_extra:  (n_steps, 3) [SVI voltage, dispatch jitter, gpu_metrics]
      intensities:(n_steps,)
    """
    if n_steps is None:
        n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    spikes = np.zeros((n_steps, N_NEURONS))
    vmem_log = np.zeros((n_steps, N_NEURONS))
    gpu_base_log = np.zeros((n_steps, 8))
    gpu_thermal_log = np.zeros((n_steps, 31))
    gpu_extra_log = np.zeros((n_steps, 3))  # SVI, jitter, gpu_metrics
    intensities = np.zeros(n_steps)
    prev_counts = None

    use_deep = deep and mode in ('COUPLED', 'COUPLED_DEEP')

    for t in range(n_steps):
        t_start = time.perf_counter()

        # ── Read ALL GPU state ──
        base = read_all_gpu_state_base()
        gpu_base_log[t] = base

        if use_deep:
            thermal_arr = read_smn_thermal_array()
            gpu_thermal_log[t] = thermal_arr
            svi = read_svi_voltage()
            gpu_extra_log[t, 0] = svi if svi is not None else 0
            # Dispatch jitter only every 10 steps (it calls synchronize)
            if t % 10 == 0:
                gpu_extra_log[t, 1] = measure_dispatch_jitter_single()
            else:
                gpu_extra_log[t, 1] = gpu_extra_log[max(0, t-1), 1]
            gm = read_gpu_metrics_raw()
            gpu_extra_log[t, 2] = float(gm) if gm is not None else 0

        # ── Compute Vg from input + GPU state ──
        vg = np.full(N_NEURONS, BASE_VG)

        if mode in ('COUPLED', 'COUPLED_DEEP', 'FPGA_ONLY'):
            vg += ALPHA * input_signal[min(t, len(input_signal)-1)] * w_in

        if mode in ('COUPLED', 'COUPLED_DEEP'):
            hw_p = gpu_base_log[t, 4]   # hwmon power
            pm_sclk = gpu_base_log[t, 3] # PM SCLK
            pm_t = gpu_base_log[t, 1]    # PM thermal

            # Running baseline
            if t >= 5:
                p_base = gpu_base_log[max(0,t-20):t, 4].mean()
                s_base = gpu_base_log[max(0,t-20):t, 3].mean()
                t_base = gpu_base_log[max(0,t-20):t, 1].mean()
            else:
                p_base, s_base, t_base = hw_p, pm_sclk, pm_t

            p_delta = (hw_p - p_base) / max(abs(p_base), 1.0)
            s_delta = (pm_sclk - s_base) / max(abs(s_base), 1.0)
            t_delta = (pm_t - t_base) / max(abs(t_base), 1.0)

            # Base GPU signals -> different neuron groups
            n3 = N_NEURONS // 3
            vg[:n3] += BETA_POWER * p_delta * w_gpu[:n3]
            vg[n3:2*n3] += BETA_CLOCK * s_delta * w_gpu[n3:2*n3]
            vg[2*n3:] += BETA_THERMAL * t_delta * w_gpu[2*n3:]

            # DEEP: per-core thermal array -> spread across all neurons
            if use_deep and _smn_thermal_available:
                thermal_mean = gpu_thermal_log[t].mean()
                thermal_var = gpu_thermal_log[t] - thermal_mean
                # Map 31 thermal sensors across 128 neurons
                for ni in range(N_NEURONS):
                    ti = ni % 31
                    vg[ni] += BETA_THERMAL_ARRAY * thermal_var[ti] * w_gpu[ni] * 0.01

            # DEEP: SVI voltage feedback
            if use_deep and gpu_extra_log[t, 0] > 0:
                svi_val = gpu_extra_log[t, 0]
                if t >= 5:
                    svi_base = gpu_extra_log[max(0,t-20):t, 0].mean()
                else:
                    svi_base = svi_val
                svi_delta = (svi_val - svi_base) / max(abs(svi_base), 0.01)
                vg += BETA_SVI * svi_delta * w_gpu * 0.1

            # DEEP: dispatch jitter modulation
            if use_deep and gpu_extra_log[t, 1] > 0:
                jitter = gpu_extra_log[t, 1]
                if t >= 10:
                    j_base = gpu_extra_log[max(0,t-20):t, 1].mean()
                else:
                    j_base = jitter
                j_delta = (jitter - j_base) / max(abs(j_base), 1e-6)
                vg += BETA_JITTER * j_delta * w_gpu * 0.05

            # SET_TEMP for physics-level BVpar coupling
            if gpu_base_log[t, 1] > 0:
                try:
                    fpga.set_temp(float(gpu_base_log[t, 1]) + 273.15)
                except:
                    pass

        vg = np.clip(vg, 0.10, 0.85)

        # ── Drive FPGA & read state ──
        if mode != 'GPU_ONLY':
            fpga.set_vg_batch(0, vg.tolist())
            time.sleep(0.0003)  # 300us settle

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

        # ── Spike-driven feedback -> GPU workload ──
        if mode in ('COUPLED', 'COUPLED_DEEP'):
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
            inp_t = input_signal[min(t, len(input_signal)-1)]
            intensity = float(0.2 + 0.6 * np.clip(inp_t, 0, 1))
            run_workload(intensity, duration_ms=WORKLOAD_MS)
        elif mode == 'UNCOUPLED':
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

    return spikes, vmem_log, gpu_base_log, gpu_thermal_log, gpu_extra_log, intensities


# ═══════════════════════════════════════════════════════════
# ANALYSIS — from z2223 + extensions
# ═══════════════════════════════════════════════════════════

def transfer_entropy(source, target, k=1, bins=8):
    """Transfer entropy TE(source->target) using binned estimator."""
    if len(source) < k + 2 or np.std(source) < 1e-10 or np.std(target) < 1e-10:
        return 0.0
    s_bins = np.digitize(source, np.linspace(source.min()-1e-10, source.max()+1e-10, bins+1)) - 1
    t_bins = np.digitize(target, np.linspace(target.min()-1e-10, target.max()+1e-10, bins+1)) - 1
    n = len(source) - k
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
    """Effective dimensionality via participation ratio of singular values."""
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


def acf(x, lag=1):
    """Autocorrelation at given lag."""
    if len(x) < lag + 2 or np.std(x) < 1e-10:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def entropy_bits(x, bins=20):
    """Shannon entropy in bits."""
    hist, _ = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


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


def memory_capacity(spikes_all, max_delay=10):
    """Linear memory capacity: how many past steps can be linearly decoded."""
    n_steps, n_feat = spikes_all.shape
    mc_total = 0.0
    for d in range(1, min(max_delay + 1, n_steps - 10)):
        X = spikes_all[d:, :]
        y = spikes_all[:-d, :].mean(axis=1)  # target = mean input d steps ago
        if np.std(y) < 1e-10:
            continue
        # Ridge regression
        mu = X.mean(axis=0); sigma = X.std(axis=0); sigma[sigma < 1e-2] = 1.0
        Xs = (X - mu) / sigma
        try:
            W = np.linalg.solve(Xs.T @ Xs + 0.01 * np.eye(n_feat), Xs.T @ y)
            y_pred = Xs @ W
            r2 = 1.0 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
            mc_total += max(0.0, r2)
        except:
            pass
    return mc_total


# ═══════════════════════════════════════════════════════════
# WAVEFORM GENERATION (4-class)
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


def build_features_full_stack(spikes, vmem, gpu_base, gpu_thermal, gpu_extra):
    """Build feature vector from ALL channels: spikes+vmem+gpu_base+thermal+extra."""
    # Full state concatenation, then mean+std pooling
    full = np.hstack([spikes, vmem, gpu_base, gpu_thermal, gpu_extra])
    return np.concatenate([full.mean(axis=0), full.std(axis=0)])


def build_features_partial(spikes, vmem, gpu_base):
    """Build features from z2223-equivalent probes only (spikes+vmem+8 GPU)."""
    full = np.hstack([spikes, vmem, gpu_base])
    return np.concatenate([full.mean(axis=0), full.std(axis=0)])


def build_features_fpga_only(spikes, vmem):
    """FPGA only."""
    full = np.hstack([spikes, vmem])
    return np.concatenate([full.mean(axis=0), full.std(axis=0)])


def build_features_gpu_only(gpu_base, gpu_thermal, gpu_extra):
    """GPU only."""
    full = np.hstack([gpu_base, gpu_thermal, gpu_extra])
    return np.concatenate([full.mean(axis=0), full.std(axis=0)])


# ═══════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("z2224: DEEP COUPLED FULL-STACK EXPERIMENT")
    print("  Extension of z2223 (17/19) — adds ALL deep probes:")
    print("  + 31 per-core SMN thermal sensors")
    print("  + SVI voltage rails")
    print("  + GPU dispatch jitter (ISA timing proxy)")
    print("  + gpu_metrics system clock")
    print("  + FPGA SET_SYNAPSE recurrence")
    print("  = 42 GPU channels + 128 FPGA neurons")
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
    base_test = read_all_gpu_state_base()
    print(f"    Base 8 channels: {['smn','pm_t','pm_p','pm_sclk','hw_p','hw_t','hw_f','busy']}")
    print(f"    Values: {[f'{v:.2f}' for v in base_test]}")

    thermal_test = read_smn_thermal_array()
    thermal_ok = _smn_thermal_available
    print(f"    Per-core thermal (31): {'OK' if thermal_ok else 'UNAVAIL'} "
          f"range=[{thermal_test.min():.1f}, {thermal_test.max():.1f}]")

    svi_test = read_svi_voltage()
    print(f"    SVI voltage: {svi_test if svi_test else 'UNAVAIL'}")

    jitter_test = measure_dispatch_jitter(8)
    print(f"    Dispatch jitter: mean={jitter_test.mean()*1e6:.1f}us std={jitter_test.std()*1e6:.1f}us")

    gm_test = read_gpu_metrics_raw()
    print(f"    gpu_metrics sys_clock: {gm_test if gm_test else 'UNAVAIL'}")

    n_channels = 8 + (31 if thermal_ok else 0) + (1 if svi_test else 0) + 1 + (1 if gm_test else 0)
    print(f"\n    Total GPU probe channels: {n_channels}")

    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, N_NEURONS)
    w_gpu = rng.uniform(-1, 1, N_NEURONS)
    w_fb = rng.uniform(-1, 1, N_NEURONS)

    results = {
        'experiment': 'z2224_deep_coupled_full_stack',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'base_from': 'z2223',
            'n_gpu_channels': n_channels,
            'new_probes': ['smn_thermal_array_31', 'svi_voltage', 'dispatch_jitter', 'gpu_metrics'],
            'fpga_recurrence': 'SET_SYNAPSE (CMD 0x04)',
            'sample_hz': SAMPLE_HZ,
            'n_neurons': N_NEURONS,
        }
    }
    tests = {}

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 1: FULL-STACK PROBE CHARACTERIZATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 1: FULL-STACK PROBE CHARACTERIZATION")
    print(f"  Reading ALL probes for {N_TRIALS_CHARACTERIZE} samples at max rate")
    print("=" * 72)

    char_base = np.zeros((N_TRIALS_CHARACTERIZE, 8))
    char_thermal = np.zeros((N_TRIALS_CHARACTERIZE, 31))
    char_svi = np.zeros(N_TRIALS_CHARACTERIZE)
    char_jitter = np.zeros(N_TRIALS_CHARACTERIZE)
    char_gm = np.zeros(N_TRIALS_CHARACTERIZE)
    char_times = np.zeros(N_TRIALS_CHARACTERIZE)

    # Run a light workload to get dynamic readings
    print("  Collecting...")
    for i in range(N_TRIALS_CHARACTERIZE):
        t0 = time.perf_counter()

        # Light workload to excite dynamics
        if i % 5 == 0:
            run_workload(0.3 + 0.4 * np.sin(2 * np.pi * i / 200), duration_ms=1.0)

        char_base[i] = read_all_gpu_state_base()
        char_thermal[i] = read_smn_thermal_array()
        svi = read_svi_voltage()
        char_svi[i] = svi if svi is not None else 0
        if i % 10 == 0:
            char_jitter[i] = measure_dispatch_jitter_single()
        else:
            char_jitter[i] = char_jitter[max(0, i-1)]
        gm = read_gpu_metrics_raw()
        char_gm[i] = float(gm) if gm is not None else 0

        char_times[i] = time.perf_counter() - t0

        if (i + 1) % 200 == 0:
            print(f"    sample {i+1}/{N_TRIALS_CHARACTERIZE} "
                  f"(rate={1.0/max(char_times[i], 1e-6):.0f} Hz)")

    actual_rate = 1.0 / char_times[char_times > 0].mean()
    print(f"  Actual sample rate: {actual_rate:.0f} Hz")

    # Characterize each probe
    probe_stats = {}
    probe_names = ['smn_temp', 'pm_thermal', 'pm_power', 'pm_sclk',
                   'hw_power', 'hw_temp', 'hw_freq', 'gpu_busy']
    for idx, name in enumerate(probe_names):
        x = char_base[:, idx]
        probe_stats[name] = {
            'mean': float(x.mean()), 'std': float(x.std()),
            'acf1': acf(x, 1), 'psd_slope': psd_slope(x, fs=actual_rate),
            'entropy': entropy_bits(x),
        }
        print(f"    {name:15s}: mean={x.mean():.3f} std={x.std():.4f} "
              f"ACF(1)={acf(x,1):.3f} PSD={psd_slope(x, fs=actual_rate):.2f}")

    # Per-core thermal analysis
    thermal_variance = char_thermal.var(axis=0)  # variance of each sensor
    thermal_across_variance = char_thermal.var(axis=1)  # variance across sensors at each time
    thermal_mean_var = float(thermal_variance.mean())
    thermal_cross_var = float(thermal_across_variance.mean())
    print(f"\n    Per-core thermal: mean temporal var={thermal_mean_var:.6f}, "
          f"cross-sensor var={thermal_cross_var:.4f}")
    for si in [0, 10, 20, 30]:
        if si < 31:
            x = char_thermal[:, si]
            print(f"      sensor[{si:2d}]: mean={x.mean():.2f} std={x.std():.4f} "
                  f"ACF(1)={acf(x,1):.3f}")

    # SVI
    svi_stats = {
        'mean': float(char_svi.mean()), 'std': float(char_svi.std()),
        'acf1': acf(char_svi, 1), 'psd_slope': psd_slope(char_svi, fs=actual_rate),
    }
    print(f"    SVI voltage: mean={char_svi.mean():.4f}V std={char_svi.std():.6f} "
          f"ACF(1)={acf(char_svi,1):.3f}")

    # Dispatch jitter
    jitter_nonzero = char_jitter[char_jitter > 0]
    jitter_stats = {
        'mean': float(jitter_nonzero.mean()) if len(jitter_nonzero) > 0 else 0,
        'std': float(jitter_nonzero.std()) if len(jitter_nonzero) > 0 else 0,
        'acf1': acf(jitter_nonzero, 1) if len(jitter_nonzero) > 10 else 0,
    }
    print(f"    Dispatch jitter: mean={jitter_stats['mean']*1e6:.1f}us "
          f"std={jitter_stats['std']*1e6:.1f}us ACF(1)={jitter_stats['acf1']:.3f}")

    # gpu_metrics
    gm_nonzero = char_gm[char_gm > 0]
    gm_ok = len(gm_nonzero) > 0
    print(f"    gpu_metrics: {'OK' if gm_ok else 'UNAVAIL'} "
          f"({len(gm_nonzero)}/{N_TRIALS_CHARACTERIZE} reads)")

    results['probe_characterization'] = {
        'base_probes': probe_stats,
        'thermal_array': {
            'temporal_variance_mean': thermal_mean_var,
            'cross_sensor_variance': thermal_cross_var,
        },
        'svi': svi_stats,
        'jitter': jitter_stats,
        'gpu_metrics_ok': gm_ok,
        'actual_rate_hz': actual_rate,
    }

    # T620: Per-core thermal variance > 0 (sensors NOT all identical)
    tests['T620'] = {'desc': 'Per-core thermal variance > 0 (sensors distinct)',
                     'val': thermal_cross_var, 'pass': thermal_cross_var > 0.0}
    print(f"\n  T620: thermal cross-sensor var = {thermal_cross_var:.6f} "
          f"{'PASS' if thermal_cross_var > 0 else 'FAIL'}")

    # T621: SVI voltage changes under load
    svi_range = float(char_svi.max() - char_svi.min())
    tests['T621'] = {'desc': 'SVI voltage changes under load (range > 0)',
                     'val': svi_range, 'pass': svi_range > 0.0}
    print(f"  T621: SVI range = {svi_range:.6f}V {'PASS' if svi_range > 0 else 'FAIL'}")

    # T622: Dispatch jitter ACF > 0.1 (temporal structure)
    tests['T622'] = {'desc': 'Dispatch jitter ACF(1) > 0.1',
                     'val': jitter_stats['acf1'],
                     'pass': jitter_stats['acf1'] > 0.1}
    print(f"  T622: jitter ACF(1) = {jitter_stats['acf1']:.3f} "
          f"{'PASS' if jitter_stats['acf1'] > 0.1 else 'FAIL'}")

    # T623: Number of distinct probe channels > 10
    tests['T623'] = {'desc': 'Distinct probe channels > 10',
                     'val': n_channels, 'pass': n_channels > 10}
    print(f"  T623: channels = {n_channels} {'PASS' if n_channels > 10 else 'FAIL'}")

    # T624: GPU metrics sys_clock reads successfully
    tests['T624'] = {'desc': 'gpu_metrics sys_clock readable',
                     'val': len(gm_nonzero), 'pass': gm_ok}
    print(f"  T624: gpu_metrics OK = {gm_ok} {'PASS' if gm_ok else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 2: FPGA RECURRENCE VIA SET_SYNAPSE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 2: FPGA RECURRENCE VIA SET_SYNAPSE")
    print("  Sparse recurrent connections (10% connectivity)")
    print("  Compare RECURRENT vs FEEDFORWARD vs STATIC")
    print("=" * 72)

    inputs_rec, labels_rec = generate_waveforms(N_TRIALS, N_STEPS, seed=123)

    # Generate recurrence matrix
    w_recur = generate_sparse_recurrence(N_NEURONS, connectivity=0.10, strength=0.3, seed=99)
    n_nonzero = np.count_nonzero(w_recur)
    print(f"  Recurrence matrix: {n_nonzero} non-zero synapses "
          f"({n_nonzero / (128*4) * 100:.1f}% fill)")

    recurrence_conditions = {
        'RECURRENT': True,
        'FEEDFORWARD': False,
        'STATIC': None,
    }

    rec_results = {}
    rec_spikes_all = {}

    for cond, use_recur in recurrence_conditions.items():
        print(f"\n  --- {cond} ---")

        if use_recur is True:
            n_set = enable_fpga_recurrence(fpga, w_recur)
            print(f"    SET_SYNAPSE: {n_set} synapses programmed")
            time.sleep(0.3)
        elif use_recur is False:
            disable_fpga_recurrence(fpga)
            time.sleep(0.3)

        mode = 'COUPLED' if cond != 'STATIC' else 'STATIC'
        feats = []
        spk_all = []
        for trial in range(N_TRIALS):
            spk, vm, gb, gt, ge, ints = run_coupled_loop(
                fpga, inputs_rec[trial], w_in, w_gpu, w_fb,
                mode=mode, deep=False)
            f = build_features_partial(spk, vm, gb)
            feats.append(f)
            spk_all.append(spk)
            if (trial + 1) % 20 == 0:
                print(f"    trial {trial+1}/{N_TRIALS}")

        X = np.array(feats)
        rec_results[cond] = classify_cv(X, labels_rec, n_classes=4)
        rec_spikes_all[cond] = spk_all
        print(f"    {cond}: {rec_results[cond]['mean']:.3f} +/- {rec_results[cond]['std']:.3f}")

    # Compute memory capacity for recurrent vs feedforward
    mc_recurrent = 0.0
    mc_feedforward = 0.0
    for trial_idx in range(min(10, N_TRIALS)):
        if 'RECURRENT' in rec_spikes_all:
            mc_recurrent += memory_capacity(rec_spikes_all['RECURRENT'][trial_idx])
        if 'FEEDFORWARD' in rec_spikes_all:
            mc_feedforward += memory_capacity(rec_spikes_all['FEEDFORWARD'][trial_idx])
    mc_recurrent /= min(10, N_TRIALS)
    mc_feedforward /= min(10, N_TRIALS)
    print(f"\n  Memory capacity: RECURRENT={mc_recurrent:.2f}, FEEDFORWARD={mc_feedforward:.2f}")

    # ACF for recurrent vs feedforward
    acf_recurrent = []
    acf_feedforward = []
    for trial_idx in range(min(10, N_TRIALS)):
        if 'RECURRENT' in rec_spikes_all:
            s = rec_spikes_all['RECURRENT'][trial_idx].mean(axis=1)
            acf_recurrent.append(acf(s, 1))
        if 'FEEDFORWARD' in rec_spikes_all:
            s = rec_spikes_all['FEEDFORWARD'][trial_idx].mean(axis=1)
            acf_feedforward.append(acf(s, 1))
    mean_acf_rec = float(np.mean(acf_recurrent)) if acf_recurrent else 0
    mean_acf_ff = float(np.mean(acf_feedforward)) if acf_feedforward else 0
    print(f"  Mean ACF(1): RECURRENT={mean_acf_rec:.3f}, FEEDFORWARD={mean_acf_ff:.3f}")

    # Clean up recurrence
    disable_fpga_recurrence(fpga)
    time.sleep(0.3)

    results['recurrence'] = {
        'classification': rec_results,
        'memory_capacity': {'recurrent': mc_recurrent, 'feedforward': mc_feedforward},
        'acf': {'recurrent': mean_acf_rec, 'feedforward': mean_acf_ff},
        'n_synapses': n_nonzero,
    }

    # T625: RECURRENT > FEEDFORWARD
    rec_acc = rec_results['RECURRENT']['mean']
    ff_acc = rec_results['FEEDFORWARD']['mean']
    tests['T625'] = {'desc': 'RECURRENT > FEEDFORWARD accuracy',
                     'val': rec_acc - ff_acc, 'pass': rec_acc > ff_acc}
    print(f"\n  T625: RECURRENT-FEEDFORWARD = {rec_acc - ff_acc:.3f} "
          f"{'PASS' if rec_acc > ff_acc else 'FAIL'}")

    # T626: RECURRENT > 0.50 (4-class)
    tests['T626'] = {'desc': 'RECURRENT accuracy > 0.50 (4-class)',
                     'val': rec_acc, 'pass': rec_acc > 0.50}
    print(f"  T626: RECURRENT acc = {rec_acc:.3f} {'PASS' if rec_acc > 0.50 else 'FAIL'}")

    # T627: Memory capacity RECURRENT > FEEDFORWARD
    tests['T627'] = {'desc': 'MC RECURRENT > FEEDFORWARD',
                     'val': mc_recurrent - mc_feedforward,
                     'pass': mc_recurrent > mc_feedforward}
    print(f"  T627: MC diff = {mc_recurrent - mc_feedforward:.3f} "
          f"{'PASS' if mc_recurrent > mc_feedforward else 'FAIL'}")

    # T628: Recurrent ACF > Feedforward ACF
    tests['T628'] = {'desc': 'Recurrent ACF(1) > Feedforward ACF(1)',
                     'val': mean_acf_rec - mean_acf_ff,
                     'pass': mean_acf_rec > mean_acf_ff}
    print(f"  T628: ACF diff = {mean_acf_rec - mean_acf_ff:.3f} "
          f"{'PASS' if mean_acf_rec > mean_acf_ff else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 3: DEEPENED BIDIRECTIONAL COUPLING
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 3: DEEPENED BIDIRECTIONAL COUPLING")
    print(f"  ALL probes (31 thermal + SVI + jitter + FPGA recurrence)")
    print(f"  {N_STEPS_LONG} steps, extended analysis")
    print("=" * 72)

    # Re-enable recurrence for deep coupling test
    n_set = enable_fpga_recurrence(fpga, w_recur)
    print(f"  Re-enabled {n_set} recurrent synapses")
    time.sleep(0.3)

    test_input_long = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N_STEPS_LONG) / 60)

    print("\n  Running COUPLED_DEEP (with recurrence)...")
    cd_spk, cd_vm, cd_gb, cd_gt, cd_ge, cd_int = run_coupled_loop(
        fpga, test_input_long, w_in, w_gpu, w_fb,
        mode='COUPLED', deep=True, n_steps=N_STEPS_LONG)
    print(f"    spikes/step: {cd_spk[1:].mean():.1f}")

    # Disable recurrence for comparison
    disable_fpga_recurrence(fpga)
    time.sleep(0.3)

    print("  Running COUPLED (no recurrence, deep probes)...")
    cn_spk, cn_vm, cn_gb, cn_gt, cn_ge, cn_int = run_coupled_loop(
        fpga, test_input_long, w_in, w_gpu, w_fb,
        mode='COUPLED', deep=True, n_steps=N_STEPS_LONG)
    print(f"    spikes/step: {cn_spk[1:].mean():.1f}")

    print("  Running UNCOUPLED...")
    u_spk, u_vm, u_gb, u_gt, u_ge, u_int = run_coupled_loop(
        fpga, test_input_long, w_in, w_gpu, w_fb,
        mode='UNCOUPLED', deep=False, n_steps=N_STEPS_LONG)
    print(f"    spikes/step: {u_spk[1:].mean():.1f}")

    skip = 20  # skip transient

    # Aggregate signals for TE
    cd_spike_mean = cd_spk[skip:].mean(axis=1)
    cd_power = cd_gb[skip:, 4]
    cn_spike_mean = cn_spk[skip:].mean(axis=1)
    cn_power = cn_gb[skip:, 4]
    u_spike_mean = u_spk[skip:].mean(axis=1)
    u_power = u_gb[skip:, 4]

    # TE FPGA->GPU
    te_fg_deep = transfer_entropy(cd_spike_mean, cd_power, k=2)
    te_fg_nrec = transfer_entropy(cn_spike_mean, cn_power, k=2)
    te_fg_uncoup = transfer_entropy(u_spike_mean, u_power, k=2)

    # TE GPU->FPGA
    te_gf_deep = transfer_entropy(cd_power, cd_spike_mean, k=2)
    te_gf_nrec = transfer_entropy(cn_power, cn_spike_mean, k=2)
    te_gf_uncoup = transfer_entropy(u_power, u_spike_mean, k=2)

    # MI
    mi_deep = mutual_information(cd_spike_mean, cd_power)
    mi_nrec = mutual_information(cn_spike_mean, cn_power)
    mi_uncoup = mutual_information(u_spike_mean, u_power)

    print(f"\n  TE(FPGA->GPU): deep={te_fg_deep:.4f}, no-rec={te_fg_nrec:.4f}, uncoup={te_fg_uncoup:.4f}")
    print(f"  TE(GPU->FPGA): deep={te_gf_deep:.4f}, no-rec={te_gf_nrec:.4f}, uncoup={te_gf_uncoup:.4f}")
    print(f"  MI:            deep={mi_deep:.4f}, no-rec={mi_nrec:.4f}, uncoup={mi_uncoup:.4f}")

    # Per-core thermal TE
    thermal_te_sum = 0.0
    thermal_te_count = 0
    for si in range(0, 31, 3):  # Sample every 3rd sensor
        th = cd_gt[skip:, si]
        if np.std(th) > 1e-10:
            te_th = transfer_entropy(th, cd_spike_mean, k=1)
            thermal_te_sum += te_th
            thermal_te_count += 1
    avg_thermal_te = thermal_te_sum / max(thermal_te_count, 1)

    # Aggregate thermal TE
    agg_thermal = cd_gt[skip:].mean(axis=1)
    agg_thermal_te = transfer_entropy(agg_thermal, cd_spike_mean, k=1) if np.std(agg_thermal) > 1e-10 else 0

    print(f"  Per-core thermal TE (avg): {avg_thermal_te:.4f}")
    print(f"  Aggregate thermal TE:      {agg_thermal_te:.4f}")

    # SVI TE
    svi_series = cd_ge[skip:, 0]
    svi_te = transfer_entropy(svi_series, cd_spike_mean, k=1) if np.std(svi_series) > 1e-10 else 0
    print(f"  SVI voltage TE:            {svi_te:.4f}")

    results['bidirectional_coupling'] = {
        'TE_FPGA_GPU_deep': te_fg_deep, 'TE_FPGA_GPU_norec': te_fg_nrec,
        'TE_GPU_FPGA_deep': te_gf_deep, 'TE_GPU_FPGA_norec': te_gf_nrec,
        'MI_deep': mi_deep, 'MI_norec': mi_nrec, 'MI_uncoup': mi_uncoup,
        'per_core_thermal_te': avg_thermal_te,
        'aggregate_thermal_te': agg_thermal_te,
        'svi_te': svi_te,
    }

    # z2223 reference values
    z2223_te_fg = 0.095
    z2223_te_gf = 0.633
    z2223_mi = 0.168

    # T629: TE(FPGA->GPU) with full probes > z2223 value
    tests['T629'] = {'desc': f'TE(FPGA->GPU) full > z2223 ({z2223_te_fg})',
                     'val': te_fg_deep, 'pass': te_fg_deep > z2223_te_fg}
    print(f"\n  T629: TE(FPGA->GPU) = {te_fg_deep:.4f} vs z2223={z2223_te_fg} "
          f"{'PASS' if te_fg_deep > z2223_te_fg else 'FAIL'}")

    # T630: TE(GPU->FPGA) with full probes > z2223 value
    tests['T630'] = {'desc': f'TE(GPU->FPGA) full > z2223 ({z2223_te_gf})',
                     'val': te_gf_deep, 'pass': te_gf_deep > z2223_te_gf}
    print(f"  T630: TE(GPU->FPGA) = {te_gf_deep:.4f} vs z2223={z2223_te_gf} "
          f"{'PASS' if te_gf_deep > z2223_te_gf else 'FAIL'}")

    # T631: MI with full probes > z2223 value
    tests['T631'] = {'desc': f'MI full > z2223 ({z2223_mi})',
                     'val': mi_deep, 'pass': mi_deep > z2223_mi}
    print(f"  T631: MI = {mi_deep:.4f} vs z2223={z2223_mi} "
          f"{'PASS' if mi_deep > z2223_mi else 'FAIL'}")

    # T632: Per-core thermal TE > aggregate thermal TE
    tests['T632'] = {'desc': 'Per-core thermal TE > aggregate thermal TE',
                     'val': avg_thermal_te - agg_thermal_te,
                     'pass': avg_thermal_te > agg_thermal_te}
    print(f"  T632: per-core({avg_thermal_te:.4f}) vs agg({agg_thermal_te:.4f}) "
          f"{'PASS' if avg_thermal_te > agg_thermal_te else 'FAIL'}")

    # T633: SVI voltage TE > 0.01
    tests['T633'] = {'desc': 'SVI voltage TE > 0.01',
                     'val': svi_te, 'pass': svi_te > 0.01}
    print(f"  T633: SVI TE = {svi_te:.4f} {'PASS' if svi_te > 0.01 else 'FAIL'}")

    # T634: Bidirectional TE with recurrence > without
    te_bidir_rec = te_fg_deep + te_gf_deep
    te_bidir_norec = te_fg_nrec + te_gf_nrec
    tests['T634'] = {'desc': 'Bidirectional TE with recurrence > without',
                     'val': te_bidir_rec - te_bidir_norec,
                     'pass': te_bidir_rec > te_bidir_norec}
    print(f"  T634: bidir TE rec={te_bidir_rec:.4f} > norec={te_bidir_norec:.4f} "
          f"{'PASS' if te_bidir_rec > te_bidir_norec else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 4: FULL-STACK CLASSIFICATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 4: FULL-STACK CLASSIFICATION")
    print(f"  Features: spikes + vmem + ALL GPU probes ({n_channels} channels)")
    print(f"  4-class waveform, {N_TRIALS_LONG} trials, 5-fold CV")
    print("=" * 72)

    inputs_cls, labels_cls = generate_waveforms(N_TRIALS_LONG, N_STEPS, seed=777)

    # FULL_STACK: all probes, deep=True
    print("\n  --- FULL_STACK ---")
    feats_full = []
    for trial in range(N_TRIALS_LONG):
        spk, vm, gb, gt, ge, ints = run_coupled_loop(
            fpga, inputs_cls[trial], w_in, w_gpu, w_fb,
            mode='COUPLED', deep=True)
        f = build_features_full_stack(spk, vm, gb, gt, ge)
        feats_full.append(f)
        if (trial + 1) % 20 == 0:
            print(f"    trial {trial+1}/{N_TRIALS_LONG}")
    X_full = np.array(feats_full)
    cls_full = classify_cv(X_full, labels_cls, n_classes=4)
    print(f"    FULL_STACK: {cls_full['mean']:.3f} +/- {cls_full['std']:.3f}")

    # PARTIAL (z2223-equivalent): base GPU only, deep=False
    print("\n  --- PARTIAL (z2223-equiv) ---")
    feats_partial = []
    for trial in range(N_TRIALS_LONG):
        spk, vm, gb, gt, ge, ints = run_coupled_loop(
            fpga, inputs_cls[trial], w_in, w_gpu, w_fb,
            mode='COUPLED', deep=False)
        f = build_features_partial(spk, vm, gb)
        feats_partial.append(f)
        if (trial + 1) % 20 == 0:
            print(f"    trial {trial+1}/{N_TRIALS_LONG}")
    X_partial = np.array(feats_partial)
    cls_partial = classify_cv(X_partial, labels_cls, n_classes=4)
    print(f"    PARTIAL: {cls_partial['mean']:.3f} +/- {cls_partial['std']:.3f}")

    # FPGA_ONLY
    print("\n  --- FPGA_ONLY ---")
    feats_fpga = []
    for trial in range(N_TRIALS_LONG):
        spk, vm, gb, gt, ge, ints = run_coupled_loop(
            fpga, inputs_cls[trial], w_in, w_gpu, w_fb,
            mode='FPGA_ONLY', deep=False)
        f = build_features_fpga_only(spk, vm)
        feats_fpga.append(f)
        if (trial + 1) % 20 == 0:
            print(f"    trial {trial+1}/{N_TRIALS_LONG}")
    X_fpga = np.array(feats_fpga)
    cls_fpga = classify_cv(X_fpga, labels_cls, n_classes=4)
    print(f"    FPGA_ONLY: {cls_fpga['mean']:.3f} +/- {cls_fpga['std']:.3f}")

    # GPU_ONLY (need deep reads even though FPGA not in loop)
    print("\n  --- GPU_ONLY ---")
    feats_gpu = []
    for trial in range(N_TRIALS_LONG):
        spk, vm, gb, gt, ge, ints = run_coupled_loop(
            fpga, inputs_cls[trial], w_in, w_gpu, w_fb,
            mode='GPU_ONLY', deep=False)
        f = build_features_gpu_only(gb, gt, ge)
        feats_gpu.append(f)
        if (trial + 1) % 20 == 0:
            print(f"    trial {trial+1}/{N_TRIALS_LONG}")
    X_gpu = np.array(feats_gpu)
    cls_gpu = classify_cv(X_gpu, labels_cls, n_classes=4)
    print(f"    GPU_ONLY: {cls_gpu['mean']:.3f} +/- {cls_gpu['std']:.3f}")

    results['classification'] = {
        'FULL_STACK': cls_full, 'PARTIAL': cls_partial,
        'FPGA_ONLY': cls_fpga, 'GPU_ONLY': cls_gpu,
    }

    # Ablation: full minus thermal array
    print("\n  --- Ablation: no thermal array ---")
    feats_no_thermal = []
    for i in range(len(feats_full)):
        # Rebuild features without thermal: use partial + extra only
        f = feats_full[i].copy()  # will compare directly
        feats_no_thermal.append(f)
    # Actually need to zero out thermal columns. Features = mean+std of [spk(128)+vm(128)+gb(8)+gt(31)+ge(3)]
    # thermal is columns 264:295 in the state, so in features it's mean[264:295] + std[264:295]
    dim_state = 128 + 128 + 8 + 31 + 3  # = 298
    X_no_thermal = X_full.copy()
    # Zero the thermal mean and std features
    X_no_thermal[:, 264:295] = 0  # mean of thermal
    X_no_thermal[:, dim_state + 264: dim_state + 295] = 0  # std of thermal
    cls_no_thermal = classify_cv(X_no_thermal, labels_cls, n_classes=4)
    print(f"    NO_THERMAL: {cls_no_thermal['mean']:.3f}")

    # Ablation: full minus SVI
    X_no_svi = X_full.copy()
    X_no_svi[:, 295] = 0  # mean of SVI
    X_no_svi[:, dim_state + 295] = 0  # std of SVI
    cls_no_svi = classify_cv(X_no_svi, labels_cls, n_classes=4)
    print(f"    NO_SVI: {cls_no_svi['mean']:.3f}")

    results['ablation'] = {
        'no_thermal': cls_no_thermal,
        'no_svi': cls_no_svi,
    }

    z2223_coupled_acc = 0.654  # z2223 reference

    # T635: FULL_STACK > z2223 COUPLED
    full_acc = cls_full['mean']
    tests['T635'] = {'desc': f'FULL_STACK > z2223 COUPLED ({z2223_coupled_acc})',
                     'val': full_acc, 'pass': full_acc > z2223_coupled_acc}
    print(f"\n  T635: FULL_STACK={full_acc:.3f} vs z2223={z2223_coupled_acc} "
          f"{'PASS' if full_acc > z2223_coupled_acc else 'FAIL'}")

    # T636: FULL_STACK > FPGA_ONLY
    tests['T636'] = {'desc': 'FULL_STACK > FPGA_ONLY',
                     'val': full_acc - cls_fpga['mean'],
                     'pass': full_acc > cls_fpga['mean']}
    print(f"  T636: FULL-FPGA = {full_acc - cls_fpga['mean']:.3f} "
          f"{'PASS' if full_acc > cls_fpga['mean'] else 'FAIL'}")

    # T637: FULL_STACK > GPU_ONLY
    tests['T637'] = {'desc': 'FULL_STACK > GPU_ONLY',
                     'val': full_acc - cls_gpu['mean'],
                     'pass': full_acc > cls_gpu['mean']}
    print(f"  T637: FULL-GPU = {full_acc - cls_gpu['mean']:.3f} "
          f"{'PASS' if full_acc > cls_gpu['mean'] else 'FAIL'}")

    # T638: FULL_STACK > PARTIAL
    tests['T638'] = {'desc': 'FULL_STACK > PARTIAL (z2223 probes)',
                     'val': full_acc - cls_partial['mean'],
                     'pass': full_acc > cls_partial['mean']}
    print(f"  T638: FULL-PARTIAL = {full_acc - cls_partial['mean']:.3f} "
          f"{'PASS' if full_acc > cls_partial['mean'] else 'FAIL'}")

    # T639: Removing thermal array hurts > 2pp
    thermal_ablation = full_acc - cls_no_thermal['mean']
    tests['T639'] = {'desc': 'Removing thermal array hurts > 2pp',
                     'val': thermal_ablation, 'pass': thermal_ablation > 0.02}
    print(f"  T639: thermal ablation = {thermal_ablation:.3f} "
          f"{'PASS' if thermal_ablation > 0.02 else 'FAIL'}")

    # T640: Removing SVI hurts > 1pp
    svi_ablation = full_acc - cls_no_svi['mean']
    tests['T640'] = {'desc': 'Removing SVI hurts > 1pp',
                     'val': svi_ablation, 'pass': svi_ablation > 0.01}
    print(f"  T640: SVI ablation = {svi_ablation:.3f} "
          f"{'PASS' if svi_ablation > 0.01 else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 5: CAUSAL EMERGENCE AT SCALE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 5: CAUSAL EMERGENCE AT SCALE")
    print("  100 micro pairs + per-core thermal as additional channels")
    print("=" * 72)

    # Use the deep coupled data from EXP 3
    # Micro: individual neuron -> individual GPU channel (100 pairs)
    micro_te_list = []
    for n_idx in range(0, N_NEURONS, 2):  # Every 2nd neuron = 64 neurons
        for g_idx in range(min(5, 8)):  # 5 GPU base channels
            if len(micro_te_list) >= 100:
                break
            s = cd_spk[skip:, n_idx]
            g = cd_gb[skip:, g_idx]
            if np.std(s) > 1e-10 and np.std(g) > 1e-10:
                micro_te_list.append(transfer_entropy(s, g, k=1))
        if len(micro_te_list) >= 100:
            break

    # Add per-core thermal as micro channels
    for si in range(31):
        for n_idx in [0, 32, 64, 96]:
            s = cd_spk[skip:, n_idx]
            th = cd_gt[skip:, si]
            if np.std(s) > 1e-10 and np.std(th) > 1e-10:
                micro_te_list.append(transfer_entropy(s, th, k=1))

    avg_micro_te = float(np.mean(micro_te_list)) if micro_te_list else 0.0
    print(f"  Micro TE ({len(micro_te_list)} pairs): mean={avg_micro_te:.4f}")

    # Macro: aggregate spike -> aggregate GPU+thermal
    macro_spike = cd_spk[skip:].mean(axis=1)
    macro_gpu = cd_gb[skip:, :5].mean(axis=1)
    macro_thermal = cd_gt[skip:].mean(axis=1)

    macro_te_gpu = transfer_entropy(macro_spike, macro_gpu, k=2)
    macro_te_thermal = transfer_entropy(macro_spike, macro_thermal, k=2) if np.std(macro_thermal) > 1e-10 else 0
    macro_te = macro_te_gpu + macro_te_thermal

    emergence = macro_te / max(avg_micro_te, 1e-6)
    print(f"  Macro TE (GPU): {macro_te_gpu:.4f}")
    print(f"  Macro TE (thermal): {macro_te_thermal:.4f}")
    print(f"  Macro TE (total): {macro_te:.4f}")
    print(f"  Emergence ratio: {emergence:.2f}")

    # Emergence with vs without thermal array
    micro_te_no_thermal = [te for te in micro_te_list[:100]]  # Only base GPU pairs
    avg_micro_no_thermal = float(np.mean(micro_te_no_thermal)) if micro_te_no_thermal else 0
    emergence_no_thermal = macro_te_gpu / max(avg_micro_no_thermal, 1e-6)
    print(f"  Emergence (no thermal): {emergence_no_thermal:.2f}")

    # Emergence with recurrence (from EXP 3 deep data) vs without (cn_* data)
    cn_macro_spike = cn_spk[skip:].mean(axis=1)
    cn_macro_gpu = cn_gb[skip:, :5].mean(axis=1)
    cn_macro_te = transfer_entropy(cn_macro_spike, cn_macro_gpu, k=2)
    cn_micro_te = []
    for n_idx in range(0, N_NEURONS, 8):
        for g_idx in range(5):
            if len(cn_micro_te) >= 40:
                break
            s = cn_spk[skip:, n_idx]
            g = cn_gb[skip:, g_idx]
            if np.std(s) > 1e-10 and np.std(g) > 1e-10:
                cn_micro_te.append(transfer_entropy(s, g, k=1))
        if len(cn_micro_te) >= 40:
            break
    cn_avg_micro = float(np.mean(cn_micro_te)) if cn_micro_te else 0
    emergence_norec = cn_macro_te / max(cn_avg_micro, 1e-6)
    print(f"  Emergence (no recurrence): {emergence_norec:.2f}")

    results['causal_emergence'] = {
        'micro_te_avg': avg_micro_te, 'n_micro_pairs': len(micro_te_list),
        'macro_te_gpu': macro_te_gpu, 'macro_te_thermal': macro_te_thermal,
        'macro_te_total': macro_te, 'emergence_ratio': emergence,
        'emergence_no_thermal': emergence_no_thermal,
        'emergence_no_recurrence': emergence_norec,
    }

    z2223_emergence = 3.05

    # T641: Emergence ratio > z2223
    tests['T641'] = {'desc': f'Emergence ratio > z2223 ({z2223_emergence})',
                     'val': emergence, 'pass': emergence > z2223_emergence}
    print(f"\n  T641: emergence={emergence:.2f} vs z2223={z2223_emergence} "
          f"{'PASS' if emergence > z2223_emergence else 'FAIL'}")

    # T642: Macro TE > 0.05
    tests['T642'] = {'desc': 'Macro TE > 0.05',
                     'val': macro_te, 'pass': macro_te > 0.05}
    print(f"  T642: macro TE = {macro_te:.4f} {'PASS' if macro_te > 0.05 else 'FAIL'}")

    # T643: Adding thermal array increases emergence
    tests['T643'] = {'desc': 'Thermal array increases emergence',
                     'val': emergence - emergence_no_thermal,
                     'pass': emergence > emergence_no_thermal}
    print(f"  T643: emergence({emergence:.2f}) vs no-thermal({emergence_no_thermal:.2f}) "
          f"{'PASS' if emergence > emergence_no_thermal else 'FAIL'}")

    # T644: Emergence with recurrence > without
    tests['T644'] = {'desc': 'Emergence with recurrence > without',
                     'val': emergence - emergence_norec,
                     'pass': emergence > emergence_norec}
    print(f"  T644: rec({emergence:.2f}) vs norec({emergence_norec:.2f}) "
          f"{'PASS' if emergence > emergence_norec else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 6: HONEST SUBSTRATE SCALING
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 6: HONEST SUBSTRATE SCALING")
    print("  SHALLOW (hwmon 3ch) vs MEDIUM (8ch) vs DEEP (42ch)")
    print("=" * 72)

    # Reuse EXP 4 data but build features at different depths
    # Need to re-collect with deep=True for all, then select features

    # We already have FULL_STACK features from EXP 4
    # Build SHALLOW and MEDIUM from the same runs

    # Actually, we need the raw data. Let's collect once with deep=True
    # and then select features at different depths.
    print("\n  Collecting data for 3-level comparison...")
    inputs_scale, labels_scale = generate_waveforms(N_TRIALS, N_STEPS, seed=999)

    all_spk = []
    all_vm = []
    all_gb = []
    all_gt = []
    all_ge = []

    for trial in range(N_TRIALS):
        spk, vm, gb, gt, ge, ints = run_coupled_loop(
            fpga, inputs_scale[trial], w_in, w_gpu, w_fb,
            mode='COUPLED', deep=True)
        all_spk.append(spk)
        all_vm.append(vm)
        all_gb.append(gb)
        all_gt.append(gt)
        all_ge.append(ge)
        if (trial + 1) % 20 == 0:
            print(f"    trial {trial+1}/{N_TRIALS}")

    # SHALLOW: hwmon only (power, temp, freq) = 3 channels
    print("\n  Building features at 3 depth levels...")
    X_shallow = []
    for i in range(N_TRIALS):
        state = np.hstack([all_spk[i], all_vm[i], all_gb[i][:, 4:7]])  # hw_p, hw_t, hw_f
        X_shallow.append(np.concatenate([state.mean(axis=0), state.std(axis=0)]))
    X_shallow = np.array(X_shallow)

    # MEDIUM: all 8 base channels (= z2223)
    X_medium = []
    for i in range(N_TRIALS):
        state = np.hstack([all_spk[i], all_vm[i], all_gb[i]])
        X_medium.append(np.concatenate([state.mean(axis=0), state.std(axis=0)]))
    X_medium = np.array(X_medium)

    # DEEP: all channels
    X_deep = []
    for i in range(N_TRIALS):
        f = build_features_full_stack(all_spk[i], all_vm[i], all_gb[i], all_gt[i], all_ge[i])
        X_deep.append(f)
    X_deep = np.array(X_deep)

    cls_shallow = classify_cv(X_shallow, labels_scale, n_classes=4)
    cls_medium = classify_cv(X_medium, labels_scale, n_classes=4)
    cls_deep = classify_cv(X_deep, labels_scale, n_classes=4)

    print(f"    SHALLOW (3ch): {cls_shallow['mean']:.3f} +/- {cls_shallow['std']:.3f}")
    print(f"    MEDIUM  (8ch): {cls_medium['mean']:.3f} +/- {cls_medium['std']:.3f}")
    print(f"    DEEP   ({n_channels}ch): {cls_deep['mean']:.3f} +/- {cls_deep['std']:.3f}")

    # TE at different depths
    # Use last trial data for TE comparison
    last_spk = all_spk[-1][skip:].mean(axis=1)
    last_hw_p = all_gb[-1][skip:, 4]
    last_thermal_agg = all_gt[-1][skip:].mean(axis=1)

    te_shallow = transfer_entropy(last_spk, last_hw_p, k=1)
    te_medium = transfer_entropy(last_spk, all_gb[-1][skip:, :5].mean(axis=1), k=2)
    # Deep TE: include thermal
    deep_target = np.hstack([all_gb[-1][skip:, :5], all_gt[-1][skip:, ::5]])
    te_deep = transfer_entropy(last_spk, deep_target.mean(axis=1), k=2) if deep_target.shape[1] > 0 else 0

    print(f"\n    TE: SHALLOW={te_shallow:.4f}, MEDIUM={te_medium:.4f}, DEEP={te_deep:.4f}")

    results['scaling'] = {
        'SHALLOW': cls_shallow, 'MEDIUM': cls_medium, 'DEEP': cls_deep,
        'TE_shallow': te_shallow, 'TE_medium': te_medium, 'TE_deep': te_deep,
    }

    shallow_acc = cls_shallow['mean']
    medium_acc = cls_medium['mean']
    deep_acc = cls_deep['mean']

    # T645: DEEP > MEDIUM > SHALLOW (monotonic)
    monotonic = (deep_acc > medium_acc) and (medium_acc > shallow_acc)
    tests['T645'] = {'desc': 'DEEP > MEDIUM > SHALLOW (monotonic)',
                     'val': f'{deep_acc:.3f}>{medium_acc:.3f}>{shallow_acc:.3f}',
                     'pass': monotonic}
    print(f"\n  T645: {deep_acc:.3f} > {medium_acc:.3f} > {shallow_acc:.3f} "
          f"{'PASS' if monotonic else 'FAIL'}")

    # T646: Each layer adds > 1pp
    gap1 = medium_acc - shallow_acc
    gap2 = deep_acc - medium_acc
    tests['T646'] = {'desc': 'Each probe layer adds > 1pp',
                     'val': min(gap1, gap2),
                     'pass': gap1 > 0.01 and gap2 > 0.01}
    print(f"  T646: gaps = {gap1:.3f}, {gap2:.3f} "
          f"{'PASS' if gap1 > 0.01 and gap2 > 0.01 else 'FAIL'}")

    # T647: DEEP accuracy > 0.70
    tests['T647'] = {'desc': 'DEEP accuracy > 0.70 (4-class)',
                     'val': deep_acc, 'pass': deep_acc > 0.70}
    print(f"  T647: DEEP = {deep_acc:.3f} {'PASS' if deep_acc > 0.70 else 'FAIL'}")

    # T648: DEEP TE > SHALLOW TE
    tests['T648'] = {'desc': 'DEEP TE > SHALLOW TE',
                     'val': te_deep - te_shallow,
                     'pass': te_deep > te_shallow}
    print(f"  T648: TE DEEP={te_deep:.4f} > SHALLOW={te_shallow:.4f} "
          f"{'PASS' if te_deep > te_shallow else 'FAIL'}")

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
        val_str = f"{t['val']}" if isinstance(t['val'], str) else f"{t['val']:.4f}"
        print(f"  {tid}: {status} -- {t['desc']} (val={val_str})")

    # Save
    out_path = RESULTS / 'z2224_deep_coupled_full_stack.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nSaved: {out_path}")

    fpga.close()


if __name__ == '__main__':
    main()
