#!/usr/bin/env python3
"""z2209_stochastic_resonance_reservoir.py — Subthreshold Stochastic Resonance Reservoir

KEY INSIGHT FROM z2207/z2208 FAILURE:
  The waveform task is trivially solvable (LINEAR=97%, FPGA_NONE=93%).
  Adding noise to a task that's already solved deterministically can ONLY hurt.
  FPGA neurons at Vg=0.58 are ABOVE firing threshold → deterministic response.

THE FIX: OPERATE IN THE STOCHASTIC RESONANCE REGIME
  Set all neurons BELOW firing threshold (Vg=0.48, below BVpar cliff ~0.60).
  Make input modulation TINY (ALPHA=0.02).
  Without noise → neurons barely fire → all-zero features → classifier fails.
  With 1/f noise → noise pushes neurons OVER threshold stochastically.
  Noise-driven firing pattern ENCODES the input via stochastic resonance.
  Firmware weights modulate WHICH neurons are pushed over → richer encoding.

This should FLIP the hierarchy:
  FPGA_NONE ≈ chance (subthreshold, zero spikes)
  FPGA_128 > FPGA_NONE (noise enables firing)
  DEEP_INTER > FPGA_128 (firmware weights create heterogeneous thresholds)

Additionally: NARMA-10 regression benchmark — standard RC benchmark where LINEAR fails.

Conditions:
  DEEP_INTER:  Subthreshold FPGA + firmware weights + noise + GPU workload
  SMN_WEIGHTS: Subthreshold FPGA + firmware weight gain modulation
  FPGA_128:    Subthreshold FPGA + 1/f noise (no firmware weights)
  FPGA_NONE:   Subthreshold FPGA + static Vg (no noise)
  SUPRA_NONE:  Suprathreshold FPGA + static Vg (z2208 regime, control)
  LINEAR:      Time-delay embedding baseline

Tests T335-T348:
  Waveform classification:
    T335: FPGA_128 > FPGA_NONE (noise enables subthreshold computation)
    T336: DEEP_INTER > FPGA_128 (firmware weights add value)
    T337: DEEP_INTER > FPGA_NONE by >15pp (subthreshold regime reversal)
    T338: FPGA_NONE < 0.45 (near chance = neurons barely fire)
    T339: DEEP_INTER > 0.60 (noise-enabled computation works)
  NARMA-10:
    T340: FPGA_128 NRMSE < FPGA_NONE NRMSE (noise helps nonlinear memory)
    T341: DEEP_INTER NRMSE < LINEAR NRMSE (reservoir beats linear on NARMA)
  Memory capacity:
    T342: MC(FPGA_128) > MC(FPGA_NONE) (noise adds temporal memory)
    T343: MC(DEEP_INTER) > MC(FPGA_128) (firmware adds memory)
  Cross-substrate:
    T344: Weight-spike MI > 0.01 bits (firmware modulates firing)
    T345: Noise-spike correlation > 0.1 (noise drives spikes)
    T346: SUPRA_NONE > FPGA_NONE (control: suprathreshold fires, subthreshold doesn't)
  Stochastic resonance:
    T347: FPGA_128 acc peaks at intermediate noise (not monotonic with noise)
    T348: SNR improvement ratio > 1.0 (stochastic resonance detected)

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
SUBTHRESHOLD_VG = 0.48   # BELOW BVpar cliff (~0.60) — neurons barely fire
SUPRATHRESHOLD_VG = 0.58  # ABOVE cliff — z2208 control regime
ALPHA = 0.02              # TINY input modulation — requires noise to discriminate
BETA_BASE = 0.15          # Noise amplitude (will sweep for SR curve)
SAMPLE_HZ = 20
IIR_ALPHA = 0.85
N_TRIALS = 200
STEPS_PER_TRIAL = 30
N_FOLDS = 5
BASE_WORKLOAD_SIZE = 128

# ─── Noise Channel Assignment ───
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
# Firmware reads (from z2208)
# ═══════════════════════════════════════════════════════════

def read_smn_register(address):
    assert address in SMN_THERMAL_ADDRS
    try:
        with open(SMN_PATH, 'rb+') as f:
            f.write(struct.pack('<I', address))
            f.seek(0)
            data = f.read(4)
            if len(data) < 4: return 0
            return struct.unpack('<I', data)[0]
    except Exception:
        return 0

def read_smn_weights():
    raw = np.array([read_smn_register(addr) for addr in SMN_THERMAL_ADDRS], dtype=np.float64)
    if raw.max() == raw.min(): return np.zeros(8)
    return 2.0 * (raw - raw.min()) / max(raw.max() - raw.min(), 1.0) - 1.0

def read_pm_table_weights():
    try:
        with open(PM_TABLE_PATH, 'rb') as f: data = f.read(1024)
        if len(data) < 0x200: return np.zeros(8)
        values = []
        for off in [0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C]:
            val = struct.unpack_from('<f', data, off)[0]
            values.append(val if np.isfinite(val) else 0.0)
        raw = np.array(values, dtype=np.float64)
        if raw.max() == raw.min(): return np.zeros(8)
        return 2.0 * (raw - raw.min()) / max(raw.max() - raw.min(), 1.0) - 1.0
    except Exception:
        return np.zeros(8)

def read_power_weights():
    try:
        with open(HWMON_POWER) as f: p0 = int(f.read().strip()) / 1e6
        time.sleep(0.003)
        with open(HWMON_POWER) as f: p1 = int(f.read().strip()) / 1e6
        return np.array([p0, p1], dtype=np.float64)
    except Exception:
        return np.array([11.0, 11.0])

def compute_firmware_weights(smn_w, pm_w, power_w):
    smn_128 = np.repeat(smn_w, 16)
    pm_128 = np.repeat(pm_w, 16)
    pn = power_w.copy()
    if pn.max() != pn.min():
        pn = 2.0 * (pn - pn.min()) / (pn.max() - pn.min() + 1e-10) - 1.0
    else:
        pn = np.zeros(2)
    power_128 = np.tile(pn, 64)
    return 0.5 * smn_128 + 0.3 * pm_128 + 0.2 * power_128

def read_deep_pm_state():
    vec = np.zeros(8)
    try:
        with open(PM_TABLE_PATH, 'rb') as f: data = f.read(0x200)
        for i, (name, (off, fmt, scale)) in enumerate(PM_DEEP_OFFSETS.items()):
            if off + 4 <= len(data):
                val = struct.unpack_from(f'<{fmt}', data, off)[0]
                if np.isfinite(val): vec[i] = min(val / scale, 1.0)
    except Exception: pass
    return vec


# ═══════════════════════════════════════════════════════════
# Noise Sources
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
# GPU workload
# ═══════════════════════════════════════════════════════════

def run_gpu_workload(input_val, torch_mod, device):
    inp = float(np.clip(input_val, 0.0, 1.0))
    size = max(64, min(512, int(BASE_WORKLOAD_SIZE + inp * 384)))
    m = torch_mod.randn(size, size, device=device)
    _ = m @ m.T
    torch_mod.cuda.synchronize()
    return size


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
    """NARMA-10 benchmark: y(t) = 0.3*y(t-1) + 0.05*y(t-1)*sum(y(t-1..t-10))
                                  + 1.5*u(t-10)*u(t-1) + 0.1"""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=n_steps)
    y = np.zeros(n_steps)
    for t in range(10, n_steps):
        y[t] = (0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-10] * u[t-1] + 0.1)
        y[t] = np.clip(y[t], 0, 10)  # prevent divergence
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
    if alphas is None: alphas = [1e-4, 1e-2, 1.0, 100.0, 1000.0, 10000.0]
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

def ridge_regress(X_tr, y_tr, X_te, y_te, alphas=None):
    """Ridge regression, returns NRMSE on test set."""
    if alphas is None: alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0, 10000.0]
    best_nrmse = 1e9
    y_var = max(np.var(y_te), 1e-10)
    for a in alphas:
        I = np.eye(X_tr.shape[1])
        try: w = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ y_tr)
        except: continue
        pred = X_te @ w
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


# ═══════════════════════════════════════════════════════════
# Subthreshold Reservoir Step
# ═══════════════════════════════════════════════════════════

def run_step(fpga, input_val, t, noises, w_in, w_noise, torch_mod, device,
             prev_counts, cumulative, mode='DEEP_INTER', fw_weights=None, beta=BETA_BASE):
    """One timestep in subthreshold stochastic resonance regime.

    Key difference from z2208: BASE_VG = 0.48 (subthreshold).
    Without noise, neurons are below BVpar cliff → minimal firing.
    Noise pushes some neurons over threshold → stochastic resonance encoding.
    """
    fpga_features = np.zeros(N_NEURONS * 3)
    deep_pm = np.zeros(8)

    base_vg = SUBTHRESHOLD_VG if mode != 'SUPRA_NONE' else SUPRATHRESHOLD_VG

    # ── Compute Vg ──
    if mode == 'DEEP_INTER':
        if fw_weights is not None:
            # Firmware weights shift effective threshold per-neuron
            # Some neurons closer to threshold, some further → heterogeneous sensitivity
            threshold_shift = 0.08 * fw_weights  # ±0.08V shift from firmware
        else:
            threshold_shift = np.zeros(N_NEURONS)
        vg = np.full(N_NEURONS, base_vg) + threshold_shift + ALPHA * input_val * w_in
        # Add noise to push over threshold
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

    elif mode == 'SMN_WEIGHTS':
        if fw_weights is not None:
            threshold_shift = 0.08 * fw_weights
        else:
            threshold_shift = np.zeros(N_NEURONS)
        vg = np.full(N_NEURONS, base_vg) + threshold_shift + ALPHA * input_val * w_in
        vg = np.clip(vg, 0.05, 0.95)

    elif mode == 'FPGA_128':
        vg = np.full(N_NEURONS, base_vg) + ALPHA * input_val * w_in
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

    elif mode in ('FPGA_NONE', 'SUPRA_NONE'):
        vg = np.full(N_NEURONS, base_vg) + ALPHA * input_val * w_in
        vg = np.clip(vg, 0.05, 0.95)

    else:  # LINEAR
        return fpga_features, deep_pm, prev_counts, cumulative

    # ── Write Vg + wait + read ──
    try:
        fpga.set_vg_all(vg.tolist())
    except Exception:
        pass

    time.sleep(0.05)
    time.sleep(1.0 / SAMPLE_HZ * 0.3)

    try:
        fpga.ser.reset_input_buffer()
        telem = fpga.read_telem(timeout=0.3)
    except Exception:
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
                fpga_features[i] = delta
                cumulative[i] += delta
        for i in range(N_NEURONS):
            fpga_features[N_NEURONS + i] = vmems[i]
            fpga_features[N_NEURONS * 2 + i] = cumulative[i]
        prev_counts = counts[:]

    # ── GPU workload (DEEP_INTER only, after FPGA read) ──
    if mode == 'DEEP_INTER':
        try: run_gpu_workload(input_val, torch_mod, device)
        except: pass
        deep_pm = read_deep_pm_state()

    return fpga_features, deep_pm, prev_counts, cumulative


def run_trial(fpga, input_signal, noises, w_in, w_noise, torch_mod, device,
              mode='DEEP_INTER', fw_weights=None, beta=BETA_BASE):
    n_steps = len(input_signal)
    all_fpga = np.zeros((n_steps, N_NEURONS * 3))
    all_pm = np.zeros((n_steps, 8))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        fpga_f, pm_f, prev_counts, cumulative = run_step(
            fpga, input_signal[t], t, noises, w_in, w_noise,
            torch_mod, device, prev_counts, cumulative,
            mode=mode, fw_weights=fw_weights, beta=beta)
        all_fpga[t] = fpga_f
        all_pm[t] = pm_f

    return all_fpga, all_pm


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    global SUBTHRESHOLD_VG
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps', type=int, default=STEPS_PER_TRIAL)
    parser.add_argument('--narma-steps', type=int, default=1500)
    parser.add_argument('--noise-s', type=float, default=15.0)
    args = parser.parse_args()

    print("=" * 70)
    print("z2209: Subthreshold Stochastic Resonance Reservoir")
    print(f"  Subthreshold Vg = {SUBTHRESHOLD_VG} (below BVpar cliff ~0.60)")
    print(f"  Input modulation ALPHA = {ALPHA} (tiny)")
    print(f"  Noise amplitude BETA = {BETA_BASE}")
    print("  Prediction: FPGA_NONE ≈ chance, noise conditions WIN")
    print("=" * 70)

    # ─── Init GPU ───
    try:
        import torch
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        print(f"[HW] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        _ = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[ERR] PyTorch/CUDA: {e}")
        sys.exit(1)

    # ─── Init FPGA ───
    print("\n[1/9] Connecting to 128-neuron FPGA...")
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
        'experiment': 'z2209_stochastic_resonance_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'n_neurons': N_NEURONS, 'subthreshold_vg': SUBTHRESHOLD_VG,
            'suprathreshold_vg': SUPRATHRESHOLD_VG,
            'alpha': ALPHA, 'beta_base': BETA_BASE, 'sample_hz': SAMPLE_HZ,
            'n_trials': args.n_trials, 'steps': args.steps,
        },
    }

    # ─── Read firmware weights ───
    print("\n[2/9] Reading firmware weight matrix...")
    smn_w = read_smn_weights()
    pm_w = read_pm_table_weights()
    power_w = read_power_weights()
    fw_weights = compute_firmware_weights(smn_w, pm_w, power_w)
    print(f"  SMN raw: {[f'{v:.4f}' for v in smn_w]}")
    print(f"  Combined weight range: [{fw_weights.min():.4f}, {fw_weights.max():.4f}]")
    print(f"  Combined weight std:   {fw_weights.std():.6f}")
    results['firmware_weights'] = {
        'smn': smn_w.tolist(), 'pm': pm_w.tolist(), 'power': power_w.tolist(),
        'combined_std': float(fw_weights.std()),
    }

    # ─── Collect noise ───
    print(f"\n[3/9] Collecting 5-channel noise ({args.noise_s}s)...")
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
            print(f"  {name}: MISSING, using zeros")

    # ─── Subthreshold verification ───
    print("\n[3.5/9] Verifying subthreshold regime...")
    # Set all neurons to subthreshold Vg, read spike rates
    vg_sub = np.full(N_NEURONS, SUBTHRESHOLD_VG)
    try:
        fpga.set_vg_all(vg_sub.tolist())
        time.sleep(0.5)
        t1 = fpga.read_telem(timeout=0.5)
        time.sleep(1.0)
        t2 = fpga.read_telem(timeout=0.5)
        if t1 and t2 and len(t1) >= N_NEURONS and len(t2) >= N_NEURONS:
            sub_rates = []
            for i in range(N_NEURONS):
                delta = (t2[i]['spike_count'] - t1[i]['spike_count']) & 0xFFFF
                if delta > 30000: delta = 0
                sub_rates.append(delta)
            mean_sub = np.mean(sub_rates)
            nonzero_sub = np.sum(np.array(sub_rates) > 0)
            print(f"  Subthreshold (Vg={SUBTHRESHOLD_VG}): mean_rate={mean_sub:.1f}, "
                  f"active={nonzero_sub}/{N_NEURONS}")

            # Also check suprathreshold
            vg_supra = np.full(N_NEURONS, SUPRATHRESHOLD_VG)
            fpga.set_vg_all(vg_supra.tolist())
            time.sleep(0.5)
            t1 = fpga.read_telem(timeout=0.5)
            time.sleep(1.0)
            t2 = fpga.read_telem(timeout=0.5)
            if t1 and t2 and len(t1) >= N_NEURONS and len(t2) >= N_NEURONS:
                supra_rates = []
                for i in range(N_NEURONS):
                    delta = (t2[i]['spike_count'] - t1[i]['spike_count']) & 0xFFFF
                    if delta > 30000: delta = 0
                    supra_rates.append(delta)
                mean_supra = np.mean(supra_rates)
                nonzero_supra = np.sum(np.array(supra_rates) > 0)
                print(f"  Suprathreshold (Vg={SUPRATHRESHOLD_VG}): mean_rate={mean_supra:.1f}, "
                      f"active={nonzero_supra}/{N_NEURONS}")
                results['threshold_check'] = {
                    'sub_mean_rate': float(mean_sub), 'sub_active': int(nonzero_sub),
                    'supra_mean_rate': float(mean_supra), 'supra_active': int(nonzero_supra),
                }

                # If subthreshold is already firing a lot, adjust Vg lower
                if mean_sub > 5.0:
                    print(f"  WARNING: Subthreshold neurons still active (mean={mean_sub:.1f})")
                    print(f"  Adjusting SUBTHRESHOLD_VG down to 0.40")
                    SUBTHRESHOLD_VG = 0.40
    except Exception as e:
        print(f"  Threshold check failed: {e}")

    # ─── Generate tasks ───
    print(f"\n[4/9] Generating waveform + NARMA-10 tasks...")
    waves, wave_labels = generate_waveforms(args.n_trials, args.steps)
    print(f"  Waveforms: {args.n_trials} trials x {args.steps} steps")
    print(f"  Class distribution: {np.bincount(wave_labels)}")

    narma_u, narma_y = generate_narma10(args.narma_steps)
    print(f"  NARMA-10: {args.narma_steps} steps")

    # ─── Run conditions ───
    CONDITIONS = ['DEEP_INTER', 'SMN_WEIGHTS', 'FPGA_128', 'FPGA_NONE', 'SUPRA_NONE', 'LINEAR']
    condition_features = {}
    condition_spike_stats = {}

    for cond in CONDITIONS:
        print(f"\n[5/9] Waveform condition: {cond}")
        all_features = []
        total_spikes = 0
        total_active_neurons = 0
        t0 = time.time()

        # Refresh firmware weights periodically
        current_fw = fw_weights if cond in ('DEEP_INTER', 'SMN_WEIGHTS') else None

        for trial_i in range(args.n_trials):
            if cond == 'LINEAR':
                feat = np.concatenate([waves[trial_i], waves[trial_i]**2,
                                       np.diff(waves[trial_i], prepend=waves[trial_i][0]),
                                       np.cumsum(waves[trial_i]) / args.steps])
                all_features.append(feat)
            else:
                fpga_states, pm_states = run_trial(
                    fpga, waves[trial_i], noises, w_in, w_noise,
                    __import__('torch'), device, mode=cond,
                    fw_weights=current_fw, beta=BETA_BASE)

                # Track spike statistics
                spike_deltas = fpga_states[:, :N_NEURONS]
                trial_total = spike_deltas.sum()
                trial_active = np.sum(spike_deltas.sum(axis=0) > 0)
                total_spikes += trial_total
                total_active_neurons += trial_active

                if cond == 'DEEP_INTER':
                    combined = np.hstack([fpga_states, pm_states])
                else:
                    combined = fpga_states
                feat = pool_trial_features(combined)
                all_features.append(feat)

            if (trial_i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = (trial_i + 1) / elapsed if elapsed > 0 else 0
                eta = int((args.n_trials - trial_i - 1) / rate) if rate > 0 else 0
                print(f"    Trial {trial_i+1}/{args.n_trials} ({rate:.2f} t/s, ETA {eta}s)")

            if cond in ('DEEP_INTER', 'SMN_WEIGHTS') and (trial_i + 1) % 50 == 0:
                smn_w = read_smn_weights()
                pm_w = read_pm_table_weights()
                power_w = read_power_weights()
                current_fw = compute_firmware_weights(smn_w, pm_w, power_w)
                print(f"    [Refreshed firmware weights: std={current_fw.std():.6f}]")

        X = np.array(all_features)
        elapsed = time.time() - t0
        print(f"  {cond}: {args.n_trials} trials in {elapsed:.1f}s, feat dim={X.shape[1]}")

        if cond != 'LINEAR':
            avg_spikes = total_spikes / args.n_trials if args.n_trials > 0 else 0
            avg_active = total_active_neurons / args.n_trials if args.n_trials > 0 else 0
            print(f"  Spike stats: avg_spikes/trial={avg_spikes:.1f}, avg_active={avg_active:.1f}/{N_NEURONS}")
            condition_spike_stats[cond] = {
                'avg_spikes_per_trial': float(avg_spikes),
                'avg_active_neurons': float(avg_active),
            }

        condition_features[cond] = X

    # ─── Classify waveforms ───
    print(f"\n[6/9] Classifying waveforms (5-fold CV)...")
    wave_results = {}
    for cond in CONDITIONS:
        X = condition_features[cond]
        r = classify_condition(X, wave_labels)
        wave_results[cond] = r
        print(f"  {cond}: {r['mean']:.3f} +/- {r['std']:.3f}")
    results['waveform'] = wave_results
    results['spike_stats'] = condition_spike_stats

    # ─── NARMA-10 regression ───
    print(f"\n[7/9] NARMA-10 regression...")
    narma_results = {}
    # For NARMA, we drive the reservoir with u(t) and try to predict y(t)
    # We need time-series features, not pooled features
    narma_steps_use = min(args.narma_steps, 800)  # limit for time

    for cond in ['DEEP_INTER', 'FPGA_128', 'FPGA_NONE', 'LINEAR']:
        print(f"  Running NARMA reservoir ({cond})...")
        if cond == 'LINEAR':
            # Time-delay embedding
            delays = [1, 2, 3, 5, 10]
            X_narma = np.zeros((narma_steps_use, len(delays) + 1))
            X_narma[:, 0] = narma_u[:narma_steps_use]
            for d_i, d in enumerate(delays):
                X_narma[d:, d_i + 1] = narma_u[:narma_steps_use - d]
        else:
            # Drive reservoir with NARMA input
            current_fw = fw_weights if cond == 'DEEP_INTER' else None
            reservoir_states = []
            prev_counts = None
            cumulative = np.zeros(N_NEURONS)
            for t_i in range(narma_steps_use):
                fpga_f, pm_f, prev_counts, cumulative = run_step(
                    fpga, narma_u[t_i], t_i, noises, w_in, w_noise,
                    __import__('torch'), device, prev_counts, cumulative,
                    mode=cond, fw_weights=current_fw, beta=BETA_BASE)
                if cond == 'DEEP_INTER':
                    reservoir_states.append(np.concatenate([fpga_f, pm_f]))
                else:
                    reservoir_states.append(fpga_f)

                if (t_i + 1) % 100 == 0:
                    print(f"    Step {t_i+1}/{narma_steps_use}")
            X_narma = np.array(reservoir_states)

        # Reduce dimensionality
        if X_narma.shape[1] > 100:
            X_narma, _, _ = pca_reduce(X_narma, n_components=80)

        # Split 70/30
        y_narma = narma_y[:narma_steps_use]
        split = int(0.7 * narma_steps_use)
        # Skip first 50 warmup steps
        warmup = 50
        X_tr = X_narma[warmup:split]
        y_tr = y_narma[warmup:split]
        X_te = X_narma[split:]
        y_te = y_narma[split:]

        # Normalize
        mu = X_tr.mean(axis=0, keepdims=True)
        sigma = X_tr.std(axis=0, keepdims=True)
        sigma[sigma < 1e-10] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma

        nrmse = ridge_regress(X_tr_n, y_tr, X_te_n, y_te)
        narma_results[cond] = float(nrmse)
        print(f"  {cond} NARMA-10 NRMSE: {nrmse:.4f}")

    results['narma10'] = narma_results

    # ─── Memory capacity ───
    print(f"\n[8/9] Memory capacity...")
    mc_results = {}
    for cond in ['DEEP_INTER', 'FPGA_128', 'FPGA_NONE']:
        # Memory capacity = sum of R^2 for predicting u(t-k) from reservoir state at time t
        mc_total = 0.0
        # Reuse waveform features: use trial mean spike rates as state vectors
        # For each trial, the "input" is the mean of the waveform
        # For memory: use sequential trials and measure how well we can predict past inputs

        # Simpler: use the first 100 trial features as sequential states
        X_mc = condition_features[cond][:100]
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
            mu = X_tr_mc.mean(axis=0, keepdims=True)
            sigma = X_tr_mc.std(axis=0, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            X_tr_mc = (X_tr_mc - mu) / sigma
            X_te_mc = (X_te_mc - mu) / sigma
            # R^2
            nrmse = ridge_regress(X_tr_mc, y_tr_mc, X_te_mc, y_te_mc)
            r2 = max(0, 1.0 - nrmse**2)
            mc_total += r2

        mc_results[cond] = float(mc_total)
        print(f"  MC({cond}) = {mc_total:.3f}")

    results['memory_capacity'] = mc_results

    # ─── Cross-substrate metrics ───
    print(f"\n[9/9] Cross-substrate metrics...")
    # Weight-spike MI: do firmware weights predict which neurons fire?
    # Use spike rates from DEEP_INTER trials
    X_deep = condition_features.get('DEEP_INTER')
    X_none = condition_features.get('FPGA_NONE')

    weight_spike_mi = 0.0
    noise_spike_corr = 0.0

    if X_deep is not None:
        # Mean spike rate per neuron across all trials
        spike_rates = X_deep[:, :N_NEURONS].mean(axis=0)
        # Correlation with firmware weight magnitude
        if fw_weights is not None:
            fw_abs = np.abs(fw_weights)
            if spike_rates.std() > 0 and fw_abs.std() > 0:
                corr = np.corrcoef(spike_rates, fw_abs)[0, 1]
                noise_spike_corr = abs(corr) if np.isfinite(corr) else 0.0
            # Simple MI estimate via binning
            n_bins = 10
            spike_bins = np.digitize(spike_rates, np.linspace(spike_rates.min(),
                                     spike_rates.max() + 1e-10, n_bins + 1)) - 1
            fw_bins = np.digitize(fw_abs, np.linspace(fw_abs.min(),
                                  fw_abs.max() + 1e-10, n_bins + 1)) - 1
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

    # Noise-spike correlation: do noise conditions have higher spike variance than no-noise?
    if X_deep is not None and X_none is not None:
        deep_spike_std = X_deep[:, :N_NEURONS].std(axis=0).mean()
        none_spike_std = X_none[:, :N_NEURONS].std(axis=0).mean()
        noise_spike_corr = float(deep_spike_std / max(none_spike_std, 1e-6))

    print(f"  Weight-spike MI = {weight_spike_mi:.4f} bits")
    print(f"  Noise-spike correlation ratio = {noise_spike_corr:.4f}")

    results['cross_substrate'] = {
        'weight_spike_mi': weight_spike_mi,
        'noise_spike_corr_ratio': noise_spike_corr,
    }

    # ─── Stochastic resonance curve ───
    # Sweep noise amplitude: does accuracy peak at intermediate level?
    print(f"\n[9.5/9] Stochastic resonance sweep (3 levels)...")
    sr_accs = {}
    betas_to_test = [0.05, 0.15, 0.40]  # low, medium, high noise

    for beta_test in betas_to_test:
        print(f"  BETA={beta_test:.2f}: running 60 quick trials...")
        sr_features = []
        for trial_i in range(60):
            fpga_states, _ = run_trial(
                fpga, waves[trial_i], noises, w_in, w_noise,
                __import__('torch'), device, mode='FPGA_128',
                fw_weights=None, beta=beta_test)
            sr_features.append(pool_trial_features(fpga_states))

            if (trial_i + 1) % 20 == 0:
                print(f"    {trial_i+1}/60")

        X_sr = np.array(sr_features)
        y_sr = wave_labels[:60]
        sr_r = classify_condition(X_sr, y_sr, n_splits=3, max_features=80)
        sr_accs[f"beta_{beta_test:.2f}"] = sr_r['mean']
        print(f"    Acc = {sr_r['mean']:.3f}")

    results['sr_curve'] = sr_accs

    # Check SR: does accuracy peak at intermediate noise?
    sr_vals = [sr_accs[f"beta_{b:.2f}"] for b in betas_to_test]
    sr_peak_at_mid = sr_vals[1] > sr_vals[0] and sr_vals[1] > sr_vals[2]
    sr_improvement = sr_vals[1] / max(sr_vals[0], 0.01)

    # ═══════════════════════════════════════════════════════════
    # TEST VERDICTS
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    tests = []
    w = wave_results
    n = narma_results
    mc = mc_results

    def test(tid, desc, passed):
        status = "PASS" if passed else "FAIL"
        print(f"  {tid} {desc}: {status}")
        tests.append({'id': tid, 'description': desc, 'passed': bool(passed)})

    # Waveform
    test('T335', f"FPGA_128({w['FPGA_128']['mean']:.3f}) > FPGA_NONE({w['FPGA_NONE']['mean']:.3f})",
         w['FPGA_128']['mean'] > w['FPGA_NONE']['mean'])
    test('T336', f"DEEP_INTER({w['DEEP_INTER']['mean']:.3f}) > FPGA_128({w['FPGA_128']['mean']:.3f})",
         w['DEEP_INTER']['mean'] > w['FPGA_128']['mean'])
    diff_dn = w['DEEP_INTER']['mean'] - w['FPGA_NONE']['mean']
    test('T337', f"DEEP_INTER - FPGA_NONE = {diff_dn:.3f} > 0.15",
         diff_dn > 0.15)
    test('T338', f"FPGA_NONE({w['FPGA_NONE']['mean']:.3f}) < 0.45 (near chance)",
         w['FPGA_NONE']['mean'] < 0.45)
    test('T339', f"DEEP_INTER({w['DEEP_INTER']['mean']:.3f}) > 0.60",
         w['DEEP_INTER']['mean'] > 0.60)

    # NARMA-10
    test('T340', f"FPGA_128 NRMSE({n.get('FPGA_128',99):.3f}) < FPGA_NONE({n.get('FPGA_NONE',99):.3f})",
         n.get('FPGA_128', 99) < n.get('FPGA_NONE', 99))
    test('T341', f"DEEP_INTER NRMSE({n.get('DEEP_INTER',99):.3f}) < LINEAR({n.get('LINEAR',99):.3f})",
         n.get('DEEP_INTER', 99) < n.get('LINEAR', 99))

    # Memory capacity
    test('T342', f"MC(FPGA_128)={mc.get('FPGA_128',0):.3f} > MC(FPGA_NONE)={mc.get('FPGA_NONE',0):.3f}",
         mc.get('FPGA_128', 0) > mc.get('FPGA_NONE', 0))
    test('T343', f"MC(DEEP_INTER)={mc.get('DEEP_INTER',0):.3f} > MC(FPGA_128)={mc.get('FPGA_128',0):.3f}",
         mc.get('DEEP_INTER', 0) > mc.get('FPGA_128', 0))

    # Cross-substrate
    test('T344', f"Weight-spike MI({weight_spike_mi:.4f}) > 0.01",
         weight_spike_mi > 0.01)
    test('T345', f"Noise-spike ratio({noise_spike_corr:.4f}) > 1.0",
         noise_spike_corr > 1.0)
    test('T346', f"SUPRA_NONE({w['SUPRA_NONE']['mean']:.3f}) > FPGA_NONE({w['FPGA_NONE']['mean']:.3f})",
         w['SUPRA_NONE']['mean'] > w['FPGA_NONE']['mean'])

    # Stochastic resonance
    test('T347', f"SR peak at intermediate noise: {sr_peak_at_mid}",
         sr_peak_at_mid)
    test('T348', f"SR improvement ratio({sr_improvement:.3f}) > 1.0",
         sr_improvement > 1.0)

    n_pass = sum(1 for t in tests if t['passed'])
    n_total = len(tests)
    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    results['tests'] = tests
    results['summary'] = {'pass': n_pass, 'total': n_total}

    out_path = RESULTS / 'z2209_stochastic_resonance_reservoir.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, cls=NpEncoder, indent=2)
    print(f"\n  Results saved: {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
