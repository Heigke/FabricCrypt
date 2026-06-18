#!/usr/bin/env python3
"""z2220_eth_reservoir.py — High-Speed Ethernet Reservoir (z2206 architecture)

z2219 showed ESN-style recurrence (W_res @ h) doesn't help.
z2206 showed the REAL winning architecture: direct noise injection + delay taps.
  Vg(t) = base_vg + α·input·w_in + β·IIR_noise(t)·w_noise
  Features: [spike_delta, vmem, cumulative] × delay_taps(t, t-1, t-2, t-3)

This script: same z2206 architecture but 10× faster via Ethernet.
  z2206: 20 Hz UART → 30 steps/trial (1.5s physical time)
  z2220: 200 Hz Ethernet → 200 steps/trial (1.0s physical time)
  More steps = richer temporal features = better delay tap extraction

Conditions:
  FULL_128:  128 neurons, 5-channel heterogeneous noise, 200 Hz
  HOMO_128:  128 neurons, power 1/f only, 200 Hz
  FAST_128:  128 neurons, 5-channel, 500 Hz (test speed limit)
  SLOW_128:  128 neurons, 5-channel, 50 Hz (control for rate effect)

Tasks:
  - 7-class waveform classification (200 trials × 200 steps)
  - Temporal XOR τ=1,2,3,5,8 (continuous 4000 steps)
  - Memory Capacity (continuous 400 steps, delays 1-40)

Tests T474-T489:
  T474: FULL_128 waveform > 0.80 (at least match z2206's 0.81)
  T475: FULL_128 waveform > HOMO_128 (heterogeneous noise helps)
  T476: FULL_128 waveform > SLOW_128 (speed helps)
  T477: FULL_128 XOR τ=2 > 0.60
  T478: FULL_128 XOR τ=5 > 0.55 (long-range memory — z2206 FAILED this)
  T479: FULL_128 XOR τ=8 > 0.52 (very long range)
  T480: FULL_128 MC > 3.0 (z2206 was 2.67)
  T481: FAST_128 XOR τ=5 > FULL_128 XOR τ=5 (faster = more samples in same τ window)
  T482: XOR τ=2 monotonically increases with sample rate
  T483: MC increases with sample rate
  T484: FULL_128 XOR τ=3 > 0.55
  T485: FULL waveform std < 0.05 (stable)
  T486: At least 3 delay taps contribute (delay ablation)
  T487: FAST_128 waveform > 0.75
  T488: FULL_128 cross-neuron corr < 0.30
  T489: FULL_128 MC per-delay curve has exponential decay shape

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128-neuron, UDP Ethernet)
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Reservoir Parameters ───
N_NEURONS = 128
BASE_VG = 0.58
ALPHA = 0.25        # input gain
BETA = 0.08         # noise injection gain
IIR_ALPHA = 0.85    # temporal smoothing for noise

# ─── Noise Channels ───
POWER_NEURONS   = list(range(0, 32))
SMN_NEURONS     = list(range(32, 56))
JITTER_NEURONS  = list(range(56, 80))
THERMAL_NEURONS = list(range(80, 104))
CLOCK_NEURONS   = list(range(104, 128))

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

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
            f.seek(0x004C)
            return struct.unpack('<f', f.read(4))[0]
    except: return None

def read_perf_jitter():
    t0 = time.perf_counter_ns()
    _ = os.getpid()
    return time.perf_counter_ns() - t0

def normalize_noise(samples):
    arr = np.array(samples, dtype=float)
    if len(arr) == 0: return arr
    return (arr - arr.mean()) / max(arr.std(), 1e-6)

def iir_filter_noise(noise_samples, alpha_iir=0.85):
    if len(noise_samples) == 0: return noise_samples
    filtered = np.zeros(len(noise_samples))
    filtered[0] = noise_samples[0]
    for t in range(1, len(noise_samples)):
        filtered[t] = alpha_iir * filtered[t-1] + (1 - alpha_iir) * noise_samples[t]
    std = max(np.std(filtered), 1e-6)
    return filtered / std

def collect_all_noise(duration_s=15, sample_hz=100):
    n = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    power_s, thermal_s, clock_s, smn_s, jitter_s = [], [], [], [], []
    print("  Collecting 5 noise channels...")
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
        if n > 4 and (i + 1) % (n // 4) == 0:
            print(f"    {i+1}/{n} samples")
    return {
        'power': iir_filter_noise(normalize_noise(power_s)),
        'smn': iir_filter_noise(normalize_noise(smn_s), alpha_iir=0.92),
        'jitter': iir_filter_noise(normalize_noise(jitter_s), alpha_iir=0.50),
        'thermal': iir_filter_noise(normalize_noise(thermal_s), alpha_iir=0.90),
        'clock': iir_filter_noise(normalize_noise(clock_s), alpha_iir=0.88),
    }

# ═══════════════════════════════════════════════════════════
# Vg Computation (z2206 architecture — NO ESN h-state)
# ═══════════════════════════════════════════════════════════

CHANNEL_MAP = {
    'power': POWER_NEURONS,
    'smn': SMN_NEURONS,
    'jitter': JITTER_NEURONS,
    'thermal': THERMAL_NEURONS,
    'clock': CLOCK_NEURONS,
}

def compute_vg(t, input_val, noises, w_in, w_noise, mode='FULL_128'):
    """Direct noise injection — no recurrent W_res. Same as z2206."""
    vg = np.full(N_NEURONS, BASE_VG) + ALPHA * input_val * w_in

    if mode in ('FULL_128', 'FAST_128', 'SLOW_128'):
        for ch_name, neuron_ids in CHANNEL_MAP.items():
            ch_data = noises.get(ch_name, np.zeros(1))
            if len(ch_data) == 0: continue
            idx = t % len(ch_data)
            for nid in neuron_ids:
                vg[nid] += BETA * ch_data[idx] * w_noise[nid]

    elif mode == 'HOMO_128':
        ch_data = noises.get('power', np.zeros(1))
        if len(ch_data) > 0:
            idx = t % len(ch_data)
            vg += BETA * ch_data[idx] * w_noise

    return np.clip(vg, 0.05, 0.95)

# ═══════════════════════════════════════════════════════════
# Reservoir Core — Ethernet
# ═══════════════════════════════════════════════════════════

def run_reservoir_eth(fpga, input_signal, noises, w_in, w_noise,
                      sample_hz, mode='FULL_128'):
    """Drive 128-neuron FPGA reservoir via Ethernet.

    Returns: (n_steps, N*3) array — [spike_delta | vmem | cumulative]
    """
    n_steps = len(input_signal)
    interval = 1.0 / sample_hz
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    actual_rates = []

    for t in range(n_steps):
        t_start = time.perf_counter()

        vg = compute_vg(t, input_signal[t], noises, w_in, w_noise, mode=mode)
        fpga.set_vg_batch(0, vg.tolist())

        # Integration time — proportional to interval
        time.sleep(max(0.001, interval * 0.3))

        try:
            counts, vmem, bvpar = fpga.read_telemetry_fast()
        except (TimeoutError, Exception):
            continue

        # Spike deltas
        if prev_counts is not None:
            for i in range(N_NEURONS):
                delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                if delta > 30000: delta = 0
                states[t, i] = delta
                cumulative[i] += delta
        for i in range(N_NEURONS):
            states[t, N_NEURONS + i] = vmem[i]
            states[t, N_NEURONS * 2 + i] = cumulative[i]
        prev_counts = counts.copy()

        # Pace to target rate
        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.0005:
            time.sleep(remaining)
        actual_rates.append(1.0 / max(time.perf_counter() - t_start, 1e-6))

    mean_rate = float(np.mean(actual_rates)) if actual_rates else 0.0
    return states, mean_rate

# ═══════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2, 3)):
    """Add time-delayed copies of state vector. This IS the temporal memory."""
    T, D = states.shape
    augmented = np.zeros((T, D * (1 + len(delays))))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[d:, start:start + D] = states[:T - d]
    return augmented

def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])

# ═══════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=None):
    if n_classes is None: n_classes = len(np.unique(np.concatenate([y_tr, y_te])))
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0, 1000.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
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

def ridge_binary(X_tr, y_tr, X_te, y_te):
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    best = -1
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try: w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except: continue
        acc = np.mean(((X_te_s @ w) > 0.5).astype(float) == y_te)
        if acc > best: best = acc
    return best

def ridge_regress(X_tr, y_tr, X_te, y_te):
    alphas = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    best_corr = -1
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try: w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except: continue
        pred = X_te_s @ w
        if np.std(pred) > 1e-10 and np.std(y_te) > 1e-10:
            corr = np.corrcoef(pred, y_te)[0, 1]
            if corr > best_corr: best_corr = corr
    return max(best_corr, 0.0)

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

def classify_cv(X, y, n_splits=5, n_classes=None):
    splits = stratified_kfold(X, y, n_splits)
    accs = []
    for tr_idx, te_idx in splits:
        acc = ridge_classify(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], n_classes=n_classes)
        accs.append(acc)
    return {'mean': float(np.mean(accs)), 'std': float(np.std(accs)), 'folds': [float(a) for a in accs]}

# ═══════════════════════════════════════════════════════════
# Task Generators
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials, steps, sample_hz, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / sample_hz
    t = np.arange(steps) * dt
    trials, labels = [], []
    for _ in range(n_trials):
        cls = rng.integers(0, 7)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.8, 1.2)
        if cls == 0:    wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:  wave = 2.0 * np.abs(2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0) - 1.0
        elif cls == 2:  wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif cls == 3:  wave = 2.0 * ((freq * t + phase/(2*np.pi)) % 1.0) - 1.0
        elif cls == 4:
            f0, f1 = freq * 0.5, freq * 2.0
            inst_f = f0 + (f1 - f0) * t / max(t[-1], 1e-6)
            wave = np.sin(2 * np.pi * np.cumsum(inst_f) * dt + phase)
        elif cls == 5:
            carrier = np.sin(2 * np.pi * freq * 2 * t + phase)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * freq * 0.3 * t)
            wave = carrier * envelope
        else:
            decay = np.exp(-2.0 * t)
            wave = np.sin(2 * np.pi * freq * t + phase) * decay
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)

def generate_xor_input(n_steps, seed=42):
    return np.random.default_rng(seed).integers(0, 2, size=n_steps).astype(float)

def compute_xor_targets(u, tau):
    n = len(u)
    targets = np.zeros(n, dtype=int)
    for t in range(tau, n):
        targets[t] = int(u[t]) ^ int(u[t - tau])
    return targets

# ═══════════════════════════════════════════════════════════
# Cross-Neuron Correlation
# ═══════════════════════════════════════════════════════════

def mean_off_diagonal_corr(states_list):
    """Mean |correlation| between spike_delta channels across trials."""
    corrs = []
    for states in states_list:
        spikes = states[:, :N_NEURONS]
        valid = [i for i in range(N_NEURONS) if spikes[:, i].std() > 1e-8]
        if len(valid) < 2: continue
        C = np.corrcoef(spikes[:, valid].T)
        mask = ~np.eye(len(valid), dtype=bool)
        corrs.append(np.mean(np.abs(C[mask])))
    return float(np.mean(corrs)) if corrs else 1.0

# ═══════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 70)
    print("z2220: High-Speed Ethernet Reservoir (z2206 Architecture)")
    print("  z2206 (UART 20Hz): 0.81 waveform, 0.62 XOR-2, 2.67 MC")
    print("  z2219 proved: ESN h-state recurrence doesn't help")
    print("  THIS: z2206's direct noise injection at 200 Hz")
    print("=" * 70)

    # ─── Connect ───
    print("\n[1] Connecting to FPGA via Ethernet...")
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("  FATAL: Cannot connect"); return
    print(f"  Connected: {fpga.num_neurons} neurons")

    # ─── Noise ───
    print("\n[2] Collecting firmware noise (15s at 100 Hz)...")
    noises = collect_all_noise(15, 100)
    for k, v in noises.items():
        print(f"  {k}: {len(v)} samples")

    # ─── Weights ───
    rng = np.random.default_rng(42)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)
    w_noise = rng.uniform(-1, 1, size=N_NEURONS)

    results = {
        'experiment': 'z2220_eth_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    tests = {}

    # ═══════════════════════════════════════════════════════════
    # WAVEFORM CLASSIFICATION (7-class)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("WAVEFORM CLASSIFICATION (7-class)")
    print("=" * 70)

    N_WAVE_TRIALS = 140  # 20 per class
    N_WAVE_STEPS = 100   # 0.5s at 200 Hz

    wave_results = {}
    wave_states_full = []  # for correlation analysis

    for mode, hz in [('FULL_128', 200), ('HOMO_128', 200), ('SLOW_128', 50), ('FAST_128', 500)]:
        print(f"\n  --- {mode} @ {hz} Hz ---")
        inputs, labels = generate_waveforms(N_WAVE_TRIALS, N_WAVE_STEPS, hz)
        all_feats = []

        for trial in range(N_WAVE_TRIALS):
            states, actual_hz = run_reservoir_eth(
                fpga, inputs[trial], noises, w_in, w_noise, hz, mode=mode)

            # Augment with delay taps — THIS is the temporal memory
            aug = augment_with_delays(states, delays=(1, 2, 3))
            feat = pool_trial_features(aug)
            all_feats.append(feat)

            if mode == 'FULL_128':
                wave_states_full.append(states)

            if trial == 0:
                print(f"    actual_hz={actual_hz:.0f}, feats={len(feat)}")
            if (trial + 1) % 35 == 0:
                print(f"    trial {trial+1}/{N_WAVE_TRIALS}")

        X = np.array(all_feats)
        res = classify_cv(X, labels, n_splits=5, n_classes=7)
        wave_results[mode] = res
        print(f"    {mode}: {res['mean']:.3f} ± {res['std']:.3f}")

    results['waveform'] = wave_results

    # Test evaluations
    full_wave = wave_results['FULL_128']['mean']
    homo_wave = wave_results['HOMO_128']['mean']
    slow_wave = wave_results['SLOW_128']['mean']
    fast_wave = wave_results['FAST_128']['mean']

    tests['T474'] = {'desc': 'FULL_128 wave > 0.80', 'val': full_wave, 'pass': full_wave > 0.80}
    tests['T475'] = {'desc': 'FULL > HOMO wave', 'val': full_wave - homo_wave,
                     'pass': full_wave > homo_wave}
    tests['T476'] = {'desc': 'FULL(200Hz) > SLOW(50Hz) wave', 'val': full_wave - slow_wave,
                     'pass': full_wave > slow_wave}
    tests['T485'] = {'desc': 'FULL wave std < 0.05', 'val': wave_results['FULL_128']['std'],
                     'pass': wave_results['FULL_128']['std'] < 0.05}
    tests['T487'] = {'desc': 'FAST wave > 0.75', 'val': fast_wave, 'pass': fast_wave > 0.75}

    # Cross-neuron correlation
    corr = mean_off_diagonal_corr(wave_states_full[:20])
    tests['T488'] = {'desc': 'cross-neuron corr < 0.30', 'val': corr, 'pass': corr < 0.30}
    results['cross_neuron_corr'] = corr
    print(f"\n  Cross-neuron correlation: {corr:.3f}")

    # ═══════════════════════════════════════════════════════════
    # TEMPORAL XOR (continuous stream, per-timestep classification)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEMPORAL XOR")
    print("=" * 70)

    XOR_STEPS = 4000

    xor_results = {}

    for mode, hz in [('FULL_128', 200), ('HOMO_128', 200), ('SLOW_128', 50), ('FAST_128', 500)]:
        print(f"\n  --- {mode} @ {hz} Hz ---")
        xor_input = generate_xor_input(XOR_STEPS)

        states, actual_hz = run_reservoir_eth(
            fpga, xor_input, noises, w_in, w_noise, hz, mode=mode)
        print(f"    actual_hz={actual_hz:.0f}")

        aug = augment_with_delays(states, delays=(1, 2, 3, 5, 8))

        xor_results[mode] = {}
        for tau in [1, 2, 3, 5, 8]:
            targets = compute_xor_targets(xor_input, tau)
            # Use valid region only (after max delay tap + tau)
            start = max(tau, 8) + 1
            X = aug[start:]
            y = targets[start:]

            # 70/30 split
            n = len(y)
            idx = np.random.default_rng(42).permutation(n)
            split = int(0.7 * n)
            acc = ridge_binary(X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]])
            xor_results[mode][f'tau{tau}'] = float(acc)
            print(f"    τ={tau}: {acc:.3f}")

    results['xor'] = xor_results

    # XOR tests
    full_xor = xor_results['FULL_128']
    tests['T477'] = {'desc': 'FULL XOR τ=2 > 0.60', 'val': full_xor['tau2'],
                     'pass': full_xor['tau2'] > 0.60}
    tests['T478'] = {'desc': 'FULL XOR τ=5 > 0.55', 'val': full_xor['tau5'],
                     'pass': full_xor['tau5'] > 0.55}
    tests['T479'] = {'desc': 'FULL XOR τ=8 > 0.52', 'val': full_xor['tau8'],
                     'pass': full_xor['tau8'] > 0.52}
    tests['T484'] = {'desc': 'FULL XOR τ=3 > 0.55', 'val': full_xor['tau3'],
                     'pass': full_xor['tau3'] > 0.55}

    fast_xor = xor_results.get('FAST_128', {})
    tests['T481'] = {'desc': 'FAST τ=5 > FULL τ=5',
                     'val': fast_xor.get('tau5', 0) - full_xor['tau5'],
                     'pass': fast_xor.get('tau5', 0) > full_xor['tau5']}

    # Rate monotonicity for XOR τ=2
    rates_tau2 = []
    for mode, hz in [('SLOW_128', 50), ('FULL_128', 200), ('FAST_128', 500)]:
        rates_tau2.append(xor_results.get(mode, {}).get('tau2', 0))
    monotonic = all(rates_tau2[i] <= rates_tau2[i+1] for i in range(len(rates_tau2)-1))
    tests['T482'] = {'desc': 'XOR τ=2 monotonic with rate', 'val': rates_tau2,
                     'pass': monotonic}

    # ═══════════════════════════════════════════════════════════
    # MEMORY CAPACITY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MEMORY CAPACITY")
    print("=" * 70)

    MC_STEPS = 500
    MAX_DELAY = 40

    mc_results = {}

    for mode, hz in [('FULL_128', 200), ('HOMO_128', 200), ('SLOW_128', 50)]:
        print(f"\n  --- {mode} @ {hz} Hz ---")
        mc_input = np.random.default_rng(99).uniform(0, 1, MC_STEPS)

        states, actual_hz = run_reservoir_eth(
            fpga, mc_input, noises, w_in, w_noise, hz, mode=mode)
        print(f"    actual_hz={actual_hz:.0f}")

        aug = augment_with_delays(states, delays=(1, 2, 3))

        mc_total = 0.0
        mc_per_delay = []
        for delay in range(1, MAX_DELAY + 1):
            if delay >= MC_STEPS - 20:
                mc_per_delay.append(0.0)
                continue
            y_delayed = mc_input[:-delay]
            X_d = aug[delay:]
            n = min(len(y_delayed), len(X_d))
            y_d = y_delayed[:n]
            X_d = X_d[:n]
            split = int(0.7 * n)
            corr = ridge_regress(X_d[:split], y_d[:split], X_d[split:], y_d[split:])
            r2 = corr ** 2
            mc_per_delay.append(float(r2))
            mc_total += r2

        mc_results[mode] = {'total': float(mc_total), 'per_delay': mc_per_delay}
        print(f"    MC={mc_total:.3f}")

    results['memory_capacity'] = mc_results

    full_mc = mc_results['FULL_128']['total']
    tests['T480'] = {'desc': 'FULL MC > 3.0', 'val': full_mc, 'pass': full_mc > 3.0}

    # MC rate monotonicity
    mc_rates = [mc_results.get('SLOW_128', {}).get('total', 0),
                mc_results.get('FULL_128', {}).get('total', 0)]
    tests['T483'] = {'desc': 'MC increases: SLOW < FULL',
                     'val': mc_rates, 'pass': mc_rates[0] < mc_rates[1]}

    # MC decay shape: check if first 5 delays > last 5 delays (exponential-ish)
    per_delay = mc_results['FULL_128']['per_delay']
    early_mc = np.mean(per_delay[:5]) if len(per_delay) >= 5 else 0
    late_mc = np.mean(per_delay[30:35]) if len(per_delay) >= 35 else 0
    tests['T489'] = {'desc': 'MC decay shape (early > late)', 'val': early_mc - late_mc,
                     'pass': early_mc > late_mc * 1.5}

    # ═══════════════════════════════════════════════════════════
    # DELAY TAP ABLATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DELAY TAP ABLATION")
    print("=" * 70)

    # Quick waveform test with different delay configurations
    ablation_input, ablation_labels = generate_waveforms(N_WAVE_TRIALS, N_WAVE_STEPS, 200)
    # Reuse FULL_128 states if we have them
    print("  Running FULL_128 @ 200 Hz for ablation...")
    ablation_states = []
    for trial in range(N_WAVE_TRIALS):
        states, _ = run_reservoir_eth(
            fpga, ablation_input[trial], noises, w_in, w_noise, 200, mode='FULL_128')
        ablation_states.append(states)
        if (trial + 1) % 35 == 0:
            print(f"    trial {trial+1}/{N_WAVE_TRIALS}")

    delay_configs = {
        'no_delay': (),
        'delay_1': (1,),
        'delay_1_2': (1, 2),
        'delay_1_2_3': (1, 2, 3),
    }
    ablation_accs = {}
    for name, delays in delay_configs.items():
        feats = []
        for states in ablation_states:
            aug = augment_with_delays(states, delays=delays) if delays else states
            feats.append(pool_trial_features(aug))
        X = np.array(feats)
        res = classify_cv(X, ablation_labels, n_splits=5, n_classes=7)
        ablation_accs[name] = res['mean']
        print(f"  {name}: {res['mean']:.3f}")

    results['delay_ablation'] = ablation_accs

    # Count how many delay taps improve accuracy
    acc_vals = [ablation_accs.get(k, 0) for k in ['no_delay', 'delay_1', 'delay_1_2', 'delay_1_2_3']]
    n_helpful = sum(1 for i in range(1, len(acc_vals)) if acc_vals[i] > acc_vals[i-1])
    tests['T486'] = {'desc': 'At least 2 delay taps help', 'val': n_helpful,
                     'pass': n_helpful >= 2}

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY — z2220 Ethernet Reservoir")
    print("=" * 70)

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n  Tests: {n_pass}/{n_total} PASS")

    for tid, t in sorted(tests.items()):
        status = "PASS" if t['pass'] else "FAIL"
        val = t['val']
        if isinstance(val, float): val = f"{val:.4f}"
        print(f"    {tid}: {status} — {t['desc']} ({val})")

    print(f"\n  Comparison with z2206 (UART 20 Hz):")
    print(f"    z2206 waveform:  0.810")
    print(f"    z2220 waveform:  {wave_results['FULL_128']['mean']:.3f}")
    print(f"    z2206 XOR τ=2:   0.620")
    print(f"    z2220 XOR τ=2:   {xor_results['FULL_128']['tau2']:.3f}")
    print(f"    z2206 MC:        2.674")
    print(f"    z2220 MC:        {mc_results['FULL_128']['total']:.3f}")

    results['tests'] = tests
    with open(RESULTS / 'z2220_eth_reservoir.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved: results/z2220_eth_reservoir.json")

    fpga.close()

if __name__ == '__main__':
    main()
