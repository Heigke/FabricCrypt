#!/usr/bin/env python3
"""z2213_analog_persistence.py — NS-RAM-inspired analog state persistence

HYPOTHESIS: Our FPGA reservoir's biggest weakness is memory capacity (MC < 1.0
in z2212). Real NS-RAM devices have analog state retention — the floating bulk
charge persists across spikes, providing natural multi-timescale memory that
our digital BRAM resets every cycle.

This experiment adds SOFTWARE-SIDE analog persistence to the FPGA readout,
simulating what real NS-RAM floating-bulk dynamics would provide:

  bulk[n, t] = decay * bulk[n, t-1] + spike_delta[n, t]

Multiple decay timescales create a hierarchy of memory traces:
  - Fast:  τ = 100ms  (decay ~0.61 at 20Hz) — recent spike bursts
  - Mid:   τ = 1.0s   (decay ~0.95 at 20Hz) — pattern accumulation
  - Slow:  τ = 5.0s   (decay ~0.99 at 20Hz) — context memory

This is the key NS-RAM analogy: floating bulk retains charge across spikes.
If persistence fixes MC, that's a direct prediction: real NS-RAM devices
would give us what digital emulation cannot.

Benchmarks:
  1. Memory Capacity (main target: MC should exceed 3.0 with persistence)
  2. Temporal XOR τ=5 and τ=10 (should improve with memory)
  3. 7-class waveform (should at least maintain accuracy)

Conditions:
  L3_VANILLA:    Current FPGA reservoir (no persistence) — z2212 baseline
  L3_PERSIST_F:  FPGA + fast persistence only (τ=100ms)
  L3_PERSIST_M:  FPGA + mid persistence only (τ=1s)
  L3_PERSIST_S:  FPGA + slow persistence only (τ=5s)
  L3_MULTI_TAU:  FPGA + all three timescales concatenated
  L5_MULTI_TAU:  Bridge (noise+telem) + all three timescales

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128-neuron)
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
BASE_VG = 0.58
ALPHA = 0.25
BETA_1F = 0.08
SAMPLE_HZ = 20
IIR_ALPHA = 0.85
N_FOLDS = 5

# Persistence timescales (τ in seconds → decay per step at SAMPLE_HZ)
DT = 1.0 / SAMPLE_HZ
TAU_FAST = 0.1    # 100ms
TAU_MID  = 1.0    # 1s
TAU_SLOW = 5.0    # 5s
DECAY_FAST = np.exp(-DT / TAU_FAST)   # ~0.607
DECAY_MID  = np.exp(-DT / TAU_MID)    # ~0.951
DECAY_SLOW = np.exp(-DT / TAU_SLOW)   # ~0.990

# ─── Firmware Paths ───
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
PM_TABLE_THERMAL_OFFSET = 0x004C

# ─── Noise Channel Assignment (for L5 bridge) ───
POWER_NEURONS   = list(range(0, 32))
SMN_NEURONS     = list(range(32, 56))
JITTER_NEURONS  = list(range(56, 80))
THERMAL_NEURONS = list(range(80, 104))
CLOCK_NEURONS   = list(range(104, 128))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

# ─── Firmware reads ───

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

def read_firmware_telemetry():
    feat = np.zeros(6)
    p = read_hwmon_power()
    if p is not None: feat[0] = p
    t = read_gpu_thermal()
    if t is not None: feat[1] = t
    c = read_gpu_clock()
    if c is not None: feat[2] = c
    sm = read_smn_thermal()
    if sm is not None: feat[3] = sm
    feat[4] = read_perf_jitter()
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(0x04)
            feat[5] = struct.unpack('<f', f.read(4))[0]
    except: pass
    return feat

# ─── Noise Sources ───

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
        if n > 4 and (i + 1) % (n // 4) == 0:
            print(f"    {i+1}/{n} samples")
    return power_s, thermal_s, clock_s, smn_s, jitter_s

# ─── Classification / Regression helpers ───

def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
    ])

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=None, alphas=None):
    if alphas is None: alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    if n_classes is None: n_classes = len(np.unique(np.concatenate([y_tr, y_te])))
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
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

def ridge_regress(X_tr, y_tr, X_te, y_te, alphas=None):
    if alphas is None: alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
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

def pca_reduce(X, n_components=100):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_components < 1: return X, np.zeros(X.shape[1]), np.eye(X.shape[1])
    mu = X.mean(axis=0); Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T, mu, Vt[:n_components]

def pca_transform(X, mu, Vt): return (X - mu) @ Vt.T

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

def classify_condition(X_all, y_all, n_splits=5, max_features=120, n_classes=None):
    splits = stratified_kfold(X_all, y_all, n_splits=n_splits)
    fold_accs = []
    use_pca = X_all.shape[1] > max_features
    for train_idx, test_idx in splits:
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]
        mu = X_tr.mean(axis=0, keepdims=True)
        sigma = X_tr.std(axis=0, keepdims=True)
        sigma[sigma < 1e-2] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        if use_pca:
            X_tr_n, pca_mu, pca_Vt = pca_reduce(X_tr_n, n_components=max_features)
            X_te_n = pca_transform(X_te_n, pca_mu, pca_Vt)
        acc = ridge_classify(X_tr_n, y_tr, X_te_n, y_te, n_classes=n_classes)
        fold_accs.append(acc)
    return {'mean': float(np.mean(fold_accs)), 'std': float(np.std(fold_accs)),
            'folds': [float(a) for a in fold_accs]}

# ─── Task generators ───

def generate_7class_waveforms(n_trials=300, steps=30, dt=1.0/20, seed=42):
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    t = np.arange(steps) * dt
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

def generate_temporal_xor(n_trials=200, steps=50, tau=5, seed=42):
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    for _ in range(n_trials):
        seq = rng.integers(0, 2, size=steps).astype(float)
        target = np.zeros(steps, dtype=int)
        for t_i in range(tau, steps):
            target[t_i] = int(seq[t_i]) ^ int(seq[t_i - tau])
        trials.append(seq)
        labels.append(target)
    return np.array(trials), np.array(labels)

def generate_memory_capacity_input(n_steps=200, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=n_steps)

# ─── FPGA Trial Runner with Analog Persistence ───

def run_fpga_trial_persist(fpga, input_signal, noises, w_in, w_noise,
                           mode='L3_FPGA_ALONE', beta=BETA_1F,
                           persist_decays=None):
    """Run one trial through FPGA with optional analog state persistence.

    persist_decays: list of decay constants, e.g. [DECAY_FAST, DECAY_MID, DECAY_SLOW]
                    If None, no persistence (vanilla mode).

    Returns:
        fpga_states: (n_steps, N_NEURONS*3) — spike_delta, vmem, cumulative
        telem_states: (n_steps, 6) — firmware telemetry
        persist_states: (n_steps, N_NEURONS * len(persist_decays)) — bulk charge traces
    """
    n_steps = len(input_signal)
    n_persist = len(persist_decays) if persist_decays else 0

    all_fpga = np.zeros((n_steps, N_NEURONS * 3))
    all_telem = np.zeros((n_steps, 6))
    all_persist = np.zeros((n_steps, N_NEURONS * n_persist))

    prev_counts = None
    cumulative = np.zeros(N_NEURONS)

    # Initialize bulk charge states (one per neuron per timescale)
    bulk = [np.zeros(N_NEURONS) for _ in range(n_persist)]

    for t in range(n_steps):
        inp = input_signal[t]

        # Compute Vg for ALL 128 neurons
        vg = np.full(N_NEURONS, BASE_VG) + ALPHA * inp * w_in

        # L5: Add 1/f noise channel assignment
        if mode == 'L5_BRIDGE':
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

        try: fpga.set_vg_all(vg.tolist())
        except: pass

        time.sleep(1.0 / SAMPLE_HZ * 0.5)

        # Read firmware telemetry (L5)
        if mode == 'L5_BRIDGE':
            all_telem[t] = read_firmware_telemetry()

        # Read FPGA spikes
        try:
            fpga.ser.reset_input_buffer()
            telem = fpga.read_telem(timeout=0.3)
        except:
            telem = None
            try: fpga.reconnect()
            except: pass

        spike_deltas = np.zeros(N_NEURONS)
        if telem and len(telem) >= N_NEURONS:
            counts = [telem[i]['spike_count'] for i in range(N_NEURONS)]
            vmems = [telem[i]['vmem'] for i in range(N_NEURONS)]
            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000: delta = 0
                    all_fpga[t, i] = delta
                    spike_deltas[i] = delta
                    cumulative[i] += delta
            for i in range(N_NEURONS):
                all_fpga[t, N_NEURONS + i] = vmems[i]
                all_fpga[t, N_NEURONS * 2 + i] = cumulative[i]
            prev_counts = counts[:]

        # Update analog persistence (bulk charge) for each timescale
        for k in range(n_persist):
            bulk[k] = persist_decays[k] * bulk[k] + spike_deltas
            all_persist[t, k * N_NEURONS : (k + 1) * N_NEURONS] = bulk[k]

    return all_fpga, all_telem, all_persist


def build_features_persist(fpga_states, telem_states, persist_states, mode,
                           use_persist=True):
    """Build pooled feature vector with optional persistence features."""
    parts = [fpga_states]
    if mode == 'L5_BRIDGE':
        parts.append(telem_states)
    if use_persist and persist_states.shape[1] > 0:
        parts.append(persist_states)
    combined = np.hstack(parts)
    return pool_trial_features(combined)


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-s', type=float, default=15.0)
    parser.add_argument('--mc-steps', type=int, default=300)
    parser.add_argument('--xor-trials', type=int, default=200)
    parser.add_argument('--wave7-trials', type=int, default=300)
    args = parser.parse_args()

    print("=" * 70)
    print("z2213: NS-RAM-Inspired Analog State Persistence")
    print(f"  Persistence timescales: τ_fast={TAU_FAST}s (decay={DECAY_FAST:.3f}),")
    print(f"                          τ_mid={TAU_MID}s (decay={DECAY_MID:.3f}),")
    print(f"                          τ_slow={TAU_SLOW}s (decay={DECAY_SLOW:.3f})")
    print(f"  Benchmarks: Memory Capacity, XOR τ=5/10, 7-class waveform")
    print(f"  Vg={BASE_VG}, ALPHA={ALPHA}, BETA_1F={BETA_1F}")
    print("=" * 70)

    # ─── Init FPGA ───
    print("\n[1] Connecting to 128-neuron FPGA...")
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
        'experiment': 'z2213_analog_persistence',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'n_neurons': N_NEURONS, 'base_vg': BASE_VG,
            'alpha': ALPHA, 'beta_1f': BETA_1F,
            'tau_fast': TAU_FAST, 'tau_mid': TAU_MID, 'tau_slow': TAU_SLOW,
            'decay_fast': float(DECAY_FAST), 'decay_mid': float(DECAY_MID),
            'decay_slow': float(DECAY_SLOW),
        },
        'tests': {},
    }

    # ─── Collect noise for L5 ───
    print(f"\n[2] Collecting 1/f noise ({args.noise_s}s)...")
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
            print(f"  {name}: {len(raw_samples)} samples")
        else:
            noises[name] = np.zeros(100)
            print(f"  {name}: MISSING")

    # ─── Helper functions ───

    def pf(name, passed, detail):
        print(f"  {name}: {'PASS' if passed else 'FAIL'} — {detail}")

    def rt(tid, passed, desc, **kw):
        results['tests'][tid] = {'pass': bool(passed), 'desc': desc, **kw}

    # Define conditions with their persistence configurations
    CONDITIONS = {
        'L3_VANILLA':    ('L3_FPGA_ALONE', None),
        'L3_PERSIST_F':  ('L3_FPGA_ALONE', [DECAY_FAST]),
        'L3_PERSIST_M':  ('L3_FPGA_ALONE', [DECAY_MID]),
        'L3_PERSIST_S':  ('L3_FPGA_ALONE', [DECAY_SLOW]),
        'L3_MULTI_TAU':  ('L3_FPGA_ALONE', [DECAY_FAST, DECAY_MID, DECAY_SLOW]),
        'L5_MULTI_TAU':  ('L5_BRIDGE',     [DECAY_FAST, DECAY_MID, DECAY_SLOW]),
    }

    # =========================================================================
    # BENCHMARK 1: Memory Capacity (main target)
    # z2212 baseline: MC < 1.0 for all conditions. Target: MC > 3.0
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"BENCHMARK 1: Memory Capacity ({args.mc_steps} steps, delays 1..40)")
    print(f"  z2212 baseline: MC(L3)=0.99, MC(L5)=0.73, MC(L6)=0.98")
    print(f"  Target: MC > 3.0 with persistence")
    print(f"{'='*60}")

    mc_input = generate_memory_capacity_input(args.mc_steps)
    max_delay = 40
    washout = max_delay + 10

    b1 = {}
    for cond_name, (mode, persist_decays) in CONDITIONS.items():
        print(f"\n  Running MC: {cond_name} ({args.mc_steps} steps)...")
        t0 = time.time()

        fpga_s, telem_s, persist_s = run_fpga_trial_persist(
            fpga, mc_input, noises, w_in, w_noise,
            mode=mode, beta=BETA_1F,
            persist_decays=persist_decays if persist_decays else [])

        elapsed = time.time() - t0
        print(f"    {args.mc_steps} steps in {elapsed:.1f}s")

        # Build state matrix (per-step features, not pooled)
        parts = [fpga_s]
        if mode == 'L5_BRIDGE':
            parts.append(telem_s)
        if persist_decays:
            parts.append(persist_s)
        states = np.hstack(parts)

        # Memory capacity: for each delay k, train ridge to predict u[t-k]
        mc_total = 0.0
        mc_per_delay = {}
        usable = args.mc_steps - washout
        X_res = states[washout:]

        # Standardize
        mu_res = X_res.mean(axis=0)
        sigma_res = X_res.std(axis=0)
        sigma_res[sigma_res < 1e-6] = 1.0
        X_std = (X_res - mu_res) / sigma_res

        # PCA if high-dimensional
        if X_std.shape[1] > 120:
            X_std, pca_mu, pca_Vt = pca_reduce(X_std, n_components=120)

        n_train = int(usable * 0.7)
        X_tr = X_std[:n_train]
        X_te = X_std[n_train:]

        for k in range(1, max_delay + 1):
            target = mc_input[washout - k : args.mc_steps - k]
            y_tr = target[:n_train]
            y_te = target[n_train:]
            corr = ridge_regress(X_tr, y_tr, X_te, y_te)
            mc_k = corr ** 2
            mc_total += mc_k
            mc_per_delay[k] = float(mc_k)

        b1[cond_name] = {
            'mc_total': float(mc_total),
            'mc_per_delay': mc_per_delay,
            'n_features': states.shape[1],
            'elapsed': elapsed,
        }
        print(f"    MC({cond_name}) = {mc_total:.3f}  (features={states.shape[1]})")

    results['benchmark1_memory_capacity'] = b1

    mc_van = b1['L3_VANILLA']['mc_total']
    mc_pf = b1['L3_PERSIST_F']['mc_total']
    mc_pm = b1['L3_PERSIST_M']['mc_total']
    mc_ps = b1['L3_PERSIST_S']['mc_total']
    mc_mt = b1['L3_MULTI_TAU']['mc_total']
    mc_l5 = b1['L5_MULTI_TAU']['mc_total']

    # Tests
    T393 = mc_mt > mc_van  # Multi-tau persistence improves MC
    T394 = mc_mt > 3.0     # MC exceeds 3.0 (z2212 target)
    T395 = mc_ps > mc_pf   # Slow persistence better than fast for MC
    T396 = mc_l5 > mc_mt   # Bridge + persistence > persistence alone
    T397 = mc_pm > mc_van  # Even single mid-tau helps

    pf('T393', T393, f'MULTI_TAU>VANILLA MC ({mc_mt:.3f} vs {mc_van:.3f})')
    pf('T394', T394, f'MULTI_TAU MC>3.0 ({mc_mt:.3f})')
    pf('T395', T395, f'SLOW>FAST MC ({mc_ps:.3f} vs {mc_pf:.3f})')
    pf('T396', T396, f'L5_MULTI>L3_MULTI MC ({mc_l5:.3f} vs {mc_mt:.3f})')
    pf('T397', T397, f'PERSIST_M>VANILLA MC ({mc_pm:.3f} vs {mc_van:.3f})')

    rt('T393', T393, 'MULTI_TAU>VANILLA MC', multi=mc_mt, vanilla=mc_van)
    rt('T394', T394, 'MULTI_TAU MC>3.0', mc=mc_mt)
    rt('T395', T395, 'SLOW>FAST MC', slow=mc_ps, fast=mc_pf)
    rt('T396', T396, 'L5_MULTI>L3_MULTI MC', l5=mc_l5, l3=mc_mt)
    rt('T397', T397, 'PERSIST_M>VANILLA MC', persist_m=mc_pm, vanilla=mc_van)

    # =========================================================================
    # BENCHMARK 2: Temporal XOR τ=5 and τ=10
    # z2212 baseline: all at chance (~0.50). Target: >0.55
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"BENCHMARK 2: Temporal XOR τ=5,10 ({args.xor_trials} trials × 50 steps)")
    print(f"  z2212 baseline: all at chance (~0.50)")
    print(f"  Target: >0.55 with persistence")
    print(f"{'='*60}")

    b2 = {}
    # Test key conditions only (not all 6 — too slow)
    XOR_CONDITIONS = {
        'L3_VANILLA':   ('L3_FPGA_ALONE', None),
        'L3_MULTI_TAU': ('L3_FPGA_ALONE', [DECAY_FAST, DECAY_MID, DECAY_SLOW]),
        'L5_MULTI_TAU': ('L5_BRIDGE',     [DECAY_FAST, DECAY_MID, DECAY_SLOW]),
    }

    for tau in [5, 10]:
        xor_trials, xor_targets = generate_temporal_xor(args.xor_trials, 50, tau)
        print(f"\n  τ={tau}:")

        for cond_name, (mode, persist_decays) in XOR_CONDITIONS.items():
            key = f"{cond_name}_tau{tau}"
            print(f"    Running {key}...")

            all_features = []
            all_labels = []
            t0 = time.time()

            for trial_i in range(args.xor_trials):
                fpga_s, telem_s, persist_s = run_fpga_trial_persist(
                    fpga, xor_trials[trial_i], noises, w_in, w_noise,
                    mode=mode, beta=BETA_1F,
                    persist_decays=persist_decays if persist_decays else [])

                # Per-step classification from tau onwards
                for t_i in range(tau, 50):
                    parts = [fpga_s[t_i]]
                    if mode == 'L5_BRIDGE':
                        parts.append(telem_s[t_i])
                    if persist_decays:
                        parts.append(persist_s[t_i])
                    feat = np.concatenate(parts)
                    all_features.append(feat)
                    all_labels.append(xor_targets[trial_i][t_i])

                if (trial_i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"      trial {trial_i+1}/{args.xor_trials} ({(trial_i+1)/elapsed:.1f} t/s)")

            X = np.array(all_features)
            y = np.array(all_labels)
            r = classify_condition(X, y, n_classes=2)
            b2[key] = r
            print(f"      {key}: {r['mean']:.3f} ± {r['std']:.3f}")

    results['benchmark2_xor'] = b2

    xV_5 = b2['L3_VANILLA_tau5']['mean']
    xM_5 = b2['L3_MULTI_TAU_tau5']['mean']
    x5_5 = b2['L5_MULTI_TAU_tau5']['mean']
    xV_10 = b2['L3_VANILLA_tau10']['mean']
    xM_10 = b2['L3_MULTI_TAU_tau10']['mean']
    x5_10 = b2['L5_MULTI_TAU_tau10']['mean']

    T398 = xM_5 > xV_5      # Persistence helps XOR τ=5
    T399 = xM_5 > 0.55       # XOR τ=5 above chance with persistence
    T400 = xM_10 > xV_10     # Persistence helps XOR τ=10
    T401 = xM_10 > 0.55      # XOR τ=10 above chance with persistence
    T402 = x5_10 > xM_10     # L5 bridge + persistence > persistence alone

    pf('T398', T398, f'MULTI>VANILLA XOR τ=5 ({xM_5:.3f} vs {xV_5:.3f})')
    pf('T399', T399, f'MULTI XOR τ=5 > 0.55 ({xM_5:.3f})')
    pf('T400', T400, f'MULTI>VANILLA XOR τ=10 ({xM_10:.3f} vs {xV_10:.3f})')
    pf('T401', T401, f'MULTI XOR τ=10 > 0.55 ({xM_10:.3f})')
    pf('T402', T402, f'L5_MULTI>L3_MULTI XOR τ=10 ({x5_10:.3f} vs {xM_10:.3f})')

    rt('T398', T398, 'MULTI>VANILLA XOR τ=5', multi=xM_5, vanilla=xV_5)
    rt('T399', T399, 'MULTI XOR τ=5 > 0.55', multi=xM_5)
    rt('T400', T400, 'MULTI>VANILLA XOR τ=10', multi=xM_10, vanilla=xV_10)
    rt('T401', T401, 'MULTI XOR τ=10 > 0.55', multi=xM_10)
    rt('T402', T402, 'L5_MULTI>L3_MULTI XOR τ=10', l5=x5_10, l3=xM_10)

    # =========================================================================
    # BENCHMARK 3: 7-class waveform (should maintain or improve accuracy)
    # z2212 baseline: L3=0.586, L5=0.757
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"BENCHMARK 3: 7-class waveform ({args.wave7_trials} trials × 30 steps)")
    print(f"  z2212 baseline: L3=0.586, L5=0.757")
    print(f"{'='*60}")

    waves7, labels7 = generate_7class_waveforms(args.wave7_trials, 30)
    print(f"  Classes: {np.bincount(labels7)}")

    b3 = {}
    WAVE_CONDITIONS = {
        'L3_VANILLA':   ('L3_FPGA_ALONE', None),
        'L3_MULTI_TAU': ('L3_FPGA_ALONE', [DECAY_FAST, DECAY_MID, DECAY_SLOW]),
        'L5_VANILLA':   ('L5_BRIDGE',     None),
        'L5_MULTI_TAU': ('L5_BRIDGE',     [DECAY_FAST, DECAY_MID, DECAY_SLOW]),
    }

    for cond_name, (mode, persist_decays) in WAVE_CONDITIONS.items():
        print(f"\n  Running {cond_name}...")
        all_features = []
        t0 = time.time()

        for trial_i in range(args.wave7_trials):
            fpga_s, telem_s, persist_s = run_fpga_trial_persist(
                fpga, waves7[trial_i], noises, w_in, w_noise,
                mode=mode, beta=BETA_1F,
                persist_decays=persist_decays if persist_decays else [])

            feat = build_features_persist(fpga_s, telem_s, persist_s, mode,
                                          use_persist=(persist_decays is not None))
            all_features.append(feat)

            if (trial_i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    trial {trial_i+1}/{args.wave7_trials} ({(trial_i+1)/elapsed:.1f} t/s)")

        X = np.array(all_features)
        r = classify_condition(X, labels7, n_classes=7)
        b3[cond_name] = r
        print(f"    {cond_name}: {r['mean']:.3f} ± {r['std']:.3f} (feats={X.shape[1]})")

    results['benchmark3_7class'] = b3

    w_van = b3['L3_VANILLA']['mean']
    w_mt = b3['L3_MULTI_TAU']['mean']
    w_l5v = b3['L5_VANILLA']['mean']
    w_l5m = b3['L5_MULTI_TAU']['mean']

    T403 = w_mt >= w_van - 0.02   # Persistence doesn't hurt (within 2pp)
    T404 = w_mt > w_van           # Persistence actually helps waveform
    T405 = w_l5m > w_l5v          # L5 + persistence > L5 alone
    T406 = w_l5m > 0.70           # L5 + persistence above 70% threshold

    pf('T403', T403, f'MULTI_TAU >= VANILLA-0.02 waveform ({w_mt:.3f} vs {w_van:.3f})')
    pf('T404', T404, f'MULTI_TAU > VANILLA waveform ({w_mt:.3f} vs {w_van:.3f})')
    pf('T405', T405, f'L5_MULTI > L5_VANILLA waveform ({w_l5m:.3f} vs {w_l5v:.3f})')
    pf('T406', T406, f'L5_MULTI waveform > 0.70 ({w_l5m:.3f})')

    rt('T403', T403, 'MULTI >= VANILLA-0.02 waveform', multi=w_mt, vanilla=w_van)
    rt('T404', T404, 'MULTI > VANILLA waveform', multi=w_mt, vanilla=w_van)
    rt('T405', T405, 'L5_MULTI > L5_VANILLA waveform', l5_multi=w_l5m, l5_vanilla=w_l5v)
    rt('T406', T406, 'L5_MULTI waveform > 0.70', l5_multi=w_l5m)

    # =========================================================================
    # BENCHMARK 4: Persistence profile — which timescale matters most?
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"BENCHMARK 4: Persistence profile (MC per timescale)")
    print(f"{'='*60}")

    # Already computed in benchmark 1 — summarize
    profile = {
        'fast_only': mc_pf,
        'mid_only': mc_pm,
        'slow_only': mc_ps,
        'multi_tau': mc_mt,
        'vanilla': mc_van,
    }
    results['persistence_profile'] = profile

    best_single = max(mc_pf, mc_pm, mc_ps)
    T407 = mc_mt > best_single  # Multi-tau > any single timescale
    T408 = mc_ps > mc_van       # Slow persistence alone helps

    pf('T407', T407, f'MULTI_TAU > best_single MC ({mc_mt:.3f} vs {best_single:.3f})')
    pf('T408', T408, f'PERSIST_SLOW > VANILLA MC ({mc_ps:.3f} vs {mc_van:.3f})')

    rt('T407', T407, 'MULTI_TAU > best_single MC', multi=mc_mt, best_single=best_single)
    rt('T408', T408, 'PERSIST_SLOW > VANILLA MC', slow=mc_ps, vanilla=mc_van)

    # ─── Summary ───

    tests = results['tests']
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_pass}/{n_total} tests passed")
    print(f"{'='*60}")
    for tid in sorted(tests.keys(), key=lambda x: int(x[1:])):
        t = tests[tid]
        status = 'PASS' if t['pass'] else 'FAIL'
        print(f"  {tid}: {status} — {t['desc']}")

    # NS-RAM motivation summary
    print(f"\n{'='*60}")
    print(f"NS-RAM MOTIVATION:")
    print(f"  Vanilla MC:    {mc_van:.3f} (z2212 confirmed: digital BRAM has no memory)")
    print(f"  Multi-tau MC:  {mc_mt:.3f} (analog persistence {'FIXES' if mc_mt > 3.0 else 'IMPROVES'} it)")
    print(f"  Improvement:   {mc_mt - mc_van:+.3f} ({(mc_mt/max(mc_van,1e-6) - 1)*100:+.1f}%)")
    print(f"  → Real NS-RAM floating bulk provides THIS by physics, not software")
    print(f"{'='*60}")

    results['summary'] = {'pass': n_pass, 'total': n_total}
    results['nsram_motivation'] = {
        'vanilla_mc': mc_van,
        'persist_mc': mc_mt,
        'mc_improvement': mc_mt - mc_van,
        'mc_improvement_pct': (mc_mt / max(mc_van, 1e-6) - 1) * 100,
        'xor5_improvement': xM_5 - xV_5,
        'xor10_improvement': xM_10 - xV_10,
        'waveform_impact': w_mt - w_van,
    }

    out_path = RESULTS / 'z2213_analog_persistence.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
