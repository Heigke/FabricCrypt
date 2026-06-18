#!/usr/bin/env python3
"""z2214_nonlinear_persistence.py — Nonlinear analog persistence + cross-neuron coupling

z2213 FINDING: Linear exponential decay persistence improves MC (+40%) but CANNOT
solve XOR because it's a linear filter. XOR requires nonlinear computation.

THIS EXPERIMENT adds three nonlinear mechanisms to the persistence framework:

1. PRODUCT FEATURES: bulk_fast[n] × bulk_slow[n] creates multiplicative gates
   where recent activity modulates long-term context (biological: fast AMPA ×
   slow NMDA receptor interaction → nonlinear coincidence detection)

2. THRESHOLD-GATED ACCUMULATION: bulk only grows when spike_delta exceeds a
   threshold, creating a binary gate on the analog trace. This is closer to
   real NS-RAM write dynamics where charge injection requires above-threshold
   programming voltage.

3. CROSS-NEURON COUPLING: local interaction features — each neuron's persistence
   is modulated by neighbors' recent spikes. This creates lateral inhibition /
   excitation dynamics absent from independent-neuron persistence.

Together, these create a nonlinear kernel on top of the FPGA reservoir,
implementing what real NS-RAM crossbar arrays provide through physical
inter-device coupling and nonlinear charge dynamics.

Conditions:
  L3_VANILLA:      Baseline FPGA (no persistence)
  L3_LINEAR:       z2213-style linear multi-tau (control)
  L3_PRODUCT:      Linear + product interaction features
  L3_THRESHOLD:    Threshold-gated persistence
  L3_COUPLED:      Cross-neuron coupling persistence
  L3_FULL_NL:      All three nonlinear mechanisms combined
  L5_FULL_NL:      Bridge + all nonlinear mechanisms

Benchmarks:
  1. Memory Capacity (target: MC > 2.0 via nonlinear terms)
  2. Temporal XOR τ=5 (main target: >0.55 via nonlinear persistence)
  3. 7-class waveform (maintain/improve 0.669 from z2213)
  4. Nonlinear contribution analysis (which mechanism helps most?)

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

# Persistence timescales
DT = 1.0 / SAMPLE_HZ
TAU_FAST = 0.1
TAU_MID  = 1.0
TAU_SLOW = 5.0
DECAY_FAST = np.exp(-DT / TAU_FAST)
DECAY_MID  = np.exp(-DT / TAU_MID)
DECAY_SLOW = np.exp(-DT / TAU_SLOW)

# Nonlinear parameters
SPIKE_THRESHOLD = 2.0    # threshold-gated: only accumulate if delta > this
COUPLING_RADIUS = 4      # cross-neuron: couple with ±4 neighbors
COUPLING_STRENGTH = 0.3  # how much neighbor spikes modulate persistence

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

# ─── FPGA Trial Runner with NONLINEAR Persistence ───

def run_fpga_trial_nonlinear(fpga, input_signal, noises, w_in, w_noise,
                              mode='L3_FPGA_ALONE', beta=BETA_1F,
                              use_linear=True, use_product=False,
                              use_threshold=False, use_coupling=False):
    """Run one trial through FPGA with nonlinear analog state persistence.

    Nonlinear mechanisms:
      use_linear:    standard multi-tau exponential decay (z2213 baseline)
      use_product:   fast×slow product interaction features
      use_threshold: threshold-gated accumulation (only above SPIKE_THRESHOLD)
      use_coupling:  cross-neuron coupling (neighbor spike modulation)

    Returns:
        fpga_states:    (n_steps, N_NEURONS*3)
        telem_states:   (n_steps, 6)
        persist_linear: (n_steps, N_NEURONS*3) — linear multi-tau traces
        persist_nl:     (n_steps, n_nl_features) — nonlinear features
    """
    n_steps = len(input_signal)

    all_fpga = np.zeros((n_steps, N_NEURONS * 3))
    all_telem = np.zeros((n_steps, 6))

    # Linear persistence traces (3 timescales × N_NEURONS)
    bulk_fast = np.zeros(N_NEURONS)
    bulk_mid  = np.zeros(N_NEURONS)
    bulk_slow = np.zeros(N_NEURONS)
    all_linear = np.zeros((n_steps, N_NEURONS * 3))

    # Threshold-gated accumulator
    bulk_thresh = np.zeros(N_NEURONS)

    # Cross-neuron coupling state
    coupling_state = np.zeros(N_NEURONS)

    # Count nonlinear feature dimensions
    n_nl = 0
    if use_product:   n_nl += N_NEURONS * 2   # fast×slow, fast×mid
    if use_threshold: n_nl += N_NEURONS        # threshold-gated trace
    if use_coupling:  n_nl += N_NEURONS        # coupling trace
    all_nl = np.zeros((n_steps, n_nl))

    prev_counts = None
    cumulative = np.zeros(N_NEURONS)

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

        # === LINEAR PERSISTENCE (z2213 baseline) ===
        if use_linear:
            bulk_fast = DECAY_FAST * bulk_fast + spike_deltas
            bulk_mid  = DECAY_MID  * bulk_mid  + spike_deltas
            bulk_slow = DECAY_SLOW * bulk_slow + spike_deltas
            all_linear[t, :N_NEURONS] = bulk_fast
            all_linear[t, N_NEURONS:2*N_NEURONS] = bulk_mid
            all_linear[t, 2*N_NEURONS:] = bulk_slow

        # === NONLINEAR FEATURES ===
        offset = 0

        # 1. Product features: fast×slow and fast×mid (multiplicative gates)
        if use_product:
            # Normalize to prevent scale explosion
            bf_norm = bulk_fast / (np.abs(bulk_fast).max() + 1e-6)
            bm_norm = bulk_mid / (np.abs(bulk_mid).max() + 1e-6)
            bs_norm = bulk_slow / (np.abs(bulk_slow).max() + 1e-6)
            all_nl[t, offset : offset + N_NEURONS] = bf_norm * bs_norm
            offset += N_NEURONS
            all_nl[t, offset : offset + N_NEURONS] = bf_norm * bm_norm
            offset += N_NEURONS

        # 2. Threshold-gated accumulation
        if use_threshold:
            gate = (spike_deltas > SPIKE_THRESHOLD).astype(float)
            bulk_thresh = 0.95 * bulk_thresh + gate * spike_deltas
            all_nl[t, offset : offset + N_NEURONS] = bulk_thresh
            offset += N_NEURONS

        # 3. Cross-neuron coupling (circular neighbor modulation)
        if use_coupling:
            neighbor_sum = np.zeros(N_NEURONS)
            for r in range(1, COUPLING_RADIUS + 1):
                neighbor_sum += np.roll(spike_deltas, r) + np.roll(spike_deltas, -r)
            neighbor_sum /= (2 * COUPLING_RADIUS)
            coupling_state = 0.9 * coupling_state + COUPLING_STRENGTH * neighbor_sum * spike_deltas
            all_nl[t, offset : offset + N_NEURONS] = coupling_state
            offset += N_NEURONS

    return all_fpga, all_telem, all_linear, all_nl


def build_features_nl(fpga_states, telem_states, linear_states, nl_states,
                      mode, use_linear=True, use_nl=True):
    """Build pooled feature vector with linear + nonlinear persistence."""
    parts = [fpga_states]
    if mode == 'L5_BRIDGE':
        parts.append(telem_states)
    if use_linear and linear_states.shape[1] > 0:
        parts.append(linear_states)
    if use_nl and nl_states.shape[1] > 0:
        parts.append(nl_states)
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
    print("z2214: Nonlinear Persistence + Cross-Neuron Coupling")
    print(f"  z2213 finding: linear persistence helps MC (+40%) but NOT XOR")
    print(f"  This adds: (1) product gates, (2) threshold accumulation,")
    print(f"             (3) cross-neuron coupling")
    print(f"  Persistence: τ_fast={TAU_FAST}s, τ_mid={TAU_MID}s, τ_slow={TAU_SLOW}s")
    print(f"  Nonlinear: threshold={SPIKE_THRESHOLD}, coupling_r={COUPLING_RADIUS}")
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
        'experiment': 'z2214_nonlinear_persistence',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'n_neurons': N_NEURONS, 'base_vg': BASE_VG,
            'alpha': ALPHA, 'beta_1f': BETA_1F,
            'tau_fast': TAU_FAST, 'tau_mid': TAU_MID, 'tau_slow': TAU_SLOW,
            'spike_threshold': SPIKE_THRESHOLD,
            'coupling_radius': COUPLING_RADIUS,
            'coupling_strength': COUPLING_STRENGTH,
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

    # ─── Helpers ───
    def pf(name, passed, detail):
        print(f"  {name}: {'PASS' if passed else 'FAIL'} — {detail}")

    def rt(tid, passed, desc, **kw):
        results['tests'][tid] = {'pass': bool(passed), 'desc': desc, **kw}

    # Define conditions: (mode, use_linear, use_product, use_threshold, use_coupling)
    CONDITIONS = {
        'L3_VANILLA':    ('L3_FPGA_ALONE', False, False, False, False),
        'L3_LINEAR':     ('L3_FPGA_ALONE', True,  False, False, False),
        'L3_PRODUCT':    ('L3_FPGA_ALONE', True,  True,  False, False),
        'L3_THRESHOLD':  ('L3_FPGA_ALONE', True,  False, True,  False),
        'L3_COUPLED':    ('L3_FPGA_ALONE', True,  False, False, True),
        'L3_FULL_NL':    ('L3_FPGA_ALONE', True,  True,  True,  True),
        'L5_FULL_NL':    ('L5_BRIDGE',     True,  True,  True,  True),
    }

    # =========================================================================
    # BENCHMARK 1: Memory Capacity
    # z2213: VANILLA=0.808, MULTI_TAU=1.131. Target: >2.0 with nonlinear
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"BENCHMARK 1: Memory Capacity ({args.mc_steps} steps, delays 1..40)")
    print(f"  z2213: VANILLA=0.808, MULTI_TAU=1.131")
    print(f"  Target: MC > 2.0 with nonlinear persistence")
    print(f"{'='*60}")

    mc_input = generate_memory_capacity_input(args.mc_steps)
    max_delay = 40
    washout = max_delay + 10

    b1 = {}
    # For MC, test key conditions only (all 7 would take too long)
    MC_CONDITIONS = {k: v for k, v in CONDITIONS.items()
                     if k in ['L3_VANILLA', 'L3_LINEAR', 'L3_FULL_NL', 'L5_FULL_NL']}

    for cond_name, (mode, ul, up, ut, uc) in MC_CONDITIONS.items():
        print(f"\n  Running MC: {cond_name} ({args.mc_steps} steps)...")
        t0 = time.time()

        fpga_s, telem_s, linear_s, nl_s = run_fpga_trial_nonlinear(
            fpga, mc_input, noises, w_in, w_noise,
            mode=mode, beta=BETA_1F,
            use_linear=ul, use_product=up, use_threshold=ut, use_coupling=uc)

        elapsed = time.time() - t0
        print(f"    {args.mc_steps} steps in {elapsed:.1f}s")

        # Build state matrix (per-step features, not pooled)
        parts = [fpga_s]
        if mode == 'L5_BRIDGE': parts.append(telem_s)
        if ul: parts.append(linear_s)
        if nl_s.shape[1] > 0: parts.append(nl_s)
        states = np.hstack(parts)

        # Memory capacity
        mc_total = 0.0
        mc_per_delay = {}
        usable = args.mc_steps - washout
        X_res = states[washout:]

        mu_res = X_res.mean(axis=0)
        sigma_res = X_res.std(axis=0)
        sigma_res[sigma_res < 1e-6] = 1.0
        X_std = (X_res - mu_res) / sigma_res

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
    mc_lin = b1['L3_LINEAR']['mc_total']
    mc_nl  = b1['L3_FULL_NL']['mc_total']
    mc_l5  = b1['L5_FULL_NL']['mc_total']

    T409 = mc_nl > mc_lin       # Nonlinear > linear persistence
    T410 = mc_nl > 2.0          # MC exceeds 2.0 with nonlinear
    T411 = mc_l5 > mc_nl        # Bridge + nonlinear > nonlinear alone
    T412 = mc_nl > mc_van       # Nonlinear > vanilla

    pf('T409', T409, f'NL>LINEAR MC ({mc_nl:.3f} vs {mc_lin:.3f})')
    pf('T410', T410, f'NL MC>2.0 ({mc_nl:.3f})')
    pf('T411', T411, f'L5_NL>L3_NL MC ({mc_l5:.3f} vs {mc_nl:.3f})')
    pf('T412', T412, f'NL>VANILLA MC ({mc_nl:.3f} vs {mc_van:.3f})')

    rt('T409', T409, 'NL>LINEAR MC', nl=mc_nl, linear=mc_lin)
    rt('T410', T410, 'NL MC>2.0', mc=mc_nl)
    rt('T411', T411, 'L5_NL>L3_NL MC', l5=mc_l5, l3=mc_nl)
    rt('T412', T412, 'NL>VANILLA MC', nl=mc_nl, vanilla=mc_van)

    # =========================================================================
    # BENCHMARK 2: Temporal XOR τ=5 (MAIN TARGET — z2213 all at chance)
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"BENCHMARK 2: Temporal XOR τ=5 ({args.xor_trials} trials × 50 steps)")
    print(f"  z2213: ALL at chance (~0.50) — linear persistence can't help")
    print(f"  Target: >0.55 with nonlinear persistence")
    print(f"{'='*60}")

    xor_trials, xor_targets = generate_temporal_xor(args.xor_trials, 50, 5)

    b2 = {}
    # Test each nonlinear mechanism separately + combined
    XOR_CONDITIONS = {k: v for k, v in CONDITIONS.items()
                      if k in ['L3_VANILLA', 'L3_LINEAR', 'L3_PRODUCT',
                               'L3_THRESHOLD', 'L3_COUPLED', 'L3_FULL_NL', 'L5_FULL_NL']}

    for cond_name, (mode, ul, up, ut, uc) in XOR_CONDITIONS.items():
        print(f"\n  Running XOR: {cond_name}...")

        all_features = []
        all_labels = []
        t0 = time.time()

        for trial_i in range(args.xor_trials):
            fpga_s, telem_s, linear_s, nl_s = run_fpga_trial_nonlinear(
                fpga, xor_trials[trial_i], noises, w_in, w_noise,
                mode=mode, beta=BETA_1F,
                use_linear=ul, use_product=up, use_threshold=ut, use_coupling=uc)

            # Per-step classification from tau onwards
            for t_i in range(5, 50):
                parts = [fpga_s[t_i]]
                if mode == 'L5_BRIDGE':
                    parts.append(telem_s[t_i])
                if ul:
                    parts.append(linear_s[t_i])
                if nl_s.shape[1] > 0:
                    parts.append(nl_s[t_i])
                feat = np.concatenate(parts)
                all_features.append(feat)
                all_labels.append(xor_targets[trial_i][t_i])

            if (trial_i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    trial {trial_i+1}/{args.xor_trials} ({(trial_i+1)/elapsed:.1f} t/s)")

        X = np.array(all_features)
        y = np.array(all_labels)
        r = classify_condition(X, y, n_classes=2)
        b2[cond_name] = r
        print(f"    {cond_name}: {r['mean']:.3f} ± {r['std']:.3f} (feats={X.shape[1]})")

    results['benchmark2_xor'] = b2

    xV = b2['L3_VANILLA']['mean']
    xL = b2['L3_LINEAR']['mean']
    xP = b2['L3_PRODUCT']['mean']
    xT = b2['L3_THRESHOLD']['mean']
    xC = b2['L3_COUPLED']['mean']
    xN = b2['L3_FULL_NL']['mean']
    x5 = b2['L5_FULL_NL']['mean']

    T413 = xN > xL           # Full nonlinear > linear persistence
    T414 = xN > 0.55         # Above chance with nonlinear
    T415 = xP > xL           # Product features help XOR
    T416 = xT > xL           # Threshold gating helps XOR
    T417 = xC > xL           # Cross-neuron coupling helps XOR
    T418 = x5 > xN           # L5 bridge helps on top of nonlinear
    T419 = max(xP, xT, xC) > xV  # At least one mechanism helps

    pf('T413', T413, f'FULL_NL>LINEAR XOR ({xN:.3f} vs {xL:.3f})')
    pf('T414', T414, f'FULL_NL XOR > 0.55 ({xN:.3f})')
    pf('T415', T415, f'PRODUCT>LINEAR XOR ({xP:.3f} vs {xL:.3f})')
    pf('T416', T416, f'THRESHOLD>LINEAR XOR ({xT:.3f} vs {xL:.3f})')
    pf('T417', T417, f'COUPLED>LINEAR XOR ({xC:.3f} vs {xL:.3f})')
    pf('T418', T418, f'L5_NL>L3_NL XOR ({x5:.3f} vs {xN:.3f})')
    pf('T419', T419, f'BEST_MECH>VANILLA XOR ({max(xP,xT,xC):.3f} vs {xV:.3f})')

    rt('T413', T413, 'FULL_NL>LINEAR XOR', nl=xN, linear=xL)
    rt('T414', T414, 'FULL_NL XOR > 0.55', nl=xN)
    rt('T415', T415, 'PRODUCT>LINEAR XOR', product=xP, linear=xL)
    rt('T416', T416, 'THRESHOLD>LINEAR XOR', threshold=xT, linear=xL)
    rt('T417', T417, 'COUPLED>LINEAR XOR', coupled=xC, linear=xL)
    rt('T418', T418, 'L5_NL>L3_NL XOR', l5=x5, l3=xN)
    rt('T419', T419, 'BEST_MECH>VANILLA XOR', best=max(xP, xT, xC), vanilla=xV)

    # =========================================================================
    # BENCHMARK 3: 7-class waveform (maintain/improve z2213's 0.669)
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"BENCHMARK 3: 7-class waveform ({args.wave7_trials} trials × 30 steps)")
    print(f"  z2213: L3_VANILLA=0.621, L3_MULTI=0.669, L5_MULTI=0.764")
    print(f"{'='*60}")

    waves7, labels7 = generate_7class_waveforms(args.wave7_trials, 30)
    print(f"  Classes: {np.bincount(labels7)}")

    b3 = {}
    WAVE_CONDITIONS = {k: v for k, v in CONDITIONS.items()
                       if k in ['L3_VANILLA', 'L3_LINEAR', 'L3_FULL_NL', 'L5_FULL_NL']}

    for cond_name, (mode, ul, up, ut, uc) in WAVE_CONDITIONS.items():
        print(f"\n  Running {cond_name}...")
        all_features = []
        t0 = time.time()

        for trial_i in range(args.wave7_trials):
            fpga_s, telem_s, linear_s, nl_s = run_fpga_trial_nonlinear(
                fpga, waves7[trial_i], noises, w_in, w_noise,
                mode=mode, beta=BETA_1F,
                use_linear=ul, use_product=up, use_threshold=ut, use_coupling=uc)

            feat = build_features_nl(fpga_s, telem_s, linear_s, nl_s, mode,
                                     use_linear=ul, use_nl=(nl_s.shape[1] > 0))
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
    w_lin = b3['L3_LINEAR']['mean']
    w_nl  = b3['L3_FULL_NL']['mean']
    w_l5  = b3['L5_FULL_NL']['mean']

    T420 = w_nl > w_lin         # Nonlinear ≥ linear for waveform
    T421 = w_nl > 0.65          # Nonlinear above 65% threshold
    T422 = w_l5 > 0.70          # L5 + NL above 70%
    T423 = w_l5 > w_nl          # Bridge helps on top of NL

    pf('T420', T420, f'NL>LINEAR wave ({w_nl:.3f} vs {w_lin:.3f})')
    pf('T421', T421, f'NL wave > 0.65 ({w_nl:.3f})')
    pf('T422', T422, f'L5_NL wave > 0.70 ({w_l5:.3f})')
    pf('T423', T423, f'L5>L3 NL wave ({w_l5:.3f} vs {w_nl:.3f})')

    rt('T420', T420, 'NL>LINEAR wave', nl=w_nl, linear=w_lin)
    rt('T421', T421, 'NL wave > 0.65', nl=w_nl)
    rt('T422', T422, 'L5_NL wave > 0.70', l5=w_l5)
    rt('T423', T423, 'L5>L3 NL wave', l5=w_l5, l3=w_nl)

    # =========================================================================
    # BENCHMARK 4: Mechanism contribution analysis
    # Which nonlinear mechanism helps the most?
    # =========================================================================

    print(f"\n{'='*60}")
    print(f"BENCHMARK 4: Mechanism Contribution Analysis (XOR τ=5)")
    print(f"{'='*60}")

    mech_analysis = {
        'vanilla': xV, 'linear': xL,
        'product': xP, 'threshold': xT,
        'coupled': xC, 'full_nl': xN,
        'l5_full_nl': x5,
    }
    results['mechanism_analysis'] = mech_analysis

    best_single = max(xP, xT, xC)
    best_name = {xP: 'product', xT: 'threshold', xC: 'coupled'}[best_single]

    T424 = xN > best_single     # Combined > any single mechanism
    T425 = best_single > xV + 0.02  # Best mechanism > vanilla by 2pp

    pf('T424', T424, f'FULL_NL>BEST_SINGLE XOR ({xN:.3f} vs {best_single:.3f} [{best_name}])')
    pf('T425', T425, f'BEST({best_name})>VANILLA+0.02 ({best_single:.3f} vs {xV+0.02:.3f})')

    rt('T424', T424, f'FULL_NL>BEST_SINGLE XOR',
       full_nl=xN, best_single=best_single, best_name=best_name)
    rt('T425', T425, f'BEST_MECH > VANILLA+0.02',
       best=best_single, best_name=best_name, vanilla=xV)

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

    # Comparison with z2213
    print(f"\n{'='*60}")
    print(f"z2213 vs z2214 COMPARISON:")
    print(f"  MC:  z2213 linear={mc_lin:.3f}, z2214 nonlinear={mc_nl:.3f} ({mc_nl-mc_lin:+.3f})")
    print(f"  XOR: z2213 linear={xL:.3f},  z2214 nonlinear={xN:.3f} ({xN-xL:+.3f})")
    print(f"  Wave: z2213 linear={w_lin:.3f}, z2214 nonlinear={w_nl:.3f} ({w_nl-w_lin:+.3f})")
    print(f"  Best XOR mechanism: {best_name} ({best_single:.3f})")
    print(f"{'='*60}")

    results['summary'] = {'pass': n_pass, 'total': n_total}
    results['z2213_comparison'] = {
        'mc_linear': mc_lin, 'mc_nonlinear': mc_nl,
        'xor_linear': xL, 'xor_nonlinear': xN,
        'wave_linear': w_lin, 'wave_nonlinear': w_nl,
        'best_xor_mechanism': best_name,
        'best_xor_accuracy': best_single,
    }

    out_path = RESULTS / 'z2214_nonlinear_persistence.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
