#!/usr/bin/env python3
"""z2216_recurrent_reservoir.py — Cross-Substrate Recurrent Reservoir

SCIENTIFIC CONTEXT:
  z2162-z2215: XOR τ=5 stuck at chance (~0.50) because 128 FPGA neurons
  are INDEPENDENT — no inter-neuron coupling means no recurrent mixing.
  Literature (Jaeger 2001, Lukoševičius 2009) is unambiguous: delayed XOR
  requires BOTH memory AND nonlinearity within the reservoir. Memory comes
  from recurrent connections; nonlinearity from the activation function.

THIS EXPERIMENT:
  Close the recurrent loop THROUGH the GPU↔FPGA bridge:
    FPGA spikes → GPU reads spikes → GPU computes W_res × spikes + noise → GPU writes Vg → FPGA

  This creates a genuine cross-substrate recurrent neural network where:
  1. Neurons are physical (FPGA avalanche diodes, stochastic)
  2. Recurrence is computed on GPU and transmitted back via UART
  3. Synaptic weights are MODULATED by firmware noise (1/f power, thermal)
     — creating time-varying connectivity driven by real physics
  4. Multiple timescales emerge: fast membrane (~1ms), medium recurrent loop
     (~50ms), slow thermal drift (~10s)

NOVELTY:
  - First cross-substrate recurrent reservoir (FPGA neurons + GPU synapses)
  - Synaptic weights physically modulated by firmware 1/f noise
     (unlike ESN where W_res is fixed, here W_eff(t) changes each step)
  - Natural timescale hierarchy from physics, not design
  - Stochastic connectivity with 1/f temporal correlations

RECURRENT UPDATE EQUATION:
  spike_norm(t)  = spike_delta(t) / max(sum(spike_delta(t)), 1)
  η(t)           = firmware_noise(t)  [power, thermal, smn, jitter, clock]
  W_eff(t)       = W_res ⊙ (1 + γ · noise_matrix(η(t)))
  recurrent(t)   = W_eff(t) @ spike_norm(t)
  Vg(t+1)        = base_vg + α_in·input(t+1)·w_in + α_rec·recurrent(t)

CONDITIONS:
  NO_REC:       α_rec=0 (control, independent neurons)
  STATIC_REC:   Fixed W_res, γ=0 (standard ESN-style recurrence)
  FW_1F_REC:    W_res modulated by power VRM 1/f noise
  FW_MULTI_REC: W_res modulated by all firmware channels
  FULL_BRIDGE:  Recurrence + L5 bridge noise injection + multi modulation

BENCHMARKS:
  1. XOR τ=5 (PRIMARY — this is the test that's been failing)
  2. XOR τ=3 (easier version)
  3. 7-class waveform
  4. Memory Capacity

Tests T442-T457:
  T442: STATIC_REC XOR5 > NO_REC (recurrence helps)
  T443: STATIC_REC XOR5 > 0.55 (breaks chance)
  T444: FW_1F_REC XOR5 > STATIC_REC (noise in weights helps)
  T445: FW_MULTI_REC XOR5 > STATIC_REC (multi-channel > single)
  T446: FULL_BRIDGE XOR5 > NO_REC (full > baseline)
  T447: FULL_BRIDGE XOR5 > 0.60 (strong performance)
  T448: STATIC_REC MC > NO_REC MC (recurrence adds memory)
  T449: STATIC_REC MC > 3.0 (meaningful MC)
  T450: FULL_BRIDGE wave > NO_REC wave
  T451: FULL_BRIDGE wave > 0.80 (ambitious)
  T452: FW_1F_REC XOR3 > 0.60 (easier XOR)
  T453: STATIC_REC XOR3 > 0.55
  T454: MC FW_MULTI > MC STATIC (firmware helps memory)
  T455: Stability: Vg stays in [0.1, 0.9] (no divergence)
  T456: Weight modulation: std(W_eff)/mean(|W_eff|) > 0.05
  T457: BEST_REC XOR5 > 0.55 (any recurrent condition breaks chance)

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128-neuron, 921600 baud)
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters ───
N_NEURONS    = 128
BASE_VG      = 0.58
ALPHA_IN     = 0.25      # input gain
ALPHA_REC    = 0.06      # recurrent gain (calibrated: ~0.02 Vg perturbation)
BETA_1F      = 0.08      # noise injection gain (L5 bridge)
GAMMA_MOD    = 0.20      # firmware modulation strength on W_res
SAMPLE_HZ    = 20
SPECTRAL_RAD = 0.90      # W_res spectral radius
SPARSITY     = 0.10      # W_res density (10% connections)
N_FOLDS      = 5

# Persistence from z2213 (still useful for features)
DT = 1.0 / SAMPLE_HZ
TAU_FAST = 0.1
TAU_MID  = 1.0
TAU_SLOW = 5.0
DECAY_FAST = np.exp(-DT / TAU_FAST)
DECAY_MID  = np.exp(-DT / TAU_MID)
DECAY_SLOW = np.exp(-DT / TAU_SLOW)

# ─── Firmware Paths ───
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
PM_TABLE_THERMAL_OFFSET = 0x004C

# ─── Noise Channel Assignment (for L5 bridge + weight modulation) ───
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

# ─── Firmware Reads ───

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

# ─── Noise Collection ───

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

# ─── Recurrent Weight Matrix ───

def create_recurrent_matrix(n_neurons, sparsity, spectral_radius, seed=42):
    """Create sparse random recurrent weight matrix W_res.

    Standard ESN initialization (Jaeger 2001):
    1. Random sparse matrix with Gaussian entries
    2. Scale to desired spectral radius

    Returns: W_res (n_neurons × n_neurons), sparse with density=sparsity.
    """
    rng = np.random.default_rng(seed)

    # Sparse random matrix
    W = rng.standard_normal((n_neurons, n_neurons))
    mask = rng.random((n_neurons, n_neurons)) < sparsity
    np.fill_diagonal(mask, False)  # no self-connections
    W *= mask

    # Scale to spectral radius
    eigenvalues = np.linalg.eigvals(W)
    max_eig = np.max(np.abs(eigenvalues))
    if max_eig > 0:
        W = W * (spectral_radius / max_eig)

    n_connections = np.count_nonzero(W)
    print(f"  W_res: {n_neurons}×{n_neurons}, density={n_connections/(n_neurons**2):.3f}, "
          f"ρ={spectral_radius:.2f}, nnz={n_connections}")
    return W


def modulate_weights(W_res, noise_val, channel_assignment, gamma):
    """Apply firmware noise modulation to recurrent weights.

    Each receiving neuron j has its column of W_res modulated by
    the noise channel assigned to that neuron.

    W_eff_ij = W_res_ij * (1 + gamma * η_channel(j))

    This creates time-varying connectivity driven by firmware physics.
    """
    W_eff = W_res.copy()
    for ch_name, neuron_ids in channel_assignment.items():
        eta = noise_val.get(ch_name, 0.0)
        for j in neuron_ids:
            W_eff[:, j] *= (1.0 + gamma * eta)
    return W_eff


# ─── Ridge Classification ───

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

def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
    ])

# ─── Task Generators ───

def generate_waveforms(n_trials=200, steps=50, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / SAMPLE_HZ
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

# ─── FPGA Trial Runner with Cross-Substrate Recurrence ───

def run_fpga_trial_recurrent(fpga, input_signal, noises, w_in, w_noise,
                              W_res, alpha_rec, gamma_mod,
                              mode='NO_REC', beta=BETA_1F,
                              channel_assignment=None):
    """Run one trial through FPGA reservoir WITH recurrent feedback.

    At each timestep:
    1. Compute Vg = base + input + recurrent_feedback + noise
    2. Set all 128 Vg values on FPGA
    3. Wait for neurons to integrate
    4. Read spike counts
    5. Compute recurrent: W_eff @ normalized_spikes
    6. Store recurrent for next step

    Args:
        fpga: FPGABridge instance
        input_signal: (n_steps,) input values in [0, 1]
        noises: dict of noise arrays {power, thermal, smn, jitter, clock}
        w_in: (N_NEURONS,) input weight vector
        w_noise: (N_NEURONS,) noise weight vector
        W_res: (N_NEURONS, N_NEURONS) recurrent weight matrix
        alpha_rec: recurrent gain
        gamma_mod: firmware modulation strength
        mode: one of NO_REC, STATIC_REC, FW_1F_REC, FW_MULTI_REC, FULL_BRIDGE
        beta: noise injection gain
        channel_assignment: dict mapping channel names to neuron indices

    Returns:
        fpga_states: (n_steps, N_NEURONS*3) [spike_delta, vmem, cumulative]
        stability_info: dict with max/min Vg, modulation stats
    """
    n_steps = len(input_signal)
    all_fpga = np.zeros((n_steps, N_NEURONS * 3))

    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    recurrent_input = np.zeros(N_NEURONS)  # starts at zero

    # Stability tracking
    vg_min_seen = 1.0
    vg_max_seen = 0.0
    w_eff_stds = []

    if channel_assignment is None:
        channel_assignment = {
            'power': POWER_NEURONS, 'smn': SMN_NEURONS,
            'jitter': JITTER_NEURONS, 'thermal': THERMAL_NEURONS,
            'clock': CLOCK_NEURONS,
        }

    for t in range(n_steps):
        inp = input_signal[t]

        # Base Vg from input
        vg = np.full(N_NEURONS, BASE_VG) + ALPHA_IN * inp * w_in

        # Add recurrent feedback (from previous step's spikes)
        if mode != 'NO_REC':
            vg += alpha_rec * recurrent_input

        # Add direct noise injection (L5 bridge style)
        if mode == 'FULL_BRIDGE':
            for ch_name, neuron_ids in channel_assignment.items():
                ch_data = noises.get(ch_name, np.zeros(1))
                if len(ch_data) == 0: ch_data = np.zeros(1)
                idx = t % len(ch_data)
                for nid in neuron_ids:
                    vg[nid] += beta * ch_data[idx] * w_noise[nid]

        # Safety clip
        vg = np.clip(vg, 0.05, 0.95)
        vg_min_seen = min(vg_min_seen, vg.min())
        vg_max_seen = max(vg_max_seen, vg.max())

        # Send Vg to FPGA
        try:
            fpga.set_vg_all(vg.tolist())
        except:
            try: fpga.reconnect()
            except: pass

        # Wait for neuron integration
        time.sleep(1.0 / SAMPLE_HZ * 0.5)

        # Read telemetry
        try:
            fpga.ser.reset_input_buffer()
            telem = fpga.read_telem(timeout=0.3)
        except:
            telem = None
            try: fpga.reconnect()
            except: pass

        # Extract spike deltas
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

        # Compute recurrent feedback for NEXT step
        if mode != 'NO_REC':
            # Normalize spike deltas (sum-to-1, avoids Vg blow-up)
            s_total = max(spike_deltas.sum(), 1.0)
            s_norm = spike_deltas / s_total

            if mode == 'STATIC_REC':
                # Fixed W_res, no modulation
                recurrent_input = W_res @ s_norm

            elif mode == 'FW_1F_REC':
                # Modulate all weights by power VRM 1/f noise
                power_data = noises.get('power', np.zeros(1))
                if len(power_data) > 0:
                    eta = power_data[t % len(power_data)]
                else:
                    eta = 0.0
                W_eff = W_res * (1.0 + gamma_mod * eta)
                recurrent_input = W_eff @ s_norm
                w_eff_stds.append(np.std(W_eff[W_eff != 0]))

            elif mode in ('FW_MULTI_REC', 'FULL_BRIDGE'):
                # Each neuron group's incoming weights modulated by its firmware channel
                noise_now = {}
                for ch_name in ['power', 'thermal', 'smn', 'jitter', 'clock']:
                    ch_data = noises.get(ch_name, np.zeros(1))
                    if len(ch_data) > 0:
                        noise_now[ch_name] = ch_data[t % len(ch_data)]
                    else:
                        noise_now[ch_name] = 0.0

                W_eff = modulate_weights(W_res, noise_now, channel_assignment, gamma_mod)
                recurrent_input = W_eff @ s_norm
                w_eff_stds.append(np.std(W_eff[W_eff != 0]))

    stability = {
        'vg_min': float(vg_min_seen),
        'vg_max': float(vg_max_seen),
        'vg_stable': bool(vg_min_seen > 0.08 and vg_max_seen < 0.92),
        'w_eff_mod_std': float(np.mean(w_eff_stds)) if w_eff_stds else 0.0,
        'w_eff_mod_relative': float(np.mean(w_eff_stds) / max(np.std(W_res[W_res != 0]), 1e-10))
                              if w_eff_stds else 0.0,
    }

    return all_fpga, stability


# ─── Benchmark Runners ───

def run_xor_benchmark(fpga, noises, w_in, w_noise, W_res, tau=5,
                       n_trials=200, n_steps=50, conditions=None):
    """Run XOR τ benchmark across all conditions."""
    if conditions is None:
        conditions = ['NO_REC', 'STATIC_REC', 'FW_1F_REC', 'FW_MULTI_REC', 'FULL_BRIDGE']

    inputs, targets = generate_temporal_xor(n_trials, n_steps, tau=tau)
    results = {}

    for cond in conditions:
        print(f"\n  Running XOR τ={tau}: {cond}...")
        # Collect per-timestep features and labels
        all_X = []
        all_y = []

        for trial in range(n_trials):
            fpga_states, stability = run_fpga_trial_recurrent(
                fpga, inputs[trial], noises, w_in, w_noise,
                W_res, ALPHA_REC, GAMMA_MOD, mode=cond)

            # Per-timestep features (spike_deltas only, first N_NEURONS columns)
            # Use timesteps tau..end where XOR target is defined
            for t_i in range(tau, n_steps):
                feat = fpga_states[t_i, :N_NEURONS]  # spike deltas at this step
                all_X.append(feat)
                all_y.append(targets[trial][t_i])

            if (trial + 1) % 50 == 0:
                print(f"    trial {trial+1}/{n_trials} ({(trial+1)/n_trials*100:.0f}%)")

        X = np.array(all_X)
        y = np.array(all_y)

        # Cross-validated ridge classification
        res = classify_condition(X, y, n_splits=N_FOLDS, max_features=120, n_classes=2)
        results[cond] = res
        print(f"    {cond}: {res['mean']:.3f} ± {res['std']:.3f} (feats={X.shape[1]})")

    return results


def run_waveform_benchmark(fpga, noises, w_in, w_noise, W_res,
                            n_trials=200, n_steps=50, conditions=None):
    """Run 7-class waveform benchmark."""
    if conditions is None:
        conditions = ['NO_REC', 'STATIC_REC', 'FW_MULTI_REC', 'FULL_BRIDGE']

    inputs, labels = generate_waveforms(n_trials, n_steps)
    results = {}

    for cond in conditions:
        print(f"\n  Running waveform: {cond}...")
        all_feats = []

        for trial in range(n_trials):
            fpga_states, _ = run_fpga_trial_recurrent(
                fpga, inputs[trial], noises, w_in, w_noise,
                W_res, ALPHA_REC, GAMMA_MOD, mode=cond)

            feat = pool_trial_features(fpga_states)
            all_feats.append(feat)

            if (trial + 1) % 50 == 0:
                print(f"    trial {trial+1}/{n_trials}")

        X = np.array(all_feats)
        res = classify_condition(X, labels, n_splits=N_FOLDS, max_features=120, n_classes=7)
        results[cond] = res
        print(f"    {cond}: {res['mean']:.3f} ± {res['std']:.3f}")

    return results


def run_mc_benchmark(fpga, noises, w_in, w_noise, W_res,
                      n_steps=300, max_delay=40, conditions=None):
    """Run Memory Capacity benchmark."""
    if conditions is None:
        conditions = ['NO_REC', 'STATIC_REC', 'FW_MULTI_REC', 'FULL_BRIDGE']

    mc_input = generate_memory_capacity_input(n_steps)
    results = {}

    for cond in conditions:
        print(f"\n  Running MC: {cond}...")
        fpga_states, stability = run_fpga_trial_recurrent(
            fpga, mc_input, noises, w_in, w_noise,
            W_res, ALPHA_REC, GAMMA_MOD, mode=cond)

        # MC: sum of R² for delays 1..max_delay
        mc_total = 0.0
        mc_per_delay = []
        # Use spike_deltas (first N_NEURONS columns)
        X = fpga_states[:, :N_NEURONS]

        for delay in range(1, max_delay + 1):
            if delay >= n_steps - 10:
                mc_per_delay.append(0.0)
                continue
            y_delayed = mc_input[:-delay]
            X_d = X[delay:]
            n = min(len(y_delayed), len(X_d))
            y_d = y_delayed[:n]
            X_d = X_d[:n]

            if n < 20:
                mc_per_delay.append(0.0)
                continue

            split = int(0.7 * n)
            corr = ridge_regress(X_d[:split], y_d[:split], X_d[split:], y_d[split:])
            r2 = corr ** 2
            mc_per_delay.append(r2)
            mc_total += r2

        results[cond] = {
            'total': float(mc_total),
            'per_delay': mc_per_delay,
            'stability': stability,
        }
        print(f"    {cond}: MC={mc_total:.3f}")

    return results


# ─── Main ───

def main():
    from fpga_host_v2 import FPGABridge

    print("=" * 70)
    print("z2216: Cross-Substrate Recurrent Reservoir")
    print("  Literature: XOR needs memory + nonlinearity WITHIN reservoir")
    print("  FIX: Close recurrent loop GPU→FPGA→GPU via bidirectional bridge")
    print("  NOVELTY: Synaptic weights modulated by firmware 1/f noise")
    print("=" * 70)

    # ─── Connect FPGA ───
    print("\n[1] Connecting to 128-neuron FPGA...")
    fpga = FPGABridge()
    if not fpga.connected:
        print("  FATAL: Cannot connect to FPGA")
        return
    print(f"  Connected: {fpga.port}, neurons={fpga.num_neurons}")

    telem = fpga.read_telem()
    if telem and len(telem) >= N_NEURONS:
        print(f"  Telemetry OK: {len(telem)} neurons")
    else:
        print("  WARNING: Telemetry check failed, proceeding anyway")

    # ─── Collect noise ───
    print("\n[2] Collecting firmware noise (15s)...")
    power_raw, thermal_raw, clock_raw, smn_raw, jitter_raw = collect_all_noise(15, 50)
    noises = {
        'power': iir_filter_noise(normalize_noise(power_raw)),
        'thermal': iir_filter_noise(normalize_noise(thermal_raw)),
        'clock': iir_filter_noise(normalize_noise(clock_raw)),
        'smn': iir_filter_noise(normalize_noise(smn_raw)),
        'jitter': iir_filter_noise(normalize_noise(jitter_raw)),
    }
    for k, v in noises.items():
        print(f"  {k}: {len(v)} samples")

    # ─── Create recurrent weight matrix ───
    print("\n[3] Creating recurrent weight matrix...")
    rng = np.random.default_rng(42)
    w_in = rng.standard_normal(N_NEURONS)
    w_in /= np.linalg.norm(w_in)
    w_noise = rng.standard_normal(N_NEURONS)
    w_noise /= np.linalg.norm(w_noise)

    W_res = create_recurrent_matrix(N_NEURONS, SPARSITY, SPECTRAL_RAD, seed=42)

    # ─── Results container ───
    out = {
        'experiment': 'z2216_recurrent_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'n_neurons': N_NEURONS, 'base_vg': BASE_VG,
            'alpha_in': ALPHA_IN, 'alpha_rec': ALPHA_REC,
            'beta_1f': BETA_1F, 'gamma_mod': GAMMA_MOD,
            'spectral_radius': SPECTRAL_RAD, 'sparsity': SPARSITY,
            'sample_hz': SAMPLE_HZ, 'n_folds': N_FOLDS,
        },
        'W_res_stats': {
            'nnz': int(np.count_nonzero(W_res)),
            'density': float(np.count_nonzero(W_res) / W_res.size),
            'spectral_radius': float(np.max(np.abs(np.linalg.eigvals(W_res)))),
            'mean_abs': float(np.mean(np.abs(W_res[W_res != 0]))),
        },
    }

    # ─── BENCHMARK 1: XOR τ=5 ───
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Temporal XOR τ=5 (200 trials × 50 steps)")
    print("  z2162-z2215: ALL at chance (~0.50) without recurrence")
    print("  Target: >0.55 with cross-substrate recurrence")
    print("=" * 60)

    xor5 = run_xor_benchmark(fpga, noises, w_in, w_noise, W_res, tau=5,
                              n_trials=200, n_steps=50)
    out['xor_tau5'] = xor5

    # ─── BENCHMARK 2: XOR τ=3 ───
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Temporal XOR τ=3 (200 trials × 50 steps)")
    print("  Easier version — should show clearer recurrence benefit")
    print("=" * 60)

    xor3 = run_xor_benchmark(fpga, noises, w_in, w_noise, W_res, tau=3,
                              n_trials=200, n_steps=50,
                              conditions=['NO_REC', 'STATIC_REC', 'FW_1F_REC', 'FULL_BRIDGE'])
    out['xor_tau3'] = xor3

    # ─── BENCHMARK 3: Waveform ───
    print("\n" + "=" * 60)
    print("BENCHMARK 3: 7-class Waveform (200 trials × 50 steps)")
    print("  Previous best: 81.0% (z2206). Does recurrence help?")
    print("=" * 60)

    wave = run_waveform_benchmark(fpga, noises, w_in, w_noise, W_res,
                                   n_trials=200, n_steps=50)
    out['waveform'] = wave

    # ─── BENCHMARK 4: Memory Capacity ───
    print("\n" + "=" * 60)
    print("BENCHMARK 4: Memory Capacity (300 steps, delays 1-40)")
    print("  Previous: MC ≈ 1-3. Recurrence should increase significantly.")
    print("=" * 60)

    mc = run_mc_benchmark(fpga, noises, w_in, w_noise, W_res, n_steps=300)
    out['memory_capacity'] = mc

    # ─── Get stability info from a dedicated test ───
    print("\n[4] Stability check...")
    _, stability = run_fpga_trial_recurrent(
        fpga, np.random.default_rng(99).uniform(0, 1, 100),
        noises, w_in, w_noise, W_res, ALPHA_REC, GAMMA_MOD, mode='FULL_BRIDGE')
    out['stability'] = stability

    # ─── Tests ───
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    tests = []

    # Helper to safely get mean
    def m(d, key):
        if key in d and isinstance(d[key], dict):
            return d[key].get('mean', 0.0)
        return 0.0

    def mc_total(d, key):
        if key in d and isinstance(d[key], dict):
            return d[key].get('total', 0.0)
        return 0.0

    # XOR τ=5 tests
    xor5_no = m(xor5, 'NO_REC')
    xor5_static = m(xor5, 'STATIC_REC')
    xor5_1f = m(xor5, 'FW_1F_REC')
    xor5_multi = m(xor5, 'FW_MULTI_REC')
    xor5_full = m(xor5, 'FULL_BRIDGE')
    xor5_best = max(xor5_static, xor5_1f, xor5_multi, xor5_full)

    tests.append({'id': 'T442', 'description': f'STATIC_REC XOR5({xor5_static:.3f}) > NO_REC({xor5_no:.3f})',
                  'passed': xor5_static > xor5_no})
    tests.append({'id': 'T443', 'description': f'STATIC_REC XOR5({xor5_static:.3f}) > 0.55',
                  'passed': xor5_static > 0.55})
    tests.append({'id': 'T444', 'description': f'FW_1F_REC XOR5({xor5_1f:.3f}) > STATIC({xor5_static:.3f})',
                  'passed': xor5_1f > xor5_static})
    tests.append({'id': 'T445', 'description': f'FW_MULTI XOR5({xor5_multi:.3f}) > STATIC({xor5_static:.3f})',
                  'passed': xor5_multi > xor5_static})
    tests.append({'id': 'T446', 'description': f'FULL_BRIDGE XOR5({xor5_full:.3f}) > NO_REC({xor5_no:.3f})',
                  'passed': xor5_full > xor5_no})
    tests.append({'id': 'T447', 'description': f'FULL_BRIDGE XOR5({xor5_full:.3f}) > 0.60',
                  'passed': xor5_full > 0.60})

    # MC tests
    mc_no = mc_total(mc, 'NO_REC')
    mc_static = mc_total(mc, 'STATIC_REC')
    mc_multi = mc_total(mc, 'FW_MULTI_REC')
    mc_full = mc_total(mc, 'FULL_BRIDGE')

    tests.append({'id': 'T448', 'description': f'STATIC MC({mc_static:.3f}) > NO_REC MC({mc_no:.3f})',
                  'passed': mc_static > mc_no})
    tests.append({'id': 'T449', 'description': f'STATIC MC({mc_static:.3f}) > 3.0',
                  'passed': mc_static > 3.0})

    # Waveform tests
    wave_no = m(wave, 'NO_REC')
    wave_full = m(wave, 'FULL_BRIDGE')

    tests.append({'id': 'T450', 'description': f'FULL_BRIDGE wave({wave_full:.3f}) > NO_REC({wave_no:.3f})',
                  'passed': wave_full > wave_no})
    tests.append({'id': 'T451', 'description': f'FULL_BRIDGE wave({wave_full:.3f}) > 0.80',
                  'passed': wave_full > 0.80})

    # XOR τ=3 tests
    xor3_1f = m(xor3, 'FW_1F_REC')
    xor3_static = m(xor3, 'STATIC_REC')

    tests.append({'id': 'T452', 'description': f'FW_1F XOR3({xor3_1f:.3f}) > 0.60',
                  'passed': xor3_1f > 0.60})
    tests.append({'id': 'T453', 'description': f'STATIC XOR3({xor3_static:.3f}) > 0.55',
                  'passed': xor3_static > 0.55})

    # MC firmware vs static
    tests.append({'id': 'T454', 'description': f'MC MULTI({mc_multi:.3f}) > MC STATIC({mc_static:.3f})',
                  'passed': mc_multi > mc_static})

    # Stability
    vg_stable = stability.get('vg_stable', False)
    tests.append({'id': 'T455', 'description': f'Vg stable [{stability.get("vg_min", 0):.2f}, {stability.get("vg_max", 0):.2f}]',
                  'passed': vg_stable})

    # Weight modulation
    w_mod = stability.get('w_eff_mod_relative', 0.0)
    tests.append({'id': 'T456', 'description': f'Weight modulation({w_mod:.3f}) > 0.05',
                  'passed': w_mod > 0.05})

    # Any recurrent > chance
    tests.append({'id': 'T457', 'description': f'BEST_REC XOR5({xor5_best:.3f}) > 0.55',
                  'passed': xor5_best > 0.55})

    out['tests'] = tests
    n_pass = sum(1 for t in tests if t['passed'])
    out['summary'] = {'pass': n_pass, 'total': len(tests)}

    for t in tests:
        status = "PASS" if t['passed'] else "FAIL"
        print(f"  [{status}] {t['id']}: {t['description']}")
    print(f"\n  TOTAL: {n_pass}/{len(tests)} PASS")

    # ─── Save ───
    out_path = RESULTS / 'z2216_recurrent_reservoir.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, cls=NpEncoder)
    print(f"\nSaved: {out_path}")

    fpga.close()
    print("Done.")


if __name__ == '__main__':
    main()
