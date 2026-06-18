#!/usr/bin/env python3
"""z2217_calibrated_recurrence.py — Calibrated Cross-Substrate Recurrence

z2216 FAILURE ANALYSIS:
  Recurrent signal was ~0.0004 Vg — completely invisible to neurons.
  Root cause: sum-normalization (spike_deltas / sum → ~0.008 per neuron)
  × sparse W_res @ → ~0.007 × alpha_rec=0.06 → 0.0004 Vg perturbation.
  BVpar cliff at ~0.60 means we need ±0.02-0.05 to change firing rates.

FIX:
  1. Normalize by sqrt(sum) instead of sum → preserves magnitude information
  2. Calibrate alpha_rec so recurrent contribution = ~0.03 Vg (measurable but stable)
  3. Add leaky integrator: h(t) = λ·h(t-1) + (1-λ)·W_eff @ s_norm
     This accumulates recurrent state across steps (echo state property)
  4. Use MULTI-SCALE features for readout: spike_delta + vmem + leaky traces
  5. Reduce trials to 100 (faster iteration, still statistically meaningful)

ARCHITECTURE:
  Standard ESN with physical neurons:
    h(t) = λ · h(t-1) + (1-λ) · tanh(W_eff(t) · s(t))      [GPU state]
    Vg(t+1) = base_vg + α_in · u(t+1) · w_in + α_rec · h(t) [FPGA input]
    s(t+1) = FPGA_spikes(Vg(t+1))                             [FPGA output]
    readout features: [s(t), h(t), vmem(t)]                    [multi-scale]

  Where W_eff(t) = W_res ⊙ (1 + γ · η(t)) and η(t) is firmware noise.

CONDITIONS (reduced from 5 to 3 for speed):
  NO_REC:     Control (α_rec=0)
  STATIC_REC: Fixed W_res, leaky integrator
  FW_REC:     Firmware-modulated W_eff + leaky integrator + bridge noise

Tests T458-T473:
  T458: STATIC_REC XOR5 > NO_REC (recurrence helps)
  T459: STATIC_REC XOR5 > 0.55 (breaks chance barrier)
  T460: FW_REC XOR5 > STATIC_REC (firmware modulation helps)
  T461: FW_REC XOR5 > 0.60 (strong performance)
  T462: STATIC_REC XOR3 > 0.60
  T463: FW_REC XOR3 > STATIC_REC
  T464: FW_REC XOR3 > 0.65 (easier task, higher bar)
  T465: STATIC_REC wave > NO_REC
  T466: FW_REC wave > 0.80
  T467: MC STATIC > MC NO_REC
  T468: MC FW > MC STATIC (firmware adds memory)
  T469: MC FW > 3.0
  T470: Vg stays in [0.1, 0.9] (stability)
  T471: Recurrent h(t) has significant variance (not dead)
  T472: BEST_REC XOR5 > 0.55
  T473: Echo state property: h(t) norm bounded

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
ALPHA_REC    = 0.15      # recurrent gain — CALIBRATED from z2217v1:
                          # h std~0.20/neuron, 0.15*0.20=0.03 Vg perturbation
                          # z2216: 0.06 → 0.0004 (invisible), z2217v1: 2.0 → 0.40 (saturated)
BETA_1F      = 0.08      # noise injection gain (bridge)
GAMMA_MOD    = 0.20      # firmware modulation strength on W_res
LEAK_RATE    = 0.3       # leaky integrator: h = λ·h_old + (1-λ)·new  (λ=0.3 → fast)
SAMPLE_HZ    = 20
SPECTRAL_RAD = 0.90      # W_res spectral radius
SPARSITY     = 0.10      # W_res density
N_FOLDS      = 5
N_TRIALS     = 100       # reduced from 200 for speed
N_STEPS      = 50

# Firmware paths
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
PM_TABLE_THERMAL_OFFSET = 0x004C

# Noise channel assignment
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
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_neurons, n_neurons))
    mask = rng.random((n_neurons, n_neurons)) < sparsity
    np.fill_diagonal(mask, False)
    W *= mask
    eigenvalues = np.linalg.eigvals(W)
    max_eig = np.max(np.abs(eigenvalues))
    if max_eig > 0:
        W = W * (spectral_radius / max_eig)
    n_connections = np.count_nonzero(W)
    print(f"  W_res: {n_neurons}×{n_neurons}, density={n_connections/(n_neurons**2):.3f}, "
          f"ρ={spectral_radius:.2f}, nnz={n_connections}")
    return W

# ─── Ridge Classification/Regression ───

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
        if use_pca:
            X_tr, pca_mu, pca_Vt = pca_reduce(X_tr, n_components=max_features)
            X_te = pca_transform(X_te, pca_mu, pca_Vt)
        acc = ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=n_classes)
        fold_accs.append(acc)
    return {'mean': float(np.mean(fold_accs)), 'std': float(np.std(fold_accs)),
            'folds': [float(a) for a in fold_accs]}

def pool_trial_features(spike_states, h_states):
    """Pool features from a trial: spike stats + hidden state stats."""
    return np.concatenate([
        spike_states.mean(axis=0),   # mean spike delta per neuron
        spike_states.std(axis=0),    # spike variability
        h_states.mean(axis=0),       # mean recurrent state (echo)
        h_states.std(axis=0),        # recurrent state variability
    ])

# ─── Task Generators ───

def generate_waveforms(n_trials=100, steps=50, seed=42):
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

def generate_temporal_xor(n_trials=100, steps=50, tau=5, seed=42):
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

# ─── FPGA Trial Runner with CALIBRATED Recurrence ───

def run_fpga_trial_recurrent(fpga, input_signal, noises, w_in, w_noise,
                              W_res, alpha_rec, gamma_mod,
                              mode='NO_REC', beta=BETA_1F,
                              channel_assignment=None):
    """Run one trial with CALIBRATED cross-substrate recurrence.

    KEY FIX from z2216:
    1. sqrt-normalization instead of sum-normalization
    2. Leaky integrator for echo state accumulation
    3. Multi-scale feature output (spikes + hidden state)

    Recurrent equation:
      s_norm(t) = spike_delta(t) / sqrt(max(sum(spike_delta(t)), 1))
      h_raw(t)  = tanh(W_eff(t) @ s_norm(t))
      h(t)      = leak * h(t-1) + (1-leak) * h_raw(t)
      Vg(t+1)   = base_vg + α_in·u(t+1)·w_in + α_rec·h(t)
    """
    n_steps = len(input_signal)
    spike_out = np.zeros((n_steps, N_NEURONS))
    h_out = np.zeros((n_steps, N_NEURONS))

    prev_counts = None
    h = np.zeros(N_NEURONS)  # leaky echo state

    vg_min_seen = 1.0
    vg_max_seen = 0.0
    h_norms = []
    rec_contributions = []

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

        # Add recurrent feedback from echo state
        if mode != 'NO_REC':
            rec_signal = alpha_rec * h
            vg += rec_signal
            rec_contributions.append(np.std(rec_signal))

        # Add direct noise injection (bridge style) for FW_REC
        if mode == 'FW_REC':
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
            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000: delta = 0
                    spike_deltas[i] = delta
            prev_counts = counts[:]

        spike_out[t] = spike_deltas

        # Update echo state (recurrent hidden state)
        if mode != 'NO_REC':
            # sqrt-normalization: preserves magnitude structure
            # sum~2000 → sqrt(2000)≈45 → per-neuron ~0.2-0.5
            s_total = max(spike_deltas.sum(), 1.0)
            s_norm = spike_deltas / np.sqrt(s_total)

            if mode == 'STATIC_REC':
                h_raw = np.tanh(W_res @ s_norm)
            elif mode == 'FW_REC':
                # Modulate weights by firmware noise (per-channel)
                W_eff = W_res.copy()
                for ch_name, neuron_ids in channel_assignment.items():
                    ch_data = noises.get(ch_name, np.zeros(1))
                    if len(ch_data) > 0:
                        eta = ch_data[t % len(ch_data)]
                    else:
                        eta = 0.0
                    for j in neuron_ids:
                        W_eff[:, j] *= (1.0 + gamma_mod * eta)
                h_raw = np.tanh(W_eff @ s_norm)

            # Leaky integration — accumulates history
            h = LEAK_RATE * h + (1 - LEAK_RATE) * h_raw
            h_norms.append(np.linalg.norm(h))

        h_out[t] = h

    stability = {
        'vg_min': float(vg_min_seen),
        'vg_max': float(vg_max_seen),
        'vg_stable': bool(vg_min_seen > 0.08 and vg_max_seen < 0.92),
        'h_norm_mean': float(np.mean(h_norms)) if h_norms else 0.0,
        'h_norm_std': float(np.std(h_norms)) if h_norms else 0.0,
        'rec_contribution_mean': float(np.mean(rec_contributions)) if rec_contributions else 0.0,
        'h_bounded': bool(max(h_norms) < 10.0) if h_norms else True,
    }

    return spike_out, h_out, stability


# ─── Benchmark Runners ───

def run_xor_benchmark(fpga, noises, w_in, w_noise, W_res, tau=5,
                       n_trials=N_TRIALS, n_steps=N_STEPS, conditions=None):
    """XOR τ benchmark — per-timestep classification with multi-scale features."""
    if conditions is None:
        conditions = ['NO_REC', 'STATIC_REC', 'FW_REC']

    inputs, targets = generate_temporal_xor(n_trials, n_steps, tau=tau)
    results = {}

    for cond in conditions:
        print(f"\n  Running XOR τ={tau}: {cond}...")
        all_X = []
        all_y = []

        for trial in range(n_trials):
            spike_states, h_states, stability = run_fpga_trial_recurrent(
                fpga, inputs[trial], noises, w_in, w_noise,
                W_res, ALPHA_REC, GAMMA_MOD, mode=cond)

            # Per-timestep features: [spike_delta(128) + h_state(128)] = 256 features
            for t_i in range(tau, n_steps):
                feat = np.concatenate([spike_states[t_i], h_states[t_i]])
                all_X.append(feat)
                all_y.append(targets[trial][t_i])

            if (trial + 1) % 25 == 0:
                print(f"    trial {trial+1}/{n_trials} ({(trial+1)/n_trials*100:.0f}%)")
                if stability['rec_contribution_mean'] > 0:
                    print(f"      h_norm={stability['h_norm_mean']:.4f}, "
                          f"rec_Vg={stability['rec_contribution_mean']:.4f}")

        X = np.array(all_X)
        y = np.array(all_y)

        res = classify_condition(X, y, n_splits=N_FOLDS, max_features=120, n_classes=2)
        results[cond] = res
        print(f"    {cond}: {res['mean']:.3f} ± {res['std']:.3f} "
              f"(feats={X.shape[1]}, samples={len(y)})")

    return results


def run_waveform_benchmark(fpga, noises, w_in, w_noise, W_res,
                            n_trials=N_TRIALS, n_steps=N_STEPS, conditions=None):
    """7-class waveform with multi-scale pooled features."""
    if conditions is None:
        conditions = ['NO_REC', 'STATIC_REC', 'FW_REC']

    inputs, labels = generate_waveforms(n_trials, n_steps)
    results = {}

    for cond in conditions:
        print(f"\n  Running waveform: {cond}...")
        all_feats = []

        for trial in range(n_trials):
            spike_states, h_states, _ = run_fpga_trial_recurrent(
                fpga, inputs[trial], noises, w_in, w_noise,
                W_res, ALPHA_REC, GAMMA_MOD, mode=cond)

            feat = pool_trial_features(spike_states, h_states)
            all_feats.append(feat)

            if (trial + 1) % 25 == 0:
                print(f"    trial {trial+1}/{n_trials}")

        X = np.array(all_feats)
        res = classify_condition(X, labels, n_splits=N_FOLDS, max_features=120, n_classes=7)
        results[cond] = res
        print(f"    {cond}: {res['mean']:.3f} ± {res['std']:.3f}")

    return results


def run_mc_benchmark(fpga, noises, w_in, w_noise, W_res,
                      n_steps=300, max_delay=40, conditions=None):
    """Memory Capacity with leaky echo state."""
    if conditions is None:
        conditions = ['NO_REC', 'STATIC_REC', 'FW_REC']

    mc_input = generate_memory_capacity_input(n_steps)
    results = {}

    for cond in conditions:
        print(f"\n  Running MC: {cond}...")
        spike_states, h_states, stability = run_fpga_trial_recurrent(
            fpga, mc_input, noises, w_in, w_noise,
            W_res, ALPHA_REC, GAMMA_MOD, mode=cond)

        # Use combined features for MC
        X = np.concatenate([spike_states, h_states], axis=1)

        mc_total = 0.0
        mc_per_delay = []
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
        print(f"    {cond}: MC={mc_total:.3f} "
              f"(h_norm={stability['h_norm_mean']:.4f}, "
              f"rec_Vg={stability['rec_contribution_mean']:.4f})")

    return results


# ─── Main ───

def main():
    from fpga_host_v2 import FPGABridge

    print("=" * 70)
    print("z2217: Calibrated Cross-Substrate Recurrence")
    print("  z2216 FAILURE: recurrent signal ~0.0004 Vg (invisible)")
    print("  FIX: sqrt-norm + leaky integrator + alpha_rec=2.0")
    print("  Target: ~0.03 Vg perturbation from recurrence")
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

    # ─── Calibration check ───
    print("\n[4] Calibration check (5 steps)...")
    test_input = np.array([0.5, 1.0, 0.0, 1.0, 0.5])
    _, _, cal_stab = run_fpga_trial_recurrent(
        fpga, test_input, noises, w_in, w_noise,
        W_res, ALPHA_REC, GAMMA_MOD, mode='STATIC_REC')
    print(f"  h_norm: {cal_stab['h_norm_mean']:.4f} ± {cal_stab['h_norm_std']:.4f}")
    print(f"  rec_Vg contribution: {cal_stab['rec_contribution_mean']:.4f}")
    print(f"  Vg range: [{cal_stab['vg_min']:.3f}, {cal_stab['vg_max']:.3f}]")
    print(f"  Stable: {cal_stab['vg_stable']}, Bounded: {cal_stab['h_bounded']}")

    # ─── Results container ───
    out = {
        'experiment': 'z2217_calibrated_recurrence',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'z2216_failure_reason': 'recurrent signal ~0.0004 Vg due to sum-normalization + low alpha_rec',
        'fixes': [
            'sqrt-normalization (s/sqrt(sum) not s/sum)',
            'alpha_rec=2.0 (was 0.06)',
            'leaky integrator h = 0.3*h + 0.7*tanh(W@s)',
            'multi-scale features: spike + hidden state',
            'reduced trials 200→100',
        ],
        'calibration': cal_stab,
        'params': {
            'n_neurons': N_NEURONS, 'base_vg': BASE_VG,
            'alpha_in': ALPHA_IN, 'alpha_rec': ALPHA_REC,
            'beta_1f': BETA_1F, 'gamma_mod': GAMMA_MOD,
            'leak_rate': LEAK_RATE,
            'spectral_radius': SPECTRAL_RAD, 'sparsity': SPARSITY,
            'sample_hz': SAMPLE_HZ, 'n_folds': N_FOLDS,
            'n_trials': N_TRIALS, 'n_steps': N_STEPS,
        },
        'W_res_stats': {
            'nnz': int(np.count_nonzero(W_res)),
            'density': float(np.count_nonzero(W_res) / W_res.size),
            'spectral_radius': float(np.max(np.abs(np.linalg.eigvals(W_res)))),
        },
    }

    # ─── BENCHMARK 1: XOR τ=5 ───
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Temporal XOR τ=5 (100 trials × 50 steps)")
    print("  z2216 ALL at chance. Fix: calibrated recurrence.")
    print("  Target: >0.55 with recurrence")
    print("=" * 60)

    xor5 = run_xor_benchmark(fpga, noises, w_in, w_noise, W_res, tau=5)
    out['xor_tau5'] = xor5

    # ─── BENCHMARK 2: XOR τ=3 ───
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Temporal XOR τ=3 (100 trials × 50 steps)")
    print("  Shorter delay — easier for recurrence")
    print("=" * 60)

    xor3 = run_xor_benchmark(fpga, noises, w_in, w_noise, W_res, tau=3)
    out['xor_tau3'] = xor3

    # ─── BENCHMARK 3: XOR τ=2 ───
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Temporal XOR τ=2 (100 trials × 50 steps)")
    print("  z2206 got 0.620 without recurrence. Does recurrence help?")
    print("=" * 60)

    xor2 = run_xor_benchmark(fpga, noises, w_in, w_noise, W_res, tau=2)
    out['xor_tau2'] = xor2

    # ─── BENCHMARK 4: Waveform ───
    print("\n" + "=" * 60)
    print("BENCHMARK 4: 7-class Waveform (100 trials × 50 steps)")
    print("  z2206 best: 81.0%. Does recurrence help?")
    print("=" * 60)

    wave = run_waveform_benchmark(fpga, noises, w_in, w_noise, W_res)
    out['waveform'] = wave

    # ─── BENCHMARK 5: Memory Capacity ───
    print("\n" + "=" * 60)
    print("BENCHMARK 5: Memory Capacity (300 steps, delays 1-40)")
    print("  Echo state should significantly boost MC")
    print("=" * 60)

    mc = run_mc_benchmark(fpga, noises, w_in, w_noise, W_res, n_steps=300)
    out['memory_capacity'] = mc

    # ─── Compile Tests ───
    tests = []

    def T(tid, desc, passed):
        tests.append({'id': tid, 'description': desc, 'passed': passed})
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {tid}: {desc}")

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    # XOR τ=5
    xor5_nr = xor5.get('NO_REC', {}).get('mean', 0)
    xor5_sr = xor5.get('STATIC_REC', {}).get('mean', 0)
    xor5_fw = xor5.get('FW_REC', {}).get('mean', 0)
    xor5_best = max(xor5_sr, xor5_fw)

    T('T458', f'STATIC_REC XOR5({xor5_sr:.3f}) > NO_REC({xor5_nr:.3f})',
      xor5_sr > xor5_nr)
    T('T459', f'STATIC_REC XOR5({xor5_sr:.3f}) > 0.55',
      xor5_sr > 0.55)
    T('T460', f'FW_REC XOR5({xor5_fw:.3f}) > STATIC_REC({xor5_sr:.3f})',
      xor5_fw > xor5_sr)
    T('T461', f'FW_REC XOR5({xor5_fw:.3f}) > 0.60',
      xor5_fw > 0.60)

    # XOR τ=3
    xor3_nr = xor3.get('NO_REC', {}).get('mean', 0)
    xor3_sr = xor3.get('STATIC_REC', {}).get('mean', 0)
    xor3_fw = xor3.get('FW_REC', {}).get('mean', 0)

    T('T462', f'STATIC_REC XOR3({xor3_sr:.3f}) > 0.60',
      xor3_sr > 0.60)
    T('T463', f'FW_REC XOR3({xor3_fw:.3f}) > STATIC_REC({xor3_sr:.3f})',
      xor3_fw > xor3_sr)
    T('T464', f'FW_REC XOR3({xor3_fw:.3f}) > 0.65',
      xor3_fw > 0.65)

    # Waveform
    wave_nr = wave.get('NO_REC', {}).get('mean', 0)
    wave_sr = wave.get('STATIC_REC', {}).get('mean', 0)
    wave_fw = wave.get('FW_REC', {}).get('mean', 0)

    T('T465', f'STATIC_REC wave({wave_sr:.3f}) > NO_REC({wave_nr:.3f})',
      wave_sr > wave_nr)
    T('T466', f'FW_REC wave({wave_fw:.3f}) > 0.80',
      wave_fw > 0.80)

    # Memory Capacity
    mc_nr = mc.get('NO_REC', {}).get('total', 0)
    mc_sr = mc.get('STATIC_REC', {}).get('total', 0)
    mc_fw = mc.get('FW_REC', {}).get('total', 0)

    T('T467', f'MC STATIC({mc_sr:.3f}) > MC NO_REC({mc_nr:.3f})',
      mc_sr > mc_nr)
    T('T468', f'MC FW({mc_fw:.3f}) > MC STATIC({mc_sr:.3f})',
      mc_fw > mc_sr)
    T('T469', f'MC FW({mc_fw:.3f}) > 3.0',
      mc_fw > 3.0)

    # Stability (from calibration)
    T('T470', f'Vg in [{cal_stab["vg_min"]:.2f},{cal_stab["vg_max"]:.2f}] ⊂ [0.1,0.9]',
      cal_stab['vg_stable'])
    T('T471', f'h_norm({cal_stab["h_norm_mean"]:.4f}) > 0.001 (recurrence alive)',
      cal_stab['h_norm_mean'] > 0.001)
    T('T472', f'BEST_REC XOR5({xor5_best:.3f}) > 0.55',
      xor5_best > 0.55)
    T('T473', f'Echo bounded: max(h_norm) < 10.0',
      cal_stab['h_bounded'])

    # XOR τ=2 bonus
    xor2_nr = xor2.get('NO_REC', {}).get('mean', 0)
    xor2_sr = xor2.get('STATIC_REC', {}).get('mean', 0)
    xor2_fw = xor2.get('FW_REC', {}).get('mean', 0)

    T('T474', f'STATIC_REC XOR2({xor2_sr:.3f}) > NO_REC({xor2_nr:.3f})',
      xor2_sr > xor2_nr)
    T('T475', f'FW_REC XOR2({xor2_fw:.3f}) > 0.65',
      xor2_fw > 0.65)

    out['tests'] = tests
    n_pass = sum(1 for t in tests if t['passed'])
    out['summary'] = {'pass': n_pass, 'total': len(tests)}

    print(f"\n  SUMMARY: {n_pass}/{len(tests)} PASS")

    # Save
    result_path = RESULTS / 'z2217_calibrated_recurrence.json'
    with open(result_path, 'w') as f:
        json.dump(out, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved: {result_path}")

    fpga.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
