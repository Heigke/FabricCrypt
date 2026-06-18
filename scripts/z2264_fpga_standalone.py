#!/usr/bin/env python3
"""z2264_fpga_standalone.py — Standalone FPGA Reservoir Benchmark

Pure FPGA reservoir benchmark (no GPU noise injection). Tests the intrinsic
computational capacity of 128 time-multiplexed LIF neurons on Arty A7-100T
driven only by waveform-modulated Vg.

Architecture:
  128 neurons, heterogeneous Vg spread ±0.08 around BASE_VG=0.58
  Input encoding: vg[n] = base_vg[n] + ALPHA * input_signal
  Readout: Ridge regression on [spike_delta | vmem | cumulative] × delay taps

Tasks:
  1. 4-class waveform classification (sine/square/triangle/sawtooth)
  2. Memory capacity (delays d=1..10)
  3. Temporal XOR (τ=1,2,3)

Hardware: Arty A7-100T FPGA via UDP Ethernet (192.168.0.50:7700)
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters ───
N_NEURONS = 128
BASE_VG = 0.58
VG_SPREAD = 0.08     # heterogeneous spread ±
ALPHA = 0.25          # input gain
SAMPLE_HZ = 200      # Ethernet telemetry rate

# ─── Waveform ───
N_WAVE_TRIALS = 120   # 30 per class
N_WAVE_STEPS = 100    # 0.5s at 200 Hz

# ─── XOR / MC ───
N_CONTINUOUS_STEPS = 2000  # for XOR and MC
MC_MAX_DELAY = 10


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# Ridge Classification / Regression
# ═══════════════════════════════════════════════════════════

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=None):
    if n_classes is None:
        n_classes = len(np.unique(np.concatenate([y_tr, y_te])))
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 100.0, 1000.0]
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_tr_s = (X_tr - mu) / sigma
    X_te_s = (X_te - mu) / sigma
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, y in enumerate(y_tr):
        Y_tr[i, int(y)] = 1.0
    best = -1
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try:
            W = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ Y_tr)
        except Exception:
            continue
        acc = np.mean(np.argmax(X_te_s @ W, axis=1) == y_te)
        if acc > best:
            best = acc
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
        try:
            w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except Exception:
            continue
        acc = np.mean(((X_te_s @ w) > 0.5).astype(float) == y_te)
        if acc > best:
            best = acc
    return best


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
    return {'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'folds': [float(a) for a in accs]}


# ═══════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2, 3)):
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
# Waveform Generation (4-class)
# ═══════════════════════════════════════════════════════════

def generate_waveforms_4class(n_trials, steps, sample_hz, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / sample_hz
    t = np.arange(steps) * dt
    trials, labels = [], []
    for _ in range(n_trials):
        cls = rng.integers(0, 4)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.8, 1.2)
        if cls == 0:    # sine
            wave = np.sin(2 * np.pi * freq * t + phase)
        elif cls == 1:  # square
            wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif cls == 2:  # triangle
            wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:           # sawtooth
            wave = 2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0
        # Normalize to [0, 1]
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
# Reservoir Core — Ethernet
# ═══════════════════════════════════════════════════════════

def run_reservoir_trial(fpga, input_signal, base_vg_per_neuron, w_in):
    """Drive 128-neuron FPGA reservoir for one trial (standalone, no GPU noise).

    vg[n] = base_vg[n] + ALPHA * input * w_in[n]

    Returns: (n_steps, N*3) array — [spike_delta | vmem | cumulative]
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)

    for t in range(n_steps):
        t_start = time.perf_counter()

        vg = base_vg_per_neuron + ALPHA * input_signal[t] * w_in
        vg = np.clip(vg, 0.05, 0.95)
        fpga.set_vg_batch(0, vg.tolist())

        # Integration time
        time.sleep(max(0.001, interval * 0.3))

        try:
            counts, vmem, refract = fpga.read_telemetry_fast()
        except (TimeoutError, Exception):
            continue

        if prev_counts is not None:
            for i in range(N_NEURONS):
                delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                if delta > 30000:
                    delta = 0
                states[t, i] = delta
                cumulative[i] += delta
        for i in range(N_NEURONS):
            states[t, N_NEURONS + i] = vmem[i]
            states[t, N_NEURONS * 2 + i] = cumulative[i]
        prev_counts = counts.copy()

        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.0005:
            time.sleep(remaining)

    return states


def run_reservoir_continuous(fpga, input_signal, base_vg_per_neuron, w_in):
    """Drive reservoir continuously for XOR/MC tasks.

    Returns: (n_steps, N*3) array
    """
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    states = np.zeros((n_steps, N_NEURONS * 3))
    prev_counts = None
    cumulative = np.zeros(N_NEURONS)
    actual_rates = []

    for t in range(n_steps):
        t_start = time.perf_counter()

        vg = base_vg_per_neuron + ALPHA * input_signal[t] * w_in
        vg = np.clip(vg, 0.05, 0.95)
        fpga.set_vg_batch(0, vg.tolist())

        time.sleep(max(0.001, interval * 0.3))

        try:
            counts, vmem, refract = fpga.read_telemetry_fast()
        except (TimeoutError, Exception):
            continue

        if prev_counts is not None:
            for i in range(N_NEURONS):
                delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                if delta > 30000:
                    delta = 0
                states[t, i] = delta
                cumulative[i] += delta
        for i in range(N_NEURONS):
            states[t, N_NEURONS + i] = vmem[i]
            states[t, N_NEURONS * 2 + i] = cumulative[i]
        prev_counts = counts.copy()

        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.0005:
            time.sleep(remaining)
        actual_rates.append(1.0 / max(time.perf_counter() - t_start, 1e-6))

        if n_steps > 100 and (t + 1) % (n_steps // 10) == 0:
            print(f"    step {t+1}/{n_steps} (rate={np.mean(actual_rates[-50:]):.0f} Hz)")

    mean_rate = float(np.mean(actual_rates)) if actual_rates else 0.0
    return states, mean_rate


# ═══════════════════════════════════════════════════════════
# Cross-Neuron Correlation
# ═══════════════════════════════════════════════════════════

def mean_off_diagonal_corr(states_list):
    corrs = []
    for states in states_list:
        spikes = states[:, :N_NEURONS]
        valid = [i for i in range(N_NEURONS) if spikes[:, i].std() > 1e-8]
        if len(valid) < 2:
            continue
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
    print("z2264: Standalone FPGA Reservoir Benchmark")
    print("  128 neurons, Ethernet 200 Hz, no GPU noise, pure FPGA")
    print("  Tasks: 4-class waveform, memory capacity (d=1-10), XOR (tau=1,2,3)")
    print("=" * 70)

    # ─── Connect ───
    print("\n[1] Connecting to FPGA via Ethernet...")
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("  FATAL: Cannot connect to FPGA")
        return
    print(f"  Connected: {fpga.num_neurons} neurons")

    # Kill switch off
    fpga.set_kill(False)
    time.sleep(0.2)

    # ─── Heterogeneous Vg setup ───
    rng = np.random.default_rng(42)
    base_vg_per_neuron = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, size=N_NEURONS)
    w_in = rng.uniform(-1, 1, size=N_NEURONS)

    print(f"\n[2] Setting heterogeneous Vg (spread ±{VG_SPREAD})...")
    print(f"  Vg range: [{base_vg_per_neuron.min():.3f}, {base_vg_per_neuron.max():.3f}]")
    fpga.set_vg_batch(0, base_vg_per_neuron.tolist())
    time.sleep(1.0)

    # Quick sanity: read telemetry
    telem = fpga.read_telemetry()
    if telem:
        sc = telem['spike_counts']
        print(f"  Sanity check: total_spikes={sc.sum()}, active={np.count_nonzero(sc)}/{N_NEURONS}")
    else:
        print("  WARNING: no telemetry response")

    results = {
        'experiment': 'z2264_fpga_standalone',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'params': {
            'n_neurons': N_NEURONS,
            'base_vg': BASE_VG,
            'vg_spread': VG_SPREAD,
            'alpha': ALPHA,
            'sample_hz': SAMPLE_HZ,
        },
    }

    # ═══════════════════════════════════════════════════════════
    # TASK 1: 4-CLASS WAVEFORM CLASSIFICATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TASK 1: 4-CLASS WAVEFORM CLASSIFICATION")
    print(f"  {N_WAVE_TRIALS} trials × {N_WAVE_STEPS} steps @ {SAMPLE_HZ} Hz")
    print("  Classes: sine, square, triangle, sawtooth")
    print("=" * 70)

    inputs, labels = generate_waveforms_4class(N_WAVE_TRIALS, N_WAVE_STEPS, SAMPLE_HZ)
    all_feats = []
    all_states = []

    for trial in range(N_WAVE_TRIALS):
        states = run_reservoir_trial(fpga, inputs[trial], base_vg_per_neuron, w_in)
        aug = augment_with_delays(states, delays=(1, 2, 3))
        feat = pool_trial_features(aug)
        all_feats.append(feat)
        all_states.append(states)

        if trial == 0:
            print(f"  Features per trial: {len(feat)}")
        if (trial + 1) % 30 == 0:
            print(f"  trial {trial+1}/{N_WAVE_TRIALS}")

    X = np.array(all_feats)
    wave_res = classify_cv(X, labels, n_splits=5, n_classes=4)
    corr = mean_off_diagonal_corr(all_states)
    print(f"\n  WAVEFORM ACCURACY: {wave_res['mean']:.3f} +/- {wave_res['std']:.3f}")
    print(f"  Cross-neuron correlation: {corr:.3f}")
    print(f"  Fold accs: {wave_res['folds']}")

    results['waveform'] = {
        'accuracy': wave_res['mean'],
        'std': wave_res['std'],
        'folds': wave_res['folds'],
        'cross_neuron_corr': corr,
    }

    # ═══════════════════════════════════════════════════════════
    # TASK 2: MEMORY CAPACITY (d=1..10)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TASK 2: MEMORY CAPACITY")
    print(f"  {N_CONTINUOUS_STEPS} continuous steps, delays d=1..{MC_MAX_DELAY}")
    print("=" * 70)

    mc_input = generate_xor_input(N_CONTINUOUS_STEPS, seed=123)
    mc_states, mc_rate = run_reservoir_continuous(fpga, mc_input, base_vg_per_neuron, w_in)
    print(f"  Mean sample rate: {mc_rate:.0f} Hz")

    # Augment with delays
    mc_aug = augment_with_delays(mc_states, delays=(1, 2, 3, 4, 5))
    warmup = 50

    mc_per_delay = {}
    mc_total = 0.0
    for d in range(1, MC_MAX_DELAY + 1):
        # Target: input delayed by d steps
        target = np.zeros(N_CONTINUOUS_STEPS)
        target[d:] = mc_input[:N_CONTINUOUS_STEPS - d]

        X = mc_aug[warmup:]
        y = target[warmup:]

        # 70/30 split
        n_tr = int(0.7 * len(X))
        X_tr, X_te = X[:n_tr], X[n_tr:]
        y_tr, y_te = y[:n_tr], y[n_tr:]

        # Ridge regression correlation
        alphas = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0)
        sigma[sigma < 1e-2] = 1.0
        X_tr_s = (X_tr - mu) / sigma
        X_te_s = (X_te - mu) / sigma

        best_r2 = 0.0
        for a in alphas:
            I = np.eye(X_tr_s.shape[1])
            try:
                w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
            except Exception:
                continue
            pred = X_te_s @ w
            if np.std(pred) > 1e-10 and np.std(y_te) > 1e-10:
                r = np.corrcoef(pred, y_te)[0, 1]
                r2 = max(r ** 2, 0.0)
                if r2 > best_r2:
                    best_r2 = r2

        mc_per_delay[d] = best_r2
        mc_total += best_r2
        print(f"  d={d:2d}: r²={best_r2:.3f}")

    print(f"\n  MEMORY CAPACITY TOTAL: {mc_total:.3f}")

    results['memory_capacity'] = {
        'per_delay': mc_per_delay,
        'total': mc_total,
        'n_steps': N_CONTINUOUS_STEPS,
        'mean_rate_hz': mc_rate,
    }

    # ═══════════════════════════════════════════════════════════
    # TASK 3: TEMPORAL XOR (tau=1,2,3)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TASK 3: TEMPORAL XOR")
    print(f"  {N_CONTINUOUS_STEPS} continuous steps, tau=1,2,3")
    print("=" * 70)

    xor_input = generate_xor_input(N_CONTINUOUS_STEPS, seed=456)
    xor_states, xor_rate = run_reservoir_continuous(fpga, xor_input, base_vg_per_neuron, w_in)
    print(f"  Mean sample rate: {xor_rate:.0f} Hz")

    xor_aug = augment_with_delays(xor_states, delays=(1, 2, 3, 4, 5))

    xor_results = {}
    for tau in [1, 2, 3]:
        targets = compute_xor_targets(xor_input, tau)
        X = xor_aug[warmup:]
        y = targets[warmup:].astype(float)

        n_tr = int(0.7 * len(X))
        X_tr, X_te = X[:n_tr], X[n_tr:]
        y_tr, y_te = y[:n_tr], y[n_tr:]

        acc = ridge_binary(X_tr, y_tr, X_te, y_te)
        xor_results[tau] = acc
        print(f"  tau={tau}: accuracy={acc:.3f}")

    results['temporal_xor'] = {str(k): v for k, v in xor_results.items()}

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY — z2264 Standalone FPGA Reservoir Benchmark")
    print("=" * 70)
    print(f"  Waveform 4-class accuracy:  {wave_res['mean']:.3f} +/- {wave_res['std']:.3f}")
    print(f"  Cross-neuron correlation:   {corr:.3f}")
    print(f"  Memory Capacity total:      {mc_total:.3f}")
    for d in range(1, MC_MAX_DELAY + 1):
        print(f"    MC d={d:2d}: {mc_per_delay[d]:.3f}")
    for tau in [1, 2, 3]:
        print(f"  XOR tau={tau}: {xor_results[tau]:.3f}")

    # Save results
    json_path = RESULTS / 'z2264_fpga_standalone.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  JSON: {json_path}")

    txt_path = RESULTS / 'z2264_fpga_standalone.txt'
    with open(txt_path, 'w') as f:
        f.write("z2264: Standalone FPGA Reservoir Benchmark\n")
        f.write(f"Date: {results['timestamp']}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  N_NEURONS = {N_NEURONS}\n")
        f.write(f"  BASE_VG = {BASE_VG}\n")
        f.write(f"  VG_SPREAD = +/-{VG_SPREAD}\n")
        f.write(f"  ALPHA = {ALPHA}\n")
        f.write(f"  SAMPLE_HZ = {SAMPLE_HZ}\n\n")
        f.write("TASK 1: 4-Class Waveform Classification\n")
        f.write(f"  Accuracy: {wave_res['mean']:.4f} +/- {wave_res['std']:.4f}\n")
        f.write(f"  Folds: {wave_res['folds']}\n")
        f.write(f"  Cross-neuron correlation: {corr:.4f}\n\n")
        f.write("TASK 2: Memory Capacity\n")
        f.write(f"  Total MC: {mc_total:.4f}\n")
        for d in range(1, MC_MAX_DELAY + 1):
            f.write(f"  d={d:2d}: r^2 = {mc_per_delay[d]:.4f}\n")
        f.write(f"\nTASK 3: Temporal XOR\n")
        for tau in [1, 2, 3]:
            f.write(f"  tau={tau}: accuracy = {xor_results[tau]:.4f}\n")
        f.write("\n")
    print(f"  TXT: {txt_path}")

    fpga.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
