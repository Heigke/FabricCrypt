#!/usr/bin/env python3
"""z2218_quick_recurrence_sweep.py — Fast parameter sweep for recurrence

Quick iteration: 30 trials, 30 steps, XOR τ=2 only (known to work at 0.62 baseline).
Sweep alpha_rec and leak_rate to find the sweet spot.

Key insight from z2216/z2217:
  - z2216: rec_Vg=0.0004 (invisible) → chance
  - z2217v1: rec_Vg=0.40 (saturated) → chance
  - z2217v2: rec_Vg=0.01 (marginal) → chance
  - LEAK_RATE=0.3 kills memory: 0.3^5=0.002 after 5 steps

This script sweeps: alpha_rec × leak_rate to find where recurrence actually helps.
~10 min per full sweep.
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

N_NEURONS    = 128
BASE_VG      = 0.58
ALPHA_IN     = 0.25
BETA_1F      = 0.08
SAMPLE_HZ    = 20
SPECTRAL_RAD = 0.90
SPARSITY     = 0.10

# Fast settings
N_TRIALS     = 30
N_STEPS      = 30

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

def read_hwmon_power():
    try: return int(open(HWMON_POWER).read().strip()) / 1e6
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

def collect_noise_fast(duration_s=8, sample_hz=50):
    n = int(duration_s * sample_hz)
    interval = 1.0 / sample_hz
    power_s, jitter_s = [], []
    for i in range(n):
        p = read_hwmon_power()
        j = read_perf_jitter()
        if p is not None: power_s.append(p)
        jitter_s.append(j)
        time.sleep(interval)
    return power_s, jitter_s

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
    return W

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=2):
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
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

def generate_temporal_xor(n_trials, steps, tau, seed=42):
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

def run_trial(fpga, input_signal, noises, w_in, W_res, alpha_rec, leak_rate,
              mode='STATIC_REC', gamma_mod=0.20):
    """Single trial with recurrent feedback. Returns spike_out, h_out."""
    n_steps = len(input_signal)
    spike_out = np.zeros((n_steps, N_NEURONS))
    h_out = np.zeros((n_steps, N_NEURONS))

    prev_counts = None
    h = np.zeros(N_NEURONS)

    for t in range(n_steps):
        inp = input_signal[t]
        vg = np.full(N_NEURONS, BASE_VG) + ALPHA_IN * inp * w_in

        if mode != 'NO_REC':
            vg += alpha_rec * h

        vg = np.clip(vg, 0.05, 0.95)

        try:
            fpga.set_vg_all(vg.tolist())
        except:
            try: fpga.reconnect()
            except: pass

        time.sleep(0.025)  # 25ms integration

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
            if prev_counts is not None:
                for i in range(N_NEURONS):
                    delta = (counts[i] - prev_counts[i]) & 0xFFFF
                    if delta > 30000: delta = 0
                    spike_deltas[i] = delta
            prev_counts = counts[:]

        spike_out[t] = spike_deltas

        if mode != 'NO_REC':
            s_total = max(spike_deltas.sum(), 1.0)
            s_norm = spike_deltas / np.sqrt(s_total)

            if mode == 'FW_REC':
                power_data = noises.get('power', np.zeros(1))
                eta = power_data[t % len(power_data)] if len(power_data) > 0 else 0.0
                W_eff = W_res * (1.0 + gamma_mod * eta)
                h_raw = np.tanh(W_eff @ s_norm)
            else:
                h_raw = np.tanh(W_res @ s_norm)

            h = leak_rate * h + (1 - leak_rate) * h_raw

        h_out[t] = h

    return spike_out, h_out

def evaluate_xor(fpga, noises, w_in, W_res, alpha_rec, leak_rate, tau, mode):
    """Quick XOR evaluation. Returns accuracy."""
    inputs, targets = generate_temporal_xor(N_TRIALS, N_STEPS, tau)

    all_X, all_y = [], []
    for trial in range(N_TRIALS):
        spikes, h_states = run_trial(fpga, inputs[trial], noises, w_in, W_res,
                                      alpha_rec, leak_rate, mode=mode)
        for t_i in range(tau, N_STEPS):
            feat = np.concatenate([spikes[t_i], h_states[t_i]])
            all_X.append(feat)
            all_y.append(targets[trial][t_i])

    X = np.array(all_X)
    y = np.array(all_y)

    # Simple 70/30 split (fast, no CV)
    n = len(y)
    idx = np.random.default_rng(42).permutation(n)
    split = int(0.7 * n)
    acc = ridge_classify(X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]])
    return acc

def main():
    from fpga_host_v2 import FPGABridge

    print("=" * 60)
    print("z2218: Quick Recurrence Parameter Sweep")
    print("  XOR τ=2 (baseline 0.62), 30 trials × 30 steps")
    print("  Sweep alpha_rec × leak_rate")
    print("=" * 60)

    fpga = FPGABridge()
    if not fpga.connected:
        print("FATAL: No FPGA"); return
    print(f"Connected: {fpga.port}, {fpga.num_neurons} neurons")

    # Quick noise
    print("\nCollecting noise (8s)...")
    power_raw, jitter_raw = collect_noise_fast(8, 50)
    noises = {
        'power': iir_filter_noise(normalize_noise(power_raw)),
        'jitter': iir_filter_noise(normalize_noise(jitter_raw)),
    }
    print(f"  power: {len(noises['power'])} samples")

    rng = np.random.default_rng(42)
    w_in = rng.standard_normal(N_NEURONS)
    w_in /= np.linalg.norm(w_in)
    W_res = create_recurrent_matrix(N_NEURONS, SPARSITY, SPECTRAL_RAD, seed=42)

    results = {}

    # ─── Phase 1: NO_REC baseline (τ=2) ───
    print("\n--- PHASE 1: Baseline NO_REC ---")
    t0 = time.time()
    acc_baseline = evaluate_xor(fpga, noises, w_in, W_res, 0, 0, tau=2, mode='NO_REC')
    dt = time.time() - t0
    print(f"  NO_REC τ=2: {acc_baseline:.3f} ({dt:.0f}s)")
    results['baseline_tau2'] = acc_baseline

    # ─── Phase 2: Sweep alpha_rec with leak=0.8 (high memory) ───
    print("\n--- PHASE 2: Sweep alpha_rec (leak=0.80, τ=2) ---")
    for alpha in [0.05, 0.10, 0.20, 0.50, 1.0]:
        t0 = time.time()
        acc = evaluate_xor(fpga, noises, w_in, W_res, alpha, 0.80, tau=2, mode='STATIC_REC')
        dt = time.time() - t0
        tag = ">>>" if acc > acc_baseline + 0.02 else "   "
        print(f"  {tag} α={alpha:.2f}: {acc:.3f} (Δ={acc-acc_baseline:+.3f}, {dt:.0f}s)")
        results[f'alpha_{alpha:.2f}_leak_0.80'] = acc

    # ─── Phase 3: Sweep leak_rate with best alpha ───
    # Pick the alpha that worked best
    best_alpha = 0.20  # default
    best_acc = 0
    for alpha in [0.05, 0.10, 0.20, 0.50, 1.0]:
        key = f'alpha_{alpha:.2f}_leak_0.80'
        if results.get(key, 0) > best_acc:
            best_acc = results[key]
            best_alpha = alpha

    print(f"\n--- PHASE 3: Sweep leak_rate (alpha={best_alpha:.2f}, τ=2) ---")
    for leak in [0.30, 0.50, 0.70, 0.85, 0.95]:
        t0 = time.time()
        acc = evaluate_xor(fpga, noises, w_in, W_res, best_alpha, leak, tau=2, mode='STATIC_REC')
        dt = time.time() - t0
        tag = ">>>" if acc > acc_baseline + 0.02 else "   "
        print(f"  {tag} λ={leak:.2f}: {acc:.3f} (Δ={acc-acc_baseline:+.3f}, {dt:.0f}s)")
        results[f'alpha_{best_alpha:.2f}_leak_{leak:.2f}'] = acc

    # ─── Phase 4: Best params on harder tasks ───
    # Find best leak
    best_leak = 0.80
    best_acc2 = 0
    for leak in [0.30, 0.50, 0.70, 0.80, 0.85, 0.95]:
        key = f'alpha_{best_alpha:.2f}_leak_{leak:.2f}'
        if results.get(key, 0) > best_acc2:
            best_acc2 = results[key]
            best_leak = leak

    print(f"\n--- PHASE 4: Best params (α={best_alpha:.2f}, λ={best_leak:.2f}) on τ=3,5 ---")

    # τ=3
    t0 = time.time()
    acc_nr3 = evaluate_xor(fpga, noises, w_in, W_res, 0, 0, tau=3, mode='NO_REC')
    dt = time.time() - t0
    print(f"  NO_REC τ=3: {acc_nr3:.3f} ({dt:.0f}s)")

    t0 = time.time()
    acc_sr3 = evaluate_xor(fpga, noises, w_in, W_res, best_alpha, best_leak, tau=3, mode='STATIC_REC')
    dt = time.time() - t0
    tag = ">>>" if acc_sr3 > acc_nr3 + 0.02 else "   "
    print(f"  {tag} STATIC τ=3: {acc_sr3:.3f} (Δ={acc_sr3-acc_nr3:+.3f}, {dt:.0f}s)")

    # τ=5
    t0 = time.time()
    acc_nr5 = evaluate_xor(fpga, noises, w_in, W_res, 0, 0, tau=5, mode='NO_REC')
    dt = time.time() - t0
    print(f"  NO_REC τ=5: {acc_nr5:.3f} ({dt:.0f}s)")

    t0 = time.time()
    acc_sr5 = evaluate_xor(fpga, noises, w_in, W_res, best_alpha, best_leak, tau=5, mode='STATIC_REC')
    dt = time.time() - t0
    tag = ">>>" if acc_sr5 > acc_nr5 + 0.02 else "   "
    print(f"  {tag} STATIC τ=5: {acc_sr5:.3f} (Δ={acc_sr5-acc_nr5:+.3f}, {dt:.0f}s)")

    # FW_REC with best params
    t0 = time.time()
    acc_fw5 = evaluate_xor(fpga, noises, w_in, W_res, best_alpha, best_leak, tau=5, mode='FW_REC')
    dt = time.time() - t0
    tag = ">>>" if acc_fw5 > acc_sr5 + 0.01 else "   "
    print(f"  {tag} FW_REC τ=5: {acc_fw5:.3f} (Δ={acc_fw5-acc_nr5:+.3f}, {dt:.0f}s)")

    results.update({
        'best_alpha': best_alpha, 'best_leak': best_leak,
        'tau3_norec': acc_nr3, 'tau3_static': acc_sr3,
        'tau5_norec': acc_nr5, 'tau5_static': acc_sr5, 'tau5_fw': acc_fw5,
    })

    # ─── Phase 5: Spectral radius sweep (if time) ───
    print(f"\n--- PHASE 5: Spectral radius sweep (α={best_alpha:.2f}, λ={best_leak:.2f}, τ=2) ---")
    for rho in [0.50, 0.70, 0.95, 1.05, 1.20]:
        W_test = create_recurrent_matrix(N_NEURONS, SPARSITY, rho, seed=42)
        t0 = time.time()
        acc = evaluate_xor(fpga, noises, w_in, W_test, best_alpha, best_leak, tau=2, mode='STATIC_REC')
        dt = time.time() - t0
        tag = ">>>" if acc > acc_baseline + 0.02 else "   "
        print(f"  {tag} ρ={rho:.2f}: {acc:.3f} (Δ={acc-acc_baseline:+.3f}, {dt:.0f}s)")
        results[f'rho_{rho:.2f}'] = acc

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline NO_REC τ=2: {acc_baseline:.3f}")
    print(f"  Best params: α={best_alpha:.2f}, λ={best_leak:.2f}")
    print(f"  Best STATIC τ=2: {best_acc2:.3f} (Δ={best_acc2-acc_baseline:+.3f})")
    print(f"  STATIC τ=3: {acc_sr3:.3f} vs NO_REC {acc_nr3:.3f}")
    print(f"  STATIC τ=5: {acc_sr5:.3f} vs NO_REC {acc_nr5:.3f}")
    print(f"  FW_REC τ=5: {acc_fw5:.3f}")

    any_improvement = (best_acc2 > acc_baseline + 0.03 or
                       acc_sr3 > acc_nr3 + 0.03 or
                       acc_sr5 > acc_nr5 + 0.03)
    print(f"\n  RECURRENCE HELPS: {'YES' if any_improvement else 'NO'}")

    with open(RESULTS / 'z2218_quick_sweep.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Saved: results/z2218_quick_sweep.json")

    fpga.close()

if __name__ == '__main__':
    main()
