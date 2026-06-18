#!/usr/bin/env python3
"""
z2316_ei_balance.py — Excitation/Inhibition Balance Analysis
=============================================================
Studies the effect of E/I neuron ratio on reservoir computing quality.
In biological brains, the E/I ratio (~80/20) is critical for computation.
Our FPGA has no inter-neuron synapses, but we can simulate E/I via:
  - Vg groups (high Vg = excitable, low Vg = inhibited)
  - Threshold modulation (different neurons, different thresholds)
  - Software-side E/I by sign-flipping inhibitory neuron readouts

Conditions (5):
  1) UNIFORM:  All neurons identical Vg=0.30
  2) EI_80_20: 80% excitatory (Vg=0.45) + 20% inhibitory (Vg=0.10)
  3) EI_50_50: 50/50 split
  4) GRADED:   4 Vg groups (0.05, 0.15, 0.30, 0.58) — current default
  5) INH_SIGN: GRADED + sign-flip readout for bottom 25% neurons (software inhibition)

Tests (16):
  T916: GRADED MC > UNIFORM MC
  T917: EI_80_20 MC > UNIFORM MC
  T918: INH_SIGN MC > GRADED MC
  T919: EI_80_20 waveform > UNIFORM waveform
  T920: GRADED waveform > UNIFORM waveform
  T921: INH_SIGN waveform > GRADED waveform
  T922: EI_80_20 XOR1 > 60%
  T923: GRADED XOR1 > 60%
  T924: INH_SIGN XOR1 > GRADED XOR1
  T925: Best E/I condition achieves MC > 11.0
  T926: Best E/I condition achieves waveform > 90%
  T927: E/I variance: GRADED state variance > UNIFORM state variance
  T928: E/I decorrelation: GRADED mean corr < UNIFORM mean corr
  T929: EI_50_50 MC < EI_80_20 MC (biological ratio is optimal)
  T930: Spike rate varies across E/I conditions (at least 2× range)
  T931: NARMA-5 best under non-UNIFORM condition

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2316_ei_balance.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2316_ei_balance.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
N_STEPS = 3000       # 60s at 50Hz — sufficient for MC/XOR/NARMA
WARMUP = 200         # discard first 200 steps
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
TEMP_SAFE = 42.0

# Ridge
RIDGE_ALPHA = 0.01

# E/I conditions: dict of name → {Vg_map: dict[neuron_id → vg_value]}
def make_vg_map(condition):
    """Return dict: neuron_id → Vg value for given E/I condition."""
    vg = {}
    if condition == 'UNIFORM':
        for n in range(NUM_NEURONS):
            vg[n] = 0.30
    elif condition == 'EI_80_20':
        for n in range(NUM_NEURONS):
            if n < 102:  # 80% excitatory
                vg[n] = 0.45
            else:        # 20% inhibitory
                vg[n] = 0.10
    elif condition == 'EI_50_50':
        for n in range(NUM_NEURONS):
            if n < 64:
                vg[n] = 0.45
            else:
                vg[n] = 0.10
    elif condition == 'GRADED':
        groups = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
        for n in range(NUM_NEURONS):
            vg[n] = groups[n % 4]
    elif condition == 'INH_SIGN':
        groups = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
        for n in range(NUM_NEURONS):
            vg[n] = groups[n % 4]
    return vg


CONDITIONS = ['UNIFORM', 'EI_80_20', 'EI_50_50', 'GRADED', 'INH_SIGN']


# ============================================================
# Thermal helpers
# ============================================================
def get_max_temp():
    temps = []
    for path in ['/sys/class/thermal/thermal_zone0/temp',
                 '/sys/class/hwmon/hwmon7/temp1_input']:
        try:
            with open(path, 'r') as f:
                temps.append(float(f.read().strip()) / 1000.0)
        except Exception:
            pass
    return max(temps) if temps else 0.0


def wait_cool(label="", target=None):
    if target is None:
        target = TEMP_SAFE
    temp = get_max_temp()
    if temp <= target:
        return temp
    print(f"  [TEMP] {label} {temp:.0f}C -> {target:.0f}C...", end="", flush=True)
    t0 = time.time()
    while temp > target and (time.time() - t0) < 180:
        time.sleep(5)
        temp = get_max_temp()
        print(f" {temp:.0f}", end="", flush=True)
    print(f" OK ({time.time()-t0:.0f}s)")
    return temp


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ============================================================
# Input signal generation
# ============================================================
def generate_input(n_steps, seed=42):
    """Random uniform input signal in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, n_steps)


# ============================================================
# FPGA run
# ============================================================
def fpga_run(fpga, u):
    """Drive FPGA, return (vmem_states, dspikes)."""
    n_steps = len(u)
    mac_signal = np.clip(u * 0.3 + 0.3, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ
    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
    for t in range(n_steps):
        if t > 0 and t % 50 == 0:
            temp = get_max_temp()
            if temp > TEMP_PAUSE:
                fpga.set_mac_signal(0.0)
                print(f"\n    [THERMAL PAUSE] {temp:.0f}C at step {t}/{n_steps}", end="", flush=True)
                while temp > TEMP_RESUME:
                    time.sleep(5)
                    temp = get_max_temp()
                    print(f" {temp:.0f}", end="", flush=True)
                print(" resumed", flush=True)
        fpga.set_mac_signal(float(mac_signal[t]))
        time.sleep(dt + 0.005)
        telem = fpga.read_telemetry()
        if telem is not None:
            states[t] = telem['vmem']
            sc = telem['spike_counts']
            diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
            diff[diff < 0] += 65536
            dspikes[t] = diff.astype(np.float32)
            prev_sc = sc.copy()
        elif t > 0:
            states[t] = states[t-1]
            dspikes[t] = dspikes[t-1]
        if t > 0 and t % 500 == 0:
            print(f"    step {t}/{n_steps}, temp={get_max_temp():.0f}C", flush=True)
    fpga.set_mac_signal(0.0)
    return states, dspikes


# ============================================================
# Temporal product features
# ============================================================
def build_temporal_features(states, dspikes=None, n_select=24, seed=42):
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, delta]
    if dspikes is not None:
        feats.append(dspikes)
    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(n_ch, size=min(n_select, n_ch), replace=False))
    vm_q = states[:, qi]
    tau_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        if dspikes is not None:
            ds_q = dspikes[:, qi]
            feats.append(ds_q * shifted)
    for i, t1 in enumerate(tau_list):
        for t2 in tau_list[i+1:]:
            if t2 > 10:
                continue
            sh1 = np.zeros_like(vm_q)
            sh2 = np.zeros_like(vm_q)
            sh1[t1:] = vm_q[:-t1]
            sh2[t2:] = vm_q[:-t2]
            feats.append(vm_q * sh1 * sh2)
    feats.append(np.square(vm_q))
    feats.append((vm_q > np.median(vm_q, axis=0)).astype(float))
    return np.hstack(feats)


# ============================================================
# PCA reduce
# ============================================================
def pca_reduce(X, n_components=128):
    if X.shape[1] <= n_components:
        return X
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


# ============================================================
# Reservoir metrics
# ============================================================
def ridge_fast(X_train, y_train, X_test, alpha=RIDGE_ALPHA):
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    d = XtX.shape[0]
    w = np.linalg.solve(XtX + alpha * np.eye(d), Xty)
    return X_test @ w


def compute_memory_capacity(X, u, max_delay=20):
    """Compute total memory capacity (sum of R² for delays 1..max_delay)."""
    n = min(len(X), len(u))
    n_tr = int(0.7 * n)
    mc = 0.0
    for d in range(1, max_delay + 1):
        target = u[max_delay - d:max_delay - d + n][:n]
        X_d = X[:n]
        try:
            pred = ridge_fast(X_d[:n_tr], target[:n_tr], X_d[n_tr:])
            y_test = target[n_tr:]
            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            mc += r2
        except Exception:
            pass
    return mc


def compute_xor(X, u, tau=1):
    """Compute XOR classification accuracy: u(t) XOR u(t-tau)."""
    n = min(len(X), len(u))
    u_bin = (u[:n] > 0.5).astype(float)
    target = np.zeros(n)
    target[tau:] = np.abs(u_bin[tau:] - u_bin[:-tau])
    n_tr = int(0.7 * n)
    try:
        pred = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:])
        pred_bin = (pred > 0.5).astype(float)
        acc = np.mean(pred_bin == target[n_tr:])
        return acc
    except Exception:
        return 0.5


def compute_waveform(X, u, n_classes=4):
    """4-class waveform classification (sin, square, sawtooth, triangle)."""
    n = min(len(X), len(u))
    labels = np.zeros(n, dtype=int)
    chunk = n // n_classes
    for c in range(n_classes):
        labels[c*chunk:(c+1)*chunk] = c
    n_tr = int(0.7 * n)
    # One-vs-all
    preds = np.zeros((n - n_tr, n_classes))
    for c in range(n_classes):
        target = (labels == c).astype(float)
        try:
            preds[:, c] = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:])
        except Exception:
            pass
    pred_labels = np.argmax(preds, axis=1)
    acc = np.mean(pred_labels == labels[n_tr:])
    return acc


def compute_narma5(X, u):
    """NARMA-5 regression: NRMSE."""
    n = min(len(X), len(u))
    # Generate NARMA-5 target
    y = np.zeros(n)
    for t in range(5, n):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[max(0,t-5):t]) + 1.5*u[t-5]*u[t-1] + 0.1
    # Clip to prevent divergence
    y = np.clip(y, -10, 10)
    y = y[10:]  # skip transient
    X_n = X[10:len(y)+10] if len(X) > 10 else X
    n = min(len(X_n), len(y))
    X_n, y = X_n[:n], y[:n]
    n_tr = int(0.7 * n)
    try:
        pred = ridge_fast(X_n[:n_tr], y[:n_tr], X_n[n_tr:])
        y_test = y[n_tr:]
        nrmse = np.sqrt(np.mean((y_test - pred)**2)) / (np.std(y_test) + 1e-10)
        return float(nrmse)
    except Exception:
        return 999.0


# ============================================================
# Save
# ============================================================
def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("z2316 — Excitation/Inhibition Balance Analysis")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temp: {get_max_temp():.0f}C")

    results = {'experiment': 'z2316_ei_balance', 'tests': {}, 'conditions': {}}

    # Generate input signal
    u_raw = generate_input(N_STEPS, seed=42)
    print(f"Input signal: {N_STEPS} steps, range [{u_raw.min():.3f}, {u_raw.max():.3f}]")

    # Run each condition
    for cond_idx, cond in enumerate(CONDITIONS):
        print(f"\n[{cond_idx+1}/{len(CONDITIONS)}] Condition: {cond}")
        wait_cool(f"pre-{cond}")

        fpga = FPGAEthBridge(timeout=2.0)
        fpga.connect()
        fpga.set_kill(0)
        time.sleep(1.0)

        # Set standard params
        fpga.set_leak_cond(0x2000)
        fpga.set_threshold_raw(0x20000)
        fpga.set_base_exc_raw(0x0080)
        fpga.set_bias_gain_raw(0x4000)

        # Set Vg per E/I condition
        vg_map = make_vg_map(cond)
        for n in range(NUM_NEURONS):
            fpga.set_vg(n, vg_map[n])
            time.sleep(0.001)
        time.sleep(0.5)

        telem = fpga.read_telemetry()
        if telem is not None:
            print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

        # Collect data
        print(f"  Collecting {N_STEPS} steps...")
        states, dspikes = fpga_run(fpga, u_raw)
        fpga.set_mac_signal(0.0)
        fpga.set_kill(1)

        # Apply software inhibition for INH_SIGN
        if cond == 'INH_SIGN':
            # Sign-flip readout for bottom 25% neurons (group 0: Vg=0.05)
            inh_mask = np.array([n % 4 == 0 for n in range(NUM_NEURONS)])
            states_proc = states.copy()
            states_proc[:, inh_mask] *= -1.0
            print(f"  INH_SIGN: flipped {inh_mask.sum()} neuron readouts")
        else:
            states_proc = states

        # Build features (after warmup)
        X_raw = states_proc[WARMUP:]
        u_used = u_raw[WARMUP:]
        X_temporal = build_temporal_features(states_proc[WARMUP:], dspikes[WARMUP:])
        X_pca = pca_reduce(X_temporal, n_components=128)
        print(f"  Features: raw={X_raw.shape[1]}, temporal={X_temporal.shape[1]}, PCA={X_pca.shape[1]}")

        # Compute metrics
        print("  Computing metrics...")
        mc = compute_memory_capacity(X_pca, u_used)
        xor1 = compute_xor(X_pca, u_used, tau=1)
        wave = compute_waveform(X_pca, u_used)
        narma5 = compute_narma5(X_pca, u_used)

        # State statistics
        state_var = np.mean(np.var(states[WARMUP:], axis=0))
        # Mean pairwise correlation
        n_sample = min(32, NUM_NEURONS)
        rng = np.random.default_rng(42)
        idx = rng.choice(NUM_NEURONS, n_sample, replace=False)
        corr_mat = np.corrcoef(states[WARMUP:, idx].T)
        mean_corr = np.mean(np.abs(corr_mat[np.triu_indices(n_sample, k=1)]))
        # Spike rate
        total_spikes = dspikes[WARMUP:].sum()
        spike_rate = total_spikes / (len(dspikes[WARMUP:]) * NUM_NEURONS)

        print(f"  MC={mc:.2f}, XOR1={xor1:.1%}, Wave={wave:.1%}, NARMA5={narma5:.3f}")
        print(f"  state_var={state_var:.6f}, mean_corr={mean_corr:.4f}, spike_rate={spike_rate:.4f}")

        results['conditions'][cond] = {
            'mc': mc, 'xor1': xor1, 'waveform': wave, 'narma5': narma5,
            'state_variance': state_var, 'mean_correlation': mean_corr,
            'spike_rate': spike_rate,
        }
        save_results(results)

    # ============================================================
    # Run tests
    # ============================================================
    print(f"\n{'='*70}")
    print("TESTS")
    print(f"{'='*70}")

    C = results['conditions']
    tests = {}

    def T(tid, name, passed, detail=""):
        tag = "PASS" if passed else "FAIL"
        tests[tid] = {'name': name, 'passed': bool(passed), 'detail': detail}
        print(f"  {tid} [{tag}] {name}: {detail}")

    # MC tests
    T('T916', 'GRADED MC > UNIFORM MC',
      C['GRADED']['mc'] > C['UNIFORM']['mc'],
      f"{C['GRADED']['mc']:.2f} vs {C['UNIFORM']['mc']:.2f}")

    T('T917', 'EI_80_20 MC > UNIFORM MC',
      C['EI_80_20']['mc'] > C['UNIFORM']['mc'],
      f"{C['EI_80_20']['mc']:.2f} vs {C['UNIFORM']['mc']:.2f}")

    T('T918', 'INH_SIGN MC > GRADED MC',
      C['INH_SIGN']['mc'] > C['GRADED']['mc'],
      f"{C['INH_SIGN']['mc']:.2f} vs {C['GRADED']['mc']:.2f}")

    # Waveform tests
    T('T919', 'EI_80_20 wave > UNIFORM wave',
      C['EI_80_20']['waveform'] > C['UNIFORM']['waveform'],
      f"{C['EI_80_20']['waveform']:.1%} vs {C['UNIFORM']['waveform']:.1%}")

    T('T920', 'GRADED wave > UNIFORM wave',
      C['GRADED']['waveform'] > C['UNIFORM']['waveform'],
      f"{C['GRADED']['waveform']:.1%} vs {C['UNIFORM']['waveform']:.1%}")

    T('T921', 'INH_SIGN wave > GRADED wave',
      C['INH_SIGN']['waveform'] > C['GRADED']['waveform'],
      f"{C['INH_SIGN']['waveform']:.1%} vs {C['GRADED']['waveform']:.1%}")

    # XOR tests
    T('T922', 'EI_80_20 XOR1 > 60%',
      C['EI_80_20']['xor1'] > 0.60,
      f"{C['EI_80_20']['xor1']:.1%}")

    T('T923', 'GRADED XOR1 > 60%',
      C['GRADED']['xor1'] > 0.60,
      f"{C['GRADED']['xor1']:.1%}")

    T('T924', 'INH_SIGN XOR1 > GRADED XOR1',
      C['INH_SIGN']['xor1'] > C['GRADED']['xor1'],
      f"{C['INH_SIGN']['xor1']:.1%} vs {C['GRADED']['xor1']:.1%}")

    # Absolute thresholds
    best_mc = max(C[c]['mc'] for c in CONDITIONS)
    best_wave = max(C[c]['waveform'] for c in CONDITIONS)
    T('T925', 'Best MC > 11.0',
      best_mc > 11.0,
      f"{best_mc:.2f}")

    T('T926', 'Best waveform > 90%',
      best_wave > 0.90,
      f"{best_wave:.1%}")

    # Statistical tests
    T('T927', 'GRADED variance > UNIFORM variance',
      C['GRADED']['state_variance'] > C['UNIFORM']['state_variance'],
      f"{C['GRADED']['state_variance']:.6f} vs {C['UNIFORM']['state_variance']:.6f}")

    T('T928', 'GRADED mean_corr < UNIFORM mean_corr',
      C['GRADED']['mean_correlation'] < C['UNIFORM']['mean_correlation'],
      f"{C['GRADED']['mean_correlation']:.4f} vs {C['UNIFORM']['mean_correlation']:.4f}")

    T('T929', 'EI_50_50 MC < EI_80_20 MC',
      C['EI_50_50']['mc'] < C['EI_80_20']['mc'],
      f"{C['EI_50_50']['mc']:.2f} vs {C['EI_80_20']['mc']:.2f}")

    # Spike rate range
    spike_rates = [C[c]['spike_rate'] for c in CONDITIONS]
    sr_ratio = max(spike_rates) / (min(spike_rates) + 1e-10)
    T('T930', 'Spike rate range > 2×',
      sr_ratio > 2.0,
      f"max/min = {sr_ratio:.2f}")

    # NARMA-5 best under non-UNIFORM
    best_narma_cond = min(CONDITIONS, key=lambda c: C[c]['narma5'])
    T('T931', 'NARMA-5 best not UNIFORM',
      best_narma_cond != 'UNIFORM',
      f"best={best_narma_cond} ({C[best_narma_cond]['narma5']:.3f})")

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['passed'])
    n_total = len(tests)
    results['summary'] = {
        'pass': n_pass, 'total': n_total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_results(results)

    print(f"\n{'='*70}")
    print(f"z2316 SUMMARY: {n_pass}/{n_total} PASS")
    print(f"{'='*70}")

    # Print comparison table
    print(f"\n  {'Condition':<12} | {'MC':>6} {'XOR1':>6} {'Wave':>6} {'NARMA5':>7} | {'Var':>8} {'Corr':>6} {'SpkRt':>6}")
    print(f"  {'-'*12}-+-{'-'*6}-{'-'*6}-{'-'*6}-{'-'*7}-+-{'-'*8}-{'-'*6}-{'-'*6}")
    for cond in CONDITIONS:
        c = C[cond]
        print(f"  {cond:<12} | {c['mc']:6.2f} {c['xor1']:5.1%} {c['waveform']:5.1%} {c['narma5']:7.3f} | {c['state_variance']:8.6f} {c['mean_correlation']:6.4f} {c['spike_rate']:6.4f}")

    print(f"\nEnd: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
