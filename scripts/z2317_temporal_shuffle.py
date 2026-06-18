#!/usr/bin/env python3
"""
z2317_temporal_shuffle.py — Temporal Shuffle Control
=====================================================
Critical control experiment: does the ORDER of FPGA readouts matter,
or is performance entirely from feature dimensionality?

If temporal products work just as well on time-shuffled states,
then the FPGA provides NO temporal processing — just a nonlinear
feature expander.

Conditions (4):
  1) ORDERED:    Normal temporal order (baseline)
  2) SHUFFLED:   Time-shuffled states (destroys temporal structure)
  3) REVERSED:   Time-reversed states (preserves statistics, destroys causality)
  4) BLOCK_SHUF: Shuffle within 50-step blocks (preserves local but not global)

Tests (12):
  T932: ORDERED MC > SHUFFLED MC
  T933: ORDERED MC > REVERSED MC
  T934: ORDERED XOR > SHUFFLED XOR
  T935: ORDERED XOR > REVERSED XOR
  T936: ORDERED waveform > SHUFFLED waveform
  T937: BLOCK_SHUF MC > SHUFFLED MC (local structure matters)
  T938: ORDERED MC > BLOCK_SHUF MC (global structure matters)
  T939: SHUFFLED MC < 1.0 (shuffling destroys memory)
  T940: Temporal feature rank: ORDERED > SHUFFLED
  T941: NARMA-5: ORDERED NRMSE < SHUFFLED NRMSE
  T942: ORDERED MC > 10.0 (absolute threshold)
  T943: Temporal product contribution: ORDERED temporal/raw MC ratio > 5

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2317_temporal_shuffle.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2317_temporal_shuffle.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
N_STEPS = 3000
WARMUP = 200
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
RIDGE_ALPHA = 0.01


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
# FPGA run
# ============================================================
def fpga_run(fpga, u):
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


def effective_rank(X):
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-15))))


# ============================================================
# Metrics
# ============================================================
def ridge_fast(X_train, y_train, X_test, alpha=RIDGE_ALPHA):
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    d = XtX.shape[0]
    w = np.linalg.solve(XtX + alpha * np.eye(d), Xty)
    return X_test @ w


def compute_mc(X, u, max_delay=20):
    n = min(len(X), len(u))
    n_tr = int(0.7 * n)
    mc = 0.0
    for d in range(1, max_delay + 1):
        target = u[max_delay - d:max_delay - d + n][:n]
        try:
            pred = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:])
            y_test = target[n_tr:]
            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            mc += r2
        except Exception:
            pass
    return mc


def compute_xor(X, u, tau=1):
    n = min(len(X), len(u))
    u_bin = (u[:n] > 0.5).astype(float)
    target = np.zeros(n)
    target[tau:] = np.abs(u_bin[tau:] - u_bin[:-tau])
    n_tr = int(0.7 * n)
    try:
        pred = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:])
        pred_bin = (pred > 0.5).astype(float)
        return float(np.mean(pred_bin == target[n_tr:]))
    except Exception:
        return 0.5


def compute_wave(X, u, n_classes=4):
    n = min(len(X), len(u))
    labels = np.zeros(n, dtype=int)
    chunk = n // n_classes
    for c in range(n_classes):
        labels[c*chunk:(c+1)*chunk] = c
    n_tr = int(0.7 * n)
    preds = np.zeros((n - n_tr, n_classes))
    for c in range(n_classes):
        target = (labels == c).astype(float)
        try:
            preds[:, c] = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:])
        except Exception:
            pass
    pred_labels = np.argmax(preds, axis=1)
    return float(np.mean(pred_labels == labels[n_tr:]))


def compute_narma5(X, u):
    n = min(len(X), len(u))
    y = np.zeros(n)
    for t in range(5, n):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[max(0,t-5):t]) + 1.5*u[t-5]*u[t-1] + 0.1
    y = np.clip(y, -10, 10)
    y = y[10:]
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
    print("z2317 — Temporal Shuffle Control")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temp: {get_max_temp():.0f}C")

    results = {'experiment': 'z2317_temporal_shuffle', 'tests': {}, 'conditions': {}}

    # Generate input
    rng = np.random.default_rng(42)
    u_raw = rng.uniform(0, 1, N_STEPS)

    # ============================================================
    # Collect FPGA data ONCE (temporal order preserved)
    # ============================================================
    print(f"\n[1] Collecting FPGA data ({N_STEPS} steps)...")
    wait_cool("pre-FPGA")

    fpga = FPGAEthBridge(timeout=2.0)
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(1.0)
    fpga.set_leak_cond(0x2000)
    fpga.set_threshold_raw(0x20000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    time.sleep(0.5)

    telem = fpga.read_telemetry()
    if telem is not None:
        print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

    states, dspikes = fpga_run(fpga, u_raw)
    fpga.set_mac_signal(0.0)
    fpga.set_kill(1)

    # After warmup
    states_w = states[WARMUP:]
    dspikes_w = dspikes[WARMUP:]
    u_w = u_raw[WARMUP:]
    n_w = len(states_w)

    # ============================================================
    # Generate 4 temporal conditions from SAME FPGA data
    # ============================================================
    conditions = {}

    # 1) ORDERED (baseline)
    print("\n[2] Building condition features...")
    conditions['ORDERED'] = {
        'states': states_w,
        'dspikes': dspikes_w,
        'u': u_w,
    }

    # 2) SHUFFLED (random permutation of time axis)
    perm = rng.permutation(n_w)
    conditions['SHUFFLED'] = {
        'states': states_w[perm],
        'dspikes': dspikes_w[perm],
        'u': u_w[perm],
    }

    # 3) REVERSED (time-reversed)
    conditions['REVERSED'] = {
        'states': states_w[::-1].copy(),
        'dspikes': dspikes_w[::-1].copy(),
        'u': u_w[::-1].copy(),
    }

    # 4) BLOCK_SHUF (shuffle within 50-step blocks)
    block_size = 50
    n_blocks = n_w // block_size
    block_states = states_w[:n_blocks*block_size].reshape(n_blocks, block_size, -1)
    block_dspikes = dspikes_w[:n_blocks*block_size].reshape(n_blocks, block_size, -1)
    block_u = u_w[:n_blocks*block_size].reshape(n_blocks, block_size)
    # Shuffle blocks but keep within-block order
    block_perm = rng.permutation(n_blocks)
    block_states = block_states[block_perm].reshape(-1, NUM_NEURONS)
    block_dspikes = block_dspikes[block_perm].reshape(-1, NUM_NEURONS)
    block_u = block_u[block_perm].reshape(-1)
    conditions['BLOCK_SHUF'] = {
        'states': block_states,
        'dspikes': block_dspikes,
        'u': block_u,
    }

    # ============================================================
    # Evaluate each condition
    # ============================================================
    for cond_name, cond_data in conditions.items():
        print(f"\n[{cond_name}] Evaluating...")
        wait_cool(f"pre-{cond_name}")

        s = cond_data['states']
        ds = cond_data['dspikes']
        u = cond_data['u']

        # Raw features
        X_raw = s
        mc_raw = compute_mc(X_raw, u)

        # Temporal features
        X_temporal = build_temporal_features(s, ds)
        X_pca = pca_reduce(X_temporal, n_components=128)

        mc = compute_mc(X_pca, u)
        xor1 = compute_xor(X_pca, u, tau=1)
        wave = compute_wave(X_pca, u)
        narma5 = compute_narma5(X_pca, u)

        # Effective rank
        eff_rank = effective_rank(X_pca[:500])  # subsample for speed

        print(f"  MC_raw={mc_raw:.2f}, MC_temporal={mc:.2f}")
        print(f"  XOR1={xor1:.1%}, Wave={wave:.1%}, NARMA5={narma5:.3f}")
        print(f"  Effective rank: {eff_rank:.2f}")

        results['conditions'][cond_name] = {
            'mc_raw': mc_raw, 'mc': mc, 'xor1': xor1,
            'waveform': wave, 'narma5': narma5,
            'effective_rank': eff_rank,
            'temporal_ratio': mc / (mc_raw + 1e-10),
        }
        save_results(results)

    # ============================================================
    # Tests
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

    T('T932', 'ORDERED MC > SHUFFLED MC',
      C['ORDERED']['mc'] > C['SHUFFLED']['mc'],
      f"{C['ORDERED']['mc']:.2f} vs {C['SHUFFLED']['mc']:.2f}")

    T('T933', 'ORDERED MC > REVERSED MC',
      C['ORDERED']['mc'] > C['REVERSED']['mc'],
      f"{C['ORDERED']['mc']:.2f} vs {C['REVERSED']['mc']:.2f}")

    T('T934', 'ORDERED XOR > SHUFFLED XOR',
      C['ORDERED']['xor1'] > C['SHUFFLED']['xor1'],
      f"{C['ORDERED']['xor1']:.1%} vs {C['SHUFFLED']['xor1']:.1%}")

    T('T935', 'ORDERED XOR > REVERSED XOR',
      C['ORDERED']['xor1'] > C['REVERSED']['xor1'],
      f"{C['ORDERED']['xor1']:.1%} vs {C['REVERSED']['xor1']:.1%}")

    T('T936', 'ORDERED wave > SHUFFLED wave',
      C['ORDERED']['waveform'] > C['SHUFFLED']['waveform'],
      f"{C['ORDERED']['waveform']:.1%} vs {C['SHUFFLED']['waveform']:.1%}")

    T('T937', 'BLOCK_SHUF MC > SHUFFLED MC',
      C['BLOCK_SHUF']['mc'] > C['SHUFFLED']['mc'],
      f"{C['BLOCK_SHUF']['mc']:.2f} vs {C['SHUFFLED']['mc']:.2f}")

    T('T938', 'ORDERED MC > BLOCK_SHUF MC',
      C['ORDERED']['mc'] > C['BLOCK_SHUF']['mc'],
      f"{C['ORDERED']['mc']:.2f} vs {C['BLOCK_SHUF']['mc']:.2f}")

    T('T939', 'SHUFFLED MC < 1.0',
      C['SHUFFLED']['mc'] < 1.0,
      f"{C['SHUFFLED']['mc']:.2f}")

    T('T940', 'Temporal rank: ORDERED > SHUFFLED',
      C['ORDERED']['effective_rank'] > C['SHUFFLED']['effective_rank'],
      f"{C['ORDERED']['effective_rank']:.2f} vs {C['SHUFFLED']['effective_rank']:.2f}")

    T('T941', 'NARMA-5: ORDERED < SHUFFLED NRMSE',
      C['ORDERED']['narma5'] < C['SHUFFLED']['narma5'],
      f"{C['ORDERED']['narma5']:.3f} vs {C['SHUFFLED']['narma5']:.3f}")

    T('T942', 'ORDERED MC > 10.0',
      C['ORDERED']['mc'] > 10.0,
      f"{C['ORDERED']['mc']:.2f}")

    T('T943', 'Temporal/raw MC ratio > 5',
      C['ORDERED']['temporal_ratio'] > 5,
      f"{C['ORDERED']['temporal_ratio']:.2f}")

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['passed'])
    n_total = len(tests)
    results['summary'] = {
        'pass': n_pass, 'total': n_total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_results(results)

    print(f"\n{'='*70}")
    print(f"z2317 SUMMARY: {n_pass}/{n_total} PASS")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<12} | {'MC_raw':>7} {'MC':>7} {'XOR1':>6} {'Wave':>6} {'NARMA5':>7} | {'Rank':>5} {'T/R':>5}")
    print(f"  {'-'*12}-+-{'-'*7}-{'-'*7}-{'-'*6}-{'-'*6}-{'-'*7}-+-{'-'*5}-{'-'*5}")
    for cond in ['ORDERED', 'SHUFFLED', 'REVERSED', 'BLOCK_SHUF']:
        c = C[cond]
        print(f"  {cond:<12} | {c['mc_raw']:7.2f} {c['mc']:7.2f} {c['xor1']:5.1%} {c['waveform']:5.1%} {c['narma5']:7.3f} | {c['effective_rank']:5.2f} {c['temporal_ratio']:5.1f}")

    print(f"\nEnd: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
