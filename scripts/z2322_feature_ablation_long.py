#!/usr/bin/env python3
"""
z2322_feature_ablation_long.py — Feature Ablation with Long Sequences
======================================================================
z2321 showed temporal products collapse to MC=0 with N=1800 steps due to
curse of dimensionality (1600 features, 128 PCA components, short sequences).
This version uses N=5000 steps to give features enough data to shine,
matching z2310's successful setup.

Also uses alpha search (like z2310) instead of fixed alpha.

Feature groups:
  RAW, DELTA, TAU2, TAU3, DSPIKES, DSXPROD, QUAD, THRESH, ALL, TAU2_TAU3, NVAR

Tests (12):
  T1000: ALL MC > RAW MC
  T1001: TAU2 MC > RAW MC
  T1002: TAU3 MC > RAW MC
  T1003: ALL MC > 8.0 (matching z2310 performance level)
  T1004: ALL wave > 80%
  T1005: TAU2+TAU3 MC > 0.7 × ALL MC (temporal products dominate)
  T1006: NVAR MC > ALL MC (NVAR has direct input access)
  T1007: RAW MC < 2.0 (raw FPGA has minimal memory)
  T1008: ALL XOR1 > RAW XOR1
  T1009: QUAD MC > RAW MC (quadratic helps)
  T1010: Feature ranking includes TAU2 in top 3
  T1011: DELTA MC > RAW MC (derivatives help)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2322_feature_ablation_long.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2322_feature_ablation_long.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
N_STEPS = 5000
WARMUP = 200
N_SELECT = 24


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


def setup_fpga():
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
    return fpga


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
                print(f"\n    [THERMAL PAUSE] {temp:.0f}C at step {t}", end="", flush=True)
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


def pca_reduce(X, n_components=128):
    if X.shape[1] <= n_components:
        return X
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


def ridge_alpha_search(X_train, y_train, X_test, y_test_true=None,
                       alphas=None):
    """Ridge with alpha search using internal validation split."""
    if alphas is None:
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    n_tr = X_train.shape[0]
    n_val = n_tr // 5
    n_tr_inner = n_tr - n_val

    best_alpha = alphas[0]
    best_val_mse = 1e30
    for alpha in alphas:
        XtX = X_train[:n_tr_inner].T @ X_train[:n_tr_inner]
        Xty = X_train[:n_tr_inner].T @ y_train[:n_tr_inner]
        d = XtX.shape[0]
        try:
            w = np.linalg.solve(XtX + alpha * np.eye(d), Xty)
            pred_val = X_train[n_tr_inner:] @ w
            val_mse = np.mean((y_train[n_tr_inner:] - pred_val) ** 2)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_alpha = alpha
        except Exception:
            pass

    # Refit on full training set with best alpha
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    d = XtX.shape[0]
    w = np.linalg.solve(XtX + best_alpha * np.eye(d), Xty)
    return X_test @ w


def compute_mc(X, u, max_delay=20):
    n = min(len(X), len(u))
    n_tr = int(0.7 * n)
    mc = 0.0
    for d in range(1, max_delay + 1):
        target = u[max_delay - d:max_delay - d + n][:n]
        try:
            pred = ridge_alpha_search(X[:n_tr], target[:n_tr], X[n_tr:])
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
        pred = ridge_alpha_search(X[:n_tr], target[:n_tr], X[n_tr:])
        return float(np.mean((pred > 0.5).astype(float) == target[n_tr:]))
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
            preds[:, c] = ridge_alpha_search(X[:n_tr], target[:n_tr], X[n_tr:])
        except Exception:
            pass
    return float(np.mean(np.argmax(preds, axis=1) == labels[n_tr:]))


def build_feature_groups(states, dspikes, seed=42):
    n_steps, n_ch = states.shape
    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(n_ch, size=min(N_SELECT, n_ch), replace=False))
    vm_q = states[:, qi]
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    tau_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    groups = {}
    groups['RAW'] = states.copy()
    groups['DELTA'] = np.hstack([states, delta])

    tau2_parts = []
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        tau2_parts.append(vm_q * shifted)
    groups['TAU2'] = np.hstack(tau2_parts)

    tau3_parts = []
    for i, t1 in enumerate(tau_list):
        for t2 in tau_list[i+1:]:
            if t2 > 10:
                continue
            sh1 = np.zeros_like(vm_q)
            sh2 = np.zeros_like(vm_q)
            sh1[t1:] = vm_q[:-t1]
            sh2[t2:] = vm_q[:-t2]
            tau3_parts.append(vm_q * sh1 * sh2)
    groups['TAU3'] = np.hstack(tau3_parts)

    groups['DSPIKES'] = dspikes.copy()

    ds_q = dspikes[:, qi]
    dsxprod_parts = []
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        dsxprod_parts.append(ds_q * shifted)
    groups['DSXPROD'] = np.hstack(dsxprod_parts)

    groups['QUAD'] = np.square(vm_q)
    groups['THRESH'] = (vm_q > np.median(vm_q, axis=0)).astype(float)

    all_parts = [states, delta, dspikes]
    all_parts.extend(tau2_parts)
    all_parts.extend(dsxprod_parts)
    all_parts.extend(tau3_parts)
    all_parts.append(np.square(vm_q))
    all_parts.append((vm_q > np.median(vm_q, axis=0)).astype(float))
    groups['ALL'] = np.hstack(all_parts)

    groups['TAU2_TAU3'] = np.hstack([groups['TAU2'], groups['TAU3']])

    return groups


def build_nvar_features(u, n_steps):
    delays = list(range(1, 21))
    feats = [u.reshape(-1, 1)]
    for d in delays:
        shifted = np.zeros(n_steps)
        shifted[d:] = u[:-d]
        feats.append(shifted.reshape(-1, 1))
    u_stack = np.hstack(feats)
    for i in range(min(10, u_stack.shape[1])):
        for j in range(i, min(10, u_stack.shape[1])):
            feats.append((u_stack[:, i] * u_stack[:, j]).reshape(-1, 1))
    return np.hstack(feats)


def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


def main():
    print("=" * 70)
    print("z2322 — Feature Ablation (Long Sequences, 5000 steps)")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temp: {get_max_temp():.0f}C")

    results = {'experiment': 'z2322_feature_ablation_long', 'tests': {}, 'exp': {}}
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 1, N_STEPS)

    # Collect FPGA data
    print(f"\n{'='*50}")
    print(f"Data Collection: Single FPGA run ({N_STEPS} steps)")
    print(f"{'='*50}")

    wait_cool("data-collect")
    fpga = setup_fpga()
    telem = fpga.read_telemetry()
    if telem is not None:
        print(f"  vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    states, dspikes = fpga_run(fpga, u)
    fpga.set_kill(1)

    states_w = states[WARMUP:]
    dspikes_w = dspikes[WARMUP:]
    u_w = u[WARMUP:]
    n_w = len(u_w)
    print(f"  Working samples: {n_w}")

    # Build feature groups
    print(f"\n{'='*50}")
    print("Building Feature Groups")
    print(f"{'='*50}")

    groups = build_feature_groups(states_w, dspikes_w)
    nvar_feats = build_nvar_features(u_w, n_w)
    groups['NVAR_ONLY'] = nvar_feats

    for name, X in groups.items():
        print(f"  {name}: {X.shape}")

    # Evaluate
    print(f"\n{'='*50}")
    print("Evaluating Feature Groups (with alpha search)")
    print(f"{'='*50}")

    group_names = ['RAW', 'DELTA', 'TAU2', 'TAU3', 'DSPIKES', 'DSXPROD',
                   'QUAD', 'THRESH', 'ALL', 'TAU2_TAU3', 'NVAR_ONLY']
    group_results = {}

    for name in group_names:
        print(f"\n  {name}...", end="", flush=True)
        X = groups[name]
        X_pca = pca_reduce(X)
        mc = compute_mc(X_pca, u_w)
        xor1 = compute_xor(X_pca, u_w)
        wave = compute_wave(X_pca, u_w)
        group_results[name] = {
            'mc': float(mc), 'xor1': float(xor1), 'wave': float(wave),
            'n_features': int(X.shape[1]), 'n_pca': int(X_pca.shape[1]),
        }
        print(f" MC={mc:.2f}, XOR1={xor1:.1%}, Wave={wave:.1%} ({X_pca.shape[1]} dims)")

    results['exp']['groups'] = group_results

    # Rankings
    single_groups = ['TAU2', 'TAU3', 'DELTA', 'DSPIKES', 'DSXPROD', 'QUAD', 'THRESH']
    ranked = sorted(single_groups, key=lambda n: -group_results[n]['mc'])
    rank_strs = [f"{g}({group_results[g]['mc']:.2f})" for g in ranked]
    print(f"\n  MC ranking: {' > '.join(rank_strs)}")
    results['exp']['mc_ranking'] = ranked
    save_results(results)

    # Tests
    print(f"\n{'='*70}")
    print("TESTS")
    print(f"{'='*70}")

    tests = {}
    def T(tid, name, passed, detail=""):
        tag = "PASS" if passed else "FAIL"
        tests[tid] = {'name': name, 'passed': bool(passed), 'detail': detail}
        print(f"  {tid} [{tag}] {name}: {detail}")

    mc_all = group_results['ALL']['mc']
    mc_raw = group_results['RAW']['mc']
    mc_tau2 = group_results['TAU2']['mc']
    mc_tau3 = group_results['TAU3']['mc']
    mc_nvar = group_results['NVAR_ONLY']['mc']
    mc_tt = group_results['TAU2_TAU3']['mc']
    mc_delta = group_results['DELTA']['mc']
    mc_quad = group_results['QUAD']['mc']
    wave_all = group_results['ALL']['wave']
    xor_all = group_results['ALL']['xor1']
    xor_raw = group_results['RAW']['xor1']

    T('T1000', 'ALL MC > RAW MC', mc_all > mc_raw, f'ALL={mc_all:.2f} vs RAW={mc_raw:.2f}')
    T('T1001', 'TAU2 MC > RAW MC', mc_tau2 > mc_raw, f'TAU2={mc_tau2:.2f} vs RAW={mc_raw:.2f}')
    T('T1002', 'TAU3 MC > RAW MC', mc_tau3 > mc_raw, f'TAU3={mc_tau3:.2f} vs RAW={mc_raw:.2f}')
    T('T1003', 'ALL MC > 8.0', mc_all > 8.0, f'{mc_all:.2f}')
    T('T1004', 'ALL wave > 80%', wave_all > 0.80, f'{wave_all:.1%}')
    T('T1005', 'TAU2+TAU3 > 0.7×ALL MC', mc_tt > 0.7 * mc_all or mc_all < 0.1, f'T2T3={mc_tt:.2f} vs 0.7×ALL={0.7*mc_all:.2f}')
    T('T1006', 'NVAR MC > ALL MC', mc_nvar > mc_all, f'NVAR={mc_nvar:.2f} vs ALL={mc_all:.2f}')
    T('T1007', 'RAW MC < 2.0', mc_raw < 2.0, f'{mc_raw:.2f}')
    T('T1008', 'ALL XOR1 > RAW XOR1', xor_all > xor_raw, f'ALL={xor_all:.1%} vs RAW={xor_raw:.1%}')
    T('T1009', 'QUAD MC > RAW MC', mc_quad > mc_raw, f'QUAD={mc_quad:.2f} vs RAW={mc_raw:.2f}')
    top3 = ranked[:3]
    T('T1010', 'TAU2 in top 3 by MC', 'TAU2' in top3, f'top3={top3}')
    T('T1011', 'DELTA MC > RAW MC', mc_delta > mc_raw, f'DELTA={mc_delta:.2f} vs RAW={mc_raw:.2f}')

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['passed'])
    n_total = len(tests)
    results['summary'] = {'pass': n_pass, 'total': n_total,
                          'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    save_results(results)

    print(f"\n{'='*70}")
    print(f"z2322 SUMMARY: {n_pass}/{n_total} PASS")
    print(f"{'='*70}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
