#!/usr/bin/env python3
"""
z2321_feature_ablation.py — Temporal Feature Group Ablation
============================================================
Since z2317 proved ALL performance comes from temporal product features
(not FPGA internal dynamics), this experiment quantifies which feature
groups contribute most, providing the basis for the paper's "nonlinear
transducer" narrative.

Uses a SINGLE FPGA data collection (2000 steps), then evaluates 8 feature
subsets on MC, XOR1, and waveform classification tasks.

Feature groups:
  RAW       — vmem only (128 channels)
  DELTA     — vmem + delta_vmem (256 channels)
  TAU2      — order-2 temporal products (vm(t) * vm(t-τ))
  TAU3      — order-3 temporal products (vm(t) * vm(t-τ1) * vm(t-τ2))
  DSPIKES   — delta spike counts
  DSXPROD   — dspikes × shifted vmem products
  QUAD      — quadratic features (vm²)
  THRESH    — threshold features (vm > median)
  ALL       — all features combined (baseline)
  NVAR_ONLY — delay embedding from input u only (no FPGA data)

Tests (16):
  T984: ALL MC > RAW MC (temporal features help)
  T985: TAU2 MC > RAW MC (order-2 products help)
  T986: TAU3 MC > TAU2 MC (order-3 adds value)
  T987: ALL MC > NVAR_ONLY MC (FPGA signal adds value over pure NVAR)
  T988: ALL wave > RAW wave
  T989: TAU2 wave > RAW wave
  T990: ALL XOR1 > RAW XOR1
  T991: TAU2 XOR1 > RAW XOR1
  T992: Feature importance ranking: TAU2 > TAU3 > DELTA > QUAD > THRESH
  T993: MC contribution: TAU2 contributes >30% of total MC
  T994: DSPIKES adds >0 MC over RAW (spike info useful)
  T995: Diminishing returns: ALL MC < 2 × best_single_group MC
  T996: NVAR MC > 5.0 (NVAR baseline is strong)
  T997: RAW MC < 1.0 (raw FPGA has minimal memory)
  T998: TAU2+TAU3 MC > 0.8 × ALL MC (these two dominate)
  T999: Waveform ceiling: ALL wave > 85%

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2321_feature_ablation.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2321_feature_ablation.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
RIDGE_ALPHA = 0.01
N_STEPS = 2000
WARMUP = 200
N_SELECT = 24  # channels for temporal products


# ============================================================
# Helpers
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
            preds[:, c] = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:])
        except Exception:
            pass
    return float(np.mean(np.argmax(preds, axis=1) == labels[n_tr:]))


def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


# ============================================================
# Feature group builders
# ============================================================
def build_feature_groups(states, dspikes, seed=42):
    """Build separate feature groups for ablation."""
    n_steps, n_ch = states.shape
    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(n_ch, size=min(N_SELECT, n_ch), replace=False))
    vm_q = states[:, qi]

    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])

    tau_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    groups = {}

    # RAW: just vmem
    groups['RAW'] = states.copy()

    # DELTA: vmem + delta
    groups['DELTA'] = np.hstack([states, delta])

    # TAU2: order-2 temporal products
    tau2_parts = []
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        tau2_parts.append(vm_q * shifted)
    groups['TAU2'] = np.hstack(tau2_parts) if tau2_parts else np.zeros((n_steps, 1))

    # TAU3: order-3 temporal products
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
    groups['TAU3'] = np.hstack(tau3_parts) if tau3_parts else np.zeros((n_steps, 1))

    # DSPIKES: delta spike counts
    groups['DSPIKES'] = dspikes.copy()

    # DSXPROD: dspikes × shifted vmem products
    ds_q = dspikes[:, qi]
    dsxprod_parts = []
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        dsxprod_parts.append(ds_q * shifted)
    groups['DSXPROD'] = np.hstack(dsxprod_parts) if dsxprod_parts else np.zeros((n_steps, 1))

    # QUAD: quadratic features
    groups['QUAD'] = np.square(vm_q)

    # THRESH: threshold features
    groups['THRESH'] = (vm_q > np.median(vm_q, axis=0)).astype(float)

    # ALL: everything combined
    all_parts = [states, delta, dspikes]
    all_parts.extend(tau2_parts)
    all_parts.extend(dsxprod_parts)
    all_parts.extend(tau3_parts)
    all_parts.append(np.square(vm_q))
    all_parts.append((vm_q > np.median(vm_q, axis=0)).astype(float))
    groups['ALL'] = np.hstack(all_parts)

    # TAU2+TAU3 combined
    groups['TAU2_TAU3'] = np.hstack([groups['TAU2'], groups['TAU3']])

    return groups


def build_nvar_features(u, n_steps):
    """Pure NVAR: delay embedding from input only, no FPGA data."""
    delays = list(range(1, 21))
    feats = [u.reshape(-1, 1)]
    for d in delays:
        shifted = np.zeros(n_steps)
        shifted[d:] = u[:-d]
        feats.append(shifted.reshape(-1, 1))
    u_stack = np.hstack(feats)
    # Add quadratic terms
    for i in range(min(10, u_stack.shape[1])):
        for j in range(i, min(10, u_stack.shape[1])):
            feats.append((u_stack[:, i] * u_stack[:, j]).reshape(-1, 1))
    return np.hstack(feats)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("z2321 — Temporal Feature Group Ablation")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temp: {get_max_temp():.0f}C")

    results = {'experiment': 'z2321_feature_ablation', 'tests': {}, 'exp': {}}
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 1, N_STEPS)

    # ============================================================
    # Collect FPGA data (single run)
    # ============================================================
    print(f"\n{'='*50}")
    print("Data Collection: Single FPGA run (2000 steps)")
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

    # ============================================================
    # Build feature groups
    # ============================================================
    print(f"\n{'='*50}")
    print("Building Feature Groups")
    print(f"{'='*50}")

    groups = build_feature_groups(states_w, dspikes_w)
    nvar_feats = build_nvar_features(u_w, n_w)
    groups['NVAR_ONLY'] = nvar_feats

    for name, X in groups.items():
        print(f"  {name}: {X.shape}")

    # ============================================================
    # Evaluate each group
    # ============================================================
    print(f"\n{'='*50}")
    print("Evaluating Feature Groups")
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
            'n_features': int(X.shape[1]),
        }
        print(f" MC={mc:.2f}, XOR1={xor1:.1%}, Wave={wave:.1%}")

    results['exp']['groups'] = group_results

    # MC contributions (relative to ALL)
    mc_all = group_results['ALL']['mc']
    mc_raw = group_results['RAW']['mc']
    mc_tau2 = group_results['TAU2']['mc']
    mc_tau3 = group_results['TAU3']['mc']
    mc_nvar = group_results['NVAR_ONLY']['mc']
    mc_tau2_tau3 = group_results['TAU2_TAU3']['mc']

    # Contribution = (group MC - RAW MC) / (ALL MC - RAW MC)
    mc_range = mc_all - mc_raw
    contributions = {}
    for name in group_names:
        if name in ('ALL', 'NVAR_ONLY', 'TAU2_TAU3'):
            continue
        c = (group_results[name]['mc'] - mc_raw) / (mc_range + 1e-10)
        contributions[name] = float(c)

    print(f"\n  MC contributions (fraction of gain over RAW):")
    for name, c in sorted(contributions.items(), key=lambda x: -x[1]):
        print(f"    {name}: {c:.2%}")

    results['exp']['contributions'] = contributions

    # Feature importance ranking by MC
    single_groups = ['TAU2', 'TAU3', 'DELTA', 'DSPIKES', 'DSXPROD', 'QUAD', 'THRESH']
    ranked = sorted(single_groups, key=lambda n: -group_results[n]['mc'])
    print(f"\n  MC ranking: {' > '.join(ranked)}")
    results['exp']['mc_ranking'] = ranked

    save_results(results)

    # ============================================================
    # Tests
    # ============================================================
    print(f"\n{'='*70}")
    print("TESTS")
    print(f"{'='*70}")

    tests = {}

    def T(tid, name, passed, detail=""):
        tag = "PASS" if passed else "FAIL"
        tests[tid] = {'name': name, 'passed': bool(passed), 'detail': detail}
        print(f"  {tid} [{tag}] {name}: {detail}")

    T('T984', 'ALL MC > RAW MC',
      mc_all > mc_raw,
      f'ALL={mc_all:.2f} vs RAW={mc_raw:.2f}')

    T('T985', 'TAU2 MC > RAW MC',
      mc_tau2 > mc_raw,
      f'TAU2={mc_tau2:.2f} vs RAW={mc_raw:.2f}')

    T('T986', 'TAU3 MC > TAU2 MC',
      mc_tau3 > mc_tau2,
      f'TAU3={mc_tau3:.2f} vs TAU2={mc_tau2:.2f}')

    T('T987', 'ALL MC > NVAR_ONLY MC',
      mc_all > mc_nvar,
      f'ALL={mc_all:.2f} vs NVAR={mc_nvar:.2f}')

    wave_all = group_results['ALL']['wave']
    wave_raw = group_results['RAW']['wave']
    wave_tau2 = group_results['TAU2']['wave']
    xor_all = group_results['ALL']['xor1']
    xor_raw = group_results['RAW']['xor1']
    xor_tau2 = group_results['TAU2']['xor1']

    T('T988', 'ALL wave > RAW wave',
      wave_all > wave_raw,
      f'ALL={wave_all:.1%} vs RAW={wave_raw:.1%}')

    T('T989', 'TAU2 wave > RAW wave',
      wave_tau2 > wave_raw,
      f'TAU2={wave_tau2:.1%} vs RAW={wave_raw:.1%}')

    T('T990', 'ALL XOR1 > RAW XOR1',
      xor_all > xor_raw,
      f'ALL={xor_all:.1%} vs RAW={xor_raw:.1%}')

    T('T991', 'TAU2 XOR1 > RAW XOR1',
      xor_tau2 > xor_raw,
      f'TAU2={xor_tau2:.1%} vs RAW={xor_raw:.1%}')

    # Check ranking: TAU2 > TAU3 > DELTA > QUAD > THRESH
    expected_order = ['TAU2', 'TAU3', 'DELTA', 'QUAD', 'THRESH']
    actual_subset = [g for g in ranked if g in expected_order]
    order_correct = actual_subset == expected_order
    T('T992', 'Ranking: TAU2 > TAU3 > DELTA > QUAD > THRESH',
      order_correct,
      f'actual: {" > ".join(actual_subset)}')

    tau2_contrib = contributions.get('TAU2', 0)
    T('T993', 'TAU2 contributes > 30% of MC gain',
      tau2_contrib > 0.30,
      f'{tau2_contrib:.1%}')

    mc_dspikes = group_results['DSPIKES']['mc']
    T('T994', 'DSPIKES MC > RAW MC',
      mc_dspikes > mc_raw,
      f'DSPIKES={mc_dspikes:.2f} vs RAW={mc_raw:.2f}')

    # Best single group
    best_single_mc = max(group_results[g]['mc'] for g in single_groups)
    T('T995', 'Diminishing returns: ALL MC < 2 × best_single',
      mc_all < 2 * best_single_mc or mc_all == 0,
      f'ALL={mc_all:.2f} vs 2×best={2*best_single_mc:.2f}')

    T('T996', 'NVAR MC > 5.0',
      mc_nvar > 5.0,
      f'{mc_nvar:.2f}')

    T('T997', 'RAW MC < 1.0',
      mc_raw < 1.0,
      f'{mc_raw:.2f}')

    T('T998', 'TAU2+TAU3 MC > 0.8 × ALL MC',
      mc_tau2_tau3 > 0.8 * mc_all or mc_all < 0.1,
      f'TAU2+TAU3={mc_tau2_tau3:.2f} vs 0.8×ALL={0.8*mc_all:.2f}')

    T('T999', 'Waveform ceiling: ALL wave > 85%',
      wave_all > 0.85,
      f'{wave_all:.1%}')

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['passed'])
    n_total = len(tests)
    results['summary'] = {'pass': n_pass, 'total': n_total,
                          'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    save_results(results)

    print(f"\n{'='*70}")
    print(f"z2321 SUMMARY: {n_pass}/{n_total} PASS")
    print(f"{'='*70}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
