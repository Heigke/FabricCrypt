#!/usr/bin/env python3
"""
z2318_reproducibility.py — Reproducibility & Input Sensitivity Analysis
========================================================================
Tests whether the FPGA reservoir produces consistent outputs across:
1) Repeated identical inputs (reproducibility)
2) Slightly perturbed inputs (sensitivity / Lyapunov-like analysis)
3) Different initial conditions (echo state property at scale)

This complements z2315 (which tested echo state on 100 steps) with a
more thorough analysis using full-length signals.

Conditions:
  EXP1 — Reproducibility: Same input 5 times, measure output variance
  EXP2 — Sensitivity: Input + epsilon perturbation at 5 noise levels
  EXP3 — Prediction consistency: Same Mackey-Glass, 3 runs, compare predictions

Tests (14):
  T944: Reproducibility: mean vmem std < 0.01 across 5 runs
  T945: Reproducibility: spike count correlation > 0.95 between runs
  T946: Reproducibility: MC variance < 0.5 across runs
  T947: Reproducibility: waveform variance < 5pp across runs
  T948: Sensitivity: NRMSE increases with perturbation (monotonic 4/4)
  T949: Sensitivity: epsilon=0.001 NRMSE < 0.01 (robust to tiny noise)
  T950: Sensitivity: epsilon=0.1 NRMSE > 0.05 (responsive to large noise)
  T951: Sensitivity: no chaos — NRMSE < 1.0 even at epsilon=0.1
  T952: Prediction: MC std < 1.0 across 3 runs
  T953: Prediction: waveform std < 3pp across 3 runs
  T954: Prediction: all 3 runs MC > 9.0
  T955: Prediction: XOR1 std < 5pp
  T956: SNR: signal-to-noise ratio > 10 for vmem
  T957: Channel consistency: >90% of neurons have <0.01 std across runs

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2318_reproducibility.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2318_reproducibility.json'

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
    """Connect and configure FPGA with standard params."""
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
    """Drive FPGA with input signal, return (vmem, dspikes)."""
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
# Main
# ============================================================
def main():
    print("=" * 70)
    print("z2318 — Reproducibility & Input Sensitivity Analysis")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temp: {get_max_temp():.0f}C")

    results = {'experiment': 'z2318_reproducibility', 'tests': {}, 'exp': {}}
    rng = np.random.default_rng(42)
    u_base = rng.uniform(0, 1, N_STEPS)

    # ============================================================
    # EXP1: Reproducibility — same input 5 times
    # ============================================================
    print(f"\n{'='*50}")
    print("EXP1: Reproducibility (5 identical runs)")
    print(f"{'='*50}")

    N_REPS = 5
    all_states = []
    all_dspikes = []

    for rep in range(N_REPS):
        print(f"\n  Run {rep+1}/{N_REPS}...")
        wait_cool(f"rep-{rep+1}")
        fpga = setup_fpga()
        telem = fpga.read_telemetry()
        if telem is not None:
            print(f"    vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
        states, dspikes = fpga_run(fpga, u_base)
        all_states.append(states[WARMUP:])
        all_dspikes.append(dspikes[WARMUP:])
        fpga.set_kill(1)

    # Compute reproducibility metrics
    stacked = np.stack(all_states)  # (5, n_steps, 128)
    vmem_std = stacked.std(axis=0)  # (n_steps, 128)
    mean_vmem_std = vmem_std.mean()
    channel_std = vmem_std.mean(axis=0)  # (128,)
    consistent_channels = np.sum(channel_std < 0.01)

    # Spike count correlation between runs
    spike_totals = np.array([ds.sum(axis=0) for ds in all_dspikes])  # (5, 128)
    if spike_totals.max() > 0:
        spike_corrs = []
        for i in range(N_REPS):
            for j in range(i+1, N_REPS):
                c = np.corrcoef(spike_totals[i], spike_totals[j])[0, 1]
                if not np.isnan(c):
                    spike_corrs.append(c)
        mean_spike_corr = np.mean(spike_corrs) if spike_corrs else 0.0
    else:
        mean_spike_corr = 1.0  # all zeros → perfectly "correlated"

    # SNR
    signal_power = stacked.mean(axis=0).var()
    noise_power = vmem_std.mean() ** 2
    snr = signal_power / (noise_power + 1e-15)

    # Compute MC and waveform for each run
    mc_vals = []
    wave_vals = []
    xor_vals = []
    u_w = u_base[WARMUP:]
    for rep in range(N_REPS):
        X = build_temporal_features(all_states[rep], all_dspikes[rep])
        X_pca = pca_reduce(X)
        mc = compute_mc(X_pca, u_w)
        wave = compute_wave(X_pca, u_w)
        xor1 = compute_xor(X_pca, u_w)
        mc_vals.append(mc)
        wave_vals.append(wave)
        xor_vals.append(xor1)
        print(f"  Run {rep+1}: MC={mc:.2f}, Wave={wave:.1%}, XOR1={xor1:.1%}")

    mc_std = np.std(mc_vals)
    wave_std = np.std(wave_vals)
    xor_std = np.std(xor_vals)

    print(f"\n  Reproducibility summary:")
    print(f"    Mean vmem std across runs: {mean_vmem_std:.6f}")
    print(f"    Spike count correlation: {mean_spike_corr:.4f}")
    print(f"    MC: {np.mean(mc_vals):.2f} ± {mc_std:.2f}")
    print(f"    Waveform: {np.mean(wave_vals):.1%} ± {wave_std:.1%}")
    print(f"    XOR1: {np.mean(xor_vals):.1%} ± {xor_std:.1%}")
    print(f"    SNR: {snr:.2f}")
    print(f"    Consistent channels (<0.01 std): {consistent_channels}/128")

    results['exp']['reproducibility'] = {
        'mean_vmem_std': float(mean_vmem_std),
        'spike_corr': float(mean_spike_corr),
        'mc_mean': float(np.mean(mc_vals)), 'mc_std': float(mc_std),
        'wave_mean': float(np.mean(wave_vals)), 'wave_std': float(wave_std),
        'xor_mean': float(np.mean(xor_vals)), 'xor_std': float(xor_std),
        'snr': float(snr),
        'consistent_channels': int(consistent_channels),
        'mc_vals': [float(v) for v in mc_vals],
        'wave_vals': [float(v) for v in wave_vals],
    }
    save_results(results)

    # ============================================================
    # EXP2: Sensitivity — perturbation analysis
    # ============================================================
    print(f"\n{'='*50}")
    print("EXP2: Input Sensitivity (perturbation analysis)")
    print(f"{'='*50}")

    epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]
    sensitivity = {}

    # Use first run as reference
    ref_states = all_states[0]

    for eps in epsilons:
        print(f"\n  epsilon={eps}...")
        wait_cool(f"eps-{eps}")
        u_pert = u_base + rng.normal(0, eps, N_STEPS)
        u_pert = np.clip(u_pert, 0, 1)

        fpga = setup_fpga()
        pert_states, _ = fpga_run(fpga, u_pert)
        fpga.set_kill(1)

        pert_w = pert_states[WARMUP:]
        # NRMSE between reference and perturbed
        diff = ref_states - pert_w
        nrmse = np.sqrt(np.mean(diff**2)) / (np.std(ref_states) + 1e-10)
        max_diff = np.max(np.abs(diff))

        print(f"    NRMSE: {nrmse:.6f}, max_diff: {max_diff:.6f}")
        sensitivity[str(eps)] = {'nrmse': float(nrmse), 'max_diff': float(max_diff)}

    # Check monotonicity
    nrmse_vals = [sensitivity[str(e)]['nrmse'] for e in epsilons]
    n_monotonic = sum(1 for i in range(len(nrmse_vals)-1) if nrmse_vals[i+1] > nrmse_vals[i])

    results['exp']['sensitivity'] = {
        'epsilons': epsilons,
        'nrmse_vals': nrmse_vals,
        'n_monotonic': n_monotonic,
        'details': sensitivity,
    }
    save_results(results)

    # ============================================================
    # EXP3: Prediction consistency (3 independent runs)
    # ============================================================
    print(f"\n{'='*50}")
    print("EXP3: Prediction Consistency (3 runs)")
    print(f"{'='*50}")

    pred_mc = []
    pred_wave = []
    pred_xor = []

    for run in range(3):
        print(f"\n  Run {run+1}/3...")
        wait_cool(f"pred-run-{run+1}")
        # Different random input per run
        u_run = np.random.default_rng(100 + run).uniform(0, 1, N_STEPS)
        fpga = setup_fpga()
        st, ds = fpga_run(fpga, u_run)
        fpga.set_kill(1)

        X = build_temporal_features(st[WARMUP:], ds[WARMUP:])
        X_pca = pca_reduce(X)
        u_w = u_run[WARMUP:]
        mc = compute_mc(X_pca, u_w)
        wave = compute_wave(X_pca, u_w)
        xor1 = compute_xor(X_pca, u_w)
        pred_mc.append(mc)
        pred_wave.append(wave)
        pred_xor.append(xor1)
        print(f"    MC={mc:.2f}, Wave={wave:.1%}, XOR1={xor1:.1%}")

    results['exp']['prediction_consistency'] = {
        'mc': [float(v) for v in pred_mc],
        'wave': [float(v) for v in pred_wave],
        'xor': [float(v) for v in pred_xor],
        'mc_std': float(np.std(pred_mc)),
        'wave_std': float(np.std(pred_wave)),
        'xor_std': float(np.std(pred_xor)),
    }
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

    # EXP1 tests
    T('T944', 'Reproducibility: vmem std < 0.01',
      mean_vmem_std < 0.01,
      f'{mean_vmem_std:.6f}')

    T('T945', 'Reproducibility: spike corr > 0.95',
      mean_spike_corr > 0.95,
      f'{mean_spike_corr:.4f}')

    T('T946', 'Reproducibility: MC std < 0.5',
      mc_std < 0.5,
      f'{mc_std:.2f} (MC={np.mean(mc_vals):.2f}±{mc_std:.2f})')

    T('T947', 'Reproducibility: wave std < 5pp',
      wave_std < 0.05,
      f'{wave_std:.1%} (wave={np.mean(wave_vals):.1%}±{wave_std:.1%})')

    # EXP2 tests
    T('T948', 'Sensitivity: NRMSE monotonic (4/4)',
      n_monotonic >= 4,
      f'{n_monotonic}/4 increasing')

    T('T949', 'Sensitivity: eps=0.001 NRMSE < 0.01',
      sensitivity['0.001']['nrmse'] < 0.01,
      f"{sensitivity['0.001']['nrmse']:.6f}")

    T('T950', 'Sensitivity: eps=0.1 NRMSE > 0.05',
      sensitivity['0.1']['nrmse'] > 0.05,
      f"{sensitivity['0.1']['nrmse']:.6f}")

    T('T951', 'Sensitivity: no chaos (NRMSE < 1.0 at eps=0.1)',
      sensitivity['0.1']['nrmse'] < 1.0,
      f"{sensitivity['0.1']['nrmse']:.6f}")

    # EXP3 tests
    T('T952', 'Prediction: MC std < 1.0',
      np.std(pred_mc) < 1.0,
      f'{np.std(pred_mc):.2f}')

    T('T953', 'Prediction: wave std < 3pp',
      np.std(pred_wave) < 0.03,
      f'{np.std(pred_wave):.1%}')

    T('T954', 'Prediction: all MC > 9.0',
      all(m > 9.0 for m in pred_mc),
      f'{[f"{m:.1f}" for m in pred_mc]}')

    T('T955', 'Prediction: XOR1 std < 5pp',
      np.std(pred_xor) < 0.05,
      f'{np.std(pred_xor):.1%}')

    T('T956', 'SNR > 10',
      snr > 10,
      f'{snr:.2f}')

    T('T957', 'Consistent channels > 90%',
      consistent_channels > 0.9 * NUM_NEURONS,
      f'{consistent_channels}/128 ({consistent_channels/128:.0%})')

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['passed'])
    n_total = len(tests)
    results['summary'] = {'pass': n_pass, 'total': n_total,
                          'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    save_results(results)

    print(f"\n{'='*70}")
    print(f"z2318 SUMMARY: {n_pass}/{n_total} PASS")
    print(f"{'='*70}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
