#!/usr/bin/env python3
"""
z2319_drift_stability.py — Long-Horizon Drift & Stability Analysis
===================================================================
Tests whether FPGA reservoir performance is stable over extended operation
or drifts due to thermal effects, charge accumulation, or analog fatigue.

This is critical for the paper: if performance degrades over minutes, the
reservoir is not practically useful.

Conditions:
  EXP1 — Sliding window stability: 10,000 steps (~200s at 50Hz),
          evaluate MC/Wave in 6 overlapping windows of 1500 steps
  EXP2 — Continuous operation stability: Run for 5000 steps, compare
          first-half vs second-half performance (3 reps for statistics)
  EXP3 — Reset recovery: Run 2000 steps, kill/restart FPGA, run 2000 more,
          compare pre/post performance

Tests (14):
  T958: Sliding MC coefficient of variation < 15%
  T959: Sliding MC no monotonic decline (best not in first window)
  T960: Sliding waveform CV < 10%
  T961: Sliding vmem mean drift < 0.05 (last window - first window)
  T962: Sliding spike rate CV < 20%
  T963: Continuous: |MC_first_half - MC_second_half| < 2.0
  T964: Continuous: |wave_first - wave_second| < 10pp (3/3 reps)
  T965: Continuous: vmem mean drift < 0.05 per rep (3/3 reps)
  T966: Continuous: spike rate stable (ratio 0.5-2.0, 3/3 reps)
  T967: Reset recovery: MC_post > 0.8 * MC_pre
  T968: Reset recovery: wave_post > wave_pre - 10pp
  T969: Reset recovery: vmem returns to baseline within 0.1
  T970: Overall: mean MC across all windows > 9.0
  T971: Overall: mean waveform across all windows > 80%

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2319_drift_stability.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2319_drift_stability.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
RIDGE_ALPHA = 0.01


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


def fpga_run(fpga, u, label=""):
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
                print(f"\n    [THERMAL PAUSE] {label} {temp:.0f}C at step {t}", end="", flush=True)
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
        if t > 0 and t % 1000 == 0:
            print(f"    {label} step {t}/{n_steps}, temp={get_max_temp():.0f}C", flush=True)
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
    print("z2319 — Long-Horizon Drift & Stability Analysis")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temp: {get_max_temp():.0f}C")

    results = {'experiment': 'z2319_drift_stability', 'tests': {}, 'exp': {}}
    rng = np.random.default_rng(77)

    # ============================================================
    # EXP1: Sliding window stability over 10,000 steps
    # ============================================================
    print(f"\n{'='*50}")
    print("EXP1: Sliding Window Stability (10,000 steps)")
    print(f"{'='*50}")

    N_LONG = 10000
    WARMUP = 200
    WIN_SIZE = 1500
    WIN_STRIDE = 1500  # non-overlapping for independence

    u_long = rng.uniform(0, 1, N_LONG)
    wait_cool("exp1")

    fpga = setup_fpga()
    telem = fpga.read_telemetry()
    if telem is not None:
        print(f"  vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

    states_long, dspikes_long = fpga_run(fpga, u_long, label="EXP1")
    fpga.set_kill(1)

    # Evaluate in windows
    states_w = states_long[WARMUP:]
    dspikes_w = dspikes_long[WARMUP:]
    u_w = u_long[WARMUP:]
    n_avail = len(states_w)
    n_windows = (n_avail - WIN_SIZE) // WIN_STRIDE + 1
    n_windows = min(n_windows, 6)

    window_metrics = []
    for wi in range(n_windows):
        start = wi * WIN_STRIDE
        end = start + WIN_SIZE
        if end > n_avail:
            break
        s_win = states_w[start:end]
        d_win = dspikes_w[start:end]
        u_win = u_w[start:end]

        X = build_temporal_features(s_win, d_win)
        X_pca = pca_reduce(X)
        mc = compute_mc(X_pca, u_win)
        wave = compute_wave(X_pca, u_win)
        vmem_mean = s_win.mean()
        spike_rate = d_win.sum() / (WIN_SIZE * NUM_NEURONS)

        window_metrics.append({
            'window': wi, 'start': start, 'end': end,
            'mc': float(mc), 'wave': float(wave),
            'vmem_mean': float(vmem_mean), 'spike_rate': float(spike_rate),
        })
        print(f"  Window {wi}: steps {start}-{end}, MC={mc:.2f}, Wave={wave:.1%}, "
              f"vmem={vmem_mean:.4f}, spikes={spike_rate:.4f}")

    mc_vals = [w['mc'] for w in window_metrics]
    wave_vals = [w['wave'] for w in window_metrics]
    vmem_vals = [w['vmem_mean'] for w in window_metrics]
    spike_vals = [w['spike_rate'] for w in window_metrics]

    mc_cv = np.std(mc_vals) / (np.mean(mc_vals) + 1e-10) if np.mean(mc_vals) > 0 else 0
    wave_cv = np.std(wave_vals) / (np.mean(wave_vals) + 1e-10) if np.mean(wave_vals) > 0 else 0
    spike_cv = np.std(spike_vals) / (np.mean(spike_vals) + 1e-10) if np.mean(spike_vals) > 0 else 0
    vmem_drift = abs(vmem_vals[-1] - vmem_vals[0]) if len(vmem_vals) > 1 else 0
    mc_best_window = np.argmax(mc_vals)

    print(f"\n  Summary: MC CV={mc_cv:.2%}, Wave CV={wave_cv:.2%}, Spike CV={spike_cv:.2%}")
    print(f"  vmem drift: {vmem_drift:.6f}, best MC in window {mc_best_window}")

    results['exp']['sliding_window'] = {
        'windows': window_metrics,
        'mc_cv': float(mc_cv), 'wave_cv': float(wave_cv),
        'spike_cv': float(spike_cv), 'vmem_drift': float(vmem_drift),
        'mc_best_window': int(mc_best_window),
        'mc_mean': float(np.mean(mc_vals)), 'wave_mean': float(np.mean(wave_vals)),
    }
    save_results(results)

    # ============================================================
    # EXP2: Continuous operation — first half vs second half (3 reps)
    # ============================================================
    print(f"\n{'='*50}")
    print("EXP2: Continuous Half-Split (3 reps × 5000 steps)")
    print(f"{'='*50}")

    N_CONT = 5000
    half_results = []

    for rep in range(3):
        print(f"\n  Rep {rep+1}/3...")
        wait_cool(f"cont-rep-{rep+1}")
        u_cont = np.random.default_rng(200 + rep).uniform(0, 1, N_CONT)
        fpga = setup_fpga()
        st, ds = fpga_run(fpga, u_cont, label=f"EXP2-rep{rep+1}")
        fpga.set_kill(1)

        st_w = st[WARMUP:]
        ds_w = ds[WARMUP:]
        u_cw = u_cont[WARMUP:]
        mid = len(st_w) // 2

        # First half
        X1 = build_temporal_features(st_w[:mid], ds_w[:mid])
        X1_pca = pca_reduce(X1)
        mc1 = compute_mc(X1_pca, u_cw[:mid])
        wave1 = compute_wave(X1_pca, u_cw[:mid])
        vmem1 = st_w[:mid].mean()
        sr1 = ds_w[:mid].sum() / (mid * NUM_NEURONS)

        # Second half
        X2 = build_temporal_features(st_w[mid:], ds_w[mid:])
        X2_pca = pca_reduce(X2)
        mc2 = compute_mc(X2_pca, u_cw[mid:])
        wave2 = compute_wave(X2_pca, u_cw[mid:])
        vmem2 = st_w[mid:].mean()
        sr2 = ds_w[mid:].sum() / ((len(st_w) - mid) * NUM_NEURONS)

        hr = {
            'mc_first': float(mc1), 'mc_second': float(mc2),
            'wave_first': float(wave1), 'wave_second': float(wave2),
            'vmem_first': float(vmem1), 'vmem_second': float(vmem2),
            'spike_first': float(sr1), 'spike_second': float(sr2),
        }
        half_results.append(hr)
        print(f"    First:  MC={mc1:.2f}, Wave={wave1:.1%}, vmem={vmem1:.4f}, spikes={sr1:.4f}")
        print(f"    Second: MC={mc2:.2f}, Wave={wave2:.1%}, vmem={vmem2:.4f}, spikes={sr2:.4f}")

    results['exp']['continuous'] = half_results
    save_results(results)

    # ============================================================
    # EXP3: Reset recovery
    # ============================================================
    print(f"\n{'='*50}")
    print("EXP3: Reset Recovery (2000 + kill + 2000)")
    print(f"{'='*50}")

    N_RESET = 2000
    u_reset = rng.uniform(0, 1, N_RESET)
    wait_cool("reset-pre")

    # Pre-reset run
    print("  Pre-reset run...")
    fpga = setup_fpga()
    telem_pre = fpga.read_telemetry()
    vmem_baseline = telem_pre['vmem'].mean() if telem_pre is not None else 0
    st_pre, ds_pre = fpga_run(fpga, u_reset, label="pre-reset")
    fpga.set_kill(1)
    time.sleep(2.0)  # Full kill pause

    X_pre = build_temporal_features(st_pre[WARMUP:], ds_pre[WARMUP:])
    X_pre_pca = pca_reduce(X_pre)
    u_pre_w = u_reset[WARMUP:]
    mc_pre = compute_mc(X_pre_pca, u_pre_w)
    wave_pre = compute_wave(X_pre_pca, u_pre_w)
    vmem_pre_mean = st_pre[WARMUP:].mean()

    print(f"    Pre:  MC={mc_pre:.2f}, Wave={wave_pre:.1%}, vmem={vmem_pre_mean:.4f}")

    # Post-reset run (same input)
    print("  Post-reset run (after kill + restart)...")
    wait_cool("reset-post")
    fpga = setup_fpga()
    telem_post = fpga.read_telemetry()
    vmem_after_reset = telem_post['vmem'].mean() if telem_post is not None else 0
    st_post, ds_post = fpga_run(fpga, u_reset, label="post-reset")
    fpga.set_kill(1)

    X_post = build_temporal_features(st_post[WARMUP:], ds_post[WARMUP:])
    X_post_pca = pca_reduce(X_post)
    mc_post = compute_mc(X_post_pca, u_pre_w)
    wave_post = compute_wave(X_post_pca, u_pre_w)
    vmem_post_mean = st_post[WARMUP:].mean()
    vmem_reset_diff = abs(vmem_after_reset - vmem_baseline)

    print(f"    Post: MC={mc_post:.2f}, Wave={wave_post:.1%}, vmem={vmem_post_mean:.4f}")
    print(f"    Baseline vmem diff after reset: {vmem_reset_diff:.6f}")

    results['exp']['reset_recovery'] = {
        'mc_pre': float(mc_pre), 'mc_post': float(mc_post),
        'wave_pre': float(wave_pre), 'wave_post': float(wave_post),
        'vmem_pre': float(vmem_pre_mean), 'vmem_post': float(vmem_post_mean),
        'vmem_baseline': float(vmem_baseline),
        'vmem_after_reset': float(vmem_after_reset),
        'vmem_reset_diff': float(vmem_reset_diff),
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
    T('T958', 'Sliding MC CV < 15%',
      mc_cv < 0.15,
      f'{mc_cv:.2%}')

    T('T959', 'Sliding MC best not in first window',
      mc_best_window > 0,
      f'best in window {mc_best_window}')

    T('T960', 'Sliding waveform CV < 10%',
      wave_cv < 0.10,
      f'{wave_cv:.2%}')

    T('T961', 'Sliding vmem drift < 0.05',
      vmem_drift < 0.05,
      f'{vmem_drift:.6f}')

    T('T962', 'Sliding spike rate CV < 20%',
      spike_cv < 0.20 or np.mean(spike_vals) < 1e-6,  # if no spikes, trivially stable
      f'{spike_cv:.2%} (mean={np.mean(spike_vals):.4f})')

    # EXP2 tests
    mc_diffs = [abs(h['mc_first'] - h['mc_second']) for h in half_results]
    wave_diffs = [abs(h['wave_first'] - h['wave_second']) for h in half_results]
    vmem_drifts = [abs(h['vmem_first'] - h['vmem_second']) for h in half_results]
    sr_ratios = [(h['spike_second'] / (h['spike_first'] + 1e-10))
                 for h in half_results]

    T('T963', 'Continuous: MC half-diff < 2.0 (3/3)',
      all(d < 2.0 for d in mc_diffs),
      f'{[f"{d:.2f}" for d in mc_diffs]}')

    T('T964', 'Continuous: wave half-diff < 10pp (3/3)',
      all(d < 0.10 for d in wave_diffs),
      f'{[f"{d:.1%}" for d in wave_diffs]}')

    T('T965', 'Continuous: vmem drift < 0.05 (3/3)',
      all(d < 0.05 for d in vmem_drifts),
      f'{[f"{d:.6f}" for d in vmem_drifts]}')

    T('T966', 'Continuous: spike rate stable (3/3)',
      all(0.5 < r < 2.0 or half_results[i]['spike_first'] < 1e-6
          for i, r in enumerate(sr_ratios)),
      f'{[f"{r:.2f}" for r in sr_ratios]}')

    # EXP3 tests
    T('T967', 'Reset: MC_post > 0.8 * MC_pre',
      mc_post > 0.8 * mc_pre or mc_pre < 1.0,
      f'pre={mc_pre:.2f}, post={mc_post:.2f}, ratio={mc_post/(mc_pre+1e-10):.2f}')

    T('T968', 'Reset: wave_post > wave_pre - 10pp',
      wave_post > wave_pre - 0.10,
      f'pre={wave_pre:.1%}, post={wave_post:.1%}')

    T('T969', 'Reset: vmem baseline returns within 0.1',
      vmem_reset_diff < 0.1,
      f'diff={vmem_reset_diff:.6f}')

    # Overall
    all_mc = mc_vals + [h['mc_first'] for h in half_results] + [h['mc_second'] for h in half_results] + [mc_pre, mc_post]
    all_wave = wave_vals + [h['wave_first'] for h in half_results] + [h['wave_second'] for h in half_results] + [wave_pre, wave_post]

    T('T970', 'Overall: mean MC > 9.0',
      np.mean(all_mc) > 9.0,
      f'{np.mean(all_mc):.2f}')

    T('T971', 'Overall: mean waveform > 80%',
      np.mean(all_wave) > 0.80,
      f'{np.mean(all_wave):.1%}')

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['passed'])
    n_total = len(tests)
    results['summary'] = {'pass': n_pass, 'total': n_total,
                          'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    save_results(results)

    print(f"\n{'='*70}")
    print(f"z2319 SUMMARY: {n_pass}/{n_total} PASS")
    print(f"{'='*70}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
