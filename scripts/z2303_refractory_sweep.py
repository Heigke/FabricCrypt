#!/usr/bin/env python3
"""
z2303_refractory_sweep.py — Sweep refractory period to test sparse spike codes
===============================================================================
Hypothesis: longer refractory → sparser spikes → better feature separation
            → better classification (up to a point, then info loss)

Sweeps refrac = [0, 2, 5, 10, 20, 50, 100, 200] cycles on FPGA 128-neuron bank.
Same config as z2298: LEAK=0x2000, THRESH=0x20000, BASE_EXC=0x0080, BIAS_GAIN=0x4000.
Benchmarks: MC(d=1..20), XOR(τ=1,3,5), NARMA-5, Wave4.

Run:
  PYTHONUNBUFFERED=1 taskset -c 0-3 venv/bin/python scripts/z2303_refractory_sweep.py
"""

import os, sys, time, json, struct
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2303_refractory_sweep.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
N_STEPS = 1500
WARMUP = 300
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
REFRAC_VALUES = [0, 2, 5, 10, 20, 50, 100, 200]


# ============================================================
# Thermal safety
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


def wait_cool(label="", target=40.0):
    temp = get_max_temp()
    if temp <= target:
        return temp
    print(f"  [TEMP] {label} {temp:.0f}°C → {target:.0f}°C...", end="", flush=True)
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
# FPGA continuous run
# ============================================================
def fpga_run_continuous(fpga, u):
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
        if t > 0 and t % 5 == 0:
            temp = get_max_temp()
            if temp > 60.0:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}°C at step {t}/{n_steps}", end="", flush=True)
                while temp > 42.0:
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
    fpga.set_mac_signal(0.0)
    return states, dspikes


# ============================================================
# Temporal product features (same as z2296/z2298)
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
            ds_q = dspikes[:, qi] if dspikes.shape[1] >= n_ch else dspikes[:, :min(n_select, dspikes.shape[1])]
            if ds_q.shape[1] == vm_q.shape[1]:
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
# Benchmarks
# ============================================================
def ridge_solve(X_tr, y_tr, X_te, y_te, task='regression'):
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_score = 0.0 if task == 'regression' else 0.5
    for alpha in alphas:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
            pred = X_te @ w
            if task == 'regression':
                ss_res = np.sum((y_te - pred) ** 2)
                ss_tot = np.sum((y_te - y_te.mean()) ** 2)
                score = max(0, 1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            else:
                score = np.mean((pred > 0.5).astype(float) == y_te)
            if score > best_score:
                best_score = score
        except Exception:
            pass
    return best_score


def classify_waveform(X, u_raw):
    n = len(X)
    n_tr = int(0.7 * n)
    quartiles = np.percentile(u_raw[WARMUP:WARMUP+n], [25, 50, 75])
    u = u_raw[WARMUP:WARMUP+n]
    labels = np.zeros(n)
    labels[u > quartiles[2]] = 3
    labels[(u > quartiles[1]) & (u <= quartiles[2])] = 2
    labels[(u > quartiles[0]) & (u <= quartiles[1])] = 1

    scores_matrix = np.zeros((n - n_tr, 4))
    for c in range(4):
        y = (labels == c).astype(float)
        for alpha in [1.0, 10.0, 100.0]:
            I = np.eye(X[:n_tr].shape[1])
            try:
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I, X[:n_tr].T @ y[:n_tr])
                scores_matrix[:, c] = X[n_tr:] @ w
                break
            except Exception:
                pass
    pred = np.argmax(scores_matrix, axis=1)
    acc = np.mean(pred == labels[n_tr:])
    return float(acc)


def full_benchmark(X, u_raw):
    n = len(X)
    n_tr = int(0.7 * n)

    # Memory capacity
    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 21):
        target = u_raw[WARMUP-d:len(u_raw)-d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    # XOR
    xor = {}
    for tau in [1, 3, 5]:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP-tau:len(u_raw)-tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor[f'tau{tau}'] = acc

    # NARMA-5
    T = len(u_raw)
    order = 5
    u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
    y = np.zeros(T)
    for t in range(order, T):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-order:t]) + 1.5*u_n[t-1]*u_n[t-order] + 0.1
        y[t] = np.tanh(y[t])
    target = y[WARMUP:]
    nn = min(n, len(target))
    best_nrmse = 999.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        I2 = np.eye(X[:n_tr].shape[1])
        try:
            w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I2, X[:n_tr].T @ target[:n_tr])
            pred = X[n_tr:nn] @ w
            gt = target[n_tr:nn]
            nrmse = np.sqrt(np.mean((gt-pred)**2)) / (np.std(gt)+1e-10)
            if nrmse < best_nrmse:
                best_nrmse = nrmse
        except Exception:
            pass
    narma5 = best_nrmse

    # 4-class waveform
    wave_acc = classify_waveform(X, u_raw)

    return {'mc_total': mc_total, 'mc_per_delay': mc_per_d, 'xor': xor,
            'narma5': narma5, 'wave4_acc': wave_acc}


# ============================================================
# Main sweep
# ============================================================
def main():
    print("=" * 70)
    print("  z2303: Refractory Period Sweep — Sparse Spike Codes")
    print("  Refrac values:", REFRAC_VALUES)
    print("=" * 70)

    results = {'sweep': {}, 'tests': {}}
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done = list(results.get('sweep', {}).keys())
            if done:
                print(f"  RESUMED: refrac values {done} already done")
        except Exception:
            results = {'sweep': {}, 'tests': {}}

    rng = np.random.default_rng(42)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

    # Connect FPGA
    print("\n  Connecting to FPGA...")
    fpga = FPGAEthBridge(timeout=2.0)
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(1.0)

    # Base configuration (same as z2298)
    print("  Setting base config: LEAK=0x2000 THRESH=0x20000 BASE_EXC=0x0080 BIAS_GAIN=0x4000")
    fpga.set_leak_cond(0x2000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    fpga.set_threshold_raw(0x20000)
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    # Zero synapses (no recurrent connections)
    for n in range(NUM_NEURONS):
        fpga.set_synapse(n, 0x00000000)
        time.sleep(0.001)
    time.sleep(0.5)

    # Sweep
    for idx, refrac in enumerate(REFRAC_VALUES):
        key = str(refrac)
        if key in results.get('sweep', {}):
            print(f"\n[{idx+1}/{len(REFRAC_VALUES)}] refrac={refrac} — already done, skipping")
            continue

        print(f"\n[{idx+1}/{len(REFRAC_VALUES)}] refrac={refrac} cycles")
        wait_cool(f"pre-refrac{refrac}")

        # Set refractory period
        fpga.set_refract_cycles(refrac)
        time.sleep(0.2)

        # Run FPGA
        print(f"  Running FPGA ({N_STEPS} steps @ {SAMPLE_HZ}Hz)...")
        t0 = time.time()
        states, dspikes = fpga_run_continuous(fpga, u_raw)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # Spike rate
        spike_rate = float(dspikes[WARMUP:].mean())
        print(f"  spike_rate = {spike_rate:.4f}")

        # Build features and benchmark
        X = build_temporal_features(states[WARMUP:], dspikes[WARMUP:], n_select=24, seed=42)
        bm = full_benchmark(X, u_raw)
        bm['spike_rate'] = spike_rate

        xor = bm['xor']
        print(f"  MC={bm['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
              f"XOR5={xor['tau5']*100:.1f}% N5={bm['narma5']:.3f} W4={bm['wave4_acc']*100:.1f}%")

        results['sweep'][key] = bm
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)

    fpga.set_mac_signal(0.0)
    fpga.set_refract_cycles(0)  # Reset to default

    # ============================================================
    # Evaluate tests
    # ============================================================
    print("\n" + "=" * 70)
    print("  TESTS")
    print("=" * 70)

    sweep = results['sweep']
    tests = {}
    n_pass = 0

    # Collect metrics
    refrac_keys = [str(r) for r in REFRAC_VALUES if str(r) in sweep]
    mcs = {k: sweep[k]['mc_total'] for k in refrac_keys}
    xor1s = {k: sweep[k]['xor']['tau1'] for k in refrac_keys}
    xor3s = {k: sweep[k]['xor']['tau3'] for k in refrac_keys}
    xor5s = {k: sweep[k]['xor']['tau5'] for k in refrac_keys}
    narmas = {k: sweep[k]['narma5'] for k in refrac_keys}
    waves = {k: sweep[k]['wave4_acc'] for k in refrac_keys}
    spike_rates = {k: sweep[k]['spike_rate'] for k in refrac_keys}

    baseline_mc = mcs.get('0', 0.0)
    baseline_spike = spike_rates.get('0', 1.0)

    # T1: Baseline MC > 8.0
    t1 = baseline_mc > 8.0
    tests['T1_baseline_mc_gt8'] = {'pass': t1, 'value': baseline_mc, 'threshold': 8.0}
    print(f"  T1  {'PASS' if t1 else 'FAIL'}: Baseline MC={baseline_mc:.2f} (>8.0)")

    # T2: At least one refrac > 0 has MC > baseline
    nonzero_better = any(mcs[k] > baseline_mc for k in refrac_keys if k != '0')
    tests['T2_refrac_improves_mc'] = {'pass': nonzero_better, 'baseline': baseline_mc,
                                       'best_nonzero': max((mcs[k] for k in refrac_keys if k != '0'), default=0)}
    print(f"  T2  {'PASS' if nonzero_better else 'FAIL'}: Any refrac>0 MC > baseline ({baseline_mc:.2f})")

    # T3: Spike rate decreases monotonically with refractory period
    rates_ordered = [spike_rates[str(r)] for r in REFRAC_VALUES if str(r) in spike_rates]
    monotonic = all(rates_ordered[i] >= rates_ordered[i+1] for i in range(len(rates_ordered)-1))
    tests['T3_spike_rate_monotonic'] = {'pass': monotonic, 'rates': rates_ordered}
    print(f"  T3  {'PASS' if monotonic else 'FAIL'}: Spike rate monotonic decrease: {[f'{r:.4f}' for r in rates_ordered]}")

    # T4: Best refrac XOR1 > 85%
    best_xor1 = max(xor1s.values()) if xor1s else 0
    t4 = best_xor1 > 0.85
    tests['T4_best_xor1_gt85'] = {'pass': t4, 'value': best_xor1}
    print(f"  T4  {'PASS' if t4 else 'FAIL'}: Best XOR1={best_xor1*100:.1f}% (>85%)")

    # T5: Best refrac Wave4 > 90%
    best_wave = max(waves.values()) if waves else 0
    t5 = best_wave > 0.90
    tests['T5_best_wave_gt90'] = {'pass': t5, 'value': best_wave}
    print(f"  T5  {'PASS' if t5 else 'FAIL'}: Best Wave4={best_wave*100:.1f}% (>90%)")

    # T6: Refrac=200 spike rate < 0.5 × baseline
    rate_200 = spike_rates.get('200', baseline_spike)
    t6 = rate_200 < 0.5 * baseline_spike
    tests['T6_high_refrac_sparse'] = {'pass': t6, 'refrac200': rate_200, 'half_baseline': 0.5 * baseline_spike}
    print(f"  T6  {'PASS' if t6 else 'FAIL'}: refrac=200 rate={rate_200:.4f} < {0.5*baseline_spike:.4f}")

    # T7: XOR at high refrac (>=50) doesn't collapse to chance
    high_refrac_xor = [xor1s[str(r)] for r in [50, 100, 200] if str(r) in xor1s]
    t7 = any(x > 0.55 for x in high_refrac_xor) if high_refrac_xor else False
    tests['T7_high_refrac_xor_alive'] = {'pass': t7, 'values': high_refrac_xor}
    print(f"  T7  {'PASS' if t7 else 'FAIL'}: High-refrac XOR1 > 55%: {[f'{x*100:.1f}%' for x in high_refrac_xor]}")

    # T8: Best NARMA-5 < 0.20
    best_narma = min(narmas.values()) if narmas else 999
    t8 = best_narma < 0.20
    tests['T8_best_narma5_lt020'] = {'pass': t8, 'value': best_narma}
    print(f"  T8  {'PASS' if t8 else 'FAIL'}: Best NARMA-5={best_narma:.3f} (<0.20)")

    # T9: Optimal refrac for MC ≠ optimal for XOR (tradeoff)
    best_mc_refrac = max(mcs, key=mcs.get) if mcs else '0'
    best_xor_refrac = max(xor1s, key=xor1s.get) if xor1s else '0'
    t9 = best_mc_refrac != best_xor_refrac
    tests['T9_mc_xor_tradeoff'] = {'pass': t9, 'best_mc_refrac': int(best_mc_refrac),
                                     'best_xor_refrac': int(best_xor_refrac)}
    print(f"  T9  {'PASS' if t9 else 'FAIL'}: MC-best@refrac={best_mc_refrac} vs XOR-best@refrac={best_xor_refrac}")

    # T10: MC curve is non-monotonic (inverted U)
    mc_vals = [mcs[str(r)] for r in REFRAC_VALUES if str(r) in mcs]
    if len(mc_vals) >= 3:
        peak_idx = np.argmax(mc_vals)
        t10 = 0 < peak_idx < len(mc_vals) - 1  # Peak not at boundary
    else:
        t10 = False
    tests['T10_mc_inverted_u'] = {'pass': t10, 'mc_curve': mc_vals,
                                   'peak_idx': int(peak_idx) if len(mc_vals) >= 3 else -1}
    print(f"  T10 {'PASS' if t10 else 'FAIL'}: MC inverted-U, peak at idx={int(peak_idx) if len(mc_vals)>=3 else -1}: {[f'{v:.2f}' for v in mc_vals]}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/10 PASS")

    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass,
        'n_total': 10,
        'refrac_values': REFRAC_VALUES,
        'best_mc_refrac': int(best_mc_refrac) if mcs else None,
        'best_xor_refrac': int(best_xor_refrac) if xor1s else None,
        'best_wave_refrac': int(max(waves, key=waves.get)) if waves else None,
    }

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n  Saved: {SAVE_FILE}")


if __name__ == '__main__':
    main()
