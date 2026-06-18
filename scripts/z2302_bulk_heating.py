#!/usr/bin/env python3
"""
z2302_bulk_heating.py — FPGA Bulk Heating Current: NS-RAM Thermal Memory Retention
===================================================================================
Tests the FPGA's BULK_HEAT register (set_temp) at normalized temperatures [0..1].

Hypothesis: Bulk heating creates analog memory retention in NS-RAM neurons,
improving memory capacity (MC) especially at higher delays.

Temperatures: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
Benchmarks:   MC(d=1..20), XOR(tau=1,3,5), NARMA-5, 4-class waveform
FPGA config:  LEAK=0x2000, THRESH=0x20000, BASE_EXC=0x0080, BIAS_GAIN=0x4000

Tests (10):
  T1:  Baseline (temp=0.0) MC > 8.0
  T2:  At least one temp>0 achieves MC > baseline MC
  T3:  At least one temp>0 achieves MC(d=10+) > baseline MC(d=10+)
  T4:  Best heated MC > 12.0
  T5:  MC(d>=10) correlation with temperature (positive = heating helps memory)
  T6:  XOR1 > 75% for at least 3 temperatures
  T7:  Best heated NARMA-5 < baseline NARMA-5
  T8:  Heating doesn't crash FPGA (all temps produce valid telemetry)
  T9:  Wave4 > 80% for all temperatures
  T10: Optimal temperature is NOT the highest (inverted U = resonance)

Run:
  PYTHONUNBUFFERED=1 taskset -c 0-3 venv/bin/python scripts/z2302_bulk_heating.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2302_bulk_heating.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
N_STEPS = 1500
WARMUP = 300
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


# ============================================================
# Helpers (same as z2298)
# ============================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


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
    print(f"  [TEMP] {label} {temp:.0f}C -> {target:.0f}C...", end="", flush=True)
    t0 = time.time()
    while temp > target and (time.time() - t0) < 180:
        time.sleep(5)
        temp = get_max_temp()
        print(f" {temp:.0f}", end="", flush=True)
    print(f" OK ({time.time()-t0:.0f}s)")
    return temp


def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)


# ============================================================
# FPGA continuous run with thermal safety
# ============================================================
def fpga_run_continuous(fpga, u, mac_signal=None):
    n_steps = len(u)
    if mac_signal is None:
        mac_signal = np.clip(u * 0.3 + 0.3, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ
    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
    valid_count = 0
    for t in range(n_steps):
        # Thermal check every 5 steps
        if t > 0 and t % 5 == 0:
            temp = get_max_temp()
            if temp > 60.0:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}C at step {t}/{n_steps}", end="", flush=True)
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
            valid_count += 1
        elif t > 0:
            states[t] = states[t-1]
            dspikes[t] = dspikes[t-1]
    fpga.set_mac_signal(0.0)
    return states, dspikes, valid_count


# ============================================================
# Temporal product features (same as z2298/z2296)
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

    # Order-2 temporal products
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        if dspikes is not None:
            ds_q = dspikes[:, qi] if dspikes.shape[1] >= n_ch else dspikes[:, :min(n_select, dspikes.shape[1])]
            if ds_q.shape[1] == vm_q.shape[1]:
                feats.append(ds_q * shifted)

    # Order-3 temporal products (limited)
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
    quartiles = np.percentile(u_raw[WARMUP:WARMUP+n], [25, 50, 75])
    u = u_raw[WARMUP:WARMUP+n]
    labels = np.zeros(n)
    labels[u > quartiles[2]] = 3
    labels[(u > quartiles[1]) & (u <= quartiles[2])] = 2
    labels[(u > quartiles[0]) & (u <= quartiles[1])] = 1

    n_tr = int(0.7 * n)
    scores_matrix = np.zeros((n - n_tr, 4))
    for c in range(4):
        y = (labels == c).astype(float)
        for alpha in [1.0, 10.0, 100.0]:
            I = np.eye(X[:n_tr].shape[1])
            try:
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I, X[:n_tr].T @ y[:n_tr])
                scores_matrix[:, c] = X[n_tr:] @ w
                break
            except:
                pass
    pred = np.argmax(scores_matrix, axis=1)
    acc = np.mean(pred == labels[n_tr:])
    return float(acc)


def full_benchmark(X, u_raw):
    n = len(X)
    n_tr = int(0.7 * n)

    # Memory capacity d=1..20
    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 21):
        target = u_raw[WARMUP-d:len(u_raw)-d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    # XOR at tau=1,3,5
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

    # 4-class waveform
    wave_acc = classify_waveform(X, u_raw)

    return {
        'mc_total': mc_total,
        'mc_per_delay': mc_per_d,
        'xor': xor,
        'narma5': best_nrmse,
        'wave4_acc': wave_acc,
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  z2302: Bulk Heating Current — NS-RAM Thermal Memory Retention")
    print("  Temperatures:", TEMPERATURES)
    print("=" * 70)

    # Resume support
    results = {'temperatures': {}, 'tests': {}}
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done = list(results.get('temperatures', {}).keys())
            if done:
                print(f"  RESUMED: temps {done} already done")
        except Exception:
            results = {'temperatures': {}, 'tests': {}}

    # Fixed input signal
    rng = np.random.default_rng(42)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

    # Connect FPGA
    print("\n  Connecting to FPGA...")
    fpga = FPGAEthBridge(timeout=2.0)
    fpga.connect()

    # Configure FPGA (same as z2298)
    fpga.set_kill(0)
    time.sleep(1.0)
    fpga.set_leak_cond(0x2000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    fpga.set_threshold_raw(0x20000)
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    # Zero synapses (independent neurons)
    for n in range(NUM_NEURONS):
        fpga.set_synapse(n, 0x00000000)
        time.sleep(0.001)
    time.sleep(0.5)
    telem = fpga.read_telemetry()
    if telem is not None:
        print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    else:
        print("  WARNING: FPGA telemetry returned None!")

    # Run each temperature
    all_valid = True
    for i, temp in enumerate(TEMPERATURES):
        key = f"temp_{temp:.2f}"
        if key in results.get('temperatures', {}):
            print(f"\n[{i+1}/{len(TEMPERATURES)}] temp={temp:.2f} -- already done, skipping")
            continue

        print(f"\n[{i+1}/{len(TEMPERATURES)}] temp={temp:.2f}")
        wait_cool(f"pre-temp{temp:.2f}")

        # Set bulk heating
        fpga.set_temp(temp)
        time.sleep(0.5)  # let FPGA settle with new heating current

        # Verify FPGA alive
        telem = fpga.read_telemetry()
        if telem is None:
            print(f"  WARNING: No telemetry at temp={temp:.2f}, FPGA may have crashed")
            results['temperatures'][key] = {'error': 'no_telemetry', 'valid': False}
            all_valid = False
            save_results(results)
            continue

        print(f"  vmem range [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

        # Run reservoir
        states, dspikes, valid_count = fpga_run_continuous(fpga, u_raw)
        valid_frac = valid_count / len(u_raw)
        print(f"  Valid telemetry: {valid_count}/{len(u_raw)} ({valid_frac*100:.1f}%)")

        if valid_frac < 0.5:
            print(f"  SKIP: too few valid readings")
            results['temperatures'][key] = {'error': 'low_valid', 'valid_frac': valid_frac, 'valid': False}
            all_valid = False
            save_results(results)
            continue

        # Build features and benchmark
        X = build_temporal_features(states[WARMUP:], dspikes[WARMUP:], n_select=24, seed=42)
        bm = full_benchmark(X, u_raw)
        bm['valid'] = True
        bm['valid_frac'] = valid_frac
        bm['temperature'] = temp

        results['temperatures'][key] = bm
        xor = bm['xor']
        print(f"  MC={bm['mc_total']:.2f}  XOR1={xor['tau1']*100:.1f}%  XOR3={xor['tau3']*100:.1f}%  "
              f"XOR5={xor['tau5']*100:.1f}%  NARMA5={bm['narma5']:.3f}  Wave4={bm['wave4_acc']*100:.1f}%")

        # MC breakdown for long delays
        mc_long = sum(bm['mc_per_delay'].get(str(d), 0) for d in range(10, 21))
        print(f"  MC(d>=10)={mc_long:.2f}")

        save_results(results)

    # Reset heating to zero
    fpga.set_temp(0.0)
    fpga.close()

    # ============================================================
    # Evaluate tests
    # ============================================================
    print("\n" + "=" * 70)
    print("  TEST EVALUATION")
    print("=" * 70)

    temps_data = results['temperatures']
    tests = {}

    # Gather valid results
    valid_temps = {k: v for k, v in temps_data.items() if v.get('valid', False)}
    baseline_key = "temp_0.00"
    baseline = valid_temps.get(baseline_key, {})
    heated = {k: v for k, v in valid_temps.items() if k != baseline_key}

    baseline_mc = baseline.get('mc_total', 0)
    baseline_narma5 = baseline.get('narma5', 999)

    # T1: Baseline MC > 8.0
    t1_pass = baseline_mc > 8.0
    tests['T1_baseline_mc_gt_8'] = {
        'pass': t1_pass,
        'baseline_mc': baseline_mc,
        'threshold': 8.0,
    }
    print(f"  T1  baseline MC > 8.0:           {'PASS' if t1_pass else 'FAIL'}  (MC={baseline_mc:.2f})")

    # T2: At least one heated MC > baseline
    heated_mcs = {k: v.get('mc_total', 0) for k, v in heated.items()}
    best_heated_mc = max(heated_mcs.values()) if heated_mcs else 0
    t2_pass = best_heated_mc > baseline_mc if baseline_mc > 0 else False
    tests['T2_heated_mc_gt_baseline'] = {
        'pass': t2_pass,
        'baseline_mc': baseline_mc,
        'best_heated_mc': best_heated_mc,
    }
    print(f"  T2  heated MC > baseline:        {'PASS' if t2_pass else 'FAIL'}  (best={best_heated_mc:.2f} vs base={baseline_mc:.2f})")

    # T3: At least one heated MC(d>=10) > baseline MC(d>=10)
    def mc_long_range(bm_data):
        mc_d = bm_data.get('mc_per_delay', {})
        return sum(mc_d.get(str(d), 0) for d in range(10, 21))

    baseline_mc_long = mc_long_range(baseline)
    heated_mc_longs = {k: mc_long_range(v) for k, v in heated.items()}
    best_heated_mc_long = max(heated_mc_longs.values()) if heated_mc_longs else 0
    t3_pass = best_heated_mc_long > baseline_mc_long if baseline_mc_long >= 0 else False
    tests['T3_heated_long_mc_gt_baseline'] = {
        'pass': t3_pass,
        'baseline_mc_long': baseline_mc_long,
        'best_heated_mc_long': best_heated_mc_long,
    }
    print(f"  T3  heated MC(d>=10) > baseline: {'PASS' if t3_pass else 'FAIL'}  (best={best_heated_mc_long:.2f} vs base={baseline_mc_long:.2f})")

    # T4: Best heated MC > 12.0
    t4_pass = best_heated_mc > 12.0
    tests['T4_best_heated_mc_gt_12'] = {
        'pass': t4_pass,
        'best_heated_mc': best_heated_mc,
        'threshold': 12.0,
    }
    print(f"  T4  best heated MC > 12.0:       {'PASS' if t4_pass else 'FAIL'}  (best={best_heated_mc:.2f})")

    # T5: MC(d>=10) correlation with temperature
    if len(valid_temps) >= 3:
        temp_vals = []
        mc_long_vals = []
        for k, v in sorted(valid_temps.items()):
            temp_vals.append(v.get('temperature', 0))
            mc_long_vals.append(mc_long_range(v))
        corr = np.corrcoef(temp_vals, mc_long_vals)[0, 1] if len(temp_vals) > 1 else 0
        t5_pass = corr > 0
    else:
        corr = 0
        t5_pass = False
    tests['T5_mc_long_temp_correlation'] = {
        'pass': t5_pass,
        'correlation': float(corr),
    }
    print(f"  T5  MC(d>=10) ~ temp corr > 0:  {'PASS' if t5_pass else 'FAIL'}  (r={corr:.3f})")

    # T6: XOR1 > 75% for at least 3 temperatures
    xor1_above = sum(1 for v in valid_temps.values()
                     if v.get('xor', {}).get('tau1', 0) > 0.75)
    t6_pass = xor1_above >= 3
    tests['T6_xor1_gt_75pct_3temps'] = {
        'pass': t6_pass,
        'count_above': xor1_above,
        'threshold_count': 3,
    }
    print(f"  T6  XOR1>75% for >=3 temps:      {'PASS' if t6_pass else 'FAIL'}  (count={xor1_above})")

    # T7: Best heated NARMA-5 < baseline NARMA-5
    heated_narmas = {k: v.get('narma5', 999) for k, v in heated.items()}
    best_heated_narma = min(heated_narmas.values()) if heated_narmas else 999
    t7_pass = best_heated_narma < baseline_narma5
    tests['T7_heated_narma5_lt_baseline'] = {
        'pass': t7_pass,
        'baseline_narma5': baseline_narma5,
        'best_heated_narma5': best_heated_narma,
    }
    print(f"  T7  heated NARMA5 < baseline:    {'PASS' if t7_pass else 'FAIL'}  (best={best_heated_narma:.3f} vs base={baseline_narma5:.3f})")

    # T8: All temps produce valid telemetry
    t8_pass = all(v.get('valid', False) for v in temps_data.values()) and len(temps_data) == len(TEMPERATURES)
    tests['T8_all_temps_valid'] = {
        'pass': t8_pass,
        'total_temps': len(TEMPERATURES),
        'valid_temps': sum(1 for v in temps_data.values() if v.get('valid', False)),
    }
    print(f"  T8  all temps valid telemetry:   {'PASS' if t8_pass else 'FAIL'}  ({tests['T8_all_temps_valid']['valid_temps']}/{len(TEMPERATURES)})")

    # T9: Wave4 > 80% for all temperatures
    wave_accs = {k: v.get('wave4_acc', 0) for k, v in valid_temps.items()}
    t9_pass = all(a > 0.80 for a in wave_accs.values()) and len(wave_accs) == len(TEMPERATURES)
    tests['T9_wave4_gt_80pct_all'] = {
        'pass': t9_pass,
        'wave4_accs': {k: round(v, 4) for k, v in wave_accs.items()},
    }
    print(f"  T9  Wave4>80% all temps:         {'PASS' if t9_pass else 'FAIL'}  (min={min(wave_accs.values())*100:.1f}%)" if wave_accs else "  T9  Wave4>80% all temps:         FAIL  (no data)")

    # T10: Optimal temperature is NOT the highest (inverted U = resonance)
    if heated_mcs:
        best_temp_key = max(heated_mcs, key=heated_mcs.get)
        best_temp_val = valid_temps[best_temp_key].get('temperature', 0)
        max_temp = max(TEMPERATURES)
        t10_pass = best_temp_val < max_temp
    else:
        best_temp_val = 0
        max_temp = max(TEMPERATURES)
        t10_pass = False
    tests['T10_optimal_not_max_temp'] = {
        'pass': t10_pass,
        'optimal_temp': best_temp_val,
        'max_temp': max_temp,
    }
    print(f"  T10 optimal temp != max:         {'PASS' if t10_pass else 'FAIL'}  (optimal={best_temp_val:.2f}, max={max_temp:.2f})")

    # Summary
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    print(f"\n  RESULT: {n_pass}/{n_total} PASS")

    results['tests'] = tests
    results['summary'] = {
        'pass': n_pass,
        'total': n_total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_results(results)
    print(f"\n  Saved to {SAVE_FILE}")


if __name__ == '__main__':
    main()
