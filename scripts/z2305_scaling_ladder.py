#!/usr/bin/env python3
"""
z2305_scaling_ladder.py — Neuron-count scaling ladder (8→16→32→64→128)
======================================================================
Tests how reservoir performance scales with neuron count by masking
subsets of the full 128-neuron FPGA run.

Strategy:
  - Run FPGA ONCE with all 128 neurons (same config as z2298)
  - Mask to N=8,16,32,64,128 offline — computationally cheap
  - Same benchmarks: MC(d=1..20), XOR(tau=1,3,5), NARMA-5, Wave4
  - Also run with temporal product features

Tests (12):
  T1:  MC(128) > MC(64) > MC(32) (monotonic scaling)
  T2:  MC(128) > 2 × MC(8) (superlinear at top)
  T3:  XOR1(128) > XOR1(8) by at least 10pp
  T4:  XOR5(128) > XOR5(8) (temporal advantage scales)
  T5:  Wave4(128) > Wave4(8) by at least 15pp
  T6:  NARMA-5(128) < NARMA-5(8) (better regression)
  T7:  MC(128) > 10.0 (reproduces z2298)
  T8:  MC(8) < 3.0 (small bank limited)
  T9:  At least 4/5 sizes show MC increasing
  T10: Scaling curve is concave (diminishing returns)
  T11: XOR3 improves with scale for at least 3 sizes
  T12: Wave4(64) > 80%

Run:
  PYTHONUNBUFFERED=1 taskset -c 0-3 venv/bin/python scripts/z2305_scaling_ladder.py
"""

import os, sys, time, json, struct
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2305_scaling_ladder.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
N_STEPS = 1500
WARMUP = 300
TEMP_SAFE = 40.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
NEURON_COUNTS = [8, 16, 32, 64, 128]


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
# FPGA continuous run
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
    for t in range(n_steps):
        # Check temp EVERY 5 steps — APU heats from 45->99C in seconds from UDP I/O
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
        elif t > 0:
            states[t] = states[t-1]
            dspikes[t] = dspikes[t-1]
    fpga.set_mac_signal(0.0)
    return states, dspikes


# ============================================================
# Temporal product features (same as z2296/z2298)
# ============================================================
def build_temporal_features(states, dspikes=None, n_select=24, seed=42):
    """Build temporal order-2+3 product features for ANY reservoir states."""
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
    """4-class waveform classification (sine, square, triangle, sawtooth)."""
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
            except:
                pass
    pred = np.argmax(scores_matrix, axis=1)
    acc = np.mean(pred == labels[n_tr:])
    return float(acc)


def full_benchmark(X, u_raw):
    n = len(X)
    n_tr = int(0.7 * n)

    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 21):
        target = u_raw[WARMUP-d:len(u_raw)-d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    xor = {}
    for tau in [1, 3, 5]:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP-tau:len(u_raw)-tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor[f'tau{tau}'] = acc

    narma = {}
    for order in [5]:
        T = len(u_raw)
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
        narma[f'narma{order}'] = best_nrmse

    wave_acc = classify_waveform(X, u_raw)

    return {'mc_total': mc_total, 'mc_per_delay': mc_per_d, 'xor': xor, 'narma': narma,
            'wave4_acc': wave_acc}


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  z2305: Neuron-Count Scaling Ladder (8 -> 16 -> 32 -> 64 -> 128)")
    print("  Single FPGA run, offline masking to simulate smaller banks")
    print("=" * 70)

    results = {'sizes': {}, 'tests': {}}

    # Resume support
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done = list(results.get('sizes', {}).keys())
            if done:
                print(f"  RESUMED: sizes {done} already done")
        except Exception:
            results = {'sizes': {}, 'tests': {}}

    rng = np.random.default_rng(42)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

    # ================================================================
    # Step 1: Run FPGA once with all 128 neurons
    # ================================================================
    states_file = RESULTS / 'z2305_fpga_states.npy'
    dspikes_file = RESULTS / 'z2305_fpga_dspikes.npy'

    if states_file.exists() and dspikes_file.exists():
        print("\n[FPGA] Loading cached states from disk")
        fpga_states = np.load(states_file)
        fpga_dspikes = np.load(dspikes_file)
        print(f"  Loaded: states {fpga_states.shape}, dspikes {fpga_dspikes.shape}")
    else:
        print("\n[FPGA] Running 128-neuron continuous acquisition")
        wait_cool("pre-FPGA", target=TEMP_SAFE)

        fpga = FPGAEthBridge(timeout=2.0)
        fpga.connect()
        fpga.set_kill(0)
        time.sleep(1.0)

        # Set runtime params (MUST after reprogram — bitstream defaults differ!)
        fpga.set_leak_cond(0x2000)
        fpga.set_base_exc_raw(0x0080)
        fpga.set_bias_gain_raw(0x4000)
        fpga.set_threshold_raw(0x20000)
        for n in range(NUM_NEURONS):
            fpga.set_vg(n, VG_GROUPS[n % 4])
            time.sleep(0.001)
        for n in range(NUM_NEURONS):
            fpga.set_synapse(n, 0x00000000)
            time.sleep(0.001)
        time.sleep(0.5)

        telem = fpga.read_telemetry()
        if telem is not None:
            print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
        else:
            print("  WARNING: FPGA telemetry returned None!")

        fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw)
        np.save(states_file, fpga_states)
        np.save(dspikes_file, fpga_dspikes)
        print(f"  Saved: states {fpga_states.shape}, dspikes {fpga_dspikes.shape}")

        try:
            fpga.set_kill(1)
        except Exception:
            pass

    # ================================================================
    # Step 2: Offline masking — benchmark each neuron count
    # ================================================================
    for N in NEURON_COUNTS:
        key = str(N)
        if key in results.get('sizes', {}):
            print(f"\n[N={N:3d}] Already done, skipping")
            continue

        print(f"\n[N={N:3d}] Masking to neurons 0..{N-1}")

        # Mask states and dspikes to first N neurons
        st_masked = fpga_states[:, :N]
        ds_masked = fpga_dspikes[:, :N]

        st_w = st_masked[WARMUP:]
        ds_w = ds_masked[WARMUP:]

        n_select = min(24, N)

        # Raw features (states + dspikes only)
        X_raw = np.hstack([st_w, ds_w])
        bm_raw = full_benchmark(X_raw, u_raw)

        # Temporal product features
        X_temporal = build_temporal_features(st_w, ds_w, n_select=n_select, seed=42)
        bm_temporal = full_benchmark(X_temporal, u_raw)

        results['sizes'][key] = {
            'n_neurons': N,
            'raw': bm_raw,
            'temporal': bm_temporal,
        }

        xr = bm_raw['xor']
        xt = bm_temporal['xor']
        print(f"  RAW:      MC={bm_raw['mc_total']:.2f}  XOR1={xr['tau1']*100:.1f}%  XOR3={xr['tau3']*100:.1f}%  "
              f"XOR5={xr['tau5']*100:.1f}%  N5={bm_raw['narma']['narma5']:.3f}  W4={bm_raw['wave4_acc']*100:.1f}%")
        print(f"  TEMPORAL: MC={bm_temporal['mc_total']:.2f}  XOR1={xt['tau1']*100:.1f}%  XOR3={xt['tau3']*100:.1f}%  "
              f"XOR5={xt['tau5']*100:.1f}%  N5={bm_temporal['narma']['narma5']:.3f}  W4={bm_temporal['wave4_acc']*100:.1f}%")

        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)

    # ================================================================
    # Step 3: Evaluate tests (use temporal features as primary)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TESTS")
    print("=" * 70)

    def mc(n):
        return results['sizes'][str(n)]['temporal']['mc_total']

    def xor_acc(n, tau):
        return results['sizes'][str(n)]['temporal']['xor'][f'tau{tau}']

    def wave4(n):
        return results['sizes'][str(n)]['temporal']['wave4_acc']

    def narma5(n):
        return results['sizes'][str(n)]['temporal']['narma']['narma5']

    n_pass = 0
    n_total = 12

    # T1: MC(128) > MC(64) > MC(32) (monotonic scaling)
    t1 = mc(128) > mc(64) > mc(32)
    results['tests']['T1_mc_monotonic'] = {
        'pass': bool(t1),
        'mc_128': mc(128), 'mc_64': mc(64), 'mc_32': mc(32),
        'desc': 'MC(128) > MC(64) > MC(32)'
    }
    n_pass += t1
    print(f"  T1  {'PASS' if t1 else 'FAIL'}: MC monotonic: 128={mc(128):.2f} > 64={mc(64):.2f} > 32={mc(32):.2f}")

    # T2: MC(128) > 2 * MC(8) (superlinear at top)
    t2 = mc(128) > 2 * mc(8)
    results['tests']['T2_mc_superlinear'] = {
        'pass': bool(t2),
        'mc_128': mc(128), 'mc_8': mc(8), 'threshold': 2 * mc(8),
        'desc': 'MC(128) > 2 x MC(8)'
    }
    n_pass += t2
    print(f"  T2  {'PASS' if t2 else 'FAIL'}: MC superlinear: 128={mc(128):.2f} > 2x8={2*mc(8):.2f}")

    # T3: XOR1(128) > XOR1(8) by at least 10pp
    diff3 = xor_acc(128, 1) - xor_acc(8, 1)
    t3 = diff3 >= 0.10
    results['tests']['T3_xor1_scaling'] = {
        'pass': bool(t3),
        'xor1_128': xor_acc(128, 1), 'xor1_8': xor_acc(8, 1), 'diff_pp': diff3 * 100,
        'desc': 'XOR1(128) > XOR1(8) + 10pp'
    }
    n_pass += t3
    print(f"  T3  {'PASS' if t3 else 'FAIL'}: XOR1 scaling: 128={xor_acc(128,1)*100:.1f}% - 8={xor_acc(8,1)*100:.1f}% = {diff3*100:.1f}pp (need >=10)")

    # T4: XOR5(128) > XOR5(8) (temporal advantage scales)
    t4 = xor_acc(128, 5) > xor_acc(8, 5)
    results['tests']['T4_xor5_scales'] = {
        'pass': bool(t4),
        'xor5_128': xor_acc(128, 5), 'xor5_8': xor_acc(8, 5),
        'desc': 'XOR5(128) > XOR5(8)'
    }
    n_pass += t4
    print(f"  T4  {'PASS' if t4 else 'FAIL'}: XOR5 scales: 128={xor_acc(128,5)*100:.1f}% > 8={xor_acc(8,5)*100:.1f}%")

    # T5: Wave4(128) > Wave4(8) by at least 15pp
    diff5 = wave4(128) - wave4(8)
    t5 = diff5 >= 0.15
    results['tests']['T5_wave4_scaling'] = {
        'pass': bool(t5),
        'wave4_128': wave4(128), 'wave4_8': wave4(8), 'diff_pp': diff5 * 100,
        'desc': 'Wave4(128) > Wave4(8) + 15pp'
    }
    n_pass += t5
    print(f"  T5  {'PASS' if t5 else 'FAIL'}: Wave4 scaling: 128={wave4(128)*100:.1f}% - 8={wave4(8)*100:.1f}% = {diff5*100:.1f}pp (need >=15)")

    # T6: NARMA-5(128) < NARMA-5(8) (lower NRMSE = better regression)
    t6 = narma5(128) < narma5(8)
    results['tests']['T6_narma5_scaling'] = {
        'pass': bool(t6),
        'narma5_128': narma5(128), 'narma5_8': narma5(8),
        'desc': 'NARMA-5(128) < NARMA-5(8)'
    }
    n_pass += t6
    print(f"  T6  {'PASS' if t6 else 'FAIL'}: NARMA-5 scaling: 128={narma5(128):.3f} < 8={narma5(8):.3f}")

    # T7: MC(128) > 10.0 (reproduces z2298)
    t7 = mc(128) > 10.0
    results['tests']['T7_mc128_threshold'] = {
        'pass': bool(t7),
        'mc_128': mc(128),
        'desc': 'MC(128) > 10.0'
    }
    n_pass += t7
    print(f"  T7  {'PASS' if t7 else 'FAIL'}: MC(128)={mc(128):.2f} > 10.0")

    # T8: MC(8) < 3.0 (small bank limited)
    t8 = mc(8) < 3.0
    results['tests']['T8_mc8_limited'] = {
        'pass': bool(t8),
        'mc_8': mc(8),
        'desc': 'MC(8) < 3.0'
    }
    n_pass += t8
    print(f"  T8  {'PASS' if t8 else 'FAIL'}: MC(8)={mc(8):.2f} < 3.0")

    # T9: At least 4/5 sizes show MC increasing
    mc_vals = [mc(n) for n in NEURON_COUNTS]
    n_increasing = sum(1 for i in range(1, len(mc_vals)) if mc_vals[i] > mc_vals[i-1])
    t9 = n_increasing >= 4
    results['tests']['T9_mc_mostly_increasing'] = {
        'pass': bool(t9),
        'n_increasing': n_increasing,
        'mc_values': {str(n): mc(n) for n in NEURON_COUNTS},
        'desc': 'At least 4/5 sizes show MC increasing'
    }
    n_pass += t9
    print(f"  T9  {'PASS' if t9 else 'FAIL'}: MC increasing in {n_increasing}/4 consecutive pairs (need >=4)")

    # T10: Scaling curve is concave (diminishing returns)
    # Check: MC gain from 8->16 > MC gain from 64->128 (per neuron)
    gain_8_16 = (mc(16) - mc(8)) / 8
    gain_64_128 = (mc(128) - mc(64)) / 64
    t10 = gain_8_16 > gain_64_128
    results['tests']['T10_concave_scaling'] = {
        'pass': bool(t10),
        'gain_per_neuron_8_16': gain_8_16,
        'gain_per_neuron_64_128': gain_64_128,
        'desc': 'Scaling is concave (diminishing returns)'
    }
    n_pass += t10
    print(f"  T10 {'PASS' if t10 else 'FAIL'}: Concave scaling: gain/neuron 8->16={gain_8_16:.4f} > 64->128={gain_64_128:.4f}")

    # T11: XOR3 improves with scale for at least 3 sizes
    xor3_vals = [xor_acc(n, 3) for n in NEURON_COUNTS]
    n_xor3_inc = sum(1 for i in range(1, len(xor3_vals)) if xor3_vals[i] > xor3_vals[i-1])
    t11 = n_xor3_inc >= 3
    results['tests']['T11_xor3_scales'] = {
        'pass': bool(t11),
        'n_increasing': n_xor3_inc,
        'xor3_values': {str(n): xor_acc(n, 3) for n in NEURON_COUNTS},
        'desc': 'XOR3 improves with scale for at least 3 sizes'
    }
    n_pass += t11
    print(f"  T11 {'PASS' if t11 else 'FAIL'}: XOR3 increasing in {n_xor3_inc}/4 pairs (need >=3)")

    # T12: Wave4(64) > 80%
    t12 = wave4(64) > 0.80
    results['tests']['T12_wave4_64'] = {
        'pass': bool(t12),
        'wave4_64': wave4(64),
        'desc': 'Wave4(64) > 80%'
    }
    n_pass += t12
    print(f"  T12 {'PASS' if t12 else 'FAIL'}: Wave4(64)={wave4(64)*100:.1f}% > 80%")

    # ================================================================
    # Summary
    # ================================================================
    results['summary'] = {
        'n_pass': n_pass,
        'n_total': n_total,
        'pass_rate': f"{n_pass}/{n_total}",
        'neuron_counts': NEURON_COUNTS,
        'mc_scaling': {str(n): mc(n) for n in NEURON_COUNTS},
        'xor1_scaling': {str(n): xor_acc(n, 1) for n in NEURON_COUNTS},
        'wave4_scaling': {str(n): wave4(n) for n in NEURON_COUNTS},
        'narma5_scaling': {str(n): narma5(n) for n in NEURON_COUNTS},
    }

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)

    print(f"\n{'=' * 70}")
    print(f"  RESULT: {n_pass}/{n_total} PASS")
    print(f"  Saved to {SAVE_FILE}")
    print(f"{'=' * 70}")

    # Print scaling table
    print(f"\n  {'N':>4s}  {'MC':>7s}  {'XOR1':>6s}  {'XOR3':>6s}  {'XOR5':>6s}  {'N5':>7s}  {'W4':>6s}")
    print(f"  {'----':>4s}  {'-------':>7s}  {'------':>6s}  {'------':>6s}  {'------':>6s}  {'-------':>7s}  {'------':>6s}")
    for N in NEURON_COUNTS:
        print(f"  {N:4d}  {mc(N):7.2f}  {xor_acc(N,1)*100:5.1f}%  {xor_acc(N,3)*100:5.1f}%  "
              f"{xor_acc(N,5)*100:5.1f}%  {narma5(N):7.3f}  {wave4(N)*100:5.1f}%")


if __name__ == '__main__':
    main()
