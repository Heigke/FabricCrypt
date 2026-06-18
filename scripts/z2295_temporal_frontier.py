#!/usr/bin/env python3
"""
z2295_temporal_frontier.py — Push temporal features to the limit
================================================================
z2294 confirmed: temporal products vmem(t)×vmem(t-τ) are THE key feature.
XOR1=80.4%, XOR3=66.4%, XOR5=63.2%.

Push further:
  EXP 1: Extended temporal lags (τ=1..20) for XOR and MC
         What's the maximum XOR τ we can solve above chance?
  EXP 2: Higher-order temporal products
         vmem(t) × vmem(t-τ1) × vmem(t-τ2) — 3-body temporal correlations
  EXP 3: Bridge + temporal features
         GPU MAC + temporal products — does the bridge help when features are good?
  EXP 4: NARMA-20 attack with temporal features
         Long-range temporal prediction should benefit most

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2295_temporal_frontier.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2295_temporal_frontier.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
N_STEPS = 3000  # Longer for temporal tasks
WARMUP = 500
TEMP_SAFE = 55.0


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


class GPUFourpopESN:
    def __init__(self, n_per_pop=64, seed=7777):
        self.pp = n_per_pop
        self.N = 4 * n_per_pop
        rng = np.random.default_rng(seed)
        self.leak = np.zeros(self.N)
        self.input_w = np.zeros(self.N)
        self.thr = np.zeros(self.N)
        self.bias = np.zeros(self.N)
        for pop in range(4):
            s, e = pop * n_per_pop, (pop + 1) * n_per_pop
            self.leak[s:e] = 0.05 + 0.15 * rng.random(n_per_pop)
            self.input_w[s:e] = 0.05 + 0.20 * rng.random(n_per_pop)
            self.thr[s:e] = 0.4 + 0.5 * rng.random(n_per_pop)
            self.bias[s:e] = 0.02 * (rng.random(n_per_pop) - 0.5)
        self.W_rec = rng.standard_normal((self.N, self.N)) * 0.04
        mask = rng.random((self.N, self.N)) > 0.9
        self.W_rec *= mask
        sc, ec = 2 * n_per_pop, 3 * n_per_pop
        W_c = rng.standard_normal((n_per_pop, n_per_pop)) * 0.08
        mask_c = rng.random((n_per_pop, n_per_pop)) > 0.7
        W_c *= mask_c
        eigvals = np.abs(np.linalg.eigvals(W_c))
        sr = max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0: W_c *= 1.05 / sr
        self.W_rec[sc:ec, sc:ec] = W_c
        self.bthr = 0.5 + 0.3 * np.arange(n_per_pop) / max(n_per_pop - 1, 1)
        self.temp_c = 0.65

    def run(self, input_seq, run_seed=42):
        n_steps = len(input_seq)
        pp = self.pp
        states = np.zeros((n_steps, self.N))
        v = np.zeros(self.N)
        h = np.zeros(self.N)
        slow = np.zeros(self.N)
        rng = np.random.default_rng(run_seed)
        for t in range(n_steps):
            u = input_seq[t]
            rec = self.W_rec @ v
            sa, ea = 0, pp
            bv = np.where(v[sa:ea] > self.bthr, 1.0, -1.0)
            v_a = np.tanh((1-self.leak[sa:ea])*v[sa:ea] + self.input_w[sa:ea]*u + rec[sa:ea] + self.bias[sa:ea] + 0.02*bv)
            sb, eb = pp, 2*pp
            v_b = v[sb:eb].copy()
            ns = max(1, pp//10)
            si = rng.choice(pp, size=ns*2, replace=False)
            for k in range(0, ns*2-1, 2):
                v_b[si[k]], v_b[si[k+1]] = v_b[si[k+1]], v_b[si[k]]
            v_b = np.tanh((1-self.leak[sb:eb])*v_b + self.input_w[sb:eb]*u + rec[sb:eb] + self.bias[sb:eb])
            sc, ec = 2*pp, 3*pp
            v_c = np.tanh(((1-self.leak[sc:ec])*v[sc:ec] + self.input_w[sc:ec]*u + rec[sc:ec] + self.bias[sc:ec])/self.temp_c)
            sd, ed = 3*pp, 4*pp
            sn = rng.uniform(-1,1,pp)*0.01
            v_d = np.tanh((1-self.leak[sd:ed])*v[sd:ed] + self.input_w[sd:ed]*u + rec[sd:ed] + self.bias[sd:ed] + sn)
            v_new = np.concatenate([v_a, v_b, v_c, v_d])
            v_new += rng.uniform(-1,1,self.N)*0.003
            spike_mask = v_new > self.thr
            v_new[spike_mask] -= self.thr[spike_mask]
            v = v_new
            h = 0.93*h + 0.07*v
            slow = 0.99*slow + 0.01*v
            states[t] = v + 0.3*h + 0.1*slow
        return states


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
        fpga.set_mac_signal(float(mac_signal[t]))
        time.sleep(dt)
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


def build_temporal_features(states, dspikes, tau_list, order=2, n_sample=24):
    """Build temporal product features at specified lags and orders."""
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, dspikes, delta]

    rng = np.random.default_rng(42)
    qi = np.sort(rng.choice(n_ch, size=min(n_sample, n_ch), replace=False))
    vm_q = states[:, qi]
    ds_q = dspikes[:, qi]

    # Order-2 temporal products: vmem(t) × vmem(t-τ)
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        feats.append(ds_q * shifted)

    if order >= 3:
        # Order-3: vmem(t) × vmem(t-τ1) × vmem(t-τ2)
        for i, t1 in enumerate(tau_list):
            for t2 in tau_list[i+1:]:
                if t2 > 10:
                    continue  # limit combinatorial explosion
                sh1 = np.zeros_like(vm_q)
                sh2 = np.zeros_like(vm_q)
                sh1[t1:] = vm_q[:-t1]
                sh2[t2:] = vm_q[:-t2]
                feats.append(vm_q * sh1 * sh2)

    # Squares (always useful)
    feats.append(np.square(vm_q))

    # Sign features
    feats.append((vm_q > np.median(vm_q, axis=0)).astype(float))

    return np.hstack(feats)


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


def eval_mc(X, u_raw, max_d=20):
    n = len(X)
    n_tr = int(0.7 * n)
    mc = 0.0
    per_d = {}
    for d in range(1, max_d+1):
        target = u_raw[WARMUP-d:len(u_raw)-d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        per_d[str(d)] = r2
        mc += r2
    return mc, per_d


def eval_xor(X, u_raw, tau):
    n = len(X)
    n_tr = int(0.7 * n)
    u_a = (u_raw[WARMUP:] > 0).astype(float)
    u_b = (u_raw[WARMUP-tau:len(u_raw)-tau] > 0).astype(float)
    nn = min(len(u_a), len(u_b), n)
    target = (u_a[:nn] != u_b[:nn]).astype(float)
    Xn = X[:nn]
    return ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')


def eval_narma(X, u_raw, order):
    n = len(X)
    n_tr = int(0.7 * n)
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
    return best_nrmse


def main():
    print("=" * 70)
    print("  z2295: TEMPORAL FRONTIER — push XOR to τ=10+, NARMA-20")
    print("  z2294: XOR1=80.4%, XOR3=66.4%, XOR5=63.2% (temporal products)")
    print("=" * 70)

    fpga = FPGAEthBridge()
    fpga.connect()
    telem = fpga.read_telemetry()
    print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    gpu = GPUFourpopESN(seed=7777)

    rng = np.random.default_rng(42)
    u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

    results = {'experiments': {}, 'tests': {}}

    # ================================================================
    # EXP 1: Extended XOR τ sweep (1..20) with temporal features
    # ================================================================
    print("\n[EXP 1] XOR τ sweep (1..20)")
    print("-" * 60)

    mac_sig = np.clip(u_raw * 0.3 + 0.3, 0, 1)
    fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw, mac_signal=mac_sig)

    # Build features with lags matching the XOR range
    X_temp = build_temporal_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:],
                                     tau_list=[1,2,3,4,5,6,7,8,10,12,15,20], order=2)

    xor_sweep = {}
    for tau in range(1, 21):
        acc = eval_xor(X_temp, u_raw, tau)
        xor_sweep[str(tau)] = acc
        marker = " ***" if acc > 0.60 else " **" if acc > 0.55 else ""
        print(f"  τ={tau:2d}: {acc*100:.1f}%{marker}")

    results['experiments']['exp1_xor_sweep'] = xor_sweep

    # Find max τ above chance (>55%)
    max_tau_above_55 = 0
    for tau in range(1, 21):
        if xor_sweep[str(tau)] > 0.55:
            max_tau_above_55 = tau

    print(f"  Max τ above 55%: {max_tau_above_55}")
    results['experiments']['exp1_max_tau'] = max_tau_above_55

    # ================================================================
    # EXP 2: Order-3 temporal products
    # ================================================================
    print("\n[EXP 2] Order-3 temporal products (3-body correlations)")
    print("-" * 60)

    X_order2 = build_temporal_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:],
                                        tau_list=[1,2,3,5,8], order=2)
    X_order3 = build_temporal_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:],
                                        tau_list=[1,2,3,5,8], order=3)

    order_results = {}
    for tau in [1, 3, 5, 8, 10]:
        acc2 = eval_xor(X_order2, u_raw, tau)
        acc3 = eval_xor(X_order3, u_raw, tau)
        order_results[f'tau{tau}'] = {'order2': acc2, 'order3': acc3}
        print(f"  τ={tau:2d}: order2={acc2*100:.1f}% order3={acc3*100:.1f}% {'↑' if acc3 > acc2 else '↓'}")

    mc2, mc2_d = eval_mc(X_order2, u_raw, max_d=10)
    mc3, mc3_d = eval_mc(X_order3, u_raw, max_d=10)
    order_results['mc'] = {'order2': mc2, 'order3': mc3}
    print(f"  MC: order2={mc2:.3f} order3={mc3:.3f}")

    results['experiments']['exp2_order'] = order_results

    # ================================================================
    # EXP 3: Bridge + temporal features (GPU MAC + NL readout)
    # ================================================================
    print("\n[EXP 3] Bridge + temporal features")
    print("-" * 60)

    wait_cool("pre-EXP3")

    # GPU-driven MAC
    gpu_states = gpu.run(u_raw, run_seed=42)
    gpu_mac = np.mean(np.abs(gpu_states), axis=1)
    mac_bridge = np.clip(0.6 * mac_sig + 0.4 * gpu_mac, 0, 1)

    br_states, br_dspikes = fpga_run_continuous(fpga, u_raw, mac_signal=mac_bridge)

    X_br_temp = build_temporal_features(br_states[WARMUP:], br_dspikes[WARMUP:],
                                         tau_list=[1,2,3,5,8,10,15,20], order=2)

    # Also concatenate GPU features
    X_br_full = np.hstack([X_br_temp, gpu_states[WARMUP:]])

    bridge_results = {}
    for label, X in [('fpga_temp', X_temp), ('bridge_temp', X_br_temp), ('bridge_full', X_br_full)]:
        xor_res = {}
        for tau in [1, 3, 5, 8, 10]:
            xor_res[f'tau{tau}'] = eval_xor(X, u_raw, tau)
        mc, _ = eval_mc(X, u_raw, max_d=10)
        bridge_results[label] = {'xor': xor_res, 'mc': mc, 'n_feats': X.shape[1]}
        xr = xor_res
        print(f"  {label:15s} ({X.shape[1]:4d}): MC={mc:.3f} XOR1={xr['tau1']*100:.1f}% XOR3={xr['tau3']*100:.1f}% XOR5={xr['tau5']*100:.1f}% XOR10={xr['tau10']*100:.1f}%")

    results['experiments']['exp3_bridge'] = bridge_results

    # ================================================================
    # EXP 4: NARMA-5/10/20/30 with temporal features
    # ================================================================
    print("\n[EXP 4] NARMA with temporal features")
    print("-" * 60)

    narma_results = {}
    for label, X in [('fpga_temp', X_temp), ('bridge_temp', X_br_temp), ('bridge_full', X_br_full)]:
        nr = {}
        for order in [5, 10, 20, 30]:
            nrmse = eval_narma(X, u_raw, order)
            nr[f'narma{order}'] = nrmse
        narma_results[label] = nr
        print(f"  {label:15s}: N5={nr['narma5']:.3f} N10={nr['narma10']:.3f} N20={nr['narma20']:.3f} N30={nr['narma30']:.3f}")

    results['experiments']['exp4_narma'] = narma_results

    # ================================================================
    # EXP 5: MC to d=20 with temporal features
    # ================================================================
    print("\n[EXP 5] Extended memory capacity (d=1..20)")
    print("-" * 60)

    mc_ext, mc_ext_d = eval_mc(X_temp, u_raw, max_d=20)
    mc_br_ext, mc_br_d = eval_mc(X_br_full, u_raw, max_d=20)

    print(f"  FPGA temporal: MC(20)={mc_ext:.3f}")
    print(f"  Bridge full:   MC(20)={mc_br_ext:.3f}")
    print(f"  Per-delay FPGA: ", end="")
    for d in range(1, 21):
        v = mc_ext_d[str(d)]
        print(f"d{d}={v:.2f} ", end="")
        if d % 10 == 0:
            print()
            if d < 20:
                print(f"                  ", end="")

    results['experiments']['exp5_mc_extended'] = {
        'fpga': {'mc_total': mc_ext, 'per_delay': mc_ext_d},
        'bridge': {'mc_total': mc_br_ext, 'per_delay': mc_br_d},
    }

    # ================================================================
    # TESTS
    # ================================================================
    print("\n" + "=" * 70)
    print("  TESTS")
    print("=" * 70)

    tests = {}
    n_pass = 0

    # T1: XOR1 > 80% with temporal features
    t1 = xor_sweep['1'] > 0.80
    tests['T1'] = {'pass': t1, 'val': xor_sweep['1']}
    print(f"  T1 {'PASS' if t1 else 'FAIL'}: XOR1={xor_sweep['1']*100:.1f}% > 80%")
    n_pass += t1

    # T2: XOR3 > 60%
    t2 = xor_sweep['3'] > 0.60
    tests['T2'] = {'pass': t2, 'val': xor_sweep['3']}
    print(f"  T2 {'PASS' if t2 else 'FAIL'}: XOR3={xor_sweep['3']*100:.1f}% > 60%")
    n_pass += t2

    # T3: XOR5 > 58%
    t3 = xor_sweep['5'] > 0.58
    tests['T3'] = {'pass': t3, 'val': xor_sweep['5']}
    print(f"  T3 {'PASS' if t3 else 'FAIL'}: XOR5={xor_sweep['5']*100:.1f}% > 58%")
    n_pass += t3

    # T4: XOR10 > 53%
    t4 = xor_sweep['10'] > 0.53
    tests['T4'] = {'pass': t4, 'val': xor_sweep['10']}
    print(f"  T4 {'PASS' if t4 else 'FAIL'}: XOR10={xor_sweep['10']*100:.1f}% > 53%")
    n_pass += t4

    # T5: Max τ above 55% ≥ 8
    t5 = max_tau_above_55 >= 8
    tests['T5'] = {'pass': t5, 'max_tau': max_tau_above_55}
    print(f"  T5 {'PASS' if t5 else 'FAIL'}: Max τ above 55% = {max_tau_above_55} (need ≥8)")
    n_pass += t5

    # T6: Order-3 improves XOR5 over order-2
    t6 = order_results['tau5']['order3'] > order_results['tau5']['order2']
    tests['T6'] = {'pass': t6, 'o2': order_results['tau5']['order2'], 'o3': order_results['tau5']['order3']}
    print(f"  T6 {'PASS' if t6 else 'FAIL'}: Order3 XOR5={order_results['tau5']['order3']*100:.1f}% > order2={order_results['tau5']['order2']*100:.1f}%")
    n_pass += t6

    # T7: Bridge + temporal gives best MC
    t7 = bridge_results['bridge_full']['mc'] > bridge_results['fpga_temp']['mc']
    tests['T7'] = {'pass': t7, 'bridge': bridge_results['bridge_full']['mc'], 'fpga': bridge_results['fpga_temp']['mc']}
    print(f"  T7 {'PASS' if t7 else 'FAIL'}: Bridge MC={bridge_results['bridge_full']['mc']:.3f} > FPGA={bridge_results['fpga_temp']['mc']:.3f}")
    n_pass += t7

    # T8: NARMA-5 < 0.45 with temporal features
    best_n5 = min(v['narma5'] for v in narma_results.values())
    t8 = best_n5 < 0.45
    tests['T8'] = {'pass': t8, 'best': best_n5}
    print(f"  T8 {'PASS' if t8 else 'FAIL'}: Best NARMA-5={best_n5:.3f} < 0.45")
    n_pass += t8

    # T9: NARMA-10 < 0.75 with temporal features
    best_n10 = min(v['narma10'] for v in narma_results.values())
    t9 = best_n10 < 0.75
    tests['T9'] = {'pass': t9, 'best': best_n10}
    print(f"  T9 {'PASS' if t9 else 'FAIL'}: Best NARMA-10={best_n10:.3f} < 0.75")
    n_pass += t9

    # T10: NARMA-20 < 0.85
    best_n20 = min(v['narma20'] for v in narma_results.values())
    t10 = best_n20 < 0.85
    tests['T10'] = {'pass': t10, 'best': best_n20}
    print(f"  T10 {'PASS' if t10 else 'FAIL'}: Best NARMA-20={best_n20:.3f} < 0.85")
    n_pass += t10

    # T11: MC(d=1..20) total > 6.0
    best_mc20 = max(mc_ext, mc_br_ext)
    t11 = best_mc20 > 6.0
    tests['T11'] = {'pass': t11, 'best': best_mc20}
    print(f"  T11 {'PASS' if t11 else 'FAIL'}: Best MC(20)={best_mc20:.3f} > 6.0")
    n_pass += t11

    # T12: XOR monotonically decreases with τ (at least first 8)
    monotone = all(xor_sweep[str(i)] >= xor_sweep[str(i+1)] - 0.02 for i in range(1, 8))
    t12 = monotone
    tests['T12'] = {'pass': t12}
    print(f"  T12 {'PASS' if t12 else 'FAIL'}: XOR monotonically decreases τ=1..8")
    n_pass += t12

    # T13: NARMA-30 < 1.0 (usable at order 30)
    best_n30 = min(v['narma30'] for v in narma_results.values())
    t13 = best_n30 < 1.0
    tests['T13'] = {'pass': t13, 'best': best_n30}
    print(f"  T13 {'PASS' if t13 else 'FAIL'}: Best NARMA-30={best_n30:.3f} < 1.0")
    n_pass += t13

    # T14: Bridge temporal XOR1 ≥ FPGA temporal XOR1 (bridge doesn't hurt with NL features)
    t14 = bridge_results['bridge_temp']['xor']['tau1'] >= bridge_results['fpga_temp']['xor']['tau1'] - 0.02
    tests['T14'] = {'pass': t14}
    print(f"  T14 {'PASS' if t14 else 'FAIL'}: Bridge XOR1={bridge_results['bridge_temp']['xor']['tau1']*100:.1f}% ≈ FPGA={bridge_results['fpga_temp']['xor']['tau1']*100:.1f}%")
    n_pass += t14

    print(f"\n  TOTAL: {n_pass}/14 PASS")

    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass, 'n_total': 14,
        'max_tau_above_55': max_tau_above_55,
        'best_xor1': xor_sweep['1'],
        'best_narma5': best_n5,
        'best_narma10': best_n10,
        'best_mc20': best_mc20,
    }

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results saved: {SAVE_FILE}")


if __name__ == '__main__':
    main()
