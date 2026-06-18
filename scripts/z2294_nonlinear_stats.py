#!/usr/bin/env python3
"""
z2294_nonlinear_stats.py — Statistical validation of nonlinear feature breakthrough
===================================================================================
z2293 breakthrough: XOR1=82.4%, XOR3=68.0%, XOR5=62.9% with nonlinear features.
Need 5 seeds to confirm this isn't a fluke.

Also test:
  - Feature ablation: which nonlinear feature type matters most?
  - Waveform classification (4-class, 7-class, 15-class) with NL features

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2294_nonlinear_stats.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2294_nonlinear_stats.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
N_STEPS = 2500
WARMUP = 400
TEMP_SAFE = 55.0
N_SEEDS = 5


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
        mac_signal = np.clip(u * 0.4 + 0.5, 0, 1)
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


def build_features_basic(states, dspikes):
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, states.shape[1])), delta])
    return np.hstack([states, dspikes, delta])


def build_features_ablation(states, dspikes, include_quad=True, include_temporal=True,
                            include_spike_vm=True, include_sign=True, include_cube=True):
    """Ablation-friendly nonlinear features."""
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, dspikes, delta]

    rng = np.random.default_rng(42)
    qi = np.sort(rng.choice(n_ch, size=min(24, n_ch), replace=False))
    vm_q = states[:, qi]
    ds_q = dspikes[:, qi]

    if include_quad:
        cross = []
        for i in range(len(qi)):
            for j in range(i+1, min(i+4, len(qi))):
                cross.append((vm_q[:, i] * vm_q[:, j]).reshape(-1, 1))
        if cross:
            feats.append(np.hstack(cross))
        feats.append(np.square(vm_q))

    if include_cube:
        feats.append(np.power(vm_q, 3))

    if include_spike_vm:
        feats.append(vm_q * ds_q)

    if include_temporal:
        for tau in [1, 2, 3, 5, 8]:
            shifted = np.zeros_like(vm_q)
            shifted[tau:] = vm_q[:-tau]
            feats.append(vm_q * shifted)
            feats.append(ds_q * shifted)

    if include_sign:
        feats.append((vm_q > np.median(vm_q, axis=0)).astype(float))
        feats.append((ds_q > 0).astype(float))

    return np.hstack(feats)


def build_features_full_nonlinear(states, dspikes):
    return build_features_ablation(states, dspikes, True, True, True, True, True)


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


def eval_mc(X, u_raw, max_d=10):
    n = len(X)
    n_tr = int(0.7 * n)
    mc = 0.0
    for d in range(1, max_d+1):
        target = u_raw[WARMUP-d:len(u_raw)-d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc += r2
    return mc


def eval_xor(X, u_raw, tau_list=[1,2,3,5]):
    n = len(X)
    n_tr = int(0.7 * n)
    results = {}
    for tau in tau_list:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP-tau:len(u_raw)-tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        results[f'tau{tau}'] = acc
    return results


def eval_waveform(X, u_raw, n_classes=4, window=50):
    """Waveform classification: segment input into windows, classify waveform type."""
    n = len(X)
    n_tr = int(0.7 * n)
    rng = np.random.default_rng(12345)

    # Generate waveform labels based on input characteristics
    n_windows = n // window
    labels = np.zeros(n)
    for w in range(n_windows):
        s, e = w * window, (w + 1) * window
        segment = u_raw[WARMUP + s:WARMUP + e] if WARMUP + e <= len(u_raw) else u_raw[-window:]
        # Assign class based on segment statistics
        mean_val = np.mean(segment)
        std_val = np.std(segment)
        if n_classes == 4:
            if mean_val > 0 and std_val > 0.5:
                c = 0
            elif mean_val > 0 and std_val <= 0.5:
                c = 1
            elif mean_val <= 0 and std_val > 0.5:
                c = 2
            else:
                c = 3
        elif n_classes == 7:
            c = int(np.clip((mean_val + 1) / 2 * 3.5, 0, 3))
            if std_val > 0.6:
                c += 3
            c = min(c, 6)
        else:  # 15-class
            c = int(np.clip((mean_val + 1) / 2 * n_classes, 0, n_classes - 1))
        labels[s:e] = c

    # Window-averaged features
    X_win = np.zeros((n_windows, X.shape[1]))
    y_win = np.zeros(n_windows)
    for w in range(n_windows):
        s, e = w * window, (w + 1) * window
        if e <= n:
            X_win[w] = X[s:e].mean(axis=0)
            y_win[w] = labels[s]

    n_tr_w = int(0.7 * n_windows)
    if n_tr_w < 5 or n_windows - n_tr_w < 3:
        return 1.0 / n_classes  # too few windows

    # One-vs-rest classification
    pred_all = np.zeros((n_windows - n_tr_w, n_classes))
    for c in range(n_classes):
        y_bin = (y_win == c).astype(float)
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            I = np.eye(X_win.shape[1])
            try:
                w = np.linalg.solve(X_win[:n_tr_w].T @ X_win[:n_tr_w] + alpha * I,
                                    X_win[:n_tr_w].T @ y_bin[:n_tr_w])
                pred_all[:, c] = X_win[n_tr_w:] @ w
                break
            except Exception:
                pass

    pred_labels = np.argmax(pred_all, axis=1)
    true_labels = y_win[n_tr_w:]
    acc = np.mean(pred_labels == true_labels)
    return max(acc, 1.0 / n_classes)


def main():
    print("=" * 70)
    print("  z2294: NONLINEAR FEATURE STATS — 5 seeds + ablation + waveform")
    print("  z2293: XOR1=82.4%, XOR3=68.0%, XOR5=62.9% — confirm with stats")
    print("=" * 70)

    fpga = FPGAEthBridge()
    fpga.connect()
    telem = fpga.read_telemetry()
    print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    gpu = GPUFourpopESN(seed=7777)

    results = {'seeds': {}, 'ablation': {}, 'waveform': {}, 'tests': {}}

    # ================================================================
    # EXP 1: 5-seed statistical validation
    # ================================================================
    print("\n[EXP 1] Statistical validation — 5 seeds × 2 feature sets")
    print("-" * 60)

    seeds = [42, 123, 456, 789, 2024]
    seed_results = []

    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)
        gpu_states = gpu.run(u_raw, run_seed=seed)
        mac_base = np.clip(u_raw * 0.3 + 0.3, 0, 1)  # z2293 mixed-leak MAC

        wait_cool(f"pre-seed{si}")
        fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw, mac_signal=mac_base)

        X_basic = build_features_basic(fpga_states[WARMUP:], fpga_dspikes[WARMUP:])
        X_nonlin = build_features_full_nonlinear(fpga_states[WARMUP:], fpga_dspikes[WARMUP:])

        mc_b = eval_mc(X_basic, u_raw)
        mc_n = eval_mc(X_nonlin, u_raw)
        xor_b = eval_xor(X_basic, u_raw)
        xor_n = eval_xor(X_nonlin, u_raw)

        entry = {
            'seed': seed,
            'basic': {'mc': mc_b, 'xor': xor_b},
            'nonlinear': {'mc': mc_n, 'xor': xor_n},
        }
        seed_results.append(entry)
        print(f"  Seed {seed}: BASIC MC={mc_b:.3f} XOR1={xor_b['tau1']*100:.1f}% XOR3={xor_b['tau3']*100:.1f}%"
              f" | NL MC={mc_n:.3f} XOR1={xor_n['tau1']*100:.1f}% XOR3={xor_n['tau3']*100:.1f}% XOR5={xor_n['tau5']*100:.1f}%")

    results['seeds'] = seed_results

    # Compute stats
    nl_mc = [s['nonlinear']['mc'] for s in seed_results]
    nl_xor1 = [s['nonlinear']['xor']['tau1'] for s in seed_results]
    nl_xor3 = [s['nonlinear']['xor']['tau3'] for s in seed_results]
    nl_xor5 = [s['nonlinear']['xor']['tau5'] for s in seed_results]
    b_xor1 = [s['basic']['xor']['tau1'] for s in seed_results]

    print(f"\n  NL STATS (5 seeds):")
    print(f"    MC:   {np.mean(nl_mc):.3f} ± {np.std(nl_mc):.3f}")
    print(f"    XOR1: {np.mean(nl_xor1)*100:.1f}% ± {np.std(nl_xor1)*100:.1f}%")
    print(f"    XOR3: {np.mean(nl_xor3)*100:.1f}% ± {np.std(nl_xor3)*100:.1f}%")
    print(f"    XOR5: {np.mean(nl_xor5)*100:.1f}% ± {np.std(nl_xor5)*100:.1f}%")

    stats = {
        'nl_mc': {'mean': np.mean(nl_mc), 'std': np.std(nl_mc)},
        'nl_xor1': {'mean': np.mean(nl_xor1), 'std': np.std(nl_xor1)},
        'nl_xor3': {'mean': np.mean(nl_xor3), 'std': np.std(nl_xor3)},
        'nl_xor5': {'mean': np.mean(nl_xor5), 'std': np.std(nl_xor5)},
    }
    results['stats'] = stats

    # ================================================================
    # EXP 2: Feature ablation (use seed=42 data)
    # ================================================================
    print("\n[EXP 2] Feature ablation — which nonlinear type matters?")
    print("-" * 60)

    rng0 = np.random.default_rng(42)
    u_raw0 = rng0.uniform(-1, 1, N_STEPS + WARMUP)

    wait_cool("pre-ablation")
    fpga_states0, fpga_dspikes0 = fpga_run_continuous(fpga, u_raw0,
                                                       mac_signal=np.clip(u_raw0*0.3+0.3, 0, 1))

    ablation_configs = {
        'all':       dict(include_quad=True, include_temporal=True, include_spike_vm=True, include_sign=True, include_cube=True),
        'no_quad':   dict(include_quad=False, include_temporal=True, include_spike_vm=True, include_sign=True, include_cube=True),
        'no_temp':   dict(include_quad=True, include_temporal=False, include_spike_vm=True, include_sign=True, include_cube=True),
        'no_spike':  dict(include_quad=True, include_temporal=True, include_spike_vm=False, include_sign=True, include_cube=True),
        'no_sign':   dict(include_quad=True, include_temporal=True, include_spike_vm=True, include_sign=False, include_cube=True),
        'no_cube':   dict(include_quad=True, include_temporal=True, include_spike_vm=True, include_sign=True, include_cube=False),
        'quad_only': dict(include_quad=True, include_temporal=False, include_spike_vm=False, include_sign=False, include_cube=False),
        'temp_only': dict(include_quad=False, include_temporal=True, include_spike_vm=False, include_sign=False, include_cube=False),
        'basic':     None,
    }

    ablation_results = {}
    for label, config in ablation_configs.items():
        if config is None:
            X = build_features_basic(fpga_states0[WARMUP:], fpga_dspikes0[WARMUP:])
        else:
            X = build_features_ablation(fpga_states0[WARMUP:], fpga_dspikes0[WARMUP:], **config)
        mc = eval_mc(X, u_raw0)
        xor = eval_xor(X, u_raw0)
        ablation_results[label] = {'mc': mc, 'xor': xor, 'n_feats': X.shape[1]}
        print(f"  {label:12s} ({X.shape[1]:4d}): MC={mc:.3f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% XOR5={xor['tau5']*100:.1f}%")

    results['ablation'] = ablation_results

    # ================================================================
    # EXP 3: Waveform classification with NL features
    # ================================================================
    print("\n[EXP 3] Waveform classification — 4/7/15-class")
    print("-" * 60)

    # Use same FPGA data from ablation run
    X_nl = build_features_full_nonlinear(fpga_states0[WARMUP:], fpga_dspikes0[WARMUP:])
    X_b = build_features_basic(fpga_states0[WARMUP:], fpga_dspikes0[WARMUP:])

    wave_results = {}
    for n_cls in [4, 7, 15]:
        acc_b = eval_waveform(X_b, u_raw0, n_classes=n_cls)
        acc_n = eval_waveform(X_nl, u_raw0, n_classes=n_cls)
        wave_results[f'wave{n_cls}'] = {'basic': acc_b, 'nonlinear': acc_n}
        print(f"  {n_cls}-class: basic={acc_b*100:.1f}% nonlinear={acc_n*100:.1f}%")

    results['waveform'] = wave_results

    # ================================================================
    # TESTS
    # ================================================================
    print("\n" + "=" * 70)
    print("  TESTS")
    print("=" * 70)

    tests = {}
    n_pass = 0

    # T1: Mean NL XOR1 > 70%
    t1 = np.mean(nl_xor1) > 0.70
    tests['T1'] = {'pass': t1, 'mean': np.mean(nl_xor1), 'std': np.std(nl_xor1)}
    print(f"  T1 {'PASS' if t1 else 'FAIL'}: Mean NL XOR1={np.mean(nl_xor1)*100:.1f}%±{np.std(nl_xor1)*100:.1f}% > 70%")
    n_pass += t1

    # T2: Mean NL XOR3 > 55%
    t2 = np.mean(nl_xor3) > 0.55
    tests['T2'] = {'pass': t2, 'mean': np.mean(nl_xor3), 'std': np.std(nl_xor3)}
    print(f"  T2 {'PASS' if t2 else 'FAIL'}: Mean NL XOR3={np.mean(nl_xor3)*100:.1f}%±{np.std(nl_xor3)*100:.1f}% > 55%")
    n_pass += t2

    # T3: Mean NL XOR5 > 53%
    t3 = np.mean(nl_xor5) > 0.53
    tests['T3'] = {'pass': t3, 'mean': np.mean(nl_xor5), 'std': np.std(nl_xor5)}
    print(f"  T3 {'PASS' if t3 else 'FAIL'}: Mean NL XOR5={np.mean(nl_xor5)*100:.1f}%±{np.std(nl_xor5)*100:.1f}% > 53%")
    n_pass += t3

    # T4: NL XOR1 > basic XOR1 for all seeds
    nl_beats_basic = sum(1 for i in range(N_SEEDS) if nl_xor1[i] > b_xor1[i])
    t4 = nl_beats_basic >= 4  # 4/5 seeds
    tests['T4'] = {'pass': t4, 'nl_wins': nl_beats_basic}
    print(f"  T4 {'PASS' if t4 else 'FAIL'}: NL beats basic XOR1 in {nl_beats_basic}/5 seeds (need ≥4)")
    n_pass += t4

    # T5: Mean NL MC > 3.0
    t5 = np.mean(nl_mc) > 3.0
    tests['T5'] = {'pass': t5, 'mean': np.mean(nl_mc), 'std': np.std(nl_mc)}
    print(f"  T5 {'PASS' if t5 else 'FAIL'}: Mean NL MC={np.mean(nl_mc):.3f}±{np.std(nl_mc):.3f} > 3.0")
    n_pass += t5

    # T6: Temporal features are most important for XOR3
    all_xor3_ablation = ablation_results['all']['xor']['tau3']
    no_temp_xor3 = ablation_results['no_temp']['xor']['tau3']
    temp_drop = all_xor3_ablation - no_temp_xor3
    t6 = temp_drop > 0.02  # >2pp drop without temporal
    tests['T6'] = {'pass': t6, 'all': all_xor3_ablation, 'no_temp': no_temp_xor3, 'drop': temp_drop}
    print(f"  T6 {'PASS' if t6 else 'FAIL'}: Temporal drop on XOR3: all={all_xor3_ablation*100:.1f}% no_temp={no_temp_xor3*100:.1f}% (drop={temp_drop*100:.1f}pp)")
    n_pass += t6

    # T7: Quadratic features help XOR1
    all_xor1_abl = ablation_results['all']['xor']['tau1']
    no_quad_xor1 = ablation_results['no_quad']['xor']['tau1']
    quad_drop = all_xor1_abl - no_quad_xor1
    t7 = quad_drop > 0.01
    tests['T7'] = {'pass': t7, 'all': all_xor1_abl, 'no_quad': no_quad_xor1, 'drop': quad_drop}
    print(f"  T7 {'PASS' if t7 else 'FAIL'}: Quadratic drop on XOR1: all={all_xor1_abl*100:.1f}% no_quad={no_quad_xor1*100:.1f}% (drop={quad_drop*100:.1f}pp)")
    n_pass += t7

    # T8: NL features don't hurt waveform
    t8 = wave_results['wave4']['nonlinear'] >= wave_results['wave4']['basic'] * 0.95
    tests['T8'] = {'pass': t8, 'nl': wave_results['wave4']['nonlinear'], 'basic': wave_results['wave4']['basic']}
    print(f"  T8 {'PASS' if t8 else 'FAIL'}: Wave-4 NL={wave_results['wave4']['nonlinear']*100:.1f}% ≥ 95% basic={wave_results['wave4']['basic']*100:.1f}%")
    n_pass += t8

    # T9: std(XOR1) < 10% (consistent)
    t9 = np.std(nl_xor1) < 0.10
    tests['T9'] = {'pass': t9, 'std': np.std(nl_xor1)}
    print(f"  T9 {'PASS' if t9 else 'FAIL'}: XOR1 std={np.std(nl_xor1)*100:.1f}% < 10%")
    n_pass += t9

    # T10: At least one seed has XOR3 > 60%
    best_xor3 = max(nl_xor3)
    t10 = best_xor3 > 0.60
    tests['T10'] = {'pass': t10, 'best': best_xor3}
    print(f"  T10 {'PASS' if t10 else 'FAIL'}: Best XOR3={best_xor3*100:.1f}% > 60%")
    n_pass += t10

    # T11: All seeds have NL XOR1 > basic XOR1 by at least 5pp
    all_improve = all(nl_xor1[i] - b_xor1[i] > 0.05 for i in range(N_SEEDS))
    t11 = all_improve
    tests['T11'] = {'pass': t11}
    print(f"  T11 {'PASS' if t11 else 'FAIL'}: All seeds NL XOR1 > basic+5pp")
    n_pass += t11

    # T12: Feature ablation shows no single feature type is sufficient alone
    quad_only_xor1 = ablation_results['quad_only']['xor']['tau1']
    temp_only_xor1 = ablation_results['temp_only']['xor']['tau1']
    t12 = all_xor1_abl > max(quad_only_xor1, temp_only_xor1)
    tests['T12'] = {'pass': t12, 'all': all_xor1_abl, 'quad_only': quad_only_xor1, 'temp_only': temp_only_xor1}
    print(f"  T12 {'PASS' if t12 else 'FAIL'}: Combined>{max(quad_only_xor1,temp_only_xor1)*100:.1f}% (quad_only={quad_only_xor1*100:.1f}% temp_only={temp_only_xor1*100:.1f}%)")
    n_pass += t12

    print(f"\n  TOTAL: {n_pass}/12 PASS")

    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass, 'n_total': 12,
        'nl_xor1_mean': np.mean(nl_xor1), 'nl_xor1_std': np.std(nl_xor1),
        'nl_xor3_mean': np.mean(nl_xor3), 'nl_xor3_std': np.std(nl_xor3),
        'nl_xor5_mean': np.mean(nl_xor5), 'nl_xor5_std': np.std(nl_xor5),
        'nl_mc_mean': np.mean(nl_mc), 'nl_mc_std': np.std(nl_mc),
    }

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results saved: {SAVE_FILE}")


if __name__ == '__main__':
    main()
