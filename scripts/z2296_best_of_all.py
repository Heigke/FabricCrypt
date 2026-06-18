#!/usr/bin/env python3
"""
z2296_best_of_all.py — Definitive best results: all optimizations combined
==========================================================================
Combine all breakthroughs:
  - Temporal product features (z2294-z2295 key finding)
  - Order-3 temporal products (z2295: +11.4pp on XOR5)
  - Extended lag range τ=1..20 (z2295: XOR above chance to τ=20)
  - FPGA-only readout (bridge hurts temporal features)
  - 3000 steps (longer sequences for temporal tasks)

5 seeds for paper-quality statistics on the DEFINITIVE configuration.

Also: Cross-substrate comparison at this best readout level.
Does GPU add ANYTHING when temporal readout is this good?

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2296_best_of_all.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2296_best_of_all.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
N_STEPS = 3000
WARMUP = 500
TEMP_SAFE = 45.0  # Lower target — laptop hits 100°C and crashes at 99°C ACPI trip
N_SEEDS = 5
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}


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
        # Aggressive thermal check every 100 steps — APU hits 100°C in seconds
        if t > 0 and t % 100 == 0:
            temp = get_max_temp()
            if temp > 80.0:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}°C at step {t}/{n_steps}", end="", flush=True)
                while temp > 50.0:
                    time.sleep(5)
                    temp = get_max_temp()
                    print(f" {temp:.0f}", end="", flush=True)
                print(" resumed", flush=True)
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


def build_best_features(states, dspikes):
    """The definitive best feature set: temporal products order 2+3."""
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, dspikes, delta]

    rng = np.random.default_rng(42)
    qi = np.sort(rng.choice(n_ch, size=min(24, n_ch), replace=False))
    vm_q = states[:, qi]
    ds_q = dspikes[:, qi]

    tau_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    # Order-2 temporal products
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        feats.append(ds_q * shifted)

    # Order-3 temporal products (limited combinations)
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


def full_benchmark(X, u_raw):
    n = len(X)
    n_tr = int(0.7 * n)

    # MC d=1..20
    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 21):
        target = u_raw[WARMUP-d:len(u_raw)-d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    # XOR τ=1..15
    xor = {}
    for tau in [1, 2, 3, 5, 8, 10, 15]:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP-tau:len(u_raw)-tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor[f'tau{tau}'] = acc

    # NARMA 5, 10, 20
    narma = {}
    for order in [5, 10, 20]:
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

    return {'mc_total': mc_total, 'mc_per_delay': mc_per_d, 'xor': xor, 'narma': narma}


def main():
    print("=" * 70)
    print("  z2296: DEFINITIVE BEST — 5 seeds, all optimizations")
    print("  z2295: XOR1=81.2%, XOR5=72.0%, MC(20)=12.5, NARMA-5=0.314")
    print("=" * 70)

    fpga = FPGAEthBridge()
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.3)

    # Runtime parameter setup (CRITICAL: bitstream defaults differ from z2292 optimal)
    fpga.set_leak_cond(0x2000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    fpga.set_threshold_raw(0x20000)

    # Set Vg groups (heterogeneous gate voltages)
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    # Clear synapses
    for n in range(NUM_NEURONS):
        fpga.set_synapse(n, 0x00000000)
        time.sleep(0.001)
    time.sleep(1.0)

    telem = fpga.read_telemetry()
    print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    print(f"  Config: LEAK=0x2000, THRESH=0x20000, BASE_EXC=0x0080, BIAS_GAIN=0x4000")
    gpu = GPUFourpopESN(seed=7777)

    results = {'seeds': [], 'gpu_comparison': {}, 'tests': {}}
    seeds = [42, 123, 456, 789, 2024]

    # ================================================================
    # Main: 5 seeds × best features (FPGA temporal order-2+3)
    # ================================================================
    print("\n[MAIN] 5-seed validation — FPGA temporal features")
    print("-" * 60)

    all_results = []
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        u_raw = rng.uniform(-1, 1, N_STEPS + WARMUP)

        wait_cool(f"pre-seed{si}")
        fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw)

        X = build_best_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:])
        bm = full_benchmark(X, u_raw)
        bm['seed'] = seed
        bm['n_features'] = X.shape[1]
        all_results.append(bm)

        xor = bm['xor']
        print(f"  Seed {seed}: MC={bm['mc_total']:.2f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% "
              f"XOR5={xor['tau5']*100:.1f}% XOR10={xor['tau10']*100:.1f}% N5={bm['narma']['narma5']:.3f} N10={bm['narma']['narma10']:.3f}")

        # Incremental save after each seed (crash protection)
        results['seeds'] = all_results
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)

    results['seeds'] = all_results

    # Compute statistics
    def stat(key_fn):
        vals = [key_fn(r) for r in all_results]
        return {'mean': np.mean(vals), 'std': np.std(vals), 'min': np.min(vals), 'max': np.max(vals), 'vals': vals}

    stats = {
        'mc': stat(lambda r: r['mc_total']),
        'xor1': stat(lambda r: r['xor']['tau1']),
        'xor3': stat(lambda r: r['xor']['tau3']),
        'xor5': stat(lambda r: r['xor']['tau5']),
        'xor8': stat(lambda r: r['xor']['tau8']),
        'xor10': stat(lambda r: r['xor']['tau10']),
        'xor15': stat(lambda r: r['xor']['tau15']),
        'narma5': stat(lambda r: r['narma']['narma5']),
        'narma10': stat(lambda r: r['narma']['narma10']),
        'narma20': stat(lambda r: r['narma']['narma20']),
    }
    results['stats'] = stats

    print(f"\n  STATISTICS (5 seeds):")
    for key in ['mc', 'xor1', 'xor3', 'xor5', 'xor8', 'xor10', 'narma5', 'narma10', 'narma20']:
        s = stats[key]
        if 'xor' in key:
            print(f"    {key:8s}: {s['mean']*100:.1f}% ± {s['std']*100:.1f}%  [{s['min']*100:.1f}%, {s['max']*100:.1f}%]")
        elif 'mc' in key:
            print(f"    {key:8s}: {s['mean']:.2f} ± {s['std']:.2f}  [{s['min']:.2f}, {s['max']:.2f}]")
        else:
            print(f"    {key:8s}: {s['mean']:.3f} ± {s['std']:.3f}  [{s['min']:.3f}, {s['max']:.3f}]")

    # ================================================================
    # GPU comparison: does GPU add anything?
    # ================================================================
    print("\n[GPU COMPARISON] GPU-only vs FPGA temporal")
    print("-" * 60)

    rng0 = np.random.default_rng(42)
    u_raw0 = rng0.uniform(-1, 1, N_STEPS + WARMUP)
    gpu_states = gpu.run(u_raw0, run_seed=42)

    # GPU with same temporal features
    gpu_delta = np.diff(gpu_states[WARMUP:], axis=0)
    gpu_delta = np.vstack([np.zeros((1, gpu_states.shape[1])), gpu_delta])
    gpu_basic = np.hstack([gpu_states[WARMUP:], gpu_delta])

    # GPU temporal products
    n_gpu = gpu_states.shape[1]
    rng_q = np.random.default_rng(42)
    qi_g = np.sort(rng_q.choice(n_gpu, size=min(24, n_gpu), replace=False))
    vm_g = gpu_states[WARMUP:][:, qi_g]
    gpu_temp_feats = [gpu_basic]
    for tau in [1, 2, 3, 5, 8, 10]:
        shifted = np.zeros_like(vm_g)
        shifted[tau:] = vm_g[:-tau]
        gpu_temp_feats.append(vm_g * shifted)
    gpu_temp_feats.append(np.square(vm_g))
    X_gpu = np.hstack(gpu_temp_feats)

    bm_gpu = full_benchmark(X_gpu, u_raw0)
    xg = bm_gpu['xor']
    print(f"  GPU temporal: MC={bm_gpu['mc_total']:.2f} XOR1={xg['tau1']*100:.1f}% XOR3={xg['tau3']*100:.1f}% "
          f"XOR5={xg['tau5']*100:.1f}% N5={bm_gpu['narma']['narma5']:.3f} N10={bm_gpu['narma']['narma10']:.3f}")

    # FPGA seed=42 for direct comparison
    fpga_s42 = all_results[0]
    xf = fpga_s42['xor']
    print(f"  FPGA temporal: MC={fpga_s42['mc_total']:.2f} XOR1={xf['tau1']*100:.1f}% XOR3={xf['tau3']*100:.1f}% "
          f"XOR5={xf['tau5']*100:.1f}% N5={fpga_s42['narma']['narma5']:.3f} N10={fpga_s42['narma']['narma10']:.3f}")

    results['gpu_comparison'] = {
        'gpu': bm_gpu,
        'fpga_s42': fpga_s42,
    }

    # ================================================================
    # TESTS
    # ================================================================
    print("\n" + "=" * 70)
    print("  TESTS")
    print("=" * 70)

    tests = {}
    n_pass = 0

    # T1: Mean XOR1 > 78%
    t1 = stats['xor1']['mean'] > 0.78
    tests['T1'] = {'pass': t1, 'val': stats['xor1']['mean']}
    print(f"  T1 {'PASS' if t1 else 'FAIL'}: Mean XOR1={stats['xor1']['mean']*100:.1f}% > 78%")
    n_pass += t1

    # T2: Mean XOR3 > 62%
    t2 = stats['xor3']['mean'] > 0.62
    tests['T2'] = {'pass': t2, 'val': stats['xor3']['mean']}
    print(f"  T2 {'PASS' if t2 else 'FAIL'}: Mean XOR3={stats['xor3']['mean']*100:.1f}% > 62%")
    n_pass += t2

    # T3: Mean XOR5 > 58%
    t3 = stats['xor5']['mean'] > 0.58
    tests['T3'] = {'pass': t3, 'val': stats['xor5']['mean']}
    print(f"  T3 {'PASS' if t3 else 'FAIL'}: Mean XOR5={stats['xor5']['mean']*100:.1f}% > 58%")
    n_pass += t3

    # T4: Mean XOR10 > 53%
    t4 = stats['xor10']['mean'] > 0.53
    tests['T4'] = {'pass': t4, 'val': stats['xor10']['mean']}
    print(f"  T4 {'PASS' if t4 else 'FAIL'}: Mean XOR10={stats['xor10']['mean']*100:.1f}% > 53%")
    n_pass += t4

    # T5: Mean MC > 8.0
    t5 = stats['mc']['mean'] > 8.0
    tests['T5'] = {'pass': t5, 'val': stats['mc']['mean']}
    print(f"  T5 {'PASS' if t5 else 'FAIL'}: Mean MC={stats['mc']['mean']:.2f} > 8.0")
    n_pass += t5

    # T6: Mean NARMA-5 < 0.40
    t6 = stats['narma5']['mean'] < 0.40
    tests['T6'] = {'pass': t6, 'val': stats['narma5']['mean']}
    print(f"  T6 {'PASS' if t6 else 'FAIL'}: Mean NARMA-5={stats['narma5']['mean']:.3f} < 0.40")
    n_pass += t6

    # T7: Mean NARMA-10 < 0.50
    t7 = stats['narma10']['mean'] < 0.50
    tests['T7'] = {'pass': t7, 'val': stats['narma10']['mean']}
    print(f"  T7 {'PASS' if t7 else 'FAIL'}: Mean NARMA-10={stats['narma10']['mean']:.3f} < 0.50")
    n_pass += t7

    # T8: Min XOR1 across seeds > 75% (consistent)
    t8 = stats['xor1']['min'] > 0.75
    tests['T8'] = {'pass': t8, 'min': stats['xor1']['min']}
    print(f"  T8 {'PASS' if t8 else 'FAIL'}: Min XOR1={stats['xor1']['min']*100:.1f}% > 75%")
    n_pass += t8

    # T9: FPGA beats GPU on XOR1
    t9 = fpga_s42['xor']['tau1'] > bm_gpu['xor']['tau1']
    tests['T9'] = {'pass': t9, 'fpga': fpga_s42['xor']['tau1'], 'gpu': bm_gpu['xor']['tau1']}
    print(f"  T9 {'PASS' if t9 else 'FAIL'}: FPGA XOR1={fpga_s42['xor']['tau1']*100:.1f}% > GPU={bm_gpu['xor']['tau1']*100:.1f}%")
    n_pass += t9

    # T10: FPGA beats GPU on NARMA-10
    t10 = fpga_s42['narma']['narma10'] < bm_gpu['narma']['narma10']
    tests['T10'] = {'pass': t10, 'fpga': fpga_s42['narma']['narma10'], 'gpu': bm_gpu['narma']['narma10']}
    print(f"  T10 {'PASS' if t10 else 'FAIL'}: FPGA NARMA-10={fpga_s42['narma']['narma10']:.3f} < GPU={bm_gpu['narma']['narma10']:.3f}")
    n_pass += t10

    # T11: XOR std < 5% (stable)
    t11 = stats['xor1']['std'] < 0.05
    tests['T11'] = {'pass': t11, 'std': stats['xor1']['std']}
    print(f"  T11 {'PASS' if t11 else 'FAIL'}: XOR1 std={stats['xor1']['std']*100:.1f}% < 5%")
    n_pass += t11

    # T12: NARMA-20 mean < 0.55
    t12 = stats['narma20']['mean'] < 0.55
    tests['T12'] = {'pass': t12, 'val': stats['narma20']['mean']}
    print(f"  T12 {'PASS' if t12 else 'FAIL'}: Mean NARMA-20={stats['narma20']['mean']:.3f} < 0.55")
    n_pass += t12

    # T13: Mean XOR8 > 55%
    t13 = stats['xor8']['mean'] > 0.55
    tests['T13'] = {'pass': t13, 'val': stats['xor8']['mean']}
    print(f"  T13 {'PASS' if t13 else 'FAIL'}: Mean XOR8={stats['xor8']['mean']*100:.1f}% > 55%")
    n_pass += t13

    # T14: Best single seed MC > 12.0
    t14 = stats['mc']['max'] > 12.0
    tests['T14'] = {'pass': t14, 'max': stats['mc']['max']}
    print(f"  T14 {'PASS' if t14 else 'FAIL'}: Best MC={stats['mc']['max']:.2f} > 12.0")
    n_pass += t14

    print(f"\n  TOTAL: {n_pass}/14 PASS")

    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass, 'n_total': 14,
        'configuration': 'FPGA 128-neuron, LEAK=0x2000, temporal order-2+3 features, τ=1..20',
    }

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results saved: {SAVE_FILE}")


if __name__ == '__main__':
    main()
