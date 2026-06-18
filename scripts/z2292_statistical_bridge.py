#!/usr/bin/env python3
"""
z2292_statistical_bridge.py — Statistical validation of bridge results
=====================================================================
z2291: Best results at LEAK=0x2000: XOR1=76.2% (BR_FPGA), MC=2.265 (FULL_BR),
NARMA-10=0.772. Need multiple repetitions for confidence intervals.

Plan:
  5 seeds × 3 conditions (FPGA, BR_FPGA, FULL_BR) at LEAK=0x2000
  + NARMA comparison (5, 10, 20)
  + Mutual information GPU↔FPGA

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2292_statistical_bridge.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2292_statistical_bridge.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
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
            v_a = np.tanh((1 - self.leak[sa:ea]) * v[sa:ea] + self.input_w[sa:ea] * u + rec[sa:ea] + self.bias[sa:ea] + 0.02 * bv)
            sb, eb = pp, 2 * pp
            v_b = v[sb:eb].copy()
            ns = max(1, pp // 10)
            si = rng.choice(pp, size=ns * 2, replace=False)
            for k in range(0, ns * 2 - 1, 2):
                v_b[si[k]], v_b[si[k + 1]] = v_b[si[k + 1]], v_b[si[k]]
            v_b = np.tanh((1 - self.leak[sb:eb]) * v_b + self.input_w[sb:eb] * u + rec[sb:eb] + self.bias[sb:eb])
            sc, ec = 2 * pp, 3 * pp
            v_c = np.tanh(((1 - self.leak[sc:ec]) * v[sc:ec] + self.input_w[sc:ec] * u + rec[sc:ec] + self.bias[sc:ec]) / self.temp_c)
            sd, ed = 3 * pp, 4 * pp
            sn = rng.uniform(-1, 1, pp) * 0.01
            v_d = np.tanh((1 - self.leak[sd:ed]) * v[sd:ed] + self.input_w[sd:ed] * u + rec[sd:ed] + self.bias[sd:ed] + sn)
            v_new = np.concatenate([v_a, v_b, v_c, v_d])
            v_new += rng.uniform(-1, 1, self.N) * 0.003
            spike_mask = v_new > self.thr
            v_new[spike_mask] -= self.thr[spike_mask]
            v = v_new
            h = 0.93 * h + 0.07 * v
            slow = 0.99 * slow + 0.01 * v
            states[t] = v + 0.3 * h + 0.1 * slow
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
            states[t] = states[t - 1]
            dspikes[t] = dspikes[t - 1]
    fpga.set_mac_signal(0.0)
    return states, dspikes


def build_features(states, dspikes):
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, states.shape[1])), delta])
    X = np.hstack([states, dspikes, delta])
    n_cols = states.shape[1]
    qi = np.arange(0, n_cols, max(1, n_cols // 32))[:32]
    vm = states[:, qi]
    ds = dspikes[:, qi]
    X = np.hstack([X, vm * ds, vm[:, :-1] * vm[:, 1:], np.square(vm)])
    return X


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


def benchmark_suite(X, u_raw, narma_orders=[5, 10, 20]):
    n = len(X)
    n_tr = int(0.7 * n)

    # MC d=1..10
    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 11):
        target = u_raw[WARMUP - d:len(u_raw) - d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    # XOR τ=1,2,3,5
    xor = {}
    for tau in [1, 2, 3, 5]:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP - tau:len(u_raw) - tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor[f'tau{tau}'] = acc

    # NARMA at multiple orders
    narma = {}
    for order in narma_orders:
        T = len(u_raw)
        u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
        y = np.zeros(T)
        for t in range(order, T):
            y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-order:t]) + 1.5 * u_n[t-1] * u_n[t-order] + 0.1
            y[t] = np.tanh(y[t])
        target_narma = y[WARMUP:]
        nn = min(n, len(target_narma))
        best_nrmse = 999.0
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            I2 = np.eye(X[:n_tr].shape[1])
            try:
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I2, X[:n_tr].T @ target_narma[:n_tr])
                pred = X[n_tr:nn] @ w
                gt = target_narma[n_tr:nn]
                nrmse = np.sqrt(np.mean((gt - pred)**2)) / (np.std(gt) + 1e-10)
                if nrmse < best_nrmse:
                    best_nrmse = nrmse
            except Exception:
                pass
        narma[f'narma{order}'] = best_nrmse

    return {
        'mc_total': mc_total, 'mc_per_delay': mc_per_d,
        'xor': xor, 'narma': narma,
    }


def mutual_information(x, y, bins=16):
    """Estimate MI between two 1D signals."""
    c_xy, xedges, yedges = np.histogram2d(x, y, bins=bins)
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return max(0.0, mi)


def main():
    print("=" * 70)
    print("  z2292: STATISTICAL BRIDGE — 5 seeds × 3 conditions")
    print("  z2291: Best at LEAK=0x2000, XOR1=76.2%, MC=2.265, NARMA=0.772")
    print("=" * 70)

    results = {
        'experiment': 'z2292_statistical_bridge',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    fpga = FPGAEthBridge()
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.3)

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
    time.sleep(1.0)

    telem = fpga.read_telemetry()
    if telem is None:
        time.sleep(0.5)
        telem = fpga.read_telemetry()
    if telem is None:
        print("  FATAL: No FPGA telemetry")
        fpga.close()
        sys.exit(1)
    print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")

    gpu_esn = GPUFourpopESN(n_per_pop=64)

    # Collect results across seeds
    all_runs = {
        'FPGA': [], 'BR_FPGA': [], 'FULL_BR': [], 'GPU': [],
    }

    seeds = [42, 123, 456, 789, 2024]

    for si, seed in enumerate(seeds):
        print(f"\n[Seed {si+1}/{N_SEEDS}] seed={seed}")
        rng = np.random.default_rng(seed)
        u = rng.uniform(-1, 1, N_STEPS).astype(np.float64)

        # GPU
        gpu_states = gpu_esn.run(u, run_seed=seed)
        gpu_X = gpu_states[WARMUP:]
        gpu_bench = benchmark_suite(gpu_X, u)
        all_runs['GPU'].append(gpu_bench)
        print(f"  GPU: MC={gpu_bench['mc_total']:.3f} XOR1={gpu_bench['xor']['tau1']*100:.1f}%")

        # FPGA
        wait_cool(f"pre-FPGA-s{si}")
        fpga_s, fpga_ds = fpga_run_continuous(fpga, u)
        fpga_X = build_features(fpga_s, fpga_ds)[WARMUP:]
        fpga_bench = benchmark_suite(fpga_X, u)
        all_runs['FPGA'].append(fpga_bench)
        print(f"  FPGA: MC={fpga_bench['mc_total']:.3f} XOR1={fpga_bench['xor']['tau1']*100:.1f}%")

        # BR_FPGA (GPU→MAC, FPGA readout)
        wait_cool(f"pre-BR-s{si}")
        gpu_mac = np.mean(np.abs(gpu_states), axis=1)
        gpu_mac_norm = gpu_mac / (gpu_mac.max() + 1e-10)
        bridge_mac = np.clip(0.6 * (u * 0.4 + 0.5) + 0.4 * gpu_mac_norm, 0, 1)
        br_s, br_ds = fpga_run_continuous(fpga, u, mac_signal=bridge_mac)
        br_fpga_X = build_features(br_s, br_ds)[WARMUP:]
        br_fpga_bench = benchmark_suite(br_fpga_X, u)
        all_runs['BR_FPGA'].append(br_fpga_bench)
        print(f"  BR_FPGA: MC={br_fpga_bench['mc_total']:.3f} XOR1={br_fpga_bench['xor']['tau1']*100:.1f}%")

        # FULL_BR
        br_full_X = np.hstack([br_fpga_X, gpu_states[WARMUP:]])
        br_full_bench = benchmark_suite(br_full_X, u)
        all_runs['FULL_BR'].append(br_full_bench)
        print(f"  FULL_BR: MC={br_full_bench['mc_total']:.3f} XOR1={br_full_bench['xor']['tau1']*100:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Mutual information (last seed's data)
    # ═══════════════════════════════════════════════════════════
    print("\n[MI] Mutual information GPU ↔ FPGA")
    gpu_mean = np.mean(gpu_states[WARMUP:], axis=1)
    fpga_mean = np.mean(br_s[WARMUP:], axis=1)
    mi_total = mutual_information(gpu_mean, fpga_mean)
    print(f"  MI(GPU_mean, FPGA_mean) = {mi_total:.4f} bits")

    # Per-channel MI (sample 8 channels)
    mi_channels = []
    for ch in range(0, NUM_NEURONS, 16):
        mi_ch = mutual_information(gpu_mean, br_s[WARMUP:, ch])
        mi_channels.append(mi_ch)
    mi_mean_ch = np.mean(mi_channels)
    mi_std_ch = np.std(mi_channels)
    print(f"  MI per channel (8 sampled): {mi_mean_ch:.4f} ± {mi_std_ch:.4f} bits")

    results['mutual_info'] = {
        'mi_total': mi_total,
        'mi_per_channel_mean': mi_mean_ch,
        'mi_per_channel_std': mi_std_ch,
    }

    # ═══════════════════════════════════════════════════════════
    # Statistics
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  STATISTICS (mean ± std over 5 seeds)")
    print(f"{'='*70}")

    stats = {}
    for mode in ['FPGA', 'BR_FPGA', 'FULL_BR', 'GPU']:
        runs = all_runs[mode]
        mc_vals = [r['mc_total'] for r in runs]
        xor1_vals = [r['xor']['tau1'] for r in runs]
        xor3_vals = [r['xor']['tau3'] for r in runs]
        xor5_vals = [r['xor']['tau5'] for r in runs]
        n5_vals = [r['narma']['narma5'] for r in runs]
        n10_vals = [r['narma']['narma10'] for r in runs]
        n20_vals = [r['narma']['narma20'] for r in runs]

        s = {
            'mc': {'mean': np.mean(mc_vals), 'std': np.std(mc_vals)},
            'xor1': {'mean': np.mean(xor1_vals), 'std': np.std(xor1_vals)},
            'xor3': {'mean': np.mean(xor3_vals), 'std': np.std(xor3_vals)},
            'xor5': {'mean': np.mean(xor5_vals), 'std': np.std(xor5_vals)},
            'narma5': {'mean': np.mean(n5_vals), 'std': np.std(n5_vals)},
            'narma10': {'mean': np.mean(n10_vals), 'std': np.std(n10_vals)},
            'narma20': {'mean': np.mean(n20_vals), 'std': np.std(n20_vals)},
        }
        stats[mode] = s

        print(f"\n  {mode}:")
        print(f"    MC:      {s['mc']['mean']:.3f} ± {s['mc']['std']:.3f}")
        print(f"    XOR1:    {s['xor1']['mean']*100:.1f}% ± {s['xor1']['std']*100:.1f}%")
        print(f"    XOR3:    {s['xor3']['mean']*100:.1f}% ± {s['xor3']['std']*100:.1f}%")
        print(f"    XOR5:    {s['xor5']['mean']*100:.1f}% ± {s['xor5']['std']*100:.1f}%")
        print(f"    NARMA5:  {s['narma5']['mean']:.3f} ± {s['narma5']['std']:.3f}")
        print(f"    NARMA10: {s['narma10']['mean']:.3f} ± {s['narma10']['std']:.3f}")
        print(f"    NARMA20: {s['narma20']['mean']:.3f} ± {s['narma20']['std']:.3f}")

    results['statistics'] = stats
    results['all_runs'] = {
        mode: [{k: v for k, v in r.items()} for r in runs]
        for mode, runs in all_runs.items()
    }

    # ═══════════════════════════════════════════════════════════
    # Tests
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  KEY TESTS")
    print(f"{'='*70}")

    tests = {}

    # T1: FULL_BR MC mean > 2.0
    t1 = stats['FULL_BR']['mc']['mean'] > 2.0
    tests['T1_fullbr_mc_gt_2'] = {'pass': t1,
        'desc': f"FULL_BR MC={stats['FULL_BR']['mc']['mean']:.3f}±{stats['FULL_BR']['mc']['std']:.3f} > 2.0"}

    # T2: FULL_BR MC > FPGA MC (mean)
    t2 = stats['FULL_BR']['mc']['mean'] > stats['FPGA']['mc']['mean']
    tests['T2_fullbr_mc_gt_fpga'] = {'pass': t2,
        'desc': f"FULL_BR MC={stats['FULL_BR']['mc']['mean']:.3f} > FPGA={stats['FPGA']['mc']['mean']:.3f}"}

    # T3: BR_FPGA XOR1 mean > 70%
    t3 = stats['BR_FPGA']['xor1']['mean'] > 0.70
    tests['T3_brfpga_xor1_gt_70'] = {'pass': t3,
        'desc': f"BR_FPGA XOR1={stats['BR_FPGA']['xor1']['mean']*100:.1f}% > 70%"}

    # T4: FPGA XOR1 > GPU XOR1 (FPGA has better nonlinearity)
    t4 = stats['FPGA']['xor1']['mean'] > stats['GPU']['xor1']['mean']
    tests['T4_fpga_xor_gt_gpu'] = {'pass': t4,
        'desc': f"FPGA XOR1={stats['FPGA']['xor1']['mean']*100:.1f}% > GPU={stats['GPU']['xor1']['mean']*100:.1f}%"}

    # T5: NARMA-10 FULL_BR < FPGA (bridge helps temporal prediction)
    t5 = stats['FULL_BR']['narma10']['mean'] < stats['FPGA']['narma10']['mean']
    tests['T5_narma10_bridge_better'] = {'pass': t5,
        'desc': f"FULL_BR NARMA10={stats['FULL_BR']['narma10']['mean']:.3f} < FPGA={stats['FPGA']['narma10']['mean']:.3f}"}

    # T6: NARMA-20 < 1.0 for FULL_BR
    t6 = stats['FULL_BR']['narma20']['mean'] < 1.0
    tests['T6_narma20_useful'] = {'pass': t6,
        'desc': f"FULL_BR NARMA20={stats['FULL_BR']['narma20']['mean']:.3f} < 1.0"}

    # T7: MI > 0.1 bits (real information coupling)
    t7 = mi_total > 0.1
    tests['T7_mi_above_01'] = {'pass': t7,
        'desc': f"MI(GPU,FPGA)={mi_total:.4f} > 0.1 bits"}

    # T8: XOR3 > 55% for any condition
    best_xor3 = max(stats[m]['xor3']['mean'] for m in stats)
    t8 = best_xor3 > 0.55
    tests['T8_xor3_gt_55'] = {'pass': t8,
        'desc': f"Best XOR3={best_xor3*100:.1f}% > 55%"}

    # T9: Std(MC) < 0.3 for FULL_BR (reliable)
    t9 = stats['FULL_BR']['mc']['std'] < 0.3
    tests['T9_mc_reliable'] = {'pass': t9,
        'desc': f"FULL_BR MC std={stats['FULL_BR']['mc']['std']:.3f} < 0.3"}

    # T10: FULL_BR best on ≥ 3 of {MC, XOR1, NARMA5, NARMA10, NARMA20}
    wins = 0
    for metric in ['mc', 'narma5', 'narma10', 'narma20']:
        if metric == 'mc':
            if stats['FULL_BR']['mc']['mean'] == max(stats[m]['mc']['mean'] for m in stats):
                wins += 1
        else:
            if stats['FULL_BR'][metric]['mean'] == min(stats[m][metric]['mean'] for m in stats):
                wins += 1
    if stats['FULL_BR']['xor1']['mean'] == max(stats[m]['xor1']['mean'] for m in stats):
        wins += 1
    t10 = wins >= 3
    tests['T10_fullbr_best_3of5'] = {'pass': t10,
        'desc': f"FULL_BR best on {wins}/5 metrics"}

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_tests = len(tests)

    for k, v in tests.items():
        tag = k.split('_', 1)[0]
        print(f"  {tag} {'PASS' if v['pass'] else 'FAIL'}: {v['desc']}")

    print(f"\n  TOTAL: {n_pass}/{n_tests} PASS")

    results['key_tests'] = tests
    results['n_pass'] = n_pass
    results['n_tests'] = n_tests

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results saved: {SAVE_FILE}")

    fpga.close()


if __name__ == '__main__':
    main()
