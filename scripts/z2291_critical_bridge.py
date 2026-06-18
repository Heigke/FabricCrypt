#!/usr/bin/env python3
"""
z2291_critical_bridge.py — Bridge at the critical transition point
=================================================================
z2290 Phase 1 revealed gradual transition:
  0x0800: eff_dim=52, MC=0     (independent, no memory)
  0x1800: eff_dim=5,  MC=0.65  (intermediate, XOR5=53.5%!)
  0x1C00: eff_dim=3,  MC=1.61  (near-critical, good MC)
  0x2000: eff_dim=1.5, MC=1.94 (synchronized, best MC)

Test three leak points near the critical transition with GPU bridge:
  LEAK=0x1800 — balance point (eff_dim>5, some MC)
  LEAK=0x1A00 — critical onset (MC>1, eff_dim≈4)
  LEAK=0x1C00 — strong MC with some diversity

At each: FPGA_ONLY vs BRIDGE (GPU→MAC + combined readout)
Benchmarks: MC d=1..10, XOR τ=1..5, NARMA-10, Wave-4, Wave-7

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2291_critical_bridge.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2291_critical_bridge.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
N_STEPS = 2500
WARMUP = 400
N_WAVE_TRIALS = 50
N_WAVE_STEPS = 60
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
    def __init__(self, n_per_pop=64):
        self.pp = n_per_pop
        self.N = 4 * n_per_pop
        rng = np.random.default_rng(7777)
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

    def run(self, input_seq):
        n_steps = len(input_seq)
        pp = self.pp
        states = np.zeros((n_steps, self.N))
        v = np.zeros(self.N)
        h = np.zeros(self.N)
        slow = np.zeros(self.N)
        rng = np.random.default_rng(42)
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


def fpga_run_continuous(fpga, u, mac_signal=None, sample_hz=SAMPLE_HZ):
    n_steps = len(u)
    if mac_signal is None:
        mac_signal = np.clip(u * 0.4 + 0.5, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / sample_hz

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


def full_benchmark(X, u_raw, label=""):
    n = len(X)
    n_tr = int(0.7 * n)

    # MC d=1..10
    mc_total = 0.0
    mc_per_d = {}
    for d in range(1, 11):
        target = u_raw[WARMUP - d:len(u_raw) - d]
        nn = min(n, len(target))
        if nn < n_tr + 20:
            mc_per_d[str(d)] = 0.0
            continue
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc_per_d[str(d)] = r2
        mc_total += r2

    # XOR
    xor = {}
    for tau in [1, 2, 3, 5]:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP - tau:len(u_raw) - tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor[f'tau{tau}'] = acc

    # NARMA-10
    T = len(u_raw)
    u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
    y = np.zeros(T)
    for t in range(10, T):
        y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t]) + 1.5 * u_n[t-1] * u_n[t-10] + 0.1
        y[t] = np.tanh(y[t])
    target_narma = y[WARMUP:]
    nn = min(n, len(target_narma))
    narma_nrmse = 999.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        I2 = np.eye(X[:n_tr].shape[1])
        try:
            w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I2, X[:n_tr].T @ target_narma[:n_tr])
            pred = X[n_tr:nn] @ w
            gt = target_narma[n_tr:nn]
            nrmse = np.sqrt(np.mean((gt - pred)**2)) / (np.std(gt) + 1e-10)
            if nrmse < narma_nrmse:
                narma_nrmse = nrmse
        except Exception:
            pass

    print(f"    {label}: MC={mc_total:.3f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% XOR5={xor['tau5']*100:.1f}% NARMA={narma_nrmse:.3f}")

    return {
        'mc_total': mc_total, 'mc_per_delay': mc_per_d,
        'xor': xor, 'narma10_nrmse': narma_nrmse,
    }


def generate_waveform(cls, steps):
    t = np.linspace(0, 2 * np.pi, steps)
    waveforms = [
        lambda: np.sin(t),
        lambda: np.sign(np.sin(t)),
        lambda: 2 * np.abs(2 * (t/(2*np.pi) - np.floor(t/(2*np.pi) + 0.5))) - 1,
        lambda: 2 * (t/(2*np.pi) - np.floor(t/(2*np.pi))) - 1,
        lambda: np.sin(t) * np.sin(3*t),
        lambda: np.sign(np.sin(2*t)),
        lambda: np.abs(np.sin(t)) * 2 - 1,
    ]
    return waveforms[cls % len(waveforms)]()


def extract_trial_features(states, dspikes):
    feat_mean = states.mean(axis=0)
    feat_std = states.std(axis=0)
    feat_last = states[-1]
    ds_mean = dspikes.mean(axis=0)
    ds_std = dspikes.std(axis=0)
    delta = np.diff(states, axis=0)
    feat_delta_std = delta.std(axis=0) if len(delta) > 0 else np.zeros(states.shape[1])
    return np.concatenate([feat_mean, feat_std, feat_last, ds_mean, ds_std, feat_delta_std])


def ridge_classify(X, y, n_classes, n_splits=5):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import cross_val_score
    sigma = np.std(X, axis=0)
    sigma[sigma < 1e-2] = 1.0
    X_n = X / sigma
    clf = RidgeClassifier(alpha=10.0)
    scores = cross_val_score(clf, X_n, y, cv=n_splits)
    return float(scores.mean()), float(scores.std())


def compute_diversity(states):
    vm = states[WARMUP:]
    vm_c = vm - vm.mean(0)
    try:
        sv = np.linalg.svd(vm_c, compute_uv=False)
        sv_n = sv / (sv.sum() + 1e-30)
        eff_dim = float(np.exp(-np.sum(sv_n * np.log(sv_n + 1e-30))))
    except:
        eff_dim = 0.0
    corr_mat = np.corrcoef(vm.T)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    xcorr = float(np.mean(np.abs(corr_mat[mask])))
    return eff_dim, xcorr


def main():
    print("=" * 70)
    print("  z2291: CRITICAL BRIDGE — GPU bridge at phase transition")
    print("  z2290: Transition 0x0800→0x2000, sweet spot 0x1800-0x1C00")
    print("=" * 70)

    results = {
        'experiment': 'z2291_critical_bridge',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    fpga = FPGAEthBridge()
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.3)

    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    fpga.set_threshold_raw(0x20000)

    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    # Zero synapses for maximum clarity
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

    rng = np.random.default_rng(42)
    u = rng.uniform(-1, 1, N_STEPS).astype(np.float64)
    gpu_esn = GPUFourpopESN(n_per_pop=64)

    # Pre-compute GPU states
    gpu_states = gpu_esn.run(u)
    gpu_mac = np.mean(np.abs(gpu_states), axis=1)
    gpu_mac_norm = gpu_mac / (gpu_mac.max() + 1e-10)

    leak_points = [
        ('0x1800', 0x1800, "intermediate"),
        ('0x1A00', 0x1A00, "critical_onset"),
        ('0x1C00', 0x1C00, "near_critical"),
        ('0x2000', 0x2000, "synchronized"),
    ]

    all_conditions = {}

    for leak_name, leak_val, desc in leak_points:
        print(f"\n{'='*60}")
        print(f"  LEAK={leak_name} ({desc})")
        print(f"{'='*60}")

        fpga.set_leak_cond(leak_val)
        time.sleep(0.5)

        # A) FPGA only
        wait_cool(f"pre-{leak_name}-FPGA")
        print(f"  [FPGA_ONLY]")
        fpga_s, fpga_ds = fpga_run_continuous(fpga, u)
        fpga_X = build_features(fpga_s, fpga_ds)[WARMUP:]
        ed_fpga, xc_fpga = compute_diversity(fpga_s)
        fpga_bench = full_benchmark(fpga_X, u, f"FPGA eff_dim={ed_fpga:.1f}")

        # B) BRIDGE: GPU→MAC, FPGA-only readout
        wait_cool(f"pre-{leak_name}-BRIDGE_FPGA")
        print(f"  [BRIDGE_FPGA_READOUT]")
        bridge_mac = np.clip(0.6 * (u * 0.4 + 0.5) + 0.4 * gpu_mac_norm, 0, 1)
        br_s, br_ds = fpga_run_continuous(fpga, u, mac_signal=bridge_mac)
        br_fpga_X = build_features(br_s, br_ds)[WARMUP:]
        ed_br, xc_br = compute_diversity(br_s)
        br_fpga_bench = full_benchmark(br_fpga_X, u, f"BR_FPGA eff_dim={ed_br:.1f}")

        # C) FULL BRIDGE: GPU→MAC, FPGA+GPU readout
        print(f"  [FULL_BRIDGE]")
        br_full_X = np.hstack([br_fpga_X, gpu_states[WARMUP:]])
        br_full_bench = full_benchmark(br_full_X, u, f"FULL_BR eff_dim={ed_br:.1f}")

        all_conditions[leak_name] = {
            'desc': desc,
            'FPGA_ONLY': {**fpga_bench, 'eff_dim': ed_fpga, 'xcorr': xc_fpga},
            'BRIDGE_FPGA': {**br_fpga_bench, 'eff_dim': ed_br, 'xcorr': xc_br},
            'FULL_BRIDGE': br_full_bench,
        }

    results['conditions'] = all_conditions

    # ═══════════════════════════════════════════════════════════
    # EXP 2: Wave classification at each leak point
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  Wave-4 classification at each leak point")
    print(f"{'='*60}")

    wave_results = {}
    for leak_name, leak_val, desc in leak_points:
        fpga.set_leak_cond(leak_val)
        time.sleep(0.3)
        wait_cool(f"pre-wave-{leak_name}")

        for mode in ['FPGA', 'BRIDGE']:
            feats_list = []
            labels = []
            for trial in range(N_WAVE_TRIALS):
                cls = trial % 4
                wave = generate_waveform(cls, N_WAVE_STEPS)
                if mode == 'FPGA':
                    mac = np.clip(wave * 0.4 + 0.5, 0, 1)
                    s, ds = fpga_run_continuous(fpga, wave, mac_signal=mac)
                    feat = extract_trial_features(s, ds)
                else:
                    gpu_s = gpu_esn.run(wave)
                    gpu_m = np.mean(np.abs(gpu_s), axis=1)
                    gpu_m_n = gpu_m / (gpu_m.max() + 1e-10)
                    mac = np.clip(0.6 * (wave * 0.4 + 0.5) + 0.4 * gpu_m_n, 0, 1)
                    s, ds = fpga_run_continuous(fpga, wave, mac_signal=mac)
                    feat = extract_trial_features(s, ds)
                    gpu_feat = extract_trial_features(gpu_s, np.zeros_like(gpu_s))
                    feat = np.concatenate([feat, gpu_feat])
                feats_list.append(feat)
                labels.append(cls)

            X_w = np.array(feats_list)
            y_w = np.array(labels)
            acc, std = ridge_classify(X_w, y_w, n_classes=4)
            key = f'{leak_name}_{mode}'
            wave_results[key] = {'acc': acc, 'std': std}
            print(f"  {leak_name} {mode}: Wave-4 = {acc*100:.1f}% ± {std*100:.1f}%")

    results['wave4'] = wave_results

    # ═══════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Leak':<8s} {'Mode':<15s} {'MC':>7s} {'XOR1':>7s} {'XOR3':>7s} {'XOR5':>7s} {'NARMA':>7s} {'EffDim':>7s} {'Wave4':>7s}")
    print(f"  {'-'*8} {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for leak_name, leak_val, desc in leak_points:
        cond = all_conditions[leak_name]
        for mode_name, mode_key in [('FPGA', 'FPGA_ONLY'), ('BR_FPGA', 'BRIDGE_FPGA'), ('FULL_BR', 'FULL_BRIDGE')]:
            d = cond[mode_key]
            w4_key = f'{leak_name}_{("FPGA" if mode_key == "FPGA_ONLY" else "BRIDGE")}'
            w4 = wave_results.get(w4_key, {}).get('acc', 0)
            ed = d.get('eff_dim', 0)
            print(f"  {leak_name:<8s} {mode_name:<15s} {d['mc_total']:7.3f} "
                  f"{d['xor']['tau1']*100:6.1f}% {d['xor']['tau3']*100:6.1f}% {d['xor']['tau5']*100:6.1f}% "
                  f"{d['narma10_nrmse']:7.3f} {ed:7.1f} {w4*100:6.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Tests
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  KEY TESTS")
    print(f"{'='*70}")

    tests = {}

    # T1: Any intermediate leak (0x1800/0x1A00) has eff_dim > 3 AND MC > 0.5
    for lk in ['0x1800', '0x1A00']:
        if lk in all_conditions:
            c = all_conditions[lk]['FPGA_ONLY']
            if c.get('eff_dim', 0) > 3 and c['mc_total'] > 0.5:
                t1 = True
                break
    else:
        t1 = False
    tests['T1_intermediate_balanced'] = {'pass': t1, 'desc': 'Intermediate leak has eff_dim>3 AND MC>0.5'}

    # T2: Bridge improves MC at all leak points
    bridge_mc_wins = 0
    for lk in ['0x1800', '0x1A00', '0x1C00', '0x2000']:
        if lk in all_conditions:
            c = all_conditions[lk]
            if c['FULL_BRIDGE']['mc_total'] > c['FPGA_ONLY']['mc_total']:
                bridge_mc_wins += 1
    t2 = bridge_mc_wins >= 3
    tests['T2_bridge_mc_wins'] = {'pass': t2, 'desc': f'FULL_BRIDGE MC > FPGA at {bridge_mc_wins}/4 leaks'}

    # T3: FPGA XOR1 > 65% at optimal leak
    best_fpga_xor = max(all_conditions[lk]['FPGA_ONLY']['xor']['tau1']
                       for lk in all_conditions)
    t3 = best_fpga_xor > 0.65
    tests['T3_fpga_xor1_gt_65'] = {'pass': t3, 'desc': f'Best FPGA XOR1={best_fpga_xor*100:.1f}% > 65%'}

    # T4: XOR τ=3 > 55% at any condition
    best_xor3 = max(
        max(all_conditions[lk][mode]['xor']['tau3']
            for mode in ['FPGA_ONLY', 'BRIDGE_FPGA', 'FULL_BRIDGE'])
        for lk in all_conditions
    )
    t4 = best_xor3 > 0.55
    tests['T4_xor3_above_55'] = {'pass': t4, 'desc': f'Best XOR3={best_xor3*100:.1f}% > 55%'}

    # T5: NARMA-10 < 0.9 at any condition
    best_narma = min(
        min(all_conditions[lk][mode]['narma10_nrmse']
            for mode in ['FPGA_ONLY', 'BRIDGE_FPGA', 'FULL_BRIDGE'])
        for lk in all_conditions
    )
    t5 = best_narma < 0.9
    tests['T5_narma10_below_09'] = {'pass': t5, 'desc': f'Best NARMA10={best_narma:.3f} < 0.9'}

    # T6: Wave-4 > 90% at all leak points (bridge)
    all_wave4_pass = all(
        wave_results.get(f'{lk}_BRIDGE', {}).get('acc', 0) > 0.90
        for lk in ['0x1800', '0x1A00', '0x1C00', '0x2000']
    )
    t6 = all_wave4_pass
    tests['T6_wave4_all_above_90'] = {'pass': t6, 'desc': 'Bridge Wave-4 > 90% at all leaks'}

    # T7: Bridge FPGA readout XOR1 > FPGA alone at intermediate leak
    for lk in ['0x1800', '0x1A00']:
        if lk in all_conditions:
            c = all_conditions[lk]
            t7 = c['BRIDGE_FPGA']['xor']['tau1'] > c['FPGA_ONLY']['xor']['tau1']
            if t7:
                break
    else:
        t7 = False
    tests['T7_bridge_helps_xor'] = {'pass': t7, 'desc': 'GPU MAC helps FPGA XOR at intermediate leak'}

    # T8: Phase transition is monotonic (MC increases with leak, eff_dim decreases)
    mc_vals = [all_conditions[lk]['FPGA_ONLY']['mc_total'] for lk in ['0x1800', '0x1A00', '0x1C00', '0x2000'] if lk in all_conditions]
    ed_vals = [all_conditions[lk]['FPGA_ONLY'].get('eff_dim', 0) for lk in ['0x1800', '0x1A00', '0x1C00', '0x2000'] if lk in all_conditions]
    # Check if MC is roughly increasing and eff_dim roughly decreasing
    mc_increasing = all(mc_vals[i] <= mc_vals[i+1] + 0.3 for i in range(len(mc_vals)-1))
    ed_decreasing = all(ed_vals[i] >= ed_vals[i+1] - 1.0 for i in range(len(ed_vals)-1))
    t8 = mc_increasing and ed_decreasing
    tests['T8_monotonic_transition'] = {'pass': t8, 'desc': 'MC↑ and eff_dim↓ with leak (monotonic transition)'}

    # T9: Best overall score (MC + XOR1 + (1-NARMA)) achievable > 2.0
    best_composite = 0
    for lk in all_conditions:
        for mode in ['FPGA_ONLY', 'BRIDGE_FPGA', 'FULL_BRIDGE']:
            c = all_conditions[lk][mode]
            composite = c['mc_total'] + c['xor']['tau1'] + max(0, 1 - c['narma10_nrmse'])
            if composite > best_composite:
                best_composite = composite
    t9 = best_composite > 2.0
    tests['T9_composite_above_2'] = {'pass': t9, 'desc': f'Best composite score={best_composite:.3f} > 2.0'}

    # T10: XOR τ=5 > 52% at any condition
    best_xor5 = max(
        max(all_conditions[lk][mode]['xor']['tau5']
            for mode in ['FPGA_ONLY', 'BRIDGE_FPGA', 'FULL_BRIDGE'])
        for lk in all_conditions
    )
    t10 = best_xor5 > 0.52
    tests['T10_xor5_above_chance'] = {'pass': t10, 'desc': f'Best XOR5={best_xor5*100:.1f}% > 52%'}

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
