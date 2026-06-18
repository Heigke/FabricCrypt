#!/usr/bin/env python3
"""
z2289_deeper_bridge.py — Deeper bridge analysis with harder tasks
================================================================
z2288: Wave-7=100% (saturated), FPGA XOR1=74.6% best (bridge dilutes).
Key insight: GPU→MAC drives FPGA, but readout should be FPGA-only for nonlinear tasks.

Plan:
  EXP 1: FPGA-only readout bridge vs full bridge vs FPGA-only (MC, XOR, NARMA-10)
  EXP 2: 15-class waveform (saturated at 7), short windows (30 steps vs 60)
  EXP 3: Transfer entropy GPU→FPGA at multiple lags

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2289_deeper_bridge.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2289_deeper_bridge.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}

N_STEPS = 2500
WARMUP = 400
N_WAVE_TRIALS = 75
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


def benchmark_mc(X, u_raw, max_delay=10):
    mc_per_d = {}
    mc_total = 0.0
    n = len(X)
    n_tr = int(0.7 * n)
    for d in range(1, max_delay + 1):
        target = u_raw[WARMUP - d:len(u_raw) - d]
        nn = min(n, len(target))
        if nn < n_tr + 20:
            mc_per_d[str(d)] = 0.0
            continue
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn], 'regression')
        mc_per_d[str(d)] = r2
        mc_total += r2
    return mc_total, mc_per_d


def benchmark_xor(X, u_raw, tau_list=[1, 2, 3, 5]):
    xor_results = {}
    n = len(X)
    n_tr = int(0.7 * n)
    for tau in tau_list:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP - tau:len(u_raw) - tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor_results[f'xor_tau{tau}'] = acc
    return xor_results


def benchmark_narma(X, u_raw, order=10):
    T = len(u_raw)
    u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
    y = np.zeros(T)
    for t in range(order, T):
        y[t] = (0.3 * y[t-1] +
                0.05 * y[t-1] * np.sum(y[t-order:t]) +
                1.5 * u_n[t-1] * u_n[t-order] +
                0.1)
        y[t] = np.tanh(y[t])
    target = y[WARMUP:]
    n = min(len(X), len(target))
    n_tr = int(0.7 * n)
    if n_tr < 50 or n - n_tr < 20:
        return 0.0, 999.0
    r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:n], target[n_tr:n], 'regression')
    best_nrmse = 999.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        I = np.eye(X[:n_tr].shape[1])
        try:
            w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I, X[:n_tr].T @ target[:n_tr])
            pred = X[n_tr:n] @ w
            gt = target[n_tr:n]
            nrmse = np.sqrt(np.mean((gt - pred)**2)) / (np.std(gt) + 1e-10)
            if nrmse < best_nrmse:
                best_nrmse = nrmse
        except Exception:
            pass
    return r2, best_nrmse


def generate_waveform(cls, steps):
    t = np.linspace(0, 2 * np.pi, steps)
    waveforms = [
        lambda: np.sin(t),                                          # 0: sine
        lambda: np.sign(np.sin(t)),                                  # 1: square
        lambda: 2 * np.abs(2 * (t/(2*np.pi) - np.floor(t/(2*np.pi) + 0.5))) - 1,  # 2: triangle
        lambda: 2 * (t/(2*np.pi) - np.floor(t/(2*np.pi))) - 1,     # 3: sawtooth
        lambda: np.sin(t) * np.sin(3*t),                            # 4: AM
        lambda: np.sign(np.sin(2*t)),                                # 5: double square
        lambda: np.abs(np.sin(t)) * 2 - 1,                          # 6: rectified sine
        lambda: np.sin(t + np.pi/4),                                # 7: phase-shifted sine
        lambda: np.sin(t)**3,                                       # 8: cubic sine
        lambda: np.tanh(3*np.sin(t)),                               # 9: saturated sine
        lambda: np.sin(t) + 0.5*np.sin(2*t),                       # 10: harmonic sum
        lambda: np.sign(np.sin(t)) * np.abs(np.sin(t))**0.5,       # 11: sqrt-square
        lambda: np.sin(t) * np.exp(-0.3*t/(2*np.pi)),              # 12: damped sine
        lambda: np.where(np.sin(t) > 0, np.sin(t), 0),             # 13: half-wave rect
        lambda: np.sin(2*t) * np.cos(t),                            # 14: product wave
    ]
    if cls < len(waveforms):
        return waveforms[cls]()
    return np.sin(t)


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


def transfer_entropy(source, target, lag=1, k=1, bins=8):
    """Estimate transfer entropy from source to target at given lag."""
    n = len(target) - lag - k
    if n < 100:
        return 0.0
    # Discretize
    s = np.digitize(source[:n], np.linspace(source.min(), source.max() + 1e-10, bins + 1)) - 1
    t_past = np.digitize(target[lag:lag+n], np.linspace(target.min(), target.max() + 1e-10, bins + 1)) - 1
    t_fut = np.digitize(target[lag+k:lag+k+n], np.linspace(target.min(), target.max() + 1e-10, bins + 1)) - 1

    # Joint probabilities via counting
    from collections import Counter
    joint_stp = Counter(zip(s, t_past, t_fut))
    joint_tp = Counter(zip(t_past, t_fut))
    joint_st = Counter(zip(s, t_past))
    marg_t = Counter(t_past)

    te = 0.0
    total = float(n)
    for (si, tpi, tfi), count in joint_stp.items():
        p_stp = count / total
        p_tp = joint_tp[(tpi, tfi)] / total
        p_st = joint_st[(si, tpi)] / total
        p_t = marg_t[tpi] / total
        if p_stp > 0 and p_t > 0 and p_tp > 0 and p_st > 0:
            te += p_stp * np.log2((p_stp * p_t) / (p_tp * p_st + 1e-30) + 1e-30)
    return max(0.0, te)


def main():
    print("=" * 70)
    print("  z2289: DEEPER BRIDGE — FPGA-only readout, 15-class, NARMA-10, TE")
    print("  z2288: Wave-7=100% (saturated), FPGA XOR=74.6% (best)")
    print("=" * 70)

    results = {
        'experiment': 'z2289_deeper_bridge',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    fpga = FPGAEthBridge()
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(0.3)

    # Optimal parameters from z2287
    fpga.set_leak_cond(0x2000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    fpga.set_threshold_raw(0x20000)

    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    for n in range(NUM_NEURONS):
        fpga.set_synapse(n, 0x40408080)
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

    # ═══════════════════════════════════════════════════════════
    # EXP 1: Three readout modes — FPGA_ONLY, BRIDGE_FPGA_READOUT, FULL_BRIDGE
    # ═══════════════════════════════════════════════════════════
    print("\n[EXP 1] Three readout modes (MC d=1..10, XOR τ=1,2,3,5, NARMA-10)")

    # Condition A: FPGA only (input→MAC directly)
    print("  A) FPGA_ONLY...", end="", flush=True)
    fpga_s, fpga_ds = fpga_run_continuous(fpga, u)
    fpga_X = build_features(fpga_s, fpga_ds)[WARMUP:]
    fpga_mc, fpga_mc_d = benchmark_mc(fpga_X, u)
    fpga_xor = benchmark_xor(fpga_X, u, [1, 2, 3, 5])
    fpga_narma_r2, fpga_narma_nrmse = benchmark_narma(fpga_X, u, order=10)
    print(f" MC={fpga_mc:.3f} XOR1={fpga_xor['xor_tau1']*100:.1f}% NARMA10={fpga_narma_nrmse:.3f}")

    # Condition B: GPU→MAC bridge, FPGA-only readout
    wait_cool("pre-B")
    print("  B) BRIDGE_FPGA_READOUT (GPU→MAC, read FPGA only)...", end="", flush=True)
    gpu_states_b = gpu_esn.run(u)
    gpu_mac = np.mean(np.abs(gpu_states_b), axis=1)
    gpu_mac_norm = gpu_mac / (gpu_mac.max() + 1e-10)
    bridge_mac = np.clip(0.6 * (u * 0.4 + 0.5) + 0.4 * gpu_mac_norm, 0, 1)
    br_s, br_ds = fpga_run_continuous(fpga, u, mac_signal=bridge_mac)
    br_fpga_X = build_features(br_s, br_ds)[WARMUP:]
    br_fpga_mc, br_fpga_mc_d = benchmark_mc(br_fpga_X, u)
    br_fpga_xor = benchmark_xor(br_fpga_X, u, [1, 2, 3, 5])
    br_fpga_narma_r2, br_fpga_narma_nrmse = benchmark_narma(br_fpga_X, u, order=10)
    print(f" MC={br_fpga_mc:.3f} XOR1={br_fpga_xor['xor_tau1']*100:.1f}% NARMA10={br_fpga_narma_nrmse:.3f}")

    # Condition C: Full bridge (GPU→MAC + GPU+FPGA readout)
    wait_cool("pre-C")
    print("  C) FULL_BRIDGE (GPU→MAC, read FPGA+GPU)...", end="", flush=True)
    br_full_X = np.hstack([br_fpga_X, gpu_states_b[WARMUP:]])
    br_full_mc, br_full_mc_d = benchmark_mc(br_full_X, u)
    br_full_xor = benchmark_xor(br_full_X, u, [1, 2, 3, 5])
    br_full_narma_r2, br_full_narma_nrmse = benchmark_narma(br_full_X, u, order=10)
    print(f" MC={br_full_mc:.3f} XOR1={br_full_xor['xor_tau1']*100:.1f}% NARMA10={br_full_narma_nrmse:.3f}")

    # GPU only
    gpu_X = gpu_states_b[WARMUP:]
    gpu_mc, gpu_mc_d = benchmark_mc(gpu_X, u)
    gpu_xor = benchmark_xor(gpu_X, u, [1, 2, 3, 5])
    gpu_narma_r2, gpu_narma_nrmse = benchmark_narma(gpu_X, u, order=10)
    print(f"  D) GPU_ONLY: MC={gpu_mc:.3f} XOR1={gpu_xor['xor_tau1']*100:.1f}% NARMA10={gpu_narma_nrmse:.3f}")

    results['exp1'] = {
        'FPGA_ONLY': {
            'mc_total': fpga_mc, 'mc_per_delay': fpga_mc_d,
            'xor': fpga_xor, 'narma10_nrmse': fpga_narma_nrmse,
        },
        'BRIDGE_FPGA_READOUT': {
            'mc_total': br_fpga_mc, 'mc_per_delay': br_fpga_mc_d,
            'xor': br_fpga_xor, 'narma10_nrmse': br_fpga_narma_nrmse,
        },
        'FULL_BRIDGE': {
            'mc_total': br_full_mc, 'mc_per_delay': br_full_mc_d,
            'xor': br_full_xor, 'narma10_nrmse': br_full_narma_nrmse,
        },
        'GPU_ONLY': {
            'mc_total': gpu_mc, 'mc_per_delay': gpu_mc_d,
            'xor': gpu_xor, 'narma10_nrmse': gpu_narma_nrmse,
        },
    }

    # ═══════════════════════════════════════════════════════════
    # EXP 2: 15-class waveform (short 30-step windows)
    # ═══════════════════════════════════════════════════════════
    wait_cool("pre-EXP2")
    print(f"\n[EXP 2] 15-class waveform (30-step window, {N_WAVE_TRIALS} trials)")

    for cond_name in ['FPGA', 'BRIDGE_FPGA', 'FULL_BRIDGE', 'GPU']:
        feats_list = []
        labels = []
        for trial in range(N_WAVE_TRIALS):
            cls = trial % 15
            wave = generate_waveform(cls, 30)

            if cond_name == 'GPU':
                gpu_s = gpu_esn.run(wave)
                feat = extract_trial_features(gpu_s, np.zeros_like(gpu_s))
            elif cond_name == 'FPGA':
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
                if cond_name == 'FULL_BRIDGE':
                    gpu_feat = extract_trial_features(gpu_s, np.zeros_like(gpu_s))
                    feat = np.concatenate([feat, gpu_feat])

            feats_list.append(feat)
            labels.append(cls)

        X_w = np.array(feats_list)
        y_w = np.array(labels)
        acc, std = ridge_classify(X_w, y_w, n_classes=15, n_splits=5)
        results[f'wave15_{cond_name}'] = {'acc': acc, 'std': std}
        print(f"  {cond_name} Wave-15: {acc*100:.1f}% ± {std*100:.1f}%")

        if cond_name != 'GPU':
            wait_cool(f"post-{cond_name}")

    # ═══════════════════════════════════════════════════════════
    # EXP 3: Transfer entropy GPU→FPGA
    # ═══════════════════════════════════════════════════════════
    print("\n[EXP 3] Transfer entropy GPU → FPGA at multiple lags")

    # Use bridge run data
    gpu_signal = np.mean(gpu_states_b, axis=1)
    fpga_signal = np.mean(br_s, axis=1)

    te_results = {}
    for lag in [1, 2, 5, 10, 20, 50]:
        te_gf = transfer_entropy(gpu_signal, fpga_signal, lag=lag)
        te_fg = transfer_entropy(fpga_signal, gpu_signal, lag=lag)
        te_results[f'lag_{lag}'] = {
            'gpu_to_fpga': te_gf,
            'fpga_to_gpu': te_fg,
            'net_flow': te_gf - te_fg,
        }
        direction = "GPU→FPGA" if te_gf > te_fg else "FPGA→GPU"
        print(f"  lag={lag:2d}: TE(GPU→FPGA)={te_gf:.4f} TE(FPGA→GPU)={te_fg:.4f} net={te_gf-te_fg:+.4f} [{direction}]")

    results['transfer_entropy'] = te_results

    # ═══════════════════════════════════════════════════════════
    # Tests
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  KEY TESTS")
    print("=" * 70)

    tests = {}

    # T1: BRIDGE_FPGA_READOUT XOR1 > FPGA_ONLY XOR1 (GPU MAC helps nonlinearity)
    t1 = br_fpga_xor['xor_tau1'] > fpga_xor['xor_tau1']
    tests['T1_bridge_fpga_xor_gt_fpga'] = {
        'pass': t1,
        'desc': f'BR_FPGA XOR1={br_fpga_xor["xor_tau1"]*100:.1f}% > FPGA={fpga_xor["xor_tau1"]*100:.1f}%'
    }

    # T2: BRIDGE_FPGA_READOUT MC > GPU MC
    t2 = br_fpga_mc > gpu_mc
    tests['T2_bridge_fpga_mc_gt_gpu'] = {
        'pass': t2,
        'desc': f'BR_FPGA MC={br_fpga_mc:.3f} > GPU={gpu_mc:.3f}'
    }

    # T3: FULL_BRIDGE MC > 2.0
    t3 = br_full_mc > 2.0
    tests['T3_full_bridge_mc'] = {
        'pass': t3,
        'desc': f'FULL MC={br_full_mc:.3f} > 2.0'
    }

    # T4: NARMA-10 < 1.0 for bridge
    t4 = br_fpga_narma_nrmse < 1.0
    tests['T4_narma10_useful'] = {
        'pass': t4,
        'desc': f'BR_FPGA NARMA10={br_fpga_narma_nrmse:.3f} < 1.0'
    }

    # T5: NARMA-10 bridge < FPGA alone
    t5 = br_fpga_narma_nrmse < fpga_narma_nrmse
    tests['T5_narma10_bridge_lt_fpga'] = {
        'pass': t5,
        'desc': f'BR_FPGA NARMA10={br_fpga_narma_nrmse:.3f} < FPGA={fpga_narma_nrmse:.3f}'
    }

    # T6: XOR τ=3 > 55% for any condition
    best_xor3 = max(
        fpga_xor.get('xor_tau3', 0.5),
        br_fpga_xor.get('xor_tau3', 0.5),
        br_full_xor.get('xor_tau3', 0.5)
    )
    t6 = best_xor3 > 0.55
    tests['T6_xor3_above_55'] = {
        'pass': t6,
        'desc': f'Best XOR3={best_xor3*100:.1f}% > 55%'
    }

    # T7: Wave-15 bridge > 50% (chance=6.7%)
    w15_br = results.get('wave15_FULL_BRIDGE', {}).get('acc', 0)
    t7 = w15_br > 0.50
    tests['T7_wave15_above_50'] = {
        'pass': t7,
        'desc': f'FULL_BRIDGE Wave15={w15_br*100:.1f}% > 50%'
    }

    # T8: Wave-15 bridge > FPGA alone
    w15_fp = results.get('wave15_FPGA', {}).get('acc', 0)
    t8 = w15_br > w15_fp
    tests['T8_wave15_bridge_gt_fpga'] = {
        'pass': t8,
        'desc': f'BRIDGE Wave15={w15_br*100:.1f}% > FPGA={w15_fp*100:.1f}%'
    }

    # T9: TE GPU→FPGA > 0.05 bits at lag=1
    te_lag1 = te_results.get('lag_1', {}).get('gpu_to_fpga', 0)
    t9 = te_lag1 > 0.05
    tests['T9_te_above_005'] = {
        'pass': t9,
        'desc': f'TE(GPU→FPGA, lag=1)={te_lag1:.4f} > 0.05'
    }

    # T10: Net TE flow is GPU→FPGA at lag=1
    net_lag1 = te_results.get('lag_1', {}).get('net_flow', 0)
    t10 = net_lag1 > 0
    tests['T10_net_te_gpu_to_fpga'] = {
        'pass': t10,
        'desc': f'Net TE(lag=1)={net_lag1:+.4f} > 0 (GPU→FPGA)'
    }

    # T11: BRIDGE_FPGA_READOUT best on ≥2 of {XOR1, MC, NARMA10} vs FPGA_ONLY
    wins = 0
    if br_fpga_xor['xor_tau1'] > fpga_xor['xor_tau1']: wins += 1
    if br_fpga_mc > fpga_mc: wins += 1
    if br_fpga_narma_nrmse < fpga_narma_nrmse: wins += 1
    t11 = wins >= 2
    tests['T11_bridge_fpga_wins_2of3'] = {
        'pass': t11,
        'desc': f'BR_FPGA wins {wins}/3 vs FPGA_ONLY ≥ 2'
    }

    # T12: Any XOR τ=5 > 52%
    best_xor5 = max(
        fpga_xor.get('xor_tau5', 0.5),
        br_fpga_xor.get('xor_tau5', 0.5),
        br_full_xor.get('xor_tau5', 0.5)
    )
    t12 = best_xor5 > 0.52
    tests['T12_xor5_above_chance'] = {
        'pass': t12,
        'desc': f'Best XOR5={best_xor5*100:.1f}% > 52%'
    }

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_tests = len(tests)

    for k, v in tests.items():
        print(f"  {k.split('_', 1)[0]} {'PASS' if v['pass'] else 'FAIL'}: {v['desc']}")

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
