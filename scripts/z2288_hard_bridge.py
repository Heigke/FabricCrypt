#!/usr/bin/env python3
"""
z2288_hard_bridge.py — Hard benchmarks with optimized bridge
=============================================================
z2287: MC=2.628, Wave-4=100%, XOR1=66.1% — all PASS at easy difficulty.
Now push harder: NARMA-5, 7-class waveform, XOR τ=3/5, MC delays 1-10,
and deeper bridge analysis.

Uses: LEAK=0x2000, BIAS=0x4000, THRESH=0x20000, hetero Vg, default synapses.
Three conditions: FPGA_ONLY, GPU_ONLY, BRIDGE (60% input + 40% GPU→MAC).

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2288_hard_bridge.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2288_hard_bridge.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}

N_STEPS = 2500
WARMUP = 400
N_WAVE_TRIALS = 50
N_WAVE_STEPS = 60
TEMP_ABORT = 90.0
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


# ═══════════════════════════════════════════════════════════
# GPU Fourpop ESN (Python version of z2286 HIP kernel)
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# FPGA continuous run
# ═══════════════════════════════════════════════════════════

def fpga_run_continuous(fpga, u, mac_signal=None, sample_hz=SAMPLE_HZ):
    """Run continuous input. mac_signal overrides the default input-based MAC."""
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


# ═══════════════════════════════════════════════════════════
# Benchmark functions
# ═══════════════════════════════════════════════════════════

def ridge_solve(X_tr, y_tr, X_te, y_te, task='regression'):
    """Ridge regression/classification with alpha sweep."""
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_score = 0.0
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
    """Memory capacity at delays 1..max_delay."""
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
    """XOR nonlinearity at multiple delays."""
    xor_results = {}
    n = len(X)
    n_tr = int(0.7 * n)
    for tau in tau_list:
        if WARMUP < tau + 1:
            xor_results[f'xor_tau{tau}'] = 0.5
            continue
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP - tau:len(u_raw) - tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        xor_results[f'xor_tau{tau}'] = acc
    return xor_results


def benchmark_narma5(X, u_raw):
    """NARMA-5 time series prediction."""
    T = len(u_raw)
    u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
    y = np.zeros(T)
    for t in range(5, T):
        y[t] = (0.3 * y[t-1] +
                0.05 * y[t-1] * np.sum(y[t-5:t]) +
                1.5 * u_n[t-1] * u_n[t-5] +
                0.1)
        y[t] = np.tanh(y[t])
    target = y[WARMUP:]
    n = min(len(X), len(target))
    n_tr = int(0.7 * n)
    if n_tr < 50 or n - n_tr < 20:
        return 0.0, 999.0
    # R² score
    r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:n], target[n_tr:n], 'regression')
    # NRMSE
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
    if cls == 0:   return np.sin(t)
    elif cls == 1: return np.sign(np.sin(t))
    elif cls == 2: return 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))) - 1
    elif cls == 3: return 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi))) - 1
    elif cls == 4: return np.sin(t) * np.sin(3 * t)
    elif cls == 5: return np.sign(np.sin(2 * t))
    elif cls == 6: return np.abs(np.sin(t)) * 2 - 1
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


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  z2288: HARD BRIDGE — Push benchmarks beyond easy regime")
    print("  z2287: MC=2.628, Wave-4=100%, XOR1=66.1% at easy difficulty")
    print("=" * 70)

    results = {
        'experiment': 'z2288_hard_bridge',
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

    # Heterogeneous Vg
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    time.sleep(1.0)

    # Default synapses
    for n in range(NUM_NEURONS):
        fpga.set_synapse(n, 0x40408080)
        time.sleep(0.001)
    time.sleep(0.5)

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
    # EXP 1: Continuous benchmarks — FPGA, GPU, BRIDGE
    # ═══════════════════════════════════════════════════════════
    print("\n[EXP 1] Continuous benchmarks (MC d=1..10, XOR τ=1,2,3,5, NARMA-5)")

    conditions = {}

    # GPU run
    print("  GPU fourpop...", end="", flush=True)
    gpu_states = gpu_esn.run(u)
    gpu_X = gpu_states[WARMUP:]
    gpu_mc_tot, gpu_mc_d = benchmark_mc(gpu_X, u, max_delay=10)
    gpu_xor = benchmark_xor(gpu_X, u, [1, 2, 3, 5])
    gpu_narma_r2, gpu_narma_nrmse = benchmark_narma5(gpu_X, u)
    conditions['GPU'] = {
        'mc_total': gpu_mc_tot, 'mc_per_delay': gpu_mc_d,
        'xor': gpu_xor,
        'narma5_r2': gpu_narma_r2, 'narma5_nrmse': gpu_narma_nrmse,
    }
    print(f" MC={gpu_mc_tot:.3f} XOR1={gpu_xor.get('xor_tau1',0)*100:.1f}% NARMA={gpu_narma_nrmse:.3f}")

    # FPGA run
    wait_cool("pre-FPGA")
    print("  FPGA (input→MAC)...", end="", flush=True)
    fpga_s, fpga_ds = fpga_run_continuous(fpga, u)
    fpga_X = build_features(fpga_s, fpga_ds)[WARMUP:]
    fpga_mc_tot, fpga_mc_d = benchmark_mc(fpga_X, u, max_delay=10)
    fpga_xor = benchmark_xor(fpga_X, u, [1, 2, 3, 5])
    fpga_narma_r2, fpga_narma_nrmse = benchmark_narma5(fpga_X, u)
    conditions['FPGA'] = {
        'mc_total': fpga_mc_tot, 'mc_per_delay': fpga_mc_d,
        'xor': fpga_xor,
        'narma5_r2': fpga_narma_r2, 'narma5_nrmse': fpga_narma_nrmse,
    }
    print(f" MC={fpga_mc_tot:.3f} XOR1={fpga_xor.get('xor_tau1',0)*100:.1f}% NARMA={fpga_narma_nrmse:.3f}")

    # BRIDGE run: 60% input + 40% GPU mean(abs(state))
    wait_cool("pre-BRIDGE")
    print("  BRIDGE (60% input + 40% GPU→MAC)...", end="", flush=True)
    gpu_states_bridge = gpu_esn.run(u)
    gpu_mac = np.mean(np.abs(gpu_states_bridge), axis=1)
    gpu_mac_norm = gpu_mac / (gpu_mac.max() + 1e-10)
    bridge_mac = np.clip(0.6 * (u * 0.4 + 0.5) + 0.4 * gpu_mac_norm, 0, 1)

    bridge_s, bridge_ds = fpga_run_continuous(fpga, u, mac_signal=bridge_mac)
    bridge_X_fpga = build_features(bridge_s, bridge_ds)[WARMUP:]
    # Combine FPGA + GPU features
    bridge_X = np.hstack([bridge_X_fpga, gpu_states_bridge[WARMUP:]])

    bridge_mc_tot, bridge_mc_d = benchmark_mc(bridge_X, u, max_delay=10)
    bridge_xor = benchmark_xor(bridge_X, u, [1, 2, 3, 5])
    bridge_narma_r2, bridge_narma_nrmse = benchmark_narma5(bridge_X, u)
    conditions['BRIDGE'] = {
        'mc_total': bridge_mc_tot, 'mc_per_delay': bridge_mc_d,
        'xor': bridge_xor,
        'narma5_r2': bridge_narma_r2, 'narma5_nrmse': bridge_narma_nrmse,
    }
    print(f" MC={bridge_mc_tot:.3f} XOR1={bridge_xor.get('xor_tau1',0)*100:.1f}% NARMA={bridge_narma_nrmse:.3f}")

    results['continuous'] = conditions

    # ═══════════════════════════════════════════════════════════
    # EXP 2: 7-class waveform classification
    # ═══════════════════════════════════════════════════════════
    wait_cool("pre-EXP2")
    print("\n[EXP 2] 7-class waveform classification")

    for cond_name in ['FPGA', 'BRIDGE']:
        feats_list = []
        labels = []
        for trial in range(N_WAVE_TRIALS):
            cls = trial % 7
            wave = generate_waveform(cls, N_WAVE_STEPS)

            if cond_name == 'FPGA':
                mac = np.clip(wave * 0.4 + 0.5, 0, 1)
            else:
                gpu_s = gpu_esn.run(wave)
                gpu_m = np.mean(np.abs(gpu_s), axis=1)
                gpu_m_n = gpu_m / (gpu_m.max() + 1e-10)
                mac = np.clip(0.6 * (wave * 0.4 + 0.5) + 0.4 * gpu_m_n, 0, 1)

            s, ds = fpga_run_continuous(fpga, wave, mac_signal=mac)
            feat = extract_trial_features(s, ds)
            if cond_name == 'BRIDGE':
                gpu_feat = extract_trial_features(gpu_s, np.zeros_like(gpu_s))
                feat = np.concatenate([feat, gpu_feat])
            feats_list.append(feat)
            labels.append(cls)

        X_w = np.array(feats_list)
        y_w = np.array(labels)
        acc, std = ridge_classify(X_w, y_w, n_classes=7)
        results[f'wave7_{cond_name}'] = {'acc': acc, 'std': std}
        print(f"  {cond_name} Wave-7: {acc*100:.1f}% ± {std*100:.1f}%")

        wait_cool(f"post-{cond_name}")

    # GPU-only wave-7
    feats_list = []
    labels = []
    for trial in range(N_WAVE_TRIALS):
        cls = trial % 7
        wave = generate_waveform(cls, N_WAVE_STEPS)
        gpu_s = gpu_esn.run(wave)
        feat = extract_trial_features(gpu_s, np.zeros_like(gpu_s))
        feats_list.append(feat)
        labels.append(cls)
    X_w = np.array(feats_list)
    y_w = np.array(labels)
    acc, std = ridge_classify(X_w, y_w, n_classes=7)
    results['wave7_GPU'] = {'acc': acc, 'std': std}
    print(f"  GPU Wave-7: {acc*100:.1f}% ± {std*100:.1f}%")

    # 4-class waveform
    wait_cool("pre-wave4")
    print("\n  4-class waveform classification")

    for cond_name in ['FPGA', 'BRIDGE']:
        feats_list = []
        labels = []
        for trial in range(N_WAVE_TRIALS):
            cls = trial % 4
            wave = generate_waveform(cls, N_WAVE_STEPS)

            if cond_name == 'FPGA':
                mac = np.clip(wave * 0.4 + 0.5, 0, 1)
            else:
                gpu_s = gpu_esn.run(wave)
                gpu_m = np.mean(np.abs(gpu_s), axis=1)
                gpu_m_n = gpu_m / (gpu_m.max() + 1e-10)
                mac = np.clip(0.6 * (wave * 0.4 + 0.5) + 0.4 * gpu_m_n, 0, 1)

            s, ds = fpga_run_continuous(fpga, wave, mac_signal=mac)
            feat = extract_trial_features(s, ds)
            if cond_name == 'BRIDGE':
                gpu_feat = extract_trial_features(gpu_s, np.zeros_like(gpu_s))
                feat = np.concatenate([feat, gpu_feat])
            feats_list.append(feat)
            labels.append(cls)

        X_w = np.array(feats_list)
        y_w = np.array(labels)
        acc, std = ridge_classify(X_w, y_w, n_classes=4)
        results[f'wave4_{cond_name}'] = {'acc': acc, 'std': std}
        print(f"  {cond_name} Wave-4: {acc*100:.1f}% ± {std*100:.1f}%")

        wait_cool(f"post-{cond_name}")

    # ═══════════════════════════════════════════════════════════
    # EXP 3: Diversity analysis
    # ═══════════════════════════════════════════════════════════
    print("\n[EXP 3] Diversity analysis")
    vm = bridge_s[WARMUP:]
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

    neuron_stds = np.std(vm, axis=0)
    mean_std = float(np.mean(neuron_stds))
    std_std = float(np.std(neuron_stds))

    results['diversity'] = {
        'eff_dim': eff_dim, 'xcorr': xcorr,
        'mean_neuron_std': mean_std, 'std_neuron_std': std_std,
    }
    print(f"  eff_dim={eff_dim:.1f}, xcorr={xcorr:.4f}, neuron_std={mean_std:.4f}±{std_std:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Tests
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  KEY TESTS")
    print("=" * 70)

    tests = {}

    # T1: Bridge MC > 2.0
    t1 = bridge_mc_tot > 2.0
    tests['T1_bridge_mc_above_2'] = {'pass': t1, 'desc': f'BRIDGE MC={bridge_mc_tot:.3f} > 2.0'}
    print(f"  T1 {'PASS' if t1 else 'FAIL'}: {tests['T1_bridge_mc_above_2']['desc']}")

    # T2: Bridge MC > GPU MC
    t2 = bridge_mc_tot > gpu_mc_tot
    tests['T2_bridge_mc_gt_gpu'] = {'pass': t2, 'desc': f'BRIDGE MC={bridge_mc_tot:.3f} > GPU MC={gpu_mc_tot:.3f}'}
    print(f"  T2 {'PASS' if t2 else 'FAIL'}: {tests['T2_bridge_mc_gt_gpu']['desc']}")

    # T3: Bridge MC > FPGA MC
    t3 = bridge_mc_tot > fpga_mc_tot
    tests['T3_bridge_mc_gt_fpga'] = {'pass': t3, 'desc': f'BRIDGE MC={bridge_mc_tot:.3f} > FPGA MC={fpga_mc_tot:.3f}'}
    print(f"  T3 {'PASS' if t3 else 'FAIL'}: {tests['T3_bridge_mc_gt_fpga']['desc']}")

    # T4: XOR τ=1 > 65%
    xor1_br = bridge_xor.get('xor_tau1', 0.5)
    t4 = xor1_br > 0.65
    tests['T4_xor1_above_65'] = {'pass': t4, 'desc': f'BRIDGE XOR1={xor1_br*100:.1f}% > 65%'}
    print(f"  T4 {'PASS' if t4 else 'FAIL'}: {tests['T4_xor1_above_65']['desc']}")

    # T5: XOR τ=3 > 55%
    xor3_br = bridge_xor.get('xor_tau3', 0.5)
    t5 = xor3_br > 0.55
    tests['T5_xor3_above_55'] = {'pass': t5, 'desc': f'BRIDGE XOR3={xor3_br*100:.1f}% > 55%'}
    print(f"  T5 {'PASS' if t5 else 'FAIL'}: {tests['T5_xor3_above_55']['desc']}")

    # T6: XOR τ=5 > 52% (above chance)
    xor5_br = bridge_xor.get('xor_tau5', 0.5)
    t6 = xor5_br > 0.52
    tests['T6_xor5_above_chance'] = {'pass': t6, 'desc': f'BRIDGE XOR5={xor5_br*100:.1f}% > 52%'}
    print(f"  T6 {'PASS' if t6 else 'FAIL'}: {tests['T6_xor5_above_chance']['desc']}")

    # T7: NARMA-5 NRMSE < 1.0 for bridge
    t7 = bridge_narma_nrmse < 1.0
    tests['T7_narma5_useful'] = {'pass': t7, 'desc': f'BRIDGE NARMA5 NRMSE={bridge_narma_nrmse:.3f} < 1.0'}
    print(f"  T7 {'PASS' if t7 else 'FAIL'}: {tests['T7_narma5_useful']['desc']}")

    # T8: NARMA-5 bridge better than FPGA
    t8 = bridge_narma_nrmse < fpga_narma_nrmse
    tests['T8_narma_bridge_gt_fpga'] = {'pass': t8, 'desc': f'BRIDGE NARMA={bridge_narma_nrmse:.3f} < FPGA={fpga_narma_nrmse:.3f}'}
    print(f"  T8 {'PASS' if t8 else 'FAIL'}: {tests['T8_narma_bridge_gt_fpga']['desc']}")

    # T9: Wave-7 bridge > 40% (chance=14.3%)
    w7_br = results.get('wave7_BRIDGE', {}).get('acc', 0)
    t9 = w7_br > 0.40
    tests['T9_wave7_above_40'] = {'pass': t9, 'desc': f'BRIDGE Wave7={w7_br*100:.1f}% > 40%'}
    print(f"  T9 {'PASS' if t9 else 'FAIL'}: {tests['T9_wave7_above_40']['desc']}")

    # T10: Wave-4 bridge > 90%
    w4_br = results.get('wave4_BRIDGE', {}).get('acc', 0)
    t10 = w4_br > 0.90
    tests['T10_wave4_above_90'] = {'pass': t10, 'desc': f'BRIDGE Wave4={w4_br*100:.1f}% > 90%'}
    print(f"  T10 {'PASS' if t10 else 'FAIL'}: {tests['T10_wave4_above_90']['desc']}")

    # T11: Bridge best on ≥3 out of 5 metrics (MC, XOR1, XOR3, NARMA, Wave7)
    bridge_wins = 0
    if bridge_mc_tot > max(fpga_mc_tot, gpu_mc_tot): bridge_wins += 1
    if xor1_br > max(fpga_xor.get('xor_tau1', 0), gpu_xor.get('xor_tau1', 0)): bridge_wins += 1
    if xor3_br > max(fpga_xor.get('xor_tau3', 0), gpu_xor.get('xor_tau3', 0)): bridge_wins += 1
    if bridge_narma_nrmse < min(fpga_narma_nrmse, gpu_narma_nrmse): bridge_wins += 1
    w7_fp = results.get('wave7_FPGA', {}).get('acc', 0)
    w7_gp = results.get('wave7_GPU', {}).get('acc', 0)
    if w7_br > max(w7_fp, w7_gp): bridge_wins += 1
    t11 = bridge_wins >= 3
    tests['T11_bridge_best_3of5'] = {'pass': t11, 'desc': f'BRIDGE best on {bridge_wins}/5 metrics ≥ 3'}
    print(f"  T11 {'PASS' if t11 else 'FAIL'}: {tests['T11_bridge_best_3of5']['desc']}")

    # T12: MC at d=5 > 0.05 for bridge (long-range memory)
    mc_d5 = float(bridge_mc_d.get('5', 0))
    t12 = mc_d5 > 0.05
    tests['T12_mc_d5'] = {'pass': t12, 'desc': f'BRIDGE MC(d=5)={mc_d5:.4f} > 0.05'}
    print(f"  T12 {'PASS' if t12 else 'FAIL'}: {tests['T12_mc_d5']['desc']}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_tests = len(tests)
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
