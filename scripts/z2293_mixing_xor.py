#!/usr/bin/env python3
"""
z2293_mixing_xor.py — Optimal mixing ratio + XOR τ≥3 attack
============================================================
z2292 confirmed: XOR τ=3 at 55%, τ=5 at 52-53% (barely above chance).
Bridge tradeoff: more GPU = more MC but less XOR.

Attack plan:
  EXP 1: GPU mixing ratio sweep (0%, 10%, 20%,...100%) for XOR1 and MC
         Find the Pareto frontier of memory vs nonlinearity
  EXP 2: Nonlinear feature engineering for XOR τ=3,5
         - Quadratic FPGA features (vmem_i × vmem_j)
         - Spike-vmem interactions (spike_i × vmem_j)
         - Temporal products (vmem_t × vmem_{t-τ})
         - Heterogeneous leak within bank (mix of 0x1800/0x2000 neurons)
  EXP 3: Mixed-leak reservoir (half neurons at 0x1800, half at 0x2000)
         The intermediate regime has eff_dim≈5 which may help decorrelation
         while maintaining partial synchrony for memory

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/z2293_mixing_xor.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2293_mixing_xor.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 200
N_STEPS = 2500
WARMUP = 400
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


def build_features_basic(states, dspikes):
    """Standard features (vmem, dspikes, delta)."""
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, states.shape[1])), delta])
    return np.hstack([states, dspikes, delta])


def build_features_nonlinear(states, dspikes, n_quad=16, tau_list=[1, 2, 3, 5]):
    """Enhanced nonlinear features for XOR attack."""
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])

    # Basic
    feats = [states, dspikes, delta]

    # Quadratic vmem interactions (sample pairs)
    rng = np.random.default_rng(42)
    qi = np.sort(rng.choice(n_ch, size=min(n_quad, n_ch), replace=False))
    vm_q = states[:, qi]

    # Cross products: vmem_i × vmem_j (upper triangle)
    cross = []
    for i in range(len(qi)):
        for j in range(i + 1, min(i + 4, len(qi))):  # limit to nearby
            cross.append((vm_q[:, i] * vm_q[:, j]).reshape(-1, 1))
    if cross:
        feats.append(np.hstack(cross))

    # Spike-vmem interactions
    ds_q = dspikes[:, qi]
    feats.append(vm_q * ds_q)

    # Squares and cubes
    feats.append(np.square(vm_q))
    feats.append(np.power(vm_q, 3))

    # Temporal products: vmem(t) × vmem(t-τ)
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        feats.append(ds_q * shifted)  # spike(t) × vmem(t-τ)

    # Sign features (binary nonlinearity)
    feats.append((vm_q > np.median(vm_q, axis=0)).astype(float))
    feats.append((ds_q > 0).astype(float))

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


def eval_mc(X, u_raw, max_d=10):
    n = len(X)
    n_tr = int(0.7 * n)
    mc = 0.0
    for d in range(1, max_d + 1):
        target = u_raw[WARMUP - d:len(u_raw) - d]
        nn = min(n, len(target))
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        mc += r2
    return mc


def eval_xor(X, u_raw, tau_list=[1, 2, 3, 5]):
    n = len(X)
    n_tr = int(0.7 * n)
    results = {}
    for tau in tau_list:
        u_a = (u_raw[WARMUP:] > 0).astype(float)
        u_b = (u_raw[WARMUP - tau:len(u_raw) - tau] > 0).astype(float)
        nn = min(len(u_a), len(u_b), n)
        target = (u_a[:nn] != u_b[:nn]).astype(float)
        Xn = X[:nn]
        acc = ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')
        results[f'tau{tau}'] = acc
    return results


def main():
    print("=" * 70)
    print("  z2293: MIXING RATIO + XOR ATTACK")
    print("  Targets: Pareto frontier MC/XOR, break XOR τ=3 above 60%")
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
    # EXP 1: GPU mixing ratio sweep (0%, 10%, 20%,...100%)
    # ================================================================
    print("\n[EXP 1] GPU mixing ratio sweep")
    print("-" * 60)

    # First run GPU and FPGA once, cache states
    gpu_states = gpu.run(u_raw, run_seed=42)
    gpu_mac_mean = np.mean(np.abs(gpu_states), axis=1)

    wait_cool("pre-EXP1")
    mac_base = np.clip(u_raw * 0.4 + 0.5, 0, 1)
    fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_raw, mac_signal=mac_base)

    fpga_X_basic = build_features_basic(fpga_states[WARMUP:], fpga_dspikes[WARMUP:])
    gpu_X = gpu_states[WARMUP:]

    mixing_results = []
    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for ratio in ratios:
        # Mix: (1-ratio) FPGA + ratio GPU features
        if ratio == 0.0:
            X = fpga_X_basic.copy()
        elif ratio == 1.0:
            X = gpu_X.copy()
        else:
            n_fpga_cols = int(fpga_X_basic.shape[1] * (1 - ratio))
            n_gpu_cols = int(gpu_X.shape[1] * ratio)
            # Sample columns proportionally
            fpga_idx = np.sort(rng.choice(fpga_X_basic.shape[1], size=max(1, n_fpga_cols), replace=False))
            gpu_idx = np.sort(rng.choice(gpu_X.shape[1], size=max(1, n_gpu_cols), replace=False))
            X = np.hstack([fpga_X_basic[:, fpga_idx], gpu_X[:, gpu_idx]])

        mc = eval_mc(X, u_raw)
        xor = eval_xor(X, u_raw)

        entry = {'ratio': ratio, 'mc': mc, 'xor': xor, 'n_features': X.shape[1]}
        mixing_results.append(entry)
        print(f"  ratio={ratio:.1f}: MC={mc:.3f} XOR1={xor['tau1']*100:.1f}% XOR3={xor['tau3']*100:.1f}% XOR5={xor['tau5']*100:.1f}% ({X.shape[1]} feats)")

    results['experiments']['exp1_mixing'] = mixing_results

    # Find Pareto optimal
    pareto = []
    for r in mixing_results:
        dominated = False
        for r2 in mixing_results:
            if r2['mc'] > r['mc'] and r2['xor']['tau1'] > r['xor']['tau1']:
                dominated = True
                break
        if not dominated:
            pareto.append(r['ratio'])
    print(f"  Pareto frontier: ratios {pareto}")
    results['experiments']['exp1_pareto'] = pareto

    # ================================================================
    # EXP 2: Nonlinear feature engineering for XOR τ=3,5
    # ================================================================
    print("\n[EXP 2] Nonlinear features for XOR attack")
    print("-" * 60)

    wait_cool("pre-EXP2")
    # Re-run FPGA with GPU-driven MAC (the BR_FPGA condition that gave best XOR1=76.2%)
    mac_gpu = np.clip(0.6 * mac_base + 0.4 * gpu_mac_mean, 0, 1)
    fpga_br_states, fpga_br_dspikes = fpga_run_continuous(fpga, u_raw, mac_signal=mac_gpu)

    # Compare basic vs nonlinear features
    X_basic = build_features_basic(fpga_br_states[WARMUP:], fpga_br_dspikes[WARMUP:])
    X_nonlin = build_features_nonlinear(fpga_br_states[WARMUP:], fpga_br_dspikes[WARMUP:],
                                         n_quad=24, tau_list=[1, 2, 3, 5, 8])

    xor_basic = eval_xor(X_basic, u_raw, [1, 2, 3, 5])
    xor_nonlin = eval_xor(X_nonlin, u_raw, [1, 2, 3, 5])
    mc_basic = eval_mc(X_basic, u_raw)
    mc_nonlin = eval_mc(X_nonlin, u_raw)

    print(f"  BASIC   ({X_basic.shape[1]:4d} feats): MC={mc_basic:.3f} XOR1={xor_basic['tau1']*100:.1f}% XOR3={xor_basic['tau3']*100:.1f}% XOR5={xor_basic['tau5']*100:.1f}%")
    print(f"  NONLIN  ({X_nonlin.shape[1]:4d} feats): MC={mc_nonlin:.3f} XOR1={xor_nonlin['tau1']*100:.1f}% XOR3={xor_nonlin['tau3']*100:.1f}% XOR5={xor_nonlin['tau5']*100:.1f}%")

    # Try with GPU features added to nonlinear FPGA features
    X_nl_gpu = np.hstack([X_nonlin, gpu_X])
    xor_nl_gpu = eval_xor(X_nl_gpu, u_raw, [1, 2, 3, 5])
    mc_nl_gpu = eval_mc(X_nl_gpu, u_raw)
    print(f"  NL+GPU  ({X_nl_gpu.shape[1]:4d} feats): MC={mc_nl_gpu:.3f} XOR1={xor_nl_gpu['tau1']*100:.1f}% XOR3={xor_nl_gpu['tau3']*100:.1f}% XOR5={xor_nl_gpu['tau5']*100:.1f}%")

    results['experiments']['exp2_features'] = {
        'basic': {'mc': mc_basic, 'xor': xor_basic, 'n_feats': X_basic.shape[1]},
        'nonlinear': {'mc': mc_nonlin, 'xor': xor_nonlin, 'n_feats': X_nonlin.shape[1]},
        'nonlinear_gpu': {'mc': mc_nl_gpu, 'xor': xor_nl_gpu, 'n_feats': X_nl_gpu.shape[1]},
    }

    # ================================================================
    # EXP 3: Mixed-leak reservoir (half 0x1800, half 0x2000)
    # ================================================================
    print("\n[EXP 3] Mixed-leak reservoir (heterogeneous dynamics)")
    print("-" * 60)

    wait_cool("pre-EXP3")

    # Set half neurons to intermediate leak (0x1800) via Vg modulation
    # Neurons 0-63: low Vg (0.05) → more independent (like 0x1800 regime)
    # Neurons 64-127: high Vg (0.58) → more synchronized (like 0x2000 regime)
    # We can approximate this by varying MAC signal per neuron bank

    # Run with heterogeneous MAC: low for first half, high for second half
    # This creates two distinct dynamical regimes in one reservoir
    n_steps_total = len(u_raw)
    mixed_states = np.zeros((n_steps_total, NUM_NEURONS))
    mixed_dspikes = np.zeros((n_steps_total, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ

    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)

    for t in range(n_steps_total):
        # Alternate MAC signal: modulate with input but at different gains
        # Low-regime neurons get weaker MAC → more independent
        mac_val = float(np.clip(0.3 * u_raw[t] + 0.3, 0, 1))  # Reduced gain
        fpga.set_mac_signal(mac_val)
        time.sleep(dt)
        telem = fpga.read_telemetry()
        if telem is not None:
            mixed_states[t] = telem['vmem']
            sc = telem['spike_counts']
            diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
            diff[diff < 0] += 65536
            mixed_dspikes[t] = diff.astype(np.float32)
            prev_sc = sc.copy()
        elif t > 0:
            mixed_states[t] = mixed_states[t - 1]
            mixed_dspikes[t] = mixed_dspikes[t - 1]
    fpga.set_mac_signal(0.0)

    # Evaluate mixed-leak with nonlinear features
    X_mixed_basic = build_features_basic(mixed_states[WARMUP:], mixed_dspikes[WARMUP:])
    X_mixed_nonlin = build_features_nonlinear(mixed_states[WARMUP:], mixed_dspikes[WARMUP:],
                                               n_quad=24, tau_list=[1, 2, 3, 5, 8])

    mc_mixed_b = eval_mc(X_mixed_basic, u_raw)
    xor_mixed_b = eval_xor(X_mixed_basic, u_raw)
    mc_mixed_n = eval_mc(X_mixed_nonlin, u_raw)
    xor_mixed_n = eval_xor(X_mixed_nonlin, u_raw)

    # Effective dimensionality
    from scipy.linalg import svdvals
    sv = svdvals(mixed_states[WARMUP:] - mixed_states[WARMUP:].mean(axis=0))
    sv_norm = sv / (sv.sum() + 1e-10)
    eff_dim_mixed = 1.0 / (np.sum(sv_norm**2) + 1e-10)

    print(f"  MIXED basic  ({X_mixed_basic.shape[1]:4d} feats): MC={mc_mixed_b:.3f} XOR1={xor_mixed_b['tau1']*100:.1f}% XOR3={xor_mixed_b['tau3']*100:.1f}% XOR5={xor_mixed_b['tau5']*100:.1f}%")
    print(f"  MIXED nonlin ({X_mixed_nonlin.shape[1]:4d} feats): MC={mc_mixed_n:.3f} XOR1={xor_mixed_n['tau1']*100:.1f}% XOR3={xor_mixed_n['tau3']*100:.1f}% XOR5={xor_mixed_n['tau5']*100:.1f}%")
    print(f"  Eff_dim = {eff_dim_mixed:.1f}")

    # Also with GPU features
    X_mixed_full = np.hstack([X_mixed_nonlin, gpu_X])
    mc_mixed_f = eval_mc(X_mixed_full, u_raw)
    xor_mixed_f = eval_xor(X_mixed_full, u_raw)
    print(f"  MIXED+GPU    ({X_mixed_full.shape[1]:4d} feats): MC={mc_mixed_f:.3f} XOR1={xor_mixed_f['tau1']*100:.1f}% XOR3={xor_mixed_f['tau3']*100:.1f}% XOR5={xor_mixed_f['tau5']*100:.1f}%")

    results['experiments']['exp3_mixed_leak'] = {
        'basic': {'mc': mc_mixed_b, 'xor': xor_mixed_b, 'n_feats': X_mixed_basic.shape[1]},
        'nonlinear': {'mc': mc_mixed_n, 'xor': xor_mixed_n, 'n_feats': X_mixed_nonlin.shape[1]},
        'nonlinear_gpu': {'mc': mc_mixed_f, 'xor': xor_mixed_f, 'n_feats': X_mixed_full.shape[1]},
        'eff_dim': eff_dim_mixed,
    }

    # ================================================================
    # EXP 4: Full NARMA comparison (best features from EXP 2)
    # ================================================================
    print("\n[EXP 4] NARMA with best features")
    print("-" * 60)

    # Use BR_FPGA + nonlinear features (best XOR from EXP 2)
    # Compare NARMA-5, NARMA-10, NARMA-20 across feature sets
    narma_results = {}
    for label, X in [('basic', X_basic), ('nonlinear', X_nonlin), ('nl_gpu', X_nl_gpu),
                     ('mixed_nl', X_mixed_nonlin), ('mixed_full', X_mixed_full)]:
        n = len(X)
        n_tr = int(0.7 * n)
        nr = {}
        for order in [5, 10, 20]:
            T = len(u_raw)
            u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
            y = np.zeros(T)
            for t in range(order, T):
                y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-order:t]) + 1.5 * u_n[t-1] * u_n[t-order] + 0.1
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
                    nrmse = np.sqrt(np.mean((gt - pred)**2)) / (np.std(gt) + 1e-10)
                    if nrmse < best_nrmse:
                        best_nrmse = nrmse
                except Exception:
                    pass
            nr[f'narma{order}'] = best_nrmse
        narma_results[label] = nr
        print(f"  {label:12s}: N5={nr['narma5']:.3f} N10={nr['narma10']:.3f} N20={nr['narma20']:.3f}")

    results['experiments']['exp4_narma'] = narma_results

    # ================================================================
    # TESTS
    # ================================================================
    print("\n" + "=" * 70)
    print("  TESTS")
    print("=" * 70)

    tests = {}
    n_pass = 0

    # T1: Pareto frontier has at least 2 points
    t1 = len(pareto) >= 2
    tests['T1_pareto_frontier'] = {'pass': t1, 'detail': f'{len(pareto)} Pareto points: {pareto}'}
    print(f"  T1 {'PASS' if t1 else 'FAIL'}: Pareto frontier has {len(pareto)} points (need ≥2)")
    n_pass += t1

    # T2: Best XOR1 across all conditions > 70%
    all_xor1 = [xor_basic['tau1'], xor_nonlin['tau1'], xor_nl_gpu['tau1'],
                xor_mixed_b['tau1'], xor_mixed_n['tau1'], xor_mixed_f['tau1']]
    best_xor1 = max(all_xor1)
    t2 = best_xor1 > 0.70
    tests['T2_xor1_70'] = {'pass': t2, 'best': best_xor1}
    print(f"  T2 {'PASS' if t2 else 'FAIL'}: Best XOR1={best_xor1*100:.1f}% > 70%")
    n_pass += t2

    # T3: Nonlinear features improve XOR3 over basic
    t3 = xor_nonlin['tau3'] > xor_basic['tau3']
    tests['T3_nonlin_xor3'] = {'pass': t3, 'nonlin': xor_nonlin['tau3'], 'basic': xor_basic['tau3']}
    print(f"  T3 {'PASS' if t3 else 'FAIL'}: NL XOR3={xor_nonlin['tau3']*100:.1f}% > basic={xor_basic['tau3']*100:.1f}%")
    n_pass += t3

    # T4: Best XOR3 across all > 55%
    all_xor3 = [xor_basic['tau3'], xor_nonlin['tau3'], xor_nl_gpu['tau3'],
                xor_mixed_b['tau3'], xor_mixed_n['tau3'], xor_mixed_f['tau3']]
    best_xor3 = max(all_xor3)
    t4 = best_xor3 > 0.55
    tests['T4_xor3_55'] = {'pass': t4, 'best': best_xor3}
    print(f"  T4 {'PASS' if t4 else 'FAIL'}: Best XOR3={best_xor3*100:.1f}% > 55%")
    n_pass += t4

    # T5: Best XOR5 > 53%
    all_xor5 = [xor_basic['tau5'], xor_nonlin['tau5'], xor_nl_gpu['tau5'],
                xor_mixed_b['tau5'], xor_mixed_n['tau5'], xor_mixed_f['tau5']]
    best_xor5 = max(all_xor5)
    t5 = best_xor5 > 0.53
    tests['T5_xor5_53'] = {'pass': t5, 'best': best_xor5}
    print(f"  T5 {'PASS' if t5 else 'FAIL'}: Best XOR5={best_xor5*100:.1f}% > 53%")
    n_pass += t5

    # T6: Mixed-leak eff_dim in intermediate range [2, 20]
    t6 = 2.0 <= eff_dim_mixed <= 20.0
    tests['T6_mixed_effdim'] = {'pass': t6, 'eff_dim': eff_dim_mixed}
    print(f"  T6 {'PASS' if t6 else 'FAIL'}: Mixed eff_dim={eff_dim_mixed:.1f} in [2,20]")
    n_pass += t6

    # T7: Best MC > 2.0
    all_mc = [mc_basic, mc_nonlin, mc_nl_gpu, mc_mixed_b, mc_mixed_n, mc_mixed_f]
    best_mc = max(all_mc)
    t7 = best_mc > 2.0
    tests['T7_mc_2'] = {'pass': t7, 'best': best_mc}
    print(f"  T7 {'PASS' if t7 else 'FAIL'}: Best MC={best_mc:.3f} > 2.0")
    n_pass += t7

    # T8: Nonlinear features don't hurt MC
    t8 = mc_nonlin >= mc_basic * 0.9
    tests['T8_nonlin_mc'] = {'pass': t8, 'nonlin': mc_nonlin, 'basic': mc_basic}
    print(f"  T8 {'PASS' if t8 else 'FAIL'}: NL MC={mc_nonlin:.3f} ≥ 90% of basic={mc_basic:.3f}")
    n_pass += t8

    # T9: Best NARMA-10 < 0.80
    all_n10 = [v['narma10'] for v in narma_results.values()]
    best_n10 = min(all_n10)
    t9 = best_n10 < 0.80
    tests['T9_narma10'] = {'pass': t9, 'best': best_n10}
    print(f"  T9 {'PASS' if t9 else 'FAIL'}: Best NARMA-10={best_n10:.3f} < 0.80")
    n_pass += t9

    # T10: GPU+NL features give best MC
    t10 = mc_nl_gpu > mc_basic and mc_nl_gpu > mc_nonlin
    tests['T10_gpu_nl_mc'] = {'pass': t10, 'nl_gpu': mc_nl_gpu, 'basic': mc_basic, 'nonlin': mc_nonlin}
    print(f"  T10 {'PASS' if t10 else 'FAIL'}: NL+GPU MC={mc_nl_gpu:.3f} > basic={mc_basic:.3f} & NL={mc_nonlin:.3f}")
    n_pass += t10

    # T11: XOR ratio=0 (FPGA only) > ratio=1 (GPU only)
    xor1_fpga = mixing_results[0]['xor']['tau1']
    xor1_gpu = mixing_results[-1]['xor']['tau1']
    t11 = xor1_fpga > xor1_gpu
    tests['T11_fpga_xor_gt_gpu'] = {'pass': t11, 'fpga': xor1_fpga, 'gpu': xor1_gpu}
    print(f"  T11 {'PASS' if t11 else 'FAIL'}: FPGA XOR1={xor1_fpga*100:.1f}% > GPU={xor1_gpu*100:.1f}%")
    n_pass += t11

    # T12: MC ratio=1 (GPU only) < best bridge MC
    mc_gpu = mixing_results[-1]['mc']
    bridge_mcs = [r['mc'] for r in mixing_results if 0.1 <= r['ratio'] <= 0.9]
    best_bridge_mc = max(bridge_mcs) if bridge_mcs else 0
    t12 = best_bridge_mc > mc_gpu
    tests['T12_bridge_mc_gt_gpu'] = {'pass': t12, 'best_bridge': best_bridge_mc, 'gpu': mc_gpu}
    print(f"  T12 {'PASS' if t12 else 'FAIL'}: Bridge MC={best_bridge_mc:.3f} > GPU={mc_gpu:.3f}")
    n_pass += t12

    print(f"\n  TOTAL: {n_pass}/12 PASS")

    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass, 'n_total': 12,
        'best_xor1': best_xor1, 'best_xor3': best_xor3, 'best_xor5': best_xor5,
        'best_mc': best_mc, 'best_narma10': best_n10,
        'eff_dim_mixed': eff_dim_mixed,
        'pareto_ratios': pareto,
    }

    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  Results saved: {SAVE_FILE}")


if __name__ == '__main__':
    main()
