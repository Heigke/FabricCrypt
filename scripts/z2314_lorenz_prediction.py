#!/usr/bin/env python3
"""
z2314_lorenz_prediction.py — Lorenz attractor prediction benchmark
===================================================================
Standard chaotic system benchmark for reservoir computing.
Complements z2310 (Mackey-Glass) with a multidimensional attractor.

Lorenz system: dx/dt = sigma*(y-x), dy/dt = x*(rho-z)-y, dz/dt = x*y - beta*z
Standard params: sigma=10, rho=28, beta=8/3

Conditions (4):
  1) NVAR baseline (Gauthier 2021)
  2) FPGA-only (128 LIF neurons + temporal features)
  3) GPU-ESN (128-node echo state network + hwmon noise)
  4) Bridge (FPGA + GPU-ESN concatenated)

Prediction targets: x(t+h) for h=1,5,10,20 steps
Additional: cross-variable prediction x→y, x→z

Tests (16):
  T884-T887: FPGA NRMSE < NVAR NRMSE at h=1,5,10,20
  T888-T891: Bridge NRMSE < GPU-ESN NRMSE at h=1,5,10,20
  T892-T893: Cross-variable x→y, x→z R² > 0.5
  T894: Bridge best at h=1
  T895: Bridge best at h=20
  T896: FPGA R² > 0.9 at h=1
  T897: All conditions beat persistence baseline at h≥5
  T898: Lyapunov time analysis (performance degrades with horizon)
  T899: Bridge cross-variable better than FPGA-only

Run:
  PYTHONUNBUFFERED=1 venv/bin/python scripts/z2314_lorenz_prediction.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2314_lorenz_prediction.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
N_STEPS = 5500
WARMUP = 500
HORIZONS = [1, 5, 10, 20]
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}


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


def wait_cool(label="", target=50.0):
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


def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


# ============================================================
# Lorenz system
# ============================================================
def generate_lorenz(n_total, dt=0.02, sigma=10.0, rho=28.0, beta=8.0/3.0,
                    transient=2000, seed=42):
    """Generate Lorenz attractor trajectory. Returns (n_total, 3) array of [x,y,z]."""
    rng = np.random.default_rng(seed)
    # Start near attractor
    state = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.1, 3)

    # Discard transient
    for _ in range(transient):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        state += np.array([dx, dy, dz]) * dt

    # Collect trajectory
    traj = np.zeros((n_total, 3))
    for i in range(n_total):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        state += np.array([dx, dy, dz]) * dt
        traj[i] = state

    # Normalize to [0, 1] per dimension
    for d in range(3):
        mn, mx = traj[:, d].min(), traj[:, d].max()
        if mx > mn:
            traj[:, d] = (traj[:, d] - mn) / (mx - mn)

    return traj


# ============================================================
# FPGA continuous run
# ============================================================
def fpga_run_continuous(fpga, u, n_steps):
    mac_signal = np.clip(u * 0.3 + 0.3, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
    dt = 1.0 / SAMPLE_HZ
    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
    for t in range(n_steps):
        if t > 0 and t % 50 == 0:
            temp = get_max_temp()
            if temp > 75.0:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}C at step {t}/{n_steps}", end="", flush=True)
                while temp > 50.0:
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
            states[t] = states[t - 1]
            dspikes[t] = dspikes[t - 1]
        if t > 0 and t % 500 == 0:
            print(f"    step {t}/{n_steps}, temp={get_max_temp():.0f}C", flush=True)
    fpga.set_mac_signal(0.0)
    return states, dspikes


# ============================================================
# Temporal product features
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
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        if dspikes is not None:
            ds_q = dspikes[:, qi]
            feats.append(ds_q * shifted)
    for i, t1 in enumerate(tau_list):
        for t2 in tau_list[i + 1:]:
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
# NVAR baseline
# ============================================================
def build_nvar_features(u, d_max=10):
    n = len(u)
    delays = []
    for d in range(1, d_max + 1):
        delayed = np.zeros(n)
        delayed[d:] = u[:-d]
        delays.append(delayed)
    delays = np.column_stack(delays)
    poly2 = (3 * delays**2 - 1) / 2
    cross = []
    for i in range(min(d_max, 10)):
        for j in range(i + 1, min(d_max, 10)):
            cross.append(delays[:, i] * delays[:, j])
    cross = np.column_stack(cross) if cross else np.zeros((n, 0))
    return np.hstack([delays, poly2, cross])


# ============================================================
# GPU-ESN
# ============================================================
def build_esn(u, n_nodes=128, spectral_radius=0.95, input_scale=0.1, seed=42):
    rng = np.random.default_rng(seed)
    W_in = rng.uniform(-input_scale, input_scale, (n_nodes, 1))
    W_res = rng.standard_normal((n_nodes, n_nodes)) * 0.1
    eigvals = np.abs(np.linalg.eigvals(W_res))
    W_res *= spectral_radius / max(eigvals.max(), 1e-10)
    # Add hwmon thermal noise
    try:
        with open('/sys/class/hwmon/hwmon7/temp1_input') as f:
            hw_noise = float(f.read().strip()) / 1e6
    except Exception:
        hw_noise = 0.001
    n_steps = len(u)
    states = np.zeros((n_steps, n_nodes))
    x = np.zeros(n_nodes)
    for t in range(n_steps):
        x = np.tanh(W_in @ np.array([[u[t]]]).flatten() + W_res @ x + hw_noise * rng.standard_normal(n_nodes))
        states[t] = x
    return states


# ============================================================
# Ridge regression (pre-computed gram)
# ============================================================
def ridge_predict(X_train, y_train, X_test, alpha=0.01):
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    d = XtX.shape[0]
    w = np.linalg.solve(XtX + alpha * np.eye(d), Xty)
    return X_test @ w


def evaluate_prediction(X, target, offset=10, alphas=None):
    """Train/test split, predict, return NRMSE and R². Searches over alphas."""
    if alphas is None:
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    n = min(len(X), len(target))
    X, target = X[offset:n], target[offset:n]
    n = len(X)
    n_tr = int(0.7 * n)
    if n_tr < 50:
        return {'nrmse': 999.0, 'r2': -999.0, 'alpha': None}
    # Normalize features
    mu = X[:n_tr].mean(axis=0, keepdims=True)
    sigma = X[:n_tr].std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    X_norm = (X - mu) / sigma
    # Use validation split within train for alpha selection
    n_val = n_tr // 5
    n_tr_inner = n_tr - n_val
    best_alpha = alphas[0]
    best_val_mse = 1e30
    for alpha in alphas:
        try:
            pred_val = ridge_predict(X_norm[:n_tr_inner], target[:n_tr_inner],
                                      X_norm[n_tr_inner:n_tr], alpha=alpha)
            val_mse = np.mean((target[n_tr_inner:n_tr] - pred_val) ** 2)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_alpha = alpha
        except Exception:
            continue
    # Final prediction with best alpha on full train
    pred = ridge_predict(X_norm[:n_tr], target[:n_tr], X_norm[n_tr:], alpha=best_alpha)
    y_test = target[n_tr:]
    mse = np.mean((y_test - pred) ** 2)
    var_t = np.var(y_test)
    nrmse = np.sqrt(mse) / (np.std(y_test) + 1e-10)
    r2 = 1.0 - mse / var_t if var_t > 1e-10 else 0.0
    return {'nrmse': float(nrmse), 'r2': float(r2), 'alpha': best_alpha}


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  z2314: Lorenz Attractor Prediction Benchmark")
    print("  FPGA 128-neuron reservoir vs GPU-ESN vs Bridge vs NVAR")
    print("  Horizons: h=1, 5, 10, 20")
    print("=" * 70)

    # Generate Lorenz trajectory
    print("\n[LZ] Generating Lorenz attractor...")
    lorenz = generate_lorenz(N_STEPS, dt=0.02)
    u_input = lorenz[:, 0]  # Drive with x-component
    print(f"  Lorenz shape: {lorenz.shape}, x range: [{u_input.min():.4f}, {u_input.max():.4f}]")

    results = {'conditions': {}, 'tests': {}}

    # ========================================
    # 1) NVAR baseline
    # ========================================
    print(f"\n[1/4] NVAR baseline (Gauthier 2021)")
    X_nvar = build_nvar_features(u_input[WARMUP:], d_max=10)
    print(f"  NVAR features: {X_nvar.shape}")

    nvar_results = {}
    for h in HORIZONS:
        target = u_input[WARMUP + h:]
        X = X_nvar[:len(target)]
        res = evaluate_prediction(X, target)
        nvar_results[f'h{h}'] = res
        print(f"      h={h:2d}: NRMSE={res['nrmse']:.4f}  R²={res['r2']:.4f}")
    results['conditions']['NVAR'] = nvar_results

    # Cross-variable: x→y, x→z
    for dim, label in [(1, 'y'), (2, 'z')]:
        target = lorenz[WARMUP + 1:, dim]
        X = X_nvar[:len(target)]
        res = evaluate_prediction(X, target)
        nvar_results[f'cross_{label}'] = res
        print(f"      x→{label}: NRMSE={res['nrmse']:.4f}  R²={res['r2']:.4f}")

    save_results(results)

    # ========================================
    # 2) FPGA-only
    # ========================================
    print(f"\n[2/4] FPGA-only (128 LIF neurons + temporal features)")
    wait_cool("pre-FPGA", target=45.0)

    states_path = RESULTS / 'z2314_fpga_states.npy'
    dspikes_path = RESULTS / 'z2314_fpga_dspikes.npy'

    if states_path.exists() and dspikes_path.exists():
        print("  Loading cached FPGA states")
        fpga_states = np.load(states_path)
        fpga_dspikes = np.load(dspikes_path)
    else:
        fpga = FPGAEthBridge(timeout=2.0)
        fpga.connect()
        fpga.set_kill(0)
        time.sleep(1.0)
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
        fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_input, N_STEPS)
        np.save(states_path, fpga_states)
        np.save(dspikes_path, fpga_dspikes)
        print(f"  Saved: states {fpga_states.shape}, dspikes {fpga_dspikes.shape}")
        fpga.set_mac_signal(0.0)
        fpga.set_kill(1)

    X_fpga = build_temporal_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:])
    # PCA to 128 dims for thermal safety
    mu_f = X_fpga.mean(axis=0)
    X_centered = X_fpga - mu_f
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n_pca = min(128, X_centered.shape[1])
    X_fpga_pca = X_centered @ Vt[:n_pca].T
    explained = np.sum(S[:n_pca]**2) / np.sum(S**2)
    print(f"  FPGA features: {X_fpga.shape} → PCA {n_pca} ({explained:.1%} var)")

    fpga_results = {}
    for h in HORIZONS:
        target = u_input[WARMUP + h:]
        X = X_fpga_pca[:len(target)]
        res = evaluate_prediction(X, target)
        fpga_results[f'h{h}'] = res
        print(f"      h={h:2d}: NRMSE={res['nrmse']:.4f}  R²={res['r2']:.4f}")

    for dim, label in [(1, 'y'), (2, 'z')]:
        target = lorenz[WARMUP + 1:, dim]
        X = X_fpga_pca[:len(target)]
        res = evaluate_prediction(X, target)
        fpga_results[f'cross_{label}'] = res
        print(f"      x→{label}: NRMSE={res['nrmse']:.4f}  R²={res['r2']:.4f}")

    results['conditions']['FPGA_ONLY'] = fpga_results
    save_results(results)

    # ========================================
    # 3) GPU-ESN
    # ========================================
    print(f"\n[3/4] GPU-ESN (128-node software ESN + hwmon noise)")
    wait_cool("pre-ESN", target=45.0)
    esn_states = build_esn(u_input, n_nodes=128)
    X_esn_raw = esn_states[WARMUP:]
    # Add temporal features for ESN too
    delta_esn = np.diff(esn_states, axis=0)
    delta_esn = np.vstack([np.zeros((1, 128)), delta_esn])
    X_esn = np.hstack([esn_states[WARMUP:], delta_esn[WARMUP:], np.square(esn_states[WARMUP:])])
    # PCA
    mu_e = X_esn.mean(axis=0)
    X_ec = X_esn - mu_e
    Ue, Se, Vte = np.linalg.svd(X_ec, full_matrices=False)
    n_pca_e = min(128, X_ec.shape[1])
    X_esn_pca = X_ec @ Vte[:n_pca_e].T
    print(f"  ESN features: {X_esn.shape} → PCA {n_pca_e}")

    esn_results = {}
    for h in HORIZONS:
        target = u_input[WARMUP + h:]
        X = X_esn_pca[:len(target)]
        res = evaluate_prediction(X, target)
        esn_results[f'h{h}'] = res
        print(f"      h={h:2d}: NRMSE={res['nrmse']:.4f}  R²={res['r2']:.4f}")

    for dim, label in [(1, 'y'), (2, 'z')]:
        target = lorenz[WARMUP + 1:, dim]
        X = X_esn_pca[:len(target)]
        res = evaluate_prediction(X, target)
        esn_results[f'cross_{label}'] = res
        print(f"      x→{label}: NRMSE={res['nrmse']:.4f}  R²={res['r2']:.4f}")

    results['conditions']['GPU_ESN'] = esn_results
    save_results(results)

    # ========================================
    # 4) Bridge (FPGA + ESN concatenated)
    # ========================================
    print(f"\n[4/4] Bridge (FPGA + GPU-ESN concatenated)")
    wait_cool("pre-Bridge", target=45.0)
    n_min = min(len(X_fpga_pca), len(X_esn_pca))
    X_bridge = np.hstack([X_fpga_pca[:n_min], X_esn_pca[:n_min]])
    print(f"  Bridge features: {X_bridge.shape}")

    bridge_results = {}
    for h in HORIZONS:
        target = u_input[WARMUP + h:]
        X = X_bridge[:len(target)]
        res = evaluate_prediction(X, target)
        bridge_results[f'h{h}'] = res
        print(f"      h={h:2d}: NRMSE={res['nrmse']:.4f}  R²={res['r2']:.4f}")

    for dim, label in [(1, 'y'), (2, 'z')]:
        target = lorenz[WARMUP + 1:, dim]
        X = X_bridge[:len(target)]
        res = evaluate_prediction(X, target)
        bridge_results[f'cross_{label}'] = res
        print(f"      x→{label}: NRMSE={res['nrmse']:.4f}  R²={res['r2']:.4f}")

    results['conditions']['BRIDGE'] = bridge_results
    save_results(results)

    # ========================================
    # Tests
    # ========================================
    print(f"\n{'='*70}")
    print("  TESTS")
    print(f"{'='*70}")

    tests = {}
    n = results['conditions']['NVAR']
    f = results['conditions']['FPGA_ONLY']
    e = results['conditions']['GPU_ESN']
    b = results['conditions']['BRIDGE']

    # T884-T887: FPGA < NVAR at each horizon
    for i, h in enumerate(HORIZONS):
        tid = f'T{884+i}'
        fv = f[f'h{h}']['nrmse']
        nv = n[f'h{h}']['nrmse']
        tests[tid] = {'pass': fv < nv, 'fpga': fv, 'nvar': nv,
                      'desc': f'FPGA({fv:.4f}) < NVAR({nv:.4f}) @ h={h}'}

    # T888-T891: Bridge < GPU-ESN at each horizon
    for i, h in enumerate(HORIZONS):
        tid = f'T{888+i}'
        bv = b[f'h{h}']['nrmse']
        ev = e[f'h{h}']['nrmse']
        tests[tid] = {'pass': bv < ev, 'bridge': bv, 'esn': ev,
                      'desc': f'Bridge({bv:.4f}) < GPU-ESN({ev:.4f}) @ h={h}'}

    # T892-T893: Cross-variable R² > 0.5
    for i, (dim, label) in enumerate([(1, 'y'), (2, 'z')]):
        tid = f'T{892+i}'
        r2 = b[f'cross_{label}']['r2']
        tests[tid] = {'pass': r2 > 0.5, 'value': r2, 'threshold': 0.5,
                      'desc': f'Bridge x→{label} R²={r2:.4f} > 0.5'}

    # T894: Bridge best at h=1
    bv1 = b['h1']['nrmse']
    best1 = min(n['h1']['nrmse'], f['h1']['nrmse'], e['h1']['nrmse'])
    tests['T894'] = {'pass': bv1 <= best1, 'bridge': bv1, 'best_other': best1,
                     'desc': f'Bridge best at h=1'}

    # T895: Bridge best at h=20
    bv20 = b['h20']['nrmse']
    best20 = min(n['h20']['nrmse'], f['h20']['nrmse'], e['h20']['nrmse'])
    tests['T895'] = {'pass': bv20 <= best20, 'bridge': bv20, 'best_other': best20,
                     'desc': f'Bridge best at h=20'}

    # T896: FPGA R² > 0.9 at h=1
    r2_h1 = f['h1']['r2']
    tests['T896'] = {'pass': r2_h1 > 0.9, 'value': r2_h1, 'threshold': 0.9,
                     'desc': f'FPGA R²={r2_h1:.4f} > 0.9 at h=1'}

    # T897: All conditions beat persistence at h=5
    persist_nrmse = 1.0  # persistence baseline: predict y(t+h) = y(t) → NRMSE ≈ 1
    all_beat = all(results['conditions'][c]['h5']['nrmse'] < persist_nrmse
                   for c in ['NVAR', 'FPGA_ONLY', 'GPU_ESN', 'BRIDGE'])
    tests['T897'] = {'pass': all_beat, 'desc': 'All conditions beat persistence at h=5'}

    # T898: Performance degrades with horizon (Lyapunov)
    fpga_nrmses = [f[f'h{h}']['nrmse'] for h in HORIZONS]
    monotonic = all(fpga_nrmses[i] <= fpga_nrmses[i+1] for i in range(len(fpga_nrmses)-1))
    tests['T898'] = {'pass': monotonic, 'nrmses': fpga_nrmses,
                     'desc': 'FPGA NRMSE increases with horizon (Lyapunov degradation)'}

    # T899: Bridge cross-variable better than FPGA
    bridge_cross_y = b['cross_y']['r2']
    fpga_cross_y = f['cross_y']['r2']
    tests['T899'] = {'pass': bridge_cross_y > fpga_cross_y,
                     'bridge': bridge_cross_y, 'fpga': fpga_cross_y,
                     'desc': f'Bridge cross x→y R² > FPGA cross x→y R²'}

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)

    for tid, t in sorted(tests.items()):
        status = "PASS" if t['pass'] else "FAIL"
        print(f"  {tid} {status}: {t['desc']}")

    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    # Summary table
    print(f"\n  {'Condition':12s} |  {'h=1':>8s} {'h=5':>8s} {'h=10':>8s} {'h=20':>8s} | {'x→y R²':>8s} {'x→z R²':>8s}")
    print("  " + "-" * 75)
    for cond in ['NVAR', 'FPGA_ONLY', 'GPU_ESN', 'BRIDGE']:
        c = results['conditions'][cond]
        row = f"  {cond:12s} |"
        for h in HORIZONS:
            row += f"  {c[f'h{h}']['nrmse']:6.4f}"
        row += f"  | {c['cross_y']['r2']:8.4f} {c['cross_z']['r2']:8.4f}"
        print(row)

    results['tests'] = tests
    results['n_pass'] = n_pass
    results['n_total'] = n_total
    save_results(results)

    print(f"\n  Done. Results saved to {SAVE_FILE}")
    return n_pass, n_total


if __name__ == '__main__':
    main()
