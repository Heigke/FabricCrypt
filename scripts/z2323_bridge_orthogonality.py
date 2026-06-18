#!/usr/bin/env python3
"""
z2323_bridge_orthogonality.py — Bridge Feature Orthogonality Analysis
=====================================================================
Tests the hypothesis from z2321/z2322: the bridge works because GPU and
FPGA features are ORTHOGONAL, not because temporal products create new
memory. With rank-1 FPGA, temporal products of FPGA data alone collapse
under PCA, but combining with GPU-ESN features provides complementary
dimensions that break this degeneracy.

Collects FPGA data + simulates GPU-ESN simultaneously, then analyzes:
1) CKA between FPGA and GPU feature spaces
2) Effective dimensionality of each and combined
3) MC/XOR/Wave for FPGA-only, GPU-only, bridge, and shuffled controls
4) PCA spectrum analysis showing how bridge adds dimensions

Tests (14):
  T1012: Bridge MC > FPGA-only MC
  T1013: Bridge MC > GPU-only MC
  T1014: Bridge MC > max(FPGA, GPU) MC  (super-additive)
  T1015: Bridge wave > FPGA-only wave
  T1016: Bridge wave > GPU-only wave
  T1017: CKA(FPGA, GPU) < 0.5 (complementary, not redundant)
  T1018: EffDim(bridge) > EffDim(FPGA) + EffDim(GPU) × 0.5
  T1019: Bridge XOR > FPGA-only XOR
  T1020: Shuffled bridge MC < real bridge MC (temporal coherence matters)
  T1021: Bridge MC > 5.0 (absolute threshold)
  T1022: FPGA-only MC < 2.0 (confirms rank-1)
  T1023: GPU-only MC > FPGA-only MC (GPU-ESN has recurrence)
  T1024: Bridge effective rank > 10
  T1025: PCA explained variance: bridge top-10 < 95% (distributed information)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2323_bridge_orthogonality.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2323_bridge_orthogonality.json'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
N_STEPS = 5000
WARMUP = 200
N_SELECT = 24
N_ESN = 128
SPECTRAL_RADIUS = 0.95


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


def setup_fpga():
    fpga = FPGAEthBridge(timeout=2.0)
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(1.0)
    fpga.set_leak_cond(0x2000)
    fpga.set_threshold_raw(0x20000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    time.sleep(0.5)
    return fpga


def fpga_run(fpga, u):
    n_steps = len(u)
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
            if temp > TEMP_PAUSE:
                fpga.set_mac_signal(0.0)
                print(f"\n    [THERMAL PAUSE] {temp:.0f}C at step {t}", end="", flush=True)
                while temp > TEMP_RESUME:
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
        if t > 0 and t % 500 == 0:
            print(f"    step {t}/{n_steps}, temp={get_max_temp():.0f}C", flush=True)
    fpga.set_mac_signal(0.0)
    return states, dspikes


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


def pca_reduce(X, n_components=128):
    if X.shape[1] <= n_components:
        return X
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


def effective_rank(X):
    """Roy & Vetterli 2007 effective dimensionality."""
    _, S, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    return float(np.exp(entropy))


def kernel_alignment(X1, X2):
    """CKA — Centered Kernel Alignment."""
    n = min(X1.shape[0], X2.shape[0])
    X1, X2 = X1[:n], X2[:n]
    H = np.eye(n) - np.ones((n, n)) / n
    K1 = H @ (X1 @ X1.T) @ H
    K2 = H @ (X2 @ X2.T) @ H
    hsic12 = np.trace(K1 @ K2) / (n - 1)**2
    hsic11 = np.trace(K1 @ K1) / (n - 1)**2
    hsic22 = np.trace(K2 @ K2) / (n - 1)**2
    denom = np.sqrt(hsic11 * hsic22)
    return float(hsic12 / denom) if denom > 1e-15 else 0.0


def pca_spectrum(X, n_top=10):
    """Return explained variance ratios for top n_top components."""
    X_c = X - X.mean(axis=0)
    _, S, _ = np.linalg.svd(X_c, full_matrices=False)
    var = S ** 2
    total = var.sum()
    ratios = var[:n_top] / (total + 1e-15)
    return ratios


def ridge_alpha_search(X_train, y_train, X_test, alphas=None):
    if alphas is None:
        alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    n_tr = X_train.shape[0]
    n_val = n_tr // 5
    n_tr_inner = n_tr - n_val
    best_alpha = alphas[0]
    best_val_mse = 1e30
    for alpha in alphas:
        XtX = X_train[:n_tr_inner].T @ X_train[:n_tr_inner]
        Xty = X_train[:n_tr_inner].T @ y_train[:n_tr_inner]
        d = XtX.shape[0]
        try:
            w = np.linalg.solve(XtX + alpha * np.eye(d), Xty)
            pred_val = X_train[n_tr_inner:] @ w
            val_mse = np.mean((y_train[n_tr_inner:] - pred_val) ** 2)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_alpha = alpha
        except Exception:
            pass
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    d = XtX.shape[0]
    w = np.linalg.solve(XtX + best_alpha * np.eye(d), Xty)
    return X_test @ w


def compute_mc(X, u, max_delay=20):
    n = min(len(X), len(u))
    n_tr = int(0.7 * n)
    mc = 0.0
    for d in range(1, max_delay + 1):
        target = u[max_delay - d:max_delay - d + n][:n]
        try:
            pred = ridge_alpha_search(X[:n_tr], target[:n_tr], X[n_tr:])
            y_test = target[n_tr:]
            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            mc += r2
        except Exception:
            pass
    return mc


def compute_xor(X, u, tau=1):
    n = min(len(X), len(u))
    u_bin = (u[:n] > 0.5).astype(float)
    target = np.zeros(n)
    target[tau:] = np.abs(u_bin[tau:] - u_bin[:-tau])
    n_tr = int(0.7 * n)
    try:
        pred = ridge_alpha_search(X[:n_tr], target[:n_tr], X[n_tr:])
        return float(np.mean((pred > 0.5).astype(float) == target[n_tr:]))
    except Exception:
        return 0.5


def compute_wave(X, u, n_classes=4):
    n = min(len(X), len(u))
    labels = np.zeros(n, dtype=int)
    chunk = n // n_classes
    for c in range(n_classes):
        labels[c*chunk:(c+1)*chunk] = c
    n_tr = int(0.7 * n)
    preds = np.zeros((n - n_tr, n_classes))
    for c in range(n_classes):
        target = (labels == c).astype(float)
        try:
            preds[:, c] = ridge_alpha_search(X[:n_tr], target[:n_tr], X[n_tr:])
        except Exception:
            pass
    return float(np.mean(np.argmax(preds, axis=1) == labels[n_tr:]))


def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


def main():
    print("=" * 70)
    print("z2323 — Bridge Feature Orthogonality Analysis")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temp: {get_max_temp():.0f}C")

    results = {'experiment': 'z2323_bridge_orthogonality', 'tests': {}, 'exp': {}}
    rng = np.random.default_rng(42)
    u = rng.uniform(0, 1, N_STEPS)

    # ============================================================
    # Step 1: Collect FPGA data
    # ============================================================
    print(f"\n{'='*50}")
    print(f"FPGA Data Collection ({N_STEPS} steps)")
    print(f"{'='*50}")

    wait_cool("fpga-collect")
    fpga = setup_fpga()
    telem = fpga.read_telemetry()
    if telem is not None:
        print(f"  vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    fpga_states, fpga_dspikes = fpga_run(fpga, u)
    fpga.set_kill(1)

    # ============================================================
    # Step 2: Simulate GPU-ESN
    # ============================================================
    print(f"\n{'='*50}")
    print("GPU-ESN Simulation")
    print(f"{'='*50}")

    rng_esn = np.random.default_rng(42)
    W_in = rng_esn.standard_normal((N_ESN, 1)) * 0.1
    W_rec = rng_esn.standard_normal((N_ESN, N_ESN))
    eigvals = np.linalg.eigvals(W_rec)
    W_rec = W_rec * (SPECTRAL_RADIUS / np.max(np.abs(eigvals)))

    x = np.zeros(N_ESN)
    gpu_states = np.zeros((N_STEPS, N_ESN))
    for t in range(N_STEPS):
        x = np.tanh(W_in @ np.array([u[t]]) + W_rec @ x)
        gpu_states[t] = x
    print(f"  GPU-ESN: {gpu_states.shape}")

    # Working data (after warmup)
    fpga_w = fpga_states[WARMUP:]
    dspikes_w = fpga_dspikes[WARMUP:]
    gpu_w = gpu_states[WARMUP:]
    u_w = u[WARMUP:]
    n_w = len(u_w)

    # ============================================================
    # Step 3: Build feature sets
    # ============================================================
    print(f"\n{'='*50}")
    print("Building Feature Sets")
    print(f"{'='*50}")

    # FPGA features (with temporal products)
    X_fpga_full = build_temporal_features(fpga_w, dspikes_w)
    X_fpga = pca_reduce(X_fpga_full, 128)

    # GPU-ESN features (raw states — ESN already has recurrence)
    X_gpu = gpu_w  # 128 dims, no PCA needed

    # Bridge: concatenate FPGA + GPU
    X_bridge_full = np.hstack([X_fpga, X_gpu])
    X_bridge = pca_reduce(X_bridge_full, 128)

    # Shuffled bridge control: shuffle GPU features temporally
    gpu_shuffled = gpu_w.copy()
    perm = rng.permutation(n_w)
    gpu_shuffled = gpu_shuffled[perm]
    X_bridge_shuffled_full = np.hstack([X_fpga, gpu_shuffled])
    X_bridge_shuffled = pca_reduce(X_bridge_shuffled_full, 128)

    # FPGA raw only (no temporal products)
    X_fpga_raw = fpga_w

    print(f"  FPGA (temporal): {X_fpga.shape}")
    print(f"  GPU-ESN: {X_gpu.shape}")
    print(f"  Bridge: {X_bridge.shape}")
    print(f"  Bridge shuffled: {X_bridge_shuffled.shape}")
    print(f"  FPGA raw: {X_fpga_raw.shape}")

    # ============================================================
    # Step 4: Orthogonality analysis
    # ============================================================
    print(f"\n{'='*50}")
    print("Orthogonality Analysis")
    print(f"{'='*50}")

    # Use subsampled data for CKA (expensive for large n)
    n_sub = min(1000, n_w)
    idx = rng.choice(n_w, n_sub, replace=False)
    cka = kernel_alignment(X_fpga[idx], X_gpu[idx])
    print(f"  CKA(FPGA, GPU): {cka:.4f}")

    eff_fpga = effective_rank(X_fpga)
    eff_gpu = effective_rank(X_gpu)
    eff_bridge = effective_rank(X_bridge)
    eff_fpga_raw = effective_rank(X_fpga_raw)
    print(f"  EffRank: FPGA_raw={eff_fpga_raw:.1f}, FPGA_temp={eff_fpga:.1f}, "
          f"GPU={eff_gpu:.1f}, Bridge={eff_bridge:.1f}")

    # PCA spectrum
    spec_fpga = pca_spectrum(X_fpga, 10)
    spec_gpu = pca_spectrum(X_gpu, 10)
    spec_bridge = pca_spectrum(X_bridge_full, 10)
    bridge_top10_var = spec_bridge.sum()
    print(f"  Bridge top-10 explained variance: {bridge_top10_var:.1%}")

    results['exp']['orthogonality'] = {
        'cka': float(cka),
        'eff_rank_fpga_raw': float(eff_fpga_raw),
        'eff_rank_fpga_temp': float(eff_fpga),
        'eff_rank_gpu': float(eff_gpu),
        'eff_rank_bridge': float(eff_bridge),
        'bridge_top10_var': float(bridge_top10_var),
        'pca_fpga': spec_fpga.tolist(),
        'pca_gpu': spec_gpu.tolist(),
        'pca_bridge': spec_bridge.tolist(),
    }

    # ============================================================
    # Step 5: Task evaluation
    # ============================================================
    print(f"\n{'='*50}")
    print("Task Evaluation")
    print(f"{'='*50}")

    conditions = {
        'FPGA_RAW': X_fpga_raw,
        'FPGA_TEMP': X_fpga,
        'GPU_ESN': X_gpu,
        'BRIDGE': X_bridge,
        'BRIDGE_SHUF': X_bridge_shuffled,
    }

    cond_results = {}
    for name, X in conditions.items():
        print(f"\n  {name}...", end="", flush=True)
        mc = compute_mc(X, u_w)
        xor1 = compute_xor(X, u_w)
        wave = compute_wave(X, u_w)
        cond_results[name] = {'mc': float(mc), 'xor1': float(xor1), 'wave': float(wave)}
        print(f" MC={mc:.2f}, XOR1={xor1:.1%}, Wave={wave:.1%}")

    results['exp']['conditions'] = cond_results
    save_results(results)

    # ============================================================
    # Tests
    # ============================================================
    print(f"\n{'='*70}")
    print("TESTS")
    print(f"{'='*70}")

    tests = {}
    def T(tid, name, passed, detail=""):
        tag = "PASS" if passed else "FAIL"
        tests[tid] = {'name': name, 'passed': bool(passed), 'detail': detail}
        print(f"  {tid} [{tag}] {name}: {detail}")

    mc_fpga_raw = cond_results['FPGA_RAW']['mc']
    mc_fpga_temp = cond_results['FPGA_TEMP']['mc']
    mc_gpu = cond_results['GPU_ESN']['mc']
    mc_bridge = cond_results['BRIDGE']['mc']
    mc_bridge_shuf = cond_results['BRIDGE_SHUF']['mc']
    wave_fpga = cond_results['FPGA_TEMP']['wave']
    wave_gpu = cond_results['GPU_ESN']['wave']
    wave_bridge = cond_results['BRIDGE']['wave']
    xor_fpga = cond_results['FPGA_TEMP']['xor1']
    xor_bridge = cond_results['BRIDGE']['xor1']

    T('T1012', 'Bridge MC > FPGA MC',
      mc_bridge > mc_fpga_temp,
      f'Bridge={mc_bridge:.2f} vs FPGA={mc_fpga_temp:.2f}')

    T('T1013', 'Bridge MC > GPU MC',
      mc_bridge > mc_gpu,
      f'Bridge={mc_bridge:.2f} vs GPU={mc_gpu:.2f}')

    T('T1014', 'Bridge MC > max(FPGA, GPU)',
      mc_bridge > max(mc_fpga_temp, mc_gpu),
      f'Bridge={mc_bridge:.2f} vs max={max(mc_fpga_temp, mc_gpu):.2f}')

    T('T1015', 'Bridge wave > FPGA wave',
      wave_bridge > wave_fpga,
      f'Bridge={wave_bridge:.1%} vs FPGA={wave_fpga:.1%}')

    T('T1016', 'Bridge wave > GPU wave',
      wave_bridge > wave_gpu,
      f'Bridge={wave_bridge:.1%} vs GPU={wave_gpu:.1%}')

    T('T1017', 'CKA < 0.5 (complementary)',
      cka < 0.5,
      f'{cka:.4f}')

    T('T1018', 'EffDim(bridge) > EffDim(FPGA) + 0.5*EffDim(GPU)',
      eff_bridge > eff_fpga + 0.5 * eff_gpu,
      f'Bridge={eff_bridge:.1f} vs threshold={eff_fpga + 0.5 * eff_gpu:.1f}')

    T('T1019', 'Bridge XOR > FPGA XOR',
      xor_bridge > xor_fpga,
      f'Bridge={xor_bridge:.1%} vs FPGA={xor_fpga:.1%}')

    T('T1020', 'Shuffled bridge MC < real bridge MC',
      mc_bridge_shuf < mc_bridge,
      f'Shuf={mc_bridge_shuf:.2f} vs Real={mc_bridge:.2f}')

    T('T1021', 'Bridge MC > 5.0',
      mc_bridge > 5.0,
      f'{mc_bridge:.2f}')

    T('T1022', 'FPGA raw MC < 2.0',
      mc_fpga_raw < 2.0,
      f'{mc_fpga_raw:.2f}')

    T('T1023', 'GPU MC > FPGA raw MC',
      mc_gpu > mc_fpga_raw,
      f'GPU={mc_gpu:.2f} vs FPGA_raw={mc_fpga_raw:.2f}')

    T('T1024', 'Bridge effective rank > 10',
      eff_bridge > 10,
      f'{eff_bridge:.1f}')

    T('T1025', 'Bridge top-10 PCA < 95% explained',
      bridge_top10_var < 0.95,
      f'{bridge_top10_var:.1%}')

    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['passed'])
    n_total = len(tests)
    results['summary'] = {'pass': n_pass, 'total': n_total,
                          'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    save_results(results)

    print(f"\n{'='*70}")
    print(f"z2323 SUMMARY: {n_pass}/{n_total} PASS")
    print(f"{'='*70}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
