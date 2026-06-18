#!/usr/bin/env python3
"""
z2311_ipc_capacity.py — Information Processing Capacity (IPC) of 128-neuron FPGA reservoir
===========================================================================================
Gold-standard task-agnostic metric for reservoir computing quality
(Dambre et al., Scientific Reports 2012).

IPC decomposes total information processing into:
  - Linear Memory Capacity (LMC): capacity for linear functions of past inputs
  - Quadratic Capacity (QC): capacity for degree-2 polynomial functions
  - Cubic Capacity (CC): capacity for degree-3 polynomial functions
  - Cross-delay terms: u(t-d1)*u(t-d2) for degree-2
  Total IPC = LMC + QC + CC + cross terms (bounded by reservoir size N)

Target functions use Legendre polynomials (orthogonal basis on [-1,1]):
  P1(x) = x
  P2(x) = (3x^2 - 1) / 2
  P3(x) = (5x^3 - 3x) / 2

Conditions (3):
  1) FPGA raw states (128 features)
  2) FPGA + temporal products (128 + product features)
  3) NVAR baseline (delayed input + polynomial, no reservoir)

Tests (10):
  T859: Total IPC (FPGA raw) > 5.0
  T860: Total IPC (FPGA+temporal) > 10.0
  T861: Linear MC contribution > 60% of total IPC
  T862: Quadratic capacity > 0 (nonlinear processing present)
  T863: Cubic capacity > 0
  T864: IPC(FPGA) > IPC(NVAR) (reservoir adds value)
  T865: IPC(FPGA+temporal) > IPC(FPGA raw) (temporal products help)
  T866: IPC/N > 0.05 (at least 5% of theoretical maximum)
  T867: Linear MC from IPC matches MC from z2296 within 20%
  T868: Total IPC decreases with shuffled states (temporal structure matters)

Run:
  PYTHONUNBUFFERED=1 venv/bin/python scripts/z2311_ipc_capacity.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2311_ipc_capacity.json'
STATES_FILE = RESULTS / 'z2311_fpga_states.npy'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
N_STEPS = 3000
WARMUP = 500
D_MAX = 30
POLY_DEGREES = [1, 2, 3]
CROSS_DELAY_MAX_SUM = 20
RIDGE_ALPHA = 0.01  # Fixed alpha (no CV — saves massive CPU heat)
N_RUNS = 3
TEMP_PAUSE = 60.0
TEMP_RESUME = 42.0
TEMP_SAFE = 42.0
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


def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


# ============================================================
# FPGA continuous run
# ============================================================
def fpga_run_continuous(fpga, u):
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
    fpga.set_mac_signal(0.0)
    return states, dspikes


# ============================================================
# Temporal product features (same as z2296/z2298/z2305)
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
            ds_q = dspikes[:, qi] if dspikes.shape[1] >= n_ch else dspikes[:, :min(n_select, dspikes.shape[1])]
            if ds_q.shape[1] == vm_q.shape[1]:
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
# Legendre polynomials
# ============================================================
def legendre_p1(x):
    return x

def legendre_p2(x):
    return (3.0 * x**2 - 1.0) / 2.0

def legendre_p3(x):
    return (5.0 * x**3 - 3.0 * x) / 2.0

LEGENDRE = {1: legendre_p1, 2: legendre_p2, 3: legendre_p3}


# ============================================================
# NVAR baseline (delay embedding + polynomial, no reservoir)
# ============================================================
def build_nvar_features(u, d_max=20):
    """Nonlinear Vector Auto-Regression baseline: delayed copies + polynomials."""
    n = len(u)
    delays = []
    for d in range(1, d_max + 1):
        delayed = np.zeros(n)
        delayed[d:] = u[:-d]
        delays.append(delayed)
    delays = np.column_stack(delays)
    # Add degree-2 and degree-3 of each delay
    poly2 = legendre_p2(delays)
    poly3 = legendre_p3(delays)
    # Cross products of first 10 delays
    cross = []
    for i in range(min(10, d_max)):
        for j in range(i + 1, min(10, d_max)):
            cross.append(delays[:, i] * delays[:, j])
    cross = np.column_stack(cross) if cross else np.zeros((n, 0))
    return np.hstack([delays, poly2, poly3, cross])


# ============================================================
# Ridge regression with cross-validation
# ============================================================
def ridge_fast(XtX, X_train, y_train, X_test, y_test, alpha=0.01):
    """Fast ridge regression using pre-computed XtX. Returns IPC_i = 1 - MSE/var(target), capped at 0."""
    try:
        Xty = X_train.T @ y_train  # O(n*d) — cheap
        d = XtX.shape[0]
        w = np.linalg.solve(XtX + alpha * np.eye(d), Xty)
        pred = X_test @ w
        mse = np.mean((y_test - pred) ** 2)
        var_t = np.var(y_test)
        return max(0.0, 1.0 - mse / var_t) if var_t > 1e-10 else 0.0
    except Exception:
        return 0.0


# ============================================================
# Compute IPC for a given feature matrix
# ============================================================
def compute_ipc(X, u_input, label=""):
    """
    Compute full IPC decomposition.
    X: (n_steps, n_features) — reservoir states after washout
    u_input: (N_STEPS + WARMUP,) — full input sequence (pre-washout)

    Returns dict with linear_mc, quadratic, cubic, cross_delay, total, per_target breakdown.
    """
    n = len(X)
    n_tr = int(0.7 * n)

    # Normalize features (zero mean, unit var) for numerical stability
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    X_norm = (X - mu) / sigma

    X_train = X_norm[:n_tr]
    X_test = X_norm[n_tr:]

    # Pre-compute gram matrix ONCE — O(n*d^2), the expensive part
    print(f"  [{label}] Pre-computing gram matrix ({X_train.shape[1]}x{X_train.shape[1]})...", flush=True)
    XtX = X_train.T @ X_train
    print(f"  [{label}] Gram matrix ready", flush=True)

    ipc_linear = {}
    ipc_quad = {}
    ipc_cubic = {}
    ipc_cross = {}

    total_targets = D_MAX * 3  # degrees 1,2,3 x delays
    # Count cross terms
    n_cross = 0
    for d1 in range(1, D_MAX + 1):
        for d2 in range(d1 + 1, D_MAX + 1):
            if d1 + d2 <= CROSS_DELAY_MAX_SUM:
                n_cross += 1
    total_targets += n_cross
    done = 0

    print(f"  [{label}] Computing IPC: {total_targets} targets ({D_MAX} delays x 3 degrees + {n_cross} cross terms)")

    def _thermal_check_ipc(done_count, total_count):
        """Check thermal every 5 targets during IPC computation."""
        if done_count % 5 == 0 and done_count > 0:
            temp = get_max_temp()
            if temp > 70.0:
                print(f"\n    [THERMAL PAUSE] {temp:.0f}C at target {done_count}/{total_count}", end="", flush=True)
                while temp > 45.0:
                    time.sleep(5)
                    temp = get_max_temp()
                    print(f" {temp:.0f}", end="", flush=True)
                print(" resumed", flush=True)
        if done_count % 10 == 0 and done_count > 0:
            print(f"    {done_count}/{total_count} targets done", flush=True)

    # Degree 1 (linear memory capacity)
    for d in range(1, D_MAX + 1):
        target_full = legendre_p1(u_input[WARMUP - d: WARMUP - d + N_STEPS - WARMUP])
        nn = min(n, len(target_full))
        if nn < n_tr + 10:
            ipc_linear[d] = 0.0
            continue
        y_tr = target_full[:n_tr]
        y_te = target_full[n_tr:nn]
        ipc_linear[d] = ridge_fast(XtX, X_train[:n_tr], y_tr, X_test[:nn - n_tr], y_te)
        done += 1
        _thermal_check_ipc(done, total_targets)

    # Degree 2 (quadratic)
    for d in range(1, D_MAX + 1):
        target_full = legendre_p2(u_input[WARMUP - d: WARMUP - d + N_STEPS - WARMUP])
        nn = min(n, len(target_full))
        if nn < n_tr + 10:
            ipc_quad[d] = 0.0
            continue
        y_tr = target_full[:n_tr]
        y_te = target_full[n_tr:nn]
        ipc_quad[d] = ridge_fast(XtX, X_train[:n_tr], y_tr, X_test[:nn - n_tr], y_te)
        done += 1
        _thermal_check_ipc(done, total_targets)

    # Degree 3 (cubic)
    for d in range(1, D_MAX + 1):
        target_full = legendre_p3(u_input[WARMUP - d: WARMUP - d + N_STEPS - WARMUP])
        nn = min(n, len(target_full))
        if nn < n_tr + 10:
            ipc_cubic[d] = 0.0
            continue
        y_tr = target_full[:n_tr]
        y_te = target_full[n_tr:nn]
        ipc_cubic[d] = ridge_fast(XtX, X_train[:n_tr], y_tr, X_test[:nn - n_tr], y_te)
        done += 1
        _thermal_check_ipc(done, total_targets)

    # Cross-delay terms (degree 2): u(t-d1) * u(t-d2)
    for d1 in range(1, D_MAX + 1):
        for d2 in range(d1 + 1, D_MAX + 1):
            if d1 + d2 > CROSS_DELAY_MAX_SUM:
                continue
            u_d1 = u_input[WARMUP - d1: WARMUP - d1 + N_STEPS - WARMUP]
            u_d2 = u_input[WARMUP - d2: WARMUP - d2 + N_STEPS - WARMUP]
            nn_raw = min(len(u_d1), len(u_d2))
            target_full = u_d1[:nn_raw] * u_d2[:nn_raw]
            nn = min(n, nn_raw)
            if nn < n_tr + 10:
                ipc_cross[(d1, d2)] = 0.0
                continue
            y_tr = target_full[:n_tr]
            y_te = target_full[n_tr:nn]
            ipc_cross[(d1, d2)] = ridge_fast(XtX, X_train[:n_tr], y_tr, X_test[:nn - n_tr], y_te)
            done += 1
            _thermal_check_ipc(done, total_targets)

    lmc = sum(ipc_linear.values())
    qc = sum(ipc_quad.values())
    cc = sum(ipc_cubic.values())
    xc = sum(ipc_cross.values())
    total = lmc + qc + cc + xc

    print(f"  [{label}] IPC breakdown: LMC={lmc:.3f}, QC={qc:.3f}, CC={cc:.3f}, Cross={xc:.3f}, Total={total:.3f}")

    return {
        'linear_mc': lmc,
        'quadratic': qc,
        'cubic': cc,
        'cross_delay': xc,
        'total': total,
        'linear_per_delay': {str(k): v for k, v in ipc_linear.items()},
        'quadratic_per_delay': {str(k): v for k, v in ipc_quad.items()},
        'cubic_per_delay': {str(k): v for k, v in ipc_cubic.items()},
        'cross_per_pair': {f"{k[0]}_{k[1]}": v for k, v in ipc_cross.items()},
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  z2311: Information Processing Capacity (IPC)")
    print("  128-neuron FPGA reservoir — Dambre et al. (2012) decomposition")
    print("=" * 70)

    results = {'runs': [], 'conditions': {}, 'tests': {}, 'summary': {}}

    # Resume support
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done_runs = len(results.get('runs', []))
            if done_runs > 0:
                print(f"  RESUMED: {done_runs} runs already done")
        except Exception:
            results = {'runs': [], 'conditions': {}, 'tests': {}, 'summary': {}}

    done_runs = len(results.get('runs', []))

    # ================================================================
    # FPGA setup (only if we need to collect states)
    # ================================================================
    fpga = None
    all_run_states = []  # list of (states, dspikes, u_input) per run

    for run_idx in range(done_runs, N_RUNS):
        seed = 42 + run_idx * 1000
        rng = np.random.default_rng(seed)
        u_input = rng.uniform(-1, 1, N_STEPS + WARMUP)

        states_path = RESULTS / f'z2311_fpga_states_run{run_idx}.npy'
        dspikes_path = RESULTS / f'z2311_fpga_dspikes_run{run_idx}.npy'

        if states_path.exists() and dspikes_path.exists():
            print(f"\n[Run {run_idx}] Loading cached FPGA states")
            fpga_states = np.load(states_path)
            fpga_dspikes = np.load(dspikes_path)
        else:
            print(f"\n[Run {run_idx}] FPGA acquisition (seed={seed}, {N_STEPS + WARMUP} steps)")
            wait_cool(f"pre-run{run_idx}", target=TEMP_SAFE)

            if fpga is None:
                fpga = FPGAEthBridge(timeout=2.0)
                fpga.connect()
                fpga.set_kill(0)
                time.sleep(1.0)

                # Set runtime params
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
                else:
                    print("  WARNING: FPGA telemetry returned None!")

            fpga_states, fpga_dspikes = fpga_run_continuous(fpga, u_input)
            np.save(states_path, fpga_states)
            np.save(dspikes_path, fpga_dspikes)
            print(f"  Saved: {states_path.name}, {dspikes_path.name}")

        # Also save first run as the canonical states file
        if run_idx == 0:
            np.save(STATES_FILE, fpga_states)

        all_run_states.append((fpga_states, fpga_dspikes, u_input))

    # Close FPGA
    if fpga is not None:
        try:
            fpga.set_mac_signal(0.0)
            fpga.set_kill(1)
        except Exception:
            pass

    # Reload already-done runs if resumed
    for run_idx in range(0, done_runs):
        seed = 42 + run_idx * 1000
        rng = np.random.default_rng(seed)
        u_input = rng.uniform(-1, 1, N_STEPS + WARMUP)
        states_path = RESULTS / f'z2311_fpga_states_run{run_idx}.npy'
        dspikes_path = RESULTS / f'z2311_fpga_dspikes_run{run_idx}.npy'
        if states_path.exists() and dspikes_path.exists():
            all_run_states.insert(run_idx, (np.load(states_path), np.load(dspikes_path), u_input))
        else:
            print(f"  WARNING: Missing states for run {run_idx}, re-running needed")

    # ================================================================
    # Compute IPC for each run x condition
    # ================================================================
    # Load partial IPC results if available (resume after crash)
    cond_results = {
        'FPGA_RAW': {'runs': []},
        'FPGA_TEMPORAL': {'runs': []},
        'NVAR': {'runs': []},
        'FPGA_SHUFFLED': {'runs': []},
    }
    if results.get('conditions'):
        for cond_name in cond_results:
            if cond_name in results['conditions'] and 'runs' in results['conditions'][cond_name]:
                cond_results[cond_name]['runs'] = results['conditions'][cond_name]['runs']
        # Only count runs where ALL 4 conditions are complete
        n_done = min(len(v['runs']) for v in cond_results.values())
        # Trim any partial runs (e.g. crash mid-run: RAW has 2 but TEMPORAL has 1)
        for cond_name in cond_results:
            cond_results[cond_name]['runs'] = cond_results[cond_name]['runs'][:n_done]
        if n_done > 0:
            print(f"  RESUMED: {n_done} complete IPC runs from partial results")

    for run_idx in range(N_RUNS):
        # Skip if all 4 conditions already done for this run
        n_done_this_run = min(len(cond_results[c]['runs']) for c in cond_results)
        if run_idx < n_done_this_run:
            print(f"\n  Run {run_idx + 1}/{N_RUNS}: already complete, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  IPC Analysis — Run {run_idx + 1}/{N_RUNS}")
        print(f"{'='*60}")

        fpga_states, fpga_dspikes, u_input = all_run_states[run_idx]

        # After-washout slicing
        X_raw = fpga_states[WARMUP:]
        X_dspikes = fpga_dspikes[WARMUP:]

        # Condition 1: FPGA raw states
        wait_cool(f"pre-cond1-run{run_idx}", target=50.0)
        print(f"\n  --- Condition 1: FPGA Raw States ({X_raw.shape[1]} features) ---")
        ipc_raw = compute_ipc(X_raw, u_input, label="FPGA_RAW")
        cond_results['FPGA_RAW']['runs'].append(ipc_raw)
        save_results({**results, 'conditions': {k: v for k, v in cond_results.items()}, 'partial': True})

        # Condition 2: FPGA + temporal products (PCA to 128 dims to avoid thermal death)
        wait_cool(f"pre-cond2-run{run_idx}", target=50.0)
        X_temporal_full = build_temporal_features(fpga_states[WARMUP:], fpga_dspikes[WARMUP:])
        # PCA reduction: 1632 features → 128 to keep gram matrix computation tractable
        mu_t = X_temporal_full.mean(axis=0)
        X_centered = X_temporal_full - mu_t
        # Use SVD on centered data (more numerically stable than covariance)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n_pca = min(128, X_centered.shape[1])
        X_temporal = X_centered @ Vt[:n_pca].T
        explained = np.sum(S[:n_pca]**2) / np.sum(S**2)
        print(f"\n  --- Condition 2: FPGA + Temporal ({X_temporal_full.shape[1]} → PCA {n_pca}, {explained:.1%} var) ---")
        ipc_temporal = compute_ipc(X_temporal, u_input, label="FPGA_TEMPORAL")
        cond_results['FPGA_TEMPORAL']['runs'].append(ipc_temporal)
        save_results({**results, 'conditions': {k: v for k, v in cond_results.items()}, 'partial': True})

        # Condition 3: NVAR baseline
        wait_cool(f"pre-cond3-run{run_idx}", target=50.0)
        u_after = u_input[WARMUP:]
        X_nvar = build_nvar_features(u_after, d_max=20)
        print(f"\n  --- Condition 3: NVAR Baseline ({X_nvar.shape[1]} features) ---")
        ipc_nvar = compute_ipc(X_nvar, u_input, label="NVAR")
        cond_results['NVAR']['runs'].append(ipc_nvar)
        save_results({**results, 'conditions': {k: v for k, v in cond_results.items()}, 'partial': True})

        # Shuffled FPGA states (for T868)
        wait_cool(f"pre-cond4-run{run_idx}", target=50.0)
        rng_shuf = np.random.default_rng(999 + run_idx)
        X_shuf = X_raw.copy()
        for col in range(X_shuf.shape[1]):
            rng_shuf.shuffle(X_shuf[:, col])
        print(f"\n  --- Condition 4: FPGA Shuffled (temporal structure destroyed) ---")
        ipc_shuf = compute_ipc(X_shuf, u_input, label="FPGA_SHUFFLED")
        cond_results['FPGA_SHUFFLED']['runs'].append(ipc_shuf)
        save_results({**results, 'conditions': {k: v for k, v in cond_results.items()}, 'partial': True})

    # ================================================================
    # Aggregate across runs
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  Aggregating {N_RUNS} runs")
    print(f"{'='*60}")

    def aggregate(cond_name):
        runs_data = cond_results[cond_name]['runs']
        totals = [r['total'] for r in runs_data]
        lmcs = [r['linear_mc'] for r in runs_data]
        qcs = [r['quadratic'] for r in runs_data]
        ccs = [r['cubic'] for r in runs_data]
        xcs = [r['cross_delay'] for r in runs_data]
        return {
            'total_mean': float(np.mean(totals)),
            'total_std': float(np.std(totals)),
            'lmc_mean': float(np.mean(lmcs)),
            'lmc_std': float(np.std(lmcs)),
            'qc_mean': float(np.mean(qcs)),
            'qc_std': float(np.std(qcs)),
            'cc_mean': float(np.mean(ccs)),
            'cc_std': float(np.std(ccs)),
            'xc_mean': float(np.mean(xcs)),
            'xc_std': float(np.std(xcs)),
        }

    summary = {}
    for cond in ['FPGA_RAW', 'FPGA_TEMPORAL', 'NVAR', 'FPGA_SHUFFLED']:
        summary[cond] = aggregate(cond)
        s = summary[cond]
        print(f"  {cond:20s}: Total={s['total_mean']:.3f}+/-{s['total_std']:.3f}  "
              f"LMC={s['lmc_mean']:.3f}  QC={s['qc_mean']:.3f}  CC={s['cc_mean']:.3f}  XC={s['xc_mean']:.3f}")

    # ================================================================
    # Tests
    # ================================================================
    print(f"\n{'='*60}")
    print("  TESTS")
    print(f"{'='*60}")

    s_raw = summary['FPGA_RAW']
    s_temp = summary['FPGA_TEMPORAL']
    s_nvar = summary['NVAR']
    s_shuf = summary['FPGA_SHUFFLED']

    tests = {}

    # T859: Total IPC (FPGA raw) > 5.0
    v = s_raw['total_mean']
    tests['T859'] = {'pass': v > 5.0, 'value': v, 'threshold': 5.0,
                     'desc': 'Total IPC (FPGA raw) > 5.0'}

    # T860: Total IPC (FPGA+temporal) > 10.0
    v = s_temp['total_mean']
    tests['T860'] = {'pass': v > 10.0, 'value': v, 'threshold': 10.0,
                     'desc': 'Total IPC (FPGA+temporal) > 10.0'}

    # T861: Linear MC contribution > 60% of total IPC
    lmc_frac = s_raw['lmc_mean'] / s_raw['total_mean'] if s_raw['total_mean'] > 0 else 0
    tests['T861'] = {'pass': lmc_frac > 0.60, 'value': lmc_frac, 'threshold': 0.60,
                     'desc': 'Linear MC > 60% of total IPC'}

    # T862: Quadratic capacity > 0
    v = s_raw['qc_mean']
    tests['T862'] = {'pass': v > 0, 'value': v, 'threshold': 0.0,
                     'desc': 'Quadratic capacity > 0'}

    # T863: Cubic capacity > 0
    v = s_raw['cc_mean']
    tests['T863'] = {'pass': v > 0, 'value': v, 'threshold': 0.0,
                     'desc': 'Cubic capacity > 0'}

    # T864: IPC(FPGA) > IPC(NVAR)
    tests['T864'] = {'pass': s_raw['total_mean'] > s_nvar['total_mean'],
                     'value': s_raw['total_mean'], 'baseline': s_nvar['total_mean'],
                     'desc': 'IPC(FPGA) > IPC(NVAR)'}

    # T865: IPC(FPGA+temporal) > IPC(FPGA raw)
    tests['T865'] = {'pass': s_temp['total_mean'] > s_raw['total_mean'],
                     'value': s_temp['total_mean'], 'baseline': s_raw['total_mean'],
                     'desc': 'IPC(FPGA+temporal) > IPC(FPGA raw)'}

    # T866: IPC/N > 0.05
    ipc_per_n = s_raw['total_mean'] / NUM_NEURONS
    tests['T866'] = {'pass': ipc_per_n > 0.05, 'value': ipc_per_n, 'threshold': 0.05,
                     'desc': 'IPC/N > 0.05 (5% of theoretical max)'}

    # T867: Linear MC from IPC matches z2296 MC within 20%
    z2296_mc = 10.73  # from z2298 reference
    lmc_val = s_raw['lmc_mean']
    ratio = abs(lmc_val - z2296_mc) / z2296_mc if z2296_mc > 0 else 999
    tests['T867'] = {'pass': ratio < 0.20, 'value': lmc_val, 'reference': z2296_mc,
                     'ratio_diff': ratio,
                     'desc': 'Linear MC from IPC matches z2296 within 20%'}

    # T868: Total IPC decreases with shuffled states
    tests['T868'] = {'pass': s_raw['total_mean'] > s_shuf['total_mean'],
                     'value': s_raw['total_mean'], 'shuffled': s_shuf['total_mean'],
                     'desc': 'IPC decreases with shuffled states'}

    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)

    for tid, t in sorted(tests.items()):
        status = "PASS" if t['pass'] else "FAIL"
        print(f"  {tid}: {status} — {t['desc']}")
        for k, v in t.items():
            if k not in ('pass', 'desc'):
                print(f"         {k} = {v}")

    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    # ================================================================
    # Save final results
    # ================================================================
    results = {
        'experiment': 'z2311_ipc_capacity',
        'description': 'Information Processing Capacity (Dambre 2012)',
        'n_runs': N_RUNS,
        'n_steps': N_STEPS,
        'warmup': WARMUP,
        'd_max': D_MAX,
        'conditions': {k: v for k, v in cond_results.items()},
        'summary': summary,
        'tests': tests,
        'n_pass': n_pass,
        'n_total': n_total,
    }
    save_results(results)

    # Print IPC breakdown by delay for FPGA raw (first run)
    print(f"\n{'='*60}")
    print("  IPC per delay (FPGA raw, run 0)")
    print(f"{'='*60}")
    run0 = cond_results['FPGA_RAW']['runs'][0]
    print(f"  {'Delay':>6s}  {'Linear':>8s}  {'Quad':>8s}  {'Cubic':>8s}")
    for d in range(1, D_MAX + 1):
        l = run0['linear_per_delay'].get(str(d), 0)
        q = run0['quadratic_per_delay'].get(str(d), 0)
        c = run0['cubic_per_delay'].get(str(d), 0)
        print(f"  {d:6d}  {l:8.4f}  {q:8.4f}  {c:8.4f}")

    print(f"\n  Done. Results saved to {SAVE_FILE}")
    return n_pass, n_total


if __name__ == '__main__':
    main()
