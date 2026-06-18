#!/usr/bin/env python3
"""
z2310_mackey_glass.py — Mackey-Glass Chaotic Time Series Prediction
====================================================================
The most standard reservoir computing benchmark. Tests FPGA 128-neuron
reservoir on multi-horizon Mackey-Glass prediction.

Mackey-Glass system:
  dx/dt = beta*x(t-tau)/(1+x(t-tau)^n) - gamma*x(t)
  Standard: beta=0.2, gamma=0.1, tau=17, n=10

Conditions (4):
  1) FPGA_ONLY:   128 LIF neurons + temporal product features
  2) GPU_ESN:     128-node software ESN perturbed by hwmon thermal noise
  3) BRIDGE:      FPGA states + GPU-ESN states concatenated
  4) NVAR:        No reservoir — time-delayed polynomial features (Gauthier 2021)

Prediction horizons: h=1, 5, 10, 20
Metrics: NRMSE, R²

Tests (16):
  T843-T846: FPGA NRMSE < 0.3 for h=1,5,10,20
  T847-T850: Bridge NRMSE < FPGA NRMSE for h=1,5,10,20
  T851-T854: FPGA NRMSE < NVAR NRMSE for h=1,5,10,20
  T855-T858: Bridge NRMSE < GPU-ESN NRMSE for h=1,5,10,20

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2310_mackey_glass.py
"""

import os, sys, time, json, struct
import numpy as np
from pathlib import Path
from itertools import product as iterproduct

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2310_mackey_glass.json'
STATES_FILE = RESULTS / 'z2310_fpga_states.npy'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
TEMP_PAUSE = 60.0
TEMP_RESUME = 42.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}

# Mackey-Glass parameters
MG_BETA = 0.2
MG_GAMMA = 0.1
MG_TAU = 17
MG_N = 10
MG_DT = 1.0
MG_TOTAL = 5500   # 500 washout + 5000 usable
MG_WASHOUT = 500

HORIZONS = [1, 5, 10, 20]

# Ridge regression alphas
RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

# Number of CV folds (expanding window)
N_FOLDS = 5


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


# ============================================================
# Mackey-Glass time series generation
# ============================================================
def generate_mackey_glass(n_total, beta=MG_BETA, gamma=MG_GAMMA, tau=MG_TAU,
                          n_exp=MG_N, dt=MG_DT, seed=42):
    """Generate Mackey-Glass chaotic time series using 4th-order Runge-Kutta."""
    rng = np.random.default_rng(seed)
    # Need history buffer of length tau
    history_len = int(tau / dt) + 1
    # Total length including transient
    total = n_total + 1000  # extra transient to discard
    x = np.zeros(total + history_len)
    # Initialize with small random perturbation around 1.2
    x[:history_len] = 1.2 + 0.1 * rng.standard_normal(history_len)

    def mg_deriv(x_now, x_delayed):
        return beta * x_delayed / (1.0 + x_delayed**n_exp) - gamma * x_now

    tau_steps = int(tau / dt)
    for i in range(history_len, len(x)):
        x_now = x[i-1]
        x_del = x[i-1 - tau_steps]
        # RK4
        k1 = dt * mg_deriv(x_now, x_del)
        k2 = dt * mg_deriv(x_now + k1/2, x_del)
        k3 = dt * mg_deriv(x_now + k2/2, x_del)
        k4 = dt * mg_deriv(x_now + k3, x_del)
        x[i] = x_now + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # Discard transient, return n_total points
    mg = x[history_len + 1000: history_len + 1000 + n_total]
    # Normalize to [0, 1]
    mg = (mg - mg.min()) / (mg.max() - mg.min() + 1e-10)
    return mg


# ============================================================
# FPGA continuous run
# ============================================================
def fpga_run_continuous(fpga, u):
    """Drive FPGA with input signal u, return (states, dspikes)."""
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
        # Thermal check every 5 steps
        if t > 0 and t % 5 == 0:
            temp = get_max_temp()
            if temp > TEMP_PAUSE:
                fpga.set_mac_signal(0.0)
                print(f"\n  [THERMAL PAUSE] {temp:.0f}C at step {t}/{n_steps}", end="", flush=True)
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


# ============================================================
# Temporal product features (same as z2296/z2298/z2305)
# ============================================================
def build_temporal_features(states, dspikes=None, n_select=24, seed=42):
    """Build temporal order-2+3 product features for ANY reservoir states."""
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

    # Order-2 temporal products
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        if dspikes is not None:
            ds_q = dspikes[:, qi] if dspikes.shape[1] >= n_ch else dspikes[:, :min(n_select, dspikes.shape[1])]
            if ds_q.shape[1] == vm_q.shape[1]:
                feats.append(ds_q * shifted)

    # Order-3 temporal products (limited)
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


# ============================================================
# Software ESN with hwmon thermal noise perturbation
# ============================================================
class GPUThermalESN:
    """Standard leaky-integrator ESN perturbed by hwmon thermal readings."""
    def __init__(self, n_neurons=128, spectral_radius=0.95, input_scale=0.1, leak=0.3, seed=42):
        rng = np.random.default_rng(seed)
        self.N = n_neurons
        self.leak = leak
        self.input_w = rng.uniform(-input_scale, input_scale, n_neurons)
        W = rng.standard_normal((n_neurons, n_neurons)) * 0.1
        mask = rng.random((n_neurons, n_neurons)) > 0.9
        W *= mask
        eigvals = np.abs(np.linalg.eigvals(W))
        sr = max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0:
            W *= spectral_radius / sr
        self.W = W
        self.bias = rng.uniform(-0.01, 0.01, n_neurons)

    def _read_thermal_noise(self):
        """Read hwmon thermal as perturbation noise (safe sysfs only)."""
        try:
            with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
                temp_mc = float(f.read().strip())
            # Convert millicelsius to small noise: ~0.001 per degree
            return (temp_mc / 1000.0 - 50.0) * 0.001
        except Exception:
            return 0.0

    def run(self, input_seq):
        n_steps = len(input_seq)
        states = np.zeros((n_steps, self.N))
        x = np.zeros(self.N)
        for t in range(n_steps):
            u = input_seq[t]
            noise = self._read_thermal_noise()
            x_new = np.tanh(self.W @ x + self.input_w * u + self.bias + noise)
            x = (1 - self.leak) * x + self.leak * x_new
            states[t] = x
        return states


# ============================================================
# NVAR baseline (Gauthier et al. 2021)
# ============================================================
def build_nvar_features(mg_series, delays=None, degree=2):
    """Build NVAR features: time-delayed copies with polynomial expansion."""
    if delays is None:
        delays = list(range(1, 11))  # delays 1..10
    max_d = max(delays)
    n = len(mg_series) - max_d
    # Delayed copies
    delayed = np.zeros((n, len(delays)))
    for i, d in enumerate(delays):
        delayed[:, i] = mg_series[max_d - d: max_d - d + n]

    if degree == 1:
        return delayed, max_d

    # Degree 2: include all pairwise products
    feats = [delayed]
    n_d = delayed.shape[1]
    products = []
    for i in range(n_d):
        for j in range(i, n_d):
            products.append(delayed[:, i] * delayed[:, j])
    feats.append(np.column_stack(products))

    # Add bias column
    feats.append(np.ones((n, 1)))

    return np.hstack(feats), max_d


# ============================================================
# Ridge regression with expanding-window time-series CV
# ============================================================
def ridge_predict_cv(X, y, alphas=None, n_folds=5):
    """
    Expanding-window time-series cross-validation.
    Returns best (nrmse, r2) averaged across folds.
    """
    if alphas is None:
        alphas = RIDGE_ALPHAS
    n = len(X)
    # Expanding window: fold k uses [0..split_k] for train, [split_k..split_k+1] for test
    min_train = n // (n_folds + 1)
    fold_size = (n - min_train) // n_folds

    best_nrmse = 999.0
    best_r2 = -999.0
    best_alpha = None

    for alpha in alphas:
        nrmses = []
        r2s = []
        for k in range(n_folds):
            tr_end = min_train + k * fold_size
            te_start = tr_end
            te_end = min(te_start + fold_size, n)
            if te_end <= te_start:
                continue
            X_tr, y_tr = X[:tr_end], y[:tr_end]
            X_te, y_te = X[te_start:te_end], y[te_start:te_end]

            I = np.eye(X_tr.shape[1])
            try:
                w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
                pred = X_te @ w
                ss_res = np.sum((y_te - pred) ** 2)
                ss_tot = np.sum((y_te - y_te.mean()) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
                nrmse = np.sqrt(np.mean((y_te - pred)**2)) / (np.std(y_te) + 1e-10)
                nrmses.append(nrmse)
                r2s.append(r2)
            except Exception:
                nrmses.append(999.0)
                r2s.append(-999.0)

        if len(nrmses) == 0:
            continue
        avg_nrmse = np.mean(nrmses)
        avg_r2 = np.mean(r2s)
        if avg_nrmse < best_nrmse:
            best_nrmse = avg_nrmse
            best_r2 = avg_r2
            best_alpha = alpha

    return float(best_nrmse), float(best_r2), best_alpha


def evaluate_condition(X, mg_usable, horizons=None):
    """Evaluate NRMSE and R² for each prediction horizon."""
    if horizons is None:
        horizons = HORIZONS
    results = {}
    for h in horizons:
        # Target: mg(t+h), features: X(t)
        # Trim: we need X[0..n-h] predicting mg[h..n]
        n = min(len(X), len(mg_usable))
        if h >= n:
            results[f'h{h}'] = {'nrmse': 999.0, 'r2': -999.0, 'alpha': None}
            continue
        X_h = X[:n-h]
        y_h = mg_usable[h:n]
        nrmse, r2, alpha = ridge_predict_cv(X_h, y_h)
        results[f'h{h}'] = {'nrmse': nrmse, 'r2': r2, 'alpha': alpha}
        print(f"      h={h:2d}: NRMSE={nrmse:.4f}  R²={r2:.4f}  alpha={alpha}", flush=True)
    return results


# ============================================================
# Incremental save
# ============================================================
def save_results(results):
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_FILE}", flush=True)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  z2310: Mackey-Glass Chaotic Time Series Prediction")
    print("  FPGA 128-neuron reservoir vs GPU-ESN vs Bridge vs NVAR")
    print("  Horizons: h=1, 5, 10, 20")
    print("=" * 70)

    results = {'conditions': {}, 'tests': {}}
    # Resume from saved results if any
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done = list(results.get('conditions', {}).keys())
            if done:
                print(f"  RESUMED: {done} already done")
        except Exception:
            results = {'conditions': {}, 'tests': {}}

    # ----------------------------------------------------------
    # Generate Mackey-Glass time series
    # ----------------------------------------------------------
    print("\n[MG] Generating Mackey-Glass time series...", flush=True)
    mg = generate_mackey_glass(MG_TOTAL, seed=42)
    mg_input = mg[:MG_TOTAL]  # full series for driving reservoir
    mg_usable = mg[MG_WASHOUT:]  # after washout, for targets
    print(f"  Total={MG_TOTAL}, Washout={MG_WASHOUT}, Usable={len(mg_usable)}")
    print(f"  MG range: [{mg.min():.4f}, {mg.max():.4f}], mean={mg.mean():.4f}, std={mg.std():.4f}")

    # ----------------------------------------------------------
    # Condition 1: NVAR baseline (no HW needed, do first)
    # ----------------------------------------------------------
    if 'NVAR' not in results.get('conditions', {}):
        print("\n[1/4] NVAR baseline (Gauthier 2021 — no reservoir)", flush=True)
        try:
            X_nvar, nvar_offset = build_nvar_features(mg_usable, delays=list(range(1, 11)), degree=2)
            print(f"  NVAR features: {X_nvar.shape} (delays 1..10, degree 2, offset={nvar_offset})")
            # Target alignment: X_nvar[i] corresponds to mg_usable[nvar_offset + i]
            mg_nvar = mg_usable[nvar_offset:]
            nvar_res = evaluate_condition(X_nvar, mg_nvar)
            results['conditions']['NVAR'] = nvar_res
            save_results(results)
        except Exception as e:
            print(f"  [ERROR] NVAR failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            results['conditions']['NVAR'] = {'error': str(e)}
            save_results(results)
    else:
        print("\n[1/4] NVAR -- already done, skipping")

    # ----------------------------------------------------------
    # Condition 2: FPGA-only
    # ----------------------------------------------------------
    fpga_states = None
    fpga_dspikes = None
    if 'FPGA_ONLY' not in results.get('conditions', {}):
        print("\n[2/4] FPGA-only (128 LIF neurons + temporal features)", flush=True)
        try:
            # Load cached states if available
            dspikes_file = RESULTS / 'z2310_fpga_dspikes.npy'
            if STATES_FILE.exists() and dspikes_file.exists():
                print("  Loading cached FPGA states from disk", flush=True)
                fpga_states = np.load(STATES_FILE)
                fpga_dspikes = np.load(dspikes_file)
                print(f"  Loaded: states {fpga_states.shape}, dspikes {fpga_dspikes.shape}")
            else:
                wait_cool("pre-FPGA", target=TEMP_SAFE)
                fpga = FPGAEthBridge(timeout=2.0)
                fpga.connect()
                fpga.set_kill(0)
                time.sleep(1.0)

                # Set runtime params (MUST after reprogram — bitstream defaults differ!)
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

                # Drive FPGA with MG input signal
                print(f"  Running {MG_TOTAL} steps at {SAMPLE_HZ}Hz...", flush=True)
                fpga_states, fpga_dspikes = fpga_run_continuous(fpga, mg_input)
                np.save(STATES_FILE, fpga_states)
                np.save(dspikes_file, fpga_dspikes)
                print(f"  Saved: states {fpga_states.shape}, dspikes {fpga_dspikes.shape}")

                try:
                    fpga.set_kill(1)
                except Exception:
                    pass

            # Build features from post-washout states
            st_w = fpga_states[MG_WASHOUT:]
            ds_w = fpga_dspikes[MG_WASHOUT:]
            X_fpga = build_temporal_features(st_w, ds_w, n_select=24, seed=42)
            print(f"  FPGA feature matrix: {X_fpga.shape}")
            fpga_res = evaluate_condition(X_fpga, mg_usable)
            results['conditions']['FPGA_ONLY'] = fpga_res
            save_results(results)
        except Exception as e:
            print(f"  [ERROR] FPGA_ONLY failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            results['conditions']['FPGA_ONLY'] = {'error': str(e)}
            save_results(results)
    else:
        print("\n[2/4] FPGA_ONLY -- already done, skipping")
        # Load states for bridge condition
        dspikes_file = RESULTS / 'z2310_fpga_dspikes.npy'
        if STATES_FILE.exists() and dspikes_file.exists():
            fpga_states = np.load(STATES_FILE)
            fpga_dspikes = np.load(dspikes_file)

    # ----------------------------------------------------------
    # Condition 3: GPU-ESN (software ESN with hwmon noise)
    # ----------------------------------------------------------
    gpu_esn_states = None
    if 'GPU_ESN' not in results.get('conditions', {}):
        print("\n[3/4] GPU-ESN (128-node software ESN + hwmon thermal noise)", flush=True)
        try:
            esn = GPUThermalESN(n_neurons=128, spectral_radius=0.95, input_scale=0.1, leak=0.3, seed=42)
            # Scale MG to [-1,1] for ESN input
            mg_scaled = mg_input * 2.0 - 1.0
            gpu_esn_states = esn.run(mg_scaled)
            print(f"  ESN states: {gpu_esn_states.shape}, range [{gpu_esn_states.min():.3f}, {gpu_esn_states.max():.3f}]")

            esn_w = gpu_esn_states[MG_WASHOUT:]
            X_esn = build_temporal_features(esn_w, n_select=24, seed=42)
            print(f"  ESN feature matrix: {X_esn.shape}")
            esn_res = evaluate_condition(X_esn, mg_usable)
            results['conditions']['GPU_ESN'] = esn_res
            save_results(results)
        except Exception as e:
            print(f"  [ERROR] GPU_ESN failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            results['conditions']['GPU_ESN'] = {'error': str(e)}
            save_results(results)
    else:
        print("\n[3/4] GPU_ESN -- already done, skipping")

    # ----------------------------------------------------------
    # Condition 4: Bridge (FPGA + GPU-ESN concatenated)
    # ----------------------------------------------------------
    if 'BRIDGE' not in results.get('conditions', {}):
        print("\n[4/4] Bridge (FPGA + GPU-ESN concatenated)", flush=True)
        try:
            # Ensure we have FPGA states
            if fpga_states is None:
                dspikes_file = RESULTS / 'z2310_fpga_dspikes.npy'
                if STATES_FILE.exists() and dspikes_file.exists():
                    fpga_states = np.load(STATES_FILE)
                    fpga_dspikes = np.load(dspikes_file)
                else:
                    raise RuntimeError("No FPGA states available for bridge")

            # Ensure we have GPU-ESN states
            if gpu_esn_states is None:
                print("  Re-running GPU-ESN for bridge...", flush=True)
                esn = GPUThermalESN(n_neurons=128, spectral_radius=0.95, input_scale=0.1, leak=0.3, seed=42)
                mg_scaled = mg_input * 2.0 - 1.0
                gpu_esn_states = esn.run(mg_scaled)

            st_fpga_w = fpga_states[MG_WASHOUT:]
            ds_fpga_w = fpga_dspikes[MG_WASHOUT:]
            esn_w = gpu_esn_states[MG_WASHOUT:]

            # Build features separately then concatenate
            X_fpga = build_temporal_features(st_fpga_w, ds_fpga_w, n_select=24, seed=42)
            X_esn = build_temporal_features(esn_w, n_select=24, seed=43)

            # Trim to same length
            n_min = min(len(X_fpga), len(X_esn))
            X_bridge = np.hstack([X_fpga[:n_min], X_esn[:n_min]])
            print(f"  Bridge feature matrix: {X_bridge.shape} (FPGA:{X_fpga.shape[1]} + ESN:{X_esn.shape[1]})")

            bridge_res = evaluate_condition(X_bridge, mg_usable[:n_min])
            results['conditions']['BRIDGE'] = bridge_res
            save_results(results)
        except Exception as e:
            print(f"  [ERROR] BRIDGE failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            results['conditions']['BRIDGE'] = {'error': str(e)}
            save_results(results)
    else:
        print("\n[4/4] BRIDGE -- already done, skipping")

    # ==============================================================
    # Evaluate tests
    # ==============================================================
    print("\n" + "=" * 70)
    print("  TEST RESULTS")
    print("=" * 70)

    conds = results.get('conditions', {})
    tests = {}

    def get_nrmse(cond_name, horizon):
        c = conds.get(cond_name, {})
        if 'error' in c:
            return None
        h_key = f'h{horizon}'
        if h_key not in c:
            return None
        return c[h_key].get('nrmse')

    # T843-T846: FPGA NRMSE < 0.3 for h=1,5,10,20
    for i, h in enumerate(HORIZONS):
        tid = f'T{843+i}'
        fpga_n = get_nrmse('FPGA_ONLY', h)
        if fpga_n is not None:
            passed = fpga_n < 0.3
            tests[tid] = {'pass': passed, 'fpga_nrmse': fpga_n, 'threshold': 0.3,
                          'desc': f'FPGA NRMSE < 0.3 @ h={h}'}
            status = "PASS" if passed else "FAIL"
            print(f"  {tid} {status}: FPGA NRMSE({h})={fpga_n:.4f} {'<' if passed else '>='} 0.3")
        else:
            tests[tid] = {'pass': False, 'desc': f'FPGA NRMSE < 0.3 @ h={h}', 'error': 'no data'}
            print(f"  {tid} SKIP: no FPGA data")

    # T847-T850: Bridge NRMSE < FPGA NRMSE for h=1,5,10,20
    for i, h in enumerate(HORIZONS):
        tid = f'T{847+i}'
        fpga_n = get_nrmse('FPGA_ONLY', h)
        bridge_n = get_nrmse('BRIDGE', h)
        if fpga_n is not None and bridge_n is not None:
            passed = bridge_n < fpga_n
            tests[tid] = {'pass': passed, 'bridge_nrmse': bridge_n, 'fpga_nrmse': fpga_n,
                          'desc': f'Bridge < FPGA @ h={h}'}
            status = "PASS" if passed else "FAIL"
            print(f"  {tid} {status}: Bridge({bridge_n:.4f}) {'<' if passed else '>='} FPGA({fpga_n:.4f}) @ h={h}")
        else:
            tests[tid] = {'pass': False, 'desc': f'Bridge < FPGA @ h={h}', 'error': 'no data'}
            print(f"  {tid} SKIP: missing data")

    # T851-T854: FPGA NRMSE < NVAR NRMSE for h=1,5,10,20
    for i, h in enumerate(HORIZONS):
        tid = f'T{851+i}'
        fpga_n = get_nrmse('FPGA_ONLY', h)
        nvar_n = get_nrmse('NVAR', h)
        if fpga_n is not None and nvar_n is not None:
            passed = fpga_n < nvar_n
            tests[tid] = {'pass': passed, 'fpga_nrmse': fpga_n, 'nvar_nrmse': nvar_n,
                          'desc': f'FPGA < NVAR @ h={h}'}
            status = "PASS" if passed else "FAIL"
            print(f"  {tid} {status}: FPGA({fpga_n:.4f}) {'<' if passed else '>='} NVAR({nvar_n:.4f}) @ h={h}")
        else:
            tests[tid] = {'pass': False, 'desc': f'FPGA < NVAR @ h={h}', 'error': 'no data'}
            print(f"  {tid} SKIP: missing data")

    # T855-T858: Bridge NRMSE < GPU-ESN NRMSE for h=1,5,10,20
    for i, h in enumerate(HORIZONS):
        tid = f'T{855+i}'
        bridge_n = get_nrmse('BRIDGE', h)
        esn_n = get_nrmse('GPU_ESN', h)
        if bridge_n is not None and esn_n is not None:
            passed = bridge_n < esn_n
            tests[tid] = {'pass': passed, 'bridge_nrmse': bridge_n, 'esn_nrmse': esn_n,
                          'desc': f'Bridge < GPU-ESN @ h={h}'}
            status = "PASS" if passed else "FAIL"
            print(f"  {tid} {status}: Bridge({bridge_n:.4f}) {'<' if passed else '>='} GPU-ESN({esn_n:.4f}) @ h={h}")
        else:
            tests[tid] = {'pass': False, 'desc': f'Bridge < GPU-ESN @ h={h}', 'error': 'no data'}
            print(f"  {tid} SKIP: missing data")

    results['tests'] = tests

    # Summary
    n_pass = sum(1 for t in tests.values() if t.get('pass'))
    n_total = len(tests)
    print(f"\n  TOTAL: {n_pass}/{n_total} PASS")

    # Summary table
    print("\n  Condition   |  h=1     h=5     h=10    h=20")
    print("  " + "-" * 55)
    for cond_name in ['NVAR', 'GPU_ESN', 'FPGA_ONLY', 'BRIDGE']:
        vals = []
        for h in HORIZONS:
            n = get_nrmse(cond_name, h)
            vals.append(f"{n:.4f}" if n is not None else "  N/A ")
        print(f"  {cond_name:12s}| {'  '.join(vals)}")

    save_results(results)
    print(f"\n  Results: {SAVE_FILE}")
    print(f"  States:  {STATES_FILE}")
    print(f"\n  Done.")


if __name__ == '__main__':
    main()
