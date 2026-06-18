#!/usr/bin/env python3
"""
z2324_synapse_validation.py — FPGA Synapse Enhancement Validation
==================================================================
Validates the inter-neuron synapse RTL changes:
  1) Spike hold increased from 8 to 128 cycles (full round-robin persistence)
  2) Small-world topology: N±1 (local excitatory) + N^32, N^64 (long-range inhibitory)
  3) Signed synaptic weights: bit[7]=sign, bits[6:0]=magnitude

Experiments (6):
  EXP1 — Effective Rank: is the reservoir no longer rank-1?
  EXP2 — Memory Capacity: does the reservoir have memory now?
  EXP3 — Spike Statistics: is there inter-neuron diversity?
  EXP4 — Classification Benchmark: 4-class waveform
  EXP5 — XOR Temporal Nonlinearity
  EXP6 — Mackey-Glass Comparison (vs z2310 baselines)

Tests (T1026-T1049, 24 total):
  T1026: eff_rank > 4.0
  T1027: eff_rank > 8.0
  T1028: MC_raw > 1.0
  T1029: MC_raw > 3.0
  T1030: MC_temporal > 10.0
  T1031: mean_corr < 0.8
  T1032: spike_rate_std > 0.1 * spike_rate_mean
  T1033: >= 64/128 channels have distinct spike rates
  T1034: waveform accuracy > 75% (raw)
  T1035: waveform accuracy > 85% (temporal)
  T1036: waveform accuracy > 46% (z2264 baseline)
  T1037: XOR_tau1 > 60%
  T1038: XOR_tau3 > 55%
  T1039: XOR_tau5 > 52%
  T1040: FPGA NRMSE h1 < 0.015
  T1041: Bridge NRMSE h1 < 0.010
  T1042: eff_rank > 2.0 (minimum improvement)
  T1043: MC_raw > 0.5 (any memory at all)
  T1044: at least 32/128 channels active (spike_count > 0)
  T1045: waveform accuracy > 60% (raw, moderate target)
  T1046: XOR_tau1 > 55% (relaxed)
  T1047: FPGA NRMSE h1 < 0.05 (relaxed)
  T1048: FPGA NRMSE h5 < 0.10
  T1049: per-channel vmem range > 0 for >= 64 channels (neurons are alive)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2324_synapse_validation.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2324_synapse_validation.json'
STATES_FILE = RESULTS / 'z2324_fpga_states.npy'
DSPIKES_FILE = RESULTS / 'z2324_fpga_dspikes.npy'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

N_STEPS = 2000
WARMUP = 200

# Mackey-Glass parameters
MG_BETA = 0.2
MG_GAMMA = 0.1
MG_TAU = 17
MG_N_EXP = 10
MG_TOTAL = 3500  # 500 washout + 3000 usable
MG_WASHOUT = 500


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
# FPGA setup and run
# ============================================================
def setup_fpga():
    """Connect and configure FPGA with standard params."""
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
        # Thermal check every 50 steps
        if t > 0 and t % 50 == 0:
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
# Feature engineering
# ============================================================
def build_temporal_features(states, dspikes=None, n_select=24, seed=42):
    """Build temporal order-2+3 product features + PCA reduce to 128 dims."""
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
            ds_q = dspikes[:, qi]
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


def pca_reduce(X, n_components=128):
    """PCA reduce to n_components dimensions if needed."""
    if X.shape[1] <= n_components:
        return X
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


# ============================================================
# Effective rank (Roy & Vetterli 2007)
# ============================================================
def effective_rank(X):
    """Effective rank = exp(entropy of normalized singular values)."""
    # Center the data
    X_c = X - X.mean(axis=0)
    S = np.linalg.svd(X_c, compute_uv=False)
    S = S[S > 1e-10]  # drop near-zero
    if len(S) == 0:
        return 1.0
    # Normalize to probability distribution
    p = S / S.sum()
    # Shannon entropy
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


# ============================================================
# Ridge regression helpers
# ============================================================
def ridge_best_alpha(X_train, y_train, X_test, y_test, alphas=None):
    """Find best ridge alpha and return predictions, R², NRMSE."""
    if alphas is None:
        alphas = RIDGE_ALPHAS
    best_nrmse = 999.0
    best_r2 = -999.0
    best_pred = None
    best_alpha = None
    d = X_train.shape[1]
    I = np.eye(d)
    for alpha in alphas:
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ y_train)
            pred = X_test @ w
            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
            nrmse = np.sqrt(np.mean((y_test - pred)**2)) / (np.std(y_test) + 1e-10)
            if nrmse < best_nrmse:
                best_nrmse = nrmse
                best_r2 = r2
                best_pred = pred
                best_alpha = alpha
        except Exception:
            pass
    return best_pred, best_r2, best_nrmse, best_alpha


def ridge_fast(X_train, y_train, X_test, alpha=0.01):
    """Simple ridge regression, return test predictions."""
    d = X_train.shape[1]
    w = np.linalg.solve(X_train.T @ X_train + alpha * np.eye(d), X_train.T @ y_train)
    return X_test @ w


def compute_mc(X, u, max_delay=20, alphas=None):
    """Memory capacity = sum of R² for delays d=1..max_delay."""
    if alphas is None:
        alphas = RIDGE_ALPHAS
    n = min(len(X), len(u))
    n_tr = int(0.7 * n)
    mc = 0.0
    per_delay = {}
    for d in range(1, max_delay + 1):
        target = u[max_delay - d:max_delay - d + n][:n]
        best_r2 = 0.0
        for alpha in alphas:
            try:
                pred = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:], alpha=alpha)
                y_test = target[n_tr:]
                ss_res = np.sum((y_test - pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
                if r2 > best_r2:
                    best_r2 = r2
            except Exception:
                pass
        mc += best_r2
        per_delay[d] = best_r2
    return mc, per_delay


def compute_xor(X, u, tau=1, alphas=None):
    """XOR between input and delayed input."""
    if alphas is None:
        alphas = RIDGE_ALPHAS
    n = min(len(X), len(u))
    u_bin = (u[:n] > 0.5).astype(float)
    target = np.zeros(n)
    target[tau:] = np.abs(u_bin[tau:] - u_bin[:-tau])
    n_tr = int(0.7 * n)
    best_acc = 0.5
    for alpha in alphas:
        try:
            pred = ridge_fast(X[:n_tr], target[:n_tr], X[n_tr:], alpha=alpha)
            acc = float(np.mean((pred > 0.5).astype(float) == target[n_tr:]))
            if acc > best_acc:
                best_acc = acc
        except Exception:
            pass
    return best_acc


# ============================================================
# Mackey-Glass generation
# ============================================================
def generate_mackey_glass(n_total, beta=MG_BETA, gamma=MG_GAMMA, tau=MG_TAU,
                          n_exp=MG_N_EXP, dt=1.0, seed=42):
    """Generate Mackey-Glass chaotic time series using RK4."""
    rng = np.random.default_rng(seed)
    history_len = int(tau / dt) + 1
    total = n_total + 1000
    x = np.zeros(total + history_len)
    x[:history_len] = 1.2 + 0.1 * rng.standard_normal(history_len)

    def mg_deriv(x_now, x_delayed):
        return beta * x_delayed / (1.0 + x_delayed**n_exp) - gamma * x_now

    tau_steps = int(tau / dt)
    for i in range(history_len, len(x)):
        x_now = x[i-1]
        x_del = x[i-1 - tau_steps]
        k1 = dt * mg_deriv(x_now, x_del)
        k2 = dt * mg_deriv(x_now + k1/2, x_del)
        k3 = dt * mg_deriv(x_now + k2/2, x_del)
        k4 = dt * mg_deriv(x_now + k3, x_del)
        x[i] = x_now + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    mg = x[history_len + 1000: history_len + 1000 + n_total]
    mg = (mg - mg.min()) / (mg.max() - mg.min() + 1e-10)
    return mg


# ============================================================
# Waveform generation for classification
# ============================================================
def generate_waveform_dataset(n_samples=400, n_steps_per=50, seed=42):
    """Generate 4-class waveform dataset: sine, square, triangle, sawtooth.
    Returns (signals, labels) where each signal is n_steps_per long."""
    rng = np.random.default_rng(seed)
    signals = []
    labels = []
    freqs = [1.0, 2.0, 3.0, 5.0]
    n_per_class = n_samples // 4
    t = np.linspace(0, 1, n_steps_per)

    for cls in range(4):
        for i in range(n_per_class):
            f = rng.choice(freqs) * (1.0 + 0.1 * rng.standard_normal())
            phase = rng.uniform(0, 2 * np.pi)
            if cls == 0:  # sine
                sig = np.sin(2 * np.pi * f * t + phase)
            elif cls == 1:  # square
                sig = np.sign(np.sin(2 * np.pi * f * t + phase))
            elif cls == 2:  # triangle
                sig = 2 * np.abs(2 * (f * t + phase / (2*np.pi)) % 1 - 0.5) * 2 - 1
            else:  # sawtooth
                sig = 2 * ((f * t + phase / (2*np.pi)) % 1) - 1
            sig = sig * 0.3 + 0.5  # normalize to [0.2, 0.8] range for MAC
            signals.append(sig)
            labels.append(cls)

    idx = rng.permutation(len(signals))
    signals = [signals[i] for i in idx]
    labels = [labels[i] for i in idx]
    return signals, labels


def classify_waveforms(fpga, signals, labels, use_temporal=False, seed=42):
    """Run each waveform through FPGA, collect states, ridge classify."""
    n_samples = len(signals)
    n_steps_per = len(signals[0])

    all_features = []
    print(f"    Collecting {n_samples} waveform responses...", flush=True)
    for idx, sig in enumerate(signals):
        if idx > 0 and idx % 20 == 0:
            temp = get_max_temp()
            if temp > TEMP_PAUSE:
                fpga.set_mac_signal(0.0)
                print(f"\n    [THERMAL PAUSE] {temp:.0f}C at sample {idx}", end="", flush=True)
                while temp > TEMP_RESUME:
                    time.sleep(5)
                    temp = get_max_temp()
                    print(f" {temp:.0f}", end="", flush=True)
                print(" resumed", flush=True)
            if idx % 100 == 0:
                print(f"    sample {idx}/{n_samples}, temp={temp:.0f}C", flush=True)

        # Drive FPGA with this waveform
        states_block = np.zeros((n_steps_per, NUM_NEURONS))
        dspikes_block = np.zeros((n_steps_per, NUM_NEURONS), dtype=np.float32)
        dt = 1.0 / SAMPLE_HZ

        telem = fpga.read_telemetry()
        prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)

        for t in range(n_steps_per):
            fpga.set_mac_signal(float(sig[t]))
            time.sleep(dt + 0.005)
            telem = fpga.read_telemetry()
            if telem is not None:
                states_block[t] = telem['vmem']
                sc = telem['spike_counts']
                diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
                diff[diff < 0] += 65536
                dspikes_block[t] = diff.astype(np.float32)
                prev_sc = sc.copy()
            elif t > 0:
                states_block[t] = states_block[t-1]
                dspikes_block[t] = dspikes_block[t-1]

        # Feature: mean + std of vmem over the waveform, plus last state
        feat_mean = states_block.mean(axis=0)
        feat_std = states_block.std(axis=0)
        feat_last = states_block[-1]
        feat_dspike_sum = dspikes_block.sum(axis=0)

        if use_temporal:
            # Add temporal product features on this block
            tf = build_temporal_features(states_block, dspikes_block, n_select=16, seed=seed)
            tf_mean = tf.mean(axis=0)
            tf_std = tf.std(axis=0)
            feat = np.concatenate([feat_mean, feat_std, feat_last, feat_dspike_sum, tf_mean, tf_std])
        else:
            feat = np.concatenate([feat_mean, feat_std, feat_last, feat_dspike_sum])

        all_features.append(feat)

    fpga.set_mac_signal(0.0)

    X = np.array(all_features)
    y = np.array(labels)

    # Normalize features
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-2] = 1.0  # floor to avoid amplifying near-constant features
    X = (X - mu) / sigma

    # Train/test split (70/30)
    n_tr = int(0.7 * n_samples)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_te, y_te = X[n_tr:], y[n_tr:]

    # One-vs-rest ridge classification
    n_classes = 4
    best_acc = 0.0
    for alpha in RIDGE_ALPHAS:
        preds = np.zeros((len(X_te), n_classes))
        for c in range(n_classes):
            y_bin = (y_tr == c).astype(float)
            try:
                pred = ridge_fast(X_tr, y_bin, X_te, alpha=alpha)
                preds[:, c] = pred
            except Exception:
                pass
        pred_labels = np.argmax(preds, axis=1)
        acc = float(np.mean(pred_labels == y_te))
        if acc > best_acc:
            best_acc = acc

    return best_acc


# ============================================================
# GPU Thermal ESN (for bridge condition in EXP6)
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
        try:
            with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
                temp_mc = float(f.read().strip())
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
# Save helper
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
    print("  z2324: FPGA Synapse Enhancement Validation")
    print("  Spike hold 8->128, small-world topology, signed weights")
    print("  128 LIF neurons via Ethernet (UDP)")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Temp: {get_max_temp():.0f}C")

    results = {'experiments': {}, 'tests': {}, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}

    # Resume from saved results if any
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done = list(results.get('experiments', {}).keys())
            if done:
                print(f"  RESUMED: {done} already done")
        except Exception:
            results = {'experiments': {}, 'tests': {}, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}

    fpga = None
    try:
        # ==============================================================
        # EXP 1 — Effective Rank
        # ==============================================================
        if 'EXP1_RANK' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 1: Effective Rank (is it still rank-1?)")
            print("=" * 70)
            wait_cool("pre-EXP1", target=TEMP_SAFE)

            fpga = setup_fpga()
            telem = fpga.read_telemetry()
            if telem is not None:
                print(f"  FPGA online: vmem [{telem['vmem'].min():.4f}, {telem['vmem'].max():.4f}]")
            else:
                print("  WARNING: FPGA telemetry returned None!")

            # Collect 2000 steps with random input
            rng = np.random.default_rng(42)
            u_rand = rng.uniform(0, 1, N_STEPS)
            print(f"  Collecting {N_STEPS} steps with random input...", flush=True)
            states, dspikes = fpga_run_continuous(fpga, u_rand)
            np.save(STATES_FILE, states)
            np.save(DSPIKES_FILE, dspikes)
            print(f"  Saved: states {states.shape}, dspikes {dspikes.shape}")

            # Compute effective rank on post-warmup states
            st_w = states[WARMUP:]
            eff_r = effective_rank(st_w)
            print(f"  Effective rank = {eff_r:.3f} (old baseline: 1.15)")

            # Also compute singular value spectrum
            X_c = st_w - st_w.mean(axis=0)
            S = np.linalg.svd(X_c, compute_uv=False)
            S_norm = S / S.sum()
            top5 = S_norm[:5].tolist()

            results['experiments']['EXP1_RANK'] = {
                'effective_rank': eff_r,
                'old_baseline': 1.15,
                'top5_singular_values_normalized': top5,
                'n_steps': N_STEPS,
                'warmup': WARMUP,
                'states_shape': list(states.shape),
                'vmem_range': [float(states.min()), float(states.max())],
                'vmem_std_mean': float(states[WARMUP:].std(axis=0).mean()),
            }
            save_results(results)

            try:
                fpga.set_kill(1)
                fpga.close()
            except Exception:
                pass
            fpga = None
        else:
            print("\n  EXP1_RANK -- already done, skipping")

        # ==============================================================
        # EXP 2 — Memory Capacity
        # ==============================================================
        if 'EXP2_MC' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 2: Memory Capacity")
            print("=" * 70)
            wait_cool("pre-EXP2", target=TEMP_SAFE)

            # Load or collect states
            if STATES_FILE.exists() and DSPIKES_FILE.exists():
                states = np.load(STATES_FILE)
                dspikes = np.load(DSPIKES_FILE)
                rng = np.random.default_rng(42)
                u_rand = rng.uniform(0, 1, N_STEPS)
                print(f"  Using cached states: {states.shape}")
            else:
                fpga = setup_fpga()
                rng = np.random.default_rng(42)
                u_rand = rng.uniform(0, 1, N_STEPS)
                print(f"  Collecting {N_STEPS} steps...", flush=True)
                states, dspikes = fpga_run_continuous(fpga, u_rand)
                np.save(STATES_FILE, states)
                np.save(DSPIKES_FILE, dspikes)
                try:
                    fpga.set_kill(1)
                    fpga.close()
                except Exception:
                    pass
                fpga = None

            st_w = states[WARMUP:]
            ds_w = dspikes[WARMUP:]
            u_w = u_rand[WARMUP:]

            # MC on raw states
            print("  Computing MC on raw FPGA states...", flush=True)
            mc_raw, mc_per_delay_raw = compute_mc(st_w, u_w, max_delay=20)
            print(f"  MC_raw = {mc_raw:.3f}")

            # MC on temporal features
            print("  Building temporal features...", flush=True)
            X_temp = build_temporal_features(st_w, ds_w, n_select=24, seed=42)
            X_temp = pca_reduce(X_temp, n_components=128)
            print(f"  Temporal features: {X_temp.shape}")
            mc_temporal, mc_per_delay_temporal = compute_mc(X_temp, u_w, max_delay=20)
            print(f"  MC_temporal = {mc_temporal:.3f}")

            results['experiments']['EXP2_MC'] = {
                'mc_raw': mc_raw,
                'mc_temporal': mc_temporal,
                'mc_per_delay_raw': mc_per_delay_raw,
                'mc_per_delay_temporal': mc_per_delay_temporal,
            }
            save_results(results)
        else:
            print("\n  EXP2_MC -- already done, skipping")

        # ==============================================================
        # EXP 3 — Spike Statistics
        # ==============================================================
        if 'EXP3_SPIKES' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 3: Spike Statistics (inter-neuron diversity)")
            print("=" * 70)

            # Load states
            if STATES_FILE.exists() and DSPIKES_FILE.exists():
                states = np.load(STATES_FILE)
                dspikes = np.load(DSPIKES_FILE)
                print(f"  Using cached states: {states.shape}")
            else:
                print("  ERROR: no cached states, run EXP1 first")
                results['experiments']['EXP3_SPIKES'] = {'error': 'no cached states'}
                save_results(results)
                return

            st_w = states[WARMUP:]
            ds_w = dspikes[WARMUP:]

            # Per-channel spike rates (spikes per step)
            spike_rates = ds_w.mean(axis=0)
            spike_rate_mean = float(spike_rates.mean())
            spike_rate_std = float(spike_rates.std())
            print(f"  Spike rate: mean={spike_rate_mean:.4f}, std={spike_rate_std:.4f}")

            # Mean pairwise correlation of vmem
            # Subsample for speed: use 500 steps, all 128 channels
            sub = st_w[::max(1, len(st_w)//500)]
            corr_matrix = np.corrcoef(sub.T)
            # Mean of off-diagonal
            mask = ~np.eye(NUM_NEURONS, dtype=bool)
            mean_corr = float(np.nanmean(np.abs(corr_matrix[mask])))
            print(f"  Mean |correlation| = {mean_corr:.4f}")

            # Count channels with distinct spike rates
            # "Distinct" = differ from at least half the other channels by > 0.01
            n_distinct = 0
            for i in range(NUM_NEURONS):
                diffs = np.abs(spike_rates - spike_rates[i])
                if np.sum(diffs > 0.01) >= NUM_NEURONS // 2:
                    n_distinct += 1
            print(f"  Channels with distinct spike rates: {n_distinct}/{NUM_NEURONS}")

            # Count active channels
            n_active = int(np.sum(ds_w.sum(axis=0) > 0))
            print(f"  Active channels (any spikes): {n_active}/{NUM_NEURONS}")

            # Per-channel vmem range
            vmem_ranges = st_w.max(axis=0) - st_w.min(axis=0)
            n_alive = int(np.sum(vmem_ranges > 0))
            print(f"  Channels with vmem range > 0: {n_alive}/{NUM_NEURONS}")

            results['experiments']['EXP3_SPIKES'] = {
                'spike_rate_mean': spike_rate_mean,
                'spike_rate_std': spike_rate_std,
                'mean_abs_correlation': mean_corr,
                'n_distinct_spike_rate': n_distinct,
                'n_active_channels': n_active,
                'n_alive_vmem': n_alive,
                'spike_rate_per_channel': spike_rates.tolist(),
                'vmem_ranges_per_channel': vmem_ranges.tolist(),
            }
            save_results(results)
        else:
            print("\n  EXP3_SPIKES -- already done, skipping")

        # ==============================================================
        # EXP 4 — Classification Benchmark
        # ==============================================================
        if 'EXP4_CLASSIFY' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 4: 4-Class Waveform Classification")
            print("=" * 70)
            wait_cool("pre-EXP4", target=TEMP_SAFE)

            fpga = setup_fpga()
            telem = fpga.read_telemetry()
            if telem is not None:
                print(f"  FPGA online: vmem [{telem['vmem'].min():.4f}, {telem['vmem'].max():.4f}]")

            signals, labels = generate_waveform_dataset(n_samples=200, n_steps_per=50, seed=42)
            print(f"  Generated {len(signals)} waveforms, 4 classes, 50 steps each")

            # Raw features
            print("\n  [RAW] Classifying without temporal features...", flush=True)
            acc_raw = classify_waveforms(fpga, signals, labels, use_temporal=False, seed=42)
            print(f"  Accuracy (raw) = {acc_raw*100:.1f}%")

            wait_cool("pre-temporal", target=TEMP_SAFE)

            # Temporal features
            print("\n  [TEMPORAL] Classifying with temporal features...", flush=True)
            acc_temporal = classify_waveforms(fpga, signals, labels, use_temporal=True, seed=42)
            print(f"  Accuracy (temporal) = {acc_temporal*100:.1f}%")

            results['experiments']['EXP4_CLASSIFY'] = {
                'accuracy_raw': acc_raw,
                'accuracy_temporal': acc_temporal,
                'n_samples': len(signals),
                'n_classes': 4,
                'n_steps_per': 50,
                'z2264_baseline': 0.46,
            }
            save_results(results)

            try:
                fpga.set_kill(1)
                fpga.close()
            except Exception:
                pass
            fpga = None
        else:
            print("\n  EXP4_CLASSIFY -- already done, skipping")

        # ==============================================================
        # EXP 5 — XOR Temporal Nonlinearity
        # ==============================================================
        if 'EXP5_XOR' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 5: XOR Temporal Nonlinearity")
            print("=" * 70)
            wait_cool("pre-EXP5", target=TEMP_SAFE)

            # Load or collect states with random binary-ish input
            if STATES_FILE.exists() and DSPIKES_FILE.exists():
                states = np.load(STATES_FILE)
                dspikes = np.load(DSPIKES_FILE)
                rng = np.random.default_rng(42)
                u_rand = rng.uniform(0, 1, N_STEPS)
                print(f"  Using cached states: {states.shape}")
            else:
                fpga = setup_fpga()
                rng = np.random.default_rng(42)
                u_rand = rng.uniform(0, 1, N_STEPS)
                print(f"  Collecting {N_STEPS} steps...", flush=True)
                states, dspikes = fpga_run_continuous(fpga, u_rand)
                np.save(STATES_FILE, states)
                np.save(DSPIKES_FILE, dspikes)
                try:
                    fpga.set_kill(1)
                    fpga.close()
                except Exception:
                    pass
                fpga = None

            st_w = states[WARMUP:]
            ds_w = dspikes[WARMUP:]
            u_w = u_rand[WARMUP:]

            # Build temporal features
            X_temp = build_temporal_features(st_w, ds_w, n_select=24, seed=42)
            X_temp = pca_reduce(X_temp, n_components=128)

            xor_results = {}
            for tau in [1, 3, 5]:
                # Raw states
                xor_raw = compute_xor(st_w, u_w, tau=tau)
                # Temporal features
                xor_temp = compute_xor(X_temp, u_w, tau=tau)
                xor_results[f'tau{tau}'] = {
                    'raw': xor_raw,
                    'temporal': xor_temp,
                    'best': max(xor_raw, xor_temp),
                }
                print(f"  XOR tau={tau}: raw={xor_raw*100:.1f}%, temporal={xor_temp*100:.1f}%")

            results['experiments']['EXP5_XOR'] = xor_results
            save_results(results)
        else:
            print("\n  EXP5_XOR -- already done, skipping")

        # ==============================================================
        # EXP 6 — Mackey-Glass Comparison
        # ==============================================================
        if 'EXP6_MG' not in results.get('experiments', {}):
            print("\n" + "=" * 70)
            print("  EXP 6: Mackey-Glass Prediction (vs z2310 baselines)")
            print("=" * 70)
            wait_cool("pre-EXP6", target=TEMP_SAFE)

            # Generate Mackey-Glass
            mg = generate_mackey_glass(MG_TOTAL, seed=42)
            mg_input = mg[:MG_TOTAL]
            mg_usable = mg[MG_WASHOUT:]
            print(f"  MG total={MG_TOTAL}, washout={MG_WASHOUT}, usable={len(mg_usable)}")
            print(f"  MG range: [{mg.min():.4f}, {mg.max():.4f}]")

            # Drive FPGA with MG
            fpga = setup_fpga()
            telem = fpga.read_telemetry()
            if telem is not None:
                print(f"  FPGA online: vmem [{telem['vmem'].min():.4f}, {telem['vmem'].max():.4f}]")

            print(f"  Running {MG_TOTAL} steps at {SAMPLE_HZ}Hz...", flush=True)
            mg_states, mg_dspikes = fpga_run_continuous(fpga, mg_input)
            mg_states_file = RESULTS / 'z2324_mg_fpga_states.npy'
            np.save(mg_states_file, mg_states)
            print(f"  Saved: mg_states {mg_states.shape}")

            try:
                fpga.set_kill(1)
                fpga.close()
            except Exception:
                pass
            fpga = None

            # FPGA-only evaluation
            st_w = mg_states[MG_WASHOUT:]
            ds_w = mg_dspikes[MG_WASHOUT:]
            X_fpga = build_temporal_features(st_w, ds_w, n_select=24, seed=42)
            X_fpga = pca_reduce(X_fpga, n_components=128)
            print(f"  FPGA feature matrix: {X_fpga.shape}")

            mg_results = {'FPGA': {}, 'BRIDGE': {}}
            horizons = [1, 5]

            print("\n  [FPGA-only] Evaluating...", flush=True)
            for h in horizons:
                n = min(len(X_fpga), len(mg_usable))
                if h >= n:
                    mg_results['FPGA'][f'h{h}'] = {'nrmse': 999.0, 'r2': -999.0}
                    continue
                X_h = X_fpga[:n-h]
                y_h = mg_usable[h:n]
                n_tr = int(0.7 * len(X_h))
                _, r2, nrmse, alpha = ridge_best_alpha(X_h[:n_tr], y_h[:n_tr], X_h[n_tr:], y_h[n_tr:])
                mg_results['FPGA'][f'h{h}'] = {'nrmse': nrmse, 'r2': r2, 'alpha': alpha}
                print(f"    h={h}: NRMSE={nrmse:.4f}, R²={r2:.4f}")

            # Bridge: FPGA + GPU-ESN
            print("\n  [BRIDGE] Running GPU-ESN + FPGA concatenation...", flush=True)
            esn = GPUThermalESN(n_neurons=128, spectral_radius=0.95, input_scale=0.1, leak=0.3, seed=42)
            mg_scaled = mg_input * 2.0 - 1.0
            esn_states = esn.run(mg_scaled)
            esn_w = esn_states[MG_WASHOUT:]
            X_esn = build_temporal_features(esn_w, n_select=24, seed=43)
            X_esn = pca_reduce(X_esn, n_components=128)

            n_min = min(len(X_fpga), len(X_esn))
            X_bridge = np.hstack([X_fpga[:n_min], X_esn[:n_min]])
            X_bridge = pca_reduce(X_bridge, n_components=256)
            print(f"  Bridge feature matrix: {X_bridge.shape}")

            for h in horizons:
                n = min(len(X_bridge), len(mg_usable))
                if h >= n:
                    mg_results['BRIDGE'][f'h{h}'] = {'nrmse': 999.0, 'r2': -999.0}
                    continue
                X_h = X_bridge[:n-h]
                y_h = mg_usable[h:n]
                n_tr = int(0.7 * len(X_h))
                _, r2, nrmse, alpha = ridge_best_alpha(X_h[:n_tr], y_h[:n_tr], X_h[n_tr:], y_h[n_tr:])
                mg_results['BRIDGE'][f'h{h}'] = {'nrmse': nrmse, 'r2': r2, 'alpha': alpha}
                print(f"    h={h}: NRMSE={nrmse:.4f}, R²={r2:.4f}")

            mg_results['z2310_baseline'] = {
                'FPGA_h1_nrmse': 0.0065,
                'note': 'z2310 used 5500 steps, different temporal features'
            }

            results['experiments']['EXP6_MG'] = mg_results
            save_results(results)
        else:
            print("\n  EXP6_MG -- already done, skipping")

        # ==============================================================
        # Evaluate all tests
        # ==============================================================
        print("\n" + "=" * 70)
        print("  TEST RESULTS")
        print("=" * 70)

        exp = results.get('experiments', {})
        tests = {}
        n_pass = 0
        n_total = 0

        def test(tid, passed, desc, **kwargs):
            nonlocal n_pass, n_total
            n_total += 1
            if passed:
                n_pass += 1
            status = "PASS" if passed else "FAIL"
            tests[tid] = {'pass': bool(passed), 'desc': desc, **kwargs}
            print(f"  {tid} {status}: {desc}", flush=True)

        # EXP1 tests
        e1 = exp.get('EXP1_RANK', {})
        er = e1.get('effective_rank', 0)
        test('T1026', er > 4.0,
             f"eff_rank({er:.2f}) > 4.0 (significant improvement from rank-1)",
             effective_rank=er)
        test('T1027', er > 8.0,
             f"eff_rank({er:.2f}) > 8.0 (substantial recurrence)",
             effective_rank=er)
        test('T1042', er > 2.0,
             f"eff_rank({er:.2f}) > 2.0 (minimum improvement)",
             effective_rank=er)

        # EXP2 tests
        e2 = exp.get('EXP2_MC', {})
        mc_raw = e2.get('mc_raw', 0)
        mc_temp = e2.get('mc_temporal', 0)
        test('T1028', mc_raw > 1.0,
             f"MC_raw({mc_raw:.3f}) > 1.0 (raw FPGA without temporal products)",
             mc_raw=mc_raw)
        test('T1029', mc_raw > 3.0,
             f"MC_raw({mc_raw:.3f}) > 3.0 (good reservoir memory)",
             mc_raw=mc_raw)
        test('T1030', mc_temp > 10.0,
             f"MC_temporal({mc_temp:.3f}) > 10.0 (with temporal products)",
             mc_temporal=mc_temp)
        test('T1043', mc_raw > 0.5,
             f"MC_raw({mc_raw:.3f}) > 0.5 (any memory at all)",
             mc_raw=mc_raw)

        # EXP3 tests
        e3 = exp.get('EXP3_SPIKES', {})
        mean_corr = e3.get('mean_abs_correlation', 1.0)
        sr_mean = e3.get('spike_rate_mean', 0)
        sr_std = e3.get('spike_rate_std', 0)
        n_distinct = e3.get('n_distinct_spike_rate', 0)
        n_active = e3.get('n_active_channels', 0)
        n_alive = e3.get('n_alive_vmem', 0)
        test('T1031', mean_corr < 0.8,
             f"mean_corr({mean_corr:.4f}) < 0.8 (neurons NOT all correlated)",
             mean_corr=mean_corr)
        test('T1032', sr_std > 0.1 * sr_mean if sr_mean > 0 else False,
             f"spike_rate_std({sr_std:.4f}) > 0.1*mean({sr_mean:.4f}) = {0.1*sr_mean:.4f} (firing rate diversity)",
             spike_rate_std=sr_std, spike_rate_mean=sr_mean)
        test('T1033', n_distinct >= 64,
             f"distinct_channels({n_distinct}) >= 64 (spike rate diversity)",
             n_distinct=n_distinct)
        test('T1044', n_active >= 32,
             f"active_channels({n_active}) >= 32 (at least 32/128 active)",
             n_active=n_active)
        test('T1049', n_alive >= 64,
             f"alive_vmem({n_alive}) >= 64 (vmem range > 0 for >= 64 channels)",
             n_alive=n_alive)

        # EXP4 tests
        e4 = exp.get('EXP4_CLASSIFY', {})
        acc_raw = e4.get('accuracy_raw', 0)
        acc_temp = e4.get('accuracy_temporal', 0)
        test('T1034', acc_raw > 0.75,
             f"accuracy_raw({acc_raw*100:.1f}%) > 75% (without temporal)",
             accuracy_raw=acc_raw)
        test('T1035', acc_temp > 0.85,
             f"accuracy_temporal({acc_temp*100:.1f}%) > 85% (with temporal)",
             accuracy_temporal=acc_temp)
        test('T1036', max(acc_raw, acc_temp) > 0.46,
             f"best_accuracy({max(acc_raw, acc_temp)*100:.1f}%) > 46% (z2264 baseline)",
             best_accuracy=max(acc_raw, acc_temp))
        test('T1045', acc_raw > 0.60,
             f"accuracy_raw({acc_raw*100:.1f}%) > 60% (moderate target)",
             accuracy_raw=acc_raw)

        # EXP5 tests
        e5 = exp.get('EXP5_XOR', {})
        xor1 = e5.get('tau1', {}).get('best', 0.5)
        xor3 = e5.get('tau3', {}).get('best', 0.5)
        xor5 = e5.get('tau5', {}).get('best', 0.5)
        test('T1037', xor1 > 0.60,
             f"XOR_tau1({xor1*100:.1f}%) > 60% (above chance)",
             xor_tau1=xor1)
        test('T1038', xor3 > 0.55,
             f"XOR_tau3({xor3*100:.1f}%) > 55%",
             xor_tau3=xor3)
        test('T1039', xor5 > 0.52,
             f"XOR_tau5({xor5*100:.1f}%) > 52% (any signal at long delay)",
             xor_tau5=xor5)
        test('T1046', xor1 > 0.55,
             f"XOR_tau1({xor1*100:.1f}%) > 55% (relaxed)",
             xor_tau1=xor1)

        # EXP6 tests
        e6 = exp.get('EXP6_MG', {})
        fpga_h1 = e6.get('FPGA', {}).get('h1', {}).get('nrmse', 999.0)
        fpga_h5 = e6.get('FPGA', {}).get('h5', {}).get('nrmse', 999.0)
        bridge_h1 = e6.get('BRIDGE', {}).get('h1', {}).get('nrmse', 999.0)
        test('T1040', fpga_h1 < 0.015,
             f"FPGA_NRMSE_h1({fpga_h1:.4f}) < 0.015 (competitive)",
             fpga_nrmse_h1=fpga_h1)
        test('T1041', bridge_h1 < 0.010,
             f"Bridge_NRMSE_h1({bridge_h1:.4f}) < 0.010",
             bridge_nrmse_h1=bridge_h1)
        test('T1047', fpga_h1 < 0.05,
             f"FPGA_NRMSE_h1({fpga_h1:.4f}) < 0.05 (relaxed)",
             fpga_nrmse_h1=fpga_h1)
        test('T1048', fpga_h5 < 0.10,
             f"FPGA_NRMSE_h5({fpga_h5:.4f}) < 0.10",
             fpga_nrmse_h5=fpga_h5)

        results['tests'] = tests
        results['summary'] = {
            'total': n_total,
            'passed': n_pass,
            'failed': n_total - n_pass,
            'pass_rate': f"{n_pass}/{n_total} ({100*n_pass/n_total:.0f}%)" if n_total > 0 else "0/0",
        }
        save_results(results)

        print("\n" + "=" * 70)
        print(f"  SUMMARY: {n_pass}/{n_total} PASS ({100*n_pass/n_total:.0f}%)" if n_total > 0 else "  SUMMARY: 0/0")
        print("=" * 70)

    except Exception as e:
        print(f"\n  [FATAL ERROR] {e}", flush=True)
        import traceback; traceback.print_exc()
        results['error'] = str(e)
        save_results(results)
    finally:
        if fpga is not None:
            try:
                fpga.set_kill(1)
                fpga.close()
            except Exception:
                pass
        print(f"\n  Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Results: {SAVE_FILE}")


if __name__ == '__main__':
    main()
