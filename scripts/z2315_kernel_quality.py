#!/usr/bin/env python3
"""
z2315_kernel_quality.py — Reservoir Kernel Quality Analysis
============================================================
Measures fundamental kernel properties of the FPGA 128-neuron reservoir:
1) Effective rank (Gretton 2005): spectral decay of kernel matrix
2) Separation property: different inputs → different states
3) Generalization rank: kernel rank under output noise
4) Echo state property: state convergence from different initial conditions
5) Fading memory profile: response decay to impulse

Conditions (4):
  1) FPGA_RAW:     128 vmem snapshots only
  2) FPGA_TEMPORAL: vmem + temporal product features
  3) GPU_ESN:      128-node software ESN + hwmon noise
  4) NVAR:         Time-delayed polynomial features

Tests (16):
  T900: FPGA effective rank > 10
  T901: FPGA_TEMPORAL effective rank > FPGA_RAW effective rank
  T902: Bridge effective rank > GPU_ESN effective rank
  T903: NVAR effective rank > 50
  T904: FPGA separation > 0.1 (different inputs → different states)
  T905: FPGA separation > GPU_ESN separation
  T906: FPGA_TEMPORAL separation > FPGA_RAW separation
  T907: Separation monotonic with input distance (3/4 bins increasing)
  T908: Generalization rank FPGA > 5 at noise=0.01
  T909: Generalization rank FPGA_TEMPORAL > 20
  T910: Echo state: state diff decays < 0.01 within 50 steps
  T911: Echo state: convergence rate FPGA < 30 steps
  T912: Fading memory: impulse response R² > 0.5 at lag=1
  T913: Fading memory: impulse response R² < 0.1 at lag=10
  T914: Kernel alignment: FPGA-GPU kernel alignment < 0.5 (different subspaces)
  T915: Combined: FPGA+GPU kernel rank > max(FPGA rank, GPU rank)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2315_kernel_quality.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2315_kernel_quality.json'
STATES_FILE = RESULTS / 'z2315_fpga_states.npy'

from fpga_host_eth import FPGAEthBridge

NUM_NEURONS = 128
SAMPLE_HZ = 50
TEMP_PAUSE = 60.0
TEMP_RESUME = 42.0
TEMP_SAFE = 42.0
VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}

# Kernel analysis parameters
N_INPUT_SIGNALS = 30       # number of distinct random input signals
SIGNAL_LENGTH = 200        # steps per signal (4s at 50Hz)
WASHOUT = 20               # discard first 20 steps

# Ridge
RIDGE_ALPHA = 0.01

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
# Generate diverse input signals
# ============================================================
def generate_input_signals(n_signals, length, seed=42):
    """Generate n_signals random input signals of given length.
    Mix of: sinusoidal, step, ramp, random walk, white noise."""
    rng = np.random.default_rng(seed)
    signals = []
    for i in range(n_signals):
        kind = i % 5
        if kind == 0:  # sinusoidal
            freq = rng.uniform(0.5, 5.0)
            phase = rng.uniform(0, 2*np.pi)
            t = np.linspace(0, length/SAMPLE_HZ, length)
            s = 0.5 + 0.4 * np.sin(2*np.pi*freq*t + phase)
        elif kind == 1:  # step function
            n_steps = rng.integers(3, 8)
            levels = rng.uniform(0.1, 0.9, n_steps)
            step_len = length // n_steps
            s = np.zeros(length)
            for j, lev in enumerate(levels):
                s[j*step_len:min((j+1)*step_len, length)] = lev
        elif kind == 2:  # random walk
            steps = rng.standard_normal(length) * 0.05
            s = np.cumsum(steps)
            s = (s - s.min()) / (s.max() - s.min() + 1e-10) * 0.8 + 0.1
        elif kind == 3:  # white noise
            s = rng.uniform(0.1, 0.9, length)
        else:  # chirp
            t = np.linspace(0, length/SAMPLE_HZ, length)
            freq = np.linspace(0.5, 10.0, length)
            s = 0.5 + 0.4 * np.sin(2*np.pi*np.cumsum(freq/SAMPLE_HZ))
        signals.append(np.clip(s, 0.0, 1.0))
    return signals


# ============================================================
# FPGA continuous run (single signal)
# ============================================================
def fpga_run_signal(fpga, u):
    """Drive FPGA with single input signal, return states."""
    n_steps = len(u)
    mac_signal = np.clip(u * 0.3 + 0.3, 0, 1)
    states = np.zeros((n_steps, NUM_NEURONS))
    dt = 1.0 / SAMPLE_HZ
    fpga.set_mac_signal(0.0)
    time.sleep(0.02)
    telem = fpga.read_telemetry()
    prev_sc = telem['spike_counts'].copy() if telem is not None else np.zeros(NUM_NEURONS, dtype=np.uint16)
    dspikes = np.zeros((n_steps, NUM_NEURONS), dtype=np.float32)
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
            states[t] = states[t-1]
            dspikes[t] = dspikes[t-1]
    fpga.set_mac_signal(0.0)
    return states, dspikes


# ============================================================
# Temporal product features (same as z2296/z2310)
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
# Software ESN
# ============================================================
class GPUThermalESN:
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
# NVAR features
# ============================================================
def build_nvar_features(signal, delays=None, degree=2):
    if delays is None:
        delays = list(range(1, 11))
    max_d = max(delays)
    n = len(signal) - max_d
    delayed = np.zeros((n, len(delays)))
    for i, d in enumerate(delays):
        delayed[:, i] = signal[max_d - d: max_d - d + n]
    if degree == 1:
        return delayed, max_d
    feats = [delayed]
    n_d = delayed.shape[1]
    products = []
    for i in range(n_d):
        for j in range(i, n_d):
            products.append(delayed[:, i] * delayed[:, j])
    feats.append(np.column_stack(products))
    feats.append(np.ones((n, 1)))
    return np.hstack(feats), max_d


# ============================================================
# Kernel quality metrics
# ============================================================
def effective_rank(X):
    """Effective rank via normalized singular value entropy (Roy & Vetterli 2007).
    Higher = more dimensions used, better kernel."""
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 0.0
    p = S / S.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    return float(np.exp(entropy))


def separation_property(state_matrix_list):
    """Measure how well different inputs produce different states.
    state_matrix_list: list of (n_steps, n_features) arrays.
    Returns mean pairwise distance normalized by dimensionality."""
    # Take mean state vector for each signal (after washout)
    centroids = []
    for S in state_matrix_list:
        centroids.append(S[WASHOUT:].mean(axis=0))
    centroids = np.array(centroids)

    # Pairwise distances
    n = len(centroids)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centroids[i] - centroids[j])
            dists.append(d)
    return float(np.mean(dists)) if dists else 0.0


def separation_vs_input_distance(signals, state_matrix_list):
    """Check if state distance is monotonic with input distance."""
    # Compute input distances (L2 of signal differences) and state distances
    n = len(signals)
    input_dists = []
    state_dists = []
    for i in range(n):
        for j in range(i+1, n):
            # Input distance
            min_len = min(len(signals[i]), len(signals[j]))
            id = np.linalg.norm(signals[i][:min_len] - signals[j][:min_len])
            input_dists.append(id)
            # State distance (centroids after washout)
            ci = state_matrix_list[i][WASHOUT:].mean(axis=0)
            cj = state_matrix_list[j][WASHOUT:].mean(axis=0)
            sd = np.linalg.norm(ci - cj)
            state_dists.append(sd)

    input_dists = np.array(input_dists)
    state_dists = np.array(state_dists)

    # Bin into quartiles and check monotonicity
    quartiles = np.percentile(input_dists, [25, 50, 75])
    bins = np.digitize(input_dists, quartiles)
    bin_means = []
    for b in range(4):
        mask = bins == b
        if mask.sum() > 0:
            bin_means.append(float(np.mean(state_dists[mask])))
        else:
            bin_means.append(0.0)

    # Count increasing pairs
    n_increasing = sum(1 for i in range(len(bin_means)-1) if bin_means[i+1] > bin_means[i])
    return n_increasing, bin_means


def generalization_rank(X, noise_level=0.01, seed=42):
    """Rank of kernel matrix under output noise perturbation.
    Higher = more robust generalization."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(X.shape) * noise_level
    X_noisy = X + noise
    _, S, _ = np.linalg.svd(X_noisy, full_matrices=False)
    # Count significant singular values (> 1% of max)
    threshold = S[0] * 0.01
    return int(np.sum(S > threshold))


def kernel_alignment(X1, X2):
    """Centered kernel alignment (CKA) between two feature matrices.
    Low alignment = complementary subspaces (good for combining)."""
    n = X1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # centering matrix
    K1 = H @ (X1 @ X1.T) @ H
    K2 = H @ (X2 @ X2.T) @ H
    hsic12 = np.trace(K1 @ K2) / (n - 1)**2
    hsic11 = np.trace(K1 @ K1) / (n - 1)**2
    hsic22 = np.trace(K2 @ K2) / (n - 1)**2
    denom = np.sqrt(hsic11 * hsic22)
    if denom < 1e-10:
        return 0.0
    return float(hsic12 / denom)


# ============================================================
# PCA for dimensionality reduction (thermal safety)
# ============================================================
def pca_reduce(X, n_components=128):
    """PCA reduce to n_components."""
    if X.shape[1] <= n_components:
        return X
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:n_components].T


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
    print("z2315 — Reservoir Kernel Quality Analysis")
    print("=" * 70)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temp: {get_max_temp():.0f}C")

    results = {'experiment': 'z2315_kernel_quality', 'tests': {}, 'conditions': {}}

    # Generate input signals
    print(f"\n[0] Generating {N_INPUT_SIGNALS} input signals ({SIGNAL_LENGTH} steps each)...")
    signals = generate_input_signals(N_INPUT_SIGNALS, SIGNAL_LENGTH, seed=42)
    print(f"  Signal types: sin, step, ramp, noise, chirp × {N_INPUT_SIGNALS//5}")

    # ============================================================
    # Condition 1: FPGA_RAW
    # ============================================================
    print(f"\n[1/4] FPGA_RAW (128 neurons, {N_INPUT_SIGNALS} signals)...")
    wait_cool("pre-FPGA")

    fpga = FPGAEthBridge(timeout=2.0)
    fpga.connect()
    fpga.set_kill(0)
    time.sleep(1.0)

    # Set runtime params
    fpga.set_leak_cond(0x2000)
    fpga.set_threshold_raw(0x20000)
    fpga.set_base_exc_raw(0x0080)
    fpga.set_bias_gain_raw(0x4000)
    for n in range(NUM_NEURONS):
        fpga.set_vg(n, VG_GROUPS[n % 4])
        time.sleep(0.001)
    time.sleep(0.5)

    telem = fpga.read_telemetry()
    if telem is not None:
        print(f"  FPGA online: vmem [{telem['vmem'].min():.3f}, {telem['vmem'].max():.3f}]")
    else:
        print("  WARNING: no telemetry!")

    fpga_raw_states = []
    fpga_temporal_states = []
    all_dspikes = []

    for sig_idx, sig in enumerate(signals):
        print(f"  Signal {sig_idx+1}/{N_INPUT_SIGNALS}...", end="", flush=True)
        states, dspikes = fpga_run_signal(fpga, sig)
        fpga_raw_states.append(states)
        all_dspikes.append(dspikes)
        # Build temporal features
        temp_feats = build_temporal_features(states, dspikes)
        fpga_temporal_states.append(temp_feats)
        print(f" done ({states.shape[1]} raw, {temp_feats.shape[1]} temporal)", flush=True)

        if (sig_idx + 1) % 5 == 0:
            wait_cool(f"after sig {sig_idx+1}")

    fpga.set_mac_signal(0.0)
    fpga.set_kill(1)

    # Save FPGA states
    np.save(STATES_FILE, np.array([s[WASHOUT:] for s in fpga_raw_states], dtype=object), allow_pickle=True)

    # Compute FPGA_RAW metrics
    print("\n  Computing FPGA_RAW metrics...")
    raw_after_washout = [s[WASHOUT:] for s in fpga_raw_states]
    raw_concat = np.vstack(raw_after_washout)
    fpga_raw_eff_rank = effective_rank(raw_concat)
    fpga_raw_sep = separation_property(fpga_raw_states)
    fpga_raw_gen_rank = generalization_rank(raw_concat)
    n_inc_raw, bin_means_raw = separation_vs_input_distance(signals, fpga_raw_states)

    print(f"  Effective rank: {fpga_raw_eff_rank:.2f}")
    print(f"  Separation: {fpga_raw_sep:.4f}")
    print(f"  Generalization rank: {fpga_raw_gen_rank}")
    print(f"  Monotonicity: {n_inc_raw}/3 bins increasing, means={bin_means_raw}")

    results['conditions']['FPGA_RAW'] = {
        'effective_rank': fpga_raw_eff_rank,
        'separation': fpga_raw_sep,
        'generalization_rank': fpga_raw_gen_rank,
        'monotonicity_increasing': n_inc_raw,
        'bin_means': bin_means_raw,
    }

    # Compute FPGA_TEMPORAL metrics (PCA first for thermal safety)
    print("\n  Computing FPGA_TEMPORAL metrics...")
    temp_after_washout = [s[WASHOUT:] for s in fpga_temporal_states]
    temp_concat = np.vstack(temp_after_washout)
    temp_concat_pca = pca_reduce(temp_concat, n_components=128)
    fpga_temp_eff_rank = effective_rank(temp_concat_pca)
    fpga_temp_sep = separation_property(fpga_temporal_states)
    fpga_temp_gen_rank = generalization_rank(temp_concat_pca)
    n_inc_temp, bin_means_temp = separation_vs_input_distance(signals, fpga_temporal_states)

    print(f"  Effective rank (PCA-128): {fpga_temp_eff_rank:.2f}")
    print(f"  Separation: {fpga_temp_sep:.4f}")
    print(f"  Generalization rank: {fpga_temp_gen_rank}")

    results['conditions']['FPGA_TEMPORAL'] = {
        'effective_rank': fpga_temp_eff_rank,
        'separation': fpga_temp_sep,
        'generalization_rank': fpga_temp_gen_rank,
        'monotonicity_increasing': n_inc_temp,
        'bin_means': bin_means_temp,
        'n_features_raw': temp_concat.shape[1],
        'n_features_pca': temp_concat_pca.shape[1],
    }
    save_results(results)

    # ============================================================
    # Condition 2: GPU_ESN
    # ============================================================
    print(f"\n[2/4] GPU_ESN (128-node ESN, {N_INPUT_SIGNALS} signals)...")
    wait_cool("pre-ESN")

    esn = GPUThermalESN(n_neurons=128, spectral_radius=0.95, input_scale=0.1, leak=0.3, seed=42)
    esn_states = []
    for sig_idx, sig in enumerate(signals):
        s = esn.run(sig)
        esn_states.append(s)
        if (sig_idx + 1) % 10 == 0:
            print(f"  Signal {sig_idx+1}/{N_INPUT_SIGNALS} done", flush=True)

    esn_after_washout = [s[WASHOUT:] for s in esn_states]
    esn_concat = np.vstack(esn_after_washout)
    esn_eff_rank = effective_rank(esn_concat)
    esn_sep = separation_property(esn_states)
    esn_gen_rank = generalization_rank(esn_concat)

    print(f"  Effective rank: {esn_eff_rank:.2f}")
    print(f"  Separation: {esn_sep:.4f}")
    print(f"  Generalization rank: {esn_gen_rank}")

    results['conditions']['GPU_ESN'] = {
        'effective_rank': esn_eff_rank,
        'separation': esn_sep,
        'generalization_rank': esn_gen_rank,
    }
    save_results(results)

    # ============================================================
    # Condition 3: NVAR
    # ============================================================
    print(f"\n[3/4] NVAR (time-delayed polynomial features)...")
    wait_cool("pre-NVAR")

    nvar_states = []
    for sig_idx, sig in enumerate(signals):
        feats, offset = build_nvar_features(sig, delays=list(range(1, 11)), degree=2)
        # Pad to match signal length
        padded = np.zeros((len(sig), feats.shape[1]))
        padded[offset:offset+len(feats)] = feats
        nvar_states.append(padded)

    nvar_after_washout = [s[WASHOUT:] for s in nvar_states]
    nvar_concat = np.vstack(nvar_after_washout)
    nvar_eff_rank = effective_rank(nvar_concat)
    nvar_sep = separation_property(nvar_states)
    nvar_gen_rank = generalization_rank(nvar_concat)

    print(f"  Effective rank: {nvar_eff_rank:.2f}")
    print(f"  Separation: {nvar_sep:.4f}")
    print(f"  Generalization rank: {nvar_gen_rank}")

    results['conditions']['NVAR'] = {
        'effective_rank': nvar_eff_rank,
        'separation': nvar_sep,
        'generalization_rank': nvar_gen_rank,
    }
    save_results(results)

    # ============================================================
    # Condition 4: Bridge (FPGA + GPU combined)
    # ============================================================
    print(f"\n[4/4] Bridge kernel analysis...")

    # Kernel alignment (CKA) between FPGA and GPU
    # Use matching samples
    min_samples = min(len(raw_concat), len(esn_concat))
    fpga_sub = raw_concat[:min_samples]
    esn_sub = esn_concat[:min_samples]
    cka = kernel_alignment(fpga_sub, esn_sub)
    print(f"  FPGA-GPU kernel alignment (CKA): {cka:.4f}")

    # Combined kernel rank
    bridge_concat = np.hstack([fpga_sub, esn_sub])
    bridge_eff_rank = effective_rank(bridge_concat)
    bridge_gen_rank = generalization_rank(bridge_concat)
    print(f"  Bridge effective rank: {bridge_eff_rank:.2f}")
    print(f"  Bridge generalization rank: {bridge_gen_rank}")

    results['conditions']['BRIDGE'] = {
        'effective_rank': bridge_eff_rank,
        'generalization_rank': bridge_gen_rank,
        'kernel_alignment_cka': cka,
    }
    save_results(results)

    # ============================================================
    # Echo state property (FPGA-specific)
    # ============================================================
    print(f"\n[5] Echo state property test...")
    wait_cool("pre-echo")

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

    # Drive with same signal twice (different start conditions)
    echo_signal = signals[0]  # first sinusoidal signal
    echo_len = min(100, len(echo_signal))  # 100 steps = 2s

    # Run 1: fresh start
    states1, _ = fpga_run_signal(fpga, echo_signal[:echo_len])
    wait_cool("echo-between")

    # Perturb: drive with noise briefly
    noise_perturb = np.random.default_rng(99).uniform(0.0, 1.0, 20)
    fpga_run_signal(fpga, noise_perturb)

    # Run 2: same signal again (different initial state)
    states2, _ = fpga_run_signal(fpga, echo_signal[:echo_len])

    fpga.set_mac_signal(0.0)
    fpga.set_kill(1)

    # Compute convergence
    state_diffs = np.linalg.norm(states1 - states2, axis=1)
    state_diffs_norm = state_diffs / (state_diffs[0] + 1e-10)
    converge_step = echo_len  # default: didn't converge
    for t in range(echo_len):
        if state_diffs_norm[t] < 0.01:
            converge_step = t
            break

    print(f"  State diff at t=0: {state_diffs[0]:.4f}")
    print(f"  State diff at t=50: {state_diffs[min(49, echo_len-1)]:.4f}")
    print(f"  Convergence step (< 1%): {converge_step}")
    print(f"  Final diff norm: {state_diffs_norm[-1]:.4f}")

    results['echo_state'] = {
        'diff_t0': float(state_diffs[0]),
        'diff_t50': float(state_diffs[min(49, echo_len-1)]),
        'converge_step': int(converge_step),
        'final_diff_norm': float(state_diffs_norm[-1]),
    }
    save_results(results)

    # ============================================================
    # Fading memory (impulse response)
    # ============================================================
    print(f"\n[6] Fading memory (impulse response)...")
    wait_cool("pre-impulse")

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

    # Generate impulse signal: background + random impulses
    rng = np.random.default_rng(123)
    impulse_len = 300
    background = np.full(impulse_len, 0.3)
    impulse_times = rng.choice(range(20, impulse_len-20), size=15, replace=False)
    impulse_amplitudes = rng.uniform(0.5, 1.0, len(impulse_times))
    for it, ia in zip(impulse_times, impulse_amplitudes):
        background[it] = ia

    imp_states, imp_dspikes = fpga_run_signal(fpga, background)
    fpga.set_mac_signal(0.0)
    fpga.set_kill(1)

    # Measure impulse response R² at different lags
    # For each impulse, see how well states at t+lag predict impulse amplitude
    fading_r2 = {}
    for lag in [1, 2, 3, 5, 10, 15, 20]:
        # Build features: state at impulse_time + lag
        X_imp = []
        y_imp = []
        for it, ia in zip(impulse_times, impulse_amplitudes):
            t_read = it + lag
            if t_read < impulse_len:
                X_imp.append(imp_states[t_read])
                y_imp.append(ia)
        if len(X_imp) < 5:
            fading_r2[f'lag_{lag}'] = 0.0
            continue
        X_imp = np.array(X_imp)
        y_imp = np.array(y_imp)
        # Simple ridge regression (leave-one-out for small n)
        n = len(X_imp)
        pred = np.zeros(n)
        for i in range(n):
            X_tr = np.delete(X_imp, i, axis=0)
            y_tr = np.delete(y_imp, i)
            try:
                w = np.linalg.solve(X_tr.T @ X_tr + RIDGE_ALPHA * np.eye(X_tr.shape[1]), X_tr.T @ y_tr)
                pred[i] = X_imp[i] @ w
            except Exception:
                pred[i] = y_imp.mean()
        ss_res = np.sum((y_imp - pred) ** 2)
        ss_tot = np.sum((y_imp - y_imp.mean()) ** 2)
        r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        fading_r2[f'lag_{lag}'] = float(r2)
        print(f"  Lag {lag:2d}: R² = {r2:.4f}")

    results['fading_memory'] = fading_r2
    save_results(results)

    # ============================================================
    # Run tests
    # ============================================================
    print(f"\n{'='*70}")
    print("TESTS")
    print(f"{'='*70}")

    tests = {}

    def T(tid, name, passed, detail=""):
        tag = "PASS" if passed else "FAIL"
        tests[tid] = {'name': name, 'passed': bool(passed), 'detail': detail}
        print(f"  {tid} [{tag}] {name}: {detail}")

    # Effective rank tests
    T('T900', 'FPGA effective rank > 10',
      fpga_raw_eff_rank > 10,
      f'{fpga_raw_eff_rank:.2f}')

    T('T901', 'FPGA_TEMPORAL rank > FPGA_RAW rank',
      fpga_temp_eff_rank > fpga_raw_eff_rank,
      f'{fpga_temp_eff_rank:.2f} vs {fpga_raw_eff_rank:.2f}')

    T('T902', 'Bridge rank > GPU_ESN rank',
      bridge_eff_rank > esn_eff_rank,
      f'{bridge_eff_rank:.2f} vs {esn_eff_rank:.2f}')

    T('T903', 'NVAR effective rank > 50',
      nvar_eff_rank > 50,
      f'{nvar_eff_rank:.2f}')

    # Separation tests
    T('T904', 'FPGA separation > 0.1',
      fpga_raw_sep > 0.1,
      f'{fpga_raw_sep:.4f}')

    T('T905', 'FPGA separation > GPU_ESN separation',
      fpga_raw_sep > esn_sep,
      f'{fpga_raw_sep:.4f} vs {esn_sep:.4f}')

    T('T906', 'FPGA_TEMPORAL separation > FPGA_RAW separation',
      fpga_temp_sep > fpga_raw_sep,
      f'{fpga_temp_sep:.4f} vs {fpga_raw_sep:.4f}')

    T('T907', 'Separation monotonic (3/4 bins increasing)',
      n_inc_raw >= 3,
      f'{n_inc_raw}/3 increasing, bins={[f"{b:.3f}" for b in bin_means_raw]}')

    # Generalization rank tests
    T('T908', 'FPGA gen rank > 5 at noise=0.01',
      fpga_raw_gen_rank > 5,
      f'{fpga_raw_gen_rank}')

    T('T909', 'FPGA_TEMPORAL gen rank > 20',
      fpga_temp_gen_rank > 20,
      f'{fpga_temp_gen_rank}')

    # Echo state tests
    T('T910', 'Echo state: diff < 0.01 within 50 steps',
      converge_step <= 50,
      f'converge at step {converge_step}')

    T('T911', 'Echo state: convergence < 30 steps',
      converge_step < 30,
      f'converge at step {converge_step}')

    # Fading memory tests
    fm_lag1 = fading_r2.get('lag_1', 0.0)
    fm_lag10 = fading_r2.get('lag_10', 0.0)
    T('T912', 'Fading memory: R² > 0.5 at lag=1',
      fm_lag1 > 0.5,
      f'{fm_lag1:.4f}')

    T('T913', 'Fading memory: R² < 0.1 at lag=10',
      fm_lag10 < 0.1,
      f'{fm_lag10:.4f}')

    # Kernel alignment
    T('T914', 'Kernel alignment FPGA-GPU < 0.5',
      cka < 0.5,
      f'CKA={cka:.4f}')

    T('T915', 'Combined rank > max(FPGA, GPU)',
      bridge_eff_rank > max(fpga_raw_eff_rank, esn_eff_rank),
      f'{bridge_eff_rank:.2f} > max({fpga_raw_eff_rank:.2f}, {esn_eff_rank:.2f})')

    results['tests'] = tests

    # Summary
    n_pass = sum(1 for t in tests.values() if t['passed'])
    n_total = len(tests)
    results['summary'] = {
        'pass': n_pass,
        'total': n_total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_results(results)

    print(f"\n{'='*70}")
    print(f"z2315 SUMMARY: {n_pass}/{n_total} PASS")
    print(f"{'='*70}")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
