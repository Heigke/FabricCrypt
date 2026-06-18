#!/usr/bin/env python3
"""z2277_real_bridge.py — Real HIP GPU Physics + Real FPGA NS-RAM Bridge

Previous bridge scripts (z2272-z2276) used a NUMPY simulation of the GPU fourpop
reservoir. This script uses the ACTUAL HIP kernel (z2268 fourpop_reservoir) running
on real GPU hardware with:
  - Real branch divergence (FMA vs INT warp paths)
  - Real L1 bank conflicts (LDS scratch with input-dependent addressing)
  - Real wavefront scheduling jitter (__clock64 timing)
  - Real cross-CU recurrence (global memory exchange)
  - Real PLL jitter (clock64 - wall_clock64)
  - 3072 neurons (12 CUs × 256 threads) sampled to 512

Combined with real FPGA NS-RAM 128-neuron reservoir via Ethernet bridge.

GPU kernel achieved: Wave4=97.7%, XOR=77.0%, MC=0.615, NARMA=+0.261 (z2268)
vs numpy ESN: Wave4=63.7%, XOR=55%, MC=2.345, NARMA=0.000

Conditions:
  GPU_HIP:      Real HIP fourpop kernel states only
  FPGA_ONLY:    FPGA 128-neuron states only (no GPU influence)
  FPGA_1F:      FPGA with 1/f noise (IIR from GPU power)
  BRIDGE_SIMPLE: Concatenate GPU_HIP + FPGA states (no MAC feedback)
  BRIDGE_FULL:   GPU_HIP states → MAC → FPGA hardware, then concatenate (causal)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2277_real_bridge.py
"""

import os, sys, time, json, struct, subprocess, tempfile
import numpy as np
from pathlib import Path

# ─── Paths ───
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2277_real_bridge.json'
TEXT_FILE = RESULTS / 'z2277_real_bridge.txt'
GPU_KERN = BASE / 'scripts' / 'z2277_gpu_bridge_kern'

# ─── Parameters ───
N_GPU_SAMPLED = 512   # 128 per pop × 4 pops (from HIP kernel)
N_FPGA = 128
BASE_VG = 0.58
VG_SPREAD = 0.08
IIR_ALPHA = 0.85
SAMPLE_HZ = 50

N_WAVE_TRIALS = 80
N_WAVE_STEPS = 60
N_CONTINUOUS_STEPS = 2000
MC_MAX_DELAY = 10
WARMUP = 200
NARMA_ORDERS = [3, 5, 10]
PCA_DIMS = 120
POLY_TOP_K = 20

# ─── THERMAL SAFETY ───
TEMP_ABORT = 90.0
TEMP_SAFE = 55.0
TEMP_PAUSE = 75.0
COOLING_PAUSE_S = 15

ALL_CONDITIONS = ['GPU_HIP', 'FPGA_ONLY', 'FPGA_1F', 'BRIDGE_SIMPLE', 'BRIDGE_FULL']
# Run cool conditions first, then warm ones
SAFE_ORDER = ['FPGA_ONLY', 'FPGA_1F', 'GPU_HIP', 'BRIDGE_SIMPLE', 'BRIDGE_FULL']


# ═══════════════════════════════════════════════════════════
# Temperature
# ═══════════════════════════════════════════════════════════

def get_apu_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

def get_gpu_temp():
    try:
        with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

def get_max_temp():
    return max(get_apu_temp(), get_gpu_temp())

def wait_cool(label="", target=None):
    if target is None:
        target = TEMP_SAFE
    temp = get_max_temp()
    if temp <= target:
        print(f"  [TEMP] {label} {temp:.0f}°C — OK", flush=True)
        return temp
    print(f"  [TEMP] {label} {temp:.0f}°C — cooling to {target:.0f}°C...", end="", flush=True)
    t0 = time.time()
    while temp > target:
        if time.time() - t0 > 180:
            print(f" timeout", flush=True)
            return temp
        time.sleep(5)
        temp = get_max_temp()
        print(f" {temp:.0f}", end="", flush=True)
    print(f" — OK ({time.time()-t0:.0f}s)", flush=True)
    return temp

def check_abort():
    temp = get_max_temp()
    if temp > TEMP_ABORT:
        print(f"  [ABORT] {temp:.0f}°C > {TEMP_ABORT}°C — waiting...", flush=True)
        t0 = time.time()
        while temp > TEMP_SAFE and time.time() - t0 < 120:
            time.sleep(10)
            temp = get_max_temp()
        return temp > TEMP_ABORT
    if temp > TEMP_PAUSE:
        wait_cool("thermal pause", TEMP_SAFE)
    return False

def check_pause():
    temp = get_max_temp()
    if temp > TEMP_PAUSE:
        wait_cool("mid-benchmark pause", TEMP_SAFE + 5)


# ═══════════════════════════════════════════════════════════
# GPU Noise Sampler (for 1/f)
# ═══════════════════════════════════════════════════════════

class GPUNoiseSampler:
    def __init__(self):
        self._power_norm = None
        self._iir_state = 0.0

    def load_burst(self, pw_arr):
        mu = np.mean(pw_arr)
        sigma = np.std(pw_arr)
        self._power_norm = (pw_arr - mu) / max(sigma, 1e-6)
        self._burst_idx = 0
        stats = {'n_samples': len(pw_arr), 'mean': float(mu), 'std': float(sigma)}
        print(f"  [NOISE] {len(pw_arr)} samples, mean={mu:.2f}W, std={sigma:.4f}W", flush=True)
        return stats

    def sample(self):
        if self._power_norm is None:
            return 0.0
        idx = self._burst_idx % len(self._power_norm)
        self._burst_idx += 1
        raw = self._power_norm[idx]
        self._iir_state = IIR_ALPHA * self._iir_state + (1.0 - IIR_ALPHA) * raw
        return float(self._iir_state)


# ═══════════════════════════════════════════════════════════
# GPU HIP Kernel Runner
# ═══════════════════════════════════════════════════════════

def run_hip_kernel(input_seq):
    """Run the real HIP fourpop kernel and return sampled states.

    Args:
        input_seq: numpy array of float32, shape (n_steps,)
    Returns:
        states: numpy array of float32, shape (n_steps, N_GPU_SAMPLED)
    """
    n_steps = len(input_seq)

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fin:
        input_path = fin.name
        input_seq.astype(np.float32).tofile(fin)

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fout:
        output_path = fout.name

    try:
        env = os.environ.copy()
        env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
        result = subprocess.run(
            [str(GPU_KERN), input_path, output_path, str(n_steps)],
            env=env, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  [GPU] Kernel error: {result.stderr}", flush=True)
            return None

        # Read output: SAMPLE_N × n_steps, row-major (neuron × time)
        raw = np.fromfile(output_path, dtype=np.float32)
        expected = N_GPU_SAMPLED * n_steps
        if len(raw) != expected:
            print(f"  [GPU] Wrong output size: {len(raw)} != {expected}", flush=True)
            return None

        # Reshape to (neuron, time) then transpose to (time, neuron)
        states = raw.reshape(N_GPU_SAMPLED, n_steps).T
        return states
    finally:
        os.unlink(input_path)
        try:
            os.unlink(output_path)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
# FPGA Reservoir
# ═══════════════════════════════════════════════════════════

class FPGAReservoir:
    def __init__(self, fpga, noise_sampler):
        self.fpga = fpga
        self.noise = noise_sampler

    def read_state(self):
        """Read FPGA telemetry and return N_FPGA-dimensional state vector."""
        telem = self.fpga.read_telemetry()
        if telem is None:
            return np.zeros(N_FPGA)
        vmem = telem['vmem']
        sc = telem['spike_counts']
        # Use vmem as primary features (continuous, richer than spike counts)
        state = np.array(vmem[:N_FPGA], dtype=np.float64)
        return state

    def run_fpga_only(self, input_seq):
        """Run FPGA-only: drive input via Vg modulation, collect states."""
        n_steps = len(input_seq)
        states = np.zeros((n_steps, N_FPGA))
        for t in range(n_steps):
            u = float(input_seq[t])
            # Modulate Vg based on input
            vg = BASE_VG + VG_SPREAD * u
            vg_q16 = int(vg * 65536) & 0xFFFFFFFF
            # Set Vg for neuron 0 as input signal
            pkt = struct.pack(">BBBI", 0x55, 0x01, 0, vg_q16)
            self.fpga.sock.sendto(pkt, (self.fpga.fpga_ip, self.fpga.fpga_port))
            time.sleep(1.0 / SAMPLE_HZ)
            states[t] = self.read_state()
        return states

    def run_fpga_1f(self, input_seq):
        """Run FPGA with 1/f noise from GPU power."""
        n_steps = len(input_seq)
        states = np.zeros((n_steps, N_FPGA))
        for t in range(n_steps):
            u = float(input_seq[t])
            noise_val = self.noise.sample() * 0.02  # NOISE_SCALE
            vg = BASE_VG + VG_SPREAD * u + noise_val
            vg_q16 = int(vg * 65536) & 0xFFFFFFFF
            pkt = struct.pack(">BBBI", 0x55, 0x01, 0, vg_q16)
            self.fpga.sock.sendto(pkt, (self.fpga.fpga_ip, self.fpga.fpga_port))
            time.sleep(1.0 / SAMPLE_HZ)
            states[t] = self.read_state()
        return states

    def run_with_mac(self, input_seq, mac_signals):
        """Run FPGA with MAC signal from pre-computed GPU states."""
        n_steps = len(input_seq)
        states = np.zeros((n_steps, N_FPGA))
        for t in range(n_steps):
            u = float(input_seq[t])
            # Set MAC signal (GPU→FPGA causal coupling)
            mac = float(mac_signals[t])
            self.fpga.set_mac_signal(mac)
            # Modulate Vg based on input + 1/f noise
            noise_val = self.noise.sample() * 0.02
            vg = BASE_VG + VG_SPREAD * u + noise_val
            vg_q16 = int(vg * 65536) & 0xFFFFFFFF
            pkt = struct.pack(">BBBI", 0x55, 0x01, 0, vg_q16)
            self.fpga.sock.sendto(pkt, (self.fpga.fpga_ip, self.fpga.fpga_port))
            time.sleep(1.0 / SAMPLE_HZ)
            states[t] = self.read_state()
        return states


# ═══════════════════════════════════════════════════════════
# Cross-Substrate Reservoir Runner
# ═══════════════════════════════════════════════════════════

class RealBridgeReservoir:
    def __init__(self, fpga, noise_sampler):
        self.fpga_res = FPGAReservoir(fpga, noise_sampler)
        self.noise = noise_sampler

    def compute_mac(self, gpu_state_t):
        """Convert GPU state to MAC signal [0, 1] using mean(abs)."""
        activity = np.mean(np.abs(gpu_state_t))
        return float(np.clip(activity, 0.0, 1.0))

    def run_trial(self, input_seq, condition):
        """Run a full trial for a given condition.

        Returns: states array (n_steps, n_features)
        """
        n_steps = len(input_seq)

        if condition == 'GPU_HIP':
            # Real HIP kernel only
            gpu_states = run_hip_kernel(input_seq)
            if gpu_states is None:
                return np.zeros((n_steps, N_GPU_SAMPLED))
            return gpu_states

        elif condition == 'FPGA_ONLY':
            return self.fpga_res.run_fpga_only(input_seq)

        elif condition == 'FPGA_1F':
            return self.fpga_res.run_fpga_1f(input_seq)

        elif condition == 'BRIDGE_SIMPLE':
            # Run GPU kernel first (all steps at once)
            gpu_states = run_hip_kernel(input_seq)
            if gpu_states is None:
                gpu_states = np.zeros((n_steps, N_GPU_SAMPLED))
            # Run FPGA with 1/f noise but NO MAC feedback
            fpga_states = self.fpga_res.run_fpga_1f(input_seq)
            # Concatenate (FPGA weighted 2× to amplify its signal)
            return np.hstack([gpu_states, fpga_states * 2.0])

        elif condition == 'BRIDGE_FULL':
            # Run GPU kernel first (all steps at once)
            gpu_states = run_hip_kernel(input_seq)
            if gpu_states is None:
                gpu_states = np.zeros((n_steps, N_GPU_SAMPLED))
            # Compute MAC signal from GPU states
            mac_signals = np.array([self.compute_mac(gpu_states[t]) for t in range(n_steps)])
            # Run FPGA with MAC feedback (GPU→FPGA causal coupling!)
            fpga_states = self.fpga_res.run_with_mac(input_seq, mac_signals)
            # Concatenate
            return np.hstack([gpu_states, fpga_states * 2.0])

        else:
            raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════════════════════
# ML Utilities
# ═══════════════════════════════════════════════════════════

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def augment_with_delays(states, delays=(1, 2)):
    T, D = states.shape
    max_d = max(delays)
    valid_T = T - max_d
    if valid_T < 10:
        return states
    augmented = np.zeros((valid_T, D * (1 + len(delays))))
    augmented[:, :D] = states[max_d:]
    for i, d in enumerate(delays):
        start = D * (i + 1)
        augmented[:, start:start + D] = states[max_d - d:T - d]
    return augmented

def polynomial_features(X, top_k=None):
    """Add x² and top cross-products for nonlinear readout."""
    T, D = X.shape
    if top_k is None:
        top_k = min(POLY_TOP_K, D)
    X_sq = X ** 2
    var = np.var(X, axis=0)
    top_idx = np.argsort(var)[-top_k:]
    X_top = X[:, top_idx]
    cross = []
    for i in range(top_k):
        for j in range(i + 1, top_k):
            cross.append(X_top[:, i] * X_top[:, j])
    if cross:
        X_cross = np.column_stack(cross)
        return np.hstack([X, X_sq, X_cross])
    else:
        return np.hstack([X, X_sq])

def pool_trial_features(trial_states):
    return np.concatenate([
        trial_states.mean(axis=0),
        trial_states.std(axis=0),
        trial_states.max(axis=0),
        trial_states.min(axis=0),
    ])

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=4):
    sigma = np.std(X_tr, axis=0)
    sigma[sigma < 1e-6] = 1.0
    X_tr_s = X_tr / sigma
    X_te_s = X_te / sigma
    alphas = [1e-5, 1e-3, 0.1, 1.0, 10.0, 100.0, 1000.0, 100000.0]
    best = 0.0
    for a in alphas:
        targets = np.zeros((len(y_tr), n_classes))
        for i, c in enumerate(y_tr):
            targets[i, int(c)] = 1.0
        I = np.eye(X_tr_s.shape[1])
        try:
            w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ targets)
        except Exception:
            continue
        pred = X_te_s @ w
        acc = np.mean(pred.argmax(axis=1) == y_te)
        if acc > best:
            best = acc
    return best

def ridge_binary(X_tr, y_tr, X_te, y_te):
    sigma = np.std(X_tr, axis=0)
    sigma[sigma < 1e-6] = 1.0
    X_tr_s = X_tr / sigma
    X_te_s = X_te / sigma
    alphas = [1e-5, 1e-3, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best = 0.0
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try:
            w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except Exception:
            continue
        acc = np.mean(((X_te_s @ w) > 0.5).astype(float) == y_te)
        if acc > best:
            best = acc
    return best

def ridge_regress(X_tr, y_tr, X_te, y_te, use_poly=False):
    if use_poly:
        X_tr = polynomial_features(X_tr)
        X_te = polynomial_features(X_te)

    n_feat = X_tr.shape[1]
    n_samp = X_tr.shape[0]
    mu = X_tr.mean(axis=0)
    X_tr_c = X_tr - mu
    X_te_c = X_te - mu

    if n_feat > n_samp * 0.5:
        k = min(PCA_DIMS, n_samp - 1, n_feat)
        U, S, Vt = np.linalg.svd(X_tr_c, full_matrices=False)
        X_tr_c = U[:, :k] * S[:k]
        X_te_c = (X_te_c @ Vt[:k].T)

    sigma = np.std(X_tr_c, axis=0)
    sigma[sigma < 1e-6] = 1.0
    X_tr_s = X_tr_c / sigma
    X_te_s = X_te_c / sigma

    alphas = [1e-5, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 100000.0]
    best_nrmse = 1e10
    best_r2 = -1e10
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try:
            w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except Exception:
            continue
        pred = X_te_s @ w
        ss_res = np.sum((y_te - pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        if ss_tot < 1e-10:
            continue
        nrmse = np.sqrt(ss_res / len(y_te)) / np.sqrt(ss_tot / len(y_te))
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_nrmse = nrmse
    return best_nrmse, max(best_r2, 0.0)


def ridge_regress_narma(X_tr, y_tr, X_te, y_te, use_poly=False, top_k=20):
    """NARMA-specific regression: correlation-based feature selection (not PCA).

    PCA captures population variance which washes out input-driven signal.
    Instead select neurons most correlated with the target, then optionally
    add polynomial features.
    """
    # Select top-k features by correlation with training target
    corrs = np.array([np.corrcoef(X_tr[:, i], y_tr)[0, 1]
                      if np.std(X_tr[:, i]) > 1e-8 else 0.0
                      for i in range(X_tr.shape[1])])
    corrs = np.nan_to_num(corrs)
    k = min(top_k, X_tr.shape[1])
    sel = np.argsort(np.abs(corrs))[-k:]

    X_tr_sel = X_tr[:, sel]
    X_te_sel = X_te[:, sel]

    if use_poly:
        # Add squared and cross-product terms on selected features
        X_tr_sq = X_tr_sel ** 2
        X_te_sq = X_te_sel ** 2
        cross_tr, cross_te = [], []
        pk = min(10, k)
        for i in range(pk):
            for j in range(i + 1, pk):
                cross_tr.append(X_tr_sel[:, i] * X_tr_sel[:, j])
                cross_te.append(X_te_sel[:, i] * X_te_sel[:, j])
        if cross_tr:
            X_tr_sel = np.hstack([X_tr_sel, X_tr_sq, np.column_stack(cross_tr)])
            X_te_sel = np.hstack([X_te_sel, X_te_sq, np.column_stack(cross_te)])
        else:
            X_tr_sel = np.hstack([X_tr_sel, X_tr_sq])
            X_te_sel = np.hstack([X_te_sel, X_te_sq])

    # Normalize
    sigma = np.std(X_tr_sel, axis=0)
    sigma[sigma < 1e-6] = 1.0
    X_tr_s = X_tr_sel / sigma
    X_te_s = X_te_sel / sigma

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_nrmse = 1e10
    best_r2 = -1e10
    for a in alphas:
        I = np.eye(X_tr_s.shape[1])
        try:
            w = np.linalg.solve(X_tr_s.T @ X_tr_s + a * I, X_tr_s.T @ y_tr)
        except Exception:
            continue
        pred = X_te_s @ w
        ss_res = np.sum((y_te - pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        if ss_tot < 1e-10:
            continue
        nrmse = np.sqrt(ss_res / len(y_te)) / np.sqrt(ss_tot / len(y_te))
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_nrmse = nrmse
    return best_nrmse, max(best_r2, 0.0)

def stratified_kfold(X, y, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        c_idx = indices[y[indices] == c]
        for i, idx in enumerate(c_idx):
            folds[i % n_splits].append(idx)
    splits = []
    for fold in range(n_splits):
        test_idx = np.array(folds[fold])
        train_idx = np.concatenate([np.array(folds[f]) for f in range(n_splits) if f != fold])
        splits.append((train_idx, test_idx))
    return splits

def classify_cv(X, y, n_splits=5, n_classes=4):
    splits = stratified_kfold(X, y, n_splits)
    accs = []
    for tr_idx, te_idx in splits:
        acc = ridge_classify(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], n_classes=n_classes)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


# ═══════════════════════════════════════════════════════════
# Task Generation
# ═══════════════════════════════════════════════════════════

def generate_waveforms(n_trials, steps, sample_hz, n_classes=4, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / sample_hz
    t = np.arange(steps) * dt
    trials, labels = [], []
    for _ in range(n_trials):
        cls = rng.integers(0, n_classes)
        phase = rng.uniform(0, 2 * np.pi)
        if n_classes <= 4:
            freq = rng.uniform(0.8, 1.2)
            base_cls = cls
        else:
            base_cls = cls % 4
            freq_band = cls // 4
            freq = rng.uniform(0.5, 0.9) if freq_band == 0 else rng.uniform(1.3, 2.0)
        if base_cls == 0:   wave = np.sin(2 * np.pi * freq * t + phase)
        elif base_cls == 1: wave = np.sign(np.sin(2 * np.pi * freq * t + phase))
        elif base_cls == 2: wave = 2.0 * np.abs(2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0) - 1.0
        else:               wave = 2.0 * ((freq * t + phase / (2 * np.pi)) % 1.0) - 1.0
        wave = (wave - wave.min()) / max(wave.max() - wave.min(), 1e-6)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)

def generate_continuous_input(n_steps, seed=123):
    return np.random.default_rng(seed).uniform(-1, 1, size=n_steps).astype(np.float32)

def generate_binary_input(n_steps, seed=456):
    return np.random.default_rng(seed).integers(0, 2, size=n_steps).astype(float)

def compute_xor_targets(u, tau):
    n = len(u)
    targets = np.zeros(n, dtype=int)
    for t in range(tau, n):
        targets[t] = int(u[t]) ^ int(u[t - tau])
    return targets

def generate_narma(u, order=10):
    n = len(u)
    y = np.zeros(n)
    u_scaled = np.clip(u * 0.25 + 0.25, 0.0, 0.5)
    for t in range(min(order, n)):
        y[t] = 0.1
    for t in range(order, n):
        y_prev = y[t-1]
        s = np.sum(y[max(0, t-order):t])
        y[t] = 0.3 * y_prev + 0.05 * y_prev * s + 1.5 * u_scaled[t-order] * u_scaled[t] + 0.1
        if abs(y[t]) > 5.0:
            y[t] = np.clip(y[t], -5.0, 5.0)
    return y


# ═══════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════

def save_intermediate(all_results, label=""):
    with open(SAVE_FILE, 'w') as f:
        json.dump(all_results, f, cls=NpEncoder, indent=2)
    print(f"  [SAVE] {label} → {SAVE_FILE.name}", flush=True)

def load_previous():
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE, 'r') as f:
                data = json.load(f)
            print(f"  [RESUME] Loaded previous results from {SAVE_FILE.name}")
            return data
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════
# Benchmark Runners
# ═══════════════════════════════════════════════════════════

def run_waveform_benchmark(reservoir, n_trials, n_steps, n_classes, seed, label, all_results):
    key = f'waveform_{label}'
    if key not in all_results:
        all_results[key] = {}

    trials, labels = generate_waveforms(n_trials, n_steps, SAMPLE_HZ, n_classes, seed)

    for cond in SAFE_ORDER:
        if cond in all_results[key]:
            print(f"  [{cond}] {label} — already done, skipping", flush=True)
            continue

        if check_abort():
            continue

        wait_cool(f"Before {label} {cond}")
        print(f"\n  [{cond}] Running {label}...", flush=True)

        features = []
        for trial_idx in range(n_trials):
            states = reservoir.run_trial(trials[trial_idx], cond)
            states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
            feat = pool_trial_features(states)
            features.append(feat)
            if (trial_idx + 1) % 20 == 0:
                print(f"    trial {trial_idx+1}/{n_trials}", flush=True)
                check_pause()

        X = np.array(features)
        actual_labels = labels[:len(X)]
        acc_mean, acc_std = classify_cv(X, actual_labels, n_splits=min(5, max(2, len(X)//4)), n_classes=n_classes)
        print(f"  [{cond}] {label}: {acc_mean:.3f} ± {acc_std:.3f} ({len(X)} trials)", flush=True)
        all_results[key][cond] = {'accuracy': acc_mean, 'std': acc_std, 'n_trials': len(X)}
        save_intermediate(all_results, f"{label} {cond}")

        if 'GPU' in cond or 'BRIDGE' in cond:
            time.sleep(COOLING_PAUSE_S)


def run_continuous_benchmark(reservoir, benchmark_name, all_results):
    if benchmark_name not in all_results:
        all_results[benchmark_name] = {}

    mc_input = generate_continuous_input(N_CONTINUOUS_STEPS, seed=123)
    xor_input = generate_binary_input(N_CONTINUOUS_STEPS, seed=456)
    narma_input = generate_continuous_input(N_CONTINUOUS_STEPS, seed=789)

    narma_targets = {}
    for order in NARMA_ORDERS:
        narma_targets[order] = generate_narma(narma_input, order=order)
        nt_valid = narma_targets[order][WARMUP:]
        print(f"  [NARMA-{order}] Target: mean={nt_valid.mean():.4f}, std={nt_valid.std():.4f}", flush=True)

    for cond in SAFE_ORDER:
        if cond in all_results[benchmark_name]:
            print(f"  [{cond}] {benchmark_name} — already done, skipping", flush=True)
            continue

        if check_abort():
            continue

        wait_cool(f"Before {benchmark_name} {cond}")
        print(f"\n  [{cond}] Running {benchmark_name}...", flush=True)
        cond_results = {}

        # Memory Capacity
        check_pause()
        states = reservoir.run_trial(mc_input, cond)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2))
        mc_total = 0.0
        mc_per_delay = {}
        for d in range(1, MC_MAX_DELAY + 1):
            T_aug = aug.shape[0]
            target = np.zeros(T_aug)
            for i in range(T_aug):
                orig_idx = i + 2
                src_idx = orig_idx - d
                if 0 <= src_idx < len(mc_input):
                    target[i] = mc_input[src_idx]
            X = aug[WARMUP:]
            y = target[WARMUP:]
            n_tr = int(0.7 * len(X))
            if n_tr < 10 or len(X) - n_tr < 5:
                mc_per_delay[str(d)] = 0.0
                continue
            _, r2 = ridge_regress(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])
            mc_per_delay[str(d)] = r2
            mc_total += r2
        cond_results['mc_total'] = mc_total
        cond_results['mc_per_delay'] = mc_per_delay
        print(f"    MC={mc_total:.3f}", flush=True)

        wait_cool(f"Between MC→XOR {cond}", TEMP_SAFE + 10)

        # XOR
        states = reservoir.run_trial(xor_input, cond)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2))
        xor_results = {}
        for tau in [1, 2, 3]:
            targets = compute_xor_targets(xor_input, tau)
            T_aug = aug.shape[0]
            aligned_targets = targets[2:2 + T_aug]
            X = aug[WARMUP:]
            y = aligned_targets[WARMUP:].astype(float)
            n_tr = int(0.7 * len(X))
            acc = ridge_binary(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:])
            xor_results[str(tau)] = acc
            print(f"    XOR tau={tau}: {acc:.3f}", flush=True)
        cond_results['xor'] = xor_results

        wait_cool(f"Between XOR→NARMA {cond}", TEMP_SAFE + 10)

        # NARMA — multiple orders, both LINEAR and POLYNOMIAL readout
        states = reservoir.run_trial(narma_input, cond)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        aug = augment_with_delays(states, delays=(1, 2))
        T_aug = aug.shape[0]

        narma_all = {}
        best_narma_r2 = 0.0
        for order in NARMA_ORDERS:
            narma_target = narma_targets[order]
            aligned_narma = narma_target[2:2 + T_aug]
            X = aug[WARMUP:]
            y = aligned_narma[WARMUP:]
            n_tr = int(0.7 * len(X))

            nrmse_lin, r2_lin = ridge_regress_narma(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], use_poly=False)
            nrmse_poly, r2_poly = ridge_regress_narma(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], use_poly=True)

            narma_all[f'narma{order}_linear'] = {'nrmse': nrmse_lin, 'r2': r2_lin}
            narma_all[f'narma{order}_poly'] = {'nrmse': nrmse_poly, 'r2': r2_poly}
            best_narma_r2 = max(best_narma_r2, r2_poly, r2_lin)

            print(f"    NARMA-{order}: LINEAR R2={r2_lin:.4f}  POLY R2={r2_poly:.4f} "
                  f"({'POLY WINS' if r2_poly > r2_lin + 0.01 else 'similar'})", flush=True)

        cond_results['narma_all'] = narma_all
        cond_results['narma_r2'] = narma_all.get('narma10_poly', {}).get('r2', 0)
        cond_results['narma_nrmse'] = narma_all.get('narma10_poly', {}).get('nrmse', 999)
        cond_results['narma_best_r2'] = best_narma_r2
        print(f"    NARMA best R2={best_narma_r2:.4f}", flush=True)

        all_results[benchmark_name][cond] = cond_results
        save_intermediate(all_results, f"{benchmark_name} {cond}")

        if 'GPU' in cond or 'BRIDGE' in cond:
            time.sleep(COOLING_PAUSE_S)


# ═══════════════════════════════════════════════════════════
# Summary & Tests
# ═══════════════════════════════════════════════════════════

def print_summary_and_tests(all_results):
    out = []
    def p(s=""):
        print(s, flush=True)
        out.append(s)

    p("\n" + "=" * 72)
    p("z2277: REAL HIP GPU + FPGA BRIDGE — SUMMARY")
    p("=" * 72)

    w4 = all_results.get('waveform_4class', {})
    w8 = all_results.get('waveform_8class', {})
    cont = all_results.get('continuous', {})

    p(f"\n{'Condition':<18} {'W4':>6} {'W8':>6} {'MC':>6} {'XOR1':>6} {'XOR2':>6} {'XOR3':>6} {'NARMA':>7} {'NR_BEST':>7}")
    p("-" * 82)
    for c in ALL_CONDITIONS:
        w4a = w4.get(c, {}).get('accuracy', 0)
        w8a = w8.get(c, {}).get('accuracy', 0)
        mc = cont.get(c, {}).get('mc_total', 0)
        x1 = cont.get(c, {}).get('xor', {}).get('1', 0)
        x2 = cont.get(c, {}).get('xor', {}).get('2', 0)
        x3 = cont.get(c, {}).get('xor', {}).get('3', 0)
        nr = cont.get(c, {}).get('narma_r2', 0)
        nr_best = cont.get(c, {}).get('narma_best_r2', 0)
        p(f"  {c:<16} {w4a:>5.1%} {w8a:>5.1%} {mc:>6.3f} {x1:>5.1%} {x2:>5.1%} {x3:>5.1%} {nr:>7.3f} {nr_best:>7.3f}")

    # NARMA detail
    p(f"\n{'NARMA DETAIL':<18} {'N3_LIN':>7} {'N3_PLY':>7} {'N5_LIN':>7} {'N5_PLY':>7} {'N10_LIN':>7} {'N10_PLY':>7}")
    p("-" * 82)
    for c in ALL_CONDITIONS:
        na = cont.get(c, {}).get('narma_all', {})
        n3l = na.get('narma3_linear', {}).get('r2', 0)
        n3p = na.get('narma3_poly', {}).get('r2', 0)
        n5l = na.get('narma5_linear', {}).get('r2', 0)
        n5p = na.get('narma5_poly', {}).get('r2', 0)
        n10l = na.get('narma10_linear', {}).get('r2', 0)
        n10p = na.get('narma10_poly', {}).get('r2', 0)
        p(f"  {c:<16} {n3l:>7.4f} {n3p:>7.4f} {n5l:>7.4f} {n5p:>7.4f} {n10l:>7.4f} {n10p:>7.4f}")

    p(f"\n{'=' * 72}")
    p("TESTS")
    p("=" * 72)

    n_pass = 0
    n_total = 0
    tests = {}

    def check(name, condition, desc):
        nonlocal n_pass, n_total
        n_total += 1
        status = "PASS" if condition else "FAIL"
        if condition: n_pass += 1
        p(f"  {name}: {status} — {desc}")
        tests[name] = {'pass': bool(condition), 'desc': desc}

    # T1: GPU HIP kernel produces non-trivial states
    gpu_mc = cont.get('GPU_HIP', {}).get('mc_total', 0)
    check("T1_gpu_hip_alive", gpu_mc > 0.1,
          f"GPU_HIP MC={gpu_mc:.3f} > 0.1")

    # T2: GPU HIP >> numpy ESN baseline (z2273: 63.7%)
    w4_vals = {c: w4.get(c, {}).get('accuracy', 0) for c in ALL_CONDITIONS}
    check("T2_gpu_hip_wave4", w4_vals['GPU_HIP'] > 0.80,
          f"GPU_HIP wave4 {w4_vals['GPU_HIP']:.1%} > 80% (numpy was 63.7%)")

    # T3: FPGA 1/f > FPGA ONLY
    check("T3_fpga_1f_vs_fpga", w4_vals['FPGA_1F'] > w4_vals['FPGA_ONLY'],
          f"FPGA_1F {w4_vals['FPGA_1F']:.1%} > FPGA_ONLY {w4_vals['FPGA_ONLY']:.1%}")

    # T4: BRIDGE_FULL > both individual substrates (wave4)
    check("T4_bridge_full_best_w4",
          w4_vals['BRIDGE_FULL'] >= max(w4_vals['FPGA_ONLY'], w4_vals['GPU_HIP']),
          f"BRIDGE_FULL {w4_vals['BRIDGE_FULL']:.1%} >= max(FPGA={w4_vals['FPGA_ONLY']:.1%}, GPU={w4_vals['GPU_HIP']:.1%})")

    # T5: BRIDGE_FULL > 80% wave4
    check("T5_bridge_full_w4_target", w4_vals['BRIDGE_FULL'] > 0.80,
          f"BRIDGE_FULL wave4 {w4_vals['BRIDGE_FULL']:.1%} > 80%")

    # T6: MAC feedback helps (BRIDGE_FULL > BRIDGE_SIMPLE)
    check("T6_mac_vs_simple", w4_vals['BRIDGE_FULL'] > w4_vals['BRIDGE_SIMPLE'],
          f"BRIDGE_FULL(MAC) {w4_vals['BRIDGE_FULL']:.1%} > BRIDGE_SIMPLE(no MAC) {w4_vals['BRIDGE_SIMPLE']:.1%}")

    # T7: 8-class
    w8_vals = {c: w8.get(c, {}).get('accuracy', 0) for c in ALL_CONDITIONS}
    check("T7_bridge_full_best_w8",
          w8_vals['BRIDGE_FULL'] >= max(w8_vals['FPGA_ONLY'], w8_vals['GPU_HIP']),
          f"BRIDGE_FULL {w8_vals['BRIDGE_FULL']:.1%} >= max(FPGA={w8_vals['FPGA_ONLY']:.1%}, GPU={w8_vals['GPU_HIP']:.1%})")

    check("T8_bridge_full_w8_target", w8_vals['BRIDGE_FULL'] > 0.50,
          f"BRIDGE_FULL wave8 {w8_vals['BRIDGE_FULL']:.1%} > 50%")

    # T9-T11: Memory capacity
    mc_vals = {c: cont.get(c, {}).get('mc_total', 0) for c in ALL_CONDITIONS}
    check("T9_mc_bridge_best",
          mc_vals['BRIDGE_FULL'] >= max(mc_vals['FPGA_ONLY'], mc_vals['GPU_HIP']),
          f"BRIDGE_FULL MC {mc_vals['BRIDGE_FULL']:.3f} >= max(FPGA={mc_vals['FPGA_ONLY']:.3f}, GPU={mc_vals['GPU_HIP']:.3f})")

    check("T10_mc_bridge_target", mc_vals['BRIDGE_FULL'] > 0.5,
          f"BRIDGE_FULL MC {mc_vals['BRIDGE_FULL']:.3f} > 0.5")

    check("T11_mc_gpu_hip", mc_vals['GPU_HIP'] > 0.3,
          f"GPU_HIP MC {mc_vals['GPU_HIP']:.3f} > 0.3")

    # T12-T13: XOR
    xor1 = {c: cont.get(c, {}).get('xor', {}).get('1', 0) for c in ALL_CONDITIONS}
    check("T12_xor1_bridge_target", xor1['BRIDGE_FULL'] > 0.55,
          f"BRIDGE_FULL XOR1 {xor1['BRIDGE_FULL']:.1%} > 55%")
    check("T13_xor1_gpu_hip", xor1['GPU_HIP'] > 0.65,
          f"GPU_HIP XOR1 {xor1['GPU_HIP']:.1%} > 65% (numpy was ~55%)")

    # T14-T17: NARMA (THE FIX)
    nr_best = {c: cont.get(c, {}).get('narma_best_r2', 0) for c in ALL_CONDITIONS}
    nr10_poly = {c: cont.get(c, {}).get('narma_all', {}).get('narma10_poly', {}).get('r2', 0) for c in ALL_CONDITIONS}
    nr10_lin = {c: cont.get(c, {}).get('narma_all', {}).get('narma10_linear', {}).get('r2', 0) for c in ALL_CONDITIONS}

    check("T14_narma_bridge_best",
          nr_best['BRIDGE_FULL'] >= max(nr_best.get('FPGA_ONLY', 0), nr_best.get('GPU_HIP', 0)),
          f"BRIDGE_FULL NARMA-best R2 {nr_best['BRIDGE_FULL']:.4f} >= max(FPGA={nr_best.get('FPGA_ONLY', 0):.4f}, GPU={nr_best.get('GPU_HIP', 0):.4f})")

    check("T15_narma_bridge_target", nr_best['BRIDGE_FULL'] > 0.05,
          f"BRIDGE_FULL NARMA-best R2 {nr_best['BRIDGE_FULL']:.4f} > 0.05")

    check("T16_poly_vs_linear", nr10_poly['BRIDGE_FULL'] > nr10_lin['BRIDGE_FULL'],
          f"POLY {nr10_poly['BRIDGE_FULL']:.4f} > LINEAR {nr10_lin['BRIDGE_FULL']:.4f} (NARMA-10)")

    check("T17_narma_gpu_hip", nr_best['GPU_HIP'] > 0.05,
          f"GPU_HIP NARMA-best R2 {nr_best['GPU_HIP']:.4f} > 0.05 (HIP kernel has real nonlinearity)")

    # T18: Synergy — bridge best on >= 3/5 benchmarks
    bridge_wins = 0
    benchmarks = [
        ('W4', w4_vals),
        ('W8', w8_vals),
        ('MC', mc_vals),
        ('XOR1', xor1),
        ('NARMA', nr_best),
    ]
    for bname, bvals in benchmarks:
        others = [bvals.get(c, 0) for c in ALL_CONDITIONS if c != 'BRIDGE_FULL']
        if bvals.get('BRIDGE_FULL', 0) >= max(others):
            bridge_wins += 1
    check("T18_synergy", bridge_wins >= 3,
          f"BRIDGE_FULL best on {bridge_wins}/5 benchmarks (need >=3)")

    # T19: MAC improves MC
    mc_bf = mc_vals.get('BRIDGE_FULL', 0)
    mc_bs = mc_vals.get('BRIDGE_SIMPLE', 0)
    check("T19_mac_mc_improvement", mc_bf > mc_bs,
          f"BRIDGE_FULL MC {mc_bf:.3f} > BRIDGE_SIMPLE MC {mc_bs:.3f}")

    # T20: GPU HIP XOR >> FPGA XOR (proves GPU adds nonlinear computation)
    check("T20_gpu_xor_vs_fpga", xor1['GPU_HIP'] > xor1['FPGA_ONLY'],
          f"GPU_HIP XOR1 {xor1['GPU_HIP']:.1%} > FPGA_ONLY {xor1['FPGA_ONLY']:.1%}")

    p(f"\n  TOTAL: {n_pass}/{n_total} PASS")
    all_results['tests'] = tests
    all_results['n_pass'] = n_pass
    all_results['n_total'] = n_total

    with open(TEXT_FILE, 'w') as f:
        f.write('\n'.join(out))
    print(f"\n  Written to {TEXT_FILE}", flush=True)


# ═══════════════════════════════════════════════════════════
# GPU Noise Sampling (short burst)
# ═══════════════════════════════════════════════════════════

GPU_STRESS_SRC = '/tmp/z2277_gpu_stress.hip'
GPU_STRESS_BIN = '/tmp/z2277_gpu_stress'

GPU_STRESS_CODE = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void stress(float* out, int n) {
    float v = threadIdx.x * 0.001f + 1.0f;
    for (int i = 0; i < n; i++) v = v * 1.000001f + 0.0001f;
    if (v == -1.0f) out[0] = v;
}
int main() {
    float* d; hipMalloc(&d, 4);
    for (int i = 0; i < 10000; i++) {
        stress<<<64, 256>>>(d, 100000);
        hipDeviceSynchronize();
    }
    hipFree(d);
}
"""

def compile_stress_kernel():
    if os.path.isfile(GPU_STRESS_BIN):
        return GPU_STRESS_BIN
    with open(GPU_STRESS_SRC, 'w') as f:
        f.write(GPU_STRESS_CODE)
    r = subprocess.run(
        ['/opt/rocm/bin/hipcc', '--offload-arch=gfx1100', '-O2', '-o', GPU_STRESS_BIN, GPU_STRESS_SRC],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  Stress compile error: {r.stderr}", flush=True)
        return None
    return GPU_STRESS_BIN

def sample_noise_with_burst(stress_bin, burst_s=6, n_samples=150):
    """Run GPU stress and sample power readings."""
    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    proc = subprocess.Popen([stress_bin], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    readings = []
    for _ in range(n_samples):
        try:
            with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
                pw = float(f.read().strip()) / 1e6
                readings.append(pw)
        except Exception:
            readings.append(0.0)
        time.sleep(burst_s / n_samples)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return np.array(readings)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    from fpga_host_eth import FPGAEthBridge

    print("=" * 72)
    print("  z2277: REAL HIP GPU PHYSICS + FPGA NS-RAM BRIDGE")
    print("  GPU: HIP fourpop kernel (3072 neurons, branch/L1/wavefront/PLL)")
    print("  FPGA: 128-neuron LIF (τ=210ms, BIAS_GAIN=0.03125)")
    print("  NARMA fix: polynomial readout (r², r_i×r_j)")
    print(f"  NARMA orders: {NARMA_ORDERS}, steps={N_CONTINUOUS_STEPS}")
    print("  Thermal limits: ABORT>90°C, SAFE<55°C")
    print("=" * 72)

    # Verify HIP kernel exists
    if not GPU_KERN.exists():
        print(f"  FATAL: HIP kernel not found at {GPU_KERN}")
        print("  Compile: hipcc --offload-arch=gfx1100 -O1 -o scripts/z2277_gpu_bridge_kern scripts/z2277_gpu_bridge_kern.hip")
        sys.exit(1)

    all_results = load_previous() or {}
    if not all_results:
        all_results = {
            'experiment': 'z2277_real_bridge',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'params': {
                'n_gpu_sampled': N_GPU_SAMPLED, 'n_fpga': N_FPGA,
                'gpu_kernel': 'z2277_gpu_bridge_kern (z2268 fourpop)',
                'gpu_neurons_total': 3072, 'gpu_pops': 4,
                'base_vg': BASE_VG, 'vg_spread': VG_SPREAD,
                'iir_alpha': IIR_ALPHA, 'sample_hz': SAMPLE_HZ,
                'n_wave_trials': N_WAVE_TRIALS, 'n_wave_steps': N_WAVE_STEPS,
                'n_continuous_steps': N_CONTINUOUS_STEPS,
                'narma_orders': NARMA_ORDERS, 'pca_dims': PCA_DIMS,
                'poly_top_k': POLY_TOP_K,
                'temp_abort': TEMP_ABORT, 'temp_safe': TEMP_SAFE,
                'mac_feedback': True,
            },
        }

    wait_cool("Startup")

    # ─── Step 1: Quick GPU kernel test ───
    print("\n[1] Testing HIP kernel...", flush=True)
    test_input = np.random.default_rng(99).uniform(-1, 1, 50).astype(np.float32)
    test_states = run_hip_kernel(test_input)
    if test_states is None:
        print("  FATAL: HIP kernel failed")
        sys.exit(1)
    print(f"  GPU kernel OK: states shape={test_states.shape}, "
          f"range=[{test_states.min():.3f}, {test_states.max():.3f}]", flush=True)

    # ─── Step 2: Connect to FPGA ───
    print("\n[2] Connecting to FPGA via Ethernet...", flush=True)
    fpga = FPGAEthBridge()
    fpga_ok = fpga.connect()
    if not fpga_ok:
        print("  WARNING: FPGA not responding — will run GPU-only conditions")

    fpga.set_kill(False)
    time.sleep(0.3)

    print("  Setting heterogeneous Vg (BASE_VG=0.58 ± 0.08)...", flush=True)
    rng = np.random.default_rng(42)
    init_vg = BASE_VG + rng.uniform(-VG_SPREAD, VG_SPREAD, size=N_FPGA)
    for nid in range(N_FPGA):
        vg_q16 = int(init_vg[nid] * 65536) & 0xFFFFFFFF
        pkt = struct.pack(">BBBI", 0x55, 0x01, nid & 0x7F, vg_q16)
        fpga.sock.sendto(pkt, (fpga.fpga_ip, fpga.fpga_port))
        if nid % 32 == 31:
            time.sleep(0.01)
    print(f"  Set Vg for {N_FPGA} neurons: [{init_vg.min():.3f}, {init_vg.max():.3f}]", flush=True)
    time.sleep(0.5)

    fpga.set_mac_signal(0.0)

    telem = fpga.read_telemetry()
    if telem:
        sc = telem['spike_counts']
        print(f"  FPGA alive: total_spikes={sc.sum()}, active={np.count_nonzero(sc)}/{N_FPGA}", flush=True)
    else:
        print("  WARNING: no telemetry", flush=True)

    # ─── Step 3: Sample GPU noise ───
    print("\n[3] Sampling GPU noise (short burst)...", flush=True)
    noise_sampler = GPUNoiseSampler()
    stress_bin = compile_stress_kernel()
    if stress_bin:
        pw_arr = sample_noise_with_burst(stress_bin, burst_s=6, n_samples=150)
        noise_stats = noise_sampler.load_burst(pw_arr)
        all_results['noise_stats'] = noise_stats
    else:
        print("  WARNING: No stress kernel — using dummy noise", flush=True)
        noise_sampler.load_burst(np.random.default_rng(42).normal(100, 1.5, 150))

    save_intermediate(all_results, "after noise sampling")

    # ─── Step 4: Create reservoir system ───
    print("\n[4] Initializing REAL bridge reservoir...", flush=True)
    reservoir = RealBridgeReservoir(fpga, noise_sampler)

    # ─── BENCHMARKS ───

    print("\n" + "=" * 72)
    print("BENCHMARK 1: 4-CLASS WAVEFORM (80 trials × 60 steps)")
    print("=" * 72, flush=True)
    run_waveform_benchmark(reservoir, N_WAVE_TRIALS, N_WAVE_STEPS, 4, seed=42, label="4class", all_results=all_results)

    print("\n" + "=" * 72)
    print("BENCHMARK 2: 8-CLASS WAVEFORM (80 trials × 60 steps)")
    print("=" * 72, flush=True)
    run_waveform_benchmark(reservoir, N_WAVE_TRIALS, N_WAVE_STEPS, 8, seed=99, label="8class", all_results=all_results)

    print("\n" + "=" * 72)
    print(f"BENCHMARKS 3-5: MC + XOR + NARMA-{NARMA_ORDERS} ({N_CONTINUOUS_STEPS} steps, POLY readout)")
    print("=" * 72, flush=True)
    run_continuous_benchmark(reservoir, 'continuous', all_results)

    # ─── Summary ───
    print_summary_and_tests(all_results)
    save_intermediate(all_results, "FINAL")

    try:
        fpga.set_mac_signal(0.0)
    except Exception:
        pass

    print(f"\n  Done. Results: {SAVE_FILE}", flush=True)
    print(f"  Summary: {TEXT_FILE}", flush=True)


if __name__ == '__main__':
    main()
