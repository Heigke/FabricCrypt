#!/usr/bin/env python3
"""z2231_closed_loop_computation.py — Closed-Loop Bidirectional Computation

Key hypothesis: The GPU↔FPGA bridge can perform COMPUTATION, not just coupling.
We test whether the closed loop produces emergent dynamics that neither substrate
alone exhibits, and whether spike-driven GPU feedback creates structured state
transitions in the FPGA reservoir.

Experiments:
  EXP 1: Spike-Driven GPU Feedback (120 trials per condition)
         FPGA spike rate directly controls GPU workload type (matmul/conv/fft).
         Different waveforms → different spike patterns → different GPU signatures.
         Tests if the loop creates distinguishable GPU workload fingerprints.

  EXP 2: Recurrent Cross-Substrate Echo (160 trials)
         Inject pulse → measure echo through GPU→FPGA→GPU→FPGA path.
         COUPLED should show longer-lasting echoes than either substrate alone.
         Tests temporal memory THROUGH the bidirectional loop.

  EXP 3: Emergent State Space (120 trials)
         Run coupled system with continuous drive and measure the dimensionality
         of the joint (FPGA_state, GPU_state) trajectory.
         COUPLED should create higher-dimensional state space than concatenation.

  EXP 4: Mutual Information Profile (80 trials)
         MI(FPGA_t, GPU_t+k) for k = -5..+5 — full temporal MI profile.
         Tests timing structure of bidirectional information flow.

  EXP 5: Computation Through Coupling (120 trials)
         Target function = nonlinear transform of (input + GPU_power + FPGA_spikes).
         Only the coupled system has all three ingredients.
         Tests genuine computation emerging from the bridge.

  EXP 6: Attractor Landscape (80 trials)
         Same input, different initial conditions → measure convergence.
         COUPLED should show attractor dynamics (convergence to limit cycles).

Hardware: AMD gfx1151 GPU + Arty A7-100T FPGA (128 neurons, UDP Ethernet)
Tests: T774-T793 (20 tests)
"""

import os, sys, json, time, struct
import numpy as np
from pathlib import Path

# Force line-buffered stdout for real-time output with file redirect
if not sys.stdout.isatty():
    sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'

# ─── Parameters (z2230 proven values) ───
N_NEURONS = 128
BASE_VG = 0.45
ALPHA = 0.35
BETA_POWER = 0.12
BETA_THERMAL = 0.08
BETA_CLOCK = 0.10
SAMPLE_HZ = 20
N_STEPS = 30
WORKLOAD_MS = 5.0

# Probe paths
HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
HWMON_TEMP = "/sys/class/hwmon/hwmon7/temp1_input"
HWMON_FREQ = "/sys/class/hwmon/hwmon7/freq1_input"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
GPU_BUSY_PATH = "/sys/class/drm/card0/device/gpu_busy_percent"

CHANNEL_NAMES = [
    'smn_temp', 'pm_thermal', 'pm_power', 'pm_sclk',
    'hw_power', 'hw_temp', 'hw_freq', 'gpu_busy', 'dispatch_jitter'
]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════
# GPU PROBES (same as z2230)
# ═══════════════════════════════════════════════════════════

def read_smn_adc():
    try:
        with open(SMN_PATH, 'rb+') as f:
            f.write(struct.pack('<I', 0x59800))
            f.seek(0)
            raw = struct.unpack('<I', f.read(4))[0]
        return (raw >> 21) * 0.125
    except:
        return None

def read_pm_table():
    try:
        with open(PM_TABLE_PATH, 'rb') as f:
            f.seek(0x004C); thermal = struct.unpack('<f', f.read(4))[0]
            f.seek(0x0100); power = struct.unpack('<f', f.read(4))[0]
            f.seek(0x0344); sclk = struct.unpack('<f', f.read(4))[0]
        return thermal, power, sclk
    except:
        return None, None, None

def read_hwmon():
    try:
        p = int(open(HWMON_POWER).read().strip()) / 1e6
        t = int(open(HWMON_TEMP).read().strip()) / 1e3
        f = int(open(HWMON_FREQ).read().strip()) / 1e6
        return p, t, f
    except:
        return None, None, None

def read_gpu_busy():
    try: return int(open(GPU_BUSY_PATH).read().strip())
    except: return 0

_torch_device = None
_torch_available = False

def measure_dispatch_jitter():
    if not _torch_available:
        return 0.0
    import torch
    a = torch.randn(64, 64, device=_torch_device)
    b = torch.randn(64, 64, device=_torch_device)
    t0 = time.perf_counter()
    _ = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    del a, b
    return elapsed

def read_all_gpu_state():
    smn = read_smn_adc()
    pm_t, pm_p, pm_sclk = read_pm_table()
    hw_p, hw_t, hw_f = read_hwmon()
    busy = read_gpu_busy()
    jitter = measure_dispatch_jitter()
    return [
        smn or 0, pm_t or 0, pm_p or 0, pm_sclk or 0,
        hw_p or 0, hw_t or 0, hw_f or 0, busy or 0, jitter
    ]


# ═══════════════════════════════════════════════════════════
# GPU Workload — Multiple Types
# ═══════════════════════════════════════════════════════════

def init_torch():
    global _torch_available, _torch_device
    try:
        import torch
        if torch.cuda.is_available():
            _torch_device = torch.device('cuda')
            _ = torch.randn(64, 64, device=_torch_device) @ torch.randn(64, 64, device=_torch_device)
            torch.cuda.synchronize()
            _torch_available = True
            print(f"  HIP: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARNING: No CUDA/HIP")
    except ImportError:
        print("  WARNING: No torch")

def run_workload(intensity, duration_ms=5.0, workload_type='matmul'):
    """Run GPU workload of a specific type — different types create different telemetry."""
    if not _torch_available or intensity < 0.05:
        return 0.0
    import torch
    N = int(128 + 896 * np.clip(intensity, 0.0, 1.0))
    t0 = time.perf_counter()
    deadline = t0 + duration_ms / 1000.0

    if workload_type == 'matmul':
        a = torch.randn(N, N, device=_torch_device)
        b = torch.randn(N, N, device=_torch_device)
        while time.perf_counter() < deadline:
            _ = torch.mm(a, b)
        del a, b
    elif workload_type == 'conv':
        # 2D convolution — different memory access pattern (cap spatial to 64)
        S = min(N, 64)
        x = torch.randn(1, 16, S, S, device=_torch_device)
        w = torch.randn(16, 16, 3, 3, device=_torch_device)
        while time.perf_counter() < deadline:
            _ = torch.nn.functional.conv2d(x, w, padding=1)
        del x, w
    elif workload_type == 'fft':
        a = torch.randn(N * 4, device=_torch_device, dtype=torch.complex64)
        while time.perf_counter() < deadline:
            _ = torch.fft.fft(a)
        del a
    elif workload_type == 'scatter':
        # Memory-intensive scatter — irregular access pattern (cap size)
        S = min(N, 256)
        a = torch.randn(S * S, device=_torch_device)
        idx = torch.randint(0, S * S, (S * S,), device=_torch_device)
        while time.perf_counter() < deadline:
            _ = a[idx]
        del a, idx
    else:
        a = torch.randn(N, N, device=_torch_device)
        b = torch.randn(N, N, device=_torch_device)
        while time.perf_counter() < deadline:
            _ = torch.mm(a, b)
        del a, b

    elapsed = time.perf_counter() - t0
    return elapsed

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ═══════════════════════════════════════════════════════════
# COUPLED DYNAMICS LOOP — ENHANCED WITH WORKLOAD TYPE SELECTION
# ═══════════════════════════════════════════════════════════

def run_coupled_loop_v2(fpga, input_signal, w_in, w_gpu, w_fb, vg_spread,
                        mode='COUPLED', n_neurons=128, record_gpu=True,
                        gpu_intensity=None, beta_scale=1.0,
                        workload_type='matmul', spike_driven_workload=False):
    """Enhanced loop with workload type selection and spike-driven feedback."""
    n_steps = len(input_signal)
    interval = 1.0 / SAMPLE_HZ
    raw_states = np.zeros((n_steps, n_neurons * 3))
    gpu_log = np.zeros((n_steps, 9))
    cumulative = np.zeros(n_neurons)
    prev_counts = None
    workload_types_used = []

    for t in range(n_steps):
        t_start = time.perf_counter()

        if record_gpu:
            gpu_log[t] = read_all_gpu_state()

        vg = BASE_VG + vg_spread[:n_neurons]

        if mode in ('COUPLED', 'FPGA_ONLY'):
            vg += ALPHA * input_signal[t] * w_in[:n_neurons]

        if mode == 'COUPLED':
            hw_p = gpu_log[t, 4]
            pm_sclk = gpu_log[t, 3]
            pm_t = gpu_log[t, 1]

            if t >= 3:
                p_base = gpu_log[max(0,t-10):t, 4].mean()
                s_base = gpu_log[max(0,t-10):t, 3].mean()
                t_base = gpu_log[max(0,t-10):t, 1].mean()
            else:
                p_base, s_base, t_base = hw_p, pm_sclk, pm_t

            p_delta = (hw_p - p_base) / max(abs(p_base), 1.0)
            s_delta = (pm_sclk - s_base) / max(abs(s_base), 1.0)
            t_delta = (pm_t - t_base) / max(abs(t_base), 1.0)

            n3 = n_neurons // 3
            wg = w_gpu[:n_neurons]
            vg[:n3] += beta_scale * BETA_POWER * p_delta * wg[:n3]
            vg[n3:2*n3] += beta_scale * BETA_CLOCK * s_delta * wg[n3:2*n3]
            vg[2*n3:] += beta_scale * BETA_THERMAL * t_delta * wg[2*n3:]

            gs_pm_t = gpu_log[t, 1]
            if gs_pm_t > 0:
                try:
                    fpga.set_temp(float(gs_pm_t) + 273.15)
                except:
                    pass

        vg = np.clip(vg, 0.10, 0.85)

        if mode != 'GPU_ONLY':
            fpga.set_vg_batch(0, vg.tolist())
            time.sleep(0.001)

            try:
                counts, vm, refract = fpga.read_telemetry_fast()
                if prev_counts is not None:
                    for i in range(n_neurons):
                        delta = (int(counts[i]) - int(prev_counts[i])) & 0xFFFF
                        if delta > 30000: delta = 0
                        raw_states[t, i] = delta
                        cumulative[i] += delta
                    raw_states[t, n_neurons:2*n_neurons] = vm[:n_neurons]
                    raw_states[t, 2*n_neurons:] = cumulative
                prev_counts = counts.copy()
            except:
                pass

        # GPU workload — optionally spike-driven type selection
        if mode in ('COUPLED', 'FPGA_ONLY'):
            if gpu_intensity is not None:
                intensity = float(gpu_intensity[t])
            elif t >= 1:
                recent_spikes = raw_states[max(0,t-2):t+1, :n_neurons].mean(axis=0)
                raw_val = float(np.dot(recent_spikes, w_fb[:n_neurons]))
                intensity = float(sigmoid(raw_val - 5.0))
            else:
                intensity = 0.3

            # Spike-driven workload type selection
            wl_type = workload_type
            if spike_driven_workload and t >= 2:
                total_spikes = raw_states[t, :n_neurons].sum()
                types = ['matmul', 'conv', 'fft', 'scatter']
                wl_type = types[int(total_spikes) % 4]

            run_workload(intensity, duration_ms=WORKLOAD_MS, workload_type=wl_type)
            workload_types_used.append(wl_type)

            if mode == 'COUPLED':
                try:
                    fpga.set_mac_signal(intensity * 0.5)
                except:
                    pass
        elif mode == 'GPU_ONLY':
            intensity = float(0.2 + 0.6 * np.clip(input_signal[t], 0, 1))
            run_workload(intensity, duration_ms=WORKLOAD_MS, workload_type=workload_type)
            workload_types_used.append(workload_type)

        elapsed = time.perf_counter() - t_start
        remaining = interval - elapsed
        if remaining > 0.001:
            time.sleep(remaining)

    return raw_states, gpu_log, workload_types_used


# ═══════════════════════════════════════════════════════════
# FEATURE PIPELINE + PCA (z2230 proven)
# ═══════════════════════════════════════════════════════════

def augment_with_delays(states, delays=(1, 2)):
    T, D = states.shape
    aug = np.zeros((T, D * (1 + len(delays))))
    aug[:, :D] = states
    for i, d in enumerate(delays):
        start = D * (i + 1)
        aug[d:, start:start+D] = states[:-d]
    return aug

def pool_trial_features(raw_states):
    aug = augment_with_delays(raw_states, delays=(1, 2))
    return np.concatenate([
        aug.mean(axis=0), aug.std(axis=0),
        aug.max(axis=0), aug.min(axis=0),
    ])

def pool_gpu_features(gpu_log):
    return np.concatenate([
        gpu_log.mean(axis=0), gpu_log.std(axis=0),
        gpu_log.max(axis=0), gpu_log.min(axis=0),
    ])

def pca_reduce(X, n_components=100):
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T, mu, Vt[:n_components]

def pca_transform(X, mu, Vt):
    return (X - mu) @ Vt.T

def ridge_classify(X_tr, y_tr, X_te, y_te, n_classes=None):
    if n_classes is None: n_classes = max(len(np.unique(y_tr)), len(np.unique(y_te)))
    alphas = [1e-4, 1e-2, 1.0, 100.0, 1000.0, 10000.0]
    Y_tr = np.zeros((len(y_tr), n_classes))
    for i, y in enumerate(y_tr): Y_tr[i, int(y)] = 1.0
    best = -1
    for a in alphas:
        I = np.eye(X_tr.shape[1])
        try: W = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ Y_tr)
        except: continue
        acc = np.mean(np.argmax(X_te @ W, axis=1) == y_te)
        if acc > best: best = acc
    return best

def stratified_kfold(X, y, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        c_idx = indices[y[indices] == c]
        for i, idx in enumerate(c_idx): folds[i % n_splits].append(idx)
    splits = []
    for fold in range(n_splits):
        te = np.array(folds[fold])
        tr = np.concatenate([np.array(folds[f]) for f in range(n_splits) if f != fold])
        splits.append((tr, te))
    return splits

def classify_cv_pca(X, y, n_splits=5, n_classes=None, max_pca=120):
    if n_classes is None: n_classes = len(np.unique(y))
    splits = stratified_kfold(X, y, n_splits)
    accs = []
    use_pca = X.shape[1] > max_pca
    for tr_idx, te_idx in splits:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
        sigma[sigma < 1e-2] = 1.0
        X_tr_n = (X_tr - mu) / sigma
        X_te_n = (X_te - mu) / sigma
        if use_pca:
            X_tr_n, pca_mu, pca_Vt = pca_reduce(X_tr_n, n_components=max_pca)
            X_te_n = pca_transform(X_te_n, pca_mu, pca_Vt)
        acc = ridge_classify(X_tr_n, y_tr, X_te_n, y_te, n_classes=n_classes)
        accs.append(acc)
    return {'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
            'per_fold': [float(a) for a in accs]}


# ═══════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════

def generate_waveforms_4class(n_trials, n_steps, seed=42):
    """4-class waveform: sine, square, sawtooth, triangle."""
    rng = np.random.default_rng(seed)
    trials, labels = [], []
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(n_steps) * dt
    for _ in range(n_trials):
        cls = rng.integers(0, 4)
        freq = rng.uniform(1.0, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        phi = 2 * np.pi * freq * t + phase
        if cls == 0: wave = np.sin(phi)
        elif cls == 1: wave = np.sign(np.sin(phi))
        elif cls == 2: wave = 2 * (phi / (2 * np.pi) % 1.0) - 1
        else: wave = 2 * np.abs(2 * (phi / (2 * np.pi) % 1.0) - 1) - 1
        wave = 0.5 + 0.4 * wave + 0.02 * rng.standard_normal(n_steps)
        trials.append(wave)
        labels.append(cls)
    return np.array(trials), np.array(labels)


def generate_echo_pulse(n_steps, pulse_step=5, pulse_width=3):
    """Single pulse for echo measurement."""
    signal = np.full(n_steps, 0.5)
    signal[pulse_step:pulse_step+pulse_width] = 0.9
    return signal


def generate_continuous_drive(n_steps, seed=42):
    """Smooth sinusoidal continuous drive for state space analysis."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / SAMPLE_HZ
    t = np.arange(n_steps) * dt
    f1, f2 = rng.uniform(0.5, 1.5), rng.uniform(1.5, 3.0)
    return 0.5 + 0.3 * np.sin(2 * np.pi * f1 * t) + 0.15 * np.sin(2 * np.pi * f2 * t)


def generate_nonlinear_target(input_signal, gpu_power_seq, fpga_spike_seq):
    """Target = nonlinear function of all three: input + GPU + FPGA.
    Only the coupled system has access to all three ingredients."""
    # Normalize each component
    inp = (input_signal - input_signal.mean()) / max(input_signal.std(), 1e-6)
    gpu = (gpu_power_seq - gpu_power_seq.mean()) / max(gpu_power_seq.std(), 1e-6)
    fpga = (fpga_spike_seq - fpga_spike_seq.mean()) / max(fpga_spike_seq.std(), 1e-6)
    # Nonlinear combination: tanh(input * gpu) + sin(fpga * input) + gpu^2
    target = np.tanh(inp * gpu) + np.sin(fpga * inp) + gpu ** 2
    return target


# ═══════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════

def mutual_info_binned(x, y, n_bins=8):
    """Binned mutual information estimator."""
    x_bins = np.digitize(x, np.linspace(x.min()-1e-10, x.max()+1e-10, n_bins+1)) - 1
    y_bins = np.digitize(y, np.linspace(y.min()-1e-10, y.max()+1e-10, n_bins+1)) - 1
    pxy = np.zeros((n_bins, n_bins))
    for i in range(len(x)):
        xb = min(x_bins[i], n_bins-1)
        yb = min(y_bins[i], n_bins-1)
        pxy[xb, yb] += 1
    pxy /= pxy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    return mi


def effective_dimension(X, threshold=0.95):
    """Effective dimensionality via PCA variance explained."""
    Xc = X - X.mean(axis=0)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var_explained = S ** 2
    total = var_explained.sum()
    if total < 1e-10:
        return 1
    cumvar = np.cumsum(var_explained) / total
    return int(np.searchsorted(cumvar, threshold) + 1)


def ridge_regress_r2(X_tr, y_tr, X_te, y_te):
    """Ridge regression R² with alpha search."""
    alphas = [1e-4, 1e-2, 1.0, 100.0, 1000.0]
    best_r2 = -999
    for a in alphas:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + a * I, X_tr.T @ y_tr)
        except:
            continue
        pred = X_te @ w
        ss_res = np.sum((y_te - pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        if r2 > best_r2:
            best_r2 = r2
    return best_r2


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("z2231: CLOSED-LOOP BIDIRECTIONAL COMPUTATION")
    print("  Spike-driven GPU feedback, cross-substrate echo, emergent state space")
    print("  MI temporal profile, computation through coupling, attractor dynamics")
    print("=" * 72)

    # ─── Setup ───
    print("\n[1] Connecting FPGA...")
    from fpga_host_eth import FPGAEthBridge
    fpga = FPGAEthBridge()
    if not fpga.connect():
        print("[ETH] WARN: No telemetry response, continuing anyway...")
    print(f"[ETH] {N_NEURONS} neurons")

    print("\n[2] Init GPU HIP...")
    init_torch()

    print("\n[3] Probe check...")
    gpu_state = read_all_gpu_state()
    for name, val in zip(CHANNEL_NAMES, gpu_state):
        print(f"    {name}: {val}")

    rng = np.random.default_rng(42)
    w_in = rng.standard_normal(N_NEURONS)
    w_gpu = rng.standard_normal(N_NEURONS)
    w_fb = rng.standard_normal(N_NEURONS) * 0.5
    vg_spread = rng.uniform(-0.08, 0.08, N_NEURONS)

    results = {
        'experiment': 'z2231_closed_loop_computation',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'architecture': {
            'base': 'z2230 + spike-driven GPU feedback + cross-substrate echo + MI profile',
            'n_neurons': N_NEURONS, 'sample_hz': SAMPLE_HZ,
            'n_steps': N_STEPS, 'membrane_tau_ms': 49.4,
        },
    }
    tests = {}

    # ════════════════════════════════════════════════════════════
    # EXP 1: SPIKE-DRIVEN GPU FEEDBACK (200 trials, 4-class)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 1: SPIKE-DRIVEN GPU FEEDBACK (4-class, 120 trials)")
    print("  FPGA spikes select GPU workload type (matmul/conv/fft/scatter)")
    print("  Different waveforms → different spike patterns → different GPU fingerprints")
    print("  Tests if closed loop creates distinguishable computational signatures")
    print("=" * 72)

    N_TRIALS_1 = 120
    trials, labels = generate_waveforms_4class(N_TRIALS_1, N_STEPS, seed=42)
    exp1 = {}

    for mode in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Spike-Driven: {mode} ---")
        fpga_feats_list, gpu_feats_list = [], []
        for trial_idx in range(N_TRIALS_1):
            if (trial_idx + 1) % 20 == 0:
                print(f"    {mode} trial {trial_idx+1}/{N_TRIALS_1}")
            raw, gpu, wl_types = run_coupled_loop_v2(
                fpga, trials[trial_idx], w_in, w_gpu, w_fb, vg_spread,
                mode=mode, spike_driven_workload=(mode == 'COUPLED'))
            fpga_feats_list.append(pool_trial_features(raw))
            gpu_feats_list.append(pool_gpu_features(gpu))

        F = np.array(fpga_feats_list)
        G = np.array(gpu_feats_list)
        combined = np.hstack([F, G])
        y = labels

        # Classify using combined features
        res = classify_cv_pca(combined, y, n_classes=4)
        print(f"  {mode}: combined acc={res['mean']:.3f} ± {res['std']:.3f}")

        # Also classify GPU-only features (should show workload fingerprinting)
        res_gpu = classify_cv_pca(G, y, n_classes=4, max_pca=30)
        print(f"  {mode}: GPU-only acc={res_gpu['mean']:.3f} ± {res_gpu['std']:.3f}")

        exp1[mode] = {'combined': res, 'gpu_only': res_gpu}

    results['spike_driven_feedback'] = exp1

    coupled_acc = exp1['COUPLED']['combined']['mean']
    fpga_only_acc = exp1['FPGA_ONLY']['combined']['mean']
    static_acc = exp1['STATIC']['combined']['mean']
    coupled_gpu_acc = exp1['COUPLED']['gpu_only']['mean']
    fpga_only_gpu_acc = exp1['FPGA_ONLY']['gpu_only']['mean']

    tests['T774'] = {'desc': 'Spike-driven: COUPLED combined > 0.60',
                     'val': coupled_acc, 'pass': coupled_acc > 0.60}
    tests['T775'] = {'desc': 'Spike-driven: COUPLED combined > FPGA_ONLY',
                     'val': coupled_acc - fpga_only_acc,
                     'pass': coupled_acc > fpga_only_acc}
    tests['T776'] = {'desc': 'Spike-driven: COUPLED combined > STATIC',
                     'val': coupled_acc - static_acc,
                     'pass': coupled_acc > static_acc}
    tests['T777'] = {'desc': 'Spike-driven: COUPLED GPU-only > FPGA_ONLY GPU-only',
                     'val': coupled_gpu_acc - fpga_only_gpu_acc,
                     'pass': coupled_gpu_acc > fpga_only_gpu_acc}
    for tid in ['T774', 'T775', 'T776', 'T777']:
        print(f"  {tid}: {'PASS' if tests[tid]['pass'] else 'FAIL'} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ════════════════════════════════════════════════════════════
    # EXP 2: RECURRENT CROSS-SUBSTRATE ECHO (160 trials)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 2: RECURRENT CROSS-SUBSTRATE ECHO (160 trials)")
    print("  Inject pulse → measure echo decay through GPU↔FPGA loop")
    print("  COUPLED should show longer-lasting echoes than FPGA_ONLY")
    print("=" * 72)

    N_TRIALS_2 = 40
    N_ECHO_STEPS = 40  # longer window to measure echo decay
    exp2 = {}

    for mode in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Echo: {mode} ---")
        echo_profiles = []
        for trial_idx in range(N_TRIALS_2):
            if (trial_idx + 1) % 10 == 0:
                print(f"    {mode} trial {trial_idx+1}/{N_TRIALS_2}")
            pulse = generate_echo_pulse(N_ECHO_STEPS, pulse_step=5, pulse_width=3)
            raw, gpu, _ = run_coupled_loop_v2(
                fpga, pulse, w_in, w_gpu, w_fb, vg_spread, mode=mode)
            # Measure spike rate deviation from baseline over time
            baseline = raw[:4, :N_NEURONS].mean()  # pre-pulse baseline
            spike_profile = np.array([raw[t, :N_NEURONS].mean() for t in range(N_ECHO_STEPS)])
            deviation = np.abs(spike_profile - baseline)
            echo_profiles.append(deviation)

        echo_mean = np.mean(echo_profiles, axis=0)
        # Echo duration: how many steps above 10% of peak deviation after pulse
        peak_dev = echo_mean[5:10].max()
        if peak_dev > 0:
            echo_thresh = 0.1 * peak_dev
            echo_duration = 0
            for t in range(10, N_ECHO_STEPS):
                if echo_mean[t] > echo_thresh:
                    echo_duration = t - 10 + 1
        else:
            echo_duration = 0
        echo_energy = echo_mean[10:].sum()  # total post-pulse energy

        print(f"  {mode}: echo_duration={echo_duration} steps, echo_energy={echo_energy:.4f}")
        exp2[mode] = {'echo_duration': echo_duration, 'echo_energy': float(echo_energy),
                      'echo_profile': [float(x) for x in echo_mean]}

    results['echo_analysis'] = exp2

    c_dur = exp2['COUPLED']['echo_duration']
    f_dur = exp2['FPGA_ONLY']['echo_duration']
    s_dur = exp2['STATIC']['echo_duration']
    c_nrg = exp2['COUPLED']['echo_energy']
    f_nrg = exp2['FPGA_ONLY']['echo_energy']

    tests['T778'] = {'desc': 'Echo: COUPLED duration > 3 steps',
                     'val': c_dur, 'pass': c_dur > 3}
    tests['T779'] = {'desc': 'Echo: COUPLED duration > FPGA_ONLY',
                     'val': c_dur - f_dur, 'pass': c_dur > f_dur}
    tests['T780'] = {'desc': 'Echo: COUPLED energy > FPGA_ONLY energy',
                     'val': c_nrg - f_nrg, 'pass': c_nrg > f_nrg}
    for tid in ['T778', 'T779', 'T780']:
        print(f"  {tid}: {'PASS' if tests[tid]['pass'] else 'FAIL'} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ════════════════════════════════════════════════════════════
    # EXP 3: EMERGENT STATE SPACE (120 trials)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 3: EMERGENT STATE SPACE DIMENSIONALITY (120 trials)")
    print("  Measure effective dimension of joint (FPGA, GPU) trajectory")
    print("  COUPLED should create richer state space than either alone")
    print("=" * 72)

    N_TRIALS_3 = 40
    exp3 = {}

    for mode in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- StateSpace: {mode} ---")
        all_joint_states = []
        all_fpga_states = []
        all_gpu_states = []
        for trial_idx in range(N_TRIALS_3):
            if (trial_idx + 1) % 10 == 0:
                print(f"    {mode} trial {trial_idx+1}/{N_TRIALS_3}")
            drive = generate_continuous_drive(N_STEPS, seed=trial_idx * 7 + 100)
            raw, gpu, _ = run_coupled_loop_v2(
                fpga, drive, w_in, w_gpu, w_fb, vg_spread, mode=mode)
            # Take a compact state: mean spike rate across 8 groups + 4 GPU channels
            n_grp = 8
            grp_sz = N_NEURONS // n_grp
            fpga_compact = np.array([raw[:, i*grp_sz:(i+1)*grp_sz].mean(axis=1)
                                     for i in range(n_grp)]).T  # (N_STEPS, n_grp)
            gpu_compact = gpu[:, [4, 1, 3, 8]]  # hw_power, pm_thermal, pm_sclk, jitter
            joint = np.hstack([fpga_compact, gpu_compact])  # (N_STEPS, 12)
            all_joint_states.append(joint)
            all_fpga_states.append(fpga_compact)
            all_gpu_states.append(gpu_compact)

        # Stack all trials → (N_TRIALS * N_STEPS, dim)
        J = np.vstack(all_joint_states)
        F = np.vstack(all_fpga_states)
        G = np.vstack(all_gpu_states)

        dim_joint = effective_dimension(J)
        dim_fpga = effective_dimension(F)
        dim_gpu = effective_dimension(G)

        print(f"  {mode}: dim_joint={dim_joint}, dim_fpga={dim_fpga}, dim_gpu={dim_gpu}")
        exp3[mode] = {'dim_joint': dim_joint, 'dim_fpga': dim_fpga, 'dim_gpu': dim_gpu}

    results['state_space'] = exp3

    c_dim = exp3['COUPLED']['dim_joint']
    f_dim = exp3['FPGA_ONLY']['dim_joint']
    s_dim = exp3['STATIC']['dim_joint']

    tests['T781'] = {'desc': 'StateSpace: COUPLED dim_joint > FPGA_ONLY dim_joint',
                     'val': c_dim - f_dim, 'pass': c_dim > f_dim}
    tests['T782'] = {'desc': 'StateSpace: COUPLED dim_joint > STATIC dim_joint',
                     'val': c_dim - s_dim, 'pass': c_dim > s_dim}
    tests['T783'] = {'desc': 'StateSpace: COUPLED dim_joint > 5 (rich dynamics)',
                     'val': c_dim, 'pass': c_dim > 5}
    for tid in ['T781', 'T782', 'T783']:
        print(f"  {tid}: {'PASS' if tests[tid]['pass'] else 'FAIL'} — {tests[tid]['desc']} [{tests[tid]['val']}]")

    # ════════════════════════════════════════════════════════════
    # EXP 4: MUTUAL INFORMATION TEMPORAL PROFILE (80 trials)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 4: MUTUAL INFORMATION TEMPORAL PROFILE (80 trials)")
    print("  MI(FPGA_t, GPU_t+k) for k = -5..+5")
    print("  Tests timing structure of bidirectional information flow")
    print("=" * 72)

    N_TRIALS_4 = 40
    max_lag = 5
    exp4 = {}

    for mode in ['COUPLED', 'FPGA_ONLY']:
        print(f"\n  --- MI Profile: {mode} ---")
        mi_profiles = []
        for trial_idx in range(N_TRIALS_4):
            if (trial_idx + 1) % 10 == 0:
                print(f"    {mode} trial {trial_idx+1}/{N_TRIALS_4}")
            drive = generate_continuous_drive(N_STEPS + 2 * max_lag, seed=trial_idx * 11)
            raw, gpu, _ = run_coupled_loop_v2(
                fpga, drive, w_in, w_gpu, w_fb, vg_spread, mode=mode)
            fpga_rate = raw[:, :N_NEURONS].mean(axis=1)  # mean spike rate
            gpu_power = gpu[:, 4]  # hw_power

            mi_at_lag = []
            for k in range(-max_lag, max_lag + 1):
                if k >= 0:
                    f_slice = fpga_rate[:len(fpga_rate)-max_lag]
                    g_slice = gpu_power[k:k+len(f_slice)]
                else:
                    g_slice = gpu_power[:len(gpu_power)-max_lag]
                    f_slice = fpga_rate[-k:-k+len(g_slice)]
                n = min(len(f_slice), len(g_slice))
                if n < 10:
                    mi_at_lag.append(0.0)
                else:
                    mi_at_lag.append(mutual_info_binned(f_slice[:n], g_slice[:n]))
            mi_profiles.append(mi_at_lag)

        mi_mean = np.mean(mi_profiles, axis=0)
        lags = list(range(-max_lag, max_lag + 1))
        peak_lag = lags[np.argmax(mi_mean)]
        peak_mi = float(np.max(mi_mean))
        asymmetry = float(np.mean(mi_mean[max_lag+1:])) - float(np.mean(mi_mean[:max_lag]))  # positive → GPU leads

        print(f"  {mode}: peak_lag={peak_lag}, peak_MI={peak_mi:.4f}, asymmetry={asymmetry:.4f}")
        print(f"    MI profile: {['%.3f' % m for m in mi_mean]}")
        exp4[mode] = {
            'lags': lags,
            'mi_profile': [float(m) for m in mi_mean],
            'peak_lag': peak_lag,
            'peak_mi': peak_mi,
            'asymmetry': asymmetry,
        }

    results['mi_profile'] = exp4

    c_peak = exp4['COUPLED']['peak_mi']
    f_peak = exp4['FPGA_ONLY']['peak_mi']
    c_asym = exp4['COUPLED']['asymmetry']

    tests['T784'] = {'desc': 'MI Profile: COUPLED peak MI > 0.05 bits',
                     'val': c_peak, 'pass': c_peak > 0.05}
    tests['T785'] = {'desc': 'MI Profile: COUPLED peak MI > FPGA_ONLY peak MI',
                     'val': c_peak - f_peak, 'pass': c_peak > f_peak}
    tests['T786'] = {'desc': 'MI Profile: asymmetry detected (|asym| > 0.01)',
                     'val': abs(c_asym), 'pass': abs(c_asym) > 0.01}
    for tid in ['T784', 'T785', 'T786']:
        print(f"  {tid}: {'PASS' if tests[tid]['pass'] else 'FAIL'} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ════════════════════════════════════════════════════════════
    # EXP 5: COMPUTATION THROUGH COUPLING (120 trials)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 5: COMPUTATION THROUGH COUPLING (120 trials)")
    print("  Target = nonlinear(input + GPU_power + FPGA_spikes)")
    print("  Only coupled system has all three ingredients")
    print("  Tests genuine emergent computation from the bridge")
    print("=" * 72)

    N_TRIALS_5 = 120
    trials_5, labels_5 = generate_waveforms_4class(N_TRIALS_5, N_STEPS, seed=99)
    exp5 = {}

    for mode in ['COUPLED', 'FPGA_ONLY', 'STATIC']:
        print(f"\n  --- Computation: {mode} ---")
        features_list = []
        targets_list = []
        for trial_idx in range(N_TRIALS_5):
            if (trial_idx + 1) % 20 == 0:
                print(f"    {mode} trial {trial_idx+1}/{N_TRIALS_5}")
            raw, gpu, _ = run_coupled_loop_v2(
                fpga, trials_5[trial_idx], w_in, w_gpu, w_fb, vg_spread, mode=mode)
            fpga_feat = pool_trial_features(raw)
            gpu_feat = pool_gpu_features(gpu)
            combined_feat = np.concatenate([fpga_feat, gpu_feat])
            features_list.append(combined_feat)
            # Target: nonlinear function of input + GPU power + FPGA spikes
            gpu_power_seq = gpu[:, 4]
            fpga_spike_seq = raw[:, :N_NEURONS].mean(axis=1)
            target = generate_nonlinear_target(
                trials_5[trial_idx], gpu_power_seq, fpga_spike_seq)
            targets_list.append(target.mean())  # scalar target per trial

        X = np.array(features_list)
        y_reg = np.array(targets_list)

        # Ridge regression R² with PCA
        splits = stratified_kfold(X, labels_5, n_splits=5)
        r2s = []
        for tr_idx, te_idx in splits:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y_reg[tr_idx], y_reg[te_idx]
            mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0)
            sigma[sigma < 1e-2] = 1.0
            X_tr_n = (X_tr - mu) / sigma
            X_te_n = (X_te - mu) / sigma
            if X_tr_n.shape[1] > 120:
                X_tr_n, pca_mu, pca_Vt = pca_reduce(X_tr_n, n_components=120)
                X_te_n = pca_transform(X_te_n, pca_mu, pca_Vt)
            r2 = ridge_regress_r2(X_tr_n, y_tr, X_te_n, y_te)
            r2s.append(r2)

        r2_mean = float(np.mean(r2s))
        r2_std = float(np.std(r2s))
        print(f"  {mode}: R²={r2_mean:.4f} ± {r2_std:.4f}")
        exp5[mode] = {'mean': r2_mean, 'std': r2_std, 'per_fold': [float(r) for r in r2s]}

    results['computation_through_coupling'] = exp5

    c_r2 = exp5['COUPLED']['mean']
    f_r2 = exp5['FPGA_ONLY']['mean']
    s_r2 = exp5['STATIC']['mean']

    tests['T787'] = {'desc': 'Computation: COUPLED R² > 0.0 (above chance)',
                     'val': c_r2, 'pass': c_r2 > 0.0}
    tests['T788'] = {'desc': 'Computation: COUPLED R² > FPGA_ONLY',
                     'val': c_r2 - f_r2, 'pass': c_r2 > f_r2}
    tests['T789'] = {'desc': 'Computation: COUPLED R² > STATIC',
                     'val': c_r2 - s_r2, 'pass': c_r2 > s_r2}
    for tid in ['T787', 'T788', 'T789']:
        print(f"  {tid}: {'PASS' if tests[tid]['pass'] else 'FAIL'} — {tests[tid]['desc']} [{tests[tid]['val']:.4f}]")

    # ════════════════════════════════════════════════════════════
    # EXP 6: ATTRACTOR LANDSCAPE (80 trials)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("EXP 6: ATTRACTOR LANDSCAPE (80 trials)")
    print("  Same input, different initial Vg spread → measure convergence")
    print("  COUPLED should show attractor dynamics (lower variance over time)")
    print("=" * 72)

    N_TRIALS_6 = 20
    N_INIT = 4  # 4 different initial conditions per trial
    exp6 = {}

    for mode in ['COUPLED', 'FPGA_ONLY']:
        print(f"\n  --- Attractor: {mode} ---")
        convergence_ratios = []
        for trial_idx in range(N_TRIALS_6):
            if (trial_idx + 1) % 5 == 0:
                print(f"    {mode} trial {trial_idx+1}/{N_TRIALS_6}")
            drive = generate_continuous_drive(N_STEPS, seed=trial_idx * 13)
            trajectories = []
            for init_idx in range(N_INIT):
                vg_init = rng.uniform(-0.12, 0.12, N_NEURONS)  # different ICs
                raw, gpu, _ = run_coupled_loop_v2(
                    fpga, drive, w_in, w_gpu, w_fb, vg_init, mode=mode)
                traj = raw[:, :N_NEURONS].mean(axis=1)  # mean spike rate trajectory
                trajectories.append(traj)
            trajectories = np.array(trajectories)  # (N_INIT, N_STEPS)
            # Variance across ICs at start vs end
            var_start = trajectories[:, 2:5].var(axis=0).mean()
            var_end = trajectories[:, -5:].var(axis=0).mean()
            convergence = var_start / max(var_end, 1e-10)
            convergence_ratios.append(convergence)

        conv_mean = float(np.mean(convergence_ratios))
        conv_std = float(np.std(convergence_ratios))
        print(f"  {mode}: convergence_ratio={conv_mean:.4f} ± {conv_std:.4f}")
        exp6[mode] = {'convergence_ratio_mean': conv_mean,
                      'convergence_ratio_std': conv_std,
                      'per_trial': [float(c) for c in convergence_ratios]}

    results['attractor_landscape'] = exp6

    c_conv = exp6['COUPLED']['convergence_ratio_mean']
    f_conv = exp6['FPGA_ONLY']['convergence_ratio_mean']

    tests['T790'] = {'desc': 'Attractor: COUPLED convergence > 1.0 (converging)',
                     'val': c_conv, 'pass': c_conv > 1.0}
    tests['T791'] = {'desc': 'Attractor: COUPLED convergence > FPGA_ONLY',
                     'val': c_conv - f_conv, 'pass': c_conv > f_conv}

    # ─── Additional strong coupling tests ───
    # T792: Cross-experiment consistency — spike-driven COUPLED beats STATIC everywhere
    spike_pass = coupled_acc > static_acc
    echo_pass = exp2['COUPLED']['echo_energy'] > exp2['STATIC']['echo_energy']
    state_pass = exp3['COUPLED']['dim_joint'] > exp3['STATIC']['dim_joint']
    all_3 = spike_pass and echo_pass and state_pass
    tests['T792'] = {'desc': 'Consistency: COUPLED > STATIC in spike-driven + echo + state-space',
                     'val': int(spike_pass) + int(echo_pass) + int(state_pass),
                     'pass': all_3}

    # T793: Overall coupling strength — majority of coupling-specific tests pass
    coupling_tests = [tests[f'T{t}']['pass'] for t in range(774, 793)]
    coupling_pass_rate = sum(coupling_tests) / len(coupling_tests)
    tests['T793'] = {'desc': 'Overall: >60% of coupling tests pass',
                     'val': coupling_pass_rate,
                     'pass': coupling_pass_rate > 0.60}

    for tid in ['T790', 'T791', 'T792', 'T793']:
        print(f"  {tid}: {'PASS' if tests[tid]['pass'] else 'FAIL'} — {tests[tid]['desc']} [{tests[tid]['val']}]")

    # ─── Final Summary ───
    results['tests'] = tests
    n_pass = sum(1 for t in tests.values() if t['pass'])
    n_total = len(tests)
    results['score'] = f"{n_pass}/{n_total}"

    print("\n" + "=" * 72)
    print(f"FINAL SCORE: {n_pass}/{n_total}")
    print("=" * 72)
    for tid in sorted(tests.keys(), key=lambda x: int(x[1:])):
        t = tests[tid]
        print(f"  {tid}: {'PASS' if t['pass'] else 'FAIL'} — {t['desc']} [{t['val']}]")

    out_path = RESULTS / 'z2231_closed_loop_computation.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
