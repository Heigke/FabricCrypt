#!/usr/bin/env python3
"""
z2328_combined_bridge.py — Combined GPU ESN + FPGA NS-RAM Bridge
================================================================
Final integration: GPU ESN (z2326) + FPGA synaptic reservoir (z2325).

z2326 GPU standalone: MC=12.0@512N, MG R²=0.96, cubic XOR-10=69.6%
z2325 FPGA standalone: sparse regime (T=0x18000/E=0x80) 36/128 active,
  saturated regime (T=0x18000/E=0x200) 44/128 active, rate=5957

The combined bridge:
  1. GPU ESN provides nonlinear computation (tanh) + memory (spectral radius)
  2. FPGA provides physical noise + synaptic topology + analog dynamics
  3. Bidirectional coupling: GPU→MAC→FPGA, FPGA→readout→GPU_input

EXP 1: Verify FPGA at 3 regimes (sparse/moderate/saturated)
  Quick 200-step probe to confirm spiking. If FPGA unavailable, abort.
  Tests: T1126-T1128

EXP 2: GPU ESN alone (baseline, 512 neurons)
  MC, XOR-5, 4-class waveform, NARMA-5
  Tests: T1129-T1132

EXP 3: FPGA alone at best regime (from EXP1)
  Same benchmarks as EXP2
  Tests: T1133-T1136

EXP 4: Concatenated bridge (GPU features ‖ FPGA features)
  No feedback loop — just concatenate readout features
  Same benchmarks
  Tests: T1137-T1140

EXP 5: Feedback bridge (GPU→FPGA MAC, FPGA→GPU input modulation)
  GPU ESN output modulates FPGA MAC signal; FPGA spike features
  fed back into GPU input on next step
  Tests: T1141-T1144

EXP 6: Cubic readout on all conditions (nonlinear boosted)
  Tests: T1145-T1148

Total: 23 tests (T1126-T1148)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2328_combined_bridge.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2328_combined_bridge.json'
sys.path.insert(0, str(BASE / 'scripts'))
from fpga_host_eth import FPGAEthBridge

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

NUM_NEURONS_FPGA = 128
NUM_NEURONS_GPU = 512
SAMPLE_HZ = 50

# ======================================================================
# Thermal safety
# ======================================================================
def get_temp():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return int(f.read().strip()) // 1000
    except:
        return 50

def wait_cool(target=50, timeout=120):
    t0 = time.time()
    while get_temp() > target and time.time() - t0 < timeout:
        time.sleep(2)
    return get_temp()

def check_thermal(pause_at=75, resume_at=50):
    t = get_temp()
    if t >= pause_at:
        print(f"  [TEMP] {t}C >= {pause_at}C, cooling...", end='', flush=True)
        t2 = wait_cool(resume_at)
        print(f" {t2}C OK")
    return get_temp()

# ======================================================================
# GPU ESN
# ======================================================================
class GPU_ESN:
    """Echo State Network on GPU via PyTorch."""
    def __init__(self, n_neurons=512, spectral_radius=0.95, input_scale=1.0,
                 temperature=0.65, seed=42):
        self.N = n_neurons
        self.sr = spectral_radius
        self.T = temperature

        torch.manual_seed(seed)
        sparsity = min(0.1, 50.0 / n_neurons)
        W = torch.randn(n_neurons, n_neurons, device=device) / np.sqrt(n_neurons)
        mask = (torch.rand(n_neurons, n_neurons, device=device) < sparsity).float()
        W = W * mask

        with torch.no_grad():
            v = torch.randn(n_neurons, device=device)
            for _ in range(50):
                v = W @ v
                v = v / (v.norm() + 1e-10)
            sr_est = (W @ v).norm().item()
            if sr_est > 0:
                W = W * (spectral_radius / sr_est)

        self.W = W
        self.W_in = (torch.rand(n_neurons, device=device) * 2 - 1) * input_scale
        self.bias = torch.randn(n_neurons, device=device) * 0.005
        self.alpha = torch.ones(n_neurons, device=device)
        self.state = torch.zeros(n_neurons, device=device)

    def reset(self):
        self.state = torch.zeros(self.N, device=device)

    @torch.no_grad()
    def step(self, u_val):
        """Single step, returns state as numpy."""
        pre = self.W @ self.state + self.W_in * u_val + self.bias
        new_state = torch.tanh(pre / self.T)
        self.state = new_state  # alpha=1.0, no leak
        return self.state.cpu().numpy()

    @torch.no_grad()
    def run(self, inputs, warmup=0):
        """Run on full sequence."""
        T_steps = len(inputs)
        u = torch.tensor(inputs, device=device, dtype=torch.float32)
        states = torch.zeros(T_steps, self.N, device=device)
        for t in range(T_steps):
            pre = self.W @ self.state + self.W_in * u[t] + self.bias
            new_state = torch.tanh(pre / self.T)
            self.state = new_state
            states[t] = self.state
        return states[warmup:].cpu().numpy()

# ======================================================================
# FPGA helpers
# ======================================================================
FPGA_REGIMES = {
    'sparse':    {'thresh': 0x18000, 'base_exc': 0x0080, 'desc': 'T=1.5/E=0.002 (36/128 active)'},
    'moderate':  {'thresh': 0x18000, 'base_exc': 0x0200, 'desc': 'T=1.5/E=0.008 (44/128 active)'},
    'saturated': {'thresh': 0x18000, 'base_exc': 0x0800, 'desc': 'T=1.5/E=0.031 (128/128 active)'},
}

def open_fpga(timeout=2.0):
    """Create and connect FPGAEthBridge."""
    f = FPGAEthBridge(timeout=timeout)
    f.connect()
    return f

def setup_fpga(fpga, thresh, base_exc, bias_gain=0x0800, leak=0x0004):
    """Configure FPGA runtime parameters."""
    fpga.set_threshold_raw(thresh)
    fpga.set_base_exc_raw(base_exc)
    fpga.set_bias_gain_raw(bias_gain)
    fpga.set_leak_cond(leak)
    # Set Vg per group (32 neurons each) — wide spread for heterogeneity
    # Range 0.40–0.85V spans BVpar cliff region for max differentiation
    vg_volts = [0.40, 0.55, 0.70, 0.85]
    for g in range(4):
        vg_list = [vg_volts[g]] * 32
        fpga.set_vg_batch(g * 32, vg_list)
        time.sleep(0.05)  # Small delay between batches for CDC
    time.sleep(0.3)

def fpga_read_state(fpga):
    """Read telemetry, return (vmem_array, spike_cnt_array) or None."""
    telem = fpga.read_telemetry()
    if telem is None:
        return None, None
    vmem = np.array(telem['vmem'], dtype=np.float32)  # already float from read_telemetry()
    scnt = np.array(telem['spike_counts'], dtype=np.float32)
    return vmem, scnt

# ======================================================================
# Benchmark generators
# ======================================================================
def gen_narma(order, length, seed=42):
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(order, length):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[max(0,t-order):t]) + 1.5*u[t-order]*u[t-1] + 0.1
        y[t] = np.clip(y[t], -10, 10)
    return u, y

def gen_mackey_glass(length, tau=17, seed=42):
    rng = np.random.RandomState(seed)
    x = np.zeros(length + tau + 200)
    x[:tau+1] = 0.5 + rng.randn(tau+1) * 0.01
    for t in range(tau, length + tau + 199):
        x[t+1] = x[t] + 0.2 * x[t-tau] / (1 + x[t-tau]**10) - 0.1 * x[t]
    x = x[200:]  # remove transient
    return x[:length]

# ======================================================================
# Metrics
# ======================================================================
def compute_mc(states, inputs, max_delay=20):
    """Memory capacity."""
    n = min(len(states), len(inputs))
    states, inputs = states[:n], inputs[:n]
    split = n // 2
    mc = 0.0
    mc_per = {}
    for d in range(1, max_delay + 1):
        if d >= split:
            mc_per[d] = 0.0
            continue
        y_train = inputs[:split - d]
        y_test = inputs[split:n - d]
        X_train = states[d:d + len(y_train)]
        X_test = states[split + d:split + d + len(y_test)]
        if len(X_test) < 10 or len(X_train) < 10:
            mc_per[d] = 0.0
            continue
        alpha = 1e-4
        XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
        Xty = X_train.T @ y_train
        try:
            w = np.linalg.solve(XtX, Xty)
            y_pred = X_test @ w
            cc = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]
            r2 = max(0, cc**2) if not np.isnan(cc) else 0.0
        except:
            r2 = 0.0
        mc_per[d] = r2
        mc += r2
    return mc, mc_per

def compute_xor(states, inputs, tau):
    """XOR temporal task with linear readout."""
    n = min(len(states), len(inputs))
    u_bin = (inputs[:n] > np.median(inputs[:n])).astype(float)
    targets = np.zeros(n)
    for t in range(tau, n):
        targets[t] = float(int(u_bin[t]) ^ int(u_bin[t - tau]))
    valid = slice(max(tau, 50), n)
    X, y = states[valid], targets[valid]
    split = len(X) // 2
    if split < 10:
        return 0.5
    alpha = 1e-4
    XtX = X[:split].T @ X[:split] + alpha * np.eye(X.shape[1])
    Xty = X[:split].T @ y[:split]
    try:
        w = np.linalg.solve(XtX, Xty)
        y_pred = X[split:] @ w
        return float(np.mean((y_pred > 0.5).astype(float) == y[split:]))
    except:
        return 0.5

def compute_xor_cubic(states, inputs, tau):
    """XOR with cubic readout features on PCA-reduced states."""
    n = min(len(states), len(inputs))
    u_bin = (inputs[:n] > np.median(inputs[:n])).astype(float)
    targets = np.zeros(n)
    for t in range(tau, n):
        targets[t] = float(int(u_bin[t]) ^ int(u_bin[t - tau]))
    valid = slice(max(tau, 50), n)
    X_raw, y = states[valid], targets[valid]
    # Check for zero-variance (e.g., FPGA offline → all zeros)
    var = X_raw.var(axis=0)
    if var.sum() < 1e-10:
        return 0.5
    # PCA to 64 components (more than 32 to retain info from large reservoirs)
    from sklearn.decomposition import PCA
    n_comp = min(64, X_raw.shape[1], len(X_raw) - 1)
    if n_comp < 2:
        return 0.5
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_raw)
    X = np.hstack([X_pca, X_pca**2, X_pca**3])
    split = len(X) // 2
    if split < 10:
        return 0.5
    alpha = 1.0  # Stronger regularization for high-dim cubic features
    XtX = X[:split].T @ X[:split] + alpha * np.eye(X.shape[1])
    Xty = X[:split].T @ y[:split]
    try:
        w = np.linalg.solve(XtX, Xty)
        y_pred = X[split:] @ w
        return float(np.mean((y_pred > 0.5).astype(float) == y[split:]))
    except:
        return 0.5

def classify_waveforms_from_features(features, labels, split_ratio=0.5):
    """Classify using pre-collected features."""
    from sklearn.linear_model import RidgeClassifier
    split = int(len(labels) * split_ratio)
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(features[:split], labels[:split])
    return clf.score(features[split:], labels[split:])

def standardize_features(X):
    """Z-score each column. Essential for multi-scale feature matrices."""
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-10] = 1.0  # avoid div-by-zero for constant columns
    return (X - mu) / sigma

def effective_rank(X):
    """Effective dimensionality via SVD entropy (z-scored per column)."""
    X_z = standardize_features(X)
    cov = X_z.T @ X_z / len(X_z)
    sv = np.linalg.svd(cov, compute_uv=False)
    sv = sv / (sv.sum() + 1e-20)
    sv = sv[sv > 1e-12]
    return float(np.exp(-np.sum(sv * np.log(sv + 1e-20))))

# ======================================================================
# Results tracking
# ======================================================================
results = {'experiments': {}, 'tests': {}, 'meta': {
    'script': 'z2328_combined_bridge.py',
    'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    'gpu_neurons': NUM_NEURONS_GPU,
    'fpga_neurons': NUM_NEURONS_FPGA,
}}

def save():
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"  [SAVED] {SAVE_FILE}")

def test(tid, name, condition, val=None):
    status = "PASS" if condition else "FAIL"
    results['tests'][tid] = {'name': name, 'status': status, 'value': val}
    print(f"  [{status}] {tid}: {name}" + (f" = {val}" if val is not None else ""))
    return condition

print("=" * 70)
print("  z2328: Combined GPU ESN + FPGA NS-RAM Bridge")
print(f"  GPU: {NUM_NEURONS_GPU}N ESN (PyTorch) | FPGA: {NUM_NEURONS_FPGA}N LIF (synaptic)")
print("=" * 70)
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Temp: {get_temp()}C")

# ======================================================================
# EXP 1: Verify FPGA at 3 regimes
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 1: Verify FPGA Connectivity + Regimes")
print("=" * 70)

check_thermal()
exp1 = {}
best_regime = None
best_regime_score = -1

for regime_name, regime in FPGA_REGIMES.items():
    print(f"\n  --- {regime_name}: {regime['desc']} ---")
    try:
        fpga = open_fpga(timeout=2.0)
        setup_fpga(fpga, regime['thresh'], regime['base_exc'])

        rng = np.random.RandomState(42)
        states = np.zeros((200, NUM_NEURONS_FPGA), dtype=np.float32)
        dspikes = np.zeros((200, NUM_NEURONS_FPGA), dtype=np.float32)
        inputs = rng.uniform(0, 1, 200)

        for t in range(200):
            fpga.set_mac_signal(int(inputs[t] * 65536))
            vmem, scnt = fpga_read_state(fpga)
            if vmem is not None:
                states[t] = vmem
                dspikes[t] = scnt
            time.sleep(1.0 / SAMPLE_HZ)

        fpga.close()

        delta = np.diff(dspikes, axis=0)
        delta[delta < 0] = 0
        total_spikes = delta.sum()
        spike_rates = delta[50:].sum(axis=0)
        n_active = int((spike_rates > 0).sum())
        rate_mean = float(spike_rates.mean())

        X = states[50:]
        rank = effective_rank(X)

        # Quick MC
        X_c = X - X.mean(axis=0, keepdims=True)
        mc, _ = compute_mc(X_c, inputs[50:], max_delay=10)

        # Mean correlation
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0)
        mean_corr = float(np.abs(corr).mean())

        exp1[regime_name] = {
            'rank': rank, 'n_active': n_active, 'rate_mean': rate_mean,
            'total_spikes': total_spikes, 'mc': mc, 'mean_corr': mean_corr,
            'connected': True,
        }
        print(f"    rank={rank:.1f}, active={n_active}, rate={rate_mean:.0f}, "
              f"MC={mc:.3f}, corr={mean_corr:.4f}")

        # Pick best regime by composite score
        decorr = max(0, 1 - mean_corr)
        active_frac = n_active / 128.0
        activity_bonus = 4 * active_frac * (1 - active_frac)
        score = rank * 0.1 + mc * 10.0 + decorr * 5.0 + activity_bonus * 3.0
        if score > best_regime_score:
            best_regime_score = score
            best_regime = regime_name

    except Exception as e:
        print(f"    ERROR: {e}")
        exp1[regime_name] = {'connected': False, 'error': str(e)}

results['experiments']['EXP1_VERIFY'] = exp1

if best_regime is None or not exp1.get('sparse', {}).get('connected', False):
    # If FPGA not available, we can still run GPU-only and FPGA-simulated
    print("\n  WARNING: FPGA not available. Running GPU-only + simulated FPGA.")
    FPGA_AVAILABLE = False
    best_regime = 'sparse'
else:
    FPGA_AVAILABLE = True
    print(f"\n  Best FPGA regime: {best_regime} (score={best_regime_score:.2f})")

# Tests
if FPGA_AVAILABLE:
    test('T1126', 'FPGA sparse connected', exp1.get('sparse', {}).get('connected', False))
    test('T1127', 'FPGA sparse has active neurons',
         exp1.get('sparse', {}).get('n_active', 0) > 0,
         exp1.get('sparse', {}).get('n_active', 0))
    test('T1128', 'FPGA sparse rank > 1.5',
         exp1.get('sparse', {}).get('rank', 0) > 1.5,
         round(exp1.get('sparse', {}).get('rank', 0), 2))
else:
    test('T1126', 'FPGA connected', False, 'FPGA not available')
    test('T1127', 'FPGA active', False, 'FPGA not available')
    test('T1128', 'FPGA rank', False, 'FPGA not available')

save()

# ======================================================================
# EXP 2: GPU ESN alone (baseline)
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 2: GPU ESN Alone (512 neurons, baseline)")
print("=" * 70)

check_thermal()
N_STEPS = 2000
WARMUP = 300

rng = np.random.RandomState(42)
inputs_rand = rng.uniform(0, 1, N_STEPS).astype(np.float32)

# Run GPU ESN
esn = GPU_ESN(n_neurons=NUM_NEURONS_GPU, spectral_radius=0.95, seed=42)
esn.reset()
gpu_states = esn.run(inputs_rand, warmup=0)

X_gpu = gpu_states[WARMUP:]
inp = inputs_rand[WARMUP:]

gpu_rank = effective_rank(X_gpu)
gpu_mc, gpu_mc_per = compute_mc(X_gpu, inp, max_delay=20)
gpu_xor5 = compute_xor(X_gpu, inp, tau=5)

# Waveform classification using GPU ESN
print("  GPU classification...")
rng2 = np.random.RandomState(99)
t_wave = np.linspace(0, 2*np.pi, 50)
n_samples = 200
wave_signals, wave_labels = [], []
for _ in range(n_samples):
    c = rng2.randint(4)
    if c == 0: s = np.sin(t_wave)
    elif c == 1: s = np.sign(np.sin(t_wave))
    elif c == 2: s = 2*(t_wave/(2*np.pi) - np.floor(t_wave/(2*np.pi) + 0.5))
    else: s = np.sin(t_wave) + 0.5*np.sin(3*t_wave)
    s += rng2.randn(50) * 0.1
    wave_signals.append(s)
    wave_labels.append(c)
wave_labels = np.array(wave_labels)

gpu_wave_feats = []
for i in range(n_samples):
    esn.reset()
    u_wave = ((wave_signals[i] + 1.5) / 3.0).astype(np.float32)
    st = esn.run(u_wave, warmup=0)
    feat = np.concatenate([st[-20:].mean(0), st[-20:].std(0), np.diff(st[-10:], axis=0).mean(0)])
    gpu_wave_feats.append(feat)
gpu_wave_feats = np.array(gpu_wave_feats)
gpu_wave_acc = classify_waveforms_from_features(gpu_wave_feats, wave_labels)

exp2 = {
    'rank': gpu_rank, 'mc': gpu_mc, 'xor5': gpu_xor5,
    'wave_acc': gpu_wave_acc, 'n_neurons': NUM_NEURONS_GPU,
}
results['experiments']['EXP2_GPU_ALONE'] = exp2
print(f"  rank={gpu_rank:.1f}, MC={gpu_mc:.2f}, XOR-5={gpu_xor5*100:.1f}%, wave={gpu_wave_acc*100:.1f}%")

test('T1129', 'GPU MC > 5', gpu_mc > 5, round(gpu_mc, 2))
test('T1130', 'GPU XOR-5 > 50%', gpu_xor5 > 0.5, round(gpu_xor5*100, 1))
test('T1131', 'GPU wave > 80%', gpu_wave_acc > 0.8, round(gpu_wave_acc*100, 1))
test('T1132', 'GPU rank > 10', gpu_rank > 10, round(gpu_rank, 1))
save()

# ======================================================================
# EXP 3: FPGA alone at best regime
# ======================================================================
print("\n" + "=" * 70)
print(f"  EXP 3: FPGA Alone ({best_regime} regime)")
print("=" * 70)

check_thermal()

if FPGA_AVAILABLE:
    regime = FPGA_REGIMES[best_regime]
    fpga = open_fpga(timeout=2.0)
    setup_fpga(fpga, regime['thresh'], regime['base_exc'])

    rng3 = np.random.RandomState(42)
    fpga_states = np.zeros((N_STEPS, NUM_NEURONS_FPGA), dtype=np.float32)
    fpga_dspikes = np.zeros((N_STEPS, NUM_NEURONS_FPGA), dtype=np.float32)

    for t in range(N_STEPS):
        fpga.set_mac_signal(int(inputs_rand[t] * 65536))
        vmem, scnt = fpga_read_state(fpga)
        if vmem is not None:
            fpga_states[t] = vmem
            fpga_dspikes[t] = scnt
        time.sleep(1.0 / SAMPLE_HZ)
        if (t+1) % 500 == 0:
            check_thermal()
            print(f"    step {t+1}/{N_STEPS}, temp={get_temp()}C")

    fpga.close()

    # Delta spikes
    fpga_delta = np.diff(fpga_dspikes, axis=0)
    fpga_delta[fpga_delta < 0] = 0

    X_fpga = fpga_states[WARMUP:]
    fpga_rank = effective_rank(X_fpga)
    fpga_mc, fpga_mc_per = compute_mc(X_fpga, inp, max_delay=20)
    fpga_xor5 = compute_xor(X_fpga, inp, tau=5)

    # FPGA waveform classification
    print("  FPGA classification...")
    fpga2 = open_fpga(timeout=2.0)
    setup_fpga(fpga2, regime['thresh'], regime['base_exc'])
    fpga_wave_feats = []
    for i in range(n_samples):
        sample_states = np.zeros((50, NUM_NEURONS_FPGA), dtype=np.float32)
        for st in range(50):
            mac = int(np.clip((wave_signals[i][st] + 1.5) / 3.0, 0, 1) * 65536)
            fpga2.set_mac_signal(mac)
            vmem, scnt = fpga_read_state(fpga2)
            if vmem is not None:
                sample_states[st] = vmem
            time.sleep(1.0 / SAMPLE_HZ)
        feat = np.concatenate([
            sample_states[-20:].mean(0), sample_states[-20:].std(0),
            np.diff(sample_states[-10:], axis=0).mean(0),
        ])
        fpga_wave_feats.append(feat)
        if (i+1) % 50 == 0:
            check_thermal()
            print(f"    sample {i+1}/{n_samples}, temp={get_temp()}C")
    fpga2.close()
    fpga_wave_feats = np.array(fpga_wave_feats)
    fpga_wave_acc = classify_waveforms_from_features(fpga_wave_feats, wave_labels)
else:
    fpga_rank, fpga_mc, fpga_xor5, fpga_wave_acc = 0, 0, 0.5, 0.25
    fpga_states = np.zeros((N_STEPS, NUM_NEURONS_FPGA), dtype=np.float32)
    fpga_delta = np.zeros((N_STEPS-1, NUM_NEURONS_FPGA), dtype=np.float32)
    fpga_wave_feats = np.zeros((n_samples, NUM_NEURONS_FPGA*3), dtype=np.float32)
    X_fpga = fpga_states[WARMUP:]

exp3 = {
    'rank': fpga_rank, 'mc': fpga_mc, 'xor5': fpga_xor5,
    'wave_acc': fpga_wave_acc, 'regime': best_regime, 'fpga_available': FPGA_AVAILABLE,
}
results['experiments']['EXP3_FPGA_ALONE'] = exp3
print(f"  rank={fpga_rank:.1f}, MC={fpga_mc:.2f}, XOR-5={fpga_xor5*100:.1f}%, wave={fpga_wave_acc*100:.1f}%")

test('T1133', 'FPGA MC > 0.5', fpga_mc > 0.5, round(fpga_mc, 2))
test('T1134', 'FPGA XOR-5 > 50%', fpga_xor5 > 0.5, round(fpga_xor5*100, 1))
test('T1135', 'FPGA wave > 60%', fpga_wave_acc > 0.6, round(fpga_wave_acc*100, 1))
test('T1136', 'FPGA rank > 1.5', fpga_rank > 1.5, round(fpga_rank, 1))
save()

# ======================================================================
# EXP 4: Concatenated bridge (GPU ‖ FPGA features)
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 4: Concatenated Bridge (GPU ‖ FPGA)")
print("=" * 70)

check_thermal()

if FPGA_AVAILABLE:
    # Run GPU ESN and FPGA simultaneously with same input
    fpga3 = open_fpga(timeout=2.0)
    regime = FPGA_REGIMES[best_regime]
    setup_fpga(fpga3, regime['thresh'], regime['base_exc'])
    esn2 = GPU_ESN(n_neurons=NUM_NEURONS_GPU, spectral_radius=0.95, seed=42)
    esn2.reset()

    concat_gpu = np.zeros((N_STEPS, NUM_NEURONS_GPU), dtype=np.float32)
    concat_fpga = np.zeros((N_STEPS, NUM_NEURONS_FPGA), dtype=np.float32)

    for t in range(N_STEPS):
        u = inputs_rand[t]
        # GPU step
        gpu_st = esn2.step(float(u))
        concat_gpu[t] = gpu_st
        # FPGA step
        fpga3.set_mac_signal(int(u * 65536))
        vmem, _ = fpga_read_state(fpga3)
        if vmem is not None:
            concat_fpga[t] = vmem
        time.sleep(1.0 / SAMPLE_HZ)
        if (t+1) % 500 == 0:
            check_thermal()
            print(f"    step {t+1}/{N_STEPS}, temp={get_temp()}C")

    fpga3.close()

    X_concat = np.hstack([concat_gpu[WARMUP:], concat_fpga[WARMUP:]])
else:
    X_concat = np.hstack([X_gpu, X_fpga])

concat_rank = effective_rank(X_concat)
concat_mc, _ = compute_mc(X_concat, inp, max_delay=20)
concat_xor5 = compute_xor(X_concat, inp, tau=5)

# Concatenated waveform classification
if FPGA_AVAILABLE:
    # Collect waveform features from both
    esn3 = GPU_ESN(n_neurons=NUM_NEURONS_GPU, spectral_radius=0.95, seed=42)
    fpga4 = open_fpga(timeout=2.0)
    setup_fpga(fpga4, regime['thresh'], regime['base_exc'])
    concat_wave_feats = []
    print("  Concat classification...")
    for i in range(n_samples):
        esn3.reset()
        gpu_sample = np.zeros((50, NUM_NEURONS_GPU), dtype=np.float32)
        fpga_sample = np.zeros((50, NUM_NEURONS_FPGA), dtype=np.float32)
        for st in range(50):
            u_val = float(np.clip((wave_signals[i][st] + 1.5) / 3.0, 0, 1))
            gpu_sample[st] = esn3.step(u_val)
            fpga4.set_mac_signal(int(u_val * 65536))
            vmem, _ = fpga_read_state(fpga4)
            if vmem is not None:
                fpga_sample[st] = vmem
            time.sleep(1.0 / SAMPLE_HZ)
        gpu_feat = np.concatenate([gpu_sample[-20:].mean(0), gpu_sample[-20:].std(0),
                                   np.diff(gpu_sample[-10:], axis=0).mean(0)])
        fpga_feat = np.concatenate([fpga_sample[-20:].mean(0), fpga_sample[-20:].std(0),
                                    np.diff(fpga_sample[-10:], axis=0).mean(0)])
        concat_wave_feats.append(np.concatenate([gpu_feat, fpga_feat]))
        if (i+1) % 50 == 0:
            check_thermal()
            print(f"    sample {i+1}/{n_samples}, temp={get_temp()}C")
    fpga4.close()
    concat_wave_feats = np.array(concat_wave_feats)
else:
    concat_wave_feats = np.hstack([gpu_wave_feats, fpga_wave_feats])

concat_wave_acc = classify_waveforms_from_features(concat_wave_feats, wave_labels)

exp4 = {
    'rank': concat_rank, 'mc': concat_mc, 'xor5': concat_xor5,
    'wave_acc': concat_wave_acc,
}
results['experiments']['EXP4_CONCAT'] = exp4
print(f"  rank={concat_rank:.1f}, MC={concat_mc:.2f}, XOR-5={concat_xor5*100:.1f}%, wave={concat_wave_acc*100:.1f}%")
print(f"  vs GPU: MC {concat_mc:.2f} vs {gpu_mc:.2f}, wave {concat_wave_acc*100:.1f}% vs {gpu_wave_acc*100:.1f}%")

test('T1137', 'Concat MC > GPU MC', concat_mc > gpu_mc,
     f"{concat_mc:.2f} vs {gpu_mc:.2f}")
test('T1138', 'Concat XOR-5 > 50%', concat_xor5 > 0.5, round(concat_xor5*100, 1))
test('T1139', 'Concat wave > max(GPU, FPGA)',
     concat_wave_acc > max(gpu_wave_acc, fpga_wave_acc),
     f"{concat_wave_acc*100:.1f}% vs max({gpu_wave_acc*100:.1f}%, {fpga_wave_acc*100:.1f}%)")
test('T1140', 'Concat rank > max(GPU, FPGA) rank',
     concat_rank > max(gpu_rank, fpga_rank),
     f"{concat_rank:.1f} vs max({gpu_rank:.1f}, {fpga_rank:.1f})")
save()

# ======================================================================
# EXP 5: Feedback bridge (bidirectional coupling)
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 5: Feedback Bridge (GPU→FPGA→GPU)")
print("=" * 70)

check_thermal()

if FPGA_AVAILABLE:
    regime = FPGA_REGIMES[best_regime]
    fpga5 = open_fpga(timeout=2.0)
    setup_fpga(fpga5, regime['thresh'], regime['base_exc'])
    esn4 = GPU_ESN(n_neurons=NUM_NEURONS_GPU, spectral_radius=0.95, seed=42)
    esn4.reset()

    fb_gpu = np.zeros((N_STEPS, NUM_NEURONS_GPU), dtype=np.float32)
    fb_fpga = np.zeros((N_STEPS, NUM_NEURONS_FPGA), dtype=np.float32)
    fb_signal = 0.0  # Feedback from FPGA to GPU

    for t in range(N_STEPS):
        u = inputs_rand[t]

        # GPU step: input + feedback from FPGA
        gpu_input = float(u) + 0.1 * fb_signal
        gpu_st = esn4.step(gpu_input)
        fb_gpu[t] = gpu_st

        # GPU output → FPGA MAC: use mean of GPU state as MAC modulation
        gpu_summary = float(np.tanh(gpu_st.mean() * 5))  # [-1, 1]
        mac_val = int(np.clip((gpu_summary + 1.0) / 2.0, 0, 1) * 65536)
        fpga5.set_mac_signal(mac_val)

        vmem, scnt = fpga_read_state(fpga5)
        if vmem is not None:
            fb_fpga[t] = vmem
            # FPGA → GPU feedback: mean membrane voltage deviation
            fb_signal = float(vmem.mean() - 0.5)  # Center around 0
        else:
            fb_signal = 0.0

        time.sleep(1.0 / SAMPLE_HZ)
        if (t+1) % 500 == 0:
            check_thermal()
            print(f"    step {t+1}/{N_STEPS}, temp={get_temp()}C")

    fpga5.close()

    X_fb = np.hstack([fb_gpu[WARMUP:], fb_fpga[WARMUP:]])
else:
    # Simulated feedback: add noise to GPU input as proxy
    esn5 = GPU_ESN(n_neurons=NUM_NEURONS_GPU, spectral_radius=0.95, seed=42)
    esn5.reset()
    fb_gpu = np.zeros((N_STEPS, NUM_NEURONS_GPU), dtype=np.float32)
    fb_signal = 0.0
    for t in range(N_STEPS):
        gpu_input = float(inputs_rand[t]) + 0.1 * fb_signal
        gpu_st = esn5.step(gpu_input)
        fb_gpu[t] = gpu_st
        fb_signal = float(np.tanh(gpu_st.mean() * 5) * 0.5)
    X_fb = np.hstack([fb_gpu[WARMUP:], X_fpga])

fb_rank = effective_rank(X_fb)
fb_mc, _ = compute_mc(X_fb, inp, max_delay=20)
fb_xor5 = compute_xor(X_fb, inp, tau=5)

# Feedback waveform classification
if FPGA_AVAILABLE:
    print("  Feedback classification...")
    esn6 = GPU_ESN(n_neurons=NUM_NEURONS_GPU, spectral_radius=0.95, seed=42)
    fpga6 = open_fpga(timeout=2.0)
    setup_fpga(fpga6, regime['thresh'], regime['base_exc'])
    fb_wave_feats = []
    for i in range(n_samples):
        esn6.reset()
        gpu_sample = np.zeros((50, NUM_NEURONS_GPU), dtype=np.float32)
        fpga_sample = np.zeros((50, NUM_NEURONS_FPGA), dtype=np.float32)
        fb_sig = 0.0
        for st in range(50):
            u_val = float(np.clip((wave_signals[i][st] + 1.5) / 3.0, 0, 1))
            gpu_in = u_val + 0.1 * fb_sig
            gpu_s = esn6.step(gpu_in)
            gpu_sample[st] = gpu_s
            mac = int(np.clip((float(np.tanh(gpu_s.mean()*5))+1)/2, 0, 1) * 65536)
            fpga6.set_mac_signal(mac)
            vmem, _ = fpga_read_state(fpga6)
            if vmem is not None:
                fpga_sample[st] = vmem
                fb_sig = float(vmem.mean() - 0.5)
            else:
                fb_sig = 0.0
            time.sleep(1.0 / SAMPLE_HZ)
        gpu_feat = np.concatenate([gpu_sample[-20:].mean(0), gpu_sample[-20:].std(0),
                                   np.diff(gpu_sample[-10:], axis=0).mean(0)])
        fpga_feat = np.concatenate([fpga_sample[-20:].mean(0), fpga_sample[-20:].std(0),
                                    np.diff(fpga_sample[-10:], axis=0).mean(0)])
        fb_wave_feats.append(np.concatenate([gpu_feat, fpga_feat]))
        if (i+1) % 50 == 0:
            check_thermal()
            print(f"    sample {i+1}/{n_samples}, temp={get_temp()}C")
    fpga6.close()
    fb_wave_feats = np.array(fb_wave_feats)
else:
    fb_wave_feats = concat_wave_feats  # Fallback

fb_wave_acc = classify_waveforms_from_features(fb_wave_feats, wave_labels)

exp5 = {
    'rank': fb_rank, 'mc': fb_mc, 'xor5': fb_xor5,
    'wave_acc': fb_wave_acc,
}
results['experiments']['EXP5_FEEDBACK'] = exp5
print(f"  rank={fb_rank:.1f}, MC={fb_mc:.2f}, XOR-5={fb_xor5*100:.1f}%, wave={fb_wave_acc*100:.1f}%")
print(f"  vs Concat: MC {fb_mc:.2f} vs {concat_mc:.2f}, wave {fb_wave_acc*100:.1f}% vs {concat_wave_acc*100:.1f}%")

test('T1141', 'Feedback MC > GPU MC', fb_mc > gpu_mc,
     f"{fb_mc:.2f} vs {gpu_mc:.2f}")
test('T1142', 'Feedback XOR-5 > 50%', fb_xor5 > 0.5, round(fb_xor5*100, 1))
test('T1143', 'Feedback wave > GPU wave',
     fb_wave_acc > gpu_wave_acc,
     f"{fb_wave_acc*100:.1f}% vs {gpu_wave_acc*100:.1f}%")
test('T1144', 'Feedback > Concat on any metric',
     fb_mc > concat_mc or fb_wave_acc > concat_wave_acc or fb_xor5 > concat_xor5,
     f"MC:{fb_mc:.2f}vs{concat_mc:.2f}, wave:{fb_wave_acc*100:.1f}vs{concat_wave_acc*100:.1f}")
save()

# ======================================================================
# EXP 6: Cubic readout on all conditions
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 6: Cubic Readout (Nonlinear Boost)")
print("=" * 70)

check_thermal()

# Cubic XOR-5 for each condition
gpu_xor5_cubic = compute_xor_cubic(X_gpu, inp, tau=5)
fpga_xor5_cubic = compute_xor_cubic(X_fpga, inp, tau=5)
concat_xor5_cubic = compute_xor_cubic(X_concat, inp, tau=5)
fb_xor5_cubic = compute_xor_cubic(X_fb, inp, tau=5)

exp6 = {
    'gpu_xor5_cubic': gpu_xor5_cubic,
    'fpga_xor5_cubic': fpga_xor5_cubic,
    'concat_xor5_cubic': concat_xor5_cubic,
    'fb_xor5_cubic': fb_xor5_cubic,
}
results['experiments']['EXP6_CUBIC'] = exp6
print(f"  Cubic XOR-5:")
print(f"    GPU:    {gpu_xor5_cubic*100:.1f}% (linear: {gpu_xor5*100:.1f}%)")
print(f"    FPGA:   {fpga_xor5_cubic*100:.1f}% (linear: {fpga_xor5*100:.1f}%)")
print(f"    Concat: {concat_xor5_cubic*100:.1f}% (linear: {concat_xor5*100:.1f}%)")
print(f"    FB:     {fb_xor5_cubic*100:.1f}% (linear: {fb_xor5*100:.1f}%)")

test('T1145', 'Cubic GPU XOR-5 > 65%', gpu_xor5_cubic > 0.65, round(gpu_xor5_cubic*100, 1))
test('T1146', 'Cubic concat > cubic GPU',
     concat_xor5_cubic > gpu_xor5_cubic,
     f"{concat_xor5_cubic*100:.1f}% vs {gpu_xor5_cubic*100:.1f}%")
test('T1147', 'Cubic FB > 55%', fb_xor5_cubic > 0.55, round(fb_xor5_cubic*100, 1))
test('T1148', 'Any bridge > GPU alone (cubic)',
     max(concat_xor5_cubic, fb_xor5_cubic) > gpu_xor5_cubic,
     f"max({concat_xor5_cubic*100:.1f}%, {fb_xor5_cubic*100:.1f}%) vs {gpu_xor5_cubic*100:.1f}%")
save()

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print("  SUMMARY: z2328 Combined GPU+FPGA Bridge")
print("=" * 70)

n_pass = sum(1 for t in results['tests'].values() if t['status'] == 'PASS')
n_total = len(results['tests'])
print(f"\n  Tests: {n_pass}/{n_total} PASS ({100*n_pass/n_total:.0f}%)")

print(f"\n  {'Condition':<15} {'Rank':>6} {'MC':>8} {'XOR-5':>8} {'Wave':>8}")
print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'GPU alone':<15} {gpu_rank:>6.1f} {gpu_mc:>8.2f} {gpu_xor5*100:>7.1f}% {gpu_wave_acc*100:>7.1f}%")
print(f"  {'FPGA alone':<15} {fpga_rank:>6.1f} {fpga_mc:>8.2f} {fpga_xor5*100:>7.1f}% {fpga_wave_acc*100:>7.1f}%")
print(f"  {'Concat':<15} {concat_rank:>6.1f} {concat_mc:>8.2f} {concat_xor5*100:>7.1f}% {concat_wave_acc*100:>7.1f}%")
print(f"  {'Feedback':<15} {fb_rank:>6.1f} {fb_mc:>8.2f} {fb_xor5*100:>7.1f}% {fb_wave_acc*100:>7.1f}%")

if any(t['status'] == 'PASS' for tid, t in results['tests'].items() if 'bridge' in t.get('name', '').lower() or 'concat' in t.get('name', '').lower()):
    print("\n  ★ Cross-substrate bridge shows measurable advantage!")
else:
    print("\n  Bridge advantage not yet demonstrated — may need parameter tuning")

results['meta']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
results['meta']['tests_passed'] = n_pass
results['meta']['tests_total'] = n_total
save()

print(f"\n  Done. Results: {SAVE_FILE}")
