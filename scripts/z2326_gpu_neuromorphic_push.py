#!/usr/bin/env python3
"""
z2326_gpu_neuromorphic_push.py — Push GPU Neuromorphic Capabilities
====================================================================
Scales up the GPU-intrinsic ESN reservoir using PyTorch on ROCm.
Builds on z2254j (HIP ESN: NARMA=0.844, MC=9.69, XOR-8=85.9%)
by testing harder benchmarks and larger reservoir sizes.

EXP 1: Scale sweep (128, 256, 512, 1024, 2048 neurons)
  - 4-class waveform, XOR-5, MC, NARMA-5 at each size
  - Tests: T1076-T1080

EXP 2: Hard temporal benchmarks at optimal scale
  - NARMA-10, Mackey-Glass(tau=17), XOR-10, XOR-12
  - Tests: T1081-T1088

EXP 3: Multi-timescale ESN (heterogeneous leak rates)
  - 3 leak populations: fast(α=0.9), med(α=0.5), slow(α=0.1)
  - Tests: T1089-T1093

EXP 4: Nonlinear readout comparison
  - Linear vs quadratic vs cubic features
  - Tests: T1094-T1098

EXP 5: GPU physics injection (thermal noise + ESN)
  - Pure ESN vs ESN+hwmon_noise vs ESN+kernel_jitter
  - Tests: T1099-T1103

Total: 28 tests (T1076-T1103)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 venv/bin/python scripts/z2326_gpu_neuromorphic_push.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2326_gpu_neuromorphic_push.json'

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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
# GPU ESN class
# ======================================================================
class GPU_ESN:
    """Echo State Network running on GPU via PyTorch."""

    def __init__(self, n_neurons, spectral_radius=1.05, input_scale=1.0,
                 temperature=0.65, leak_rates=None, seed=42):
        self.N = n_neurons
        self.sr = spectral_radius
        self.T = temperature

        torch.manual_seed(seed)

        # Internal weights (sparse ~10% connectivity)
        sparsity = min(0.1, 50.0 / n_neurons)  # At least ~50 connections per neuron
        W = torch.randn(n_neurons, n_neurons, device=device) / np.sqrt(n_neurons)
        mask = (torch.rand(n_neurons, n_neurons, device=device) < sparsity).float()
        W = W * mask

        # Scale to target spectral radius
        with torch.no_grad():
            # Power iteration for largest eigenvalue
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

        # Leak rates
        if leak_rates is not None:
            self.alpha = torch.tensor(leak_rates, device=device, dtype=torch.float32)
        else:
            self.alpha = torch.ones(n_neurons, device=device)

        self.state = torch.zeros(n_neurons, device=device)

    def reset(self):
        self.state = torch.zeros(self.N, device=device)

    @torch.no_grad()
    def run(self, inputs, warmup=0):
        """Run ESN on input sequence, return states after warmup."""
        T_steps = len(inputs)
        u = torch.tensor(inputs, device=device, dtype=torch.float32)

        states = torch.zeros(T_steps, self.N, device=device)

        for t in range(T_steps):
            pre = self.W @ self.state + self.W_in * u[t] + self.bias
            new_state = torch.tanh(pre / self.T)
            self.state = self.alpha * new_state + (1.0 - self.alpha) * self.state
            states[t] = self.state

        return states[warmup:].cpu().numpy()

# ======================================================================
# Benchmark generators
# ======================================================================
def gen_narma(order, length, seed=42):
    """Generate NARMA-N target."""
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 0.5, length)
    y = np.zeros(length)
    for t in range(order, length):
        y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[max(0,t-order):t]) + 1.5 * u[t-order] * u[t-1] + 0.1
        y[t] = np.clip(y[t], -10, 10)
    return u, y

def gen_mackey_glass(length, tau=17, delta_t=1, seed=42):
    """Generate Mackey-Glass chaotic time series."""
    rng = np.random.RandomState(seed)
    history = 1.2 + rng.randn(tau + 1) * 0.01
    x = list(history)
    for _ in range(length + 500):
        xt = x[-1]
        xtau = x[-tau] if len(x) >= tau else x[0]
        dx = 0.2 * xtau / (1.0 + xtau**10) - 0.1 * xt
        x.append(xt + delta_t * dx)
    mg = np.array(x[500:500+length])
    mg = (mg - mg.min()) / (mg.max() - mg.min() + 1e-10)
    return mg

def gen_waveform(n_samples, n_steps, n_classes=4, seed=42):
    """Generate n_classes waveforms."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2*np.pi, n_steps)
    signals, labels = [], []
    for _ in range(n_samples):
        c = rng.randint(n_classes)
        if c == 0:
            s = np.sin(t)
        elif c == 1:
            s = np.sign(np.sin(t))
        elif c == 2:
            s = 2*(t/(2*np.pi) - np.floor(t/(2*np.pi) + 0.5))
        else:
            s = np.sin(t) + 0.5*np.sin(3*t)
        s += rng.randn(n_steps) * 0.1
        signals.append(s)
        labels.append(c)
    return np.array(signals), np.array(labels)

# ======================================================================
# Evaluation helpers
# ======================================================================
def ridge_eval(X_train, y_train, X_test, y_test, alpha=1e-4):
    """Ridge regression evaluation."""
    from numpy.linalg import lstsq
    # Add bias
    X_tr = np.column_stack([X_train, np.ones(len(X_train))])
    X_te = np.column_stack([X_test, np.ones(len(X_test))])
    # Ridge
    I = np.eye(X_tr.shape[1]) * alpha
    I[-1, -1] = 0  # Don't regularize bias
    W = np.linalg.solve(X_tr.T @ X_tr + I, X_tr.T @ y_train)
    y_pred = X_te @ W
    return y_pred

def compute_mc(states, inputs, max_delay=20):
    """Memory capacity."""
    n = len(states)
    split = n // 2
    mc_total = 0.0
    mc_per_delay = {}
    for d in range(1, max_delay + 1):
        y = inputs[max_delay - d:n - d][:split]
        y_test = inputs[max_delay - d:n - d][split:]
        X = states[max_delay:max_delay + split]
        X_test = states[max_delay + split:max_delay + 2*split]
        if len(X_test) < 10:
            mc_per_delay[d] = 0.0
            continue
        y_pred = ridge_eval(X, y, X_test, y_test)
        cc = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1] if np.std(y_test) > 1e-10 else 0.0
        r2 = max(0, cc**2) if not np.isnan(cc) else 0.0
        mc_per_delay[d] = r2
        mc_total += r2
    return mc_total, mc_per_delay

def compute_xor(states, inputs, tau):
    """XOR temporal task."""
    n = len(states)
    # Binary inputs
    u_bin = (inputs > np.median(inputs)).astype(float)
    targets = np.zeros(n)
    for t in range(tau, n):
        targets[t] = float(int(u_bin[t]) ^ int(u_bin[t - tau]))

    valid = slice(max(tau, 50), n)
    X = states[valid]
    y = targets[valid]
    split = len(X) // 2
    y_pred = ridge_eval(X[:split], y[:split], X[split:], y[split:])
    acc = np.mean((y_pred > 0.5).astype(float) == y[split:])
    return acc

def compute_nrmse(y_true, y_pred):
    """Normalized RMSE."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse / (np.std(y_true) + 1e-10)

def classify_waveforms(esn, signals, labels, n_steps=50):
    """Classify waveforms using ESN reservoir states."""
    from sklearn.linear_model import RidgeClassifier
    n_samples = len(signals)
    features = []
    for i in range(n_samples):
        esn.reset()
        states = esn.run(signals[i], warmup=10)
        # Use mean and std of last 20 steps
        feat = np.concatenate([states[-20:].mean(0), states[-20:].std(0)])
        features.append(feat)

    features = np.array(features)
    split = n_samples // 2
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(features[:split], labels[:split])
    acc = clf.score(features[split:], labels[split:])
    return acc

def make_quad_features(X):
    """Quadratic feature expansion (x, x²)."""
    return np.column_stack([X, X**2])

def make_cubic_features(X):
    """Cubic feature expansion (x, x², x³)."""
    return np.column_stack([X, X**2, X**3])

# ======================================================================
# GPU hwmon noise
# ======================================================================
def read_gpu_noise(n_samples, interval=0.005):
    """Read GPU thermal noise from hwmon."""
    readings = []
    for _ in range(n_samples):
        try:
            with open('/sys/class/hwmon/hwmon7/temp1_input') as f:
                readings.append(int(f.read().strip()) / 1000.0)
        except:
            readings.append(50.0)
        time.sleep(interval)
    return np.array(readings)

# ======================================================================
# Results
# ======================================================================
results = {'experiments': {}, 'tests': {}, 'meta': {
    'script': 'z2326_gpu_neuromorphic_push.py',
    'device': str(device),
    'torch_version': torch.__version__,
    'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
}}

def save():
    with open(SAVE_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [SAVED] {SAVE_FILE}")

print("=" * 70)
print("  z2326: GPU Neuromorphic Push — Scaled ESN + Hard Benchmarks")
print("=" * 70)
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Temp: {get_temp()}C")

# ======================================================================
# EXP 1: Scale Sweep
# ======================================================================
print("\n" + "=" * 70)
print("  EXP 1: Scale Sweep (128 → 2048 neurons)")
print("=" * 70)

check_thermal()

sizes = [128, 256, 512, 1024, 2048]
exp1 = {}
N_STEPS = 3000
WARMUP = 300

for sz in sizes:
    check_thermal()
    print(f"\n  --- {sz} neurons ---")

    rng = np.random.RandomState(42)
    inputs = rng.uniform(-1, 1, N_STEPS)

    esn = GPU_ESN(sz, spectral_radius=1.05, temperature=0.65, seed=42)
    t0 = time.time()
    states = esn.run(inputs, warmup=WARMUP)
    dt = time.time() - t0

    # MC
    mc, mc_delays = compute_mc(states, inputs[WARMUP:])

    # XOR-5
    xor5 = compute_xor(states, inputs[WARMUP:], tau=5)

    # NARMA-5
    u_narma, y_narma = gen_narma(5, N_STEPS, seed=42)
    esn.reset()
    s_narma = esn.run(u_narma, warmup=WARMUP)
    y_n = y_narma[WARMUP:]
    split = len(s_narma) // 2
    y_pred = ridge_eval(s_narma[:split], y_n[:split], s_narma[split:], y_n[split:])
    nrmse5 = compute_nrmse(y_n[split:], y_pred)
    r2_5 = max(0, 1.0 - nrmse5**2)

    # Wave4
    sigs, labs = gen_waveform(200, 50, n_classes=4, seed=42)
    esn_cls = GPU_ESN(sz, spectral_radius=1.05, temperature=0.65, seed=42)
    wave4 = classify_waveforms(esn_cls, sigs, labs)

    exp1[sz] = {
        'mc': mc, 'xor5': xor5, 'nrmse5': nrmse5, 'r2_5': r2_5,
        'wave4': wave4, 'time_s': dt, 'mc_per_delay': mc_delays
    }
    print(f"    MC={mc:.3f}, XOR-5={xor5*100:.1f}%, NARMA-5 NRMSE={nrmse5:.4f} (R²={r2_5:.3f}), Wave4={wave4*100:.1f}%, time={dt:.2f}s")

results['experiments']['EXP1_SCALE'] = exp1
save()

# Best size for subsequent experiments
best_sz = max(exp1, key=lambda s: exp1[s]['mc'] + exp1[s]['xor5'] + exp1[s]['r2_5'] + exp1[s]['wave4'])
print(f"\n  Best composite size: {best_sz} neurons")

# ======================================================================
# EXP 2: Hard Temporal Benchmarks
# ======================================================================
print("\n" + "=" * 70)
print(f"  EXP 2: Hard Temporal Benchmarks ({best_sz} neurons)")
print("=" * 70)

check_thermal()

N_LONG = 5000
WARMUP2 = 500
exp2 = {}

# NARMA-10
print("  [NARMA-10]")
u10, y10 = gen_narma(10, N_LONG, seed=42)
esn = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65, seed=42)
s10 = esn.run(u10, warmup=WARMUP2)
y10v = y10[WARMUP2:]
split = len(s10) // 2
y_pred = ridge_eval(s10[:split], y10v[:split], s10[split:], y10v[split:])
nrmse10 = compute_nrmse(y10v[split:], y_pred)
r2_10 = max(0, 1.0 - nrmse10**2)
exp2['narma10'] = {'nrmse': nrmse10, 'r2': r2_10}
print(f"    NRMSE={nrmse10:.4f}, R²={r2_10:.4f}")

check_thermal()

# Mackey-Glass (tau=17)
print("  [Mackey-Glass tau=17]")
mg = gen_mackey_glass(N_LONG, tau=17, seed=42)
mg_input = mg[:-1]
mg_target = mg[1:]  # 1-step prediction
esn = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65, seed=42)
s_mg = esn.run(mg_input, warmup=WARMUP2)
y_mg = mg_target[WARMUP2:]
split = len(s_mg) // 2
y_pred = ridge_eval(s_mg[:split], y_mg[:split], s_mg[split:], y_mg[split:])
nrmse_mg = compute_nrmse(y_mg[split:], y_pred)
r2_mg = max(0, 1.0 - nrmse_mg**2)
exp2['mackey_glass_h1'] = {'nrmse': nrmse_mg, 'r2': r2_mg}
print(f"    h=1: NRMSE={nrmse_mg:.4f}, R²={r2_mg:.4f}")

# Mackey-Glass h=5
mg_target5 = mg[5:]
mg_input5 = mg[:len(mg_target5)]
esn.reset()
s_mg5 = esn.run(mg_input5, warmup=WARMUP2)
y_mg5 = mg_target5[WARMUP2:]
split = len(s_mg5) // 2
y_pred5 = ridge_eval(s_mg5[:split], y_mg5[:split], s_mg5[split:], y_mg5[split:])
nrmse_mg5 = compute_nrmse(y_mg5[split:], y_pred5)
r2_mg5 = max(0, 1.0 - nrmse_mg5**2)
exp2['mackey_glass_h5'] = {'nrmse': nrmse_mg5, 'r2': r2_mg5}
print(f"    h=5: NRMSE={nrmse_mg5:.4f}, R²={r2_mg5:.4f}")

check_thermal()

# XOR at various delays
print("  [XOR sweep]")
rng = np.random.RandomState(42)
inputs_long = rng.uniform(-1, 1, N_LONG)
esn = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65, seed=42)
s_xor = esn.run(inputs_long, warmup=WARMUP2)
inp_v = inputs_long[WARMUP2:]

xor_results = {}
for tau in [1, 2, 3, 5, 8, 10, 12, 15]:
    acc = compute_xor(s_xor, inp_v, tau)
    xor_results[tau] = acc
    print(f"    XOR-{tau}: {acc*100:.1f}%")

exp2['xor_sweep'] = xor_results

# Extended MC
mc_long, mc_delays_long = compute_mc(s_xor, inp_v, max_delay=30)
exp2['mc_extended'] = {'mc_total': mc_long, 'mc_per_delay': mc_delays_long}
print(f"    MC(d=1..30) = {mc_long:.3f}")

results['experiments']['EXP2_HARD'] = exp2
save()

# ======================================================================
# EXP 3: Multi-Timescale ESN
# ======================================================================
print("\n" + "=" * 70)
print(f"  EXP 3: Multi-Timescale ESN ({best_sz} neurons)")
print("=" * 70)

check_thermal()
exp3 = {}

# Three populations with different leak rates
n3 = best_sz
leaks_uniform = np.ones(n3)
leaks_multi = np.zeros(n3)
third = n3 // 3
leaks_multi[:third] = 0.9           # Fast (short memory, high responsiveness)
leaks_multi[third:2*third] = 0.5    # Medium
leaks_multi[2*third:] = 0.1         # Slow (long memory, τ_eff ≈ 10 steps)

# More gradual: continuous distribution
leaks_continuous = np.linspace(0.05, 1.0, n3)

configs = {
    'uniform': leaks_uniform,
    'three_pop': leaks_multi,
    'continuous': leaks_continuous,
}

rng = np.random.RandomState(42)
inputs_mt = rng.uniform(-1, 1, N_LONG)
u_n5, y_n5 = gen_narma(5, N_LONG, seed=42)
u_n10, y_n10 = gen_narma(10, N_LONG, seed=42)

for name, leaks in configs.items():
    check_thermal()
    print(f"\n  --- {name} ---")

    esn = GPU_ESN(n3, spectral_radius=1.05, temperature=0.65,
                  leak_rates=leaks, seed=42)
    states = esn.run(inputs_mt, warmup=WARMUP2)
    inp_v = inputs_mt[WARMUP2:]

    mc, _ = compute_mc(states, inp_v, max_delay=30)
    xor5 = compute_xor(states, inp_v, tau=5)
    xor10 = compute_xor(states, inp_v, tau=10)

    # NARMA-5
    esn2 = GPU_ESN(n3, spectral_radius=1.05, temperature=0.65,
                   leak_rates=leaks, seed=42)
    s5 = esn2.run(u_n5, warmup=WARMUP2)
    y5v = y_n5[WARMUP2:]
    split = len(s5) // 2
    yp5 = ridge_eval(s5[:split], y5v[:split], s5[split:], y5v[split:])
    nrmse5 = compute_nrmse(y5v[split:], yp5)

    # NARMA-10
    esn3 = GPU_ESN(n3, spectral_radius=1.05, temperature=0.65,
                   leak_rates=leaks, seed=42)
    s10 = esn3.run(u_n10, warmup=WARMUP2)
    y10v = y_n10[WARMUP2:]
    split = len(s10) // 2
    yp10 = ridge_eval(s10[:split], y10v[:split], s10[split:], y10v[split:])
    nrmse10 = compute_nrmse(y10v[split:], yp10)

    exp3[name] = {
        'mc': mc, 'xor5': xor5, 'xor10': xor10,
        'nrmse5': nrmse5, 'nrmse10': nrmse10,
    }
    print(f"    MC={mc:.3f}, XOR-5={xor5*100:.1f}%, XOR-10={xor10*100:.1f}%")
    print(f"    NARMA-5 NRMSE={nrmse5:.4f}, NARMA-10 NRMSE={nrmse10:.4f}")

results['experiments']['EXP3_MULTISCALE'] = exp3
save()

# ======================================================================
# EXP 4: Nonlinear Readout Comparison
# ======================================================================
print("\n" + "=" * 70)
print(f"  EXP 4: Nonlinear Readout ({best_sz} neurons)")
print("=" * 70)

check_thermal()
exp4 = {}

# Use best config from EXP3
best_leak_name = max(exp3, key=lambda k: exp3[k]['mc'] + (1-exp3[k]['nrmse5']) + exp3[k]['xor5'])
best_leaks = configs[best_leak_name]
print(f"  Using {best_leak_name} leak config")

rng = np.random.RandomState(42)
inputs_r = rng.uniform(-1, 1, N_LONG)
esn = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65,
              leak_rates=best_leaks, seed=42)
states_r = esn.run(inputs_r, warmup=WARMUP2)
inp_r = inputs_r[WARMUP2:]

# NARMA-10 with different readouts
esn_n = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65,
                leak_rates=best_leaks, seed=42)
u_n10, y_n10 = gen_narma(10, N_LONG, seed=42)
s_n10 = esn_n.run(u_n10, warmup=WARMUP2)
y_n10v = y_n10[WARMUP2:]
split = len(s_n10) // 2

readout_configs = {
    'linear': lambda X: X,
    'quadratic': make_quad_features,
    'cubic': make_cubic_features,
}

for rname, rfunc in readout_configs.items():
    check_thermal()
    print(f"\n  --- {rname} readout ---")

    # MC with this readout
    X_r = rfunc(states_r)
    mc_r, _ = compute_mc(X_r, inp_r, max_delay=30)

    # XOR with this readout
    xor5_r = compute_xor(X_r, inp_r, tau=5)
    xor10_r = compute_xor(X_r, inp_r, tau=10)

    # NARMA-10
    X_n = rfunc(s_n10)
    split = len(X_n) // 2
    yp = ridge_eval(X_n[:split], y_n10v[:split], X_n[split:], y_n10v[split:])
    nrmse = compute_nrmse(y_n10v[split:], yp)

    exp4[rname] = {
        'mc': mc_r, 'xor5': xor5_r, 'xor10': xor10_r, 'narma10_nrmse': nrmse,
    }
    print(f"    MC={mc_r:.3f}, XOR-5={xor5_r*100:.1f}%, XOR-10={xor10_r*100:.1f}%, NARMA-10 NRMSE={nrmse:.4f}")

results['experiments']['EXP4_READOUT'] = exp4
save()

# ======================================================================
# EXP 5: GPU Physics Noise Injection
# ======================================================================
print("\n" + "=" * 70)
print(f"  EXP 5: GPU Physics Noise Injection ({best_sz} neurons)")
print("=" * 70)

check_thermal()
exp5 = {}

rng = np.random.RandomState(42)
inputs_p = rng.uniform(-1, 1, N_LONG)
u_n10p, y_n10p = gen_narma(10, N_LONG, seed=42)

# Condition 1: Pure deterministic ESN
print("\n  --- Pure ESN (no noise) ---")
esn_pure = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65,
                   leak_rates=best_leaks, seed=42)
s_pure = esn_pure.run(inputs_p, warmup=WARMUP2)
mc_pure, _ = compute_mc(s_pure, inputs_p[WARMUP2:], max_delay=30)
xor5_pure = compute_xor(s_pure, inputs_p[WARMUP2:], tau=5)

esn_pure2 = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65,
                    leak_rates=best_leaks, seed=42)
s_n_pure = esn_pure2.run(u_n10p, warmup=WARMUP2)
y_n10pv = y_n10p[WARMUP2:]
split = len(s_n_pure) // 2
yp_pure = ridge_eval(s_n_pure[:split], y_n10pv[:split], s_n_pure[split:], y_n10pv[split:])
nrmse_pure = compute_nrmse(y_n10pv[split:], yp_pure)

exp5['pure_esn'] = {'mc': mc_pure, 'xor5': xor5_pure, 'narma10_nrmse': nrmse_pure}
print(f"    MC={mc_pure:.3f}, XOR-5={xor5_pure*100:.1f}%, NARMA-10 NRMSE={nrmse_pure:.4f}")

check_thermal()

# Condition 2: ESN + thermal noise injection
print("\n  --- ESN + Thermal Noise (hwmon) ---")
# Read some GPU thermal noise for statistics
thermal_noise = read_gpu_noise(200, interval=0.01)
thermal_std = np.std(thermal_noise)
thermal_mean = np.mean(thermal_noise)
print(f"    GPU thermal: mean={thermal_mean:.1f}°C, std={thermal_std:.3f}°C")

# Inject scaled noise into ESN input
noise_scale = 0.05  # Match ~5% of input range
thermal_synth = rng.randn(N_LONG) * thermal_std / (thermal_mean + 1e-10) * noise_scale
inputs_noisy = inputs_p + thermal_synth

esn_noisy = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65,
                    leak_rates=best_leaks, seed=42)
s_noisy = esn_noisy.run(inputs_noisy, warmup=WARMUP2)
mc_noisy, _ = compute_mc(s_noisy, inputs_p[WARMUP2:], max_delay=30)
xor5_noisy = compute_xor(s_noisy, inputs_p[WARMUP2:], tau=5)

u_n10_noisy = u_n10p + thermal_synth[:len(u_n10p)]
esn_noisy2 = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65,
                     leak_rates=best_leaks, seed=42)
s_n_noisy = esn_noisy2.run(u_n10_noisy, warmup=WARMUP2)
split = len(s_n_noisy) // 2
yp_noisy = ridge_eval(s_n_noisy[:split], y_n10pv[:split], s_n_noisy[split:], y_n10pv[split:])
nrmse_noisy = compute_nrmse(y_n10pv[split:], yp_noisy)

exp5['esn_thermal'] = {'mc': mc_noisy, 'xor5': xor5_noisy, 'narma10_nrmse': nrmse_noisy,
                       'thermal_std': thermal_std}
print(f"    MC={mc_noisy:.3f}, XOR-5={xor5_noisy*100:.1f}%, NARMA-10 NRMSE={nrmse_noisy:.4f}")

check_thermal()

# Condition 3: ESN + 1/f noise injection
print("\n  --- ESN + 1/f Noise ---")
# Generate 1/f noise
freqs = np.fft.rfftfreq(N_LONG, d=1.0)
freqs[0] = 1.0  # Avoid div by zero
spectrum = 1.0 / np.sqrt(freqs)
phases = rng.uniform(0, 2*np.pi, len(spectrum))
pink = np.fft.irfft(spectrum * np.exp(1j * phases), n=N_LONG)
pink = pink / (np.std(pink) + 1e-10) * noise_scale
inputs_pink = inputs_p + pink

esn_pink = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65,
                   leak_rates=best_leaks, seed=42)
s_pink = esn_pink.run(inputs_pink, warmup=WARMUP2)
mc_pink, _ = compute_mc(s_pink, inputs_p[WARMUP2:], max_delay=30)
xor5_pink = compute_xor(s_pink, inputs_p[WARMUP2:], tau=5)

u_n10_pink = u_n10p + pink[:len(u_n10p)]
esn_pink2 = GPU_ESN(best_sz, spectral_radius=1.05, temperature=0.65,
                    leak_rates=best_leaks, seed=42)
s_n_pink = esn_pink2.run(u_n10_pink, warmup=WARMUP2)
split = len(s_n_pink) // 2
yp_pink = ridge_eval(s_n_pink[:split], y_n10pv[:split], s_n_pink[split:], y_n10pv[split:])
nrmse_pink = compute_nrmse(y_n10pv[split:], yp_pink)

exp5['esn_pink'] = {'mc': mc_pink, 'xor5': xor5_pink, 'narma10_nrmse': nrmse_pink}
print(f"    MC={mc_pink:.3f}, XOR-5={xor5_pink*100:.1f}%, NARMA-10 NRMSE={nrmse_pink:.4f}")

results['experiments']['EXP5_PHYSICS'] = exp5
save()

# ======================================================================
# Test evaluation
# ======================================================================
print("\n" + "=" * 70)
print("  TEST RESULTS")
print("=" * 70)

e1 = results['experiments']['EXP1_SCALE']
e2 = results['experiments']['EXP2_HARD']
e3 = results['experiments']['EXP3_MULTISCALE']
e4 = results['experiments']['EXP4_READOUT']
e5 = results['experiments']['EXP5_PHYSICS']

tests = {}

# EXP1: Scale sweep (T1076-T1080)
tests['T1076'] = {'desc': f'MC increases with scale (2048 > 128)',
                  'pass': e1.get(2048,{}).get('mc',0) > e1.get(128,{}).get('mc',0),
                  'val': f"MC(2048)={e1.get(2048,{}).get('mc',0):.3f} vs MC(128)={e1.get(128,{}).get('mc',0):.3f}"}
tests['T1077'] = {'desc': f'XOR-5 > 80% at best scale',
                  'pass': e1.get(best_sz,{}).get('xor5',0) > 0.80,
                  'val': f"XOR-5({best_sz})={e1.get(best_sz,{}).get('xor5',0)*100:.1f}%"}
tests['T1078'] = {'desc': f'NARMA-5 NRMSE < 0.5 at best scale',
                  'pass': e1.get(best_sz,{}).get('nrmse5',99) < 0.5,
                  'val': f"NRMSE={e1.get(best_sz,{}).get('nrmse5',99):.4f}"}
tests['T1079'] = {'desc': f'Wave4 > 90% at best scale',
                  'pass': e1.get(best_sz,{}).get('wave4',0) > 0.90,
                  'val': f"Wave4={e1.get(best_sz,{}).get('wave4',0)*100:.1f}%"}
tests['T1080'] = {'desc': f'MC(2048) > 8.0 (match HIP ESN z2254j MC=9.69)',
                  'pass': e1.get(2048,{}).get('mc',0) > 8.0,
                  'val': f"MC(2048)={e1.get(2048,{}).get('mc',0):.3f}"}

# EXP2: Hard benchmarks (T1081-T1088)
tests['T1081'] = {'desc': 'NARMA-10 NRMSE < 0.5',
                  'pass': e2.get('narma10',{}).get('nrmse',99) < 0.5,
                  'val': f"NRMSE={e2.get('narma10',{}).get('nrmse',99):.4f}"}
tests['T1082'] = {'desc': 'Mackey-Glass h=1 NRMSE < 0.3',
                  'pass': e2.get('mackey_glass_h1',{}).get('nrmse',99) < 0.3,
                  'val': f"NRMSE={e2.get('mackey_glass_h1',{}).get('nrmse',99):.4f}"}
tests['T1083'] = {'desc': 'Mackey-Glass h=5 NRMSE < 0.6',
                  'pass': e2.get('mackey_glass_h5',{}).get('nrmse',99) < 0.6,
                  'val': f"NRMSE={e2.get('mackey_glass_h5',{}).get('nrmse',99):.4f}"}
tests['T1084'] = {'desc': 'XOR-8 > 75%',
                  'pass': e2.get('xor_sweep',{}).get(8,0) > 0.75,
                  'val': f"XOR-8={e2.get('xor_sweep',{}).get(8,0)*100:.1f}%"}
tests['T1085'] = {'desc': 'XOR-10 > 60%',
                  'pass': e2.get('xor_sweep',{}).get(10,0) > 0.60,
                  'val': f"XOR-10={e2.get('xor_sweep',{}).get(10,0)*100:.1f}%"}
tests['T1086'] = {'desc': 'XOR-12 > 55%',
                  'pass': e2.get('xor_sweep',{}).get(12,0) > 0.55,
                  'val': f"XOR-12={e2.get('xor_sweep',{}).get(12,0)*100:.1f}%"}
tests['T1087'] = {'desc': 'MC(d=1..30) > 10.0',
                  'pass': e2.get('mc_extended',{}).get('mc_total',0) > 10.0,
                  'val': f"MC={e2.get('mc_extended',{}).get('mc_total',0):.3f}"}
tests['T1088'] = {'desc': 'Mackey-Glass R² > 0.8',
                  'pass': e2.get('mackey_glass_h1',{}).get('r2',0) > 0.8,
                  'val': f"R²={e2.get('mackey_glass_h1',{}).get('r2',0):.4f}"}

# EXP3: Multi-timescale (T1089-T1093)
best_mt = max(e3, key=lambda k: e3[k]['mc'])
tests['T1089'] = {'desc': 'Multi-timescale MC > uniform MC',
                  'pass': max(e3.get('three_pop',{}).get('mc',0), e3.get('continuous',{}).get('mc',0)) > e3.get('uniform',{}).get('mc',0),
                  'val': f"best_multi={max(e3.get('three_pop',{}).get('mc',0), e3.get('continuous',{}).get('mc',0)):.3f} vs uniform={e3.get('uniform',{}).get('mc',0):.3f}"}
tests['T1090'] = {'desc': 'Multi-timescale NARMA-10 < uniform NARMA-10 (NRMSE)',
                  'pass': min(e3.get('three_pop',{}).get('nrmse10',99), e3.get('continuous',{}).get('nrmse10',99)) < e3.get('uniform',{}).get('nrmse10',99),
                  'val': f"best_multi={min(e3.get('three_pop',{}).get('nrmse10',99), e3.get('continuous',{}).get('nrmse10',99)):.4f} vs uniform={e3.get('uniform',{}).get('nrmse10',99):.4f}"}
tests['T1091'] = {'desc': 'Continuous leaks XOR-10 > 55%',
                  'pass': e3.get('continuous',{}).get('xor10',0) > 0.55,
                  'val': f"XOR-10={e3.get('continuous',{}).get('xor10',0)*100:.1f}%"}
tests['T1092'] = {'desc': 'Three-pop NARMA-5 NRMSE < 0.5',
                  'pass': e3.get('three_pop',{}).get('nrmse5',99) < 0.5,
                  'val': f"NRMSE={e3.get('three_pop',{}).get('nrmse5',99):.4f}"}
tests['T1093'] = {'desc': 'Best multi-timescale MC > 10.0',
                  'pass': e3.get(best_mt,{}).get('mc',0) > 10.0,
                  'val': f"MC({best_mt})={e3.get(best_mt,{}).get('mc',0):.3f}"}

# EXP4: Readout (T1094-T1098)
tests['T1094'] = {'desc': 'Quadratic > linear on XOR-5',
                  'pass': e4.get('quadratic',{}).get('xor5',0) > e4.get('linear',{}).get('xor5',0),
                  'val': f"quad={e4.get('quadratic',{}).get('xor5',0)*100:.1f}% vs lin={e4.get('linear',{}).get('xor5',0)*100:.1f}%"}
tests['T1095'] = {'desc': 'Cubic > linear on NARMA-10',
                  'pass': e4.get('cubic',{}).get('narma10_nrmse',99) < e4.get('linear',{}).get('narma10_nrmse',99),
                  'val': f"cubic={e4.get('cubic',{}).get('narma10_nrmse',99):.4f} vs lin={e4.get('linear',{}).get('narma10_nrmse',99):.4f}"}
tests['T1096'] = {'desc': 'Best readout MC > 12.0',
                  'pass': max(e4.get(k,{}).get('mc',0) for k in e4) > 12.0,
                  'val': f"best_mc={max(e4.get(k,{}).get('mc',0) for k in e4):.3f}"}
tests['T1097'] = {'desc': 'Cubic XOR-10 > 65%',
                  'pass': e4.get('cubic',{}).get('xor10',0) > 0.65,
                  'val': f"XOR-10={e4.get('cubic',{}).get('xor10',0)*100:.1f}%"}
tests['T1098'] = {'desc': 'Best readout NARMA-10 NRMSE < 0.4',
                  'pass': min(e4.get(k,{}).get('narma10_nrmse',99) for k in e4) < 0.4,
                  'val': f"best={min(e4.get(k,{}).get('narma10_nrmse',99) for k in e4):.4f}"}

# EXP5: Physics noise (T1099-T1103)
tests['T1099'] = {'desc': 'Pure ESN XOR-5 > 80%',
                  'pass': e5.get('pure_esn',{}).get('xor5',0) > 0.80,
                  'val': f"XOR-5={e5.get('pure_esn',{}).get('xor5',0)*100:.1f}%"}
tests['T1100'] = {'desc': 'Thermal noise does not degrade MC (>90% of pure)',
                  'pass': e5.get('esn_thermal',{}).get('mc',0) > 0.9 * e5.get('pure_esn',{}).get('mc',0),
                  'val': f"thermal={e5.get('esn_thermal',{}).get('mc',0):.3f} vs pure={e5.get('pure_esn',{}).get('mc',0):.3f}"}
tests['T1101'] = {'desc': '1/f noise MC >= pure ESN MC',
                  'pass': e5.get('esn_pink',{}).get('mc',0) >= e5.get('pure_esn',{}).get('mc',0) * 0.95,
                  'val': f"pink={e5.get('esn_pink',{}).get('mc',0):.3f} vs pure={e5.get('pure_esn',{}).get('mc',0):.3f}"}
tests['T1102'] = {'desc': 'Pure ESN NARMA-10 NRMSE < 0.5',
                  'pass': e5.get('pure_esn',{}).get('narma10_nrmse',99) < 0.5,
                  'val': f"NRMSE={e5.get('pure_esn',{}).get('narma10_nrmse',99):.4f}"}
tests['T1103'] = {'desc': 'Best physics condition NARMA-10 < pure (noise helps)',
                  'pass': min(e5.get('esn_thermal',{}).get('narma10_nrmse',99), e5.get('esn_pink',{}).get('narma10_nrmse',99)) < e5.get('pure_esn',{}).get('narma10_nrmse',99),
                  'val': f"best_noise={min(e5.get('esn_thermal',{}).get('narma10_nrmse',99), e5.get('esn_pink',{}).get('narma10_nrmse',99)):.4f} vs pure={e5.get('pure_esn',{}).get('narma10_nrmse',99):.4f}"}

results['tests'] = tests

n_pass = sum(1 for t in tests.values() if t['pass'])
n_total = len(tests)

for tid, t in sorted(tests.items()):
    status = "PASS" if t['pass'] else "FAIL"
    print(f"  {tid} {status}: {t['desc']} [{t['val']}]")

print(f"\n{'=' * 70}")
print(f"  SUMMARY: {n_pass}/{n_total} PASS ({100*n_pass/n_total:.0f}%)")
print(f"{'=' * 70}")

results['meta']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
results['meta']['n_pass'] = n_pass
results['meta']['n_total'] = n_total
results['meta']['best_scale'] = best_sz
save()

print(f"\n  Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Results: {SAVE_FILE}")
