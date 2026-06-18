#!/usr/bin/env python3
"""
z1602 - Nonlinear Readout & Task Suite for GPU Thermal Reservoir
================================================================

Builds on z1600 finding: GPU thermal reservoir has MC=3.55 but lost to
linear delay embedding on NARMA-10 (which is substantially linear).

Key question: Does the GPU thermal reservoir provide NONLINEAR computation?
If yes, a nonlinear readout (MLP) should beat a linear readout (Ridge)
on the SAME physical states, and the gap should be larger for tasks
requiring nonlinearity.

Experimental design (2x3 factorial + baselines):

Readouts:
  - Ridge regression (linear)
  - MLP (2-layer, 64 hidden, ReLU — nonlinear)

Tasks:
  1. NARMA-10: Standard RC benchmark (substantial linear component)
  2. Temporal parity: XOR(sign(u(t)-0.25), sign(u(t-1)-0.25))
     Requires nonlinearity — linear readout CANNOT solve XOR
  3. Self-prediction: predict GPU power(t+k) from reservoir state(t)
     Tests privileged access to internal dynamics

Conditions:
  A: Physical GPU reservoir (real hardware dynamics)
  B: ESN (software baseline, matched dimensionality)
  C: Linear delay embedding (no nonlinear transform)

If Physical+MLP >> Physical+Ridge on parity task, GPU provides nonlinear
computation. If Physical > ESN on self-prediction, GPU has privileged
access to its own dynamics (true embodiment).

Usage:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z1602_nonlinear_readout_tasks.py
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z1602_nonlinear_readout_tasks.py --steps 500 --dt 1.0
"""

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys
import json
import time
import argparse
import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Telemetry (reuse from z1600)
# ---------------------------------------------------------------------------
_telemetry = None
_use_mock = False


def init_telemetry():
    global _telemetry, _use_mock
    try:
        from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
        _telemetry = SysfsHwmonTelemetry()
        s = _telemetry.read_sample()
        print(f"[telemetry] sysfs hwmon OK: {s.temp_edge_c:.1f}°C, {s.power_w:.1f}W")
        _use_mock = False
        return True
    except Exception as e:
        print(f"[telemetry] unavailable ({e}), using mock")
        _use_mock = True
        return False


def read_gpu_state():
    """Read 7-dim GPU state vector. Fixed: use temp_edge (index 1) for primary temp."""
    if _use_mock:
        return np.array([50.0, 45.0, 20.0, 2.0, 0.5, 40.0, 3.0]) + np.random.randn(7) * 0.5
    s = _telemetry.read_sample()
    return np.array([
        s.temp_edge_c,           # 0: edge temp (WORKING on gfx1151)
        s.power_w,               # 1: instantaneous power
        s.freq_sclk_mhz / 1000, # 2: GPU clock (normalized)
        s.gpu_busy_pct / 100.0,  # 3: GPU utilization (normalized)
        s.temp_junction_c,       # 4: junction temp (may be 0 on gfx1151)
        s.temp_mem_c,            # 5: mem temp (may be 0 on gfx1151)
    ])


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def generate_narma10(T, seed=42):
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for t in range(10, T):
        y_sum = sum(y[t - 1 - i] for i in range(10))
        y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * y_sum + 1.5 * u[t - 10] * u[t] + 0.1
        y[t] = np.clip(y[t], -1e6, 1e6)
    return u, y


def generate_temporal_parity(u, delay=1):
    """
    Temporal XOR: target(t) = XOR(bit(u(t)), bit(u(t-delay)))
    where bit(x) = 1 if x > 0.25 else 0.
    Requires NONLINEARITY — linear readout cannot solve XOR.
    """
    T = len(u)
    bits = (u > 0.25).astype(float)
    targets = np.zeros(T)
    for t in range(delay, T):
        targets[t] = float(int(bits[t]) ^ int(bits[t - delay]))
    return targets


def generate_temporal_parity_3way(u, delays=(1, 2)):
    """3-way temporal parity: XOR(bit(u(t)), bit(u(t-d1)), bit(u(t-d2)))."""
    T = len(u)
    bits = (u > 0.25).astype(float)
    targets = np.zeros(T)
    d_max = max(delays)
    for t in range(d_max, T):
        val = int(bits[t])
        for d in delays:
            val ^= int(bits[t - d])
        targets[t] = float(val)
    return targets


# ---------------------------------------------------------------------------
# Physical GPU Reservoir
# ---------------------------------------------------------------------------
def run_physical_reservoir(u, dt=1.0, min_size=128, max_size=3072,
                           constant_load=False, device_str='cuda'):
    T = len(u)
    state_dim = 7  # 6 telemetry + kernel time
    states = np.zeros((T, state_dim))
    power_trace = np.zeros(T)  # separate power trace for self-prediction

    device = torch.device(device_str)

    # Warmup
    print(f"  Warming up GPU...")
    for _ in range(5):
        A = torch.randn(1024, 1024, device=device)
        B = torch.randn(1024, 1024, device=device)
        _ = torch.mm(A, B)
        torch.cuda.synchronize()
    del A, B
    time.sleep(1.0)

    est_min = T * dt / 60
    print(f"  Running: {T} steps, dt={dt}s (~{est_min:.1f} min)")

    for t in range(T):
        step_start = time.monotonic()

        if constant_load:
            size = (min_size + max_size) // 2
        else:
            size = int(min_size + (max_size - min_size) * (u[t] / 0.5))
            size = max(min_size, min(max_size, size))

        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        _ = torch.mm(A, B)
        torch.cuda.synchronize()
        del A, B

        kernel_time = time.monotonic() - step_start

        elapsed = time.monotonic() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

        gpu_state = read_gpu_state()
        states[t, :6] = gpu_state
        states[t, 6] = kernel_time * 1000  # ms
        power_trace[t] = gpu_state[1]      # power_w

        if t % 100 == 0:
            print(f"    step {t}/{T}: temp={gpu_state[0]:.1f}°C, "
                  f"power={gpu_state[1]:.1f}W, freq={gpu_state[2]*1000:.0f}MHz, "
                  f"size={size}, kernel={kernel_time*1000:.1f}ms")

    return states, power_trace


# ---------------------------------------------------------------------------
# Echo State Network
# ---------------------------------------------------------------------------
class EchoStateNetwork:
    def __init__(self, input_dim=1, reservoir_size=100,
                 spectral_radius=0.95, input_scaling=0.1,
                 leak_rate=0.3, seed=42):
        rng = np.random.RandomState(seed)
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.W_in = rng.randn(reservoir_size, input_dim) * input_scaling
        W = rng.randn(reservoir_size, reservoir_size)
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / rho)
        self.state = np.zeros(reservoir_size)

    def reset(self):
        self.state = np.zeros(self.reservoir_size)

    def run(self, inputs):
        T = len(inputs)
        states = np.zeros((T, self.reservoir_size))
        for t in range(T):
            x = np.atleast_1d(inputs[t])
            pre = np.tanh(self.W @ self.state + self.W_in @ x)
            self.state = (1 - self.leak_rate) * self.state + self.leak_rate * pre
            states[t] = self.state.copy()
        return states


# ---------------------------------------------------------------------------
# State augmentation & normalization
# ---------------------------------------------------------------------------
def augment_with_delays(states, delays=(1, 2, 3)):
    T, D = states.shape
    n_delays = len(delays)
    augmented = np.zeros((T, D * (1 + n_delays)))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start_col = D * (i + 1)
        augmented[d:, start_col:start_col + D] = states[:T - d]
    return augmented


def normalize_states(states):
    mean = states.mean(axis=0, keepdims=True)
    std = states.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0
    return (states - mean) / std, mean, std


def prepare_states(states_raw, delays=(1, 2, 3)):
    """Full pipeline: augment, normalize, add bias."""
    aug = augment_with_delays(states_raw, delays)
    normed, mean, std = normalize_states(aug)
    T = normed.shape[0]
    return np.hstack([normed, np.ones((T, 1))]), mean, std


# ---------------------------------------------------------------------------
# Readout: Ridge Regression
# ---------------------------------------------------------------------------
def ridge_fit_predict(X_train, y_train, X_test, alphas=None):
    if alphas is None:
        alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0]
    best_w, best_alpha, best_err = None, alphas[0], float('inf')
    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue
        err = np.mean((y_train - X_train @ w) ** 2)
        if err < best_err:
            best_err, best_w, best_alpha = err, w, alpha
    return X_test @ best_w, best_alpha


# ---------------------------------------------------------------------------
# Readout: MLP (PyTorch)
# ---------------------------------------------------------------------------
class MLPReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_fit_predict(X_train, y_train, X_test, hidden_dim=64,
                    epochs=300, lr=1e-3, device='cpu'):
    """Train small MLP readout and predict."""
    input_dim = X_train.shape[1]
    model = MLPReadout(input_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_tr = torch.from_numpy(X_train).float().to(device)
    y_tr = torch.from_numpy(y_train).float().to(device)
    X_te = torch.from_numpy(X_test).float().to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).cpu().numpy()

    return y_pred, float(loss.item())


# ---------------------------------------------------------------------------
# Readout: Polynomial Ridge (degree 2)
# ---------------------------------------------------------------------------
def polynomial_features(X, degree=2):
    """Generate degree-2 polynomial features (pairs of columns)."""
    T, D = X.shape
    # Keep original features + all pairwise products
    # For D features, this gives D + D*(D+1)/2 features
    n_pairs = D * (D + 1) // 2
    X_poly = np.zeros((T, D + n_pairs))
    X_poly[:, :D] = X

    idx = D
    for i in range(D):
        for j in range(i, D):
            X_poly[:, idx] = X[:, i] * X[:, j]
            idx += 1

    return X_poly


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    sigma = np.std(y_true)
    return rmse / sigma if sigma > 1e-10 else rmse


def compute_accuracy(y_true, y_pred, threshold=0.5):
    """Binary classification accuracy."""
    pred_binary = (y_pred > threshold).astype(float)
    return np.mean(pred_binary == y_true)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = np.percentile(vals, (1 - ci) / 2 * 100)
    hi = np.percentile(vals, (1 + ci) / 2 * 100)
    return lo, hi


def compute_memory_capacity(states, u, washout, train_end, max_delay=20):
    mc_total = 0.0
    mc_per_delay = []
    T = len(u)
    for k in range(1, max_delay + 1):
        target = np.zeros(T)
        target[k:] = u[:T - k]
        X_train = states[washout:train_end]
        y_train = target[washout:train_end]
        X_test = states[train_end:]
        y_test = target[train_end:]
        if np.std(y_test) < 1e-10:
            mc_per_delay.append(0.0)
            continue
        try:
            y_pred, _ = ridge_fit_predict(X_train, y_train, X_test)
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            r2 = corr ** 2 if not np.isnan(corr) else 0.0
        except Exception:
            r2 = 0.0
        mc_per_delay.append(r2)
        mc_total += r2
    return mc_total, mc_per_delay


# ---------------------------------------------------------------------------
# Evaluate one reservoir on all tasks with all readouts
# ---------------------------------------------------------------------------
def evaluate_reservoir(name, states_raw, u, y_narma, power_trace,
                       washout, train_end, max_delay=20, device='cpu'):
    """Evaluate a reservoir with multiple readouts on multiple tasks."""
    print(f"\n  Evaluating {name}...")
    states, _, _ = prepare_states(states_raw)
    T = states.shape[0]
    input_dim = states.shape[1]

    # Split indices
    X_train = states[washout:train_end]
    X_test = states[train_end:]

    results = {'reservoir': name, 'state_dim': input_dim}

    # ---- Task 1: NARMA-10 ----
    y_tr = y_narma[washout:train_end]
    y_te = y_narma[train_end:]

    # Ridge
    pred_ridge, alpha = ridge_fit_predict(X_train, y_tr, X_test)
    nrmse_ridge = compute_nrmse(y_te, pred_ridge)

    # MLP
    pred_mlp, mlp_loss = mlp_fit_predict(X_train, y_tr, X_test,
                                         hidden_dim=64, epochs=300, device=device)
    nrmse_mlp = compute_nrmse(y_te, pred_mlp)

    # Polynomial Ridge (degree 2)
    X_poly_tr = polynomial_features(X_train, degree=2)
    X_poly_te = polynomial_features(X_test, degree=2)
    pred_poly, alpha_poly = ridge_fit_predict(X_poly_tr, y_tr, X_poly_te)
    nrmse_poly = compute_nrmse(y_te, pred_poly)

    results['narma10'] = {
        'ridge_nrmse': float(nrmse_ridge),
        'mlp_nrmse': float(nrmse_mlp),
        'poly2_nrmse': float(nrmse_poly),
        'ridge_ci': [float(x) for x in bootstrap_ci(y_te, pred_ridge, compute_nrmse)],
        'mlp_ci': [float(x) for x in bootstrap_ci(y_te, pred_mlp, compute_nrmse)],
        'nonlinear_gain_mlp': float(nrmse_ridge - nrmse_mlp),  # positive = MLP better
        'nonlinear_gain_poly': float(nrmse_ridge - nrmse_poly),
    }
    print(f"    NARMA-10: Ridge={nrmse_ridge:.4f}, MLP={nrmse_mlp:.4f}, "
          f"Poly2={nrmse_poly:.4f} (gain={nrmse_ridge-nrmse_mlp:+.4f})")

    # ---- Task 2: Temporal Parity (XOR) ----
    y_parity = generate_temporal_parity(u, delay=1)
    y_tr_p = y_parity[washout:train_end]
    y_te_p = y_parity[train_end:]

    # Ridge
    pred_ridge_p, _ = ridge_fit_predict(X_train, y_tr_p, X_test)
    acc_ridge = compute_accuracy(y_te_p, pred_ridge_p)

    # MLP
    pred_mlp_p, _ = mlp_fit_predict(X_train, y_tr_p, X_test,
                                    hidden_dim=64, epochs=300, device=device)
    acc_mlp = compute_accuracy(y_te_p, pred_mlp_p)

    # Polynomial Ridge
    pred_poly_p, _ = ridge_fit_predict(X_poly_tr, y_tr_p, X_poly_te)
    acc_poly = compute_accuracy(y_te_p, pred_poly_p)

    results['temporal_parity'] = {
        'ridge_accuracy': float(acc_ridge),
        'mlp_accuracy': float(acc_mlp),
        'poly2_accuracy': float(acc_poly),
        'ridge_ci': [float(x) for x in bootstrap_ci(y_te_p, pred_ridge_p, compute_accuracy)],
        'mlp_ci': [float(x) for x in bootstrap_ci(y_te_p, pred_mlp_p, compute_accuracy)],
        'chance_level': 0.5,
        'nonlinear_gain': float(acc_mlp - acc_ridge),  # positive = MLP better
    }
    print(f"    Parity:   Ridge={acc_ridge:.3f}, MLP={acc_mlp:.3f}, "
          f"Poly2={acc_poly:.3f} (gain={acc_mlp-acc_ridge:+.3f})")

    # ---- Task 2b: 3-way temporal parity (harder) ----
    y_par3 = generate_temporal_parity_3way(u, delays=(1, 2))
    y_tr_p3 = y_par3[washout:train_end]
    y_te_p3 = y_par3[train_end:]

    pred_ridge_p3, _ = ridge_fit_predict(X_train, y_tr_p3, X_test)
    acc_ridge_3 = compute_accuracy(y_te_p3, pred_ridge_p3)

    pred_mlp_p3, _ = mlp_fit_predict(X_train, y_tr_p3, X_test,
                                     hidden_dim=64, epochs=300, device=device)
    acc_mlp_3 = compute_accuracy(y_te_p3, pred_mlp_p3)

    results['temporal_parity_3way'] = {
        'ridge_accuracy': float(acc_ridge_3),
        'mlp_accuracy': float(acc_mlp_3),
        'chance_level': 0.5,
        'nonlinear_gain': float(acc_mlp_3 - acc_ridge_3),
    }
    print(f"    Parity3:  Ridge={acc_ridge_3:.3f}, MLP={acc_mlp_3:.3f} "
          f"(gain={acc_mlp_3-acc_ridge_3:+.3f})")

    # ---- Task 3: Self-prediction (predict power k steps ahead) ----
    if power_trace is not None and np.std(power_trace) > 0.01:
        self_pred_results = {}
        for k in [1, 2, 3, 5]:
            target = np.zeros(T)
            target[:-k] = power_trace[k:]  # power at t+k
            y_tr_s = target[washout:train_end - k]  # avoid edge effects
            y_te_s = target[train_end:T - k]

            X_tr_s = X_train[:len(y_tr_s)]
            X_te_s = X_test[:len(y_te_s)]

            if len(y_te_s) < 10:
                continue

            pred_ridge_s, _ = ridge_fit_predict(X_tr_s, y_tr_s, X_te_s)
            nrmse_s = compute_nrmse(y_te_s, pred_ridge_s)

            pred_mlp_s, _ = mlp_fit_predict(X_tr_s, y_tr_s, X_te_s,
                                            hidden_dim=32, epochs=200, device=device)
            nrmse_mlp_s = compute_nrmse(y_te_s, pred_mlp_s)

            self_pred_results[f'k={k}'] = {
                'ridge_nrmse': float(nrmse_s),
                'mlp_nrmse': float(nrmse_mlp_s),
                'power_std_target': float(np.std(y_te_s)),
            }
            print(f"    Self t+{k}: Ridge={nrmse_s:.4f}, MLP={nrmse_mlp_s:.4f}")

        results['self_prediction'] = self_pred_results
    else:
        results['self_prediction'] = 'skipped (no power trace or zero variance)'
        print(f"    Self-prediction: skipped")

    # ---- Memory Capacity ----
    mc_total, mc_per_delay = compute_memory_capacity(states, u, washout, train_end, max_delay)
    results['memory_capacity'] = float(mc_total)
    results['mc_per_delay'] = [float(x) for x in mc_per_delay]
    print(f"    MC: {mc_total:.2f}")

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='z1602: Nonlinear Readout & Task Suite')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--dt', type=float, default=1.0,
                        help='Time step seconds (default: 1.0, matching GPU thermal tau)')
    parser.add_argument('--min-size', type=int, default=128)
    parser.add_argument('--max-size', type=int, default=3072)
    parser.add_argument('--esn-size', type=int, default=50)
    parser.add_argument('--max-delay', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-physical', action='store_true')
    args = parser.parse_args()

    T = args.steps
    dt = args.dt
    washout = int(T * 0.15)
    train_end = int(T * 0.75)
    test_size = T - train_end

    print("=" * 70)
    print("z1602: Nonlinear Readout & Task Suite")
    print("=" * 70)
    print(f"  Steps: {T}, dt: {dt}s")
    print(f"  Washout: {washout}, Train: {train_end - washout}, Test: {test_size}")
    print(f"  Tasks: NARMA-10, Temporal Parity (1-step, 3-way), Self-Prediction")
    print(f"  Readouts: Ridge, MLP(64h), Polynomial-Ridge(deg=2)")
    print()

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    has_telemetry = init_telemetry()
    mlp_device = 'cuda' if has_cuda else 'cpu'
    print()

    # Generate signals
    print("[1/5] Generating signals...")
    u, y_narma = generate_narma10(T, seed=args.seed)
    print(f"  Input range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"  NARMA-10 target std: {np.std(y_narma[train_end:]):.4f}")

    y_parity = generate_temporal_parity(u, delay=1)
    parity_balance = np.mean(y_parity[washout:])
    print(f"  Parity balance: {parity_balance:.3f} (0.5 = perfectly balanced)")

    results = {
        'experiment': 'z1602_nonlinear_readout_tasks',
        'timestamp': datetime.datetime.now().isoformat(),
        'hypothesis': 'GPU thermal reservoir provides nonlinear computation detectable '
                      'by comparing linear (Ridge) vs nonlinear (MLP) readouts. '
                      'Nonlinear gain should be largest on XOR parity task.',
        'hardware': {
            'gpu': torch.cuda.get_device_name(0) if has_cuda else 'CPU',
            'telemetry': 'sysfs_hwmon (real)' if has_telemetry else 'mock',
        },
        'config': {
            'steps': T, 'dt_s': dt, 'washout': washout,
            'train_size': train_end - washout, 'test_size': test_size,
            'esn_size': args.esn_size, 'max_delay': args.max_delay,
        },
        'conditions': {},
    }

    # ------------------------------------------------------------------
    # Condition A: Physical GPU Reservoir
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[2/5] Condition A: Physical GPU Thermal Reservoir")
        t0 = time.time()
        states_A, power_A = run_physical_reservoir(
            u, dt=dt, min_size=args.min_size, max_size=args.max_size
        )
        time_A = time.time() - t0
        print(f"  Physical run: {time_A:.1f}s")
        print(f"  Temp range: {states_A[:, 0].min():.1f}-{states_A[:, 0].max():.1f}°C")
        print(f"  Power range: {power_A.min():.1f}-{power_A.max():.1f}W")

        eval_A = evaluate_reservoir(
            'A_physical', states_A, u, y_narma, power_A,
            washout, train_end, args.max_delay, mlp_device
        )
        eval_A['time_s'] = time_A
        results['conditions']['A_physical'] = eval_A

        print("  Cooling down (30s)...")
        time.sleep(30)

    # ------------------------------------------------------------------
    # Condition B: Echo State Network
    # ------------------------------------------------------------------
    print("\n[3/5] Condition B: Echo State Network")
    esn = EchoStateNetwork(
        input_dim=1, reservoir_size=args.esn_size,
        spectral_radius=0.95, input_scaling=0.1, leak_rate=0.3,
        seed=args.seed,
    )
    esn.reset()
    states_B = esn.run(u)
    eval_B = evaluate_reservoir(
        'B_esn', states_B, u, y_narma, None,
        washout, train_end, args.max_delay, mlp_device
    )
    eval_B['time_s'] = 0.0
    results['conditions']['B_esn'] = eval_B

    # ------------------------------------------------------------------
    # Condition C: Linear delay embedding
    # ------------------------------------------------------------------
    print("\n[4/5] Condition C: Linear delay embedding")
    max_embed = 28
    states_C = np.zeros((T, max_embed))
    for d in range(max_embed):
        states_C[d:, d] = u[:T - d]
    eval_C = evaluate_reservoir(
        'C_linear', states_C, u, y_narma, None,
        washout, train_end, args.max_delay, mlp_device
    )
    eval_C['time_s'] = 0.0
    results['conditions']['C_linear'] = eval_C

    # ------------------------------------------------------------------
    # Condition D: Physical with shuffled inputs (control)
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[5/5] Condition D: Physical GPU, shuffled inputs")
        rng = np.random.RandomState(args.seed + 1)
        u_shuf = u.copy()
        rng.shuffle(u_shuf)
        t0 = time.time()
        states_D, power_D = run_physical_reservoir(
            u_shuf, dt=dt, min_size=args.min_size, max_size=args.max_size
        )
        time_D = time.time() - t0

        eval_D = evaluate_reservoir(
            'D_shuffled', states_D, u_shuf, y_narma, power_D,
            washout, train_end, args.max_delay, mlp_device
        )
        eval_D['time_s'] = time_D
        results['conditions']['D_shuffled'] = eval_D

    # ------------------------------------------------------------------
    # Analysis & Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Summary table
    header = f"{'Condition':15s} | {'NARMA10(Ridge)':>14s} | {'NARMA10(MLP)':>12s} | " \
             f"{'Parity(Ridge)':>13s} | {'Parity(MLP)':>11s} | {'MC':>5s}"
    print(header)
    print("-" * len(header))

    for name, cond in results['conditions'].items():
        n10r = cond.get('narma10', {}).get('ridge_nrmse', -1)
        n10m = cond.get('narma10', {}).get('mlp_nrmse', -1)
        pr = cond.get('temporal_parity', {}).get('ridge_accuracy', -1)
        pm = cond.get('temporal_parity', {}).get('mlp_accuracy', -1)
        mc = cond.get('memory_capacity', 0)
        print(f"{name:15s} | {n10r:14.4f} | {n10m:12.4f} | {pr:13.3f} | {pm:11.3f} | {mc:5.2f}")

    # Verdict logic
    verdict_parts = []

    # Test 1: Nonlinear gain on parity task for physical reservoir
    if 'A_physical' in results['conditions']:
        phys = results['conditions']['A_physical']
        parity_gain = phys['temporal_parity']['nonlinear_gain']
        if parity_gain > 0.05:
            verdict_parts.append(
                f"PASS: Physical parity MLP-Ridge gain = {parity_gain:+.3f} (nonlinear computation detected)")
        else:
            verdict_parts.append(
                f"FAIL: Physical parity MLP-Ridge gain = {parity_gain:+.3f} (no nonlinear advantage)")

        # Test 2: Physical parity should beat chance (0.5)
        phys_parity_mlp = phys['temporal_parity']['mlp_accuracy']
        if phys_parity_mlp > 0.55:
            verdict_parts.append(
                f"PASS: Physical parity MLP = {phys_parity_mlp:.3f} > 0.55 (above chance)")
        else:
            verdict_parts.append(
                f"FAIL: Physical parity MLP = {phys_parity_mlp:.3f} <= 0.55")

    # Test 3: ESN vs Physical on nonlinear tasks
    if 'A_physical' in results['conditions'] and 'B_esn' in results['conditions']:
        phys_parity = results['conditions']['A_physical']['temporal_parity']['mlp_accuracy']
        esn_parity = results['conditions']['B_esn']['temporal_parity']['mlp_accuracy']
        if phys_parity > esn_parity - 0.05:
            verdict_parts.append(
                f"PASS: Physical parity ({phys_parity:.3f}) competitive with ESN ({esn_parity:.3f})")
        else:
            verdict_parts.append(
                f"FAIL: Physical parity ({phys_parity:.3f}) << ESN ({esn_parity:.3f})")

    # Test 4: Self-prediction (physical should have an advantage)
    if 'A_physical' in results['conditions']:
        sp = results['conditions']['A_physical'].get('self_prediction', {})
        if isinstance(sp, dict) and 'k=1' in sp:
            sp_nrmse = sp['k=1']['ridge_nrmse']
            if sp_nrmse < 0.8:
                verdict_parts.append(
                    f"PASS: Self-prediction NRMSE(k=1) = {sp_nrmse:.4f} < 0.8")
            else:
                verdict_parts.append(
                    f"FAIL: Self-prediction NRMSE(k=1) = {sp_nrmse:.4f} >= 0.8")

    # Test 5: Shuffled control should be worse on parity
    if 'A_physical' in results['conditions'] and 'D_shuffled' in results['conditions']:
        phys_p = results['conditions']['A_physical']['temporal_parity']['mlp_accuracy']
        shuf_p = results['conditions']['D_shuffled']['temporal_parity']['mlp_accuracy']
        if phys_p > shuf_p:
            verdict_parts.append(
                f"PASS: Physical parity ({phys_p:.3f}) > Shuffled ({shuf_p:.3f})")
        else:
            verdict_parts.append(
                f"FAIL: Physical parity ({phys_p:.3f}) <= Shuffled ({shuf_p:.3f})")

    n_pass = sum(1 for v in verdict_parts if v.startswith('PASS'))
    n_total = len(verdict_parts)

    verdict = "NONLINEAR COMPUTATION PROVEN" if n_pass >= 4 else \
              "PARTIAL EVIDENCE" if n_pass >= 2 else "NOT PROVEN"

    results['verdict'] = verdict
    results['verdict_details'] = verdict_parts
    results['tests_passed'] = f"{n_pass}/{n_total}"

    print()
    for v in verdict_parts:
        print(f"  {v}")
    print(f"\n  VERDICT: {verdict} ({n_pass}/{n_total} tests passed)")

    # Save
    out_path = Path(__file__).parent.parent / 'results' / 'z1602_nonlinear_readout.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == '__main__':
    main()
