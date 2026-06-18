#!/usr/bin/env python3
"""
z1604 - High Duty Cycle Thermal Reservoir Computing
====================================================

Addresses the key limitation of z1600-z1603: at dt=1.0s with single matmul,
the GPU duty cycle is only ~0.5%, creating negligible thermal variation.

This experiment MAXIMIZES thermal variation by:
1. Using dt=0.1s (10Hz sampling — fast enough for thermal dynamics)
2. Running MULTIPLE matmuls per step to fill the time interval
3. Input u(t) controls BOTH matmul size AND number of iterations
4. This pushes GPU duty cycle to 50-90%, creating real temperature swings

Expected improvement:
- Temperature variation: ~3°C (was ~1°C in z1602)
- Power variation: ~30W range (was ~13W range)
- Intrinsic memory: should increase due to genuine thermal inertia
- Self-prediction: should improve with larger power/temp signal

Also computes intrinsic MC (without delay augmentation) to separate
artificial from genuine thermal memory.

Usage:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z1604_high_duty_thermal_rc.py
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
# Telemetry
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
    if _use_mock:
        return np.array([50.0, 60.0, 2.0, 0.8, 3.0]) + np.random.randn(5) * 1.0
    s = _telemetry.read_sample()
    return np.array([
        s.temp_edge_c,
        s.power_w,
        s.freq_sclk_mhz / 1000.0,
        s.gpu_busy_pct / 100.0,
        0.0,  # kernel_time filled by caller
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
    T = len(u)
    bits = (u > 0.25).astype(float)
    targets = np.zeros(T)
    for t in range(delay, T):
        targets[t] = float(int(bits[t]) ^ int(bits[t - delay]))
    return targets


# ---------------------------------------------------------------------------
# HIGH DUTY CYCLE Physical GPU Reservoir
# ---------------------------------------------------------------------------
def run_high_duty_reservoir(u, dt=0.1, min_size=256, max_size=2048,
                            min_iters=1, max_iters=8, device_str='cuda'):
    """
    Drive GPU at HIGH duty cycle by running MULTIPLE matmuls per time step.

    Input encoding:
      u(t) in [0, 0.5] maps to:
        - matmul size: [min_size, max_size]
        - iteration count: [min_iters, max_iters]
      Combined: u=0 gives small*1, u=0.5 gives large*8

    At dt=0.1s with max 8 iterations of 2048x2048 matmul:
      Each matmul takes ~5ms, so 8*5=40ms active out of 100ms = 40% duty cycle.
      Plus GPU doesn't fully cool in 60ms idle, creating thermal buildup.
    """
    T = len(u)
    state_dim = 5  # temp, power, freq, busy, kernel_time
    states = np.zeros((T, state_dim))

    device = torch.device(device_str)

    # Warmup to operating temperature
    print(f"  Warming up GPU (heavy, 10 iterations)...")
    for _ in range(10):
        A = torch.randn(2048, 2048, device=device)
        B = torch.randn(2048, 2048, device=device)
        for _ in range(5):
            _ = torch.mm(A, B)
        torch.cuda.synchronize()
    del A, B
    time.sleep(2.0)

    s = read_gpu_state()
    print(f"  After warmup: temp={s[0]:.1f}°C, power={s[1]:.1f}W")

    est_min = T * dt / 60
    print(f"  Running: {T} steps, dt={dt}s (~{est_min:.1f} min)")

    for t in range(T):
        step_start = time.monotonic()

        # Encode input: both size and iteration count scale with u
        u_norm = u[t] / 0.5  # [0, 1]
        size = int(min_size + (max_size - min_size) * u_norm)
        size = max(min_size, min(max_size, size))
        n_iters = int(min_iters + (max_iters - min_iters) * u_norm)
        n_iters = max(min_iters, min(max_iters, n_iters))

        # Run multiple matmuls for high duty cycle
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        for _ in range(n_iters):
            C = torch.mm(A, B)
            # Prevent compiler optimization
            A = C
        torch.cuda.synchronize()
        del A, B, C

        kernel_time = time.monotonic() - step_start

        # Wait for remainder (may be very short or negative)
        elapsed = time.monotonic() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

        gpu_state = read_gpu_state()
        gpu_state[4] = kernel_time * 1000  # ms
        states[t] = gpu_state

        if t % 200 == 0:
            duty = kernel_time / dt * 100
            print(f"    step {t}/{T}: temp={gpu_state[0]:.1f}°C, "
                  f"power={gpu_state[1]:.1f}W, freq={gpu_state[2]*1000:.0f}MHz, "
                  f"size={size}x{n_iters}, kernel={kernel_time*1000:.1f}ms, "
                  f"duty={duty:.0f}%")

    return states


# ---------------------------------------------------------------------------
# ESN
# ---------------------------------------------------------------------------
class EchoStateNetwork:
    def __init__(self, input_dim=1, reservoir_size=50,
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
# Readouts
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
# Metrics
# ---------------------------------------------------------------------------
def compute_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    sigma = np.std(y_true)
    return rmse / sigma if sigma > 1e-10 else rmse


def compute_accuracy(y_true, y_pred, threshold=0.5):
    return np.mean((y_pred > threshold).astype(float) == y_true)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = [metric_fn(y_true[rng.choice(n, n, replace=True)],
                      y_pred[rng.choice(n, n, replace=True)]) for _ in range(n_boot)]
    return float(np.percentile(vals, (1 - ci) / 2 * 100)), \
           float(np.percentile(vals, (1 + ci) / 2 * 100))


def compute_memory_capacity(states, u, washout, train_end, max_delay=30):
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
# Evaluate reservoir on multiple tasks
# ---------------------------------------------------------------------------
def evaluate_full(name, states_raw, u, y_narma, washout, train_end,
                  max_delay=30, device='cpu'):
    """Evaluate with both augmented and raw states."""
    print(f"\n  Evaluating {name}...")
    T = states_raw.shape[0]

    results = {'reservoir': name}

    # ---- Augmented states (standard, with delays 1,2,3) ----
    aug = augment_with_delays(states_raw, delays=(1, 2, 3))
    normed_aug, _, _ = normalize_states(aug)
    states_aug = np.hstack([normed_aug, np.ones((T, 1))])

    X_train_aug = states_aug[washout:train_end]
    X_test_aug = states_aug[train_end:]

    # ---- Raw states (NO delay augmentation — tests intrinsic memory) ----
    normed_raw, _, _ = normalize_states(states_raw)
    states_raw_n = np.hstack([normed_raw, np.ones((T, 1))])

    X_train_raw = states_raw_n[washout:train_end]
    X_test_raw = states_raw_n[train_end:]

    # NARMA-10
    y_tr = y_narma[washout:train_end]
    y_te = y_narma[train_end:]

    pred_aug, _ = ridge_fit_predict(X_train_aug, y_tr, X_test_aug)
    pred_raw, _ = ridge_fit_predict(X_train_raw, y_tr, X_test_raw)
    pred_mlp, _ = mlp_fit_predict(X_train_aug, y_tr, X_test_aug,
                                  hidden_dim=64, epochs=300, device=device)

    results['narma10'] = {
        'ridge_augmented_nrmse': float(compute_nrmse(y_te, pred_aug)),
        'ridge_raw_nrmse': float(compute_nrmse(y_te, pred_raw)),
        'mlp_augmented_nrmse': float(compute_nrmse(y_te, pred_mlp)),
    }
    print(f"    NARMA-10: Ridge(aug)={results['narma10']['ridge_augmented_nrmse']:.4f}, "
          f"Ridge(raw)={results['narma10']['ridge_raw_nrmse']:.4f}, "
          f"MLP(aug)={results['narma10']['mlp_augmented_nrmse']:.4f}")

    # Temporal Parity
    y_parity = generate_temporal_parity(u, delay=1)
    y_tr_p = y_parity[washout:train_end]
    y_te_p = y_parity[train_end:]

    pred_ridge_p, _ = ridge_fit_predict(X_train_aug, y_tr_p, X_test_aug)
    pred_mlp_p, _ = mlp_fit_predict(X_train_aug, y_tr_p, X_test_aug,
                                    hidden_dim=64, epochs=300, device=device)
    pred_raw_p, _ = ridge_fit_predict(X_train_raw, y_tr_p, X_test_raw)
    pred_mlp_raw_p, _ = mlp_fit_predict(X_train_raw, y_tr_p, X_test_raw,
                                        hidden_dim=64, epochs=300, device=device)

    results['parity'] = {
        'ridge_augmented_acc': float(compute_accuracy(y_te_p, pred_ridge_p)),
        'mlp_augmented_acc': float(compute_accuracy(y_te_p, pred_mlp_p)),
        'ridge_raw_acc': float(compute_accuracy(y_te_p, pred_raw_p)),
        'mlp_raw_acc': float(compute_accuracy(y_te_p, pred_mlp_raw_p)),
    }
    print(f"    Parity:   Ridge(aug)={results['parity']['ridge_augmented_acc']:.3f}, "
          f"MLP(aug)={results['parity']['mlp_augmented_acc']:.3f}, "
          f"Ridge(raw)={results['parity']['ridge_raw_acc']:.3f}, "
          f"MLP(raw)={results['parity']['mlp_raw_acc']:.3f}")

    # Self-prediction (power at t+k)
    power_trace = states_raw[:, 1]
    if np.std(power_trace) > 0.5:  # need meaningful variation
        sp_results = {}
        for k in [1, 2, 3, 5, 10]:
            if k >= T - train_end:
                continue
            target = np.zeros(T)
            target[:T - k] = power_trace[k:]
            y_tr_s = target[washout:train_end - k]
            y_te_s = target[train_end:T - k]
            X_tr_s = states_aug[washout:washout + len(y_tr_s)]
            X_te_s = states_aug[train_end:train_end + len(y_te_s)]
            if len(y_te_s) < 10 or np.std(y_te_s) < 0.1:
                continue
            pred_s, _ = ridge_fit_predict(X_tr_s, y_tr_s, X_te_s)
            nrmse_s = compute_nrmse(y_te_s, pred_s)
            r2 = 1 - np.sum((y_te_s - pred_s)**2) / np.sum((y_te_s - y_te_s.mean())**2)
            sp_results[f'k={k}'] = {
                'nrmse': float(nrmse_s), 'r2': float(r2),
                'target_std': float(np.std(y_te_s))
            }
            print(f"    Self t+{k}: NRMSE={nrmse_s:.4f}, R²={r2:.4f}")
        results['self_prediction'] = sp_results
    else:
        print(f"    Self-pred: skipped (power std={np.std(power_trace):.2f}W too low)")
        results['self_prediction'] = 'skipped'

    # Memory Capacity: BOTH augmented and raw
    mc_aug, mc_aug_per = compute_memory_capacity(states_aug, u, washout, train_end, max_delay)
    mc_raw, mc_raw_per = compute_memory_capacity(states_raw_n, u, washout, train_end, max_delay)

    results['memory_capacity'] = {
        'augmented': float(mc_aug),
        'raw_intrinsic': float(mc_raw),
        'augmented_per_delay': [float(x) for x in mc_aug_per],
        'raw_per_delay': [float(x) for x in mc_raw_per],
    }
    print(f"    MC: augmented={mc_aug:.2f}, raw(intrinsic)={mc_raw:.2f}")

    # State statistics
    results['state_stats'] = {
        'temp_range': [float(states_raw[:, 0].min()), float(states_raw[:, 0].max())],
        'temp_std': float(np.std(states_raw[:, 0])),
        'power_range': [float(states_raw[:, 1].min()), float(states_raw[:, 1].max())],
        'power_std': float(np.std(states_raw[:, 1])),
        'freq_range': [float(states_raw[:, 2].min() * 1000),
                       float(states_raw[:, 2].max() * 1000)],
        'kernel_ms_range': [float(states_raw[:, 4].min()), float(states_raw[:, 4].max())],
        'duty_cycle_pct': float(np.mean(states_raw[:, 4]) / (1000 * 0.1) * 100)
        if states_raw[:, 4].mean() > 0 else 0.0,
    }

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='z1604: High Duty Cycle Thermal RC')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Total steps (default: 2000)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step seconds (default: 0.1)')
    parser.add_argument('--min-size', type=int, default=256)
    parser.add_argument('--max-size', type=int, default=2048)
    parser.add_argument('--min-iters', type=int, default=1)
    parser.add_argument('--max-iters', type=int, default=8)
    parser.add_argument('--esn-size', type=int, default=50)
    parser.add_argument('--max-delay', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-physical', action='store_true')
    args = parser.parse_args()

    T = args.steps
    dt = args.dt
    washout = int(T * 0.15)
    train_end = int(T * 0.75)
    test_size = T - train_end

    print("=" * 70)
    print("z1604: High Duty Cycle Thermal Reservoir Computing")
    print("=" * 70)
    print(f"  Steps: {T}, dt: {dt}s, Total: {T * dt / 60:.1f} min per condition")
    print(f"  Washout: {washout}, Train: {train_end - washout}, Test: {test_size}")
    print(f"  Matmul range: {args.min_size}-{args.max_size}, "
          f"Iters: {args.min_iters}-{args.max_iters}")
    print(f"  Expected duty cycle: 10-50% (vs <1% in z1600-z1603)")
    print(f"  Readouts: Ridge (aug+raw), MLP (aug)")
    print()

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    has_telemetry = init_telemetry()
    mlp_device = 'cuda' if has_cuda else 'cpu'
    print()

    # Generate signals
    print("[1/4] Generating signals...")
    u, y_narma = generate_narma10(T, seed=args.seed)
    print(f"  Input range: [{u.min():.3f}, {u.max():.3f}]")

    results = {
        'experiment': 'z1604_high_duty_thermal_rc',
        'timestamp': datetime.datetime.now().isoformat(),
        'hypothesis': 'Higher GPU duty cycle creates meaningful thermal variation, '
                      'improving intrinsic memory capacity and self-prediction ability. '
                      'Raw (non-augmented) MC should be significantly higher than in z1600-z1603.',
        'hardware': {
            'gpu': torch.cuda.get_device_name(0) if has_cuda else 'CPU',
            'telemetry': 'sysfs_hwmon (real)' if has_telemetry else 'mock',
        },
        'config': {
            'steps': T, 'dt_s': dt, 'washout': washout,
            'train_size': train_end - washout, 'test_size': test_size,
            'min_size': args.min_size, 'max_size': args.max_size,
            'min_iters': args.min_iters, 'max_iters': args.max_iters,
            'esn_size': args.esn_size, 'max_delay': args.max_delay,
        },
        'conditions': {},
    }

    # ------------------------------------------------------------------
    # Condition A: Physical GPU High-Duty Reservoir
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[2/4] Condition A: Physical GPU High-Duty Reservoir")
        t0 = time.time()
        states_A = run_high_duty_reservoir(
            u, dt=dt, min_size=args.min_size, max_size=args.max_size,
            min_iters=args.min_iters, max_iters=args.max_iters
        )
        time_A = time.time() - t0
        print(f"  Run time: {time_A:.1f}s")

        eval_A = evaluate_full(
            'A_physical_high_duty', states_A, u, y_narma,
            washout, train_end, args.max_delay, mlp_device
        )
        eval_A['time_s'] = time_A
        results['conditions']['A_physical_high_duty'] = eval_A

        # Cool down
        print("  Cooling down (30s)...")
        time.sleep(30)

    # ------------------------------------------------------------------
    # Condition B: ESN (matched)
    # ------------------------------------------------------------------
    print("\n[3/4] Condition B: Echo State Network")
    esn = EchoStateNetwork(
        input_dim=1, reservoir_size=args.esn_size,
        spectral_radius=0.95, input_scaling=0.1, leak_rate=0.3,
        seed=args.seed,
    )
    esn.reset()
    states_B = esn.run(u)
    eval_B = evaluate_full(
        'B_esn', states_B, u, y_narma,
        washout, train_end, args.max_delay, mlp_device
    )
    eval_B['time_s'] = 0.0
    results['conditions']['B_esn'] = eval_B

    # ------------------------------------------------------------------
    # Condition C: Linear delay embedding
    # ------------------------------------------------------------------
    print("\n[4/4] Condition C: Linear delay embedding")
    max_embed = 20
    states_C = np.zeros((T, max_embed))
    for d in range(max_embed):
        states_C[d:, d] = u[:T - d]
    eval_C = evaluate_full(
        'C_linear', states_C, u, y_narma,
        washout, train_end, args.max_delay, mlp_device
    )
    eval_C['time_s'] = 0.0
    results['conditions']['C_linear'] = eval_C

    # ------------------------------------------------------------------
    # Comparison and Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Print comparison table
    print(f"\n{'Condition':20s} | {'NARMA Ridge':>11s} | {'NARMA MLP':>9s} | "
          f"{'Parity MLP':>10s} | {'MC(aug)':>7s} | {'MC(raw)':>7s} | "
          f"{'Temp Std':>8s} | {'Pwr Std':>7s}")
    print("-" * 100)

    for name, cond in results['conditions'].items():
        n10 = cond.get('narma10', {}).get('ridge_augmented_nrmse', -1)
        n10m = cond.get('narma10', {}).get('mlp_augmented_nrmse', -1)
        p_mlp = cond.get('parity', {}).get('mlp_augmented_acc', -1)
        mc_a = cond.get('memory_capacity', {})
        mc_aug = mc_a.get('augmented', 0) if isinstance(mc_a, dict) else 0
        mc_raw = mc_a.get('raw_intrinsic', 0) if isinstance(mc_a, dict) else 0
        ss = cond.get('state_stats', {})
        t_std = ss.get('temp_std', 0)
        p_std = ss.get('power_std', 0)
        print(f"{name:20s} | {n10:11.4f} | {n10m:9.4f} | "
              f"{p_mlp:10.3f} | {mc_aug:7.2f} | {mc_raw:7.2f} | "
              f"{t_std:7.2f}°C | {p_std:6.2f}W")

    # Verdict
    verdict_parts = []

    if 'A_physical_high_duty' in results['conditions']:
        phys = results['conditions']['A_physical_high_duty']
        ss = phys.get('state_stats', {})

        # Test 1: Did we achieve higher thermal variation?
        t_std = ss.get('temp_std', 0)
        if t_std > 0.5:
            verdict_parts.append(f"PASS: Temp std = {t_std:.2f}°C > 0.5 (meaningful variation)")
        else:
            verdict_parts.append(f"FAIL: Temp std = {t_std:.2f}°C <= 0.5 (insufficient variation)")

        # Test 2: Intrinsic MC > 1.0
        mc_info = phys.get('memory_capacity', {})
        mc_raw = mc_info.get('raw_intrinsic', 0) if isinstance(mc_info, dict) else 0
        if mc_raw > 1.0:
            verdict_parts.append(f"PASS: Intrinsic MC = {mc_raw:.2f} > 1.0 (real thermal memory)")
        else:
            verdict_parts.append(f"FAIL: Intrinsic MC = {mc_raw:.2f} <= 1.0")

        # Test 3: Parity MLP > 0.7
        p_mlp = phys.get('parity', {}).get('mlp_augmented_acc', 0)
        if p_mlp > 0.7:
            verdict_parts.append(f"PASS: Parity MLP = {p_mlp:.3f} > 0.7")
        else:
            verdict_parts.append(f"FAIL: Parity MLP = {p_mlp:.3f} <= 0.7")

        # Test 4: Self-prediction works
        sp = phys.get('self_prediction', {})
        if isinstance(sp, dict) and 'k=1' in sp:
            sp_r2 = sp['k=1']['r2']
            if sp_r2 > 0.3:
                verdict_parts.append(f"PASS: Self-pred R²(k=1) = {sp_r2:.4f} > 0.3")
            else:
                verdict_parts.append(f"FAIL: Self-pred R²(k=1) = {sp_r2:.4f} <= 0.3")
        else:
            verdict_parts.append("FAIL: Self-prediction skipped")

        # Test 5: Higher MC(raw) than z1600-z1603 baseline (~0.3)
        if mc_raw > 0.5:
            verdict_parts.append(
                f"PASS: MC(raw) = {mc_raw:.2f} > 0.5 (improved over z1602's ~0.3)")
        else:
            verdict_parts.append(
                f"FAIL: MC(raw) = {mc_raw:.2f} <= 0.5 (no improvement)")

    n_pass = sum(1 for v in verdict_parts if v.startswith('PASS'))
    n_total = len(verdict_parts)
    verdict = "HIGH DUTY CYCLE IMPROVES THERMAL RC" if n_pass >= 4 else \
              "PARTIAL IMPROVEMENT" if n_pass >= 2 else "NO IMPROVEMENT"

    results['verdict'] = verdict
    results['verdict_details'] = verdict_parts
    results['tests_passed'] = f"{n_pass}/{n_total}"

    print()
    for v in verdict_parts:
        print(f"  {v}")
    print(f"\n  VERDICT: {verdict} ({n_pass}/{n_total} tests passed)")

    # Compare with z1600-z1603
    print("\n  Comparison with previous experiments:")
    print(f"    z1600 (dt=0.3s, low duty): MC(aug)=3.55, temp_std=~0")
    print(f"    z1602 (dt=1.0s, low duty): MC(aug)=3.31, temp_std=~1°C")
    if 'A_physical_high_duty' in results['conditions']:
        mc_info = results['conditions']['A_physical_high_duty'].get('memory_capacity', {})
        mc_a = mc_info.get('augmented', 0) if isinstance(mc_info, dict) else 0
        mc_r = mc_info.get('raw_intrinsic', 0) if isinstance(mc_info, dict) else 0
        t_s = results['conditions']['A_physical_high_duty'].get('state_stats', {}).get('temp_std', 0)
        print(f"    z1604 (dt={dt}s, HIGH duty): MC(aug)={mc_a:.2f}, "
              f"MC(raw)={mc_r:.2f}, temp_std={t_s:.2f}°C")

    # Save
    out_path = Path(__file__).parent.parent / 'results' / 'z1604_high_duty_thermal_rc.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == '__main__':
    main()
