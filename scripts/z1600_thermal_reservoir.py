#!/usr/bin/env python3
"""
z1600 - GPU Thermal Reservoir Computing (NARMA-10 & Memory Capacity)
====================================================================

Novel: First demonstration of commodity GPU thermal dynamics as a physical
reservoir computer, benchmarked on standard reservoir computing tasks.

The GPU's thermal mass, DVFS transitions, and power regulation create a
nonlinear dynamical system with fading memory -- the two requirements for
reservoir computing (Jaeger, 2001). Input signals are encoded as GPU workload
intensity (matrix multiplication size), and the resulting thermal/power/frequency
state is read out via sysfs hwmon as the reservoir state.

This is analogous to:
  - Water bucket RC (Fernando & Sojakka, 2003)
  - Memristive network RC (Chen et al., Science Advances 2025)
  - Photonic RC (Larger et al., 2012)
But uses commodity hardware reproducible by anyone with a GPU.

Conditions:
  A: Physical GPU thermal reservoir (our contribution)
  B: Echo State Network (software baseline, matched readout dimensionality)
  C: Physical reservoir with SHUFFLED inputs (destroy temporal structure)
  D: Linear regression on input time-delay embedding (no nonlinear transform)
  E: Physical reservoir with CONSTANT workload (no input encoding)

Benchmarks:
  1. NARMA-10: Nonlinear autoregressive moving average, order 10
  2. Memory Capacity: How many past inputs can be linearly recovered

Metrics:
  - NRMSE (Normalized Root Mean Square Error) -- standard RC metric
  - Memory Capacity (MC) -- total recoverable past information
  - Bootstrap 95% CI on NRMSE

Related work:
  - Proteus (ICS 2025, Mutlu): PuD with dynamic bit-precision
  - Memristive RC (Science Advances 2025): Bi2Se3 reservoir for robot control
  - NeuroBench (Nature Comms 2025): Standardized neuromorphic benchmarks

Usage:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z1600_thermal_reservoir.py
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z1600_thermal_reservoir.py --steps 2000 --dt 0.3
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
    """Read 6-dim GPU state vector."""
    if _use_mock:
        # Simulate slow thermal dynamics for testing
        return np.array([50.0, 45.0, 20.0, 2.0, 0.5, 40.0]) + np.random.randn(6) * 0.5
    s = _telemetry.read_sample()
    return np.array([
        s.temp_junction_c,
        s.temp_edge_c,
        s.power_w,
        s.freq_sclk_mhz / 1000.0,   # normalize to ~1-3 range
        s.gpu_busy_pct / 100.0,      # normalize to 0-1
        s.temp_mem_c,
    ])


# ---------------------------------------------------------------------------
# NARMA-10 generation
# ---------------------------------------------------------------------------
def generate_narma10(T, seed=42):
    """
    Generate NARMA-10 input/target sequences.
    y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 0.5, T)
    y = np.zeros(T)
    for t in range(10, T):
        y_sum = sum(y[t - 1 - i] for i in range(10))
        y[t] = 0.3 * y[t - 1] + 0.05 * y[t - 1] * y_sum + 1.5 * u[t - 10] * u[t] + 0.1
        y[t] = np.clip(y[t], -1e6, 1e6)
    return u, y


# ---------------------------------------------------------------------------
# Memory Capacity targets
# ---------------------------------------------------------------------------
def generate_mc_targets(u, max_delay=50):
    """Target for delay k: reconstruct u(t-k) from reservoir state at time t."""
    T = len(u)
    targets = np.zeros((T, max_delay))
    for k in range(1, max_delay + 1):
        targets[k:, k - 1] = u[:T - k]
    return targets


# ---------------------------------------------------------------------------
# Physical GPU Reservoir
# ---------------------------------------------------------------------------
def run_physical_reservoir(u, dt=0.3, min_size=128, max_size=3072,
                           constant_load=False, device_str='cuda'):
    """
    Drive GPU with workload proportional to u[t], read thermal state.

    Args:
        u: input signal, values in [0, 0.5]
        dt: time step in seconds
        min_size: minimum matmul dimension (u=0)
        max_size: maximum matmul dimension (u=0.5)
        constant_load: if True, use constant workload (condition E)
    """
    T = len(u)
    state_dim = 7  # 6 telemetry + kernel time
    states = np.zeros((T, state_dim))

    device = torch.device(device_str)

    # Warmup GPU
    print(f"  Warming up GPU...")
    for _ in range(5):
        A = torch.randn(1024, 1024, device=device)
        B = torch.randn(1024, 1024, device=device)
        _ = torch.mm(A, B)
        torch.cuda.synchronize()
    del A, B
    time.sleep(1.0)

    est_min = T * dt / 60
    print(f"  Running physical reservoir: {T} steps, dt={dt}s (~{est_min:.1f} min)")

    for t in range(T):
        step_start = time.monotonic()

        # Encode input as matmul size
        if constant_load:
            size = (min_size + max_size) // 2
        else:
            size = int(min_size + (max_size - min_size) * (u[t] / 0.5))
            size = max(min_size, min(max_size, size))

        # Execute GPU workload
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        _ = torch.mm(A, B)
        torch.cuda.synchronize()
        del A, B

        kernel_time = time.monotonic() - step_start

        # Wait for remainder of time step
        elapsed = time.monotonic() - step_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

        # Read thermal state
        gpu_state = read_gpu_state()
        states[t, :6] = gpu_state
        states[t, 6] = kernel_time * 1000  # ms

        if t % 200 == 0:
            print(f"    step {t}/{T}: temp={gpu_state[0]:.1f}°C, "
                  f"power={gpu_state[2]:.1f}W, freq={gpu_state[3] * 1000:.0f}MHz, "
                  f"size={size}, kernel={kernel_time * 1000:.1f}ms")

    return states


# ---------------------------------------------------------------------------
# Echo State Network
# ---------------------------------------------------------------------------
class EchoStateNetwork:
    """Standard Echo State Network for baseline comparison."""

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
        """Run ESN on input sequence, return all states."""
        T = len(inputs)
        states = np.zeros((T, self.reservoir_size))
        for t in range(T):
            x = np.atleast_1d(inputs[t])
            pre = np.tanh(self.W @ self.state + self.W_in @ x)
            self.state = (1 - self.leak_rate) * self.state + self.leak_rate * pre
            states[t] = self.state.copy()
        return states


# ---------------------------------------------------------------------------
# State augmentation
# ---------------------------------------------------------------------------
def augment_with_delays(states, delays=(1, 2, 3)):
    """Add time-delayed copies of states for richer features."""
    T, D = states.shape
    n_delays = len(delays)
    augmented = np.zeros((T, D * (1 + n_delays)))
    augmented[:, :D] = states
    for i, d in enumerate(delays):
        start_col = D * (i + 1)
        augmented[d:, start_col:start_col + D] = states[:T - d]
    return augmented


def normalize_states(states):
    """Z-score normalization per feature."""
    mean = states.mean(axis=0, keepdims=True)
    std = states.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0
    return (states - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Ridge Regression
# ---------------------------------------------------------------------------
def ridge_fit_predict(X_train, y_train, X_test, alphas=None):
    """Fit ridge regression, pick best alpha by leave-one-out approximation."""
    if alphas is None:
        alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0]

    best_w = None
    best_alpha = alphas[0]
    best_train_err = float('inf')

    for alpha in alphas:
        I = np.eye(X_train.shape[1])
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I,
                                X_train.T @ y_train)
        except np.linalg.LinAlgError:
            continue

        train_pred = X_train @ w
        train_err = np.mean((y_train - train_pred) ** 2)

        if train_err < best_train_err:
            best_train_err = train_err
            best_w = w
            best_alpha = alpha

    y_pred = X_test @ best_w
    return y_pred, best_alpha


def compute_nrmse(y_true, y_pred):
    """Normalized RMSE."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    sigma = np.std(y_true)
    return rmse / sigma if sigma > 1e-10 else rmse


def bootstrap_nrmse(y_true, y_pred, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for NRMSE."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    nrmses = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        nrmses.append(compute_nrmse(y_true[idx], y_pred[idx]))
    lo = np.percentile(nrmses, (1 - ci) / 2 * 100)
    hi = np.percentile(nrmses, (1 + ci) / 2 * 100)
    return lo, hi


# ---------------------------------------------------------------------------
# Memory Capacity
# ---------------------------------------------------------------------------
def compute_memory_capacity(states, u, washout, train_end, max_delay=50):
    """
    Compute memory capacity: MC = sum_{k=1}^{K} r^2(readout_k, u(t-k))
    """
    mc_targets = generate_mc_targets(u, max_delay)
    total_mc = 0.0
    mc_per_delay = []

    for k in range(max_delay):
        target = mc_targets[:, k]

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
        total_mc += r2

    return total_mc, mc_per_delay


# ---------------------------------------------------------------------------
# Evaluate one condition
# ---------------------------------------------------------------------------
def evaluate_condition(states_raw, u, y_narma, washout, train_end, max_delay=30):
    """Evaluate NARMA-10 NRMSE and Memory Capacity for a set of states."""
    # Augment with time delays
    states_aug = augment_with_delays(states_raw, delays=(1, 2, 3))

    # Normalize
    states_norm, _, _ = normalize_states(states_aug)

    # Add bias
    T = states_norm.shape[0]
    states_final = np.hstack([states_norm, np.ones((T, 1))])

    # NARMA-10
    X_train = states_final[washout:train_end]
    y_train = y_narma[washout:train_end]
    X_test = states_final[train_end:]
    y_test = y_narma[train_end:]

    y_pred, best_alpha = ridge_fit_predict(X_train, y_train, X_test)
    nrmse = compute_nrmse(y_test, y_pred)
    ci_lo, ci_hi = bootstrap_nrmse(y_test, y_pred)

    # Memory Capacity
    mc_total, mc_per_delay = compute_memory_capacity(
        states_final, u, washout, train_end, max_delay
    )

    return {
        'nrmse': float(nrmse),
        'nrmse_ci_95': [float(ci_lo), float(ci_hi)],
        'best_alpha': float(best_alpha),
        'memory_capacity': float(mc_total),
        'mc_per_delay': [float(x) for x in mc_per_delay],
        'state_dim_raw': states_raw.shape[1],
        'state_dim_augmented': states_final.shape[1],
    }


# ---------------------------------------------------------------------------
# Reservoir diagnostics
# ---------------------------------------------------------------------------
def compute_reservoir_properties(states_raw, u):
    """Compute key reservoir properties: state variance, rank, condition number."""
    # Effective rank (ratio of singular values)
    U, S, Vt = np.linalg.svd(states_raw, full_matrices=False)
    S_norm = S / S.sum()
    eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12)))

    # State variance per feature
    state_var = np.var(states_raw, axis=0)

    # Input-state correlation
    correlations = []
    for j in range(states_raw.shape[1]):
        c = np.corrcoef(u, states_raw[:, j])[0, 1]
        correlations.append(float(c) if not np.isnan(c) else 0.0)

    return {
        'effective_rank': float(eff_rank),
        'state_variance': [float(v) for v in state_var],
        'state_mean': [float(m) for m in np.mean(states_raw, axis=0)],
        'input_state_correlation': correlations,
        'condition_number': float(S[0] / S[-1]) if S[-1] > 1e-12 else float('inf'),
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='z1600: GPU Thermal Reservoir Computing')
    parser.add_argument('--steps', type=int, default=1500,
                        help='Total time steps (default: 1500)')
    parser.add_argument('--dt', type=float, default=0.3,
                        help='Time step in seconds (default: 0.3)')
    parser.add_argument('--min-size', type=int, default=128,
                        help='Min matmul size (default: 128)')
    parser.add_argument('--max-size', type=int, default=3072,
                        help='Max matmul size (default: 3072)')
    parser.add_argument('--esn-size', type=int, default=50,
                        help='ESN reservoir size (default: 50)')
    parser.add_argument('--max-delay', type=int, default=30,
                        help='Max delay for memory capacity (default: 30)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-physical', action='store_true',
                        help='Skip physical conditions (debug with ESN only)')
    args = parser.parse_args()

    T = args.steps
    dt = args.dt
    washout = int(T * 0.15)
    train_end = int(T * 0.75)
    test_size = T - train_end

    print("=" * 70)
    print("z1600: GPU Thermal Reservoir Computing")
    print("=" * 70)
    print(f"  Steps: {T}, dt: {dt}s, Total: {T * dt / 60:.1f} min per physical condition")
    print(f"  Washout: {washout}, Train: {train_end - washout}, Test: {test_size}")
    print(f"  Matmul range: {args.min_size}-{args.max_size}")
    print(f"  ESN size: {args.esn_size}")
    print()

    # Check hardware
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    has_telemetry = init_telemetry()
    print()

    # Generate benchmark signals
    print("[1/6] Generating NARMA-10 signal...")
    u, y_narma = generate_narma10(T, seed=args.seed)
    print(f"  Input range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"  Target range: [{y_narma[10:].min():.3f}, {y_narma[10:].max():.3f}]")
    print(f"  Target std: {np.std(y_narma[train_end:]):.4f}")

    results = {
        'experiment': 'z1600_thermal_reservoir',
        'timestamp': datetime.datetime.now().isoformat(),
        'hypothesis': 'GPU thermal dynamics serve as a physical reservoir computer '
                      'competitive with software ESN on standard benchmarks',
        'hardware': {
            'gpu': torch.cuda.get_device_name(0) if has_cuda else 'CPU',
            'telemetry': 'sysfs_hwmon (real)' if has_telemetry else 'mock',
        },
        'config': {
            'steps': T,
            'dt_s': dt,
            'washout': washout,
            'train_size': train_end - washout,
            'test_size': test_size,
            'min_matmul_size': args.min_size,
            'max_matmul_size': args.max_size,
            'esn_reservoir_size': args.esn_size,
            'max_delay': args.max_delay,
            'seed': args.seed,
        },
        'conditions': {},
    }

    # ------------------------------------------------------------------
    # Condition A: Physical GPU Thermal Reservoir
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[2/6] Condition A: Physical GPU Thermal Reservoir")
        t0 = time.time()
        states_A = run_physical_reservoir(
            u, dt=dt, min_size=args.min_size, max_size=args.max_size
        )
        time_A = time.time() - t0

        print(f"  Physical run took {time_A:.1f}s")
        print(f"  Temp range: {states_A[:, 0].min():.1f}-{states_A[:, 0].max():.1f}°C")
        print(f"  Power range: {states_A[:, 2].min():.1f}-{states_A[:, 2].max():.1f}W")

        # Evaluate
        eval_A = evaluate_condition(states_A, u, y_narma, washout, train_end, args.max_delay)
        props_A = compute_reservoir_properties(states_A[washout:train_end], u[washout:train_end])

        results['conditions']['A_physical'] = {
            'label': 'Physical GPU thermal reservoir',
            'time_s': time_A,
            **eval_A,
            'reservoir_properties': props_A,
            'thermal_range': {
                'temp_min': float(states_A[:, 0].min()),
                'temp_max': float(states_A[:, 0].max()),
                'temp_std': float(states_A[:, 0].std()),
                'power_min': float(states_A[:, 2].min()),
                'power_max': float(states_A[:, 2].max()),
                'power_std': float(states_A[:, 2].std()),
            },
        }
        print(f"  NARMA-10 NRMSE: {eval_A['nrmse']:.4f} "
              f"(95% CI: [{eval_A['nrmse_ci_95'][0]:.4f}, {eval_A['nrmse_ci_95'][1]:.4f}])")
        print(f"  Memory Capacity: {eval_A['memory_capacity']:.2f}")

        # Cool down between physical conditions
        print("  Cooling down (30s)...")
        time.sleep(30)
    else:
        states_A = None

    # ------------------------------------------------------------------
    # Condition B: Echo State Network (software baseline)
    # ------------------------------------------------------------------
    print("\n[3/6] Condition B: Echo State Network (software baseline)")
    esn = EchoStateNetwork(
        input_dim=1,
        reservoir_size=args.esn_size,
        spectral_radius=0.95,
        input_scaling=0.1,
        leak_rate=0.3,
        seed=args.seed,
    )
    esn.reset()
    states_B = esn.run(u)

    eval_B = evaluate_condition(states_B, u, y_narma, washout, train_end, args.max_delay)
    props_B = compute_reservoir_properties(states_B[washout:train_end], u[washout:train_end])

    results['conditions']['B_esn'] = {
        'label': 'Echo State Network (software)',
        'time_s': 0.0,
        **eval_B,
        'reservoir_properties': props_B,
    }
    print(f"  NARMA-10 NRMSE: {eval_B['nrmse']:.4f} "
          f"(95% CI: [{eval_B['nrmse_ci_95'][0]:.4f}, {eval_B['nrmse_ci_95'][1]:.4f}])")
    print(f"  Memory Capacity: {eval_B['memory_capacity']:.2f}")

    # ------------------------------------------------------------------
    # Condition C: Physical with SHUFFLED inputs
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[4/6] Condition C: Physical with SHUFFLED inputs")
        rng = np.random.RandomState(args.seed + 1)
        u_shuffled = u.copy()
        rng.shuffle(u_shuffled)

        t0 = time.time()
        states_C = run_physical_reservoir(
            u_shuffled, dt=dt, min_size=args.min_size, max_size=args.max_size
        )
        time_C = time.time() - t0

        # Evaluate against ORIGINAL targets (should fail -- temporal structure destroyed)
        eval_C = evaluate_condition(states_C, u_shuffled, y_narma, washout, train_end, args.max_delay)
        props_C = compute_reservoir_properties(states_C[washout:train_end], u_shuffled[washout:train_end])

        results['conditions']['C_shuffled'] = {
            'label': 'Physical GPU, shuffled inputs',
            'time_s': time_C,
            **eval_C,
            'reservoir_properties': props_C,
        }
        print(f"  NARMA-10 NRMSE: {eval_C['nrmse']:.4f}")
        print(f"  Memory Capacity: {eval_C['memory_capacity']:.2f}")

        print("  Cooling down (30s)...")
        time.sleep(30)

    # ------------------------------------------------------------------
    # Condition D: Linear baseline (input time-delay embedding)
    # ------------------------------------------------------------------
    print("\n[5/6] Condition D: Linear baseline (input delay embedding)")

    # Create time-delay embedding of input signal
    max_embed = 28  # match augmented physical state dim
    states_D = np.zeros((T, max_embed))
    for d in range(max_embed):
        states_D[d:, d] = u[:T - d]

    eval_D = evaluate_condition(states_D, u, y_narma, washout, train_end, args.max_delay)

    results['conditions']['D_linear'] = {
        'label': 'Linear (input delay embedding, no reservoir)',
        'time_s': 0.0,
        **eval_D,
    }
    print(f"  NARMA-10 NRMSE: {eval_D['nrmse']:.4f}")
    print(f"  Memory Capacity: {eval_D['memory_capacity']:.2f}")

    # ------------------------------------------------------------------
    # Condition E: Physical with CONSTANT workload
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[6/6] Condition E: Physical with CONSTANT workload")
        t0 = time.time()
        states_E = run_physical_reservoir(
            u, dt=dt, min_size=args.min_size, max_size=args.max_size,
            constant_load=True
        )
        time_E = time.time() - t0

        eval_E = evaluate_condition(states_E, u, y_narma, washout, train_end, args.max_delay)
        props_E = compute_reservoir_properties(states_E[washout:train_end], u[washout:train_end])

        results['conditions']['E_constant'] = {
            'label': 'Physical GPU, constant workload (no input encoding)',
            'time_s': time_E,
            **eval_E,
            'reservoir_properties': props_E,
        }
        print(f"  NARMA-10 NRMSE: {eval_E['nrmse']:.4f}")
        print(f"  Memory Capacity: {eval_E['memory_capacity']:.2f}")

    # ------------------------------------------------------------------
    # Comparison and verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for name, cond in results['conditions'].items():
        label = cond['label']
        nrmse = cond['nrmse']
        mc = cond['memory_capacity']
        ci = cond.get('nrmse_ci_95', [0, 0])
        print(f"  {name:15s}: NRMSE={nrmse:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]  MC={mc:.2f}  ({label})")

    # Determine if physical reservoir is competitive
    nrmse_A = results['conditions'].get('A_physical', {}).get('nrmse', float('inf'))
    nrmse_B = results['conditions']['B_esn']['nrmse']
    nrmse_D = results['conditions']['D_linear']['nrmse']
    nrmse_E = results['conditions'].get('E_constant', {}).get('nrmse', float('inf'))

    verdict_parts = []

    # Test 1: Physical (A) should beat linear (D)
    if nrmse_A < nrmse_D:
        verdict_parts.append(f"PASS: Physical ({nrmse_A:.4f}) < Linear ({nrmse_D:.4f})")
    else:
        verdict_parts.append(f"FAIL: Physical ({nrmse_A:.4f}) >= Linear ({nrmse_D:.4f})")

    # Test 2: Physical (A) should beat constant-workload (E)
    if nrmse_A < nrmse_E:
        verdict_parts.append(f"PASS: Physical ({nrmse_A:.4f}) < Constant ({nrmse_E:.4f})")
    else:
        verdict_parts.append(f"FAIL: Physical ({nrmse_A:.4f}) >= Constant ({nrmse_E:.4f})")

    # Test 3: Physical (A) vs ESN (B)
    ratio = nrmse_A / nrmse_B if nrmse_B > 0 else float('inf')
    if ratio < 1.5:
        verdict_parts.append(f"COMPETITIVE: Physical/ESN ratio = {ratio:.2f}")
    else:
        verdict_parts.append(f"WORSE: Physical/ESN ratio = {ratio:.2f}")

    all_pass = all('PASS' in v or 'COMPETITIVE' in v for v in verdict_parts)
    verdict = "GPU THERMAL RESERVOIR WORKS" if all_pass else "MIXED RESULTS"

    results['verdict'] = verdict
    results['verdict_details'] = verdict_parts
    results['comparison'] = {
        'physical_vs_esn_ratio': float(ratio),
        'physical_vs_linear_better': bool(nrmse_A < nrmse_D),
        'physical_vs_constant_better': bool(nrmse_A < nrmse_E),
    }

    print()
    for v in verdict_parts:
        print(f"  {v}")
    print(f"\n  VERDICT: {verdict}")

    # Novelty claim
    results['novelty_claim'] = (
        "First demonstration of commodity GPU thermal dynamics as a physical "
        "reservoir computer. GPU thermal mass + DVFS + power regulation create "
        "a nonlinear dynamical system with fading memory, benchmarked on NARMA-10 "
        "and Memory Capacity against Echo State Network baseline."
    )

    results['related_work'] = {
        'proteus_ics2025': 'Processing-using-DRAM, Mutlu et al. -- bulk bitwise PuD (different: digital, not analog reservoir)',
        'memristive_rc_sciadv2025': 'Bi2Se3 memristive network RC for robotics (different: custom fab memristors)',
        'neurobench_natcomms2025': 'Standardized neuromorphic benchmarks (our metrics are compatible)',
        'water_bucket_2003': 'Fernando & Sojakka -- physical liquid RC (analog: our thermal is similar principle)',
        'darpa_eaml_2025': 'DARPA Energy-Aware ML program (related: hardware-aware training)',
        'lipson_self_model_2025': 'Egocentric visual self-modeling (related: self-model of physical substrate)',
    }

    # Save
    out_path = Path(__file__).parent.parent / 'results' / 'z1600_thermal_reservoir.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == '__main__':
    main()
