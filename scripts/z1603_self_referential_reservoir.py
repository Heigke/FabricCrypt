#!/usr/bin/env python3
"""
z1603 - Self-Referential Thermal Reservoir: GPU Predicts Its Own Dynamics
=========================================================================

Novel contribution: A physical reservoir computing system where the TARGET
is the reservoir's own future state. This creates a genuine self-referential
(ouroboros) loop: computation → heat → reservoir state → prediction of heat.

No software reservoir can replicate this because the prediction task is
intrinsically tied to the physical substrate. The physical reservoir has
PRIVILEGED ACCESS to its own internal state (thermal inertia, DVFS controller
state, power regulation dynamics) that is invisible to any external observer.

Hypothesis: The physical GPU reservoir achieves lower self-prediction error
than an ESN or linear model with access to the SAME input sequence, because
the reservoir state vector implicitly encodes unobserved internal dynamics.

Protocol:
  Phase 1: Drive GPU with structured input (sum of sinusoids + noise)
           Collect state trajectory: [temp, power, freq, busy, kernel_time]
  Phase 2: Train readouts to predict power(t+k) from state(t)
           for k=1,2,3,5,10 (prediction horizons from 1s to 10s at dt=1)
  Phase 3: Compare physical self-prediction vs ESN-based prediction vs linear

Key metric: Physical self-prediction NRMSE should be LOWER than ESN prediction
for short horizons (k=1,2,3) where thermal inertia provides information.

Usage:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z1603_self_referential_reservoir.py
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
    if _use_mock:
        return np.array([50.0, 20.0, 2.0, 0.5, 3.0]) + np.random.randn(5) * 0.5
    s = _telemetry.read_sample()
    return np.array([
        s.temp_edge_c,
        s.power_w,
        s.freq_sclk_mhz / 1000.0,
        s.gpu_busy_pct / 100.0,
        0.0,  # placeholder for kernel_time, filled by caller
    ])


# ---------------------------------------------------------------------------
# Input signal: structured (predictable component + noise)
# ---------------------------------------------------------------------------
def generate_structured_input(T, seed=42):
    """
    Input with predictable temporal structure: sum of slow sinusoids + noise.
    This gives the reservoir something to LEARN about future dynamics.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(T, dtype=float)

    # Slow oscillations (periods of 20, 50, 100 steps)
    signal = 0.15 * np.sin(2 * np.pi * t / 20)
    signal += 0.10 * np.sin(2 * np.pi * t / 50 + 0.5)
    signal += 0.05 * np.sin(2 * np.pi * t / 100 + 1.0)

    # Add noise
    signal += rng.uniform(-0.05, 0.05, T)

    # Shift to [0, 0.5] range
    signal = signal - signal.min()
    signal = signal / (signal.max() + 1e-10) * 0.5

    return signal


def generate_random_input(T, seed=42):
    """Pure random input (no temporal structure). Control condition."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 0.5, T)


# ---------------------------------------------------------------------------
# Physical Reservoir
# ---------------------------------------------------------------------------
def run_physical_reservoir(u, dt=1.0, min_size=128, max_size=3072,
                           device_str='cuda'):
    T = len(u)
    state_dim = 5
    states = np.zeros((T, state_dim))

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
        gpu_state[4] = kernel_time * 1000  # kernel time in ms
        states[t] = gpu_state

        if t % 100 == 0:
            print(f"    step {t}/{T}: temp={gpu_state[0]:.1f}°C, "
                  f"power={gpu_state[1]:.1f}W, freq={gpu_state[2]*1000:.0f}MHz, "
                  f"size={size}, kernel={kernel_time*1000:.1f}ms")

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
# Readout
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


def compute_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    sigma = np.std(y_true)
    return rmse / sigma if sigma > 1e-10 else rmse


def bootstrap_nrmse(y_true, y_pred, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        vals.append(compute_nrmse(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, (1 - ci) / 2 * 100)), \
           float(np.percentile(vals, (1 + ci) / 2 * 100))


# ---------------------------------------------------------------------------
# Self-prediction evaluation
# ---------------------------------------------------------------------------
def evaluate_self_prediction(name, states_raw, u, power_trace,
                             washout, train_end, horizons=(1, 2, 3, 5, 10)):
    """
    Evaluate how well a reservoir can predict the GPU's power at t+k.

    Returns results dict with NRMSE for each prediction horizon.
    """
    print(f"\n  Self-prediction: {name}")

    # Prepare states
    aug = augment_with_delays(states_raw, delays=(1, 2, 3))
    normed, _, _ = normalize_states(aug)
    T = normed.shape[0]
    states = np.hstack([normed, np.ones((T, 1))])

    results = {'reservoir': name, 'state_dim': states.shape[1]}
    horizon_results = {}

    for k in horizons:
        if k >= T - train_end:
            print(f"    k={k}: skipped (too few test samples)")
            continue

        # Target: power at t+k
        target = np.zeros(T)
        target[:T - k] = power_trace[k:]

        # Train/test split (avoid edge effects)
        valid_train_end = train_end - k
        valid_test_end = T - k

        if valid_train_end <= washout + 10:
            continue

        X_train = states[washout:valid_train_end]
        y_train = target[washout:valid_train_end]
        X_test = states[train_end:valid_test_end]
        y_test = target[train_end:valid_test_end]

        if len(y_test) < 10 or np.std(y_test) < 0.01:
            print(f"    k={k}: skipped (insufficient data or zero variance)")
            continue

        # Ridge prediction
        y_pred, alpha = ridge_fit_predict(X_train, y_train, X_test)
        nrmse = compute_nrmse(y_test, y_pred)
        ci_lo, ci_hi = bootstrap_nrmse(y_test, y_pred)

        # R² (explained variance)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

        # Correlation
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        corr = float(corr) if not np.isnan(corr) else 0.0

        horizon_results[f'k={k}'] = {
            'nrmse': float(nrmse),
            'nrmse_ci': [ci_lo, ci_hi],
            'r2': float(r2),
            'correlation': corr,
            'best_alpha': float(alpha),
            'n_test': len(y_test),
            'target_std': float(np.std(y_test)),
            'target_mean': float(np.mean(y_test)),
        }
        print(f"    k={k:2d}: NRMSE={nrmse:.4f} [{ci_lo:.4f},{ci_hi:.4f}], "
              f"R²={r2:.4f}, corr={corr:.4f}")

    results['horizons'] = horizon_results
    return results


# ---------------------------------------------------------------------------
# Transfer function analysis
# ---------------------------------------------------------------------------
def analyze_transfer_function(states, u):
    """Analyze the GPU's input-output transfer function."""
    T = states.shape[0]
    results = {}

    # Cross-correlation between input and each state feature
    for j in range(states.shape[1]):
        feat_name = ['temp', 'power', 'freq', 'busy', 'kernel_ms'][j]
        # Compute cross-correlation for different lags
        max_lag = 15
        cross_corr = []
        for lag in range(max_lag):
            if lag < T:
                c = np.corrcoef(u[:T - lag], states[lag:, j])[0, 1]
                cross_corr.append(float(c) if not np.isnan(c) else 0.0)
            else:
                cross_corr.append(0.0)

        # Peak lag (where cross-correlation is maximum)
        peak_lag = int(np.argmax(np.abs(cross_corr)))

        results[feat_name] = {
            'cross_correlation': cross_corr,
            'peak_lag': peak_lag,
            'peak_corr': float(cross_corr[peak_lag]),
            'variance': float(np.var(states[:, j])),
            'mean': float(np.mean(states[:, j])),
        }

    # Power spectrum of state features
    for j in range(min(3, states.shape[1])):
        feat_name = ['temp', 'power', 'freq'][j]
        fft = np.fft.rfft(states[:, j] - np.mean(states[:, j]))
        psd = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(T)
        # Dominant frequency
        dom_idx = np.argmax(psd[1:]) + 1  # skip DC
        results[feat_name]['dominant_freq'] = float(freqs[dom_idx])
        results[feat_name]['psd_peak'] = float(psd[dom_idx])

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='z1603: Self-Referential Thermal Reservoir')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--min-size', type=int, default=128)
    parser.add_argument('--max-size', type=int, default=3072)
    parser.add_argument('--esn-size', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-physical', action='store_true')
    args = parser.parse_args()

    T = args.steps
    dt = args.dt
    washout = int(T * 0.15)
    train_end = int(T * 0.75)
    test_size = T - train_end

    print("=" * 70)
    print("z1603: Self-Referential Thermal Reservoir")
    print("=" * 70)
    print(f"  Steps: {T}, dt: {dt}s, Total: {T*dt/60:.1f} min per condition")
    print(f"  Washout: {washout}, Train: {train_end - washout}, Test: {test_size}")
    print(f"  Prediction horizons: k=1,2,3,5,10 steps ({dt}s per step)")
    print(f"  Input: structured sinusoid + noise (predictable dynamics)")
    print()

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    has_telemetry = init_telemetry()
    print()

    # Generate structured input
    print("[1/5] Generating structured input...")
    u_struct = generate_structured_input(T, seed=args.seed)
    u_random = generate_random_input(T, seed=args.seed + 100)
    print(f"  Structured input: range [{u_struct.min():.3f}, {u_struct.max():.3f}], "
          f"std={np.std(u_struct):.4f}")
    print(f"  Random input: range [{u_random.min():.3f}, {u_random.max():.3f}], "
          f"std={np.std(u_random):.4f}")

    horizons = (1, 2, 3, 5, 10)

    results = {
        'experiment': 'z1603_self_referential_reservoir',
        'timestamp': datetime.datetime.now().isoformat(),
        'hypothesis': 'Physical GPU reservoir achieves lower self-prediction error '
                      'than software ESN because it has privileged access to internal '
                      'thermal/electrical dynamics invisible to external observers.',
        'hardware': {
            'gpu': torch.cuda.get_device_name(0) if has_cuda else 'CPU',
            'telemetry': 'sysfs_hwmon (real)' if has_telemetry else 'mock',
        },
        'config': {
            'steps': T, 'dt_s': dt, 'washout': washout,
            'train_size': train_end - washout, 'test_size': test_size,
            'horizons': list(horizons),
            'input_type': 'structured_sinusoid',
        },
        'conditions': {},
    }

    # ------------------------------------------------------------------
    # Condition A: Physical reservoir, structured input
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[2/5] Condition A: Physical GPU, structured input")
        t0 = time.time()
        states_A = run_physical_reservoir(
            u_struct, dt=dt, min_size=args.min_size, max_size=args.max_size
        )
        time_A = time.time() - t0
        power_trace_A = states_A[:, 1].copy()  # power is column 1

        print(f"  Physical run: {time_A:.1f}s")
        print(f"  Temp range: {states_A[:, 0].min():.1f}-{states_A[:, 0].max():.1f}°C")
        print(f"  Power range: {power_trace_A.min():.1f}-{power_trace_A.max():.1f}W")
        print(f"  Power std: {np.std(power_trace_A):.2f}W")

        eval_A = evaluate_self_prediction(
            'A_physical_structured', states_A, u_struct, power_trace_A,
            washout, train_end, horizons
        )
        eval_A['time_s'] = time_A

        # Transfer function analysis
        tf_A = analyze_transfer_function(states_A, u_struct)
        eval_A['transfer_function'] = tf_A
        print(f"\n  Transfer function:")
        for feat, info in tf_A.items():
            if 'peak_lag' in info:
                print(f"    {feat}: peak_lag={info['peak_lag']}*{dt}s, "
                      f"peak_corr={info['peak_corr']:.3f}, var={info['variance']:.3f}")

        results['conditions']['A_physical_structured'] = eval_A

        print("  Cooling down (30s)...")
        time.sleep(30)

    # ------------------------------------------------------------------
    # Condition B: Physical reservoir, random input (control)
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[3/5] Condition B: Physical GPU, random input")
        t0 = time.time()
        states_B = run_physical_reservoir(
            u_random, dt=dt, min_size=args.min_size, max_size=args.max_size
        )
        time_B = time.time() - t0
        power_trace_B = states_B[:, 1].copy()

        print(f"  Power std: {np.std(power_trace_B):.2f}W")

        eval_B = evaluate_self_prediction(
            'B_physical_random', states_B, u_random, power_trace_B,
            washout, train_end, horizons
        )
        eval_B['time_s'] = time_B
        results['conditions']['B_physical_random'] = eval_B

        print("  Cooling down (30s)...")
        time.sleep(30)

    # ------------------------------------------------------------------
    # Condition C: ESN predicting physical power (uses real power trace from A)
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[4/5] Condition C: ESN predicting GPU power")
        print("  (ESN driven by same input, but predicts REAL GPU power trace)")

        esn = EchoStateNetwork(
            input_dim=1, reservoir_size=args.esn_size,
            spectral_radius=0.95, input_scaling=0.1, leak_rate=0.3,
            seed=args.seed,
        )
        esn.reset()
        states_C = esn.run(u_struct)

        eval_C = evaluate_self_prediction(
            'C_esn_predicting_gpu', states_C, u_struct, power_trace_A,
            washout, train_end, horizons
        )
        eval_C['time_s'] = 0.0
        results['conditions']['C_esn_predicting_gpu'] = eval_C

    # ------------------------------------------------------------------
    # Condition D: Linear delay embedding predicting physical power
    # ------------------------------------------------------------------
    if not args.skip_physical:
        print("\n[5/5] Condition D: Linear delay embedding predicting GPU power")
        max_embed = 20
        states_D = np.zeros((T, max_embed))
        for d in range(max_embed):
            states_D[d:, d] = u_struct[:T - d]

        eval_D = evaluate_self_prediction(
            'D_linear_predicting_gpu', states_D, u_struct, power_trace_A,
            washout, train_end, horizons
        )
        eval_D['time_s'] = 0.0
        results['conditions']['D_linear_predicting_gpu'] = eval_D

    # ------------------------------------------------------------------
    # Comparison and Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SELF-PREDICTION COMPARISON")
    print("=" * 70)

    # Build comparison table
    header = f"{'Horizon':>8s}"
    cond_names = list(results['conditions'].keys())
    for name in cond_names:
        short = name.split('_')[0] + '_' + '_'.join(name.split('_')[1:3])
        header += f" | {short:>18s}"
    print(header)
    print("-" * len(header))

    for k in horizons:
        key = f'k={k}'
        row = f"{'t+' + str(k):>8s}"
        for name in cond_names:
            horizs = results['conditions'][name].get('horizons', {})
            if key in horizs:
                nrmse = horizs[key]['nrmse']
                row += f" | {nrmse:>18.4f}"
            else:
                row += f" | {'---':>18s}"
        print(row)

    # Verdict
    verdict_parts = []

    if 'A_physical_structured' in results['conditions'] and \
       'C_esn_predicting_gpu' in results['conditions']:

        phys_h = results['conditions']['A_physical_structured'].get('horizons', {})
        esn_h = results['conditions']['C_esn_predicting_gpu'].get('horizons', {})

        wins = 0
        total = 0
        for k in [1, 2, 3]:
            key = f'k={k}'
            if key in phys_h and key in esn_h:
                total += 1
                p_nrmse = phys_h[key]['nrmse']
                e_nrmse = esn_h[key]['nrmse']
                if p_nrmse < e_nrmse:
                    wins += 1
                    verdict_parts.append(
                        f"PASS: Physical self-pred(k={k}) {p_nrmse:.4f} < ESN {e_nrmse:.4f}")
                else:
                    verdict_parts.append(
                        f"FAIL: Physical self-pred(k={k}) {p_nrmse:.4f} >= ESN {e_nrmse:.4f}")

        if total > 0:
            if wins == total:
                verdict_parts.append(
                    f"PASS: Physical wins {wins}/{total} short-horizon predictions")
            elif wins > total // 2:
                verdict_parts.append(
                    f"PARTIAL: Physical wins {wins}/{total} short-horizon predictions")
            else:
                verdict_parts.append(
                    f"FAIL: Physical wins only {wins}/{total} short-horizon predictions")

    # Check if structured input gives better self-prediction than random
    if 'A_physical_structured' in results['conditions'] and \
       'B_physical_random' in results['conditions']:
        phys_s = results['conditions']['A_physical_structured'].get('horizons', {})
        phys_r = results['conditions']['B_physical_random'].get('horizons', {})

        if 'k=1' in phys_s and 'k=1' in phys_r:
            s_nrmse = phys_s['k=1']['nrmse']
            r_nrmse = phys_r['k=1']['nrmse']
            if s_nrmse < r_nrmse:
                verdict_parts.append(
                    f"PASS: Structured input self-pred ({s_nrmse:.4f}) < "
                    f"Random ({r_nrmse:.4f}) — input structure matters")
            else:
                verdict_parts.append(
                    f"FAIL: Structured input self-pred ({s_nrmse:.4f}) >= "
                    f"Random ({r_nrmse:.4f})")

    # Check physical reservoir has good absolute self-prediction
    if 'A_physical_structured' in results['conditions']:
        phys_h = results['conditions']['A_physical_structured'].get('horizons', {})
        if 'k=1' in phys_h:
            nrmse_1 = phys_h['k=1']['nrmse']
            r2_1 = phys_h['k=1']['r2']
            if r2_1 > 0.3:
                verdict_parts.append(
                    f"PASS: Self-prediction R²(k=1) = {r2_1:.4f} > 0.3")
            else:
                verdict_parts.append(
                    f"FAIL: Self-prediction R²(k=1) = {r2_1:.4f} <= 0.3")

    n_pass = sum(1 for v in verdict_parts if v.startswith('PASS'))
    n_total = len(verdict_parts)

    verdict = "SELF-REFERENTIAL PREDICTION PROVEN" if n_pass >= n_total - 1 and n_total >= 3 else \
              "PARTIAL EVIDENCE" if n_pass >= 2 else "NOT PROVEN"

    results['verdict'] = verdict
    results['verdict_details'] = verdict_parts
    results['tests_passed'] = f"{n_pass}/{n_total}"

    print()
    for v in verdict_parts:
        print(f"  {v}")
    print(f"\n  VERDICT: {verdict} ({n_pass}/{n_total} tests passed)")

    # Save
    out_path = Path(__file__).parent.parent / 'results' / 'z1603_self_referential.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == '__main__':
    main()
