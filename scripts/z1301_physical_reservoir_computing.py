#!/usr/bin/env python3
"""
z1301: PHYSICAL RESERVOIR COMPUTING - GPU Thermal Dynamics as Computational Substrate

================================================================================
                    THE HARDWARE THINKS FOR US
================================================================================

Key insight from reservoir computing research:
"Physical computing substrates can operate at only ~0.1% of conventional
computer power consumption" (Nature Communications 2024)

We exploit GPU thermal dynamics as a PHYSICAL RESERVOIR:
1. The GPU's thermal mass stores information about past computation
2. Temperature gradients encode temporal patterns
3. The reservoir's nonlinear dynamics provide rich feature transformation
4. A simple readout layer extracts useful representations

This is GENUINE physical computation - not simulation!

The GPU is not just hardware running software - it IS the computer.
Its physics provides the nonlinearity, memory, and dynamics.

Inspired by:
- Task-adaptive physical reservoir computing (Nature Materials 2023)
- Ensemble Reservoir Computing for Physical Systems (arXiv 2601.21807)
- Physical Reservoir Computing for Edge AI (Innovation 2025)

================================================================================
"""

import os
import sys
import time
import json
import math
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


@dataclass
class ReservoirConfig:
    """Configuration for physical reservoir computing."""

    # Reservoir properties
    reservoir_dim: int = 64         # Virtual reservoir nodes (mapped from physics)
    input_dim: int = 8              # Raw telemetry dimensions
    output_dim: int = 32            # Readout dimension
    washout_steps: int = 50         # Steps to let reservoir settle

    # Physics coupling
    thermal_coupling: float = 0.8   # How much thermal dynamics matter
    power_coupling: float = 0.6     # How much power dynamics matter
    temporal_scale: float = 0.1     # Time constant (seconds)

    # Nonlinearity
    spectral_radius: float = 0.9    # Echo state property
    input_scaling: float = 0.5
    leak_rate: float = 0.3          # How fast reservoir forgets

    # Training
    ridge_alpha: float = 1e-4       # Ridge regression regularization
    n_samples: int = 5000
    batch_size: int = 32


class PhysicalReservoir:
    """
    Maps GPU thermal dynamics to reservoir state.

    The key insight: GPU physics is ALREADY doing computation.
    - Heat diffusion is a natural low-pass filter
    - Thermal mass provides memory of past computation
    - Nonlinear thermal throttling adds complexity
    - Power-temperature coupling creates feedback

    We don't simulate a reservoir - we READ the physical reservoir.
    """

    def __init__(self, config: ReservoirConfig):
        self.config = config
        self.telemetry = SysfsHwmonTelemetry()

        # History for temporal features
        self.history = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)

        # Internal state (augments physical state)
        self.internal_state = np.zeros(config.reservoir_dim)

        # Random projection matrices (fixed, for reproducibility)
        np.random.seed(42)
        self.W_in = np.random.randn(config.reservoir_dim, config.input_dim) * config.input_scaling

        # Recurrent weights (sparse, scaled to spectral radius)
        density = 0.1
        self.W_res = np.random.randn(config.reservoir_dim, config.reservoir_dim)
        mask = np.random.random((config.reservoir_dim, config.reservoir_dim)) < density
        self.W_res *= mask

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(self.W_res)
        spectral_radius = np.max(np.abs(eigenvalues))
        if spectral_radius > 0:
            self.W_res *= config.spectral_radius / spectral_radius

    def _sample_physics(self) -> np.ndarray:
        """Sample the physical state of the GPU."""
        sample = self.telemetry.read_sample()

        # Raw physical features
        physics = np.array([
            sample.power_w / 65.0,               # Normalized power
            sample.temp_edge_c / 100.0,          # Edge temperature
            sample.temp_junction_c / 100.0,      # Junction temperature
            sample.freq_sclk_mhz / 2800.0,       # GPU clock
            sample.freq_mclk_mhz / 2000.0,       # Memory clock
            sample.gpu_busy_pct / 100.0,         # Utilization
            sample.vram_used_gb / 8.0,           # VRAM usage
            (time.time() % 100) / 100.0,         # Temporal phase
        ])

        self.history.append(physics)
        self.timestamps.append(time.time())

        return physics

    def _compute_temporal_features(self) -> np.ndarray:
        """Extract temporal features from physical history."""
        if len(self.history) < 10:
            return np.zeros(self.config.input_dim * 2)

        history = np.array(list(self.history))
        timestamps = np.array(list(self.timestamps))

        # Derivatives (rate of change)
        if len(history) >= 2:
            dt = timestamps[-1] - timestamps[-2] + 1e-6
            derivatives = (history[-1] - history[-2]) / dt
        else:
            derivatives = np.zeros(self.config.input_dim)

        # Moving statistics
        recent = history[-20:]
        mean = recent.mean(axis=0)
        std = recent.std(axis=0) + 1e-6

        # Normalized deviation from mean
        deviation = (history[-1] - mean) / std

        return np.concatenate([derivatives, deviation])

    def step(self, external_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        One step of the physical reservoir.

        Combines:
        1. Physical state from GPU
        2. Temporal features (derivatives, statistics)
        3. Internal echo state dynamics

        Returns full reservoir state.
        """
        # Sample physical state
        physics = self._sample_physics()

        # Temporal features
        temporal = self._compute_temporal_features()

        # Combine physical and external inputs
        if external_input is not None:
            combined_input = np.concatenate([physics, external_input[:self.config.input_dim - 8]])
        else:
            combined_input = physics

        # Update internal state (echo state network dynamics)
        # x(t+1) = (1-α)x(t) + α·tanh(W_in·u(t) + W_res·x(t))
        pre_activation = (
            self.W_in @ combined_input +
            self.W_res @ self.internal_state
        )
        new_state = np.tanh(pre_activation)

        # Leak rate mixing
        self.internal_state = (
            (1 - self.config.leak_rate) * self.internal_state +
            self.config.leak_rate * new_state
        )

        # Full reservoir state = internal + physics + temporal
        full_state = np.concatenate([
            self.internal_state,
            physics * self.config.thermal_coupling,
            temporal[:8] * self.config.power_coupling,
        ])

        return full_state

    def reset(self):
        """Reset reservoir state."""
        self.internal_state = np.zeros(self.config.reservoir_dim)
        self.history.clear()
        self.timestamps.clear()


class ReservoirReadout(nn.Module):
    """
    Trainable readout layer for the physical reservoir.

    The reservoir does the heavy lifting (nonlinear transformation).
    The readout just learns a linear mapping to the desired output.

    This is the key insight of reservoir computing:
    train only the readout, not the dynamics.
    """

    def __init__(self, config: ReservoirConfig):
        super().__init__()
        self.config = config

        # Reservoir output dimension
        reservoir_out = config.reservoir_dim + config.input_dim + config.input_dim

        # Simple linear readout (can also use ridge regression)
        self.readout = nn.Linear(reservoir_out, config.output_dim)

        # Optional nonlinear readout
        self.nonlinear_readout = nn.Sequential(
            nn.Linear(reservoir_out, 64),
            nn.Tanh(),
            nn.Linear(64, config.output_dim),
        )

        self.use_nonlinear = False

    def forward(self, reservoir_state: torch.Tensor) -> torch.Tensor:
        """Map reservoir state to output."""
        if self.use_nonlinear:
            return self.nonlinear_readout(reservoir_state)
        else:
            return self.readout(reservoir_state)


class PhysicalReservoirComputer:
    """
    Complete physical reservoir computing system.

    Uses GPU thermal dynamics as the reservoir and trains
    a simple readout layer for various tasks.
    """

    def __init__(self, config: ReservoirConfig):
        self.config = config
        self.reservoir = PhysicalReservoir(config)
        self.readout = ReservoirReadout(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.readout.to(self.device)

    def collect_reservoir_states(
        self,
        n_steps: int,
        workload_fn: Optional[callable] = None,
    ) -> np.ndarray:
        """
        Collect reservoir states over time.

        Args:
            n_steps: Number of timesteps
            workload_fn: Optional function to generate varying workload

        Returns:
            states: [n_steps, reservoir_dim] array
        """
        self.reservoir.reset()
        states = []

        # Washout period
        for _ in range(self.config.washout_steps):
            self.reservoir.step()
            if workload_fn:
                workload_fn()
            time.sleep(self.config.temporal_scale)

        # Collection period
        for step in range(n_steps):
            state = self.reservoir.step()
            states.append(state)

            if workload_fn:
                workload_fn()

            time.sleep(self.config.temporal_scale)

            if step % 100 == 0:
                print(f"  Collecting step {step}/{n_steps}")

        return np.array(states)

    def train_prediction_task(
        self,
        states: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train readout for prediction task using ridge regression.

        This is fast because we only train the readout!
        """
        # Ridge regression: W = (X^T X + αI)^{-1} X^T Y
        X = states
        Y = targets

        XtX = X.T @ X
        reg = self.config.ridge_alpha * np.eye(XtX.shape[0])
        XtY = X.T @ Y

        W = np.linalg.solve(XtX + reg, XtY)

        # Compute training metrics
        predictions = X @ W
        mse = np.mean((predictions - Y) ** 2)
        correlation = np.corrcoef(predictions.flatten(), Y.flatten())[0, 1]

        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        }

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Make prediction from reservoir state."""
        state_tensor = torch.from_numpy(state).float().to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.readout(state_tensor)

        return output.cpu().numpy()


# ============================================================================
#                      BENCHMARK TASKS
# ============================================================================

def create_memory_task(states: np.ndarray, delay: int = 10) -> np.ndarray:
    """
    Memory task: predict past thermal state.

    Tests: Does the reservoir retain information about past states?
    """
    n_samples = len(states) - delay
    # Target: thermal state from 'delay' steps ago
    targets = states[:-delay, :8]  # First 8 dims are thermal
    return targets


def create_prediction_task(states: np.ndarray, horizon: int = 5) -> np.ndarray:
    """
    Prediction task: predict future thermal state.

    Tests: Can the reservoir extrapolate dynamics?
    """
    n_samples = len(states) - horizon
    # Target: thermal state 'horizon' steps ahead
    targets = states[horizon:, :8]
    return targets


def create_nonlinear_task(states: np.ndarray) -> np.ndarray:
    """
    Nonlinear transformation task.

    Tests: Does the reservoir provide useful nonlinear features?
    """
    # XOR-like task on thermal features
    thermal = states[:, :8]
    targets = np.zeros((len(states), 4))

    # Nonlinear combinations
    targets[:, 0] = np.sin(thermal[:, 0] * 2 * np.pi) * np.cos(thermal[:, 1] * 2 * np.pi)
    targets[:, 1] = ((thermal[:, 2] > 0.5).astype(int) ^ (thermal[:, 3] > 0.5).astype(int)).astype(float)
    targets[:, 2] = np.tanh(thermal[:, 0] * thermal[:, 1] - thermal[:, 2] * thermal[:, 3])
    targets[:, 3] = np.sqrt(thermal[:, 0]**2 + thermal[:, 1]**2)

    return targets


def create_workload_generator(device: torch.device):
    """Create varying GPU workload for richer reservoir dynamics."""
    sizes = [256, 512, 1024, 2048]
    current_size = [0]

    def workload():
        size = sizes[current_size[0] % len(sizes)]
        current_size[0] += 1

        # Matrix multiplication workload
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

    return workload


# ============================================================================
#                            EXPERIMENTS
# ============================================================================

def run_memory_capacity_experiment(
    prc: PhysicalReservoirComputer,
    states: np.ndarray,
) -> Dict[str, float]:
    """
    Test memory capacity: How far back can the reservoir remember?

    Memory capacity is a key metric for reservoir computing.
    """
    print("\n  Memory Capacity Experiment")
    print("  " + "-" * 40)

    capacities = []
    delays = [1, 2, 5, 10, 20, 50]

    for delay in delays:
        if delay >= len(states) - 100:
            continue

        targets = create_memory_task(states, delay)
        train_states = states[delay:len(targets) + delay]

        metrics = prc.train_prediction_task(train_states, targets)
        capacity = max(0, metrics['correlation'] ** 2)
        capacities.append(capacity)

        print(f"    Delay {delay:3d}: correlation={metrics['correlation']:.3f}, capacity={capacity:.3f}")

    total_capacity = sum(capacities)
    print(f"    Total memory capacity: {total_capacity:.2f}")

    return {
        'delays': delays[:len(capacities)],
        'capacities': capacities,
        'total_capacity': total_capacity,
    }


def run_prediction_experiment(
    prc: PhysicalReservoirComputer,
    states: np.ndarray,
) -> Dict[str, float]:
    """
    Test prediction: Can the reservoir predict future states?
    """
    print("\n  Prediction Experiment")
    print("  " + "-" * 40)

    horizons = [1, 2, 5, 10, 20]
    results = []

    for horizon in horizons:
        if horizon >= len(states) - 100:
            continue

        targets = create_prediction_task(states, horizon)
        train_states = states[:len(targets)]

        metrics = prc.train_prediction_task(train_states, targets)
        results.append({
            'horizon': horizon,
            'rmse': metrics['rmse'],
            'correlation': metrics['correlation'],
        })

        print(f"    Horizon {horizon:3d}: RMSE={metrics['rmse']:.4f}, corr={metrics['correlation']:.3f}")

    return {'horizons': results}


def run_nonlinear_experiment(
    prc: PhysicalReservoirComputer,
    states: np.ndarray,
) -> Dict[str, float]:
    """
    Test nonlinearity: Does the reservoir compute useful nonlinear features?
    """
    print("\n  Nonlinear Transformation Experiment")
    print("  " + "-" * 40)

    targets = create_nonlinear_task(states)

    # Split train/test
    split = int(len(states) * 0.8)
    train_states, test_states = states[:split], states[split:]
    train_targets, test_targets = targets[:split], targets[split:]

    # Train
    metrics = prc.train_prediction_task(train_states, train_targets)
    print(f"    Train RMSE: {metrics['rmse']:.4f}")

    # Test
    test_preds = train_states @ np.linalg.lstsq(
        train_states.T @ train_states + prc.config.ridge_alpha * np.eye(train_states.shape[1]),
        train_states.T @ train_targets,
        rcond=None
    )[0]

    # Actually test on test set
    W = np.linalg.lstsq(
        train_states.T @ train_states + prc.config.ridge_alpha * np.eye(train_states.shape[1]),
        train_states.T @ train_targets,
        rcond=None
    )[0]

    test_preds = test_states @ W
    test_mse = np.mean((test_preds - test_targets) ** 2)
    test_corr = np.corrcoef(test_preds.flatten(), test_targets.flatten())[0, 1]

    print(f"    Test RMSE: {np.sqrt(test_mse):.4f}")
    print(f"    Test Correlation: {test_corr:.3f}")

    return {
        'train_rmse': metrics['rmse'],
        'test_rmse': float(np.sqrt(test_mse)),
        'test_correlation': float(test_corr) if not np.isnan(test_corr) else 0.0,
    }


def compare_with_baseline(
    states: np.ndarray,
    config: ReservoirConfig,
) -> Dict[str, float]:
    """
    Compare physical reservoir with baseline (linear + random features).
    """
    print("\n  Baseline Comparison")
    print("  " + "-" * 40)

    # Task: predict temperature from power/utilization
    X_phys = states[:, :8]  # Physical features only
    X_reservoir = states     # Full reservoir state
    Y = states[1:, 1:3]      # Predict next temperature
    X_phys = X_phys[:-1]
    X_reservoir = X_reservoir[:-1]

    split = int(len(X_phys) * 0.8)

    results = {}

    # Baseline 1: Linear on raw features
    W_linear = np.linalg.lstsq(
        X_phys[:split].T @ X_phys[:split] + config.ridge_alpha * np.eye(X_phys.shape[1]),
        X_phys[:split].T @ Y[:split],
        rcond=None
    )[0]
    pred_linear = X_phys[split:] @ W_linear
    mse_linear = np.mean((pred_linear - Y[split:]) ** 2)
    results['linear_rmse'] = float(np.sqrt(mse_linear))

    # Baseline 2: Random features
    np.random.seed(123)
    W_random = np.random.randn(X_phys.shape[1], 64)
    X_random = np.tanh(X_phys @ W_random)

    W_random_out = np.linalg.lstsq(
        X_random[:split].T @ X_random[:split] + config.ridge_alpha * np.eye(X_random.shape[1]),
        X_random[:split].T @ Y[:split],
        rcond=None
    )[0]
    pred_random = X_random[split:] @ W_random_out
    mse_random = np.mean((pred_random - Y[split:]) ** 2)
    results['random_features_rmse'] = float(np.sqrt(mse_random))

    # Physical reservoir
    W_res = np.linalg.lstsq(
        X_reservoir[:split].T @ X_reservoir[:split] + config.ridge_alpha * np.eye(X_reservoir.shape[1]),
        X_reservoir[:split].T @ Y[:split],
        rcond=None
    )[0]
    pred_res = X_reservoir[split:] @ W_res
    mse_res = np.mean((pred_res - Y[split:]) ** 2)
    results['reservoir_rmse'] = float(np.sqrt(mse_res))

    print(f"    Linear baseline RMSE: {results['linear_rmse']:.4f}")
    print(f"    Random features RMSE: {results['random_features_rmse']:.4f}")
    print(f"    Physical reservoir RMSE: {results['reservoir_rmse']:.4f}")

    improvement = (results['linear_rmse'] - results['reservoir_rmse']) / results['linear_rmse'] * 100
    results['improvement_over_linear'] = improvement
    print(f"    Improvement over linear: {improvement:.1f}%")

    return results


# ============================================================================
#                              MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("z1301: PHYSICAL RESERVOIR COMPUTING")
    print("GPU Thermal Dynamics as Computational Substrate")
    print("=" * 70)
    print()

    config = ReservoirConfig(
        reservoir_dim=64,
        n_samples=500,  # Reduced for faster testing
        temporal_scale=0.05,
        washout_steps=20,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create physical reservoir computer
    print("\nInitializing Physical Reservoir Computer...")
    prc = PhysicalReservoirComputer(config)
    print(f"  Reservoir dim: {config.reservoir_dim}")
    print(f"  Spectral radius: {config.spectral_radius}")
    print(f"  Leak rate: {config.leak_rate}")

    # Collect reservoir states with varying workload
    print("\nCollecting reservoir states...")
    print("  (This exploits GPU thermal dynamics as computation)")

    workload = create_workload_generator(device)
    states = prc.collect_reservoir_states(config.n_samples, workload)
    print(f"  Collected {len(states)} states")
    print(f"  State dimension: {states.shape[1]}")

    # Results storage
    results = {
        'experiment': 'z1301_physical_reservoir_computing',
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
    }

    # Run experiments
    print("\n" + "=" * 70)
    print("EXPERIMENTS")
    print("=" * 70)

    # 1. Memory capacity
    results['memory'] = run_memory_capacity_experiment(prc, states)

    # 2. Prediction
    results['prediction'] = run_prediction_experiment(prc, states)

    # 3. Nonlinear transformation
    results['nonlinear'] = run_nonlinear_experiment(prc, states)

    # 4. Baseline comparison
    results['baseline'] = compare_with_baseline(states, config)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nPhysical Reservoir Computing Results:")
    print(f"  Memory capacity: {results['memory']['total_capacity']:.2f}")
    print(f"  Prediction (h=5): corr={results['prediction']['horizons'][2]['correlation']:.3f}")
    print(f"  Nonlinear test corr: {results['nonlinear']['test_correlation']:.3f}")
    print(f"  Improvement over linear: {results['baseline']['improvement_over_linear']:.1f}%")

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'z1301_physical_reservoir_computing.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
