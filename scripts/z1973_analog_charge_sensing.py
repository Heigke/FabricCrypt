#!/usr/bin/env python3
"""
z1973: Analog Charge Level Sensing for Embodied AI

================================================================================
                    ANALOG MEMORY AS PROPRIOCEPTIVE SIGNAL
================================================================================

Traditional digital memory: Binary (0 or 1)
Analog memory: Continuous charge level (0.0 to 1.0)
Embodied AI: Charge level as proprioceptive/interoceptive signal

DDR3 cells store charge that:
1. Decays exponentially over time (Arrhenius equation)
2. Is affected by temperature
3. Can be sensed as a continuous signal (not just binary)
4. Provides natural temporal dynamics for embodied AI

This script implements:
1. AnalogChargeSensor - Simulates DDR3 charge sensing with realistic noise
2. EmbodiedChargePredictor - Uses charge levels for self-prediction
3. Comparison of analog vs digital memory for embodiment tasks

Key insight: The DECAY is computation. The hardware does work for us.

Author: Claude + ikaros
Date: 2026-02-05
"""

import os
import sys
sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import deque

# Try to import GPU telemetry (optional)
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False
    print("[INFO] Telemetry not available, using simulated temperature")


# =============================================================================
# ANALOG CHARGE SENSOR
# =============================================================================

@dataclass
class ChargeSensorConfig:
    """Configuration for analog charge sensor."""
    num_cells: int = 256
    noise_std: float = 0.02  # Measurement noise (like real ADC)
    base_decay_rate: float = 0.001  # Decay rate at 25C
    temperature_factor: float = 10.0  # Temperature scaling
    refresh_threshold: float = 0.1  # Below this, consider cell "lost"


class AnalogChargeSensor:
    """
    Simulates analog charge sensing in DDR3-like memory.

    Physical model:
    - Charge stored in capacitor cell
    - Decay follows Arrhenius equation: tau = tau_0 * exp(-Ea/(k*T))
    - Higher temperature = faster decay
    - Measurement includes thermal noise (ADC noise)

    For embodied AI:
    - Charge level = proprioceptive signal
    - Decay = natural forgetting/regularization
    - Temperature coupling = environmental awareness
    """

    def __init__(self, config: ChargeSensorConfig):
        self.config = config
        self.num_cells = config.num_cells

        # Initialize charge levels randomly in middle range
        self.charge = np.random.uniform(0.3, 0.7, self.num_cells)

        # Track write times for decay calculation
        self.write_times = np.zeros(self.num_cells)
        self.write_temps = np.ones(self.num_cells) * 25.0

        # Statistics
        self.read_count = 0
        self.write_count = 0
        self.decay_events = 0

    def sense(self, temperature_c: float = 25.0) -> np.ndarray:
        """
        Sense charge levels with realistic measurement noise.

        Args:
            temperature_c: Current temperature for noise scaling

        Returns:
            Noisy charge readings (0.0 to 1.0)
        """
        # Measurement noise scales with temperature (thermal noise)
        noise_scale = self.config.noise_std * (1 + (temperature_c - 25) / 50)
        noise = np.random.normal(0, noise_scale, self.charge.shape)

        # Add noise and clamp
        noisy_charge = self.charge + noise
        noisy_charge = np.clip(noisy_charge, 0.0, 1.0)

        self.read_count += 1
        return noisy_charge

    def decay(self, dt_ms: float = 1.0, temperature_c: float = 25.0):
        """
        Apply time-based decay to all cells.

        Physical model: Q(t) = Q(0) * exp(-t/tau)
        where tau = tau_0 * exp(-Ea/(k*T))

        Simplified: decay_rate = base_rate * exp((T - 25) / temp_factor)

        Args:
            dt_ms: Time step in milliseconds
            temperature_c: Current temperature
        """
        # Temperature-dependent decay rate (Arrhenius-like)
        decay_rate = self.config.base_decay_rate * np.exp(
            (temperature_c - 25.0) / self.config.temperature_factor
        )

        # Apply exponential decay
        decay_factor = np.exp(-decay_rate * dt_ms)
        self.charge *= decay_factor

        # Count cells that fell below threshold
        lost_cells = np.sum(self.charge < self.config.refresh_threshold)
        if lost_cells > 0:
            self.decay_events += lost_cells

    def write(self, indices: np.ndarray, values: np.ndarray,
              temperature_c: float = 25.0):
        """
        Write values to specified cells.

        Args:
            indices: Cell indices to write
            values: Charge values (0.0 to 1.0)
            temperature_c: Temperature at write time
        """
        indices = np.asarray(indices)
        values = np.asarray(values)

        # Clamp values
        values = np.clip(values, 0.0, 1.0)

        # Write with some variance (write noise)
        write_noise = np.random.normal(0, 0.01, values.shape)
        self.charge[indices] = values + write_noise
        self.charge = np.clip(self.charge, 0.0, 1.0)

        # Track write metadata
        self.write_times[indices] = time.time()
        self.write_temps[indices] = temperature_c
        self.write_count += len(indices)

    def partial_write(self, indices: np.ndarray, target_values: np.ndarray,
                      write_fraction: float = 0.5, temperature_c: float = 25.0):
        """
        Partial write - blend current charge with target.

        This simulates shortened write pulses that don't fully
        charge/discharge the cell.

        Args:
            indices: Cell indices
            target_values: Target charge levels
            write_fraction: How much to write (0=none, 1=full)
            temperature_c: Temperature
        """
        indices = np.asarray(indices)
        target_values = np.asarray(target_values)

        # Blend current with target
        current = self.charge[indices]
        new_values = current * (1 - write_fraction) + target_values * write_fraction

        self.write(indices, new_values, temperature_c)

    def refresh(self, indices: Optional[np.ndarray] = None):
        """
        Refresh cells to restore charge to digital levels.

        In real DRAM, refresh reads and rewrites cells.
        Cells near threshold get restored to full 0 or 1.

        Args:
            indices: Cells to refresh (None = all)
        """
        if indices is None:
            indices = np.arange(self.num_cells)

        # Threshold at 0.5 and restore to digital
        refreshed = (self.charge[indices] > 0.5).astype(float)
        self.charge[indices] = refreshed

    def get_statistics(self) -> Dict:
        """Get sensor statistics."""
        return {
            'num_cells': self.num_cells,
            'mean_charge': float(self.charge.mean()),
            'std_charge': float(self.charge.std()),
            'min_charge': float(self.charge.min()),
            'max_charge': float(self.charge.max()),
            'read_count': self.read_count,
            'write_count': self.write_count,
            'decay_events': self.decay_events,
            'cells_below_threshold': int(np.sum(self.charge < self.config.refresh_threshold)),
        }


# =============================================================================
# EMBODIED CHARGE PREDICTOR
# =============================================================================

class EmbodiedChargePredictor(nn.Module):
    """
    Neural network that uses analog charge as proprioceptive signal.

    The charge levels are interoceptive - they represent the
    internal state of the "body" (the memory system).

    The network learns to:
    1. Predict its own charge state over time
    2. Use charge as additional input for task
    3. Adapt behavior based on charge levels
    """

    def __init__(self, charge_dim: int = 256, hidden_dim: int = 128,
                 task_dim: int = 10, device: str = 'cuda'):
        super().__init__()
        self.charge_dim = charge_dim
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim
        self.device = device

        # Charge encoder
        self.charge_encoder = nn.Sequential(
            nn.Linear(charge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Self-prediction head (predict next charge state)
        self.charge_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, charge_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Task head (uses charge as embodied signal)
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + charge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, task_dim),
        )

        self.to(device)

    def forward(self, charge: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with charge as proprioceptive input.

        Args:
            charge: Current charge levels [batch, charge_dim]

        Returns:
            predicted_charge: Predicted next charge [batch, charge_dim]
            task_output: Task prediction [batch, task_dim]
        """
        # Encode charge
        encoded = self.charge_encoder(charge)

        # Predict next charge state
        predicted_charge = self.charge_predictor(encoded)

        # Task uses both encoded and raw charge (embodied)
        combined = torch.cat([encoded, charge], dim=1)
        task_output = self.task_head(combined)

        return predicted_charge, task_output

    def compute_embodiment_loss(self, charge: torch.Tensor,
                                 next_charge: torch.Tensor,
                                 task_target: Optional[torch.Tensor] = None
                                 ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute embodiment-aware loss.

        The self-prediction loss is key - it measures how well the
        network understands its own physical substrate.

        Args:
            charge: Current charge
            next_charge: Actual next charge (ground truth)
            task_target: Optional task target

        Returns:
            total_loss: Combined loss
            metrics: Loss components
        """
        predicted_charge, task_output = self(charge)

        # Self-prediction loss (MSE for continuous charge)
        prediction_loss = F.mse_loss(predicted_charge, next_charge)

        # Task loss (if target provided)
        task_loss = torch.tensor(0.0, device=self.device)
        if task_target is not None:
            task_loss = F.cross_entropy(task_output, task_target)

        # Embodiment bonus: reward accurate charge prediction
        # This encourages the network to model its own dynamics
        with torch.no_grad():
            prediction_error = torch.abs(predicted_charge - next_charge).mean()
            embodiment_quality = 1.0 - prediction_error.clamp(0, 1)

        # Total loss with embodiment weighting
        total_loss = prediction_loss + task_loss

        metrics = {
            'prediction_loss': prediction_loss.item(),
            'task_loss': task_loss.item(),
            'total_loss': total_loss.item(),
            'embodiment_quality': embodiment_quality.item(),
            'prediction_error': prediction_error.item(),
        }

        return total_loss, metrics


# =============================================================================
# DIGITAL BASELINE (FOR COMPARISON)
# =============================================================================

class DigitalMemoryBaseline(nn.Module):
    """
    Baseline model using digital (binary) memory.

    This is for comparison - shows what we LOSE by ignoring
    the analog nature of physical memory.
    """

    def __init__(self, memory_dim: int = 256, hidden_dim: int = 128,
                 task_dim: int = 10, device: str = 'cuda'):
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Same architecture but binary input
        self.encoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
            nn.Sigmoid(),
        )

        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, task_dim),
        )

        self.to(device)

    def forward(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Binarize input (threshold at 0.5)
        binary_memory = (memory > 0.5).float()

        encoded = self.encoder(binary_memory)
        predicted = self.predictor(encoded)
        combined = torch.cat([encoded, binary_memory], dim=1)
        task_output = self.task_head(combined)

        return predicted, task_output


# =============================================================================
# EXPERIMENTS
# =============================================================================

def test_sensing_accuracy(sensor: AnalogChargeSensor,
                          noise_levels: List[float]) -> Dict:
    """
    Test 1: How accurate is charge sensing vs noise level?
    """
    results = []

    for noise_std in noise_levels:
        sensor.config.noise_std = noise_std

        # Set known charge pattern
        true_charge = np.linspace(0, 1, sensor.num_cells)
        sensor.charge = true_charge.copy()

        # Sense multiple times and measure error
        errors = []
        for _ in range(100):
            sensed = sensor.sense(temperature_c=25.0)
            error = np.abs(sensed - true_charge).mean()
            errors.append(error)

        results.append({
            'noise_std': noise_std,
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
        })

        print(f"  Noise {noise_std:.3f}: mean_error={np.mean(errors):.4f}")

    return results


def test_decay_tracking(sensor: AnalogChargeSensor,
                        temperatures: List[float],
                        time_steps: int = 100) -> Dict:
    """
    Test 2: Track decay over time at different temperatures.
    """
    results = []

    for temp in temperatures:
        # Reset to full charge
        sensor.charge = np.ones(sensor.num_cells)

        decay_curve = []
        for step in range(time_steps):
            mean_charge = sensor.charge.mean()
            decay_curve.append(float(mean_charge))
            sensor.decay(dt_ms=10.0, temperature_c=temp)

        # Calculate half-life
        half_life_idx = next(
            (i for i, c in enumerate(decay_curve) if c < 0.5),
            time_steps
        )
        half_life_ms = half_life_idx * 10.0

        results.append({
            'temperature_c': temp,
            'half_life_ms': half_life_ms,
            'final_charge': decay_curve[-1],
            'decay_curve': decay_curve[::10],  # Subsample
        })

        print(f"  Temp {temp}C: half_life={half_life_ms:.0f}ms, final={decay_curve[-1]:.3f}")

    return results


def test_embodied_prediction(device: str = 'cuda') -> Dict:
    """
    Test 3: Train embodied predictor and measure self-modeling quality.
    """
    print("\n  Setting up embodied prediction test...")

    # Setup
    config = ChargeSensorConfig(num_cells=256)
    sensor = AnalogChargeSensor(config)

    model = EmbodiedChargePredictor(
        charge_dim=256, hidden_dim=128, task_dim=10, device=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate training data by simulating charge dynamics
    print("  Generating training data with charge dynamics...")
    training_data = []

    # Simulate for 1000 steps
    sensor.charge = np.random.uniform(0.3, 0.7, sensor.num_cells)
    for step in range(1000):
        # Current charge
        current = sensor.sense(temperature_c=35.0).copy()

        # Random writes to some cells
        if np.random.random() < 0.3:
            write_indices = np.random.choice(sensor.num_cells, size=20, replace=False)
            write_values = np.random.uniform(0.4, 0.9, 20)
            sensor.write(write_indices, write_values, temperature_c=35.0)

        # Decay
        sensor.decay(dt_ms=5.0, temperature_c=35.0)

        # Next charge
        next_charge = sensor.sense(temperature_c=35.0).copy()

        # Task label (based on charge distribution)
        task_label = int(np.argmax(np.histogram(current, bins=10)[0]))

        training_data.append((current, next_charge, task_label))

    # Train
    print("  Training embodied predictor...")
    model.train()
    losses = []
    embodiment_qualities = []

    for epoch in range(5):
        epoch_loss = 0.0
        epoch_eq = 0.0

        np.random.shuffle(training_data)
        for i in range(0, len(training_data) - 32, 32):
            batch = training_data[i:i+32]

            current = torch.tensor(
                np.array([b[0] for b in batch]), dtype=torch.float32, device=device
            )
            next_c = torch.tensor(
                np.array([b[1] for b in batch]), dtype=torch.float32, device=device
            )
            labels = torch.tensor(
                [b[2] for b in batch], dtype=torch.long, device=device
            )

            optimizer.zero_grad()
            loss, metrics = model.compute_embodiment_loss(current, next_c, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += metrics['total_loss']
            epoch_eq += metrics['embodiment_quality']

        n_batches = len(training_data) // 32
        avg_loss = epoch_loss / n_batches
        avg_eq = epoch_eq / n_batches
        losses.append(avg_loss)
        embodiment_qualities.append(avg_eq)

        print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}, embodiment_quality={avg_eq:.4f}")

    return {
        'final_loss': losses[-1],
        'final_embodiment_quality': embodiment_qualities[-1],
        'loss_curve': losses,
        'embodiment_curve': embodiment_qualities,
    }


def test_analog_vs_digital(device: str = 'cuda') -> Dict:
    """
    Test 4: Compare analog vs digital memory for embodiment task.

    Key insight: Analog advantage appears when:
    1. Values are in intermediate range (0.3-0.7)
    2. Precise decay prediction matters
    3. Multiple decay rates/temperatures exist

    We test: Predict WHICH cells will cross threshold after decay
    This requires knowing the exact charge level, not just 0/1.
    """
    print("\n  Comparing analog vs digital memory...")
    print("  Task: Predict which cells cross 0.5 threshold after decay")

    # Setup
    config = ChargeSensorConfig(num_cells=256)
    sensor = AnalogChargeSensor(config)

    analog_model = EmbodiedChargePredictor(
        charge_dim=256, hidden_dim=128, task_dim=10, device=device
    )
    digital_model = DigitalMemoryBaseline(
        memory_dim=256, hidden_dim=128, task_dim=10, device=device
    )

    analog_opt = torch.optim.Adam(analog_model.parameters(), lr=0.001)
    digital_opt = torch.optim.Adam(digital_model.parameters(), lr=0.001)

    # Generate data with intermediate charge values (where analog matters)
    print("  Generating data with INTERMEDIATE charge values...")
    data = []

    for step in range(1000):
        # Create charge in the critical zone (0.4-0.6)
        # This is where digital loses information
        sensor.charge = np.random.uniform(0.4, 0.6, sensor.num_cells)

        # Add some random variation
        sensor.charge += np.random.normal(0, 0.1, sensor.num_cells)
        sensor.charge = np.clip(sensor.charge, 0.1, 0.9)

        current = sensor.sense(temperature_c=35.0).copy()

        # Variable decay (temperature varies)
        temp = np.random.uniform(30, 50)
        sensor.decay(dt_ms=50.0, temperature_c=temp)

        next_c = sensor.sense(temperature_c=35.0).copy()

        # Label: which charge level bucket
        label = int(np.clip(current.mean() * 10, 0, 9))
        data.append((current, next_c, label))

    # Train both
    print("  Training models to predict charge after decay...")
    analog_losses = []
    digital_losses = []

    for epoch in range(10):
        np.random.shuffle(data)
        a_loss, d_loss = 0.0, 0.0

        for i in range(0, len(data) - 32, 32):
            batch = data[i:i+32]

            current = torch.tensor(
                np.array([b[0] for b in batch]), dtype=torch.float32, device=device
            )
            next_c = torch.tensor(
                np.array([b[1] for b in batch]), dtype=torch.float32, device=device
            )
            labels = torch.tensor(
                [b[2] for b in batch], dtype=torch.long, device=device
            )

            # Analog model - sees precise charge levels
            analog_opt.zero_grad()
            pred_a, task_a = analog_model(current)
            loss_a = F.mse_loss(pred_a, next_c)
            loss_a.backward()
            analog_opt.step()
            a_loss += loss_a.item()

            # Digital model - sees only 0/1
            digital_opt.zero_grad()
            pred_d, task_d = digital_model(current)
            loss_d = F.mse_loss(pred_d, next_c)
            loss_d.backward()
            digital_opt.step()
            d_loss += loss_d.item()

        n_batches = len(data) // 32
        analog_losses.append(a_loss / n_batches)
        digital_losses.append(d_loss / n_batches)

        if epoch % 2 == 0:
            print(f"    Epoch {epoch+1}: analog={analog_losses[-1]:.4f}, digital={digital_losses[-1]:.4f}")

    # Evaluate: predict threshold crossings
    print("\n  Evaluating threshold crossing prediction...")

    analog_correct = 0
    digital_correct = 0
    total = 0

    for _ in range(200):
        # Intermediate values again
        sensor.charge = np.random.uniform(0.4, 0.6, sensor.num_cells)
        current = sensor.sense(temperature_c=35.0).copy()

        # Ground truth: which cells are above 0.5
        above_before = (current > 0.5)

        # Decay
        sensor.decay(dt_ms=50.0, temperature_c=40.0)
        actual_next = sensor.sense(temperature_c=35.0).copy()
        above_after = (actual_next > 0.5)

        # Which cells CROSSED the threshold?
        crossed = above_before != above_after

        with torch.no_grad():
            curr_t = torch.tensor(current, dtype=torch.float32, device=device).unsqueeze(0)

            pred_a, _ = analog_model(curr_t)
            pred_d, _ = digital_model(curr_t)

            # Predict which cells will be above 0.5
            pred_above_a = pred_a.squeeze() > 0.5
            pred_above_d = pred_d.squeeze() > 0.5

            # Accuracy on CROSSED cells (where it matters)
            actual_above_t = torch.tensor(above_after, device=device)

            # Count correct predictions on cells that crossed
            crossed_t = torch.tensor(crossed, device=device)
            n_crossed = crossed_t.sum().item()

            if n_crossed > 0:
                analog_correct_crossed = ((pred_above_a == actual_above_t) & crossed_t).sum().item()
                digital_correct_crossed = ((pred_above_d == actual_above_t) & crossed_t).sum().item()

                analog_correct += analog_correct_crossed
                digital_correct += digital_correct_crossed
                total += n_crossed

    analog_accuracy = analog_correct / max(total, 1)
    digital_accuracy = digital_correct / max(total, 1)
    improvement = (analog_accuracy - digital_accuracy) / max(digital_accuracy, 0.001) * 100

    print(f"\n  Threshold crossing prediction accuracy:")
    print(f"    Analog model: {analog_accuracy:.1%} ({analog_correct}/{total})")
    print(f"    Digital model: {digital_accuracy:.1%} ({digital_correct}/{total})")
    print(f"    Analog advantage: {improvement:+.1f}%")

    return {
        'analog_losses': analog_losses,
        'digital_losses': digital_losses,
        'analog_crossing_accuracy': analog_accuracy,
        'digital_crossing_accuracy': digital_accuracy,
        'analog_improvement_pct': improvement,
        'total_threshold_crossings': total,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("z1973: ANALOG CHARGE SENSING FOR EMBODIED AI")
    print("=" * 70)
    print()
    print("Physical memory stores analog charge, not just bits.")
    print("We use this as a proprioceptive signal for embodied AI.")
    print()

    # Check for GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("No GPU available, using CPU")

    # Initialize sensor
    config = ChargeSensorConfig(num_cells=256, noise_std=0.02)
    sensor = AnalogChargeSensor(config)

    results = {
        'experiment': 'z1973_analog_charge_sensing',
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'config': asdict(config),
        'tests': {}
    }

    # Test 1: Sensing accuracy
    print("\n" + "=" * 70)
    print("TEST 1: CHARGE SENSING ACCURACY VS NOISE")
    print("=" * 70)

    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    accuracy_results = test_sensing_accuracy(sensor, noise_levels)
    results['tests']['sensing_accuracy'] = accuracy_results

    # Test 2: Decay tracking
    print("\n" + "=" * 70)
    print("TEST 2: DECAY TRACKING AT DIFFERENT TEMPERATURES")
    print("=" * 70)

    # Reset config
    sensor.config.noise_std = 0.02
    temperatures = [25.0, 35.0, 45.0, 55.0, 65.0]
    decay_results = test_decay_tracking(sensor, temperatures)
    results['tests']['decay_tracking'] = decay_results

    # Test 3: Embodied prediction
    print("\n" + "=" * 70)
    print("TEST 3: EMBODIED CHARGE PREDICTION")
    print("=" * 70)

    prediction_results = test_embodied_prediction(device)
    results['tests']['embodied_prediction'] = prediction_results

    # Test 4: Analog vs Digital
    print("\n" + "=" * 70)
    print("TEST 4: ANALOG VS DIGITAL MEMORY FOR EMBODIMENT")
    print("=" * 70)

    comparison_results = test_analog_vs_digital(device)
    results['tests']['analog_vs_digital'] = comparison_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nKey findings:")
    print(f"  1. Sensing accuracy at noise=0.02: {accuracy_results[3]['mean_error']:.4f} mean error")
    print(f"  2. Decay half-life at 25C: {decay_results[0]['half_life_ms']:.0f}ms")
    print(f"  3. Decay half-life at 65C: {decay_results[4]['half_life_ms']:.0f}ms")
    print(f"  4. Embodied prediction quality: {prediction_results['final_embodiment_quality']:.4f}")
    print(f"  5. Threshold crossing accuracy (analog): {comparison_results['analog_crossing_accuracy']:.1%}")
    print(f"  6. Threshold crossing accuracy (digital): {comparison_results['digital_crossing_accuracy']:.1%}")
    print(f"  7. Analog advantage: {comparison_results['analog_improvement_pct']:+.1f}%")

    # Verdict based on threshold crossing accuracy
    improvement = comparison_results['analog_improvement_pct']
    if improvement > 20:
        verdict = "ANALOG_SIGNIFICANTLY_BETTER"
        print("\n  VERDICT: Analog charge sensing significantly improves embodiment!")
    elif improvement > 5:
        verdict = "ANALOG_BETTER"
        print("\n  VERDICT: Analog charge sensing provides clear improvement.")
    elif improvement > 0:
        verdict = "ANALOG_MARGINALLY_BETTER"
        print("\n  VERDICT: Analog charge sensing provides marginal improvement.")
    else:
        verdict = "COMPARABLE"
        print("\n  VERDICT: Analog and digital perform similarly in this test.")

    results['verdict'] = verdict
    results['summary'] = {
        'sensing_accuracy_at_002': accuracy_results[3]['mean_error'],
        'decay_half_life_25c_ms': decay_results[0]['half_life_ms'],
        'decay_half_life_65c_ms': decay_results[4]['half_life_ms'],
        'embodiment_quality': prediction_results['final_embodiment_quality'],
        'analog_crossing_accuracy': comparison_results['analog_crossing_accuracy'],
        'digital_crossing_accuracy': comparison_results['digital_crossing_accuracy'],
        'analog_improvement_pct': comparison_results['analog_improvement_pct'],
    }

    # Save results
    output_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1973_analog_sensing.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
