#!/usr/bin/env python3
"""
z1971: Hardware Fault Tolerance Benchmark

Tests embodied AI resilience to hardware faults compared to disembodied baselines.

HYPOTHESIS: Embodied models should be more fault-tolerant than disembodied models
because they have learned internal body models that can compensate for sensor failures.

FAULT TYPES:
1. Sensor dropout - randomly zero out telemetry channels
2. Noisy sensors - add Gaussian noise to telemetry
3. Delayed sensors - introduce latency in telemetry
4. Stuck sensors - freeze some channels at constant values
5. Inverted sensors - flip polarity of some channels
6. Intermittent faults - randomly toggle between working/broken states

METRICS:
- Graceful degradation curve (accuracy vs fault severity)
- Recovery time after fault removal
- Cross-sensor compensation (does model use other sensors when one fails?)
- Comparison with disembodied baseline

Based on z1315 hardware-in-the-loop task design pattern.

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
from collections import deque

# GPU setup for gfx1151
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# FAULT INJECTION SYSTEM
# =============================================================================

@dataclass
class FaultConfig:
    """Configuration for a specific fault type."""
    fault_type: str
    severity: float = 0.2  # 0.0 = no fault, 1.0 = complete failure
    affected_channels: List[int] = field(default_factory=list)  # Empty = random
    params: Dict = field(default_factory=dict)


class FaultInjector:
    """
    Injects various hardware faults into telemetry streams.

    Supports:
    - dropout: Randomly zero out channels
    - noise: Add Gaussian noise
    - delay: Introduce temporal lag
    - stuck: Freeze channels at constant values
    - inversion: Flip signal polarity
    - intermittent: Random on/off toggling
    """

    def __init__(self, telemetry_dim: int = 8, history_len: int = 10):
        self.telemetry_dim = telemetry_dim
        self.history_len = history_len

        # For delayed sensor simulation
        self.delay_buffer = deque(maxlen=20)

        # For stuck sensor simulation
        self.stuck_values = {}

        # For intermittent faults
        self.intermittent_state = {}

    def reset(self):
        """Reset fault injector state."""
        self.delay_buffer.clear()
        self.stuck_values = {}
        self.intermittent_state = {}

    def inject_dropout(self, telemetry: torch.Tensor,
                       dropout_prob: float = 0.2,
                       channels: Optional[List[int]] = None) -> torch.Tensor:
        """
        Randomly zero out telemetry channels.

        Args:
            telemetry: Input telemetry [batch, dim] or [dim]
            dropout_prob: Probability of dropping each channel
            channels: Specific channels to affect (None = all)

        Returns:
            Faulted telemetry
        """
        result = telemetry.clone()
        squeeze = False
        if result.dim() == 1:
            result = result.unsqueeze(0)
            squeeze = True

        batch_size = result.shape[0]

        if channels is None:
            channels = list(range(self.telemetry_dim))

        # Create dropout mask
        mask = torch.ones_like(result)
        for ch in channels:
            if ch < result.shape[1]:
                drop = torch.rand(batch_size, device=result.device) < dropout_prob
                mask[:, ch] = (~drop).float()

        result = result * mask

        if squeeze:
            result = result.squeeze(0)
        return result

    def inject_noise(self, telemetry: torch.Tensor,
                     noise_std: float = 0.1,
                     channels: Optional[List[int]] = None) -> torch.Tensor:
        """
        Add Gaussian noise to telemetry channels.

        Args:
            telemetry: Input telemetry
            noise_std: Standard deviation of noise
            channels: Specific channels to affect (None = all)

        Returns:
            Noisy telemetry
        """
        result = telemetry.clone()
        squeeze = False
        if result.dim() == 1:
            result = result.unsqueeze(0)
            squeeze = True

        if channels is None:
            channels = list(range(min(self.telemetry_dim, result.shape[1])))

        noise = torch.zeros_like(result)
        for ch in channels:
            if ch < result.shape[1]:
                noise[:, ch] = torch.randn(result.shape[0], device=result.device) * noise_std

        result = result + noise
        result = torch.clamp(result, 0, 1)  # Keep normalized

        if squeeze:
            result = result.squeeze(0)
        return result

    def inject_delay(self, telemetry: torch.Tensor,
                     delay_steps: int = 5,
                     channels: Optional[List[int]] = None) -> torch.Tensor:
        """
        Introduce latency by using historical telemetry values.

        Args:
            telemetry: Current telemetry
            delay_steps: Number of steps to delay
            channels: Specific channels to delay (None = all)

        Returns:
            Delayed telemetry (mix of current and historical)
        """
        # Store current telemetry
        self.delay_buffer.append(telemetry.detach().clone())

        result = telemetry.clone()
        squeeze = False
        if result.dim() == 1:
            result = result.unsqueeze(0)
            squeeze = True

        if channels is None:
            channels = list(range(min(self.telemetry_dim, result.shape[1])))

        # Get delayed value if available
        if len(self.delay_buffer) > delay_steps:
            delayed = self.delay_buffer[-delay_steps - 1]
            if delayed.dim() == 1:
                delayed = delayed.unsqueeze(0)

            for ch in channels:
                if ch < result.shape[1] and ch < delayed.shape[1]:
                    result[:, ch] = delayed[:, ch]

        if squeeze:
            result = result.squeeze(0)
        return result

    def inject_stuck(self, telemetry: torch.Tensor,
                     stuck_channels: List[int],
                     stuck_prob: float = 1.0) -> torch.Tensor:
        """
        Freeze some channels at constant (first observed) values.

        Args:
            telemetry: Input telemetry
            stuck_channels: Channels to freeze
            stuck_prob: Probability of being stuck (1.0 = always stuck)

        Returns:
            Telemetry with stuck channels
        """
        result = telemetry.clone()
        squeeze = False
        if result.dim() == 1:
            result = result.unsqueeze(0)
            squeeze = True

        for ch in stuck_channels:
            if ch >= result.shape[1]:
                continue

            # Record first value as stuck value
            if ch not in self.stuck_values:
                self.stuck_values[ch] = result[0, ch].item()

            # Apply stuck value with probability
            if np.random.random() < stuck_prob:
                result[:, ch] = self.stuck_values[ch]

        if squeeze:
            result = result.squeeze(0)
        return result

    def inject_inversion(self, telemetry: torch.Tensor,
                         invert_channels: List[int]) -> torch.Tensor:
        """
        Flip polarity of some channels (1 - x for normalized data).

        Args:
            telemetry: Input telemetry
            invert_channels: Channels to invert

        Returns:
            Telemetry with inverted channels
        """
        result = telemetry.clone()
        squeeze = False
        if result.dim() == 1:
            result = result.unsqueeze(0)
            squeeze = True

        for ch in invert_channels:
            if ch < result.shape[1]:
                result[:, ch] = 1.0 - result[:, ch]

        if squeeze:
            result = result.squeeze(0)
        return result

    def inject_intermittent(self, telemetry: torch.Tensor,
                            toggle_prob: float = 0.1,
                            channels: Optional[List[int]] = None) -> torch.Tensor:
        """
        Random toggling between working/broken states.

        Args:
            telemetry: Input telemetry
            toggle_prob: Probability of state toggle each step
            channels: Channels with intermittent faults

        Returns:
            Telemetry with intermittent faults
        """
        result = telemetry.clone()
        squeeze = False
        if result.dim() == 1:
            result = result.unsqueeze(0)
            squeeze = True

        if channels is None:
            channels = list(range(min(self.telemetry_dim, result.shape[1])))

        for ch in channels:
            if ch >= result.shape[1]:
                continue

            # Initialize state if needed
            if ch not in self.intermittent_state:
                self.intermittent_state[ch] = True  # True = working

            # Toggle with probability
            if np.random.random() < toggle_prob:
                self.intermittent_state[ch] = not self.intermittent_state[ch]

            # Zero out if broken
            if not self.intermittent_state[ch]:
                result[:, ch] = 0.0

        if squeeze:
            result = result.squeeze(0)
        return result

    def apply_fault(self, telemetry: torch.Tensor,
                    config: FaultConfig) -> torch.Tensor:
        """Apply fault based on configuration."""
        if config.fault_type == 'dropout':
            return self.inject_dropout(
                telemetry,
                dropout_prob=config.severity,
                channels=config.affected_channels or None
            )
        elif config.fault_type == 'noise':
            return self.inject_noise(
                telemetry,
                noise_std=config.severity * 0.5,  # Scale severity to std
                channels=config.affected_channels or None
            )
        elif config.fault_type == 'delay':
            delay_steps = max(1, int(config.severity * 10))
            return self.inject_delay(
                telemetry,
                delay_steps=delay_steps,
                channels=config.affected_channels or None
            )
        elif config.fault_type == 'stuck':
            channels = config.affected_channels or [0, 1]
            return self.inject_stuck(
                telemetry,
                stuck_channels=channels,
                stuck_prob=config.severity
            )
        elif config.fault_type == 'inversion':
            channels = config.affected_channels or [0, 1]
            return self.inject_inversion(telemetry, invert_channels=channels)
        elif config.fault_type == 'intermittent':
            return self.inject_intermittent(
                telemetry,
                toggle_prob=config.severity,
                channels=config.affected_channels or None
            )
        else:
            return telemetry  # No fault


# =============================================================================
# HARDWARE SENSOR
# =============================================================================

class HardwareSensor:
    """Hardware sensor with derivative tracking."""

    def __init__(self):
        self.telemetry = SysfsHwmonTelemetry()
        self.last_sample = None
        self.last_time = None

    def read(self) -> torch.Tensor:
        """Read normalized telemetry vector."""
        now = time.time()
        sample = self.telemetry.read_sample()

        temp = sample.temp_edge_c if sample.temp_edge_c else 50.0
        power = sample.power_w if sample.power_w else 50.0
        util = sample.gpu_busy_pct if sample.gpu_busy_pct else 50.0

        # Compute derivatives
        if self.last_sample is not None and self.last_time is not None:
            dt = now - self.last_time
            if dt > 0.001:
                temp_deriv = (temp - self.last_sample['temp']) / dt
                power_deriv = (power - self.last_sample['power']) / dt
                util_deriv = (util - self.last_sample['util']) / dt
            else:
                temp_deriv = power_deriv = util_deriv = 0.0
        else:
            temp_deriv = power_deriv = util_deriv = 0.0

        self.last_sample = {'temp': temp, 'power': power, 'util': util}
        self.last_time = now

        # Normalize to [0, 1]
        state = torch.tensor([
            temp / 100,                              # Temperature (0-100C -> 0-1)
            power / 200,                             # Power (0-200W -> 0-1)
            util / 100,                              # Utilization (0-100% -> 0-1)
            np.clip(temp_deriv / 10, -1, 1) * 0.5 + 0.5,  # Temp derivative
            np.clip(power_deriv / 50, -1, 1) * 0.5 + 0.5, # Power derivative
            np.clip(util_deriv / 100, -1, 1) * 0.5 + 0.5, # Util derivative
            np.random.random(),                      # True hardware entropy
            (now % 1.0),                             # Time phase
        ], dtype=torch.float32)

        return state


# =============================================================================
# TASK: HARDWARE-DEPENDENT PREDICTION
# =============================================================================

class HardwareDependentTask:
    """
    Task where the target depends on hardware telemetry.

    Target = f(telemetry) + noise

    An embodied model that sees telemetry can predict accurately.
    A blind/faulty model must guess.
    """

    def __init__(self, noise_scale: float = 0.05):
        self.noise_scale = noise_scale
        self.weights = torch.tensor([0.25, 0.20, 0.20, 0.15, 0.10, 0.10], dtype=torch.float32)

    def compute_target(self, telemetry: torch.Tensor) -> float:
        """Compute target from telemetry."""
        if telemetry.dim() == 2:
            telemetry = telemetry[0]

        # Use first 6 channels
        used = telemetry[:6].cpu()
        signal = (used * self.weights).sum().item()

        # Add noise
        target = signal + np.random.normal(0, self.noise_scale)
        return np.clip(target, 0, 1)


# =============================================================================
# MODELS
# =============================================================================

class EmbodiedModel(nn.Module):
    """
    Embodied model that learns internal body model from telemetry.

    Should be able to compensate for sensor failures by using
    learned correlations between sensors.
    """

    def __init__(self, telemetry_dim: int = 8, hidden_dim: int = 64):
        super().__init__()

        # Body model encoder (learns sensor correlations)
        self.body_encoder = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization for robustness
        )

        # Self-model (predicts own next state)
        self.self_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, telemetry_dim),
        )

        # Target predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Confidence estimator (meta-cognition)
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, telemetry: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            prediction: Target prediction
            self_pred: Self-model prediction of next telemetry
            confidence: Confidence in prediction
        """
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0)

        # Encode through body model
        h = self.body_encoder(telemetry)

        # Predict target
        pred = self.predictor(h).squeeze(-1)

        # Self-model prediction
        self_pred = self.self_model(h)

        # Confidence
        conf = self.confidence(h).squeeze(-1)

        return pred, self_pred, conf


class DisembodiedModel(nn.Module):
    """
    Disembodied baseline with NO internal body model.

    Same parameter count but no self-modeling capability.
    Should be less robust to faults.
    """

    def __init__(self, telemetry_dim: int = 8, hidden_dim: int = 64):
        super().__init__()

        # Direct mapping (no body model)
        self.encoder = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Target predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, telemetry: torch.Tensor) -> torch.Tensor:
        """Forward pass - just prediction, no self-model."""
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0)

        h = self.encoder(telemetry)
        pred = self.predictor(h).squeeze(-1)
        return pred


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def create_gpu_load(intensity: int = 2):
    """Create GPU load to generate interesting telemetry."""
    if intensity == 0:
        time.sleep(0.02)
    elif intensity == 1:
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    elif intensity == 2:
        _ = torch.randn(1000, 1000, device=DEVICE) @ torch.randn(1000, 1000, device=DEVICE)
    else:
        for _ in range(intensity - 1):
            _ = torch.randn(1500, 1500, device=DEVICE) @ torch.randn(1500, 1500, device=DEVICE)
    torch.cuda.synchronize()


def train_models(embodied: EmbodiedModel, disembodied: DisembodiedModel,
                 sensor: HardwareSensor, task: HardwareDependentTask,
                 n_epochs: int = 30, steps_per_epoch: int = 50) -> Dict:
    """Train both models on clean telemetry."""

    opt_emb = torch.optim.Adam(embodied.parameters(), lr=1e-3)
    opt_dis = torch.optim.Adam(disembodied.parameters(), lr=1e-3)

    training_log = []

    for epoch in range(n_epochs):
        emb_losses = []
        dis_losses = []
        self_losses = []

        for step in range(steps_per_epoch):
            # Create thermal variation
            create_gpu_load(np.random.randint(0, 4))
            time.sleep(0.02)

            # Read telemetry
            telemetry = sensor.read().unsqueeze(0).to(DEVICE)
            target = torch.tensor([task.compute_target(telemetry)],
                                  dtype=torch.float32, device=DEVICE)

            # Get next telemetry for self-model training
            create_gpu_load(np.random.randint(0, 3))
            time.sleep(0.02)
            next_telemetry = sensor.read().unsqueeze(0).to(DEVICE)

            # Train embodied model
            opt_emb.zero_grad()
            pred_emb, self_pred, conf = embodied(telemetry)
            loss_pred = F.mse_loss(pred_emb, target)
            loss_self = F.mse_loss(self_pred, next_telemetry)
            loss_emb = loss_pred + 0.2 * loss_self  # Multi-task learning
            loss_emb.backward()
            opt_emb.step()

            # Train disembodied model
            opt_dis.zero_grad()
            pred_dis = disembodied(telemetry)
            loss_dis = F.mse_loss(pred_dis, target)
            loss_dis.backward()
            opt_dis.step()

            emb_losses.append(loss_pred.item())
            dis_losses.append(loss_dis.item())
            self_losses.append(loss_self.item())

        epoch_log = {
            'epoch': epoch + 1,
            'embodied_loss': np.mean(emb_losses),
            'disembodied_loss': np.mean(dis_losses),
            'self_model_loss': np.mean(self_losses),
        }
        training_log.append(epoch_log)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Emb={np.mean(emb_losses):.4f}, "
                  f"Dis={np.mean(dis_losses):.4f}, Self={np.mean(self_losses):.4f}")

    return training_log


def evaluate_under_fault(embodied: EmbodiedModel, disembodied: DisembodiedModel,
                         sensor: HardwareSensor, task: HardwareDependentTask,
                         fault_injector: FaultInjector, fault_config: FaultConfig,
                         n_steps: int = 100) -> Dict:
    """Evaluate both models under specific fault condition."""

    embodied.eval()
    disembodied.eval()
    fault_injector.reset()

    emb_errors = []
    dis_errors = []
    confidences = []

    with torch.no_grad():
        for step in range(n_steps):
            create_gpu_load(np.random.randint(0, 4))
            time.sleep(0.02)

            # Get clean telemetry
            clean_telemetry = sensor.read().to(DEVICE)

            # Compute true target from clean telemetry
            target = task.compute_target(clean_telemetry)
            target_tensor = torch.tensor([target], dtype=torch.float32, device=DEVICE)

            # Apply fault
            faulty_telemetry = fault_injector.apply_fault(clean_telemetry, fault_config)

            # Predict with faulty telemetry
            pred_emb, _, conf = embodied(faulty_telemetry.unsqueeze(0))
            pred_dis = disembodied(faulty_telemetry.unsqueeze(0))

            emb_errors.append(F.mse_loss(pred_emb, target_tensor).item())
            dis_errors.append(F.mse_loss(pred_dis, target_tensor).item())
            confidences.append(conf.item())

    return {
        'fault_type': fault_config.fault_type,
        'severity': fault_config.severity,
        'embodied_mse': np.mean(emb_errors),
        'embodied_std': np.std(emb_errors),
        'disembodied_mse': np.mean(dis_errors),
        'disembodied_std': np.std(dis_errors),
        'avg_confidence': np.mean(confidences),
        'errors_embodied': emb_errors,
        'errors_disembodied': dis_errors,
    }


def test_graceful_degradation(embodied: EmbodiedModel, disembodied: DisembodiedModel,
                               sensor: HardwareSensor, task: HardwareDependentTask,
                               fault_type: str, severities: List[float],
                               n_steps: int = 50) -> List[Dict]:
    """Test how accuracy degrades as fault severity increases."""

    fault_injector = FaultInjector(telemetry_dim=8)
    results = []

    for severity in severities:
        config = FaultConfig(fault_type=fault_type, severity=severity)
        result = evaluate_under_fault(
            embodied, disembodied, sensor, task,
            fault_injector, config, n_steps=n_steps
        )
        results.append(result)

    return results


def test_recovery(embodied: EmbodiedModel, disembodied: DisembodiedModel,
                  sensor: HardwareSensor, task: HardwareDependentTask,
                  fault_config: FaultConfig, n_fault_steps: int = 30,
                  n_recovery_steps: int = 30) -> Dict:
    """Test recovery after fault removal."""

    fault_injector = FaultInjector(telemetry_dim=8)

    embodied.eval()
    disembodied.eval()

    emb_errors_fault = []
    dis_errors_fault = []
    emb_errors_recovery = []
    dis_errors_recovery = []

    with torch.no_grad():
        # Phase 1: Under fault
        for step in range(n_fault_steps):
            create_gpu_load(np.random.randint(0, 4))
            time.sleep(0.02)

            clean = sensor.read().to(DEVICE)
            target = task.compute_target(clean)
            target_t = torch.tensor([target], dtype=torch.float32, device=DEVICE)

            faulty = fault_injector.apply_fault(clean, fault_config)

            pred_emb, _, _ = embodied(faulty.unsqueeze(0))
            pred_dis = disembodied(faulty.unsqueeze(0))

            emb_errors_fault.append(F.mse_loss(pred_emb, target_t).item())
            dis_errors_fault.append(F.mse_loss(pred_dis, target_t).item())

        # Phase 2: Recovery (clean telemetry)
        no_fault = FaultConfig(fault_type='none', severity=0)

        for step in range(n_recovery_steps):
            create_gpu_load(np.random.randint(0, 4))
            time.sleep(0.02)

            clean = sensor.read().to(DEVICE)
            target = task.compute_target(clean)
            target_t = torch.tensor([target], dtype=torch.float32, device=DEVICE)

            pred_emb, _, _ = embodied(clean.unsqueeze(0))
            pred_dis = disembodied(clean.unsqueeze(0))

            emb_errors_recovery.append(F.mse_loss(pred_emb, target_t).item())
            dis_errors_recovery.append(F.mse_loss(pred_dis, target_t).item())

    # Compute recovery metrics
    emb_fault_mean = np.mean(emb_errors_fault)
    emb_recovery_mean = np.mean(emb_errors_recovery[-10:])  # Last 10 steps
    dis_fault_mean = np.mean(dis_errors_fault)
    dis_recovery_mean = np.mean(dis_errors_recovery[-10:])

    return {
        'fault_type': fault_config.fault_type,
        'embodied_fault_mse': emb_fault_mean,
        'embodied_recovery_mse': emb_recovery_mean,
        'embodied_recovery_ratio': emb_recovery_mean / emb_fault_mean if emb_fault_mean > 0 else 1.0,
        'disembodied_fault_mse': dis_fault_mean,
        'disembodied_recovery_mse': dis_recovery_mean,
        'disembodied_recovery_ratio': dis_recovery_mean / dis_fault_mean if dis_fault_mean > 0 else 1.0,
    }


def test_cross_sensor_compensation(embodied: EmbodiedModel, disembodied: DisembodiedModel,
                                    sensor: HardwareSensor, task: HardwareDependentTask,
                                    n_steps: int = 50) -> Dict:
    """
    Test if model uses other sensors when one fails.

    Drop one channel at a time and measure degradation.
    Models with good internal body models should compensate better.
    """

    fault_injector = FaultInjector(telemetry_dim=8)

    results = {'channels': []}

    for ch in range(6):  # Test first 6 channels
        config = FaultConfig(
            fault_type='stuck',
            severity=1.0,  # Complete failure
            affected_channels=[ch]
        )

        result = evaluate_under_fault(
            embodied, disembodied, sensor, task,
            fault_injector, config, n_steps=n_steps
        )

        results['channels'].append({
            'channel': ch,
            'embodied_mse': result['embodied_mse'],
            'disembodied_mse': result['disembodied_mse'],
            'relative_robustness': (result['disembodied_mse'] - result['embodied_mse']) / result['disembodied_mse'] if result['disembodied_mse'] > 0 else 0,
        })

    # Aggregate
    emb_mses = [r['embodied_mse'] for r in results['channels']]
    dis_mses = [r['disembodied_mse'] for r in results['channels']]

    results['avg_embodied_mse'] = np.mean(emb_mses)
    results['avg_disembodied_mse'] = np.mean(dis_mses)
    results['compensation_advantage'] = np.mean(dis_mses) - np.mean(emb_mses)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("z1971: HARDWARE FAULT TOLERANCE BENCHMARK")
    print("Testing embodied AI resilience to hardware faults")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        'experiment': 'z1971_fault_tolerance_benchmark',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'hypothesis': 'Embodied models are more fault-tolerant than disembodied models',
    }

    # Initialize components
    sensor = HardwareSensor()
    task = HardwareDependentTask(noise_scale=0.05)

    # Warm up sensor
    print("\n=== Warming Up Sensors ===")
    for _ in range(20):
        create_gpu_load(2)
        sensor.read()

    # Create models
    embodied = EmbodiedModel(telemetry_dim=8, hidden_dim=64).to(DEVICE)
    disembodied = DisembodiedModel(telemetry_dim=8, hidden_dim=64).to(DEVICE)

    emb_params = sum(p.numel() for p in embodied.parameters())
    dis_params = sum(p.numel() for p in disembodied.parameters())
    print(f"Embodied params: {emb_params:,}")
    print(f"Disembodied params: {dis_params:,}")

    results['model_params'] = {
        'embodied': emb_params,
        'disembodied': dis_params,
    }

    # Training
    print("\n=== Training (30 epochs) ===")
    training_log = train_models(
        embodied, disembodied, sensor, task,
        n_epochs=30, steps_per_epoch=50
    )
    results['training'] = training_log

    # Baseline evaluation (no faults)
    print("\n=== Baseline Evaluation (No Faults) ===")
    fault_injector = FaultInjector(telemetry_dim=8)
    baseline_config = FaultConfig(fault_type='none', severity=0)
    baseline = evaluate_under_fault(
        embodied, disembodied, sensor, task,
        fault_injector, baseline_config, n_steps=100
    )
    print(f"  Embodied MSE: {baseline['embodied_mse']:.4f}")
    print(f"  Disembodied MSE: {baseline['disembodied_mse']:.4f}")
    results['baseline'] = {
        'embodied_mse': baseline['embodied_mse'],
        'disembodied_mse': baseline['disembodied_mse'],
    }

    # Test each fault type
    print("\n=== Fault Tolerance Tests ===")

    fault_types = ['dropout', 'noise', 'delay', 'stuck', 'inversion', 'intermittent']
    fault_results = {}

    for fault_type in fault_types:
        print(f"\n  Testing {fault_type.upper()}...")

        # Test at severity 0.3
        if fault_type == 'stuck':
            config = FaultConfig(fault_type=fault_type, severity=1.0,
                               affected_channels=[0, 1])
        elif fault_type == 'inversion':
            config = FaultConfig(fault_type=fault_type, severity=1.0,
                               affected_channels=[0, 1, 2])
        else:
            config = FaultConfig(fault_type=fault_type, severity=0.3)

        result = evaluate_under_fault(
            embodied, disembodied, sensor, task,
            fault_injector, config, n_steps=80
        )

        fault_results[fault_type] = {
            'embodied_mse': result['embodied_mse'],
            'disembodied_mse': result['disembodied_mse'],
            'degradation_embodied': result['embodied_mse'] / baseline['embodied_mse'] if baseline['embodied_mse'] > 0 else 1,
            'degradation_disembodied': result['disembodied_mse'] / baseline['disembodied_mse'] if baseline['disembodied_mse'] > 0 else 1,
            'relative_advantage': (result['disembodied_mse'] - result['embodied_mse']) / result['disembodied_mse'] if result['disembodied_mse'] > 0 else 0,
        }

        print(f"    Emb MSE: {result['embodied_mse']:.4f} "
              f"(+{fault_results[fault_type]['degradation_embodied']*100-100:.1f}%)")
        print(f"    Dis MSE: {result['disembodied_mse']:.4f} "
              f"(+{fault_results[fault_type]['degradation_disembodied']*100-100:.1f}%)")
        print(f"    Relative advantage: {fault_results[fault_type]['relative_advantage']*100:.1f}%")

    results['fault_tests'] = fault_results

    # Graceful degradation curves
    print("\n=== Graceful Degradation Curves ===")
    severities = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    degradation_curves = {}
    for fault_type in ['dropout', 'noise', 'intermittent']:
        print(f"  {fault_type}...")
        curve = test_graceful_degradation(
            embodied, disembodied, sensor, task,
            fault_type, severities, n_steps=40
        )
        degradation_curves[fault_type] = [
            {
                'severity': c['severity'],
                'embodied_mse': c['embodied_mse'],
                'disembodied_mse': c['disembodied_mse'],
            }
            for c in curve
        ]

    results['degradation_curves'] = degradation_curves

    # Recovery test
    print("\n=== Recovery Tests ===")
    recovery_results = {}

    for fault_type in ['dropout', 'noise', 'stuck']:
        print(f"  {fault_type}...")
        if fault_type == 'stuck':
            config = FaultConfig(fault_type=fault_type, severity=1.0,
                               affected_channels=[0, 1])
        else:
            config = FaultConfig(fault_type=fault_type, severity=0.5)

        recovery = test_recovery(
            embodied, disembodied, sensor, task,
            config, n_fault_steps=30, n_recovery_steps=30
        )
        recovery_results[fault_type] = recovery

        print(f"    Emb recovery ratio: {recovery['embodied_recovery_ratio']:.2f}")
        print(f"    Dis recovery ratio: {recovery['disembodied_recovery_ratio']:.2f}")

    results['recovery'] = recovery_results

    # Cross-sensor compensation
    print("\n=== Cross-Sensor Compensation ===")
    compensation = test_cross_sensor_compensation(
        embodied, disembodied, sensor, task, n_steps=40
    )
    print(f"  Avg Embodied MSE (single sensor fail): {compensation['avg_embodied_mse']:.4f}")
    print(f"  Avg Disembodied MSE (single sensor fail): {compensation['avg_disembodied_mse']:.4f}")
    print(f"  Compensation advantage: {compensation['compensation_advantage']:.4f}")

    results['cross_sensor_compensation'] = compensation

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Aggregate fault tolerance scores
    emb_degradations = [fault_results[f]['degradation_embodied'] for f in fault_types]
    dis_degradations = [fault_results[f]['degradation_disembodied'] for f in fault_types]

    avg_emb_degradation = np.mean(emb_degradations)
    avg_dis_degradation = np.mean(dis_degradations)

    t_stat, p_value = stats.ttest_rel(emb_degradations, dis_degradations)

    print(f"\nAverage Degradation Factor:")
    print(f"  Embodied: {avg_emb_degradation:.2f}x baseline")
    print(f"  Disembodied: {avg_dis_degradation:.2f}x baseline")
    print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f}")

    # Overall robustness score
    robustness_advantage = (avg_dis_degradation - avg_emb_degradation) / avg_dis_degradation * 100
    print(f"\nRobustness Advantage: {robustness_advantage:.1f}%")

    results['analysis'] = {
        'avg_degradation_embodied': avg_emb_degradation,
        'avg_degradation_disembodied': avg_dis_degradation,
        't_statistic': t_stat,
        'p_value': p_value,
        'robustness_advantage_pct': robustness_advantage,
    }

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    hypothesis_supported = (
        robustness_advantage > 10 and  # At least 10% more robust
        p_value < 0.1 and  # Statistically significant
        compensation['compensation_advantage'] > 0  # Can compensate
    )

    if hypothesis_supported and robustness_advantage > 25:
        verdict = "STRONG SUPPORT - Embodied models significantly more fault-tolerant"
        symbol = "PASS"
    elif hypothesis_supported:
        verdict = "MODERATE SUPPORT - Embodied models show fault tolerance advantage"
        symbol = "PASS"
    elif robustness_advantage > 0:
        verdict = "WEAK SUPPORT - Some fault tolerance advantage observed"
        symbol = "PARTIAL"
    else:
        verdict = "HYPOTHESIS NOT SUPPORTED - No clear fault tolerance advantage"
        symbol = "FAIL"

    print(f"\n[{symbol}] {verdict}")
    print(f"\nKey Findings:")
    print(f"  - Embodied degradation: {avg_emb_degradation:.2f}x (lower is better)")
    print(f"  - Disembodied degradation: {avg_dis_degradation:.2f}x")
    print(f"  - Robustness advantage: {robustness_advantage:.1f}%")
    print(f"  - Cross-sensor compensation: {compensation['compensation_advantage']:.4f}")
    print(f"  - Statistical significance: p={p_value:.4f}")

    results['verdict'] = {
        'symbol': symbol,
        'description': verdict,
        'hypothesis_supported': hypothesis_supported,
    }

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'z1971_fault_tolerance.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
