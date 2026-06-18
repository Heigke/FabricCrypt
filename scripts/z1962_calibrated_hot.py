#!/usr/bin/env python3
"""
z1962: Calibrated Higher-Order Theory (HOT) Confidence

PROBLEM: z1913 showed HOT confidence calibration is -0.395 (NEGATIVE!)
TARGET: Confidence-accuracy correlation > 0 (POSITIVE)

KEY FIXES:
1. Expected Calibration Error (ECE) loss during training
2. Temperature scaling for confidence calibration
3. Platt scaling (logistic regression) post-hoc calibration
4. Isotonic regression as alternative calibrator
5. Two-phase training: task first, then calibration
6. Reliability diagrams to verify calibration

CALIBRATION LOSS:
    ECE = sum(|accuracy_bin - confidence_bin| * weight_bin)
    L_total = L_task + lambda_cal * L_calibration

Uses TRUE hardware entropy sources from z1950 design.

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
import struct
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# GPU setup for gfx1151
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Hardware Entropy (from z1950)
# =============================================================================

class TrueHardwareEntropy:
    """Multi-source TRUE hardware entropy collection."""

    def __init__(self):
        try:
            self.random_fd = open('/dev/random', 'rb')
            print("  [Entropy] /dev/random opened")
        except:
            self.random_fd = None

        self.last_interrupts = self._read_interrupts()
        self.last_interrupt_time = time.time()
        self.entropy_history = deque(maxlen=100)

    def _read_interrupts(self) -> int:
        try:
            with open('/proc/interrupts', 'r') as f:
                total = 0
                for line in f:
                    parts = line.split()
                    if len(parts) > 1:
                        for p in parts[1:]:
                            try:
                                total += int(p)
                            except:
                                break
                return total
        except:
            return 0

    def _get_entropy_avail(self) -> int:
        try:
            with open('/proc/sys/kernel/random/entropy_avail', 'r') as f:
                return int(f.read().strip())
        except:
            return 0

    def read_true_random(self, n_bytes: int = 4) -> float:
        if self.random_fd:
            try:
                entropy = self._get_entropy_avail()
                self.entropy_history.append(entropy)
                if entropy < 64:
                    return self._rdrand_fallback()
                data = self.random_fd.read(n_bytes)
                if len(data) == n_bytes:
                    val = struct.unpack('>I', data)[0]
                    return val / (2**32)
            except:
                pass
        return self._rdrand_fallback()

    def _rdrand_fallback(self) -> float:
        seed_bytes = os.urandom(8)
        seed = struct.unpack('>Q', seed_bytes)[0]
        return (seed & 0xFFFFFFFF) / (2**32)

    def get_interrupt_jitter(self) -> float:
        now = time.time()
        current_interrupts = self._read_interrupts()
        dt = now - self.last_interrupt_time
        if dt > 0.001:
            rate = (current_interrupts - self.last_interrupts) / dt
            normalized = np.clip(rate / 50000, 0, 1)
        else:
            normalized = 0.5
        self.last_interrupts = current_interrupts
        self.last_interrupt_time = now
        return normalized

    def close(self):
        if self.random_fd:
            self.random_fd.close()


class GPUSensor:
    """GPU interoceptive sensing with derivatives."""

    def __init__(self):
        self.card = '/sys/class/drm/card1/device'
        self.temp_history = deque(maxlen=10)
        self.power_history = deque(maxlen=10)
        self.util_history = deque(maxlen=10)
        self.time_history = deque(maxlen=10)

    def _hwmon(self, p, d=0):
        try:
            for h in os.listdir(f'{self.card}/hwmon'):
                f = f'{self.card}/hwmon/{h}/{p}'
                if os.path.exists(f):
                    with open(f) as fp:
                        return float(fp.read().strip())
        except:
            pass
        return d

    def _read(self, f, d=0):
        try:
            with open(f'{self.card}/{f}') as fp:
                return float(fp.read().strip())
        except:
            return d

    def read(self) -> Dict:
        now = time.time()
        temp = self._hwmon('temp1_input', 50000) / 1000
        power = self._hwmon('power1_average', 50e6) / 1e6
        util = self._read('gpu_busy_percent', 50)

        self.temp_history.append(temp)
        self.power_history.append(power)
        self.util_history.append(util)
        self.time_history.append(now)

        # Derivatives
        if len(self.time_history) >= 2:
            dt = self.time_history[-1] - self.time_history[-2]
            if dt > 0.001:
                temp_deriv = (self.temp_history[-1] - self.temp_history[-2]) / dt
                power_deriv = (self.power_history[-1] - self.power_history[-2]) / dt
                util_deriv = (self.util_history[-1] - self.util_history[-2]) / dt
            else:
                temp_deriv = power_deriv = util_deriv = 0
        else:
            temp_deriv = power_deriv = util_deriv = 0

        # Second derivative (acceleration)
        if len(self.temp_history) >= 3 and len(self.time_history) >= 3:
            dt1 = self.time_history[-1] - self.time_history[-2]
            dt2 = self.time_history[-2] - self.time_history[-3]
            if dt1 > 0.001 and dt2 > 0.001:
                d1 = (self.temp_history[-1] - self.temp_history[-2]) / dt1
                d2 = (self.temp_history[-2] - self.temp_history[-3]) / dt2
                temp_accel = (d1 - d2) / ((dt1 + dt2) / 2)
            else:
                temp_accel = 0
        else:
            temp_accel = 0

        return {
            'temp': temp,
            'power': power,
            'util': util / 100,
            'temp_deriv': np.clip(temp_deriv / 10, -1, 1),
            'power_deriv': np.clip(power_deriv / 50, -1, 1),
            'util_deriv': np.clip(util_deriv / 500, -1, 1),
            'temp_accel': np.clip(temp_accel / 10, -1, 1),
        }


# =============================================================================
# Calibration Utilities
# =============================================================================

def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum(|accuracy_bin - confidence_bin| * weight_bin)

    Args:
        confidences: Array of confidence values [0, 1]
        accuracies: Array of binary accuracy (correct=1, incorrect=0)
        n_bins: Number of bins for calibration

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence_in_bin = confidences[in_bin].mean()
            avg_accuracy_in_bin = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece


def compute_mce(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error - worst case bin error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_confidence_in_bin = confidences[in_bin].mean()
            avg_accuracy_in_bin = accuracies[in_bin].mean()
            mce = max(mce, np.abs(avg_accuracy_in_bin - avg_confidence_in_bin))

    return mce


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for confidence calibration.

    p_calibrated = softmax(logits / T)

    For regression, we scale: confidence_calibrated = sigmoid(logit(confidence) / T)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling."""
        return logits / self.temperature


class PlattScaling:
    """
    Platt scaling using logistic regression.

    Maps raw confidence to calibrated confidence via:
    p_calibrated = 1 / (1 + exp(-(a * logit + b)))
    """

    def __init__(self):
        self.model = LogisticRegression()
        self.fitted = False

    def fit(self, confidences: np.ndarray, labels: np.ndarray):
        """Fit Platt scaling on validation data."""
        # Convert confidence to logits
        confidences = np.clip(confidences, 1e-7, 1 - 1e-7)
        logits = np.log(confidences / (1 - confidences)).reshape(-1, 1)

        self.model.fit(logits, labels)
        self.fitted = True

    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to calibrate confidences."""
        if not self.fitted:
            return confidences

        confidences = np.clip(confidences, 1e-7, 1 - 1e-7)
        logits = np.log(confidences / (1 - confidences)).reshape(-1, 1)

        return self.model.predict_proba(logits)[:, 1]


class IsotonicCalibration:
    """
    Isotonic regression for calibration.

    Non-parametric, monotonic calibration.
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False

    def fit(self, confidences: np.ndarray, labels: np.ndarray):
        """Fit isotonic regression."""
        self.model.fit(confidences, labels)
        self.fitted = True

    def calibrate(self, confidences: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if not self.fitted:
            return confidences
        return self.model.predict(confidences)


def compute_reliability_diagram(confidences: np.ndarray, accuracies: np.ndarray,
                                n_bins: int = 10) -> Dict:
    """
    Compute reliability diagram data.

    Returns bin centers, average confidence, average accuracy per bin.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts.append(in_bin.sum())

        if in_bin.sum() > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())

    return {
        'bin_centers': bin_centers,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
    }


# =============================================================================
# Differentiable ECE Loss
# =============================================================================

class ECELoss(nn.Module):
    """
    Differentiable Expected Calibration Error loss.

    Uses soft binning with Gaussian kernels for differentiability.
    """

    def __init__(self, n_bins: int = 15, temperature: float = 0.1):
        super().__init__()
        self.n_bins = n_bins
        self.temperature = temperature

        # Bin centers
        self.register_buffer('bin_centers',
                           torch.linspace(0.5/n_bins, 1 - 0.5/n_bins, n_bins))

    def forward(self, confidences: torch.Tensor, accuracies: torch.Tensor) -> torch.Tensor:
        """
        Compute soft ECE loss.

        Args:
            confidences: [B] confidence values in [0, 1]
            accuracies: [B] binary accuracy (1 if correct, 0 if not)
        """
        # Compute soft bin assignments using Gaussian kernels
        # Shape: [B, n_bins]
        distances = (confidences.unsqueeze(1) - self.bin_centers.unsqueeze(0)) ** 2
        soft_bins = F.softmax(-distances / self.temperature, dim=1)

        # Weighted average confidence and accuracy per bin
        # Shape: [n_bins]
        bin_weights = soft_bins.sum(dim=0) + 1e-8
        bin_confidences = (soft_bins * confidences.unsqueeze(1)).sum(dim=0) / bin_weights
        bin_accuracies = (soft_bins * accuracies.unsqueeze(1)).sum(dim=0) / bin_weights

        # ECE = sum(|acc - conf| * weight)
        bin_weights_normalized = bin_weights / bin_weights.sum()
        ece = (torch.abs(bin_accuracies - bin_confidences) * bin_weights_normalized).sum()

        return ece


# =============================================================================
# Calibrated HOT Model
# =============================================================================

class FirstOrderModule(nn.Module):
    """First-order processing: perceives input."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CalibratedHigherOrderModule(nn.Module):
    """
    Higher-order processing with CALIBRATED confidence.

    Key improvements over z1913:
    1. Separate confidence head with calibration-aware architecture
    2. Temperature scaling built-in
    3. Explicit accuracy target for confidence training
    4. Focal loss to handle easy/hard examples
    """

    def __init__(self, first_order_dim: int, meta_dim: int, telemetry_dim: int):
        super().__init__()

        # Meta-representation of first-order states
        self.meta_encoder = nn.Sequential(
            nn.Linear(first_order_dim, meta_dim),
            nn.GELU(),
            nn.Linear(meta_dim, meta_dim),
        )

        # Telemetry gate
        self.telemetry_gate = nn.Sequential(
            nn.Linear(telemetry_dim, meta_dim),
            nn.Sigmoid(),
        )

        # First-order predictor (self-model)
        self.first_order_predictor = nn.Sequential(
            nn.Linear(meta_dim, first_order_dim * 2),
            nn.GELU(),
            nn.Linear(first_order_dim * 2, first_order_dim),
        )

        # CALIBRATED confidence head - deeper with explicit calibration structure
        self.confidence_features = nn.Sequential(
            nn.Linear(meta_dim + first_order_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
        )

        # Pre-sigmoid logit output (for temperature scaling)
        self.confidence_logit = nn.Linear(32, 1)

        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        # Auxiliary uncertainty head (aleatoric + epistemic)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus(),  # Positive uncertainty
        )

    def forward(
        self,
        first_order_state: torch.Tensor,
        telemetry: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Meta-representation
        meta_rep = self.meta_encoder(first_order_state)

        # Modulate by telemetry
        telem_gate = self.telemetry_gate(telemetry)
        meta_rep_embodied = meta_rep * telem_gate

        # Self-model prediction
        predicted_first_order = self.first_order_predictor(meta_rep_embodied)

        # Confidence computation with temperature scaling
        # Concatenate meta and first-order for confidence estimation
        conf_input = torch.cat([meta_rep_embodied, first_order_state], dim=-1)
        conf_features = self.confidence_features(conf_input)

        # Logit before sigmoid (for calibration)
        confidence_logit = self.confidence_logit(conf_features)

        # Temperature-scaled sigmoid
        confidence = torch.sigmoid(confidence_logit / self.temperature)

        # Uncertainty estimate
        uncertainty = self.uncertainty_head(conf_features)

        return {
            'meta_representation': meta_rep_embodied,
            'predicted_first_order': predicted_first_order,
            'confidence': confidence.squeeze(-1),
            'confidence_logit': confidence_logit.squeeze(-1),
            'uncertainty': uncertainty.squeeze(-1),
            'raw_meta': meta_rep,
        }


class CalibratedHOTModel(nn.Module):
    """
    Higher-Order Theory model with CALIBRATED confidence.

    Architecture:
    Input -> FirstOrder -> HigherOrder (calibrated) -> Task output
                              ^
                              |
                          Telemetry + Hardware Entropy
    """

    def __init__(
        self,
        hw_state_dim: int = 6,  # From z1950: hw_random, jitter, power_d, temp_d, util_d, temp_accel
        hidden_dim: int = 128,
        meta_dim: int = 64,
        telemetry_dim: int = 6,
    ):
        super().__init__()

        self.hw_encoder = nn.Sequential(
            nn.Linear(hw_state_dim, hidden_dim),
            nn.GELU(),
        )

        # First-order processing
        self.first_order = FirstOrderModule(hidden_dim, hidden_dim)

        # Higher-order (calibrated)
        self.higher_order = CalibratedHigherOrderModule(hidden_dim, meta_dim, telemetry_dim)

        # Target prediction head
        self.predictor = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        hw_state: torch.Tensor,
        telemetry: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Encode hardware state
        x = self.hw_encoder(hw_state)

        # First-order processing
        first_order_state = self.first_order(x)

        # Higher-order processing
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0).expand(x.size(0), -1)

        ho_output = self.higher_order(first_order_state, telemetry)

        # Task prediction
        prediction = self.predictor(ho_output['meta_representation']).squeeze(-1)

        if return_all:
            return {
                'prediction': prediction,
                'first_order_state': first_order_state,
                **ho_output,
            }

        return {'prediction': prediction, 'confidence': ho_output['confidence']}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Task Definition (from z1950)
# =============================================================================

class HardwareEntropyTask:
    """Task where target depends on TRUE hardware entropy."""

    def __init__(self, noise_scale=0.12):  # Higher noise for more error variance
        self.noise_scale = noise_scale
        self.w_random = 0.35
        self.w_interrupt = 0.25
        self.w_power = 0.20
        self.w_temp = 0.10
        self.w_util = 0.10
        self.y = 0.5

    def compute_target(self, hw_random: float, interrupt_jitter: float,
                       gpu: Dict) -> Tuple[float, Dict]:
        signal = (
            self.w_random * hw_random +
            self.w_interrupt * interrupt_jitter +
            self.w_power * (gpu['power_deriv'] + 1) / 2 +
            self.w_temp * (gpu['temp_accel'] + 1) / 2 +
            self.w_util * (gpu['util_deriv'] + 1) / 2
        )

        noise = np.random.normal(0, self.noise_scale)
        target = float(np.clip(signal + noise, 0, 1))
        self.y = target

        return target, {'signal': signal}


# =============================================================================
# Training Functions
# =============================================================================

def create_gpu_load(intensity: int = 2):
    """Create varying GPU load."""
    if intensity == 0:
        time.sleep(0.02)
    elif intensity == 1:
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    elif intensity == 2:
        _ = torch.randn(1000, 1000, device=DEVICE) @ torch.randn(1000, 1000, device=DEVICE)
    else:
        for _ in range(intensity):
            _ = torch.randn(1500, 1500, device=DEVICE) @ torch.randn(1500, 1500, device=DEVICE)
    torch.cuda.synchronize()


def train_phase1_task(model, entropy_src, gpu_sensor, task, n_episodes=40, steps_per_ep=60):
    """
    Phase 1: Train the task prediction head.

    Focus on accurate predictions first, before calibration.
    """
    print("\n=== PHASE 1: Task Training ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes)

    training_log = []

    for ep in range(n_episodes):
        model.train()
        ep_losses = []
        ep_pred_errors = []
        ep_self_errors = []

        for step in range(steps_per_ep):
            create_gpu_load(np.random.randint(0, 4))

            hw_random = entropy_src.read_true_random()
            jitter = entropy_src.get_interrupt_jitter()
            gpu = gpu_sensor.read()

            target, _ = task.compute_target(hw_random, jitter, gpu)

            # Hardware state tensor
            hw_tensor = torch.tensor([
                hw_random, jitter,
                gpu['power_deriv'], gpu['temp_deriv'],
                gpu['util_deriv'], gpu['temp_accel'],
            ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

            target_tensor = torch.tensor([target], dtype=torch.float32, device=DEVICE)

            # Forward
            out = model(hw_tensor, hw_tensor, return_all=True)

            # Task loss
            pred_loss = F.mse_loss(out['prediction'], target_tensor)

            # Self-model loss
            self_loss = F.mse_loss(out['predicted_first_order'], out['first_order_state'].detach())

            # Total loss (no calibration yet)
            loss = pred_loss + 0.1 * self_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_losses.append(loss.item())
            ep_pred_errors.append(pred_loss.item())
            ep_self_errors.append(self_loss.item())

        scheduler.step()

        training_log.append({
            'phase': 1,
            'episode': ep + 1,
            'loss': np.mean(ep_losses),
            'pred_error': np.mean(ep_pred_errors),
            'self_error': np.mean(ep_self_errors),
        })

        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1}/{n_episodes}: loss={np.mean(ep_losses):.4f}, "
                  f"pred={np.mean(ep_pred_errors):.4f}, self={np.mean(ep_self_errors):.4f}")

    return training_log


def train_phase2_calibration(model, entropy_src, gpu_sensor, task,
                             n_episodes=50, steps_per_ep=80):
    """
    Phase 2: Train confidence calibration.

    IMPROVED APPROACH:
    1. Use direct supervision: confidence should predict 1 - normalized_error
    2. Train confidence to directly track accuracy (soft labels, not binary)
    3. Heavy weight on correlation loss
    4. Batch processing for stable gradient estimates
    """
    print("\n=== PHASE 2: Calibration Training ===")

    # Freeze non-confidence parameters
    for name, param in model.named_parameters():
        if 'confidence' not in name and 'temperature' not in name and 'uncertainty' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Only optimize confidence-related parameters
    conf_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(conf_params, lr=2e-3)  # Higher LR

    ece_loss_fn = ECELoss(n_bins=15, temperature=0.1).to(DEVICE)

    training_log = []

    for ep in range(n_episodes):
        model.train()

        # Collect batch for calibration - larger batch for better gradient
        confidences_all = []
        soft_accuracies_all = []  # Soft targets for confidence
        binary_accuracies_all = []  # For ECE

        for step in range(steps_per_ep):
            create_gpu_load(np.random.randint(0, 4))

            hw_random = entropy_src.read_true_random()
            jitter = entropy_src.get_interrupt_jitter()
            gpu = gpu_sensor.read()

            target, _ = task.compute_target(hw_random, jitter, gpu)

            hw_tensor = torch.tensor([
                hw_random, jitter,
                gpu['power_deriv'], gpu['temp_deriv'],
                gpu['util_deriv'], gpu['temp_accel'],
            ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

            target_tensor = torch.tensor([target], dtype=torch.float32, device=DEVICE)

            out = model(hw_tensor, hw_tensor, return_all=True)

            # Compute soft accuracy: 1 - normalized_error (clipped to [0, 1])
            pred_error = torch.abs(out['prediction'].detach() - target_tensor)
            soft_accuracy = torch.clamp(1.0 - pred_error * 5, 0, 1)  # Scale error

            # Binary accuracy for ECE
            threshold = 0.05
            binary_accuracy = (pred_error < threshold).float()

            confidences_all.append(out['confidence'])
            soft_accuracies_all.append(soft_accuracy)
            binary_accuracies_all.append(binary_accuracy)

        # Stack all samples
        confidences = torch.cat(confidences_all)
        soft_accuracies = torch.cat(soft_accuracies_all)
        binary_accuracies = torch.cat(binary_accuracies_all)

        # LOSS 1: Direct supervision - confidence should match soft accuracy
        supervision_loss = F.mse_loss(confidences, soft_accuracies)

        # LOSS 2: ECE loss
        ece = ece_loss_fn(confidences, binary_accuracies)

        # LOSS 3: Correlation-based loss
        conf_centered = confidences - confidences.mean()
        acc_centered = binary_accuracies - binary_accuracies.mean()
        cov = (conf_centered * acc_centered).mean()
        conf_std = conf_centered.std() + 1e-8
        acc_std = acc_centered.std() + 1e-8
        correlation = cov / (conf_std * acc_std)
        corr_loss = 1 - correlation  # Want high positive correlation

        # LOSS 4: Sharpness - encourage variance in confidence
        variance = confidences.var()
        sharpness_loss = -torch.log(variance + 1e-8)  # Log encourages min variance

        # Total loss - heavy weight on supervision and correlation
        loss = 2.0 * supervision_loss + 0.3 * ece + 1.0 * corr_loss + 0.1 * sharpness_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(conf_params, 1.0)
        optimizer.step()

        training_log.append({
            'phase': 2,
            'episode': ep + 1,
            'supervision_loss': supervision_loss.item(),
            'ece': ece.item(),
            'correlation': correlation.item(),
            'variance': variance.item(),
        })

        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1}/{n_episodes}: sup={supervision_loss.item():.4f}, "
                  f"ECE={ece.item():.4f}, corr={correlation.item():.4f}, var={variance.item():.4f}")

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    return training_log


def collect_calibration_data(model, entropy_src, gpu_sensor, task, n_samples=500):
    """Collect data for post-hoc calibration (Platt/Isotonic)."""
    print("\n=== Collecting Calibration Data ===")

    model.eval()
    confidences = []
    accuracies = []

    with torch.no_grad():
        for _ in range(n_samples):
            create_gpu_load(np.random.randint(0, 4))

            hw_random = entropy_src.read_true_random()
            jitter = entropy_src.get_interrupt_jitter()
            gpu = gpu_sensor.read()

            target, _ = task.compute_target(hw_random, jitter, gpu)

            hw_tensor = torch.tensor([
                hw_random, jitter,
                gpu['power_deriv'], gpu['temp_deriv'],
                gpu['util_deriv'], gpu['temp_accel'],
            ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

            target_tensor = torch.tensor([target], dtype=torch.float32, device=DEVICE)

            out = model(hw_tensor, hw_tensor, return_all=True)

            pred_error = torch.abs(out['prediction'] - target_tensor).item()
            accuracy = 1.0 if pred_error < 0.05 else 0.0  # Tighter threshold

            confidences.append(out['confidence'].item())
            accuracies.append(accuracy)

    return np.array(confidences), np.array(accuracies)


def evaluate_calibration(model, entropy_src, gpu_sensor, task,
                        platt=None, isotonic=None, n_episodes=30, steps_per_ep=80):
    """
    Comprehensive calibration evaluation.

    Tests raw confidence and post-hoc calibrated confidences.
    """
    print("\n=== Calibration Evaluation ===")

    model.eval()

    raw_confidences = []
    platt_confidences = []
    isotonic_confidences = []
    accuracies = []
    predictions = []
    targets = []

    with torch.no_grad():
        for ep in range(n_episodes):
            for step in range(steps_per_ep):
                create_gpu_load(np.random.randint(0, 4))

                hw_random = entropy_src.read_true_random()
                jitter = entropy_src.get_interrupt_jitter()
                gpu = gpu_sensor.read()

                target, _ = task.compute_target(hw_random, jitter, gpu)

                hw_tensor = torch.tensor([
                    hw_random, jitter,
                    gpu['power_deriv'], gpu['temp_deriv'],
                    gpu['util_deriv'], gpu['temp_accel'],
                ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

                out = model(hw_tensor, hw_tensor, return_all=True)

                pred = out['prediction'].item()
                conf = out['confidence'].item()

                pred_error = abs(pred - target)
                accuracy = 1.0 if pred_error < 0.05 else 0.0  # Tighter threshold

                raw_confidences.append(conf)
                accuracies.append(accuracy)
                predictions.append(pred)
                targets.append(target)

                # Post-hoc calibration
                if platt is not None and platt.fitted:
                    platt_confidences.append(platt.calibrate(np.array([conf]))[0])
                else:
                    platt_confidences.append(conf)

                if isotonic is not None and isotonic.fitted:
                    isotonic_confidences.append(isotonic.calibrate(np.array([conf]))[0])
                else:
                    isotonic_confidences.append(conf)

    raw_confidences = np.array(raw_confidences)
    platt_confidences = np.array(platt_confidences)
    isotonic_confidences = np.array(isotonic_confidences)
    accuracies = np.array(accuracies)
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    results = {}

    # Raw confidence
    raw_ece = compute_ece(raw_confidences, accuracies)
    raw_mce = compute_mce(raw_confidences, accuracies)
    raw_corr = np.corrcoef(raw_confidences, accuracies)[0, 1]
    if np.isnan(raw_corr):
        raw_corr = 0.0

    results['raw'] = {
        'ece': raw_ece,
        'mce': raw_mce,
        'correlation': raw_corr,
        'mean_confidence': raw_confidences.mean(),
        'mean_accuracy': accuracies.mean(),
        'reliability_diagram': compute_reliability_diagram(raw_confidences, accuracies),
    }

    # Platt calibrated
    platt_ece = compute_ece(platt_confidences, accuracies)
    platt_corr = np.corrcoef(platt_confidences, accuracies)[0, 1]
    if np.isnan(platt_corr):
        platt_corr = 0.0

    results['platt'] = {
        'ece': platt_ece,
        'correlation': platt_corr,
        'mean_confidence': platt_confidences.mean(),
    }

    # Isotonic calibrated
    iso_ece = compute_ece(isotonic_confidences, accuracies)
    iso_corr = np.corrcoef(isotonic_confidences, accuracies)[0, 1]
    if np.isnan(iso_corr):
        iso_corr = 0.0

    results['isotonic'] = {
        'ece': iso_ece,
        'correlation': iso_corr,
        'mean_confidence': isotonic_confidences.mean(),
    }

    # Prediction accuracy
    pred_mse = np.mean((predictions - targets) ** 2)
    results['prediction_mse'] = pred_mse

    return results


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 70)
    print("z1962: Calibrated Higher-Order Theory (HOT) Confidence")
    print("Fixing negative confidence calibration from z1913")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        'experiment': 'z1962_calibrated_hot',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'problem': 'z1913 showed HOT confidence calibration = -0.395 (NEGATIVE)',
        'target': 'Confidence-accuracy correlation > 0 (POSITIVE)',
    }

    # Initialize hardware
    print("\n=== Hardware Initialization ===")
    entropy_src = TrueHardwareEntropy()
    gpu_sensor = GPUSensor()
    task = HardwareEntropyTask(noise_scale=0.03)

    # Warm up
    print("  Warming up sensors...")
    for _ in range(20):
        create_gpu_load(2)
        entropy_src.read_true_random()
        gpu_sensor.read()

    # Create model
    model = CalibratedHOTModel(
        hw_state_dim=6,
        hidden_dim=128,
        meta_dim=64,
        telemetry_dim=6,
    ).to(DEVICE)

    print(f"  Model parameters: {model.count_parameters():,}")
    results['model_params'] = model.count_parameters()

    # PHASE 1: Task training
    phase1_log = train_phase1_task(model, entropy_src, gpu_sensor, task,
                                   n_episodes=40, steps_per_ep=60)
    results['phase1_training'] = phase1_log

    # PHASE 2: Calibration training with ECE loss
    phase2_log = train_phase2_calibration(model, entropy_src, gpu_sensor, task,
                                          n_episodes=30, steps_per_ep=60)
    results['phase2_training'] = phase2_log

    # Collect data for post-hoc calibration
    cal_confidences, cal_accuracies = collect_calibration_data(
        model, entropy_src, gpu_sensor, task, n_samples=500
    )

    # Fit post-hoc calibrators
    print("\n=== Fitting Post-hoc Calibrators ===")
    platt = PlattScaling()
    isotonic = IsotonicCalibration()

    # Split data
    split_idx = int(0.7 * len(cal_confidences))
    train_conf, val_conf = cal_confidences[:split_idx], cal_confidences[split_idx:]
    train_acc, val_acc = cal_accuracies[:split_idx], cal_accuracies[split_idx:]

    platt.fit(train_conf, train_acc)
    isotonic.fit(train_conf, train_acc)
    print("  Platt and Isotonic calibrators fitted")

    # Evaluate calibration
    eval_results = evaluate_calibration(
        model, entropy_src, gpu_sensor, task,
        platt=platt, isotonic=isotonic,
        n_episodes=30, steps_per_ep=80
    )
    results['evaluation'] = eval_results

    # Print results
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<15} | {'ECE':>10} | {'Correlation':>12} | {'Mean Conf':>10}")
    print("-" * 55)
    print(f"{'Raw':<15} | {eval_results['raw']['ece']:>10.4f} | "
          f"{eval_results['raw']['correlation']:>12.4f} | "
          f"{eval_results['raw']['mean_confidence']:>10.4f}")
    print(f"{'Platt':<15} | {eval_results['platt']['ece']:>10.4f} | "
          f"{eval_results['platt']['correlation']:>12.4f} | "
          f"{eval_results['platt']['mean_confidence']:>10.4f}")
    print(f"{'Isotonic':<15} | {eval_results['isotonic']['ece']:>10.4f} | "
          f"{eval_results['isotonic']['correlation']:>12.4f} | "
          f"{eval_results['isotonic']['mean_confidence']:>10.4f}")

    print(f"\nPrediction MSE: {eval_results['prediction_mse']:.4f}")
    print(f"Mean Accuracy: {eval_results['raw']['mean_accuracy']:.4f}")

    # Verdict
    raw_corr = eval_results['raw']['correlation']
    platt_corr = eval_results['platt']['correlation']
    iso_corr = eval_results['isotonic']['correlation']
    best_corr = max(raw_corr, platt_corr, iso_corr)

    tests = {
        'T1_raw_positive': raw_corr > 0,
        'T2_platt_positive': platt_corr > 0,
        'T3_isotonic_positive': iso_corr > 0,
        'T4_best_significant': best_corr > 0.1,
        'T5_ece_reasonable': eval_results['raw']['ece'] < 0.2,
    }

    tests_passed = sum(tests.values())
    results['tests'] = {k: str(v) for k, v in tests.items()}
    results['tests_passed'] = tests_passed
    results['tests_total'] = len(tests)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    print(f"\nz1913 baseline: confidence-accuracy correlation = -0.395")
    print(f"z1962 raw:      confidence-accuracy correlation = {raw_corr:.4f}")
    print(f"z1962 platt:    confidence-accuracy correlation = {platt_corr:.4f}")
    print(f"z1962 isotonic: confidence-accuracy correlation = {iso_corr:.4f}")

    improvement = raw_corr - (-0.395)
    print(f"\nImprovement over z1913: {improvement:+.4f}")

    if raw_corr > 0 and best_corr > 0.1:
        verdict = "CALIBRATION FIXED - POSITIVE CORRELATION ACHIEVED"
        print(f"\n[SUCCESS] {verdict}")
    elif raw_corr > 0 or best_corr > 0:
        verdict = "PARTIAL SUCCESS - POSITIVE BUT WEAK CORRELATION"
        print(f"\n[PARTIAL] {verdict}")
    else:
        verdict = "CALIBRATION STILL NEEDS WORK"
        print(f"\n[NEEDS WORK] {verdict}")

    results['verdict'] = verdict
    results['improvement_over_z1913'] = improvement
    results['z1913_baseline'] = -0.395

    # Reliability diagram data
    print("\nReliability Diagram (Raw):")
    rd = eval_results['raw']['reliability_diagram']
    for i, (center, acc, conf, count) in enumerate(zip(
        rd['bin_centers'], rd['bin_accuracies'],
        rd['bin_confidences'], rd['bin_counts']
    )):
        gap = abs(acc - conf)
        bar = '*' * int(count / 10) if count > 0 else ''
        print(f"  Bin {center:.2f}: acc={acc:.2f}, conf={conf:.2f}, gap={gap:.2f} {bar}")

    # Cleanup
    entropy_src.close()

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'z1962_calibrated_hot.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
