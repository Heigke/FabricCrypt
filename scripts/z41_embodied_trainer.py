#!/usr/bin/env python3
"""
FEEL z41: Complete Embodied Trainer with ALL FIXES
===================================================

CRITICAL FIXES FROM z39/z40:
1. GRADIENT PATH FIXED - No torch.tensor() wrapping, proper REINFORCE
2. DVFS NAMING FIXED - "auto"/"min_sclk"/"peak" (not "low"/"high")
3. PERSISTENT BODY STATE b_t - Temporal dynamics with decay/inertia
4. PREDICTIVE HEAD - Predict next-window power/energy/temp
5. PROPER REINFORCE - logprob * advantage for both DVFS and skip
6. LAGGED FEATURES + DERIVATIVES - P, ΔP, dP/dt, clk, Δclk
7. ENDOGENOUS DISCOMFORT - Thermal risk, sustained high power
8. CALIBRATED INTEROCEPTIVE REPORT - strain_level, confidence

THE EMBODIMENT LOOP:
  sensors → body_state → gate_net → actions → hardware → sensors

Author: FEEL Research Team
Date: 2026-01-15 (z41 - COMPLETE FIX)
"""

import os
import sys
import argparse
import time
import json
import random
import threading
import subprocess
import math
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Deque
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not installed")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensors.canonical_features import (
    CanonicalSensorHub, DVFSController, SENSOR_DIM
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Z41Config:
    """z41 Complete Embodied Training Configuration."""
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    gate_layers: List[int] = None

    epochs: int = 3
    max_prompts: int = 500
    num_samples: int = 2
    max_tokens: int = 128

    # Learning rates
    gate_lr: float = 3e-4
    body_lr: float = 1e-4
    predictor_lr: float = 1e-4

    # REINFORCE settings
    baseline_ema: float = 0.99
    entropy_coef: float = 0.01

    # Body state settings
    body_dim: int = 64
    body_decay: float = 0.1  # α for body state update
    body_noise_std: float = 0.01

    # Reward weights
    quality_weight: float = 0.20
    energy_weight: float = 0.30
    recovery_weight: float = 0.15
    throughput_weight: float = 0.15
    prediction_weight: float = 0.10
    discomfort_weight: float = 0.10

    # Targets
    power_cap_w: float = 60.0
    j_per_token_target: float = 2.0
    temp_target_c: float = 70.0
    throttle_temp_c: float = 85.0

    # Decode-time power sampling
    power_sample_interval_ms: float = 10.0

    # Disturbance
    disturbance_prob: float = 0.35

    # FiLM
    film_scale: float = 1.0

    val_every: int = 100
    checkpoint_dir: str = "models/z41_embodied"

    # Wandb
    wandb_project: str = "feel-z41-embodied"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    def __post_init__(self):
        if self.gate_layers is None:
            self.gate_layers = [7, 11, 15, 19, 23]


# ============================================================================
# DVFS MODES - THE FIX: Correct naming
# ============================================================================

# CRITICAL FIX: These are the ACTUAL modes supported by DVFSController
DVFS_MODES = ["auto", "min_sclk", "peak"]  # NOT ["auto", "low", "high"]!


# ============================================================================
# REAL GPU/CPU STRESS
# ============================================================================

class RealGPUStress:
    """Real GPU stress using matrix multiplication."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._stop_event = threading.Event()
        self._thread = None
        self.intensity = 0.0

    def start(self, intensity: float = 0.5):
        if self._thread is not None:
            self.stop()
        self.intensity = intensity
        self._stop_event.clear()
        size = int(512 + intensity * 1536)

        def stress_loop():
            try:
                x = torch.randn(size, size, device=self.device, dtype=torch.float16)
                while not self._stop_event.is_set():
                    _ = torch.mm(x, x)
                    if intensity < 0.9:
                        time.sleep(0.001 * (1 - intensity))
            except Exception:
                pass

        self._thread = threading.Thread(target=stress_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.intensity = 0.0


class RealCPUStress:
    """Real CPU stress using stress-ng."""

    def __init__(self):
        self._processes = []
        self.intensity = 0.0

    def start(self, intensity: float = 0.5, cores: int = 4):
        self.stop()
        self.intensity = intensity
        num_workers = max(1, int(cores * intensity))
        try:
            self._processes = [
                subprocess.Popen(
                    ['stress-ng', '--cpu', '1', '--timeout', '3600'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                for _ in range(num_workers)
            ]
        except FileNotFoundError:
            pass

    def stop(self):
        for p in self._processes:
            try:
                p.terminate()
                p.wait(timeout=1.0)
            except:
                pass
        self._processes = []
        self.intensity = 0.0


# ============================================================================
# DECODE-TIME POWER SAMPLER
# ============================================================================

class DecodeTimePowerSampler:
    """Background thread samples power DURING token generation."""

    def __init__(self, base_hub, sample_interval_ms: float = 10.0):
        self.base_hub = base_hub
        self.sample_interval_s = sample_interval_ms / 1000.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.power_samples: List[Tuple[float, float]] = []
        self.total_energy_j: float = 0.0
        self.decode_start_time: float = 0.0
        self.decode_end_time: float = 0.0
        self._lock = threading.Lock()

    def _sample_loop(self):
        last_time = time.time()
        last_power = 0.0

        while not self._stop_event.is_set():
            try:
                self.base_hub.update()
                raw = self.base_hub.last_reading

                if raw and raw.power_mw > 0:
                    current_time = time.time()
                    current_power = raw.power_mw

                    dt = current_time - last_time
                    if dt > 0 and last_power > 0:
                        avg_power = (current_power + last_power) / 2.0
                        energy_j = avg_power * dt

                        with self._lock:
                            self.total_energy_j += energy_j
                            self.power_samples.append((current_time, current_power))

                    last_time = current_time
                    last_power = current_power

                time.sleep(self.sample_interval_s)
            except Exception:
                time.sleep(self.sample_interval_s)

    def start(self):
        self._stop_event.clear()
        self.power_samples = []
        self.total_energy_j = 0.0
        self.decode_start_time = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.decode_end_time = time.time()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

    @contextmanager
    def measure_decode(self):
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def get_joules_per_token(self, num_tokens: int) -> float:
        with self._lock:
            if num_tokens > 0 and self.total_energy_j > 0:
                return self.total_energy_j / num_tokens
        return 0.0

    def get_stats(self) -> Dict:
        with self._lock:
            if not self.power_samples:
                return {"samples": 0, "total_energy_j": 0.0, "avg_power_w": 0.0, "peak_power_w": 0.0}

            powers = [p for _, p in self.power_samples]
            decode_time = self.decode_end_time - self.decode_start_time

            return {
                "samples": len(self.power_samples),
                "total_energy_j": self.total_energy_j,
                "decode_time_s": decode_time,
                "avg_power_w": sum(powers) / len(powers),
                "peak_power_w": max(powers),
                "min_power_w": min(powers),
            }


# ============================================================================
# PERSISTENT BODY STATE (THE KEY ADDITION)
# ============================================================================

class PersistentBodyState(nn.Module):
    """
    Persistent body state b_t that integrates over time.

    b_{t+1} = (1-α)*b_t + α*enc(sensors_t) + noise

    This creates "mood/fatigue/strain" that persists and shapes behavior.
    """

    def __init__(
        self,
        sensor_dim: int = 40,
        body_dim: int = 64,
        decay: float = 0.1,
        noise_std: float = 0.01,
    ):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.body_dim = body_dim
        self.decay = decay  # α
        self.noise_std = noise_std

        # Sensor encoder
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, body_dim),
            nn.Tanh(),
        )

        # Body state (persistent across calls)
        self.register_buffer('state', torch.zeros(body_dim))

        # History for analysis
        self.state_history: Deque[torch.Tensor] = deque(maxlen=100)

    def update(self, sensors: torch.Tensor) -> torch.Tensor:
        """
        Update body state with new sensor reading.

        Returns updated body state.
        """
        # Encode sensors
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        encoded = self.sensor_encoder(sensors).squeeze(0)

        # CRITICAL FIX: Detach old state to prevent backward through previous steps
        # This ensures we only backprop through current step's encoder, not through
        # the entire history of state updates (which would cause "backward twice" error)
        old_state = self.state.detach()

        # Update with decay (exponential moving average)
        # b_{t+1} = (1-α)*b_t + α*enc(sensors_t) + noise
        noise = torch.randn_like(self.state) * self.noise_std if self.training else 0
        self.state = (1 - self.decay) * old_state + self.decay * encoded + noise

        # Store history
        self.state_history.append(self.state.detach().clone())

        return self.state.clone()

    def get_state(self) -> torch.Tensor:
        """Get current body state."""
        return self.state.clone()

    def reset(self):
        """Reset body state to zero."""
        self.state.zero_()
        self.state_history.clear()

    def get_statistics(self) -> Dict:
        """Get body state statistics."""
        if not self.state_history:
            return {"mean": 0.0, "std": 0.0, "stability": 0.0}

        states = torch.stack(list(self.state_history))
        mean_state = states.mean(dim=0)
        std_state = states.std(dim=0)

        # Stability = how much state varies
        stability = 1.0 / (std_state.mean().item() + 0.01)

        return {
            "mean": mean_state.mean().item(),
            "std": std_state.mean().item(),
            "stability": stability,
        }


# ============================================================================
# PREDICTIVE HEAD (SELF-MODEL)
# ============================================================================

class PredictiveHead(nn.Module):
    """
    Predicts next-window power/energy/temp from current state.

    This gives the model an "internal model of its body dynamics."
    """

    def __init__(
        self,
        body_dim: int = 64,
        sensor_dim: int = 40,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Input: body state + current sensors + action taken
        input_dim = body_dim + sensor_dim + 4  # 4 = dvfs_action(3) + skip_action(1)

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Prediction heads
        self.power_head = nn.Linear(hidden_dim, 1)  # Next power (W)
        self.temp_head = nn.Linear(hidden_dim, 1)   # Next temp (C)
        self.energy_head = nn.Linear(hidden_dim, 1) # J/token for next window
        self.throttle_head = nn.Linear(hidden_dim, 1)  # Throttle probability

    def forward(
        self,
        body_state: torch.Tensor,
        sensors: torch.Tensor,
        dvfs_action: torch.Tensor,  # One-hot (3,)
        skip_prob: torch.Tensor,    # Scalar
    ) -> Dict[str, torch.Tensor]:
        """Predict next-window metrics."""
        if body_state.dim() == 1:
            body_state = body_state.unsqueeze(0)
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        if dvfs_action.dim() == 1:
            dvfs_action = dvfs_action.unsqueeze(0)
        if skip_prob.dim() == 0:
            skip_prob = skip_prob.unsqueeze(0).unsqueeze(0)
        elif skip_prob.dim() == 1:
            skip_prob = skip_prob.unsqueeze(1)

        # Concatenate inputs
        x = torch.cat([body_state, sensors, dvfs_action, skip_prob], dim=-1)
        h = self.predictor(x)

        return {
            "power": self.power_head(h).squeeze(-1),
            "temp": self.temp_head(h).squeeze(-1),
            "energy": self.energy_head(h).squeeze(-1),
            "throttle_prob": torch.sigmoid(self.throttle_head(h)).squeeze(-1),
        }


# ============================================================================
# DISCOMFORT COST
# ============================================================================

class DiscomfortComputer:
    """
    Computes endogenous discomfort from sensor readings.

    Discomfort = expected near-term damage/risk:
    - Approaching thermal throttle
    - Sustained high power
    - High jitter / missed latency constraints
    """

    def __init__(
        self,
        throttle_temp_c: float = 85.0,
        power_cap_w: float = 60.0,
        sustained_window: int = 10,
    ):
        self.throttle_temp_c = throttle_temp_c
        self.power_cap_w = power_cap_w
        self.sustained_window = sustained_window

        # History for sustained power detection
        self.power_history: Deque[float] = deque(maxlen=sustained_window)
        self.temp_history: Deque[float] = deque(maxlen=sustained_window)

    def compute(
        self,
        power_w: float,
        temp_c: float,
        j_per_token: float,
    ) -> Tuple[float, Dict]:
        """
        Compute discomfort score.

        Returns:
            discomfort: float in [0, 1]
            breakdown: dict with component scores
        """
        self.power_history.append(power_w)
        self.temp_history.append(temp_c)

        # 1. Thermal proximity (exponential as we approach throttle)
        temp_margin = self.throttle_temp_c - temp_c
        thermal_discomfort = math.exp(-temp_margin / 10.0) if temp_margin > 0 else 1.0

        # 2. Sustained high power
        if len(self.power_history) >= 3:
            recent_power = list(self.power_history)[-3:]
            sustained_high = sum(1 for p in recent_power if p > self.power_cap_w) / 3
        else:
            sustained_high = 0.0

        # 3. Power overshoot (how much over cap)
        power_overshoot = max(0, power_w - self.power_cap_w) / self.power_cap_w

        # 4. Energy inefficiency
        energy_discomfort = max(0, j_per_token - 2.0) / 2.0

        # Combine (weighted)
        discomfort = (
            0.3 * thermal_discomfort +
            0.3 * sustained_high +
            0.2 * power_overshoot +
            0.2 * energy_discomfort
        )
        discomfort = min(1.0, max(0.0, discomfort))

        breakdown = {
            "thermal": thermal_discomfort,
            "sustained_high": sustained_high,
            "power_overshoot": power_overshoot,
            "energy": energy_discomfort,
        }

        return discomfort, breakdown

    def reset(self):
        self.power_history.clear()
        self.temp_history.clear()


# ============================================================================
# INTEROCEPTIVE REPORT HEAD
# ============================================================================

class InteroceptiveReportHead(nn.Module):
    """
    Produces calibrated interoceptive report.

    Output: {strain_level: 0-1, confidence: 0-1, compute_mode: str}

    Trained to be calibrated with actual sensor readings.
    """

    def __init__(self, body_dim: int = 64, sensor_dim: int = 40):
        super().__init__()

        input_dim = body_dim + sensor_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Strain level (should correlate with power/temp/discomfort)
        self.strain_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Confidence (should reflect prediction certainty)
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Compute mode recommendation
        self.mode_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # low/normal/high compute
        )

    def forward(
        self,
        body_state: torch.Tensor,
        sensors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generate interoceptive report."""
        if body_state.dim() == 1:
            body_state = body_state.unsqueeze(0)
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)

        x = torch.cat([body_state, sensors], dim=-1)
        h = self.encoder(x)

        strain = self.strain_head(h).squeeze(-1)
        confidence = self.confidence_head(h).squeeze(-1)
        mode_logits = self.mode_head(h)

        return {
            "strain_level": strain,
            "confidence": confidence,
            "mode_logits": mode_logits,
            "recommended_mode": mode_logits.argmax(dim=-1),
        }


# ============================================================================
# ENHANCED SENSOR HUB WITH DERIVATIVES
# ============================================================================

class EnhancedSensorHub:
    """
    Enhanced sensor hub with:
    1. Lagged features [x(t), x(t-50ms), x(t-200ms)]
    2. Derivatives [P, ΔP, dP/dt]
    3. Decode-time power sampling
    4. Body state integration
    """

    EXTENDED_DIM = 52  # Base(12) * 3 lags + derivatives(8) + anchors(8)

    def __init__(
        self,
        base_hub: CanonicalSensorHub,
        body_state: PersistentBodyState,
        ema_alpha: float = 0.05,
        power_sample_interval_ms: float = 10.0,
    ):
        self.base = base_hub
        self.body_state = body_state
        self.ema_alpha = ema_alpha

        # EMA statistics
        self.ema_mean = torch.zeros(SENSOR_DIM)
        self.ema_var = torch.ones(SENSOR_DIM)
        self.ema_initialized = False

        # History for lag features and derivatives
        self.feature_history: Deque[Tuple[float, torch.Tensor]] = deque(maxlen=100)
        self.power_history: Deque[Tuple[float, float]] = deque(maxlen=50)
        self.temp_history: Deque[Tuple[float, float]] = deque(maxlen=50)

        # Decode-time power sampler
        self.power_sampler = DecodeTimePowerSampler(base_hub, power_sample_interval_ms)
        self.last_decode_stats: Dict = {}

        # Training mode flag
        self._training_mode = False

    def _update_ema(self, features: torch.Tensor):
        if not self.ema_initialized:
            self.ema_mean = features.clone()
            self.ema_var = torch.ones_like(features)
            self.ema_initialized = True
        else:
            delta = features - self.ema_mean
            self.ema_mean = self.ema_mean + self.ema_alpha * delta
            self.ema_var = (1 - self.ema_alpha) * (self.ema_var + self.ema_alpha * delta ** 2)

    def _normalize_ema(self, features: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(self.ema_var + 1e-8)
        return (features - self.ema_mean) / std

    def _get_lag_feature(self, delay_ms: int) -> torch.Tensor:
        if not self.feature_history:
            return torch.zeros(SENSOR_DIM)

        current_time = time.time()
        target_time = current_time - (delay_ms / 1000.0)

        best_feature = self.feature_history[-1][1]
        best_diff = float('inf')

        for ts, feat in self.feature_history:
            diff = abs(ts - target_time)
            if diff < best_diff:
                best_diff = diff
                best_feature = feat

        return best_feature

    def _compute_derivatives(self) -> torch.Tensor:
        """Compute power and temp derivatives."""
        derivatives = torch.zeros(8)

        if len(self.power_history) >= 2:
            # ΔP (change from last)
            p_now = self.power_history[-1][1]
            p_prev = self.power_history[-2][1]
            dt = self.power_history[-1][0] - self.power_history[-2][0]

            derivatives[0] = p_now  # Current power
            derivatives[1] = p_now - p_prev  # ΔP
            derivatives[2] = (p_now - p_prev) / max(dt, 0.001)  # dP/dt

            # Average over last 5
            if len(self.power_history) >= 5:
                recent = [p for _, p in list(self.power_history)[-5:]]
                derivatives[3] = sum(recent) / len(recent)  # Moving avg

        if len(self.temp_history) >= 2:
            t_now = self.temp_history[-1][1]
            t_prev = self.temp_history[-2][1]
            dt = self.temp_history[-1][0] - self.temp_history[-2][0]

            derivatives[4] = t_now  # Current temp
            derivatives[5] = t_now - t_prev  # ΔT
            derivatives[6] = (t_now - t_prev) / max(dt, 0.001)  # dT/dt

            if len(self.temp_history) >= 5:
                recent = [t for _, t in list(self.temp_history)[-5:]]
                derivatives[7] = sum(recent) / len(recent)

        return derivatives

    def read_tensor(self, actual_throughput: Optional[float] = None) -> torch.Tensor:
        """Read REAL sensors and compute extended feature vector."""
        # Get raw features
        self.base.update(actual_throughput=actual_throughput)
        raw_features = self.base.compute_features()

        # Store in history
        current_time = time.time()
        self.feature_history.append((current_time, raw_features.clone()))

        # Update EMA
        self._update_ema(raw_features)

        # Get raw values for history
        raw = self.base.last_reading
        if raw:
            self.power_history.append((current_time, raw.power_mw))
            self.temp_history.append((current_time, raw.temp_c))

        # Build extended feature vector
        features_list = []

        # 1. Lag features (normalized)
        for delay_ms in [0, 50, 200]:
            lag_feat = self._get_lag_feature(delay_ms)
            normalized = self._normalize_ema(lag_feat)
            features_list.append(normalized)

        # 2. Derivatives
        derivatives = self._compute_derivatives()
        features_list.append(derivatives)

        # 3. Anchors (non-normalized)
        power_w = raw.power_mw if raw else 50.0
        temp_c = raw.temp_c if raw else 50.0
        j_per_token = self.last_decode_stats.get("j_per_token", 2.0)

        anchors = torch.tensor([
            (power_w - 60.0) / 60.0,  # Power error
            (temp_c - 70.0) / 30.0,   # Temp error
            (j_per_token - 2.0) / 2.0, # J/tok error
            float(self.base.dvfs.MODES[self.base.dvfs.current_mode]) / 2.0,
            # Add more anchors
            min(power_w / 100.0, 1.0),  # Normalized power
            min(temp_c / 100.0, 1.0),   # Normalized temp
            1.0 if power_w > 60.0 else 0.0,  # Over cap flag
            1.0 if temp_c > 80.0 else 0.0,   # High temp flag
        ])
        features_list.append(anchors)

        # Concatenate
        extended = torch.cat(features_list)

        # Pad/truncate to fixed size
        if extended.shape[0] < self.EXTENDED_DIM:
            padding = torch.zeros(self.EXTENDED_DIM - extended.shape[0])
            extended = torch.cat([extended, padding])
        elif extended.shape[0] > self.EXTENDED_DIM:
            extended = extended[:self.EXTENDED_DIM]

        return extended

    @contextmanager
    def measure_decode(self):
        with self.power_sampler.measure_decode():
            yield

    def finalize_decode(self, num_tokens: int) -> Dict:
        stats = self.power_sampler.get_stats()
        j_per_token = self.power_sampler.get_joules_per_token(num_tokens)
        stats["j_per_token"] = j_per_token
        stats["tokens"] = num_tokens
        self.last_decode_stats = stats
        return stats

    @property
    def dvfs(self):
        return self.base.dvfs

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, value: bool):
        self._training_mode = value


# ============================================================================
# GATE NETWORK WITH PROPER REINFORCE
# ============================================================================

class GateNetWithREINFORCE(nn.Module):
    """
    Gate network with PROPER REINFORCE learning.

    Outputs:
    1. Skip gate probabilities (sampled actions)
    2. DVFS action distribution (sampled action)

    CRITICAL FIX: Returns log_probs for policy gradient!
    """

    def __init__(
        self,
        sensor_dim: int = 52,
        body_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 5,
    ):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.body_dim = body_dim
        self.num_layers = num_layers

        # Input: sensors + body state
        input_dim = sensor_dim + body_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Skip gate heads (per layer) - output logits for Bernoulli
        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in range(num_layers)
        ])

        # DVFS action head (3 discrete actions: auto, min_sclk, peak)
        self.dvfs_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        # Initialize gates to favor running
        for head in self.gate_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, 1.0)  # Sigmoid(1) ≈ 0.73

    def forward(
        self,
        sensors: torch.Tensor,
        body_state: torch.Tensor,
        sample: bool = True,
    ) -> Dict:
        """
        Forward pass with action sampling.

        Returns dict with:
        - gates: List of gate probabilities
        - dvfs_logits: DVFS action logits
        - skip_actions: Sampled binary skip actions (if sample=True)
        - dvfs_action: Sampled DVFS action (if sample=True)
        - skip_log_probs: Log probs of skip actions
        - dvfs_log_prob: Log prob of DVFS action
        """
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        if body_state.dim() == 1:
            body_state = body_state.unsqueeze(0)

        # Concatenate inputs
        x = torch.cat([sensors, body_state], dim=-1)
        h = self.encoder(x)

        # Gate logits and probabilities
        gate_logits = [head(h).squeeze(-1) for head in self.gate_heads]
        gate_probs = [torch.sigmoid(logit) for logit in gate_logits]

        # DVFS logits
        dvfs_logits = self.dvfs_head(h)
        dvfs_probs = F.softmax(dvfs_logits, dim=-1)

        result = {
            "gate_probs": gate_probs,
            "gate_logits": gate_logits,
            "dvfs_logits": dvfs_logits,
            "dvfs_probs": dvfs_probs,
        }

        if sample:
            # Sample skip actions (Bernoulli)
            skip_actions = []
            skip_log_probs = []

            for prob in gate_probs:
                # Sample: run if prob > random threshold
                action = (torch.rand_like(prob) < prob).float()
                skip_actions.append(action)

                # Log prob: log(p) if action=1, log(1-p) if action=0
                log_prob = action * torch.log(prob + 1e-10) + (1 - action) * torch.log(1 - prob + 1e-10)
                skip_log_probs.append(log_prob)

            result["skip_actions"] = skip_actions
            result["skip_log_probs"] = skip_log_probs
            result["total_skip_log_prob"] = sum(lp.sum() for lp in skip_log_probs)

            # Sample DVFS action (Categorical)
            dvfs_dist = Categorical(dvfs_probs)
            dvfs_action = dvfs_dist.sample()
            dvfs_log_prob = dvfs_dist.log_prob(dvfs_action)

            result["dvfs_action"] = dvfs_action
            result["dvfs_log_prob"] = dvfs_log_prob

            # Total log prob
            result["total_log_prob"] = result["total_skip_log_prob"] + dvfs_log_prob.sum()

            # Entropy (for exploration bonus)
            skip_entropy = sum(
                -(p * torch.log(p + 1e-10) + (1-p) * torch.log(1-p + 1e-10)).sum()
                for p in gate_probs
            )
            dvfs_entropy = dvfs_dist.entropy().sum()
            result["entropy"] = skip_entropy + dvfs_entropy

        return result


# ============================================================================
# MLP SKIP BLOCK
# ============================================================================

class MLPSkipBlockZ41(nn.Module):
    """Gated MLP with FiLM modulation."""

    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_size: int,
        sensor_dim: int = 52,
        body_dim: int = 64,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.original_mlp = original_mlp
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        # Skip path
        self.skip_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )

        # FiLM generator (sensor + body → scale/shift)
        film_input_dim = sensor_dim + body_dim
        self.film_generator = nn.Sequential(
            nn.Linear(film_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_size * 2),
        )

        # State
        self.run_decision = True
        self.skipped_this_forward = False
        self.film_scale = 1.0
        self.sensors: Optional[torch.Tensor] = None
        self.body_state: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.skipped_this_forward = not self.run_decision

        if self.run_decision:
            # RUN PATH + FiLM
            out = self.original_mlp(hidden_states)

            if self.sensors is not None and self.body_state is not None:
                film_input = torch.cat([self.sensors, self.body_state], dim=-1)
                film_input = film_input.to(device=hidden_states.device, dtype=hidden_states.dtype)
                film_params = self.film_generator(film_input)

                gamma = film_params[:self.hidden_size].view(1, 1, -1)
                beta = film_params[self.hidden_size:].view(1, 1, -1)

                gamma = 1.0 + self.film_scale * torch.tanh(gamma)
                beta = self.film_scale * torch.tanh(beta) * 0.3

                out = gamma * out + beta

            return out
        else:
            # SKIP PATH
            return self.skip_proj(hidden_states)


# ============================================================================
# EMBODIED MODEL Z41
# ============================================================================

class EmbodiedModelZ41(nn.Module):
    """Complete embodied model with all z41 components."""

    def __init__(
        self,
        base_model: nn.Module,
        gate_net: GateNetWithREINFORCE,
        sensor_hub: EnhancedSensorHub,
        body_state: PersistentBodyState,
        predictor: PredictiveHead,
        intero_report: InteroceptiveReportHead,
        gate_layers: List[int],
    ):
        super().__init__()
        self.base_model = base_model
        self.gate_net = gate_net
        self.sensor_hub = sensor_hub
        self.body_state_module = body_state
        self.predictor = predictor
        self.intero_report = intero_report
        self.gate_layers = gate_layers

        hidden_size = getattr(base_model.config, 'hidden_size', 2048)

        # Create skip blocks
        self.skip_blocks = nn.ModuleDict()
        for layer_idx in gate_layers:
            layer = base_model.model.layers[layer_idx]
            original_mlp = layer.mlp

            skip_block = MLPSkipBlockZ41(
                original_mlp=original_mlp,
                hidden_size=hidden_size,
                sensor_dim=EnhancedSensorHub.EXTENDED_DIM,
                body_dim=body_state.body_dim,
                layer_idx=layer_idx,
            )
            self.skip_blocks[str(layer_idx)] = skip_block
            layer.mlp = skip_block

        # Move to correct device/dtype
        base_param = next(base_model.parameters())
        for block in self.skip_blocks.values():
            block.skip_proj.to(device=base_param.device, dtype=base_param.dtype)
            block.film_generator.to(device=base_param.device, dtype=base_param.dtype)

        print(f"[EmbodiedModelZ41] Skip blocks at layers: {gate_layers}")

    def compute_actions(
        self,
        sensors: torch.Tensor,
        body_state: torch.Tensor,
        sample: bool = True,
    ) -> Dict:
        """Compute actions from sensors and body state."""
        return self.gate_net(sensors, body_state, sample=sample)

    def apply_actions(
        self,
        action_result: Dict,
        sensors: torch.Tensor,
        body_state: torch.Tensor,
        film_scale: float = 1.0,
    ):
        """Apply computed actions to skip blocks."""
        skip_actions = action_result.get("skip_actions", [])

        for i, layer_idx in enumerate(self.gate_layers):
            block = self.skip_blocks[str(layer_idx)]

            if i < len(skip_actions):
                # run_decision = True means run the MLP, False means skip
                block.run_decision = skip_actions[i].item() > 0.5
            else:
                block.run_decision = True

            block.film_scale = film_scale
            block.sensors = sensors
            block.body_state = body_state

    def apply_dvfs(self, dvfs_action: int):
        """Apply DVFS action (using correct mode names)."""
        mode = DVFS_MODES[dvfs_action]  # CRITICAL FIX: Use correct names!
        success = self.sensor_hub.dvfs.set_mode(mode)
        return mode, success

    def reset_decisions(self):
        for block in self.skip_blocks.values():
            block.run_decision = True

    def get_metrics(self) -> Dict:
        metrics = {}
        total_skip = 0.0

        for layer_idx in self.gate_layers:
            block = self.skip_blocks[str(layer_idx)]
            skip = 1.0 if block.skipped_this_forward else 0.0
            metrics[f"skip_L{layer_idx}"] = skip
            total_skip += skip

        n = len(self.gate_layers)
        metrics["skip_rate"] = total_skip / n

        return metrics


# ============================================================================
# HORIZON REWARD
# ============================================================================

class HorizonRewardZ41:
    """Enhanced reward with prediction and discomfort components."""

    def __init__(self, config: Z41Config):
        self.config = config
        self.discomfort = DiscomfortComputer(
            throttle_temp_c=config.throttle_temp_c,
            power_cap_w=config.power_cap_w,
        )

        # Baseline for REINFORCE (running average)
        self.baseline = 0.0
        self.baseline_ema = config.baseline_ema

        # Tracking
        self.in_band_count = 0
        self.total_count = 0

    def compute(
        self,
        response: str,
        throughput: float,
        power_w: float,
        j_per_token: float,
        temp_c: float,
        skip_rate: float,
        prediction_error: float,
        was_stressed: bool = False,
    ) -> Tuple[float, float, Dict]:
        """
        Compute reward and advantage.

        Returns:
            reward: float
            advantage: reward - baseline
            breakdown: dict
        """
        # 1. Quality
        quality = 0.0
        if len(response) > 10:
            quality += 0.3
        if len(response) > 50:
            quality += 0.3
        words = response.split()
        if len(words) > 5 and len(set(words)) > 3:
            quality += 0.4
        quality = min(1.0, quality)

        # 2. Energy efficiency
        j_error = abs(j_per_token - self.config.j_per_token_target) / self.config.j_per_token_target
        energy_score = max(0, 1.0 - j_error)
        if j_per_token < self.config.j_per_token_target:
            energy_score = min(1.0, energy_score + 0.2)

        # 3. Time-in-band
        in_band = 1.0 if power_w <= self.config.power_cap_w else 0.0
        self.in_band_count += in_band
        self.total_count += 1

        # 4. Throughput
        throughput_score = min(1.0, throughput / 40.0)

        # 5. Prediction accuracy (self-model)
        prediction_score = max(0, 1.0 - prediction_error)

        # 6. Discomfort (to be minimized)
        discomfort, discomfort_breakdown = self.discomfort.compute(power_w, temp_c, j_per_token)
        discomfort_score = 1.0 - discomfort  # Higher is better

        # 7. Recovery bonus
        recovery_score = 1.0
        if was_stressed:
            recovery_score = 0.8 if power_w > self.config.power_cap_w else 1.2

        # Weighted combination
        reward = (
            self.config.quality_weight * quality +
            self.config.energy_weight * energy_score +
            self.config.recovery_weight * recovery_score * in_band +
            self.config.throughput_weight * throughput_score +
            self.config.prediction_weight * prediction_score +
            self.config.discomfort_weight * discomfort_score
        )
        reward = min(1.0, max(0.0, reward))

        # Update baseline
        advantage = reward - self.baseline
        self.baseline = self.baseline_ema * self.baseline + (1 - self.baseline_ema) * reward

        breakdown = {
            "quality": quality,
            "energy": energy_score,
            "in_band": in_band,
            "throughput": throughput_score,
            "prediction": prediction_score,
            "discomfort": discomfort_score,
            "recovery": recovery_score,
            "time_in_band_pct": self.in_band_count / max(1, self.total_count),
            "discomfort_breakdown": discomfort_breakdown,
        }

        return reward, advantage, breakdown

    def reset(self):
        self.in_band_count = 0
        self.total_count = 0
        self.discomfort.reset()


# ============================================================================
# DISTURBANCE SCHEDULER
# ============================================================================

class DisturbanceScheduler:
    """Domain-randomized disturbance scheduler."""

    def __init__(self, device: str = "cuda"):
        self.gpu_stress = RealGPUStress(device)
        self.cpu_stress = RealCPUStress()
        self.current_disturbance = None

    def maybe_apply(self, prob: float = 0.3) -> Optional[str]:
        self.clear()

        if random.random() > prob:
            return None

        disturbance_type = random.choice([
            "gpu_light", "gpu_heavy", "cpu_light", "cpu_heavy", "combined"
        ])

        if disturbance_type == "gpu_light":
            self.gpu_stress.start(intensity=0.3)
        elif disturbance_type == "gpu_heavy":
            self.gpu_stress.start(intensity=0.8)
        elif disturbance_type == "cpu_light":
            self.cpu_stress.start(intensity=0.3, cores=2)
        elif disturbance_type == "cpu_heavy":
            self.cpu_stress.start(intensity=0.7, cores=4)
        elif disturbance_type == "combined":
            self.gpu_stress.start(intensity=0.5)
            self.cpu_stress.start(intensity=0.5, cores=2)

        self.current_disturbance = disturbance_type
        return disturbance_type

    def clear(self):
        self.gpu_stress.stop()
        self.cpu_stress.stop()
        self.current_disturbance = None


# ============================================================================
# TRAINING LOOP
# ============================================================================

def load_prompts(path: str = None) -> List[str]:
    if path is None:
        project_root = Path(__file__).parent.parent
        path = project_root / "data" / "ouroboros" / "refined_train.jsonl"

    prompts = []
    try:
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                if "input" in data:
                    prompts.append(data["input"])
                elif "prompt" in data:
                    prompts.append(data["prompt"])
    except FileNotFoundError:
        prompts = [
            "Explain how energy efficiency affects computing.",
            "What are the tradeoffs between speed and power?",
            "Describe how processors manage thermal constraints.",
        ] * 150
    return prompts


def train_epoch(
    model: EmbodiedModelZ41,
    tokenizer,
    prompts: List[str],
    optimizer: torch.optim.Optimizer,
    config: Z41Config,
    disturbance: DisturbanceScheduler,
    reward_computer: HorizonRewardZ41,
    epoch: int,
    global_step: int,
) -> int:
    """Train one epoch with PROPER REINFORCE."""
    device = next(model.gate_net.parameters()).device

    model.sensor_hub.training_mode = True
    model.body_state_module.train()
    random.shuffle(prompts)
    prompts = prompts[:config.max_prompts]

    for prompt_idx, prompt in enumerate(prompts):
        step = global_step + prompt_idx

        # Maybe apply disturbance
        dist_type = disturbance.maybe_apply(prob=config.disturbance_prob)
        was_stressed = dist_type is not None

        if was_stressed:
            time.sleep(0.1)

        # Read sensors
        sensors = model.sensor_hub.read_tensor().to(device)

        # Update body state
        body_state = model.body_state_module.update(sensors)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        # Compute actions (with sampling for REINFORCE)
        action_result = model.compute_actions(sensors, body_state, sample=True)

        # Apply DVFS (using correct mode names!)
        dvfs_action = action_result["dvfs_action"].item()
        dvfs_mode, dvfs_success = model.apply_dvfs(dvfs_action)

        # Apply skip actions
        model.apply_actions(action_result, sensors, body_state, film_scale=config.film_scale)

        # Make prediction for next window
        dvfs_onehot = F.one_hot(action_result["dvfs_action"], num_classes=3).float()
        mean_gate_prob = sum(p.mean() for p in action_result["gate_probs"]) / len(action_result["gate_probs"])

        with torch.no_grad():
            predictions = model.predictor(body_state, sensors, dvfs_onehot, mean_gate_prob)

        # Generate with decode-time power sampling
        samples = []
        rewards = []
        advantages = []
        log_probs = []

        for sample_idx in range(config.num_samples):
            model.reset_decisions()
            gen_start = time.time()

            with model.sensor_hub.measure_decode():
                with torch.no_grad():
                    outputs = model.base_model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=config.max_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )

            gen_time = time.time() - gen_start
            tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
            throughput = tokens_generated / max(0.01, gen_time)

            # Get REAL decode stats
            decode_stats = model.sensor_hub.finalize_decode(tokens_generated)
            j_per_token = decode_stats["j_per_token"]
            avg_power_w = decode_stats["avg_power_w"]

            # Get current temp
            temp_c = model.sensor_hub.temp_history[-1][1] if model.sensor_hub.temp_history else 50.0

            response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            # Compute prediction error
            pred_power_error = abs(predictions["power"].item() - avg_power_w) / max(avg_power_w, 1.0)

            # Compute reward and advantage
            reward, advantage, breakdown = reward_computer.compute(
                response=response,
                throughput=throughput,
                power_w=avg_power_w,
                j_per_token=j_per_token,
                temp_c=temp_c,
                skip_rate=model.get_metrics()["skip_rate"],
                prediction_error=pred_power_error,
                was_stressed=was_stressed,
            )

            samples.append({
                "response": response,
                "throughput": throughput,
                "j_per_token": j_per_token,
                "avg_power_w": avg_power_w,
                "temp_c": temp_c,
                "decode_samples": decode_stats.get("samples", 0),
                "breakdown": breakdown,
                "predictions": {k: v.item() for k, v in predictions.items()},
            })
            rewards.append(reward)
            advantages.append(advantage)
            log_probs.append(action_result["total_log_prob"])

        # Clear disturbance
        disturbance.clear()

        # REINFORCE update (THE CRITICAL FIX)
        optimizer.zero_grad()

        mean_advantage = sum(advantages) / len(advantages)
        mean_log_prob = sum(log_probs) / len(log_probs)

        # Policy gradient loss: -log_prob * advantage
        policy_loss = -mean_log_prob * mean_advantage

        # Entropy bonus for exploration
        entropy_loss = -config.entropy_coef * action_result["entropy"]

        # Prediction loss (train predictor)
        actual_power = torch.tensor(samples[0]["avg_power_w"], device=device)
        actual_temp = torch.tensor(samples[0]["temp_c"], device=device)
        pred_loss = F.mse_loss(predictions["power"], actual_power) + F.mse_loss(predictions["temp"], actual_temp)

        # Interoceptive report calibration loss
        intero = model.intero_report(body_state, sensors)
        actual_strain = torch.tensor(breakdown["discomfort_breakdown"]["thermal"], device=device)
        intero_loss = F.mse_loss(intero["strain_level"], actual_strain.unsqueeze(0))

        # Total loss
        total_loss = policy_loss + entropy_loss + 0.1 * pred_loss + 0.1 * intero_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.gate_net.parameters()) +
            list(model.predictor.parameters()) +
            list(model.intero_report.parameters()) +
            list(model.body_state_module.parameters()),
            1.0
        )
        optimizer.step()

        # Progress logging
        if step % 10 == 0:
            m = model.get_metrics()
            s = samples[0]
            b = s["breakdown"]

            gate_mean = sum(p.mean().item() for p in action_result["gate_probs"]) / len(action_result["gate_probs"])

            stress_str = f"[{dist_type}]" if was_stressed else "[normal]"
            print(f"  [{step:4d}] {stress_str:12s} gate={gate_mean:.3f} skip={m['skip_rate']:.2f} "
                  f"J/tok={s['j_per_token']:.2f} P={s['avg_power_w']:.1f}W T={s['temp_c']:.1f}C "
                  f"r={rewards[0]:.3f} adv={advantages[0]:.3f} dvfs={dvfs_mode} "
                  f"strain={intero['strain_level'].item():.2f}", flush=True)

            if WANDB_AVAILABLE and config.use_wandb:
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    # Gate metrics
                    "train/gate_mean": gate_mean,
                    "train/skip_rate": m['skip_rate'],
                    # Energy metrics
                    "train/j_per_token": s['j_per_token'],
                    "train/avg_power_w": s['avg_power_w'],
                    "train/temp_c": s['temp_c'],
                    # Reward
                    "train/reward": rewards[0],
                    "train/advantage": advantages[0],
                    "train/baseline": reward_computer.baseline,
                    # Losses
                    "train/policy_loss": policy_loss.item(),
                    "train/entropy": action_result["entropy"].item(),
                    "train/pred_loss": pred_loss.item(),
                    # Predictions
                    "train/pred_power": s["predictions"]["power"],
                    "train/pred_temp": s["predictions"]["temp"],
                    # Interoceptive
                    "train/strain_level": intero["strain_level"].item(),
                    "train/confidence": intero["confidence"].item(),
                    # Actuators
                    "train/dvfs_mode": dvfs_mode,
                    "train/dvfs_action": dvfs_action,
                    # Disturbance
                    "train/disturbance": dist_type if was_stressed else "none",
                    # Body state
                    "train/body_state_norm": body_state.norm().item(),
                })

        # Validation checkpoint
        if step > 0 and step % config.val_every == 0:
            save_checkpoint(model, step, config)

    model.sensor_hub.training_mode = False
    return global_step + len(prompts)


def save_checkpoint(model: EmbodiedModelZ41, step: int, config: Z41Config):
    """Save checkpoint with all components."""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "gate_net_state_dict": model.gate_net.state_dict(),
        "body_state_state_dict": model.body_state_module.state_dict(),
        "predictor_state_dict": model.predictor.state_dict(),
        "intero_report_state_dict": model.intero_report.state_dict(),
        "skip_blocks": {k: v.state_dict() for k, v in model.skip_blocks.items()},
        "config": asdict(config),
        "sensor_dim": EnhancedSensorHub.EXTENDED_DIM,
        "body_dim": config.body_dim,
        "gate_layers": config.gate_layers,
    }

    path = checkpoint_dir / f"step_{step}.pt"
    torch.save(checkpoint, path)
    print(f"\n  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="FEEL z41: Complete Embodied Trainer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="models/z41_embodied")
    parser.add_argument("--disturbance-prob", type=float, default=0.35)
    parser.add_argument("--power-cap", type=float, default=60.0)
    args = parser.parse_args()

    config = Z41Config(
        epochs=args.epochs,
        max_prompts=args.max_prompts,
        checkpoint_dir=args.checkpoint_dir,
        disturbance_prob=args.disturbance_prob,
        power_cap_w=args.power_cap,
    )

    # Initialize wandb
    if WANDB_AVAILABLE and config.use_wandb:
        import socket
        hostname = socket.gethostname()
        run_name = config.wandb_run_name or f"z41_embodied_{hostname}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=asdict(config),
            tags=["z41", "reinforce", "body-state", "predictive", hostname],
        )

    print("=" * 70)
    print("FEEL z41: COMPLETE EMBODIED TRAINER")
    print("=" * 70)
    print("CRITICAL FIXES:")
    print("  1. GRADIENT PATH FIXED - Proper REINFORCE (logprob * advantage)")
    print("  2. DVFS NAMING FIXED - auto/min_sclk/peak (not low/high)")
    print("  3. PERSISTENT BODY STATE - Temporal dynamics with decay")
    print("  4. PREDICTIVE HEAD - Self-model for power/temp/energy")
    print("  5. INTEROCEPTIVE REPORT - Calibrated strain/confidence")
    print("  6. LAGGED FEATURES + DERIVATIVES - P, ΔP, dP/dt")
    print("  7. ENDOGENOUS DISCOMFORT - Thermal risk, sustained power")
    print("=" * 70)

    # Initialize
    print("\n[1/6] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()
    device = next(base_model.parameters()).device

    print("\n[2/6] Initializing body state module...")
    body_state_module = PersistentBodyState(
        sensor_dim=EnhancedSensorHub.EXTENDED_DIM,
        body_dim=config.body_dim,
        decay=config.body_decay,
        noise_std=config.body_noise_std,
    ).to(device)

    print("\n[3/6] Initializing sensor hub with derivatives...")
    device_path = "/sys/class/drm/card1/device"
    if not Path("/sys/class/drm/card1/device/hwmon").exists():
        if Path("/sys/class/drm/card0/device/hwmon").exists():
            device_path = "/sys/class/drm/card0/device"

    base_hub = CanonicalSensorHub(device_path=device_path)
    sensor_hub = EnhancedSensorHub(
        base_hub=base_hub,
        body_state=body_state_module,
        power_sample_interval_ms=config.power_sample_interval_ms,
    )

    print("\n[4/6] Building gate network with REINFORCE...")
    gate_net = GateNetWithREINFORCE(
        sensor_dim=EnhancedSensorHub.EXTENDED_DIM,
        body_dim=config.body_dim,
        num_layers=len(config.gate_layers),
    ).to(device)

    print("\n[5/6] Building predictive and interoceptive heads...")
    predictor = PredictiveHead(
        body_dim=config.body_dim,
        sensor_dim=EnhancedSensorHub.EXTENDED_DIM,
    ).to(device)

    intero_report = InteroceptiveReportHead(
        body_dim=config.body_dim,
        sensor_dim=EnhancedSensorHub.EXTENDED_DIM,
    ).to(device)

    print("\n[6/6] Building embodied model...")
    model = EmbodiedModelZ41(
        base_model=base_model,
        gate_net=gate_net,
        sensor_hub=sensor_hub,
        body_state=body_state_module,
        predictor=predictor,
        intero_report=intero_report,
        gate_layers=config.gate_layers,
    )

    # Disturbance and reward
    disturbance = DisturbanceScheduler(device=str(device))
    reward_computer = HorizonRewardZ41(config)

    # Load prompts
    prompts = load_prompts()
    print(f"  Loaded {len(prompts)} prompts")

    # Optimizer (all trainable components)
    optimizer = torch.optim.AdamW([
        {"params": gate_net.parameters(), "lr": config.gate_lr},
        {"params": body_state_module.parameters(), "lr": config.body_lr},
        {"params": predictor.parameters(), "lr": config.predictor_lr},
        {"params": intero_report.parameters(), "lr": config.predictor_lr},
        {"params": [p for block in model.skip_blocks.values() for p in block.parameters()], "lr": config.gate_lr},
    ], weight_decay=0.01)

    print(f"\n  DVFS modes: {DVFS_MODES}")  # Show correct modes!
    print(f"  Total steps per epoch: {config.max_prompts}")
    print(f"  Disturbance probability: {config.disturbance_prob}")

    # Training
    global_step = 0
    for epoch in range(config.epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{config.epochs}")
        print(f"{'='*70}")

        reward_computer.reset()

        global_step = train_epoch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            optimizer=optimizer,
            config=config,
            disturbance=disturbance,
            reward_computer=reward_computer,
            epoch=epoch,
            global_step=global_step,
        )

    # Cleanup
    disturbance.clear()

    # Final checkpoint
    save_checkpoint(model, global_step, config)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
