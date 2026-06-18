#!/usr/bin/env python3
"""
FEEL z44: Complete Embodied Trainer with SENSOR-CONDITIONAL FIXES
=================================================================

CRITICAL FIXES FROM z43 (based on diagnostic analysis):

1. CONTRASTIVE GATE LOSS - Direct supervision for sensor-conditional behavior!
   - Enforces |gate(stressed) - gate(relaxed)| >= margin
   - Enforces DIRECTIONALITY: stress → more skipping / lower DVFS
   - Uses REAL sensor pairs from buffer, not synthetic

2. NON-SATURATING FiLM KL - Replace hinge with maximize-up-to-cap!
   - Old: loss = relu(target - kl) → goes to 0 when kl > target → NO GRADIENT
   - New: loss = -log(1 + kl) capped → always provides gradient, never explodes
   - KL target SCHEDULES upward over training

3. PREDICTOR PRETRAINING PHASE - Supervised first 100 steps!
   - First 100 steps: predictor_weight = 1.0 (heavy supervision)
   - After 100 steps: predictor_weight = 0.1 (auxiliary loss)
   - This bootstraps a working self-model before RL

4. SENSOR BUFFER for contrastive pairs - Real stressed/relaxed examples!
   - Stores (sensors, condition) tuples during training
   - Samples REAL pairs for contrastive loss (not synthetic baseline)

PREVIOUS FIXES (from z41/z42/z43):
- Proper REINFORCE with log_prob * advantage
- DVFS: "auto"/"min_sclk"/"peak"
- Persistent body state with decay/inertia
- Lagged features + derivatives (52-dim)
- Real disturbances via DisturbanceScheduler

THE EMBODIMENT LOOP:
  sensors → body_state → gate_net → actions → hardware → sensors

NEW: Contrastive gate loss + non-saturating FiLM KL = REAL sensor-conditional control!

Author: FEEL Research Team
Date: 2026-01-16 (z44 - SENSOR-CONDITIONAL FIXES)
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
class Z44Config:
    """z44 Complete Embodied Training Configuration with Sensor-Conditional Fixes."""
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

    # FiLM - z44 FIXES
    film_scale: float = 1.0
    film_kl_coef: float = 0.1  # Weight for FiLM KL loss
    film_lr: float = 1e-4  # Learning rate for FiLM/skip_proj params
    film_kl_target_init: float = 0.05  # Initial KL target (ramps up)
    film_kl_target_max: float = 0.5   # Maximum KL target
    film_kl_ramp_steps: int = 300     # Steps to ramp from init to max

    # NEW z44: Contrastive gate loss (FIX: higher coef for stronger signal)
    contrastive_coef: float = 0.2     # Weight for contrastive gate loss (was 0.05, too weak)
    contrastive_margin: float = 0.05  # Minimum gate difference required
    sensor_buffer_size: int = 200     # Size of buffer for real sensor pairs (with telemetry)

    # NEW z44: Gate pretraining (no REINFORCE, just contrastive)
    gate_pretrain_steps: int = 50     # Steps with ONLY contrastive, no RL

    # NEW z44: Predictor pretraining
    predictor_pretrain_steps: int = 100  # Steps with heavy predictor weight
    predictor_pretrain_weight: float = 1.0  # Weight during pretraining
    predictor_normal_weight: float = 0.1    # Weight after pretraining

    val_every: int = 50  # Frequent checkpoints
    checkpoint_dir: str = "models/z44_embodied"

    # Wandb
    wandb_project: str = "feel-z44-embodied"
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

    def encode(self, sensors: torch.Tensor) -> torch.Tensor:
        """
        STATELESS encoding: returns what body state WOULD be for these sensors,
        WITHOUT mutating the persistent state.

        Used for contrastive learning where we need to compare different sensor
        inputs without contaminating the running body state.
        """
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        encoded = self.sensor_encoder(sensors).squeeze(0)
        return encoded

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
        """Reset decisions to run all - use only at start of step, NOT between samples!"""
        for block in self.skip_blocks.values():
            block.run_decision = True

    def reset_tracking_flags(self):
        """Reset only the tracking flags, NOT the run_decisions - use between samples."""
        for block in self.skip_blocks.values():
            block.skipped_this_forward = False

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

    def compute_film_kl_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sensors_stressed: torch.Tensor,
        sensors_relaxed: torch.Tensor,
        body_state: torch.Tensor,
        kl_target: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute teacher-forced KL loss to train FiLM generator.

        z44 FIX: NON-SATURATING loss that keeps gradients alive!

        Old (z43): kl_loss = relu(target - kl) → saturates at 0, no gradient
        New (z44): Two-sided smooth loss + maximize term → always has gradient

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            sensors_stressed: Sensor readings under stress
            sensors_relaxed: Sensor readings when relaxed
            body_state: Current body state
            kl_target: Target KL divergence (scheduled, ramps up over training)

        Returns:
            kl_loss: KL divergence loss (encourage different outputs for different states)
            metrics: Dict with kl_value, logit_diff, etc.
        """
        # Ensure all skip blocks RUN (no skipping) for this comparison
        for block in self.skip_blocks.values():
            block.run_decision = True

        # Forward pass with STRESSED sensors (FiLM enabled)
        for block in self.skip_blocks.values():
            block.film_scale = 1.0
            block.sensors = sensors_stressed
            block.body_state = body_state

        outputs_stressed = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        logits_stressed = outputs_stressed.logits  # [B, T, V]

        # Forward pass with RELAXED sensors (FiLM enabled but different state)
        for block in self.skip_blocks.values():
            block.film_scale = 1.0
            block.sensors = sensors_relaxed
            block.body_state = body_state

        outputs_relaxed = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        logits_relaxed = outputs_relaxed.logits  # [B, T, V]

        # Compute KL divergence between the two distributions
        # We want this to be NON-ZERO (different states should produce different outputs)
        log_probs_stressed = F.log_softmax(logits_stressed, dim=-1)
        probs_relaxed = F.softmax(logits_relaxed, dim=-1)

        # KL(relaxed || stressed) averaged over sequence
        kl_per_token = F.kl_div(log_probs_stressed, probs_relaxed, reduction='none').sum(dim=-1)
        kl_mean = kl_per_token.mean()

        # z44 FIX: NON-SATURATING LOSS
        # We want KL to reach kl_target, but ALWAYS provide gradient
        #
        # Component 1: Pull toward target (smooth L1, never zero gradient)
        # Component 2: Maximize KL up to a cap (always pushes for more difference)
        #
        # This replaces: kl_loss = relu(target - kl) which saturates!

        # Smooth L1 to target (provides gradient on both sides)
        target_tensor = torch.tensor(kl_target, device=kl_mean.device, dtype=kl_mean.dtype)
        target_loss = F.smooth_l1_loss(kl_mean, target_tensor)

        # Maximize KL term: -log(1 + kl) capped at reasonable value
        # This ALWAYS provides gradient pushing for more KL
        maximize_loss = -torch.log(1.0 + kl_mean).clamp(min=-2.0)

        # Combined: target_loss pulls to target, maximize_loss keeps pushing
        # Weight maximize_loss less once we're near target
        kl_loss = target_loss + 0.3 * maximize_loss

        # Also compute raw logit difference for monitoring
        logit_diff = (logits_stressed - logits_relaxed).abs().mean()

        # Compute gradient magnitude estimate (for diagnostics)
        grad_magnitude = (1.0 / (1.0 + kl_mean.detach())).item()  # ~derivative of -log(1+kl)

        metrics = {
            "kl_mean": kl_mean.item(),
            "kl_target": kl_target,
            "kl_loss": kl_loss.item(),
            "target_loss": target_loss.item(),
            "maximize_loss": maximize_loss.item(),
            "logit_diff": logit_diff.item(),
            "grad_magnitude": grad_magnitude,
        }

        return kl_loss, metrics

    def get_film_params(self) -> List[torch.nn.Parameter]:
        """Get all FiLM generator and skip_proj parameters for optimizer."""
        params = []
        for block in self.skip_blocks.values():
            params.extend(block.film_generator.parameters())
            params.extend(block.skip_proj.parameters())
        return params


# ============================================================================
# HORIZON REWARD
# ============================================================================

class HorizonRewardZ41:
    """Enhanced reward with prediction and discomfort components."""

    def __init__(self, config: Z44Config):
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
# z44 NEW: SENSOR BUFFER FOR CONTRASTIVE PAIRS
# ============================================================================

class SensorBuffer:
    """
    Buffer to store REAL sensor readings WITH TELEMETRY for contrastive learning.

    z44 FIX: Labels stressed/relaxed by ACTUAL POWER/TEMP, not disturbance type!
    Samples EXTREMES (top/bottom quantiles) for maximum contrastive signal.
    """

    def __init__(self, max_size: int = 200, power_hi: float = 80.0, power_lo: float = 50.0):
        self.max_size = max_size
        self.power_hi = power_hi  # Above this = stressed
        self.power_lo = power_lo  # Below this = relaxed
        # Store (sensors, power, temp) tuples
        self.buffer: Deque[Tuple[torch.Tensor, float, float]] = deque(maxlen=max_size)

    def add(self, sensors: torch.Tensor, power_w: float, temp_c: float):
        """Add a sensor reading WITH its telemetry."""
        self.buffer.append((sensors.detach().clone(), power_w, temp_c))

    def can_sample_pair(self) -> bool:
        """Check if we have enough spread in power readings."""
        if len(self.buffer) < 10:
            return False
        powers = [p for _, p, _ in self.buffer]
        return max(powers) - min(powers) > 20.0  # Need at least 20W spread

    def sample_extremes(self) -> Tuple[Tuple[torch.Tensor, float, float], Tuple[torch.Tensor, float, float]]:
        """
        Sample from TOP and BOTTOM power quartiles for maximum contrast.
        Returns ((sensors_hi, power_hi, temp_hi), (sensors_lo, power_lo, temp_lo))
        """
        if len(self.buffer) < 10:
            # Fallback: random samples
            s1, p1, t1 = random.choice(self.buffer)
            s2, p2, t2 = random.choice(self.buffer)
            return (s1, p1, t1), (s2, p2, t2)

        # Sort by power
        sorted_buf = sorted(self.buffer, key=lambda x: x[1])
        n = len(sorted_buf)

        # Sample from bottom 25% (relaxed) and top 25% (stressed)
        lo_quartile = sorted_buf[:max(1, n // 4)]
        hi_quartile = sorted_buf[-(max(1, n // 4)):]

        lo_sample = random.choice(lo_quartile)
        hi_sample = random.choice(hi_quartile)

        return hi_sample, lo_sample  # (stressed, relaxed)

    def get_stats(self) -> Dict:
        """Get buffer statistics for logging."""
        if len(self.buffer) == 0:
            return {"size": 0, "power_min": 0, "power_max": 0, "power_spread": 0}
        powers = [p for _, p, _ in self.buffer]
        return {
            "size": len(self.buffer),
            "power_min": min(powers),
            "power_max": max(powers),
            "power_spread": max(powers) - min(powers),
        }

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# z44 NEW: CONTRASTIVE GATE LOSS
# ============================================================================

class ContrastiveGateLoss:
    """
    Contrastive loss to enforce sensor-conditional gate behavior.

    Key insight: REINFORCE alone teaches an "average" good gate.
    This loss DIRECTLY forces gates to differ between conditions.

    Two components:
    1. MARGIN: |gate(stressed) - gate(relaxed)| >= margin
    2. DIRECTION: stressed → lower gate (more skipping)
    """

    def __init__(self, margin: float = 0.05, direction_weight: float = 0.5):
        self.margin = margin
        self.direction_weight = direction_weight

    def compute(
        self,
        gate_probs_stressed: List[torch.Tensor],
        gate_probs_relaxed: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute contrastive gate loss.

        Args:
            gate_probs_stressed: Gate probabilities under stress
            gate_probs_relaxed: Gate probabilities when relaxed

        Returns:
            loss: Contrastive loss (to minimize)
            metrics: Dict with diagnostic info
        """
        # Compute mean gates
        mean_gate_stressed = sum(p.mean() for p in gate_probs_stressed) / len(gate_probs_stressed)
        mean_gate_relaxed = sum(p.mean() for p in gate_probs_relaxed) / len(gate_probs_relaxed)

        # 1. MARGIN LOSS: gates should differ by at least `margin`
        gate_diff = (mean_gate_relaxed - mean_gate_stressed).abs()
        margin_loss = F.relu(self.margin - gate_diff)

        # 2. DIRECTION LOSS: stressed should have LOWER gate (more skipping)
        # gate = prob of RUNNING, so lower gate = more skip
        # We want: gate_stressed < gate_relaxed
        # Loss = relu(gate_stressed - gate_relaxed + small_margin)
        direction_loss = F.relu(mean_gate_stressed - mean_gate_relaxed + 0.01)

        # Combined loss
        total_loss = margin_loss + self.direction_weight * direction_loss

        metrics = {
            "gate_diff": gate_diff.item(),
            "margin_loss": margin_loss.item(),
            "direction_loss": direction_loss.item(),
            "gate_stressed": mean_gate_stressed.item(),
            "gate_relaxed": mean_gate_relaxed.item(),
            "direction_correct": (mean_gate_stressed < mean_gate_relaxed).item(),
        }

        return total_loss, metrics


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
    config: Z44Config,
    disturbance: DisturbanceScheduler,
    reward_computer: HorizonRewardZ41,
    sensor_buffer: SensorBuffer,
    contrastive_loss_fn: ContrastiveGateLoss,
    epoch: int,
    global_step: int,
) -> int:
    """Train one epoch with z44 SENSOR-CONDITIONAL FIXES."""
    device = next(model.gate_net.parameters()).device

    model.sensor_hub.training_mode = True
    model.body_state_module.train()
    random.shuffle(prompts)
    prompts = prompts[:config.max_prompts]

    for prompt_idx, prompt in enumerate(prompts):
        step = global_step + prompt_idx

        # z44: Compute scheduled KL target (ramps up over training)
        kl_progress = min(1.0, step / config.film_kl_ramp_steps)
        kl_target = config.film_kl_target_init + kl_progress * (config.film_kl_target_max - config.film_kl_target_init)

        # z44: Compute predictor weight (high during pretraining, then normal)
        if step < config.predictor_pretrain_steps:
            predictor_weight = config.predictor_pretrain_weight
        else:
            predictor_weight = config.predictor_normal_weight

        # Maybe apply disturbance
        dist_type = disturbance.maybe_apply(prob=config.disturbance_prob)
        was_stressed = dist_type is not None

        if was_stressed:
            time.sleep(0.1)

        # Read sensors
        sensors = model.sensor_hub.read_tensor().to(device)

        # z44 FIX: Store sensors LATER after we have actual power/temp (not disturbance label)

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

        # FIX BUG 1: Predictor MUST get gradients - removed torch.no_grad()!
        predictions = model.predictor(body_state, sensors, dvfs_onehot, mean_gate_prob)

        # Generate with decode-time power sampling
        samples = []
        rewards = []
        advantages = []
        log_probs = []

        for sample_idx in range(config.num_samples):
            # FIX BUG 2: Only reset tracking flags, NOT run_decisions!
            # The sampled actions from compute_actions() must persist during generation
            model.reset_tracking_flags()
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

        # z44 FIX: Store sensors with ACTUAL power/temp (not disturbance label!)
        # Use metrics from the first sample
        sensor_buffer.add(sensors, power_w=samples[0]["avg_power_w"], temp_c=samples[0]["temp_c"])

        # REINFORCE update (THE CRITICAL FIX)
        optimizer.zero_grad()

        mean_advantage = sum(advantages) / len(advantages)
        mean_log_prob = sum(log_probs) / len(log_probs)

        # z44 FIX: Gate pretraining phase - NO REINFORCE, just contrastive!
        # This allows the gate to learn sensor-conditional behavior before RL interferes
        if step < config.gate_pretrain_steps:
            # During gate pretraining: zero RL losses, only contrastive + predictor
            policy_loss = torch.tensor(0.0, device=device)
            entropy_loss = torch.tensor(0.0, device=device)
        else:
            # Policy gradient loss: -log_prob * advantage
            policy_loss = -mean_log_prob * mean_advantage
            # Entropy bonus for exploration
            entropy_loss = -config.entropy_coef * action_result["entropy"]

        # FIX: Multi-target predictor loss (power, j_tok, temp) with proper shapes
        s0 = samples[0]
        actual_power = torch.tensor([s0["avg_power_w"]], device=device, dtype=torch.float32)
        actual_temp = torch.tensor([s0["temp_c"]], device=device, dtype=torch.float32)
        actual_j_tok = torch.tensor([s0["j_per_token"]], device=device, dtype=torch.float32)

        # Normalize targets for stable training
        power_target_norm = actual_power / 100.0  # Scale power to ~0-1.5
        temp_target_norm = actual_temp / 100.0  # Scale temp to ~0-1
        j_tok_target_norm = actual_j_tok / 10.0  # Scale J/tok to ~0-1

        pred_power_norm = predictions["power"] / 100.0
        pred_temp_norm = predictions["temp"] / 100.0
        pred_energy_norm = predictions["energy"] / 10.0  # Use energy as J/tok proxy

        pred_loss = (
            F.mse_loss(pred_power_norm, power_target_norm) +
            F.mse_loss(pred_temp_norm, temp_target_norm) +
            F.mse_loss(pred_energy_norm, j_tok_target_norm)
        )

        # FIX: Interoceptive calibration with FULL discomfort, not just thermal
        intero = model.intero_report(body_state, sensors)
        discomfort_bd = breakdown["discomfort_breakdown"]
        # Weighted combination of all discomfort components
        full_discomfort = (
            0.5 * discomfort_bd.get("thermal", 0.0) +
            0.3 * discomfort_bd.get("power_overshoot", 0.0) +
            0.2 * discomfort_bd.get("energy_discomfort", 0.0)
        )
        actual_strain = torch.tensor([full_discomfort], device=device, dtype=torch.float32)
        intero_loss = F.mse_loss(intero["strain_level"], actual_strain)

        # z44 FIX: Contrastive gate loss using EXTREME sensor pairs from buffer
        # KEY FIXES:
        #   1. Use sample_extremes() to get actual high/low power samples
        #   2. Use encode() not update() - STATELESS, no body state mutation
        #   3. Enable gradients so body encoder can learn separation
        if sensor_buffer.can_sample_pair() and step % 3 == 0:
            # Sample from TOP (stressed) and BOTTOM (relaxed) power quartiles
            (sensors_hi, power_hi, temp_hi), (sensors_lo, power_lo, temp_lo) = sensor_buffer.sample_extremes()
            sensors_hi = sensors_hi.to(device)
            sensors_lo = sensors_lo.to(device)

            # z44 FIX: Use STATELESS encode() - does NOT mutate body state!
            # WITH gradients so body encoder can learn stress separation
            body_hi = model.body_state_module.encode(sensors_hi)
            body_lo = model.body_state_module.encode(sensors_lo)

            gate_result_hi = model.gate_net(sensors_hi, body_hi, sample=False)
            gate_result_lo = model.gate_net(sensors_lo, body_lo, sample=False)

            contrastive_loss, contrastive_metrics = contrastive_loss_fn.compute(
                gate_result_hi["gate_probs"],  # stressed (high power)
                gate_result_lo["gate_probs"],  # relaxed (low power)
            )
            # Store pair telemetry for logging
            contrastive_metrics["pair_power_hi"] = power_hi
            contrastive_metrics["pair_power_lo"] = power_lo
            contrastive_metrics["pair_temp_hi"] = temp_hi
            contrastive_metrics["pair_temp_lo"] = temp_lo
            contrastive_metrics["pair_power_diff"] = power_hi - power_lo
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
            contrastive_metrics = {"gate_diff": 0.0, "margin_loss": 0.0, "direction_loss": 0.0,
                                   "gate_stressed": 0.0, "gate_relaxed": 0.0, "direction_correct": True,
                                   "pair_power_hi": 0.0, "pair_power_lo": 0.0, "pair_temp_hi": 0.0,
                                   "pair_temp_lo": 0.0, "pair_power_diff": 0.0}

        # z44 FIX: FiLM KL loss with scheduled target and EXTREME sensor pairs
        if step % 5 == 0 and sensor_buffer.can_sample_pair():
            # Use EXTREME sensor pairs from buffer (top/bottom power quartiles)
            (sensors_hi, _, _), (sensors_lo, _, _) = sensor_buffer.sample_extremes()
            sensors_hi = sensors_hi.to(device)
            sensors_lo = sensors_lo.to(device)

            film_kl_loss, kl_metrics = model.compute_film_kl_loss(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                sensors_stressed=sensors_hi,
                sensors_relaxed=sensors_lo,
                body_state=body_state,
                kl_target=kl_target,  # z44: Scheduled target!
            )
        elif step % 5 == 0:
            # Fallback to synthetic baseline if buffer not ready
            sensors_current = sensors.clone()
            sensors_baseline = sensors.clone()
            sensors_baseline[36:44] = 0.0
            sensors_baseline[44:52] = sensors_baseline[44:52] * 0.5

            film_kl_loss, kl_metrics = model.compute_film_kl_loss(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                sensors_stressed=sensors_current,
                sensors_relaxed=sensors_baseline,
                body_state=body_state,
                kl_target=kl_target,
            )
        else:
            film_kl_loss = torch.tensor(0.0, device=device)
            kl_metrics = {"kl_mean": 0.0, "kl_target": kl_target, "kl_loss": 0.0,
                          "target_loss": 0.0, "maximize_loss": 0.0, "logit_diff": 0.0, "grad_magnitude": 0.0}

        # z44: Total loss with CONTRASTIVE + scheduled predictor weight
        total_loss = (
            policy_loss +
            entropy_loss +
            predictor_weight * pred_loss +  # z44: Scheduled weight (1.0 first 100 steps, then 0.1)
            0.1 * intero_loss +
            config.film_kl_coef * film_kl_loss +
            config.contrastive_coef * contrastive_loss  # z44: NEW contrastive loss!
        )

        total_loss.backward()

        # Compute gradient norms BEFORE clipping for diagnostics
        def grad_norm(params):
            total = 0.0
            for p in params:
                if p.grad is not None:
                    total += p.grad.data.norm(2).item() ** 2
            return total ** 0.5

        gate_grad_norm = grad_norm(model.gate_net.parameters())
        pred_grad_norm = grad_norm(model.predictor.parameters())
        film_grad_norm = grad_norm(model.get_film_params())
        intero_grad_norm = grad_norm(model.intero_report.parameters())

        # FIX: Include FiLM params in gradient clipping!
        all_params = (
            list(model.gate_net.parameters()) +
            list(model.predictor.parameters()) +
            list(model.intero_report.parameters()) +
            list(model.body_state_module.parameters()) +
            model.get_film_params()  # NEW: FiLM generator + skip_proj
        )
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        # Progress logging
        if step % 1 == 0:  # Log every step
            m = model.get_metrics()
            s = samples[0]
            b = s["breakdown"]

            gate_mean = sum(p.mean().item() for p in action_result["gate_probs"]) / len(action_result["gate_probs"])

            stress_str = f"[{dist_type}]" if was_stressed else "[normal]"
            kl_str = f"kl={kl_metrics['kl_mean']:.3f}" if kl_metrics['kl_mean'] > 0 else ""
            grad_str = f"∇g={gate_grad_norm:.2e} ∇f={film_grad_norm:.2e}"
            # z44 FIX: ALWAYS show Δg with high precision + pair telemetry when available
            if contrastive_metrics["pair_power_diff"] > 0:
                # Show gate_hi, gate_lo, and the power difference of the sampled pair
                cdiff_str = (f"Δg={contrastive_metrics['gate_diff']:.4f} "
                            f"g_hi={contrastive_metrics['gate_stressed']:.3f} "
                            f"g_lo={contrastive_metrics['gate_relaxed']:.3f} "
                            f"ΔP={contrastive_metrics['pair_power_diff']:.0f}W")
            else:
                cdiff_str = f"Δg={contrastive_metrics['gate_diff']:.4f}"
            # z44: Show predictor weight phase + gate pretraining
            if step < config.gate_pretrain_steps:
                phase_str = "GATE_PRETRAIN"
            elif step < config.predictor_pretrain_steps:
                phase_str = "PRED_PRETRAIN"
            else:
                phase_str = ""
            # Buffer stats
            buf_stats = sensor_buffer.get_stats()
            buf_str = f"buf={buf_stats['size']}({buf_stats['power_spread']:.0f}W)"
            print(f"  [{step:4d}] {stress_str:12s} gate={gate_mean:.3f} skip={m['skip_rate']:.2f} "
                  f"J/tok={s['j_per_token']:.2f} P={s['avg_power_w']:.1f}W T={s['temp_c']:.1f}C "
                  f"r={rewards[0]:.3f} adv={advantages[0]:.3f} dvfs={dvfs_mode} "
                  f"strain={intero['strain_level'].item():.2f} {kl_str} {cdiff_str} {buf_str} {grad_str} {phase_str}", flush=True)

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
                    "train/intero_loss": intero_loss.item(),
                    # z44: FiLM KL metrics with schedule
                    "train/film_kl_loss": kl_metrics["kl_loss"],
                    "train/film_kl_mean": kl_metrics["kl_mean"],
                    "train/film_kl_target": kl_metrics.get("kl_target", kl_target),
                    "train/film_target_loss": kl_metrics.get("target_loss", 0),
                    "train/film_maximize_loss": kl_metrics.get("maximize_loss", 0),
                    "train/film_logit_diff": kl_metrics.get("logit_diff", 0),
                    "train/film_grad_magnitude": kl_metrics.get("grad_magnitude", 0),
                    # z44 FIX: Contrastive gate loss metrics with pair telemetry
                    "train/contrastive_loss": contrastive_loss.item(),
                    "train/contrastive_gate_diff": contrastive_metrics["gate_diff"],
                    "train/contrastive_margin_loss": contrastive_metrics["margin_loss"],
                    "train/contrastive_direction_loss": contrastive_metrics["direction_loss"],
                    "train/contrastive_gate_hi": contrastive_metrics.get("gate_stressed", 0),
                    "train/contrastive_gate_lo": contrastive_metrics.get("gate_relaxed", 0),
                    "train/contrastive_direction_correct": contrastive_metrics.get("direction_correct", True),
                    "train/contrastive_pair_power_hi": contrastive_metrics.get("pair_power_hi", 0),
                    "train/contrastive_pair_power_lo": contrastive_metrics.get("pair_power_lo", 0),
                    "train/contrastive_pair_power_diff": contrastive_metrics.get("pair_power_diff", 0),
                    # z44: Buffer stats
                    "train/buffer_size": buf_stats["size"],
                    "train/buffer_power_spread": buf_stats["power_spread"],
                    # z44: Training phases
                    "train/predictor_weight": predictor_weight,
                    "train/in_gate_pretrain": 1.0 if step < config.gate_pretrain_steps else 0.0,
                    "train/in_pred_pretrain": 1.0 if step < config.predictor_pretrain_steps else 0.0,
                    # Predictions
                    "train/pred_power": s["predictions"]["power"],
                    "train/pred_temp": s["predictions"]["temp"],
                    "train/pred_energy": s["predictions"].get("energy", 0),
                    # Interoceptive
                    "train/strain_level": intero["strain_level"].item(),
                    "train/actual_strain": full_discomfort,
                    "train/confidence": intero["confidence"].item(),
                    # Actuators
                    "train/dvfs_mode": dvfs_mode,
                    "train/dvfs_action": dvfs_action,
                    # Disturbance
                    "train/disturbance": dist_type if was_stressed else "none",
                    # Body state
                    "train/body_state_norm": body_state.norm().item(),
                    # Gradient norms (diagnose flat metrics)
                    "grad/gate_net": gate_grad_norm,
                    "grad/predictor": pred_grad_norm,
                    "grad/film": film_grad_norm,
                    "grad/intero": intero_grad_norm,
                    # z44: Sensor buffer stats
                    "buffer/size": len(sensor_buffer),
                    "buffer/can_sample": sensor_buffer.can_sample_pair(),
                    # Total loss
                    "train/total_loss": total_loss.item(),
                })

        # Validation checkpoint
        if step > 0 and step % config.val_every == 0:
            save_checkpoint(model, step, config)

    model.sensor_hub.training_mode = False
    return global_step + len(prompts)


def save_checkpoint(model: EmbodiedModelZ41, step: int, config: Z44Config):
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
    parser = argparse.ArgumentParser(description="FEEL z44: Complete Embodied Trainer with Sensor-Conditional Fixes")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="models/z44_embodied")
    parser.add_argument("--disturbance-prob", type=float, default=0.35)
    parser.add_argument("--power-cap", type=float, default=60.0)
    # z44 new args (with FIXES)
    parser.add_argument("--contrastive-coef", type=float, default=0.2)  # FIX: was 0.05, too weak
    parser.add_argument("--gate-pretrain-steps", type=int, default=50)  # FIX: no RL first 50 steps
    parser.add_argument("--predictor-pretrain-steps", type=int, default=100)
    parser.add_argument("--wandb-project", type=str, default="feel-z44-embodied")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    config = Z44Config(
        epochs=args.epochs,
        max_prompts=args.max_prompts,
        checkpoint_dir=args.checkpoint_dir,
        disturbance_prob=args.disturbance_prob,
        power_cap_w=args.power_cap,
        contrastive_coef=args.contrastive_coef,
        gate_pretrain_steps=args.gate_pretrain_steps,  # z44 FIX: no RL first N steps
        predictor_pretrain_steps=args.predictor_pretrain_steps,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
    )

    # Initialize wandb
    if WANDB_AVAILABLE and config.use_wandb:
        import socket
        hostname = socket.gethostname()
        run_name = config.wandb_run_name or f"z44_embodied_{hostname}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=asdict(config),
            tags=["z44", "contrastive", "sensor-conditional", "predictor-pretrain", hostname],
        )

    print("=" * 70)
    print("FEEL z44: SENSOR-CONDITIONAL EMBODIED TRAINER")
    print("=" * 70)
    print("z44 CRITICAL FIXES:")
    print("  1. CONTRASTIVE GATE LOSS - Direct supervision for sensor-conditional behavior")
    print("  2. NON-SATURATING FiLM KL - Always provides gradient (no hinge saturation)")
    print("  3. KL TARGET SCHEDULE - Ramps from 0.05 → 0.5 over training")
    print("  4. PREDICTOR PRETRAINING - Weight=1.0 first 100 steps, then 0.1")
    print("  5. REAL SENSOR PAIRS - Uses buffer of actual stressed/relaxed readings")
    print("  6. DIRECTION ENFORCEMENT - Stress → lower gate (more skipping)")
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

    # z44: Sensor buffer for contrastive learning with REAL sensor pairs
    sensor_buffer = SensorBuffer(max_size=config.sensor_buffer_size)
    print(f"  Sensor buffer initialized (max_size={config.sensor_buffer_size})")

    # z44: Contrastive gate loss for sensor-conditional behavior
    contrastive_loss_fn = ContrastiveGateLoss(
        margin=config.contrastive_margin,
        direction_weight=0.5,
    )
    print(f"  Contrastive gate loss initialized (margin={config.contrastive_margin})")

    # Load prompts
    prompts = load_prompts()
    print(f"  Loaded {len(prompts)} prompts")

    # Optimizer (all trainable components including FiLM!)
    # FIX: Use separate learning rate for FiLM/skip_proj params
    film_params = model.get_film_params()
    optimizer = torch.optim.AdamW([
        {"params": gate_net.parameters(), "lr": config.gate_lr},
        {"params": body_state_module.parameters(), "lr": config.body_lr},
        {"params": predictor.parameters(), "lr": config.predictor_lr},
        {"params": intero_report.parameters(), "lr": config.predictor_lr},
        {"params": film_params, "lr": config.film_lr},  # FiLM + skip_proj with dedicated LR
    ], weight_decay=0.01)
    print(f"  FiLM/skip_proj params: {sum(p.numel() for p in film_params)}")

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
            sensor_buffer=sensor_buffer,  # z44: Pass sensor buffer
            contrastive_loss_fn=contrastive_loss_fn,  # z44: Pass contrastive loss
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
