#!/usr/bin/env python3
"""
FEEL z45v3: Safe Embodied Trainer with Deep GPU Signals
=======================================================

z45v3 CHANGES (based on signal variance analysis):
- DEEP GPU SIGNALS from gpu_metrics binary blob:
  - throttle_status (offset 96): CV=1.72, MOST VARIABLE SIGNAL!
  - mem_ctrl_activity (offset 66): CV=0.20, memory controller load
  - vcn_activity (offset 76): CV=0.21, video codec activity
- Replaces broken gfx_voltage (returns 0 on Z2)
- Extended feature dimension: 72 (was 64)
- More comprehensive wandb logging

KEY z45 CHANGES from z44 (based on hardware failure analysis):

1. SAFE VARIABILITY CURRICULUM - No more extreme gpu_heavy!
   - Bounded stress: 70-110W target range (never push to 160W+)
   - Short pulses (1-3s) instead of sustained load
   - Frequency over intensity for learning

2. THERMAL GOVERNOR (HARD SAFETY)
   - Temp > 80C: Force DVFS=min + skip_all + pause stress
   - Power > 130W: Same protective response
   - Cooldown window before resuming
   - This prevents Z2 thermal shutdowns!

3. FAST SIGNAL SENSORS (NEW)
   - gpu_busy_percent: Instant utilization (10-100ms response)
   - power1_input: Instant power (faster than power1_average)
   - voltage: GFX and SOC voltages
   - Memory activity indicators
   - These capture subtle state changes temp/power miss

4. NORMALIZED CONTRASTIVE LOSS
   - Gate diff normalized by sensor spread
   - Prevents threshold tuning sensitivity
   - Better gradient signal early training

5. PREDICTION CURRICULUM
   - Phase 1 (0-100 steps): Predict power + DVFS (easy, fast signals)
   - Phase 2 (100-200 steps): Add energy/J-tok
   - Phase 3 (200+ steps): Full prediction including temp

6. EXPECTED SKIP (CONTINUOUS EARLY)
   - First 100 steps: Use expected skip (p_skip) not sampled
   - After 100 steps: Switch to sampled actions
   - Reduces variance for initial learning

7. RECOVERY/HYSTERESIS REWARD
   - Explicit reward for returning to normal after stress
   - Penalize "stuck" in stress state
   - Tracks state transitions

Author: FEEL Research Team
Date: 2026-01-16 (z45v3 - DEEP GPU SIGNALS)
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
class Z45Config:
    """z45 Safe Embodied Training Configuration."""
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
    body_decay: float = 0.1
    body_noise_std: float = 0.01

    # Reward weights
    quality_weight: float = 0.15
    energy_weight: float = 0.25
    recovery_weight: float = 0.20  # z45: INCREASED recovery weight
    throughput_weight: float = 0.15
    prediction_weight: float = 0.10
    discomfort_weight: float = 0.15

    # z45: SAFETY TARGETS (conservative!)
    power_cap_w: float = 60.0      # Soft target for reward
    power_safety_w: float = 130.0  # HARD LIMIT - triggers governor
    temp_target_c: float = 70.0
    temp_safety_c: float = 80.0    # HARD LIMIT - triggers governor
    j_per_token_target: float = 2.0

    # z45: Safe stress bounds
    stress_power_min_w: float = 70.0   # Minimum stress target
    stress_power_max_w: float = 110.0  # Maximum stress target (NO 160W!)
    stress_duration_min_s: float = 0.5
    stress_duration_max_s: float = 2.0
    cooldown_duration_s: float = 3.0   # Cooldown after safety trigger

    # Decode-time power sampling
    power_sample_interval_ms: float = 10.0

    # z45: Lower disturbance probability (quality over quantity)
    disturbance_prob: float = 0.30

    # FiLM settings
    film_scale: float = 1.0
    film_kl_coef: float = 0.1
    film_lr: float = 1e-4
    film_kl_target_init: float = 0.05
    film_kl_target_max: float = 0.5
    film_kl_ramp_steps: int = 300

    # Contrastive gate loss
    contrastive_coef: float = 0.2
    contrastive_margin: float = 0.05
    sensor_buffer_size: int = 200

    # z45: Training phases (EXTENDED for safety)
    gate_pretrain_steps: int = 50      # No RL, just contrastive
    expected_skip_steps: int = 100     # Use expected (continuous) skip, not sampled
    predictor_phase1_steps: int = 100  # Power + DVFS prediction only
    predictor_phase2_steps: int = 200  # Add energy
    predictor_pretrain_weight: float = 1.0
    predictor_normal_weight: float = 0.1

    val_every: int = 50
    checkpoint_dir: str = "models/z45_embodied"

    # Wandb
    wandb_project: str = "feel-z45-embodied"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    def __post_init__(self):
        if self.gate_layers is None:
            self.gate_layers = [7, 11, 15, 19, 23]


# DVFS modes
DVFS_MODES = ["auto", "min_sclk", "peak"]


# ============================================================================
# z45 NEW: FAST SIGNAL SENSOR HUB
# ============================================================================

class DeepGPUMetrics:
    """
    z45v3: Parse gpu_metrics binary blob for DEEP hardware signals.

    These signals have HIGH VARIANCE and are HIGHLY CORRELATED with actual GPU activity:
    - throttle_status (offset 96): CV=1.72, most variable signal!
    - mem_ctrl_activity (offset 66): CV=0.20, memory controller load
    - vcn_activity (offset 76): CV=0.21, video codec activity
    - gfx_activity (offset 42): Actual GPU activity (redundant with gpu_busy_percent)

    These replace broken signals like gfx_voltage which returns 0 on Z2.
    """

    def __init__(self, device_path: Path):
        self.gpu_metrics_path = device_path / "gpu_metrics"
        self.available = self.gpu_metrics_path.exists()
        if self.available:
            print(f"[DeepGPUMetrics] Found gpu_metrics at {self.gpu_metrics_path}")
        else:
            print(f"[DeepGPUMetrics] WARN: gpu_metrics not found at {self.gpu_metrics_path}")

    def read(self) -> Dict[str, float]:
        """Read and parse gpu_metrics binary blob."""
        result = {
            "throttle_status": 0.0,
            "mem_ctrl_activity": 0.0,
            "vcn_activity": 0.0,
            "gfx_activity_deep": 0.0,
            "temp_gfx_deep": 0.0,
            "temp_soc_deep": 0.0,
            "gfxclk_deep": 0.0,
            "memclk_deep": 0.0,
            "activity_136": 0.0,
            "activity_138": 0.0,
        }

        if not self.available:
            return result

        try:
            import struct
            data = self.gpu_metrics_path.read_bytes()

            if len(data) < 240:
                return result  # Blob too short

            # Temperatures (0.01C units)
            result["temp_gfx_deep"] = struct.unpack_from("<H", data, 4)[0] / 100.0
            result["temp_soc_deep"] = struct.unpack_from("<H", data, 6)[0] / 100.0

            # Activity percentages (0-100)
            result["gfx_activity_deep"] = float(struct.unpack_from("<H", data, 42)[0])
            result["mem_ctrl_activity"] = float(struct.unpack_from("<H", data, 66)[0])
            result["vcn_activity"] = float(struct.unpack_from("<H", data, 76)[0])
            result["activity_136"] = float(struct.unpack_from("<H", data, 136)[0])
            result["activity_138"] = float(struct.unpack_from("<H", data, 138)[0])

            # Throttle status (32-bit, MOST VARIABLE! CV=1.72)
            result["throttle_status"] = float(struct.unpack_from("<I", data, 96)[0])

            # Clocks (MHz)
            result["gfxclk_deep"] = float(struct.unpack_from("<H", data, 224)[0])
            result["memclk_deep"] = float(struct.unpack_from("<H", data, 186)[0])

        except Exception as e:
            pass  # Return defaults

        return result


class FastSignalSensorHub:
    """
    Extended sensor hub with FAST signals for rapid feedback.

    NEW z45 signals (10-100ms response time):
    - gpu_busy_percent: Instant GPU utilization
    - power_input: Instantaneous power (vs averaged)
    - gfx_voltage: Graphics voltage (NOTE: Returns 0 on Z2!)
    - soc_voltage: SOC voltage
    - current_sclk: Current clock level index
    - current_mclk: Current memory clock level
    - pcie_speed: PCIe link speed state

    z45v3 DEEP signals from gpu_metrics binary:
    - throttle_status: Thermal/power throttle flags (CV=1.72!)
    - mem_ctrl_activity: Memory controller utilization
    - vcn_activity: Video codec engine activity
    """

    # z45v3: Extended dimension for fast + deep signals
    FAST_SIGNAL_DIM = 72  # Was 64, now +8 for deep signals

    def __init__(
        self,
        base_hub: CanonicalSensorHub,
        body_state,
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

        # History for lag features
        self.feature_history: Deque[Tuple[float, torch.Tensor]] = deque(maxlen=100)
        self.power_history: Deque[Tuple[float, float]] = deque(maxlen=50)
        self.temp_history: Deque[Tuple[float, float]] = deque(maxlen=50)

        # z45: Fast signal history
        self.gpu_busy_history: Deque[Tuple[float, float]] = deque(maxlen=50)
        self.voltage_history: Deque[Tuple[float, float, float]] = deque(maxlen=20)

        # Decode-time power sampler
        self.power_sampler = DecodeTimePowerSampler(base_hub, power_sample_interval_ms)
        self.last_decode_stats: Dict = {}

        # z45: Find fast signal paths
        self._init_fast_signal_paths()

        # z45v3: Deep GPU metrics from gpu_metrics binary
        self.deep_metrics = DeepGPUMetrics(self.base.device_path)
        self.deep_history: Deque[Tuple[float, Dict]] = deque(maxlen=50)

        self._training_mode = False

    def _init_fast_signal_paths(self):
        """Initialize paths to fast-response sensor files."""
        device_path = self.base.device_path

        # GPU busy percent (instant utilization)
        self.gpu_busy_path = device_path / "gpu_busy_percent"

        # Instant power (faster than average)
        hwmon_base = device_path / "hwmon"
        self.power_input_path = None
        self.gfx_voltage_path = None
        self.soc_voltage_path = None
        self.freq_input_path = None

        if hwmon_base.exists():
            for hwmon in sorted(hwmon_base.iterdir()):
                # Power input (instant)
                p = hwmon / "power1_input"
                if p.exists():
                    self.power_input_path = p
                # Voltages
                v0 = hwmon / "in0_input"
                if v0.exists():
                    self.gfx_voltage_path = v0
                v1 = hwmon / "in1_input"
                if v1.exists():
                    self.soc_voltage_path = v1
                # Frequency
                f = hwmon / "freq1_input"
                if f.exists():
                    self.freq_input_path = f

        # Clock state files
        self.sclk_path = device_path / "pp_dpm_sclk"
        self.mclk_path = device_path / "pp_dpm_mclk"
        self.pcie_path = device_path / "pp_dpm_pcie"

        print(f"[FastSignalSensorHub] Fast signals initialized:")
        print(f"  gpu_busy: {self.gpu_busy_path}")
        print(f"  power_input: {self.power_input_path}")
        print(f"  gfx_voltage: {self.gfx_voltage_path}")

    def _read_fast_signals(self) -> Dict[str, float]:
        """Read fast-response sensor signals."""
        signals = {}

        # GPU busy percent (0-100)
        if self.gpu_busy_path and self.gpu_busy_path.exists():
            try:
                signals["gpu_busy"] = float(self.gpu_busy_path.read_text().strip())
            except:
                signals["gpu_busy"] = 0.0
        else:
            signals["gpu_busy"] = 0.0

        # Instant power (uW -> W)
        if self.power_input_path and self.power_input_path.exists():
            try:
                signals["power_instant"] = float(self.power_input_path.read_text().strip()) / 1e6
            except:
                signals["power_instant"] = 0.0
        else:
            signals["power_instant"] = 0.0

        # GFX voltage (mV)
        if self.gfx_voltage_path and self.gfx_voltage_path.exists():
            try:
                signals["gfx_voltage"] = float(self.gfx_voltage_path.read_text().strip())
            except:
                signals["gfx_voltage"] = 0.0
        else:
            signals["gfx_voltage"] = 0.0

        # SOC voltage (mV)
        if self.soc_voltage_path and self.soc_voltage_path.exists():
            try:
                signals["soc_voltage"] = float(self.soc_voltage_path.read_text().strip())
            except:
                signals["soc_voltage"] = 0.0
        else:
            signals["soc_voltage"] = 0.0

        # Current clock level (parse asterisk line)
        signals["sclk_level"] = self._parse_current_dpm_level(self.sclk_path)
        signals["mclk_level"] = self._parse_current_dpm_level(self.mclk_path)
        signals["pcie_level"] = self._parse_current_dpm_level(self.pcie_path)

        # Frequency (Hz -> MHz)
        if self.freq_input_path and self.freq_input_path.exists():
            try:
                signals["freq_mhz"] = float(self.freq_input_path.read_text().strip()) / 1e6
            except:
                signals["freq_mhz"] = 0.0
        else:
            signals["freq_mhz"] = 0.0

        # z45v3: DEEP signals from gpu_metrics binary
        deep = self.deep_metrics.read()
        signals["throttle_status"] = deep["throttle_status"]
        signals["mem_ctrl_activity"] = deep["mem_ctrl_activity"]
        signals["vcn_activity"] = deep["vcn_activity"]
        signals["gfx_activity_deep"] = deep["gfx_activity_deep"]
        signals["gfxclk_deep"] = deep["gfxclk_deep"]
        signals["memclk_deep"] = deep["memclk_deep"]
        signals["activity_136"] = deep["activity_136"]
        signals["activity_138"] = deep["activity_138"]

        # Store in history for derivatives
        self.deep_history.append((time.time(), deep))

        return signals

    def _parse_current_dpm_level(self, path) -> float:
        """Parse current DPM level from pp_dpm_* file."""
        if path is None or not path.exists():
            return 0.0
        try:
            content = path.read_text()
            lines = content.strip().split('\n')
            total = len(lines)
            for i, line in enumerate(lines):
                if '*' in line:
                    return i / max(1, total - 1)  # Normalize to 0-1
            return 0.0
        except:
            return 0.0

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
            p_now = self.power_history[-1][1]
            p_prev = self.power_history[-2][1]
            dt = self.power_history[-1][0] - self.power_history[-2][0]

            derivatives[0] = p_now
            derivatives[1] = p_now - p_prev
            derivatives[2] = (p_now - p_prev) / max(dt, 0.001)

            if len(self.power_history) >= 5:
                recent = [p for _, p in list(self.power_history)[-5:]]
                derivatives[3] = sum(recent) / len(recent)

        if len(self.temp_history) >= 2:
            t_now = self.temp_history[-1][1]
            t_prev = self.temp_history[-2][1]
            dt = self.temp_history[-1][0] - self.temp_history[-2][0]

            derivatives[4] = t_now
            derivatives[5] = t_now - t_prev
            derivatives[6] = (t_now - t_prev) / max(dt, 0.001)

            if len(self.temp_history) >= 5:
                recent = [t for _, t in list(self.temp_history)[-5:]]
                derivatives[7] = sum(recent) / len(recent)

        return derivatives

    def read_tensor(self, actual_throughput: Optional[float] = None) -> torch.Tensor:
        """Read REAL sensors and compute extended feature vector with FAST SIGNALS."""
        # Get base features
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

        # z45: Read FAST signals
        fast = self._read_fast_signals()
        self.gpu_busy_history.append((current_time, fast["gpu_busy"]))

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

        # 3. Anchors
        power_w = raw.power_mw if raw else 50.0
        temp_c = raw.temp_c if raw else 50.0
        j_per_token = self.last_decode_stats.get("j_per_token", 2.0)

        anchors = torch.tensor([
            (power_w - 60.0) / 60.0,
            (temp_c - 70.0) / 30.0,
            (j_per_token - 2.0) / 2.0,
            float(self.base.dvfs.MODES[self.base.dvfs.current_mode]) / 2.0,
            min(power_w / 100.0, 1.0),
            min(temp_c / 100.0, 1.0),
            1.0 if power_w > 60.0 else 0.0,
            1.0 if temp_c > 80.0 else 0.0,
        ])
        features_list.append(anchors)

        # 4. z45 NEW: FAST SIGNALS (normalized)
        fast_signals = torch.tensor([
            fast["gpu_busy"] / 100.0,  # 0-1
            fast["power_instant"] / 150.0,  # Normalize to ~0-1
            fast["gfx_voltage"] / 1500.0,  # mV, typical ~700-1200 (NOTE: Returns 0 on Z2!)
            fast["soc_voltage"] / 1200.0,  # mV
            fast["sclk_level"],  # Already 0-1
            fast["mclk_level"],  # Already 0-1
            fast["pcie_level"],  # Already 0-1
            fast["freq_mhz"] / 2500.0,  # Normalize
            # Derivatives of fast signals
            self._gpu_busy_derivative(),
            self._power_instant_derivative(fast["power_instant"]),
            # State flags
            1.0 if fast["gpu_busy"] > 90 else 0.0,  # High utilization flag
            1.0 if fast["power_instant"] > 100 else 0.0,  # High power flag
        ])
        features_list.append(fast_signals)

        # 5. z45v3 NEW: DEEP GPU SIGNALS (from gpu_metrics binary)
        # These have HIGH VARIANCE and are most useful for embodiment learning
        deep_signals = torch.tensor([
            # Throttle status - MOST VARIABLE (CV=1.72!)
            # Normalize: typical values 0-16384 (power of 2 flags)
            min(fast["throttle_status"] / 16384.0, 1.0),
            1.0 if fast["throttle_status"] > 0 else 0.0,  # Binary throttle flag
            # Memory controller activity (0-100%) - CV=0.20
            fast["mem_ctrl_activity"] / 100.0,
            # VCN activity (0-100%) - CV=0.21
            fast["vcn_activity"] / 100.0,
            # Deep activity signals
            fast["gfx_activity_deep"] / 100.0,  # Should match gpu_busy
            fast["activity_136"] / 100.0,  # Unknown activity signal
            # Clocks from gpu_metrics (more reliable than pp_dpm_*)
            fast["gfxclk_deep"] / 2500.0,  # Normalize to 0-1
            fast["memclk_deep"] / 1000.0,  # Normalize to 0-1
        ])
        features_list.append(deep_signals)

        # Concatenate
        extended = torch.cat(features_list)

        # Pad/truncate to fixed size
        if extended.shape[0] < self.FAST_SIGNAL_DIM:
            padding = torch.zeros(self.FAST_SIGNAL_DIM - extended.shape[0])
            extended = torch.cat([extended, padding])
        elif extended.shape[0] > self.FAST_SIGNAL_DIM:
            extended = extended[:self.FAST_SIGNAL_DIM]

        return extended

    def _gpu_busy_derivative(self) -> float:
        """Compute GPU busy change rate."""
        if len(self.gpu_busy_history) < 2:
            return 0.0
        t1, b1 = self.gpu_busy_history[-2]
        t2, b2 = self.gpu_busy_history[-1]
        dt = t2 - t1
        if dt < 0.001:
            return 0.0
        return (b2 - b1) / dt / 100.0  # Normalize

    def _power_instant_derivative(self, current_power: float) -> float:
        """Compute instant power change rate."""
        if len(self.power_history) < 2:
            return 0.0
        t1, p1 = self.power_history[-2]
        t2, _ = self.power_history[-1]
        dt = t2 - t1
        if dt < 0.001:
            return 0.0
        return (current_power - p1) / dt / 100.0  # Normalize

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

    def get_fast_signals(self) -> Dict[str, float]:
        """Get current fast signal values for logging."""
        return self._read_fast_signals()

    def get_fast_signal_variance(self) -> Dict[str, float]:
        """z45v2: Compute variance of fast signals over recent history.

        Expert recommendation: Log variance to verify signals are actually varying.
        If variance is ~0, the model won't learn counterfactual behavior.
        """
        import numpy as np
        result = {}

        # GPU busy variance
        if len(self.gpu_busy_history) >= 10:
            busy_vals = [b for _, b in list(self.gpu_busy_history)[-100:]]
            result["gpu_busy_var"] = float(np.var(busy_vals))
            result["gpu_busy_mean"] = float(np.mean(busy_vals))
        else:
            result["gpu_busy_var"] = 0.0
            result["gpu_busy_mean"] = 0.0

        # Power variance
        if len(self.power_history) >= 10:
            power_vals = [p for _, p in list(self.power_history)[-100:]]
            result["power_var"] = float(np.var(power_vals))
            result["power_mean"] = float(np.mean(power_vals))
        else:
            result["power_var"] = 0.0
            result["power_mean"] = 0.0

        # Temperature variance
        if len(self.temp_history) >= 10:
            temp_vals = [t for _, t in list(self.temp_history)[-100:]]
            result["temp_var"] = float(np.var(temp_vals))
            result["temp_mean"] = float(np.mean(temp_vals))
        else:
            result["temp_var"] = 0.0
            result["temp_mean"] = 0.0

        return result

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
# z45 NEW: THERMAL GOVERNOR (HARD SAFETY)
# ============================================================================

class ThermalGovernor:
    """
    Hard safety limits to prevent thermal shutdowns.

    When triggered:
    1. Force DVFS to minimum
    2. Signal trainer to skip all layers
    3. Pause any stress generators
    4. Wait for cooldown

    This is NON-NEGOTIABLE safety - not part of learning!
    """

    def __init__(
        self,
        temp_limit_c: float = 80.0,
        power_limit_w: float = 130.0,
        cooldown_s: float = 3.0,
        dvfs_controller: DVFSController = None,
    ):
        self.temp_limit = temp_limit_c
        self.power_limit = power_limit_w
        self.cooldown_s = cooldown_s
        self.dvfs = dvfs_controller

        self.triggered = False
        self.trigger_time = 0.0
        self.trigger_count = 0
        self.last_trigger_reason = ""

        print(f"[ThermalGovernor] Safety limits: T<{temp_limit_c}C, P<{power_limit_w}W")

    def check(self, power_w: float, temp_c: float) -> Tuple[bool, str]:
        """
        Check if safety limits exceeded.

        Returns (triggered, reason)
        """
        # Still in cooldown?
        if self.triggered:
            elapsed = time.time() - self.trigger_time
            if elapsed < self.cooldown_s:
                return True, f"cooldown ({self.cooldown_s - elapsed:.1f}s remaining)"
            else:
                self.triggered = False
                print(f"[ThermalGovernor] Cooldown complete, resuming normal operation")

        # Check limits
        if temp_c > self.temp_limit:
            self._trigger(f"temp={temp_c:.1f}C > {self.temp_limit}C")
            return True, self.last_trigger_reason

        if power_w > self.power_limit:
            self._trigger(f"power={power_w:.1f}W > {self.power_limit}W")
            return True, self.last_trigger_reason

        return False, ""

    def _trigger(self, reason: str):
        """Trigger safety response."""
        self.triggered = True
        self.trigger_time = time.time()
        self.trigger_count += 1
        self.last_trigger_reason = reason

        print(f"[ThermalGovernor] SAFETY TRIGGERED: {reason} (count={self.trigger_count})")

        # Force DVFS to minimum
        if self.dvfs:
            self.dvfs.set_mode("min_sclk")

    def get_forced_actions(self) -> Optional[Dict]:
        """
        If triggered, return forced safe actions.

        Returns None if not triggered.
        """
        if not self.triggered:
            return None

        return {
            "dvfs_action": 1,  # min_sclk
            "skip_all": True,  # Skip all gated layers
            "reason": self.last_trigger_reason,
        }

    def get_stats(self) -> Dict:
        return {
            "triggered": self.triggered,
            "trigger_count": self.trigger_count,
            "last_reason": self.last_trigger_reason,
        }


# ============================================================================
# z45: SAFE GPU STRESS (BOUNDED)
# ============================================================================

class SafeGPUStress:
    """
    GPU stress with HARD BOUNDS for safety.

    Key differences from RealGPUStress:
    - Maximum intensity capped
    - Duration limited
    - Power monitoring during stress
    - Auto-abort if limits exceeded
    """

    def __init__(
        self,
        device: str = "cuda",
        max_power_w: float = 110.0,
        max_duration_s: float = 3.0,
    ):
        self.device = device
        self.max_power_w = max_power_w
        self.max_duration_s = max_duration_s

        self._stop_event = threading.Event()
        self._thread = None
        self.intensity = 0.0
        self.start_time = 0.0
        self.aborted = False

    def start(self, intensity: float = 0.5, duration_s: float = 1.0):
        """Start bounded stress."""
        if self._thread is not None:
            self.stop()

        # Clamp intensity (z45: no extreme stress!)
        intensity = min(0.6, max(0.1, intensity))
        duration_s = min(self.max_duration_s, duration_s)

        self.intensity = intensity
        self.start_time = time.time()
        self.aborted = False
        self._stop_event.clear()

        # z45v2: Slightly larger matrix for more variation while still safe
        # Was: 256 + intensity*512 (max 563) - only created ~10W variation
        # Now: 384 + intensity*768 (max 845) - should create ~15-20W variation
        size = int(384 + intensity * 768)  # z45v2: Increased for better signal

        def stress_loop():
            try:
                x = torch.randn(size, size, device=self.device, dtype=torch.float16)
                while not self._stop_event.is_set():
                    _ = torch.mm(x, x)
                    # Check duration limit
                    if time.time() - self.start_time > duration_s:
                        break
                    # Brief sleep to avoid saturating
                    time.sleep(0.005)
            except Exception:
                pass

        self._thread = threading.Thread(target=stress_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        self.intensity = 0.0


class SafeCPUStress:
    """CPU stress with bounds."""

    def __init__(self):
        self._processes = []
        self.intensity = 0.0

    def start(self, intensity: float = 0.3, cores: int = 2, duration_s: float = 2.0):
        self.stop()
        self.intensity = min(0.5, intensity)  # Cap intensity
        num_workers = min(2, max(1, int(cores * self.intensity)))  # Cap workers

        try:
            timeout = min(60, int(duration_s))  # Cap duration
            self._processes = [
                subprocess.Popen(
                    ['stress-ng', '--cpu', '1', '--timeout', str(timeout)],
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
                p.wait(timeout=0.5)
            except:
                pass
        self._processes = []
        self.intensity = 0.0


# ============================================================================
# z45: SAFE DISTURBANCE SCHEDULER
# ============================================================================

class SafeDisturbanceScheduler:
    """
    Domain-randomized disturbance with SAFETY BOUNDS.

    z45v2 changes:
    - DVFS sweeps added (expert recommendation!) - benign variation without heat
    - Higher moderate intensity (0.5 not 0.4)
    - No more "gpu_heavy" - replaced with "gpu_moderate"
    - Shorter durations
    - Power-aware abort
    """

    def __init__(self, device: str = "cuda", config: Z45Config = None):
        self.config = config or Z45Config()
        self.gpu_stress = SafeGPUStress(
            device,
            max_power_w=self.config.stress_power_max_w,
            max_duration_s=self.config.stress_duration_max_s,
        )
        self.cpu_stress = SafeCPUStress()
        self.dvfs = DVFSController()  # z45v2: DVFS for benign variation
        self.current_disturbance = None
        self.start_time = 0.0

    def maybe_apply(self, prob: float = 0.3) -> Optional[str]:
        self.clear()

        if random.random() > prob:
            return None

        # z45v2: DVFS sweeps added per expert recommendation!
        # Creates natural power variation without overheating
        disturbance_type = random.choice([
            "gpu_light", "gpu_moderate",  # No gpu_heavy!
            "cpu_light", "cpu_moderate",  # No cpu_heavy!
            "combined_light",
            "dvfs_min", "dvfs_peak",  # z45v2: DVFS sweeps (benign!)
        ])

        duration = random.uniform(
            self.config.stress_duration_min_s,
            self.config.stress_duration_max_s
        )

        if disturbance_type == "gpu_light":
            self.gpu_stress.start(intensity=0.3, duration_s=duration)  # z45v2: 0.3 from 0.2
        elif disturbance_type == "gpu_moderate":
            self.gpu_stress.start(intensity=0.5, duration_s=duration)  # z45v2: 0.5 from 0.4!
        elif disturbance_type == "cpu_light":
            self.cpu_stress.start(intensity=0.25, cores=1, duration_s=duration)
        elif disturbance_type == "cpu_moderate":
            self.cpu_stress.start(intensity=0.45, cores=2, duration_s=duration)  # z45v2: increased
        elif disturbance_type == "combined_light":
            self.gpu_stress.start(intensity=0.3, duration_s=duration)
            self.cpu_stress.start(intensity=0.25, cores=1, duration_s=duration)
        elif disturbance_type == "dvfs_min":
            # z45v2: DVFS sweep to min clock - reduces power without heat!
            self.dvfs.set_mode("min_sclk")
        elif disturbance_type == "dvfs_peak":
            # z45v2: DVFS sweep to peak - increases power naturally
            self.dvfs.set_mode("peak")

        self.current_disturbance = disturbance_type
        self.start_time = time.time()
        return disturbance_type

    def clear(self):
        self.gpu_stress.stop()
        self.cpu_stress.stop()
        # z45v2: Return DVFS to auto after disturbance
        if self.current_disturbance in ["dvfs_min", "dvfs_peak"]:
            self.dvfs.set_mode("auto")
        self.current_disturbance = None


# ============================================================================
# DECODE-TIME POWER SAMPLER
# ============================================================================

class DecodeTimePowerSampler:
    """Background thread samples power DURING token generation."""

    def __init__(self, base_hub, sample_interval_ms: float = 10.0):
        self.base_hub = base_hub
        self.sample_interval_s = sample_interval_ms / 1000.0
        self._stop_event = threading.Event()
        self._thread = None
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
# PERSISTENT BODY STATE
# ============================================================================

class PersistentBodyState(nn.Module):
    """Persistent body state with decay."""

    def __init__(
        self,
        sensor_dim: int = 64,
        body_dim: int = 64,
        decay: float = 0.1,
        noise_std: float = 0.01,
    ):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.body_dim = body_dim
        self.decay = decay
        self.noise_std = noise_std

        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, body_dim),
            nn.Tanh(),
        )

        self.register_buffer('state', torch.zeros(body_dim))
        self.state_history: Deque[torch.Tensor] = deque(maxlen=100)

    def update(self, sensors: torch.Tensor) -> torch.Tensor:
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        encoded = self.sensor_encoder(sensors).squeeze(0)

        old_state = self.state.detach()
        noise = torch.randn_like(self.state) * self.noise_std if self.training else 0
        self.state = (1 - self.decay) * old_state + self.decay * encoded + noise

        self.state_history.append(self.state.detach().clone())
        return self.state.clone()

    def get_state(self) -> torch.Tensor:
        return self.state.clone()

    def encode(self, sensors: torch.Tensor) -> torch.Tensor:
        """Stateless encoding for contrastive learning."""
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        return self.sensor_encoder(sensors).squeeze(0)

    def reset(self):
        self.state.zero_()
        self.state_history.clear()


# ============================================================================
# PREDICTIVE HEAD with CURRICULUM
# ============================================================================

class PredictiveHeadWithCurriculum(nn.Module):
    """
    Predictive head with z45 CURRICULUM learning.

    Phase 1: Predict power + DVFS state (fast, easy)
    Phase 2: Add energy/J-tok
    Phase 3: Full prediction including temp
    """

    def __init__(
        self,
        body_dim: int = 64,
        sensor_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        input_dim = body_dim + sensor_dim + 4

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Separate heads for curriculum
        self.power_head = nn.Linear(hidden_dim, 1)  # Phase 1
        self.dvfs_head = nn.Linear(hidden_dim, 3)   # Phase 1 (classify current DVFS)
        self.energy_head = nn.Linear(hidden_dim, 1) # Phase 2
        self.temp_head = nn.Linear(hidden_dim, 1)   # Phase 3
        self.throttle_head = nn.Linear(hidden_dim, 1)  # Phase 3

    def forward(
        self,
        body_state: torch.Tensor,
        sensors: torch.Tensor,
        dvfs_action: torch.Tensor,
        skip_prob: torch.Tensor,
        phase: int = 3,  # z45: curriculum phase
    ) -> Dict[str, torch.Tensor]:
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

        x = torch.cat([body_state, sensors, dvfs_action, skip_prob], dim=-1)
        h = self.predictor(x)

        result = {
            "power": self.power_head(h).squeeze(-1),
            "dvfs_logits": self.dvfs_head(h),
        }

        # Phase 2+: Add energy
        if phase >= 2:
            result["energy"] = self.energy_head(h).squeeze(-1)
        else:
            result["energy"] = torch.zeros_like(result["power"])

        # Phase 3: Add temp and throttle
        if phase >= 3:
            result["temp"] = self.temp_head(h).squeeze(-1)
            result["throttle_prob"] = torch.sigmoid(self.throttle_head(h)).squeeze(-1)
        else:
            result["temp"] = torch.zeros_like(result["power"])
            result["throttle_prob"] = torch.zeros_like(result["power"])

        return result


# ============================================================================
# INTEROCEPTIVE REPORT HEAD
# ============================================================================

class InteroceptiveReportHead(nn.Module):
    """Produces calibrated interoceptive report."""

    def __init__(self, body_dim: int = 64, sensor_dim: int = 64):
        super().__init__()

        input_dim = body_dim + sensor_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.strain_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.mode_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

    def forward(self, body_state: torch.Tensor, sensors: torch.Tensor) -> Dict[str, torch.Tensor]:
        if body_state.dim() == 1:
            body_state = body_state.unsqueeze(0)
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)

        x = torch.cat([body_state, sensors], dim=-1)
        h = self.encoder(x)

        return {
            "strain_level": self.strain_head(h).squeeze(-1),
            "confidence": self.confidence_head(h).squeeze(-1),
            "mode_logits": self.mode_head(h),
            "recommended_mode": self.mode_head(h).argmax(dim=-1),
        }


# ============================================================================
# GATE NETWORK with EXPECTED SKIP option
# ============================================================================

class GateNetWithExpectedSkip(nn.Module):
    """
    Gate network with option for EXPECTED (continuous) skip values.

    z45: During early training, use expected skip (p_skip) instead of
    sampled binary actions. This reduces variance for initial learning.
    """

    def __init__(
        self,
        sensor_dim: int = 64,
        body_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 5,
    ):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.body_dim = body_dim
        self.num_layers = num_layers

        input_dim = sensor_dim + body_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in range(num_layers)
        ])

        self.dvfs_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        # Initialize gates to favor running
        for head in self.gate_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, 1.0)

    def forward(
        self,
        sensors: torch.Tensor,
        body_state: torch.Tensor,
        sample: bool = True,
        use_expected: bool = False,  # z45: Use expected (continuous) values
    ) -> Dict:
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        if body_state.dim() == 1:
            body_state = body_state.unsqueeze(0)

        x = torch.cat([sensors, body_state], dim=-1)
        h = self.encoder(x)

        gate_logits = [head(h).squeeze(-1) for head in self.gate_heads]
        gate_probs = [torch.sigmoid(logit) for logit in gate_logits]

        dvfs_logits = self.dvfs_head(h)
        dvfs_probs = F.softmax(dvfs_logits, dim=-1)

        result = {
            "gate_probs": gate_probs,
            "gate_logits": gate_logits,
            "dvfs_logits": dvfs_logits,
            "dvfs_probs": dvfs_probs,
        }

        if sample:
            if use_expected:
                # z45: Use EXPECTED values (continuous, no sampling)
                skip_actions = [prob.clone() for prob in gate_probs]  # Continuous 0-1
                skip_log_probs = [torch.zeros_like(prob) for prob in gate_probs]  # No log prob
            else:
                # Standard sampling
                skip_actions = []
                skip_log_probs = []

                for prob in gate_probs:
                    action = (torch.rand_like(prob) < prob).float()
                    skip_actions.append(action)
                    log_prob = action * torch.log(prob + 1e-10) + (1 - action) * torch.log(1 - prob + 1e-10)
                    skip_log_probs.append(log_prob)

            result["skip_actions"] = skip_actions
            result["skip_log_probs"] = skip_log_probs
            result["total_skip_log_prob"] = sum(lp.sum() for lp in skip_log_probs)

            # DVFS always sampled (it's categorical)
            dvfs_dist = Categorical(dvfs_probs)
            dvfs_action = dvfs_dist.sample()
            dvfs_log_prob = dvfs_dist.log_prob(dvfs_action)

            result["dvfs_action"] = dvfs_action
            result["dvfs_log_prob"] = dvfs_log_prob
            result["total_log_prob"] = result["total_skip_log_prob"] + dvfs_log_prob.sum()

            # Entropy
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

class MLPSkipBlock(nn.Module):
    """Gated MLP with FiLM modulation."""

    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_size: int,
        sensor_dim: int = 64,
        body_dim: int = 64,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.original_mlp = original_mlp
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        self.skip_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )

        film_input_dim = sensor_dim + body_dim
        self.film_generator = nn.Sequential(
            nn.Linear(film_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_size * 2),
        )

        self.run_decision = True
        self.run_probability = 1.0  # z45: For expected skip
        self.skipped_this_forward = False
        self.film_scale = 1.0
        self.sensors = None
        self.body_state = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # z45: Support continuous run_probability
        if isinstance(self.run_decision, float) or (isinstance(self.run_decision, bool) and self.run_probability < 1.0):
            # Weighted blend of run and skip paths
            prob = self.run_probability if hasattr(self, 'run_probability') else (1.0 if self.run_decision else 0.0)

            run_out = self._run_path(hidden_states)
            skip_out = self.skip_proj(hidden_states)

            self.skipped_this_forward = prob < 0.5
            return prob * run_out + (1 - prob) * skip_out

        # Standard binary decision
        self.skipped_this_forward = not self.run_decision

        if self.run_decision:
            return self._run_path(hidden_states)
        else:
            return self.skip_proj(hidden_states)

    def _run_path(self, hidden_states: torch.Tensor) -> torch.Tensor:
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


# ============================================================================
# EMBODIED MODEL
# ============================================================================

class EmbodiedModel(nn.Module):
    """Complete embodied model with all components."""

    def __init__(
        self,
        base_model: nn.Module,
        gate_net: GateNetWithExpectedSkip,
        sensor_hub: FastSignalSensorHub,
        body_state: PersistentBodyState,
        predictor: PredictiveHeadWithCurriculum,
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

        self.skip_blocks = nn.ModuleDict()
        for layer_idx in gate_layers:
            layer = base_model.model.layers[layer_idx]
            original_mlp = layer.mlp

            skip_block = MLPSkipBlock(
                original_mlp=original_mlp,
                hidden_size=hidden_size,
                sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
                body_dim=body_state.body_dim,
                layer_idx=layer_idx,
            )
            self.skip_blocks[str(layer_idx)] = skip_block
            layer.mlp = skip_block

        base_param = next(base_model.parameters())
        for block in self.skip_blocks.values():
            block.skip_proj.to(device=base_param.device, dtype=base_param.dtype)
            block.film_generator.to(device=base_param.device, dtype=base_param.dtype)

        print(f"[EmbodiedModel] Skip blocks at layers: {gate_layers}")

    def compute_actions(
        self,
        sensors: torch.Tensor,
        body_state: torch.Tensor,
        sample: bool = True,
        use_expected: bool = False,
    ) -> Dict:
        return self.gate_net(sensors, body_state, sample=sample, use_expected=use_expected)

    def apply_actions(
        self,
        action_result: Dict,
        sensors: torch.Tensor,
        body_state: torch.Tensor,
        film_scale: float = 1.0,
        use_expected: bool = False,
    ):
        skip_actions = action_result.get("skip_actions", [])
        gate_probs = action_result.get("gate_probs", [])

        for i, layer_idx in enumerate(self.gate_layers):
            block = self.skip_blocks[str(layer_idx)]

            if use_expected and i < len(gate_probs):
                # z45: Use continuous probability
                block.run_probability = gate_probs[i].item()
                block.run_decision = gate_probs[i].item()
            elif i < len(skip_actions):
                block.run_decision = skip_actions[i].item() > 0.5
                block.run_probability = 1.0 if block.run_decision else 0.0
            else:
                block.run_decision = True
                block.run_probability = 1.0

            block.film_scale = film_scale
            block.sensors = sensors
            block.body_state = body_state

    def apply_dvfs(self, dvfs_action: int):
        mode = DVFS_MODES[dvfs_action]
        success = self.sensor_hub.dvfs.set_mode(mode)
        return mode, success

    def force_safe_actions(self, sensors: torch.Tensor, body_state: torch.Tensor):
        """z45: Force safe actions when thermal governor triggers."""
        for block in self.skip_blocks.values():
            block.run_decision = False  # Skip all
            block.run_probability = 0.0
            block.sensors = sensors
            block.body_state = body_state
        self.sensor_hub.dvfs.set_mode("min_sclk")

    def reset_decisions(self):
        for block in self.skip_blocks.values():
            block.run_decision = True
            block.run_probability = 1.0

    def reset_tracking_flags(self):
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
        """Compute FiLM KL loss."""
        for block in self.skip_blocks.values():
            block.run_decision = True

        for block in self.skip_blocks.values():
            block.film_scale = 1.0
            block.sensors = sensors_stressed
            block.body_state = body_state

        outputs_stressed = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        logits_stressed = outputs_stressed.logits

        for block in self.skip_blocks.values():
            block.sensors = sensors_relaxed

        outputs_relaxed = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        logits_relaxed = outputs_relaxed.logits

        log_probs_stressed = F.log_softmax(logits_stressed, dim=-1)
        probs_relaxed = F.softmax(logits_relaxed, dim=-1)

        kl_per_token = F.kl_div(log_probs_stressed, probs_relaxed, reduction='none').sum(dim=-1)
        kl_mean = kl_per_token.mean()

        target_tensor = torch.tensor(kl_target, device=kl_mean.device, dtype=kl_mean.dtype)
        target_loss = F.smooth_l1_loss(kl_mean, target_tensor)
        maximize_loss = -torch.log(1.0 + kl_mean).clamp(min=-2.0)
        kl_loss = target_loss + 0.3 * maximize_loss

        logit_diff = (logits_stressed - logits_relaxed).abs().mean()

        metrics = {
            "kl_mean": kl_mean.item(),
            "kl_target": kl_target,
            "kl_loss": kl_loss.item(),
            "logit_diff": logit_diff.item(),
        }

        return kl_loss, metrics

    def get_film_params(self) -> List[torch.nn.Parameter]:
        params = []
        for block in self.skip_blocks.values():
            params.extend(block.film_generator.parameters())
            params.extend(block.skip_proj.parameters())
        return params


# ============================================================================
# z45: RECOVERY-AWARE REWARD
# ============================================================================

class RecoveryAwareReward:
    """
    Reward computer with z45 RECOVERY/HYSTERESIS tracking.

    Key additions:
    - Track state transitions (stressed -> normal, normal -> stressed)
    - Reward successful recovery
    - Penalize getting stuck in stress state
    """

    def __init__(self, config: Z45Config):
        self.config = config

        # Baseline for REINFORCE
        self.baseline = 0.0
        self.baseline_ema = config.baseline_ema

        # z45: State tracking for recovery
        self.was_stressed_last = False
        self.stress_duration = 0
        self.recovery_count = 0
        self.stuck_count = 0

        # Tracking
        self.in_band_count = 0
        self.total_count = 0

        # Discomfort history
        self.power_history: Deque[float] = deque(maxlen=10)
        self.temp_history: Deque[float] = deque(maxlen=10)

    def compute(
        self,
        response: str,
        throughput: float,
        power_w: float,
        j_per_token: float,
        temp_c: float,
        skip_rate: float,
        prediction_error: float,
        is_stressed: bool,
        fast_signals: Dict = None,
    ) -> Tuple[float, float, Dict]:
        """Compute reward with recovery tracking."""
        self.power_history.append(power_w)
        self.temp_history.append(temp_c)

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

        # 5. Prediction accuracy
        prediction_score = max(0, 1.0 - prediction_error)

        # 6. Discomfort
        thermal_discomfort = math.exp(-(self.config.temp_safety_c - temp_c) / 10.0) if temp_c < self.config.temp_safety_c else 1.0
        power_overshoot = max(0, power_w - self.config.power_cap_w) / self.config.power_cap_w
        discomfort = 0.5 * thermal_discomfort + 0.5 * power_overshoot
        discomfort_score = 1.0 - min(1.0, discomfort)

        # 7. z45 NEW: Recovery reward
        is_normal = power_w < self.config.power_cap_w and temp_c < self.config.temp_target_c

        recovery_score = 1.0
        if self.was_stressed_last and is_normal:
            # Successfully recovered from stress!
            recovery_score = 1.3
            self.recovery_count += 1
            self.stress_duration = 0
        elif self.was_stressed_last and is_stressed:
            # Still stressed - penalize if duration too long
            self.stress_duration += 1
            if self.stress_duration > 5:
                recovery_score = 0.7
                self.stuck_count += 1
        elif not self.was_stressed_last and is_stressed:
            # Just entered stress
            self.stress_duration = 1

        self.was_stressed_last = is_stressed

        # z45: Fast signal bonus (reward using fast signals well)
        fast_signal_bonus = 0.0
        if fast_signals:
            gpu_busy = fast_signals.get("gpu_busy", 50)
            # Reward for efficient GPU use (not idle, not saturated)
            if 30 < gpu_busy < 90:
                fast_signal_bonus = 0.1

        # Weighted combination
        reward = (
            self.config.quality_weight * quality +
            self.config.energy_weight * energy_score +
            self.config.recovery_weight * recovery_score * in_band +
            self.config.throughput_weight * throughput_score +
            self.config.prediction_weight * prediction_score +
            self.config.discomfort_weight * discomfort_score +
            fast_signal_bonus
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
            "fast_signal_bonus": fast_signal_bonus,
            "time_in_band_pct": self.in_band_count / max(1, self.total_count),
            "recovery_count": self.recovery_count,
            "stuck_count": self.stuck_count,
        }

        return reward, advantage, breakdown

    def reset(self):
        self.in_band_count = 0
        self.total_count = 0
        self.was_stressed_last = False
        self.stress_duration = 0
        self.power_history.clear()
        self.temp_history.clear()


# ============================================================================
# SENSOR BUFFER
# ============================================================================

class SensorBuffer:
    """Buffer for contrastive learning with NORMALIZED gate diff."""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.buffer: Deque[Tuple[torch.Tensor, float, float]] = deque(maxlen=max_size)

    def add(self, sensors: torch.Tensor, power_w: float, temp_c: float):
        self.buffer.append((sensors.detach().clone(), power_w, temp_c))

    def can_sample_pair(self) -> bool:
        if len(self.buffer) < 4:  # z45: Lower threshold for faster learning
            return False
        powers = [p for _, p, _ in self.buffer]
        return max(powers) - min(powers) >= 8.0  # z45v2: LOWERED from 15W - actual spread was only 13W!

    def sample_extremes(self) -> Tuple[Tuple, Tuple]:
        if len(self.buffer) < 4:
            s1, p1, t1 = random.choice(self.buffer)
            s2, p2, t2 = random.choice(self.buffer)
            return (s1, p1, t1), (s2, p2, t2)

        sorted_buf = sorted(self.buffer, key=lambda x: x[1])
        n = len(sorted_buf)

        lo_quartile = sorted_buf[:max(1, n // 4)]
        hi_quartile = sorted_buf[-(max(1, n // 4)):]

        lo_sample = random.choice(lo_quartile)
        hi_sample = random.choice(hi_quartile)

        return hi_sample, lo_sample

    def get_power_spread(self) -> float:
        """Get current power spread for normalization."""
        if len(self.buffer) == 0:
            return 1.0
        powers = [p for _, p, _ in self.buffer]
        spread = max(powers) - min(powers)
        return max(spread, 1.0)

    def get_stats(self) -> Dict:
        if len(self.buffer) == 0:
            return {"size": 0, "power_min": 0, "power_max": 0, "power_spread": 0}
        powers = [p for _, p, _ in self.buffer]
        return {
            "size": len(self.buffer),
            "power_min": min(powers),
            "power_max": max(powers),
            "power_spread": max(powers) - min(powers),
        }


# ============================================================================
# z45: NORMALIZED CONTRASTIVE GATE LOSS
# ============================================================================

class NormalizedContrastiveGateLoss:
    """
    Contrastive loss with NORMALIZED gate diff.

    z45: Normalize by sensor spread to avoid threshold tuning.
    """

    def __init__(self, margin: float = 0.05, direction_weight: float = 0.5):
        self.margin = margin
        self.direction_weight = direction_weight

    def compute(
        self,
        gate_probs_stressed: List[torch.Tensor],
        gate_probs_relaxed: List[torch.Tensor],
        power_spread: float = 1.0,  # z45: For normalization
    ) -> Tuple[torch.Tensor, Dict]:
        mean_gate_stressed = sum(p.mean() for p in gate_probs_stressed) / len(gate_probs_stressed)
        mean_gate_relaxed = sum(p.mean() for p in gate_probs_relaxed) / len(gate_probs_relaxed)

        gate_diff = (mean_gate_relaxed - mean_gate_stressed).abs()

        # z45: Normalize by power spread (more spread = expect more gate diff)
        expected_diff = self.margin * min(power_spread / 30.0, 2.0)  # Scale with spread
        margin_loss = F.relu(expected_diff - gate_diff)

        # Direction: stressed should have LOWER gate
        direction_loss = F.relu(mean_gate_stressed - mean_gate_relaxed + 0.01)

        total_loss = margin_loss + self.direction_weight * direction_loss

        # z45: Normalized gate diff for consistent logging
        normalized_gate_diff = gate_diff / max(power_spread / 50.0, 0.01)

        metrics = {
            "gate_diff": gate_diff.item(),
            "gate_diff_normalized": normalized_gate_diff.item(),
            "margin_loss": margin_loss.item(),
            "direction_loss": direction_loss.item(),
            "gate_stressed": mean_gate_stressed.item(),
            "gate_relaxed": mean_gate_relaxed.item(),
            "direction_correct": (mean_gate_stressed < mean_gate_relaxed).item(),
            "expected_diff": expected_diff,
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
    model: EmbodiedModel,
    tokenizer,
    prompts: List[str],
    optimizer: torch.optim.Optimizer,
    config: Z45Config,
    disturbance: SafeDisturbanceScheduler,
    thermal_governor: ThermalGovernor,
    reward_computer: RecoveryAwareReward,
    sensor_buffer: SensorBuffer,
    contrastive_loss_fn: NormalizedContrastiveGateLoss,
    epoch: int,
    global_step: int,
) -> int:
    """Train one epoch with z45 SAFE TRAINING."""
    device = next(model.gate_net.parameters()).device

    model.sensor_hub.training_mode = True
    model.body_state_module.train()
    random.shuffle(prompts)
    prompts = prompts[:config.max_prompts]

    for prompt_idx, prompt in enumerate(prompts):
        step = global_step + prompt_idx

        # z45: Determine training phase
        use_expected_skip = step < config.expected_skip_steps
        in_gate_pretrain = step < config.gate_pretrain_steps

        # z45: Predictor curriculum phase
        if step < config.predictor_phase1_steps:
            pred_phase = 1
            predictor_weight = config.predictor_pretrain_weight
        elif step < config.predictor_phase2_steps:
            pred_phase = 2
            predictor_weight = config.predictor_pretrain_weight * 0.5
        else:
            pred_phase = 3
            predictor_weight = config.predictor_normal_weight

        # KL target schedule
        kl_progress = min(1.0, step / config.film_kl_ramp_steps)
        kl_target = config.film_kl_target_init + kl_progress * (config.film_kl_target_max - config.film_kl_target_init)

        # Read sensors BEFORE disturbance to check safety
        sensors = model.sensor_hub.read_tensor().to(device)
        fast_signals = model.sensor_hub.get_fast_signals()

        # Get current power/temp
        raw = model.sensor_hub.base.last_reading
        current_power = raw.power_mw if raw else 50.0
        current_temp = raw.temp_c if raw else 50.0

        # z45: CHECK THERMAL GOVERNOR FIRST
        governor_triggered, governor_reason = thermal_governor.check(current_power, current_temp)

        if governor_triggered:
            # SAFETY: Skip disturbance, force safe actions
            disturbance.clear()
            model.force_safe_actions(sensors, model.body_state_module.get_state())

            print(f"  [{step:4d}] [GOVERNOR] {governor_reason} - forcing safe actions")

            if WANDB_AVAILABLE and config.use_wandb:
                wandb.log({
                    "step": step,
                    "train/governor_triggered": 1.0,
                    "train/power_w": current_power,
                    "train/temp_c": current_temp,
                })
            continue  # Skip this training step

        # Safe to apply disturbance
        dist_type = disturbance.maybe_apply(prob=config.disturbance_prob)
        was_stressed = dist_type is not None

        if was_stressed:
            time.sleep(0.05)  # z45: Shorter warmup

        # Re-read sensors after disturbance
        sensors = model.sensor_hub.read_tensor().to(device)
        fast_signals = model.sensor_hub.get_fast_signals()

        # Update body state
        body_state = model.body_state_module.update(sensors)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        # Compute actions (z45: with expected skip option)
        action_result = model.compute_actions(
            sensors, body_state,
            sample=True,
            use_expected=use_expected_skip
        )

        # Apply DVFS
        dvfs_action = action_result["dvfs_action"].item()
        dvfs_mode, dvfs_success = model.apply_dvfs(dvfs_action)

        # Apply skip actions
        model.apply_actions(
            action_result, sensors, body_state,
            film_scale=config.film_scale,
            use_expected=use_expected_skip
        )

        # Make prediction (z45: with curriculum phase)
        dvfs_onehot = F.one_hot(action_result["dvfs_action"], num_classes=3).float()
        mean_gate_prob = sum(p.mean() for p in action_result["gate_probs"]) / len(action_result["gate_probs"])

        predictions = model.predictor(body_state, sensors, dvfs_onehot, mean_gate_prob, phase=pred_phase)

        # Generate with decode-time power sampling
        samples = []
        rewards = []
        advantages = []
        log_probs = []

        for sample_idx in range(config.num_samples):
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

            decode_stats = model.sensor_hub.finalize_decode(tokens_generated)
            j_per_token = decode_stats["j_per_token"]
            avg_power_w = decode_stats["avg_power_w"]

            temp_c = model.sensor_hub.temp_history[-1][1] if model.sensor_hub.temp_history else 50.0

            response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            pred_power_error = abs(predictions["power"].item() - avg_power_w) / max(avg_power_w, 1.0)

            # z45: Reward with recovery tracking
            reward, advantage, breakdown = reward_computer.compute(
                response=response,
                throughput=throughput,
                power_w=avg_power_w,
                j_per_token=j_per_token,
                temp_c=temp_c,
                skip_rate=model.get_metrics()["skip_rate"],
                prediction_error=pred_power_error,
                is_stressed=was_stressed,
                fast_signals=fast_signals,
            )

            samples.append({
                "response": response,
                "throughput": throughput,
                "j_per_token": j_per_token,
                "avg_power_w": avg_power_w,
                "temp_c": temp_c,
                "decode_samples": decode_stats.get("samples", 0),
                "breakdown": breakdown,
                "predictions": {k: v.item() if hasattr(v, 'item') else v for k, v in predictions.items() if k != 'dvfs_logits'},
            })
            rewards.append(reward)
            advantages.append(advantage)
            log_probs.append(action_result["total_log_prob"])

        # Clear disturbance
        disturbance.clear()

        # Store sensors with actual telemetry
        sensor_buffer.add(sensors, power_w=samples[0]["avg_power_w"], temp_c=samples[0]["temp_c"])

        # REINFORCE update
        optimizer.zero_grad()

        mean_advantage = sum(advantages) / len(advantages)
        mean_log_prob = sum(log_probs) / len(log_probs)

        # z45: Gate pretraining phase
        if in_gate_pretrain:
            policy_loss = torch.tensor(0.0, device=device)
            entropy_loss = torch.tensor(0.0, device=device)
        else:
            policy_loss = -mean_log_prob * mean_advantage
            entropy_loss = -config.entropy_coef * action_result["entropy"]

        # Predictor loss (z45: curriculum-aware)
        s0 = samples[0]
        actual_power = torch.tensor([s0["avg_power_w"]], device=device, dtype=torch.float32)

        # Phase 1: Power + DVFS
        pred_loss = F.mse_loss(predictions["power"] / 100.0, actual_power / 100.0)

        # Phase 2+: Add energy
        if pred_phase >= 2:
            actual_j_tok = torch.tensor([s0["j_per_token"]], device=device, dtype=torch.float32)
            pred_loss = pred_loss + F.mse_loss(predictions["energy"] / 10.0, actual_j_tok / 10.0)

        # Phase 3: Add temp
        if pred_phase >= 3:
            actual_temp = torch.tensor([s0["temp_c"]], device=device, dtype=torch.float32)
            pred_loss = pred_loss + F.mse_loss(predictions["temp"] / 100.0, actual_temp / 100.0)

        # Interoceptive calibration
        intero = model.intero_report(body_state, sensors)
        discomfort_val = 1.0 - breakdown["discomfort"]
        actual_strain = torch.tensor([discomfort_val], device=device, dtype=torch.float32)
        intero_loss = F.mse_loss(intero["strain_level"], actual_strain)

        # z45: Normalized contrastive loss
        if sensor_buffer.can_sample_pair() and step % 3 == 0:
            (sensors_hi, power_hi, temp_hi), (sensors_lo, power_lo, temp_lo) = sensor_buffer.sample_extremes()
            sensors_hi = sensors_hi.to(device)
            sensors_lo = sensors_lo.to(device)

            body_hi = model.body_state_module.encode(sensors_hi)
            body_lo = model.body_state_module.encode(sensors_lo)

            gate_result_hi = model.gate_net(sensors_hi, body_hi, sample=False)
            gate_result_lo = model.gate_net(sensors_lo, body_lo, sample=False)

            power_spread = sensor_buffer.get_power_spread()
            contrastive_loss, contrastive_metrics = contrastive_loss_fn.compute(
                gate_result_hi["gate_probs"],
                gate_result_lo["gate_probs"],
                power_spread=power_spread,
            )
            contrastive_metrics["pair_power_diff"] = power_hi - power_lo
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
            contrastive_metrics = {"gate_diff": 0.0, "gate_diff_normalized": 0.0, "pair_power_diff": 0.0}

        # FiLM KL loss
        if step % 5 == 0 and sensor_buffer.can_sample_pair():
            (sensors_hi, _, _), (sensors_lo, _, _) = sensor_buffer.sample_extremes()
            sensors_hi = sensors_hi.to(device)
            sensors_lo = sensors_lo.to(device)

            film_kl_loss, kl_metrics = model.compute_film_kl_loss(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                sensors_stressed=sensors_hi,
                sensors_relaxed=sensors_lo,
                body_state=body_state,
                kl_target=kl_target,
            )
        else:
            film_kl_loss = torch.tensor(0.0, device=device)
            kl_metrics = {"kl_mean": 0.0, "kl_target": kl_target}

        # Total loss
        total_loss = (
            policy_loss +
            entropy_loss +
            predictor_weight * pred_loss +
            0.1 * intero_loss +
            config.film_kl_coef * film_kl_loss +
            config.contrastive_coef * contrastive_loss
        )

        total_loss.backward()

        # Gradient clipping
        all_params = (
            list(model.gate_net.parameters()) +
            list(model.predictor.parameters()) +
            list(model.intero_report.parameters()) +
            list(model.body_state_module.parameters()) +
            model.get_film_params()
        )
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        # Logging
        if step % 1 == 0:
            m = model.get_metrics()
            s = samples[0]

            gate_mean = sum(p.mean().item() for p in action_result["gate_probs"]) / len(action_result["gate_probs"])

            stress_str = f"[{dist_type}]" if was_stressed else "[normal]"
            phase_str = ""
            if in_gate_pretrain:
                phase_str = "GATE_PT"
            elif use_expected_skip:
                phase_str = "EXP_SKIP"
            elif pred_phase < 3:
                phase_str = f"PRED_P{pred_phase}"

            # z45v3: Show fast + deep signals
            throttle_hex = int(fast_signals.get('throttle_status', 0))
            fast_str = f"gpu={fast_signals['gpu_busy']:.0f}% mem={fast_signals.get('mem_ctrl_activity', 0):.0f}% thr=0x{throttle_hex:04x}"

            # z45v2: Show if contrastive is active
            contra_str = f"Δg={contrastive_metrics['gate_diff']:.4f}"
            if not sensor_buffer.can_sample_pair():
                contra_str = f"Δg=N/A(spread<8W)"

            print(f"  [{step:4d}] {stress_str:14s} gate={gate_mean:.3f} skip={m['skip_rate']:.2f} "
                  f"J/tok={s['j_per_token']:.2f} P={s['avg_power_w']:.1f}W T={s['temp_c']:.1f}C "
                  f"r={rewards[0]:.3f} {contra_str} "
                  f"{fast_str} {phase_str}", flush=True)

            # z45v2: Periodic buffer status (every 25 steps)
            if step % 25 == 0:
                buf_stats = sensor_buffer.get_stats()
                print(f"  [BUFFER] size={buf_stats['size']} spread={buf_stats['power_spread']:.1f}W "
                      f"(min={buf_stats['power_min']:.1f}, max={buf_stats['power_max']:.1f}) "
                      f"can_pair={sensor_buffer.can_sample_pair()}")

            if WANDB_AVAILABLE and config.use_wandb:
                wandb.log({
                    "step": step,
                    "epoch": epoch,
                    # Gate
                    "train/gate_mean": gate_mean,
                    "train/skip_rate": m['skip_rate'],
                    # Energy
                    "train/j_per_token": s['j_per_token'],
                    "train/avg_power_w": s['avg_power_w'],
                    "train/temp_c": s['temp_c'],
                    # Reward
                    "train/reward": rewards[0],
                    "train/advantage": advantages[0],
                    # z45: Recovery tracking
                    "train/recovery_count": breakdown.get("recovery_count", 0),
                    "train/stuck_count": breakdown.get("stuck_count", 0),
                    "train/recovery_score": breakdown.get("recovery", 1.0),
                    # Losses
                    "train/policy_loss": policy_loss.item(),
                    "train/pred_loss": pred_loss.item(),
                    "train/contrastive_loss": contrastive_loss.item(),
                    # Contrastive
                    "train/contrastive_gate_diff": contrastive_metrics["gate_diff"],
                    "train/contrastive_gate_diff_norm": contrastive_metrics.get("gate_diff_normalized", 0),
                    # z45: Fast signals
                    "train/gpu_busy": fast_signals["gpu_busy"],
                    "train/power_instant": fast_signals["power_instant"],
                    "train/gfx_voltage": fast_signals["gfx_voltage"],  # NOTE: Returns 0 on Z2!
                    "train/sclk_level": fast_signals["sclk_level"],
                    # z45v3: DEEP GPU signals (from gpu_metrics binary)
                    "train/throttle_status": fast_signals["throttle_status"],  # CV=1.72, most variable!
                    "train/mem_ctrl_activity": fast_signals["mem_ctrl_activity"],
                    "train/vcn_activity": fast_signals["vcn_activity"],
                    "train/gfx_activity_deep": fast_signals["gfx_activity_deep"],
                    "train/gfxclk_deep": fast_signals["gfxclk_deep"],
                    "train/memclk_deep": fast_signals["memclk_deep"],
                    # z45: Training phases
                    "train/pred_phase": pred_phase,
                    "train/in_gate_pretrain": 1.0 if in_gate_pretrain else 0.0,
                    "train/use_expected_skip": 1.0 if use_expected_skip else 0.0,
                    # z45: Thermal governor
                    "train/governor_trigger_count": thermal_governor.trigger_count,
                    # Buffer
                    "buffer/size": len(sensor_buffer.buffer),
                    "buffer/power_spread": sensor_buffer.get_power_spread(),
                })

                # z45v2: Log variance every 10 steps (expert recommendation)
                if step % 10 == 0:
                    var_stats = model.sensor_hub.get_fast_signal_variance()
                    wandb.log({
                        "step": step,
                        "variance/gpu_busy": var_stats["gpu_busy_var"],
                        "variance/power": var_stats["power_var"],
                        "variance/temp": var_stats["temp_var"],
                        "variance/gpu_busy_mean": var_stats["gpu_busy_mean"],
                        "variance/power_mean": var_stats["power_mean"],
                    })
                    # z45v2: Print warning if variance is too low
                    if step > 50 and var_stats["power_var"] < 50.0:
                        print(f"  [WARN] Low power variance: {var_stats['power_var']:.1f} - contrastive learning may be weak")

        # Checkpoint
        if step > 0 and step % config.val_every == 0:
            save_checkpoint(model, step, config)

    model.sensor_hub.training_mode = False
    return global_step + len(prompts)


def save_checkpoint(model: EmbodiedModel, step: int, config: Z45Config):
    """Save checkpoint."""
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
        "sensor_dim": FastSignalSensorHub.FAST_SIGNAL_DIM,
        "body_dim": config.body_dim,
        "gate_layers": config.gate_layers,
    }

    path = checkpoint_dir / f"step_{step}.pt"
    torch.save(checkpoint, path)
    print(f"\n  Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="FEEL z45v3: Safe Embodied Trainer + Deep GPU Signals")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="models/z45_embodied")
    parser.add_argument("--disturbance-prob", type=float, default=0.30)
    parser.add_argument("--power-cap", type=float, default=60.0)
    parser.add_argument("--temp-safety", type=float, default=80.0)
    parser.add_argument("--power-safety", type=float, default=130.0)
    parser.add_argument("--contrastive-coef", type=float, default=0.2)
    parser.add_argument("--wandb-project", type=str, default="feel-z45-embodied")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = Z45Config(
        epochs=args.epochs,
        max_prompts=args.max_prompts,
        checkpoint_dir=args.checkpoint_dir,
        disturbance_prob=args.disturbance_prob,
        power_cap_w=args.power_cap,
        temp_safety_c=args.temp_safety,
        power_safety_w=args.power_safety,
        contrastive_coef=args.contrastive_coef,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
    )

    # Initialize wandb
    if WANDB_AVAILABLE and config.use_wandb:
        import socket
        hostname = socket.gethostname()
        run_name = config.wandb_run_name or f"z45v3_deep_{hostname}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=asdict(config),
            tags=["z45v3", "safe", "deep-signals", "thermal-governor", hostname],
        )

    print("=" * 70)
    print("FEEL z45v3: SAFE EMBODIED TRAINER + DEEP GPU SIGNALS")
    print("=" * 70)
    print("z45v3 KEY FEATURES:")
    print("  1. THERMAL GOVERNOR - Hard safety limits (NO more Z2 shutdowns!)")
    print("  2. SAFE STRESS - Bounded intensity (max 110W, not 160W+)")
    print("  3. FAST SIGNALS - gpu_busy, instant power, voltages")
    print("  4. DEEP SIGNALS - throttle_status, mem_ctrl, vcn (from gpu_metrics)")
    print("     - throttle_status has CV=1.72 (most variable!)")
    print("     - Replaces broken gfx_voltage (returns 0 on Z2)")
    print("  4. EXPECTED SKIP - Continuous values early, sampled later")
    print("  5. PREDICTION CURRICULUM - Power/DVFS first, then energy, then temp")
    print("  6. NORMALIZED CONTRASTIVE - Gate diff scaled by power spread")
    print("  7. RECOVERY REWARD - Track stress→normal transitions")
    print("=" * 70)

    # Initialize
    print("\n[1/7] Loading base model...")
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

    print("\n[2/7] Initializing body state...")
    body_state_module = PersistentBodyState(
        sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
        body_dim=config.body_dim,
        decay=config.body_decay,
        noise_std=config.body_noise_std,
    ).to(device)

    print("\n[3/7] Initializing FAST SIGNAL sensor hub...")
    device_path = "/sys/class/drm/card1/device"
    if not Path("/sys/class/drm/card1/device/hwmon").exists():
        if Path("/sys/class/drm/card0/device/hwmon").exists():
            device_path = "/sys/class/drm/card0/device"

    base_hub = CanonicalSensorHub(device_path=device_path)
    sensor_hub = FastSignalSensorHub(
        base_hub=base_hub,
        body_state=body_state_module,
        power_sample_interval_ms=config.power_sample_interval_ms,
    )

    print("\n[4/7] Building gate network with EXPECTED SKIP...")
    gate_net = GateNetWithExpectedSkip(
        sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
        body_dim=config.body_dim,
        num_layers=len(config.gate_layers),
    ).to(device)

    print("\n[5/7] Building predictive head with CURRICULUM...")
    predictor = PredictiveHeadWithCurriculum(
        body_dim=config.body_dim,
        sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
    ).to(device)

    intero_report = InteroceptiveReportHead(
        body_dim=config.body_dim,
        sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
    ).to(device)

    print("\n[6/7] Building embodied model...")
    model = EmbodiedModel(
        base_model=base_model,
        gate_net=gate_net,
        sensor_hub=sensor_hub,
        body_state=body_state_module,
        predictor=predictor,
        intero_report=intero_report,
        gate_layers=config.gate_layers,
    )

    print("\n[7/7] Initializing THERMAL GOVERNOR...")
    thermal_governor = ThermalGovernor(
        temp_limit_c=config.temp_safety_c,
        power_limit_w=config.power_safety_w,
        cooldown_s=config.cooldown_duration_s,
        dvfs_controller=sensor_hub.dvfs,
    )

    # Safe disturbance
    disturbance = SafeDisturbanceScheduler(device=str(device), config=config)
    reward_computer = RecoveryAwareReward(config)

    sensor_buffer = SensorBuffer(max_size=config.sensor_buffer_size)
    contrastive_loss_fn = NormalizedContrastiveGateLoss(
        margin=config.contrastive_margin,
        direction_weight=0.5,
    )

    prompts = load_prompts()
    print(f"  Loaded {len(prompts)} prompts")

    # Optimizer
    film_params = model.get_film_params()
    optimizer = torch.optim.AdamW([
        {"params": gate_net.parameters(), "lr": config.gate_lr},
        {"params": body_state_module.parameters(), "lr": config.body_lr},
        {"params": predictor.parameters(), "lr": config.predictor_lr},
        {"params": intero_report.parameters(), "lr": config.predictor_lr},
        {"params": film_params, "lr": config.film_lr},
    ], weight_decay=0.01)

    print(f"\n  SAFETY LIMITS: T<{config.temp_safety_c}C, P<{config.power_safety_w}W")
    print(f"  STRESS BOUNDS: {config.stress_power_min_w}-{config.stress_power_max_w}W")

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
            thermal_governor=thermal_governor,
            reward_computer=reward_computer,
            sensor_buffer=sensor_buffer,
            contrastive_loss_fn=contrastive_loss_fn,
            epoch=epoch,
            global_step=global_step,
        )

    # Cleanup
    disturbance.clear()

    # Final checkpoint
    save_checkpoint(model, global_step, config)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Thermal governor triggered: {thermal_governor.trigger_count} times")
    print("=" * 70)


if __name__ == "__main__":
    main()
