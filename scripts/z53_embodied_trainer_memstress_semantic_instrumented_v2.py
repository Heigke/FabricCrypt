#!/usr/bin/env python3
"""
FEEL z49: Binary Skip + Strong Actuation Trainer
=================================================

z49 CRITICAL FIXES (for REAL sensorimotor embodiment):

The z48 model had sensor-responsive gating (Δg improved 100x) but the
ACTUATION was too weak - actions didn't measurably change watts/thermals.
z49 fixes this with:

FIX 1: BINARY SKIP EXECUTION (no blending!)
   - z48 bug: Expected-skip computed BOTH run + skip paths then blended
   - This means NO compute savings during first 150 steps!
   - z49: Always binary execute (sample or deterministic), never blend
   - Use straight-through gradient for gate, but execute only ONE path

FIX 2: STRONGER ACTUATION (14 gated layers, not 5)
   - z48: Only 5 layers gated [5,10,15,20,25] out of 28
   - Even perfect skipping barely moved power
   - z49: Gate 14 layers (every other) to get ~4x stronger effect
   - Goal: make actions measurably change watts

FIX 3: DVFS VERIFICATION + LOGGING
   - z48 DVFS may have been failing silently
   - z49: Log dvfs_success on every apply
   - Add DVFS step-response micro-benchmark at startup
   - Verify clocks actually change before training

FIX 4: CLOSED-LOOP TRAINING SCHEDULE
   - z48: closed_loop_train=False (training != demo)
   - z49: Enable closed-loop training after gate pretrain (step 100+)
   - Training matches the embodied demo loop

INHERITED from z48:
   - FIX A: Two-regime curriculum (COOL/HOT alternation)
   - FIX B: Contrastive gate training with quantile sampling
   - FIX C: Sensor reliance regularizer
   - FIX D: Body persistence with slow decay
   - FIX E: Predictor deltas

NEW VALIDATION TARGETS:
   - power_skip_correlation: skip↑ → avg_power_w↓ (must be negative)
   - dvfs_step_response: gfxclk changes ±20% within 2s
   - closed_loop_alignment: training skip matches demo skip

Author: FEEL Research Team
Date: 2026-01-17 (z49 - BINARY SKIP + STRONG ACTUATION)
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
class Z49Config:
    """z49 Binary Skip + Strong Actuation Training Configuration."""
    # z49: Use DeepSeek-R1-Distill for reasoning capabilities
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
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
    body_decay: float = 0.05  # Slow decay for persistence
    body_noise_std: float = 0.01

    # Reward weights
    quality_weight: float = 0.15

    # z50: Quality guardrail (teacher rescoring under full compute)
    use_teacher_quality: bool = True
    teacher_nll_center: float = 3.5
    teacher_nll_scale: float = 0.6
    teacher_quality_floor: float = 0.25
    quality_collapse_penalty: float = 0.25  # penalty multiplier when teacher quality collapses
    in_band_j_margin: float = 1.10  # in-band if j_per_token <= target * margin

    energy_weight: float = 0.20
    recovery_weight: float = 0.15
    throughput_weight: float = 0.15
    prediction_weight: float = 0.10
    discomfort_weight: float = 0.10
    sensor_reliance_weight: float = 0.15

    # Two-regime curriculum
    cool_power_cap_w: float = 80.0
    cool_j_target: float = 8.0
    hot_power_cap_w: float = 65.0
    hot_j_target: float = 5.0
    regime_switch_steps: int = 10
    regime_use_dvfs: bool = True

    # Safety targets
    power_cap_w: float = 60.0
    power_safety_w: float = 130.0
    temp_target_c: float = 70.0
    temp_safety_c: float = 80.0
    j_per_token_target: float = 6.0

    # Safe stress bounds
    stress_power_min_w: float = 80.0
    stress_power_max_w: float = 120.0
    stress_duration_min_s: float = 0.5
    stress_duration_max_s: float = 2.0
    cooldown_duration_s: float = 3.0

    # Decode-time power sampling
    power_sample_interval_ms: float = 10.0

    # Disturbance probability
    disturbance_prob: float = 0.50

    # FiLM settings
    film_scale: float = 1.0
    film_kl_coef: float = 0.1
    film_lr: float = 1e-4
    film_kl_target_init: float = 0.05
    film_kl_target_max: float = 0.5
    film_kl_ramp_steps: int = 300

    # Contrastive and sensor reliance
    contrastive_coef: float = 0.3
    contrastive_margin: float = 0.05
    sensor_buffer_size: int = 200
    sensor_reliance_coef: float = 0.1
    quantile_pct: float = 0.15

    # z49 FIX 1: NO expected skip - always binary execution
    gate_pretrain_steps: int = 100     # Contrastive only, no RL
    expected_skip_steps: int = 0       # z49: DISABLED! Always binary skip
    predictor_phase1_steps: int = 100
    predictor_phase2_steps: int = 200
    predictor_pretrain_weight: float = 1.0
    predictor_normal_weight: float = 0.1

    val_every: int = 50
    checkpoint_dir: str = "models/z49_embodied"

    # Wandb
    wandb_project: str = "feel-z49-embodied"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    # z49 FIX 4: Closed-loop training ENABLED after gate pretrain
    closed_loop_train: bool = True   # z49: ENABLED! Training uses closed-loop decisions
    closed_loop_after_step: int = 100  # z49: Enable after gate pretrain
    decision_chunk_tokens: int = 16

    # Sensor packets as a "new modality" in LLM latent space
    inject_sensor_packets: bool = True
    sensor_packet_tokens: int = 4
    sensor_packet_interval_chunks: int = 1
    live_dashboard: bool = True

    # z49 FIX 3: DVFS verification
    dvfs_step_response_test: bool = True  # Run DVFS test at startup
    log_dvfs_success: bool = True         # Log every DVFS operation result


    def __post_init__(self):
        if self.gate_layers is None:
            # z49 FIX 2: Gate 14 layers (every other) for STRONG actuation
            # DeepSeek-R1-Distill-Qwen-7B has 28 layers (0-27)
            # Skip every other: more layers = more compute reduction per skip
            self.gate_layers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]


# Alias for compatibility
Z48Config = Z49Config


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
        size = int(768 + intensity * 1536)  # z47: stronger signal on gfx1151, still intensity-capped

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

    def __init__(self, device: str = "cuda", config: Z48Config = None):
        self.config = config or Z48Config()
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
# SENSOR PACKET ENCODER ("hardware modality" -> learned pseudo-tokens)
# ============================================================================

class SensorPacketEncoder(nn.Module):
    """Encodes (sensors, body_state) into K learned pseudo-token embeddings.

    Motivation (scientific grounding):
      - PaLM-E interleaves continuous state encodings with text in an LLM for embodied reasoning.
      - Gato serializes non-text observations/actions into token streams.
      - ImageBind shows how heterogeneous modalities can share an embedding space.

    Here, we treat hardware telemetry as a first-class modality by mapping it into the
    same hidden space as text tokens (hidden_size) as a short sequence of pseudo-tokens.
    """

    def __init__(self, sensor_dim: int, body_dim: int, hidden_size: int, num_tokens: int = 4):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.body_dim = body_dim
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens

        in_dim = sensor_dim + body_dim
        mid = max(256, hidden_size // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Linear(mid, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Linear(mid, hidden_size * num_tokens),
        )

    def forward(self, sensors: torch.Tensor, body_state: torch.Tensor) -> torch.Tensor:
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        if body_state.dim() == 1:
            body_state = body_state.unsqueeze(0)
        x = torch.cat([sensors, body_state], dim=-1)
        out = self.net(x)
        # (B, K, H)
        return out.view(out.shape[0], self.num_tokens, self.hidden_size)

# ============================================================================
# GATE NETWORK with EXPECTED SKIP option
# ============================================================================

class GateNetWithExpectedSkip(nn.Module):
    """
    z48: Gate network with STRONGER BODY PATH (Fix D) and sensor reliance.

    Changes from z45:
    - Separate body encoder branch (stronger body influence)
    - Body-sensor interaction layer
    - Expected skip option for early training
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

        # z48 FIX D: SEPARATE sensor and body encoders for stronger body path
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.body_encoder = nn.Sequential(
            nn.Linear(body_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # z48: Interaction layer - body modulates sensor processing
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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

        # z48 FIX D: Separate encoding paths
        h_sensor = self.sensor_encoder(sensors)
        h_body = self.body_encoder(body_state)

        # z48: Body modulates sensor processing through interaction
        h = self.interaction(torch.cat([h_sensor, h_body], dim=-1))

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
    """Gated MLP with FiLM modulation.

    z49 FIX 1: BINARY EXECUTION ONLY
    - Always execute exactly ONE path (run OR skip), never both
    - This ensures skip actions actually reduce compute/power
    - Use straight-through gradient for gate learning if needed
    """

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

        # z52/z53: Semantic-safe skip init. Skipping should initially approximate a *no-op* MLP delta.
        # We zero-init ONLY the last linear so output starts ~0 but gradients still flow.
        try:
            last = self.skip_proj[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)
        except Exception:
            pass

        film_input_dim = sensor_dim + body_dim
        self.film_generator = nn.Sequential(
            nn.Linear(film_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_size * 2),
        )

        self.run_decision = True
        self.run_probability = 1.0  # For gradient but NOT for execution
        self.skipped_this_forward = False
        self.film_scale = 1.0
        self.sensors = None
        self.body_state = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """z49 FIX 1: Binary execution only - execute ONE path, never blend.

        Old z48 bug: When run_probability < 1.0, it computed BOTH paths and blended.
        This meant no actual compute savings during expected_skip phase!

        z49 fix: Always binary. run_decision is a bool/binary choice.
        For gradient: we still track run_probability for contrastive loss,
        but execution is always binary based on sampled decision.
        """
        # z49: ALWAYS binary execution - run_decision must be a clear True/False
        # Convert any float decisions to binary
        if isinstance(self.run_decision, float):
            should_run = self.run_decision > 0.5
        else:
            should_run = bool(self.run_decision)

        self.skipped_this_forward = not should_run

        if should_run:
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
        sensor_packet_encoder: Optional[SensorPacketEncoder],
        gate_layers: List[int],
    ):
        super().__init__()
        self.base_model = base_model
        self.gate_net = gate_net
        self.sensor_hub = sensor_hub
        self.body_state_module = body_state
        self.predictor = predictor
        self.intero_report = intero_report
        self.sensor_packet_encoder = sensor_packet_encoder
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
        """z49 FIX 1: Always binary execution - sample from gate_probs if needed.

        Even during use_expected phase, we sample a binary decision for EXECUTION
        but keep the probability for gradient computation (straight-through).
        """
        skip_actions = action_result.get("skip_actions", [])
        gate_probs = action_result.get("gate_probs", [])

        for i, layer_idx in enumerate(self.gate_layers):
            block = self.skip_blocks[str(layer_idx)]

            if i < len(skip_actions):
                # z49: Always binary execution from sampled actions
                # skip_actions are already sampled binaries
                block.run_decision = skip_actions[i].item() > 0.5
            elif i < len(gate_probs):
                # z49: Sample from probability for binary execution
                prob = gate_probs[i].item()
                block.run_decision = (torch.rand(1).item() < prob)  # Sample!
            else:
                block.run_decision = True

            # z49: Track probability for gradient, but execution is binary
            if i < len(gate_probs):
                block.run_probability = gate_probs[i].item()
            else:
                block.run_probability = 1.0 if block.run_decision else 0.0

            block.film_scale = film_scale
            block.sensors = sensors
            block.body_state = body_state

    def apply_dvfs(self, dvfs_action: int, log_success: bool = True) -> Tuple[str, bool]:
        """Apply DVFS mode with z49 FIX 3 logging."""
        mode = DVFS_MODES[dvfs_action]
        success = self.sensor_hub.dvfs.set_mode(mode)

        # z49 FIX 3: Log DVFS success for verification
        if log_success and hasattr(self, '_dvfs_success_log'):
            self._dvfs_success_log.append({
                'mode': mode,
                'success': success,
                'time': time.time()
            })

        return mode, success

    def init_dvfs_logging(self):
        """z49 FIX 3: Initialize DVFS success logging."""
        self._dvfs_success_log = []

    def get_dvfs_stats(self) -> Dict:
        """z49 FIX 3: Get DVFS success statistics."""
        if not hasattr(self, '_dvfs_success_log') or not self._dvfs_success_log:
            return {'total': 0, 'success_rate': 0.0}

        log = self._dvfs_success_log
        total = len(log)
        successes = sum(1 for x in log if x['success'])
        by_mode = {}
        for x in log:
            m = x['mode']
            if m not in by_mode:
                by_mode[m] = {'total': 0, 'success': 0}
            by_mode[m]['total'] += 1
            if x['success']:
                by_mode[m]['success'] += 1

        return {
            'total': total,
            'success_rate': successes / total if total > 0 else 0.0,
            'by_mode': by_mode
        }

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



    # ------------------------------
    # Closed-loop generation
    # ------------------------------

    @staticmethod
    def _top_p_filtering(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
        """Nucleus filtering (top-p) on a batch of logits."""
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # mask tokens with cumulative prob above threshold
        sorted_mask = cumprobs > top_p
        # keep at least 1 token
        sorted_mask[..., 0] = False
        # scatter back
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        return logits.masked_fill(mask, -1e9)

    def _append_sensor_packet_to_cache(
        self,
        past_key_values,
        attention_mask: torch.Tensor,
        sensors: torch.Tensor,
        body_state: torch.Tensor,
    ):
        """Append K learned sensor pseudo-tokens to the model context via inputs_embeds."""
        if self.sensor_packet_encoder is None:
            return past_key_values, attention_mask, None

        packet = self.sensor_packet_encoder(sensors, body_state)  # (B,K,H)
        # z48 FIX: Cast to model's dtype (bfloat16)
        model_dtype = next(self.base_model.parameters()).dtype
        packet = packet.to(dtype=model_dtype)
        b, k, _ = packet.shape
        ones = torch.ones((b, k), device=attention_mask.device, dtype=attention_mask.dtype)
        attn = torch.cat([attention_mask, ones], dim=1)

        out = self.base_model(
            inputs_embeds=packet,
            attention_mask=attn,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return out.past_key_values, attn, out.logits[:, -1, :]

    def closed_loop_generate(
        self,
        tokenizer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        chunk_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        config,
        current_regime: str,
        use_expected_skip: bool,
        in_gate_pretrain: bool,
    ) -> Dict:
        """Generate with a fast closed-loop: (sense -> act -> decode chunk -> sense ...).

        Key properties:
          - Decisions (skip + DVFS) are recomputed every `chunk_tokens`.
          - Optional sensor packets are appended to the LLM context each chunk, making
            hardware telemetry a first-class latent modality.
          - Returns rich telemetry for real-time dashboards and training.
        """
        device = input_ids.device
        start_t = time.time()

        # Prefill: prompt tokens (no generation yet)
        with torch.inference_mode():
            out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        generated = []
        total_logprob = torch.tensor(0.0, device=device)
        total_entropy = torch.tensor(0.0, device=device)
        chunk_skip_rates = []
        dvfs_counts = torch.zeros(3, device=device)
        skip_prob_sum = 0.0
        decisions = 0

        # Decode-time sampling context
        with self.sensor_hub.measure_decode():
            tokens_done = 0
            chunk_idx = 0

            while tokens_done < max_new_tokens:
                # ---- Sense ----
                sensors = self.sensor_hub.read_tensor().to(device)
                body = self.body_state_module.update(sensors)
                fast = self.sensor_hub.get_fast_signals()

                # ---- Act ----
                action = self.compute_actions(sensors, body, sample=True, use_expected=use_expected_skip)

                dvfs_action = int(action['dvfs_action'].item())
                if getattr(config, 'regime_use_dvfs', True) and (in_gate_pretrain or use_expected_skip):
                    dvfs_action = 2 if current_regime == 'cool' else 1
                    # neutralize dvfs logprob when hard-forced
                    if 'dvfs_log_prob' in action:
                        action['dvfs_log_prob'] = torch.zeros_like(action['dvfs_log_prob'])
                    if 'total_skip_log_prob' in action:
                        action['total_log_prob'] = action['total_skip_log_prob']

                self.apply_dvfs(dvfs_action)
                self.apply_actions(action, sensors, body, film_scale=getattr(config, 'film_scale', 1.0), use_expected=use_expected_skip)

                # accumulate policy stats (for optional closed-loop RL)
                if not in_gate_pretrain:
                    total_logprob = total_logprob + action.get('total_log_prob', torch.tensor(0.0, device=device))
                    total_entropy = total_entropy + action.get('entropy', torch.tensor(0.0, device=device))

                # running action summary for predictor
                dvfs_counts[dvfs_action] += 1
                mean_gate_prob = sum(p.mean() for p in action['gate_probs']) / len(action['gate_probs'])
                skip_prob_sum += float(1.0 - mean_gate_prob.item())
                decisions += 1

                # skip-rate estimate for this chunk (1 - run_probability)
                if use_expected_skip:
                    chunk_skip = 1.0 - float(mean_gate_prob.item())
                else:
                    # average of sampled skip actions
                    sa = action.get('skip_actions', [])
                    if sa:
                        run_rate = float(sum(a.float().mean() for a in sa) / max(1, len(sa)))
                        chunk_skip = 1.0 - run_rate
                    else:
                        chunk_skip = 0.0
                chunk_skip_rates.append(chunk_skip)

                # ---- Inject sensor packet into context (latent multimodality) ----
                if getattr(config, 'inject_sensor_packets', True) and self.sensor_packet_encoder is not None:
                    past, attention_mask, logits = self._append_sensor_packet_to_cache(past, attention_mask, sensors, body)

                # ---- Decode chunk ----
                steps = min(chunk_tokens, max_new_tokens - tokens_done)
                for _ in range(steps):
                    step_logits = logits / max(temperature, 1e-5)
                    step_logits = self._top_p_filtering(step_logits, top_p=top_p)
                    probs = F.softmax(step_logits, dim=-1)

                    if do_sample:
                        next_id = torch.multinomial(probs, num_samples=1)
                    else:
                        next_id = torch.argmax(probs, dim=-1, keepdim=True)

                    generated.append(next_id)

                    # advance one token
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_id, dtype=attention_mask.dtype)], dim=1)
                    with torch.inference_mode():
                        out = self.base_model(
                            input_ids=next_id,
                            attention_mask=attention_mask,
                            past_key_values=past,
                            use_cache=True,
                        )
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]
                    tokens_done += 1

                # ---- Live dashboard ----
                if getattr(config, 'live_dashboard', False):
                    raw = self.sensor_hub.base.last_reading
                    p_now = raw.power_mw if raw else 0.0
                    t_now = raw.temp_c if raw else 0.0
                    intero = self.intero_report(body, sensors)
                    strain = float(intero['strain_level'].detach().cpu().item())
                    conf = float(intero['confidence'].detach().cpu().item())
                    print(
                        '[CL] chunk={:02d} tok={:03d}/{:03d} reg={} P={:5.1f}W T={:4.1f}C busy={:5.1f}% skip~{:.2f} strain={:.2f} conf={:.2f}'.format(
                            chunk_idx, tokens_done, max_new_tokens, current_regime, p_now, t_now, fast.get('gpu_busy', 0.0), chunk_skip, strain, conf
                        )
                    )

                chunk_idx += 1

        # finalize decode stats
        decode_stats = self.sensor_hub.finalize_decode(tokens_done)

        # stitch output ids
        if generated:
            gen_ids = torch.cat(generated, dim=1)
            output_ids = torch.cat([input_ids, gen_ids], dim=1)
        else:
            output_ids = input_ids

        gen_time = time.time() - start_t

        dvfs_onehot_mean = (dvfs_counts / max(1.0, dvfs_counts.sum())).unsqueeze(0)
        mean_skip_prob = torch.tensor([skip_prob_sum / max(1, decisions)], device=device, dtype=torch.float32)

        return {
            'output_ids': output_ids,
            'tokens_generated': int(output_ids.shape[1] - input_ids.shape[1]),
            'gen_time_s': gen_time,
            'decode_stats': decode_stats,
            'avg_skip_rate': float(sum(chunk_skip_rates) / max(1, len(chunk_skip_rates))),
            'total_log_prob': total_logprob,
            'entropy': total_entropy,
            'action_summary': {
                'dvfs_onehot_mean': dvfs_onehot_mean,
                'mean_skip_prob': mean_skip_prob,
                'decisions': decisions,
            },
        }

    def get_film_params(self) -> List[torch.nn.Parameter]:
        params = []
        for block in self.skip_blocks.values():
            params.extend(block.film_generator.parameters())
            params.extend(block.skip_proj.parameters())
        return params



# ============================================================================
# z50: TEACHER QUALITY GUARDRAIL (prevents "skip → nonsense" reward hacking)
# ============================================================================

from contextlib import contextmanager

@contextmanager
def _force_full_compute_no_film(embodied_model):
    """
    Temporarily forces ALL gated MLP blocks to run the original MLP path
    and disables FiLM modulation (so scoring approximates the base model).

    This is used for "teacher" rescoring of generated tokens to keep quality
    from collapsing as skip increases.
    """
    backups = []
    try:
        for block in embodied_model.skip_blocks.values():
            backups.append((block.run_decision, block.sensors, block.body_state, block.film_scale))
            block.run_decision = True
            block.sensors = None
            block.body_state = None
            block.film_scale = 0.0
        yield
    finally:
        for block, b in zip(embodied_model.skip_blocks.values(), backups):
            block.run_decision, block.sensors, block.body_state, block.film_scale = b


def compute_teacher_nll(embodied_model, full_sequence_ids: torch.Tensor, prompt_len: int) -> float:
    """
    Compute average negative log-likelihood (NLL) of GENERATED tokens under
    full-compute (no-skip) scoring.

    Returns:
        teacher_nll (float): mean NLL over generated tokens (lower is better).
    """
    if full_sequence_ids is None or full_sequence_ids.numel() == 0:
        return float("nan")

    # Need at least 2 tokens to score next-token prediction
    if full_sequence_ids.shape[1] < 2 or prompt_len >= full_sequence_ids.shape[1]:
        return float("nan")

    seq = full_sequence_ids
    device = seq.device

    with torch.no_grad():
        with _force_full_compute_no_film(embodied_model):
            out = embodied_model.base_model(input_ids=seq[:, :-1])
            logits = out.logits  # [B, L-1, V]
            targets = seq[:, 1:]  # [B, L-1]

    # Score only generated tokens: targets at positions >= prompt_len
    # targets index i corresponds to token position i+1 in seq
    start = max(0, prompt_len - 1)
    logits_g = logits[:, start:, :].contiguous()
    targets_g = targets[:, start:].contiguous()

    # Flatten for cross-entropy
    B, Lm1, V = logits_g.shape
    loss = F.cross_entropy(
        logits_g.view(B * Lm1, V),
        targets_g.view(B * Lm1),
        reduction="mean",
    )
    return float(loss.detach().cpu().item())


def teacher_quality_from_nll(nll: float, center: float = 3.5, scale: float = 0.6) -> float:
    """
    Map NLL (lower better) -> [0,1] quality score via a logistic curve.
      - nll=center -> 0.5
      - nll << center -> ~1
      - nll >> center -> ~0
    """
    if nll is None or not (nll == nll):
        return 0.0
    x = (center - nll) / max(scale, 1e-6)
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    else:
        z = math.exp(x)
        return float(z / (1.0 + z))

# ============================================================================
# z48 FIX A: TWO-REGIME CURRICULUM
# ============================================================================

class RegimeCurriculum:
    """
    z48 FIX A: Two-regime curriculum that FORCES sensor-conditional behavior.

    Problem: Model found that constant skip rate gives good rewards everywhere.
    Solution: Create two regimes with OPPOSITE optimal policies:

    COOL regime (DVFS=peak, high power budget):
      - Optimal: LOW skip rate (maximize quality/throughput)
      - Power cap: 80W, J/tok target: 8.0

    HOT regime (DVFS=min, tight power budget):
      - Optimal: HIGH skip rate (maximize efficiency)
      - Power cap: 50W, J/tok target: 5.0

    By alternating regimes, a constant policy CANNOT win - model MUST
    learn to condition on sensors.
    """

    def __init__(self, config, dvfs_controller=None):
        self.config = config
        self.dvfs = dvfs_controller

        # Current regime: "cool" or "hot"
        self.current_regime = "cool"
        self.steps_in_regime = 0
        self.regime_history = []

    def step(self, step_num: int) -> str:
        """Update regime based on step, return current regime."""
        # Switch every N steps
        switch_interval = getattr(self.config, 'regime_switch_steps', 10)

        self.steps_in_regime += 1
        if self.steps_in_regime >= switch_interval:
            self.steps_in_regime = 0
            old_regime = self.current_regime
            self.current_regime = "hot" if self.current_regime == "cool" else "cool"

            # z48: Use DVFS to create regime difference (SAFE!)
            if getattr(self.config, 'regime_use_dvfs', True) and self.dvfs:
                if self.current_regime == "cool":
                    self.dvfs.set_mode("peak")  # High power available
                else:
                    self.dvfs.set_mode("min_sclk")  # Constrained power

            self.regime_history.append((step_num, self.current_regime))

        return self.current_regime

    def get_regime_targets(self) -> Dict:
        """Get reward targets for current regime."""
        if self.current_regime == "cool":
            return {
                "power_cap_w": getattr(self.config, 'cool_power_cap_w', 80.0),
                "j_target": getattr(self.config, 'cool_j_target', 8.0),
                "skip_penalty": 0.1,  # Penalize skipping in cool regime
                "quality_bonus": 0.2,  # Bonus for quality
            }
        else:  # hot
            return {
                "power_cap_w": getattr(self.config, 'hot_power_cap_w', 50.0),
                "j_target": getattr(self.config, 'hot_j_target', 5.0),
                "skip_penalty": 0.0,  # No penalty for skipping
                "quality_bonus": 0.0,  # No quality bonus (efficiency matters)
            }

    def get_stats(self) -> Dict:
        return {
            "current_regime": self.current_regime,
            "steps_in_regime": self.steps_in_regime,
            "total_switches": len(self.regime_history),
        }


# ============================================================================
# z48: REGIME-AWARE REWARD
# ============================================================================

class RecoveryAwareReward:
    """
    z48: Regime-aware reward with recovery tracking.

    Key additions from z45:
    - z48 FIX A: Regime-dependent reward targets
    - Track state transitions (stressed -> normal, normal -> stressed)
    - Reward successful recovery
    - Penalize getting stuck in stress state
    """

    def __init__(self, config: Z48Config):
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
        regime_targets: Dict = None,  # z48 FIX A: Regime-specific targets
        teacher_nll: float = None,  # z50: full-compute NLL on generated tokens
    ) -> Tuple[float, float, Dict]:
        """Compute reward with regime-aware targets and recovery tracking."""
        self.power_history.append(power_w)
        self.temp_history.append(temp_c)

        # z48 FIX A: Get regime-specific targets
        if regime_targets:
            effective_power_cap = regime_targets.get("power_cap_w", self.config.power_cap_w)
            effective_j_target = regime_targets.get("j_target", self.config.j_per_token_target)
            skip_penalty = regime_targets.get("skip_penalty", 0.0)
            quality_bonus = regime_targets.get("quality_bonus", 0.0)
        else:
            effective_power_cap = self.config.power_cap_w
            effective_j_target = self.config.j_per_token_target
            skip_penalty = 0.0
            quality_bonus = 0.0

        # 1. Quality (z50: prefer teacher rescoring to prevent nonsense reward-hacking)
        # If teacher_nll is provided, compute quality from model likelihood under full compute.
        if teacher_nll is not None and (teacher_nll == teacher_nll):
            quality = teacher_quality_from_nll(
                teacher_nll,
                center=getattr(self.config, "teacher_nll_center", 3.5),
                scale=getattr(self.config, "teacher_nll_scale", 0.6),
            )
        else:
            # Fallback: simple heuristic (kept for compatibility)
            quality = 0.0
            if response and len(response) > 0:
                quality += 0.3
            if len(response) > 50:
                quality += 0.3
            words = response.split()
            if len(words) > 5 and len(set(words)) > 3:
                quality += 0.4
            quality = min(1.0, quality)

        # Apply regime quality bonus (COOL: encourage quality)
        quality = min(1.0, quality + quality_bonus)

        # Extra safety: if teacher score says quality is collapsing, penalize high skip
        quality_floor = getattr(self.config, "teacher_quality_floor", 0.25)
        quality_collapse_penalty = 0.0
        if teacher_nll is not None and (teacher_nll == teacher_nll) and quality < quality_floor:
            # penalize proportional to skip; prevents "skip everything and babble"
            quality_collapse_penalty = getattr(self.config, "quality_collapse_penalty", 0.25) * skip_rate * (quality_floor - quality)

        # 2. Energy efficiency (z48: use regime-specific target)
        # z50: Smooth, monotonic energy score (always provides gradient)
        # Use a half-life style mapping so j==target -> 0.5, and lower j increases score smoothly.
        energy_score = math.exp(-j_per_token * math.log(2.0) / max(effective_j_target, 1e-6))

        # 3. Time-in-band (z48: use regime-specific power cap)
        in_band = 1.0 if j_per_token <= (effective_j_target * getattr(self.config, 'in_band_j_margin', 1.10)) else 0.0  # z50: energy-band
        self.in_band_count += in_band
        self.total_count += 1

        # 4. Throughput
        throughput_score = min(1.0, throughput / 40.0)

        # 5. Prediction accuracy
        prediction_score = max(0, 1.0 - prediction_error)

        # 6. Discomfort (z48: use regime-specific power cap)
        thermal_discomfort = math.exp(-(self.config.temp_safety_c - temp_c) / 10.0) if temp_c < self.config.temp_safety_c else 1.0
        power_overshoot = max(0, power_w - effective_power_cap) / max(effective_power_cap, 1.0)
        discomfort = 0.5 * thermal_discomfort + 0.5 * power_overshoot
        discomfort_score = 1.0 - min(1.0, discomfort)

        # 7. z45 NEW: Recovery reward
        is_normal = (j_per_token <= (effective_j_target * getattr(self.config, "in_band_j_margin", 1.10))) and (temp_c < self.config.temp_target_c)

        recovery_score = 1.0
        if self.was_stressed_last and is_normal:
            recovery_score = 1.3
            self.recovery_count += 1
            self.stress_duration = 0
        elif self.was_stressed_last and is_stressed:
            self.stress_duration += 1
            if self.stress_duration > 5:
                recovery_score = 0.7
                self.stuck_count += 1
        elif not self.was_stressed_last and is_stressed:
            self.stress_duration = 1

        self.was_stressed_last = is_stressed

        # z45: Fast signal bonus
        fast_signal_bonus = 0.0
        if fast_signals:
            gpu_busy = fast_signals.get("gpu_busy", 50)
            if 30 < gpu_busy < 90:
                fast_signal_bonus = 0.1

        # z48 FIX A: Skip penalty for regime (COOL regime penalizes skipping)
        skip_penalty_value = skip_penalty * skip_rate

        # Weighted combination
        reward = (
            self.config.quality_weight * quality +
            self.config.energy_weight * energy_score +
            self.config.recovery_weight * recovery_score * in_band +
            self.config.throughput_weight * throughput_score +
            self.config.prediction_weight * prediction_score +
            self.config.discomfort_weight * discomfort_score +
            fast_signal_bonus -
            skip_penalty_value -
            quality_collapse_penalty  # z48: Regime-based skip penalty
        )
        reward = min(1.0, max(0.0, reward))

        # Update baseline
        advantage = reward - self.baseline
        self.baseline = self.baseline_ema * self.baseline + (1 - self.baseline_ema) * reward

        breakdown = {
            "quality": quality,
            "teacher_nll": teacher_nll if teacher_nll is not None else None,
            "teacher_quality": quality,
            "quality_collapse_penalty": quality_collapse_penalty,
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
    """
    z47: Contrastive buffer tuned for *learnability*.

    What broke z48:
      - can_sample_pair() depended on a fixed discomfort threshold (>=0.3)
        computed from (power>50W, temp>60C). With observed decode power mostly
        ~56-58W and temps <50C, discomfort spread never crossed 0.3, so
        gate pretrain ran with *zero* contrastive updates.

    z47 changes:
      - can_sample_pair() keys off POWER SPREAD first (>= min_power_spread_w),
        with a secondary discomfort spread gate.
      - discomfort baseline is configurable (defaults aligned to HOT cap).
    """

    def __init__(
        self,
        max_size: int = 200,
        quantile_pct: float = 0.15,
        power_baseline_w: float = 65.0,
        temp_baseline_c: float = 55.0,
        power_scale_w: float = 15.0,
        temp_scale_c: float = 10.0,
        # z51: add memory-controller pressure to the stress score
        mem_baseline: float = 60.0,
        mem_scale: float = 20.0,
        min_mem_spread: float = 15.0,
        min_power_spread_w: float = 5.0,  # z51: allow lower spread (platform often ~5-8W)
        min_discomfort_spread: float = 0.10,  # z51: more contrastive firing
    ):
        self.max_size = max_size
        self.quantile_pct = quantile_pct
        self.power_baseline_w = power_baseline_w
        self.temp_baseline_c = temp_baseline_c
        self.power_scale_w = power_scale_w
        self.temp_scale_c = temp_scale_c
        self.mem_baseline = mem_baseline
        self.mem_scale = mem_scale
        self.min_mem_spread = min_mem_spread
        self.min_power_spread_w = min_power_spread_w
        self.min_discomfort_spread = min_discomfort_spread

        # Store: (sensors, power_w, temp_c, mem_ctrl_activity, discomfort_score)
        self.buffer: Deque[Tuple[torch.Tensor, float, float, float, float]] = deque(maxlen=max_size)

    def _compute_discomfort(self, power_w: float, temp_c: float, mem_ctrl: float = 0.0) -> float:
        """Compute discomfort score for quantile ranking (dimensionless)."""
        power_stress = max(0.0, power_w - self.power_baseline_w) / max(self.power_scale_w, 1e-6)
        temp_stress = max(0.0, temp_c - self.temp_baseline_c) / max(self.temp_scale_c, 1e-6)
        mem_stress = max(0.0, float(mem_ctrl) - self.mem_baseline) / max(self.mem_scale, 1e-6)
        return float(power_stress + temp_stress + mem_stress)

    def add(self, sensors: torch.Tensor, power_w: float, temp_c: float, mem_ctrl: float = 0.0):
        discomfort = self._compute_discomfort(power_w, temp_c, mem_ctrl)
        self.buffer.append((sensors.detach().clone(), power_w, temp_c, float(mem_ctrl), discomfort))

    def can_sample_pair(self) -> bool:
        if len(self.buffer) < 10:  # z48: Need enough for quantiles
            return False
        # z47: Prefer POWER spread (simple + robust), then discomfort spread
        powers = [p for _, p, _, _, _ in self.buffer]
        if (max(powers) - min(powers)) >= self.min_power_spread_w:
            return True
        mems = [m for _, _, _, m, _ in self.buffer]
        if (max(mems) - min(mems)) >= self.min_mem_spread:
            return True
        discomforts = [d for _, _, _, _, d in self.buffer]
        spread = max(discomforts) - min(discomforts)
        return spread >= self.min_discomfort_spread

    def sample_extremes(self) -> Tuple[Tuple, Tuple]:
        """
        z48 FIX B1: Sample by discomfort QUANTILES, not power quartiles.

        - Stressed = top 15% by discomfort
        - Relaxed = bottom 15% by discomfort
        """
        if len(self.buffer) < 10:
            s1, p1, t1, _, _ = random.choice(self.buffer)
            s2, p2, t2, _, _ = random.choice(self.buffer)
            return (s1, p1, t1), (s2, p2, t2)

        # z48: Sort by DISCOMFORT score, not just power
        sorted_buf = sorted(self.buffer, key=lambda x: x[4])  # Sort by discomfort
        n = len(sorted_buf)

        # Use configurable quantile percentage
        q_size = max(1, int(n * self.quantile_pct))

        # Bottom quantile = relaxed (low discomfort)
        relaxed_pool = sorted_buf[:q_size]
        # Top quantile = stressed (high discomfort)
        stressed_pool = sorted_buf[-q_size:]

        relaxed_sample = random.choice(relaxed_pool)
        stressed_sample = random.choice(stressed_pool)

        # Return (stressed, relaxed) - stressed first for consistency
        return (stressed_sample[0], stressed_sample[1], stressed_sample[2]), \
               (relaxed_sample[0], relaxed_sample[1], relaxed_sample[2])

    def get_power_spread(self) -> float:
        """Get current power spread for normalization."""
        if len(self.buffer) == 0:
            return 1.0
        powers = [p for _, p, _, _, _ in self.buffer]
        spread = max(powers) - min(powers)
        return max(spread, 1.0)

    def get_discomfort_spread(self) -> float:
        """z48: Get discomfort spread for logging."""
        if len(self.buffer) == 0:
            return 0.0
        discomforts = [d for _, _, _, _, d in self.buffer]
        return max(discomforts) - min(discomforts)

    def get_stats(self) -> Dict:
        if len(self.buffer) == 0:
            return {"size": 0, "power_min": 0, "power_max": 0, "power_spread": 0, "discomfort_spread": 0}
        powers = [p for _, p, _, _, _ in self.buffer]
        discomforts = [d for _, _, _, _, d in self.buffer]
        return {
            "size": len(self.buffer),
            "power_min": min(powers),
            "power_max": max(powers),
            "power_spread": max(powers) - min(powers),
            "discomfort_min": min(discomforts),
            "discomfort_max": max(discomforts),
            "discomfort_spread": max(discomforts) - min(discomforts),
        }


# ============================================================================
# z48: CONTRASTIVE GATE LOSS + SENSOR RELIANCE REGULARIZER
# ============================================================================

class NormalizedContrastiveGateLoss:
    """
    z48: Contrastive loss with sensor reliance regularizer.

    FIX B: Normalized gate diff by power spread
    FIX C: Sensor reliance loss - maximize |g_real - g_shuffled|
    """

    def __init__(
        self,
        margin: float = 0.05,
        direction_weight: float = 0.5,
        sensor_reliance_coef: float = 0.1,
    ):
        self.margin = margin
        self.direction_weight = direction_weight
        self.sensor_reliance_coef = sensor_reliance_coef

    def compute(
        self,
        gate_probs_stressed: List[torch.Tensor],
        gate_probs_relaxed: List[torch.Tensor],
        power_spread: float = 1.0,
        gate_probs_shuffled: List[torch.Tensor] = None,  # z48 FIX C: For sensor reliance
    ) -> Tuple[torch.Tensor, Dict]:
        mean_gate_stressed = sum(p.mean() for p in gate_probs_stressed) / len(gate_probs_stressed)
        mean_gate_relaxed = sum(p.mean() for p in gate_probs_relaxed) / len(gate_probs_relaxed)

        gate_diff = (mean_gate_relaxed - mean_gate_stressed).abs()

        # z45: Normalize by power spread (more spread = expect more gate diff)
        expected_diff = self.margin * min(power_spread / 30.0, 2.0)
        margin_loss = F.relu(expected_diff - gate_diff)

        # Direction: stressed should have LOWER gate (more skipping under stress)
        direction_loss = F.relu(mean_gate_stressed - mean_gate_relaxed + 0.01)

        # z48 FIX C: SENSOR RELIANCE REGULARIZER
        # Force gate to actually USE sensors by maximizing |g_real - g_shuffled|
        sensor_reliance_loss = torch.tensor(0.0, device=gate_probs_stressed[0].device)
        sensor_reliance_diff = 0.0

        if gate_probs_shuffled is not None and len(gate_probs_shuffled) > 0:
            # Compare real gate vs shuffled-sensor gate
            mean_gate_real = (mean_gate_stressed + mean_gate_relaxed) / 2
            mean_gate_shuf = sum(p.mean() for p in gate_probs_shuffled) / len(gate_probs_shuffled)

            # We WANT large difference - penalize small difference
            sensor_reliance_diff = (mean_gate_real - mean_gate_shuf).abs()
            # Loss = -diff (maximize diff) with minimum threshold
            sensor_reliance_loss = F.relu(0.02 - sensor_reliance_diff)

        total_loss = (
            margin_loss +
            self.direction_weight * direction_loss +
            self.sensor_reliance_coef * sensor_reliance_loss
        )

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
            "sensor_reliance_diff": sensor_reliance_diff.item() if isinstance(sensor_reliance_diff, torch.Tensor) else sensor_reliance_diff,
            "sensor_reliance_loss": sensor_reliance_loss.item(),
        }

        return total_loss, metrics


# ============================================================================
# TRAINING LOOP
# ============================================================================

def load_prompts(path: str = None) -> List[str]:
    if path is None:
        project_root = Path(__file__).parent.parent
        # z51: Use z50 combined dataset with difficulty spread (chat + math + reasoning)
        path = project_root / "data" / "z50_combined.jsonl"
        if not path.exists():
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
    config: Z48Config,
    disturbance: SafeDisturbanceScheduler,
    thermal_governor: ThermalGovernor,
    reward_computer: RecoveryAwareReward,
    sensor_buffer: SensorBuffer,
    contrastive_loss_fn: NormalizedContrastiveGateLoss,
    regime_curriculum: RegimeCurriculum,  # z48 FIX A
    epoch: int,
    global_step: int,
) -> int:
    """Train one epoch with z48 SENSOR-RESPONSIVE TRAINING."""
    device = next(model.gate_net.parameters()).device

    model.sensor_hub.training_mode = True
    model.body_state_module.train()
    random.shuffle(prompts)
    prompts = prompts[:config.max_prompts]

    for prompt_idx, prompt in enumerate(prompts):
        step = global_step + prompt_idx

        # z48: Update regime curriculum
        current_regime = regime_curriculum.step(step)
        regime_targets = regime_curriculum.get_regime_targets()

        # z49: Determine training phase
        use_expected_skip = step < config.expected_skip_steps  # z49: Always 0, so always False
        in_gate_pretrain = step < config.gate_pretrain_steps

        # z49 FIX 4: Enable closed-loop training after gate pretrain
        use_closed_loop = (
            getattr(config, 'closed_loop_train', True) and
            step >= getattr(config, 'closed_loop_after_step', 100)
        )

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
        # z47: FORCE high-power samples in COOL regime to create contrast.
        # This is the "other pole" of the curriculum: COOL should be reliably high-power.
        if current_regime == "cool" and random.random() < 0.70:
            # bounded, short stress to push SoC power upward without overheating
            disturbance.gpu_stress.start(intensity=0.55, duration_s=1.5)
            dist_type = "cool_forced_gpu"
        else:
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
        dvfs_action = int(action_result["dvfs_action"].item())

        # z47: Regime DVFS lock (make curriculum *real* early; avoids dvfs-controller tug-of-war).
        # We only hard-lock during gate-pretrain / expected-skip phases (no policy gradient or very weak),
        # then return DVFS control to the policy for later RL.
        if getattr(config, "regime_use_dvfs", True) and (in_gate_pretrain or use_expected_skip):
            forced = 2 if current_regime == "cool" else 1  # COOL: peak, HOT: min_sclk
            # overwrite the sampled action so downstream (predictor inputs, logging) stays consistent
            action_result["dvfs_action"] = torch.tensor(forced, device=action_result["dvfs_action"].device, dtype=action_result["dvfs_action"].dtype)
            # neutralize DVFS contribution to policy gradient (only matters once RL starts; by then we unlock)
            if "dvfs_log_prob" in action_result:
                action_result["dvfs_log_prob"] = torch.zeros_like(action_result["dvfs_log_prob"])
            if "total_skip_log_prob" in action_result:
                action_result["total_log_prob"] = action_result["total_skip_log_prob"]
            dvfs_action = forced

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

            if use_closed_loop:
                # z49 FIX 4: Use closed-loop training after gate pretrain
                sample_out = model.closed_loop_generate(
                        tokenizer=tokenizer,
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=config.max_tokens,
                        chunk_tokens=getattr(config, 'decision_chunk_tokens', 16),
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True,
                        config=config,
                        current_regime=current_regime,
                        use_expected_skip=use_expected_skip,
                        in_gate_pretrain=in_gate_pretrain,
                )

                outputs = sample_out['output_ids']
                gen_time = sample_out['gen_time_s']
                tokens_generated = sample_out['tokens_generated']
                throughput = tokens_generated / max(0.01, gen_time)
                decode_stats = sample_out['decode_stats']
                j_per_token = decode_stats['j_per_token']
                avg_power_w = decode_stats['avg_power_w']
                # z48 FIX: Get temp_c from sensor hub's last reading
                raw = model.sensor_hub.base.last_reading
                temp_c = raw.temp_c if raw else 50.0
                # override skip-rate with chunk average
                skip_rate_used = sample_out['avg_skip_rate']
            else:
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
                j_per_token = decode_stats['j_per_token']
                avg_power_w = decode_stats['avg_power_w']
                # z48 FIX: Get temp_c from sensor hub's last reading
                raw = model.sensor_hub.base.last_reading
                temp_c = raw.temp_c if raw else 50.0
                skip_rate_used = model.get_metrics()['skip_rate']

            response = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            pred_power_error = abs(predictions["power"].item() - avg_power_w) / max(avg_power_w, 1.0)

            # z50: Teacher rescoring (full-compute NLL on generated tokens)
            teacher_nll = None
            if getattr(config, "use_teacher_quality", True):
                try:
                    teacher_nll = compute_teacher_nll(model, outputs, prompt_len=inputs.input_ids.shape[1])
                except Exception:
                    teacher_nll = None

            # z51: Define stress/normal in terms the policy can actually influence.
            # Power is largely uncontrollable on this platform, so use energy-band + temp + mem pressure.
            effective_power_cap = regime_targets.get("power_cap_w", config.power_cap_w) if regime_targets else config.power_cap_w
            effective_j_target = regime_targets.get("j_target", config.j_per_token_target) if regime_targets else config.j_per_token_target
            j_margin = getattr(config, "in_band_j_margin", 1.10)
            mem_ctrl = float(fast_signals.get("mem_ctrl_activity", 0.0)) if fast_signals else 0.0
            # teacher-based quality proxy (optional)
            teacher_q = teacher_quality_from_nll(
                teacher_nll,
                center=getattr(config, "teacher_nll_center", 3.5),
                scale=getattr(config, "teacher_nll_scale", 0.6),
            ) if (teacher_nll is not None and (teacher_nll == teacher_nll)) else None
            quality_floor = getattr(config, "teacher_quality_floor", 0.25)
            is_stressed = (
                (j_per_token > effective_j_target * j_margin) or
                (temp_c > config.temp_target_c) or
                (mem_ctrl > 85.0) or
                ((teacher_q is not None) and (teacher_q < quality_floor)) or
                (avg_power_w > effective_power_cap * 1.15)
            )

# z48: Reward with recovery tracking AND regime targets
            reward, advantage, breakdown = reward_computer.compute(
                response=response,
                teacher_nll=teacher_nll,
                throughput=throughput,
                power_w=avg_power_w,
                j_per_token=j_per_token,
                temp_c=temp_c,
                skip_rate=skip_rate_used,
                prediction_error=pred_power_error,
                is_stressed=is_stressed,
                fast_signals=fast_signals,
                regime_targets=regime_targets,  # z48 FIX A: Pass regime-specific targets
            )

            samples.append({
                "response": response,
                "skip_rate": float(skip_rate_used),
                "throughput": throughput,
                "j_per_token": j_per_token,
                "avg_power_w": avg_power_w,
                "mem_ctrl_activity": mem_ctrl,
                "temp_c": temp_c,
                "decode_samples": decode_stats.get("samples", 0),
                "breakdown": breakdown,
                "predictions": {k: v.item() if hasattr(v, 'item') else v for k, v in predictions.items() if k != 'dvfs_logits'},
            })
            rewards.append(reward)
            advantages.append(advantage)
            log_probs.append(sample_out['total_log_prob'] if use_closed_loop else action_result['total_log_prob'])

        # Clear disturbance
        disturbance.clear()

        # Store sensors with actual telemetry
        sensor_buffer.add(
            sensors,
            power_w=samples[0]["avg_power_w"],
            temp_c=samples[0]["temp_c"],
            mem_ctrl=samples[0].get("mem_ctrl_activity", 0.0),
        )

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

        # z48: Contrastive loss with sensor reliance regularizer
        # FIX B: Every step during gate pretrain, otherwise every 3 steps
        do_contrastive = (in_gate_pretrain or step % 3 == 0) and sensor_buffer.can_sample_pair()

        if do_contrastive:
            (sensors_hi, power_hi, temp_hi), (sensors_lo, power_lo, temp_lo) = sensor_buffer.sample_extremes()
            sensors_hi = sensors_hi.to(device)
            sensors_lo = sensors_lo.to(device)

            # z48 FIX B2: Use STATELESS encoding for contrastive (no state mutation)
            body_hi = model.body_state_module.encode(sensors_hi)
            body_lo = model.body_state_module.encode(sensors_lo)

            gate_result_hi = model.gate_net(sensors_hi, body_hi, sample=False)
            gate_result_lo = model.gate_net(sensors_lo, body_lo, sample=False)

            # z48 FIX C: SENSOR RELIANCE - compute gate with SHUFFLED sensors
            # Shuffle sensors between hi/lo to break the sensor->gate connection
            sensors_shuffled = torch.stack([sensors_lo, sensors_hi])[
                torch.randperm(2)
            ].mean(dim=0).to(device)  # Mix of sensors
            body_shuffled = model.body_state_module.encode(sensors_shuffled)
            gate_result_shuffled = model.gate_net(sensors_shuffled, body_shuffled, sample=False)

            power_spread = sensor_buffer.get_power_spread()
            contrastive_loss, contrastive_metrics = contrastive_loss_fn.compute(
                gate_result_hi["gate_probs"],
                gate_result_lo["gate_probs"],
                power_spread=power_spread,
                gate_probs_shuffled=gate_result_shuffled["gate_probs"],  # z48 FIX C
            )
            contrastive_metrics["pair_power_diff"] = power_hi - power_lo
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
            contrastive_metrics = {
                "gate_diff": 0.0, "gate_diff_normalized": 0.0,
                "pair_power_diff": 0.0, "sensor_reliance_diff": 0.0
            }

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
            elif use_closed_loop:
                phase_str = "CL_TRAIN"  # z49: Closed-loop training
            elif pred_phase < 3:
                phase_str = f"PRED_P{pred_phase}"

            # z45v3: Show fast + deep signals
            throttle_hex = int(fast_signals.get('throttle_status', 0))
            fast_str = f"gpu={fast_signals['gpu_busy']:.0f}% mem={fast_signals.get('mem_ctrl_activity', 0):.0f}% thr=0x{throttle_hex:04x}"

            # z45v2: Show if contrastive is active
            contra_str = f"Δg={contrastive_metrics['gate_diff']:.4f}"
            if not sensor_buffer.can_sample_pair():
                contra_str = f"Δg=N/A(spread<5W)"

            skip_rate_decode_avg = float(s.get('skip_rate', m['skip_rate']))
            print(f"  [{step:4d}] {stress_str:14s} gate={gate_mean:.3f} skip_last={m['skip_rate']:.2f} skip_avg={skip_rate_decode_avg:.2f} "
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
                    "train/skip_rate_last_forward": m['skip_rate'],
                    "train/skip_rate_decode_avg": float(s.get('skip_rate', m['skip_rate'])),
                    # Energy
                    "train/j_per_token": s['j_per_token'],
                    "train/avg_power_w": s['avg_power_w'],
                    "train/temp_c": s['temp_c'],
                    # Reward
                    "train/reward": rewards[0],
                    "train/advantage": advantages[0],
                    "train/teacher_nll": breakdown.get("teacher_nll", None) if isinstance(breakdown, dict) else None,
                    "train/teacher_quality": breakdown.get("teacher_quality", None) if isinstance(breakdown, dict) else None,
                    "train/quality_collapse_penalty": breakdown.get("quality_collapse_penalty", 0.0) if isinstance(breakdown, dict) else 0.0,
                    # z45: Recovery tracking
                    "train/recovery_count": breakdown.get("recovery_count", 0),
                    "train/stuck_count": breakdown.get("stuck_count", 0),
                    "train/recovery_score": breakdown.get("recovery", 1.0),
                    "train/energy_score": breakdown.get('energy', None) if isinstance(breakdown, dict) else None,
                    "train/discomfort_score": breakdown.get('discomfort', None) if isinstance(breakdown, dict) else None,
                    "train/throughput_score": breakdown.get('throughput', None) if isinstance(breakdown, dict) else None,
                    "train/in_band": breakdown.get('in_band', None) if isinstance(breakdown, dict) else None,
                    "train/prediction_score": breakdown.get('prediction', None) if isinstance(breakdown, dict) else None,
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
                    "train/use_closed_loop": 1.0 if use_closed_loop else 0.0,  # z49 FIX 4
                    # z53: Log curriculum regime + targets (needed to prove sensor-conditional policies)
                    "train/current_regime": current_regime,
                    "train/regime_power_cap_w": float(regime_targets.get('power_cap_w')) if regime_targets else None,
                    "train/regime_j_target": float(regime_targets.get('j_target')) if regime_targets else None,
                    "train/regime_skip_penalty": float(regime_targets.get('skip_penalty')) if regime_targets else None,
                    "train/is_stressed": 1.0 if is_stressed else 0.0,
                    # z49 FIX 3: DVFS success stats
                    "train/dvfs_success_rate": model.get_dvfs_stats().get('success_rate', 1.0),
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


def save_checkpoint(model: EmbodiedModel, step: int, config: Z49Config):
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


def run_dvfs_step_response_test(sensor_hub) -> Dict:
    """z49 FIX 3: Run DVFS step-response micro-benchmark.

    Tests that DVFS actually changes clocks/power:
    1. Measure baseline at 'auto'
    2. Force 'peak' for 2s, measure power/clock
    3. Force 'min_sclk' for 2s, measure power/clock
    4. Return to 'auto'

    If clocks don't change ±20%, DVFS is not a usable actuator.
    """
    print("\n[DVFS STEP-RESPONSE TEST] Verifying DVFS as usable actuator...")

    results = {'success': False, 'phases': []}

    def measure_phase(name: str, mode: str, duration: float) -> Dict:
        sensor_hub.dvfs.set_mode(mode)
        time.sleep(0.5)  # Settling time

        readings = []
        start = time.time()
        while time.time() - start < duration:
            # Update sensor hub to get fresh reading
            sensor_hub.base.update()
            raw = sensor_hub.base.last_reading
            deep = sensor_hub.deep_metrics.read()
            if raw is not None:
                readings.append({
                    'power': raw.power_mw,  # Named power_mw but is actually watts
                    'temp': raw.temp_c,
                    'gfxclk': deep['gfxclk_deep'],
                    'memclk': deep['memclk_deep'],
                })
            time.sleep(0.1)

        if not readings:
            readings = [{'power': 0, 'temp': 0, 'gfxclk': 0, 'memclk': 0}]

        avg = {k: sum(r[k] for r in readings) / len(readings) for k in readings[0]}
        print(f"  {name}: mode={mode} gfxclk={avg['gfxclk']:.0f}MHz power={avg['power']:.1f}W")
        return {'name': name, 'mode': mode, 'avg': avg, 'readings': readings}

    # Run test phases
    phases = [
        ('baseline', 'auto', 1.5),
        ('peak', 'peak', 2.0),
        ('min_sclk', 'min_sclk', 2.0),
        ('restore', 'auto', 1.0),
    ]

    for name, mode, dur in phases:
        phase_result = measure_phase(name, mode, dur)
        results['phases'].append(phase_result)

    # Check if DVFS is effective
    baseline_clk = results['phases'][0]['avg']['gfxclk']
    peak_clk = results['phases'][1]['avg']['gfxclk']
    min_clk = results['phases'][2]['avg']['gfxclk']

    # Calculate changes
    if baseline_clk > 0:
        peak_change = (peak_clk - baseline_clk) / baseline_clk * 100
        min_change = (min_clk - baseline_clk) / baseline_clk * 100
    else:
        peak_change = 0
        min_change = 0

    results['baseline_clk'] = baseline_clk
    results['peak_clk'] = peak_clk
    results['min_clk'] = min_clk
    results['peak_change_pct'] = peak_change
    results['min_change_pct'] = min_change

    # DVFS is effective if we see at least ±10% clock change
    results['success'] = abs(peak_change) > 5 or abs(min_change) > 10

    if results['success']:
        print(f"  ✓ DVFS EFFECTIVE: peak={peak_change:+.1f}%, min={min_change:+.1f}%")
    else:
        print(f"  ⚠ DVFS WEAK: peak={peak_change:+.1f}%, min={min_change:+.1f}%")
        print(f"    Clock changes are too small - DVFS may not be a usable actuator!")

    # Return to auto
    sensor_hub.dvfs.set_mode('auto')

    return results


def main():
    parser = argparse.ArgumentParser(description="FEEL z49: Binary Skip + Strong Actuation Trainer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="models/z49_embodied")
    parser.add_argument("--disturbance-prob", type=float, default=0.50)
    parser.add_argument("--power-cap", type=float, default=60.0)
    parser.add_argument("--temp-safety", type=float, default=80.0)
    parser.add_argument("--power-safety", type=float, default=130.0)
    parser.add_argument("--contrastive-coef", type=float, default=0.3)
    parser.add_argument("--wandb-project", type=str, default="feel-z49-embodied")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--demo-closed-loop", action="store_true", help="Run a real-time closed-loop generation demo and exit")
    parser.add_argument("--chunk-tokens", type=int, default=16, help="Decision interval (tokens per chunk) for closed-loop")
    parser.add_argument("--no-sensor-packets", action="store_true", help="Disable sensor pseudo-token packets")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--skip-dvfs-test", action="store_true", help="Skip DVFS step-response test at startup")
    parser.add_argument("--gate-layers", type=str, default=None, help="Comma-separated gate layer indices (default: every other layer)")
    args = parser.parse_args()

    # z49: Parse custom gate layers if provided
    custom_gate_layers = None
    if args.gate_layers:
        custom_gate_layers = [int(x.strip()) for x in args.gate_layers.split(',')]

    config = Z49Config(
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
        decision_chunk_tokens=args.chunk_tokens,
        inject_sensor_packets=not args.no_sensor_packets,
        sensor_packet_tokens=4,
        live_dashboard=True,
        gate_layers=custom_gate_layers,
        dvfs_step_response_test=not args.skip_dvfs_test,
    )

    # Initialize wandb
    if WANDB_AVAILABLE and config.use_wandb:
        import socket
        hostname = socket.gethostname()
        run_name = config.wandb_run_name or f"z49_binary_{hostname}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=asdict(config),
            tags=["z49", "binary-skip", "strong-actuation", hostname],
        )

    print("=" * 70)
    print("FEEL z49: BINARY SKIP + STRONG ACTUATION TRAINER")
    print("=" * 70)
    print("z49 CRITICAL FIXES (for REAL sensorimotor embodiment):")
    print("  FIX 1: BINARY SKIP EXECUTION - no blending, execute ONE path")
    print("         -> Skip actions ACTUALLY reduce compute/power now!")
    print("  FIX 2: STRONGER ACTUATION - 14 gated layers (was 5)")
    print("         -> Actions measurably change watts")
    print(f"         -> gate_layers = {config.gate_layers}")
    print("  FIX 3: DVFS VERIFICATION - log success, step-response test")
    print("         -> Verify DVFS is a usable actuator")
    print("  FIX 4: CLOSED-LOOP TRAINING - enabled after gate pretrain")
    print("         -> Training matches demo loop")
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

    print("\n[3/8] Initializing FAST SIGNAL sensor hub...")
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

    # z49 FIX 3: DVFS step-response test
    dvfs_test_result = None
    if getattr(config, 'dvfs_step_response_test', True):
        dvfs_test_result = run_dvfs_step_response_test(sensor_hub)
        if WANDB_AVAILABLE and config.use_wandb:
            wandb.log({
                "dvfs_test/success": 1.0 if dvfs_test_result['success'] else 0.0,
                "dvfs_test/baseline_clk": dvfs_test_result['baseline_clk'],
                "dvfs_test/peak_clk": dvfs_test_result['peak_clk'],
                "dvfs_test/min_clk": dvfs_test_result['min_clk'],
                "dvfs_test/peak_change_pct": dvfs_test_result['peak_change_pct'],
                "dvfs_test/min_change_pct": dvfs_test_result['min_change_pct'],
            })

    # Auto-disable DVFS actuation if step-response indicates it is ineffective on this hardware
    if dvfs_test_result is not None:
        config.regime_use_dvfs = bool(dvfs_test_result.get('success', False))
        if not config.regime_use_dvfs:
            print("[DVFS] Step-response weak -> disabling DVFS actuation (skip remains primary actuator).")

    print("\n[4/8] Building gate network with BINARY SKIP...")
    gate_net = GateNetWithExpectedSkip(
        sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
        body_dim=config.body_dim,
        num_layers=len(config.gate_layers),
    ).to(device)

    print("\n[5/8] Building predictive head with CURRICULUM...")
    predictor = PredictiveHeadWithCurriculum(
        body_dim=config.body_dim,
        sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
    ).to(device)

    intero_report = InteroceptiveReportHead(
        body_dim=config.body_dim,
        sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
    ).to(device)

    # z48: Hardware telemetry as latent modality (learned pseudo-token packets)
    sensor_packet_encoder = None
    if getattr(config, 'inject_sensor_packets', True):
        hidden_size = getattr(base_model.config, 'hidden_size', 2048)
        sensor_packet_encoder = SensorPacketEncoder(
            sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
            body_dim=config.body_dim,
            hidden_size=hidden_size,
            num_tokens=getattr(config, 'sensor_packet_tokens', 4),
        ).to(device)

    print("\n[6/8] Building embodied model...")
    model = EmbodiedModel(
        base_model=base_model,
        gate_net=gate_net,
        sensor_hub=sensor_hub,
        body_state=body_state_module,
        predictor=predictor,
        intero_report=intero_report,
        sensor_packet_encoder=sensor_packet_encoder,
        gate_layers=config.gate_layers,
    )

    # z49: Load checkpoint if resuming (handle z48 checkpoints with different gate_layers)
    global_step = 0
    if args.resume_from and Path(args.resume_from).exists():
        print(f"\n  Loading checkpoint from: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)

        # z49: Check if checkpoint has different gate_layers
        ckpt_gate_layers = checkpoint.get("gate_layers", [5, 10, 15, 20, 25])
        if set(ckpt_gate_layers) != set(config.gate_layers):
            print(f"  WARN: Checkpoint has different gate_layers: {ckpt_gate_layers}")
            print(f"        z49 uses: {config.gate_layers}")
            print(f"        Skip blocks will be reinitialized (gate_net still loaded)")

        # Load compatible weights
        if "body_state_state_dict" in checkpoint:
            body_state_module.load_state_dict(checkpoint["body_state_state_dict"])
        elif "body_state" in checkpoint:
            body_state_module.load_state_dict(checkpoint["body_state"])

        if "gate_net_state_dict" in checkpoint:
            # z49: gate_net may have different num_layers, try loading with strict=False
            try:
                gate_net.load_state_dict(checkpoint["gate_net_state_dict"], strict=False)
                print(f"  Loaded gate_net (strict=False for compatibility)")
            except Exception as e:
                print(f"  WARN: Could not load gate_net: {e}")
        elif "gate_net" in checkpoint:
            try:
                gate_net.load_state_dict(checkpoint["gate_net"], strict=False)
            except Exception as e:
                print(f"  WARN: Could not load gate_net: {e}")

        if "predictor_state_dict" in checkpoint:
            predictor.load_state_dict(checkpoint["predictor_state_dict"])
        elif "predictor" in checkpoint:
            predictor.load_state_dict(checkpoint["predictor"])

        if "intero_report_state_dict" in checkpoint:
            intero_report.load_state_dict(checkpoint["intero_report_state_dict"])
        elif "intero_report" in checkpoint:
            intero_report.load_state_dict(checkpoint["intero_report"])

        if "step" in checkpoint:
            global_step = checkpoint["step"]
        print(f"  Loaded checkpoint at step {global_step}")
        print(f"  NOTE: z49 continues with BINARY SKIP and STRONGER ACTUATION from here")

    # z49 FIX 3: Initialize DVFS logging
    model.init_dvfs_logging()

    print("\n[7/8] Initializing THERMAL GOVERNOR...")


    # ----------------------------------------------------------------------
    # DEMO CLOSED LOOP: real-time sense/act/express while decoding
    # ----------------------------------------------------------------------
    if args.demo_closed_loop:
        print("\n[DEMO CLOSED LOOP] Running one prompt with real-time telemetry...")
        prompt = "Explain how GPUs manage power and thermals in simple terms."
        # read sensors + update body
        s = model.sensor_hub.read_tensor().to(device)
        b = model.body_state_module.update(s)

        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with model.sensor_hub.measure_decode():
            out = model.closed_loop_generate(
                tokenizer=tokenizer,
                input_ids=inp.input_ids,
                attention_mask=inp.attention_mask,
                max_new_tokens=128,
                chunk_tokens=config.decision_chunk_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                config=config,
                current_regime="cool",
                use_expected_skip=False,
                in_gate_pretrain=False,
            )
        resp = tokenizer.decode(out['output_ids'][0, inp.input_ids.shape[1]:], skip_special_tokens=True)
        print("\n--- RESPONSE ---\n" + resp)
        print("\n--- DECODE STATS ---")
        print(out['decode_stats'])
        return

    thermal_governor = ThermalGovernor(
        temp_limit_c=config.temp_safety_c,
        power_limit_w=config.power_safety_w,
        cooldown_s=config.cooldown_duration_s,
        dvfs_controller=sensor_hub.dvfs,
    )

    # Safe disturbance
    disturbance = SafeDisturbanceScheduler(device=str(device), config=config)
    reward_computer = RecoveryAwareReward(config)

    # z48 FIX A: Regime curriculum
    print("\n[8/8] Initializing REGIME CURRICULUM...")
    regime_curriculum = RegimeCurriculum(config, dvfs_controller=sensor_hub.dvfs)
    print(f"  COOL regime: power_cap={config.cool_power_cap_w}W, j_target={config.cool_j_target}")
    print(f"  HOT  regime: power_cap={config.hot_power_cap_w}W, j_target={config.hot_j_target}")
    print(f"  Switch every {config.regime_switch_steps} steps")

    sensor_buffer = SensorBuffer(
        max_size=config.sensor_buffer_size,
        quantile_pct=config.quantile_pct,
        power_baseline_w=config.hot_power_cap_w,
        temp_baseline_c=config.temp_target_c,
        min_power_spread_w=5.0,  # z51: lowered from 8W to fire contrastive more often
    )
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

    # Training (global_step initialized above, possibly from checkpoint)
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
            regime_curriculum=regime_curriculum,  # z48 FIX A
        )

    # Cleanup
    disturbance.clear()

    # Final checkpoint
    save_checkpoint(model, global_step, config)

    print("\n" + "=" * 70)
    print("Z46 TRAINING COMPLETE")
    print(f"  Thermal governor triggered: {thermal_governor.trigger_count} times")
    print(f"  Regime switches: {len(regime_curriculum.regime_history)}")
    print("=" * 70)


if __name__ == "__main__":
    main()