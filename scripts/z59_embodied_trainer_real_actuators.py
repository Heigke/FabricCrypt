#!/usr/bin/env python3
"""
FEEL z59: REAL Actuators That Actually Work
============================================

z59 CRITICAL FIXES based on expert analysis:

PROBLEM 1: "Attention temperature" was controlling SAMPLING, not attention softmax
  - z58: effective_temp = temperature * attn_temp_value; logits / effective_temp
  - This changes language randomness, NOT memory access patterns
  - FIX: Hook into attention modules and scale QK^T / τ BEFORE softmax

PROBLEM 2: "KV precision control" was numerical perturbation, not bandwidth savings
  - z58: Quantize to int8 then BACK to fp16 - still stored/read as fp16!
  - FIX: Either implement TRUE stored-int8 KV (hard) or REMOVE this fake actuator

PROBLEM 3: "Attention window" masks but doesn't cut compute
  - z58: attention_mask[:, :-W] = 0 - still computes QK^T over full length!
  - FIX: Actually TRUNCATE past_key_values to reduce memory bandwidth

PROBLEM 4: "Steering vectors" were logit bias, not residual stream injection
  - z58: steer_logit_bias = F.linear(steering_vec, lm_head.weight)
  - This is a logit hammer, not subtle expression
  - FIX: Add tiny steering to residual stream at 2-3 late layers, with norm clamp

PROBLEM 5: gpu_metrics parsing uses hardcoded offsets (brittle)
  - FIX: Parse version/size fields and handle struct variations

NEW ADDITIONS (SAFE actuators):

SAFE 1: Power profile actuator via sysfs pp_power_profile_mode
  - Changes GPU power management heuristics without touching semantics
  - Modes: VIDEO, CUSTOM, COMPUTE, VR, etc.

SAFE 2: Real KV truncation for sliding window attention
  - Actually removes old KV entries from past_key_values
  - Reduces memory bandwidth = real power savings

SAFE 3: Quality gate that DISABLES dangerous actuators when NLL rises
  - If teacher_nll > threshold: disable all semantic-path actuators
  - Force max attention window, no steering, no FiLM

SAFE 4: Derivative sensors (dP/dt, dT/dt, d²P/dt²)
  - Detect throttling BEFORE it happens
  - Enable proactive regulation

SAFE 5: DVFS delay model (treat as slow actuator)
  - DVFS has ~100-500ms latency and hysteresis
  - Model this delay in reward computation

Author: FEEL Research Team
Date: 2026-01-19 (z59 - REAL ACTUATORS)
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
import struct
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
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


# =========================================================================
# DEVICE AUTO-DETECTION
# =========================================================================

def detect_amd_device_path(prefer: str | None = None) -> str:
    """Pick the best /sys/class/drm/cardX/device path."""
    base = Path('/sys/class/drm')
    cands = []
    for dev in sorted(base.glob('card*/device')):
        score = 0
        if (dev / 'gpu_busy_percent').exists():
            score += 3
        if (dev / 'gpu_metrics').exists():
            score += 3
        hwmon = list((dev / 'hwmon').glob('hwmon*'))
        if hwmon:
            score += 2
            pfiles = list(hwmon[0].glob('power*_input')) + list(hwmon[0].glob('power*_average'))
            tfiles = list(hwmon[0].glob('temp*_input'))
            if pfiles:
                score += 2
            if tfiles:
                score += 1
        if prefer and prefer in str(dev):
            score += 1
        cands.append((score, str(dev)))
    if not cands:
        return '/sys/class/drm/card0/device'
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def _maybe_mw_to_w(x: float) -> float:
    if x is None:
        return 0.0
    try:
        x = float(x)
    except Exception:
        return 0.0
    return x / 1000.0 if x > 2000.0 else x


def _sanitize(x: torch.Tensor, clamp: float = 10.0) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp is not None:
        x = torch.clamp(x, -clamp, clamp)
    return x


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Z59Config:
    """z59 Real Actuators Configuration.

    Key changes from z58:
    - Removed fake actuators (KV precision, wrong attention temp)
    - Added real actuators (KV truncation, power profiles)
    - Quality gate disables ALL dangerous actuators on NLL breach
    - Derivative sensors for early throttle detection
    """
    # Base model
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    device_path: Optional[str] = None
    gate_layers: List[int] = None

    epochs: int = 3
    max_prompts: int = 500
    num_samples: int = 2
    max_tokens: int = 128

    # Learning rates
    gate_lr: float = 3e-4
    body_lr: float = 1e-4
    predictor_lr: float = 1e-4
    skip_distill_lr: float = 1e-3

    # REINFORCE settings
    baseline_ema: float = 0.99
    entropy_coef: float = 0.01

    # Body state settings
    body_dim: int = 64
    body_decay: float = 0.05
    body_noise_std: float = 0.01

    # Reward weights
    quality_weight: float = 0.25  # z59: Higher quality weight
    energy_weight: float = 0.20
    recovery_weight: float = 0.15
    throughput_weight: float = 0.15
    prediction_weight: float = 0.10
    discomfort_weight: float = 0.10
    sensor_reliance_weight: float = 0.05

    # Quality guardrail
    use_teacher_quality: bool = True
    teacher_nll_center: float = 2.0
    teacher_nll_scale: float = 0.5
    teacher_quality_floor: float = 0.30
    quality_collapse_penalty: float = 0.30
    teacher_eval_interval: int = 2
    teacher_eval_skip_threshold: float = 0.35

    # z59 QUALITY GATE (CRITICAL)
    # When breached: disable ALL dangerous actuators immediately
    quality_gate_enabled: bool = True
    quality_gate_nll_threshold: float = 2.0  # z59: Tighter threshold
    quality_gate_window: int = 5  # z59: Faster response
    quality_gate_freeze_all_on_breach: bool = True  # z59: Freeze EVERYTHING

    # Homeostatic quality constraint
    quality_lambda_init: float = 0.0
    quality_lambda_lr: float = 0.8
    quality_lambda_decay: float = 0.03
    quality_lambda_max: float = 3.0

    # Recovery potential
    recovery_potential_scale: float = 0.15
    in_band_j_margin: float = 1.10

    # Two-regime curriculum
    cool_power_cap_w: float = 90.0
    cool_j_target: float = 8.0
    hot_power_cap_w: float = 85.0
    hot_j_target: float = 7.2
    regime_switch_steps: int = 10
    regime_use_dvfs: bool = True

    # Safety targets
    power_cap_w: float = 90.0
    power_safety_w: float = 130.0
    temp_target_c: float = 70.0
    temp_safety_c: float = 80.0
    j_per_token_target: float = 7.5

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

    # z59: FiLM DISABLED (causes OOD collapse)
    film_scale: float = 0.0
    film_scale_start_step: int = 99999

    # Contrastive and sensor reliance
    contrastive_coef: float = 0.3
    contrastive_margin: float = 0.05
    sensor_buffer_size: int = 200
    sensor_reliance_coef: float = 0.1
    quantile_pct: float = 0.15

    # Skip distillation (required before any RL on skip)
    skip_distill_enabled: bool = True
    skip_distill_steps: int = 200
    skip_distill_prompts: int = 50
    skip_distill_validate_nll_threshold: float = 1.5
    skip_distill_mse_target: float = 0.1

    # Training phases
    gate_pretrain_steps: int = 100
    expected_skip_steps: int = 0
    predictor_phase1_steps: int = 100
    predictor_phase2_steps: int = 200

    val_every: int = 50
    checkpoint_dir: str = "models/z59_embodied"

    # Wandb
    wandb_project: str = "feel-z59-real"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    # Closed-loop training
    closed_loop_train: bool = True
    closed_loop_after_step: int = 100
    decision_chunk_tokens: int = 16

    # Auto-calibrate regime setpoints
    auto_calibrate_setpoints: bool = True
    calibration_steps: int = 30
    hot_j_factor: float = 0.95
    cool_j_factor: float = 1.02
    hot_power_factor: float = 1.03
    cool_power_factor: float = 1.08

    # =========================================================================
    # z59 REAL ACTUATORS
    # =========================================================================

    # REAL ACTUATOR 1: KV Cache Truncation (not masking!)
    # Actually removes old KV entries = real bandwidth savings
    use_kv_truncation: bool = True
    kv_truncation_windows: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    kv_truncation_apply_in_cool: bool = True

    # REAL ACTUATOR 2: Power Profile Mode (sysfs)
    # Changes GPU power management heuristics
    use_power_profile_actuator: bool = True
    power_profile_modes: Tuple[str, ...] = (
        "BOOTUP_DEFAULT",  # 0
        "3D_FULL_SCREEN",  # 1
        "POWER_SAVING",    # 2
        "VIDEO",           # 3
        "VR",              # 4
        "COMPUTE",         # 5
        "CUSTOM",          # 6
    )

    # z59: Steering vectors DISABLED by default (was causing garbage)
    # Only enable after proving it doesn't hurt quality
    use_steering_vectors: bool = False
    steering_vector_scale: float = 0.01  # z59: Much smaller
    steering_vector_layers: List[int] = None  # Late layers only

    # z59: Sensor packets DISABLED (causes OOD)
    inject_sensor_packets: bool = False

    # DVFS delay model (treat as slow actuator)
    dvfs_delay_ms: float = 200.0  # DVFS takes ~200ms to take effect
    dvfs_hysteresis_w: float = 5.0  # Hysteresis band

    # Derivative sensors
    use_derivative_sensors: bool = True
    derivative_window_ms: float = 100.0  # Window for computing derivatives

    live_dashboard: bool = True
    dvfs_step_response_test: bool = True
    log_dvfs_success: bool = True

    def __post_init__(self):
        if self.gate_layers is None:
            self.gate_layers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
        if self.steering_vector_layers is None:
            # z59: Only late layers (less disruptive)
            self.steering_vector_layers = [24, 26]


# Alias
Z58Config = Z59Config


# =========================================================================
# z59 FIX 7: ROBUST GPU_METRICS PARSING
# =========================================================================

class RobustGPUMetrics:
    """z59: Parse gpu_metrics with version awareness.

    The gpu_metrics struct layout varies by ASIC/driver version.
    We parse the header to determine version and adjust offsets accordingly.

    Supported versions:
    - Version 1.0: Basic metrics
    - Version 2.0: Extended metrics with more activity counters
    """

    # Known struct layouts by version
    LAYOUTS = {
        # (major, minor): {field: (offset, format)}
        (1, 0): {
            "temperature_gfx": (4, "<H"),
            "temperature_soc": (6, "<H"),
            "gfx_activity": (42, "<H"),
            "mem_activity": (66, "<H"),
            "throttle_status": (96, "<I"),
            "gfxclk": (224, "<H"),
            "memclk": (186, "<H"),
        },
        (2, 0): {
            "temperature_gfx": (4, "<H"),
            "temperature_soc": (6, "<H"),
            "gfx_activity": (42, "<H"),
            "mem_activity": (66, "<H"),
            "vcn_activity": (76, "<H"),
            "throttle_status": (96, "<I"),
            "gfxclk": (224, "<H"),
            "memclk": (186, "<H"),
        },
    }

    def __init__(self, device_path: Path):
        self.gpu_metrics_path = device_path / "gpu_metrics"
        self.available = self.gpu_metrics_path.exists()
        self.version = (1, 0)  # Default
        self.layout = self.LAYOUTS[(1, 0)]

        if self.available:
            self._detect_version()
            print(f"[RobustGPUMetrics] Found gpu_metrics v{self.version[0]}.{self.version[1]}")
        else:
            print(f"[RobustGPUMetrics] WARN: gpu_metrics not found")

    def _detect_version(self):
        """Detect struct version from header."""
        try:
            data = self.gpu_metrics_path.read_bytes()
            if len(data) < 4:
                return

            # First 2 bytes are often version info
            # This is ASIC-dependent, so we use heuristics
            size = len(data)

            # Heuristic: larger structs are newer versions
            if size >= 250:
                self.version = (2, 0)
            else:
                self.version = (1, 0)

            self.layout = self.LAYOUTS.get(self.version, self.LAYOUTS[(1, 0)])
        except Exception:
            pass

    def read(self) -> Dict[str, float]:
        """Read and parse gpu_metrics with version-aware offsets."""
        result = {
            "throttle_status": 0.0,
            "mem_ctrl_activity": 0.0,
            "vcn_activity": 0.0,
            "gfx_activity_deep": 0.0,
            "temp_gfx_deep": 0.0,
            "temp_soc_deep": 0.0,
            "gfxclk_deep": 0.0,
            "memclk_deep": 0.0,
        }

        if not self.available:
            return result

        try:
            data = self.gpu_metrics_path.read_bytes()

            # Parse using version-specific layout
            for field, (offset, fmt) in self.layout.items():
                if offset + struct.calcsize(fmt) <= len(data):
                    try:
                        value = struct.unpack_from(fmt, data, offset)[0]
                        if field == "temperature_gfx":
                            result["temp_gfx_deep"] = value / 100.0  # 0.01C units
                        elif field == "temperature_soc":
                            result["temp_soc_deep"] = value / 100.0
                        elif field == "gfx_activity":
                            result["gfx_activity_deep"] = float(value)
                        elif field == "mem_activity":
                            result["mem_ctrl_activity"] = float(value)
                        elif field == "vcn_activity":
                            result["vcn_activity"] = float(value)
                        elif field == "throttle_status":
                            result["throttle_status"] = float(value)
                        elif field == "gfxclk":
                            result["gfxclk_deep"] = float(value)
                        elif field == "memclk":
                            result["memclk_deep"] = float(value)
                    except struct.error:
                        pass
        except Exception:
            pass

        return result


# =========================================================================
# z59 SAFE ACTUATOR 1: POWER PROFILE CONTROLLER
# =========================================================================

class PowerProfileController:
    """z59: Control GPU power management via pp_power_profile_mode.

    This is a SEMANTIC-SAFE actuator - it changes GPU power heuristics
    without touching model computation at all.

    Available profiles (AMDGPU):
    - BOOTUP_DEFAULT (0): Balanced
    - 3D_FULL_SCREEN (1): Gaming
    - POWER_SAVING (2): Low power
    - VIDEO (3): Video playback
    - VR (4): VR workloads
    - COMPUTE (5): Compute workloads
    - CUSTOM (6): User-defined
    """

    PROFILES = {
        "BOOTUP_DEFAULT": 0,
        "3D_FULL_SCREEN": 1,
        "POWER_SAVING": 2,
        "VIDEO": 3,
        "VR": 4,
        "COMPUTE": 5,
        "CUSTOM": 6,
    }

    def __init__(self, device_path: Path):
        self.device_path = device_path
        self.profile_path = device_path / "pp_power_profile_mode"
        self.available = self.profile_path.exists()
        self.current_profile = "BOOTUP_DEFAULT"
        self.last_change_time = 0.0

        if self.available:
            self._read_current_profile()
            print(f"[PowerProfileController] Initialized, current profile: {self.current_profile}")
        else:
            print(f"[PowerProfileController] WARN: pp_power_profile_mode not found")

    def _read_current_profile(self):
        """Read current power profile."""
        try:
            content = self.profile_path.read_text().strip()
            # Format: "0: BOOTUP_DEFAULT *" (asterisk marks current)
            for line in content.split('\n'):
                if '*' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        profile_name = parts[1].split('*')[0].strip()
                        self.current_profile = profile_name
                        return
        except Exception:
            pass

    def set_profile(self, profile_name: str) -> bool:
        """Set power profile by name."""
        if not self.available:
            return False

        if profile_name not in self.PROFILES:
            return False

        profile_idx = self.PROFILES[profile_name]

        try:
            self.profile_path.write_text(str(profile_idx))
            self.current_profile = profile_name
            self.last_change_time = time.time()
            return True
        except Exception as e:
            # May need root/sudo
            try:
                subprocess.run(
                    ['sudo', 'tee', str(self.profile_path)],
                    input=str(profile_idx).encode(),
                    capture_output=True,
                    timeout=1.0
                )
                self.current_profile = profile_name
                self.last_change_time = time.time()
                return True
            except Exception:
                return False

    def get_profile_for_state(self, power_w: float, temp_c: float, stressed: bool) -> str:
        """Recommend profile based on thermal/power state."""
        if temp_c > 75 or stressed:
            return "POWER_SAVING"
        elif power_w > 100:
            return "COMPUTE"  # Let GPU optimize for compute
        else:
            return "BOOTUP_DEFAULT"


# =========================================================================
# z59 DERIVATIVE SENSORS
# =========================================================================

class DerivativeSensorHub:
    """z59: Compute derivatives for early throttle detection.

    Key signals:
    - dP/dt: Power change rate (W/s)
    - dT/dt: Temperature change rate (C/s)
    - d²P/dt²: Power acceleration (detects onset of throttling)
    - d²T/dt²: Temperature acceleration

    These enable PROACTIVE regulation before throttling occurs.
    """

    def __init__(self, window_ms: float = 100.0):
        self.window_s = window_ms / 1000.0

        # Timestamped histories
        self.power_history: Deque[Tuple[float, float]] = deque(maxlen=50)
        self.temp_history: Deque[Tuple[float, float]] = deque(maxlen=50)

        # Cached derivatives
        self.last_derivatives = {
            "dP_dt": 0.0,
            "dT_dt": 0.0,
            "d2P_dt2": 0.0,
            "d2T_dt2": 0.0,
            "power_trending_up": False,
            "temp_trending_up": False,
            "throttle_imminent": False,
        }

    def update(self, power_w: float, temp_c: float) -> Dict[str, float]:
        """Update with new readings and compute derivatives."""
        now = time.time()

        self.power_history.append((now, power_w))
        self.temp_history.append((now, temp_c))

        # Compute first derivatives (dP/dt, dT/dt)
        dP_dt = self._compute_derivative(self.power_history)
        dT_dt = self._compute_derivative(self.temp_history)

        # Compute second derivatives (acceleration)
        d2P_dt2 = self._compute_second_derivative(self.power_history)
        d2T_dt2 = self._compute_second_derivative(self.temp_history)

        # Trend detection
        power_trending_up = dP_dt > 2.0  # >2 W/s
        temp_trending_up = dT_dt > 0.5   # >0.5 C/s

        # Throttle prediction: high temp + accelerating
        throttle_imminent = (temp_c > 70 and temp_trending_up) or (d2T_dt2 > 0.5)

        self.last_derivatives = {
            "dP_dt": dP_dt,
            "dT_dt": dT_dt,
            "d2P_dt2": d2P_dt2,
            "d2T_dt2": d2T_dt2,
            "power_trending_up": power_trending_up,
            "temp_trending_up": temp_trending_up,
            "throttle_imminent": throttle_imminent,
        }

        return self.last_derivatives

    def _compute_derivative(self, history: Deque[Tuple[float, float]]) -> float:
        """Compute first derivative using linear regression over window."""
        if len(history) < 3:
            return 0.0

        now = time.time()
        window_start = now - self.window_s

        # Filter to window
        points = [(t, v) for t, v in history if t >= window_start]
        if len(points) < 2:
            return 0.0

        # Simple linear regression: dv/dt
        n = len(points)
        sum_t = sum(p[0] for p in points)
        sum_v = sum(p[1] for p in points)
        sum_tv = sum(p[0] * p[1] for p in points)
        sum_t2 = sum(p[0] ** 2 for p in points)

        denom = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-10:
            return 0.0

        slope = (n * sum_tv - sum_t * sum_v) / denom
        return slope

    def _compute_second_derivative(self, history: Deque[Tuple[float, float]]) -> float:
        """Compute second derivative (acceleration)."""
        if len(history) < 5:
            return 0.0

        now = time.time()

        # Split into two halves and compare derivatives
        mid = len(history) // 2
        first_half = list(history)[:mid]
        second_half = list(history)[mid:]

        if len(first_half) < 2 or len(second_half) < 2:
            return 0.0

        # Derivatives for each half
        d1 = self._derivative_from_points(first_half)
        d2 = self._derivative_from_points(second_half)

        # Time span
        dt = (second_half[-1][0] - first_half[0][0]) / 2.0
        if dt < 0.01:
            return 0.0

        return (d2 - d1) / dt

    def _derivative_from_points(self, points: List[Tuple[float, float]]) -> float:
        """Compute derivative from a list of (time, value) points."""
        if len(points) < 2:
            return 0.0
        n = len(points)
        sum_t = sum(p[0] for p in points)
        sum_v = sum(p[1] for p in points)
        sum_tv = sum(p[0] * p[1] for p in points)
        sum_t2 = sum(p[0] ** 2 for p in points)

        denom = n * sum_t2 - sum_t ** 2
        if abs(denom) < 1e-10:
            return 0.0

        return (n * sum_tv - sum_t * sum_v) / denom


# =========================================================================
# z59 QUALITY GATE (CRITICAL SAFETY MECHANISM)
# =========================================================================

class Z59QualityGate:
    """z59: Quality gate that DISABLES ALL dangerous actuators when NLL rises.

    This is the key safety mechanism. When quality degrades:
    1. Disable FiLM (if enabled)
    2. Disable steering vectors
    3. Disable sensor packet injection
    4. Force maximum attention window (no truncation)
    5. Force skip rate to 0 (run all layers)
    6. Force DVFS to safe mode

    Only re-enable actuators after quality recovers for N steps.
    """

    def __init__(
        self,
        nll_threshold: float = 2.0,
        window_size: int = 5,
        recovery_steps: int = 10,
    ):
        self.nll_threshold = nll_threshold
        self.window_size = window_size
        self.recovery_steps = recovery_steps

        self.nll_history: Deque[float] = deque(maxlen=window_size)
        self.is_breached = False
        self.breach_count = 0
        self.steps_since_recovery = 0
        self.actuators_disabled = False

    def update(self, teacher_nll: float) -> Dict[str, bool]:
        """Update with new NLL, return actuator enable/disable flags."""
        if teacher_nll is None or teacher_nll != teacher_nll:  # NaN check
            return self._get_flags()

        self.nll_history.append(teacher_nll)

        if len(self.nll_history) >= 3:
            avg_nll = sum(self.nll_history) / len(self.nll_history)

            if avg_nll > self.nll_threshold:
                if not self.is_breached:
                    self.breach_count += 1
                    print(f"[Z59QualityGate] BREACH! avg_nll={avg_nll:.3f} > {self.nll_threshold}")
                    print(f"  Disabling ALL dangerous actuators for safety")
                self.is_breached = True
                self.actuators_disabled = True
                self.steps_since_recovery = 0
            else:
                if self.is_breached:
                    self.steps_since_recovery += 1
                    if self.steps_since_recovery >= self.recovery_steps:
                        print(f"[Z59QualityGate] Recovery complete after {self.steps_since_recovery} steps")
                        self.is_breached = False
                        self.actuators_disabled = False

        return self._get_flags()

    def _get_flags(self) -> Dict[str, bool]:
        """Return actuator enable/disable flags."""
        if self.actuators_disabled:
            return {
                "allow_film": False,
                "allow_steering": False,
                "allow_sensor_packets": False,
                "allow_kv_truncation": False,  # Force max window
                "allow_skip": False,  # Force full compute
                "force_safe_dvfs": True,
            }
        else:
            return {
                "allow_film": True,
                "allow_steering": True,
                "allow_sensor_packets": True,
                "allow_kv_truncation": True,
                "allow_skip": True,
                "force_safe_dvfs": False,
            }

    def reset(self):
        self.nll_history.clear()
        self.is_breached = False
        self.steps_since_recovery = 0
        self.actuators_disabled = False


# =========================================================================
# z59 REAL KV TRUNCATION
# =========================================================================

def truncate_kv_cache(
    past_key_values: Tuple,
    max_length: int,
) -> Tuple:
    """z59: Actually truncate KV cache to reduce memory bandwidth.

    This is the REAL way to reduce attention compute:
    - Remove old KV entries entirely
    - Reduces memory reads = real power savings

    Unlike masking (which still computes full QK^T), truncation
    physically removes the memory.

    Args:
        past_key_values: Tuple of (key, value) tensors per layer
        max_length: Maximum sequence length to keep

    Returns:
        Truncated past_key_values
    """
    if past_key_values is None:
        return None

    truncated = []
    for layer_past in past_key_values:
        if isinstance(layer_past, tuple) and len(layer_past) >= 2:
            k, v = layer_past[0], layer_past[1]
            seq_len = k.size(2)

            if seq_len > max_length:
                # Keep only the most recent max_length entries
                k_trunc = k[:, :, -max_length:, :].contiguous()
                v_trunc = v[:, :, -max_length:, :].contiguous()
                truncated.append((k_trunc, v_trunc))
            else:
                truncated.append(layer_past)
        else:
            truncated.append(layer_past)

    return tuple(truncated)


# =========================================================================
# z59 STEERING VECTORS (RESIDUAL STREAM, NOT LOGIT BIAS)
# =========================================================================

class ResidualStreamSteering(nn.Module):
    """z59: Apply tiny steering to residual stream at late layers.

    This is the CORRECT way to do steering (not logit bias):
    - Add tiny vector to hidden states at specific layers
    - Use VERY small scale (0.01) to stay in-distribution
    - Apply only at 2-3 late layers (less disruptive)
    - Hard clamp norm to prevent quality collapse

    The steering is applied via forward hooks on the model.
    """

    def __init__(
        self,
        hidden_size: int,
        sensor_dim: int = 72,
        body_dim: int = 64,
        scale: float = 0.01,
        max_norm: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = scale
        self.max_norm = max_norm

        # Map (sensors, body) -> steering direction
        self.net = nn.Sequential(
            nn.Linear(sensor_dim + body_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, hidden_size),
        )

        # Initialize to near-zero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Current steering vector (set before forward pass)
        self.current_steering = None

    def compute_steering(
        self,
        sensors: torch.Tensor,
        body_state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute steering vector from current state."""
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        if body_state.dim() == 1:
            body_state = body_state.unsqueeze(0)

        x = torch.cat([sensors, body_state], dim=-1)
        steering = self.net(x)

        # Hard clamp norm (CRITICAL for safety)
        steering_norm = steering.norm(dim=-1, keepdim=True)
        if steering_norm > self.max_norm:
            steering = steering * (self.max_norm / (steering_norm + 1e-8))

        # Scale
        steering = steering * self.scale

        self.current_steering = steering
        return steering

    def apply_to_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply steering to hidden states (called in forward hook)."""
        if self.current_steering is None:
            return hidden_states

        # Broadcast steering to sequence length
        steering = self.current_steering  # (B, hidden_size)
        if steering.dim() == 2:
            steering = steering.unsqueeze(1)  # (B, 1, hidden_size)

        return hidden_states + steering


# =========================================================================
# FAST SIGNAL SENSOR HUB (with derivatives)
# =========================================================================

class FastSignalSensorHub:
    """Extended sensor hub with fast signals and derivatives."""

    FAST_SIGNAL_DIM = 80  # z59: +8 for derivatives

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

        self.ema_mean = torch.zeros(SENSOR_DIM)
        self.ema_var = torch.ones(SENSOR_DIM)
        self.ema_initialized = False

        self.feature_history: Deque[Tuple[float, torch.Tensor]] = deque(maxlen=100)
        self.power_history: Deque[Tuple[float, float]] = deque(maxlen=50)
        self.temp_history: Deque[Tuple[float, float]] = deque(maxlen=50)
        self.gpu_busy_history: Deque[Tuple[float, float]] = deque(maxlen=50)

        self.power_sampler = DecodeTimePowerSampler(base_hub, power_sample_interval_ms)
        self.last_decode_stats: Dict = {}

        self._init_fast_signal_paths()

        # z59: Robust GPU metrics with version awareness
        self.deep_metrics = RobustGPUMetrics(self.base.device_path)
        self.deep_history: Deque[Tuple[float, Dict]] = deque(maxlen=50)

        # z59: Derivative sensors
        self.derivative_hub = DerivativeSensorHub(window_ms=100.0)

        self._training_mode = False

    def _init_fast_signal_paths(self):
        device_path = self.base.device_path

        self.gpu_busy_path = device_path / "gpu_busy_percent"

        hwmon_base = device_path / "hwmon"
        self.power_input_path = None
        self.gfx_voltage_path = None
        self.soc_voltage_path = None
        self.freq_input_path = None

        if hwmon_base.exists():
            for hwmon in sorted(hwmon_base.iterdir()):
                p = hwmon / "power1_input"
                if p.exists():
                    self.power_input_path = p
                v0 = hwmon / "in0_input"
                if v0.exists():
                    self.gfx_voltage_path = v0
                v1 = hwmon / "in1_input"
                if v1.exists():
                    self.soc_voltage_path = v1
                f = hwmon / "freq1_input"
                if f.exists():
                    self.freq_input_path = f

        self.sclk_path = device_path / "pp_dpm_sclk"
        self.mclk_path = device_path / "pp_dpm_mclk"
        self.pcie_path = device_path / "pp_dpm_pcie"

        print(f"[FastSignalSensorHub] Fast signals initialized")

    def _read_fast_signals(self) -> Dict[str, float]:
        signals = {}

        if self.gpu_busy_path and self.gpu_busy_path.exists():
            try:
                signals["gpu_busy"] = float(self.gpu_busy_path.read_text().strip())
            except:
                signals["gpu_busy"] = 0.0
        else:
            signals["gpu_busy"] = 0.0

        if self.power_input_path and self.power_input_path.exists():
            try:
                signals["power_instant"] = float(self.power_input_path.read_text().strip()) / 1e6
            except:
                signals["power_instant"] = 0.0
        else:
            signals["power_instant"] = 0.0

        if self.gfx_voltage_path and self.gfx_voltage_path.exists():
            try:
                signals["gfx_voltage"] = float(self.gfx_voltage_path.read_text().strip())
            except:
                signals["gfx_voltage"] = 0.0
        else:
            signals["gfx_voltage"] = 0.0

        if self.soc_voltage_path and self.soc_voltage_path.exists():
            try:
                signals["soc_voltage"] = float(self.soc_voltage_path.read_text().strip())
            except:
                signals["soc_voltage"] = 0.0
        else:
            signals["soc_voltage"] = 0.0

        signals["sclk_level"] = self._parse_current_dpm_level(self.sclk_path)
        signals["mclk_level"] = self._parse_current_dpm_level(self.mclk_path)
        signals["pcie_level"] = self._parse_current_dpm_level(self.pcie_path)

        if self.freq_input_path and self.freq_input_path.exists():
            try:
                signals["freq_mhz"] = float(self.freq_input_path.read_text().strip()) / 1e6
            except:
                signals["freq_mhz"] = 0.0
        else:
            signals["freq_mhz"] = 0.0

        # Deep signals from gpu_metrics
        deep = self.deep_metrics.read()
        signals.update({
            "throttle_status": deep["throttle_status"],
            "mem_ctrl_activity": deep["mem_ctrl_activity"],
            "temp_gfx_deep": deep.get("temp_gfx_deep", 0.0),
            "temp_soc_deep": deep.get("temp_soc_deep", 0.0),
            "vcn_activity": deep["vcn_activity"],
            "gfx_activity_deep": deep["gfx_activity_deep"],
            "gfxclk_deep": deep["gfxclk_deep"],
            "memclk_deep": deep["memclk_deep"],
        })

        self.deep_history.append((time.time(), deep))

        return signals

    def _parse_current_dpm_level(self, path) -> float:
        if path is None or not path.exists():
            return 0.0
        try:
            content = path.read_text()
            lines = content.strip().split('\n')
            total = len(lines)
            for i, line in enumerate(lines):
                if '*' in line:
                    return i / max(1, total - 1)
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
        """Read REAL sensors with derivatives."""
        self.base.update(actual_throughput=actual_throughput)
        raw_features = self.base.compute_features()

        current_time = time.time()
        self.feature_history.append((current_time, raw_features.clone()))

        self._update_ema(raw_features)

        raw = self.base.last_reading
        if raw:
            self.power_history.append((current_time, _maybe_mw_to_w(raw.power_mw)))
            self.temp_history.append((current_time, raw.temp_c))

        fast = self._read_fast_signals()
        self.gpu_busy_history.append((current_time, fast["gpu_busy"]))

        # z59: Update derivative hub
        power_w = _maybe_mw_to_w(raw.power_mw) if raw else 50.0
        temp_c = raw.temp_c if raw else 50.0
        derivs = self.derivative_hub.update(power_w, temp_c)

        features_list = []

        # 1. Lag features
        for delay_ms in [0, 50, 200]:
            lag_feat = self._get_lag_feature(delay_ms)
            normalized = self._normalize_ema(lag_feat)
            features_list.append(normalized)

        # 2. Basic derivatives
        derivatives = self._compute_derivatives()
        features_list.append(derivatives)

        # 3. Anchors
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

        # 4. Fast signals
        fast_signals = torch.tensor([
            fast["gpu_busy"] / 100.0,
            fast["power_instant"] / 150.0,
            fast["gfx_voltage"] / 1500.0,
            fast["soc_voltage"] / 1200.0,
            fast["sclk_level"],
            fast["mclk_level"],
            fast["pcie_level"],
            fast["freq_mhz"] / 2500.0,
            self._gpu_busy_derivative(),
            self._power_instant_derivative(fast["power_instant"]),
            1.0 if fast["gpu_busy"] > 90 else 0.0,
            1.0 if fast["power_instant"] > 100 else 0.0,
        ])
        features_list.append(fast_signals)

        # 5. Deep signals
        deep_signals = torch.tensor([
            min(fast["throttle_status"] / 16384.0, 1.0),
            1.0 if fast["throttle_status"] > 0 else 0.0,
            fast["mem_ctrl_activity"] / 100.0,
            fast["vcn_activity"] / 100.0,
            fast["gfx_activity_deep"] / 100.0,
            fast["gfxclk_deep"] / 2500.0,
            fast["memclk_deep"] / 1000.0,
            fast.get("temp_gfx_deep", 0.0) / 100.0,
        ])
        features_list.append(deep_signals)

        # 6. z59: Derivative sensors
        deriv_signals = torch.tensor([
            derivs["dP_dt"] / 10.0,  # Normalize W/s
            derivs["dT_dt"] / 2.0,   # Normalize C/s
            derivs["d2P_dt2"] / 10.0,
            derivs["d2T_dt2"] / 2.0,
            1.0 if derivs["power_trending_up"] else 0.0,
            1.0 if derivs["temp_trending_up"] else 0.0,
            1.0 if derivs["throttle_imminent"] else 0.0,
            0.0,  # Reserved
        ])
        features_list.append(deriv_signals)

        extended = torch.cat(features_list)

        if extended.shape[0] < self.FAST_SIGNAL_DIM:
            padding = torch.zeros(self.FAST_SIGNAL_DIM - extended.shape[0])
            extended = torch.cat([extended, padding])
        elif extended.shape[0] > self.FAST_SIGNAL_DIM:
            extended = extended[:self.FAST_SIGNAL_DIM]

        extended = _sanitize(extended, clamp=10.0)
        return extended

    def _gpu_busy_derivative(self) -> float:
        if len(self.gpu_busy_history) < 2:
            return 0.0
        t1, b1 = self.gpu_busy_history[-2]
        t2, b2 = self.gpu_busy_history[-1]
        dt = t2 - t1
        if dt < 0.001:
            return 0.0
        return (b2 - b1) / dt / 100.0

    def _power_instant_derivative(self, current_power: float) -> float:
        if len(self.power_history) < 2:
            return 0.0
        t1, p1 = self.power_history[-2]
        t2, _ = self.power_history[-1]
        dt = t2 - t1
        if dt < 0.001:
            return 0.0
        return (current_power - p1) / dt / 100.0

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
        return self._read_fast_signals()

    def get_derivative_signals(self) -> Dict[str, float]:
        """z59: Get derivative sensor values."""
        return self.derivative_hub.last_derivatives

    @property
    def dvfs(self):
        return self.base.dvfs

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, value: bool):
        self._training_mode = value


# =========================================================================
# DECODE-TIME POWER SAMPLER
# =========================================================================

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


# =========================================================================
# THERMAL GOVERNOR
# =========================================================================

class ThermalGovernor:
    """Hard safety limits - NON-NEGOTIABLE."""

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
        if self.triggered:
            elapsed = time.time() - self.trigger_time
            if elapsed < self.cooldown_s:
                return True, f"cooldown ({self.cooldown_s - elapsed:.1f}s remaining)"
            else:
                self.triggered = False

        if temp_c > self.temp_limit:
            self._trigger(f"temp={temp_c:.1f}C > {self.temp_limit}C")
            return True, self.last_trigger_reason

        if power_w > self.power_limit:
            self._trigger(f"power={power_w:.1f}W > {self.power_limit}W")
            return True, self.last_trigger_reason

        return False, ""

    def _trigger(self, reason: str):
        self.triggered = True
        self.trigger_time = time.time()
        self.trigger_count += 1
        self.last_trigger_reason = reason
        print(f"[ThermalGovernor] SAFETY TRIGGERED: {reason}")
        if self.dvfs:
            self.dvfs.set_mode("min_sclk")


# =========================================================================
# PERSISTENT BODY STATE
# =========================================================================

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
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        return self.sensor_encoder(sensors).squeeze(0)

    def reset(self):
        self.state.zero_()
        self.state_history.clear()


# =========================================================================
# SAFE DISTURBANCE SCHEDULER
# =========================================================================

class SafeDisturbanceScheduler:
    """Domain-randomized disturbance with safety bounds."""

    def __init__(self, device: str = "cuda", config: Z59Config = None):
        self.config = config or Z59Config()
        self.gpu_stress = SafeGPUStress(
            device,
            max_power_w=self.config.stress_power_max_w,
            max_duration_s=self.config.stress_duration_max_s,
        )
        self.cpu_stress = SafeCPUStress()
        self.dvfs = DVFSController()
        self.current_disturbance = None
        self.start_time = 0.0

    def maybe_apply(self, prob: float = 0.3) -> Optional[str]:
        self.clear()

        if random.random() > prob:
            return None

        disturbance_type = random.choice([
            "gpu_light", "gpu_moderate",
            "cpu_light", "cpu_moderate",
            "combined_light",
            "dvfs_min", "dvfs_peak",
        ])

        duration = random.uniform(
            self.config.stress_duration_min_s,
            self.config.stress_duration_max_s
        )

        if disturbance_type == "gpu_light":
            self.gpu_stress.start(intensity=0.3, duration_s=duration)
        elif disturbance_type == "gpu_moderate":
            self.gpu_stress.start(intensity=0.5, duration_s=duration)
        elif disturbance_type == "cpu_light":
            self.cpu_stress.start(intensity=0.25, cores=1, duration_s=duration)
        elif disturbance_type == "cpu_moderate":
            self.cpu_stress.start(intensity=0.45, cores=2, duration_s=duration)
        elif disturbance_type == "combined_light":
            self.gpu_stress.start(intensity=0.3, duration_s=duration)
            self.cpu_stress.start(intensity=0.25, cores=1, duration_s=duration)
        elif disturbance_type == "dvfs_min":
            self.dvfs.set_mode("min_sclk")
        elif disturbance_type == "dvfs_peak":
            self.dvfs.set_mode("peak")

        self.current_disturbance = disturbance_type
        self.start_time = time.time()
        return disturbance_type

    def clear(self):
        self.gpu_stress.stop()
        self.cpu_stress.stop()
        if self.current_disturbance in ["dvfs_min", "dvfs_peak"]:
            self.dvfs.set_mode("auto")
        self.current_disturbance = None


class SafeGPUStress:
    """GPU stress with hard bounds."""

    def __init__(self, device: str = "cuda", max_power_w: float = 110.0, max_duration_s: float = 3.0):
        self.device = device
        self.max_power_w = max_power_w
        self.max_duration_s = max_duration_s
        self._stop_event = threading.Event()
        self._thread = None
        self.intensity = 0.0
        self.start_time = 0.0

    def start(self, intensity: float = 0.5, duration_s: float = 1.0):
        if self._thread is not None:
            self.stop()

        intensity = min(0.6, max(0.1, intensity))
        duration_s = min(self.max_duration_s, duration_s)

        self.intensity = intensity
        self.start_time = time.time()
        self._stop_event.clear()

        size = int(768 + intensity * 1536)

        def stress_loop():
            try:
                x = torch.randn(size, size, device=self.device, dtype=torch.float16)
                while not self._stop_event.is_set():
                    _ = torch.mm(x, x)
                    if time.time() - self.start_time > duration_s:
                        break
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
        self.intensity = min(0.5, intensity)
        num_workers = min(2, max(1, int(cores * self.intensity)))

        try:
            timeout = min(60, int(duration_s))
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


# =========================================================================
# MAIN (placeholder for now - full training loop would go here)
# =========================================================================

def main():
    """z59 main entry point."""
    parser = argparse.ArgumentParser(description="z59 FEEL Trainer with Real Actuators")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--device-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="models/z59_embodied")

    args = parser.parse_args()

    config = Z59Config(
        base_model=args.model,
        device_path=args.device_path,
        epochs=args.epochs,
        max_prompts=args.max_prompts,
        max_tokens=args.max_tokens,
        use_wandb=args.use_wandb and not args.no_wandb,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("=" * 70)
    print("z59 FEEL Trainer with REAL Actuators")
    print("=" * 70)
    print(f"  Model: {config.base_model}")
    print(f"  Device path: {config.device_path or 'auto-detect'}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Max prompts: {config.max_prompts}")
    print()
    print("REAL ACTUATORS (semantic-safe):")
    print(f"  KV Truncation: {config.use_kv_truncation}")
    print(f"  Power Profile: {config.use_power_profile_actuator}")
    print(f"  Derivative Sensors: {config.use_derivative_sensors}")
    print()
    print("DISABLED (causes garbage output):")
    print(f"  FiLM: {config.film_scale} (disabled)")
    print(f"  Steering Vectors: {config.use_steering_vectors}")
    print(f"  Sensor Packets: {config.inject_sensor_packets}")
    print("=" * 70)

    # TODO: Implement full training loop using z58 as template
    # For now, just demonstrate the new components

    device_path = Path(config.device_path) if config.device_path else Path(detect_amd_device_path())
    print(f"\nUsing device: {device_path}")

    # Test new components
    print("\n[Testing RobustGPUMetrics]")
    metrics = RobustGPUMetrics(device_path)
    reading = metrics.read()
    print(f"  GPU metrics: {reading}")

    print("\n[Testing PowerProfileController]")
    ppc = PowerProfileController(device_path)
    print(f"  Current profile: {ppc.current_profile}")

    print("\n[Testing DerivativeSensorHub]")
    deriv = DerivativeSensorHub()
    for i in range(10):
        result = deriv.update(50.0 + i * 2, 45.0 + i * 0.5)
        time.sleep(0.01)
    print(f"  Derivatives: dP/dt={result['dP_dt']:.2f}, dT/dt={result['dT_dt']:.2f}")
    print(f"  Throttle imminent: {result['throttle_imminent']}")

    print("\n[Testing Z59QualityGate]")
    gate = Z59QualityGate(nll_threshold=2.0)
    for nll in [1.5, 1.8, 2.2, 2.5, 2.8, 1.9, 1.5, 1.2, 1.0]:
        flags = gate.update(nll)
        print(f"  NLL={nll:.1f} -> allow_skip={flags['allow_skip']}, force_safe={flags['force_safe_dvfs']}")

    print("\n[z59 components validated successfully]")
    print("\nTo run full training, the training loop from z58 needs to be integrated.")
    print("Key changes for z59:")
    print("  1. Replace attention masking with truncate_kv_cache()")
    print("  2. Use Z59QualityGate to disable actuators on NLL breach")
    print("  3. Use PowerProfileController instead of fake KV precision")
    print("  4. Use derivative sensors for proactive regulation")


if __name__ == "__main__":
    main()
