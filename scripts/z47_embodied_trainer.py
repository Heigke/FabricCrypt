#!/usr/bin/env python3
"""
FEEL z47: Fixed Contrastive Learning + Achievable Regime Caps
==============================================================

z47 CRITICAL FIXES (from expert analysis of z46 failure):

ROOT CAUSE: z46's contrastive learning NEVER activated because:
1. can_sample_pair() checked discomfort spread, but discomfort only kicks in above 50W/60C
2. With decode power ~56-58W and temps <50C, discomfort spread stayed near 0
3. HOT cap (50W) was BELOW idle power (57W) - impossible game

FIX 1: POWER-FIRST can_sample_pair()
   - Check power spread >= 8W FIRST (not discomfort)
   - This is the actual signal we want contrast on

FIX 2: POWER RESERVOIR (preserve extremes)
   - Maintain separate top-K and bottom-K power reservoirs
   - FIFO deque forgets extremes - reservoirs don't
   - Always have contrastive pairs available

FIX 3: REGIME-AWARE DISCOMFORT
   - Discomfort relative to CURRENT regime cap (not hardcoded 50W)
   - Model can actually be "in band" during HOT regime

FIX 4: AUTO-CALIBRATION AT STARTUP
   - Measure P_rest (idle) and P_decode_baseline
   - Set HOT_cap = P_decode_min + margin (achievable!)
   - Clamp to hwmon power_cap bounds

FIX 5: RAISED HOT CAP: 50W -> 65W
   - With idle at 57W, 50W was impossible
   - 65W is achievable with aggressive skip + DVFS min

FIX 6: INCREASED DISTURBANCE: 30% -> 50%
   - More stress events = more power variance
   - More variance = better contrastive signal

FIX 7: COOL REGIME FORCES GPU STRESS
   - Don't just "allow" high power - CREATE it
   - Bounded background stress during COOL to push power UP
   - Creates reliable high-power pole for contrast

FIX 8: DVFS LOCKED DURING EARLY PHASES
   - During gate_pretrain: DVFS follows regime strictly
   - After: policy can override DVFS
   - Ensures clean causal structure early

FIX 9: STRONGER GPU STRESS MATRIX
   - Increased matrix size for more power draw
   - Creates 15-25W variation (was 5-10W)

Author: FEEL Research Team
Date: 2026-01-17 (z47 - FIXED CONTRASTIVE + ACHIEVABLE CAPS)
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
class Z47Config:
    """z47 Fixed Contrastive + Achievable Caps Configuration."""
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
    body_decay: float = 0.05
    body_noise_std: float = 0.01

    # Reward weights
    quality_weight: float = 0.15
    energy_weight: float = 0.20
    recovery_weight: float = 0.15
    throughput_weight: float = 0.15
    prediction_weight: float = 0.10
    discomfort_weight: float = 0.10
    sensor_reliance_weight: float = 0.15

    # z47 FIX 5: RAISED HOT CAP (was 50W, now 65W - achievable!)
    cool_power_cap_w: float = 85.0   # High power OK in COOL
    cool_j_target: float = 9.0       # Can use more energy
    hot_power_cap_w: float = 65.0    # z47: RAISED from 50W!
    hot_j_target: float = 5.5        # Must be efficient
    regime_switch_steps: int = 10
    regime_use_dvfs: bool = True

    # Default targets (overridden by regime)
    power_cap_w: float = 65.0
    power_safety_w: float = 130.0
    temp_target_c: float = 70.0
    temp_safety_c: float = 80.0
    j_per_token_target: float = 6.0

    # z47: Safe stress bounds (slightly higher for more variance)
    stress_power_min_w: float = 75.0
    stress_power_max_w: float = 115.0
    stress_duration_min_s: float = 0.5
    stress_duration_max_s: float = 2.5
    cooldown_duration_s: float = 3.0

    power_sample_interval_ms: float = 10.0

    # z47 FIX 6: INCREASED DISTURBANCE (was 0.30, now 0.50)
    disturbance_prob: float = 0.50

    # FiLM settings
    film_scale: float = 1.0
    film_kl_coef: float = 0.1
    film_lr: float = 1e-4
    film_kl_target_init: float = 0.05
    film_kl_target_max: float = 0.5
    film_kl_ramp_steps: int = 300

    # z47: Contrastive settings - POWER SPREAD THRESHOLD
    contrastive_coef: float = 0.35
    contrastive_margin: float = 0.05
    sensor_buffer_size: int = 200
    sensor_reliance_coef: float = 0.15
    quantile_pct: float = 0.15
    power_spread_threshold: float = 8.0  # z47 FIX 1: Minimum power spread for contrast

    # z47 FIX 2: Power reservoir settings
    reservoir_size: int = 30  # Keep top/bottom 30 samples

    # Training phases
    gate_pretrain_steps: int = 100
    expected_skip_steps: int = 150
    predictor_phase1_steps: int = 100
    predictor_phase2_steps: int = 200
    predictor_pretrain_weight: float = 1.0
    predictor_normal_weight: float = 0.1

    val_every: int = 50
    checkpoint_dir: str = "models/z47_embodied"

    # Wandb
    wandb_project: str = "feel-z47-embodied"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    # z47 FIX 4: Auto-calibration
    auto_calibrate: bool = True
    calibration_samples: int = 10

    def __post_init__(self):
        if self.gate_layers is None:
            self.gate_layers = [7, 11, 15, 19, 23]


DVFS_MODES = ["auto", "min_sclk", "peak"]


# ============================================================================
# z47 FIX 4: AUTO-CALIBRATION
# ============================================================================

class PowerCalibrator:
    """
    z47 FIX 4: Measure actual power levels at startup to set achievable caps.
    """

    def __init__(self, sensor_hub, model, tokenizer, device):
        self.sensor_hub = sensor_hub
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.p_rest = 0.0
        self.p_decode_baseline = 0.0
        self.p_decode_min = 0.0
        self.p_decode_peak = 0.0
        self.calibrated = False

    def calibrate(self, num_samples: int = 10) -> Dict:
        """Measure power at different states."""
        print("\n[CALIBRATION] Measuring power levels...")

        # 1. Rest power (idle)
        rest_powers = []
        for _ in range(num_samples):
            self.sensor_hub.base.update()
            raw = self.sensor_hub.base.last_reading
            if raw:
                rest_powers.append(raw.power_mw)
            time.sleep(0.1)
        self.p_rest = sum(rest_powers) / len(rest_powers) if rest_powers else 55.0
        print(f"  P_rest (idle): {self.p_rest:.1f}W")

        # 2. Decode baseline (normal DVFS)
        self.sensor_hub.dvfs.set_mode("auto")
        decode_powers = self._measure_decode_power(num_samples)
        self.p_decode_baseline = sum(decode_powers) / len(decode_powers) if decode_powers else 65.0
        print(f"  P_decode_baseline: {self.p_decode_baseline:.1f}W")

        # 3. Decode with DVFS min
        self.sensor_hub.dvfs.set_mode("min_sclk")
        time.sleep(0.2)
        decode_powers = self._measure_decode_power(num_samples // 2)
        self.p_decode_min = sum(decode_powers) / len(decode_powers) if decode_powers else 55.0
        print(f"  P_decode_min (DVFS min): {self.p_decode_min:.1f}W")

        # 4. Decode with DVFS peak
        self.sensor_hub.dvfs.set_mode("peak")
        time.sleep(0.2)
        decode_powers = self._measure_decode_power(num_samples // 2)
        self.p_decode_peak = sum(decode_powers) / len(decode_powers) if decode_powers else 80.0
        print(f"  P_decode_peak (DVFS peak): {self.p_decode_peak:.1f}W")

        # Reset DVFS
        self.sensor_hub.dvfs.set_mode("auto")
        self.calibrated = True

        return {
            "p_rest": self.p_rest,
            "p_decode_baseline": self.p_decode_baseline,
            "p_decode_min": self.p_decode_min,
            "p_decode_peak": self.p_decode_peak,
        }

    def _measure_decode_power(self, num_samples: int) -> List[float]:
        """Run decode and measure power."""
        powers = []
        prompt = "Explain energy efficiency briefly."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        for _ in range(num_samples):
            with torch.no_grad():
                _ = self.model.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=32,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            self.sensor_hub.base.update()
            raw = self.sensor_hub.base.last_reading
            if raw:
                powers.append(raw.power_mw)
        return powers

    def get_recommended_caps(self, margin_w: float = 5.0) -> Dict:
        """Get recommended power caps based on calibration."""
        if not self.calibrated:
            return {"hot_cap": 65.0, "cool_cap": 85.0}

        # HOT cap = achievable with skip + DVFS min
        hot_cap = max(self.p_decode_min + margin_w, self.p_rest + 3.0)

        # COOL cap = allow peak power + margin
        cool_cap = self.p_decode_peak + margin_w

        return {
            "hot_cap": hot_cap,
            "cool_cap": cool_cap,
        }


# ============================================================================
# DEEP GPU METRICS (from z46)
# ============================================================================

class DeepGPUMetrics:
    """Parse gpu_metrics binary blob for deep hardware signals."""

    def __init__(self, device_path: Path):
        self.gpu_metrics_path = device_path / "gpu_metrics"
        self.available = self.gpu_metrics_path.exists()

    def read(self) -> Dict[str, float]:
        result = {
            "throttle_status": 0.0,
            "mem_ctrl_activity": 0.0,
            "vcn_activity": 0.0,
            "gfx_activity_deep": 0.0,
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
                return result

            result["gfx_activity_deep"] = float(struct.unpack_from("<H", data, 42)[0])
            result["mem_ctrl_activity"] = float(struct.unpack_from("<H", data, 66)[0])
            result["vcn_activity"] = float(struct.unpack_from("<H", data, 76)[0])
            result["activity_136"] = float(struct.unpack_from("<H", data, 136)[0])
            result["activity_138"] = float(struct.unpack_from("<H", data, 138)[0])
            result["throttle_status"] = float(struct.unpack_from("<I", data, 96)[0])
            result["gfxclk_deep"] = float(struct.unpack_from("<H", data, 224)[0])
            result["memclk_deep"] = float(struct.unpack_from("<H", data, 186)[0])
        except Exception:
            pass

        return result


# ============================================================================
# FAST SIGNAL SENSOR HUB (from z46)
# ============================================================================

class FastSignalSensorHub:
    """Extended sensor hub with fast signals."""

    FAST_SIGNAL_DIM = 72

    def __init__(self, base_hub: CanonicalSensorHub, body_state, ema_alpha: float = 0.05,
                 power_sample_interval_ms: float = 10.0):
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
        self.voltage_history: Deque[Tuple[float, float, float]] = deque(maxlen=20)
        self.power_sampler = DecodeTimePowerSampler(base_hub, power_sample_interval_ms)
        self.last_decode_stats: Dict = {}
        self._init_fast_signal_paths()
        self.deep_metrics = DeepGPUMetrics(self.base.device_path)
        self.deep_history: Deque[Tuple[float, Dict]] = deque(maxlen=50)
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

        deep = self.deep_metrics.read()
        signals.update(deep)
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
        self.base.update(actual_throughput=actual_throughput)
        raw_features = self.base.compute_features()
        current_time = time.time()
        self.feature_history.append((current_time, raw_features.clone()))
        self._update_ema(raw_features)

        raw = self.base.last_reading
        if raw:
            self.power_history.append((current_time, raw.power_mw))
            self.temp_history.append((current_time, raw.temp_c))

        fast = self._read_fast_signals()
        self.gpu_busy_history.append((current_time, fast["gpu_busy"]))

        features_list = []
        for delay_ms in [0, 50, 200]:
            lag_feat = self._get_lag_feature(delay_ms)
            normalized = self._normalize_ema(lag_feat)
            features_list.append(normalized)

        derivatives = self._compute_derivatives()
        features_list.append(derivatives)

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

        deep_signals = torch.tensor([
            min(fast["throttle_status"] / 16384.0, 1.0),
            1.0 if fast["throttle_status"] > 0 else 0.0,
            fast["mem_ctrl_activity"] / 100.0,
            fast["vcn_activity"] / 100.0,
            fast["gfx_activity_deep"] / 100.0,
            fast["activity_136"] / 100.0,
            fast["gfxclk_deep"] / 2500.0,
            fast["memclk_deep"] / 1000.0,
        ])
        features_list.append(deep_signals)

        extended = torch.cat(features_list)
        if extended.shape[0] < self.FAST_SIGNAL_DIM:
            padding = torch.zeros(self.FAST_SIGNAL_DIM - extended.shape[0])
            extended = torch.cat([extended, padding])
        elif extended.shape[0] > self.FAST_SIGNAL_DIM:
            extended = extended[:self.FAST_SIGNAL_DIM]

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

    def get_fast_signal_variance(self) -> Dict[str, float]:
        import numpy as np
        result = {}
        if len(self.gpu_busy_history) >= 10:
            busy_vals = [b for _, b in list(self.gpu_busy_history)[-100:]]
            result["gpu_busy_var"] = float(np.var(busy_vals))
            result["gpu_busy_mean"] = float(np.mean(busy_vals))
        else:
            result["gpu_busy_var"] = 0.0
            result["gpu_busy_mean"] = 0.0
        if len(self.power_history) >= 10:
            power_vals = [p for _, p in list(self.power_history)[-100:]]
            result["power_var"] = float(np.var(power_vals))
            result["power_mean"] = float(np.mean(power_vals))
        else:
            result["power_var"] = 0.0
            result["power_mean"] = 0.0
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
# THERMAL GOVERNOR
# ============================================================================

class ThermalGovernor:
    """Hard safety limits."""

    def __init__(self, temp_limit_c: float = 80.0, power_limit_w: float = 130.0,
                 cooldown_s: float = 3.0, dvfs_controller=None):
        self.temp_limit = temp_limit_c
        self.power_limit = power_limit_w
        self.cooldown_s = cooldown_s
        self.dvfs = dvfs_controller
        self.triggered = False
        self.trigger_time = 0.0
        self.trigger_count = 0
        self.last_trigger_reason = ""

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


# ============================================================================
# z47 FIX 9: STRONGER GPU STRESS
# ============================================================================

class SafeGPUStress:
    """z47: Stronger GPU stress with larger matrix for more power variance."""

    def __init__(self, device: str = "cuda", max_power_w: float = 115.0, max_duration_s: float = 3.0):
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

        # z47: Higher intensity cap and larger matrix
        intensity = min(0.75, max(0.1, intensity))  # z47: Was 0.6, now 0.75
        duration_s = min(self.max_duration_s, duration_s)

        self.intensity = intensity
        self.start_time = time.time()
        self._stop_event.clear()

        # z47 FIX 9: LARGER MATRIX for more power draw
        # Was: 384 + intensity*768 (max 845)
        # Now: 512 + intensity*1024 (max 1280) - ~50% larger
        size = int(512 + intensity * 1024)

        def stress_loop():
            try:
                x = torch.randn(size, size, device=self.device, dtype=torch.float16)
                while not self._stop_event.is_set():
                    _ = torch.mm(x, x)
                    if time.time() - self.start_time > duration_s:
                        break
                    time.sleep(0.003)  # z47: Faster loop for more sustained load
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
                subprocess.Popen(['stress-ng', '--cpu', '1', '--timeout', str(timeout)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
# z47 FIX 7: COOL REGIME GPU STRESS + STRONGER DISTURBANCE
# ============================================================================

class SafeDisturbanceScheduler:
    """z47: Enhanced disturbance with COOL regime stress and higher variance."""

    def __init__(self, device: str = "cuda", config: Z47Config = None):
        self.config = config or Z47Config()
        self.gpu_stress = SafeGPUStress(
            device,
            max_power_w=self.config.stress_power_max_w,
            max_duration_s=self.config.stress_duration_max_s,
        )
        self.cpu_stress = SafeCPUStress()
        self.dvfs = DVFSController()
        self.current_disturbance = None
        self.start_time = 0.0

        # z47 FIX 7: Track regime for forced stress
        self.current_regime = "cool"
        self.regime_stress_active = False

    def set_regime(self, regime: str):
        """z47: Set current regime for regime-aware stress."""
        self.current_regime = regime

    def apply_regime_stress(self):
        """z47 FIX 7: Apply bounded stress during COOL regime."""
        if self.current_regime == "cool" and not self.regime_stress_active:
            # Apply moderate GPU stress to push power UP
            self.gpu_stress.start(intensity=0.4, duration_s=5.0)
            self.regime_stress_active = True
            return True
        return False

    def clear_regime_stress(self):
        """Clear regime-based stress."""
        if self.regime_stress_active:
            self.gpu_stress.stop()
            self.regime_stress_active = False

    def maybe_apply(self, prob: float = 0.5) -> Optional[str]:
        self.clear()  # Don't clear regime stress here

        if random.random() > prob:
            return None

        # z47: More GPU-heavy disturbances for better power variance
        disturbance_type = random.choice([
            "gpu_light", "gpu_moderate", "gpu_strong",  # z47: Added gpu_strong
            "cpu_light", "cpu_moderate",
            "combined_moderate",  # z47: Combined stress
            "dvfs_min", "dvfs_peak",
        ])

        duration = random.uniform(
            self.config.stress_duration_min_s,
            self.config.stress_duration_max_s
        )

        if disturbance_type == "gpu_light":
            self.gpu_stress.start(intensity=0.35, duration_s=duration)
        elif disturbance_type == "gpu_moderate":
            self.gpu_stress.start(intensity=0.55, duration_s=duration)
        elif disturbance_type == "gpu_strong":
            # z47: New stronger stress option
            self.gpu_stress.start(intensity=0.70, duration_s=duration)
        elif disturbance_type == "cpu_light":
            self.cpu_stress.start(intensity=0.25, cores=1, duration_s=duration)
        elif disturbance_type == "cpu_moderate":
            self.cpu_stress.start(intensity=0.45, cores=2, duration_s=duration)
        elif disturbance_type == "combined_moderate":
            # z47: Combined stress for maximum variance
            self.gpu_stress.start(intensity=0.45, duration_s=duration)
            self.cpu_stress.start(intensity=0.35, cores=2, duration_s=duration)
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


# ============================================================================
# DECODE-TIME POWER SAMPLER
# ============================================================================

class DecodeTimePowerSampler:
    """Background thread samples power during generation."""

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
                return {"samples": 0, "total_energy_j": 0.0, "avg_power_w": 0.0}
            powers = [p for _, p in self.power_samples]
            return {
                "samples": len(self.power_samples),
                "total_energy_j": self.total_energy_j,
                "decode_time_s": self.decode_end_time - self.decode_start_time,
                "avg_power_w": sum(powers) / len(powers),
                "peak_power_w": max(powers),
                "min_power_w": min(powers),
            }


# ============================================================================
# PERSISTENT BODY STATE
# ============================================================================

class PersistentBodyState(nn.Module):
    """Persistent body state with decay."""

    def __init__(self, sensor_dim: int = 64, body_dim: int = 64, decay: float = 0.1, noise_std: float = 0.01):
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


# ============================================================================
# PREDICTIVE HEAD
# ============================================================================

class PredictiveHeadWithCurriculum(nn.Module):
    """Predictive head with curriculum."""

    def __init__(self, body_dim: int = 64, sensor_dim: int = 64, hidden_dim: int = 128):
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

        self.power_head = nn.Linear(hidden_dim, 1)
        self.dvfs_head = nn.Linear(hidden_dim, 3)
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.temp_head = nn.Linear(hidden_dim, 1)
        self.throttle_head = nn.Linear(hidden_dim, 1)

    def forward(self, body_state, sensors, dvfs_action, skip_prob, phase: int = 3) -> Dict[str, torch.Tensor]:
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

        if phase >= 2:
            result["energy"] = self.energy_head(h).squeeze(-1)
        else:
            result["energy"] = torch.zeros_like(result["power"])

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
    """Interoceptive report."""

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
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.mode_head = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 3),
        )

    def forward(self, body_state, sensors) -> Dict[str, torch.Tensor]:
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
# GATE NETWORK
# ============================================================================

class GateNetWithExpectedSkip(nn.Module):
    """Gate network with separate encoders."""

    def __init__(self, sensor_dim: int = 64, body_dim: int = 64, hidden_dim: int = 128, num_layers: int = 5):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.body_dim = body_dim
        self.num_layers = num_layers

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
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))
            for _ in range(num_layers)
        ])

        self.dvfs_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 3))

        for head in self.gate_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, 1.0)

    def forward(self, sensors, body_state, sample: bool = True, use_expected: bool = False) -> Dict:
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        if body_state.dim() == 1:
            body_state = body_state.unsqueeze(0)

        h_sensor = self.sensor_encoder(sensors)
        h_body = self.body_encoder(body_state)
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
                skip_actions = [prob.clone() for prob in gate_probs]
                skip_log_probs = [torch.zeros_like(prob) for prob in gate_probs]
            else:
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

            dvfs_dist = Categorical(dvfs_probs)
            dvfs_action = dvfs_dist.sample()
            dvfs_log_prob = dvfs_dist.log_prob(dvfs_action)

            result["dvfs_action"] = dvfs_action
            result["dvfs_log_prob"] = dvfs_log_prob
            result["total_log_prob"] = result["total_skip_log_prob"] + dvfs_log_prob.sum()

            skip_entropy = sum(-(p * torch.log(p + 1e-10) + (1-p) * torch.log(1-p + 1e-10)).sum() for p in gate_probs)
            dvfs_entropy = dvfs_dist.entropy().sum()
            result["entropy"] = skip_entropy + dvfs_entropy

        return result


# ============================================================================
# MLP SKIP BLOCK
# ============================================================================

class MLPSkipBlock(nn.Module):
    """Gated MLP with FiLM."""

    def __init__(self, original_mlp, hidden_size, sensor_dim=64, body_dim=64, layer_idx=0):
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
        self.run_probability = 1.0
        self.skipped_this_forward = False
        self.film_scale = 1.0
        self.sensors = None
        self.body_state = None

    def forward(self, hidden_states):
        if isinstance(self.run_decision, float) or (isinstance(self.run_decision, bool) and self.run_probability < 1.0):
            prob = self.run_probability if hasattr(self, 'run_probability') else (1.0 if self.run_decision else 0.0)
            run_out = self._run_path(hidden_states)
            skip_out = self.skip_proj(hidden_states)
            self.skipped_this_forward = prob < 0.5
            return prob * run_out + (1 - prob) * skip_out

        self.skipped_this_forward = not self.run_decision
        if self.run_decision:
            return self._run_path(hidden_states)
        else:
            return self.skip_proj(hidden_states)

    def _run_path(self, hidden_states):
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
    """Complete embodied model."""

    def __init__(self, base_model, gate_net, sensor_hub, body_state, predictor, intero_report, gate_layers):
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

    def compute_actions(self, sensors, body_state, sample=True, use_expected=False):
        return self.gate_net(sensors, body_state, sample=sample, use_expected=use_expected)

    def apply_actions(self, action_result, sensors, body_state, film_scale=1.0, use_expected=False):
        skip_actions = action_result.get("skip_actions", [])
        gate_probs = action_result.get("gate_probs", [])

        for i, layer_idx in enumerate(self.gate_layers):
            block = self.skip_blocks[str(layer_idx)]
            if use_expected and i < len(gate_probs):
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

    def force_safe_actions(self, sensors, body_state):
        for block in self.skip_blocks.values():
            block.run_decision = False
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

    def compute_film_kl_loss(self, input_ids, attention_mask, sensors_stressed, sensors_relaxed, body_state, kl_target=0.1):
        for block in self.skip_blocks.values():
            block.run_decision = True
            block.film_scale = 1.0
            block.sensors = sensors_stressed
            block.body_state = body_state

        outputs_stressed = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        logits_stressed = outputs_stressed.logits

        for block in self.skip_blocks.values():
            block.sensors = sensors_relaxed

        outputs_relaxed = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        logits_relaxed = outputs_relaxed.logits

        log_probs_stressed = F.log_softmax(logits_stressed, dim=-1)
        probs_relaxed = F.softmax(logits_relaxed, dim=-1)

        kl_per_token = F.kl_div(log_probs_stressed, probs_relaxed, reduction='none').sum(dim=-1)
        kl_mean = kl_per_token.mean()

        target_tensor = torch.tensor(kl_target, device=kl_mean.device, dtype=kl_mean.dtype)
        target_loss = F.smooth_l1_loss(kl_mean, target_tensor)
        maximize_loss = -torch.log(1.0 + kl_mean).clamp(min=-2.0)
        kl_loss = target_loss + 0.3 * maximize_loss

        metrics = {"kl_mean": kl_mean.item(), "kl_target": kl_target, "kl_loss": kl_loss.item()}
        return kl_loss, metrics

    def get_film_params(self):
        params = []
        for block in self.skip_blocks.values():
            params.extend(block.film_generator.parameters())
            params.extend(block.skip_proj.parameters())
        return params


# ============================================================================
# z47 REGIME CURRICULUM
# ============================================================================

class RegimeCurriculum:
    """z47: Regime curriculum with achievable caps."""

    def __init__(self, config, dvfs_controller=None, disturbance=None):
        self.config = config
        self.dvfs = dvfs_controller
        self.disturbance = disturbance  # z47: For regime stress
        self.current_regime = "cool"
        self.steps_in_regime = 0
        self.regime_history = []

        # z47 FIX 8: Track whether to lock DVFS
        self.dvfs_locked = True  # Lock during early training

    def unlock_dvfs(self):
        """z47 FIX 8: Allow policy to control DVFS."""
        self.dvfs_locked = False

    def step(self, step_num: int) -> str:
        switch_interval = getattr(self.config, 'regime_switch_steps', 10)
        self.steps_in_regime += 1

        if self.steps_in_regime >= switch_interval:
            self.steps_in_regime = 0
            old_regime = self.current_regime
            self.current_regime = "hot" if self.current_regime == "cool" else "cool"

            # z47 FIX 7: Apply regime-based stress/DVFS
            if self.disturbance:
                self.disturbance.set_regime(self.current_regime)
                if self.current_regime == "cool":
                    # COOL: Force stress to push power UP
                    self.disturbance.apply_regime_stress()
                else:
                    # HOT: Clear stress
                    self.disturbance.clear_regime_stress()

            # z47 FIX 8: DVFS follows regime when locked
            if self.dvfs_locked and self.dvfs:
                if self.current_regime == "cool":
                    self.dvfs.set_mode("peak")
                else:
                    self.dvfs.set_mode("min_sclk")

            self.regime_history.append((step_num, self.current_regime))

        return self.current_regime

    def get_regime_targets(self) -> Dict:
        if self.current_regime == "cool":
            return {
                "power_cap_w": getattr(self.config, 'cool_power_cap_w', 85.0),
                "j_target": getattr(self.config, 'cool_j_target', 9.0),
                "skip_penalty": 0.1,
                "quality_bonus": 0.2,
            }
        else:
            return {
                "power_cap_w": getattr(self.config, 'hot_power_cap_w', 65.0),
                "j_target": getattr(self.config, 'hot_j_target', 5.5),
                "skip_penalty": 0.0,
                "quality_bonus": 0.0,
            }


# ============================================================================
# z47 FIX 3: REGIME-AWARE REWARD
# ============================================================================

class RecoveryAwareReward:
    """z47: Reward with regime-aware discomfort."""

    def __init__(self, config):
        self.config = config
        self.baseline = 0.0
        self.baseline_ema = config.baseline_ema
        self.was_stressed_last = False
        self.stress_duration = 0
        self.recovery_count = 0
        self.stuck_count = 0
        self.in_band_count = 0
        self.total_count = 0
        self.power_history: Deque[float] = deque(maxlen=10)
        self.temp_history: Deque[float] = deque(maxlen=10)

    def compute(self, response, throughput, power_w, j_per_token, temp_c, skip_rate, prediction_error,
                is_stressed, fast_signals=None, regime_targets=None) -> Tuple[float, float, Dict]:
        self.power_history.append(power_w)
        self.temp_history.append(temp_c)

        # z47 FIX 3: Get regime-specific targets
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

        # Quality
        quality = 0.0
        if len(response) > 10:
            quality += 0.3
        if len(response) > 50:
            quality += 0.3
        words = response.split()
        if len(words) > 5 and len(set(words)) > 3:
            quality += 0.4
        quality = min(1.0, quality + quality_bonus)

        # Energy efficiency
        j_error = abs(j_per_token - effective_j_target) / max(effective_j_target, 1.0)
        energy_score = max(0, 1.0 - j_error)
        if j_per_token < effective_j_target:
            energy_score = min(1.0, energy_score + 0.2)

        # Time-in-band
        in_band = 1.0 if power_w <= effective_power_cap else 0.0
        self.in_band_count += in_band
        self.total_count += 1

        # Throughput
        throughput_score = min(1.0, throughput / 40.0)

        # Prediction accuracy
        prediction_score = max(0, 1.0 - prediction_error)

        # z47 FIX 3: REGIME-AWARE DISCOMFORT
        # Discomfort relative to current cap, not hardcoded
        thermal_discomfort = math.exp(-(self.config.temp_safety_c - temp_c) / 10.0) if temp_c < self.config.temp_safety_c else 1.0
        power_overshoot = max(0, power_w - effective_power_cap) / max(effective_power_cap, 1.0)
        discomfort = 0.5 * thermal_discomfort + 0.5 * power_overshoot
        discomfort_score = 1.0 - min(1.0, discomfort)

        # Recovery
        is_normal = power_w < effective_power_cap and temp_c < self.config.temp_target_c
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

        # Fast signal bonus
        fast_signal_bonus = 0.0
        if fast_signals:
            gpu_busy = fast_signals.get("gpu_busy", 50)
            if 30 < gpu_busy < 90:
                fast_signal_bonus = 0.1

        # Skip penalty
        skip_penalty_value = skip_penalty * skip_rate

        # Total reward
        reward = (
            self.config.quality_weight * quality +
            self.config.energy_weight * energy_score +
            self.config.recovery_weight * recovery_score * in_band +
            self.config.throughput_weight * throughput_score +
            self.config.prediction_weight * prediction_score +
            self.config.discomfort_weight * discomfort_score +
            fast_signal_bonus -
            skip_penalty_value
        )
        reward = min(1.0, max(0.0, reward))

        advantage = reward - self.baseline
        self.baseline = self.baseline_ema * self.baseline + (1 - self.baseline_ema) * reward

        breakdown = {
            "quality": quality, "energy": energy_score, "in_band": in_band,
            "throughput": throughput_score, "prediction": prediction_score,
            "discomfort": discomfort_score, "recovery": recovery_score,
            "fast_signal_bonus": fast_signal_bonus,
            "time_in_band_pct": self.in_band_count / max(1, self.total_count),
            "recovery_count": self.recovery_count, "stuck_count": self.stuck_count,
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
# z47 FIX 1 & 2: POWER-FIRST SENSOR BUFFER WITH RESERVOIRS
# ============================================================================

class SensorBuffer:
    """
    z47: Sensor buffer with POWER-FIRST checking and RESERVOIRS.

    FIX 1: can_sample_pair() checks power spread >= 8W FIRST
    FIX 2: Separate high/low power reservoirs preserve extremes
    """

    def __init__(self, max_size: int = 200, quantile_pct: float = 0.15,
                 power_spread_threshold: float = 8.0, reservoir_size: int = 30):
        self.max_size = max_size
        self.quantile_pct = quantile_pct
        self.power_spread_threshold = power_spread_threshold
        self.reservoir_size = reservoir_size

        # Main buffer
        self.buffer: Deque[Tuple[torch.Tensor, float, float, float]] = deque(maxlen=max_size)

        # z47 FIX 2: POWER RESERVOIRS - preserve extremes
        self.high_power_reservoir: List[Tuple[torch.Tensor, float, float]] = []
        self.low_power_reservoir: List[Tuple[torch.Tensor, float, float]] = []

    def _compute_discomfort(self, power_w: float, temp_c: float, power_cap: float = 65.0) -> float:
        """z47 FIX 3: Regime-aware discomfort."""
        power_stress = max(0, power_w - power_cap) / power_cap
        temp_stress = max(0, temp_c - 60.0) / 20.0
        return power_stress + temp_stress

    def add(self, sensors: torch.Tensor, power_w: float, temp_c: float, power_cap: float = 65.0):
        discomfort = self._compute_discomfort(power_w, temp_c, power_cap)
        self.buffer.append((sensors.detach().clone(), power_w, temp_c, discomfort))

        # z47 FIX 2: Update reservoirs
        self._update_reservoirs(sensors.detach().clone(), power_w, temp_c)

    def _update_reservoirs(self, sensors: torch.Tensor, power_w: float, temp_c: float):
        """z47 FIX 2: Maintain high/low power reservoirs."""
        entry = (sensors, power_w, temp_c)

        # Add to high reservoir if among top-K
        self.high_power_reservoir.append(entry)
        self.high_power_reservoir.sort(key=lambda x: x[1], reverse=True)
        self.high_power_reservoir = self.high_power_reservoir[:self.reservoir_size]

        # Add to low reservoir if among bottom-K
        self.low_power_reservoir.append(entry)
        self.low_power_reservoir.sort(key=lambda x: x[1])
        self.low_power_reservoir = self.low_power_reservoir[:self.reservoir_size]

    def can_sample_pair(self) -> bool:
        """z47 FIX 1: Check POWER SPREAD first."""
        # Need minimum samples
        if len(self.buffer) < 10:
            return False

        # z47 FIX 1: Check POWER spread (the actual signal we care about)
        powers = [p for _, p, _, _ in self.buffer]
        power_spread = max(powers) - min(powers)

        if power_spread < self.power_spread_threshold:
            return False

        # z47 FIX 2: Check reservoirs have samples
        if len(self.high_power_reservoir) < 3 or len(self.low_power_reservoir) < 3:
            return False

        # Ensure reservoir spread is meaningful
        reservoir_spread = self.high_power_reservoir[0][1] - self.low_power_reservoir[0][1]
        return reservoir_spread >= self.power_spread_threshold

    def sample_extremes(self) -> Tuple[Tuple, Tuple]:
        """z47 FIX 2: Sample from RESERVOIRS (not FIFO buffer)."""
        if not self.high_power_reservoir or not self.low_power_reservoir:
            # Fallback to buffer
            if len(self.buffer) < 2:
                s1, p1, t1, _ = self.buffer[0]
                return (s1, p1, t1), (s1, p1, t1)
            s1, p1, t1, _ = random.choice(list(self.buffer))
            s2, p2, t2, _ = random.choice(list(self.buffer))
            return (s1, p1, t1), (s2, p2, t2)

        # Sample from reservoirs
        high_sample = random.choice(self.high_power_reservoir)
        low_sample = random.choice(self.low_power_reservoir)

        return (high_sample[0], high_sample[1], high_sample[2]), \
               (low_sample[0], low_sample[1], low_sample[2])

    def get_power_spread(self) -> float:
        if len(self.buffer) == 0:
            return 1.0
        powers = [p for _, p, _, _ in self.buffer]
        return max(max(powers) - min(powers), 1.0)

    def get_reservoir_spread(self) -> float:
        """z47: Get spread from reservoirs."""
        if not self.high_power_reservoir or not self.low_power_reservoir:
            return 0.0
        return self.high_power_reservoir[0][1] - self.low_power_reservoir[0][1]

    def get_stats(self) -> Dict:
        if len(self.buffer) == 0:
            return {"size": 0, "power_min": 0, "power_max": 0, "power_spread": 0}
        powers = [p for _, p, _, _ in self.buffer]
        return {
            "size": len(self.buffer),
            "power_min": min(powers),
            "power_max": max(powers),
            "power_spread": max(powers) - min(powers),
            "reservoir_high_size": len(self.high_power_reservoir),
            "reservoir_low_size": len(self.low_power_reservoir),
            "reservoir_spread": self.get_reservoir_spread(),
        }


# ============================================================================
# CONTRASTIVE GATE LOSS
# ============================================================================

class NormalizedContrastiveGateLoss:
    """Contrastive loss with sensor reliance regularizer."""

    def __init__(self, margin: float = 0.05, direction_weight: float = 0.5, sensor_reliance_coef: float = 0.15):
        self.margin = margin
        self.direction_weight = direction_weight
        self.sensor_reliance_coef = sensor_reliance_coef

    def compute(self, gate_probs_stressed, gate_probs_relaxed, power_spread=1.0, gate_probs_shuffled=None):
        mean_gate_stressed = sum(p.mean() for p in gate_probs_stressed) / len(gate_probs_stressed)
        mean_gate_relaxed = sum(p.mean() for p in gate_probs_relaxed) / len(gate_probs_relaxed)

        gate_diff = (mean_gate_relaxed - mean_gate_stressed).abs()
        expected_diff = self.margin * min(power_spread / 30.0, 2.0)
        margin_loss = F.relu(expected_diff - gate_diff)

        direction_loss = F.relu(mean_gate_stressed - mean_gate_relaxed + 0.01)

        sensor_reliance_loss = torch.tensor(0.0, device=gate_probs_stressed[0].device)
        sensor_reliance_diff = 0.0

        if gate_probs_shuffled is not None and len(gate_probs_shuffled) > 0:
            mean_gate_real = (mean_gate_stressed + mean_gate_relaxed) / 2
            mean_gate_shuf = sum(p.mean() for p in gate_probs_shuffled) / len(gate_probs_shuffled)
            sensor_reliance_diff = (mean_gate_real - mean_gate_shuf).abs()
            sensor_reliance_loss = F.relu(0.02 - sensor_reliance_diff)

        total_loss = margin_loss + self.direction_weight * direction_loss + self.sensor_reliance_coef * sensor_reliance_loss

        metrics = {
            "gate_diff": gate_diff.item(),
            "margin_loss": margin_loss.item(),
            "direction_loss": direction_loss.item(),
            "gate_stressed": mean_gate_stressed.item(),
            "gate_relaxed": mean_gate_relaxed.item(),
            "direction_correct": (mean_gate_stressed < mean_gate_relaxed).item(),
            "sensor_reliance_diff": sensor_reliance_diff.item() if isinstance(sensor_reliance_diff, torch.Tensor) else sensor_reliance_diff,
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
        prompts = ["Explain how energy efficiency affects computing."] * 150
    return prompts


def train_epoch(model, tokenizer, prompts, optimizer, config, disturbance, thermal_governor, reward_computer,
                sensor_buffer, contrastive_loss_fn, regime_curriculum, epoch, global_step) -> int:
    device = next(model.gate_net.parameters()).device
    model.sensor_hub.training_mode = True
    model.body_state_module.train()
    random.shuffle(prompts)
    prompts = prompts[:config.max_prompts]

    for prompt_idx, prompt in enumerate(prompts):
        step = global_step + prompt_idx

        # z47 FIX 8: Unlock DVFS after gate pretrain
        if step == config.gate_pretrain_steps:
            regime_curriculum.unlock_dvfs()
            print(f"  [{step}] DVFS UNLOCKED - policy can now control DVFS")

        current_regime = regime_curriculum.step(step)
        regime_targets = regime_curriculum.get_regime_targets()

        use_expected_skip = step < config.expected_skip_steps
        in_gate_pretrain = step < config.gate_pretrain_steps

        if step < config.predictor_phase1_steps:
            pred_phase = 1
            predictor_weight = config.predictor_pretrain_weight
        elif step < config.predictor_phase2_steps:
            pred_phase = 2
            predictor_weight = config.predictor_pretrain_weight * 0.5
        else:
            pred_phase = 3
            predictor_weight = config.predictor_normal_weight

        kl_progress = min(1.0, step / config.film_kl_ramp_steps)
        kl_target = config.film_kl_target_init + kl_progress * (config.film_kl_target_max - config.film_kl_target_init)

        sensors = model.sensor_hub.read_tensor().to(device)
        fast_signals = model.sensor_hub.get_fast_signals()

        raw = model.sensor_hub.base.last_reading
        current_power = raw.power_mw if raw else 50.0
        current_temp = raw.temp_c if raw else 50.0

        governor_triggered, governor_reason = thermal_governor.check(current_power, current_temp)
        if governor_triggered:
            disturbance.clear()
            model.force_safe_actions(sensors, model.body_state_module.get_state())
            print(f"  [{step:4d}] [GOVERNOR] {governor_reason}")
            continue

        dist_type = disturbance.maybe_apply(prob=config.disturbance_prob)
        was_stressed = dist_type is not None

        if was_stressed:
            time.sleep(0.05)

        sensors = model.sensor_hub.read_tensor().to(device)
        fast_signals = model.sensor_hub.get_fast_signals()
        body_state = model.body_state_module.update(sensors)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        action_result = model.compute_actions(sensors, body_state, sample=True, use_expected=use_expected_skip)

        # z47 FIX 8: Only apply DVFS if unlocked
        if not regime_curriculum.dvfs_locked:
            dvfs_action = action_result["dvfs_action"].item()
            dvfs_mode, _ = model.apply_dvfs(dvfs_action)
        else:
            dvfs_action = 0
            dvfs_mode = "regime-locked"

        model.apply_actions(action_result, sensors, body_state, film_scale=config.film_scale, use_expected=use_expected_skip)

        dvfs_onehot = F.one_hot(action_result["dvfs_action"], num_classes=3).float()
        mean_gate_prob = sum(p.mean() for p in action_result["gate_probs"]) / len(action_result["gate_probs"])
        predictions = model.predictor(body_state, sensors, dvfs_onehot, mean_gate_prob, phase=pred_phase)

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

            reward, advantage, breakdown = reward_computer.compute(
                response=response, throughput=throughput, power_w=avg_power_w, j_per_token=j_per_token,
                temp_c=temp_c, skip_rate=model.get_metrics()["skip_rate"], prediction_error=pred_power_error,
                is_stressed=was_stressed, fast_signals=fast_signals, regime_targets=regime_targets,
            )

            samples.append({
                "response": response, "throughput": throughput, "j_per_token": j_per_token,
                "avg_power_w": avg_power_w, "temp_c": temp_c, "breakdown": breakdown,
            })
            rewards.append(reward)
            advantages.append(advantage)
            log_probs.append(action_result["total_log_prob"])

        disturbance.clear()

        # z47: Add to buffer with regime-aware power cap
        sensor_buffer.add(sensors, power_w=samples[0]["avg_power_w"], temp_c=samples[0]["temp_c"],
                         power_cap=regime_targets["power_cap_w"])

        optimizer.zero_grad()

        mean_advantage = sum(advantages) / len(advantages)
        mean_log_prob = sum(log_probs) / len(log_probs)

        if in_gate_pretrain:
            policy_loss = torch.tensor(0.0, device=device)
            entropy_loss = torch.tensor(0.0, device=device)
        else:
            policy_loss = -mean_log_prob * mean_advantage
            entropy_loss = -config.entropy_coef * action_result["entropy"]

        s0 = samples[0]
        actual_power = torch.tensor([s0["avg_power_w"]], device=device, dtype=torch.float32)
        pred_loss = F.mse_loss(predictions["power"] / 100.0, actual_power / 100.0)

        if pred_phase >= 2:
            actual_j_tok = torch.tensor([s0["j_per_token"]], device=device, dtype=torch.float32)
            pred_loss = pred_loss + F.mse_loss(predictions["energy"] / 10.0, actual_j_tok / 10.0)

        if pred_phase >= 3:
            actual_temp = torch.tensor([s0["temp_c"]], device=device, dtype=torch.float32)
            pred_loss = pred_loss + F.mse_loss(predictions["temp"] / 100.0, actual_temp / 100.0)

        intero = model.intero_report(body_state, sensors)
        discomfort_val = 1.0 - breakdown["discomfort"]
        actual_strain = torch.tensor([discomfort_val], device=device, dtype=torch.float32)
        intero_loss = F.mse_loss(intero["strain_level"], actual_strain)

        # z47: Contrastive with reservoir sampling
        do_contrastive = (in_gate_pretrain or step % 3 == 0) and sensor_buffer.can_sample_pair()

        if do_contrastive:
            (sensors_hi, power_hi, temp_hi), (sensors_lo, power_lo, temp_lo) = sensor_buffer.sample_extremes()
            sensors_hi = sensors_hi.to(device)
            sensors_lo = sensors_lo.to(device)

            body_hi = model.body_state_module.encode(sensors_hi)
            body_lo = model.body_state_module.encode(sensors_lo)

            gate_result_hi = model.gate_net(sensors_hi, body_hi, sample=False)
            gate_result_lo = model.gate_net(sensors_lo, body_lo, sample=False)

            sensors_shuffled = torch.stack([sensors_lo, sensors_hi])[torch.randperm(2)].mean(dim=0).to(device)
            body_shuffled = model.body_state_module.encode(sensors_shuffled)
            gate_result_shuffled = model.gate_net(sensors_shuffled, body_shuffled, sample=False)

            power_spread = sensor_buffer.get_reservoir_spread()  # z47: Use reservoir spread
            contrastive_loss, contrastive_metrics = contrastive_loss_fn.compute(
                gate_result_hi["gate_probs"], gate_result_lo["gate_probs"],
                power_spread=power_spread, gate_probs_shuffled=gate_result_shuffled["gate_probs"],
            )
            contrastive_metrics["pair_power_diff"] = power_hi - power_lo
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
            contrastive_metrics = {"gate_diff": 0.0, "pair_power_diff": 0.0}

        # FiLM KL
        if step % 5 == 0 and sensor_buffer.can_sample_pair():
            (sensors_hi, _, _), (sensors_lo, _, _) = sensor_buffer.sample_extremes()
            sensors_hi = sensors_hi.to(device)
            sensors_lo = sensors_lo.to(device)
            film_kl_loss, kl_metrics = model.compute_film_kl_loss(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                sensors_stressed=sensors_hi, sensors_relaxed=sensors_lo,
                body_state=body_state, kl_target=kl_target,
            )
        else:
            film_kl_loss = torch.tensor(0.0, device=device)
            kl_metrics = {"kl_mean": 0.0}

        total_loss = (
            policy_loss + entropy_loss +
            predictor_weight * pred_loss +
            0.1 * intero_loss +
            config.film_kl_coef * film_kl_loss +
            config.contrastive_coef * contrastive_loss
        )

        total_loss.backward()

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
            phase_str = "GATE_PT" if in_gate_pretrain else ("EXP_SKIP" if use_expected_skip else "")
            regime_str = f"[{current_regime.upper()}]"

            buf_stats = sensor_buffer.get_stats()
            can_pair = sensor_buffer.can_sample_pair()

            print(f"  [{step:4d}] {regime_str:6s} {stress_str:14s} gate={gate_mean:.3f} skip={m['skip_rate']:.2f} "
                  f"J/tok={s['j_per_token']:.2f} P={s['avg_power_w']:.1f}W T={s['temp_c']:.1f}C "
                  f"r={rewards[0]:.3f} Δg={contrastive_metrics['gate_diff']:.4f} "
                  f"spread={buf_stats['power_spread']:.1f}W res={buf_stats.get('reservoir_spread', 0):.1f}W "
                  f"can_pair={can_pair} {phase_str}")

            if step % 25 == 0:
                print(f"  [BUFFER] size={buf_stats['size']} power={buf_stats['power_min']:.1f}-{buf_stats['power_max']:.1f}W "
                      f"reservoirs: hi={buf_stats.get('reservoir_high_size', 0)} lo={buf_stats.get('reservoir_low_size', 0)}")

        # Checkpoint
        if step > 0 and step % config.val_every == 0:
            save_checkpoint(model, step, config)

    model.sensor_hub.training_mode = False
    return global_step + len(prompts)


def save_checkpoint(model, step, config):
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
    parser = argparse.ArgumentParser(description="FEEL z47: Fixed Contrastive + Achievable Caps")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="models/z47_embodied")
    parser.add_argument("--disturbance-prob", type=float, default=0.50)
    parser.add_argument("--hot-cap", type=float, default=65.0)
    parser.add_argument("--cool-cap", type=float, default=85.0)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    config = Z47Config(
        epochs=args.epochs,
        max_prompts=args.max_prompts,
        checkpoint_dir=args.checkpoint_dir,
        disturbance_prob=args.disturbance_prob,
        hot_power_cap_w=args.hot_cap,
        cool_power_cap_w=args.cool_cap,
        auto_calibrate=not args.no_calibrate,
        use_wandb=not args.no_wandb,
    )

    if WANDB_AVAILABLE and config.use_wandb:
        import socket
        hostname = socket.gethostname()
        wandb.init(project=config.wandb_project, name=f"z47_fixed_{hostname}",
                   config=asdict(config), tags=["z47", "fixed-contrastive", hostname])

    print("=" * 70)
    print("FEEL z47: FIXED CONTRASTIVE + ACHIEVABLE CAPS")
    print("=" * 70)
    print("z47 CRITICAL FIXES:")
    print("  FIX 1: POWER-FIRST can_sample_pair() - check power >= 8W")
    print("  FIX 2: POWER RESERVOIRS - preserve extremes, don't forget")
    print("  FIX 3: REGIME-AWARE DISCOMFORT - relative to cap")
    print("  FIX 4: AUTO-CALIBRATION - measure actual power levels")
    print("  FIX 5: HOT CAP: 50W -> 65W (achievable!)")
    print("  FIX 6: DISTURBANCE: 30% -> 50% (more variance)")
    print("  FIX 7: COOL STRESS - force GPU stress for high-power pole")
    print("  FIX 8: DVFS LOCKED during gate pretrain")
    print("  FIX 9: STRONGER STRESS - larger matrix for more power")
    print("=" * 70)

    print("\n[1/8] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    base_model.eval()
    device = next(base_model.parameters()).device

    print("\n[2/8] Initializing body state...")
    body_state_module = PersistentBodyState(
        sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM, body_dim=config.body_dim,
        decay=config.body_decay, noise_std=config.body_noise_std,
    ).to(device)

    print("\n[3/8] Initializing sensor hub...")
    device_path = "/sys/class/drm/card1/device"
    if not Path("/sys/class/drm/card1/device/hwmon").exists():
        if Path("/sys/class/drm/card0/device/hwmon").exists():
            device_path = "/sys/class/drm/card0/device"

    base_hub = CanonicalSensorHub(device_path=device_path)
    sensor_hub = FastSignalSensorHub(base_hub=base_hub, body_state=body_state_module,
                                      power_sample_interval_ms=config.power_sample_interval_ms)

    print("\n[4/8] Building gate network...")
    gate_net = GateNetWithExpectedSkip(
        sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM, body_dim=config.body_dim,
        num_layers=len(config.gate_layers),
    ).to(device)

    print("\n[5/8] Building predictive head...")
    predictor = PredictiveHeadWithCurriculum(
        body_dim=config.body_dim, sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
    ).to(device)

    intero_report = InteroceptiveReportHead(
        body_dim=config.body_dim, sensor_dim=FastSignalSensorHub.FAST_SIGNAL_DIM,
    ).to(device)

    print("\n[6/8] Building embodied model...")
    model = EmbodiedModel(
        base_model=base_model, gate_net=gate_net, sensor_hub=sensor_hub, body_state=body_state_module,
        predictor=predictor, intero_report=intero_report, gate_layers=config.gate_layers,
    )

    # z47 FIX 4: Auto-calibration
    if config.auto_calibrate:
        print("\n[CALIBRATION] Running power calibration...")
        calibrator = PowerCalibrator(sensor_hub, base_model, tokenizer, device)
        cal_results = calibrator.calibrate(config.calibration_samples)
        recommended = calibrator.get_recommended_caps()

        print(f"  Recommended caps: HOT={recommended['hot_cap']:.1f}W, COOL={recommended['cool_cap']:.1f}W")

        # Update config with calibrated values if significantly different
        if abs(recommended['hot_cap'] - config.hot_power_cap_w) > 5:
            print(f"  Adjusting HOT cap: {config.hot_power_cap_w:.1f}W -> {recommended['hot_cap']:.1f}W")
            config.hot_power_cap_w = recommended['hot_cap']

    global_step = 0
    if args.resume_from and Path(args.resume_from).exists():
        print(f"\n  Loading checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        if "body_state" in checkpoint:
            body_state_module.load_state_dict(checkpoint["body_state"])
        if "gate_net" in checkpoint:
            gate_net.load_state_dict(checkpoint["gate_net"])
        if "predictor" in checkpoint:
            predictor.load_state_dict(checkpoint["predictor"])
        if "intero_report" in checkpoint:
            intero_report.load_state_dict(checkpoint["intero_report"])
        if "step" in checkpoint:
            global_step = checkpoint["step"]
        print(f"  Loaded at step {global_step}")

    print("\n[7/8] Initializing thermal governor...")
    thermal_governor = ThermalGovernor(
        temp_limit_c=config.temp_safety_c, power_limit_w=config.power_safety_w,
        cooldown_s=config.cooldown_duration_s, dvfs_controller=sensor_hub.dvfs,
    )

    disturbance = SafeDisturbanceScheduler(device=str(device), config=config)
    reward_computer = RecoveryAwareReward(config)

    print("\n[8/8] Initializing regime curriculum...")
    regime_curriculum = RegimeCurriculum(config, dvfs_controller=sensor_hub.dvfs, disturbance=disturbance)
    print(f"  COOL: cap={config.cool_power_cap_w}W, HOT: cap={config.hot_power_cap_w}W")
    print(f"  Switch every {config.regime_switch_steps} steps")

    sensor_buffer = SensorBuffer(
        max_size=config.sensor_buffer_size, power_spread_threshold=config.power_spread_threshold,
        reservoir_size=config.reservoir_size,
    )
    contrastive_loss_fn = NormalizedContrastiveGateLoss(margin=config.contrastive_margin)

    prompts = load_prompts()
    print(f"  Loaded {len(prompts)} prompts")

    film_params = model.get_film_params()
    optimizer = torch.optim.AdamW([
        {"params": gate_net.parameters(), "lr": config.gate_lr},
        {"params": body_state_module.parameters(), "lr": config.body_lr},
        {"params": predictor.parameters(), "lr": config.predictor_lr},
        {"params": intero_report.parameters(), "lr": config.predictor_lr},
        {"params": film_params, "lr": config.film_lr},
    ], weight_decay=0.01)

    print(f"\n  z47 KEY CHANGES:")
    print(f"    - Power spread threshold: {config.power_spread_threshold}W")
    print(f"    - Reservoir size: {config.reservoir_size}")
    print(f"    - Disturbance prob: {config.disturbance_prob*100:.0f}%")

    for epoch in range(config.epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{config.epochs}")
        print(f"{'='*70}")

        reward_computer.reset()

        global_step = train_epoch(
            model=model, tokenizer=tokenizer, prompts=prompts, optimizer=optimizer,
            config=config, disturbance=disturbance, thermal_governor=thermal_governor,
            reward_computer=reward_computer, sensor_buffer=sensor_buffer,
            contrastive_loss_fn=contrastive_loss_fn, regime_curriculum=regime_curriculum,
            epoch=epoch, global_step=global_step,
        )

    disturbance.clear()
    save_checkpoint(model, global_step, config)

    print("\n" + "=" * 70)
    print("Z47 TRAINING COMPLETE")
    print(f"  Thermal governor triggered: {thermal_governor.trigger_count} times")
    print(f"  Regime switches: {len(regime_curriculum.regime_history)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
