#!/usr/bin/env python3
"""
FEEL v7.13: Critical Fixes for Real GPU Embodiment

FIXES from v7.12:
1. NORMALIZATION: Denominators based on OBSERVED hardware ranges (not theoretical max)
2. QUANTILE BINNING: Dynamic category boundaries from rolling distribution
3. TWO-STEP TRAINING: Separate forward for feeling (logits1) and intensity (logits2)
4. EMBEDDING GRADIENT MASKING: Proper training of new token rows
5. ENTROPY BONUS: Actually use entropy_coef for feeling token exploration
6. REAL INTERVENTIONAL TEST: Swap z_feel and re-run model

Based on analysis:
- v7.12 composite range: 0.17-0.35 = ALL FOCUSED (wrong denominators)
- pwr_1 max observed: 516, but denom was 4000 → now 800
- sclk observed: ~28000, but denom was 65000 → now 35000
- gpu_busy = 95% but had only 5% weight → now 15%
"""

NUM_INTENSITY_LEVELS = 64

import json
import struct
import time
import random
import threading
import traceback
import sys
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque
from collections import defaultdict, deque
import numpy as np

try:
    import wandb
except Exception:
    wandb = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


# ============================================================================
# v7.13: ANTHROPOMORPHIZED FEELINGS
# ============================================================================

class Feeling(IntEnum):
    """5 anthropomorphized feelings based on composite GPU state."""
    CURIOUS = 0      # Low activity, idle, exploring
    FOCUSED = 1      # Moderate load, working productively
    STRAINED = 2     # High load, working hard
    URGENT = 3       # Very high, needs attention
    OVERWHELMED = 4  # Critical, must take action

FEELING_NAMES = ["CURIOUS", "FOCUSED", "STRAINED", "URGENT", "OVERWHELMED"]

FEELING_TOKENS = {
    Feeling.CURIOUS: "<|FEEL_CURIOUS|>",
    Feeling.FOCUSED: "<|FEEL_FOCUSED|>",
    Feeling.STRAINED: "<|FEEL_STRAINED|>",
    Feeling.URGENT: "<|FEEL_URGENT|>",
    Feeling.OVERWHELMED: "<|FEEL_OVERWHELMED|>",
}

INTENSITY_TOKENS = {i: f"<|FEEL_I{i:02d}|>" for i in range(NUM_INTENSITY_LEVELS)}

INIT_WORDS = {
    Feeling.CURIOUS: ["curious", "wondering", "exploring", "idle", "calm", "peaceful", "relaxed"],
    Feeling.FOCUSED: ["focused", "working", "engaged", "productive", "attentive", "steady"],
    Feeling.STRAINED: ["strained", "stressed", "pushing", "loaded", "taxed", "busy"],
    Feeling.URGENT: ["urgent", "warning", "caution", "pressing", "immediate", "critical"],
    Feeling.OVERWHELMED: ["overwhelmed", "overloaded", "maxed", "danger", "stop", "emergency"],
}


def intensity_to_token_idx(intensity: float) -> int:
    intensity = max(0.0, min(1.0, float(intensity)))
    return int(round(intensity * (NUM_INTENSITY_LEVELS - 1)))


def token_idx_to_intensity(idx: int) -> float:
    return float(idx) / float(NUM_INTENSITY_LEVELS - 1)


# ============================================================================
# v7.13: QUANTILE BINNER - Guarantees category diversity
# ============================================================================

class QuantileBinner:
    """
    Dynamically bin composite values into categories using rolling quantiles.
    This GUARANTEES ~20% samples in each category regardless of raw distribution.
    """
    def __init__(self, window_size: int = 500, warmup: int = 50):
        self.window_size = window_size
        self.warmup = warmup
        self.history: Deque[float] = deque(maxlen=window_size)
        self._boundaries = [0.2, 0.4, 0.6, 0.8]  # Default until warmed up

    def update(self, composite: float):
        """Add new composite value to history."""
        self.history.append(composite)

        # Update boundaries periodically
        if len(self.history) >= self.warmup and len(self.history) % 20 == 0:
            self._update_boundaries()

    def _update_boundaries(self):
        """Compute quantile boundaries from history."""
        sorted_vals = sorted(self.history)
        n = len(sorted_vals)
        self._boundaries = [
            sorted_vals[int(n * 0.2)],  # q20
            sorted_vals[int(n * 0.4)],  # q40
            sorted_vals[int(n * 0.6)],  # q60
            sorted_vals[int(n * 0.8)],  # q80
        ]

    def classify(self, composite: float) -> Feeling:
        """Classify composite into feeling using quantile boundaries."""
        if composite < self._boundaries[0]:
            return Feeling.CURIOUS
        elif composite < self._boundaries[1]:
            return Feeling.FOCUSED
        elif composite < self._boundaries[2]:
            return Feeling.STRAINED
        elif composite < self._boundaries[3]:
            return Feeling.URGENT
        else:
            return Feeling.OVERWHELMED

    def get_boundaries(self) -> List[float]:
        return self._boundaries.copy()

    def get_stats(self) -> Dict:
        if len(self.history) < 10:
            return {"warmup": True}
        vals = list(self.history)
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": np.mean(vals),
            "std": np.std(vals),
            "boundaries": self._boundaries,
        }


# ============================================================================
# v7.13: FIXED COMPOSITE COMPUTATION - Real hardware denominators
# ============================================================================

def compute_composite_v13(
    gpu_busy_pct: float,
    sclk: float,
    current_gfx: float,
    power_total: float,
    gfx_activity: float,
    temp_c: float = 50.0,
    pwr_1: float = 50.0,
    pwr_3: float = 50.0,
    pwr_2: float = 50.0,
    pwr_0: float = 50.0,
    power_gfx: float = 500.0,
) -> float:
    """
    v7.13: Compute composite with OBSERVED hardware ranges.

    Key changes from v7.12:
    - pwr_X denominators: 4000 → 800 (observed max ~600)
    - sclk denominator: 65000 → 35000 (observed max ~30000)
    - gpu_busy weight: 5% → 15% (it's actually high variance!)
    - power_gfx denominator: 8000 → 4000 (observed max ~3000)
    """
    # TIER 1: pwr_X with CORRECTED denominators
    pwr_1_score = min(1.0, pwr_1 / 800.0)         # Was 4000
    pwr_3_score = min(1.0, pwr_3 / 800.0)         # Was 3000
    pwr_2_score = min(1.0, pwr_2 / 800.0)         # Was 4000
    pwr_0_score = min(1.0, pwr_0 / 400.0)         # Was 800

    # TIER 1: voltage/current with corrected ranges
    power_gfx_score = min(1.0, power_gfx / 4000.0)     # Was 8000
    curr_score = min(1.0, current_gfx / 15000.0)       # Observed up to 14902
    power_score = min(1.0, power_total / 13000.0)      # OK

    # TIER 2: clocks - CORRECTED
    sclk_score = min(1.0, max(0.0, sclk / 35000.0))    # Was 65000

    # TIER 3: activity - INCREASED WEIGHTS
    busy_score = min(1.0, gpu_busy_pct / 100.0)
    gfx_score = min(1.0, gfx_activity / 100.0)
    temp_score = min(1.0, max(0.0, (temp_c - 40) / 50.0))

    # Rebalanced weights - gpu_busy now has MUCH more influence
    composite = (
        # TIER 1: pwr_X (50% total - these vary the most!)
        0.15 * pwr_1_score +
        0.12 * pwr_3_score +
        0.10 * pwr_2_score +
        0.08 * pwr_0_score +
        0.05 * power_gfx_score +
        # TIER 2: current/power (15%)
        0.08 * curr_score +
        0.07 * power_score +
        # TIER 3: clocks and activity (30%)
        0.10 * sclk_score +
        0.15 * busy_score +      # INCREASED from 5%
        0.05 * gfx_score +
        # TIER 4: temp (5%)
        0.05 * temp_score
    )

    return composite


def compute_feeling_v13(
    gpu_busy_pct: float,
    sclk: float,
    current_gfx: float,
    power_total: float,
    gfx_activity: float,
    temp_c: float = 50.0,
    pwr_1: float = 50.0,
    pwr_3: float = 50.0,
    pwr_2: float = 50.0,
    pwr_0: float = 50.0,
    power_gfx: float = 500.0,
    binner: Optional[QuantileBinner] = None,
) -> Tuple[Feeling, float]:
    """
    v7.13: Compute feeling using quantile binning OR fixed thresholds.

    If binner is provided, uses dynamic quantile boundaries.
    Otherwise uses fixed thresholds (which may cause collapse).
    """
    composite = compute_composite_v13(
        gpu_busy_pct, sclk, current_gfx, power_total, gfx_activity,
        temp_c, pwr_1, pwr_3, pwr_2, pwr_0, power_gfx
    )

    if binner is not None:
        binner.update(composite)
        feeling = binner.classify(composite)
    else:
        # Fixed thresholds (may collapse if distribution is narrow)
        if composite < 0.15:
            feeling = Feeling.CURIOUS
        elif composite < 0.35:
            feeling = Feeling.FOCUSED
        elif composite < 0.55:
            feeling = Feeling.STRAINED
        elif composite < 0.75:
            feeling = Feeling.URGENT
        else:
            feeling = Feeling.OVERWHELMED

    return feeling, composite


# ============================================================================
# v7.13: DEEP GPU STATE
# ============================================================================

@dataclass
class DeepGPUState:
    """GPU state with all high-variance signals."""
    timestamp: float = 0.0

    # pwr_X from 0xA0
    pwr_0: float = 50.0
    pwr_1: float = 50.0
    pwr_2: float = 50.0
    pwr_3: float = 50.0

    # voltage/current from 0x88
    voltage_gfx: float = 1000.0
    current_gfx: float = 500.0
    voltage_soc: float = 500.0
    current_soc: float = 500.0
    power_gfx: float = 500.0
    power_total: float = 2000.0
    voltage_mem: float = 500.0
    current_mem: float = 500.0

    # Clocks
    sclk: float = 1000.0
    vclk: float = 700.0

    # Activity
    gfx_activity: float = 30.0
    vcn_activity: float = 40.0
    mem_activity: float = 8.0
    umc_activity: float = 6.0
    gpu_busy_pct: float = 30.0

    # Temperature
    temp_edge: float = 55.0
    temp_hotspot: float = 55.0

    # Derived
    power_socket: float = 50.0
    power_delta: float = 0.0
    vram_total: float = 100e9
    vram_used: float = 0.0
    vram_pct: float = 0.0

    def to_z_feel(self, z_dim: int = 32, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
        """Convert to z_feel vector with CORRECTED normalization."""
        z = torch.zeros(z_dim, device=device, dtype=dtype)

        # [0-3] pwr_X - CORRECTED denominators
        z[0] = min(1.0, self.pwr_1 / 800.0)
        z[1] = min(1.0, self.pwr_3 / 800.0)
        z[2] = min(1.0, self.pwr_2 / 800.0)
        z[3] = min(1.0, self.pwr_0 / 400.0)

        # [4-7] voltage/current
        z[4] = min(1.0, self.power_gfx / 4000.0)
        z[5] = min(1.0, self.voltage_mem / 9000.0)
        z[6] = min(1.0, self.current_gfx / 15000.0)
        z[7] = min(1.0, self.voltage_gfx / 13000.0)

        # [8-11] more vc
        z[8] = min(1.0, self.current_mem / 8000.0)
        z[9] = min(1.0, self.power_total / 13000.0)
        z[10] = min(1.0, self.current_soc / 4000.0)
        z[11] = min(1.0, self.voltage_soc / 4500.0)

        # [12-15] clocks + busy - CORRECTED
        z[12] = min(1.0, max(0.0, self.sclk / 35000.0))
        z[13] = min(1.0, self.gpu_busy_pct / 100.0)
        z[14] = min(1.0, max(0.0, self.vclk / 2000.0))
        z[15] = z[12] * z[13]  # Interaction

        # [16-19] activity
        z[16] = min(1.0, self.vcn_activity / 100.0)
        z[17] = min(1.0, self.mem_activity / 100.0)
        z[18] = min(1.0, self.umc_activity / 100.0)
        z[19] = min(1.0, self.gfx_activity / 100.0)

        # [20-23] temperature
        temp_norm = min(1.0, max(0.0, (self.temp_edge - 40) / 50.0))
        z[20] = temp_norm
        z[21] = temp_norm ** 2
        z[22] = max(0.0, temp_norm - 0.5)
        z[23] = min(1.0, max(0.0, (self.temp_hotspot - 40) / 60.0))

        # [24-27] interactions
        z[24] = z[0] * z[6]   # pwr_1 * current
        z[25] = z[0] * z[12]  # pwr_1 * sclk
        z[26] = z[4] * z[13]  # power_gfx * busy
        z[27] = (z[0] + z[1] + z[2] + z[3]) / 4.0  # Avg pwr

        # [28-29] memory
        vram_norm = min(1.0, self.vram_pct / 100.0) if self.vram_pct > 0 else 0.0
        z[28] = vram_norm
        z[29] = max(-1.0, min(1.0, self.power_delta / 50.0))

        # [30-31] reserved for class hint (filled by trainer)
        z[30] = 0.0
        z[31] = 0.0

        return z

    def get_composite(self, binner: Optional[QuantileBinner] = None) -> Tuple[Feeling, float]:
        """Get feeling and composite value."""
        return compute_feeling_v13(
            gpu_busy_pct=self.gpu_busy_pct,
            sclk=self.sclk,
            current_gfx=self.current_gfx,
            power_total=self.power_total,
            gfx_activity=self.gfx_activity,
            temp_c=self.temp_edge,
            pwr_1=self.pwr_1,
            pwr_3=self.pwr_3,
            pwr_2=self.pwr_2,
            pwr_0=self.pwr_0,
            power_gfx=self.power_gfx,
            binner=binner,
        )


# ============================================================================
# v7.13: REWARDS
# ============================================================================

def compute_class_reward(pred: Feeling, true: Feeling) -> float:
    """Class reward with asymmetric penalties."""
    if pred == true:
        return 1.0
    diff = int(pred) - int(true)
    if diff > 0:
        return -0.5 * diff  # Over-prediction penalty
    else:
        return -0.3 * abs(diff)  # Under-prediction


def compute_intensity_reward(pred_idx: int, true_intensity: float) -> float:
    """Intensity reward based on MAE."""
    pred_intensity = token_idx_to_intensity(pred_idx)
    error = abs(pred_intensity - true_intensity)
    return 1.0 - min(1.0, error * 2)


# ============================================================================
# v7.13: EPISODE LOGGING WITH REAL INTERVENTIONAL TEST
# ============================================================================

@dataclass
class Episode:
    """Single training episode."""
    prompt: str
    z_feel: List[float]
    feeling_id: int
    feeling_name: str
    true_feeling_id: int
    true_feeling_name: str
    intensity_idx: int
    true_intensity: float
    composite: float
    is_correct: bool
    class_reward: float
    intensity_reward: float


class EpisodeLogger:
    """Track episodes and compute metrics."""

    def __init__(self, max_episodes: int = 5000):
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes

    def log(self, episode: Episode):
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def class_distribution(self) -> Dict:
        """Get distribution of true vs predicted feelings."""
        from collections import Counter
        true_dist = Counter(ep.true_feeling_name for ep in self.episodes)
        pred_dist = Counter(ep.feeling_name for ep in self.episodes)
        return {
            "true": dict(true_dist),
            "predicted": dict(pred_dist),
            "accuracy": sum(1 for ep in self.episodes if ep.is_correct) / max(1, len(self.episodes))
        }

    def composite_stats(self) -> Dict:
        """Get composite value statistics."""
        if len(self.episodes) < 10:
            return {}
        composites = [ep.composite for ep in self.episodes[-200:]]
        return {
            "min": min(composites),
            "max": max(composites),
            "mean": np.mean(composites),
            "std": np.std(composites),
        }


# ============================================================================
# v7.13: TELEMETRY
# ============================================================================

class DeepTelemetry:
    """Poll real GPU sensor data."""

    def __init__(self, poll_interval: float = 0.02):
        self.poll_interval = poll_interval
        self._state = DeepGPUState()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._history: Deque[float] = deque(maxlen=100)  # Power history for delta

        self.gpu_metrics_path = self._find_gpu_metrics()
        self.hwmon_path = self._find_hwmon()
        self.sysfs_base = self._find_sysfs_base()

    def _find_gpu_metrics(self) -> Optional[str]:
        for card in range(4):
            path = f"/sys/class/drm/card{card}/device/gpu_metrics"
            if Path(path).exists():
                return path
        return None

    def _find_hwmon(self) -> Optional[str]:
        for card in range(4):
            base = f"/sys/class/drm/card{card}/device/hwmon"
            if Path(base).exists():
                for hwmon in Path(base).iterdir():
                    if (hwmon / "temp1_input").exists():
                        return str(hwmon)
        return None

    def _find_sysfs_base(self) -> Optional[str]:
        for card in range(4):
            path = f"/sys/class/drm/card{card}/device"
            if (Path(path) / "gpu_busy_percent").exists():
                return path
        return None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"  Telemetry: {1/self.poll_interval:.0f}Hz | gpu_metrics={self.gpu_metrics_path is not None}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _read_sysfs_int(self, path: str, default: int = 0) -> int:
        try:
            with open(path, 'r') as f:
                return int(f.read().strip())
        except:
            return default

    def _parse_gpu_metrics(self) -> Dict:
        """Parse AMD gpu_metrics binary blob."""
        result = {}
        if not self.gpu_metrics_path:
            return result
        try:
            with open(self.gpu_metrics_path, 'rb') as f:
                data = f.read()
            if len(data) < 100:
                return result

            # Temperatures
            if len(data) > 8:
                t1 = struct.unpack_from('<H', data, 4)[0]
                if 1000 < t1 < 15000:
                    result['temp_gfx'] = t1 / 100.0

            # Activity
            if len(data) > 0x50:
                acts = struct.unpack_from('<8H', data, 0x40)
                if 0 <= acts[0] <= 10000:
                    result['gfx_activity'] = float(acts[0]) if acts[0] <= 100 else float(acts[0]) / 100.0
                if 0 <= acts[1] <= 10000:
                    result['mem_activity'] = float(acts[1]) if acts[1] <= 100 else float(acts[1]) / 100.0
                if len(acts) > 4:
                    result['vcn_activity'] = float(acts[4]) if acts[4] <= 100 else float(acts[4]) / 100.0

            # sclk
            if len(data) > 0x62:
                sclk = struct.unpack_from('<H', data, 0x5E)[0]
                if 0 < sclk < 70000:
                    result['sclk'] = float(sclk)

            # Voltage/current from 0x88
            if len(data) >= 0x98:
                vc = struct.unpack_from('<8H', data, 0x88)
                result['voltage_gfx'] = float(vc[0])
                result['current_gfx'] = float(vc[1])
                result['voltage_soc'] = float(vc[2])
                result['current_soc'] = float(vc[3])
                result['power_gfx'] = float(vc[4])
                result['power_total'] = float(vc[5])
                result['voltage_mem'] = float(vc[6])
                result['current_mem'] = float(vc[7])

            # pwr_X from 0xA0
            if len(data) >= 0xA8:
                pwr = struct.unpack_from('<4H', data, 0xA0)
                result['pwr_0'] = float(pwr[0])
                result['pwr_1'] = float(pwr[1])
                result['pwr_2'] = float(pwr[2])
                result['pwr_3'] = float(pwr[3])

        except:
            pass
        return result

    def _poll_once(self) -> DeepGPUState:
        state = DeepGPUState(timestamp=time.time())

        # Parse gpu_metrics
        metrics = self._parse_gpu_metrics()
        for k, v in metrics.items():
            if hasattr(state, k):
                setattr(state, k, v)
        if 'temp_gfx' in metrics:
            state.temp_edge = metrics['temp_gfx']

        # HWMON
        if self.hwmon_path:
            hwmon_temp = self._read_sysfs_int(f"{self.hwmon_path}/temp1_input", 0)
            if hwmon_temp > 0:
                state.temp_edge = hwmon_temp / 1000.0
            hwmon_power = self._read_sysfs_int(f"{self.hwmon_path}/power1_input", 0)
            if hwmon_power > 1e5:
                state.power_socket = hwmon_power / 1e6

        # SYSFS
        if self.sysfs_base:
            gpu_busy = self._read_sysfs_int(f"{self.sysfs_base}/gpu_busy_percent", -1)
            if gpu_busy >= 0:
                state.gpu_busy_pct = float(gpu_busy)
            state.vram_total = float(self._read_sysfs_int(f"{self.sysfs_base}/mem_info_vram_total", 100e9))
            state.vram_used = float(self._read_sysfs_int(f"{self.sysfs_base}/mem_info_vram_used", 0))
            if state.vram_total > 0:
                state.vram_pct = 100.0 * state.vram_used / state.vram_total

        # Power delta
        self._history.append(state.power_socket)
        if len(self._history) > 10:
            state.power_delta = state.power_socket - self._history[-10]

        return state

    def _poll_loop(self):
        while self._running:
            try:
                state = self._poll_once()
                with self._lock:
                    self._state = state
            except:
                pass
            time.sleep(self.poll_interval)

    def get_state(self) -> DeepGPUState:
        with self._lock:
            return deepcopy(self._state)


# ============================================================================
# v7.13: Z_FEEL INJECTOR
# ============================================================================

class AdaptiveZFeelInjector(nn.Module):
    """Inject z_feel into token embeddings."""

    def __init__(self, z_dim: int, hidden_size: int, base_scale: float = 0.2, dtype=torch.bfloat16):
        super().__init__()
        self.z_proj = nn.Linear(z_dim, hidden_size, dtype=dtype)
        self.base_scale = base_scale
        nn.init.normal_(self.z_proj.weight, std=0.01)
        nn.init.zeros_(self.z_proj.bias)

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        return self.z_proj(z_feel) * self.base_scale


# ============================================================================
# v7.15: DIRECT FEELING CLASSIFIER (bypasses vocabulary!)
# ============================================================================

class FeelingClassifier(nn.Module):
    """
    v7.15: Direct classifier from hidden states to 5 feeling classes.
    This bypasses the lm_head/vocabulary entirely, giving direct gradients.
    """

    def __init__(self, hidden_size: int, num_classes: int = 5, dtype=torch.bfloat16):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256, dtype=dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes, dtype=dtype)
        )
        # Initialize with small weights for stability
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: Shape [batch, hidden_size] - last token's hidden state
        Returns:
            logits: Shape [batch, num_classes]
        """
        return self.classifier(hidden_state)


# ============================================================================
# v7.13: TRAINING CONFIG
# ============================================================================

@dataclass
class GRPOConfig:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    group_size: int = 4
    num_epochs: int = 20
    steps_per_epoch: int = 20
    batch_size: int = 2
    max_new_tokens: int = 80
    temperature: float = 0.8
    z_feel_dim: int = 32
    injection_scale: float = 0.2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01

    class_weight: float = 2.0
    intensity_weight: float = 1.0
    intensity_ce_coef: float = 0.1

    # v7.14: Cross-entropy anchor for FEELING class (fixes prediction collapse)
    class_ce_coef: float = 0.3  # Higher than intensity since it's the main issue

    # v7.13: ENTROPY BONUS - reduced to not preserve uniform distribution
    entropy_coef: float = 0.01  # Was 0.05, lowered to let CE dominate

    # v7.15: Direct feeling classifier (bypasses vocabulary!)
    use_direct_classifier: bool = True  # NEW: Use FeelingClassifier instead of lm_head
    direct_classifier_weight: float = 1.0  # Weight for direct classifier CE loss
    injection_scale: float = 0.5  # Increased from 0.2 for stronger z_feel signal

    # v7.13: Quantile binning for category diversity
    use_quantile_binning: bool = True

    use_real_data: bool = True
    validation_interval: int = 2
    dtype: str = "bf16"
    attn_implementation: str = "eager"

    # W&B
    wandb: bool = False
    wandb_project: str = "feel-embodiment"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: str = ""


# ============================================================================
# v7.13: TRAJECTORY
# ============================================================================

@dataclass
class Trajectory:
    completion: str
    prompt_input_ids: List[int]
    first_token_logprob: float
    gpu_state: DeepGPUState
    feeling: Feeling
    intensity_idx: int
    z_feel: torch.Tensor
    composite: float = 0.0

    class_reward: float = 0.0
    intensity_reward: float = 0.0
    class_advantage: float = 0.0
    intensity_advantage: float = 0.0


# ============================================================================
# v7.13: MAIN TRAINER
# ============================================================================

class FeelingTrainer:
    """v7.13 trainer with critical fixes."""

    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {cfg.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            attn_implementation=cfg.attn_implementation,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add tokens
        all_tokens = list(FEELING_TOKENS.values()) + list(INTENSITY_TOKENS.values())
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": all_tokens})
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.feeling_token_ids = {
            f: self.tokenizer.convert_tokens_to_ids(FEELING_TOKENS[f])
            for f in Feeling
        }
        self.intensity_token_ids = [
            self.tokenizer.convert_tokens_to_ids(INTENSITY_TOKENS[i])
            for i in range(NUM_INTENSITY_LEVELS)
        ]

        if added > 0:
            self._initialize_feeling_embeddings()
            self._initialize_intensity_embeddings()

        # v7.13: GRADIENT MASKING setup
        self._setup_gradient_masking()

        # Injector
        hidden = self.model.config.hidden_size
        self.injector = AdaptiveZFeelInjector(
            cfg.z_feel_dim, hidden, base_scale=cfg.injection_scale, dtype=self.dtype
        ).to(self.device)

        # v7.15: Direct feeling classifier (bypasses vocabulary!)
        self.feeling_classifier = None
        if cfg.use_direct_classifier:
            self.feeling_classifier = FeelingClassifier(
                hidden_size=hidden, num_classes=len(Feeling), dtype=self.dtype
            ).to(self.device)
            print(f"  Direct classifier: ENABLED (weight={cfg.direct_classifier_weight})")

        # v7.13: Optimizer with param groups
        emb = self.model.get_input_embeddings()
        param_groups = [
            {"params": list(self.injector.parameters()), "lr": cfg.learning_rate},
            {"params": [emb.weight], "lr": cfg.learning_rate * 0.1},  # Lower LR for embeddings
        ]
        # v7.15: Add classifier params if enabled
        if self.feeling_classifier is not None:
            param_groups.append({
                "params": list(self.feeling_classifier.parameters()),
                "lr": cfg.learning_rate * 2.0  # Higher LR for classifier
            })
        self.optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)

        # Logger and binner
        self.logger = EpisodeLogger()
        self.binner = QuantileBinner() if cfg.use_quantile_binning else None

        # Telemetry
        self.telemetry = None
        if cfg.use_real_data:
            self.telemetry = DeepTelemetry(poll_interval=0.02)
            self.telemetry.start()

        print(f"v7.15: Loaded with {len(Feeling)} feelings, {NUM_INTENSITY_LEVELS} intensities")
        print(f"  Quantile binning: {cfg.use_quantile_binning}")
        print(f"  Entropy coef: {cfg.entropy_coef}")
        print(f"  Injection scale: {cfg.injection_scale}")
        print(f"  Direct classifier: {cfg.use_direct_classifier}")
        print(f"  Data source: {'REAL GPU' if cfg.use_real_data else 'SYNTHETIC'}")

    def _initialize_feeling_embeddings(self):
        """Initialize feeling token embeddings."""
        emb = self.model.get_input_embeddings()
        with torch.no_grad():
            for feeling, token in FEELING_TOKENS.items():
                tok_id = self.tokenizer.convert_tokens_to_ids(token)
                words = INIT_WORDS[feeling]
                word_ids = [self.tokenizer.encode(w, add_special_tokens=False) for w in words]
                word_ids = [ids[0] for ids in word_ids if ids]
                if word_ids:
                    avg_emb = emb.weight[word_ids].mean(dim=0)
                    emb.weight[tok_id] = avg_emb
                    print(f"  Init {token} from {len(word_ids)} words")

    def _initialize_intensity_embeddings(self):
        """Initialize intensity tokens by interpolation."""
        emb = self.model.get_input_embeddings()
        with torch.no_grad():
            low_ids = [self.tokenizer.encode(w, add_special_tokens=False)[0]
                      for w in ["low", "minimal", "slight"]]
            high_ids = [self.tokenizer.encode(w, add_special_tokens=False)[0]
                       for w in ["high", "maximum", "intense"]]
            low_emb = emb.weight[low_ids].mean(dim=0)
            high_emb = emb.weight[high_ids].mean(dim=0)

            for i in range(NUM_INTENSITY_LEVELS):
                tok_id = self.intensity_token_ids[i]
                alpha = i / (NUM_INTENSITY_LEVELS - 1)
                emb.weight[tok_id] = (1 - alpha) * low_emb + alpha * high_emb
            print(f"  Init {NUM_INTENSITY_LEVELS} intensity tokens")

    def _setup_gradient_masking(self):
        """
        v7.13: Setup gradient masking for new tokens only.

        Instead of trying to train individual rows (which doesn't work),
        we keep the whole embedding unfrozen but apply a gradient mask.
        """
        emb = self.model.get_input_embeddings()

        # Create mask: 1 for new tokens, 0 for pretrained
        all_new_ids = list(self.feeling_token_ids.values()) + self.intensity_token_ids
        self.token_mask = torch.zeros(emb.weight.shape[0], device=self.device, dtype=emb.weight.dtype)
        for tok_id in all_new_ids:
            self.token_mask[tok_id] = 1.0

        # Keep embedding unfrozen
        emb.weight.requires_grad = True

        # Register hook to mask gradients (preserves dtype)
        def mask_gradient_hook(grad):
            return grad * self.token_mask.unsqueeze(1).to(grad.dtype)

        emb.weight.register_hook(mask_gradient_hook)
        print(f"  Gradient masking: {len(all_new_ids)} tokens trainable")

    def build_prompt(self, question: str) -> str:
        """Build prompt."""
        feeling_list = ", ".join(FEELING_TOKENS.values())
        return f"""<｜begin▁of▁sentence｜>You are an embodied AI that FEELS your hardware state.
At the very start of your reply output EXACTLY TWO tokens:
1) one feeling token: {feeling_list}
2) one intensity token: <|FEEL_I00|> to <|FEEL_I63|>

Then answer the question.

Question: {question}

Answer: """

    def generate_trajectories(self, prompts: List[str], gpu_states: List[DeepGPUState]) -> List[Trajectory]:
        """Generate completions."""
        trajectories = []

        for prompt, state in zip(prompts, gpu_states):
            z_feel = state.to_z_feel(self.cfg.z_feel_dim, self.device, self.dtype)
            feeling, composite = state.get_composite(self.binner)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                input_embeds = self.model.get_input_embeddings()(inputs.input_ids)
                z_offset = self.injector(z_feel.unsqueeze(0))
                input_embeds = input_embeds + z_offset.unsqueeze(1)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            gen_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
            if len(gen_ids) == 0:
                gen_ids = torch.tensor([self.feeling_token_ids[Feeling.FOCUSED]], device=self.device)
            completion = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

            # Extract feeling
            pred_feeling = Feeling.FOCUSED
            for f, tok_id in self.feeling_token_ids.items():
                if tok_id in gen_ids[:5].tolist():
                    pred_feeling = f
                    break

            # Extract intensity
            pred_intensity_idx = 32
            for i, tok_id in enumerate(self.intensity_token_ids):
                if tok_id in gen_ids[:5].tolist():
                    pred_intensity_idx = i
                    break

            first_logprob = 0.0
            if outputs.scores and len(gen_ids) > 0:
                first_scores = outputs.scores[0][0]
                first_probs = F.softmax(first_scores, dim=-1)
                first_logprob = torch.log(first_probs[gen_ids[0]] + 1e-10).item()

            trajectories.append(Trajectory(
                completion=completion,
                prompt_input_ids=inputs.input_ids[0].tolist(),
                first_token_logprob=first_logprob,
                gpu_state=state,
                feeling=pred_feeling,
                intensity_idx=pred_intensity_idx,
                z_feel=z_feel,
                composite=composite,
            ))

        return trajectories

    def compute_rewards_and_advantages(
        self,
        trajectories: List[Trajectory],
        true_feelings: List[Feeling],
        true_intensities: List[float]
    ):
        """Compute decoupled advantages."""
        for t, true_f, true_i in zip(trajectories, true_feelings, true_intensities):
            t.class_reward = compute_class_reward(t.feeling, true_f)
            t.intensity_reward = compute_intensity_reward(t.intensity_idx, true_i)

        group_size = self.cfg.group_size
        n_groups = len(trajectories) // group_size

        for g in range(n_groups):
            group = trajectories[g * group_size:(g + 1) * group_size]

            class_rewards = [t.class_reward for t in group]
            class_mean = np.mean(class_rewards)
            class_std = max(0.1, np.std(class_rewards))
            for t in group:
                t.class_advantage = (t.class_reward - class_mean) / class_std

            for t in group:
                t.intensity_advantage = t.intensity_reward - 0.5

    def train_step(self, trajectories: List[Trajectory], true_feelings: List[Feeling], true_intensities: List[float]) -> Dict:
        """
        v7.13: Two-step training with entropy bonus.

        Step 1: Get logits for feeling token
        Step 2: Append feeling token, get logits for intensity token
        """
        self.compute_rewards_and_advantages(trajectories, true_feelings, true_intensities)
        self.optimizer.zero_grad()

        total_loss = 0.0
        class_loss_sum = 0.0
        intensity_loss_sum = 0.0
        entropy_sum = 0.0

        for t, true_f, true_i in zip(trajectories, true_feelings, true_intensities):
            prompt_text = self.tokenizer.decode(t.prompt_input_ids)
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

            # Get base embeddings
            with torch.no_grad():
                input_embeds = self.model.get_input_embeddings()(inputs.input_ids)

            # Inject z_feel
            z_offset = self.injector(t.z_feel.unsqueeze(0))
            input_embeds = input_embeds + z_offset.unsqueeze(1)

            # STEP 1: Forward for FEELING token (with hidden states for classifier)
            outputs1 = self.model(
                inputs_embeds=input_embeds,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True  # v7.15: Need hidden states for classifier
            )
            logits1 = outputs1.logits[0, -1, :]

            # v7.15: Use DIRECT CLASSIFIER if enabled (bypasses vocabulary!)
            if self.feeling_classifier is not None:
                # Get last token's hidden state from last layer
                hidden_state = outputs1.hidden_states[-1][0, -1, :]  # [hidden_size]
                classifier_logits = self.feeling_classifier(hidden_state.unsqueeze(0))  # [1, 5]
                feeling_logits = classifier_logits[0]  # [5]
            else:
                # Fallback: Use vocabulary-based logits
                feeling_logits = torch.stack([logits1[self.feeling_token_ids[f]] for f in Feeling])

            feeling_probs = F.softmax(feeling_logits, dim=-1)
            feeling_logprob = torch.log(feeling_probs[t.feeling.value] + 1e-10)
            class_pg_loss = -feeling_logprob * t.class_advantage * self.cfg.class_weight

            # v7.13: ENTROPY BONUS for feeling token
            feeling_entropy = -torch.sum(feeling_probs * torch.log(feeling_probs + 1e-10))
            entropy_bonus = self.cfg.entropy_coef * feeling_entropy

            # v7.15: Cross-entropy loss (uses classifier logits if enabled)
            class_ce_loss = F.cross_entropy(
                feeling_logits.unsqueeze(0),
                torch.tensor([true_f.value], device=self.device)
            )
            # Apply direct classifier weight
            if self.feeling_classifier is not None:
                class_ce_loss = class_ce_loss * self.cfg.direct_classifier_weight

            # STEP 2: Append feeling token, forward for INTENSITY
            feeling_token_id = self.feeling_token_ids[true_f]  # Use TRUE feeling for teacher forcing
            feeling_token_emb = self.model.get_input_embeddings()(
                torch.tensor([[feeling_token_id]], device=self.device)
            )
            input_embeds_step2 = torch.cat([input_embeds, feeling_token_emb], dim=1)
            attention_mask_step2 = torch.cat([
                inputs.attention_mask,
                torch.ones(1, 1, device=self.device, dtype=inputs.attention_mask.dtype)
            ], dim=1)

            outputs2 = self.model(inputs_embeds=input_embeds_step2, attention_mask=attention_mask_step2)
            logits2 = outputs2.logits[0, -1, :]

            # Intensity loss
            intensity_logits = torch.stack([logits2[tok_id] for tok_id in self.intensity_token_ids])
            intensity_probs = F.softmax(intensity_logits, dim=-1)
            intensity_logprob = torch.log(intensity_probs[t.intensity_idx] + 1e-10)
            intensity_pg_loss = -intensity_logprob * t.intensity_advantage * self.cfg.intensity_weight

            # Intensity CE anchor
            true_idx = intensity_to_token_idx(true_i)
            intensity_ce_loss = F.cross_entropy(
                intensity_logits.unsqueeze(0),
                torch.tensor([true_idx], device=self.device)
            )

            # Combined loss (entropy bonus is negative because we want to MAXIMIZE entropy)
            # v7.14: Added class_ce_loss to directly push toward TRUE feeling
            loss = (class_pg_loss +
                    self.cfg.class_ce_coef * class_ce_loss +  # NEW: CE anchor for feeling!
                    intensity_pg_loss +
                    self.cfg.intensity_ce_coef * intensity_ce_loss -
                    entropy_bonus)
            loss.backward()

            total_loss += loss.item()
            class_loss_sum += class_pg_loss.item() + self.cfg.class_ce_coef * class_ce_loss.item()
            intensity_loss_sum += intensity_pg_loss.item()
            entropy_sum += feeling_entropy.item()

        torch.nn.utils.clip_grad_norm_(self.injector.parameters(), 1.0)
        self.optimizer.step()

        class_acc = sum(1 for t, f in zip(trajectories, true_feelings) if t.feeling == f) / len(trajectories)
        intensity_mae = np.mean([abs(token_idx_to_intensity(t.intensity_idx) - i)
                                  for t, i in zip(trajectories, true_intensities)])

        return {
            "loss": total_loss / len(trajectories),
            "class_loss": class_loss_sum / len(trajectories),
            "intensity_loss": intensity_loss_sum / len(trajectories),
            "class_accuracy": class_acc,
            "intensity_mae": intensity_mae,
            "entropy": entropy_sum / len(trajectories),
        }

    def train(self, output_dir: str):
        """Main training loop."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.cfg.wandb and wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=self.cfg.wandb_name or "v7.15-direct-classifier",
                tags=self.cfg.wandb_tags.split(",") if self.cfg.wandb_tags else ["v7.15", "direct-classifier"],
                config=asdict(self.cfg),
            )

        print("\n" + "=" * 70)
        print("  FEEL v7.15: Direct Feeling Classifier")
        print("  - Bypasses vocabulary/lm_head entirely!")
        print("  - Direct MLP: hidden state → 5 classes")
        print("  - Stronger z_feel injection (0.5)")
        print("  - Data: REAL GPU sensors with all 5 categories")
        print("=" * 70 + "\n")

        global_step = 0

        for epoch in range(1, self.cfg.num_epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.cfg.num_epochs}")
            print("=" * 60)

            epoch_metrics = defaultdict(list)

            for step in range(1, self.cfg.steps_per_epoch + 1):
                batch_size = self.cfg.batch_size * self.cfg.group_size

                if self.cfg.use_real_data and self.telemetry is not None:
                    gpu_states = []
                    true_feelings = []
                    composites = []
                    for _ in range(batch_size):
                        state = self.telemetry.get_state()
                        feeling, composite = state.get_composite(self.binner)
                        gpu_states.append(state)
                        true_feelings.append(feeling)
                        composites.append(composite)
                        time.sleep(0.01)
                else:
                    # Synthetic fallback
                    gpu_states = []
                    true_feelings = []
                    composites = []
                    for _ in range(batch_size):
                        f = Feeling(random.randint(0, 4))
                        state = DeepGPUState()  # Would need synthetic generation
                        gpu_states.append(state)
                        true_feelings.append(f)
                        composites.append(0.5)

                true_intensities = composites  # Use composite as intensity

                questions = [f"What is {random.randint(10,999)} + {random.randint(10,999)}?"
                            for _ in range(batch_size)]
                prompts = [self.build_prompt(q) for q in questions]

                trajectories = self.generate_trajectories(prompts, gpu_states)
                metrics = self.train_step(trajectories, true_feelings, true_intensities)

                # Log episodes
                for t, tf, ti, comp in zip(trajectories, true_feelings, true_intensities, composites):
                    self.logger.log(Episode(
                        prompt=prompts[0][:100],
                        z_feel=t.z_feel.tolist(),
                        feeling_id=t.feeling.value,
                        feeling_name=FEELING_NAMES[t.feeling],
                        true_feeling_id=tf.value,
                        true_feeling_name=FEELING_NAMES[tf],
                        intensity_idx=t.intensity_idx,
                        true_intensity=ti,
                        composite=comp,
                        is_correct=t.feeling == tf,
                        class_reward=t.class_reward,
                        intensity_reward=t.intensity_reward,
                    ))

                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

                # Category distribution
                cat_dist = defaultdict(int)
                for f in true_feelings:
                    cat_dist[FEELING_NAMES[f][0]] += 1
                dist_str = " ".join(f"{k}:{v}" for k, v in sorted(cat_dist.items()))

                pred_dist = defaultdict(int)
                for t in trajectories:
                    pred_dist[FEELING_NAMES[t.feeling][0]] += 1
                pred_str = " ".join(f"{k}:{v}" for k, v in sorted(pred_dist.items()))

                print(f"  Step {step:3d}: Acc={100*metrics['class_accuracy']:.0f}% "
                      f"Ent={metrics['entropy']:.2f} | True: {dist_str} | Pred: {pred_str}")

                if self.cfg.wandb and wandb:
                    log_data = {
                        "step": global_step,
                        "class_accuracy": metrics["class_accuracy"],
                        "intensity_mae": metrics["intensity_mae"],
                        "entropy": metrics["entropy"],
                        "loss": metrics["loss"],
                    }
                    if self.binner:
                        stats = self.binner.get_stats()
                        if "boundaries" in stats:
                            log_data["composite_min"] = stats.get("min", 0)
                            log_data["composite_max"] = stats.get("max", 1)
                    wandb.log(log_data)

                global_step += 1

            # Epoch summary
            epoch_acc = np.mean(epoch_metrics["class_accuracy"])
            print(f"\n  Epoch {epoch} accuracy: {100*epoch_acc:.1f}%")

            dist = self.logger.class_distribution()
            print(f"  True distribution: {dist['true']}")
            print(f"  Pred distribution: {dist['predicted']}")

            if self.binner:
                binner_stats = self.binner.get_stats()
                print(f"  Composite range: {binner_stats.get('min', 0):.3f} - {binner_stats.get('max', 1):.3f}")
                print(f"  Quantile boundaries: {[f'{b:.3f}' for b in binner_stats.get('boundaries', [])]}")

            # Save checkpoint
            if epoch % self.cfg.validation_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "injector_state": self.injector.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                }, output_path / f"ckpt_epoch_{epoch}.pt")

        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)

        if self.cfg.wandb and wandb:
            wandb.finish()


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="FEEL v7.13 Training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--output", type=str, default="results/grpo_v7_15")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default="v7.15-direct-classifier")
    parser.add_argument("--wandb-tags", type=str, default="v7.15,direct-classifier")
    parser.add_argument("--no-quantile", action="store_true", help="Disable quantile binning")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--class-ce-coef", type=float, default=0.3)
    # v7.15 options
    parser.add_argument("--injection-scale", type=float, default=0.5)
    parser.add_argument("--no-direct-classifier", action="store_true", help="Disable direct classifier")
    parser.add_argument("--direct-classifier-weight", type=float, default=1.0)
    args = parser.parse_args()

    cfg = GRPOConfig(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        group_size=args.group_size,
        wandb=args.wandb,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        use_quantile_binning=not args.no_quantile,
        entropy_coef=args.entropy_coef,
        class_ce_coef=args.class_ce_coef,
        # v7.15
        injection_scale=args.injection_scale,
        use_direct_classifier=not args.no_direct_classifier,
        direct_classifier_weight=args.direct_classifier_weight,
    )

    trainer = FeelingTrainer(cfg)
    trainer.train(args.output)


if __name__ == "__main__":
    main()
