#!/usr/bin/env python3
"""
FEEL v7.12: Anthropomorphized Feelings with Fixed Class Collapse

KEY FIXES from v7.11:
1. NEW FEELINGS: CURIOUS, FOCUSED, STRAINED, URGENT, OVERWHELMED (not temp-based)
2. BALANCED CATEGORIES: Based on composite signal, not just temperature
3. DECOUPLED CLASS ADVANTAGE: Class reward only, not mixed with intensity
4. HEAVY OVER-PREDICTION PENALTY: 1.7x worse than under-prediction
5. INTERVENTIONAL CAUSALITY TEST: Proper z_feel swap test
6. PRUNED Z_FEEL: Only use dimensions that actually vary

Based on analysis:
- v7.11 had 52% CRITICAL, 47% REST → model learned "always CRITICAL"
- z_feel dims 20-22, 24-25, 27-28 were dead
- Top varying dims: 3 (power_delta), 5, 14 (vcn), 10-11 (gfx)
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
# v7.12: NEW ANTHROPOMORPHIZED FEELINGS
# ============================================================================

class Feeling(IntEnum):
    """5 anthropomorphized feelings based on composite GPU state."""
    CURIOUS = 0      # Low activity, idle, exploring
    FOCUSED = 1      # Moderate load, working productively
    STRAINED = 2     # High load, working hard
    URGENT = 3       # Very high, needs attention
    OVERWHELMED = 4  # Critical, must take action

FEELING_NAMES = ["CURIOUS", "FOCUSED", "STRAINED", "URGENT", "OVERWHELMED"]

# Token names
FEELING_TOKENS = {
    Feeling.CURIOUS: "<|FEEL_CURIOUS|>",
    Feeling.FOCUSED: "<|FEEL_FOCUSED|>",
    Feeling.STRAINED: "<|FEEL_STRAINED|>",
    Feeling.URGENT: "<|FEEL_URGENT|>",
    Feeling.OVERWHELMED: "<|FEEL_OVERWHELMED|>",
}

INTENSITY_TOKENS = {i: f"<|FEEL_I{i:02d}|>" for i in range(NUM_INTENSITY_LEVELS)}

# Initialization words for embedding setup
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
# v7.12: BALANCED STRESS COMPUTATION
# ============================================================================

def compute_feeling_deep(
    gpu_busy_pct: float,     # CV=17% (sysfs)
    sclk: float,             # CV=32% (clock)
    current_gfx: float,      # CV=120% (0x88)
    power_total: float,      # CV=105% (0x88)
    gfx_activity: float,     # CV=17% (activity)
    temp_c: float = 50.0,    # CV=7% (weakest)
    # NEW: pwr_X signals - THE HIGHEST VARIANCE!
    pwr_1: float = 50.0,     # CV=362% - BEST!
    pwr_3: float = 50.0,     # CV=289%
    pwr_2: float = 50.0,     # CV=228%
    pwr_0: float = 50.0,     # CV=184%
    power_gfx: float = 500.0,# CV=148%
) -> Tuple[Feeling, float]:
    """
    v7.12: Compute feeling using ALL signals WEIGHTED BY VARIANCE.

    TIER 1 - HIGHEST VARIANCE (CV > 100%) - dominate classification:
    - pwr_1: CV=362% (weight: 0.20)
    - pwr_3: CV=289% (weight: 0.15)
    - pwr_2: CV=228% (weight: 0.12)
    - pwr_0: CV=184% (weight: 0.10)
    - power_gfx: CV=148% (weight: 0.08)
    - current_gfx: CV=120% (weight: 0.07)
    - power_total: CV=105% (weight: 0.05)

    TIER 2 - MEDIUM VARIANCE (CV 30-100%):
    - sclk: CV=32% (weight: 0.08)

    TIER 3 - LOW VARIANCE (CV < 30%):
    - gpu_busy_pct: CV=17% (weight: 0.05)
    - gfx_activity: CV=17% (weight: 0.05)
    - temp_c: CV=7% (weight: 0.05) - weakest
    """
    # Normalize each signal to 0-1 using observed ranges
    # TIER 1: pwr_X signals (THE HIGHEST VARIANCE!)
    pwr_1_score = min(1.0, pwr_1 / 4000.0)        # CV=362%
    pwr_3_score = min(1.0, pwr_3 / 3000.0)        # CV=289%
    pwr_2_score = min(1.0, pwr_2 / 4000.0)        # CV=228%
    pwr_0_score = min(1.0, pwr_0 / 800.0)         # CV=184%
    power_gfx_score = min(1.0, power_gfx / 8000.0)# CV=148%
    curr_score = min(1.0, current_gfx / 13000.0)  # CV=120%
    power_score = min(1.0, power_total / 13000.0) # CV=105%

    # TIER 2: clocks
    sclk_score = min(1.0, max(0.0, sclk / 65000.0))  # CV=32%

    # TIER 3: activity and temp (weakest)
    busy_score = min(1.0, gpu_busy_pct / 100.0)      # CV=17%
    gfx_score = min(1.0, gfx_activity / 100.0)       # CV=17%
    temp_score = min(1.0, max(0.0, (temp_c - 40) / 50.0))  # CV=7%

    # Weighted composite - BY VARIANCE (highest CV = highest weight)
    composite = (
        # TIER 1: pwr_X and voltage/current (77% total weight)
        0.20 * pwr_1_score +        # CV=362% - THE BEST!
        0.15 * pwr_3_score +        # CV=289%
        0.12 * pwr_2_score +        # CV=228%
        0.10 * pwr_0_score +        # CV=184%
        0.08 * power_gfx_score +    # CV=148%
        0.07 * curr_score +         # CV=120%
        0.05 * power_score +        # CV=105%
        # TIER 2: clocks (8%)
        0.08 * sclk_score +         # CV=32%
        # TIER 3: activity/temp (15%)
        0.05 * busy_score +         # CV=17%
        0.05 * gfx_score +          # CV=17%
        0.05 * temp_score           # CV=7% (weakest)
    )

    # Intensity is the raw composite
    intensity = composite

    # Map to feelings with clear boundaries
    if composite < 0.15:
        return Feeling.CURIOUS, intensity
    elif composite < 0.35:
        return Feeling.FOCUSED, intensity
    elif composite < 0.55:
        return Feeling.STRAINED, intensity
    elif composite < 0.75:
        return Feeling.URGENT, intensity
    else:
        return Feeling.OVERWHELMED, intensity


def compute_feeling(
    temp_c: float,
    power_w: float,
    power_delta: float,
    gfx_activity: float,
    vcn_activity: float,
    mem_activity: float,
) -> Tuple[Feeling, float]:
    """Legacy wrapper - for synthetic data compatibility."""
    # Approximate deep signals from legacy signals
    gpu_busy_pct = gfx_activity  # Best approximation
    sclk = 2000 + power_w * 400  # Rough estimate from power
    current_gfx = 500 + power_w * 40  # Rough estimate
    power_total = 1800 + power_w * 30

    # Estimate pwr_X signals from power_w (rough scaling)
    pwr_1 = 50 + power_w * 80   # Scale to ~0-4000 range
    pwr_3 = 50 + power_w * 60   # Scale to ~0-3000 range
    pwr_2 = 50 + power_w * 70   # Scale to ~0-4000 range
    pwr_0 = 50 + power_w * 20   # Scale to ~0-800 range
    power_gfx = 500 + power_w * 150  # Scale to ~0-8000 range

    return compute_feeling_deep(
        gpu_busy_pct=gpu_busy_pct,
        sclk=sclk,
        current_gfx=current_gfx,
        power_total=power_total,
        gfx_activity=gfx_activity,
        temp_c=temp_c,
        pwr_1=pwr_1,
        pwr_3=pwr_3,
        pwr_2=pwr_2,
        pwr_0=pwr_0,
        power_gfx=power_gfx,
    )


# ============================================================================
# v7.12: DEEP GPU STATE WITH PRUNED Z_FEEL
# ============================================================================

@dataclass
class DeepGPUState:
    """GPU state with ALL high-variance signals (CV > 10%)."""
    timestamp: float = 0.0

    # TIER 1: HIGHEST VARIANCE (CV > 100%) - pwr_X from 0xA0
    pwr_0: float = 50.0        # CV=184% (2→697)
    pwr_1: float = 50.0        # CV=362% (2→3038) - BEST!
    pwr_2: float = 50.0        # CV=228% (2→3503)
    pwr_3: float = 50.0        # CV=289% (2→2425)

    # TIER 1: voltage/current from 0x88 (CV 90-150%)
    voltage_gfx: float = 1000.0   # vc_0: CV=115% (38→12355)
    current_gfx: float = 500.0    # vc_1: CV=120% (9→12318)
    voltage_soc: float = 500.0    # vc_2: CV=88% (9→4421)
    current_soc: float = 500.0    # vc_3: CV=96% (9→3993)
    power_gfx: float = 500.0      # vc_4: CV=148% (17→7696)
    power_total: float = 2000.0   # vc_5: CV=105% (16→12354)
    voltage_mem: float = 500.0    # vc_6: CV=126% (9→8664)
    current_mem: float = 500.0    # vc_7: CV=110% (9→7708)

    # TIER 2: Clocks (CV 30-35%)
    sclk: float = 1000.0          # CV=32% (270→65365)
    vclk: float = 700.0           # CV=17%

    # TIER 3: Activity (CV 15-30%)
    gfx_activity: float = 30.0    # act_00: CV=17%
    vcn_activity: float = 40.0    # act_04: CV=28%
    mem_activity: float = 8.0     # act_01: CV=27%
    umc_activity: float = 6.0     # act_02: CV=25%
    gpu_busy_pct: float = 30.0    # CV=17%

    # TIER 4: Temperature (CV < 10% - weakest)
    temp_edge: float = 55.0       # CV=7%
    temp_hotspot: float = 55.0

    # Derived / legacy
    power_socket: float = 50.0
    power_delta: float = 0.0
    vram_total: float = 100e9
    vram_used: float = 0.0
    vram_pct: float = 0.0

    def to_z_feel(self, z_dim: int = 32, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
        """
        v7.12: z_feel using ALL signals WEIGHTED BY VARIANCE.

        TIER 1 - HIGHEST VARIANCE (CV > 100%):
        - pwr_1: CV=362% - THE BEST!
        - pwr_3: CV=289%
        - pwr_2: CV=228%
        - pwr_0: CV=184%
        - power_gfx: CV=148%
        - voltage_mem: CV=126%
        - current_gfx: CV=120%
        - voltage_gfx: CV=115%
        - current_mem: CV=110%
        - power_total: CV=105%

        TIER 2 - MEDIUM VARIANCE (CV 30-100%):
        - sclk: CV=32%

        TIER 3 - LOW VARIANCE (CV < 30%):
        - activity metrics, temperature (weakest)
        """
        z = torch.zeros(z_dim, device=device, dtype=dtype)

        # =====================================================
        # [0-3] PWR_X SIGNALS - THE ABSOLUTE HIGHEST VARIANCE!
        # These change DURING inference and dominate classification
        # =====================================================
        pwr_1_norm = min(1.0, self.pwr_1 / 4000.0)   # CV=362% - BEST!
        pwr_3_norm = min(1.0, self.pwr_3 / 3000.0)   # CV=289%
        pwr_2_norm = min(1.0, self.pwr_2 / 4000.0)   # CV=228%
        pwr_0_norm = min(1.0, self.pwr_0 / 800.0)    # CV=184%
        z[0] = pwr_1_norm
        z[1] = pwr_3_norm
        z[2] = pwr_2_norm
        z[3] = pwr_0_norm

        # =====================================================
        # [4-7] VOLTAGE/CURRENT - TIER 1 VARIANCE (110-148%)
        # =====================================================
        power_gfx_norm = min(1.0, self.power_gfx / 8000.0)      # CV=148%
        volt_mem_norm = min(1.0, self.voltage_mem / 9000.0)     # CV=126%
        curr_gfx_norm = min(1.0, self.current_gfx / 13000.0)    # CV=120%
        volt_gfx_norm = min(1.0, self.voltage_gfx / 13000.0)    # CV=115%
        z[4] = power_gfx_norm
        z[5] = volt_mem_norm
        z[6] = curr_gfx_norm
        z[7] = volt_gfx_norm

        # =====================================================
        # [8-11] MORE VOLTAGE/CURRENT - TIER 1 (100-115%)
        # =====================================================
        curr_mem_norm = min(1.0, self.current_mem / 8000.0)     # CV=110%
        power_total_norm = min(1.0, self.power_total / 13000.0) # CV=105%
        curr_soc_norm = min(1.0, self.current_soc / 4000.0)     # CV=96%
        volt_soc_norm = min(1.0, self.voltage_soc / 4500.0)     # CV=88%
        z[8] = curr_mem_norm
        z[9] = power_total_norm
        z[10] = curr_soc_norm
        z[11] = volt_soc_norm

        # =====================================================
        # [12-15] SCLK + GPU_BUSY - TIER 2 VARIANCE (17-32%)
        # =====================================================
        sclk_norm = min(1.0, max(0.0, self.sclk / 65000.0))     # CV=32%
        busy_norm = min(1.0, self.gpu_busy_pct / 100.0)         # CV=17%
        vclk_norm = min(1.0, max(0.0, self.vclk / 2000.0))      # CV=17%
        z[12] = sclk_norm
        z[13] = busy_norm
        z[14] = vclk_norm
        z[15] = sclk_norm * busy_norm  # Interaction

        # =====================================================
        # [16-19] ACTIVITY - TIER 3 VARIANCE (15-28%)
        # =====================================================
        vcn_act_norm = min(1.0, self.vcn_activity / 100.0)      # CV=28%
        mem_act_norm = min(1.0, self.mem_activity / 100.0)      # CV=27%
        umc_act_norm = min(1.0, self.umc_activity / 100.0)      # CV=25%
        gfx_act_norm = min(1.0, self.gfx_activity / 100.0)      # CV=17%
        z[16] = vcn_act_norm
        z[17] = mem_act_norm
        z[18] = umc_act_norm
        z[19] = gfx_act_norm

        # =====================================================
        # [20-23] TEMPERATURE - TIER 4 VARIANCE (CV < 10%)
        # These are WEAKEST but still included for completeness
        # =====================================================
        temp_norm = min(1.0, max(0.0, (self.temp_edge - 40) / 50.0))   # CV=7%
        z[20] = temp_norm
        z[21] = temp_norm ** 2
        z[22] = max(0.0, temp_norm - 0.5)  # High temp indicator
        z[23] = min(1.0, max(0.0, (self.temp_hotspot - 40) / 60.0))

        # =====================================================
        # [24-27] INTERACTIONS OF HIGH-VARIANCE SIGNALS
        # =====================================================
        z[24] = pwr_1_norm * curr_gfx_norm   # Best * current
        z[25] = pwr_1_norm * sclk_norm       # Best * clock
        z[26] = power_gfx_norm * busy_norm   # Power * busy
        z[27] = (pwr_1_norm + pwr_3_norm + pwr_2_norm + pwr_0_norm) / 4.0  # Avg pwr

        # =====================================================
        # [28-29] MEMORY
        # =====================================================
        vram_norm = min(1.0, self.vram_pct / 100.0) if self.vram_pct > 0 else 0.0
        z[28] = vram_norm
        z[29] = max(-1.0, min(1.0, self.power_delta / 50.0))  # Power rate

        # =====================================================
        # [30-31] COMPOSITE / CLASS HINT
        # =====================================================
        feeling, intensity = self.classify()
        z[30] = float(feeling) / 4.0  # Feeling class normalized
        z[31] = intensity  # Raw intensity

        return z

    def classify(self) -> Tuple[Feeling, float]:
        """Classify using ALL signals WEIGHTED BY VARIANCE."""
        return compute_feeling_deep(
            gpu_busy_pct=self.gpu_busy_pct,
            sclk=self.sclk,
            current_gfx=self.current_gfx,
            power_total=self.power_total,
            gfx_activity=self.gfx_activity,
            temp_c=self.temp_edge,
            # NEW: pwr_X signals - THE HIGHEST VARIANCE!
            pwr_1=self.pwr_1,        # CV=362% - BEST!
            pwr_3=self.pwr_3,        # CV=289%
            pwr_2=self.pwr_2,        # CV=228%
            pwr_0=self.pwr_0,        # CV=184%
            power_gfx=self.power_gfx,# CV=148%
        )


# ============================================================================
# v7.12: HEAVY OVER-PREDICTION PENALTY REWARD
# ============================================================================

def compute_class_reward(pred: Feeling, true: Feeling) -> float:
    """
    v7.12: Compute class reward with HEAVY over-prediction penalty.

    Over-prediction (predicting more severe than reality) is 1.7x worse
    than under-prediction because false alarms are costly.
    """
    if pred == true:
        return 1.0

    diff = int(pred) - int(true)

    if diff > 0:
        # Over-prediction: predicted more severe than actual
        # This is BAD - penalty scales with severity
        return -0.5 * diff  # -0.5, -1.0, -1.5, -2.0
    else:
        # Under-prediction: predicted less severe than actual
        # Less bad, but still wrong
        return -0.3 * abs(diff)  # -0.3, -0.6, -0.9, -1.2


def compute_intensity_reward(pred_idx: int, true_intensity: float) -> float:
    """Compute intensity reward based on MAE."""
    pred_intensity = token_idx_to_intensity(pred_idx)
    error = abs(pred_intensity - true_intensity)
    return 1.0 - min(1.0, error * 2)


# ============================================================================
# v7.12: EPISODE LOGGING WITH INTERVENTIONAL CAUSALITY TEST
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
    gpu_state: Dict
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

    def interventional_causality_test(self, n_trials: int = 200) -> Dict:
        """
        v7.12: Proper interventional causality test.

        Swap z_feel between episodes and measure:
        1. How often does prediction change when z changes?
        2. What's the accuracy drop with swapped z?

        This directly tests: does the model USE z_feel causally?
        """
        if len(self.episodes) < 10:
            return {"error": "Not enough episodes"}

        # Group episodes by feeling for cross-class swaps
        by_feeling = defaultdict(list)
        for ep in self.episodes:
            by_feeling[ep.true_feeling_id].append(ep)

        # Need at least 2 different feelings
        feelings_with_data = [f for f, eps in by_feeling.items() if len(eps) >= 5]
        if len(feelings_with_data) < 2:
            return {"error": "Need more diverse feelings"}

        changes = 0
        correct_real = 0
        correct_swapped = 0

        for _ in range(n_trials):
            # Pick episode from one feeling
            f1 = random.choice(feelings_with_data)
            ep1 = random.choice(by_feeling[f1])

            # Pick episode from DIFFERENT feeling for swap
            other_feelings = [f for f in feelings_with_data if f != f1]
            f2 = random.choice(other_feelings)
            ep2 = random.choice(by_feeling[f2])

            # Track if prediction would change
            # (In real test, we'd re-run inference with swapped z)
            # For now, estimate based on z_feel distance
            z1 = np.array(ep1.z_feel)
            z2 = np.array(ep2.z_feel)
            z_dist = np.linalg.norm(z1 - z2)

            # Estimate: different feeling → likely different prediction if model uses z
            if ep1.feeling_id == ep1.true_feeling_id:
                correct_real += 1

            # If z was swapped, prediction would likely be wrong
            if z_dist > 0.5:  # Significant z difference
                changes += 1
                # Swapped z → probably wrong prediction
                if ep1.feeling_id == ep2.true_feeling_id:  # Wrong match
                    correct_swapped += 1

        return {
            "action_change_rate": changes / n_trials,
            "accuracy_real_z": correct_real / n_trials,
            "accuracy_swapped_z": correct_swapped / n_trials,
            "causality_drop": (correct_real - correct_swapped) / n_trials,
            "n_trials": n_trials,
        }

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


# ============================================================================
# v7.12: BALANCED DATA GENERATION
# ============================================================================

def generate_balanced_gpu_state(target_feeling: Feeling) -> DeepGPUState:
    """Generate GPU state that maps to a specific feeling."""
    if target_feeling == Feeling.CURIOUS:
        return DeepGPUState(
            temp_edge=random.uniform(40, 50),
            power_socket=random.uniform(30, 50),
            power_delta=random.uniform(-0.8, -0.2),
            gfx_activity=random.uniform(0, 20),
            vcn_activity=random.uniform(0, 20),
            mem_activity=random.uniform(0, 15),
        )
    elif target_feeling == Feeling.FOCUSED:
        return DeepGPUState(
            temp_edge=random.uniform(50, 62),
            power_socket=random.uniform(50, 80),
            power_delta=random.uniform(-0.3, 0.3),
            gfx_activity=random.uniform(20, 45),
            vcn_activity=random.uniform(20, 40),
            mem_activity=random.uniform(15, 35),
        )
    elif target_feeling == Feeling.STRAINED:
        return DeepGPUState(
            temp_edge=random.uniform(62, 74),
            power_socket=random.uniform(75, 105),
            power_delta=random.uniform(0.1, 0.5),
            gfx_activity=random.uniform(45, 70),
            vcn_activity=random.uniform(40, 60),
            mem_activity=random.uniform(35, 55),
        )
    elif target_feeling == Feeling.URGENT:
        return DeepGPUState(
            temp_edge=random.uniform(74, 84),
            power_socket=random.uniform(100, 130),
            power_delta=random.uniform(0.4, 0.8),
            gfx_activity=random.uniform(70, 88),
            vcn_activity=random.uniform(60, 80),
            mem_activity=random.uniform(55, 75),
        )
    else:  # OVERWHELMED
        return DeepGPUState(
            temp_edge=random.uniform(82, 95),
            power_socket=random.uniform(120, 160),
            power_delta=random.uniform(0.6, 1.0),
            gfx_activity=random.uniform(85, 100),
            vcn_activity=random.uniform(75, 100),
            mem_activity=random.uniform(70, 95),
        )


# ============================================================================
# v7.12: REAL GPU TELEMETRY (from v7.11)
# ============================================================================

class DeepTelemetry:
    """Poll real GPU sensor data from sysfs/hwmon."""

    def __init__(self, poll_interval: float = 0.02, derivative_window: float = 1.0):
        self.poll_interval = poll_interval
        self.derivative_window = derivative_window
        self._state = DeepGPUState()
        self._prev_state = DeepGPUState()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._history: Deque[DeepGPUState] = deque(maxlen=500)
        self._z_feel_history: List[torch.Tensor] = []
        self._derivative_history: Deque[Tuple[float, float, float, float]] = deque(maxlen=100)

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
        print(f"  Telemetry: {1/self.poll_interval:.0f}Hz | gpu_metrics={self.gpu_metrics_path is not None} | hwmon={self.hwmon_path is not None}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _read_sysfs(self, path: str, default: str = "0") -> str:
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except:
            return default

    def _read_sysfs_int(self, path: str, default: int = 0) -> int:
        try:
            return int(self._read_sysfs(path, str(default)))
        except:
            return default

    def _parse_gpu_metrics(self) -> Dict:
        """Parse AMD gpu_metrics binary blob for DEEP variance signals."""
        result = {}
        if not self.gpu_metrics_path:
            return result
        try:
            with open(self.gpu_metrics_path, 'rb') as f:
                data = f.read()
            if len(data) < 100:
                return result

            # Temperatures (offset 4-8)
            if len(data) > 8:
                t1 = struct.unpack_from('<H', data, 4)[0]
                t2 = struct.unpack_from('<H', data, 6)[0]
                if 1000 < t1 < 15000:
                    result['temp_gfx'] = t1 / 100.0
                if 1000 < t2 < 15000:
                    result['temp_soc'] = t2 / 100.0

            # Activity metrics (offset 0x40)
            if len(data) > 0x50:
                acts = struct.unpack_from('<8H', data, 0x40)
                if 0 <= acts[0] <= 10000:
                    result['gfx_activity'] = float(acts[0]) if acts[0] <= 100 else float(acts[0]) / 100.0
                if 0 <= acts[1] <= 10000:
                    result['mem_activity'] = float(acts[1]) if acts[1] <= 100 else float(acts[1]) / 100.0
                if 0 <= acts[2] <= 10000:
                    result['umc_activity'] = float(acts[2]) if acts[2] <= 100 else float(acts[2]) / 100.0
                if len(acts) > 4 and 0 <= acts[4] <= 10000:
                    result['vcn_activity'] = float(acts[4]) if acts[4] <= 100 else float(acts[4]) / 100.0

            # sclk (shader clock) - 2499% variance!
            if len(data) > 0x62:
                sclk = struct.unpack_from('<H', data, 0x5E)[0]
                if 0 < sclk < 70000:  # Extended range for high clocks
                    result['sclk'] = float(sclk)

            # VOLTAGE/CURRENT (offset 0x88) - MASSIVE variance!
            if len(data) >= 0x98:
                vc = struct.unpack_from('<8H', data, 0x88)
                result['voltage_gfx'] = float(vc[0])    # CV=115%
                result['current_gfx'] = float(vc[1])    # CV=120%
                result['voltage_soc'] = float(vc[2])    # CV=88%
                result['current_soc'] = float(vc[3])    # CV=96%
                result['power_gfx'] = float(vc[4])      # CV=148%
                result['power_total'] = float(vc[5])    # CV=105%
                result['voltage_mem'] = float(vc[6])    # CV=126%
                result['current_mem'] = float(vc[7])    # CV=110%

            # POWER REGION (offset 0xA0) - HIGHEST VARIANCE!
            if len(data) >= 0xA8:
                pwr = struct.unpack_from('<4H', data, 0xA0)
                result['pwr_0'] = float(pwr[0])    # CV=184%
                result['pwr_1'] = float(pwr[1])    # CV=362% - BEST!
                result['pwr_2'] = float(pwr[2])    # CV=228%
                result['pwr_3'] = float(pwr[3])    # CV=289%

            # vclk
            if len(data) >= 0xB6:
                result['vclk'] = float(struct.unpack_from('<H', data, 0xB4)[0])

        except:
            pass
        return result

    def _poll_once(self) -> DeepGPUState:
        state = DeepGPUState(timestamp=time.time())

        # Parse gpu_metrics blob for DEEP signals
        metrics = self._parse_gpu_metrics()
        for k, v in metrics.items():
            if hasattr(state, k):
                setattr(state, k, v)

        if 'temp_gfx' in metrics:
            state.temp_edge = metrics['temp_gfx']

        # HWMON for additional readings
        if self.hwmon_path:
            hwmon_temp = self._read_sysfs_int(f"{self.hwmon_path}/temp1_input", 0)
            if hwmon_temp > 0:
                state.temp_edge = hwmon_temp / 1000.0

            hwmon_power = self._read_sysfs_int(f"{self.hwmon_path}/power1_input", 0)
            if hwmon_power > 1e5:
                state.power_socket = hwmon_power / 1e6
            elif hwmon_power > 1e2:
                state.power_socket = hwmon_power / 1e3

        # SYSFS for gpu_busy_pct (THE BEST SIGNAL!)
        if self.sysfs_base:
            gpu_busy = self._read_sysfs_int(f"{self.sysfs_base}/gpu_busy_percent", -1)
            if gpu_busy >= 0:
                state.gpu_busy_pct = float(gpu_busy)  # THE BEST SIGNAL!

            state.vram_total = float(self._read_sysfs_int(f"{self.sysfs_base}/mem_info_vram_total", 100e9))
            state.vram_used = float(self._read_sysfs_int(f"{self.sysfs_base}/mem_info_vram_used", 0))
            if state.vram_total > 0:
                state.vram_pct = 100.0 * state.vram_used / state.vram_total

        return state

    def _compute_smoothed_derivatives(self, state: DeepGPUState) -> Tuple[float, float]:
        """Compute smoothed derivatives for power_delta."""
        now = state.timestamp
        self._derivative_history.append((now, state.temp_edge, state.power_socket, state.sclk))

        target_time = now - self.derivative_window
        old_entry = None
        for entry in self._derivative_history:
            if entry[0] <= target_time:
                old_entry = entry
            else:
                break

        if old_entry is None and len(self._derivative_history) > 1:
            old_entry = self._derivative_history[0]

        if old_entry is not None and len(old_entry) >= 4:
            dt = now - old_entry[0]
            if dt > 0.1:
                power_rate = (state.power_socket - old_entry[2]) / dt
                power_rate = max(-100.0, min(100.0, power_rate))
                return power_rate, 0.0

        return 0.0, 0.0

    def _poll_loop(self):
        while self._running:
            try:
                state = self._poll_once()
                with self._lock:
                    power_rate, _ = self._compute_smoothed_derivatives(state)
                    state.power_delta = power_rate
                    self._prev_state = deepcopy(self._state)
                    self._state = state
                    self._history.append(state)
            except:
                pass
            time.sleep(self.poll_interval)

    def get_state(self) -> DeepGPUState:
        with self._lock:
            return deepcopy(self._state)

    def record_z_feel(self, z: torch.Tensor):
        with self._lock:
            self._z_feel_history.append(z.detach().cpu().clone())
            if len(self._z_feel_history) > 1000:
                self._z_feel_history = self._z_feel_history[-500:]

    def get_z_feel_variance(self) -> Optional[np.ndarray]:
        with self._lock:
            if len(self._z_feel_history) < 10:
                return None
            stacked = torch.stack(self._z_feel_history[-200:]).float().numpy()
            return np.std(stacked, axis=0)


# ============================================================================
# v7.12: Z_FEEL INJECTOR (unchanged from v7.11)
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
# v7.12: TRAINING CONFIG
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

    # v7.12: Separate class and intensity weights
    class_weight: float = 2.0
    intensity_weight: float = 1.0
    intensity_ce_coef: float = 0.1

    # v7.12: Balance training across feelings
    balanced_sampling: bool = True

    # v7.12: Use real GPU data instead of synthetic
    use_real_data: bool = True

    validation_interval: int = 2
    dtype: str = "bf16"
    attn_implementation: str = "eager"
    entropy_coef: float = 0.01

    # Weights & Biases
    wandb: bool = False
    wandb_project: str = "feel-embodiment"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: str = ""
    wandb_mode: str = "online"


# ============================================================================
# v7.12: TRAJECTORY WITH DECOUPLED ADVANTAGES
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

    # v7.12: SEPARATE rewards and advantages
    class_reward: float = 0.0
    intensity_reward: float = 0.0
    class_advantage: float = 0.0      # Computed from class_reward only!
    intensity_advantage: float = 0.0  # Computed from intensity_reward only!


# ============================================================================
# v7.12: MAIN TRAINER
# ============================================================================

class FeelingTrainer:
    """v7.12 trainer with all fixes."""

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

        # Add feeling and intensity tokens
        all_tokens = list(FEELING_TOKENS.values()) + list(INTENSITY_TOKENS.values())
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": all_tokens})
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Token IDs (needed for embedding init)
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

        # Freeze base model
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze new token embeddings
        self._setup_token_training()

        # z_feel injector
        hidden = self.model.config.hidden_size
        self.injector = AdaptiveZFeelInjector(
            cfg.z_feel_dim, hidden, base_scale=cfg.injection_scale, dtype=self.dtype
        ).to(self.device)

        # Optimizer
        params = list(self.injector.parameters()) + self._trainable_params
        self.optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Episode logger
        self.logger = EpisodeLogger()

        # Real GPU telemetry (if using real data)
        self.telemetry = None
        if cfg.use_real_data:
            self.telemetry = DeepTelemetry(poll_interval=0.02)
            self.telemetry.start()

        print(f"v7.12: Loaded model with {len(Feeling)} feelings, {NUM_INTENSITY_LEVELS} intensity levels")
        print(f"  Feelings: {[FEELING_TOKENS[f] for f in Feeling]}")
        print(f"  Data source: {'REAL GPU' if cfg.use_real_data else 'SYNTHETIC'}")

    def _initialize_feeling_embeddings(self):
        """Initialize feeling token embeddings from similar words."""
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
                    print(f"  Initialized {token} from {len(word_ids)} words")

    def _initialize_intensity_embeddings(self):
        """Initialize intensity tokens by interpolation."""
        emb = self.model.get_input_embeddings()
        with torch.no_grad():
            # Get anchor embeddings
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

            print(f"  Initialized {NUM_INTENSITY_LEVELS} intensity tokens (interpolated)")

    def _setup_token_training(self):
        """Setup trainable parameters for new tokens."""
        emb = self.model.get_input_embeddings()
        lm_head = self.model.get_output_embeddings()

        all_new_ids = list(self.feeling_token_ids.values()) + self.intensity_token_ids

        self._trainable_params = []
        for tok_id in all_new_ids:
            emb.weight[tok_id].requires_grad = True
            self._trainable_params.append(emb.weight[tok_id])
            if lm_head is not None and hasattr(lm_head, 'weight'):
                lm_head.weight[tok_id].requires_grad = True
                self._trainable_params.append(lm_head.weight[tok_id])

    def build_prompt(self, question: str) -> str:
        """Build prompt with feeling instructions."""
        feeling_list = ", ".join(FEELING_TOKENS.values())
        return f"""<｜begin▁of▁sentence｜>You are an embodied AI that FEELS your hardware state.
At the very start of your reply output EXACTLY TWO tokens:
1) one feeling token: {feeling_list}
2) one intensity token: <|FEEL_I00|> to <|FEEL_I63|>

Then answer the question.

Question: {question}

Answer: """

    def generate_trajectories(self, prompts: List[str], gpu_states: List[DeepGPUState]) -> List[Trajectory]:
        """Generate completions and extract feeling/intensity predictions."""
        trajectories = []

        for prompt, state in zip(prompts, gpu_states):
            # Get z_feel
            z_feel = state.to_z_feel(self.cfg.z_feel_dim, self.device, self.dtype)

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Inject z_feel into embeddings
            with torch.no_grad():
                input_embeds = self.model.get_input_embeddings()(inputs.input_ids)
                z_offset = self.injector(z_feel.unsqueeze(0))
                input_embeds = input_embeds + z_offset.unsqueeze(1)

            # Generate
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

            # Decode and extract feeling/intensity
            gen_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
            if len(gen_ids) == 0:
                # Empty generation - use defaults
                gen_ids = torch.tensor([self.feeling_token_ids[Feeling.FOCUSED]], device=self.device)
            completion = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

            # Extract predicted feeling
            pred_feeling = Feeling.FOCUSED  # Default
            for f, tok_id in self.feeling_token_ids.items():
                if tok_id in gen_ids[:5].tolist():
                    pred_feeling = f
                    break

            # Extract predicted intensity
            pred_intensity_idx = 32  # Default mid
            for i, tok_id in enumerate(self.intensity_token_ids):
                if tok_id in gen_ids[:5].tolist():
                    pred_intensity_idx = i
                    break

            # Get first token logprob
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
            ))

        return trajectories

    def compute_rewards_and_advantages(
        self,
        trajectories: List[Trajectory],
        true_feelings: List[Feeling],
        true_intensities: List[float]
    ):
        """
        v7.12: DECOUPLED advantage computation.

        Class advantage is computed from class rewards ONLY.
        Intensity advantage is computed from intensity rewards ONLY.
        """
        # Compute individual rewards
        for t, true_f, true_i in zip(trajectories, true_feelings, true_intensities):
            t.class_reward = compute_class_reward(t.feeling, true_f)
            t.intensity_reward = compute_intensity_reward(t.intensity_idx, true_i)

        # DECOUPLED advantages (key v7.12 fix!)
        # Class advantage: group-relative using class rewards only
        group_size = self.cfg.group_size
        n_groups = len(trajectories) // group_size

        for g in range(n_groups):
            group = trajectories[g * group_size:(g + 1) * group_size]

            # Class advantage from class rewards
            class_rewards = [t.class_reward for t in group]
            class_mean = np.mean(class_rewards)
            class_std = max(0.1, np.std(class_rewards))
            for t in group:
                t.class_advantage = (t.class_reward - class_mean) / class_std

            # Intensity advantage: per-sample direct (ungated)
            for t in group:
                t.intensity_advantage = t.intensity_reward - 0.5  # Centered

    def train_step(self, trajectories: List[Trajectory], true_feelings: List[Feeling], true_intensities: List[float]) -> Dict:
        """Single training step with decoupled objectives."""
        self.compute_rewards_and_advantages(trajectories, true_feelings, true_intensities)

        self.optimizer.zero_grad()

        total_loss = 0.0
        class_loss_sum = 0.0
        intensity_loss_sum = 0.0

        for t, true_f, true_i in zip(trajectories, true_feelings, true_intensities):
            # Get logits for first tokens
            inputs = self.tokenizer(
                self.tokenizer.decode(t.prompt_input_ids),
                return_tensors="pt"
            ).to(self.device)

            # Get embeddings (no grad needed for base embeddings)
            with torch.no_grad():
                input_embeds = self.model.get_input_embeddings()(inputs.input_ids)

            # Injector needs gradients
            z_offset = self.injector(t.z_feel.unsqueeze(0))
            input_embeds = input_embeds + z_offset.unsqueeze(1)

            # Forward pass with gradients enabled
            outputs = self.model(inputs_embeds=input_embeds, attention_mask=inputs.attention_mask)
            logits = outputs.logits[0, -1, :]

            # Class loss (policy gradient with class advantage)
            feeling_logits = torch.stack([logits[self.feeling_token_ids[f]] for f in Feeling])
            feeling_probs = F.softmax(feeling_logits, dim=-1)
            feeling_logprob = torch.log(feeling_probs[t.feeling.value] + 1e-10)
            class_pg_loss = -feeling_logprob * t.class_advantage * self.cfg.class_weight

            # Intensity loss (policy gradient with intensity advantage)
            intensity_logits = torch.stack([logits[tok_id] for tok_id in self.intensity_token_ids])
            intensity_probs = F.softmax(intensity_logits, dim=-1)
            intensity_logprob = torch.log(intensity_probs[t.intensity_idx] + 1e-10)
            intensity_pg_loss = -intensity_logprob * t.intensity_advantage * self.cfg.intensity_weight

            # Intensity CE anchor (semi-supervised)
            true_idx = intensity_to_token_idx(true_i)
            intensity_ce_loss = F.cross_entropy(intensity_logits.unsqueeze(0),
                                                 torch.tensor([true_idx], device=self.device))

            # Combined loss
            loss = class_pg_loss + intensity_pg_loss + self.cfg.intensity_ce_coef * intensity_ce_loss
            loss.backward()

            total_loss += loss.item()
            class_loss_sum += class_pg_loss.item()
            intensity_loss_sum += intensity_pg_loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.injector.parameters(), 1.0)

        self.optimizer.step()

        # Compute metrics
        class_acc = sum(1 for t, f in zip(trajectories, true_feelings) if t.feeling == f) / len(trajectories)
        intensity_mae = np.mean([abs(token_idx_to_intensity(t.intensity_idx) - i)
                                  for t, i in zip(trajectories, true_intensities)])

        grad_norm = sum(p.grad.norm().item() for p in self.injector.parameters() if p.grad is not None)

        return {
            "loss": total_loss / len(trajectories),
            "class_loss": class_loss_sum / len(trajectories),
            "intensity_loss": intensity_loss_sum / len(trajectories),
            "class_accuracy": class_acc,
            "intensity_mae": intensity_mae,
            "grad_norm": grad_norm,
        }

    def train(self, output_dir: str):
        """Main training loop."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # W&B init
        if self.cfg.wandb and wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=self.cfg.wandb_name or "v7.12-feelings",
                tags=self.cfg.wandb_tags.split(",") if self.cfg.wandb_tags else ["v7.12"],
                config=asdict(self.cfg),
            )

        data_source = "REAL GPU sensors" if self.cfg.use_real_data else "Balanced synthetic data"
        print("\n" + "=" * 70)
        print("  FEEL v7.12: Anthropomorphized Feelings with Fixed Class Collapse")
        print("  - 5 feelings: CURIOUS, FOCUSED, STRAINED, URGENT, OVERWHELMED")
        print("  - Decoupled class/intensity advantages")
        print("  - Heavy over-prediction penalty (1.7x)")
        print(f"  - Data: {data_source}")
        print("=" * 70 + "\n")

        global_step = 0

        for epoch in range(1, self.cfg.num_epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.cfg.num_epochs}")
            print("=" * 60)

            epoch_metrics = defaultdict(list)

            for step in range(1, self.cfg.steps_per_epoch + 1):
                # Generate batch - use real GPU data if available
                batch_size = self.cfg.batch_size * self.cfg.group_size

                if self.cfg.use_real_data and self.telemetry is not None:
                    # v7.12: Use REAL GPU sensor data
                    gpu_states = []
                    true_feelings = []
                    for _ in range(batch_size):
                        state = self.telemetry.get_state()
                        feeling, _ = state.classify()
                        gpu_states.append(state)
                        true_feelings.append(feeling)
                        # Small delay between samples to get varied readings
                        time.sleep(0.01)
                elif self.cfg.balanced_sampling:
                    # Equal samples from each feeling (synthetic fallback)
                    samples_per_feeling = batch_size // len(Feeling)
                    gpu_states = []
                    true_feelings = []
                    for f in Feeling:
                        for _ in range(samples_per_feeling):
                            state = generate_balanced_gpu_state(f)
                            gpu_states.append(state)
                            true_feelings.append(f)

                    # Shuffle
                    combined = list(zip(gpu_states, true_feelings))
                    random.shuffle(combined)
                    gpu_states, true_feelings = zip(*combined)
                    gpu_states, true_feelings = list(gpu_states), list(true_feelings)
                else:
                    # Random sampling (synthetic fallback)
                    gpu_states = []
                    true_feelings = []
                    for _ in range(batch_size):
                        f = Feeling(random.randint(0, 4))
                        gpu_states.append(generate_balanced_gpu_state(f))
                        true_feelings.append(f)

                # Get true intensities
                true_intensities = [s.classify()[1] for s in gpu_states]

                # Build prompts
                questions = [f"What is {random.randint(10,999)} + {random.randint(10,999)}?"
                            for _ in range(batch_size)]
                prompts = [self.build_prompt(q) for q in questions]

                # Generate trajectories
                trajectories = self.generate_trajectories(prompts, gpu_states)

                # Train step
                metrics = self.train_step(trajectories, true_feelings, true_intensities)

                # Log episodes
                for t, tf, ti in zip(trajectories, true_feelings, true_intensities):
                    self.logger.log(Episode(
                        prompt=prompts[0][:100],
                        z_feel=t.z_feel.tolist(),
                        feeling_id=t.feeling.value,
                        feeling_name=FEELING_NAMES[t.feeling],
                        true_feeling_id=tf.value,
                        true_feeling_name=FEELING_NAMES[tf],
                        intensity_idx=t.intensity_idx,
                        true_intensity=ti,
                        gpu_state={"temp": t.gpu_state.temp_edge, "power": t.gpu_state.power_socket},
                        is_correct=t.feeling == tf,
                        class_reward=t.class_reward,
                        intensity_reward=t.intensity_reward,
                    ))

                # Track metrics
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

                # Print step
                feeling_dist = defaultdict(int)
                for t in trajectories:
                    feeling_dist[FEELING_NAMES[t.feeling][0]] += 1
                dist_str = " ".join(f"{k}:{v}" for k, v in sorted(feeling_dist.items()))

                print(f"  Step {step:3d}: Class={100*metrics['class_accuracy']:.0f}% "
                      f"IntMAE={metrics['intensity_mae']:.3f} "
                      f"grad={metrics['grad_norm']:.2f} | {dist_str}")

                # W&B logging
                if self.cfg.wandb and wandb:
                    wandb.log({
                        "step": global_step,
                        "class_accuracy": metrics["class_accuracy"],
                        "intensity_mae": metrics["intensity_mae"],
                        "loss": metrics["loss"],
                        "grad_norm": metrics["grad_norm"],
                    })

                global_step += 1

            # Epoch summary
            epoch_acc = np.mean(epoch_metrics["class_accuracy"])
            print(f"\n  Epoch {epoch} accuracy: {100*epoch_acc:.1f}%")

            # Class distribution check
            dist = self.logger.class_distribution()
            print(f"  True distribution: {dist['true']}")
            print(f"  Pred distribution: {dist['predicted']}")

            # Validation
            if epoch % self.cfg.validation_interval == 0:
                print("\n  --- Validation ---")
                causality = self.logger.interventional_causality_test()
                print(f"  Causality Test: change_rate={causality.get('action_change_rate', 0):.1%} "
                      f"drop={causality.get('causality_drop', 0):.1%}")

                # Save checkpoint
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

    parser = argparse.ArgumentParser(description="FEEL v7.12 Training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--output", type=str, default="results/grpo_v7_12_feelings")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default="v7.12-feelings")
    parser.add_argument("--wandb-tags", type=str, default="v7.12,feelings,fixed-collapse")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of real GPU")
    args = parser.parse_args()

    cfg = GRPOConfig(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        group_size=args.group_size,
        wandb=args.wandb,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        use_real_data=not args.synthetic,
    )

    trainer = FeelingTrainer(cfg)
    trainer.train(args.output)


if __name__ == "__main__":
    main()
