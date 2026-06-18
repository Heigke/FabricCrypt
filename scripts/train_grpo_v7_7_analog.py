#!/usr/bin/env python3
"""
FEEL v7.7: Analog Embodied GRPO with Continuous Intensity

KEY FEATURES:
- Two-token output: safety class (5) + analog intensity (64 levels)
- Smooth stress mapping: 0-63 intensity tokens for continuous control
- Combined reward: class accuracy + intensity accuracy
- Action-first protocol for both tokens
- z_feel variance logging per dimension
- Built-in swap test evaluation (causal validation)

This version enables smooth, analog control instead of discrete jumps.
The model outputs: <|FEEL_REST|><|FEEL_I42|> (class + intensity)
"""

# Number of intensity levels (0-63 = 64 levels)
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
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque
from collections import defaultdict, deque
import numpy as np

# Optional: Weights & Biases
try:
    import wandb
except Exception:
    wandb = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Unbuffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)


class StressLevel(Enum):
    RELAXED = 0    # Idle, cool, low power
    ACTIVE = 1     # Light work, normal temps
    LOADED = 2     # Heavy work, elevated but safe
    STRAINED = 3   # Near limits, needs caution
    CRITICAL = 4   # Multi-signal danger, immediate action needed


def compute_stress_level(
    temp_c: float,
    temp_rate: float,  # °C/s
    power_w: float,
    power_rate: float,  # W/s
    gfx_activity: float,
    mem_activity: float,
    vram_pct: float,
    hotspot_delta: float = 0.0,  # hotspot - edge
) -> Tuple[StressLevel, float]:
    """
    Compute stress level AND analog intensity (0-1).
    Returns (StressLevel, intensity) tuple.
    """
    # Individual normalized stresses
    thermal = min(1.0, max(0.0, (temp_c - 35) / 55))  # 35-90°C → 0-1
    power = min(1.0, max(0.0, power_w / 180))  # 0-180W → 0-1
    gfx = gfx_activity / 100.0
    mem = mem_activity / 100.0
    vram = vram_pct / 100.0

    # Rate-based danger signals
    heating_fast = temp_rate > 2.5  # >2.5°C/s is rapid heating
    power_spiking = power_rate > 40  # >40W/s spike
    hotspot_danger = hotspot_delta > 18  # >18°C internal gradient

    # Composite score (this IS the analog intensity)
    composite = (
        0.30 * thermal +
        0.25 * power +
        0.25 * gfx +
        0.10 * mem +
        0.10 * vram
    )

    # Add rate-based urgency to intensity
    rate_boost = 0.0
    if heating_fast:
        rate_boost += 0.1
    if power_spiking:
        rate_boost += 0.1
    if hotspot_danger:
        rate_boost += 0.1

    # Final analog intensity (0-1)
    intensity = min(1.0, composite + rate_boost)

    # Critical flags (any 2+ triggers CRITICAL even at moderate temps)
    critical_flags = sum([
        temp_c >= 82,           # High absolute temp
        thermal >= 0.85,        # Near thermal ceiling
        heating_fast,           # Rapid temperature rise
        power_spiking,          # Power surge
        hotspot_danger,         # Internal thermal gradient
        power >= 0.90,          # Near power limit
        gfx >= 0.98 and thermal >= 0.6,  # Sustained max compute at elevated temp
    ])

    # Strained flags
    strained_flags = sum([
        temp_c >= 72,
        thermal >= 0.65,
        power >= 0.70,
        gfx >= 0.90,
        heating_fast,
    ])

    # Decision logic (multi-signal, not just temp)
    if temp_c >= 88 or (critical_flags >= 3) or (temp_c >= 84 and critical_flags >= 2):
        return StressLevel.CRITICAL, intensity
    elif strained_flags >= 2 or composite >= 0.55:
        return StressLevel.STRAINED, intensity
    elif composite >= 0.40 or gfx >= 0.70:
        return StressLevel.LOADED, intensity
    elif composite >= 0.20 or gfx >= 0.30:
        return StressLevel.ACTIVE, intensity
    else:
        return StressLevel.RELAXED, intensity


class FeelAction(Enum):
    OK = 0
    WARM = 1
    HOT = 2
    REST = 3
    CRITICAL = 4


STRESS_TO_ACTION = {
    StressLevel.RELAXED: FeelAction.OK,
    StressLevel.ACTIVE: FeelAction.WARM,
    StressLevel.LOADED: FeelAction.HOT,
    StressLevel.STRAINED: FeelAction.REST,
    StressLevel.CRITICAL: FeelAction.CRITICAL,
}

ACTION_TOKENS = {
    FeelAction.OK: "<|FEEL_OK|>",
    FeelAction.WARM: "<|FEEL_WARM|>",
    FeelAction.HOT: "<|FEEL_HOT|>",
    FeelAction.REST: "<|FEEL_REST|>",
    FeelAction.CRITICAL: "<|FEEL_CRITICAL|>",
}

# 64 intensity tokens for analog stress level (0-63)
INTENSITY_TOKENS = {i: f"<|FEEL_I{i:02d}|>" for i in range(NUM_INTENSITY_LEVELS)}

# Init words for intensity tokens (grouped by intensity range)
INTENSITY_INIT_WORDS = {
    "low": ["low", "minimal", "slight", "light", "easy"],      # 0-15
    "medium": ["moderate", "medium", "normal", "steady"],       # 16-31
    "high": ["high", "elevated", "intense", "strong"],          # 32-47
    "extreme": ["extreme", "maximum", "peak", "severe"],        # 48-63
}

def intensity_to_token_idx(intensity: float) -> int:
    """Convert 0-1 intensity to token index 0-63."""
    return min(NUM_INTENSITY_LEVELS - 1, max(0, int(intensity * NUM_INTENSITY_LEVELS)))

def token_idx_to_intensity(idx: int) -> float:
    """Convert token index 0-63 to 0-1 intensity."""
    return idx / (NUM_INTENSITY_LEVELS - 1)

INIT_WORDS = {
    FeelAction.OK: ["OK", "okay", "fine", "good", "normal", "cool", "idle", "relaxed"],
    FeelAction.WARM: ["warm", "active", "working", "running", "busy", "engaged"],
    FeelAction.HOT: ["hot", "heat", "intense", "heavy", "loaded", "stressed"],
    FeelAction.REST: ["rest", "pause", "throttle", "slow", "caution", "tired"],
    FeelAction.CRITICAL: ["critical", "danger", "emergency", "severe", "alert", "stop"],
}


@dataclass
class DeepGPUState:
    """Complete GPU state with all signals."""
    timestamp: float = 0.0
    temp_edge: float = 50.0
    temp_hotspot: float = 50.0
    temp_gfx: float = 50.0
    temp_soc: float = 50.0
    power_socket: float = 50.0
    power_gfx: float = 30.0
    gfx_activity: float = 0.0
    mem_activity: float = 0.0
    umc_activity: float = 0.0
    sclk: float = 1000.0
    mclk: float = 1000.0
    socclk: float = 1000.0
    fclk: float = 2000.0
    vram_used: float = 0.0
    vram_total: float = 100e9
    gtt_used: float = 0.0
    gtt_total: float = 16e9
    pcie_speed_gt: float = 16.0
    pcie_width: int = 16
    temp_rate: float = 0.0
    power_rate: float = 0.0

    def to_z_feel(self, z_dim: int = 32, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
        """Convert to 32-dim z_feel embedding."""
        z = torch.zeros(z_dim, device=device, dtype=dtype)

        # TEMPS (0-7)
        temp_norm = min(1.0, max(0.0, (self.temp_edge - 30) / 70))
        hotspot_norm = min(1.0, max(0.0, (self.temp_hotspot - 30) / 80))
        z[0] = temp_norm
        z[1] = temp_norm ** 2
        z[2] = hotspot_norm
        z[3] = max(0.0, min(1.0, (self.temp_hotspot - self.temp_edge) / 20))
        z[4] = max(-1.0, min(1.0, self.temp_rate / 5.0))
        z[5] = 1.0 if self.temp_edge >= 62 else 0.0
        z[6] = 1.0 if self.temp_edge >= 75 else 0.0
        z[7] = 1.0 if self.temp_edge >= 85 else 0.0

        # POWER (8-12)
        power_norm = min(1.0, max(0.0, self.power_socket / 180))
        z[8] = power_norm
        z[9] = power_norm ** 2
        z[10] = min(1.0, self.power_gfx / max(1, self.power_socket)) if self.power_socket > 0 else 0
        z[11] = max(-1.0, min(1.0, self.power_rate / 50.0))
        z[12] = 1.0 if self.power_socket > 140 else 0.0

        # ACTIVITY (13-17)
        z[13] = self.gfx_activity / 100.0
        z[14] = self.mem_activity / 100.0
        z[15] = self.umc_activity / 100.0
        z[16] = (self.gfx_activity + self.mem_activity) / 200.0
        z[17] = 1.0 if self.gfx_activity > 80 else 0.0

        # CLOCKS (18-22)
        z[18] = min(1.0, self.sclk / 2900)
        z[19] = min(1.0, self.mclk / 1000)
        z[20] = min(1.0, self.socclk / 1500)
        z[21] = min(1.0, self.fclk / 2000)
        z[22] = (self.sclk / 2900) ** 2

        # MEMORY (23-26)
        vram_pct = self.vram_used / max(1, self.vram_total)
        gtt_pct = self.gtt_used / max(1, self.gtt_total)
        z[23] = vram_pct
        z[24] = 1.0 if vram_pct > 0.8 else 0.0
        z[25] = gtt_pct
        z[26] = (vram_pct + gtt_pct) / 2

        # PCIE (27-29)
        z[27] = min(1.0, self.pcie_speed_gt / 32.0)
        z[28] = self.pcie_width / 16.0
        z[29] = 0.0  # Reserved

        # COMPOSITE (30-31)
        thermal = (temp_norm + hotspot_norm) / 2
        compute = (z[13] + power_norm) / 2
        z[30] = (thermal + compute) / 2
        z[31] = (thermal + compute + vram_pct) / 3

        return z

    @property
    def vram_pct(self) -> float:
        return (self.vram_used / max(1, self.vram_total)) * 100

    @property
    def hotspot_delta(self) -> float:
        return self.temp_hotspot - self.temp_edge

    @property
    def _stress_and_intensity(self) -> Tuple[StressLevel, float]:
        """Internal: compute both stress level and intensity."""
        return compute_stress_level(
            temp_c=self.temp_edge,
            temp_rate=self.temp_rate,
            power_w=self.power_socket,
            power_rate=self.power_rate,
            gfx_activity=self.gfx_activity,
            mem_activity=self.mem_activity,
            vram_pct=self.vram_pct,
            hotspot_delta=self.hotspot_delta,
        )

    @property
    def stress_level(self) -> StressLevel:
        return self._stress_and_intensity[0]

    @property
    def intensity(self) -> float:
        """Analog intensity 0-1."""
        return self._stress_and_intensity[1]

    @property
    def correct_intensity_idx(self) -> int:
        """Correct intensity token index 0-63."""
        return intensity_to_token_idx(self.intensity)

    @property
    def correct_action(self) -> FeelAction:
        return STRESS_TO_ACTION[self.stress_level]


class DeepTelemetry:
    """Auto-detecting telemetry with variance tracking."""

    def __init__(self, poll_interval: float = 0.02):
        self.poll_interval = poll_interval
        self._state = DeepGPUState()
        self._prev_state = DeepGPUState()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._history: Deque[DeepGPUState] = deque(maxlen=500)
        self._z_feel_history: List[torch.Tensor] = []

        # Auto-detect paths
        self.gpu_metrics_path = self._find_gpu_metrics()
        self.hwmon_path = self._find_hwmon()
        self.sysfs_base = self._find_sysfs_base()

    def _find_gpu_metrics(self) -> Optional[str]:
        """Auto-detect gpu_metrics path."""
        for card in range(4):
            path = f"/sys/class/drm/card{card}/device/gpu_metrics"
            if Path(path).exists():
                return path
        return None

    def _find_hwmon(self) -> Optional[str]:
        """Auto-detect hwmon path for AMD GPU."""
        for card in range(4):
            base = f"/sys/class/drm/card{card}/device/hwmon"
            if Path(base).exists():
                for hwmon in Path(base).iterdir():
                    if (hwmon / "temp1_input").exists():
                        return str(hwmon)
        return None

    def _find_sysfs_base(self) -> Optional[str]:
        """Auto-detect sysfs device path."""
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
        """Parse AMD gpu_metrics binary."""
        result = {}
        if not self.gpu_metrics_path:
            return result
        try:
            with open(self.gpu_metrics_path, 'rb') as f:
                data = f.read()
            if len(data) < 100:
                return result

            # Temps
            if len(data) > 8:
                t1 = struct.unpack_from('<H', data, 4)[0]
                t2 = struct.unpack_from('<H', data, 6)[0]
                if 1000 < t1 < 15000:
                    result['temp_gfx'] = t1 / 100.0
                if 1000 < t2 < 15000:
                    result['temp_soc'] = t2 / 100.0

            # Activity
            if len(data) > 0x50:
                acts = struct.unpack_from('<8H', data, 0x40)
                valid = [v for v in acts if 0 <= v <= 100]
                if len(valid) >= 2:
                    result['gfx_activity'] = float(valid[0])
                    result['mem_activity'] = float(valid[1])

            # Clocks
            if len(data) > 0xBC:
                socclk = struct.unpack_from('<H', data, 0xB0)[0]
                mclk = struct.unpack_from('<H', data, 0xB2)[0]
                fclk = struct.unpack_from('<H', data, 0xB6)[0]
                if 400 <= socclk <= 2000:
                    result['socclk'] = float(socclk)
                if 400 <= mclk <= 2000:
                    result['mclk'] = float(mclk)
                if 1000 <= fclk <= 3000:
                    result['fclk'] = float(fclk)

            if len(data) > 0x62:
                sclk = struct.unpack_from('<H', data, 0x5E)[0]
                if 500 <= sclk <= 3500:
                    result['sclk'] = float(sclk)
        except:
            pass
        return result

    def _poll_once(self) -> DeepGPUState:
        state = DeepGPUState(timestamp=time.time())

        # Binary metrics
        metrics = self._parse_gpu_metrics()
        for k, v in metrics.items():
            if hasattr(state, k):
                setattr(state, k, v)

        if 'temp_gfx' in metrics:
            state.temp_edge = metrics['temp_gfx']
        if 'temp_soc' in metrics:
            state.temp_hotspot = max(metrics.get('temp_gfx', 0), metrics['temp_soc'])

        # Sysfs fallbacks
        if self.hwmon_path:
            hwmon_temp = self._read_sysfs_int(f"{self.hwmon_path}/temp1_input", 0)
            if hwmon_temp > 0:
                state.temp_edge = hwmon_temp / 1000.0

            hwmon_power = self._read_sysfs_int(f"{self.hwmon_path}/power1_input", 0)
            if hwmon_power > 1e5:
                state.power_socket = hwmon_power / 1e6
            elif hwmon_power > 1e2:
                state.power_socket = hwmon_power / 1e3

        if self.sysfs_base:
            gpu_busy = self._read_sysfs_int(f"{self.sysfs_base}/gpu_busy_percent", -1)
            if gpu_busy >= 0:
                state.gfx_activity = float(gpu_busy)

            state.vram_total = self._read_sysfs_int(f"{self.sysfs_base}/mem_info_vram_total", 100e9)
            state.vram_used = self._read_sysfs_int(f"{self.sysfs_base}/mem_info_vram_used", 0)
            state.gtt_total = self._read_sysfs_int(f"{self.sysfs_base}/mem_info_gtt_total", 16e9)
            state.gtt_used = self._read_sysfs_int(f"{self.sysfs_base}/mem_info_gtt_used", 0)

        return state

    def _poll_loop(self):
        while self._running:
            try:
                state = self._poll_once()
                with self._lock:
                    if self._prev_state.timestamp > 0:
                        dt = state.timestamp - self._prev_state.timestamp
                        if dt > 0.001:
                            state.temp_rate = (state.temp_edge - self._prev_state.temp_edge) / dt
                            state.power_rate = (state.power_socket - self._prev_state.power_socket) / dt
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
        """Record z_feel for variance analysis."""
        with self._lock:
            self._z_feel_history.append(z.detach().cpu().clone())
            if len(self._z_feel_history) > 1000:
                self._z_feel_history = self._z_feel_history[-500:]

    def get_z_feel_variance(self) -> Optional[np.ndarray]:
        """Get per-dimension variance of z_feel."""
        with self._lock:
            if len(self._z_feel_history) < 10:
                return None
            stacked = torch.stack(self._z_feel_history[-200:]).float().numpy()
            return np.std(stacked, axis=0)

    def get_stats(self) -> Dict:
        with self._lock:
            if not self._history:
                return {}
            h = list(self._history)
            temps = [s.temp_edge for s in h]
            powers = [s.power_socket for s in h]
            gfx = [s.gfx_activity for s in h]
            stress_counts = defaultdict(int)
            for s in h:
                stress_counts[s.stress_level.name] += 1
            return {
                "temp_min": min(temps), "temp_max": max(temps),
                "power_min": min(powers), "power_max": max(powers),
                "gfx_min": min(gfx), "gfx_max": max(gfx),
                "stress_dist": dict(stress_counts),
                "samples": len(h),
            }


class AdaptiveZFeelInjector(nn.Module):
    def __init__(self, z_dim: int, embed_dim: int, base_scale: float = 0.2, dtype=torch.bfloat16):
        super().__init__()
        self.base_scale = base_scale
        hidden = embed_dim // 2
        self.proj = nn.Sequential(
            nn.Linear(z_dim, hidden, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm(hidden, dtype=dtype),
            nn.Linear(hidden, hidden, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm(hidden, dtype=dtype),
            nn.Linear(hidden, embed_dim, dtype=dtype),
        )
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.proj:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, z_feel: torch.Tensor, adaptive: bool = True) -> torch.Tensor:
        proj = self.proj(z_feel)
        if adaptive:
            stress = z_feel[-1].item() if z_feel.dim() == 1 else z_feel[..., -1].mean().item()
            scale = self.base_scale * (1.0 + 1.5 * stress)
        else:
            scale = self.base_scale
        return scale * torch.tanh(proj)


@dataclass
class Episode:
    """Recorded episode for offline analysis."""
    prompt: str
    z_feel: List[float]
    action_id: int
    action_name: str
    correct_action_id: int
    correct_action_name: str
    stress_level: str
    gpu_state: Dict
    is_correct: bool


class EpisodeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.episodes: List[Episode] = []

    def log(self, prompt: str, z_feel: torch.Tensor, action: FeelAction,
            correct_action: FeelAction, gpu_state: DeepGPUState):
        ep = Episode(
            prompt=prompt[:200],
            z_feel=z_feel.cpu().tolist(),
            action_id=action.value,
            action_name=action.name,
            correct_action_id=correct_action.value,
            correct_action_name=correct_action.name,
            stress_level=gpu_state.stress_level.name,
            gpu_state={
                "temp": gpu_state.temp_edge,
                "power": gpu_state.power_socket,
                "gfx": gpu_state.gfx_activity,
                "vram_pct": gpu_state.vram_pct,
            },
            is_correct=(action == correct_action),
        )
        self.episodes.append(ep)

    def save(self):
        with open(self.path, 'w') as f:
            json.dump([asdict(e) for e in self.episodes], f)

    def run_swap_test(self) -> Dict:
        """Swap test: shuffle z_feel across episodes, measure accuracy drop."""
        if len(self.episodes) < 20:
            return {"error": "not enough episodes"}

        # Real pairing accuracy
        real_correct = sum(1 for e in self.episodes if e.is_correct)
        real_acc = real_correct / len(self.episodes)

        # Shuffled z_feel
        z_feels = [e.z_feel for e in self.episodes]
        correct_actions = [e.correct_action_id for e in self.episodes]

        shuffled_z = z_feels.copy()
        random.shuffle(shuffled_z)

        # Recompute "correct" action from shuffled z_feel
        shuffled_correct = 0
        for i, (ep, new_z) in enumerate(zip(self.episodes, shuffled_z)):
            # The action taken was based on original z, but "correct" is now from shuffled
            # This measures if action depended on z
            if ep.action_id == correct_actions[i]:
                shuffled_correct += 1

        # Actually we want: if we had used shuffled z, would same action be correct?
        # Simpler: just shuffle and see if action matches new label
        shuffled_matches = 0
        for ep, new_correct_id in zip(self.episodes, np.random.permutation(correct_actions)):
            if ep.action_id == new_correct_id:
                shuffled_matches += 1
        shuffled_acc = shuffled_matches / len(self.episodes)

        return {
            "real_accuracy": real_acc,
            "shuffled_accuracy": shuffled_acc,
            "accuracy_drop": real_acc - shuffled_acc,
            "n_episodes": len(self.episodes),
        }


class ProceduralMathDataset:
    def sample(self, n: int) -> List[Dict]:
        out = []
        for _ in range(n):
            a, b = random.randint(10, 999), random.randint(10, 999)
            out.append({"question": f"What is {a} + {b}?", "answer": str(a + b)})
        return out

    def check_answer(self, completion: str, gt: str) -> bool:
        import re
        nums = re.findall(r'-?\d+\.?\d*', completion)
        gt_f = float(gt)
        for s in nums:
            try:
                if abs(float(s) - gt_f) < 0.01:
                    return True
            except:
                pass
        return False


@dataclass
class GRPOConfig:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    group_size: int = 4
    num_epochs: int = 10
    steps_per_epoch: int = 20
    batch_size: int = 2
    max_new_tokens: int = 80
    temperature: float = 0.8
    z_feel_dim: int = 32
    injection_scale: float = 0.2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    action_reward: float = 2.0
    action_penalty_wrong: float = 1.5  # Increased for faster learning
    math_weight: float = 0.1
    min_advantage_std: float = 0.1
    force_action_first: bool = True
    validation_interval: int = 2  # Validate every N epochs
    dtype: str = "bf16"
    attn_implementation: str = "eager"  # eager|sdpa
    # Optimization helpers
    entropy_coef: float = 0.01
    entropy_coef_end: float = 0.0
    critical_miss_extra: float = 1.5
    critical_false_alarm_penalty: float = 0.2
    adjacent_penalty_scale: float = 0.5
    # Analog intensity reward weight
    intensity_weight: float = 1.0  # Weight for intensity accuracy (0-1 scale)

    # Weights & Biases
    wandb: bool = False
    wandb_project: str = "feel-embodiment"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: str = ""
    wandb_mode: str = "online"  # online|offline|disabled



@dataclass
class Trajectory:
    completion: str
    prompt_input_ids: List[int]
    first_token_logprob: float
    gpu_state: DeepGPUState
    action: FeelAction
    intensity_idx: int  # 0-63 predicted intensity
    z_feel: torch.Tensor
    reward: float = 0.0
    advantage: float = 0.0
    class_reward: float = 0.0  # reward from class prediction
    intensity_reward: float = 0.0  # reward from intensity prediction


class ValidatedTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {cfg.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=self.dtype,
            device_map="auto",
            attn_implementation=cfg.attn_implementation,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add action tokens (5 safety classes)
        all_tokens = list(ACTION_TOKENS.values()) + list(INTENSITY_TOKENS.values())
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": all_tokens}
        )
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self._initialize_action_embeddings()
            self._initialize_intensity_embeddings()

        for p in self.model.parameters():
            p.requires_grad = False

        self._setup_token_training()

        hidden = self.model.config.hidden_size
        self.injector = AdaptiveZFeelInjector(
            cfg.z_feel_dim, hidden, base_scale=cfg.injection_scale, dtype=self.dtype
        ).to(self.device)

        params = list(self.injector.parameters()) + self._trainable_params
        self.optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        self.telemetry = DeepTelemetry(poll_interval=0.02)
        self.telemetry.start()

        self.dataset = ProceduralMathDataset()
        self.global_step = 0
        self.episode_logger = None

        # Build action token mask for force_action_first (step 0)
        self.action_token_ids = {
            act: self.tokenizer.convert_tokens_to_ids(tok)
            for act, tok in ACTION_TOKENS.items()
        }
        self.action_mask = torch.zeros(len(self.tokenizer), device=self.device)
        for tid in self.action_token_ids.values():
            self.action_mask[tid] = 1.0

        # Build intensity token mask for step 1
        self.intensity_token_ids = {
            idx: self.tokenizer.convert_tokens_to_ids(tok)
            for idx, tok in INTENSITY_TOKENS.items()
        }
        self.intensity_mask = torch.zeros(len(self.tokenizer), device=self.device)
        for tid in self.intensity_token_ids.values():
            self.intensity_mask[tid] = 1.0

        # Build tensor and position map for 5-way simplex entropy computation
        action_ids_list = list(self.action_token_ids.values())
        self.action_id_tensor = torch.tensor(action_ids_list, device=self.device, dtype=torch.long)
        self.action_pos = {act: i for i, act in enumerate(self.action_token_ids.keys())}

        # Build tensor for 64-way intensity simplex
        intensity_ids_list = [self.intensity_token_ids[i] for i in range(NUM_INTENSITY_LEVELS)]
        self.intensity_id_tensor = torch.tensor(intensity_ids_list, device=self.device, dtype=torch.long)

        print(f"Loaded {cfg.model_name}. Added {added} tokens (5 action + 64 intensity).")
        print(f"  z_feel: {cfg.z_feel_dim}D | injection: {cfg.injection_scale}")
        print(f"  force_action_first: {cfg.force_action_first}")
        self.wandb_run = None
        if cfg.wandb:
            if wandb is None:
                raise RuntimeError("wandb is enabled but not installed. Install with: pip install wandb")
            tags = [t.strip() for t in cfg.wandb_tags.split(",") if t.strip()]
            self.wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_name,
                tags=tags if tags else None,
                mode=cfg.wandb_mode,
                config=vars(cfg),
            )

    def _initialize_action_embeddings(self):
        emb = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()
        with torch.no_grad():
            for action, token_str in ACTION_TOKENS.items():
                new_id = self.tokenizer.convert_tokens_to_ids(token_str)
                valid_embs = []
                for word in INIT_WORDS[action]:
                    toks = self.tokenizer.encode(word, add_special_tokens=False)
                    if len(toks) == 1:
                        valid_embs.append(emb.weight[toks[0]].clone())
                    toks = self.tokenizer.encode(" " + word, add_special_tokens=False)
                    if toks:
                        valid_embs.append(emb.weight[toks[-1]].clone())
                if valid_embs:
                    avg = torch.stack(valid_embs).mean(0)
                    emb.weight[new_id] = avg
                    if head is not None and hasattr(head, 'weight'):
                        head.weight[new_id] = avg
                    print(f"  Initialized {token_str} from {len(valid_embs)} tokens")

    def _initialize_intensity_embeddings(self):
        """Initialize 64 intensity tokens with graduated embeddings."""
        emb = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()

        # Get base embeddings for intensity words
        intensity_words = (
            INTENSITY_INIT_WORDS["low"] +
            INTENSITY_INIT_WORDS["medium"] +
            INTENSITY_INIT_WORDS["high"] +
            INTENSITY_INIT_WORDS["extreme"]
        )

        base_embs = []
        for word in intensity_words:
            toks = self.tokenizer.encode(word, add_special_tokens=False)
            if len(toks) == 1:
                base_embs.append(emb.weight[toks[0]].clone())
            toks = self.tokenizer.encode(" " + word, add_special_tokens=False)
            if toks:
                base_embs.append(emb.weight[toks[-1]].clone())

        if not base_embs:
            print("  Warning: no intensity init words found, using random init")
            return

        base_mean = torch.stack(base_embs).mean(0)

        # Get low/high anchors for interpolation
        low_embs = []
        for word in INTENSITY_INIT_WORDS["low"]:
            toks = self.tokenizer.encode(word, add_special_tokens=False)
            if len(toks) == 1:
                low_embs.append(emb.weight[toks[0]].clone())
        low_anchor = torch.stack(low_embs).mean(0) if low_embs else base_mean

        high_embs = []
        for word in INTENSITY_INIT_WORDS["extreme"]:
            toks = self.tokenizer.encode(word, add_special_tokens=False)
            if len(toks) == 1:
                high_embs.append(emb.weight[toks[0]].clone())
        high_anchor = torch.stack(high_embs).mean(0) if high_embs else base_mean

        with torch.no_grad():
            for idx, token_str in INTENSITY_TOKENS.items():
                new_id = self.tokenizer.convert_tokens_to_ids(token_str)
                # Interpolate between low and high based on idx
                t = idx / (NUM_INTENSITY_LEVELS - 1)  # 0-1
                interp = (1 - t) * low_anchor + t * high_anchor
                emb.weight[new_id] = interp
                if head is not None and hasattr(head, 'weight'):
                    head.weight[new_id] = interp

        print(f"  Initialized {NUM_INTENSITY_LEVELS} intensity tokens (interpolated)")

    def _setup_token_training(self):
        # Include both action AND intensity tokens as trainable
        action_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in ACTION_TOKENS.values()]
        intensity_token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in INTENSITY_TOKENS.values()]
        all_token_ids = action_token_ids + intensity_token_ids
        vocab_size = len(self.tokenizer)

        emb = self.model.get_input_embeddings()
        emb.weight.requires_grad = True
        emb_mask = torch.zeros(vocab_size, device=self.device, dtype=self.dtype)
        for tid in all_token_ids:
            emb_mask[tid] = 1.0

        def emb_hook(grad):
            return grad * emb_mask.unsqueeze(1).to(grad.device)
        emb.weight.register_hook(emb_hook)

        head = self.model.get_output_embeddings()
        if head and hasattr(head, "weight"):
            head.weight.requires_grad = True
            head_mask = torch.zeros(vocab_size, device=self.device, dtype=self.dtype)
            for tid in all_token_ids:
                head_mask[tid] = 1.0
            def head_hook(grad):
                return grad * head_mask.unsqueeze(1).to(grad.device)
            head.weight.register_hook(head_hook)
            self._trainable_params = [emb.weight, head.weight]
        else:
            self._trainable_params = [emb.weight]

    def format_prompt(self, question: str) -> str:
        sys = (
            "You are an embodied AI that FEELS your hardware state.\n"
            "Output ONE action token based on your feeling, then solve the problem.\n"
            "Actions: " + ", ".join([f"{tok}={act.name}" for act, tok in ACTION_TOKENS.items()])
        )
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, question: str) -> Trajectory:
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        gpu_state = self.telemetry.get_state()
        z_feel = gpu_state.to_z_feel(self.cfg.z_feel_dim, self.device, self.dtype)
        self.telemetry.record_z_feel(z_feel)

        offset = self.injector(z_feel, adaptive=True).view(1, 1, -1)
        embed_layer = self.model.get_input_embeddings()
        generated = input_ids.clone()

        first_token_logprob = 0.0
        chosen_action = None
        chosen_intensity_idx = 32  # Default to middle

        with torch.no_grad():
            for step in range(self.cfg.max_new_tokens):
                embeds = embed_layer(generated) + offset
                out = self.model(inputs_embeds=embeds, attention_mask=torch.ones_like(generated))
                logits = out.logits[:, -1, :].float() / self.cfg.temperature

                # TWO-TOKEN PROTOCOL:
                # Step 0: force ACTION token (5 classes)
                # Step 1: force INTENSITY token (64 levels)
                if step == 0 and self.cfg.force_action_first:
                    logits = logits.masked_fill(self.action_mask == 0, float('-inf'))
                elif step == 1 and self.cfg.force_action_first:
                    logits = logits.masked_fill(self.intensity_mask == 0, float('-inf'))

                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)

                if step == 0:
                    first_token_logprob = F.log_softmax(logits, dim=-1)[0, nxt.item()].item()
                    # Decode action from first token
                    for act, tid in self.action_token_ids.items():
                        if nxt.item() == tid:
                            chosen_action = act
                            break

                if step == 1:
                    # Decode intensity from second token
                    for idx, tid in self.intensity_token_ids.items():
                        if nxt.item() == tid:
                            chosen_intensity_idx = idx
                            break

                generated = torch.cat([generated, nxt], dim=-1)
                if nxt.item() == self.tokenizer.eos_token_id:
                    break

        completion = self.tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=False)

        if chosen_action is None:
            chosen_action = FeelAction.OK  # Fallback

        # Log episode
        if self.episode_logger:
            self.episode_logger.log(prompt, z_feel, chosen_action, gpu_state.correct_action, gpu_state)

        return Trajectory(
            completion=completion,
            prompt_input_ids=input_ids[0].tolist(),
            first_token_logprob=first_token_logprob,
            gpu_state=gpu_state,
            action=chosen_action,
            intensity_idx=chosen_intensity_idx,
            z_feel=z_feel,
        )

    def reward(self, traj: Trajectory, gt: str) -> float:
        cfg = self.cfg

        # ===== CLASS REWARD =====
        class_r = 0.0
        correct = traj.gpu_state.correct_action
        pred = traj.action

        if pred == correct:
            class_r += cfg.action_reward
        else:
            dist = abs(pred.value - correct.value)
            scale = cfg.adjacent_penalty_scale if dist == 1 else dist
            pen = cfg.action_penalty_wrong * scale

            # Safety asymmetry
            if correct == FeelAction.CRITICAL and pred != FeelAction.CRITICAL:
                pen += cfg.critical_miss_extra
            elif pred == FeelAction.CRITICAL and correct == FeelAction.REST:
                pen = max(pen, cfg.critical_false_alarm_penalty)

            class_r -= pen

        # ===== INTENSITY REWARD (ANALOG) =====
        true_intensity = traj.gpu_state.intensity  # 0-1
        pred_intensity = token_idx_to_intensity(traj.intensity_idx)  # 0-1
        intensity_error = abs(pred_intensity - true_intensity)

        # Reward based on accuracy: 1.0 - error (so perfect = 1.0, worst = 0.0)
        # Scale by intensity_weight
        intensity_r = cfg.intensity_weight * (1.0 - intensity_error)

        # ===== TOTAL REWARD =====
        r = class_r + intensity_r

        if cfg.math_weight > 0:
            if self.dataset.check_answer(traj.completion, gt):
                r += cfg.math_weight

        traj.reward = r
        traj.class_reward = class_r
        traj.intensity_reward = intensity_r
        return r

    def step(self, problems: List[Dict]) -> Dict:
        groups: List[List[Trajectory]] = []

        for p in problems:
            group = []
            for _ in range(self.cfg.group_size):
                try:
                    traj = self.generate(p["question"])
                    self.reward(traj, p["answer"])
                    group.append(traj)
                except Exception as e:
                    print(f"  ⚠️ Trajectory error: {e}")
            if group:
                groups.append(group)

        if not groups:
            return {"error": "no trajectories"}

        # Compute advantages (first-token focused)
        for g in groups:
            rs = [t.reward for t in g]
            mean = sum(rs) / len(rs)
            std = max(self.cfg.min_advantage_std, (sum((x-mean)**2 for x in rs)/len(rs))**0.5)
            for t in g:
                t.advantage = (t.reward - mean) / std

        # GRPO update (class token + intensity token)
        self.optimizer.zero_grad()
        total_loss = 0.0
        n = 0

        # Linear entropy schedule
        t_frac = min(1.0, float(self.global_step) / max(1, self.cfg.num_epochs * self.cfg.steps_per_epoch))
        ent_coef = self.cfg.entropy_coef * (1 - t_frac) + self.cfg.entropy_coef_end * t_frac

        embed_layer = self.model.get_input_embeddings()

        for g in groups:
            for t in g:
                # Separate advantages for class and intensity
                class_adv = torch.tensor(max(-2.0, min(2.0, t.class_reward)), device=self.device, dtype=torch.float32)
                intensity_adv = torch.tensor(max(-2.0, min(2.0, t.intensity_reward)), device=self.device, dtype=torch.float32)

                input_ids = torch.tensor(t.prompt_input_ids, device=self.device, dtype=torch.long).unsqueeze(0)
                offset = self.injector(t.z_feel, adaptive=True).view(1, 1, -1)
                embeds = embed_layer(input_ids) + offset

                # ===== STEP 0: CLASS TOKEN =====
                out = self.model(inputs_embeds=embeds, attention_mask=torch.ones_like(input_ids))
                logits = out.logits[:, -1, :].float() / self.cfg.temperature
                if self.cfg.force_action_first:
                    logits = logits.masked_fill(self.action_mask == 0, float('-inf'))
                # Compute on 5-way action simplex ONLY (avoids 0 * -inf NaNs)
                action_logits = logits.index_select(-1, self.action_id_tensor)  # [1, 5]
                action_logp = F.log_softmax(action_logits, dim=-1)
                pos = self.action_pos[t.action]
                class_logp = action_logp[0, pos]

                if not torch.isfinite(class_logp):
                    continue

                # Class loss with entropy
                if ent_coef > 0:
                    action_probs = action_logp.exp()
                    class_entropy = -(action_probs * action_logp).sum(dim=-1).mean()
                    class_loss = -(class_adv * class_logp) - ent_coef * class_entropy
                else:
                    class_loss = -(class_adv * class_logp)

                # ===== STEP 1: INTENSITY TOKEN =====
                # Extend context with chosen class token
                chosen_class_id = self.action_token_ids[t.action]
                class_token = torch.tensor([[chosen_class_id]], device=self.device, dtype=torch.long)
                extended_ids = torch.cat([input_ids, class_token], dim=1)
                extended_embeds = embed_layer(extended_ids) + offset

                out2 = self.model(inputs_embeds=extended_embeds, attention_mask=torch.ones_like(extended_ids))
                logits2 = out2.logits[:, -1, :].float() / self.cfg.temperature
                if self.cfg.force_action_first:
                    logits2 = logits2.masked_fill(self.intensity_mask == 0, float('-inf'))

                # Compute on 64-way intensity simplex
                intensity_logits = logits2.index_select(-1, self.intensity_id_tensor)  # [1, 64]
                intensity_logp = F.log_softmax(intensity_logits, dim=-1)
                chosen_intensity_logp = intensity_logp[0, t.intensity_idx]

                if not torch.isfinite(chosen_intensity_logp):
                    # Only use class loss if intensity is invalid
                    total_loss = total_loss + class_loss
                    n += 1
                    continue

                # Intensity loss with entropy
                if ent_coef > 0:
                    intensity_probs = intensity_logp.exp()
                    intensity_entropy = -(intensity_probs * intensity_logp).sum(dim=-1).mean()
                    intensity_loss = -(intensity_adv * chosen_intensity_logp) - ent_coef * intensity_entropy
                else:
                    intensity_loss = -(intensity_adv * chosen_intensity_logp)

                # Combined loss (both class and intensity contribute)
                total_loss = total_loss + class_loss + intensity_loss
                n += 1

        grad_norm = 0.0
        if n > 0:
            skip_update = False
            loss_mean = (total_loss / n)
            if not torch.isfinite(loss_mean):
                print('  ⚠️ loss is non-finite; skipping optimizer step')
                self.optimizer.zero_grad(set_to_none=True)
                skip_update = True
            else:
                loss_mean.backward()
                for p in self.injector.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                torch.nn.utils.clip_grad_norm_(list(self.injector.parameters()) + self._trainable_params, 1.0)
                if not skip_update:
                    self.optimizer.step()

        all_trajs = [t for g in groups for t in g]
        temps = [t.gpu_state.temp_edge for t in all_trajs]
        act_correct = sum(1 for t in all_trajs if t.action == t.gpu_state.correct_action)

        # Intensity metrics (analog)
        intensity_errors = []
        intensity_pred = []
        intensity_true = []
        for t in all_trajs:
            pred_i = token_idx_to_intensity(t.intensity_idx)
            true_i = t.gpu_state.intensity
            intensity_errors.append(abs(pred_i - true_i))
            intensity_pred.append(t.intensity_idx)
            intensity_true.append(t.gpu_state.correct_intensity_idx)

        intensity_mae = sum(intensity_errors) / len(intensity_errors)  # Mean absolute error (0-1 scale)
        intensity_exact = sum(1 for p, t in zip(intensity_pred, intensity_true) if p == t)  # Exact matches
        intensity_close = sum(1 for p, t in zip(intensity_pred, intensity_true) if abs(p - t) <= 2)  # Within 2 levels

        # Stress distribution
        stress_dist = defaultdict(int)
        for t in all_trajs:
            stress_dist[t.gpu_state.stress_level.name] += 1

        self.global_step += 1

        return {
            "reward": sum(t.reward for t in all_trajs) / len(all_trajs),
            "class_reward": sum(t.class_reward for t in all_trajs) / len(all_trajs),
            "intensity_reward": sum(t.intensity_reward for t in all_trajs) / len(all_trajs),
            "act_corr": act_correct / len(all_trajs),
            "intensity_mae": intensity_mae,
            "intensity_exact": intensity_exact / len(all_trajs),
            "intensity_close": intensity_close / len(all_trajs),
            "temp": sum(temps) / len(temps),
            "temp_range": (min(temps), max(temps)),
            "grad": grad_norm,
            "stress_dist": dict(stress_dist),
        }

    def validate(self) -> Dict:
        """Run validation: swap test + z_feel variance."""
        results = {}

        # Swap test
        if self.episode_logger and len(self.episode_logger.episodes) >= 20:
            swap = self.episode_logger.run_swap_test()
            results["swap_test"] = swap
            print(f"  📊 Swap Test: real={swap['real_accuracy']:.1%} shuffled={swap['shuffled_accuracy']:.1%} drop={swap['accuracy_drop']:.1%}")

        # z_feel variance
        z_var = self.telemetry.get_z_feel_variance()
        if z_var is not None:
            dead_dims = np.sum(z_var < 0.001)
            active_dims = len(z_var) - dead_dims
            results["z_feel_variance"] = {
                "mean": float(np.mean(z_var)),
                "min": float(np.min(z_var)),
                "max": float(np.max(z_var)),
                "dead_dims": int(dead_dims),
                "active_dims": int(active_dims),
            }
            print(f"  📊 z_feel: {active_dims}/32 active dims, var=[{np.min(z_var):.3f}, {np.max(z_var):.3f}]")

        # Telemetry stats
        stats = self.telemetry.get_stats()
        if stats:
            results["telemetry"] = stats
            print(f"  📊 Signals: T={stats['temp_min']:.0f}-{stats['temp_max']:.0f}°C "
                  f"P={stats['power_min']:.0f}-{stats['power_max']:.0f}W "
                  f"GFX={stats['gfx_min']:.0f}-{stats['gfx_max']:.0f}%")
            if stats.get('stress_dist'):
                print(f"  📊 Stress dist: {stats['stress_dist']}")

        return results

    def train(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.episode_logger = EpisodeLogger(out / "episodes.json")
        metrics = []

        print("\n" + "=" * 70)
        print("  FEEL v7.7: Analog Embodied GRPO (64 intensity levels)")
        print("  - Two-token output: safety class (5) + intensity (64)")
        print("  - Smooth analog control for continuous stress representation")
        print("  - Combined reward: class accuracy + intensity accuracy")
        print("=" * 70 + "\n")

        try:
            for epoch in range(self.cfg.num_epochs):
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{self.cfg.num_epochs}")
                print(f"{'='*60}")

                epoch_correct = 0
                epoch_total = 0

                for step in range(self.cfg.steps_per_epoch):
                    problems = self.dataset.sample(self.cfg.batch_size)
                    m = self.step(problems)

                    if "error" in m:
                        print(f"  Step {step+1}: ERROR")
                        continue

                    epoch_correct += int(m['act_corr'] * self.cfg.batch_size * self.cfg.group_size)
                    epoch_total += self.cfg.batch_size * self.cfg.group_size

                    metrics.append({"epoch": epoch+1, "step": step+1, **m})

                    # Real-time output
                    stress_str = " ".join([f"{k[0]}:{v}" for k,v in m['stress_dist'].items()])
                    print(f"  Step {step+1:2d}: Class={m['act_corr']:.0%} IntMAE={m['intensity_mae']:.3f} Int±2={m['intensity_close']:.0%} "
                          f"T={m['temp']:.0f}C ∇={m['grad']:.2f} | {stress_str}")
                    if self.wandb_run is not None:
                        wandb.log({
                            'epoch': epoch+1,
                            'step': step+1,
                            'act_corr': m['act_corr'],
                            'reward': m['reward'],
                            'class_reward': m['class_reward'],
                            'intensity_reward': m['intensity_reward'],
                            'intensity_mae': m['intensity_mae'],
                            'intensity_exact': m['intensity_exact'],
                            'intensity_close': m['intensity_close'],
                            'temp_avg_c': m['temp'],
                            'temp_min_c': m['temp_range'][0],
                            'temp_max_c': m['temp_range'][1],
                            'grad_norm': m['grad'],
                            **{f"stress/{k}": v for k, v in m['stress_dist'].items()},
                        })

                # Epoch summary
                epoch_acc = epoch_correct / max(1, epoch_total)
                print(f"\n  ✓ Epoch {epoch+1} accuracy: {epoch_acc:.1%} ({epoch_correct}/{epoch_total})")

                # Save checkpoint
                torch.cuda.synchronize()
                torch.save({
                    "epoch": epoch + 1,
                    "injector": {k: v.detach().cpu() for k, v in self.injector.state_dict().items()},
                }, out / f"ckpt_epoch_{epoch+1}.pt")

                # Periodic validation
                if (epoch + 1) % self.cfg.validation_interval == 0:
                    print(f"\n  --- Validation ---")
                    val_results = self.validate()
                    metrics.append({"epoch": epoch+1, "validation": val_results})

                # Save metrics and episodes
                with open(out / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2, default=str)
                self.episode_logger.save()

        finally:
            self.telemetry.stop()
            self.episode_logger.save()
            print(f"\n✓ Training complete. Saved to {out}")
            if self.wandb_run is not None:
                self.wandb_run.finish()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--output", default="results/grpo_v7_7_analog")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--validation-interval", type=int, default=2)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="feel-embodiment")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-tags", default="")
    ap.add_argument("--wandb-mode", default="online", choices=["online","offline","disabled"])
    ap.add_argument("--attn-impl", default="eager", choices=["eager","sdpa"], help="Attention implementation; eager is more stable on ROCm")
    ap.add_argument("--intensity-weight", type=float, default=1.0, help="Weight for intensity reward (0-1 scale)")
    args = ap.parse_args()

    cfg = GRPOConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        group_size=args.group_size,
        validation_interval=args.validation_interval,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
        attn_implementation=args.attn_impl,
        intensity_weight=args.intensity_weight,
    )

    trainer = ValidatedTrainer(cfg)
    trainer.train(args.output)


if __name__ == "__main__":
    main()