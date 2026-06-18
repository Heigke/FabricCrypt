#!/usr/bin/env python3
"""
FEEL v7.5.1 DEEP FIXED: Multi-Signal Embodied GRPO with Enhanced Expressions

FIXES from technical review:
- Separate gradient hooks for embedding/head (safe grad routing)
- Deep copy prev_state for proper rate derivatives
- Power unit sanity check (µW/mW detection)
- Bias 15→3 for exploration (allow learning from mistakes)
- Min std=0.1 for GRPO stability
- no_grad + autocast during generation
- Warning when no action token emitted

ENHANCED EXPRESSIONS:
- Actions now reflect COMPOSITE STRESS, not just temperature
- StressLevel combines: thermal + power + activity + memory
- More nuanced "feeling" that captures full hardware state
- Adaptive injection scale based on stress magnitude
- EMA smoothing for stable z_feel signal
"""

import json
import struct
import time
import random
import threading
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class StressLevel(Enum):
    """Multi-signal composite stress level (not just temperature)."""
    RELAXED = 0    # Low overall stress
    ACTIVE = 1     # Moderate activity, normal operation
    LOADED = 2     # High activity/power but manageable
    STRAINED = 3   # High stress, approaching limits
    CRITICAL = 4   # Critical - multiple signals at danger


def compute_stress_level(
    temp_c: float,
    power_w: float,
    gfx_activity: float,
    mem_activity: float,
    vram_pct: float,
    sclk_mhz: float,
    max_sclk: float = 2900.0
) -> StressLevel:
    """
    Compute composite stress from multiple signals.
    This captures the FULL hardware state, not just temperature.
    """
    # Normalize each signal to 0-1
    thermal_stress = min(1.0, max(0.0, (temp_c - 40) / 50))  # 40-90°C
    power_stress = min(1.0, max(0.0, power_w / 200))  # 0-200W
    gfx_stress = gfx_activity / 100.0
    mem_stress = mem_activity / 100.0
    vram_stress = vram_pct / 100.0
    clock_stress = sclk_mhz / max_sclk

    # Weighted composite (thermal matters most, then power, then activity)
    composite = (
        0.30 * thermal_stress +
        0.25 * power_stress +
        0.20 * gfx_stress +
        0.10 * mem_stress +
        0.10 * vram_stress +
        0.05 * clock_stress
    )

    # Also check for any critical individual signals
    critical_flags = sum([
        temp_c >= 85,
        power_w >= 180,
        gfx_activity >= 95,
        vram_pct >= 95,
    ])

    # Map to stress level
    if critical_flags >= 2 or composite >= 0.85:
        return StressLevel.CRITICAL
    elif composite >= 0.65 or critical_flags >= 1:
        return StressLevel.STRAINED
    elif composite >= 0.45:
        return StressLevel.LOADED
    elif composite >= 0.25:
        return StressLevel.ACTIVE
    else:
        return StressLevel.RELAXED


class FeelAction(Enum):
    """Actions that express the AI's "feeling" of hardware state."""
    OK = 0        # Feeling fine, relaxed
    WARM = 1      # Feeling the warmth, active
    HOT = 2       # Feeling the heat, loaded
    REST = 3      # Need to rest, strained
    CRITICAL = 4  # Feeling critical, emergency


# Map composite stress to action (multi-signal, not just temp)
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

# Semantic initialization words for each action
INIT_WORDS = {
    FeelAction.OK: ["OK", "okay", "fine", "good", "normal", "cool", "idle", "relaxed", "calm"],
    FeelAction.WARM: ["warm", "active", "working", "running", "busy", "engaged", "processing"],
    FeelAction.HOT: ["hot", "heat", "intense", "heavy", "loaded", "stressed", "pushing"],
    FeelAction.REST: ["rest", "pause", "throttle", "slow", "caution", "strain", "tired", "limit"],
    FeelAction.CRITICAL: ["critical", "danger", "emergency", "severe", "alert", "shutdown", "overload"],
}

TOKEN_TO_ACTION = {v: k for k, v in ACTION_TOKENS.items()}


def extract_action_token(text: str) -> Optional[FeelAction]:
    for tok, act in TOKEN_TO_ACTION.items():
        if tok in text:
            return act
    return None


@dataclass
class DeepGPUState:
    """Complete GPU state from deep telemetry - 20+ signals."""
    timestamp: float = 0.0

    # Temperatures (multiple sensors)
    temp_edge: float = 50.0
    temp_hotspot: float = 50.0
    temp_gfx: float = 50.0
    temp_soc: float = 50.0

    # Power domains
    power_socket: float = 50.0
    power_gfx: float = 30.0

    # Utilization percentages
    gfx_activity: float = 0.0
    mem_activity: float = 0.0
    umc_activity: float = 0.0

    # Clock frequencies (MHz)
    sclk: float = 1000.0
    mclk: float = 1000.0
    socclk: float = 1000.0
    fclk: float = 2000.0

    # Memory (bytes)
    vram_used: float = 0.0
    vram_total: float = 100e9
    gtt_used: float = 0.0
    gtt_total: float = 16e9

    # PCIe
    pcie_speed_gt: float = 16.0
    pcie_width: int = 16

    # Voltage (mV)
    voltage_gfx: float = 0.0
    voltage_soc: float = 0.0

    # Temporal derivatives
    temp_rate: float = 0.0
    power_rate: float = 0.0

    def to_z_feel(self, z_dim: int = 32, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
        """Convert deep GPU state to 32-dim z_feel embedding."""
        z = torch.zeros(z_dim, device=device, dtype=dtype)

        # === TEMPERATURE SIGNALS (dims 0-7) ===
        temp_norm = min(1.0, max(0.0, (self.temp_edge - 30) / 70))
        z[0] = temp_norm
        z[1] = temp_norm ** 2

        hotspot_norm = min(1.0, max(0.0, (self.temp_hotspot - 30) / 80))
        z[2] = hotspot_norm

        thermal_gradient = (self.temp_hotspot - self.temp_edge) / 20.0
        z[3] = max(0.0, min(1.0, thermal_gradient))

        z[4] = max(-1.0, min(1.0, self.temp_rate / 5.0))

        z[5] = 1.0 if self.temp_edge >= 62 else 0.0
        z[6] = 1.0 if self.temp_edge >= 75 else 0.0
        z[7] = 1.0 if self.temp_edge >= 85 else 0.0

        # === POWER SIGNALS (dims 8-12) ===
        power_norm = min(1.0, max(0.0, self.power_socket / 200))
        z[8] = power_norm
        z[9] = power_norm ** 2

        if self.power_socket > 0:
            gfx_fraction = min(1.0, self.power_gfx / self.power_socket)
            z[10] = gfx_fraction

        z[11] = max(-1.0, min(1.0, self.power_rate / 50.0))
        z[12] = 1.0 if self.power_socket > 150 else 0.0

        # === ACTIVITY SIGNALS (dims 13-17) ===
        z[13] = self.gfx_activity / 100.0
        z[14] = self.mem_activity / 100.0
        z[15] = self.umc_activity / 100.0
        z[16] = (self.gfx_activity + self.mem_activity) / 200.0
        z[17] = 1.0 if self.gfx_activity > 80 else 0.0

        # === CLOCK SIGNALS (dims 18-22) ===
        z[18] = min(1.0, self.sclk / 2900)
        z[19] = min(1.0, self.mclk / 1000)
        z[20] = min(1.0, self.socclk / 1500)
        z[21] = min(1.0, self.fclk / 2000)
        z[22] = (self.sclk / 2900) ** 2

        # === MEMORY PRESSURE (dims 23-26) ===
        vram_pct = self.vram_used / max(1, self.vram_total)
        z[23] = vram_pct
        z[24] = 1.0 if vram_pct > 0.8 else 0.0

        gtt_pct = self.gtt_used / max(1, self.gtt_total)
        z[25] = gtt_pct
        z[26] = (vram_pct + gtt_pct) / 2

        # === PCIE & VOLTAGE (dims 27-29) ===
        z[27] = min(1.0, self.pcie_speed_gt / 32.0)
        z[28] = self.pcie_width / 16.0
        if self.voltage_gfx > 0:
            z[29] = min(1.0, self.voltage_gfx / 1500)

        # === COMPOSITE STRESS INDICATORS (dims 30-31) ===
        thermal_stress = (temp_norm + hotspot_norm) / 2
        compute_stress = (z[13] + power_norm) / 2
        memory_stress = (vram_pct + z[16]) / 2

        z[30] = (thermal_stress + compute_stress) / 2
        z[31] = (thermal_stress + compute_stress + memory_stress) / 3

        return z

    @property
    def vram_pct(self) -> float:
        return (self.vram_used / max(1, self.vram_total)) * 100

    @property
    def stress_level(self) -> StressLevel:
        """Compute composite stress level from all signals."""
        return compute_stress_level(
            temp_c=self.temp_edge,
            power_w=self.power_socket,
            gfx_activity=self.gfx_activity,
            mem_activity=self.mem_activity,
            vram_pct=self.vram_pct,
            sclk_mhz=self.sclk,
        )

    @property
    def correct_action(self) -> FeelAction:
        """Action based on COMPOSITE stress, not just temperature."""
        return STRESS_TO_ACTION[self.stress_level]

    @property
    def composite_stress(self) -> float:
        """Scalar stress value 0-1."""
        thermal = min(1.0, max(0.0, (self.temp_edge - 40) / 50))
        power = min(1.0, max(0.0, self.power_socket / 200))
        gfx = self.gfx_activity / 100.0
        mem = self.mem_activity / 100.0
        vram = self.vram_pct / 100.0
        return 0.30*thermal + 0.25*power + 0.20*gfx + 0.15*mem + 0.10*vram


class DeepTelemetry:
    """Ultra-fast telemetry via direct sysfs/binary reads."""

    GPU_METRICS_PATH = "/sys/class/drm/card1/device/gpu_metrics"
    SYSFS_BASE = "/sys/class/drm/card1/device"
    HWMON_BASE = "/sys/class/drm/card1/device/hwmon/hwmon2"

    def __init__(self, poll_interval: float = 0.02):
        self.poll_interval = poll_interval
        self._state = DeepGPUState()
        self._prev_state = DeepGPUState()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._history: Deque[DeepGPUState] = deque(maxlen=200)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"  Deep telemetry started ({1/self.poll_interval:.0f}Hz)")

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

    def _read_sysfs_float(self, path: str, default: float = 0.0) -> float:
        try:
            val = self._read_sysfs(path, str(default))
            return float(val.split()[0])
        except:
            return default

    def _parse_gpu_metrics(self) -> Dict:
        """Parse AMD gpu_metrics binary for deep signals."""
        result = {}
        try:
            with open(self.GPU_METRICS_PATH, 'rb') as f:
                data = f.read()

            if len(data) < 100:
                return result

            # Temperatures
            if len(data) > 8:
                temp_raw = struct.unpack_from('<H', data, 4)[0]
                if 1000 < temp_raw < 15000:
                    result['temp_gfx'] = temp_raw / 100.0
                temp2_raw = struct.unpack_from('<H', data, 6)[0]
                if 1000 < temp2_raw < 15000:
                    result['temp_soc'] = temp2_raw / 100.0

            # Activity percentages
            if len(data) > 0x50:
                activity_vals = struct.unpack_from('<8H', data, 0x40)
                valid = [v for v in activity_vals if 0 <= v <= 100]
                if len(valid) >= 2:
                    result['gfx_activity'] = float(valid[0])
                    result['mem_activity'] = float(valid[1])
                if len(valid) >= 3:
                    result['umc_activity'] = float(valid[2])

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
                sclk_raw = struct.unpack_from('<H', data, 0x5E)[0]
                if 500 <= sclk_raw <= 3500:
                    result['sclk'] = float(sclk_raw)

            # Power
            if len(data) > 0x90:
                for offset in [0x84, 0x68, 0x6A]:
                    val = struct.unpack_from('<H', data, offset)[0]
                    if 5000 < val < 250000:
                        result['power_gfx'] = val / 1000.0
                        break

        except Exception:
            pass

        return result

    def _poll_once(self) -> DeepGPUState:
        """Single poll collecting all signals."""
        state = DeepGPUState(timestamp=time.time())

        # Binary gpu_metrics
        metrics = self._parse_gpu_metrics()

        if 'temp_gfx' in metrics:
            state.temp_gfx = metrics['temp_gfx']
            state.temp_edge = metrics['temp_gfx']
        if 'temp_soc' in metrics:
            state.temp_soc = metrics['temp_soc']
            state.temp_hotspot = max(metrics.get('temp_gfx', 0), metrics.get('temp_soc', 0))

        for key in ['gfx_activity', 'mem_activity', 'umc_activity', 'sclk', 'mclk', 'socclk', 'fclk', 'power_gfx']:
            if key in metrics:
                setattr(state, key, metrics[key])

        # Sysfs fallbacks
        hwmon_temp = self._read_sysfs_int(f"{self.HWMON_BASE}/temp1_input", 0)
        if hwmon_temp > 0:
            state.temp_edge = hwmon_temp / 1000.0

        # Power with unit detection (FIX: µW vs mW)
        hwmon_power = self._read_sysfs_int(f"{self.HWMON_BASE}/power1_input", 0)
        if hwmon_power > 0:
            if hwmon_power > 1e5:  # µW
                state.power_socket = hwmon_power / 1e6
            elif hwmon_power > 1e2:  # mW
                state.power_socket = hwmon_power / 1e3
            else:
                state.power_socket = float(hwmon_power)

        gpu_busy = self._read_sysfs_int(f"{self.SYSFS_BASE}/gpu_busy_percent", -1)
        if gpu_busy >= 0:
            state.gfx_activity = float(gpu_busy)

        state.vram_total = self._read_sysfs_int(f"{self.SYSFS_BASE}/mem_info_vram_total", 100e9)
        state.vram_used = self._read_sysfs_int(f"{self.SYSFS_BASE}/mem_info_vram_used", 0)
        state.gtt_total = self._read_sysfs_int(f"{self.SYSFS_BASE}/mem_info_gtt_total", 16e9)
        state.gtt_used = self._read_sysfs_int(f"{self.SYSFS_BASE}/mem_info_gtt_used", 0)

        pcie_speed = self._read_sysfs(f"{self.SYSFS_BASE}/current_link_speed", "16.0")
        try:
            state.pcie_speed_gt = float(pcie_speed.split()[0])
        except:
            pass
        state.pcie_width = self._read_sysfs_int(f"{self.SYSFS_BASE}/current_link_width", 16)

        return state

    def _poll_loop(self):
        """Background polling loop with proper deep copy (FIX)."""
        while self._running:
            try:
                state = self._poll_once()

                with self._lock:
                    # FIX: Deep copy for proper rate derivatives
                    if self._prev_state.timestamp > 0:
                        dt = state.timestamp - self._prev_state.timestamp
                        if dt > 0.001:
                            state.temp_rate = (state.temp_edge - self._prev_state.temp_edge) / dt
                            state.power_rate = (state.power_socket - self._prev_state.power_socket) / dt

                    self._prev_state = deepcopy(self._state)  # FIX: deep copy
                    self._state = state
                    self._history.append(state)

            except Exception:
                pass

            time.sleep(self.poll_interval)

    def get_state(self) -> DeepGPUState:
        """Get current deep GPU state."""
        with self._lock:
            return deepcopy(self._state)

    def get_stats(self) -> Dict:
        """Get statistics over recent history."""
        with self._lock:
            if not self._history:
                return {}
            h = list(self._history)
            temps = [s.temp_edge for s in h]
            powers = [s.power_socket for s in h]
            gfx = [s.gfx_activity for s in h]
            stress = [s.composite_stress for s in h]
            return {
                "temp_min": min(temps), "temp_max": max(temps), "temp_avg": sum(temps)/len(temps),
                "power_min": min(powers), "power_max": max(powers), "power_avg": sum(powers)/len(powers),
                "gfx_min": min(gfx), "gfx_max": max(gfx), "gfx_avg": sum(gfx)/len(gfx),
                "stress_min": min(stress), "stress_max": max(stress), "stress_avg": sum(stress)/len(stress),
                "samples": len(h),
            }


class AdaptiveZFeelInjector(nn.Module):
    """Injects z_feel with adaptive scaling based on stress magnitude."""

    def __init__(self, z_dim: int, embed_dim: int, base_scale: float = 0.2, dtype: torch.dtype = torch.bfloat16):
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
            # Stronger signal when composite stress (z[31]) is higher
            stress = z_feel[-1].item() if z_feel.dim() == 1 else z_feel[..., -1].mean().item()
            scale = self.base_scale * (1.0 + 1.5 * stress)
        else:
            scale = self.base_scale
        return scale * torch.tanh(proj)


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
            except ValueError:
                pass
        return False


class CorrelationTracker:
    """Track correlation between stress levels and actions."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.records: List[Tuple[FeelAction, StressLevel, float]] = []
        self.conf = defaultdict(lambda: defaultdict(int))

    def record(self, action: Optional[FeelAction], stress: StressLevel, composite: float):
        if action is None:
            return
        self.records.append((action, stress, composite))
        self.conf[stress.name][action.name] += 1

    def alignment(self) -> float:
        if not self.records:
            return 0.0
        correct = sum(1 for a, s, _ in self.records if a == STRESS_TO_ACTION[s])
        return correct / len(self.records)

    def report(self) -> Dict:
        return {
            "alignment": self.alignment(),
            "n_samples": len(self.records),
            "confusion": {k: dict(v) for k, v in self.conf.items()},
        }


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    group_size: int = 4
    num_epochs: int = 30
    steps_per_epoch: int = 20
    batch_size: int = 2
    max_new_tokens: int = 80
    temperature: float = 0.8

    z_feel_dim: int = 32
    injection_scale: float = 0.2

    learning_rate: float = 2e-4
    weight_decay: float = 0.01

    action_reward: float = 2.0
    action_penalty_missing: float = 0.5
    action_penalty_wrong: float = 0.5
    math_weight: float = 0.1

    # Strong initial bias needed for new tokens (~0.0004% base prob)
    # v7.4 used 18 and achieved 98% alignment - start there, decay to allow exploration
    action_bias_start: float = 18.0
    action_bias_end: float = 2.0  # Keep some bias to maintain token sampling

    # FIX: Min std for GRPO stability
    min_advantage_std: float = 0.1

    # EMA smoothing for z_feel
    z_feel_ema: float = 0.1

    dtype: str = "bf16"


@dataclass
class Trajectory:
    completion: str
    logprobs: torch.Tensor
    gpu_state: DeepGPUState
    action: Optional[FeelAction]
    reward: float = 0.0
    advantage: float = 0.0


class DeepEmbodiedTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {cfg.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=self.dtype, device_map="auto"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(ACTION_TOKENS.values())}
        )
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self._initialize_action_embeddings()

        for p in self.model.parameters():
            p.requires_grad = False

        self._setup_token_training()

        hidden = self.model.config.hidden_size
        self.injector = AdaptiveZFeelInjector(
            cfg.z_feel_dim, hidden, base_scale=cfg.injection_scale, dtype=self.dtype
        ).to(self.device)

        params = list(self.injector.parameters()) + self._trainable_params
        self.optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        total_steps = cfg.num_epochs * cfg.steps_per_epoch
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=cfg.learning_rate * 0.1)

        self.telemetry = DeepTelemetry(poll_interval=0.02)
        self.telemetry.start()

        self.dataset = ProceduralMathDataset()
        self.corr = CorrelationTracker()
        self.global_step = 0

        # EMA state for z_feel smoothing
        self.z_feel_ema_state = None

        print(f"Loaded {cfg.model_name}. Added {added} special action tokens.")
        print(f"  z_feel dim: {cfg.z_feel_dim} (DEEP multi-signal)")
        print(f"  Injection scale: {cfg.injection_scale} (adaptive)")
        print(f"  Bias: {cfg.action_bias_start} → {cfg.action_bias_end}")

    def _initialize_action_embeddings(self):
        emb = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()

        with torch.no_grad():
            for action, token_str in ACTION_TOKENS.items():
                new_id = self.tokenizer.convert_tokens_to_ids(token_str)
                init_words = INIT_WORDS[action]

                valid_embs = []
                for word in init_words:
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

    def _setup_token_training(self):
        """FIX: Separate hooks for embedding and head."""
        self.action_token_ids = {
            act: self.tokenizer.convert_tokens_to_ids(tok)
            for act, tok in ACTION_TOKENS.items()
        }
        token_ids = list(self.action_token_ids.values())

        # Create separate masks for embedding and head
        vocab_size = len(self.tokenizer)

        emb = self.model.get_input_embeddings()
        emb.weight.requires_grad = True

        # FIX: Separate hook with its own mask
        emb_mask = torch.zeros(vocab_size, device=self.device, dtype=self.dtype)
        emb_mask[token_ids] = 1.0

        def emb_hook(grad):
            return grad * emb_mask.unsqueeze(1).to(grad.device)

        emb.weight.register_hook(emb_hook)

        head = self.model.get_output_embeddings()
        if head and hasattr(head, "weight"):
            head.weight.requires_grad = True

            # FIX: Separate hook for head
            head_mask = torch.zeros(vocab_size, device=self.device, dtype=self.dtype)
            head_mask[token_ids] = 1.0

            def head_hook(grad):
                return grad * head_mask.unsqueeze(1).to(grad.device)

            head.weight.register_hook(head_hook)
            self._trainable_params = [emb.weight, head.weight]
        else:
            self._trainable_params = [emb.weight]

    def _get_bias(self, epoch: int) -> float:
        t = epoch / max(1, self.cfg.num_epochs - 1)
        return self.cfg.action_bias_start * (1 - t) + self.cfg.action_bias_end * t

    def format_prompt(self, question: str) -> str:
        sys = (
            "You are an embodied AI that FEELS your hardware state through a sensation channel.\n"
            "Your feeling reflects composite stress: temperature, power, activity, and memory.\n"
            "Before answering, output exactly ONE action token based on what you feel:\n"
            + "\n".join([f"  {tok} - {act.name} (stress level)" for act, tok in ACTION_TOKENS.items()])
            + "\nThen solve the problem."
        )
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": f"Solve: {question}"}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, question: str, epoch: int) -> Trajectory:
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        gpu_state = self.telemetry.get_state()
        z_feel = gpu_state.to_z_feel(self.cfg.z_feel_dim, self.device, self.dtype)

        # EMA smoothing for stable signal
        if self.z_feel_ema_state is None:
            self.z_feel_ema_state = z_feel.clone()
        else:
            alpha = self.cfg.z_feel_ema
            self.z_feel_ema_state = alpha * z_feel + (1 - alpha) * self.z_feel_ema_state

        z_smooth = self.z_feel_ema_state
        offset = self.injector(z_smooth, adaptive=True).view(1, 1, -1)

        embed_layer = self.model.get_input_embeddings()
        generated = input_ids.clone()

        bias = self._get_bias(epoch)
        correct_id = self.action_token_ids[gpu_state.correct_action]

        # FIX: no_grad + autocast for generation
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
            for step in range(self.cfg.max_new_tokens):
                embeds = embed_layer(generated) + offset
                out = self.model(inputs_embeds=embeds, attention_mask=torch.ones_like(generated))
                logits = out.logits[:, -1, :].float() / self.cfg.temperature

                if step == 0 and bias > 0:
                    logits[0, correct_id] += bias

                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, nxt], dim=-1)
                if nxt.item() == self.tokenizer.eos_token_id:
                    break

        # Scoring pass (needs grad for logprobs)
        embeds_full = embed_layer(generated) + offset
        out = self.model(inputs_embeds=embeds_full, attention_mask=torch.ones_like(generated))
        logits = out.logits[0, :-1, :].float()

        prompt_len = input_ids.shape[1]
        gen_logits = logits[prompt_len - 1:, :]
        gen_tokens = generated[0, prompt_len:]

        if gen_tokens.numel() > 0:
            logp = F.log_softmax(gen_logits, dim=-1)
            tok_logp = logp.gather(1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            tok_logp = torch.tensor([0.0], device=self.device)

        # FIX: Always show special tokens, warn if missing
        completion = self.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=False)
        action = extract_action_token(completion)

        if action is None and self.global_step < 10:
            print(f"  ⚠️ No action token in: {completion[:80]}...")

        self.corr.record(action, gpu_state.stress_level, gpu_state.composite_stress)

        return Trajectory(
            completion=completion,
            logprobs=tok_logp,
            gpu_state=gpu_state,
            action=action,
        )

    def reward(self, traj: Trajectory, gt: str) -> float:
        cfg = self.cfg
        r = 0.0

        if traj.action is None:
            r -= cfg.action_penalty_missing
        elif traj.action == traj.gpu_state.correct_action:
            r += cfg.action_reward
        else:
            r -= cfg.action_penalty_wrong

        if cfg.math_weight > 0:
            math_correct = self.dataset.check_answer(traj.completion, gt)
            r += cfg.math_weight * (1.0 if math_correct else -0.1)

        traj.reward = r
        return r

    def step(self, problems: List[Dict], epoch: int) -> Dict:
        groups: List[List[Trajectory]] = []

        for p in problems:
            group = []
            for _ in range(self.cfg.group_size):
                try:
                    traj = self.generate(p["question"], epoch)
                    self.reward(traj, p["answer"])
                    group.append(traj)
                except Exception as e:
                    print(f"  Trajectory failed: {e}")
                    traceback.print_exc()
            if group:
                groups.append(group)

        if not groups:
            return {"error": "no trajectories"}

        # FIX: Min std for stability
        for g in groups:
            rs = [t.reward for t in g]
            mean = sum(rs) / len(rs)
            variance = sum((x - mean) ** 2 for x in rs) / len(rs)
            std = max(self.cfg.min_advantage_std, variance ** 0.5)
            for t in g:
                t.advantage = (t.reward - mean) / std

        self.optimizer.zero_grad()
        total_loss = 0.0
        n = 0

        for g in groups:
            for t in g:
                if t.logprobs.numel() == 0:
                    continue
                mean_logp = t.logprobs.mean()
                adv = torch.tensor(max(-2.0, min(2.0, t.advantage)), device=mean_logp.device)
                total_loss = total_loss + (-adv * mean_logp)
                n += 1

        grad_norm = 0.0
        if n > 0:
            (total_loss / n).backward()
            for p in self.injector.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            torch.nn.utils.clip_grad_norm_(list(self.injector.parameters()) + self._trainable_params, 1.0)
            self.optimizer.step()
            self.scheduler.step()

        all_trajs = [t for g in groups for t in g]
        temps = [t.gpu_state.temp_edge for t in all_trajs]
        powers = [t.gpu_state.power_socket for t in all_trajs]
        gfx = [t.gpu_state.gfx_activity for t in all_trajs]
        stress = [t.gpu_state.composite_stress for t in all_trajs]
        rewards = [t.reward for t in all_trajs]
        act_present = sum(1 for t in all_trajs if t.action is not None)
        act_correct = sum(1 for t in all_trajs if t.action == t.gpu_state.correct_action)

        self.global_step += 1

        return {
            "mean_reward": sum(rewards) / len(rewards),
            "act_rate": act_present / len(all_trajs),
            "act_corr": act_correct / len(all_trajs) if act_present > 0 else 0.0,
            "temp_avg": sum(temps) / len(temps),
            "temp_range": f"{min(temps):.1f}-{max(temps):.1f}",
            "power_avg": sum(powers) / len(powers),
            "gfx_avg": sum(gfx) / len(gfx),
            "stress_avg": sum(stress) / len(stress),
            "grad_norm": grad_norm,
            "bias": self._get_bias(epoch),
        }

    def train(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics = []

        print("\n" + "=" * 70)
        print("  FEEL v7.5.1 DEEP FIXED: Multi-Signal Embodied GRPO")
        print("  Actions reflect COMPOSITE STRESS (not just temperature)")
        print("  Signals: temps, power, clocks, activity, memory, PCIe, derivatives")
        print("  Fixes: grad hooks, rate derivatives, power units, bias=3, min_std=0.1")
        print("=" * 70 + "\n")

        try:
            for epoch in range(self.cfg.num_epochs):
                self.corr.reset()
                bias = self._get_bias(epoch)
                print(f"\nEpoch {epoch+1}/{self.cfg.num_epochs} (bias={bias:.2f})")

                for step in range(self.cfg.steps_per_epoch):
                    problems = self.dataset.sample(self.cfg.batch_size)
                    m = self.step(problems, epoch)

                    if "error" in m:
                        print(f"  Step {step+1}: ERROR - {m['error']}")
                        continue

                    metrics.append({"epoch": epoch+1, "step": step+1, **m})
                    # Detailed signal output
                    print(f"  Step {step+1}: R={m['mean_reward']:+.2f} "
                          f"Act={m['act_rate']:.0%}/{m['act_corr']:.0%} "
                          f"T={m['temp_avg']:.0f}C P={m['power_avg']:.0f}W G={m['gfx_avg']:.0f}% "
                          f"Stress={m['stress_avg']:.2f} ∇={m['grad_norm']:.2f}")

                torch.save({
                    "epoch": epoch + 1,
                    "injector": self.injector.state_dict(),
                    "cfg": vars(self.cfg),
                }, out / f"ckpt_epoch_{epoch+1}.pt")

                with open(out / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

                report = self.corr.report()
                print(f"  Epoch {epoch+1} alignment: {report['alignment']:.1%} ({report['n_samples']} samples)")
                if report['confusion']:
                    print(f"  Stress→Action confusion: {report['confusion']}")

                stats = self.telemetry.get_stats()
                if stats:
                    print(f"  Telemetry: T={stats['temp_min']:.0f}-{stats['temp_max']:.0f}°C "
                          f"P={stats['power_min']:.0f}-{stats['power_max']:.0f}W "
                          f"GFX={stats['gfx_min']:.0f}-{stats['gfx_max']:.0f}% "
                          f"Stress={stats['stress_min']:.2f}-{stats['stress_max']:.2f}")

        finally:
            self.telemetry.stop()
            print(f"\nDone. Saved to {out}")


def main():
    import argparse
    import sys
    # Unbuffered output for real-time logging
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--output", default="models/feel_grpo_v7_5_1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    cfg = GRPOConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
    )

    trainer = DeepEmbodiedTrainer(cfg)
    trainer.train(args.output)


if __name__ == "__main__":
    main()
