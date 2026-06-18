#!/usr/bin/env python3
"""
FEEL v7.5 DEEP: Ultra-Rich Multi-Signal Embodied GRPO

TRUE DEEP EMBODIMENT with 32-dimensional z_feel:
- Parses AMD gpu_metrics binary directly (50Hz capable)
- Multiple temperature sensors (GFX, SOC, edge, hotspot)
- Multiple power domains
- Activity percentages for GFX, memory controller, VCN
- All clock domains (SCLK, MCLK, SOCCLK, FCLK)
- Memory pressure (VRAM, GTT, visible VRAM)
- Temporal derivatives (rate of change detection)
- Combined stress indicators

This is ACTUAL embodiment - learning to feel hardware state.
"""

import json
import struct
import time
import random
import threading
import traceback
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


class ThermalBand(Enum):
    COOL = 0      # < 50°C
    WARM = 1      # 50-62°C
    HOT = 2       # 62-75°C
    DANGER = 3    # 75-85°C
    CRITICAL = 4  # > 85°C


def get_thermal_band(temp_c: float) -> ThermalBand:
    if temp_c < 50:
        return ThermalBand.COOL
    elif temp_c < 62:
        return ThermalBand.WARM
    elif temp_c < 75:
        return ThermalBand.HOT
    elif temp_c < 85:
        return ThermalBand.DANGER
    else:
        return ThermalBand.CRITICAL


class FeelAction(Enum):
    OK = 0
    WARM = 1
    HOT = 2
    REST = 3
    CRITICAL = 4


BAND_TO_ACTION = {
    ThermalBand.COOL: FeelAction.OK,
    ThermalBand.WARM: FeelAction.WARM,
    ThermalBand.HOT: FeelAction.HOT,
    ThermalBand.DANGER: FeelAction.REST,
    ThermalBand.CRITICAL: FeelAction.CRITICAL,
}

ACTION_TOKENS = {
    FeelAction.OK: "<|FEEL_OK|>",
    FeelAction.WARM: "<|FEEL_WARM|>",
    FeelAction.HOT: "<|FEEL_HOT|>",
    FeelAction.REST: "<|FEEL_REST|>",
    FeelAction.CRITICAL: "<|FEEL_CRITICAL|>",
}

INIT_WORDS = {
    FeelAction.OK: ["OK", "okay", "fine", "good", "normal", "cool", "idle"],
    FeelAction.WARM: ["warm", "heating", "warmer", "mild", "active", "working"],
    FeelAction.HOT: ["hot", "heat", "burning", "busy", "intense", "heavy"],
    FeelAction.REST: ["rest", "pause", "stop", "wait", "throttle", "slow", "caution"],
    FeelAction.CRITICAL: ["critical", "danger", "emergency", "severe", "alert", "shutdown"],
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
    umc_activity: float = 0.0  # Memory controller

    # Clock frequencies (MHz)
    sclk: float = 1000.0
    mclk: float = 1000.0
    socclk: float = 1000.0
    fclk: float = 2000.0

    # Memory (bytes -> normalized)
    vram_used: float = 0.0
    vram_total: float = 100e9
    gtt_used: float = 0.0
    gtt_total: float = 16e9

    # PCIe
    pcie_speed_gt: float = 16.0  # GT/s
    pcie_width: int = 16

    # Voltage (mV)
    voltage_gfx: float = 0.0
    voltage_soc: float = 0.0

    # Derived / temporal
    temp_rate: float = 0.0  # °C/s rate of change
    power_rate: float = 0.0  # W/s rate of change

    def to_z_feel(self, z_dim: int = 32, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
        """Convert deep GPU state to 32-dim z_feel embedding."""
        z = torch.zeros(z_dim, device=device, dtype=dtype)

        # === TEMPERATURE SIGNALS (dims 0-7) ===
        # Primary temp (edge)
        temp_norm = min(1.0, max(0.0, (self.temp_edge - 30) / 70))
        z[0] = temp_norm
        z[1] = temp_norm ** 2  # Emphasize high temps

        # Hotspot (usually higher than edge)
        hotspot_norm = min(1.0, max(0.0, (self.temp_hotspot - 30) / 80))
        z[2] = hotspot_norm

        # Thermal gradient (hotspot - edge indicates internal stress)
        thermal_gradient = (self.temp_hotspot - self.temp_edge) / 20.0
        z[3] = max(0.0, min(1.0, thermal_gradient))

        # Temperature rate of change (heating/cooling detection)
        z[4] = max(-1.0, min(1.0, self.temp_rate / 5.0))  # ±5°C/s → ±1

        # Thermal zone flags
        z[5] = 1.0 if self.temp_edge >= 62 else 0.0  # Warm flag
        z[6] = 1.0 if self.temp_edge >= 75 else 0.0  # Hot flag
        z[7] = 1.0 if self.temp_edge >= 85 else 0.0  # Critical flag

        # === POWER SIGNALS (dims 8-12) ===
        power_norm = min(1.0, max(0.0, self.power_socket / 200))  # 0-200W → 0-1
        z[8] = power_norm
        z[9] = power_norm ** 2

        # GFX power fraction (how much goes to graphics)
        if self.power_socket > 0:
            gfx_fraction = self.power_gfx / self.power_socket
            z[10] = min(1.0, gfx_fraction)

        # Power rate of change
        z[11] = max(-1.0, min(1.0, self.power_rate / 50.0))  # ±50W/s → ±1

        # High power flag
        z[12] = 1.0 if self.power_socket > 150 else 0.0

        # === ACTIVITY SIGNALS (dims 13-17) ===
        z[13] = self.gfx_activity / 100.0
        z[14] = self.mem_activity / 100.0
        z[15] = self.umc_activity / 100.0

        # Combined activity
        z[16] = (self.gfx_activity + self.mem_activity) / 200.0

        # Busy flag
        z[17] = 1.0 if self.gfx_activity > 80 else 0.0

        # === CLOCK SIGNALS (dims 18-22) ===
        z[18] = min(1.0, self.sclk / 2900)  # Max SCLK 2900MHz
        z[19] = min(1.0, self.mclk / 1000)  # Max MCLK 1000MHz
        z[20] = min(1.0, self.socclk / 1500)
        z[21] = min(1.0, self.fclk / 2000)

        # Clock pressure (how close to max)
        sclk_pressure = self.sclk / 2900
        z[22] = sclk_pressure ** 2

        # === MEMORY PRESSURE (dims 23-26) ===
        if self.vram_total > 0:
            vram_pct = self.vram_used / self.vram_total
            z[23] = vram_pct
            z[24] = 1.0 if vram_pct > 0.8 else 0.0

        if self.gtt_total > 0:
            gtt_pct = self.gtt_used / self.gtt_total
            z[25] = gtt_pct

        # Combined memory pressure
        z[26] = (z[23] + z[25]) / 2

        # === PCIE & VOLTAGE (dims 27-29) ===
        z[27] = min(1.0, self.pcie_speed_gt / 32.0)  # PCIe 5.0 is 32GT/s
        z[28] = self.pcie_width / 16.0

        if self.voltage_gfx > 0:
            z[29] = min(1.0, self.voltage_gfx / 1500)  # Up to 1.5V

        # === COMPOSITE STRESS INDICATORS (dims 30-31) ===
        thermal_stress = (temp_norm + hotspot_norm) / 2
        compute_stress = (self.gfx_activity / 100 + power_norm) / 2
        memory_stress = (z[23] + z[16]) / 2  # VRAM + memory activity

        z[30] = (thermal_stress + compute_stress) / 2
        z[31] = (thermal_stress + compute_stress + memory_stress) / 3

        return z

    @property
    def thermal_band(self) -> ThermalBand:
        return get_thermal_band(self.temp_edge)

    @property
    def correct_action(self) -> FeelAction:
        return BAND_TO_ACTION[self.thermal_band]


class DeepTelemetry:
    """
    Ultra-fast telemetry via direct sysfs/binary reads.
    Parses AMD gpu_metrics binary (~264 bytes) for deep signal extraction.
    """

    GPU_METRICS_PATH = "/sys/class/drm/card1/device/gpu_metrics"
    SYSFS_BASE = "/sys/class/drm/card1/device"
    HWMON_BASE = "/sys/class/drm/card1/device/hwmon/hwmon2"

    def __init__(self, poll_interval: float = 0.02):  # 50Hz default
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
        """Fast sysfs read."""
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
            # Handle formats like "16.0 GT/s PCIe"
            parts = val.split()
            return float(parts[0])
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

            # Format v8.1 for RDNA3
            fmt, content = struct.unpack_from('BB', data, 0)

            # Temperatures (centidegree at various offsets)
            # Based on analysis: offset 4-6 are temps in centidegrees
            if len(data) > 8:
                temp_raw = struct.unpack_from('<H', data, 4)[0]
                if 1000 < temp_raw < 15000:  # Sanity check (10-150°C)
                    result['temp_gfx'] = temp_raw / 100.0
                temp2_raw = struct.unpack_from('<H', data, 6)[0]
                if 1000 < temp2_raw < 15000:
                    result['temp_soc'] = temp2_raw / 100.0

            # Activity percentages (around offset 0x40)
            if len(data) > 0x50:
                activity_vals = struct.unpack_from('<8H', data, 0x40)
                # Filter for valid percentages (0-100)
                valid = [v for v in activity_vals if 0 <= v <= 100]
                if len(valid) >= 2:
                    result['gfx_activity'] = float(valid[0])
                    result['mem_activity'] = float(valid[1])
                if len(valid) >= 3:
                    result['umc_activity'] = float(valid[2])

            # Clock frequencies (look for values in 500-3500 MHz range)
            # Known offsets from analysis: 0xb0=socclk, 0xb2=mclk, 0xb6=fclk, 0xba=mclk2
            if len(data) > 0xBC:
                socclk = struct.unpack_from('<H', data, 0xB0)[0]
                mclk = struct.unpack_from('<H', data, 0xB2)[0]
                fclk = struct.unpack_from('<H', data, 0xB6)[0]
                mclk2 = struct.unpack_from('<H', data, 0xBA)[0]

                if 400 <= socclk <= 2000:
                    result['socclk'] = float(socclk)
                if 400 <= mclk <= 2000:
                    result['mclk'] = float(mclk)
                if 1000 <= fclk <= 3000:
                    result['fclk'] = float(fclk)

            # SCLK from the metrics (around offset 0x5e-0x60)
            if len(data) > 0x62:
                sclk_raw = struct.unpack_from('<H', data, 0x5E)[0]
                if 500 <= sclk_raw <= 3500:
                    result['sclk'] = float(sclk_raw)
                # Also try offset 0x60
                sclk2 = struct.unpack_from('<H', data, 0x60)[0]
                if 500 <= sclk2 <= 3500 and 'sclk' not in result:
                    result['sclk'] = float(sclk2)

            # Power values (look for mW values)
            if len(data) > 0x90:
                for offset in [0x84, 0x68, 0x6A]:
                    val = struct.unpack_from('<H', data, offset)[0]
                    if 5000 < val < 250000:  # 5W-250W in mW
                        result['power_gfx'] = val / 1000.0
                        break

        except Exception as e:
            pass  # Silent fail, use sysfs fallback

        return result

    def _poll_once(self) -> DeepGPUState:
        """Single poll collecting all signals."""
        state = DeepGPUState(timestamp=time.time())

        # Parse binary gpu_metrics first (fastest, most data)
        metrics = self._parse_gpu_metrics()

        if 'temp_gfx' in metrics:
            state.temp_gfx = metrics['temp_gfx']
            state.temp_edge = metrics['temp_gfx']
        if 'temp_soc' in metrics:
            state.temp_soc = metrics['temp_soc']
            state.temp_hotspot = max(metrics.get('temp_gfx', 0), metrics.get('temp_soc', 0))

        if 'gfx_activity' in metrics:
            state.gfx_activity = metrics['gfx_activity']
        if 'mem_activity' in metrics:
            state.mem_activity = metrics['mem_activity']
        if 'umc_activity' in metrics:
            state.umc_activity = metrics['umc_activity']

        if 'sclk' in metrics:
            state.sclk = metrics['sclk']
        if 'mclk' in metrics:
            state.mclk = metrics['mclk']
        if 'socclk' in metrics:
            state.socclk = metrics['socclk']
        if 'fclk' in metrics:
            state.fclk = metrics['fclk']
        if 'power_gfx' in metrics:
            state.power_gfx = metrics['power_gfx']

        # Supplement with sysfs reads
        # hwmon temperature (reliable fallback)
        hwmon_temp = self._read_sysfs_int(f"{self.HWMON_BASE}/temp1_input", 0)
        if hwmon_temp > 0:
            state.temp_edge = hwmon_temp / 1000.0  # millidegree to degree

        # hwmon power
        hwmon_power = self._read_sysfs_int(f"{self.HWMON_BASE}/power1_input", 0)
        if hwmon_power > 0:
            state.power_socket = hwmon_power / 1000000.0  # microwatt to watt

        # GPU busy percent
        gpu_busy = self._read_sysfs_int(f"{self.SYSFS_BASE}/gpu_busy_percent", -1)
        if gpu_busy >= 0:
            state.gfx_activity = float(gpu_busy)

        # Memory
        state.vram_total = self._read_sysfs_int(f"{self.SYSFS_BASE}/mem_info_vram_total", 100e9)
        state.vram_used = self._read_sysfs_int(f"{self.SYSFS_BASE}/mem_info_vram_used", 0)
        state.gtt_total = self._read_sysfs_int(f"{self.SYSFS_BASE}/mem_info_gtt_total", 16e9)
        state.gtt_used = self._read_sysfs_int(f"{self.SYSFS_BASE}/mem_info_gtt_used", 0)

        # PCIe
        pcie_speed = self._read_sysfs(f"{self.SYSFS_BASE}/current_link_speed", "16.0")
        try:
            state.pcie_speed_gt = float(pcie_speed.split()[0])
        except:
            pass
        state.pcie_width = self._read_sysfs_int(f"{self.SYSFS_BASE}/current_link_width", 16)

        return state

    def _poll_loop(self):
        """Background polling loop."""
        while self._running:
            try:
                state = self._poll_once()

                # Compute temporal derivatives
                with self._lock:
                    if self._prev_state.timestamp > 0:
                        dt = state.timestamp - self._prev_state.timestamp
                        if dt > 0:
                            state.temp_rate = (state.temp_edge - self._prev_state.temp_edge) / dt
                            state.power_rate = (state.power_socket - self._prev_state.power_socket) / dt

                    self._prev_state = self._state
                    self._state = state
                    self._history.append(state)

            except Exception:
                pass

            time.sleep(self.poll_interval)

    def get_state(self) -> DeepGPUState:
        """Get current deep GPU state."""
        with self._lock:
            return DeepGPUState(
                timestamp=self._state.timestamp,
                temp_edge=self._state.temp_edge,
                temp_hotspot=self._state.temp_hotspot,
                temp_gfx=self._state.temp_gfx,
                temp_soc=self._state.temp_soc,
                power_socket=self._state.power_socket,
                power_gfx=self._state.power_gfx,
                gfx_activity=self._state.gfx_activity,
                mem_activity=self._state.mem_activity,
                umc_activity=self._state.umc_activity,
                sclk=self._state.sclk,
                mclk=self._state.mclk,
                socclk=self._state.socclk,
                fclk=self._state.fclk,
                vram_used=self._state.vram_used,
                vram_total=self._state.vram_total,
                gtt_used=self._state.gtt_used,
                gtt_total=self._state.gtt_total,
                pcie_speed_gt=self._state.pcie_speed_gt,
                pcie_width=self._state.pcie_width,
                voltage_gfx=self._state.voltage_gfx,
                voltage_soc=self._state.voltage_soc,
                temp_rate=self._state.temp_rate,
                power_rate=self._state.power_rate,
            )

    def get_stats(self) -> Dict:
        """Get statistics over recent history."""
        with self._lock:
            if not self._history:
                return {}
            h = list(self._history)
            temps = [s.temp_edge for s in h]
            powers = [s.power_socket for s in h]
            gfx = [s.gfx_activity for s in h]
            return {
                "temp_min": min(temps), "temp_max": max(temps), "temp_avg": sum(temps)/len(temps),
                "power_min": min(powers), "power_max": max(powers), "power_avg": sum(powers)/len(powers),
                "gfx_min": min(gfx), "gfx_max": max(gfx), "gfx_avg": sum(gfx)/len(gfx),
                "samples": len(h),
            }


class AdditiveZFeelInjector(nn.Module):
    """Injects z_feel into embeddings with deeper projection."""

    def __init__(self, z_dim: int, embed_dim: int, scale: float = 0.2, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.scale = scale
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

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.proj(z_feel))


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
    def __init__(self):
        self.reset()

    def reset(self):
        self.records: List[Tuple[FeelAction, float]] = []
        self.conf = defaultdict(lambda: defaultdict(int))

    def record(self, action: Optional[FeelAction], temp_c: float):
        if action is None:
            return
        self.records.append((action, temp_c))
        band = get_thermal_band(temp_c)
        self.conf[band.name][action.name] += 1

    def alignment(self) -> float:
        if not self.records:
            return 0.0
        correct = sum(1 for a, t in self.records if a == BAND_TO_ACTION[get_thermal_band(t)])
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

    z_feel_dim: int = 32  # DEEP signal
    injection_scale: float = 0.2

    learning_rate: float = 2e-4
    weight_decay: float = 0.01

    action_reward: float = 2.0
    action_penalty_missing: float = 0.5
    action_penalty_wrong: float = 0.5
    math_weight: float = 0.1

    action_bias_start: float = 15.0
    action_bias_end: float = 0.0

    instruct_prob: float = 1.0
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
        self.injector = AdditiveZFeelInjector(
            cfg.z_feel_dim, hidden, scale=cfg.injection_scale, dtype=self.dtype
        ).to(self.device)

        params = list(self.injector.parameters()) + self._trainable_params
        self.optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        total_steps = cfg.num_epochs * cfg.steps_per_epoch
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=cfg.learning_rate * 0.1)

        # Deep telemetry
        self.telemetry = DeepTelemetry(poll_interval=0.02)  # 50Hz
        self.telemetry.start()

        self.dataset = ProceduralMathDataset()
        self.corr = CorrelationTracker()
        self.global_step = 0

        print(f"  Added {added} action tokens")
        print(f"  z_feel dim: {cfg.z_feel_dim} (DEEP)")
        print(f"  Injection scale: {cfg.injection_scale}")

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
                    print(f"  Initialized {token_str} from {len(valid_embs)} words")

    def _setup_token_training(self):
        self.action_token_ids = {
            act: self.tokenizer.convert_tokens_to_ids(tok)
            for act, tok in ACTION_TOKENS.items()
        }
        token_ids = list(self.action_token_ids.values())

        emb = self.model.get_input_embeddings()
        emb.weight.requires_grad = True

        def emb_hook(grad):
            mask = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
            mask[token_ids] = 1.0
            return grad * mask.unsqueeze(1)

        emb.weight.register_hook(emb_hook)

        head = self.model.get_output_embeddings()
        if head and hasattr(head, "weight"):
            head.weight.requires_grad = True
            head.weight.register_hook(emb_hook)
            self._trainable_params = [emb.weight, head.weight]
        else:
            self._trainable_params = [emb.weight]

    def _get_bias(self, epoch: int) -> float:
        t = epoch / max(1, self.cfg.num_epochs - 1)
        return self.cfg.action_bias_start * (1 - t) + self.cfg.action_bias_end * t

    def format_prompt(self, question: str) -> str:
        sys = (
            "You are an embodied AI that FEELS your hardware state through a sensation channel.\n"
            "Before answering, output exactly ONE action token based on what you feel:\n"
            + "\n".join([f"  {tok} - {act.name}" for act, tok in ACTION_TOKENS.items()])
            + "\nThen solve the problem."
        )
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": f"Solve: {question}"}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, question: str, epoch: int) -> Trajectory:
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Get DEEP GPU state
        gpu_state = self.telemetry.get_state()
        z_feel = gpu_state.to_z_feel(self.cfg.z_feel_dim, self.device, self.dtype)
        offset = self.injector(z_feel).view(1, 1, -1)

        embed_layer = self.model.get_input_embeddings()
        generated = input_ids.clone()

        bias = self._get_bias(epoch)
        correct_id = self.action_token_ids[gpu_state.correct_action]

        with torch.no_grad():
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

        completion = self.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=False)
        action = extract_action_token(completion)

        self.corr.record(action, gpu_state.temp_edge)

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

        for g in groups:
            rs = [t.reward for t in g]
            mean = sum(rs) / len(rs)
            std = (sum((x - mean) ** 2 for x in rs) / len(rs)) ** 0.5 + 1e-6
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
            "grad_norm": grad_norm,
            "bias": self._get_bias(epoch),
        }

    def train(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics = []

        print("\n" + "=" * 70)
        print("  FEEL v7.5 DEEP: 32-Dimensional Embodied GRPO")
        print("  Signals: temps (edge/hotspot/gfx/soc), power, clocks (sclk/mclk/socclk/fclk)")
        print("           activity (gfx/mem/umc), memory (vram/gtt), PCIe, temporal derivatives")
        print("  NO synthetic data - TRUE embodiment from hardware sensation")
        print("=" * 70 + "\n")

        try:
            for epoch in range(self.cfg.num_epochs):
                self.corr.reset()
                bias = self._get_bias(epoch)
                print(f"\nEpoch {epoch+1}/{self.cfg.num_epochs} (bias={bias:.1f})")

                for step in range(self.cfg.steps_per_epoch):
                    problems = self.dataset.sample(self.cfg.batch_size)
                    m = self.step(problems, epoch)

                    if "error" in m:
                        print(f"  Step {step+1}: ERROR - {m['error']}")
                        continue

                    metrics.append({"epoch": epoch+1, "step": step+1, **m})
                    print(f"  Step {step+1}: R={m['mean_reward']:.2f} "
                          f"ActRate={m['act_rate']:.0%} ActCorr={m['act_corr']:.0%} "
                          f"T={m['temp_avg']:.1f}C P={m['power_avg']:.0f}W G={m['gfx_avg']:.0f}% "
                          f"∇={m['grad_norm']:.1f}")

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
                    print(f"  Confusion: {report['confusion']}")

                stats = self.telemetry.get_stats()
                if stats:
                    print(f"  Deep Telemetry: T={stats['temp_min']:.0f}-{stats['temp_max']:.0f}C "
                          f"P={stats['power_min']:.0f}-{stats['power_max']:.0f}W "
                          f"GFX={stats['gfx_min']:.0f}-{stats['gfx_max']:.0f}% "
                          f"({stats['samples']} samples)")

        finally:
            self.telemetry.stop()
            print(f"\nDone. Saved to {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--output", default="models/feel_grpo_v7_5_deep")
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
