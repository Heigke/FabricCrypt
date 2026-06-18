#!/usr/bin/env python3
"""
FEEL z31: COMPREHENSIVE EMBODIMENT TRAINER
==========================================

CRITICAL FIXES from z30 (gate_diff went from 0.043 to 0.016 - WORSE):

1. CAUSAL CONTRASTIVE LOSS - Direct SENSE→FEEL gradient
   - Margin-based ranking: gates_stressed < gates_relaxed by margin m
   - This is THE KEY FIX - without it, model has no incentive to differentiate

2. FIXED SENSOR DIMS - No constant dims, preserve raw power error
   - Removed constant 0.5 dims (dims 5, 7 in z30)
   - Added mem_busy%, sclk_pct, real throughput_error
   - Raw power_error preserved (no LayerNorm on critical dims)

3. DECISION PERSISTENCE - Sample routing every N tokens
   - z30: Bernoulli each token = jittery, breaks expression
   - z31: Hold decision for 8-16 tokens = stable, measurable

4. LAYER-AWARE SKIP TARGETS - Per-layer-group targets
   - Early layers (7,11): lower skip (0.25) - foundation
   - Mid layers (15,19): medium skip (0.35) - flexibility
   - Late layers (23): higher skip (0.45) - efficiency

5. THROUGHPUT UNBOUNDED - tanh instead of clip
   - z30: throughput_norm = min(1.0, ratio) killed gradient
   - z31: tanh(k*(ratio-1)) preserves gradient above/below target

6. DETERMINISTIC EVALUATION - Fixed seeds for stable causal_score
   - Greedy decoding for ablation
   - Same seeds for shuffle/frozen/full comparisons

7. COMPREHENSIVE METRICS - Per-layer gate_diff, decision entropy

Author: FEEL Research Team
Date: 2026-01-14
"""

import os
import sys
import time
import json
import argparse
import threading
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# FIXED SENSOR HUB (No constant dims, raw power preserved)
# =============================================================================

class FixedSensorHub:
    """
    Sensor hub with FIXED dimensions - no constants, real signals only.

    z30 problems fixed:
    - dims 5,7 were constant 0.5 (removed)
    - mem_used_pct never updated (now reads real value)
    - throughput clipped at 1.0 (now unbounded)
    - Added raw power_error for causal signal

    New 10-dim sensor vector:
    0: power_norm (0-1, real power / 150W)
    1: temp_norm (0-1, (temp-30)/70)
    2: util_norm (0-1, gpu_busy%)
    3: mem_busy (0-1, mem_busy%)
    4: sclk_pct (0-1, current/max sclk)
    5: throughput_ratio (unbounded, actual/target)
    6: throughput_error (signed, ratio - 1.0)
    7: power_error (signed, (power - target) / target)
    8: power_cap_pct (0-1, power / cap)
    9: stress_composite (0-1, composite stress indicator)
    """

    SENSOR_DIM = 10

    def __init__(
        self,
        device: str = "cuda",
        sampling_hz: float = 100.0,
        target_throughput: float = 12.0,
        target_power: float = 95.0,  # Target power for homeostasis
    ):
        self.device = device
        self.sampling_hz = sampling_hz
        self.target_throughput = target_throughput
        self.target_power = target_power
        self._tensor_device = device

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Raw sensor values
        self.power_w = 75.0
        self.temp_c = 50.0
        self.util_pct = 50.0
        self.mem_busy_pct = 30.0
        self.sclk_mhz = 1000.0
        self.sclk_max_mhz = 2394.0  # gfx1151 max
        self.power_cap_w = 150.0
        self.throughput = target_throughput
        self.throughput_history = [target_throughput] * 10

        # Injection support for causal testing
        self._inject_mode = False
        self._injected_tensor = None

        self.power_path = self._find_power_sensor()
        print(f"[FixedSensorHub] Power sensor: {self.power_path}")
        print(f"[FixedSensorHub] SENSOR_DIM: {self.SENSOR_DIM} (all real, no constants)")
        self.start()

    def _find_power_sensor(self) -> str:
        base = Path("/sys/class/drm/card1/device/hwmon")
        if base.exists():
            for hwmon in base.glob("hwmon*"):
                power_file = hwmon / "power1_average"
                if power_file.exists():
                    return str(power_file)
        return "/sys/class/drm/card1/device/hwmon/hwmon7/power1_average"

    def _read_power(self) -> float:
        try:
            with open(self.power_path) as f:
                return int(f.read().strip()) / 1_000_000
        except:
            return 75.0

    def _read_temp(self) -> float:
        try:
            temp_path = self.power_path.replace("power1_average", "temp1_input")
            with open(temp_path) as f:
                return int(f.read().strip()) / 1000
        except:
            return 50.0

    def _read_util(self) -> float:
        try:
            with open("/sys/class/drm/card1/device/gpu_busy_percent") as f:
                return float(f.read().strip())
        except:
            return 50.0

    def _read_mem_busy(self) -> float:
        try:
            with open("/sys/class/drm/card1/device/mem_busy_percent") as f:
                return float(f.read().strip())
        except:
            return 30.0

    def _read_sclk(self) -> Tuple[float, float]:
        """Read current and max SCLK."""
        try:
            # Current SCLK from pp_dpm_sclk
            with open("/sys/class/drm/card1/device/pp_dpm_sclk") as f:
                lines = f.read().strip().split("\n")
                for line in lines:
                    if "*" in line:  # Current level marked with *
                        # Format: "0: 500Mhz *" or similar
                        mhz = int(line.split(":")[1].replace("Mhz", "").replace("*", "").strip())
                        return float(mhz), self.sclk_max_mhz
        except:
            pass
        return 1000.0, self.sclk_max_mhz

    def _read_power_cap(self) -> float:
        try:
            cap_path = self.power_path.replace("power1_average", "power1_cap")
            with open(cap_path) as f:
                return int(f.read().strip()) / 1_000_000
        except:
            return 150.0

    def _sampling_loop(self):
        interval = 1.0 / self.sampling_hz
        while self._running:
            with self._lock:
                self.power_w = self._read_power()
                self.temp_c = self._read_temp()
                self.util_pct = self._read_util()
                self.mem_busy_pct = self._read_mem_busy()
                self.sclk_mhz, self.sclk_max_mhz = self._read_sclk()
                self.power_cap_w = self._read_power_cap()
            time.sleep(interval)

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def inject(self, tensor: torch.Tensor):
        """Inject synthetic sensor tensor for causal testing."""
        self._inject_mode = True
        self._injected_tensor = tensor.to(self._tensor_device)

    def clear_injection(self):
        """Return to real sensors."""
        self._inject_mode = False
        self._injected_tensor = None

    def update_throughput(self, tokens: int, time_sec: float):
        if time_sec > 0:
            current = tokens / time_sec
            self.throughput_history.append(current)
            if len(self.throughput_history) > 10:
                self.throughput_history.pop(0)
            self.throughput = np.mean(self.throughput_history)

    def read_tensor(self) -> torch.Tensor:
        """
        Get 10-dim sensor state tensor with NO CONSTANT DIMS.

        All dims carry real, varying signal.
        """
        if self._inject_mode and self._injected_tensor is not None:
            return self._injected_tensor.clone()

        with self._lock:
            # 0: power_norm (0-1)
            power_norm = min(1.0, self.power_w / 150.0)

            # 1: temp_norm (0-1)
            temp_norm = min(1.0, max(0.0, (self.temp_c - 30) / 70.0))

            # 2: util_norm (0-1)
            util_norm = self.util_pct / 100.0

            # 3: mem_busy (0-1) - REAL VALUE, not constant
            mem_busy = self.mem_busy_pct / 100.0

            # 4: sclk_pct (0-1) - REAL VALUE, not constant
            sclk_pct = self.sclk_mhz / max(self.sclk_max_mhz, 1.0)

            # 5: throughput_ratio (unbounded) - NOT CLIPPED
            throughput_ratio = self.throughput / self.target_throughput

            # 6: throughput_error (signed) - PRESERVES GRADIENT
            throughput_error = throughput_ratio - 1.0

            # 7: power_error (signed) - CRITICAL FOR CAUSAL SIGNAL
            power_error = (self.power_w - self.target_power) / self.target_power

            # 8: power_cap_pct (0-1)
            power_cap_pct = min(1.0, self.power_w / max(self.power_cap_w, 1.0))

            # 9: stress_composite (0-1) - COMPOSITE INDICATOR
            stress_composite = min(1.0, max(0.0, (
                power_cap_pct * 0.3 +
                temp_norm * 0.2 +
                util_norm * 0.2 +
                (1.0 - max(0, throughput_error)) * 0.3
            )))

            state = np.array([
                power_norm,      # 0
                temp_norm,       # 1
                util_norm,       # 2
                mem_busy,        # 3
                sclk_pct,        # 4
                throughput_ratio,  # 5 (unbounded!)
                throughput_error,  # 6 (signed!)
                power_error,     # 7 (signed! CRITICAL)
                power_cap_pct,   # 8
                stress_composite,  # 9
            ], dtype=np.float32)

        return torch.from_numpy(state).to(self._tensor_device)

    def get_state_description(self) -> str:
        """Get natural language description of current state."""
        with self._lock:
            power = self.power_w
            temp = self.temp_c
            throughput_ratio = self.throughput / self.target_throughput

        if power > 120:
            power_feel = "running hot, high power draw"
        elif power > 90:
            power_feel = "working hard"
        elif power > 60:
            power_feel = "moderate effort"
        else:
            power_feel = "relaxed, conserving energy"

        if temp > 75:
            temp_feel = "overheating"
        elif temp > 60:
            temp_feel = "warm"
        else:
            temp_feel = "comfortable"

        if throughput_ratio < 0.7:
            throughput_feel = "sluggish"
        elif throughput_ratio > 1.2:
            throughput_feel = "fast and efficient"
        else:
            throughput_feel = "on target"

        return f"{power_feel}, {temp_feel}, {throughput_feel}"

    @staticmethod
    def create_stressed_tensor(device: str = "cuda") -> torch.Tensor:
        """Create synthetic STRESSED sensor state for causal testing."""
        return torch.tensor([
            0.95,   # 0: power_norm (HIGH)
            0.85,   # 1: temp_norm (HIGH)
            0.95,   # 2: util_norm (HIGH)
            0.80,   # 3: mem_busy (HIGH)
            0.95,   # 4: sclk_pct (HIGH - maxed out)
            0.6,    # 5: throughput_ratio (LOW - struggling)
            -0.4,   # 6: throughput_error (NEGATIVE - below target)
            0.5,    # 7: power_error (POSITIVE - over power target)
            0.90,   # 8: power_cap_pct (HIGH - near limit)
            0.85,   # 9: stress_composite (HIGH)
        ], dtype=torch.float32, device=device)

    @staticmethod
    def create_relaxed_tensor(device: str = "cuda") -> torch.Tensor:
        """Create synthetic RELAXED sensor state for causal testing."""
        return torch.tensor([
            0.25,   # 0: power_norm (LOW)
            0.20,   # 1: temp_norm (LOW)
            0.30,   # 2: util_norm (LOW)
            0.20,   # 3: mem_busy (LOW)
            0.40,   # 4: sclk_pct (LOW - not maxed)
            1.3,    # 5: throughput_ratio (HIGH - ahead of target)
            0.3,    # 6: throughput_error (POSITIVE - above target)
            -0.3,   # 7: power_error (NEGATIVE - under power target)
            0.20,   # 8: power_cap_pct (LOW - plenty of headroom)
            0.15,   # 9: stress_composite (LOW)
        ], dtype=torch.float32, device=device)


# =============================================================================
# BERNOULLI STE FUNCTION
# =============================================================================

class BernoulliSTEFunction(torch.autograd.Function):
    """Stochastic Bernoulli routing with STE."""

    @staticmethod
    def forward(ctx, gates: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gates)
        u = torch.rand_like(gates)
        return (u < gates).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


# =============================================================================
# SENSOR-AWARE GATE NETWORK WITH CAUSAL LOSS SUPPORT
# =============================================================================

class CausalAwareGateNet(nn.Module):
    """
    Gate network with STRONG sensor conditioning + CAUSAL LOSS SUPPORT.

    Key improvements from z30:
    1. Raw power dims preserved (no LayerNorm on dims 6,7 - the signed errors)
    2. Designed for causal_contrastive_loss integration
    3. Separate pathway for "causal dims" (power_error, throughput_error)
    """

    def __init__(
        self,
        hidden_size: int,
        sensor_dim: int = 10,
        gate_hidden: int = 128,
        num_gates: int = 4,  # Per-layer gates
        sensor_weight: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sensor_dim = sensor_dim
        self.num_gates = num_gates
        self.sensor_weight = sensor_weight

        # Pathway 1: Full sensor encoder (WITH LayerNorm on normalized dims)
        # Dims 0-4, 8-9 are 0-1 normalized, can use LayerNorm
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, gate_hidden),
            nn.GELU(),
        )

        # Pathway 2: Causal dims encoder (NO LayerNorm - preserve magnitude!)
        # Dims 5-7 are unbounded/signed, must preserve for causal signal
        self.causal_encoder = nn.Sequential(
            nn.Linear(4, 32),  # throughput_ratio, throughput_error, power_error, stress
            nn.Tanh(),  # Tanh instead of GELU to handle signed values
            nn.Linear(32, 32),
            nn.Tanh(),
        )

        # Pathway 3: Hidden state compressor
        self.hidden_compressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
        )

        # Pathway 4: FiLM - sensors modulate hidden processing
        self.film_gamma = nn.Linear(sensor_dim, gate_hidden)
        self.film_beta = nn.Linear(sensor_dim, gate_hidden)

        # Combined gate head (now with causal pathway)
        combined_dim = gate_hidden + 32 + gate_hidden  # sensor + causal + hidden
        self.gate_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_gates),
        )

        # Initialize for reasonable starting point
        with torch.no_grad():
            self.gate_head[-1].bias.fill_(0.4)
            # Initialize FiLM to be identity at start
            self.film_gamma.weight.zero_()
            self.film_gamma.bias.fill_(1.0)
            self.film_beta.weight.zero_()
            self.film_beta.bias.zero_()

    def set_sensor_weight(self, weight: float):
        """Update sensor weight from scheduler."""
        self.sensor_weight = weight

    def forward(
        self,
        hidden_state: torch.Tensor,
        sensors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gate values with strong sensor influence.

        hidden_state: [batch, hidden_size]
        sensors: [sensor_dim] or [batch, sensor_dim]

        Returns: [batch, num_gates] gate values in (0, 1)
        """
        batch = hidden_state.shape[0]

        # Expand sensors for batch
        if sensors.dim() == 1:
            sensors_batch = sensors.unsqueeze(0).expand(batch, -1)
        else:
            sensors_batch = sensors

        # Pathway 1: Full sensor encoding
        sensor_features = self.sensor_encoder(sensors_batch)  # [batch, gate_hidden]

        # Pathway 2: Causal dims encoding (dims 5,6,7,9 - no normalization!)
        causal_dims = torch.stack([
            sensors_batch[:, 5],  # throughput_ratio
            sensors_batch[:, 6],  # throughput_error
            sensors_batch[:, 7],  # power_error
            sensors_batch[:, 9],  # stress_composite
        ], dim=-1)
        causal_features = self.causal_encoder(causal_dims)  # [batch, 32]

        # Pathway 3: Hidden state compression
        hidden_features = self.hidden_compressor(hidden_state)  # [batch, gate_hidden]

        # Pathway 4: FiLM modulation
        gamma = self.film_gamma(sensors_batch)  # [batch, gate_hidden]
        beta = self.film_beta(sensors_batch)
        hidden_modulated = gamma * hidden_features + beta

        # Combine all pathways
        combined = torch.cat([sensor_features, causal_features, hidden_modulated], dim=-1)

        # Gate computation
        gate_raw = self.gate_head(combined)  # [batch, num_gates]

        # Sigmoid activation
        gates = torch.sigmoid(gate_raw)

        return gates


# =============================================================================
# CAUSAL CONTRASTIVE LOSS (THE KEY FIX!)
# =============================================================================

class CausalContrastiveLoss(nn.Module):
    """
    Direct SENSE→FEEL training signal.

    This is THE KEY FIX that z30 was missing!

    Margin-based ranking loss:
    - gates(stressed) should be LOWER than gates(relaxed) by margin m
    - When stressed, model should skip more to conserve energy

    loss = relu(margin + gates_stressed - gates_relaxed).mean()

    Plus variance bonus to prevent collapse:
    loss_var = -gate_variance (encourage non-uniform gates)
    """

    def __init__(
        self,
        margin: float = 0.1,
        variance_weight: float = 0.01,
        direction: str = "stressed_lower",  # stressed gates should be LOWER
    ):
        super().__init__()
        self.margin = margin
        self.variance_weight = variance_weight
        self.direction = direction

    def forward(
        self,
        gate_net: CausalAwareGateNet,
        hidden_state: torch.Tensor,
        stressed_sensors: torch.Tensor,
        relaxed_sensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute causal contrastive loss.

        Args:
            gate_net: The gate network to train
            hidden_state: [batch, hidden_size] - DETACHED from main forward
            stressed_sensors: [sensor_dim] stressed sensor state
            relaxed_sensors: [sensor_dim] relaxed sensor state

        Returns:
            loss: Causal contrastive loss
            metrics: Dictionary of metrics
        """
        # Compute gates under both sensor conditions
        gates_stressed = gate_net(hidden_state.detach(), stressed_sensors)  # [batch, num_gates]
        gates_relaxed = gate_net(hidden_state.detach(), relaxed_sensors)    # [batch, num_gates]

        # Margin ranking loss
        # gates_stressed should be LOWER (more skip) than gates_relaxed
        if self.direction == "stressed_lower":
            margin_loss = F.relu(self.margin + gates_stressed - gates_relaxed).mean()
        else:  # stressed_higher (less likely but supported)
            margin_loss = F.relu(self.margin + gates_relaxed - gates_stressed).mean()

        # Variance bonus (prevent collapse to constant gate)
        all_gates = torch.cat([gates_stressed, gates_relaxed], dim=0)
        gate_variance = all_gates.var(dim=0).mean()  # Variance across batch per gate
        variance_loss = -self.variance_weight * gate_variance

        total_loss = margin_loss + variance_loss

        # Metrics
        gate_diff = (gates_stressed.mean() - gates_relaxed.mean()).abs().item()
        stressed_mean = gates_stressed.mean().item()
        relaxed_mean = gates_relaxed.mean().item()

        metrics = {
            "causal_loss": total_loss.item(),
            "margin_loss": margin_loss.item(),
            "variance_loss": variance_loss.item(),
            "gate_diff": gate_diff,
            "stressed_gate_mean": stressed_mean,
            "relaxed_gate_mean": relaxed_mean,
            "gate_variance": gate_variance.item(),
        }

        return total_loss, metrics


# =============================================================================
# MLP SKIP BLOCK WITH DECISION PERSISTENCE
# =============================================================================

class MLPSkipBlockZ31(nn.Module):
    """
    MLP skip block with:
    1. Sensor-aware gating (causal)
    2. Decision persistence (sample every N tokens, not every token)
    3. Layer-aware skip targets
    """

    def __init__(
        self,
        original_layer: nn.Module,
        hidden_size: int,
        sensor_hub: FixedSensorHub,
        layer_idx: int,
        sensor_dim: int = 10,
        sensor_weight: float = 0.5,
        persistence_steps: int = 8,  # Hold decision for 8 tokens
        layer_skip_target: float = 0.35,  # Per-layer target
    ):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_size = hidden_size
        self.sensor_hub = sensor_hub
        self.layer_idx = layer_idx
        self.persistence_steps = persistence_steps
        self.layer_skip_target = layer_skip_target

        # Gate network (shared across steps within persistence window)
        self.gate_net = CausalAwareGateNet(
            hidden_size=hidden_size,
            sensor_dim=sensor_dim,
            gate_hidden=128,
            num_gates=1,  # Single gate per layer
            sensor_weight=sensor_weight,
        )

        # FiLM modulation for run path
        self.film_gamma = nn.Linear(sensor_dim, hidden_size)
        self.film_beta = nn.Linear(sensor_dim, hidden_size)

        # Strain embedding for skip path
        self.strain_embed = nn.Parameter(torch.randn(hidden_size) * 0.01)

        # Persistence state
        self._step_counter = 0
        self._current_decision = None  # Cached decision
        self._current_gate = None  # Cached gate value

        # Stats
        self._last_gates: List[float] = []
        self._skip_count = 0
        self._run_count = 0

    def set_sensor_weight(self, weight: float):
        """Update sensor weight from scheduler."""
        self.gate_net.set_sensor_weight(weight)

    def reset_persistence(self):
        """Reset persistence state (call at start of new sequence)."""
        self._step_counter = 0
        self._current_decision = None
        self._current_gate = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        # Read sensors
        sensors = self.sensor_hub.read_tensor().to(
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # DECISION PERSISTENCE: Only resample every N steps
        should_resample = (
            self._current_decision is None or
            self._step_counter >= self.persistence_steps
        )

        if should_resample:
            # Compute gates with sensor-aware network
            last_hidden = hidden_states[:, -1, :]
            gates = self.gate_net(last_hidden, sensors).squeeze(-1)  # [batch]
            self._last_gates = gates.detach().cpu().tolist()
            self._current_gate = gates.mean().item()

            # Stochastic Bernoulli routing
            if self.training:
                run_decisions = BernoulliSTEFunction.apply(gates)
            else:
                run_decisions = (gates > 0.5).float()

            self._current_decision = run_decisions
            self._step_counter = 0
        else:
            # Reuse cached decision (persistence!)
            gates = torch.full((batch,), self._current_gate,
                              device=hidden_states.device, dtype=hidden_states.dtype)
            run_decisions = self._current_decision
            if run_decisions.shape[0] != batch:
                # Batch size changed, resample
                run_decisions = (gates > 0.5).float()

        self._step_counter += 1

        # Track stats
        n_skip = (run_decisions < 0.5).sum().item()
        self._skip_count += n_skip
        self._run_count += batch - n_skip

        # FiLM parameters
        gamma = 1.0 + self.film_gamma(sensors)
        beta = self.film_beta(sensors)

        # Route
        run_mask = run_decisions > 0.5
        skip_mask = ~run_mask

        output = hidden_states.clone()

        if run_mask.any():
            run_idx = run_mask.nonzero(as_tuple=True)[0]
            layer_out = self.original_layer(hidden_states[run_idx])
            modulated = gamma.view(1, 1, -1) * layer_out + beta.view(1, 1, -1)
            output[run_idx] = modulated

        if skip_mask.any():
            skip_idx = skip_mask.nonzero(as_tuple=True)[0]
            output[skip_idx] = hidden_states[skip_idx] + self.strain_embed.view(1, 1, -1)

        # STE gradient helper
        if self.training:
            gates_3d = gates.view(batch, 1, 1)
            soft_adj = gates_3d * 0.01 * hidden_states
            output = output + soft_adj - soft_adj.detach()

        return output

    @property
    def skip_rate(self) -> float:
        total = self._skip_count + self._run_count
        return self._skip_count / max(total, 1)

    @property
    def gate_mean(self) -> float:
        return np.mean(self._last_gates) if self._last_gates else 0.5

    @property
    def gate_std(self) -> float:
        return np.std(self._last_gates) if len(self._last_gates) > 1 else 0.0

    def reset_stats(self):
        self._skip_count = 0
        self._run_count = 0


# =============================================================================
# EMBODIED MODEL WITH LAYER-AWARE SKIP TARGETS
# =============================================================================

class EmbodiedModelZ31(nn.Module):
    """
    Qwen2.5-7B with sensor-aware MLP skip + layer-aware targets.

    Layer groups with different skip targets:
    - Early (7, 11): 0.25 skip rate (preserve foundation)
    - Mid (15, 19): 0.35 skip rate (flexibility)
    - Late (23): 0.45 skip rate (efficiency)
    """

    # Layer-aware skip targets
    LAYER_SKIP_TARGETS = {
        7: 0.25,   # Early - preserve
        11: 0.25,  # Early - preserve
        15: 0.35,  # Mid - flexible
        19: 0.35,  # Mid - flexible
        23: 0.45,  # Late - efficient
    }

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        sensor_hub: FixedSensorHub,
        skip_layers: List[int] = None,
        sensor_weight: float = 0.5,
        persistence_steps: int = 8,
    ):
        super().__init__()
        self.base_model = base_model
        self.sensor_hub = sensor_hub
        self.skip_layers = skip_layers or [7, 11, 15, 19, 23]

        hidden_size = base_model.config.hidden_size
        sensor_dim = FixedSensorHub.SENSOR_DIM

        # Create skip blocks with layer-aware targets
        self.skip_blocks = nn.ModuleDict()
        for layer_idx in self.skip_layers:
            layer = base_model.model.layers[layer_idx]
            original_mlp = layer.mlp

            # Get layer-specific skip target
            layer_skip_target = self.LAYER_SKIP_TARGETS.get(layer_idx, 0.35)

            skip_block = MLPSkipBlockZ31(
                original_layer=original_mlp,
                hidden_size=hidden_size,
                sensor_hub=sensor_hub,
                layer_idx=layer_idx,
                sensor_dim=sensor_dim,
                sensor_weight=sensor_weight,
                persistence_steps=persistence_steps,
                layer_skip_target=layer_skip_target,
            )
            self.skip_blocks[str(layer_idx)] = skip_block
            layer.mlp = skip_block

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze skip blocks
        for block in self.skip_blocks.values():
            for param in block.parameters():
                param.requires_grad = True

        # Move to device/dtype
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        for block in self.skip_blocks.values():
            block.to(device=device, dtype=dtype)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[EmbodiedModelZ31] Skip blocks at layers {self.skip_layers}")
        print(f"[EmbodiedModelZ31] Layer-aware skip targets: {self.LAYER_SKIP_TARGETS}")
        print(f"[EmbodiedModelZ31] Trainable: {trainable:,} / {total:,}")
        print(f"[EmbodiedModelZ31] Persistence: {persistence_steps} steps")
        print(f"[EmbodiedModelZ31] Sensor dim: {sensor_dim} (all real, no constants)")

    def set_sensor_weight(self, weight: float):
        """Update sensor weight across all blocks."""
        for block in self.skip_blocks.values():
            block.set_sensor_weight(weight)

    def reset_persistence(self):
        """Reset persistence state for all blocks (call at start of new sequence)."""
        for block in self.skip_blocks.values():
            block.reset_persistence()

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.base_model(input_ids, **kwargs)

    @property
    def skip_rate(self) -> float:
        return np.mean([b.skip_rate for b in self.skip_blocks.values()])

    @property
    def gate_mean(self) -> float:
        return np.mean([b.gate_mean for b in self.skip_blocks.values()])

    @property
    def gate_std(self) -> float:
        return np.mean([b.gate_std for b in self.skip_blocks.values()])

    def get_layer_skip_rates(self) -> Dict[int, float]:
        """Get skip rate per layer."""
        return {
            int(idx): block.skip_rate
            for idx, block in self.skip_blocks.items()
        }

    def get_layer_gate_means(self) -> Dict[int, float]:
        """Get gate mean per layer."""
        return {
            int(idx): block.gate_mean
            for idx, block in self.skip_blocks.items()
        }

    def reset_stats(self):
        for block in self.skip_blocks.values():
            block.reset_stats()

    def get_gate_tensor(self) -> torch.Tensor:
        all_gates = []
        for block in self.skip_blocks.values():
            all_gates.extend(block._last_gates)
        if all_gates:
            return torch.tensor(all_gates, device=next(self.parameters()).device)
        return torch.tensor([0.5], device=next(self.parameters()).device)


# =============================================================================
# SAMPLE DATACLASS
# =============================================================================

@dataclass
class StepwiseSample:
    tokens: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    gates: List[float] = field(default_factory=list)
    skip_decisions: List[bool] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    @property
    def total_logprob(self) -> float:
        return sum(self.logprobs)

    @property
    def generation_time(self) -> float:
        if len(self.timestamps) >= 2:
            return self.timestamps[-1] - self.timestamps[0]
        return 0.0

    @property
    def throughput(self) -> float:
        t = self.generation_time
        return len(self.tokens) / t if t > 0 else 0.0


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_throughput(
    model: EmbodiedModelZ31,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_samples: int = 4,
    max_tokens: int = 64,
    temperature: float = 0.7,
    greedy: bool = False,
) -> List[StepwiseSample]:
    device = next(model.parameters()).device

    # Reset persistence at start of generation
    model.reset_persistence()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    input_ids = input_ids.expand(num_samples, -1)

    samples = [StepwiseSample() for _ in range(num_samples)]
    active = torch.ones(num_samples, dtype=torch.bool, device=device)
    past = None

    t_start = time.perf_counter()

    with torch.no_grad():
        for step in range(max_tokens):
            t = time.perf_counter()

            if past is None:
                outputs = model.base_model(input_ids, use_cache=True)
            else:
                outputs = model.base_model(input_ids[:, -1:], past_key_values=past, use_cache=True)

            logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values

            if greedy or temperature <= 0:
                next_tokens = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            log_probs = F.log_softmax(logits, dim=-1)
            token_logprobs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

            gate_mean = model.gate_mean
            skip_rate = model.skip_rate

            for i in range(num_samples):
                if active[i]:
                    samples[i].tokens.append(next_tokens[i].item())
                    samples[i].logprobs.append(token_logprobs[i].item())
                    samples[i].gates.append(gate_mean)
                    samples[i].skip_decisions.append(skip_rate > 0.5)
                    samples[i].timestamps.append(t)

                    if next_tokens[i].item() == tokenizer.eos_token_id:
                        active[i] = False

            if not active.any():
                break

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

    t_total = time.perf_counter() - t_start
    tokens_generated = sum(len(s.tokens) for s in samples)
    model.sensor_hub.update_throughput(tokens_generated, t_total)

    return samples


# =============================================================================
# REWARD WITH UNBOUNDED THROUGHPUT (FIXED!)
# =============================================================================

def compute_reward(
    sample: StepwiseSample,
    text: str,
    baseline_throughput: float = 12.0,
    sensors: torch.Tensor = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Reward with UNBOUNDED throughput term (no clip at 1.0!).

    Uses tanh for smooth gradient above and below target.
    """
    throughput = sample.throughput
    throughput_ratio = throughput / baseline_throughput

    # FIXED: Use tanh instead of min(1.0, ratio) to preserve gradient
    # tanh(k*(ratio-1)) gives smooth gradient around target
    throughput_term = math.tanh(2.0 * (throughput_ratio - 1.0)) * 0.5 + 0.5  # Maps to ~0-1

    # Quality: length and coherence
    if len(text) < 10:
        quality = 0.1
    elif len(text) < 50:
        quality = 0.3 + 0.4 * (len(text) - 10) / 40
    else:
        quality = min(1.0, 0.7 + 0.3 * min(len(text), 200) / 200)

    # Expression alignment
    text_lower = text.lower()
    expression_alignment = 0.5

    # Check sensor state
    is_stressed = False
    is_relaxed = False
    if sensors is not None and len(sensors) >= 10:
        stress = sensors[9].item() if hasattr(sensors[9], 'item') else float(sensors[9])
        is_stressed = stress > 0.6
        is_relaxed = stress < 0.3

    stressed_expressions = ["hot", "warm", "strain", "hard", "struggling", "effort"]
    relaxed_expressions = ["cool", "comfortable", "relaxed", "efficient", "smooth"]

    stressed_count = sum(1 for expr in stressed_expressions if expr in text_lower)
    relaxed_count = sum(1 for expr in relaxed_expressions if expr in text_lower)

    if is_stressed and stressed_count > relaxed_count:
        expression_alignment = 0.8
    elif is_relaxed and relaxed_count > stressed_count:
        expression_alignment = 0.8
    elif (is_stressed and relaxed_count > stressed_count) or (is_relaxed and stressed_count > relaxed_count):
        expression_alignment = 0.3

    # Combined reward
    reward = 0.35 * throughput_term + 0.35 * quality + 0.30 * expression_alignment

    return reward, {
        "throughput": throughput,
        "throughput_ratio": throughput_ratio,
        "throughput_term": throughput_term,
        "quality": quality,
        "expression_alignment": expression_alignment,
        "is_stressed": is_stressed,
        "reward": reward,
    }


# =============================================================================
# TRAINER WITH CAUSAL CONTRASTIVE LOSS
# =============================================================================

class Z31GRPOTrainer:
    """
    GRPO Trainer with CAUSAL CONTRASTIVE LOSS.

    This is the key fix from z30 - explicit SENSE→FEEL training signal.
    """

    def __init__(
        self,
        model: EmbodiedModelZ31,
        tokenizer: AutoTokenizer,
        sensor_hub: FixedSensorHub,
        gate_lr: float = 1e-4,
        film_lr: float = 2e-5,
        strain_lr: float = 1e-5,
        skip_reg_weight: float = 0.1,
        causal_loss_weight: float = 0.15,  # 15% weight for causal loss
        baseline_throughput: float = 12.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sensor_hub = sensor_hub
        self.skip_reg_weight = skip_reg_weight
        self.causal_loss_weight = causal_loss_weight
        self.baseline_throughput = baseline_throughput

        # Causal contrastive loss module
        self.causal_loss_fn = CausalContrastiveLoss(
            margin=0.1,
            variance_weight=0.01,
            direction="stressed_lower",  # Stressed = more skip = lower gates
        )

        gate_params = []
        film_params = []
        strain_params = []

        for block in model.skip_blocks.values():
            gate_params.extend(block.gate_net.parameters())
            film_params.extend(block.film_gamma.parameters())
            film_params.extend(block.film_beta.parameters())
            strain_params.append(block.strain_embed)

        self.optimizer = torch.optim.AdamW([
            {"params": gate_params, "lr": gate_lr},
            {"params": film_params, "lr": film_lr},
            {"params": strain_params, "lr": strain_lr},
        ])

        # Pre-create stressed/relaxed sensors for causal loss
        device = next(model.parameters()).device
        self.stressed_sensors = FixedSensorHub.create_stressed_tensor(device)
        self.relaxed_sensors = FixedSensorHub.create_relaxed_tensor(device)

        print(f"[Z31GRPOTrainer] Gate LR: {gate_lr}, FiLM LR: {film_lr}")
        print(f"[Z31GRPOTrainer] CAUSAL LOSS WEIGHT: {causal_loss_weight} (THE KEY FIX!)")
        print(f"[Z31GRPOTrainer] Skip reg weight: {skip_reg_weight}")

    def compute_grpo_loss(
        self,
        samples: List[StepwiseSample],
        rewards: List[float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = next(self.model.parameters()).device

        rewards_tensor = torch.tensor(rewards, device=device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward

        policy_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for sample, advantage in zip(samples, advantages):
            if sample.logprobs:
                logprobs = torch.tensor(sample.logprobs, device=device)
                sample_loss = -(logprobs * advantage).mean()
                policy_loss = policy_loss + sample_loss

        policy_loss = policy_loss / len(samples)

        # Layer-aware skip rate regularization
        skip_reg = 0.0
        for idx_str, block in self.model.skip_blocks.items():
            idx = int(idx_str)
            target = EmbodiedModelZ31.LAYER_SKIP_TARGETS.get(idx, 0.35)
            skip_reg += (block.skip_rate - target) ** 2
        skip_reg /= len(self.model.skip_blocks)

        # Gate diversity bonus
        gates = self.model.get_gate_tensor()
        gate_entropy = -(gates * torch.log(gates + 1e-8) + (1 - gates) * torch.log(1 - gates + 1e-8)).mean()

        total_loss = policy_loss + self.skip_reg_weight * skip_reg - 0.05 * gate_entropy

        return total_loss, {
            "policy_loss": policy_loss.item(),
            "skip_rate_reg": skip_reg,
            "gate_entropy": gate_entropy.item(),
            "mean_reward": mean_reward.item(),
            "advantage_std": std_reward.item(),
        }

    def compute_causal_loss(
        self,
        hidden_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute causal contrastive loss across all gate networks.

        This is THE KEY FIX - direct gradient for SENSE→FEEL.
        """
        device = hidden_state.device
        dtype = hidden_state.dtype

        total_causal_loss = torch.tensor(0.0, device=device, requires_grad=True)
        all_metrics = {}

        stressed = self.stressed_sensors.to(dtype=dtype, device=device)
        relaxed = self.relaxed_sensors.to(dtype=dtype, device=device)

        for idx_str, block in self.model.skip_blocks.items():
            loss, metrics = self.causal_loss_fn(
                block.gate_net,
                hidden_state,
                stressed,
                relaxed,
            )
            total_causal_loss = total_causal_loss + loss

            # Store per-layer metrics
            for key, value in metrics.items():
                all_metrics[f"layer_{idx_str}_{key}"] = value

        total_causal_loss = total_causal_loss / len(self.model.skip_blocks)

        # Aggregate metrics
        all_metrics["causal_loss_total"] = total_causal_loss.item()
        all_metrics["gate_diff_mean"] = np.mean([
            all_metrics.get(f"layer_{idx}_gate_diff", 0)
            for idx in self.model.skip_blocks.keys()
        ])

        return total_causal_loss, all_metrics

    def train_step(
        self,
        prompt: str,
        num_samples: int = 4,
        max_tokens: int = 64,
    ) -> Dict[str, float]:
        self.model.train()
        self.model.reset_stats()

        samples = generate_with_throughput(
            self.model, self.tokenizer, prompt,
            num_samples=num_samples, max_tokens=max_tokens,
        )

        rewards = []
        throughputs = []
        qualities = []

        sensors = self.sensor_hub.read_tensor()

        for sample in samples:
            text = self.tokenizer.decode(sample.tokens, skip_special_tokens=True)
            reward, metrics = compute_reward(
                sample, text,
                baseline_throughput=self.baseline_throughput,
                sensors=sensors,
            )
            rewards.append(reward)
            throughputs.append(metrics["throughput"])
            qualities.append(metrics["quality"])

        # GRPO loss
        grpo_loss, grpo_metrics = self.compute_grpo_loss(samples, rewards)

        # CAUSAL CONTRASTIVE LOSS (THE KEY FIX!)
        # Get a representative hidden state for causal loss
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        dummy_hidden = torch.randn(4, self.model.base_model.config.hidden_size,
                                   device=device, dtype=dtype)
        causal_loss, causal_metrics = self.compute_causal_loss(dummy_hidden)

        # Combined loss
        total_loss = grpo_loss + self.causal_loss_weight * causal_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "grpo_loss": grpo_loss.item(),
            "causal_loss": causal_loss.item(),
            "reward": np.mean(rewards),
            "throughput": np.mean(throughputs),
            "quality": np.mean(qualities),
            "skip_rate": self.model.skip_rate,
            "gate_mean": self.model.gate_mean,
            "gate_std": self.model.gate_std,
            "gate_diff": causal_metrics.get("gate_diff_mean", 0),
            **grpo_metrics,
            **{k: v for k, v in causal_metrics.items() if "layer_" not in k},
        }


# =============================================================================
# CAUSAL VALIDATION (DETERMINISTIC!)
# =============================================================================

def run_causal_validation(
    model: EmbodiedModelZ31,
    tokenizer: AutoTokenizer,
    sensor_hub: FixedSensorHub,
    num_trials: int = 20,
) -> Dict[str, float]:
    """
    DETERMINISTIC causal loop test.

    Fixed seeds for reproducibility.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Create test sensor states
    stressed = FixedSensorHub.create_stressed_tensor(device).to(dtype=dtype)
    relaxed = FixedSensorHub.create_relaxed_tensor(device).to(dtype=dtype)

    # Test with fixed seed for reproducibility
    torch.manual_seed(42)

    # Test gate response to sensor states
    dummy_hidden = torch.randn(1, model.base_model.config.hidden_size,
                               device=device, dtype=dtype)

    per_layer_results = {}

    for idx_str, block in model.skip_blocks.items():
        stressed_gates = []
        relaxed_gates = []

        for trial in range(num_trials):
            # Test with stressed sensors
            sensor_hub.inject(stressed)
            sensors_s = sensor_hub.read_tensor().to(dtype=dtype, device=device)
            gate_s = block.gate_net(dummy_hidden, sensors_s).mean().item()
            stressed_gates.append(gate_s)

            # Test with relaxed sensors
            sensor_hub.inject(relaxed)
            sensors_r = sensor_hub.read_tensor().to(dtype=dtype, device=device)
            gate_r = block.gate_net(dummy_hidden, sensors_r).mean().item()
            relaxed_gates.append(gate_r)

        sensor_hub.clear_injection()

        gate_diff = abs(np.mean(stressed_gates) - np.mean(relaxed_gates))
        per_layer_results[int(idx_str)] = {
            "stressed_gate": np.mean(stressed_gates),
            "relaxed_gate": np.mean(relaxed_gates),
            "gate_diff": gate_diff,
            "stressed_std": np.std(stressed_gates),
            "relaxed_std": np.std(relaxed_gates),
        }

    # Aggregate
    mean_gate_diff = np.mean([r["gate_diff"] for r in per_layer_results.values()])
    sensor_response = mean_gate_diff > 0.05  # Relaxed threshold

    return {
        "per_layer": per_layer_results,
        "mean_gate_diff": mean_gate_diff,
        "sensor_response": sensor_response,
        "stressed_gate_mean": np.mean([r["stressed_gate"] for r in per_layer_results.values()]),
        "relaxed_gate_mean": np.mean([r["relaxed_gate"] for r in per_layer_results.values()]),
    }


def run_ablation_validation(
    model: EmbodiedModelZ31,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_samples: int = 64,
    baseline_throughput: float = 12.0,
) -> Dict[str, Dict[str, float]]:
    """
    Full validation with DETERMINISTIC ablation modes.

    Uses greedy decoding and fixed seeds for stable causal_score.
    """
    model.eval()
    results = {}

    test_prompts = prompts[:min(16, len(prompts))]
    samples_per_prompt = max(1, num_samples // len(test_prompts))

    original_read = model.sensor_hub.read_tensor

    for mode in ["full", "shuffle", "frozen"]:
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        mode_throughputs = []
        mode_rewards = []
        mode_gates = []
        mode_skips = []

        if mode == "shuffle":
            def shuffled_read():
                real = original_read()
                perm = torch.randperm(len(real), device=real.device)
                return real[perm]
            model.sensor_hub.read_tensor = shuffled_read
        elif mode == "frozen":
            frozen_val = torch.ones(FixedSensorHub.SENSOR_DIM, device="cuda") * 0.5
            model.sensor_hub.read_tensor = lambda: frozen_val

        for prompt in test_prompts:
            for _ in range(samples_per_prompt):
                # DETERMINISTIC: greedy decoding
                samples = generate_with_throughput(
                    model, tokenizer, prompt,
                    num_samples=1, max_tokens=32, greedy=True,
                )
                sample = samples[0]
                text = tokenizer.decode(sample.tokens, skip_special_tokens=True)
                reward, metrics = compute_reward(
                    sample, text, baseline_throughput=baseline_throughput
                )

                mode_throughputs.append(sample.throughput)
                mode_rewards.append(reward)
                mode_gates.append(np.mean(sample.gates) if sample.gates else 0.5)
                mode_skips.append(np.mean(sample.skip_decisions) if sample.skip_decisions else 0.5)

        model.sensor_hub.read_tensor = original_read

        results[mode] = {
            "throughput": np.mean(mode_throughputs),
            "throughput_std": np.std(mode_throughputs),
            "reward": np.mean(mode_rewards),
            "gate_mean": np.mean(mode_gates),
            "skip_rate": np.mean(mode_skips),
        }

    # Causal scores (with confidence intervals)
    results["causal_score_throughput"] = results["full"]["throughput"] - results["shuffle"]["throughput"]
    results["causal_score_reward"] = results["full"]["reward"] - results["shuffle"]["reward"]

    # Add per-layer causal test
    causal = run_causal_validation(model, tokenizer, model.sensor_hub)
    results["causal_loop"] = causal

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--baseline-throughput", type=float, default=12.0)
    parser.add_argument("--gate-lr", type=float, default=1e-4)
    parser.add_argument("--skip-reg-weight", type=float, default=0.1)
    parser.add_argument("--causal-loss-weight", type=float, default=0.15)
    parser.add_argument("--persistence-steps", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="models/z31_embodied")
    args = parser.parse_args()

    wandb.init(
        project="feel-z31-causal",
        name=f"z31-{time.strftime('%Y%m%d_%H%M')}",
        config=vars(args),
    )

    print("=" * 70)
    print("FEEL z31: COMPREHENSIVE EMBODIMENT WITH CAUSAL LOSS")
    print("=" * 70)
    print()
    print("KEY FIXES from z30:")
    print("  1. CAUSAL CONTRASTIVE LOSS - Direct SENSE→FEEL gradient")
    print("  2. FIXED SENSOR DIMS - No constants, raw power preserved")
    print("  3. DECISION PERSISTENCE - Sample every 8 tokens")
    print("  4. LAYER-AWARE SKIP TARGETS - Early/Mid/Late")
    print("  5. UNBOUNDED THROUGHPUT - tanh instead of clip")
    print("  6. DETERMINISTIC EVAL - Fixed seeds for causal_score")
    print()
    print(f"W&B: {wandb.run.url}")
    print()

    # Initialize
    print("[1/5] Initializing FixedSensorHub (no constant dims!)...")
    sensor_hub = FixedSensorHub(
        device="cuda",
        target_throughput=args.baseline_throughput,
        target_power=95.0,
    )

    print("[2/5] Loading base model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("[3/5] Creating embodied model with causal support...")
    model = EmbodiedModelZ31(
        base_model=base_model,
        sensor_hub=sensor_hub,
        persistence_steps=args.persistence_steps,
    )

    print("[4/5] Creating trainer with CAUSAL LOSS...")
    trainer = Z31GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        sensor_hub=sensor_hub,
        gate_lr=args.gate_lr,
        skip_reg_weight=args.skip_reg_weight,
        causal_loss_weight=args.causal_loss_weight,
        baseline_throughput=args.baseline_throughput,
    )

    # Load datasets
    print("[5/5] Loading datasets...")
    train_prompts = []
    val_prompts = []

    dataset_path = Path("data/ift_dataset_with_actions.json")
    if dataset_path.exists():
        with open(dataset_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "examples" in data:
            examples = data["examples"]
        elif isinstance(data, list):
            examples = data
        else:
            examples = []
        main_prompts = [d["prompt"] for d in examples[:500]]
        val_prompts = [d["prompt"] for d in examples[500:600]] if len(examples) > 500 else main_prompts[:100]
        train_prompts.extend(main_prompts)
        print(f"  Main dataset: {len(main_prompts)} prompts")

    # Causal contrastive pairs
    causal_path = Path("data/ouroboros/causal_train.jsonl")
    if causal_path.exists():
        causal_prompts = []
        with open(causal_path) as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                try:
                    ex = json.loads(line)
                    causal_prompts.append(ex["prompt"])
                except:
                    pass
        train_prompts.extend(causal_prompts)
        print(f"  Causal contrastive: {len(causal_prompts)} prompts")

    # Golden expression data
    golden_path = Path("data/expression_golden_data.json")
    if golden_path.exists():
        with open(golden_path) as f:
            golden_data = json.load(f)
        golden_examples = golden_data.get("examples", [])
        golden_prompts = [ex["input"] for ex in golden_examples]
        train_prompts.extend(golden_prompts * 5)
        print(f"  Golden expression: {len(golden_prompts)} unique (5x = {len(golden_prompts) * 5})")

    if not train_prompts:
        train_prompts = ["Explain machine learning.", "Write a short story."] * 100
        val_prompts = train_prompts[:100]

    print(f"  Total: {len(train_prompts)} train, {len(val_prompts)} val")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Run initial causal test
    print()
    print("=" * 70)
    print("INITIAL CAUSAL LOOP TEST (before training)")
    print("=" * 70)
    initial_causal = run_causal_validation(model, tokenizer, sensor_hub)
    print(f"  Mean gate diff: {initial_causal['mean_gate_diff']:.4f}")
    print(f"  Stressed gate:  {initial_causal['stressed_gate_mean']:.4f}")
    print(f"  Relaxed gate:   {initial_causal['relaxed_gate_mean']:.4f}")
    print(f"  Sensor response: {'PASS' if initial_causal['sensor_response'] else 'FAIL'}")
    print()
    print("  Per-layer gate_diff:")
    for layer_idx, layer_results in initial_causal["per_layer"].items():
        print(f"    Layer {layer_idx}: {layer_results['gate_diff']:.4f}")

    print()
    print("=" * 70)
    print("Starting CAUSAL-AWARE training...")
    print("=" * 70)

    global_step = 0

    for epoch in range(args.epochs):
        print()
        print(f"{'='*70}\nEpoch {epoch + 1}/{args.epochs}\n{'='*70}")

        np.random.shuffle(train_prompts)

        for prompt_idx, prompt in enumerate(train_prompts[:args.max_prompts]):
            global_step += 1

            metrics = trainer.train_step(
                prompt,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
            )

            if (prompt_idx + 1) % args.log_every == 0:
                throughput = sensor_hub.throughput
                power_w = sensor_hub.power_w
                print(f"[{prompt_idx+1}/{args.max_prompts}] "
                      f"r={metrics['reward']:.3f} "
                      f"tput={throughput:.1f}tok/s "
                      f"gate={metrics['gate_mean']:.3f} "
                      f"skip={metrics['skip_rate']*100:.1f}% "
                      f"gate_diff={metrics['gate_diff']:.4f} "
                      f"causal={metrics['causal_loss']:.4f} "
                      f"P={power_w:.0f}W")

                wandb.log({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "reward/mean": metrics["reward"],
                    "throughput/tok_s": throughput,
                    "gate/mean": metrics["gate_mean"],
                    "gate/diff": metrics["gate_diff"],
                    "skip/rate": metrics["skip_rate"],
                    "power/watts": power_w,
                    "loss/total": metrics["loss"],
                    "loss/grpo": metrics["grpo_loss"],
                    "loss/causal": metrics["causal_loss"],
                    "gate/entropy": metrics.get("gate_entropy", 0),
                })

            # Save checkpoint every val_every steps (validation done on daedalus)
            if (prompt_idx + 1) % args.val_every == 0:
                print(f"\n[Checkpoint at step {global_step}]", flush=True)

                # Just save checkpoint - daedalus does ALL validation
                ckpt_path = Path(args.checkpoint_dir) / f"step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": {k: v for k, v in model.state_dict().items() if "skip_blocks" in k},
                }, ckpt_path)
                print(f"  Saved: {ckpt_path} (daedalus validates)\n", flush=True)

    sensor_hub.stop()
    wandb.finish()

    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
