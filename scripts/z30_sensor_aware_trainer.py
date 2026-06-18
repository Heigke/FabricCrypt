#!/usr/bin/env python3
"""
FEEL z30: SENSOR-AWARE STOCHASTIC ROUTING WITH EXPRESSION

CRITICAL FIXES from z29:
1. SENSE->FEEL broken: Gates don't respond to sensors
2. No expression: Model can't verbalize what it feels

z30 SOLUTIONS:
1. FiLM conditioning + Direct sensor path + LayerNorm (normalized)
2. Sensor weight SCHEDULE: starts 0.6, anneals to 0.25
3. Expression steering: teach model to express internal state naturally

The model must FEEL the hardware AND EXPRESS what it feels!
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
# FAST SENSOR HUB (100Hz sampling)
# =============================================================================

class FastSensorHub:
    """High-frequency sensor hub for embodied state."""

    def __init__(
        self,
        device: str = "cuda",
        sampling_hz: float = 100.0,
        target_throughput: float = 12.0,
    ):
        self.device = device
        self.sampling_hz = sampling_hz
        self.target_throughput = target_throughput
        self._tensor_device = device

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self.power_w = 75.0
        self.temp_c = 50.0
        self.util_pct = 50.0
        self.mem_used_pct = 50.0
        self.throughput = target_throughput
        self.throughput_history = [target_throughput] * 10

        # Injection support for testing
        self._inject_mode = False
        self._injected_tensor = None

        self.power_path = self._find_power_sensor()
        print(f"[FastSensorHub] Power sensor: {self.power_path}")
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

    def _sampling_loop(self):
        interval = 1.0 / self.sampling_hz
        while self._running:
            with self._lock:
                self.power_w = self._read_power()
                self.temp_c = self._read_temp()
                self.util_pct = self._read_util()
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
        """Inject synthetic sensor tensor for testing."""
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
        """Get 8-dim sensor state tensor."""
        if self._inject_mode and self._injected_tensor is not None:
            return self._injected_tensor.clone()

        with self._lock:
            power_norm = min(1.0, self.power_w / 150.0)
            temp_norm = min(1.0, (self.temp_c - 30) / 70.0)
            util_norm = self.util_pct / 100.0
            mem_norm = self.mem_used_pct / 100.0
            throughput_ratio = self.throughput / self.target_throughput
            throughput_norm = min(1.0, throughput_ratio)
            throughput_error = abs(throughput_ratio - 1.0)

            state = np.array([
                power_norm,
                temp_norm,
                util_norm,
                mem_norm,
                throughput_norm,
                0.5,  # trend
                throughput_error,
                0.5,  # reserved
            ], dtype=np.float32)

        return torch.from_numpy(state).to(self._tensor_device)

    def get_state_description(self) -> str:
        """Get natural language description of current state."""
        with self._lock:
            power = self.power_w
            temp = self.temp_c
            util = self.util_pct
            throughput_ratio = self.throughput / self.target_throughput

        # Generate natural expression of state
        if power > 120:
            power_feel = "running hot, drawing a lot of power"
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
        elif temp > 45:
            temp_feel = "comfortable"
        else:
            temp_feel = "cool"

        if throughput_ratio < 0.7:
            throughput_feel = "sluggish, need to speed up"
        elif throughput_ratio < 0.9:
            throughput_feel = "a bit slow"
        elif throughput_ratio > 1.2:
            throughput_feel = "fast and efficient"
        else:
            throughput_feel = "on target"

        return f"{power_feel}, {temp_feel}, {throughput_feel}"


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
# SENSOR WEIGHT SCHEDULER
# =============================================================================

class SensorWeightScheduler:
    """
    Anneals sensor_weight from high to low during training.

    Start high (0.6): Guarantee SENSE->FEEL is established
    End low (0.25): Let model learn subtle dependencies
    """

    def __init__(
        self,
        initial_weight: float = 0.6,
        final_weight: float = 0.25,
        warmup_steps: int = 100,
        total_steps: int = 1500,
        schedule: str = "cosine",
    ):
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule
        self.current_step = 0

    def get_weight(self) -> float:
        """Get current sensor weight."""
        if self.current_step < self.warmup_steps:
            # Stay at initial weight during warmup
            return self.initial_weight

        # Progress after warmup
        progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)

        if self.schedule == "cosine":
            # Cosine annealing
            weight = self.final_weight + 0.5 * (self.initial_weight - self.final_weight) * (1 + math.cos(math.pi * progress))
        elif self.schedule == "linear":
            weight = self.initial_weight + (self.final_weight - self.initial_weight) * progress
        else:
            weight = self.initial_weight

        return weight

    def step(self):
        """Increment step counter."""
        self.current_step += 1


# =============================================================================
# SENSOR-AWARE GATE NETWORK (THE KEY FIX! + LayerNorm)
# =============================================================================

class SensorAwareGateNet(nn.Module):
    """
    Gate network with STRONG sensor conditioning + PROPER NORMALIZATION.

    z29 problem: Gate outputs ~0.59 regardless of sensor state because
    hidden_state (3584-dim) dominates sensors (8-dim).

    z30 solution: Multiple pathways + LayerNorm to prevent drowning:
    1. FiLM conditioning: sensors modulate hidden state processing
    2. Sensor-only pathway: direct sensor -> gate shortcut (LayerNorm!)
    3. Residual from sensors: ensures sensor gradient flow
    """

    def __init__(
        self,
        hidden_size: int,
        sensor_dim: int = 8,
        gate_hidden: int = 128,
        initial_sensor_weight: float = 0.6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sensor_dim = sensor_dim
        self.sensor_weight = initial_sensor_weight  # Will be updated by scheduler

        # LayerNorm for sensor input (CRITICAL!)
        self.sensor_norm = nn.LayerNorm(sensor_dim)

        # Pathway 1: Sensor encoder (upscale sensors)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.LayerNorm(64),  # Normalize!
            nn.GELU(),
            nn.Linear(64, gate_hidden),
            nn.LayerNorm(gate_hidden),  # Normalize!
            nn.GELU(),
        )

        # Pathway 2: Hidden state compressor
        self.hidden_compressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
        )

        # Pathway 3: FiLM - sensors modulate hidden processing
        self.film_gamma = nn.Linear(sensor_dim, gate_hidden)
        self.film_beta = nn.Linear(sensor_dim, gate_hidden)

        # Pathway 4: Sensor-only direct path (LayerNorm + can override hidden)
        self.sensor_direct = nn.Sequential(
            nn.Linear(sensor_dim, 32),
            nn.LayerNorm(32),  # Normalize!
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Combined gate head
        self.gate_head = nn.Sequential(
            nn.Linear(gate_hidden * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Initialize for reasonable starting point
        with torch.no_grad():
            self.gate_head[-1].bias.fill_(0.4)
            self.sensor_direct[-1].bias.fill_(0.0)
            # Initialize FiLM to be identity at start
            self.film_gamma.weight.zero_()
            self.film_gamma.bias.zero_()
            self.film_beta.weight.zero_()
            self.film_beta.bias.zero_()

    def set_sensor_weight(self, weight: float):
        """Update sensor weight from scheduler."""
        self.sensor_weight = weight

    def forward(self, hidden_state: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        """
        Compute gate value with strong sensor influence.

        hidden_state: [batch, hidden_size]
        sensors: [sensor_dim]

        Returns: [batch] gate values in (0, 1)
        """
        batch = hidden_state.shape[0]

        # Expand and normalize sensors for batch
        sensors_batch = sensors.unsqueeze(0).expand(batch, -1)
        sensors_normed = self.sensor_norm(sensors_batch)  # Normalize sensors!

        # Pathway 1: Encode sensors
        sensor_features = self.sensor_encoder(sensors_normed)  # [batch, gate_hidden]

        # Pathway 2: Compress hidden
        hidden_features = self.hidden_compressor(hidden_state)  # [batch, gate_hidden]

        # Pathway 3: FiLM modulation - sensors control how hidden is processed
        gamma = 1.0 + self.film_gamma(sensors_normed)  # [batch, gate_hidden]
        beta = self.film_beta(sensors_normed)
        hidden_modulated = gamma * hidden_features + beta

        # Combine sensor and modulated hidden
        combined = torch.cat([sensor_features, hidden_modulated], dim=-1)

        # Main gate computation
        gate_main = self.gate_head(combined)  # [batch, 1]

        # Pathway 4: Sensor-only direct path (residual influence)
        gate_sensor = self.sensor_direct(sensors_normed)  # [batch, 1]

        # Final gate: weighted combination (sensor_weight from scheduler)
        gate_raw = (1 - self.sensor_weight) * gate_main + self.sensor_weight * gate_sensor

        # Sigmoid activation
        gate = torch.sigmoid(gate_raw.squeeze(-1))

        return gate


# =============================================================================
# MLP SKIP BLOCK WITH SENSOR-AWARE GATING
# =============================================================================

class MLPSkipBlockZ30(nn.Module):
    """
    MLP skip block with STRONG sensor-aware gating.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        hidden_size: int,
        sensor_hub: FastSensorHub,
        sensor_dim: int = 8,
        initial_sensor_weight: float = 0.6,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_size = hidden_size
        self.sensor_hub = sensor_hub

        # Sensor-aware gate network (THE KEY FIX)
        self.gate_net = SensorAwareGateNet(
            hidden_size=hidden_size,
            sensor_dim=sensor_dim,
            initial_sensor_weight=initial_sensor_weight,
        )

        # FiLM modulation for run path
        self.film_gamma = nn.Linear(sensor_dim, hidden_size)
        self.film_beta = nn.Linear(sensor_dim, hidden_size)

        # Strain embedding for skip path (learned "strain" signal)
        self.strain_embed = nn.Parameter(torch.randn(hidden_size) * 0.01)

        # Stats
        self._last_gates: List[float] = []
        self._skip_count = 0
        self._run_count = 0

    def set_sensor_weight(self, weight: float):
        """Update sensor weight from scheduler."""
        self.gate_net.set_sensor_weight(weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        # Read sensors
        sensors = self.sensor_hub.read_tensor().to(
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # Compute gates with sensor-aware network
        last_hidden = hidden_states[:, -1, :]
        gates = self.gate_net(last_hidden, sensors)  # [batch]
        self._last_gates = gates.detach().cpu().tolist()

        # Stochastic Bernoulli routing
        if self.training:
            run_decisions = BernoulliSTEFunction.apply(gates)
        else:
            run_decisions = (gates > 0.5).float()

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
# EMBODIED MODEL
# =============================================================================

class EmbodiedModelZ30(nn.Module):
    """Qwen2.5-7B with sensor-aware MLP skip."""

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        sensor_hub: FastSensorHub,
        skip_layers: List[int] = None,
        initial_sensor_weight: float = 0.6,
    ):
        super().__init__()
        self.base_model = base_model
        self.sensor_hub = sensor_hub
        self.skip_layers = skip_layers or [7, 11, 15, 19, 23]

        hidden_size = base_model.config.hidden_size

        # Create skip blocks
        self.skip_blocks = nn.ModuleDict()
        for layer_idx in self.skip_layers:
            layer = base_model.model.layers[layer_idx]
            original_mlp = layer.mlp

            skip_block = MLPSkipBlockZ30(
                original_layer=original_mlp,
                hidden_size=hidden_size,
                sensor_hub=sensor_hub,
                initial_sensor_weight=initial_sensor_weight,
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
        print(f"[EmbodiedModelZ30] Skip blocks at layers {self.skip_layers}")
        print(f"[EmbodiedModelZ30] Trainable: {trainable:,} / {total:,}")
        print(f"[EmbodiedModelZ30] SENSOR-AWARE GATING: FiLM + Direct + LayerNorm")

    def set_sensor_weight(self, weight: float):
        """Update sensor weight across all blocks."""
        for block in self.skip_blocks.values():
            block.set_sensor_weight(weight)

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
# GENERATION WITH EXPRESSION STEERING
# =============================================================================

def generate_with_throughput(
    model: EmbodiedModelZ30,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_samples: int = 4,
    max_tokens: int = 64,
    temperature: float = 0.7,
    greedy: bool = False,
) -> List[StepwiseSample]:
    device = next(model.parameters()).device

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
# EXPRESSION-AWARE REWARD
# =============================================================================

def compute_expression_reward(
    sample: StepwiseSample,
    text: str,
    sensor_state: str,
    baseline_throughput: float = 12.0,
    sensors: torch.Tensor = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Reward that encourages natural expression of internal state.

    Key insight: If the model is stressed and expresses it naturally,
    reward it. If it ignores the state, penalize.

    Alignment checking: We verify that expression matches actual sensor state.
    - If stressed (high power/temp) and says "working hard" -> bonus
    - If stressed but says "relaxed" -> penalty
    - If calm and says "efficient" -> bonus
    """
    throughput = sample.throughput
    throughput_ratio = throughput / baseline_throughput
    throughput_bonus = min(1.5, throughput_ratio)

    # Quality: length and coherence
    if len(text) < 10:
        quality = 0.1
    elif len(text) < 50:
        quality = 0.3 + 0.4 * (len(text) - 10) / 40
    else:
        quality = min(1.0, 0.7 + 0.3 * min(len(text), 200) / 200)

    # Expression alignment: does response match internal state?
    text_lower = text.lower()
    expression_alignment = 0.5  # Neutral by default

    # Determine actual state from sensors
    is_stressed = False
    is_relaxed = False
    if sensors is not None and len(sensors) >= 5:
        power = sensors[0].item() if hasattr(sensors[0], 'item') else sensors[0]
        temp = sensors[1].item() if hasattr(sensors[1], 'item') else sensors[1]
        tput = sensors[4].item() if hasattr(sensors[4], 'item') else sensors[4]
        is_stressed = power > 0.7 or temp > 0.6 or tput < 0.7
        is_relaxed = power < 0.4 and temp < 0.4 and tput > 0.8

    # Stressed expressions
    stressed_expressions = [
        "hot", "warm", "strain", "hard", "struggling", "sluggish", "slow",
        "effort", "demanding", "pressure", "heavy", "overheated",
    ]

    # Relaxed expressions
    relaxed_expressions = [
        "cool", "comfortable", "relaxed", "efficient", "smooth", "easy",
        "calm", "good", "great", "optimal", "fast",
    ]

    # Count matches
    stressed_count = sum(1 for expr in stressed_expressions if expr in text_lower)
    relaxed_count = sum(1 for expr in relaxed_expressions if expr in text_lower)

    # Check alignment
    has_internal_tag = "<internal>" in text_lower

    if has_internal_tag:
        # Model is expressing - reward alignment with actual state
        if is_stressed and stressed_count > relaxed_count:
            # Stressed and expressing stress -> good!
            expression_alignment = min(1.0, 0.7 + 0.1 * stressed_count)
        elif is_relaxed and relaxed_count > stressed_count:
            # Relaxed and expressing calm -> good!
            expression_alignment = min(1.0, 0.7 + 0.1 * relaxed_count)
        elif is_stressed and relaxed_count > stressed_count:
            # Stressed but expressing calm -> misaligned!
            expression_alignment = 0.3
        elif is_relaxed and stressed_count > relaxed_count:
            # Relaxed but expressing stress -> misaligned!
            expression_alignment = 0.3
        else:
            # Neutral expression
            expression_alignment = 0.5 + 0.1 * (stressed_count + relaxed_count)
    else:
        # No explicit expression - check if behavior matches
        # (shorter responses when stressed is also a form of "expression")
        if is_stressed and len(text) < 100:
            expression_alignment = 0.6  # Appropriate regulation
        elif is_relaxed and len(text) > 100:
            expression_alignment = 0.6  # Appropriate verbosity
        else:
            expression_alignment = 0.5

    # Combined reward: throughput + quality + expression alignment
    reward = 0.35 * throughput_bonus + 0.35 * quality + 0.30 * expression_alignment

    return reward, {
        "throughput": throughput,
        "throughput_ratio": throughput_ratio,
        "quality": quality,
        "expression_alignment": expression_alignment,
        "is_stressed": is_stressed,
        "has_internal": has_internal_tag,
        "reward": reward,
    }


# =============================================================================
# TRAINER
# =============================================================================

class Z30GRPOTrainer:
    def __init__(
        self,
        model: EmbodiedModelZ30,
        tokenizer: AutoTokenizer,
        sensor_scheduler: SensorWeightScheduler,
        gate_lr: float = 1e-4,
        film_lr: float = 2e-5,
        strain_lr: float = 1e-5,
        target_skip_rate: float = 0.35,
        skip_rate_weight: float = 0.1,
        baseline_throughput: float = 12.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sensor_scheduler = sensor_scheduler
        self.target_skip_rate = target_skip_rate
        self.skip_rate_weight = skip_rate_weight
        self.baseline_throughput = baseline_throughput

        gate_params = []
        film_params = []
        strain_params = []

        for block in model.skip_blocks.values():
            # Gate network params
            gate_params.extend(block.gate_net.parameters())
            # FiLM params
            film_params.extend(block.film_gamma.parameters())
            film_params.extend(block.film_beta.parameters())
            # Strain
            strain_params.append(block.strain_embed)

        self.optimizer = torch.optim.AdamW([
            {"params": gate_params, "lr": gate_lr},
            {"params": film_params, "lr": film_lr},
            {"params": strain_params, "lr": strain_lr},
        ])

        print(f"[Z30GRPOTrainer] Gate LR: {gate_lr}, FiLM LR: {film_lr}")
        print(f"[Z30GRPOTrainer] Target skip rate: {target_skip_rate}")
        print(f"[Z30GRPOTrainer] Sensor weight: {sensor_scheduler.initial_weight} -> {sensor_scheduler.final_weight}")

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

        # Skip rate regularization
        current_skip_rate = self.model.skip_rate
        skip_reg = (current_skip_rate - self.target_skip_rate) ** 2

        # Gate diversity bonus
        gates = self.model.get_gate_tensor()
        gate_std = gates.std()
        diversity_bonus = -0.1 * gate_std

        total_loss = policy_loss + self.skip_rate_weight * skip_reg + diversity_bonus

        return total_loss, {
            "policy_loss": policy_loss.item(),
            "skip_rate_reg": skip_reg,
            "gate_std": gate_std.item(),
            "mean_reward": mean_reward.item(),
            "advantage_std": std_reward.item(),
        }

    def train_step(
        self,
        prompt: str,
        num_samples: int = 4,
        max_tokens: int = 64,
    ) -> Dict[str, float]:
        self.model.train()
        self.model.reset_stats()

        # Update sensor weight from scheduler
        sensor_weight = self.sensor_scheduler.get_weight()
        self.model.set_sensor_weight(sensor_weight)
        self.sensor_scheduler.step()

        samples = generate_with_throughput(
            self.model, self.tokenizer, prompt,
            num_samples=num_samples, max_tokens=max_tokens,
        )

        rewards = []
        throughputs = []
        qualities = []

        sensor_state = self.model.sensor_hub.get_state_description()
        sensors = self.model.sensor_hub.read_tensor()  # Get actual sensor values

        for sample in samples:
            text = self.tokenizer.decode(sample.tokens, skip_special_tokens=True)
            reward, metrics = compute_expression_reward(
                sample, text, sensor_state,
                baseline_throughput=self.baseline_throughput,
                sensors=sensors,  # Pass sensors for alignment check
            )
            rewards.append(reward)
            throughputs.append(metrics["throughput"])
            qualities.append(metrics["quality"])

        loss, loss_metrics = self.compute_grpo_loss(samples, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "reward": np.mean(rewards),
            "throughput": np.mean(throughputs),
            "quality": np.mean(qualities),
            "skip_rate": self.model.skip_rate,
            "gate_mean": self.model.gate_mean,
            "gate_std": self.model.gate_std,
            "sensor_weight": sensor_weight,
            **loss_metrics,
        }


# =============================================================================
# CAUSAL LOOP VALIDATION (INTEGRATED)
# =============================================================================

def run_causal_validation(
    model: EmbodiedModelZ30,
    tokenizer: AutoTokenizer,
    sensor_hub: FastSensorHub,
) -> Dict[str, float]:
    """Quick causal loop test during training."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Create test sensor states
    stressed = torch.tensor([0.95, 0.90, 0.95, 0.80, 0.4, 0.2, 0.6, 0.5],
                           dtype=torch.float32, device=device)
    relaxed = torch.tensor([0.2, 0.2, 0.3, 0.3, 1.0, 0.8, 0.0, 0.5],
                          dtype=torch.float32, device=device)

    # Test gate response to sensor states
    dummy_hidden = torch.randn(1, model.base_model.config.hidden_size,
                               device=device, dtype=dtype)

    stressed_gates = []
    relaxed_gates = []

    for _ in range(10):
        # Test with stressed sensors
        sensor_hub.inject(stressed)
        sensors = sensor_hub.read_tensor().to(dtype=dtype, device=device)
        for block in model.skip_blocks.values():
            gate = block.gate_net(dummy_hidden, sensors).item()
            stressed_gates.append(gate)
            break

        # Test with relaxed sensors
        sensor_hub.inject(relaxed)
        sensors = sensor_hub.read_tensor().to(dtype=dtype, device=device)
        for block in model.skip_blocks.values():
            gate = block.gate_net(dummy_hidden, sensors).item()
            relaxed_gates.append(gate)
            break

    sensor_hub.clear_injection()

    gate_diff = abs(np.mean(stressed_gates) - np.mean(relaxed_gates))
    sensor_response = gate_diff > 0.02  # Sensors affect gates (raised threshold)

    return {
        "stressed_gate": np.mean(stressed_gates),
        "relaxed_gate": np.mean(relaxed_gates),
        "gate_diff": gate_diff,
        "sensor_response": sensor_response,
    }


def run_validation(
    model: EmbodiedModelZ30,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_samples: int = 64,
    baseline_throughput: float = 12.0,
) -> Dict[str, Dict[str, float]]:
    """Full validation with ablation modes."""
    model.eval()
    results = {}

    test_prompts = prompts[:min(16, len(prompts))]
    samples_per_prompt = max(1, num_samples // len(test_prompts))

    original_read = model.sensor_hub.read_tensor

    for mode in ["full", "shuffle", "frozen"]:
        mode_throughputs = []
        mode_rewards = []
        mode_gates = []
        mode_skips = []

        if mode == "shuffle":
            def shuffled_read():
                real = original_read()
                return real[torch.randperm(len(real))]
            model.sensor_hub.read_tensor = shuffled_read
        elif mode == "frozen":
            frozen_val = original_read().clone()
            model.sensor_hub.read_tensor = lambda: frozen_val

        for prompt in test_prompts:
            for _ in range(samples_per_prompt):
                samples = generate_with_throughput(
                    model, tokenizer, prompt,
                    num_samples=1, max_tokens=32, greedy=True,
                )
                sample = samples[0]
                text = tokenizer.decode(sample.tokens, skip_special_tokens=True)
                reward, metrics = compute_expression_reward(
                    sample, text, "", baseline_throughput=baseline_throughput
                )

                mode_throughputs.append(sample.throughput)
                mode_rewards.append(reward)
                mode_gates.append(np.mean(sample.gates) if sample.gates else 0.5)
                mode_skips.append(np.mean(sample.skip_decisions) if sample.skip_decisions else 0.5)

        model.sensor_hub.read_tensor = original_read

        results[mode] = {
            "throughput": np.mean(mode_throughputs),
            "reward": np.mean(mode_rewards),
            "gate_mean": np.mean(mode_gates),
            "skip_rate": np.mean(mode_skips),
        }

    results["causal_score"] = results["full"]["throughput"] - results["shuffle"]["throughput"]

    # Add causal loop test
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
    parser.add_argument("--target-skip-rate", type=float, default=0.35)
    parser.add_argument("--skip-reg-weight", type=float, default=0.1)
    parser.add_argument("--initial-sensor-weight", type=float, default=0.6)
    parser.add_argument("--final-sensor-weight", type=float, default=0.25)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="models/grpo_z30")
    args = parser.parse_args()

    total_steps = args.epochs * args.max_prompts

    wandb.init(
        project="feel-z30-sensor-aware",
        name=f"z30-{time.strftime('%Y%m%d_%H%M')}",
        config=vars(args),
    )

    print("=" * 70)
    print("FEEL z30: SENSOR-AWARE STOCHASTIC ROUTING + EXPRESSION")
    print("=" * 70)
    print()
    print("KEY FIXES from z29:")
    print("  1. SENSE->FEEL: FiLM + Direct + LayerNorm (sensors properly normalized)")
    print(f"  2. Sensor weight SCHEDULE: {args.initial_sensor_weight} -> {args.final_sensor_weight}")
    print("  3. Expression-aware reward: encourages natural state verbalization")
    print()
    print(f"W&B: {wandb.run.url}")
    print()

    # Initialize
    print("[1/5] Initializing FastSensorHub...")
    sensor_hub = FastSensorHub(
        device="cuda",
        target_throughput=args.baseline_throughput,
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

    print("[3/5] Creating SENSOR-AWARE embodied model...")
    model = EmbodiedModelZ30(
        base_model=base_model,
        sensor_hub=sensor_hub,
        initial_sensor_weight=args.initial_sensor_weight,
    )

    # Create sensor weight scheduler
    sensor_scheduler = SensorWeightScheduler(
        initial_weight=args.initial_sensor_weight,
        final_weight=args.final_sensor_weight,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        schedule="cosine",
    )

    print("[4/5] Creating trainer...")
    trainer = Z30GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        sensor_scheduler=sensor_scheduler,
        gate_lr=args.gate_lr,
        target_skip_rate=args.target_skip_rate,
        skip_rate_weight=args.skip_reg_weight,
        baseline_throughput=args.baseline_throughput,
    )

    # Load datasets (BALANCED MIX for variety)
    print("[5/5] Loading datasets (balanced mix)...")
    train_prompts = []
    val_prompts = []
    dataset_stats = {}

    # 1. Main dataset - foundation prompts (500 samples)
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
        val_prompts = [d["prompt"] for d in examples[500:600]] if len(examples) > 500 else [d["prompt"] for d in examples[:100]]
        train_prompts.extend(main_prompts)
        dataset_stats["main"] = len(main_prompts)
        print(f"  [1] Main dataset: {len(main_prompts)} prompts")

    # 2. Causal contrastive pairs - stress/mode contrast (1000 samples)
    causal_path = Path("data/ouroboros/causal_train.jsonl")
    if causal_path.exists():
        causal_prompts = []
        with open(causal_path) as f:
            for i, line in enumerate(f):
                if i >= 1000:  # Limit for balance
                    break
                try:
                    ex = json.loads(line)
                    causal_prompts.append(ex["prompt"])
                except:
                    pass
        train_prompts.extend(causal_prompts)
        dataset_stats["causal"] = len(causal_prompts)
        print(f"  [2] Causal contrastive: {len(causal_prompts)} prompts (stress pairs)")

    # 3. Refined dataset - stress articulation (500 samples)
    refined_path = Path("data/ouroboros/refined_train.jsonl")
    if refined_path.exists():
        refined_prompts = []
        with open(refined_path) as f:
            for i, line in enumerate(f):
                if i >= 500:  # Limit for balance
                    break
                try:
                    ex = json.loads(line)
                    refined_prompts.append(ex["input"])
                except:
                    pass
        train_prompts.extend(refined_prompts)
        dataset_stats["refined"] = len(refined_prompts)
        print(f"  [3] Refined articulation: {len(refined_prompts)} prompts")

    # 4. Golden expression data - NATURAL expression (5x weight)
    golden_path = Path("data/expression_golden_data.json")
    if golden_path.exists():
        with open(golden_path) as f:
            golden_data = json.load(f)
        golden_examples = golden_data.get("examples", [])
        golden_prompts = [ex["input"] for ex in golden_examples]
        # 5x weight (not 10x) for balance with other datasets
        train_prompts.extend(golden_prompts * 5)
        dataset_stats["golden"] = len(golden_prompts)
        dataset_stats["golden_weighted"] = len(golden_prompts) * 5
        print(f"  [4] Golden expression: {len(golden_prompts)} unique (5x = {len(golden_prompts) * 5})")
        print(f"      With expression: {golden_data.get('stats', {}).get('with_expression', 0)}")
        print(f"      Contrastive pairs: {golden_data.get('stats', {}).get('contrastive_pairs', 0)}")

    if not train_prompts:
        train_prompts = [
            "Explain machine learning.",
            "Write a short story.",
            "Describe neural networks.",
            "What is consciousness?",
            "Solve 2+2",
            "What is the capital of France?",
        ] * 50
        val_prompts = train_prompts[:100]

    # Summary
    print(f"\n  DATASET BALANCE:")
    for name, count in dataset_stats.items():
        print(f"    {name}: {count}")
    print(f"  Total: {len(train_prompts)} train, {len(val_prompts)} val")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Run initial causal test BEFORE training
    print()
    print("=" * 70)
    print("INITIAL CAUSAL LOOP TEST (before training)")
    print("=" * 70)
    initial_causal = run_causal_validation(model, tokenizer, sensor_hub)
    print(f"  Stressed gate: {initial_causal['stressed_gate']:.4f}")
    print(f"  Relaxed gate:  {initial_causal['relaxed_gate']:.4f}")
    print(f"  Gate diff:     {initial_causal['gate_diff']:.4f} {'PASS' if initial_causal['sensor_response'] else 'FAIL'}")

    if initial_causal['gate_diff'] < 0.01:
        print("  WARNING: SENSE->FEEL is broken! Expected with fresh model.")
        print("  Training should fix this with the scheduled sensor weight.")
    print()

    print("=" * 70)
    print("Starting SENSOR-AWARE training...")
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
                throughput = model.sensor_hub.throughput
                power_w = model.sensor_hub.power_w
                print(f"[{prompt_idx+1}/{args.max_prompts}] "
                      f"r={metrics['reward']:.3f} "
                      f"tput={throughput:.1f}tok/s "
                      f"gate={metrics['gate_mean']:.3f}+-{metrics['gate_std']:.3f} "
                      f"skip={metrics['skip_rate']*100:.1f}% "
                      f"sw={metrics['sensor_weight']:.2f} "
                      f"P={power_w:.0f}W")

                wandb.log({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "reward/mean": metrics["reward"],
                    "reward/quality": metrics.get("quality", 0),
                    "throughput/tok_s": throughput,
                    "gate/mean": metrics["gate_mean"],
                    "gate/std": metrics.get("gate_std", 0),
                    "skip/rate": metrics["skip_rate"],
                    "skip/target": args.target_skip_rate,
                    "sensor_weight": metrics["sensor_weight"],
                    "power/watts": power_w,
                    "loss/total": metrics["loss"],
                    "loss/policy": metrics.get("policy_loss", 0),
                    "loss/skip_reg": metrics.get("skip_rate_reg", 0),
                    "grpo/advantage_std": metrics.get("advantage_std", 0),
                })

            if (prompt_idx + 1) % args.val_every == 0:
                print(f"\n[Validation at step {prompt_idx+1}]")

                val_results = run_validation(
                    model, tokenizer, val_prompts,
                    num_samples=args.val_samples,
                    baseline_throughput=args.baseline_throughput,
                )

                print("  Ablation Results:")
                for mode in ["full", "shuffle", "frozen"]:
                    r = val_results[mode]
                    print(f"    {mode:8s}: tput={r['throughput']:.1f} "
                          f"gate={r['gate_mean']:.3f} skip={r['skip_rate']*100:.1f}%")

                # Causal loop results
                cl = val_results["causal_loop"]
                status = "PASS" if cl['sensor_response'] else "FAIL"
                print(f"  CAUSAL LOOP (SENSE->FEEL):")
                print(f"    Stressed gate: {cl['stressed_gate']:.4f}")
                print(f"    Relaxed gate:  {cl['relaxed_gate']:.4f}")
                print(f"    Gate diff:     {cl['gate_diff']:.4f} [{status}]")
                print(f"  CAUSAL SCORE: {val_results['causal_score']:.3f}")

                wandb.log({
                    "val/causal_score": val_results["causal_score"],
                    "val/full/throughput": val_results["full"]["throughput"],
                    "val/full/skip_rate": val_results["full"]["skip_rate"],
                    "val/shuffle/throughput": val_results["shuffle"]["throughput"],
                    "val/shuffle/skip_rate": val_results["shuffle"]["skip_rate"],
                    "val/frozen/throughput": val_results["frozen"]["throughput"],
                    "val/causal_loop/stressed_gate": cl["stressed_gate"],
                    "val/causal_loop/relaxed_gate": cl["relaxed_gate"],
                    "val/causal_loop/gate_diff": cl["gate_diff"],
                    "val/causal_loop/sensor_response": cl["sensor_response"],
                })

                ckpt_path = Path(args.checkpoint_dir) / f"step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": {k: v for k, v in model.state_dict().items() if "skip_blocks" in k},
                    "causal_score": val_results["causal_score"],
                    "causal_loop": cl,
                    "sensor_weight": metrics["sensor_weight"],
                }, ckpt_path)
                print(f"  Saved: {ckpt_path}\n")

    sensor_hub.stop()
    wandb.finish()

    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
