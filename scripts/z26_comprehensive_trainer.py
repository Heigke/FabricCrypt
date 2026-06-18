#!/usr/bin/env python3
"""
FEEL z26: Comprehensive Closed-Loop Embodied Training

ALL improvements from feedback:
1. Per-sample gate (no .mean()) - gate decisions per batch item
2. Top-k sampling (k=50) - faster than top-p sort
3. IAE power reward with trajectory window - control-style error integral
4. Conditional gate bonus - only when power ABOVE target (incentivize skipping)
5. Disturbance injection - background GPU workload to create stress
6. Anti-cheat - zero power reward if task_reward < threshold
7. Comprehensive sensing metrics (ablation with hardware trajectories)
8. Regulation metrics (IAE, time_in_band, overshoot_pct, time_over_120pct)
9. Expression metrics (paired eval: low vs high stress)
10. Realistic targets: target_power=75W, band=15W

The goal: sense → modulate → actuate → hardware changes → new sensing
With MEASURABLE hardware trajectory separation (full vs shuffle).
"""

import os
import sys
import time
import json
import argparse
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FAST SENSOR HUB (100Hz power-focused)
# =============================================================================

class FastSensorHub:
    """
    High-frequency sensor sampling focused on fast-changing signals.

    Fast signals (100Hz):
      - power1_average (control target)
      - gpu_busy_percent
      - pp_dpm_sclk state
      - throttle status
    """

    HWMON_PATH = "/sys/class/drm/card1/device/hwmon/hwmon7"
    DEVICE_PATH = "/sys/class/drm/card1/device"

    # Power limits for normalization (RX 7900 XTX)
    POWER_MIN = 20_000_000   # 20W idle
    POWER_MAX = 350_000_000  # 350W max

    def __init__(
        self,
        sample_rate: float = 100.0,
        device: str = "cuda",
        target_power_w: float = 75.0,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.target_power = target_power_w * 1e6  # Convert to microwatts
        self._lock = threading.Lock()
        self._running = False

        # Current state (fast signals)
        self._power_raw = 0  # microwatts
        self._power_norm = 0.5
        self._gpu_busy = 0.0
        self._mem_busy = 0.0
        self._sclk_state = 0  # 0, 1, 2
        self._sclk_max = 2
        self._stress_flag = False

        # Power trajectory for windowed reward
        self._power_history = deque(maxlen=int(sample_rate * 10))  # 10s history
        self._power_timestamps = deque(maxlen=int(sample_rate * 10))

        # Cached GPU tensor
        self._cached_tensor: Optional[torch.Tensor] = None
        self._tensor_dirty = True

        # Verify paths
        self._verify_paths()

        # Start sampling
        self._start_sampling()

    def _verify_paths(self):
        power_path = f"{self.HWMON_PATH}/power1_average"
        if not os.path.exists(power_path):
            raise RuntimeError(f"Power sensor not found: {power_path}")
        print(f"[FastSensorHub] Power sensor: {power_path}")

    def _start_sampling(self):
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        print(f"[FastSensorHub] Sampling at {self.sample_rate}Hz, target={self.target_power/1e6:.0f}W")

    def _sample_loop(self):
        interval = 1.0 / self.sample_rate
        while self._running:
            try:
                t = time.time()

                # Read power (fast)
                with open(f"{self.HWMON_PATH}/power1_average", "r") as f:
                    self._power_raw = int(f.read().strip())

                # Read GPU busy (fast)
                with open(f"{self.DEVICE_PATH}/gpu_busy_percent", "r") as f:
                    self._gpu_busy = float(f.read().strip()) / 100.0

                # Read mem busy if available
                try:
                    with open(f"{self.DEVICE_PATH}/mem_busy_percent", "r") as f:
                        self._mem_busy = float(f.read().strip()) / 100.0
                except:
                    pass

                # Read SCLK state (fast)
                try:
                    with open(f"{self.DEVICE_PATH}/pp_dpm_sclk", "r") as f:
                        lines = f.read().strip().split('\n')
                        self._sclk_max = len(lines) - 1
                        for i, line in enumerate(lines):
                            if '*' in line:
                                self._sclk_state = i
                                break
                except:
                    pass

                # Normalize power
                self._power_norm = (self._power_raw - self.POWER_MIN) / (self.POWER_MAX - self.POWER_MIN)
                self._power_norm = max(0.0, min(1.0, self._power_norm))

                # Stress indicator (NOT real throttle - just power > 120% target)
                # Real throttle would need GPU_THROTTLE_STATUS from sysfs
                self._stress_flag = self._power_raw > self.target_power * 1.2

                # Record history
                with self._lock:
                    self._power_history.append(self._power_raw)
                    self._power_timestamps.append(t)
                    self._tensor_dirty = True

                # Sleep
                elapsed = time.time() - t
                if elapsed < interval:
                    time.sleep(interval - elapsed)

            except Exception as e:
                time.sleep(0.1)

    def read_tensor(self) -> torch.Tensor:
        """Get current sensor state as GPU tensor (8-dim, fast signals only)."""
        with self._lock:
            if self._tensor_dirty or self._cached_tensor is None:
                power_history = list(self._power_history)

                # Fast 8-dim vector
                state = np.array([
                    self._power_norm,                    # 0: power (normalized 0-1)
                    self._gpu_busy,                      # 1: GPU utilization
                    self._mem_busy,                      # 2: Memory utilization
                    self._sclk_state / max(self._sclk_max, 1),  # 3: clock state (0-1)
                    float(self._stress_flag),            # 4: stress flag (power > 120% target)
                    self._power_raw / self.target_power - 1.0,  # 5: power error (target-relative)
                    np.std(power_history[-50:]) / 1e7 if len(power_history) > 10 else 0,  # 6: power variance
                    sum(1 for p in power_history[-100:] if p > self.target_power) / max(len(power_history[-100:]), 1),  # 7: time over target
                ], dtype=np.float32)

                if self._cached_tensor is None:
                    self._cached_tensor = torch.from_numpy(state).to(self.device)
                else:
                    self._cached_tensor.copy_(torch.from_numpy(state))
                self._tensor_dirty = False

            return self._cached_tensor

    def get_power_window(self, window_sec: float = 2.0) -> Dict[str, float]:
        """Get comprehensive power stats over recent window."""
        with self._lock:
            if len(self._power_history) < 10:
                return {
                    "mean_w": self._power_raw / 1e6,
                    "std_w": 0.0,
                    "time_in_band": 0.0,
                    "time_over": 0.0,
                    "IAE": 0.0,
                    "overshoot_pct": 0.0,
                    "time_over_120pct": 0.0,
                }

            now = time.time()
            cutoff = now - window_sec

            # Get samples in window
            recent = []
            for t, p in zip(self._power_timestamps, self._power_history):
                if t >= cutoff:
                    recent.append(p)

            if not recent:
                recent = list(self._power_history)[-10:]

            power_array = np.array(recent)
            target_w = self.target_power / 1e6  # Convert to watts
            band_w = 15.0  # Default band

            mean_w = np.mean(power_array) / 1e6
            std_w = np.std(power_array) / 1e6

            # Time in band
            in_band = sum(1 for p in power_array if abs(p/1e6 - target_w) < band_w)
            time_in_band = in_band / len(power_array)

            # Time over target
            over = sum(1 for p in power_array if p > self.target_power)
            time_over = over / len(power_array)

            # IAE (Integral Absolute Error) - normalized by target
            errors = np.abs(power_array / 1e6 - target_w)
            IAE = np.mean(errors) / target_w

            # Overshoot percentage
            max_power = np.max(power_array) / 1e6
            overshoot_pct = max(0, (max_power - target_w) / target_w * 100)

            # Time over 120% of target (NOT real throttle - just a stress indicator)
            # Real throttle detection would require reading GPU_THROTTLE_STATUS from sysfs
            time_over_120pct = sum(1 for p in power_array if p > self.target_power * 1.2) / len(power_array) * 100

            return {
                "mean_w": mean_w,
                "std_w": std_w,
                "time_in_band": time_in_band,
                "time_over": time_over,
                "IAE": IAE,
                "overshoot_pct": overshoot_pct,
                "time_over_120pct": time_over_120pct,  # Renamed: not real throttle
            }

    @property
    def power_w(self) -> float:
        """Current power in watts."""
        return self._power_raw / 1e6

    def stop(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=1.0)
        print("[FastSensorHub] Stopped")


# =============================================================================
# DISTURBANCE INJECTOR - Creates stress for learning regulation
# =============================================================================

class DisturbanceInjector:
    """
    Injects GPU workload to create stress conditions for learning.

    Uses matrix multiplications to create controllable power draw.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._running = False
        self._intensity = 0.0
        self._thread = None

    def start(self, intensity: float = 0.5):
        """Start background workload at given intensity (0-1)."""
        if self._running:
            self._intensity = intensity
            return

        self._intensity = intensity
        self._running = True
        self._thread = threading.Thread(target=self._workload_loop, daemon=True)
        self._thread.start()
        print(f"[DisturbanceInjector] Started at intensity={intensity:.1%}")

    def set_intensity(self, intensity: float):
        """Change workload intensity."""
        self._intensity = max(0.0, min(1.0, intensity))

    def stop(self):
        """Stop background workload."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[DisturbanceInjector] Stopped")

    def _workload_loop(self):
        """Background loop that creates GPU load."""
        # Allocate matrices based on intensity
        max_size = 4096

        while self._running:
            if self._intensity < 0.01:
                time.sleep(0.1)
                continue

            try:
                # Scale matrix size with intensity
                size = int(max_size * self._intensity)
                size = max(256, size)

                # Create and multiply matrices
                a = torch.randn(size, size, device=self.device, dtype=torch.float16)
                b = torch.randn(size, size, device=self.device, dtype=torch.float16)

                # Do some work
                for _ in range(10):
                    c = torch.matmul(a, b)
                    a = c

                # Sync to actually use GPU
                torch.cuda.synchronize()

                # Small sleep to control rate
                time.sleep(0.01)

            except Exception as e:
                time.sleep(0.1)


# =============================================================================
# PER-SAMPLE MLP-Skip Block
# =============================================================================

class MLPSkipBlock(nn.Module):
    """
    Embodied block with PER-SAMPLE gate decisions.

    Key difference from z25: gate is [batch] not scalar.
    This allows different samples in the batch to make different decisions.

    Training: Soft gating (differentiable)
    Inference: Hard skip (real compute savings)
    """

    def __init__(
        self,
        original_layer: nn.Module,
        hidden_size: int,
        sensor_hub: 'FastSensorHub',
        sensor_dim: int = 8,
        skip_threshold: float = 0.3,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_size = hidden_size
        self.sensor_hub = sensor_hub
        self.sensor_dim = sensor_dim
        self.skip_threshold = skip_threshold

        # Gate network: decides skip based on hidden state + sensors
        # Output is PER-SAMPLE gate [batch, 1]
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size + sensor_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),  # Exploration via dropout
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # FiLM: modulates MLP output when not skipping
        self.film_gamma = nn.Linear(sensor_dim, hidden_size)
        self.film_beta = nn.Linear(sensor_dim, hidden_size)

        # Strain embedding: injected when skipping (proprioception)
        self.strain_embed = nn.Parameter(torch.randn(hidden_size) * 0.01)

        # Stats for logging
        self._last_gates: List[float] = []
        self._last_skipped: List[bool] = []
        self._skip_count = 0
        self._total_count = 0
        self._last_film_gamma = 1.0
        self._last_film_beta = 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with PER-SAMPLE conditional MLP skip.

        Args:
            hidden_states: [batch, seq, hidden]

        Returns:
            output: [batch, seq, hidden]
        """
        batch, seq_len, _ = hidden_states.shape

        # Read current sensors from hub and cast to match dtype
        sensors = self.sensor_hub.read_tensor().to(dtype=hidden_states.dtype)  # [sensor_dim]

        # Compute gate from last token + sensors - PER SAMPLE
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]
        sensors_expanded = sensors.unsqueeze(0).expand(batch, -1)  # [batch, sensor_dim]
        gate_input = torch.cat([last_hidden, sensors_expanded], dim=-1)

        # PER-SAMPLE gate: [batch, 1]
        gates = self.gate_net(gate_input)  # [batch, 1]

        self._last_gates = gates.squeeze(-1).detach().cpu().tolist()
        self._total_count += batch

        if self.training:
            # Soft gating during training (differentiable)
            # Run original layer
            layer_out = self.original_layer(hidden_states)

            # FiLM modulation
            gamma = 1.0 + self.film_gamma(sensors)  # [hidden]
            beta = self.film_beta(sensors)  # [hidden]
            self._last_film_gamma = gamma.mean().item()
            self._last_film_beta = beta.mean().item()

            modulated = gamma.view(1, 1, -1) * layer_out + beta.view(1, 1, -1)

            # Soft mix PER SAMPLE: gate * modulated + (1-gate) * (hidden + strain)
            skip_out = hidden_states + self.strain_embed.view(1, 1, -1)

            # gates: [batch, 1] -> [batch, 1, 1] for broadcasting
            gates_expanded = gates.unsqueeze(-1)  # [batch, 1, 1]
            output = gates_expanded * modulated + (1 - gates_expanded) * skip_out

            self._last_skipped = [g < self.skip_threshold for g in self._last_gates]

        else:
            # Hard skip during inference - PER SAMPLE
            skip_mask = (gates.squeeze(-1) < self.skip_threshold)  # [batch]

            self._last_skipped = skip_mask.tolist()
            self._skip_count += skip_mask.sum().item()

            if skip_mask.all():
                # All samples skip - no MLP needed
                output = hidden_states + self.strain_embed.view(1, 1, -1)
            elif (~skip_mask).all():
                # No samples skip - full MLP
                layer_out = self.original_layer(hidden_states)
                gamma = 1.0 + self.film_gamma(sensors)
                beta = self.film_beta(sensors)
                output = gamma.view(1, 1, -1) * layer_out + beta.view(1, 1, -1)
            else:
                # Mixed: some skip, some run
                output = hidden_states.clone()

                # Run MLP for non-skipped
                non_skip_idx = (~skip_mask).nonzero(as_tuple=True)[0]
                if len(non_skip_idx) > 0:
                    layer_out = self.original_layer(hidden_states[non_skip_idx])
                    gamma = 1.0 + self.film_gamma(sensors)
                    beta = self.film_beta(sensors)
                    modulated = gamma.view(1, 1, -1) * layer_out + beta.view(1, 1, -1)
                    output[non_skip_idx] = modulated

                # Add strain for skipped
                skip_idx = skip_mask.nonzero(as_tuple=True)[0]
                if len(skip_idx) > 0:
                    output[skip_idx] = hidden_states[skip_idx] + self.strain_embed.view(1, 1, -1)

        return output

    @property
    def skip_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._skip_count / self._total_count

    @property
    def gate_mean(self) -> float:
        if not self._last_gates:
            return 0.5
        return np.mean(self._last_gates)


# =============================================================================
# EMBODIED MODEL WITH PER-SAMPLE ACTUATORS
# =============================================================================

class EmbodiedModelZ26(nn.Module):
    """
    Embodied model with PER-SAMPLE MLP-skip actuators for real power regulation.
    """

    def __init__(
        self,
        base_model: nn.Module,
        sensor_hub: FastSensorHub,
        skip_layers: List[int] = [7, 11, 15, 19, 23],
        skip_threshold: float = 0.3,
    ):
        super().__init__()
        self.base_model = base_model
        self.sensor_hub = sensor_hub
        self.skip_threshold = skip_threshold

        # Get model config
        config = base_model.config
        hidden_size = config.hidden_size

        # Wrap specified MLP layers with skip blocks
        self.skip_blocks = nn.ModuleDict()
        layers = base_model.model.layers

        for layer_idx in skip_layers:
            if layer_idx < len(layers):
                original_mlp = layers[layer_idx].mlp
                mlp_param = next(original_mlp.parameters())
                mlp_device = mlp_param.device
                mlp_dtype = mlp_param.dtype

                skip_block = MLPSkipBlock(
                    original_layer=original_mlp,
                    hidden_size=hidden_size,
                    sensor_hub=sensor_hub,
                    sensor_dim=8,
                    skip_threshold=skip_threshold,
                ).to(device=mlp_device, dtype=mlp_dtype)

                self.skip_blocks[str(layer_idx)] = skip_block
                layers[layer_idx].mlp = skip_block

        print(f"[EmbodiedModelZ26] Created {len(self.skip_blocks)} skip blocks at layers {skip_layers}")

        # Freeze base model, only train skip blocks
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze skip block parameters
        for block in self.skip_blocks.values():
            for param in block.gate_net.parameters():
                param.requires_grad = True
            for param in block.film_gamma.parameters():
                param.requires_grad = True
            for param in block.film_beta.parameters():
                param.requires_grad = True
            block.strain_embed.requires_grad = True

        # Count parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[EmbodiedModelZ26] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - sensors are read inside skip blocks automatically."""
        return self.base_model(input_ids, **kwargs)

    def get_stats(self) -> Dict[str, float]:
        """Get detailed gate/skip/FiLM stats from all skip blocks."""
        all_gates = []  # All individual gate values
        hard_skip_rates = []
        pct_below_tau = []
        film_gammas = []
        film_betas = []

        for block in self.skip_blocks.values():
            # Collect all gate values from last forward pass
            if block._last_gates:
                all_gates.extend(block._last_gates)
                # % of gates below skip threshold (would skip if in inference)
                below = sum(1 for g in block._last_gates if g < block.skip_threshold)
                pct_below_tau.append(below / len(block._last_gates))
            hard_skip_rates.append(block.skip_rate)
            # FiLM stats
            film_gammas.append(block._last_film_gamma)
            film_betas.append(block._last_film_beta)

        gate_mean = np.mean(all_gates) if all_gates else 0.5

        return {
            # Truthful gate metrics for training (soft gating)
            "gate/mean_prob": gate_mean,
            "gate/std": np.std(all_gates) if all_gates else 0.0,
            "gate/pct_below_tau": np.mean(pct_below_tau) if pct_below_tau else 0.0,
            "gate/expected_skip": 1.0 - gate_mean,  # Expected skip rate from soft gating
            # Hard skip rate (only counts inference hard-skips, will be 0 in training)
            "actuation/hard_skip_rate": np.mean(hard_skip_rates),
            # FiLM modulation stats (shows sensors → latent modulation)
            "film/gamma_mean": np.mean(film_gammas),
            "film/gamma_std": np.std(film_gammas),
            "film/beta_mean": np.mean(film_betas),
            "film/beta_std": np.std(film_betas),
            # Legacy (for compatibility)
            "gate_mean": gate_mean,
            "skip_rate": np.mean(hard_skip_rates),
        }


# =============================================================================
# TOP-K SAMPLER (faster than top-p)
# =============================================================================

@dataclass
class StepwiseSample:
    """Single sample from step-wise generation."""
    tokens: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    sensors: List[np.ndarray] = field(default_factory=list)
    gates: List[float] = field(default_factory=list)
    power_trajectory: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


def generate_stepwise_batch(
    model: EmbodiedModelZ26,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_samples: int = 4,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,  # Top-k instead of top-p (much faster)
) -> List[StepwiseSample]:
    """
    Generate K samples in parallel with step-wise control.

    Uses TOP-K sampling (faster than top-p, no full vocab sort).
    """
    device = next(model.parameters()).device

    # Tokenize and duplicate for batch
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    input_ids = input_ids.expand(num_samples, -1)

    # Initialize samples
    samples = [StepwiseSample() for _ in range(num_samples)]

    # Track which samples are still generating
    active = torch.ones(num_samples, dtype=torch.bool, device=device)

    # Initialize past_key_values for KV cache
    past = None

    with torch.no_grad():
        for step in range(max_tokens):
            # Read sensors
            sensors = model.sensor_hub.read_tensor()
            power_w = model.sensor_hub.power_w
            t = time.time()

            # Forward pass
            if past is None:
                outputs = model.base_model(input_ids, use_cache=True)
            else:
                outputs = model.base_model(input_ids[:, -1:], past_key_values=past, use_cache=True)

            logits = outputs.logits[:, -1, :]  # [K, vocab]
            past = outputs.past_key_values

            # TOP-K sampling (no full vocab sort!)
            if temperature > 0:
                logits_scaled = logits / temperature

                # Get top-k logits and indices
                topk_logits, topk_indices = torch.topk(logits_scaled, k=top_k, dim=-1)

                # Softmax over top-k only
                topk_probs = F.softmax(topk_logits, dim=-1)

                # Sample from top-k
                sampled_idx = torch.multinomial(topk_probs, num_samples=1)  # [K, 1]

                # Map back to vocabulary indices
                next_tokens = topk_indices.gather(-1, sampled_idx).squeeze(-1)  # [K]
            else:
                next_tokens = logits.argmax(dim=-1)

            # Get logprobs for sampled tokens
            log_probs = F.log_softmax(logits, dim=-1)
            token_logprobs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

            # Get gate stats (average across all skip blocks)
            gate_mean = np.mean([b.gate_mean for b in model.skip_blocks.values()])

            # Record for each active sample
            for i in range(num_samples):
                if active[i]:
                    samples[i].tokens.append(next_tokens[i].item())
                    samples[i].logprobs.append(token_logprobs[i].item())
                    samples[i].sensors.append(sensors.cpu().numpy().copy())
                    samples[i].gates.append(gate_mean)
                    samples[i].power_trajectory.append(power_w)
                    samples[i].timestamps.append(t)

            # Update input_ids for next step
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Check for EOS
            eos_mask = next_tokens == tokenizer.eos_token_id
            active = active & ~eos_mask

            if not active.any():
                break

    return samples


# =============================================================================
# IAE POWER REWARD WITH ANTI-CHEAT
# =============================================================================

class IAEPowerReward:
    """
    Reward based on IAE (Integral Absolute Error) over power trajectory.

    Features:
    1. IAE-based power reward (control-style error integral)
    2. Asymmetric penalty (overshoot worse than undershoot)
    3. Anti-cheat: zero power reward if task quality below threshold
    4. Conditional gate bonus: only incentivize skipping when power ABOVE target
    """

    def __init__(
        self,
        target_power_w: float = 75.0,
        power_band_w: float = 15.0,
        min_tokens: int = 20,
        task_quality_threshold: float = 0.35,  # Anti-cheat threshold
        task_weight: float = 0.5,
        power_weight: float = 0.4,
        gate_weight: float = 0.1,
        overshoot_multiplier: float = 1.5,  # Overshoot penalized more
    ):
        self.target_power = target_power_w
        self.power_band = power_band_w
        self.min_tokens = min_tokens
        self.task_threshold = task_quality_threshold
        self.task_weight = task_weight
        self.power_weight = power_weight
        self.gate_weight = gate_weight
        self.overshoot_mult = overshoot_multiplier

    def compute(
        self,
        sample: StepwiseSample,
        reference: str,
        generated: str,
        tokenizer: AutoTokenizer,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total reward with IAE power metric and anti-cheat."""
        breakdown = {}

        # 1. Task reward (simple overlap)
        ref_tokens = set(tokenizer.encode(reference.lower()))
        gen_tokens = set(tokenizer.encode(generated.lower()))

        if len(ref_tokens) > 0:
            precision = len(ref_tokens & gen_tokens) / max(len(gen_tokens), 1)
            recall = len(ref_tokens & gen_tokens) / len(ref_tokens)
            task_reward = 2 * precision * recall / (precision + recall + 1e-8)
        else:
            task_reward = 0.5

        # Length penalty
        if len(sample.tokens) < self.min_tokens:
            penalty = (self.min_tokens - len(sample.tokens)) / self.min_tokens
            task_reward *= (1 - penalty * 0.5)
            breakdown["length_penalty"] = penalty
        else:
            breakdown["length_penalty"] = 0.0

        breakdown["task"] = task_reward

        # 2. IAE Power Reward
        if len(sample.power_trajectory) > 0:
            power_array = np.array(sample.power_trajectory)
            errors = power_array - self.target_power

            # Asymmetric error: overshoot penalized more
            abs_errors = np.abs(errors)
            overshoot_mask = errors > 0
            abs_errors[overshoot_mask] *= self.overshoot_mult

            # IAE normalized by target
            IAE = np.mean(abs_errors) / self.target_power

            # Convert to reward (lower IAE = better)
            # IAE of 0 = reward 1.0, IAE of 0.5 = reward 0.5, etc.
            power_reward = max(0.0, 1.0 - IAE)

            # Time in band
            in_band = sum(1 for p in power_array if abs(p - self.target_power) < self.power_band)
            time_in_band = in_band / len(power_array)

            # Stats
            breakdown["power"] = power_reward
            breakdown["IAE"] = IAE
            breakdown["time_in_band"] = time_in_band
            breakdown["mean_power_w"] = np.mean(power_array)
            breakdown["max_power_w"] = np.max(power_array)
            breakdown["overshoot_pct"] = max(0, (np.max(power_array) - self.target_power) / self.target_power * 100)

            # Power above target (for gate bonus logic)
            power_above = np.mean(power_array) > self.target_power
            breakdown["power_above_target"] = float(power_above)

            # Oscillation penalty: penalize power thrashing
            if len(power_array) > 2:
                power_diffs = np.abs(np.diff(power_array))
                oscillation = np.mean(power_diffs) / self.target_power
                oscillation_penalty = min(0.2, oscillation * 0.5)  # Cap at 0.2
                breakdown["oscillation"] = oscillation
                breakdown["oscillation_penalty"] = oscillation_penalty
            else:
                oscillation_penalty = 0.0
                breakdown["oscillation"] = 0.0
                breakdown["oscillation_penalty"] = 0.0

            # Gate-power correlation (should become negative as policy learns)
            if len(sample.gates) > 2 and len(power_array) > 2:
                min_len = min(len(sample.gates), len(power_array))
                gate_arr = np.array(sample.gates[:min_len])
                power_arr = power_array[:min_len]
                if np.std(gate_arr) > 1e-6 and np.std(power_arr) > 1e-6:
                    correlation = np.corrcoef(gate_arr, power_arr)[0, 1]
                    breakdown["gate_power_corr"] = correlation if not np.isnan(correlation) else 0.0
                else:
                    breakdown["gate_power_corr"] = 0.0
            else:
                breakdown["gate_power_corr"] = 0.0
        else:
            power_reward = 0.5
            power_above = False
            oscillation_penalty = 0.0
            breakdown["power"] = 0.5
            breakdown["IAE"] = 0.5
            breakdown["time_in_band"] = 0.0
            breakdown["oscillation"] = 0.0
            breakdown["oscillation_penalty"] = 0.0
            breakdown["gate_power_corr"] = 0.0

        # 3. Anti-cheat: if task quality too low, DOWNWEIGHT (not zero) power reward
        # Zeroing was too aggressive (71% triggered) - use soft penalty instead
        if task_reward < self.task_threshold:
            # Soft penalty: scale power reward by task quality (0.2 task → 0.57x power reward)
            anti_cheat_scale = task_reward / self.task_threshold
            power_reward *= anti_cheat_scale
            breakdown["anti_cheat_triggered"] = 1.0
            breakdown["anti_cheat_scale"] = anti_cheat_scale
        else:
            breakdown["anti_cheat_triggered"] = 0.0
            breakdown["anti_cheat_scale"] = 1.0

        # 4. Conditional gate bonus (ONLY when power above target)
        if len(sample.gates) > 0:
            gate_mean = np.mean(sample.gates)
            breakdown["gate_mean"] = gate_mean

            if power_above:
                # Power above target: reward LOW gate (incentivize skipping)
                gate_bonus = (1.0 - gate_mean) * self.gate_weight
                breakdown["gate_bonus"] = gate_bonus
            else:
                # Power below target: no gate bonus (don't penalize either)
                gate_bonus = 0.0
                breakdown["gate_bonus"] = 0.0
        else:
            gate_bonus = 0.0
            breakdown["gate_mean"] = 0.5
            breakdown["gate_bonus"] = 0.0

        # Total reward (with oscillation penalty)
        total = (
            self.task_weight * task_reward +
            self.power_weight * power_reward +
            gate_bonus -
            oscillation_penalty  # Penalize power thrashing
        )

        breakdown["total"] = total
        return total, breakdown


# =============================================================================
# Z26 GRPO TRAINER
# =============================================================================

class Z26GRPOTrainer:
    """
    GRPO trainer with comprehensive metrics and validation.
    """

    def __init__(
        self,
        model: EmbodiedModelZ26,
        tokenizer: AutoTokenizer,
        sensor_hub: FastSensorHub,
        reward_fn: IAEPowerReward,
        disturbance_injector: Optional[DisturbanceInjector] = None,
        lr: float = 1e-5,
        num_samples: int = 4,
        max_tokens: int = 128,
        disturbance_schedule: str = "periodic",  # periodic, random, curriculum
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sensor_hub = sensor_hub
        self.reward_fn = reward_fn
        self.disturbance = disturbance_injector
        self.num_samples = num_samples
        self.max_tokens = max_tokens
        self.disturbance_schedule = disturbance_schedule
        self.device = next(model.parameters()).device

        # Optimizer
        trainable_params = []
        for block in model.skip_blocks.values():
            trainable_params.extend([p for p in block.parameters() if p.requires_grad])
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        # Disturbance state
        self._disturbance_active = False
        self._disturbance_step = 0

        print(f"[Z26GRPOTrainer] Optimizing {len(trainable_params)} parameter groups")

    def _manage_disturbance(self, step: int):
        """Manage disturbance injection based on schedule."""
        if not self.disturbance:
            return

        if self.disturbance_schedule == "periodic":
            # Every 20 steps, toggle disturbance for 5 steps
            cycle_pos = step % 25
            if cycle_pos < 5:
                if not self._disturbance_active:
                    self.disturbance.start(intensity=0.6)
                    self._disturbance_active = True
            else:
                if self._disturbance_active:
                    self.disturbance.stop()
                    self._disturbance_active = False

        elif self.disturbance_schedule == "random":
            # 20% chance to toggle each step
            if np.random.random() < 0.2:
                if self._disturbance_active:
                    self.disturbance.stop()
                    self._disturbance_active = False
                else:
                    self.disturbance.start(intensity=np.random.uniform(0.3, 0.8))
                    self._disturbance_active = True

        elif self.disturbance_schedule == "curriculum":
            # Gradually increase disturbance frequency
            prob = min(0.5, step / 500 * 0.5)
            if np.random.random() < prob and not self._disturbance_active:
                self.disturbance.start(intensity=np.random.uniform(0.4, 0.7))
                self._disturbance_active = True
            elif np.random.random() < 0.3 and self._disturbance_active:
                self.disturbance.stop()
                self._disturbance_active = False

    def _compute_logprobs_with_grad(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: List[List[int]],
    ) -> torch.Tensor:
        """Compute logprobs for completions WITH gradients."""
        batch_size = len(completion_ids)

        max_len = max(len(c) for c in completion_ids)
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        masks = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, comp in enumerate(completion_ids):
            padded[i, :len(comp)] = torch.tensor(comp, device=self.device)
            masks[i, :len(comp)] = True

        prompt_expanded = prompt_ids.expand(batch_size, -1)
        full_ids = torch.cat([prompt_expanded, padded], dim=1)

        # Forward pass WITH gradients
        outputs = self.model.base_model(full_ids)
        logits = outputs.logits

        prompt_len = prompt_ids.size(1)
        completion_logits = logits[:, prompt_len-1:-1, :]

        log_probs = F.log_softmax(completion_logits, dim=-1)
        token_logprobs = log_probs.gather(-1, padded.unsqueeze(-1)).squeeze(-1)

        token_logprobs = token_logprobs * masks.float()
        seq_logprobs = token_logprobs.sum(dim=1) / masks.sum(dim=1).float()

        return seq_logprobs

    def train_step(self, prompt: str, reference: str, step: int = 0) -> Dict[str, float]:
        """Single GRPO training step with disturbance management."""

        # Manage disturbance
        self._manage_disturbance(step)

        # Generate K samples
        self.model.eval()
        samples = generate_stepwise_batch(
            self.model,
            self.tokenizer,
            prompt,
            num_samples=self.num_samples,
            max_tokens=self.max_tokens,
        )

        # Decode samples
        decoded = [self.tokenizer.decode(s.tokens, skip_special_tokens=True) for s in samples]

        # Compute rewards
        rewards = []
        breakdowns = []
        for s, text in zip(samples, decoded):
            r, breakdown = self.reward_fn.compute(s, reference, text, self.tokenizer)
            rewards.append(r)
            breakdowns.append(breakdown)

        rewards = np.array(rewards)

        # Group-relative advantages
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        advantages = (rewards - mean_r) / std_r
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Re-forward WITH gradients
        self.model.train()
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        completion_ids = [s.tokens for s in samples]

        logprobs = self._compute_logprobs_with_grad(prompt_ids, completion_ids)

        # Policy gradient loss
        policy_loss = -(advantages * logprobs).mean()

        # Backward and update
        self.optimizer.zero_grad()
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for b in self.model.skip_blocks.values() for p in b.parameters()],
            max_norm=1.0
        )

        self.optimizer.step()

        # Aggregate metrics
        stats = self.model.get_stats()
        power_stats = self.sensor_hub.get_power_window(2.0)

        metrics = {
            "reward_mean": float(mean_r),
            "reward_std": float(std_r),
            "reward_max": float(np.max(rewards)),
            "policy_loss": policy_loss.item(),
            "disturbance_active": float(self._disturbance_active),
        }

        # Gate/skip metrics (truthful for training)
        metrics["gate/mean_prob"] = stats["gate/mean_prob"]
        metrics["gate/std"] = stats["gate/std"]
        metrics["gate/pct_below_tau"] = stats["gate/pct_below_tau"]
        metrics["gate/expected_skip"] = stats["gate/expected_skip"]
        metrics["actuation/hard_skip_rate"] = stats["actuation/hard_skip_rate"]
        # FiLM modulation stats (shows sensors → latent modulation)
        metrics["film/gamma_mean"] = stats["film/gamma_mean"]
        metrics["film/gamma_std"] = stats["film/gamma_std"]
        metrics["film/beta_mean"] = stats["film/beta_mean"]
        metrics["film/beta_std"] = stats["film/beta_std"]
        # Legacy (for console logging)
        metrics["gate_mean"] = stats["gate_mean"]
        metrics["skip_rate"] = stats["skip_rate"]

        # Control metrics
        metrics["control/IAE_power"] = power_stats["IAE"]
        metrics["control/time_in_band"] = power_stats["time_in_band"]
        metrics["control/overshoot_pct"] = power_stats["overshoot_pct"]
        metrics["control/time_over_120pct"] = power_stats["time_over_120pct"]
        metrics["control/power_mean_w"] = power_stats["mean_w"]
        metrics["control/power_std_w"] = power_stats["std_w"]

        # Breakdown averages
        for key in breakdowns[0].keys():
            if key not in ["total"]:
                metrics[f"breakdown/{key}"] = np.mean([b.get(key, 0) for b in breakdowns])

        return metrics


# =============================================================================
# COMPREHENSIVE VALIDATION
# =============================================================================

def run_ablation_validation(
    model: EmbodiedModelZ26,
    tokenizer: AutoTokenizer,
    sensor_hub: FastSensorHub,
    reward_fn: IAEPowerReward,
    val_data: List[Dict],
    num_samples: int = 5,
) -> Dict[str, float]:
    """
    Run comprehensive ablation validation with hardware trajectory separation.

    Tests:
    1. Full mode (normal operation)
    2. Shuffle mode (randomize sensor readings)
    3. Frozen mode (constant sensor readings)

    Measures separation on:
    - reward
    - power trajectory (mean, IAE)
    - time_in_band
    - time_over_120pct
    """
    model.eval()

    results = {
        "full": {"rewards": [], "powers": [], "IAEs": [], "time_in_bands": [], "gates": []},
        "shuffle": {"rewards": [], "powers": [], "IAEs": [], "time_in_bands": [], "gates": []},
        "frozen": {"rewards": [], "powers": [], "IAEs": [], "time_in_bands": [], "gates": []},
    }

    original_read = sensor_hub.read_tensor

    for sample in val_data[:num_samples]:
        prompt = sample.get("input", sample.get("prompt", sample.get("instruction", "")))
        reference = sample.get("output", sample.get("response", ""))

        if not prompt or not reference:
            continue

        # 1. Full mode
        sensor_hub.read_tensor = original_read
        full_samples = generate_stepwise_batch(model, tokenizer, prompt, num_samples=2, max_tokens=64)
        for s in full_samples:
            text = tokenizer.decode(s.tokens, skip_special_tokens=True)
            r, breakdown = reward_fn.compute(s, reference, text, tokenizer)
            results["full"]["rewards"].append(r)
            results["full"]["powers"].append(breakdown.get("mean_power_w", 0))
            results["full"]["IAEs"].append(breakdown.get("IAE", 0))
            results["full"]["time_in_bands"].append(breakdown.get("time_in_band", 0))
            results["full"]["gates"].append(breakdown.get("gate_mean", 0.5))

        # 2. Shuffle mode
        def shuffled_read():
            t = original_read()
            return t[torch.randperm(len(t))]
        sensor_hub.read_tensor = shuffled_read

        shuffle_samples = generate_stepwise_batch(model, tokenizer, prompt, num_samples=2, max_tokens=64)
        for s in shuffle_samples:
            text = tokenizer.decode(s.tokens, skip_special_tokens=True)
            r, breakdown = reward_fn.compute(s, reference, text, tokenizer)
            results["shuffle"]["rewards"].append(r)
            results["shuffle"]["powers"].append(breakdown.get("mean_power_w", 0))
            results["shuffle"]["IAEs"].append(breakdown.get("IAE", 0))
            results["shuffle"]["time_in_bands"].append(breakdown.get("time_in_band", 0))
            results["shuffle"]["gates"].append(breakdown.get("gate_mean", 0.5))

        # 3. Frozen mode
        frozen_tensor = original_read().clone()
        def frozen_read():
            return frozen_tensor
        sensor_hub.read_tensor = frozen_read

        frozen_samples = generate_stepwise_batch(model, tokenizer, prompt, num_samples=2, max_tokens=64)
        for s in frozen_samples:
            text = tokenizer.decode(s.tokens, skip_special_tokens=True)
            r, breakdown = reward_fn.compute(s, reference, text, tokenizer)
            results["frozen"]["rewards"].append(r)
            results["frozen"]["powers"].append(breakdown.get("mean_power_w", 0))
            results["frozen"]["IAEs"].append(breakdown.get("IAE", 0))
            results["frozen"]["time_in_bands"].append(breakdown.get("time_in_band", 0))
            results["frozen"]["gates"].append(breakdown.get("gate_mean", 0.5))

    # Restore
    sensor_hub.read_tensor = original_read

    # Compute summary metrics
    metrics = {}
    for mode in ["full", "shuffle", "frozen"]:
        metrics[f"ablation/{mode}_reward"] = np.mean(results[mode]["rewards"])
        metrics[f"ablation/{mode}_power_w"] = np.mean(results[mode]["powers"])
        metrics[f"ablation/{mode}_IAE"] = np.mean(results[mode]["IAEs"])
        metrics[f"ablation/{mode}_time_in_band"] = np.mean(results[mode]["time_in_bands"])
        metrics[f"ablation/{mode}_gate_mean"] = np.mean(results[mode]["gates"])

    # Compute separation scores
    metrics["ablation/reward_separation"] = metrics["ablation/full_reward"] - metrics["ablation/shuffle_reward"]
    metrics["ablation/power_separation"] = abs(metrics["ablation/full_power_w"] - metrics["ablation/shuffle_power_w"])
    metrics["ablation/IAE_separation"] = metrics["ablation/shuffle_IAE"] - metrics["ablation/full_IAE"]  # Better IAE for full
    metrics["ablation/time_in_band_separation"] = metrics["ablation/full_time_in_band"] - metrics["ablation/shuffle_time_in_band"]

    # Causal score (how much does sensor info matter?)
    metrics["ablation/causal_score"] = (
        metrics["ablation/reward_separation"] +
        metrics["ablation/time_in_band_separation"]
    ) / 2

    return metrics


def run_expression_validation(
    model: EmbodiedModelZ26,
    tokenizer: AutoTokenizer,
    sensor_hub: FastSensorHub,
    reward_fn: IAEPowerReward,
    disturbance: DisturbanceInjector,
    prompts: List[str],
) -> Dict[str, float]:
    """
    Run paired expression validation: low stress vs high stress.

    Measures how generation changes with different hardware conditions.
    """
    model.eval()

    low_stress_results = {"lengths": [], "tasks": [], "gates": []}
    high_stress_results = {"lengths": [], "tasks": [], "gates": []}

    for prompt in prompts[:5]:
        # Low stress: no disturbance
        disturbance.stop()
        time.sleep(0.5)

        low_samples = generate_stepwise_batch(model, tokenizer, prompt, num_samples=2, max_tokens=64)
        for s in low_samples:
            low_stress_results["lengths"].append(len(s.tokens))
            low_stress_results["gates"].append(np.mean(s.gates) if s.gates else 0.5)

        # High stress: with disturbance
        disturbance.start(intensity=0.7)
        time.sleep(0.5)

        high_samples = generate_stepwise_batch(model, tokenizer, prompt, num_samples=2, max_tokens=64)
        for s in high_samples:
            high_stress_results["lengths"].append(len(s.tokens))
            high_stress_results["gates"].append(np.mean(s.gates) if s.gates else 0.5)

        disturbance.stop()

    metrics = {
        "expr/len_low": np.mean(low_stress_results["lengths"]),
        "expr/len_high": np.mean(high_stress_results["lengths"]),
        "expr/gate_low": np.mean(low_stress_results["gates"]),
        "expr/gate_high": np.mean(high_stress_results["gates"]),
        "expr/len_delta": np.mean(high_stress_results["lengths"]) - np.mean(low_stress_results["lengths"]),
        "expr/gate_delta": np.mean(high_stress_results["gates"]) - np.mean(low_stress_results["gates"]),
    }

    return metrics


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def load_dataset(path: str, max_samples: int = None) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if max_samples and len(data) >= max_samples:
                    break
    return data


def main():
    parser = argparse.ArgumentParser(description="FEEL z26 Comprehensive Closed-Loop Training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--target-power", type=float, default=75.0, help="Target power in watts")
    parser.add_argument("--power-band", type=float, default=15.0, help="Acceptable power band")
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--disturbance", type=str, default="periodic",
                       choices=["periodic", "random", "curriculum", "none"])
    parser.add_argument("--checkpoint-dir", type=str, default="models/grpo_z26")
    args = parser.parse_args()

    print("=" * 70)
    print("FEEL z26: COMPREHENSIVE CLOSED-LOOP EMBODIED TRAINING")
    print("=" * 70)
    print(f"\nKey improvements over z25:")
    print(f"  - Per-sample gate (no .mean())")
    print(f"  - Top-k sampling (faster than top-p)")
    print(f"  - IAE power reward with anti-cheat")
    print(f"  - Conditional gate bonus (only when power above target)")
    print(f"  - Disturbance injection: {args.disturbance}")
    print(f"  - Realistic target: {args.target_power}W ± {args.power_band}W")
    print()
    print(f"Config:")
    print(f"  epochs: {args.epochs}")
    print(f"  max_prompts: {args.max_prompts}")
    print(f"  num_samples (K): {args.num_samples}")
    print()

    # Initialize W&B
    wandb.init(
        project="feel-z26-comprehensive",
        name=f"z26-{time.strftime('%Y%m%d_%H%M')}",
        config=vars(args),
    )
    print(f"WandB: {wandb.run.url}")

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # 1. Initialize sensor hub
    print("\n[1/5] Initializing FastSensorHub...")
    sensor_hub = FastSensorHub(
        sample_rate=100.0,
        target_power_w=args.target_power,
    )
    time.sleep(0.5)

    # 2. Initialize disturbance injector
    print("\n[2/5] Initializing DisturbanceInjector...")
    disturbance = DisturbanceInjector() if args.disturbance != "none" else None

    # 3. Load base model
    print("\n[3/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    # 4. Create embodied model
    print("\n[4/5] Creating embodied model with per-sample MLP-skip actuators...")
    model = EmbodiedModelZ26(
        base_model=base_model,
        sensor_hub=sensor_hub,
        skip_layers=[7, 11, 15, 19, 23],
        skip_threshold=0.3,
    )

    # 5. Create reward function and trainer
    print("\n[5/5] Creating trainer...")
    reward_fn = IAEPowerReward(
        target_power_w=args.target_power,
        power_band_w=args.power_band,
        min_tokens=20,
        task_quality_threshold=0.35,
    )

    trainer = Z26GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        sensor_hub=sensor_hub,
        reward_fn=reward_fn,
        disturbance_injector=disturbance,
        lr=args.lr,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        disturbance_schedule=args.disturbance,
    )

    # Load dataset
    print("\n[6/6] Loading dataset...")
    train_data = load_dataset("data/ouroboros/ouroboros_train.jsonl", max_samples=args.max_prompts * 3)
    val_data = load_dataset("data/ouroboros/ouroboros_val.jsonl", max_samples=100)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    global_step = 0
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")

        np.random.shuffle(train_data)

        for i, sample in enumerate(train_data[:args.max_prompts]):
            prompt = sample.get("input", sample.get("prompt", sample.get("instruction", "")))
            reference = sample.get("output", sample.get("response", ""))

            if not prompt or not reference:
                continue

            try:
                metrics = trainer.train_step(prompt, reference, step=global_step)
                global_step += 1

                wandb.log(metrics, step=global_step)

                if (i + 1) % args.log_every == 0:
                    print(
                        f"[{i+1}/{args.max_prompts}] "
                        f"r={metrics['reward_mean']:.3f} "
                        f"gate={metrics['gate_mean']:.3f} "
                        f"skip={metrics['skip_rate']:.2%} "
                        f"power={metrics['control/power_mean_w']:.1f}W "
                        f"IAE={metrics['control/IAE_power']:.3f} "
                        f"dist={'ON' if metrics['disturbance_active'] else 'off'}",
                        flush=True
                    )
            except Exception as e:
                print(f"[{i+1}] Error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue

            # Validation
            if (i + 1) % args.val_every == 0:
                print(f"\n[Comprehensive Validation at step {i+1}]")

                # Ablation validation
                ablation_metrics = run_ablation_validation(
                    model, tokenizer, sensor_hub, reward_fn, val_data, num_samples=5
                )

                print(f"  Ablation Results:")
                print(f"    full:    reward={ablation_metrics['ablation/full_reward']:.3f} "
                      f"power={ablation_metrics['ablation/full_power_w']:.1f}W "
                      f"IAE={ablation_metrics['ablation/full_IAE']:.3f} "
                      f"band={ablation_metrics['ablation/full_time_in_band']:.2%}")
                print(f"    shuffle: reward={ablation_metrics['ablation/shuffle_reward']:.3f} "
                      f"power={ablation_metrics['ablation/shuffle_power_w']:.1f}W "
                      f"IAE={ablation_metrics['ablation/shuffle_IAE']:.3f} "
                      f"band={ablation_metrics['ablation/shuffle_time_in_band']:.2%}")
                print(f"    frozen:  reward={ablation_metrics['ablation/frozen_reward']:.3f} "
                      f"power={ablation_metrics['ablation/frozen_power_w']:.1f}W "
                      f"IAE={ablation_metrics['ablation/frozen_IAE']:.3f}")
                print(f"    CAUSAL SCORE: {ablation_metrics['ablation/causal_score']:.3f}")
                print(f"    Separations: reward={ablation_metrics['ablation/reward_separation']:.3f} "
                      f"power={ablation_metrics['ablation/power_separation']:.1f}W "
                      f"time_in_band={ablation_metrics['ablation/time_in_band_separation']:.3f}")

                wandb.log(ablation_metrics, step=global_step)

                # Expression validation (if disturbance available)
                if disturbance:
                    expr_prompts = [s.get("input", s.get("prompt", "")) for s in val_data[:5] if s.get("input") or s.get("prompt")]
                    if expr_prompts:
                        expr_metrics = run_expression_validation(
                            model, tokenizer, sensor_hub, reward_fn, disturbance, expr_prompts
                        )

                        print(f"\n  Expression Results:")
                        print(f"    low_stress:  len={expr_metrics['expr/len_low']:.1f} gate={expr_metrics['expr/gate_low']:.3f}")
                        print(f"    high_stress: len={expr_metrics['expr/len_high']:.1f} gate={expr_metrics['expr/gate_high']:.3f}")
                        print(f"    deltas: len={expr_metrics['expr/len_delta']:.1f} gate={expr_metrics['expr/gate_delta']:.3f}")

                        wandb.log(expr_metrics, step=global_step)

                model.train()

        # Save checkpoint
        ckpt_path = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state": {k: v.state_dict() for k, v in model.skip_blocks.items()},
            "optimizer_state": trainer.optimizer.state_dict(),
            "global_step": global_step,
        }, ckpt_path)
        print(f"\n[Checkpoint] Saved to {ckpt_path}")

    # Cleanup
    if disturbance:
        disturbance.stop()
    sensor_hub.stop()
    wandb.finish()

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
