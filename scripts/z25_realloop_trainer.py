#!/usr/bin/env python3
"""
FEEL z25: Real Closed-Loop Embodied Training

Key improvements over z24:
1. Custom step-wise sampler (no output_scores overhead)
2. Batched K-sample generation in one forward pass
3. Power-window reward over 2-5s with anti-cheat
4. Strong MLP-skip actuator with real compute savings
5. Comprehensive W&B metrics for sensing/regulation/expression

The goal: sense → modulate → actuate → hardware changes → new sensing
"""

import os
import sys
import time
import json
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
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

    Slow signals (10Hz):
      - temperature
    """

    HWMON_PATH = "/sys/class/drm/card1/device/hwmon/hwmon7"
    DEVICE_PATH = "/sys/class/drm/card1/device"

    # Power limits for normalization (RX 7900 XTX)
    POWER_MIN = 20_000_000   # 20W idle
    POWER_MAX = 350_000_000  # 350W max
    POWER_TARGET = 200_000_000  # 200W target for regulation

    def __init__(self, sample_rate: float = 100.0, device: str = "cuda"):
        self.sample_rate = sample_rate
        self.device = device
        self._lock = threading.Lock()
        self._running = False

        # Current state (fast signals)
        self._power_raw = 0  # microwatts
        self._power_norm = 0.5
        self._gpu_busy = 0.0
        self._sclk_state = 0  # 0, 1, 2
        self._throttle = False

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
        print(f"[FastSensorHub] Sampling at {self.sample_rate}Hz")

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

                # Read SCLK state (fast)
                with open(f"{self.DEVICE_PATH}/pp_dpm_sclk", "r") as f:
                    lines = f.read().strip().split('\n')
                    for i, line in enumerate(lines):
                        if '*' in line:
                            self._sclk_state = i
                            break

                # Normalize power
                self._power_norm = (self._power_raw - self.POWER_MIN) / (self.POWER_MAX - self.POWER_MIN)
                self._power_norm = max(0.0, min(1.0, self._power_norm))

                # Throttle detection (high power + not max clock)
                self._throttle = self._power_raw > self.POWER_TARGET * 0.95 and self._sclk_state < 2

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
                # Fast 8-dim vector
                state = np.array([
                    self._power_norm,           # 0: power (normalized)
                    self._gpu_busy,             # 1: GPU utilization
                    self._sclk_state / 2.0,     # 2: clock state (0-1)
                    float(self._throttle),      # 3: throttle flag
                    self._power_raw / 1e8,      # 4: raw power (100W scale)
                    np.std(list(self._power_history)[-50:]) / 1e7 if len(self._power_history) > 10 else 0,  # 5: power variance
                    len([p for p in list(self._power_history)[-100:] if p > self.POWER_TARGET]) / 100.0 if len(self._power_history) > 10 else 0,  # 6: time over target
                    0.0,  # 7: reserved
                ], dtype=np.float32)

                if self._cached_tensor is None:
                    self._cached_tensor = torch.from_numpy(state).to(self.device)
                else:
                    self._cached_tensor.copy_(torch.from_numpy(state))
                self._tensor_dirty = False

            return self._cached_tensor

    def get_power_window(self, window_sec: float = 2.0) -> Tuple[float, float, float]:
        """Get power stats over recent window: (mean, std, time_over_target)."""
        with self._lock:
            if len(self._power_history) < 10:
                return self._power_raw / 1e6, 0.0, 0.0

            now = time.time()
            cutoff = now - window_sec

            # Get samples in window
            recent = []
            for t, p in zip(self._power_timestamps, self._power_history):
                if t >= cutoff:
                    recent.append(p)

            if not recent:
                recent = list(self._power_history)[-10:]

            mean_w = np.mean(recent) / 1e6  # Convert to watts
            std_w = np.std(recent) / 1e6
            time_over = sum(1 for p in recent if p > self.POWER_TARGET) / len(recent)

            return mean_w, std_w, time_over

    def stop(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=1.0)
        print("[FastSensorHub] Stopped")


# =============================================================================
# STRONG ACTUATOR: MLP-Skip Block
# =============================================================================

class MLPSkipBlock(nn.Module):
    """
    Embodied block that can skip MLP entirely for real compute savings.

    When gate < threshold:
      - Skip MLP computation entirely (real power savings)
      - Inject strain embedding as proprioceptive signal

    When gate >= threshold:
      - Run full MLP with FiLM modulation from sensors

    Sensors are read from the sensor_hub reference stored in the block.
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
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size + sensor_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # FiLM: modulates MLP output when not skipping
        self.film_gamma = nn.Linear(sensor_dim, hidden_size)
        self.film_beta = nn.Linear(sensor_dim, hidden_size)

        # Strain embedding: injected when skipping (proprioception)
        self.strain_embed = nn.Parameter(torch.randn(hidden_size) * 0.01)

        # Stats for logging
        self._last_gate = 0.5
        self._last_skipped = False
        self._skip_count = 0
        self._total_count = 0
        self._last_film_gamma = 1.0
        self._last_film_beta = 0.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional MLP skip.

        Args:
            hidden_states: [batch, seq, hidden]

        Note: Sensors are read directly from sensor_hub, not passed as argument.
        This allows the block to work as a drop-in MLP replacement.
        """
        batch, seq_len, _ = hidden_states.shape

        # Read current sensors from hub and cast to match hidden_states dtype
        sensors = self.sensor_hub.read_tensor().to(dtype=hidden_states.dtype)  # [sensor_dim]

        # Compute gate from last token + sensors
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]
        sensors_expanded = sensors.unsqueeze(0).expand(batch, -1)  # [batch, sensor_dim]
        gate_input = torch.cat([last_hidden, sensors_expanded], dim=-1)
        gate = self.gate_net(gate_input).mean()  # Scalar gate for whole batch

        self._last_gate = gate.item()
        self._total_count += 1

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

            # Soft mix: gate * modulated + (1-gate) * (hidden + strain)
            skip_out = hidden_states + self.strain_embed.view(1, 1, -1)
            output = gate * modulated + (1 - gate) * skip_out

            self._last_skipped = gate.item() < self.skip_threshold

        else:
            # Hard skip during inference (real compute savings)
            if gate.item() < self.skip_threshold:
                # SKIP: Don't run MLP at all - real power savings!
                output = hidden_states + self.strain_embed.view(1, 1, -1)
                self._last_skipped = True
                self._skip_count += 1
            else:
                # RUN: Full computation with FiLM
                layer_out = self.original_layer(hidden_states)

                gamma = 1.0 + self.film_gamma(sensors)
                beta = self.film_beta(sensors)
                self._last_film_gamma = gamma.mean().item()
                self._last_film_beta = beta.mean().item()

                output = gamma.view(1, 1, -1) * layer_out + beta.view(1, 1, -1)
                self._last_skipped = False

        return output

    @property
    def skip_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._skip_count / self._total_count


# =============================================================================
# EMBODIED MODEL WITH STRONG ACTUATORS
# =============================================================================

class EmbodiedModelZ25(nn.Module):
    """
    Embodied model with MLP-skip actuators for real power regulation.
    """

    def __init__(
        self,
        base_model: nn.Module,
        sensor_hub: FastSensorHub,
        skip_layers: List[int] = [7, 11, 15, 19, 23],  # MLP layers to make skippable
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
                # Get device and dtype from original MLP
                mlp_param = next(original_mlp.parameters())
                mlp_device = mlp_param.device
                mlp_dtype = mlp_param.dtype

                skip_block = MLPSkipBlock(
                    original_layer=original_mlp,
                    hidden_size=hidden_size,
                    sensor_hub=sensor_hub,  # Pass hub reference
                    sensor_dim=8,
                    skip_threshold=skip_threshold,
                ).to(device=mlp_device, dtype=mlp_dtype)  # Match device AND dtype

                self.skip_blocks[str(layer_idx)] = skip_block
                # Replace MLP with skip block
                layers[layer_idx].mlp = skip_block

        print(f"[EmbodiedModelZ25] Created {len(self.skip_blocks)} skip blocks at layers {skip_layers}")

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
        print(f"[EmbodiedModelZ25] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - sensors are read inside skip blocks automatically."""
        return self.base_model(input_ids, **kwargs)

    def get_stats(self) -> Dict[str, float]:
        """Get current stats from all skip blocks."""
        gates = []
        skip_rates = []
        for block in self.skip_blocks.values():
            gates.append(block._last_gate)
            skip_rates.append(block.skip_rate)
        return {
            "gate_mean": np.mean(gates),
            "gate_std": np.std(gates),
            "skip_rate": np.mean(skip_rates),
        }


# =============================================================================
# CUSTOM STEP-WISE SAMPLER (No output_scores overhead)
# =============================================================================

@dataclass
class StepwiseSample:
    """Single sample from step-wise generation."""
    tokens: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    sensors: List[np.ndarray] = field(default_factory=list)
    gates: List[float] = field(default_factory=list)
    power_trajectory: List[float] = field(default_factory=list)


def generate_stepwise_batch(
    model: EmbodiedModelZ25,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_samples: int = 4,
    max_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[StepwiseSample]:
    """
    Generate K samples in parallel with step-wise control.

    Key advantages:
    - No output_scores overhead (vocab-size tensors)
    - Batched generation (K samples in one forward pass)
    - Per-step sensor/gate logging
    - Power trajectory capture for windowed reward
    """
    device = next(model.parameters()).device

    # Tokenize and duplicate for batch
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)  # [1, seq]
    input_ids = input_ids.expand(num_samples, -1)  # [K, seq]

    # Initialize samples
    samples = [StepwiseSample() for _ in range(num_samples)]

    # Track which samples are still generating
    active = torch.ones(num_samples, dtype=torch.bool, device=device)

    # Initialize past_key_values for KV cache
    past = None

    with torch.no_grad():
        for step in range(max_tokens):
            # Read sensors (same for all samples in batch)
            sensors = model.sensor_hub.read_tensor()
            power_w = model.sensor_hub._power_raw / 1e6

            # Forward pass
            if past is None:
                outputs = model.base_model(input_ids, use_cache=True)
            else:
                outputs = model.base_model(input_ids[:, -1:], past_key_values=past, use_cache=True)

            logits = outputs.logits[:, -1, :]  # [K, vocab]
            past = outputs.past_key_values

            # Sample next tokens
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                # Top-p sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # Sample from filtered distribution
                next_tokens = torch.multinomial(sorted_probs, 1)  # [K, 1]
                next_tokens = sorted_indices.gather(-1, next_tokens)  # Map back to original indices
            else:
                next_tokens = logits.argmax(dim=-1, keepdim=True)

            next_tokens = next_tokens.squeeze(-1)  # [K]

            # Get logprobs for sampled tokens
            log_probs = F.log_softmax(logits, dim=-1)
            token_logprobs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)  # [K]

            # Get gate stats
            gate_mean = np.mean([b._last_gate for b in model.skip_blocks.values()])

            # Record for each active sample
            for i in range(num_samples):
                if active[i]:
                    samples[i].tokens.append(next_tokens[i].item())
                    samples[i].logprobs.append(token_logprobs[i].item())
                    samples[i].sensors.append(sensors.cpu().numpy().copy())
                    samples[i].gates.append(gate_mean)
                    samples[i].power_trajectory.append(power_w)

            # Update input_ids for next step
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Check for EOS
            eos_mask = next_tokens == tokenizer.eos_token_id
            active = active & ~eos_mask

            if not active.any():
                break

    return samples


# =============================================================================
# POWER-WINDOW REWARD WITH ANTI-CHEAT
# =============================================================================

class PowerWindowReward:
    """
    Reward based on power trajectory over time window.

    Components:
    1. Task reward (quality)
    2. Power regulation reward (stay in target band)
    3. Anti-cheat (minimum quality threshold)
    4. Oscillation penalty (discourage thrash)
    """

    def __init__(
        self,
        target_power_w: float = 200.0,
        power_band_w: float = 50.0,  # Acceptable range: target ± band
        window_sec: float = 2.0,
        min_tokens: int = 20,
        task_weight: float = 0.6,
        power_weight: float = 0.3,
        oscillation_weight: float = 0.1,
    ):
        self.target_power = target_power_w
        self.power_band = power_band_w
        self.window_sec = window_sec
        self.min_tokens = min_tokens
        self.task_weight = task_weight
        self.power_weight = power_weight
        self.oscillation_weight = oscillation_weight

    def compute(
        self,
        sample: StepwiseSample,
        reference: str,
        generated: str,
        tokenizer: AutoTokenizer,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward with breakdown.

        Returns:
            (total_reward, breakdown_dict)
        """
        breakdown = {}

        # 1. Task reward (simple overlap for now)
        ref_tokens = set(tokenizer.encode(reference.lower()))
        gen_tokens = set(tokenizer.encode(generated.lower()))

        if len(ref_tokens) > 0:
            precision = len(ref_tokens & gen_tokens) / max(len(gen_tokens), 1)
            recall = len(ref_tokens & gen_tokens) / len(ref_tokens)
            task_reward = 2 * precision * recall / (precision + recall + 1e-8)
        else:
            task_reward = 0.5

        breakdown["task"] = task_reward

        # Anti-cheat: minimum tokens required
        if len(sample.tokens) < self.min_tokens:
            penalty = (self.min_tokens - len(sample.tokens)) / self.min_tokens
            task_reward *= (1 - penalty * 0.5)
            breakdown["length_penalty"] = penalty

        # 2. Power regulation reward
        if len(sample.power_trajectory) > 0:
            power_array = np.array(sample.power_trajectory)
            mean_power = np.mean(power_array)

            # Distance from target band
            if mean_power < self.target_power - self.power_band:
                # Too low (mild penalty)
                power_reward = 0.8 - 0.2 * abs(mean_power - (self.target_power - self.power_band)) / self.target_power
            elif mean_power > self.target_power + self.power_band:
                # Too high (strong penalty)
                power_reward = 0.5 - 0.5 * (mean_power - (self.target_power + self.power_band)) / self.target_power
            else:
                # In band (reward)
                power_reward = 1.0

            power_reward = max(0.0, min(1.0, power_reward))
            breakdown["power"] = power_reward
            breakdown["mean_power_w"] = mean_power

            # Time in band
            in_band = sum(1 for p in power_array if abs(p - self.target_power) < self.power_band)
            breakdown["time_in_band"] = in_band / len(power_array)
        else:
            power_reward = 0.5
            breakdown["power"] = 0.5

        # 3. Oscillation penalty
        if len(sample.power_trajectory) > 2:
            diffs = np.abs(np.diff(sample.power_trajectory))
            oscillation = np.mean(diffs) / self.target_power  # Normalized
            oscillation_penalty = min(1.0, oscillation * 10)  # Scale up
            breakdown["oscillation"] = oscillation_penalty
        else:
            oscillation_penalty = 0.0
            breakdown["oscillation"] = 0.0

        # 4. Gate usage bonus (encourage using embodied computation)
        if len(sample.gates) > 0:
            gate_mean = np.mean(sample.gates)
            gate_bonus = gate_mean * 0.1  # Small bonus for using gates
            breakdown["gate_mean"] = gate_mean
        else:
            gate_bonus = 0.0

        # Total reward
        total = (
            self.task_weight * task_reward +
            self.power_weight * power_reward -
            self.oscillation_weight * oscillation_penalty +
            gate_bonus
        )

        breakdown["total"] = total
        return total, breakdown


# =============================================================================
# Z25 GRPO TRAINER
# =============================================================================

class Z25GRPOTrainer:
    """
    GRPO trainer with real closed-loop regulation.

    Uses REINFORCE-style policy gradient with:
    1. Generate K samples (no grad)
    2. Compute rewards
    3. Re-forward with grad to get logprobs
    4. Policy gradient update
    """

    def __init__(
        self,
        model: EmbodiedModelZ25,
        tokenizer: AutoTokenizer,
        sensor_hub: FastSensorHub,
        reward_fn: PowerWindowReward,
        lr: float = 1e-5,
        num_samples: int = 4,
        max_tokens: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sensor_hub = sensor_hub
        self.reward_fn = reward_fn
        self.num_samples = num_samples
        self.max_tokens = max_tokens
        self.device = next(model.parameters()).device

        # Optimizer (only skip block parameters)
        trainable_params = []
        for block in model.skip_blocks.values():
            trainable_params.extend([p for p in block.parameters() if p.requires_grad])
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        print(f"[Z25GRPOTrainer] Optimizing {len(trainable_params)} parameter groups")

    def _compute_logprobs_with_grad(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: List[List[int]],
    ) -> torch.Tensor:
        """
        Compute logprobs for completions WITH gradients.

        This is the critical part for policy gradient - we need gradients
        to flow through the skip blocks' gate decisions.
        """
        batch_size = len(completion_ids)

        # Pad completions to same length
        max_len = max(len(c) for c in completion_ids)
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        masks = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, comp in enumerate(completion_ids):
            padded[i, :len(comp)] = torch.tensor(comp, device=self.device)
            masks[i, :len(comp)] = True

        # Expand prompt for batch
        prompt_expanded = prompt_ids.expand(batch_size, -1)

        # Full sequences
        full_ids = torch.cat([prompt_expanded, padded], dim=1)

        # Forward pass WITH gradients
        outputs = self.model.base_model(full_ids)
        logits = outputs.logits  # [batch, seq, vocab]

        # Get logprobs for completion tokens only
        prompt_len = prompt_ids.size(1)
        completion_logits = logits[:, prompt_len-1:-1, :]  # Shifted for next-token prediction

        # Log softmax
        log_probs = F.log_softmax(completion_logits, dim=-1)

        # Gather logprobs for actual tokens
        token_logprobs = log_probs.gather(-1, padded.unsqueeze(-1)).squeeze(-1)  # [batch, max_len]

        # Mask and sum
        token_logprobs = token_logprobs * masks.float()
        seq_logprobs = token_logprobs.sum(dim=1) / masks.sum(dim=1).float()  # Mean logprob per sample

        return seq_logprobs  # [batch]

    def train_step(self, prompt: str, reference: str) -> Dict[str, float]:
        """
        Single GRPO training step.

        1. Generate K samples (no grad) with step-wise control
        2. Compute rewards for each
        3. Re-forward with grad to get logprobs
        4. Policy gradient update weighted by advantages
        """
        # Generate K samples (no grad, but capture power/gate trajectories)
        self.model.eval()
        samples = generate_stepwise_batch(
            self.model,
            self.tokenizer,
            prompt,
            num_samples=self.num_samples,
            max_tokens=self.max_tokens,
        )

        # Decode samples
        decoded = []
        for s in samples:
            text = self.tokenizer.decode(s.tokens, skip_special_tokens=True)
            decoded.append(text)

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

        # Policy gradient loss: -E[A * log π(a|s)]
        policy_loss = -(advantages * logprobs).mean()

        # Backward and update
        self.optimizer.zero_grad()
        policy_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for b in self.model.skip_blocks.values() for p in b.parameters()],
            max_norm=1.0
        )

        self.optimizer.step()

        # Aggregate metrics
        stats = self.model.get_stats()
        power_mean, power_std, time_over = self.sensor_hub.get_power_window(2.0)

        metrics = {
            "reward_mean": float(mean_r),
            "reward_std": float(std_r),
            "reward_max": float(np.max(rewards)),
            "gate_mean": stats["gate_mean"],
            "skip_rate": stats["skip_rate"],
            "power_mean_w": power_mean,
            "power_std_w": power_std,
            "time_over_target": time_over,
            "policy_loss": policy_loss.item(),
        }

        # Add breakdown averages
        for key in breakdowns[0].keys():
            if key != "total":
                metrics[f"breakdown/{key}"] = np.mean([b.get(key, 0) for b in breakdowns])

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
    parser = argparse.ArgumentParser(description="FEEL z25 Real Closed-Loop Training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-prompts", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--target-power", type=float, default=200.0, help="Target power in watts")
    parser.add_argument("--power-band", type=float, default=50.0, help="Acceptable power band")
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="models/grpo_z25")
    args = parser.parse_args()

    print("=" * 70)
    print("FEEL z25: REAL CLOSED-LOOP EMBODIED TRAINING")
    print("=" * 70)
    print(f"Config:")
    print(f"  epochs: {args.epochs}")
    print(f"  max_prompts: {args.max_prompts}")
    print(f"  num_samples (K): {args.num_samples}")
    print(f"  target_power: {args.target_power}W")
    print(f"  power_band: ±{args.power_band}W")
    print()

    # Initialize W&B
    wandb.init(
        project="feel-z25-realloop",
        name=f"z25-{time.strftime('%Y%m%d_%H%M')}",
        config=vars(args),
    )
    print(f"WandB: {wandb.run.url}")

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # 1. Initialize sensor hub
    print("\n[1/4] Initializing FastSensorHub...")
    sensor_hub = FastSensorHub(sample_rate=100.0)
    time.sleep(0.5)  # Let it collect initial samples

    # 2. Load base model
    print("\n[2/4] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    # 3. Create embodied model with skip blocks
    print("\n[3/4] Creating embodied model with MLP-skip actuators...")
    model = EmbodiedModelZ25(
        base_model=base_model,
        sensor_hub=sensor_hub,
        skip_layers=[7, 11, 15, 19, 23],
        skip_threshold=0.3,
    )

    # 4. Create reward function and trainer
    print("\n[4/4] Creating trainer...")
    reward_fn = PowerWindowReward(
        target_power_w=args.target_power,
        power_band_w=args.power_band,
        min_tokens=20,
    )

    trainer = Z25GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        sensor_hub=sensor_hub,
        reward_fn=reward_fn,
        lr=args.lr,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
    )

    # Load dataset
    print("\n[5/5] Loading dataset...")
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

        # Shuffle data
        np.random.shuffle(train_data)

        for i, sample in enumerate(train_data[:args.max_prompts]):
            # Support multiple dataset formats
            prompt = sample.get("input", sample.get("prompt", sample.get("instruction", "")))
            reference = sample.get("output", sample.get("response", ""))

            if not prompt or not reference:
                print(f"[{i+1}] Skipping - no prompt/reference", flush=True)
                continue

            try:
                # Train step
                metrics = trainer.train_step(prompt, reference)
                global_step += 1

                # Log to W&B
                wandb.log(metrics, step=global_step)

                # Console log
                if (i + 1) % args.log_every == 0:
                    print(
                        f"[{i+1}/{args.max_prompts}] "
                        f"reward={metrics['reward_mean']:.3f} "
                        f"gate={metrics['gate_mean']:.3f} "
                        f"skip={metrics['skip_rate']:.2%} "
                        f"power={metrics['power_mean_w']:.1f}W",
                        flush=True
                    )
            except Exception as e:
                print(f"[{i+1}] Error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue

            # Validation
            if (i + 1) % args.val_every == 0:
                print(f"\n[Validation at step {i+1}]")

                # Quick validation on subset
                val_rewards = []
                val_powers = []
                val_gates = []
                model.eval()

                for v in val_data[:10]:
                    v_prompt = v.get("input", v.get("prompt", v.get("instruction", "")))
                    v_ref = v.get("output", v.get("response", ""))

                    samples = generate_stepwise_batch(
                        model, tokenizer, v_prompt,
                        num_samples=1, max_tokens=64
                    )

                    text = tokenizer.decode(samples[0].tokens, skip_special_tokens=True)
                    r, _ = reward_fn.compute(samples[0], v_ref, text, tokenizer)
                    val_rewards.append(r)
                    if samples[0].power_trajectory:
                        val_powers.append(np.mean(samples[0].power_trajectory))
                    if samples[0].gates:
                        val_gates.append(np.mean(samples[0].gates))

                val_r = np.mean(val_rewards)
                val_p = np.mean(val_powers) if val_powers else 0
                val_g = np.mean(val_gates) if val_gates else 0
                print(f"  val_reward={val_r:.3f} val_power={val_p:.1f}W val_gate={val_g:.3f}")

                wandb.log({
                    "val/reward": val_r,
                    "val/power_w": val_p,
                    "val/gate_mean": val_g,
                }, step=global_step)

                # Ablation test: Compare power trajectories across conditions
                print("\n[Ablation Test - Hardware Trajectories]")
                ablation_prompt = val_data[0].get("prompt", val_data[0].get("instruction", ""))
                ablation_ref = val_data[0].get("response", val_data[0].get("output", ""))

                # Full mode (normal)
                full_samples = generate_stepwise_batch(
                    model, tokenizer, ablation_prompt, num_samples=2, max_tokens=64
                )
                full_power = np.mean([np.mean(s.power_trajectory) for s in full_samples if s.power_trajectory])
                full_reward = np.mean([reward_fn.compute(s, ablation_ref, tokenizer.decode(s.tokens, skip_special_tokens=True), tokenizer)[0] for s in full_samples])

                # Shuffle sensors (randomize sensor readings)
                original_read = sensor_hub.read_tensor
                def shuffled_read():
                    t = original_read()
                    return t[torch.randperm(len(t))]
                sensor_hub.read_tensor = shuffled_read

                shuffle_samples = generate_stepwise_batch(
                    model, tokenizer, ablation_prompt, num_samples=2, max_tokens=64
                )
                shuffle_power = np.mean([np.mean(s.power_trajectory) for s in shuffle_samples if s.power_trajectory])
                shuffle_reward = np.mean([reward_fn.compute(s, ablation_ref, tokenizer.decode(s.tokens, skip_special_tokens=True), tokenizer)[0] for s in shuffle_samples])

                # Restore
                sensor_hub.read_tensor = original_read

                # Frozen (use cached tensor, no updates)
                frozen_tensor = sensor_hub.read_tensor().clone()
                def frozen_read():
                    return frozen_tensor
                sensor_hub.read_tensor = frozen_read

                frozen_samples = generate_stepwise_batch(
                    model, tokenizer, ablation_prompt, num_samples=2, max_tokens=64
                )
                frozen_power = np.mean([np.mean(s.power_trajectory) for s in frozen_samples if s.power_trajectory])
                frozen_reward = np.mean([reward_fn.compute(s, ablation_ref, tokenizer.decode(s.tokens, skip_special_tokens=True), tokenizer)[0] for s in frozen_samples])

                # Restore
                sensor_hub.read_tensor = original_read

                # Compute causal score
                causal_reward = (full_reward - shuffle_reward) / (full_reward + 1e-8)
                power_delta = full_power - shuffle_power

                print(f"  full:    reward={full_reward:.3f} power={full_power:.1f}W")
                print(f"  shuffle: reward={shuffle_reward:.3f} power={shuffle_power:.1f}W")
                print(f"  frozen:  reward={frozen_reward:.3f} power={frozen_power:.1f}W")
                print(f"  causal_score={causal_reward:.3f} power_delta={power_delta:.1f}W")

                wandb.log({
                    "ablation/full_reward": full_reward,
                    "ablation/shuffle_reward": shuffle_reward,
                    "ablation/frozen_reward": frozen_reward,
                    "ablation/causal_score": causal_reward,
                    "ablation/full_power_w": full_power,
                    "ablation/shuffle_power_w": shuffle_power,
                    "ablation/frozen_power_w": frozen_power,
                    "ablation/power_delta_w": power_delta,
                }, step=global_step)

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
    sensor_hub.stop()
    wandb.finish()

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
