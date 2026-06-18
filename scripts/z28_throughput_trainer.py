#!/usr/bin/env python3
"""
FEEL z28: Throughput-Based Embodied Training

CRITICAL FIXES from z27:
1. REWARD: Throughput (tok/s) instead of power IAE
   - MLP skip INCREASES power but IMPROVES throughput
   - Throughput aligns with what skip actually controls

2. GATE INIT: Bias to ~0.58 (above start threshold)
   - z27: gates ~0.527, threshold 0.55 → 87% skip immediately
   - z28: gates ~0.58, threshold 0.40 → mixed run/skip from start

3. THRESHOLD: Start at 0.40, increase to 0.55
   - Flipped curriculum: start easy (low bar to run), get harder
   - Ensures mixed behavior from step 1

4. REGULARIZATION: Target skip rate, not symmetric
   - z27: -|gate - 0.5| pushed gates toward 0
   - z28: (skip_rate - target)² keeps balanced

The model learns: "Skip when I can maintain throughput, run when needed"
"""

import os
import sys
import time
import json
import argparse
import threading
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
    """
    High-frequency sensor hub for embodied state.
    Provides 8-dim sensor vector for gate decisions.
    """

    def __init__(
        self,
        device: str = "cuda",
        sampling_hz: float = 100.0,
        target_throughput: float = 12.0,  # baseline tok/s
    ):
        self.device = device
        self.sampling_hz = sampling_hz
        self.target_throughput = target_throughput

        # Find power sensor
        self.power_path = self._find_power_sensor()
        print(f"[FastSensorHub] Power sensor: {self.power_path}")
        print(f"[FastSensorHub] Sampling at {sampling_hz}Hz, target throughput={target_throughput} tok/s")

        # State
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Sensor readings
        self.power_w = 75.0
        self.temp_c = 50.0
        self.util_pct = 50.0
        self.mem_used_pct = 50.0
        self.throughput = target_throughput
        self.throughput_history = [target_throughput] * 10

        # Cached tensor
        self._cached_tensor = None
        self._tensor_device = device

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

    def update_throughput(self, tokens: int, time_sec: float):
        """Update throughput measurement."""
        if time_sec > 0:
            current = tokens / time_sec
            self.throughput_history.append(current)
            if len(self.throughput_history) > 10:
                self.throughput_history.pop(0)
            self.throughput = np.mean(self.throughput_history)

    def read_tensor(self) -> torch.Tensor:
        """
        Get 8-dim sensor state tensor.
        All values normalized to [0, 1] range.
        """
        with self._lock:
            # Normalize sensors
            power_norm = min(1.0, self.power_w / 150.0)
            temp_norm = min(1.0, (self.temp_c - 30) / 70.0)
            util_norm = self.util_pct / 100.0
            mem_norm = self.mem_used_pct / 100.0

            # Throughput-based sensors
            throughput_ratio = self.throughput / self.target_throughput
            throughput_norm = min(1.0, throughput_ratio)
            throughput_trend = 0.5  # Could track derivative

            # Error from target
            throughput_error = abs(throughput_ratio - 1.0)

            state = np.array([
                power_norm,           # 0: Power level
                temp_norm,            # 1: Temperature
                util_norm,            # 2: GPU utilization
                mem_norm,             # 3: Memory usage
                throughput_norm,      # 4: Current throughput ratio
                throughput_trend,     # 5: Throughput trend
                throughput_error,     # 6: Error from target
                0.5,                  # 7: Reserved
            ], dtype=np.float32)

        tensor = torch.from_numpy(state).to(self._tensor_device)
        return tensor


# =============================================================================
# STE SKIP FUNCTION
# =============================================================================

class STESkipFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for hard skip decisions.
    Forward: Hard threshold
    Backward: Gradient passes through as identity
    """

    @staticmethod
    def forward(ctx, gates: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(gates)
        # Hard decision: gate >= threshold → 1 (run), else 0 (skip)
        return (gates >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output, None


# =============================================================================
# MLP SKIP BLOCK WITH PROPER INITIALIZATION
# =============================================================================

class MLPSkipBlockZ28(nn.Module):
    """
    Embodied MLP skip block with FIXED initialization.

    Key changes from z27:
    1. Gate bias initialized to give ~0.58 mean output
    2. Skip decision based on throughput, not power
    """

    def __init__(
        self,
        original_layer: nn.Module,
        hidden_size: int,
        sensor_hub: FastSensorHub,
        sensor_dim: int = 8,
        skip_threshold: float = 0.40,  # Lower start threshold!
        gate_bias: float = 0.4,  # Bias to shift sigmoid output up
    ):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_size = hidden_size
        self.sensor_hub = sensor_hub
        self.sensor_dim = sensor_dim
        self.skip_threshold = skip_threshold

        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size + sensor_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # CRITICAL FIX: Bias the final linear layer to output higher values
        # This shifts sigmoid input, making gate output ~0.58 instead of ~0.52
        with torch.no_grad():
            self.gate_net[-2].bias.fill_(gate_bias)

        # FiLM modulation
        self.film_gamma = nn.Linear(sensor_dim, hidden_size)
        self.film_beta = nn.Linear(sensor_dim, hidden_size)

        # Strain embedding for skip path
        self.strain_embed = nn.Parameter(torch.randn(hidden_size) * 0.01)

        # Stats tracking
        self._last_gates: List[float] = []
        self._last_skip_decisions: List[bool] = []
        self._skip_count = 0
        self._run_count = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        # Read sensors - CRITICAL: move to same device as hidden_states
        sensors = self.sensor_hub.read_tensor().to(
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # Compute gates
        last_hidden = hidden_states[:, -1, :]
        sensors_expanded = sensors.unsqueeze(0).expand(batch, -1)
        gate_input = torch.cat([last_hidden, sensors_expanded], dim=-1)

        gates = self.gate_net(gate_input).squeeze(-1)  # [batch]
        self._last_gates = gates.detach().cpu().tolist()

        # STE hard decision
        hard_decisions = STESkipFunction.apply(gates, self.skip_threshold)
        self._last_skip_decisions = (hard_decisions < 0.5).tolist()

        # Update stats
        n_skip = (hard_decisions < 0.5).sum().item()
        self._skip_count += n_skip
        self._run_count += batch - n_skip

        # FiLM parameters
        gamma = 1.0 + self.film_gamma(sensors)
        beta = self.film_beta(sensors)

        # Route based on decisions
        run_mask = hard_decisions > 0.5
        skip_mask = ~run_mask

        output = hidden_states.clone()

        # Run MLP for non-skipped
        if run_mask.any():
            run_idx = run_mask.nonzero(as_tuple=True)[0]
            layer_out = self.original_layer(hidden_states[run_idx])
            modulated = gamma.view(1, 1, -1) * layer_out + beta.view(1, 1, -1)
            output[run_idx] = modulated

        # Skip path
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

    def reset_stats(self):
        self._skip_count = 0
        self._run_count = 0


# =============================================================================
# EMBODIED MODEL
# =============================================================================

class EmbodiedModelZ28(nn.Module):
    """
    Qwen2.5-7B with throughput-aware MLP skip.
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        sensor_hub: FastSensorHub,
        skip_layers: List[int] = None,
        skip_threshold: float = 0.40,
        gate_bias: float = 0.4,
    ):
        super().__init__()
        self.base_model = base_model
        self.sensor_hub = sensor_hub
        self.skip_layers = skip_layers or [7, 11, 15, 19, 23]
        self._current_threshold = skip_threshold

        hidden_size = base_model.config.hidden_size

        # Create skip blocks
        self.skip_blocks = nn.ModuleDict()
        for layer_idx in self.skip_layers:
            layer = base_model.model.layers[layer_idx]
            original_mlp = layer.mlp

            skip_block = MLPSkipBlockZ28(
                original_layer=original_mlp,
                hidden_size=hidden_size,
                sensor_hub=sensor_hub,
                skip_threshold=skip_threshold,
                gate_bias=gate_bias,
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

        # CRITICAL: Move skip blocks to same device AND dtype as base model
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        for block in self.skip_blocks.values():
            block.to(device=device, dtype=dtype)

        # Count params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[EmbodiedModelZ28] Skip blocks at layers {self.skip_layers}")
        print(f"[EmbodiedModelZ28] Trainable: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")

    def set_threshold(self, threshold: float):
        self._current_threshold = threshold
        for block in self.skip_blocks.values():
            block.skip_threshold = threshold

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.base_model(input_ids, **kwargs)

    @property
    def skip_rate(self) -> float:
        rates = [b.skip_rate for b in self.skip_blocks.values()]
        return np.mean(rates)

    @property
    def gate_mean(self) -> float:
        means = [b.gate_mean for b in self.skip_blocks.values()]
        return np.mean(means)

    def reset_stats(self):
        for block in self.skip_blocks.values():
            block.reset_stats()

    def get_gate_tensor(self) -> torch.Tensor:
        """Get all recent gate values as tensor for regularization."""
        all_gates = []
        for block in self.skip_blocks.values():
            all_gates.extend(block._last_gates)
        if all_gates:
            return torch.tensor(all_gates, device=next(self.parameters()).device)
        return torch.tensor([0.5], device=next(self.parameters()).device)


# =============================================================================
# THRESHOLD CURRICULUM (FLIPPED from z27)
# =============================================================================

class ThresholdCurriculumZ28:
    """
    FLIPPED curriculum: Start LOW, increase to HIGH.

    z27: 0.55 → 0.35 (everything skipped initially)
    z28: 0.40 → 0.55 (mixed behavior from start, harder to skip later)
    """

    def __init__(
        self,
        start_threshold: float = 0.40,
        end_threshold: float = 0.55,
        warmup_steps: int = 50,
        total_steps: int = 1500,
    ):
        self.start = start_threshold
        self.end = end_threshold
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_threshold(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.start

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        return self.start + progress * (self.end - self.start)


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
        if t > 0:
            return len(self.tokens) / t
        return 0.0


# =============================================================================
# GENERATION WITH THROUGHPUT TRACKING
# =============================================================================

def generate_with_throughput(
    model: EmbodiedModelZ28,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_samples: int = 4,
    max_tokens: int = 64,
    temperature: float = 0.7,
    greedy: bool = False,
) -> List[StepwiseSample]:
    """Generate samples and track throughput."""
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
                logits_scaled = logits / temperature
                probs = F.softmax(logits_scaled, dim=-1)
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

    # Update sensor hub with throughput
    t_total = time.perf_counter() - t_start
    tokens_generated = sum(len(s.tokens) for s in samples)
    model.sensor_hub.update_throughput(tokens_generated, t_total)

    return samples


# =============================================================================
# THROUGHPUT-BASED REWARD
# =============================================================================

def compute_throughput_reward(
    sample: StepwiseSample,
    text: str,
    baseline_throughput: float = 12.0,
    min_quality: float = 0.3,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute reward based on throughput and quality.

    reward = throughput_bonus * quality_factor

    Where:
    - throughput_bonus = throughput / baseline (capped at 1.5)
    - quality_factor = text quality score
    """
    # Throughput component
    throughput = sample.throughput
    throughput_ratio = throughput / baseline_throughput
    throughput_bonus = min(1.5, throughput_ratio)  # Cap at 1.5x baseline

    # Quality component (simple heuristic)
    if len(text) < 10:
        quality = 0.1
    elif len(text) < 50:
        quality = 0.3 + 0.4 * (len(text) - 10) / 40
    else:
        quality = min(1.0, 0.7 + 0.3 * min(len(text), 200) / 200)

    # Combined reward
    # High throughput with good quality = high reward
    # High throughput with bad quality = medium reward
    # Low throughput = low reward regardless of quality
    reward = 0.6 * throughput_bonus + 0.4 * quality

    metrics = {
        "throughput": throughput,
        "throughput_ratio": throughput_ratio,
        "quality": quality,
        "reward": reward,
    }

    return reward, metrics


# =============================================================================
# GRPO TRAINER WITH SKIP RATE REGULARIZATION
# =============================================================================

class Z28GRPOTrainer:
    """
    GRPO trainer with:
    1. Throughput-based reward
    2. Skip rate regularization (not symmetric!)
    """

    def __init__(
        self,
        model: EmbodiedModelZ28,
        tokenizer: AutoTokenizer,
        gate_lr: float = 1e-4,
        film_lr: float = 2e-5,
        strain_lr: float = 1e-5,
        target_skip_rate: float = 0.35,
        skip_rate_weight: float = 0.1,
        baseline_throughput: float = 12.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_skip_rate = target_skip_rate
        self.skip_rate_weight = skip_rate_weight
        self.baseline_throughput = baseline_throughput

        # Separate parameter groups
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

        print(f"[Z28GRPOTrainer] Gate LR: {gate_lr}, FiLM LR: {film_lr}")
        print(f"[Z28GRPOTrainer] Target skip rate: {target_skip_rate}")
        print(f"[Z28GRPOTrainer] Baseline throughput: {baseline_throughput} tok/s")

    def compute_grpo_loss(
        self,
        samples: List[StepwiseSample],
        rewards: List[float],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """GRPO loss with skip rate regularization."""
        device = next(self.model.parameters()).device

        # Group-relative advantages
        rewards_tensor = torch.tensor(rewards, device=device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward

        # Policy gradient loss
        policy_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for sample, advantage in zip(samples, advantages):
            if sample.logprobs:
                logprobs = torch.tensor(sample.logprobs, device=device)
                sample_loss = -(logprobs * advantage).mean()
                policy_loss = policy_loss + sample_loss

        policy_loss = policy_loss / len(samples)

        # Skip rate regularization
        # Target a specific skip rate, not "away from 0.5"
        current_skip_rate = self.model.skip_rate
        skip_reg = (current_skip_rate - self.target_skip_rate) ** 2

        # Gate distribution regularization (encourage spread, not collapse)
        gates = self.model.get_gate_tensor()
        gate_std = gates.std()
        diversity_bonus = -0.1 * gate_std  # Reward diversity

        # Total loss
        total_loss = policy_loss + self.skip_rate_weight * skip_reg + diversity_bonus

        metrics = {
            "policy_loss": policy_loss.item(),
            "skip_rate_reg": skip_reg,
            "gate_std": gate_std.item(),
            "mean_reward": mean_reward.item(),
            "advantage_std": std_reward.item(),
        }

        return total_loss, metrics

    def train_step(
        self,
        prompt: str,
        num_samples: int = 4,
        max_tokens: int = 64,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.model.reset_stats()

        # Generate samples
        samples = generate_with_throughput(
            self.model, self.tokenizer, prompt,
            num_samples=num_samples, max_tokens=max_tokens,
        )

        # Compute rewards
        rewards = []
        throughputs = []
        qualities = []

        for sample in samples:
            text = self.tokenizer.decode(sample.tokens, skip_special_tokens=True)
            reward, metrics = compute_throughput_reward(
                sample, text, baseline_throughput=self.baseline_throughput
            )
            rewards.append(reward)
            throughputs.append(metrics["throughput"])
            qualities.append(metrics["quality"])

        # Compute loss
        loss, loss_metrics = self.compute_grpo_loss(samples, rewards)

        # Optimize
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
            **loss_metrics,
        }


# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(
    model: EmbodiedModelZ28,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_samples: int = 64,
    baseline_throughput: float = 12.0,
) -> Dict[str, Dict[str, float]]:
    """
    Validation with ablation modes.
    """
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

        # Set up sensor mode
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
                reward, metrics = compute_throughput_reward(
                    sample, text, baseline_throughput=baseline_throughput
                )

                mode_throughputs.append(sample.throughput)
                mode_rewards.append(reward)
                mode_gates.append(np.mean(sample.gates) if sample.gates else 0.5)
                mode_skips.append(np.mean(sample.skip_decisions) if sample.skip_decisions else 0.5)

        # Restore
        model.sensor_hub.read_tensor = original_read

        results[mode] = {
            "throughput": np.mean(mode_throughputs),
            "reward": np.mean(mode_rewards),
            "gate_mean": np.mean(mode_gates),
            "skip_rate": np.mean(mode_skips),
        }

    # Causal score: full vs shuffle throughput difference
    results["causal_score"] = results["full"]["throughput"] - results["shuffle"]["throughput"]

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
    parser.add_argument("--start-threshold", type=float, default=0.40)
    parser.add_argument("--end-threshold", type=float, default=0.55)
    parser.add_argument("--gate-bias", type=float, default=0.4)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="models/grpo_z28")
    args = parser.parse_args()

    # W&B init
    wandb.init(
        project="feel-z28-throughput",
        name=f"z28-{time.strftime('%Y%m%d_%H%M')}",
        config=vars(args),
    )

    print("=" * 70)
    print("FEEL z28: THROUGHPUT-BASED EMBODIED TRAINING")
    print("=" * 70)
    print()
    print("KEY FIXES from z27:")
    print("  1. Reward: Throughput instead of power IAE")
    print("  2. Gate bias: +0.4 → ~0.58 mean (above start threshold)")
    print(f"  3. Threshold: {args.start_threshold} → {args.end_threshold} (flipped)")
    print(f"  4. Regularization: Target {args.target_skip_rate*100:.0f}% skip rate")
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

    print("[3/5] Creating embodied model with biased gates...")
    model = EmbodiedModelZ28(
        base_model=base_model,
        sensor_hub=sensor_hub,
        skip_threshold=args.start_threshold,
        gate_bias=args.gate_bias,
    )

    print("[4/5] Creating trainer...")
    trainer = Z28GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        gate_lr=args.gate_lr,
        target_skip_rate=args.target_skip_rate,
        skip_rate_weight=args.skip_reg_weight,
        baseline_throughput=args.baseline_throughput,
    )

    # Curriculum
    total_steps = args.epochs * args.max_prompts
    curriculum = ThresholdCurriculumZ28(
        start_threshold=args.start_threshold,
        end_threshold=args.end_threshold,
        warmup_steps=50,
        total_steps=total_steps,
    )

    # Load dataset
    print("[5/5] Loading dataset...")
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
        train_prompts = [d["prompt"] for d in examples[:1500]]
        val_prompts = [d["prompt"] for d in examples[1500:1600]] if len(examples) > 1500 else train_prompts[:100]
    else:
        train_prompts = [
            "Explain the concept of machine learning.",
            "Write a short story about discovery.",
            "Describe how neural networks work.",
            "What is the meaning of life?",
        ] * 50
        val_prompts = train_prompts[:100]

    print(f"  Train: {len(train_prompts)} samples")
    print(f"  Val: {len(val_prompts)} samples")

    # Checkpoint dir
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    global_step = 0

    for epoch in range(args.epochs):
        print()
        print("=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 70)

        np.random.shuffle(train_prompts)

        for prompt_idx, prompt in enumerate(train_prompts[:args.max_prompts]):
            global_step += 1

            # Update threshold
            new_threshold = curriculum.get_threshold(global_step)
            model.set_threshold(new_threshold)

            # Train step
            metrics = trainer.train_step(
                prompt,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
            )

            # Log
            if (prompt_idx + 1) % args.log_every == 0:
                throughput = model.sensor_hub.throughput
                power_w = model.sensor_hub.power_w
                print(f"[{prompt_idx+1}/{args.max_prompts}] "
                      f"r={metrics['reward']:.3f} "
                      f"tput={throughput:.1f}tok/s "
                      f"gate={metrics['gate_mean']:.3f} "
                      f"skip={metrics['skip_rate']*100:.1f}% "
                      f"τ={new_threshold:.2f} "
                      f"P={power_w:.0f}W")

                wandb.log({
                    # Step info
                    "step": global_step,
                    "epoch": epoch + 1,
                    # Reward
                    "reward/mean": metrics["reward"],
                    "reward/quality": metrics.get("quality", 0),
                    # Throughput (primary metric for z28)
                    "throughput/tok_s": throughput,
                    "throughput/sample_mean": metrics.get("throughput", throughput),
                    # Gate stats
                    "gate/mean": metrics["gate_mean"],
                    "gate/std": metrics.get("gate_std", 0),
                    # Skip behavior
                    "skip/rate": metrics["skip_rate"],
                    "skip/threshold": new_threshold,
                    # Power (still tracked for reference)
                    "power/watts": power_w,
                    # Loss breakdown
                    "loss/total": metrics["loss"],
                    "loss/policy": metrics.get("policy_loss", 0),
                    "loss/skip_reg": metrics.get("skip_rate_reg", 0),
                    # GRPO stats
                    "grpo/advantage_std": metrics.get("advantage_std", 0),
                })

            # Validation
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
                print(f"  CAUSAL SCORE: {val_results['causal_score']:.3f}")

                wandb.log({
                    # Causal validation score
                    "val/causal_score": val_results["causal_score"],
                    # Full (normal) mode
                    "val/full/throughput": val_results["full"]["throughput"],
                    "val/full/skip_rate": val_results["full"]["skip_rate"],
                    "val/full/gate_mean": val_results["full"]["gate_mean"],
                    # Shuffle (randomized sensors) mode
                    "val/shuffle/throughput": val_results["shuffle"]["throughput"],
                    "val/shuffle/skip_rate": val_results["shuffle"]["skip_rate"],
                    "val/shuffle/gate_mean": val_results["shuffle"]["gate_mean"],
                    # Frozen (fixed gates) mode
                    "val/frozen/throughput": val_results["frozen"]["throughput"],
                    "val/frozen/skip_rate": val_results["frozen"]["skip_rate"],
                    "val/frozen/gate_mean": val_results["frozen"]["gate_mean"],
                    # Deltas (full - shuffle) to show sensor influence
                    "val/delta/throughput": val_results["full"]["throughput"] - val_results["shuffle"]["throughput"],
                    "val/delta/skip_rate": val_results["full"]["skip_rate"] - val_results["shuffle"]["skip_rate"],
                })

                # Save checkpoint
                ckpt_path = Path(args.checkpoint_dir) / f"step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": {k: v for k, v in model.state_dict().items() if "skip_blocks" in k},
                    "threshold": new_threshold,
                    "causal_score": val_results["causal_score"],
                }, ckpt_path)
                print(f"  Saved: {ckpt_path}")
                print()

    # Final
    sensor_hub.stop()
    wandb.finish()

    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
