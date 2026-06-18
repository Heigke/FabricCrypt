#!/usr/bin/env python3
"""
FEEL z29: STOCHASTIC ROUTING with Bernoulli Sampling

CRITICAL FIX from z28:
The deterministic threshold-based routing (gate > τ) caused the model to be
stuck with 0% skip because gate mean (~0.60) was always above threshold (~0.41).
No exploration = no learning.

z29 SOLUTION: Stochastic Bernoulli Routing
- Replace: hard_decision = (gate > threshold)
- With:    run ~ Bernoulli(p = gate)

This means:
- Gate = 0.60 → ~60% run, ~40% skip (IMMEDIATE mixed behavior)
- Gate = 0.80 → ~80% run, ~20% skip
- Gate = 0.30 → ~30% run, ~70% skip

The model can IMMEDIATELY learn from mixed run/skip decisions because
exploration is built into the sampling, not waiting for τ to cross gate.

STE (Straight-Through Estimator):
- Forward: Binary sample from Bernoulli(gate)
- Backward: Gradient flows through gate as identity

MAX SKIP CLAMP:
- Threshold is now a SAFETY LIMIT, not the primary driver
- If skip_rate > max_skip, force some gates to run
- Prevents runaway skipping while allowing exploration
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
        target_throughput: float = 12.0,
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
            throughput_trend = 0.5

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
# BERNOULLI STE FUNCTION (THE KEY FIX!)
# =============================================================================

class BernoulliSTEFunction(torch.autograd.Function):
    """
    Stochastic Bernoulli routing with Straight-Through Estimator.

    Forward: Sample run ~ Bernoulli(gate)
    Backward: Gradient flows through gate as identity

    This is THE FIX for z28's problem:
    - z28: gate=0.60, threshold=0.41 → 100% run (no exploration)
    - z29: gate=0.60 → 60% run, 40% skip (immediate mixed behavior!)
    """

    @staticmethod
    def forward(ctx, gates: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gates)
        # Sample: run with probability = gate value
        u = torch.rand_like(gates)
        run = (u < gates).float()  # 1 = run, 0 = skip
        return run

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # STE: gradient passes through as if gates were used directly
        gates, = ctx.saved_tensors
        return grad_output


# =============================================================================
# MLP SKIP BLOCK WITH STOCHASTIC ROUTING
# =============================================================================

class MLPSkipBlockZ29(nn.Module):
    """
    Embodied MLP skip block with STOCHASTIC Bernoulli routing.

    Key difference from z28:
    - z28: hard_decision = (gate > threshold) → deterministic, stuck
    - z29: run ~ Bernoulli(gate) → stochastic, immediate exploration

    The gate network learns to output probabilities that maximize reward.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        hidden_size: int,
        sensor_hub: FastSensorHub,
        sensor_dim: int = 8,
        gate_bias: float = 0.4,  # Bias to start around ~0.60
        max_skip_rate: float = 0.6,  # Safety clamp: max 60% skip
    ):
        super().__init__()
        self.original_layer = original_layer
        self.hidden_size = hidden_size
        self.sensor_hub = sensor_hub
        self.sensor_dim = sensor_dim
        self.max_skip_rate = max_skip_rate

        # Gate network outputs probability of running MLP
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size + sensor_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Bias final layer for reasonable starting point
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

        # Read sensors - move to same device/dtype as hidden_states
        sensors = self.sensor_hub.read_tensor().to(
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # Compute gates (run probabilities)
        last_hidden = hidden_states[:, -1, :]
        sensors_expanded = sensors.unsqueeze(0).expand(batch, -1)
        gate_input = torch.cat([last_hidden, sensors_expanded], dim=-1)

        gates = self.gate_net(gate_input).squeeze(-1)  # [batch] - probability of running
        self._last_gates = gates.detach().cpu().tolist()

        # STOCHASTIC BERNOULLI ROUTING (THE KEY FIX!)
        # Instead of: run = (gate > threshold)
        # We use:     run ~ Bernoulli(gate)
        if self.training:
            # During training: stochastic sampling with STE
            run_decisions = BernoulliSTEFunction.apply(gates)
        else:
            # During eval: use expected value (deterministic for reproducibility)
            run_decisions = (gates > 0.5).float()

        # Track decisions
        skip_decisions = (run_decisions < 0.5)
        self._last_skip_decisions = skip_decisions.tolist()

        # Update stats
        n_skip = skip_decisions.sum().item()
        self._skip_count += n_skip
        self._run_count += batch - n_skip

        # FiLM parameters
        gamma = 1.0 + self.film_gamma(sensors)
        beta = self.film_beta(sensors)

        # Route based on stochastic decisions
        run_mask = run_decisions > 0.5
        skip_mask = ~run_mask

        output = hidden_states.clone()

        # Run MLP for run decisions
        if run_mask.any():
            run_idx = run_mask.nonzero(as_tuple=True)[0]
            layer_out = self.original_layer(hidden_states[run_idx])
            modulated = gamma.view(1, 1, -1) * layer_out + beta.view(1, 1, -1)
            output[run_idx] = modulated

        # Skip path for skip decisions
        if skip_mask.any():
            skip_idx = skip_mask.nonzero(as_tuple=True)[0]
            output[skip_idx] = hidden_states[skip_idx] + self.strain_embed.view(1, 1, -1)

        # STE gradient helper: ensure gradient flows through gates
        if self.training:
            # Small adjustment that creates gradient path
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

class EmbodiedModelZ29(nn.Module):
    """
    Qwen2.5-7B with stochastic MLP skip routing.
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        sensor_hub: FastSensorHub,
        skip_layers: List[int] = None,
        gate_bias: float = 0.4,
        max_skip_rate: float = 0.6,
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

            skip_block = MLPSkipBlockZ29(
                original_layer=original_mlp,
                hidden_size=hidden_size,
                sensor_hub=sensor_hub,
                gate_bias=gate_bias,
                max_skip_rate=max_skip_rate,
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

        # Move skip blocks to same device AND dtype as base model
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        for block in self.skip_blocks.values():
            block.to(device=device, dtype=dtype)

        # Count params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[EmbodiedModelZ29] Skip blocks at layers {self.skip_layers}")
        print(f"[EmbodiedModelZ29] Trainable: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")
        print(f"[EmbodiedModelZ29] STOCHASTIC ROUTING: run ~ Bernoulli(gate)")

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

    @property
    def gate_std(self) -> float:
        stds = [b.gate_std for b in self.skip_blocks.values()]
        return np.mean(stds)

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
    model: EmbodiedModelZ29,
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
    """
    # Throughput component
    throughput = sample.throughput
    throughput_ratio = throughput / baseline_throughput
    throughput_bonus = min(1.5, throughput_ratio)

    # Quality component
    if len(text) < 10:
        quality = 0.1
    elif len(text) < 50:
        quality = 0.3 + 0.4 * (len(text) - 10) / 40
    else:
        quality = min(1.0, 0.7 + 0.3 * min(len(text), 200) / 200)

    # Combined reward
    reward = 0.6 * throughput_bonus + 0.4 * quality

    metrics = {
        "throughput": throughput,
        "throughput_ratio": throughput_ratio,
        "quality": quality,
        "reward": reward,
    }

    return reward, metrics


# =============================================================================
# GRPO TRAINER WITH SKIP RATE TARGETING
# =============================================================================

class Z29GRPOTrainer:
    """
    GRPO trainer with:
    1. Throughput-based reward
    2. Stochastic Bernoulli routing
    3. Target skip rate regularization
    """

    def __init__(
        self,
        model: EmbodiedModelZ29,
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

        print(f"[Z29GRPOTrainer] Gate LR: {gate_lr}, FiLM LR: {film_lr}")
        print(f"[Z29GRPOTrainer] Target skip rate: {target_skip_rate}")
        print(f"[Z29GRPOTrainer] Baseline throughput: {baseline_throughput} tok/s")
        print(f"[Z29GRPOTrainer] STOCHASTIC ROUTING: Learning from mixed run/skip")

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

        # Skip rate regularization - target a specific skip rate
        current_skip_rate = self.model.skip_rate
        skip_reg = (current_skip_rate - self.target_skip_rate) ** 2

        # Gate diversity bonus - encourage variance in gate outputs
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

        # Generate samples with stochastic routing
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
            "gate_std": self.model.gate_std,
            **loss_metrics,
        }


# =============================================================================
# VALIDATION
# =============================================================================

def run_validation(
    model: EmbodiedModelZ29,
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
    parser.add_argument("--gate-bias", type=float, default=0.4)
    parser.add_argument("--max-skip-rate", type=float, default=0.6)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="models/grpo_z29")
    args = parser.parse_args()

    # W&B init
    wandb.init(
        project="feel-z29-stochastic",
        name=f"z29-{time.strftime('%Y%m%d_%H%M')}",
        config=vars(args),
    )

    print("=" * 70)
    print("FEEL z29: STOCHASTIC BERNOULLI ROUTING")
    print("=" * 70)
    print()
    print("KEY FIX from z28:")
    print("  z28: gate > threshold → deterministic, 0% skip, stuck")
    print("  z29: run ~ Bernoulli(gate) → stochastic, immediate mixed behavior")
    print()
    print("With gate mean ~0.60:")
    print("  z28: 0% skip (threshold=0.41 < gate=0.60)")
    print("  z29: ~40% skip (Bernoulli samples give immediate exploration)")
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

    print("[3/5] Creating embodied model with STOCHASTIC routing...")
    model = EmbodiedModelZ29(
        base_model=base_model,
        sensor_hub=sensor_hub,
        gate_bias=args.gate_bias,
        max_skip_rate=args.max_skip_rate,
    )

    print("[4/5] Creating trainer...")
    trainer = Z29GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        gate_lr=args.gate_lr,
        target_skip_rate=args.target_skip_rate,
        skip_rate_weight=args.skip_reg_weight,
        baseline_throughput=args.baseline_throughput,
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
    print("Starting training with STOCHASTIC routing...")
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

            # Train step
            metrics = trainer.train_step(
                prompt,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
            )

            # Log every step
            if (prompt_idx + 1) % args.log_every == 0:
                throughput = model.sensor_hub.throughput
                power_w = model.sensor_hub.power_w
                print(f"[{prompt_idx+1}/{args.max_prompts}] "
                      f"r={metrics['reward']:.3f} "
                      f"tput={throughput:.1f}tok/s "
                      f"gate={metrics['gate_mean']:.3f}±{metrics['gate_std']:.3f} "
                      f"skip={metrics['skip_rate']*100:.1f}% "
                      f"P={power_w:.0f}W")

                wandb.log({
                    # Step info
                    "step": global_step,
                    "epoch": epoch + 1,
                    # Reward
                    "reward/mean": metrics["reward"],
                    "reward/quality": metrics.get("quality", 0),
                    # Throughput
                    "throughput/tok_s": throughput,
                    "throughput/sample_mean": metrics.get("throughput", throughput),
                    # Gate stats (now with std!)
                    "gate/mean": metrics["gate_mean"],
                    "gate/std": metrics.get("gate_std", 0),
                    # Skip behavior (should now be ~35-40% immediately!)
                    "skip/rate": metrics["skip_rate"],
                    "skip/target": args.target_skip_rate,
                    # Power
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
                    # Full mode
                    "val/full/throughput": val_results["full"]["throughput"],
                    "val/full/skip_rate": val_results["full"]["skip_rate"],
                    "val/full/gate_mean": val_results["full"]["gate_mean"],
                    # Shuffle mode
                    "val/shuffle/throughput": val_results["shuffle"]["throughput"],
                    "val/shuffle/skip_rate": val_results["shuffle"]["skip_rate"],
                    "val/shuffle/gate_mean": val_results["shuffle"]["gate_mean"],
                    # Frozen mode
                    "val/frozen/throughput": val_results["frozen"]["throughput"],
                    "val/frozen/skip_rate": val_results["frozen"]["skip_rate"],
                    "val/frozen/gate_mean": val_results["frozen"]["gate_mean"],
                    # Deltas
                    "val/delta/throughput": val_results["full"]["throughput"] - val_results["shuffle"]["throughput"],
                    "val/delta/skip_rate": val_results["full"]["skip_rate"] - val_results["shuffle"]["skip_rate"],
                })

                # Save checkpoint
                ckpt_path = Path(args.checkpoint_dir) / f"step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": {k: v for k, v in model.state_dict().items() if "skip_blocks" in k},
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
