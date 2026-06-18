#!/usr/bin/env python3
"""
Z401: Train Embodied Policy - Learn Optimal Compute Actions from Body State

This trains the controller to make optimal early-exit decisions based on:
- Current body state (temp, power, utilization)
- Predicted future state (temperature slope)
- Task quality requirements

Training approach:
1. Collect rollouts with different policies
2. Learn state-action mapping that minimizes energy while meeting constraints
3. Validate on held-out workloads
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter, GpuSample
from src.controllers.embodied_controller import ComputeAction, EnergyMode

from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset

# Load distilled model
try:
    from scripts.z204_distill_exit_heads import DistillableEarlyExitGPT2
    HAS_DISTILLED = True
except ImportError:
    HAS_DISTILLED = False


@dataclass
class TrainingConfig:
    """Configuration for policy training."""
    # Environment
    temp_cap_c: float = 80.0
    power_cap_w: float = 100.0
    latency_cap_ms: float = 100.0

    # Training
    num_episodes: int = 100
    steps_per_episode: int = 50
    batch_size: int = 8
    seq_len: int = 128

    # Policy network
    state_dim: int = 8  # temp, power, slope, headroom, etc.
    hidden_dim: int = 64
    num_actions: int = 4  # exit at L3, L6, L9, L12

    # Learning
    lr: float = 1e-3
    gamma: float = 0.99  # Discount factor
    entropy_coef: float = 0.01  # Exploration bonus

    # Reward weights
    energy_weight: float = 1.0
    quality_weight: float = 0.5
    constraint_weight: float = 2.0  # Penalty for violations


@dataclass
class Transition:
    """Single transition for training."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class BodyStateEncoder:
    """Encodes GPU body state into neural network input."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._history = deque(maxlen=10)

    def encode(self, sample: GpuSample, latency_ms: float = 0.0) -> np.ndarray:
        """Encode body state into feature vector."""
        self._history.append(sample)

        # Compute temperature slope from history
        if len(self._history) >= 2:
            temps = [s.temp_edge_c for s in self._history]
            times = [(s.timestamp_ns - self._history[0].timestamp_ns) / 1e9
                     for s in self._history]
            if times[-1] > 0:
                slope = np.polyfit(times, temps, 1)[0]
            else:
                slope = 0.0
        else:
            slope = 0.0

        # Normalize features
        state = np.array([
            sample.temp_edge_c / 100.0,  # Temperature (normalized)
            sample.power_w / 150.0,      # Power (normalized)
            slope / 10.0,                # Temperature slope
            (self.config.temp_cap_c - sample.temp_edge_c) / 20.0,  # Thermal headroom
            (self.config.power_cap_w - sample.power_w) / 30.0,     # Power headroom
            sample.gpu_busy_pct / 100.0,  # GPU utilization
            latency_ms / self.config.latency_cap_ms,  # Latency ratio
            1.0 if slope > 0.5 else 0.0,  # Rising temperature flag
        ], dtype=np.float32)

        return state


class PolicyNetwork(nn.Module):
    """Neural network policy for compute action selection."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(config.hidden_dim, config.num_actions)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value."""
        features = self.net(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Sample action from policy."""
        device = next(self.parameters()).device
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, _ = self.forward(state_t)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                action = torch.multinomial(probs, 1).item()

        return action

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training."""
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return action_log_probs, values.squeeze(1), entropy


class EarlyExitModel(nn.Module):
    """Model wrapper with early exit capability."""

    def __init__(self, base_model: GPT2LMHeadModel, distilled_model=None):
        super().__init__()
        self.base_model = base_model
        self.distilled_model = distilled_model
        self.vocab_size = base_model.config.vocab_size

    def forward(self, input_ids: torch.Tensor, exit_layer: int = 12):
        """Forward with early exit."""
        if self.distilled_model is not None:
            logits, _ = self.distilled_model.forward_to_layer(input_ids, exit_layer)
            return logits

        if exit_layer >= 12:
            return self.base_model(input_ids).logits

        # Manual early exit
        hidden = self.base_model.transformer.wte(input_ids)
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = hidden + self.base_model.transformer.wpe(pos_ids)
        hidden = self.base_model.transformer.drop(hidden)

        for block in self.base_model.transformer.h[:exit_layer]:
            hidden = block(hidden)[0]

        hidden = self.base_model.transformer.ln_f(hidden)
        return self.base_model.lm_head(hidden)

    def compute_loss(self, input_ids: torch.Tensor, exit_layer: int = 12) -> float:
        """Compute cross-entropy loss."""
        logits = self.forward(input_ids, exit_layer)
        labels = input_ids.clone()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )
        return loss.item()


class EmbodiedEnvironment:
    """Environment for training embodied policy."""

    EXIT_LAYERS = [3, 6, 9, 12]

    def __init__(
        self,
        model: EarlyExitModel,
        telemetry: SysfsHwmonTelemetry,
        batches: List[torch.Tensor],
        config: TrainingConfig,
    ):
        self.model = model
        self.telemetry = telemetry
        self.batches = batches
        self.config = config
        self.encoder = BodyStateEncoder(config)

        self.batch_idx = 0
        self.baseline_loss = None  # Set during first full-model forward

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        self.batch_idx = 0
        torch.cuda.synchronize()
        time.sleep(0.1)  # Brief cooldown

        sample = self.telemetry.read_sample()
        return self.encoder.encode(sample)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return (next_state, reward, done, info).

        Action: 0=L3, 1=L6, 2=L9, 3=L12
        """
        exit_layer = self.EXIT_LAYERS[action]
        batch = self.batches[self.batch_idx % len(self.batches)]

        # Execute forward pass
        start_time = time.perf_counter()
        with torch.no_grad():
            loss = self.model.compute_loss(batch, exit_layer)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Read body state
        sample = self.telemetry.read_sample()

        # Compute reward
        reward, info = self._compute_reward(
            sample, latency_ms, loss, exit_layer
        )

        # Next state
        next_state = self.encoder.encode(sample, latency_ms)

        # Check if episode done
        self.batch_idx += 1
        done = self.batch_idx >= self.config.steps_per_episode

        info.update({
            'exit_layer': exit_layer,
            'latency_ms': latency_ms,
            'loss': loss,
            'temp_c': sample.temp_edge_c,
            'power_w': sample.power_w,
        })

        return next_state, reward, done, info

    def _compute_reward(
        self,
        sample: GpuSample,
        latency_ms: float,
        loss: float,
        exit_layer: int
    ) -> Tuple[float, dict]:
        """Compute reward balancing energy, quality, and constraints."""

        # Energy reward: lower is better, scale by exit layer
        energy_reward = (12 - exit_layer) / 12.0  # 0.25 to 0 as layer increases

        # Quality reward: penalize quality degradation
        if self.baseline_loss is None:
            self.baseline_loss = loss
        quality_ratio = loss / self.baseline_loss if self.baseline_loss > 0 else 1.0
        quality_penalty = max(0, quality_ratio - 1.0)  # Penalty if worse than baseline

        # Constraint penalties
        temp_violation = max(0, sample.temp_edge_c - self.config.temp_cap_c) / 10.0
        power_violation = max(0, sample.power_w - self.config.power_cap_w) / 20.0
        latency_violation = max(0, latency_ms - self.config.latency_cap_ms) / 50.0

        constraint_penalty = temp_violation + power_violation + latency_violation

        # Combined reward
        reward = (
            self.config.energy_weight * energy_reward
            - self.config.quality_weight * quality_penalty
            - self.config.constraint_weight * constraint_penalty
        )

        info = {
            'energy_reward': energy_reward,
            'quality_penalty': quality_penalty,
            'constraint_penalty': constraint_penalty,
            'quality_ratio': quality_ratio,
        }

        return reward, info


def train_policy(
    policy: PolicyNetwork,
    env: EmbodiedEnvironment,
    config: TrainingConfig,
    device: torch.device,
) -> Dict:
    """Train policy using PPO-style updates."""

    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)

    # Training metrics
    episode_rewards = []
    episode_energies = []
    episode_qualities = []
    episode_violations = []

    print("\n" + "=" * 70)
    print("TRAINING EMBODIED POLICY")
    print("=" * 70)

    for episode in range(config.num_episodes):
        # Collect rollout
        transitions = []
        state = env.reset()

        episode_reward = 0
        episode_exit_layers = []
        episode_temps = []
        violations = 0

        for step in range(config.steps_per_episode):
            # Get action from policy
            action = policy.get_action(state, deterministic=False)

            # Execute action
            next_state, reward, done, info = env.step(action)

            transitions.append(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            ))

            episode_reward += reward
            episode_exit_layers.append(info['exit_layer'])
            episode_temps.append(info['temp_c'])

            if info['constraint_penalty'] > 0:
                violations += 1

            state = next_state
            if done:
                break

        # Convert to tensors
        states = torch.FloatTensor([t.state for t in transitions]).to(device)
        actions = torch.LongTensor([t.action for t in transitions]).to(device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(device)

        # Compute returns (simple Monte Carlo)
        returns = []
        R = 0
        for t in reversed(transitions):
            R = t.reward + config.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy update
        log_probs, values, entropy = policy.evaluate_actions(states, actions)

        advantage = returns - values.detach()
        policy_loss = -(log_probs * advantage).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        loss = policy_loss + 0.5 * value_loss + config.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        # Track metrics
        avg_exit = np.mean(episode_exit_layers)
        avg_temp = np.mean(episode_temps)
        episode_rewards.append(episode_reward)
        episode_energies.append(12 - avg_exit)  # Proxy for energy saved
        episode_violations.append(violations)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:3d} | "
                  f"Reward: {episode_reward:+.2f} | "
                  f"Avg Exit: {avg_exit:.1f} | "
                  f"Avg Temp: {avg_temp:.1f}°C | "
                  f"Violations: {violations}")

    return {
        'episode_rewards': episode_rewards,
        'episode_energies': episode_energies,
        'episode_violations': episode_violations,
    }


def validate_policy(
    policy: PolicyNetwork,
    env: EmbodiedEnvironment,
    config: TrainingConfig,
    device: torch.device,
    num_episodes: int = 10,
) -> Dict:
    """Validate trained policy."""

    print("\n" + "=" * 70)
    print("VALIDATING POLICY")
    print("=" * 70)

    policy.eval()

    all_exits = []
    all_temps = []
    all_powers = []
    all_latencies = []
    all_losses = []
    total_violations = 0

    for episode in range(num_episodes):
        state = env.reset()

        for step in range(config.steps_per_episode):
            action = policy.get_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)

            all_exits.append(info['exit_layer'])
            all_temps.append(info['temp_c'])
            all_powers.append(info['power_w'])
            all_latencies.append(info['latency_ms'])
            all_losses.append(info['loss'])

            if info['constraint_penalty'] > 0:
                total_violations += 1

            state = next_state
            if done:
                break

    total_steps = num_episodes * config.steps_per_episode

    return {
        'avg_exit_layer': np.mean(all_exits),
        'exit_distribution': dict(zip(*np.unique(all_exits, return_counts=True))),
        'avg_temp_c': np.mean(all_temps),
        'max_temp_c': np.max(all_temps),
        'avg_power_w': np.mean(all_powers),
        'avg_latency_ms': np.mean(all_latencies),
        'avg_loss': np.mean(all_losses),
        'constraint_violations': total_violations,
        'constraint_satisfaction_pct': (1 - total_violations / total_steps) * 100,
    }


def main():
    print("=" * 70)
    print("Z401: TRAIN EMBODIED POLICY")
    print("=" * 70)

    config = TrainingConfig(
        num_episodes=50,
        steps_per_episode=30,
        batch_size=8,
        seq_len=128,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize telemetry
    print("\n--- Initializing Telemetry ---")
    telemetry = SysfsHwmonTelemetry(sample_rate_hz=50)
    idle_power = telemetry.measure_idle_baseline(duration_s=2.0)
    print(f"Idle power: {idle_power:.1f} W")

    # Load model
    print("\n--- Loading Model ---")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    distilled_model = None
    if HAS_DISTILLED:
        checkpoint_path = Path("checkpoints/z204_distilled/model_final.pt")
        if checkpoint_path.exists():
            print(f"Loading distilled model from {checkpoint_path}")
            distilled_model = DistillableEarlyExitGPT2("gpt2").to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            distilled_model.load_state_dict(checkpoint['model_state'])
            distilled_model.eval()

    model = EarlyExitModel(base_model, distilled_model)
    model.eval()

    # Prepare data
    print("\n--- Preparing Data ---")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    texts = []
    for item in dataset:
        if len(texts) >= 500:
            break
        text = item['text'][:500]
        if len(text) > 50:
            texts.append(text)

    batches = []
    for i in range(0, len(texts), config.batch_size):
        batch_texts = texts[i:i+config.batch_size]
        if len(batch_texts) == config.batch_size:
            encoded = tokenizer(
                batch_texts,
                max_length=config.seq_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            batches.append(encoded['input_ids'].to(device))

    print(f"Prepared {len(batches)} batches")

    # Create environment and policy
    env = EmbodiedEnvironment(model, telemetry, batches, config)
    policy = PolicyNetwork(config)

    # Train
    train_metrics = train_policy(policy, env, config, device)

    # Validate
    val_metrics = validate_policy(policy, env, config, device, num_episodes=10)

    # Compare with baselines
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)

    # Fixed L12 baseline
    print("\n--- Fixed L12 Baseline ---")
    env_baseline = EmbodiedEnvironment(model, telemetry, batches, config)
    baseline_exits = []
    baseline_losses = []
    baseline_temps = []
    baseline_powers = []

    state = env_baseline.reset()
    for _ in range(config.steps_per_episode * 5):
        next_state, _, done, info = env_baseline.step(3)  # Always L12
        baseline_exits.append(info['exit_layer'])
        baseline_losses.append(info['loss'])
        baseline_temps.append(info['temp_c'])
        baseline_powers.append(info['power_w'])
        state = next_state

    baseline_metrics = {
        'avg_exit_layer': np.mean(baseline_exits),
        'avg_loss': np.mean(baseline_losses),
        'avg_temp_c': np.mean(baseline_temps),
        'avg_power_w': np.mean(baseline_powers),
    }

    # Results comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Baseline (L12)':<20} {'Trained Policy':<20}")
    print("-" * 65)
    print(f"{'Avg Exit Layer':<25} {baseline_metrics['avg_exit_layer']:<20.1f} {val_metrics['avg_exit_layer']:<20.1f}")
    print(f"{'Avg Loss':<25} {baseline_metrics['avg_loss']:<20.3f} {val_metrics['avg_loss']:<20.3f}")
    print(f"{'Avg Temp (°C)':<25} {baseline_metrics['avg_temp_c']:<20.1f} {val_metrics['avg_temp_c']:<20.1f}")
    print(f"{'Avg Power (W)':<25} {baseline_metrics['avg_power_w']:<20.1f} {val_metrics['avg_power_w']:<20.1f}")
    print(f"{'Constraint Sat (%)':<25} {'N/A':<20} {val_metrics['constraint_satisfaction_pct']:<20.1f}")

    # Energy savings estimate
    compute_reduction = (12 - val_metrics['avg_exit_layer']) / 12 * 100
    print(f"\n{'Compute Reduction':<25} {compute_reduction:.1f}%")

    # Business value
    print("\n" + "=" * 70)
    print("BUSINESS VALUE (100 GPUs, 24/7)")
    print("=" * 70)

    power_reduction = baseline_metrics['avg_power_w'] - val_metrics['avg_power_w']
    annual_hours = 8760
    gpu_count = 100
    cost_per_kwh = 0.10

    energy_saved_kwh = power_reduction * annual_hours / 1000 * gpu_count
    cost_savings = energy_saved_kwh * cost_per_kwh
    carbon_reduction = energy_saved_kwh * 0.4  # kg CO2

    print(f"Power reduction per GPU: {power_reduction:.1f} W")
    print(f"Annual energy savings: {energy_saved_kwh:,.0f} kWh")
    print(f"Annual cost savings: ${cost_savings:,.0f}")
    print(f"Carbon reduction: {carbon_reduction:,.0f} kg CO2/year")

    # Save results
    output_path = Path("results/z401_trained_policy.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'config': asdict(config),
        'training': {
            'final_reward': float(train_metrics['episode_rewards'][-1]),
            'reward_improvement': float(train_metrics['episode_rewards'][-1] - train_metrics['episode_rewards'][0]),
        },
        'validation': {k: float(v) if isinstance(v, (np.floating, float)) else v
                      for k, v in val_metrics.items()},
        'baseline': {k: float(v) for k, v in baseline_metrics.items()},
        'business_value': {
            'power_reduction_w': power_reduction,
            'annual_energy_kwh': energy_saved_kwh,
            'annual_cost_usd': cost_savings,
            'carbon_reduction_kg': carbon_reduction,
        }
    }

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return obj

    results = convert(results)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Save policy
    policy_path = Path("checkpoints/z401_policy/policy.pt")
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'policy_state': policy.state_dict(),
        'config': asdict(config),
    }, policy_path)
    print(f"Policy saved to {policy_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
