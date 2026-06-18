#!/usr/bin/env python3
"""
Z64: Smart Metabolic Trainer - Throughput/Watt Optimization
============================================================

Key insight from z61-z63: Lower power ≠ lower energy per token.
This experiment uses a SMARTER reward function:

1. MAXIMIZE throughput/watt (tokens/sec per watt)
2. TEMPERATURE CONSTRAINT: Penalize if >80°C
3. WORKLOAD-ADAPTIVE: Learn when to throttle vs run full
4. IDLE DETECTION: Power down during low utilization

Reward function:
  R = throughput_per_watt * quality_factor - temp_penalty - idle_bonus

Where:
  - throughput_per_watt = tokens_per_sec / power_watts
  - quality_factor = 1.0 if PPL < threshold else decay
  - temp_penalty = max(0, temp - 80) * 0.1
  - idle_bonus = reward for correctly detecting idle and throttling

Author: FEEL Research Team
Date: 2026-01-19
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# SMART REWARD CONFIGURATION
# ============================================================
@dataclass
class SmartRewardConfig:
    """Configuration for smart reward function."""
    # Throughput/watt scaling
    throughput_weight: float = 1.0

    # Temperature constraints
    temp_target: float = 75.0  # Ideal temperature
    temp_max: float = 85.0     # Max before heavy penalty
    temp_penalty_scale: float = 0.05

    # Quality constraints
    ppl_threshold: float = 2.0  # PPL above this degrades reward
    quality_weight: float = 0.5

    # Idle detection bonus
    idle_util_threshold: float = 20.0  # Below this = idle
    idle_throttle_bonus: float = 0.5   # Bonus for throttling when idle
    idle_full_power_penalty: float = 0.3  # Penalty for full power when idle

    # Action smoothing (penalize rapid switching)
    action_switch_penalty: float = 0.1


def compute_smart_reward(
    tokens_generated: int,
    time_elapsed: float,
    power_watts: float,
    temperature: float,
    utilization: float,
    action_idx: int,
    prev_action_idx: int,
    loss: float,
    config: SmartRewardConfig,
) -> Tuple[float, Dict]:
    """
    Compute smart reward optimizing throughput/watt with constraints.

    Returns:
        reward: float
        components: dict with breakdown
    """
    components = {}

    # 1. Throughput per watt (main objective)
    tokens_per_sec = tokens_generated / max(time_elapsed, 0.001)
    throughput_per_watt = tokens_per_sec / max(power_watts, 1.0)
    # Normalize to ~1.0 range (typical is 100-300 tok/s/W)
    normalized_tpw = throughput_per_watt / 200.0
    components['throughput_per_watt'] = normalized_tpw * config.throughput_weight

    # 2. Temperature penalty (soft constraint)
    if temperature > config.temp_max:
        temp_penalty = (temperature - config.temp_max) * config.temp_penalty_scale * 2
    elif temperature > config.temp_target:
        temp_penalty = (temperature - config.temp_target) * config.temp_penalty_scale
    else:
        temp_penalty = 0.0
    components['temp_penalty'] = -temp_penalty

    # 3. Quality factor (penalize high loss/PPL)
    ppl = np.exp(min(loss, 10))
    if ppl > config.ppl_threshold:
        quality_factor = config.ppl_threshold / ppl
    else:
        quality_factor = 1.0
    components['quality_factor'] = quality_factor * config.quality_weight

    # 4. Idle detection bonus/penalty
    is_idle = utilization < config.idle_util_threshold
    is_throttled = action_idx in [0, 1]  # ECO or BALANCED

    if is_idle and is_throttled:
        idle_reward = config.idle_throttle_bonus
    elif is_idle and not is_throttled:
        idle_reward = -config.idle_full_power_penalty
    else:
        idle_reward = 0.0
    components['idle_reward'] = idle_reward

    # 5. Action switching penalty
    if action_idx != prev_action_idx:
        switch_penalty = config.action_switch_penalty
    else:
        switch_penalty = 0.0
    components['switch_penalty'] = -switch_penalty

    # Total reward
    reward = sum(components.values())
    components['total'] = reward

    return reward, components


# ============================================================
# WORKLOAD GENERATOR (Variable intensity)
# ============================================================
class VariableWorkloadDataset(Dataset):
    """Dataset with variable workload intensity for testing adaptive behavior."""

    def __init__(self, corpus: str, seq_len: int, total_samples: int = 1000):
        self.data = torch.tensor([ord(c) % 256 for c in corpus], dtype=torch.long)
        self.seq_len = seq_len
        self.total_samples = total_samples

        # Create workload pattern: alternating heavy/light/idle periods
        self.workload_pattern = []
        for i in range(total_samples):
            phase = (i // 50) % 4  # 50 samples per phase, 4 phases
            if phase == 0:
                self.workload_pattern.append('heavy')  # Full batch
            elif phase == 1:
                self.workload_pattern.append('medium')  # Half batch effective
            elif phase == 2:
                self.workload_pattern.append('light')  # Quarter batch
            else:
                self.workload_pattern.append('idle')  # Minimal work

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        workload = self.workload_pattern[idx]

        # All return same size, but 'effective' work varies
        start = torch.randint(0, max(1, len(self.data) - self.seq_len - 1), (1,)).item()
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]

        # Return workload type as metadata
        return x, y, workload


# ============================================================
# SMART METABOLIC TRAINER
# ============================================================
class SmartMetabolicTrainer:
    """Trainer with smart reward function and workload adaptation."""

    def __init__(
        self,
        model,
        telemetry,
        actuator,
        device,
        reward_config: SmartRewardConfig = None,
    ):
        self.model = model
        self.telemetry = telemetry
        self.actuator = actuator
        self.device = device
        self.reward_config = reward_config or SmartRewardConfig()

        self.action_names = ['ECO', 'BALANCED', 'PERFORMANCE', 'MAX']
        self.prev_action = 1  # Start with BALANCED

        # Metrics tracking
        self.episode_rewards = []
        self.episode_components = []
        self.action_history = []
        self.workload_history = []

    def train_episode(
        self,
        dataloader,
        optimizer,
        num_batches: int = 50,
    ) -> Dict:
        """Train one episode with smart rewards."""

        self.model.train()
        episode_reward = 0
        episode_loss = 0
        batch_metrics = []

        for i, (input_ids, targets, workloads) in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            workload = workloads[0]  # First item in batch

            # Read telemetry
            snap_before = self.telemetry.read()
            body_state = self.telemetry.read_body_state()
            telem_tensor = torch.from_numpy(body_state).float().to(self.device)
            telem_tensor = telem_tensor.unsqueeze(0).expand(input_ids.size(0), -1)

            # Forward pass with timing
            batch_start = time.time()

            output = self.model(input_ids, telem_tensor)
            logits = output['logits']
            action_logits = output['action_logits']

            # Compute LM loss
            lm_loss = F.cross_entropy(
                logits.view(-1, self.model.config.vocab_size),
                targets.view(-1)
            )

            # Sample action
            action_probs = F.softmax(action_logits[0], dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_idx = action.item()

            # Apply action to hardware
            self.actuator.set_mode_from_action(action_idx)

            torch.cuda.synchronize()
            batch_time = time.time() - batch_start

            # Read telemetry after
            snap_after = self.telemetry.read()

            # Compute smart reward
            tokens_generated = targets.numel()
            avg_power = (snap_before.power_watts + snap_after.power_watts) / 2
            avg_temp = (snap_before.temp_c + snap_after.temp_c) / 2
            utilization = snap_after.utilization

            reward, components = compute_smart_reward(
                tokens_generated=tokens_generated,
                time_elapsed=batch_time,
                power_watts=avg_power,
                temperature=avg_temp,
                utilization=utilization,
                action_idx=action_idx,
                prev_action_idx=self.prev_action,
                loss=lm_loss.item(),
                config=self.reward_config,
            )

            # RL loss (policy gradient)
            log_prob = action_dist.log_prob(action)
            rl_loss = -log_prob * reward

            # Combined loss and single backward
            total_loss = lm_loss * 0.1 + rl_loss  # Small LM weight during RL
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            episode_reward += reward
            episode_loss += lm_loss.item()
            self.prev_action = action_idx

            batch_metrics.append({
                'batch': i,
                'workload': workload,
                'action': self.action_names[action_idx],
                'reward': reward,
                'power': avg_power,
                'temp': avg_temp,
                'util': utilization,
                'loss': lm_loss.item(),
                'components': components,
            })

            self.action_history.append(action_idx)
            self.workload_history.append(workload)

            if i % 10 == 0:
                tpw = components.get('throughput_per_watt', 0) * 200  # Denormalize
                logger.info(
                    f"  B{i:3d} | {workload:6s} | {self.action_names[action_idx]:11s} | "
                    f"R:{reward:+.3f} | P:{avg_power:.0f}W | T:{avg_temp:.0f}C | "
                    f"TPW:{tpw:.1f}"
                )

        avg_reward = episode_reward / num_batches
        avg_loss = episode_loss / num_batches

        self.episode_rewards.append(avg_reward)
        self.episode_components.append(batch_metrics)

        return {
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
            'batch_metrics': batch_metrics,
        }

    def analyze_workload_adaptation(self) -> Dict:
        """Analyze how well the model adapts to different workloads."""

        if not self.action_history:
            return {}

        workload_actions = {'heavy': [], 'medium': [], 'light': [], 'idle': []}

        for action, workload in zip(self.action_history, self.workload_history):
            workload_actions[workload].append(action)

        analysis = {}
        for workload, actions in workload_actions.items():
            if actions:
                action_counts = {i: actions.count(i) for i in range(4)}
                dominant_action = max(action_counts, key=action_counts.get)
                analysis[workload] = {
                    'action_distribution': {self.action_names[k]: v/len(actions)
                                          for k, v in action_counts.items()},
                    'dominant_action': self.action_names[dominant_action],
                    'samples': len(actions),
                }

        return analysis


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_smart_metabolic_experiment():
    """Run the smart metabolic experiment."""

    from src.metabolic.film_transformer import MetabolicTransformer, MetabolicConfig
    from src.metabolic.telemetry_unified import UnifiedTelemetryReader
    from src.metabolic.actuation_unified import UnifiedActuator

    logger.info("=" * 60)
    logger.info("Z64: SMART METABOLIC EXPERIMENT")
    logger.info("Optimizing throughput/watt with temperature constraints")
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/z64_smart_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Initialize hardware
    telemetry = UnifiedTelemetryReader()
    actuator = UnifiedActuator()

    gpu_info = telemetry.get_device_info()
    logger.info(f"GPU: {gpu_info}")

    # Model config (40M params)
    config = MetabolicConfig(
        vocab_size=256,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        max_seq_len=256,
        dropout=0.1,
        telemetry_dim=12,
        film_hidden_dim=64,
        condition_every_layer=True,
        num_actions=4,
    )

    # Create model
    model = MetabolicTransformer(config).to(device)
    logger.info(f"Model params: {model.get_num_parameters():,}")

    # Create variable workload dataset
    corpus = "The quick brown fox jumps over the lazy dog. " * 5000
    dataset = VariableWorkloadDataset(corpus, config.max_seq_len, total_samples=2000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Smart reward config
    reward_config = SmartRewardConfig(
        throughput_weight=1.0,
        temp_target=75.0,
        temp_max=85.0,
        temp_penalty_scale=0.05,
        ppl_threshold=2.0,
        quality_weight=0.5,
        idle_util_threshold=20.0,
        idle_throttle_bonus=0.5,
        idle_full_power_penalty=0.3,
        action_switch_penalty=0.1,
    )

    # Optimizer (only FiLM + action head for RL)
    film_params = list(model.get_film_parameters())
    action_params = list(model.action_head.parameters())
    optimizer = torch.optim.Adam(film_params + action_params, lr=1e-4)

    logger.info(f"Trainable params: {sum(p.numel() for p in film_params + action_params):,}")

    # Create trainer
    trainer = SmartMetabolicTrainer(
        model=model,
        telemetry=telemetry,
        actuator=actuator,
        device=device,
        reward_config=reward_config,
    )

    # ============================================================
    # Stage 1: LM Pretraining (freeze FiLM, train LM)
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 1: LM Pretraining")
    logger.info("=" * 50)

    # Freeze FiLM, train full model
    for p in film_params:
        p.requires_grad = False

    lm_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(2):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for i, (input_ids, targets, _) in enumerate(dataloader):
            if i >= 100:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Random telemetry for pretraining
            telem = torch.rand(input_ids.size(0), 12, device=device)

            output = model(input_ids, telem)
            loss = F.cross_entropy(
                output['logits'].view(-1, config.vocab_size),
                targets.view(-1)
            )

            lm_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lm_optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if i % 30 == 0:
                snap = telemetry.read()
                logger.info(f"  E{epoch+1} B{i} | Loss: {loss.item():.4f} | Power: {snap.power_watts:.0f}W")

        logger.info(f"  Epoch {epoch+1} | Avg Loss: {epoch_loss/num_batches:.4f}")

    # Unfreeze FiLM
    for p in film_params:
        p.requires_grad = True

    # ============================================================
    # Stage 2: Smart RL Training
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 2: Smart RL Training (throughput/watt)")
    logger.info("=" * 50)

    for epoch in range(5):
        logger.info(f"\nEpoch {epoch + 1}/5")
        result = trainer.train_episode(dataloader, optimizer, num_batches=40)
        logger.info(f"  Avg Reward: {result['avg_reward']:.4f}")

    # Analyze workload adaptation
    logger.info("\n" + "=" * 50)
    logger.info("WORKLOAD ADAPTATION ANALYSIS")
    logger.info("=" * 50)

    adaptation = trainer.analyze_workload_adaptation()
    for workload, data in adaptation.items():
        logger.info(f"\n{workload.upper()} workload:")
        logger.info(f"  Dominant action: {data['dominant_action']}")
        logger.info(f"  Distribution: {data['action_distribution']}")

    # ============================================================
    # Stage 3: Evaluation
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 3: Evaluation")
    logger.info("=" * 50)

    model.eval()
    actuator.reset_to_default()

    eval_metrics = {
        'total_tokens': 0,
        'total_time': 0,
        'total_energy': 0,
        'powers': [],
        'temps': [],
        'actions': [],
    }

    with torch.no_grad():
        for i, (input_ids, targets, workloads) in enumerate(dataloader):
            if i >= 50:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            snap_before = telemetry.read()
            body_state = telemetry.read_body_state()
            telem = torch.from_numpy(body_state).float().to(device)
            telem = telem.unsqueeze(0).expand(input_ids.size(0), -1)

            start = time.time()
            output = model(input_ids, telem)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            snap_after = telemetry.read()

            # Apply learned action
            action_probs = F.softmax(output['action_logits'][0], dim=-1)
            action_idx = torch.argmax(action_probs).item()
            actuator.set_mode_from_action(action_idx)

            avg_power = (snap_before.power_watts + snap_after.power_watts) / 2
            energy = avg_power * elapsed

            eval_metrics['total_tokens'] += targets.numel()
            eval_metrics['total_time'] += elapsed
            eval_metrics['total_energy'] += energy
            eval_metrics['powers'].append(avg_power)
            eval_metrics['temps'].append(snap_after.temp_c)
            eval_metrics['actions'].append(action_idx)

    actuator.reset_to_default()

    # Compute final metrics
    tokens_per_sec = eval_metrics['total_tokens'] / eval_metrics['total_time']
    j_per_token = eval_metrics['total_energy'] / eval_metrics['total_tokens']
    avg_power = np.mean(eval_metrics['powers'])
    avg_temp = np.mean(eval_metrics['temps'])
    throughput_per_watt = tokens_per_sec / avg_power

    action_counts = {i: eval_metrics['actions'].count(i) for i in range(4)}

    logger.info(f"\nFinal Metrics:")
    logger.info(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    logger.info(f"  Power: {avg_power:.1f}W")
    logger.info(f"  Throughput/Watt: {throughput_per_watt:.2f} tok/s/W")
    logger.info(f"  mJ/token: {j_per_token * 1000:.3f}")
    logger.info(f"  Temperature: {avg_temp:.1f}C")
    logger.info(f"  Action distribution: {action_counts}")

    # Save results
    results = {
        'timestamp': timestamp,
        'config': asdict(config),
        'reward_config': asdict(reward_config),
        'training': {
            'episode_rewards': trainer.episode_rewards,
        },
        'adaptation': adaptation,
        'evaluation': {
            'tokens_per_sec': tokens_per_sec,
            'avg_power': avg_power,
            'throughput_per_watt': throughput_per_watt,
            'mj_per_token': j_per_token * 1000,
            'avg_temp': avg_temp,
            'action_distribution': action_counts,
        },
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = run_smart_metabolic_experiment()
