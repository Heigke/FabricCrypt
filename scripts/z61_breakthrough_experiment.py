#!/usr/bin/env python3
"""
Z61: Breakthrough Metabolic Transformer Experiment
===================================================
Larger model (40M params) with real power control to achieve
actual energy savings through learned hardware actions.

Key changes from z60:
- 40M param model (vs 1M) for real GPU utilization
- Batch size 32 (vs 1) to stress GPU
- Verified power limit control via sudo nvidia-smi
- Proper energy delta reward for RL

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
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from src.metabolic.film_transformer import (
    MetabolicTransformer, BaselineTransformer, MetabolicConfig
)
from src.metabolic.telemetry_unified import UnifiedTelemetryReader
from src.metabolic.actuation_unified import UnifiedActuator, MetabolicMode
from src.metabolic.energy_tracker import EnergyTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class CharDataset(Dataset):
    """Character-level dataset for language modeling."""

    def __init__(self, text: str, seq_len: int):
        self.data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def prepare_corpus(size: int = 500000) -> str:
    """Prepare larger training corpus."""
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 500,
        "Machine learning models process data efficiently using neural networks. " * 500,
        "def function(x): return x * 2 + 1\nclass Model(nn.Module): pass\n" * 500,
        "Energy efficiency matters for sustainable computing and green AI. " * 500,
        "Hardware and software must work together seamlessly for optimal performance. " * 500,
        "The transformer architecture revolutionized natural language processing. " * 500,
        "GPU acceleration enables training of large-scale deep learning models. " * 500,
    ]
    corpus = " ".join(texts)
    while len(corpus) < size:
        corpus += corpus[:size // 5]
    return corpus[:size]


def train_lm_stage(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    telemetry: UnifiedTelemetryReader,
    epochs: int = 2,
    is_metabolic: bool = True,
    max_batches: int = 150,
) -> Dict:
    """Stage 1: LM pretraining."""
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            if is_metabolic:
                telem_np = telemetry.read_body_state()
                telem = torch.from_numpy(telem_np).float().to(device)
                telem = telem.unsqueeze(0).expand(input_ids.size(0), -1)
                output = model(input_ids, telem)
            else:
                output = model(input_ids)

            loss = F.cross_entropy(
                output['logits'].view(-1, model.config.vocab_size),
                targets.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            losses.append(loss.item())
            batch_count += 1

            if batch_idx % 30 == 0:
                snap = telemetry.read()
                logger.info(f"  E{epoch+1} B{batch_idx} | Loss: {loss.item():.4f} | Power: {snap.power_watts:.0f}W")

        logger.info(f"  Epoch {epoch+1} | Avg Loss: {epoch_loss / batch_count:.4f}")

    return {
        'final_loss': losses[-1] if losses else 0,
        'avg_loss': np.mean(losses),
        'train_time': time.time() - start_time,
    }


def train_rl_stage(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    telemetry: UnifiedTelemetryReader,
    actuator: UnifiedActuator,
    epochs: int = 3,
    max_batches: int = 80,
) -> Dict:
    """Stage 2: RL training with energy reward."""

    # Freeze LM, train only action head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_action_head_parameters():
        param.requires_grad = True
    for param in model.get_film_parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  RL trainable params: {trainable:,}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4
    )

    model.train()
    rewards_history = []
    action_counts = [0, 0, 0, 0]
    energy_by_action = {0: [], 1: [], 2: [], 3: []}

    # Baseline energy (no action)
    baseline_energy = None

    for epoch in range(epochs):
        epoch_rewards = []

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Get telemetry
            telem_np = telemetry.read_body_state()
            telem = torch.from_numpy(telem_np).float().to(device)
            telem = telem.unsqueeze(0).expand(input_ids.size(0), -1)

            # Forward pass
            output = model(input_ids, telem)

            # Sample action (use first sample in batch for action decision)
            action_logits = output['action_logits'][0]  # Take first sample
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_idx = action.item()

            # Apply action
            actuator.set_mode_from_action(action_idx)

            # Measure energy for this batch
            snap_before = telemetry.read()
            start = time.time()

            # Run inference (simulate token generation)
            with torch.no_grad():
                _ = model(input_ids, telem)
            torch.cuda.synchronize()

            elapsed = time.time() - start
            snap_after = telemetry.read()

            # Energy = avg_power * time
            avg_power = (snap_before.power_watts + snap_after.power_watts) / 2
            energy_j = avg_power * elapsed
            energy_per_token = energy_j / (input_ids.numel())

            # Track energy by action
            energy_by_action[action_idx].append(energy_per_token)
            action_counts[action_idx] += 1

            # Compute baseline on first batch
            if baseline_energy is None:
                baseline_energy = energy_per_token

            # Reward: negative energy (want to minimize)
            # Normalize by baseline to get meaningful gradients
            reward = -energy_per_token / max(baseline_energy, 1e-9)

            # Add quality penalty if loss is bad
            with torch.no_grad():
                lm_loss = F.cross_entropy(
                    output['logits'].view(-1, model.config.vocab_size),
                    targets.view(-1)
                )
            quality_penalty = max(0, lm_loss.item() - 0.5) * 0.1
            reward -= quality_penalty

            rewards_history.append(reward)
            epoch_rewards.append(reward)

            # Policy gradient loss
            log_prob = action_dist.log_prob(action)
            pg_loss = -log_prob * reward

            # Add entropy bonus for exploration
            entropy = action_dist.entropy()
            loss = pg_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if batch_idx % 20 == 0:
                logger.info(f"  E{epoch+1} B{batch_idx} | R: {reward:.3f} | E: {energy_per_token*1000:.2f}mJ | A: {action_idx} | P: {avg_power:.0f}W")

        # Log epoch stats
        avg_reward = np.mean(epoch_rewards)
        logger.info(f"  Epoch {epoch+1} | Avg Reward: {avg_reward:.4f}")

    # Reset to default
    actuator.reset_to_default()

    # Compute energy stats by action
    energy_stats = {}
    for a in range(4):
        if energy_by_action[a]:
            energy_stats[f'action_{a}_avg_mj'] = np.mean(energy_by_action[a]) * 1000
            energy_stats[f'action_{a}_count'] = len(energy_by_action[a])

    return {
        'rewards': rewards_history,
        'avg_reward': np.mean(rewards_history),
        'action_counts': action_counts,
        'energy_stats': energy_stats,
    }


def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    telemetry: UnifiedTelemetryReader,
    actuator: UnifiedActuator,
    energy_tracker: EnergyTracker,
    is_metabolic: bool = True,
    num_samples: int = 50,
) -> Dict:
    """Evaluate model and measure energy."""
    model.eval()

    total_loss = 0
    total_tokens = 0
    action_counts = [0, 0, 0, 0]
    powers = []
    temps = []
    energies = []

    energy_tracker.start_session()
    start_time = time.time()

    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(eval_loader):
            if i >= num_samples:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Get telemetry
            telem_np = telemetry.read_body_state()
            telem = torch.from_numpy(telem_np).float().to(device)
            telem = telem.unsqueeze(0).expand(input_ids.size(0), -1)

            # Measure batch energy
            snap_before = telemetry.read()
            batch_start = time.time()

            if is_metabolic:
                output = model(input_ids, telem)

                # Get and apply action (use first sample in batch for action)
                action_probs = F.softmax(output['action_logits'], dim=-1)
                action_idx = torch.argmax(action_probs[0], dim=-1).item()
                action_counts[action_idx] += 1
                actuator.set_mode_from_action(action_idx)
            else:
                output = model(input_ids)

            torch.cuda.synchronize()
            batch_time = time.time() - batch_start
            snap_after = telemetry.read()

            # Calculate batch energy
            avg_power = (snap_before.power_watts + snap_after.power_watts) / 2
            batch_energy = avg_power * batch_time
            batch_tokens = targets.numel()

            energies.append(batch_energy / batch_tokens)
            powers.append(avg_power)
            temps.append(snap_after.temp_c)

            loss = F.cross_entropy(
                output['logits'].view(-1, model.config.vocab_size),
                targets.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += batch_tokens
            energy_tracker.record_tokens(batch_tokens)

    end_time = time.time()
    session = energy_tracker.stop_session()
    actuator.reset_to_default()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 10))  # Cap to avoid overflow
    tokens_per_sec = total_tokens / max(end_time - start_time, 0.001)

    action_dist = [c / max(sum(action_counts), 1) for c in action_counts]

    return {
        'perplexity': perplexity,
        'j_per_token': np.mean(energies) if energies else 0,
        'tokens_per_sec': tokens_per_sec,
        'total_tokens': total_tokens,
        'power_avg': np.mean(powers) if powers else 0,
        'power_std': np.std(powers) if powers else 0,
        'temp_avg': np.mean(temps) if temps else 0,
        'temp_max': max(temps) if temps else 0,
        'action_distribution': action_dist,
    }


def run_breakthrough_experiment():
    """Run the breakthrough experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/z61_breakthrough_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BREAKTHROUGH METABOLIC TRANSFORMER EXPERIMENT")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Hardware interfaces
    telemetry = UnifiedTelemetryReader()
    actuator = UnifiedActuator()
    energy_tracker = EnergyTracker(telemetry)

    device_info = telemetry.get_device_info()
    logger.info(f"GPU: {device_info}")

    # LARGER model configuration for real GPU utilization
    config = MetabolicConfig(
        hidden_dim=512,       # 4x larger
        num_layers=12,        # 3x more layers
        num_heads=8,
        ff_dim=2048,
        telemetry_dim=12,
        max_seq_len=256,
    )

    # Prepare data
    logger.info("Preparing training corpus...")
    corpus = prepare_corpus(500000)
    dataset = CharDataset(corpus, config.max_seq_len)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)

    results = {
        'timestamp': timestamp,
        'device': str(device),
        'device_info': device_info,
        'config': asdict(config),
    }

    # ===== BASELINE =====
    logger.info("\n" + "=" * 50)
    logger.info("BASELINE MODEL (No conditioning)")
    logger.info("=" * 50)

    baseline = BaselineTransformer(config).to(device)
    logger.info(f"Baseline params: {baseline.get_num_parameters():,}")

    baseline_train = train_lm_stage(
        baseline, train_loader, device, telemetry,
        epochs=2, is_metabolic=False, max_batches=150
    )

    logger.info("Evaluating BASELINE...")
    baseline_eval = evaluate_model(
        baseline, eval_loader, device,
        telemetry, actuator, energy_tracker,
        is_metabolic=False, num_samples=50
    )

    results['baseline'] = {'train': baseline_train, 'eval': baseline_eval}
    logger.info(f"Baseline PPL: {baseline_eval['perplexity']:.2f}")
    logger.info(f"Baseline mJ/token: {baseline_eval['j_per_token']*1000:.2f}")
    logger.info(f"Baseline power: {baseline_eval['power_avg']:.1f}W")

    # ===== METABOLIC STAGE 1 =====
    logger.info("\n" + "=" * 50)
    logger.info("METABOLIC MODEL - Stage 1 (LM Pretrain)")
    logger.info("=" * 50)

    metabolic = MetabolicTransformer(config).to(device)
    logger.info(f"Metabolic params: {metabolic.get_num_parameters():,}")
    logger.info(f"  FiLM params: {sum(p.numel() for p in metabolic.get_film_parameters()):,}")
    logger.info(f"  Action head: {sum(p.numel() for p in metabolic.get_action_head_parameters()):,}")

    metabolic_s1_train = train_lm_stage(
        metabolic, train_loader, device, telemetry,
        epochs=2, is_metabolic=True, max_batches=150
    )

    logger.info("Evaluating METABOLIC (Stage 1)...")
    metabolic_s1_eval = evaluate_model(
        metabolic, eval_loader, device,
        telemetry, actuator, energy_tracker,
        is_metabolic=True, num_samples=50
    )

    results['metabolic_s1'] = {'train': metabolic_s1_train, 'eval': metabolic_s1_eval}
    logger.info(f"Stage 1 PPL: {metabolic_s1_eval['perplexity']:.2f}")
    logger.info(f"Stage 1 mJ/token: {metabolic_s1_eval['j_per_token']*1000:.2f}")
    logger.info(f"Stage 1 actions: {[f'{a:.0%}' for a in metabolic_s1_eval['action_distribution']]}")

    # ===== METABOLIC STAGE 2 (RL) =====
    logger.info("\n" + "=" * 50)
    logger.info("METABOLIC MODEL - Stage 2 (RL with Energy Reward)")
    logger.info("=" * 50)

    metabolic_s2_train = train_rl_stage(
        metabolic, train_loader, device,
        telemetry, actuator,
        epochs=3, max_batches=80
    )

    logger.info("Evaluating METABOLIC (Stage 2)...")
    metabolic_s2_eval = evaluate_model(
        metabolic, eval_loader, device,
        telemetry, actuator, energy_tracker,
        is_metabolic=True, num_samples=50
    )

    results['metabolic_s2'] = {'train': metabolic_s2_train, 'eval': metabolic_s2_eval}
    logger.info(f"Stage 2 PPL: {metabolic_s2_eval['perplexity']:.2f}")
    logger.info(f"Stage 2 mJ/token: {metabolic_s2_eval['j_per_token']*1000:.2f}")
    logger.info(f"Stage 2 actions: {[f'{a:.0%}' for a in metabolic_s2_eval['action_distribution']]}")

    # ===== COMPARISON =====
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)

    b_energy = baseline_eval['j_per_token'] * 1000
    s1_energy = metabolic_s1_eval['j_per_token'] * 1000
    s2_energy = metabolic_s2_eval['j_per_token'] * 1000

    s1_delta = (s1_energy - b_energy) / b_energy * 100 if b_energy > 0 else 0
    s2_delta = (s2_energy - b_energy) / b_energy * 100 if b_energy > 0 else 0

    results['comparison'] = {
        'baseline_mj': b_energy,
        's1_mj': s1_energy,
        's2_mj': s2_energy,
        's1_delta_pct': s1_delta,
        's2_delta_pct': s2_delta,
        'baseline_ppl': baseline_eval['perplexity'],
        's1_ppl': metabolic_s1_eval['perplexity'],
        's2_ppl': metabolic_s2_eval['perplexity'],
    }

    print("\n" + "=" * 60)
    print("BREAKTHROUGH EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"{'Model':<25} {'PPL':<10} {'mJ/token':<12} {'Power':<10} {'Actions'}")
    print("-" * 70)
    print(f"{'Baseline':<25} {baseline_eval['perplexity']:<10.2f} {b_energy:<12.2f} {baseline_eval['power_avg']:<10.1f} N/A")
    print(f"{'Metabolic (Stage 1)':<25} {metabolic_s1_eval['perplexity']:<10.2f} {s1_energy:<12.2f} {metabolic_s1_eval['power_avg']:<10.1f} {[f'{a:.0%}' for a in metabolic_s1_eval['action_distribution']]}")
    print(f"{'Metabolic (Stage 2 RL)':<25} {metabolic_s2_eval['perplexity']:<10.2f} {s2_energy:<12.2f} {metabolic_s2_eval['power_avg']:<10.1f} {[f'{a:.0%}' for a in metabolic_s2_eval['action_distribution']]}")
    print()
    print(f"Energy change vs baseline:")
    print(f"  Stage 1: {s1_delta:+.1f}%")
    print(f"  Stage 2: {s2_delta:+.1f}%")

    if s2_delta < 0:
        print(f"\n🎉 BREAKTHROUGH: {abs(s2_delta):.1f}% energy savings achieved!")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = run_breakthrough_experiment()
