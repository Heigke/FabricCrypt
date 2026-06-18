#!/usr/bin/env python3
"""
Z60: Full Metabolic Transformer Comparison
==========================================
Complete experiment with:
1. Baseline training
2. Metabolic Stage 1 (LM pretrain with telemetry conditioning)
3. Metabolic Stage 2 (RL training with energy reward)
4. Comprehensive evaluation comparing all variants

This demonstrates whether the metabolic model can actually learn
to be energy-efficient through the RL training stage.

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
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.metabolic.film_transformer import (
    MetabolicTransformer, BaselineTransformer, MetabolicConfig
)
from src.metabolic.telemetry_unified import UnifiedTelemetryReader
from src.metabolic.actuation_unified import UnifiedActuator, MetabolicMode
from src.metabolic.energy_tracker import EnergyTracker, TokenLevelEnergyTracker, EnergyRewardShaper
from src.metabolic.metabolic_trainer import CharDataset, TelemetryScheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def get_best_device() -> torch.device:
    """Get best available device with fallback."""
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1, device='cuda')
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.synchronize()
            return torch.device('cuda')
        except Exception as e:
            logger.warning(f"GPU not functional: {e}, using CPU")
    return torch.device('cpu')


def prepare_corpus(size: int = 200000) -> str:
    """Prepare training corpus with diverse content."""
    texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Machine learning models process data efficiently. ",
        "def function(x): return x * 2 ",
        "Energy efficiency is crucial for sustainable AI. ",
        "Hardware aware computing optimizes performance. ",
        "The transformer architecture revolutionized NLP. ",
        "Metabolic processes regulate energy in organisms. ",
        "Neural networks learn through gradient descent. ",
    ]
    corpus = " ".join([t * 100 for t in texts])
    while len(corpus) < size:
        corpus += corpus[:size // 5]
    return corpus[:size]


def train_lm_stage(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    telemetry_reader: Optional[UnifiedTelemetryReader] = None,
    is_metabolic: bool = True,
) -> Dict:
    """Stage 1: Language model pretraining."""
    logger.info(f"Stage 1: LM Pretraining ({'metabolic' if is_metabolic else 'baseline'})...")

    model.train()
    optimizer = AdamW(model.parameters(), lr=3e-4)
    telem_scheduler = TelemetryScheduler(dim=12, schedule='random')

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            if is_metabolic:
                if telemetry_reader:
                    telem_np = telemetry_reader.read_body_state()
                    telemetry = torch.from_numpy(telem_np).float().to(device)
                    telemetry = telemetry.unsqueeze(0).expand(input_ids.size(0), -1)
                else:
                    telemetry = telem_scheduler.get_telemetry(input_ids.size(0)).to(device)
                output = model(input_ids, telemetry)
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

            if batch_idx % 100 == 0:
                logger.info(f"  Epoch {epoch+1} Batch {batch_idx} | Loss: {loss.item():.4f}")

            if batch_idx >= 200:
                break

        logger.info(f"  Epoch {epoch+1} complete | Avg Loss: {epoch_loss / min(len(train_loader), 200):.4f}")

    return {'losses': losses, 'final_loss': losses[-1] if losses else 0}


def train_rl_stage(
    model: MetabolicTransformer,
    train_loader: DataLoader,
    device: torch.device,
    telemetry_reader: UnifiedTelemetryReader,
    actuator: UnifiedActuator,
    epochs: int = 2,
    energy_coef: float = 1.0,
) -> Dict:
    """Stage 2: RL training with energy reward."""
    logger.info("Stage 2: RL Training with Energy Reward...")

    # Freeze base model, train only action head + FiLM
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []
    for param in model.get_action_head_parameters():
        param.requires_grad = True
        trainable_params.append(param)
    for param in model.get_film_parameters():
        param.requires_grad = True
        trainable_params.append(param)

    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info(f"  Trainable parameters: {trainable_count:,}")

    optimizer = AdamW(trainable_params, lr=1e-4)
    energy_tracker = TokenLevelEnergyTracker(telemetry_reader)
    reward_shaper = EnergyRewardShaper(alpha=energy_coef)

    model.train()
    rewards = []
    energies = []
    action_history = []

    for epoch in range(epochs):
        epoch_rewards = []

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            if batch_idx >= 50:  # Limit batches for RL
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Get telemetry
            telem_np = telemetry_reader.read_body_state()
            telemetry = torch.from_numpy(telem_np).float().to(device)
            telemetry = telemetry.unsqueeze(0).expand(input_ids.size(0), -1)

            # Reset energy tracker
            energy_tracker.reset()

            # Forward
            output = model(input_ids, telemetry)
            action_logits = output['action_logits']

            # Sample actions
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)

            # Apply actions to hardware
            for action_idx in actions.tolist():
                actuator.set_mode_from_action(action_idx)
                action_history.append(action_idx)

            # Measure energy
            time.sleep(0.01)  # Small delay for hardware to respond
            energy = energy_tracker.tick()
            energies.append(energy)

            # Compute reward (lower energy = higher reward)
            reward = reward_shaper.linear_reward(energy)
            rewards.append(reward)
            epoch_rewards.append(reward)

            # Policy gradient loss
            entropy = action_dist.entropy().mean()
            pg_loss = -(log_probs * reward).mean() - 0.01 * entropy

            # LM loss (to maintain quality)
            lm_loss = F.cross_entropy(
                output['logits'].view(-1, model.config.vocab_size),
                targets.view(-1)
            )

            loss = pg_loss + 0.1 * lm_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            if batch_idx % 10 == 0:
                logger.info(
                    f"  Epoch {epoch+1} Batch {batch_idx} | "
                    f"Reward: {reward:.4f} | Energy: {energy*1000:.2f}mJ | "
                    f"Action: {actions[0].item()}"
                )

        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        logger.info(f"  Epoch {epoch+1} complete | Avg Reward: {avg_reward:.4f}")

    # Restore auto mode
    actuator.reset_to_default()

    # Unfreeze model
    for param in model.parameters():
        param.requires_grad = True

    # Analyze action distribution after RL
    from collections import Counter
    action_counts = Counter(action_history)
    action_dist_rl = [action_counts.get(i, 0) / len(action_history) for i in range(4)]

    return {
        'rewards': rewards,
        'energies': energies,
        'action_distribution_rl': action_dist_rl,
        'final_reward': rewards[-1] if rewards else 0,
    }


def evaluate_model(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    telemetry_reader: UnifiedTelemetryReader,
    actuator: UnifiedActuator,
    energy_tracker: EnergyTracker,
    is_metabolic: bool = True,
    num_samples: int = 100,
    label: str = "",
) -> Dict:
    """Evaluate model with energy measurement."""
    logger.info(f"Evaluating {label}...")

    model.eval()
    total_loss = 0
    total_tokens = 0
    action_counts = [0, 0, 0, 0]
    temps = []

    energy_tracker.start_session()
    start_time = time.time()

    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(eval_loader):
            if i >= num_samples:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            telem_np = telemetry_reader.read_body_state()
            telemetry = torch.from_numpy(telem_np).float().to(device).unsqueeze(0)

            if is_metabolic:
                output = model(input_ids, telemetry)
                action_probs = F.softmax(output['action_logits'], dim=-1)
                action_idx = torch.argmax(action_probs, dim=-1).item()
                action_counts[action_idx] += 1
                actuator.set_mode_from_action(action_idx)
            else:
                output = model(input_ids)

            loss = F.cross_entropy(
                output['logits'].view(-1, model.config.vocab_size),
                targets.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
            energy_tracker.record_tokens(targets.numel())

            snap = telemetry_reader.read()
            temps.append(snap.temp_c)

    end_time = time.time()
    session = energy_tracker.stop_session()
    actuator.reset_to_default()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)

    return {
        'perplexity': perplexity,
        'j_per_token': session.j_per_token,
        'tokens_per_sec': total_tokens / max(end_time - start_time, 0.001),
        'total_energy_j': session.total_energy_j,
        'temp_mean': np.mean(temps) if temps else 0,
        'action_distribution': [c / max(sum(action_counts), 1) for c in action_counts],
    }


def plot_comparison(results: Dict, output_dir: Path):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # PPL comparison
    ax = axes[0, 0]
    models = ['Baseline', 'Metabolic\n(Stage 1)', 'Metabolic\n(Stage 2)']
    ppls = [
        results['baseline']['eval']['perplexity'],
        results['metabolic_stage1']['eval']['perplexity'],
        results['metabolic_stage2']['eval']['perplexity'],
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    ax.bar(models, ppls, color=colors)
    ax.set_ylabel('Perplexity')
    ax.set_title('Language Model Quality')
    ax.set_ylim(0, max(ppls) * 1.2)
    for i, v in enumerate(ppls):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center')

    # Energy comparison
    ax = axes[0, 1]
    energies = [
        results['baseline']['eval']['j_per_token'] * 1000,
        results['metabolic_stage1']['eval']['j_per_token'] * 1000,
        results['metabolic_stage2']['eval']['j_per_token'] * 1000,
    ]
    ax.bar(models, energies, color=colors)
    ax.set_ylabel('Energy (mJ/token)')
    ax.set_title('Energy Consumption')
    ax.set_ylim(0, max(energies) * 1.2)
    for i, v in enumerate(energies):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center')

    # Action distribution (Stage 2)
    ax = axes[1, 0]
    action_names = ['ECO', 'BALANCED', 'PERFORMANCE', 'MAX']
    stage1_actions = results['metabolic_stage1']['eval']['action_distribution']
    stage2_actions = results['metabolic_stage2']['eval']['action_distribution']

    x = np.arange(len(action_names))
    width = 0.35
    ax.bar(x - width/2, stage1_actions, width, label='After Stage 1', color='#ff7f0e')
    ax.bar(x + width/2, stage2_actions, width, label='After Stage 2 (RL)', color='#2ca02c')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(action_names)
    ax.legend()

    # Training curves
    ax = axes[1, 1]
    if 'rl_training' in results['metabolic_stage2']:
        rewards = results['metabolic_stage2']['rl_training']['rewards']
        ax.plot(rewards, alpha=0.5, label='Raw rewards')
        # Smoothed
        window = min(10, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, label='Smoothed', linewidth=2)
    ax.set_xlabel('RL Step')
    ax.set_ylabel('Reward')
    ax.set_title('RL Training Progress')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150)
    plt.close()

    logger.info(f"Saved comparison plot to {output_dir / 'comparison.png'}")


def run_full_comparison():
    """Run full comparison experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/z60_full_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("METABOLIC TRANSFORMER: FULL COMPARISON")
    logger.info("="*60)

    device = get_best_device()
    logger.info(f"Device: {device}")

    # Hardware interfaces
    telemetry = UnifiedTelemetryReader()
    actuator = UnifiedActuator()
    energy_tracker = EnergyTracker(telemetry)

    device_info = telemetry.get_device_info()
    logger.info(f"GPU: {device_info}")

    # Config
    config = MetabolicConfig(
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=512,
        telemetry_dim=12,
        max_seq_len=128,
    )

    # Data
    logger.info("Preparing corpus...")
    corpus = prepare_corpus(200000)
    dataset = CharDataset(corpus, config.max_seq_len)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = {
        'timestamp': timestamp,
        'device': str(device),
        'device_info': device_info,
        'config': asdict(config),
    }

    # ========== BASELINE ==========
    logger.info("\n" + "="*50)
    logger.info("BASELINE MODEL")
    logger.info("="*50)

    baseline = BaselineTransformer(config).to(device)
    baseline_train = train_lm_stage(
        baseline, train_loader, device,
        epochs=3, telemetry_reader=None, is_metabolic=False
    )
    baseline_eval = evaluate_model(
        baseline, eval_loader, device,
        telemetry, actuator, energy_tracker,
        is_metabolic=False, num_samples=100, label="BASELINE"
    )
    results['baseline'] = {'train': baseline_train, 'eval': baseline_eval}

    # ========== METABOLIC STAGE 1 ==========
    logger.info("\n" + "="*50)
    logger.info("METABOLIC MODEL - STAGE 1 (LM Pretrain)")
    logger.info("="*50)

    metabolic = MetabolicTransformer(config).to(device)
    metabolic_train = train_lm_stage(
        metabolic, train_loader, device,
        epochs=3, telemetry_reader=telemetry, is_metabolic=True
    )
    metabolic_stage1_eval = evaluate_model(
        metabolic, eval_loader, device,
        telemetry, actuator, energy_tracker,
        is_metabolic=True, num_samples=100, label="METABOLIC (Stage 1)"
    )
    results['metabolic_stage1'] = {'train': metabolic_train, 'eval': metabolic_stage1_eval}

    # ========== METABOLIC STAGE 2 ==========
    logger.info("\n" + "="*50)
    logger.info("METABOLIC MODEL - STAGE 2 (RL with Energy Reward)")
    logger.info("="*50)

    rl_training = train_rl_stage(
        metabolic, train_loader, device,
        telemetry, actuator,
        epochs=2, energy_coef=5.0  # Strong energy penalty
    )
    metabolic_stage2_eval = evaluate_model(
        metabolic, eval_loader, device,
        telemetry, actuator, energy_tracker,
        is_metabolic=True, num_samples=100, label="METABOLIC (Stage 2)"
    )
    results['metabolic_stage2'] = {
        'rl_training': rl_training,
        'eval': metabolic_stage2_eval
    }

    # ========== COMPARISON ==========
    logger.info("\n" + "="*60)
    logger.info("RESULTS COMPARISON")
    logger.info("="*60)

    b_ppl = results['baseline']['eval']['perplexity']
    m1_ppl = results['metabolic_stage1']['eval']['perplexity']
    m2_ppl = results['metabolic_stage2']['eval']['perplexity']

    b_energy = results['baseline']['eval']['j_per_token']
    m1_energy = results['metabolic_stage1']['eval']['j_per_token']
    m2_energy = results['metabolic_stage2']['eval']['j_per_token']

    print(f"\n{'Model':<25} {'PPL':<10} {'mJ/token':<10} {'Actions'}")
    print("-"*60)
    print(f"{'Baseline':<25} {b_ppl:<10.2f} {b_energy*1000:<10.2f} N/A")
    print(f"{'Metabolic (Stage 1)':<25} {m1_ppl:<10.2f} {m1_energy*1000:<10.2f} {[f'{a:.0%}' for a in results['metabolic_stage1']['eval']['action_distribution']]}")
    print(f"{'Metabolic (Stage 2 RL)':<25} {m2_ppl:<10.2f} {m2_energy*1000:<10.2f} {[f'{a:.0%}' for a in results['metabolic_stage2']['eval']['action_distribution']]}")

    # Compute improvements
    ppl_change_s1 = (m1_ppl - b_ppl) / b_ppl * 100
    ppl_change_s2 = (m2_ppl - b_ppl) / b_ppl * 100
    energy_change_s1 = (m1_energy - b_energy) / max(b_energy, 1e-9) * 100
    energy_change_s2 = (m2_energy - b_energy) / max(b_energy, 1e-9) * 100

    print(f"\n{'Change vs Baseline:':<25}")
    print(f"{'  Stage 1':<25} PPL: {ppl_change_s1:+.1f}%, Energy: {energy_change_s1:+.1f}%")
    print(f"{'  Stage 2 (RL)':<25} PPL: {ppl_change_s2:+.1f}%, Energy: {energy_change_s2:+.1f}%")

    results['comparison'] = {
        'ppl_change_s1': ppl_change_s1,
        'ppl_change_s2': ppl_change_s2,
        'energy_change_s1': energy_change_s1,
        'energy_change_s2': energy_change_s2,
    }

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate plots
    try:
        plot_comparison(results, output_dir)
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    results = run_full_comparison()
