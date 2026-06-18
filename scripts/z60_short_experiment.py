#!/usr/bin/env python3
"""
Z60: Short Metabolic Transformer Experiment
============================================
Quick experiment to verify the full pipeline works and produces
meaningful results for comparison.

Runs a shorter version of the full experiment suitable for
initial validation.

Usage:
    python scripts/z60_short_experiment.py

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
from src.metabolic.energy_tracker import EnergyTracker, TokenLevelEnergyTracker
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
    return torch.device('cpu')


def prepare_corpus(size: int = 100000) -> str:
    """Prepare training corpus."""
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 200,
        "Machine learning models process data efficiently. " * 200,
        "def function(): return value " * 200,
        "Energy efficiency matters for sustainable computing. " * 200,
        "Hardware and software must work together seamlessly. " * 200,
    ]
    corpus = " ".join(texts)
    while len(corpus) < size:
        corpus += corpus[:size // 5]
    return corpus[:size]


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 2,
    telemetry_reader: Optional[UnifiedTelemetryReader] = None,
    is_metabolic: bool = True,
) -> Dict:
    """Train model and return metrics."""
    model.train()
    optimizer = AdamW(model.parameters(), lr=3e-4)
    telem_scheduler = TelemetryScheduler(dim=12)

    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Get telemetry
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

            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1} Batch {batch_idx} | Loss: {loss.item():.4f}")

            if batch_idx >= 100:  # Limit batches per epoch for quick experiment
                break

        logger.info(f"Epoch {epoch+1} complete | Avg Loss: {epoch_loss / min(len(train_loader), 100):.4f}")

    train_time = time.time() - start_time
    return {
        'final_loss': losses[-1] if losses else 0,
        'avg_loss': np.mean(losses),
        'train_time': train_time,
        'losses': losses,
    }


def evaluate_model(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    telemetry_reader: UnifiedTelemetryReader,
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
    temps = []

    energy_tracker.start_session()
    start_time = time.time()

    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(eval_loader):
            if i >= num_samples:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Get telemetry
            telem_np = telemetry_reader.read_body_state()
            telemetry = torch.from_numpy(telem_np).float().to(device).unsqueeze(0)

            if is_metabolic:
                output = model(input_ids, telemetry)
                # Get and apply action
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

            # Record temperature
            snap = telemetry_reader.read()
            temps.append(snap.temp_c)

    end_time = time.time()
    session = energy_tracker.stop_session()

    actuator.reset_to_default()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    tokens_per_sec = total_tokens / max(end_time - start_time, 0.001)

    action_dist = [c / max(sum(action_counts), 1) for c in action_counts]

    return {
        'perplexity': perplexity,
        'j_per_token': session.j_per_token,
        'tokens_per_sec': tokens_per_sec,
        'total_tokens': total_tokens,
        'total_energy_j': session.total_energy_j,
        'temp_mean': np.mean(temps) if temps else 0,
        'temp_max': max(temps) if temps else 0,
        'action_distribution': action_dist,
    }


def run_short_experiment():
    """Run short experiment comparing baseline and metabolic."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/z60_short_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("METABOLIC TRANSFORMER: SHORT EXPERIMENT")
    logger.info("="*60)

    device = get_best_device()
    logger.info(f"Device: {device}")

    # Hardware interfaces
    telemetry = UnifiedTelemetryReader()
    actuator = UnifiedActuator()
    energy_tracker = EnergyTracker(telemetry)

    device_info = telemetry.get_device_info()
    logger.info(f"GPU: {device_info}")

    # Model configuration - small for quick experiment
    config = MetabolicConfig(
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=512,
        telemetry_dim=12,
        max_seq_len=128,
    )

    # Prepare data
    logger.info("Preparing training corpus...")
    corpus = prepare_corpus(100000)
    dataset = CharDataset(corpus, config.max_seq_len)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = {
        'timestamp': timestamp,
        'device': str(device),
        'device_info': device_info,
        'config': asdict(config),
    }

    # ===== BASELINE =====
    logger.info("\n" + "="*40)
    logger.info("Training BASELINE model...")
    logger.info("="*40)

    baseline = BaselineTransformer(config).to(device)
    logger.info(f"Baseline params: {baseline.get_num_parameters():,}")

    baseline_train = train_model(
        baseline, train_loader, device,
        epochs=2, telemetry_reader=None, is_metabolic=False
    )

    logger.info("Evaluating BASELINE...")
    baseline_eval = evaluate_model(
        baseline, eval_loader, device,
        telemetry, actuator, energy_tracker,
        is_metabolic=False, num_samples=50
    )

    results['baseline'] = {
        'train': baseline_train,
        'eval': baseline_eval,
    }

    logger.info(f"Baseline PPL: {baseline_eval['perplexity']:.2f}")
    logger.info(f"Baseline J/token: {baseline_eval['j_per_token']*1000:.2f} mJ")

    # ===== METABOLIC =====
    logger.info("\n" + "="*40)
    logger.info("Training METABOLIC model...")
    logger.info("="*40)

    metabolic = MetabolicTransformer(config).to(device)
    logger.info(f"Metabolic params: {metabolic.get_num_parameters():,}")
    logger.info(f"  FiLM params: {sum(p.numel() for p in metabolic.get_film_parameters()):,}")
    logger.info(f"  Action head params: {sum(p.numel() for p in metabolic.get_action_head_parameters()):,}")

    metabolic_train = train_model(
        metabolic, train_loader, device,
        epochs=2, telemetry_reader=telemetry, is_metabolic=True
    )

    logger.info("Evaluating METABOLIC...")
    metabolic_eval = evaluate_model(
        metabolic, eval_loader, device,
        telemetry, actuator, energy_tracker,
        is_metabolic=True, num_samples=50
    )

    results['metabolic'] = {
        'train': metabolic_train,
        'eval': metabolic_eval,
    }

    logger.info(f"Metabolic PPL: {metabolic_eval['perplexity']:.2f}")
    logger.info(f"Metabolic J/token: {metabolic_eval['j_per_token']*1000:.2f} mJ")
    logger.info(f"Action distribution: {metabolic_eval['action_distribution']}")

    # ===== COMPARISON =====
    ppl_delta = (metabolic_eval['perplexity'] - baseline_eval['perplexity']) / baseline_eval['perplexity'] * 100
    energy_delta = (metabolic_eval['j_per_token'] - baseline_eval['j_per_token']) / max(baseline_eval['j_per_token'], 1e-9) * 100

    results['comparison'] = {
        'ppl_delta_percent': ppl_delta,
        'energy_delta_percent': energy_delta,
        'baseline_ppl': baseline_eval['perplexity'],
        'metabolic_ppl': metabolic_eval['perplexity'],
        'baseline_j_per_token': baseline_eval['j_per_token'],
        'metabolic_j_per_token': metabolic_eval['j_per_token'],
    }

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE: SHORT METABOLIC TRANSFORMER TEST")
    print("="*60)
    print(f"Device: {device} ({device_info.get('device_name', 'Unknown')})")
    print(f"\nBASELINE:")
    print(f"  Perplexity: {baseline_eval['perplexity']:.2f}")
    print(f"  J/token: {baseline_eval['j_per_token']*1000:.2f} mJ")
    print(f"  Tokens/sec: {baseline_eval['tokens_per_sec']:.1f}")
    print(f"\nMETABOLIC:")
    print(f"  Perplexity: {metabolic_eval['perplexity']:.2f}")
    print(f"  J/token: {metabolic_eval['j_per_token']*1000:.2f} mJ")
    print(f"  Tokens/sec: {metabolic_eval['tokens_per_sec']:.1f}")
    print(f"  Action distribution: {[f'{a:.1%}' for a in metabolic_eval['action_distribution']]}")
    print(f"\nCOMPARISON:")
    print(f"  PPL change: {ppl_delta:+.1f}%")
    print(f"  Energy change: {energy_delta:+.1f}%")
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = run_short_experiment()
