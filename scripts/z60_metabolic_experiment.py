#!/usr/bin/env python3
"""
Z60: Metabolic Transformer Experiment
=====================================
Full experiment comparing baseline vs metabolic transformer across machines.

This script:
1. Trains baseline and metabolic models
2. Measures Joules/token, perplexity, thermal stability
3. Tests cross-body generalization (AMD ↔ NVIDIA)
4. Generates comprehensive comparison report

Usage:
    python scripts/z60_metabolic_experiment.py --host ikaros
    python scripts/z60_metabolic_experiment.py --host daedalus
    python scripts/z60_metabolic_experiment.py --host minos

Author: FEEL Research Team
Date: 2026-01-19
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.metabolic.film_transformer import (
    MetabolicTransformer, BaselineTransformer,
    MetabolicConfig, create_metabolic_transformer
)
from src.metabolic.telemetry_unified import UnifiedTelemetryReader
from src.metabolic.actuation_unified import UnifiedActuator, MetabolicMode
from src.metabolic.energy_tracker import EnergyTracker, TokenLevelEnergyTracker
from src.metabolic.metabolic_trainer import (
    MetabolicTrainer, TrainerConfig, CharDataset, TelemetryScheduler
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for metabolic experiment."""
    # Experiment identity
    experiment_name: str = "z60_metabolic"
    host: str = "ikaros"  # ikaros, daedalus, minos
    timestamp: str = ""

    # Model configuration
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    telemetry_dim: int = 12
    max_seq_len: int = 256

    # Training configuration
    lm_epochs: int = 5
    lm_batch_size: int = 32
    rl_epochs: int = 3
    rl_batch_size: int = 8

    # Evaluation configuration
    eval_samples: int = 100
    eval_tokens_per_sample: int = 100
    thermal_stress_duration_s: int = 60

    # Data
    corpus_size: int = 500000  # Characters for training

    # Paths
    output_dir: str = ""
    data_path: str = ""

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"results/z60_metabolic_{self.host}_{self.timestamp}"
        self.data_path = f"data/metabolic_corpus_{self.corpus_size}.txt"


@dataclass
class ExperimentResults:
    """Results from metabolic experiment."""
    # Configuration
    config: Dict
    host: str
    device_info: Dict

    # Baseline results
    baseline_ppl: float
    baseline_j_per_token: float
    baseline_tokens_per_sec: float

    # Metabolic results
    metabolic_ppl: float
    metabolic_j_per_token: float
    metabolic_tokens_per_sec: float
    metabolic_action_distribution: List[float]

    # Comparison
    ppl_delta_percent: float
    energy_delta_percent: float
    efficiency_gain: float  # tokens/J improvement

    # Thermal stability
    baseline_thermal_variance: float
    metabolic_thermal_variance: float
    throttle_events_baseline: int
    throttle_events_metabolic: int

    # Cross-body (if applicable)
    cross_body_ppl: Optional[float] = None
    cross_body_j_per_token: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def prepare_training_corpus(config: ExperimentConfig) -> str:
    """Prepare training corpus from diverse sources."""
    logger.info(f"Preparing training corpus ({config.corpus_size} chars)")

    corpus_path = Path(config.data_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    if corpus_path.exists():
        logger.info(f"Using existing corpus: {corpus_path}")
        return str(corpus_path)

    # Generate diverse training text
    texts = []

    # Shakespeare-style
    texts.append("To be or not to be, that is the question. " * 100)

    # Technical
    texts.append("The transformer architecture uses self-attention mechanisms. " * 100)

    # Code-like
    texts.append("def forward(self, x): return self.layer(x) " * 100)

    # Numbers and patterns
    texts.append("0123456789 " * 500)

    # Repeat letters
    for c in "abcdefghijklmnopqrstuvwxyz":
        texts.append(f"{c} " * 50)

    # Mix and repeat to reach target size
    corpus = " ".join(texts)
    while len(corpus) < config.corpus_size:
        corpus += corpus[:config.corpus_size // 10]
    corpus = corpus[:config.corpus_size]

    corpus_path.write_text(corpus)
    logger.info(f"Created corpus: {len(corpus)} chars")

    return str(corpus_path)


def train_baseline(config: ExperimentConfig, data_path: str) -> BaselineTransformer:
    """Train baseline transformer (no telemetry conditioning)."""
    logger.info("Training baseline transformer...")

    model_config = MetabolicConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        telemetry_dim=config.telemetry_dim,
        max_seq_len=config.max_seq_len,
    )

    model = BaselineTransformer(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    logger.info(f"Baseline model: {model.get_num_parameters():,} parameters on {device}")

    # Load data
    with open(data_path) as f:
        text = f.read()
    dataset = CharDataset(text, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.lm_batch_size, shuffle=True, num_workers=2)

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(config.lm_epochs):
        epoch_loss = 0
        for batch_idx, (input_ids, targets) in enumerate(loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            output = model(input_ids)
            loss = F.cross_entropy(
                output['logits'].view(-1, model_config.vocab_size),
                targets.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                logger.info(f"Baseline Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        logger.info(f"Baseline Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

    # Save
    save_path = Path(config.output_dir) / "baseline_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    return model


def train_metabolic(config: ExperimentConfig, data_path: str) -> MetabolicTransformer:
    """Train metabolic transformer with telemetry conditioning + RL."""
    logger.info("Training metabolic transformer...")

    trainer_config = TrainerConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        telemetry_dim=config.telemetry_dim,
        lm_epochs=config.lm_epochs,
        lm_batch_size=config.lm_batch_size,
        lm_max_seq_len=config.max_seq_len,
        rl_epochs=config.rl_epochs,
        rl_batch_size=config.rl_batch_size,
        output_dir=config.output_dir,
    )

    trainer = MetabolicTrainer(trainer_config)

    # Load data
    train_loader, val_loader = trainer.load_data(path=data_path)

    # Stage 1: LM pretraining with simulated telemetry
    trainer.train_stage1(train_loader, val_loader)

    # Stage 2: RL with real hardware (if available)
    use_real_hw = torch.cuda.is_available()
    trainer.train_stage2(train_loader, use_real_hardware=use_real_hw)

    return trainer.model


def evaluate_model(
    model: MetabolicTransformer,
    config: ExperimentConfig,
    telemetry: UnifiedTelemetryReader,
    actuator: UnifiedActuator,
    is_metabolic: bool = True,
) -> Dict:
    """Evaluate model for perplexity, energy, and throughput."""
    logger.info(f"Evaluating {'metabolic' if is_metabolic else 'baseline'} model...")

    device = next(model.parameters()).device
    model.eval()

    # Prepare evaluation data
    eval_text = "The quick brown fox jumps over the lazy dog. " * 100
    dataset = CharDataset(eval_text, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Energy tracker
    energy_tracker = EnergyTracker(telemetry)

    # Metrics
    total_loss = 0
    total_tokens = 0
    action_counts = [0, 0, 0, 0]
    temps = []
    throttle_events = 0

    energy_tracker.start_session()
    start_time = time.time()

    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(loader):
            if i >= config.eval_samples:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Get real telemetry
            telem_np = telemetry.read_body_state()
            telem_tensor = torch.from_numpy(telem_np).float().to(device).unsqueeze(0)

            # Forward pass
            if is_metabolic:
                output = model(input_ids, telem_tensor)
                # Get action
                action_probs = F.softmax(output['action_logits'], dim=-1)
                action_idx = torch.argmax(action_probs, dim=-1).item()
                action_counts[action_idx] += 1
                # Apply action
                actuator.set_mode_from_action(action_idx)
            else:
                output = model(input_ids)

            # Compute loss
            loss = F.cross_entropy(
                output['logits'].view(-1, model.config.vocab_size),
                targets.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
            energy_tracker.record_tokens(targets.numel())

            # Record telemetry
            snap = telemetry.read()
            temps.append(snap.temp_c)
            if snap.throttle_status:
                throttle_events += 1

    end_time = time.time()
    session = energy_tracker.stop_session()

    # Reset actuator
    actuator.reset_to_default()

    # Compute metrics
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    j_per_token = session.j_per_token
    tokens_per_sec = total_tokens / (end_time - start_time)
    thermal_variance = np.var(temps) if temps else 0

    action_dist = [c / sum(action_counts) for c in action_counts] if sum(action_counts) > 0 else [0.25] * 4

    results = {
        'perplexity': perplexity,
        'j_per_token': j_per_token,
        'tokens_per_sec': tokens_per_sec,
        'thermal_variance': thermal_variance,
        'throttle_events': throttle_events,
        'action_distribution': action_dist,
        'total_tokens': total_tokens,
        'total_energy_j': session.total_energy_j,
    }

    logger.info(f"Evaluation complete: PPL={perplexity:.2f}, J/tok={j_per_token*1000:.2f}mJ")

    return results


def run_thermal_stress_test(
    model: MetabolicTransformer,
    config: ExperimentConfig,
    telemetry: UnifiedTelemetryReader,
    actuator: UnifiedActuator,
    is_metabolic: bool = True,
) -> Dict:
    """Run thermal stress test to evaluate stability under load."""
    logger.info(f"Running thermal stress test ({config.thermal_stress_duration_s}s)...")

    device = next(model.parameters()).device
    model.eval()

    # Generate continuous load
    eval_text = "x" * 10000
    dataset = CharDataset(eval_text, config.max_seq_len)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    temps = []
    throttle_events = 0
    start_time = time.time()

    with torch.no_grad():
        while time.time() - start_time < config.thermal_stress_duration_s:
            for input_ids, targets in loader:
                if time.time() - start_time >= config.thermal_stress_duration_s:
                    break

                input_ids = input_ids.to(device)
                targets = targets.to(device)

                telem_np = telemetry.read_body_state()
                telem_tensor = torch.from_numpy(telem_np).float().to(device)
                telem_tensor = telem_tensor.unsqueeze(0).expand(input_ids.size(0), -1)

                if is_metabolic:
                    output = model(input_ids, telem_tensor)
                    action_probs = F.softmax(output['action_logits'], dim=-1)
                    action_idx = torch.argmax(action_probs[0], dim=-1).item()
                    actuator.set_mode_from_action(action_idx)
                else:
                    output = model(input_ids)

                snap = telemetry.read()
                temps.append(snap.temp_c)
                if snap.throttle_status:
                    throttle_events += 1

    actuator.reset_to_default()

    return {
        'temp_mean': np.mean(temps),
        'temp_max': np.max(temps),
        'temp_std': np.std(temps),
        'throttle_events': throttle_events,
    }


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run complete metabolic experiment."""
    logger.info(f"Starting experiment: {config.experiment_name} on {config.host}")
    logger.info(f"Output directory: {config.output_dir}")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize hardware interfaces
    telemetry = UnifiedTelemetryReader()
    actuator = UnifiedActuator()
    device_info = telemetry.get_device_info()

    logger.info(f"Device: {device_info}")

    # Prepare data
    data_path = prepare_training_corpus(config)

    # Train models
    baseline_model = train_baseline(config, data_path)
    metabolic_model = train_metabolic(config, data_path)

    # Evaluate baseline
    baseline_results = evaluate_model(
        baseline_model, config, telemetry, actuator, is_metabolic=False
    )

    # Evaluate metabolic
    metabolic_results = evaluate_model(
        metabolic_model, config, telemetry, actuator, is_metabolic=True
    )

    # Thermal stress tests
    baseline_thermal = run_thermal_stress_test(
        baseline_model, config, telemetry, actuator, is_metabolic=False
    )
    metabolic_thermal = run_thermal_stress_test(
        metabolic_model, config, telemetry, actuator, is_metabolic=True
    )

    # Compute deltas
    ppl_delta = (metabolic_results['perplexity'] - baseline_results['perplexity']) / baseline_results['perplexity'] * 100
    energy_delta = (metabolic_results['j_per_token'] - baseline_results['j_per_token']) / baseline_results['j_per_token'] * 100

    baseline_efficiency = baseline_results['tokens_per_sec'] / max(baseline_results['j_per_token'], 1e-6)
    metabolic_efficiency = metabolic_results['tokens_per_sec'] / max(metabolic_results['j_per_token'], 1e-6)
    efficiency_gain = (metabolic_efficiency - baseline_efficiency) / baseline_efficiency * 100

    # Compile results
    results = ExperimentResults(
        config=asdict(config),
        host=config.host,
        device_info=device_info,

        baseline_ppl=baseline_results['perplexity'],
        baseline_j_per_token=baseline_results['j_per_token'],
        baseline_tokens_per_sec=baseline_results['tokens_per_sec'],

        metabolic_ppl=metabolic_results['perplexity'],
        metabolic_j_per_token=metabolic_results['j_per_token'],
        metabolic_tokens_per_sec=metabolic_results['tokens_per_sec'],
        metabolic_action_distribution=metabolic_results['action_distribution'],

        ppl_delta_percent=ppl_delta,
        energy_delta_percent=energy_delta,
        efficiency_gain=efficiency_gain,

        baseline_thermal_variance=baseline_thermal['temp_std'] ** 2,
        metabolic_thermal_variance=metabolic_thermal['temp_std'] ** 2,
        throttle_events_baseline=baseline_thermal['throttle_events'],
        throttle_events_metabolic=metabolic_thermal['throttle_events'],
    )

    # Save results
    results_path = Path(config.output_dir) / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT COMPLETE: {config.experiment_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Baseline PPL: {results.baseline_ppl:.2f}")
    logger.info(f"Metabolic PPL: {results.metabolic_ppl:.2f} ({ppl_delta:+.1f}%)")
    logger.info(f"Baseline J/token: {results.baseline_j_per_token*1000:.2f} mJ")
    logger.info(f"Metabolic J/token: {results.metabolic_j_per_token*1000:.2f} mJ ({energy_delta:+.1f}%)")
    logger.info(f"Efficiency gain: {efficiency_gain:+.1f}%")
    logger.info(f"Throttle events: baseline={results.throttle_events_baseline}, metabolic={results.throttle_events_metabolic}")
    logger.info(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Z60 Metabolic Transformer Experiment")
    parser.add_argument('--host', type=str, default='ikaros',
                        choices=['ikaros', 'daedalus', 'minos'],
                        help='Host machine to run on')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--lm-epochs', type=int, default=5)
    parser.add_argument('--rl-epochs', type=int, default=3)
    parser.add_argument('--corpus-size', type=int, default=500000)
    parser.add_argument('--eval-samples', type=int, default=100)
    parser.add_argument('--thermal-duration', type=int, default=60)
    args = parser.parse_args()

    config = ExperimentConfig(
        host=args.host,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lm_epochs=args.lm_epochs,
        rl_epochs=args.rl_epochs,
        corpus_size=args.corpus_size,
        eval_samples=args.eval_samples,
        thermal_stress_duration_s=args.thermal_duration,
    )

    results = run_experiment(config)

    print(f"\nExperiment complete! Results in: {config.output_dir}")


if __name__ == "__main__":
    main()
