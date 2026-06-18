#!/usr/bin/env python3
"""
z104_ablation_study.py - FEEL-SLM Ablation Study

Runs the 8-configuration ablation matrix to isolate contributions:
1. Baseline (no body path)
2. +FiLM (body-conditioned normalization only)
3. +Gating (context-aware gate only)
4. +FiLM+Gating (both modulation mechanisms)
5. +Reporter (body state prediction head)
6. +Invariance (semantic stability loss)
7. +Policy (actuation head with bandit)
8. Full FEEL (all components)

Each configuration trains for N steps and measures:
- Validation loss (perplexity proxy)
- Energy per token (mJ/tok)
- SLO compliance
- Gate activation statistics
- FiLM modulation magnitude

This produces the ablation table for the paper.
"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import argparse

import torch
import torch.nn as nn

from src.feel_slm.config import FEELConfig
from src.feel_slm.model import FEELSLM, BaselineSLM
from src.feel_slm.trainer import FEELTrainer, TrainerConfig
from src.feel_slm.data import create_dataloaders
from src.feel_slm.benchmark import BenchmarkHarness, create_test_prompts


@dataclass
class AblationConfig:
    """Configuration for one ablation variant."""
    name: str
    use_film: bool = False
    use_gated_injection: bool = False
    use_reporter: bool = False
    use_invariance: bool = False
    use_policy: bool = False


# 8 ablation configurations
ABLATION_CONFIGS = [
    AblationConfig("baseline"),
    AblationConfig("+film", use_film=True),
    AblationConfig("+gating", use_gated_injection=True),
    AblationConfig("+film+gating", use_film=True, use_gated_injection=True),
    AblationConfig("+reporter", use_film=True, use_gated_injection=True, use_reporter=True),
    AblationConfig("+invariance", use_film=True, use_gated_injection=True, use_invariance=True),
    AblationConfig("+policy", use_film=True, use_gated_injection=True, use_policy=True),
    AblationConfig("full_feel", use_film=True, use_gated_injection=True,
                   use_reporter=True, use_invariance=True, use_policy=True),
]


@dataclass
class AblationResult:
    """Results from one ablation run."""
    name: str
    train_loss: float
    val_loss: float
    mj_per_token: float
    power_w: float
    latency_ms: float
    gate_openness: float  # Mean gate activation (if gating enabled)
    film_magnitude: float  # Mean FiLM gamma magnitude (if FiLM enabled)
    parameters: int
    training_time_s: float


def create_ablation_model(config: AblationConfig, model_config: FEELConfig) -> nn.Module:
    """Create model for ablation configuration."""

    if config.name == "baseline":
        return BaselineSLM(model_config)

    # Modify model config for ablation
    ablation_model_config = FEELConfig(
        # Copy base config
        vocab_size=model_config.vocab_size,
        hidden_dim=model_config.hidden_dim,
        intermediate_dim=model_config.intermediate_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout,
        # Ablation flags
        use_film=config.use_film,
        use_gated_injection=config.use_gated_injection,
    )

    return FEELSLM(ablation_model_config)


def measure_model_internals(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Measure internal model statistics for ablation analysis."""
    stats = {
        'gate_openness': 0.0,
        'film_magnitude': 0.0,
    }

    if not isinstance(model, FEELSLM):
        return stats

    # Get gate statistics
    with torch.no_grad():
        if hasattr(model.lm.layers[0], 'gated_injection') and model.lm.layers[0].gated_injection is not None:
            # Forward pass to get gate values
            input_ids = batch['input_ids']
            telemetry = batch['telemetry']
            body_embed = model.body_encoder(telemetry)

            # Get gate stats from first layer
            h = model.lm.tok_emb(input_ids)
            gate_stats = model.lm.layers[0].gated_injection.get_gate_stats(h, body_embed)
            stats['gate_openness'] = gate_stats['gate_mean']

        # Get FiLM magnitude
        if hasattr(model.lm.layers[0], 'film_attn') and model.lm.layers[0].film_attn is not None:
            telemetry = batch['telemetry']
            body_embed = model.body_encoder(telemetry)
            gamma, _ = model.lm.layers[0].film_attn(body_embed)
            stats['film_magnitude'] = gamma.abs().mean().item()

    return stats


def run_ablation_study(
    output_dir: str = "results/ablation",
    model_size: str = "tiny",
    train_steps: int = 1000,
    batch_size: int = 16,
    train_samples: int = 5000,
    val_samples: int = 500,
    daemon_url: Optional[str] = None,
    device: str = 'auto',
):
    """
    Run full ablation study.

    Args:
        output_dir: Where to save results
        model_size: tiny/small/medium
        train_steps: Steps per ablation
        batch_size: Training batch size
        train_samples: Training samples
        val_samples: Validation samples
        daemon_url: Daemon URL for energy measurement (optional)
        device: Device to use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Select device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"FEEL-SLM Ablation Study")
    print(f"=" * 60)
    print(f"Model size: {model_size}")
    print(f"Train steps: {train_steps}")
    print(f"Device: {device}")
    print(f"Daemon URL: {daemon_url or 'None (no energy measurement)'}")
    print()

    # Create base model config
    config_fn = {
        "tiny": FEELConfig.tiny,
        "small": FEELConfig.small,
        "medium": FEELConfig.medium,
    }.get(model_size, FEELConfig.tiny)
    model_config = config_fn()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        max_length=model_config.max_seq_len,
        train_samples=train_samples,
        val_samples=val_samples,
        num_workers=0,  # For stability
    )

    results: List[AblationResult] = []

    # Benchmark harness (if daemon available)
    benchmark = None
    if daemon_url:
        benchmark = BenchmarkHarness(daemon_url, device=device)

    # Run each ablation
    for ablation_config in ABLATION_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"ABLATION: {ablation_config.name}")
        print(f"{'=' * 60}")

        # Create model
        model = create_ablation_model(ablation_config, model_config)
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Create trainer config
        trainer_config = TrainerConfig(
            output_dir=str(output_path / ablation_config.name),
            max_steps=train_steps,
            batch_size=batch_size,
            eval_interval=train_steps // 5,
            save_interval=train_steps,
            log_interval=train_steps // 10,
            lm_weight=1.0,
            reporter_weight=0.1 if ablation_config.use_reporter else 0.0,
            policy_weight=0.01 if ablation_config.use_policy else 0.0,
            invariance_weight=0.05 if ablation_config.use_invariance else 0.0,
            use_amp=True,
        )

        # Train
        trainer = FEELTrainer(
            model=model,
            config=trainer_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        train_start = time.time()
        trainer.train()
        train_time = time.time() - train_start

        # Get final losses
        val_loss = trainer.evaluate()

        # Get model internals
        sample_batch = next(iter(train_loader))
        sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
        internal_stats = measure_model_internals(model, sample_batch)

        # Energy benchmark (if available)
        mj_per_token = 0.0
        power_w = 0.0
        latency_ms = 0.0

        if benchmark:
            print("\nRunning energy benchmark...")
            test_prompts = create_test_prompts(n=5, max_len=32, vocab_size=model_config.vocab_size)

            if ablation_config.name == "baseline":
                benchmark.baseline_model = model
                benchmark.feel_model = None
            else:
                benchmark.baseline_model = None
                benchmark.feel_model = model

            # Quick benchmark
            for prompt in test_prompts[:3]:
                telemetry = torch.rand(1, model_config.body_dim, device=device) if ablation_config.name != "baseline" else None
                condition = "baseline" if ablation_config.name == "baseline" else "feel"

                result = benchmark.benchmark_single(
                    model, condition, "balanced", prompt, max_tokens=50, telemetry=telemetry
                )
                mj_per_token += result.mj_per_token
                power_w += result.avg_power_w
                latency_ms += result.latency_ms

            mj_per_token /= 3
            power_w /= 3
            latency_ms /= 3

        # Store result
        result = AblationResult(
            name=ablation_config.name,
            train_loss=trainer.best_val_loss,
            val_loss=val_loss,
            mj_per_token=mj_per_token,
            power_w=power_w,
            latency_ms=latency_ms,
            gate_openness=internal_stats['gate_openness'],
            film_magnitude=internal_stats['film_magnitude'],
            parameters=n_params,
            training_time_s=train_time,
        )
        results.append(result)

        print(f"\nResult: val_loss={val_loss:.4f}, mJ/tok={mj_per_token:.1f}")

        # Clear GPU memory
        del model
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary table
    print("\n" + "=" * 100)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Name':<18} {'Params':>10} {'Val Loss':>10} {'mJ/tok':>10} {'Power(W)':>10} {'Gate':>8} {'FiLM':>8}")
    print("-" * 100)

    for r in results:
        print(f"{r.name:<18} {r.parameters:>10,} {r.val_loss:>10.4f} "
              f"{r.mj_per_token:>10.1f} {r.power_w:>10.0f} "
              f"{r.gate_openness:>8.3f} {r.film_magnitude:>8.4f}")

    # Save results
    results_file = output_path / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Generate LaTeX table
    latex_file = output_path / "ablation_table.tex"
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{FEEL-SLM Ablation Study}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Configuration & Params & Val Loss & mJ/tok & Gate & FiLM \\\\\n")
        f.write("\\midrule\n")
        for r in results:
            f.write(f"{r.name} & {r.parameters:,} & {r.val_loss:.4f} & "
                   f"{r.mj_per_token:.1f} & {r.gate_openness:.3f} & {r.film_magnitude:.4f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved to {latex_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEEL-SLM Ablation Study")
    parser.add_argument("--output-dir", default="results/z104_ablation")
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"], default="tiny")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--daemon-url", default=None, help="Daemon URL for energy measurement")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_ablation_study(
        output_dir=args.output_dir,
        model_size=args.model_size,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        daemon_url=args.daemon_url,
        device=args.device,
    )
