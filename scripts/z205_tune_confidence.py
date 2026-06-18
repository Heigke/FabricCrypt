#!/usr/bin/env python3
"""
Tune Confidence-Based Early Exit

The z204 distillation showed that confidence thresholds 0.7-0.9 always
fall back to full model (avg exit layer = 12.0). This script:

1. Analyzes the confidence distribution from trained model
2. Finds optimal thresholds for different quality-energy tradeoffs
3. Validates with real energy measurement

The key insight: we need to understand what confidence values the model
actually produces to set meaningful thresholds.
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from src.telemetry.real_amd import AMDTelemetry, RocmSmiReader

# Import the distillation model
from scripts.z204_distill_exit_heads import DistillableEarlyExitGPT2


@dataclass
class ConfidenceStats:
    """Statistics for confidence at one exit layer."""
    layer: int
    mean: float
    std: float
    min: float
    max: float
    percentiles: Dict[str, float]  # 10, 25, 50, 75, 90


@dataclass
class ThresholdResult:
    """Result for one confidence threshold."""
    threshold: float
    avg_exit_layer: float
    exit_distribution: Dict[int, float]
    quality_loss: float
    quality_ratio: float
    energy_per_token_mj: float
    energy_savings_pct: float
    throughput_tok_s: float


def analyze_confidence_distribution(
    model: DistillableEarlyExitGPT2,
    batches: List[torch.Tensor],
    device: torch.device
) -> Dict[int, ConfidenceStats]:
    """Analyze what confidence values the model produces."""

    model.eval()
    confidence_values = {3: [], 6: [], 9: []}

    with torch.no_grad():
        for batch in tqdm(batches, desc="Analyzing confidence"):
            outputs = model.forward_all_exits(batch)

            for layer in [3, 6, 9]:
                _, confidence, _ = outputs[layer]
                confidence_values[layer].extend(confidence.cpu().numpy().flatten().tolist())

    stats = {}
    for layer in [3, 6, 9]:
        values = np.array(confidence_values[layer])
        stats[layer] = ConfidenceStats(
            layer=layer,
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            percentiles={
                '10': float(np.percentile(values, 10)),
                '25': float(np.percentile(values, 25)),
                '50': float(np.percentile(values, 50)),
                '75': float(np.percentile(values, 75)),
                '90': float(np.percentile(values, 90)),
            }
        )

    return stats


def evaluate_threshold(
    model: DistillableEarlyExitGPT2,
    batches: List[torch.Tensor],
    threshold: float,
    baseline_loss: float,
    baseline_energy: float,
    device: torch.device,
    num_iterations: int = 5
) -> ThresholdResult:
    """Evaluate model performance at a specific threshold."""

    model.eval()

    # Count tokens
    total_tokens = sum(
        (batch != 50256).sum().item()
        for batch in batches
    ) * num_iterations

    # Track exits and measure
    exit_counts = {3: 0, 6: 0, 9: 0, 12: 0}
    power_samples = []

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        for batch in batches:
            with torch.no_grad():
                _, exit_layer, _ = model.forward_with_confidence_exit(batch, threshold)
            exit_counts[exit_layer] += batch.size(0)

            power = RocmSmiReader.read_power()
            if power:
                power_samples.append(power)

    torch.cuda.synchronize()
    end_time = time.time()

    duration_ms = (end_time - start_time) * 1000
    avg_power = np.mean(power_samples) if power_samples else 0
    total_energy_mj = avg_power * (duration_ms / 1000) * 1000

    # Calculate average exit layer
    total_exits = sum(exit_counts.values())
    avg_exit_layer = sum(layer * count for layer, count in exit_counts.items()) / total_exits

    exit_distribution = {layer: count / total_exits for layer, count in exit_counts.items()}

    # Quality measurement
    with torch.no_grad():
        test_logits, _, _ = model.forward_with_confidence_exit(batches[0], threshold)
        labels = batches[0].clone()
        shift_logits = test_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, model.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        ).item()

    energy_per_token = total_energy_mj / total_tokens if total_tokens > 0 else 0

    return ThresholdResult(
        threshold=threshold,
        avg_exit_layer=avg_exit_layer,
        exit_distribution=exit_distribution,
        quality_loss=loss,
        quality_ratio=loss / baseline_loss if baseline_loss > 0 else 0,
        energy_per_token_mj=energy_per_token,
        energy_savings_pct=(1 - energy_per_token / baseline_energy) * 100 if baseline_energy > 0 else 0,
        throughput_tok_s=total_tokens / (duration_ms / 1000) if duration_ms > 0 else 0
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/z204_distilled/model_final.pt")
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="results/z205_confidence_tuning.json")
    args = parser.parse_args()

    print("="*60)
    print("Confidence-Based Early Exit Tuning")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize telemetry
    print("\nInitializing hardware telemetry...")
    telemetry = AMDTelemetry()
    initial = telemetry.read()
    print(f"  Power: {initial.power_w:.1f}W, Temp: {initial.temp_c:.0f}°C")

    # Load model
    print("\nLoading distilled model...")
    model = DistillableEarlyExitGPT2("gpt2").to(device)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"  Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"  Warning: No checkpoint found at {checkpoint_path}, using untrained model")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load validation data
    print(f"\nLoading validation data...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    texts = []
    for item in dataset:
        if len(texts) >= args.num_batches * args.batch_size:
            break
        text = item['text'][:500]
        if len(text) > 50:
            texts.append(text)

    batches = []
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i+args.batch_size]
        if len(batch_texts) == args.batch_size:
            encoded = tokenizer(
                batch_texts,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            batches.append(encoded['input_ids'].to(device))

    print(f"  Prepared {len(batches)} batches")

    # ========== ANALYZE CONFIDENCE DISTRIBUTION ==========
    print("\n" + "="*60)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("="*60)

    stats = analyze_confidence_distribution(model, batches, device)

    for layer in [3, 6, 9]:
        s = stats[layer]
        print(f"\n  Layer {layer}:")
        print(f"    Mean: {s.mean:.4f} ± {s.std:.4f}")
        print(f"    Range: [{s.min:.4f}, {s.max:.4f}]")
        print(f"    Percentiles: 10%={s.percentiles['10']:.3f}, "
              f"25%={s.percentiles['25']:.3f}, "
              f"50%={s.percentiles['50']:.3f}, "
              f"75%={s.percentiles['75']:.3f}, "
              f"90%={s.percentiles['90']:.3f}")

    # ========== BASELINE MEASUREMENT ==========
    print("\n" + "="*60)
    print("BASELINE MEASUREMENT (full model)")
    print("="*60)

    # Measure baseline
    baseline_result = evaluate_threshold(model, batches, 1.0, 1.0, 1.0, device)
    baseline_loss = baseline_result.quality_loss
    baseline_energy = baseline_result.energy_per_token_mj

    print(f"  Loss: {baseline_loss:.3f}")
    print(f"  Energy: {baseline_energy:.3f} mJ/token")
    print(f"  Throughput: {baseline_result.throughput_tok_s:.0f} tok/s")

    # ========== THRESHOLD SWEEP ==========
    print("\n" + "="*60)
    print("THRESHOLD SWEEP")
    print("="*60)

    # Calculate thresholds based on actual confidence distribution
    # Use percentiles from layer 9 (most likely to exit early)
    layer9_stats = stats[9]

    # Test a range of thresholds from 10th percentile to 90th percentile
    thresholds = [
        0.0,  # Always exit at first confident layer
        layer9_stats.percentiles['10'],
        layer9_stats.percentiles['25'],
        layer9_stats.percentiles['50'],
        layer9_stats.percentiles['75'],
        layer9_stats.percentiles['90'],
        layer9_stats.max + 0.01,  # Never exit early (baseline)
    ]

    # Also add some fixed thresholds
    for fixed in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if fixed not in thresholds:
            thresholds.append(fixed)

    thresholds = sorted(set(thresholds))

    results = {}

    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        result = evaluate_threshold(
            model, batches, threshold,
            baseline_loss, baseline_energy, device
        )
        results[threshold] = result

        print(f"\n  Threshold {threshold:.3f}:")
        print(f"    Avg exit layer: {result.avg_exit_layer:.2f}")
        print(f"    Exit distribution: " +
              ", ".join(f"L{l}:{p:.0%}" for l, p in sorted(result.exit_distribution.items())))
        print(f"    Quality: {result.quality_loss:.3f} ({result.quality_ratio:.2f}x baseline)")
        print(f"    Energy: {result.energy_per_token_mj:.3f} mJ/tok ({result.energy_savings_pct:+.1f}%)")
        print(f"    Throughput: {result.throughput_tok_s:.0f} tok/s")

    # ========== PARETO ANALYSIS ==========
    print("\n" + "="*60)
    print("PARETO-OPTIMAL CONFIGURATIONS")
    print("="*60)

    # Find Pareto-optimal points (best quality for each energy level)
    pareto_points = []
    for threshold, result in sorted(results.items()):
        is_dominated = False
        for other_threshold, other_result in results.items():
            if other_threshold == threshold:
                continue
            # Check if other dominates this point
            if (other_result.quality_loss <= result.quality_loss and
                other_result.energy_savings_pct >= result.energy_savings_pct and
                (other_result.quality_loss < result.quality_loss or
                 other_result.energy_savings_pct > result.energy_savings_pct)):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append((threshold, result))

    print("\nPareto-optimal thresholds:")
    for threshold, result in sorted(pareto_points, key=lambda x: x[1].energy_savings_pct, reverse=True):
        print(f"\n  Threshold {threshold:.3f}:")
        print(f"    Avg exit: {result.avg_exit_layer:.2f}")
        print(f"    Quality ratio: {result.quality_ratio:.2f}x")
        print(f"    Energy savings: {result.energy_savings_pct:.1f}%")
        print(f"    Throughput: {result.throughput_tok_s:.0f} tok/s")

    # ========== RECOMMENDATIONS ==========
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    # Find best threshold for different use cases
    best_balanced = min(results.items(), key=lambda x: x[1].quality_ratio + (1 - x[1].energy_savings_pct/100))
    best_quality = min(results.items(), key=lambda x: x[1].quality_loss)
    best_efficiency = max(results.items(), key=lambda x: x[1].energy_savings_pct)

    print(f"\n  Best Balanced (quality + efficiency):")
    print(f"    Threshold: {best_balanced[0]:.3f}")
    print(f"    Quality: {best_balanced[1].quality_ratio:.2f}x, Energy: {best_balanced[1].energy_savings_pct:.1f}%")

    print(f"\n  Best Quality:")
    print(f"    Threshold: {best_quality[0]:.3f}")
    print(f"    Quality: {best_quality[1].quality_ratio:.2f}x, Energy: {best_quality[1].energy_savings_pct:.1f}%")

    print(f"\n  Best Efficiency:")
    print(f"    Threshold: {best_efficiency[0]:.3f}")
    print(f"    Quality: {best_efficiency[1].quality_ratio:.2f}x, Energy: {best_efficiency[1].energy_savings_pct:.1f}%")

    # ========== SAVE RESULTS ==========
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'confidence_stats': {str(k): asdict(v) for k, v in stats.items()},
        'baseline': {
            'loss': baseline_loss,
            'energy_mj_per_token': baseline_energy,
            'throughput_tok_s': baseline_result.throughput_tok_s
        },
        'threshold_results': {str(k): asdict(v) for k, v in results.items()},
        'pareto_optimal': [
            {'threshold': t, **asdict(r)} for t, r in pareto_points
        ],
        'recommendations': {
            'balanced': {'threshold': best_balanced[0], **asdict(best_balanced[1])},
            'quality': {'threshold': best_quality[0], **asdict(best_quality[1])},
            'efficiency': {'threshold': best_efficiency[0], **asdict(best_efficiency[1])}
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Cleanup
    telemetry.shutdown()

    print("\n" + "="*60)
    print("Confidence tuning complete!")
    print("="*60)


if __name__ == "__main__":
    main()
