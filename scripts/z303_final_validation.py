#!/usr/bin/env python3
"""
Final Validation - Compare All Optimizations with Distilled Models

This uses the TRAINED distilled exit heads from z204 for fair comparison.
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset

from src.telemetry.real_amd import AMDTelemetry, RocmSmiReader

# Import the distilled model architecture
from scripts.z204_distill_exit_heads import DistillableEarlyExitGPT2


@dataclass
class ValidationResult:
    name: str
    description: str
    total_tokens: int
    total_time_ms: float
    tokens_per_second: float
    energy_per_token_mj: float
    avg_power_w: float
    avg_loss: float
    avg_temp_c: float
    energy_savings_pct: float = 0.0
    throughput_gain_pct: float = 0.0
    quality_ratio: float = 1.0


def run_validation(
    model,
    batches,
    telemetry,
    device,
    forward_fn,
    num_iterations=10,
    warmup=3,
    name="test",
    description=""
):
    """Run validation with a custom forward function."""

    model.eval()

    # Warmup
    for _ in range(warmup):
        for batch in batches[:2]:
            with torch.no_grad():
                forward_fn(model, batch)
            torch.cuda.synchronize()

    # Count tokens
    total_tokens = sum((b != 50256).sum().item() for b in batches) * num_iterations

    # Measure
    power_samples = []
    temp_samples = []
    losses = []

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iterations):
        for batch in batches:
            with torch.no_grad():
                logits, loss = forward_fn(model, batch)
                if loss is not None:
                    losses.append(loss.item() if hasattr(loss, 'item') else loss)

            power = RocmSmiReader.read_power()
            if power:
                power_samples.append(power)
            temp_samples.append(telemetry.read().temp_c)

    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000

    avg_power = np.mean(power_samples) if power_samples else 0
    energy_mj = avg_power * (elapsed_ms / 1000) * 1000

    return ValidationResult(
        name=name,
        description=description,
        total_tokens=total_tokens,
        total_time_ms=elapsed_ms,
        tokens_per_second=total_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        energy_per_token_mj=energy_mj / total_tokens if total_tokens > 0 else 0,
        avg_power_w=avg_power,
        avg_loss=np.mean(losses) if losses else 0,
        avg_temp_c=np.mean(temp_samples) if temp_samples else 0,
    )


def main():
    print("=" * 70)
    print("FINAL VALIDATION - With Distilled Exit Heads")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    telemetry = AMDTelemetry()
    initial = telemetry.read()
    print(f"Initial: Power={initial.power_w:.1f}W, Temp={initial.temp_c:.0f}°C")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("\nLoading validation data...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    texts = []
    for item in dataset:
        if len(texts) >= 200:
            break
        text = item['text'][:500]
        if len(text) > 50:
            texts.append(text)

    batches = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        if len(batch_texts) == batch_size:
            encoded = tokenizer(
                batch_texts,
                max_length=128,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            batches.append(encoded['input_ids'].to(device))

    print(f"Prepared {len(batches)} batches")

    results = {}

    # 1. Baseline - Standard GPT-2
    print("\n" + "=" * 70)
    print("1. BASELINE - Standard GPT-2 (Full Model)")
    print("=" * 70)

    baseline_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    baseline_model.eval()

    def baseline_forward(model, batch):
        labels = batch.clone()
        out = model(batch, labels=labels)
        return out.logits, out.loss

    results['baseline'] = run_validation(
        baseline_model, batches, telemetry, device,
        baseline_forward, num_iterations=10,
        name="baseline", description="Standard GPT-2"
    )
    print(f"  Tokens/sec: {results['baseline'].tokens_per_second:.0f}")
    print(f"  Energy: {results['baseline'].energy_per_token_mj:.3f} mJ/tok")
    print(f"  Power: {results['baseline'].avg_power_w:.1f}W")
    print(f"  Loss: {results['baseline'].avg_loss:.3f}")

    del baseline_model
    torch.cuda.empty_cache()
    time.sleep(2)

    # 2. Load Distilled Model
    print("\n" + "=" * 70)
    print("2. Loading DISTILLED exit heads model...")
    print("=" * 70)

    distilled_model = DistillableEarlyExitGPT2("gpt2").to(device)

    # Load trained weights
    checkpoint_path = Path("checkpoints/z204_distilled/model_final.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        distilled_model.load_state_dict(checkpoint['model_state'])
        print(f"  Loaded distilled weights from {checkpoint_path}")
    else:
        print(f"  Warning: No distilled checkpoint found, using untrained exit heads")

    distilled_model.eval()

    # 3. Distilled Exit - Layer 9
    print("\n" + "=" * 70)
    print("3. DISTILLED EXIT - Layer 9 (75% compute)")
    print("=" * 70)

    def distilled_forward_9(model, batch):
        labels = batch.clone()
        logits, _ = model.forward_to_layer(batch, exit_layer=9)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, model.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return logits, loss.item()

    results['distilled_9'] = run_validation(
        distilled_model, batches, telemetry, device,
        distilled_forward_9, num_iterations=10,
        name="distilled_exit_9", description="Distilled exit at layer 9"
    )
    print(f"  Tokens/sec: {results['distilled_9'].tokens_per_second:.0f}")
    print(f"  Energy: {results['distilled_9'].energy_per_token_mj:.3f} mJ/tok")
    print(f"  Power: {results['distilled_9'].avg_power_w:.1f}W")
    print(f"  Loss: {results['distilled_9'].avg_loss:.3f}")

    time.sleep(2)

    # 4. Distilled Exit - Layer 6
    print("\n" + "=" * 70)
    print("4. DISTILLED EXIT - Layer 6 (50% compute)")
    print("=" * 70)

    def distilled_forward_6(model, batch):
        labels = batch.clone()
        logits, _ = model.forward_to_layer(batch, exit_layer=6)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, model.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return logits, loss.item()

    results['distilled_6'] = run_validation(
        distilled_model, batches, telemetry, device,
        distilled_forward_6, num_iterations=10,
        name="distilled_exit_6", description="Distilled exit at layer 6"
    )
    print(f"  Tokens/sec: {results['distilled_6'].tokens_per_second:.0f}")
    print(f"  Energy: {results['distilled_6'].energy_per_token_mj:.3f} mJ/tok")
    print(f"  Power: {results['distilled_6'].avg_power_w:.1f}W")
    print(f"  Loss: {results['distilled_6'].avg_loss:.3f}")

    time.sleep(2)

    # 5. Distilled Exit - Layer 3
    print("\n" + "=" * 70)
    print("5. DISTILLED EXIT - Layer 3 (25% compute)")
    print("=" * 70)

    def distilled_forward_3(model, batch):
        labels = batch.clone()
        logits, _ = model.forward_to_layer(batch, exit_layer=3)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, model.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return logits, loss.item()

    results['distilled_3'] = run_validation(
        distilled_model, batches, telemetry, device,
        distilled_forward_3, num_iterations=10,
        name="distilled_exit_3", description="Distilled exit at layer 3"
    )
    print(f"  Tokens/sec: {results['distilled_3'].tokens_per_second:.0f}")
    print(f"  Energy: {results['distilled_3'].energy_per_token_mj:.3f} mJ/tok")
    print(f"  Power: {results['distilled_3'].avg_power_w:.1f}W")
    print(f"  Loss: {results['distilled_3'].avg_loss:.3f}")

    time.sleep(2)

    # 6. Confidence-Based Exit (threshold 0.5)
    print("\n" + "=" * 70)
    print("6. CONFIDENCE-BASED EXIT (threshold=0.5)")
    print("=" * 70)

    exit_layers_used = []

    def confidence_forward(model, batch):
        labels = batch.clone()
        logits, exit_layer, _ = model.forward_with_confidence_exit(batch, confidence_threshold=0.5)
        exit_layers_used.append(exit_layer)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, model.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return logits, loss.item()

    results['confidence_05'] = run_validation(
        distilled_model, batches, telemetry, device,
        confidence_forward, num_iterations=10,
        name="confidence_0.5", description="Confidence-based exit (threshold=0.5)"
    )
    avg_exit = np.mean(exit_layers_used)
    print(f"  Avg Exit Layer: {avg_exit:.1f}")
    print(f"  Tokens/sec: {results['confidence_05'].tokens_per_second:.0f}")
    print(f"  Energy: {results['confidence_05'].energy_per_token_mj:.3f} mJ/tok")
    print(f"  Power: {results['confidence_05'].avg_power_w:.1f}W")
    print(f"  Loss: {results['confidence_05'].avg_loss:.3f}")

    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    baseline = results['baseline']

    # Calculate relative metrics
    for name, result in results.items():
        if name != 'baseline':
            result.energy_savings_pct = (1 - result.energy_per_token_mj / baseline.energy_per_token_mj) * 100
            result.throughput_gain_pct = (result.tokens_per_second / baseline.tokens_per_second - 1) * 100
            result.quality_ratio = result.avg_loss / baseline.avg_loss

    print(f"\nBaseline: {baseline.tokens_per_second:.0f} tok/s, {baseline.energy_per_token_mj:.3f} mJ/tok, loss={baseline.avg_loss:.3f}")

    print("\n" + "-" * 70)
    print(f"{'Configuration':<20} {'Throughput':>10} {'Energy':>10} {'Quality':>10} {'Savings':>10}")
    print(f"{'':<20} {'(tok/s)':>10} {'(mJ/tok)':>10} {'(ratio)':>10} {'(%)':>10}")
    print("-" * 70)

    for name in ['baseline', 'distilled_9', 'distilled_6', 'distilled_3', 'confidence_05']:
        if name not in results:
            continue
        result = results[name]
        savings = f"{result.energy_savings_pct:+.1f}%" if name != 'baseline' else "baseline"
        quality = f"{result.quality_ratio:.2f}x" if name != 'baseline' else "1.00x"
        print(f"{result.name:<20} {result.tokens_per_second:>10.0f} {result.energy_per_token_mj:>10.3f} {quality:>10} {savings:>10}")

    # Business Value
    print("\n" + "=" * 70)
    print("BUSINESS VALUE (100 GPUs, 24/7 operation)")
    print("=" * 70)

    best_balanced = results.get('distilled_6') or results.get('distilled_9')
    if best_balanced:
        annual_gpu_hours = 8760
        gpu_count = 100
        cost_per_kwh = 0.10

        power_reduction = baseline.avg_power_w - best_balanced.avg_power_w
        savings_kwh = power_reduction * annual_gpu_hours / 1000 * gpu_count
        cost_savings = savings_kwh * cost_per_kwh
        carbon_reduction = savings_kwh * 0.4

        print(f"\nBest Balanced Configuration: {best_balanced.name}")
        print(f"  Energy savings: {best_balanced.energy_savings_pct:.1f}%")
        print(f"  Throughput gain: {best_balanced.throughput_gain_pct:.1f}%")
        print(f"  Quality ratio: {best_balanced.quality_ratio:.2f}x")
        print(f"\n  Power reduction: {power_reduction:.1f}W per GPU")
        print(f"  Annual energy savings: {savings_kwh:,.0f} kWh")
        print(f"  Annual cost savings: ${cost_savings:,.0f}")
        print(f"  Carbon reduction: {carbon_reduction:,.0f} kg CO2/year")

    # Save results
    output_path = Path("results/z303_final_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'results': {name: asdict(r) for name, r in results.items()},
        'baseline': {
            'tokens_per_second': baseline.tokens_per_second,
            'energy_per_token_mj': baseline.energy_per_token_mj,
            'avg_loss': baseline.avg_loss,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    telemetry.shutdown()

    print("\n" + "=" * 70)
    print("Final validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
