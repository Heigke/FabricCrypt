#!/usr/bin/env python3
"""
Energy Validation with Proper Methodology

Key fixes:
1. Run multiple iterations per measurement (accumulate compute)
2. Measure energy over batches, not single samples
3. Add proper warmup
4. Use longer measurement windows for stable power readings
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.telemetry.real_amd import AMDTelemetry, EnergyMeter, RocmSmiReader


class ExitHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.out(self.norm(F.gelu(self.proj(x)) + x))


class BenchmarkModel(nn.Module):
    """Model for benchmarking different configurations."""

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        for p in self.base.parameters():
            p.requires_grad = False

        self.hidden_dim = self.base.config.hidden_size
        self.vocab_size = self.base.config.vocab_size
        self.num_layers = self.base.config.num_hidden_layers

        self.exit_layers = [3, 6, 9, 12]
        self.exit_heads = nn.ModuleList([
            ExitHead(self.hidden_dim, self.vocab_size)
            for _ in self.exit_layers
        ])

    def forward(self, input_ids, exit_layer: int = 12):
        with torch.no_grad():
            out = self.base(input_ids, output_hidden_states=True)
        hidden_states = out.hidden_states

        if exit_layer in self.exit_layers:
            exit_idx = self.exit_layers.index(exit_layer)
        else:
            exit_idx = -1

        h = hidden_states[exit_layer]
        logits = self.exit_heads[exit_idx](h)
        return logits


@dataclass
class BenchmarkResult:
    condition: str
    exit_layer: int
    batch_size: int
    num_iterations: int
    total_tokens: int
    total_energy_mj: float
    total_time_ms: float
    energy_per_token_mj: float
    tokens_per_second: float
    avg_power_w: float
    quality_loss: float


def run_benchmark(
    model: BenchmarkModel,
    tokenizer,
    batches: List[torch.Tensor],
    exit_layer: int,
    telemetry: AMDTelemetry,
    device: torch.device,
    num_iterations: int = 10,
    warmup_iterations: int = 3
) -> BenchmarkResult:
    """Run benchmark with proper energy measurement."""

    energy_meter = EnergyMeter(telemetry, sample_interval_ms=10)

    # Warmup
    for _ in range(warmup_iterations):
        for batch in batches:
            _ = model(batch, exit_layer)
            torch.cuda.synchronize()

    # Count tokens
    total_tokens = sum(
        (batch != tokenizer.pad_token_id).sum().item()
        for batch in batches
    ) * num_iterations

    # Measure
    torch.cuda.synchronize()
    power_samples = []

    start_time = time.time()
    with energy_meter.measure():
        for _ in range(num_iterations):
            for batch in batches:
                _ = model(batch, exit_layer)
                # Sample power during computation
                power = RocmSmiReader.read_power()
                if power:
                    power_samples.append(power)
        torch.cuda.synchronize()
    end_time = time.time()

    result = energy_meter.result
    duration_ms = (end_time - start_time) * 1000

    # Compute quality loss on one batch
    labels = batches[0].clone()
    logits = model(batches[0], exit_layer)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, model.vocab_size),
        shift_labels.view(-1),
        ignore_index=-100
    ).item()

    # Use power sampling if energy measurement failed
    if result.energy_mj < 1.0 and power_samples:
        # Estimate energy from power samples
        avg_power = np.mean(power_samples)
        estimated_energy_mj = avg_power * (duration_ms / 1000) * 1000  # W * s * 1000 = mJ
    else:
        estimated_energy_mj = result.energy_mj
        avg_power = result.avg_power_w if result.avg_power_w > 0 else np.mean(power_samples) if power_samples else 0

    return BenchmarkResult(
        condition=f"exit_{exit_layer}",
        exit_layer=exit_layer,
        batch_size=batches[0].size(0),
        num_iterations=num_iterations,
        total_tokens=total_tokens,
        total_energy_mj=estimated_energy_mj,
        total_time_ms=duration_ms,
        energy_per_token_mj=estimated_energy_mj / total_tokens if total_tokens > 0 else 0,
        tokens_per_second=total_tokens / (duration_ms / 1000) if duration_ms > 0 else 0,
        avg_power_w=avg_power,
        quality_loss=loss
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default="results/z202_energy_validation.json")
    args = parser.parse_args()

    print("="*60)
    print("Energy Validation with Proper Methodology")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize telemetry
    print("\nInitializing hardware telemetry...")
    telemetry = AMDTelemetry()
    initial = telemetry.read()
    print(f"  Power: {initial.power_w:.1f}W, Temp: {initial.temp_c:.0f}°C")

    # Load model
    print("\nLoading model...")
    model = BenchmarkModel("gpt2").to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare batches
    print(f"\nPreparing {args.num_batches} batches of size {args.batch_size}...")
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

    # Run benchmarks for different exit layers
    print(f"\nRunning benchmarks ({args.iterations} iterations each)...")
    results = {}

    for exit_layer in [3, 6, 9, 12]:
        print(f"\n--- Exit Layer {exit_layer} ---")

        result = run_benchmark(
            model, tokenizer, batches, exit_layer,
            telemetry, device, num_iterations=args.iterations
        )

        results[f"exit_{exit_layer}"] = result

        print(f"  Tokens: {result.total_tokens:,}")
        print(f"  Time: {result.total_time_ms:.0f} ms")
        print(f"  Energy: {result.total_energy_mj:.1f} mJ ({result.energy_per_token_mj:.3f} mJ/token)")
        print(f"  Throughput: {result.tokens_per_second:.0f} tok/s")
        print(f"  Avg Power: {result.avg_power_w:.1f} W")
        print(f"  Quality: {result.quality_loss:.3f}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    baseline = results["exit_12"]

    print(f"\nBaseline (Exit 12):")
    print(f"  {baseline.energy_per_token_mj:.3f} mJ/token")
    print(f"  {baseline.tokens_per_second:.0f} tok/s")
    print(f"  {baseline.avg_power_w:.1f} W")

    print(f"\nSavings vs Baseline:")
    for exit_layer in [3, 6, 9]:
        r = results[f"exit_{exit_layer}"]
        energy_savings = (1 - r.energy_per_token_mj / baseline.energy_per_token_mj) * 100 if baseline.energy_per_token_mj > 0 else 0
        throughput_gain = (r.tokens_per_second / baseline.tokens_per_second - 1) * 100 if baseline.tokens_per_second > 0 else 0
        quality_diff = (r.quality_loss - baseline.quality_loss) / baseline.quality_loss * 100 if baseline.quality_loss > 0 else 0

        print(f"\n  Exit Layer {exit_layer}:")
        print(f"    Energy: {energy_savings:+.1f}%")
        print(f"    Throughput: {throughput_gain:+.1f}%")
        print(f"    Quality: {quality_diff:+.1f}%")

    # Compute theoretical combined savings
    # ECC with 45% L3, 19% L6, 36% L9 distribution
    ecc_weights = {3: 0.45, 6: 0.19, 9: 0.36, 12: 0.0}
    ecc_energy = sum(
        ecc_weights[l] * results[f"exit_{l}"].energy_per_token_mj
        for l in [3, 6, 9]
    )
    ecc_savings = (1 - ecc_energy / baseline.energy_per_token_mj) * 100 if baseline.energy_per_token_mj > 0 else 0

    print(f"\n--- ECC Weighted Average (based on training distribution) ---")
    print(f"  Expected energy/token: {ecc_energy:.3f} mJ")
    print(f"  Expected savings: {ecc_savings:.1f}%")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'config': {
            'batch_size': args.batch_size,
            'num_batches': args.num_batches,
            'iterations': args.iterations
        },
        'results': {k: asdict(v) for k, v in results.items()},
        'analysis': {
            'baseline_energy_mj_per_token': baseline.energy_per_token_mj,
            'baseline_throughput': baseline.tokens_per_second,
            'ecc_weighted_energy_mj_per_token': ecc_energy,
            'ecc_expected_savings_pct': ecc_savings
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Cleanup
    telemetry.shutdown()

    print("\n" + "="*60)
    print("Validation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
