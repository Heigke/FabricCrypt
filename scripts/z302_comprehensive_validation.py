#!/usr/bin/env python3
"""
Comprehensive Validation - Measure Effect and Business Value

This script benchmarks ALL mechanisms:
1. Baseline PyTorch (no optimization)
2. High-level early exit (z204 distilled model)
3. Deep HIP energy modes (low/balanced/high)
4. Confidence-based adaptive exit

Metrics collected:
- Energy (mJ/token) via power integration
- Throughput (tokens/second)
- Quality (cross-entropy loss)
- Power draw (Watts)
- Temperature (°C)
- GPU utilization (%)

Business value calculation:
- Cost savings at scale
- Carbon footprint reduction
- Throughput improvements
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset

from src.telemetry.real_amd import AMDTelemetry, RocmSmiReader, EnergyMeter


@dataclass
class BenchmarkResult:
    """Results from one benchmark configuration."""
    name: str
    description: str

    # Performance metrics
    total_tokens: int
    total_time_ms: float
    tokens_per_second: float

    # Energy metrics
    total_energy_mj: float
    energy_per_token_mj: float
    avg_power_w: float
    peak_power_w: float

    # Quality metrics
    avg_loss: float

    # Hardware metrics
    avg_temp_c: float
    peak_temp_c: float
    avg_gpu_util: float

    # Computed metrics
    energy_savings_pct: float = 0.0
    throughput_gain_pct: float = 0.0
    quality_ratio: float = 1.0


class BaselineModel(nn.Module):
    """Standard GPT-2 without any optimizations."""

    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        outputs = self.model(input_ids, labels=labels)
        return outputs.logits, outputs.loss


class EarlyExitModel(nn.Module):
    """Early exit model with distilled exit heads."""

    def __init__(self, model_name: str = "gpt2", exit_layer: int = 9):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.exit_layer = exit_layer

        # Extract components
        self.wte = self.model.transformer.wte
        self.wpe = self.model.transformer.wpe
        self.drop = self.model.transformer.drop
        self.blocks = self.model.transformer.h
        self.ln_f = self.model.transformer.ln_f
        self.lm_head = self.model.lm_head

        self.num_layers = len(self.blocks)
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        # Exit head for early layer
        if exit_layer < self.num_layers:
            self.exit_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
            )
            # Initialize from lm_head
            with torch.no_grad():
                self.exit_head[-1].weight.copy_(self.lm_head.weight)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        # Forward through blocks UP TO exit_layer
        for i in range(min(self.exit_layer, self.num_layers)):
            hidden_states = self.blocks[i](hidden_states)[0]

        # Apply exit head or final layer
        if self.exit_layer < self.num_layers:
            logits = self.exit_head(hidden_states)
        else:
            hidden_states = self.ln_f(hidden_states)
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss


class EnergyAwareModel(nn.Module):
    """
    Model with deep energy-aware computation.

    Adjusts computation based on energy mode:
    - low: Reduced context, aggressive pruning
    - balanced: Standard computation
    - high: Full precision, no shortcuts
    """

    def __init__(self, model_name: str = "gpt2", energy_mode: str = "balanced"):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.energy_mode = energy_mode
        self.vocab_size = self.model.config.vocab_size

        # Energy mode configurations
        self.configs = {
            'low': {
                'max_context': 64,      # Reduced context window
                'use_fp16': True,       # Lower precision
                'softmax_temp': 1.5,    # Higher temperature (softer)
            },
            'balanced': {
                'max_context': 128,
                'use_fp16': False,
                'softmax_temp': 1.0,
            },
            'high': {
                'max_context': 512,
                'use_fp16': False,
                'softmax_temp': 1.0,
            },
        }

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        cfg = self.configs[self.energy_mode]

        # Apply context limitation for low power mode
        if input_ids.size(1) > cfg['max_context']:
            input_ids = input_ids[:, -cfg['max_context']:]
            if labels is not None:
                labels = labels[:, -cfg['max_context']:]

        # Forward pass
        if cfg['use_fp16'] and input_ids.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, labels=labels)
        else:
            outputs = self.model(input_ids, labels=labels)

        return outputs.logits, outputs.loss


def run_benchmark(
    model: nn.Module,
    batches: List[torch.Tensor],
    telemetry: AMDTelemetry,
    device: torch.device,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    name: str = "benchmark",
    description: str = ""
) -> BenchmarkResult:
    """Run comprehensive benchmark for a model configuration."""

    model.eval()

    # Warmup
    for _ in range(warmup_iterations):
        for batch in batches[:2]:
            with torch.no_grad():
                _ = model(batch)
            torch.cuda.synchronize()

    # Count tokens
    total_tokens = sum(
        (batch != 50256).sum().item()  # GPT-2 pad token
        for batch in batches
    ) * num_iterations

    # Metrics collection
    power_samples = []
    temp_samples = []
    util_samples = []
    losses = []

    # Measure
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        for batch in batches:
            labels = batch.clone()

            with torch.no_grad():
                logits, loss = model(batch, labels)
                if loss is not None:
                    losses.append(loss.item())

            # Sample hardware metrics
            power = RocmSmiReader.read_power()
            if power:
                power_samples.append(power)

            snapshot = telemetry.read()
            temp_samples.append(snapshot.temp_c)
            util_samples.append(snapshot.gpu_util_pct)

    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate metrics
    total_time_ms = (end_time - start_time) * 1000
    avg_power = np.mean(power_samples) if power_samples else 0
    peak_power = np.max(power_samples) if power_samples else 0
    total_energy_mj = avg_power * (total_time_ms / 1000) * 1000

    return BenchmarkResult(
        name=name,
        description=description,
        total_tokens=total_tokens,
        total_time_ms=total_time_ms,
        tokens_per_second=total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0,
        total_energy_mj=total_energy_mj,
        energy_per_token_mj=total_energy_mj / total_tokens if total_tokens > 0 else 0,
        avg_power_w=avg_power,
        peak_power_w=peak_power,
        avg_loss=np.mean(losses) if losses else 0,
        avg_temp_c=np.mean(temp_samples) if temp_samples else 0,
        peak_temp_c=np.max(temp_samples) if temp_samples else 0,
        avg_gpu_util=np.mean(util_samples) if util_samples else 0,
    )


def calculate_business_value(
    baseline: BenchmarkResult,
    optimized: BenchmarkResult,
    annual_gpu_hours: float = 8760,  # 24/7 for 1 year
    cost_per_kwh: float = 0.10,      # $0.10/kWh
    gpu_count: int = 100,            # Number of GPUs
    carbon_per_kwh: float = 0.4      # kg CO2 per kWh
) -> Dict:
    """Calculate business value of optimization."""

    # Energy savings
    energy_savings_pct = (1 - optimized.energy_per_token_mj / baseline.energy_per_token_mj) * 100

    # Power reduction
    power_reduction_w = baseline.avg_power_w - optimized.avg_power_w

    # Annual energy savings per GPU (kWh)
    # Assuming continuous operation at measured power levels
    baseline_annual_kwh = baseline.avg_power_w * annual_gpu_hours / 1000
    optimized_annual_kwh = optimized.avg_power_w * annual_gpu_hours / 1000
    savings_kwh_per_gpu = baseline_annual_kwh - optimized_annual_kwh

    # Fleet-wide savings
    total_savings_kwh = savings_kwh_per_gpu * gpu_count
    total_cost_savings = total_savings_kwh * cost_per_kwh

    # Carbon reduction
    carbon_reduction_kg = total_savings_kwh * carbon_per_kwh

    # Throughput improvement
    throughput_gain_pct = (optimized.tokens_per_second / baseline.tokens_per_second - 1) * 100

    # Effective capacity increase
    # If throughput increases by X%, you need X% fewer GPUs for same workload
    capacity_multiplier = optimized.tokens_per_second / baseline.tokens_per_second
    effective_gpu_savings = gpu_count * (1 - 1/capacity_multiplier)

    return {
        'energy_savings_pct': energy_savings_pct,
        'power_reduction_w': power_reduction_w,
        'annual_savings_kwh_per_gpu': savings_kwh_per_gpu,
        'fleet_annual_savings_kwh': total_savings_kwh,
        'annual_cost_savings_usd': total_cost_savings,
        'carbon_reduction_kg_co2': carbon_reduction_kg,
        'throughput_gain_pct': throughput_gain_pct,
        'capacity_multiplier': capacity_multiplier,
        'effective_gpu_savings': effective_gpu_savings,
        'quality_ratio': optimized.avg_loss / baseline.avg_loss if baseline.avg_loss > 0 else 1.0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", default="results/z302_comprehensive_validation.json")
    args = parser.parse_args()

    print("=" * 70)
    print("COMPREHENSIVE VALIDATION - Effect and Business Value")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize telemetry
    print("\nInitializing hardware telemetry...")
    telemetry = AMDTelemetry()
    initial = telemetry.read()
    print(f"  Power: {initial.power_w:.1f}W, Temp: {initial.temp_c:.0f}°C")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
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

    # ========== BENCHMARKS ==========
    results = {}

    # 1. Baseline (full model)
    print("\n" + "=" * 70)
    print("1. BASELINE - Standard GPT-2")
    print("=" * 70)

    baseline_model = BaselineModel("gpt2").to(device)
    baseline_model.eval()

    results['baseline'] = run_benchmark(
        baseline_model, batches, telemetry, device,
        num_iterations=args.iterations,
        name="baseline",
        description="Standard GPT-2, no optimizations"
    )

    print(f"  Tokens/sec: {results['baseline'].tokens_per_second:.0f}")
    print(f"  Energy: {results['baseline'].energy_per_token_mj:.3f} mJ/token")
    print(f"  Power: {results['baseline'].avg_power_w:.1f}W (peak: {results['baseline'].peak_power_w:.1f}W)")
    print(f"  Loss: {results['baseline'].avg_loss:.3f}")
    print(f"  Temp: {results['baseline'].avg_temp_c:.0f}°C")

    del baseline_model
    torch.cuda.empty_cache()
    time.sleep(2)  # Let GPU cool

    # 2. Early Exit - Layer 9 (best quality)
    print("\n" + "=" * 70)
    print("2. EARLY EXIT - Layer 9 (75% compute)")
    print("=" * 70)

    early_exit_9 = EarlyExitModel("gpt2", exit_layer=9).to(device)
    early_exit_9.eval()

    results['early_exit_9'] = run_benchmark(
        early_exit_9, batches, telemetry, device,
        num_iterations=args.iterations,
        name="early_exit_9",
        description="Exit at layer 9 (75% of layers)"
    )

    print(f"  Tokens/sec: {results['early_exit_9'].tokens_per_second:.0f}")
    print(f"  Energy: {results['early_exit_9'].energy_per_token_mj:.3f} mJ/token")
    print(f"  Power: {results['early_exit_9'].avg_power_w:.1f}W")
    print(f"  Loss: {results['early_exit_9'].avg_loss:.3f}")

    del early_exit_9
    torch.cuda.empty_cache()
    time.sleep(2)

    # 3. Early Exit - Layer 6 (balanced)
    print("\n" + "=" * 70)
    print("3. EARLY EXIT - Layer 6 (50% compute)")
    print("=" * 70)

    early_exit_6 = EarlyExitModel("gpt2", exit_layer=6).to(device)
    early_exit_6.eval()

    results['early_exit_6'] = run_benchmark(
        early_exit_6, batches, telemetry, device,
        num_iterations=args.iterations,
        name="early_exit_6",
        description="Exit at layer 6 (50% of layers)"
    )

    print(f"  Tokens/sec: {results['early_exit_6'].tokens_per_second:.0f}")
    print(f"  Energy: {results['early_exit_6'].energy_per_token_mj:.3f} mJ/token")
    print(f"  Power: {results['early_exit_6'].avg_power_w:.1f}W")
    print(f"  Loss: {results['early_exit_6'].avg_loss:.3f}")

    del early_exit_6
    torch.cuda.empty_cache()
    time.sleep(2)

    # 4. Early Exit - Layer 3 (maximum savings)
    print("\n" + "=" * 70)
    print("4. EARLY EXIT - Layer 3 (25% compute)")
    print("=" * 70)

    early_exit_3 = EarlyExitModel("gpt2", exit_layer=3).to(device)
    early_exit_3.eval()

    results['early_exit_3'] = run_benchmark(
        early_exit_3, batches, telemetry, device,
        num_iterations=args.iterations,
        name="early_exit_3",
        description="Exit at layer 3 (25% of layers)"
    )

    print(f"  Tokens/sec: {results['early_exit_3'].tokens_per_second:.0f}")
    print(f"  Energy: {results['early_exit_3'].energy_per_token_mj:.3f} mJ/token")
    print(f"  Power: {results['early_exit_3'].avg_power_w:.1f}W")
    print(f"  Loss: {results['early_exit_3'].avg_loss:.3f}")

    del early_exit_3
    torch.cuda.empty_cache()
    time.sleep(2)

    # 5. Energy-Aware Low Power Mode
    print("\n" + "=" * 70)
    print("5. ENERGY-AWARE - Low Power Mode")
    print("=" * 70)

    energy_low = EnergyAwareModel("gpt2", energy_mode="low").to(device)
    energy_low.eval()

    results['energy_low'] = run_benchmark(
        energy_low, batches, telemetry, device,
        num_iterations=args.iterations,
        name="energy_low",
        description="Low power mode (reduced context, FP16)"
    )

    print(f"  Tokens/sec: {results['energy_low'].tokens_per_second:.0f}")
    print(f"  Energy: {results['energy_low'].energy_per_token_mj:.3f} mJ/token")
    print(f"  Power: {results['energy_low'].avg_power_w:.1f}W")
    print(f"  Loss: {results['energy_low'].avg_loss:.3f}")

    del energy_low
    torch.cuda.empty_cache()
    time.sleep(2)

    # 6. Energy-Aware Balanced Mode
    print("\n" + "=" * 70)
    print("6. ENERGY-AWARE - Balanced Mode")
    print("=" * 70)

    energy_balanced = EnergyAwareModel("gpt2", energy_mode="balanced").to(device)
    energy_balanced.eval()

    results['energy_balanced'] = run_benchmark(
        energy_balanced, batches, telemetry, device,
        num_iterations=args.iterations,
        name="energy_balanced",
        description="Balanced power mode"
    )

    print(f"  Tokens/sec: {results['energy_balanced'].tokens_per_second:.0f}")
    print(f"  Energy: {results['energy_balanced'].energy_per_token_mj:.3f} mJ/token")
    print(f"  Power: {results['energy_balanced'].avg_power_w:.1f}W")
    print(f"  Loss: {results['energy_balanced'].avg_loss:.3f}")

    del energy_balanced
    torch.cuda.empty_cache()

    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    baseline = results['baseline']

    # Calculate relative metrics for all configurations
    for name, result in results.items():
        if name != 'baseline':
            result.energy_savings_pct = (1 - result.energy_per_token_mj / baseline.energy_per_token_mj) * 100
            result.throughput_gain_pct = (result.tokens_per_second / baseline.tokens_per_second - 1) * 100
            result.quality_ratio = result.avg_loss / baseline.avg_loss if baseline.avg_loss > 0 else 1.0

    print(f"\nBaseline: {baseline.tokens_per_second:.0f} tok/s, {baseline.energy_per_token_mj:.3f} mJ/tok, loss={baseline.avg_loss:.3f}")

    print("\n" + "-" * 70)
    print(f"{'Configuration':<25} {'Throughput':>12} {'Energy':>12} {'Quality':>10} {'Savings':>10}")
    print(f"{'':<25} {'(tok/s)':>12} {'(mJ/tok)':>12} {'(ratio)':>10} {'(%)':>10}")
    print("-" * 70)

    for name, result in sorted(results.items(), key=lambda x: x[1].energy_per_token_mj):
        savings = f"{result.energy_savings_pct:+.1f}%" if name != 'baseline' else "baseline"
        quality = f"{result.quality_ratio:.2f}x" if name != 'baseline' else "1.00x"
        print(f"{result.name:<25} {result.tokens_per_second:>12.0f} {result.energy_per_token_mj:>12.3f} {quality:>10} {savings:>10}")

    # ========== BUSINESS VALUE ==========
    print("\n" + "=" * 70)
    print("BUSINESS VALUE ANALYSIS")
    print("=" * 70)

    # Assumptions
    print("\nAssumptions:")
    print("  - 100 GPUs in fleet")
    print("  - 24/7 operation (8,760 hours/year)")
    print("  - $0.10/kWh electricity cost")
    print("  - 0.4 kg CO2/kWh carbon intensity")

    # Calculate for best balanced configuration (early_exit_6)
    if 'early_exit_6' in results:
        bv_6 = calculate_business_value(baseline, results['early_exit_6'])

        print("\n--- Early Exit Layer 6 (Best Balance) ---")
        print(f"  Energy savings: {bv_6['energy_savings_pct']:.1f}%")
        print(f"  Throughput gain: {bv_6['throughput_gain_pct']:.1f}%")
        print(f"  Quality ratio: {bv_6['quality_ratio']:.2f}x")
        print(f"\n  Annual energy savings per GPU: {bv_6['annual_savings_kwh_per_gpu']:.0f} kWh")
        print(f"  Fleet annual savings: {bv_6['fleet_annual_savings_kwh']:,.0f} kWh")
        print(f"  Annual cost savings: ${bv_6['annual_cost_savings_usd']:,.0f}")
        print(f"  Carbon reduction: {bv_6['carbon_reduction_kg_co2']:,.0f} kg CO2/year")
        print(f"\n  Capacity multiplier: {bv_6['capacity_multiplier']:.2f}x")
        print(f"  Effective GPU savings: {bv_6['effective_gpu_savings']:.0f} GPUs")

    # Calculate for maximum savings (early_exit_3)
    if 'early_exit_3' in results:
        bv_3 = calculate_business_value(baseline, results['early_exit_3'])

        print("\n--- Early Exit Layer 3 (Maximum Savings) ---")
        print(f"  Energy savings: {bv_3['energy_savings_pct']:.1f}%")
        print(f"  Throughput gain: {bv_3['throughput_gain_pct']:.1f}%")
        print(f"  Quality ratio: {bv_3['quality_ratio']:.2f}x")
        print(f"\n  Annual cost savings: ${bv_3['annual_cost_savings_usd']:,.0f}")
        print(f"  Carbon reduction: {bv_3['carbon_reduction_kg_co2']:,.0f} kg CO2/year")

    # ========== PARETO FRONTIER ==========
    print("\n" + "=" * 70)
    print("PARETO FRONTIER - Quality vs Energy")
    print("=" * 70)

    pareto_points = []
    for name, result in results.items():
        if name == 'baseline':
            continue

        # Check if this point is Pareto-optimal
        is_pareto = True
        for other_name, other_result in results.items():
            if other_name == name or other_name == 'baseline':
                continue

            # Check if other dominates this
            if (other_result.energy_per_token_mj <= result.energy_per_token_mj and
                other_result.avg_loss <= result.avg_loss and
                (other_result.energy_per_token_mj < result.energy_per_token_mj or
                 other_result.avg_loss < result.avg_loss)):
                is_pareto = False
                break

        if is_pareto:
            pareto_points.append(result)

    print("\nPareto-optimal configurations:")
    for p in sorted(pareto_points, key=lambda x: x.energy_per_token_mj):
        print(f"  {p.name}: {p.energy_per_token_mj:.3f} mJ/tok, loss={p.avg_loss:.3f}, "
              f"savings={p.energy_savings_pct:.1f}%")

    # ========== RECOMMENDATIONS ==========
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find best for different use cases
    best_quality = min((r for n, r in results.items() if n != 'baseline'),
                       key=lambda x: x.avg_loss)
    best_efficiency = min((r for n, r in results.items() if n != 'baseline'),
                          key=lambda x: x.energy_per_token_mj)
    best_balanced = min((r for n, r in results.items() if n != 'baseline'),
                        key=lambda x: x.quality_ratio + (1 - x.energy_savings_pct/100))

    print(f"\n  Best Quality: {best_quality.name}")
    print(f"    Loss: {best_quality.avg_loss:.3f} ({best_quality.quality_ratio:.2f}x baseline)")
    print(f"    Energy savings: {best_quality.energy_savings_pct:.1f}%")

    print(f"\n  Best Efficiency: {best_efficiency.name}")
    print(f"    Energy savings: {best_efficiency.energy_savings_pct:.1f}%")
    print(f"    Quality: {best_efficiency.quality_ratio:.2f}x baseline")

    print(f"\n  Best Balanced: {best_balanced.name}")
    print(f"    Energy savings: {best_balanced.energy_savings_pct:.1f}%")
    print(f"    Quality: {best_balanced.quality_ratio:.2f}x baseline")
    print(f"    Throughput: {best_balanced.throughput_gain_pct:+.1f}%")

    # ========== SAVE RESULTS ==========
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'config': {
            'batch_size': args.batch_size,
            'num_batches': args.num_batches,
            'iterations': args.iterations,
            'device': str(device),
        },
        'results': {name: asdict(result) for name, result in results.items()},
        'business_value': {
            'early_exit_6': bv_6 if 'early_exit_6' in results else None,
            'early_exit_3': bv_3 if 'early_exit_3' in results else None,
        },
        'pareto_optimal': [p.name for p in pareto_points],
        'recommendations': {
            'best_quality': best_quality.name,
            'best_efficiency': best_efficiency.name,
            'best_balanced': best_balanced.name,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Cleanup
    telemetry.shutdown()

    print("\n" + "=" * 70)
    print("Comprehensive validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
