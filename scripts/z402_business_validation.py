#!/usr/bin/env python3
"""
Z402: Comprehensive Business Validation

Uses the proven embodied controller to compute full business metrics:
- Energy savings at scale
- Cost reduction
- Carbon footprint
- ROI calculations
- Quality/accuracy trade-offs
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
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter
from src.controllers.embodied_controller import (
    EmbodiedController, FixedController, ControllerConfig
)

from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset

try:
    from scripts.z204_distill_exit_heads import DistillableEarlyExitGPT2
    HAS_DISTILLED = True
except ImportError:
    HAS_DISTILLED = False


@dataclass
class BusinessMetrics:
    """Comprehensive business metrics."""
    # Performance
    throughput_tokens_per_sec: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p99: float = 0.0

    # Energy
    energy_per_token_mj: float = 0.0
    avg_power_w: float = 0.0
    energy_savings_pct: float = 0.0

    # Quality
    avg_loss: float = 0.0
    quality_ratio: float = 1.0  # vs baseline

    # Compute
    avg_exit_layer: float = 12.0
    compute_reduction_pct: float = 0.0

    # Constraints
    constraint_satisfaction_pct: float = 0.0
    temp_violations: int = 0
    max_temp_c: float = 0.0

    # Business (per 100 GPUs, annual)
    annual_energy_kwh: float = 0.0
    annual_cost_savings_usd: float = 0.0
    carbon_reduction_kg: float = 0.0
    roi_months: float = 0.0


class EarlyExitModel(torch.nn.Module):
    """Model with early exit."""

    def __init__(self, base_model, distilled_model=None):
        super().__init__()
        self.base_model = base_model
        self.distilled_model = distilled_model
        self.vocab_size = base_model.config.vocab_size

    def forward(self, input_ids, exit_layer=12):
        if self.distilled_model is not None:
            logits, _ = self.distilled_model.forward_to_layer(input_ids, exit_layer)
            return logits

        if exit_layer >= 12:
            return self.base_model(input_ids).logits

        hidden = self.base_model.transformer.wte(input_ids)
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = hidden + self.base_model.transformer.wpe(pos)
        hidden = self.base_model.transformer.drop(hidden)

        for block in self.base_model.transformer.h[:exit_layer]:
            hidden = block(hidden)[0]

        hidden = self.base_model.transformer.ln_f(hidden)
        return self.base_model.lm_head(hidden)

    def compute_loss(self, input_ids, exit_layer=12):
        logits = self.forward(input_ids, exit_layer)
        labels = input_ids.clone()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )
        return loss.item()


def run_benchmark(
    model,
    batches,
    controller,
    telemetry,
    config,
    name="test",
    warmup=5,
) -> BusinessMetrics:
    """Run benchmark and collect metrics."""

    model.eval()
    metrics = BusinessMetrics()

    # Tracking
    latencies = []
    losses = []
    exit_layers = []
    temps = []
    powers = []
    total_tokens = 0

    # Warmup
    for batch in batches[:warmup]:
        with torch.no_grad():
            _ = model.forward(batch, exit_layer=12)
        torch.cuda.synchronize()

    time.sleep(3)  # Cool down

    controller.start()

    # Main benchmark with energy measurement
    with EnergyMeter(telemetry) as meter:
        for batch in tqdm(batches, desc=f"  {name}"):
            batch_start = time.perf_counter()

            with torch.no_grad():
                action = controller.get_compute_action()
                exit_layer = action.value
                loss = model.compute_loss(batch, exit_layer)

            torch.cuda.synchronize()
            batch_latency = (time.perf_counter() - batch_start) * 1000

            sample = telemetry.get_latest_sample()
            if sample:
                temps.append(sample.temp_edge_c)
                powers.append(sample.power_w)
                if sample.temp_edge_c > config.temp_cap_c:
                    metrics.temp_violations += 1

            latencies.append(batch_latency)
            losses.append(loss)
            exit_layers.append(exit_layer)
            total_tokens += (batch != 50256).sum().item()

            controller.report_latency(batch_latency)

    controller.stop()

    # Compute metrics
    metrics.throughput_tokens_per_sec = total_tokens / meter.duration_s
    metrics.latency_ms_p50 = np.percentile(latencies, 50)
    metrics.latency_ms_p99 = np.percentile(latencies, 99)

    metrics.energy_per_token_mj = (meter.energy_j * 1000) / total_tokens
    metrics.avg_power_w = meter.avg_power_w

    metrics.avg_loss = np.mean(losses)
    metrics.avg_exit_layer = np.mean(exit_layers)
    metrics.compute_reduction_pct = (12 - metrics.avg_exit_layer) / 12 * 100

    metrics.max_temp_c = max(temps) if temps else 0
    latency_violations = sum(1 for l in latencies if l > config.latency_cap_ms)
    total_checks = len(batches) * 2
    violations = metrics.temp_violations + latency_violations
    metrics.constraint_satisfaction_pct = (1 - violations / total_checks) * 100

    return metrics


def compute_business_value(
    baseline: BusinessMetrics,
    optimized: BusinessMetrics,
    gpu_count: int = 100,
    cost_per_kwh: float = 0.10,
    carbon_per_kwh: float = 0.4,
    implementation_cost: float = 5000,
) -> BusinessMetrics:
    """Compute business value metrics."""

    result = BusinessMetrics()

    # Copy optimized metrics
    result.throughput_tokens_per_sec = optimized.throughput_tokens_per_sec
    result.latency_ms_p50 = optimized.latency_ms_p50
    result.latency_ms_p99 = optimized.latency_ms_p99
    result.energy_per_token_mj = optimized.energy_per_token_mj
    result.avg_power_w = optimized.avg_power_w
    result.avg_loss = optimized.avg_loss
    result.avg_exit_layer = optimized.avg_exit_layer
    result.compute_reduction_pct = optimized.compute_reduction_pct
    result.constraint_satisfaction_pct = optimized.constraint_satisfaction_pct
    result.temp_violations = optimized.temp_violations
    result.max_temp_c = optimized.max_temp_c

    # Relative metrics
    result.energy_savings_pct = (1 - optimized.energy_per_token_mj / baseline.energy_per_token_mj) * 100
    result.quality_ratio = optimized.avg_loss / baseline.avg_loss

    # Business calculations
    power_reduction_w = baseline.avg_power_w - optimized.avg_power_w
    annual_hours = 8760

    result.annual_energy_kwh = power_reduction_w * annual_hours / 1000 * gpu_count
    result.annual_cost_savings_usd = result.annual_energy_kwh * cost_per_kwh
    result.carbon_reduction_kg = result.annual_energy_kwh * carbon_per_kwh

    # ROI (months to recover implementation cost)
    if result.annual_cost_savings_usd > 0:
        result.roi_months = implementation_cost / (result.annual_cost_savings_usd / 12)
    else:
        result.roi_months = float('inf')

    return result


def main():
    print("=" * 70)
    print("Z402: COMPREHENSIVE BUSINESS VALIDATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config
    config = ControllerConfig(
        temp_cap_c=85.0,
        power_cap_w=120.0,
        latency_cap_ms=150.0,
    )

    # Initialize telemetry
    print("\n--- Initializing Telemetry ---")
    telemetry = SysfsHwmonTelemetry(sample_rate_hz=50)
    idle_power = telemetry.measure_idle_baseline(duration_s=2.0)
    print(f"Idle power: {idle_power:.1f} W")

    # Load model
    print("\n--- Loading Model ---")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    distilled_model = None
    if HAS_DISTILLED:
        checkpoint_path = Path("checkpoints/z204_distilled/model_final.pt")
        if checkpoint_path.exists():
            print(f"Loading distilled model")
            distilled_model = DistillableEarlyExitGPT2("gpt2").to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            distilled_model.load_state_dict(checkpoint['model_state'])
            distilled_model.eval()

    model = EarlyExitModel(base_model, distilled_model).to(device)
    model.eval()

    # Prepare data
    print("\n--- Preparing Data ---")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    texts = []
    for item in dataset:
        if len(texts) >= 800:
            break
        text = item['text'][:500]
        if len(text) > 50:
            texts.append(text)

    batch_size = 8
    seq_len = 128
    batches = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        if len(batch_texts) == batch_size:
            encoded = tokenizer(
                batch_texts,
                max_length=seq_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            batches.append(encoded['input_ids'].to(device))

    print(f"Prepared {len(batches)} batches")

    results = {}

    # 1. Baseline (Full Model)
    print("\n" + "=" * 70)
    print("1. BASELINE - Full Model (L12)")
    print("=" * 70)

    baseline_controller = FixedController(exit_layer=12)
    results['baseline'] = run_benchmark(
        model, batches, baseline_controller, telemetry, config, "baseline"
    )

    print(f"  Throughput: {results['baseline'].throughput_tokens_per_sec:.0f} tok/s")
    print(f"  Energy: {results['baseline'].energy_per_token_mj:.3f} mJ/tok")
    print(f"  Loss: {results['baseline'].avg_loss:.3f}")

    time.sleep(5)

    # 2. Embodied Controller
    print("\n" + "=" * 70)
    print("2. EMBODIED CONTROLLER - Predictive 3-Timescale")
    print("=" * 70)

    embodied_controller = EmbodiedController(telemetry, config)
    results['embodied'] = run_benchmark(
        model, batches, embodied_controller, telemetry, config, "embodied"
    )

    print(f"  Throughput: {results['embodied'].throughput_tokens_per_sec:.0f} tok/s")
    print(f"  Energy: {results['embodied'].energy_per_token_mj:.3f} mJ/tok")
    print(f"  Avg Exit: {results['embodied'].avg_exit_layer:.1f}")
    print(f"  Constraints: {results['embodied'].constraint_satisfaction_pct:.1f}%")

    time.sleep(5)

    # 3. Fixed L6 (50% compute)
    print("\n" + "=" * 70)
    print("3. FIXED L6 - 50% Compute")
    print("=" * 70)

    fixed_l6_controller = FixedController(exit_layer=6)
    results['fixed_l6'] = run_benchmark(
        model, batches, fixed_l6_controller, telemetry, config, "fixed_l6"
    )

    print(f"  Throughput: {results['fixed_l6'].throughput_tokens_per_sec:.0f} tok/s")
    print(f"  Energy: {results['fixed_l6'].energy_per_token_mj:.3f} mJ/tok")
    print(f"  Loss: {results['fixed_l6'].avg_loss:.3f}")

    # Compute business value
    print("\n" + "=" * 70)
    print("BUSINESS VALUE ANALYSIS")
    print("=" * 70)

    business = compute_business_value(
        results['baseline'],
        results['embodied'],
        gpu_count=100,
        cost_per_kwh=0.10,
    )

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'Metric':<30} {'Baseline':<15} {'Embodied':<15} {'Improvement':<15}")
    print("-" * 70)

    baseline = results['baseline']
    embodied = results['embodied']

    throughput_gain = (embodied.throughput_tokens_per_sec / baseline.throughput_tokens_per_sec - 1) * 100
    energy_savings = (1 - embodied.energy_per_token_mj / baseline.energy_per_token_mj) * 100

    print(f"{'Throughput (tok/s)':<30} {baseline.throughput_tokens_per_sec:<15.0f} {embodied.throughput_tokens_per_sec:<15.0f} {f'+{throughput_gain:.1f}%':<15}")
    print(f"{'Energy (mJ/tok)':<30} {baseline.energy_per_token_mj:<15.3f} {embodied.energy_per_token_mj:<15.3f} {f'{energy_savings:+.1f}%':<15}")
    print(f"{'Avg Power (W)':<30} {baseline.avg_power_w:<15.1f} {embodied.avg_power_w:<15.1f} {f'{baseline.avg_power_w - embodied.avg_power_w:+.1f}W':<15}")
    print(f"{'Loss':<30} {baseline.avg_loss:<15.3f} {embodied.avg_loss:<15.3f} {f'{embodied.avg_loss/baseline.avg_loss:.2f}x':<15}")
    print(f"{'Avg Exit Layer':<30} {'12.0':<15} {embodied.avg_exit_layer:<15.1f} {f'{embodied.compute_reduction_pct:.0f}% less':<15}")
    print(f"{'Constraint Satisfaction':<30} {'N/A':<15} {f'{embodied.constraint_satisfaction_pct:.1f}%':<15} {'':<15}")

    print("\n" + "=" * 70)
    print("BUSINESS VALUE (100 GPUs, 24/7 Operation)")
    print("=" * 70)

    print(f"\n  Energy Savings:      {business.energy_savings_pct:.1f}%")
    print(f"  Throughput Gain:     {throughput_gain:.1f}%")
    print(f"  Quality Ratio:       {business.quality_ratio:.2f}x")
    print(f"\n  Power Reduction:     {baseline.avg_power_w - embodied.avg_power_w:.1f} W per GPU")
    print(f"  Annual Energy Saved: {business.annual_energy_kwh:,.0f} kWh")
    print(f"  Annual Cost Savings: ${business.annual_cost_savings_usd:,.0f}")
    print(f"  Carbon Reduction:    {business.carbon_reduction_kg:,.0f} kg CO2/year")
    print(f"  ROI:                 {business.roi_months:.1f} months")

    # Scale projections
    print("\n" + "=" * 70)
    print("SCALE PROJECTIONS")
    print("=" * 70)

    for scale in [100, 1000, 10000]:
        scaled = compute_business_value(baseline, embodied, gpu_count=scale)
        print(f"\n  {scale:,} GPUs:")
        print(f"    Annual Energy: {scaled.annual_energy_kwh:,.0f} kWh")
        print(f"    Annual Savings: ${scaled.annual_cost_savings_usd:,.0f}")
        print(f"    Carbon Reduction: {scaled.carbon_reduction_kg:,.0f} kg CO2")

    # Save results
    output_path = Path("results/z402_business_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return convert(vars(obj))
        return obj

    output = {
        'results': {name: asdict(m) for name, m in results.items()},
        'business_value': asdict(business),
        'summary': {
            'energy_savings_pct': energy_savings,
            'throughput_gain_pct': throughput_gain,
            'quality_ratio': business.quality_ratio,
            'annual_cost_savings_100gpu': business.annual_cost_savings_usd,
            'carbon_reduction_100gpu_kg': business.carbon_reduction_kg,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(convert(output), f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
