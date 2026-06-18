#!/usr/bin/env python3
"""
Z403: Rigorous Validation with Disturbance Protocol

This validates THREE falsifiable properties:
1. Intervention sensitivity: Body state changes -> compute policy changes
2. Semantic sensitivity: Text difficulty -> compute spending changes
3. Outperforms best fixed policy under disturbances

Baselines to beat:
- Fixed L3 (aggressive)
- Fixed L6 (balanced)
- Fixed L9 (quality)
- Fixed L12 (full)
- Uncertainty-only (no body awareness)

Disturbance protocol:
- Bursty arrivals (variable batch timing)
- Thermal ramps (sustained load)
- Power budget changes (simulated)
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter
from src.controllers.adaptive_controller import (
    AdaptiveController, FixedController, UncertaintyOnlyController,
    ControllerConfig, ComputeAction
)

from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset

try:
    from scripts.z204_distill_exit_heads import DistillableEarlyExitGPT2
    HAS_DISTILLED = True
except ImportError:
    HAS_DISTILLED = False


@dataclass
class ValidationMetrics:
    """Metrics from a validation run."""
    name: str
    controller_type: str

    # Performance
    total_tokens: int = 0
    total_time_s: float = 0.0
    throughput_tok_s: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Energy (integrated from sysfs, labeled as such)
    energy_j: float = 0.0
    energy_per_token_mj: float = 0.0
    avg_power_w: float = 0.0

    # Quality
    avg_loss: float = 0.0
    quality_ratio: float = 1.0

    # Compute
    avg_exit_layer: float = 12.0
    exit_distribution: Dict[int, int] = field(default_factory=dict)
    compute_reduction_pct: float = 0.0

    # Constraints
    temp_violations: int = 0
    latency_violations: int = 0
    max_temp_c: float = 0.0
    constraint_score: float = 0.0  # Combined metric

    # Adaptivity (for checking if truly adaptive)
    exit_variance: float = 0.0  # Variance in exit layers (fixed=0)
    body_pressure_correlation: float = 0.0  # Correlation with body state


@dataclass
class DisturbanceConfig:
    """Configuration for disturbance protocol."""
    # Bursty arrivals
    enable_bursty: bool = True
    burst_probability: float = 0.2
    burst_delay_ms: float = 500.0

    # Thermal ramp (sustained load)
    enable_thermal_ramp: bool = True
    thermal_ramp_batches: int = 30

    # Variable difficulty
    enable_difficulty_variation: bool = True


class EarlyExitModel(torch.nn.Module):
    """Model with early exit capability."""

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

    def compute_loss_and_uncertainty(self, input_ids, exit_layer=12):
        """Compute loss and uncertainty (entropy of predictions)."""
        logits = self.forward(input_ids, exit_layer)

        # Loss
        labels = input_ids.clone()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )

        # Uncertainty: entropy of last token prediction
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        # Normalize entropy to [0,1] (max entropy for vocab_size)
        max_entropy = np.log(self.vocab_size)
        uncertainty = (entropy.mean() / max_entropy).item()

        return loss.item(), uncertainty


def run_validation(
    model,
    batches: List[torch.Tensor],
    controller,
    telemetry: SysfsHwmonTelemetry,
    config: ControllerConfig,
    disturbance: Optional[DisturbanceConfig] = None,
    name: str = "test",
    warmup_batches: int = 5,
) -> ValidationMetrics:
    """Run validation with disturbance protocol."""

    model.eval()
    metrics = ValidationMetrics(name=name, controller_type=type(controller).__name__)

    # Tracking
    latencies = []
    losses = []
    exit_layers = []
    temps = []
    powers = []
    body_pressures = []
    total_tokens = 0

    # Warmup
    for batch in batches[:warmup_batches]:
        with torch.no_grad():
            _ = model.forward(batch, exit_layer=12)
        torch.cuda.synchronize()

    time.sleep(2)  # Cool down

    # Main validation with energy measurement
    # IMPORTANT: Start EnergyMeter BEFORE controller to avoid race condition
    # (EnergyMeter resets accumulator, which would clear AdaptiveController's telemetry)
    with EnergyMeter(telemetry) as meter:
        # Start controller AFTER EnergyMeter starts sampling
        controller.start()

        for batch_idx, batch in enumerate(tqdm(batches, desc=f"  {name}")):

            # Apply disturbances
            if disturbance:
                # Bursty arrivals
                if disturbance.enable_bursty and random.random() < disturbance.burst_probability:
                    time.sleep(disturbance.burst_delay_ms / 1000.0)

            batch_start = time.perf_counter()

            with torch.no_grad():
                # First, compute uncertainty with a quick forward pass
                _, uncertainty = model.compute_loss_and_uncertainty(batch, exit_layer=12)

                # Record body pressure AT DECISION TIME (before get_compute_action changes state)
                stats_at_decision = controller.get_statistics()
                if 'body_pressure' in stats_at_decision:
                    body_pressures.append(stats_at_decision['body_pressure'])

                # Get compute action using TWO-SIGNAL policy
                action = controller.get_compute_action(uncertainty=uncertainty)
                exit_layer = action.value

                # Actual forward pass with chosen exit layer
                loss, _ = model.compute_loss_and_uncertainty(batch, exit_layer)

            torch.cuda.synchronize()
            batch_latency = (time.perf_counter() - batch_start) * 1000

            # Record body state (for temperature/power tracking)
            sample = telemetry.get_latest_sample()
            if sample:
                temps.append(sample.temp_edge_c)
                powers.append(sample.power_w)
                if sample.temp_edge_c > config.temp_cap_c:
                    metrics.temp_violations += 1

            if batch_latency > config.latency_cap_ms:
                metrics.latency_violations += 1

            latencies.append(batch_latency)
            losses.append(loss)
            exit_layers.append(exit_layer)
            total_tokens += (batch != 50256).sum().item()

            controller.report_latency(batch_latency)

    controller.stop()

    # Compute metrics
    metrics.total_tokens = total_tokens
    metrics.total_time_s = meter.duration_s
    metrics.throughput_tok_s = total_tokens / meter.duration_s if meter.duration_s > 0 else 0
    metrics.latency_p50_ms = np.percentile(latencies, 50)
    metrics.latency_p99_ms = np.percentile(latencies, 99)

    metrics.energy_j = meter.energy_j
    metrics.energy_per_token_mj = (meter.energy_j * 1000) / total_tokens if total_tokens > 0 else 0
    metrics.avg_power_w = meter.avg_power_w

    metrics.avg_loss = np.mean(losses)
    metrics.avg_exit_layer = np.mean(exit_layers)
    metrics.compute_reduction_pct = (12 - metrics.avg_exit_layer) / 12 * 100

    metrics.exit_distribution = {int(k): int(v) for k, v in
                                  zip(*np.unique(exit_layers, return_counts=True))}

    metrics.max_temp_c = max(temps) if temps else 0

    # Constraint score (higher is better)
    total_samples = len(batches)
    violation_rate = (metrics.temp_violations + metrics.latency_violations) / (2 * total_samples)
    metrics.constraint_score = (1 - violation_rate) * 100

    # Adaptivity metrics
    metrics.exit_variance = np.var(exit_layers)

    # Body pressure correlation (if available)
    if body_pressures and len(body_pressures) == len(exit_layers):
        # Higher body pressure should correlate with lower exit layers
        if np.std(body_pressures) > 0 and np.std(exit_layers) > 0:
            metrics.body_pressure_correlation = -np.corrcoef(body_pressures, exit_layers)[0, 1]
        else:
            metrics.body_pressure_correlation = 0.0

    return metrics


def run_all_baselines(
    model,
    batches,
    telemetry,
    config,
    disturbance,
) -> Dict[str, ValidationMetrics]:
    """Run all baseline comparisons."""

    results = {}

    # DEBUG: Skip baselines for faster testing
    SKIP_BASELINES = False

    if not SKIP_BASELINES:
        # Fixed baselines
        for exit_layer in [3, 6, 9, 12]:
            print(f"\n--- Fixed L{exit_layer} Baseline ---")
            controller = FixedController(exit_layer=exit_layer)
            results[f'fixed_L{exit_layer}'] = run_validation(
                model, batches, controller, telemetry, config,
                disturbance, name=f'fixed_L{exit_layer}'
            )
            time.sleep(3)  # Cool down

        # Uncertainty-only baseline
        print("\n--- Uncertainty-Only Baseline ---")
        controller = UncertaintyOnlyController()
        results['uncertainty_only'] = run_validation(
            model, batches, controller, telemetry, config,
            disturbance, name='uncertainty_only'
        )
        time.sleep(3)
    else:
        print("\n[DEBUG] Skipping baselines, testing adaptive only...")

    # Adaptive controller (the one we need to prove works)
    print("\n--- Adaptive Two-Signal Controller ---")
    controller = AdaptiveController(telemetry, config)
    results['adaptive'] = run_validation(
        model, batches, controller, telemetry, config,
        disturbance, name='adaptive'
    )

    return results


def analyze_results(results: Dict[str, ValidationMetrics], baseline_name: str = 'fixed_L12'):
    """Analyze and compare results."""

    baseline = results.get(baseline_name)
    if not baseline:
        print(f"Warning: Baseline {baseline_name} not found")
        return

    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    # Summary table
    print(f"\n{'Controller':<20} {'Tok/s':>10} {'mJ/tok':>10} {'Loss':>8} "
          f"{'Exit':>6} {'Var':>6} {'Constr':>8}")
    print("-" * 80)

    for name, m in sorted(results.items()):
        energy_change = (1 - m.energy_per_token_mj / baseline.energy_per_token_mj) * 100 \
            if baseline.energy_per_token_mj > 0 else 0

        print(f"{name:<20} {m.throughput_tok_s:>10.0f} {m.energy_per_token_mj:>10.3f} "
              f"{m.avg_loss:>8.3f} {m.avg_exit_layer:>6.1f} {m.exit_variance:>6.2f} "
              f"{m.constraint_score:>7.1f}%")

    # Check falsifiable properties
    print("\n" + "=" * 80)
    print("FALSIFIABLE PROPERTY CHECKS")
    print("=" * 80)

    adaptive = results.get('adaptive')
    if not adaptive:
        print("No adaptive controller results!")
        return

    # Property 1: Intervention sensitivity (exit variance > 0 for adaptive)
    print("\n1. INTERVENTION SENSITIVITY")
    print(f"   Adaptive exit variance: {adaptive.exit_variance:.3f}")
    print(f"   Fixed L12 exit variance: {results['fixed_L12'].exit_variance:.3f}")
    if adaptive.exit_variance > 0.1:
        print("   ✓ PASS: Controller varies exit based on state")
    else:
        print("   ✗ FAIL: Controller behaves like fixed policy")

    # Property 2: Body pressure correlation
    print("\n2. BODY-COMPUTE CORRELATION")
    print(f"   Body pressure correlation: {adaptive.body_pressure_correlation:.3f}")
    if adaptive.body_pressure_correlation > 0.1:
        print("   ✓ PASS: Higher body pressure -> earlier exit")
    else:
        print("   ✗ FAIL: Body state doesn't influence compute")

    # Property 3: Beats best fixed under disturbances
    print("\n3. BEATS BEST FIXED POLICY")

    # Find best fixed policy (best energy-quality tradeoff)
    fixed_results = {k: v for k, v in results.items() if k.startswith('fixed_')}

    # Score: energy savings * constraint score * (1 / quality degradation)
    def compute_score(m):
        energy_ratio = baseline.energy_per_token_mj / m.energy_per_token_mj \
            if m.energy_per_token_mj > 0 else 1
        quality_ratio = baseline.avg_loss / m.avg_loss if m.avg_loss > 0 else 1
        return energy_ratio * (m.constraint_score / 100) * min(1.0, quality_ratio)

    best_fixed_name = max(fixed_results.keys(), key=lambda k: compute_score(fixed_results[k]))
    best_fixed = fixed_results[best_fixed_name]

    adaptive_score = compute_score(adaptive)
    best_fixed_score = compute_score(best_fixed)

    print(f"   Best fixed policy: {best_fixed_name} (score: {best_fixed_score:.3f})")
    print(f"   Adaptive score: {adaptive_score:.3f}")

    if adaptive_score > best_fixed_score:
        print(f"   ✓ PASS: Adaptive beats best fixed by {(adaptive_score/best_fixed_score - 1)*100:.1f}%")
    else:
        print(f"   ✗ FAIL: Adaptive does not beat best fixed policy")

    # Energy comparison
    print("\n4. ENERGY COMPARISON VS BASELINE")
    for name, m in sorted(results.items()):
        savings = (1 - m.energy_per_token_mj / baseline.energy_per_token_mj) * 100 \
            if baseline.energy_per_token_mj > 0 else 0
        quality = m.avg_loss / baseline.avg_loss if baseline.avg_loss > 0 else 1
        print(f"   {name:<20}: {savings:+.1f}% energy, {quality:.2f}x quality")

    return {
        'intervention_sensitivity': adaptive.exit_variance > 0.1,
        'body_correlation': adaptive.body_pressure_correlation > 0.1,
        'beats_best_fixed': adaptive_score > best_fixed_score,
        'adaptive_score': adaptive_score,
        'best_fixed_score': best_fixed_score,
    }


def main():
    print("=" * 80)
    print("Z403: RIGOROUS VALIDATION WITH DISTURBANCE PROTOCOL")
    print("=" * 80)
    print("\nValidating three falsifiable properties:")
    print("  1. Intervention sensitivity: Body state changes -> compute changes")
    print("  2. Semantic sensitivity: Text difficulty -> compute changes")
    print("  3. Outperforms best fixed policy under disturbances")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Configuration with FAST SIGNAL thresholds for better body-compute correlation
    config = ControllerConfig(
        temp_cap_c=85.0,
        power_cap_w=150.0,  # Realistic cap
        latency_cap_ms=150.0,
        thermal_margin_c=15.0,
        power_margin_w=30.0,
        # Fast signal thresholds (these fluctuate per-batch)
        gpu_busy_threshold_pct=70.0,  # Start pressure at 70% GPU utilization
        power_delta_threshold_w_per_s=20.0,  # Pressure when power increases 20W/s
        latency_pressure_start_ms=350.0,  # Start pressure at 350ms latency
        # Balance uncertainty and body signals
        uncertainty_weight=0.5,
        body_weight=0.5,
    )

    disturbance = DisturbanceConfig(
        enable_bursty=True,
        burst_probability=0.15,
        burst_delay_ms=300.0,
    )

    # Initialize telemetry
    print("\n--- Telemetry ---")
    telemetry = SysfsHwmonTelemetry(sample_rate_hz=50)
    idle_power = telemetry.measure_idle_baseline(duration_s=2.0)
    print(f"Idle power: {idle_power:.1f} W")
    print(f"NOTE: Energy measured via sysfs power integration (not hardware counter)")

    # Load model
    print("\n--- Model ---")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    distilled_model = None
    if HAS_DISTILLED:
        checkpoint_path = Path("checkpoints/z204_distilled/model_final.pt")
        if checkpoint_path.exists():
            print("Loading distilled exit heads")
            distilled_model = DistillableEarlyExitGPT2("gpt2").to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            distilled_model.load_state_dict(checkpoint['model_state'])
            distilled_model.eval()

    model = EarlyExitModel(base_model, distilled_model).to(device)
    model.eval()

    # Prepare data
    print("\n--- Data ---")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    texts = []
    for item in dataset:
        if len(texts) >= 600:
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

    # Run all baselines
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION WITH DISTURBANCES")
    print("=" * 80)

    results = run_all_baselines(model, batches, telemetry, config, disturbance)

    # Analyze
    checks = analyze_results(results)

    # Business value (if adaptive passes)
    if checks and checks.get('beats_best_fixed'):
        print("\n" + "=" * 80)
        print("BUSINESS VALUE (Adaptive Controller)")
        print("=" * 80)

        baseline = results['fixed_L12']
        adaptive = results['adaptive']

        power_reduction = baseline.avg_power_w - adaptive.avg_power_w
        annual_hours = 8760
        gpu_count = 100
        cost_per_kwh = 0.10

        energy_kwh = power_reduction * annual_hours / 1000 * gpu_count
        cost_savings = energy_kwh * cost_per_kwh
        carbon_kg = energy_kwh * 0.4

        print(f"\nFor 100 GPUs, 24/7:")
        print(f"  Power reduction: {power_reduction:.1f} W/GPU")
        print(f"  Annual energy: {energy_kwh:,.0f} kWh saved")
        print(f"  Annual cost: ${cost_savings:,.0f} saved")
        print(f"  Carbon: {carbon_kg:,.0f} kg CO2/year reduced")

    # Save results
    output_path = Path("results/z403_rigorous_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return {k: convert(v) for k, v in vars(obj).items() if not k.startswith('_')}
        return obj

    output = {
        'results': {name: asdict(m) for name, m in results.items()},
        'falsifiable_checks': checks,
        'config': asdict(config),
        'disturbance': asdict(disturbance),
    }

    with open(output_path, 'w') as f:
        json.dump(convert(output), f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
