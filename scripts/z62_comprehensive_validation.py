#!/usr/bin/env python3
"""
Z62: Comprehensive FEEL Validation Framework
=============================================
Full validation of the Sense-Feel-Regulate-Express-HW_Change loop
with benchmarks against normal operation and business value calculation.

FEEL Loop:
1. SENSE   - Read hardware telemetry (power, temp, clocks, utilization)
2. FEEL    - Convert to body state with EMA smoothing + derivatives
3. REGULATE- FiLM conditioning modulates transformer computation
4. EXPRESS - Output action logits for hardware control
5. HW_CHANGE - Actually set power limits via nvidia-smi/sysfs
6. MEASURE - Track energy, tokens, quality metrics

Validation:
- Compare against baseline (no FEEL loop)
- Compare against fixed power modes
- Measure actual energy savings
- Calculate business value ($/token, CO2 savings)

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
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# BUSINESS VALUE CONSTANTS
# ============================================================
ELECTRICITY_COST_PER_KWH = 0.12  # USD
CO2_PER_KWH = 0.42  # kg CO2 (US average)
GPU_COST_PER_HOUR = 1.50  # USD (cloud A6000 rate)


@dataclass
class FEELLoopMetrics:
    """Metrics for each component of the FEEL loop."""
    # SENSE metrics
    telemetry_latency_ms: float = 0.0
    telemetry_samples: int = 0

    # FEEL metrics
    body_state_updates: int = 0
    ema_alpha: float = 0.1

    # REGULATE metrics
    film_activations: int = 0
    conditioning_strength: float = 0.0

    # EXPRESS metrics
    actions_taken: Dict[int, int] = None
    action_entropy: float = 0.0

    # HW_CHANGE metrics
    power_limit_changes: int = 0
    actuation_latency_ms: float = 0.0

    # EFFICIENCY metrics
    total_energy_j: float = 0.0
    total_tokens: int = 0
    j_per_token: float = 0.0

    # QUALITY metrics
    perplexity: float = 0.0
    tokens_per_sec: float = 0.0

    def __post_init__(self):
        if self.actions_taken is None:
            self.actions_taken = {0: 0, 1: 0, 2: 0, 3: 0}


@dataclass
class BusinessMetrics:
    """Business value metrics."""
    energy_saved_j: float = 0.0
    energy_saved_kwh: float = 0.0
    cost_saved_usd: float = 0.0
    co2_saved_kg: float = 0.0
    efficiency_gain_pct: float = 0.0
    tokens_per_dollar: float = 0.0
    roi_pct: float = 0.0  # Return on investment


def calculate_business_value(
    baseline_j_per_token: float,
    metabolic_j_per_token: float,
    total_tokens: int,
    inference_time_hours: float,
) -> BusinessMetrics:
    """Calculate business value of energy savings."""

    # Energy savings
    energy_saved_per_token = baseline_j_per_token - metabolic_j_per_token
    total_energy_saved_j = energy_saved_per_token * total_tokens
    energy_saved_kwh = total_energy_saved_j / 3_600_000  # J to kWh

    # Cost savings
    electricity_saved = energy_saved_kwh * ELECTRICITY_COST_PER_KWH

    # CO2 savings
    co2_saved = energy_saved_kwh * CO2_PER_KWH

    # Efficiency gain
    if baseline_j_per_token > 0:
        efficiency_gain = (baseline_j_per_token - metabolic_j_per_token) / baseline_j_per_token * 100
    else:
        efficiency_gain = 0

    # Tokens per dollar (using metabolic energy)
    if metabolic_j_per_token > 0:
        kwh_per_token = metabolic_j_per_token / 3_600_000
        cost_per_token = kwh_per_token * ELECTRICITY_COST_PER_KWH
        tokens_per_dollar = 1 / cost_per_token if cost_per_token > 0 else 0
    else:
        tokens_per_dollar = 0

    # Simple ROI calculation (savings vs overhead)
    # Assume 5% computational overhead for FEEL loop
    overhead_cost = inference_time_hours * GPU_COST_PER_HOUR * 0.05
    roi = (electricity_saved - overhead_cost) / max(overhead_cost, 0.01) * 100

    return BusinessMetrics(
        energy_saved_j=total_energy_saved_j,
        energy_saved_kwh=energy_saved_kwh,
        cost_saved_usd=electricity_saved,
        co2_saved_kg=co2_saved,
        efficiency_gain_pct=efficiency_gain,
        tokens_per_dollar=tokens_per_dollar,
        roi_pct=roi,
    )


class FEELValidator:
    """Validates each component of the FEEL loop."""

    def __init__(self, telemetry, actuator, model, device):
        self.telemetry = telemetry
        self.actuator = actuator
        self.model = model
        self.device = device
        self.metrics = FEELLoopMetrics()

    def validate_sense(self, num_samples: int = 20) -> Dict:
        """Validate SENSE: Hardware telemetry reading."""
        logger.info("Validating SENSE (telemetry reading)...")

        latencies = []
        readings = []

        for i in range(num_samples):
            start = time.time()
            snap = self.telemetry.read()
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            readings.append({
                'power': snap.power_watts,
                'temp': snap.temp_c,
                'clock': snap.clock_mhz,
                'util': snap.utilization,
            })
            time.sleep(0.05)

        self.metrics.telemetry_latency_ms = np.mean(latencies)
        self.metrics.telemetry_samples = num_samples

        result = {
            'status': 'PASS' if np.mean(latencies) < 100 else 'WARN',
            'avg_latency_ms': np.mean(latencies),
            'power_range': [min(r['power'] for r in readings), max(r['power'] for r in readings)],
            'temp_range': [min(r['temp'] for r in readings), max(r['temp'] for r in readings)],
            'samples': num_samples,
        }

        logger.info(f"  SENSE: {result['status']} - {result['avg_latency_ms']:.1f}ms latency, "
                   f"power {result['power_range'][0]:.0f}-{result['power_range'][1]:.0f}W")
        return result

    def validate_feel(self, num_updates: int = 20) -> Dict:
        """Validate FEEL: Body state computation with EMA."""
        logger.info("Validating FEEL (body state EMA)...")

        states = []
        for i in range(num_updates):
            body_state = self.telemetry.read_body_state()
            states.append(body_state.copy())
            time.sleep(0.05)

        self.metrics.body_state_updates = num_updates

        # Check smoothing (derivatives should be smaller than raw changes)
        states_arr = np.array(states)
        raw_vars = np.var(states_arr[:, :6], axis=0)  # First 6 are values
        deriv_vars = np.var(states_arr[:, 6:], axis=0)  # Last 6 are derivatives

        smoothing_effective = np.mean(deriv_vars) < np.mean(raw_vars)

        result = {
            'status': 'PASS' if smoothing_effective else 'WARN',
            'body_state_dim': len(states[0]),
            'smoothing_effective': smoothing_effective,
            'value_variance': float(np.mean(raw_vars)),
            'derivative_variance': float(np.mean(deriv_vars)),
        }

        logger.info(f"  FEEL: {result['status']} - {result['body_state_dim']}D state, "
                   f"smoothing={'effective' if smoothing_effective else 'weak'}")
        return result

    def validate_regulate(self, batch_size: int = 4) -> Dict:
        """Validate REGULATE: FiLM conditioning effect."""
        logger.info("Validating REGULATE (FiLM conditioning)...")

        self.model.eval()
        x = torch.randint(0, 256, (batch_size, 64), device=self.device)
        telem = torch.rand(batch_size, 12, device=self.device)

        # Test with conditioning ON
        self.model.enable_conditioning(True)
        with torch.no_grad():
            out_on = self.model(x, telem)
            logits_on = out_on['logits'].clone()

        # Test with conditioning OFF
        self.model.enable_conditioning(False)
        with torch.no_grad():
            out_off = self.model(x, telem)
            logits_off = out_off['logits'].clone()

        self.model.enable_conditioning(True)  # Reset

        # Measure conditioning effect
        diff = (logits_on - logits_off).abs().mean().item()
        self.metrics.conditioning_strength = diff

        result = {
            'status': 'PASS' if diff > 0.01 else 'FAIL',
            'conditioning_effect': diff,
            'logits_diff_mean': diff,
            'film_params': sum(p.numel() for p in self.model.get_film_parameters()),
        }

        logger.info(f"  REGULATE: {result['status']} - conditioning effect {diff:.4f}, "
                   f"{result['film_params']:,} FiLM params")
        return result

    def validate_express(self, num_samples: int = 50) -> Dict:
        """Validate EXPRESS: Action output distribution."""
        logger.info("Validating EXPRESS (action output)...")

        self.model.eval()
        actions = []
        entropies = []

        with torch.no_grad():
            for i in range(num_samples):
                x = torch.randint(0, 256, (1, 64), device=self.device)
                telem_np = self.telemetry.read_body_state()
                telem = torch.from_numpy(telem_np).float().to(self.device).unsqueeze(0)

                output = self.model(x, telem)
                action_probs = F.softmax(output['action_logits'], dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()
                actions.append(action)

                # Compute entropy
                entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum().item()
                entropies.append(entropy)

        action_counts = {i: actions.count(i) for i in range(4)}
        self.metrics.actions_taken = action_counts
        self.metrics.action_entropy = np.mean(entropies)

        # Check diversity (not always same action)
        max_action_pct = max(action_counts.values()) / num_samples
        diverse = max_action_pct < 0.9

        result = {
            'status': 'PASS' if diverse else 'WARN',
            'action_distribution': {k: v/num_samples for k, v in action_counts.items()},
            'avg_entropy': np.mean(entropies),
            'max_entropy': 1.386,  # log(4)
            'diversity': diverse,
        }

        logger.info(f"  EXPRESS: {result['status']} - actions {action_counts}, "
                   f"entropy {np.mean(entropies):.2f}/{1.386:.2f}")
        return result

    def validate_hw_change(self) -> Dict:
        """Validate HW_CHANGE: Actual hardware control."""
        logger.info("Validating HW_CHANGE (power control)...")

        from src.metabolic.actuation_unified import MetabolicMode

        results = {}
        latencies = []

        for mode in MetabolicMode:
            start = time.time()
            result = self.actuator.set_metabolic_mode(mode)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            time.sleep(0.3)
            snap = self.telemetry.read()

            results[mode.name] = {
                'success': result.success,
                'latency_ms': latency,
                'power_after': snap.power_watts,
            }

        self.actuator.reset_to_default()

        self.metrics.actuation_latency_ms = np.mean(latencies)
        self.metrics.power_limit_changes = len(MetabolicMode)

        all_success = all(r['success'] for r in results.values())

        result = {
            'status': 'PASS' if all_success else 'FAIL',
            'modes': results,
            'avg_latency_ms': np.mean(latencies),
        }

        logger.info(f"  HW_CHANGE: {result['status']} - "
                   f"{'all modes work' if all_success else 'some modes failed'}, "
                   f"{np.mean(latencies):.1f}ms avg latency")
        return result

    def run_full_validation(self) -> Dict:
        """Run complete FEEL loop validation."""
        logger.info("=" * 60)
        logger.info("FEEL LOOP VALIDATION")
        logger.info("=" * 60)

        results = {
            'sense': self.validate_sense(),
            'feel': self.validate_feel(),
            'regulate': self.validate_regulate(),
            'express': self.validate_express(),
            'hw_change': self.validate_hw_change(),
        }

        # Overall status
        statuses = [r['status'] for r in results.values()]
        if all(s == 'PASS' for s in statuses):
            overall = 'PASS'
        elif any(s == 'FAIL' for s in statuses):
            overall = 'FAIL'
        else:
            overall = 'WARN'

        results['overall'] = overall
        results['metrics'] = asdict(self.metrics)

        logger.info("=" * 60)
        logger.info(f"FEEL LOOP VALIDATION: {overall}")
        logger.info("=" * 60)

        return results


def run_benchmark_comparison(
    model,
    baseline_model,
    eval_loader,
    device,
    telemetry,
    actuator,
    num_samples: int = 100,
) -> Dict:
    """Compare metabolic model against baseline and fixed power modes."""
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPARISON")
    logger.info("=" * 60)

    from src.metabolic.actuation_unified import MetabolicMode

    results = {}

    def measure_mode(model_to_test, mode_name, is_metabolic, fixed_mode=None):
        """Measure performance for a specific configuration."""
        model_to_test.eval()

        total_loss = 0
        total_tokens = 0
        total_energy = 0
        powers = []

        if fixed_mode is not None:
            actuator.set_metabolic_mode(fixed_mode)
            time.sleep(0.5)

        start_time = time.time()

        with torch.no_grad():
            for i, (input_ids, targets) in enumerate(eval_loader):
                if i >= num_samples:
                    break

                input_ids = input_ids.to(device)
                targets = targets.to(device)

                snap_before = telemetry.read()
                batch_start = time.time()

                if is_metabolic:
                    telem_np = telemetry.read_body_state()
                    telem = torch.from_numpy(telem_np).float().to(device)
                    telem = telem.unsqueeze(0).expand(input_ids.size(0), -1)
                    output = model_to_test(input_ids, telem)

                    # Apply learned action (use first sample in batch)
                    action_probs = F.softmax(output['action_logits'][0], dim=-1)
                    action_idx = torch.argmax(action_probs, dim=-1).item()
                    actuator.set_mode_from_action(action_idx)
                else:
                    output = model_to_test(input_ids)

                torch.cuda.synchronize()
                batch_time = time.time() - batch_start
                snap_after = telemetry.read()

                avg_power = (snap_before.power_watts + snap_after.power_watts) / 2
                batch_energy = avg_power * batch_time

                powers.append(avg_power)
                total_energy += batch_energy

                loss = F.cross_entropy(
                    output['logits'].view(-1, model_to_test.config.vocab_size),
                    targets.view(-1),
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += targets.numel()

        elapsed = time.time() - start_time
        actuator.reset_to_default()

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = np.exp(min(avg_loss, 10))
        j_per_token = total_energy / max(total_tokens, 1)

        return {
            'perplexity': ppl,
            'j_per_token': j_per_token,
            'mj_per_token': j_per_token * 1000,
            'tokens_per_sec': total_tokens / elapsed,
            'avg_power': np.mean(powers),
            'total_tokens': total_tokens,
        }

    # 1. Baseline (no conditioning, default power)
    logger.info("Measuring: Baseline (no FEEL)...")
    results['baseline'] = measure_mode(baseline_model, 'baseline', is_metabolic=False)
    logger.info(f"  PPL: {results['baseline']['perplexity']:.2f}, "
               f"mJ/token: {results['baseline']['mj_per_token']:.2f}, "
               f"Power: {results['baseline']['avg_power']:.0f}W")

    # 2. Fixed ECO mode
    logger.info("Measuring: Fixed ECO mode...")
    results['fixed_eco'] = measure_mode(baseline_model, 'fixed_eco', is_metabolic=False,
                                        fixed_mode=MetabolicMode.ECO)
    logger.info(f"  PPL: {results['fixed_eco']['perplexity']:.2f}, "
               f"mJ/token: {results['fixed_eco']['mj_per_token']:.2f}, "
               f"Power: {results['fixed_eco']['avg_power']:.0f}W")

    # 3. Fixed PERFORMANCE mode
    logger.info("Measuring: Fixed PERFORMANCE mode...")
    results['fixed_perf'] = measure_mode(baseline_model, 'fixed_perf', is_metabolic=False,
                                         fixed_mode=MetabolicMode.PERFORMANCE)
    logger.info(f"  PPL: {results['fixed_perf']['perplexity']:.2f}, "
               f"mJ/token: {results['fixed_perf']['mj_per_token']:.2f}, "
               f"Power: {results['fixed_perf']['avg_power']:.0f}W")

    # 4. Metabolic (learned actions)
    logger.info("Measuring: Metabolic FEEL (learned actions)...")
    results['metabolic'] = measure_mode(model, 'metabolic', is_metabolic=True)
    logger.info(f"  PPL: {results['metabolic']['perplexity']:.2f}, "
               f"mJ/token: {results['metabolic']['mj_per_token']:.2f}, "
               f"Power: {results['metabolic']['avg_power']:.0f}W")

    return results


def print_final_report(
    feel_validation: Dict,
    benchmark: Dict,
    business: BusinessMetrics,
    output_dir: Path,
):
    """Print comprehensive final report."""

    print("\n" + "=" * 70)
    print("COMPREHENSIVE FEEL VALIDATION REPORT")
    print("=" * 70)

    print("\n1. FEEL LOOP VALIDATION")
    print("-" * 40)
    for component in ['sense', 'feel', 'regulate', 'express', 'hw_change']:
        status = feel_validation[component]['status']
        icon = '✓' if status == 'PASS' else ('!' if status == 'WARN' else '✗')
        print(f"  {icon} {component.upper()}: {status}")
    print(f"\n  Overall: {feel_validation['overall']}")

    print("\n2. BENCHMARK COMPARISON")
    print("-" * 40)
    print(f"  {'Configuration':<25} {'PPL':<8} {'mJ/tok':<10} {'Power':<10} {'tok/s':<10}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    for name, data in benchmark.items():
        print(f"  {name:<25} {data['perplexity']:<8.2f} {data['mj_per_token']:<10.2f} "
              f"{data['avg_power']:<10.0f} {data['tokens_per_sec']:<10.0f}")

    print("\n3. ENERGY EFFICIENCY ANALYSIS")
    print("-" * 40)

    baseline_mj = benchmark['baseline']['mj_per_token']
    metabolic_mj = benchmark['metabolic']['mj_per_token']
    fixed_eco_mj = benchmark['fixed_eco']['mj_per_token']

    vs_baseline = (metabolic_mj - baseline_mj) / baseline_mj * 100
    vs_fixed_eco = (metabolic_mj - fixed_eco_mj) / fixed_eco_mj * 100

    print(f"  Metabolic vs Baseline:   {vs_baseline:+.1f}%")
    print(f"  Metabolic vs Fixed ECO:  {vs_fixed_eco:+.1f}%")

    if vs_baseline < 0:
        print(f"\n  🎉 ENERGY SAVINGS ACHIEVED: {abs(vs_baseline):.1f}%")

    print("\n4. BUSINESS VALUE (per 1M tokens)")
    print("-" * 40)
    print(f"  Energy saved:      {business.energy_saved_kwh * 1e6:.4f} kWh")
    print(f"  Cost saved:        ${business.cost_saved_usd * 1e6:.4f}")
    print(f"  CO2 saved:         {business.co2_saved_kg * 1e6:.4f} kg")
    print(f"  Efficiency gain:   {business.efficiency_gain_pct:.1f}%")
    print(f"  Tokens per $1:     {business.tokens_per_dollar:,.0f}")

    print("\n5. EXTRAPOLATED ANNUAL SAVINGS (1B tokens/day)")
    print("-" * 40)
    daily_tokens = 1_000_000_000
    yearly_tokens = daily_tokens * 365

    yearly_energy_kwh = business.energy_saved_kwh * yearly_tokens
    yearly_cost_usd = business.cost_saved_usd * yearly_tokens
    yearly_co2_kg = business.co2_saved_kg * yearly_tokens

    print(f"  Energy saved:      {yearly_energy_kwh:,.0f} kWh/year")
    print(f"  Cost saved:        ${yearly_cost_usd:,.0f}/year")
    print(f"  CO2 saved:         {yearly_co2_kg:,.0f} kg/year ({yearly_co2_kg/1000:.1f} tonnes)")

    print("\n" + "=" * 70)

    # Save report
    report = {
        'feel_validation': feel_validation,
        'benchmark': benchmark,
        'business': asdict(business),
        'summary': {
            'energy_change_vs_baseline_pct': vs_baseline,
            'energy_change_vs_fixed_eco_pct': vs_fixed_eco,
            'yearly_savings_usd': yearly_cost_usd,
            'yearly_co2_saved_kg': yearly_co2_kg,
        }
    }

    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")


def main():
    """Run comprehensive validation."""
    from src.metabolic.film_transformer import MetabolicTransformer, BaselineTransformer, MetabolicConfig
    from src.metabolic.telemetry_unified import UnifiedTelemetryReader
    from src.metabolic.actuation_unified import UnifiedActuator

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/z62_validation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Initialize hardware
    telemetry = UnifiedTelemetryReader()
    actuator = UnifiedActuator()

    # Model config
    config = MetabolicConfig(
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        max_seq_len=256,
    )

    # Create models
    metabolic = MetabolicTransformer(config).to(device)
    baseline = BaselineTransformer(config).to(device)

    logger.info(f"Model params: {metabolic.get_num_parameters():,}")

    # Create data
    from src.metabolic.metabolic_trainer import CharDataset
    corpus = "The quick brown fox jumps. " * 10000
    dataset = CharDataset(corpus, config.max_seq_len)
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 1. FEEL Loop Validation
    validator = FEELValidator(telemetry, actuator, metabolic, device)
    feel_results = validator.run_full_validation()

    # 2. Benchmark Comparison
    benchmark = run_benchmark_comparison(
        metabolic, baseline, eval_loader, device, telemetry, actuator,
        num_samples=50
    )

    # 3. Business Value Calculation
    business = calculate_business_value(
        baseline_j_per_token=benchmark['baseline']['j_per_token'],
        metabolic_j_per_token=benchmark['metabolic']['j_per_token'],
        total_tokens=benchmark['metabolic']['total_tokens'],
        inference_time_hours=0.1,
    )

    # 4. Print Report
    print_final_report(feel_results, benchmark, business, output_dir)

    return {
        'feel': feel_results,
        'benchmark': benchmark,
        'business': asdict(business),
    }


if __name__ == "__main__":
    results = main()
