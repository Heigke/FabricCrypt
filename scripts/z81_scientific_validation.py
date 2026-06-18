#!/usr/bin/env python3
"""
Z81 Scientific Validation Experiment
=====================================

Comprehensive validation of FEEL controllers against faithful baselines:
- GreenLLM (phase-split dual-loop) per arXiv:2508.16449
- throttLL'eM (predictive scaling) per arXiv:2408.05235
- Fixed baselines (ECO, MED, PERF)
- Bandit controller (online learning)
- Multiscale controller (fast/slow loops)

This experiment:
1. Runs all controllers on the same workload
2. Measures: J/token, TTFT, TBT, throughput, thermal
3. Compares against scientifically faithful baselines
4. Generates publication-ready comparison data

Usage:
    python scripts/z81_scientific_validation.py --quick
    python scripts/z81_scientific_validation.py --full --controllers all

Author: FEEL Research Team
Date: 2026-01-20
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.atom.atom import TokenMetabolismStep
from src.atom.schema import AtomConfig, ActionLevel, InferencePhase
from src.atom.decide import (
    Controller, FixedController, BanditController,
    GreenLLMController, ThrottLLeMController, create_controller
)
from src.atom.multiscale import MultiScaleController, MultiScaleConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for validation experiment."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"
    num_warmup_tokens: int = 20
    num_eval_tokens: int = 200
    num_requests: int = 5
    prompt: str = "Explain the concept of energy efficiency in computing. Discuss:"
    temp_threshold_c: float = 75.0
    tbt_slo_ms: float = 50.0
    ttft_slo_ms: float = 500.0
    output_dir: str = "results/z81_validation"


@dataclass
class ValidationResult:
    """Result from a single validation run."""
    controller_name: str
    coupling_mode: str
    num_tokens: int

    # Timing metrics
    total_time_ms: float
    ttft_ms: float
    tbt_p50_ms: float
    tbt_p95_ms: float
    tbt_p99_ms: float
    throughput_tps: float

    # Energy metrics
    total_energy_j: float
    j_per_token: float
    avg_power_w: float

    # Thermal metrics
    avg_temp_c: float
    max_temp_c: float
    time_above_threshold_frac: float
    throttle_residency_frac: float

    # SLO metrics
    tbt_slo_violations: int
    tbt_slo_violation_rate: float
    ttft_slo_violations: int

    # Phase metrics (new)
    prefill_count: int
    decode_count: int
    prefill_energy_j: float
    decode_energy_j: float

    # Controller stats
    controller_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_all_controllers(config: ExperimentConfig) -> Dict[str, Controller]:
    """Create all controllers for comparison."""
    atom_config = AtomConfig(
        control_interval_ms=100,
        rate_limit_ms=50,
        latency_slo_ms=config.tbt_slo_ms,
        thermal_margin_c=5.0,
    )

    controllers = {
        # Fixed baselines
        'fixed_eco': FixedController(level=ActionLevel.ECO),
        'fixed_med': FixedController(level=ActionLevel.MED),
        'fixed_perf': FixedController(level=ActionLevel.PERF),

        # Faithful baselines (per papers)
        'greenllm': GreenLLMController(
            tbt_slo_ms=config.tbt_slo_ms,
            ttft_slo_ms=config.ttft_slo_ms,
            slo_margin=0.1,
        ),
        'throttllem': ThrottLLeMController(
            tbt_slo_ms=config.tbt_slo_ms,
            slo_margin=0.2,
        ),

        # FEEL controllers
        'bandit': BanditController(config=atom_config, epsilon=0.1),
        'multiscale': MultiScaleController(config=MultiScaleConfig(
            fast_interval_ms=30.0,
            slow_interval_ms=500.0,
            temp_setpoint_frac=config.temp_threshold_c / 100.0,  # Convert to fraction
        )),
    }

    return controllers


def run_validation(
    config: ExperimentConfig,
    controller_name: str,
    controller: Controller,
    model: Any,
    tokenizer: Any,
    device_id: int = 0,
) -> Optional[ValidationResult]:
    """Run validation with a specific controller."""
    logger.info(f"Running validation: {controller_name}")

    atom_config = AtomConfig(
        control_interval_ms=100,
        rate_limit_ms=50,
        latency_slo_ms=config.tbt_slo_ms,
        device_id=device_id,
    )

    atom = TokenMetabolismStep(config=atom_config, controller=controller)

    try:
        atom.initialize(model, tokenizer, device_id=device_id)
    except Exception as e:
        logger.warning(f"Atom initialization failed: {e}, trying fallback")
        try:
            atom.initialize(model, tokenizer, device_id=0)
        except Exception as e2:
            logger.error(f"Validation failed for {controller_name}: {e2}")
            return None

    # Warmup
    logger.info(f"Warming up ({config.num_warmup_tokens} tokens)...")
    input_ids = tokenizer.encode(config.prompt, return_tensors="pt")
    for _ in range(config.num_warmup_tokens):
        atom.step(input_ids)
        input_ids = None

    # Reset for eval
    atom.reset_metrics()

    # Evaluation
    logger.info(f"Evaluating ({config.num_eval_tokens} tokens x {config.num_requests} requests)...")

    all_latencies = []  # All latencies (for total time)
    decode_tbts = []    # DECODE-ONLY TBTs (for percentile computation)
    all_ttfts = []
    all_powers = []
    all_temps = []
    all_energies = []
    throttle_samples = 0
    throttle_count = 0
    above_threshold_count = 0
    tbt_violations = 0
    ttft_violations = 0
    prefill_count = 0
    decode_count = 0
    prefill_energy = 0.0
    decode_energy = 0.0

    for req_idx in range(config.num_requests):
        input_ids = tokenizer.encode(config.prompt, return_tensors="pt")
        first_token = True

        for tok_idx in range(config.num_eval_tokens):
            record = atom.step(input_ids)
            input_ids = None

            if record:
                # Timing
                latency_ms = record.tokens.latency_ms
                all_latencies.append(latency_ms)

                if first_token:
                    # First token is TTFT (prefill) - don't include in TBT stats
                    all_ttfts.append(latency_ms)
                    if latency_ms > config.ttft_slo_ms:
                        ttft_violations += 1
                    first_token = False
                else:
                    # Subsequent tokens are decode - track for TBT percentiles
                    decode_tbts.append(latency_ms)
                    if latency_ms > config.tbt_slo_ms:
                        tbt_violations += 1

                # Energy
                all_energies.append(record.energy_delta_joules)

                # Power & thermal
                if record.sense_post:
                    all_powers.append(record.sense_post.power_watts)
                    all_temps.append(record.sense_post.temp_c)
                    throttle_samples += 1
                    if record.sense_post.is_throttled:
                        throttle_count += 1
                    if record.sense_post.temp_c > config.temp_threshold_c:
                        above_threshold_count += 1

                # Phase tracking
                if record.body_state:
                    if record.body_state.phase == InferencePhase.PREFILL:
                        prefill_count += 1
                        prefill_energy += record.energy_delta_joules
                    else:
                        decode_count += 1
                        decode_energy += record.energy_delta_joules

    # Compute metrics
    total_tokens = len(all_latencies)
    if total_tokens == 0:
        logger.warning(f"No tokens generated for {controller_name}")
        atom.shutdown()
        return None

    total_time_ms = sum(all_latencies)
    total_energy_j = sum(all_energies)

    # TBT percentiles computed ONLY on decode tokens (excludes TTFT)
    # This is scientifically correct - TTFT and TBT are different metrics
    if decode_tbts:
        tbt_p50 = np.percentile(decode_tbts, 50)
        tbt_p95 = np.percentile(decode_tbts, 95)
        tbt_p99 = np.percentile(decode_tbts, 99)
    else:
        tbt_p50 = tbt_p95 = tbt_p99 = 0.0

    result = ValidationResult(
        controller_name=controller_name,
        coupling_mode="on",
        num_tokens=total_tokens,
        total_time_ms=total_time_ms,
        ttft_ms=np.mean(all_ttfts) if all_ttfts else 0.0,
        tbt_p50_ms=tbt_p50,
        tbt_p95_ms=tbt_p95,
        tbt_p99_ms=tbt_p99,
        throughput_tps=total_tokens / (total_time_ms / 1000.0) if total_time_ms > 0 else 0.0,
        total_energy_j=total_energy_j,
        j_per_token=total_energy_j / total_tokens,
        avg_power_w=np.mean(all_powers) if all_powers else 0.0,
        avg_temp_c=np.mean(all_temps) if all_temps else 0.0,
        max_temp_c=max(all_temps) if all_temps else 0.0,
        time_above_threshold_frac=above_threshold_count / max(1, throttle_samples),
        throttle_residency_frac=throttle_count / max(1, throttle_samples),
        tbt_slo_violations=tbt_violations,
        tbt_slo_violation_rate=tbt_violations / max(1, total_tokens - len(all_ttfts)),
        ttft_slo_violations=ttft_violations,
        prefill_count=prefill_count,
        decode_count=decode_count,
        prefill_energy_j=prefill_energy,
        decode_energy_j=decode_energy,
        controller_stats=controller.get_stats(),
    )

    atom.shutdown()

    logger.info(f"  Total tokens: {total_tokens} (prefill: {len(all_ttfts)}, decode: {len(decode_tbts)})")
    logger.info(f"  J/token: {result.j_per_token:.4f}")
    logger.info(f"  TBT p50/p95 (decode-only): {result.tbt_p50_ms:.1f}/{result.tbt_p95_ms:.1f} ms")
    logger.info(f"  TTFT: {result.ttft_ms:.1f} ms")
    logger.info(f"  SLO violations: {tbt_violations}/{len(decode_tbts)} TBT ({result.tbt_slo_violation_rate:.1%}), {ttft_violations} TTFT")
    logger.info(f"  Avg temp: {result.avg_temp_c:.1f}C")

    return result


def numpy_serializer(x):
    """JSON serializer for numpy types."""
    if isinstance(x, (np.bool_,)):
        return bool(x)
    elif isinstance(x, (np.floating,)):
        return float(x)
    elif isinstance(x, (np.integer,)):
        return int(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    parser = argparse.ArgumentParser(description="Z81 Scientific Validation")
    parser.add_argument("--quick", action="store_true", help="Quick validation (fewer tokens)")
    parser.add_argument("--full", action="store_true", help="Full validation (more tokens)")
    parser.add_argument("--controllers", nargs="+", default=["fixed_eco", "fixed_med", "greenllm", "throttllem", "multiscale"],
                       help="Controllers to test")
    parser.add_argument("--output-dir", default="results/z81_validation", help="Output directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model to use")
    args = parser.parse_args()

    # Configure experiment
    config = ExperimentConfig(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    if args.quick:
        config.num_warmup_tokens = 10
        config.num_eval_tokens = 100
        config.num_requests = 3
    elif args.full:
        config.num_warmup_tokens = 50
        config.num_eval_tokens = 500
        config.num_requests = 10

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Use same config as z80 which worked
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.device != "cpu" else torch.float32,
        device_map=config.device if config.device != "cpu" else None,
        trust_remote_code=True,
    )

    # Create controllers
    all_controllers = create_all_controllers(config)

    # Filter to requested controllers
    if "all" in args.controllers:
        controllers = all_controllers
    else:
        controllers = {k: v for k, v in all_controllers.items() if k in args.controllers}

    logger.info(f"Testing controllers: {list(controllers.keys())}")

    # Run validation
    results = []
    for name, controller in controllers.items():
        result = run_validation(config, name, controller, model, tokenizer)
        if result:
            results.append(result)

            # Save individual result
            result_file = Path(config.output_dir) / f"{name}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=numpy_serializer)

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = Path(config.output_dir) / f"validation_{timestamp}.json"
    with open(combined_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'results': [r.to_dict() for r in results],
            'timestamp': timestamp,
        }, f, indent=2, default=numpy_serializer)

    logger.info(f"\nResults saved to: {combined_file}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("SCIENTIFIC VALIDATION RESULTS")
    print("=" * 100)
    print(f"{'Controller':<15} {'J/token':>10} {'TBT p50':>10} {'TBT p95':>10} {'TTFT':>10} "
          f"{'SLO Viol':>10} {'Temp':>8} {'Throughput':>12}")
    print("-" * 100)

    # Sort by J/token
    results.sort(key=lambda r: r.j_per_token)

    for r in results:
        print(f"{r.controller_name:<15} {r.j_per_token:>10.4f} {r.tbt_p50_ms:>10.2f} "
              f"{r.tbt_p95_ms:>10.2f} {r.ttft_ms:>10.2f} {r.tbt_slo_violation_rate:>9.1%} "
              f"{r.avg_temp_c:>7.1f}C {r.throughput_tps:>10.1f} tps")

    print("=" * 100)

    # Summary statistics
    if len(results) >= 2:
        best = results[0]
        worst = results[-1]
        savings = (worst.j_per_token - best.j_per_token) / worst.j_per_token * 100
        print(f"\nBest efficiency: {best.controller_name} ({best.j_per_token:.4f} J/token)")
        print(f"Worst efficiency: {worst.controller_name} ({worst.j_per_token:.4f} J/token)")
        print(f"Energy savings: {savings:.1f}%")

        # Find best SLO compliance
        slo_sorted = sorted(results, key=lambda r: r.tbt_slo_violation_rate)
        print(f"Best SLO compliance: {slo_sorted[0].controller_name} "
              f"({slo_sorted[0].tbt_slo_violation_rate:.1%} violations)")

    # Return summary for scripts
    return {
        "best_efficiency": results[0].controller_name if results else "N/A",
        "best_latency": min(results, key=lambda r: r.tbt_p50_ms).controller_name if results else "N/A",
        "best_slo": min(results, key=lambda r: r.tbt_slo_violation_rate).controller_name if results else "N/A",
        "all_results": len(results),
    }


if __name__ == "__main__":
    summary = main()
    print(json.dumps(summary))
