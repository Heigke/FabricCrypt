#!/usr/bin/env python3
"""
Z82: Embodied Expression Validation
====================================

This script validates the interoceptive latent space and expression modulation
by comparing:

1. Baseline: Standard generation (no modulation)
2. Embodied: Generation with body-state-aware expression modulation

Key metrics:
- Energy efficiency (J/token)
- Latency (TBT p50/p95)
- Expression distribution (how much modulation affects sampling)
- Quality preservation (perplexity should be similar)

The hypothesis: Embodied expression allows the model to adapt its generation
style based on hardware state WITHOUT changing semantic content.

Author: FEEL Research Team
Date: 2026-01-20
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.atom import (
    AtomConfig,
    BodyState,
    InferencePhase,
    ActionLevel,
    create_sensor,
    create_actuator,
    BodyStateTracker,
    TokenGenerator,
    SyncMode,
    # Interoception
    BodyLatent,
    ExpressionMode,
    ExpressionParams,
    EmbodiedExpressionController,
    # Multi-timescale controller
    MultiScaleController,
)
from src.atom.decide import create_controller, FixedController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EmbodiedValidationConfig:
    """Configuration for embodied expression validation."""
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"

    # Validation params
    num_warmup_tokens: int = 20
    num_eval_tokens: int = 200
    num_requests: int = 5
    prompt: str = "Explain the concept of energy efficiency in computing. Discuss:"

    # SLO thresholds
    tbt_slo_ms: float = 50.0
    ttft_slo_ms: float = 500.0
    temp_threshold_c: float = 75.0

    # Embodied expression settings
    modulation_strengths: List[float] = None  # Test different strengths

    # Controller for hardware (we use multiscale for best results)
    hw_controller: str = "multiscale"

    # Output
    output_dir: str = "results/z82_embodied"

    def __post_init__(self):
        if self.modulation_strengths is None:
            self.modulation_strengths = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass
class ExpressionMetrics:
    """Metrics for a single expression modulation level."""
    modulation_strength: float
    num_tokens: int
    total_time_ms: float
    ttft_ms: float
    tbt_p50_ms: float
    tbt_p95_ms: float
    throughput_tps: float
    total_energy_j: float
    j_per_token: float
    avg_power_w: float
    avg_temp_c: float

    # Expression-specific metrics
    mode_distribution: Dict[str, int]
    avg_temperature: float  # Sampling temperature
    avg_top_k: float
    avg_strain: float
    avg_urgency: float
    avg_margin: float

    # Quality metrics
    avg_entropy: float  # Token entropy (diversity)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmbodiedExpressionValidator:
    """Validates embodied expression with different modulation strengths."""

    def __init__(
        self,
        config: EmbodiedValidationConfig,
        model=None,
        tokenizer=None,
    ):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Load model if not provided
        if model is None:
            self._load_model()
        else:
            self.model = model
            self.tokenizer = tokenizer

        # Initialize components
        self._init_hardware()

    def _load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def _init_hardware(self):
        """Initialize hardware monitoring and control."""
        self.atom_config = AtomConfig(
            rate_limit_ms=50,
            latency_slo_ms=self.config.tbt_slo_ms,
            thermal_margin_c=5.0,
        )

        self.sensor = create_sensor(device_id=0)
        self.actuator = create_actuator(device_id=0, config=self.atom_config)
        self.body_tracker = BodyStateTracker(config=self.atom_config)

        # Hardware controller (multiscale recommended)
        if self.config.hw_controller == "multiscale":
            self.hw_controller = MultiScaleController()
        else:
            self.hw_controller = create_controller(
                self.config.hw_controller,
                config=self.atom_config
            )

        logger.info(f"Hardware initialized: {self.sensor.get_vendor().name}")
        logger.info(f"Actuator mode: {self.actuator.get_actuation_mode()}")

    def validate_modulation_strength(
        self,
        modulation_strength: float,
    ) -> ExpressionMetrics:
        """
        Run validation at a specific modulation strength.

        Args:
            modulation_strength: How much body state affects generation (0-1)

        Returns:
            ExpressionMetrics for this configuration
        """
        logger.info(f"Validating modulation_strength={modulation_strength:.2f}")

        # Create expression controller
        expression_controller = EmbodiedExpressionController(
            j_per_token_target=0.1,
            tbt_slo_ms=self.config.tbt_slo_ms,
            temp_safe_c=self.config.temp_threshold_c,
            modulation_strength=modulation_strength,
            enabled=(modulation_strength > 0),
        )

        # Reset hardware to default
        self.actuator.reset_to_default()
        time.sleep(0.5)

        # Warm up
        logger.info(f"Warming up ({self.config.num_warmup_tokens} tokens)...")
        self._generate_tokens(
            self.config.num_warmup_tokens,
            expression_controller,
            collect_metrics=False,
        )

        # Evaluation
        logger.info(f"Evaluating ({self.config.num_eval_tokens} tokens x {self.config.num_requests} requests)...")

        all_tbts = []
        all_ttfts = []
        all_temps = []
        all_powers = []
        mode_counts = {}
        temperatures = []
        top_ks = []
        strains = []
        urgencies = []
        margins = []
        entropies = []

        start_energy = self.sensor.read().energy_joules
        start_time = time.time()

        for req_idx in range(self.config.num_requests):
            # Reset expression controller for new request
            expression_controller.reset()
            self.body_tracker.reset()

            # Get initial snapshot
            snap = self.sensor.read()
            self.body_tracker.reset_energy_window(snap.energy_joules)

            # Generate tokens for this request
            is_first_token = True
            for tok_idx in range(self.config.num_eval_tokens):
                # Sense
                snap_pre = self.sensor.read()

                # Feel
                phase = InferencePhase.PREFILL if is_first_token else InferencePhase.DECODE
                body_state = self.body_tracker.update(
                    snap_pre,
                    tokens_generated=1,
                    phase=phase,
                    latency_ms=all_tbts[-1] if all_tbts else 10.0,
                    actuation_applied=False,
                )

                # Get expression parameters from body state
                expr_params = expression_controller.step(body_state)

                # Track mode distribution
                mode_name = expr_params.mode.name
                mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1

                # Track expression params
                temperatures.append(expr_params.temperature)
                top_ks.append(expr_params.top_k)

                # Track latent
                latent = expression_controller.get_current_latent()
                if latent:
                    strains.append(latent.strain)
                    urgencies.append(latent.urgency)
                    margins.append(latent.margin)

                # Decide hardware action
                power_min, power_max, _ = self.actuator.get_power_limits()
                hw_action = self.hw_controller.decide(body_state, power_min, power_max)

                # Actuate
                self.actuator.apply(hw_action)

                # Generate token with expression parameters
                gen_start = time.time()
                output = self._generate_single_token(expr_params)
                gen_end = time.time()

                latency_ms = (gen_end - gen_start) * 1000

                # Track entropy (token diversity proxy)
                if hasattr(output, 'scores') and output.scores:
                    # Get logits entropy
                    logits = output.scores[0][0]  # [vocab_size]
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                    entropies.append(entropy)

                # Sense after
                snap_post = self.sensor.read()

                # Track metrics
                if is_first_token:
                    all_ttfts.append(latency_ms)
                    is_first_token = False
                else:
                    all_tbts.append(latency_ms)

                all_temps.append(snap_post.temp_c)
                all_powers.append(snap_post.power_watts)

        end_time = time.time()
        end_energy = self.sensor.read().energy_joules

        total_time_ms = (end_time - start_time) * 1000
        total_energy_j = end_energy - start_energy
        total_tokens = self.config.num_eval_tokens * self.config.num_requests

        # Compute metrics
        metrics = ExpressionMetrics(
            modulation_strength=modulation_strength,
            num_tokens=total_tokens,
            total_time_ms=total_time_ms,
            ttft_ms=np.mean(all_ttfts) if all_ttfts else 0.0,
            tbt_p50_ms=np.percentile(all_tbts, 50) if all_tbts else 0.0,
            tbt_p95_ms=np.percentile(all_tbts, 95) if all_tbts else 0.0,
            throughput_tps=total_tokens / (total_time_ms / 1000),
            total_energy_j=total_energy_j,
            j_per_token=total_energy_j / total_tokens,
            avg_power_w=np.mean(all_powers) if all_powers else 0.0,
            avg_temp_c=np.mean(all_temps) if all_temps else 0.0,
            mode_distribution=mode_counts,
            avg_temperature=np.mean(temperatures) if temperatures else 1.0,
            avg_top_k=np.mean(top_ks) if top_ks else 50,
            avg_strain=np.mean(strains) if strains else 0.5,
            avg_urgency=np.mean(urgencies) if urgencies else 0.0,
            avg_margin=np.mean(margins) if margins else 1.0,
            avg_entropy=np.mean(entropies) if entropies else 0.0,
        )

        # Log stats
        expr_stats = expression_controller.get_stats()
        logger.info(f"  Tokens: {total_tokens}")
        logger.info(f"  J/token: {metrics.j_per_token:.4f}")
        logger.info(f"  TBT p50/p95: {metrics.tbt_p50_ms:.1f}/{metrics.tbt_p95_ms:.1f} ms")
        logger.info(f"  Avg temp: {metrics.avg_temp_c:.1f}C")
        logger.info(f"  Mode dist: {mode_counts}")
        logger.info(f"  Avg sampling temp: {metrics.avg_temperature:.3f}")

        return metrics

    def _generate_single_token(
        self,
        expr_params: ExpressionParams,
    ) -> Any:
        """Generate a single token with expression parameters."""
        # Encode prompt
        inputs = self.tokenizer(
            self.config.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Apply expression parameters to generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=True,
                temperature=max(0.01, expr_params.temperature),
                top_k=max(1, expr_params.top_k),
                top_p=min(0.99, max(0.1, expr_params.top_p)),
                repetition_penalty=expr_params.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        return outputs

    def _generate_tokens(
        self,
        num_tokens: int,
        expression_controller: EmbodiedExpressionController,
        collect_metrics: bool = False,
    ) -> None:
        """Generate tokens (for warmup)."""
        for _ in range(num_tokens):
            snap = self.sensor.read()
            body_state = self.body_tracker.update(snap, tokens_generated=1)
            expr_params = expression_controller.step(body_state)
            self._generate_single_token(expr_params)

    def run_validation(self) -> Dict[str, Any]:
        """Run full validation across all modulation strengths."""
        results = {
            'config': asdict(self.config),
            'results': [],
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        }

        for strength in self.config.modulation_strengths:
            metrics = self.validate_modulation_strength(strength)
            results['results'].append(metrics.to_dict())

        # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = results['timestamp']
        output_path = output_dir / f"embodied_validation_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print summary table."""
        print("\n" + "=" * 100)
        print("EMBODIED EXPRESSION VALIDATION RESULTS")
        print("=" * 100)
        print(f"{'Modulation':<12} {'J/token':>10} {'TBT p50':>10} {'TBT p95':>10} "
              f"{'Throughput':>12} {'Avg Temp':>10} {'Sampling T':>10} {'Mode Dist':>25}")
        print("-" * 100)

        baseline_jpt = None
        for r in results['results']:
            strength = r['modulation_strength']
            jpt = r['j_per_token']

            if baseline_jpt is None and strength == 0.0:
                baseline_jpt = jpt

            # Format mode distribution
            modes = r['mode_distribution']
            total = sum(modes.values())
            mode_str = ", ".join([f"{k[:3]}:{v/total*100:.0f}%" for k, v in modes.items()])

            print(f"{strength:<12.2f} {jpt:>10.4f} {r['tbt_p50_ms']:>10.2f} {r['tbt_p95_ms']:>10.2f} "
                  f"{r['throughput_tps']:>12.1f} {r['avg_temp_c']:>10.1f} {r['avg_temperature']:>10.3f} "
                  f"{mode_str:>25}")

        print("=" * 100)

        if baseline_jpt and len(results['results']) > 1:
            # Find best efficiency
            best = min(results['results'], key=lambda x: x['j_per_token'])
            if best['modulation_strength'] > 0:
                savings = (baseline_jpt - best['j_per_token']) / baseline_jpt * 100
                print(f"\nBest efficiency: modulation_strength={best['modulation_strength']:.2f} "
                      f"({best['j_per_token']:.4f} J/token)")
                print(f"Energy savings vs baseline: {savings:.1f}%")

    def shutdown(self):
        """Clean up."""
        self.actuator.reset_to_default()


def main():
    parser = argparse.ArgumentParser(description='Z82: Embodied Expression Validation')
    parser.add_argument('--model', default="Qwen/Qwen2.5-0.5B-Instruct", help='Model name')
    parser.add_argument('--tokens', type=int, default=200, help='Tokens per request')
    parser.add_argument('--requests', type=int, default=5, help='Number of requests')
    parser.add_argument('--output', default="results/z82_embodied", help='Output directory')
    parser.add_argument('--hw-controller', default="multiscale", help='Hardware controller')
    parser.add_argument('--quick', action='store_true', help='Quick validation')
    args = parser.parse_args()

    config = EmbodiedValidationConfig(
        model_name=args.model,
        num_eval_tokens=args.tokens if not args.quick else 50,
        num_requests=args.requests if not args.quick else 2,
        output_dir=args.output,
        hw_controller=args.hw_controller,
        modulation_strengths=[0.0, 0.5, 1.0] if args.quick else [0.0, 0.25, 0.5, 0.75, 1.0],
    )

    validator = EmbodiedExpressionValidator(config)

    try:
        results = validator.run_validation()
        print(json.dumps({
            'best_modulation': min(results['results'], key=lambda x: x['j_per_token'])['modulation_strength'],
            'baseline_jpt': results['results'][0]['j_per_token'],
            'best_jpt': min(r['j_per_token'] for r in results['results']),
        }))
    finally:
        validator.shutdown()


if __name__ == "__main__":
    main()
