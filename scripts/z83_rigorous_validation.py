#!/usr/bin/env python3
"""
Z83: Rigorous Embodied Validation
==================================

This script provides SCIENTIFICALLY RIGOROUS validation of the FEEL architecture.
It fixes the issues in Z82:

1. PROPER DECODE LOOP: Uses KV cache, no re-encoding per token
2. SEPARATED METRICS: Prefill vs decode metrics tracked separately
3. QUALITY MEASUREMENT: Tracks output perplexity/entropy for quality drift
4. PER-MACHINE CALIBRATION: Uses MachineCalibration for accurate BodyLatent
5. DELTA ENERGY TRACKING: Uses incremental energy, not running averages

Key insight from analysis:
- Z82 showed 10.1% savings but expression params never changed (always BALANCED)
- This was due to: wrong calibration (0.1 J/token vs real 1.2-2.6), re-encode overhead
- This script tests whether expression modulation ACTUALLY affects generation

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
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

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
    # Interoception with calibration
    BodyLatent,
    ExpressionMode,
    ExpressionParams,
    MachineCalibration,
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
class RigorousConfig:
    """Configuration for rigorous validation."""
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cuda"

    # Validation params
    num_warmup_tokens: int = 50
    num_prefill_tokens: int = 100  # Fixed prompt length
    num_decode_tokens: int = 200   # Tokens to generate
    num_requests: int = 5

    # Prompt for generation
    prompt: str = "Write a detailed explanation of how computers process information. Include discussion of:"

    # SLO thresholds
    tbt_slo_ms: float = 50.0
    ttft_slo_ms: float = 500.0
    temp_threshold_c: float = 75.0

    # Modulation strengths to test
    modulation_strengths: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])

    # Machine type for calibration
    machine_type: str = "amd_apu"  # amd_apu, amd_dgpu, nvidia_gpu

    # Output
    output_dir: str = "results/z83_rigorous"

    # Quality measurement
    measure_quality: bool = True


@dataclass
class PhaseSeparatedMetrics:
    """Metrics separated by inference phase."""
    # Prefill metrics
    prefill_tokens: int = 0
    prefill_time_ms: float = 0.0
    prefill_energy_j: float = 0.0
    prefill_j_per_token: float = 0.0

    # Decode metrics (the real test)
    decode_tokens: int = 0
    decode_time_ms: float = 0.0
    decode_energy_j: float = 0.0
    decode_j_per_token: float = 0.0
    decode_tbt_p50_ms: float = 0.0
    decode_tbt_p95_ms: float = 0.0
    decode_tbt_p99_ms: float = 0.0
    decode_throughput_tps: float = 0.0

    # Overall
    total_tokens: int = 0
    total_time_ms: float = 0.0
    total_energy_j: float = 0.0
    j_per_token: float = 0.0

    # Thermal/power
    avg_temp_c: float = 0.0
    max_temp_c: float = 0.0
    avg_power_w: float = 0.0

    # SLO compliance
    decode_slo_violations: int = 0
    decode_slo_violation_rate: float = 0.0


@dataclass
class ExpressionMetrics:
    """Expression-specific metrics."""
    mode_distribution: Dict[str, int] = field(default_factory=dict)
    mode_transitions: int = 0  # How often mode changed

    # Average expression params (only during decode)
    avg_temperature: float = 1.0
    avg_top_k: float = 50.0
    avg_top_p: float = 0.9
    temperature_variance: float = 0.0

    # Latent averages
    avg_strain: float = 0.5
    avg_urgency: float = 0.0
    avg_debt: float = 0.0
    avg_margin: float = 1.0

    # Did expression actually change?
    params_changed: bool = False
    num_unique_temperatures: int = 1


@dataclass
class QualityMetrics:
    """Quality drift measurement."""
    # Output entropy (diversity)
    avg_token_entropy: float = 0.0
    entropy_variance: float = 0.0

    # Repetition
    repetition_ratio: float = 0.0  # Fraction of repeated n-grams

    # Length distribution
    avg_output_length: int = 0

    # Perplexity proxy (if measurable)
    avg_log_prob: float = 0.0


@dataclass
class ValidationResult:
    """Complete result for one modulation strength."""
    modulation_strength: float
    phase_metrics: PhaseSeparatedMetrics
    expression_metrics: ExpressionMetrics
    quality_metrics: QualityMetrics
    calibration_used: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'modulation_strength': self.modulation_strength,
            'phase_metrics': asdict(self.phase_metrics),
            'expression_metrics': asdict(self.expression_metrics),
            'quality_metrics': asdict(self.quality_metrics),
            'calibration_used': self.calibration_used,
        }


class RigorousValidator:
    """
    Rigorous validation with proper KV-cache decode loop.
    """

    def __init__(self, config: RigorousConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Get calibration for this machine
        self.calibration = self._get_calibration()

        # Load model
        self._load_model()

        # Initialize hardware
        self._init_hardware()

    def _get_calibration(self) -> MachineCalibration:
        """Get machine-specific calibration."""
        if self.config.machine_type == "amd_apu":
            return MachineCalibration.for_amd_apu()
        elif self.config.machine_type == "amd_dgpu":
            return MachineCalibration.for_amd_dgpu()
        elif self.config.machine_type == "nvidia_gpu":
            return MachineCalibration.for_nvidia_gpu()
        else:
            logger.warning(f"Unknown machine type: {self.config.machine_type}, using AMD APU defaults")
            return MachineCalibration.for_amd_apu()

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

        # Try GPU first, fall back to CPU if HIP/CUDA fails
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            # Quick test to verify GPU works
            test_input = self.tokenizer("test", return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(**test_input)
            logger.info("Model loaded on GPU successfully")
        except Exception as e:
            logger.warning(f"GPU loading failed ({e}), falling back to CPU")
            self.device = torch.device("cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            logger.info("Model loaded on CPU (fallback)")

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

        # Use multiscale controller for hardware
        self.hw_controller = MultiScaleController()

        logger.info(f"Hardware initialized: {self.sensor.get_vendor().name}")
        logger.info(f"Actuator mode: {self.actuator.get_actuation_mode()}")
        logger.info(f"Calibration: {asdict(self.calibration)}")

    def _generate_with_kv_cache(
        self,
        expression_controller: EmbodiedExpressionController,
        num_tokens: int,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Generate tokens using proper KV-cache autoregressive decoding.

        This is the CORRECT way to test embodied expression:
        - Encode prompt ONCE
        - Generate tokens one-by-one with KV cache
        - Apply expression params to EACH token's sampling

        Returns:
            Tuple of (generated_token_ids, per_token_records)
        """
        # Encode prompt
        inputs = self.tokenizer(
            self.config.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Track per-token data
        records = []
        generated_tokens = []
        past_key_values = None

        # Reset body tracker
        self.body_tracker.reset()
        snap = self.sensor.read()
        self.body_tracker.reset_energy_window(snap.energy_joules)
        last_energy = snap.energy_joules

        for tok_idx in range(num_tokens):
            # === SENSE (before generation) ===
            snap_pre = self.sensor.read()
            energy_before = snap_pre.energy_joules

            # === FEEL (body state) - use PREVIOUS token's energy delta ===
            phase = InferencePhase.PREFILL if tok_idx == 0 else InferencePhase.DECODE

            # Calculate delta energy from PREVIOUS token (we don't have this token's energy yet)
            prev_delta_energy_j = energy_before - last_energy if energy_before > 0 and last_energy > 0 else 0.0

            body_state = self.body_tracker.update(
                snap_pre,
                tokens_generated=1,
                phase=phase,
                latency_ms=records[-1]['latency_ms'] if records else 10.0,
                actuation_applied=False,
            )

            # Get expression params - ALWAYS encode (even when disabled) for logging
            # The modulator will return defaults if disabled, but we still want latent values
            expr_params = expression_controller.step(body_state, delta_energy_j=prev_delta_energy_j)

            # === DECIDE (hardware action) ===
            power_min, power_max, _ = self.actuator.get_power_limits()
            hw_action = self.hw_controller.decide(body_state, power_min, power_max)

            # === ACTUATE (hardware) ===
            self.actuator.apply(hw_action)

            # === GENERATE (single token with KV cache) ===
            gen_start = time.time()

            with torch.no_grad():
                if tok_idx == 0:
                    # First token: full forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                else:
                    # Subsequent tokens: use KV cache
                    outputs = self.model(
                        input_ids=next_token_id.unsqueeze(0),
                        attention_mask=torch.cat([attention_mask, torch.ones(1, 1, device=self.device)], dim=1),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=self.device)], dim=1)
                    logits = outputs.logits[:, -1, :]

                # Apply expression-modulated sampling
                next_token_id, token_entropy, log_prob = self._sample_with_params(logits, expr_params)

            gen_end = time.time()
            latency_ms = (gen_end - gen_start) * 1000

            # === SENSE (after generation) ===
            snap_post = self.sensor.read()

            # Calculate this token's energy
            token_energy_j = snap_post.energy_joules - snap_pre.energy_joules if snap_post.energy_joules > 0 else 0.0

            # Store record
            generated_tokens.append(next_token_id.item())
            records.append({
                'token_idx': tok_idx,
                'phase': phase.name,
                'latency_ms': latency_ms,
                'energy_j': token_energy_j,
                'temp_c': snap_post.temp_c,
                'power_w': snap_post.power_watts,
                'expr_mode': expr_params.mode.name,
                'expr_temperature': expr_params.temperature,
                'expr_top_k': expr_params.top_k,
                'expr_top_p': expr_params.top_p,
                'latent_strain': expression_controller.get_current_latent().strain if expression_controller.get_current_latent() else 0.5,
                'latent_urgency': expression_controller.get_current_latent().urgency if expression_controller.get_current_latent() else 0.0,
                'latent_debt': expression_controller.get_current_latent().debt if expression_controller.get_current_latent() else 0.0,
                'latent_margin': expression_controller.get_current_latent().margin if expression_controller.get_current_latent() else 1.0,
                'token_entropy': token_entropy,
                'log_prob': log_prob,
            })

            # Stop on EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return generated_tokens, records

    def _sample_with_params(
        self,
        logits: torch.Tensor,
        params: ExpressionParams,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Sample from logits using expression parameters.

        Returns:
            (token_id, entropy, log_prob)
        """
        # Apply temperature
        temp = max(0.01, params.temperature)
        scaled_logits = logits / temp

        # Apply top-k filtering
        top_k = max(1, params.top_k)
        if top_k < logits.shape[-1]:
            indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
            scaled_logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        top_p = min(0.99, max(0.1, params.top_p))
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scaled_logits[indices_to_remove] = float('-inf')

        # Compute probabilities
        probs = torch.softmax(scaled_logits, dim=-1)

        # Compute entropy
        valid_probs = probs[probs > 0]
        entropy = -torch.sum(valid_probs * torch.log(valid_probs)).item()

        # Sample
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Get log prob of selected token
        log_prob = torch.log(probs[0, next_token.item()] + 1e-10).item()

        return next_token, entropy, log_prob

    def validate_modulation_strength(
        self,
        modulation_strength: float,
    ) -> ValidationResult:
        """
        Run validation at a specific modulation strength.
        """
        logger.info(f"Validating modulation_strength={modulation_strength:.2f}")

        # Create expression controller with proper calibration
        expression_controller = EmbodiedExpressionController(
            modulation_strength=modulation_strength,
            enabled=(modulation_strength > 0),
            calibration=self.calibration,
        )

        # Reset hardware
        self.actuator.reset_to_default()
        time.sleep(0.5)

        # Warmup
        logger.info(f"Warming up ({self.config.num_warmup_tokens} tokens)...")
        self._warmup(expression_controller, self.config.num_warmup_tokens)

        # Run validation requests
        all_records = []
        for req_idx in range(self.config.num_requests):
            logger.info(f"Request {req_idx + 1}/{self.config.num_requests}")
            expression_controller.reset()

            tokens, records = self._generate_with_kv_cache(
                expression_controller,
                self.config.num_decode_tokens,
            )
            all_records.extend(records)

        # Compute metrics
        result = self._compute_metrics(modulation_strength, all_records)

        # Log summary
        pm = result.phase_metrics
        em = result.expression_metrics
        logger.info(f"  Decode J/token: {pm.decode_j_per_token:.4f}")
        logger.info(f"  Decode TBT p50/p95: {pm.decode_tbt_p50_ms:.1f}/{pm.decode_tbt_p95_ms:.1f} ms")
        logger.info(f"  Mode distribution: {em.mode_distribution}")
        logger.info(f"  Params changed: {em.params_changed}")
        logger.info(f"  Avg temperature: {em.avg_temperature:.3f} (variance: {em.temperature_variance:.4f})")

        return result

    def _warmup(
        self,
        expression_controller: EmbodiedExpressionController,
        num_tokens: int,
    ) -> None:
        """Warmup generation."""
        self._generate_with_kv_cache(expression_controller, num_tokens)
        expression_controller.reset()

    def _compute_metrics(
        self,
        modulation_strength: float,
        records: List[Dict[str, Any]],
    ) -> ValidationResult:
        """Compute all metrics from records."""

        # Separate prefill vs decode
        prefill_records = [r for r in records if r['phase'] == 'PREFILL']
        decode_records = [r for r in records if r['phase'] == 'DECODE']

        # Phase-separated metrics
        phase_metrics = PhaseSeparatedMetrics()

        # Prefill
        if prefill_records:
            phase_metrics.prefill_tokens = len(prefill_records)
            phase_metrics.prefill_time_ms = sum(r['latency_ms'] for r in prefill_records)
            phase_metrics.prefill_energy_j = sum(r['energy_j'] for r in prefill_records)
            phase_metrics.prefill_j_per_token = phase_metrics.prefill_energy_j / phase_metrics.prefill_tokens

        # Decode (the real test)
        if decode_records:
            phase_metrics.decode_tokens = len(decode_records)
            phase_metrics.decode_time_ms = sum(r['latency_ms'] for r in decode_records)
            phase_metrics.decode_energy_j = sum(r['energy_j'] for r in decode_records)
            phase_metrics.decode_j_per_token = phase_metrics.decode_energy_j / phase_metrics.decode_tokens

            decode_latencies = [r['latency_ms'] for r in decode_records]
            phase_metrics.decode_tbt_p50_ms = np.percentile(decode_latencies, 50)
            phase_metrics.decode_tbt_p95_ms = np.percentile(decode_latencies, 95)
            phase_metrics.decode_tbt_p99_ms = np.percentile(decode_latencies, 99)
            phase_metrics.decode_throughput_tps = phase_metrics.decode_tokens / (phase_metrics.decode_time_ms / 1000.0)

            # SLO violations (decode only)
            violations = sum(1 for r in decode_records if r['latency_ms'] > self.config.tbt_slo_ms)
            phase_metrics.decode_slo_violations = violations
            phase_metrics.decode_slo_violation_rate = violations / len(decode_records)

        # Overall
        phase_metrics.total_tokens = len(records)
        phase_metrics.total_time_ms = sum(r['latency_ms'] for r in records)
        phase_metrics.total_energy_j = sum(r['energy_j'] for r in records)
        phase_metrics.j_per_token = phase_metrics.total_energy_j / max(1, phase_metrics.total_tokens)

        # Thermal/power
        temps = [r['temp_c'] for r in records]
        powers = [r['power_w'] for r in records]
        phase_metrics.avg_temp_c = np.mean(temps) if temps else 0.0
        phase_metrics.max_temp_c = max(temps) if temps else 0.0
        phase_metrics.avg_power_w = np.mean(powers) if powers else 0.0

        # Expression metrics (decode only)
        expression_metrics = ExpressionMetrics()

        if decode_records:
            # Mode distribution
            modes = [r['expr_mode'] for r in decode_records]
            mode_counts = {}
            for m in modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            expression_metrics.mode_distribution = mode_counts

            # Mode transitions
            transitions = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])
            expression_metrics.mode_transitions = transitions

            # Expression param averages
            temperatures = [r['expr_temperature'] for r in decode_records]
            top_ks = [r['expr_top_k'] for r in decode_records]
            top_ps = [r['expr_top_p'] for r in decode_records]

            expression_metrics.avg_temperature = np.mean(temperatures)
            expression_metrics.avg_top_k = np.mean(top_ks)
            expression_metrics.avg_top_p = np.mean(top_ps)
            expression_metrics.temperature_variance = np.var(temperatures)

            # Did params actually change?
            unique_temps = len(set(round(t, 3) for t in temperatures))
            expression_metrics.num_unique_temperatures = unique_temps
            expression_metrics.params_changed = unique_temps > 1 or len(mode_counts) > 1

            # Latent averages
            expression_metrics.avg_strain = np.mean([r['latent_strain'] for r in decode_records])
            expression_metrics.avg_urgency = np.mean([r['latent_urgency'] for r in decode_records])
            expression_metrics.avg_debt = np.mean([r['latent_debt'] for r in decode_records])
            expression_metrics.avg_margin = np.mean([r['latent_margin'] for r in decode_records])

        # Quality metrics
        quality_metrics = QualityMetrics()

        if decode_records and self.config.measure_quality:
            entropies = [r['token_entropy'] for r in decode_records]
            log_probs = [r['log_prob'] for r in decode_records]

            quality_metrics.avg_token_entropy = np.mean(entropies) if entropies else 0.0
            quality_metrics.entropy_variance = np.var(entropies) if entropies else 0.0
            quality_metrics.avg_log_prob = np.mean(log_probs) if log_probs else 0.0
            quality_metrics.avg_output_length = len(decode_records)

        return ValidationResult(
            modulation_strength=modulation_strength,
            phase_metrics=phase_metrics,
            expression_metrics=expression_metrics,
            quality_metrics=quality_metrics,
            calibration_used=asdict(self.calibration),
        )

    def run_validation(self) -> Dict[str, Any]:
        """Run full validation."""
        results = {
            'config': asdict(self.config),
            'calibration': asdict(self.calibration),
            'results': [],
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        }

        for strength in self.config.modulation_strengths:
            result = self.validate_modulation_strength(strength)
            results['results'].append(result.to_dict())

        # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"rigorous_validation_{results['timestamp']}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print summary table."""
        print("\n" + "=" * 120)
        print("Z83 RIGOROUS VALIDATION RESULTS")
        print("=" * 120)
        print(f"Model: {self.config.model_name}")
        print(f"Machine: {self.config.machine_type}")
        print(f"Calibration: target={self.calibration.j_per_token_target}, max={self.calibration.j_per_token_max}")
        print("=" * 120)
        print(f"{'Mod':>5} {'Decode J/tok':>12} {'Dec TBT p50':>11} {'Dec TBT p95':>11} "
              f"{'Dec Thru':>10} {'SLO Viol':>10} {'Params Δ':>10} {'Modes':>20} {'Avg Temp':>10}")
        print("-" * 120)

        baseline_jpt = None
        for r in results['results']:
            strength = r['modulation_strength']
            pm = r['phase_metrics']
            em = r['expression_metrics']

            if baseline_jpt is None and strength == 0.0:
                baseline_jpt = pm['decode_j_per_token']

            # Format mode distribution
            modes = em['mode_distribution']
            total = sum(modes.values()) if modes else 1
            mode_str = ", ".join([f"{k[:3]}:{v/total*100:.0f}%" for k, v in modes.items()]) if modes else "N/A"

            params_changed = "YES" if em['params_changed'] else "NO"

            print(f"{strength:>5.2f} {pm['decode_j_per_token']:>12.4f} "
                  f"{pm['decode_tbt_p50_ms']:>11.2f} {pm['decode_tbt_p95_ms']:>11.2f} "
                  f"{pm['decode_throughput_tps']:>10.1f} {pm['decode_slo_violation_rate']:>9.1%} "
                  f"{params_changed:>10} {mode_str:>20} {em['avg_temperature']:>10.3f}")

        print("=" * 120)

        if baseline_jpt and len(results['results']) > 1:
            best = min(results['results'], key=lambda x: x['phase_metrics']['decode_j_per_token'])
            if best['modulation_strength'] > 0:
                best_jpt = best['phase_metrics']['decode_j_per_token']
                savings = (baseline_jpt - best_jpt) / baseline_jpt * 100
                print(f"\nBest efficiency: modulation={best['modulation_strength']:.2f} "
                      f"({best_jpt:.4f} J/token, {savings:.1f}% savings vs baseline)")

                # Report whether expression params actually changed
                if best['expression_metrics']['params_changed']:
                    print(f"✓ Expression parameters DID change during generation")
                else:
                    print(f"⚠ Expression parameters did NOT change - savings may be from other factors")

    def shutdown(self):
        """Clean up."""
        self.actuator.reset_to_default()


def detect_machine_type() -> str:
    """Detect machine type from hostname or GPU."""
    import socket
    hostname = socket.gethostname().lower()

    if 'ikaros' in hostname:
        return 'amd_apu'
    elif 'daedalus' in hostname:
        return 'amd_dgpu'
    elif 'minos' in hostname:
        return 'nvidia_gpu'

    # Fallback: check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'nvidia' in gpu_name:
            return 'nvidia_gpu'
        elif 'amd' in gpu_name or 'radeon' in gpu_name:
            return 'amd_dgpu'

    return 'amd_apu'  # Default


def main():
    parser = argparse.ArgumentParser(description='Z83: Rigorous Embodied Validation')
    parser.add_argument('--model', default="Qwen/Qwen2.5-0.5B-Instruct", help='Model name')
    parser.add_argument('--decode-tokens', type=int, default=200, help='Decode tokens per request')
    parser.add_argument('--requests', type=int, default=5, help='Number of requests')
    parser.add_argument('--output', default="results/z83_rigorous", help='Output directory')
    parser.add_argument('--machine', choices=['amd_apu', 'amd_dgpu', 'nvidia_gpu', 'auto'],
                       default='auto', help='Machine type for calibration')
    parser.add_argument('--quick', action='store_true', help='Quick validation')
    parser.add_argument('--modulation', type=float, nargs='+', default=None,
                       help='Specific modulation strengths to test')
    args = parser.parse_args()

    # Auto-detect machine if needed
    machine_type = args.machine if args.machine != 'auto' else detect_machine_type()
    logger.info(f"Machine type: {machine_type}")

    config = RigorousConfig(
        model_name=args.model,
        num_decode_tokens=args.decode_tokens if not args.quick else 100,
        num_requests=args.requests if not args.quick else 2,
        output_dir=args.output,
        machine_type=machine_type,
        modulation_strengths=args.modulation if args.modulation else ([0.0, 1.0] if args.quick else [0.0, 0.5, 1.0]),
    )

    validator = RigorousValidator(config)

    try:
        results = validator.run_validation()

        # Output machine-readable summary
        summary = {
            'machine': machine_type,
            'baseline_decode_jpt': results['results'][0]['phase_metrics']['decode_j_per_token'],
            'best_decode_jpt': min(r['phase_metrics']['decode_j_per_token'] for r in results['results']),
            'best_modulation': min(results['results'], key=lambda x: x['phase_metrics']['decode_j_per_token'])['modulation_strength'],
            'params_changed': any(r['expression_metrics']['params_changed'] for r in results['results'] if r['modulation_strength'] > 0),
        }
        print(json.dumps(summary))
    finally:
        validator.shutdown()


if __name__ == "__main__":
    main()
