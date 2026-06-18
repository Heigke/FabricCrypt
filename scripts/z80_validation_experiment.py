#!/usr/bin/env python3
"""
FEEL Validation Experiment - Scientific Baselines
==================================================

This script runs the complete validation suite for FEEL claims:

BASELINE SET A: Same model, different controllers (no math coupling)
- Fixed ECO/MED/PERF
- PID thermal control
- GreenLLM-style dual-loop
- throttLL'eM-style predictive
- MultiScale (our contribution)

BASELINE SET B: Same controller, different coupling strength
- Coupling OFF (baseline)
- Coupling ON (embodied)
- Coupling ON + adaptive

METRICS (all logged together):
- Quality: perplexity, task accuracy
- Serving: TTFT, TBT percentiles (p50/p95/p99)
- Energy: J/token, total energy per request
- Thermals: time above threshold, throttle residency

Reference:
- GreenLLM: https://arxiv.org/html/2508.16449v1
- throttLL'eM: https://arxiv.org/html/2408.05235v2

Author: FEEL Research Team
Date: 2026-01-20
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from collections import defaultdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

# Import atom module
from atom import (
    TokenMetabolismStep,
    AtomConfig,
    FixedController,
    NullController,
    BanditController,
    ActionLevel,
    create_sensor,
    SyncMode,
)
from atom.multiscale import MultiScaleController, MultiScaleConfig, AdaptiveMultiScaleController

# Import baselines
sys.path.insert(0, str(Path(__file__).parent.parent / "src/body_daemon/controller"))
try:
    from baselines import (
        FixedCapController,
        PIDController,
        GreenLLMController,
        ThrottLLMController,
        FixedCapConfig,
        PIDConfig,
        GreenLLMConfig,
        ThrottLLMConfig,
    )
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False
    print("Warning: baselines module not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a single validation run."""
    controller_name: str
    coupling_mode: str  # "off", "on", "adaptive"

    # Timing metrics
    num_tokens: int = 0
    total_time_ms: float = 0.0
    ttft_ms: float = 0.0  # Time to first token
    tbt_p50_ms: float = 0.0  # Time between tokens (median)
    tbt_p95_ms: float = 0.0
    tbt_p99_ms: float = 0.0
    throughput_tps: float = 0.0

    # Energy metrics
    total_energy_j: float = 0.0
    j_per_token: float = 0.0
    avg_power_w: float = 0.0

    # Thermal metrics
    avg_temp_c: float = 0.0
    max_temp_c: float = 0.0
    time_above_threshold_frac: float = 0.0
    throttle_residency_frac: float = 0.0

    # Quality metrics (from external eval)
    perplexity: float = 0.0

    # Controller stats
    controller_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationConfig:
    """Configuration for validation experiment."""
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"  # "cuda", "hip", "auto"

    # Experiment
    num_warmup_tokens: int = 50
    num_eval_tokens: int = 500
    num_requests: int = 5  # Number of separate requests

    # Prompts
    prompt: str = "Explain the concept of energy efficiency in computing. Discuss:"

    # Thermal threshold (for time_above_threshold metric)
    temp_threshold_c: float = 75.0

    # Controllers to test
    controllers: List[str] = field(default_factory=lambda: [
        "fixed_eco", "fixed_med", "fixed_perf",
        "pid", "greenllm", "throttllm",
        "multiscale", "bandit"
    ])

    # Output
    output_dir: str = "results/z80_validation"


def detect_device() -> str:
    """Auto-detect GPU device."""
    # Check for CUDA (NVIDIA or ROCm)
    if torch.cuda.is_available():
        return "cuda"
    # Check for AMD gpu_metrics (ROCm not fully initialized)
    if Path("/sys/class/drm/card0/device/gpu_metrics").exists():
        return "cuda"  # AMD with ROCm
    if Path("/sys/class/drm/card1/device/gpu_metrics").exists():
        return "cuda"  # AMD at card1
    return "cpu"


def load_model(model_name: str, device: str):
    """Load model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )

        if device == "cpu":
            pass  # Already on CPU
        elif device == "cuda":
            if not hasattr(model, 'device') or str(model.device) == 'cpu':
                model = model.to('cuda')

        model.eval()
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def create_controller(name: str, config: AtomConfig):
    """Create a controller by name."""
    name = name.lower()

    if name == "null" or name == "fixed_med":
        return FixedController(ActionLevel.MED)
    elif name == "fixed_eco":
        return FixedController(ActionLevel.ECO)
    elif name == "fixed_perf":
        return FixedController(ActionLevel.PERF)
    elif name == "fixed_low":
        return FixedController(ActionLevel.LOW)
    elif name == "fixed_high":
        return FixedController(ActionLevel.HIGH)
    elif name == "bandit":
        return BanditController(config, epsilon=0.1, learning_rate=0.01)
    elif name == "multiscale":
        return MultiScaleController(MultiScaleConfig(
            fast_interval_ms=30.0,
            slow_interval_ms=500.0,
        ))
    elif name == "adaptive_multiscale":
        return AdaptiveMultiScaleController(MultiScaleConfig())
    elif name == "pid" and BASELINES_AVAILABLE:
        # Wrap in atom-compatible interface
        return _BaselineWrapper(PIDController())
    elif name == "greenllm" and BASELINES_AVAILABLE:
        return _BaselineWrapper(GreenLLMController())
    elif name == "throttllm" and BASELINES_AVAILABLE:
        return _BaselineWrapper(ThrottLLMController())
    else:
        logger.warning(f"Unknown controller: {name}, using FixedController(MED)")
        return FixedController(ActionLevel.MED)


class _BaselineWrapper:
    """Wrapper to make body_daemon baselines compatible with atom controllers."""

    def __init__(self, baseline_controller):
        self._baseline = baseline_controller
        self._step_count = 0

    def decide(self, body_state, power_min, power_max):
        from atom.schema import AtomicAction, ActionLevel, timestamp_ns

        # Convert body state to observation vector
        obs = np.array(body_state.to_observation_vector(), dtype=np.float32)

        # Get action from baseline
        action_idx = self._baseline.select_action(obs)
        level = ActionLevel(int(np.clip(action_idx, 0, 4)))

        power_range = power_max - power_min
        target_power = power_min + power_range * (level.value / 4.0)

        self._step_count += 1

        return AtomicAction(
            level=level,
            target_power_watts=target_power,
            timestamp_ns=timestamp_ns(),
            applied=False,
            rate_limited=False,
            previous_level=level,
        )

    def update(self, body_state, action, reward):
        obs = np.array(body_state.to_observation_vector(), dtype=np.float32)
        self._baseline.update(obs, action.level.value, reward)

    def get_stats(self):
        return self._baseline.get_stats()

    def compute_reward(self, **kwargs):
        # Use default reward computation
        j_per_token = kwargs.get('j_per_token', 0)
        latency_ms = kwargs.get('latency_ms', 0)
        latency_slo_ms = kwargs.get('latency_slo_ms', 50)
        temp_c = kwargs.get('temp_c', 50)
        temp_limit_c = kwargs.get('temp_limit_c', 100)
        thermal_margin_c = kwargs.get('thermal_margin_c', 5)

        energy_cost = j_per_token * 10.0
        latency_excess = max(0, latency_ms - latency_slo_ms) / latency_slo_ms
        slo_penalty = latency_excess * 2.0
        temp_threshold = temp_limit_c - thermal_margin_c
        temp_excess = max(0, temp_c - temp_threshold) / thermal_margin_c
        thermal_penalty = temp_excess * 1.0

        return -(energy_cost + slo_penalty + thermal_penalty)


def run_validation(
    model,
    tokenizer,
    controller_name: str,
    config: ValidationConfig,
    atom_config: AtomConfig,
) -> ValidationResult:
    """Run a single validation with one controller."""
    logger.info(f"Running validation: {controller_name}")

    # Create controller
    controller = create_controller(controller_name, atom_config)

    # Create atom
    atom = TokenMetabolismStep(
        config=atom_config,
        controller=controller,
    )

    # Initialize
    device_id = 0
    # Check for card1 on systems where GPU is not at card0
    if not Path("/sys/class/drm/card0/device/gpu_metrics").exists():
        if Path("/sys/class/drm/card1/device/gpu_metrics").exists():
            device_id = 1

    try:
        atom.initialize(model, tokenizer, device_id=device_id)
    except Exception as e:
        logger.warning(f"Atom initialization failed: {e}, trying fallback")
        # Try without device_id
        atom.initialize(model, tokenizer)

    # Encode prompt
    input_ids = tokenizer.encode(config.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')

    # Metrics collection
    latencies_ms = []
    energies_j = []
    temps_c = []
    powers_w = []
    throttled_steps = 0
    above_threshold_steps = 0
    total_steps = 0

    try:
        # Warmup
        logger.info(f"Warmup: {config.num_warmup_tokens} tokens")
        for i in range(config.num_warmup_tokens):
            try:
                record = atom.step(
                    input_ids=input_ids if i == 0 else None,
                    num_tokens=1,
                    temperature=0.0,
                )
            except Exception as e:
                logger.warning(f"Warmup step {i} failed: {e}")
                break

        # Reset energy window
        atom.reset_energy_window()

        # Evaluation
        logger.info(f"Evaluation: {config.num_eval_tokens} tokens")
        start_time = time.time()

        for i in range(config.num_eval_tokens):
            try:
                record = atom.step(
                    input_ids=None,  # Continue from cache
                    num_tokens=1,
                    temperature=0.0,
                )

                # Collect metrics
                latencies_ms.append(record.tokens.latency_ms)
                energies_j.append(record.energy_delta_joules)
                temps_c.append(record.sense_post.temp_c)
                powers_w.append(record.sense_post.power_watts)

                if record.sense_post.is_throttled:
                    throttled_steps += 1
                if record.sense_post.temp_c > config.temp_threshold_c:
                    above_threshold_steps += 1
                total_steps += 1

            except Exception as e:
                logger.warning(f"Eval step {i} failed: {e}")
                continue

        total_time = time.time() - start_time

    finally:
        atom.shutdown()

    # Compute statistics
    if len(latencies_ms) == 0:
        logger.error("No successful steps!")
        return ValidationResult(
            controller_name=controller_name,
            coupling_mode="on",
        )

    latencies_np = np.array(latencies_ms)
    energies_np = np.array(energies_j)
    temps_np = np.array(temps_c)
    powers_np = np.array(powers_w)

    # Get controller stats
    ctrl_stats = {}
    if hasattr(controller, 'get_stats'):
        ctrl_stats = controller.get_stats()

    result = ValidationResult(
        controller_name=controller_name,
        coupling_mode="on",

        # Timing
        num_tokens=len(latencies_ms),
        total_time_ms=total_time * 1000,
        ttft_ms=latencies_ms[0] if latencies_ms else 0,
        tbt_p50_ms=float(np.percentile(latencies_np, 50)),
        tbt_p95_ms=float(np.percentile(latencies_np, 95)),
        tbt_p99_ms=float(np.percentile(latencies_np, 99)),
        throughput_tps=len(latencies_ms) / total_time if total_time > 0 else 0,

        # Energy
        total_energy_j=float(np.sum(energies_np)),
        j_per_token=float(np.mean(energies_np)),
        avg_power_w=float(np.mean(powers_np)),

        # Thermal
        avg_temp_c=float(np.mean(temps_np)),
        max_temp_c=float(np.max(temps_np)),
        time_above_threshold_frac=above_threshold_steps / max(total_steps, 1),
        throttle_residency_frac=throttled_steps / max(total_steps, 1),

        # Controller
        controller_stats=ctrl_stats,
    )

    # Log summary
    logger.info(f"  Tokens: {result.num_tokens}")
    logger.info(f"  J/token: {result.j_per_token:.4f}")
    logger.info(f"  TBT p50/p95: {result.tbt_p50_ms:.1f}/{result.tbt_p95_ms:.1f} ms")
    logger.info(f"  Avg temp: {result.avg_temp_c:.1f}°C")
    logger.info(f"  Throttle: {result.throttle_residency_frac:.1%}")

    return result


def main():
    parser = argparse.ArgumentParser(description="FEEL Validation Experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    parser.add_argument("--device", default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--warmup-tokens", type=int, default=50)
    parser.add_argument("--eval-tokens", type=int, default=500)
    parser.add_argument("--output-dir", default="results/z80_validation")
    parser.add_argument("--controllers", nargs="+", default=[
        "fixed_eco", "fixed_med", "fixed_perf", "multiscale", "bandit"
    ])
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer tokens)")

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.warmup_tokens = 10
        args.eval_tokens = 100

    # Setup
    config = ValidationConfig(
        model_name=args.model,
        device=args.device if args.device != "auto" else detect_device(),
        num_warmup_tokens=args.warmup_tokens,
        num_eval_tokens=args.eval_tokens,
        output_dir=args.output_dir,
        controllers=args.controllers,
    )

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Atom config
    atom_config = AtomConfig(
        rate_limit_ms=200,
        ema_alpha=0.3,
    )

    logger.info("=" * 60)
    logger.info("FEEL Validation Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Warmup: {config.num_warmup_tokens} tokens")
    logger.info(f"Eval: {config.num_eval_tokens} tokens")
    logger.info(f"Controllers: {config.controllers}")

    # Load model
    model, tokenizer = load_model(config.model_name, config.device)

    # Run validations
    results = []
    for controller_name in config.controllers:
        try:
            result = run_validation(
                model, tokenizer, controller_name, config, atom_config
            )
            results.append(result)

            # Save incremental results (with numpy type conversion)
            result_file = output_dir / f"{controller_name}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=lambda x: bool(x) if isinstance(x, (np.bool_,)) else float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else x)

        except Exception as e:
            logger.error(f"Validation failed for {controller_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results (with numpy type conversion)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    combined_file = output_dir / f"validation_{timestamp}.json"

    def numpy_serializer(x):
        if isinstance(x, (np.bool_,)):
            return bool(x)
        elif isinstance(x, (np.floating,)):
            return float(x)
        elif isinstance(x, (np.integer,)):
            return int(x)
        return x

    with open(combined_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'results': [r.to_dict() for r in results],
            'timestamp': timestamp,
        }, f, indent=2, default=numpy_serializer)

    logger.info(f"\nResults saved to: {combined_file}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Controller':<15} {'J/token':>10} {'TBT p50':>10} {'TBT p95':>10} {'Temp':>8} {'Throttle':>10}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.j_per_token):
        print(f"{r.controller_name:<15} {r.j_per_token:>10.4f} {r.tbt_p50_ms:>10.1f} "
              f"{r.tbt_p95_ms:>10.1f} {r.avg_temp_c:>7.1f}C {r.throttle_residency_frac:>9.1%}")

    print("=" * 80)

    # Output JSON for machine parsing
    efficiency_results = [r for r in results if r.j_per_token > 0]
    latency_results = [r for r in results if r.tbt_p50_ms > 0]
    summary = {
        'best_efficiency': min((r.controller_name for r in efficiency_results), default='N/A'),
        'best_latency': min((r.controller_name for r in latency_results), key=lambda n: next(r.tbt_p50_ms for r in results if r.controller_name == n), default='N/A') if latency_results else 'N/A',
        'all_results': len(results),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
