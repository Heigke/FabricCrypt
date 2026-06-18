#!/usr/bin/env python3
"""
Meta-Cognitive Energy Experiments.

This script runs experiments with temperature > 0 to observe "effort allocation"
behavior in LLMs. We use prompts that:
1. Ask the model about its own states (meta-cognition)
2. Vary in difficulty to induce different hidden state patterns
3. Explicitly mention energy efficiency to observe model adaptation

Research goal: Generate quality-vs-Joules curves to prove "effort allocation" behavior.
"""

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.hf_infer_extended import (
    run_single_inference,
    ExperimentCondition,
    EnhancedPowerRecorder,
)
from src.energy_harness.enhanced_gpu_metrics import DPMStateController
from src.energy_harness.internal_signals import (
    LatentStateController,
    ZeroOverheadLatentCapture,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PromptResult:
    """Result from a single prompt experiment."""
    prompt_text: str
    difficulty: str
    category: str
    temperature: float

    # Timing
    total_s: float
    ttft_s: float
    tpot_mean_s: float
    tpot_p95_s: float

    # Energy
    total_energy_j: float
    avg_power_w: float

    # Latent metrics
    hidden_norm_mean: float = 0.0
    hidden_delta_mean: float = 0.0
    margin_mean: float = 0.0
    entropy_mean: float = 0.0

    # Quality proxies
    output_length: int = 0
    unique_tokens: int = 0

    # Policy
    policy: str = "latent_v2"
    policy_switches: int = 0


def load_prompts(config_path: Path) -> Dict[str, List[Dict]]:
    """Load prompts from YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("prompts", {})


def run_prompt_experiment(
    model,
    tokenizer,
    prompt_text: str,
    difficulty: str,
    category: str,
    temperature: float,
    policy: str,
    dpm_controller: Optional[DPMStateController],
    max_tokens: int = 128,
    energy_context: str = "",
) -> PromptResult:
    """Run a single prompt experiment with energy measurement."""

    # Prepend energy context if using meta-cognitive mode
    if energy_context and difficulty in ("meta", "energy_aware"):
        full_prompt = f"{energy_context}\n\n{prompt_text}"
    else:
        full_prompt = prompt_text

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]

    # Setup controllers
    latent_controller = None
    latent_capture = None

    if policy in ("latent", "latent_v2"):
        latent_controller = LatentStateController(
            window_tokens=32,  # Smaller window for prompt-level analysis
            high_delta_threshold=0.15,
            low_delta_threshold=0.05,
            min_dwell_tokens=16,
        )

    if policy == "latent_v2":
        latent_capture = ZeroOverheadLatentCapture(model)

    # Setup recorder
    recorder = EnhancedPowerRecorder(sample_interval_ms=10)
    recorder.start(metadata={"prompt": prompt_text[:100], "difficulty": difficulty})

    t_start = time.perf_counter()

    # Run inference with temperature
    metrics, decode_times, signal_trace = run_single_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=full_prompt,
        max_new_tokens=max_tokens,
        recorder=recorder,
        dpm_controller=dpm_controller,
        policy=policy,
        latent_controller=latent_controller,
        latent_capture=latent_capture,
    )

    # Get energy measurement
    measurement = recorder.stop()

    # Get latent controller summary
    latent_summary = {}
    if latent_controller:
        latent_summary = latent_controller.get_summary()

    # Decode output for quality metrics
    output_ids = tokenizer.encode(metrics.generated_text if hasattr(metrics, 'generated_text') else "", add_special_tokens=False)
    unique_tokens = len(set(output_ids)) if output_ids else 0

    return PromptResult(
        prompt_text=prompt_text[:200],  # Truncate for storage
        difficulty=difficulty,
        category=category,
        temperature=temperature,
        total_s=metrics.total_s,
        ttft_s=metrics.ttft_s,
        tpot_mean_s=metrics.tpot_s_mean,
        tpot_p95_s=metrics.tpot_s_p95,
        total_energy_j=measurement.energy_joules,
        avg_power_w=measurement.avg_power_watts,
        hidden_norm_mean=latent_summary.get("hidden_norm_mean", 0.0),
        hidden_delta_mean=latent_summary.get("delta_mean", 0.0),
        margin_mean=latent_summary.get("margin_mean", 0.0),
        entropy_mean=metrics.entropy_mean if hasattr(metrics, 'entropy_mean') else 0.0,
        output_length=len(output_ids),
        unique_tokens=unique_tokens,
        policy=policy,
        policy_switches=latent_summary.get("policy_switches", 0),
    )


def run_difficulty_comparison(
    model,
    tokenizer,
    prompts_config: Dict,
    temperatures: List[float],
    policy: str,
    dpm_controller: Optional[DPMStateController],
    energy_context: str,
    repeats: int = 3,
) -> List[PromptResult]:
    """Run experiments across difficulty levels and temperatures."""

    results = []

    for difficulty, prompts in prompts_config.items():
        logger.info(f"Running {difficulty} prompts ({len(prompts)} prompts)...")

        for prompt_info in prompts:
            prompt_text = prompt_info.get("text", prompt_info) if isinstance(prompt_info, dict) else prompt_info
            category = prompt_info.get("category", "unknown") if isinstance(prompt_info, dict) else "unknown"

            for temp in temperatures:
                for rep in range(repeats):
                    try:
                        result = run_prompt_experiment(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_text=prompt_text,
                            difficulty=difficulty,
                            category=category,
                            temperature=temp,
                            policy=policy,
                            dpm_controller=dpm_controller,
                            energy_context=energy_context,
                        )
                        results.append(result)

                        logger.info(
                            f"  {difficulty}/{category} T={temp:.1f} rep{rep+1}: "
                            f"{result.total_s:.2f}s, {result.total_energy_j:.1f}J, "
                            f"switches={result.policy_switches}"
                        )

                    except Exception as e:
                        logger.error(f"Error on {difficulty}/{category}: {e}")

    return results


def compute_effort_allocation_stats(results: List[PromptResult]) -> Dict[str, Any]:
    """Compute effort allocation statistics by difficulty."""

    stats = {}

    for difficulty in ["easy", "medium", "hard", "meta", "energy_aware"]:
        diff_results = [r for r in results if r.difficulty == difficulty]
        if not diff_results:
            continue

        stats[difficulty] = {
            "n": len(diff_results),
            "energy_mean": statistics.mean(r.total_energy_j for r in diff_results),
            "energy_std": statistics.stdev([r.total_energy_j for r in diff_results]) if len(diff_results) > 1 else 0,
            "time_mean": statistics.mean(r.total_s for r in diff_results),
            "joules_per_sec": statistics.mean(r.total_energy_j / r.total_s for r in diff_results if r.total_s > 0),
            "tpot_mean": statistics.mean(r.tpot_mean_s * 1000 for r in diff_results),  # ms
            "policy_switches_mean": statistics.mean(r.policy_switches for r in diff_results),
        }

    # Compute effort ratios (hard vs easy)
    if "easy" in stats and "hard" in stats:
        stats["effort_ratio"] = {
            "energy_hard_vs_easy": stats["hard"]["energy_mean"] / stats["easy"]["energy_mean"],
            "time_hard_vs_easy": stats["hard"]["time_mean"] / stats["easy"]["time_mean"],
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Meta-Cognitive Energy Experiments")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompts-config", type=Path,
                       default=Path("config/prompts/meta_cognitive_prompts.yaml"))
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument("--policy", choices=["auto", "latent", "latent_v2"], default="latent_v2")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--difficulties", nargs="+",
                       default=["easy", "medium", "hard", "meta"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/meta_cognitive"))
    parser.add_argument("--enable-dpm", action="store_true", help="Enable DPM control")
    parser.add_argument("--quick", action="store_true", help="Quick test with subset")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    logger.info(f"Loading prompts from {args.prompts_config}")
    all_prompts = load_prompts(args.prompts_config)

    # Filter to selected difficulties
    prompts_config = {k: v for k, v in all_prompts.items() if k in args.difficulties}

    # Quick mode: use subset
    if args.quick:
        prompts_config = {k: v[:2] for k, v in prompts_config.items()}
        args.repeats = 1
        args.temperatures = [1.0]

    # Load energy context
    with open(args.prompts_config) as f:
        full_config = yaml.safe_load(f)
    energy_context = full_config.get("energy_context", "")

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )

    # Setup DPM controller
    dpm_controller = None
    if args.enable_dpm:
        try:
            dpm_controller = DPMStateController()
            logger.info(f"DPM Controller enabled: SCLK={dpm_controller.sclk_states}")
        except Exception as e:
            logger.warning(f"DPM control unavailable: {e}")

    # Run experiments
    logger.info(f"Running experiments: {len(prompts_config)} difficulties, "
                f"{args.temperatures} temps, {args.repeats} repeats")

    results = run_difficulty_comparison(
        model=model,
        tokenizer=tokenizer,
        prompts_config=prompts_config,
        temperatures=args.temperatures,
        policy=args.policy,
        dpm_controller=dpm_controller,
        energy_context=energy_context,
        repeats=args.repeats,
    )

    # Compute statistics
    stats = compute_effort_allocation_stats(results)

    # Save results
    results_data = {
        "config": {
            "model": args.model,
            "policy": args.policy,
            "temperatures": args.temperatures,
            "repeats": args.repeats,
            "difficulties": args.difficulties,
        },
        "stats": stats,
        "results": [asdict(r) for r in results],
    }

    output_file = args.output_dir / "meta_cognitive_results.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("EFFORT ALLOCATION SUMMARY")
    print("=" * 80)
    print(f"{'Difficulty':<15} | {'N':>4} | {'Energy (J)':>12} | {'Time (s)':>10} | {'TPOT (ms)':>10}")
    print("-" * 80)

    for diff, s in stats.items():
        if diff == "effort_ratio":
            continue
        print(f"{diff:<15} | {s['n']:>4} | {s['energy_mean']:>9.1f}±{s['energy_std']:.1f} | "
              f"{s['time_mean']:>10.2f} | {s['tpot_mean']:>10.1f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
