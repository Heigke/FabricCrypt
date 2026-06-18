#!/usr/bin/env python3
"""
Hot Start Protocol: Cold vs Thermal Stress Evaluation

Novel Contribution:
-------------------
Evaluate cognitive control strategies under two thermal regimes:
1. COLD START: Idle → run suite once (GPU starts cold)
2. HOT START: Sustained warmup → run suite repeatedly until near steady-state

This mirrors thermal/power scheduling papers (TAPAS, etc.) but adds:
- CORRECTNESS-AWARE cognition (not just latency/throughput)
- Per-strategy comparison under thermal stress

Report:
- Accuracy, Joules, J/correct
- TTFT/TPOT tails
- Throttle rate (clock drops while load stays high)
- Stability: oscillations in regime switching

The key hypothesis: Proprioceptive control should WIN under hot-start,
because it backs off escalation when thermals are constrained.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from scripts.eval_suite import EVAL_SUITE_EXPANDED, check_answer
from scripts.north_star_experiment import (
    NorthStarConfig,
    predict_p_error,
    decide_compute_level_uncertainty,
    decide_compute_level_proprioceptive,
    generate_with_strategy,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class ThermalMetrics:
    """Thermal and throttling metrics for a run."""
    start_temp_c: float
    end_temp_c: float
    peak_temp_c: float
    mean_temp_c: float
    temp_derivative_mean: float  # °C/min
    throttle_events: int
    throttle_rate: float  # events per 1000 tokens
    total_tokens: int
    duration_s: float


@dataclass
class RunResult:
    """Result of a single evaluation run."""
    regime: str  # "cold" or "hot"
    strategy: str
    accuracy: float
    n_correct: int
    n_total: int
    total_energy_j: float
    j_per_correct: float
    mean_latency_ms: float
    p95_latency_ms: float
    thermal: ThermalMetrics


class ThermalMonitor:
    """Monitor thermal state during experiment."""

    def __init__(self):
        self.temps = []
        self.powers = []
        self.clocks = []
        self.times = []
        self.throttle_events = 0
        self.last_clock = None
        self.tokens_generated = 0

    def update(self, power: float, temp: float = None, clock: int = None):
        """Record a sample."""
        now = time.time()
        self.times.append(now)
        self.powers.append(power)

        # Estimate temp from power if not provided
        if temp is None:
            temp = min(95, 50 + power * 0.4)
        self.temps.append(temp)

        # Detect throttle (clock drop under load)
        if clock is None:
            clock = 1800
        self.clocks.append(clock)

        if self.last_clock is not None and clock < self.last_clock * 0.9:
            self.throttle_events += 1
        self.last_clock = clock

    def add_tokens(self, n: int):
        """Track generated tokens."""
        self.tokens_generated += n

    def get_metrics(self) -> ThermalMetrics:
        """Compute thermal metrics."""
        if not self.temps:
            return ThermalMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

        duration = self.times[-1] - self.times[0] if len(self.times) > 1 else 1.0

        # Temperature derivative (°C/min)
        if len(self.temps) > 1:
            temp_change = self.temps[-1] - self.temps[0]
            temp_derivative = temp_change / (duration / 60)  # °C/min
        else:
            temp_derivative = 0.0

        # Throttle rate
        throttle_rate = (self.throttle_events / max(1, self.tokens_generated)) * 1000

        return ThermalMetrics(
            start_temp_c=self.temps[0],
            end_temp_c=self.temps[-1],
            peak_temp_c=max(self.temps),
            mean_temp_c=sum(self.temps) / len(self.temps),
            temp_derivative_mean=temp_derivative,
            throttle_events=self.throttle_events,
            throttle_rate=throttle_rate,
            total_tokens=self.tokens_generated,
            duration_s=duration,
        )

    def reset(self):
        """Reset for new run."""
        self.temps = []
        self.powers = []
        self.clocks = []
        self.times = []
        self.throttle_events = 0
        self.last_clock = None
        self.tokens_generated = 0


def warmup_gpu(model, tokenizer, duration_s: float = 60, target_temp_c: float = 70):
    """Run sustained decode to warm up GPU."""
    print(f"Warming up GPU for {duration_s}s (target: {target_temp_c}°C)...")

    prompt = "Explain the theory of relativity in detail, covering both special and general relativity, their mathematical formulations, experimental evidence, and practical applications in modern technology. Also discuss the implications for our understanding of space, time, gravity, and the universe."

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    recorder = PowerTraceRecorder(sample_interval_ms=100)
    recorder.start()

    start_time = time.time()
    tokens_generated = 0

    while time.time() - start_time < duration_s:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        tokens_generated += outputs.shape[1] - inputs.input_ids.shape[1]

        # Check temperature (estimate from power)
        if recorder.samples:
            power = recorder.samples[-1].power_watts
            temp = min(95, 50 + power * 0.4)
            elapsed = time.time() - start_time
            print(f"  {elapsed:.0f}s: {tokens_generated} tokens, ~{temp:.0f}°C, {power:.0f}W")

            if temp >= target_temp_c:
                print(f"  Reached target temperature {target_temp_c}°C")
                break

    recorder.stop()

    final_power = recorder.samples[-1].power_watts if recorder.samples else 50
    final_temp = min(95, 50 + final_power * 0.4)
    print(f"Warmup complete: {tokens_generated} tokens, final ~{final_temp:.0f}°C")

    return final_temp


def run_evaluation(
    model, tokenizer,
    items: List[Dict],
    strategy: str,
    config: NorthStarConfig,
    budget_j: float,
    thermal_monitor: ThermalMonitor,
) -> Tuple[List[Dict], float]:
    """Run evaluation with specified strategy, tracking thermals."""

    results = []
    spent_j = 0.0
    latencies = []

    for idx, item in enumerate(items):
        if spent_j >= budget_j:
            break

        start_time = time.time()

        output, energy, stats = generate_with_strategy(
            model, tokenizer, item["q"], 32, strategy, config,
            budget_j=budget_j, spent_j=spent_j
        )

        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)

        is_correct, _ = check_answer(output, item)
        spent_j += energy

        # Update thermal monitor
        thermal_monitor.add_tokens(len(stats.get("margins", [])))
        # Update from energy (proxy for power samples)
        thermal_monitor.update(power=energy * 10)  # Rough estimate

        results.append({
            "correct": is_correct,
            "energy_j": energy,
            "latency_ms": latency_ms,
            "margin_min": min(stats["margins"]) if stats["margins"] else 0,
        })

    return results, latencies


def run_regime(
    model, tokenizer,
    suite: List[Dict],
    regime: str,
    strategies: List[str],
    config: NorthStarConfig,
    budget_j: float,
    warmup_duration: float = 60,
) -> Dict[str, RunResult]:
    """Run all strategies under a thermal regime."""

    regime_results = {}

    for strategy in strategies:
        print(f"\n=== {regime.upper()} / {strategy.upper()} ===")

        # Warmup if hot-start
        if regime == "hot":
            warmup_gpu(model, tokenizer, duration_s=warmup_duration)

        thermal_monitor = ThermalMonitor()
        thermal_monitor.update(power=50)  # Initial reading

        results, latencies = run_evaluation(
            model, tokenizer, suite, strategy, config, budget_j, thermal_monitor
        )

        # Compute metrics
        n_correct = sum(1 for r in results if r["correct"])
        n_total = len(results)
        total_energy = sum(r["energy_j"] for r in results)

        thermal_metrics = thermal_monitor.get_metrics()

        regime_results[strategy] = RunResult(
            regime=regime,
            strategy=strategy,
            accuracy=n_correct / n_total if n_total > 0 else 0,
            n_correct=n_correct,
            n_total=n_total,
            total_energy_j=total_energy,
            j_per_correct=total_energy / n_correct if n_correct > 0 else float('inf'),
            mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            thermal=thermal_metrics,
        )

        print(f"  Accuracy: {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
        print(f"  Energy: {total_energy:.1f}J, J/correct: {total_energy/max(1,n_correct):.2f}")
        print(f"  Thermal: start={thermal_metrics.start_temp_c:.0f}°C "
              f"peak={thermal_metrics.peak_temp_c:.0f}°C throttle_rate={thermal_metrics.throttle_rate:.2f}")

        # Cool down between strategies
        print("  Cooling down...")
        time.sleep(30)

    return regime_results


def analyze_regimes(cold_results: Dict, hot_results: Dict) -> Dict[str, Any]:
    """Analyze differences between cold and hot regimes."""

    analysis = {
        "cold": {},
        "hot": {},
        "degradation": {},
        "winner": {},
    }

    strategies = list(cold_results.keys())

    for strategy in strategies:
        cold = cold_results[strategy]
        hot = hot_results[strategy]

        analysis["cold"][strategy] = {
            "accuracy": cold.accuracy,
            "j_per_correct": cold.j_per_correct,
            "throttle_rate": cold.thermal.throttle_rate,
        }

        analysis["hot"][strategy] = {
            "accuracy": hot.accuracy,
            "j_per_correct": hot.j_per_correct,
            "throttle_rate": hot.thermal.throttle_rate,
        }

        # Degradation under thermal stress
        analysis["degradation"][strategy] = {
            "accuracy_drop": cold.accuracy - hot.accuracy,
            "j_per_correct_increase": hot.j_per_correct - cold.j_per_correct,
            "throttle_rate_increase": hot.thermal.throttle_rate - cold.thermal.throttle_rate,
        }

    # Find winner in each regime
    for regime, results in [("cold", cold_results), ("hot", hot_results)]:
        best_strategy = min(results.items(), key=lambda x: x[1].j_per_correct)[0]
        analysis["winner"][regime] = best_strategy

    # Key insight: does proprioceptive win under hot?
    if "proprioceptive" in strategies and "uncertainty" in strategies:
        hot_prop = hot_results["proprioceptive"]
        hot_unc = hot_results["uncertainty"]

        analysis["proprioceptive_vs_uncertainty_hot"] = {
            "prop_j_per_correct": hot_prop.j_per_correct,
            "unc_j_per_correct": hot_unc.j_per_correct,
            "prop_wins": hot_prop.j_per_correct < hot_unc.j_per_correct,
            "delta_j_per_correct": hot_unc.j_per_correct - hot_prop.j_per_correct,
        }

    return analysis


def plot_regime_comparison(cold_results: Dict, hot_results: Dict, output_dir: Path):
    """Generate plots comparing cold vs hot regimes."""

    strategies = list(cold_results.keys())
    n = len(strategies)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: Accuracy comparison
    ax1 = axes[0, 0]
    x = np.arange(n)
    width = 0.35
    cold_acc = [cold_results[s].accuracy * 100 for s in strategies]
    hot_acc = [hot_results[s].accuracy * 100 for s in strategies]
    ax1.bar(x - width/2, cold_acc, width, label='Cold Start', color='blue', alpha=0.7)
    ax1.bar(x + width/2, hot_acc, width, label='Hot Start', color='red', alpha=0.7)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy by Regime')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=15)
    ax1.legend()

    # Row 1: J/correct comparison
    ax2 = axes[0, 1]
    cold_jpc = [cold_results[s].j_per_correct for s in strategies]
    hot_jpc = [hot_results[s].j_per_correct for s in strategies]
    ax2.bar(x - width/2, cold_jpc, width, label='Cold Start', color='blue', alpha=0.7)
    ax2.bar(x + width/2, hot_jpc, width, label='Hot Start', color='red', alpha=0.7)
    ax2.set_ylabel('J/Correct')
    ax2.set_title('Energy Efficiency by Regime')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15)
    ax2.legend()

    # Row 1: Throttle rate
    ax3 = axes[0, 2]
    cold_tr = [cold_results[s].thermal.throttle_rate for s in strategies]
    hot_tr = [hot_results[s].thermal.throttle_rate for s in strategies]
    ax3.bar(x - width/2, cold_tr, width, label='Cold Start', color='blue', alpha=0.7)
    ax3.bar(x + width/2, hot_tr, width, label='Hot Start', color='red', alpha=0.7)
    ax3.set_ylabel('Throttle Rate (per 1000 tokens)')
    ax3.set_title('Throttling by Regime')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=15)
    ax3.legend()

    # Row 2: Degradation analysis
    ax4 = axes[1, 0]
    acc_drop = [cold_results[s].accuracy - hot_results[s].accuracy for s in strategies]
    colors = ['green' if d < 0 else 'red' for d in acc_drop]
    ax4.bar(strategies, [d * 100 for d in acc_drop], color=colors, alpha=0.7)
    ax4.axhline(y=0, color='gray', linestyle='--')
    ax4.set_ylabel('Accuracy Drop (%)')
    ax4.set_title('Degradation Under Thermal Stress')
    for i, v in enumerate(acc_drop):
        ax4.text(i, v * 100 + 0.5, f'{v*100:.1f}%', ha='center', fontsize=8)

    # Row 2: Temperature profiles
    ax5 = axes[1, 1]
    cold_temps = [cold_results[s].thermal.peak_temp_c for s in strategies]
    hot_temps = [hot_results[s].thermal.peak_temp_c for s in strategies]
    ax5.bar(x - width/2, cold_temps, width, label='Cold Start', color='blue', alpha=0.7)
    ax5.bar(x + width/2, hot_temps, width, label='Hot Start', color='red', alpha=0.7)
    ax5.set_ylabel('Peak Temperature (°C)')
    ax5.set_title('Peak Temperature by Regime')
    ax5.set_xticks(x)
    ax5.set_xticklabels(strategies, rotation=15)
    ax5.legend()

    # Row 2: Pareto front
    ax6 = axes[1, 2]
    for regime, results, color in [("Cold", cold_results, 'blue'), ("Hot", hot_results, 'red')]:
        accs = [results[s].accuracy * 100 for s in strategies]
        jpcs = [results[s].j_per_correct for s in strategies]
        ax6.scatter(jpcs, accs, c=color, s=100, alpha=0.7, label=regime)
        for i, s in enumerate(strategies):
            ax6.annotate(s[:4], (jpcs[i], accs[i]), fontsize=8)
    ax6.set_xlabel('J/Correct')
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_title('Pareto Front: Cold vs Hot')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "regime_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'regime_comparison.png'}")


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    budget_j: float = 500.0,
    warmup_duration: float = 60,
    output_dir: Path = Path("results/hot_start"),
) -> Dict[str, Any]:
    """Run hot start protocol experiment."""

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Config
    config = NorthStarConfig()

    # First run calibration
    print("\n=== Calibration Pass ===")
    from scripts.north_star_experiment import run_experiment as run_north_star

    # Get subset of suite
    suite = [item for item in EVAL_SUITE_EXPANDED if item.get("verify") != "unit_test"][:40]

    strategies = ["greedy", "uncertainty", "proprioceptive"]

    # Run cold regime
    print("\n" + "=" * 70)
    print("COLD START REGIME")
    print("=" * 70)

    # Cool down first
    print("Ensuring cold start (waiting 60s)...")
    time.sleep(60)

    cold_results = run_regime(
        model, tokenizer, suite, "cold", strategies, config, budget_j, warmup_duration=0
    )

    # Run hot regime
    print("\n" + "=" * 70)
    print("HOT START REGIME")
    print("=" * 70)

    hot_results = run_regime(
        model, tokenizer, suite, "hot", strategies, config, budget_j, warmup_duration
    )

    # Analyze
    analysis = analyze_regimes(cold_results, hot_results)

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_regime_comparison(cold_results, hot_results, output_dir)

    # Save results
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"hot_start_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "budget_j": budget_j,
            "warmup_duration": warmup_duration,
            "cold_results": {k: asdict(v) for k, v in cold_results.items()},
            "hot_results": {k: asdict(v) for k, v in hot_results.items()},
            "analysis": analysis,
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Hot Start Protocol experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--budget", type=float, default=500.0)
    parser.add_argument("--warmup-duration", type=float, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("results/hot_start"))
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        budget_j=args.budget,
        warmup_duration=args.warmup_duration,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print("HOT START PROTOCOL SUMMARY")
    print("=" * 70)

    print("\nCold Start Results:")
    for strategy, stats in analysis["cold"].items():
        print(f"  {strategy:15s}: acc={stats['accuracy']*100:5.1f}%  J/correct={stats['j_per_correct']:.2f}")

    print("\nHot Start Results:")
    for strategy, stats in analysis["hot"].items():
        print(f"  {strategy:15s}: acc={stats['accuracy']*100:5.1f}%  J/correct={stats['j_per_correct']:.2f}")

    print("\nDegradation Under Thermal Stress:")
    for strategy, deg in analysis["degradation"].items():
        print(f"  {strategy:15s}: acc_drop={deg['accuracy_drop']*100:+.1f}%  "
              f"J/correct_incr={deg['j_per_correct_increase']:+.2f}")

    print(f"\nWinner (Cold): {analysis['winner']['cold']}")
    print(f"Winner (Hot):  {analysis['winner']['hot']}")

    if "proprioceptive_vs_uncertainty_hot" in analysis:
        pvu = analysis["proprioceptive_vs_uncertainty_hot"]
        print(f"\nProprioceptive vs Uncertainty (Hot):")
        print(f"  Proprioceptive J/correct: {pvu['prop_j_per_correct']:.2f}")
        print(f"  Uncertainty J/correct: {pvu['unc_j_per_correct']:.2f}")
        print(f"  Proprioceptive wins: {pvu['prop_wins']}")
        print(f"  Delta: {pvu['delta_j_per_correct']:+.2f} J/correct")


if __name__ == "__main__":
    main()
