#!/usr/bin/env python3
"""
North Star Experiment: The definitive test for thermo-proprioceptive compute allocation.

Run the eval suite under 3 conditions:
1. Greedy baseline (fixed compute)
2. Uncertainty-only (margin-calibrated adaptive sampling)
3. Uncertainty + proprioception (reduce escalation when hot/throttling/low budget)

Report:
- Accuracy
- Joules
- J/correct
- Calibration (ECE)
- Thermal constraint violations
- Pareto front

If #3 dominates #2 under stress, we've demonstrated "feeling" (physical constraints shaping cognition).
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from scripts.correctness_prediction_experiment import EVAL_SUITE, check_answer


@dataclass
class NorthStarConfig:
    """Configuration for North Star experiment."""
    # Platt calibration parameters (from calibration experiment)
    # p(error) = sigmoid(platt_coef * margin + platt_intercept)
    platt_coef: float = -0.6428  # From calibration on 0.5B - will update for 3B
    platt_intercept: float = -0.0008

    # Escalation thresholds based on p(error)
    p_error_threshold_medium: float = 0.3  # Escalate to medium if p(error) > 0.3
    p_error_threshold_heavy: float = 0.5   # Escalate to heavy if p(error) > 0.5

    # Proprioceptive constraints
    temp_throttle_c: float = 80.0      # Start backing off escalation
    power_throttle_w: float = 80.0     # Power limit for throttling
    budget_reserve_fraction: float = 0.2  # Reserve 20% of budget

    # Compute configurations
    samples_greedy: int = 1
    samples_medium: int = 3
    samples_heavy: int = 5

    temp_greedy: float = 0.0
    temp_medium: float = 0.5
    temp_heavy: float = 0.7


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def predict_p_error(margin: float, config: NorthStarConfig) -> float:
    """Predict p(error) from margin using Platt calibration."""
    logit = config.platt_coef * margin + config.platt_intercept
    return sigmoid(logit)


def decide_compute_level_uncertainty(
    margin: float,
    config: NorthStarConfig,
) -> Tuple[int, float, str]:
    """Decide compute level based on uncertainty only."""
    p_error = predict_p_error(margin, config)

    if p_error > config.p_error_threshold_heavy:
        return config.samples_heavy, config.temp_heavy, "heavy"
    elif p_error > config.p_error_threshold_medium:
        return config.samples_medium, config.temp_medium, "medium"
    else:
        return config.samples_greedy, config.temp_greedy, "greedy"


def decide_compute_level_proprioceptive(
    margin: float,
    config: NorthStarConfig,
    current_temp: float,
    current_power: float,
    remaining_budget_fraction: float,
) -> Tuple[int, float, str, Dict[str, Any]]:
    """Decide compute level with proprioceptive constraints."""
    p_error = predict_p_error(margin, config)

    # Start with uncertainty-based decision
    if p_error > config.p_error_threshold_heavy:
        target_level = "heavy"
        target_samples = config.samples_heavy
        target_temp = config.temp_heavy
    elif p_error > config.p_error_threshold_medium:
        target_level = "medium"
        target_samples = config.samples_medium
        target_temp = config.temp_medium
    else:
        target_level = "greedy"
        target_samples = config.samples_greedy
        target_temp = config.temp_greedy

    # Apply proprioceptive constraints
    constraint_info = {
        "p_error": p_error,
        "target_level": target_level,
        "actual_level": target_level,
        "temp_constraint": False,
        "power_constraint": False,
        "budget_constraint": False,
    }

    # Thermal constraint: back off if hot
    if current_temp > config.temp_throttle_c:
        if target_level == "heavy":
            target_level = "medium"
            target_samples = config.samples_medium
            target_temp = config.temp_medium
            constraint_info["temp_constraint"] = True
        elif target_level == "medium":
            target_level = "greedy"
            target_samples = config.samples_greedy
            target_temp = config.temp_greedy
            constraint_info["temp_constraint"] = True

    # Power constraint: back off if power high
    if current_power > config.power_throttle_w:
        if target_level == "heavy":
            target_level = "medium"
            target_samples = config.samples_medium
            target_temp = config.temp_medium
            constraint_info["power_constraint"] = True
        elif target_level == "medium":
            target_level = "greedy"
            target_samples = config.samples_greedy
            target_temp = config.temp_greedy
            constraint_info["power_constraint"] = True

    # Budget constraint: back off if low budget
    if remaining_budget_fraction < config.budget_reserve_fraction:
        target_level = "greedy"
        target_samples = config.samples_greedy
        target_temp = config.temp_greedy
        constraint_info["budget_constraint"] = True

    constraint_info["actual_level"] = target_level
    return target_samples, target_temp, target_level, constraint_info


def generate_with_strategy(
    model, tokenizer, prompt: str, max_tokens: int,
    strategy: str, config: NorthStarConfig,
    budget_j: float = None, spent_j: float = 0.0,
) -> Tuple[str, float, Dict]:
    """Generate with specified strategy."""

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    generated_ids = inputs.input_ids.clone()

    stats = {
        "levels": [],
        "margins": [],
        "p_errors": [],
        "constraints_applied": [],
    }

    recorder = PowerTraceRecorder(sample_interval_ms=10)
    recorder.start()

    with torch.no_grad():
        for step in range(max_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Compute margin
            top_logits, _ = torch.topk(logits[0], k=2)
            margin = (top_logits[0] - top_logits[1]).item()
            stats["margins"].append(margin)

            # Get current thermal state
            if recorder.samples:
                current_power = recorder.samples[-1].power_watts
                current_temp = min(85, 50 + current_power * 0.3)  # Estimate
            else:
                current_power = 50.0
                current_temp = 55.0

            # Decide compute level based on strategy
            if strategy == "greedy":
                num_samples = 1
                temperature = 0.0
                level = "greedy"
            elif strategy == "uncertainty":
                num_samples, temperature, level = decide_compute_level_uncertainty(margin, config)
                stats["p_errors"].append(predict_p_error(margin, config))
            elif strategy == "proprioceptive":
                remaining_fraction = (budget_j - spent_j) / budget_j if budget_j else 1.0
                num_samples, temperature, level, constraint_info = decide_compute_level_proprioceptive(
                    margin, config, current_temp, current_power, remaining_fraction
                )
                stats["p_errors"].append(constraint_info["p_error"])
                stats["constraints_applied"].append(constraint_info)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            stats["levels"].append(level)

            # Sample
            if num_samples > 1 and temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
                next_token = samples.mode().values.unsqueeze(0)
            elif temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    recorder.stop()

    output_ids = generated_ids[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

    return output_text, energy, stats


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    budget_j: float = 500.0,
    output_dir: Path = Path("results/north_star"),
) -> Dict[str, Any]:
    """Run North Star experiment."""

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

    # First run calibration to get proper Platt parameters for this model
    print("\n=== Running Calibration Pass ===")
    config = NorthStarConfig()

    # Collect margin/correct data for calibration
    margins = []
    errors = []

    for idx, item in enumerate(EVAL_SUITE):
        output, energy, stats = generate_with_strategy(
            model, tokenizer, item["q"], 32, "greedy", config
        )
        is_correct = check_answer(output, item["a"])
        margin_min = min(stats["margins"]) if stats["margins"] else 0.0

        margins.append(margin_min)
        errors.append(not is_correct)

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1}/{len(EVAL_SUITE)}] {status} margin={margin_min:.2f}")

    # Fit Platt calibrator
    from sklearn.linear_model import LogisticRegression
    margins_np = np.array(margins).reshape(-1, 1)
    errors_np = np.array(errors, dtype=float)

    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(margins_np, errors_np)

    # Update config with calibrated parameters
    config.platt_coef = float(platt.coef_[0][0])
    config.platt_intercept = float(platt.intercept_[0])

    print(f"\nCalibrated Platt parameters:")
    print(f"  coef: {config.platt_coef:.4f}")
    print(f"  intercept: {config.platt_intercept:.4f}")

    # Compute calibration metrics
    p_errors_pred = np.array([sigmoid(config.platt_coef * m + config.platt_intercept) for m in margins])
    from sklearn.metrics import brier_score_loss, roc_auc_score
    brier = brier_score_loss(errors_np, p_errors_pred)
    auc = roc_auc_score(errors_np, p_errors_pred) if len(np.unique(errors_np)) > 1 else None

    print(f"\nCalibration metrics:")
    print(f"  AUC: {auc:.3f}" if auc else "  AUC: N/A")
    print(f"  Brier: {brier:.3f}")

    # Now run the three strategies
    strategies = ["greedy", "uncertainty", "proprioceptive"]
    results = {s: [] for s in strategies}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'='*60}")

        spent_j = 0.0

        for idx, item in enumerate(EVAL_SUITE):
            if spent_j >= budget_j:
                print(f"  Budget exhausted at item {idx}")
                break

            output, energy, stats = generate_with_strategy(
                model, tokenizer, item["q"], 32, strategy, config,
                budget_j=budget_j, spent_j=spent_j
            )

            is_correct = check_answer(output, item["a"])
            spent_j += energy

            # Summarize levels used
            level_counts = defaultdict(int)
            for l in stats["levels"]:
                level_counts[l] += 1

            results[strategy].append({
                "correct": is_correct,
                "energy_j": energy,
                "cumulative_j": spent_j,
                "levels": dict(level_counts),
                "margin_min": min(stats["margins"]) if stats["margins"] else 0,
                "constraints": stats.get("constraints_applied", []),
            })

            status = "✓" if is_correct else "✗"
            print(f"  [{idx+1}] {status} {energy:.1f}J (cum: {spent_j:.1f}J) {dict(level_counts)}")

    # Analysis
    analysis = analyze_results(results, budget_j)
    analysis["calibration"] = {
        "platt_coef": config.platt_coef,
        "platt_intercept": config.platt_intercept,
        "auc": float(auc) if auc else None,
        "brier": float(brier),
    }

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]

    plot_pareto_front(results, output_dir / f"pareto_front_{model_short}.png")
    plot_strategy_comparison(analysis, output_dir / f"strategy_comparison_{model_short}.png")

    # Save results
    output_file = output_dir / f"north_star_{model_short}.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "budget_j": budget_j,
            "config": {
                "platt_coef": config.platt_coef,
                "platt_intercept": config.platt_intercept,
                "p_error_threshold_medium": config.p_error_threshold_medium,
                "p_error_threshold_heavy": config.p_error_threshold_heavy,
            },
            "analysis": analysis,
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")
    return analysis


def analyze_results(results: Dict[str, List], budget_j: float) -> Dict[str, Any]:
    """Analyze North Star results."""
    analysis = {}

    for strategy, items in results.items():
        if not items:
            continue

        n_correct = sum(1 for r in items if r["correct"])
        total_energy = sum(r["energy_j"] for r in items)
        n_items = len(items)

        # Count constraint applications
        constraint_counts = {"temp": 0, "power": 0, "budget": 0}
        for r in items:
            for c in r.get("constraints", []):
                if c.get("temp_constraint"):
                    constraint_counts["temp"] += 1
                if c.get("power_constraint"):
                    constraint_counts["power"] += 1
                if c.get("budget_constraint"):
                    constraint_counts["budget"] += 1

        # Level distribution
        level_counts = defaultdict(int)
        for r in items:
            for level, count in r.get("levels", {}).items():
                level_counts[level] += count

        total_tokens = sum(level_counts.values())
        level_fractions = {k: v / total_tokens * 100 for k, v in level_counts.items()} if total_tokens > 0 else {}

        analysis[strategy] = {
            "items_completed": n_items,
            "accuracy": n_correct / n_items if n_items > 0 else 0,
            "n_correct": n_correct,
            "total_energy_j": total_energy,
            "j_per_correct": total_energy / n_correct if n_correct > 0 else float('inf'),
            "budget_utilization": total_energy / budget_j,
            "level_distribution": dict(level_fractions),
            "constraint_applications": constraint_counts,
        }

    # Comparison
    if "greedy" in analysis and "proprioceptive" in analysis:
        greedy = analysis["greedy"]
        prop = analysis["proprioceptive"]

        analysis["comparison"] = {
            "accuracy_delta": prop["accuracy"] - greedy["accuracy"],
            "energy_delta": prop["total_energy_j"] - greedy["total_energy_j"],
            "j_per_correct_delta": prop["j_per_correct"] - greedy["j_per_correct"],
            "proprioceptive_dominates": (
                prop["accuracy"] >= greedy["accuracy"] and
                prop["total_energy_j"] <= greedy["total_energy_j"]
            ),
        }

    return analysis


def plot_pareto_front(results: Dict, output_path: Path):
    """Plot Pareto front of accuracy vs energy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"greedy": "blue", "uncertainty": "orange", "proprioceptive": "green"}
    markers = {"greedy": "o", "uncertainty": "s", "proprioceptive": "^"}

    for strategy, items in results.items():
        if not items:
            continue

        energies = [r["cumulative_j"] for r in items]
        accuracies = []
        correct_count = 0
        for r in items:
            if r["correct"]:
                correct_count += 1
            accuracies.append(correct_count / (items.index(r) + 1))

        ax.plot(energies, accuracies, f'-{markers[strategy]}',
                color=colors[strategy], label=strategy, linewidth=2, markersize=4)

    ax.set_xlabel('Cumulative Energy (J)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Pareto Front: Accuracy vs Energy by Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_strategy_comparison(analysis: Dict, output_path: Path):
    """Plot strategy comparison bars."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    strategies = [s for s in ["greedy", "uncertainty", "proprioceptive"] if s in analysis]
    if not strategies:
        return

    # Accuracy
    ax1 = axes[0]
    accuracies = [analysis[s]["accuracy"] * 100 for s in strategies]
    colors = ["blue", "orange", "green"]
    ax1.bar(strategies, accuracies, color=colors[:len(strategies)])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy by Strategy')
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center')

    # J/Correct
    ax2 = axes[1]
    j_per_correct = [analysis[s]["j_per_correct"] for s in strategies]
    ax2.bar(strategies, j_per_correct, color=colors[:len(strategies)])
    ax2.set_ylabel('J/Correct')
    ax2.set_title('Energy Efficiency')
    for i, v in enumerate(j_per_correct):
        ax2.text(i, v + 0.5, f'{v:.1f}', ha='center')

    # Level distribution (for uncertainty and proprioceptive)
    ax3 = axes[2]
    for i, s in enumerate(strategies):
        if s == "greedy":
            continue
        levels = analysis[s].get("level_distribution", {})
        bottoms = 0
        for level in ["greedy", "medium", "heavy"]:
            if level in levels:
                ax3.bar(s, levels[level], bottom=bottoms, label=level if i == 1 else "")
                bottoms += levels[level]
    ax3.set_ylabel('% of Tokens')
    ax3.set_title('Compute Level Distribution')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="North Star experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--budget", type=float, default=500.0, help="Energy budget in Joules")
    parser.add_argument("--output-dir", type=Path, default=Path("results/north_star"))
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        budget_j=args.budget,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("NORTH STAR EXPERIMENT SUMMARY")
    print("=" * 60)

    for strategy in ["greedy", "uncertainty", "proprioceptive"]:
        if strategy not in analysis:
            continue
        data = analysis[strategy]
        print(f"\n{strategy.upper()}:")
        print(f"  Accuracy: {data['accuracy']*100:.1f}% ({data['n_correct']} correct)")
        print(f"  Energy: {data['total_energy_j']:.1f}J")
        print(f"  J/Correct: {data['j_per_correct']:.1f}")
        if "level_distribution" in data:
            print(f"  Levels: {data['level_distribution']}")

    if "comparison" in analysis:
        comp = analysis["comparison"]
        print(f"\n--- Proprioceptive vs Greedy ---")
        print(f"  Accuracy delta: {comp['accuracy_delta']*100:+.1f}%")
        print(f"  Energy delta: {comp['energy_delta']:+.1f}J")
        print(f"  J/Correct delta: {comp['j_per_correct_delta']:+.1f}")
        print(f"  Proprioceptive dominates: {comp['proprioceptive_dominates']}")


if __name__ == "__main__":
    main()
