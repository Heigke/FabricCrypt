#!/usr/bin/env python3
"""
Expected Value of Compute (EVC) Experiment.

Produces the three key figures from research feedback:
1. Budget curves: Accuracy vs Joules (Pareto frontier)
2. Joules-per-correct by category/difficulty
3. Calibration curve of p(error) predictor

This is the "killer experiment" for the research claim:
"Use cheap internal correctness predictor to decide when to escalate compute,
achieving better accuracy-per-Joule than static policies."
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from src.energy_harness.expected_value_compute import (
    ExpectedValueComputeController,
    EVCConfig,
    ComputeLevel,
    compute_marginal_utility,
)

from scripts.correctness_prediction_experiment import EVAL_SUITE, check_answer


def generate_with_evc(
    model, tokenizer, prompt: str, max_tokens: int,
    controller: ExpectedValueComputeController,
    remaining_budget_frac: float = 1.0,
) -> tuple:
    """Generate with Expected Value of Compute control."""

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    generated_ids = inputs.input_ids.clone()

    stats = {
        "levels_used": [],
        "margins": [],
        "p_errors": [],
        "evc_values": [],
    }

    with torch.no_grad():
        for step in range(max_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Compute margin
            top_logits, _ = torch.topk(logits[0], k=2)
            margin = (top_logits[0] - top_logits[1]).item()

            # Get EVC decision
            level, decision = controller.decide_level(margin, remaining_budget_frac)
            params = controller.get_sampling_params(level)

            stats["levels_used"].append(level.value)
            stats["margins"].append(margin)
            stats["p_errors"].append(decision["p_error"])
            stats["evc_values"].append(decision["evc_used"])

            # Sample based on level
            num_samples = params.get("num_samples", 1)

            if num_samples > 1:
                temp = params.get("temperature", 0.7)
                if temp > 0:
                    probs = F.softmax(logits / temp, dim=-1)
                    samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
                    # Majority vote
                    next_token = samples.mode().values.unsqueeze(0)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                temp = params.get("temperature", 0)
                if temp > 0 and params.get("do_sample", False):
                    probs = F.softmax(logits / temp, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    output_ids = generated_ids[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, stats


def generate_baseline(model, tokenizer, prompt: str, max_tokens: int) -> tuple:
    """Standard greedy baseline."""
    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_ids = outputs[0, inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text, {"mode": "baseline"}


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens: int = 32,
    output_dir: Path = Path("results/evc"),
) -> Dict[str, Any]:
    """Run EVC experiment with comprehensive analysis."""
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

    # Warmup
    print("Warming up...")
    for _ in range(3):
        generate_baseline(model, tokenizer, "Hello", 5)

    results = {
        "baseline": [],
        "evc": [],
    }

    # === Baseline Run ===
    print("\n=== Baseline (Greedy) ===")

    for idx, item in enumerate(EVAL_SUITE):
        recorder = PowerTraceRecorder(sample_interval_ms=10)
        recorder.start()
        start = time.perf_counter()

        output, stats = generate_baseline(model, tokenizer, item["q"], max_tokens)

        end = time.perf_counter()
        recorder.stop()

        is_correct = check_answer(output, item["a"])
        energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

        results["baseline"].append({
            "question": item["q"],
            "expected": item["a"],
            "output": output[:100],
            "correct": is_correct,
            "energy_j": energy,
            "time_s": end - start,
            "category": item["cat"],
            "difficulty": item["diff"],
        })

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1}/{len(EVAL_SUITE)}] {status} {item['cat']}/{item['diff']}: {energy:.1f}J")

    # === EVC Run ===
    print("\n=== Expected Value of Compute ===")
    controller = ExpectedValueComputeController(budget_j=500.0)

    for idx, item in enumerate(EVAL_SUITE):
        remaining_frac = 1 - (controller.state.energy_spent_j / controller.state.budget_j)

        recorder = PowerTraceRecorder(sample_interval_ms=10)
        recorder.start()
        start = time.perf_counter()

        output, stats = generate_with_evc(
            model, tokenizer, item["q"], max_tokens, controller, remaining_frac
        )

        end = time.perf_counter()
        recorder.stop()

        is_correct = check_answer(output, item["a"])
        energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

        # Record outcome for calibration
        controller.record_outcome(not is_correct, energy)

        results["evc"].append({
            "question": item["q"],
            "expected": item["a"],
            "output": output[:100],
            "correct": is_correct,
            "energy_j": energy,
            "time_s": end - start,
            "category": item["cat"],
            "difficulty": item["diff"],
            "levels_used": stats["levels_used"],
            "avg_p_error": sum(stats["p_errors"]) / len(stats["p_errors"]) if stats["p_errors"] else 0,
        })

        level_counts = defaultdict(int)
        for l in stats["levels_used"]:
            level_counts[l] += 1

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1}/{len(EVAL_SUITE)}] {status} {item['cat']}/{item['diff']}: "
              f"{energy:.1f}J, levels={dict(level_counts)}")

    # === Analysis ===
    analysis = analyze_results(results, controller)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]
    output_file = output_dir / f"evc_{model_short}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "n_items": len(EVAL_SUITE),
            "analysis": analysis,
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")

    # Generate plots
    generate_plots(results, analysis, output_dir, model_short)

    return analysis


def analyze_results(results: Dict[str, List], controller) -> Dict[str, Any]:
    """Comprehensive analysis for the three key figures."""
    analysis = {}

    # === Figure 1: Budget Curves (Accuracy vs Joules) ===
    budget_curves = {}
    for method in ["baseline", "evc"]:
        items = results[method]
        sorted_items = sorted(items, key=lambda x: x["energy_j"])

        cumulative_correct = 0
        cumulative_energy = 0
        curve_points = []

        for r in sorted_items:
            cumulative_energy += r["energy_j"]
            if r["correct"]:
                cumulative_correct += 1
            curve_points.append({
                "energy_j": cumulative_energy,
                "correct": cumulative_correct,
                "accuracy": cumulative_correct / (len(curve_points) + 1),
            })

        budget_curves[method] = curve_points

    analysis["budget_curves"] = budget_curves

    # === Figure 2: Joules-per-correct by difficulty ===
    jpc_by_diff = {}
    for method in ["baseline", "evc"]:
        items = results[method]
        by_diff = defaultdict(list)

        for r in items:
            by_diff[r["difficulty"]].append(r)

        jpc_by_diff[method] = {}
        for diff, rs in by_diff.items():
            correct_items = [r for r in rs if r["correct"]]
            if correct_items:
                total_energy = sum(r["energy_j"] for r in correct_items)
                jpc_by_diff[method][diff] = {
                    "n_correct": len(correct_items),
                    "n_total": len(rs),
                    "accuracy": len(correct_items) / len(rs),
                    "total_energy_j": total_energy,
                    "joules_per_correct": total_energy / len(correct_items),
                }

    analysis["jpc_by_difficulty"] = jpc_by_diff

    # === Figure 3: Calibration ===
    analysis["calibration"] = controller.get_calibration_metrics()

    # === Summary statistics ===
    for method in ["baseline", "evc"]:
        items = results[method]
        n_correct = sum(1 for r in items if r["correct"])
        total_energy = sum(r["energy_j"] for r in items)

        analysis[method] = {
            "accuracy": n_correct / len(items),
            "n_correct": n_correct,
            "n_total": len(items),
            "total_energy_j": total_energy,
            "joules_per_correct": total_energy / n_correct if n_correct > 0 else float('inf'),
        }

    # Marginal utility
    baseline = analysis["baseline"]
    evc = analysis["evc"]
    analysis["marginal_utility"] = compute_marginal_utility(
        results["evc"],
        baseline["accuracy"],
        baseline["total_energy_j"],
    )

    # EVC-specific stats
    analysis["evc_summary"] = controller.get_summary()

    return analysis


def generate_plots(results: Dict, analysis: Dict, output_dir: Path, model_name: str):
    """Generate the three key figures."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # === Figure 1: Budget Curves ===
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, color, label in [
        ("baseline", "#3498db", "Baseline (Greedy)"),
        ("evc", "#e74c3c", "EVC (Introspective)"),
    ]:
        curve = analysis["budget_curves"][method]
        energies = [p["energy_j"] for p in curve]
        accuracies = [p["accuracy"] * 100 for p in curve]
        ax.plot(energies, accuracies, color=color, linewidth=2, label=label)

    ax.set_xlabel("Cumulative Energy (Joules)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Budget Curve: Accuracy vs Energy\n{model_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"budget_curve_{model_name}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / f'budget_curve_{model_name}.png'}")

    # === Figure 2: Joules per Correct by Difficulty ===
    fig, ax = plt.subplots(figsize=(10, 6))

    difficulties = ["easy", "medium", "hard"]
    x = np.arange(len(difficulties))
    width = 0.35

    baseline_jpc = [
        analysis["jpc_by_difficulty"]["baseline"].get(d, {}).get("joules_per_correct", 0)
        for d in difficulties
    ]
    evc_jpc = [
        analysis["jpc_by_difficulty"]["evc"].get(d, {}).get("joules_per_correct", 0)
        for d in difficulties
    ]

    ax.bar(x - width/2, baseline_jpc, width, label="Baseline", color="#3498db")
    ax.bar(x + width/2, evc_jpc, width, label="EVC", color="#e74c3c")

    ax.set_xlabel("Difficulty", fontsize=12)
    ax.set_ylabel("Joules per Correct Answer", fontsize=12)
    ax.set_title(f"Energy Efficiency by Difficulty\n{model_name}", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars, values in [(ax.containers[0], baseline_jpc), (ax.containers[1], evc_jpc)]:
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / f"jpc_by_difficulty_{model_name}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / f'jpc_by_difficulty_{model_name}.png'}")

    # === Figure 3: Calibration Curve ===
    cal = analysis.get("calibration", {})
    reliability = cal.get("reliability_data", [])

    if reliability:
        fig, ax = plt.subplots(figsize=(8, 8))

        predicted = [r["avg_predicted"] for r in reliability]
        actual = [r["avg_actual"] for r in reliability]
        counts = [r["count"] for r in reliability]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')

        # Actual calibration
        scatter = ax.scatter(predicted, actual, s=[c*20 for c in counts],
                            c='#e74c3c', alpha=0.7, edgecolors='black')

        ax.set_xlabel("Predicted p(error)", fontsize=12)
        ax.set_ylabel("Observed Error Rate", fontsize=12)
        ax.set_title(f"Calibration of Error Predictor\n{model_name}\n"
                    f"ECE={cal.get('ece', 0):.3f}, Brier={cal.get('brier_score', 0):.3f}",
                    fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"calibration_{model_name}.png", dpi=150)
        plt.close()
        print(f"Saved: {output_dir / f'calibration_{model_name}.png'}")


def main():
    parser = argparse.ArgumentParser(description="Expected Value of Compute experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/evc"))
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("EXPECTED VALUE OF COMPUTE ANALYSIS")
    print("=" * 60)

    for method in ["baseline", "evc"]:
        if method not in analysis:
            continue
        data = analysis[method]
        print(f"\n{method}:")
        print(f"  Accuracy: {data['accuracy']*100:.1f}% ({data['n_correct']}/{data['n_total']})")
        print(f"  Total energy: {data['total_energy_j']:.1f}J")
        print(f"  J/Correct: {data['joules_per_correct']:.1f}")

    if "marginal_utility" in analysis:
        mu = analysis["marginal_utility"]
        print(f"\n--- Marginal Utility ---")
        print(f"  Δ Accuracy: {mu['delta_accuracy']*100:+.1f}%")
        print(f"  Δ Energy: {mu['delta_energy_j']:+.1f}J")
        print(f"  Marginal utility: {mu['marginal_utility']:.4f} accuracy/J")
        print(f"  Interpretation: {mu['interpretation']}")

    if "calibration" in analysis:
        cal = analysis["calibration"]
        print(f"\n--- Calibration ---")
        print(f"  ECE: {cal.get('ece', 'N/A'):.4f}")
        print(f"  Brier score: {cal.get('brier_score', 'N/A'):.4f}")

    if "evc_summary" in analysis:
        evc = analysis["evc_summary"]
        print(f"\n--- EVC Decision Distribution ---")
        for level, pct in evc.get("escalation_distribution", {}).items():
            print(f"  {level}: {pct:.1f}%")


if __name__ == "__main__":
    main()
