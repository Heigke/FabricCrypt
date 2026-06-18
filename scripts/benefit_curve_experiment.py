#!/usr/bin/env python3
"""
Benefit Curve Experiment: Measure what extra compute actually buys.

Key insight from feedback: EVC can't be tuned from vibes.
We need empirical "what do I get if I spend more?"

For each item, run:
- 1 sample @ T=0 (cheap/greedy)
- 3 samples @ T=0.5 (medium)
- 5 samples @ T=0.7 (heavy)
- Optional: self-consistency vote

Then compute:
- Δaccuracy vs ΔJoules
- Conditioned on margin bucket (margin_min in [0-1], [1-2], [2-4], [4+])

This yields the controller's "metabolic law":
> When uncertainty is this high, extra compute buys this much correctness.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

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


# Compute configurations
COMPUTE_CONFIGS = [
    {"name": "cheap", "samples": 1, "temperature": 0.0},
    {"name": "medium", "samples": 3, "temperature": 0.5},
    {"name": "heavy", "samples": 5, "temperature": 0.7},
]

# Margin buckets for analysis
MARGIN_BUCKETS = [
    (0, 1, "very_low"),      # Very uncertain
    (1, 2, "low"),           # Uncertain
    (2, 4, "medium"),        # Moderately confident
    (4, float('inf'), "high"),  # Confident
]


@dataclass
class ItemResult:
    """Results for a single item across all compute configs."""
    question: str
    expected: str
    category: str
    difficulty: str
    margin_min: float
    margin_bucket: str
    configs: Dict[str, Dict] = field(default_factory=dict)


def get_margin_bucket(margin: float) -> str:
    """Get the bucket name for a margin value."""
    for low, high, name in MARGIN_BUCKETS:
        if low <= margin < high:
            return name
    return "high"


def generate_with_config(
    model, tokenizer, prompt: str, max_tokens: int,
    num_samples: int, temperature: float,
) -> Tuple[str, float, float, List[float]]:
    """Generate with specified compute configuration."""

    messages = [{"role": "user", "content": f"{prompt}\nAnswer with just the final answer, no explanation."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    generated_ids = inputs.input_ids.clone()
    margins = []

    recorder = PowerTraceRecorder(sample_interval_ms=10)
    recorder.start()
    start_time = time.perf_counter()

    with torch.no_grad():
        for step in range(max_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Compute margin (always, for tracking)
            top_logits, _ = torch.topk(logits[0], k=2)
            margin = (top_logits[0] - top_logits[1]).item()
            margins.append(margin)

            # Sample based on config
            if num_samples > 1 and temperature > 0:
                # Multi-sample voting
                probs = F.softmax(logits / temperature, dim=-1)
                samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
                # Majority vote
                next_token = samples.mode().values.unsqueeze(0)
            elif temperature > 0:
                # Single sample with temperature
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    end_time = time.perf_counter()
    recorder.stop()

    output_ids = generated_ids[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0
    latency = end_time - start_time

    return output_text, energy, latency, margins


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_tokens: int = 32,
    output_dir: Path = Path("results/benefit_curve"),
) -> Dict[str, Any]:
    """Run benefit curve experiment."""

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
        generate_with_config(model, tokenizer, "Hello", 5, 1, 0.0)

    results = []

    print(f"\n=== Running Benefit Curve Experiment ===")
    print(f"Items: {len(EVAL_SUITE)}")
    print(f"Configs: {[c['name'] for c in COMPUTE_CONFIGS]}")

    for idx, item in enumerate(EVAL_SUITE):
        print(f"\n[{idx+1}/{len(EVAL_SUITE)}] {item['cat']}/{item['diff']}")

        item_result = ItemResult(
            question=item["q"],
            expected=item["a"],
            category=item["cat"],
            difficulty=item["diff"],
            margin_min=0.0,
            margin_bucket="",
        )

        # Run each config
        for config in COMPUTE_CONFIGS:
            output, energy, latency, margins = generate_with_config(
                model, tokenizer, item["q"], max_tokens,
                config["samples"], config["temperature"]
            )

            is_correct = check_answer(output, item["a"])

            item_result.configs[config["name"]] = {
                "output": output[:100],
                "correct": is_correct,
                "energy_j": energy,
                "latency_s": latency,
                "margins": margins,
                "margin_min": min(margins) if margins else 0.0,
                "margin_mean": np.mean(margins) if margins else 0.0,
            }

            status = "✓" if is_correct else "✗"
            print(f"  {config['name']}: {status} {energy:.1f}J")

        # Use cheap config's margin_min as the "base" margin
        item_result.margin_min = item_result.configs["cheap"]["margin_min"]
        item_result.margin_bucket = get_margin_bucket(item_result.margin_min)

        results.append(item_result)

    # Analyze results
    analysis = analyze_results(results)

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]

    plot_benefit_curves(analysis, output_dir / f"benefit_curves_{model_short}.png")
    plot_accuracy_by_margin_bucket(analysis, output_dir / f"accuracy_by_margin_{model_short}.png")
    plot_energy_vs_accuracy_gain(analysis, output_dir / f"energy_vs_accuracy_{model_short}.png")

    # Save results
    output_file = output_dir / f"benefit_curve_{model_short}.json"
    with open(output_file, "w") as f:
        # Convert dataclass results to dict
        results_dict = []
        for r in results:
            results_dict.append({
                "question": r.question,
                "expected": r.expected,
                "category": r.category,
                "difficulty": r.difficulty,
                "margin_min": r.margin_min,
                "margin_bucket": r.margin_bucket,
                "configs": r.configs,
            })

        json.dump({
            "model": model_name,
            "compute_configs": COMPUTE_CONFIGS,
            "margin_buckets": [(l, h, n) for l, h, n in MARGIN_BUCKETS],
            "analysis": analysis,
            "results": results_dict,
        }, f, indent=2)

    print(f"\nSaved: {output_file}")
    return analysis


def analyze_results(results: List[ItemResult]) -> Dict[str, Any]:
    """Analyze benefit curve results."""

    analysis = {
        "overall": {},
        "by_margin_bucket": {},
        "by_category": {},
        "by_difficulty": {},
        "error_reduction": {},  # Key metric for EVC
    }

    # Overall stats per config
    for config in COMPUTE_CONFIGS:
        correct = sum(1 for r in results if r.configs[config["name"]]["correct"])
        energy = sum(r.configs[config["name"]]["energy_j"] for r in results)
        analysis["overall"][config["name"]] = {
            "accuracy": correct / len(results),
            "n_correct": correct,
            "total_energy_j": energy,
            "j_per_correct": energy / correct if correct > 0 else float('inf'),
        }

    # By margin bucket - THE KEY ANALYSIS
    for _, _, bucket_name in MARGIN_BUCKETS:
        bucket_results = [r for r in results if r.margin_bucket == bucket_name]
        if not bucket_results:
            continue

        bucket_analysis = {}
        for config in COMPUTE_CONFIGS:
            correct = sum(1 for r in bucket_results if r.configs[config["name"]]["correct"])
            energy = sum(r.configs[config["name"]]["energy_j"] for r in bucket_results)
            bucket_analysis[config["name"]] = {
                "accuracy": correct / len(bucket_results),
                "n_correct": correct,
                "n_total": len(bucket_results),
                "total_energy_j": energy,
                "avg_energy_j": energy / len(bucket_results),
            }

        analysis["by_margin_bucket"][bucket_name] = bucket_analysis

    # Compute error reduction metrics (for EVC calibration)
    # Error reduction = (cheap_errors - escalated_errors) / cheap_errors
    cheap_errors = sum(1 for r in results if not r.configs["cheap"]["correct"])
    medium_errors = sum(1 for r in results if not r.configs["medium"]["correct"])
    heavy_errors = sum(1 for r in results if not r.configs["heavy"]["correct"])

    if cheap_errors > 0:
        analysis["error_reduction"]["medium_vs_cheap"] = (cheap_errors - medium_errors) / cheap_errors
        analysis["error_reduction"]["heavy_vs_cheap"] = (cheap_errors - heavy_errors) / cheap_errors
    else:
        analysis["error_reduction"]["medium_vs_cheap"] = 0.0
        analysis["error_reduction"]["heavy_vs_cheap"] = 0.0

    # Error reduction by margin bucket
    for _, _, bucket_name in MARGIN_BUCKETS:
        bucket_results = [r for r in results if r.margin_bucket == bucket_name]
        if not bucket_results:
            continue

        bucket_cheap_errors = sum(1 for r in bucket_results if not r.configs["cheap"]["correct"])
        bucket_medium_errors = sum(1 for r in bucket_results if not r.configs["medium"]["correct"])
        bucket_heavy_errors = sum(1 for r in bucket_results if not r.configs["heavy"]["correct"])

        if bucket_cheap_errors > 0:
            analysis["error_reduction"][f"{bucket_name}_medium"] = (
                (bucket_cheap_errors - bucket_medium_errors) / bucket_cheap_errors
            )
            analysis["error_reduction"][f"{bucket_name}_heavy"] = (
                (bucket_cheap_errors - bucket_heavy_errors) / bucket_cheap_errors
            )

    # By category
    for cat in set(r.category for r in results):
        cat_results = [r for r in results if r.category == cat]
        cat_analysis = {}
        for config in COMPUTE_CONFIGS:
            correct = sum(1 for r in cat_results if r.configs[config["name"]]["correct"])
            cat_analysis[config["name"]] = {
                "accuracy": correct / len(cat_results),
                "n_correct": correct,
                "n_total": len(cat_results),
            }
        analysis["by_category"][cat] = cat_analysis

    # By difficulty
    for diff in set(r.difficulty for r in results):
        diff_results = [r for r in results if r.difficulty == diff]
        diff_analysis = {}
        for config in COMPUTE_CONFIGS:
            correct = sum(1 for r in diff_results if r.configs[config["name"]]["correct"])
            diff_analysis[config["name"]] = {
                "accuracy": correct / len(diff_results),
                "n_correct": correct,
                "n_total": len(diff_results),
            }
        analysis["by_difficulty"][diff] = diff_analysis

    return analysis


def plot_benefit_curves(analysis: Dict, output_path: Path):
    """Plot accuracy vs energy for each margin bucket."""

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'very_low': 'red', 'low': 'orange', 'medium': 'blue', 'high': 'green'}
    markers = {'very_low': 'o', 'low': 's', 'medium': '^', 'high': 'D'}

    for bucket_name, bucket_data in analysis["by_margin_bucket"].items():
        energies = []
        accuracies = []
        for config in COMPUTE_CONFIGS:
            if config["name"] in bucket_data:
                energies.append(bucket_data[config["name"]]["avg_energy_j"])
                accuracies.append(bucket_data[config["name"]]["accuracy"])

        if energies:
            n = bucket_data[COMPUTE_CONFIGS[0]["name"]]["n_total"]
            ax.plot(energies, accuracies, f'-{markers.get(bucket_name, "o")}',
                    color=colors.get(bucket_name, 'gray'),
                    label=f'{bucket_name} (n={n})', linewidth=2, markersize=8)

    ax.set_xlabel('Average Energy per Item (J)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Benefit Curves: Accuracy vs Energy by Uncertainty Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_by_margin_bucket(analysis: Dict, output_path: Path):
    """Plot accuracy improvement by margin bucket."""

    fig, ax = plt.subplots(figsize=(10, 6))

    bucket_names = list(analysis["by_margin_bucket"].keys())
    x = np.arange(len(bucket_names))
    width = 0.25

    for i, config in enumerate(COMPUTE_CONFIGS):
        accuracies = []
        for bucket in bucket_names:
            if config["name"] in analysis["by_margin_bucket"][bucket]:
                accuracies.append(analysis["by_margin_bucket"][bucket][config["name"]]["accuracy"])
            else:
                accuracies.append(0)

        ax.bar(x + i * width, accuracies, width, label=config["name"])

    ax.set_xlabel('Margin Bucket (Uncertainty Level)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Margin Bucket and Compute Level')
    ax.set_xticks(x + width)
    ax.set_xticklabels(bucket_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_energy_vs_accuracy_gain(analysis: Dict, output_path: Path):
    """Plot energy cost vs accuracy gain from escalation."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Overall error reduction
    ax1 = axes[0]
    reductions = analysis["error_reduction"]

    configs = ["medium_vs_cheap", "heavy_vs_cheap"]
    values = [reductions.get(c, 0) * 100 for c in configs]

    ax1.bar(["Medium (3 samples)", "Heavy (5 samples)"], values, color=['orange', 'red'])
    ax1.set_ylabel('Error Reduction (%)')
    ax1.set_title('Overall Error Reduction from Escalation')
    ax1.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(values):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center')

    # Right: Error reduction by margin bucket
    ax2 = axes[1]

    bucket_names = []
    medium_reductions = []
    heavy_reductions = []

    for _, _, bucket in MARGIN_BUCKETS:
        if f"{bucket}_medium" in reductions:
            bucket_names.append(bucket)
            medium_reductions.append(reductions[f"{bucket}_medium"] * 100)
            heavy_reductions.append(reductions[f"{bucket}_heavy"] * 100)

    if bucket_names:
        x = np.arange(len(bucket_names))
        width = 0.35

        ax2.bar(x - width/2, medium_reductions, width, label='Medium (3 samples)', color='orange')
        ax2.bar(x + width/2, heavy_reductions, width, label='Heavy (5 samples)', color='red')

        ax2.set_xlabel('Margin Bucket (Uncertainty)')
        ax2.set_ylabel('Error Reduction (%)')
        ax2.set_title('Error Reduction by Uncertainty Level')
        ax2.set_xticks(x)
        ax2.set_xticklabels(bucket_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benefit curve experiment: measure compute benefit")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/benefit_curve"))
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("BENEFIT CURVE SUMMARY")
    print("=" * 60)

    print("\n--- Overall Performance ---")
    for config, data in analysis["overall"].items():
        print(f"  {config}: {data['accuracy']*100:.1f}% accuracy, {data['total_energy_j']:.1f}J total")

    print("\n--- Error Reduction (Key for EVC) ---")
    reductions = analysis["error_reduction"]
    print(f"  Medium (3 samples) reduces errors by: {reductions.get('medium_vs_cheap', 0)*100:.1f}%")
    print(f"  Heavy (5 samples) reduces errors by: {reductions.get('heavy_vs_cheap', 0)*100:.1f}%")

    print("\n--- By Margin Bucket ---")
    for bucket, data in analysis["by_margin_bucket"].items():
        cheap_acc = data.get("cheap", {}).get("accuracy", 0)
        heavy_acc = data.get("heavy", {}).get("accuracy", 0)
        n = data.get("cheap", {}).get("n_total", 0)
        print(f"  {bucket} (n={n}): cheap={cheap_acc*100:.1f}% → heavy={heavy_acc*100:.1f}% ({(heavy_acc-cheap_acc)*100:+.1f}%)")

    print("\n--- Metabolic Law (for EVC config) ---")
    print("  Use these error reduction values in EVCConfig:")
    print(f"    medium_error_reduction = {reductions.get('medium_vs_cheap', 0):.2f}")
    print(f"    heavy_error_reduction = {reductions.get('heavy_vs_cheap', 0):.2f}")


if __name__ == "__main__":
    main()
