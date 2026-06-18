#!/usr/bin/env python3
"""Plot budget curves: quality vs energy Pareto frontiers."""

import json
from pathlib import Path
import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available")


def plot_quality_vs_energy(results: dict, output_dir: Path):
    """Plot accuracy vs energy consumption."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (model_file, model_data) in enumerate(results.items()):
        if ax_idx >= 2:
            break

        ax = axes[ax_idx]
        model_name = model_file.replace("budget_curves_", "").replace(".json", "")

        # Get aggregated data
        configs = {k: v for k, v in model_data.items()
                   if k.startswith("temp_") and "budget" in k}

        temps = sorted(set(v["temperature"] for v in configs.values()))
        budgets = sorted(set(v["budget"] for v in configs.values()))

        colors = {'0.0': '#3498db', '0.3': '#2ecc71', '0.7': '#f1c40f', '1.0': '#e74c3c'}
        markers = {16: 's', 32: 'o', 64: '^', 128: 'D'}

        for key, data in configs.items():
            temp = data["temperature"]
            budget = data["budget"]
            accuracy = data["accuracy"] * 100
            energy = data["energy_mean_j"]

            color = colors.get(str(temp), '#95a5a6')
            marker = markers.get(budget, 'o')

            ax.scatter(energy, accuracy, c=color, marker=marker, s=150,
                       label=f"T={temp}, B={budget}", edgecolors='black', linewidths=1)

        ax.set_xlabel('Energy (Joules)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{model_name}\nQuality vs Energy', fontsize=13)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

        # Annotate Pareto-optimal region
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.annotate('Random guess (50%)', xy=(ax.get_xlim()[1]*0.7, 50),
                    fontsize=9, color='gray')

    plt.suptitle('Budget Curves: Quality vs Energy Trade-off\n'
                 '(Higher accuracy + Lower energy = Better)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'budget_quality_vs_energy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'budget_quality_vs_energy.png'}")


def plot_by_category(results: dict, output_dir: Path):
    """Plot accuracy by task category."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    categories = ["math", "qa", "code"]
    x = np.arange(len(categories))
    width = 0.35

    for idx, model_file in enumerate(models[:2]):
        model_data = results[model_file]
        model_name = model_file.replace("budget_curves_", "").replace(".json", "")

        by_cat = model_data.get("by_category", {})
        accuracies = [by_cat.get(cat, {}).get("accuracy", 0) * 100 for cat in categories]

        offset = width * (idx - 0.5)
        bars = ax.bar(x + offset, accuracies, width, label=model_name, edgecolor='black')

        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Task Category', fontsize=12)
    ax.set_title('Accuracy by Task Category', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in categories])
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'budget_accuracy_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'budget_accuracy_by_category.png'}")


def plot_joules_per_correct(results: dict, output_dir: Path):
    """Plot Joules per correct answer - the key efficiency metric."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_file, model_data in results.items():
        model_name = model_file.replace("budget_curves_", "").replace(".json", "")

        configs = {k: v for k, v in model_data.items()
                   if k.startswith("temp_") and "budget" in k}

        # Sort by budget
        items = sorted(configs.items(), key=lambda x: (x[1]["temperature"], x[1]["budget"]))

        labels = []
        jpc_values = []

        for key, data in items:
            jpc = data.get("joules_per_correct", float('inf'))
            if jpc < 1000:  # Filter out infinity
                labels.append(f"T{data['temperature']}/B{data['budget']}")
                jpc_values.append(jpc)

        x = range(len(labels))
        ax.bar(x, jpc_values, label=model_name, alpha=0.7, edgecolor='black')

    ax.set_ylabel('Joules per Correct Answer', fontsize=12)
    ax.set_xlabel('Configuration (Temperature/Budget)', fontsize=12)
    ax.set_title('Energy Efficiency: Joules per Correct Answer\n(Lower is Better)', fontsize=13)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'budget_joules_per_correct.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'budget_joules_per_correct.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot budget curves results")
    parser.add_argument("--results-dir", type=Path, default=Path("results/budget_curves"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/budget_curves"))
    args = parser.parse_args()

    # Load all results
    results = {}
    for result_file in args.results_dir.glob("budget_curves_*.json"):
        with open(result_file) as f:
            data = json.load(f)
            results[result_file.name] = data.get("aggregated", data)

    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(results)} result files")

    plot_quality_vs_energy(results, args.output_dir)
    plot_by_category(results, args.output_dir)
    plot_joules_per_correct(results, args.output_dir)


if __name__ == "__main__":
    main()
