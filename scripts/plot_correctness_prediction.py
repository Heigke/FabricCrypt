#!/usr/bin/env python3
"""Plot correctness prediction results - the key research finding."""

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


def plot_auc_comparison(results: dict, output_dir: Path):
    """Plot AUC comparison: signals predicting errors vs TPOT."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = list(results.keys())
    signals = ["margin_mean", "margin_min", "entropy_max", "delta_mean", "delta_max"]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    for ax_idx, model_file in enumerate(models[:2]):
        ax = axes[ax_idx]
        model_name = model_file.replace("correctness_prediction_", "").replace(".json", "")
        data = results[model_file]

        # Get AUCs for temp=0.0
        analysis = data.get("analysis", {})
        temp_data = analysis.get("temp_0.0", {})

        signal_names = []
        auc_values = []

        for sig in signals:
            auc_key = f"auc_{sig}"
            if auc_key in temp_data and temp_data[auc_key] is not None:
                signal_names.append(sig.replace("_", "\n"))
                auc_values.append(temp_data[auc_key])

        if not auc_values:
            continue

        x = np.arange(len(signal_names))
        bars = ax.bar(x, auc_values, color=colors[:len(auc_values)], edgecolor='black')

        # Color bars by predictive power
        for bar, val in zip(bars, auc_values):
            if val > 0.7:
                bar.set_facecolor('#2ecc71')  # Green - strong
            elif val > 0.6:
                bar.set_facecolor('#f1c40f')  # Yellow - moderate
            else:
                bar.set_facecolor('#e74c3c')  # Red - weak

        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random (0.5)')
        ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Strong (0.7)')

        ax.set_ylabel('AUC for Error Prediction', fontsize=12)
        ax.set_xlabel('Signal', fontsize=12)
        ax.set_title(f'{model_name}\nT=0.0, Accuracy={temp_data.get("accuracy", 0)*100:.1f}%', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(signal_names, fontsize=10)
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, auc_values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('KEY FINDING: Latent Signals Predict ERRORS (not TPOT)\n'
                 'margin_min AUC=0.94 for 3B model - near-perfect error predictor!',
                 fontsize=13, fontweight='bold', color='green')
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'correctness_prediction_auc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'correctness_prediction_auc.png'}")


def plot_research_pivot(output_dir: Path):
    """Plot the research pivot: TPOT prediction vs Error prediction."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Data from experiments
    categories = ['Predict TPOT Spikes\n(Result 10)', 'Predict Errors\n(Result 12)']
    models = ['0.5B', '3B']

    # AUC values (best signal for each)
    tpot_aucs = [0.52, 0.51]  # From signal_prediction experiment
    error_aucs = [0.72, 0.94]  # margin_min from correctness_prediction

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, [tpot_aucs[0], error_aucs[0]], width, label='0.5B', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, [tpot_aucs[1], error_aucs[1]], width, label='3B', color='#e74c3c', edgecolor='black')

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Random')
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Strong predictor')

    ax.set_ylabel('AUC (Best Signal)', fontsize=12)
    ax.set_title('Research Pivot: What Do Latent Signals Predict?', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            color = 'green' if height > 0.7 else ('black' if height > 0.55 else 'red')
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=color)

    # Add annotations
    ax.annotate('AUC ≈ 0.5\n(Random!)',
                xy=(0, 0.52), xytext=(-0.5, 0.65),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.annotate('AUC = 0.94\n(Near-perfect!)',
                xy=(1.2, 0.94), xytext=(1.5, 0.8),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig(output_dir / 'research_pivot_tpot_vs_error.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'research_pivot_tpot_vs_error.png'}")


def plot_accuracy_by_difficulty(results: dict, output_dir: Path):
    """Plot accuracy by difficulty level."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    difficulties = ['easy', 'medium', 'hard']
    x = np.arange(len(difficulties))
    width = 0.35

    for idx, model_file in enumerate(models[:2]):
        model_name = model_file.replace("correctness_prediction_", "").replace(".json", "")
        data = results[model_file]
        analysis = data.get("analysis", {})
        temp_data = analysis.get("temp_0.0", {})
        by_diff = temp_data.get("by_difficulty", {})

        accuracies = [by_diff.get(d, {}).get("accuracy", 0) * 100 for d in difficulties]

        offset = width * (idx - 0.5)
        bars = ax.bar(x + offset, accuracies, width, label=model_name, edgecolor='black')

        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Difficulty', fontsize=12)
    ax.set_title('Accuracy by Difficulty Level (T=0.0)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'correctness_accuracy_by_difficulty.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'correctness_accuracy_by_difficulty.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot correctness prediction results")
    parser.add_argument("--results-dir", type=Path, default=Path("results/correctness_prediction"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/correctness_prediction"))
    args = parser.parse_args()

    # Load all results
    results = {}
    for result_file in args.results_dir.glob("correctness_prediction_*.json"):
        with open(result_file) as f:
            results[result_file.name] = json.load(f)

    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(results)} result files")

    plot_auc_comparison(results, args.output_dir)
    plot_research_pivot(args.output_dir)
    plot_accuracy_by_difficulty(results, args.output_dir)


if __name__ == "__main__":
    main()
