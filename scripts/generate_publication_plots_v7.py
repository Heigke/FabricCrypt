#!/usr/bin/env python3
"""
FEEL Publication Plots v7.0
===========================

Generates publication-quality plots for the v7 publication battery results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(path: str = "results/feel_experiments/publication_v7_results.json"):
    """Load publication results."""
    with open(path) as f:
        return json.load(f)


def plot_accuracy_by_condition(results: dict, save_path: str):
    """Plot accuracy by condition with 95% CI error bars."""
    fig, ax = plt.subplots(figsize=(12, 6))

    conditions = results.get("conditions", {})
    names = []
    accuracies = []
    ci_low = []
    ci_high = []

    # Order conditions meaningfully
    order = ["baseline", "feel", "feel_off", "random_feel", "shuffled",
             "cross_prompt_swap", "hardware_only", "internal_only"]

    for name in order:
        if name in conditions:
            cond = conditions[name]
            names.append(name.replace("_", "\n"))
            acc = cond["accuracy"]
            accuracies.append(acc)
            ci = cond["accuracy_ci"]
            ci_low.append(acc - ci[1])
            ci_high.append(ci[2] - acc)

    x = np.arange(len(names))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    bars = ax.bar(x, accuracies, yerr=[ci_low, ci_high], capsize=5,
                  color=colors, edgecolor='black', linewidth=1, alpha=0.8)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("FEEL v7.0 Publication Battery: Accuracy by Condition (95% CI)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, max(accuracies) * 1.3)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ci_high[i] + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=accuracies[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_category_breakdown(results: dict, save_path: str):
    """Plot accuracy breakdown by prompt category."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    conditions = results.get("conditions", {})
    categories = ["math", "factual", "coding", "open_ended"]
    titles = ["Math", "Factual", "Coding", "Open-Ended"]

    for ax, category, title in zip(axes.flat, categories, titles):
        cond_names = []
        accuracies = []
        ci_low = []
        ci_high = []

        order = ["baseline", "feel", "shuffled", "random_feel"]
        for name in order:
            if name in conditions:
                cond = conditions[name]
                per_cat = cond.get("per_category", {})
                if category in per_cat:
                    cat_data = per_cat[category]
                    cond_names.append(name.replace("_", "\n"))
                    acc = cat_data["accuracy"]
                    accuracies.append(acc)
                    ci = cat_data["accuracy_ci"]
                    ci_low.append(acc - ci[1])
                    ci_high.append(ci[2] - acc)

        x = np.arange(len(cond_names))
        colors = ['steelblue', 'forestgreen', 'coral', 'mediumpurple'][:len(cond_names)]

        bars = ax.bar(x, accuracies, yerr=[ci_low, ci_high], capsize=4,
                      color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title} (n={per_cat[category]['n']})", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cond_names, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{acc:.2f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle("FEEL v7.0: Per-Category Accuracy Breakdown", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_lag_sweep(results: dict, save_path: str):
    """Plot accuracy across lag values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    lag_sweep = results.get("lag_sweep", {})
    baseline = results.get("conditions", {}).get("baseline", {})

    lags = sorted([int(k) for k in lag_sweep.keys()])
    accuracies = []
    ci_low = []
    ci_high = []

    for k in lags:
        data = lag_sweep[str(k)]
        acc = data["accuracy"]
        accuracies.append(acc)
        ci = data["accuracy_ci"]
        ci_low.append(acc - ci[1])
        ci_high.append(ci[2] - acc)

    ax.errorbar(lags, accuracies, yerr=[ci_low, ci_high],
                marker='o', markersize=8, capsize=5, linewidth=2,
                color='steelblue', label='FEEL with Lag')

    # Baseline reference
    if baseline:
        base_acc = baseline["accuracy"]
        ax.axhline(y=base_acc, color='gray', linestyle='--', alpha=0.7, label='Baseline (no FEEL)')

    ax.set_xlabel("Lag (k steps)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("FEEL v7.0: Lag Sweep Ablation\n(Expected: Accuracy should degrade as lag increases)", fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_xticks(lags)
    ax.set_xticklabels(lags)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Note about expected behavior
    ax.text(0.98, 0.02, "Note: Flat line indicates no measurable\nFEEL effect at current alpha level",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_telemetry_validity(results: dict, save_path: str):
    """Plot telemetry validity status."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    telem = results.get("telemetry_validity", {})

    # Left: Availability
    ax1 = axes[0]
    availability = telem.get("availability", {})
    channels = list(availability.keys())
    avail_values = [availability[c] * 100 for c in channels]
    colors = ['forestgreen' if v > 80 else 'coral' for v in avail_values]

    bars = ax1.bar(channels, avail_values, color=colors, edgecolor='black', alpha=0.8)
    ax1.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% threshold')
    ax1.set_ylabel("Availability (%)")
    ax1.set_title("Telemetry Channel Availability", fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.legend()

    for bar, val in zip(bars, avail_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10)

    # Right: Validity status
    ax2 = axes[1]
    valid = telem.get("valid", {})
    channels = list(valid.keys())
    valid_values = [1 if valid[c] else 0 for c in channels]
    colors = ['forestgreen' if v else 'coral' for v in valid_values]

    bars = ax2.bar(channels, valid_values, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel("Valid (1=yes, 0=no)")
    ax2.set_title("Telemetry Channel Validity", fontsize=12)
    ax2.set_ylim(0, 1.3)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Invalid', 'Valid'])

    for bar, val, c in zip(bars, valid_values, channels):
        label = "VALID" if val else "INVALID"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    source = telem.get("source", "unknown")
    n_samples = telem.get("n_samples", 0)
    fig.suptitle(f"FEEL v7.0: Telemetry Validity Report\nSource: {source}, Samples: {n_samples}",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_benefit_analysis(results: dict, save_path: str):
    """Plot benefit analysis summary."""
    fig, ax = plt.subplots(figsize=(10, 6))

    benefit = results.get("benefit_analysis", {})

    # Create summary table
    feel_benefit = benefit.get("feel_benefit_accuracy", [0, 0, 0])
    collapse_shuffled = benefit.get("benefit_collapse_shuffled", False)
    collapse_random = benefit.get("benefit_collapse_random", False)
    collapse_cross = benefit.get("benefit_collapse_cross_prompt", False)

    # Left side: Benefit bar
    labels = ['FEEL Benefit']
    values = [feel_benefit[0]]
    ci_low = [feel_benefit[0] - feel_benefit[1]]
    ci_high = [feel_benefit[2] - feel_benefit[0]]

    x = [0]
    bars = ax.barh(x, values, xerr=[ci_low, ci_high], capsize=5,
                   color='steelblue', edgecolor='black', height=0.4, alpha=0.8)

    ax.axvline(x=0, color='gray', linestyle='-', linewidth=2)
    ax.set_yticks([0])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Accuracy Benefit (FEEL - Baseline)")
    ax.set_title("FEEL v7.0: Benefit Analysis", fontsize=14, fontweight='bold')
    ax.set_xlim(-0.15, 0.15)

    # Add collapse status as text
    collapse_text = f"""Falsification Tests (Benefit Collapse):

    Shuffled sensors: {'PASS' if collapse_shuffled else 'FAIL'}
    Random direction: {'PASS' if collapse_random else 'FAIL'}
    Cross-prompt swap: {'PASS' if collapse_cross else 'FAIL'}

Overall: {'ALL PASS' if (collapse_shuffled and collapse_random and collapse_cross) else 'SOME FAIL'}"""

    ax.text(0.98, 0.5, collapse_text, transform=ax.transAxes, ha='right', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_dashboard(results: dict, save_path: str):
    """Create comprehensive summary dashboard."""
    fig = plt.figure(figsize=(16, 12))

    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Main accuracy comparison (top row, full width)
    ax1 = fig.add_subplot(gs[0, :])
    conditions = results.get("conditions", {})
    order = ["baseline", "feel", "feel_off", "random_feel", "shuffled", "cross_prompt_swap"]
    names = []
    accuracies = []
    ece_values = []

    for name in order:
        if name in conditions:
            cond = conditions[name]
            names.append(name.replace("_", " ").title())
            accuracies.append(cond["accuracy"])
            ece_values.append(cond["ece"])

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', alpha=0.8)
    ax1.set_ylabel('Accuracy', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, 0.5)

    ax1b = ax1.twinx()
    bars2 = ax1b.bar(x + width/2, ece_values, width, label='ECE', color='coral', alpha=0.8)
    ax1b.set_ylabel('ECE (lower=better)', color='coral')
    ax1b.tick_params(axis='y', labelcolor='coral')
    ax1b.set_ylim(0, 0.25)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.set_title('Accuracy and Calibration by Condition', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1b.legend(loc='upper right')

    # 2. Category breakdown (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    baseline = conditions.get("baseline", {}).get("per_category", {})
    cats = ["math", "factual", "coding", "open_ended"]
    cat_acc = [baseline.get(c, {}).get("accuracy", 0) for c in cats]
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(cats)))

    bars = ax2.bar(cats, cat_acc, color=colors, edgecolor='black')
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Per-Category Accuracy\n(Baseline)", fontsize=11)
    ax2.set_ylim(0, 0.6)
    for bar, acc in zip(bars, cat_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)

    # 3. Lag sweep (middle center)
    ax3 = fig.add_subplot(gs[1, 1])
    lag_sweep = results.get("lag_sweep", {})
    lags = sorted([int(k) for k in lag_sweep.keys()])
    lag_acc = [lag_sweep[str(k)]["accuracy"] for k in lags]

    ax3.plot(lags, lag_acc, 'o-', markersize=8, linewidth=2, color='forestgreen')
    ax3.axhline(y=accuracies[0], color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlabel("Lag (k)")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Lag Sweep\n(Expected: decay)", fontsize=11)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(lags)
    ax3.set_xticklabels(lags)

    # 4. Telemetry status (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    telem = results.get("telemetry_validity", {})
    valid = telem.get("valid", {})
    channels = list(valid.keys())
    valid_colors = ['forestgreen' if valid[c] else 'coral' for c in channels]

    ax4.bar(channels, [1]*len(channels), color=valid_colors, edgecolor='black', alpha=0.8)
    ax4.set_ylim(0, 1.2)
    ax4.set_title(f"Telemetry Validity\n({telem.get('source', 'unknown')})", fontsize=11)
    ax4.set_ylabel("Valid")
    for i, (c, v) in enumerate(zip(channels, valid.values())):
        ax4.text(i, 1.05, "Y" if v else "N", ha='center', fontweight='bold')

    # 5. Key metrics table (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')

    n_prompts = results.get("n_prompts", 0)
    n_bootstrap = results.get("n_bootstrap", 0)
    timestamp = results.get("timestamp", "")[:19]

    table_text = f"""Publication Battery Summary

Prompts: {n_prompts}
Bootstrap: {n_bootstrap}
Timestamp: {timestamp}
Model: DeepSeek-R1-1.5B
Alpha: 0.000436"""

    ax5.text(0.5, 0.5, table_text, transform=ax5.transAxes, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 6. Falsification results (bottom center)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    benefit = results.get("benefit_analysis", {})
    collapse_shuffled = benefit.get("benefit_collapse_shuffled", False)
    collapse_random = benefit.get("benefit_collapse_random", False)
    collapse_cross = benefit.get("benefit_collapse_cross_prompt", False)

    def checkmark(v): return "PASS" if v else "FAIL"

    falsif_text = f"""Falsification Tests

Shuffled:     {checkmark(collapse_shuffled)}
Random:       {checkmark(collapse_random)}
Cross-prompt: {checkmark(collapse_cross)}

FEEL Benefit: {benefit.get('feel_benefit_accuracy', [0])[0]:+.4f}"""

    color = 'lightgreen' if all([collapse_shuffled, collapse_random, collapse_cross]) else 'lightyellow'
    ax6.text(0.5, 0.5, falsif_text, transform=ax6.transAxes, ha='center', va='center',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    # 7. Interpretation (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    interp_text = """Interpretation

Current Results:
- No measurable FEEL benefit
- Ablations collapse as expected
- Baseline stable (reproducible)

Next Steps:
- Increase alpha for larger effect
- Train projector for task benefit
- Add stochastic sampling"""

    ax7.text(0.5, 0.5, interp_text, transform=ax7.transAxes, ha='center', va='center',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle("FEEL v7.0 Publication Battery Dashboard", fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all publication plots."""
    results_path = "results/feel_experiments/publication_v7_results.json"
    output_dir = Path("results/feel_experiments/plots_v7")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    print(f"\nGenerating publication plots in: {output_dir}")

    plot_accuracy_by_condition(results, str(output_dir / "accuracy_by_condition_v7.png"))
    plot_per_category_breakdown(results, str(output_dir / "per_category_breakdown_v7.png"))
    plot_lag_sweep(results, str(output_dir / "lag_sweep_v7.png"))
    plot_telemetry_validity(results, str(output_dir / "telemetry_validity_v7.png"))
    plot_benefit_analysis(results, str(output_dir / "benefit_analysis_v7.png"))
    plot_summary_dashboard(results, str(output_dir / "v7_summary_dashboard.png"))

    print(f"\nGenerated 6 plots in {output_dir}")


if __name__ == "__main__":
    main()
