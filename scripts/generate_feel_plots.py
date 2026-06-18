#!/usr/bin/env python3
"""
FEEL Publication Plots Generator
=================================
Generates publication-grade plots from breakthrough experiment results.

Run: python scripts/generate_feel_plots.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'feel_on': '#2ecc71',   # Green
    'feel_off': '#e74c3c',  # Red
    'baseline': '#3498db',  # Blue
    'permuted': '#9b59b6',  # Purple
    'swapped': '#f39c12',   # Orange
    'ci': '#95a5a6',        # Gray
}


def load_results(path: str = "results/feel_experiments/breakthrough_results.json"):
    """Load experiment results."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_aux_head_control(results: dict, output_dir: Path):
    """Plot aux head control experiment results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = results["aux_head_control"]
    summary = data["summary"]

    # Bar chart: correlation comparison
    ax = axes[0]
    x = ['Alpha = 0\n(FEEL OFF)', 'Alpha > 0\n(FEEL ON)']
    y = [summary["alpha_0_avg_corr"], summary["alpha_pos_avg_corr"]]
    colors = [COLORS['feel_off'], COLORS['feel_on']]

    bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Hidden State - Entropy Correlation')
    ax.set_title('Experiment 1: Aux Head Control\n(Does FEEL improve hidden state predictability?)')

    for bar, val in zip(bars, y):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   ha='center', va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')

    # KL comparison
    ax = axes[1]
    x = ['Alpha = 0', 'Alpha > 0']
    y = [summary["alpha_0_avg_kl"], summary["alpha_pos_avg_kl"]]

    bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average KL Divergence')
    ax.set_title('KL Divergence (FEEL influence)')

    for bar, val in zip(bars, y):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'exp1_aux_head_control.png')
    plt.close()


def plot_counterfactual(results: dict, output_dir: Path):
    """Plot counterfactual experiment results."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    data = results["counterfactual"]
    raw = data["raw_data"]

    # Hidden state divergence
    ax = axes[0]
    divs = [r["hidden_state_divergence"] for r in raw]
    ax.bar(range(len(divs)), divs, color=COLORS['baseline'], edgecolor='black')
    ax.axhline(y=np.mean(divs), color='red', linestyle='--', label=f'Mean: {np.mean(divs):.3f}')
    ax.set_xlabel('Prompt Index')
    ax.set_ylabel('Hidden State Divergence')
    ax.set_title('Hidden State Δ (same tokens, swapped sensors)')
    ax.legend()

    # KL difference
    ax = axes[1]
    kl_diffs = [r["kl_difference"] for r in raw]
    ax.bar(range(len(kl_diffs)), kl_diffs, color=COLORS['swapped'], edgecolor='black')
    ax.axhline(y=np.mean(kl_diffs), color='red', linestyle='--', label=f'Mean: {np.mean(kl_diffs):.4f}')
    ax.set_xlabel('Prompt Index')
    ax.set_ylabel('|KL_baseline - KL_swapped|')
    ax.set_title('KL Divergence Change')
    ax.legend()

    # Logit delta difference
    ax = axes[2]
    logit_diffs = [r["logit_delta_diff"] for r in raw]
    ax.bar(range(len(logit_diffs)), logit_diffs, color=COLORS['permuted'], edgecolor='black')
    ax.axhline(y=np.mean(logit_diffs), color='red', linestyle='--', label=f'Mean: {np.mean(logit_diffs):.3f}')
    ax.set_xlabel('Prompt Index')
    ax.set_ylabel('|Δlogit_baseline - Δlogit_swapped|')
    ax.set_title('Prediction Change')
    ax.legend()

    fig.suptitle('Experiment 2: Counterfactual Test (Same Tokens, Different Sensors)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'exp2_counterfactual.png')
    plt.close()


def plot_extensive_suite(results: dict, output_dir: Path):
    """Plot extensive suite with bootstrap CIs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    data = results["extensive_suite"]
    summary = data["summary"]
    per_prompt = data["per_prompt"]

    metrics = ["avg_kl", "max_kl", "p95_kl", "avg_logit_delta"]
    titles = ["Average KL Divergence", "Maximum KL Divergence", "95th Percentile KL", "Average |Δlogit|"]

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        values = [p[metric] for p in per_prompt]
        stats = summary[metric]

        # Histogram
        ax.hist(values, bins=15, color=COLORS['baseline'], edgecolor='black', alpha=0.7)

        # CI shading
        ax.axvline(stats["mean"], color='red', linestyle='-', linewidth=2, label=f'Mean: {stats["mean"]:.4f}')
        ax.axvspan(stats["ci_lower"], stats["ci_upper"], alpha=0.2, color='red',
                  label=f'95% CI: [{stats["ci_lower"]:.4f}, {stats["ci_upper"]:.4f}]')

        ax.set_xlabel(title)
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend(loc='upper right')

    fig.suptitle(f'Experiment 3: Extensive Suite ({data["n_prompts"]} prompts, {data["n_bootstrap"]} bootstrap)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'exp3_extensive_suite.png')
    plt.close()


def plot_falsification(results: dict, output_dir: Path):
    """Plot falsification battery results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data = results["falsification"]
    summary = data["summary"]

    # Ratio comparison
    ax = axes[0]
    conditions = ['Baseline', 'Permuted', 'Cross-Swap']
    ratios = [1.0, summary["permute_ratio"], summary["cross_swap_ratio"]]
    colors = [COLORS['baseline'], COLORS['permuted'], COLORS['swapped']]

    bars = ax.bar(conditions, ratios, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('KL Ratio (vs Baseline)')
    ax.set_title('Falsification: Sensor Manipulation Effects')
    ax.set_ylim(0, max(ratios) * 1.3)

    for bar, val in zip(bars, ratios):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Lag sweep
    ax = axes[1]
    lag_ratios = summary["lag_ratios"]
    lags = sorted([int(k) for k in lag_ratios.keys()])
    ratios = [lag_ratios[str(lag)] for lag in lags]

    ax.plot(lags, ratios, 'o-', color=COLORS['baseline'], markersize=10, linewidth=2)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Lag (tokens)')
    ax.set_ylabel('KL Ratio (vs lag=0)')
    ax.set_title('Lag Sweep: Temporal Coupling')
    ax.set_xticks(lags)

    for lag, ratio in zip(lags, ratios):
        ax.annotate(f'{ratio:.3f}', xy=(lag, ratio), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9)

    fig.suptitle('Experiment 4: Strengthened Falsification Battery', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'exp4_falsification.png')
    plt.close()


def plot_predictive_test(results: dict, output_dir: Path):
    """Plot ridge regression predictive test."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = results["predictive_test"]
    summary = data["summary"]

    # R² comparison
    conditions = ['Sensors Only', 'z_feel Only', 'Combined']
    r2_train = [summary["r2_sensors_only"], summary["r2_z_feel_only"], summary["r2_combined"]]
    r2_cv = [summary["cv_sensors_only"], summary["cv_z_feel_only"], summary["cv_combined"]]

    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, r2_train, width, label='R² (Train)', color=COLORS['baseline'], edgecolor='black')
    bars2 = ax.bar(x + width/2, r2_cv, width, label='R² (5-fold CV)', color=COLORS['feel_on'], edgecolor='black')

    ax.set_ylabel('R² Score')
    ax.set_title('Experiment 5: Ridge Regression Predictive Test\n(Predicting Future Entropy)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # Annotate bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)

    # Add incremental gain annotation
    gain = summary["z_feel_incremental_gain"]
    ax.annotate(f'Incremental Gain: {gain:.4f}', xy=(0.7, 0.95), xycoords='axes fraction',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'exp5_predictive_test.png')
    plt.close()


def plot_compute_multiplier(results: dict, output_dir: Path):
    """Plot effective compute multiplier."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = results["compute_multiplier"]
    summary = data["summary"]

    # Entropy comparison
    ax = axes[0]
    conditions = ['FEEL OFF\n(1 sample)', 'FEEL OFF\n(2 samples)', 'FEEL ON\n(1 sample)']
    entropies = [
        summary["feel_off_1_avg_entropy"],
        summary["feel_off_2_avg_entropy"],
        summary["feel_on_avg_entropy"]
    ]
    colors = [COLORS['feel_off'], COLORS['feel_off'], COLORS['feel_on']]

    bars = ax.bar(conditions, entropies, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Entropy')
    ax.set_title('Prediction Entropy (lower = more confident)')

    for bar, val in zip(bars, entropies):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Multiplier visualization
    ax = axes[1]
    multiplier = summary["effective_compute_multiplier"]
    reduction = summary["entropy_reduction_pct"]

    # Create gauge-like visualization
    theta = np.linspace(0, np.pi, 100)
    r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, 'k-', linewidth=2)

    # Fill based on multiplier
    fill_angle = min(np.pi, (multiplier - 1) / 3 * np.pi)  # Map 1-4x to 0-π
    theta_fill = np.linspace(0, fill_angle, 50)
    x_fill = np.concatenate([[0], r * np.cos(theta_fill), [0]])
    y_fill = np.concatenate([[0], r * np.sin(theta_fill), [0]])
    ax.fill(x_fill, y_fill, color=COLORS['feel_on'], alpha=0.6)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.1, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0, 0.5, f'{multiplier:.2f}x', ha='center', va='center', fontsize=36, fontweight='bold')
    ax.text(0, 0.1, f'Entropy reduction: {reduction:.1f}%', ha='center', va='center', fontsize=12)
    ax.set_title('Effective Compute Multiplier', fontsize=14)

    fig.suptitle('Experiment 6: Effective Compute Multiplier', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'exp6_compute_multiplier.png')
    plt.close()


def plot_gpu_interoception(results: dict, output_dir: Path):
    """Plot GPU interoception results."""
    if "gpu_interoception" not in results:
        print("  Skipping GPU interoception plot (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = results["gpu_interoception"]
    summary = data["summary"]

    # Correlation bars
    ax = axes[0]
    metrics = ['z_feel-Temp', 'z_feel-Power', 'Sensors-Temp']
    values = [
        summary["z_feel_temp_correlation"],
        summary["z_feel_power_correlation"],
        summary["sensor_temp_correlation"]
    ]
    colors = [COLORS['feel_on'] if abs(v) > 0.3 else COLORS['ci'] for v in values]

    bars = ax.barh(metrics, values, color=colors, edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Threshold (0.3)')
    ax.axvline(x=-0.3, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Correlation')
    ax.set_title('z_feel - GPU State Correlations')
    ax.set_xlim(-1, 1)
    ax.legend()

    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.3f}', xy=(val, bar.get_y() + bar.get_height()/2),
                   ha='left' if val > 0 else 'right', va='center', fontsize=10)

    # Counterfactual identifiability
    ax = axes[1]
    if "counterfactual_results" in summary:
        cf = summary["counterfactual_results"]
        n_toks = [r["n_tokens"] for r in cf]
        z_means = [r["z_feel_mean"] for r in cf]
        z_vars = [r["z_feel_var"] for r in cf]

        ax.plot(n_toks, z_means, 'o-', color=COLORS['baseline'], label='z_feel mean', markersize=10)
        ax2 = ax.twinx()
        ax2.plot(n_toks, z_vars, 's--', color=COLORS['permuted'], label='z_feel var', markersize=10)

        ax.set_xlabel('Tokens Generated')
        ax.set_ylabel('z_feel Mean', color=COLORS['baseline'])
        ax2.set_ylabel('z_feel Variance', color=COLORS['permuted'])
        ax.set_title('Counterfactual Identifiability\n(same prompt, different compute load)')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Evidence source badge
    evidence = summary["evidence_source"]
    color = COLORS['feel_on'] if evidence != "NONE" else COLORS['feel_off']
    ax.annotate(f'Evidence: {evidence}', xy=(0.98, 0.02), xycoords='axes fraction',
               ha='right', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    fig.suptitle('Experiment 7: Deep GPU Interoception', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'exp7_gpu_interoception.png')
    plt.close()


def plot_final_summary(results: dict, output_dir: Path):
    """Plot final verdict summary."""
    fig, ax = plt.subplots(figsize=(10, 8))

    summary = results["final_summary"]
    verdicts = summary["verdicts"]

    # Create verdict table
    tests = list(verdicts.keys())
    passed = [verdicts[t] for t in tests]

    # Format test names
    test_labels = [
        'Causal Channel Real',
        'Counterfactual Works',
        'Falsification Passed',
        'z_feel Adds Value',
        'Compute Benefit',
        'GPU Interoception',
    ]

    y_pos = np.arange(len(tests))
    colors = [COLORS['feel_on'] if p else COLORS['feel_off'] for p in passed]

    bars = ax.barh(y_pos, [1]*len(tests), color=colors, edgecolor='black', linewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(test_labels, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    # Add pass/fail labels
    for i, (bar, p) in enumerate(zip(bars, passed)):
        status = '✓ PASS' if p else '✗ FAIL'
        ax.annotate(status, xy=(0.5, bar.get_y() + bar.get_height()/2),
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='white')

    # Overall verdict
    overall = summary["overall_pass"]
    overall_text = "✓ BREAKTHROUGH" if overall else "✗ MORE WORK NEEDED"
    overall_color = COLORS['feel_on'] if overall else COLORS['feel_off']

    ax.text(0.5, -0.15, overall_text, transform=ax.transAxes, ha='center',
           fontsize=20, fontweight='bold', color=overall_color,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=overall_color, linewidth=3))

    # Stats
    n_pass = sum(passed)
    ax.text(0.5, 1.05, f'Passed: {n_pass}/{len(tests)}', transform=ax.transAxes,
           ha='center', fontsize=14)

    ax.set_title('FEEL Breakthrough Experiments - Final Verdict', fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'final_summary.png')
    plt.close()


def generate_all_plots(results_path: str = "results/feel_experiments/breakthrough_results.json"):
    """Generate all plots."""
    print("="*60)
    print("  FEEL Publication Plots Generator")
    print("="*60)

    output_dir = Path("results/feel_experiments/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)

    print("\n  Generating plots...")

    plot_aux_head_control(results, output_dir)
    print("    ✓ exp1_aux_head_control.png")

    plot_counterfactual(results, output_dir)
    print("    ✓ exp2_counterfactual.png")

    plot_extensive_suite(results, output_dir)
    print("    ✓ exp3_extensive_suite.png")

    plot_falsification(results, output_dir)
    print("    ✓ exp4_falsification.png")

    plot_predictive_test(results, output_dir)
    print("    ✓ exp5_predictive_test.png")

    plot_compute_multiplier(results, output_dir)
    print("    ✓ exp6_compute_multiplier.png")

    plot_gpu_interoception(results, output_dir)
    print("    ✓ exp7_gpu_interoception.png")

    plot_final_summary(results, output_dir)
    print("    ✓ final_summary.png")

    print(f"\n  All plots saved to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    generate_all_plots()
