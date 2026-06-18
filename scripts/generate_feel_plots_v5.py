#!/usr/bin/env python3
"""
Generate publication-quality plots for FEEL v5.0 experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

def load_results():
    """Load v5.0 results."""
    with open("results/feel_experiments/breakthrough_v5_results.json") as f:
        return json.load(f)

def plot_teacher_forced(results, save_path):
    """Plot teacher-forced counterfactual results."""
    data = results["teacher_forced_counterfactual"]["raw_data"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # L2 divergence
    l2_vals = [d["avg_l2_divergence"] for d in data]
    axes[0].bar(range(len(l2_vals)), l2_vals, color='steelblue', alpha=0.8)
    axes[0].set_xlabel("Prompt Index")
    axes[0].set_ylabel("L2 Hidden Divergence")
    axes[0].set_title("Teacher-Forced: Hidden State Divergence")
    axes[0].axhline(y=np.mean(l2_vals), color='red', linestyle='--', label=f'Mean: {np.mean(l2_vals):.2f}')
    axes[0].legend()

    # KL divergence
    kl_vals = [d["avg_kl"] * 1000 for d in data]  # Scale for visibility
    axes[1].bar(range(len(kl_vals)), kl_vals, color='darkorange', alpha=0.8)
    axes[1].set_xlabel("Prompt Index")
    axes[1].set_ylabel("KL Divergence (×1000)")
    axes[1].set_title("Teacher-Forced: KL Divergence")
    axes[1].axhline(y=np.mean(kl_vals), color='red', linestyle='--', label=f'Mean: {np.mean(kl_vals):.2f}')
    axes[1].legend()

    # Cosine similarity
    cos_vals = [d["avg_cosine_sim"] for d in data]
    axes[2].bar(range(len(cos_vals)), cos_vals, color='seagreen', alpha=0.8)
    axes[2].set_xlabel("Prompt Index")
    axes[2].set_ylabel("Cosine Similarity")
    axes[2].set_title("Teacher-Forced: Cosine Similarity")
    axes[2].set_ylim(0.9, 1.0)
    axes[2].axhline(y=np.mean(cos_vals), color='red', linestyle='--', label=f'Mean: {np.mean(cos_vals):.4f}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_falsification(results, save_path):
    """Plot falsification battery results."""
    summary = results["falsification"]["summary"]

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['Baseline\n(Aligned)', 'Permuted\n(Shuffled)']
    values = [summary["baseline_avg_kl"] * 1000, summary["permuted_avg_kl"] * 1000]
    colors = ['steelblue', 'coral']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')

    # Add ratio annotation
    ratio = summary["permute_ratio"]
    ax.annotate(f'{ratio:.2f}×', xy=(1, values[1]), xytext=(0.5, values[1] * 1.1),
                ha='center', fontsize=14, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_ylabel("KL Divergence (×1000)")
    ax.set_title("Falsification: Sensor Permutation Effect\n(Higher = sensors matter)")

    # Add pass/fail indicator
    status = "PASS" if summary["falsification_passed"] else "FAIL"
    color = "green" if summary["falsification_passed"] else "red"
    ax.text(0.98, 0.95, status, transform=ax.transAxes, ha='right', va='top',
            fontsize=16, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_compute_multiplier(results, save_path):
    """Plot compute multiplier results."""
    summary = results["compute_multiplier"]["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Entropy comparison
    categories = ['FEEL ON', 'FEEL OFF']
    values = [summary["feel_on_entropy"], summary["feel_off_entropy"]]
    colors = ['seagreen', 'coral']

    axes[0].bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel("Average Entropy")
    axes[0].set_title("Entropy: FEEL ON vs OFF")

    # Add reduction annotation
    reduction = summary["entropy_reduction_pct"]
    axes[0].annotate(f'{reduction:.1f}% reduction', xy=(0.5, min(values) * 0.95),
                     ha='center', fontsize=12, fontweight='bold', color='green')

    # Effective multiplier gauge
    ax = axes[1]
    multiplier = summary.get("effective_multiplier", summary.get("effective_compute_multiplier", 1.0))

    # Create gauge-like visualization
    theta = np.linspace(0, np.pi, 100)
    r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, 'k-', linewidth=2)

    # Fill based on multiplier (1.0 = 0%, 2.0 = 100%)
    fill_angle = np.pi * min((multiplier - 1.0), 1.0)
    theta_fill = np.linspace(0, fill_angle, 50)
    x_fill = np.concatenate([[0], r * np.cos(theta_fill), [0]])
    y_fill = np.concatenate([[0], r * np.sin(theta_fill), [0]])
    ax.fill(x_fill, y_fill, color='seagreen', alpha=0.6)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.3)
    ax.axis('off')
    ax.set_title(f"Effective Compute Multiplier: {multiplier:.2f}×")
    ax.text(0, 0.5, f"{multiplier:.2f}×", ha='center', va='center', fontsize=24, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_summary(results, save_path):
    """Plot overall summary."""
    verdicts = results["final_summary"]["verdicts"]

    fig, ax = plt.subplots(figsize=(10, 5))

    tests = list(verdicts.keys())
    passed = [1 if v else 0 for v in verdicts.values()]
    colors = ['seagreen' if p else 'coral' for p in passed]

    bars = ax.barh(tests, passed, color=colors, alpha=0.8, edgecolor='black')

    # Add PASS/FAIL labels
    for i, (bar, p) in enumerate(zip(bars, passed)):
        label = "PASS" if p else "FAIL"
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontweight='bold',
                color='green' if p else 'red')

    ax.set_xlim(0, 1.5)
    ax.set_xlabel("Result")
    ax.set_title(f"FEEL v5.0 Breakthrough Experiments\nScore: {sum(passed)}/{len(passed)}")

    # Remove x ticks
    ax.set_xticks([])

    # Add overall verdict
    overall = "BREAKTHROUGH" if results["final_summary"]["overall_pass"] else "MORE WORK NEEDED"
    color = "green" if results["final_summary"]["overall_pass"] else "red"
    ax.text(0.98, 0.05, overall, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=18, fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_training_history(save_path):
    """Plot training history."""
    try:
        with open("results/feel_training/canonical_v5_history.json") as f:
            history = json.load(f)
    except:
        print("No training history found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    epochs = range(1, len(history["aux_loss"]) + 1)

    # Aux loss
    axes[0, 0].plot(epochs, history["aux_loss"], 'b-', linewidth=2, marker='o')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Aux Loss")
    axes[0, 0].set_title("Auxiliary Task Loss (Lower = Better)")
    axes[0, 0].grid(True, alpha=0.3)

    # KL divergence
    axes[0, 1].plot(epochs, history["kl"], 'r-', linewidth=2, marker='s')
    axes[0, 1].axhline(y=0.01, color='green', linestyle='--', label='KL Budget (0.01)')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("KL Divergence")
    axes[0, 1].set_title("KL Divergence (Constraint)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Future prediction loss
    axes[1, 0].plot(epochs, history["future_loss"], 'g-', linewidth=2, marker='^')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Future Loss")
    axes[1, 0].set_title("Future Entropy Prediction Loss")
    axes[1, 0].grid(True, alpha=0.3)

    # Alpha and Lambda
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()

    l1 = ax1.plot(epochs, history["alpha"], 'b-', linewidth=2, marker='o', label='Alpha')
    l2 = ax2.plot(epochs, history["lambda"], 'r-', linewidth=2, marker='s', label='Lambda')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Alpha", color='blue')
    ax2.set_ylabel("Lambda", color='red')
    ax1.set_title("Alpha (FEEL strength) and Lambda (KL constraint)")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    """Generate all plots."""
    plot_dir = Path("results/feel_experiments/plots_v5")
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = load_results()

    print("\nGenerating v5.0 plots...")
    plot_teacher_forced(results, plot_dir / "teacher_forced_counterfactual.png")
    plot_falsification(results, plot_dir / "falsification_battery.png")
    plot_compute_multiplier(results, plot_dir / "compute_multiplier.png")
    plot_summary(results, plot_dir / "summary.png")
    plot_training_history(plot_dir / "training_history.png")

    print(f"\nAll plots saved to {plot_dir}")

if __name__ == "__main__":
    main()
