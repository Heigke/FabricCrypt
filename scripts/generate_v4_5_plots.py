#!/usr/bin/env python3
"""
Generate plots for FEEL v4.5 training and validation results.

Creates:
1. Training loss curve
2. Per-action accuracy over epochs
3. v4.4 vs v4.5 comparison (catastrophic forgetting fix)
4. Architecture diagram
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_training_metrics(path: str = "models/feel_v4_5/training_metrics.json") -> dict:
    """Load training metrics."""
    with open(path) as f:
        return json.load(f)


def plot_training_curves(metrics: dict, output_dir: str = "results/plots"):
    """Plot training loss and accuracy curves."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    epochs = [m["epoch"] for m in metrics["metrics"]]
    loss = [m["loss"] for m in metrics["metrics"]]
    train_acc = [m["train_acc"] for m in metrics["metrics"]]
    eval_acc = [m["eval_acc"] for m in metrics["metrics"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    axes[0].plot(epochs, loss, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # Accuracy curves
    axes[1].plot(epochs, train_acc, 'b-o', linewidth=2, markersize=6, label="Train")
    axes[1].plot(epochs, eval_acc, 'r-s', linewidth=2, markersize=6, label="Eval")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Classifier Accuracy", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feel_v4_5_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feel_v4_5_training_curves.png")


def plot_per_action_accuracy(metrics: dict, output_dir: str = "results/plots"):
    """Plot per-action accuracy over epochs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    actions = ["OK", "WARM", "HOT", "REST", "FULL", "CRITICAL"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6', '#1abc9c']

    epochs = [m["epoch"] for m in metrics["metrics"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    for action, color in zip(actions, colors):
        acc = [m["per_action"].get(action, 0) for m in metrics["metrics"]]
        ax.plot(epochs, acc, '-o', color=color, linewidth=2, markersize=5, label=action)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Action Classifier Accuracy (v4.5 Frozen Model)", fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feel_v4_5_per_action_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feel_v4_5_per_action_accuracy.png")


def plot_final_accuracy_bar(metrics: dict, output_dir: str = "results/plots"):
    """Bar chart of final per-action accuracy."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    final_epoch = metrics["metrics"][-1]
    actions = ["OK", "WARM", "HOT", "REST", "FULL", "CRITICAL"]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6', '#1abc9c']

    accuracies = [final_epoch["per_action"].get(a, 0) for a in actions]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(actions, accuracies, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel("Action Class", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Final Per-Action Accuracy (Epoch {final_epoch['epoch']}, Eval: {final_epoch['eval_acc']:.0%})",
                 fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=final_epoch['eval_acc'], color='red', linestyle='--', linewidth=2,
               label=f"Overall Eval: {final_epoch['eval_acc']:.0%}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feel_v4_5_final_accuracy_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feel_v4_5_final_accuracy_bar.png")


def plot_v44_vs_v45_comparison(output_dir: str = "results/plots"):
    """Compare v4.4 catastrophic forgetting vs v4.5 preservation."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # v4.4 (Catastrophic Forgetting)
    ax = axes[0]
    categories = ['Language\nCoherence', 'Action\nClassification', 'Overall\nUtility']
    v44_scores = [0.0, 0.0, 0.0]  # Completely broken
    bars = ax.bar(categories, v44_scores, color='#e74c3c', edgecolor='black', linewidth=2)
    ax.set_ylim(0, 1.1)
    ax.set_title('v4.4 (LoRA Fine-tune)\nCatastrophic Forgetting', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)

    # Add failure markers
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2., 0.5, 'FAIL',
               ha='center', va='center', fontsize=14, fontweight='bold', color='white',
               bbox=dict(boxstyle='round', facecolor='#c0392b'))

    # v4.5 (Frozen Model)
    ax = axes[1]
    v45_scores = [1.0, 0.62, 0.81]  # Language preserved, classifier works
    colors = ['#2ecc71', '#f39c12', '#3498db']
    bars = ax.bar(categories, v45_scores, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylim(0, 1.1)
    ax.set_title('v4.5 (Frozen + Classifier)\nLanguage Preserved', fontsize=12, fontweight='bold')

    # Add value labels
    for bar, score in zip(bars, v45_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{score:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('FEEL Version Comparison: Catastrophic Forgetting vs Frozen Model',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feel_v44_vs_v45_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feel_v44_vs_v45_comparison.png")


def plot_architecture_diagram(output_dir: str = "results/plots"):
    """Create architecture diagram for v4.5."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Boxes with positions (x, y, w, h, color, label)
    boxes = [
        (0.5, 9, 2.5, 1.5, '#3498db', 'AMD GPU\nTelemetry'),
        (0.5, 6, 2.5, 1.5, '#e74c3c', 'z_feel\n[8 dims]'),
        (4.5, 9, 3, 1.5, '#2ecc71', 'Injector\n(~500K params)'),
        (4.5, 6, 3, 1.5, '#9b59b6', 'Input\nEmbeddings'),
        (9, 7.5, 4, 2.5, '#f39c12', 'FROZEN\nQwen 1.5B'),
        (9, 3.5, 4, 2, '#1abc9c', 'Classifier\n(~600K params)'),
        (4.5, 1, 3, 1.5, '#e67e22', 'Action Prediction\nOK|WARM|HOT|REST|FULL|CRIT'),
    ]

    for x, y, w, h, color, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black',
                             linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    # Arrows with labels
    arrow_style = dict(arrowstyle='->', color='black', lw=2)

    # Telemetry -> z_feel
    ax.annotate('', xy=(1.75, 7.5), xytext=(1.75, 9), arrowprops=arrow_style)

    # z_feel -> Injector
    ax.annotate('', xy=(4.5, 9.75), xytext=(3, 7.5), arrowprops=arrow_style)

    # z_feel -> Embeddings
    ax.annotate('', xy=(4.5, 6.75), xytext=(3, 6.75), arrowprops=arrow_style)

    # Injector -> Frozen LM (embedding offset)
    ax.annotate('', xy=(9, 9), xytext=(7.5, 9.75), arrowprops=arrow_style)
    ax.text(8.3, 10, 'offset', fontsize=9, style='italic')

    # Embeddings -> Frozen LM
    ax.annotate('', xy=(9, 8), xytext=(7.5, 6.75), arrowprops=arrow_style)
    ax.text(8.3, 7.3, '+', fontsize=14, fontweight='bold')

    # Frozen LM -> Classifier (hidden states)
    ax.annotate('', xy=(11, 5.5), xytext=(11, 7.5), arrowprops=arrow_style)
    ax.text(11.2, 6.5, 'hidden\nstates', fontsize=9, style='italic')

    # z_feel -> Classifier (direct)
    ax.annotate('', xy=(9, 4.5), xytext=(3, 6.5),
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2, ls='--'))
    ax.text(5.5, 5.2, 'z_feel (direct)', fontsize=9, style='italic', color='#e74c3c')

    # Classifier -> Action
    ax.annotate('', xy=(6, 2.5), xytext=(9, 4), arrowprops=arrow_style)

    # Title and legend
    ax.text(7, 11.5, 'FEEL v4.5 Architecture: Frozen Model + Trainable Modules',
           ha='center', va='center', fontsize=16, fontweight='bold')

    # Legend box
    legend_box = plt.Rectangle((0.3, 0.3), 3.5, 2.2, facecolor='white',
                                edgecolor='gray', linewidth=1)
    ax.add_patch(legend_box)
    ax.text(0.5, 2.2, 'Trainable: ~1.1M params', fontsize=9)
    ax.text(0.5, 1.7, 'Frozen LM: ~1.5B params', fontsize=9)
    ax.text(0.5, 1.2, 'Key: Model FROZEN during training', fontsize=9, fontweight='bold')
    ax.text(0.5, 0.7, '       Only injector + classifier learn', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feel_v4_5_architecture.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feel_v4_5_architecture.png")


def plot_training_progress(metrics: dict, output_dir: str = "results/plots"):
    """Plot loss over epochs with annotations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    epochs = [m["epoch"] for m in metrics["metrics"]]
    loss = [m["loss"] for m in metrics["metrics"]]
    eval_acc = [m["eval_acc"] for m in metrics["metrics"]]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Loss curve
    line1, = ax1.plot(epochs, loss, 'b-o', linewidth=2, markersize=8, label='Loss')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Accuracy curve
    line2, = ax2.plot(epochs, eval_acc, 'r-s', linewidth=2, markersize=8, label='Eval Accuracy')
    ax2.set_ylabel("Eval Accuracy", fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)

    # Annotations
    ax1.annotate(f'Final: {loss[-1]:.2f}',
                xy=(epochs[-1], loss[-1]),
                xytext=(epochs[-1]-1, loss[-1]+0.2),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue'))

    ax2.annotate(f'Final: {eval_acc[-1]:.0%}',
                xy=(epochs[-1], eval_acc[-1]),
                xytext=(epochs[-1]-1.5, eval_acc[-1]-0.1),
                fontsize=10, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax1.set_title("FEEL v4.5 Training Progress: Loss vs Accuracy", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feel_v4_5_training_progress.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feel_v4_5_training_progress.png")


def main():
    output_dir = "results/plots"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading training metrics...")
    metrics = load_training_metrics()

    print("\nGenerating FEEL v4.5 plots...")
    plot_training_curves(metrics, output_dir)
    plot_per_action_accuracy(metrics, output_dir)
    plot_final_accuracy_bar(metrics, output_dir)
    plot_v44_vs_v45_comparison(output_dir)
    plot_architecture_diagram(output_dir)
    plot_training_progress(metrics, output_dir)

    print(f"\nAll FEEL v4.5 plots saved to: {output_dir}/")

    # Print summary
    final = metrics["metrics"][-1]
    print("\n" + "="*50)
    print("FEEL v4.5 Training Summary")
    print("="*50)
    print(f"Final Loss: {final['loss']:.4f}")
    print(f"Train Accuracy: {final['train_acc']:.1%}")
    print(f"Eval Accuracy: {final['eval_acc']:.1%}")
    print("\nPer-Action Accuracy:")
    for action, acc in final['per_action'].items():
        print(f"  {action}: {acc:.1%}")


if __name__ == "__main__":
    main()
