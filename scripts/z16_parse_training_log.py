#!/usr/bin/env python3
"""
Parse training log and extract metrics for plotting.
Works with both running and completed training logs.
"""

import re
import json
import argparse
from pathlib import Path

def parse_training_log(log_path: str, output_path: str = None):
    """Extract loss and learning rate from training log."""

    losses = []
    learning_rates = []
    steps = []

    # Pattern to match: loss=6.5189, lr=1.00e-04
    pattern = r'loss=([0-9.]+), lr=([0-9.e+-]+)'

    with open(log_path, 'r') as f:
        content = f.read()

    # Find all matches
    matches = re.findall(pattern, content)

    # Deduplicate (tqdm updates same line multiple times)
    seen = set()
    step = 0
    for loss, lr in matches:
        key = (loss, lr)
        if key not in seen:
            seen.add(key)
            step += 1
            losses.append(float(loss))
            learning_rates.append(float(lr))
            steps.append(step)

    metrics = {
        "train_losses": losses,
        "learning_rates": learning_rates,
        "steps": steps,
        "total_steps": len(losses),
    }

    # Summary stats
    if losses:
        metrics["min_loss"] = min(losses)
        metrics["max_loss"] = max(losses)
        metrics["final_loss"] = losses[-1]
        metrics["loss_reduction"] = losses[0] - losses[-1] if len(losses) > 1 else 0

    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")

    return metrics


def print_summary(metrics: dict):
    """Print training summary."""
    print("\n" + "=" * 50)
    print("TRAINING METRICS SUMMARY")
    print("=" * 50)
    print(f"Total steps:     {metrics.get('total_steps', 0)}")
    print(f"Initial loss:    {metrics['train_losses'][0]:.4f}" if metrics.get('train_losses') else "")
    print(f"Final loss:      {metrics.get('final_loss', 'N/A'):.4f}" if metrics.get('final_loss') else "")
    print(f"Min loss:        {metrics.get('min_loss', 'N/A'):.4f}" if metrics.get('min_loss') else "")
    print(f"Loss reduction:  {metrics.get('loss_reduction', 0):.4f}")
    print("=" * 50)


def plot_metrics(metrics: dict, save_path: str = None):
    """Plot training metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1 = axes[0]
    ax1.plot(metrics["steps"], metrics["train_losses"], 'b-', alpha=0.7, linewidth=0.5)
    # Add smoothed line
    if len(metrics["train_losses"]) > 10:
        window = min(50, len(metrics["train_losses"]) // 10)
        smoothed = []
        for i in range(len(metrics["train_losses"])):
            start = max(0, i - window)
            smoothed.append(sum(metrics["train_losses"][start:i+1]) / (i - start + 1))
        ax1.plot(metrics["steps"], smoothed, 'r-', linewidth=2, label='Smoothed')
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Learning rate
    ax2 = axes[1]
    ax2.plot(metrics["steps"], metrics["learning_rates"], 'g-', linewidth=1)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse training log for metrics")
    parser.add_argument("--log", type=str, default="results/z16_hardware_training.log")
    parser.add_argument("--output", type=str, default="results/z16_training_metrics.json")
    parser.add_argument("--plot", type=str, default=None, help="Save plot to this path")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    metrics = parse_training_log(args.log, args.output)
    print_summary(metrics)

    if args.plot or args.show:
        plot_metrics(metrics, args.plot)
