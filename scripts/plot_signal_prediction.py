#!/usr/bin/env python3
"""Plot signal prediction experiment results."""

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


def plot_lead_lag_correlations(results: dict, output_dir: Path):
    """Plot lead/lag correlation analysis."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    models = list(results.keys())
    temps = ["temp_0.0", "temp_0.7"]

    for row, model_file in enumerate(models):
        model_name = model_file.replace("signal_prediction_", "").replace(".json", "")
        data = results[model_file]

        for col, temp in enumerate(temps):
            ax = axes[row, col]

            if temp not in data:
                continue

            analysis = data[temp]["analysis"]
            corrs = analysis["correlations"]

            # Plot each signal type
            for signal_name, signal_corrs in corrs.items():
                lags = sorted(signal_corrs.keys(), key=int)
                values = [signal_corrs[str(lag)] for lag in lags]

                label = signal_name.replace("_vs_tpot", "").replace("_", " ")
                ax.plot(lags, values, 'o-', label=label, markersize=4)

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=0.2, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axhline(y=-0.2, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

            ax.set_xlabel('Lag (tokens)')
            ax.set_ylabel('Correlation (r)')
            ax.set_title(f'{model_name}\nT={temp.replace("temp_", "")}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.3, 0.3)

    plt.suptitle('Lead/Lag Correlation Analysis: Signal vs TPOT\n(Red dashed = significance threshold)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'signal_leadlag_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'signal_leadlag_correlation.png'}")


def plot_spike_prediction_auc(results: dict, output_dir: Path):
    """Plot spike prediction AUC results."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = list(results.keys())

    for ax_idx, model_file in enumerate(models):
        ax = axes[ax_idx]
        model_name = model_file.replace("signal_prediction_", "").replace(".json", "")
        data = results[model_file]

        lookaheads = [0, 1, 2, 4, 8]
        width = 0.25
        x = np.arange(len(lookaheads))

        signals = ["latent_delta_norm", "logit_margin", "entropy_approx"]
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        for temp_idx, temp in enumerate(["temp_0.0", "temp_0.7"]):
            if temp not in data:
                continue

            aucs = data[temp]["analysis"]["spike_aucs"]

            for sig_idx, signal in enumerate(signals):
                values = [aucs.get(f"{signal}_lookahead_{la}", 0.5) for la in lookaheads]
                offset = (temp_idx - 0.5) * width + (sig_idx - 1) * width * 0.3
                label = f"{signal.replace('_', ' ')} T={temp.replace('temp_', '')}"
                ax.bar(x + offset, values, width * 0.3, label=label,
                       color=colors[sig_idx], alpha=0.7 if temp_idx == 0 else 0.4)

        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random (AUC=0.5)')
        ax.axhline(y=0.6, color='green', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Lookahead (tokens)')
        ax.set_ylabel('AUC for Spike Prediction')
        ax.set_title(f'{model_name}\nPredicting Top-10% TPOT Spikes')
        ax.set_xticks(x)
        ax.set_xticklabels(lookaheads)
        ax.set_ylim(0.3, 0.8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Spike Prediction AUC: Can Latent Signals Predict TPOT Spikes?\n(AUC > 0.6 = meaningful, 0.5 = random)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_dir / 'signal_spike_prediction_auc.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'signal_spike_prediction_auc.png'}")


def plot_key_finding_summary(results: dict, output_dir: Path):
    """Create a summary plot of the key finding: no prediction power."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all AUC values
    all_aucs = []
    labels = []

    for model_file, model_data in results.items():
        model_name = model_file.replace("signal_prediction_", "").replace(".json", "")

        for temp, temp_data in model_data.items():
            if not temp.startswith("temp_"):
                continue

            aucs = temp_data["analysis"]["spike_aucs"]

            for key, value in aucs.items():
                if "latent" in key and "lookahead_1" in key:
                    labels.append(f"{model_name}\n{temp.replace('temp_', 'T=')}")
                    all_aucs.append(value)

    x = range(len(all_aucs))
    colors = ['#e74c3c' if v < 0.55 else '#2ecc71' for v in all_aucs]

    bars = ax.bar(x, all_aucs, color=colors, edgecolor='black')

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random Chance')
    ax.axhline(y=0.6, color='green', linestyle='--', linewidth=1, label='Meaningful Prediction')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('AUC for Predicting TPOT Spikes', fontsize=12)
    ax.set_ylim(0.3, 0.8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, all_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title('KEY FINDING: Latent Delta Does NOT Predict TPOT Spikes\n'
              '(All AUC values ≈ 0.5 = random chance)',
              fontsize=14, fontweight='bold', color='red')
    plt.tight_layout()

    plt.savefig(output_dir / 'signal_prediction_key_finding.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'signal_prediction_key_finding.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot signal prediction results")
    parser.add_argument("--results-dir", type=Path, default=Path("results/signal_prediction"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/signal_prediction"))
    args = parser.parse_args()

    # Load all results
    results = {}
    for result_file in args.results_dir.glob("signal_prediction_*.json"):
        with open(result_file) as f:
            results[result_file.name] = json.load(f)

    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Found {len(results)} result files")

    plot_lead_lag_correlations(results, args.output_dir)
    plot_spike_prediction_auc(results, args.output_dir)
    plot_key_finding_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
