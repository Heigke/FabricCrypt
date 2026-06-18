#!/usr/bin/env python3
"""
Plot Stress & Recovery Results

Visualizes:
1. Fatigue accumulation and recovery curves
2. Feeling state transitions
3. K regulation over time
4. Temperature dynamics
"""

import json
import argparse
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available")


def plot_stress_recovery(timeline_path: str, output_dir: str):
    """Create comprehensive stress/recovery visualization."""
    with open(timeline_path) as f:
        timeline = json.load(f)

    if not HAS_MATPLOTLIB:
        print("Cannot generate plots without matplotlib")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    generations = [e['generation'] for e in timeline]
    phases = [e['phase'] for e in timeline]
    fatigue = [e['fatigue'] for e in timeline]
    feelings = [e['feeling'] for e in timeline]
    ks = [e['k'] for e in timeline]
    temps = [e['temp'] for e in timeline]
    metabolic = [e['metabolic'] for e in timeline]
    thermal = [e['thermal'] for e in timeline]
    cognitive = [e['cognitive'] for e in timeline]

    # Find phase boundary
    stress_end = max(i for i, p in enumerate(phases) if p == 'stress') + 1

    # Color mappings
    feeling_colors = {
        'FOCUSED': '#3498db',
        'FLOW_STATE': '#2ecc71',
        'CURIOUS': '#9b59b6',
        'DETERMINATION': '#f39c12',
        'STRAINED': '#e74c3c',
        'EXHAUSTED': '#8b0000',
        'OVERHEATED': '#ff0000',
    }

    # Create figure with 5 panels
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # Panel 1: Fatigue Curve (the main story)
    ax1 = axes[0]
    ax1.fill_between(generations[:stress_end], fatigue[:stress_end],
                     alpha=0.3, color='red', label='Stress Phase')
    ax1.fill_between(generations[stress_end:], fatigue[stress_end:],
                     alpha=0.3, color='green', label='Recovery Phase')
    ax1.plot(generations, fatigue, 'k-', linewidth=2.5)
    ax1.axvline(x=stress_end, color='gray', linestyle='--', linewidth=2, label='Phase Boundary')
    ax1.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='High Fatigue')
    ax1.set_ylabel('Fatigue', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.set_title('Stress & Recovery: The Organism Breathes', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Somatic Signals
    ax2 = axes[1]
    ax2.plot(generations, metabolic, 'r-', linewidth=1.5, label='Metabolic', alpha=0.8)
    ax2.plot(generations, thermal, 'orange', linewidth=1.5, label='Thermal', alpha=0.8)
    ax2.plot(generations, cognitive, 'purple', linewidth=1.5, label='Cognitive', alpha=0.8)
    ax2.axvline(x=stress_end, color='gray', linestyle='--', linewidth=2)
    ax2.set_ylabel('Somatic Signal', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Feeling States (color-coded bars)
    ax3 = axes[2]
    for i, (g, f) in enumerate(zip(generations, feelings)):
        color = feeling_colors.get(f, 'gray')
        ax3.bar(g, 1, width=0.8, color=color, alpha=0.8)
    ax3.axvline(x=stress_end, color='gray', linestyle='--', linewidth=2)
    ax3.set_ylabel('Feeling', fontsize=12)
    ax3.set_yticks([])

    # Add legend for feelings
    unique_feelings = list(set(feelings))
    handles = [mpatches.Patch(color=feeling_colors.get(f, 'gray'), label=f) for f in unique_feelings]
    ax3.legend(handles=handles, loc='upper right', fontsize=9)

    # Panel 4: K Regulation
    ax4 = axes[3]
    ax4.step(generations, ks, where='mid', linewidth=2.5, color='brown')
    ax4.fill_between(generations, ks, step='mid', alpha=0.3, color='brown')
    ax4.axvline(x=stress_end, color='gray', linestyle='--', linewidth=2)
    ax4.axhline(y=2, color='red', linestyle=':', alpha=0.5, label='Survival Mode (K≤2)')
    ax4.set_ylabel('K (Compute)', fontsize=12)
    ax4.set_ylim(0, 5)
    ax4.set_yticks([1, 2, 3, 4])
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Temperature
    ax5 = axes[4]
    ax5.plot(generations, temps, 'b-', linewidth=2)
    ax5.fill_between(generations, temps, alpha=0.2, color='blue')
    ax5.axvline(x=stress_end, color='gray', linestyle='--', linewidth=2)
    ax5.set_ylabel('Temperature (°C)', fontsize=12)
    ax5.set_xlabel('Generation', fontsize=12)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "stress_recovery_organism.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()

    # Create second figure: Recovery Analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Fatigue vs Generation with exponential fit
    ax_left = axes2[0]
    recovery_gens = generations[stress_end:]
    recovery_fatigue = fatigue[stress_end:]

    ax_left.scatter(recovery_gens, recovery_fatigue, c='green', s=80, alpha=0.7, edgecolors='black')
    ax_left.plot(recovery_gens, recovery_fatigue, 'g--', linewidth=1.5, alpha=0.5)

    # Add recovery percentage annotations
    peak = max(fatigue)
    for i, (g, f) in enumerate(zip(recovery_gens, recovery_fatigue)):
        if i % 3 == 0:  # Every 3rd point
            pct = (1 - f/peak) * 100
            ax_left.annotate(f'{pct:.0f}%', (g, f), textcoords="offset points",
                           xytext=(5, 5), fontsize=8, alpha=0.7)

    ax_left.set_xlabel('Generation', fontsize=12)
    ax_left.set_ylabel('Fatigue', fontsize=12)
    ax_left.set_title('Recovery Curve: Biological Inertia', fontsize=14)
    ax_left.grid(True, alpha=0.3)

    # Right: Feeling distribution
    ax_right = axes2[1]
    feeling_counts = {}
    for f in feelings:
        feeling_counts[f] = feeling_counts.get(f, 0) + 1

    colors = [feeling_colors.get(f, 'gray') for f in feeling_counts.keys()]
    ax_right.bar(feeling_counts.keys(), feeling_counts.values(), color=colors, alpha=0.8, edgecolor='black')
    ax_right.set_ylabel('Count', fontsize=12)
    ax_right.set_title('Feeling State Distribution', fontsize=14)
    ax_right.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plot_path2 = output_dir / "stress_recovery_analysis.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path2}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Stress & Recovery Results")
    parser.add_argument("--input", required=True, help="Path to timeline JSON")
    parser.add_argument("--output-dir", default="results/stress_recovery/plots")
    args = parser.parse_args()

    plot_stress_recovery(args.input, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
