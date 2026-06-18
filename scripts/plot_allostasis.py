#!/usr/bin/env python3
"""
Plotting for Allostasis Validation Results

Creates visualizations for the three triangulation tests:
- Test A: Causality (Hardware vs Chemistry correlation)
- Test B: Mediation (Intensity vs Expression characteristics)
- Test C: Allostasis (Mode transitions and temperature oscillation)
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
    print("matplotlib not available, will generate text summaries only")


def plot_causality(timeline_path: str, output_path: str):
    """Plot Test A: Hardware vs Chemistry correlation."""
    with open(timeline_path) as f:
        timeline = json.load(f)

    timestamps = [e['timestamp'] for e in timeline]
    pwr_1 = [e['pwr_1'] for e in timeline]
    stress = [e['stress_intensity'] for e in timeline]
    phases = [e['phase'] for e in timeline]

    if not HAS_MATPLOTLIB:
        print(f"\n=== CAUSALITY PLOT (Text Summary) ===")
        print(f"Samples: {len(timeline)}")
        print(f"pwr_1 range: {min(pwr_1):.1f} - {max(pwr_1):.1f}")
        print(f"stress range: {min(stress):.3f} - {max(stress):.3f}")
        correlation = np.corrcoef(pwr_1, stress)[0, 1]
        print(f"Correlation: {correlation:.3f}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Hardware (pwr_1)
    ax1 = axes[0]
    colors = ['red' if p == 'stress' else 'blue' for p in phases]
    ax1.scatter(timestamps, pwr_1, c=colors, alpha=0.6, s=10)
    ax1.set_ylabel('pwr_1 (raw)', fontsize=12)
    ax1.set_title('Test A: Causality Check - Hardware Drives Chemistry', fontsize=14)
    ax1.axhline(y=np.mean(pwr_1), color='gray', linestyle='--', alpha=0.5)
    stress_patch = mpatches.Patch(color='red', label='Stress phase')
    cool_patch = mpatches.Patch(color='blue', label='Cool phase')
    ax1.legend(handles=[stress_patch, cool_patch], loc='upper right')

    # Panel 2: Chemistry (stress intensity)
    ax2 = axes[1]
    ax2.plot(timestamps, stress, 'g-', linewidth=1.5)
    ax2.fill_between(timestamps, stress, alpha=0.3, color='green')
    ax2.set_ylabel('Stress Intensity', fontsize=12)
    ax2.set_ylim(0, 1)

    # Panel 3: Scatter correlation
    ax3 = axes[2]
    ax3.scatter(pwr_1, stress, c=colors, alpha=0.5, s=15)
    # Fit line
    z = np.polyfit(pwr_1, stress, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(pwr_1), max(pwr_1), 100)
    ax3.plot(x_line, p(x_line), 'k--', linewidth=2, label=f'r = {np.corrcoef(pwr_1, stress)[0,1]:.3f}')
    ax3.set_xlabel('pwr_1 (Hardware)', fontsize=12)
    ax3.set_ylabel('Stress (Chemistry)', fontsize=12)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_mediation(samples_path: str, output_path: str):
    """Plot Test B: Intensity vs Expression characteristics."""
    with open(samples_path) as f:
        samples = json.load(f)

    intensities = [s['intensity'] for s in samples]
    lengths = [s['length'] for s in samples]
    sentiments = [s['sentiment_score'] for s in samples]
    urgency = [s['urgency_count'] for s in samples]

    if not HAS_MATPLOTLIB:
        print(f"\n=== MEDIATION PLOT (Text Summary) ===")
        print(f"Samples: {len(samples)}")
        print(f"Intensity range: {min(intensities):.2f} - {max(intensities):.2f}")
        print(f"Sentiment range: {min(sentiments)} - {max(sentiments)}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Intensity vs Length
    ax1 = axes[0, 0]
    ax1.scatter(intensities, lengths, c=intensities, cmap='RdYlGn_r', s=50, alpha=0.7)
    ax1.set_xlabel('Injection Intensity')
    ax1.set_ylabel('Response Length (words)')
    ax1.set_title('Response Length vs Intensity')

    # Panel 2: Intensity vs Sentiment
    ax2 = axes[0, 1]
    ax2.scatter(intensities, sentiments, c=intensities, cmap='RdYlGn_r', s=50, alpha=0.7)
    ax2.set_xlabel('Injection Intensity')
    ax2.set_ylabel('Sentiment Score (urgency - calm)')
    ax2.set_title('Sentiment vs Intensity')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Panel 3: Intensity vs Urgency keywords
    ax3 = axes[1, 0]
    ax3.scatter(intensities, urgency, c=intensities, cmap='RdYlGn_r', s=50, alpha=0.7)
    ax3.set_xlabel('Injection Intensity')
    ax3.set_ylabel('Urgency Keywords Count')
    ax3.set_title('Urgency Words vs Intensity')

    # Panel 4: Summary by intensity bin
    ax4 = axes[1, 1]
    bins = [0.0, 0.3, 0.7, 1.0]
    labels = ['Low', 'Medium', 'High']
    binned_sentiment = []
    for i in range(len(bins) - 1):
        bin_samples = [s['sentiment_score'] for s in samples
                       if bins[i] <= s['intensity'] < bins[i+1]]
        binned_sentiment.append(np.mean(bin_samples) if bin_samples else 0)

    colors = ['green', 'yellow', 'red']
    ax4.bar(labels, binned_sentiment, color=colors, alpha=0.7)
    ax4.set_ylabel('Avg Sentiment Score')
    ax4.set_title('Sentiment by Intensity Level')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Test B: Mediation Check - Chemistry Drives Expression', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_allostasis(timeline_path: str, output_path: str):
    """Plot Test C: Mode transitions and temperature oscillation."""
    with open(timeline_path) as f:
        timeline = json.load(f)

    timestamps = [e['timestamp'] for e in timeline]
    temps = [e['temp'] for e in timeline]
    stress = [e['stress_intensity'] for e in timeline]
    modes = [e['mode'] for e in timeline]
    ks = [e['k'] for e in timeline]

    # Check if v9.0 somatic data exists
    has_somatic = 'fatigue' in timeline[0] if timeline else False

    if not HAS_MATPLOTLIB:
        print(f"\n=== ALLOSTASIS PLOT (Text Summary) ===")
        print(f"Samples: {len(timeline)}")
        print(f"Temperature range: {min(temps):.1f} - {max(temps):.1f}")
        transitions = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])
        print(f"Mode transitions: {transitions}")
        if has_somatic:
            fatigue = [e['fatigue'] for e in timeline]
            print(f"v9.0 Somatic - Fatigue range: {min(fatigue):.2f} - {max(fatigue):.2f}")
        return

    # Determine plot layout based on data availability
    if has_somatic:
        fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
        fatigue = [e.get('fatigue', 0) for e in timeline]
        metabolic = [e.get('metabolic', 0) for e in timeline]
        thermal = [e.get('thermal', 0) for e in timeline]
        cognitive = [e.get('cognitive', 0) for e in timeline]
        feelings = [e.get('feeling', 'UNKNOWN') for e in timeline]
    else:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Color mapping for modes
    mode_colors = ['green' if m == 'AMBITION' else 'red' for m in modes]

    # Panel 1: Temperature
    ax1 = axes[0]
    ax1.plot(timestamps, temps, 'b-', linewidth=2)
    ax1.fill_between(timestamps, temps, alpha=0.3, color='blue')
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    version_str = "v9.0 Somatic Nervous System" if has_somatic else "v8.x Binary Hysteresis"
    ax1.set_title(f'Test C: Allostasis Check - {version_str}', fontsize=14)

    # Panel 2: Stress/Somatic Intensity
    ax2 = axes[1]
    if has_somatic:
        # Plot multi-dimensional somatic signals
        ax2.plot(timestamps, metabolic, 'r-', linewidth=1.5, label='Metabolic', alpha=0.8)
        ax2.plot(timestamps, thermal, 'orange', linewidth=1.5, label='Thermal', alpha=0.8)
        ax2.plot(timestamps, cognitive, 'purple', linewidth=1.5, label='Cognitive', alpha=0.8)
        ax2.plot(timestamps, stress, 'k--', linewidth=1, label='Overall Stress', alpha=0.6)
        ax2.set_ylabel('Somatic Signals', fontsize=12)
        ax2.legend(loc='upper right', fontsize=9)
    else:
        ax2.fill_between(timestamps, stress, alpha=0.4, color='orange')
        ax2.plot(timestamps, stress, 'r-', linewidth=1.5)
        ax2.set_ylabel('Stress Intensity', fontsize=12)
        ax2.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='High threshold')
        ax2.axhline(y=0.25, color='green', linestyle='--', alpha=0.5, label='Low threshold')
        ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)

    # Panel 3: Mode/Feeling
    ax3 = axes[2]
    if has_somatic:
        # Color-coded feelings
        feeling_colors = {
            'FOCUSED': 'blue', 'FLOW_STATE': 'green', 'CURIOUS': 'cyan',
            'DETERMINATION': 'orange', 'STRAINED': 'red', 'EXHAUSTED': 'darkred',
            'OVERHEATED': 'maroon', 'UNKNOWN': 'gray'
        }
        for i, (t, f) in enumerate(zip(timestamps, feelings)):
            color = feeling_colors.get(f, 'gray')
            ax3.bar(t, 1, width=timestamps[1]-timestamps[0] if len(timestamps) > 1 else 1,
                   color=color, alpha=0.7)
        ax3.set_ylabel('Feeling', fontsize=12)
        ax3.set_yticks([])
        # Add legend
        unique_feelings = list(set(feelings))
        handles = [mpatches.Patch(color=feeling_colors.get(f, 'gray'), label=f) for f in unique_feelings]
        ax3.legend(handles=handles, loc='upper right', fontsize=9)
    else:
        mode_values = [1 if m == 'AMBITION' else 0 for m in modes]
        ax3.fill_between(timestamps, mode_values, step='mid', alpha=0.5, color='purple')
        ax3.step(timestamps, mode_values, where='mid', linewidth=2, color='purple')
        ax3.set_ylabel('Mode', fontsize=12)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['SURVIVAL', 'AMBITION'])
    ax3.set_ylim(-0.1, 1.1)

    # Panel 4: K value
    ax4 = axes[3]
    ax4.step(timestamps, ks, where='mid', linewidth=2, color='brown')
    ax4.fill_between(timestamps, ks, step='mid', alpha=0.3, color='brown')
    ax4.set_ylabel('K (Compute Budget)', fontsize=12)
    ax4.set_ylim(0, max(ks) + 1)
    if not has_somatic:
        ax4.set_xlabel('Time (seconds)', fontsize=12)

    # Panel 5: Fatigue (v9.0 only)
    if has_somatic:
        ax5 = axes[4]
        ax5.fill_between(timestamps, fatigue, alpha=0.4, color='purple')
        ax5.plot(timestamps, fatigue, 'purple', linewidth=2, label='Fatigue')
        ax5.set_ylabel('Fatigue', fontsize=12)
        ax5.set_xlabel('Time (seconds)', fontsize=12)
        ax5.set_ylim(0, max(0.5, max(fatigue) * 1.1))
        ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Recovery needed')
        ax5.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_timeline(
    causality_path: str,
    allostasis_path: str,
    output_path: str
):
    """Create combined timeline showing all signals together."""
    if not HAS_MATPLOTLIB:
        print("Combined timeline requires matplotlib")
        return

    # Load data
    with open(causality_path) as f:
        causality = json.load(f)
    with open(allostasis_path) as f:
        allostasis = json.load(f)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Panel 1: Causality test timeline
    ax1 = axes[0]
    timestamps = [e['timestamp'] for e in causality]
    pwr_1 = [e['pwr_1'] for e in causality]
    stress = [e['stress_intensity'] * 1000 for e in causality]  # Scale for visibility

    ax1.plot(timestamps, pwr_1, 'b-', label='pwr_1 (Hardware)', alpha=0.7)
    ax1.plot(timestamps, stress, 'r-', label='Stress x1000 (Chemistry)', alpha=0.7)
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title('Causality: Hardware → Chemistry Coupling')
    ax1.legend()

    # Panel 2: Allostasis test timeline
    ax2 = axes[1]
    timestamps = [e['timestamp'] for e in allostasis]
    temps = [e['temp'] for e in allostasis]
    modes = [e['mode'] for e in allostasis]
    stress = [e['stress_intensity'] * 50 for e in allostasis]  # Scale

    ax2.plot(timestamps, temps, 'b-', label='Temperature (°C)', linewidth=2)
    ax2.plot(timestamps, stress, 'r-', label='Stress x50', alpha=0.7)

    # Shade SURVIVAL periods
    in_survival = False
    survival_start = 0
    for i, (t, m) in enumerate(zip(timestamps, modes)):
        if m == 'SURVIVAL' and not in_survival:
            survival_start = t
            in_survival = True
        elif m == 'AMBITION' and in_survival:
            ax2.axvspan(survival_start, t, alpha=0.2, color='red', label='SURVIVAL' if i < 3 else '')
            in_survival = False

    ax2.set_ylabel('Value')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Allostasis: Action → Body Loop')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Allostasis Validation Results")
    parser.add_argument("--input_dir", default="results/allostasis_validation")
    parser.add_argument("--output_dir", default="results/allostasis_validation/plots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPlotting Allostasis Validation Results")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Plot each test if data exists
    causality_path = input_dir / "causality_timeline.json"
    if causality_path.exists():
        plot_causality(str(causality_path), str(output_dir / "causality.png"))

    mediation_path = input_dir / "mediation_samples.json"
    if mediation_path.exists():
        plot_mediation(str(mediation_path), str(output_dir / "mediation.png"))

    allostasis_path = input_dir / "allostasis_timeline.json"
    if allostasis_path.exists():
        plot_allostasis(str(allostasis_path), str(output_dir / "allostasis.png"))

    # Combined timeline if both exist
    if causality_path.exists() and allostasis_path.exists():
        plot_combined_timeline(
            str(causality_path),
            str(allostasis_path),
            str(output_dir / "combined_timeline.png")
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
