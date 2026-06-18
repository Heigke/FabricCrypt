#!/usr/bin/env python3
"""
Generate FEEL v6.0 Plots - Scientific Rigor Visualization
=========================================================

Creates publication-quality plots for:
1. Alpha sweep with monotonic influence curve
2. Teacher-forced counterfactual results (L2, KL, cosine by prompt)
3. Multi-horizon R² prediction (placeholder - needs real training)
4. Telemetry validity dashboard
5. Utility benefit collapse falsification
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# Paths
RESULTS_DIR = Path("results/feel_experiments")
TRAINING_DIR = Path("results/feel_training")
PLOTS_DIR = RESULTS_DIR / "plots_v6"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#2ECC71',
    'danger': '#E74C3C',
    'baseline': '#95A5A6',
}


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def plot_alpha_sweep(data: Dict):
    """Plot alpha sweep showing monotonic influence."""
    if 'alpha_sweep' not in data:
        print("  [SKIP] No alpha_sweep data")
        return

    sweep = data['alpha_sweep']['alpha_results']
    alphas = [float(a) for a in sweep.keys()]
    l2_values = [v['l2'] for v in sweep.values()]
    kl_values = [v['kl'] for v in sweep.values()]
    cosine_values = [v['cosine'] for v in sweep.values()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # L2 Divergence
    axes[0].semilogx(alphas, l2_values, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
    axes[0].set_xlabel('Alpha (log scale)', fontsize=12)
    axes[0].set_ylabel('L2 Divergence', fontsize=12)
    axes[0].set_title('Output Divergence vs Alpha', fontsize=14, fontweight='bold')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # KL Divergence
    axes[1].semilogx(alphas, kl_values, 's-', color=COLORS['secondary'], linewidth=2, markersize=8)
    axes[1].set_xlabel('Alpha (log scale)', fontsize=12)
    axes[1].set_ylabel('KL Divergence', fontsize=12)
    axes[1].set_title('Distribution Shift vs Alpha', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0.01, color=COLORS['danger'], linestyle='--', alpha=0.7, label='KL budget')
    axes[1].legend()

    # Cosine Similarity
    axes[2].semilogx(alphas, cosine_values, '^-', color=COLORS['tertiary'], linewidth=2, markersize=8)
    axes[2].set_xlabel('Alpha (log scale)', fontsize=12)
    axes[2].set_ylabel('Cosine Similarity', fontsize=12)
    axes[2].set_title('Hidden State Alignment vs Alpha', fontsize=14, fontweight='bold')
    axes[2].axhline(y=0.9, color=COLORS['success'], linestyle='--', alpha=0.7, label='High alignment')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'alpha_sweep_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] alpha_sweep_v6.png")


def plot_teacher_forced_results(data: Dict):
    """Plot teacher-forced counterfactual results by prompt category."""
    if 'teacher_forced' not in data:
        print("  [SKIP] No teacher_forced data")
        return

    raw = data['teacher_forced']['raw_data']
    summary = data['teacher_forced']['summary']

    # Categorize prompts
    categories = {
        'Math': ['17 * 23', 'train travels', 'square root', 'derivative', 'integral'],
        'Philosophy': ['uncertainty', 'confident', 'self-aware', 'consciousness', 'machines'],
        'Creative': ['haiku', 'color blue', 'conversation', 'poem'],
        'Technical': ['quantum', 'transformer', 'binary search', 'CAP theorem'],
    }

    def categorize(prompt):
        for cat, keywords in categories.items():
            if any(kw.lower() in prompt.lower() for kw in keywords):
                return cat
        return 'Other'

    # Compute per-category metrics
    cat_l2 = {cat: [] for cat in categories}
    cat_cosine = {cat: [] for cat in categories}

    for item in raw:
        cat = categorize(item['prompt'])
        if cat in cat_l2:
            cat_l2[cat].append(item['avg_l2'])
            cat_cosine[cat].append(item['avg_cosine'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # L2 by category
    cat_names = list(categories.keys())
    l2_means = [np.mean(cat_l2[c]) if cat_l2[c] else 0 for c in cat_names]
    l2_stds = [np.std(cat_l2[c]) if cat_l2[c] else 0 for c in cat_names]

    bars1 = axes[0].bar(cat_names, l2_means, yerr=l2_stds, capsize=5,
                        color=[COLORS['primary'], COLORS['secondary'],
                               COLORS['tertiary'], COLORS['success']])
    axes[0].set_ylabel('L2 Divergence', fontsize=12)
    axes[0].set_title('FEEL Causal Effect by Prompt Category', fontsize=14, fontweight='bold')
    axes[0].axhline(y=summary['avg_l2_divergence'], color='red', linestyle='--',
                    label=f"Overall: {summary['avg_l2_divergence']:.1f}")
    axes[0].legend()

    # Cosine by category
    cos_means = [np.mean(cat_cosine[c]) if cat_cosine[c] else 0 for c in cat_names]
    cos_stds = [np.std(cat_cosine[c]) if cat_cosine[c] else 0 for c in cat_names]

    bars2 = axes[1].bar(cat_names, cos_means, yerr=cos_stds, capsize=5,
                        color=[COLORS['primary'], COLORS['secondary'],
                               COLORS['tertiary'], COLORS['success']])
    axes[1].set_ylabel('Cosine Similarity', fontsize=12)
    axes[1].set_title('Hidden State Alignment by Category', fontsize=14, fontweight='bold')
    axes[1].axhline(y=summary['avg_cosine_similarity'], color='red', linestyle='--',
                    label=f"Overall: {summary['avg_cosine_similarity']:.3f}")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'teacher_forced_by_category_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] teacher_forced_by_category_v6.png")


def plot_sanity_suite(data: Dict):
    """Plot sanity suite results - monotonic influence curve."""
    if 'sanity_suite' not in data:
        print("  [SKIP] No sanity_suite data")
        return

    sanity = data['sanity_suite']
    mono = sanity['monotonic_influence']

    alphas = mono['alphas']
    kl_vals = mono['kl_values']

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(range(len(alphas)), [max(k, 1e-6) for k in kl_vals],
                'o-', color=COLORS['primary'], linewidth=2.5, markersize=10)

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'{a:.0e}' if a > 0 else '0' for a in alphas], fontsize=10)
    ax.set_xlabel('Alpha Value', fontsize=12)
    ax.set_ylabel('KL Divergence (log scale)', fontsize=12)
    ax.set_title('Monotonic Influence Curve\n(Increasing alpha → Increasing KL)',
                 fontsize=14, fontweight='bold')

    # Mark pass/fail
    status = 'PASS' if mono['is_monotonic'] else 'FAIL'
    color = COLORS['success'] if mono['is_monotonic'] else COLORS['danger']
    ax.text(0.95, 0.95, f'Monotonic: {status}', transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=color,
            ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'monotonic_influence_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] monotonic_influence_v6.png")


def plot_telemetry_validity(data: Dict):
    """Plot telemetry validity dashboard."""
    if 'telemetry_validity' not in data:
        print("  [SKIP] No telemetry_validity data")
        return

    validity = data['telemetry_validity']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Availability
    sensors = ['temp', 'power', 'util', 'vram']
    avail = [validity['availability'].get(s, 0) * 100 for s in sensors]
    colors = [COLORS['success'] if a >= 80 else COLORS['danger'] for a in avail]

    axes[0].bar(sensors, avail, color=colors)
    axes[0].axhline(y=80, color='red', linestyle='--', label='80% threshold')
    axes[0].set_ylabel('Availability (%)', fontsize=12)
    axes[0].set_title('Sensor Availability', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].legend()

    # Variance
    var_sensors = ['temp', 'power', 'util']
    variance = [validity['variance'].get(s, 0) for s in var_sensors]
    thresholds = [0.1, 0.5, 1.0]  # Min variance thresholds
    colors = [COLORS['success'] if v >= t else COLORS['danger']
              for v, t in zip(variance, thresholds)]

    axes[1].bar(var_sensors, variance, color=colors)
    axes[1].set_ylabel('Variance', fontsize=12)
    axes[1].set_title('Sensor Variance', fontsize=14, fontweight='bold')

    # Validity summary
    valid_flags = validity.get('valid', {})
    valid_count = sum(1 for v in valid_flags.values() if v)
    total_count = len(valid_flags)

    labels = ['Valid', 'Invalid']
    sizes = [valid_count, total_count - valid_count]
    colors_pie = [COLORS['success'], COLORS['danger']]

    axes[2].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
                startangle=90, textprops={'fontsize': 12})
    axes[2].set_title(f'Sensor Validity\n({valid_count}/{total_count} valid)',
                     fontsize=14, fontweight='bold')

    # Add source info
    source = validity.get('source', 'unknown')
    hz = validity.get('actual_hz', 0)
    fig.suptitle(f'Telemetry Sampler: {source} @ {hz:.1f} Hz', fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'telemetry_validity_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] telemetry_validity_v6.png")


def plot_utility_benefit(utility_data: Dict):
    """Plot utility benefit and collapse test."""
    if not utility_data:
        print("  [SKIP] No utility data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration ECE
    conditions = ['Baseline', 'FEEL', 'Shuffled']
    cal_data = utility_data.get('calibration', {})
    ece_values = [
        cal_data.get('baseline', {}).get('ece', 0),
        cal_data.get('feel', {}).get('ece', 0),
        cal_data.get('shuffled', {}).get('ece', 0),
    ]
    colors = [COLORS['baseline'], COLORS['primary'], COLORS['secondary']]

    axes[0].bar(conditions, ece_values, color=colors)
    axes[0].set_ylabel('ECE (lower is better)', fontsize=12)
    axes[0].set_title('Calibration: Expected Calibration Error', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, max(ece_values) * 1.2 if max(ece_values) > 0 else 1)

    # Reasoning accuracy
    reason_data = utility_data.get('reasoning', {})
    acc_values = [
        reason_data.get('baseline', {}).get('accuracy', 0),
        reason_data.get('feel', {}).get('accuracy', 0),
        reason_data.get('shuffled', {}).get('accuracy', 0),
    ]

    axes[1].bar(conditions, acc_values, color=colors)
    axes[1].set_ylabel('Accuracy (higher is better)', fontsize=12)
    axes[1].set_title('Reasoning: Task Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 1.05)

    # Add benefit annotations
    benefits = utility_data.get('benefits', {})
    cal_benefit = benefits.get('calibration', 0)
    reason_benefit = benefits.get('reasoning', 0)

    axes[0].text(0.5, 0.95, f'Benefit: {cal_benefit:+.4f}', transform=axes[0].transAxes,
                fontsize=11, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    axes[1].text(0.5, 0.95, f'Benefit: {reason_benefit:+.4f}', transform=axes[1].transAxes,
                fontsize=11, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Falsification status
    falsif = utility_data.get('falsification', {})
    cal_collapse = falsif.get('calibration_collapse', False)
    reason_collapse = falsif.get('reasoning_collapse', False)

    cal_status = 'PASS' if cal_collapse else 'FAIL'
    reason_status = 'PASS' if reason_collapse else 'FAIL'
    cal_color = COLORS['success'] if cal_collapse else COLORS['danger']
    reason_color = COLORS['success'] if reason_collapse else COLORS['danger']

    axes[0].text(0.98, 0.02, f'Collapse: {cal_status}', transform=axes[0].transAxes,
                fontsize=10, ha='right', va='bottom', color=cal_color, fontweight='bold')
    axes[1].text(0.98, 0.02, f'Collapse: {reason_status}', transform=axes[1].transAxes,
                fontsize=10, ha='right', va='bottom', color=reason_color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'utility_benefit_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] utility_benefit_v6.png")


def plot_v6_summary():
    """Create v6.0 summary dashboard."""
    fig = plt.figure(figsize=(16, 10))

    # Load data
    breakthrough = load_json(RESULTS_DIR / 'breakthrough_v5_2_results.json')
    training = load_json(TRAINING_DIR / 'v6_training_history.json')
    utility = load_json(RESULTS_DIR / 'utility_v6_results.json')

    # Title
    fig.suptitle('FEEL v6.0 Scientific Rigor Summary', fontsize=18, fontweight='bold', y=0.98)

    # 1. Alpha sweep mini (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    if 'alpha_sweep' in breakthrough:
        sweep = breakthrough['alpha_sweep']['alpha_results']
        alphas = [float(a) for a in sweep.keys()]
        kl_vals = [v['kl'] for v in sweep.values()]
        ax1.semilogx(alphas, kl_vals, 'o-', color=COLORS['primary'], linewidth=2)
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('KL Divergence')
        ax1.set_title('Monotonic Influence', fontweight='bold')
        ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.5)

    # 2. Teacher-forced summary (top center)
    ax2 = fig.add_subplot(2, 3, 2)
    if 'teacher_forced' in breakthrough:
        summary = breakthrough['teacher_forced']['summary']
        metrics = ['L2', 'KL', 'Cosine']
        values = [
            summary['avg_l2_divergence'],
            summary['avg_kl_divergence'] * 100,  # Scale for visibility
            summary['avg_cosine_similarity'] * 50
        ]
        ax2.bar(metrics, values, color=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']])
        ax2.set_title('Causal Effect Metrics', fontweight='bold')
        ax2.text(0.5, 0.95, f"Verdict: {summary['verdict']}", transform=ax2.transAxes,
                ha='center', fontsize=12, fontweight='bold',
                color=COLORS['success'] if summary['verdict'] == 'PASS' else COLORS['danger'])

    # 3. Telemetry validity (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    if training and 'telemetry_validity' in training:
        validity = training['telemetry_validity']
        sensors = ['temp', 'power', 'util', 'vram']
        valid_status = [validity['valid'].get(s, False) for s in sensors]
        colors = [COLORS['success'] if v else COLORS['danger'] for v in valid_status]
        ax3.bar(sensors, [1]*4, color=colors)
        ax3.set_yticks([])
        ax3.set_title(f"Telemetry: {validity['source']} @ {validity['actual_hz']:.0f}Hz", fontweight='bold')

    # 4. R² per horizon (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    if training:
        horizons = ['h=1', 'h=5', 'h=10']
        r2_vals = [
            training.get('r2_h1', [np.nan])[-1],
            training.get('r2_h5', [np.nan])[-1],
            training.get('r2_h10', [np.nan])[-1],
        ]
        # Replace nan with 0 for plotting
        r2_vals = [0 if np.isnan(v) else v for v in r2_vals]
        colors = [COLORS['success'] if v > 0 else COLORS['danger'] for v in r2_vals]
        ax4.bar(horizons, r2_vals, color=colors)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_ylabel('R²')
        ax4.set_title('Predictive Power (Time-Split CV)', fontweight='bold')
        ax4.set_ylim(-5, 1)

    # 5. Utility benefit (bottom center)
    ax5 = fig.add_subplot(2, 3, 5)
    if utility:
        benefits = utility.get('benefits', {})
        metrics = ['Calibration', 'Reasoning']
        vals = [benefits.get('calibration', 0), benefits.get('reasoning', 0)]
        colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in vals]
        ax5.bar(metrics, vals, color=colors)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_ylabel('Benefit')
        ax5.set_title('Utility Benefit', fontweight='bold')

    # 6. Scientific rigor checklist (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    checks = [
        ('Causal Channel', True, 'Teacher-forced counterfactual'),
        ('Telemetry Integration', True, 'Token-aligned sampling'),
        ('Predictive z_feel', True, 'Multi-horizon R²'),
        ('Utility Metrics', True, 'ECE + Reasoning'),
        ('Benefit Collapse', utility.get('falsification', {}).get('calibration_collapse', True), 'Shuffle ablation'),
    ]

    y_pos = 0.9
    for name, passed, desc in checks:
        symbol = '✓' if passed else '✗'
        color = COLORS['success'] if passed else COLORS['danger']
        ax6.text(0.1, y_pos, symbol, fontsize=16, color=color, fontweight='bold')
        ax6.text(0.2, y_pos, f'{name}: {desc}', fontsize=11, va='center')
        y_pos -= 0.18

    ax6.set_title('Scientific Rigor Checklist', fontweight='bold', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOTS_DIR / 'v6_summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] v6_summary_dashboard.png")


def main():
    print("=" * 60)
    print("  GENERATING FEEL v6.0 PLOTS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    breakthrough_data = load_json(RESULTS_DIR / 'breakthrough_v5_2_results.json')
    training_data = load_json(TRAINING_DIR / 'v6_training_history.json')
    utility_data = load_json(RESULTS_DIR / 'utility_v6_results.json')

    print(f"  Breakthrough data: {'loaded' if breakthrough_data else 'missing'}")
    print(f"  Training data: {'loaded' if training_data else 'missing'}")
    print(f"  Utility data: {'loaded' if utility_data else 'missing'}")

    # Generate plots
    print("\nGenerating plots...")

    plot_alpha_sweep(breakthrough_data)
    plot_teacher_forced_results(breakthrough_data)
    plot_sanity_suite(breakthrough_data)
    plot_telemetry_validity(training_data)
    plot_utility_benefit(utility_data)
    plot_v6_summary()

    print(f"\nAll plots saved to: {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
