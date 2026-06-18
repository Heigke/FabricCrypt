#!/usr/bin/env python3
"""Generate comparison plot for latent controller vs auto."""

from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available, skipping plot generation")


def main():
    if not HAS_MPL:
        return

    # Data from experiments (Qwen2-0.5B, p64/d64, 3 reps each)
    policies = ['auto', 'latent']

    # Results from the runs
    time_s = [1.667, 1.861]  # mean total_s
    energy_j = [100.4, 118.5]  # mean total_energy_j
    tok_per_j = [1.19, 1.00]  # mean tok/J
    tpot_p95_ms = [26.0, 29.2]  # mean TPOT p95 in ms

    colors = {'auto': '#2ecc71', 'latent': '#9b59b6'}

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    # 1. Time comparison
    ax = axes[0]
    bars = ax.bar(policies, time_s, color=[colors[p] for p in policies], edgecolor='black')
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_title('Inference Time', fontsize=12)
    ax.set_ylim(0, max(time_s) * 1.2)
    for bar, val in zip(bars, time_s):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}s',
                ha='center', va='bottom', fontsize=10)
    # Add overhead annotation
    overhead = (time_s[1] - time_s[0]) / time_s[0] * 100
    ax.annotate(f'+{overhead:.0f}%', xy=(1, time_s[1]), xytext=(1.3, time_s[1]*0.9),
                fontsize=10, color='red', arrowprops=dict(arrowstyle='->', color='red'))

    # 2. Energy comparison
    ax = axes[1]
    bars = ax.bar(policies, energy_j, color=[colors[p] for p in policies], edgecolor='black')
    ax.set_ylabel('Energy (J)', fontsize=11)
    ax.set_title('Total Energy', fontsize=12)
    ax.set_ylim(0, max(energy_j) * 1.2)
    for bar, val in zip(bars, energy_j):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}J',
                ha='center', va='bottom', fontsize=10)
    overhead = (energy_j[1] - energy_j[0]) / energy_j[0] * 100
    ax.annotate(f'+{overhead:.0f}%', xy=(1, energy_j[1]), xytext=(1.3, energy_j[1]*0.9),
                fontsize=10, color='red', arrowprops=dict(arrowstyle='->', color='red'))

    # 3. Efficiency comparison
    ax = axes[2]
    bars = ax.bar(policies, tok_per_j, color=[colors[p] for p in policies], edgecolor='black')
    ax.set_ylabel('Efficiency (tok/J)', fontsize=11)
    ax.set_title('Token Efficiency', fontsize=12)
    ax.set_ylim(0, max(tok_per_j) * 1.3)
    for bar, val in zip(bars, tok_per_j):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03, f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)
    delta = (tok_per_j[1] - tok_per_j[0]) / tok_per_j[0] * 100
    ax.annotate(f'{delta:.0f}%', xy=(1, tok_per_j[1]), xytext=(1.3, tok_per_j[1]*1.1),
                fontsize=10, color='red', arrowprops=dict(arrowstyle='->', color='red'))

    # 4. TPOT p95 comparison
    ax = axes[3]
    bars = ax.bar(policies, tpot_p95_ms, color=[colors[p] for p in policies], edgecolor='black')
    ax.set_ylabel('TPOT p95 (ms)', fontsize=11)
    ax.set_title('Tail Latency', fontsize=12)
    ax.set_ylim(0, max(tpot_p95_ms) * 1.3)
    for bar, val in zip(bars, tpot_p95_ms):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}ms',
                ha='center', va='bottom', fontsize=10)
    overhead = (tpot_p95_ms[1] - tpot_p95_ms[0]) / tpot_p95_ms[0] * 100
    ax.annotate(f'+{overhead:.0f}%', xy=(1, tpot_p95_ms[1]), xytext=(1.3, tpot_p95_ms[1]*0.9),
                fontsize=10, color='red', arrowprops=dict(arrowstyle='->', color='red'))

    plt.suptitle('Latent Controller Overhead Analysis\n(Qwen2-0.5B, 64 prompt + 64 decode tokens)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()

    output_dir = Path('reports/latent_controller')
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'latent_overhead_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'latent_overhead_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'latent_overhead_comparison.png'}")

    # Also create a simple overhead summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ['Time', 'Energy', 'TPOT p95', 'Efficiency']
    overhead_pct = [
        (time_s[1] - time_s[0]) / time_s[0] * 100,
        (energy_j[1] - energy_j[0]) / energy_j[0] * 100,
        (tpot_p95_ms[1] - tpot_p95_ms[0]) / tpot_p95_ms[0] * 100,
        (tok_per_j[1] - tok_per_j[0]) / tok_per_j[0] * 100,
    ]

    colors_overhead = ['#e74c3c' if v > 0 else '#2ecc71' for v in overhead_pct]
    bars = ax.bar(metrics, overhead_pct, color=colors_overhead, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Change vs Auto (%)', fontsize=11)
    ax.set_title('Latent Controller Overhead from output_hidden_states=True\n(0.5B model, need optimization)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, overhead_pct):
        y_pos = val + (1 if val > 0 else -2)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.0f}%',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'latent_overhead_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'latent_overhead_summary.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'latent_overhead_summary.png'}")


if __name__ == '__main__':
    main()
