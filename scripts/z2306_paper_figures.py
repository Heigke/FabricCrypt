#!/usr/bin/env python3
"""z2306_paper_figures.py — Publication-quality figures from z2298 results.

Generates 4 figures from existing z2298 data (no hardware needed):
  1. MC Decay Curve
  2. Benchmark Comparison Radar
  3. Cross-Substrate Synergy Bars
  4. Temporal Feature Importance

Usage:
    venv/bin/python scripts/z2306_paper_figures.py
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

JSON_PATH = RESULTS / "z2298_hip_temporal.json"
STATES_PATH = RESULTS / "z2298_fpga_states.npy"
DSPIKES_PATH = RESULTS / "z2298_fpga_dspikes.npy"

# ── Style ──────────────────────────────────────────────────────────────
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        pass  # fall back to default

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Condition display config ───────────────────────────────────────────
CONDITIONS = ['SOFTWARE_ESN', 'GPU_HIP', 'FPGA_ONLY', 'BRIDGE_CONCAT', 'BRIDGE_MAC']
COLORS = {
    'SOFTWARE_ESN':  '#888888',
    'GPU_HIP':       '#1f77b4',
    'FPGA_ONLY':     '#2ca02c',
    'BRIDGE_CONCAT': '#d62728',
    'BRIDGE_MAC':    '#ff7f0e',
}
LABELS = {
    'SOFTWARE_ESN':  'Software ESN',
    'GPU_HIP':       'GPU HIP',
    'FPGA_ONLY':     'FPGA Only',
    'BRIDGE_CONCAT': 'Bridge (Concat)',
    'BRIDGE_MAC':    'Bridge (MAC)',
}

# ── Load data ──────────────────────────────────────────────────────────
print("Loading z2298 results...")
with open(JSON_PATH) as f:
    data = json.load(f)
conds = data["conditions"]

states = np.load(STATES_PATH)
dspikes = np.load(DSPIKES_PATH)
print(f"  JSON: {len(conds)} conditions")
print(f"  States: {states.shape}, Dspikes: {dspikes.shape}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: MC Decay Curve
# ═══════════════════════════════════════════════════════════════════════
def fig_mc_decay():
    fig, ax = plt.subplots(figsize=(8, 5))
    delays = list(range(1, 21))

    for cond in CONDITIONS:
        mc = conds[cond]["mc_per_delay"]
        r2 = [mc[str(d)] for d in delays]
        ax.plot(delays, r2, 'o-', color=COLORS[cond], label=LABELS[cond],
                markersize=4, linewidth=1.8)

    ax.set_xlabel("Delay $d$")
    ax.set_ylabel("$R^2$ (memory capacity per delay)")
    ax.set_title("Memory Capacity Decay by Substrate")
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES / f"fig_mc_decay.{fmt}")
    plt.close(fig)
    print("  Saved fig_mc_decay.pdf/.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Benchmark Comparison Radar
# ═══════════════════════════════════════════════════════════════════════
def fig_benchmark_radar():
    metrics = ['MC', 'XOR1', 'XOR5', '1-NARMA5', 'Wave4']
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for cond in CONDITIONS:
        c = conds[cond]
        vals = [
            c["mc_total"] / 15.0,                    # MC normalized to 0-1 (max ~15)
            c["xor"]["tau1"],                         # XOR1 already 0-1
            c["xor"]["tau5"],                         # XOR5 already 0-1
            1.0 - c["narma"]["narma5"],               # 1-NARMA (lower NRMSE = better)
            c["wave4_acc"],                           # Wave4 already 0-1
        ]
        vals += vals[:1]  # close polygon
        ax.plot(angles, vals, 'o-', color=COLORS[cond], label=LABELS[cond],
                linewidth=1.8, markersize=4)
        ax.fill(angles, vals, color=COLORS[cond], alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title("Benchmark Comparison by Substrate", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES / f"fig_benchmark_radar.{fmt}")
    plt.close(fig)
    print("  Saved fig_benchmark_radar.pdf/.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Cross-Substrate Synergy
# ═══════════════════════════════════════════════════════════════════════
def fig_synergy_bar():
    metric_keys = ['MC', 'XOR1', 'XOR5', 'NARMA-5', 'Wave4']
    trio = ['FPGA_ONLY', 'GPU_HIP', 'BRIDGE_CONCAT']
    trio_colors = [COLORS[c] for c in trio]
    trio_labels = [LABELS[c] for c in trio]

    def get_val(cond, metric):
        c = conds[cond]
        if metric == 'MC':
            return c["mc_total"]
        elif metric == 'XOR1':
            return c["xor"]["tau1"] * 100
        elif metric == 'XOR5':
            return c["xor"]["tau5"] * 100
        elif metric == 'NARMA-5':
            return c["narma"]["narma5"]
        elif metric == 'Wave4':
            return c["wave4_acc"] * 100
        return 0

    fig, ax = plt.subplots(figsize=(10, 5))
    n_groups = len(metric_keys)
    n_bars = len(trio)
    bar_width = 0.25
    x = np.arange(n_groups)

    for i, cond in enumerate(trio):
        vals = [get_val(cond, m) for m in metric_keys]
        bars = ax.bar(x + i * bar_width, vals, bar_width,
                      color=trio_colors[i], label=trio_labels[i], edgecolor='white')

    # Annotate synergy: BRIDGE_CONCAT exceeds BOTH individual substrates
    for j, m in enumerate(metric_keys):
        v_fpga = get_val('FPGA_ONLY', m)
        v_gpu = get_val('GPU_HIP', m)
        v_bridge = get_val('BRIDGE_CONCAT', m)
        best_indiv = max(v_fpga, v_gpu)

        # For NARMA, lower is better
        if m == 'NARMA-5':
            is_synergy = v_bridge < min(v_fpga, v_gpu)
            if is_synergy:
                margin = min(v_fpga, v_gpu) - v_bridge
                ax.annotate(f'-{margin:.2f}',
                            xy=(x[j] + 2 * bar_width, v_bridge),
                            xytext=(0, -18), textcoords='offset points',
                            fontsize=8, color='#d62728', fontweight='bold',
                            ha='center')
        else:
            is_synergy = v_bridge > best_indiv
            if is_synergy:
                margin = v_bridge - best_indiv
                unit = '' if m == 'MC' else '%'
                ax.annotate(f'+{margin:.1f}{unit}',
                            xy=(x[j] + 2 * bar_width, v_bridge),
                            xytext=(0, 5), textcoords='offset points',
                            fontsize=8, color='#d62728', fontweight='bold',
                            ha='center')

    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Score")
    ax.set_title("Cross-Substrate Synergy: Bridge vs. Individual Substrates")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(metric_keys)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES / f"fig_synergy_bar.{fmt}")
    plt.close(fig)
    print("  Saved fig_synergy_bar.pdf/.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Temporal Feature Importance
# ═══════════════════════════════════════════════════════════════════════
def build_temporal_features(states, dspikes=None, n_select=24, seed=42):
    """Replicate build_temporal_features from z2298."""
    n_steps, n_ch = states.shape
    delta = np.diff(states, axis=0)
    delta = np.vstack([np.zeros((1, n_ch)), delta])
    feats = [states, delta]
    feat_groups = ['raw'] * n_ch + ['delta'] * n_ch

    if dspikes is not None:
        feats.append(dspikes)
        feat_groups += ['raw'] * dspikes.shape[1]  # dspikes counted as raw input

    rng = np.random.default_rng(seed)
    qi = np.sort(rng.choice(n_ch, size=min(n_select, n_ch), replace=False))
    vm_q = states[:, qi]
    n_q = len(qi)

    tau_list = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    # Order-2 temporal products
    for tau in tau_list:
        shifted = np.zeros_like(vm_q)
        shifted[tau:] = vm_q[:-tau]
        feats.append(vm_q * shifted)
        feat_groups += ['order-2'] * n_q
        if dspikes is not None:
            ds_q = dspikes[:, qi] if dspikes.shape[1] >= n_ch else \
                   dspikes[:, :min(n_select, dspikes.shape[1])]
            if ds_q.shape[1] == vm_q.shape[1]:
                feats.append(ds_q * shifted)
                feat_groups += ['order-2'] * n_q

    # Order-3 temporal products
    for i, t1 in enumerate(tau_list):
        for t2 in tau_list[i+1:]:
            if t2 > 10:
                continue
            sh1 = np.zeros_like(vm_q)
            sh2 = np.zeros_like(vm_q)
            sh1[t1:] = vm_q[:-t1]
            sh2[t2:] = vm_q[:-t2]
            feats.append(vm_q * sh1 * sh2)
            feat_groups += ['order-3'] * n_q

    feats.append(np.square(vm_q))
    feat_groups += ['squared'] * n_q

    feats.append((vm_q > np.median(vm_q, axis=0)).astype(float))
    feat_groups += ['binary'] * n_q

    return np.hstack(feats), feat_groups


def fig_temporal_ablation():
    print("  Building temporal features from FPGA states...")
    X, groups = build_temporal_features(states, dspikes)
    n_steps = X.shape[0]

    # Build MC target: input signal delayed by 1
    # Use random input that was driving the reservoir (reconstruct from states)
    rng = np.random.default_rng(42)
    u = rng.standard_normal(n_steps)

    # Target = u(t-1) for memory capacity d=1
    y = np.zeros(n_steps)
    y[1:] = u[:-1]

    # Trim warmup
    warmup = 50
    X_tr = X[warmup:]
    y_tr = y[warmup:]

    # Normalize features
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd < 1e-10] = 1.0
    X_norm = (X_tr - mu) / sd

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_norm, y_tr)
    weights = np.abs(ridge.coef_)

    # Aggregate by group
    groups_arr = np.array(groups)
    group_names = ['raw', 'delta', 'order-2', 'order-3', 'squared', 'binary']
    group_means = []
    for g in group_names:
        mask = groups_arr == g
        if mask.sum() > 0:
            group_means.append(weights[mask].mean())
        else:
            group_means.append(0.0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_bar = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']
    bars = ax.bar(group_names, group_means, color=colors_bar, edgecolor='white',
                  linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, group_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Feature Group")
    ax.set_ylabel("Mean |weight| (Ridge, MC d=1)")
    ax.set_title("Temporal Feature Importance by Product Order")
    ax.grid(True, axis='y', alpha=0.3)

    for fmt in ('pdf', 'png'):
        fig.savefig(FIGURES / f"fig_temporal_ablation.{fmt}")
    plt.close(fig)
    print("  Saved fig_temporal_ablation.pdf/.png")


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== z2306: Paper Figures ===\n")

    print("[1/4] MC Decay Curve")
    fig_mc_decay()

    print("[2/4] Benchmark Radar")
    fig_benchmark_radar()

    print("[3/4] Cross-Substrate Synergy")
    fig_synergy_bar()

    print("[4/4] Temporal Feature Importance")
    fig_temporal_ablation()

    print(f"\nAll figures saved to {FIGURES}/")
    print("Done.")
