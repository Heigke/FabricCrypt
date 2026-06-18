#!/usr/bin/env python3
"""z2159_bridge_plots.py — Generate publication figures for z2153-z2158 bridge results."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIGDIR = RESULTS / 'FEEL_paper' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-4' / 'figures'

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_z2153_isi():
    """z2153: ISI CV across regimes and jitter amplitudes."""
    with open(RESULTS / 'z2153_isi_injection.json') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel (a): ISI CV by regime and amplitude
    ax = axes[0]
    regimes = ['Subcritical\n($V_g=0.30$)', 'Critical\n($V_g=0.60$)', 'Supercritical\n($V_g=0.90$)']
    regime_keys = ['subcritical', 'critical', 'supercritical']
    amps = [0.00, 0.08, 0.15]
    colors_amp = ['#4C72B0', '#DD8452', '#55A868']
    width = 0.22
    x = np.arange(3)

    for j, amp in enumerate(amps):
        cvs = []
        for rk in regime_keys:
            key = f'{rk}_amp_{amp:.2f}'
            if key in data['test_results']:
                cvs.append(data['test_results'][key].get('cv', 0))
            else:
                cvs.append(0)
        ax.bar(x + j * width - width, cvs, width, label=f'Amp={amp:.2f}',
               color=colors_amp[j], edgecolor='white', linewidth=0.5)

    ax.axhspan(0.3, 2.0, alpha=0.1, color='green', label='Lanza range [0.3, 2.0]')
    ax.axhline(0.3, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=8)
    ax.set_ylabel('ISI CV')
    ax.set_title('(a) ISI CV by regime and jitter amplitude')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_ylim(0, 0.85)

    # Panel (b): Spike counts showing regime differentiation
    ax = axes[1]
    for j, amp in enumerate(amps):
        spikes = []
        for rk in regime_keys:
            key = f'{rk}_amp_{amp:.2f}'
            if key in data['test_results']:
                spikes.append(data['test_results'][key].get('spikes', 0))
            else:
                spikes.append(0)
        ax.bar(x + j * width - width, spikes, width, label=f'Amp={amp:.2f}',
               color=colors_amp[j], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=8)
    ax.set_ylabel('Total spikes (8s)')
    ax.set_title('(b) Spike counts by regime')
    ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    out = FIGDIR / 'fig_z2153_isi_injection.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


def plot_z2154_1f():
    """z2154: 1/f filter PSD and ISI differentiation."""
    with open(RESULTS / 'z2154_iir_conductance.json') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))

    # Panel (a): Voss-McCartney filter concept
    ax = axes[0]
    # Generate sample white and 1/f PSD for illustration
    freqs = np.logspace(-2, 1, 200)
    white_psd = np.ones_like(freqs) * 1.0
    pink_psd = 1.0 / (freqs ** 1.238)
    ax.loglog(freqs, white_psd, 'b-', alpha=0.7, linewidth=1.5, label='White (GPU raw)')
    ax.loglog(freqs, pink_psd, 'r-', alpha=0.7, linewidth=1.5, label='1/$f$ filtered')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('(a) Voss-McCartney filter')
    ax.legend(fontsize=7)
    ax.text(0.05, 0.12, f'Slope = {data["filter"]["filtered_psd_slope"]:.2f}',
            transform=ax.transAxes, fontsize=7, color='red')

    # Panel (b): ISI CV comparison
    ax = axes[1]
    conditions = ['White', '1/$f$']
    cvs = [data['results']['white']['aggregate_isi_cv'],
           data['results']['1/f']['aggregate_isi_cv']]
    bars = ax.bar(conditions, cvs, color=['#4C72B0', '#C44E52'],
                  edgecolor='white', linewidth=0.5, width=0.5)
    ax.axhspan(0.3, 2.0, alpha=0.08, color='green')
    ax.set_ylabel('ISI CV')
    ax.set_title('(b) ISI CV: white vs 1/$f$')
    for bar, cv in zip(bars, cvs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{cv:.3f}', ha='center', fontsize=8)
    ax.set_ylim(0, 0.95)

    # Panel (c): ACF showing temporal memory
    ax = axes[2]
    acf_val = data['filter']['filtered_acf1']
    ax.bar(['ACF(1)'], [acf_val], color='#C44E52', width=0.4)
    ax.axhline(0.5, color='green', linestyle='--', linewidth=1, label='Threshold')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('(c) Temporal memory')
    ax.set_ylim(0, 1.0)
    ax.text(0, acf_val + 0.03, f'{acf_val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = FIGDIR / 'fig_z2154_1f_conductance.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


def plot_z2155_heterogeneous():
    """z2155: Process variation decorrelation."""
    with open(RESULTS / 'z2155_heterogeneous_thresholds.json') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    spreads = data['spreads_tested']
    rate_cvs = [data['results'][f'spread_{s:.2f}']['rate_cv'] for s in spreads]
    corrs = [data['results'][f'spread_{s:.2f}']['cross_neuron_correlation'] for s in spreads]

    # Panel (a): Rate CV vs spread
    ax = axes[0]
    ax.plot(spreads, rate_cvs, 'o-', color='#4C72B0', linewidth=1.5, markersize=6)
    ax.axhline(data['evaluation']['baseline_cv'], color='gray', linestyle='--',
               linewidth=0.8, label=f'Baseline ({data["evaluation"]["baseline_cv"]:.4f})')
    ax.fill_between([0, 0.12], [0.02]*2, [0]*2, alpha=0.0)
    ax.set_xlabel('$V_g$ spread ($\\pm$)')
    ax.set_ylabel('Spike rate CV across neurons')
    ax.set_title('(a) Rate heterogeneity vs process spread')
    peak_idx = np.argmax(rate_cvs)
    ax.annotate(f'Peak: {rate_cvs[peak_idx]:.3f}\n(±{spreads[peak_idx]:.2f})',
                xy=(spreads[peak_idx], rate_cvs[peak_idx]),
                xytext=(spreads[peak_idx] + 0.02, rate_cvs[peak_idx] + 0.01),
                fontsize=7, arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
    ax.legend(fontsize=7)

    # Panel (b): Cross-neuron correlation vs spread
    ax = axes[1]
    ax.plot(spreads, corrs, 's-', color='#C44E52', linewidth=1.5, markersize=6)
    ax.set_xlabel('$V_g$ spread ($\\pm$)')
    ax.set_ylabel('Cross-neuron correlation')
    ax.set_title('(b) Decorrelation with process variation')
    ax.set_ylim(0.9, 1.01)
    min_idx = np.argmin(corrs)
    ax.annotate(f'Min: {corrs[min_idx]:.3f}\n(±{spreads[min_idx]:.2f})',
                xy=(spreads[min_idx], corrs[min_idx]),
                xytext=(spreads[min_idx] + 0.02, corrs[min_idx] + 0.02),
                fontsize=7, arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    plt.tight_layout()
    out = FIGDIR / 'fig_z2155_heterogeneous.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


def plot_z2158_psd():
    """z2158: Cross-substrate PSD correlation."""
    with open(RESULTS / 'z2158_gpu_fpga_psd_correlation.json') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel (a): PSD slopes comparison
    ax = axes[0]
    labels = ['GPU\njitter', 'GPU\nPERF', 'GPU\ncycles', 'FPGA\nspikes', 'FPGA\nVmem']
    slopes = [
        data['gpu']['jitter_psd_slope'],
        data['gpu']['perf_psd_slope'],
        data['gpu']['cycle_psd_slope'],
        data['fpga']['aggregate_psd_slope'],
        data['fpga']['vmem_psd_slope'],
    ]
    colors = ['#4C72B0', '#4C72B0', '#4C72B0', '#C44E52', '#C44E52']
    bars = ax.bar(labels, slopes, color=colors, edgecolor='white', linewidth=0.5, width=0.6)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axhline(-1.0, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='1/$f$ target')
    ax.set_ylabel('PSD slope')
    ax.set_title('(a) PSD slopes: GPU (blue) vs FPGA (red)')
    ax.legend(fontsize=7)
    for bar, s in zip(bars, slopes):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.02 if y >= 0 else y - 0.06,
                f'{s:.2f}', ha='center', fontsize=7)

    # Panel (b): Bridge metrics scorecard
    ax = axes[1]
    metrics = ['Slope\nsimilarity', 'Spectral\ncoherence', 'Pearson\n|$r$|']
    values = [
        1.0 - data['cross_substrate']['psd_slope_diff'],  # normalized similarity
        data['cross_substrate']['mean_coherence'],
        abs(data['cross_substrate']['pearson_r']),
    ]
    thresholds = [0.0, 0.05, 0.01]  # min values for pass
    passes = [
        data['cross_substrate']['psd_slope_diff'] < 1.0,
        data['cross_substrate']['mean_coherence'] > 0.05,
        abs(data['cross_substrate']['pearson_r']) > 0.01,
    ]
    colors_bar = ['#55A868' if p else '#C44E52' for p in passes]
    bars = ax.bar(metrics, values, color=colors_bar, edgecolor='white', linewidth=0.5, width=0.5)
    for bar, v, p in zip(bars, values, passes):
        label = 'PASS' if p else 'FAIL'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}\n{label}', ha='center', fontsize=7, fontweight='bold')
    ax.set_ylabel('Metric value')
    ax.set_title('(b) Cross-substrate coupling metrics')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    out = FIGDIR / 'fig_z2158_psd_correlation.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


def plot_bridge_summary():
    """Summary scorecard for all z2153-z2161 bridge metrics."""
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    experiments = [
        ('z2153: T52 ISI CV\n(critical)', 0.587, True),
        ('z2153: Regime\ndifferentiation', 0.339, True),
        ('z2154: 1/$f$ PSD\nslope', 1.238, True),
        ('z2154: ACF(1)\ntemporal memory', 0.852, True),
        ('z2154: ISI CV\nwhite vs 1/$f$', 0.119, True),
        ('z2155: Rate CV\nincrease', 0.063, True),
        ('z2155: Neuron\ndecorrelation', 0.067, True),
        ('z2158: Spectral\ncoherence', 0.187, True),
        ('z2158: PSD slope\nsimilarity', 0.879, True),
        ('z2158: Pearson\n|$r$|', 0.008, False),
        ('z2160: Native 1/$f$\nPSD slope', 1.546, True),
        ('z2160: ISI CV\n(thermal Lanza)', 1.449, True),
        ('z2160: Power\nACF(1)', 0.989, True),
        ('z2160: Thermal vs\nwhite diff.', 1.548, True),
        ('z2161: Fused PSD\nslope', 0.395, True),
        ('z2161: Fused ISI CV\n(Lanza)', 1.479, True),
        ('z2161: Entropy\n$H$ gain', 0.0, False),
    ]

    names = [e[0] for e in experiments]
    values = [e[1] for e in experiments]
    passes = [e[2] for e in experiments]
    colors = ['#55A868' if p else '#C44E52' for p in passes]

    y = np.arange(len(experiments))
    bars = ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    for bar, v, p in zip(bars, values, passes):
        label = 'PASS' if p else 'FAIL'
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{v:.3f} ({label})', va='center', fontsize=7, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('Metric value')
    ax.set_title('GPU$\\leftrightarrow$FPGA Deep Bridge Battery (z2153--z2161): 15/17 PASS')
    ax.invert_yaxis()
    ax.set_xlim(0, 1.8)

    # Add pass count annotation
    n_pass = sum(passes)
    ax.text(0.95, 0.05, f'{n_pass}/{len(experiments)} PASS',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='right', va='bottom', color='#55A868',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#55A868', alpha=0.9))

    plt.tight_layout()
    out = FIGDIR / 'fig_z2153_z2158_bridge_summary.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


if __name__ == '__main__':
    print("Generating z2153-z2158 bridge figures...")
    plot_z2153_isi()
    plot_z2154_1f()
    plot_z2155_heterogeneous()
    plot_z2158_psd()
    plot_bridge_summary()
    print("Done.")
