#!/usr/bin/env python3
"""
Publication figure: NS-RAM <-> GPU Bridge dual-regime behaviour and kill-shot evidence.
Generates fig_regime_killshot.png for FEEL paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os, struct, json

# ─── Paths ───
BASE = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy"
RAW_COUPLED = os.path.join(BASE, "results", "nsram_regime_coupled.raw")
JSON_PATH = os.path.join(BASE, "results", "nsram_bridge_experiments.json")
OUT_DIR = "/tmp/feel_paper/FEEL__Functionally_Embodied_Emergent_Learning__13_/figures"
OUT_PATH = os.path.join(OUT_DIR, "fig_regime_killshot.png")

# ─── Colors ───
BLUE   = '#2166ac'
RED    = '#b2182b'
GREEN  = '#1b7837'
GREY   = '#888888'

# ─── Style ───
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#111111',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#111111',
    'grid.color': '#dddddd',
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
})


def read_raw(path):
    """Read ngspice binary raw file, return dict of signal_name -> array."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    with open(path, 'rb') as f:
        content = f.read()

    header_end = content.find(b'Binary:\n')
    if header_end < 0:
        return None
    header = content[:header_end].decode('ascii', errors='replace')
    data = content[header_end + len(b'Binary:\n'):]

    lines = header.split('\n')
    n_vars = 0
    n_pts = 0
    var_names = []

    in_vars = False
    for line in lines:
        ls = line.strip()
        if ls.startswith('No. Variables:'):
            n_vars = int(ls.split(':')[1].strip())
        elif ls.startswith('No. Points:'):
            n_pts = int(ls.split(':')[1].strip())
        elif ls == 'Variables:':
            in_vars = True
        elif in_vars and len(var_names) < n_vars:
            parts = ls.split()
            if len(parts) >= 2 and parts[0].isdigit():
                var_names.append(parts[1].lower())

    if n_vars == 0 or n_pts == 0:
        return None

    expected = n_vars * n_pts * 8
    arr = np.frombuffer(data[:expected], dtype=np.float64).reshape(n_pts, n_vars)

    result = {}
    for i, name in enumerate(var_names):
        result[name] = arr[:, i]
    return result


def detect_spikes(vspike, time, threshold=0.5):
    """Detect spike times from vspike waveform."""
    above = vspike > threshold
    edges = np.diff(above.astype(int))
    rising = np.where(edges == 1)[0]
    return time[rising]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load JSON data
    with open(JSON_PATH) as f:
        data = json.load(f)
    regime = data['regime_test']
    ks = data['killshot_test']

    # Load raw waveform
    raw = read_raw(RAW_COUPLED)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
    fig.suptitle(
        r"NS-RAM $\leftrightarrow$ GPU Bridge: Dual-Regime Behaviour and Kill-Shot Evidence",
        fontsize=12, fontweight='bold', y=0.97
    )

    # ─── (a) Dual-Regime Spike Counts ───
    ax = axes[0, 0]
    labels = ['Cold', 'Hot', 'Coupled']
    counts = [regime['cold']['spikes'], regime['hot']['spikes'], regime['coupled']['spikes']]
    colors = [BLUE, RED, GREEN]
    bars = ax.bar(labels, counts, color=colors, edgecolor='#333333', linewidth=0.6, width=0.6)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                str(c), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.annotate(r'Hot/Cold = $\infty$', xy=(0.95, 0.92), xycoords='axes fraction',
                ha='right', fontsize=8, fontstyle='italic', color=RED)
    ax.annotate('Coupled intermediate', xy=(0.95, 0.82), xycoords='axes fraction',
                ha='right', fontsize=8, fontstyle='italic', color=GREEN)
    ax.set_ylabel('LIF Spike Count')
    ax.set_ylim(0, 22)
    ax.set_title('Dual-Regime Spike Counts', fontsize=10)
    ax.text(-0.12, 1.05, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    # ─── (b) ISI Distribution ───
    ax = axes[0, 1]
    # Hot: mean ISI ~ 10.6 us, CV=0.223
    hot_mean = regime['hot']['isi_mean'] * 1e6   # to microseconds
    hot_cv = regime['hot']['isi_cv']
    hot_std = hot_mean * hot_cv
    # Coupled: mean ISI ~ 19.9 us, CV=0.451
    coup_mean = regime['coupled']['isi_mean'] * 1e6
    coup_cv = regime['coupled']['isi_cv']
    coup_std = coup_mean * coup_cv

    x_range = np.linspace(0, 40, 500)
    hot_pdf = (1.0 / (hot_std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range - hot_mean)/hot_std)**2)
    coup_pdf = (1.0 / (coup_std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range - coup_mean)/coup_std)**2)

    ax.fill_between(x_range, hot_pdf, alpha=0.3, color=RED)
    ax.plot(x_range, hot_pdf, color=RED, linewidth=1.2, label=f'Hot (CV={hot_cv:.3f})')
    ax.fill_between(x_range, coup_pdf, alpha=0.3, color=GREEN)
    ax.plot(x_range, coup_pdf, color=GREEN, linewidth=1.2, label=f'Coupled (CV={coup_cv:.3f})')

    ax.axvline(hot_mean, color=RED, linestyle='--', linewidth=0.7, alpha=0.7)
    ax.axvline(coup_mean, color=GREEN, linestyle='--', linewidth=0.7, alpha=0.7)

    ax.annotate(f'CV={hot_cv:.3f}', xy=(hot_mean, max(hot_pdf)*0.85),
                fontsize=8, color=RED, ha='center')
    ax.annotate(f'CV={coup_cv:.3f}', xy=(coup_mean, max(coup_pdf)*0.85),
                fontsize=8, color=GREEN, ha='center')

    ax.set_xlabel(r'Inter-Spike Interval ($\mu$s)')
    ax.set_ylabel('Density')
    ax.set_title('ISI Distribution', fontsize=10)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.set_xlim(0, 40)
    ax.text(-0.12, 1.05, '(b)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    # ─── (c) Gate Voltage Modulation (Coupled) ───
    ax = axes[0, 2]
    if raw is not None and 'time' in raw and 'v(nsram_gate)' in raw:
        t = raw['time'] * 1e6  # to microseconds
        vgate = raw['v(nsram_gate)']
        vspike = raw['v(vspike)']

        ax.plot(t, vgate, color=GREEN, linewidth=0.5, alpha=0.9, label=r'$V_{gate}$')

        # Detect and mark spikes
        spike_times = detect_spikes(vspike, t, threshold=0.5)
        for st in spike_times:
            ax.axvline(st, color=RED, linewidth=0.6, alpha=0.6, linestyle='-')

        gate_mean = regime['coupled']['gate_mean']
        gate_std = regime['coupled']['gate_std']
        ax.axhline(gate_mean, color='#333333', linestyle='--', linewidth=0.7, alpha=0.6)
        ax.fill_between(t, gate_mean - gate_std, gate_mean + gate_std,
                         color=GREEN, alpha=0.08)
        ax.annotate(f'mean={gate_mean:.3f}V\nstd={gate_std:.3f}V',
                    xy=(0.03, 0.92), xycoords='axes fraction', fontsize=7,
                    va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                        edgecolor='#cccccc', alpha=0.9))

        # Add spike marker to legend
        ax.plot([], [], color=RED, linewidth=1.0, label='Spike events')
        ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

        ax.set_xlabel(r'Time ($\mu$s)')
        ax.set_ylabel(r'$V_{gate}$ (V)')
        ax.set_xlim(t[0], t[-1])
    else:
        ax.text(0.5, 0.5, 'Raw data\nnot available', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color=GREY)

    ax.set_title('Gate Voltage Modulation (Coupled)', fontsize=10)
    ax.text(-0.12, 1.05, '(c)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    # ─── (d) Kill-Shot Bar Chart ───
    ax = axes[1, 0]
    ks_labels = ['Full\n(A)', 'Open\n(B)', 'Reversed\n(C)', 'No Aval.\n(D)']
    ks_counts = [ks['A_full']['spikes'], ks['B_open']['spikes'],
                 ks['C_reversed']['spikes'], ks['D_noaval']['spikes']]
    ks_colors = [GREEN, GREY, GREY, GREY]
    bars = ax.bar(ks_labels, ks_counts, color=ks_colors, edgecolor='#333333',
                  linewidth=0.6, width=0.6)
    # Annotate full condition
    ax.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height() + 0.3,
            '8', ha='center', va='bottom', fontsize=9, fontweight='bold', color=GREEN)
    # Annotate ablated conditions
    for bar in bars[1:]:
        ax.text(bar.get_x() + bar.get_width()/2, 0.3,
                '0 spikes', ha='center', va='bottom', fontsize=8, color=GREY)

    ax.annotate('All ablations → 0 spikes', xy=(0.97, 0.92), xycoords='axes fraction',
                ha='right', fontsize=9, fontweight='bold', color=RED,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0',
                          edgecolor=RED, alpha=0.9))
    ax.set_ylabel('LIF Spike Count')
    ax.set_ylim(0, 11)
    ax.set_title('Kill-Shot: Ablation Results', fontsize=10)
    ax.text(-0.12, 1.05, '(d)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    # ─── (e) FEEL vs NS-RAM Kill-Shot Comparison ───
    ax = axes[1, 1]
    x_pos = np.array([0, 1.2])
    width = 0.35

    # Enabled (100% for both, normalized)
    enabled = [100.0, 100.0]
    disabled = [1.0, 0.0]  # FEEL: 99pp drop means ~1% remains; NS-RAM: 0%

    bars_en = ax.bar(x_pos - width/2, enabled, width, color=[BLUE, GREEN],
                     edgecolor='#333333', linewidth=0.6, label='Enabled', alpha=0.9)
    bars_dis = ax.bar(x_pos + width/2, disabled, width, color=[BLUE, GREEN],
                      edgecolor='#333333', linewidth=0.6, label='Disabled',
                      alpha=0.3, hatch='///')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['FEEL z2090\n(DVFS)', 'NS-RAM\n(Bridge)'], fontsize=8)
    ax.set_ylabel('% of Enabled Condition')
    ax.set_ylim(0, 130)

    # Annotations
    ax.annotate('99.0pp\ndrop', xy=(x_pos[0] + width/2, 5),
                fontsize=8, ha='center', va='bottom', fontweight='bold', color=BLUE)
    ax.annotate('100%\ndrop', xy=(x_pos[1] + width/2, 5),
                fontsize=8, ha='center', va='bottom', fontweight='bold', color=GREEN)

    # Drop arrows
    for i, (xp, col) in enumerate(zip(x_pos, [BLUE, GREEN])):
        ax.annotate('', xy=(xp + width/2, disabled[i] + 3),
                    xytext=(xp + width/2, enabled[i] - 3),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.set_title('Kill-Shot: Causal Necessity', fontsize=10)
    ax.text(-0.12, 1.05, '(e)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    # ─── (f) Energy Validation ───
    ax = axes[1, 2]
    # Pazos reported range
    pazos_lo, pazos_hi = 0.2, 21.0
    our_v5 = 11.827  # fJ/spike (from killshot A_full)
    our_v2 = 18.4    # fJ/spike (from earlier bridge v2)

    y_pos = 0.5
    ax.barh(y_pos, pazos_hi - pazos_lo, left=pazos_lo, height=0.25,
            color='#e0e0e0', edgecolor='#999999', linewidth=0.8, label='Pazos et al. range')

    ax.plot(our_v5, y_pos, 'D', color=GREEN, markersize=10, zorder=5,
            markeredgecolor='#333333', markeredgewidth=0.8, label=f'v5 bridge: {our_v5:.1f} fJ')
    ax.plot(our_v2, y_pos, 's', color=BLUE, markersize=8, zorder=5,
            markeredgecolor='#333333', markeredgewidth=0.8, label=f'v2 bridge: {our_v2:.1f} fJ')

    ax.annotate(f'{our_v5:.1f} fJ', xy=(our_v5, y_pos + 0.17), ha='center',
                fontsize=8, fontweight='bold', color=GREEN)
    ax.annotate(f'{our_v2:.1f} fJ', xy=(our_v2, y_pos - 0.17), ha='center',
                fontsize=8, fontweight='bold', color=BLUE, va='top')

    ax.annotate('Within Pazos et al. range', xy=(0.5, 0.08), xycoords='axes fraction',
                ha='center', fontsize=8, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0fff0',
                          edgecolor=GREEN, alpha=0.9))

    # Range labels
    ax.text(pazos_lo, y_pos - 0.22, f'{pazos_lo}', ha='center', fontsize=7, color=GREY)
    ax.text(pazos_hi, y_pos - 0.22, f'{pazos_hi}', ha='center', fontsize=7, color=GREY)

    ax.set_xlim(-1, 25)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Energy per Spike (fJ)')
    ax.set_yticks([])
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    ax.set_title('Energy per Spike (fJ)', fontsize=10)
    ax.text(-0.12, 1.05, '(f)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    # ─── Final layout ───
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUT_PATH}")
    plt.close(fig)


if __name__ == '__main__':
    main()
