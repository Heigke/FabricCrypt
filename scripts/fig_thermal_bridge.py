#!/usr/bin/env python3
"""
fig_thermal_bridge.py — Publication figure: NS-RAM <-> ThermalSoftmax thermal parallel.
2x2 panel: (a) spike count vs T, (b) waveform at 328K, (c) Boltzmann parallel, (d) math.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, struct, json

# ── Paths ──
BASE = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'
JSON_PATH = os.path.join(BASE, 'results/nsram_bridge_experiments.json')
RAW_PATH  = os.path.join(BASE, 'results/nsram_thermal_T328.raw')
OUT_DIR   = '/tmp/feel_paper/FEEL__Functionally_Embodied_Emergent_Learning__13_/figures'
OUT_PATH  = os.path.join(OUT_DIR, 'fig_thermal_bridge.png')

# ── Colors (accessible) ──
BLUE   = '#2166ac'
RED    = '#b2182b'
GREEN  = '#1b7837'
BLACK  = '#222222'

# ── rcParams: white, serif, publication ──
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': BLACK,
    'axes.labelcolor': BLACK,
    'axes.linewidth': 0.8,
    'xtick.color': BLACK,
    'ytick.color': BLACK,
    'text.color': BLACK,
    'font.family': 'serif',
    'font.size': 9,
    'mathtext.fontset': 'cm',
})


# ── Raw file reader (from plot_bridge_experiments.py) ──
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
    flags_real = True

    in_vars = False
    for line in lines:
        ls = line.strip()
        if ls.startswith('No. Variables:'):
            n_vars = int(ls.split(':')[1].strip())
        elif ls.startswith('No. Points:'):
            n_pts = int(ls.split(':')[1].strip())
        elif ls.startswith('Flags:'):
            flags_real = 'real' in ls.lower()
        elif ls == 'Variables:':
            in_vars = True
        elif in_vars and len(var_names) < n_vars:
            parts = ls.split()
            if len(parts) >= 2 and parts[0].isdigit():
                var_names.append(parts[1].lower())

    if n_vars == 0 or n_pts == 0:
        return None

    if flags_real:
        expected = n_vars * n_pts * 8
        arr = np.frombuffer(data[:expected], dtype=np.float64).reshape(n_pts, n_vars)
    else:
        expected = n_vars * n_pts * 16
        arr_c = np.frombuffer(data[:expected], dtype=np.complex128).reshape(n_pts, n_vars)
        arr = np.real(arr_c)

    result = {}
    for i, name in enumerate(var_names):
        result[name] = arr[:, i]
    return result


def spike_times(time, vspike, threshold=0.75):
    above = vspike > threshold
    edges = np.where(np.diff(above.astype(int)) > 0)[0]
    return time[edges] if len(edges) > 0 else np.array([])


# ── Load data ──
with open(JSON_PATH) as f:
    jdata = json.load(f)

thermal = jdata['thermal_test']
temps   = [300, 313, 328, 343, 358]
spikes  = [thermal[str(t)]['spikes'] for t in temps]
bvpars  = [thermal[str(t)]['bvpar_mean'] for t in temps]

raw = read_raw(RAW_PATH)

# ── Figure ──
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(
    r'NS-RAM $\leftrightarrow$ ThermalSoftmax: Temperature as Computational Modulator',
    fontsize=11, fontweight='bold', y=0.97,
)

# ════════════════════════════════════════════════════════════
# (a) Spike Count vs Temperature
# ════════════════════════════════════════════════════════════
ax = axes[0, 0]
cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=290, vmax=365)
bar_colors = [cmap(norm(t)) for t in temps]
bars = ax.bar([str(t) for t in temps], spikes, color=bar_colors, edgecolor=BLACK, linewidth=0.6)

# BVpar annotation on each bar
for i, (bar, bv) in enumerate(zip(bars, bvpars)):
    y_pos = bar.get_height() + 0.4
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'BV={bv:.2f}', ha='center', va='bottom', fontsize=7, color='#555555')

# Threshold annotation between 313K and 328K
ax.annotate('', xy=(2, 12), xytext=(1, 1),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5, ls='--'))
ax.text(1.5, 7, 'threshold\ntransition', ha='center', va='center',
        fontsize=7.5, color=RED, fontstyle='italic')

ax.set_xlabel('Temperature (K)', fontsize=9)
ax.set_ylabel('LIF Spike Count', fontsize=9)
ax.set_ylim(0, 17)
ax.text(0.03, 0.95, r'$\mathbf{(a)}$', transform=ax.transAxes,
        fontsize=10, va='top', ha='left')

# ════════════════════════════════════════════════════════════
# (b) NS-RAM Waveform at 328K
# ════════════════════════════════════════════════════════════
ax = axes[0, 1]
if raw is not None:
    time_s = raw.get('time', raw.get('v-sweep', None))
    vmem   = raw.get('v(vmem)', None)
    vspk   = raw.get('v(vspike)', None)

    if time_s is None:
        # first column is usually time
        names = list(raw.keys())
        time_s = raw[names[0]]

    if time_s is not None and vmem is not None and vspk is not None:
        t_us = time_s * 1e6  # convert to us

        # Find spike cluster and zoom to ~55us window with 4 clean spikes
        st = spike_times(time_s, vspk, threshold=0.75)
        if len(st) >= 6:
            # Show spikes 3-6 (indices 2-5): covers ~40-91 us
            t_start = st[2] - 5e-6
            t_end   = st[5] + 5e-6
        elif len(st) >= 2:
            t_start = st[0] - 5e-6
            t_end   = st[-1] + 5e-6
        else:
            t_start = time_s[len(time_s)//3]
            t_end   = t_start + 50e-6

        mask = (time_s >= t_start) & (time_s <= t_end)
        t_plot = t_us[mask]
        ax.plot(t_plot, vmem[mask], color=BLUE, lw=0.8, label='V(vmem)')
        ax.plot(t_plot, vspk[mask], color=RED,  lw=0.8, label='V(vspike)')

        # Annotate ISI on one pair
        st_us = st * 1e6
        vis = st_us[(st_us >= t_plot.min()) & (st_us <= t_plot.max())]
        if len(vis) >= 2:
            isi_us = vis[1] - vis[0]
            ymid = 1.15
            ax.annotate('', xy=(vis[1], ymid), xytext=(vis[0], ymid),
                        arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.2))
            ax.text((vis[0]+vis[1])/2, ymid + 0.06,
                    f'ISI={isi_us:.1f} \u00b5s', ha='center', fontsize=7, color=GREEN)

        ax.set_xlabel(r'Time ($\mu$s)', fontsize=9)
        ax.set_ylabel('Voltage (V)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.9, edgecolor='#cccccc')
    else:
        ax.text(0.5, 0.5, 'Signal names not found\nin raw file', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='gray')
        print(f"  Available signals: {list(raw.keys())}")
else:
    ax.text(0.5, 0.5, 'Raw file not found', transform=ax.transAxes,
            ha='center', va='center', fontsize=9, color='gray')

ax.text(0.03, 0.95, r'$\mathbf{(b)}$', transform=ax.transAxes,
        fontsize=10, va='top', ha='left')

# ════════════════════════════════════════════════════════════
# (c) Boltzmann Parallel — dual y-axis
# ════════════════════════════════════════════════════════════
ax = axes[1, 0]
ax.plot(temps, bvpars, 'o-', color=BLUE, lw=1.5, ms=6, label=r'BV$_{\mathrm{par}}$(T)', zorder=3)
ax.set_xlabel('Temperature (K)', fontsize=9)
ax.set_ylabel(r'BV$_{\mathrm{par}}$ (V)', fontsize=9, color=BLUE)
ax.tick_params(axis='y', labelcolor=BLUE)

ax2 = ax.twinx()
fT = [1 + 0.3 * (t - 300) / 58 for t in temps]
ax2.plot(temps, fT, 's--', color=RED, lw=1.5, ms=6, label=r'$f(T_{\mathrm{die}})$', zorder=3)
ax2.set_ylabel(r'ThermalSoftmax $f(T_{\mathrm{die}})$', fontsize=9, color=RED)
ax2.tick_params(axis='y', labelcolor=RED)

# Annotations
ax.annotate(r'NS-RAM: BV$_{\mathrm{par}}$(T) $\downarrow$ $\rightarrow$ fires more easily',
            xy=(340, 2.70), fontsize=7, color=BLUE,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BLUE, alpha=0.85))
ax2.annotate(r'FEEL: $f(T_{\mathrm{die}})$ $\uparrow$ $\rightarrow$ softer attention',
             xy=(305, 1.27), fontsize=7, color=RED,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=RED, alpha=0.85))

ax.text(0.03, 0.95, r'$\mathbf{(c)}$', transform=ax.transAxes,
        fontsize=10, va='top', ha='left')

# ════════════════════════════════════════════════════════════
# (d) Shared Mathematical Structure
# ════════════════════════════════════════════════════════════
ax = axes[1, 1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Panel label
ax.text(0.03, 0.97, r'$\mathbf{(d)}$', fontsize=10, va='top', ha='left',
        transform=ax.transAxes)

# Title
ax.text(0.5, 0.93, 'Shared Mathematical Structure', fontsize=10,
        ha='center', va='top', fontweight='bold', transform=ax.transAxes)

# NS-RAM block
y0 = 0.78
ax.text(0.5, y0, r'$\mathbf{NS\text{-}RAM\ avalanche\ multiplication:}$',
        fontsize=9, ha='center', va='top', transform=ax.transAxes)
ax.text(0.5, y0 - 0.08,
        r'$M = \frac{1}{1 - \left(\frac{V_{cb}}{BV_{\mathrm{par}}(T)}\right)^{\!n}}$',
        fontsize=11, ha='center', va='top', transform=ax.transAxes)
ax.text(0.5, y0 - 0.22,
        r'$BV_{\mathrm{par}}(T) = 3.5 - 1.5 \cdot V_g(T)$',
        fontsize=9, ha='center', va='top', transform=ax.transAxes)

# Divider
ax.plot([0.15, 0.85], [0.48, 0.48], '-', color='#999999', lw=0.6,
        transform=ax.transAxes)

# FEEL block
y1 = 0.44
ax.text(0.5, y1, r'$\mathbf{FEEL\ ThermalSoftmax:}$',
        fontsize=9, ha='center', va='top', transform=ax.transAxes)
ax.text(0.5, y1 - 0.08,
        r'$\mathrm{Attn} = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}\;\cdot\;f(T_{\mathrm{die}})}\right)$',
        fontsize=11, ha='center', va='top', transform=ax.transAxes)

# Unifying principle box
ax.text(0.5, 0.12,
        r'Physical temperature $\rightarrow$ transfer function modulation',
        fontsize=9.5, ha='center', va='center', transform=ax.transAxes,
        color=GREEN, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', fc='#f0f8f0', ec=GREEN, lw=1.5))

# ── Save ──
os.makedirs(OUT_DIR, exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PATH}")
plt.close(fig)
