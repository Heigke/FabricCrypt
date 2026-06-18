#!/usr/bin/env python3
"""
Generate LinkedIn thumbnail for FEEL security depth post.
Dark tech aesthetic with layered depth visualization.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(12, 6.3), dpi=200)  # LinkedIn 1200x630

# Dark background
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.3)
ax.axis('off')

# === LAYER BARS (left side) — depth visualization ===
layer_data = [
    (4, 'HIP API',           '#3a7bd5', 'STANDARD',    True),
    (3, 'ISA: s_setreg',     '#00d4aa', 'WRITABLE',    True),
    (2, 'MMIO ~100 MHz',     '#f5a623', 'OBSERVABLE',  True),
    (1, 'SMN Thermal/Power', '#e74c3c', 'ANALOG',      True),
    (0, 'PSP Firmware',      '#333344', 'LOCKED',      False),
]

for i, (layer, name, color, tag, accessible) in enumerate(layer_data):
    y = 5.2 - i * 1.05
    width = 4.8 if accessible else 4.8
    alpha = 0.9 if accessible else 0.3

    # Bar
    rect = patches.FancyBboxPatch(
        (0.4, y - 0.35), width, 0.7,
        boxstyle="round,pad=0.08",
        facecolor=color, alpha=alpha,
        edgecolor='white' if accessible else '#444455',
        linewidth=1.5 if accessible else 0.5
    )
    ax.add_patch(rect)

    # Layer number
    ax.text(0.75, y, str(layer), fontsize=14, fontweight='bold',
            color='white' if accessible else '#666677',
            ha='center', va='center', fontfamily='monospace')

    # Name
    ax.text(1.15, y + 0.05, name, fontsize=11,
            color='white' if accessible else '#555566',
            ha='left', va='center', fontweight='bold')

    # Tag
    tag_color = color if accessible else '#444455'
    ax.text(5.0, y, tag, fontsize=8,
            color=tag_color, ha='right', va='center',
            fontweight='bold', fontfamily='monospace',
            alpha=0.8)

    # Lock/unlock icon
    if not accessible:
        ax.text(5.35, y, 'X', fontsize=13, ha='center', va='center',
                color='#e74c3c', fontweight='bold', fontfamily='monospace')
    else:
        ax.text(5.35, y, '✓', fontsize=13, ha='center', va='center',
                color=color, fontweight='bold')

# === Depth arrow (left edge) ===
ax.annotate('', xy=(0.15, 0.5), xytext=(0.15, 5.6),
            arrowprops=dict(arrowstyle='->', color='#ffffff', lw=2))
ax.text(0.15, 3.0, 'D\nE\nP\nT\nH', fontsize=7, color='#666677',
        ha='center', va='center', fontfamily='monospace', linespacing=1.5)

# === RIGHT SIDE — Key findings ===
# Title
ax.text(8.5, 5.5, 'HOW DEEP CAN', fontsize=22, fontweight='bold',
        color='white', ha='center', va='center', fontfamily='sans-serif')
ax.text(8.5, 4.85, 'YOU GO?', fontsize=28, fontweight='bold',
        color='#00d4aa', ha='center', va='center', fontfamily='sans-serif')

# Divider line
ax.plot([6.2, 6.2], [0.5, 5.8], color='#333344', lw=1, alpha=0.5)

# Stats boxes
stats = [
    ('18', 'attack vectors\ntested', '#e74c3c'),
    ('5', 'protection layers\nmapped', '#f5a623'),
    ('4', 'usable layers\nfor neuromorphic', '#00d4aa'),
    ('110k', 'words firmware\nwritten to TMR', '#3a7bd5'),
]

for i, (num, desc, color) in enumerate(stats):
    x = 7.0 + (i % 2) * 2.8
    y = 3.5 - (i // 2) * 1.6

    ax.text(x, y + 0.25, num, fontsize=24, fontweight='bold',
            color=color, ha='center', va='center', fontfamily='monospace')
    ax.text(x, y - 0.35, desc, fontsize=7.5,
            color='#999999', ha='center', va='center',
            linespacing=1.3)

# Bottom tagline
ax.text(8.5, 0.55, 'First public RDNA4 PSP security characterisation',
        fontsize=9, color='#666677', ha='center', va='center',
        fontstyle='italic')

# Subtle scan lines effect
for y_line in np.arange(0, 6.3, 0.15):
    ax.axhline(y=y_line, color='white', alpha=0.008, lw=0.5)

# Subtle grid dots on right side
for gx in np.arange(6.5, 11.5, 0.4):
    for gy in np.arange(0.3, 6.0, 0.4):
        ax.plot(gx, gy, '.', color='white', alpha=0.02, markersize=1)

plt.tight_layout(pad=0.1)
out = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/linkedin_depth_thumb.png'
plt.savefig(out, dpi=200, facecolor='#0a0a0f',
            bbox_inches='tight', pad_inches=0.05)
plt.close()
print(f"Saved: {out}")

# Also make a second variant — more dramatic
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6.3), dpi=200)
fig2.patch.set_facecolor('#0a0a0f')
ax2.set_facecolor('#0a0a0f')
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 6.3)
ax2.axis('off')

# Big dramatic text
ax2.text(6, 4.8, 'I PROBED 5 LAYERS', fontsize=30, fontweight='bold',
         color='white', ha='center', va='center')
ax2.text(6, 3.9, 'OF GPU SILICON', fontsize=30, fontweight='bold',
         color='#00d4aa', ha='center', va='center')

# Separator
ax2.plot([1.5, 10.5], [3.2, 3.2], color='#333344', lw=1)

# Three columns of findings
cols = [
    ('s_setreg', 'Writable from\nshader ISA', '#00d4aa'),
    ('MMIO @100MHz', 'GPU internal state\nvisible in real-time', '#f5a623'),
    ('TMR breached', '110k words written\n(encrypted, no exec)', '#e74c3c'),
]

for i, (title, desc, color) in enumerate(cols):
    x = 2.5 + i * 3.5
    ax2.text(x, 2.5, title, fontsize=14, fontweight='bold',
             color=color, ha='center', va='center', fontfamily='monospace')
    ax2.text(x, 1.7, desc, fontsize=9, color='#888888',
             ha='center', va='center', linespacing=1.4)

# Bottom
ax2.text(6, 0.6, 'Commodity GPUs have 4 usable neuromorphic layers below the API',
         fontsize=11, color='#666677', ha='center', va='center', fontstyle='italic')

# Scan lines
for y_line in np.arange(0, 6.3, 0.15):
    ax2.axhline(y=y_line, color='white', alpha=0.008, lw=0.5)

plt.tight_layout(pad=0.1)
out2 = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/linkedin_depth_thumb_v2.png'
plt.savefig(out2, dpi=200, facecolor='#0a0a0f',
            bbox_inches='tight', pad_inches=0.05)
plt.close()
print(f"Saved: {out2}")
