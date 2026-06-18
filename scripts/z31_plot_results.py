#!/usr/bin/env python3
"""Generate plots for z31 causal proof results."""

import matplotlib.pyplot as plt
import numpy as np

# Data from causal proof validation
conditions = ['Stressed', 'Relaxed', 'Natural']

# Gate values
gate_means = [0.4481, 0.5798, 0.5054]
gate_stds = [0.0305, 0.0292, 0.0150]

# Skip rates
skip_rates = [22.8, 83.8, 53.0]

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Colors
colors = ['#e74c3c', '#27ae60', '#3498db']

# Plot 1: Gate Values
ax1 = axes[0]
bars1 = ax1.bar(conditions, gate_means, yerr=gate_stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold')
ax1.set_ylabel('Gate Value', fontsize=12)
ax1.set_title('SENSE→FEEL: Gate Response\n(p < 10⁻⁹³)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 0.8)
ax1.legend()

# Add significance markers
ax1.annotate('', xy=(0, 0.65), xytext=(1, 0.65),
            arrowprops=dict(arrowstyle='<->', color='black'))
ax1.text(0.5, 0.67, 'Δ=0.13***', ha='center', fontsize=10, fontweight='bold')

# Plot 2: Skip Rates
ax2 = axes[1]
bars2 = ax2.bar(conditions, skip_rates, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Skip Rate (%)', fontsize=12)
ax2.set_title('FEEL→REGULATE: Compute Routing\n(p < 10⁻⁷⁹)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 100)

# Add significance markers
ax2.annotate('', xy=(0, 90), xytext=(1, 90),
            arrowprops=dict(arrowstyle='<->', color='black'))
ax2.text(0.5, 93, 'Δ=61%***', ha='center', fontsize=10, fontweight='bold')

# Plot 3: The Full Loop
ax3 = axes[2]

# Create a circular flow diagram
circle_angles = np.linspace(0, 2*np.pi, 6)[:-1]  # 5 points
radius = 0.35
center = (0.5, 0.5)

labels = ['SENSE', 'FEEL', 'REGULATE', 'LATENT', 'EXPRESS']
check_marks = ['✓', '✓', '✓', '?', '✓']
label_colors = ['#27ae60', '#27ae60', '#27ae60', '#e67e22', '#27ae60']

for i, (angle, label, check, lcolor) in enumerate(zip(circle_angles, labels, check_marks, label_colors)):
    x = center[0] + radius * np.cos(angle - np.pi/2)
    y = center[1] + radius * np.sin(angle - np.pi/2)

    # Draw node
    circle = plt.Circle((x, y), 0.12, color=lcolor, alpha=0.3, transform=ax3.transAxes)
    ax3.add_patch(circle)
    ax3.text(x, y, f'{label}\n{check}', ha='center', va='center',
             fontsize=9, fontweight='bold', transform=ax3.transAxes)

    # Draw arrow to next node
    next_i = (i + 1) % 5
    next_angle = circle_angles[next_i]
    next_x = center[0] + radius * np.cos(next_angle - np.pi/2)
    next_y = center[1] + radius * np.sin(next_angle - np.pi/2)

    # Arrow from edge of current circle to edge of next
    dx = next_x - x
    dy = next_y - y
    dist = np.sqrt(dx**2 + dy**2)

    ax3.annotate('',
                xy=(next_x - 0.13*dx/dist, next_y - 0.13*dy/dist),
                xytext=(x + 0.13*dx/dist, y + 0.13*dy/dist),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2),
                transform=ax3.transAxes)

ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('Embodiment Loop Status\n(4/5 proven)', fontsize=12, fontweight='bold')

# Add text summary
ax3.text(0.5, 0.05, 'Word Overlap: 15.4% (outputs differ!)',
         ha='center', fontsize=10, transform=ax3.transAxes)

plt.tight_layout()
plt.savefig('results/z31_causal_proof.png', dpi=150, bbox_inches='tight')
plt.savefig('results/z31_causal_proof.svg', bbox_inches='tight')
print("Saved: results/z31_causal_proof.png")
print("Saved: results/z31_causal_proof.svg")

# Second figure: Per-layer analysis
fig2, ax = plt.subplots(figsize=(10, 5))

layers = [7, 11, 15, 19, 23]
layer_labels = ['L7\n(Early)', 'L11\n(Early)', 'L15\n(Mid)', 'L19\n(Mid)', 'L23\n(Late)']

# Data from validation
stressed_gates = [0.469, 0.555, 0.523, 0.498, 0.629]
relaxed_gates = [0.629, 0.695, 0.680, 0.633, 0.734]

x = np.arange(len(layers))
width = 0.35

bars1 = ax.bar(x - width/2, stressed_gates, width, label='Stressed', color='#e74c3c', edgecolor='black')
bars2 = ax.bar(x + width/2, relaxed_gates, width, label='Relaxed', color='#27ae60', edgecolor='black')

ax.set_ylabel('Gate Value', fontsize=12)
ax.set_xlabel('Layer', fontsize=12)
ax.set_title('Per-Layer Gate Response: Stressed vs Relaxed Sensors', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(layer_labels)
ax.legend()
ax.set_ylim(0, 0.9)

# Add gate_diff annotations
for i, (s, r) in enumerate(zip(stressed_gates, relaxed_gates)):
    diff = abs(r - s)
    ax.annotate(f'Δ={diff:.2f}', xy=(i, max(s, r) + 0.03), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/z31_per_layer_gates.png', dpi=150, bbox_inches='tight')
print("Saved: results/z31_per_layer_gates.png")

print("\nPlots generated successfully!")
