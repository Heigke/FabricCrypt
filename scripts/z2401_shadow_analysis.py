#!/usr/bin/env python3
"""
z2401_shadow_analysis.py — Shadow Neuromorphic Signal Analysis on Trained MNIST MLP

Loads trained MLP weights (784→128→64→10), runs inference on MNIST test set,
computes shadow signals (sign_agreement, timing_proxy, population) in numpy
to mirror what the HIP kernel harvests, then evaluates whether these shadow
signals are predictive of correctness.

Generates:
  1. ROC: shadow confidence vs correctness
  2. ROC: softmax confidence vs correctness (baseline)
  3. Calibration plot: binned accuracy vs confidence
  4. Per-digit breakdown: shadow signal by digit
  5. Scatter: softmax vs shadow confidence colored by correct/incorrect

Results saved to results/z2401_shadow_analysis/
"""

import os
import sys
import struct
import gzip
import numpy as np
import json
from pathlib import Path

# Paths
BASE = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
MODEL_DIR = BASE / "models" / "mnist_mlp"
MNIST_DIR = BASE / "data" / "MNIST" / "raw"
OUT_DIR = BASE / "results" / "z2401_shadow_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load MNIST
# ============================================================
def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        assert magic == 2051, f"Bad magic: {magic}"
        n = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
        return data.astype(np.float32) / 255.0

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        assert magic == 2049, f"Bad magic: {magic}"
        n = struct.unpack('>I', f.read(4))[0]
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)

# ============================================================
# Load trained weights (binary float32, row-major)
# ============================================================
def load_bin(path, n_floats):
    data = np.fromfile(str(path), dtype=np.float32)
    assert len(data) == n_floats, f"{path}: expected {n_floats}, got {len(data)}"
    return data

# Network dims: 784 → 128 → 64 → 10
INPUT_DIM = 784
HIDDEN1 = 128
HIDDEN2 = 64
OUTPUT_DIM = 10

print("=" * 60)
print("z2401: SHADOW NEUROMORPHIC ANALYSIS (Python/numpy)")
print("=" * 60)

# Load data
images = load_mnist_images(MNIST_DIR / "t10k-images-idx3-ubyte")
labels = load_mnist_labels(MNIST_DIR / "t10k-labels-idx1-ubyte")
n_test = len(labels)
print(f"Loaded MNIST test: {n_test} images, dim={images.shape[1]}")

# Load weights
w1 = load_bin(MODEL_DIR / "w1.bin", HIDDEN1 * INPUT_DIM).reshape(HIDDEN1, INPUT_DIM)
b1 = load_bin(MODEL_DIR / "b1.bin", HIDDEN1)
w2 = load_bin(MODEL_DIR / "w2.bin", HIDDEN2 * HIDDEN1).reshape(HIDDEN2, HIDDEN1)
b2 = load_bin(MODEL_DIR / "b2.bin", HIDDEN2)
w3 = load_bin(MODEL_DIR / "w3.bin", OUTPUT_DIM * HIDDEN2).reshape(OUTPUT_DIM, HIDDEN2)
b3 = load_bin(MODEL_DIR / "b3.bin", OUTPUT_DIM)
print(f"Loaded weights: w1={w1.shape}, w2={w2.shape}, w3={w3.shape}")

# ============================================================
# Shadow signal computation — mirrors HIP kernel logic
# ============================================================
def compute_shadow_layer(inp, weight, bias, apply_relu=True):
    """
    Compute matmul + shadow signals for one layer.
    inp: (batch, in_dim), weight: (out_dim, in_dim), bias: (out_dim,)

    Shadow signals per (sample, output_neuron):
      - sign_agreement: count of groups-of-4 where all products have same sign
      - timing_proxy: count of near-zero products (proxy for computational difficulty)
      - membrane: integrated shadow signal (same formula as HIP kernel)
      - spike_count: number of threshold crossings
    """
    batch_size, in_dim = inp.shape
    out_dim = weight.shape[0]

    # Traditional matmul
    output = inp @ weight.T + bias  # (batch, out_dim)

    # Shadow computation — iterate groups of 4 like the HIP kernel
    n_groups = in_dim // 4
    remainder = in_dim % 4

    # Reshape for group-of-4 processing
    # inp: (batch, n_groups, 4), weight: (out_dim, n_groups, 4)
    inp_g = inp[:, :n_groups * 4].reshape(batch_size, n_groups, 4)
    w_g = weight[:, :n_groups * 4].reshape(out_dim, n_groups, 4)

    # Products: (batch, out_dim, n_groups, 4)
    # This is the element-wise product for each (sample, neuron, group, element)
    products = inp_g[:, np.newaxis, :, :] * w_g[np.newaxis, :, :, :]

    # Sign agreement: all 4 same sign?
    positive = (products > 0).astype(np.int32)  # (batch, out_dim, n_groups, 4)
    pos_count = positive.sum(axis=3)  # (batch, out_dim, n_groups)
    same_sign = ((pos_count == 0) | (pos_count == 4)).astype(np.float32)
    sign_agreement = same_sign.sum(axis=2)  # (batch, out_dim) — count of groups

    # Timing proxy: near-zero products are "hard" (more pipeline stalls in HW)
    # A product near zero means the weight or input is near zero — ambiguous
    NEAR_ZERO_THRESH = 0.01
    near_zero = (np.abs(products) < NEAR_ZERO_THRESH).astype(np.float32)
    timing_proxy = near_zero.sum(axis=(2, 3))  # (batch, out_dim)

    # Membrane integration (mirrors HIP kernel formula)
    # For each group: signal = (4 - pos_count) * 0.25 + timing_spike * 0.5
    sign_disagree = (4 - pos_count).astype(np.float32) * 0.25  # (batch, out_dim, n_groups)

    # Simulate timing spikes: use product magnitude variance as proxy
    prod_sum = products.sum(axis=3)  # (batch, out_dim, n_groups)
    timing_spike = (np.abs(prod_sum) < 0.05).astype(np.float32) * 0.5

    signal_per_group = sign_disagree + timing_spike  # (batch, out_dim, n_groups)

    # Integrate membrane with resets (spike at threshold 5.0)
    membrane = np.zeros((batch_size, out_dim), dtype=np.float32)
    spike_count = np.zeros((batch_size, out_dim), dtype=np.int32)
    for g in range(n_groups):
        membrane += signal_per_group[:, :, g]
        spikes = (membrane > 5.0).astype(np.int32)
        spike_count += spikes
        membrane = np.where(spikes > 0, 0.0, membrane)

    # Population: count of neurons with positive activation (per sample)
    # (computed after relu for intermediate, or on raw logits for output)

    if apply_relu:
        output = np.maximum(0, output)

    return output, {
        'sign_agreement': sign_agreement,   # (batch, out_dim)
        'timing_proxy': timing_proxy,        # (batch, out_dim)
        'membrane': membrane,                # (batch, out_dim)
        'spike_count': spike_count,          # (batch, out_dim)
    }


# ============================================================
# Run inference with shadow signals
# ============================================================
print("\nRunning inference with shadow signal computation...")

h1, shadow1 = compute_shadow_layer(images, w1, b1, apply_relu=True)
h2, shadow2 = compute_shadow_layer(h1, w2, b2, apply_relu=True)
logits, shadow3 = compute_shadow_layer(h2, w3, b3, apply_relu=False)

# Predictions
preds = np.argmax(logits, axis=1)
correct = (preds == labels)
accuracy = correct.mean()
print(f"Model accuracy: {accuracy * 100:.2f}% ({correct.sum()}/{n_test})")

# ============================================================
# Compute confidence measures
# ============================================================

# 1. Softmax confidence (traditional)
logits_shifted = logits - logits.max(axis=1, keepdims=True)
exp_logits = np.exp(logits_shifted)
softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
softmax_conf = softmax.max(axis=1)  # max class probability

# 2. Shadow confidence: normalized sign agreement averaged across layers
# Higher sign agreement → computation was "cleaner" → more confident
# Normalize by max possible groups per layer
max_groups_l1 = INPUT_DIM // 4   # 196
max_groups_l2 = HIDDEN1 // 4     # 32
max_groups_l3 = HIDDEN2 // 4     # 16

# Per-sample average sign agreement across all neurons in each layer
sa1_avg = shadow1['sign_agreement'].mean(axis=1) / max_groups_l1  # (n_test,)
sa2_avg = shadow2['sign_agreement'].mean(axis=1) / max_groups_l2
sa3_avg = shadow3['sign_agreement'].mean(axis=1) / max_groups_l3

# Combine layers (weighted by depth — later layers more diagnostic)
shadow_conf = 0.2 * sa1_avg + 0.3 * sa2_avg + 0.5 * sa3_avg

# Also compute per-output-neuron shadow for the predicted class
# sign agreement of the winning neuron in output layer
pred_sa3 = shadow3['sign_agreement'][np.arange(n_test), preds] / max_groups_l3
shadow_conf_pred = 0.2 * sa1_avg + 0.3 * sa2_avg + 0.5 * pred_sa3

# 3. Spike-based confidence: total spikes across layers
spikes_total = (shadow1['spike_count'].sum(axis=1) +
                shadow2['spike_count'].sum(axis=1) +
                shadow3['spike_count'].sum(axis=1)).astype(np.float32)
# Normalize
spikes_max = spikes_total.max() if spikes_total.max() > 0 else 1.0
spike_conf = spikes_total / spikes_max

# 4. Population coding: fraction of neurons with positive activation per layer
pop1 = (h1 > 0).mean(axis=1)  # fraction of hidden1 neurons active
pop2 = (h2 > 0).mean(axis=1)
# For output: use fraction near the max (within 20% of max logit)
logit_range = logits.max(axis=1) - logits.min(axis=1)
pop3 = ((logits > (logits.max(axis=1, keepdims=True) - 0.2 * logit_range[:, np.newaxis]))).mean(axis=1)
# Lower population near max → more decisive → higher confidence
pop_conf = 1.0 - pop3  # inverted: fewer neurons near max = more confident

# Combined shadow: sign + spike + population
shadow_combined = 0.4 * shadow_conf + 0.3 * (1.0 - spike_conf) + 0.3 * pop_conf

# ============================================================
# Print summary statistics
# ============================================================
print(f"\n{'Signal':<25} {'Correct':>12} {'Incorrect':>12} {'Ratio':>8}")
print("-" * 60)
for name, vals in [
    ("Softmax confidence", softmax_conf),
    ("Shadow sign_agree", shadow_conf),
    ("Shadow pred_neuron_sa", shadow_conf_pred),
    ("Spike count (total)", spikes_total),
    ("Population conf", pop_conf),
    ("Shadow combined", shadow_combined),
]:
    c_mean = vals[correct].mean()
    i_mean = vals[~correct].mean()
    ratio = c_mean / i_mean if i_mean != 0 else float('inf')
    print(f"{name:<25} {c_mean:>12.4f} {i_mean:>12.4f} {ratio:>8.3f}")

# ============================================================
# ROC analysis
# ============================================================
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

confidence_methods = {
    'Softmax': softmax_conf,
    'Shadow (sign agree)': shadow_conf,
    'Shadow (pred neuron)': shadow_conf_pred,
    'Shadow (combined)': shadow_combined,
    'Spike count': 1.0 - spike_conf,  # invert: fewer spikes → more confident
    'Population': pop_conf,
}

print(f"\n{'Method':<25} {'AUC':>8}")
print("-" * 35)
aucs = {}
for name, conf in confidence_methods.items():
    try:
        auc = roc_auc_score(correct.astype(int), conf)
        aucs[name] = auc
        print(f"{name:<25} {auc:>8.4f}")
    except Exception as e:
        print(f"{name:<25} ERROR: {e}")
        aucs[name] = 0.5

# ============================================================
# Plot 1: ROC curves
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for name, conf in confidence_methods.items():
    fpr, tpr, _ = roc_curve(correct.astype(int), conf)
    ax.plot(fpr, tpr, label=f"{name} (AUC={aucs.get(name, 0.5):.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'z2401: Shadow vs Softmax Confidence ROC\n(MNIST MLP, acc={accuracy*100:.1f}%)')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / 'roc_comparison.png', dpi=150)
plt.close()
print(f"\nSaved: {OUT_DIR / 'roc_comparison.png'}")

# ============================================================
# Plot 2: Calibration plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (name, conf) in zip(axes, [('Softmax', softmax_conf), ('Shadow combined', shadow_combined)]):
    n_bins = 10
    bin_edges = np.linspace(conf.min() - 1e-8, conf.max() + 1e-8, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accs.append(correct[mask].mean())
            bin_counts.append(mask.sum())

    bin_centers = np.array(bin_centers)
    bin_accs = np.array(bin_accs)
    bin_counts = np.array(bin_counts)

    ax.bar(bin_centers, bin_accs, width=(bin_edges[1] - bin_edges[0]) * 0.8,
           alpha=0.6, color='steelblue', label='Accuracy')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel(f'{name} Confidence')
    ax.set_ylabel('Actual Accuracy')
    ax.set_title(f'{name} Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add count annotations
    for x, y, c in zip(bin_centers, bin_accs, bin_counts):
        ax.annotate(f'n={c}', (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=7)

plt.suptitle(f'z2401: Calibration — Shadow vs Softmax (acc={accuracy*100:.1f}%)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / 'calibration.png', dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'calibration.png'}")

# ============================================================
# Plot 3: Per-digit breakdown
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (name, vals) in zip(axes.flat, [
    ('Shadow sign_agree (avg)', shadow_conf),
    ('Softmax confidence', softmax_conf),
    ('Spike count (total)', spikes_total),
    ('Population conf', pop_conf),
]):
    digit_means_correct = []
    digit_means_incorrect = []
    digit_acc = []
    for d in range(10):
        mask_d = labels == d
        digit_acc.append(correct[mask_d].mean())
        if (mask_d & correct).sum() > 0:
            digit_means_correct.append(vals[mask_d & correct].mean())
        else:
            digit_means_correct.append(0)
        if (mask_d & ~correct).sum() > 0:
            digit_means_incorrect.append(vals[mask_d & ~correct].mean())
        else:
            digit_means_incorrect.append(0)

    x = np.arange(10)
    w = 0.35
    ax.bar(x - w/2, digit_means_correct, w, label='Correct', color='#2196F3', alpha=0.8)
    ax.bar(x + w/2, digit_means_incorrect, w, label='Incorrect', color='#F44336', alpha=0.8)
    ax.set_xlabel('Digit')
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Add accuracy on twin axis
    ax2 = ax.twinx()
    ax2.plot(x, digit_acc, 'k--o', markersize=4, alpha=0.5, label='Accuracy')
    ax2.set_ylabel('Accuracy', color='gray')
    ax2.set_ylim(0.8, 1.02)
    ax2.tick_params(axis='y', labelcolor='gray')

plt.suptitle(f'z2401: Per-Digit Shadow Signal Breakdown (acc={accuracy*100:.1f}%)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / 'per_digit_breakdown.png', dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'per_digit_breakdown.png'}")

# ============================================================
# Plot 4: Scatter — softmax vs shadow colored by correctness
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, sconf) in zip(axes, [
    ('Shadow sign agree', shadow_conf),
    ('Shadow combined', shadow_combined),
    ('Shadow pred neuron', shadow_conf_pred),
]):
    # Plot incorrect first (so correct is on top)
    ax.scatter(softmax_conf[~correct], sconf[~correct],
               c='red', alpha=0.15, s=3, label=f'Incorrect (n={int((~correct).sum())})')
    ax.scatter(softmax_conf[correct], sconf[correct],
               c='blue', alpha=0.05, s=3, label=f'Correct (n={int(correct.sum())})')
    ax.set_xlabel('Softmax Confidence')
    ax.set_ylabel(name)
    ax.set_title(f'Softmax vs {name}')
    ax.legend(fontsize=8, markerscale=5)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'z2401: Softmax vs Shadow Confidence (acc={accuracy*100:.1f}%)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / 'scatter_softmax_vs_shadow.png', dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'scatter_softmax_vs_shadow.png'}")

# ============================================================
# Plot 5: Shadow signal distributions
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, (name, vals) in zip(axes.flat, [
    ('Shadow sign agree', shadow_conf),
    ('Shadow combined', shadow_combined),
    ('Softmax', softmax_conf),
    ('Spike count', spikes_total),
    ('Population conf', pop_conf),
    ('Timing proxy (L3 avg)', shadow3['timing_proxy'].mean(axis=1)),
]):
    ax.hist(vals[correct], bins=50, alpha=0.6, density=True, label='Correct', color='blue')
    ax.hist(vals[~correct], bins=50, alpha=0.6, density=True, label='Incorrect', color='red')
    ax.set_xlabel(name)
    ax.set_ylabel('Density')
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'z2401: Signal Distributions — Correct vs Incorrect (acc={accuracy*100:.1f}%)', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / 'distributions.png', dpi=150)
plt.close()
print(f"Saved: {OUT_DIR / 'distributions.png'}")

# ============================================================
# Save numerical results
# ============================================================
results = {
    'experiment': 'z2401_shadow_analysis',
    'model': '784->128->64->10 MLP (trained)',
    'test_samples': int(n_test),
    'accuracy': float(accuracy),
    'n_correct': int(correct.sum()),
    'n_incorrect': int((~correct).sum()),
    'aucs': {k: float(v) for k, v in aucs.items()},
    'signal_stats': {},
}

for name, vals in [
    ("softmax_conf", softmax_conf),
    ("shadow_sign_agree", shadow_conf),
    ("shadow_pred_neuron", shadow_conf_pred),
    ("shadow_combined", shadow_combined),
    ("spike_total", spikes_total),
    ("pop_conf", pop_conf),
]:
    results['signal_stats'][name] = {
        'correct_mean': float(vals[correct].mean()),
        'correct_std': float(vals[correct].std()),
        'incorrect_mean': float(vals[~correct].mean()),
        'incorrect_std': float(vals[~correct].std()),
        'ratio': float(vals[correct].mean() / vals[~correct].mean()) if vals[~correct].mean() != 0 else None,
    }

# Per-digit accuracy
results['per_digit'] = {}
for d in range(10):
    mask = labels == d
    results['per_digit'][str(d)] = {
        'count': int(mask.sum()),
        'accuracy': float(correct[mask].mean()),
        'shadow_sign_correct': float(shadow_conf[mask & correct].mean()) if (mask & correct).sum() > 0 else None,
        'shadow_sign_incorrect': float(shadow_conf[mask & ~correct].mean()) if (mask & ~correct).sum() > 0 else None,
        'softmax_correct': float(softmax_conf[mask & correct].mean()) if (mask & correct).sum() > 0 else None,
        'softmax_incorrect': float(softmax_conf[mask & ~correct].mean()) if (mask & ~correct).sum() > 0 else None,
    }

with open(OUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUT_DIR / 'results.json'}")

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print(f"\nAUC comparison (higher = better at distinguishing correct/incorrect):")
for name, auc in sorted(aucs.items(), key=lambda x: -x[1]):
    marker = " <<<" if "Shadow" in name and auc > aucs.get('Softmax', 0) else ""
    print(f"  {name:<25} AUC = {auc:.4f}{marker}")

best_shadow = max((v, k) for k, v in aucs.items() if 'Shadow' in k or 'Spike' in k or 'Pop' in k)
print(f"\nBest shadow method: {best_shadow[1]} (AUC={best_shadow[0]:.4f})")
print(f"Softmax baseline:  AUC={aucs.get('Softmax', 0.5):.4f}")
gap = best_shadow[0] - aucs.get('Softmax', 0.5)
print(f"Gap: {gap:+.4f} ({'shadow better' if gap > 0 else 'softmax better'})")

# Key insight
print(f"\nKey finding:")
if best_shadow[0] > 0.55:
    print(f"  Shadow signals ARE predictive of correctness (AUC > 0.55)")
    print(f"  This validates the shadow neuromorphic confidence concept:")
    print(f"  computational side-effects carry useful information about prediction quality.")
else:
    print(f"  Shadow signals show WEAK predictivity (AUC near 0.5)")
    print(f"  The numpy simulation may miss real HW effects (timing, CU identity).")
    print(f"  Need to re-test with actual HIP kernel on GPU for hardware-level signals.")

print(f"\nPlots saved to: {OUT_DIR}")
