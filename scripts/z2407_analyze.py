#!/usr/bin/env python3
"""
z2407_analyze.py — ROC + complementarity analysis of REAL GPU shadow signals
Reads binary dump from z2407_shadow_dump.hip
Tests: does shadow + softmax beat softmax alone?
"""
import numpy as np
import struct
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'
dump_path = f'{base}/results/z2407_shadow_dump.bin'
out_dir = f'{base}/results/z2407_analysis'
os.makedirs(out_dir, exist_ok=True)

# Load binary dump
with open(dump_path, 'rb') as f:
    N, nc = struct.unpack('<ii', f.read(8))
    print(f"Loading {N} samples, {nc} classes")

    logits = np.zeros((N, nc), dtype=np.float32)
    shadow = np.zeros((N, nc, 6), dtype=np.float32)  # 6 shadow signals per class
    labels = np.zeros(N, dtype=np.int32)
    preds = np.zeros(N, dtype=np.int32)
    correct = np.zeros(N, dtype=np.int32)

    for i in range(N):
        # 10 logits
        logits[i] = np.frombuffer(f.read(nc*4), dtype=np.float32)
        # 10 shadow structs (6 floats each)
        for c in range(nc):
            shadow[i, c] = np.frombuffer(f.read(6*4), dtype=np.float32)
        # label, pred, correct
        meta = struct.unpack('<iii', f.read(12))
        labels[i], preds[i], correct[i] = meta

acc = correct.mean() * 100
print(f"Accuracy: {acc:.2f}% ({correct.sum()}/{N})")

# Compute confidence scores
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

probs = softmax(logits)
softmax_conf = probs.max(axis=1)  # max softmax prob
margin_conf = np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2]  # logit gap

# Shadow signals (average across output neurons per sample)
shadow_names = ['sign_ratio', 'timing_cv', 'neighbor_diff', 'pop_minority', 'spike_count', 'perf_entropy']
shadow_avg = shadow.mean(axis=1)  # [N, 6]

# Also compute shadow signals for the PREDICTED class only
shadow_pred = np.array([shadow[i, preds[i]] for i in range(N)])  # [N, 6]

# ROC computation
def compute_roc(scores, labels, n_thresholds=1000):
    """Compute ROC curve. Higher score = more likely positive (correct)."""
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    tpr_list, fpr_list = [], []
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0, 1], [0, 1], 0.5

    for t in thresholds:
        pred_pos = scores >= t
        tp = (pred_pos & (labels == 1)).sum()
        fp = (pred_pos & (labels == 0)).sum()
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Sort by FPR
    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    # AUC via trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_sorted)):
        auc += (fpr_sorted[i] - fpr_sorted[i-1]) * (tpr_sorted[i] + tpr_sorted[i-1]) / 2

    return fpr_sorted, tpr_sorted, auc

# Compute ROC for all signals
print("\n=== ROC AUC ===")
roc_results = {}

# Softmax
fpr, tpr, auc = compute_roc(softmax_conf, correct)
roc_results['Softmax confidence'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
print(f"  Softmax confidence:        AUC = {auc:.4f}")

# Margin
fpr, tpr, auc = compute_roc(margin_conf, correct)
roc_results['Margin (logit gap)'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
print(f"  Margin (logit gap):        AUC = {auc:.4f}")

# Individual shadow signals (averaged across classes)
for s in range(6):
    name = f"Shadow {shadow_names[s]} (avg)"
    fpr, tpr, auc = compute_roc(shadow_avg[:, s], correct)
    roc_results[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
    print(f"  {name:35s} AUC = {auc:.4f}")

# Shadow signals for predicted class only
for s in range(6):
    name = f"Shadow {shadow_names[s]} (pred)"
    fpr, tpr, auc = compute_roc(shadow_pred[:, s], correct)
    roc_results[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
    print(f"  {name:35s} AUC = {auc:.4f}")

# Combined shadow: normalize and sum best signals
best_shadow_indices = []
for s in range(6):
    if roc_results[f"Shadow {shadow_names[s]} (avg)"]['auc'] > 0.52:
        best_shadow_indices.append(s)

if best_shadow_indices:
    combined = np.zeros(N)
    for s in best_shadow_indices:
        sig = shadow_avg[:, s]
        sig_norm = (sig - sig.mean()) / (sig.std() + 1e-10)
        combined += sig_norm
    fpr, tpr, auc = compute_roc(combined, correct)
    roc_results['Shadow combined (best)'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
    print(f"  {'Shadow combined (best)':35s} AUC = {auc:.4f}")
    print(f"    (using: {[shadow_names[s] for s in best_shadow_indices]})")

# === COMPLEMENTARITY TEST ===
print("\n=== COMPLEMENTARITY: Softmax + Shadow ===")

# Normalize softmax_conf to [0,1]
sm_norm = (softmax_conf - softmax_conf.min()) / (softmax_conf.max() - softmax_conf.min() + 1e-10)

# Try different weights for combining softmax + shadow
best_combo_auc = 0
best_alpha = 0
for alpha in np.arange(0, 1.01, 0.05):
    combo = alpha * sm_norm
    for s in best_shadow_indices:
        sig = shadow_avg[:, s]
        sig_norm = (sig - sig.mean()) / (sig.std() + 1e-10)
        combo += (1 - alpha) / len(best_shadow_indices) * sig_norm
    _, _, auc = compute_roc(combo, correct)
    if auc > best_combo_auc:
        best_combo_auc = auc
        best_alpha = alpha

print(f"  Softmax alone AUC:          {roc_results['Softmax confidence']['auc']:.4f}")
print(f"  Best combo AUC:             {best_combo_auc:.4f} (alpha={best_alpha:.2f})")
print(f"  Improvement:                {best_combo_auc - roc_results['Softmax confidence']['auc']:.4f}")

if best_combo_auc > roc_results['Softmax confidence']['auc'] + 0.001:
    print(f"  >>> SHADOW ADDS COMPLEMENTARY INFORMATION! <<<")
else:
    print(f"  Shadow does not improve over softmax alone")

# Compute final combo ROC for plotting
combo_final = best_alpha * sm_norm
for s in best_shadow_indices:
    sig = shadow_avg[:, s]
    sig_norm = (sig - sig.mean()) / (sig.std() + 1e-10)
    combo_final += (1 - best_alpha) / len(best_shadow_indices) * sig_norm
fpr, tpr, auc = compute_roc(combo_final, correct)
roc_results['Softmax + Shadow'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}

# === REJECTION ANALYSIS ===
print("\n=== REJECTION ANALYSIS ===")
print(f"  {'Rate':>6s} {'Softmax':>10s} {'Shadow':>10s} {'Combo':>10s}")
for reject_pct in [1, 2, 5, 10, 15, 20, 30, 50]:
    n_reject = int(N * reject_pct / 100)
    n_keep = N - n_reject

    # Softmax rejection
    sm_order = np.argsort(softmax_conf)[::-1]  # highest confidence first
    sm_keep = sm_order[:n_keep]
    sm_acc = correct[sm_keep].mean() * 100

    # Shadow rejection (combined)
    if best_shadow_indices:
        sh_order = np.argsort(combined)[::-1]
        sh_keep = sh_order[:n_keep]
        sh_acc = correct[sh_keep].mean() * 100
    else:
        sh_acc = acc

    # Combo rejection
    co_order = np.argsort(combo_final)[::-1]
    co_keep = co_order[:n_keep]
    co_acc = correct[co_keep].mean() * 100

    print(f"  {reject_pct:5d}% {sm_acc:9.2f}% {sh_acc:9.2f}% {co_acc:9.2f}%")

# === PLOT ROC ===
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
colors = {'Softmax confidence': 'blue', 'Margin (logit gap)': 'orange',
          'Shadow combined (best)': 'red', 'Softmax + Shadow': 'green'}
for name, data in roc_results.items():
    if name in colors:
        ax.plot(data['fpr'], data['tpr'], color=colors[name],
                label=f"{name} (AUC={data['auc']:.3f})", linewidth=2)
ax.plot([0,1],[0,1],'--',color='gray',alpha=0.5, label='Random (0.5)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('z2407: ROC — Real GPU Shadow Signals vs Softmax')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{out_dir}/roc_hw_comparison.png', dpi=150)
plt.close()

# === PLOT REJECTION ===
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
reject_rates = [0, 1, 2, 5, 10, 15, 20, 30, 50]
sm_accs, sh_accs, co_accs = [], [], []
for r in reject_rates:
    n_reject = int(N * r / 100)
    n_keep = N - n_reject
    sm_order = np.argsort(softmax_conf)[::-1][:n_keep]
    sm_accs.append(correct[sm_order].mean() * 100)
    if best_shadow_indices:
        sh_order = np.argsort(combined)[::-1][:n_keep]
        sh_accs.append(correct[sh_order].mean() * 100)
    else:
        sh_accs.append(acc)
    co_order = np.argsort(combo_final)[::-1][:n_keep]
    co_accs.append(correct[co_order].mean() * 100)

ax.plot(reject_rates, sm_accs, 'b-o', label='Softmax', linewidth=2)
ax.plot(reject_rates, sh_accs, 'r-s', label='Shadow (HW)', linewidth=2)
ax.plot(reject_rates, co_accs, 'g-^', label='Softmax + Shadow', linewidth=2)
ax.axhline(y=acc, color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({acc:.1f}%)')
ax.set_xlabel('Rejection Rate (%)'); ax.set_ylabel('Accuracy on Remaining (%)')
ax.set_title('z2407: Selective Rejection — Real GPU Signals')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{out_dir}/rejection_hw.png', dpi=150)
plt.close()

# === PER-DIGIT ===
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for s, ax in enumerate(axes.flat):
    if s >= 6: break
    per_digit_corr = []
    per_digit_incorr = []
    for d in range(10):
        mask_d = labels == d
        mask_c = mask_d & (correct == 1)
        mask_i = mask_d & (correct == 0)
        per_digit_corr.append(shadow_avg[mask_c, s].mean() if mask_c.sum() > 0 else 0)
        per_digit_incorr.append(shadow_avg[mask_i, s].mean() if mask_i.sum() > 0 else 0)
    x = np.arange(10)
    ax.bar(x - 0.15, per_digit_corr, 0.3, label='Correct', color='green', alpha=0.7)
    ax.bar(x + 0.15, per_digit_incorr, 0.3, label='Incorrect', color='red', alpha=0.7)
    ax.set_title(shadow_names[s], fontsize=10)
    ax.set_xticks(x)
    if s == 0: ax.legend(fontsize=8)
plt.suptitle('z2407: Shadow Signals per Digit (Real GPU)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{out_dir}/per_digit_hw.png', dpi=150)
plt.close()

# Save JSON summary
summary = {
    'accuracy': float(acc),
    'n_samples': int(N),
    'softmax_auc': float(roc_results['Softmax confidence']['auc']),
    'margin_auc': float(roc_results['Margin (logit gap)']['auc']),
    'shadow_combined_auc': float(roc_results.get('Shadow combined (best)', {}).get('auc', 0)),
    'combo_auc': float(best_combo_auc),
    'combo_alpha': float(best_alpha),
    'improvement_over_softmax': float(best_combo_auc - roc_results['Softmax confidence']['auc']),
    'shadow_signals_used': [shadow_names[s] for s in best_shadow_indices],
    'per_signal_auc': {shadow_names[s]: float(roc_results[f"Shadow {shadow_names[s]} (avg)"]['auc']) for s in range(6)},
}
with open(f'{out_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nPlots saved to {out_dir}/")
print("Done.")
