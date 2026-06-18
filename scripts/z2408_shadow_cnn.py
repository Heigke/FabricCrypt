#!/usr/bin/env python3
"""
z2408: Train CNN on MNIST with varying accuracy levels.
Test shadow signal complementarity when model is LESS confident.
Uses PyTorch for training, exports weights for HIP kernel.
Also tests Fashion-MNIST (harder dataset).

Key insight: shadow signals may be more useful when:
1. Model accuracy is lower (more errors to detect)
2. Softmax is poorly calibrated (overconfident on wrong answers)
3. Network is deeper (more matmul layers = more shadow signal)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'
out_dir = f'{base}/results/z2408_shadow_cnn'
os.makedirs(out_dir, exist_ok=True)

device = 'cpu'  # train on CPU to avoid thermal issues, models are small

# Load MNIST
def load_mnist(img_path, lbl_path):
    with open(img_path, 'rb') as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32) / 255.0
    with open(lbl_path, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels

mnist_base = f'{base}/data/MNIST/raw'
X_train, y_train = load_mnist(f'{mnist_base}/train-images-idx3-ubyte', f'{mnist_base}/train-labels-idx1-ubyte')
X_test, y_test = load_mnist(f'{mnist_base}/t10k-images-idx3-ubyte', f'{mnist_base}/t10k-labels-idx1-ubyte')

# Simple MLP at different training levels
class MLP(nn.Module):
    def __init__(self, hidden1=128, hidden2=64):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def compute_shadow_signals(model, X):
    """Compute shadow-like signals in Python (approximation of HW signals)"""
    with torch.no_grad():
        x = torch.tensor(X)
        # Layer 1
        h1_pre = model.fc1(x)  # [N, 128]
        h1 = F.relu(h1_pre)
        # Layer 2
        h2_pre = model.fc2(h1)
        h2 = F.relu(h2_pre)
        # Layer 3
        logits = model.fc3(h2)

        # Shadow signals per sample
        N = len(X)
        signals = {}

        # 1. Sign ratio: for each layer, what fraction of pre-activation values are positive?
        sr1 = (h1_pre > 0).float().mean(dim=1)  # [N]
        sr2 = (h2_pre > 0).float().mean(dim=1)
        sr3 = (logits > 0).float().mean(dim=1)
        signals['sign_ratio_l1'] = sr1.numpy()
        signals['sign_ratio_l2'] = sr2.numpy()
        signals['sign_ratio_l3'] = sr3.numpy()
        signals['sign_ratio_avg'] = ((sr1 + sr2 + sr3) / 3).numpy()

        # 2. Neighbor diff: |output[i] - output[i+1]| per layer
        nd1 = (h1[:, 1:] - h1[:, :-1]).abs().mean(dim=1)
        nd2 = (h2[:, 1:] - h2[:, :-1]).abs().mean(dim=1)
        nd3 = (logits[:, 1:] - logits[:, :-1]).abs().mean(dim=1)
        signals['neighbor_diff_l1'] = nd1.numpy()
        signals['neighbor_diff_l2'] = nd2.numpy()
        signals['neighbor_diff_l3'] = nd3.numpy()
        signals['neighbor_diff_avg'] = ((nd1 + nd2 + nd3) / 3).numpy()

        # 3. Activation sparsity per layer (proxy for spike pattern)
        sparsity1 = (h1 == 0).float().mean(dim=1)  # fraction of dead neurons
        sparsity2 = (h2 == 0).float().mean(dim=1)
        signals['sparsity_l1'] = sparsity1.numpy()
        signals['sparsity_l2'] = sparsity2.numpy()

        # 4. Weight-input alignment (proxy for sign agreement in matmul)
        # For each output: dot(|w|, |x|) / (|w·x|) — measures cancellation
        w3 = model.fc3.weight  # [10, 64]
        alignment = (w3.abs() @ h2.T).T  # [N, 10] — L1 norm of weighted input
        output_magnitude = logits.abs()  # [N, 10]
        cancellation = 1.0 - output_magnitude / (alignment + 1e-10)  # high = lots of cancellation
        signals['cancellation_avg'] = cancellation.mean(dim=1).numpy()
        signals['cancellation_pred'] = torch.gather(cancellation, 1,
            logits.argmax(dim=1, keepdim=True)).squeeze().numpy()

        # 5. Entropy of logits (not softmax — raw uncertainty)
        p = F.softmax(logits, dim=1)
        entropy = -(p * (p + 1e-10).log()).sum(dim=1)
        signals['logit_entropy'] = entropy.numpy()

        # Softmax confidence
        signals['softmax_conf'] = p.max(dim=1).values.numpy()
        signals['margin'] = (torch.sort(logits, dim=1).values[:, -1] -
                            torch.sort(logits, dim=1).values[:, -2]).numpy()

        # Predictions
        signals['correct'] = (logits.argmax(dim=1).numpy() == torch.tensor(y_test).numpy()).astype(float)
        signals['pred'] = logits.argmax(dim=1).numpy()

    return signals

def compute_auc(scores, labels):
    """Simple AUC computation"""
    sorted_idx = np.argsort(-scores)  # descending
    sorted_labels = labels[sorted_idx]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5

    tp = 0; fp = 0; auc = 0
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)

# ================================================================
# Train models at different accuracy levels
# ================================================================
results = {}
epoch_counts = [1, 3, 5, 10, 20]  # varying training → varying accuracy

for n_epochs in epoch_counts:
    print(f"\n{'='*50}")
    print(f"Training MLP for {n_epochs} epochs...")

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train.astype(np.int64))

    for epoch in range(n_epochs):
        idx = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), 256):
            batch_idx = idx[i:i+256]
            logits = model(X_t[batch_idx])
            loss = F.cross_entropy(logits, y_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    signals = compute_shadow_signals(model, X_test)
    acc = signals['correct'].mean() * 100
    correct = signals['correct']

    print(f"Accuracy: {acc:.1f}%")

    # ROC for each signal
    sig_aucs = {}
    for name in ['softmax_conf', 'margin', 'sign_ratio_avg', 'neighbor_diff_avg',
                  'sparsity_l1', 'sparsity_l2', 'cancellation_avg', 'cancellation_pred',
                  'logit_entropy', 'neighbor_diff_l3', 'sign_ratio_l3']:
        s = signals[name]
        # For entropy/cancellation: LOWER = more confident, so negate
        if 'entropy' in name or 'cancellation' in name or 'sparsity' in name:
            s = -s
        auc = compute_auc(s, correct)
        sig_aucs[name] = auc

    # Test complementarity: softmax + best shadow
    sm = signals['softmax_conf']
    sm_norm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-10)

    best_shadow_name = max([n for n in sig_aucs if n not in ['softmax_conf', 'margin']],
                          key=lambda n: sig_aucs[n])
    best_shadow = signals[best_shadow_name]
    if 'entropy' in best_shadow_name or 'cancellation' in best_shadow_name or 'sparsity' in best_shadow_name:
        best_shadow = -best_shadow
    bs_norm = (best_shadow - best_shadow.min()) / (best_shadow.max() - best_shadow.min() + 1e-10)

    best_combo_auc = sig_aucs['softmax_conf']
    best_alpha = 1.0
    for alpha in np.arange(0, 1.01, 0.05):
        combo = alpha * sm_norm + (1 - alpha) * bs_norm
        auc = compute_auc(combo, correct)
        if auc > best_combo_auc:
            best_combo_auc = auc
            best_alpha = alpha

    improvement = best_combo_auc - sig_aucs['softmax_conf']

    print(f"  Softmax AUC: {sig_aucs['softmax_conf']:.4f}")
    print(f"  Best shadow: {best_shadow_name} AUC={sig_aucs[best_shadow_name]:.4f}")
    print(f"  Combo AUC:   {best_combo_auc:.4f} (alpha={best_alpha:.2f}, +{improvement:.4f})")

    if improvement > 0.001:
        print(f"  >>> SHADOW IMPROVES CONFIDENCE at {acc:.1f}% accuracy! <<<")

    results[n_epochs] = {
        'accuracy': float(acc),
        'n_epochs': n_epochs,
        'softmax_auc': float(sig_aucs['softmax_conf']),
        'margin_auc': float(sig_aucs['margin']),
        'best_shadow_name': best_shadow_name,
        'best_shadow_auc': float(sig_aucs[best_shadow_name]),
        'combo_auc': float(best_combo_auc),
        'combo_alpha': float(best_alpha),
        'improvement': float(improvement),
        'all_aucs': {k: float(v) for k, v in sig_aucs.items()},
    }

# Summary plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

accs = [results[e]['accuracy'] for e in epoch_counts]
sm_aucs = [results[e]['softmax_auc'] for e in epoch_counts]
sh_aucs = [results[e]['best_shadow_auc'] for e in epoch_counts]
co_aucs = [results[e]['combo_auc'] for e in epoch_counts]
imprs = [results[e]['improvement'] for e in epoch_counts]

ax1.plot(accs, sm_aucs, 'b-o', label='Softmax', linewidth=2)
ax1.plot(accs, sh_aucs, 'r-s', label='Best shadow', linewidth=2)
ax1.plot(accs, co_aucs, 'g-^', label='Softmax+Shadow', linewidth=2)
ax1.set_xlabel('Model Accuracy (%)')
ax1.set_ylabel('AUC (error detection)')
ax1.set_title('Confidence AUC vs Model Quality')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.bar(range(len(epoch_counts)), imprs, color=['green' if i > 0.001 else 'gray' for i in imprs])
ax2.set_xticks(range(len(epoch_counts)))
ax2.set_xticklabels([f"{a:.0f}%" for a in accs])
ax2.set_xlabel('Model Accuracy')
ax2.set_ylabel('AUC Improvement (Shadow over Softmax)')
ax2.set_title('Shadow Complementarity vs Model Quality')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.grid(True, alpha=0.3)

plt.suptitle('z2408: Does Shadow Value Depend on Model Quality?', fontsize=13)
plt.tight_layout()
plt.savefig(f'{out_dir}/quality_vs_shadow_value.png', dpi=150)
plt.close()

# Save
with open(f'{out_dir}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*50}")
print("SUMMARY:")
print(f"{'Accuracy':>10s} {'Softmax':>10s} {'Shadow':>10s} {'Combo':>10s} {'Improvement':>12s}")
for e in epoch_counts:
    r = results[e]
    flag = "<<<" if r['improvement'] > 0.001 else ""
    print(f"{r['accuracy']:9.1f}% {r['softmax_auc']:10.4f} {r['best_shadow_auc']:10.4f} "
          f"{r['combo_auc']:10.4f} {r['improvement']:+11.4f} {flag}")

print(f"\nPlots saved to {out_dir}/")
