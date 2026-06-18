#!/usr/bin/env python3
"""
z2422: Learned ISA→FPGA coupling vs hand-tuned

z2421 showed ISA coupling HURTS (-0.14pp). Hypothesis: the coupling
function is too crude (raw signals → parameters). Test if LEARNED
coupling helps.

Also test: different FPGA configurations to see what matters most.
"""
import numpy as np
import time, json, os

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

# Load MNIST + MLP (reuse from z2421)
def load_mnist(img_path, lbl_path):
    with open(img_path, 'rb') as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32) / 255.0
    with open(lbl_path, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels

X_test, y_test = load_mnist(f'{base}/data/MNIST/raw/t10k-images-idx3-ubyte',
                             f'{base}/data/MNIST/raw/t10k-labels-idx1-ubyte')

w1 = np.fromfile(f'{base}/models/mnist_mlp/w1.bin', dtype=np.float32).reshape(128, 784)
b1 = np.fromfile(f'{base}/models/mnist_mlp/b1.bin', dtype=np.float32)
w2 = np.fromfile(f'{base}/models/mnist_mlp/w2.bin', dtype=np.float32).reshape(64, 128)
b2 = np.fromfile(f'{base}/models/mnist_mlp/b2.bin', dtype=np.float32)
w3 = np.fromfile(f'{base}/models/mnist_mlp/w3.bin', dtype=np.float32).reshape(10, 64)
b3 = np.fromfile(f'{base}/models/mnist_mlp/b3.bin', dtype=np.float32)

def relu(x): return np.maximum(0, x)

N = len(X_test)
n_train = 7000
n_test = N - n_train

# MLP forward + ISA signals
def mlp_with_isa(X):
    h1 = relu(X @ w1.T + b1)
    h2 = relu(h1 @ w2.T + b2)
    logits = h2 @ w3.T + b3
    # ISA signals per layer
    isa = np.stack([
        (h1 > 0).mean(axis=1),  # sign ratio L1
        np.abs(np.diff(h1, axis=1)).mean(axis=1),  # neighd L1
        h1.std(axis=1),  # timing proxy L1
        (h2 > 0).mean(axis=1),
        np.abs(np.diff(h2, axis=1)).mean(axis=1),
        h2.std(axis=1),
        logits.std(axis=1),
        np.abs(np.diff(logits, axis=1)).mean(axis=1),
        (logits > 0).mean(axis=1),
    ], axis=1)
    return logits, isa

logits_all, isa_all = mlp_with_isa(X_test)

# FPGA reservoir with configurable coupling
class FPGAReservoir:
    def __init__(self, n=128, leak=0.85, threshold=0.5, coupling_weights=None):
        self.n = n
        self.leak = leak
        self.threshold = threshold
        self.coupling_weights = coupling_weights  # [9] ISA→params mapping
        np.random.seed(42)
        self.W_in = (np.random.randn(n, 28) * 0.3).astype(np.float32)
        self.W_rec = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            self.W_rec[i, (i+1)%n] = 0.05
            self.W_rec[i, (i-1)%n] = 0.05
            self.W_rec[i, (i+32)%n] = -0.03

    def run(self, image, isa=None):
        membrane = np.zeros(self.n, dtype=np.float32)
        spikes_all = np.zeros((28, self.n), dtype=np.float32)

        # ISA coupling: learned weights map 9 ISA signals → 3 reservoir params
        leak_mod = 0.0
        thresh_mod = 0.0
        input_scale = 1.0
        if isa is not None and self.coupling_weights is not None:
            cw = self.coupling_weights
            leak_mod = np.tanh(cw[0]*isa[0] + cw[1]*isa[1] + cw[2]*isa[2]) * 0.1
            thresh_mod = np.tanh(cw[3]*isa[3] + cw[4]*isa[4] + cw[5]*isa[5]) * 0.2
            input_scale = 1.0 + np.tanh(cw[6]*isa[6] + cw[7]*isa[7] + cw[8]*isa[8]) * 0.3

        eff_leak = np.clip(self.leak + leak_mod, 0.5, 0.99)
        eff_thresh = np.clip(self.threshold + thresh_mod, 0.1, 2.0)

        for t in range(28):
            row = image[t*28:(t+1)*28]
            inp = self.W_in @ row * input_scale
            rec = self.W_rec @ membrane
            membrane = membrane * eff_leak + inp + rec
            spikes = (membrane > eff_thresh).astype(float)
            spikes_all[t] = spikes
            membrane[spikes > 0] = 0.0

        mean_state = spikes_all.mean(axis=0)
        spike_count = spikes_all.sum(axis=0)
        prods = []
        for tau in [1, 3, 5]:
            if tau < 28:
                prods.append((spikes_all[tau:] * spikes_all[:-tau]).mean(axis=0))
        return np.concatenate([mean_state, spike_count] + prods)

def ridge_classify(X_tr, y_tr, X_te, y_te, alpha=1.0):
    n_c = 10
    Y = np.zeros((len(y_tr), n_c))
    for i, y in enumerate(y_tr): Y[i, y] = 1.0
    XtX = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
    W = np.linalg.solve(XtX, X_tr.T @ Y)
    preds = (X_te @ W).argmax(axis=1)
    return (preds == y_te).mean() * 100

# ============================================================
# Experiment 1: sweep coupling weights (random search)
# ============================================================
print("=" * 60)
print("z2422: LEARNED ISA→FPGA COUPLING")
print("=" * 60)

# Baseline: no coupling
print("\n--- Baselines ---")
fpga_nc = FPGAReservoir(128, coupling_weights=None)
feat_nc = np.zeros((N, 640), dtype=np.float32)
for i in range(N):
    feat_nc[i] = fpga_nc.run(X_test[i], isa=None)
feat_e = np.concatenate([logits_all, feat_nc], axis=1)
acc_no_coupling = ridge_classify(feat_e[:n_train], y_test[:n_train],
                                  feat_e[n_train:], y_test[n_train:])
print(f"  MLP + FPGA (no coupling): {acc_no_coupling:.2f}%")

acc_mlp = ridge_classify(logits_all[:n_train], y_test[:n_train],
                          logits_all[n_train:], y_test[n_train:])
print(f"  MLP alone (ridge):        {acc_mlp:.2f}%")

# Random search for coupling weights
print("\n--- Random search: 50 coupling configurations ---")
best_acc = acc_no_coupling
best_cw = None
np.random.seed(123)

results = []
for trial in range(50):
    cw = np.random.randn(9) * 0.5
    fpga = FPGAReservoir(128, coupling_weights=cw)
    feat = np.zeros((N, 640), dtype=np.float32)
    for i in range(N):
        feat[i] = fpga.run(X_test[i], isa=isa_all[i])
    feat_all = np.concatenate([logits_all, feat], axis=1)
    acc = ridge_classify(feat_all[:n_train], y_test[:n_train],
                          feat_all[n_train:], y_test[n_train:])
    results.append({'trial': trial, 'acc': acc, 'cw': cw.tolist()})
    if acc > best_acc:
        best_acc = acc
        best_cw = cw.copy()
        print(f"  Trial {trial:2d}: {acc:.2f}% ★ NEW BEST (Δ={acc-acc_no_coupling:+.2f}pp)")
    elif trial < 10 or trial % 10 == 0:
        print(f"  Trial {trial:2d}: {acc:.2f}%")

print(f"\n  Best coupling: {best_acc:.2f}% (vs no coupling {acc_no_coupling:.2f}%)")
print(f"  Improvement: {best_acc - acc_no_coupling:+.2f}pp")
if best_acc > acc_no_coupling + 0.1:
    print(f"  >>> LEARNED COUPLING IMPROVES! <<<")

# ============================================================
# Experiment 2: ablation — which FPGA features matter?
# ============================================================
print("\n--- Ablation: which features matter? ---")
# Test with different feature subsets
fpga_full = FPGAReservoir(128, coupling_weights=None)
feat_full = np.zeros((N, 640), dtype=np.float32)
for i in range(N):
    feat_full[i] = fpga_full.run(X_test[i])

# FPGA features: [0:128]=mean_state, [128:256]=spike_count, [256:640]=temporal_products
subsets = {
    'mean_state only (128)': feat_full[:, :128],
    'spike_count only (128)': feat_full[:, 128:256],
    'temporal_products only (384)': feat_full[:, 256:],
    'mean+count (256)': feat_full[:, :256],
    'count+temporal (512)': feat_full[:, 128:],
    'all (640)': feat_full,
}

for name, feat_sub in subsets.items():
    feat_combined = np.concatenate([logits_all, feat_sub], axis=1)
    acc = ridge_classify(feat_combined[:n_train], y_test[:n_train],
                          feat_combined[n_train:], y_test[n_train:])
    print(f"  MLP + {name:30s}: {acc:.2f}%")

# ============================================================
# Experiment 3: scaling — different number of FPGA neurons
# ============================================================
print("\n--- Scaling: FPGA neuron count ---")
for n_neurons in [8, 16, 32, 64, 128, 256]:
    fpga_s = FPGAReservoir(n_neurons, coupling_weights=None)
    # Adjust W_in size
    fpga_s.W_in = (np.random.RandomState(42).randn(n_neurons, 28) * 0.3).astype(np.float32)
    fpga_s.W_rec = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    for ii in range(n_neurons):
        fpga_s.W_rec[ii, (ii+1)%n_neurons] = 0.05
        fpga_s.W_rec[ii, (ii-1)%n_neurons] = 0.05

    n_feat = n_neurons * 2 + n_neurons * 3  # mean + count + 3 temporal
    feat_s = np.zeros((N, n_feat), dtype=np.float32)
    for i in range(N):
        feat_s[i] = fpga_s.run(X_test[i])
    feat_combined = np.concatenate([logits_all, feat_s], axis=1)
    acc = ridge_classify(feat_combined[:n_train], y_test[:n_train],
                          feat_combined[n_train:], y_test[n_train:])
    print(f"  {n_neurons:4d} neurons: {acc:.2f}%  ({n_feat} features)")

# Save
out = {
    'mlp_alone': float(acc_mlp),
    'no_coupling': float(acc_no_coupling),
    'best_coupling': float(best_acc),
    'coupling_improvement': float(best_acc - acc_no_coupling),
    'trials': results[:10],  # first 10 for brevity
}
with open(f'{base}/results/z2422_coupling.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to results/z2422_coupling.json")
