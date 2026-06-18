#!/usr/bin/env python3
"""
z2421: Integrated pipeline — GPU traditional + ISA harvest + FPGA reservoir

Architecture:
  1. GPU runs trained MLP inference (traditional matmul)
  2. DURING matmul, ISA mechanisms harvest side-effect signals (timing, shfl)
  3. Side-effect signals feed into FPGA reservoir (128 LIF neurons)
  4. FPGA reservoir output is concatenated with MLP logits
  5. Combined readout makes final prediction

The FPGA reservoir adds TEMPORAL processing that the MLP doesn't have.
Each MNIST image is processed over T=50 timesteps (scanning rows).
MLP sees the whole image at once. FPGA sees it row-by-row with temporal dynamics.

Test:
  A: MLP alone (94.3%)
  B: MLP + simulated ISA signals (timing_cv, neighbor_diff per layer)
  C: MLP + ISA + FPGA reservoir (128 LIF neurons with ISA-signal input)
  D: FPGA reservoir alone (no MLP)

If C > A: integration works — the combination is stronger.
"""
import numpy as np
import struct
import os
import time
import json

base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

# Load MNIST
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

# Load trained MLP weights
def load_bin(path, n):
    return np.fromfile(path, dtype=np.float32, count=n)

w1 = load_bin(f'{base}/models/mnist_mlp/w1.bin', 128*784).reshape(128, 784)
b1 = load_bin(f'{base}/models/mnist_mlp/b1.bin', 128)
w2 = load_bin(f'{base}/models/mnist_mlp/w2.bin', 64*128).reshape(64, 128)
b2 = load_bin(f'{base}/models/mnist_mlp/b2.bin', 64)
w3 = load_bin(f'{base}/models/mnist_mlp/w3.bin', 10*64).reshape(10, 64)
b3 = load_bin(f'{base}/models/mnist_mlp/b3.bin', 10)

def relu(x): return np.maximum(0, x)
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# ============================================================
# MLP forward pass WITH ISA signal simulation
# ============================================================
def mlp_forward_with_isa(X):
    """MLP inference + simulated ISA harvest signals per layer."""
    N = len(X)

    # Layer 1
    h1_pre = X @ w1.T + b1
    h1 = relu(h1_pre)

    # ISA signals from layer 1 (simulate what HIP kernel measures)
    # Sign ratio: fraction of positive products per neuron group of 4
    sign_ratios_l1 = (h1_pre > 0).astype(float).reshape(N, -1, 4).mean(axis=2).mean(axis=1)
    # Neighbor diff: |h1[i] - h1[i+1]| averaged
    neighd_l1 = np.abs(np.diff(h1, axis=1)).mean(axis=1)
    # Timing proxy: std of activation magnitudes (proxy for pipeline variation)
    timing_l1 = h1.std(axis=1)

    # Layer 2
    h2_pre = h1 @ w2.T + b2
    h2 = relu(h2_pre)
    sign_ratios_l2 = (h2_pre > 0).astype(float).reshape(N, -1, 4).mean(axis=2).mean(axis=1)
    neighd_l2 = np.abs(np.diff(h2, axis=1)).mean(axis=1)
    timing_l2 = h2.std(axis=1)

    # Layer 3
    logits = h2 @ w3.T + b3
    sign_ratios_l3 = (logits > 0).astype(float).mean(axis=1)
    neighd_l3 = np.abs(np.diff(logits, axis=1)).mean(axis=1)
    timing_l3 = logits.std(axis=1)

    # ISA features: 9 values per sample (3 per layer × 3 layers)
    isa_features = np.stack([
        sign_ratios_l1, neighd_l1, timing_l1,
        sign_ratios_l2, neighd_l2, timing_l2,
        sign_ratios_l3, neighd_l3, timing_l3
    ], axis=1)

    return logits, isa_features, h1, h2

# ============================================================
# FPGA NS-RAM reservoir (simulated, same model as real FPGA)
# ============================================================
class FPGAReservoir:
    """Simulated 128-neuron LIF reservoir matching real FPGA behavior."""
    def __init__(self, n_neurons=128, leak=0.85, threshold=0.5):
        self.n = n_neurons
        self.leak = leak
        self.threshold = threshold
        # Random input weights
        np.random.seed(42)
        self.W_in = (np.random.randn(n_neurons, 28) * 0.3).astype(np.float32)  # 28 = image row width
        # Recurrent weights (sparse, small-world)
        self.W_rec = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        for i in range(n_neurons):
            self.W_rec[i, (i+1) % n_neurons] = 0.05  # nearest neighbor exc
            self.W_rec[i, (i-1) % n_neurons] = 0.05
            self.W_rec[i, (i+32) % n_neurons] = -0.03  # long-range inh
            self.W_rec[i, (i+64) % n_neurons] = -0.02

    def run(self, image, isa_signals=None):
        """Process one 28×28 image row-by-row (28 timesteps)."""
        membrane = np.zeros(self.n, dtype=np.float32)
        spike_history = np.zeros((28, self.n), dtype=np.float32)

        for t in range(28):
            # Input: one row of image
            row = image[t*28:(t+1)*28]
            inp = self.W_in @ row

            # ISA modulation: if ISA signals provided, use them
            # to modulate the reservoir (cross-substrate coupling)
            if isa_signals is not None:
                # ISA timing signal modulates leak rate
                timing_mod = isa_signals[2] * 0.1  # timing_l1 scaled
                # ISA neighbor diff modulates threshold
                neighd_mod = isa_signals[1] * 0.05
            else:
                timing_mod = 0
                neighd_mod = 0

            # Leaky integration
            rec = self.W_rec @ membrane
            effective_leak = self.leak + timing_mod
            effective_leak = np.clip(effective_leak, 0.5, 0.99)
            membrane = membrane * effective_leak + inp + rec

            # Spike
            spikes = (membrane > (self.threshold + neighd_mod)).astype(float)
            spike_history[t] = spikes
            membrane[spikes > 0] = 0.0  # reset

        # Readout features: mean membrane + spike count + temporal products
        mean_state = spike_history.mean(axis=0)  # [128]
        spike_count = spike_history.sum(axis=0)   # [128]

        # Temporal product features (the breakthrough from z2296)
        prod_features = []
        for tau in [1, 3, 5]:
            if tau < 28:
                prod = (spike_history[tau:] * spike_history[:-tau]).mean(axis=0)
                prod_features.append(prod)
        prod_features = np.concatenate(prod_features) if prod_features else np.array([])

        return np.concatenate([mean_state, spike_count, prod_features])

# ============================================================
# Run experiments
# ============================================================
print("=" * 60)
print("z2421: INTEGRATED PIPELINE")
print("GPU traditional + ISA harvest + FPGA reservoir")
print("=" * 60)

N = len(X_test)
n_train = 7000
n_test = N - n_train

# A: MLP alone
print("\n--- A: MLP alone ---")
logits_all, isa_all, _, _ = mlp_forward_with_isa(X_test)
preds_mlp = logits_all.argmax(axis=1)
acc_mlp = (preds_mlp == y_test).mean() * 100
print(f"  Accuracy: {acc_mlp:.2f}%")

# B: MLP + ISA features (concatenated readout)
print("\n--- B: MLP + ISA features ---")
features_b = np.concatenate([logits_all, isa_all], axis=1)  # 10 + 9 = 19 features

# Ridge regression on combined features
from numpy.linalg import lstsq
def ridge_classify(X_train, y_train, X_test, y_test_true, alpha=1.0):
    n_classes = 10
    n_feat = X_train.shape[1]
    # One-hot targets
    Y = np.zeros((len(y_train), n_classes))
    for i, y in enumerate(y_train):
        Y[i, y] = 1.0
    # Ridge: W = (XtX + αI)^-1 XtY
    XtX = X_train.T @ X_train + alpha * np.eye(n_feat)
    XtY = X_train.T @ Y
    W = np.linalg.solve(XtX, XtY)
    # Predict
    scores = X_test @ W
    preds = scores.argmax(axis=1)
    acc = (preds == y_test_true).mean() * 100
    return acc, preds

acc_b, _ = ridge_classify(features_b[:n_train], y_test[:n_train],
                          features_b[n_train:], y_test[n_train:])
print(f"  Accuracy: {acc_b:.2f}% (MLP logits + ISA signals → ridge)")

# C: MLP + ISA + FPGA reservoir
print("\n--- C: MLP + ISA + FPGA reservoir ---")
print("  Running FPGA reservoir (simulated, 128 LIF neurons)...")
fpga = FPGAReservoir(128)
fpga_features = np.zeros((N, 128*2 + 128*3), dtype=np.float32)  # mean+count+temporal

t0 = time.time()
for i in range(N):
    fpga_features[i] = fpga.run(X_test[i], isa_signals=isa_all[i])
    if (i+1) % 2000 == 0:
        print(f"    {i+1}/{N} ({time.time()-t0:.1f}s)")
print(f"  FPGA done ({time.time()-t0:.1f}s)")

# Combine: MLP logits + ISA signals + FPGA features
features_c = np.concatenate([logits_all, isa_all, fpga_features], axis=1)
print(f"  Feature dims: MLP={logits_all.shape[1]} + ISA={isa_all.shape[1]} + FPGA={fpga_features.shape[1]} = {features_c.shape[1]}")

acc_c, _ = ridge_classify(features_c[:n_train], y_test[:n_train],
                          features_c[n_train:], y_test[n_train:])
print(f"  Accuracy: {acc_c:.2f}%")

# D: FPGA reservoir alone
print("\n--- D: FPGA reservoir alone ---")
acc_d, _ = ridge_classify(fpga_features[:n_train], y_test[:n_train],
                          fpga_features[n_train:], y_test[n_train:])
print(f"  Accuracy: {acc_d:.2f}%")

# E: MLP + FPGA (without ISA coupling)
print("\n--- E: MLP + FPGA (no ISA coupling) ---")
fpga_no_isa = FPGAReservoir(128)
fpga_features_no_isa = np.zeros_like(fpga_features)
for i in range(N):
    fpga_features_no_isa[i] = fpga_no_isa.run(X_test[i], isa_signals=None)
features_e = np.concatenate([logits_all, fpga_features_no_isa], axis=1)
acc_e, _ = ridge_classify(features_e[:n_train], y_test[:n_train],
                          features_e[n_train:], y_test[n_train:])
print(f"  Accuracy: {acc_e:.2f}%")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  A: MLP alone:           {acc_mlp:.2f}%")
print(f"  B: MLP + ISA:           {acc_b:.2f}%  (Δ={acc_b-acc_mlp:+.2f}pp)")
print(f"  C: MLP + ISA + FPGA:    {acc_c:.2f}%  (Δ={acc_c-acc_mlp:+.2f}pp)")
print(f"  D: FPGA alone:          {acc_d:.2f}%")
print(f"  E: MLP + FPGA (no ISA): {acc_e:.2f}%  (Δ={acc_e-acc_mlp:+.2f}pp)")
print()
if acc_c > acc_mlp:
    print(f"  >>> INTEGRATION IMPROVES: +{acc_c-acc_mlp:.2f}pp over MLP alone <<<")
    if acc_c > acc_e:
        print(f"  >>> ISA COUPLING HELPS: +{acc_c-acc_e:.2f}pp over MLP+FPGA without ISA <<<")
else:
    print(f"  Integration does not improve over MLP alone")

# Save
results = {
    'mlp_alone': float(acc_mlp),
    'mlp_isa': float(acc_b),
    'mlp_isa_fpga': float(acc_c),
    'fpga_alone': float(acc_d),
    'mlp_fpga_no_isa': float(acc_e),
    'delta_integration': float(acc_c - acc_mlp),
    'delta_isa_coupling': float(acc_c - acc_e),
    'n_train': n_train,
    'n_test': n_test,
}
with open(f'{base}/results/z2421_integrated.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2421_integrated.json")
