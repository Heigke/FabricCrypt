#!/usr/bin/env python3
"""
z2427: VG-batch as 32-dimensional FPGA input

z2423-z2426 all failed because MAC is a SINGLE SCALAR input.
128 neurons all receive the same stimulus → no spatial differentiation.

VG controls gate voltage PER NEURON GROUP (4 groups of 32).
But we can use set_vg_batch to set 32 individual Vg values rapidly.
This gives us 32 distinct input channels → spatial information!

Architecture:
  1. MLP computes h1 (128d) → PCA to 32d
  2. 32 components → 32 VG values → 32 groups of 4 neurons each
  3. Different activations → different Vg → different spiking regimes per neuron
  4. This creates SPATIAL DIFFERENTIATION that MAC alone cannot provide

Key insight: Vg controls avalanche sensitivity. High Vg → more spiking.
So we're encoding MLP's learned features into WHICH neurons spike,
not just HOW MUCH they spike.

Also test: using TOP-10 logit-derived features mapped to VG.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/scripts')
base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

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
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def mlp_full(x):
    h1 = relu(w1 @ x + b1)
    h2 = relu(w2 @ h1 + b2)
    logits = w3 @ h2 + b3
    return logits, h1, h2

from fpga_host_eth import FPGAEthBridge

print("=" * 60)
print("z2427: VG-BATCH AS 32D FPGA INPUT")
print("=" * 60)

fpga = FPGAEthBridge(local_port=7725, timeout=2.0)
ok = fpga.connect()
if not ok:
    print("FPGA connection failed!")
    sys.exit(1)

def check_temp():
    try:
        return int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
    except:
        return 0

def wait_cool(target=55):
    t = check_temp()
    if t > 70:
        print(f"  Cooling ({t:.0f}°C)...", end='', flush=True)
        while t > target:
            time.sleep(3)
            t = check_temp()
        print(f" {t:.0f}°C OK")

def ridge_classify(X_tr, y_tr, X_te, y_te, alpha=1.0):
    n_c = 10
    Y = np.zeros((len(y_tr), n_c))
    for i, y in enumerate(y_tr): Y[i, y] = 1.0
    XtX = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
    W = np.linalg.solve(XtX, X_tr.T @ Y)
    preds = (X_te @ W).argmax(axis=1)
    return (preds == y_te).mean() * 100

# Base FPGA config
fpga.set_leak_cond(0x0004)
fpga.set_threshold_raw(0x8000)
fpga.set_base_exc_raw(0x0333)
fpga.set_bias_gain_raw(0x0800)
fpga.set_mac_signal(0.5)  # constant moderate MAC
time.sleep(0.5)

N = 2000
n_train = 1400

print(f"\nUsing {N} samples, temp={check_temp():.0f}°C")
print("Running MLP...")
all_logits = np.zeros((N, 10), dtype=np.float32)
all_h1 = np.zeros((N, 128), dtype=np.float32)
all_h2 = np.zeros((N, 64), dtype=np.float32)

for i in range(N):
    all_logits[i], all_h1[i], all_h2[i] = mlp_full(X_test[i])

acc_mlp = (all_logits.argmax(axis=1) == y_test[:N]).mean() * 100
acc_mlp_ridge = ridge_classify(all_logits[:n_train], y_test[:n_train],
                                all_logits[n_train:N], y_test[n_train:N])
print(f"MLP: {acc_mlp:.2f}% (argmax), {acc_mlp_ridge:.2f}% (ridge)")

# PCA h1 → 32d (to match 4 VG groups × 8 sub-values, or 32 individual neurons)
# Simple approach: reshape h1 (128) → mean of groups of 4 → 32 values
h1_32 = all_h1.reshape(N, 32, 4).mean(axis=2)  # (N, 32)
# Normalize to VG range [0.01, 0.65] — BVpar cliff at ~0.60
h1_max = h1_32.max(axis=0, keepdims=True) + 1e-10
h1_vg = h1_32 / h1_max * 0.55 + 0.05  # [0.05, 0.60]

# Enable auto-telemetry
fpga.enable_auto_telemetry(2000)
time.sleep(0.1)
for _ in range(50):
    fpga.recv_auto_telemetry(timeout=0.002)

# ================================================================
# Condition A: VG encoding of h1 features
# ================================================================
print("\n--- A: VG encoding of h1 (32d) ---")
fpga_feat_a = np.zeros((N, 256), dtype=np.float32)
t0 = time.time()

for i in range(N):
    if i % 50 == 0:
        fpga.disable_auto_telemetry()
        fpga.set_kill(True); time.sleep(0.005)
        fpga.set_kill(False); time.sleep(0.005)
        fpga.enable_auto_telemetry(2000)
        time.sleep(0.01)
        for _ in range(20):
            fpga.recv_auto_telemetry(timeout=0.002)

    # Set VG values encoding h1 features
    vg_vals = h1_vg[i].tolist()
    # Send in 4 batches of 32 neurons, each batch gets 8 VG values
    # But set_vg_batch takes start_id and list — each VG applies to 1 neuron
    # Actually set_vg_batch sets 32 neurons starting at start_id
    # We have 128 neurons, so 4 batches
    for grp in range(4):
        # Map 8 h1 components to 32 neurons (each h1 → 4 neurons)
        vg_32 = []
        for j in range(8):
            v = float(vg_vals[grp * 8 + j])
            vg_32.extend([v] * 4)  # 4 neurons per VG value
        fpga.set_vg_batch(grp * 32, vg_32)

    # Let FPGA process for ~30ms
    time.sleep(0.030)

    # Read telemetry
    frames = []
    for _ in range(10):
        t = fpga.recv_auto_telemetry(timeout=0.003)
        if t: frames.append(t)

    if frames:
        fpga_feat_a[i, :128] = frames[-1]['spike_counts'].astype(np.float32)
        fpga_feat_a[i, 128:] = frames[-1]['vmem'].astype(np.float32)

    if (i+1) % 200 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (N - i - 1)
        print(f"  {i+1}/{N} ({elapsed:.0f}s, ETA {eta:.0f}s, {check_temp():.0f}°C)")
        if check_temp() > 75: wait_cool()

elapsed_a = time.time() - t0
print(f"  Done ({elapsed_a:.1f}s)")

# ================================================================
# Condition B: VG encoding + temporal (send image rows via MAC while VG is set)
# ================================================================
print("\n--- B: VG(h1) + MAC(image rows) ---")
wait_cool()
fpga_feat_b = np.zeros((N, 256), dtype=np.float32)
t0 = time.time()

for i in range(N):
    if i % 50 == 0:
        fpga.disable_auto_telemetry()
        fpga.set_kill(True); time.sleep(0.005)
        fpga.set_kill(False); time.sleep(0.005)
        fpga.enable_auto_telemetry(2000)
        time.sleep(0.01)
        for _ in range(20):
            fpga.recv_auto_telemetry(timeout=0.002)

    # Set VG encoding h1
    vg_vals = h1_vg[i].tolist()
    for grp in range(4):
        vg_32 = []
        for j in range(8):
            v = float(vg_vals[grp * 8 + j])
            vg_32.extend([v] * 4)
        fpga.set_vg_batch(grp * 32, vg_32)

    # Then feed image rows via MAC for temporal dynamics
    image = X_test[i].reshape(28, 28)
    for row in range(0, 28, 2):  # 14 rows
        fpga.set_mac_signal(float(np.clip(image[row].mean(), 0, 0.99)))
        time.sleep(0.003)

    frames = []
    for _ in range(10):
        t = fpga.recv_auto_telemetry(timeout=0.003)
        if t: frames.append(t)

    if frames:
        fpga_feat_b[i, :128] = frames[-1]['spike_counts'].astype(np.float32)
        if len(frames) > 1:
            fpga_feat_b[i, 128:] = frames[-1]['spike_counts'].astype(np.float32) - frames[0]['spike_counts'].astype(np.float32)
        else:
            fpga_feat_b[i, 128:] = frames[-1]['vmem'].astype(np.float32)

    if (i+1) % 200 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (N - i - 1)
        print(f"  {i+1}/{N} ({elapsed:.0f}s, ETA {eta:.0f}s, {check_temp():.0f}°C)")
        if check_temp() > 75: wait_cool()

elapsed_b = time.time() - t0
print(f"  Done ({elapsed_b:.1f}s)")

fpga.disable_auto_telemetry()

# Normalize
for feat in [fpga_feat_a, fpga_feat_b]:
    for j in range(256):
        col = feat[:, j]
        std = col.std()
        if std > 1e-2:
            feat[:, j] = (col - col.mean()) / std
        else:
            feat[:, j] = 0.0

# ================================================================
# Results
# ================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  MLP argmax:          {acc_mlp:.2f}%")
print(f"  MLP ridge:           {acc_mlp_ridge:.2f}%")

# h1 features directly
acc_h1 = ridge_classify(all_h1[:n_train], y_test[:n_train],
                          all_h1[n_train:N], y_test[n_train:N])
print(f"  h1 ridge (128d):     {acc_h1:.2f}%")

feat_lh = np.concatenate([all_logits[:N], all_h1[:N]], axis=1)
acc_lh = ridge_classify(feat_lh[:n_train], y_test[:n_train],
                          feat_lh[n_train:N], y_test[n_train:N])
print(f"  logits+h1 ridge:     {acc_lh:.2f}%")

# A: VG encoding
feat_a = np.concatenate([all_logits[:N], fpga_feat_a], axis=1)
acc_a = ridge_classify(feat_a[:n_train], y_test[:n_train],
                        feat_a[n_train:N], y_test[n_train:N])
print(f"\n  A: logits+FPGA(VG):  {acc_a:.2f}%  (Δ={acc_a-acc_mlp_ridge:+.2f}pp)")

# B: VG + MAC temporal
feat_b = np.concatenate([all_logits[:N], fpga_feat_b], axis=1)
acc_b = ridge_classify(feat_b[:n_train], y_test[:n_train],
                        feat_b[n_train:N], y_test[n_train:N])
print(f"  B: logits+FPGA(VG+MAC): {acc_b:.2f}%  (Δ={acc_b-acc_mlp_ridge:+.2f}pp)")

# A with h1
feat_ah = np.concatenate([all_logits[:N], all_h1[:N], fpga_feat_a], axis=1)
acc_ah = ridge_classify(feat_ah[:n_train], y_test[:n_train],
                          feat_ah[n_train:N], y_test[n_train:N])
print(f"  A+h1: full:          {acc_ah:.2f}%  (Δ={acc_ah-acc_mlp_ridge:+.2f}pp)")

# FPGA alone
acc_fa = ridge_classify(fpga_feat_a[:n_train], y_test[:n_train],
                          fpga_feat_a[n_train:N], y_test[n_train:N])
acc_fb = ridge_classify(fpga_feat_b[:n_train], y_test[:n_train],
                          fpga_feat_b[n_train:N], y_test[n_train:N])
print(f"\n  FPGA alone (A):      {acc_fa:.2f}%")
print(f"  FPGA alone (B):      {acc_fb:.2f}%")

n_useful_a = (fpga_feat_a.var(axis=0) > 0.01).sum()
n_useful_b = (fpga_feat_b.var(axis=0) > 0.01).sum()
print(f"  Useful features: A={n_useful_a}/256, B={n_useful_b}/256")

best = max(acc_a, acc_b, acc_ah)
print(f"\n  BEST: {best:.2f}%")
if best > acc_mlp:
    print(f"  >>> BEATS MLP ARGMAX <<<")
elif best > acc_mlp_ridge:
    print(f"  >>> BEATS MLP RIDGE <<<")
else:
    print(f"  FPGA still hurts ({best-acc_mlp_ridge:+.2f}pp vs ridge)")

results = {
    'mlp_argmax': float(acc_mlp), 'mlp_ridge': float(acc_mlp_ridge),
    'h1_ridge': float(acc_h1), 'logits_h1': float(acc_lh),
    'A_vg': float(acc_a), 'B_vg_mac': float(acc_b),
    'A_full': float(acc_ah),
    'fpga_alone_a': float(acc_fa), 'fpga_alone_b': float(acc_fb),
}
with open(f'{base}/results/z2427_vg_input.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2427_vg_input.json")

fpga.close()
