#!/usr/bin/env python3
"""
z2426: MLP activation → FPGA pipeline

z2423-z2425 all failed: FPGA features don't correlate with MNIST class.
Root cause: MAC signal = mean pixel intensity per row → too little information.

New idea: Feed MLP's INTERNAL ACTIVATIONS to FPGA, not raw pixels.
- MLP h1 is 128-dimensional — each activation represents a learned feature detector
- These activations differ dramatically between classes (that's what makes MLP work!)
- We feed h1 activations sequentially to FPGA → temporal processing of MLP internals
- FPGA acts as a SECOND PROCESSING STAGE on MLP's features, not an independent classifier

This is NOT the FPGA classifying images. It's the FPGA processing MLP's learned features
temporally, potentially capturing inter-activation relationships that ridge regression misses.

Also test: feed MLP ERRORS (logit entropy, confidence) to modulate FPGA.
High uncertainty → stronger FPGA activation → more temporal processing on hard cases.
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
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    confidence = probs.max()
    return logits, h1, h2, entropy, confidence

from fpga_host_eth import FPGAEthBridge

print("=" * 60)
print("z2426: MLP ACTIVATION → FPGA PIPELINE")
print("=" * 60)

fpga = FPGAEthBridge(local_port=7724, timeout=2.0)
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

# Configure FPGA
fpga.set_leak_cond(0x0004)
fpga.set_threshold_raw(0x8000)
fpga.set_base_exc_raw(0x0333)
fpga.set_bias_gain_raw(0x0800)
VG = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
for grp, vg in VG.items():
    fpga.set_vg_batch(grp*32, [vg]*32)
time.sleep(0.5)

# Enable auto-telemetry
fpga.enable_auto_telemetry(2000)
time.sleep(0.1)

# Drain buffer
for _ in range(50):
    fpga.recv_auto_telemetry(timeout=0.002)

N = 2000
n_train = 1400
n_test = N - n_train

print(f"\nUsing {N} samples, temp={check_temp():.0f}°C")
print("Running MLP...")
all_logits = np.zeros((N, 10), dtype=np.float32)
all_h1 = np.zeros((N, 128), dtype=np.float32)
all_h2 = np.zeros((N, 64), dtype=np.float32)
all_entropy = np.zeros(N, dtype=np.float32)
all_confidence = np.zeros(N, dtype=np.float32)

for i in range(N):
    all_logits[i], all_h1[i], all_h2[i], all_entropy[i], all_confidence[i] = mlp_full(X_test[i])

acc_mlp = (all_logits.argmax(axis=1) == y_test[:N]).mean() * 100
acc_mlp_ridge = ridge_classify(all_logits[:n_train], y_test[:n_train],
                                all_logits[n_train:N], y_test[n_train:N])
print(f"MLP: {acc_mlp:.2f}% (argmax), {acc_mlp_ridge:.2f}% (ridge)")
print(f"Entropy: mean={all_entropy.mean():.2f}, std={all_entropy.std():.2f}")
print(f"Confidence: mean={all_confidence.mean():.3f}\n")

# ================================================================
# Condition A: Feed MLP h1 activations sequentially to FPGA
# h1 is 128-d, we normalize to [0,1] and feed 8 values at a time
# → 16 MAC steps per image, each encoding different h1 groups
# ================================================================
print("--- A: h1 activations → FPGA ---")

# Normalize h1 to [0,1] for MAC signal
h1_max = all_h1.max()
h1_norm = all_h1 / (h1_max + 1e-10)

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

    # Feed h1 in 16 groups of 8
    h1 = h1_norm[i]
    for g in range(16):
        mac_val = float(h1[g*8:(g+1)*8].mean())
        fpga.set_mac_signal(np.clip(mac_val, 0, 0.99))
        time.sleep(0.003)

    # Collect telemetry
    frames = []
    for _ in range(10):
        t = fpga.recv_auto_telemetry(timeout=0.002)
        if t:
            frames.append(t)

    if frames:
        last = frames[-1]
        fpga_feat_a[i, :128] = last['spike_counts'].astype(np.float32)
        fpga_feat_a[i, 128:] = last['vmem'].astype(np.float32)
        # Use temporal delta if we have multiple frames
        if len(frames) > 1:
            delta = frames[-1]['spike_counts'].astype(np.float32) - frames[0]['spike_counts'].astype(np.float32)
            # Overwrite vmem slot with delta (more informative)
            fpga_feat_a[i, 128:] = delta

    if (i+1) % 200 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (N - i - 1)
        print(f"  {i+1}/{N} ({elapsed:.0f}s, ETA {eta:.0f}s, {check_temp():.0f}°C)")
        if check_temp() > 75: wait_cool()

elapsed_a = time.time() - t0
print(f"  Done ({elapsed_a:.1f}s)")

# Normalize
for j in range(256):
    col = fpga_feat_a[:, j]
    std = col.std()
    if std > 1e-2:
        fpga_feat_a[:, j] = (col - col.mean()) / std
    else:
        fpga_feat_a[:, j] = 0.0

# ================================================================
# Condition B: Entropy-modulated MAC
# High entropy (uncertain) → stronger signal → more FPGA activity
# ================================================================
print("\n--- B: Entropy-modulated h1 → FPGA ---")
wait_cool()

ent_norm = all_entropy / (all_entropy.max() + 1e-10)  # [0,1]
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

    h1 = h1_norm[i]
    ent_mod = 0.5 + ent_norm[i] * 0.5  # range [0.5, 1.0]

    for g in range(16):
        mac_val = float(h1[g*8:(g+1)*8].mean()) * ent_mod
        fpga.set_mac_signal(np.clip(mac_val, 0, 0.99))
        time.sleep(0.003)

    frames = []
    for _ in range(10):
        t = fpga.recv_auto_telemetry(timeout=0.002)
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

for j in range(256):
    col = fpga_feat_b[:, j]
    std = col.std()
    if std > 1e-2:
        fpga_feat_b[:, j] = (col - col.mean()) / std
    else:
        fpga_feat_b[:, j] = 0.0

# ================================================================
# Results
# ================================================================
fpga.disable_auto_telemetry()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# MLP baselines
print(f"  MLP argmax:          {acc_mlp:.2f}%")
print(f"  MLP ridge:           {acc_mlp_ridge:.2f}%")

# h1 alone (no FPGA)
acc_h1 = ridge_classify(all_h1[:n_train], y_test[:n_train],
                          all_h1[n_train:N], y_test[n_train:N])
print(f"  h1 ridge (128d):     {acc_h1:.2f}%")

# logits + h1 (upper bound without FPGA)
feat_lh = np.concatenate([all_logits[:N], all_h1[:N]], axis=1)
acc_lh = ridge_classify(feat_lh[:n_train], y_test[:n_train],
                          feat_lh[n_train:N], y_test[n_train:N])
print(f"  logits+h1 ridge:     {acc_lh:.2f}%")

# Condition A
feat_a = np.concatenate([all_logits[:N], fpga_feat_a], axis=1)
acc_a = ridge_classify(feat_a[:n_train], y_test[:n_train],
                        feat_a[n_train:N], y_test[n_train:N])
print(f"\n  A: logits+FPGA(h1):  {acc_a:.2f}%  (Δ={acc_a-acc_mlp_ridge:+.2f}pp vs ridge)")

# Condition B
feat_b = np.concatenate([all_logits[:N], fpga_feat_b], axis=1)
acc_b = ridge_classify(feat_b[:n_train], y_test[:n_train],
                        feat_b[n_train:N], y_test[n_train:N])
print(f"  B: logits+FPGA(ent): {acc_b:.2f}%  (Δ={acc_b-acc_mlp_ridge:+.2f}pp vs ridge)")

# A + h1
feat_ah = np.concatenate([all_logits[:N], all_h1[:N], fpga_feat_a], axis=1)
acc_ah = ridge_classify(feat_ah[:n_train], y_test[:n_train],
                          feat_ah[n_train:N], y_test[n_train:N])
print(f"  A+h1: full pipeline: {acc_ah:.2f}%  (Δ={acc_ah-acc_mlp_ridge:+.2f}pp)")

# FPGA alone
acc_fa = ridge_classify(fpga_feat_a[:n_train], y_test[:n_train],
                          fpga_feat_a[n_train:N], y_test[n_train:N])
print(f"\n  FPGA alone (A):      {acc_fa:.2f}%")

n_useful = (fpga_feat_a.var(axis=0) > 0.01).sum()
print(f"  Useful FPGA features: {n_useful}/256")

best = max(acc_a, acc_b, acc_ah)
print(f"\n  BEST: {best:.2f}% (vs MLP argmax {acc_mlp:.2f}%, ridge {acc_mlp_ridge:.2f}%)")

results = {
    'mlp_argmax': float(acc_mlp),
    'mlp_ridge': float(acc_mlp_ridge),
    'h1_ridge': float(acc_h1),
    'logits_h1_ridge': float(acc_lh),
    'A_logits_fpga_h1': float(acc_a),
    'B_logits_fpga_ent': float(acc_b),
    'A_full_pipeline': float(acc_ah),
    'fpga_alone': float(acc_fa),
    'n_samples': N,
}
with open(f'{base}/results/z2426_activation_stream.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2426_activation_stream.json")

fpga.close()
