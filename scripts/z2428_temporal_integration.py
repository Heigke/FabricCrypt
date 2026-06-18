#!/usr/bin/env python3
"""
z2428: Honest integrated demo — GPU + FPGA on temporal tasks

z2423-z2427 proved: Real FPGA cannot improve MNIST classification.
Root cause: Single MAC channel → no spatial input diversity → all 128 neurons
receive identical stimulus → spike patterns don't encode class information.

BUT FPGA excels at TEMPORAL tasks (proven: MC=12.27, XOR5=88.3%, waveform=81%).
So let's build an integrated system where each substrate does what it's BEST at:

  GPU: Fast parallel spatial processing (classification, feature extraction)
  FPGA: Temporal pattern recognition (time series, sequence detection)

DEMO: Temporal digit sequence recognition
  1. GPU classifies individual MNIST digits (fast, parallel, 92% accurate)
  2. Sequences of digits are fed to FPGA over time via MAC
  3. FPGA recognizes TEMPORAL PATTERNS in the digit sequence
  4. Task: "Is this a monotonically increasing sequence?" (e.g., 1-3-5-7 = yes, 3-1-5-2 = no)
  5. This task REQUIRES temporal memory — GPU alone can do it, but FPGA does it naturally

Also test:
  - Anomaly detection in digit sequences (sudden outlier)
  - Sequence prediction (what digit comes next?)
  - Temporal XOR on GPU-produced confidence signals

This is a REAL use case: monitoring a production ML pipeline's outputs over time.
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

def mlp_predict(x):
    h1 = relu(w1 @ x + b1)
    h2 = relu(w2 @ h1 + b2)
    logits = w3 @ h2 + b3
    probs = softmax(logits)
    return logits.argmax(), probs.max(), probs

from fpga_host_eth import FPGAEthBridge

print("=" * 60)
print("z2428: TEMPORAL INTEGRATION — GPU classifies, FPGA remembers")
print("=" * 60)

fpga = FPGAEthBridge(local_port=7726, timeout=2.0)
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

# Configure FPGA
fpga.set_leak_cond(0x0004)       # τ≈210ms
fpga.set_threshold_raw(0x8000)   # 0.5V
fpga.set_base_exc_raw(0x0333)
fpga.set_bias_gain_raw(0x0800)   # gentle MAC
VG = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
for grp, vg in VG.items():
    fpga.set_vg_batch(grp*32, [vg]*32)
time.sleep(0.5)

fpga.enable_auto_telemetry(2000)
time.sleep(0.1)
for _ in range(50):
    fpga.recv_auto_telemetry(timeout=0.002)

# ================================================================
# Generate sequences from MNIST
# ================================================================
np.random.seed(42)

# Pre-classify all test images with GPU (MLP)
print("\nGPU: Classifying MNIST...")
gpu_preds = np.zeros(len(X_test), dtype=int)
gpu_confs = np.zeros(len(X_test), dtype=np.float32)
for i in range(len(X_test)):
    gpu_preds[i], gpu_confs[i], _ = mlp_predict(X_test[i])

gpu_acc = (gpu_preds == y_test).mean() * 100
print(f"  GPU accuracy: {gpu_acc:.2f}%")

# Create digit sequences for temporal tasks
SEQ_LEN = 8  # 8 digits per sequence
N_SEQ = 500  # 500 sequences

def make_sequences(n_seq, seq_len, task='monotone'):
    """Generate labeled sequences from classified MNIST digits."""
    sequences = []
    labels = []
    indices = []  # which MNIST images to use

    # Group images by predicted digit
    digit_indices = {d: np.where(gpu_preds == d)[0] for d in range(10)}

    for _ in range(n_seq):
        if task == 'monotone':
            # Task: is the sequence monotonically increasing?
            if np.random.random() < 0.5:
                # Positive: sorted increasing
                digits = sorted(np.random.choice(10, seq_len, replace=True))
                label = 1
            else:
                # Negative: random (unlikely to be monotone)
                digits = np.random.choice(10, seq_len, replace=True).tolist()
                # Make sure it's NOT monotone
                while all(digits[i] <= digits[i+1] for i in range(len(digits)-1)):
                    digits = np.random.choice(10, seq_len, replace=True).tolist()
                label = 0

        elif task == 'even_odd':
            # Task: are >50% of digits even?
            digits = np.random.choice(10, seq_len, replace=True).tolist()
            label = 1 if sum(d % 2 == 0 for d in digits) > seq_len // 2 else 0

        elif task == 'sum_threshold':
            # Task: does the running sum exceed 30?
            digits = np.random.choice(10, seq_len, replace=True).tolist()
            label = 1 if sum(digits) > 30 else 0

        elif task == 'temporal_xor':
            # Task: XOR of first half parity vs second half parity
            digits = np.random.choice(10, seq_len, replace=True).tolist()
            first_even = sum(d % 2 == 0 for d in digits[:seq_len//2]) > seq_len//4
            second_even = sum(d % 2 == 0 for d in digits[seq_len//2:]) > seq_len//4
            label = 1 if (first_even != second_even) else 0

        # Pick random images with matching predicted digits
        img_idx = []
        for d in digits:
            if len(digit_indices[d]) > 0:
                img_idx.append(np.random.choice(digit_indices[d]))
            else:
                img_idx.append(np.random.choice(len(X_test)))

        sequences.append(digits)
        labels.append(label)
        indices.append(img_idx)

    return sequences, np.array(labels), indices

# ================================================================
# Task 1: Monotone sequence detection
# ================================================================
print("\n--- Task 1: Monotone sequence detection ---")
seqs, labels, img_indices = make_sequences(N_SEQ, SEQ_LEN, 'monotone')
print(f"  {N_SEQ} sequences, {SEQ_LEN} digits each")
print(f"  Balance: {labels.mean():.2f} positive")

n_train = 350
n_test_seq = N_SEQ - n_train

# Method A: GPU only — use GPU predictions directly
# Simple: count if predictions are sorted
gpu_features = np.zeros((N_SEQ, SEQ_LEN + 5), dtype=np.float32)
for i in range(N_SEQ):
    preds = [gpu_preds[j] for j in img_indices[i]]
    confs = [gpu_confs[j] for j in img_indices[i]]
    gpu_features[i, :SEQ_LEN] = preds
    # Engineered features
    gpu_features[i, SEQ_LEN] = all(preds[k] <= preds[k+1] for k in range(SEQ_LEN-1))  # is_sorted
    gpu_features[i, SEQ_LEN+1] = np.mean(confs)  # avg confidence
    gpu_features[i, SEQ_LEN+2] = np.std(preds)   # digit spread
    gpu_features[i, SEQ_LEN+3] = max(preds) - min(preds)  # range
    gpu_features[i, SEQ_LEN+4] = sum(preds[k] <= preds[k+1] for k in range(SEQ_LEN-1)) / (SEQ_LEN-1)

# Ridge on GPU features
def ridge_binary(X_tr, y_tr, X_te, y_te, alpha=1.0):
    Y = y_tr.reshape(-1, 1).astype(float)
    XtX = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
    W = np.linalg.solve(XtX, X_tr.T @ Y)
    preds = (X_te @ W).ravel() > 0.5
    return (preds == y_te).mean() * 100

acc_gpu = ridge_binary(gpu_features[:n_train], labels[:n_train],
                        gpu_features[n_train:], labels[n_train:])
print(f"  GPU only: {acc_gpu:.1f}%")

# Method B: FPGA temporal processing of GPU predictions
# Feed each digit prediction as MAC signal over time
fpga_features = np.zeros((N_SEQ, 128 + 128), dtype=np.float32)  # spikes + vmem
t0 = time.time()

for i in range(N_SEQ):
    # Reset
    if i % 25 == 0:
        fpga.disable_auto_telemetry()
        fpga.set_kill(True); time.sleep(0.005)
        fpga.set_kill(False); time.sleep(0.005)
        fpga.enable_auto_telemetry(2000)
        time.sleep(0.01)
        for _ in range(20):
            fpga.recv_auto_telemetry(timeout=0.002)

    # Feed GPU predictions as temporal signal
    for j in range(SEQ_LEN):
        pred = gpu_preds[img_indices[i][j]]
        conf = gpu_confs[img_indices[i][j]]
        # MAC = predicted_digit/9 × confidence → [0, 1]
        mac_val = (pred / 9.0) * conf
        fpga.set_mac_signal(float(np.clip(mac_val, 0, 0.99)))
        time.sleep(0.010)  # 10ms per digit → 80ms per sequence

    # Collect final state
    time.sleep(0.005)
    frames = []
    for _ in range(5):
        t = fpga.recv_auto_telemetry(timeout=0.005)
        if t: frames.append(t)

    if frames:
        fpga_features[i, :128] = frames[-1]['spike_counts'].astype(np.float32)
        fpga_features[i, 128:] = frames[-1]['vmem'].astype(np.float32)

    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{N_SEQ} ({elapsed:.0f}s, {check_temp():.0f}°C)")
        if check_temp() > 75: wait_cool()

elapsed_1 = time.time() - t0
print(f"  FPGA done ({elapsed_1:.1f}s)")

# Normalize FPGA features
for j in range(256):
    col = fpga_features[:, j]
    std = col.std()
    if std > 1e-2:
        fpga_features[:, j] = (col - col.mean()) / std
    else:
        fpga_features[:, j] = 0.0

# FPGA alone
acc_fpga = ridge_binary(fpga_features[:n_train], labels[:n_train],
                          fpga_features[n_train:], labels[n_train:])
print(f"  FPGA alone: {acc_fpga:.1f}%")

# GPU + FPGA
combined = np.concatenate([gpu_features, fpga_features], axis=1)
acc_combined = ridge_binary(combined[:n_train], labels[:n_train],
                             combined[n_train:], labels[n_train:])
print(f"  GPU + FPGA: {acc_combined:.1f}% (Δ={acc_combined-acc_gpu:+.1f}pp)")

# ================================================================
# Task 2: Sum threshold (harder — requires memory of accumulation)
# ================================================================
print("\n--- Task 2: Sum threshold (Σdigits > 30?) ---")
seqs2, labels2, img_indices2 = make_sequences(N_SEQ, SEQ_LEN, 'sum_threshold')
print(f"  Balance: {labels2.mean():.2f} positive")
wait_cool()

# GPU features for sum task
gpu_features2 = np.zeros((N_SEQ, SEQ_LEN + 3), dtype=np.float32)
for i in range(N_SEQ):
    preds = [gpu_preds[j] for j in img_indices2[i]]
    gpu_features2[i, :SEQ_LEN] = preds
    gpu_features2[i, SEQ_LEN] = sum(preds)
    gpu_features2[i, SEQ_LEN+1] = np.mean(preds)
    gpu_features2[i, SEQ_LEN+2] = np.mean([gpu_confs[j] for j in img_indices2[i]])

acc_gpu2 = ridge_binary(gpu_features2[:n_train], labels2[:n_train],
                          gpu_features2[n_train:], labels2[n_train:])
print(f"  GPU only: {acc_gpu2:.1f}%")

# FPGA on sum task
fpga_features2 = np.zeros((N_SEQ, 256), dtype=np.float32)
t0 = time.time()

for i in range(N_SEQ):
    if i % 25 == 0:
        fpga.disable_auto_telemetry()
        fpga.set_kill(True); time.sleep(0.005)
        fpga.set_kill(False); time.sleep(0.005)
        fpga.enable_auto_telemetry(2000)
        time.sleep(0.01)
        for _ in range(20):
            fpga.recv_auto_telemetry(timeout=0.002)

    for j in range(SEQ_LEN):
        pred = gpu_preds[img_indices2[i][j]]
        conf = gpu_confs[img_indices2[i][j]]
        mac_val = (pred / 9.0) * conf
        fpga.set_mac_signal(float(np.clip(mac_val, 0, 0.99)))
        time.sleep(0.010)

    time.sleep(0.005)
    frames = []
    for _ in range(5):
        t = fpga.recv_auto_telemetry(timeout=0.005)
        if t: frames.append(t)

    if frames:
        fpga_features2[i, :128] = frames[-1]['spike_counts'].astype(np.float32)
        fpga_features2[i, 128:] = frames[-1]['vmem'].astype(np.float32)

    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{N_SEQ} ({elapsed:.0f}s, {check_temp():.0f}°C)")

elapsed_2 = time.time() - t0

for j in range(256):
    col = fpga_features2[:, j]
    std = col.std()
    if std > 1e-2:
        fpga_features2[:, j] = (col - col.mean()) / std
    else:
        fpga_features2[:, j] = 0.0

acc_fpga2 = ridge_binary(fpga_features2[:n_train], labels2[:n_train],
                           fpga_features2[n_train:], labels2[n_train:])
print(f"  FPGA alone: {acc_fpga2:.1f}%")

combined2 = np.concatenate([gpu_features2, fpga_features2], axis=1)
acc_combined2 = ridge_binary(combined2[:n_train], labels2[:n_train],
                              combined2[n_train:], labels2[n_train:])
print(f"  GPU + FPGA: {acc_combined2:.1f}% (Δ={acc_combined2-acc_gpu2:+.1f}pp)")

# ================================================================
# Task 3: Temporal XOR (nonlinear temporal)
# ================================================================
print("\n--- Task 3: Temporal XOR (first-half vs second-half parity) ---")
seqs3, labels3, img_indices3 = make_sequences(N_SEQ, SEQ_LEN, 'temporal_xor')
print(f"  Balance: {labels3.mean():.2f} positive")
wait_cool()

gpu_features3 = np.zeros((N_SEQ, SEQ_LEN + 4), dtype=np.float32)
for i in range(N_SEQ):
    preds = [gpu_preds[j] for j in img_indices3[i]]
    gpu_features3[i, :SEQ_LEN] = preds
    first_half_even = sum(p % 2 == 0 for p in preds[:SEQ_LEN//2])
    second_half_even = sum(p % 2 == 0 for p in preds[SEQ_LEN//2:])
    gpu_features3[i, SEQ_LEN] = first_half_even
    gpu_features3[i, SEQ_LEN+1] = second_half_even
    gpu_features3[i, SEQ_LEN+2] = first_half_even != second_half_even  # cheating feature
    gpu_features3[i, SEQ_LEN+3] = np.mean([gpu_confs[j] for j in img_indices3[i]])

acc_gpu3 = ridge_binary(gpu_features3[:n_train], labels3[:n_train],
                          gpu_features3[n_train:], labels3[n_train:])
# Also test WITHOUT the cheating feature
acc_gpu3_fair = ridge_binary(gpu_features3[:n_train, :SEQ_LEN+2], labels3[:n_train],
                              gpu_features3[n_train:, :SEQ_LEN+2], labels3[n_train:])
print(f"  GPU (with XOR feat): {acc_gpu3:.1f}%")
print(f"  GPU (fair, no XOR):  {acc_gpu3_fair:.1f}%")

fpga_features3 = np.zeros((N_SEQ, 256), dtype=np.float32)
t0 = time.time()

for i in range(N_SEQ):
    if i % 25 == 0:
        fpga.disable_auto_telemetry()
        fpga.set_kill(True); time.sleep(0.005)
        fpga.set_kill(False); time.sleep(0.005)
        fpga.enable_auto_telemetry(2000)
        time.sleep(0.01)
        for _ in range(20):
            fpga.recv_auto_telemetry(timeout=0.002)

    for j in range(SEQ_LEN):
        pred = gpu_preds[img_indices3[i][j]]
        conf = gpu_confs[img_indices3[i][j]]
        mac_val = (pred / 9.0) * conf
        fpga.set_mac_signal(float(np.clip(mac_val, 0, 0.99)))
        time.sleep(0.010)

    time.sleep(0.005)
    frames = []
    for _ in range(5):
        t = fpga.recv_auto_telemetry(timeout=0.005)
        if t: frames.append(t)

    if frames:
        fpga_features3[i, :128] = frames[-1]['spike_counts'].astype(np.float32)
        fpga_features3[i, 128:] = frames[-1]['vmem'].astype(np.float32)

for j in range(256):
    col = fpga_features3[:, j]
    std = col.std()
    if std > 1e-2:
        fpga_features3[:, j] = (col - col.mean()) / std
    else:
        fpga_features3[:, j] = 0.0

acc_fpga3 = ridge_binary(fpga_features3[:n_train], labels3[:n_train],
                           fpga_features3[n_train:], labels3[n_train:])

combined3 = np.concatenate([gpu_features3[:, :SEQ_LEN+2], fpga_features3], axis=1)
acc_combined3 = ridge_binary(combined3[:n_train], labels3[:n_train],
                              combined3[n_train:], labels3[n_train:])
print(f"  FPGA alone: {acc_fpga3:.1f}%")
print(f"  GPU(fair) + FPGA: {acc_combined3:.1f}% (Δ={acc_combined3-acc_gpu3_fair:+.1f}pp)")

fpga.disable_auto_telemetry()

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY — GPU classifies, FPGA remembers")
print("=" * 60)
print(f"  Task 1 (monotone):     GPU={acc_gpu:.1f}%  FPGA={acc_fpga:.1f}%  BOTH={acc_combined:.1f}%  Δ={acc_combined-acc_gpu:+.1f}pp")
print(f"  Task 2 (sum>30):       GPU={acc_gpu2:.1f}%  FPGA={acc_fpga2:.1f}%  BOTH={acc_combined2:.1f}%  Δ={acc_combined2-acc_gpu2:+.1f}pp")
print(f"  Task 3 (temporal XOR): GPU={acc_gpu3_fair:.1f}%  FPGA={acc_fpga3:.1f}%  BOTH={acc_combined3:.1f}%  Δ={acc_combined3-acc_gpu3_fair:+.1f}pp")

any_improvement = any([
    acc_combined > acc_gpu + 0.5,
    acc_combined2 > acc_gpu2 + 0.5,
    acc_combined3 > acc_gpu3_fair + 0.5
])
if any_improvement:
    print("\n  >>> FPGA ADDS VALUE on temporal tasks! <<<")
else:
    print("\n  FPGA does not add value even on temporal tasks with this setup")

results = {
    'task1_monotone': {'gpu': float(acc_gpu), 'fpga': float(acc_fpga), 'both': float(acc_combined)},
    'task2_sum': {'gpu': float(acc_gpu2), 'fpga': float(acc_fpga2), 'both': float(acc_combined2)},
    'task3_xor': {'gpu_fair': float(acc_gpu3_fair), 'gpu_cheat': float(acc_gpu3),
                  'fpga': float(acc_fpga3), 'both': float(acc_combined3)},
    'gpu_accuracy': float(gpu_acc),
    'n_sequences': N_SEQ,
    'seq_len': SEQ_LEN,
}
with open(f'{base}/results/z2428_temporal_integration.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2428_temporal_integration.json")

fpga.close()
