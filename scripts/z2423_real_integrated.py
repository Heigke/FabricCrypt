#!/usr/bin/env python3
"""
z2423: REAL integrated pipeline — GPU MLP + ISA harvest + REAL FPGA 128 neurons

Architecture:
  1. GPU runs trained MLP (94.3% accuracy)
  2. ISA-simulated signals extracted per layer (timing proxy, neighbor diff, sign ratio)
  3. ISA signals + MLP intermediate activations → FPGA as MAC current
  4. FPGA 128 LIF neurons process temporally (image scanned row-by-row, 28 timesteps)
  5. FPGA telemetry (spike counts, vmem) → concatenated with MLP logits
  6. Ridge regression readout on combined features → final prediction

Conditions:
  A: MLP alone (baseline)
  B: MLP + FPGA (no ISA coupling — FPGA gets raw image rows)
  C: MLP + FPGA + ISA coupling (FPGA gets image rows MODULATED by ISA signals)
  D: FPGA alone

All with REAL FPGA hardware.
"""
import sys, os, time, json, struct
import numpy as np

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/scripts')
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

# Load MLP weights
w1 = np.fromfile(f'{base}/models/mnist_mlp/w1.bin', dtype=np.float32).reshape(128, 784)
b1 = np.fromfile(f'{base}/models/mnist_mlp/b1.bin', dtype=np.float32)
w2 = np.fromfile(f'{base}/models/mnist_mlp/w2.bin', dtype=np.float32).reshape(64, 128)
b2 = np.fromfile(f'{base}/models/mnist_mlp/b2.bin', dtype=np.float32)
w3 = np.fromfile(f'{base}/models/mnist_mlp/w3.bin', dtype=np.float32).reshape(10, 64)
b3 = np.fromfile(f'{base}/models/mnist_mlp/b3.bin', dtype=np.float32)

def relu(x): return np.maximum(0, x)

# MLP forward + ISA signal extraction
def mlp_with_isa(x):
    """Single sample MLP + ISA signals."""
    h1 = relu(w1 @ x + b1)
    h2 = relu(w2 @ h1 + b2)
    logits = w3 @ h2 + b3

    # ISA signals (simulate what HIP kernel measures)
    isa = np.array([
        (h1 > 0).mean(),                    # sign ratio L1
        np.abs(np.diff(h1)).mean(),          # neighbor diff L1
        h1.std(),                            # activation spread L1
        (h2 > 0).mean(),                     # sign ratio L2
        np.abs(np.diff(h2)).mean(),          # neighbor diff L2
        h2.std(),                            # activation spread L2
    ], dtype=np.float32)

    return logits, isa, h1

# Connect to REAL FPGA
from fpga_host_eth import FPGAEthBridge

print("=" * 60)
print("z2423: REAL INTEGRATED PIPELINE")
print("GPU MLP + ISA harvest + REAL FPGA 128 neurons")
print("=" * 60)

fpga = FPGAEthBridge(local_port=7720, timeout=2.0)
ok = fpga.connect()
if not ok:
    print("FPGA connection failed!")
    sys.exit(1)

# Configure FPGA
fpga.set_leak_cond(0x2000)
fpga.set_threshold_raw(0x20000)
fpga.set_base_exc_raw(0x0080)
fpga.set_bias_gain_raw(0x4000)
VG = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
for grp, vg in VG.items():
    fpga.set_vg_batch(grp*32, [vg]*32)
time.sleep(0.5)

# Verify FPGA
telem = fpga.read_telemetry()
if telem:
    print(f"FPGA: {(telem['spike_counts'] > 0).sum()}/128 neurons active")
else:
    print("WARNING: No telemetry")

# Thermal monitoring
def check_temp():
    try:
        t = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
        return t
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

# Ridge classifier
def ridge_classify(X_tr, y_tr, X_te, y_te, alpha=1.0):
    n_c = 10
    Y = np.zeros((len(y_tr), n_c))
    for i, y in enumerate(y_tr): Y[i, y] = 1.0
    XtX = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
    W = np.linalg.solve(XtX, X_tr.T @ Y)
    preds = (X_te @ W).argmax(axis=1)
    return (preds == y_te).mean() * 100

# ================================================================
# Run experiments
# ================================================================
N = 2000  # Use 2000 samples (FPGA is slow at 50Hz)
n_train = 1400
n_test = N - n_train
print(f"\nUsing {N} samples (train={n_train}, test={n_test})")
print(f"CPU temp: {check_temp():.0f}°C\n")

# Pre-compute MLP for all samples
print("Running MLP on all samples...")
all_logits = np.zeros((N, 10), dtype=np.float32)
all_isa = np.zeros((N, 6), dtype=np.float32)
all_h1 = np.zeros((N, 128), dtype=np.float32)
for i in range(N):
    all_logits[i], all_isa[i], all_h1[i] = mlp_with_isa(X_test[i])
preds_mlp = all_logits.argmax(axis=1)
acc_mlp = (preds_mlp[:N] == y_test[:N]).mean() * 100
print(f"MLP accuracy: {acc_mlp:.2f}%\n")

# ================================================================
# Condition B: MLP + FPGA (no ISA coupling)
# Feed each image row-by-row to FPGA, collect spike dynamics
# ================================================================
print("--- B: MLP + FPGA (raw image → FPGA) ---")
wait_cool()

fpga_features_b = np.zeros((N, 128*2), dtype=np.float32)  # spike_counts + vmem
t0 = time.time()

for i in range(N):
    # Reset FPGA state by sending kill then unkill
    if i % 100 == 0:
        fpga.set_kill(True)
        time.sleep(0.01)
        fpga.set_kill(False)
        time.sleep(0.01)

    # Feed image as MAC signal (use mean pixel intensity per row as current)
    image = X_test[i].reshape(28, 28)
    for row in range(0, 28, 4):  # every 4th row (7 timesteps, ~140ms at 50Hz)
        row_mean = image[row].mean()
        fpga.set_mac_signal(float(np.clip(row_mean, 0, 0.99)))
        time.sleep(0.015)  # ~66Hz effective

    # Read final state
    telem = fpga.read_telemetry()
    if telem:
        fpga_features_b[i, :128] = telem['spike_counts'].astype(np.float32)
        fpga_features_b[i, 128:] = telem['vmem'].astype(np.float32)

    if (i+1) % 200 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (N - i - 1)
        t = check_temp()
        print(f"  {i+1}/{N} ({elapsed:.0f}s, ETA {eta:.0f}s, {t:.0f}°C)")
        if t > 75:
            wait_cool()

elapsed_b = time.time() - t0
print(f"  Done ({elapsed_b:.1f}s)")

# Normalize FPGA features
for j in range(256):
    col = fpga_features_b[:, j]
    std = col.std()
    if std > 1e-10:
        fpga_features_b[:, j] = (col - col.mean()) / std

# B accuracy
feat_b = np.concatenate([all_logits[:N], fpga_features_b], axis=1)
acc_b = ridge_classify(feat_b[:n_train], y_test[:n_train],
                        feat_b[n_train:N], y_test[n_train:N])
print(f"  Accuracy: {acc_b:.2f}%\n")

# ================================================================
# Condition C: MLP + FPGA + ISA coupling
# ISA signals modulate MAC signal to FPGA
# ================================================================
print("--- C: MLP + FPGA + ISA coupling ---")
wait_cool()

fpga_features_c = np.zeros((N, 128*2), dtype=np.float32)
t0 = time.time()

for i in range(N):
    if i % 100 == 0:
        fpga.set_kill(True)
        time.sleep(0.01)
        fpga.set_kill(False)
        time.sleep(0.01)

    image = X_test[i].reshape(28, 28)
    isa = all_isa[i]

    # ISA coupling: modulate MAC signal with ISA features
    # ISA neighbor_diff (index 1) indicates MLP's internal disagreement
    # High disagreement → stronger MAC → more FPGA excitation
    isa_modulation = 1.0 + (isa[1] - all_isa[:N, 1].mean()) / (all_isa[:N, 1].std() + 1e-10) * 0.3

    for row in range(0, 28, 4):
        row_mean = image[row].mean()
        # Modulated MAC: raw signal × ISA modulation
        modulated = row_mean * np.clip(isa_modulation, 0.5, 2.0)
        fpga.set_mac_signal(float(np.clip(modulated, 0, 0.99)))
        time.sleep(0.015)

    telem = fpga.read_telemetry()
    if telem:
        fpga_features_c[i, :128] = telem['spike_counts'].astype(np.float32)
        fpga_features_c[i, 128:] = telem['vmem'].astype(np.float32)

    if (i+1) % 200 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (N - i - 1)
        t = check_temp()
        print(f"  {i+1}/{N} ({elapsed:.0f}s, ETA {eta:.0f}s, {t:.0f}°C)")
        if t > 75:
            wait_cool()

elapsed_c = time.time() - t0
print(f"  Done ({elapsed_c:.1f}s)")

for j in range(256):
    col = fpga_features_c[:, j]
    std = col.std()
    if std > 1e-10:
        fpga_features_c[:, j] = (col - col.mean()) / std

feat_c = np.concatenate([all_logits[:N], all_isa[:N], fpga_features_c], axis=1)
acc_c = ridge_classify(feat_c[:n_train], y_test[:n_train],
                        feat_c[n_train:N], y_test[n_train:N])
print(f"  Accuracy: {acc_c:.2f}%\n")

# ================================================================
# Condition D: FPGA alone
# ================================================================
print("--- D: FPGA alone ---")
acc_d = ridge_classify(fpga_features_c[:n_train], y_test[:n_train],
                        fpga_features_c[n_train:N], y_test[n_train:N])
print(f"  Accuracy: {acc_d:.2f}%\n")

# ================================================================
# Condition E: MLP + ISA only (no FPGA)
# ================================================================
print("--- E: MLP + ISA only ---")
feat_e = np.concatenate([all_logits[:N], all_isa[:N]], axis=1)
acc_e = ridge_classify(feat_e[:n_train], y_test[:n_train],
                        feat_e[n_train:N], y_test[n_train:N])
print(f"  Accuracy: {acc_e:.2f}%\n")

# ================================================================
# SUMMARY
# ================================================================
print("=" * 60)
print("SUMMARY — REAL INTEGRATED PIPELINE")
print("=" * 60)
print(f"  A: MLP alone:              {acc_mlp:.2f}%")
print(f"  B: MLP + FPGA:             {acc_b:.2f}%  (Δ={acc_b-acc_mlp:+.2f}pp)")
print(f"  C: MLP + ISA + FPGA:       {acc_c:.2f}%  (Δ={acc_c-acc_mlp:+.2f}pp)")
print(f"  D: FPGA alone:             {acc_d:.2f}%")
print(f"  E: MLP + ISA (no FPGA):    {acc_e:.2f}%  (Δ={acc_e-acc_mlp:+.2f}pp)")
print()
if acc_c > acc_b:
    print(f"  ISA coupling helps: C-B = +{acc_c-acc_b:.2f}pp")
elif acc_b > acc_c:
    print(f"  ISA coupling hurts: C-B = {acc_c-acc_b:.2f}pp")
if max(acc_b, acc_c) > acc_mlp:
    best = max(acc_b, acc_c)
    best_name = "B (MLP+FPGA)" if acc_b > acc_c else "C (MLP+ISA+FPGA)"
    print(f"  BEST: {best_name} = {best:.2f}% (+{best-acc_mlp:.2f}pp over MLP)")
print()
print(f"  FPGA processing time: B={elapsed_b:.0f}s C={elapsed_c:.0f}s")
print(f"  Per sample: B={elapsed_b/N*1000:.0f}ms C={elapsed_c/N*1000:.0f}ms")

# Save
results = {
    'mlp_alone': float(acc_mlp),
    'mlp_fpga': float(acc_b),
    'mlp_isa_fpga': float(acc_c),
    'fpga_alone': float(acc_d),
    'mlp_isa': float(acc_e),
    'n_samples': N, 'n_train': n_train, 'n_test': n_test,
    'fpga_time_b': float(elapsed_b),
    'fpga_time_c': float(elapsed_c),
}
with open(f'{base}/results/z2423_real_integrated.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2423_real_integrated.json")

fpga.close()
