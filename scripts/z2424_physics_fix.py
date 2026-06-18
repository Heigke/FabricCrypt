#!/usr/bin/env python3
"""
z2424: Physics-fixed integrated pipeline

z2423 FAILED (MLP+FPGA = -3.5pp vs MLP alone) because:
  1. LEAK_COND=0x2000 → τ≈49ms, too fast for 20Hz sampling (79% decay/step)
  2. BIAS_GAIN=0x4000 → 0.25, too strong (overwhelms avalanche dynamics)
  3. Only 7 timesteps per image (every 4th row) — not enough temporal dynamics

Fixes:
  A. LEAK_COND=0x0004 → τ≈210ms (RTL default, designed for this!)
  B. BIAS_GAIN=0x0800 → 0.03125 (RTL default, gentle MAC injection)
  C. All 28 rows per image (28 timesteps, 5ms spacing → ~140ms total)
  D. Richer features: spike counts + vmem + delta_spikes between rows
  E. Multiple readout points (after row 7, 14, 21, 28 → temporal trajectory)

Also test with RTL-default threshold (0x8000 = 0.5) vs old (0x20000 = 2.0)
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

def mlp_with_isa(x):
    h1 = relu(w1 @ x + b1)
    h2 = relu(w2 @ h1 + b2)
    logits = w3 @ h2 + b3
    isa = np.array([
        (h1 > 0).mean(), np.abs(np.diff(h1)).mean(), h1.std(),
        (h2 > 0).mean(), np.abs(np.diff(h2)).mean(), h2.std(),
    ], dtype=np.float32)
    return logits, isa, h1

from fpga_host_eth import FPGAEthBridge

print("=" * 60)
print("z2424: PHYSICS-FIXED INTEGRATED PIPELINE")
print("=" * 60)

fpga = FPGAEthBridge(local_port=7722, timeout=2.0)
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

# ================================================================
# Test multiple FPGA configurations
# ================================================================
N = 2000
n_train = 1400
n_test = N - n_train

# Pre-compute MLP
print(f"\nUsing {N} samples, temp={check_temp():.0f}°C")
print("Running MLP...")
all_logits = np.zeros((N, 10), dtype=np.float32)
all_isa = np.zeros((N, 6), dtype=np.float32)
for i in range(N):
    all_logits[i], all_isa[i], _ = mlp_with_isa(X_test[i])
acc_mlp = (all_logits.argmax(axis=1) == y_test[:N]).mean() * 100
print(f"MLP accuracy: {acc_mlp:.2f}%\n")

# Ridge on MLP logits alone
acc_mlp_ridge = ridge_classify(all_logits[:n_train], y_test[:n_train],
                                all_logits[n_train:N], y_test[n_train:N])
print(f"MLP ridge accuracy: {acc_mlp_ridge:.2f}%\n")

configs = {
    'OLD': {  # z2423 config that failed
        'leak': 0x2000, 'thresh': 0x20000, 'base_exc': 0x0080,
        'bias_gain': 0x4000, 'rows': 7, 'row_step': 4, 'delay': 0.015,
    },
    'PHYSICS': {  # RTL defaults — slow membrane, gentle MAC
        'leak': 0x0004, 'thresh': 0x8000, 'base_exc': 0x0333,
        'bias_gain': 0x0800, 'rows': 28, 'row_step': 1, 'delay': 0.005,
    },
    'PHYSICS_FAST': {  # RTL defaults + faster sampling
        'leak': 0x0004, 'thresh': 0x8000, 'base_exc': 0x0333,
        'bias_gain': 0x0800, 'rows': 14, 'row_step': 2, 'delay': 0.003,
    },
    'MID_LEAK': {  # Medium leak — between old and physics
        'leak': 0x0040, 'thresh': 0x10000, 'base_exc': 0x0100,
        'bias_gain': 0x1000, 'rows': 28, 'row_step': 1, 'delay': 0.005,
    },
}

# VG groups (same for all)
VG = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}

results = {'mlp_alone': float(acc_mlp), 'mlp_ridge': float(acc_mlp_ridge), 'configs': {}}

for cfg_name, cfg in configs.items():
    print(f"--- {cfg_name} ---")
    wait_cool()

    # Configure FPGA
    fpga.set_leak_cond(cfg['leak'])
    fpga.set_threshold_raw(cfg['thresh'])
    fpga.set_base_exc_raw(cfg['base_exc'])
    fpga.set_bias_gain_raw(cfg['bias_gain'])
    for grp, vg in VG.items():
        fpga.set_vg_batch(grp*32, [vg]*32)
    time.sleep(0.5)

    # Verify
    telem = fpga.read_telemetry()
    if telem:
        active = (telem['spike_counts'] > 0).sum()
        print(f"  FPGA: {active}/128 active, config: leak=0x{cfg['leak']:04X} thresh=0x{cfg['thresh']:05X} bias=0x{cfg['bias_gain']:04X}")
    else:
        print("  WARNING: no telemetry")

    n_rows = cfg['rows']
    row_step = cfg['row_step']
    delay = cfg['delay']

    # Features: spike_counts(128) + vmem(128) + mid-point spike snapshot(128) = 384
    fpga_feat = np.zeros((N, 384), dtype=np.float32)
    t0 = time.time()

    for i in range(N):
        # Reset every 50 samples
        if i % 50 == 0:
            fpga.set_kill(True)
            time.sleep(0.005)
            fpga.set_kill(False)
            time.sleep(0.005)

        image = X_test[i].reshape(28, 28)
        isa = all_isa[i]

        # ISA modulation
        isa_mod = 1.0 + (isa[1] - all_isa[:N, 1].mean()) / (all_isa[:N, 1].std() + 1e-10) * 0.2

        # Read mid-point telemetry
        mid_row = n_rows // 2
        mid_spikes = np.zeros(128, dtype=np.float32)

        for r_idx in range(n_rows):
            row = r_idx * row_step
            if row >= 28:
                break
            row_mean = image[row].mean()
            modulated = row_mean * np.clip(isa_mod, 0.7, 1.5)
            fpga.set_mac_signal(float(np.clip(modulated, 0, 0.99)))
            time.sleep(delay)

            # Mid-point snapshot
            if r_idx == mid_row:
                t_mid = fpga.read_telemetry()
                if t_mid:
                    mid_spikes = t_mid['spike_counts'].astype(np.float32)

        # Final telemetry
        telem = fpga.read_telemetry()
        if telem:
            fpga_feat[i, :128] = telem['spike_counts'].astype(np.float32)
            fpga_feat[i, 128:256] = telem['vmem'].astype(np.float32)
            # Delta: final - mid (temporal change)
            fpga_feat[i, 256:384] = telem['spike_counts'].astype(np.float32) - mid_spikes

        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (N - i - 1)
            t = check_temp()
            print(f"  {i+1}/{N} ({elapsed:.0f}s, ETA {eta:.0f}s, {t:.0f}°C)")
            if t > 75:
                wait_cool()

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s, {elapsed/N*1000:.0f}ms/sample)")

    # Normalize features
    for j in range(384):
        col = fpga_feat[:, j]
        std = col.std()
        if std > 1e-2:  # sigma floor (memory: must be 1e-2, not 1e-10!)
            fpga_feat[:, j] = (col - col.mean()) / std
        else:
            fpga_feat[:, j] = 0.0

    # Test: MLP + FPGA
    feat_combined = np.concatenate([all_logits[:N], fpga_feat], axis=1)
    acc_combined = ridge_classify(feat_combined[:n_train], y_test[:n_train],
                                   feat_combined[n_train:N], y_test[n_train:N])

    # Test: MLP + FPGA + ISA
    feat_full = np.concatenate([all_logits[:N], all_isa[:N], fpga_feat], axis=1)
    acc_full = ridge_classify(feat_full[:n_train], y_test[:n_train],
                               feat_full[n_train:N], y_test[n_train:N])

    # Test: FPGA alone
    acc_fpga = ridge_classify(fpga_feat[:n_train], y_test[:n_train],
                               fpga_feat[n_train:N], y_test[n_train:N])

    # Variance check
    feat_var = fpga_feat.var(axis=0)
    n_useful = (feat_var > 0.01).sum()

    print(f"  MLP+FPGA:       {acc_combined:.2f}%  (Δ={acc_combined-acc_mlp_ridge:+.2f}pp)")
    print(f"  MLP+ISA+FPGA:   {acc_full:.2f}%  (Δ={acc_full-acc_mlp_ridge:+.2f}pp)")
    print(f"  FPGA alone:     {acc_fpga:.2f}%")
    print(f"  Useful features: {n_useful}/384 (var>0.01)")
    print()

    results['configs'][cfg_name] = {
        'mlp_fpga': float(acc_combined),
        'mlp_isa_fpga': float(acc_full),
        'fpga_alone': float(acc_fpga),
        'time': float(elapsed),
        'useful_features': int(n_useful),
        'delta_vs_mlp': float(acc_combined - acc_mlp_ridge),
    }

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  MLP alone:  {acc_mlp:.2f}% (argmax)  {acc_mlp_ridge:.2f}% (ridge)")
for name, r in results['configs'].items():
    delta = r['delta_vs_mlp']
    marker = "★" if delta > 0 else ""
    print(f"  {name:15s}: {r['mlp_fpga']:.2f}% (Δ={delta:+.2f}pp) FPGA={r['fpga_alone']:.2f}% feat={r['useful_features']} {marker}")

best_name = max(results['configs'], key=lambda k: results['configs'][k]['mlp_fpga'])
best = results['configs'][best_name]
print(f"\n  BEST: {best_name} = {best['mlp_fpga']:.2f}% ({best['delta_vs_mlp']:+.2f}pp vs MLP ridge)")

with open(f'{base}/results/z2424_physics_fix.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to results/z2424_physics_fix.json")

fpga.close()
