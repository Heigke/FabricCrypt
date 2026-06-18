#!/usr/bin/env python3
"""
z2425: Temporal stream pipeline — auto-telemetry at 2kHz

z2423-z2424 failed because:
  - Polling at 20Hz gives ~7-28 snapshots per image, no real temporal dynamics
  - FPGA neuron activity is either silent (old config) or uncorrelated (physics config)
  - Ridge regression on 384 noisy FPGA features + 10 MLP logits → worse than MLP alone

New approach:
  - Auto-telemetry at 2000Hz → ~100 temporal samples per image (50ms window)
  - Collect spike count TIME SERIES, not just final snapshot
  - Extract proper temporal features: spike rate slope, variance, temporal correlation
  - Feed image via MAC as temporal signal (one row every ~2ms)
  - Key insight: the VALUE is in temporal dynamics, not in spatial spike patterns

Architecture:
  1. MLP computes logits + ISA signals (as before)
  2. FPGA auto-pushes telemetry at 2kHz
  3. Image rows fed sequentially via MAC (28 rows × ~2ms = 56ms)
  4. Collect ~100+ telemetry frames during feeding
  5. Extract: spike_rate_slope, spike_rate_var, vmem_trend, temporal_corr per neuron group
  6. Ridge readout on MLP logits + temporal features → prediction
"""
import sys, os, time, json, socket
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
    return logits, isa

from fpga_host_eth import FPGAEthBridge

print("=" * 60)
print("z2425: TEMPORAL STREAM PIPELINE (2kHz auto-telemetry)")
print("=" * 60)

fpga = FPGAEthBridge(local_port=7723, timeout=2.0)
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

# Configure FPGA with PHYSICS defaults
fpga.set_leak_cond(0x0004)       # τ≈210ms
fpga.set_threshold_raw(0x8000)   # 0.5V
fpga.set_base_exc_raw(0x0333)    # moderate avalanche
fpga.set_bias_gain_raw(0x0800)   # gentle MAC injection
VG = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
for grp, vg in VG.items():
    fpga.set_vg_batch(grp*32, [vg]*32)
time.sleep(0.5)

# Test auto-telemetry
print("\nTesting auto-telemetry...")
fpga.enable_auto_telemetry(2000)  # 2kHz
time.sleep(0.1)
frames = []
for _ in range(20):
    t = fpga.recv_auto_telemetry(timeout=0.02)
    if t:
        frames.append(t)
fpga.disable_auto_telemetry()
print(f"  Received {len(frames)} frames in 0.4s (expected ~8-20 at 2kHz)")
if len(frames) == 0:
    print("  WARNING: No auto-telemetry frames! Falling back to polling.")
    USE_AUTO = False
else:
    USE_AUTO = True
    # Check what's in the frames
    sc = np.array([f['spike_counts'] for f in frames])
    print(f"  Spike counts range: {sc.min():.0f}-{sc.max():.0f}")
    print(f"  Active neurons per frame: {[(f['spike_counts'] > 0).sum() for f in frames[:5]]}")

# ================================================================
# Temporal feature extraction
# ================================================================
def extract_temporal_features(spike_series, vmem_series, n_groups=4):
    """
    Extract temporal features from a time series of telemetry frames.

    spike_series: (T, 128) spike counts over time
    vmem_series: (T, 128) membrane voltages over time
    n_groups: divide 128 neurons into groups for aggregation

    Returns feature vector.
    """
    T = len(spike_series)
    if T < 3:
        return np.zeros(n_groups * 10, dtype=np.float32)

    # Group neurons (32 per group — matches VG groups)
    feats = []
    group_size = 128 // n_groups

    for g in range(n_groups):
        s = g * group_size
        e = s + group_size

        # Spike dynamics per group
        spk = spike_series[:, s:e].sum(axis=1).astype(np.float32)  # total spikes per timestep
        vm = vmem_series[:, s:e].mean(axis=1).astype(np.float32)   # mean vmem per timestep

        # Temporal features:
        # 1. Mean spike rate
        feats.append(spk.mean())
        # 2. Spike rate variance (temporal variability)
        feats.append(spk.var())
        # 3. Spike rate slope (linear trend) — does activity increase/decrease?
        t_axis = np.arange(T, dtype=np.float32)
        if spk.std() > 0:
            slope = np.polyfit(t_axis, spk, 1)[0]
        else:
            slope = 0.0
        feats.append(slope)
        # 4. Temporal autocorrelation lag-1
        if T > 1 and spk.std() > 0:
            acf1 = np.corrcoef(spk[:-1], spk[1:])[0, 1]
            if np.isnan(acf1): acf1 = 0.0
        else:
            acf1 = 0.0
        feats.append(acf1)
        # 5. Mean vmem
        feats.append(vm.mean())
        # 6. Vmem variance
        feats.append(vm.var())
        # 7. Vmem trend
        if vm.std() > 0:
            vm_slope = np.polyfit(t_axis, vm, 1)[0]
        else:
            vm_slope = 0.0
        feats.append(vm_slope)
        # 8. Spike-vmem correlation
        if spk.std() > 0 and vm.std() > 0:
            sv_corr = np.corrcoef(spk, vm)[0, 1]
            if np.isnan(sv_corr): sv_corr = 0.0
        else:
            sv_corr = 0.0
        feats.append(sv_corr)
        # 9. Max spike burst (max consecutive timesteps with spikes)
        binary = (spk > 0).astype(int)
        max_burst = 0
        curr = 0
        for b in binary:
            if b: curr += 1; max_burst = max(max_burst, curr)
            else: curr = 0
        feats.append(float(max_burst) / T)
        # 10. First-spike latency (normalized)
        first_spike = np.argmax(spk > 0) / T if (spk > 0).any() else 1.0
        feats.append(first_spike)

    return np.array(feats, dtype=np.float32)


# ================================================================
# Main experiment
# ================================================================
N = 1500  # fewer samples but richer features
n_train = 1000
n_test = N - n_train

print(f"\nUsing {N} samples, temp={check_temp():.0f}°C")
print("Running MLP...")
all_logits = np.zeros((N, 10), dtype=np.float32)
all_isa = np.zeros((N, 6), dtype=np.float32)
for i in range(N):
    all_logits[i], all_isa[i] = mlp_with_isa(X_test[i])
acc_mlp = (all_logits.argmax(axis=1) == y_test[:N]).mean() * 100
acc_mlp_ridge = ridge_classify(all_logits[:n_train], y_test[:n_train],
                                all_logits[n_train:N], y_test[n_train:N])
print(f"MLP: {acc_mlp:.2f}% (argmax), {acc_mlp_ridge:.2f}% (ridge)\n")

# Temporal features: 4 groups × 10 features = 40
n_feat = 40
fpga_temporal = np.zeros((N, n_feat), dtype=np.float32)
# Also collect snapshot features for comparison
fpga_snapshot = np.zeros((N, 256), dtype=np.float32)  # spike_counts + vmem

print(f"--- Streaming experiment (auto={USE_AUTO}) ---")

if USE_AUTO:
    fpga.enable_auto_telemetry(2000)
    time.sleep(0.1)
    # Drain any buffered frames
    for _ in range(50):
        fpga.recv_auto_telemetry(timeout=0.002)

t0 = time.time()
frames_per_image = []

for i in range(N):
    # Reset periodically
    if i % 50 == 0:
        if USE_AUTO:
            fpga.disable_auto_telemetry()
        fpga.set_kill(True)
        time.sleep(0.005)
        fpga.set_kill(False)
        time.sleep(0.005)
        if USE_AUTO:
            fpga.enable_auto_telemetry(2000)
            time.sleep(0.01)
            # Drain
            for _ in range(20):
                fpga.recv_auto_telemetry(timeout=0.002)

    image = X_test[i].reshape(28, 28)
    isa = all_isa[i]
    isa_mod = 1.0 + (isa[1] - all_isa[:N, 1].mean()) / (all_isa[:N, 1].std() + 1e-10) * 0.2

    # Feed image rows and collect temporal telemetry
    spike_series = []
    vmem_series = []

    for row in range(28):
        row_mean = image[row].mean()
        modulated = row_mean * np.clip(isa_mod, 0.7, 1.5)
        fpga.set_mac_signal(float(np.clip(modulated, 0, 0.99)))

        # Collect any available telemetry
        if USE_AUTO:
            for _ in range(3):  # try to get 2-3 frames per row
                t = fpga.recv_auto_telemetry(timeout=0.001)
                if t:
                    spike_series.append(t['spike_counts'].copy())
                    vmem_series.append(t['vmem'].copy())

        time.sleep(0.002)  # ~2ms per row = 56ms total per image

    # Collect remaining frames
    if USE_AUTO:
        for _ in range(10):
            t = fpga.recv_auto_telemetry(timeout=0.002)
            if t:
                spike_series.append(t['spike_counts'].copy())
                vmem_series.append(t['vmem'].copy())

    frames_per_image.append(len(spike_series))

    # Extract temporal features
    if len(spike_series) >= 3:
        spike_arr = np.array(spike_series)
        vmem_arr = np.array(vmem_series)
        fpga_temporal[i] = extract_temporal_features(spike_arr, vmem_arr)

    # Also get final snapshot
    if not USE_AUTO:
        telem = fpga.read_telemetry()
        if telem:
            fpga_snapshot[i, :128] = telem['spike_counts'].astype(np.float32)
            fpga_snapshot[i, 128:] = telem['vmem'].astype(np.float32)
    elif len(spike_series) > 0:
        fpga_snapshot[i, :128] = spike_series[-1].astype(np.float32)
        fpga_snapshot[i, 128:] = vmem_series[-1].astype(np.float32)

    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (N - i - 1)
        avg_frames = np.mean(frames_per_image[-100:])
        t = check_temp()
        print(f"  {i+1}/{N} ({elapsed:.0f}s, ETA {eta:.0f}s, {t:.0f}°C, {avg_frames:.1f} frames/img)")
        if t > 75:
            wait_cool()

elapsed = time.time() - t0
avg_frames = np.mean(frames_per_image)
print(f"  Done ({elapsed:.1f}s, {elapsed/N*1000:.0f}ms/sample, avg {avg_frames:.1f} frames/img)")

if USE_AUTO:
    fpga.disable_auto_telemetry()

# Normalize features
def normalize_features(feat):
    for j in range(feat.shape[1]):
        col = feat[:, j]
        std = col.std()
        if std > 1e-2:
            feat[:, j] = (col - col.mean()) / std
        else:
            feat[:, j] = 0.0
    return feat

fpga_temporal = normalize_features(fpga_temporal.copy())
fpga_snapshot = normalize_features(fpga_snapshot.copy())

# ================================================================
# Results
# ================================================================
print("\n--- Results ---")

# 1. MLP + temporal features
feat_t = np.concatenate([all_logits[:N], fpga_temporal], axis=1)
acc_temporal = ridge_classify(feat_t[:n_train], y_test[:n_train],
                               feat_t[n_train:N], y_test[n_train:N])
print(f"  MLP + temporal (40d):  {acc_temporal:.2f}%  (Δ={acc_temporal-acc_mlp_ridge:+.2f}pp)")

# 2. MLP + snapshot features
feat_s = np.concatenate([all_logits[:N], fpga_snapshot], axis=1)
acc_snapshot = ridge_classify(feat_s[:n_train], y_test[:n_train],
                               feat_s[n_train:N], y_test[n_train:N])
print(f"  MLP + snapshot (256d): {acc_snapshot:.2f}%  (Δ={acc_snapshot-acc_mlp_ridge:+.2f}pp)")

# 3. MLP + temporal + ISA
feat_ti = np.concatenate([all_logits[:N], all_isa[:N], fpga_temporal], axis=1)
acc_temp_isa = ridge_classify(feat_ti[:n_train], y_test[:n_train],
                               feat_ti[n_train:N], y_test[n_train:N])
print(f"  MLP + temporal + ISA:  {acc_temp_isa:.2f}%  (Δ={acc_temp_isa-acc_mlp_ridge:+.2f}pp)")

# 4. MLP + all FPGA features
feat_all = np.concatenate([all_logits[:N], all_isa[:N], fpga_temporal, fpga_snapshot], axis=1)
acc_all = ridge_classify(feat_all[:n_train], y_test[:n_train],
                          feat_all[n_train:N], y_test[n_train:N])
print(f"  MLP + ALL (312d):      {acc_all:.2f}%  (Δ={acc_all-acc_mlp_ridge:+.2f}pp)")

# 5. Temporal alone
acc_t_alone = ridge_classify(fpga_temporal[:n_train], y_test[:n_train],
                              fpga_temporal[n_train:N], y_test[n_train:N])
print(f"  Temporal alone (40d):  {acc_t_alone:.2f}%")

# 6. Snapshot alone
acc_s_alone = ridge_classify(fpga_snapshot[:n_train], y_test[:n_train],
                              fpga_snapshot[n_train:N], y_test[n_train:N])
print(f"  Snapshot alone (256d): {acc_s_alone:.2f}%")

# Feature quality
n_useful_t = (fpga_temporal.var(axis=0) > 0.01).sum()
n_useful_s = (fpga_snapshot.var(axis=0) > 0.01).sum()
print(f"\n  Useful features: temporal={n_useful_t}/40, snapshot={n_useful_s}/256")
print(f"  Frames per image: min={min(frames_per_image)}, max={max(frames_per_image)}, mean={avg_frames:.1f}")

print(f"\n  MLP baseline: {acc_mlp:.2f}% (argmax), {acc_mlp_ridge:.2f}% (ridge)")

best_acc = max(acc_temporal, acc_snapshot, acc_temp_isa, acc_all)
best_name = ['temporal', 'snapshot', 'temporal+ISA', 'ALL'][
    [acc_temporal, acc_snapshot, acc_temp_isa, acc_all].index(best_acc)]
delta = best_acc - acc_mlp_ridge
print(f"  BEST: {best_name} = {best_acc:.2f}% ({delta:+.2f}pp vs ridge)")
if best_acc > acc_mlp:
    print(f"  >>> BEATS MLP ARGMAX ({acc_mlp:.2f}%) by {best_acc-acc_mlp:+.2f}pp <<<")

results = {
    'mlp_argmax': float(acc_mlp),
    'mlp_ridge': float(acc_mlp_ridge),
    'mlp_temporal': float(acc_temporal),
    'mlp_snapshot': float(acc_snapshot),
    'mlp_temporal_isa': float(acc_temp_isa),
    'mlp_all': float(acc_all),
    'temporal_alone': float(acc_t_alone),
    'snapshot_alone': float(acc_s_alone),
    'avg_frames_per_image': float(avg_frames),
    'n_samples': N,
    'time_s': float(elapsed),
    'use_auto_telem': USE_AUTO,
}
with open(f'{base}/results/z2425_temporal_stream.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2425_temporal_stream.json")

fpga.close()
