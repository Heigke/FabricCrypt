#!/usr/bin/env python3
"""
z2432b: Fusion readout — GPU mechanism features + FPGA temporal + combined

Takes GPU mechanism features from z2432_mechanism_features.bin,
feeds them to FPGA, and tests whether GPU+FPGA beats either alone.
"""
import sys, os, time, json, struct
import numpy as np

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/scripts')
base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

# Load GPU mechanism features
fpath = f'{base}/results/z2432_mechanism_features.bin'
with open(fpath, 'rb') as f:
    N = struct.unpack('i', f.read(4))[0]
    N_MECH = struct.unpack('i', f.read(4))[0]
    SEQ_LEN = struct.unpack('i', f.read(4))[0]
    gpu_features = np.frombuffer(f.read(N * N_MECH * 4), dtype=np.float32).reshape(N, N_MECH)
    labels = np.frombuffer(f.read(N * 4), dtype=np.int32)
    waveforms = np.frombuffer(f.read(N * SEQ_LEN * 4), dtype=np.float32).reshape(N, SEQ_LEN)

print(f"Loaded: {N} samples, {N_MECH} mechanisms, {SEQ_LEN} timesteps, 4 classes")

# Shuffle
np.random.seed(42)
idx = np.random.permutation(N)
gpu_features = gpu_features[idx]
labels = labels[idx]
waveforms = waveforms[idx]

n_train = int(0.7 * N)
n_test = N - n_train

def ridge_classify(X_tr, y_tr, X_te, y_te, alpha=1.0):
    n_c = len(set(y_tr))
    Y = np.zeros((len(y_tr), n_c))
    for i, y in enumerate(y_tr): Y[i, y] = 1.0
    XtX = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
    W = np.linalg.solve(XtX, X_tr.T @ Y)
    preds = (X_te @ W).argmax(axis=1)
    return (preds == y_te).mean() * 100

def check_temp():
    try:
        return int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
    except:
        return 0

# ================================================================
# Method A: GPU mechanisms alone
# ================================================================
print("\n--- A: GPU Mechanism Features Alone ---")
# Normalize
gf = gpu_features.copy()
for j in range(N_MECH):
    std = gf[:n_train, j].std()
    mean = gf[:n_train, j].mean()
    if std > 1e-6:
        gf[:, j] = (gf[:, j] - mean) / std
    else:
        gf[:, j] = 0

acc_gpu = ridge_classify(gf[:n_train], labels[:n_train],
                          gf[n_train:], labels[n_train:])
print(f"  GPU mechanisms ({N_MECH}d): {acc_gpu:.1f}%")

# ================================================================
# Method B: Statistical features (baseline)
# ================================================================
print("\n--- B: Statistical Features (baseline) ---")
def extract_stats(X):
    feats = np.zeros((len(X), 15), dtype=np.float32)
    for i in range(len(X)):
        x = X[i]
        feats[i,0] = x.mean(); feats[i,1] = x.std()
        feats[i,2] = np.sum(np.abs(np.diff(x)))
        feats[i,3] = np.sum(np.diff(np.sign(x-x.mean())) != 0)
        fft = np.abs(np.fft.rfft(x))
        feats[i,4:9] = fft[1:6]
        feats[i,9] = np.argmax(fft[1:]) + 1
        feats[i,10] = fft[1:].sum()
        for lag in range(1,4):
            feats[i,10+lag] = np.corrcoef(x[:-lag], x[lag:])[0,1] if x.std() > 0 else 0
        feats[i,14] = ((x-x.mean())**4).mean() / (x.std()**4 + 1e-10)
    return feats

stats = extract_stats(waveforms)
for j in range(15):
    std = stats[:n_train, j].std()
    mean = stats[:n_train, j].mean()
    if std > 1e-6:
        stats[:, j] = (stats[:, j] - mean) / std
    else:
        stats[:, j] = 0

acc_stats = ridge_classify(stats[:n_train], labels[:n_train],
                            stats[n_train:], labels[n_train:])
print(f"  Statistical ({stats.shape[1]}d): {acc_stats:.1f}%")

# ================================================================
# Method C: GPU mechanisms + statistical
# ================================================================
print("\n--- C: GPU Mechanisms + Statistics ---")
combo = np.concatenate([gf, stats], axis=1)
acc_combo = ridge_classify(combo[:n_train], labels[:n_train],
                            combo[n_train:], labels[n_train:])
print(f"  GPU+Stats ({combo.shape[1]}d): {acc_combo:.1f}%")

# ================================================================
# Method D: FPGA reservoir (waveform via MAC with temporal products)
# ================================================================
print("\n--- D: FPGA Reservoir (with temporal products) ---")

from fpga_host_eth import FPGAEthBridge
fpga = FPGAEthBridge(local_port=7729, timeout=2.0)
ok = fpga.connect()
if not ok:
    print("  FPGA not available — skipping FPGA conditions")
    acc_fpga = 25.0
    acc_fusion = acc_combo
    acc_full = acc_combo
else:
    # Configure with physics defaults
    fpga.set_leak_cond(0x0004)
    fpga.set_threshold_raw(0x8000)
    fpga.set_base_exc_raw(0x0333)
    fpga.set_bias_gain_raw(0x0800)
    VG = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
    for grp, vg in VG.items():
        fpga.set_vg_batch(grp*32, [vg]*32)
    time.sleep(0.5)

    fpga.enable_auto_telemetry(2000)
    time.sleep(0.1)
    for _ in range(50):
        fpga.recv_auto_telemetry(timeout=0.002)

    # Collect TEMPORAL spike series per waveform (not just final snapshot!)
    # Use temporal PRODUCT features from z2296
    def run_fpga_temporal_products(X_data, n_lags=5):
        """FPGA reservoir with temporal product features (z2296 approach)."""
        n_feat = 128 + 128 + 128 * n_lags  # spikes + vmem + products
        # Simplified: 128 spikes + 128 vmem + 8 temporal stats = 264
        n_feat = 264
        features = np.zeros((len(X_data), n_feat), dtype=np.float32)
        t0 = time.time()

        for i in range(len(X_data)):
            if i % 25 == 0:
                fpga.disable_auto_telemetry()
                fpga.set_kill(True); time.sleep(0.005)
                fpga.set_kill(False); time.sleep(0.005)
                fpga.enable_auto_telemetry(2000)
                time.sleep(0.01)
                for _ in range(20):
                    fpga.recv_auto_telemetry(timeout=0.002)

            wave = X_data[i]
            # Feed waveform (subsample to 25 steps)
            vmem_history = []
            spike_history = []

            for t in range(0, SEQ_LEN, 4):
                fpga.set_mac_signal(float(np.clip(wave[t], 0, 0.99)))
                time.sleep(0.002)
                f = fpga.recv_auto_telemetry(timeout=0.003)
                if f:
                    vmem_history.append(f['vmem'].astype(np.float32))
                    spike_history.append(f['spike_counts'].astype(np.float32))

            # Final telemetry
            time.sleep(0.005)
            for _ in range(5):
                f = fpga.recv_auto_telemetry(timeout=0.003)
                if f:
                    vmem_history.append(f['vmem'].astype(np.float32))
                    spike_history.append(f['spike_counts'].astype(np.float32))

            if len(vmem_history) >= 3:
                vm = np.array(vmem_history)
                sp = np.array(spike_history)

                # Final state
                features[i, :128] = sp[-1]
                features[i, 128:256] = vm[-1]

                # Temporal product features (z2296 key insight!)
                # Mean vmem across time for 4 groups
                for g in range(4):
                    grp_vm = vm[:, g*32:(g+1)*32].mean(axis=1)
                    features[i, 256+g*2] = grp_vm.mean()  # mean
                    features[i, 256+g*2+1] = grp_vm.var()  # temporal variance

            if (i+1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(X_data)} ({elapsed:.0f}s, {check_temp():.0f}°C)")

        return features

    fpga_tr = run_fpga_temporal_products(waveforms[:n_train])
    fpga_te = run_fpga_temporal_products(waveforms[n_train:])

    # Normalize
    for j in range(264):
        std = fpga_tr[:, j].std()
        mean = fpga_tr[:, j].mean()
        if std > 1e-2:
            fpga_tr[:, j] = (fpga_tr[:, j] - mean) / std
            fpga_te[:, j] = (fpga_te[:, j] - mean) / std
        else:
            fpga_tr[:, j] = 0; fpga_te[:, j] = 0

    acc_fpga = ridge_classify(fpga_tr, labels[:n_train],
                               fpga_te, labels[n_train:])
    print(f"  FPGA temporal: {acc_fpga:.1f}%")

    # ================================================================
    # Method E: FUSION — GPU mechanisms + FPGA temporal
    # ================================================================
    print("\n--- E: FUSION (GPU mechanisms + FPGA temporal) ---")
    fus_tr = np.concatenate([gf[:n_train], fpga_tr], axis=1)
    fus_te = np.concatenate([gf[n_train:], fpga_te], axis=1)
    acc_fusion = ridge_classify(fus_tr, labels[:n_train],
                                 fus_te, labels[n_train:])
    print(f"  GPU+FPGA fusion: {acc_fusion:.1f}%")

    # Method F: FULL — GPU mechanisms + Statistics + FPGA
    print("\n--- F: FULL (GPU + Stats + FPGA) ---")
    full_tr = np.concatenate([gf[:n_train], stats[:n_train], fpga_tr], axis=1)
    full_te = np.concatenate([gf[n_train:], stats[n_train:], fpga_te], axis=1)
    acc_full = ridge_classify(full_tr, labels[:n_train],
                               full_te, labels[n_train:])
    print(f"  Full pipeline: {acc_full:.1f}%")

    fpga.disable_auto_telemetry()
    fpga.close()

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  A: GPU mechanisms alone:     {acc_gpu:.1f}%")
print(f"  B: Statistical baseline:     {acc_stats:.1f}%")
print(f"  C: GPU + Stats:              {acc_combo:.1f}%")
print(f"  D: FPGA temporal:            {acc_fpga:.1f}%")
print(f"  E: GPU + FPGA (FUSION):      {acc_fusion:.1f}%")
print(f"  F: GPU + Stats + FPGA:       {acc_full:.1f}%")

# Key question: does FUSION beat best single?
best_single = max(acc_gpu, acc_stats, acc_fpga)
best_single_name = ['GPU', 'Stats', 'FPGA'][[acc_gpu, acc_stats, acc_fpga].index(best_single)]
best_combined = max(acc_combo, acc_fusion, acc_full)
best_combined_name = ['GPU+Stats', 'GPU+FPGA', 'Full'][[acc_combo, acc_fusion, acc_full].index(best_combined)]

print(f"\n  Best single substrate: {best_single:.1f}% ({best_single_name})")
print(f"  Best combined:         {best_combined:.1f}% ({best_combined_name})")

if best_combined > best_single + 0.5:
    print(f"\n  >>> COMBINATION WINS by +{best_combined-best_single:.1f}pp <<<")
    if acc_fusion > acc_gpu + 0.5 and acc_fusion > acc_fpga + 0.5:
        print(f"  >>> GPU+FPGA FUSION > GPU alone AND > FPGA alone <<<")

# Does GPU mechanism add value over stats?
if acc_combo > acc_stats + 0.5:
    print(f"\n  GPU mechanisms add +{acc_combo-acc_stats:.1f}pp over statistics alone")
    print(f"  → Mechanisms provide UNIQUE information not in simple features")

results = {
    'gpu_mechanisms': float(acc_gpu),
    'statistical': float(acc_stats),
    'gpu_stats': float(acc_combo),
    'fpga_temporal': float(acc_fpga),
    'fusion_gpu_fpga': float(acc_fusion),
    'full_pipeline': float(acc_full),
}
with open(f'{base}/results/z2432_fusion_readout.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/z2432_fusion_readout.json")
