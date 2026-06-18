#!/usr/bin/env python3
"""
z2458: The Access Level Proof

Demonstrate that ANALOG INTERMEDIATE ACCESS gives measurable advantage
over DIGITAL-ONLY OUTPUT — using the same hardware, same data, same readout.

The FPGA NCU (Neuromorphic Compute Unit) is equivalent to:
  GPU ALU: MAC → bias → ReLU → digital output
  FPGA:    MAC → integration → threshold → spike (digital) + vmem (analog)

Three readout levels (all from the SAME FPGA run):
  Level 1: SPIKE ONLY (= what GPU gives you: binary post-activation)
  Level 2: VMEM ONLY (= analog intermediate state: what PSP hides)
  Level 3: SPIKE + VMEM + TEMPORAL PRODUCTS (= full access + feature engineering)

Also compare against:
  Level 0: SOFTWARE ESN (no hardware access at all)

If Level 2 >> Level 1 → analog intermediate access IS the key
If Level 3 >> Level 2 → feature engineering on top of access adds more
If Level 1 ≈ Level 0 → hardware adds nothing beyond software

Tasks:
  - Waveform classification (4-class: sine/square/sawtooth/triangle)
  - Memory capacity (MC)
  - Temporal XOR at multiple delays
  - Mackey-Glass chaotic prediction (if time permits)
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/scripts')
base = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy'

from fpga_host_eth import FPGAEthBridge

def check_temp():
    try: return int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
    except: return 0

def wait_cool(target=55):
    t = check_temp()
    if t > 70:
        print(f"  Cooling ({t:.0f}°C)...", end='', flush=True)
        while t > target: time.sleep(3); t = check_temp()
        print(f" {t:.0f}°C OK")

# ================================================================
# Connect FPGA
# ================================================================
print("=" * 70)
print("z2458: ACCESS LEVEL PROOF")
print("  Same hardware, same data — does analog access help?")
print("=" * 70)

fpga = FPGAEthBridge(local_port=7732, timeout=2.0)
ok = fpga.connect()
if not ok:
    print("FPGA connection failed!"); sys.exit(1)

# Configure
fpga.set_leak_cond(0x2000)
fpga.set_threshold_raw(0x20000)
fpga.set_base_exc_raw(0x0080)
fpga.set_bias_gain_raw(0x4000)
VG = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
for g, v in VG.items():
    fpga.set_vg_batch(g*32, [v]*32)
time.sleep(1.0)
fpga.enable_auto_telemetry(2000)
time.sleep(0.1)
for _ in range(50): fpga.recv_auto_telemetry(timeout=0.002)

# ================================================================
# Generate temporal benchmark signals
# ================================================================
np.random.seed(42)
N_STEPS = 3000
dt = 0.05  # 50ms per step = 20Hz

# Input: random uniform [0, 1]
u = np.random.rand(N_STEPS).astype(np.float32)

# Waveform classification targets
# Every 100 steps = one waveform sample
WAVE_LEN = 100
N_WAVES = N_STEPS // WAVE_LEN
wave_labels = np.zeros(N_WAVES, dtype=int)
for i in range(N_WAVES):
    cls = i % 4
    wave_labels[i] = cls
    t = np.linspace(0, 2*np.pi, WAVE_LEN)
    freq = 2 + (i % 7) * 0.5
    if cls == 0: sig = np.sin(freq * t)
    elif cls == 1: sig = np.sign(np.sin(freq * t))
    elif cls == 2: sig = 2*(freq*t/(2*np.pi) % 1) - 1
    else: sig = 2*np.abs(2*(freq*t/(2*np.pi) % 1) - 1) - 1
    u[i*WAVE_LEN:(i+1)*WAVE_LEN] = (sig * 0.4 + 0.5).clip(0, 1)

# XOR targets at various delays
xor_binary = (u > 0.5).astype(int)
xor_targets = {}
for delay in [1, 3, 5]:
    xor_targets[delay] = np.zeros(N_STEPS, dtype=int)
    for t in range(delay, N_STEPS):
        xor_targets[delay][t] = xor_binary[t] ^ xor_binary[t - delay]

# Memory capacity targets
mc_targets = {}
for delay in range(1, 21):
    mc_targets[delay] = np.zeros(N_STEPS, dtype=float)
    for t in range(delay, N_STEPS):
        mc_targets[delay][t] = u[t - delay]

# ================================================================
# Run FPGA: collect BOTH spike counts AND vmem at each timestep
# ================================================================
print(f"\nRunning FPGA ({N_STEPS} steps)...")
spikes_all = np.zeros((N_STEPS, 128), dtype=np.float32)
vmem_all = np.zeros((N_STEPS, 128), dtype=np.float32)

t0 = time.time()
for t in range(N_STEPS):
    if t % 50 == 0 and t > 0:
        fpga.disable_auto_telemetry()
        fpga.set_kill(True); time.sleep(0.003)
        fpga.set_kill(False); time.sleep(0.003)
        fpga.enable_auto_telemetry(2000)
        time.sleep(0.005)
        for _ in range(10): fpga.recv_auto_telemetry(timeout=0.002)

    fpga.set_mac_signal(float(u[t]))
    time.sleep(0.003)

    telem = fpga.recv_auto_telemetry(timeout=0.01)
    if telem:
        spikes_all[t] = telem['spike_counts'].astype(np.float32)
        vmem_all[t] = telem['vmem'].astype(np.float32)

    if (t+1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  {t+1}/{N_STEPS} ({elapsed:.0f}s, {check_temp():.0f}°C)")
        if check_temp() > 75: wait_cool()

fpga.disable_auto_telemetry()
fpga.close()
elapsed = time.time() - t0
print(f"  Done ({elapsed:.1f}s)")

# Stats
print(f"  Spikes: mean={spikes_all.mean():.2f} max={spikes_all.max():.0f} active_frac={(spikes_all>0).mean():.2f}")
print(f"  Vmem: mean={vmem_all.mean():.2f} std={vmem_all.std():.4f} range=[{vmem_all.min():.4f}, {vmem_all.max():.4f}]")

# ================================================================
# Build readout features for each ACCESS LEVEL
# ================================================================
def build_features(spikes, vmem, level):
    """Build features at different access levels."""
    T, N = spikes.shape

    if level == 'spike_only':
        # Level 1: Only spike counts (= digital post-activation output)
        return spikes.copy()

    elif level == 'vmem_only':
        # Level 2: Only membrane voltage (= analog intermediate state)
        return vmem.copy()

    elif level == 'full_access':
        # Level 3: Spikes + vmem + temporal products
        feats = [spikes, vmem]
        # Temporal products: vmem(t) × vmem(t-τ) for τ=1,2,3,5
        for tau in [1, 2, 3, 5]:
            product = np.zeros_like(vmem)
            product[tau:] = vmem[tau:] * vmem[:-tau]
            feats.append(product)
        # Spike-vmem cross: spike(t) × vmem(t)
        feats.append(spikes * vmem)
        return np.concatenate(feats, axis=1)

    elif level == 'software_esn':
        # Level 0: Software ESN (no hardware at all)
        N_res = 128
        np.random.seed(42)
        W_in = np.random.randn(N_res, 1).astype(np.float32) * 0.5
        W_res = np.random.randn(N_res, N_res).astype(np.float32)
        eigvals = np.abs(np.linalg.eigvals(W_res))
        W_res *= 0.9 / eigvals.max()
        states = np.zeros((T, N_res), dtype=np.float32)
        x = np.zeros(N_res, dtype=np.float32)
        for t in range(T):
            x = 0.7 * x + 0.3 * np.tanh(W_in.ravel() * u[t] + W_res @ x)
            states[t] = x
        return states

def ridge_score(X_train, y_train, X_test, y_test, alpha=1.0):
    """Ridge regression R² score."""
    XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
    XtY = X_train.T @ y_train.reshape(-1, 1)
    try:
        W = np.linalg.solve(XtX, XtY)
    except:
        return 0.0
    pred = X_test @ W
    ss_res = ((y_test.reshape(-1,1) - pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    return max(0, 1 - ss_res / (ss_tot + 1e-10))

def ridge_classify(X_train, y_train, X_test, y_test, alpha=1.0):
    """Ridge classification accuracy."""
    n_c = len(set(y_train))
    Y = np.zeros((len(y_train), n_c))
    for i, y in enumerate(y_train): Y[i, y] = 1.0
    XtX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
    try:
        W = np.linalg.solve(XtX, X_train.T @ Y)
    except:
        return 0.0
    preds = (X_test @ W).argmax(axis=1)
    return (preds == y_test).mean() * 100

# ================================================================
# Evaluate each access level
# ================================================================
levels = ['software_esn', 'spike_only', 'vmem_only', 'full_access']
level_labels = {
    'software_esn': 'L0: Software ESN (no HW)',
    'spike_only':   'L1: Spike only (digital output)',
    'vmem_only':    'L2: Vmem only (analog intermediate)',
    'full_access':  'L3: Full access (analog + temporal)',
}

# Train/test split
split = N_STEPS // 2

results = {}
print(f"\n{'='*70}")
print("ACCESS LEVEL COMPARISON")
print(f"{'='*70}")

for level in levels:
    feat = build_features(spikes_all, vmem_all, level)
    feat_tr = feat[:split]
    feat_te = feat[split:]

    # Normalize
    for j in range(feat.shape[1]):
        m, s = feat_tr[:, j].mean(), feat_tr[:, j].std()
        if s > 1e-6:
            feat_tr[:, j] = (feat_tr[:, j] - m) / s
            feat_te[:, j] = (feat_te[:, j] - m) / s
        else:
            feat_tr[:, j] = 0; feat_te[:, j] = 0

    r = {'features': feat.shape[1]}

    # Waveform classification
    n_wave_tr = split // WAVE_LEN
    n_wave_te = N_WAVES - n_wave_tr
    wave_feat_tr = np.array([feat_tr[i*WAVE_LEN:(i+1)*WAVE_LEN].mean(axis=0) for i in range(n_wave_tr)])
    wave_feat_te = np.array([feat_te[(i-n_wave_tr)*WAVE_LEN:((i-n_wave_tr)+1)*WAVE_LEN].mean(axis=0) for i in range(n_wave_tr, N_WAVES)])
    wave_acc = ridge_classify(wave_feat_tr, wave_labels[:n_wave_tr],
                               wave_feat_te, wave_labels[n_wave_tr:N_WAVES])
    r['wave4'] = wave_acc

    # XOR at delays
    for delay in [1, 3, 5]:
        target = xor_targets[delay]
        acc = ridge_classify(feat_tr[delay:], target[delay:split],
                              feat_te[delay:], target[split+delay:]) if split+delay < N_STEPS else 50.0
        r[f'xor{delay}'] = acc

    # Memory capacity
    mc = 0
    for delay in range(1, 21):
        target = mc_targets[delay]
        r2 = ridge_score(feat_tr[delay:], target[delay:split],
                          feat_te[delay:], target[split+delay:]) if split+delay < N_STEPS else 0
        mc += max(0, r2)
    r['mc20'] = mc

    results[level] = r

# Print results
print(f"\n{'Level':>35} {'Feat':>5} {'Wave4':>7} {'XOR1':>7} {'XOR3':>7} {'XOR5':>7} {'MC(20)':>8}")
print("-" * 85)
for level in levels:
    r = results[level]
    print(f"{level_labels[level]:>35} {r['features']:>5} {r['wave4']:>6.1f}% {r['xor1']:>6.1f}% "
          f"{r['xor3']:>6.1f}% {r['xor5']:>6.1f}% {r['mc20']:>7.2f}")

# Compute deltas
print(f"\n--- Analog Access Value (L2 - L1) ---")
for metric in ['wave4', 'xor1', 'xor3', 'xor5', 'mc20']:
    l1 = results['spike_only'][metric]
    l2 = results['vmem_only'][metric]
    unit = '%' if 'xor' in metric or 'wave' in metric else ''
    print(f"  {metric:>6}: spike={l1:.1f}{unit} vmem={l2:.1f}{unit} Δ={l2-l1:+.1f}")

print(f"\n--- Full Access Value (L3 - L2) ---")
for metric in ['wave4', 'xor1', 'xor3', 'xor5', 'mc20']:
    l2 = results['vmem_only'][metric]
    l3 = results['full_access'][metric]
    unit = '%' if 'xor' in metric or 'wave' in metric else ''
    print(f"  {metric:>6}: vmem={l2:.1f}{unit} full={l3:.1f}{unit} Δ={l3-l2:+.1f}")

print(f"\n--- Hardware vs Software (L2 - L0) ---")
for metric in ['wave4', 'xor1', 'xor3', 'xor5', 'mc20']:
    l0 = results['software_esn'][metric]
    l2 = results['vmem_only'][metric]
    unit = '%' if 'xor' in metric or 'wave' in metric else ''
    print(f"  {metric:>6}: SW={l0:.1f}{unit} vmem={l2:.1f}{unit} Δ={l2-l0:+.1f}")

# ================================================================
# The argument to AMD/ARM
# ================================================================
print(f"\n{'='*70}")
print("ARGUMENT TO HARDWARE ENGINEERS")
print(f"{'='*70}")
l0 = results['software_esn']
l1 = results['spike_only']
l2 = results['vmem_only']
l3 = results['full_access']
print(f"""
Your ALU computes: MAC → activation → digital output
We only see the final digital output (post-ReLU).

But INSIDE, the accumulator holds analog intermediate state
that your firmware/PSP hides from us.

On our FPGA proof-of-concept (same compute flow, full access):

  Software only (no HW):   MC={l0['mc20']:.1f}  XOR5={l0['xor5']:.1f}%  Wave={l0['wave4']:.1f}%
  Digital output only:      MC={l1['mc20']:.1f}  XOR5={l1['xor5']:.1f}%  Wave={l1['wave4']:.1f}%
  Analog intermediate:      MC={l2['mc20']:.1f}  XOR5={l2['xor5']:.1f}%  Wave={l2['wave4']:.1f}%
  Full access + temporal:   MC={l3['mc20']:.1f}  XOR5={l3['xor5']:.1f}%  Wave={l3['wave4']:.1f}%

The gap between 'digital only' and 'analog intermediate' =
  what you LOSE by hiding intermediate state.

The gap between 'software' and 'analog intermediate' =
  what hardware access UNIQUELY provides.
""")

with open(f'{base}/results/z2458_access_level_proof.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"Saved to results/z2458_access_level_proof.json")
