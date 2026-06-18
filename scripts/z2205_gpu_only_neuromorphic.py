#!/usr/bin/env python3
"""z2205_gpu_only_neuromorphic.py — GPU-Only Neuromorphic Computation (No FPGA)

THE STANDALONE CLAIM: The GPU's firmware physics — thermal registers, power rail
dynamics, DVFS regulation, PM table state — can serve as a neuromorphic computing
substrate WITHOUT any external neural hardware (no FPGA).

Architecture: "Firmware Neurons"
  Each of 8 GPU firmware channels acts as a virtual neuron:
    Channel 0-1: hwmon power1_average (VRM switching, 2 time-lagged versions)
    Channel 2-3: SMN thermal registers (0x59800, 0x59804)
    Channel 4-5: PM table regulatory state (offsets 0x10, 0x14)
    Channel 6:   GPU clock (DVFS dynamics)
    Channel 7:   System timing jitter (perf_counter noise)

  Input encoding: Run GPU tensor operations whose SIZE is modulated by the input
  signal. This creates a causal chain:
    input → torch.randn(size) → GPU power/thermal → firmware registers → readout

  The input LITERALLY changes the GPU's physics, and we read the physics back.

3 Conditions:
  FIRMWARE_NEURONS: Input → GPU workload → firmware readout → classify
  SOFTWARE_NEURONS: Input → 8-node ESN (software) → classify (ceiling)
  RANDOM_READOUT:   Ignore input, read firmware at random times → classify (floor)

Task: 3-class waveform classification (180 trials, 30 steps)

Tests T299-T306:
  T299: FIRMWARE > RANDOM (firmware dynamics are input-dependent)
  T300: FIRMWARE accuracy > 40% (genuinely useful)
  T301: FIRMWARE / SOFTWARE ratio > 0.5 (at least half as good as ESN)
  T302: Firmware channel diversity: std of per-channel MI > 0.01
  T303: Temporal memory: ACF(lag=2) of power channel > 0.3
  T304: Power PSD slope in [-2.5, -0.3] (native 1/f)
  T305: Input-firmware mutual information > 0.05 bits
  T306: At least 3 firmware channels show input sensitivity > noise floor

Hardware: AMD gfx1151 GPU (NO FPGA needed!)
Firmware: ryzen_smu kernel module for SMN/PM table
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

N_CHANNELS = 8
STEPS_PER_TRIAL = 30
SAMPLE_HZ = 10  # Slower for firmware dynamics to propagate
N_TRIALS = 180
N_FOLDS = 5
BASE_WORKLOAD_SIZE = 128  # base matrix size for GPU workload

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"
SMN_PATH = "/sys/kernel/ryzen_smu_drv/smn"
PM_TABLE_PATH = "/sys/kernel/ryzen_smu_drv/pm_table"
SMN_ADDRS = [0x59800, 0x59804]
PM_OFFSETS = [0x10, 0x14]


def read_firmware_state():
    """Read all 8 firmware neuron channels. NO processing."""
    channels = np.zeros(N_CHANNELS)

    # Ch 0-1: Power rail (2 consecutive reads = time-lagged)
    try:
        with open(HWMON_POWER) as f:
            channels[0] = int(f.read().strip()) / 1e6
        time.sleep(0.005)
        with open(HWMON_POWER) as f:
            channels[1] = int(f.read().strip()) / 1e6
    except:
        pass

    # Ch 2-3: SMN thermal registers
    for i, addr in enumerate(SMN_ADDRS):
        try:
            with open(SMN_PATH, 'rb') as f:
                f.seek(addr)
                channels[2 + i] = struct.unpack('<I', f.read(4))[0]
        except:
            pass

    # Ch 4-5: PM table
    for i, off in enumerate(PM_OFFSETS):
        try:
            with open(PM_TABLE_PATH, 'rb') as f:
                f.seek(off)
                channels[4 + i] = struct.unpack('<I', f.read(4))[0]
        except:
            pass

    # Ch 6: GPU clock
    try:
        import glob as gl
        paths = gl.glob('/sys/class/drm/card*/device/gpu_metrics')
        if paths:
            with open(paths[0], 'rb') as f:
                data = f.read()
            if len(data) >= 30:
                channels[6] = struct.unpack('<H', data[26:28])[0]
    except:
        pass

    # Ch 7: Timing jitter
    t0 = time.perf_counter_ns()
    _ = os.getpid()  # minimal syscall
    t1 = time.perf_counter_ns()
    channels[7] = (t1 - t0)

    return channels


def generate_gpu_workload(input_val, torch):
    """Generate GPU workload proportional to input signal.
    The input PHYSICALLY changes what the GPU does."""
    # Map input [-1, 1] to workload size [64, 512]
    size = max(64, min(512, int(BASE_WORKLOAD_SIZE + input_val * 200)))

    # Run actual GPU computation — this changes power, thermal, clock
    m = torch.randn(size, size, device='cuda')
    result = m @ m.T
    # Force synchronization so the power draw is immediate
    torch.cuda.synchronize()

    return size


def generate_waveforms(n_trials, steps, hz, rng):
    labels = np.repeat([0, 1, 2], (n_trials + 2) // 3)[:n_trials]
    rng.shuffle(labels)
    t = np.linspace(0, steps / hz, steps)
    signals = []
    for label in labels:
        freq = rng.uniform(0.3, 1.5)
        if label == 0:
            sig = np.sin(2 * np.pi * freq * t)
        elif label == 1:
            sig = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        else:
            sig = np.sign(np.sin(2 * np.pi * freq * t))
        signals.append(sig)
    return np.array(signals), labels


class EchoStateNetwork:
    """Standard ESN for comparison (software baseline)."""
    def __init__(self, n_input, n_reservoir, n_output, spectral_radius=0.9, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.n_reservoir = n_reservoir
        self.W_in = rng.randn(n_reservoir, n_input) * 0.5
        W = rng.randn(n_reservoir, n_reservoir)
        eigenvalues = np.linalg.eigvals(W)
        W *= spectral_radius / np.max(np.abs(eigenvalues))
        self.W = W
        self.state = np.zeros(n_reservoir)

    def step(self, u):
        self.state = np.tanh(self.W_in @ u + self.W @ self.state)
        return self.state.copy()

    def reset(self):
        self.state = np.zeros(self.n_reservoir)


def run_firmware_trial(signal, torch):
    """Run one trial: input → GPU workload → firmware readout."""
    all_channels = []

    for step_i in range(len(signal)):
        inp = signal[step_i]

        # Generate GPU workload based on input (causal!)
        _ = generate_gpu_workload(inp, torch)

        # Read firmware state
        time.sleep(1.0 / SAMPLE_HZ)
        channels = read_firmware_state()
        all_channels.append(channels)

    ch_arr = np.array(all_channels)  # (steps, 8)
    # Pool: mean, max, std, last
    pooled = np.concatenate([ch_arr.mean(0), ch_arr.max(0), ch_arr.std(0), ch_arr[-1]])
    return pooled, ch_arr


def run_esn_trial(signal, esn):
    """Run ESN on same signal (software baseline)."""
    esn.reset()
    states = []
    for step_i in range(len(signal)):
        state = esn.step(np.array([signal[step_i]]))
        states.append(state)
    st = np.array(states)
    pooled = np.concatenate([st.mean(0), st.max(0), st.std(0), st[-1]])
    return pooled


def run_random_trial(signal):
    """Read firmware at random — control for input-independent dynamics."""
    all_channels = []
    for _ in range(len(signal)):
        # NO input-dependent workload!
        time.sleep(1.0 / SAMPLE_HZ)
        channels = read_firmware_state()
        all_channels.append(channels)
    ch_arr = np.array(all_channels)
    pooled = np.concatenate([ch_arr.mean(0), ch_arr.max(0), ch_arr.std(0), ch_arr[-1]])
    return pooled, ch_arr


def compute_psd_slope(signal, fs=10):
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=fs, nperseg=min(64, len(signal)))
    mask = freqs > 0
    if mask.sum() < 2:
        return 0.0
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask] + 1e-20)
    return float(np.polyfit(log_f, log_p, 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    args = parser.parse_args()

    print("=" * 65)
    print("z2205: GPU-Only Neuromorphic Computation (No FPGA)")
    print("=" * 65)
    print("  Can GPU firmware physics alone perform neuromorphic computation?")

    try:
        import torch
        assert torch.cuda.is_available()
        print(f"[HW] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[ERR] PyTorch/CUDA: {e}")
        sys.exit(1)

    smn_ok = os.path.exists(SMN_PATH)
    pm_ok = os.path.exists(PM_TABLE_PATH)
    print(f"[HW] SMN: {smn_ok}, PM table: {pm_ok}")

    rng = np.random.RandomState(42)
    signals, labels = generate_waveforms(args.n_trials, STEPS_PER_TRIAL, SAMPLE_HZ, rng)

    # ─── Condition 1: FIRMWARE_NEURONS ───
    print(f"\n[1/3] FIRMWARE_NEURONS ({args.n_trials} trials)...")
    fw_features = []
    fw_channels_all = []

    for ti in range(args.n_trials):
        if (ti + 1) % 30 == 0:
            print(f"    trial {ti+1}/{args.n_trials}")
        feat, ch_arr = run_firmware_trial(signals[ti], torch)
        fw_features.append(feat)
        fw_channels_all.append(ch_arr)

    X_fw = np.array(fw_features)

    # ─── Condition 2: SOFTWARE_NEURONS (ESN) ───
    print(f"\n[2/3] SOFTWARE_NEURONS (ESN, {args.n_trials} trials)...")
    esn = EchoStateNetwork(n_input=1, n_reservoir=8, n_output=3,
                           spectral_radius=0.9, rng=rng)
    esn_features = []
    for ti in range(args.n_trials):
        feat = run_esn_trial(signals[ti], esn)
        esn_features.append(feat)
    X_esn = np.array(esn_features)

    # ─── Condition 3: RANDOM_READOUT ───
    print(f"\n[3/3] RANDOM_READOUT ({args.n_trials} trials)...")
    rand_features = []
    rand_channels_all = []

    for ti in range(args.n_trials):
        if (ti + 1) % 30 == 0:
            print(f"    trial {ti+1}/{args.n_trials}")
        feat, ch_arr = run_random_trial(signals[ti])
        rand_features.append(feat)
        rand_channels_all.append(ch_arr)

    X_rand = np.array(rand_features)

    # ─── Classify all conditions ───
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    results = {}
    for name, X in [('FIRMWARE_NEURONS', X_fw), ('SOFTWARE_NEURONS', X_esn), ('RANDOM_READOUT', X_rand)]:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in skf.split(X, labels):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X_tr, labels[train_idx])
            accs.append(clf.score(X_te, labels[test_idx]))

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        print(f"\n  {name}: {mean_acc:.4f} ± {std_acc:.4f}")
        results[name] = {
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'fold_accs': [float(a) for a in accs],
        }

    # ─── Per-channel analysis ───
    print("\n  Per-channel analysis (FIRMWARE_NEURONS):")
    ch_names = ['Power_0', 'Power_1', 'SMN_0', 'SMN_1', 'PM_0', 'PM_1', 'Clock', 'Jitter']
    channel_stats = {}

    for ci in range(N_CHANNELS):
        # Collect all channel values across trials
        ch_vals = np.array([fw_channels_all[ti][:, ci] for ti in range(len(fw_channels_all))])

        # Per-channel MI with labels
        from sklearn.metrics import mutual_info_score
        ch_mean = ch_vals.mean(axis=1)
        ch_binned = np.digitize(ch_mean, np.linspace(ch_mean.min(), ch_mean.max() + 1e-9, 10))
        mi = mutual_info_score(ch_binned, labels[:len(ch_binned)])

        # Input sensitivity: correlation between input signal amplitude and channel response
        input_amps = np.array([np.std(signals[ti]) for ti in range(len(signals))])
        ch_stds = ch_vals.std(axis=1)
        corr = float(np.corrcoef(input_amps[:len(ch_stds)], ch_stds)[0, 1]) if len(ch_stds) > 2 else 0

        # ACF at lag 2
        all_traces = ch_vals.flatten()
        if len(all_traces) > 10 and np.std(all_traces) > 1e-10:
            centered = all_traces - all_traces.mean()
            acf = np.correlate(centered, centered, mode='full')
            acf = acf[len(acf)//2:]
            acf2 = float(acf[2] / (acf[0] + 1e-10)) if len(acf) > 2 else 0
        else:
            acf2 = 0

        channel_stats[ch_names[ci]] = {
            'mi': float(mi),
            'input_sensitivity': float(corr),
            'acf_lag2': acf2,
        }
        print(f"    {ch_names[ci]:<10}: MI={mi:.4f}, sensitivity={corr:.3f}, ACF(2)={acf2:.3f}")

    # Power channel PSD
    power_trace = np.array([fw_channels_all[ti][:, 0] for ti in range(len(fw_channels_all))]).flatten()
    psd_slope = compute_psd_slope(power_trace)
    print(f"\n  Power PSD slope: {psd_slope:.3f}")

    # Power ACF(2)
    power_acf2 = channel_stats['Power_0']['acf_lag2']

    # ─── Tests T299-T306 ───
    print("\n" + "=" * 65)
    print("  Tests T299-T306: GPU-Only Neuromorphic Computation")
    print("=" * 65)

    fw_acc = results['FIRMWARE_NEURONS']['accuracy_mean']
    sw_acc = results['SOFTWARE_NEURONS']['accuracy_mean']
    rd_acc = results['RANDOM_READOUT']['accuracy_mean']

    tests = {}

    t299 = fw_acc > rd_acc
    tests['T299'] = {'pass': t299, 'firmware': fw_acc, 'random': rd_acc}
    print(f"\n  T299 fw>random:         {'PASS' if t299 else 'FAIL'}  FW={fw_acc:.4f} vs RANDOM={rd_acc:.4f}")

    t300 = fw_acc > 0.40
    tests['T300'] = {'pass': t300, 'firmware': fw_acc}
    print(f"  T300 useful:            {'PASS' if t300 else 'FAIL'}  FW={fw_acc:.4f} (need >0.40)")

    ratio = fw_acc / max(sw_acc, 0.01)
    t301 = ratio > 0.5
    tests['T301'] = {'pass': t301, 'ratio': float(ratio), 'firmware': fw_acc, 'software': sw_acc}
    print(f"  T301 fw/esn>0.5:        {'PASS' if t301 else 'FAIL'}  ratio={ratio:.3f} (FW={fw_acc:.4f}/ESN={sw_acc:.4f})")

    mi_values = [channel_stats[ch]['mi'] for ch in ch_names]
    mi_std = float(np.std(mi_values))
    t302 = mi_std > 0.01
    tests['T302'] = {'pass': t302, 'mi_std': mi_std}
    print(f"  T302 channel_diversity: {'PASS' if t302 else 'FAIL'}  MI_std={mi_std:.4f}")

    t303 = power_acf2 > 0.3
    tests['T303'] = {'pass': t303, 'acf2': power_acf2}
    print(f"  T303 temporal_memory:   {'PASS' if t303 else 'FAIL'}  ACF(2)={power_acf2:.3f}")

    t304 = -2.5 < psd_slope < -0.3
    tests['T304'] = {'pass': t304, 'slope': psd_slope}
    print(f"  T304 native_1f:         {'PASS' if t304 else 'FAIL'}  slope={psd_slope:.3f}")

    # T305: overall firmware-input MI
    all_fw_means = np.array([fw_channels_all[ti].mean() for ti in range(len(fw_channels_all))])
    fw_binned = np.digitize(all_fw_means, np.linspace(all_fw_means.min(), all_fw_means.max() + 1e-9, 10))
    overall_mi = mutual_info_score(fw_binned, labels[:len(fw_binned)])
    t305 = overall_mi > 0.05
    tests['T305'] = {'pass': t305, 'mi': float(overall_mi)}
    print(f"  T305 input_mi:          {'PASS' if t305 else 'FAIL'}  MI={overall_mi:.4f}")

    # T306: channels with sensitivity above noise floor
    noise_floor = 0.05
    n_sensitive = sum(1 for ch in ch_names if abs(channel_stats[ch]['input_sensitivity']) > noise_floor)
    t306 = n_sensitive >= 3
    tests['T306'] = {'pass': t306, 'n_sensitive': n_sensitive}
    print(f"  T306 multi_channel:     {'PASS' if t306 else 'FAIL'}  {n_sensitive} channels sensitive")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/8 PASS")

    # ─── Save ───
    out = {
        'experiment': 'z2205_gpu_only_neuromorphic',
        'n_trials': args.n_trials,
        'conditions': results,
        'channel_stats': channel_stats,
        'power_psd_slope': psd_slope,
        'tests': tests,
        'n_pass': n_pass,
        'n_total': 8,
    }
    out_path = RESULTS / 'z2205_gpu_only_neuromorphic.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results: {out_path}")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Accuracy comparison
        conds = ['FIRMWARE_NEURONS', 'SOFTWARE_NEURONS', 'RANDOM_READOUT']
        accs_m = [results[c]['accuracy_mean'] for c in conds]
        accs_s = [results[c]['accuracy_std'] for c in conds]
        axes[0,0].bar(conds, accs_m, yerr=accs_s,
                      color=['#e74c3c', '#3498db', '#95a5a6'], capsize=3)
        axes[0,0].axhline(0.333, ls='--', color='gray', alpha=0.5, label='chance')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('GPU-Only Neuromorphic vs Controls')
        axes[0,0].tick_params(axis='x', rotation=15)
        axes[0,0].legend()

        # Per-channel MI
        axes[0,1].bar(ch_names, mi_values, color='#e74c3c')
        axes[0,1].set_ylabel('Mutual Information (bits)')
        axes[0,1].set_title('Per-Channel Label MI')
        axes[0,1].tick_params(axis='x', rotation=45)

        # Power trace example
        if fw_channels_all:
            example_power = fw_channels_all[0][:, 0]
            axes[1,0].plot(example_power, color='#e74c3c', linewidth=1)
            axes[1,0].set_xlabel('Step')
            axes[1,0].set_ylabel('Power (W)')
            axes[1,0].set_title('Example: Power Rail During Trial')

        # Input sensitivity
        sens = [channel_stats[ch]['input_sensitivity'] for ch in ch_names]
        axes[1,1].bar(ch_names, sens, color='#e74c3c')
        axes[1,1].axhline(noise_floor, ls='--', color='green', alpha=0.5, label='noise floor')
        axes[1,1].axhline(-noise_floor, ls='--', color='green', alpha=0.5)
        axes[1,1].set_ylabel('Input Sensitivity (r)')
        axes[1,1].set_title('Channel Sensitivity to Input Signal')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend()

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2205_gpu_only_neuromorphic.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Figure: {fig_path}")
    except Exception as e:
        print(f"  [WARN] Figure failed: {e}")


if __name__ == '__main__':
    main()
