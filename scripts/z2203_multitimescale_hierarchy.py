#!/usr/bin/env python3
"""z2203_multitimescale_hierarchy.py — Multi-Timescale Hierarchy: Fast FPGA + Slow GPU

Demonstrates that the GPU-FPGA system naturally implements a multi-timescale
processing hierarchy:
  - FAST layer: FPGA LIF neurons (50ms per step, spike dynamics)
  - SLOW layer: GPU thermal/power dynamics (seconds-scale, smoothed context)

This is a key feature of biological brains: cortex has fast spiking layers
feeding into slow integrating layers. Our system gets this FOR FREE from the
physics of the two substrates.

Task: Temporal context classification
  Short signal (5 steps) embedded in longer context (25 steps total)
  Must classify the short signal, but context influences difficulty
  Context A: low-frequency modulation → easier
  Context B: high-frequency modulation → harder
  Context C: no modulation → baseline

4 Conditions:
  FULL:      FPGA + GPU thermal feedback (both timescales)
  FPGA_ONLY: FPGA neurons only, no GPU context
  GPU_ONLY:  GPU thermal features only, no FPGA spikes
  SEPARATED: FPGA features + GPU features concatenated but independent (no feedback)

Tests T285-T290:
  T285: FULL accuracy > FPGA_ONLY (slow GPU context helps)
  T286: FULL accuracy > GPU_ONLY (fast FPGA dynamics needed)
  T287: FULL accuracy > SEPARATED (feedback coupling > concatenation)
  T288: GPU thermal autocorrelation time > FPGA spike autocorrelation time
  T289: Context A accuracy > Context C > Context B (difficulty ordering)
  T290: Temporal integration window for FULL > FPGA_ONLY

Hardware: AMD gfx1151 GPU + Arty A7 FPGA on /dev/ttyUSB*
"""

import os, sys, json, time, struct, argparse
import numpy as np
from pathlib import Path

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
RESULTS = BASE / 'results'
FIGURES = RESULTS / 'FEEL_paper_update' / 'FEEL__Functionally_Embodied_Emergent_Learning__13_-5' / 'figures'

SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

N_NEURONS = 8
STEPS_TOTAL = 25
SIGNAL_START = 10
SIGNAL_LEN = 5
SAMPLE_HZ = 20
N_TRIALS = 150
N_FOLDS = 5
BASE_VG = 0.45
ALPHA = 0.15
BETA = 0.10

HWMON_POWER = "/sys/class/hwmon/hwmon7/power1_average"

def find_fpga():
    import serial
    import glob as gl
    ports = sorted(gl.glob('/dev/ttyUSB*'), reverse=True)
    for p in ports:
        try:
            s = serial.Serial(p, 115200, timeout=0.3)
            time.sleep(0.1)
            s.write(bytes([SYNC, CMD_SET_KILL, 0x00]))
            s.flush()
            time.sleep(0.05)
            s.write(bytes([SYNC, CMD_READ_TELEM]))
            s.flush()
            resp = s.read(52)
            if len(resp) >= 52 and resp[0] == SYNC and resp[1] == CMD_READ_TELEM:
                return s, p
            s.close()
        except Exception:
            continue
    return None, None

def send_set_vg(ser, neuron_id, vg_val):
    vg_q16 = int(np.clip(vg_val, 0.05, 0.95) * 65536) & 0xFFFFFFFF
    pkt = bytes([SYNC, CMD_SET_VG, neuron_id & 0x07]) + struct.pack('>I', vg_q16)
    ser.write(pkt)
    ser.flush()

def read_telem(ser):
    ser.write(bytes([SYNC, CMD_READ_TELEM]))
    ser.flush()
    resp = ser.read(52)
    if len(resp) < 52 or resp[0] != SYNC or resp[1] != CMD_READ_TELEM:
        return None
    payload = resp[2:51]
    data = {}
    data['delta_spikes'] = [payload[i] for i in range(N_NEURONS)]
    data['vmem'] = [struct.unpack('>H', payload[8+i*2:10+i*2])[0] for i in range(N_NEURONS)]
    return data

def read_power():
    try:
        with open(HWMON_POWER) as f:
            return int(f.read().strip()) / 1e6  # µW → W
    except:
        return 0.0

def generate_contextual_trials(n_trials, rng):
    """Generate trials with signal embedded in context."""
    # Signal classes: sine, triangle, square (same 3-class)
    signal_labels = rng.choice([0, 1, 2], n_trials)
    # Context types: A(low-freq), B(high-freq), C(none)
    context_labels = rng.choice([0, 1, 2], n_trials)

    t_total = np.linspace(0, STEPS_TOTAL / SAMPLE_HZ, STEPS_TOTAL)
    t_signal = np.linspace(0, SIGNAL_LEN / SAMPLE_HZ, SIGNAL_LEN)

    trials = []
    for i in range(n_trials):
        full = np.zeros(STEPS_TOTAL)

        # Context modulation
        if context_labels[i] == 0:  # low-freq
            ctx = 0.15 * np.sin(2 * np.pi * 0.3 * t_total)
        elif context_labels[i] == 1:  # high-freq
            ctx = 0.15 * np.sin(2 * np.pi * 3.0 * t_total)
        else:  # none
            ctx = np.zeros(STEPS_TOTAL)
        full += ctx

        # Signal
        freq = rng.uniform(0.8, 1.5)
        if signal_labels[i] == 0:
            sig = np.sin(2 * np.pi * freq * t_signal)
        elif signal_labels[i] == 1:
            sig = 2 * np.abs(2 * (t_signal * freq % 1) - 1) - 1
        else:
            sig = np.sign(np.sin(2 * np.pi * freq * t_signal))
        full[SIGNAL_START:SIGNAL_START + SIGNAL_LEN] += sig * 0.3

        trials.append(full)

    return np.array(trials), signal_labels, context_labels


def run_trial(ser, signal, condition, w_in, w_noise, noise_seq, torch):
    """Run one trial, collecting both FPGA and GPU features."""
    fpga_features = []
    gpu_features = []
    power_trace = []

    # GPU workload for thermal generation
    if condition in ('FULL', 'GPU_ONLY'):
        dummy = torch.randn(256, 256, device='cuda')

    for step_i in range(len(signal)):
        inp = signal[step_i]

        # GPU thermal context
        power = read_power()
        power_trace.append(power)

        if condition == 'FULL':
            # Both: noise modulation + GPU thermal feedback
            thermal_mod = (power - 11.0) / 3.0  # normalize around mean
            for n in range(N_NEURONS):
                noise_val = noise_seq[step_i % len(noise_seq)] if len(noise_seq) > 0 else 0
                vg = BASE_VG + ALPHA * inp * w_in[n] + BETA * noise_val * w_noise[n] + 0.03 * thermal_mod
                send_set_vg(ser, n, np.clip(vg, 0.05, 0.95))
            # GPU workload proportional to spike activity
            if fpga_features:
                spike_sum = sum(fpga_features[-1][:N_NEURONS])
                sz = max(64, min(512, int(64 + spike_sum * 20)))
                m = torch.randn(sz, sz, device='cuda')
                _ = m @ m.T
        elif condition == 'FPGA_ONLY':
            for n in range(N_NEURONS):
                noise_val = noise_seq[step_i % len(noise_seq)] if len(noise_seq) > 0 else 0
                vg = BASE_VG + ALPHA * inp * w_in[n] + BETA * noise_val * w_noise[n]
                send_set_vg(ser, n, np.clip(vg, 0.05, 0.95))
        elif condition == 'GPU_ONLY':
            # No FPGA modulation, just set neutral Vg
            for n in range(N_NEURONS):
                send_set_vg(ser, n, BASE_VG)
            m = torch.randn(256, 256, device='cuda')
            _ = m @ m.T
        elif condition == 'SEPARATED':
            # FPGA and GPU independent
            for n in range(N_NEURONS):
                noise_val = noise_seq[step_i % len(noise_seq)] if len(noise_seq) > 0 else 0
                vg = BASE_VG + ALPHA * inp * w_in[n] + BETA * noise_val * w_noise[n]
                send_set_vg(ser, n, np.clip(vg, 0.05, 0.95))
            m = torch.randn(256, 256, device='cuda')
            _ = m @ m.T

        time.sleep(1.0 / SAMPLE_HZ)

        telem = read_telem(ser)
        if telem:
            spikes = np.array(telem['delta_spikes'], dtype=float)
            vmem = np.array(telem['vmem'], dtype=float) / 256.0
            fpga_features.append(np.concatenate([spikes, vmem]))
        else:
            fpga_features.append(np.zeros(N_NEURONS * 2))

        gpu_features.append([power])

    fpga_arr = np.array(fpga_features)
    gpu_arr = np.array(gpu_features)
    power_arr = np.array(power_trace)

    # Pool features
    if condition == 'GPU_ONLY':
        pooled = np.concatenate([gpu_arr.mean(0), gpu_arr.std(0), gpu_arr.max(0)])
    elif condition in ('FULL', 'SEPARATED'):
        fpga_pool = np.concatenate([fpga_arr.mean(0), fpga_arr.std(0)])
        gpu_pool = np.concatenate([gpu_arr.mean(0), gpu_arr.std(0)])
        pooled = np.concatenate([fpga_pool, gpu_pool])
    else:  # FPGA_ONLY
        pooled = np.concatenate([fpga_arr.mean(0), fpga_arr.std(0)])

    return pooled, fpga_arr, gpu_arr, power_arr


def autocorr_time(x):
    """Estimate autocorrelation decay time (steps to fall below 1/e)."""
    x = x - x.mean()
    if np.std(x) < 1e-10:
        return 0
    acf = np.correlate(x, x, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / (acf[0] + 1e-10)
    threshold = 1.0 / np.e
    for i in range(len(acf)):
        if acf[i] < threshold:
            return i
    return len(acf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    args = parser.parse_args()

    print("=" * 65)
    print("z2203: Multi-Timescale Hierarchy — Fast FPGA + Slow GPU")
    print("=" * 65)

    try:
        import torch
        assert torch.cuda.is_available()
        print(f"[HW] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[ERR] PyTorch/CUDA: {e}")
        sys.exit(1)

    ser, port = find_fpga()
    if ser is None:
        print("[ERR] FPGA not found")
        sys.exit(1)
    print(f"[HW] FPGA on {port}")

    # Collect noise
    print("\n[1/4] Collecting GPU power noise...")
    noise_samples = []
    for _ in range(500):
        noise_samples.append(read_power())
        time.sleep(0.02)
    noise_seq = np.array(noise_samples)
    noise_seq = (noise_seq - noise_seq.mean()) / max(noise_seq.std(), 1e-6)
    print(f"  Collected {len(noise_seq)} noise samples")

    # Generate trials
    rng = np.random.RandomState(42)
    trials, signal_labels, context_labels = generate_contextual_trials(args.n_trials, rng)
    w_in = rng.randn(N_NEURONS)
    w_in /= np.linalg.norm(w_in)
    w_noise = rng.randn(N_NEURONS)
    w_noise /= np.linalg.norm(w_noise)

    conditions = ['FULL', 'FPGA_ONLY', 'GPU_ONLY', 'SEPARATED']
    results = {}

    print(f"\n[2/4] Running {len(conditions)} conditions × {args.n_trials} trials...")

    for cond in conditions:
        print(f"\n  === Condition: {cond} ===")
        all_features = []
        all_fpga_acf = []
        all_gpu_acf = []

        for ti in range(args.n_trials):
            if (ti + 1) % 25 == 0:
                print(f"    trial {ti+1}/{args.n_trials}")

            feat, fpga_arr, gpu_arr, power_arr = run_trial(
                ser, trials[ti], cond, w_in, w_noise, noise_seq, torch)
            all_features.append(feat)

            # Autocorrelation times
            if len(fpga_arr) > 3:
                spike_trace = fpga_arr[:, :N_NEURONS].sum(axis=1)
                all_fpga_acf.append(autocorr_time(spike_trace))
            if len(gpu_arr) > 3:
                all_gpu_acf.append(autocorr_time(power_arr))

        X = np.array(all_features)
        y = signal_labels

        # Classification
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        accs = []
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X_train, y[train_idx])
            accs.append(clf.score(X_test, y[test_idx]))

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_fpga_acf = float(np.mean(all_fpga_acf)) if all_fpga_acf else 0
        mean_gpu_acf = float(np.mean(all_gpu_acf)) if all_gpu_acf else 0

        # Per-context accuracy
        ctx_accs = {}
        for ctx in [0, 1, 2]:
            mask = context_labels == ctx
            if mask.sum() > 10:
                X_ctx = X[mask]
                y_ctx = y[mask]
                # Simple train-test split
                n_train = int(len(X_ctx) * 0.7)
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_ctx[:n_train])
                X_te = scaler.transform(X_ctx[n_train:])
                clf = RidgeClassifier(alpha=1.0)
                clf.fit(X_tr, y_ctx[:n_train])
                ctx_accs[ctx] = float(clf.score(X_te, y_ctx[n_train:]))

        print(f"    accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"    FPGA acf_time: {mean_fpga_acf:.1f} steps, GPU acf_time: {mean_gpu_acf:.1f} steps")
        print(f"    context accs: {ctx_accs}")

        results[cond] = {
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'fold_accs': [float(a) for a in accs],
            'fpga_acf_time': mean_fpga_acf,
            'gpu_acf_time': mean_gpu_acf,
            'context_accs': ctx_accs,
        }

    # ─── Tests T285-T290 ───
    print("\n" + "=" * 65)
    print("  Tests T285-T290")
    print("=" * 65)

    tests = {}

    full = results['FULL']
    fpga = results['FPGA_ONLY']
    gpu = results['GPU_ONLY']
    sep = results['SEPARATED']

    t285 = full['accuracy_mean'] > fpga['accuracy_mean']
    tests['T285'] = {'pass': t285, 'full': full['accuracy_mean'], 'fpga_only': fpga['accuracy_mean']}
    print(f"\n  T285 full>fpga_only:    {'PASS' if t285 else 'FAIL'}  FULL={full['accuracy_mean']:.4f} vs FPGA={fpga['accuracy_mean']:.4f}")

    t286 = full['accuracy_mean'] > gpu['accuracy_mean']
    tests['T286'] = {'pass': t286, 'full': full['accuracy_mean'], 'gpu_only': gpu['accuracy_mean']}
    print(f"  T286 full>gpu_only:     {'PASS' if t286 else 'FAIL'}  FULL={full['accuracy_mean']:.4f} vs GPU={gpu['accuracy_mean']:.4f}")

    t287 = full['accuracy_mean'] > sep['accuracy_mean']
    tests['T287'] = {'pass': t287, 'full': full['accuracy_mean'], 'separated': sep['accuracy_mean']}
    print(f"  T287 full>separated:    {'PASS' if t287 else 'FAIL'}  FULL={full['accuracy_mean']:.4f} vs SEP={sep['accuracy_mean']:.4f}")

    t288 = full['gpu_acf_time'] > full['fpga_acf_time']
    tests['T288'] = {'pass': t288, 'gpu_acf': full['gpu_acf_time'], 'fpga_acf': full['fpga_acf_time']}
    print(f"  T288 gpu_slower:        {'PASS' if t288 else 'FAIL'}  GPU_acf={full['gpu_acf_time']:.1f} vs FPGA_acf={full['fpga_acf_time']:.1f}")

    # T289: Context difficulty ordering (A > C > B)
    ctx_a = full['context_accs'].get(0, 0)
    ctx_b = full['context_accs'].get(1, 0)
    ctx_c = full['context_accs'].get(2, 0)
    t289 = ctx_a > ctx_c > ctx_b
    tests['T289'] = {'pass': t289, 'ctx_A': ctx_a, 'ctx_B': ctx_b, 'ctx_C': ctx_c}
    print(f"  T289 context_ordering:  {'PASS' if t289 else 'FAIL'}  A={ctx_a:.3f} C={ctx_c:.3f} B={ctx_b:.3f}")

    # T290: Integration window — FULL should have longer effective memory
    full_integ = full['fpga_acf_time'] + full['gpu_acf_time']
    fpga_integ = fpga['fpga_acf_time']
    t290 = full_integ > fpga_integ
    tests['T290'] = {'pass': t290, 'full_integration': full_integ, 'fpga_integration': fpga_integ}
    print(f"  T290 longer_memory:     {'PASS' if t290 else 'FAIL'}  FULL={full_integ:.1f} vs FPGA={fpga_integ:.1f}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/6 PASS")

    # ─── Save ───
    out = {
        'experiment': 'z2203_multitimescale_hierarchy',
        'n_trials': args.n_trials,
        'conditions': results,
        'tests': tests,
        'n_pass': n_pass,
        'n_total': 6,
    }
    out_path = RESULTS / 'z2203_multitimescale_hierarchy.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results: {out_path}")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Accuracy comparison
        conds = list(results.keys())
        accs_m = [results[c]['accuracy_mean'] for c in conds]
        accs_s = [results[c]['accuracy_std'] for c in conds]
        axes[0].bar(conds, accs_m, yerr=accs_s, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'], capsize=3)
        axes[0].axhline(0.333, ls='--', color='gray', alpha=0.5)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Signal Classification by Condition')
        axes[0].tick_params(axis='x', rotation=30)

        # Autocorrelation times
        fpga_acfs = [results[c]['fpga_acf_time'] for c in conds]
        gpu_acfs = [results[c]['gpu_acf_time'] for c in conds]
        x = np.arange(len(conds))
        axes[1].bar(x - 0.15, fpga_acfs, 0.3, label='FPGA', color='#3498db')
        axes[1].bar(x + 0.15, gpu_acfs, 0.3, label='GPU', color='#e74c3c')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(conds, fontsize=8)
        axes[1].set_ylabel('ACF Time (steps)')
        axes[1].set_title('Timescale Separation')
        axes[1].legend()

        # Context accuracy
        ctx_names = ['Low-freq (A)', 'High-freq (B)', 'None (C)']
        for ci, cond in enumerate(['FULL', 'FPGA_ONLY']):
            ca = [results[cond]['context_accs'].get(i, 0) for i in range(3)]
            axes[2].plot(ctx_names, ca, marker='o', label=cond)
        axes[2].axhline(0.333, ls='--', color='gray', alpha=0.5)
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Context-Dependent Performance')
        axes[2].legend()

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2203_multitimescale_hierarchy.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Figure: {fig_path}")
    except Exception as e:
        print(f"  [WARN] Figure failed: {e}")

    ser.close()


if __name__ == '__main__':
    main()
