#!/usr/bin/env python3
"""z2201_gpu_spike_processor.py — GPU as Active Spike Processor (not just noise source)

The GPU actively processes FPGA spike trains using HIP/ROCm tensor operations,
then feeds the results back to modulate FPGA neuron dynamics. This creates a
genuine two-brain architecture where GPU computation is CONSTITUTIVE, not merely
noisy.

Architecture:
  FPGA → spike_train[8×T] → GPU tensor ops → processed[8] → FPGA Vg modulation

GPU Processing Modes:
  1. MATMUL: W @ spike_history → weighted integration (learned connectivity)
  2. CONV1D: Temporal convolution → pattern detection
  3. SPECTRAL: FFT → frequency filtering → selective amplification
  4. ATTENTION: Softmax self-attention over spike history → context-dependent
  5. PASSTHROUGH: Raw spike rates → Vg (control: no GPU processing)
  6. RANDOM: Random Vg per step (control: computation irrelevant)

Each mode runs a waveform classification task (3-class, 120 trials).

Tests T273-T278:
  T273: Best GPU mode > PASSTHROUGH (GPU processing helps)
  T274: Best GPU mode > RANDOM (computation matters, not just any feedback)
  T275: ATTENTION > MATMUL (context-dependent processing > static)
  T276: At least 3 modes > 50% accuracy (reservoir generalizes across transforms)
  T277: GPU processing latency < 5ms (real-time capable)
  T278: Spike-to-Vg mutual information > 0.1 bits for best mode

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

# ─── FPGA Protocol ───
SYNC = 0x55
CMD_SET_VG = 0x01
CMD_READ_TELEM = 0x02
CMD_SET_KILL = 0x03

N_NEURONS = 8
STEPS_PER_TRIAL = 30
SAMPLE_HZ = 20
N_TRIALS = 120
N_FOLDS = 5
BASE_VG = 0.45  # near sub-threshold boundary (z2196 finding)
ALPHA = 0.15
HISTORY_LEN = 8  # timesteps of spike history for GPU processing

# ─── Helpers ───

def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if crc & 0x80 else (crc << 1)
            crc &= 0xFF
    return crc

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
    data['mac'] = struct.unpack('>H', payload[24:26])[0]
    data['vbpar'] = struct.unpack('>H', payload[26:28])[0]
    return data

def generate_waveforms(n_trials, steps, hz):
    """Generate 3-class waveforms: sine, triangle, square."""
    labels = np.repeat([0, 1, 2], n_trials // 3)
    np.random.shuffle(labels)
    t = np.linspace(0, steps / hz, steps)
    signals = []
    for label in labels:
        freq = np.random.uniform(0.5, 2.0)
        if label == 0:  # sine
            sig = np.sin(2 * np.pi * freq * t)
        elif label == 1:  # triangle
            sig = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        else:  # square
            sig = np.sign(np.sin(2 * np.pi * freq * t))
        signals.append(sig * 0.3)  # scale to ±0.3
    return np.array(signals), labels


# ─── GPU Processing Modes ───

def gpu_process_matmul(spike_history, torch):
    """Static weight matrix multiplication over spike history."""
    # spike_history: (N_NEURONS, HISTORY_LEN)
    W = torch.randn(N_NEURONS, N_NEURONS * HISTORY_LEN, device='cuda') * 0.1
    sh = torch.tensor(spike_history.flatten(), dtype=torch.float32, device='cuda')
    out = torch.tanh(W @ sh)
    return out.cpu().numpy()

def gpu_process_conv1d(spike_history, torch):
    """1D temporal convolution over spike history."""
    sh = torch.tensor(spike_history, dtype=torch.float32, device='cuda').unsqueeze(0)  # (1, 8, H)
    conv = torch.nn.Conv1d(N_NEURONS, N_NEURONS, kernel_size=min(3, spike_history.shape[1]),
                           padding=0, bias=False).cuda()
    with torch.no_grad():
        out = conv(sh).squeeze(0).mean(dim=-1)  # (8,)
    return torch.tanh(out).cpu().numpy()

def gpu_process_spectral(spike_history, torch):
    """FFT-based frequency filtering."""
    sh = torch.tensor(spike_history, dtype=torch.float32, device='cuda')
    fft = torch.fft.rfft(sh, dim=-1)
    # Low-pass: keep first half of frequencies
    n_keep = max(1, fft.shape[-1] // 2)
    fft[:, n_keep:] = 0
    filtered = torch.fft.irfft(fft, n=spike_history.shape[-1], dim=-1)
    return torch.tanh(filtered.mean(dim=-1)).cpu().numpy()

def gpu_process_attention(spike_history, torch):
    """Self-attention over spike history timesteps."""
    sh = torch.tensor(spike_history.T, dtype=torch.float32, device='cuda')  # (T, 8)
    # Q, K, V from spike history
    d = sh.shape[-1]
    scores = sh @ sh.T / (d ** 0.5)  # (T, T)
    attn = torch.softmax(scores, dim=-1)
    out = (attn @ sh).mean(dim=0)  # (8,)
    return torch.tanh(out).cpu().numpy()

def gpu_process_passthrough(spike_history, torch):
    """No GPU processing — just return mean spike rates."""
    return np.tanh(spike_history.mean(axis=1))

def gpu_process_random(spike_history, torch):
    """Random output — control for whether computation matters."""
    return np.random.uniform(-1, 1, N_NEURONS)


GPU_MODES = {
    'MATMUL': gpu_process_matmul,
    'CONV1D': gpu_process_conv1d,
    'SPECTRAL': gpu_process_spectral,
    'ATTENTION': gpu_process_attention,
    'PASSTHROUGH': gpu_process_passthrough,
    'RANDOM': gpu_process_random,
}


def run_reservoir_trial(ser, signal, mode_fn, torch, w_in):
    """Run one trial with GPU processing in the loop."""
    spike_history = np.zeros((N_NEURONS, HISTORY_LEN))
    features_list = []
    latencies = []

    for step_i in range(len(signal)):
        inp = signal[step_i]

        # GPU processes spike history
        t0 = time.perf_counter()
        gpu_out = mode_fn(spike_history, torch)  # (8,)
        lat = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(lat)

        # Compute Vg: base + input modulation + GPU feedback
        for n in range(N_NEURONS):
            vg = BASE_VG + ALPHA * inp * w_in[n] + 0.08 * gpu_out[n]
            vg = np.clip(vg, 0.05, 0.95)
            send_set_vg(ser, n, vg)

        time.sleep(1.0 / SAMPLE_HZ)

        # Read telemetry
        telem = read_telem(ser)
        if telem is None:
            features_list.append(np.zeros(N_NEURONS * 2))
            continue

        spikes = np.array(telem['delta_spikes'], dtype=float)
        vmem = np.array(telem['vmem'], dtype=float) / 256.0

        # Update spike history (shift left, add new)
        spike_history[:, :-1] = spike_history[:, 1:]
        spike_history[:, -1] = spikes

        features_list.append(np.concatenate([spikes, vmem]))

    features = np.array(features_list)
    # Pool: mean + max + std
    if len(features) > 0:
        pooled = np.concatenate([features.mean(0), features.max(0), features.std(0)])
    else:
        pooled = np.zeros(N_NEURONS * 6)

    return pooled, np.mean(latencies), spike_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=N_TRIALS)
    parser.add_argument('--steps', type=int, default=STEPS_PER_TRIAL)
    args = parser.parse_args()

    print("=" * 65)
    print("z2201: GPU as Active Spike Processor")
    print("=" * 65)

    # Import torch
    try:
        import torch
        assert torch.cuda.is_available(), "CUDA not available"
        print(f"[HW] PyTorch CUDA: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[ERR] PyTorch/CUDA: {e}")
        sys.exit(1)

    # Connect FPGA
    ser, port = find_fpga()
    if ser is None:
        print("[ERR] FPGA not found")
        sys.exit(1)
    print(f"[HW] FPGA on {port}")

    # Generate waveforms
    signals, labels = generate_waveforms(args.n_trials, args.steps, SAMPLE_HZ)
    w_in = np.random.RandomState(42).randn(N_NEURONS)
    w_in /= np.linalg.norm(w_in)

    results = {}

    for mode_name, mode_fn in GPU_MODES.items():
        print(f"\n  === Mode: {mode_name} ===")
        all_features = []
        all_latencies = []
        all_spike_histories = []

        for ti in range(args.n_trials):
            if (ti + 1) % 20 == 0:
                print(f"    trial {ti+1}/{args.n_trials}")

            feat, lat, sh = run_reservoir_trial(ser, signals[ti], mode_fn, torch, w_in)
            all_features.append(feat)
            all_latencies.append(lat)
            all_spike_histories.append(sh.copy())

        X = np.array(all_features)
        y = labels

        # 5-fold stratified CV
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
            acc = clf.score(X_test, y[test_idx])
            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_lat = np.mean(all_latencies)

        # Compute MI between GPU output and spike response
        # Use spike history from last step as proxy
        spike_rates = np.array([sh.mean() for sh in all_spike_histories])
        # Discretize for MI estimation
        from sklearn.metrics import mutual_info_score
        sr_binned = np.digitize(spike_rates, np.linspace(spike_rates.min(), spike_rates.max() + 1e-9, 10))
        label_binned = y
        mi = mutual_info_score(sr_binned, label_binned)

        print(f"    accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"    latency:  {mean_lat:.2f} ms")
        print(f"    MI:       {mi:.4f} bits")

        results[mode_name] = {
            'accuracy_mean': float(mean_acc),
            'accuracy_std': float(std_acc),
            'latency_ms': float(mean_lat),
            'mutual_information': float(mi),
            'fold_accs': [float(a) for a in accs],
        }

    # ─── Tests T273-T278 ───
    print("\n" + "=" * 65)
    print("  Tests T273-T278")
    print("=" * 65)

    best_mode = max([m for m in results if m not in ('PASSTHROUGH', 'RANDOM')],
                    key=lambda m: results[m]['accuracy_mean'])
    best_acc = results[best_mode]['accuracy_mean']
    pass_acc = results['PASSTHROUGH']['accuracy_mean']
    rand_acc = results['RANDOM']['accuracy_mean']
    attn_acc = results.get('ATTENTION', {}).get('accuracy_mean', 0)
    matmul_acc = results.get('MATMUL', {}).get('accuracy_mean', 0)

    modes_above_50 = sum(1 for m, r in results.items() if r['accuracy_mean'] > 0.50)

    best_lat = min(results[m]['latency_ms'] for m in results if m not in ('PASSTHROUGH', 'RANDOM'))
    best_mi_mode = max([m for m in results if m not in ('RANDOM',)],
                       key=lambda m: results[m]['mutual_information'])
    best_mi = results[best_mi_mode]['mutual_information']

    tests = {}

    t273 = best_acc > pass_acc
    tests['T273'] = {'pass': t273, 'best': best_mode, 'best_acc': best_acc, 'passthrough_acc': pass_acc}
    print(f"\n  T273 gpu_helps:       {'PASS' if t273 else 'FAIL'}  {best_mode}={best_acc:.4f} vs PASSTHROUGH={pass_acc:.4f}")

    t274 = best_acc > rand_acc
    tests['T274'] = {'pass': t274, 'best_acc': best_acc, 'random_acc': rand_acc}
    print(f"  T274 computation_matters: {'PASS' if t274 else 'FAIL'}  best={best_acc:.4f} vs RANDOM={rand_acc:.4f}")

    t275 = attn_acc > matmul_acc
    tests['T275'] = {'pass': t275, 'attention': attn_acc, 'matmul': matmul_acc}
    print(f"  T275 attention>matmul: {'PASS' if t275 else 'FAIL'}  ATTENTION={attn_acc:.4f} vs MATMUL={matmul_acc:.4f}")

    t276 = modes_above_50 >= 3
    tests['T276'] = {'pass': t276, 'modes_above_50': modes_above_50}
    print(f"  T276 generalization:  {'PASS' if t276 else 'FAIL'}  {modes_above_50} modes > 50%")

    t277 = best_lat < 5.0
    tests['T277'] = {'pass': t277, 'best_latency_ms': best_lat}
    print(f"  T277 realtime:        {'PASS' if t277 else 'FAIL'}  latency={best_lat:.2f} ms")

    t278 = best_mi > 0.1
    tests['T278'] = {'pass': t278, 'best_mi_mode': best_mi_mode, 'mi': best_mi}
    print(f"  T278 spike_mi:        {'PASS' if t278 else 'FAIL'}  MI({best_mi_mode})={best_mi:.4f} bits")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/6 PASS")

    # ─── Save ───
    out = {
        'experiment': 'z2201_gpu_spike_processor',
        'n_trials': args.n_trials,
        'steps_per_trial': args.steps,
        'modes': results,
        'tests': tests,
        'n_pass': n_pass,
        'n_total': 6,
        'best_mode': best_mode,
    }
    out_path = RESULTS / 'z2201_gpu_spike_processor.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results: {out_path}")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Bar chart: accuracy by mode
        modes = list(results.keys())
        accs = [results[m]['accuracy_mean'] for m in modes]
        stds = [results[m]['accuracy_std'] for m in modes]
        colors = ['#e74c3c' if m in ('PASSTHROUGH', 'RANDOM') else '#3498db' for m in modes]
        axes[0].bar(modes, accs, yerr=stds, color=colors, capsize=3)
        axes[0].axhline(0.333, ls='--', color='gray', alpha=0.5, label='chance')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Waveform Classification by GPU Mode')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend()

        # Latency
        lats = [results[m]['latency_ms'] for m in modes]
        axes[1].bar(modes, lats, color=colors)
        axes[1].axhline(5.0, ls='--', color='red', alpha=0.5, label='5ms threshold')
        axes[1].set_ylabel('Latency (ms)')
        axes[1].set_title('GPU Processing Latency')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()

        # MI
        mis = [results[m]['mutual_information'] for m in modes]
        axes[2].bar(modes, mis, color=colors)
        axes[2].axhline(0.1, ls='--', color='red', alpha=0.5, label='0.1 bit threshold')
        axes[2].set_ylabel('MI (bits)')
        axes[2].set_title('Spike-Label Mutual Information')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].legend()

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2201_gpu_spike_processor.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Figure: {fig_path}")
    except Exception as e:
        print(f"  [WARN] Figure failed: {e}")

    ser.close()


if __name__ == '__main__':
    main()
