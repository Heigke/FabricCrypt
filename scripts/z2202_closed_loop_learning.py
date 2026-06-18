#!/usr/bin/env python3
"""z2202_closed_loop_learning.py — Closed-Loop Learning: FPGA Error → GPU Weight Update

First demonstration of LEARNING in the GPU-FPGA bridge: FPGA classification error
modifies GPU-side weights that control Vg modulation, creating an online gradient-free
learning loop across substrates.

Architecture:
  Input → GPU weights → Vg → FPGA spikes → readout → error → GPU weight update

Learning Rule: Reward-modulated Hebbian
  Δw_ij = η * reward * (spike_i - <spike_i>) * input_j
  reward = +1 if correct, -1 if wrong (per trial)

3 Conditions:
  LEARNING:  weights updated after each trial based on error
  FROZEN:    same architecture, weights frozen after initialization
  RANDOM:    random weights, no learning (control)

Task: 3-class waveform classification (same as reservoir benchmarks)
Training: 200 trials online learning, then 60 test trials (weights frozen)

Tests T279-T284:
  T279: LEARNING test acc > FROZEN test acc (learning happened)
  T280: LEARNING test acc > RANDOM test acc (learned weights > random)
  T281: LEARNING acc improves from first 30 to last 30 training trials
  T282: Weight change magnitude > 0.01 (weights actually moved)
  T283: LEARNING test acc > 40% (above chance 33.3%)
  T284: Learning curve shows monotonic improvement (rank corr > 0.3)

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
STEPS_PER_TRIAL = 25
SAMPLE_HZ = 20
N_TRAIN = 200
N_TEST = 60
BASE_VG = 0.45
ALPHA = 0.15
ETA = 0.02  # learning rate
N_FOLDS = 5

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
    return data

def generate_waveforms(n_trials, steps, hz):
    labels = np.repeat([0, 1, 2], (n_trials + 2) // 3)[:n_trials]
    np.random.shuffle(labels)
    t = np.linspace(0, steps / hz, steps)
    signals = []
    for label in labels:
        freq = np.random.uniform(0.5, 2.0)
        if label == 0:
            sig = np.sin(2 * np.pi * freq * t)
        elif label == 1:
            sig = 2 * np.abs(2 * (t * freq % 1) - 1) - 1
        else:
            sig = np.sign(np.sin(2 * np.pi * freq * t))
        signals.append(sig * 0.3)
    return np.array(signals), labels


def run_trial(ser, signal, W, learn=False, label=None, eta=ETA):
    """Run one trial, optionally updating weights."""
    spike_accum = np.zeros(N_NEURONS)
    vmem_accum = np.zeros(N_NEURONS)
    input_accum = np.zeros(STEPS_PER_TRIAL)

    for step_i in range(len(signal)):
        inp = signal[step_i]
        input_accum[step_i] = inp

        # Vg = base + W @ [input, 1] per neuron
        for n in range(N_NEURONS):
            vg = BASE_VG + ALPHA * inp * W[n, 0] + 0.05 * W[n, 1]
            vg = np.clip(vg, 0.05, 0.95)
            send_set_vg(ser, n, vg)

        time.sleep(1.0 / SAMPLE_HZ)

        telem = read_telem(ser)
        if telem is None:
            continue

        spike_accum += np.array(telem['delta_spikes'], dtype=float)
        vmem_accum += np.array(telem['vmem'], dtype=float) / 256.0

    # Normalize
    spike_rates = spike_accum / max(len(signal), 1)
    vmem_mean = vmem_accum / max(len(signal), 1)

    # Simple readout: argmax of spike rates binned into 3 groups
    features = np.concatenate([spike_rates, vmem_mean])

    # Online classification: nearest centroid per class
    # (We use a simple linear readout that's part of W)
    readout = W[:, 2:5]  # (8, 3) — one column per class
    scores = features[:N_NEURONS] @ readout  # (3,)
    predicted = int(np.argmax(scores))

    correct = (predicted == label) if label is not None else None

    # Reward-modulated Hebbian update
    if learn and label is not None:
        reward = 1.0 if correct else -1.0
        spike_centered = spike_rates - spike_rates.mean()
        mean_input = input_accum.mean()

        # Update input weights
        W[:, 0] += eta * reward * spike_centered * mean_input
        # Update bias
        W[:, 1] += eta * reward * spike_centered * 0.1
        # Update readout: reinforce correct class column
        for c in range(3):
            r = 1.0 if c == label else -0.5
            W[:, 2 + c] += eta * r * reward * spike_centered

        # Clip weights
        W[:, :] = np.clip(W, -2.0, 2.0)

    return features, predicted, correct, W


def run_condition(ser, condition, train_signals, train_labels, test_signals, test_labels):
    """Run one condition (LEARNING, FROZEN, RANDOM)."""
    rng = np.random.RandomState(42)
    W = rng.randn(N_NEURONS, 5) * 0.1  # (8, 5): [input_w, bias, readout_0, readout_1, readout_2]
    W_init = W.copy()

    learn = (condition == 'LEARNING')
    if condition == 'RANDOM':
        W = rng.randn(N_NEURONS, 5) * 0.5  # Larger random weights

    # Training phase
    train_correct = []
    train_preds = []
    for ti in range(len(train_signals)):
        if (ti + 1) % 40 == 0:
            print(f"    train {ti+1}/{len(train_signals)}")

        _, pred, corr, W = run_trial(ser, train_signals[ti], W,
                                      learn=learn, label=train_labels[ti])
        train_correct.append(corr)
        train_preds.append(pred)

    # Test phase (no learning)
    test_correct = []
    for ti in range(len(test_signals)):
        _, pred, corr, _ = run_trial(ser, test_signals[ti], W,
                                      learn=False, label=test_labels[ti])
        test_correct.append(corr)

    W_final = W.copy()
    weight_change = float(np.mean(np.abs(W_final - W_init)))

    train_acc = float(np.mean(train_correct)) if train_correct else 0.0
    test_acc = float(np.mean(test_correct)) if test_correct else 0.0

    # Learning curve: accuracy in windows of 30 trials
    window = 30
    curve = []
    for i in range(0, len(train_correct) - window + 1, window):
        curve.append(float(np.mean(train_correct[i:i+window])))

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'weight_change': weight_change,
        'learning_curve': curve,
        'n_train': len(train_signals),
        'n_test': len(test_signals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train', type=int, default=N_TRAIN)
    parser.add_argument('--n-test', type=int, default=N_TEST)
    args = parser.parse_args()

    print("=" * 65)
    print("z2202: Closed-Loop Learning — FPGA Error → GPU Weight Update")
    print("=" * 65)

    ser, port = find_fpga()
    if ser is None:
        print("[ERR] FPGA not found")
        sys.exit(1)
    print(f"[HW] FPGA on {port}")

    # Generate data
    total = args.n_train + args.n_test
    signals, labels = generate_waveforms(total, STEPS_PER_TRIAL, SAMPLE_HZ)
    train_signals, test_signals = signals[:args.n_train], signals[args.n_train:]
    train_labels, test_labels = labels[:args.n_train], labels[args.n_train:]

    conditions = ['LEARNING', 'FROZEN', 'RANDOM']
    results = {}

    for cond in conditions:
        print(f"\n  === Condition: {cond} ===")
        r = run_condition(ser, cond, train_signals, train_labels, test_signals, test_labels)
        results[cond] = r
        print(f"    train_acc: {r['train_acc']:.4f}")
        print(f"    test_acc:  {r['test_acc']:.4f}")
        print(f"    Δw:        {r['weight_change']:.4f}")
        print(f"    curve:     {[f'{v:.3f}' for v in r['learning_curve']]}")

    # ─── Tests T279-T284 ───
    print("\n" + "=" * 65)
    print("  Tests T279-T284")
    print("=" * 65)

    learn = results['LEARNING']
    frozen = results['FROZEN']
    rand = results['RANDOM']

    tests = {}

    t279 = learn['test_acc'] > frozen['test_acc']
    tests['T279'] = {'pass': t279, 'learn_test': learn['test_acc'], 'frozen_test': frozen['test_acc']}
    print(f"\n  T279 learning_helps:    {'PASS' if t279 else 'FAIL'}  LEARN={learn['test_acc']:.4f} vs FROZEN={frozen['test_acc']:.4f}")

    t280 = learn['test_acc'] > rand['test_acc']
    tests['T280'] = {'pass': t280, 'learn_test': learn['test_acc'], 'random_test': rand['test_acc']}
    print(f"  T280 learned>random:    {'PASS' if t280 else 'FAIL'}  LEARN={learn['test_acc']:.4f} vs RANDOM={rand['test_acc']:.4f}")

    curve = learn['learning_curve']
    t281 = False
    if len(curve) >= 2:
        t281 = curve[-1] > curve[0]
    tests['T281'] = {'pass': t281, 'first_window': curve[0] if curve else 0, 'last_window': curve[-1] if curve else 0}
    print(f"  T281 improvement:       {'PASS' if t281 else 'FAIL'}  first={curve[0]:.3f} → last={curve[-1]:.3f}" if curve else "  T281: FAIL (no curve)")

    t282 = learn['weight_change'] > 0.01
    tests['T282'] = {'pass': t282, 'weight_change': learn['weight_change']}
    print(f"  T282 weights_moved:     {'PASS' if t282 else 'FAIL'}  Δw={learn['weight_change']:.4f}")

    t283 = learn['test_acc'] > 0.40
    tests['T283'] = {'pass': t283, 'test_acc': learn['test_acc']}
    print(f"  T283 above_chance:      {'PASS' if t283 else 'FAIL'}  test_acc={learn['test_acc']:.4f}")

    # Rank correlation of learning curve
    from scipy.stats import spearmanr
    if len(curve) >= 3:
        rho, _ = spearmanr(range(len(curve)), curve)
    else:
        rho = 0.0
    t284 = rho > 0.3
    tests['T284'] = {'pass': t284, 'spearman_rho': float(rho)}
    print(f"  T284 monotonic_curve:   {'PASS' if t284 else 'FAIL'}  rho={rho:.3f}")

    n_pass = sum(1 for t in tests.values() if t['pass'])
    print(f"\n  TOTAL: {n_pass}/6 PASS")

    # ─── Save ───
    out = {
        'experiment': 'z2202_closed_loop_learning',
        'n_train': args.n_train,
        'n_test': args.n_test,
        'conditions': results,
        'tests': tests,
        'n_pass': n_pass,
        'n_total': 6,
    }
    out_path = RESULTS / 'z2202_closed_loop_learning.json'
    out_path.write_text(json.dumps(out, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x)))
    print(f"\n  Results: {out_path}")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Accuracy comparison
        conds = list(results.keys())
        test_accs = [results[c]['test_acc'] for c in conds]
        train_accs = [results[c]['train_acc'] for c in conds]
        x = np.arange(len(conds))
        axes[0].bar(x - 0.15, train_accs, 0.3, label='Train', color='#3498db')
        axes[0].bar(x + 0.15, test_accs, 0.3, label='Test', color='#e74c3c')
        axes[0].axhline(0.333, ls='--', color='gray', alpha=0.5)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(conds)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Classification Accuracy')
        axes[0].legend()

        # Learning curve
        for cond in conds:
            c = results[cond]['learning_curve']
            if c:
                axes[1].plot(c, marker='o', label=cond, markersize=4)
        axes[1].axhline(0.333, ls='--', color='gray', alpha=0.5)
        axes[1].set_xlabel('Training Window (30 trials)')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Learning Curves')
        axes[1].legend()

        # Weight change
        wcs = [results[c]['weight_change'] for c in conds]
        axes[2].bar(conds, wcs, color=['#2ecc71', '#95a5a6', '#e67e22'])
        axes[2].axhline(0.01, ls='--', color='red', alpha=0.5, label='threshold')
        axes[2].set_ylabel('Mean |Δw|')
        axes[2].set_title('Weight Change Magnitude')
        axes[2].legend()

        plt.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES / 'z2202_closed_loop_learning.png'
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Figure: {fig_path}")
    except Exception as e:
        print(f"  [WARN] Figure failed: {e}")

    ser.close()


if __name__ == '__main__':
    main()
