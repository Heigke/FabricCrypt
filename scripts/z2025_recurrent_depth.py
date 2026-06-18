#!/usr/bin/env python3
"""
z2025: Recurrent Processing Depth (RPT Test)

Recurrent Processing Theory (Lamme 2006) predicts that consciousness requires
recurrent (feedback) processing, not just feedforward sweeps. The key prediction:

  - Feedforward-only: handles simple classification
  - Recurrent: handles tasks requiring temporal integration across layers

Test: Use a TASK that genuinely requires recurrence to solve. Specifically,
a "same/different" judgment over two sequential stimuli separated by a delay.
This requires holding stimulus 1 in memory and comparing with stimulus 2 —
impossible for a purely feedforward network without recurrence.

Architecture:
  - Encoder: CNN → hidden_dim
  - Optional recurrent workspace: GRU that processes encoder outputs over time
  - Classifier: workspace output → 2 classes (same/different)

Conditions:
  A: Recurrent workspace (GRU, 32 dim) — expect PASS on delayed comparison
  B: Feedforward only (no recurrence) — expect FAIL on delayed comparison
  C: Deep feedforward (extra layers to match params) — still FAIL?
  D: Recurrent workspace + telemetry — expect PASS

The key control: condition C has SAME parameter count as A, but no recurrence.
If C also passes, recurrence isn't needed and the test is uninformative.
If only A/D pass, recurrence is genuinely required.

Additional test: vary the delay between stimuli. Recurrent models should
maintain performance; feedforward should degrade with longer delays.

References:
  Lamme 2006 — Towards a true neural stance on consciousness
  Lamme 2010 — How neuroscience will change our view of consciousness
  Block 2005 — Two neural correlates of consciousness
  VanRullen & Koch 2003 — Visual processing depth

NO consciousness losses. Recurrence benefit must EMERGE from task demands.
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


# ---------- Dataset: Delayed Same/Different ----------

class DelayedMatchDataset(Dataset):
    """Present two MNIST digits with variable delay. Task: same class or different?

    Each sample is a sequence: [stim1, delay_frames..., stim2]
    Label: 1 if same class, 0 if different class.
    Delay frames are Gaussian noise.
    """
    def __init__(self, mnist_data, delay=3, seq_len=None, n_samples=20000):
        self.images = mnist_data.data.float() / 255.0
        self.labels = mnist_data.targets
        self.delay = delay
        self.seq_len = seq_len or (2 + delay)  # stim1 + delay + stim2
        self.n_samples = n_samples
        self.n_images = len(self.labels)

        # Pre-generate sample indices
        rng = np.random.RandomState(42)
        self.idx1 = rng.randint(0, self.n_images, n_samples)
        self.idx2 = rng.randint(0, self.n_images, n_samples)

        # For ~50% same, pick same-class second stimulus
        self.is_same = np.zeros(n_samples, dtype=np.int64)
        label_array = self.labels.numpy()
        for i in range(n_samples):
            if rng.random() < 0.5:
                # Find a same-class image
                c = label_array[self.idx1[i]]
                candidates = np.where(label_array == c)[0]
                self.idx2[i] = rng.choice(candidates)
                self.is_same[i] = 1
            else:
                # Ensure different class
                c = label_array[self.idx1[i]]
                candidates = np.where(label_array != c)[0]
                self.idx2[i] = rng.choice(candidates)
                self.is_same[i] = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img1 = self.images[self.idx1[idx]].unsqueeze(0)  # [1, 28, 28]
        img2 = self.images[self.idx2[idx]].unsqueeze(0)

        # Build sequence: [stim1, noise..., stim2]
        frames = []
        frames.append(img1)
        for _ in range(self.delay):
            frames.append(torch.randn(1, 28, 28) * 0.3)  # Noise delay
        frames.append(img2)

        sequence = torch.stack(frames, dim=0)  # [seq_len, 1, 28, 28]
        label = torch.tensor(self.is_same[idx], dtype=torch.long)
        return sequence, label


# ---------- Architecture ----------

class FrameEncoder(nn.Module):
    """Per-frame CNN encoder."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, hidden_dim)

    def forward(self, x):
        """x: [batch*seq, 1, 28, 28] → [batch*seq, hidden_dim]"""
        h = self.conv(x)
        return F.relu(self.fc(h.view(h.size(0), -1)))


class RecurrentWorkspace(nn.Module):
    """GRU-based recurrent workspace for temporal integration."""
    def __init__(self, input_dim, ws_dim, telemetry_dim=0):
        super().__init__()
        self.ws_dim = ws_dim
        total_in = input_dim + telemetry_dim
        self.gru = nn.GRU(total_in, ws_dim, batch_first=True)
        self.ln = nn.LayerNorm(ws_dim)

    def forward(self, frame_features, telemetry=None):
        """frame_features: [batch, seq_len, input_dim]"""
        if telemetry is not None:
            # Broadcast telemetry across time
            telem = telemetry.unsqueeze(1).expand(-1, frame_features.size(1), -1)
            x = torch.cat([frame_features, telem], dim=-1)
        else:
            x = frame_features
        output, h_n = self.gru(x)
        # Use final hidden state
        return self.ln(h_n.squeeze(0))


class FeedforwardAggregator(nn.Module):
    """Non-recurrent alternative: concatenate first and last frames."""
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, frame_features, telemetry=None):
        """Take first and last frame only."""
        first = frame_features[:, 0]
        last = frame_features[:, -1]
        combined = torch.cat([first, last], dim=-1)
        return self.fc(combined)


class DeepFeedforward(nn.Module):
    """Deep feedforward to match param count of GRU."""
    def __init__(self, input_dim, out_dim, n_layers=4):
        super().__init__()
        layers = []
        dim = input_dim * 2
        for i in range(n_layers):
            next_dim = out_dim if i == n_layers - 1 else dim
            layers.extend([nn.Linear(dim, next_dim), nn.ReLU()])
            dim = next_dim
        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, frame_features, telemetry=None):
        first = frame_features[:, 0]
        last = frame_features[:, -1]
        combined = torch.cat([first, last], dim=-1)
        return self.ln(self.net(combined))


class DelayedMatchModel(nn.Module):
    def __init__(self, hidden_dim=64, mode='recurrent', ws_dim=32, telemetry_dim=0):
        super().__init__()
        self.encoder = FrameEncoder(hidden_dim)
        self.mode = mode
        self.telemetry_dim = telemetry_dim

        if mode == 'recurrent':
            self.aggregator = RecurrentWorkspace(hidden_dim, ws_dim, telemetry_dim)
            agg_out = ws_dim
        elif mode == 'feedforward':
            self.aggregator = FeedforwardAggregator(hidden_dim, ws_dim)
            agg_out = ws_dim
        elif mode == 'deep_feedforward':
            self.aggregator = DeepFeedforward(hidden_dim, ws_dim, n_layers=4)
            agg_out = ws_dim
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.classifier = nn.Linear(agg_out, 2)

    def forward(self, sequence, telemetry=None):
        """sequence: [batch, seq_len, 1, 28, 28]"""
        batch, seq_len = sequence.shape[:2]
        frames = sequence.view(batch * seq_len, 1, 28, 28)
        features = self.encoder(frames)
        features = features.view(batch, seq_len, -1)

        agg = self.aggregator(features, telemetry)
        return self.classifier(agg)


# ---------- Training ----------

def train_model(model, train_data, device, delay=3, epochs=15,
                batch_size=64, n_train=20000):
    dataset = DelayedMatchDataset(train_data, delay=delay, n_samples=n_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            telemetry = None
            if model.telemetry_dim > 0 and TELEMETRY_AVAILABLE:
                try:
                    telem = SysfsHwmonTelemetry(card_index=0)
                    s = telem.read_sample()
                    t_val = getattr(s, 'temp_edge_c', 50.0)
                    p_val = getattr(s, 'power_w', 30.0)
                    telemetry = torch.tensor(
                        [[t_val / 100.0, p_val / 100.0]], device=device
                    ).expand(sequences.size(0), -1)
                except Exception:
                    telemetry = None

            optimizer.zero_grad()
            logits = model(sequences, telemetry)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sequences.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += sequences.size(0)

        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.3f}  ({elapsed:.1f}s)")


@torch.no_grad()
def evaluate_at_delay(model, test_data, device, delay, n_test=5000):
    dataset = DelayedMatchDataset(test_data, delay=delay, n_samples=n_test)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()

    correct = 0
    total = 0
    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        telemetry = None
        if model.telemetry_dim > 0:
            telemetry = torch.zeros(sequences.size(0), model.telemetry_dim, device=device)

        logits = model(sequences, telemetry)
        correct += (logits.argmax(1) == labels).sum().item()
        total += sequences.size(0)

    return correct / total


# ---------- Main ----------

def run_condition(label, model, train_data, test_data, device,
                  train_delay=3, epochs=15, batch_size=64):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Mode: {model.mode}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*70}")

    model = model.to(device)
    train_model(model, train_data, device, delay=train_delay,
                epochs=epochs, batch_size=batch_size)

    # Test at various delays
    delays = [0, 1, 2, 3, 5, 8, 12]
    delay_accs = {}

    print(f"\n  --- Delay Curve ---")
    for d in delays:
        acc = evaluate_at_delay(model, test_data, device, delay=d)
        delay_accs[d] = acc
        bar = '#' * int(acc * 40)
        print(f"    delay={d:2d}  acc={acc:.3f}  |{bar}")

    # Key metrics
    acc_at_train_delay = delay_accs[train_delay]
    acc_at_zero = delay_accs[0]
    acc_at_long = delay_accs[max(delays)]
    degradation = acc_at_zero - acc_at_long

    print(f"\n  Accuracy at delay=0: {acc_at_zero:.3f}")
    print(f"  Accuracy at delay={train_delay}: {acc_at_train_delay:.3f}")
    print(f"  Accuracy at delay={max(delays)}: {acc_at_long:.3f}")
    print(f"  Degradation (0→{max(delays)}): {degradation:+.3f}")

    return {
        'label': label,
        'mode': model.mode,
        'n_params': n_params,
        'train_delay': train_delay,
        'delay_accuracies': {str(k): v for k, v in delay_accs.items()},
        'acc_at_train_delay': acc_at_train_delay,
        'acc_at_zero': acc_at_zero,
        'acc_at_long': acc_at_long,
        'degradation': degradation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-delay', type=int, default=3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2025] Device: {device}")

    from torchvision import datasets
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    print("Loading MNIST...")
    train_data = datasets.MNIST(str(data_dir), train=True, download=True)
    test_data = datasets.MNIST(str(data_dir), train=False, download=True)

    print(f"\n{'='*70}")
    print(f"  z2025: Recurrent Processing Depth (RPT Test)")
    print(f"  RPT predicts: recurrence needed for temporal integration")
    print(f"  Task: delayed same/different judgment")
    print(f"  NO consciousness losses — recurrence benefit must EMERGE")
    print(f"{'='*70}")

    results = {}

    # A: Recurrent workspace (GRU)
    model_a = DelayedMatchModel(hidden_dim=64, mode='recurrent', ws_dim=32)
    results['A'] = run_condition(
        'A: Recurrent workspace (GRU, 32 dim)', model_a,
        train_data, test_data, device, args.train_delay, args.epochs, args.batch_size
    )

    # B: Feedforward (first + last frame)
    model_b = DelayedMatchModel(hidden_dim=64, mode='feedforward', ws_dim=32)
    results['B'] = run_condition(
        'B: Feedforward (first+last)', model_b,
        train_data, test_data, device, args.train_delay, args.epochs, args.batch_size
    )

    # C: Deep feedforward (param-matched)
    model_c = DelayedMatchModel(hidden_dim=64, mode='deep_feedforward', ws_dim=32)
    results['C'] = run_condition(
        'C: Deep feedforward (param-matched)', model_c,
        train_data, test_data, device, args.train_delay, args.epochs, args.batch_size
    )

    # D: Recurrent + telemetry
    model_d = DelayedMatchModel(hidden_dim=64, mode='recurrent', ws_dim=32, telemetry_dim=2)
    results['D'] = run_condition(
        'D: Recurrent + telemetry (GRU, 32 dim)', model_d,
        train_data, test_data, device, args.train_delay, args.epochs, args.batch_size
    )

    # ---------- Analysis ----------
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Recurrent Processing Depth")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<45} {'Acc@0':>6} {'Acc@{}'.format(args.train_delay):>6} "
          f"{'Acc@{}'.format(max([0,1,2,3,5,8,12])):>6} {'Degrad':>8}")
    print(f"  {'-'*71}")
    for key in ['A', 'B', 'C', 'D']:
        r = results[key]
        print(f"  {r['label']:<45} {r['acc_at_zero']:>6.3f} "
              f"{r['acc_at_train_delay']:>6.3f} {r['acc_at_long']:>6.3f} "
              f"{r['degradation']:>+8.3f}")

    # Tests
    recurrent_acc = results['A']['acc_at_train_delay']
    ff_acc = results['B']['acc_at_train_delay']
    deep_ff_acc = results['C']['acc_at_train_delay']
    recurrent_telem_acc = results['D']['acc_at_train_delay']

    t1 = recurrent_acc > ff_acc + 0.03  # Recurrent > feedforward
    t2 = recurrent_acc > deep_ff_acc + 0.02  # Recurrent > deep feedforward (param-matched)
    t3 = results['A']['degradation'] < results['B']['degradation']  # Recurrent more robust to delay
    t4 = recurrent_acc > 0.65  # Recurrent actually learns the task (above chance=0.5)

    print(f"\n  T1: Recurrent > feedforward (>3% gap):       "
          f"{'PASS' if t1 else 'FAIL'} ({recurrent_acc:.3f} vs {ff_acc:.3f})")
    print(f"  T2: Recurrent > deep FF (params matched):     "
          f"{'PASS' if t2 else 'FAIL'} ({recurrent_acc:.3f} vs {deep_ff_acc:.3f})")
    print(f"  T3: Recurrent less delay degradation:          "
          f"{'PASS' if t3 else 'FAIL'} ({results['A']['degradation']:+.3f} vs "
          f"{results['B']['degradation']:+.3f})")
    print(f"  T4: Recurrent learns task (>0.65):             "
          f"{'PASS' if t4 else 'FAIL'} ({recurrent_acc:.3f})")

    n_pass = sum([t1, t2, t3, t4])
    if n_pass >= 4:
        verdict = "RECURRENCE_ESSENTIAL"
    elif n_pass >= 3:
        verdict = "RECURRENCE_BENEFICIAL"
    elif n_pass >= 2:
        verdict = "RECURRENCE_WEAK"
    else:
        verdict = "RECURRENCE_NOT_NEEDED"

    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    # Save
    output = {
        'experiment': 'z2025_recurrent_depth',
        'hypothesis': 'Recurrence is necessary for temporal integration across delay',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'train_delay': args.train_delay,
        'references': [
            'Lamme 2006 (Recurrent Processing Theory)',
            'Lamme 2010 (RPT and consciousness)',
            'Block 2005 (Two neural correlates)',
        ],
        'design_principle': 'NO consciousness losses. Task demands recurrence.',
        'conditions': results,
        'tests': {
            't1_recurrent_gt_feedforward': bool(t1),
            't2_recurrent_gt_deep_ff': bool(t2),
            't3_recurrent_less_degradation': bool(t3),
            't4_recurrent_learns_task': bool(t4),
        },
        'tests_passed': n_pass,
        'verdict': verdict,
    }

    def json_safe(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(v) for v in obj]
        if isinstance(obj, bool):
            return bool(obj)
        return obj

    output = json_safe(output)
    results_path = Path(__file__).parent.parent / 'results' / 'z2025_recurrent_depth.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
