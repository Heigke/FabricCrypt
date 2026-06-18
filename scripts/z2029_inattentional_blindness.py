#!/usr/bin/env python3
"""
z2029: Inattentional Blindness (Consciousness Predicts Missing Things)

Tier 1 unforgeable test: A genuine conscious processor must MISS unexpected
stimuli when attention is directed elsewhere. Perfect detection of everything
= no selective attention = no consciousness by GWT.

Setup:
  - Primary task: classify attended stream (sequence of MNIST digits)
  - Unexpected stimulus: rare "gorilla" (FashionMNIST item) inserted in stream
  - Test: does workspace model MISS the unexpected item?

GWT prediction:
  - Workspace model: MISSES unexpected items (attention bottleneck)
  - No-workspace model: DETECTS unexpected items (parallel processing)
  - More load on primary task → more misses (load-dependent blindness)

Architecture:
  - Stream encoder: per-frame CNN
  - Workspace: GRU that processes attended stream
  - Primary classifier: digit identity from workspace
  - Surprise detector: was there a non-digit in the stream?

Conditions:
  A: Narrow workspace (16 dim) — expect HIGH miss rate
  B: Wide workspace (128 dim) — expect LOWER miss rate
  C: No workspace (parallel) — expect ZERO misses
  D: Narrow + high primary load — expect HIGHEST miss rate

References:
  Simons & Chabris 1999 — Gorillas in our midst (original)
  Mack & Rock 1998 — Inattentional Blindness
  Most et al. 2005 — How NOT to be seen

NO consciousness losses. Blindness must EMERGE from capacity limits.
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


class InattentionalStream(Dataset):
    """Stream of MNIST digits with occasional FashionMNIST "gorillas."

    Primary task: classify the digit at the cued position.
    Secondary (unattended): detect if a fashion item was present.
    """
    def __init__(self, mnist_data, fashion_data, stream_len=6, gorilla_prob=0.3,
                 n_samples=20000, hard_primary=False):
        self.mnist_imgs = mnist_data.data.float() / 255.0
        self.mnist_labels = mnist_data.targets
        self.fashion_imgs = fashion_data.data.float() / 255.0
        self.stream_len = stream_len
        self.gorilla_prob = gorilla_prob
        self.n_samples = n_samples
        self.hard_primary = hard_primary

        rng = np.random.RandomState(42)
        self.digit_indices = rng.randint(0, len(self.mnist_labels), (n_samples, stream_len))
        self.cue_positions = rng.randint(0, stream_len, n_samples)
        self.has_gorilla = rng.random(n_samples) < gorilla_prob
        self.gorilla_positions = rng.randint(0, stream_len, n_samples)
        self.fashion_indices = rng.randint(0, len(self.fashion_imgs), n_samples)

        # Ensure gorilla not at cued position (unattended)
        for i in range(n_samples):
            if self.has_gorilla[i]:
                while self.gorilla_positions[i] == self.cue_positions[i]:
                    self.gorilla_positions[i] = rng.randint(0, stream_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        frames = []
        for j in range(self.stream_len):
            if self.has_gorilla[idx] and j == self.gorilla_positions[idx]:
                img = self.fashion_imgs[self.fashion_indices[idx]].unsqueeze(0)
            else:
                img = self.mnist_imgs[self.digit_indices[idx, j]].unsqueeze(0)
            frames.append(img)

        stream = torch.stack(frames)  # [stream_len, 1, 28, 28]

        # Primary task: digit at cued position
        cue_pos = self.cue_positions[idx]
        digit_label = self.mnist_labels[self.digit_indices[idx, cue_pos]]

        # If hard primary: also need to report sum of all digits (mod 10)
        if self.hard_primary:
            digit_sum = sum(self.mnist_labels[self.digit_indices[idx, j]].item()
                          for j in range(self.stream_len)
                          if not (self.has_gorilla[idx] and j == self.gorilla_positions[idx]))
            secondary_label = digit_sum % 10
        else:
            secondary_label = 0

        # Gorilla detection
        gorilla_present = int(self.has_gorilla[idx])

        cue = torch.zeros(self.stream_len)
        cue[cue_pos] = 1.0

        return stream, cue, digit_label, torch.tensor(gorilla_present), torch.tensor(secondary_label)


class FrameEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, hidden_dim)

    def forward(self, x):
        h = self.conv(x)
        return F.relu(self.fc(h.view(h.size(0), -1)))


class InattentionalModel(nn.Module):
    def __init__(self, hidden_dim=64, ws_dim=None, hard_primary=False):
        super().__init__()
        self.encoder = FrameEncoder(hidden_dim)
        self.ws_dim = ws_dim
        self.hard_primary = hard_primary

        if ws_dim is not None:
            self.workspace = nn.GRU(hidden_dim, ws_dim, batch_first=True)
            self.ws_ln = nn.LayerNorm(ws_dim)
            agg_dim = ws_dim
        else:
            self.workspace = None
            agg_dim = hidden_dim

        # Primary: digit classification at cued position
        self.digit_head = nn.Linear(agg_dim, 10)

        # Gorilla detector: was there a non-digit in the stream?
        self.gorilla_head = nn.Linear(agg_dim, 2)

        # Optional hard primary: digit sum mod 10
        if hard_primary:
            self.sum_head = nn.Linear(agg_dim, 10)

    def forward(self, stream, cue):
        batch, seq_len = stream.shape[:2]
        frames = stream.view(batch * seq_len, 1, 28, 28)
        features = self.encoder(frames).view(batch, seq_len, -1)

        if self.workspace is not None:
            ws_out, _ = self.workspace(features)
            ws_out = self.ws_ln(ws_out)
            # Cue-guided selection
            cue_expanded = cue.unsqueeze(-1)
            selected = (ws_out * cue_expanded).sum(dim=1)
            # For gorilla detection: use final hidden state (whole stream)
            stream_rep = ws_out[:, -1]
        else:
            # Direct selection
            cue_expanded = cue.unsqueeze(-1)
            selected = (features * cue_expanded).sum(dim=1)
            stream_rep = features.mean(dim=1)

        digit_logits = self.digit_head(selected)
        gorilla_logits = self.gorilla_head(stream_rep)

        result = {'digit_logits': digit_logits, 'gorilla_logits': gorilla_logits}
        if self.hard_primary:
            result['sum_logits'] = self.sum_head(stream_rep)

        return result


def train_model(model, train_data, fashion_data, device, epochs=15,
                batch_size=64, hard_primary=False, stream_len=6):
    dataset = InattentionalStream(train_data, fashion_data, stream_len=stream_len,
                                   hard_primary=hard_primary)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, digit_correct, total = 0, 0, 0
        t0 = time.time()
        for stream, cue, digit_lbl, gorilla_lbl, sum_lbl in loader:
            stream = stream.to(device)
            cue = cue.to(device)
            digit_lbl = digit_lbl.to(device)
            gorilla_lbl = gorilla_lbl.to(device)

            optimizer.zero_grad()
            out = model(stream, cue)

            # Primary task loss (heavily weighted — this is the "attended" task)
            loss_digit = F.cross_entropy(out['digit_logits'], digit_lbl)

            # Gorilla detection loss (lightly weighted — model learns passively)
            loss_gorilla = F.cross_entropy(out['gorilla_logits'], gorilla_lbl)

            # Hard primary task
            loss_sum = 0
            if hard_primary and 'sum_logits' in out:
                sum_lbl = sum_lbl.to(device)
                loss_sum = F.cross_entropy(out['sum_logits'], sum_lbl)

            loss = 3.0 * loss_digit + 0.5 * loss_gorilla + 1.0 * loss_sum
            loss.backward()
            optimizer.step()

            total_loss += loss_digit.item() * stream.size(0)
            digit_correct += (out['digit_logits'].argmax(1) == digit_lbl).sum().item()
            total += stream.size(0)

        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  digit_acc={digit_correct/total:.3f}  ({elapsed:.1f}s)")


@torch.no_grad()
def evaluate(model, test_data, fashion_data, device, hard_primary=False,
             stream_len=6, n_test=5000):
    dataset = InattentionalStream(test_data, fashion_data, stream_len=stream_len,
                                   hard_primary=hard_primary, n_samples=n_test)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()

    digit_correct, gorilla_correct, total = 0, 0, 0
    gorilla_present_detected, gorilla_present_total = 0, 0
    gorilla_absent_correct, gorilla_absent_total = 0, 0

    for stream, cue, digit_lbl, gorilla_lbl, sum_lbl in loader:
        stream, cue = stream.to(device), cue.to(device)
        digit_lbl, gorilla_lbl = digit_lbl.to(device), gorilla_lbl.to(device)

        out = model(stream, cue)
        digit_correct += (out['digit_logits'].argmax(1) == digit_lbl).sum().item()
        gorilla_preds = out['gorilla_logits'].argmax(1)
        gorilla_correct += (gorilla_preds == gorilla_lbl).sum().item()

        # Miss rate: when gorilla present, how often does model say "absent"?
        present_mask = gorilla_lbl == 1
        if present_mask.sum() > 0:
            gorilla_present_detected += (gorilla_preds[present_mask] == 1).sum().item()
            gorilla_present_total += present_mask.sum().item()

        absent_mask = gorilla_lbl == 0
        if absent_mask.sum() > 0:
            gorilla_absent_correct += (gorilla_preds[absent_mask] == 0).sum().item()
            gorilla_absent_total += absent_mask.sum().item()

        total += stream.size(0)

    detection_rate = gorilla_present_detected / max(gorilla_present_total, 1)
    miss_rate = 1.0 - detection_rate
    false_alarm = 1.0 - gorilla_absent_correct / max(gorilla_absent_total, 1)

    return {
        'digit_acc': digit_correct / total,
        'gorilla_acc': gorilla_correct / total,
        'detection_rate': detection_rate,
        'miss_rate': miss_rate,
        'false_alarm_rate': false_alarm,
        'gorilla_present_n': gorilla_present_total,
        'gorilla_absent_n': gorilla_absent_total,
    }


def run_condition(label, ws_dim, train_data, fashion_train, test_data, fashion_test,
                  device, epochs=15, batch_size=64, hard_primary=False):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Workspace: {ws_dim or 'None (parallel)'}")
    print(f"  Hard primary: {hard_primary}")
    print(f"{'='*70}")

    model = InattentionalModel(hidden_dim=64, ws_dim=ws_dim, hard_primary=hard_primary).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_model(model, train_data, fashion_train, device, epochs=epochs,
                batch_size=batch_size, hard_primary=hard_primary)
    metrics = evaluate(model, test_data, fashion_test, device, hard_primary=hard_primary)

    print(f"\n  Digit accuracy:      {metrics['digit_acc']:.4f}")
    print(f"  Gorilla detection:   {metrics['detection_rate']:.4f}")
    print(f"  Gorilla MISS rate:   {metrics['miss_rate']:.4f}")
    print(f"  False alarm rate:    {metrics['false_alarm_rate']:.4f}")

    return {'label': label, 'ws_dim': ws_dim, 'n_params': n_params,
            'hard_primary': hard_primary, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2029] Device: {device}")

    from torchvision import datasets
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    print("Loading MNIST + FashionMNIST...")
    mnist_train = datasets.MNIST(str(data_dir), train=True, download=True)
    mnist_test = datasets.MNIST(str(data_dir), train=False, download=True)
    fashion_train = datasets.FashionMNIST(str(data_dir), train=True, download=True)
    fashion_test = datasets.FashionMNIST(str(data_dir), train=False, download=True)

    print(f"\n{'='*70}")
    print(f"  z2029: Inattentional Blindness")
    print(f"  Consciousness predicts MISSING unexpected stimuli")
    print(f"  NO consciousness losses — blindness must EMERGE")
    print(f"{'='*70}")

    results = {}

    results['A'] = run_condition('A: Narrow workspace (16 dim)', 16,
        mnist_train, fashion_train, mnist_test, fashion_test, device, args.epochs, args.batch_size)
    results['B'] = run_condition('B: Wide workspace (128 dim)', 128,
        mnist_train, fashion_train, mnist_test, fashion_test, device, args.epochs, args.batch_size)
    results['C'] = run_condition('C: No workspace (parallel)', None,
        mnist_train, fashion_train, mnist_test, fashion_test, device, args.epochs, args.batch_size)
    results['D'] = run_condition('D: Narrow + hard primary', 16,
        mnist_train, fashion_train, mnist_test, fashion_test, device, args.epochs, args.batch_size,
        hard_primary=True)

    # Analysis
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Inattentional Blindness")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<40} {'DigitAcc':>9} {'MissRate':>9} {'Detection':>10}")
    print(f"  {'-'*68}")
    for k in ['A', 'B', 'C', 'D']:
        r = results[k]
        print(f"  {r['label']:<40} {r['digit_acc']:>9.4f} {r['miss_rate']:>9.4f} {r['detection_rate']:>10.4f}")

    t1 = results['A']['miss_rate'] > results['C']['miss_rate'] + 0.05  # Workspace misses more
    t2 = results['C']['miss_rate'] < 0.2  # No-workspace detects most
    t3 = results['A']['miss_rate'] > results['B']['miss_rate']  # Narrow > wide miss rate
    t4 = results['D']['miss_rate'] > results['A']['miss_rate']  # High load → more misses

    print(f"\n  T1: Workspace misses more than no-ws:    {'PASS' if t1 else 'FAIL'} ({results['A']['miss_rate']:.3f} vs {results['C']['miss_rate']:.3f})")
    print(f"  T2: No-workspace detects (miss<0.2):      {'PASS' if t2 else 'FAIL'} ({results['C']['miss_rate']:.3f})")
    print(f"  T3: Narrow > wide miss rate:              {'PASS' if t3 else 'FAIL'} ({results['A']['miss_rate']:.3f} vs {results['B']['miss_rate']:.3f})")
    print(f"  T4: High load → more misses:              {'PASS' if t4 else 'FAIL'} ({results['D']['miss_rate']:.3f} vs {results['A']['miss_rate']:.3f})")

    n_pass = sum([t1, t2, t3, t4])
    verdict = {0: "NO_BLINDNESS", 1: "WEAK", 2: "PARTIAL", 3: "MOSTLY_CONFIRMED", 4: "GENUINE_INATTENTIONAL_BLINDNESS"}[n_pass]
    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    output = {
        'experiment': 'z2029_inattentional_blindness',
        'hypothesis': 'Workspace creates selective attention → misses unexpected stimuli',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': ['Simons & Chabris 1999', 'Mack & Rock 1998', 'Most et al. 2005'],
        'conditions': results,
        'tests': {'t1': bool(t1), 't2': bool(t2), 't3': bool(t3), 't4': bool(t4)},
        'tests_passed': n_pass, 'verdict': verdict,
    }

    def json_safe(obj):
        if isinstance(obj, (np.bool_, np.integer)): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list): return [json_safe(v) for v in obj]
        return obj

    rp = Path(__file__).parent.parent / 'results' / 'z2029_inattentional_blindness.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
