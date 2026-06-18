#!/usr/bin/env python3
"""
z2033: Backward Masking (Temporal Window of Consciousness)

Tests whether workspace creates a temporal integration window that can be
disrupted by masking — a hallmark of conscious visual processing.

Setup:
  - Brief target stimulus (1-2 frames of MNIST digit)
  - Variable-delay mask (noise burst)
  - Measure: target identification accuracy as function of SOA (stimulus onset asynchrony)
  - Workspace models should show STEEP drop at short SOA (mask disrupts broadcasting)
  - No-workspace models should show GRADUAL degradation (no integration window to disrupt)

Predictions:
  - Workspace: sigmoid accuracy curve with steep transition at ~2-4 frames SOA
  - No-workspace: linear or gradual accuracy curve
  - Narrow workspace: steeper transition (tighter integration window)
  - Critical test: workspace accuracy at short SOA should be LOWER (masking works)

References:
  Breitmeyer & Ogmen 2006 — Visual masking
  Dehaene et al. 2006 — Conscious, preconscious, subliminal
  Del Cul et al. 2007 — Masking threshold correlates with consciousness

NO consciousness losses. Masking dynamics must EMERGE from architecture.
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


class MaskedSequenceDataset(Dataset):
    """Sequences: target(1-2 frames) → delay → mask → blank → report target class."""
    def __init__(self, base_dataset, soa_frames=3, seq_len=12, n_samples=10000, seed=42):
        self.base = base_dataset
        self.soa = soa_frames
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)

        self.indices = self.rng.randint(0, len(base_dataset), n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        # img shape: [1, 28, 28]

        # Build sequence of frames: [seq_len, 1, 28, 28]
        frames = torch.zeros(self.seq_len, 1, 28, 28)

        # Frame 0: target stimulus
        frames[0] = img

        # Frame 1 (if target duration > 1): also target
        if self.seq_len > 1:
            frames[1] = img

        # Frame at soa: mask (noise burst)
        mask_frame = min(self.soa, self.seq_len - 1)
        frames[mask_frame] = torch.randn(1, 28, 28) * 0.5 + 0.5
        frames[mask_frame] = frames[mask_frame].clamp(0, 1)

        # Also add mask at mask_frame+1 if room
        if mask_frame + 1 < self.seq_len:
            frames[mask_frame + 1] = torch.randn(1, 28, 28) * 0.5 + 0.5
            frames[mask_frame + 1] = frames[mask_frame + 1].clamp(0, 1)

        # Remaining frames: blank (zero)

        return frames, label


class RecurrentWorkspace(nn.Module):
    """Processes temporal sequence through workspace bottleneck."""
    def __init__(self, ws_dim=32):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.rnn = nn.GRU(32, ws_dim, batch_first=True)
        self.ws_ln = nn.LayerNorm(ws_dim)
        self.classifier = nn.Linear(ws_dim, 10)

    def forward(self, frames):
        # frames: [B, T, 1, 28, 28]
        B, T = frames.shape[:2]
        flat = frames.view(B * T, 1, 28, 28)
        feats = self.frame_encoder(flat).view(B, T, -1)  # [B, T, 32]
        output, h_n = self.rnn(feats)
        ws = self.ws_ln(h_n.squeeze(0))
        logits = self.classifier(ws)
        return {'logits': logits, 'hidden': ws}


class FeedforwardModel(nn.Module):
    """Processes only first frame (no temporal integration)."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(32, hidden_dim), nn.ReLU())
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, frames):
        # Only use first frame
        first = frames[:, 0]  # [B, 1, 28, 28]
        h = self.frame_encoder(first).view(first.size(0), -1)
        feat = self.fc(h)
        logits = self.classifier(feat)
        return {'logits': logits, 'hidden': feat}


def train_at_soa(model, device, base_train, soa, epochs=10, batch_size=128):
    dataset = MaskedSequenceDataset(base_train, soa_frames=soa, n_samples=15000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(frames)
            loss = F.cross_entropy(out['logits'], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * frames.size(0)
            correct += (out['logits'].argmax(1) == labels).sum().item()
            total += frames.size(0)


@torch.no_grad()
def evaluate_at_soa(model, device, base_test, soa, n_test=3000):
    dataset = MaskedSequenceDataset(base_test, soa_frames=soa, n_samples=n_test, seed=99)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()

    correct, total = 0, 0
    all_conf = []
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        out = model(frames)
        preds = out['logits'].argmax(1)
        correct += (preds == labels).sum().item()
        total += frames.size(0)
        probs = F.softmax(out['logits'], dim=-1)
        all_conf.extend(probs.max(1).values.cpu().numpy())

    return {
        'accuracy': correct / total,
        'mean_confidence': float(np.mean(all_conf)),
    }


def compute_steepness(soa_values, accuracies):
    """Compute max slope of accuracy curve (steeper = more threshold-like)."""
    if len(soa_values) < 2:
        return 0.0
    slopes = []
    for i in range(len(soa_values) - 1):
        dx = soa_values[i + 1] - soa_values[i]
        dy = accuracies[i + 1] - accuracies[i]
        if dx > 0:
            slopes.append(dy / dx)
    return float(max(slopes)) if slopes else 0.0


def run_condition(label, model_fn, device, base_train, base_test, soa_list, epochs=10, batch_size=128):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    soa_results = {}
    for soa in soa_list:
        model = model_fn().to(device)
        train_at_soa(model, device, base_train, soa, epochs=epochs, batch_size=batch_size)
        metrics = evaluate_at_soa(model, device, base_test, soa)
        soa_results[soa] = metrics
        print(f"  SOA={soa:2d}: acc={metrics['accuracy']:.4f}  conf={metrics['mean_confidence']:.4f}")

    accuracies = [soa_results[s]['accuracy'] for s in soa_list]
    steepness = compute_steepness(soa_list, accuracies)
    masking_effect = soa_results[soa_list[-1]]['accuracy'] - soa_results[soa_list[0]]['accuracy']

    print(f"  Steepness: {steepness:.4f}")
    print(f"  Masking effect: {masking_effect:+.4f}")

    return {
        'label': label,
        'soa_results': {str(k): v for k, v in soa_results.items()},
        'steepness': steepness,
        'masking_effect': masking_effect,
        'accuracies': accuracies,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2033] Device: {device}")

    from torchvision import datasets, transforms
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    tf = transforms.ToTensor()
    train_base = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    test_base = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)

    print(f"\n{'='*70}")
    print(f"  z2033: Backward Masking (Temporal Window)")
    print(f"  Workspace should show STEEP masking threshold")
    print(f"  NO consciousness losses — masking must EMERGE")
    print(f"{'='*70}")

    soa_list = [1, 2, 3, 4, 6, 8, 10]

    results = {}
    results['A'] = run_condition(
        'A: Narrow workspace (16)', lambda: RecurrentWorkspace(ws_dim=16),
        device, train_base, test_base, soa_list, args.epochs, args.batch_size
    )
    results['B'] = run_condition(
        'B: Wide workspace (64)', lambda: RecurrentWorkspace(ws_dim=64),
        device, train_base, test_base, soa_list, args.epochs, args.batch_size
    )
    results['C'] = run_condition(
        'C: Feedforward (first frame only)', lambda: FeedforwardModel(),
        device, train_base, test_base, soa_list, args.epochs, args.batch_size
    )

    # Analysis
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Backward Masking")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<40} {'Steepness':>10} {'MaskEffect':>11}")
    print(f"  {'-'*61}")
    for k in ['A', 'B', 'C']:
        r = results[k]
        print(f"  {r['label']:<40} {r['steepness']:>10.4f} {r['masking_effect']:>+11.4f}")

    t1 = results['A']['steepness'] > results['C']['steepness']  # Workspace steeper
    t2 = results['A']['masking_effect'] > 0.05  # Masking has measurable effect
    t3 = results['A']['steepness'] > results['B']['steepness']  # Narrow steeper than wide
    t4_short_soa = results['A']['soa_results']['1']['accuracy']
    t4_long_soa = results['A']['soa_results']['10']['accuracy']
    t4 = t4_long_soa - t4_short_soa > 0.1  # Large SOA range effect

    print(f"\n  T1: Workspace steeper than FF:          {'PASS' if t1 else 'FAIL'} ({results['A']['steepness']:.4f} vs {results['C']['steepness']:.4f})")
    print(f"  T2: Masking effect >5%:                 {'PASS' if t2 else 'FAIL'} ({results['A']['masking_effect']:+.4f})")
    print(f"  T3: Narrow > wide steepness:            {'PASS' if t3 else 'FAIL'} ({results['A']['steepness']:.4f} vs {results['B']['steepness']:.4f})")
    print(f"  T4: Large SOA range effect (>10%):      {'PASS' if t4 else 'FAIL'} ({t4_short_soa:.3f} → {t4_long_soa:.3f})")

    n_pass = sum([t1, t2, t3, t4])
    verdict = {0: "NO_MASKING", 1: "WEAK", 2: "PARTIAL", 3: "MOSTLY", 4: "GENUINE_MASKING_CONFIRMED"}[n_pass]
    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    output = {
        'experiment': 'z2033_backward_masking',
        'hypothesis': 'Workspace creates temporal integration window disrupted by masking',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': ['Breitmeyer & Ogmen 2006', 'Dehaene et al. 2006', 'Del Cul et al. 2007'],
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

    rp = Path(__file__).parent.parent / 'results' / 'z2033_backward_masking.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
