#!/usr/bin/env python3
"""
z2032: Binocular Rivalry (Bistable Perception)

Tests whether a limited-capacity workspace exhibits winner-take-all dynamics
when presented with two conflicting stimuli — analogous to binocular rivalry
where consciousness alternates between two incompatible percepts.

Setup:
  - Present TWO conflicting MNIST digits simultaneously (e.g., "3" and "7")
  - Model must classify one or the other (not both)
  - Workspace model should show BISTABLE behavior: high confidence for one,
    suppression of the other, with clean switching
  - No-workspace model should show mixed/blended responses

Predictions (GWT-based):
  - Workspace: winner-take-all dynamics — one percept dominates, other suppressed
  - No-workspace: mixed responses, both percepts partially active
  - Narrow workspace → stronger rivalry (more suppression)
  - Repeated presentations → stable percept preference (persistence)

References:
  Blake & Logothetis 2002 — Binocular rivalry review
  Dehaene & Changeux 2011 — GWT and winner-take-all broadcasting
  Tononi & Koch 2008 — Neural correlates of consciousness

NO consciousness losses. Rivalry must EMERGE from architecture + task.
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


class RivalryDataset(Dataset):
    """Two conflicting MNIST digits presented simultaneously."""
    def __init__(self, base_dataset, n_samples=20000, seed=42):
        self.base = base_dataset
        self.n_samples = n_samples
        rng = np.random.RandomState(seed)

        self.indices_a = rng.randint(0, len(base_dataset), n_samples)
        self.indices_b = rng.randint(0, len(base_dataset), n_samples)

        # Ensure conflicting (different) labels
        for i in range(n_samples):
            while self.base[self.indices_b[i]][1] == self.base[self.indices_a[i]][1]:
                self.indices_b[i] = rng.randint(0, len(base_dataset))

        # Randomly choose which is the "correct" answer (50/50)
        self.target_is_a = rng.random(n_samples) < 0.5

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img_a, label_a = self.base[self.indices_a[idx]]
        img_b, label_b = self.base[self.indices_b[idx]]

        # Overlay: average the two images (simulates rivalry)
        combined = (img_a + img_b) / 2.0

        # Target: whichever was designated
        if self.target_is_a[idx]:
            target = label_a
            other = label_b
        else:
            target = label_b
            other = label_a

        return combined, target, other


class WorkspaceEncoder(nn.Module):
    def __init__(self, ws_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ws_bottleneck = nn.Sequential(
            nn.Linear(128, ws_dim),
            nn.ReLU(),
            nn.LayerNorm(ws_dim),
        )
        self.classifier = nn.Linear(ws_dim, 10)

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        ws = self.ws_bottleneck(h)
        logits = self.classifier(ws)
        return {'logits': logits, 'hidden': ws, 'pre_ws': h}


class NoWorkspaceEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        feat = self.fc(h)
        logits = self.classifier(feat)
        return {'logits': logits, 'hidden': feat, 'pre_ws': h}


def train_model(model, loader, device, epochs=15, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        t0 = time.time()
        for combined, target, other in loader:
            combined, target = combined.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(combined)
            loss = F.cross_entropy(out['logits'], target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * combined.size(0)
            correct += (out['logits'].argmax(1) == target).sum().item()
            total += combined.size(0)
        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  loss={total_loss/total:.4f}  acc={correct/total:.3f}  ({elapsed:.1f}s)")


@torch.no_grad()
def evaluate_rivalry(model, loader, device):
    """Measure winner-take-all dynamics."""
    model.eval()
    all_max_prob = []
    all_entropy = []
    all_suppression = []  # prob(other) / prob(target)
    correct, total = 0, 0

    for combined, target, other in loader:
        combined, target, other = combined.to(device), target.to(device), other.to(device)
        out = model(combined)
        probs = F.softmax(out['logits'], dim=-1)

        # Max probability (winner strength)
        max_prob = probs.max(dim=-1).values
        all_max_prob.extend(max_prob.cpu().numpy())

        # Entropy (low = decisive, high = mixed)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        all_entropy.extend(entropy.cpu().numpy())

        # Suppression ratio: how much is the OTHER percept suppressed?
        target_prob = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        other_prob = probs.gather(1, other.unsqueeze(1)).squeeze(1)
        suppression = other_prob / (target_prob + 1e-10)
        all_suppression.extend(suppression.cpu().numpy())

        correct += (out['logits'].argmax(1) == target).sum().item()
        total += combined.size(0)

    max_prob = np.array(all_max_prob)
    entropy = np.array(all_entropy)
    suppression = np.array(all_suppression)

    # Bistability: measure bimodality of confidence distribution
    # High confidence (>0.8) vs low confidence (<0.3)
    high_conf = (max_prob > 0.8).mean()
    low_conf = (max_prob < 0.3).mean()
    mid_conf = 1.0 - high_conf - low_conf

    return {
        'accuracy': correct / total,
        'mean_max_prob': float(max_prob.mean()),
        'std_max_prob': float(max_prob.std()),
        'mean_entropy': float(entropy.mean()),
        'mean_suppression': float(suppression.mean()),
        'high_conf_frac': float(high_conf),
        'mid_conf_frac': float(mid_conf),
        'low_conf_frac': float(low_conf),
        'winner_take_all': float(high_conf) > 0.5,  # Most responses are decisive
    }


def run_condition(label, model, device, train_loader, test_loader, epochs=15):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_model(model, train_loader, device, epochs=epochs)
    metrics = evaluate_rivalry(model, test_loader, device)

    print(f"\n  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Mean max prob:     {metrics['mean_max_prob']:.4f}")
    print(f"  Mean entropy:      {metrics['mean_entropy']:.4f}")
    print(f"  Mean suppression:  {metrics['mean_suppression']:.4f}")
    print(f"  High conf (>0.8):  {metrics['high_conf_frac']:.3f}")
    print(f"  Mid conf:          {metrics['mid_conf_frac']:.3f}")
    print(f"  Winner-take-all:   {metrics['winner_take_all']}")

    return {'label': label, 'n_params': n_params, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2032] Device: {device}")

    from torchvision import datasets, transforms
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    tf = transforms.ToTensor()
    train_base = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    test_base = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)

    train_rivalry = RivalryDataset(train_base, n_samples=30000)
    test_rivalry = RivalryDataset(test_base, n_samples=5000, seed=99)

    train_loader = DataLoader(train_rivalry, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_rivalry, batch_size=256, shuffle=False)

    print(f"\n{'='*70}")
    print(f"  z2032: Binocular Rivalry (Bistable Perception)")
    print(f"  Workspace should show winner-take-all: one percept dominates")
    print(f"  NO consciousness losses — rivalry must EMERGE")
    print(f"{'='*70}")

    results = {}

    # A: Narrow workspace (strong bottleneck → strong rivalry)
    model_a = WorkspaceEncoder(ws_dim=16).to(device)
    results['A'] = run_condition('A: Narrow workspace (16 dim)', model_a, device, train_loader, test_loader, args.epochs)

    # B: Wide workspace (weak bottleneck)
    model_b = WorkspaceEncoder(ws_dim=128).to(device)
    results['B'] = run_condition('B: Wide workspace (128 dim)', model_b, device, train_loader, test_loader, args.epochs)

    # C: No workspace
    model_c = NoWorkspaceEncoder(hidden_dim=128).to(device)
    results['C'] = run_condition('C: No workspace', model_c, device, train_loader, test_loader, args.epochs)

    # Analysis
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Binocular Rivalry")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<30} {'Acc':>6} {'MaxProb':>8} {'Entropy':>8} {'Suppress':>9} {'WTA':>5}")
    print(f"  {'-'*66}")
    for k in ['A', 'B', 'C']:
        r = results[k]
        print(f"  {r['label']:<30} {r['accuracy']:>6.3f} {r['mean_max_prob']:>8.4f} "
              f"{r['mean_entropy']:>8.4f} {r['mean_suppression']:>9.4f} {'Y' if r['winner_take_all'] else 'N':>5}")

    # Tests
    t1 = results['A']['mean_suppression'] < results['C']['mean_suppression']  # Workspace suppresses more
    t2 = results['A']['mean_entropy'] < results['C']['mean_entropy']  # Workspace more decisive
    t3 = results['A']['mean_suppression'] < results['B']['mean_suppression']  # Narrow > wide suppression
    t4 = results['A']['high_conf_frac'] > 0.5  # Most responses are winner-take-all

    print(f"\n  T1: Workspace suppresses other more:    {'PASS' if t1 else 'FAIL'} ({results['A']['mean_suppression']:.4f} vs {results['C']['mean_suppression']:.4f})")
    print(f"  T2: Workspace lower entropy:            {'PASS' if t2 else 'FAIL'} ({results['A']['mean_entropy']:.4f} vs {results['C']['mean_entropy']:.4f})")
    print(f"  T3: Narrow > wide suppression:          {'PASS' if t3 else 'FAIL'} ({results['A']['mean_suppression']:.4f} vs {results['B']['mean_suppression']:.4f})")
    print(f"  T4: Winner-take-all (>50% high conf):   {'PASS' if t4 else 'FAIL'} ({results['A']['high_conf_frac']:.3f})")

    n_pass = sum([t1, t2, t3, t4])
    verdict = {0: "NO_RIVALRY", 1: "WEAK", 2: "PARTIAL", 3: "MOSTLY", 4: "GENUINE_RIVALRY_CONFIRMED"}[n_pass]
    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    output = {
        'experiment': 'z2032_binocular_rivalry',
        'hypothesis': 'Workspace exhibits winner-take-all rivalry dynamics',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': ['Blake & Logothetis 2002', 'Dehaene & Changeux 2011', 'Tononi & Koch 2008'],
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

    rp = Path(__file__).parent.parent / 'results' / 'z2032_binocular_rivalry.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
