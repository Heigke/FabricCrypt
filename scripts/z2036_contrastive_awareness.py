#!/usr/bin/env python3
"""
z2036: Contrastive Awareness (Seen vs Unseen Dissociation)

Tests the CONTRASTIVE aspect of consciousness: a genuinely aware system
should show qualitatively different internal states for stimuli it
successfully processes ("seen") vs stimuli it fails on ("unseen"),
AND this difference should be PREDICTABLE from internal state alone.

This combines our strongest patterns:
- Ablation-dissociation (z2021 pattern)
- Cost measurement (z2026 pattern)
- Information content (z2027 pattern)

Setup:
  - Train classifier on hard task with variable difficulty
  - AFTER training, examine internal representations:
    - For correct trials ("seen"): what does the workspace look like?
    - For incorrect trials ("unseen"): what does it look like?
  - Train a LINEAR PROBE on workspace states to predict seen/unseen
  - Compare workspace vs no-workspace models

Predictions:
  - Workspace: seen/unseen are LINEARLY SEPARABLE in workspace space
  - No-workspace: seen/unseen are NOT cleanly separable
  - Workspace seen states have LOWER entropy than unseen
  - Ablating workspace destroys seen/unseen distinction

References:
  Dehaene et al. 2006 — Conscious vs subliminal processing
  Sergent & Dehaene 2004 — All-or-none access to consciousness
  Phua 2025 — Ablation-based consciousness markers

NO consciousness losses. Awareness must EMERGE from task training.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))


class HardMNIST(Dataset):
    """MNIST with variable noise to create seen/unseen trials."""
    def __init__(self, base_dataset, noise_range=(0.0, 3.0), seed=42):
        self.base = base_dataset
        self.noise_range = noise_range
        self.rng = np.random.RandomState(seed)
        # Pre-generate noise levels per sample
        self.noise_levels = self.rng.uniform(noise_range[0], noise_range[1], len(base_dataset)).astype(np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        noise = torch.randn_like(img) * self.noise_levels[idx]
        noisy = (img + noise).clamp(0, 1)
        return noisy, label, self.noise_levels[idx]


class WorkspaceModel(nn.Module):
    def __init__(self, ws_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.workspace = nn.Sequential(
            nn.Linear(128, ws_dim), nn.LayerNorm(ws_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(ws_dim, 10)
        self.ws_dim = ws_dim

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        ws = self.workspace(h)
        logits = self.classifier(ws)
        return {'logits': logits, 'workspace': ws, 'pre_ws': h}


class DirectModel(nn.Module):
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
        return {'logits': logits, 'workspace': feat, 'pre_ws': h}


def train_model(model, loader, device, epochs=15, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        t0 = time.time()
        for noisy, labels, noise_lvl in loader:
            noisy, labels = noisy.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(noisy)
            loss = F.cross_entropy(out['logits'], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * noisy.size(0)
            correct += (out['logits'].argmax(1) == labels).sum().item()
            total += noisy.size(0)
        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  loss={total_loss/total:.4f}  acc={correct/total:.3f}  ({elapsed:.1f}s)")


@torch.no_grad()
def collect_representations(model, loader, device):
    """Collect workspace representations and correctness labels."""
    model.eval()
    all_ws, all_correct, all_noise = [], [], []

    for noisy, labels, noise_lvl in loader:
        noisy, labels = noisy.to(device), labels.to(device)
        out = model(noisy)
        preds = out['logits'].argmax(1)
        correct = (preds == labels).cpu().numpy().astype(float)
        ws = out['workspace'].cpu().numpy()

        all_ws.append(ws)
        all_correct.append(correct)
        all_noise.append(noise_lvl.numpy())

    return {
        'workspace': np.concatenate(all_ws),
        'correct': np.concatenate(all_correct),
        'noise': np.concatenate(all_noise),
    }


def analyze_awareness(reps):
    """Analyze seen/unseen separability in workspace space."""
    ws = reps['workspace']
    correct = reps['correct']

    n_correct = int(correct.sum())
    n_wrong = int((1 - correct).sum())

    if n_correct < 10 or n_wrong < 10:
        return {
            'probe_auroc': 0.5,
            'entropy_seen': 0.0,
            'entropy_unseen': 0.0,
            'entropy_gap': 0.0,
            'norm_seen': 0.0,
            'norm_unseen': 0.0,
            'n_correct': n_correct,
            'n_wrong': n_wrong,
        }

    # 1. Linear probe: can we predict seen/unseen from workspace?
    from sklearn.model_selection import cross_val_score
    probe = LogisticRegression(max_iter=1000, C=1.0)
    # Use cross-validation for robust estimate
    scores = cross_val_score(probe, ws, correct, cv=5, scoring='roc_auc')
    probe_auroc = float(scores.mean())

    # 2. Entropy comparison
    ws_abs = np.abs(ws) + 1e-10
    ws_dist = ws_abs / ws_abs.sum(axis=1, keepdims=True)
    entropy = -(ws_dist * np.log(ws_dist)).sum(axis=1)

    entropy_seen = float(entropy[correct == 1].mean())
    entropy_unseen = float(entropy[correct == 0].mean())

    # 3. Norm comparison
    norms = np.linalg.norm(ws, axis=1)
    norm_seen = float(norms[correct == 1].mean())
    norm_unseen = float(norms[correct == 0].mean())

    # 4. Dimensionality (effective rank)
    from numpy.linalg import svd
    ws_seen = ws[correct == 1]
    ws_unseen = ws[correct == 0]

    def effective_dim(X):
        if len(X) < 2:
            return 0.0
        X_centered = X - X.mean(axis=0)
        _, s, _ = svd(X_centered, full_matrices=False)
        s = s / s.sum()
        return float(np.exp(-(s * np.log(s + 1e-10)).sum()))

    dim_seen = effective_dim(ws_seen)
    dim_unseen = effective_dim(ws_unseen)

    return {
        'probe_auroc': probe_auroc,
        'entropy_seen': entropy_seen,
        'entropy_unseen': entropy_unseen,
        'entropy_gap': entropy_unseen - entropy_seen,
        'norm_seen': norm_seen,
        'norm_unseen': norm_unseen,
        'dim_seen': dim_seen,
        'dim_unseen': dim_unseen,
        'n_correct': n_correct,
        'n_wrong': n_wrong,
        'accuracy': n_correct / (n_correct + n_wrong),
    }


def run_condition(label, model, device, train_loader, test_loader, epochs=15):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_model(model, train_loader, device, epochs=epochs)
    reps = collect_representations(model, test_loader, device)
    analysis = analyze_awareness(reps)

    print(f"\n  Accuracy:        {analysis['accuracy']:.4f} ({analysis['n_correct']} correct, {analysis['n_wrong']} wrong)")
    print(f"  Probe AUROC:     {analysis['probe_auroc']:.4f}")
    print(f"  Entropy seen:    {analysis['entropy_seen']:.4f}")
    print(f"  Entropy unseen:  {analysis['entropy_unseen']:.4f}")
    print(f"  Entropy gap:     {analysis['entropy_gap']:+.4f}")
    print(f"  Norm seen:       {analysis['norm_seen']:.4f}")
    print(f"  Norm unseen:     {analysis['norm_unseen']:.4f}")

    return {'label': label, 'n_params': n_params, **analysis}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2036] Device: {device}")

    from torchvision import datasets, transforms
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    tf = transforms.ToTensor()
    train_base = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    test_base = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)

    # High noise for meaningful seen/unseen split
    train_data = HardMNIST(train_base, noise_range=(0.0, 3.0))
    test_data = HardMNIST(test_base, noise_range=(0.0, 3.0), seed=99)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    print(f"\n{'='*70}")
    print(f"  z2036: Contrastive Awareness")
    print(f"  Seen vs unseen should be separable in workspace space")
    print(f"  NO consciousness losses — separability must EMERGE")
    print(f"{'='*70}")

    results = {}

    results['A'] = run_condition('A: Narrow workspace (16)',
        WorkspaceModel(ws_dim=16).to(device), device, train_loader, test_loader, args.epochs)

    results['B'] = run_condition('B: Wide workspace (64)',
        WorkspaceModel(ws_dim=64).to(device), device, train_loader, test_loader, args.epochs)

    results['C'] = run_condition('C: No workspace (128)',
        DirectModel(hidden_dim=128).to(device), device, train_loader, test_loader, args.epochs)

    # Analysis
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Contrastive Awareness")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<30} {'Acc':>6} {'Probe':>7} {'EntrGap':>8} {'NormS':>7} {'NormU':>7}")
    print(f"  {'-'*65}")
    for k in ['A', 'B', 'C']:
        r = results[k]
        print(f"  {r['label']:<30} {r['accuracy']:>6.3f} {r['probe_auroc']:>7.4f} "
              f"{r['entropy_gap']:>+8.4f} {r['norm_seen']:>7.3f} {r['norm_unseen']:>7.3f}")

    t1 = results['A']['probe_auroc'] > 0.6  # Seen/unseen separable
    t2 = results['A']['probe_auroc'] > results['C']['probe_auroc']  # Workspace more separable
    t3 = results['A']['entropy_gap'] > 0  # Unseen higher entropy
    t4 = results['A']['accuracy'] < 0.95  # Task is actually hard

    print(f"\n  T1: Seen/unseen separable (AUROC>0.6): {'PASS' if t1 else 'FAIL'} ({results['A']['probe_auroc']:.4f})")
    print(f"  T2: Workspace more separable than FF:   {'PASS' if t2 else 'FAIL'} ({results['A']['probe_auroc']:.4f} vs {results['C']['probe_auroc']:.4f})")
    print(f"  T3: Unseen has higher entropy:          {'PASS' if t3 else 'FAIL'} ({results['A']['entropy_gap']:+.4f})")
    print(f"  T4: Task is actually hard (<95%):       {'PASS' if t4 else 'FAIL'} ({results['A']['accuracy']:.3f})")

    n_pass = sum([t1, t2, t3, t4])
    verdict = {0: "NO_AWARENESS", 1: "WEAK", 2: "PARTIAL", 3: "MOSTLY", 4: "CONTRASTIVE_AWARENESS_CONFIRMED"}[n_pass]
    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    output = {
        'experiment': 'z2036_contrastive_awareness',
        'hypothesis': 'Workspace creates linearly separable seen/unseen representations',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': ['Dehaene et al. 2006', 'Sergent & Dehaene 2004', 'Phua 2025'],
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

    rp = Path(__file__).parent.parent / 'results' / 'z2036_contrastive_awareness.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
