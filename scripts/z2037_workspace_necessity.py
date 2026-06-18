#!/usr/bin/env python3
"""
z2037: Workspace Necessity (Causal Intervention)

Tests CAUSAL NECESSITY: does the workspace actually DO something,
or is it just a passthrough? Uses the strongest test paradigm:
ablate the workspace at test time and measure SPECIFIC degradation patterns.

This is different from z2021's self-model ablation — here we ablate
the WORKSPACE ITSELF (not the self-model) and measure what breaks.

Setup:
  - Train workspace model on composite task requiring integration
  - At test time, ablate workspace in 5 ways:
    1. Full model (baseline)
    2. Zero workspace (set to zeros)
    3. Random workspace (replace with noise)
    4. Frozen workspace (fix to training mean)
    5. Narrow workspace (reduce dimensionality)
  - Measure: accuracy, confidence calibration, representation quality

Predictions:
  - Zero/random workspace: accuracy drops, BUT encoder still works
  - Frozen workspace: some accuracy preserved (static features)
  - Accuracy drop should be LARGER for tasks requiring integration

References:
  Pearl 2009 — Causal inference (do-calculus)
  Phua 2025 — Ablation-based consciousness markers
  Luppi et al. 2024 — Workspace as integration medium

NO consciousness losses. Workspace necessity must be CAUSAL.
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


class IntegrationTask(Dataset):
    """Task requiring integration of multiple digit properties."""
    def __init__(self, base_dataset, task='composite', seed=42):
        self.base = base_dataset
        self.task = task

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]

        if self.task == 'simple':
            # Simple: just even/odd (no integration needed)
            target = label % 2
        elif self.task == 'composite':
            # Composite: (digit > 4) XOR (digit is even) — requires two properties
            target = int((label > 4) != (label % 2 == 0))
        elif self.task == 'triple':
            # Triple: combines three properties
            target = int((label > 4) != (label % 2 == 0) != (label % 3 == 0))
        else:
            target = label % 2

        return img, target


class WorkspaceNet(nn.Module):
    def __init__(self, ws_dim=32, n_classes=2):
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
        self.classifier = nn.Linear(ws_dim, n_classes)
        self.ws_dim = ws_dim

    def forward(self, x, ws_override=None):
        h = self.encoder(x).view(x.size(0), -1)
        if ws_override is not None:
            ws = ws_override
        else:
            ws = self.workspace(h)
        logits = self.classifier(ws)
        return {'logits': logits, 'workspace': ws, 'pre_ws': h}


def train_model(model, loader, device, epochs=15, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        t0 = time.time()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = F.cross_entropy(out['logits'], labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (out['logits'].argmax(1) == labels).sum().item()
            total += images.size(0)
        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  loss={total_loss/total:.4f}  acc={correct/total:.3f}  ({elapsed:.1f}s)")


@torch.no_grad()
def collect_ws_stats(model, loader, device):
    """Collect workspace statistics for frozen/mean ablation."""
    model.eval()
    all_ws = []
    for images, labels in loader:
        images = images.to(device)
        out = model(images)
        all_ws.append(out['workspace'].cpu())
    return torch.cat(all_ws)


@torch.no_grad()
def evaluate_condition(model, loader, device, ws_mode='normal', ws_stats=None):
    model.eval()
    correct, total = 0, 0
    all_conf = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)

        if ws_mode == 'normal':
            out = model(images)
        elif ws_mode == 'zero':
            h = model.encoder(images).view(B, -1)
            ws = torch.zeros(B, model.ws_dim, device=device)
            out = model(images, ws_override=ws)
        elif ws_mode == 'random':
            h = model.encoder(images).view(B, -1)
            ws = torch.randn(B, model.ws_dim, device=device)
            out = model(images, ws_override=ws)
        elif ws_mode == 'frozen':
            ws_mean = ws_stats.mean(dim=0).to(device).unsqueeze(0).expand(B, -1)
            out = model(images, ws_override=ws_mean)
        elif ws_mode == 'noisy':
            out_normal = model(images)
            ws = out_normal['workspace'] + torch.randn_like(out_normal['workspace']) * 0.5
            out = model(images, ws_override=ws)
        else:
            out = model(images)

        preds = out['logits'].argmax(1)
        correct += (preds == labels).sum().item()
        total += B
        probs = F.softmax(out['logits'], dim=-1)
        all_conf.extend(probs.max(1).values.cpu().numpy())

    return {
        'accuracy': correct / total,
        'mean_confidence': float(np.mean(all_conf)),
    }


def run_task_level(task_name, model, device, train_base, test_base, epochs=15, batch_size=128):
    print(f"\n  --- Task: {task_name} ---")

    train_data = IntegrationTask(train_base, task=task_name)
    test_data = IntegrationTask(test_base, task=task_name)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    train_model(model, train_loader, device, epochs=epochs)

    # Collect workspace stats for frozen condition
    ws_stats = collect_ws_stats(model, train_loader, device)

    conditions = {}
    for mode in ['normal', 'zero', 'random', 'frozen', 'noisy']:
        result = evaluate_condition(model, test_loader, device, ws_mode=mode, ws_stats=ws_stats)
        conditions[mode] = result
        print(f"    {mode:>8}: acc={result['accuracy']:.4f}  conf={result['mean_confidence']:.4f}")

    # Compute necessity score: how much does ablation hurt?
    baseline = conditions['normal']['accuracy']
    zero_drop = baseline - conditions['zero']['accuracy']
    random_drop = baseline - conditions['random']['accuracy']
    frozen_drop = baseline - conditions['frozen']['accuracy']

    return {
        'task': task_name,
        'conditions': conditions,
        'baseline_acc': baseline,
        'zero_drop': zero_drop,
        'random_drop': random_drop,
        'frozen_drop': frozen_drop,
        'necessity_score': (zero_drop + random_drop) / 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2037] Device: {device}")

    from torchvision import datasets, transforms
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    tf = transforms.ToTensor()
    train_base = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    test_base = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)

    print(f"\n{'='*70}")
    print(f"  z2037: Workspace Necessity (Causal Intervention)")
    print(f"  Does the workspace actually DO something? Ablate and measure.")
    print(f"  NO consciousness losses — necessity must be CAUSAL.")
    print(f"{'='*70}")

    results = {}

    # Test on simple task (less integration needed)
    model_simple = WorkspaceNet(ws_dim=32, n_classes=2).to(device)
    results['simple'] = run_task_level('simple', model_simple, device, train_base, test_base,
                                       epochs=args.epochs, batch_size=args.batch_size)

    # Test on composite task (more integration needed)
    model_composite = WorkspaceNet(ws_dim=32, n_classes=2).to(device)
    results['composite'] = run_task_level('composite', model_composite, device, train_base, test_base,
                                          epochs=args.epochs, batch_size=args.batch_size)

    # Test on triple task (most integration needed)
    model_triple = WorkspaceNet(ws_dim=32, n_classes=2).to(device)
    results['triple'] = run_task_level('triple', model_triple, device, train_base, test_base,
                                       epochs=args.epochs, batch_size=args.batch_size)

    # Analysis
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Workspace Necessity")
    print(f"{'='*70}")

    print(f"\n  {'Task':<12} {'Baseline':>9} {'ZeroDrop':>9} {'RandDrop':>9} {'FrozDrop':>9} {'Necessity':>10}")
    print(f"  {'-'*58}")
    for task in ['simple', 'composite', 'triple']:
        r = results[task]
        print(f"  {task:<12} {r['baseline_acc']:>9.4f} {r['zero_drop']:>+9.4f} "
              f"{r['random_drop']:>+9.4f} {r['frozen_drop']:>+9.4f} {r['necessity_score']:>10.4f}")

    t1 = results['composite']['necessity_score'] > 0.05  # Workspace necessary for composite
    t2 = results['composite']['necessity_score'] > results['simple']['necessity_score']  # More necessary for harder task
    t3 = results['composite']['conditions']['frozen']['accuracy'] > results['composite']['conditions']['zero']['accuracy']  # Frozen > zero
    t4 = results['composite']['conditions']['normal']['accuracy'] > 0.7  # Model actually learns

    print(f"\n  T1: Workspace necessary (>5% drop):    {'PASS' if t1 else 'FAIL'} ({results['composite']['necessity_score']:.4f})")
    print(f"  T2: More necessary for harder task:     {'PASS' if t2 else 'FAIL'} ({results['composite']['necessity_score']:.4f} vs {results['simple']['necessity_score']:.4f})")
    print(f"  T3: Frozen > zero (static features):    {'PASS' if t3 else 'FAIL'}")
    print(f"  T4: Model learns composite (>70%):      {'PASS' if t4 else 'FAIL'} ({results['composite']['baseline_acc']:.3f})")

    n_pass = sum([t1, t2, t3, t4])
    verdict = {0: "NOT_NECESSARY", 1: "WEAK", 2: "PARTIAL", 3: "MOSTLY", 4: "WORKSPACE_CAUSALLY_NECESSARY"}[n_pass]
    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    output = {
        'experiment': 'z2037_workspace_necessity',
        'hypothesis': 'Workspace is causally necessary, especially for integration tasks',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': ['Pearl 2009 (Causal inference)', 'Phua 2025', 'Luppi et al. 2024'],
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

    rp = Path(__file__).parent.parent / 'results' / 'z2037_workspace_necessity.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
