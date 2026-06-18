#!/usr/bin/env python3
"""
z2034: Workspace Cost Scaling (Weber-Fechner Law for Consciousness)

Tests a COST-based prediction: the capacity cost of workspace broadcasting
should scale predictably with the amount of information being integrated.
This follows our design pattern — tests measuring COSTS are unforgeable.

Setup:
  - Classify composite stimuli requiring integration of 1, 2, 3, 4, or 5 features
  - Measure: workspace utilization cost (entropy, dimensionality used)
  - Prediction: cost scales logarithmically with feature count (Weber-Fechner)
  - No-workspace: cost should be flat (no broadcasting overhead)

Predictions:
  - Workspace cost increases monotonically with feature count
  - Workspace accuracy DECREASES with more features (capacity limit)
  - No-workspace accuracy constant or decreasing less
  - Cost scaling approximately logarithmic (Weber-Fechner style)

References:
  Baars 2005 — Global workspace capacity
  Cowan 2001 — Magical number four in short-term memory
  Dehaene et al. 2003 — Workspace capacity limitations

NO consciousness losses. Cost scaling must EMERGE.
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


class MultiFeatureDataset(Dataset):
    """Stimuli requiring integration of N features to classify."""
    def __init__(self, n_features, n_samples=15000, input_dim=32, seed=42):
        rng = np.random.RandomState(seed)
        self.n_features = n_features

        # Generate N independent features
        self.features = rng.randn(n_samples, n_features, input_dim).astype(np.float32)

        # Target depends on ALL n_features jointly
        # Use XOR-like combination: target = sum of signs of feature means
        feature_signs = (self.features.mean(axis=2) > 0).astype(np.int64)  # [n_samples, n_features]

        # Binary target: XOR of all feature signs (requires integration)
        self.targets = feature_signs.sum(axis=1) % 2  # Even/odd count

        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Concatenate all features into single input
        x = torch.tensor(self.features[idx].flatten())
        y = torch.tensor(self.targets[idx])
        return x, y


class WorkspaceIntegrator(nn.Module):
    def __init__(self, input_dim, ws_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.workspace = nn.Sequential(
            nn.Linear(64, ws_dim),
            nn.LayerNorm(ws_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(ws_dim, 2)

    def forward(self, x):
        h = self.encoder(x)
        ws = self.workspace(h)
        logits = self.classifier(ws)
        return {'logits': logits, 'workspace': ws, 'pre_ws': h}


class DirectClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        h = self.net[:-1](x)
        logits = self.net[-1](h)
        return {'logits': logits, 'workspace': h, 'pre_ws': h}


def train_model(model, loader, device, epochs=15, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out['logits'], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out['logits'].argmax(1) == y).sum().item()
            total += x.size(0)


@torch.no_grad()
def evaluate_cost(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_ws_norms = []
    all_ws_entropy = []
    all_ws_dims_used = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = out['logits'].argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        ws = out['workspace']  # [B, ws_dim]

        # Cost metric 1: L2 norm (activation intensity)
        norms = ws.norm(dim=-1)
        all_ws_norms.extend(norms.cpu().numpy())

        # Cost metric 2: entropy of workspace activations
        ws_abs = ws.abs() + 1e-10
        ws_dist = ws_abs / ws_abs.sum(dim=-1, keepdim=True)
        entropy = -(ws_dist * torch.log(ws_dist)).sum(dim=-1)
        all_ws_entropy.extend(entropy.cpu().numpy())

        # Cost metric 3: effective dimensionality (how many dims are active)
        # Count dims with activation > 10% of max
        max_act = ws.abs().max(dim=-1, keepdim=True).values
        dims_used = (ws.abs() > 0.1 * max_act).float().sum(dim=-1)
        all_ws_dims_used.extend(dims_used.cpu().numpy())

    return {
        'accuracy': correct / total,
        'mean_ws_norm': float(np.mean(all_ws_norms)),
        'mean_ws_entropy': float(np.mean(all_ws_entropy)),
        'mean_dims_used': float(np.mean(all_ws_dims_used)),
        'std_ws_norm': float(np.std(all_ws_norms)),
    }


def run_sweep(label, model_fn, device, input_dim_base=32, feature_counts=[1, 2, 3, 4, 5],
              epochs=15, batch_size=128):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    sweep_results = {}
    for n_feat in feature_counts:
        input_dim = n_feat * input_dim_base
        dataset = MultiFeatureDataset(n_features=n_feat, input_dim=input_dim_base)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        test_data = MultiFeatureDataset(n_features=n_feat, input_dim=input_dim_base, n_samples=5000, seed=99)
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

        model = model_fn(input_dim).to(device)
        train_model(model, loader, device, epochs=epochs)
        metrics = evaluate_cost(model, test_loader, device)
        sweep_results[n_feat] = metrics

        print(f"  N={n_feat}: acc={metrics['accuracy']:.4f}  norm={metrics['mean_ws_norm']:.4f}  "
              f"entropy={metrics['mean_ws_entropy']:.4f}  dims={metrics['mean_dims_used']:.1f}")

    # Compute scaling
    accs = [sweep_results[n]['accuracy'] for n in feature_counts]
    norms = [sweep_results[n]['mean_ws_norm'] for n in feature_counts]
    entropies = [sweep_results[n]['mean_ws_entropy'] for n in feature_counts]
    dims = [sweep_results[n]['mean_dims_used'] for n in feature_counts]

    # Monotonicity of cost increase
    norm_monotonic = all(norms[i] <= norms[i + 1] for i in range(len(norms) - 1))
    acc_decreasing = accs[0] > accs[-1]
    cost_increase = norms[-1] - norms[0]

    return {
        'label': label,
        'sweep': {str(k): v for k, v in sweep_results.items()},
        'accuracies': accs,
        'norms': norms,
        'entropies': entropies,
        'dims_used': dims,
        'norm_monotonic': norm_monotonic,
        'acc_decreasing': acc_decreasing,
        'cost_increase': cost_increase,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2034] Device: {device}")

    print(f"\n{'='*70}")
    print(f"  z2034: Workspace Cost Scaling")
    print(f"  Workspace cost should increase with integration demands")
    print(f"  NO consciousness losses — cost scaling must EMERGE")
    print(f"{'='*70}")

    feature_counts = [1, 2, 3, 4, 5]

    results = {}
    results['A'] = run_sweep(
        'A: Narrow workspace (16)',
        lambda d: WorkspaceIntegrator(d, ws_dim=16),
        device, feature_counts=feature_counts, epochs=args.epochs, batch_size=args.batch_size
    )
    results['B'] = run_sweep(
        'B: Wide workspace (64)',
        lambda d: WorkspaceIntegrator(d, ws_dim=64),
        device, feature_counts=feature_counts, epochs=args.epochs, batch_size=args.batch_size
    )
    results['C'] = run_sweep(
        'C: No workspace (direct)',
        lambda d: DirectClassifier(d, hidden_dim=64),
        device, feature_counts=feature_counts, epochs=args.epochs, batch_size=args.batch_size
    )

    # Analysis
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Workspace Cost Scaling")
    print(f"{'='*70}")

    for k in ['A', 'B', 'C']:
        r = results[k]
        print(f"\n  {r['label']}")
        print(f"    Norm monotonic:  {r['norm_monotonic']}")
        print(f"    Acc decreasing:  {r['acc_decreasing']}")
        print(f"    Cost increase:   {r['cost_increase']:+.4f}")

    t1 = results['A']['norm_monotonic']  # Workspace cost monotonically increases
    t2 = results['A']['acc_decreasing']  # Workspace accuracy decreases with features
    t3 = results['A']['cost_increase'] > results['C']['cost_increase']  # Workspace cost grows more
    t4 = not results['C']['norm_monotonic'] or results['C']['cost_increase'] < results['A']['cost_increase'] * 0.5

    print(f"\n  T1: Workspace cost monotonic:           {'PASS' if t1 else 'FAIL'}")
    print(f"  T2: Workspace acc decreases:            {'PASS' if t2 else 'FAIL'}")
    print(f"  T3: Workspace cost grows more than FF:  {'PASS' if t3 else 'FAIL'} ({results['A']['cost_increase']:.4f} vs {results['C']['cost_increase']:.4f})")
    print(f"  T4: FF cost flat or grows less:         {'PASS' if t4 else 'FAIL'}")

    n_pass = sum([t1, t2, t3, t4])
    verdict = {0: "NO_SCALING", 1: "WEAK", 2: "PARTIAL", 3: "MOSTLY", 4: "GENUINE_COST_SCALING"}[n_pass]
    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    output = {
        'experiment': 'z2034_workspace_cost_scaling',
        'hypothesis': 'Workspace cost scales with integration demands',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': ['Baars 2005 (Workspace capacity)', 'Cowan 2001 (Magical number 4)', 'Dehaene et al. 2003'],
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

    rp = Path(__file__).parent.parent / 'results' / 'z2034_workspace_cost_scaling.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
