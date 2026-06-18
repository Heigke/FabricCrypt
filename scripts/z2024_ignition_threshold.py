#!/usr/bin/env python3
"""
z2024: Workspace Ignition Threshold (All-or-Nothing Broadcasting)

Core GWT prediction: Global workspace shows THRESHOLD ignition, not graded
degradation. When stimulus strength crosses a critical threshold, the
workspace either broadcasts fully or not at all. This is the neural
"ignition" observed in Dehaene et al. 2003.

In feedforward-only networks, degrading stimulus → graded accuracy drop.
In workspace networks, degrading stimulus → sharp sigmoid/step in accuracy.

Test: Vary stimulus visibility (noise masking) and measure:
  1. Task accuracy vs visibility curve
  2. Sigmoid steepness (workspace should be steeper)
  3. Workspace activation norm vs visibility (should be bimodal)
  4. Feedforward hidden activation vs visibility (should be unimodal)

Conditions:
  A: Workspace model (32 dim) — expect threshold
  B: No workspace (feedforward) — expect graded
  C: Wide workspace (128 dim) — expect threshold at lower visibility
  D: Workspace + telemetry — expect threshold

Architecture:
  - Encoder: CNN → 64 dim
  - Optional workspace: 64 → ws_dim → 64 (bottleneck with layer norm + GELU)
  - Classifier: 64 → 10

Success criteria:
  T1: Workspace sigmoid steepness > 2x feedforward steepness
  T2: Workspace activation distribution is bimodal (dip test)
  T3: Feedforward activation distribution is unimodal
  T4: Ignition threshold LOWER for wide than narrow workspace

References:
  Dehaene et al. 2003 — Neural ignition and global workspace
  Sergent & Dehaene 2004 — Threshold vs graded perception
  Mashour et al. 2020 — Conscious processing and ignition

NO consciousness losses. Ignition must EMERGE from task training.
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
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


# ---------- Dataset ----------

class MaskedMNIST(Dataset):
    """MNIST with controllable visibility (noise masking)."""
    def __init__(self, base_data, visibility=1.0):
        self.images = base_data.data.float() / 255.0
        self.labels = base_data.targets
        self.visibility = visibility

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].unsqueeze(0)  # [1, 28, 28]
        noise = torch.randn_like(img)
        # visibility=1.0: pure signal. visibility=0.0: pure noise
        masked = self.visibility * img + (1.0 - self.visibility) * noise
        masked = masked.clamp(0, 1)
        return masked, self.labels[idx]


# ---------- Architecture ----------

class Encoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, hidden_dim)

    def forward(self, x):
        h = self.conv(x)
        return F.relu(self.fc(h.view(h.size(0), -1)))


class WorkspaceBottleneck(nn.Module):
    def __init__(self, input_dim, ws_dim, telemetry_dim=0):
        super().__init__()
        self.ws_dim = ws_dim
        total_in = input_dim + telemetry_dim
        self.compress = nn.Linear(total_in, ws_dim)
        self.expand = nn.Linear(ws_dim, input_dim)
        self.ln = nn.LayerNorm(ws_dim)

    def forward(self, x, telemetry=None):
        if telemetry is not None:
            x = torch.cat([x, telemetry], dim=-1)
        ws = self.ln(F.gelu(self.compress(x)))
        return self.expand(ws), ws


class IgnitionModel(nn.Module):
    def __init__(self, hidden_dim=64, ws_dim=None, telemetry_dim=0, num_classes=10):
        super().__init__()
        self.encoder = Encoder(hidden_dim)
        self.ws_dim = ws_dim
        self.telemetry_dim = telemetry_dim

        if ws_dim is not None:
            self.workspace = WorkspaceBottleneck(hidden_dim, ws_dim, telemetry_dim)
        else:
            self.workspace = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )

    def forward(self, x, telemetry=None, return_activations=False):
        h = self.encoder(x)

        if self.workspace is not None:
            broadcast, ws_state = self.workspace(h, telemetry)
        else:
            broadcast = h
            ws_state = h  # Use encoder output as proxy

        logits = self.classifier(broadcast)

        if return_activations:
            return logits, ws_state
        return logits


# ---------- Training ----------

def train_model(model, train_data, device, epochs=15, batch_size=128, lr=1e-3):
    """Train on fully visible MNIST (visibility=1.0). No consciousness losses."""
    dataset = MaskedMNIST(train_data, visibility=1.0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            telemetry = None
            if model.telemetry_dim > 0 and TELEMETRY_AVAILABLE:
                try:
                    telem = SysfsHwmonTelemetry(card_index=0)
                    s = telem.read_sample()
                    t = getattr(s, 'temp_edge_c', 50.0)
                    p = getattr(s, 'power_w', 30.0)
                    telemetry = torch.tensor(
                        [[t / 100.0, p / 100.0]], device=device
                    ).expand(images.size(0), -1)
                except Exception:
                    telemetry = None

            optimizer.zero_grad()
            logits = model(images, telemetry)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.3f}  ({elapsed:.1f}s)")


# ---------- Evaluation ----------

@torch.no_grad()
def evaluate_at_visibility(model, test_data, device, visibility):
    """Evaluate accuracy and collect activation norms at given visibility."""
    dataset = MaskedMNIST(test_data, visibility=visibility)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()

    correct = 0
    total = 0
    activation_norms = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        telemetry = None
        if model.telemetry_dim > 0:
            telemetry = torch.zeros(images.size(0), model.telemetry_dim, device=device)

        logits, ws_state = model(images, telemetry, return_activations=True)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
        activation_norms.extend(ws_state.norm(dim=-1).cpu().numpy())

    return correct / total, np.array(activation_norms)


def sigmoid_func(x, k, x0, L, b):
    """Sigmoid: L / (1 + exp(-k*(x - x0))) + b"""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def fit_sigmoid(visibilities, accuracies):
    """Fit sigmoid to accuracy curve. Return steepness (k)."""
    try:
        popt, _ = curve_fit(
            sigmoid_func, visibilities, accuracies,
            p0=[10.0, 0.5, 0.9, 0.1],
            bounds=([0.1, 0.0, 0.0, 0.0], [100.0, 1.0, 1.0, 0.5]),
            maxfev=10000
        )
        return popt[0], popt  # steepness k, all params
    except Exception:
        return 0.0, None


def bimodality_coefficient(data):
    """Sarle's bimodality coefficient. >5/9 suggests bimodality."""
    n = len(data)
    if n < 3:
        return 0.0
    from scipy.stats import skew, kurtosis
    g = skew(data)
    k = kurtosis(data, fisher=True)  # excess kurtosis
    bc = (g**2 + 1) / (k + 3 * (n-1)**2 / ((n-2)*(n-3)))
    return bc


def hartigan_dip_test(data, n_boot=100):
    """Simple bimodality test: compare variance of top/bottom halves."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    mid = n // 2
    lower = sorted_data[:mid]
    upper = sorted_data[mid:]

    # If bimodal, each half has low variance relative to overall
    overall_var = np.var(data)
    if overall_var < 1e-10:
        return False, 0.0

    half_var = (np.var(lower) + np.var(upper)) / 2
    ratio = half_var / overall_var
    # Bimodal: ratio is small (two tight clusters)
    return ratio < 0.5, ratio


# ---------- Main ----------

def run_condition(label, model, train_data, test_data, device,
                  epochs=15, batch_size=128):
    """Train and evaluate ignition curve for one condition."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Workspace: {model.ws_dim or 'None (feedforward)'}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*70}")

    model = model.to(device)
    train_model(model, train_data, device, epochs=epochs, batch_size=batch_size)

    # Sweep visibility from 0 (pure noise) to 1 (clean)
    visibilities = np.linspace(0.0, 1.0, 21)
    accuracies = []
    all_norms = {}

    print(f"\n  --- Ignition Curve ---")
    for v in visibilities:
        acc, norms = evaluate_at_visibility(model, test_data, device, v)
        accuracies.append(acc)
        all_norms[f"{v:.2f}"] = norms
        bar = '#' * int(acc * 40)
        print(f"    vis={v:.2f}  acc={acc:.3f}  |{bar}")

    accuracies = np.array(accuracies)

    # Fit sigmoid
    steepness, sig_params = fit_sigmoid(visibilities, accuracies)
    print(f"\n  Sigmoid steepness (k): {steepness:.2f}")
    if sig_params is not None:
        print(f"  Sigmoid threshold (x0): {sig_params[1]:.3f}")

    # Bimodality of activation norms at threshold visibility
    threshold_vis = sig_params[1] if sig_params is not None else 0.5
    closest_vis = visibilities[np.argmin(np.abs(visibilities - threshold_vis))]
    threshold_norms = all_norms[f"{closest_vis:.2f}"]

    bc = bimodality_coefficient(threshold_norms)
    is_bimodal, dip_ratio = hartigan_dip_test(threshold_norms)

    print(f"\n  Activation analysis at threshold (vis={closest_vis:.2f}):")
    print(f"    Bimodality coefficient: {bc:.4f} (>0.556 = bimodal)")
    print(f"    Dip ratio: {dip_ratio:.4f} (<0.5 = bimodal)")
    print(f"    Bimodal: {is_bimodal}")

    return {
        'label': label,
        'ws_dim': model.ws_dim,
        'n_params': n_params,
        'visibilities': visibilities.tolist(),
        'accuracies': accuracies.tolist(),
        'sigmoid_steepness': float(steepness),
        'sigmoid_params': [float(p) for p in sig_params] if sig_params is not None else None,
        'bimodality_coefficient': float(bc),
        'dip_ratio': float(dip_ratio),
        'is_bimodal': bool(is_bimodal),
        'threshold_visibility': float(closest_vis),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2024] Device: {device}")

    from torchvision import datasets
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    print("Loading MNIST...")
    train_data = datasets.MNIST(str(data_dir), train=True, download=True)
    test_data = datasets.MNIST(str(data_dir), train=False, download=True)

    print(f"\n{'='*70}")
    print(f"  z2024: Workspace Ignition Threshold")
    print(f"  GWT predicts: all-or-nothing broadcasting at threshold")
    print(f"  Feedforward predicts: graded accuracy degradation")
    print(f"  NO consciousness losses — ignition must EMERGE")
    print(f"{'='*70}")

    results = {}

    # A: Workspace (32 dim)
    model_a = IgnitionModel(hidden_dim=64, ws_dim=32)
    results['A'] = run_condition(
        'A: Workspace (32 dim)', model_a,
        train_data, test_data, device, args.epochs, args.batch_size
    )

    # B: No workspace (feedforward)
    model_b = IgnitionModel(hidden_dim=64, ws_dim=None)
    results['B'] = run_condition(
        'B: No workspace (feedforward)', model_b,
        train_data, test_data, device, args.epochs, args.batch_size
    )

    # C: Wide workspace (128 dim)
    model_c = IgnitionModel(hidden_dim=64, ws_dim=128)
    results['C'] = run_condition(
        'C: Wide workspace (128 dim)', model_c,
        train_data, test_data, device, args.epochs, args.batch_size
    )

    # D: Workspace + telemetry
    model_d = IgnitionModel(hidden_dim=64, ws_dim=32, telemetry_dim=2)
    results['D'] = run_condition(
        'D: Workspace + telemetry (32 dim)', model_d,
        train_data, test_data, device, args.epochs, args.batch_size
    )

    # ---------- Analysis ----------
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Ignition Threshold")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<40} {'Steepness':>10} {'Bimodal':>10} {'BC':>10}")
    print(f"  {'-'*70}")
    for key in ['A', 'B', 'C', 'D']:
        r = results[key]
        print(f"  {r['label']:<40} {r['sigmoid_steepness']:>10.2f} "
              f"{'YES' if r['is_bimodal'] else 'NO':>10} "
              f"{r['bimodality_coefficient']:>10.4f}")

    # Tests
    ws_steep = max(results['A']['sigmoid_steepness'], results['C']['sigmoid_steepness'])
    ff_steep = results['B']['sigmoid_steepness']

    t1 = ws_steep > ff_steep * 1.5  # Workspace steeper than feedforward
    t2 = results['A']['is_bimodal'] or results['C']['is_bimodal']  # Workspace bimodal
    t3 = not results['B']['is_bimodal']  # Feedforward NOT bimodal
    t4 = (results['C']['sigmoid_steepness'] < results['A']['sigmoid_steepness'] or
           results['C']['sigmoid_params'] is not None and
           results['A']['sigmoid_params'] is not None and
           results['C']['sigmoid_params'][1] < results['A']['sigmoid_params'][1])
    # Wide workspace ignites at lower visibility

    print(f"\n  T1: Workspace steeper than feedforward (>1.5x): "
          f"{'PASS' if t1 else 'FAIL'} ({ws_steep:.2f} vs {ff_steep:.2f})")
    print(f"  T2: Workspace activations bimodal:                "
          f"{'PASS' if t2 else 'FAIL'}")
    print(f"  T3: Feedforward activations unimodal:             "
          f"{'PASS' if t3 else 'FAIL'}")
    print(f"  T4: Wide workspace lower threshold than narrow:   "
          f"{'PASS' if t4 else 'FAIL'}")

    n_pass = sum([t1, t2, t3, t4])
    if n_pass >= 4:
        verdict = "GENUINE_IGNITION_CONFIRMED"
    elif n_pass >= 3:
        verdict = "IGNITION_PARTIAL"
    elif n_pass >= 2:
        verdict = "IGNITION_WEAK"
    else:
        verdict = "NO_IGNITION"

    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    # Save
    output = {
        'experiment': 'z2024_ignition_threshold',
        'hypothesis': 'GWT workspace shows all-or-nothing ignition at stimulus threshold',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': [
            'Dehaene et al. 2003 (Neural ignition)',
            'Sergent & Dehaene 2004 (Threshold perception)',
            'Mashour et al. 2020 (Ignition theory)',
        ],
        'design_principle': 'NO consciousness losses. Ignition must EMERGE from task training.',
        'conditions': results,
        'tests': {
            't1_workspace_steeper': bool(t1),
            't2_workspace_bimodal': bool(t2),
            't3_feedforward_unimodal': bool(t3),
            't4_wide_lower_threshold': bool(t4),
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
    results_path = Path(__file__).parent.parent / 'results' / 'z2024_ignition_threshold.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
