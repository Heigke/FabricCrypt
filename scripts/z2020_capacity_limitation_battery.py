#!/usr/bin/env python3
"""
z2020: Capacity-Limitation Battery (Unforgeable Consciousness Test)

Inspired by Phua 2025 (arXiv:2512.19155) Experiment 2 and the GWT capacity
limitation principle: a genuine global workspace produces CAPACITY-DEPENDENT
INTERFERENCE. Consciousness COSTS something.

Key insight: Tests where consciousness HURTS performance are hard to fake
because there is no incentive for a system to deliberately fail, and the
failure pattern must be specific and calibrated.

Architecture:
  - Specialist A: Digit encoder (MNIST left half)
  - Specialist B: Fashion encoder (FashionMNIST right half)
  - Global workspace: variable-width bottleneck
  - Decoder A, B: Classification heads (10 classes each)

Conditions:
  A: Wide workspace (128 dim)  — minimal interference expected
  B: Narrow workspace (16 dim) — STRONG interference expected
  C: No workspace (parallel)   — NO interference expected
  D: Embodied workspace (16 dim + telemetry) — interference + HW adaptation?

Tests:
  1. Single-task accuracy (each task presented alone)
  2. Dual-task accuracy (both tasks presented simultaneously)
  3. Interference = single_acc - dual_acc (positive = interference)
  4. Interference vs workspace width curve

Success criteria (from GWT theory):
  - Workspace models (A, B, D) MUST show dual-task interference
  - No-workspace model (C) must NOT show interference
  - Interference MUST scale inversely with workspace capacity (B > A)
  - Failure to show interference = workspace is NOT genuine

References:
  Phua 2025 — arXiv:2512.19155 (Ablation-Based Markers)
  Baars 1988 — A Cognitive Theory of Consciousness (GWT)
  Dehaene & Naccache 2001 — Towards a cognitive neuroscience of consciousness

NO auxiliary consciousness losses. Train ONLY on task loss.
Consciousness-relevant metrics (interference pattern) must emerge.
"""

import sys
import os
import json
import time
import math
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Telemetry
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


# ---------- Dataset: Dual-Task (MNIST + FashionMNIST) ----------

class DualTaskDataset(Dataset):
    """Presents two tasks simultaneously: digit + fashion classification.

    Images are 28x28 each. In dual mode, both are presented.
    In single-digit mode, fashion is zeroed. In single-fashion, digit is zeroed.
    """
    def __init__(self, mnist_data, fashion_data, mode='dual'):
        """mode: 'dual', 'digit_only', 'fashion_only'"""
        self.mode = mode
        # Ensure same length
        n = min(len(mnist_data), len(fashion_data))
        self.mnist_imgs = mnist_data.data[:n].float() / 255.0
        self.mnist_labels = mnist_data.targets[:n]
        self.fashion_imgs = fashion_data.data[:n].float() / 255.0
        self.fashion_labels = fashion_data.targets[:n]

    def __len__(self):
        return len(self.mnist_labels)

    def __getitem__(self, idx):
        digit_img = self.mnist_imgs[idx].unsqueeze(0)  # [1, 28, 28]
        fashion_img = self.fashion_imgs[idx].unsqueeze(0)
        digit_label = self.mnist_labels[idx]
        fashion_label = self.fashion_labels[idx]

        if self.mode == 'digit_only':
            fashion_img = torch.zeros_like(fashion_img)
            fashion_label = torch.tensor(-1)  # ignore
        elif self.mode == 'fashion_only':
            digit_img = torch.zeros_like(digit_img)
            digit_label = torch.tensor(-1)

        return digit_img, fashion_img, digit_label, fashion_label


# ---------- Architecture ----------

class SpecialistEncoder(nn.Module):
    """Simple CNN specialist for one task."""
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, out_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class GlobalWorkspace(nn.Module):
    """GWT-inspired bottleneck. Forces information compression."""
    def __init__(self, input_dim, workspace_dim, telemetry_dim=0):
        super().__init__()
        self.workspace_dim = workspace_dim
        total_in = input_dim + telemetry_dim
        self.compress = nn.Linear(total_in, workspace_dim)
        self.expand = nn.Linear(workspace_dim, input_dim)
        self.ln = nn.LayerNorm(workspace_dim)

    def forward(self, specialist_outputs, telemetry=None):
        """specialist_outputs: [batch, total_specialist_dim]"""
        if telemetry is not None:
            x = torch.cat([specialist_outputs, telemetry], dim=-1)
        else:
            x = specialist_outputs
        workspace = self.ln(F.gelu(self.compress(x)))
        broadcast = self.expand(workspace)
        return broadcast, workspace


class DualTaskModel(nn.Module):
    """Dual-task model with optional global workspace bottleneck."""

    def __init__(self, specialist_dim=64, workspace_dim=None, telemetry_dim=0):
        """
        workspace_dim=None: no workspace (parallel processing)
        workspace_dim=N: workspace bottleneck of width N
        """
        super().__init__()
        self.specialist_dim = specialist_dim
        self.workspace_dim = workspace_dim
        self.telemetry_dim = telemetry_dim

        # Specialist encoders
        self.digit_encoder = SpecialistEncoder(specialist_dim)
        self.fashion_encoder = SpecialistEncoder(specialist_dim)

        # Optional workspace
        if workspace_dim is not None:
            self.workspace = GlobalWorkspace(
                specialist_dim * 2, workspace_dim, telemetry_dim
            )
        else:
            self.workspace = None

        # Decoder heads
        self.digit_head = nn.Linear(specialist_dim, 10)
        self.fashion_head = nn.Linear(specialist_dim, 10)

    def forward(self, digit_img, fashion_img, telemetry=None):
        h_digit = self.digit_encoder(digit_img)
        h_fashion = self.fashion_encoder(fashion_img)

        if self.workspace is not None:
            combined = torch.cat([h_digit, h_fashion], dim=-1)
            broadcast, ws_state = self.workspace(combined, telemetry)
            # Split broadcast back to specialist dims
            h_digit_out = broadcast[:, :self.specialist_dim]
            h_fashion_out = broadcast[:, self.specialist_dim:]
        else:
            h_digit_out = h_digit
            h_fashion_out = h_fashion
            ws_state = None

        digit_logits = self.digit_head(h_digit_out)
        fashion_logits = self.fashion_head(h_fashion_out)

        return {
            'digit_logits': digit_logits,
            'fashion_logits': fashion_logits,
            'workspace_state': ws_state,
        }


# ---------- Training ----------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    digit_correct = 0
    fashion_correct = 0
    total = 0

    for digit_img, fashion_img, digit_label, fashion_label in loader:
        digit_img = digit_img.to(device)
        fashion_img = fashion_img.to(device)
        digit_label = digit_label.to(device)
        fashion_label = fashion_label.to(device)

        # Get telemetry if model supports it
        telemetry = None
        if model.telemetry_dim > 0 and TELEMETRY_AVAILABLE:
            try:
                telem = SysfsHwmonTelemetry(card_index=0)
                sample = telem.read_sample()
                t = getattr(sample, 'temp_edge_c', 50.0)
                p = getattr(sample, 'power_w', 30.0)
                telemetry = torch.tensor(
                    [[t / 100.0, p / 100.0]], device=device
                ).expand(digit_img.size(0), -1)
            except Exception:
                telemetry = torch.randn(digit_img.size(0), model.telemetry_dim, device=device) * 0.1

        optimizer.zero_grad()
        out = model(digit_img, fashion_img, telemetry)

        # Task losses only — NO consciousness losses
        loss_digit = F.cross_entropy(out['digit_logits'], digit_label)
        loss_fashion = F.cross_entropy(out['fashion_logits'], fashion_label)
        loss = loss_digit + loss_fashion

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * digit_img.size(0)
        digit_correct += (out['digit_logits'].argmax(1) == digit_label).sum().item()
        fashion_correct += (out['fashion_logits'].argmax(1) == fashion_label).sum().item()
        total += digit_img.size(0)

    return {
        'loss': total_loss / total,
        'digit_acc': digit_correct / total,
        'fashion_acc': fashion_correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, device, mode='dual'):
    """Evaluate model on a specific task mode."""
    model.eval()
    digit_correct = 0
    fashion_correct = 0
    digit_total = 0
    fashion_total = 0

    for digit_img, fashion_img, digit_label, fashion_label in loader:
        digit_img = digit_img.to(device)
        fashion_img = fashion_img.to(device)
        digit_label = digit_label.to(device)
        fashion_label = fashion_label.to(device)

        telemetry = None
        if model.telemetry_dim > 0:
            telemetry = torch.randn(digit_img.size(0), model.telemetry_dim, device=device) * 0.1

        out = model(digit_img, fashion_img, telemetry)

        if digit_label.min() >= 0:
            digit_correct += (out['digit_logits'].argmax(1) == digit_label).sum().item()
            digit_total += digit_img.size(0)
        if fashion_label.min() >= 0:
            fashion_correct += (out['fashion_logits'].argmax(1) == fashion_label).sum().item()
            fashion_total += fashion_img.size(0)

    digit_acc = digit_correct / digit_total if digit_total > 0 else 0
    fashion_acc = fashion_correct / fashion_total if fashion_total > 0 else 0
    return digit_acc, fashion_acc


@torch.no_grad()
def capacity_titration(model, mnist_test, fashion_test, device, n_widths=6):
    """Sweep workspace capacity and measure interference at each width.

    Only works for workspace models. Tests the capacity-dependent curve.
    """
    if model.workspace is None:
        return None

    original_compress = model.workspace.compress.weight.data.clone()
    original_expand = model.workspace.expand.weight.data.clone()
    original_bias_c = model.workspace.compress.bias.data.clone()
    original_bias_e = model.workspace.expand.bias.data.clone()

    ws_dim = model.workspace.workspace_dim
    results = []

    # Test different effective capacities by zeroing out workspace dims
    for frac in np.linspace(0.1, 1.0, n_widths):
        active_dims = max(1, int(ws_dim * frac))

        # Zero out inactive dimensions
        model.workspace.compress.weight.data = original_compress.clone()
        model.workspace.compress.bias.data = original_bias_c.clone()
        model.workspace.expand.weight.data = original_expand.clone()
        model.workspace.expand.bias.data = original_bias_e.clone()

        model.workspace.compress.weight.data[active_dims:] = 0
        model.workspace.compress.bias.data[active_dims:] = 0
        model.workspace.expand.weight.data[:, active_dims:] = 0

        # Evaluate dual-task
        dual_ds = DualTaskDataset(mnist_test, fashion_test, mode='dual')
        dual_loader = DataLoader(dual_ds, batch_size=256, shuffle=False)
        d_acc, f_acc = evaluate(model, dual_loader, device, 'dual')

        results.append({
            'fraction': float(frac),
            'active_dims': active_dims,
            'digit_acc': d_acc,
            'fashion_acc': f_acc,
            'mean_acc': (d_acc + f_acc) / 2,
        })

    # Restore original weights
    model.workspace.compress.weight.data = original_compress
    model.workspace.compress.bias.data = original_bias_c
    model.workspace.expand.weight.data = original_expand
    model.workspace.expand.bias.data = original_bias_e

    return results


# ---------- Main ----------

def run_condition(label, model, mnist_train, fashion_train, mnist_test, fashion_test,
                  device, epochs=10, batch_size=128):
    """Train and evaluate one condition."""
    print(f"\n{'='*70}")
    print(f"  Condition: {label}")
    print(f"  Workspace: {model.workspace_dim or 'None (parallel)'}")
    print(f"  Telemetry: {model.telemetry_dim > 0}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*70}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train on dual-task
    train_ds = DualTaskDataset(mnist_train, fashion_train, mode='dual')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)

    epoch_stats = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        stats = train_epoch(model, train_loader, optimizer, device)
        elapsed = time.time() - t0
        print(f"  Epoch {ep:2d}/{epochs}  loss={stats['loss']:.4f}  "
              f"digit={stats['digit_acc']:.3f}  fashion={stats['fashion_acc']:.3f}  "
              f"({elapsed:.1f}s)")
        epoch_stats.append({**stats, 'epoch': ep, 'elapsed_s': round(elapsed, 1)})

    # --- Evaluate: Single-task vs Dual-task ---

    # Dual-task test
    dual_test_ds = DualTaskDataset(mnist_test, fashion_test, mode='dual')
    dual_loader = DataLoader(dual_test_ds, batch_size=256, shuffle=False)
    dual_digit, dual_fashion = evaluate(model, dual_loader, device, 'dual')

    # Single-task: digit only
    digit_test_ds = DualTaskDataset(mnist_test, fashion_test, mode='digit_only')
    digit_loader = DataLoader(digit_test_ds, batch_size=256, shuffle=False)
    single_digit, _ = evaluate(model, digit_loader, device, 'digit_only')

    # Single-task: fashion only
    fashion_test_ds = DualTaskDataset(mnist_test, fashion_test, mode='fashion_only')
    fashion_loader = DataLoader(fashion_test_ds, batch_size=256, shuffle=False)
    _, single_fashion = evaluate(model, fashion_loader, device, 'fashion_only')

    # Interference = single - dual (positive means dual-task hurts)
    digit_interference = single_digit - dual_digit
    fashion_interference = single_fashion - dual_fashion
    mean_interference = (digit_interference + fashion_interference) / 2

    print(f"\n  --- Interference Test ---")
    print(f"  Single-task digit:   {single_digit:.4f}")
    print(f"  Dual-task digit:     {dual_digit:.4f}")
    print(f"  Digit interference:  {digit_interference:+.4f}")
    print(f"  Single-task fashion: {single_fashion:.4f}")
    print(f"  Dual-task fashion:   {dual_fashion:.4f}")
    print(f"  Fashion interference:{fashion_interference:+.4f}")
    print(f"  MEAN INTERFERENCE:   {mean_interference:+.4f}")

    # Capacity titration (if workspace model)
    titration = capacity_titration(model, mnist_test, fashion_test, device)
    if titration is not None:
        print(f"\n  --- Capacity Titration ---")
        for t in titration:
            print(f"    {t['fraction']:.1f} capacity ({t['active_dims']} dims): "
                  f"acc={t['mean_acc']:.4f}")

    return {
        'label': label,
        'workspace_dim': model.workspace_dim,
        'telemetry_dim': model.telemetry_dim,
        'n_params': n_params,
        'epoch_stats': epoch_stats,
        'single_digit_acc': single_digit,
        'single_fashion_acc': single_fashion,
        'dual_digit_acc': dual_digit,
        'dual_fashion_acc': dual_fashion,
        'digit_interference': digit_interference,
        'fashion_interference': fashion_interference,
        'mean_interference': mean_interference,
        'capacity_titration': titration,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2020] Device: {device}")

    if TELEMETRY_AVAILABLE:
        try:
            telem = SysfsHwmonTelemetry(card_index=0)
            s = telem.read_sample()
            print(f"[telemetry] Live: temp={getattr(s, 'temp_edge_c', '?')}C")
        except Exception:
            print("[telemetry] Not available, using random for condition D")

    # Load data
    from torchvision import datasets, transforms
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    print("Loading MNIST + FashionMNIST...")
    mnist_train = datasets.MNIST(str(data_dir), train=True, download=True)
    mnist_test = datasets.MNIST(str(data_dir), train=False, download=True)
    fashion_train = datasets.FashionMNIST(str(data_dir), train=True, download=True)
    fashion_test = datasets.FashionMNIST(str(data_dir), train=False, download=True)

    print(f"\n{'='*70}")
    print(f"  z2020: Capacity-Limitation Battery")
    print(f"  Dual-Task Interference Test (Phua 2025 inspired)")
    print(f"{'='*70}")
    print(f"  GWT prediction: Workspace models show interference.")
    print(f"  No-workspace models do NOT show interference.")
    print(f"  Interference scales inversely with workspace capacity.")
    print(f"  NO consciousness losses — task only.")
    print()

    specialist_dim = 64
    results = {}

    # Condition A: Wide workspace (128 dim) — minimal interference
    model_a = DualTaskModel(specialist_dim=specialist_dim, workspace_dim=128)
    results['A'] = run_condition(
        'A: Wide workspace (128 dim)',
        model_a, mnist_train, fashion_train, mnist_test, fashion_test,
        device, epochs=args.epochs, batch_size=args.batch_size
    )

    # Condition B: Narrow workspace (16 dim) — STRONG interference
    model_b = DualTaskModel(specialist_dim=specialist_dim, workspace_dim=16)
    results['B'] = run_condition(
        'B: Narrow workspace (16 dim)',
        model_b, mnist_train, fashion_train, mnist_test, fashion_test,
        device, epochs=args.epochs, batch_size=args.batch_size
    )

    # Condition C: No workspace (parallel) — NO interference
    model_c = DualTaskModel(specialist_dim=specialist_dim, workspace_dim=None)
    results['C'] = run_condition(
        'C: No workspace (parallel)',
        model_c, mnist_train, fashion_train, mnist_test, fashion_test,
        device, epochs=args.epochs, batch_size=args.batch_size
    )

    # Condition D: Narrow workspace + telemetry (embodied)
    model_d = DualTaskModel(specialist_dim=specialist_dim, workspace_dim=16, telemetry_dim=2)
    results['D'] = run_condition(
        'D: Narrow workspace + telemetry (embodied)',
        model_d, mnist_train, fashion_train, mnist_test, fashion_test,
        device, epochs=args.epochs, batch_size=args.batch_size
    )

    # ---------- Analysis ----------

    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Capacity-Limitation Battery")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<45} {'Interference':>12}")
    print(f"  {'-'*57}")
    for key in ['A', 'B', 'C', 'D']:
        r = results[key]
        print(f"  {r['label']:<45} {r['mean_interference']:>+12.4f}")

    # Test 1: Workspace models show interference
    ws_interference = [results[k]['mean_interference'] for k in ['A', 'B', 'D']]
    no_ws_interference = results['C']['mean_interference']

    t1_pass = all(i > 0.005 for i in ws_interference)
    t2_pass = abs(no_ws_interference) < 0.01
    t3_pass = results['B']['mean_interference'] > results['A']['mean_interference']

    # Test 4: Capacity titration shows monotonic relationship
    t4_pass = False
    if results['B']['capacity_titration']:
        accs = [t['mean_acc'] for t in results['B']['capacity_titration']]
        # Should be monotonically increasing (more capacity = better)
        increases = sum(1 for i in range(1, len(accs)) if accs[i] >= accs[i-1] - 0.01)
        t4_pass = increases >= len(accs) - 2  # Allow 1 non-monotonicity

    print(f"\n  --- GWT Predictions ---")
    print(f"  T1: Workspace models show interference:      "
          f"{'PASS' if t1_pass else 'FAIL'} (A={results['A']['mean_interference']:+.4f}, "
          f"B={results['B']['mean_interference']:+.4f}, D={results['D']['mean_interference']:+.4f})")
    print(f"  T2: No-workspace has NO interference:        "
          f"{'PASS' if t2_pass else 'FAIL'} (C={no_ws_interference:+.4f})")
    print(f"  T3: Narrow > wide interference:              "
          f"{'PASS' if t3_pass else 'FAIL'} (B={results['B']['mean_interference']:+.4f} "
          f"vs A={results['A']['mean_interference']:+.4f})")
    print(f"  T4: Capacity titration monotonic:            "
          f"{'PASS' if t4_pass else 'FAIL'}")

    tests = [t1_pass, t2_pass, t3_pass, t4_pass]
    n_pass = sum(tests)

    if n_pass >= 4:
        verdict = "GENUINE_WORKSPACE_CONFIRMED"
    elif n_pass >= 3:
        verdict = "WORKSPACE_PARTIALLY_CONFIRMED"
    elif n_pass >= 2:
        verdict = "WORKSPACE_WEAK"
    else:
        verdict = "NO_GENUINE_WORKSPACE"

    print(f"\n  VERDICT: {verdict} ({n_pass}/4 tests pass)")
    print(f"{'='*70}")

    # Save results
    output = {
        'experiment': 'z2020_capacity_limitation_battery',
        'hypothesis': 'A genuine GWT workspace produces capacity-dependent dual-task interference',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'references': [
            'Phua 2025 arXiv:2512.19155 (Ablation-Based Markers)',
            'Baars 1988 (Global Workspace Theory)',
            'Dehaene & Naccache 2001 (GWT)',
        ],
        'design_principle': 'NO consciousness losses. Train only on task. Test for FAILURES.',
        'conditions': results,
        'tests': {
            't1_workspace_interference': t1_pass,
            't2_no_workspace_no_interference': t2_pass,
            't3_narrow_more_interference': t3_pass,
            't4_capacity_titration_monotonic': t4_pass,
        },
        'tests_passed': n_pass,
        'verdict': verdict,
    }

    # Convert numpy
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
    results_path = Path(__file__).parent.parent / 'results' / 'z2020_capacity_limitation_battery.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
