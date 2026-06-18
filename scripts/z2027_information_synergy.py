#!/usr/bin/env python3
"""
z2027: Information Synergy (IIT-Inspired, Not Phi)

IIT's Phi is computationally intractable and inverts for artificial systems
(z2023 confirmed PCI inversion). But the CORE IIT intuition — that consciousness
involves information that exists in the WHOLE but not in the PARTS — can be
tested using Partial Information Decomposition (PID).

PID decomposes mutual information I(sources; target) into:
  - Redundancy: info available from EITHER source alone
  - Unique: info available from ONLY one source
  - Synergy: info available ONLY from BOTH sources together

If workspace creates genuine integration, it should increase SYNERGY:
information that exists in the combined workspace state but NOT in any
individual specialist output.

Test: Train workspace model on task that requires combining information from
two specialists. Measure synergy between specialist outputs about the task label,
mediated through workspace.

Architecture:
  - Two specialists: digit classifier + parity detector
  - Workspace bottleneck: combines specialist outputs
  - Task: composite judgment requiring both (e.g., "digit > 5 AND even")

PID calculation:
  - Source 1: specialist A hidden state
  - Source 2: specialist B hidden state
  - Target: task label
  - Measure synergy using Williams & Beer 2010 PID

Conditions:
  A: Workspace (32 dim) — expect HIGH synergy
  B: No workspace (parallel) — expect LOW synergy
  C: Wide workspace (128 dim) — synergy present but lower?
  D: Random model — expect NO synergy

References:
  Tononi 2004 — An information integration theory of consciousness
  Williams & Beer 2010 — Nonnegative decomposition of multivariate information
  Luppi et al. 2024 (eLife) — Synergistic workspace in consciousness
  Mediano et al. 2021 — Towards an extended taxonomy of information dynamics

NO consciousness losses. Synergy must EMERGE from task demands.
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


# ---------- Dataset ----------

class CompositeTaskDataset(Dataset):
    """Task that requires integrating two properties of a digit.

    Property A: digit value (0-9) → is it > 4?
    Property B: digit value → is it even?
    Composite: (A AND B) = (digit > 4 AND even) → {6, 8} vs rest

    This requires COMBINING information that neither specialist has alone.
    """
    def __init__(self, mnist_data):
        self.images = mnist_data.data.float().unsqueeze(1) / 255.0  # [N, 1, 28, 28]
        self.labels = mnist_data.targets

        # Composite labels
        digits = self.labels.numpy()
        self.is_large = (digits > 4).astype(np.int64)      # Property A
        self.is_even = (digits % 2 == 0).astype(np.int64)  # Property B
        self.composite = ((digits > 4) & (digits % 2 == 0)).astype(np.int64)  # A AND B

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.images[idx],
                torch.tensor(self.is_large[idx]),
                torch.tensor(self.is_even[idx]),
                torch.tensor(self.composite[idx]))


# ---------- Architecture ----------

class PropertyEncoder(nn.Module):
    """Specialist encoder for one property."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, hidden_dim)

    def forward(self, x):
        h = self.conv(x)
        return F.relu(self.fc(h.view(h.size(0), -1)))


class SynergyModel(nn.Module):
    """Model with two specialists and optional workspace for integration."""

    def __init__(self, hidden_dim=32, ws_dim=None, telemetry_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ws_dim = ws_dim
        self.telemetry_dim = telemetry_dim

        # Two specialist encoders
        self.enc_a = PropertyEncoder(hidden_dim)  # Learns large/small
        self.enc_b = PropertyEncoder(hidden_dim)  # Learns even/odd

        # Individual property heads (for property-specific training)
        self.head_a = nn.Linear(hidden_dim, 2)  # large vs small
        self.head_b = nn.Linear(hidden_dim, 2)  # even vs odd

        # Integration path
        if ws_dim is not None:
            total_in = hidden_dim * 2 + telemetry_dim
            self.workspace = nn.Sequential(
                nn.Linear(total_in, ws_dim),
                nn.LayerNorm(ws_dim),
                nn.GELU(),
            )
            self.composite_head = nn.Linear(ws_dim, 2)
        else:
            # No workspace: direct concatenation
            self.workspace = None
            self.composite_head = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x, telemetry=None, return_hidden=False):
        h_a = self.enc_a(x)
        h_b = self.enc_b(x)

        logits_a = self.head_a(h_a)
        logits_b = self.head_b(h_b)

        combined = torch.cat([h_a, h_b], dim=-1)

        if self.workspace is not None:
            if telemetry is not None:
                combined_with_telem = torch.cat([combined, telemetry], dim=-1)
            else:
                combined_with_telem = combined
            ws_state = self.workspace(combined_with_telem)
            logits_composite = self.composite_head(ws_state)
        else:
            ws_state = combined
            logits_composite = self.composite_head(combined)

        result = {
            'logits_a': logits_a,
            'logits_b': logits_b,
            'logits_composite': logits_composite,
        }

        if return_hidden:
            result['h_a'] = h_a
            result['h_b'] = h_b
            result['ws_state'] = ws_state

        return result


# ---------- PID Estimation ----------

def estimate_synergy(h_a, h_b, labels, n_bins=8):
    """Estimate synergy using binned PID approximation.

    PID decomposition: I(A,B; Y) = Redundancy + Unique_A + Unique_B + Synergy

    We estimate:
      I(A; Y): MI between specialist A output and label
      I(B; Y): MI between specialist B output and label
      I(A,B; Y): MI between combined output and label

    Synergy ≈ I(A,B; Y) - I(A; Y) - I(B; Y) + Redundancy

    Using the minimum-MI definition of redundancy (Williams & Beer 2010):
      Redundancy = min(I(A; Y), I(B; Y))
      Synergy = I(A,B; Y) - max(I(A; Y), I(B; Y))

    (This is the simpler "redundancy lattice" approximation.)
    """
    # Project high-dim features to 1D for binning
    h_a_1d = np.dot(h_a, np.random.RandomState(0).randn(h_a.shape[1]))
    h_b_1d = np.dot(h_b, np.random.RandomState(1).randn(h_b.shape[1]))

    def binned_mi(x, y, n_bins):
        """Estimate MI(X; Y) using binned histogram."""
        x_bins = np.digitize(x, np.linspace(x.min(), x.max() + 1e-10, n_bins + 1)) - 1

        # P(x, y)
        n = len(x)
        joint = np.zeros((n_bins, len(np.unique(y))))
        y_unique = np.unique(y)
        y_map = {v: i for i, v in enumerate(y_unique)}

        for i in range(n):
            xi = min(x_bins[i], n_bins - 1)
            yi = y_map[y[i]]
            joint[xi, yi] += 1

        joint /= n
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)

        # Avoid log(0)
        mask = joint > 0
        mi = 0.0
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                if joint[i, j] > 0 and px[i, 0] > 0 and py[0, j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (px[i, 0] * py[0, j]))

        return max(0.0, mi)

    def binned_mi_2d(x1, x2, y, n_bins):
        """Estimate MI(X1, X2; Y) using joint 2D binning."""
        x1_bins = np.digitize(x1, np.linspace(x1.min(), x1.max() + 1e-10, n_bins + 1)) - 1
        x2_bins = np.digitize(x2, np.linspace(x2.min(), x2.max() + 1e-10, n_bins + 1)) - 1

        # Combined bins
        combined_bins = x1_bins * n_bins + x2_bins
        return binned_mi(combined_bins.astype(float), y, n_bins * n_bins)

    mi_a = binned_mi(h_a_1d, labels, n_bins)
    mi_b = binned_mi(h_b_1d, labels, n_bins)
    mi_ab = binned_mi_2d(h_a_1d, h_b_1d, labels, n_bins)

    # PID approximation (Williams & Beer style)
    redundancy = min(mi_a, mi_b)
    unique_a = mi_a - redundancy
    unique_b = mi_b - redundancy
    synergy = mi_ab - mi_a - mi_b + redundancy  # Can be negative (sampling noise)
    synergy = max(0.0, synergy)

    return {
        'mi_a': mi_a,
        'mi_b': mi_b,
        'mi_ab': mi_ab,
        'redundancy': redundancy,
        'unique_a': unique_a,
        'unique_b': unique_b,
        'synergy': synergy,
        'synergy_ratio': synergy / max(mi_ab, 1e-10),
    }


# ---------- Training & Evaluation ----------

def train_model(model, train_data, device, epochs=20, batch_size=128, lr=1e-3):
    dataset = CompositeTaskDataset(train_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        comp_correct = 0
        total = 0
        t0 = time.time()

        for images, lbl_a, lbl_b, lbl_composite in loader:
            images = images.to(device)
            lbl_a, lbl_b = lbl_a.to(device), lbl_b.to(device)
            lbl_composite = lbl_composite.to(device)

            telemetry = None
            if model.telemetry_dim > 0 and TELEMETRY_AVAILABLE:
                try:
                    telem = SysfsHwmonTelemetry(card_index=0)
                    s = telem.read_sample()
                    t_val = getattr(s, 'temp_edge_c', 50.0)
                    p_val = getattr(s, 'power_w', 30.0)
                    telemetry = torch.tensor(
                        [[t_val / 100.0, p_val / 100.0]], device=device
                    ).expand(images.size(0), -1)
                except Exception:
                    telemetry = None

            optimizer.zero_grad()
            out = model(images, telemetry)

            # Multi-task: train all three heads
            loss_a = F.cross_entropy(out['logits_a'], lbl_a)
            loss_b = F.cross_entropy(out['logits_b'], lbl_b)
            loss_composite = F.cross_entropy(out['logits_composite'], lbl_composite)
            loss = loss_a + loss_b + loss_composite

            loss.backward()
            optimizer.step()

            total_loss += loss_composite.item() * images.size(0)
            comp_correct += (out['logits_composite'].argmax(1) == lbl_composite).sum().item()
            total += images.size(0)

        elapsed = time.time() - t0
        if ep % 4 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  comp_loss={total_loss/total:.4f}  "
                  f"comp_acc={comp_correct/total:.3f}  ({elapsed:.1f}s)")


@torch.no_grad()
def evaluate_and_measure_synergy(model, test_data, device):
    dataset = CompositeTaskDataset(test_data)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()

    all_h_a = []
    all_h_b = []
    all_composite_labels = []
    acc_a_correct = 0
    acc_b_correct = 0
    acc_comp_correct = 0
    total = 0

    for images, lbl_a, lbl_b, lbl_composite in loader:
        images = images.to(device)
        lbl_a, lbl_b = lbl_a.to(device), lbl_b.to(device)
        lbl_composite = lbl_composite.to(device)

        telemetry = None
        if model.telemetry_dim > 0:
            telemetry = torch.zeros(images.size(0), model.telemetry_dim, device=device)

        out = model(images, telemetry, return_hidden=True)

        all_h_a.append(out['h_a'].cpu().numpy())
        all_h_b.append(out['h_b'].cpu().numpy())
        all_composite_labels.extend(lbl_composite.cpu().numpy())

        acc_a_correct += (out['logits_a'].argmax(1) == lbl_a).sum().item()
        acc_b_correct += (out['logits_b'].argmax(1) == lbl_b).sum().item()
        acc_comp_correct += (out['logits_composite'].argmax(1) == lbl_composite).sum().item()
        total += images.size(0)

    h_a = np.concatenate(all_h_a)
    h_b = np.concatenate(all_h_b)
    labels = np.array(all_composite_labels)

    pid = estimate_synergy(h_a, h_b, labels)

    return {
        'acc_a': acc_a_correct / total,
        'acc_b': acc_b_correct / total,
        'acc_composite': acc_comp_correct / total,
        **pid,
    }


# ---------- Main ----------

def run_condition(label, model, train_data, test_data, device,
                  epochs=20, batch_size=128, skip_train=False):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Workspace: {model.ws_dim or 'None (direct concat)'}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*70}")

    model = model.to(device)

    if not skip_train:
        train_model(model, train_data, device, epochs=epochs, batch_size=batch_size)
    else:
        print("  [Skipping training — random weights]")

    metrics = evaluate_and_measure_synergy(model, test_data, device)

    print(f"\n  --- Accuracies ---")
    print(f"  Property A (large?):    {metrics['acc_a']:.4f}")
    print(f"  Property B (even?):     {metrics['acc_b']:.4f}")
    print(f"  Composite (A AND B):    {metrics['acc_composite']:.4f}")

    print(f"\n  --- PID (Partial Information Decomposition) ---")
    print(f"  I(A; Y):       {metrics['mi_a']:.4f} bits")
    print(f"  I(B; Y):       {metrics['mi_b']:.4f} bits")
    print(f"  I(A,B; Y):     {metrics['mi_ab']:.4f} bits")
    print(f"  Redundancy:    {metrics['redundancy']:.4f} bits")
    print(f"  Unique A:      {metrics['unique_a']:.4f} bits")
    print(f"  Unique B:      {metrics['unique_b']:.4f} bits")
    print(f"  SYNERGY:       {metrics['synergy']:.4f} bits")
    print(f"  Synergy ratio: {metrics['synergy_ratio']:.4f}")

    return {
        'label': label,
        'ws_dim': model.ws_dim,
        'n_params': n_params,
        'skip_train': skip_train,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2027] Device: {device}")

    from torchvision import datasets
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    print("Loading MNIST...")
    train_data = datasets.MNIST(str(data_dir), train=True, download=True)
    test_data = datasets.MNIST(str(data_dir), train=False, download=True)

    print(f"\n{'='*70}")
    print(f"  z2027: Information Synergy (IIT-Inspired PID)")
    print(f"  Luppi et al. 2024: synergistic workspace → consciousness")
    print(f"  Task: composite judgment requiring integration")
    print(f"  Measure: Partial Information Decomposition (synergy)")
    print(f"  NO consciousness losses — synergy must EMERGE")
    print(f"{'='*70}")

    results = {}

    # A: Workspace (32 dim)
    model_a = SynergyModel(hidden_dim=32, ws_dim=32)
    results['A'] = run_condition(
        'A: Workspace (32 dim)', model_a,
        train_data, test_data, device, args.epochs, args.batch_size
    )

    # B: No workspace
    model_b = SynergyModel(hidden_dim=32, ws_dim=None)
    results['B'] = run_condition(
        'B: No workspace (direct concat)', model_b,
        train_data, test_data, device, args.epochs, args.batch_size
    )

    # C: Wide workspace (128 dim)
    model_c = SynergyModel(hidden_dim=32, ws_dim=128)
    results['C'] = run_condition(
        'C: Wide workspace (128 dim)', model_c,
        train_data, test_data, device, args.epochs, args.batch_size
    )

    # D: Random (untrained)
    model_d = SynergyModel(hidden_dim=32, ws_dim=32)
    results['D'] = run_condition(
        'D: Random (untrained)', model_d,
        train_data, test_data, device, skip_train=True
    )

    # ---------- Analysis ----------
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Information Synergy")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<40} {'CompAcc':>8} {'Synergy':>8} {'SynRatio':>9} {'MI(A,B;Y)':>10}")
    print(f"  {'-'*75}")
    for key in ['A', 'B', 'C', 'D']:
        r = results[key]
        print(f"  {r['label']:<40} {r['acc_composite']:>8.4f} {r['synergy']:>8.4f} "
              f"{r['synergy_ratio']:>9.4f} {r['mi_ab']:>10.4f}")

    # Tests
    t1 = results['A']['synergy'] > results['B']['synergy']  # Workspace > no workspace
    t2 = results['A']['synergy'] > results['D']['synergy'] + 0.01  # Trained > random
    t3 = results['A']['acc_composite'] > 0.75  # Actually learns composite task
    t4 = results['A']['synergy_ratio'] > 0.05  # Synergy is non-trivial fraction of total MI

    print(f"\n  T1: Workspace > no-workspace synergy:         "
          f"{'PASS' if t1 else 'FAIL'} ({results['A']['synergy']:.4f} vs {results['B']['synergy']:.4f})")
    print(f"  T2: Trained > random synergy (>0.01):          "
          f"{'PASS' if t2 else 'FAIL'} ({results['A']['synergy']:.4f} vs {results['D']['synergy']:.4f})")
    print(f"  T3: Composite task learned (>0.75):            "
          f"{'PASS' if t3 else 'FAIL'} ({results['A']['acc_composite']:.4f})")
    print(f"  T4: Synergy ratio > 5%:                        "
          f"{'PASS' if t4 else 'FAIL'} ({results['A']['synergy_ratio']:.4f})")

    n_pass = sum([t1, t2, t3, t4])
    if n_pass >= 4:
        verdict = "GENUINE_SYNERGY_CONFIRMED"
    elif n_pass >= 3:
        verdict = "SYNERGY_PARTIAL"
    elif n_pass >= 2:
        verdict = "SYNERGY_WEAK"
    else:
        verdict = "NO_SYNERGY"

    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    # Save
    output = {
        'experiment': 'z2027_information_synergy',
        'hypothesis': 'Workspace creates synergistic information integration (PID synergy)',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': [
            'Tononi 2004 (IIT)',
            'Williams & Beer 2010 (PID)',
            'Luppi et al. 2024 eLife (Synergistic workspace)',
            'Mediano et al. 2021 (Extended information dynamics)',
        ],
        'design_principle': 'NO consciousness losses. Synergy must EMERGE from task demands.',
        'conditions': results,
        'tests': {
            't1_workspace_gt_no_workspace': bool(t1),
            't2_trained_gt_random': bool(t2),
            't3_composite_learned': bool(t3),
            't4_synergy_ratio_nontrivial': bool(t4),
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
    results_path = Path(__file__).parent.parent / 'results' / 'z2027_information_synergy.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
