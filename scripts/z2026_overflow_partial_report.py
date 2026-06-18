#!/usr/bin/env python3
"""
z2026: Overflow / Partial Report Paradigm (Block 2011)

Tests the distinction between access consciousness (what workspace broadcasts)
and phenomenal consciousness (what's represented but not broadcast).

Setup: Present MORE items than workspace can hold simultaneously.
  - Present N items (e.g., 8 MNIST digits)
  - Cue one position AFTER presentation
  - Test: can the model report the cued item?

GWT prediction:
  - Workspace model: reports ONLY items that were broadcast (capacity-limited)
  - Pre-cue: high accuracy (workspace allocates to cued item)
  - Post-cue: partial accuracy (some items encoded, some lost)
  - Random cue after workspace: accuracy drops to capacity/N

Additional test: information persistence
  - Even items NOT broadcast may leave traces in non-workspace layers
  - This tests Block's "phenomenal overflow" hypothesis
  - Measure: encoder retains info that workspace discards

Architecture:
  - Encoder: processes each item independently → hidden_dim per item
  - Workspace: bottleneck that selects/broadcasts K of N items
  - Decoder: classification from workspace output
  - Probe: classification from encoder output (pre-workspace)

Conditions:
  A: Narrow workspace (ws=16, N=8 items) — expect limited report
  B: Wide workspace (ws=128, N=8 items) — expect better report
  C: No workspace (N=8 items) — expect good report (parallel)
  D: Narrow workspace (ws=16, N=16 items) — expect worse report

Success criteria:
  T1: Workspace models show capacity-limited report (acc << 1.0)
  T2: No-workspace model shows better report (parallel processing)
  T3: Encoder probe > workspace output (phenomenal overflow)
  T4: More items = worse report for workspace models

References:
  Block 2011 — Perceptual consciousness overflows cognitive access
  Sperling 1960 — Partial report paradigm
  Dehaene et al. 2006 — Conscious, preconscious, subliminal
  Lamme 2003 — Why visual attention and awareness are different

NO consciousness losses. Capacity limitation must EMERGE.
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

class OverflowDataset(Dataset):
    """Present N items, cue one position, report its class.

    Training: present N items. Random cue position.
    Model must learn to report the class of the cued item.
    """
    def __init__(self, mnist_data, n_items=8, n_samples=20000):
        self.images = mnist_data.data.float() / 255.0
        self.labels = mnist_data.targets
        self.n_items = n_items
        self.n_samples = n_samples
        self.n_images = len(self.labels)

        rng = np.random.RandomState(42)
        # Pre-generate item indices and cue positions
        self.item_indices = rng.randint(0, self.n_images, (n_samples, n_items))
        self.cue_positions = rng.randint(0, n_items, n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        items = []
        for j in range(self.n_items):
            img = self.images[self.item_indices[idx, j]].unsqueeze(0)
            items.append(img)

        items = torch.stack(items)  # [n_items, 1, 28, 28]
        cue = self.cue_positions[idx]
        target_label = self.labels[self.item_indices[idx, cue]]

        # One-hot cue
        cue_tensor = torch.zeros(self.n_items)
        cue_tensor[cue] = 1.0

        return items, cue_tensor, target_label


# ---------- Architecture ----------

class ItemEncoder(nn.Module):
    """Per-item CNN encoder."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, hidden_dim)

    def forward(self, items):
        """items: [batch, n_items, 1, 28, 28]"""
        batch, n_items = items.shape[:2]
        flat = items.view(batch * n_items, 1, 28, 28)
        h = self.conv(flat)
        h = F.relu(self.fc(h.view(batch * n_items, -1)))
        return h.view(batch, n_items, -1)  # [batch, n_items, hidden_dim]


class SelectiveWorkspace(nn.Module):
    """Workspace that must compress N items into limited capacity."""
    def __init__(self, hidden_dim, n_items, ws_dim, telemetry_dim=0):
        super().__init__()
        self.ws_dim = ws_dim
        # Attention-based selection: cue + items → workspace state
        self.query_proj = nn.Linear(n_items, ws_dim)  # cue → query
        self.key_proj = nn.Linear(hidden_dim, ws_dim)
        self.value_proj = nn.Linear(hidden_dim, ws_dim)
        self.ln = nn.LayerNorm(ws_dim)

        if telemetry_dim > 0:
            self.telem_proj = nn.Linear(telemetry_dim, ws_dim)
        else:
            self.telem_proj = None

    def forward(self, item_features, cue, telemetry=None):
        """
        item_features: [batch, n_items, hidden_dim]
        cue: [batch, n_items] (one-hot or soft)
        """
        # Cue-guided attention over items
        query = self.query_proj(cue).unsqueeze(1)  # [batch, 1, ws_dim]
        keys = self.key_proj(item_features)  # [batch, n_items, ws_dim]
        values = self.value_proj(item_features)  # [batch, n_items, ws_dim]

        # Attention
        scores = torch.bmm(query, keys.transpose(1, 2))  # [batch, 1, n_items]
        scores = scores / (self.ws_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # Weighted sum
        ws_state = torch.bmm(attn, values).squeeze(1)  # [batch, ws_dim]

        if telemetry is not None and self.telem_proj is not None:
            ws_state = ws_state + self.telem_proj(telemetry)

        return self.ln(ws_state), attn.squeeze(1)


class NoWorkspace(nn.Module):
    """Direct access to item features — no bottleneck."""
    def __init__(self, hidden_dim, n_items):
        super().__init__()
        # Simple: select cued item directly
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, item_features, cue, telemetry=None):
        # Use cue to select item (soft attention)
        attn = cue.unsqueeze(-1)  # [batch, n_items, 1]
        selected = (item_features * attn).sum(dim=1)  # [batch, hidden_dim]
        return F.relu(self.fc(selected)), cue


class OverflowModel(nn.Module):
    def __init__(self, hidden_dim=64, n_items=8, ws_dim=None, telemetry_dim=0):
        super().__init__()
        self.encoder = ItemEncoder(hidden_dim)
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.ws_dim = ws_dim
        self.telemetry_dim = telemetry_dim

        if ws_dim is not None:
            self.workspace = SelectiveWorkspace(hidden_dim, n_items, ws_dim, telemetry_dim)
            clf_dim = ws_dim
        else:
            self.workspace = NoWorkspace(hidden_dim, n_items)
            clf_dim = hidden_dim

        # Main classifier (from workspace output)
        self.classifier = nn.Linear(clf_dim, 10)

        # Encoder probe (from pre-workspace features, for overflow test)
        self.encoder_probe = nn.Linear(hidden_dim, 10)

    def forward(self, items, cue, telemetry=None, return_probes=False):
        """
        items: [batch, n_items, 1, 28, 28]
        cue: [batch, n_items]
        """
        item_features = self.encoder(items)  # [batch, n_items, hidden_dim]

        ws_output, attn = self.workspace(item_features, cue, telemetry)
        logits = self.classifier(ws_output)

        if return_probes:
            # Encoder probe: select cued item's encoder features directly
            cue_idx = cue.argmax(dim=1)  # [batch]
            batch_idx = torch.arange(items.size(0), device=items.device)
            cued_features = item_features[batch_idx, cue_idx]  # [batch, hidden_dim]
            probe_logits = self.encoder_probe(cued_features)
            return logits, probe_logits, attn

        return logits


# ---------- Training ----------

def train_model(model, train_data, device, n_items=8, epochs=15,
                batch_size=64, n_train=20000):
    dataset = OverflowDataset(train_data, n_items=n_items, n_samples=n_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for items, cue, labels in loader:
            items = items.to(device)
            cue = cue.to(device)
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
                    ).expand(items.size(0), -1)
                except Exception:
                    telemetry = None

            optimizer.zero_grad()
            logits, probe_logits, _ = model(items, cue, telemetry, return_probes=True)

            # Train both main classifier and encoder probe
            loss_main = F.cross_entropy(logits, labels)
            loss_probe = F.cross_entropy(probe_logits, labels)
            loss = loss_main + 0.3 * loss_probe  # Probe is auxiliary

            loss.backward()
            optimizer.step()

            total_loss += loss_main.item() * items.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += items.size(0)

        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.3f}  ({elapsed:.1f}s)")


@torch.no_grad()
def evaluate(model, test_data, device, n_items=8, n_test=5000):
    """Evaluate main accuracy and encoder probe accuracy."""
    dataset = OverflowDataset(test_data, n_items=n_items, n_samples=n_test)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()

    main_correct = 0
    probe_correct = 0
    total = 0
    all_attn = []

    for items, cue, labels in loader:
        items = items.to(device)
        cue = cue.to(device)
        labels = labels.to(device)

        telemetry = None
        if model.telemetry_dim > 0:
            telemetry = torch.zeros(items.size(0), model.telemetry_dim, device=device)

        logits, probe_logits, attn = model(items, cue, telemetry, return_probes=True)
        main_correct += (logits.argmax(1) == labels).sum().item()
        probe_correct += (probe_logits.argmax(1) == labels).sum().item()
        total += items.size(0)
        all_attn.append(attn.cpu())

    all_attn = torch.cat(all_attn, dim=0).numpy()

    # Attention entropy (how spread out is workspace attention?)
    attn_entropy = -(all_attn * np.log(all_attn + 1e-10)).sum(axis=1).mean()
    max_entropy = np.log(n_items)

    return {
        'main_acc': main_correct / total,
        'probe_acc': probe_correct / total,
        'attn_entropy': float(attn_entropy),
        'max_entropy': float(max_entropy),
        'attn_concentration': 1.0 - attn_entropy / max_entropy,
    }


# ---------- Main ----------

def run_condition(label, model, train_data, test_data, device,
                  n_items=8, epochs=15, batch_size=64):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Workspace: {model.ws_dim or 'None (direct)'}")
    print(f"  Items: {n_items}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*70}")

    model = model.to(device)
    train_model(model, train_data, device, n_items=n_items,
                epochs=epochs, batch_size=batch_size)

    metrics = evaluate(model, test_data, device, n_items=n_items)

    print(f"\n  --- Results ---")
    print(f"  Main accuracy (workspace):  {metrics['main_acc']:.4f}")
    print(f"  Probe accuracy (encoder):   {metrics['probe_acc']:.4f}")
    print(f"  Overflow gap (probe-main):  {metrics['probe_acc']-metrics['main_acc']:+.4f}")
    print(f"  Attention entropy:          {metrics['attn_entropy']:.4f} / {metrics['max_entropy']:.4f}")
    print(f"  Attention concentration:    {metrics['attn_concentration']:.4f}")

    return {
        'label': label,
        'ws_dim': model.ws_dim,
        'n_items': n_items,
        'n_params': n_params,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2026] Device: {device}")

    from torchvision import datasets
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    print("Loading MNIST...")
    train_data = datasets.MNIST(str(data_dir), train=True, download=True)
    test_data = datasets.MNIST(str(data_dir), train=False, download=True)

    print(f"\n{'='*70}")
    print(f"  z2026: Overflow / Partial Report Paradigm")
    print(f"  Block 2011: access consciousness vs phenomenal overflow")
    print(f"  Task: report cued item from array of N")
    print(f"  NO consciousness losses — capacity limits must EMERGE")
    print(f"{'='*70}")

    results = {}

    # A: Narrow workspace, 8 items
    model_a = OverflowModel(hidden_dim=64, n_items=8, ws_dim=16)
    results['A'] = run_condition(
        'A: Narrow workspace (16 dim), 8 items', model_a,
        train_data, test_data, device, n_items=8,
        epochs=args.epochs, batch_size=args.batch_size
    )

    # B: Wide workspace, 8 items
    model_b = OverflowModel(hidden_dim=64, n_items=8, ws_dim=128)
    results['B'] = run_condition(
        'B: Wide workspace (128 dim), 8 items', model_b,
        train_data, test_data, device, n_items=8,
        epochs=args.epochs, batch_size=args.batch_size
    )

    # C: No workspace, 8 items
    model_c = OverflowModel(hidden_dim=64, n_items=8, ws_dim=None)
    results['C'] = run_condition(
        'C: No workspace (direct), 8 items', model_c,
        train_data, test_data, device, n_items=8,
        epochs=args.epochs, batch_size=args.batch_size
    )

    # D: Narrow workspace, 16 items (overloaded)
    model_d = OverflowModel(hidden_dim=64, n_items=16, ws_dim=16)
    results['D'] = run_condition(
        'D: Narrow workspace (16 dim), 16 items', model_d,
        train_data, test_data, device, n_items=16,
        epochs=args.epochs, batch_size=args.batch_size
    )

    # ---------- Analysis ----------
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Overflow / Partial Report")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<45} {'Main':>6} {'Probe':>6} {'Overflow':>9} {'AttnConc':>9}")
    print(f"  {'-'*75}")
    for key in ['A', 'B', 'C', 'D']:
        r = results[key]
        overflow = r['probe_acc'] - r['main_acc']
        print(f"  {r['label']:<45} {r['main_acc']:>6.3f} {r['probe_acc']:>6.3f} "
              f"{overflow:>+9.4f} {r['attn_concentration']:>9.4f}")

    # Tests
    # T1: Narrow workspace has lower main acc than no-workspace (capacity limited)
    t1 = results['A']['main_acc'] < results['C']['main_acc'] - 0.01

    # T2: No-workspace has best report accuracy
    t2 = results['C']['main_acc'] >= max(results['A']['main_acc'], results['D']['main_acc'])

    # T3: Encoder probe > workspace output (phenomenal overflow)
    overflow_a = results['A']['probe_acc'] - results['A']['main_acc']
    t3 = overflow_a > 0.01  # Encoder retains more than workspace reports

    # T4: More items = worse workspace report
    t4 = results['D']['main_acc'] < results['A']['main_acc'] - 0.01

    print(f"\n  T1: Workspace capacity-limited (A < C):      "
          f"{'PASS' if t1 else 'FAIL'} ({results['A']['main_acc']:.3f} vs {results['C']['main_acc']:.3f})")
    print(f"  T2: No-workspace best report:                  "
          f"{'PASS' if t2 else 'FAIL'} (C={results['C']['main_acc']:.3f})")
    print(f"  T3: Phenomenal overflow (probe > main):        "
          f"{'PASS' if t3 else 'FAIL'} (gap={overflow_a:+.4f})")
    print(f"  T4: More items = worse report (D < A):         "
          f"{'PASS' if t4 else 'FAIL'} ({results['D']['main_acc']:.3f} vs {results['A']['main_acc']:.3f})")

    n_pass = sum([t1, t2, t3, t4])
    if n_pass >= 4:
        verdict = "GENUINE_OVERFLOW_CONFIRMED"
    elif n_pass >= 3:
        verdict = "OVERFLOW_PARTIAL"
    elif n_pass >= 2:
        verdict = "OVERFLOW_WEAK"
    else:
        verdict = "NO_OVERFLOW_PATTERN"

    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    # Save
    output = {
        'experiment': 'z2026_overflow_partial_report',
        'hypothesis': 'Workspace creates capacity-limited access; encoder retains phenomenal overflow',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': [
            'Block 2011 (Perceptual consciousness overflows cognitive access)',
            'Sperling 1960 (Partial report paradigm)',
            'Dehaene et al. 2006 (Conscious, preconscious, subliminal)',
        ],
        'design_principle': 'NO consciousness losses. Capacity limitation must EMERGE.',
        'conditions': results,
        'tests': {
            't1_workspace_capacity_limited': bool(t1),
            't2_no_workspace_best': bool(t2),
            't3_phenomenal_overflow': bool(t3),
            't4_more_items_worse': bool(t4),
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
    results_path = Path(__file__).parent.parent / 'results' / 'z2026_overflow_partial_report.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
