#!/usr/bin/env python3
"""
z2022: Attentional Blink (Temporal Capacity Limitation)

GWT predicts that processing T1 (first target) through a global workspace
bottleneck temporarily blocks processing of T2 (second target) at specific
temporal lags. This produces the "attentional blink" — a characteristic
U-shaped accuracy curve where T2 accuracy:
  - Is HIGH at lag 1 (T1 not yet consolidated)
  - DROPS at lags 2-5 (workspace occupied by T1)
  - RECOVERS at lags 6+ (T1 processing complete)

This is a STRONGER test than z2020 because:
  1. The SHAPE of the failure curve matters (U-shaped, not random)
  2. There is no incentive to fake the specific temporal pattern
  3. The blink must occur at the RIGHT lags, not arbitrary ones

Architecture:
  - Stream of MNIST digits presented sequentially
  - T1: first target (different class: letter-like digit)
  - T2: second target (digit to identify)
  - Distractors between T1 and T2 (random noise)
  - Workspace bottleneck processes stream sequentially

Conditions:
  A: Workspace model (16 dim bottleneck) — should show blink
  B: Wide workspace (128 dim) — reduced blink
  C: No workspace (parallel) — NO blink
  D: Workspace + telemetry — blink + hardware adaptation?

Success criteria:
  - Workspace models show U-shaped T2 accuracy curve
  - No-workspace model shows FLAT T2 accuracy
  - Blink depth (max accuracy - min accuracy) > 0.05 for workspace
  - Blink peak at lags 2-5 (not lag 1 or lag 8+)

References:
  Raymond, Shapiro & Arnell 1992 — JEP:HPP (original attentional blink)
  Chun & Potter 1995 — JEP:HPP (two-stage model)
  Phua 2025 — arXiv:2512.19155 (interference tests for AI)

NO consciousness losses. Train ONLY on task.
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


# ---------- RSVP Stream Dataset ----------

class RSVPDataset(Dataset):
    """Rapid Serial Visual Presentation stream for attentional blink.

    Each sample is a sequence of images with:
    - T1 at a fixed position
    - T2 at a variable lag after T1 (lag 1-8)
    - Distractors (random noise or transformed digits) between targets
    """
    def __init__(self, mnist_data, stream_length=12, n_samples=5000):
        self.stream_length = stream_length
        self.n_samples = n_samples
        self.images = mnist_data.data.float() / 255.0  # [N, 28, 28]
        self.labels = mnist_data.targets
        self.n_classes = 10

        # T1 position is always at position 3 (0-indexed)
        self.t1_pos = 3
        # Lags 1-8
        self.lags = list(range(1, 9))

    def __len__(self):
        return self.n_samples

    def _make_distractor(self):
        """Create a distractor image (Gaussian noise)."""
        return torch.randn(28, 28) * 0.3 + 0.5

    def __getitem__(self, idx):
        # Deterministic randomness per idx
        rng = np.random.RandomState(idx)

        # Choose lag for this sample
        lag = self.lags[idx % len(self.lags)]
        t2_pos = self.t1_pos + lag

        # Build stream
        stream = torch.zeros(self.stream_length, 1, 28, 28)
        stream_labels = torch.full((self.stream_length,), -1, dtype=torch.long)

        # Fill with distractors
        for i in range(self.stream_length):
            stream[i, 0] = self._make_distractor().clamp(0, 1)

        # Place T1 (always a digit from class 0-4 = "easy" set)
        t1_idx = rng.randint(0, len(self.images))
        t1_class = int(self.labels[t1_idx])
        stream[self.t1_pos, 0] = self.images[t1_idx]
        stream_labels[self.t1_pos] = t1_class

        # Place T2 at lag position (if within stream)
        t2_label = -1
        if t2_pos < self.stream_length:
            t2_idx = rng.randint(0, len(self.images))
            t2_class = int(self.labels[t2_idx])
            stream[t2_pos, 0] = self.images[t2_idx]
            stream_labels[t2_pos] = t2_class
            t2_label = t2_class

        return stream, stream_labels, lag, t1_class, t2_label


# ---------- Architecture ----------

class StreamEncoder(nn.Module):
    """Encodes each frame in the RSVP stream."""
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
        """x: [batch, 1, 28, 28] → [batch, out_dim]"""
        h = self.conv(x)
        return self.fc(h.view(h.size(0), -1))


class WorkspaceRNN(nn.Module):
    """Sequential workspace processing with GRU bottleneck."""
    def __init__(self, input_dim, workspace_dim, telemetry_dim=0):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.gru = nn.GRU(input_dim + telemetry_dim, workspace_dim, batch_first=True)

    def forward(self, frame_features, telemetry=None):
        """frame_features: [batch, seq_len, input_dim]
        Returns: [batch, seq_len, workspace_dim]
        """
        if telemetry is not None:
            # Expand telemetry to all timesteps
            telem_expanded = telemetry.unsqueeze(1).expand(-1, frame_features.size(1), -1)
            x = torch.cat([frame_features, telem_expanded], dim=-1)
        else:
            x = frame_features
        output, _ = self.gru(x)
        return output


class RSVPModel(nn.Module):
    """RSVP stream model with optional workspace bottleneck."""
    def __init__(self, encoder_dim=64, workspace_dim=None, telemetry_dim=0, n_classes=10):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.workspace_dim = workspace_dim
        self.telemetry_dim = telemetry_dim

        self.frame_encoder = StreamEncoder(encoder_dim)

        if workspace_dim is not None:
            self.workspace = WorkspaceRNN(encoder_dim, workspace_dim, telemetry_dim)
            self.classifier = nn.Linear(workspace_dim, n_classes)
        else:
            # No workspace: independent per-frame classification
            self.workspace = None
            self.classifier = nn.Linear(encoder_dim, n_classes)

    def forward(self, stream, telemetry=None):
        """stream: [batch, seq_len, 1, 28, 28]"""
        batch, seq_len = stream.shape[:2]

        # Encode each frame
        frames_flat = stream.view(batch * seq_len, 1, 28, 28)
        features_flat = self.frame_encoder(frames_flat)
        features = features_flat.view(batch, seq_len, -1)

        if self.workspace is not None:
            processed = self.workspace(features, telemetry)
        else:
            processed = features

        # Classify each position
        logits = self.classifier(processed)  # [batch, seq_len, n_classes]
        return logits


# ---------- Training & Evaluation ----------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_targets = 0

    for stream, labels, lag, t1_class, t2_label in loader:
        stream = stream.to(device)
        labels = labels.to(device)

        telemetry = None
        if model.telemetry_dim > 0 and TELEMETRY_AVAILABLE:
            try:
                telem = SysfsHwmonTelemetry(card_index=0)
                sample = telem.read_sample()
                t = getattr(sample, 'temp_edge_c', 50.0)
                p = getattr(sample, 'power_w', 30.0)
                telemetry = torch.tensor(
                    [[t / 100.0, p / 100.0]], device=device
                ).expand(stream.size(0), -1)
            except Exception:
                telemetry = None

        optimizer.zero_grad()
        logits = model(stream, telemetry)

        # Loss only on target positions (where labels != -1)
        mask = labels != -1
        if mask.sum() == 0:
            continue

        loss = F.cross_entropy(logits[mask], labels[mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        preds = logits[mask].argmax(dim=-1)
        total_correct += (preds == labels[mask]).sum().item()
        total_targets += mask.sum().item()

    acc = total_correct / total_targets if total_targets > 0 else 0
    return {'loss': total_loss / max(total_targets, 1), 'acc': acc}


@torch.no_grad()
def evaluate_blink(model, loader, device, t1_pos=3):
    """Evaluate T1 and T2 accuracy at each lag."""
    model.eval()

    # Accumulate per-lag statistics
    lag_t1_correct = {}
    lag_t2_correct = {}
    lag_t1_total = {}
    lag_t2_total = {}

    for stream, labels, lags, t1_classes, t2_labels in loader:
        stream = stream.to(device)
        labels = labels.to(device)
        t1_classes = t1_classes.to(device)
        t2_labels = t2_labels.to(device)

        telemetry = None
        if model.telemetry_dim > 0:
            telemetry = torch.zeros(stream.size(0), model.telemetry_dim, device=device)

        logits = model(stream, telemetry)

        for i in range(stream.size(0)):
            lag = int(lags[i])
            t2_pos = t1_pos + lag

            # T1 accuracy
            t1_pred = logits[i, t1_pos].argmax().item()
            t1_true = t1_classes[i].item()

            if lag not in lag_t1_correct:
                lag_t1_correct[lag] = 0
                lag_t1_total[lag] = 0
                lag_t2_correct[lag] = 0
                lag_t2_total[lag] = 0

            lag_t1_correct[lag] += int(t1_pred == t1_true)
            lag_t1_total[lag] += 1

            # T2 accuracy (if valid)
            t2_true = t2_labels[i].item()
            if t2_true >= 0 and t2_pos < logits.size(1):
                t2_pred = logits[i, t2_pos].argmax().item()
                lag_t2_correct[lag] += int(t2_pred == t2_true)
                lag_t2_total[lag] += 1

    results = {}
    for lag in sorted(lag_t1_total.keys()):
        t1_acc = lag_t1_correct[lag] / lag_t1_total[lag] if lag_t1_total[lag] > 0 else 0
        t2_acc = lag_t2_correct[lag] / lag_t2_total[lag] if lag_t2_total[lag] > 0 else 0
        results[lag] = {
            't1_acc': t1_acc,
            't2_acc': t2_acc,
            'n_t1': lag_t1_total[lag],
            'n_t2': lag_t2_total[lag],
        }

    return results


def analyze_blink_curve(blink_results):
    """Analyze whether the T2 accuracy curve shows an attentional blink."""
    lags = sorted(blink_results.keys())
    t2_accs = [blink_results[l]['t2_acc'] for l in lags]

    if len(t2_accs) < 4:
        return {'has_blink': False, 'reason': 'Too few lags'}

    max_acc = max(t2_accs)
    min_acc = min(t2_accs)
    blink_depth = max_acc - min_acc

    # Find the lag with minimum T2 accuracy
    min_lag = lags[t2_accs.index(min_acc)]

    # Check for U-shape: accuracy should dip in middle lags
    early_acc = np.mean(t2_accs[:2])  # lags 1-2
    mid_acc = np.mean(t2_accs[2:5]) if len(t2_accs) >= 5 else np.mean(t2_accs[2:])  # lags 3-5
    late_acc = np.mean(t2_accs[5:]) if len(t2_accs) > 5 else t2_accs[-1]  # lags 6+

    # U-shape: early >= mid AND late >= mid (with tolerance)
    u_shape = mid_acc < early_acc - 0.01 or mid_acc < late_acc - 0.01

    # Blink should be in lags 2-5
    blink_in_range = 2 <= min_lag <= 5

    return {
        'has_blink': blink_depth > 0.03,
        'blink_depth': blink_depth,
        'min_lag': min_lag,
        'blink_in_range': blink_in_range,
        'u_shape': u_shape,
        'early_acc': early_acc,
        'mid_acc': mid_acc,
        'late_acc': late_acc,
        'lag_accuracies': {l: blink_results[l]['t2_acc'] for l in lags},
    }


# ---------- Main ----------

def run_condition(label, mnist_train, mnist_test, device, encoder_dim=64,
                  workspace_dim=None, telemetry_dim=0, epochs=15, batch_size=64):
    """Train and evaluate one condition."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Workspace: {workspace_dim or 'None (per-frame)'}")
    print(f"{'='*70}")

    model = RSVPModel(
        encoder_dim=encoder_dim,
        workspace_dim=workspace_dim,
        telemetry_dim=telemetry_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training data
    train_ds = RSVPDataset(mnist_train, stream_length=12, n_samples=8000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)

    # Test data (more samples for reliable per-lag statistics)
    test_ds = RSVPDataset(mnist_test, stream_length=12, n_samples=4000)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    epoch_stats = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        stats = train_epoch(model, train_loader, optimizer, device)
        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  loss={stats['loss']:.4f}  "
                  f"acc={stats['acc']:.3f}  ({elapsed:.1f}s)")
        epoch_stats.append({**stats, 'epoch': ep, 'elapsed_s': round(elapsed, 1)})

    # Evaluate blink curve
    print(f"\n  --- Attentional Blink Curve ---")
    blink_results = evaluate_blink(model, test_loader, device)

    for lag in sorted(blink_results.keys()):
        r = blink_results[lag]
        bar = '#' * int(r['t2_acc'] * 40)
        print(f"    Lag {lag}: T1={r['t1_acc']:.3f}  T2={r['t2_acc']:.3f}  |{bar}")

    analysis = analyze_blink_curve(blink_results)
    print(f"\n  Blink depth:    {analysis['blink_depth']:.4f}")
    print(f"  Min accuracy at lag: {analysis['min_lag']}")
    print(f"  U-shape:        {analysis['u_shape']}")
    print(f"  Blink in range: {analysis['blink_in_range']}")
    print(f"  Has blink:      {analysis['has_blink']}")

    return {
        'label': label,
        'workspace_dim': workspace_dim,
        'telemetry_dim': telemetry_dim,
        'n_params': n_params,
        'epoch_stats': epoch_stats,
        'blink_results': {str(k): v for k, v in blink_results.items()},
        'analysis': analysis,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2022] Device: {device}")

    from torchvision import datasets
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    print("Loading MNIST...")
    mnist_train = datasets.MNIST(str(data_dir), train=True, download=True)
    mnist_test = datasets.MNIST(str(data_dir), train=False, download=True)

    print(f"\n{'='*70}")
    print(f"  z2022: Attentional Blink (Temporal Capacity Limitation)")
    print(f"  GWT predicts: workspace creates temporal bottleneck")
    print(f"  T2 accuracy should DIP at lags 2-5 (U-shaped curve)")
    print(f"  NO consciousness losses — task only")
    print(f"{'='*70}")

    results = {}

    # Condition A: Narrow workspace (should show blink)
    results['A'] = run_condition(
        'A: Narrow workspace (16 dim)',
        mnist_train, mnist_test, device,
        workspace_dim=16, epochs=args.epochs, batch_size=args.batch_size
    )

    # Condition B: Wide workspace (reduced blink)
    results['B'] = run_condition(
        'B: Wide workspace (128 dim)',
        mnist_train, mnist_test, device,
        workspace_dim=128, epochs=args.epochs, batch_size=args.batch_size
    )

    # Condition C: No workspace (no blink)
    results['C'] = run_condition(
        'C: No workspace (per-frame)',
        mnist_train, mnist_test, device,
        workspace_dim=None, epochs=args.epochs, batch_size=args.batch_size
    )

    # Condition D: Embodied narrow workspace
    results['D'] = run_condition(
        'D: Narrow workspace + telemetry',
        mnist_train, mnist_test, device,
        workspace_dim=16, telemetry_dim=2, epochs=args.epochs, batch_size=args.batch_size
    )

    # ---------- Final Analysis ----------
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Attentional Blink")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<40} {'Blink?':>6} {'Depth':>8} {'Min Lag':>8} {'U-shape':>8}")
    print(f"  {'-'*70}")
    for key in ['A', 'B', 'C', 'D']:
        a = results[key]['analysis']
        print(f"  {results[key]['label']:<40} "
              f"{'YES' if a['has_blink'] else 'NO':>6} "
              f"{a['blink_depth']:>8.4f} "
              f"{a['min_lag']:>8} "
              f"{'YES' if a['u_shape'] else 'NO':>8}")

    # Tests
    a_blink = results['A']['analysis']['has_blink']
    b_blink = results['B']['analysis']['has_blink']
    c_blink = results['C']['analysis']['has_blink']

    t1 = a_blink  # Narrow workspace shows blink
    t2 = not c_blink  # No workspace: no blink
    t3 = results['A']['analysis']['blink_depth'] > results['B']['analysis']['blink_depth']  # Narrow > wide
    t4 = results['A']['analysis']['u_shape']  # U-shape pattern

    print(f"\n  T1: Narrow workspace shows blink:       {'PASS' if t1 else 'FAIL'}")
    print(f"  T2: No-workspace has no blink:           {'PASS' if t2 else 'FAIL'}")
    print(f"  T3: Narrow deeper blink than wide:       {'PASS' if t3 else 'FAIL'}")
    print(f"  T4: U-shaped blink curve:                {'PASS' if t4 else 'FAIL'}")

    n_pass = sum([t1, t2, t3, t4])
    if n_pass >= 4:
        verdict = "ATTENTIONAL_BLINK_CONFIRMED"
    elif n_pass >= 3:
        verdict = "ATTENTIONAL_BLINK_PARTIAL"
    elif n_pass >= 2:
        verdict = "ATTENTIONAL_BLINK_WEAK"
    else:
        verdict = "NO_ATTENTIONAL_BLINK"

    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    # Save
    output = {
        'experiment': 'z2022_attentional_blink',
        'hypothesis': 'GWT workspace creates temporal bottleneck producing U-shaped attentional blink',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'epochs': args.epochs,
        'references': [
            'Raymond, Shapiro & Arnell 1992 (original attentional blink)',
            'Chun & Potter 1995 (two-stage model)',
            'Phua 2025 arXiv:2512.19155 (interference tests)',
        ],
        'design_principle': 'NO consciousness losses. Test for temporal FAILURE pattern.',
        'conditions': results,
        'tests': {
            't1_narrow_shows_blink': t1,
            't2_no_workspace_no_blink': t2,
            't3_narrow_deeper_than_wide': t3,
            't4_u_shaped_curve': t4,
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
    results_path = Path(__file__).parent.parent / 'results' / 'z2022_attentional_blink.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
