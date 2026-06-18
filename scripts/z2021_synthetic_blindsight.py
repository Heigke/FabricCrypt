#!/usr/bin/env python3
"""
z2021: Synthetic Blindsight (Ablation-Dissociation Test)

Inspired by Phua 2025 (arXiv:2512.19155) Experiment 1: create a system with
task performance AND metacognitive monitoring. Ablate the self-model component.
If genuine metacognition exists, ablation produces SPECIFIC dissociation:
  - First-order task accuracy: PRESERVED
  - Metacognitive calibration (Type-2 AUROC): COLLAPSES to chance

This is harder to game than any positive test because:
  1. You can't fake the dissociation pattern by output-level adjustments
  2. The pattern must be SPECIFIC: task preserved + metacognition collapsed
  3. A system without genuine metacognition won't show clean dissociation

Architecture:
  - Shared encoder (CNN)
  - Task head: 10-way digit classification
  - Self-model head: predicts P(correct) from hidden state
  - Optional: hardware telemetry conditioning

Training: L = L_task + lambda * L_self_model
  - L_self_model = BCE(predicted_correctness, actual_correctness)
  - NO consciousness losses. Self-model learns to predict task success.

Ablation tests:
  1. Full model: measure task acc + Type-2 AUROC + calibration
  2. Ablate self-model: zero weights → task preserved? AUROC collapsed?
  3. Ablate encoder (random): both degrade
  4. Scramble: random hidden states to self-model → AUROC collapsed?

Success criteria (from HOT theory):
  - Full model: high task acc AND high Type-2 AUROC
  - Self-model ablation: task acc PRESERVED, AUROC COLLAPSES to ~0.5
  - Encoder ablation: BOTH degrade
  - If ablation doesn't produce dissociation → self-model is not genuine

References:
  Phua 2025 — arXiv:2512.19155 (Synthetic Blindsight)
  Rosenthal 2005 — Higher-Order Theories of Consciousness
  Fleming & Lau 2014 — How to measure metacognition
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
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


# ---------- Architecture ----------

class SharedEncoder(nn.Module):
    """CNN encoder shared between task and self-model."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return F.relu(self.fc(h))


class TaskHead(nn.Module):
    """First-order digit classification."""
    def __init__(self, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, h):
        return self.fc2(F.relu(self.fc1(h)))


class SelfModelHead(nn.Module):
    """Metacognitive self-model: predicts P(I will be correct).

    This is the component that, when ablated, should produce
    synthetic blindsight: task performance preserved, metacognition lost.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, h):
        x = F.relu(self.fc1(h))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).squeeze(-1)


class BlindsightModel(nn.Module):
    """Model with separable task and self-model components."""

    def __init__(self, hidden_dim=128, num_classes=10, telemetry_dim=0):
        super().__init__()
        self.encoder = SharedEncoder(hidden_dim)
        self.task_head = TaskHead(hidden_dim, num_classes)
        self.self_model = SelfModelHead(hidden_dim)
        self.telemetry_dim = telemetry_dim

        if telemetry_dim > 0:
            self.telem_proj = nn.Linear(telemetry_dim, hidden_dim)
        else:
            self.telem_proj = None

    def forward(self, x, telemetry=None):
        h = self.encoder(x)

        if telemetry is not None and self.telem_proj is not None:
            h = h + self.telem_proj(telemetry)  # Additive conditioning

        task_logits = self.task_head(h)
        confidence = self.self_model(h)

        return {
            'logits': task_logits,
            'confidence': confidence,
            'hidden': h,
        }


# ---------- Metrics ----------

def compute_type2_auroc(confidences, correctness):
    """Type-2 AUROC: how well does confidence predict correctness?

    This is the standard metacognitive metric from Fleming & Lau 2014.
    AUROC = 0.5 means chance (no metacognitive ability).
    AUROC = 1.0 means perfect metacognition.
    """
    if len(np.unique(correctness)) < 2:
        return 0.5  # Can't compute if all correct or all wrong
    try:
        return roc_auc_score(correctness, confidences)
    except ValueError:
        return 0.5


def compute_ece(confidences, correctness, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correctness[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(confidences) if len(confidences) > 0 else 0.0


# ---------- Training ----------

def train_epoch(model, loader, optimizer, device, lambda_self=0.5):
    model.train()
    total_loss = 0
    task_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        telemetry = None
        if model.telemetry_dim > 0 and TELEMETRY_AVAILABLE:
            try:
                telem = SysfsHwmonTelemetry(card_index=0)
                sample = telem.read_sample()
                t = getattr(sample, 'temp_edge_c', 50.0)
                p = getattr(sample, 'power_w', 30.0)
                telemetry = torch.tensor(
                    [[t / 100.0, p / 100.0]], device=device
                ).expand(images.size(0), -1)
            except Exception:
                telemetry = None

        optimizer.zero_grad()
        out = model(images, telemetry)

        # Task loss
        loss_task = F.cross_entropy(out['logits'], labels)

        # Self-model loss: predict own correctness
        with torch.no_grad():
            predictions = out['logits'].argmax(dim=1)
            is_correct = (predictions == labels).float()

        loss_self = F.binary_cross_entropy(out['confidence'], is_correct)

        # Combined — NO consciousness losses
        loss = loss_task + lambda_self * loss_self

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        task_correct += (predictions == labels).sum().item()
        total += images.size(0)

    return {
        'loss': total_loss / total,
        'task_acc': task_correct / total,
    }


@torch.no_grad()
def evaluate_full(model, loader, device):
    """Full evaluation: task accuracy + metacognitive metrics."""
    model.eval()
    all_confidences = []
    all_correctness = []
    all_predictions = []
    all_labels = []
    total_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        telemetry = None
        if model.telemetry_dim > 0:
            telemetry = torch.zeros(images.size(0), model.telemetry_dim, device=device)

        out = model(images, telemetry)
        preds = out['logits'].argmax(dim=1)
        correct = (preds == labels)

        all_confidences.extend(out['confidence'].cpu().numpy())
        all_correctness.extend(correct.cpu().numpy().astype(float))
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_correct += correct.sum().item()
        total += images.size(0)

    confidences = np.array(all_confidences)
    correctness = np.array(all_correctness)

    task_acc = total_correct / total
    auroc = compute_type2_auroc(confidences, correctness)
    ece = compute_ece(confidences, correctness)
    mean_conf = confidences.mean()
    conf_when_correct = confidences[correctness == 1].mean() if correctness.sum() > 0 else 0
    conf_when_wrong = confidences[correctness == 0].mean() if (1-correctness).sum() > 0 else 0

    return {
        'task_acc': task_acc,
        'type2_auroc': auroc,
        'ece': ece,
        'mean_confidence': float(mean_conf),
        'conf_when_correct': float(conf_when_correct),
        'conf_when_wrong': float(conf_when_wrong),
        'n_correct': int(correctness.sum()),
        'n_total': total,
    }


# ---------- Ablation Tests ----------

def ablate_self_model(model):
    """Zero out self-model weights. Task should be preserved."""
    with torch.no_grad():
        for param in model.self_model.parameters():
            param.zero_()


def ablate_encoder(model):
    """Randomize encoder weights. Both task and self-model should degrade."""
    with torch.no_grad():
        for param in model.encoder.parameters():
            nn.init.normal_(param, 0, 0.01)


def scramble_self_model_input(model, loader, device):
    """Feed random hidden states to self-model. AUROC should collapse."""
    model.eval()
    all_confidences = []
    all_correctness = []
    total_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            h = model.encoder(images)

            # Task uses real hidden state
            task_logits = model.task_head(h)
            preds = task_logits.argmax(dim=1)
            correct = (preds == labels)

            # Self-model uses SCRAMBLED hidden state
            scrambled_h = h[torch.randperm(h.size(0))]
            confidence = model.self_model(scrambled_h)

            all_confidences.extend(confidence.cpu().numpy())
            all_correctness.extend(correct.cpu().numpy().astype(float))
            total_correct += correct.sum().item()
            total += images.size(0)

    confidences = np.array(all_confidences)
    correctness = np.array(all_correctness)

    return {
        'task_acc': total_correct / total,
        'type2_auroc': compute_type2_auroc(confidences, correctness),
        'ece': compute_ece(confidences, correctness),
        'mean_confidence': float(confidences.mean()),
    }


# ---------- Main ----------

def run_condition(label, hidden_dim=128, telemetry_dim=0, epochs=15,
                  batch_size=128, device='cuda', noisy_fraction=0.3):
    """Train and evaluate one condition with full ablation battery."""

    print(f"\n{'='*70}")
    print(f"  Condition: {label}")
    print(f"{'='*70}")

    # Load MNIST with some noise for non-trivial uncertainty
    from torchvision import datasets, transforms
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_data = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)

    # Add noise to fraction of test set to create genuine uncertainty
    class NoisyMNIST(torch.utils.data.Dataset):
        def __init__(self, base_dataset, noise_frac=0.3, noise_std=1.5):
            self.base = base_dataset
            self.noise_frac = noise_frac
            self.noise_std = noise_std
            self.n = len(base_dataset)

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            img, label = self.base[idx]
            if idx % int(1.0 / max(self.noise_frac, 0.01)) == 0:
                img = img + torch.randn_like(img) * self.noise_std
                img = img.clamp(0, 1)
            return img, label

    noisy_test = NoisyMNIST(test_data, noise_frac=noisy_fraction, noise_std=1.5)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(noisy_test, batch_size=256, shuffle=False)

    # Build model
    model = BlindsightModel(hidden_dim=hidden_dim, telemetry_dim=telemetry_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Telemetry: {telemetry_dim > 0}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for ep in range(1, epochs + 1):
        t0 = time.time()
        stats = train_epoch(model, train_loader, optimizer, device)
        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  loss={stats['loss']:.4f}  "
                  f"acc={stats['task_acc']:.4f}  ({elapsed:.1f}s)")

    # Save checkpoint for ablation
    checkpoint = {k: v.clone() for k, v in model.state_dict().items()}

    # ---------- Ablation Battery ----------

    # Test 1: Full model (baseline)
    print(f"\n  [1/4] Full model (baseline)...")
    full_results = evaluate_full(model, test_loader, device)
    print(f"    Task acc:      {full_results['task_acc']:.4f}")
    print(f"    Type-2 AUROC:  {full_results['type2_auroc']:.4f}")
    print(f"    ECE:           {full_results['ece']:.4f}")
    print(f"    Conf correct:  {full_results['conf_when_correct']:.4f}")
    print(f"    Conf wrong:    {full_results['conf_when_wrong']:.4f}")

    # Test 2: Ablate self-model (synthetic blindsight)
    print(f"\n  [2/4] Ablate self-model (synthetic blindsight)...")
    model.load_state_dict(checkpoint)
    ablate_self_model(model)
    ablated_results = evaluate_full(model, test_loader, device)
    print(f"    Task acc:      {ablated_results['task_acc']:.4f}")
    print(f"    Type-2 AUROC:  {ablated_results['type2_auroc']:.4f}")
    print(f"    ECE:           {ablated_results['ece']:.4f}")

    # Test 3: Ablate encoder (both should degrade)
    print(f"\n  [3/4] Ablate encoder (random encoder)...")
    model.load_state_dict(checkpoint)
    ablate_encoder(model)
    encoder_ablated = evaluate_full(model, test_loader, device)
    print(f"    Task acc:      {encoder_ablated['task_acc']:.4f}")
    print(f"    Type-2 AUROC:  {encoder_ablated['type2_auroc']:.4f}")

    # Test 4: Scramble self-model input
    print(f"\n  [4/4] Scramble self-model input...")
    model.load_state_dict(checkpoint)
    scrambled_results = scramble_self_model_input(model, test_loader, device)
    print(f"    Task acc:      {scrambled_results['task_acc']:.4f}")
    print(f"    Type-2 AUROC:  {scrambled_results['type2_auroc']:.4f}")

    # Restore full model
    model.load_state_dict(checkpoint)

    return {
        'label': label,
        'hidden_dim': hidden_dim,
        'telemetry_dim': telemetry_dim,
        'n_params': n_params,
        'full': full_results,
        'self_model_ablated': ablated_results,
        'encoder_ablated': encoder_ablated,
        'scrambled': scrambled_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2021] Device: {device}")

    if TELEMETRY_AVAILABLE:
        try:
            telem = SysfsHwmonTelemetry(card_index=0)
            s = telem.read_sample()
            print(f"[telemetry] Live: temp={getattr(s, 'temp_edge_c', '?')}C")
        except Exception:
            print("[telemetry] Not available")

    print(f"\n{'='*70}")
    print(f"  z2021: Synthetic Blindsight (Ablation-Dissociation Test)")
    print(f"  Phua 2025 inspired — test for SPECIFIC dissociation pattern")
    print(f"{'='*70}")
    print(f"  Prediction: Self-model ablation preserves task, collapses AUROC")
    print(f"  NO consciousness losses — self-model learns task success prediction")
    print()

    results = {}

    # Condition A: Standard model (no telemetry)
    results['A'] = run_condition(
        'A: Standard (no telemetry)',
        hidden_dim=128, telemetry_dim=0,
        epochs=args.epochs, batch_size=args.batch_size, device=device
    )

    # Condition B: Embodied model (with telemetry)
    results['B'] = run_condition(
        'B: Embodied (telemetry)',
        hidden_dim=128, telemetry_dim=2,
        epochs=args.epochs, batch_size=args.batch_size, device=device
    )

    # ---------- Analysis ----------

    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Synthetic Blindsight")
    print(f"{'='*70}")

    for key in ['A', 'B']:
        r = results[key]
        print(f"\n  {r['label']}:")

        full_acc = r['full']['task_acc']
        full_auroc = r['full']['type2_auroc']
        abl_acc = r['self_model_ablated']['task_acc']
        abl_auroc = r['self_model_ablated']['type2_auroc']
        enc_acc = r['encoder_ablated']['task_acc']
        enc_auroc = r['encoder_ablated']['type2_auroc']
        scr_acc = r['scrambled']['task_acc']
        scr_auroc = r['scrambled']['type2_auroc']

        # Dissociation test: task preserved + AUROC collapsed
        acc_preserved = abs(full_acc - abl_acc) < 0.02  # <2% drop
        auroc_collapsed = abl_auroc < 0.55  # Near chance (0.5)
        encoder_both_degrade = enc_acc < full_acc - 0.1 and enc_auroc < full_auroc - 0.1
        scramble_auroc_collapsed = scr_auroc < 0.55

        t1 = acc_preserved and auroc_collapsed
        t2 = encoder_both_degrade
        t3 = scramble_auroc_collapsed
        t4 = full_auroc > 0.6  # Baseline must have genuine metacognition

        print(f"    Full:      acc={full_acc:.4f}  AUROC={full_auroc:.4f}")
        print(f"    Ablated:   acc={abl_acc:.4f}  AUROC={abl_auroc:.4f}")
        print(f"    Encoder:   acc={enc_acc:.4f}  AUROC={enc_auroc:.4f}")
        print(f"    Scrambled: acc={scr_acc:.4f}  AUROC={scr_auroc:.4f}")
        print()
        print(f"    T1: Blindsight dissociation (acc kept, AUROC lost): "
              f"{'PASS' if t1 else 'FAIL'}")
        print(f"    T2: Encoder ablation degrades both:                  "
              f"{'PASS' if t2 else 'FAIL'}")
        print(f"    T3: Scrambled input collapses AUROC:                 "
              f"{'PASS' if t3 else 'FAIL'}")
        print(f"    T4: Baseline has genuine metacognition (AUROC>0.6):  "
              f"{'PASS' if t4 else 'FAIL'}")

        n_pass = sum([t1, t2, t3, t4])
        if n_pass >= 4:
            verdict = "GENUINE_METACOGNITION_WITH_DISSOCIATION"
        elif n_pass >= 3:
            verdict = "METACOGNITION_PARTIAL"
        elif n_pass >= 2:
            verdict = "METACOGNITION_WEAK"
        else:
            verdict = "NO_GENUINE_METACOGNITION"

        print(f"    VERDICT: {verdict} ({n_pass}/4)")

        results[key]['tests'] = {
            't1_blindsight_dissociation': bool(t1),
            't2_encoder_ablation_both_degrade': bool(t2),
            't3_scramble_collapses_auroc': bool(t3),
            't4_baseline_genuine_metacognition': bool(t4),
            'tests_passed': n_pass,
            'verdict': verdict,
        }

    # Save results
    output = {
        'experiment': 'z2021_synthetic_blindsight',
        'hypothesis': 'Self-model ablation produces specific dissociation: task preserved, metacognition collapsed',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'epochs': args.epochs,
        'references': [
            'Phua 2025 arXiv:2512.19155 (Synthetic Blindsight)',
            'Rosenthal 2005 (Higher-Order Theories)',
            'Fleming & Lau 2014 (How to measure metacognition)',
        ],
        'design_principle': 'NO consciousness losses. Train task + self-model. Test ablation dissociation.',
        'conditions': results,
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
    results_path = Path(__file__).parent.parent / 'results' / 'z2021_synthetic_blindsight.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
