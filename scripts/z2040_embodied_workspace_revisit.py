#!/usr/bin/env python3
"""
z2040: Embodied Workspace Revisit

Revisits embodiment using PROVEN test patterns from z2020-z2037 (ablation-
dissociation, workspace necessity, contrastive awareness) applied to
GPU-embodied computation.

Background:
  - z900 series (embodiment experiments) showed MIXED results: z907 "kill shot"
    showed ALL p=1.0 -- high-level conditioning (FiLM) DOES NOT WORK
  - z2020-z2037 showed workspace bottleneck architecture consistently passes
    ablation-dissociation tests
  - Key insight: the WORKSPACE ITSELF should be embodied, not the conditioning.
    Instead of FiLM conditioning where telemetry modulates activations, the
    workspace bottleneck DIMENSIONS should change based on hardware state.

Hypothesis:
  A workspace whose effective dimensionality is modulated by live GPU telemetry
  will show different ablation patterns (and measurably different representations)
  from a fixed-dimensionality workspace.

Architecture (~200K params):
  - Encoder: same CNN as z2037 (Conv2d stack -> 128-dim)
  - Embodied Workspace:
      Full workspace: 128 -> 64-dim
      Telemetry gate: 2-dim telemetry -> 64-dim sigmoid mask
      Active dims = workspace * sigmoid(W_gate @ telemetry + b_gate)
      GPU temp/power SELECTS which workspace dims are active
  - Classifier: 64 -> n_classes

Key difference from FiLM (z907):
  FiLM scales ALL activations by a single factor. Here, telemetry gates
  SPECIFIC dimensions, creating genuinely different subspaces under
  different hardware states.

Three Conditions:
  A: Embodied workspace -- real GPU telemetry gates workspace dims
  B: Fixed workspace -- no gating (standard z2037 architecture)
  C: Random-gated workspace -- random values instead of telemetry

Four Tests:
  T1: Workspace Necessity (ablation-dissociation from z2037)
  T2: Gate Utilization (correlation with GPU temperature)
  T3: Representation Divergence (CKA across thermal states)
  T4: Task Performance (>90% on MNIST composite)

Verdict:
  EMBODIED_WORKSPACE_CONFIRMED if 4/4 pass
  PARTIAL if 2-3
  WEAK if 1
  FAIL if 0

References:
  Baars 1988 -- Global Workspace Theory
  Pearl 2009 -- Causal inference
  Phua 2025 -- Ablation-based consciousness markers
  Luppi et al. 2024 -- Workspace as integration medium

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
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


# ============================================================================
# Dataset
# ============================================================================

class CompositeTask(Dataset):
    """Composite task requiring integration of multiple digit properties."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # (digit > 4) XOR (digit is even) -- requires two properties
        target = int((label > 4) != (label % 2 == 0))
        return img, target


# ============================================================================
# Models
# ============================================================================

class EmbodiedWorkspaceNet(nn.Module):
    """Condition A: Workspace gated by live GPU telemetry."""
    def __init__(self, ws_dim=64, n_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.workspace_proj = nn.Sequential(
            nn.Linear(128, ws_dim),
            nn.LayerNorm(ws_dim),
            nn.ReLU(),
        )
        # Telemetry gate: 2-dim telemetry -> 64-dim sigmoid mask
        self.gate = nn.Linear(2, ws_dim)
        self.classifier = nn.Linear(ws_dim, n_classes)
        self.ws_dim = ws_dim

    def forward(self, x, telemetry_vec=None, ws_override=None):
        h = self.encoder(x).view(x.size(0), -1)
        ws_raw = self.workspace_proj(h)

        if ws_override is not None:
            ws = ws_override
        elif telemetry_vec is not None:
            gate_mask = torch.sigmoid(self.gate(telemetry_vec))
            ws = ws_raw * gate_mask
        else:
            ws = ws_raw

        logits = self.classifier(ws)
        return {
            'logits': logits,
            'workspace': ws,
            'workspace_raw': ws_raw,
            'pre_ws': h,
            'gate_mask': torch.sigmoid(self.gate(telemetry_vec)) if telemetry_vec is not None else None,
        }


class FixedWorkspaceNet(nn.Module):
    """Condition B: Standard workspace with no gating (z2037 pattern)."""
    def __init__(self, ws_dim=64, n_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.workspace = nn.Sequential(
            nn.Linear(128, ws_dim),
            nn.LayerNorm(ws_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(ws_dim, n_classes)
        self.ws_dim = ws_dim

    def forward(self, x, telemetry_vec=None, ws_override=None):
        h = self.encoder(x).view(x.size(0), -1)
        if ws_override is not None:
            ws = ws_override
        else:
            ws = self.workspace(h)
        logits = self.classifier(ws)
        return {
            'logits': logits,
            'workspace': ws,
            'workspace_raw': ws,
            'pre_ws': h,
            'gate_mask': None,
        }


class RandomGatedWorkspaceNet(nn.Module):
    """Condition C: Workspace gated by random values instead of telemetry."""
    def __init__(self, ws_dim=64, n_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.workspace_proj = nn.Sequential(
            nn.Linear(128, ws_dim),
            nn.LayerNorm(ws_dim),
            nn.ReLU(),
        )
        self.gate = nn.Linear(2, ws_dim)
        self.classifier = nn.Linear(ws_dim, n_classes)
        self.ws_dim = ws_dim

    def forward(self, x, telemetry_vec=None, ws_override=None):
        h = self.encoder(x).view(x.size(0), -1)
        ws_raw = self.workspace_proj(h)

        if ws_override is not None:
            ws = ws_override
        else:
            # Always use random gate values, ignoring telemetry
            rand_vec = torch.rand(x.size(0), 2, device=x.device)
            gate_mask = torch.sigmoid(self.gate(rand_vec))
            ws = ws_raw * gate_mask

        logits = self.classifier(ws)
        return {
            'logits': logits,
            'workspace': ws,
            'workspace_raw': ws_raw,
            'pre_ws': h,
            'gate_mask': None,
        }


# ============================================================================
# Telemetry helpers
# ============================================================================

def read_telemetry(sensor):
    """Read GPU telemetry and return normalized 2-dim vector."""
    sample = sensor.read_sample()
    temp = getattr(sample, 'temp_edge_c', 50.0)
    power = getattr(sample, 'power_w', 30.0)
    temp_norm = (temp - 40.0) / 40.0
    power_norm = (power - 10.0) / 50.0
    return temp, power, temp_norm, power_norm


def make_telemetry_tensor(temp_norm, power_norm, batch_size, device):
    """Create batch of 2-dim telemetry vectors."""
    vec = torch.tensor([temp_norm, power_norm], dtype=torch.float32, device=device)
    return vec.unsqueeze(0).expand(batch_size, -1)


# ============================================================================
# Thermal variation protocol
# ============================================================================

def heat_gpu(device, duration_s=5.0):
    """Create GPU heat by doing heavy matmul."""
    mat = torch.randn(4000, 4000, device=device)
    start = time.time()
    while time.time() - start < duration_s:
        mat = mat @ mat.t()
        mat = mat / (mat.norm() + 1e-8)
    del mat
    torch.cuda.synchronize()


def cool_gpu(duration_s=3.0):
    """Let GPU cool."""
    time.sleep(duration_s)


# ============================================================================
# Training
# ============================================================================

def train_model(model, loader, device, sensor, epochs=15, lr=1e-3,
                condition='embodied', thermal_cycling=True):
    """Train model, optionally with thermal cycling."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    is_embodied = (condition == 'embodied')

    for ep in range(1, epochs + 1):
        # Thermal cycling: alternate heat/cool every few epochs
        if thermal_cycling and ep % 4 == 0:
            heat_gpu(device, duration_s=3.0)
        elif thermal_cycling and ep % 4 == 2:
            cool_gpu(duration_s=1.5)

        model.train()
        total_loss, correct, total = 0, 0, 0
        t0 = time.time()

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            B = images.size(0)

            # Read live telemetry
            _, _, temp_norm, power_norm = read_telemetry(sensor)
            telem_vec = make_telemetry_tensor(temp_norm, power_norm, B, device)

            optimizer.zero_grad()

            if is_embodied:
                out = model(images, telemetry_vec=telem_vec)
            else:
                out = model(images, telemetry_vec=telem_vec)

            loss = F.cross_entropy(out['logits'], labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            correct += (out['logits'].argmax(1) == labels).sum().item()
            total += B

        elapsed = time.time() - t0
        if ep % 3 == 0 or ep == 1 or ep == epochs:
            _, _, tn, pn = read_telemetry(sensor)
            print(f"    Epoch {ep:2d}/{epochs}  loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.3f}  temp_n={tn:.2f}  ({elapsed:.1f}s)")


# ============================================================================
# T1: Workspace Necessity (ablation-dissociation)
# ============================================================================

@torch.no_grad()
def collect_ws_stats(model, loader, device, sensor):
    """Collect workspace statistics for frozen ablation."""
    model.eval()
    all_ws = []
    for images, _ in loader:
        images = images.to(device)
        B = images.size(0)
        _, _, tn, pn = read_telemetry(sensor)
        tv = make_telemetry_tensor(tn, pn, B, device)
        out = model(images, telemetry_vec=tv)
        all_ws.append(out['workspace'].cpu())
    return torch.cat(all_ws)


@torch.no_grad()
def evaluate_ablation(model, loader, device, sensor, ws_mode='normal', ws_stats=None):
    """Evaluate model under different workspace ablation conditions."""
    model.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)
        _, _, tn, pn = read_telemetry(sensor)
        tv = make_telemetry_tensor(tn, pn, B, device)

        if ws_mode == 'normal':
            out = model(images, telemetry_vec=tv)
        elif ws_mode == 'zero':
            ws = torch.zeros(B, model.ws_dim, device=device)
            out = model(images, telemetry_vec=tv, ws_override=ws)
        elif ws_mode == 'random':
            ws = torch.randn(B, model.ws_dim, device=device)
            out = model(images, telemetry_vec=tv, ws_override=ws)
        elif ws_mode == 'frozen':
            ws_mean = ws_stats.mean(dim=0).to(device).unsqueeze(0).expand(B, -1)
            out = model(images, telemetry_vec=tv, ws_override=ws_mean)
        else:
            out = model(images, telemetry_vec=tv)

        correct += (out['logits'].argmax(1) == labels).sum().item()
        total += B

    return correct / total


def run_t1_workspace_necessity(model, test_loader, device, sensor, label):
    """T1: Ablate workspace and measure accuracy drops."""
    print(f"\n    T1 Workspace Necessity [{label}]:")
    ws_stats = collect_ws_stats(model, test_loader, device, sensor)

    results = {}
    for mode in ['normal', 'zero', 'random', 'frozen']:
        acc = evaluate_ablation(model, test_loader, device, sensor,
                                ws_mode=mode, ws_stats=ws_stats)
        results[mode] = acc
        print(f"      {mode:>8}: acc={acc:.4f}")

    baseline = results['normal']
    zero_drop = baseline - results['zero']
    random_drop = baseline - results['random']
    frozen_drop = baseline - results['frozen']
    necessity = (zero_drop + random_drop) / 2

    print(f"      drops: zero={zero_drop:+.4f}  random={random_drop:+.4f}  "
          f"frozen={frozen_drop:+.4f}  necessity={necessity:.4f}")

    return {
        'accuracies': results,
        'baseline_acc': baseline,
        'zero_drop': zero_drop,
        'random_drop': random_drop,
        'frozen_drop': frozen_drop,
        'necessity_score': necessity,
    }


# ============================================================================
# T2: Gate Utilization
# ============================================================================

@torch.no_grad()
def run_t2_gate_utilization(model, test_loader, device, sensor, label):
    """T2: Measure how many workspace dims are actively gated."""
    print(f"\n    T2 Gate Utilization [{label}]:")

    if not hasattr(model, 'gate'):
        print(f"      [No gate layer -- skipping for {label}]")
        return {
            'has_gate': False,
            'mean_active_dims': model.ws_dim,
            'gate_variance': 0.0,
            'temp_correlation': 0.0,
        }

    model.eval()
    temps_collected = []
    active_dims_collected = []
    gate_means_collected = []

    # Collect over multiple thermal states
    # Phase 1: Cool state
    cool_gpu(2.0)
    for images, _ in test_loader:
        images = images.to(device)
        B = images.size(0)
        temp, _, tn, pn = read_telemetry(sensor)
        tv = make_telemetry_tensor(tn, pn, B, device)
        out = model(images, telemetry_vec=tv)
        if out['gate_mask'] is not None:
            gate = out['gate_mask']
            active = (gate > 0.5).float().sum(dim=1).mean().item()
            active_dims_collected.append(active)
            gate_means_collected.append(gate.mean(dim=0).cpu().numpy())
            temps_collected.append(temp)
        break  # One batch per thermal state

    # Phase 2: Heat and collect
    heat_gpu(device, duration_s=5.0)
    for images, _ in test_loader:
        images = images.to(device)
        B = images.size(0)
        temp, _, tn, pn = read_telemetry(sensor)
        tv = make_telemetry_tensor(tn, pn, B, device)
        out = model(images, telemetry_vec=tv)
        if out['gate_mask'] is not None:
            gate = out['gate_mask']
            active = (gate > 0.5).float().sum(dim=1).mean().item()
            active_dims_collected.append(active)
            gate_means_collected.append(gate.mean(dim=0).cpu().numpy())
            temps_collected.append(temp)
        break

    # Phase 3: After more heat
    heat_gpu(device, duration_s=5.0)
    for images, _ in test_loader:
        images = images.to(device)
        B = images.size(0)
        temp, _, tn, pn = read_telemetry(sensor)
        tv = make_telemetry_tensor(tn, pn, B, device)
        out = model(images, telemetry_vec=tv)
        if out['gate_mask'] is not None:
            gate = out['gate_mask']
            active = (gate > 0.5).float().sum(dim=1).mean().item()
            active_dims_collected.append(active)
            gate_means_collected.append(gate.mean(dim=0).cpu().numpy())
            temps_collected.append(temp)
        break

    # Phase 4: Cool again
    cool_gpu(3.0)
    for images, _ in test_loader:
        images = images.to(device)
        B = images.size(0)
        temp, _, tn, pn = read_telemetry(sensor)
        tv = make_telemetry_tensor(tn, pn, B, device)
        out = model(images, telemetry_vec=tv)
        if out['gate_mask'] is not None:
            gate = out['gate_mask']
            active = (gate > 0.5).float().sum(dim=1).mean().item()
            active_dims_collected.append(active)
            gate_means_collected.append(gate.mean(dim=0).cpu().numpy())
            temps_collected.append(temp)
        break

    if len(temps_collected) < 2:
        print("      [Insufficient thermal variation]")
        return {
            'has_gate': True,
            'mean_active_dims': 0,
            'gate_variance': 0,
            'temp_correlation': 0,
        }

    temps_arr = np.array(temps_collected)
    active_arr = np.array(active_dims_collected)

    # Correlation between temperature and active dims
    if temps_arr.std() > 0 and active_arr.std() > 0:
        corr = np.corrcoef(temps_arr, active_arr)[0, 1]
    else:
        corr = 0.0

    # Gate variance across thermal states
    gate_stacked = np.stack(gate_means_collected)
    gate_var = gate_stacked.var(axis=0).mean()

    mean_active = float(active_arr.mean())

    print(f"      Temps observed: {temps_arr.tolist()}")
    print(f"      Active dims: {active_arr.tolist()}")
    print(f"      Mean active dims: {mean_active:.1f}/{model.ws_dim}")
    print(f"      Gate variance across states: {gate_var:.6f}")
    print(f"      Temp-active correlation: {corr:.4f}")

    return {
        'has_gate': True,
        'temps': temps_arr.tolist(),
        'active_dims_per_state': active_arr.tolist(),
        'mean_active_dims': mean_active,
        'gate_variance': float(gate_var),
        'temp_correlation': float(corr) if not np.isnan(corr) else 0.0,
    }


# ============================================================================
# T3: Representation Divergence (CKA)
# ============================================================================

def linear_cka(X, Y):
    """Centered Kernel Alignment between two representation matrices.
    X, Y: (n_samples, n_features) numpy arrays.
    Returns CKA similarity in [0, 1]."""
    n = X.shape[0]
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices
    KX = X @ X.T
    KY = Y @ Y.T

    # HSIC
    hsic_xy = np.trace(KX @ KY) / ((n - 1) ** 2)
    hsic_xx = np.trace(KX @ KX) / ((n - 1) ** 2)
    hsic_yy = np.trace(KY @ KY) / ((n - 1) ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 1.0
    return float(hsic_xy / denom)


@torch.no_grad()
def collect_representations_at_temp(model, loader, device, temp_norm, power_norm,
                                     max_samples=500):
    """Collect workspace representations at a specific thermal state."""
    model.eval()
    all_ws = []
    count = 0
    for images, _ in loader:
        images = images.to(device)
        B = images.size(0)
        tv = make_telemetry_tensor(temp_norm, power_norm, B, device)
        out = model(images, telemetry_vec=tv)
        all_ws.append(out['workspace'].cpu().numpy())
        count += B
        if count >= max_samples:
            break
    return np.concatenate(all_ws)[:max_samples]


def run_t3_representation_divergence(model, test_loader, device, sensor, label):
    """T3: Check if representations differ across thermal states."""
    print(f"\n    T3 Representation Divergence [{label}]:")

    # Get current actual temp for baseline
    temp_actual, power_actual, _, _ = read_telemetry(sensor)

    # Simulate 3 thermal states by setting telemetry directly
    states = {
        'cool': ((40.0 - 40.0) / 40.0, (15.0 - 10.0) / 50.0),   # 40C, 15W
        'warm': ((55.0 - 40.0) / 40.0, (35.0 - 10.0) / 50.0),   # 55C, 35W
        'hot':  ((70.0 - 40.0) / 40.0, (60.0 - 10.0) / 50.0),   # 70C, 60W
    }

    reps = {}
    for state_name, (tn, pn) in states.items():
        reps[state_name] = collect_representations_at_temp(
            model, test_loader, device, tn, pn, max_samples=500)
        print(f"      {state_name}: shape={reps[state_name].shape}, "
              f"mean_norm={np.linalg.norm(reps[state_name], axis=1).mean():.4f}")

    # Compute CKA between all pairs
    pairs = [('cool', 'warm'), ('cool', 'hot'), ('warm', 'hot')]
    cka_values = {}
    for a, b in pairs:
        cka = linear_cka(reps[a], reps[b])
        cka_values[f"{a}_vs_{b}"] = cka
        print(f"      CKA({a}, {b}) = {cka:.4f}")

    mean_cka = np.mean(list(cka_values.values()))
    print(f"      Mean CKA: {mean_cka:.4f}")

    return {
        'cka_values': cka_values,
        'mean_cka': float(mean_cka),
        'actual_temp': float(temp_actual),
    }


# ============================================================================
# T4: Task Performance
# ============================================================================

@torch.no_grad()
def run_t4_task_performance(model, test_loader, device, sensor, label):
    """T4: Check that the model achieves >90% accuracy."""
    print(f"\n    T4 Task Performance [{label}]:")
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)
        _, _, tn, pn = read_telemetry(sensor)
        tv = make_telemetry_tensor(tn, pn, B, device)
        out = model(images, telemetry_vec=tv)
        correct += (out['logits'].argmax(1) == labels).sum().item()
        total += B
    acc = correct / total
    print(f"      Accuracy: {acc:.4f}")
    return {'accuracy': acc}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2040] Device: {device}")

    # Initialize telemetry
    sensor = SysfsHwmonTelemetry()
    temp, power, _, _ = read_telemetry(sensor)
    print(f"[z2040] Initial GPU temp: {temp:.1f}C, power: {power:.1f}W")

    from torchvision import datasets, transforms
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    tf = transforms.ToTensor()
    train_base = datasets.MNIST(str(data_dir), train=True, download=True, transform=tf)
    test_base = datasets.MNIST(str(data_dir), train=False, download=True, transform=tf)

    train_data = CompositeTask(train_base)
    test_data = CompositeTask(test_base)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    print(f"\n{'='*70}")
    print(f"  z2040: Embodied Workspace Revisit")
    print(f"  Workspace gated by GPU telemetry vs fixed vs random")
    print(f"  Key insight: embody the WORKSPACE, not the conditioning")
    print(f"  NO consciousness losses -- all tests are CAUSAL")
    print(f"{'='*70}")

    # ----------------------------------------------------------------
    # Train three conditions
    # ----------------------------------------------------------------
    conditions = {}

    # Condition A: Embodied workspace (real telemetry gating)
    print(f"\n{'='*70}")
    print(f"  CONDITION A: Embodied Workspace (GPU telemetry gates dims)")
    print(f"{'='*70}")
    model_a = EmbodiedWorkspaceNet(ws_dim=64, n_classes=2).to(device)
    n_params_a = sum(p.numel() for p in model_a.parameters())
    print(f"  Parameters: {n_params_a:,}")
    train_model(model_a, train_loader, device, sensor, epochs=args.epochs,
                condition='embodied', thermal_cycling=True)

    # Condition B: Fixed workspace (no gating)
    print(f"\n{'='*70}")
    print(f"  CONDITION B: Fixed Workspace (standard z2037 pattern)")
    print(f"{'='*70}")
    model_b = FixedWorkspaceNet(ws_dim=64, n_classes=2).to(device)
    n_params_b = sum(p.numel() for p in model_b.parameters())
    print(f"  Parameters: {n_params_b:,}")
    train_model(model_b, train_loader, device, sensor, epochs=args.epochs,
                condition='fixed', thermal_cycling=True)

    # Condition C: Random-gated workspace
    print(f"\n{'='*70}")
    print(f"  CONDITION C: Random-Gated Workspace (random instead of telemetry)")
    print(f"{'='*70}")
    model_c = RandomGatedWorkspaceNet(ws_dim=64, n_classes=2).to(device)
    n_params_c = sum(p.numel() for p in model_c.parameters())
    print(f"  Parameters: {n_params_c:,}")
    train_model(model_c, train_loader, device, sensor, epochs=args.epochs,
                condition='random', thermal_cycling=True)

    # ----------------------------------------------------------------
    # Run tests
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RUNNING TESTS")
    print(f"{'='*70}")

    models = {'A_embodied': model_a, 'B_fixed': model_b, 'C_random': model_c}

    # T1: Workspace Necessity
    print(f"\n  === T1: Workspace Necessity ===")
    t1_results = {}
    for label, model in models.items():
        t1_results[label] = run_t1_workspace_necessity(
            model, test_loader, device, sensor, label)

    # T2: Gate Utilization
    print(f"\n  === T2: Gate Utilization ===")
    t2_results = {}
    for label, model in models.items():
        t2_results[label] = run_t2_gate_utilization(
            model, test_loader, device, sensor, label)

    # T3: Representation Divergence
    print(f"\n  === T3: Representation Divergence ===")
    t3_results = {}
    for label, model in models.items():
        t3_results[label] = run_t3_representation_divergence(
            model, test_loader, device, sensor, label)

    # T4: Task Performance
    print(f"\n  === T4: Task Performance ===")
    t4_results = {}
    for label, model in models.items():
        t4_results[label] = run_t4_task_performance(
            model, test_loader, device, sensor, label)

    # ----------------------------------------------------------------
    # Final Analysis
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS")
    print(f"{'='*70}")

    # T1 verdict: All conditions should show workspace necessity (>5% drop)
    # AND embodied should show different drop pattern from fixed
    t1_a_necessary = t1_results['A_embodied']['necessity_score'] > 0.05
    t1_b_necessary = t1_results['B_fixed']['necessity_score'] > 0.05
    t1_c_necessary = t1_results['C_random']['necessity_score'] > 0.05
    # Embodied should have different drop magnitude from fixed
    t1_drop_diff = abs(t1_results['A_embodied']['necessity_score'] -
                       t1_results['B_fixed']['necessity_score'])
    t1_pass = t1_a_necessary and t1_b_necessary and t1_drop_diff > 0.01
    print(f"\n  T1: Workspace Necessity")
    print(f"      A necessity: {t1_results['A_embodied']['necessity_score']:.4f} "
          f"({'PASS' if t1_a_necessary else 'FAIL'} >0.05)")
    print(f"      B necessity: {t1_results['B_fixed']['necessity_score']:.4f} "
          f"({'PASS' if t1_b_necessary else 'FAIL'} >0.05)")
    print(f"      C necessity: {t1_results['C_random']['necessity_score']:.4f} "
          f"({'PASS' if t1_c_necessary else 'FAIL'} >0.05)")
    print(f"      A vs B drop difference: {t1_drop_diff:.4f} "
          f"({'PASS' if t1_drop_diff > 0.01 else 'FAIL'} >0.01)")
    print(f"      T1 OVERALL: {'PASS' if t1_pass else 'FAIL'}")

    # T2 verdict: Embodied gate should correlate with temp;
    # random should NOT correlate
    t2_a = t2_results['A_embodied']
    t2_c = t2_results['C_random']
    t2_a_var = t2_a.get('gate_variance', 0)
    t2_pass = t2_a_var > 1e-4  # Gate values actually change with thermal state
    print(f"\n  T2: Gate Utilization")
    print(f"      A gate variance: {t2_a_var:.6f} "
          f"({'PASS' if t2_a_var > 1e-4 else 'FAIL'} >1e-4)")
    if t2_a.get('has_gate', False):
        print(f"      A temp-active correlation: {t2_a.get('temp_correlation', 0):.4f}")
    print(f"      T2 OVERALL: {'PASS' if t2_pass else 'FAIL'}")

    # T3 verdict: Embodied should show CKA < 0.9 (representations differ
    # across temps); Fixed should show CKA > 0.95 (same across temps)
    t3_a_cka = t3_results['A_embodied']['mean_cka']
    t3_b_cka = t3_results['B_fixed']['mean_cka']
    t3_embodied_diverges = t3_a_cka < 0.9
    t3_fixed_stable = t3_b_cka > 0.95
    t3_pass = t3_embodied_diverges and t3_fixed_stable
    print(f"\n  T3: Representation Divergence")
    print(f"      A mean CKA: {t3_a_cka:.4f} "
          f"({'PASS' if t3_embodied_diverges else 'FAIL'} <0.90, want divergent)")
    print(f"      B mean CKA: {t3_b_cka:.4f} "
          f"({'PASS' if t3_fixed_stable else 'FAIL'} >0.95, want stable)")
    print(f"      T3 OVERALL: {'PASS' if t3_pass else 'FAIL'}")

    # T4 verdict: All three should achieve >90% accuracy
    t4_a_acc = t4_results['A_embodied']['accuracy']
    t4_b_acc = t4_results['B_fixed']['accuracy']
    t4_c_acc = t4_results['C_random']['accuracy']
    t4_pass = t4_a_acc > 0.90 and t4_b_acc > 0.90 and t4_c_acc > 0.90
    print(f"\n  T4: Task Performance (all >90%)")
    print(f"      A accuracy: {t4_a_acc:.4f} ({'PASS' if t4_a_acc > 0.90 else 'FAIL'})")
    print(f"      B accuracy: {t4_b_acc:.4f} ({'PASS' if t4_b_acc > 0.90 else 'FAIL'})")
    print(f"      C accuracy: {t4_c_acc:.4f} ({'PASS' if t4_c_acc > 0.90 else 'FAIL'})")
    print(f"      T4 OVERALL: {'PASS' if t4_pass else 'FAIL'}")

    # Overall verdict
    tests = [t1_pass, t2_pass, t3_pass, t4_pass]
    n_pass = sum(tests)

    verdicts = {
        4: "EMBODIED_WORKSPACE_CONFIRMED",
        3: "PARTIAL",
        2: "PARTIAL",
        1: "WEAK",
        0: "FAIL",
    }
    verdict = verdicts[n_pass]

    print(f"\n{'='*70}")
    print(f"  VERDICT: {verdict} ({n_pass}/4)")
    print(f"  T1 workspace necessity:      {'PASS' if t1_pass else 'FAIL'}")
    print(f"  T2 gate utilization:         {'PASS' if t2_pass else 'FAIL'}")
    print(f"  T3 representation diverge:   {'PASS' if t3_pass else 'FAIL'}")
    print(f"  T4 task performance:         {'PASS' if t4_pass else 'FAIL'}")
    print(f"{'='*70}")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
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
        return obj

    output = {
        'experiment': 'z2040_embodied_workspace_revisit',
        'hypothesis': ('A workspace whose effective dimensionality is modulated by '
                       'live GPU telemetry will show different ablation patterns '
                       'and measurably different representations from a fixed '
                       'workspace'),
        'key_insight': ('Embody the WORKSPACE, not the conditioning. FiLM scales '
                        'ALL activations by a single factor; here, telemetry gates '
                        'SPECIFIC dimensions, creating genuinely different subspaces '
                        'under different hardware states.'),
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'ws_dim': 64,
            'n_classes': 2,
        },
        'model_params': {
            'A_embodied': n_params_a,
            'B_fixed': n_params_b,
            'C_random': n_params_c,
        },
        'references': [
            'Baars 1988 (Global Workspace Theory)',
            'Pearl 2009 (Causal inference)',
            'Phua 2025 (Ablation-based consciousness markers)',
            'Luppi et al. 2024 (Workspace as integration medium)',
        ],
        't1_workspace_necessity': t1_results,
        't2_gate_utilization': t2_results,
        't3_representation_divergence': t3_results,
        't4_task_performance': t4_results,
        'tests': {
            't1': bool(t1_pass),
            't2': bool(t2_pass),
            't3': bool(t3_pass),
            't4': bool(t4_pass),
        },
        'tests_passed': n_pass,
        'verdict': verdict,
    }

    rp = Path(__file__).parent.parent / 'results' / 'z2040_embodied_workspace_revisit.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
