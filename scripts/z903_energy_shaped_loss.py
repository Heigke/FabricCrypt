#!/usr/bin/env python3
"""
z903_energy_shaped_loss.py - Energy-Shaped Loss Landscape Experiment

Hypothesis: When the loss function is modulated by measured energy
    L = L_task * (1 + beta * max(0, E - E_budget) / E_budget)
the model converges to structurally different solutions on the
accuracy-energy Pareto frontier.

Key novelty: Energy directly modifies gradient magnitude. When E > budget,
gradients are amplified. This is NOT reinforcement learning -- it is direct
gradient-based optimization where the landscape shape depends on hardware state.

Architecture: MNIST classifier 784->256->128->10 with ReLU (~200K params)

Controls:
    A: beta=0 (standard training, fixed loss) -- run once
    B: Energy-shaped with SIMULATED energy (random uniform, same mean/std as real)
    C: Energy-shaped with PREVIOUS batch energy (decorrelated from current batch)

Sweep beta in {0.0, 0.1, 0.5, 1.0, 2.0} with per-batch energy from EnergyMeter.
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Telemetry: real sysfs or mock fallback
# ---------------------------------------------------------------------------
USE_REAL_TELEMETRY = False
telemetry_instance = None

try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter
    # Probe whether hwmon paths exist
    _probe = SysfsHwmonTelemetry(sample_rate_hz=100)
    _ = _probe.read_sample()
    telemetry_instance = _probe
    USE_REAL_TELEMETRY = True
    print("[telemetry] Using real SysfsHwmonTelemetry (sysfs hwmon)")
except Exception as e:
    print(f"[telemetry] sysfs hwmon unavailable ({e}); using mock energy")


class MockEnergyMeter:
    """Fallback energy meter producing plausible random values."""

    def __init__(self, mean_j: float = 0.05, std_j: float = 0.015):
        self.mean_j = mean_j
        self.std_j = std_j
        self.energy_j: float = 0.0

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *args):
        dt = time.time() - self._t0
        # Correlated with wall-clock but noisy
        self.energy_j = max(0.001, np.random.normal(self.mean_j + dt * 0.5,
                                                      self.std_j))


def make_energy_meter():
    """Return an energy meter (real or mock) for one batch measurement."""
    if USE_REAL_TELEMETRY and telemetry_instance is not None:
        return EnergyMeter(telemetry_instance, sync_cuda=True)
    return MockEnergyMeter()


# ---------------------------------------------------------------------------
# MNIST model (784 -> 256 -> 128 -> 10, ReLU, ~200K params)
# ---------------------------------------------------------------------------
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def weight_norm(model: nn.Module) -> float:
    """L2 norm of all parameters concatenated."""
    return torch.norm(
        torch.cat([p.detach().flatten() for p in model.parameters()])
    ).item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return accuracy (fraction) on a data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def measure_inference_energy(model: nn.Module, loader: DataLoader,
                              device: torch.device, n_batches: int = 100):
    """Measure per-batch inference energy for n_batches. Returns list of joules."""
    model.eval()
    energies = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(device)
            meter = make_energy_meter()
            with meter:
                _ = model(images)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            energies.append(meter.energy_j)
    return energies


# ---------------------------------------------------------------------------
# Core training loop for one (beta, condition) pair
# ---------------------------------------------------------------------------
def train_run(beta: float, condition: str, epochs: int, batch_size: int,
              device: torch.device, train_loader: DataLoader,
              test_loader: DataLoader, seed: int = 42):
    """
    Train a fresh model with energy-shaped loss.

    condition:
        'A' = standard training (beta forced to 0)
        'B' = simulated energy (random uniform matching real stats)
        'C' = previous-batch energy (decorrelated)
        'real' = actual measured energy for the current batch
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    model = MNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    effective_beta = 0.0 if condition == 'A' else beta

    # -- Phase 1: calibrate energy budget from first epoch ----------------
    print(f"    [calibrate] Measuring E_budget over epoch 0 ...")
    energy_samples_calib = []
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        meter = make_energy_meter()
        with meter:
            output = model(images)
            loss_task = F.cross_entropy(output, labels)
        energy_samples_calib.append(meter.energy_j)
        # Still do a training step so calibration epoch isn't wasted
        loss_task.backward()
        optimizer.step()
        optimizer.zero_grad()

    e_budget = float(np.mean(energy_samples_calib))
    e_std = float(np.std(energy_samples_calib))
    print(f"    [calibrate] E_budget = {e_budget:.5f} J  (std={e_std:.5f}, "
          f"n={len(energy_samples_calib)})")

    # -- Phase 2: main training with energy-shaped loss --------------------
    weight_norms = []
    loss_history = []
    energy_history = []
    penalty_history = []
    prev_batch_energy = e_budget  # initialise for condition C

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # --- Measure energy for this batch ---
            meter = make_energy_meter()
            with meter:
                output = model(images)
                loss_task = F.cross_entropy(output, labels)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
            real_energy = meter.energy_j

            # --- Pick energy value based on condition ---
            if condition == 'A':
                energy_for_penalty = 0.0  # doesn't matter, beta=0
            elif condition == 'B':
                # Simulated: random uniform with same mean/std as real
                energy_for_penalty = float(np.random.uniform(
                    max(0.001, e_budget - e_std * np.sqrt(3)),
                    e_budget + e_std * np.sqrt(3)
                ))
            elif condition == 'C':
                energy_for_penalty = prev_batch_energy
            else:  # 'real'
                energy_for_penalty = real_energy

            prev_batch_energy = real_energy  # update for next iteration

            # --- Energy-shaped loss ---
            energy_penalty = max(0.0, energy_for_penalty - e_budget) / e_budget
            loss = loss_task * (1.0 + effective_beta * energy_penalty)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = output.max(1)
            epoch_correct += predicted.eq(labels).sum().item()
            epoch_total += labels.size(0)

            energy_history.append(real_energy)
            penalty_history.append(energy_penalty)

            global_step += 1
            if global_step % 100 == 0:
                wn = weight_norm(model)
                weight_norms.append({'step': global_step, 'weight_norm': wn})
                loss_history.append({'step': global_step, 'loss': loss.item()})

        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        print(f"    epoch {epoch:2d}/{epochs}  loss={epoch_loss / len(train_loader):.4f}  "
              f"train_acc={train_acc:.4f}")

    # -- Evaluate ----------------------------------------------------------
    test_acc = evaluate(model, test_loader, device)
    inference_energies = measure_inference_energy(model, test_loader, device,
                                                  n_batches=100)

    return {
        'test_accuracy': test_acc,
        'weight_norms': weight_norms,
        'loss_history': loss_history,
        'energy_history_summary': {
            'mean': float(np.mean(energy_history)),
            'std': float(np.std(energy_history)),
            'min': float(np.min(energy_history)),
            'max': float(np.max(energy_history)),
        },
        'penalty_summary': {
            'mean': float(np.mean(penalty_history)),
            'std': float(np.std(penalty_history)),
            'frac_active': float(np.mean([p > 0 for p in penalty_history])),
        },
        'e_budget': e_budget,
        'inference_energy': {
            'mean_j': float(np.mean(inference_energies)),
            'std_j': float(np.std(inference_energies)),
            'median_j': float(np.median(inference_energies)),
            'values': inference_energies[:20],  # first 20 for plotting
        },
        'final_weight_norm': weight_norm(model),
        'model_state': model.state_dict(),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="z903: Energy-shaped loss landscape experiment")
    parser.add_argument('--epochs', type=int, default=15,
                        help='Training epochs per run (default: 15)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--betas', type=str, default='0.0,0.1,0.5,1.0,2.0',
                        help='Comma-separated beta values (default: 0.0,0.1,0.5,1.0,2.0)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu (default: auto)')
    args = parser.parse_args()

    betas = [float(b) for b in args.betas.split(',')]

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 72)
    print("z903  Energy-Shaped Loss Landscape Experiment")
    print("=" * 72)
    print(f"Device          : {device}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Betas           : {betas}")
    print(f"Real telemetry  : {USE_REAL_TELEMETRY}")
    print()

    # -- Data --------------------------------------------------------------
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(str(data_dir), train=True, download=True,
                                    transform=transform)
    test_dataset = datasets.MNIST(str(data_dir), train=False, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    model_tmp = MNISTClassifier()
    print(f"Model params    : {model_tmp.param_count():,}")
    del model_tmp
    print()

    # -- Results container --------------------------------------------------
    all_results = {
        'experiment': 'z903_energy_shaped_loss',
        'hypothesis': ('Energy-shaped loss L = L_task * (1 + beta * max(0, E-E_budget)/E_budget) '
                       'produces structurally different solutions on the accuracy-energy Pareto frontier.'),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'betas': betas,
            'device': str(device),
            'real_telemetry': USE_REAL_TELEMETRY,
        },
        'runs': {},
    }

    ckpt_dir = Path(__file__).parent.parent / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    # ======================================================================
    # Condition A: beta=0 baseline (run once)
    # ======================================================================
    print("-" * 72)
    print("Condition A  |  beta=0.0  |  Standard training (baseline)")
    print("-" * 72)
    result_a = train_run(
        beta=0.0, condition='A', epochs=args.epochs,
        batch_size=args.batch_size, device=device,
        train_loader=train_loader, test_loader=test_loader,
    )
    state_a = result_a.pop('model_state')
    torch.save(state_a, ckpt_dir / 'z903_beta_0.0_A.pt')
    all_results['runs']['beta_0.0_A'] = result_a
    print(f"  => test accuracy: {result_a['test_accuracy']:.4f}  "
          f"inference energy: {result_a['inference_energy']['mean_j']:.5f} J")
    print()

    # ======================================================================
    # Conditions B, C, real  for each beta > 0
    # ======================================================================
    for beta in betas:
        if beta == 0.0:
            continue  # already covered by condition A

        for cond_label, cond_code in [('B_simulated', 'B'),
                                       ('C_previous', 'C'),
                                       ('real', 'real')]:
            run_key = f'beta_{beta}_{cond_label}'
            print("-" * 72)
            print(f"Condition {cond_label}  |  beta={beta}  |  "
                  f"{'Simulated energy' if cond_code == 'B' else 'Previous-batch energy' if cond_code == 'C' else 'Real energy'}")
            print("-" * 72)

            result = train_run(
                beta=beta, condition=cond_code, epochs=args.epochs,
                batch_size=args.batch_size, device=device,
                train_loader=train_loader, test_loader=test_loader,
            )
            state = result.pop('model_state')
            torch.save(state, ckpt_dir / f'z903_beta_{beta}_{cond_label}.pt')
            all_results['runs'][run_key] = result
            print(f"  => test accuracy: {result['test_accuracy']:.4f}  "
                  f"inference energy: {result['inference_energy']['mean_j']:.5f} J")
            print()

    # ======================================================================
    # Pareto summary table
    # ======================================================================
    print("=" * 72)
    print("PARETO TABLE: Accuracy vs Average Inference Energy")
    print("=" * 72)
    print(f"{'Run':<30s} {'Accuracy':>10s} {'Inf Energy (J)':>15s} "
          f"{'Weight Norm':>12s} {'Penalty Frac':>13s}")
    print("-" * 82)

    pareto_rows = []
    for run_key, result in all_results['runs'].items():
        row = {
            'run': run_key,
            'accuracy': result['test_accuracy'],
            'inference_energy_mean_j': result['inference_energy']['mean_j'],
            'final_weight_norm': result['final_weight_norm'],
            'penalty_frac_active': result['penalty_summary']['frac_active'],
        }
        pareto_rows.append(row)
        print(f"{run_key:<30s} {row['accuracy']:>10.4f} "
              f"{row['inference_energy_mean_j']:>15.5f} "
              f"{row['final_weight_norm']:>12.2f} "
              f"{row['penalty_frac_active']:>13.3f}")

    all_results['pareto_table'] = pareto_rows

    # ======================================================================
    # Weight norm trajectory comparison
    # ======================================================================
    print()
    print("=" * 72)
    print("WEIGHT NORM TRAJECTORIES (sampled every 100 steps)")
    print("=" * 72)
    for run_key, result in all_results['runs'].items():
        norms = result.get('weight_norms', [])
        if norms:
            first_wn = norms[0]['weight_norm']
            last_wn = norms[-1]['weight_norm']
            print(f"  {run_key:<30s}  start={first_wn:.2f}  end={last_wn:.2f}  "
                  f"delta={last_wn - first_wn:+.2f}")

    # ======================================================================
    # Save results
    # ======================================================================
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / 'z903_energy_shaped_loss.json'

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print()
    print(f"Results saved to {results_path}")
    print(f"Checkpoints saved to {ckpt_dir}/z903_beta_*.pt")
    print()

    # ======================================================================
    # Quick verdict
    # ======================================================================
    baseline_acc = all_results['runs']['beta_0.0_A']['test_accuracy']
    baseline_energy = all_results['runs']['beta_0.0_A']['inference_energy']['mean_j']
    baseline_wnorm = all_results['runs']['beta_0.0_A']['final_weight_norm']

    print("=" * 72)
    print("VERDICT vs BASELINE (beta=0.0_A)")
    print("=" * 72)
    print(f"  Baseline accuracy     : {baseline_acc:.4f}")
    print(f"  Baseline inf energy   : {baseline_energy:.5f} J")
    print(f"  Baseline weight norm  : {baseline_wnorm:.2f}")
    print()

    found_pareto_improvement = False
    for run_key, result in all_results['runs'].items():
        if run_key == 'beta_0.0_A':
            continue
        acc = result['test_accuracy']
        eng = result['inference_energy']['mean_j']
        wn = result['final_weight_norm']
        acc_delta = acc - baseline_acc
        eng_delta = eng - baseline_energy
        wn_delta = wn - baseline_wnorm
        dominates = (acc >= baseline_acc and eng < baseline_energy)
        if dominates:
            found_pareto_improvement = True
        tag = " << PARETO-DOMINATES BASELINE" if dominates else ""
        print(f"  {run_key:<30s}  acc={acc_delta:+.4f}  energy={eng_delta:+.5f}  "
              f"wnorm={wn_delta:+.2f}{tag}")

    print()
    if found_pareto_improvement:
        print("  HYPOTHESIS SUPPORTED: At least one energy-shaped run "
              "Pareto-dominates the baseline.")
    else:
        print("  HYPOTHESIS NOT YET SUPPORTED: No energy-shaped run "
              "Pareto-dominates the baseline on this run.")
        print("  (Structural differences in weight norms may still indicate "
              "different solution geometry.)")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
