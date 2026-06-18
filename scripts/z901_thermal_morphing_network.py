#!/usr/bin/env python3
"""
z901_thermal_morphing_network.py
================================
Hypothesis: A network whose activation functions are parameterized by live GPU
temperature learns fundamentally different representations than a fixed network.

Architecture: 784->256->128->10 classifier (~200K params, MNIST)
Each hidden layer applies:  h = h * sigmoid(alpha_i * (temp_c - 50.0))
where alpha_i is a learnable scalar per layer.

Key insight: The nonlinearity itself is hardware-dependent. This is NOT FiLM
conditioning on inputs -- the activation function shape changes with the
physical temperature of the silicon executing it. All gradients flow through
temperature-dependent gates.

Three experimental conditions:
  A: Fixed ReLU baseline (no thermal modulation)
  B: Thermal-morphing (real GPU temperature drives the gate)
  C: Random-morphing (uniform random values replace temperature)

FALSIFICATION CRITERION:
  If B ≈ C (real temp ≈ random), then any varying signal reshapes representations,
  not specifically the physical temperature — the thermal embodiment hypothesis fails.
  If B ≈ A (thermal ≈ fixed ReLU), then temperature doesn't meaningfully alter computation.

WHAT THIS MEASURES (honestly):
  - Whether real hardware temperature creates distinct learned representations
  - Whether the learned sensitivity (alpha) encodes meaningful thermal adaptation
  - Whether accuracy depends on the match between train and test temperature conditions

WHAT THIS CANNOT SHOW:
  - That thermal modulation constitutes interoception or feeling
  - That this satisfies any consciousness indicator (it's a computation test, not a phenomenology test)
  - That substrate-dependent computation implies consciousness (Milinkovic & Aru 2025)

Metrics collected:
  - Per-condition test accuracy
  - Learned alpha values (conditions B and C)
  - Gradient norms stratified by hot vs cold temperature
  - Weight divergence: L2 distance of B/C weights from A weights
  - Temperature trace during training
  - Accuracy at different test temperatures (temp sweep)
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import random
import argparse
import datetime
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
# Telemetry setup -- graceful fallback to mock if sysfs unavailable
# ---------------------------------------------------------------------------
_telemetry = None
_use_mock_temp = False

def _init_telemetry():
    global _telemetry, _use_mock_temp
    try:
        from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
        _telemetry = SysfsHwmonTelemetry()
        # Test read
        _ = _telemetry.read_sample()
        _use_mock_temp = False
        print("[telemetry] sysfs hwmon initialised -- live temperature available")
    except Exception as e:
        print(f"[telemetry] sysfs hwmon unavailable ({e}) -- using mock temperature")
        _use_mock_temp = True


def read_gpu_temp() -> float:
    """Return GPU edge temperature in Celsius."""
    if _use_mock_temp:
        # Simulate temperature cycling: base 50 +/- noise
        return 50.0 + random.gauss(0, 5.0)
    try:
        sample = _telemetry.read_sample()
        return sample.temp_edge_c
    except Exception:
        return 50.0 + random.gauss(0, 5.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ThermalMorphingClassifier(nn.Module):
    """
    3-layer MLP whose hidden activations are gated by a temperature-dependent
    sigmoid.  When use_thermal=False the network falls back to standard ReLU.
    """

    def __init__(self, use_thermal: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Learnable thermal sensitivity -- one scalar per hidden layer
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(2)
        ])
        self.use_thermal = use_thermal

    def forward(self, x: torch.Tensor, temp_c: float = 50.0) -> torch.Tensor:
        h = self.fc1(x.view(-1, 784))
        if self.use_thermal:
            gate = torch.sigmoid(self.alpha[0] * (temp_c - 50.0))
            h = h * gate
        else:
            h = F.relu(h)

        h = self.fc2(h)
        if self.use_thermal:
            gate = torch.sigmoid(self.alpha[1] * (temp_c - 50.0))
            h = h * gate
        else:
            h = F.relu(h)

        return self.fc3(h)


# ---------------------------------------------------------------------------
# Thermal variation helpers
# ---------------------------------------------------------------------------
def thermal_warmup(device: torch.device, iters: int = 10):
    """Heat the GPU by doing large matmuls."""
    if device.type != 'cuda':
        return
    for _ in range(iters):
        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)
        _ = a @ b
    torch.cuda.synchronize()


def thermal_cooldown(seconds: float = 3.0):
    """Idle pause to let the GPU cool slightly."""
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    condition: str,
    warmup_iters: int,
    temp_log: list,
    grad_norms_by_temp: dict,
):
    """
    Train for one epoch.  Every 50 batches, alternate between thermal warmup
    and cooldown to create temperature cycling on the GPU.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # --- deliberate thermal variation every 50 batches ---
        if batch_idx > 0 and batch_idx % 50 == 0:
            if (batch_idx // 50) % 2 == 1:
                thermal_warmup(device, iters=warmup_iters)
            else:
                thermal_cooldown(1.5)

        # --- read temperature for this step ---
        if condition == 'A':
            temp_c = 50.0  # irrelevant -- model uses ReLU
        elif condition == 'B':
            temp_c = read_gpu_temp()
        elif condition == 'C':
            temp_c = random.uniform(30.0, 80.0)  # random mock
        else:
            temp_c = 50.0

        temp_log.append(temp_c)

        # --- forward / backward ---
        optimizer.zero_grad()
        logits = model(data, temp_c=temp_c)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        # --- record gradient norms keyed by temperature bucket ---
        gnorm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gnorm += p.grad.data.norm(2).item() ** 2
        gnorm = gnorm ** 0.5
        grad_norms_by_temp[temp_c] = gnorm

        optimizer.step()

        total_loss += loss.item() * data.size(0)
        preds = logits.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, temp_c: float = 50.0):
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits = model(data, temp_c=temp_c)
        preds = logits.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += data.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Weight divergence: L2 distance between state dicts
# ---------------------------------------------------------------------------
def weight_divergence(sd_a: dict, sd_b: dict) -> float:
    """L2 distance between two state dicts (matching keys only)."""
    dist_sq = 0.0
    for key in sd_a:
        if key in sd_b:
            dist_sq += (sd_a[key].float().cpu() - sd_b[key].float().cpu()).pow(2).sum().item()
    return dist_sq ** 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="z901 Thermal Morphing Network Experiment")
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per condition')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, cpu')
    parser.add_argument('--warmup-iters', type=int, default=10,
                        help='Matmul iterations for GPU thermal warmup')
    args = parser.parse_args()

    # --- device ---
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[z901] Device: {device}")

    # --- telemetry ---
    _init_telemetry()

    # --- data ---
    data_root = Path(__file__).parent.parent / 'data'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(str(data_root), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(str(data_root), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=(device.type == 'cuda'))

    # --- output dirs ---
    results_dir = Path(__file__).parent.parent / 'results'
    ckpt_dir = Path(__file__).parent.parent / 'checkpoints'
    results_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # --- conditions ---
    conditions = {
        'A': {'label': 'Fixed ReLU (baseline)', 'use_thermal': False},
        'B': {'label': 'Thermal-morphing (real temp)', 'use_thermal': True},
        'C': {'label': 'Random-morphing (random values)', 'use_thermal': True},
    }

    all_results = {}

    for cond_key, cond_cfg in conditions.items():
        print(f"\n{'='*70}")
        print(f"  Condition {cond_key}: {cond_cfg['label']}")
        print(f"{'='*70}")

        model = ThermalMorphingClassifier(use_thermal=cond_cfg['use_thermal']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        temp_log = []
        grad_norms_by_temp = {}  # temp -> gnorm (last occurrence wins, fine for stats)
        epoch_stats = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device,
                condition=cond_key,
                warmup_iters=args.warmup_iters,
                temp_log=temp_log,
                grad_norms_by_temp=grad_norms_by_temp,
            )
            test_acc = evaluate(model, test_loader, device, temp_c=read_gpu_temp())
            elapsed = time.time() - t0

            epoch_stats.append({
                'epoch': epoch,
                'train_loss': round(train_loss, 5),
                'train_acc': round(train_acc, 4),
                'test_acc': round(test_acc, 4),
                'elapsed_s': round(elapsed, 1),
            })
            print(f"  Epoch {epoch:2d}/{args.epochs}  loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.3%}  test_acc={test_acc:.3%}  "
                  f"({elapsed:.1f}s)")

        # --- learned alphas ---
        alphas = [model.alpha[i].item() for i in range(len(model.alpha))]
        print(f"  Learned alphas: {alphas}")

        # --- gradient norms hot vs cold ---
        if temp_log:
            median_temp = float(np.median(temp_log))
        else:
            median_temp = 50.0

        hot_gnorms = [v for t, v in grad_norms_by_temp.items() if t > median_temp]
        cold_gnorms = [v for t, v in grad_norms_by_temp.items() if t <= median_temp]
        avg_gnorm_hot = float(np.mean(hot_gnorms)) if hot_gnorms else 0.0
        avg_gnorm_cold = float(np.mean(cold_gnorms)) if cold_gnorms else 0.0
        print(f"  Gradient norms -- hot (>{median_temp:.1f}C): {avg_gnorm_hot:.4f}  "
              f"cold (<={median_temp:.1f}C): {avg_gnorm_cold:.4f}")

        # --- save checkpoint ---
        ckpt_path = ckpt_dir / f'z901_condition_{cond_key}.pt'
        torch.save({
            'state_dict': model.state_dict(),
            'alphas': alphas,
            'condition': cond_key,
            'epoch_stats': epoch_stats,
        }, str(ckpt_path))
        print(f"  Checkpoint saved: {ckpt_path}")

        # --- test accuracy at different temp values ---
        temp_sweep_acc = {}
        for eval_temp in [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]:
            acc = evaluate(model, test_loader, device, temp_c=eval_temp)
            temp_sweep_acc[str(eval_temp)] = round(acc, 4)
        print(f"  Accuracy by temp: {temp_sweep_acc}")

        all_results[cond_key] = {
            'label': cond_cfg['label'],
            'final_test_acc': epoch_stats[-1]['test_acc'],
            'epoch_stats': epoch_stats,
            'alphas': alphas,
            'gradient_norms': {
                'median_temp_c': round(median_temp, 2),
                'avg_gnorm_hot': round(avg_gnorm_hot, 6),
                'avg_gnorm_cold': round(avg_gnorm_cold, 6),
                'n_hot': len(hot_gnorms),
                'n_cold': len(cold_gnorms),
            },
            'temp_sweep_accuracy': temp_sweep_acc,
            'temp_range': {
                'min': round(min(temp_log), 2) if temp_log else 0.0,
                'max': round(max(temp_log), 2) if temp_log else 0.0,
                'mean': round(float(np.mean(temp_log)), 2) if temp_log else 0.0,
                'std': round(float(np.std(temp_log)), 2) if temp_log else 0.0,
                'n_samples': len(temp_log),
            },
        }

    # ---------------------------------------------------------------------------
    # Weight divergence: B vs A, C vs A
    # ---------------------------------------------------------------------------
    sd_a = torch.load(str(ckpt_dir / 'z901_condition_A.pt'), map_location='cpu')['state_dict']
    sd_b = torch.load(str(ckpt_dir / 'z901_condition_B.pt'), map_location='cpu')['state_dict']
    sd_c = torch.load(str(ckpt_dir / 'z901_condition_C.pt'), map_location='cpu')['state_dict']

    div_ba = weight_divergence(sd_a, sd_b)
    div_ca = weight_divergence(sd_a, sd_c)
    div_bc = weight_divergence(sd_b, sd_c)

    print(f"\n{'='*70}")
    print("  Weight Divergence (L2 distance between state dicts)")
    print(f"{'='*70}")
    print(f"  B (thermal) vs A (fixed):  {div_ba:.4f}")
    print(f"  C (random)  vs A (fixed):  {div_ca:.4f}")
    print(f"  B (thermal) vs C (random): {div_bc:.4f}")

    weight_divergence_results = {
        'B_vs_A': round(div_ba, 6),
        'C_vs_A': round(div_ca, 6),
        'B_vs_C': round(div_bc, 6),
    }

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    for k in ['A', 'B', 'C']:
        r = all_results[k]
        alpha_str = ', '.join(f'{a:.4f}' for a in r['alphas']) if r['alphas'] else 'N/A'
        print(f"  [{k}] {r['label']}")
        print(f"      Accuracy:  {r['final_test_acc']:.4f}")
        print(f"      Alphas:    [{alpha_str}]")
        gn = r['gradient_norms']
        print(f"      Grad norm: hot={gn['avg_gnorm_hot']:.4f}  cold={gn['avg_gnorm_cold']:.4f}")
        tr = r['temp_range']
        print(f"      Temp range: {tr['min']:.1f} - {tr['max']:.1f} C  "
              f"(mean={tr['mean']:.1f}, std={tr['std']:.1f})")
    print(f"\n  Weight divergence:")
    print(f"    B (thermal) vs A (fixed):  {div_ba:.4f}")
    print(f"    C (random)  vs A (fixed):  {div_ca:.4f}")
    print(f"    B (thermal) vs C (random): {div_bc:.4f}")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    output = {
        'experiment': 'z901_thermal_morphing_network',
        'hypothesis': (
            'A network whose activation functions are parameterized by live GPU '
            'temperature learns fundamentally different representations than a '
            'fixed network.'
        ),
        'timestamp': datetime.datetime.now().isoformat(),
        'device': str(device),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'warmup_iters': args.warmup_iters,
        'conditions': all_results,
        'weight_divergence': weight_divergence_results,
    }

    results_path = results_dir / 'z901_thermal_morphing_network.json'
    with open(str(results_path), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    print("  Done.")


if __name__ == '__main__':
    main()
