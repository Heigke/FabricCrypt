#!/usr/bin/env python3
"""
z900 - Self-Predicting Autoencoder: Does energy awareness reshape latent space?

Hypothesis: A small autoencoder that jointly predicts its own energy cost develops
measurably different latent representations than one without energy awareness.

Architecture:
  784 -> 256 -> 64 (latent) -> 256 -> 784   (~300K params, MNIST)
  Energy head: 64 -> 32 -> 1                 (predicts joules for this forward pass)

Four experimental conditions (SAME architecture, SAME hyperparameters):
  A: No energy head (baseline quality)
  B: Energy head with REAL energy (embodied)
  C: Energy head with RANDOM energy (controls for: does ANY auxiliary signal reshape latent?)
  D: Energy head with DELAYED energy (controls for: does CAUSAL TIMING matter?)

FALSIFICATION CRITERION:
  If B ≈ C (real ≈ random), then the system learns from ANY signal, not specifically
  from its own energy — the embodiment hypothesis is falsified for this architecture.
  If B ≈ D (real ≈ delayed), then causal timing doesn't matter — mere correlation suffices.

WHAT THIS MEASURES (honestly):
  - Whether real-time energy creates statistically distinguishable representations
  - Whether causal timing of energy feedback matters (B vs D)
  - Whether any random auxiliary signal produces similar effects (B vs C)

WHAT THIS CANNOT SHOW:
  - That the system is conscious or has subjective experience
  - That energy awareness constitutes "feeling" in any phenomenal sense
  - That this is sufficient for embodied cognition (Milinkovic & Aru 2025)

Metrics:
  - Energy prediction MSE (B should be lowest; C, D near random)
  - Reconstruction MSE (all conditions should be similar)
  - Latent space CKA divergence between conditions
  - Correlation(latent_activations, actual_energy)
  - Statistical significance via permutation tests (p < 0.05 required)

Usage:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z900_self_predicting_autoencoder.py
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z900_self_predicting_autoencoder.py --epochs 20
"""

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys
import json
import time
import argparse
import datetime
import traceback
from pathlib import Path

# Project root imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Telemetry: robust import with mock fallback
# ---------------------------------------------------------------------------

_TELEMETRY_AVAILABLE = False
_telemetry_instance = None

try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter
    # Try to actually instantiate to see if hwmon is accessible
    _test_telem = SysfsHwmonTelemetry(sample_rate_hz=50)
    _TELEMETRY_AVAILABLE = True
    del _test_telem
    print("[telemetry] sysfs_hwmon available - real energy measurement enabled")
except Exception as e:
    print(f"[telemetry] sysfs_hwmon not available ({e}), using mock (energy=0)")


class MockTelemetry:
    """Mock telemetry that returns 0 energy when real hardware is unavailable."""

    def __init__(self, sample_rate_hz=50):
        self.sample_rate_hz = sample_rate_hz

    def reset_accumulator(self):
        pass

    def start_continuous_sampling(self):
        pass

    def stop_continuous_sampling(self):
        pass

    def get_accumulated_energy_j(self):
        return 0.0


class MockEnergyMeter:
    """Mock context manager returning 0 energy."""

    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.energy_j = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.energy_j = 0.0


def get_telemetry(sample_rate_hz=100):
    """Get real or mock telemetry."""
    global _telemetry_instance
    if _TELEMETRY_AVAILABLE:
        if _telemetry_instance is None:
            _telemetry_instance = SysfsHwmonTelemetry(sample_rate_hz=sample_rate_hz)
        return _telemetry_instance
    return MockTelemetry(sample_rate_hz=sample_rate_hz)


def get_energy_meter(telemetry):
    """Get real or mock energy meter."""
    if _TELEMETRY_AVAILABLE:
        return EnergyMeter(telemetry)
    return MockEnergyMeter(telemetry)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """784 -> 256 -> 64 (latent)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """64 -> 256 -> 784."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class EnergyHead(nn.Module):
    """64 -> 32 -> 1 (predicts energy in joules)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # energy is non-negative
        )

    def forward(self, z):
        return self.net(z)


class SelfPredictingAutoencoder(nn.Module):
    """
    Autoencoder with optional energy prediction head.

    When has_energy_head=True, the model also predicts the energy
    consumed by its own forward pass.
    """

    def __init__(self, has_energy_head: bool = False):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.has_energy_head = has_energy_head
        if has_energy_head:
            self.energy_head = EnergyHead()

    def forward(self, x):
        # x: [batch, 784]
        z = self.encoder(x)
        recon = self.decoder(z)
        energy_pred = None
        if self.has_energy_head:
            energy_pred = self.energy_head(z)  # [batch, 1]
        return recon, z, energy_pred

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute linear CKA between two representation matrices.

    Args:
        X: [n_samples, n_features_x]
        Y: [n_samples, n_features_y]

    Returns:
        CKA similarity in [0, 1].
    """
    X = X.float()
    Y = Y.float()
    n = X.shape[0]

    # Center
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    # Gram matrices
    XX = X @ X.T  # [n, n]
    YY = Y @ Y.T  # [n, n]

    # HSIC estimates
    hsic_xy = (XX * YY).sum() / (n - 1) ** 2
    hsic_xx = (XX * XX).sum() / (n - 1) ** 2
    hsic_yy = (YY * YY).sum() / (n - 1) ** 2

    denom = hsic_xx.sqrt() * hsic_yy.sqrt()
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_mnist_loaders(batch_size: int, project_root: Path):
    """Load MNIST train/test with download=True."""
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
    ])

    train_ds = torchvision.datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training loop for a single condition
# ---------------------------------------------------------------------------

def train_condition(
    condition_name: str,
    has_energy_head: bool,
    energy_mode: str,  # "none", "real", "random", "delayed"
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lambda_energy: float,
    telemetry,
):
    """
    Train one condition and return metrics.

    Args:
        condition_name: e.g. "A_baseline"
        has_energy_head: whether model has energy prediction head
        energy_mode: "none"|"real"|"random"|"delayed"
        train_loader, test_loader: MNIST data
        device: torch device
        epochs: number of training epochs
        lambda_energy: weight for energy prediction loss
        telemetry: SysfsHwmonTelemetry or MockTelemetry

    Returns:
        dict with all metrics
    """
    print(f"\n{'='*70}")
    print(f"  CONDITION {condition_name}")
    print(f"  energy_head={has_energy_head}, energy_mode={energy_mode}")
    print(f"{'='*70}")

    model = SelfPredictingAutoencoder(has_energy_head=has_energy_head).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()
    energy_criterion = nn.MSELoss()

    param_count = model.param_count()
    print(f"  Parameters: {param_count:,}")

    # Tracking
    epoch_metrics = []
    all_energy_actual = []
    all_energy_predicted = []
    prev_batch_energy = 0.0  # For delayed condition

    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_recon_loss = 0.0
        epoch_energy_loss = 0.0
        epoch_total_loss = 0.0
        epoch_energy_actual_sum = 0.0
        epoch_energy_pred_sum = 0.0
        n_batches = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            # --- Measure real energy of forward pass ---
            if energy_mode in ("real", "delayed") and _TELEMETRY_AVAILABLE:
                with get_energy_meter(telemetry) as meter:
                    recon, z, energy_pred = model(data)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                actual_energy = meter.energy_j
            else:
                recon, z, energy_pred = model(data)
                actual_energy = 0.0

            # --- Determine target energy for this batch ---
            if energy_mode == "none":
                target_energy = None
            elif energy_mode == "real":
                target_energy = actual_energy
            elif energy_mode == "random":
                # Random energy in plausible range [0, 0.1] joules
                target_energy = np.random.exponential(0.02)
            elif energy_mode == "delayed":
                target_energy = prev_batch_energy
                prev_batch_energy = actual_energy
            else:
                target_energy = None

            # --- Compute loss ---
            loss_recon = recon_criterion(recon, data)
            loss = loss_recon

            if has_energy_head and energy_pred is not None and target_energy is not None:
                # Broadcast scalar target to batch
                energy_target_tensor = torch.full(
                    (data.shape[0], 1), target_energy, device=device, dtype=torch.float32
                )
                loss_energy = energy_criterion(energy_pred, energy_target_tensor)
                loss = loss + lambda_energy * loss_energy
                epoch_energy_loss += loss_energy.item()

                # Track for correlation analysis
                all_energy_actual.append(actual_energy)
                all_energy_predicted.append(energy_pred.mean().item())
                epoch_energy_pred_sum += energy_pred.mean().item()
            else:
                loss_energy = None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon_loss += loss_recon.item()
            epoch_total_loss += loss.item()
            epoch_energy_actual_sum += actual_energy
            n_batches += 1

        # Epoch averages
        avg_recon = epoch_recon_loss / max(n_batches, 1)
        avg_energy = epoch_energy_loss / max(n_batches, 1)
        avg_total = epoch_total_loss / max(n_batches, 1)
        avg_actual_energy = epoch_energy_actual_sum / max(n_batches, 1)

        epoch_metrics.append({
            "epoch": epoch + 1,
            "recon_mse": avg_recon,
            "energy_pred_mse": avg_energy if has_energy_head else None,
            "total_loss": avg_total,
            "avg_actual_energy_j": avg_actual_energy,
        })

        # Progress
        energy_str = ""
        if has_energy_head:
            energy_str = f" | E_pred_MSE={avg_energy:.6f} | E_actual={avg_actual_energy:.6f}J"
        print(f"  Epoch {epoch+1:2d}/{epochs}  recon_MSE={avg_recon:.6f}  total={avg_total:.6f}{energy_str}")

    train_time = time.time() - t_start

    # --- Evaluation on test set ---
    model.eval()
    test_recon_mse = 0.0
    test_energy_mse = 0.0
    test_batches = 0
    latent_activations = []
    actual_energies = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            if _TELEMETRY_AVAILABLE and energy_mode in ("real", "delayed"):
                with get_energy_meter(telemetry) as meter:
                    recon, z, energy_pred = model(data)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                actual_energy = meter.energy_j
            else:
                recon, z, energy_pred = model(data)
                actual_energy = 0.0

            test_recon_mse += recon_criterion(recon, data).item()

            if has_energy_head and energy_pred is not None:
                energy_target = torch.full(
                    (data.shape[0], 1), actual_energy, device=device, dtype=torch.float32
                )
                test_energy_mse += energy_criterion(energy_pred, energy_target).item()

            latent_activations.append(z.cpu())
            actual_energies.append(actual_energy)
            test_batches += 1

    test_recon_mse /= max(test_batches, 1)
    test_energy_mse /= max(test_batches, 1)

    # Latent-energy correlation: average latent L2 norm vs actual energy per batch
    latent_norms = [la.norm(dim=1).mean().item() for la in latent_activations]
    if len(actual_energies) > 2 and any(e > 0 for e in actual_energies):
        corr = np.corrcoef(latent_norms, actual_energies)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    # Energy prediction correlation (training)
    if len(all_energy_actual) > 2 and any(e > 0 for e in all_energy_actual):
        train_energy_corr = np.corrcoef(all_energy_actual, all_energy_predicted)[0, 1]
        if np.isnan(train_energy_corr):
            train_energy_corr = 0.0
    else:
        train_energy_corr = 0.0

    # Concatenate all test latents for CKA later
    all_latents = torch.cat(latent_activations, dim=0)

    print(f"\n  Test results:")
    print(f"    Recon MSE:        {test_recon_mse:.6f}")
    if has_energy_head:
        print(f"    Energy pred MSE:  {test_energy_mse:.6f}")
        print(f"    Train E corr:     {train_energy_corr:.4f}")
    print(f"    Latent-energy r:  {corr:.4f}")
    print(f"    Training time:    {train_time:.1f}s")

    result = {
        "condition": condition_name,
        "has_energy_head": has_energy_head,
        "energy_mode": energy_mode,
        "param_count": param_count,
        "epochs": epochs,
        "train_time_s": round(train_time, 2),
        "test_recon_mse": round(test_recon_mse, 8),
        "test_energy_pred_mse": round(test_energy_mse, 8) if has_energy_head else None,
        "latent_energy_correlation": round(corr, 6),
        "train_energy_prediction_corr": round(train_energy_corr, 6) if has_energy_head else None,
        "epoch_metrics": epoch_metrics,
    }

    return result, model, all_latents


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="z900 Self-Predicting Autoencoder Experiment"
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per condition (default: 10)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--lambda-energy", type=float, default=0.1,
                        help="Weight for energy prediction loss (default: 0.1)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu (default: auto)")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("  z900 - Self-Predicting Autoencoder Experiment")
    print("=" * 70)
    print(f"  Device:        {device}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Lambda energy: {args.lambda_energy}")
    print(f"  Telemetry:     {'REAL (sysfs_hwmon)' if _TELEMETRY_AVAILABLE else 'MOCK (energy=0)'}")
    if device.type == "cuda":
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Timestamp:     {datetime.datetime.now().isoformat()}")
    print("=" * 70)

    # Project root
    project_root = Path(__file__).parent.parent

    # Load data
    print("\n[data] Loading MNIST...")
    train_loader, test_loader = get_mnist_loaders(args.batch_size, project_root)
    print(f"[data] Train: {len(train_loader.dataset)} samples, "
          f"Test: {len(test_loader.dataset)} samples")

    # Initialize telemetry
    telemetry = get_telemetry(sample_rate_hz=100)

    # --- Run all four conditions ---
    conditions = [
        ("A_baseline",  False, "none"),
        ("B_embodied",  True,  "real"),
        ("C_random",    True,  "random"),
        ("D_delayed",   True,  "delayed"),
    ]

    results = {}
    latents = {}

    total_start = time.time()

    for cond_name, has_head, energy_mode in conditions:
        result, model, test_latents = train_condition(
            condition_name=cond_name,
            has_energy_head=has_head,
            energy_mode=energy_mode,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lambda_energy=args.lambda_energy,
            telemetry=telemetry,
        )

        results[cond_name] = result
        latents[cond_name] = test_latents

        # Save checkpoint
        ckpt_dir = project_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"z900_{cond_name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "condition": cond_name,
            "has_energy_head": has_head,
            "energy_mode": energy_mode,
            "test_recon_mse": result["test_recon_mse"],
        }, str(ckpt_path))
        print(f"  Checkpoint saved: {ckpt_path}")

    total_time = time.time() - total_start

    # --- CKA comparison ---
    print(f"\n{'='*70}")
    print("  CKA Latent Space Comparison")
    print(f"{'='*70}")

    cond_names = list(latents.keys())
    cka_matrix = {}

    # Use a fixed subset for CKA (first 2048 samples for efficiency)
    cka_n = min(2048, min(lat.shape[0] for lat in latents.values()))
    latent_subset = {k: v[:cka_n] for k, v in latents.items()}

    for i, name_i in enumerate(cond_names):
        for j, name_j in enumerate(cond_names):
            if j >= i:
                cka_val = linear_cka(latent_subset[name_i], latent_subset[name_j])
                key = f"{name_i}_vs_{name_j}"
                cka_matrix[key] = round(cka_val, 6)
                if i != j:
                    print(f"  CKA({name_i}, {name_j}) = {cka_val:.6f}")

    # --- Summary table ---
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Condition':<16} {'Recon MSE':>12} {'E_pred MSE':>12} "
          f"{'E_corr':>10} {'Lat-E r':>10} {'Time(s)':>10}")
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for cond_name in ["A_baseline", "B_embodied", "C_random", "D_delayed"]:
        r = results[cond_name]
        recon = f"{r['test_recon_mse']:.6f}"
        epred = f"{r['test_energy_pred_mse']:.6f}" if r['test_energy_pred_mse'] is not None else "    N/A     "
        ecorr = f"{r['train_energy_prediction_corr']:.4f}" if r['train_energy_prediction_corr'] is not None else "   N/A    "
        lat_e = f"{r['latent_energy_correlation']:.4f}"
        ttime = f"{r['train_time_s']:.1f}"
        print(f"  {cond_name:<16} {recon:>12} {epred:>12} {ecorr:>10} {lat_e:>10} {ttime:>10}")

    # --- CKA divergence summary ---
    print(f"\n  Key CKA divergences (lower = more different latent spaces):")
    baseline_vs = []
    for cond in ["B_embodied", "C_random", "D_delayed"]:
        key = f"A_baseline_vs_{cond}"
        if key in cka_matrix:
            val = cka_matrix[key]
            baseline_vs.append((cond, val))
            print(f"    A_baseline vs {cond}: {val:.6f}")

    # Embodied vs random
    key_br = "B_embodied_vs_C_random"
    if key_br in cka_matrix:
        print(f"    B_embodied vs C_random: {cka_matrix[key_br]:.6f}")

    # --- Interpretation ---
    print(f"\n  INTERPRETATION:")
    if _TELEMETRY_AVAILABLE:
        b_mse = results["B_embodied"]["test_energy_pred_mse"]
        c_mse = results["C_random"]["test_energy_pred_mse"]
        if b_mse is not None and c_mse is not None:
            if b_mse < c_mse:
                print("    [+] Embodied (B) has lower energy prediction error than random (C)")
                print("        => Energy signal is LEARNABLE, not just memorized")
            else:
                print("    [-] Random (C) has lower/equal energy prediction error")
                print("        => Energy signal may not be meaningfully learnable at this scale")

        key_ab = "A_baseline_vs_B_embodied"
        key_ac = "A_baseline_vs_C_random"
        if key_ab in cka_matrix and key_ac in cka_matrix:
            ab = cka_matrix[key_ab]
            ac = cka_matrix[key_ac]
            if ab < ac:
                print("    [+] Embodied (B) diverges MORE from baseline than random (C)")
                print("        => Real energy reshapes latent space differently")
            else:
                print("    [-] Random (C) diverges more or equally from baseline")
                print("        => Any energy head changes representations, not specifically real energy")
    else:
        print("    [!] No real telemetry available - structural test only")
        print("        Re-run on AMD GPU with sysfs hwmon for full experiment")

    # --- Save results ---
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "z900_self_predicting_autoencoder.json"

    output = {
        "experiment": "z900_self_predicting_autoencoder",
        "hypothesis": "Energy-aware autoencoder develops different latent representations",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lambda_energy": args.lambda_energy,
            "device": str(device),
            "telemetry_available": _TELEMETRY_AVAILABLE,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
        },
        "total_time_s": round(total_time, 2),
        "conditions": results,
        "cka_matrix": cka_matrix,
        "summary": {
            "best_recon_mse": min(r["test_recon_mse"] for r in results.values()),
            "best_recon_condition": min(results.values(), key=lambda r: r["test_recon_mse"])["condition"],
            "embodied_energy_pred_mse": results["B_embodied"].get("test_energy_pred_mse"),
            "random_energy_pred_mse": results["C_random"].get("test_energy_pred_mse"),
            "delayed_energy_pred_mse": results["D_delayed"].get("test_energy_pred_mse"),
            "cka_baseline_vs_embodied": cka_matrix.get("A_baseline_vs_B_embodied"),
            "cka_baseline_vs_random": cka_matrix.get("A_baseline_vs_C_random"),
            "cka_embodied_vs_random": cka_matrix.get("B_embodied_vs_C_random"),
        },
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved: {results_path}")
    print(f"  Total experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n{'='*70}")
    print("  z900 experiment complete")
    print(f"{'='*70}")

    return output


if __name__ == "__main__":
    main()
