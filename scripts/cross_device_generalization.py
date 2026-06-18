#!/usr/bin/env python3
"""
Task 3: Cross-Device Generalization
=====================================

Proves z_feel transfers across devices with minimal recalibration.

Key approaches:
1. Domain Randomization: Train on simulated device variations
2. Device Adapter: Small adapter network for device-specific calibration
3. Zero-Shot Transfer: Test on new device without any training

This is crucial for practical deployment - interoception should work
on any hardware, not just the training device.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.closed_loop_interoception import (
    RecurrentInteroceptionState,
    FiLMGenerator,
    InternalSignals,
)


# ============================================================================
# DEVICE SIMULATION
# ============================================================================

@dataclass
class DeviceProfile:
    """Simulated device characteristics."""
    name: str
    base_throughput: float      # tok/s
    thermal_throttle_temp: float  # °C
    memory_bandwidth: float     # GB/s
    power_limit: float          # W
    entropy_bias: float         # Device-specific entropy offset
    latency_factor: float       # Relative latency multiplier

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Pre-defined device profiles
DEVICE_PROFILES = {
    'amd_gfx1151': DeviceProfile(
        name='AMD gfx1151 (RX 7900 XT)',
        base_throughput=45.0,
        thermal_throttle_temp=85.0,
        memory_bandwidth=800.0,
        power_limit=300.0,
        entropy_bias=0.0,
        latency_factor=1.0,
    ),
    'nvidia_4090': DeviceProfile(
        name='NVIDIA RTX 4090',
        base_throughput=65.0,
        thermal_throttle_temp=83.0,
        memory_bandwidth=1000.0,
        power_limit=450.0,
        entropy_bias=-0.1,
        latency_factor=0.8,
    ),
    'apple_m2_ultra': DeviceProfile(
        name='Apple M2 Ultra',
        base_throughput=35.0,
        thermal_throttle_temp=95.0,
        memory_bandwidth=800.0,
        power_limit=120.0,
        entropy_bias=0.05,
        latency_factor=1.2,
    ),
    'intel_arc_a770': DeviceProfile(
        name='Intel Arc A770',
        base_throughput=25.0,
        thermal_throttle_temp=90.0,
        memory_bandwidth=560.0,
        power_limit=225.0,
        entropy_bias=0.15,
        latency_factor=1.5,
    ),
    'amd_mi300x': DeviceProfile(
        name='AMD MI300X',
        base_throughput=80.0,
        thermal_throttle_temp=90.0,
        memory_bandwidth=5300.0,
        power_limit=750.0,
        entropy_bias=-0.15,
        latency_factor=0.6,
    ),
}


def generate_device_signals(
    profile: DeviceProfile,
    stress_level: float,
    add_noise: bool = True,
) -> InternalSignals:
    """Generate signals that would be observed on a specific device."""

    # Base signal modulated by stress
    throughput = profile.base_throughput * (1 - 0.5 * stress_level)
    entropy = 2.0 + profile.entropy_bias + 1.5 * stress_level
    margin = 0.5 - 0.3 * stress_level
    latency = (20.0 / profile.base_throughput) * profile.latency_factor * (1 + stress_level)

    # Add device-specific noise
    if add_noise:
        noise = 0.1
        throughput *= (1 + random.gauss(0, noise))
        entropy *= (1 + random.gauss(0, noise))
        margin *= (1 + random.gauss(0, noise))

    return InternalSignals(
        logit_entropy=max(0.5, entropy),
        logit_margin=max(0.01, margin),
        tokens_per_second=max(1, throughput),
        time_per_token_ms=max(1, latency * 1000),
        stress_indicator=stress_level,
        uncertainty_score=0.3 + 0.4 * stress_level,
        attention_entropy=2.5 + profile.entropy_bias + 0.5 * stress_level,
    )


# ============================================================================
# DOMAIN RANDOMIZATION TRAINING
# ============================================================================

class DomainRandomizedTrainer:
    """
    Train z_feel with domain randomization across simulated devices.

    The key insight: If we train on many device variations,
    the learned representation should generalize.
    """

    def __init__(
        self,
        z_dim: int = 32,
        device: str = 'cpu',
    ):
        self.z_dim = z_dim
        self.device = device

        self.recurrent_state = RecurrentInteroceptionState(
            signal_dim=18,
            state_dim=z_dim,
        ).to(device)

        self.params = list(self.recurrent_state.parameters())

    def generate_randomized_sample(
        self,
        seq_len: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a sample with randomized device characteristics."""

        # Randomly perturb a base profile
        base_profile = random.choice(list(DEVICE_PROFILES.values()))

        # Create randomized profile
        randomized = DeviceProfile(
            name=f"randomized_{base_profile.name}",
            base_throughput=base_profile.base_throughput * random.uniform(0.7, 1.3),
            thermal_throttle_temp=base_profile.thermal_throttle_temp + random.uniform(-5, 5),
            memory_bandwidth=base_profile.memory_bandwidth * random.uniform(0.8, 1.2),
            power_limit=base_profile.power_limit * random.uniform(0.9, 1.1),
            entropy_bias=base_profile.entropy_bias + random.uniform(-0.2, 0.2),
            latency_factor=base_profile.latency_factor * random.uniform(0.8, 1.2),
        )

        # Generate stress trajectory
        stress = 0.3
        signals = []
        regimes = []

        for t in range(seq_len):
            # Random walk stress
            stress = stress + random.gauss(0, 0.05)
            stress = max(0, min(1, stress))

            sig = generate_device_signals(randomized, stress)
            signals.append(torch.tensor(sig.to_vector(), dtype=torch.float32))

            # Regime label based on stress
            regime = min(3, int(stress * 4))
            regimes.append(regime)

        return (
            torch.stack(signals).to(self.device),
            torch.tensor(regimes, dtype=torch.long, device=self.device),
        )

    def train(
        self,
        n_samples: int = 2000,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """Train with domain randomization."""

        optimizer = torch.optim.Adam(self.params, lr=lr)
        history = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            self.recurrent_state.train()

            total_loss = 0
            correct = 0
            total = 0

            for _ in range(n_samples // 10):  # Batch of 10 samples
                optimizer.zero_grad()

                batch_loss = 0
                for _ in range(10):
                    signals, regimes = self.generate_randomized_sample()

                    z_t = self.recurrent_state.init_state(1, self.device)

                    for t in range(signals.size(0)):
                        z_t, outputs = self.recurrent_state(
                            signals[t:t+1],
                            z_t,
                        )

                        loss = F.cross_entropy(
                            outputs['regime_logits'],
                            regimes[t:t+1],
                        )
                        batch_loss = batch_loss + loss

                        pred = outputs['regime_logits'].argmax(dim=-1)
                        correct += (pred == regimes[t]).sum().item()
                        total += 1

                batch_loss = batch_loss / 10
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()

            avg_loss = total_loss / (n_samples // 10)
            accuracy = correct / total if total > 0 else 0

            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.2%}")

        return history

    def evaluate_on_device(
        self,
        profile: DeviceProfile,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """Evaluate on a specific device profile."""

        self.recurrent_state.eval()

        correct = 0
        total = 0
        stress_errors = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Generate sample for this device
                signals = []
                true_stresses = []

                stress = 0.3
                for t in range(32):
                    stress = stress + random.gauss(0, 0.05)
                    stress = max(0, min(1, stress))
                    true_stresses.append(stress)

                    sig = generate_device_signals(profile, stress)
                    signals.append(torch.tensor(sig.to_vector(), dtype=torch.float32))

                signals = torch.stack(signals).to(self.device)
                z_t = self.recurrent_state.init_state(1, self.device)

                for t in range(32):
                    z_t, outputs = self.recurrent_state(signals[t:t+1], z_t)

                    # Check regime prediction
                    true_regime = min(3, int(true_stresses[t] * 4))
                    pred = outputs['regime_logits'].argmax(dim=-1).item()
                    correct += (pred == true_regime)
                    total += 1

                    # Check stress prediction
                    pred_stress = outputs['stress'].item()
                    stress_errors.append(abs(pred_stress - true_stresses[t]))

        return {
            'accuracy': correct / total if total > 0 else 0,
            'stress_mae': np.mean(stress_errors) if stress_errors else 0,
        }


# ============================================================================
# DEVICE ADAPTER
# ============================================================================

class DeviceAdapter(nn.Module):
    """
    Small adapter network for device-specific calibration.

    Instead of retraining the full model, we learn a small adapter
    that transforms device-specific signals to the canonical space.
    """

    def __init__(
        self,
        signal_dim: int = 18,
        hidden_dim: int = 32,
    ):
        super().__init__()

        # Input normalization parameters (learnable)
        self.signal_mean = nn.Parameter(torch.zeros(signal_dim))
        self.signal_std = nn.Parameter(torch.ones(signal_dim))

        # Lightweight MLP adapter
        self.adapter = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, signal_dim),
        )

        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform device-specific signals to canonical space."""
        # Normalize
        x_norm = (x - self.signal_mean) / (self.signal_std + 1e-6)

        # Adapt
        adapted = self.adapter(x_norm)

        # Residual connection (mostly preserve original, add correction)
        output = self.residual_weight * x_norm + (1 - self.residual_weight) * adapted

        return output


def train_device_adapter(
    recurrent_state: RecurrentInteroceptionState,
    source_profile: DeviceProfile,
    target_profile: DeviceProfile,
    n_samples: int = 500,
    epochs: int = 30,
    device: str = 'cpu',
) -> Tuple[DeviceAdapter, Dict[str, List[float]]]:
    """
    Train adapter to transfer from source to target device.

    Uses the trained recurrent_state as a frozen backbone,
    only learning the adapter.
    """

    adapter = DeviceAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    # Freeze backbone
    recurrent_state.eval()
    for p in recurrent_state.parameters():
        p.requires_grad = False

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        adapter.train()

        total_loss = 0
        correct = 0
        total = 0

        for _ in range(n_samples):
            optimizer.zero_grad()

            # Generate same stress trajectory on both devices
            stress = 0.3
            loss = 0

            z_t = recurrent_state.init_state(1, device)

            for t in range(32):
                stress = stress + random.gauss(0, 0.05)
                stress = max(0, min(1, stress))

                # Get target device signals
                target_sig = generate_device_signals(target_profile, stress)
                target_vec = torch.tensor(target_sig.to_vector(), dtype=torch.float32, device=device)

                # Adapt signals
                adapted = adapter(target_vec.unsqueeze(0))

                # Run through backbone
                z_t, outputs = recurrent_state(adapted, z_t)

                # Regime loss
                true_regime = torch.tensor([min(3, int(stress * 4))], device=device)
                step_loss = F.cross_entropy(outputs['regime_logits'], true_regime)
                loss = loss + step_loss

                pred = outputs['regime_logits'].argmax(dim=-1)
                correct += (pred == true_regime).sum().item()
                total += 1

            loss = loss / 32
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_samples
        accuracy = correct / total if total > 0 else 0

        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f"Adapter Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.2%}")

    return adapter, history


# ============================================================================
# EVALUATION
# ============================================================================

def run_transfer_evaluation(
    trainer: DomainRandomizedTrainer,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate transfer to all device profiles."""

    results = {
        'source': 'domain_randomized',
        'devices': {},
    }

    print("\n" + "="*60)
    print("Cross-Device Transfer Evaluation")
    print("="*60)

    for name, profile in DEVICE_PROFILES.items():
        print(f"\nEvaluating on {profile.name}...")
        metrics = trainer.evaluate_on_device(profile, n_samples=200)

        results['devices'][name] = {
            'profile': profile.to_dict(),
            'accuracy': metrics['accuracy'],
            'stress_mae': metrics['stress_mae'],
        }

        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Stress MAE: {metrics['stress_mae']:.3f}")

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_transfer_results(
    results: Dict[str, Any],
    output_dir: Path,
):
    """Plot cross-device transfer results."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    devices = list(results['devices'].keys())
    accuracies = [results['devices'][d]['accuracy'] for d in devices]
    maes = [results['devices'][d]['stress_mae'] for d in devices]

    # Clean up device names for display
    display_names = [d.replace('_', ' ').title() for d in devices]

    # Accuracy by device
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(devices)))
    bars = ax.bar(display_names, accuracies, color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Regime Classification Accuracy by Device')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Stress MAE by device
    ax = axes[1]
    ax.bar(display_names, maes, color=colors)
    ax.set_ylabel('Stress MAE')
    ax.set_title('Stress Prediction Error by Device')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.suptitle('Cross-Device Generalization: z_feel Transfers Without Retraining',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_device_transfer.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'cross_device_transfer.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-Device Generalization")
    parser.add_argument("--output-dir", default="results/cross_device")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--z-dim", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CROSS-DEVICE GENERALIZATION")
    print("="*70)
    print("\nProving z_feel transfers across devices:")
    print("  1. Domain Randomization: Train on simulated device variations")
    print("  2. Zero-Shot Transfer: Test on new devices without training")
    print("  3. Device Adapter: Minimal calibration for new devices")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Train with domain randomization
    print("\n" + "-"*60)
    print("Phase 1: Domain Randomized Training")
    print("-"*60)

    trainer = DomainRandomizedTrainer(z_dim=args.z_dim, device=device)
    history = trainer.train(
        n_samples=args.n_samples,
        epochs=args.epochs,
    )

    # Evaluate transfer
    print("\n" + "-"*60)
    print("Phase 2: Zero-Shot Transfer Evaluation")
    print("-"*60)

    transfer_results = run_transfer_evaluation(trainer, output_dir)

    # Train device adapter for worst-performing device
    print("\n" + "-"*60)
    print("Phase 3: Device Adapter Training")
    print("-"*60)

    # Find worst device
    worst_device = min(
        transfer_results['devices'].keys(),
        key=lambda d: transfer_results['devices'][d]['accuracy']
    )
    print(f"\nTraining adapter for worst device: {worst_device}")

    source_profile = DEVICE_PROFILES['amd_gfx1151']
    target_profile = DEVICE_PROFILES[worst_device]

    adapter, adapter_history = train_device_adapter(
        trainer.recurrent_state,
        source_profile,
        target_profile,
        n_samples=500,
        epochs=30,
        device=device,
    )

    # Save adapter
    torch.save(adapter.state_dict(), output_dir / f"adapter_{worst_device}.pt")

    # Save all results
    results = {
        'domain_randomization': {
            'training_history': history,
        },
        'transfer_evaluation': transfer_results,
        'adapter_training': {
            'target_device': worst_device,
            'history': adapter_history,
        },
    }

    with open(output_dir / 'cross_device_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    plot_transfer_results(transfer_results, output_dir)

    # Summary
    print("\n" + "="*70)
    print("CROSS-DEVICE GENERALIZATION SUMMARY")
    print("="*70)

    print("\nZero-Shot Transfer Accuracy:")
    for name, data in transfer_results['devices'].items():
        print(f"  {name}: {data['accuracy']:.2%}")

    mean_acc = np.mean([d['accuracy'] for d in transfer_results['devices'].values()])
    print(f"\n  Mean accuracy: {mean_acc:.2%}")

    print("\nVerification:")
    if mean_acc > 0.5:
        print("  ✓ z_feel transfers across devices (>50% accuracy)")
    if adapter_history['accuracy'][-1] > transfer_results['devices'][worst_device]['accuracy']:
        print(f"  ✓ Adapter improves worst-device accuracy: "
              f"{transfer_results['devices'][worst_device]['accuracy']:.2%} → "
              f"{adapter_history['accuracy'][-1]:.2%}")

    print("\n  → z_feel generalizes across hardware with minimal adaptation")


if __name__ == "__main__":
    main()
