#!/usr/bin/env python3
"""
z1411: CrossSim Forward-Forward with Real GPU + Analog DRAM

The complete embodied neuromorphic system:
1. Real GPU telemetry (power, temp, utilization) via sysfs
2. CrossSim-style analog DRAM simulation (partial writes, decay, noise)
3. Forward-Forward learning (no backprop, local learning rules)
4. Thermal coupling: GPU temp → DRAM decay rate
5. Memory consolidation: Important patterns written to DRAM

This is what neuromorphic researchers would build.

Author: Claude + ikaros
Date: 2026-02-03
"""

import sys
sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from collections import deque

# Import our modules
from src.neuromorphic.crosssim_dram import (
    NeuromorphicDRAMInterface, create_neuromorphic_dram, DRAMDeviceParams
)
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


@dataclass
class EmbodiedConfig:
    """Configuration for embodied FF system."""
    # Network
    input_dim: int = 784
    hidden_dims: List[int] = None
    num_classes: int = 10

    # Forward-Forward
    ff_threshold: float = 2.0
    ff_base_lr: float = 0.03

    # DRAM
    dram_rows: int = 512
    dram_cols: int = 64
    dram_retention_ms: float = 2000.0  # 2 seconds (accelerated)
    consolidation_interval: int = 20

    # Thermal coupling
    thermal_coupling: float = 0.8  # How much GPU temp affects DRAM

    # Training
    batch_size: int = 64
    n_epochs: int = 5

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [400, 400]


class FFLayer(nn.Module):
    """Forward-Forward layer with local learning."""

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.threshold = threshold
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Normalize weights
        with torch.no_grad():
            self.linear.weight.data = F.normalize(self.linear.weight.data, dim=1)

        # Track goodness history for consolidation
        self.goodness_history = deque(maxlen=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        return F.relu(self.linear(x_norm))

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        return (h ** 2).mean(dim=1)

    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                   lr_scale: float = 1.0) -> Dict:
        """Local FF training with optional LR scaling."""
        # Scale learning rate
        for pg in self.optimizer.param_groups:
            pg['lr'] = 0.03 * lr_scale

        self.optimizer.zero_grad()

        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        g_pos = self.goodness(h_pos)
        g_neg = self.goodness(h_neg)

        # FF loss
        loss = F.softplus(self.threshold - g_pos).mean() + \
               F.softplus(g_neg - self.threshold).mean()

        loss.backward()
        self.optimizer.step()

        # Normalize
        with torch.no_grad():
            self.linear.weight.data = F.normalize(self.linear.weight.data, dim=1)

        # Track goodness
        self.goodness_history.append(g_pos.mean().item())

        return {
            'pos_goodness': g_pos.mean().item(),
            'neg_goodness': g_neg.mean().item(),
            'loss': loss.item(),
            'lr_scale': lr_scale,
        }, h_pos.detach(), h_neg.detach()

    def get_consolidation_pattern(self) -> Optional[torch.Tensor]:
        """Get weight pattern for DRAM consolidation."""
        if len(self.goodness_history) < 10:
            return None
        # Return weights if goodness is high
        if np.mean(list(self.goodness_history)[-10:]) > self.threshold:
            return self.linear.weight.data.clone()
        return None


class CrossSimFFNetwork(nn.Module):
    """FF network integrated with CrossSim DRAM."""

    def __init__(self, config: EmbodiedConfig, device: str = 'cuda'):
        super().__init__()
        self.config = config
        self.device = device

        # FF layers
        dims = [config.input_dim + config.num_classes] + config.hidden_dims
        self.layers = nn.ModuleList([
            FFLayer(dims[i], dims[i+1], config.ff_threshold, config.ff_base_lr)
            for i in range(len(dims) - 1)
        ])

        self.to(device)

    def embed_label(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        one_hot = torch.zeros(batch_size, self.config.num_classes, device=x.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return torch.cat([one_hot, x], dim=1)

    def train_step(self, x: torch.Tensor, labels: torch.Tensor,
                   lr_scale: float = 1.0) -> List[Dict]:
        """Train all layers with FF."""
        batch_size = x.shape[0]

        # Positive: correct labels
        x_pos = self.embed_label(x, labels)

        # Negative: wrong labels
        wrong = torch.randint(0, self.config.num_classes, (batch_size,), device=x.device)
        mask = wrong == labels
        wrong[mask] = (wrong[mask] + 1) % self.config.num_classes
        x_neg = self.embed_label(x, wrong)

        # Train each layer
        metrics = []
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            m, h_pos, h_neg = layer.train_step(h_pos, h_neg, lr_scale)
            m['layer'] = i
            metrics.append(m)

        return metrics

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict by finding label with highest goodness."""
        batch_size = x.shape[0]
        all_g = []

        for label in range(self.config.num_classes):
            labels = torch.full((batch_size,), label, device=x.device, dtype=torch.long)
            h = self.embed_label(x, labels)

            total_g = torch.zeros(batch_size, device=x.device)
            for layer in self.layers:
                h = layer(h)
                total_g += layer.goodness(h)

            all_g.append(total_g)

        return torch.stack(all_g, dim=1).argmax(dim=1)

    def get_consolidation_patterns(self) -> List[torch.Tensor]:
        """Get patterns from all layers for DRAM storage."""
        patterns = []
        for layer in self.layers:
            p = layer.get_consolidation_pattern()
            if p is not None:
                patterns.append(p)
        return patterns


class EmbodiedFFTrainer:
    """
    Complete embodied system with real GPU + CrossSim DRAM.
    """

    def __init__(self, config: EmbodiedConfig, device: str = 'cuda'):
        self.config = config
        self.device = device

        # Create FF network
        self.network = CrossSimFFNetwork(config, device)

        # Create CrossSim DRAM
        dram_params = DRAMDeviceParams(
            retention_time_ms_25C=config.dram_retention_ms,
            write_noise_std=0.03,
            read_noise_std=0.01,
            cell_to_cell_variation=0.1,
        )
        self.dram = NeuromorphicDRAMInterface(
            config.dram_rows, config.dram_cols, dram_params, device
        )

        # Real GPU telemetry
        self.gpu_telemetry = SysfsHwmonTelemetry(card_index=0, sample_rate_hz=100.0)

        # Metrics
        self.step_count = 0
        self.consolidation_count = 0
        self.metrics_history = []

        # Thermal history for LR modulation
        self.temp_history = deque(maxlen=50)
        self.power_history = deque(maxlen=50)

    def get_gpu_state(self) -> Dict:
        """Read real GPU telemetry."""
        sample = self.gpu_telemetry.read_sample()
        if sample:
            return {
                'power_w': sample.power_w,
                'temp_c': sample.temp_edge_c,
                'util_pct': sample.gpu_busy_pct,
                'freq_mhz': sample.freq_sclk_mhz,
            }
        return {'power_w': 0, 'temp_c': 25, 'util_pct': 0, 'freq_mhz': 0}

    def compute_lr_scale(self) -> float:
        """
        Compute LR scaling based on thermal state.
        Reduce LR when GPU is hot to prevent thermal runaway.
        """
        if len(self.temp_history) < 5:
            return 1.0

        avg_temp = np.mean(list(self.temp_history)[-10:])

        # Scale down LR when hot
        if avg_temp > 60:
            return 0.5
        elif avg_temp > 50:
            return 0.75
        return 1.0

    def update_dram_temperature(self, gpu_temp: float):
        """
        Thermal coupling: GPU temperature affects DRAM decay rate.
        """
        # DRAM temp is coupled to GPU temp
        dram_temp = 25.0 + (gpu_temp - 25.0) * self.config.thermal_coupling
        self.dram.set_gpu_telemetry(gpu_temp, 0)
        self.dram.params.temperature_C = dram_temp

    def consolidate_to_dram(self) -> int:
        """
        Consolidate important patterns to DRAM.
        This is memory formation - storing learned representations.
        """
        patterns = self.network.get_consolidation_patterns()

        if not patterns:
            return 0

        for i, pattern in enumerate(patterns):
            # Normalize pattern to [0, 1]
            p_norm = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)

            # Take a slice that fits DRAM
            rows = min(p_norm.shape[0], self.config.dram_rows // len(patterns))
            cols = min(p_norm.shape[1], self.config.dram_cols)

            # Store with partial write (consolidation is gradual)
            start_row = i * rows
            self.dram.store_pattern(
                p_norm[:rows, :cols],
                start_row=start_row,
                strength=0.3  # Gradual consolidation
            )

        self.consolidation_count += len(patterns)
        return len(patterns)

    def train_batch(self, x: torch.Tensor, labels: torch.Tensor,
                    epoch: int, batch_idx: int) -> Dict:
        """
        One training step with full embodied loop.
        """
        # 1. Get GPU state (REAL hardware telemetry)
        gpu_pre = self.get_gpu_state()
        self.temp_history.append(gpu_pre['temp_c'])
        self.power_history.append(gpu_pre['power_w'])

        # 2. Update DRAM temperature (thermal coupling)
        self.update_dram_temperature(gpu_pre['temp_c'])

        # 3. Apply DRAM decay based on time since last step
        dram_metrics = self.dram.step()

        # 4. Compute adaptive LR
        lr_scale = self.compute_lr_scale()

        # 5. Forward-Forward training
        start_time = time.time()
        layer_metrics = self.network.train_step(x, labels, lr_scale)
        torch.cuda.synchronize()
        compute_ms = (time.time() - start_time) * 1000

        # 6. Get GPU state after compute
        gpu_post = self.get_gpu_state()

        # 7. Consolidation (memory formation)
        n_consolidated = 0
        if self.step_count % self.config.consolidation_interval == 0:
            n_consolidated = self.consolidate_to_dram()

        # 8. Compute accuracy
        with torch.no_grad():
            preds = self.network.predict(x)
            accuracy = (preds == labels).float().mean().item()

        # Compile metrics
        metrics = {
            'step': self.step_count,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'accuracy': accuracy,
            'total_loss': sum(m['loss'] for m in layer_metrics),
            'total_pos_goodness': sum(m['pos_goodness'] for m in layer_metrics),
            'total_neg_goodness': sum(m['neg_goodness'] for m in layer_metrics),
            'lr_scale': lr_scale,
            'compute_ms': compute_ms,
            # GPU (REAL)
            'gpu_power_w': gpu_post['power_w'],
            'gpu_temp_c': gpu_post['temp_c'],
            'gpu_util_pct': gpu_post['util_pct'],
            'gpu_power_delta': gpu_post['power_w'] - gpu_pre['power_w'],
            'gpu_temp_delta': gpu_post['temp_c'] - gpu_pre['temp_c'],
            # DRAM (CrossSim simulation)
            'dram_temp_c': self.dram.params.temperature_C,
            'dram_mean_charge': dram_metrics['distribution']['mean'],
            'dram_decay': dram_metrics['decay']['charge_lost'],
            'dram_tau_ms': dram_metrics['decay']['mean_tau_ms'],
            'n_consolidated': n_consolidated,
            # Layer details
            'layer_metrics': layer_metrics,
        }

        self.metrics_history.append(metrics)
        self.step_count += 1

        return metrics

    def train_epoch(self, dataloader, epoch: int, print_every: int = 20) -> Dict:
        """Train one epoch."""
        epoch_metrics = {
            'losses': [], 'accuracies': [], 'powers': [],
            'temps': [], 'dram_charges': [], 'lr_scales': []
        }

        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(self.device)
            labels = labels.to(self.device)

            m = self.train_batch(x, labels, epoch, batch_idx)

            epoch_metrics['losses'].append(m['total_loss'])
            epoch_metrics['accuracies'].append(m['accuracy'])
            epoch_metrics['powers'].append(m['gpu_power_w'])
            epoch_metrics['temps'].append(m['gpu_temp_c'])
            epoch_metrics['dram_charges'].append(m['dram_mean_charge'])
            epoch_metrics['lr_scales'].append(m['lr_scale'])

            if batch_idx % print_every == 0:
                cons = f" cons={m['n_consolidated']}" if m['n_consolidated'] > 0 else ""
                print(f"  [{batch_idx:3d}] loss={m['total_loss']:.3f} "
                      f"acc={m['accuracy']:.3f} "
                      f"GPU:{m['gpu_power_w']:.0f}W/{m['gpu_temp_c']:.0f}°C "
                      f"DRAM:{m['dram_mean_charge']:.3f}@{m['dram_temp_c']:.0f}°C "
                      f"lr×{m['lr_scale']:.2f}{cons}")

        return {
            'mean_loss': np.mean(epoch_metrics['losses']),
            'mean_accuracy': np.mean(epoch_metrics['accuracies']),
            'mean_power': np.mean(epoch_metrics['powers']),
            'mean_temp': np.mean(epoch_metrics['temps']),
            'mean_dram_charge': np.mean(epoch_metrics['dram_charges']),
            'mean_lr_scale': np.mean(epoch_metrics['lr_scales']),
        }


def create_synthetic_data(n_samples: int, input_dim: int,
                          num_classes: int, device: str):
    """Create synthetic classification data."""
    centers = torch.randn(num_classes, input_dim, device=device) * 2
    labels = torch.randint(0, num_classes, (n_samples,), device=device)
    x = centers[labels] + torch.randn(n_samples, input_dim, device=device) * 0.3
    return F.normalize(x, dim=1), labels


class SimpleLoader:
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.n = x.shape[0]

    def __iter__(self):
        idx = torch.randperm(self.n, device=self.x.device)
        for i in range(0, self.n, self.batch_size):
            b = idx[i:i+self.batch_size]
            yield self.x[b], self.y[b]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def main():
    print("=" * 70)
    print("z1411: CrossSim Forward-Forward with Real GPU + Analog DRAM")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Configuration
    config = EmbodiedConfig(
        input_dim=784,
        hidden_dims=[400, 400],
        num_classes=10,
        dram_rows=512,
        dram_cols=64,
        dram_retention_ms=2000.0,  # 2 sec retention (accelerated)
        consolidation_interval=25,
        thermal_coupling=0.7,
        batch_size=64,
        n_epochs=5,
    )

    # Create trainer
    print("\n[1] Creating embodied FF system...")
    trainer = EmbodiedFFTrainer(config, device)

    total_params = sum(p.numel() for p in trainer.network.parameters())
    print(f"    Network params: {total_params:,}")
    print(f"    DRAM size: {config.dram_rows}×{config.dram_cols} = "
          f"{config.dram_rows * config.dram_cols:,} cells")
    print(f"    Thermal coupling: {config.thermal_coupling}")

    # Initial GPU state
    gpu_state = trainer.get_gpu_state()
    print(f"    Initial GPU: {gpu_state['power_w']:.1f}W, {gpu_state['temp_c']:.1f}°C")

    # Create data
    print("\n[2] Creating synthetic dataset...")
    n_train = 8000
    x_train, y_train = create_synthetic_data(
        n_train, config.input_dim, config.num_classes, device
    )
    train_loader = SimpleLoader(x_train, y_train, config.batch_size)
    print(f"    Samples: {n_train}, Batches: {len(train_loader)}")

    # Training
    print("\n[3] Training with embodied FF + CrossSim DRAM...")
    print("-" * 70)

    results = {
        'config': asdict(config),
        'epochs': [],
    }

    start_time = time.time()

    for epoch in range(config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        epoch_metrics = trainer.train_epoch(train_loader, epoch, print_every=25)

        print(f"  Summary: loss={epoch_metrics['mean_loss']:.3f} "
              f"acc={epoch_metrics['mean_accuracy']:.3f} "
              f"GPU:{epoch_metrics['mean_power']:.0f}W/{epoch_metrics['mean_temp']:.0f}°C "
              f"DRAM:{epoch_metrics['mean_dram_charge']:.3f}")

        results['epochs'].append(epoch_metrics)

    training_time = time.time() - start_time

    # Final evaluation
    print("\n[4] Final evaluation...")
    with torch.no_grad():
        preds = trainer.network.predict(x_train[:2000])
        final_acc = (preds == y_train[:2000]).float().mean().item()

    # Final states
    final_gpu = trainer.get_gpu_state()
    final_dram = trainer.dram.dram.get_charge_distribution()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Real GPU + CrossSim DRAM Embodied Learning")
    print("=" * 70)

    results['final'] = {
        'accuracy': final_acc,
        'training_time_s': training_time,
        'total_steps': trainer.step_count,
        'total_consolidated': trainer.consolidation_count,
        'final_gpu_power_w': final_gpu['power_w'],
        'final_gpu_temp_c': final_gpu['temp_c'],
        'final_dram_mean_charge': final_dram['mean'],
    }

    print(f"""
  Training time: {training_time:.1f}s
  Final accuracy: {final_acc:.4f}
  Total steps: {trainer.step_count}

  REAL GPU (Hardware-in-Loop):
    Final power: {final_gpu['power_w']:.1f}W
    Final temp: {final_gpu['temp_c']:.1f}°C

  CrossSim DRAM (Neuromorphic Simulation):
    Mean charge: {final_dram['mean']:.4f}
    Partial cells: {final_dram['partial']}
    Patterns consolidated: {trainer.consolidation_count}

  EMBODIED FEATURES:
    ✓ Real GPU telemetry drives learning rate
    ✓ GPU temp → DRAM decay rate (thermal coupling)
    ✓ FF local learning (no backprop)
    ✓ Memory consolidation to analog DRAM
    ✓ Partial writes with device noise
""")

    # Save results
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1411_crosssim_ff_embodied.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x if not hasattr(x, 'tolist') else x.tolist())
    print(f"Results saved to: {results_path}")

    print("\n✓ CrossSim + FF + Real GPU embodied learning complete!")

    return results


if __name__ == "__main__":
    main()
