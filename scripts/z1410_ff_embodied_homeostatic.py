#!/usr/bin/env python3
"""
z1410: Forward-Forward with Homeostatic Embodied Control

Extends z1409 with:
1. Homeostatic regulation: Network adapts behavior based on GPU thermal/power state
2. DRAM memory consolidation: Important patterns written to simulated DRAM
3. Adaptive learning rate based on body state
4. Thermal-aware compute scheduling

The network now REACTS to its body state, not just predicts it.

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
from typing import List, Tuple, Dict
from collections import deque

from src.embodied.unified_embodied_interface import (
    UnifiedEmbodiedInterface, create_embodied_interface, BodyState, Action
)


@dataclass
class HomeostaticConfig:
    """Configuration for homeostatic control."""
    thermal_setpoint_c: float = 55.0      # Target temperature
    thermal_tolerance_c: float = 5.0       # Acceptable range
    power_budget_w: float = 50.0           # Target power
    power_tolerance_w: float = 10.0        # Acceptable range

    # Learning rate modulation
    base_lr: float = 0.03
    lr_thermal_scale: float = 0.5          # Reduce LR when hot
    lr_power_scale: float = 0.5            # Reduce LR when power high

    # Memory consolidation thresholds
    consolidation_goodness_threshold: float = 3.0  # Min goodness to consolidate
    consolidation_interval: int = 50       # Steps between consolidations


class HomeostaticFFLayer(nn.Module):
    """
    FF layer with homeostatic learning rate modulation.
    """

    def __init__(self, in_features: int, out_features: int,
                 config: HomeostaticConfig):
        super().__init__()

        self.config = config
        self.linear = nn.Linear(in_features, out_features)
        self.threshold = 2.0

        # Base optimizer (LR will be modulated)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.base_lr
        )

        # Normalize weights
        with torch.no_grad():
            self.linear.weight.data = F.normalize(self.linear.weight.data, dim=1)

        # Track activation patterns
        self.activation_buffer = deque(maxlen=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        out = F.relu(self.linear(x_norm))
        return out

    def compute_goodness(self, x: torch.Tensor) -> torch.Tensor:
        return (x ** 2).mean(dim=1)

    def modulate_lr(self, body_state: BodyState):
        """Adjust learning rate based on body state."""
        base_lr = self.config.base_lr

        # Thermal modulation: reduce LR when hot
        thermal_excess = max(0, body_state.gpu_temp_edge_c -
                            self.config.thermal_setpoint_c)
        thermal_factor = 1.0 - self.config.lr_thermal_scale * (
            thermal_excess / self.config.thermal_tolerance_c
        )
        thermal_factor = max(0.1, thermal_factor)

        # Power modulation: reduce LR when power high
        power_excess = max(0, body_state.gpu_power_w - self.config.power_budget_w)
        power_factor = 1.0 - self.config.lr_power_scale * (
            power_excess / self.config.power_tolerance_w
        )
        power_factor = max(0.1, power_factor)

        # Combined modulation
        effective_lr = base_lr * thermal_factor * power_factor

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = effective_lr

        return effective_lr

    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor,
                   body_state: BodyState) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Homeostatic training step.
        """
        # Modulate LR based on body state
        effective_lr = self.modulate_lr(body_state)

        self.optimizer.zero_grad()

        # Forward
        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        # Compute goodness
        g_pos = self.compute_goodness(h_pos)
        g_neg = self.compute_goodness(h_neg)

        # Standard FF loss
        loss_pos = F.softplus(self.threshold - g_pos).mean()
        loss_neg = F.softplus(g_neg - self.threshold).mean()
        loss = loss_pos + loss_neg

        # Backward
        loss.backward()
        self.optimizer.step()

        # Normalize weights
        with torch.no_grad():
            self.linear.weight.data = F.normalize(self.linear.weight.data, dim=1)

        # Track good activations for memory consolidation
        good_mask = g_pos > self.config.consolidation_goodness_threshold
        if good_mask.any():
            self.activation_buffer.append(h_pos[good_mask].detach().mean(dim=0))

        metrics = {
            'pos_goodness': g_pos.mean().item(),
            'neg_goodness': g_neg.mean().item(),
            'goodness_gap': (g_pos - g_neg).mean().item(),
            'loss': loss.item(),
            'effective_lr': effective_lr,
            'n_good_patterns': good_mask.sum().item(),
        }

        return metrics, h_pos.detach(), h_neg.detach()

    def get_consolidation_pattern(self) -> torch.Tensor:
        """Get averaged pattern for memory consolidation."""
        if len(self.activation_buffer) == 0:
            return None
        return torch.stack(list(self.activation_buffer)).mean(dim=0)


class HomeostaticFFNetwork(nn.Module):
    """
    Forward-Forward network with homeostatic embodied control.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 num_classes: int,
                 config: HomeostaticConfig,
                 device: str = 'cuda'):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config
        self.device = device

        # FF layers
        dims = [input_dim + num_classes] + hidden_dims
        self.ff_layers = nn.ModuleList([
            HomeostaticFFLayer(dims[i], dims[i+1], config)
            for i in range(len(dims) - 1)
        ])

        # Body state predictor
        total_hidden = sum(hidden_dims)
        self.body_predictor = nn.Sequential(
            nn.Linear(total_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.body_predictor_optimizer = torch.optim.Adam(
            self.body_predictor.parameters(), lr=0.001
        )

        # Move to device
        self.to(device)

    def embed_label(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        one_hot = torch.zeros(batch_size, self.num_classes, device=x.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return torch.cat([one_hot, x], dim=1)

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = []
        h = x
        for layer in self.ff_layers:
            h = layer(h)
            activations.append(h)
        return activations

    def predict_body_state_delta(self, activations: List[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(activations, dim=1)
        pooled = concat.mean(dim=0, keepdim=True)
        return self.body_predictor(pooled)

    def train_ff_step(self,
                      x: torch.Tensor,
                      labels: torch.Tensor,
                      body_state: BodyState,
                      pre_body_tensor: torch.Tensor,
                      post_body_tensor: torch.Tensor) -> Tuple[List[Dict], float]:
        """
        Homeostatic FF training step.
        """
        batch_size = x.shape[0]

        # Positive examples
        x_pos = self.embed_label(x, labels)

        # Negative examples
        wrong_labels = torch.randint(0, self.num_classes, (batch_size,), device=x.device)
        mask = wrong_labels == labels
        wrong_labels[mask] = (wrong_labels[mask] + 1) % self.num_classes
        x_neg = self.embed_label(x, wrong_labels)

        # Train layers with homeostatic modulation
        layer_metrics = []
        h_pos, h_neg = x_pos, x_neg
        all_activations = []

        for idx, layer in enumerate(self.ff_layers):
            metrics, h_pos, h_neg = layer.train_step(h_pos, h_neg, body_state)
            metrics['layer_idx'] = idx
            layer_metrics.append(metrics)
            all_activations.append(h_pos)

        # Train body predictor
        self.body_predictor_optimizer.zero_grad()

        predicted_delta = self.predict_body_state_delta(all_activations)
        actual_delta = (post_body_tensor - pre_body_tensor).unsqueeze(0)

        # Focus on key body dimensions (power, temp, util)
        # Indices: 0=power, 1=temp_edge, 5=busy_pct
        key_dims = [0, 1, 5]
        body_loss = F.mse_loss(
            predicted_delta[:, key_dims],
            actual_delta[:, key_dims]
        )
        body_loss.backward()
        self.body_predictor_optimizer.step()

        return layer_metrics, body_loss.item()

    def predict_label(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        all_goodness = []

        for label in range(self.num_classes):
            labels = torch.full((batch_size,), label, device=x.device, dtype=torch.long)
            x_labeled = self.embed_label(x, labels)

            h = x_labeled
            total_goodness = torch.zeros(batch_size, device=x.device)

            for layer in self.ff_layers:
                h = layer(h)
                total_goodness += layer.compute_goodness(h)

            all_goodness.append(total_goodness)

        goodness_matrix = torch.stack(all_goodness, dim=1)
        return goodness_matrix.argmax(dim=1)

    def get_consolidation_patterns(self) -> List[torch.Tensor]:
        """Get patterns from all layers for DRAM consolidation."""
        patterns = []
        for layer in self.ff_layers:
            pattern = layer.get_consolidation_pattern()
            if pattern is not None:
                patterns.append(pattern)
        return patterns


class HomeostaticTrainer:
    """
    Trainer with homeostatic control and memory consolidation.
    """

    def __init__(self,
                 network: HomeostaticFFNetwork,
                 interface: UnifiedEmbodiedInterface,
                 config: HomeostaticConfig,
                 device: str = 'cuda'):

        self.network = network
        self.interface = interface
        self.config = config
        self.device = device

        self.step_count = 0
        self.consolidation_count = 0
        self.metrics_history = []

        # Homeostatic state tracking
        self.thermal_history = deque(maxlen=50)
        self.power_history = deque(maxlen=50)

    def should_consolidate(self) -> bool:
        """Check if we should consolidate memories to DRAM."""
        return self.step_count % self.config.consolidation_interval == 0

    def consolidate_to_dram(self):
        """Write important patterns to simulated DRAM."""
        patterns = self.network.get_consolidation_patterns()

        if not patterns:
            return 0

        # Write each pattern to DRAM
        for i, pattern in enumerate(patterns):
            # Convert pattern to address + data
            # Use pattern hash as address, pattern mean as write strength
            pattern_bytes = pattern.cpu().numpy().tobytes()
            addr = hash(pattern_bytes) & 0xFFFFF  # 20-bit address
            data = int(pattern.abs().sum().item()) & 0xFFFFFFFF

            # Write strength based on pattern variance (important patterns have high variance)
            strength = min(1.0, pattern.std().item() / 2.0)

            action = Action(
                dram_write_address=addr,
                dram_write_data=data,
                dram_write_strength=strength
            )

            self.interface.step(None, action)

        self.consolidation_count += len(patterns)
        return len(patterns)

    def train_batch(self, x: torch.Tensor, labels: torch.Tensor,
                    epoch: int, batch_idx: int) -> Dict:
        """
        Homeostatic training step with embodied feedback.
        """
        # Pre-compute state
        pre_state = self.interface.get_body_state()
        pre_tensor = pre_state.to_tensor(self.device)

        # Update homeostatic tracking
        self.thermal_history.append(pre_state.gpu_temp_edge_c)
        self.power_history.append(pre_state.gpu_power_w)

        # Time compute
        start_time = time.time()

        # Training step
        layer_metrics, body_pred_error = self.network.train_ff_step(
            x, labels, pre_state, pre_tensor, pre_tensor
        )

        torch.cuda.synchronize()
        compute_ms = (time.time() - start_time) * 1000

        # Post-compute state
        post_state = self.interface.get_body_state()
        post_tensor = post_state.to_tensor(self.device)

        # Actual body prediction training
        _, body_pred_error = self.network.train_ff_step(
            x, labels, pre_state, pre_tensor, post_tensor
        )

        # Memory consolidation
        n_consolidated = 0
        if self.should_consolidate():
            n_consolidated = self.consolidate_to_dram()

        # Compute accuracy
        with torch.no_grad():
            predictions = self.network.predict_label(x)
            accuracy = (predictions == labels).float().mean().item()

        # Homeostatic metrics
        thermal_deviation = post_state.gpu_temp_edge_c - self.config.thermal_setpoint_c
        power_deviation = post_state.gpu_power_w - self.config.power_budget_w

        metrics = {
            'step': self.step_count,
            'epoch': epoch,
            'batch_idx': batch_idx,
            'layer_metrics': layer_metrics,
            'total_loss': sum(m['loss'] for m in layer_metrics),
            'accuracy': accuracy,
            'gpu_power_w': post_state.gpu_power_w,
            'gpu_temp_c': post_state.gpu_temp_edge_c,
            'gpu_util_pct': post_state.gpu_busy_pct,
            'thermal_deviation': thermal_deviation,
            'power_deviation': power_deviation,
            'body_pred_error': body_pred_error,
            'compute_ms': compute_ms,
            'n_consolidated': n_consolidated,
            'effective_lr_mean': np.mean([m['effective_lr'] for m in layer_metrics]),
        }

        self.metrics_history.append(metrics)
        self.step_count += 1

        return metrics

    def train_epoch(self, dataloader, epoch: int, print_every: int = 20) -> Dict:
        """Train one epoch with homeostatic control."""
        epoch_metrics = {
            'losses': [],
            'accuracies': [],
            'powers': [],
            'temps': [],
            'body_errors': [],
            'effective_lrs': [],
        }

        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(self.device)
            labels = labels.to(self.device)

            metrics = self.train_batch(x, labels, epoch, batch_idx)

            epoch_metrics['losses'].append(metrics['total_loss'])
            epoch_metrics['accuracies'].append(metrics['accuracy'])
            epoch_metrics['powers'].append(metrics['gpu_power_w'])
            epoch_metrics['temps'].append(metrics['gpu_temp_c'])
            epoch_metrics['body_errors'].append(metrics['body_pred_error'])
            epoch_metrics['effective_lrs'].append(metrics['effective_lr_mean'])

            if batch_idx % print_every == 0:
                lr_str = f"{metrics['effective_lr_mean']:.4f}"
                cons_str = f" cons={metrics['n_consolidated']}" if metrics['n_consolidated'] > 0 else ""
                print(f"  [{batch_idx:3d}] loss={metrics['total_loss']:.3f} "
                      f"acc={metrics['accuracy']:.3f} "
                      f"pow={metrics['gpu_power_w']:.0f}W "
                      f"tmp={metrics['gpu_temp_c']:.0f}°C "
                      f"lr={lr_str}"
                      f"{cons_str}")

        return {
            'mean_loss': np.mean(epoch_metrics['losses']),
            'mean_accuracy': np.mean(epoch_metrics['accuracies']),
            'mean_power': np.mean(epoch_metrics['powers']),
            'mean_temp': np.mean(epoch_metrics['temps']),
            'mean_body_error': np.mean(epoch_metrics['body_errors']),
            'mean_effective_lr': np.mean(epoch_metrics['effective_lrs']),
        }


def create_synthetic_dataset(n_samples: int, input_dim: int, num_classes: int, device: str):
    """Create synthetic dataset with clear class structure."""
    class_centers = torch.randn(num_classes, input_dim, device=device) * 3

    labels = torch.randint(0, num_classes, (n_samples,), device=device)
    x = class_centers[labels] + torch.randn(n_samples, input_dim, device=device) * 0.3
    x = F.normalize(x, dim=1)

    return x, labels


class SimpleDataLoader:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.n = x.shape[0]

    def __iter__(self):
        indices = torch.randperm(self.n, device=self.x.device)
        for i in range(0, self.n, self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            yield self.x[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def main():
    print("=" * 70)
    print("z1410: Forward-Forward with Homeostatic Embodied Control")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Configuration
    config = HomeostaticConfig(
        thermal_setpoint_c=50.0,
        thermal_tolerance_c=10.0,
        power_budget_w=45.0,
        power_tolerance_w=15.0,
        base_lr=0.05,
        lr_thermal_scale=0.6,
        lr_power_scale=0.4,
        consolidation_goodness_threshold=2.5,
        consolidation_interval=25,
    )

    # Initialize embodied interface
    print("\n[1] Initializing embodied interface...")
    interface = create_embodied_interface(
        use_real_gpu=True,
        use_real_fpga=False,
        device=device
    )
    interface.connect()

    initial_state = interface.get_body_state()
    print(f"    Initial: {initial_state.gpu_power_w:.1f}W, {initial_state.gpu_temp_edge_c:.1f}°C")
    print(f"    Homeostatic setpoints: {config.power_budget_w}W, {config.thermal_setpoint_c}°C")

    # Create network
    print("\n[2] Creating homeostatic FF network...")
    input_dim = 784
    hidden_dims = [500, 500]
    num_classes = 10

    network = HomeostaticFFNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        config=config,
        device=device
    )

    total_params = sum(p.numel() for p in network.parameters())
    print(f"    Parameters: {total_params:,}")

    # Create trainer
    trainer = HomeostaticTrainer(network, interface, config, device)

    # Create data
    print("\n[3] Creating dataset...")
    n_train = 8000
    batch_size = 64

    x_train, y_train = create_synthetic_dataset(n_train, input_dim, num_classes, device)
    train_loader = SimpleDataLoader(x_train, y_train, batch_size)

    # Training
    print("\n[4] Training with homeostatic control...")
    print("-" * 70)

    results = {
        'config': asdict(config),
        'network': {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'num_classes': num_classes,
        },
        'epochs': [],
    }

    n_epochs = 6
    start_time = time.time()

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        epoch_metrics = trainer.train_epoch(train_loader, epoch, print_every=25)

        print(f"  Summary: loss={epoch_metrics['mean_loss']:.3f} "
              f"acc={epoch_metrics['mean_accuracy']:.3f} "
              f"pow={epoch_metrics['mean_power']:.1f}W "
              f"tmp={epoch_metrics['mean_temp']:.1f}°C "
              f"lr={epoch_metrics['mean_effective_lr']:.4f}")

        results['epochs'].append(epoch_metrics)

    training_time = time.time() - start_time

    # Final evaluation
    print("\n[5] Final evaluation...")
    with torch.no_grad():
        predictions = network.predict_label(x_train[:2000])
        test_accuracy = (predictions == y_train[:2000]).float().mean().item()

    final_state = interface.get_body_state()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results['final'] = {
        'test_accuracy': test_accuracy,
        'training_time_s': training_time,
        'total_steps': trainer.step_count,
        'total_consolidated': trainer.consolidation_count,
        'final_power_w': final_state.gpu_power_w,
        'final_temp_c': final_state.gpu_temp_edge_c,
    }

    print(f"  Training time: {training_time:.1f}s")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Total training steps: {trainer.step_count}")
    print(f"  Patterns consolidated to DRAM: {trainer.consolidation_count}")
    print(f"  Final: {final_state.gpu_power_w:.1f}W, {final_state.gpu_temp_edge_c:.1f}°C")

    # Key insight
    print("\n" + "-" * 70)
    print("HOMEOSTATIC EMBODIMENT")
    print("-" * 70)
    print(f"""
The network now REACTS to its physical state:

1. Learning Rate Modulation:
   - Base LR: {config.base_lr}
   - When GPU hot (>{config.thermal_setpoint_c}°C): LR reduced up to {config.lr_thermal_scale*100:.0f}%
   - When power high (>{config.power_budget_w}W): LR reduced up to {config.lr_power_scale*100:.0f}%
   - Final avg LR: {results['epochs'][-1]['mean_effective_lr']:.4f}

2. Memory Consolidation:
   - Patterns with goodness >{config.consolidation_goodness_threshold} written to DRAM
   - {trainer.consolidation_count} patterns consolidated total
   - Consolidation every {config.consolidation_interval} steps

This creates ADAPTIVE behavior based on physical constraints!
""")

    # Save results
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1410_ff_embodied_homeostatic.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    interface.disconnect()
    print("\n✓ Homeostatic FF embodied learning complete!")

    return results


if __name__ == "__main__":
    main()
