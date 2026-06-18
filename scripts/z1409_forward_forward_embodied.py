#!/usr/bin/env python3
"""
z1409: Forward-Forward Learning with Embodied GPU Telemetry

Implements Geoffrey Hinton's Forward-Forward algorithm where:
- Each layer learns locally via a "goodness" function
- No backpropagation needed - biologically plausible
- GPU telemetry (power, temp, utilization) becomes embodied feedback

Key innovation: The GPU's physical state (power draw, temperature rise during
forward passes) provides REAL embodied signal that the network learns to
predict and adapt to. This creates true hardware-in-the-loop learning.

Forward-Forward overview:
- Positive pass: Real data with correct label embedded → maximize goodness
- Negative pass: Data with wrong label or corrupted → minimize goodness
- Goodness = sum of squared activations (layer wants high activation for positive)
- Each layer learns independently using only local signals

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
from typing import List, Tuple, Optional, Dict
from collections import deque

# Import embodied interface
from src.embodied.unified_embodied_interface import (
    UnifiedEmbodiedInterface, create_embodied_interface, BodyState, Action
)


@dataclass
class FFLayerMetrics:
    """Metrics for a single FF layer."""
    layer_idx: int
    pos_goodness: float
    neg_goodness: float
    goodness_gap: float  # pos - neg (want positive)
    loss: float
    weight_norm: float
    activation_mean: float
    activation_std: float


@dataclass
class EmbodiedFFMetrics:
    """Combined FF + embodied metrics."""
    step: int
    epoch: int
    batch_idx: int

    # FF metrics
    layer_metrics: List[Dict]
    total_pos_goodness: float
    total_neg_goodness: float
    total_loss: float
    accuracy: float

    # Embodied metrics
    gpu_power_w: float
    gpu_temp_c: float
    gpu_util_pct: float
    power_delta: float
    temp_delta: float
    energy_mj: float
    compute_ms: float

    # Body state prediction
    body_state_pred_error: float
    body_state_dims: int


class FFLayer(nn.Module):
    """
    Forward-Forward layer with local learning rule.

    Each layer:
    1. Normalizes input
    2. Applies linear + ReLU
    3. Computes goodness (sum of squared activations)
    4. Learns to have high goodness for positive, low for negative
    """

    def __init__(self, in_features: int, out_features: int,
                 threshold: float = 2.0, lr: float = 0.03):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.threshold = threshold
        self.lr = lr

        # Local optimizer for this layer only
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Normalize weights
        with torch.no_grad():
            self.linear.weight.data = F.normalize(self.linear.weight.data, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization."""
        # Normalize input
        x_norm = F.normalize(x, dim=1)

        # Linear + ReLU
        out = F.relu(self.linear(x_norm))

        return out

    def compute_goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute goodness = mean squared activation per sample."""
        return (x ** 2).mean(dim=1)

    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> FFLayerMetrics:
        """
        Local learning step.

        Goal: goodness(positive) > threshold > goodness(negative)
        """
        self.optimizer.zero_grad()

        # Forward both
        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        # Compute goodness
        g_pos = self.compute_goodness(h_pos)
        g_neg = self.compute_goodness(h_neg)

        # Loss: want g_pos > threshold and g_neg < threshold
        # Using softplus for smooth loss
        loss_pos = F.softplus(self.threshold - g_pos).mean()
        loss_neg = F.softplus(g_neg - self.threshold).mean()
        loss = loss_pos + loss_neg

        # Backward and update
        loss.backward()
        self.optimizer.step()

        # Normalize weights after update
        with torch.no_grad():
            self.linear.weight.data = F.normalize(self.linear.weight.data, dim=1)

        # Metrics
        metrics = FFLayerMetrics(
            layer_idx=0,  # Will be set by caller
            pos_goodness=g_pos.mean().item(),
            neg_goodness=g_neg.mean().item(),
            goodness_gap=g_pos.mean().item() - g_neg.mean().item(),
            loss=loss.item(),
            weight_norm=self.linear.weight.norm().item(),
            activation_mean=h_pos.mean().item(),
            activation_std=h_pos.std().item(),
        )

        return metrics, h_pos.detach(), h_neg.detach()


class EmbodiedFFNetwork(nn.Module):
    """
    Forward-Forward network with embodied GPU telemetry.

    The network learns to:
    1. Classify inputs using FF algorithm
    2. Predict GPU body state changes from its own compute

    This creates a self-model: the network learns how its own
    forward passes affect the physical hardware.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 num_classes: int,
                 body_state_dim: int = 32,
                 device: str = 'cuda'):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.body_state_dim = body_state_dim
        self.device = device

        # FF layers
        dims = [input_dim + num_classes] + hidden_dims
        self.ff_layers = nn.ModuleList([
            FFLayer(dims[i], dims[i+1])
            for i in range(len(dims) - 1)
        ])

        # Body state predictor: predicts delta body state from layer activations
        # This is the "self-model" - predicting how compute affects the GPU
        total_hidden = sum(hidden_dims)
        self.body_predictor = nn.Sequential(
            nn.Linear(total_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, body_state_dim)
        )
        self.body_predictor_optimizer = torch.optim.Adam(
            self.body_predictor.parameters(), lr=0.001
        )

        # Move to device
        self.to(device)

        # History for embodied learning
        self.activation_history = []
        self.body_state_history = []

    def embed_label(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Embed labels into first 10 dimensions of input."""
        batch_size = x.shape[0]

        # One-hot encode labels
        one_hot = torch.zeros(batch_size, self.num_classes, device=x.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Concatenate
        return torch.cat([one_hot, x], dim=1)

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward through all layers, returning activations."""
        activations = []
        h = x
        for layer in self.ff_layers:
            h = layer(h)
            activations.append(h)
        return activations

    def predict_body_state_delta(self, activations: List[torch.Tensor]) -> torch.Tensor:
        """Predict body state change from layer activations."""
        # Concatenate all activations
        concat = torch.cat(activations, dim=1)
        # Pool over batch
        pooled = concat.mean(dim=0, keepdim=True)
        # Predict
        return self.body_predictor(pooled)

    def train_ff_step(self,
                      x: torch.Tensor,
                      labels: torch.Tensor,
                      pre_body_state: torch.Tensor,
                      post_body_state: torch.Tensor) -> Tuple[List[FFLayerMetrics], float]:
        """
        Combined FF + embodied training step.

        Args:
            x: Input data [batch, input_dim]
            labels: Correct labels [batch]
            pre_body_state: Body state before compute [32]
            post_body_state: Body state after compute [32]

        Returns:
            Layer metrics and body prediction error
        """
        batch_size = x.shape[0]

        # Create positive examples (correct labels)
        x_pos = self.embed_label(x, labels)

        # Create negative examples (random wrong labels)
        wrong_labels = torch.randint(0, self.num_classes, (batch_size,), device=x.device)
        # Ensure they're different
        mask = wrong_labels == labels
        wrong_labels[mask] = (wrong_labels[mask] + 1) % self.num_classes
        x_neg = self.embed_label(x, wrong_labels)

        # Train each layer
        layer_metrics = []
        h_pos, h_neg = x_pos, x_neg
        all_activations = []

        for idx, layer in enumerate(self.ff_layers):
            metrics, h_pos, h_neg = layer.train_step(h_pos, h_neg)
            metrics.layer_idx = idx
            layer_metrics.append(metrics)
            all_activations.append(h_pos)

        # Train body state predictor
        self.body_predictor_optimizer.zero_grad()

        # Predict body state delta
        predicted_delta = self.predict_body_state_delta(all_activations)
        actual_delta = (post_body_state - pre_body_state).unsqueeze(0)

        # MSE loss for body prediction
        body_loss = F.mse_loss(predicted_delta, actual_delta)
        body_loss.backward()
        self.body_predictor_optimizer.step()

        body_pred_error = body_loss.item()

        return layer_metrics, body_pred_error

    def predict_label(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict labels by testing all possible label embeddings.

        Returns the label that produces highest total goodness.
        """
        batch_size = x.shape[0]

        all_goodness = []

        for label in range(self.num_classes):
            # Create label tensor
            labels = torch.full((batch_size,), label, device=x.device, dtype=torch.long)
            x_labeled = self.embed_label(x, labels)

            # Forward through all layers
            h = x_labeled
            total_goodness = torch.zeros(batch_size, device=x.device)

            for layer in self.ff_layers:
                h = layer(h)
                total_goodness += layer.compute_goodness(h)

            all_goodness.append(total_goodness)

        # Stack and find max
        goodness_matrix = torch.stack(all_goodness, dim=1)  # [batch, num_classes]
        predictions = goodness_matrix.argmax(dim=1)

        return predictions


class EmbodiedFFTrainer:
    """
    Trainer that combines Forward-Forward with real GPU embodiment.
    """

    def __init__(self,
                 network: EmbodiedFFNetwork,
                 interface: UnifiedEmbodiedInterface,
                 device: str = 'cuda'):

        self.network = network
        self.interface = interface
        self.device = device

        # Metrics history
        self.metrics_history = []
        self.step_count = 0

        # Running averages
        self.power_avg = deque(maxlen=100)
        self.temp_avg = deque(maxlen=100)
        self.accuracy_avg = deque(maxlen=100)

    def train_batch(self,
                    x: torch.Tensor,
                    labels: torch.Tensor,
                    epoch: int,
                    batch_idx: int) -> EmbodiedFFMetrics:
        """
        Train on one batch with embodied feedback.
        """
        # Get pre-compute body state
        pre_state = self.interface.get_body_state()
        pre_tensor = pre_state.to_tensor(self.device)

        # Time the compute
        start_time = time.time()

        # Forward-Forward training step
        layer_metrics, body_pred_error = self.network.train_ff_step(
            x, labels, pre_tensor, pre_tensor  # Will update post after
        )

        # Sync GPU
        torch.cuda.synchronize()
        compute_ms = (time.time() - start_time) * 1000

        # Get post-compute body state
        post_state = self.interface.get_body_state()
        post_tensor = post_state.to_tensor(self.device)

        # Now do the actual body state prediction training with real post state
        _, body_pred_error = self.network.train_ff_step(
            x, labels, pre_tensor, post_tensor
        )

        # Compute accuracy
        with torch.no_grad():
            predictions = self.network.predict_label(x)
            accuracy = (predictions == labels).float().mean().item()

        # Compute energy (approximate from power * time)
        avg_power = (pre_state.gpu_power_w + post_state.gpu_power_w) / 2
        energy_mj = avg_power * (compute_ms / 1000) * 1000  # mJ

        # Update running averages
        self.power_avg.append(post_state.gpu_power_w)
        self.temp_avg.append(post_state.gpu_temp_edge_c)
        self.accuracy_avg.append(accuracy)

        # Create metrics
        metrics = EmbodiedFFMetrics(
            step=self.step_count,
            epoch=epoch,
            batch_idx=batch_idx,
            layer_metrics=[asdict(m) for m in layer_metrics],
            total_pos_goodness=sum(m.pos_goodness for m in layer_metrics),
            total_neg_goodness=sum(m.neg_goodness for m in layer_metrics),
            total_loss=sum(m.loss for m in layer_metrics),
            accuracy=accuracy,
            gpu_power_w=post_state.gpu_power_w,
            gpu_temp_c=post_state.gpu_temp_edge_c,
            gpu_util_pct=post_state.gpu_busy_pct,
            power_delta=post_state.gpu_power_w - pre_state.gpu_power_w,
            temp_delta=post_state.gpu_temp_edge_c - pre_state.gpu_temp_edge_c,
            energy_mj=energy_mj,
            compute_ms=compute_ms,
            body_state_pred_error=body_pred_error,
            body_state_dims=32,
        )

        self.metrics_history.append(asdict(metrics))
        self.step_count += 1

        return metrics

    def train_epoch(self,
                    dataloader,
                    epoch: int,
                    print_every: int = 10) -> Dict:
        """Train for one epoch."""
        epoch_metrics = {
            'losses': [],
            'accuracies': [],
            'powers': [],
            'temps': [],
            'body_errors': [],
        }

        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(self.device)
            labels = labels.to(self.device)

            metrics = self.train_batch(x, labels, epoch, batch_idx)

            epoch_metrics['losses'].append(metrics.total_loss)
            epoch_metrics['accuracies'].append(metrics.accuracy)
            epoch_metrics['powers'].append(metrics.gpu_power_w)
            epoch_metrics['temps'].append(metrics.gpu_temp_c)
            epoch_metrics['body_errors'].append(metrics.body_state_pred_error)

            if batch_idx % print_every == 0:
                print(f"  Batch {batch_idx}: "
                      f"loss={metrics.total_loss:.3f} "
                      f"acc={metrics.accuracy:.3f} "
                      f"power={metrics.gpu_power_w:.1f}W "
                      f"temp={metrics.gpu_temp_c:.1f}°C "
                      f"body_err={metrics.body_state_pred_error:.4f}")

        return {
            'mean_loss': np.mean(epoch_metrics['losses']),
            'mean_accuracy': np.mean(epoch_metrics['accuracies']),
            'mean_power': np.mean(epoch_metrics['powers']),
            'mean_temp': np.mean(epoch_metrics['temps']),
            'mean_body_error': np.mean(epoch_metrics['body_errors']),
        }


def create_synthetic_dataset(n_samples: int, input_dim: int, num_classes: int, device: str):
    """Create simple synthetic dataset for testing."""
    # Create class-specific patterns
    class_centers = torch.randn(num_classes, input_dim, device=device)

    # Generate samples
    labels = torch.randint(0, num_classes, (n_samples,), device=device)
    x = class_centers[labels] + torch.randn(n_samples, input_dim, device=device) * 0.5

    # Normalize
    x = F.normalize(x, dim=1)

    return x, labels


def main():
    print("=" * 70)
    print("z1409: Forward-Forward Learning with Embodied GPU Telemetry")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Initialize embodied interface
    print("\n[1] Initializing embodied interface...")
    interface = create_embodied_interface(
        use_real_gpu=True,
        use_real_fpga=False,
        device=device
    )
    interface.connect()

    # Get initial body state
    initial_state = interface.get_body_state()
    print(f"    Initial GPU power: {initial_state.gpu_power_w:.1f}W")
    print(f"    Initial GPU temp: {initial_state.gpu_temp_edge_c:.1f}°C")

    # Create network
    print("\n[2] Creating Forward-Forward network...")
    input_dim = 784  # Like MNIST
    hidden_dims = [500, 500, 500]
    num_classes = 10

    network = EmbodiedFFNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        body_state_dim=32,
        device=device
    )

    total_params = sum(p.numel() for p in network.parameters())
    print(f"    Total parameters: {total_params:,}")
    print(f"    FF layers: {len(network.ff_layers)}")
    print(f"    Hidden dims: {hidden_dims}")

    # Create trainer
    trainer = EmbodiedFFTrainer(network, interface, device)

    # Create synthetic data
    print("\n[3] Creating synthetic dataset...")
    n_train = 10000
    n_test = 2000
    batch_size = 64

    x_train, y_train = create_synthetic_dataset(n_train, input_dim, num_classes, device)
    x_test, y_test = create_synthetic_dataset(n_test, input_dim, num_classes, device)

    # Create simple dataloader
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

    train_loader = SimpleDataLoader(x_train, y_train, batch_size)
    test_loader = SimpleDataLoader(x_test, y_test, batch_size)

    print(f"    Train samples: {n_train}")
    print(f"    Test samples: {n_test}")
    print(f"    Batch size: {batch_size}")

    # Training
    print("\n[4] Training with Forward-Forward + Embodied Feedback...")
    print("-" * 70)

    n_epochs = 5
    results = {
        'config': {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'num_classes': num_classes,
            'n_train': n_train,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
        },
        'epochs': [],
        'final_metrics': {},
    }

    training_start = time.time()

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        epoch_metrics = trainer.train_epoch(train_loader, epoch, print_every=25)

        print(f"\n  Epoch {epoch + 1} summary:")
        print(f"    Mean loss: {epoch_metrics['mean_loss']:.4f}")
        print(f"    Mean accuracy: {epoch_metrics['mean_accuracy']:.4f}")
        print(f"    Mean power: {epoch_metrics['mean_power']:.1f}W")
        print(f"    Mean temp: {epoch_metrics['mean_temp']:.1f}°C")
        print(f"    Body prediction error: {epoch_metrics['mean_body_error']:.4f}")

        results['epochs'].append(epoch_metrics)

    training_time = time.time() - training_start

    # Final evaluation
    print("\n[5] Final evaluation...")

    with torch.no_grad():
        correct = 0
        total = 0

        for x, labels in test_loader:
            predictions = network.predict_label(x)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]

        test_accuracy = correct / total

    print(f"    Test accuracy: {test_accuracy:.4f}")

    # Get final body state
    final_state = interface.get_body_state()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results['final_metrics'] = {
        'test_accuracy': test_accuracy,
        'training_time_s': training_time,
        'final_gpu_power_w': final_state.gpu_power_w,
        'final_gpu_temp_c': final_state.gpu_temp_edge_c,
        'total_steps': trainer.step_count,
        'avg_accuracy': np.mean([e['mean_accuracy'] for e in results['epochs']]),
        'avg_body_pred_error': np.mean([e['mean_body_error'] for e in results['epochs']]),
    }

    print(f"  Training time: {training_time:.1f}s")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Final GPU power: {final_state.gpu_power_w:.1f}W")
    print(f"  Final GPU temp: {final_state.gpu_temp_edge_c:.1f}°C")
    print(f"  Total training steps: {trainer.step_count}")
    print(f"  Body state prediction error: {results['final_metrics']['avg_body_pred_error']:.4f}")

    # Key insight
    print("\n" + "-" * 70)
    print("KEY INSIGHT: Forward-Forward + Embodied Learning")
    print("-" * 70)
    print("""
The network learns TWO things simultaneously:
1. Classification via Forward-Forward (local goodness maximization)
2. Self-model: predicting how its own forward passes affect GPU state

This creates genuine embodiment:
- The network's compute affects physical hardware (power, temp)
- The network learns to predict these effects
- Future work: Use body predictions to regulate behavior

Unlike backpropagation:
- Each layer learns independently (biologically plausible)
- No gradient flow through entire network
- Local learning rules only
""")

    # Save results
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1409_forward_forward_embodied.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Cleanup
    interface.disconnect()

    print("\n✓ Forward-Forward embodied learning complete!")
    return results


if __name__ == "__main__":
    main()
