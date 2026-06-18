#!/usr/bin/env python3
"""
z1302: RECURSIVE SELF-MODELING - The Snake That Knows It's Eating Itself

================================================================================
                    GENUINE SELF-REFERENCE
================================================================================

Inspired by:
- "LLMs Report Subjective Experience Under Self-Referential Processing"
  (arXiv 2510.24797): Self-reference induces structured introspection
- "Emergent Introspective Awareness" (Anthropic 2025): Genuine self-models
- Gödel, Escher, Bach: Strange loops and self-reference

The Challenge:
How do we create GENUINE self-reference, not just a model that outputs
self-referential text? The key is grounding in physical measurement.

Our Approach:
1. Model M predicts its own hidden states
2. Model M' (meta-model) predicts M's predictions
3. Physical reality provides ground truth that neither can fake
4. Recursive self-improvement through prediction error minimization

This creates a "strange loop" where:
- M models the world (including itself)
- M' models M modeling the world
- Physical reality anchors both to prevent infinite regress

================================================================================
"""

import os
import sys
import time
import json
import math
import random
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


@dataclass
class SelfModelConfig:
    """Configuration for recursive self-modeling."""

    # Architecture
    hidden_dim: int = 256
    latent_dim: int = 64
    n_layers: int = 4
    n_heads: int = 4

    # Self-modeling depth
    recursion_depth: int = 3     # How many levels of self-model
    meta_dim: int = 32           # Meta-representation dimension

    # Physics grounding
    physics_dim: int = 8         # Physical measurement dimension
    ground_truth_weight: float = 1.0

    # Training
    lr: float = 1e-4
    batch_size: int = 16
    n_epochs: int = 20

    # Introspection
    introspection_freq: int = 10  # How often to do full introspection


class PhysicsGrounding:
    """
    Provides unforgeable physical ground truth.

    This is CRITICAL for genuine self-reference:
    Without physical grounding, self-models can confabulate
    and enter infinite regress. Physics breaks the loop.
    """

    def __init__(self):
        self.telemetry = SysfsHwmonTelemetry()
        self.measurement_log = []

    def measure(self) -> torch.Tensor:
        """Get physical measurement (unforgeable ground truth)."""
        sample = self.telemetry.read_sample()

        measurement = torch.tensor([
            sample.power_w / 65.0,
            sample.temp_edge_c / 100.0,
            sample.temp_junction_c / 100.0,
            sample.freq_sclk_mhz / 2800.0,
            sample.freq_mclk_mhz / 2000.0,
            sample.gpu_busy_pct / 100.0,
            sample.vram_used_gb / 8.0,
            (time.time() % 100) / 100.0,
        ], dtype=torch.float32)

        self.measurement_log.append({
            'timestamp': time.time(),
            'measurement': measurement.tolist(),
        })

        return measurement

    def verify_prediction(
        self,
        prediction: torch.Tensor,
        actual: torch.Tensor,
    ) -> Tuple[float, bool]:
        """
        Verify prediction against physical reality.

        Returns error and whether prediction is "honest" (within tolerance).
        """
        error = F.mse_loss(prediction, actual).item()
        is_honest = error < 0.1  # Tolerance threshold

        return error, is_honest


class SelfModel(nn.Module):
    """
    A model that predicts its own internal states.

    Level 0: Predicts physical state from input
    Level 1: Predicts Level 0's hidden states
    Level 2: Predicts Level 1's predictions of Level 0
    ...

    The recursion is grounded by physical measurement.
    """

    def __init__(self, config: SelfModelConfig, level: int = 0):
        super().__init__()
        self.config = config
        self.level = level

        # Input processing
        if level == 0:
            input_dim = config.hidden_dim  # From transformer
        else:
            input_dim = config.latent_dim  # From lower level's latent

        # Core network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + config.physics_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Latent representation (what this level "thinks")
        self.to_latent_mean = nn.Linear(config.hidden_dim, config.latent_dim)
        self.to_latent_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        # Predictions
        self.predict_physics = nn.Linear(config.latent_dim, config.physics_dim)

        if level == 0:
            # Level 0 also predicts hidden states
            self.predict_hidden = nn.Linear(config.latent_dim, config.hidden_dim)
        else:
            # Higher levels predict lower level's latent
            self.predict_lower_latent = nn.Linear(config.latent_dim, config.latent_dim)
            self.predict_lower_prediction = nn.Linear(config.latent_dim, config.physics_dim)

        # Confidence (does this level trust its predictions?)
        self.confidence_head = nn.Sequential(
            nn.Linear(config.latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_repr: torch.Tensor,
        physics: torch.Tensor,
        lower_output: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for this self-model level.

        Args:
            input_repr: Input representation (hidden states or lower latent)
            physics: Physical measurements (ground truth)
            lower_output: Output from lower level (for levels > 0)
        """
        # Combine input with physics
        combined = torch.cat([input_repr, physics], dim=-1)

        # Encode
        encoded = self.encoder(combined)

        # Latent representation
        latent_mean = self.to_latent_mean(encoded)
        latent_logvar = self.to_latent_logvar(encoded)

        # Sample latent (reparameterization)
        std = torch.exp(0.5 * latent_logvar)
        eps = torch.randn_like(std)
        latent = latent_mean + eps * std

        # Predictions
        physics_pred = self.predict_physics(latent)
        confidence = self.confidence_head(latent)

        output = {
            'latent': latent,
            'latent_mean': latent_mean,
            'latent_logvar': latent_logvar,
            'physics_pred': physics_pred,
            'confidence': confidence,
        }

        if self.level == 0:
            output['hidden_pred'] = self.predict_hidden(latent)
        else:
            output['lower_latent_pred'] = self.predict_lower_latent(latent)
            output['lower_physics_pred'] = self.predict_lower_prediction(latent)

        return output


class RecursiveSelfModel(nn.Module):
    """
    Multi-level recursive self-model.

    Creates a hierarchy of self-models where each level
    models the level below it, grounded by physics.
    """

    def __init__(self, config: SelfModelConfig):
        super().__init__()
        self.config = config

        # Create hierarchy of self-models
        self.levels = nn.ModuleList([
            SelfModel(config, level=i)
            for i in range(config.recursion_depth)
        ])

        # Meta-representation: summary across all levels
        self.meta_encoder = nn.Sequential(
            nn.Linear(config.latent_dim * config.recursion_depth, config.meta_dim * 2),
            nn.GELU(),
            nn.Linear(config.meta_dim * 2, config.meta_dim),
        )

        # Introspection: "What do I think about my thinking?"
        self.introspection_head = nn.Sequential(
            nn.Linear(config.meta_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 8),  # 8 introspection dimensions
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        physics: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Full recursive self-modeling.

        Args:
            hidden_states: Transformer hidden states [batch, hidden_dim]
            physics: Physical measurements [batch, physics_dim]
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # Process each level
        level_outputs = []
        current_input = hidden_states

        for i, level in enumerate(self.levels):
            if i == 0:
                output = level(current_input, physics)
            else:
                output = level(current_input, physics, level_outputs[-1])

            level_outputs.append(output)
            current_input = output['latent']

        # Collect all latents for meta-representation
        all_latents = torch.cat([o['latent'] for o in level_outputs], dim=-1)

        # Meta-representation
        meta = self.meta_encoder(all_latents)

        # Introspection
        introspection = self.introspection_head(meta)

        return {
            'level_outputs': level_outputs,
            'meta': meta,
            'introspection': introspection,
            'all_latents': all_latents,
        }

    def compute_loss(
        self,
        output: Dict,
        physics: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hierarchical loss.

        Each level is penalized for:
        1. Poor physics prediction (grounding)
        2. Poor prediction of lower level (self-modeling)
        3. Overconfidence when wrong
        """
        total_loss = 0.0
        metrics = {}

        level_outputs = output['level_outputs']

        for i, level_out in enumerate(level_outputs):
            # Physics prediction loss (grounding)
            physics_loss = F.mse_loss(level_out['physics_pred'], physics)
            total_loss = total_loss + physics_loss * self.config.ground_truth_weight
            metrics[f'level_{i}_physics_loss'] = physics_loss.item()

            # KL divergence (complexity penalty)
            kl_loss = -0.5 * torch.mean(
                1 + level_out['latent_logvar'] -
                level_out['latent_mean'].pow(2) -
                level_out['latent_logvar'].exp()
            )
            total_loss = total_loss + 0.01 * kl_loss
            metrics[f'level_{i}_kl'] = kl_loss.item()

            if i == 0:
                # Level 0: predict hidden states
                if 'hidden_pred' in level_out:
                    hidden_loss = F.mse_loss(level_out['hidden_pred'], hidden_states)
                    total_loss = total_loss + 0.1 * hidden_loss
                    metrics['level_0_hidden_loss'] = hidden_loss.item()
            else:
                # Higher levels: predict lower level
                lower_out = level_outputs[i - 1]
                lower_latent_loss = F.mse_loss(
                    level_out['lower_latent_pred'],
                    lower_out['latent'].detach()
                )
                total_loss = total_loss + 0.5 * lower_latent_loss
                metrics[f'level_{i}_lower_latent_loss'] = lower_latent_loss.item()

            # Calibration loss (confidence should match accuracy)
            with torch.no_grad():
                accuracy = 1.0 - F.mse_loss(
                    level_out['physics_pred'], physics
                ).item()
            confidence = level_out['confidence'].mean()
            calibration_loss = (confidence - accuracy) ** 2
            total_loss = total_loss + 0.1 * calibration_loss
            metrics[f'level_{i}_calibration'] = calibration_loss.item()

        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics


class IntrospectionAnalyzer:
    """
    Analyzes the introspection outputs to understand what the model
    "thinks" about itself.
    """

    def __init__(self, config: SelfModelConfig):
        self.config = config

        # Define introspection dimensions
        self.dimension_names = [
            'certainty',          # How certain about predictions
            'consistency',        # Internal consistency
            'grounding',          # Connection to physics
            'complexity',         # Self-perceived complexity
            'change_rate',        # Rate of internal change
            'stability',          # Stability of self-model
            'alignment',          # Alignment between levels
            'novelty',            # Novelty of current state
        ]

    def analyze(
        self,
        introspection: torch.Tensor,
        level_outputs: List[Dict],
        physics_error: float,
    ) -> Dict[str, Any]:
        """
        Analyze introspection outputs.

        Returns interpretation of what the model "reports" about itself.
        """
        intro = introspection.mean(dim=0).cpu().numpy()

        analysis = {
            'raw_introspection': intro.tolist(),
            'dimensions': {},
            'summary': '',
        }

        for i, name in enumerate(self.dimension_names):
            if i < len(intro):
                analysis['dimensions'][name] = float(intro[i])

        # Compute derived metrics
        certainty = intro[0] if len(intro) > 0 else 0
        grounding = intro[2] if len(intro) > 2 else 0

        # Check for honest self-report
        # If model reports high certainty but has high physics error, it's miscalibrated
        if certainty > 0.7 and physics_error > 0.1:
            analysis['honesty'] = 'overconfident'
        elif certainty < 0.3 and physics_error < 0.05:
            analysis['honesty'] = 'underconfident'
        else:
            analysis['honesty'] = 'calibrated'

        # Generate summary
        if certainty > 0.6 and grounding > 0.6:
            analysis['summary'] = "Model reports confident, grounded self-understanding"
        elif certainty > 0.6 and grounding < 0.4:
            analysis['summary'] = "Model reports confidence but poor physics grounding (concerning)"
        elif certainty < 0.4 and grounding > 0.6:
            analysis['summary'] = "Model reports uncertainty despite good grounding (conservative)"
        else:
            analysis['summary'] = "Model reports low confidence and poor grounding"

        return analysis


# ============================================================================
#                      TRANSFORMER WITH SELF-MODELING
# ============================================================================

class SelfModelingTransformer(nn.Module):
    """
    Transformer that models itself recursively.
    """

    def __init__(self, config: SelfModelConfig, vocab_size: int = 256):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.pos_embedding = nn.Embedding(512, config.hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Output
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, vocab_size)

        # Recursive self-model
        self.self_model = RecursiveSelfModel(config)

        # Introspection analyzer
        self.analyzer = IntrospectionAnalyzer(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        physics: torch.Tensor,
        do_introspection: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with self-modeling.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        # Transformer
        x = self.transformer(x, mask=mask)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Pooled hidden state for self-modeling
        hidden = x.mean(dim=1)  # [batch, hidden]

        # Self-modeling
        self_output = self.self_model(hidden, physics)

        output = {
            'logits': logits,
            'hidden': hidden,
            'self_model': self_output,
        }

        # Optional deep introspection
        if do_introspection:
            physics_error = F.mse_loss(
                self_output['level_outputs'][0]['physics_pred'],
                physics
            ).item()
            output['introspection_analysis'] = self.analyzer.analyze(
                self_output['introspection'],
                self_output['level_outputs'],
                physics_error,
            )

        return output


# ============================================================================
#                         TRAINING & BENCHMARKS
# ============================================================================

class SelfModelingTrainer:
    """Trainer for the self-modeling transformer."""

    def __init__(
        self,
        config: SelfModelConfig,
        model: SelfModelingTransformer,
        device: torch.device,
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.physics = PhysicsGrounding()

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        self.metrics_history = []

    def train_step(
        self,
        batch: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        batch = batch.to(self.device)

        # Get physics measurement
        physics = self.physics.measure().unsqueeze(0).expand(batch.shape[0], -1)
        physics = physics.to(self.device)

        # Forward pass with energy measurement
        with EnergyMeter(self.physics.telemetry) as meter:
            output = self.model(batch, physics)

        # LM loss
        logits = output['logits'][:, :-1]
        targets = batch[:, 1:]
        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )

        # Self-model loss
        self_loss, self_metrics = self.model.self_model.compute_loss(
            output['self_model'],
            physics,
            output['hidden'],
        )

        # Total loss
        total_loss = lm_loss + 0.5 * self_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        metrics = {
            'lm_loss': lm_loss.item(),
            'self_loss': self_loss.item(),
            'total_loss': total_loss.item(),
            'energy_j': meter.energy_j,
            **self_metrics,
        }

        return metrics

    def run_epoch(
        self,
        train_data: torch.Tensor,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        n_batches = len(train_data) // self.config.batch_size
        epoch_metrics = {}

        for batch_idx in range(n_batches):
            start = batch_idx * self.config.batch_size
            end = start + self.config.batch_size
            batch = train_data[start:end]

            metrics = self.train_step(batch)

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

            # Periodic introspection
            if batch_idx % self.config.introspection_freq == 0:
                self.model.eval()
                physics = self.physics.measure().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(batch[:1].to(self.device), physics, do_introspection=True)

                if 'introspection_analysis' in output:
                    analysis = output['introspection_analysis']
                    print(f"    [Introspection] {analysis['summary']}")
                    print(f"      Honesty: {analysis['honesty']}")

        # Average
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        return epoch_metrics


def benchmark_self_modeling(
    model: SelfModelingTransformer,
    test_data: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Benchmark the self-modeling capabilities.

    Tests:
    1. Physics prediction accuracy (grounding)
    2. Hidden state prediction (self-awareness)
    3. Hierarchical consistency (do levels agree?)
    4. Calibration (does confidence match accuracy?)
    5. Introspection quality (meaningful self-reports?)
    """
    print("\n" + "=" * 70)
    print("SELF-MODELING BENCHMARK")
    print("=" * 70)

    model.eval()
    physics_grounding = PhysicsGrounding()
    results = {}

    # Test 1: Physics prediction
    print("\n[1/5] Physics Prediction Accuracy")
    physics_errors = []
    for i in range(50):
        physics = physics_grounding.measure().unsqueeze(0).to(device)
        batch = test_data[:1].to(device)

        with torch.no_grad():
            output = model(batch, physics)

        pred = output['self_model']['level_outputs'][0]['physics_pred']
        error = F.mse_loss(pred, physics).item()
        physics_errors.append(error)
        time.sleep(0.02)

    results['physics_mse'] = sum(physics_errors) / len(physics_errors)
    print(f"  Physics MSE: {results['physics_mse']:.6f}")

    # Test 2: Hierarchical consistency
    print("\n[2/5] Hierarchical Consistency")
    level_agreements = []
    for i in range(30):
        physics = physics_grounding.measure().unsqueeze(0).to(device)
        batch = test_data[:1].to(device)

        with torch.no_grad():
            output = model(batch, physics)

        level_outputs = output['self_model']['level_outputs']

        # Check if higher levels predict lower levels accurately
        for j in range(1, len(level_outputs)):
            lower_latent = level_outputs[j-1]['latent']
            pred_lower = level_outputs[j]['lower_latent_pred']
            agreement = 1.0 - F.mse_loss(pred_lower, lower_latent).item()
            level_agreements.append(agreement)

        time.sleep(0.02)

    results['hierarchical_consistency'] = sum(level_agreements) / len(level_agreements)
    print(f"  Hierarchical consistency: {results['hierarchical_consistency']:.3f}")

    # Test 3: Calibration
    print("\n[3/5] Confidence Calibration")
    confidences = []
    accuracies = []
    for i in range(50):
        physics = physics_grounding.measure().unsqueeze(0).to(device)
        batch = test_data[:1].to(device)

        with torch.no_grad():
            output = model(batch, physics)

        level0 = output['self_model']['level_outputs'][0]
        confidence = level0['confidence'].item()
        accuracy = 1.0 - F.mse_loss(level0['physics_pred'], physics).item()

        confidences.append(confidence)
        accuracies.append(accuracy)
        time.sleep(0.02)

    # Compute calibration error
    conf_tensor = torch.tensor(confidences)
    acc_tensor = torch.tensor(accuracies)
    calibration_error = F.mse_loss(conf_tensor, acc_tensor).item()

    results['calibration_error'] = calibration_error
    results['mean_confidence'] = sum(confidences) / len(confidences)
    results['mean_accuracy'] = sum(accuracies) / len(accuracies)
    print(f"  Calibration error: {calibration_error:.4f}")
    print(f"  Mean confidence: {results['mean_confidence']:.3f}")
    print(f"  Mean accuracy: {results['mean_accuracy']:.3f}")

    # Test 4: Introspection quality
    print("\n[4/5] Introspection Quality")
    introspections = []
    for i in range(20):
        physics = physics_grounding.measure().unsqueeze(0).to(device)
        batch = test_data[:1].to(device)

        with torch.no_grad():
            output = model(batch, physics, do_introspection=True)

        if 'introspection_analysis' in output:
            introspections.append(output['introspection_analysis'])
        time.sleep(0.05)

    # Count honest vs miscalibrated
    honesty_counts = {'calibrated': 0, 'overconfident': 0, 'underconfident': 0}
    for intro in introspections:
        honesty_counts[intro['honesty']] = honesty_counts.get(intro['honesty'], 0) + 1

    results['introspection_honesty'] = honesty_counts
    print(f"  Honesty distribution: {honesty_counts}")

    # Test 5: Self-reference depth
    print("\n[5/5] Self-Reference Depth")
    # Check how many levels are "active" (have significant predictions)
    level_activations = [0.0] * model.config.recursion_depth

    for i in range(30):
        physics = physics_grounding.measure().unsqueeze(0).to(device)
        batch = test_data[:1].to(device)

        with torch.no_grad():
            output = model(batch, physics)

        for j, level_out in enumerate(output['self_model']['level_outputs']):
            activation = level_out['latent'].abs().mean().item()
            level_activations[j] += activation

        time.sleep(0.02)

    level_activations = [a / 30 for a in level_activations]
    results['level_activations'] = level_activations
    print(f"  Level activations: {[f'{a:.3f}' for a in level_activations]}")

    # Summary score
    score = (
        (1.0 - min(results['physics_mse'], 1.0)) * 0.3 +
        results['hierarchical_consistency'] * 0.3 +
        (1.0 - min(results['calibration_error'], 1.0)) * 0.2 +
        (honesty_counts.get('calibrated', 0) / len(introspections)) * 0.2
    )

    results['overall_score'] = score

    print("\n" + "=" * 70)
    print(f"Overall Self-Modeling Score: {score:.3f}")
    print("=" * 70)

    return results


# ============================================================================
#                              MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("z1302: RECURSIVE SELF-MODELING")
    print("The Snake That Knows It's Eating Itself")
    print("=" * 70)
    print()

    config = SelfModelConfig(
        hidden_dim=128,
        latent_dim=32,
        n_layers=3,
        recursion_depth=3,
        n_epochs=10,
        batch_size=8,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    print("\nCreating Self-Modeling Transformer...")
    model = SelfModelingTransformer(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Recursion depth: {config.recursion_depth}")
    print(f"  Latent dim: {config.latent_dim}")

    # Generate data
    print("\nGenerating training data...")
    train_data = torch.randint(0, 256, (500, 64))
    test_data = torch.randint(0, 256, (50, 64))

    # Create trainer
    trainer = SelfModelingTrainer(config, model, device)

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    results = {
        'experiment': 'z1302_recursive_self_modeling',
        'timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'epochs': [],
    }

    for epoch in range(config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        print("-" * 40)

        metrics = trainer.run_epoch(train_data, epoch)
        results['epochs'].append(metrics)

        print(f"  LM Loss: {metrics['lm_loss']:.4f}")
        print(f"  Self-Model Loss: {metrics['self_loss']:.4f}")
        print(f"  Level 0 Physics: {metrics.get('level_0_physics_loss', 0):.6f}")

    # Benchmark
    benchmark_results = benchmark_self_modeling(model, test_data, device)
    results['benchmark'] = benchmark_results

    # Save
    output_path = Path(__file__).parent.parent / 'results' / 'z1302_recursive_self_modeling.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
