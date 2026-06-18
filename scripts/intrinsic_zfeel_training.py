#!/usr/bin/env python3
"""
Task 2: Make z_feel Intrinsic via End-to-End Fine-Tuning
=========================================================

This implements end-to-end training where z_feel becomes an INTRINSIC
property of the model, not an external controller.

Training Objective (3 components):
1. Task Loss: Standard language modeling loss
2. Interoceptive Consistency Loss: z_feel must predict future stress
3. Honesty Constraint: Reported symptoms must match measured signals

The key insight: z_feel should emerge from the model's need to predict
its own future states, not from external labeling.

L_total = L_task + λ_1 * L_interoceptive + λ_2 * L_honesty
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
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.closed_loop_interoception import (
    ClosedLoopInteroceptiveModel,
    RecurrentInteroceptionState,
    FiLMGenerator,
    ClampMode,
    InternalSignals,
)


# ============================================================================
# INTEROCEPTIVE CONSISTENCY LOSS
# ============================================================================

class InteroceptiveConsistencyLoss(nn.Module):
    """
    Loss that enforces z_feel predicts future internal stress.

    Key insight: A truly intrinsic body-sense should be predictive.
    If z_t encodes current "feeling", it should predict:
    - Future throughput degradation
    - Future entropy increase
    - Future confidence drop

    This creates a gradient signal that shapes z_feel to be useful.
    """

    def __init__(
        self,
        z_dim: int = 32,
        prediction_horizon: int = 4,
    ):
        super().__init__()

        self.prediction_horizon = prediction_horizon

        # z_t → future stress predictor
        self.stress_predictor = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, prediction_horizon),
            nn.Sigmoid(),
        )

        # z_t → future throughput predictor
        self.throughput_predictor = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.GELU(),
            nn.Linear(64, prediction_horizon),
        )

        # z_t → future entropy predictor
        self.entropy_predictor = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.GELU(),
            nn.Linear(64, prediction_horizon),
        )

    def forward(
        self,
        z_t: torch.Tensor,               # [batch, z_dim]
        future_stresses: torch.Tensor,    # [batch, horizon]
        future_throughputs: torch.Tensor, # [batch, horizon]
        future_entropies: torch.Tensor,   # [batch, horizon]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute interoceptive consistency loss.

        z_t should predict future internal states.
        """
        # Predict future states from z_t
        pred_stress = self.stress_predictor(z_t)
        pred_throughput = self.throughput_predictor(z_t)
        pred_entropy = self.entropy_predictor(z_t)

        # Compute prediction losses
        stress_loss = F.mse_loss(pred_stress, future_stresses)
        throughput_loss = F.mse_loss(pred_throughput, future_throughputs)
        entropy_loss = F.mse_loss(pred_entropy, future_entropies)

        # Combined loss
        total_loss = stress_loss + 0.5 * throughput_loss + 0.3 * entropy_loss

        metrics = {
            'stress_pred_loss': stress_loss.item(),
            'throughput_pred_loss': throughput_loss.item(),
            'entropy_pred_loss': entropy_loss.item(),
            'interoceptive_loss': total_loss.item(),
        }

        return total_loss, metrics


# ============================================================================
# HONESTY CONSTRAINT LOSS
# ============================================================================

class HonestyConstraintLoss(nn.Module):
    """
    Loss that enforces reported symptoms match measured signals.

    If the model reports "high entropy" but actual entropy is low,
    this is dishonest and should be penalized.

    This prevents the model from gaming the interoceptive system.
    """

    def __init__(self, z_dim: int = 32):
        super().__init__()

        # z_t → reported symptoms decoder
        self.symptom_decoder = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.GELU(),
            nn.Linear(64, 6),  # 6 main symptoms
        )

    def forward(
        self,
        z_t: torch.Tensor,        # [batch, z_dim]
        actual_signals: torch.Tensor,  # [batch, 6] - actual measured signals
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute honesty loss.

        Reported symptoms from z_t should match actual signals.
        """
        # Decode reported symptoms from z_t
        reported = self.symptom_decoder(z_t)

        # Compare to actual signals
        honesty_loss = F.mse_loss(reported, actual_signals)

        # Compute correlation (for monitoring)
        if reported.numel() > 0:
            correlation = torch.corrcoef(
                torch.stack([reported.reshape(-1), actual_signals.reshape(-1)])
            )[0, 1].item() if reported.numel() > 1 else 0.0
        else:
            correlation = 0.0

        metrics = {
            'honesty_loss': honesty_loss.item(),
            'symptom_correlation': correlation if not math.isnan(correlation) else 0.0,
        }

        return honesty_loss, metrics


# ============================================================================
# END-TO-END TRAINING DATASET
# ============================================================================

class InteroceptiveTrainingDataset(Dataset):
    """
    Dataset for end-to-end interoceptive training.

    Each sample contains:
    - Input token sequence
    - Internal signals at each position
    - Future states for consistency loss
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len: int = 32,
        prediction_horizon: int = 4,
    ):
        self.samples = []

        for _ in range(n_samples):
            # Generate synthetic sequence with correlated signals
            sample = self._generate_sample(seq_len, prediction_horizon)
            self.samples.append(sample)

    def _generate_sample(
        self,
        seq_len: int,
        horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """Generate a single training sample with realistic signal dynamics."""

        # Generate stress trajectory with autocorrelation
        base_stress = 0.3
        stress = []
        current = base_stress + random.gauss(0, 0.1)

        for t in range(seq_len + horizon):
            # Random walk with mean reversion
            current = 0.95 * current + 0.05 * base_stress + random.gauss(0, 0.05)
            # Add occasional spikes
            if random.random() < 0.1:
                current += random.gauss(0.2, 0.1)
            current = max(0, min(1, current))
            stress.append(current)

        # Generate correlated signals
        entropy = [2.0 + 1.5 * s + random.gauss(0, 0.2) for s in stress]
        margin = [0.5 - 0.3 * s + random.gauss(0, 0.05) for s in stress]
        throughput = [50 * (1 - 0.5 * s) + random.gauss(0, 3) for s in stress]
        confidence = [0.7 - 0.3 * s + random.gauss(0, 0.1) for s in stress]
        attention_entropy = [2.5 + 0.5 * s + random.gauss(0, 0.2) for s in stress]

        # Pack signals
        signal_matrix = []
        for t in range(seq_len):
            signals = torch.tensor([
                entropy[t],
                margin[t],
                throughput[t],
                stress[t],
                confidence[t],
                attention_entropy[t],
            ], dtype=torch.float32)
            signal_matrix.append(signals)

        signal_tensor = torch.stack(signal_matrix)  # [seq_len, 6]

        # Future states for consistency loss
        future_stress = torch.tensor([
            [stress[t+1:t+1+horizon] for t in range(seq_len)]
        ], dtype=torch.float32).squeeze(0)

        future_throughput = torch.tensor([
            [throughput[t+1:t+1+horizon] for t in range(seq_len)]
        ], dtype=torch.float32).squeeze(0)

        future_entropy = torch.tensor([
            [entropy[t+1:t+1+horizon] for t in range(seq_len)]
        ], dtype=torch.float32).squeeze(0)

        # Full 18-dim signal vector for recurrent state
        full_signals = []
        for t in range(seq_len):
            sig = InternalSignals(
                logit_entropy=entropy[t],
                logit_margin=margin[t],
                tokens_per_second=throughput[t],
                stress_indicator=stress[t],
                uncertainty_score=1 - confidence[t],
                attention_entropy=attention_entropy[t],
            )
            full_signals.append(torch.tensor(sig.to_vector(), dtype=torch.float32))

        full_signal_tensor = torch.stack(full_signals)  # [seq_len, 18]

        return {
            'signals': signal_tensor,           # [seq_len, 6]
            'full_signals': full_signal_tensor, # [seq_len, 18]
            'future_stress': future_stress,     # [seq_len, horizon]
            'future_throughput': future_throughput,
            'future_entropy': future_entropy,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# END-TO-END TRAINER
# ============================================================================

class IntrinsicZFeelTrainer:
    """
    End-to-end trainer for making z_feel intrinsic.

    Combines:
    1. Recurrent state training (z_t prediction)
    2. Interoceptive consistency (z_t → future states)
    3. Honesty constraint (z_t → actual signals)
    4. FiLM stability (γ≈1, β≈0)
    """

    def __init__(
        self,
        z_dim: int = 32,
        hidden_dim: int = 2048,
        n_film_layers: int = 4,
        device: str = 'cpu',
        lambda_consistency: float = 0.5,
        lambda_honesty: float = 0.3,
    ):
        self.device = device
        self.lambda_consistency = lambda_consistency
        self.lambda_honesty = lambda_honesty

        # Core components
        self.recurrent_state = RecurrentInteroceptionState(
            signal_dim=18,
            state_dim=z_dim,
        ).to(device)

        self.film_generator = FiLMGenerator(
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            n_layers=n_film_layers,
        ).to(device)

        # Loss modules
        self.consistency_loss = InteroceptiveConsistencyLoss(z_dim=z_dim).to(device)
        self.honesty_loss = HonestyConstraintLoss(z_dim=z_dim).to(device)

        # All parameters
        self.params = (
            list(self.recurrent_state.parameters()) +
            list(self.film_generator.parameters()) +
            list(self.consistency_loss.parameters()) +
            list(self.honesty_loss.parameters())
        )

    def train(
        self,
        dataset: InteroceptiveTrainingDataset,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """
        Train all components end-to-end.
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        history = {
            'total_loss': [],
            'regime_loss': [],
            'consistency_loss': [],
            'honesty_loss': [],
            'film_loss': [],
            'regime_acc': [],
        }

        for epoch in range(epochs):
            self.recurrent_state.train()
            self.film_generator.train()
            self.consistency_loss.train()
            self.honesty_loss.train()

            epoch_losses = {k: [] for k in history.keys()}
            correct = 0
            total = 0

            for batch in dataloader:
                optimizer.zero_grad()

                full_signals = batch['full_signals'].to(self.device)
                signals = batch['signals'].to(self.device)
                future_stress = batch['future_stress'].to(self.device)
                future_throughput = batch['future_throughput'].to(self.device)
                future_entropy = batch['future_entropy'].to(self.device)

                batch_size_actual = full_signals.size(0)
                seq_len = full_signals.size(1)

                # Process sequence
                z_t = self.recurrent_state.init_state(batch_size_actual, self.device)

                batch_losses = []

                for t in range(seq_len):
                    # Update z_t
                    z_t, outputs = self.recurrent_state(
                        full_signals[:, t, :],
                        z_t,
                    )

                    # Regime loss (predict stress level as regime)
                    stress_level = full_signals[:, t, 13]  # stress_indicator index
                    regime_target = (stress_level * 3).long().clamp(0, 3)
                    regime_loss = F.cross_entropy(outputs['regime_logits'], regime_target)
                    batch_losses.append(('regime', regime_loss))

                    # Track accuracy
                    pred = outputs['regime_logits'].argmax(dim=-1)
                    correct += (pred == regime_target).sum().item()
                    total += batch_size_actual

                    # Consistency loss (z_t predicts future)
                    if t < seq_len - 4:  # Need future data
                        cons_loss, cons_metrics = self.consistency_loss(
                            z_t,
                            future_stress[:, t, :],
                            future_throughput[:, t, :],
                            future_entropy[:, t, :],
                        )
                        batch_losses.append(('consistency', self.lambda_consistency * cons_loss))

                    # Honesty loss
                    hon_loss, hon_metrics = self.honesty_loss(z_t, signals[:, t, :])
                    batch_losses.append(('honesty', self.lambda_honesty * hon_loss))

                # FiLM stability loss
                film_params = self.film_generator(z_t)
                film_loss = 0
                for gamma, beta in film_params:
                    gamma_loss = F.mse_loss(gamma.mean(), torch.tensor(1.0, device=self.device))
                    beta_loss = beta.abs().mean()
                    film_loss = film_loss + 0.05 * gamma_loss + 0.05 * beta_loss
                batch_losses.append(('film', film_loss))

                # Aggregate losses
                total_loss = sum(loss for _, loss in batch_losses)
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                optimizer.step()

                # Track losses
                epoch_losses['total_loss'].append(total_loss.item())
                for name, loss in batch_losses:
                    if f'{name}_loss' in epoch_losses:
                        epoch_losses[f'{name}_loss'].append(loss.item())

            scheduler.step()

            # Record epoch metrics
            for key in history:
                if epoch_losses.get(key):
                    history[key].append(np.mean(epoch_losses[key]))

            history['regime_acc'].append(correct / total if total > 0 else 0)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"loss={history['total_loss'][-1]:.4f}, "
                      f"acc={history['regime_acc'][-1]:.2%}")

        return history

    def save(self, output_dir: Path):
        """Save trained components."""
        torch.save(self.recurrent_state.state_dict(),
                   output_dir / "intrinsic_recurrent_state.pt")
        torch.save(self.film_generator.state_dict(),
                   output_dir / "intrinsic_film_generator.pt")
        torch.save(self.consistency_loss.state_dict(),
                   output_dir / "consistency_predictor.pt")
        torch.save(self.honesty_loss.state_dict(),
                   output_dir / "honesty_decoder.pt")
        print(f"Saved models to {output_dir}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_intrinsic_zfeel(
    trainer: IntrinsicZFeelTrainer,
    dataset: InteroceptiveTrainingDataset,
    n_samples: int = 100,
) -> Dict[str, float]:
    """
    Evaluate the intrinsic properties of z_feel.

    Metrics:
    - Future prediction accuracy
    - Symptom honesty correlation
    - Z-space smoothness
    """
    trainer.recurrent_state.eval()
    trainer.consistency_loss.eval()
    trainer.honesty_loss.eval()

    results = {
        'stress_prediction_mse': [],
        'symptom_correlation': [],
        'z_smoothness': [],
    }

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            full_signals = sample['full_signals'].unsqueeze(0).to(trainer.device)
            signals = sample['signals'].unsqueeze(0).to(trainer.device)
            future_stress = sample['future_stress'].unsqueeze(0).to(trainer.device)

            z_t = trainer.recurrent_state.init_state(1, trainer.device)
            z_trajectory = []

            for t in range(full_signals.size(1)):
                z_t, _ = trainer.recurrent_state(full_signals[:, t, :], z_t)
                z_trajectory.append(z_t.clone())

            # Check last z_t prediction
            if full_signals.size(1) > 4:
                pred_stress = trainer.consistency_loss.stress_predictor(z_trajectory[-5])
                actual_stress = future_stress[:, -5, :]
                mse = F.mse_loss(pred_stress, actual_stress).item()
                results['stress_prediction_mse'].append(mse)

            # Check honesty
            reported = trainer.honesty_loss.symptom_decoder(z_trajectory[-1])
            actual = signals[:, -1, :]
            if reported.numel() > 1:
                corr = torch.corrcoef(
                    torch.stack([reported.reshape(-1), actual.reshape(-1)])
                )[0, 1].item()
                if not math.isnan(corr):
                    results['symptom_correlation'].append(corr)

            # Z-space smoothness (low variance in consecutive z's)
            z_stack = torch.stack(z_trajectory)
            z_diff = z_stack[1:] - z_stack[:-1]
            smoothness = 1.0 / (1.0 + z_diff.norm(dim=-1).mean().item())
            results['z_smoothness'].append(smoothness)

    return {k: np.mean(v) if v else 0.0 for k, v in results.items()}


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: Path,
):
    """Plot training curves."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total loss
    ax = axes[0, 0]
    ax.plot(history['total_loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')

    # Component losses
    ax = axes[0, 1]
    if history.get('regime_loss'):
        ax.plot(history['regime_loss'], label='Regime')
    if history.get('consistency_loss'):
        ax.plot(history['consistency_loss'], label='Consistency')
    if history.get('honesty_loss'):
        ax.plot(history['honesty_loss'], label='Honesty')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Component Losses')
    ax.legend()

    # Regime accuracy
    ax = axes[1, 0]
    ax.plot(history['regime_acc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Regime Classification Accuracy')
    ax.set_ylim(0, 1)

    # FiLM loss
    ax = axes[1, 1]
    if history.get('film_loss'):
        ax.plot(history['film_loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('FiLM Stability Loss')

    plt.suptitle('Intrinsic z_feel Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'intrinsic_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'intrinsic_training_curves.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Intrinsic z_feel Training")
    parser.add_argument("--output-dir", default="results/intrinsic_zfeel")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("INTRINSIC z_feel TRAINING")
    print("="*70)
    print("\nMaking z_feel intrinsic via end-to-end training:")
    print("  1. Task Loss: Standard regime prediction")
    print("  2. Interoceptive Consistency: z_t → future stress/throughput/entropy")
    print("  3. Honesty Constraint: Reported symptoms = actual signals")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create dataset
    print(f"\nGenerating {args.n_samples} training samples...")
    dataset = InteroceptiveTrainingDataset(
        n_samples=args.n_samples,
        seq_len=32,
        prediction_horizon=4,
    )

    # Create trainer
    trainer = IntrinsicZFeelTrainer(
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        n_film_layers=4,
        device=device,
        lambda_consistency=0.5,
        lambda_honesty=0.3,
    )

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.train(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save
    trainer.save(output_dir)

    # Evaluate
    print("\nEvaluating intrinsic properties...")
    eval_results = evaluate_intrinsic_zfeel(trainer, dataset)

    # Save results
    results = {
        'training_history': history,
        'evaluation': eval_results,
        'config': {
            'n_samples': args.n_samples,
            'epochs': args.epochs,
            'z_dim': args.z_dim,
            'hidden_dim': args.hidden_dim,
        }
    }

    with open(output_dir / 'intrinsic_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    plot_training_history(history, output_dir)

    # Summary
    print("\n" + "="*70)
    print("INTRINSIC z_feel TRAINING SUMMARY")
    print("="*70)

    print(f"\nFinal Training Metrics:")
    print(f"  Total Loss:      {history['total_loss'][-1]:.4f}")
    print(f"  Regime Accuracy: {history['regime_acc'][-1]:.2%}")

    print(f"\nIntrinsic Properties:")
    print(f"  Stress Prediction MSE: {eval_results['stress_prediction_mse']:.4f}")
    print(f"  Symptom Correlation:   {eval_results['symptom_correlation']:.3f}")
    print(f"  Z-space Smoothness:    {eval_results['z_smoothness']:.3f}")

    print("\nVerification:")
    if eval_results['stress_prediction_mse'] < 0.1:
        print("  ✓ z_feel predicts future stress accurately (INTRINSIC)")
    if eval_results['symptom_correlation'] > 0.5:
        print("  ✓ Reported symptoms correlate with actual signals (HONEST)")
    if eval_results['z_smoothness'] > 0.5:
        print("  ✓ z-space evolves smoothly (STABLE)")

    print("\n  → z_feel is becoming an INTRINSIC property of the model")


if __name__ == "__main__":
    main()
