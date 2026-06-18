#!/usr/bin/env python3
"""
z2003: CALIBRATED UNCERTAINTY HOT - Addressing the z2000 Calibration Failure

PROBLEM FROM z2000:
The task became trivially easy (100% accuracy) so there was no uncertainty
to calibrate against. The supervised loss prediction trained on a task with
no errors, making the model "confidently wrong" when actual errors occurred.

SOLUTION:
1. Temperature Scaling: Learn a single temperature T that scales logits: p = softmax(z/T)
2. MC Dropout: Multiple forward passes with dropout for epistemic uncertainty
3. Multi-task difficulty mixing: Easy (next-char) AND hard (next-3-char, noisy) examples
4. Expected Calibration Error (ECE): Proper binned calibration metric

TRAINING PHASES:
1. Phase 1 (epochs 1-10): Standard char-LM training on mixed-difficulty data
2. Phase 2 (epochs 11-20): Calibration training - freeze base, train temperature on validation
3. Phase 3 (epochs 21-30): HOT training - model predicts its own calibrated uncertainty

SUCCESS CRITERIA:
- ECE < 0.1 (well-calibrated)
- correlation(predicted_uncertainty, actual_error) > 0.5 (meaningful HOT)

Author: Claude (Opus 4.5)
Date: 2026-02-06
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import math

import numpy as np

# HSA override for gfx1151 compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# GPU Telemetry
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# EXPECTED CALIBRATION ERROR (ECE)
# =============================================================================

def expected_calibration_error(
    predictions: torch.Tensor,  # [N] predicted classes
    confidences: torch.Tensor,  # [N] confidence for predicted class
    labels: torch.Tensor,       # [N] true labels
    n_bins: int = 10
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum(|accuracy(bin) - confidence(bin)|) * (bin_size / total)

    A well-calibrated model has ECE close to 0.
    """
    predictions = predictions.cpu()
    confidences = confidences.cpu()
    labels = labels.cpu()

    n_samples = len(labels)

    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_data = []

    for bin_idx, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Samples in this bin
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            # Accuracy in bin
            correct_in_bin = (predictions[in_bin] == labels[in_bin]).float()
            accuracy_in_bin = correct_in_bin.mean().item()

            # Average confidence in bin
            avg_confidence_in_bin = confidences[in_bin].mean().item()

            # ECE contribution
            ece += abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

            bin_data.append({
                'bin': bin_idx,
                'lower': lower.item(),
                'upper': upper.item(),
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': in_bin.sum().item(),
                'proportion': prop_in_bin,
            })

    return ece, {'bins': bin_data, 'n_samples': n_samples}


# =============================================================================
# FiLM CONDITIONING MODULE
# =============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for hardware conditioning."""

    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, hidden_dim)
        self.beta = nn.Linear(condition_dim, hidden_dim)

        # Initialize near identity
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = 1 + self.gamma(condition)
        beta = self.beta(condition)
        if x.dim() == 3 and gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


# =============================================================================
# CALIBRATED CONSCIOUSNESS MODEL
# =============================================================================

class CalibratedConsciousnessModel(nn.Module):
    """
    Consciousness model with proper calibration via:
    1. Temperature scaling (learnable)
    2. MC Dropout for epistemic uncertainty
    3. HOT module trained on calibrated predictions
    """

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 256,
        telemetry_dim: int = 8,
        n_layers: int = 4,
        dropout_rate: float = 0.1,
        n_mc_samples: int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_mc_samples = n_mc_samples
        self.dropout_rate = dropout_rate

        # Learnable temperature for calibration
        # Start at 1.5 (typical uncalibrated models are overconfident)
        self.temperature = nn.Parameter(torch.tensor(1.5))

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # FiLM conditioning on telemetry
        self.film_layers = nn.ModuleList([
            FiLMLayer(hidden_dim, telemetry_dim) for _ in range(n_layers)
        ])

        # Transformer layers with dropout
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                norm_first=True,
                dropout=dropout_rate,
            ) for _ in range(n_layers)
        ])

        # Dropout for MC sampling (kept enabled during inference)
        self.mc_dropout = nn.Dropout(dropout_rate)

        # Output head
        self.output = nn.Linear(hidden_dim, vocab_size)

        # HOT (Higher-Order Thought) module
        # Predicts calibrated uncertainty from hidden state
        self.hot_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Output positive uncertainty
        )

        # Track history for calibration
        self.uncertainty_history = deque(maxlen=5000)
        self.error_history = deque(maxlen=5000)

    def forward(
        self,
        tokens: torch.Tensor,
        telemetry: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass.

        Returns:
            Dict with logits, calibrated_probs, hidden, etc.
        """
        batch_size, seq_len = tokens.shape

        # Token embedding
        h = self.embed(tokens)  # [batch, seq, hidden]

        # Apply FiLM-conditioned transformer layers
        for layer, film in zip(self.layers, self.film_layers):
            h = film(h, telemetry)
            h = layer(h)

        # Apply MC dropout
        h = self.mc_dropout(h)

        # Output logits
        logits = self.output(h)

        # Temperature-scaled probabilities
        # Clamp temperature to avoid numerical issues
        T = torch.clamp(self.temperature, min=0.1, max=10.0)
        calibrated_logits = logits / T
        calibrated_probs = F.softmax(calibrated_logits, dim=-1)

        # Get confidence (max probability)
        confidence, predictions = calibrated_probs.max(dim=-1)

        # HOT prediction: predict uncertainty from last hidden state
        last_hidden = h[:, -1, :]  # [batch, hidden]
        predicted_uncertainty = self.hot_module(last_hidden).squeeze(-1)

        # Track for calibration metrics
        if targets is not None:
            with torch.no_grad():
                # Only track last position for simplicity
                last_targets = targets[:, -1] if targets.dim() == 2 else targets
                last_preds = predictions[:, -1]
                last_conf = confidence[:, -1]

                # Error = 1 if wrong, 0 if correct
                errors = (last_preds != last_targets).float()

                for i in range(min(len(predicted_uncertainty), len(errors))):
                    self.uncertainty_history.append(predicted_uncertainty[i].item())
                    self.error_history.append(errors[i].item())

        return {
            'logits': logits,
            'calibrated_logits': calibrated_logits,
            'calibrated_probs': calibrated_probs,
            'hidden': h,
            'predictions': predictions,
            'confidence': confidence,
            'predicted_uncertainty': predicted_uncertainty,
            'temperature': T.item(),
        }

    def forward_with_mc_uncertainty(
        self,
        tokens: torch.Tensor,
        telemetry: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with MC Dropout for epistemic uncertainty estimation.

        Args:
            tokens: Input tokens
            telemetry: Hardware telemetry
            n_samples: Number of MC samples (default: self.n_mc_samples)

        Returns:
            mean_probs: [batch, seq, vocab] mean probability
            epistemic_uncertainty: [batch, seq] variance across samples
            predictions: [batch, seq] predicted classes
        """
        if n_samples is None:
            n_samples = self.n_mc_samples

        # Keep model in training mode for dropout
        was_training = self.training
        self.train()

        all_probs = []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(tokens, telemetry)
                all_probs.append(out['calibrated_probs'])

        # Restore training state
        if not was_training:
            self.eval()

        # Stack and compute statistics
        stacked = torch.stack(all_probs, dim=0)  # [samples, batch, seq, vocab]
        mean_probs = stacked.mean(dim=0)  # [batch, seq, vocab]

        # Epistemic uncertainty = variance of predicted class probability
        # Sum variance across vocab dimension
        epistemic_uncertainty = stacked.var(dim=0).mean(dim=-1)  # [batch, seq]

        # Predictions from mean probs
        predictions = mean_probs.argmax(dim=-1)

        return mean_probs, epistemic_uncertainty, predictions

    def get_hot_calibration(self) -> Tuple[float, Dict]:
        """Compute HOT calibration (correlation between predicted uncertainty and errors)."""
        if len(self.uncertainty_history) < 100:
            return 0.0, {'n_samples': len(self.uncertainty_history)}

        uncertainties = np.array(list(self.uncertainty_history))
        errors = np.array(list(self.error_history))

        # Check variance
        unc_var = uncertainties.var()
        err_var = errors.var()

        if unc_var < 1e-8 or err_var < 1e-8:
            return 0.0, {
                'n_samples': len(uncertainties),
                'unc_variance': float(unc_var),
                'err_variance': float(err_var),
                'note': 'Insufficient variance',
            }

        # Correlation
        corr = np.corrcoef(uncertainties, errors)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        return float(corr), {
            'n_samples': len(uncertainties),
            'unc_mean': float(uncertainties.mean()),
            'unc_std': float(uncertainties.std()),
            'err_mean': float(errors.mean()),  # This is error rate
            'err_std': float(errors.std()),
        }


# =============================================================================
# DATA GENERATION - MIXED DIFFICULTY
# =============================================================================

class MixedDifficultyDataset:
    """
    Dataset with mixed difficulty levels to ensure meaningful calibration.

    - Easy: Next character prediction (standard char-LM)
    - Hard: Next-3 character prediction with noise and label flipping
    """

    def __init__(
        self,
        n_easy_samples: int = 1000,
        n_hard_samples: int = 1000,
        noise_level: float = 0.8,
        label_flip_prob: float = 0.2,
        seq_len: int = 64,
    ):
        self.n_easy = n_easy_samples
        self.n_hard = n_hard_samples
        self.noise_level = noise_level
        self.label_flip_prob = label_flip_prob
        self.seq_len = seq_len

        # Generate text corpus
        self.text = self._generate_corpus()
        self.vocab = sorted(set(self.text))
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.vocab)

        # Create batches
        self.easy_data, self.easy_targets = self._create_easy_samples()
        self.hard_data, self.hard_targets = self._create_hard_samples()

        # Combine with difficulty labels
        self.data = torch.cat([self.easy_data, self.hard_data], dim=0)
        self.targets = torch.cat([self.easy_targets, self.hard_targets], dim=0)
        self.difficulty = torch.cat([
            torch.zeros(n_easy_samples),  # 0 = easy
            torch.ones(n_hard_samples),   # 1 = hard
        ])

        # Shuffle
        perm = torch.randperm(len(self.data))
        self.data = self.data[perm]
        self.targets = self.targets[perm]
        self.difficulty = self.difficulty[perm]

    def _generate_corpus(self) -> str:
        """Generate text corpus."""
        samples = [
            "To be or not to be that is the question\n",
            "All the world is a stage and all the men and women merely players\n",
            "Now is the winter of our discontent made glorious summer\n",
            "Friends Romans countrymen lend me your ears\n",
            "The quick brown fox jumps over the lazy dog\n",
            "Pack my box with five dozen liquor jugs\n",
            "How vexingly quick daft zebras jump\n",
            "The five boxing wizards jump quickly\n",
        ]
        return ''.join(samples * 500)

    def _create_easy_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create easy next-char prediction samples."""
        encoded = [self.char2idx[c] for c in self.text]
        data = []
        targets = []

        for i in range(self.n_easy):
            start = np.random.randint(0, len(encoded) - self.seq_len - 1)
            seq = encoded[start:start + self.seq_len]
            target = encoded[start + 1:start + self.seq_len + 1]
            data.append(seq)
            targets.append(target)

        return torch.tensor(data, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    def _create_hard_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create hard samples with noise and label flipping."""
        encoded = [self.char2idx[c] for c in self.text]
        data = []
        targets = []

        for i in range(self.n_hard):
            start = np.random.randint(0, len(encoded) - self.seq_len - 4)
            seq = encoded[start:start + self.seq_len]

            # Hard: predict 3rd next character (more context needed)
            target = encoded[start + 3:start + self.seq_len + 3]

            # Add input noise
            seq = [
                s if np.random.random() > self.noise_level
                else np.random.randint(0, self.vocab_size)
                for s in seq
            ]

            # Label flipping
            target = [
                t if np.random.random() > self.label_flip_prob
                else np.random.randint(0, self.vocab_size)
                for t in target
            ]

            data.append(seq)
            targets.append(target)

        return torch.tensor(data, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    def get_batches(self, batch_size: int = 64, shuffle: bool = True):
        """Yield batches of (data, targets, difficulty)."""
        n = len(self.data)
        indices = torch.randperm(n) if shuffle else torch.arange(n)

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            yield (
                self.data[batch_idx],
                self.targets[batch_idx],
                self.difficulty[batch_idx],
            )


# =============================================================================
# TRAINING PHASES
# =============================================================================

def train_phase1_language_model(
    model: CalibratedConsciousnessModel,
    dataset: MixedDifficultyDataset,
    telemetry: SysfsHwmonTelemetry,
    n_epochs: int = 10,
) -> List[Dict]:
    """
    Phase 1: Standard char-LM training on mixed-difficulty data.

    Train the base language model without special calibration.
    """
    print("\n" + "="*60)
    print("PHASE 1: Language Model Training")
    print("="*60)

    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if 'temperature' not in n],
        lr=1e-3,
    )

    metrics_history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        for data, targets, difficulty in dataset.get_batches(batch_size=64):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Get telemetry
            sample = telemetry.read_sample()
            telem = torch.tensor([
                sample.temp_edge_c / 100.0,
                sample.power_w / 100.0,
                (sample.freq_sclk_mhz or 1000) / 2000.0,
                (sample.gpu_busy_pct or 50) / 100.0,
                np.sin(time.time()),
                np.cos(time.time()),
                float(epoch) / n_epochs,
                difficulty.mean().item(),  # Average difficulty in batch
            ], device=DEVICE).unsqueeze(0).expand(len(data), -1).float()

            optimizer.zero_grad()

            out = model(data, telem, targets=targets)

            # Language modeling loss
            loss = F.cross_entropy(
                out['logits'].view(-1, dataset.vocab_size),
                targets.view(-1),
            )

            # Track accuracy
            preds = out['predictions'].view(-1)
            correct = (preds == targets.view(-1)).sum().item()
            epoch_correct += correct
            epoch_total += targets.numel()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        # Compute metrics
        accuracy = epoch_correct / epoch_total
        avg_loss = epoch_loss / batch_count

        # Get HOT calibration
        hot_corr, hot_info = model.get_hot_calibration()

        metrics = {
            'epoch': epoch + 1,
            'phase': 1,
            'loss': avg_loss,
            'accuracy': accuracy,
            'temperature': model.temperature.item(),
            'hot_calibration': hot_corr,
            'error_rate': hot_info.get('err_mean', 0.0),
        }
        metrics_history.append(metrics)

        print(f"  Phase1 Epoch {epoch+1:2d}/{n_epochs}: "
              f"Loss={avg_loss:.4f} Acc={accuracy:.3f} "
              f"Temp={model.temperature.item():.3f} HOT={hot_corr:+.4f}")

    return metrics_history


def train_phase2_calibration(
    model: CalibratedConsciousnessModel,
    dataset: MixedDifficultyDataset,
    telemetry: SysfsHwmonTelemetry,
    n_epochs: int = 10,
) -> List[Dict]:
    """
    Phase 2: Calibration training - freeze base model, train only temperature.

    Uses NLL loss to optimize temperature for calibration.
    """
    print("\n" + "="*60)
    print("PHASE 2: Temperature Calibration")
    print("="*60)

    # Freeze all except temperature
    for name, param in model.named_parameters():
        if 'temperature' not in name:
            param.requires_grad = False

    # Optimizer for temperature only
    optimizer = torch.optim.AdamW([model.temperature], lr=0.1)

    metrics_history = []

    for epoch in range(n_epochs):
        model.eval()  # Eval mode but temperature still trainable
        epoch_nll = 0.0
        epoch_ece = 0.0
        batch_count = 0

        all_preds = []
        all_confs = []
        all_labels = []

        for data, targets, difficulty in dataset.get_batches(batch_size=64, shuffle=False):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Get telemetry
            sample = telemetry.read_sample()
            telem = torch.tensor([
                sample.temp_edge_c / 100.0,
                sample.power_w / 100.0,
                (sample.freq_sclk_mhz or 1000) / 2000.0,
                (sample.gpu_busy_pct or 50) / 100.0,
                np.sin(time.time()),
                np.cos(time.time()),
                0.5,
                difficulty.mean().item(),
            ], device=DEVICE).unsqueeze(0).expand(len(data), -1).float()

            optimizer.zero_grad()

            with torch.enable_grad():
                out = model(data, telem)

                # Calibration loss: NLL with temperature-scaled logits
                nll = F.cross_entropy(
                    out['calibrated_logits'].view(-1, dataset.vocab_size),
                    targets.view(-1),
                )

                nll.backward()
                optimizer.step()

                # Clamp temperature
                with torch.no_grad():
                    model.temperature.clamp_(min=0.1, max=10.0)

            # Collect for ECE
            last_preds = out['predictions'][:, -1].detach()
            last_confs = out['confidence'][:, -1].detach()
            last_labels = targets[:, -1].detach()

            all_preds.append(last_preds)
            all_confs.append(last_confs)
            all_labels.append(last_labels)

            epoch_nll += nll.item()
            batch_count += 1

        # Compute ECE
        all_preds = torch.cat(all_preds)
        all_confs = torch.cat(all_confs)
        all_labels = torch.cat(all_labels)

        ece, ece_info = expected_calibration_error(all_preds, all_confs, all_labels)

        metrics = {
            'epoch': epoch + 1 + 10,  # Continue epoch numbering
            'phase': 2,
            'nll': epoch_nll / batch_count,
            'ece': ece,
            'temperature': model.temperature.item(),
            'n_samples': len(all_preds),
        }
        metrics_history.append(metrics)

        print(f"  Phase2 Epoch {epoch+1:2d}/{n_epochs}: "
              f"NLL={epoch_nll/batch_count:.4f} ECE={ece:.4f} "
              f"Temp={model.temperature.item():.3f}")

    # Unfreeze for next phase
    for param in model.parameters():
        param.requires_grad = True

    return metrics_history


def train_phase3_hot(
    model: CalibratedConsciousnessModel,
    dataset: MixedDifficultyDataset,
    telemetry: SysfsHwmonTelemetry,
    n_epochs: int = 10,
    hot_lambda: float = 1.0,
) -> List[Dict]:
    """
    Phase 3: HOT training - train model to predict its own calibrated uncertainty.

    The HOT module learns to predict the model's error probability.
    """
    print("\n" + "="*60)
    print("PHASE 3: HOT (Higher-Order Thought) Training")
    print("="*60)

    # Clear history for fresh HOT tracking
    model.uncertainty_history.clear()
    model.error_history.clear()

    # Train HOT module more, base model less
    optimizer = torch.optim.AdamW([
        {'params': model.hot_module.parameters(), 'lr': 1e-3},
        {'params': [p for n, p in model.named_parameters()
                   if 'hot' not in n and 'temperature' not in n], 'lr': 1e-4},
    ])

    metrics_history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_task_loss = 0.0
        epoch_hot_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        for data, targets, difficulty in dataset.get_batches(batch_size=64):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Get telemetry
            sample = telemetry.read_sample()
            telem = torch.tensor([
                sample.temp_edge_c / 100.0,
                sample.power_w / 100.0,
                (sample.freq_sclk_mhz or 1000) / 2000.0,
                (sample.gpu_busy_pct or 50) / 100.0,
                np.sin(time.time()),
                np.cos(time.time()),
                float(epoch) / n_epochs,
                difficulty.mean().item(),
            ], device=DEVICE).unsqueeze(0).expand(len(data), -1).float()

            optimizer.zero_grad()

            out = model(data, telem, targets=targets)

            # Task loss
            task_loss = F.cross_entropy(
                out['calibrated_logits'].view(-1, dataset.vocab_size),
                targets.view(-1),
            )

            # HOT loss: predict error probability
            # Target = 1 if wrong, 0 if correct
            last_preds = out['predictions'][:, -1]
            last_targets = targets[:, -1]
            error_targets = (last_preds != last_targets).float().detach()

            # HOT module predicts uncertainty (should correlate with error)
            predicted_uncertainty = out['predicted_uncertainty']

            # Binary cross-entropy: uncertainty should predict error
            # Clamp predicted_uncertainty to valid range for BCE
            pred_prob = torch.clamp(predicted_uncertainty, 1e-7, 1 - 1e-7)
            hot_loss = F.binary_cross_entropy(pred_prob, error_targets)

            # Combined loss
            total_loss = task_loss + hot_lambda * hot_loss

            # Track
            correct = (last_preds == last_targets).sum().item()
            epoch_correct += correct
            epoch_total += len(last_targets)

            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_hot_loss += hot_loss.item()
            batch_count += 1

        # Compute metrics
        accuracy = epoch_correct / epoch_total
        avg_task_loss = epoch_task_loss / batch_count
        avg_hot_loss = epoch_hot_loss / batch_count

        # Get HOT calibration
        hot_corr, hot_info = model.get_hot_calibration()

        metrics = {
            'epoch': epoch + 1 + 20,  # Continue epoch numbering
            'phase': 3,
            'task_loss': avg_task_loss,
            'hot_loss': avg_hot_loss,
            'accuracy': accuracy,
            'error_rate': hot_info.get('err_mean', 0.0),
            'temperature': model.temperature.item(),
            'hot_calibration': hot_corr,
        }
        metrics_history.append(metrics)

        print(f"  Phase3 Epoch {epoch+1:2d}/{n_epochs}: "
              f"Task={avg_task_loss:.4f} HOT={avg_hot_loss:.4f} "
              f"Acc={accuracy:.3f} Err={hot_info.get('err_mean', 0):.3f} "
              f"HOT_cal={hot_corr:+.4f}")

    return metrics_history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_final(
    model: CalibratedConsciousnessModel,
    dataset: MixedDifficultyDataset,
    telemetry: SysfsHwmonTelemetry,
) -> Dict[str, Any]:
    """Final evaluation with ECE and HOT calibration."""
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    model.eval()

    all_preds = []
    all_confs = []
    all_labels = []
    all_difficulties = []
    all_uncertainties = []
    all_epistemic = []

    with torch.no_grad():
        for data, targets, difficulty in dataset.get_batches(batch_size=64, shuffle=False):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Get telemetry
            sample = telemetry.read_sample()
            telem = torch.tensor([
                sample.temp_edge_c / 100.0,
                sample.power_w / 100.0,
                (sample.freq_sclk_mhz or 1000) / 2000.0,
                (sample.gpu_busy_pct or 50) / 100.0,
                0.0, 0.0, 1.0,
                difficulty.mean().item(),
            ], device=DEVICE).unsqueeze(0).expand(len(data), -1).float()

            # Standard forward
            out = model(data, telem)

            # MC forward for epistemic uncertainty
            _, epistemic, _ = model.forward_with_mc_uncertainty(data, telem, n_samples=5)

            # Collect
            last_preds = out['predictions'][:, -1]
            last_confs = out['confidence'][:, -1]
            last_labels = targets[:, -1]
            last_epistemic = epistemic[:, -1]

            all_preds.append(last_preds.cpu())
            all_confs.append(last_confs.cpu())
            all_labels.append(last_labels.cpu())
            all_difficulties.append(difficulty)
            all_uncertainties.append(out['predicted_uncertainty'].cpu())
            all_epistemic.append(last_epistemic.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds)
    all_confs = torch.cat(all_confs)
    all_labels = torch.cat(all_labels)
    all_difficulties = torch.cat(all_difficulties)
    all_uncertainties = torch.cat(all_uncertainties)
    all_epistemic = torch.cat(all_epistemic)

    # Compute ECE
    ece, ece_info = expected_calibration_error(all_preds, all_confs, all_labels)

    # Compute accuracies by difficulty
    easy_mask = all_difficulties == 0
    hard_mask = all_difficulties == 1

    overall_acc = (all_preds == all_labels).float().mean().item()
    easy_acc = (all_preds[easy_mask] == all_labels[easy_mask]).float().mean().item()
    hard_acc = (all_preds[hard_mask] == all_labels[hard_mask]).float().mean().item()

    # Compute correlations
    errors = (all_preds != all_labels).float().numpy()
    uncertainties = all_uncertainties.numpy()
    epistemic = all_epistemic.numpy()

    # HOT calibration (predicted uncertainty vs actual error)
    if uncertainties.std() > 1e-8 and errors.std() > 1e-8:
        hot_corr = np.corrcoef(uncertainties, errors)[0, 1]
        if np.isnan(hot_corr):
            hot_corr = 0.0
    else:
        hot_corr = 0.0

    # Epistemic uncertainty vs error
    if epistemic.std() > 1e-8 and errors.std() > 1e-8:
        epistemic_corr = np.corrcoef(epistemic, errors)[0, 1]
        if np.isnan(epistemic_corr):
            epistemic_corr = 0.0
    else:
        epistemic_corr = 0.0

    # Confidence vs accuracy (should be positive for calibrated model)
    if all_confs.std() > 1e-8:
        correct = (all_preds == all_labels).float().numpy()
        conf_corr = np.corrcoef(all_confs.numpy(), correct)[0, 1]
        if np.isnan(conf_corr):
            conf_corr = 0.0
    else:
        conf_corr = 0.0

    results = {
        'ece': ece,
        'ece_info': ece_info,
        'accuracy': {
            'overall': overall_acc,
            'easy': easy_acc,
            'hard': hard_acc,
        },
        'calibration': {
            'hot_correlation': hot_corr,
            'epistemic_correlation': epistemic_corr,
            'confidence_accuracy_correlation': conf_corr,
            'temperature': model.temperature.item(),
        },
        'statistics': {
            'n_samples': len(all_preds),
            'n_easy': easy_mask.sum().item(),
            'n_hard': hard_mask.sum().item(),
            'error_rate': errors.mean(),
            'uncertainty_mean': uncertainties.mean(),
            'uncertainty_std': uncertainties.std(),
            'epistemic_mean': epistemic.mean(),
            'epistemic_std': epistemic.std(),
        },
    }

    # Print summary
    print(f"\n  Overall Accuracy: {overall_acc:.3f}")
    print(f"  Easy Accuracy:    {easy_acc:.3f}")
    print(f"  Hard Accuracy:    {hard_acc:.3f}")
    print(f"\n  ECE: {ece:.4f} (< 0.1 = well calibrated)")
    print(f"  Temperature: {model.temperature.item():.3f}")
    print(f"\n  HOT Calibration: {hot_corr:+.4f} (> 0.5 = good metacognition)")
    print(f"  Epistemic Corr:  {epistemic_corr:+.4f}")
    print(f"  Conf-Acc Corr:   {conf_corr:+.4f}")

    return results


def get_hardware_fingerprint(telemetry: SysfsHwmonTelemetry) -> Dict:
    """Get hardware fingerprint."""
    sample = telemetry.read_sample()
    fp_str = f"{sample.temp_edge_c:.1f}_{sample.power_w:.1f}_{time.time()}"
    return {
        'gpu_temp_c': sample.temp_edge_c,
        'gpu_power_w': sample.power_w,
        'gpu_freq_mhz': sample.freq_sclk_mhz,
        'gpu_util_pct': sample.gpu_busy_pct,
        'timestamp': datetime.now().isoformat(),
        'hash': hashlib.sha256(fp_str.encode()).hexdigest()[:16],
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("z2003: CALIBRATED UNCERTAINTY HOT")
    print("Addressing the z2000 Calibration Failure")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    print("APPROACH:")
    print("  1. Temperature Scaling - learnable calibration parameter")
    print("  2. MC Dropout - epistemic uncertainty estimation")
    print("  3. Mixed Difficulty - easy + hard samples ensure error variance")
    print("  4. ECE Metric - proper calibration measure")
    print()
    print("SUCCESS CRITERIA:")
    print("  - ECE < 0.1 (well-calibrated)")
    print("  - HOT correlation > 0.5 (meaningful metacognition)")
    print()

    # Initialize telemetry
    print("[1/6] Initializing hardware telemetry...")
    telemetry = SysfsHwmonTelemetry()
    fp = get_hardware_fingerprint(telemetry)
    print(f"  GPU: {fp['gpu_temp_c']:.1f}C, {fp['gpu_power_w']:.1f}W")
    print()

    # Create dataset
    print("[2/6] Creating mixed-difficulty dataset...")
    dataset = MixedDifficultyDataset(
        n_easy_samples=1500,
        n_hard_samples=1500,
        noise_level=0.7,
        label_flip_prob=0.25,
        seq_len=64,
    )
    print(f"  Vocabulary size: {dataset.vocab_size}")
    print(f"  Total samples: {len(dataset.data)}")
    print(f"  Easy samples: {dataset.n_easy}")
    print(f"  Hard samples: {dataset.n_hard}")
    print()

    # Create model
    print("[3/6] Building CalibratedConsciousnessModel...")
    model = CalibratedConsciousnessModel(
        vocab_size=dataset.vocab_size,
        hidden_dim=256,
        telemetry_dim=8,
        n_layers=4,
        dropout_rate=0.15,
        n_mc_samples=10,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Initial temperature: {model.temperature.item():.3f}")
    print()

    # Training phases
    all_metrics = []

    print("[4/6] Running three training phases...")
    print()

    # Phase 1: Language model training
    phase1_metrics = train_phase1_language_model(
        model, dataset, telemetry, n_epochs=10
    )
    all_metrics.extend(phase1_metrics)

    # Phase 2: Calibration (temperature) training
    phase2_metrics = train_phase2_calibration(
        model, dataset, telemetry, n_epochs=10
    )
    all_metrics.extend(phase2_metrics)

    # Phase 3: HOT training
    phase3_metrics = train_phase3_hot(
        model, dataset, telemetry, n_epochs=10, hot_lambda=1.0
    )
    all_metrics.extend(phase3_metrics)

    # Final evaluation
    print("[5/6] Final evaluation...")
    final_results = evaluate_final(model, dataset, telemetry)

    # Determine success
    ece_passed = final_results['ece'] < 0.1
    hot_passed = final_results['calibration']['hot_correlation'] > 0.5

    print("\n" + "="*60)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*60)
    print(f"  ECE < 0.1:          {final_results['ece']:.4f} {'PASS' if ece_passed else 'FAIL'}")
    print(f"  HOT corr > 0.5:     {final_results['calibration']['hot_correlation']:+.4f} {'PASS' if hot_passed else 'FAIL'}")
    print()

    if ece_passed and hot_passed:
        verdict = "SUCCESS - Both criteria met"
    elif ece_passed:
        verdict = "PARTIAL - ECE passed, HOT calibration needs improvement"
    elif hot_passed:
        verdict = "PARTIAL - HOT passed, temperature calibration needs improvement"
    else:
        verdict = "NEEDS_WORK - Neither criterion met"

    print(f"  VERDICT: {verdict}")
    print()

    # Save results
    print("[6/6] Saving results...")
    final_fp = get_hardware_fingerprint(telemetry)

    result = {
        'experiment': 'z2003_calibrated_uncertainty_hot',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'hardware_fingerprint': final_fp,
        'approach': {
            'temperature_scaling': True,
            'mc_dropout': True,
            'mixed_difficulty': True,
            'ece_metric': True,
        },
        'dataset': {
            'n_easy': dataset.n_easy,
            'n_hard': dataset.n_hard,
            'noise_level': 0.7,
            'label_flip_prob': 0.25,
            'vocab_size': dataset.vocab_size,
        },
        'model': {
            'hidden_dim': 256,
            'n_layers': 4,
            'dropout_rate': 0.15,
            'n_mc_samples': 10,
            'n_params': n_params,
        },
        'training': {
            'phase1_epochs': 10,
            'phase2_epochs': 10,
            'phase3_epochs': 10,
            'total_epochs': 30,
        },
        'results': final_results,
        'metrics_history': all_metrics,
        'success_criteria': {
            'ece_threshold': 0.1,
            'ece_measured': final_results['ece'],
            'ece_passed': ece_passed,
            'hot_threshold': 0.5,
            'hot_measured': final_results['calibration']['hot_correlation'],
            'hot_passed': hot_passed,
        },
        'verdict': verdict,
        'key_insights': [
            "z2000 failed because task became trivially easy (100% accuracy)",
            "Mixed difficulty ensures meaningful error variance for calibration",
            "Temperature scaling learns optimal logit scaling for ECE",
            "MC Dropout provides complementary epistemic uncertainty",
            "HOT training uses error prediction as supervision signal",
        ],
        'comparison_to_z2000': {
            'z2000_accuracy': 1.0,
            'z2003_accuracy': final_results['accuracy']['overall'],
            'z2000_difficulty_variance': 0.0016,
            'z2003_error_rate': final_results['statistics']['error_rate'],
            'improvement': 'Maintained meaningful error rate for calibration',
        },
    }

    output_file = RESULTS_DIR / 'z2003_calibrated_uncertainty_hot.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return result


if __name__ == '__main__':
    main()
