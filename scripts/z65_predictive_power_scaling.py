#!/usr/bin/env python3
"""
Z65: Predictive Power Scaling - Anticipate Optimal Mode
========================================================

Key insight: Reactive power control is too slow - by the time we measure
and adjust, the batch is already done. This experiment uses PREDICTIVE
power scaling:

1. PREDICT workload intensity from input characteristics
2. PRE-SET power mode before batch execution
3. LEARN from outcome to improve predictions

Architecture:
  Input tokens -> Workload Predictor -> Predicted intensity
  Predicted intensity + Temperature -> Power Mode Selection
  Execute batch at predicted power mode
  Measure actual metrics -> Update predictor

This enables PROACTIVE rather than REACTIVE power management.

Author: FEEL Research Team
Date: 2026-01-19
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# WORKLOAD PREDICTOR
# ============================================================
class WorkloadPredictor(nn.Module):
    """
    Predicts workload intensity from input tokens.

    Takes the first few tokens of input and predicts:
    - Estimated compute intensity (0-1)
    - Estimated memory pressure (0-1)
    - Recommended power mode (0-3)
    """

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.memory_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.power_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),  # + temp, current_power
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 power modes
        )

    def forward(
        self,
        input_tokens: torch.Tensor,  # (batch, seq_len)
        current_temp: torch.Tensor,  # (batch,)
        current_power: torch.Tensor,  # (batch,)
    ) -> Dict[str, torch.Tensor]:
        """Predict workload characteristics and optimal power mode."""

        # Embed tokens (use first 32 tokens for prediction)
        x = self.token_embed(input_tokens[:, :32])  # (batch, 32, hidden)
        _, (h, _) = self.lstm(x)  # h: (1, batch, hidden)
        h = h.squeeze(0)  # (batch, hidden)

        # Predict intensity and memory
        intensity = self.intensity_head(h)  # (batch, 1)
        memory = self.memory_head(h)  # (batch, 1)

        # Predict power mode (conditioned on temp and current power)
        context = torch.cat([
            h,
            current_temp.unsqueeze(1) / 100.0,  # Normalize temp
            current_power.unsqueeze(1) / 300.0,  # Normalize power
        ], dim=1)
        power_logits = self.power_head(context)  # (batch, 4)

        return {
            'intensity': intensity,
            'memory': memory,
            'power_logits': power_logits,
        }


# ============================================================
# PREDICTIVE METABOLIC TRANSFORMER
# ============================================================
class PredictiveMetabolicTransformer(nn.Module):
    """Transformer with predictive power scaling."""

    def __init__(self, base_model, predictor: WorkloadPredictor):
        super().__init__()
        self.base_model = base_model
        self.predictor = predictor
        self.config = base_model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: torch.Tensor,
        current_temp: torch.Tensor,
        current_power: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with workload prediction.

        Returns both LM outputs and power mode prediction.
        """
        # Predict workload and power mode
        pred = self.predictor(input_ids, current_temp, current_power)

        # Run base model with telemetry conditioning
        output = self.base_model(input_ids, telemetry)

        # Combine outputs
        return {
            'logits': output['logits'],
            'action_logits': output['action_logits'],
            'predicted_intensity': pred['intensity'],
            'predicted_memory': pred['memory'],
            'predicted_power_logits': pred['power_logits'],
        }


# ============================================================
# PREDICTIVE TRAINER
# ============================================================
class PredictiveTrainer:
    """Trainer with predictive power scaling."""

    def __init__(
        self,
        model: PredictiveMetabolicTransformer,
        telemetry,
        actuator,
        device,
    ):
        self.model = model
        self.telemetry = telemetry
        self.actuator = actuator
        self.device = device

        self.action_names = ['ECO', 'BALANCED', 'PERFORMANCE', 'MAX']
        self.prediction_history = []

    def train_predictive(
        self,
        dataloader,
        optimizer,
        predictor_optimizer,
        num_batches: int = 50,
    ) -> Dict:
        """Train with predictive power scaling."""

        self.model.train()
        total_lm_loss = 0
        total_pred_loss = 0
        prediction_accuracy = []

        for i, (input_ids, targets) in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            # Get current telemetry
            snap = self.telemetry.read()
            current_temp = torch.tensor([snap.temp_c], device=self.device).expand(input_ids.size(0))
            current_power = torch.tensor([snap.power_watts], device=self.device).expand(input_ids.size(0))

            body_state = self.telemetry.read_body_state()
            telem = torch.from_numpy(body_state).float().to(self.device)
            telem = telem.unsqueeze(0).expand(input_ids.size(0), -1)

            # Forward pass with prediction
            output = self.model(input_ids, telem, current_temp, current_power)

            # Get predicted power mode and apply BEFORE execution timing
            pred_power_probs = F.softmax(output['predicted_power_logits'][0], dim=-1)
            pred_action = torch.argmax(pred_power_probs).item()
            self.actuator.set_mode_from_action(pred_action)

            # Now measure actual execution
            batch_start = time.time()
            torch.cuda.synchronize()

            # Compute LM loss
            lm_loss = F.cross_entropy(
                output['logits'].view(-1, self.model.config.vocab_size),
                targets.view(-1)
            )

            torch.cuda.synchronize()
            batch_time = time.time() - batch_start

            # Get actual metrics
            snap_after = self.telemetry.read()
            actual_power = snap_after.power_watts
            actual_throughput = targets.numel() / batch_time
            actual_tpw = actual_throughput / max(actual_power, 1)

            # Determine what the OPTIMAL action would have been
            # (based on actual outcome - this is hindsight learning)
            optimal_action = self._determine_optimal_action(
                actual_tpw, snap_after.temp_c, pred_action
            )

            # Prediction loss (cross-entropy with optimal action)
            pred_loss = F.cross_entropy(
                output['predicted_power_logits'],
                torch.tensor([optimal_action], device=self.device).expand(input_ids.size(0))
            )

            # Track accuracy
            is_correct = pred_action == optimal_action
            prediction_accuracy.append(float(is_correct))

            # Combined backward
            total_loss = lm_loss + pred_loss * 0.5
            optimizer.zero_grad()
            predictor_optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            predictor_optimizer.step()

            total_lm_loss += lm_loss.item()
            total_pred_loss += pred_loss.item()

            self.prediction_history.append({
                'predicted': pred_action,
                'optimal': optimal_action,
                'correct': is_correct,
                'tpw': actual_tpw,
                'temp': snap_after.temp_c,
            })

            if i % 10 == 0:
                logger.info(
                    f"  B{i:3d} | Pred:{self.action_names[pred_action]:11s} | "
                    f"Opt:{self.action_names[optimal_action]:11s} | "
                    f"TPW:{actual_tpw:.1f} | T:{snap_after.temp_c:.0f}C | "
                    f"Acc:{np.mean(prediction_accuracy[-10:]):.1%}"
                )

        return {
            'avg_lm_loss': total_lm_loss / num_batches,
            'avg_pred_loss': total_pred_loss / num_batches,
            'prediction_accuracy': np.mean(prediction_accuracy),
        }

    def _determine_optimal_action(
        self,
        actual_tpw: float,
        actual_temp: float,
        executed_action: int,
    ) -> int:
        """
        Determine what the optimal action would have been.

        This is learned from experience - we track what TPW we get
        at each power mode and learn the mapping.
        """
        # Simple heuristic for now:
        # - If temp > 85: should have used ECO (0)
        # - If temp > 80 and TPW is good: BALANCED (1)
        # - If temp < 75 and TPW could be better: PERFORMANCE (2) or MAX (3)

        if actual_temp > 85:
            return 0  # ECO - too hot
        elif actual_temp > 80:
            return 1  # BALANCED - warm
        elif actual_tpw > 250:  # Good TPW
            return executed_action  # Current is fine
        elif actual_temp < 70:
            return 3  # MAX - cold and could go faster
        else:
            return 2  # PERFORMANCE - moderate

    def evaluate_prediction(self) -> Dict:
        """Evaluate prediction quality."""
        if not self.prediction_history:
            return {}

        predictions = [h['predicted'] for h in self.prediction_history]
        optimals = [h['optimal'] for h in self.prediction_history]

        accuracy = np.mean([p == o for p, o in zip(predictions, optimals)])

        # Per-action accuracy
        action_accuracy = {}
        for a in range(4):
            indices = [i for i, o in enumerate(optimals) if o == a]
            if indices:
                correct = sum(1 for i in indices if predictions[i] == a)
                action_accuracy[self.action_names[a]] = correct / len(indices)

        return {
            'overall_accuracy': accuracy,
            'action_accuracy': action_accuracy,
            'total_predictions': len(predictions),
        }


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_predictive_experiment():
    """Run predictive power scaling experiment."""

    from src.metabolic.film_transformer import MetabolicTransformer, MetabolicConfig
    from src.metabolic.telemetry_unified import UnifiedTelemetryReader
    from src.metabolic.actuation_unified import UnifiedActuator

    logger.info("=" * 60)
    logger.info("Z65: PREDICTIVE POWER SCALING")
    logger.info("Anticipate optimal power mode from input")
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/z65_predictive_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Initialize hardware
    telemetry = UnifiedTelemetryReader()
    actuator = UnifiedActuator()

    # Model config
    config = MetabolicConfig(
        vocab_size=256,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        max_seq_len=256,
    )

    # Create models
    base_model = MetabolicTransformer(config).to(device)
    predictor = WorkloadPredictor(vocab_size=256, hidden_dim=128).to(device)
    model = PredictiveMetabolicTransformer(base_model, predictor).to(device)

    logger.info(f"Base model params: {base_model.get_num_parameters():,}")
    logger.info(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    # Dataset
    from src.metabolic.metabolic_trainer import CharDataset
    corpus = "The quick brown fox jumps. " * 10000
    dataset = CharDataset(corpus, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Optimizers
    base_optimizer = torch.optim.AdamW(base_model.parameters(), lr=3e-4)
    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

    # Trainer
    trainer = PredictiveTrainer(model, telemetry, actuator, device)

    # ============================================================
    # Stage 1: LM Pretraining
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 1: LM Pretraining")
    logger.info("=" * 50)

    for epoch in range(2):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for i, (input_ids, targets) in enumerate(dataloader):
            if i >= 100:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)
            telem = torch.rand(input_ids.size(0), 12, device=device)
            current_temp = torch.ones(input_ids.size(0), device=device) * 70
            current_power = torch.ones(input_ids.size(0), device=device) * 200

            output = model(input_ids, telem, current_temp, current_power)
            loss = F.cross_entropy(
                output['logits'].view(-1, config.vocab_size),
                targets.view(-1)
            )

            base_optimizer.zero_grad()
            loss.backward()
            base_optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if i % 30 == 0:
                snap = telemetry.read()
                logger.info(f"  E{epoch+1} B{i} | Loss: {loss.item():.4f} | Power: {snap.power_watts:.0f}W")

        logger.info(f"  Epoch {epoch+1} | Avg Loss: {epoch_loss/num_batches:.4f}")

    # ============================================================
    # Stage 2: Predictive Training
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 2: Predictive Power Scaling Training")
    logger.info("=" * 50)

    for epoch in range(5):
        logger.info(f"\nEpoch {epoch + 1}/5")
        result = trainer.train_predictive(
            dataloader, base_optimizer, predictor_optimizer,
            num_batches=40
        )
        logger.info(f"  LM Loss: {result['avg_lm_loss']:.4f}")
        logger.info(f"  Pred Loss: {result['avg_pred_loss']:.4f}")
        logger.info(f"  Pred Accuracy: {result['prediction_accuracy']:.1%}")

    # Evaluate
    eval_result = trainer.evaluate_prediction()
    logger.info("\n" + "=" * 50)
    logger.info("PREDICTION EVALUATION")
    logger.info("=" * 50)
    logger.info(f"Overall accuracy: {eval_result['overall_accuracy']:.1%}")
    for action, acc in eval_result.get('action_accuracy', {}).items():
        logger.info(f"  {action}: {acc:.1%}")

    # ============================================================
    # Stage 3: Comparison vs Reactive
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 3: Predictive vs Reactive Comparison")
    logger.info("=" * 50)

    model.eval()

    # Test predictive mode
    logger.info("\nPredictive mode:")
    pred_metrics = {'tokens': 0, 'time': 0, 'energy': 0}

    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(dataloader):
            if i >= 30:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            snap = telemetry.read()
            current_temp = torch.tensor([snap.temp_c], device=device).expand(input_ids.size(0))
            current_power = torch.tensor([snap.power_watts], device=device).expand(input_ids.size(0))
            telem = torch.from_numpy(telemetry.read_body_state()).float().to(device)
            telem = telem.unsqueeze(0).expand(input_ids.size(0), -1)

            output = model(input_ids, telem, current_temp, current_power)

            # Apply predicted power BEFORE timing
            pred_action = torch.argmax(output['predicted_power_logits'][0]).item()
            actuator.set_mode_from_action(pred_action)

            start = time.time()
            _ = model(input_ids, telem, current_temp, current_power)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            snap_after = telemetry.read()
            energy = snap_after.power_watts * elapsed

            pred_metrics['tokens'] += targets.numel()
            pred_metrics['time'] += elapsed
            pred_metrics['energy'] += energy

    pred_tpw = (pred_metrics['tokens'] / pred_metrics['time']) / (pred_metrics['energy'] / pred_metrics['time'])
    pred_mj = pred_metrics['energy'] / pred_metrics['tokens'] * 1000

    logger.info(f"  Throughput/Watt: {pred_tpw:.2f}")
    logger.info(f"  mJ/token: {pred_mj:.3f}")

    # Test fixed mode for comparison
    logger.info("\nFixed PERFORMANCE mode:")
    from src.metabolic.actuation_unified import MetabolicMode
    actuator.set_metabolic_mode(MetabolicMode.PERFORMANCE)

    fixed_metrics = {'tokens': 0, 'time': 0, 'energy': 0}

    with torch.no_grad():
        for i, (input_ids, targets) in enumerate(dataloader):
            if i >= 30:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)
            telem = torch.rand(input_ids.size(0), 12, device=device)
            current_temp = torch.ones(input_ids.size(0), device=device) * 70
            current_power = torch.ones(input_ids.size(0), device=device) * 200

            snap_before = telemetry.read()
            start = time.time()
            output = base_model(input_ids, telem)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            snap_after = telemetry.read()

            avg_power = (snap_before.power_watts + snap_after.power_watts) / 2
            energy = avg_power * elapsed

            fixed_metrics['tokens'] += targets.numel()
            fixed_metrics['time'] += elapsed
            fixed_metrics['energy'] += energy

    actuator.reset_to_default()

    fixed_tpw = (fixed_metrics['tokens'] / fixed_metrics['time']) / (fixed_metrics['energy'] / fixed_metrics['time'])
    fixed_mj = fixed_metrics['energy'] / fixed_metrics['tokens'] * 1000

    logger.info(f"  Throughput/Watt: {fixed_tpw:.2f}")
    logger.info(f"  mJ/token: {fixed_mj:.3f}")

    # Comparison
    logger.info("\nComparison:")
    tpw_improvement = (pred_tpw - fixed_tpw) / fixed_tpw * 100
    mj_improvement = (fixed_mj - pred_mj) / fixed_mj * 100
    logger.info(f"  TPW improvement: {tpw_improvement:+.1f}%")
    logger.info(f"  mJ/token improvement: {mj_improvement:+.1f}%")

    # Save results
    results = {
        'timestamp': timestamp,
        'prediction_eval': eval_result,
        'predictive': {
            'throughput_per_watt': pred_tpw,
            'mj_per_token': pred_mj,
        },
        'fixed_performance': {
            'throughput_per_watt': fixed_tpw,
            'mj_per_token': fixed_mj,
        },
        'improvement': {
            'tpw_pct': tpw_improvement,
            'mj_pct': mj_improvement,
        },
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = run_predictive_experiment()
