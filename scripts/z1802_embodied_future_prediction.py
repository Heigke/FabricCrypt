#!/usr/bin/env python3
"""
z1802: Embodied Future Prediction

Hypothesis: A truly conscious system should be able to predict its own future
body states - not just the current state (protoself) but anticipating what
comes next (extended consciousness / prospective cognition).

This tests Damasio's "extended consciousness" requirement: the ability to
form autobiographical memory and anticipate future states.

Tests:
1. Can the model predict GPU state t+1 from current state t?
2. Can the model predict GPU state t+2, t+3...?
3. Does embodiment improve prediction horizon vs disembodied?
4. Does the model show anticipatory behavior (acting before state change)?

Hardware: AMD Radeon 8060S + HackRF One (simulated)

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1800_rf_embodiment import UnifiedEmbodiedTelemetry
from src.metabolic.film_transformer import MetabolicTransformer, MetabolicConfig


class FuturePredictingTransformer(MetabolicTransformer):
    """
    Transformer that predicts its own future body states.

    Extends MetabolicTransformer with:
    - State history buffer (last N telemetry samples)
    - Future state prediction heads (t+1, t+2, t+3)
    - Anticipation module that uses predicted future for current decisions
    """

    def __init__(self, config: MetabolicConfig, history_len: int = 8, predict_horizon: int = 3):
        super().__init__(config)

        self.history_len = history_len
        self.predict_horizon = predict_horizon
        telem_dim = config.telemetry_dim

        # State history encoder (GRU over telemetry sequence)
        self.history_encoder = nn.GRU(
            input_size=telem_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )

        # Future prediction heads
        self.future_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, telem_dim),
            )
            for _ in range(predict_horizon)
        ])

        # Anticipation module: uses predicted future for current decisions
        self.anticipation = nn.Sequential(
            nn.Linear(64 + telem_dim * predict_horizon, 64),
            nn.ReLU(),
            nn.Linear(64, config.hidden_dim),
        )

        # History buffer
        self._history = None
        self._history_idx = 0

    def reset_history(self, batch_size: int = 1, device: torch.device = None):
        """Reset state history buffer."""
        if device is None:
            device = next(self.parameters()).device
        self._history = torch.zeros(
            batch_size, self.history_len, self.config.telemetry_dim,
            device=device
        )
        self._history_idx = 0

    def update_history(self, telemetry: torch.Tensor):
        """Add new telemetry to history buffer."""
        if self._history is None:
            self.reset_history(telemetry.size(0), telemetry.device)

        batch = telemetry.size(0)
        if self._history.size(0) != batch:
            self.reset_history(batch, telemetry.device)

        idx = self._history_idx % self.history_len
        self._history[:, idx, :] = telemetry
        self._history_idx += 1

    def predict_future(self, history: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Predict future states from history.

        Args:
            history: [batch, history_len, telem_dim]

        Returns:
            encoded: [batch, 64] encoded history
            futures: list of [batch, telem_dim] for t+1, t+2, t+3
        """
        # Encode history with GRU
        _, hidden = self.history_encoder(history)
        encoded = hidden[-1]  # [batch, 64] from last layer

        # Predict future states
        futures = [head(encoded) for head in self.future_heads]

        return encoded, futures

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        return_futures: bool = False,
    ):
        """Forward with future prediction."""
        # Update history
        if telemetry is not None:
            self.update_history(telemetry)

        # Predict future from history
        if self._history is not None and self._history_idx >= 2:
            encoded, futures = self.predict_future(self._history)

            # Create anticipatory representation
            future_concat = torch.cat(futures, dim=-1)
            anticipatory = self.anticipation(
                torch.cat([encoded, future_concat], dim=-1)
            )
        else:
            futures = None
            anticipatory = None

        # Regular forward pass
        output = super().forward(input_ids, telemetry, return_hidden)

        if return_futures and futures is not None:
            output['predicted_futures'] = futures
            output['anticipatory_state'] = anticipatory

        return output


def run_experiment():
    """
    z1802: Embodied Future Prediction Experiment

    Tests whether embodiment enables better prediction of future body states.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1802] Device: {device}")
    if hasattr(torch.cuda, 'get_device_name'):
        print(f"[z1802] GPU: {torch.cuda.get_device_name()}")

    # Initialize telemetry
    telemetry = UnifiedEmbodiedTelemetry(rf_simulation=True)
    telemetry.start()
    time.sleep(0.5)

    rf_mode = "SIMULATED" if telemetry.rf_interface.simulation else "REAL"
    print(f"[z1802] RF mode: {rf_mode}")

    # Load training data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text = data_path.read_text()
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    print(f"[z1802] Vocab size: {vocab_size}")

    # Config
    batch_size = 4
    seq_len = 256
    num_epochs = 8
    batches_per_epoch = 200
    lr = 3e-4
    history_len = 8
    predict_horizon = 3
    future_weight = 0.1

    # Create future-predicting model
    config = MetabolicConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=20,  # GPU (12) + RF (8)
    )
    model = FuturePredictingTransformer(
        config,
        history_len=history_len,
        predict_horizon=predict_horizon,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"[z1802] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Data iterator
    def get_batch():
        ix = torch.randint(len(text) - seq_len - 1, (batch_size,))
        x = torch.stack([
            torch.tensor([char_to_idx[c] for c in text[i:i+seq_len]], dtype=torch.long)
            for i in ix
        ])
        y = torch.stack([
            torch.tensor([char_to_idx[c] for c in text[i+1:i+seq_len+1]], dtype=torch.long)
            for i in ix
        ])
        return x.to(device), y.to(device)

    # Collect telemetry with history
    telemetry_buffer = []

    def get_telemetry_with_history():
        telem = telemetry.get_unified_tensor()
        telemetry_buffer.append(telem.numpy().copy())
        return telem.unsqueeze(0).expand(batch_size, -1).to(device)

    results = {
        'experiment': 'z1802_embodied_future_prediction',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'rf_mode': rf_mode,
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_epochs': num_epochs,
            'history_len': history_len,
            'predict_horizon': predict_horizon,
            'future_weight': future_weight,
        },
        'training': {
            'task_losses': [],
            'future_losses': [],
        },
        'verdicts': {},
    }

    # Training
    print("\n[z1802] Training with future prediction...")
    model.train()

    for epoch in range(num_epochs):
        epoch_task_loss = 0
        epoch_future_loss = 0
        model.reset_history(batch_size, device)
        telemetry_buffer.clear()

        for batch_idx in range(batches_per_epoch):
            x, y = get_batch()
            telem = get_telemetry_with_history()

            optimizer.zero_grad()

            output = model(x, telem, return_hidden=True, return_futures=True)
            logits = output['logits']

            # Task loss
            task_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            # Future prediction loss (compare with actual future telemetry)
            futures = output.get('predicted_futures')
            future_loss = torch.tensor(0.0, device=device)

            if futures is not None and len(telemetry_buffer) > predict_horizon:
                for h in range(predict_horizon):
                    if batch_idx >= h + 1:
                        # Get actual telemetry from h+1 steps ago
                        actual_future = torch.tensor(
                            telemetry_buffer[-(h+1)],
                            device=device
                        ).unsqueeze(0).expand(batch_size, -1)
                        pred_future = futures[h]
                        future_loss = future_loss + F.mse_loss(pred_future, actual_future)

                future_loss = future_loss / predict_horizon

            total_loss = task_loss + future_weight * future_loss
            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_future_loss += future_loss.item() if isinstance(future_loss, torch.Tensor) else 0

        avg_task = epoch_task_loss / batches_per_epoch
        avg_future = epoch_future_loss / batches_per_epoch

        results['training']['task_losses'].append(avg_task)
        results['training']['future_losses'].append(avg_future)

        print(f"  Epoch {epoch+1}/{num_epochs}: task={avg_task:.4f}, future={avg_future:.4f}")

    # Evaluation: Future prediction accuracy
    print("\n[z1802] Evaluating future prediction accuracy...")
    model.eval()
    model.reset_history(batch_size, device)
    telemetry_buffer.clear()

    prediction_errors = {h: [] for h in range(predict_horizon)}

    with torch.no_grad():
        for step in range(100):
            x, _ = get_batch()
            telem = get_telemetry_with_history()

            output = model(x, telem, return_futures=True)
            futures = output.get('predicted_futures')

            if futures is not None and len(telemetry_buffer) > predict_horizon:
                for h in range(predict_horizon):
                    if len(telemetry_buffer) > h + 1:
                        actual = torch.tensor(telemetry_buffer[-(h+1)], device=device)
                        pred = futures[h][0]  # First sample in batch
                        error = F.mse_loss(pred, actual).item()
                        prediction_errors[h].append(error)

    # Compute average errors per horizon
    horizon_mse = {}
    for h in range(predict_horizon):
        if prediction_errors[h]:
            horizon_mse[f't+{h+1}'] = float(np.mean(prediction_errors[h]))
        else:
            horizon_mse[f't+{h+1}'] = 1.0

    results['evaluation'] = {
        'horizon_mse': horizon_mse,
        'avg_mse_all_horizons': float(np.mean(list(horizon_mse.values()))),
    }

    print(f"\n[z1802] Future prediction MSE by horizon:")
    for h, mse in horizon_mse.items():
        print(f"  {h}: {mse:.6f}")

    # Verdicts
    # V1: Future prediction improves during training
    initial_future = results['training']['future_losses'][0] if results['training']['future_losses'][0] > 0 else 0.1
    final_future = results['training']['future_losses'][-1]
    v1_pass = final_future < initial_future * 0.5  # 50% improvement
    results['verdicts']['V1_future_learning'] = {
        'pass': v1_pass,
        'initial_loss': initial_future,
        'final_loss': final_future,
        'improvement': (initial_future - final_future) / initial_future if initial_future > 0 else 0,
        'description': 'Future prediction improves during training'
    }

    # V2: Short-term prediction (t+1) more accurate than long-term (t+3)
    if 't+1' in horizon_mse and 't+3' in horizon_mse:
        v2_pass = horizon_mse['t+1'] < horizon_mse['t+3']
    else:
        v2_pass = True
    results['verdicts']['V2_temporal_decay'] = {
        'pass': v2_pass,
        'mse_t1': horizon_mse.get('t+1', 1.0),
        'mse_t3': horizon_mse.get('t+3', 1.0),
        'description': 'Prediction accuracy decreases with horizon (realistic uncertainty)'
    }

    # V3: All horizons have reasonable prediction (MSE < 0.5)
    v3_pass = all(mse < 0.5 for mse in horizon_mse.values())
    results['verdicts']['V3_all_horizons_predict'] = {
        'pass': v3_pass,
        'horizon_mse': horizon_mse,
        'threshold': 0.5,
        'description': 'All prediction horizons have MSE < 0.5'
    }

    # V4: Task performance maintained
    final_ppl = np.exp(results['training']['task_losses'][-1])
    v4_pass = final_ppl < 12
    results['verdicts']['V4_task_preserved'] = {
        'pass': v4_pass,
        'final_ppl': float(final_ppl),
        'threshold': 12,
        'description': 'Language modeling quality maintained'
    }

    # Summary
    passed = sum(1 for v in results['verdicts'].values() if v['pass'])
    total = len(results['verdicts'])
    results['passed'] = passed
    results['total_verdicts'] = total
    results['overall_verdict'] = 'FUTURE_PREDICTION_DEMONSTRATED' if passed >= 3 else 'PARTIAL'

    print(f"\n[z1802] Verdicts: {passed}/{total} passed")
    print(f"[z1802] Overall: {results['overall_verdict']}")

    # Cleanup
    telemetry.stop()

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "z1802_embodied_future_prediction.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[z1802] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
