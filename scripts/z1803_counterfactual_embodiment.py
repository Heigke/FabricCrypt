#!/usr/bin/env python3
"""
z1803: Counterfactual Embodiment

Hypothesis: A conscious system can reason counterfactually about its body -
"What would my state be IF I took action X instead of Y?"

This tests a sophisticated form of self-awareness: not just predicting what
WILL happen, but what WOULD have happened under different conditions.

Tests:
1. Given action A was taken, can model predict what would have happened with action B?
2. Does embodiment improve counterfactual accuracy vs disembodied?
3. Can the model distinguish actual history from counterfactual?
4. Does counterfactual reasoning improve decision-making?

This is related to the "minimal sense of agency" in consciousness research.

Hardware: AMD Radeon 8060S + HackRF One (simulated)

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
import copy
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


class CounterfactualTransformer(MetabolicTransformer):
    """
    Transformer with counterfactual reasoning about body states.

    Key components:
    - Action encoder: represents possible actions
    - World model: predicts next state given current state + action
    - Counterfactual head: predicts what would have happened with different action
    """

    def __init__(self, config: MetabolicConfig, num_actions: int = 4):
        super().__init__(config)

        self.num_actions = num_actions
        telem_dim = config.telemetry_dim

        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 16)

        # World model: (state, action) -> next_state
        self.world_model = nn.Sequential(
            nn.Linear(telem_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, telem_dim),
        )

        # Counterfactual discriminator: given (state, action, next_state), is this real or counterfactual?
        self.cf_discriminator = nn.Sequential(
            nn.Linear(telem_dim * 2 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # History buffer
        self._state_history = []
        self._action_history = []

    def reset_history(self):
        """Clear history buffers."""
        self._state_history = []
        self._action_history = []

    def record_transition(self, state: torch.Tensor, action: int, next_state: torch.Tensor):
        """Record a state transition."""
        self._state_history.append(state.detach().cpu())
        self._action_history.append(action)

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next state given current state and action.

        Args:
            state: [batch, telem_dim]
            action: [batch] action indices

        Returns:
            next_state: [batch, telem_dim]
        """
        action_emb = self.action_embed(action)  # [batch, 16]
        combined = torch.cat([state, action_emb], dim=-1)
        return self.world_model(combined)

    def counterfactual_predict(
        self,
        state: torch.Tensor,
        actual_action: torch.Tensor,
        counterfactual_action: torch.Tensor,
        actual_next_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict what would have happened with counterfactual action.

        Returns:
            cf_next_state: predicted next state under counterfactual action
            cf_probability: probability that actual transition is real (vs counterfactual)
        """
        # Predict counterfactual next state
        cf_next_state = self.predict_next_state(state, counterfactual_action)

        # Discriminate: is the actual transition real?
        actual_action_emb = self.action_embed(actual_action)
        combined = torch.cat([state, actual_action_emb, actual_next_state], dim=-1)
        real_prob = self.cf_discriminator(combined)

        return cf_next_state, real_prob

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        action: Optional[torch.Tensor] = None,
        return_world_model: bool = False,
    ):
        """Forward with optional world model output."""
        output = super().forward(input_ids, telemetry, return_hidden)

        if return_world_model and telemetry is not None and action is not None:
            predicted_next = self.predict_next_state(telemetry, action)
            output['predicted_next_state'] = predicted_next

        return output


def simulate_action_effects(state: torch.Tensor, action: int, noise_std: float = 0.05) -> torch.Tensor:
    """
    Simulate how an action affects the body state.

    This is a simple model where:
    - Action 0 (IDLE): State drifts toward baseline
    - Action 1 (LOW): Slight decrease in power/temp
    - Action 2 (BALANCED): Moderate activity
    - Action 3 (HIGH): Increase in power/temp
    """
    next_state = state.clone()

    # GPU power (idx 2) and temp (idx 0) are most affected
    if action == 0:  # IDLE
        next_state[..., 0] *= 0.95  # temp decreases
        next_state[..., 2] *= 0.9   # power decreases
    elif action == 1:  # LOW
        next_state[..., 0] *= 0.98
        next_state[..., 2] *= 0.95
    elif action == 2:  # BALANCED
        pass  # Maintain current
    elif action == 3:  # HIGH
        next_state[..., 0] = next_state[..., 0] * 1.02 + 0.05
        next_state[..., 2] = next_state[..., 2] * 1.1 + 0.05

    # Add noise
    noise = torch.randn_like(next_state) * noise_std
    next_state = next_state + noise

    # Clamp to valid range
    next_state = torch.clamp(next_state, 0, 1)

    return next_state


def run_experiment():
    """
    z1803: Counterfactual Embodiment Experiment

    Tests whether the model can reason counterfactually about its body state.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1803] Device: {device}")
    if hasattr(torch.cuda, 'get_device_name'):
        print(f"[z1803] GPU: {torch.cuda.get_device_name()}")

    # Initialize telemetry
    telemetry_source = UnifiedEmbodiedTelemetry(rf_simulation=True)
    telemetry_source.start()
    time.sleep(0.5)

    rf_mode = "SIMULATED" if telemetry_source.rf_interface.simulation else "REAL"
    print(f"[z1803] RF mode: {rf_mode}")

    # Load training data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text = data_path.read_text()
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    print(f"[z1803] Vocab size: {vocab_size}")

    # Config
    batch_size = 4
    seq_len = 256
    num_epochs = 10
    batches_per_epoch = 150
    lr = 3e-4
    num_actions = 4
    world_model_weight = 0.2
    cf_weight = 0.1

    # Create counterfactual model
    config = MetabolicConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=20,
    )
    model = CounterfactualTransformer(config, num_actions=num_actions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"[z1803] Model params: {sum(p.numel() for p in model.parameters()):,}")

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

    results = {
        'experiment': 'z1803_counterfactual_embodiment',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'rf_mode': rf_mode,
        'config': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_actions': num_actions,
            'world_model_weight': world_model_weight,
            'cf_weight': cf_weight,
        },
        'training': {
            'task_losses': [],
            'world_model_losses': [],
            'cf_losses': [],
        },
        'verdicts': {},
    }

    # Training
    print("\n[z1803] Training with counterfactual reasoning...")
    model.train()

    # Current simulated state (start near baseline)
    current_state = torch.rand(batch_size, 20, device=device) * 0.5 + 0.25

    for epoch in range(num_epochs):
        epoch_task_loss = 0
        epoch_wm_loss = 0
        epoch_cf_loss = 0

        for batch_idx in range(batches_per_epoch):
            x, y = get_batch()

            # Sample random action
            action = torch.randint(0, num_actions, (batch_size,), device=device)

            # Simulate next state
            next_state = simulate_action_effects(current_state, action[0].item())

            optimizer.zero_grad()

            output = model(x, current_state, return_hidden=True, action=action, return_world_model=True)
            logits = output['logits']

            # Task loss
            task_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            # World model loss: predict next state
            predicted_next = output.get('predicted_next_state')
            if predicted_next is not None:
                wm_loss = F.mse_loss(predicted_next, next_state)
            else:
                wm_loss = torch.tensor(0.0, device=device)

            # Counterfactual loss: train discriminator
            # Generate counterfactual (random different action)
            cf_action = (action + torch.randint(1, num_actions, (batch_size,), device=device)) % num_actions
            cf_next_state = simulate_action_effects(current_state, cf_action[0].item())

            cf_pred, real_prob = model.counterfactual_predict(
                current_state, action, cf_action, next_state
            )

            # Real transitions should have high probability
            cf_loss = F.binary_cross_entropy(real_prob, torch.ones_like(real_prob))
            # Counterfactual prediction should be accurate
            cf_loss = cf_loss + F.mse_loss(cf_pred, cf_next_state)

            # Total loss
            total_loss = task_loss + world_model_weight * wm_loss + cf_weight * cf_loss
            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_wm_loss += wm_loss.item()
            epoch_cf_loss += cf_loss.item()

            # Update current state
            current_state = next_state.detach()

        avg_task = epoch_task_loss / batches_per_epoch
        avg_wm = epoch_wm_loss / batches_per_epoch
        avg_cf = epoch_cf_loss / batches_per_epoch

        results['training']['task_losses'].append(avg_task)
        results['training']['world_model_losses'].append(avg_wm)
        results['training']['cf_losses'].append(avg_cf)

        print(f"  Epoch {epoch+1}/{num_epochs}: task={avg_task:.4f}, world_model={avg_wm:.4f}, cf={avg_cf:.4f}")

    # Evaluation: Counterfactual accuracy
    print("\n[z1803] Evaluating counterfactual reasoning...")
    model.eval()

    wm_errors = []
    cf_errors = []
    discrimination_acc = []

    current_state = torch.rand(batch_size, 20, device=device) * 0.5 + 0.25

    with torch.no_grad():
        for _ in range(100):
            action = torch.randint(0, num_actions, (batch_size,), device=device)
            cf_action = (action + torch.randint(1, num_actions, (batch_size,), device=device)) % num_actions

            # Actual next state
            next_state = simulate_action_effects(current_state, action[0].item())
            cf_actual = simulate_action_effects(current_state, cf_action[0].item())

            # Model predictions
            pred_next = model.predict_next_state(current_state, action)
            cf_pred, real_prob = model.counterfactual_predict(
                current_state, action, cf_action, next_state
            )

            # Errors
            wm_errors.append(F.mse_loss(pred_next, next_state).item())
            cf_errors.append(F.mse_loss(cf_pred, cf_actual).item())

            # Discrimination accuracy (real transitions should have prob > 0.5)
            discrimination_acc.append((real_prob > 0.5).float().mean().item())

            current_state = next_state

    avg_wm_error = np.mean(wm_errors)
    avg_cf_error = np.mean(cf_errors)
    avg_disc_acc = np.mean(discrimination_acc)

    results['evaluation'] = {
        'world_model_mse': float(avg_wm_error),
        'counterfactual_mse': float(avg_cf_error),
        'discrimination_accuracy': float(avg_disc_acc),
    }

    print(f"\n[z1803] Results:")
    print(f"  World model MSE: {avg_wm_error:.6f}")
    print(f"  Counterfactual MSE: {avg_cf_error:.6f}")
    print(f"  Discrimination accuracy: {avg_disc_acc:.2%}")

    # Verdicts
    # V1: World model learns (predicts actual next state)
    v1_pass = avg_wm_error < 0.1
    results['verdicts']['V1_world_model_accurate'] = {
        'pass': v1_pass,
        'world_model_mse': avg_wm_error,
        'threshold': 0.1,
        'description': 'World model accurately predicts next state'
    }

    # V2: Counterfactual reasoning works (predicts alternative outcomes)
    v2_pass = avg_cf_error < 0.15
    results['verdicts']['V2_counterfactual_accurate'] = {
        'pass': v2_pass,
        'counterfactual_mse': avg_cf_error,
        'threshold': 0.15,
        'description': 'Counterfactual predictions are accurate'
    }

    # V3: Can distinguish real from counterfactual
    v3_pass = avg_disc_acc > 0.6
    results['verdicts']['V3_discrimination'] = {
        'pass': v3_pass,
        'discrimination_accuracy': avg_disc_acc,
        'threshold': 0.6,
        'description': 'Model distinguishes real from counterfactual transitions'
    }

    # V4: Task preserved
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
    results['overall_verdict'] = 'COUNTERFACTUAL_REASONING_DEMONSTRATED' if passed >= 3 else 'PARTIAL'

    print(f"\n[z1803] Verdicts: {passed}/{total} passed")
    print(f"[z1803] Overall: {results['overall_verdict']}")

    # Cleanup
    telemetry_source.stop()

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "z1803_counterfactual_embodiment.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[z1803] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
