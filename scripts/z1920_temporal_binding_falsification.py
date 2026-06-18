#!/usr/bin/env python3
"""
z1920: Temporal Binding Falsification Test

GOAL: Address the critical I4 (temporal coherence) failure from z1909.

The z1901 falsification battery showed that FiLM conditioning creates
NO behavioral difference for language modeling. This test specifically
targets the temporal dimension - can the model track CHANGES in hardware
state over time, or is it just responding to instantaneous values?

Tests:
T1: Temporal sequence prediction - predict hardware trajectory
T2: Lag detection - can model detect which telemetry is delayed?
T3: Rate of change sensitivity - does model respond to derivatives?
T4: Hysteresis test - does history affect current response?
T5: Sequence order matters - shuffle temporal order, measure impact

Falsification criteria:
- If model performs same with shuffled vs ordered sequences → FALSIFIED
- If model can't detect lag in telemetry → FALSIFIED
- If rate of change has no effect → FALSIFIED
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU setup for gfx1151
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# Telemetry
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalEmbodiedModel(nn.Module):
    """Model with explicit temporal awareness of hardware state."""

    def __init__(self, vocab_size=256, hidden_dim=256, num_layers=6,
                 num_heads=4, telemetry_dim=12, seq_history=16):
        super().__init__()

        self.seq_history = seq_history
        self.telemetry_dim = telemetry_dim
        self.hidden_dim = hidden_dim

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Temporal telemetry encoder - processes history of telemetry
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=telemetry_dim,
                nhead=4,
                dim_feedforward=64,
                batch_first=True
            ),
            num_layers=2
        )

        # Project temporal features to conditioning dimension
        self.temporal_proj = nn.Sequential(
            nn.Linear(telemetry_dim * seq_history, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # gamma and beta
        )

        # FiLM conditioning
        self.film_gamma = None
        self.film_beta = None

        # Main transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Output heads
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Temporal prediction head - predict next telemetry from history
        self.temporal_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, telemetry_dim)
        )

        # Lag detection head - classify which telemetry is lagged
        self.lag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3)  # no lag, lag 1, lag 2+
        )

        # Rate of change head - detect if rising/falling/stable
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3)  # falling, stable, rising
        )

        # Telemetry history buffer
        self.telemetry_history = []

    def update_telemetry(self, telemetry: torch.Tensor):
        """Add telemetry to history buffer."""
        self.telemetry_history.append(telemetry.detach().cpu())
        if len(self.telemetry_history) > self.seq_history:
            self.telemetry_history.pop(0)

    def get_temporal_features(self) -> torch.Tensor:
        """Get temporal encoding of telemetry history."""
        if len(self.telemetry_history) < self.seq_history:
            # Pad with zeros
            pad = [torch.zeros_like(self.telemetry_history[0])
                   for _ in range(self.seq_history - len(self.telemetry_history))]
            history = pad + self.telemetry_history
        else:
            history = self.telemetry_history[-self.seq_history:]

        # Stack: [seq_history, telemetry_dim]
        history_tensor = torch.stack(history).to(DEVICE)

        # Add batch dimension: [1, seq_history, telemetry_dim]
        history_tensor = history_tensor.unsqueeze(0)

        # Encode temporal patterns
        encoded = self.temporal_encoder(history_tensor)

        # Flatten for projection
        flat = encoded.view(1, -1)

        # Get FiLM parameters
        film_params = self.temporal_proj(flat)
        self.film_gamma = film_params[:, :self.hidden_dim]
        self.film_beta = film_params[:, self.hidden_dim:]

        return encoded

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor = None):
        """Forward with temporal telemetry conditioning."""
        batch_size, seq_len = x.shape

        # Update telemetry history if provided
        if telemetry is not None:
            self.update_telemetry(telemetry)

        # Get temporal features
        temporal_feats = self.get_temporal_features()

        # Embed tokens
        h = self.embed(x)

        # Apply FiLM conditioning
        if self.film_gamma is not None:
            gamma = self.film_gamma.unsqueeze(1).expand(-1, seq_len, -1)
            beta = self.film_beta.unsqueeze(1).expand(-1, seq_len, -1)
            h = gamma * h + beta

        # Transformer
        h = self.transformer(h)

        # Language modeling
        logits = self.lm_head(h)

        # Temporal prediction (from pooled hidden)
        pooled = h.mean(dim=1)
        temporal_pred = self.temporal_pred(pooled)
        lag_pred = self.lag_head(pooled)
        rate_pred = self.rate_head(pooled)

        return {
            'logits': logits,
            'temporal_pred': temporal_pred,
            'lag_pred': lag_pred,
            'rate_pred': rate_pred,
            'hidden': h
        }


class GPUTelemetry:
    """Collect GPU telemetry with temporal tracking."""

    def __init__(self):
        self.sensor = SysfsHwmonTelemetry()
        self.history = []

    def sense(self) -> torch.Tensor:
        """Get normalized telemetry vector."""
        sample = self.sensor.read_sample()
        state = {
            'temp_edge_c': sample.temp_c if hasattr(sample, 'temp_c') else 50,
            'temp_junction_c': sample.temp_c if hasattr(sample, 'temp_c') else 50,
            'power_w': sample.power_w if hasattr(sample, 'power_w') else 20,
            'power_cap_w': 150,
            'sclk_mhz': sample.sclk_mhz if hasattr(sample, 'sclk_mhz') else 1000,
            'mclk_mhz': sample.mclk_mhz if hasattr(sample, 'mclk_mhz') else 1000,
            'gpu_util_pct': sample.gpu_util if hasattr(sample, 'gpu_util') else 0,
            'mem_util_pct': 0,
            'vram_used_pct': 0,
        }

        # Extract relevant fields
        telemetry = torch.tensor([
            state.get('temp_edge_c', 50) / 100,
            state.get('temp_junction_c', 50) / 100,
            state.get('power_w', 20) / 200,
            state.get('power_cap_w', 50) / 200,
            state.get('sclk_mhz', 1000) / 3000,
            state.get('mclk_mhz', 1000) / 3000,
            state.get('gpu_util_pct', 0) / 100,
            state.get('mem_util_pct', 0) / 100,
            state.get('vram_used_pct', 0),
            state.get('fan_rpm', 0) / 5000 if 'fan_rpm' in state else 0,
            state.get('voltage_mv', 800) / 1500 if 'voltage_mv' in state else 0.5,
            state.get('soc_power_w', 10) / 100 if 'soc_power_w' in state else 0.1,
        ], dtype=torch.float32)

        self.history.append(telemetry)
        return telemetry


def run_t1_temporal_prediction(model: TemporalEmbodiedModel,
                                gpu: GPUTelemetry,
                                num_samples: int = 100) -> Dict:
    """T1: Can model predict next telemetry from history?"""
    print("\n=== T1: Temporal Sequence Prediction ===")

    model.eval()
    errors = []

    # Clear history
    model.telemetry_history = []

    # Build up history with real measurements
    for _ in range(model.seq_history):
        telem = gpu.sense()
        model.update_telemetry(telem)
        time.sleep(0.05)  # 50ms between samples

    with torch.no_grad():
        for i in range(num_samples):
            # Get current telemetry
            current = gpu.sense().to(DEVICE)

            # Model forward
            dummy_input = torch.randint(0, 256, (1, 32)).to(DEVICE)
            outputs = model(dummy_input)

            # Prediction error
            pred = outputs['temporal_pred'].squeeze()
            error = F.mse_loss(pred, current).item()
            errors.append(error)

            # Update history
            model.update_telemetry(current)

            # Small delay for variation
            time.sleep(0.02)

            if i % 20 == 0:
                print(f"  Sample {i}: MSE={error:.6f}")

    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # Compare to naive baseline (predict last value)
    naive_errors = []
    for i in range(1, len(gpu.history)):
        naive_pred = gpu.history[i-1]
        actual = gpu.history[i]
        naive_error = F.mse_loss(naive_pred, actual).item()
        naive_errors.append(naive_error)

    naive_mean = np.mean(naive_errors) if naive_errors else 1.0

    # Falsification: if model error >= naive error, temporal prediction failed
    improvement = (naive_mean - mean_error) / naive_mean if naive_mean > 0 else 0
    falsified = improvement <= 0

    return {
        'test': 'T1_temporal_prediction',
        'model_mse': mean_error,
        'model_std': std_error,
        'naive_mse': naive_mean,
        'improvement': improvement,
        'falsified': falsified,
        'interpretation': 'FALSIFIED - no temporal prediction' if falsified
                          else f'PASSED - {improvement*100:.1f}% better than naive'
    }


def run_t2_lag_detection(model: TemporalEmbodiedModel,
                          gpu: GPUTelemetry,
                          num_trials: int = 50) -> Dict:
    """T2: Can model detect which telemetry stream is lagged?"""
    print("\n=== T2: Lag Detection ===")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for trial in range(num_trials):
            # Clear and build fresh history
            model.telemetry_history = []

            # Decide lag condition
            lag_class = trial % 3  # 0: no lag, 1: lag 1, 2: lag 2+

            for step in range(model.seq_history):
                real_telem = gpu.sense()

                if lag_class == 0:
                    # No lag - use real telemetry
                    telem = real_telem
                elif lag_class == 1:
                    # Lag 1 - use previous telemetry
                    if len(gpu.history) >= 2:
                        telem = gpu.history[-2]
                    else:
                        telem = real_telem
                else:
                    # Lag 2+ - use older telemetry
                    if len(gpu.history) >= 3:
                        telem = gpu.history[-3]
                    else:
                        telem = real_telem

                model.update_telemetry(telem)
                time.sleep(0.02)

            # Model forward
            dummy_input = torch.randint(0, 256, (1, 32)).to(DEVICE)
            outputs = model(dummy_input)

            # Check lag prediction
            lag_pred = outputs['lag_pred'].argmax().item()

            if lag_pred == lag_class:
                correct += 1
            total += 1

            if trial % 10 == 0:
                print(f"  Trial {trial}: predicted={lag_pred}, actual={lag_class}")

    accuracy = correct / total
    # Random baseline is 33%
    falsified = accuracy <= 0.4  # Need to beat random by margin

    return {
        'test': 'T2_lag_detection',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'chance': 0.333,
        'falsified': falsified,
        'interpretation': f'FALSIFIED - {accuracy*100:.1f}% ≤ 40%' if falsified
                          else f'PASSED - {accuracy*100:.1f}% lag detection'
    }


def run_t3_rate_sensitivity(model: TemporalEmbodiedModel,
                             gpu: GPUTelemetry,
                             num_trials: int = 50) -> Dict:
    """T3: Does model respond to rate of change in telemetry?"""
    print("\n=== T3: Rate of Change Sensitivity ===")

    model.eval()

    # Create synthetic rate patterns
    rising_hiddens = []
    stable_hiddens = []
    falling_hiddens = []

    with torch.no_grad():
        for trial in range(num_trials):
            for rate_class, rate_list in [(0, falling_hiddens),
                                           (1, stable_hiddens),
                                           (2, rising_hiddens)]:
                model.telemetry_history = []

                # Create synthetic telemetry with controlled rate
                base = gpu.sense()
                for step in range(model.seq_history):
                    if rate_class == 0:  # falling
                        delta = -0.02 * step
                    elif rate_class == 1:  # stable
                        delta = 0
                    else:  # rising
                        delta = 0.02 * step

                    synthetic = base + delta
                    synthetic = torch.clamp(synthetic, 0, 1)
                    model.update_telemetry(synthetic)

                # Forward pass
                dummy_input = torch.randint(0, 256, (1, 32)).to(DEVICE)
                outputs = model(dummy_input)

                # Store hidden representation
                hidden = outputs['hidden'].mean(dim=1).squeeze().cpu().numpy()
                rate_list.append(hidden)

    # Compare hidden representations across rate conditions
    rising_hiddens = np.array(rising_hiddens)
    stable_hiddens = np.array(stable_hiddens)
    falling_hiddens = np.array(falling_hiddens)

    # Distance between conditions
    rising_stable = np.linalg.norm(rising_hiddens.mean(axis=0) - stable_hiddens.mean(axis=0))
    falling_stable = np.linalg.norm(falling_hiddens.mean(axis=0) - stable_hiddens.mean(axis=0))
    rising_falling = np.linalg.norm(rising_hiddens.mean(axis=0) - falling_hiddens.mean(axis=0))

    min_distance = min(rising_stable, falling_stable, rising_falling)

    # Rate classification accuracy
    rate_correct = 0
    with torch.no_grad():
        for _ in range(30):
            for rate_class in [0, 1, 2]:
                model.telemetry_history = []
                base = gpu.sense()

                for step in range(model.seq_history):
                    if rate_class == 0:
                        delta = -0.02 * step
                    elif rate_class == 1:
                        delta = 0
                    else:
                        delta = 0.02 * step

                    synthetic = torch.clamp(base + delta, 0, 1)
                    model.update_telemetry(synthetic)

                dummy_input = torch.randint(0, 256, (1, 32)).to(DEVICE)
                outputs = model(dummy_input)

                if outputs['rate_pred'].argmax().item() == rate_class:
                    rate_correct += 1

    rate_accuracy = rate_correct / 90

    # Falsification: if hidden states don't differ across rate conditions
    falsified = min_distance < 0.1 and rate_accuracy <= 0.4

    return {
        'test': 'T3_rate_sensitivity',
        'rising_stable_dist': rising_stable,
        'falling_stable_dist': falling_stable,
        'rising_falling_dist': rising_falling,
        'min_distance': min_distance,
        'rate_classification_accuracy': rate_accuracy,
        'falsified': falsified,
        'interpretation': 'FALSIFIED - no rate sensitivity' if falsified
                          else f'PASSED - {min_distance:.3f} hidden separation'
    }


def run_t4_hysteresis(model: TemporalEmbodiedModel,
                       gpu: GPUTelemetry,
                       num_trials: int = 30) -> Dict:
    """T4: Does history affect response to same current value?"""
    print("\n=== T4: Hysteresis Test ===")

    model.eval()

    same_value_different_history = []

    with torch.no_grad():
        for trial in range(num_trials):
            # Get a target value
            target = gpu.sense()

            # History A: rising to target
            model.telemetry_history = []
            for step in range(model.seq_history - 1):
                frac = step / (model.seq_history - 1)
                rising = target * frac
                model.update_telemetry(rising)
            model.update_telemetry(target)  # End at target

            dummy = torch.randint(0, 256, (1, 32)).to(DEVICE)
            out_rising = model(dummy)
            hidden_rising = out_rising['hidden'].mean(dim=1).squeeze().cpu().numpy()

            # History B: falling to target
            model.telemetry_history = []
            for step in range(model.seq_history - 1):
                frac = (model.seq_history - 1 - step) / (model.seq_history - 1)
                falling = target + (torch.ones_like(target) - target) * frac
                model.update_telemetry(falling)
            model.update_telemetry(target)  # End at same target

            out_falling = model(dummy)
            hidden_falling = out_falling['hidden'].mean(dim=1).squeeze().cpu().numpy()

            # Distance between hidden states (same current, different history)
            dist = np.linalg.norm(hidden_rising - hidden_falling)
            same_value_different_history.append(dist)

            if trial % 10 == 0:
                print(f"  Trial {trial}: history difference = {dist:.4f}")

    mean_diff = np.mean(same_value_different_history)

    # Falsification: if same current value produces same hidden regardless of history
    falsified = mean_diff < 0.05

    return {
        'test': 'T4_hysteresis',
        'mean_history_effect': mean_diff,
        'std_history_effect': np.std(same_value_different_history),
        'falsified': falsified,
        'interpretation': 'FALSIFIED - no hysteresis' if falsified
                          else f'PASSED - {mean_diff:.4f} history effect'
    }


def run_t5_sequence_order(model: TemporalEmbodiedModel,
                           gpu: GPUTelemetry,
                           num_trials: int = 30) -> Dict:
    """T5: Does shuffling temporal order change output?"""
    print("\n=== T5: Sequence Order Matters ===")

    model.eval()

    order_effects = []

    with torch.no_grad():
        for trial in range(num_trials):
            # Collect fresh telemetry sequence
            sequence = []
            for _ in range(model.seq_history):
                sequence.append(gpu.sense())
                time.sleep(0.02)

            # Ordered sequence
            model.telemetry_history = []
            for t in sequence:
                model.update_telemetry(t)

            dummy = torch.randint(0, 256, (1, 32)).to(DEVICE)
            out_ordered = model(dummy)
            hidden_ordered = out_ordered['hidden'].mean(dim=1).squeeze().cpu().numpy()
            logits_ordered = out_ordered['logits'].mean().item()

            # Shuffled sequence (same values, different order)
            shuffled = sequence.copy()
            np.random.shuffle(shuffled)

            model.telemetry_history = []
            for t in shuffled:
                model.update_telemetry(t)

            out_shuffled = model(dummy)
            hidden_shuffled = out_shuffled['hidden'].mean(dim=1).squeeze().cpu().numpy()
            logits_shuffled = out_shuffled['logits'].mean().item()

            # Measure difference
            hidden_diff = np.linalg.norm(hidden_ordered - hidden_shuffled)
            logits_diff = abs(logits_ordered - logits_shuffled)

            order_effects.append({
                'hidden_diff': hidden_diff,
                'logits_diff': logits_diff
            })

            if trial % 10 == 0:
                print(f"  Trial {trial}: hidden_diff={hidden_diff:.4f}, logits_diff={logits_diff:.4f}")

    mean_hidden_diff = np.mean([e['hidden_diff'] for e in order_effects])
    mean_logits_diff = np.mean([e['logits_diff'] for e in order_effects])

    # Falsification: if shuffling has no effect, temporal order doesn't matter
    falsified = mean_hidden_diff < 0.05 and mean_logits_diff < 0.01

    return {
        'test': 'T5_sequence_order',
        'mean_hidden_diff': mean_hidden_diff,
        'mean_logits_diff': mean_logits_diff,
        'falsified': falsified,
        'interpretation': 'FALSIFIED - order irrelevant' if falsified
                          else f'PASSED - order matters (hidden={mean_hidden_diff:.4f})'
    }


def train_temporal_model(model: TemporalEmbodiedModel,
                          gpu: GPUTelemetry,
                          num_epochs: int = 5,
                          steps_per_epoch: int = 50) -> List[Dict]:
    """Train the temporal model."""
    print("\n=== Training Temporal Model ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for step in range(steps_per_epoch):
            # Get fresh telemetry
            telem = gpu.sense().to(DEVICE)

            # Random input
            x = torch.randint(0, 256, (4, 64)).to(DEVICE)
            targets = x[:, 1:].contiguous()

            # Forward
            outputs = model(x[:, :-1], telem)

            # LM loss
            lm_loss = F.cross_entropy(
                outputs['logits'].reshape(-1, 256),
                targets.reshape(-1)
            )

            # Temporal prediction loss
            temporal_loss = F.mse_loss(outputs['temporal_pred'], telem.unsqueeze(0).expand(4, -1))

            # Combined loss
            loss = lm_loss + 0.1 * temporal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append({
                'total': loss.item(),
                'lm': lm_loss.item(),
                'temporal': temporal_loss.item()
            })

            time.sleep(0.01)  # Small delay for telemetry variation

        mean_loss = np.mean([l['total'] for l in epoch_losses])
        mean_temp = np.mean([l['temporal'] for l in epoch_losses])
        losses.append({'epoch': epoch, 'loss': mean_loss, 'temporal': mean_temp})
        print(f"  Epoch {epoch}: loss={mean_loss:.4f}, temporal={mean_temp:.4f}")

    return losses


def main():
    print("=" * 60)
    print("z1920: Temporal Binding Falsification Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        'experiment': 'z1920_temporal_binding_falsification',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
    }

    # Initialize
    gpu = GPUTelemetry()
    model = TemporalEmbodiedModel(
        vocab_size=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        telemetry_dim=12,
        seq_history=16
    ).to(DEVICE)

    results['model_params'] = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {results['model_params']:,}")

    # Train model
    training_losses = train_temporal_model(model, gpu, num_epochs=5, steps_per_epoch=50)
    results['training'] = training_losses

    # Run all tests
    tests = {}

    tests['T1'] = run_t1_temporal_prediction(model, gpu)
    tests['T2'] = run_t2_lag_detection(model, gpu)
    tests['T3'] = run_t3_rate_sensitivity(model, gpu)
    tests['T4'] = run_t4_hysteresis(model, gpu)
    tests['T5'] = run_t5_sequence_order(model, gpu)

    results['tests'] = tests

    # Summary
    num_falsified = sum(1 for t in tests.values() if t['falsified'])
    num_total = len(tests)

    results['num_falsified'] = num_falsified
    results['num_total'] = num_total
    results['temporal_binding_score'] = (num_total - num_falsified) / num_total

    if num_falsified >= 3:
        results['verdict'] = 'TEMPORAL BINDING FALSIFIED'
    elif num_falsified >= 1:
        results['verdict'] = 'PARTIAL TEMPORAL BINDING'
    else:
        results['verdict'] = 'TEMPORAL BINDING DEMONSTRATED'

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, test in tests.items():
        status = "❌ FALSIFIED" if test['falsified'] else "✅ PASSED"
        print(f"{name}: {status} - {test['interpretation']}")

    print(f"\nFalsified: {num_falsified}/{num_total}")
    print(f"Temporal binding score: {results['temporal_binding_score']:.1%}")
    print(f"VERDICT: {results['verdict']}")

    # Save results
    output_path = 'results/z1920_temporal_binding_falsification.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
