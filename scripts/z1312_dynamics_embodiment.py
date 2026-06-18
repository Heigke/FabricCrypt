#!/usr/bin/env python3
"""
z1312: Dynamics-Based Embodiment

The problem with z1311: Hardware state changes slowly, so delay baseline ties.
Copying t-5 is almost as good as copying t.

NEW APPROACH: Predict DYNAMICS (rate of change, direction, future trajectory)
- Not "what is the temperature?" but "is temperature rising or falling?"
- Not "current state" but "where is state going?"

This is harder to fake with delay/shuffle baselines because dynamics
require understanding temporal patterns, not just current values.

Key changes:
1. Predict DELTA (change), not absolute value
2. Multi-step rollout prediction
3. Stress onset detection (predict WHEN load will change)
4. Direction accuracy (did we predict the right sign of change?)
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class Sample:
    timestamp: float
    temp: float
    power: float
    util: float

    def to_array(self) -> np.ndarray:
        return np.array([self.temp, self.power, self.util], dtype=np.float32)


class TelemetryStream:
    """Continuous telemetry with computed derivatives"""

    def __init__(self, history_len: int = 64):
        self.card = '/sys/class/drm/card1/device'
        self.history: deque = deque(maxlen=history_len)
        self.timestamps: deque = deque(maxlen=history_len)

    def _hwmon(self, p, d=0):
        try:
            for h in os.listdir(f'{self.card}/hwmon'):
                f = f'{self.card}/hwmon/{h}/{p}'
                if os.path.exists(f):
                    with open(f) as fp:
                        return float(fp.read().strip())
        except:
            pass
        return d

    def _read(self, f, d=0):
        try:
            with open(f'{self.card}/{f}') as fp:
                return float(fp.read().strip())
        except:
            return d

    def read(self) -> Sample:
        sample = Sample(
            timestamp=time.time(),
            temp=self._hwmon('temp1_input', 50000) / 1000,
            power=self._hwmon('power1_average', 50e6) / 1e6,
            util=self._read('gpu_busy_percent', 50) / 100,
        )
        self.history.append(sample)
        self.timestamps.append(sample.timestamp)
        return sample

    def get_sequence(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get last n samples as (values, deltas) arrays"""
        if len(self.history) < n + 1:
            # Pad with current
            current = self.read()
            while len(self.history) < n + 1:
                self.history.appendleft(current)
                self.timestamps.appendleft(current.timestamp - 0.03)

        samples = list(self.history)[-n-1:]
        values = np.array([s.to_array() for s in samples[1:]])  # [n, 3]

        # Compute deltas (change from previous)
        deltas = np.zeros_like(values)
        for i in range(1, len(samples)):
            dt = max(samples[i].timestamp - samples[i-1].timestamp, 0.001)
            deltas[i-1] = (samples[i].to_array() - samples[i-1].to_array()) / dt

        # Normalize
        values[:, 0] /= 100  # temp
        values[:, 1] /= 100  # power
        deltas = np.clip(deltas, -1, 1)  # clip derivatives

        return values, deltas


class DynamicsModel(nn.Module):
    """
    Model that predicts DYNAMICS (change), not just state.

    Input: sequence of (value, delta) pairs
    Output: predicted delta for next K steps
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 seq_len: int = 16, pred_horizons: List[int] = [1, 3, 5, 10]):
        super().__init__()
        self.pred_horizons = pred_horizons
        self.seq_len = seq_len

        # Temporal encoder (1D conv over sequence)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Prediction heads for each horizon
        self.delta_heads = nn.ModuleDict()
        self.direction_heads = nn.ModuleDict()  # Binary: up or down

        for h in pred_horizons:
            self.delta_heads[f'h{h}'] = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3),  # Predict delta for each channel
            )
            self.direction_heads[f'h{h}'] = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # Binary direction per channel
                nn.Sigmoid(),
            )

    def forward(self, values: torch.Tensor, deltas: torch.Tensor) -> Dict:
        """
        Args:
            values: [B, T, 3] normalized sensor values
            deltas: [B, T, 3] rate of change
        """
        # Concatenate values and deltas
        x = torch.cat([values, deltas], dim=-1)  # [B, T, 6]
        x = x.transpose(1, 2)  # [B, 6, T] for conv1d

        # Temporal encoding
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)  # [B, hidden_dim]

        # Predictions for each horizon
        delta_preds = {}
        direction_preds = {}

        for horizon in self.pred_horizons:
            key = f'h{horizon}'
            delta_preds[key] = self.delta_heads[key](h)
            direction_preds[key] = self.direction_heads[key](h)

        return {
            'hidden': h,
            'delta_preds': delta_preds,
            'direction_preds': direction_preds,
        }


def collect_training_data(stream: TelemetryStream, n_sequences: int = 200,
                          seq_len: int = 16, max_horizon: int = 10) -> Dict:
    """Collect sequences with future labels"""

    print("Collecting training data with varied GPU load...")

    sequences = []
    future_values = []
    future_deltas = []

    # Warm up
    for _ in range(seq_len + max_horizon + 5):
        stream.read()
        time.sleep(0.025)

    for i in range(n_sequences):
        # Random stress pattern
        stress_type = np.random.choice(['none', 'light', 'heavy', 'burst'])

        if stress_type == 'light':
            _ = torch.randn(300, 300, device=DEVICE) @ torch.randn(300, 300, device=DEVICE)
        elif stress_type == 'heavy':
            _ = torch.randn(1500, 1500, device=DEVICE) @ torch.randn(1500, 1500, device=DEVICE)
        elif stress_type == 'burst':
            for _ in range(3):
                _ = torch.randn(800, 800, device=DEVICE) @ torch.randn(800, 800, device=DEVICE)
                time.sleep(0.01)

        # Get current sequence
        values, deltas = stream.get_sequence(seq_len)
        sequences.append((values.copy(), deltas.copy()))

        # Collect future states
        future_v = []
        future_d = []

        for step in range(max_horizon):
            time.sleep(0.025)

            # Occasional stress during collection
            if np.random.random() < 0.2:
                _ = torch.randn(400, 400, device=DEVICE) @ torch.randn(400, 400, device=DEVICE)

            stream.read()
            v, d = stream.get_sequence(1)
            future_v.append(v[-1])
            future_d.append(d[-1])

        future_values.append(np.array(future_v))
        future_deltas.append(np.array(future_d))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_sequences} sequences collected")

    return {
        'sequences': sequences,
        'future_values': future_values,
        'future_deltas': future_deltas,
    }


def train_dynamics_model(model: DynamicsModel, train_data: Dict,
                         epochs: int = 50, lr: float = 1e-3) -> Dict:
    """Train on predicting future dynamics"""

    print("\nTraining dynamics model...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    sequences = train_data['sequences']
    future_deltas = train_data['future_deltas']
    future_values = train_data['future_values']

    n = len(sequences)
    history = []

    model.train()

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        epoch_losses = {f'h{h}': [] for h in model.pred_horizons}
        epoch_dir_acc = {f'h{h}': [] for h in model.pred_horizons}

        for idx in indices:
            values, deltas = sequences[idx]
            fut_deltas = future_deltas[idx]
            fut_values = future_values[idx]

            # To tensors
            v_t = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            d_t = torch.tensor(deltas, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Forward
            out = model(v_t, d_t)

            total_loss = 0

            for h in model.pred_horizons:
                key = f'h{h}'
                if h <= len(fut_deltas):
                    # Delta prediction loss
                    target_delta = torch.tensor(fut_deltas[h-1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    delta_loss = F.mse_loss(out['delta_preds'][key], target_delta)

                    # Direction loss (binary cross-entropy)
                    # Direction = 1 if delta > 0, else 0
                    target_dir = (fut_deltas[h-1] > 0).astype(np.float32)
                    target_dir = torch.tensor(target_dir, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    dir_loss = F.binary_cross_entropy(out['direction_preds'][key], target_dir)

                    total_loss = total_loss + delta_loss + dir_loss * 0.5
                    epoch_losses[key].append(delta_loss.item())

                    # Direction accuracy
                    pred_dir = (out['direction_preds'][key] > 0.5).float()
                    dir_acc = (pred_dir == target_dir).float().mean().item()
                    epoch_dir_acc[key].append(dir_acc)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Log
        avg_losses = {k: np.mean(v) if v else 0 for k, v in epoch_losses.items()}
        avg_dir_acc = {k: np.mean(v) if v else 0 for k, v in epoch_dir_acc.items()}

        history.append({
            'epoch': epoch,
            'losses': avg_losses,
            'direction_accuracy': avg_dir_acc,
        })

        if (epoch + 1) % 10 == 0:
            loss_str = " ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
            dir_str = " ".join(f"{k}={v:.1%}" for k, v in avg_dir_acc.items())
            print(f"  Epoch {epoch+1}: Loss: {loss_str}")
            print(f"           Dir:  {dir_str}")

    return {'history': history}


def evaluate_dynamics(model: DynamicsModel, stream: TelemetryStream,
                      n_eval: int = 100, seq_len: int = 16) -> Dict:
    """
    Evaluate dynamics prediction.

    Key metrics:
    1. Direction accuracy: Did we predict the correct sign of change?
    2. Delta MAE: How close were we to actual rate of change?
    3. Comparison to naive baselines
    """

    print("\nEvaluating dynamics prediction...")

    model.eval()
    max_horizon = max(model.pred_horizons)

    results = {h: {'pred_deltas': [], 'actual_deltas': [],
                   'pred_dirs': [], 'actual_dirs': []}
               for h in model.pred_horizons}

    # Naive baseline: predict zero change (no dynamics)
    # Delay baseline: use delta from 5 steps ago

    for i in range(n_eval):
        # Random stress
        if np.random.random() < 0.4:
            intensity = np.random.choice([300, 800, 1500])
            _ = torch.randn(intensity, intensity, device=DEVICE) @ torch.randn(intensity, intensity, device=DEVICE)

        # Get sequence
        values, deltas = stream.get_sequence(seq_len)

        v_t = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        d_t = torch.tensor(deltas, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(v_t, d_t)

        # Collect actual future
        for h in model.pred_horizons:
            for _ in range(h):
                time.sleep(0.02)
                if np.random.random() < 0.15:
                    _ = torch.randn(400, 400, device=DEVICE) @ torch.randn(400, 400, device=DEVICE)
                stream.read()

            _, actual_d = stream.get_sequence(1)
            actual_delta = actual_d[-1]

            pred_delta = out['delta_preds'][f'h{h}'].cpu().numpy().squeeze()
            pred_dir = out['direction_preds'][f'h{h}'].cpu().numpy().squeeze()

            results[h]['pred_deltas'].append(pred_delta)
            results[h]['actual_deltas'].append(actual_delta)
            results[h]['pred_dirs'].append(pred_dir > 0.5)
            results[h]['actual_dirs'].append(actual_delta > 0)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_eval} evaluations")

    # Compute metrics
    metrics = {}

    print("\n" + "=" * 70)
    print("DYNAMICS PREDICTION RESULTS")
    print("=" * 70)

    print(f"\n{'Horizon':<10} | {'Dir Acc':<10} | {'Naive Dir':<10} | {'Delta MAE':<10} | {'Naive MAE':<10}")
    print("-" * 60)

    for h in model.pred_horizons:
        pred_d = np.array(results[h]['pred_deltas'])
        actual_d = np.array(results[h]['actual_deltas'])
        pred_dir = np.array(results[h]['pred_dirs'])
        actual_dir = np.array(results[h]['actual_dirs'])

        # Direction accuracy
        dir_acc = (pred_dir == actual_dir).mean()

        # Naive baseline: predict no change (direction = last direction)
        naive_dir = np.zeros_like(actual_dir)  # Predict "no change"
        naive_dir_acc = (naive_dir == actual_dir).mean()

        # Delta MAE
        delta_mae = np.mean(np.abs(pred_d - actual_d))

        # Naive: predict zero delta
        naive_mae = np.mean(np.abs(actual_d))  # MAE of predicting 0

        metrics[f'h{h}'] = {
            'direction_accuracy': float(dir_acc),
            'naive_direction_accuracy': float(naive_dir_acc),
            'delta_mae': float(delta_mae),
            'naive_delta_mae': float(naive_mae),
            'direction_improvement': float(dir_acc - naive_dir_acc),
            'mae_improvement': float((naive_mae - delta_mae) / naive_mae) if naive_mae > 0 else 0,
        }

        print(f"t+{h:<8} | {dir_acc:>8.1%} | {naive_dir_acc:>8.1%} | {delta_mae:>10.4f} | {naive_mae:>10.4f}")

    return metrics


def run_comparison_baselines(model: DynamicsModel, stream: TelemetryStream,
                             n_eval: int = 80, seq_len: int = 16) -> Dict:
    """
    Compare to fair baselines (delay, shuffle).

    For dynamics prediction, delay baseline = use delta from k steps ago.
    """

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (Fair)")
    print("=" * 70)

    model.eval()

    # Collect data with baselines
    live_results = []
    delay_results = []
    shuffle_results = []

    delay_k = 5

    for i in range(n_eval):
        # Stress variation
        if np.random.random() < 0.4:
            _ = torch.randn(np.random.choice([300, 800, 1200]),
                           np.random.choice([300, 800, 1200]), device=DEVICE).sum()

        values, deltas = stream.get_sequence(seq_len)

        # LIVE: Real aligned input
        v_live = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        d_live = torch.tensor(deltas, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # DELAY: Shift deltas by k (use old derivatives)
        d_delay = np.roll(deltas, delay_k, axis=0)
        d_delay[:delay_k] = 0  # Zero out the wrapped part
        d_delay = torch.tensor(d_delay, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # SHUFFLE: Randomly permute delta channels
        d_shuffle = deltas.copy()
        for t in range(len(d_shuffle)):
            perm = np.random.permutation(3)
            d_shuffle[t] = d_shuffle[t, perm]
        d_shuffle = torch.tensor(d_shuffle, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Get predictions
        with torch.no_grad():
            out_live = model(v_live, d_live)
            out_delay = model(v_live, d_delay)  # Same values, delayed deltas
            out_shuffle = model(v_live, d_shuffle)  # Same values, shuffled deltas

        # Collect actual future (horizon 3)
        for _ in range(3):
            time.sleep(0.02)
            stream.read()

        _, actual_d = stream.get_sequence(1)
        actual_dir = actual_d[-1] > 0

        # Direction predictions
        live_dir = out_live['direction_preds']['h3'].cpu().numpy().squeeze() > 0.5
        delay_dir = out_delay['direction_preds']['h3'].cpu().numpy().squeeze() > 0.5
        shuffle_dir = out_shuffle['direction_preds']['h3'].cpu().numpy().squeeze() > 0.5

        live_results.append((live_dir == actual_dir).mean())
        delay_results.append((delay_dir == actual_dir).mean())
        shuffle_results.append((shuffle_dir == actual_dir).mean())

    live_acc = np.mean(live_results)
    delay_acc = np.mean(delay_results)
    shuffle_acc = np.mean(shuffle_results)

    print(f"\nDirection Accuracy (horizon=3):")
    print(f"  LIVE (aligned):    {live_acc:.1%}")
    print(f"  DELAY-{delay_k} baseline: {delay_acc:.1%}")
    print(f"  SHUFFLE baseline:  {shuffle_acc:.1%}")

    # Statistical comparison
    live_vs_delay = live_acc - delay_acc
    live_vs_shuffle = live_acc - shuffle_acc

    print(f"\nDifferences:")
    print(f"  Live vs Delay:   {live_vs_delay:+.1%}")
    print(f"  Live vs Shuffle: {live_vs_shuffle:+.1%}")

    # Bootstrap CI for live
    n_boot = 1000
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(live_results, size=len(live_results), replace=True)
        boot_means.append(np.mean(sample))

    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)

    print(f"\nLive accuracy 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")

    return {
        'live_accuracy': float(live_acc),
        'delay_accuracy': float(delay_acc),
        'shuffle_accuracy': float(shuffle_acc),
        'live_vs_delay': float(live_vs_delay),
        'live_vs_shuffle': float(live_vs_shuffle),
        'ci_95': [float(ci_low), float(ci_high)],
    }


def main():
    print("=" * 70)
    print("  z1312: DYNAMICS-BASED EMBODIMENT")
    print("  Predicting change, not just state")
    print("=" * 70)
    print()
    print("Key insight: Hardware state changes slowly, so copying works.")
    print("Solution: Predict DYNAMICS (rate of change, direction)")
    print()

    stream = TelemetryStream()

    # Create model
    model = DynamicsModel(
        input_dim=6,  # 3 values + 3 deltas
        hidden_dim=128,
        seq_len=16,
        pred_horizons=[1, 3, 5, 10],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Collect training data
    train_data = collect_training_data(stream, n_sequences=250, seq_len=16, max_horizon=10)

    # Train
    train_results = train_dynamics_model(model, train_data, epochs=60, lr=1e-3)

    # Evaluate
    dynamics_results = evaluate_dynamics(model, stream, n_eval=100, seq_len=16)

    # Fair baseline comparison
    baseline_results = run_comparison_baselines(model, stream, n_eval=100, seq_len=16)

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    # Check if we beat baselines
    wins = 0

    # Direction improvement over naive
    avg_dir_imp = np.mean([v['direction_improvement'] for v in dynamics_results.values()])
    if avg_dir_imp > 0.05:
        print(f"\n✓ Direction prediction: +{avg_dir_imp:.1%} over naive")
        wins += 1
    else:
        print(f"\n✗ Direction prediction: {avg_dir_imp:+.1%} (not better than naive)")

    # Live vs delay
    if baseline_results['live_vs_delay'] > 0.03:
        print(f"✓ Live beats delay baseline: +{baseline_results['live_vs_delay']:.1%}")
        wins += 1
    else:
        print(f"✗ Live vs delay: {baseline_results['live_vs_delay']:+.1%} (not significant)")

    # Live vs shuffle
    if baseline_results['live_vs_shuffle'] > 0.03:
        print(f"✓ Live beats shuffle baseline: +{baseline_results['live_vs_shuffle']:.1%}")
        wins += 1
    else:
        print(f"✗ Live vs shuffle: {baseline_results['live_vs_shuffle']:+.1%} (not significant)")

    print(f"\nTests passed: {wins}/3")

    if wins >= 2:
        verdict = "DYNAMICS MODELING SHOWS ADVANTAGE"
        print(f"\n✅ {verdict}")
        print("   Model learns temporal patterns beyond naive baselines")
    else:
        verdict = "NO SIGNIFICANT ADVANTAGE"
        print(f"\n❌ {verdict}")

    # Save
    output = {
        'experiment': 'z1312_dynamics_embodiment',
        'timestamp': datetime.now().isoformat(),
        'dynamics_results': dynamics_results,
        'baseline_comparison': baseline_results,
        'wins': wins,
        'verdict': verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1312_dynamics_embodiment.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
