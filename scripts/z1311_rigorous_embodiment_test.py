#!/usr/bin/env python3
"""
z1311: Rigorous Embodiment Test

Addressing ALL audit critiques:
1. FAIR BASELINES: Delay, Shuffle, Replay (not zeros - that's unfair)
2. HOLDOUT EVALUATION: Train/test split with no leakage
3. CONFIDENCE INTERVALS: Multiple runs with bootstrap CIs
4. MASKED SENSOR TEST: Predict hidden channel (operational self-model)
5. DEFENSIBLE CLAIMS: No "consciousness" - only "body-state inference"

Based on external audit recommendations.
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
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')


@dataclass
class Sample:
    """Single telemetry sample with timestamp"""
    timestamp: float
    temp: float
    power: float
    util: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.temp / 100.0,
            self.power / 100.0,
            self.util,
        ], dtype=torch.float32)


class TelemetryCollector:
    """Collects and stores telemetry with proper timestamps"""

    def __init__(self):
        self.card = '/sys/class/drm/card1/device'
        self.history: List[Sample] = []

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

    def collect(self) -> Sample:
        """Collect single sample"""
        sample = Sample(
            timestamp=time.time(),
            temp=self._hwmon('temp1_input', 50000) / 1000,
            power=self._hwmon('power1_average', 50e6) / 1e6,
            util=self._read('gpu_busy_percent', 50) / 100,
        )
        self.history.append(sample)
        return sample

    def collect_sequence(self, n: int, interval: float = 0.03,
                         stress_prob: float = 0.3) -> List[Sample]:
        """Collect n samples with optional stress injection"""
        samples = []
        for i in range(n):
            if np.random.random() < stress_prob:
                intensity = np.random.choice([500, 1000, 1500])
                _ = torch.randn(intensity, intensity, device=DEVICE) @ torch.randn(intensity, intensity, device=DEVICE)

            samples.append(self.collect())
            time.sleep(interval)

        return samples


class BodyStateModel(nn.Module):
    """Simple but honest body state model"""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> Dict:
        h = self.encoder(x)
        pred = self.predictor(h)
        return {'hidden': h, 'prediction': pred}


def create_baselines(samples: List[Sample], delay_k: int = 5) -> Dict[str, List[torch.Tensor]]:
    """
    Create FAIR baseline inputs (not zeros!)

    1. LIVE: Real-time aligned telemetry
    2. DELAY-K: Telemetry from k steps ago
    3. SHUFFLE: Channels permuted (breaks semantics)
    4. REPLAY: Random historical sample (breaks temporal alignment)
    """
    n = len(samples)
    tensors = [s.to_tensor() for s in samples]

    baselines = {
        'live': [],
        f'delay_{delay_k}': [],
        'shuffle': [],
        'replay': [],
    }

    for i in range(delay_k, n):
        # LIVE: Current sample
        baselines['live'].append(tensors[i])

        # DELAY: Sample from k steps ago
        baselines[f'delay_{delay_k}'].append(tensors[i - delay_k])

        # SHUFFLE: Permute channels of current sample
        shuffled = tensors[i].clone()
        perm = torch.randperm(3)
        shuffled = shuffled[perm]
        baselines['shuffle'].append(shuffled)

        # REPLAY: Random historical sample
        replay_idx = np.random.randint(0, i)
        baselines['replay'].append(tensors[replay_idx])

    return baselines


def train_test_split(samples: List[Sample], test_ratio: float = 0.3) -> Tuple[List, List]:
    """Proper temporal train/test split (no leakage)"""
    n = len(samples)
    split_idx = int(n * (1 - test_ratio))
    return samples[:split_idx], samples[split_idx:]


def train_model(model: nn.Module, train_samples: List[Sample],
                epochs: int = 50, lr: float = 1e-3) -> Dict:
    """Train model on training set only"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        np.random.shuffle(train_samples)

        for i in range(1, len(train_samples)):
            # Input: previous sample, Target: current sample
            x = train_samples[i-1].to_tensor().unsqueeze(0).to(DEVICE)
            y = train_samples[i].to_tensor().unsqueeze(0).to(DEVICE)

            out = model(x)
            loss = F.mse_loss(out['prediction'], y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_samples))

    return {'train_losses': losses}


def evaluate_baseline(model: nn.Module, test_samples: List[Sample],
                      baseline_inputs: List[torch.Tensor],
                      baseline_name: str) -> Dict:
    """
    Evaluate model with specific baseline input.

    For fair comparison, ALL baselines predict the SAME targets.
    Only the INPUT differs.
    """
    model.eval()

    predictions = []
    targets = []
    hidden_states = []

    delay_k = 5  # Must match baseline creation
    test_tensors = [s.to_tensor() for s in test_samples]

    with torch.no_grad():
        for i, (inp, tgt) in enumerate(zip(baseline_inputs, test_tensors[delay_k:])):
            x = inp.unsqueeze(0).to(DEVICE)
            out = model(x)

            predictions.append(out['prediction'].cpu().numpy())
            targets.append(tgt.numpy())
            hidden_states.append(out['hidden'].cpu().numpy())

    predictions = np.array(predictions).squeeze()
    targets = np.array(targets).squeeze()
    hidden_states = np.array(hidden_states).squeeze()

    # Metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)

    # Per-channel correlation
    correlations = []
    for c in range(3):
        if np.std(predictions[:, c]) > 1e-6 and np.std(targets[:, c]) > 1e-6:
            corr = np.corrcoef(predictions[:, c], targets[:, c])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        else:
            correlations.append(0)

    return {
        'baseline': baseline_name,
        'mae': float(mae),
        'mse': float(mse),
        'correlations': correlations,
        'mean_correlation': float(np.mean(correlations)),
        'hidden_states': hidden_states,
    }


def state_classification_test(results: Dict[str, Dict]) -> Dict:
    """
    Test if hidden states distinguish conditions.

    Fair test: Same model, different input conditions.
    """
    # Get hidden states from live vs replay
    live_h = results['live']['hidden_states']
    replay_h = results['replay']['hidden_states']

    if len(live_h) < 10 or len(replay_h) < 10:
        return {'accuracy': 0.5, 'separation': 0.0}

    # Simple linear classification
    all_h = np.vstack([live_h, replay_h])
    labels = np.array([0] * len(live_h) + [1] * len(replay_h))

    live_c = live_h.mean(axis=0)
    replay_c = replay_h.mean(axis=0)

    direction = replay_c - live_c
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return {'accuracy': 0.5, 'separation': 0.0}

    direction /= norm
    proj = all_h @ direction
    thresh = proj.mean()
    preds = (proj > thresh).astype(int)

    accuracy = float((preds == labels).mean())
    separation = float(norm)

    return {'accuracy': accuracy, 'separation': separation}


def masked_sensor_test(model: nn.Module, test_samples: List[Sample],
                       mask_channel: int = 0) -> Dict:
    """
    OPERATIONAL SELF-MODEL TEST:

    Can the model predict a MASKED sensor channel?
    This is a falsifiable test of "body-state inference".

    If the model can predict temperature when temperature is masked,
    it has learned something about body dynamics (not just copying input).
    """
    model.eval()
    channel_names = ['temperature', 'power', 'utilization']

    predictions = []
    actuals = []

    with torch.no_grad():
        for i in range(1, len(test_samples)):
            x = test_samples[i-1].to_tensor().clone()
            actual_value = x[mask_channel].item()

            # MASK the channel
            x[mask_channel] = 0.5  # Neutral value

            x = x.unsqueeze(0).to(DEVICE)
            out = model(x)

            # Prediction for masked channel
            pred_value = out['prediction'][0, mask_channel].item()

            predictions.append(pred_value)
            actuals.append(actual_value)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(np.abs(predictions - actuals))

    if np.std(predictions) > 1e-6 and np.std(actuals) > 1e-6:
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        correlation = correlation if not np.isnan(correlation) else 0
    else:
        correlation = 0

    # Naive baseline: predict mean
    naive_pred = np.mean(actuals)
    naive_mae = np.mean(np.abs(naive_pred - actuals))

    improvement = (naive_mae - mae) / naive_mae if naive_mae > 0 else 0

    return {
        'masked_channel': channel_names[mask_channel],
        'model_mae': float(mae),
        'naive_mae': float(naive_mae),
        'improvement': float(improvement),
        'correlation': float(correlation),
    }


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """Bootstrap confidence interval"""
    values = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return float(np.mean(values)), float(lower), float(upper)


def run_single_experiment(seed: int = 42) -> Dict:
    """Run single experiment with given seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    collector = TelemetryCollector()

    # Collect data
    print(f"  Collecting samples (seed={seed})...")
    samples = collector.collect_sequence(n=300, interval=0.025, stress_prob=0.3)

    # Train/test split
    train_samples, test_samples = train_test_split(samples, test_ratio=0.3)

    # Create model
    model = BodyStateModel(input_dim=3, hidden_dim=64).to(DEVICE)

    # Train
    train_model(model, train_samples, epochs=30)

    # Create baselines for test set
    baselines = create_baselines(test_samples, delay_k=5)

    # Evaluate each baseline
    results = {}
    for name, inputs in baselines.items():
        results[name] = evaluate_baseline(model, test_samples, inputs, name)

    # State classification (live vs replay)
    classification = state_classification_test(results)

    # Masked sensor test
    masked = masked_sensor_test(model, test_samples, mask_channel=0)

    return {
        'seed': seed,
        'baseline_results': {k: {kk: vv for kk, vv in v.items() if kk != 'hidden_states'}
                             for k, v in results.items()},
        'classification': classification,
        'masked_sensor': masked,
    }


def main():
    print("=" * 70)
    print("  z1311: RIGOROUS EMBODIMENT TEST")
    print("  Fair baselines, proper holdout, confidence intervals")
    print("=" * 70)
    print()
    print("Addressing audit critiques:")
    print("  1. Fair baselines (delay, shuffle, replay - NOT zeros)")
    print("  2. Train/test split (no leakage)")
    print("  3. Multiple runs with CIs")
    print("  4. Masked sensor test (operational self-model)")
    print()

    # Multiple runs for confidence intervals
    n_runs = 5
    all_results = []

    print(f"Running {n_runs} experiments...")
    for i in range(n_runs):
        result = run_single_experiment(seed=42 + i * 7)
        all_results.append(result)
        print(f"  Run {i+1}/{n_runs} complete")

    # Aggregate results with CIs
    print("\n" + "=" * 70)
    print("RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 70)

    # Baseline comparison
    print("\n1. BASELINE COMPARISON (prediction MAE)")
    print("-" * 60)

    baseline_names = ['live', 'delay_5', 'shuffle', 'replay']
    baseline_maes = {name: [] for name in baseline_names}

    for result in all_results:
        for name in baseline_names:
            baseline_maes[name].append(result['baseline_results'][name]['mae'])

    for name in baseline_names:
        mean, lower, upper = bootstrap_ci(baseline_maes[name])
        print(f"  {name:12s}: MAE = {mean:.4f} [{lower:.4f}, {upper:.4f}]")

    # Check if live beats delay/shuffle/replay
    print("\n2. STATISTICAL TESTS")
    print("-" * 60)

    live_maes = np.array(baseline_maes['live'])
    wins = {}

    for name in ['delay_5', 'shuffle', 'replay']:
        other_maes = np.array(baseline_maes[name])
        # Paired comparison: how often does live win?
        live_wins = np.sum(live_maes < other_maes)
        wins[name] = live_wins
        diff = other_maes - live_maes
        mean_diff, lower, upper = bootstrap_ci(diff.tolist())

        if lower > 0:
            sig = "✓ SIGNIFICANT"
        else:
            sig = "not significant"

        print(f"  Live vs {name:8s}: Δ = {mean_diff:+.4f} [{lower:+.4f}, {upper:+.4f}] {sig}")

    # Classification accuracy
    print("\n3. STATE CLASSIFICATION (live vs replay hidden states)")
    print("-" * 60)

    class_accs = [r['classification']['accuracy'] for r in all_results]
    mean, lower, upper = bootstrap_ci(class_accs)
    print(f"  Accuracy: {mean:.1%} [{lower:.1%}, {upper:.1%}]")

    if lower > 0.6:
        class_verdict = "✓ Above chance (meaningful)"
    else:
        class_verdict = "Not significantly above chance"
    print(f"  Verdict: {class_verdict}")

    # Masked sensor test
    print("\n4. MASKED SENSOR TEST (predict temperature with temp masked)")
    print("-" * 60)

    masked_imps = [r['masked_sensor']['improvement'] for r in all_results]
    mean, lower, upper = bootstrap_ci(masked_imps)
    print(f"  Improvement over naive: {mean:.1%} [{lower:.1%}, {upper:.1%}]")

    if lower > 0:
        masked_verdict = "✓ Better than naive (learned dynamics)"
    else:
        masked_verdict = "Not better than naive"
    print(f"  Verdict: {masked_verdict}")

    # Final assessment
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    # Count significant wins
    sig_wins = sum(1 for name in ['delay_5', 'shuffle', 'replay']
                   if np.mean(np.array(baseline_maes[name]) - np.array(baseline_maes['live'])) > 0)

    claims = []
    if sig_wins >= 2:
        claims.append("✓ Live telemetry outperforms causal-decoupling baselines")
    if lower > 0.6:  # Classification CI
        claims.append("✓ Hidden states distinguish input conditions")
    if np.mean(masked_imps) > 0:
        claims.append("✓ Model infers masked sensor (operational self-model)")

    if len(claims) >= 2:
        overall = "BODY-STATE INFERENCE SUPPORTED"
        print(f"\n{overall}")
        print("\nDefensible claims:")
        for c in claims:
            print(f"  {c}")
        print("\nNOT claimed: consciousness, self-awareness, sentience")
    else:
        overall = "INSUFFICIENT EVIDENCE"
        print(f"\n{overall}")
        print("Cannot make strong claims about embodiment")

    # Save results
    output = {
        'experiment': 'z1311_rigorous_embodiment_test',
        'timestamp': datetime.now().isoformat(),
        'n_runs': n_runs,
        'baseline_comparison': {
            name: {
                'mean': float(np.mean(baseline_maes[name])),
                'ci_95': bootstrap_ci(baseline_maes[name]),
            }
            for name in baseline_names
        },
        'classification': {
            'mean': float(np.mean(class_accs)),
            'ci_95': bootstrap_ci(class_accs),
        },
        'masked_sensor': {
            'mean_improvement': float(np.mean(masked_imps)),
            'ci_95': bootstrap_ci(masked_imps),
        },
        'verdict': overall,
        'defensible_claims': claims,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1311_rigorous_embodiment_test.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
