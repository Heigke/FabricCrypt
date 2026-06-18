#!/usr/bin/env python3
"""
z1313: Task-Based Embodiment Test

Previous approaches failed because:
- Predicting hardware state is trivial (autocorrelated)
- Baselines (delay, shuffle) work almost as well

NEW APPROACH: Does hardware awareness improve a PRIMARY TASK?

The test: Train a simple sequence prediction task.
- Embodied: Model receives hardware state, learns to adapt
- Blind: Model does NOT receive hardware state
- Stress test: Heavy GPU load degrades computation (numerical noise)

If embodied outperforms blind under stress, we have evidence that
hardware awareness provides functional benefit.

Key insight: Under heavy GPU load, there's actual computational noise
(thermal throttling, memory contention). An embodied model could
learn to compensate or at least track its own degradation.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HardwareSensor:
    """Simple hardware sensor"""
    def __init__(self):
        self.card = '/sys/class/drm/card1/device'

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

    def read(self) -> torch.Tensor:
        """Get hardware state as tensor [3]"""
        return torch.tensor([
            self._hwmon('temp1_input', 50000) / 1000 / 100,  # Normalized temp
            self._hwmon('power1_average', 50e6) / 1e6 / 100,  # Normalized power
            self._read('gpu_busy_percent', 50) / 100,  # Utilization
        ], dtype=torch.float32)


class SequenceTask:
    """
    Simple sequence prediction task.

    Generate sequences with patterns, model must predict next element.
    Under stress, patterns may be harder to detect due to computational load.
    """

    def __init__(self, seq_len: int = 32, vocab_size: int = 16):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def generate_batch(self, batch_size: int, pattern_type: str = 'random') -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of sequences with patterns"""

        sequences = []
        targets = []

        for _ in range(batch_size):
            if pattern_type == 'repeat':
                # Repeating pattern: 1,2,3,4,1,2,3,4,...
                period = np.random.randint(2, 6)
                base = np.random.randint(0, self.vocab_size, size=period)
                seq = np.tile(base, self.seq_len // period + 1)[:self.seq_len]
                target = base[(self.seq_len) % period]

            elif pattern_type == 'increment':
                # Incrementing: 1,2,3,4,5,... (with wrap-around)
                start = np.random.randint(0, self.vocab_size)
                seq = np.arange(start, start + self.seq_len) % self.vocab_size
                target = (start + self.seq_len) % self.vocab_size

            elif pattern_type == 'fibonacci':
                # Fibonacci-like modular arithmetic
                seq = np.zeros(self.seq_len, dtype=np.int64)
                seq[0] = np.random.randint(0, self.vocab_size)
                seq[1] = np.random.randint(0, self.vocab_size)
                for i in range(2, self.seq_len):
                    seq[i] = (seq[i-1] + seq[i-2]) % self.vocab_size
                target = (seq[-1] + seq[-2]) % self.vocab_size

            else:  # random
                seq = np.random.randint(0, self.vocab_size, size=self.seq_len)
                target = np.random.randint(0, self.vocab_size)

            sequences.append(seq)
            targets.append(target)

        return (torch.tensor(np.array(sequences), dtype=torch.long),
                torch.tensor(targets, dtype=torch.long))


class EmbodiedSequenceModel(nn.Module):
    """
    Sequence model that receives hardware state.

    Hardware state is used to modulate the model's behavior
    via FiLM conditioning.
    """

    def __init__(self, vocab_size: int = 16, hidden_dim: int = 64,
                 hw_dim: int = 3, embodied: bool = True):
        super().__init__()
        self.embodied = embodied
        self.hidden_dim = hidden_dim

        # Sequence encoder
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Hardware encoder (only used if embodied)
        if embodied:
            self.hw_encoder = nn.Sequential(
                nn.Linear(hw_dim, 32),
                nn.ReLU(),
                nn.Linear(32, hidden_dim * 2),  # gamma and beta for FiLM
            )

        # Output head
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq: torch.Tensor, hw_state: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            seq: [B, T] sequence of tokens
            hw_state: [B, 3] hardware state (ignored if not embodied)
        """
        # Embed sequence
        x = self.embed(seq)  # [B, T, H]

        # LSTM encoding
        out, (h, c) = self.lstm(x)
        h = h.squeeze(0)  # [B, H]

        # FiLM conditioning from hardware (if embodied)
        if self.embodied and hw_state is not None:
            film_params = self.hw_encoder(hw_state)  # [B, 2H]
            gamma = film_params[:, :self.hidden_dim]
            beta = film_params[:, self.hidden_dim:]
            h = gamma * h + beta

        # Output
        logits = self.output(h)  # [B, vocab_size]
        return logits


def create_gpu_stress(intensity: str = 'none'):
    """Create GPU stress of varying intensity"""
    if intensity == 'none':
        pass
    elif intensity == 'light':
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    elif intensity == 'medium':
        _ = torch.randn(1000, 1000, device=DEVICE) @ torch.randn(1000, 1000, device=DEVICE)
    elif intensity == 'heavy':
        _ = torch.randn(2000, 2000, device=DEVICE) @ torch.randn(2000, 2000, device=DEVICE)
    elif intensity == 'extreme':
        for _ in range(3):
            _ = torch.randn(2000, 2000, device=DEVICE) @ torch.randn(2000, 2000, device=DEVICE)


def train_model(model: nn.Module, task: SequenceTask, sensor: HardwareSensor,
                epochs: int = 50, batch_size: int = 32, with_stress: bool = True) -> Dict:
    """Train model with optional stress injection"""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []

    patterns = ['repeat', 'increment', 'fibonacci']
    stress_levels = ['none', 'light', 'medium', 'heavy'] if with_stress else ['none']

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        n_batches = 0

        for pattern in patterns:
            for stress in stress_levels:
                # Create stress BEFORE inference (like real conditions)
                create_gpu_stress(stress)

                # Get hardware state
                hw = sensor.read().unsqueeze(0).expand(batch_size, -1).to(DEVICE)

                # Generate batch
                seq, target = task.generate_batch(batch_size, pattern)
                seq, target = seq.to(DEVICE), target.to(DEVICE)

                # Forward
                if model.embodied:
                    logits = model(seq, hw)
                else:
                    logits = model(seq)

                # Loss
                loss = F.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accuracy
                preds = logits.argmax(dim=-1)
                acc = (preds == target).float().mean()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        history.append({'epoch': epoch, 'loss': avg_loss, 'accuracy': avg_acc})

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.1%}")

    return {'history': history}


def evaluate_under_stress(model: nn.Module, task: SequenceTask, sensor: HardwareSensor,
                          n_eval: int = 50, batch_size: int = 32) -> Dict:
    """Evaluate model under different stress levels"""

    model.eval()
    results = {}

    stress_levels = ['none', 'light', 'medium', 'heavy', 'extreme']
    patterns = ['repeat', 'increment', 'fibonacci']

    for stress in stress_levels:
        stress_results = {'loss': [], 'accuracy': []}

        for _ in range(n_eval):
            # Create stress
            create_gpu_stress(stress)

            # Get hardware state
            hw = sensor.read().unsqueeze(0).expand(batch_size, -1).to(DEVICE)

            # Random pattern
            pattern = np.random.choice(patterns)
            seq, target = task.generate_batch(batch_size, pattern)
            seq, target = seq.to(DEVICE), target.to(DEVICE)

            with torch.no_grad():
                if model.embodied:
                    logits = model(seq, hw)
                else:
                    logits = model(seq)

                loss = F.cross_entropy(logits, target)
                preds = logits.argmax(dim=-1)
                acc = (preds == target).float().mean()

                stress_results['loss'].append(loss.item())
                stress_results['accuracy'].append(acc.item())

        results[stress] = {
            'mean_loss': float(np.mean(stress_results['loss'])),
            'std_loss': float(np.std(stress_results['loss'])),
            'mean_accuracy': float(np.mean(stress_results['accuracy'])),
            'std_accuracy': float(np.std(stress_results['accuracy'])),
        }

    return results


def main():
    print("=" * 70)
    print("  z1313: TASK-BASED EMBODIMENT TEST")
    print("  Does hardware awareness improve task performance under stress?")
    print("=" * 70)
    print()

    sensor = HardwareSensor()
    task = SequenceTask(seq_len=32, vocab_size=16)

    # Create two models: embodied and blind
    embodied_model = EmbodiedSequenceModel(
        vocab_size=16, hidden_dim=64, hw_dim=3, embodied=True
    ).to(DEVICE)

    blind_model = EmbodiedSequenceModel(
        vocab_size=16, hidden_dim=64, hw_dim=3, embodied=False
    ).to(DEVICE)

    # Copy initial weights for fair comparison
    blind_model.load_state_dict(
        {k: v for k, v in embodied_model.state_dict().items() if 'hw_encoder' not in k},
        strict=False
    )

    print(f"Embodied model params: {sum(p.numel() for p in embodied_model.parameters()):,}")
    print(f"Blind model params: {sum(p.numel() for p in blind_model.parameters()):,}")

    # Train both models with stress
    print("\nTraining EMBODIED model...")
    embodied_train = train_model(embodied_model, task, sensor, epochs=50, with_stress=True)

    print("\nTraining BLIND model...")
    blind_train = train_model(blind_model, task, sensor, epochs=50, with_stress=True)

    # Evaluate under stress
    print("\nEvaluating EMBODIED model under stress...")
    embodied_eval = evaluate_under_stress(embodied_model, task, sensor, n_eval=60)

    print("\nEvaluating BLIND model under stress...")
    blind_eval = evaluate_under_stress(blind_model, task, sensor, n_eval=60)

    # Compare results
    print("\n" + "=" * 70)
    print("RESULTS: Accuracy by Stress Level")
    print("=" * 70)

    print(f"\n{'Stress':<10} | {'Embodied':>12} | {'Blind':>12} | {'Difference':>12}")
    print("-" * 52)

    embodied_wins = 0
    total_diff = 0

    for stress in ['none', 'light', 'medium', 'heavy', 'extreme']:
        e_acc = embodied_eval[stress]['mean_accuracy']
        b_acc = blind_eval[stress]['mean_accuracy']
        diff = e_acc - b_acc

        if diff > 0.01:
            embodied_wins += 1
            marker = "✓"
        elif diff < -0.01:
            marker = "✗"
        else:
            marker = "-"

        total_diff += diff
        print(f"{stress:<10} | {e_acc:>11.1%} | {b_acc:>11.1%} | {diff:>+11.1%} {marker}")

    avg_diff = total_diff / 5

    print("-" * 52)
    print(f"{'Average':<10} | {'':<12} | {'':<12} | {avg_diff:>+11.1%}")

    # Degradation analysis: How much does each model degrade under stress?
    print("\n" + "=" * 70)
    print("DEGRADATION ANALYSIS (Heavy stress vs None)")
    print("=" * 70)

    e_degrade = embodied_eval['none']['mean_accuracy'] - embodied_eval['heavy']['mean_accuracy']
    b_degrade = blind_eval['none']['mean_accuracy'] - blind_eval['heavy']['mean_accuracy']

    print(f"\nEmbodied degradation: {e_degrade:+.1%}")
    print(f"Blind degradation:    {b_degrade:+.1%}")

    if e_degrade < b_degrade:
        print("✓ Embodied is MORE RESILIENT to stress")
        resilience_win = True
    else:
        print("✗ Embodied is NOT more resilient")
        resilience_win = False

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if embodied_wins >= 3 and avg_diff > 0.02:
        verdict = "EMBODIED MODEL SHOWS TASK ADVANTAGE"
        print(f"\n✅ {verdict}")
        print(f"   Embodied wins {embodied_wins}/5 stress levels")
        print(f"   Average advantage: {avg_diff:+.1%}")
    elif resilience_win and embodied_wins >= 2:
        verdict = "EMBODIED MODEL SHOWS RESILIENCE ADVANTAGE"
        print(f"\n✅ {verdict}")
        print(f"   Less degradation under stress")
    else:
        verdict = "NO CLEAR EMBODIMENT ADVANTAGE"
        print(f"\n❌ {verdict}")
        print(f"   Embodied wins only {embodied_wins}/5 stress levels")
        print(f"   Average difference: {avg_diff:+.1%}")

    # Save results
    output = {
        'experiment': 'z1313_task_embodiment',
        'timestamp': datetime.now().isoformat(),
        'embodied_eval': embodied_eval,
        'blind_eval': blind_eval,
        'embodied_wins': embodied_wins,
        'average_diff': float(avg_diff),
        'embodied_degradation': float(e_degrade),
        'blind_degradation': float(b_degrade),
        'resilience_advantage': resilience_win,
        'verdict': verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1313_task_embodiment.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
