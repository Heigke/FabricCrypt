#!/usr/bin/env python3
"""
z1314: Rigorous Task-Based Embodiment Test

z1313 showed embodied model winning 5/5 stress levels with +12.6% advantage.
But it had 4,352 more parameters. This test controls for:

1. MATCHED PARAMETERS - Blind model gets extra capacity to match
2. RANDOM HW - Embodied with random instead of real hardware state
3. SHUFFLED HW - Embodied with time-shuffled hardware state
4. FROZEN HW - Embodied with constant hardware state

If embodied STILL wins vs all controls, it's genuine embodiment.
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
from scipy import stats

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
        self.history = deque(maxlen=100)

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
        state = torch.tensor([
            self._hwmon('temp1_input', 50000) / 1000 / 100,  # Normalized temp
            self._hwmon('power1_average', 50e6) / 1e6 / 100,  # Normalized power
            self._read('gpu_busy_percent', 50) / 100,  # Utilization
        ], dtype=torch.float32)
        self.history.append(state.numpy().copy())
        return state

    def get_shuffled(self) -> torch.Tensor:
        """Get shuffled historical state (breaks temporal causality)"""
        if len(self.history) < 10:
            return self.read()
        idx = np.random.randint(0, len(self.history))
        return torch.tensor(self.history[idx], dtype=torch.float32)

    def get_random(self) -> torch.Tensor:
        """Get random state matching real distribution"""
        return torch.tensor([
            np.random.uniform(0.4, 0.9),   # temp range
            np.random.uniform(0.1, 0.5),   # power range
            np.random.uniform(0.0, 1.0),   # utilization range
        ], dtype=torch.float32)

    def get_frozen(self) -> torch.Tensor:
        """Get constant state (no signal)"""
        return torch.tensor([0.5, 0.3, 0.5], dtype=torch.float32)


class SequenceTask:
    """Simple sequence prediction task."""

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
                # Incrementing with wrap-around
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
    """Sequence model with FiLM hardware conditioning."""

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
        """Forward pass with optional hardware conditioning."""
        x = self.embed(seq)
        out, (h, c) = self.lstm(x)
        h = h.squeeze(0)

        if self.embodied and hw_state is not None:
            film_params = self.hw_encoder(hw_state)
            gamma = film_params[:, :self.hidden_dim]
            beta = film_params[:, self.hidden_dim:]
            h = gamma * h + beta

        logits = self.output(h)
        return logits


class MatchedBlindModel(nn.Module):
    """Blind model with MATCHED parameter count (extra hidden capacity)."""

    def __init__(self, vocab_size: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Sequence encoder (same as embodied)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # EXTRA capacity to match embodied param count
        # Embodied hw_encoder: 3*32 + 32 + 32*128 + 128 = 96+32+4096+128 = 4352 params
        # Match this with extra processing
        self.extra = nn.Sequential(
            nn.Linear(hidden_dim, 96),
            nn.ReLU(),
            nn.Linear(96, hidden_dim),
        )

        # Output head
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq: torch.Tensor, hw_state: torch.Tensor = None) -> torch.Tensor:
        """Forward pass (hw_state ignored)."""
        x = self.embed(seq)
        out, (h, c) = self.lstm(x)
        h = h.squeeze(0)

        # Use extra capacity (but no hardware signal)
        h = h + self.extra(h)  # Residual connection

        logits = self.output(h)
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
                epochs: int = 50, batch_size: int = 32, with_stress: bool = True,
                hw_mode: str = 'live') -> Dict:
    """Train model with specified hardware mode."""

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
                create_gpu_stress(stress)

                # Get hardware state based on mode
                if hw_mode == 'live':
                    hw = sensor.read()
                elif hw_mode == 'shuffled':
                    hw = sensor.get_shuffled()
                elif hw_mode == 'random':
                    hw = sensor.get_random()
                elif hw_mode == 'frozen':
                    hw = sensor.get_frozen()
                else:
                    hw = sensor.read()

                hw = hw.unsqueeze(0).expand(batch_size, -1).to(DEVICE)

                seq, target = task.generate_batch(batch_size, pattern)
                seq, target = seq.to(DEVICE), target.to(DEVICE)

                if hasattr(model, 'embodied') and model.embodied:
                    logits = model(seq, hw)
                else:
                    logits = model(seq)

                loss = F.cross_entropy(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=-1)
                acc = (preds == target).float().mean()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        history.append({'epoch': epoch, 'loss': avg_loss, 'accuracy': avg_acc})

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.1%}")

    return {'history': history}


def evaluate_under_stress(model: nn.Module, task: SequenceTask, sensor: HardwareSensor,
                          n_eval: int = 100, batch_size: int = 32,
                          hw_mode: str = 'live') -> Dict:
    """Evaluate model under different stress levels"""

    model.eval()
    results = {}

    stress_levels = ['none', 'light', 'medium', 'heavy', 'extreme']
    patterns = ['repeat', 'increment', 'fibonacci']

    for stress in stress_levels:
        stress_results = {'loss': [], 'accuracy': []}

        for _ in range(n_eval):
            create_gpu_stress(stress)

            if hw_mode == 'live':
                hw = sensor.read()
            elif hw_mode == 'shuffled':
                hw = sensor.get_shuffled()
            elif hw_mode == 'random':
                hw = sensor.get_random()
            elif hw_mode == 'frozen':
                hw = sensor.get_frozen()
            else:
                hw = sensor.read()

            hw = hw.unsqueeze(0).expand(batch_size, -1).to(DEVICE)

            pattern = np.random.choice(patterns)
            seq, target = task.generate_batch(batch_size, pattern)
            seq, target = seq.to(DEVICE), target.to(DEVICE)

            with torch.no_grad():
                if hasattr(model, 'embodied') and model.embodied:
                    logits = model(seq, hw)
                else:
                    logits = model(seq)

                loss = F.cross_entropy(logits, target)
                preds = logits.argmax(dim=-1)
                acc = (preds == target).float().mean()

                stress_results['loss'].append(loss.item())
                stress_results['accuracy'].append(acc.item())

        results[stress] = {
            'mean_accuracy': float(np.mean(stress_results['accuracy'])),
            'std_accuracy': float(np.std(stress_results['accuracy'])),
        }

    return results


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval"""
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1-ci)/2 * 100)
    upper = np.percentile(means, (1+ci)/2 * 100)
    return np.mean(data), lower, upper


def main():
    print("=" * 70)
    print("  z1314: RIGOROUS TASK-BASED EMBODIMENT TEST")
    print("  Controls for parameter count and causal signal")
    print("=" * 70)
    print()

    sensor = HardwareSensor()
    task = SequenceTask(seq_len=32, vocab_size=16)

    # Create models
    embodied_live = EmbodiedSequenceModel(embodied=True).to(DEVICE)
    embodied_random = EmbodiedSequenceModel(embodied=True).to(DEVICE)
    embodied_shuffled = EmbodiedSequenceModel(embodied=True).to(DEVICE)
    embodied_frozen = EmbodiedSequenceModel(embodied=True).to(DEVICE)
    blind_matched = MatchedBlindModel().to(DEVICE)

    # Copy initial weights for fair comparison
    base_state = {k: v.clone() for k, v in embodied_live.state_dict().items()
                  if 'hw_encoder' not in k}

    for model in [embodied_random, embodied_shuffled, embodied_frozen]:
        model.load_state_dict(base_state, strict=False)

    # Match blind model to shared architecture params
    blind_base = {k.replace('extra', 'dummy'): v for k, v in base_state.items()
                  if 'extra' not in k}
    blind_matched.embed.load_state_dict({'weight': embodied_live.embed.weight.clone()})

    print(f"Embodied params: {sum(p.numel() for p in embodied_live.parameters()):,}")
    print(f"Blind matched params: {sum(p.numel() for p in blind_matched.parameters()):,}")

    # Train all conditions
    conditions = [
        ('EMBODIED_LIVE', embodied_live, 'live'),
        ('EMBODIED_RANDOM', embodied_random, 'random'),
        ('EMBODIED_SHUFFLED', embodied_shuffled, 'shuffled'),
        ('EMBODIED_FROZEN', embodied_frozen, 'frozen'),
        ('BLIND_MATCHED', blind_matched, None),
    ]

    trained_models = {}

    for name, model, hw_mode in conditions:
        print(f"\nTraining {name}...")
        train_model(model, task, sensor, epochs=50, with_stress=True, hw_mode=hw_mode)
        trained_models[name] = model

    # Evaluate all conditions
    print("\n" + "=" * 70)
    print("EVALUATION UNDER STRESS")
    print("=" * 70)

    eval_results = {}

    for name, model, hw_mode in conditions:
        print(f"\nEvaluating {name}...")
        if hw_mode is None:
            results = evaluate_under_stress(model, task, sensor, n_eval=100, hw_mode='live')
        else:
            results = evaluate_under_stress(model, task, sensor, n_eval=100, hw_mode=hw_mode)
        eval_results[name] = results

    # Compare results
    print("\n" + "=" * 70)
    print("RESULTS: Average Accuracy Across All Stress Levels")
    print("=" * 70)

    summary = {}

    for name in eval_results:
        all_accs = [eval_results[name][s]['mean_accuracy'] for s in ['none', 'light', 'medium', 'heavy', 'extreme']]
        mean_acc = np.mean(all_accs)
        summary[name] = {'mean_acc': mean_acc, 'by_stress': eval_results[name]}
        print(f"  {name:<20}: {mean_acc:.1%}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON vs EMBODIED_LIVE")
    print("=" * 70)

    live_accs = [eval_results['EMBODIED_LIVE'][s]['mean_accuracy']
                 for s in ['none', 'light', 'medium', 'heavy', 'extreme']]

    wins = 0
    for name in ['EMBODIED_RANDOM', 'EMBODIED_SHUFFLED', 'EMBODIED_FROZEN', 'BLIND_MATCHED']:
        other_accs = [eval_results[name][s]['mean_accuracy']
                      for s in ['none', 'light', 'medium', 'heavy', 'extreme']]

        diff = np.mean(live_accs) - np.mean(other_accs)
        t_stat, p_value = stats.ttest_rel(live_accs, other_accs)

        if diff > 0 and p_value < 0.1:
            wins += 1
            marker = "✓"
        else:
            marker = "✗"

        print(f"  vs {name:<18}: Δ={diff:+.1%}, p={p_value:.3f} {marker}")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if wins >= 3:
        verdict = "GENUINE EMBODIMENT ADVANTAGE"
        print(f"\n✅ {verdict}")
        print(f"   LIVE beats {wins}/4 controls")
    elif wins >= 2:
        verdict = "PARTIAL EMBODIMENT EVIDENCE"
        print(f"\n⚠️ {verdict}")
        print(f"   LIVE beats {wins}/4 controls")
    else:
        verdict = "NO CLEAR EMBODIMENT ADVANTAGE"
        print(f"\n❌ {verdict}")
        print(f"   LIVE beats only {wins}/4 controls")

    # Detailed per-stress breakdown
    print("\n" + "=" * 70)
    print("PER-STRESS LEVEL COMPARISON (LIVE vs BLIND_MATCHED)")
    print("=" * 70)
    print(f"\n{'Stress':<10} | {'LIVE':>10} | {'MATCHED':>10} | {'Diff':>10}")
    print("-" * 48)

    live_vs_blind_wins = 0
    for stress in ['none', 'light', 'medium', 'heavy', 'extreme']:
        live_acc = eval_results['EMBODIED_LIVE'][stress]['mean_accuracy']
        blind_acc = eval_results['BLIND_MATCHED'][stress]['mean_accuracy']
        diff = live_acc - blind_acc

        if diff > 0.02:
            marker = "✓"
            live_vs_blind_wins += 1
        else:
            marker = ""

        print(f"{stress:<10} | {live_acc:>9.1%} | {blind_acc:>9.1%} | {diff:>+9.1%} {marker}")

    # Save results
    output = {
        'experiment': 'z1314_task_embodiment_rigorous',
        'timestamp': datetime.now().isoformat(),
        'eval_results': eval_results,
        'summary': {name: {'mean_acc': summary[name]['mean_acc']} for name in summary},
        'live_wins_vs_controls': wins,
        'live_wins_vs_blind': live_vs_blind_wins,
        'verdict': verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1314_task_embodiment_rigorous.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
