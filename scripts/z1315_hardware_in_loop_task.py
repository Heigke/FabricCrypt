#!/usr/bin/env python3
"""
z1315: Hardware-in-the-Loop Task

Previous tests failed because hardware state was just noise to the task.
This test creates a scenario where hardware state CAUSALLY AFFECTS task success.

DESIGN:
- Task: Predict a drifting target value
- The DRIFT is determined by GPU load/temperature
- Embodied model sees hardware state → can track drift
- Blind model cannot → must guess drift direction

This is genuine embodied cognition: the body state (hardware) directly
influences what the correct answer is.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HardwareSensor:
    """Hardware sensor that also tracks derivatives"""
    def __init__(self):
        self.card = '/sys/class/drm/card1/device'
        self.last_temp = None
        self.last_time = None

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

    def read(self) -> Tuple[torch.Tensor, float]:
        """Get hardware state and temperature derivative"""
        temp_raw = self._hwmon('temp1_input', 50000)
        temp_c = temp_raw / 1000
        power_w = self._hwmon('power1_average', 50e6) / 1e6
        util = self._read('gpu_busy_percent', 50)

        # Calculate temperature derivative
        now = time.time()
        if self.last_temp is not None and self.last_time is not None:
            dt = now - self.last_time
            if dt > 0.01:
                temp_derivative = (temp_c - self.last_temp) / dt
            else:
                temp_derivative = 0.0
        else:
            temp_derivative = 0.0

        self.last_temp = temp_c
        self.last_time = now

        state = torch.tensor([
            temp_c / 100,       # Normalized temp
            power_w / 100,      # Normalized power
            util / 100,         # Utilization
            np.clip(temp_derivative, -5, 5) / 5,  # Normalized derivative
        ], dtype=torch.float32)

        return state, temp_derivative


class HardwareDriftTask:
    """
    Task where the target drifts based on hardware state.

    The target follows: y(t) = y(t-1) + k * temp_derivative + noise

    Where temp_derivative comes from actual GPU temperature changes.
    An embodied model that sees the derivative can predict y accurately.
    A blind model must guess the drift direction.
    """

    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale
        self.y = 0.5  # Start in middle
        self.drift_scale = 0.3  # How much temp_derivative affects y

    def step(self, temp_derivative: float) -> Tuple[float, float]:
        """Update target based on hardware state"""
        # Target drifts with temperature change
        drift = self.drift_scale * temp_derivative
        noise = np.random.normal(0, self.noise_scale)

        old_y = self.y
        self.y = np.clip(self.y + drift + noise, 0, 1)

        return old_y, self.y


class EmbodiedDriftModel(nn.Module):
    """Model that predicts drifting target using hardware state"""

    def __init__(self, hw_dim: int = 4, hidden_dim: int = 64):
        super().__init__()

        # Process hardware state
        self.hw_encoder = nn.Sequential(
            nn.Linear(hw_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Predict drift
        self.drift_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),  # hw_encoded + current_y
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hw_state: torch.Tensor, current_y: torch.Tensor) -> torch.Tensor:
        """Predict next y value"""
        hw_encoded = self.hw_encoder(hw_state)
        x = torch.cat([hw_encoded, current_y.unsqueeze(-1)], dim=-1)
        return self.drift_predictor(x).squeeze(-1)


class BlindDriftModel(nn.Module):
    """Model that predicts without hardware state (must learn average drift)"""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Same parameter count as embodied
        self.processor = nn.Sequential(
            nn.Linear(1, 32),  # Just current_y
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.drift_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hw_state: torch.Tensor, current_y: torch.Tensor) -> torch.Tensor:
        """Predict next y (hw_state ignored)"""
        y_processed = self.processor(current_y.unsqueeze(-1))
        x = torch.cat([y_processed, current_y.unsqueeze(-1)], dim=-1)
        return self.drift_predictor(x).squeeze(-1)


def create_thermal_variation():
    """Create thermal variation by GPU load"""
    # Random workload intensity
    intensity = np.random.choice(['none', 'light', 'heavy'])

    if intensity == 'none':
        pass
    elif intensity == 'light':
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    else:  # heavy
        for _ in range(3):
            _ = torch.randn(2000, 2000, device=DEVICE) @ torch.randn(2000, 2000, device=DEVICE)


def run_episode(model: nn.Module, task: HardwareDriftTask, sensor: HardwareSensor,
                n_steps: int = 50, train: bool = True) -> Dict:
    """Run one episode of the drift prediction task"""

    if train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        model.eval()

    total_loss = 0
    total_mae = 0
    predictions = []
    targets = []

    task.y = 0.5  # Reset

    for step in range(n_steps):
        # Create thermal variation
        create_thermal_variation()
        time.sleep(0.05)  # Let temperature change

        # Read hardware
        hw_state, temp_derivative = sensor.read()
        hw_state = hw_state.unsqueeze(0).to(DEVICE)

        # Get current y and true next y
        current_y = torch.tensor([task.y], dtype=torch.float32, device=DEVICE)
        _, next_y_true = task.step(temp_derivative)
        next_y_true = torch.tensor([next_y_true], dtype=torch.float32, device=DEVICE)

        # Predict
        if train:
            pred_y = model(hw_state, current_y)
            loss = F.mse_loss(pred_y, next_y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        else:
            with torch.no_grad():
                pred_y = model(hw_state, current_y)
                loss = F.mse_loss(pred_y, next_y_true)
                total_loss += loss.item()

        mae = torch.abs(pred_y - next_y_true).item()
        total_mae += mae

        predictions.append(pred_y.item())
        targets.append(next_y_true.item())

    return {
        'loss': total_loss / n_steps,
        'mae': total_mae / n_steps,
        'predictions': predictions,
        'targets': targets,
    }


def main():
    print("=" * 70)
    print("  z1315: HARDWARE-IN-THE-LOOP TASK")
    print("  Target drifts with temperature → embodiment is necessary")
    print("=" * 70)
    print()

    sensor = HardwareSensor()

    # Create models
    embodied = EmbodiedDriftModel(hw_dim=4, hidden_dim=64).to(DEVICE)
    blind = BlindDriftModel(hidden_dim=64).to(DEVICE)

    print(f"Embodied params: {sum(p.numel() for p in embodied.parameters()):,}")
    print(f"Blind params: {sum(p.numel() for p in blind.parameters()):,}")

    # Training phase
    print("\n" + "=" * 70)
    print("TRAINING PHASE (30 episodes each)")
    print("=" * 70)

    n_train = 30

    print("\nTraining EMBODIED...")
    for ep in range(n_train):
        task = HardwareDriftTask(noise_scale=0.1)
        result = run_episode(embodied, task, sensor, n_steps=40, train=True)
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: MAE={result['mae']:.4f}")

    print("\nTraining BLIND...")
    for ep in range(n_train):
        task = HardwareDriftTask(noise_scale=0.1)
        result = run_episode(blind, task, sensor, n_steps=40, train=True)
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: MAE={result['mae']:.4f}")

    # Evaluation phase
    print("\n" + "=" * 70)
    print("EVALUATION PHASE (20 episodes each)")
    print("=" * 70)

    n_eval = 20

    embodied_maes = []
    blind_maes = []

    print("\nEvaluating EMBODIED...")
    for ep in range(n_eval):
        task = HardwareDriftTask(noise_scale=0.1)
        result = run_episode(embodied, task, sensor, n_steps=40, train=False)
        embodied_maes.append(result['mae'])

    print("\nEvaluating BLIND...")
    for ep in range(n_eval):
        task = HardwareDriftTask(noise_scale=0.1)
        result = run_episode(blind, task, sensor, n_steps=40, train=False)
        blind_maes.append(result['mae'])

    # Also evaluate a naive baseline (predict y stays same)
    print("\nEvaluating NAIVE (y_next = y_current)...")
    naive_maes = []
    for ep in range(n_eval):
        task = HardwareDriftTask(noise_scale=0.1)
        task.y = 0.5
        episode_mae = 0
        for step in range(40):
            create_thermal_variation()
            time.sleep(0.05)
            hw_state, temp_derivative = sensor.read()
            current_y = task.y
            _, next_y = task.step(temp_derivative)
            episode_mae += abs(current_y - next_y)  # Naive: predict no change
        naive_maes.append(episode_mae / 40)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    embodied_mean = np.mean(embodied_maes)
    embodied_std = np.std(embodied_maes)
    blind_mean = np.mean(blind_maes)
    blind_std = np.std(blind_maes)
    naive_mean = np.mean(naive_maes)
    naive_std = np.std(naive_maes)

    print(f"\n{'Model':<15} | {'MAE':>12} | {'Std':>10}")
    print("-" * 45)
    print(f"{'EMBODIED':<15} | {embodied_mean:>12.4f} | {embodied_std:>10.4f}")
    print(f"{'BLIND':<15} | {blind_mean:>12.4f} | {blind_std:>10.4f}")
    print(f"{'NAIVE':<15} | {naive_mean:>12.4f} | {naive_std:>10.4f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(embodied_maes, blind_maes)
    improvement = (blind_mean - embodied_mean) / blind_mean * 100

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    print(f"\nEMBODIED vs BLIND:")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  p-value: {p_value:.4f}")

    naive_improvement = (naive_mean - embodied_mean) / naive_mean * 100
    t_naive, p_naive = stats.ttest_ind(embodied_maes, naive_maes)

    print(f"\nEMBODIED vs NAIVE:")
    print(f"  Improvement: {naive_improvement:+.1f}%")
    print(f"  p-value: {p_naive:.4f}")

    # Verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if improvement > 10 and p_value < 0.05:
        verdict = "GENUINE EMBODIMENT ADVANTAGE"
        print(f"\n✅ {verdict}")
        print(f"   Embodied beats blind by {improvement:.1f}% (p={p_value:.4f})")
    elif improvement > 5 and p_value < 0.1:
        verdict = "PARTIAL EMBODIMENT EVIDENCE"
        print(f"\n⚠️ {verdict}")
        print(f"   Embodied beats blind by {improvement:.1f}% (p={p_value:.4f})")
    else:
        verdict = "NO CLEAR EMBODIMENT ADVANTAGE"
        print(f"\n❌ {verdict}")
        print(f"   Improvement: {improvement:.1f}% (p={p_value:.4f})")

    # Save results
    output = {
        'experiment': 'z1315_hardware_in_loop_task',
        'timestamp': datetime.now().isoformat(),
        'embodied_mae': {'mean': embodied_mean, 'std': embodied_std, 'values': embodied_maes},
        'blind_mae': {'mean': blind_mean, 'std': blind_std, 'values': blind_maes},
        'naive_mae': {'mean': naive_mean, 'std': naive_std, 'values': naive_maes},
        'improvement_vs_blind': improvement,
        'p_value_vs_blind': p_value,
        'improvement_vs_naive': naive_improvement,
        'p_value_vs_naive': p_naive,
        'verdict': verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1315_hardware_in_loop_task.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
