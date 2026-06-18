#!/usr/bin/env python3
"""
z1316: Verify Causal Signal Usage

z1315 showed 85% improvement. But does the model actually use the temp_derivative?
This test masks different features to verify which signal is causal.

Controls:
1. FULL - All features (should be best)
2. MASK_DERIVATIVE - Zero out temp_derivative (should be worst)
3. MASK_TEMP - Zero out temperature (should still work)
4. MASK_POWER - Zero out power (should still work)
5. MASK_UTIL - Zero out utilization (should still work)

If MASK_DERIVATIVE performs much worse than FULL, the model genuinely
uses the causal hardware signal.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HardwareSensor:
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

    def read(self, mask: str = None) -> Tuple[torch.Tensor, float]:
        """Get hardware state with optional masking"""
        temp_raw = self._hwmon('temp1_input', 50000)
        temp_c = temp_raw / 1000
        power_w = self._hwmon('power1_average', 50e6) / 1e6
        util = self._read('gpu_busy_percent', 50)

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

        # Build state with masking
        state = [
            temp_c / 100,
            power_w / 100,
            util / 100,
            np.clip(temp_derivative, -5, 5) / 5,
        ]

        # Apply mask
        if mask == 'derivative':
            state[3] = 0.0
        elif mask == 'temp':
            state[0] = 0.0
        elif mask == 'power':
            state[1] = 0.0
        elif mask == 'util':
            state[2] = 0.0

        return torch.tensor(state, dtype=torch.float32), temp_derivative


class HardwareDriftTask:
    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale
        self.y = 0.5
        self.drift_scale = 0.3

    def step(self, temp_derivative: float) -> Tuple[float, float]:
        drift = self.drift_scale * temp_derivative
        noise = np.random.normal(0, self.noise_scale)
        old_y = self.y
        self.y = np.clip(self.y + drift + noise, 0, 1)
        return old_y, self.y


class EmbodiedDriftModel(nn.Module):
    def __init__(self, hw_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.hw_encoder = nn.Sequential(
            nn.Linear(hw_dim, 32),
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
        hw_encoded = self.hw_encoder(hw_state)
        x = torch.cat([hw_encoded, current_y.unsqueeze(-1)], dim=-1)
        return self.drift_predictor(x).squeeze(-1)


def create_thermal_variation():
    intensity = np.random.choice(['none', 'light', 'heavy'])
    if intensity == 'none':
        pass
    elif intensity == 'light':
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    else:
        for _ in range(3):
            _ = torch.randn(2000, 2000, device=DEVICE) @ torch.randn(2000, 2000, device=DEVICE)


def train_model(sensor: HardwareSensor, mask: str = None, n_episodes: int = 30) -> nn.Module:
    """Train model with specified feature mask"""
    model = EmbodiedDriftModel(hw_dim=4, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(n_episodes):
        task = HardwareDriftTask(noise_scale=0.1)
        task.y = 0.5

        for step in range(40):
            create_thermal_variation()
            time.sleep(0.05)

            hw_state, temp_derivative = sensor.read(mask=mask)
            hw_state = hw_state.unsqueeze(0).to(DEVICE)
            current_y = torch.tensor([task.y], dtype=torch.float32, device=DEVICE)
            _, next_y_true = task.step(temp_derivative)
            next_y_true = torch.tensor([next_y_true], dtype=torch.float32, device=DEVICE)

            pred_y = model(hw_state, current_y)
            loss = F.mse_loss(pred_y, next_y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model: nn.Module, sensor: HardwareSensor, mask: str = None,
                   n_episodes: int = 20) -> list:
    """Evaluate model with specified feature mask"""
    model.eval()
    maes = []

    for ep in range(n_episodes):
        task = HardwareDriftTask(noise_scale=0.1)
        task.y = 0.5
        episode_mae = 0

        for step in range(40):
            create_thermal_variation()
            time.sleep(0.05)

            hw_state, temp_derivative = sensor.read(mask=mask)
            hw_state = hw_state.unsqueeze(0).to(DEVICE)
            current_y = torch.tensor([task.y], dtype=torch.float32, device=DEVICE)
            _, next_y_true = task.step(temp_derivative)

            with torch.no_grad():
                pred_y = model(hw_state, current_y)
                mae = abs(pred_y.item() - next_y_true)
                episode_mae += mae

        maes.append(episode_mae / 40)

    return maes


def main():
    print("=" * 70)
    print("  z1316: VERIFY CAUSAL SIGNAL USAGE")
    print("  Which feature is the model actually using?")
    print("=" * 70)
    print()

    sensor = HardwareSensor()

    conditions = [
        ('FULL', None),
        ('MASK_DERIVATIVE', 'derivative'),
        ('MASK_TEMP', 'temp'),
        ('MASK_POWER', 'power'),
        ('MASK_UTIL', 'util'),
    ]

    results = {}

    for name, mask in conditions:
        print(f"\n{'='*40}")
        print(f"Condition: {name}")
        print(f"{'='*40}")

        print(f"Training...")
        model = train_model(sensor, mask=mask, n_episodes=25)

        print(f"Evaluating...")
        maes = evaluate_model(model, sensor, mask=mask, n_episodes=15)

        mean_mae = np.mean(maes)
        std_mae = np.std(maes)

        results[name] = {
            'maes': maes,
            'mean': mean_mae,
            'std': std_mae,
        }

        print(f"  MAE: {mean_mae:.4f} ± {std_mae:.4f}")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    full_mae = results['FULL']['mean']
    print(f"\n{'Condition':<20} | {'MAE':>10} | {'vs FULL':>12} | {'Verdict':>12}")
    print("-" * 60)

    for name in ['FULL', 'MASK_DERIVATIVE', 'MASK_TEMP', 'MASK_POWER', 'MASK_UTIL']:
        mae = results[name]['mean']
        diff_pct = (mae - full_mae) / full_mae * 100 if name != 'FULL' else 0

        if name == 'FULL':
            verdict = "baseline"
        elif diff_pct > 100:
            verdict = "CRITICAL ⚠️"
        elif diff_pct > 30:
            verdict = "important"
        elif diff_pct > 10:
            verdict = "useful"
        else:
            verdict = "not used"

        print(f"{name:<20} | {mae:>10.4f} | {diff_pct:>+11.1f}% | {verdict}")

    # Statistical test: FULL vs MASK_DERIVATIVE
    t_stat, p_value = stats.ttest_ind(results['FULL']['maes'], results['MASK_DERIVATIVE']['maes'])
    derivative_importance = (results['MASK_DERIVATIVE']['mean'] - results['FULL']['mean']) / results['FULL']['mean'] * 100

    print("\n" + "=" * 70)
    print("CAUSAL VERIFICATION")
    print("=" * 70)

    print(f"\nMasking temp_derivative increases error by {derivative_importance:+.1f}%")
    print(f"p-value (FULL vs MASK_DERIVATIVE): {p_value:.6f}")

    if derivative_importance > 100 and p_value < 0.01:
        verdict = "VERIFIED: Model uses temp_derivative causally"
        causal_verified = True
        print(f"\n✅ {verdict}")
    elif derivative_importance > 30 and p_value < 0.05:
        verdict = "PARTIALLY VERIFIED: Some causal use of temp_derivative"
        causal_verified = True
        print(f"\n⚠️ {verdict}")
    else:
        verdict = "NOT VERIFIED: Model may not use temp_derivative"
        causal_verified = False
        print(f"\n❌ {verdict}")

    # Save results
    output = {
        'experiment': 'z1316_verify_causal_signal',
        'timestamp': datetime.now().isoformat(),
        'results': {name: {'mean': results[name]['mean'], 'std': results[name]['std']}
                    for name in results},
        'derivative_importance_pct': derivative_importance,
        'p_value': p_value,
        'causal_verified': causal_verified,
        'verdict': verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1316_verify_causal_signal.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
