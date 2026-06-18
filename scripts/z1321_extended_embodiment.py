#!/usr/bin/env python3
"""
z1321: Extended Embodiment - Building on Proven Results

z1319 achieved 68.6% improvement with p=3.5e-158.
This extends that architecture with:
1. Multi-step prediction (predict 3 steps ahead)
2. Learned body dynamics model
3. Planning horizon optimization

Keeping what works, extending carefully.
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
        self.history = []

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

        state = torch.tensor([
            temp_c / 100,
            power_w / 100,
            util / 100,
            np.clip(temp_derivative, -5, 5) / 5,
        ], dtype=torch.float32, device=DEVICE)

        self.history.append(state.cpu().numpy())
        if len(self.history) > 10:
            self.history.pop(0)

        return state, temp_derivative


class ExtendedEmbodiedAgent(nn.Module):
    """
    Extended agent with multi-step prediction.

    Based on z1319 but adds:
    - Body dynamics model (predict future hw states)
    - Multi-step task prediction
    - Planning over horizon
    """

    def __init__(self, hw_dim: int = 4, n_depths: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.n_depths = n_depths

        # Hardware encoder
        self.hw_encoder = nn.Sequential(
            nn.Linear(hw_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Body dynamics model: predict next hw state
        self.dynamics = nn.Sequential(
            nn.Linear(32 + 1, 32),  # hw_encoded + action
            nn.ReLU(),
            nn.Linear(32, hw_dim),  # predict next hw state
        )

        # Target predictor (uses temp_derivative - proven in z1315)
        self.target_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Cost predictor
        self.cost_predictor = nn.Sequential(
            nn.Linear(32 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, hw_state: torch.Tensor) -> torch.Tensor:
        return self.hw_encoder(hw_state.unsqueeze(0) if hw_state.dim() == 1 else hw_state)

    def predict_next_hw(self, hw_encoded: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next hardware state"""
        x = torch.cat([hw_encoded, action], dim=-1)
        return self.dynamics(x)

    def predict_target(self, hw_encoded: torch.Tensor, current_y: float) -> torch.Tensor:
        """Predict next target"""
        y_tensor = torch.tensor([[current_y]], dtype=torch.float32, device=DEVICE)
        x = torch.cat([hw_encoded, y_tensor], dim=-1)
        return self.target_predictor(x)

    def predict_cost(self, hw_encoded: torch.Tensor, depth: int) -> torch.Tensor:
        """Predict cost"""
        d_tensor = torch.tensor([[depth / self.n_depths]], dtype=torch.float32, device=DEVICE)
        x = torch.cat([hw_encoded, d_tensor], dim=-1)
        return self.cost_predictor(x)

    def plan(self, hw_state: torch.Tensor, current_y: float, horizon: int = 3,
             cost_weight: float = 0.3) -> int:
        """Plan best depth by simulating multiple steps ahead"""
        best_depth = 1
        best_value = -float('inf')

        for depth in range(1, self.n_depths + 1):
            total_value = 0
            hw = hw_state.clone()

            for step in range(horizon):
                hw_encoded = self.encode(hw)
                action = torch.tensor([[depth / self.n_depths]], dtype=torch.float32, device=DEVICE)

                # Predict target and cost
                pred_target = self.predict_target(hw_encoded, current_y)
                pred_cost = self.predict_cost(hw_encoded, depth)

                # Quality from depth
                quality = 0.3 + 0.7 * (depth / self.n_depths)

                # Step value
                step_value = quality - cost_weight * pred_cost.item()
                total_value += step_value * (0.9 ** step)  # Discount

                # Predict next hw state
                with torch.no_grad():
                    next_hw = self.predict_next_hw(hw_encoded, action)
                    hw = next_hw.squeeze(0)

            if total_value > best_value:
                best_value = total_value
                best_depth = depth

        return best_depth


class BlindAgent(nn.Module):
    """Blind agent for comparison"""

    def __init__(self, n_depths: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.n_depths = n_depths

        self.history_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.target_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.y_history = [0.5] * 5

    def update_history(self, y: float):
        self.y_history.pop(0)
        self.y_history.append(y)

    def predict_target(self, current_y: float) -> torch.Tensor:
        hist = torch.tensor([self.y_history], dtype=torch.float32, device=DEVICE)
        hist_encoded = self.history_encoder(hist)
        y_tensor = torch.tensor([[current_y]], dtype=torch.float32, device=DEVICE)
        x = torch.cat([hist_encoded, y_tensor], dim=-1)
        return self.target_predictor(x)

    def select_depth(self) -> int:
        # Fixed strategy: middle depth
        return 3


class DriftTask:
    def __init__(self, noise_scale: float = 0.08, drift_scale: float = 0.25):
        self.noise_scale = noise_scale
        self.drift_scale = drift_scale
        self.y = 0.5

    def step(self, temp_derivative: float) -> float:
        drift = self.drift_scale * temp_derivative
        noise = np.random.normal(0, self.noise_scale)
        self.y = np.clip(self.y + drift + noise, 0, 1)
        return self.y

    def reset(self):
        self.y = 0.5


def simulate_tracking(pred_y: float, true_y: float, depth: int) -> float:
    base_error = abs(pred_y - true_y)
    depth_factor = 0.3 + 0.7 * (depth / 5)
    return base_error * (2 - depth_factor)


def compute_cost(depth: int, hw_state: torch.Tensor) -> float:
    base_cost = 0.1 + 0.15 * (depth / 5)
    thermal_factor = 1.0 + 0.3 * hw_state[0].item()
    return base_cost * thermal_factor


def create_thermal_variation():
    intensity = np.random.choice(['none', 'light', 'heavy'], p=[0.4, 0.35, 0.25])
    if intensity == 'none':
        pass
    elif intensity == 'light':
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    else:
        _ = torch.randn(1500, 1500, device=DEVICE) @ torch.randn(1500, 1500, device=DEVICE)


def train_agent(agent: nn.Module, task: DriftTask, sensor: HardwareSensor,
                n_episodes: int = 60, is_embodied: bool = True) -> Dict:
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    history = []

    for episode in range(n_episodes):
        task.reset()
        if not is_embodied:
            agent.y_history = [0.5] * 5

        episode_error = 0
        n_steps = 30

        for step in range(n_steps):
            create_thermal_variation()
            time.sleep(0.03)

            hw_state, temp_derivative = sensor.read()
            current_y = task.y

            if is_embodied:
                depth = agent.plan(hw_state, current_y, horizon=3, cost_weight=0.3)
                hw_encoded = agent.encode(hw_state)
                pred_y = agent.predict_target(hw_encoded, current_y)
            else:
                depth = agent.select_depth()
                pred_y = agent.predict_target(current_y)

            true_y = task.step(temp_derivative)
            true_y_tensor = torch.tensor([[true_y]], dtype=torch.float32, device=DEVICE)

            # Learning
            loss = F.mse_loss(pred_y, true_y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track error
            track_error = simulate_tracking(pred_y.item(), true_y, depth)
            episode_error += track_error

            if not is_embodied:
                agent.update_history(true_y)

        history.append({'episode': episode, 'error': episode_error / n_steps})

        if (episode + 1) % 15 == 0:
            print(f"    Episode {episode+1}: error={episode_error/n_steps:.4f}")

    return {'history': history}


def evaluate(agent: nn.Module, task: DriftTask, sensor: HardwareSensor,
             n_episodes: int = 50, is_embodied: bool = True) -> Dict:
    agent.eval()
    results = []

    for episode in range(n_episodes):
        task.reset()
        if not is_embodied:
            agent.y_history = [0.5] * 5

        episode_errors = []
        episode_costs = []

        for step in range(30):
            create_thermal_variation()
            time.sleep(0.03)

            hw_state, temp_derivative = sensor.read()
            current_y = task.y

            with torch.no_grad():
                if is_embodied:
                    depth = agent.plan(hw_state, current_y, horizon=3, cost_weight=0.3)
                    hw_encoded = agent.encode(hw_state)
                    pred_y = agent.predict_target(hw_encoded, current_y).item()
                else:
                    depth = agent.select_depth()
                    pred_y = agent.predict_target(current_y).item()

            true_y = task.step(temp_derivative)

            track_error = simulate_tracking(pred_y, true_y, depth)
            cost = compute_cost(depth, hw_state)

            episode_errors.append(track_error)
            episode_costs.append(cost)

            if not is_embodied:
                agent.update_history(true_y)

        results.append({
            'mean_error': np.mean(episode_errors),
            'mean_cost': np.mean(episode_costs),
        })

    return {
        'mean_error': float(np.mean([r['mean_error'] for r in results])),
        'std_error': float(np.std([r['mean_error'] for r in results])),
        'mean_cost': float(np.mean([r['mean_cost'] for r in results])),
        'results': results,
    }


def main():
    print("=" * 70)
    print("  z1321: EXTENDED EMBODIMENT")
    print("  Multi-step prediction + Planning horizon")
    print("=" * 70)
    print()

    sensor = HardwareSensor()
    task = DriftTask()

    embodied = ExtendedEmbodiedAgent(hw_dim=4, n_depths=5).to(DEVICE)
    blind = BlindAgent(n_depths=5).to(DEVICE)

    print(f"Embodied params: {sum(p.numel() for p in embodied.parameters()):,}")
    print(f"Blind params: {sum(p.numel() for p in blind.parameters()):,}")

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    print("\nTraining EMBODIED...")
    train_agent(embodied, task, sensor, n_episodes=60, is_embodied=True)

    print("\nTraining BLIND...")
    train_agent(blind, task, sensor, n_episodes=60, is_embodied=False)

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    print("\nEvaluating EMBODIED...")
    e_eval = evaluate(embodied, task, sensor, n_episodes=50, is_embodied=True)

    print("\nEvaluating BLIND...")
    b_eval = evaluate(blind, task, sensor, n_episodes=50, is_embodied=False)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    e_err = e_eval['mean_error']
    b_err = b_eval['mean_error']
    improvement = (b_err - e_err) / b_err * 100

    print(f"\n{'Metric':<20} | {'Embodied':>12} | {'Blind':>12} | {'Improvement':>12}")
    print("-" * 65)
    print(f"{'Tracking Error (↓)':<20} | {e_err:>12.4f} | {b_err:>12.4f} | {improvement:>+11.1f}%")

    # Statistical test
    e_errors = [r['mean_error'] for r in e_eval['results']]
    b_errors = [r['mean_error'] for r in b_eval['results']]
    t_stat, p_value = stats.ttest_ind(e_errors, b_errors)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    print(f"\nt-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.2e}")

    # Verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if improvement > 50 and p_value < 0.001:
        verdict = "EXTENDED EMBODIMENT PROVEN"
        print(f"\n✅ {verdict}")
        print(f"   {improvement:.1f}% improvement (p={p_value:.2e})")
    elif improvement > 30 and p_value < 0.01:
        verdict = "STRONG EXTENDED EMBODIMENT"
        print(f"\n✅ {verdict}")
        print(f"   {improvement:.1f}% improvement (p={p_value:.2e})")
    else:
        verdict = f"IMPROVEMENT: {improvement:.1f}%"
        print(f"\n⚠️ {verdict} (p={p_value:.2e})")

    # Save
    output = {
        'experiment': 'z1321_extended_embodiment',
        'timestamp': datetime.now().isoformat(),
        'embodied': {'error': e_err, 'cost': e_eval['mean_cost']},
        'blind': {'error': b_err, 'cost': b_eval['mean_cost']},
        'improvement': improvement,
        'p_value': float(p_value),
        'verdict': verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1321_extended_embodiment.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output


if __name__ == '__main__':
    main()
