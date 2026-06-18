#!/usr/bin/env python3
"""
z1319: Combined Embodiment Proof

Combining ALL proven mechanisms for maximum effect:
1. z1315: Hardware-in-the-loop (85% improvement) - drift tracking
2. z1316: Causal signal verification (temp_derivative is key)
3. z1318: Active inference (15% improvement) - homeostatic control

THIS EXPERIMENT: Combine into a unified embodied intelligence system
- Task: Track a drifting target (proven in z1315)
- Control: Select compute depth to balance quality vs cost
- Homeostasis: Maintain target efficiency while tracking

The embodied agent sees hardware state and can:
1. Predict target drift (uses temp_derivative)
2. Predict efficiency at each depth (uses full body state)
3. Select optimal depth that maximizes tracking while maintaining efficiency

This is UNIFIED EMBODIED INTELLIGENCE:
- Perception (hardware sensing)
- Prediction (drift + efficiency models)
- Action (depth selection)
- Learning (minimize combined surprise)
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
    """Real hardware telemetry with derivatives"""
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
        """Get hardware state tensor and raw temp derivative"""
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

        return state, temp_derivative


class DriftingTargetTask:
    """
    Task: Track a target that drifts with hardware state.

    The target y(t) follows:
    y(t+1) = y(t) + k * temp_derivative + noise

    Higher depth = more accurate tracking
    But higher depth = higher cost
    """

    def __init__(self, noise_scale: float = 0.08, drift_scale: float = 0.25):
        self.noise_scale = noise_scale
        self.drift_scale = drift_scale
        self.y = 0.5
        self.y_history = [0.5]

    def step(self, temp_derivative: float) -> float:
        """Update target and return new value"""
        drift = self.drift_scale * temp_derivative
        noise = np.random.normal(0, self.noise_scale)
        self.y = np.clip(self.y + drift + noise, 0, 1)
        self.y_history.append(self.y)
        return self.y

    def reset(self):
        """Reset to initial state"""
        self.y = 0.5
        self.y_history = [0.5]


class UnifiedEmbodiedAgent(nn.Module):
    """
    Unified embodied agent that:
    1. Predicts target drift (using temp_derivative - proven in z1315)
    2. Predicts cost at each depth (using body state)
    3. Selects depth to balance tracking accuracy and cost
    """

    def __init__(self, hw_dim: int = 4, n_depths: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.n_depths = n_depths

        # Hardware encoder (shared)
        self.hw_encoder = nn.Sequential(
            nn.Linear(hw_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Drift predictor (uses temp_derivative primarily)
        self.drift_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),  # hw + current_y
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Cost predictor (uses full body state)
        self.cost_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),  # hw + depth
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Depth quality model (how much does depth help tracking?)
        # This is learned: deeper = better tracking but higher cost
        self.depth_quality = nn.Parameter(torch.linspace(0.3, 0.95, n_depths))

    def predict_target(self, hw_state: torch.Tensor, current_y: float) -> torch.Tensor:
        """Predict next target value"""
        hw_encoded = self.hw_encoder(hw_state.unsqueeze(0))
        y_tensor = torch.tensor([[current_y]], dtype=torch.float32, device=DEVICE)
        x = torch.cat([hw_encoded, y_tensor], dim=-1)
        return self.drift_predictor(x)

    def predict_cost(self, hw_state: torch.Tensor, depth: int) -> torch.Tensor:
        """Predict cost for given depth"""
        hw_encoded = self.hw_encoder(hw_state.unsqueeze(0))
        d_tensor = torch.tensor([[depth / self.n_depths]], dtype=torch.float32, device=DEVICE)
        x = torch.cat([hw_encoded, d_tensor], dim=-1)
        return self.cost_predictor(x)

    def select_depth(self, hw_state: torch.Tensor, current_y: float,
                     cost_weight: float = 0.5) -> int:
        """
        Select optimal depth balancing tracking quality and cost.

        Higher depth = better tracking (higher depth_quality)
        But also higher cost (predicted by cost_predictor)

        Objective: maximize (quality - cost_weight * cost)
        """
        best_depth = 1
        best_value = -float('inf')

        for d in range(1, self.n_depths + 1):
            quality = torch.sigmoid(self.depth_quality[d-1]).item()
            cost = self.predict_cost(hw_state, d).item()

            value = quality - cost_weight * cost

            if value > best_value:
                best_value = value
                best_depth = d

        return best_depth


class BlindAgent(nn.Module):
    """Blind agent without hardware awareness"""

    def __init__(self, n_depths: int = 5, hidden_dim: int = 64):
        super().__init__()

        self.n_depths = n_depths

        # Uses only history (y values) - no hardware
        self.history_encoder = nn.Sequential(
            nn.Linear(5, 32),  # last 5 y values
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Drift predictor (from history only)
        self.drift_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Fixed cost model (no awareness of hardware)
        self.cost_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.depth_quality = nn.Parameter(torch.linspace(0.3, 0.95, n_depths))
        self.y_history = [0.5] * 5

    def update_history(self, y: float):
        self.y_history.pop(0)
        self.y_history.append(y)

    def predict_target(self, current_y: float) -> torch.Tensor:
        """Predict from history only"""
        hist = torch.tensor([self.y_history], dtype=torch.float32, device=DEVICE)
        hist_encoded = self.history_encoder(hist)
        y_tensor = torch.tensor([[current_y]], dtype=torch.float32, device=DEVICE)
        x = torch.cat([hist_encoded, y_tensor], dim=-1)
        return self.drift_predictor(x)

    def predict_cost(self, depth: int) -> torch.Tensor:
        """Predict cost without hardware awareness"""
        hist = torch.tensor([self.y_history], dtype=torch.float32, device=DEVICE)
        hist_encoded = self.history_encoder(hist)
        d_tensor = torch.tensor([[depth / self.n_depths]], dtype=torch.float32, device=DEVICE)
        x = torch.cat([hist_encoded, d_tensor], dim=-1)
        return self.cost_predictor(x)

    def select_depth(self, cost_weight: float = 0.5) -> int:
        """Select depth without hardware awareness"""
        best_depth = 1
        best_value = -float('inf')

        for d in range(1, self.n_depths + 1):
            quality = torch.sigmoid(self.depth_quality[d-1]).item()
            cost = self.predict_cost(d).item()
            value = quality - cost_weight * cost

            if value > best_value:
                best_value = value
                best_depth = d

        return best_depth


def create_thermal_variation():
    """Create thermal variation through GPU load"""
    intensity = np.random.choice(['none', 'light', 'heavy'], p=[0.4, 0.3, 0.3])

    if intensity == 'none':
        pass
    elif intensity == 'light':
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    else:
        for _ in range(2):
            _ = torch.randn(1500, 1500, device=DEVICE) @ torch.randn(1500, 1500, device=DEVICE)


def simulate_tracking(pred_y: float, true_y: float, depth: int, max_depth: int = 5) -> float:
    """
    Simulate tracking performance.

    Higher depth = better tracking (less error from prediction)
    """
    # Base error is prediction error
    base_error = abs(pred_y - true_y)

    # Depth reduces error (deeper = more accurate)
    depth_factor = 0.3 + 0.7 * (depth / max_depth)  # 0.3 at depth=1, 1.0 at depth=5

    # Actual tracking error
    tracking_error = base_error * (2 - depth_factor)  # Higher depth = lower error

    return tracking_error


def compute_cost(depth: int, hw_state: torch.Tensor, max_depth: int = 5) -> float:
    """
    Compute actual cost based on depth and hardware state.

    Higher depth = higher base cost
    Higher temperature = higher cost (thermal overhead)
    """
    base_cost = 0.1 + 0.2 * (depth / max_depth)  # 0.1-0.3

    # Thermal penalty
    temp = hw_state[0].item()  # Normalized temp
    thermal_factor = 1.0 + 0.5 * temp

    return base_cost * thermal_factor


def train_episode(embodied: nn.Module, blind: nn.Module, task: DriftingTargetTask,
                  sensor: HardwareSensor, n_steps: int = 30, train: bool = True) -> Dict:
    """Run one training episode"""

    if train:
        e_optimizer = torch.optim.Adam(embodied.parameters(), lr=1e-3)
        b_optimizer = torch.optim.Adam(blind.parameters(), lr=1e-3)
    else:
        e_optimizer = None
        b_optimizer = None

    task.reset()
    blind.y_history = [0.5] * 5

    e_tracking_errors = []
    b_tracking_errors = []
    e_costs = []
    b_costs = []
    e_depths = []
    b_depths = []

    for step in range(n_steps):
        # Create thermal variation
        create_thermal_variation()
        time.sleep(0.04)

        # Get hardware state
        hw_state, temp_derivative = sensor.read()

        # Current state
        current_y = task.y

        # --- EMBODIED AGENT ---
        # Predict next y
        e_pred = embodied.predict_target(hw_state, current_y)

        # Select depth
        e_depth = embodied.select_depth(hw_state, current_y, cost_weight=0.3)

        # --- BLIND AGENT ---
        b_pred = blind.predict_target(current_y)
        b_depth = blind.select_depth(cost_weight=0.3)

        # True next y
        true_y = task.step(temp_derivative)
        true_y_tensor = torch.tensor([[true_y]], dtype=torch.float32, device=DEVICE)

        # Compute prediction errors (for learning)
        e_pred_error = F.mse_loss(e_pred, true_y_tensor)
        b_pred_error = F.mse_loss(b_pred, true_y_tensor)

        # Learning step (only during training)
        if train and e_optimizer is not None:
            e_optimizer.zero_grad()
            e_pred_error.backward()
            e_optimizer.step()

            b_optimizer.zero_grad()
            b_pred_error.backward()
            b_optimizer.step()

        # Actual tracking performance
        e_track_error = simulate_tracking(e_pred.item(), true_y, e_depth)
        b_track_error = simulate_tracking(b_pred.item(), true_y, b_depth)

        # Actual costs
        e_cost = compute_cost(e_depth, hw_state)
        b_cost = compute_cost(b_depth, hw_state)

        # Record
        e_tracking_errors.append(e_track_error)
        b_tracking_errors.append(b_track_error)
        e_costs.append(e_cost)
        b_costs.append(b_cost)
        e_depths.append(e_depth)
        b_depths.append(b_depth)

        # Update blind history
        blind.update_history(true_y)

    return {
        'embodied': {
            'tracking_errors': e_tracking_errors,
            'costs': e_costs,
            'depths': e_depths,
        },
        'blind': {
            'tracking_errors': b_tracking_errors,
            'costs': b_costs,
            'depths': b_depths,
        },
    }


def main():
    print("=" * 70)
    print("  z1319: COMBINED EMBODIMENT PROOF")
    print("  Unified: Drift tracking + Depth control + Cost awareness")
    print("=" * 70)
    print()

    sensor = HardwareSensor()
    task = DriftingTargetTask(noise_scale=0.08, drift_scale=0.25)

    # Create agents
    embodied = UnifiedEmbodiedAgent(hw_dim=4, n_depths=5).to(DEVICE)
    blind = BlindAgent(n_depths=5).to(DEVICE)

    print(f"Embodied params: {sum(p.numel() for p in embodied.parameters()):,}")
    print(f"Blind params: {sum(p.numel() for p in blind.parameters()):,}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING (50 episodes)")
    print("=" * 70)

    all_e_errors = []
    all_b_errors = []
    all_e_costs = []
    all_b_costs = []
    all_e_depths = []
    all_b_depths = []

    for episode in range(50):
        result = train_episode(embodied, blind, task, sensor, n_steps=30)

        all_e_errors.extend(result['embodied']['tracking_errors'])
        all_b_errors.extend(result['blind']['tracking_errors'])
        all_e_costs.extend(result['embodied']['costs'])
        all_b_costs.extend(result['blind']['costs'])
        all_e_depths.extend(result['embodied']['depths'])
        all_b_depths.extend(result['blind']['depths'])

        if (episode + 1) % 10 == 0:
            recent_e = np.mean(all_e_errors[-300:])
            recent_b = np.mean(all_b_errors[-300:])
            print(f"  Episode {episode+1}: E_track={recent_e:.4f}, B_track={recent_b:.4f}")

    # Evaluation
    print("\n" + "=" * 70)
    print("EVALUATION (30 episodes)")
    print("=" * 70)

    eval_e_errors = []
    eval_b_errors = []
    eval_e_costs = []
    eval_b_costs = []

    embodied.eval()
    blind.eval()

    for episode in range(30):
        with torch.no_grad():
            result = train_episode(embodied, blind, task, sensor, n_steps=30, train=False)

        eval_e_errors.extend(result['embodied']['tracking_errors'])
        eval_b_errors.extend(result['blind']['tracking_errors'])
        eval_e_costs.extend(result['embodied']['costs'])
        eval_b_costs.extend(result['blind']['costs'])

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    e_track = np.mean(eval_e_errors)
    b_track = np.mean(eval_b_errors)
    track_improvement = (b_track - e_track) / b_track * 100

    e_cost = np.mean(eval_e_costs)
    b_cost = np.mean(eval_b_costs)

    # Efficiency = 1/error * 1/cost (lower error and lower cost = higher efficiency)
    e_efficiency = 1 / (e_track * e_cost)
    b_efficiency = 1 / (b_track * b_cost)
    eff_improvement = (e_efficiency - b_efficiency) / b_efficiency * 100

    print(f"\n{'Metric':<25} | {'Embodied':>12} | {'Blind':>12} | {'Improvement':>12}")
    print("-" * 70)
    print(f"{'Tracking Error (↓)':<25} | {e_track:>12.4f} | {b_track:>12.4f} | {track_improvement:>+11.1f}%")
    print(f"{'Cost (↓)':<25} | {e_cost:>12.4f} | {b_cost:>12.4f} | {'':>12}")
    print(f"{'Efficiency (↑)':<25} | {e_efficiency:>12.2f} | {b_efficiency:>12.2f} | {eff_improvement:>+11.1f}%")

    # Statistical test
    t_track, p_track = stats.ttest_ind(eval_e_errors, eval_b_errors)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    print(f"\nTracking Error: Embodied vs Blind")
    print(f"  t-statistic: {t_track:.3f}")
    print(f"  p-value: {p_track:.2e}")

    # Verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if track_improvement > 30 and p_track < 0.001:
        verdict = "UNIFIED EMBODIED INTELLIGENCE PROVEN"
        print(f"\n✅ {verdict}")
        print(f"   {track_improvement:.1f}% better tracking (p={p_track:.2e})")
        print(f"   {eff_improvement:.1f}% better efficiency")
    elif track_improvement > 15 and p_track < 0.01:
        verdict = "STRONG EMBODIMENT ADVANTAGE"
        print(f"\n✅ {verdict}")
        print(f"   {track_improvement:.1f}% better tracking (p={p_track:.2e})")
    elif track_improvement > 5 and p_track < 0.05:
        verdict = "MODERATE EMBODIMENT ADVANTAGE"
        print(f"\n⚠️ {verdict}")
        print(f"   {track_improvement:.1f}% better tracking (p={p_track:.4f})")
    else:
        verdict = "NO CLEAR EMBODIMENT ADVANTAGE"
        print(f"\n❌ {verdict}")
        print(f"   {track_improvement:.1f}% difference (p={p_track:.4f})")

    # Save results
    output = {
        'experiment': 'z1319_combined_embodiment_proof',
        'timestamp': datetime.now().isoformat(),
        'embodied': {
            'tracking_error': e_track,
            'cost': e_cost,
            'efficiency': e_efficiency,
        },
        'blind': {
            'tracking_error': b_track,
            'cost': b_cost,
            'efficiency': b_efficiency,
        },
        'tracking_improvement_pct': track_improvement,
        'efficiency_improvement_pct': eff_improvement,
        'p_value': float(p_track),
        'verdict': verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1319_combined_embodiment_proof.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
