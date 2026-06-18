#!/usr/bin/env python3
"""
z1318: Active Inference Embodiment

Based on Free Energy Principle research:
- https://arxiv.org/abs/2306.06792 (Neural Network Implementation for FEP)
- https://www.nature.com/articles/s42003-021-02994-2 (Canonical neural networks perform active inference)

KEY INSIGHT: Active inference = predict + act to minimize surprise

Previous experiments failed because:
- z1313-z1314: Task was independent of hardware state
- z1315-z1316: PROVED that when task DEPENDS on hardware, embodiment wins (85%)

THIS EXPERIMENT: Active inference for computational homeostasis
1. Model PREDICTS its own efficiency (J/token proxy)
2. Model SELECTS compute depth to maintain target efficiency
3. Under varying load, embodied model adapts, blind cannot

The prediction error (surprise) IS the homeostatic deviation.
Actions minimize prediction error by adapting computation.

This implements the "closed-loop homeostatic control" from our architecture docs.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class BodyState:
    """Unified body state from hardware telemetry"""
    temperature: float  # Normalized 0-1
    power: float        # Normalized 0-1
    utilization: float  # Normalized 0-1
    temp_derivative: float  # Rate of change
    compute_cost: float  # Last measured compute cost


class HardwareSensor:
    """Real hardware telemetry"""
    def __init__(self):
        self.card = '/sys/class/drm/card1/device'
        self.last_temp = None
        self.last_time = None
        self.last_cost = 0.5

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

    def read(self) -> BodyState:
        """Get current body state"""
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

        return BodyState(
            temperature=temp_c / 100,
            power=power_w / 100,
            utilization=util / 100,
            temp_derivative=np.clip(temp_derivative, -5, 5) / 5,
            compute_cost=self.last_cost,
        )

    def update_cost(self, cost: float):
        """Update measured compute cost"""
        self.last_cost = cost


class ComputeEnvironment:
    """
    Environment that models REAL computational efficiency.

    Efficiency depends on:
    1. Compute depth (more layers = more cost)
    2. Hardware state (high temp = lower efficiency due to throttling)
    3. Background load (contention reduces efficiency)

    This creates a CAUSAL link: hardware → efficiency → optimal depth
    """

    def __init__(self, target_efficiency: float = 0.7):
        self.target_efficiency = target_efficiency
        self.base_cost_per_layer = 0.1
        self.throttling_factor = 0.3  # How much temp affects cost

    def compute(self, depth: int, body_state: BodyState,
                background_load: float = 0.0) -> Tuple[float, float]:
        """
        Perform computation and return (output_quality, actual_cost).

        Quality increases with depth (diminishing returns).
        Cost depends on depth AND hardware state.
        """
        # Quality: diminishing returns with depth
        # quality = 1 - exp(-depth/4) roughly
        quality = 1.0 - np.exp(-depth / 4)

        # Base cost: proportional to depth
        base_cost = depth * self.base_cost_per_layer

        # Thermal penalty: higher temp = higher cost (throttling overhead)
        thermal_penalty = 1.0 + self.throttling_factor * body_state.temperature

        # Load penalty: background load increases cost
        load_penalty = 1.0 + background_load

        # Actual cost
        actual_cost = base_cost * thermal_penalty * load_penalty

        # Add noise
        actual_cost *= (1 + 0.05 * np.random.randn())

        return float(quality), float(actual_cost)

    def efficiency(self, quality: float, cost: float) -> float:
        """Compute efficiency = quality / cost"""
        return quality / max(cost, 0.01)


class ActiveInferenceAgent(nn.Module):
    """
    Agent that uses active inference for homeostatic control.

    1. PERCEIVES hardware state
    2. PREDICTS achievable efficiency at each depth
    3. SELECTS depth to minimize |predicted_efficiency - target|
    4. LEARNS from prediction error (surprise minimization)
    """

    def __init__(self, body_dim: int = 5, n_depths: int = 6, hidden_dim: int = 64):
        super().__init__()

        self.n_depths = n_depths

        # Body state encoder
        self.body_encoder = nn.Sequential(
            nn.Linear(body_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Efficiency predictor: predicts efficiency for each depth given body state
        self.efficiency_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),  # body + depth
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # predicted efficiency
        )

    def encode_body(self, body_state: BodyState) -> torch.Tensor:
        """Encode body state to latent"""
        x = torch.tensor([
            body_state.temperature,
            body_state.power,
            body_state.utilization,
            body_state.temp_derivative,
            body_state.compute_cost,
        ], dtype=torch.float32, device=DEVICE)
        return self.body_encoder(x.unsqueeze(0))

    def predict_efficiency(self, body_encoded: torch.Tensor, depth: int) -> torch.Tensor:
        """Predict efficiency for a specific depth"""
        depth_tensor = torch.tensor([[depth / self.n_depths]], dtype=torch.float32, device=DEVICE)
        x = torch.cat([body_encoded, depth_tensor], dim=-1)
        return self.efficiency_predictor(x)

    def select_depth(self, body_state: BodyState, target_efficiency: float) -> int:
        """Select depth that minimizes |predicted_efficiency - target|"""
        body_encoded = self.encode_body(body_state)

        best_depth = 1
        best_error = float('inf')

        for depth in range(1, self.n_depths + 1):
            pred_eff = self.predict_efficiency(body_encoded, depth).item()
            error = abs(pred_eff - target_efficiency)
            if error < best_error:
                best_error = error
                best_depth = depth

        return best_depth

    def forward(self, body_state: BodyState, depth: int) -> torch.Tensor:
        """Forward pass: predict efficiency for given body state and depth"""
        body_encoded = self.encode_body(body_state)
        return self.predict_efficiency(body_encoded, depth)


class BlindAgent(nn.Module):
    """
    Blind agent without body state awareness.

    Uses fixed policy or learns from history only (no current body state).
    """

    def __init__(self, n_depths: int = 6, hidden_dim: int = 64):
        super().__init__()

        self.n_depths = n_depths

        # History encoder (learns from past efficiency only)
        self.history_encoder = nn.Sequential(
            nn.Linear(5, 32),  # last 5 efficiency values
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Efficiency predictor (same architecture for fair comparison)
        self.efficiency_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.history = [0.5] * 5  # Initial history

    def update_history(self, efficiency: float):
        """Update efficiency history"""
        self.history.pop(0)
        self.history.append(efficiency)

    def predict_efficiency(self, depth: int) -> torch.Tensor:
        """Predict efficiency without body state"""
        history_tensor = torch.tensor([self.history], dtype=torch.float32, device=DEVICE)
        history_encoded = self.history_encoder(history_tensor)
        depth_tensor = torch.tensor([[depth / self.n_depths]], dtype=torch.float32, device=DEVICE)
        x = torch.cat([history_encoded, depth_tensor], dim=-1)
        return self.efficiency_predictor(x)

    def select_depth(self, target_efficiency: float) -> int:
        """Select depth based on history only"""
        best_depth = 1
        best_error = float('inf')

        for depth in range(1, self.n_depths + 1):
            pred_eff = self.predict_efficiency(depth).item()
            error = abs(pred_eff - target_efficiency)
            if error < best_error:
                best_error = error
                best_depth = depth

        return best_depth


def create_load_variation() -> float:
    """Create varying background load to simulate real conditions"""
    load_type = np.random.choice(['none', 'light', 'medium', 'heavy'], p=[0.3, 0.3, 0.25, 0.15])

    if load_type == 'none':
        load = 0.0
    elif load_type == 'light':
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
        load = 0.2
    elif load_type == 'medium':
        _ = torch.randn(1000, 1000, device=DEVICE) @ torch.randn(1000, 1000, device=DEVICE)
        load = 0.5
    else:
        _ = torch.randn(2000, 2000, device=DEVICE) @ torch.randn(2000, 2000, device=DEVICE)
        load = 0.8

    return load


def train_agent(agent: nn.Module, env: ComputeEnvironment, sensor: HardwareSensor,
                n_episodes: int = 100, is_embodied: bool = True) -> Dict:
    """Train agent using active inference (surprise minimization)"""

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    history = []

    for episode in range(n_episodes):
        # Create varying load
        load = create_load_variation()
        time.sleep(0.03)  # Let hardware state change

        # Get body state
        body_state = sensor.read()

        # Select depth based on agent type
        if is_embodied:
            depth = agent.select_depth(body_state, env.target_efficiency)
            pred_eff = agent(body_state, depth)
        else:
            depth = agent.select_depth(env.target_efficiency)
            pred_eff = agent.predict_efficiency(depth)
            agent.update_history(env.target_efficiency)  # Will be updated with actual

        # Execute computation
        quality, cost = env.compute(depth, body_state, load)
        actual_eff = env.efficiency(quality, cost)

        # Update sensor with actual cost
        sensor.update_cost(cost)

        # Compute prediction error (surprise)
        actual_eff_tensor = torch.tensor([[actual_eff]], dtype=torch.float32, device=DEVICE)
        prediction_error = F.mse_loss(pred_eff, actual_eff_tensor)

        # Minimize surprise (free energy)
        optimizer.zero_grad()
        prediction_error.backward()
        optimizer.step()

        # Update blind agent history
        if not is_embodied:
            agent.update_history(actual_eff)

        history.append({
            'episode': episode,
            'load': load,
            'depth': depth,
            'quality': quality,
            'cost': cost,
            'efficiency': actual_eff,
            'prediction_error': prediction_error.item(),
        })

        if (episode + 1) % 25 == 0:
            recent_eff = np.mean([h['efficiency'] for h in history[-25:]])
            recent_err = np.mean([h['prediction_error'] for h in history[-25:]])
            print(f"    Ep {episode+1}: eff={recent_eff:.3f}, pred_err={recent_err:.4f}")

    return {'history': history}


def evaluate_homeostasis(agent: nn.Module, env: ComputeEnvironment, sensor: HardwareSensor,
                         n_episodes: int = 100, is_embodied: bool = True) -> Dict:
    """Evaluate homeostatic performance (maintaining target efficiency)"""

    agent.eval()
    results = []

    for episode in range(n_episodes):
        load = create_load_variation()
        time.sleep(0.03)

        body_state = sensor.read()

        if is_embodied:
            depth = agent.select_depth(body_state, env.target_efficiency)
        else:
            depth = agent.select_depth(env.target_efficiency)
            agent.update_history(env.target_efficiency)

        quality, cost = env.compute(depth, body_state, load)
        actual_eff = env.efficiency(quality, cost)

        # Homeostatic error = deviation from target
        homeostatic_error = abs(actual_eff - env.target_efficiency)

        if not is_embodied:
            agent.update_history(actual_eff)

        results.append({
            'load': load,
            'depth': depth,
            'efficiency': actual_eff,
            'homeostatic_error': homeostatic_error,
            'quality': quality,
        })

    return {
        'mean_efficiency': float(np.mean([r['efficiency'] for r in results])),
        'std_efficiency': float(np.std([r['efficiency'] for r in results])),
        'mean_homeostatic_error': float(np.mean([r['homeostatic_error'] for r in results])),
        'mean_quality': float(np.mean([r['quality'] for r in results])),
        'mean_depth': float(np.mean([r['depth'] for r in results])),
        'results': results,
    }


def main():
    print("=" * 70)
    print("  z1318: ACTIVE INFERENCE EMBODIMENT")
    print("  Predict + Act to minimize surprise (Free Energy Principle)")
    print("=" * 70)
    print()

    sensor = HardwareSensor()
    env = ComputeEnvironment(target_efficiency=0.7)

    # Create agents
    embodied_agent = ActiveInferenceAgent(body_dim=5, n_depths=6).to(DEVICE)
    blind_agent = BlindAgent(n_depths=6).to(DEVICE)

    print(f"Embodied params: {sum(p.numel() for p in embodied_agent.parameters()):,}")
    print(f"Blind params: {sum(p.numel() for p in blind_agent.parameters()):,}")
    print(f"Target efficiency: {env.target_efficiency}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING (Active Inference - Minimize Prediction Error)")
    print("=" * 70)

    print("\nTraining EMBODIED agent...")
    embodied_train = train_agent(embodied_agent, env, sensor, n_episodes=150, is_embodied=True)

    print("\nTraining BLIND agent...")
    blind_train = train_agent(blind_agent, env, sensor, n_episodes=150, is_embodied=False)

    # Evaluation
    print("\n" + "=" * 70)
    print("EVALUATION (Homeostatic Performance)")
    print("=" * 70)

    print("\nEvaluating EMBODIED agent...")
    embodied_eval = evaluate_homeostasis(embodied_agent, env, sensor, n_episodes=100, is_embodied=True)

    print("\nEvaluating BLIND agent...")
    blind_eval = evaluate_homeostasis(blind_agent, env, sensor, n_episodes=100, is_embodied=False)

    # Compare
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} | {'Embodied':>12} | {'Blind':>12} | {'Advantage':>12}")
    print("-" * 70)

    # Homeostatic error (lower is better)
    e_error = embodied_eval['mean_homeostatic_error']
    b_error = blind_eval['mean_homeostatic_error']
    error_improvement = (b_error - e_error) / b_error * 100 if b_error > 0 else 0
    print(f"{'Homeostatic Error':<25} | {e_error:>12.4f} | {b_error:>12.4f} | {error_improvement:>+11.1f}%")

    # Efficiency (closer to target is better)
    e_eff = embodied_eval['mean_efficiency']
    b_eff = blind_eval['mean_efficiency']
    print(f"{'Mean Efficiency':<25} | {e_eff:>12.4f} | {b_eff:>12.4f} | {'':>12}")

    # Quality
    e_qual = embodied_eval['mean_quality']
    b_qual = blind_eval['mean_quality']
    print(f"{'Mean Quality':<25} | {e_qual:>12.4f} | {b_qual:>12.4f} | {'':>12}")

    # Depth adaptation
    e_depth = embodied_eval['mean_depth']
    b_depth = blind_eval['mean_depth']
    print(f"{'Mean Depth':<25} | {e_depth:>12.2f} | {b_depth:>12.2f} | {'':>12}")

    # Statistical test
    e_errors = [r['homeostatic_error'] for r in embodied_eval['results']]
    b_errors = [r['homeostatic_error'] for r in blind_eval['results']]
    t_stat, p_value = stats.ttest_ind(e_errors, b_errors)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    print(f"\nHomeostatic Error: Embodied vs Blind")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.6f}")

    # Verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if error_improvement > 20 and p_value < 0.05:
        verdict = "ACTIVE INFERENCE EMBODIMENT SUCCESSFUL"
        print(f"\n✅ {verdict}")
        print(f"   Embodied maintains homeostasis {error_improvement:.1f}% better (p={p_value:.4f})")
    elif error_improvement > 10 and p_value < 0.1:
        verdict = "PARTIAL ACTIVE INFERENCE ADVANTAGE"
        print(f"\n⚠️ {verdict}")
        print(f"   {error_improvement:.1f}% improvement, p={p_value:.4f}")
    else:
        verdict = "NO CLEAR ACTIVE INFERENCE ADVANTAGE"
        print(f"\n❌ {verdict}")
        print(f"   {error_improvement:.1f}% improvement, p={p_value:.4f}")

    # Save results
    output = {
        'experiment': 'z1318_active_inference_embodiment',
        'timestamp': datetime.now().isoformat(),
        'target_efficiency': env.target_efficiency,
        'embodied_eval': {
            'mean_efficiency': embodied_eval['mean_efficiency'],
            'mean_homeostatic_error': embodied_eval['mean_homeostatic_error'],
            'mean_quality': embodied_eval['mean_quality'],
            'mean_depth': embodied_eval['mean_depth'],
        },
        'blind_eval': {
            'mean_efficiency': blind_eval['mean_efficiency'],
            'mean_homeostatic_error': blind_eval['mean_homeostatic_error'],
            'mean_quality': blind_eval['mean_quality'],
            'mean_depth': blind_eval['mean_depth'],
        },
        'error_improvement_pct': error_improvement,
        'p_value': p_value,
        'verdict': verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1318_active_inference_embodiment.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Active Inference = Predict + Act")
    print("=" * 70)
    print("""
    The Free Energy Principle states that organisms minimize SURPRISE.

    For embodied AI, this means:
    1. PREDICT future efficiency given current body state
    2. SELECT actions (depth) to achieve target efficiency
    3. LEARN from prediction errors to improve model

    When hardware state CAUSALLY affects efficiency:
    - Embodied agent can PREDICT the effect and ADAPT
    - Blind agent must guess based on history alone

    This is GENUINE embodiment: body → prediction → action → outcome

    References:
    - https://arxiv.org/abs/2306.06792
    - https://www.nature.com/articles/s42003-021-02994-2
    """)

    return output


if __name__ == '__main__':
    main()
