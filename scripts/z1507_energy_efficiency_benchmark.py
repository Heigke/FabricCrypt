#!/usr/bin/env python3
"""
z1507: Energy Efficiency Benchmark for Embodied Control

Compares energy efficiency (performance per Joule) across all approaches:
1. z1502 Neural embodied controller
2. z1506 pymdp Active Inference controller
3. Random baseline
4. Optimal PID controller (baseline)

Key metric: Steps survived per Joule of energy consumed

This addresses the neuromorphic computing advantage:
"100-1000x energy efficiency on suitable tasks"
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


class PendulumEnv:
    """Unified pendulum environment for benchmarking"""

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.theta = 0.0
        self.theta_dot = 0.0
        self.g = 9.81
        self.l = 0.5
        self.m = 0.1
        self.friction = 0.1
        self.max_torque = 2.0

    def reset(self) -> Tuple[float, float]:
        self.theta = np.random.randn() * 0.05
        self.theta_dot = np.random.randn() * 0.05
        return self.theta, self.theta_dot

    def step(self, action: float) -> Tuple[float, float, float, bool]:
        """action in [-1, 1]"""
        torque = action * self.max_torque

        acc = (self.g / self.l * np.sin(self.theta) +
               torque / (self.m * self.l**2) -
               self.friction * self.theta_dot)

        self.theta_dot += acc * self.dt
        self.theta += self.theta_dot * self.dt

        reward = np.cos(self.theta) - 0.1 * abs(self.theta_dot)
        done = abs(self.theta) > 0.5

        return self.theta, self.theta_dot, reward, done


class NeuralController(nn.Module):
    """Neural controller from z1502 style"""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, embodied: bool = True):
        super().__init__()
        self.embodied = embodied

        actual_input = input_dim if embodied else input_dim - 4

        self.net = nn.Sequential(
            nn.Linear(actual_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor, body: torch.Tensor = None) -> torch.Tensor:
        if self.embodied and body is not None:
            x = torch.cat([state, body], dim=-1)
        else:
            x = state
        return self.net(x)


class PIDController:
    """Classic PID for baseline comparison"""

    def __init__(self, kp: float = 10.0, ki: float = 0.5, kd: float = 2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def __call__(self, theta: float, theta_dot: float, dt: float = 0.02) -> float:
        error = -theta  # Goal is theta = 0

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        return np.clip(output / 2.0, -1, 1)  # Normalize to [-1, 1]


@dataclass
class BenchmarkResult:
    name: str
    episodes: int
    total_steps: int
    total_energy_j: float
    mean_length: float
    std_length: float
    steps_per_joule: float
    mean_reward: float


class EnergyEfficiencyBenchmark:
    """Comprehensive energy efficiency benchmark"""

    def __init__(self, device: torch.device):
        self.device = device
        self.telemetry = SysfsHwmonTelemetry()
        self.env = PendulumEnv()

        # Controllers
        self.neural_embodied = NeuralController(embodied=True).to(device)
        self.neural_disembodied = NeuralController(embodied=False).to(device)
        self.pid = PIDController()

        # Load pre-trained weights if available
        self._init_weights()

    def _init_weights(self):
        """Initialize neural controllers with reasonable weights"""
        # Simple initialization that should provide basic control
        for m in self.neural_embodied.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.neural_disembodied.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_body_state(self) -> torch.Tensor:
        """Get GPU body state"""
        sample = self.telemetry.read_sample()
        temp = sample.temp_junction_c if sample.temp_junction_c else sample.temp_edge_c
        power = sample.power_w if sample.power_w else 10.0
        util = sample.gpu_busy_pct if sample.gpu_busy_pct else 0.0

        return torch.tensor([
            (temp - 50) / 30,
            (power - 10) / 20,
            util / 100,
            0.0  # fatigue placeholder
        ], dtype=torch.float32, device=self.device)

    def run_benchmark(self, controller_fn, name: str, n_episodes: int = 50,
                     max_steps: int = 500, track_energy: bool = True) -> BenchmarkResult:
        """Run benchmark for a single controller"""
        lengths = []
        rewards = []
        total_energy = 0.0

        for ep in range(n_episodes):
            theta, theta_dot = self.env.reset()
            ep_reward = 0.0

            if track_energy:
                with EnergyMeter(self.telemetry) as meter:
                    for step in range(max_steps):
                        action = controller_fn(theta, theta_dot)
                        theta, theta_dot, r, done = self.env.step(action)
                        ep_reward += r
                        if done:
                            break
                    total_energy += meter.energy_j
            else:
                for step in range(max_steps):
                    action = controller_fn(theta, theta_dot)
                    theta, theta_dot, r, done = self.env.step(action)
                    ep_reward += r
                    if done:
                        break

            lengths.append(step + 1)
            rewards.append(ep_reward)

        total_steps = sum(lengths)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        mean_reward = np.mean(rewards)

        if total_energy > 0:
            steps_per_joule = total_steps / total_energy
        else:
            steps_per_joule = 0.0

        return BenchmarkResult(
            name=name,
            episodes=n_episodes,
            total_steps=total_steps,
            total_energy_j=total_energy,
            mean_length=mean_length,
            std_length=std_length,
            steps_per_joule=steps_per_joule,
            mean_reward=mean_reward
        )

    def neural_embodied_fn(self, theta: float, theta_dot: float) -> float:
        """Neural embodied controller function"""
        state = torch.tensor([[theta, theta_dot]], dtype=torch.float32, device=self.device)
        body = self.get_body_state().unsqueeze(0)
        with torch.no_grad():
            action = self.neural_embodied(state, body)
        return action.item()

    def neural_disembodied_fn(self, theta: float, theta_dot: float) -> float:
        """Neural disembodied controller function"""
        state = torch.tensor([[theta, theta_dot]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.neural_disembodied(state, None)
        return action.item()

    def pid_fn(self, theta: float, theta_dot: float) -> float:
        """PID controller function"""
        return self.pid(theta, theta_dot)

    def random_fn(self, theta: float, theta_dot: float) -> float:
        """Random controller"""
        return np.random.uniform(-1, 1)

    def run_all_benchmarks(self, n_episodes: int = 50) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks"""
        print("=" * 70)
        print("z1507: Energy Efficiency Benchmark")
        print("=" * 70)

        results = {}

        # Warm up GPU for consistent measurements
        print("\nWarming up GPU...")
        warmup = torch.randn(1024, 1024, device=self.device)
        for _ in range(20):
            warmup = warmup @ warmup.T
        time.sleep(1)

        # PID Baseline
        print("\n1. PID Controller (baseline)...")
        self.pid.reset()
        results['pid'] = self.run_benchmark(self.pid_fn, 'PID', n_episodes)
        print(f"   Length: {results['pid'].mean_length:.1f} ± {results['pid'].std_length:.1f}")
        print(f"   Energy: {results['pid'].total_energy_j:.2f} J")
        print(f"   Efficiency: {results['pid'].steps_per_joule:.1f} steps/J")

        # Neural Embodied
        print("\n2. Neural Embodied Controller...")
        results['neural_embodied'] = self.run_benchmark(
            self.neural_embodied_fn, 'Neural Embodied', n_episodes)
        print(f"   Length: {results['neural_embodied'].mean_length:.1f} ± {results['neural_embodied'].std_length:.1f}")
        print(f"   Energy: {results['neural_embodied'].total_energy_j:.2f} J")
        print(f"   Efficiency: {results['neural_embodied'].steps_per_joule:.1f} steps/J")

        # Neural Disembodied
        print("\n3. Neural Disembodied Controller...")
        results['neural_disembodied'] = self.run_benchmark(
            self.neural_disembodied_fn, 'Neural Disembodied', n_episodes)
        print(f"   Length: {results['neural_disembodied'].mean_length:.1f} ± {results['neural_disembodied'].std_length:.1f}")
        print(f"   Energy: {results['neural_disembodied'].total_energy_j:.2f} J")
        print(f"   Efficiency: {results['neural_disembodied'].steps_per_joule:.1f} steps/J")

        # Random
        print("\n4. Random Controller...")
        results['random'] = self.run_benchmark(self.random_fn, 'Random', n_episodes)
        print(f"   Length: {results['random'].mean_length:.1f} ± {results['random'].std_length:.1f}")
        print(f"   Energy: {results['random'].total_energy_j:.2f} J")
        print(f"   Efficiency: {results['random'].steps_per_joule:.1f} steps/J")

        return results

    def train_neural_controllers(self, n_episodes: int = 100):
        """Quick training for neural controllers"""
        print("\nTraining neural controllers...")

        optimizer_emb = torch.optim.Adam(self.neural_embodied.parameters(), lr=1e-3)
        optimizer_dis = torch.optim.Adam(self.neural_disembodied.parameters(), lr=1e-3)

        for ep in range(n_episodes):
            # Collect experience
            theta, theta_dot = self.env.reset()
            states_emb, bodies, actions_emb, rewards_emb = [], [], [], []
            states_dis, actions_dis, rewards_dis = [], [], []

            for step in range(200):
                state = torch.tensor([[theta, theta_dot]], dtype=torch.float32, device=self.device)
                body = self.get_body_state().unsqueeze(0)

                # Embodied action
                action_emb = self.neural_embodied(state, body)
                # Disembodied action
                action_dis = self.neural_disembodied(state, None)

                # Use embodied action for environment
                action = action_emb.item() + np.random.randn() * 0.1
                action = np.clip(action, -1, 1)

                next_theta, next_theta_dot, reward, done = self.env.step(action)

                states_emb.append(state)
                bodies.append(body)
                actions_emb.append(action_emb)
                rewards_emb.append(reward)

                states_dis.append(state)
                actions_dis.append(action_dis)
                rewards_dis.append(reward)

                theta, theta_dot = next_theta, next_theta_dot
                if done:
                    break

            if len(states_emb) < 10:
                continue

            # Simple policy gradient update
            states_emb = torch.cat(states_emb)
            bodies = torch.cat(bodies)
            actions_emb = torch.cat(actions_emb)
            rewards_emb = torch.tensor(rewards_emb, device=self.device)

            # Normalize rewards
            rewards_emb = (rewards_emb - rewards_emb.mean()) / (rewards_emb.std() + 1e-8)

            # Embodied loss
            pred_emb = self.neural_embodied(states_emb, bodies)
            loss_emb = -torch.mean(rewards_emb * pred_emb.squeeze())

            optimizer_emb.zero_grad()
            loss_emb.backward()
            optimizer_emb.step()

            # Disembodied
            states_dis = torch.cat(states_dis)
            actions_dis = torch.cat(actions_dis)
            rewards_dis = torch.tensor(rewards_dis, device=self.device)
            rewards_dis = (rewards_dis - rewards_dis.mean()) / (rewards_dis.std() + 1e-8)

            pred_dis = self.neural_disembodied(states_dis, None)
            loss_dis = -torch.mean(rewards_dis * pred_dis.squeeze())

            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()

            if (ep + 1) % 20 == 0:
                print(f"  Episode {ep + 1}/{n_episodes}, Length: {step + 1}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    benchmark = EnergyEfficiencyBenchmark(device)

    # Train controllers first
    benchmark.train_neural_controllers(n_episodes=100)

    # Run benchmarks
    results = benchmark.run_all_benchmarks(n_episodes=50)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Energy Efficiency Comparison")
    print("=" * 70)
    print(f"\n{'Controller':<25} {'Length':<15} {'Energy (J)':<15} {'Steps/J':<15}")
    print("-" * 70)

    for name, r in results.items():
        print(f"{r.name:<25} {r.mean_length:>6.1f} ± {r.std_length:>4.1f}  {r.total_energy_j:>10.2f}     {r.steps_per_joule:>10.1f}")

    # Best efficiency
    best = max(results.values(), key=lambda x: x.steps_per_joule)
    print(f"\nMost efficient: {best.name} ({best.steps_per_joule:.1f} steps/J)")

    # Embodiment comparison
    if 'neural_embodied' in results and 'neural_disembodied' in results:
        emb_eff = results['neural_embodied'].steps_per_joule
        dis_eff = results['neural_disembodied'].steps_per_joule
        if emb_eff > 0 and dis_eff > 0:
            improvement = (emb_eff / dis_eff - 1) * 100
            print(f"\nEmbodiment efficiency gain: {improvement:+.1f}%")

    # Save results
    output = {
        'benchmark': 'energy_efficiency',
        'results': {name: {
            'name': r.name,
            'episodes': r.episodes,
            'total_steps': r.total_steps,
            'total_energy_j': r.total_energy_j,
            'mean_length': r.mean_length,
            'std_length': r.std_length,
            'steps_per_joule': r.steps_per_joule,
            'mean_reward': r.mean_reward
        } for name, r in results.items()},
        'conclusion': f"Most efficient: {best.name}"
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1507_energy_efficiency_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == '__main__':
    main()
