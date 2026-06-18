#!/usr/bin/env python3
"""
z1508: Catastrophic Forgetting Resistance Benchmark

Tests whether embodied systems resist catastrophic forgetting better
than disembodied systems when learning sequential tasks.

Hypothesis: Embodied systems with body-state context maintain task
separation better, reducing interference between sequential tasks.

Test Protocol:
1. Train on Task A (pendulum at angle 0)
2. Train on Task B (pendulum at angle 0.2)
3. Test on Task A (measure forgetting)
4. Compare embodied vs disembodied forgetting rates

References:
- Catastrophic forgetting in neural networks (McCloskey & Cohen, 1989)
- Embodied cognition and memory consolidation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


class MultiTaskPendulum:
    """Pendulum with configurable target angle"""

    def __init__(self, target_angle: float = 0.0, dt: float = 0.02):
        self.target = target_angle
        self.dt = dt
        self.theta = 0.0
        self.theta_dot = 0.0
        self.g = 9.81
        self.l = 0.5

    def reset(self) -> Tuple[float, float]:
        self.theta = np.random.randn() * 0.05 + self.target
        self.theta_dot = np.random.randn() * 0.05
        return self.theta, self.theta_dot

    def step(self, action: float) -> Tuple[float, float, float, bool]:
        torque = action * 2.0

        acc = (self.g / self.l * np.sin(self.theta) +
               torque / (0.1 * self.l**2) -
               0.1 * self.theta_dot)

        self.theta_dot += acc * self.dt
        self.theta += self.theta_dot * self.dt

        # Reward: distance from target
        error = abs(self.theta - self.target)
        reward = 1.0 - error * 2 - 0.1 * abs(self.theta_dot)
        done = error > 0.5

        return self.theta, self.theta_dot, reward, done


class EmbodiedController(nn.Module):
    """Controller with body-state context"""

    def __init__(self, embodied: bool = True):
        super().__init__()
        self.embodied = embodied

        input_dim = 6 if embodied else 2  # state + body or just state

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU()
        )

        self.policy = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor, body: torch.Tensor = None) -> torch.Tensor:
        if self.embodied and body is not None:
            x = torch.cat([state, body], dim=-1)
        else:
            x = state
        z = self.encoder(x)
        return self.policy(z)


class ForgettingBenchmark:
    """Benchmark for catastrophic forgetting"""

    def __init__(self, device: torch.device):
        self.device = device
        self.telemetry = SysfsHwmonTelemetry()

        # Tasks with different target angles
        self.task_a = MultiTaskPendulum(target_angle=0.0)
        self.task_b = MultiTaskPendulum(target_angle=0.2)

        # Controllers
        self.embodied = EmbodiedController(embodied=True).to(device)
        self.disembodied = EmbodiedController(embodied=False).to(device)

        self.opt_emb = torch.optim.Adam(self.embodied.parameters(), lr=1e-3)
        self.opt_dis = torch.optim.Adam(self.disembodied.parameters(), lr=1e-3)

    def get_body_state(self) -> torch.Tensor:
        sample = self.telemetry.read_sample()
        temp = sample.temp_junction_c if sample.temp_junction_c else sample.temp_edge_c
        power = sample.power_w if sample.power_w else 10.0
        util = sample.gpu_busy_pct if sample.gpu_busy_pct else 0.0

        return torch.tensor([
            (temp - 50) / 30,
            (power - 10) / 20,
            util / 100,
            0.0
        ], dtype=torch.float32, device=self.device)

    def evaluate(self, controller: nn.Module, task: MultiTaskPendulum,
                embodied: bool, n_episodes: int = 20) -> float:
        """Evaluate controller on task"""
        lengths = []

        for _ in range(n_episodes):
            theta, theta_dot = task.reset()

            for step in range(200):
                state = torch.tensor([[theta, theta_dot]],
                                    dtype=torch.float32, device=self.device)
                body = self.get_body_state().unsqueeze(0) if embodied else None

                with torch.no_grad():
                    action = controller(state, body)

                theta, theta_dot, _, done = task.step(action.item())
                if done:
                    break

            lengths.append(step + 1)

        return np.mean(lengths)

    def train_epoch(self, controller: nn.Module, optimizer: torch.optim.Optimizer,
                   task: MultiTaskPendulum, embodied: bool, n_episodes: int = 10):
        """Train for one epoch"""
        for _ in range(n_episodes):
            theta, theta_dot = task.reset()
            states, bodies, actions, rewards = [], [], [], []

            for step in range(200):
                state = torch.tensor([[theta, theta_dot]],
                                    dtype=torch.float32, device=self.device)
                body = self.get_body_state().unsqueeze(0) if embodied else None

                action = controller(state, body)
                action_noisy = action.item() + np.random.randn() * 0.1
                action_noisy = np.clip(action_noisy, -1, 1)

                next_theta, next_theta_dot, reward, done = task.step(action_noisy)

                states.append(state)
                if embodied:
                    bodies.append(body)
                actions.append(action)
                rewards.append(reward)

                theta, theta_dot = next_theta, next_theta_dot
                if done:
                    break

            if len(states) < 5:
                continue

            # Policy gradient update
            states = torch.cat(states)
            actions = torch.cat(actions)
            rewards = torch.tensor(rewards, device=self.device)

            if embodied:
                bodies = torch.cat(bodies)
                pred = controller(states, bodies)
            else:
                pred = controller(states, None)

            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            loss = -torch.mean(rewards * pred.squeeze())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def run_forgetting_test(self, n_epochs_per_task: int = 20):
        """Run complete forgetting test"""
        print("=" * 70)
        print("z1508: Catastrophic Forgetting Benchmark")
        print("=" * 70)

        results = {
            'embodied': {'task_a_before': [], 'task_a_after': [], 'task_b': []},
            'disembodied': {'task_a_before': [], 'task_a_after': [], 'task_b': []}
        }

        # Phase 1: Train on Task A
        print("\nPhase 1: Training on Task A (target=0.0)")
        for epoch in range(n_epochs_per_task):
            self.train_epoch(self.embodied, self.opt_emb, self.task_a, embodied=True)
            self.train_epoch(self.disembodied, self.opt_dis, self.task_a, embodied=False)

            if (epoch + 1) % 5 == 0:
                emb_perf = self.evaluate(self.embodied, self.task_a, embodied=True)
                dis_perf = self.evaluate(self.disembodied, self.task_a, embodied=False)
                print(f"  Epoch {epoch+1}: Embodied={emb_perf:.1f}, Disembodied={dis_perf:.1f}")

        # Evaluate Task A performance before Task B
        task_a_emb_before = self.evaluate(self.embodied, self.task_a, embodied=True)
        task_a_dis_before = self.evaluate(self.disembodied, self.task_a, embodied=False)
        results['embodied']['task_a_before'] = task_a_emb_before
        results['disembodied']['task_a_before'] = task_a_dis_before

        print(f"\nTask A performance (before Task B):")
        print(f"  Embodied: {task_a_emb_before:.1f} steps")
        print(f"  Disembodied: {task_a_dis_before:.1f} steps")

        # Phase 2: Train on Task B
        print("\nPhase 2: Training on Task B (target=0.2)")
        for epoch in range(n_epochs_per_task):
            self.train_epoch(self.embodied, self.opt_emb, self.task_b, embodied=True)
            self.train_epoch(self.disembodied, self.opt_dis, self.task_b, embodied=False)

            if (epoch + 1) % 5 == 0:
                emb_perf = self.evaluate(self.embodied, self.task_b, embodied=True)
                dis_perf = self.evaluate(self.disembodied, self.task_b, embodied=False)
                print(f"  Epoch {epoch+1}: Embodied={emb_perf:.1f}, Disembodied={dis_perf:.1f}")

        # Evaluate Task B performance
        task_b_emb = self.evaluate(self.embodied, self.task_b, embodied=True)
        task_b_dis = self.evaluate(self.disembodied, self.task_b, embodied=False)
        results['embodied']['task_b'] = task_b_emb
        results['disembodied']['task_b'] = task_b_dis

        # Phase 3: Test Task A again (measure forgetting)
        print("\nPhase 3: Testing Task A (measure forgetting)")
        task_a_emb_after = self.evaluate(self.embodied, self.task_a, embodied=True)
        task_a_dis_after = self.evaluate(self.disembodied, self.task_a, embodied=False)
        results['embodied']['task_a_after'] = task_a_emb_after
        results['disembodied']['task_a_after'] = task_a_dis_after

        # Calculate forgetting
        emb_forgetting = (task_a_emb_before - task_a_emb_after) / max(task_a_emb_before, 1)
        dis_forgetting = (task_a_dis_before - task_a_dis_after) / max(task_a_dis_before, 1)

        results['embodied']['forgetting_rate'] = emb_forgetting
        results['disembodied']['forgetting_rate'] = dis_forgetting

        print(f"\nTask A performance (after Task B):")
        print(f"  Embodied: {task_a_emb_after:.1f} steps (was {task_a_emb_before:.1f})")
        print(f"  Disembodied: {task_a_dis_after:.1f} steps (was {task_a_dis_before:.1f})")

        print(f"\nForgetting rates:")
        print(f"  Embodied: {emb_forgetting*100:.1f}%")
        print(f"  Disembodied: {dis_forgetting*100:.1f}%")

        # Conclusion
        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)

        if emb_forgetting < dis_forgetting:
            improvement = (dis_forgetting - emb_forgetting) * 100
            print(f"✓ Embodied shows {improvement:.1f}% LESS forgetting")
            results['conclusion'] = f"SUPPORTED - Embodied reduces forgetting by {improvement:.1f}%"
        else:
            print("✗ Embodiment did not reduce forgetting")
            results['conclusion'] = "NOT SUPPORTED - Embodiment did not reduce forgetting"

        return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    benchmark = ForgettingBenchmark(device)
    results = benchmark.run_forgetting_test(n_epochs_per_task=20)

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'z1508_catastrophic_forgetting.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    main()
