#!/usr/bin/env python3
"""
z1320: World Model-Based Embodiment

Based on cutting-edge research:
- RSSM (Recurrent State Space Model) from DreamerV3
- Robotic World Model dual-autoregressive mechanism
- Interoception research on bodily self-awareness

KEY ARCHITECTURE:
1. Latent body state with BOTH stochastic and deterministic components
2. World model predicts future states AND outcomes
3. Actions selected to minimize expected prediction error
4. Self-supervised learning from body-environment interaction

This is the most advanced embodiment architecture, combining:
- z1315's proven drift tracking (temp_derivative)
- z1318's active inference (homeostatic control)
- z1319's unified system (tracking + cost awareness)
- PLUS world model for multi-step planning

References:
- https://medium.com/@lukasbierling/recurrent-state-space-models-pytorch-implementation
- https://arxiv.org/abs/2501.10100
- https://www.sciencedirect.com/science/article/pii/S0149763424003336
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HardwareSensor:
    """Real hardware telemetry"""
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
        """Get hardware state and temp derivative"""
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


class RSSM(nn.Module):
    """
    Recurrent State Space Model for body state.

    Combines:
    - Deterministic state h_t (memory, like GRU hidden)
    - Stochastic state z_t (captures uncertainty)

    This models the body's internal state with both:
    - Predictable dynamics (deterministic)
    - Uncertainty/variability (stochastic)
    """

    def __init__(self, obs_dim: int = 4, action_dim: int = 1,
                 hidden_dim: int = 32, stoch_dim: int = 16):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim

        # Deterministic state transition (GRU-like)
        self.gru = nn.GRUCell(stoch_dim + action_dim, hidden_dim)

        # Prior: p(z_t | h_t) - what we predict z will be
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, stoch_dim * 2),  # mean and log_std
        )

        # Posterior: q(z_t | h_t, o_t) - what z actually is given observation
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, stoch_dim * 2),
        )

        # Observation decoder: p(o_t | h_t, z_t)
        self.obs_decoder = nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim),
        )

    def get_initial_state(self, batch_size: int = 1):
        """Get initial hidden and stochastic states"""
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        z = torch.zeros(batch_size, self.stoch_dim, device=DEVICE)
        return h, z

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute prior distribution p(z|h)"""
        stats = self.prior_net(h)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(log_std) + 0.1
        return mean, std

    def posterior(self, h: torch.Tensor, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution q(z|h,o)"""
        x = torch.cat([h, obs], dim=-1)
        stats = self.posterior_net(x)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(log_std) + 0.1
        return mean, std

    def sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterized sample"""
        eps = torch.randn_like(mean)
        return mean + std * eps

    def step(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor,
             obs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Single step of RSSM.

        If obs is provided: use posterior (training)
        If obs is None: use prior (imagination/planning)
        """
        # Update deterministic state
        gru_input = torch.cat([z, action], dim=-1)
        h_new = self.gru(gru_input, h)

        # Get prior
        prior_mean, prior_std = self.prior(h_new)

        if obs is not None:
            # Training: use posterior
            post_mean, post_std = self.posterior(h_new, obs)
            z_new = self.sample(post_mean, post_std)
            kl_loss = self._kl_divergence(post_mean, post_std, prior_mean, prior_std)
        else:
            # Imagination: use prior
            z_new = self.sample(prior_mean, prior_std)
            kl_loss = torch.tensor(0.0, device=DEVICE)

        # Decode observation
        state = torch.cat([h_new, z_new], dim=-1)
        obs_pred = self.obs_decoder(state)

        info = {
            'prior_mean': prior_mean,
            'prior_std': prior_std,
            'kl_loss': kl_loss,
            'obs_pred': obs_pred,
        }

        return h_new, z_new, info

    def _kl_divergence(self, mean1, std1, mean2, std2):
        """KL divergence between two Gaussians"""
        var1 = std1 ** 2
        var2 = std2 ** 2
        kl = 0.5 * (var1/var2 + (mean2-mean1)**2/var2 - 1 + torch.log(var2/var1))
        return kl.sum(dim=-1).mean()


class WorldModelAgent(nn.Module):
    """
    Agent with world model for embodied planning.

    Uses RSSM to model body dynamics, then plans actions
    to optimize both task performance and body homeostasis.
    """

    def __init__(self, obs_dim: int = 4, n_actions: int = 5,
                 hidden_dim: int = 32, stoch_dim: int = 16):
        super().__init__()

        self.n_actions = n_actions

        # World model (RSSM)
        self.rssm = RSSM(obs_dim, action_dim=1, hidden_dim=hidden_dim, stoch_dim=stoch_dim)

        # Task predictor: predicts task outcome given state and action
        self.task_predictor = nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim + 1, 32),  # state + action
            nn.ReLU(),
            nn.Linear(32, 1),  # predicted task performance
        )

        # Cost predictor: predicts compute cost given state and action
        self.cost_predictor = nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Current RSSM state
        self.h, self.z = None, None

    def reset(self):
        """Reset agent state"""
        self.h, self.z = self.rssm.get_initial_state(1)

    def observe(self, obs: torch.Tensor, action: torch.Tensor) -> Dict:
        """Update world model with new observation"""
        if self.h is None:
            self.reset()

        self.h, self.z, info = self.rssm.step(self.h, self.z, action, obs.unsqueeze(0))
        return info

    def imagine(self, action: torch.Tensor, steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Imagine future states without observation"""
        h, z = self.h.clone(), self.z.clone()

        task_preds = []
        cost_preds = []

        for _ in range(steps):
            h, z, _ = self.rssm.step(h, z, action, obs=None)

            state = torch.cat([h, z], dim=-1)
            state_action = torch.cat([state, action], dim=-1)

            task_pred = self.task_predictor(state_action)
            cost_pred = self.cost_predictor(state_action)

            task_preds.append(task_pred)
            cost_preds.append(cost_pred)

        return torch.stack(task_preds), torch.stack(cost_preds)

    def select_action(self, obs: torch.Tensor, task_weight: float = 0.7) -> int:
        """Select action by imagining outcomes for each action"""
        if self.h is None:
            self.reset()

        best_action = 0
        best_value = -float('inf')

        for a in range(self.n_actions):
            action = torch.tensor([[a / self.n_actions]], dtype=torch.float32, device=DEVICE)

            # Imagine 3 steps ahead
            task_preds, cost_preds = self.imagine(action, steps=3)

            # Value = task performance - cost (weighted)
            task_value = task_preds.mean().item()
            cost_value = cost_preds.mean().item()

            value = task_weight * task_value - (1 - task_weight) * cost_value

            if value > best_value:
                best_value = value
                best_action = a

        return best_action + 1  # Depth is 1-indexed


class BlindWorldModelAgent(nn.Module):
    """Blind agent with world model but no hardware observation"""

    def __init__(self, n_actions: int = 5, hidden_dim: int = 32):
        super().__init__()

        self.n_actions = n_actions

        # Simplified world model (no hardware input)
        self.gru = nn.GRUCell(1 + 1, hidden_dim)  # action + last_task

        self.task_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.cost_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.h = None
        self.last_task = 0.5

    def reset(self):
        self.h = torch.zeros(1, 32, device=DEVICE)
        self.last_task = 0.5

    def observe(self, task_result: float, action: torch.Tensor) -> Dict:
        if self.h is None:
            self.reset()

        x = torch.cat([action, torch.tensor([[self.last_task]], dtype=torch.float32, device=DEVICE)], dim=-1)
        self.h = self.gru(x, self.h)
        self.last_task = float(task_result)

        return {}

    def select_action(self, task_weight: float = 0.7) -> int:
        if self.h is None:
            self.reset()

        best_action = 0
        best_value = -float('inf')

        for a in range(self.n_actions):
            action = torch.tensor([[a / self.n_actions]], dtype=torch.float32, device=DEVICE)

            state_action = torch.cat([self.h, action], dim=-1)
            task_pred = self.task_predictor(state_action).item()
            cost_pred = self.cost_predictor(state_action).item()

            value = task_weight * task_pred - (1 - task_weight) * cost_pred

            if value > best_value:
                best_value = value
                best_action = a

        return best_action + 1


class DriftTask:
    """Drifting target task (proven in z1315)"""

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


def compute_task_performance(pred_y: float, true_y: float, depth: int) -> float:
    """Task performance: tracking accuracy scaled by depth"""
    base_error = abs(pred_y - true_y)
    depth_factor = 0.3 + 0.7 * (depth / 5)  # Depth helps
    return 1.0 - base_error * (2 - depth_factor)


def compute_cost(depth: int, hw_state: torch.Tensor) -> float:
    """Compute cost based on depth and hardware"""
    base_cost = 0.1 + 0.15 * (depth / 5)
    thermal_factor = 1.0 + 0.3 * hw_state[0].item()
    return base_cost * thermal_factor


def create_thermal_variation():
    """Create thermal variation"""
    intensity = np.random.choice(['none', 'light', 'heavy'], p=[0.4, 0.35, 0.25])
    if intensity == 'none':
        pass
    elif intensity == 'light':
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    else:
        _ = torch.randn(1500, 1500, device=DEVICE) @ torch.randn(1500, 1500, device=DEVICE)


def train_world_model_agent(agent: WorldModelAgent, task: DriftTask,
                            sensor: HardwareSensor, n_episodes: int = 60) -> Dict:
    """Train world model agent"""

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    history = []

    for episode in range(n_episodes):
        agent.reset()
        task.reset()

        episode_loss = 0
        episode_perf = 0
        n_steps = 25

        for step in range(n_steps):
            create_thermal_variation()
            time.sleep(0.03)

            hw_state, temp_derivative = sensor.read()

            # Select action
            depth = agent.select_action(hw_state, task_weight=0.7)
            action = torch.tensor([[depth / 5]], dtype=torch.float32, device=DEVICE)

            # Execute task
            true_y = task.step(temp_derivative)

            # World model prediction
            with torch.no_grad():
                task_preds, cost_preds = agent.imagine(action, steps=1)
                pred_y = task_preds[0].item()

            # Actual performance
            perf = compute_task_performance(pred_y, true_y, depth)
            cost = compute_cost(depth, hw_state)

            # Update world model - need fresh forward pass for gradients
            optimizer.zero_grad()

            # Fresh forward pass for gradient computation
            h_temp, z_temp = agent.rssm.get_initial_state(1)
            action_for_grad = action.detach()
            h_new, z_new, info = agent.rssm.step(h_temp, z_temp, action_for_grad, hw_state.unsqueeze(0))

            obs_pred = info.get('obs_pred')
            if obs_pred is not None:
                recon_loss = F.mse_loss(obs_pred, hw_state.unsqueeze(0))
                kl_loss = info.get('kl_loss', torch.tensor(0.0, device=DEVICE))
                loss = recon_loss + 0.1 * kl_loss

                loss.backward()
                optimizer.step()

                episode_loss += loss.item()

            # Update agent state (detached)
            with torch.no_grad():
                agent.observe(hw_state, action.detach())

            episode_perf += perf

        history.append({
            'episode': episode,
            'loss': episode_loss / n_steps,
            'perf': episode_perf / n_steps,
        })

        if (episode + 1) % 15 == 0:
            print(f"    Episode {episode+1}: loss={episode_loss/n_steps:.4f}, perf={episode_perf/n_steps:.3f}")

    return {'history': history}


def train_blind_agent(agent: BlindWorldModelAgent, task: DriftTask,
                      sensor: HardwareSensor, n_episodes: int = 60) -> Dict:
    """Train blind agent"""

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    history = []

    for episode in range(n_episodes):
        agent.reset()
        task.reset()

        episode_perf = 0
        n_steps = 25

        for step in range(n_steps):
            create_thermal_variation()
            time.sleep(0.03)

            hw_state, temp_derivative = sensor.read()

            # Select action (no hardware)
            depth = agent.select_action(task_weight=0.7)
            action = torch.tensor([[depth / 5]], dtype=torch.float32, device=DEVICE)

            # Execute
            true_y = task.step(temp_derivative)

            # Blind prediction (just guess 0.5)
            pred_y = 0.5

            perf = compute_task_performance(pred_y, true_y, depth)

            # Update
            agent.observe(perf, action)

            episode_perf += perf

        history.append({'episode': episode, 'perf': episode_perf / n_steps})

        if (episode + 1) % 15 == 0:
            print(f"    Episode {episode+1}: perf={episode_perf/n_steps:.3f}")

    return {'history': history}


def evaluate(agent, task: DriftTask, sensor: HardwareSensor,
             n_episodes: int = 40, is_embodied: bool = True) -> Dict:
    """Evaluate agent"""

    results = []

    for episode in range(n_episodes):
        if is_embodied:
            agent.reset()
        else:
            agent.reset()
        task.reset()

        episode_perfs = []
        episode_costs = []

        for step in range(25):
            create_thermal_variation()
            time.sleep(0.03)

            hw_state, temp_derivative = sensor.read()

            if is_embodied:
                depth = agent.select_action(hw_state, task_weight=0.7)
                action = torch.tensor([[depth / 5]], dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    task_preds, _ = agent.imagine(action, steps=1)
                    pred_y = task_preds[0].item()

                agent.observe(hw_state, action)
            else:
                depth = agent.select_action(task_weight=0.7)
                pred_y = 0.5
                action = torch.tensor([[depth / 5]], dtype=torch.float32, device=DEVICE)

            true_y = task.step(temp_derivative)

            perf = compute_task_performance(pred_y, true_y, depth)
            cost = compute_cost(depth, hw_state)

            if not is_embodied:
                agent.observe(perf, action)

            episode_perfs.append(perf)
            episode_costs.append(cost)

        results.append({
            'mean_perf': np.mean(episode_perfs),
            'mean_cost': np.mean(episode_costs),
        })

    return {
        'mean_perf': float(np.mean([r['mean_perf'] for r in results])),
        'std_perf': float(np.std([r['mean_perf'] for r in results])),
        'mean_cost': float(np.mean([r['mean_cost'] for r in results])),
        'results': results,
    }


def main():
    print("=" * 70)
    print("  z1320: WORLD MODEL-BASED EMBODIMENT")
    print("  RSSM + Active Inference + Drift Tracking")
    print("=" * 70)
    print()

    sensor = HardwareSensor()
    task = DriftTask()

    # Create agents
    embodied = WorldModelAgent(obs_dim=4, n_actions=5).to(DEVICE)
    blind = BlindWorldModelAgent(n_actions=5).to(DEVICE)

    print(f"Embodied params: {sum(p.numel() for p in embodied.parameters()):,}")
    print(f"Blind params: {sum(p.numel() for p in blind.parameters()):,}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    print("\nTraining EMBODIED (World Model)...")
    e_train = train_world_model_agent(embodied, task, sensor, n_episodes=60)

    print("\nTraining BLIND...")
    b_train = train_blind_agent(blind, task, sensor, n_episodes=60)

    # Evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    print("\nEvaluating EMBODIED...")
    e_eval = evaluate(embodied, task, sensor, n_episodes=40, is_embodied=True)

    print("\nEvaluating BLIND...")
    b_eval = evaluate(blind, task, sensor, n_episodes=40, is_embodied=False)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    e_perf = e_eval['mean_perf']
    b_perf = b_eval['mean_perf']
    perf_improvement = (e_perf - b_perf) / abs(b_perf) * 100 if b_perf != 0 else 0

    print(f"\n{'Metric':<20} | {'Embodied':>12} | {'Blind':>12} | {'Improvement':>12}")
    print("-" * 65)
    print(f"{'Task Performance':<20} | {e_perf:>12.4f} | {b_perf:>12.4f} | {perf_improvement:>+11.1f}%")
    print(f"{'Cost':<20} | {e_eval['mean_cost']:>12.4f} | {b_eval['mean_cost']:>12.4f} | {'':>12}")

    # Statistical test
    e_perfs = [r['mean_perf'] for r in e_eval['results']]
    b_perfs = [r['mean_perf'] for r in b_eval['results']]
    t_stat, p_value = stats.ttest_ind(e_perfs, b_perfs)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    print(f"\nt-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.2e}")

    # Verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if perf_improvement > 30 and p_value < 0.001:
        verdict = "WORLD MODEL EMBODIMENT PROVEN"
        print(f"\n✅ {verdict}")
        print(f"   {perf_improvement:.1f}% improvement (p={p_value:.2e})")
    elif perf_improvement > 15 and p_value < 0.01:
        verdict = "STRONG WORLD MODEL ADVANTAGE"
        print(f"\n✅ {verdict}")
        print(f"   {perf_improvement:.1f}% improvement (p={p_value:.2e})")
    elif perf_improvement > 5 and p_value < 0.05:
        verdict = "MODERATE WORLD MODEL ADVANTAGE"
        print(f"\n⚠️ {verdict}")
    else:
        verdict = "NO CLEAR ADVANTAGE"
        print(f"\n❌ {verdict}")

    # Save
    output = {
        'experiment': 'z1320_world_model_embodiment',
        'timestamp': datetime.now().isoformat(),
        'embodied': {'perf': e_perf, 'cost': e_eval['mean_cost']},
        'blind': {'perf': b_perf, 'cost': b_eval['mean_cost']},
        'improvement': perf_improvement,
        'p_value': float(p_value),
        'verdict': verdict,
        'references': [
            'https://medium.com/@lukasbierling/recurrent-state-space-models-pytorch-implementation',
            'https://arxiv.org/abs/2501.10100',
            'https://www.sciencedirect.com/science/article/pii/S0149763424003336',
        ],
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1320_world_model_embodiment.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
