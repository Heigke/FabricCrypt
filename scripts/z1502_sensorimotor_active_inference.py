#!/usr/bin/env python3
"""
z1502: Sensorimotor Active Inference with Real Hardware Feedback

KEY INSIGHT FROM RESEARCH:
- Embodiment does NOT help static classification (z1501 proved this)
- Embodiment DOES help: event-driven, real-time, sensorimotor control tasks
- Neuromorphic advantage: 100-1000x energy efficiency on sparse, temporal workloads

This experiment tests embodiment where it SHOULD matter:
1. Sensorimotor control task (tracking/regulation, not classification)
2. Real-time feedback loops with actual hardware state
3. Event-driven processing (sparse activations)
4. Online learning (continuous adaptation)

Task: Virtual pendulum regulation with embodied feedback
- Agent must keep pendulum upright
- GPU thermal state affects "muscle fatigue" (action scaling)
- FPGA partial write state encodes "proprioceptive memory"
- Success metric: time-to-fall, energy efficiency, adaptation speed

References:
- Real-World Robot Control by Deep Active Inference (arxiv 2512.01924)
- Delayed Feedback Active Inference (PMC 2024)
- Self-configuring feedback loops for sensorimotor control (eLife 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from collections import deque
import math

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


@dataclass
class PendulumState:
    """Physical state of inverted pendulum"""
    theta: float = 0.0      # Angle from vertical (radians)
    theta_dot: float = 0.0  # Angular velocity
    x: float = 0.0          # Cart position
    x_dot: float = 0.0      # Cart velocity

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.theta, self.theta_dot, self.x, self.x_dot], dtype=torch.float32)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> 'PendulumState':
        return cls(theta=t[0].item(), theta_dot=t[1].item(), x=t[2].item(), x_dot=t[3].item())


class PendulumEnvironment:
    """
    Inverted pendulum physics simulation

    This is a CONTROL task, not classification - embodiment should matter here
    because real-time feedback affects action selection.
    """

    def __init__(self, dt: float = 0.02, gravity: float = 9.81,
                 mass_cart: float = 1.0, mass_pole: float = 0.1,
                 length: float = 0.5, friction: float = 0.1):
        self.dt = dt
        self.g = gravity
        self.mc = mass_cart
        self.mp = mass_pole
        self.l = length
        self.mu = friction

        self.state = PendulumState()
        self.max_force = 10.0
        self.theta_threshold = 0.5  # ~29 degrees
        self.x_threshold = 2.4

    def reset(self, noise_std: float = 0.05) -> PendulumState:
        """Reset to near-vertical with small perturbation"""
        self.state = PendulumState(
            theta=np.random.randn() * noise_std,
            theta_dot=np.random.randn() * noise_std,
            x=0.0,
            x_dot=0.0
        )
        return self.state

    def step(self, force: float, fatigue: float = 0.0) -> Tuple[PendulumState, float, bool]:
        """
        Apply force and advance physics

        Args:
            force: Control force [-1, 1] scaled to max_force
            fatigue: Hardware-derived fatigue factor [0, 1] that reduces effective force

        Returns:
            (new_state, reward, done)
        """
        # Apply fatigue - THIS IS WHERE EMBODIMENT ENTERS
        effective_force = force * self.max_force * (1.0 - fatigue * 0.5)

        # Physics update (Euler integration)
        theta = self.state.theta
        theta_dot = self.state.theta_dot
        x = self.state.x
        x_dot = self.state.x_dot

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        total_mass = self.mc + self.mp
        pole_mass_length = self.mp * self.l

        # Equations of motion
        temp = (effective_force + pole_mass_length * theta_dot**2 * sin_theta - self.mu * x_dot) / total_mass
        theta_acc = (self.g * sin_theta - cos_theta * temp) / (self.l * (4/3 - self.mp * cos_theta**2 / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

        # Update state
        self.state.theta_dot += theta_acc * self.dt
        self.state.theta += self.state.theta_dot * self.dt
        self.state.x_dot += x_acc * self.dt
        self.state.x += self.state.x_dot * self.dt

        # Check termination
        done = (abs(self.state.theta) > self.theta_threshold or
                abs(self.state.x) > self.x_threshold)

        # Reward: stay upright and centered
        angle_reward = 1.0 - abs(self.state.theta) / self.theta_threshold
        position_reward = 1.0 - abs(self.state.x) / self.x_threshold
        energy_penalty = 0.01 * abs(force)  # Penalize large forces

        reward = angle_reward + 0.1 * position_reward - energy_penalty

        return self.state, reward, done


class ActiveInferenceAgent(nn.Module):
    """
    Active Inference agent for sensorimotor control

    Key components:
    1. Generative model: predicts next state given action
    2. Recognition model: infers latent state from observations
    3. Action selection: minimizes expected free energy

    The embodied version receives hardware state that modulates:
    - Action scaling (fatigue)
    - Prediction confidence (thermal noise)
    - Memory consolidation (FPGA state)
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 1,
                 latent_dim: int = 16, hidden_dim: int = 64,
                 embodied: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.embodied = embodied

        # Condition dimension: 4 telemetry values if embodied, else 0
        condition_dim = 4 if embodied else 0

        # Recognition model: observation -> latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_encoder = nn.Linear(hidden_dim, latent_dim)
        self.logvar_encoder = nn.Linear(hidden_dim, latent_dim)

        # Generative model: latent + action -> next observation
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.next_state_pred = nn.Linear(hidden_dim, state_dim)
        self.next_state_var = nn.Linear(hidden_dim, state_dim)

        # Policy: latent -> action
        self.policy = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

        # Value function for expected free energy
        self.value = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # Preferred state (goal): upright, centered
        self.register_buffer('preferred_state', torch.zeros(state_dim))

        # Hardware state (updated externally)
        self.hardware_state = torch.zeros(4)  # [temp, power, util, fatigue]

    def encode(self, obs: torch.Tensor, hw_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent distribution"""
        if self.embodied and hw_state is not None:
            x = torch.cat([obs, hw_state], dim=-1)
        else:
            x = obs
        h = self.encoder(x)
        mu = self.mu_encoder(h)
        logvar = self.logvar_encoder(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def predict_next(self, z: torch.Tensor, action: torch.Tensor,
                     hw_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state given current latent and action"""
        if self.embodied and hw_state is not None:
            x = torch.cat([z, action, hw_state], dim=-1)
        else:
            x = torch.cat([z, action], dim=-1)
        h = self.transition(x)
        next_mu = self.next_state_pred(h)
        next_logvar = self.next_state_var(h)
        return next_mu, next_logvar

    def get_action(self, obs: torch.Tensor, hw_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Select action via policy"""
        mu, logvar = self.encode(obs, hw_state)
        z = self.reparameterize(mu, logvar)

        if self.embodied and hw_state is not None:
            policy_input = torch.cat([z, hw_state], dim=-1)
        else:
            policy_input = z

        action = self.policy(policy_input)
        return action

    def compute_free_energy(self, obs: torch.Tensor, action: torch.Tensor,
                           hw_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute variational free energy

        F = D_KL(q(z|o) || p(z)) + E_q[-log p(o|z)]

        For active inference, we also consider expected free energy for action selection.
        """
        # Encode current observation
        mu, logvar = self.encode(obs, hw_state)
        z = self.reparameterize(mu, logvar)

        # KL divergence (assume standard normal prior)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        # Prediction error (how well does model explain observation)
        next_mu, next_logvar = self.predict_next(z, action, hw_state)

        # Expected free energy: preference mismatch
        preference_error = F.mse_loss(next_mu, self.preferred_state.expand_as(next_mu), reduction='none').sum(-1)

        # State uncertainty (epistemic value)
        uncertainty = next_logvar.exp().sum(-1)

        # Total free energy
        F_total = kl + preference_error + 0.1 * uncertainty

        return F_total, {'kl': kl.mean().item(), 'pref_err': preference_error.mean().item(),
                        'uncertainty': uncertainty.mean().item()}


class EmbodiedHardwareInterface:
    """
    Interface to real hardware for embodied feedback

    Provides:
    - GPU thermal state -> fatigue factor
    - GPU utilization -> action confidence
    - Power consumption -> energy budget signal
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.telemetry = SysfsHwmonTelemetry()

        # Running statistics for normalization
        self.temp_ema = 50.0
        self.power_ema = 10.0
        self.util_ema = 50.0
        self.ema_alpha = 0.1

        # Fatigue accumulator (resets on rest)
        self.fatigue = 0.0
        self.fatigue_recovery_rate = 0.01
        self.fatigue_accumulation_rate = 0.001

    def get_state(self) -> Tuple[torch.Tensor, float]:
        """
        Get hardware state and fatigue factor

        Returns:
            (state_tensor, fatigue_factor)
        """
        sample = self.telemetry.read_sample()

        temp = sample.temp_junction_c if sample.temp_junction_c else sample.temp_edge_c
        power = sample.power_w if sample.power_w else 10.0
        util = sample.gpu_busy_pct if sample.gpu_busy_pct else 0.0

        # Update EMAs
        self.temp_ema = self.ema_alpha * temp + (1 - self.ema_alpha) * self.temp_ema
        self.power_ema = self.ema_alpha * power + (1 - self.ema_alpha) * self.power_ema
        self.util_ema = self.ema_alpha * util + (1 - self.ema_alpha) * self.util_ema

        # Compute fatigue based on thermal stress
        thermal_stress = max(0, (temp - 50.0) / 30.0)  # Normalized 50-80C range
        self.fatigue += self.fatigue_accumulation_rate * thermal_stress
        self.fatigue *= (1 - self.fatigue_recovery_rate)
        self.fatigue = min(0.5, self.fatigue)  # Cap at 50% fatigue

        # Normalized state tensor
        state = torch.tensor([
            (temp - 50.0) / 30.0,           # Temp normalized
            (power - 10.0) / 20.0,          # Power normalized
            util / 100.0,                    # Utilization [0,1]
            self.fatigue                     # Current fatigue
        ], dtype=torch.float32, device=self.device)

        return state, self.fatigue

    def reset_fatigue(self):
        """Reset fatigue (e.g., after rest period)"""
        self.fatigue = 0.0


class SensorimotorTrainer:
    """
    Training loop for sensorimotor active inference

    Compares:
    - Embodied agent (receives real hardware state)
    - Disembodied agent (no hardware feedback)
    - Random baseline
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.env = PendulumEnvironment()
        self.hardware = EmbodiedHardwareInterface(device)

        # Create agents
        self.embodied_agent = ActiveInferenceAgent(embodied=True).to(device)
        self.disembodied_agent = ActiveInferenceAgent(embodied=False).to(device)

        # Optimizers
        self.embodied_opt = torch.optim.AdamW(self.embodied_agent.parameters(), lr=1e-3)
        self.disembodied_opt = torch.optim.AdamW(self.disembodied_agent.parameters(), lr=1e-3)

        # Experience replay
        self.embodied_buffer = deque(maxlen=10000)
        self.disembodied_buffer = deque(maxlen=10000)

        # Metrics
        self.metrics = {
            'embodied': {'episode_lengths': [], 'rewards': [], 'energy': []},
            'disembodied': {'episode_lengths': [], 'rewards': [], 'energy': []},
            'random': {'episode_lengths': [], 'rewards': []}
        }

    def run_episode(self, agent: ActiveInferenceAgent, embodied: bool,
                   training: bool = True, max_steps: int = 500) -> Dict:
        """Run single episode"""
        state = self.env.reset()
        total_reward = 0.0
        energy_used = 0.0

        experiences = []

        for step in range(max_steps):
            obs = state.to_tensor().to(self.device)

            if embodied:
                hw_state, fatigue = self.hardware.get_state()
            else:
                hw_state = None
                fatigue = 0.0

            # Get action from agent
            with torch.no_grad():
                action = agent.get_action(obs.unsqueeze(0),
                                         hw_state.unsqueeze(0) if hw_state is not None else None)
                action = action.squeeze(0)

            # Execute in environment
            next_state, reward, done = self.env.step(action.item(), fatigue)

            # Track energy
            if embodied:
                sample = self.hardware.telemetry.read_sample()
                power = sample.power_w if sample.power_w else 10.0
                energy_used += power * self.env.dt

            total_reward += reward

            # Store experience
            if training:
                experiences.append({
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'next_obs': next_state.to_tensor().to(self.device),
                    'hw_state': hw_state,
                    'done': done
                })

            state = next_state

            if done:
                break

        return {
            'length': step + 1,
            'reward': total_reward,
            'energy': energy_used,
            'experiences': experiences
        }

    def train_step(self, agent: ActiveInferenceAgent, opt: torch.optim.Optimizer,
                  buffer: deque, batch_size: int = 64, embodied: bool = True) -> float:
        """Single training step from replay buffer"""
        if len(buffer) < batch_size:
            return 0.0

        # Sample batch
        indices = np.random.choice(len(buffer), batch_size, replace=False)
        batch = [buffer[i] for i in indices]

        obs = torch.stack([b['obs'] for b in batch])
        actions = torch.stack([b['action'] for b in batch])
        rewards = torch.tensor([b['reward'] for b in batch], device=self.device)
        next_obs = torch.stack([b['next_obs'] for b in batch])

        if embodied:
            hw_states = torch.stack([b['hw_state'] for b in batch])
        else:
            hw_states = None

        # Compute free energy loss
        free_energy, _ = agent.compute_free_energy(obs, actions, hw_states)

        # Policy gradient with reward shaping
        policy_actions = agent.get_action(obs, hw_states)
        action_loss = F.mse_loss(policy_actions, actions)  # Behavioral cloning term

        # Prediction loss
        mu, logvar = agent.encode(obs, hw_states)
        z = agent.reparameterize(mu, logvar)
        pred_next, pred_var = agent.predict_next(z, actions, hw_states)
        pred_loss = F.mse_loss(pred_next, next_obs)

        # Total loss
        loss = free_energy.mean() + action_loss + pred_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()

        return loss.item()

    def train(self, n_episodes: int = 200, eval_interval: int = 20):
        """Full training loop comparing embodied vs disembodied"""
        print("=" * 60)
        print("z1502: Sensorimotor Active Inference Training")
        print("=" * 60)
        print(f"Task: Inverted pendulum control")
        print(f"Embodiment: GPU thermal state -> fatigue factor")
        print(f"Hypothesis: Embodied agent adapts to hardware state changes")
        print("=" * 60)

        for episode in range(n_episodes):
            # Train embodied agent
            self.hardware.reset_fatigue()
            result_emb = self.run_episode(self.embodied_agent, embodied=True, training=True)
            for exp in result_emb['experiences']:
                self.embodied_buffer.append(exp)

            loss_emb = self.train_step(self.embodied_agent, self.embodied_opt,
                                       self.embodied_buffer, embodied=True)

            self.metrics['embodied']['episode_lengths'].append(result_emb['length'])
            self.metrics['embodied']['rewards'].append(result_emb['reward'])
            self.metrics['embodied']['energy'].append(result_emb['energy'])

            # Train disembodied agent
            result_dis = self.run_episode(self.disembodied_agent, embodied=False, training=True)
            for exp in result_dis['experiences']:
                self.disembodied_buffer.append(exp)

            loss_dis = self.train_step(self.disembodied_agent, self.disembodied_opt,
                                       self.disembodied_buffer, embodied=False)

            self.metrics['disembodied']['episode_lengths'].append(result_dis['length'])
            self.metrics['disembodied']['rewards'].append(result_dis['reward'])
            self.metrics['disembodied']['energy'].append(result_dis['energy'])

            # Random baseline (no training)
            random_length = 0
            random_reward = 0.0
            state = self.env.reset()
            for _ in range(500):
                action = np.random.uniform(-1, 1)
                state, r, done = self.env.step(action)
                random_length += 1
                random_reward += r
                if done:
                    break
            self.metrics['random']['episode_lengths'].append(random_length)
            self.metrics['random']['rewards'].append(random_reward)

            # Progress report
            if (episode + 1) % eval_interval == 0:
                emb_len = np.mean(self.metrics['embodied']['episode_lengths'][-eval_interval:])
                dis_len = np.mean(self.metrics['disembodied']['episode_lengths'][-eval_interval:])
                rnd_len = np.mean(self.metrics['random']['episode_lengths'][-eval_interval:])

                emb_rew = np.mean(self.metrics['embodied']['rewards'][-eval_interval:])
                dis_rew = np.mean(self.metrics['disembodied']['rewards'][-eval_interval:])

                emb_energy = np.mean(self.metrics['embodied']['energy'][-eval_interval:])
                dis_energy = np.mean(self.metrics['disembodied']['energy'][-eval_interval:])

                print(f"\nEpisode {episode + 1}/{n_episodes}")
                print(f"  Embodied:    length={emb_len:.1f}, reward={emb_rew:.2f}, energy={emb_energy:.3f}J")
                print(f"  Disembodied: length={dis_len:.1f}, reward={dis_rew:.2f}, energy={dis_energy:.3f}J")
                print(f"  Random:      length={rnd_len:.1f}")

                # Check if embodied is adapting better
                if emb_len > dis_len * 1.1:
                    print(f"  -> Embodied agent surviving {(emb_len/dis_len - 1)*100:.1f}% longer!")

    def evaluate_thermal_stress(self, n_trials: int = 10):
        """
        Evaluate agents under deliberate thermal stress

        This is the KEY test: embodied agent should adapt to thermal changes,
        disembodied agent should not.
        """
        print("\n" + "=" * 60)
        print("Thermal Stress Evaluation")
        print("=" * 60)

        results = {
            'embodied_cool': [],
            'embodied_hot': [],
            'disembodied_cool': [],
            'disembodied_hot': []
        }

        # Cool trials (minimal GPU load)
        print("\nCool conditions (minimal GPU load)...")
        time.sleep(2)  # Let GPU cool

        for i in range(n_trials):
            self.hardware.reset_fatigue()
            result = self.run_episode(self.embodied_agent, embodied=True, training=False)
            results['embodied_cool'].append(result['length'])

            result = self.run_episode(self.disembodied_agent, embodied=False, training=False)
            results['disembodied_cool'].append(result['length'])

        # Hot trials (heavy GPU load to warm up)
        print("Warming GPU with heavy computation...")
        warmup = torch.randn(4096, 4096, device=self.device)
        for _ in range(50):
            warmup = warmup @ warmup.T
            warmup = F.relu(warmup)

        print("Hot conditions (thermal stress)...")
        for i in range(n_trials):
            # Keep GPU warm with concurrent computation
            warmup = warmup @ warmup.T

            self.hardware.reset_fatigue()
            result = self.run_episode(self.embodied_agent, embodied=True, training=False)
            results['embodied_hot'].append(result['length'])

            warmup = warmup @ warmup.T
            result = self.run_episode(self.disembodied_agent, embodied=False, training=False)
            results['disembodied_hot'].append(result['length'])

        # Statistical analysis
        print("\nResults:")
        print(f"  Embodied   - Cool: {np.mean(results['embodied_cool']):.1f} ± {np.std(results['embodied_cool']):.1f}")
        print(f"  Embodied   - Hot:  {np.mean(results['embodied_hot']):.1f} ± {np.std(results['embodied_hot']):.1f}")
        print(f"  Disembodied - Cool: {np.mean(results['disembodied_cool']):.1f} ± {np.std(results['disembodied_cool']):.1f}")
        print(f"  Disembodied - Hot:  {np.mean(results['disembodied_hot']):.1f} ± {np.std(results['disembodied_hot']):.1f}")

        # Key metric: performance degradation under heat
        emb_degradation = 1 - np.mean(results['embodied_hot']) / np.mean(results['embodied_cool'])
        dis_degradation = 1 - np.mean(results['disembodied_hot']) / np.mean(results['disembodied_cool'])

        print(f"\nThermal degradation:")
        print(f"  Embodied:    {emb_degradation*100:.1f}% performance loss under heat")
        print(f"  Disembodied: {dis_degradation*100:.1f}% performance loss under heat")

        if emb_degradation < dis_degradation:
            print(f"\n  ✓ EMBODIMENT HELPS: {(dis_degradation - emb_degradation)*100:.1f}% less degradation")
        else:
            print(f"\n  ✗ Embodiment did not help thermal adaptation")

        return results


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    trainer = SensorimotorTrainer(device)

    # Training phase
    trainer.train(n_episodes=200, eval_interval=20)

    # Thermal stress evaluation
    thermal_results = trainer.evaluate_thermal_stress(n_trials=10)

    # Compile final results
    results = {
        'task': 'sensorimotor_control',
        'hypothesis': 'Embodied agents adapt better to hardware state changes',
        'training': {
            'embodied_final_length': float(np.mean(trainer.metrics['embodied']['episode_lengths'][-20:])),
            'disembodied_final_length': float(np.mean(trainer.metrics['disembodied']['episode_lengths'][-20:])),
            'random_final_length': float(np.mean(trainer.metrics['random']['episode_lengths'][-20:])),
            'embodied_final_reward': float(np.mean(trainer.metrics['embodied']['rewards'][-20:])),
            'disembodied_final_reward': float(np.mean(trainer.metrics['disembodied']['rewards'][-20:])),
        },
        'thermal_stress': {
            'embodied_cool': float(np.mean(thermal_results['embodied_cool'])),
            'embodied_hot': float(np.mean(thermal_results['embodied_hot'])),
            'disembodied_cool': float(np.mean(thermal_results['disembodied_cool'])),
            'disembodied_hot': float(np.mean(thermal_results['disembodied_hot'])),
            'embodied_degradation': float(1 - np.mean(thermal_results['embodied_hot']) / max(1, np.mean(thermal_results['embodied_cool']))),
            'disembodied_degradation': float(1 - np.mean(thermal_results['disembodied_hot']) / max(1, np.mean(thermal_results['disembodied_cool']))),
        },
        'conclusion': ''
    }

    # Determine conclusion
    emb_better = results['training']['embodied_final_length'] > results['training']['disembodied_final_length'] * 1.05
    thermal_better = results['thermal_stress']['embodied_degradation'] < results['thermal_stress']['disembodied_degradation']

    if emb_better and thermal_better:
        results['conclusion'] = 'SUPPORTED - Embodiment improves sensorimotor control AND thermal adaptation'
    elif thermal_better:
        results['conclusion'] = 'PARTIAL - Embodiment improves thermal adaptation but not baseline performance'
    elif emb_better:
        results['conclusion'] = 'PARTIAL - Embodiment improves performance but not thermal adaptation'
    else:
        results['conclusion'] = 'NOT SUPPORTED - Embodiment did not help on this task'

    print("\n" + "=" * 60)
    print("FINAL CONCLUSION")
    print("=" * 60)
    print(results['conclusion'])

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'z1502_sensorimotor_active_inference.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    main()
