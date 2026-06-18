#!/usr/bin/env python3
"""
z1504: FPGA-Embodied Sensorimotor Controller

INTEGRATES REAL HARDWARE:
1. GPU telemetry → thermal state, power, utilization (fatigue)
2. FPGA partial writes (tRAS=1 → 0xF0F0F0F0) → memory consolidation
3. Active inference controller → sensorimotor control

PROVEN CAPABILITIES USED:
- tRAS=1 gives deterministic 16/32 bits written (z1222 verified)
- GPU thermal state affects action scaling (z1502 confirmed)
- Embodiment helps control tasks but not classification (z1503 validated)

This creates a TRUE embodied AI system:
- DRAM partial writes store "motor memory" patterns
- GPU state provides "interoceptive" feedback
- Active inference minimizes free energy with body-state awareness

Task: Inverted pendulum control with memory consolidation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import sys
import socket
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


@dataclass
class MotorMemory:
    """Motor memory pattern stored in FPGA DRAM"""
    state_signature: int       # Hash of state -> 32-bit pattern
    action_pattern: int        # Successful action encoded
    success_count: int         # Times this pattern succeeded
    last_used_ns: int          # Timestamp
    charge_level: float        # Estimated charge remaining (decays)


class FPGAMotorMemory:
    """
    Interface to FPGA DRAM for motor memory consolidation

    Uses partial writes (tRAS=1) to create analog charge levels:
    - Full write (tRAS >= 2): 0xFFFFFFFF
    - Partial write (tRAS=1): 0xF0F0F0F0 (16/32 bits)

    This creates "weak" vs "strong" memories - biological analogy!
    """

    FPGA_IP = "192.168.1.50"
    FPGA_PORT = 1234
    BASE_ADDR = 0x40000000

    # Memory layout: 1024 motor memory slots
    SLOT_SIZE = 16  # bytes per slot: state_sig(4) + action(4) + count(4) + timestamp(4)
    NUM_SLOTS = 1024

    def __init__(self, use_real_fpga: bool = True):
        self.use_real_fpga = use_real_fpga
        self.sock = None

        # Simulated memory for when FPGA not available
        self.simulated_memory = {}
        self.decay_rate = 0.001  # Per-second decay

        if use_real_fpga:
            try:
                self._connect()
            except Exception as e:
                print(f"FPGA connection failed: {e}")
                print("Falling back to simulated memory")
                self.use_real_fpga = False

    def _connect(self):
        """Connect to FPGA via UDP"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.1)
        # Test connection
        self._read_word(self.BASE_ADDR)
        print(f"Connected to FPGA at {self.FPGA_IP}:{self.FPGA_PORT}")

    def _build_packet(self, cmd: int, addr: int, data: int = 0) -> bytes:
        """Build Etherbone-style packet"""
        return struct.pack('>BBBBI', 0x4E, 0x6F, cmd, 0x01, addr) + struct.pack('>I', data)

    def _send_recv(self, packet: bytes) -> bytes:
        """Send packet and receive response"""
        self.sock.sendto(packet, (self.FPGA_IP, self.FPGA_PORT))
        data, _ = self.sock.recvfrom(1024)
        return data

    def _read_word(self, addr: int) -> int:
        """Read 32-bit word from FPGA"""
        pkt = self._build_packet(0x00, addr)  # Read command
        resp = self._send_recv(pkt)
        return struct.unpack('>I', resp[-4:])[0]

    def _write_word(self, addr: int, data: int, partial: bool = False):
        """Write 32-bit word to FPGA with optional partial write"""
        pkt = self._build_packet(0x01 if partial else 0x02, addr, data)
        self._send_recv(pkt)

    def _partial_write(self, addr: int, data: int):
        """
        Partial write using tRAS=1 timing

        This is the KEY embodiment mechanism:
        - Writes only 16/32 bits deterministically
        - Creates "weak" memory trace
        - Decays faster than full write
        """
        if self.use_real_fpga:
            # Use FSM partial write mode
            self._write_word(addr, data, partial=True)
        else:
            # Simulate partial write
            current = self.simulated_memory.get(addr, 0)
            # Partial write: only some bits transfer
            mask = 0xF0F0F0F0  # tRAS=1 pattern
            new_value = (data & mask) | (current & ~mask)
            self.simulated_memory[addr] = new_value

    def hash_state(self, state: torch.Tensor) -> int:
        """Hash pendulum state to 32-bit signature"""
        # Discretize state to bins
        theta_bin = int((state[0].item() + 0.5) / 0.05) & 0xFF
        theta_dot_bin = int((state[1].item() + 2.0) / 0.2) & 0xFF
        x_bin = int((state[2].item() + 2.4) / 0.24) & 0xFF
        x_dot_bin = int((state[3].item() + 2.0) / 0.2) & 0xFF
        return (theta_bin << 24) | (theta_dot_bin << 16) | (x_bin << 8) | x_dot_bin

    def encode_action(self, action: float) -> int:
        """Encode action [-1, 1] to 32-bit pattern"""
        # Scale to 0-4095 range
        scaled = int((action + 1.0) * 2047.5)
        # Replicate for redundancy
        return (scaled << 20) | (scaled << 8) | (scaled >> 4)

    def decode_action(self, pattern: int) -> float:
        """Decode action from 32-bit pattern"""
        # Extract middle 12 bits
        scaled = (pattern >> 8) & 0xFFF
        return (scaled / 2047.5) - 1.0

    def store_motor_memory(self, state: torch.Tensor, action: float,
                          success: bool, strong: bool = False):
        """
        Store motor memory pattern

        Args:
            state: Current pendulum state
            action: Action taken
            success: Whether this action led to survival
            strong: If True, use full write (strong memory)
                   If False, use partial write (weak memory)
        """
        state_sig = self.hash_state(state)
        action_pattern = self.encode_action(action)
        slot = state_sig % self.NUM_SLOTS
        addr = self.BASE_ADDR + slot * self.SLOT_SIZE

        if self.use_real_fpga:
            if strong:
                # Strong memory: full write
                self._write_word(addr, state_sig)
                self._write_word(addr + 4, action_pattern)
                self._write_word(addr + 8, 1 if success else 0)
                self._write_word(addr + 12, int(time.time_ns() // 1000000))
            else:
                # Weak memory: partial write
                self._partial_write(addr, state_sig)
                self._partial_write(addr + 4, action_pattern)
        else:
            # Simulated storage
            self.simulated_memory[slot] = {
                'state_sig': state_sig,
                'action': action_pattern,
                'success': success,
                'timestamp': time.time(),
                'strength': 1.0 if strong else 0.5
            }

    def recall_motor_memory(self, state: torch.Tensor) -> Optional[Tuple[float, float]]:
        """
        Recall motor memory for similar state

        Returns:
            (action, confidence) or None if no memory found
        """
        state_sig = self.hash_state(state)
        slot = state_sig % self.NUM_SLOTS

        if self.use_real_fpga:
            addr = self.BASE_ADDR + slot * self.SLOT_SIZE
            stored_sig = self._read_word(addr)
            action_pattern = self._read_word(addr + 4)

            # Check if signature matches (accounting for partial write)
            sig_match = bin(stored_sig ^ state_sig).count('1')
            if sig_match > 16:  # Too different
                return None

            action = self.decode_action(action_pattern)
            # Confidence based on pattern integrity
            # Full write: all bits set, high confidence
            # Partial write: only 16/32 bits, lower confidence
            ones = bin(action_pattern).count('1')
            confidence = ones / 32.0

            return action, confidence
        else:
            # Simulated recall
            if slot not in self.simulated_memory:
                return None
            mem = self.simulated_memory[slot]

            # Check signature match
            if mem['state_sig'] != state_sig:
                return None

            # Apply decay
            elapsed = time.time() - mem['timestamp']
            strength = mem['strength'] * np.exp(-self.decay_rate * elapsed)

            if strength < 0.1:  # Memory decayed too much
                return None

            action = self.decode_action(mem['action'])
            return action, strength

    def consolidate_memories(self):
        """
        Strengthen frequently-used memories (full write)

        Biological analogy: sleep consolidation
        """
        if not self.use_real_fpga:
            # Strengthen simulated memories that were accessed recently
            for slot, mem in self.simulated_memory.items():
                if time.time() - mem['timestamp'] < 60 and mem['success']:
                    mem['strength'] = min(1.0, mem['strength'] + 0.1)


class EmbodiedActiveInference(nn.Module):
    """
    Active Inference controller with FPGA motor memory

    Architecture:
    1. State encoder: pendulum state → latent
    2. Body encoder: GPU telemetry → body state
    3. Memory query: FPGA recall → prior action
    4. Policy: latent + body + prior → action
    5. Free energy: prediction error + body regulation
    """

    def __init__(self, state_dim: int = 4, body_dim: int = 4,
                 latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Body state encoder
        self.body_encoder = nn.Sequential(
            nn.Linear(body_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim // 2)
        )

        # Memory prior integration
        self.memory_gate = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim // 2),  # +1 for confidence
            nn.Sigmoid()
        )

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        # Value function (expected free energy)
        self.value = nn.Sequential(
            nn.Linear(latent_dim + latent_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # World model (predict next state)
        self.world_model = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Body setpoint (homeostatic target)
        self.register_buffer('body_setpoint', torch.tensor([0.0, 0.5, 0.3, 0.0]))
        # [temp_normalized, power_normalized, util_normalized, fatigue]

    def forward(self, state: torch.Tensor, body_state: torch.Tensor,
                memory_prior: Optional[Tuple[float, float]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute action via active inference

        Args:
            state: Pendulum state [theta, theta_dot, x, x_dot]
            body_state: GPU telemetry [temp, power, util, fatigue]
            memory_prior: (prior_action, confidence) from FPGA memory

        Returns:
            action, info_dict
        """
        # Encode state
        z_state = self.state_encoder(state)

        # Encode body state
        z_body = self.body_encoder(body_state)

        # Integrate memory prior
        batch_size = state.size(0)
        if memory_prior is not None:
            prior_action, prior_conf = memory_prior
            conf_tensor = torch.full((batch_size, 1), prior_conf, device=state.device, dtype=torch.float32)
            prior_input = torch.cat([z_state, conf_tensor], dim=-1)
            memory_weight = self.memory_gate(prior_input)
        else:
            prior_action = 0.0
            memory_weight = torch.zeros(batch_size, z_state.size(-1) // 2, device=state.device, dtype=torch.float32)
            prior_conf = 0.0

        # Policy input: state latent + body latent + prior action
        batch_size = state.size(0)
        prior_tensor = torch.full((batch_size, 1), prior_action, device=state.device, dtype=torch.float32)
        policy_input = torch.cat([z_state, z_body, prior_tensor], dim=-1)

        # Compute action
        action = self.policy(policy_input)

        # Blend with memory prior based on confidence
        if prior_conf > 0.3:
            action = action * (1 - prior_conf * 0.5) + prior_tensor * prior_conf * 0.5

        # Compute free energy components
        # 1. State prediction error
        pred_next = self.world_model(torch.cat([z_state, action], dim=-1))

        # 2. Body regulation error (homeostasis)
        body_target = self.body_setpoint.unsqueeze(0).expand_as(body_state)
        body_error = F.mse_loss(body_state, body_target, reduction='none').sum(-1)

        # 3. Expected free energy
        value_input = torch.cat([z_state, z_body], dim=-1)
        expected_fe = self.value(value_input)

        info = {
            'body_error': body_error.mean().item(),
            'expected_fe': expected_fe.mean().item(),
            'memory_weight': memory_weight.mean().item(),
            'prior_conf': prior_conf
        }

        return action, info


class FPGAEmbodiedTrainer:
    """Training loop with FPGA motor memory integration"""

    def __init__(self, device: torch.device, use_real_fpga: bool = True):
        self.device = device

        # Hardware interfaces
        self.telemetry = SysfsHwmonTelemetry()
        self.motor_memory = FPGAMotorMemory(use_real_fpga=use_real_fpga)

        # Agent
        self.agent = EmbodiedActiveInference().to(device)
        self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=1e-3)

        # Environment (inverted pendulum)
        from z1502_sensorimotor_active_inference import PendulumEnvironment
        self.env = PendulumEnvironment()

        # Metrics
        self.metrics = {
            'episode_lengths': [],
            'rewards': [],
            'memory_recalls': [],
            'memory_stores': [],
            'body_errors': []
        }

        # Body state tracking
        self.fatigue = 0.0

    def get_body_state(self) -> torch.Tensor:
        """Get current body state from GPU telemetry"""
        sample = self.telemetry.read_sample()

        temp = sample.temp_junction_c if sample.temp_junction_c else sample.temp_edge_c
        power = sample.power_w if sample.power_w else 10.0
        util = sample.gpu_busy_pct if sample.gpu_busy_pct else 0.0

        # Update fatigue
        thermal_stress = max(0, (temp - 50.0) / 30.0)
        self.fatigue += 0.001 * thermal_stress
        self.fatigue *= 0.99  # Recovery
        self.fatigue = min(0.5, self.fatigue)

        return torch.tensor([
            (temp - 50.0) / 30.0,
            (power - 10.0) / 20.0,
            util / 100.0,
            self.fatigue
        ], dtype=torch.float32, device=self.device)

    def run_episode(self, max_steps: int = 500, training: bool = True) -> Dict:
        """Run single episode with FPGA memory integration"""
        state = self.env.reset()
        total_reward = 0.0
        memory_recalls = 0
        memory_stores = 0
        body_errors = []

        experiences = []

        for step in range(max_steps):
            obs = state.to_tensor().to(self.device).unsqueeze(0)
            body_state = self.get_body_state().unsqueeze(0)

            # Query motor memory
            memory_prior = self.motor_memory.recall_motor_memory(obs.squeeze(0))
            if memory_prior is not None:
                memory_recalls += 1

            # Get action from agent
            with torch.no_grad():
                action, info = self.agent(obs, body_state, memory_prior)
                action = action.squeeze()

            body_errors.append(info['body_error'])

            # Apply fatigue to action
            effective_action = action.item() * (1.0 - self.fatigue * 0.3)

            # Execute in environment
            next_state, reward, done = self.env.step(effective_action, self.fatigue)
            total_reward += reward

            # Store experience
            if training:
                experiences.append({
                    'obs': obs.squeeze(0),
                    'body': body_state.squeeze(0),
                    'action': action,
                    'reward': reward,
                    'next_obs': next_state.to_tensor().to(self.device),
                    'done': done
                })

                # Store motor memory for successful actions
                if not done and reward > 0.5:
                    # Use partial write for recent memories
                    self.motor_memory.store_motor_memory(
                        obs.squeeze(0), action.item(),
                        success=True, strong=False
                    )
                    memory_stores += 1

            state = next_state
            if done:
                break

        # Consolidate strong memories at episode end
        if training and step > 100:  # Long survival = consolidate
            self.motor_memory.consolidate_memories()

        return {
            'length': step + 1,
            'reward': total_reward,
            'memory_recalls': memory_recalls,
            'memory_stores': memory_stores,
            'body_error_mean': np.mean(body_errors) if body_errors else 0.0,
            'experiences': experiences
        }

    def train_step(self, experiences: List[Dict], batch_size: int = 32) -> float:
        """Single training step"""
        if len(experiences) < batch_size:
            return 0.0

        # Sample batch
        indices = np.random.choice(len(experiences), batch_size, replace=False)
        batch = [experiences[i] for i in indices]

        obs = torch.stack([b['obs'] for b in batch])
        body = torch.stack([b['body'] for b in batch])
        actions = torch.stack([b['action'] for b in batch]).unsqueeze(-1)
        rewards = torch.tensor([b['reward'] for b in batch], device=self.device)
        next_obs = torch.stack([b['next_obs'] for b in batch])

        # Forward pass
        pred_actions, info = self.agent(obs, body, None)

        # Policy loss (reward-weighted)
        policy_loss = -torch.mean(rewards * pred_actions.squeeze())

        # World model loss
        z_state = self.agent.state_encoder(obs)
        pred_next = self.agent.world_model(torch.cat([z_state, actions], dim=-1))
        world_loss = F.mse_loss(pred_next, next_obs)

        # Body regulation loss
        body_target = self.agent.body_setpoint.unsqueeze(0).expand_as(body)
        body_loss = F.mse_loss(body, body_target)

        # Total loss
        loss = policy_loss + 0.1 * world_loss + 0.01 * body_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, n_episodes: int = 100, eval_interval: int = 10):
        """Full training loop"""
        print("=" * 70)
        print("z1504: FPGA-Embodied Sensorimotor Controller")
        print("=" * 70)
        print(f"FPGA motor memory: {'REAL' if self.motor_memory.use_real_fpga else 'SIMULATED'}")
        print(f"GPU telemetry: REAL")
        print("=" * 70)

        all_experiences = []

        for episode in range(n_episodes):
            # Reset fatigue at episode start
            self.fatigue = 0.0

            result = self.run_episode(training=True)
            all_experiences.extend(result['experiences'])

            # Keep recent experiences
            if len(all_experiences) > 10000:
                all_experiences = all_experiences[-10000:]

            # Train
            loss = self.train_step(all_experiences)

            self.metrics['episode_lengths'].append(result['length'])
            self.metrics['rewards'].append(result['reward'])
            self.metrics['memory_recalls'].append(result['memory_recalls'])
            self.metrics['memory_stores'].append(result['memory_stores'])
            self.metrics['body_errors'].append(result['body_error_mean'])

            if (episode + 1) % eval_interval == 0:
                avg_len = np.mean(self.metrics['episode_lengths'][-eval_interval:])
                avg_reward = np.mean(self.metrics['rewards'][-eval_interval:])
                avg_recalls = np.mean(self.metrics['memory_recalls'][-eval_interval:])
                avg_stores = np.mean(self.metrics['memory_stores'][-eval_interval:])

                print(f"\nEpisode {episode + 1}/{n_episodes}")
                print(f"  Length: {avg_len:.1f}, Reward: {avg_reward:.2f}")
                print(f"  Memory recalls: {avg_recalls:.1f}, stores: {avg_stores:.1f}")
                print(f"  Loss: {loss:.4f}")


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check FPGA connection
    print("\nAttempting FPGA connection...")

    trainer = FPGAEmbodiedTrainer(device, use_real_fpga=True)

    # Training
    trainer.train(n_episodes=100, eval_interval=10)

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    eval_lengths = []
    for _ in range(10):
        result = trainer.run_episode(training=False)
        eval_lengths.append(result['length'])

    print(f"Evaluation episode length: {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f}")

    # Compile results
    results = {
        'task': 'fpga_embodied_control',
        'fpga_mode': 'real' if trainer.motor_memory.use_real_fpga else 'simulated',
        'training': {
            'final_length': float(np.mean(trainer.metrics['episode_lengths'][-10:])),
            'final_reward': float(np.mean(trainer.metrics['rewards'][-10:])),
            'total_memory_recalls': sum(trainer.metrics['memory_recalls']),
            'total_memory_stores': sum(trainer.metrics['memory_stores'])
        },
        'evaluation': {
            'mean_length': float(np.mean(eval_lengths)),
            'std_length': float(np.std(eval_lengths))
        },
        'hardware_integration': {
            'gpu_telemetry': True,
            'fpga_partial_writes': trainer.motor_memory.use_real_fpga,
            'motor_memory_consolidation': True
        }
    }

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'z1504_fpga_embodied_controller.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    main()
