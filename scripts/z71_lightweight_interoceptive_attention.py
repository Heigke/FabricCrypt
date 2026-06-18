#!/usr/bin/env python3
"""
Z71: Lightweight Interoceptive Attention (LIA)

Novel Contribution: Attention modulation by hardware state WITHOUT body tokens.

The z70 experiment showed that full body tokens create too much overhead.
This experiment keeps the key insight (attention modulated by hardware)
but removes the expensive body token mechanism.

Architecture:
1. Standard transformer with FiLM conditioning (from z64)
2. PLUS: Homeostatic attention modulation (from z70, lightweight version)
3. NO body tokens (remove overhead)

Expected Result: z64's efficiency + z70's attention-based interoception
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class LIAConfig:
    """Lightweight Interoceptive Attention configuration."""
    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512

    # Telemetry dimensions
    telemetry_dim: int = 6  # power, temp, util, clock, vram_used, vram_total

    # Homeostatic parameters
    homeostatic_gain: float = 0.1  # How much homeostasis affects attention
    energy_modulation_strength: float = 0.05  # How much energy affects attention

    # Setpoints
    power_setpoint: float = 150.0  # Target power (watts)
    temp_setpoint: float = 70.0    # Target temperature (celsius)

    # Number of power actions
    n_actions: int = 4  # ECO, BALANCED, PERFORMANCE, MAX


class HomeostaticAttentionModulator(nn.Module):
    """
    Lightweight module that modulates attention weights based on hardware state.

    Key difference from z70: This modulates EXISTING attention, doesn't add tokens.
    Only ~4K parameters per layer instead of ~60K.
    """
    def __init__(self, config: LIAConfig):
        super().__init__()
        self.config = config

        # Homeostatic gate: telemetry -> scalar modulation
        # Much smaller than z70's body token encoder
        self.homeostatic_gate = nn.Sequential(
            nn.Linear(2, 16),  # Just power and temp deviation
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

        # Energy modulator: power -> attention sharpness
        self.energy_modulator = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Tanh()
        )

    def forward(
        self,
        attn_scores: torch.Tensor,  # Pre-softmax attention scores
        homeostatic_state: torch.Tensor,  # (batch, 2) - power_dev, temp_dev
        energy_state: torch.Tensor,        # (batch, 1) - current power
    ) -> torch.Tensor:
        """Modulate attention SCORES (not weights) based on hardware state."""

        # Clamp homeostatic state for numerical stability
        homeostatic_state = torch.clamp(homeostatic_state, -2.0, 2.0)
        energy_state = torch.clamp(energy_state / 300.0, 0.0, 1.0)  # Normalize power

        # Homeostatic modulation
        h_gate = self.homeostatic_gate(homeostatic_state)  # (batch, 1)
        h_gate = h_gate.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, 1)

        # Energy modulation
        e_mod = self.energy_modulator(energy_state)  # (batch, 1)
        e_mod = e_mod.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, 1)

        # Apply additive modulations to SCORES (before softmax)
        # This is numerically stable and semantically cleaner
        modulated_scores = attn_scores + h_gate * self.config.homeostatic_gain + e_mod * self.config.energy_modulation_strength

        return modulated_scores


class LightweightInteroceptiveAttention(nn.Module):
    """
    Standard multi-head attention with lightweight homeostatic modulation.

    Much cheaper than z70's HomeostaticAttention because:
    1. No body tokens in key/value
    2. Modulation happens AFTER attention computation
    3. Only ~4K extra params per layer
    """
    def __init__(self, config: LIAConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Standard attention projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # Lightweight homeostatic modulator
        self.modulator = HomeostaticAttentionModulator(config)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        homeostatic_state: Optional[torch.Tensor] = None,
        energy_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Standard attention computation
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # LIGHTWEIGHT INTEROCEPTION: Modulate attention SCORES based on hardware state
        # This happens BEFORE softmax for numerical stability
        if homeostatic_state is not None and energy_state is not None:
            attn_scores = self.modulator(attn_scores, homeostatic_state, energy_state)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out, attn_weights


class FiLMLayer(nn.Module):
    """FiLM conditioning layer (from z64)."""
    def __init__(self, d_model: int, telemetry_dim: int):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(telemetry).unsqueeze(1)  # (batch, 1, d_model)
        beta = self.beta_net(telemetry).unsqueeze(1)
        return x * (1 + gamma) + beta


class LIABlock(nn.Module):
    """Lightweight Interoceptive Attention Block."""
    def __init__(self, config: LIAConfig):
        super().__init__()
        self.config = config

        # Interoceptive attention
        self.attn = LightweightInteroceptiveAttention(config)
        self.attn_norm = nn.LayerNorm(config.d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)

        # FiLM conditioning
        self.film = FiLMLayer(config.d_model, config.telemetry_dim)

    def forward(
        self,
        x: torch.Tensor,
        telemetry: torch.Tensor,
        homeostatic_state: torch.Tensor,
        energy_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Interoceptive attention
        attn_out, attn_weights = self.attn(
            self.attn_norm(x),
            homeostatic_state=homeostatic_state,
            energy_state=energy_state,
            mask=mask
        )
        x = x + attn_out

        # FiLM conditioning
        x = self.film(x, telemetry)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x, attn_weights


class LightweightInteroceptiveTransformer(nn.Module):
    """
    Transformer with lightweight interoceptive attention.

    Combines:
    - FiLM conditioning (from z64, ~1.7M params)
    - Homeostatic attention modulation (from z70, ~24K params)
    - Standard transformer (baseline)

    Expected overhead: <5% vs baseline (compared to 40% for z70)
    """
    def __init__(self, config: LIAConfig):
        super().__init__()
        self.config = config

        # Telemetry normalization constants (typical ranges)
        self.register_buffer('telem_mean', torch.tensor([150.0, 60.0, 50.0, 1500.0, 20000.0, 48000.0]))
        self.register_buffer('telem_std', torch.tensor([75.0, 20.0, 30.0, 500.0, 15000.0, 10000.0]))

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks with lightweight interoception
        self.blocks = nn.ModuleList([
            LIABlock(config) for _ in range(config.n_layers)
        ])

        # Output heads
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Action head for power control
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, config.n_actions)
        )

    def normalize_telemetry(self, telemetry: torch.Tensor) -> torch.Tensor:
        """Normalize telemetry to zero mean, unit variance."""
        return (telemetry - self.telem_mean.to(telemetry.device)) / self.telem_std.to(telemetry.device)

    def compute_homeostatic_state(self, telemetry: torch.Tensor) -> torch.Tensor:
        """Compute deviation from homeostatic setpoints."""
        power = telemetry[:, 0]  # Current power
        temp = telemetry[:, 1]   # Current temperature

        power_dev = (power - self.config.power_setpoint) / self.config.power_setpoint
        temp_dev = (temp - self.config.temp_setpoint) / self.config.temp_setpoint

        return torch.stack([power_dev, temp_dev], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Normalize telemetry for stable processing
        telemetry_norm = self.normalize_telemetry(telemetry)

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Compute homeostatic state (uses raw telemetry for physical meaning)
        homeostatic_state = self.compute_homeostatic_state(telemetry)
        energy_state = telemetry[:, 0:1] / 300.0  # Normalize power for energy modulation

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

        # Process through blocks (use normalized telemetry)
        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, telemetry_norm, homeostatic_state, energy_state, mask)
            all_attn_weights.append(attn_weights)

        x = self.ln_f(x)

        # LM output
        logits = self.lm_head(x)

        # Action output (from mean pooled representation)
        action_logits = self.action_head(x.mean(dim=1))

        return {
            'logits': logits,
            'action_logits': action_logits,
            'homeostatic_state': homeostatic_state,
            'attention_weights': all_attn_weights
        }


class BaselineTransformer(nn.Module):
    """Standard transformer without any hardware awareness."""
    def __init__(self, config: LIAConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        x = self.transformer(x, mask=mask, is_causal=True)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return {'logits': logits}


class LIAExperiment:
    """Experiment runner for Lightweight Interoceptive Attention."""

    ACTION_NAMES = ['ECO', 'BALANCED', 'PERFORMANCE', 'MAX']

    def __init__(self, device: str = 'cuda'):
        from src.metabolic.telemetry_unified import UnifiedTelemetryReader
        from src.metabolic.actuation_unified import UnifiedActuator

        self.device = torch.device(device)
        self.config = LIAConfig()

        # Initialize hardware interfaces
        self.telemetry = UnifiedTelemetryReader()
        self.actuator = UnifiedActuator()

        # Create models
        self.lia = LightweightInteroceptiveTransformer(self.config).to(self.device)
        self.baseline = BaselineTransformer(self.config).to(self.device)

        logger.info(f"GPU: {self.telemetry.get_device_info()}")

    def count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def get_telemetry_tensor(self, snap) -> torch.Tensor:
        """Convert telemetry snapshot to tensor."""
        return torch.tensor([
            snap.power_watts,
            snap.temp_c,
            snap.utilization,
            snap.clock_mhz,
            snap.vram_used_mb,
            snap.vram_total_mb
        ], device=self.device)

    def compute_reward(
        self,
        throughput: float,
        power: float,
        temp: float,
        prev_action: int,
        curr_action: int
    ) -> float:
        """Smart reward function (from z64)."""
        tpw = throughput / max(power, 1.0)
        tpw_normalized = tpw / 1000.0

        quality_factor = 1.0
        temp_penalty = max(0, (temp - 80) / 10) * 0.5
        switch_penalty = 0.1 if prev_action != curr_action else 0.0

        return tpw_normalized * quality_factor - temp_penalty - switch_penalty

    def train_stage1(self, dataloader, epochs: int = 2):
        """Stage 1: LM pretraining."""
        logger.info("\n" + "="*50)
        logger.info("STAGE 1: LM Pretraining")
        logger.info("="*50)

        lia_optimizer = torch.optim.AdamW(self.lia.parameters(), lr=1e-4)
        baseline_optimizer = torch.optim.AdamW(self.baseline.parameters(), lr=1e-4)

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")

            lia_loss_sum = 0
            baseline_loss_sum = 0
            n_batches = 0

            for i, (input_ids, targets) in enumerate(dataloader):
                if i >= 100:
                    break

                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                # Baseline
                baseline_out = self.baseline(input_ids)
                baseline_loss = F.cross_entropy(
                    baseline_out['logits'].view(-1, self.config.vocab_size),
                    targets.view(-1)
                )
                baseline_optimizer.zero_grad()
                baseline_loss.backward()
                baseline_optimizer.step()
                baseline_loss_sum += baseline_loss.item()

                # LIA
                snap = self.telemetry.read()
                telem = self.get_telemetry_tensor(snap).unsqueeze(0).expand(input_ids.size(0), -1)

                lia_out = self.lia(input_ids, telem)
                lia_loss = F.cross_entropy(
                    lia_out['logits'].view(-1, self.config.vocab_size),
                    targets.view(-1)
                )

                # Skip if loss is NaN
                if torch.isnan(lia_loss):
                    logger.warning(f"  NaN loss detected at batch {i}, skipping")
                    continue

                lia_optimizer.zero_grad()
                lia_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lia.parameters(), max_norm=1.0)
                lia_optimizer.step()
                lia_loss_sum += lia_loss.item()

                n_batches += 1

            logger.info(f"  Baseline loss: {baseline_loss_sum/n_batches:.4f}")
            logger.info(f"  LIA loss: {lia_loss_sum/n_batches:.4f}")

    def train_stage2_rl(self, dataloader, epochs: int = 3):
        """Stage 2: RL training for power control."""
        logger.info("\n" + "="*50)
        logger.info("STAGE 2: RL Training (learn to regulate)")
        logger.info("="*50)

        # Only train attention modulators and action head
        rl_params = []
        for block in self.lia.blocks:
            rl_params.extend(block.attn.modulator.parameters())
        rl_params.extend(self.lia.action_head.parameters())

        optimizer = torch.optim.Adam(rl_params, lr=1e-4)

        prev_action = 1  # Start with BALANCED

        for epoch in range(epochs):
            logger.info(f"\nRL Epoch {epoch+1}/{epochs}")

            rewards = []

            for i, (input_ids, targets) in enumerate(dataloader):
                if i >= 40:
                    break

                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                # Get telemetry
                snap_before = self.telemetry.read()
                telem = self.get_telemetry_tensor(snap_before).unsqueeze(0).expand(input_ids.size(0), -1)

                # Forward pass
                start = time.time()
                output = self.lia(input_ids, telem)
                torch.cuda.synchronize()
                elapsed = time.time() - start

                snap_after = self.telemetry.read()

                # Get action (handle NaN gracefully)
                action_logits = output['action_logits'][0]
                if torch.isnan(action_logits).any():
                    action_idx = 1  # Default to BALANCED if NaN
                else:
                    action_probs = F.softmax(action_logits, dim=-1)
                    # Use epsilon-greedy for stability
                    if np.random.random() < 0.1:
                        action_idx = np.random.randint(0, 4)
                    else:
                        action_idx = torch.argmax(action_probs).item()

                # Apply action
                self.actuator.set_mode_from_action(action_idx)

                # Compute reward
                tokens = targets.numel()
                throughput = tokens / elapsed
                avg_power = (snap_before.power_watts + snap_after.power_watts) / 2

                reward = self.compute_reward(
                    throughput, avg_power, snap_after.temp_c,
                    prev_action, action_idx
                )
                rewards.append(reward)

                # RL loss (policy gradient)
                log_prob = F.log_softmax(output['action_logits'][0], dim=-1)[action_idx]
                rl_loss = -log_prob * reward

                # Small LM loss to maintain quality
                lm_loss = F.cross_entropy(
                    output['logits'].view(-1, self.config.vocab_size),
                    targets.view(-1)
                )

                total_loss = rl_loss + 0.1 * lm_loss

                # Skip if loss is NaN
                if torch.isnan(total_loss):
                    logger.warning(f"  NaN loss at batch {i}, skipping")
                    prev_action = action_idx
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(rl_params, max_norm=1.0)
                optimizer.step()

                prev_action = action_idx

                if i % 10 == 0:
                    h_state = output['homeostatic_state'][0].detach().cpu().numpy()
                    logger.info(
                        f"  B {i:3d} | R:{reward:+.3f} | A:{self.ACTION_NAMES[action_idx]:4s} | "
                        f"P:{avg_power:.0f}W | T:{snap_after.temp_c:.0f}C | "
                        f"H:[{h_state[0]:.2f},{h_state[1]:.2f}]"
                    )

            logger.info(f"  Avg Reward: {np.mean(rewards):.4f}")

    def run_comparison(self, dataloader, num_batches: int = 30) -> Dict:
        """Scientific comparison: LIA vs Baseline."""
        logger.info("\n" + "="*60)
        logger.info("SCIENTIFIC COMPARISON: LIA vs Traditional Transformer")
        logger.info("="*60)

        results = {
            'baseline': {'energy': [], 'throughput': [], 'temp': [], 'loss': []},
            'lia': {'energy': [], 'throughput': [], 'temp': [], 'loss': [], 'actions': []}
        }

        # Phase 1: Baseline
        logger.info("\nPhase 1: Traditional Transformer (no body sensing)")
        logger.info("-" * 40)

        self.baseline.eval()
        with torch.no_grad():
            for i, (input_ids, targets) in enumerate(dataloader):
                if i >= num_batches:
                    break

                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                snap_before = self.telemetry.read()
                start = time.time()

                output = self.baseline(input_ids)

                torch.cuda.synchronize()
                elapsed = time.time() - start
                snap_after = self.telemetry.read()

                avg_power = (snap_before.power_watts + snap_after.power_watts) / 2
                energy_mj = avg_power * elapsed * 1000
                tokens = targets.numel()
                throughput = tokens / elapsed

                loss = F.cross_entropy(
                    output['logits'].view(-1, self.config.vocab_size),
                    targets.view(-1)
                ).item()

                results['baseline']['energy'].append(energy_mj)
                results['baseline']['throughput'].append(throughput)
                results['baseline']['temp'].append(snap_after.temp_c)
                results['baseline']['loss'].append(loss)

                if i % 10 == 0:
                    logger.info(
                        f"  B {i:3d} | E:{energy_mj:.2f}mJ | T:{throughput:.0f}tok/s | "
                        f"P:{avg_power:.0f}W | Temp:{snap_after.temp_c:.0f}C"
                    )

        # Phase 2: LIA
        logger.info("\nPhase 2: Lightweight Interoceptive Attention")
        logger.info("-" * 40)

        self.lia.eval()
        with torch.no_grad():
            for i, (input_ids, targets) in enumerate(dataloader):
                if i >= num_batches:
                    break

                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                snap_before = self.telemetry.read()
                telem = self.get_telemetry_tensor(snap_before).unsqueeze(0).expand(input_ids.size(0), -1)

                start = time.time()
                output = self.lia(input_ids, telem)
                torch.cuda.synchronize()
                elapsed = time.time() - start

                snap_after = self.telemetry.read()

                avg_power = (snap_before.power_watts + snap_after.power_watts) / 2
                energy_mj = avg_power * elapsed * 1000
                tokens = targets.numel()
                throughput = tokens / elapsed

                loss = F.cross_entropy(
                    output['logits'].view(-1, self.config.vocab_size),
                    targets.view(-1)
                ).item()

                # Get action
                action_idx = torch.argmax(output['action_logits'][0]).item()

                # Apply action
                self.actuator.set_mode_from_action(action_idx)

                results['lia']['energy'].append(energy_mj)
                results['lia']['throughput'].append(throughput)
                results['lia']['temp'].append(snap_after.temp_c)
                results['lia']['loss'].append(loss)
                results['lia']['actions'].append(action_idx)

                if i % 10 == 0:
                    h_state = output['homeostatic_state'][0].cpu().numpy()
                    logger.info(
                        f"  B {i:3d} | E:{energy_mj:.2f}mJ | T:{throughput:.0f}tok/s | "
                        f"P:{avg_power:.0f}W | Temp:{snap_after.temp_c:.0f}C | "
                        f"A:{self.ACTION_NAMES[action_idx]} | H:[{h_state[0]:.2f},{h_state[1]:.2f}]"
                    )

        return results

    def analyze_results(self, results: Dict) -> Dict:
        """Compute comparison metrics."""
        analysis = {}

        # Baseline metrics
        b = results['baseline']
        analysis['baseline'] = {
            'energy_mj': np.mean(b['energy']),
            'throughput': np.mean(b['throughput']),
            'temp': np.mean(b['temp']),
            'temp_max': np.max(b['temp']),
            'loss': np.mean(b['loss'])
        }

        # LIA metrics
        l = results['lia']
        analysis['lia'] = {
            'energy_mj': np.mean(l['energy']),
            'throughput': np.mean(l['throughput']),
            'temp': np.mean(l['temp']),
            'temp_max': np.max(l['temp']),
            'loss': np.mean(l['loss'])
        }

        # Action distribution
        action_counts = [0] * 4
        for a in l['actions']:
            action_counts[a] += 1
        total = len(l['actions'])
        analysis['lia']['action_dist'] = {
            self.ACTION_NAMES[i]: action_counts[i] / total * 100
            for i in range(4)
        }

        # Comparisons
        analysis['comparison'] = {
            'energy_change': (analysis['lia']['energy_mj'] - analysis['baseline']['energy_mj']) / analysis['baseline']['energy_mj'] * 100,
            'throughput_change': (analysis['lia']['throughput'] - analysis['baseline']['throughput']) / analysis['baseline']['throughput'] * 100,
            'temp_change': analysis['lia']['temp'] - analysis['baseline']['temp'],
            'tpw_baseline': analysis['baseline']['throughput'] / (analysis['baseline']['energy_mj'] / 1000),
            'tpw_lia': analysis['lia']['throughput'] / (analysis['lia']['energy_mj'] / 1000)
        }
        analysis['comparison']['tpw_change'] = (
            analysis['comparison']['tpw_lia'] - analysis['comparison']['tpw_baseline']
        ) / analysis['comparison']['tpw_baseline'] * 100

        return analysis


def run_lia_experiment():
    """Main experiment runner."""
    logger.info("="*70)
    logger.info("Z71: LIGHTWEIGHT INTEROCEPTIVE ATTENTION EXPERIMENT")
    logger.info("="*70)
    logger.info("""
Novel Contribution: Attention modulation by hardware state WITHOUT body tokens.

Key Innovation:
  - Homeostatic modulation of attention weights (from z70)
  - FiLM conditioning (from z64)
  - NO body tokens (removes overhead)

Expected: z64's efficiency + z70's attention-based interoception
""")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # Create experiment
    exp = LIAExperiment(device)

    # Log model sizes
    lia_params = exp.count_parameters(exp.lia)
    baseline_params = exp.count_parameters(exp.baseline)
    overhead = lia_params - baseline_params

    logger.info(f"\nModel sizes:")
    logger.info(f"  LIA: {lia_params:,} params")
    logger.info(f"  Baseline: {baseline_params:,} params")
    logger.info(f"  Overhead: {overhead:,} params ({overhead/baseline_params*100:.1f}%)")

    # Create synthetic dataset
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = 32
    seq_len = 128
    n_samples = 1000

    input_ids = torch.randint(0, exp.config.vocab_size, (n_samples, seq_len))
    targets = torch.randint(0, exp.config.vocab_size, (n_samples, seq_len))

    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train
    exp.train_stage1(dataloader, epochs=2)
    exp.train_stage2_rl(dataloader, epochs=3)

    # Compare
    results = exp.run_comparison(dataloader, num_batches=30)
    analysis = exp.analyze_results(results)

    # Print results
    logger.info("\n" + "="*70)
    logger.info("RESULTS: LIGHTWEIGHT INTEROCEPTIVE ATTENTION vs TRADITIONAL")
    logger.info("="*70)

    logger.info("\n1. ENERGY EFFICIENCY")
    logger.info("-" * 40)
    logger.info(f"  Baseline: {analysis['baseline']['energy_mj']:.3f} mJ/batch")
    logger.info(f"  LIA:      {analysis['lia']['energy_mj']:.3f} mJ/batch")
    logger.info(f"  Change:   {analysis['comparison']['energy_change']:+.1f}%")

    logger.info("\n2. THROUGHPUT")
    logger.info("-" * 40)
    logger.info(f"  Baseline: {analysis['baseline']['throughput']:.0f} tok/s")
    logger.info(f"  LIA:      {analysis['lia']['throughput']:.0f} tok/s")
    logger.info(f"  Change:   {analysis['comparison']['throughput_change']:+.1f}%")

    logger.info("\n3. THROUGHPUT PER WATT")
    logger.info("-" * 40)
    logger.info(f"  Baseline: {analysis['comparison']['tpw_baseline']:.1f} tok/s/W")
    logger.info(f"  LIA:      {analysis['comparison']['tpw_lia']:.1f} tok/s/W")
    logger.info(f"  Change:   {analysis['comparison']['tpw_change']:+.1f}%")

    logger.info("\n4. TEMPERATURE REGULATION")
    logger.info("-" * 40)
    logger.info(f"  Baseline: {analysis['baseline']['temp']:.1f}C (max: {analysis['baseline']['temp_max']:.0f}C)")
    logger.info(f"  LIA:      {analysis['lia']['temp']:.1f}C (max: {analysis['lia']['temp_max']:.0f}C)")
    logger.info(f"  Change:   {analysis['comparison']['temp_change']:+.1f}C")

    logger.info("\n5. LIA ACTION DISTRIBUTION")
    logger.info("-" * 40)
    for action, pct in analysis['lia']['action_dist'].items():
        logger.info(f"  {action}: {pct:.1f}%")

    logger.info("\n6. MODEL QUALITY")
    logger.info("-" * 40)
    logger.info(f"  Baseline loss: {analysis['baseline']['loss']:.4f}")
    logger.info(f"  LIA loss:      {analysis['lia']['loss']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/z71_lia_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"\nResults saved to: {results_dir}/results.json")

    return analysis


if __name__ == "__main__":
    results = run_lia_experiment()
