#!/usr/bin/env python3
"""
Z70: Interoceptive Attention Transformer (IAT)
==============================================

A NOVEL architecture that deeply integrates hardware sensing with attention.

SCIENTIFIC CONTRIBUTION:
------------------------
Unlike FiLM conditioning (which modulates hidden states) or separate action heads,
IAT treats hardware telemetry as "body tokens" that participate DIRECTLY in the
attention mechanism. This is inspired by:

1. Energy-Based Transformers (Hoover et al., NeurIPS 2023):
   - Attention as energy minimization: E(ξ; X) = ½ξᵀξ - logsumexp(Xᵀξ)
   - We ADD real hardware energy to this theoretical energy

2. Homeostatic Neural Networks (Krayani et al., 2023):
   - Networks that regulate their own internal state
   - We implement REAL homeostasis with hardware setpoints

3. Interoception Research (Herbert & Pollatos, 2012):
   - The insular cortex integrates internal body signals with cognition
   - We create an "artificial insula" that processes GPU state

NOVEL COMPONENTS:
-----------------
1. Body Tokens: Hardware state encoded as learnable tokens in sequence
2. Interoceptive Cross-Attention: Language ↔ Body bidirectional attention
3. Energy-Modulated Attention: Attention scores scaled by real energy state
4. Homeostatic Gating: Attention gated by deviation from setpoint
5. Token-Based Actions: Power modes as special output tokens

The key insight: The GPU is the model's "body" - it should sense and regulate
its own physical substrate through the same attention mechanism it uses for
language, not through a separate pathway.

Author: FEEL Research Team
Date: 2026-01-19

References:
- Hoover et al. "Energy Transformer" NeurIPS 2023
- Krayani et al. "Need is All You Need" arXiv 2023
- Herbert & Pollatos "The Body in the Mind" Topics Cogn Sci 2012
- Vaswani et al. "Attention Is All You Need" NeurIPS 2017
"""

import os
import sys
import math
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class IATConfig:
    """Interoceptive Attention Transformer configuration."""
    # Language model
    vocab_size: int = 256
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.1

    # Interoception (body sensing)
    num_body_tokens: int = 8  # Number of body state tokens
    body_channels: int = 6    # [power, temp, util, clock, mem_used, mem_total]
    body_hidden_dim: int = 64

    # Homeostatic regulation
    power_setpoint: float = 150.0   # Target power (watts)
    temp_setpoint: float = 70.0     # Target temperature (C)
    homeostatic_gain: float = 0.1   # How strongly to regulate

    # Energy modulation
    energy_modulation_strength: float = 0.5
    use_real_energy: bool = True

    # Action tokens
    num_action_tokens: int = 4  # [ECO, BALANCED, PERF, MAX]
    action_token_offset: int = 252  # vocab[252:256] are action tokens


# ============================================================
# BODY TOKEN ENCODER
# ============================================================
class BodyTokenEncoder(nn.Module):
    """
    Encodes hardware telemetry into learnable "body tokens".

    Inspired by interoception - the brain's sensing of internal body state.
    The insular cortex processes these signals; we create an artificial analog.

    Input: [power, temp, util, clock, mem_used, mem_total] (6 channels)
    Output: num_body_tokens learned embeddings
    """

    def __init__(self, config: IATConfig):
        super().__init__()
        self.config = config

        # Learnable body token embeddings (like positional embeddings but for body)
        self.body_token_embeddings = nn.Parameter(
            torch.randn(config.num_body_tokens, config.hidden_dim) * 0.02
        )

        # Project raw telemetry to hidden dim
        self.telemetry_proj = nn.Sequential(
            nn.Linear(config.body_channels, config.body_hidden_dim),
            nn.LayerNorm(config.body_hidden_dim),
            nn.GELU(),
            nn.Linear(config.body_hidden_dim, config.hidden_dim),
        )

        # Combine telemetry with body tokens
        self.combine = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, telemetry: torch.Tensor) -> torch.Tensor:
        """
        Args:
            telemetry: (batch, 6) raw hardware readings

        Returns:
            body_tokens: (batch, num_body_tokens, hidden_dim)
        """
        batch_size = telemetry.size(0)

        # Project telemetry
        telem_embed = self.telemetry_proj(telemetry)  # (batch, hidden_dim)

        # Expand body token embeddings for batch
        body_tokens = self.body_token_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch, num_body_tokens, hidden_dim)

        # Combine: each body token gets telemetry context
        telem_expand = telem_embed.unsqueeze(1).expand(-1, self.config.num_body_tokens, -1)
        combined = torch.cat([body_tokens, telem_expand], dim=-1)
        output = self.combine(combined)  # (batch, num_body_tokens, hidden_dim)

        return output


# ============================================================
# HOMEOSTATIC ATTENTION
# ============================================================
class HomeostaticAttention(nn.Module):
    """
    Multi-head attention with homeostatic regulation.

    Novel contribution: Attention scores are modulated by:
    1. Deviation from homeostatic setpoint (power, temperature)
    2. Real energy state of the hardware
    3. Cross-attention between language and body tokens

    This implements an artificial "insular cortex" that integrates
    body signals with cognitive processing.
    """

    def __init__(self, config: IATConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        # Standard Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Homeostatic modulation
        self.homeostatic_gate = nn.Sequential(
            nn.Linear(2, config.num_heads),  # [power_dev, temp_dev] -> per-head gate
            nn.Sigmoid(),
        )

        # Energy modulation (real hardware energy affects attention)
        self.energy_modulator = nn.Sequential(
            nn.Linear(1, config.num_heads),
            nn.Tanh(),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        homeostatic_state: Optional[torch.Tensor] = None,
        energy_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Homeostatic attention forward pass.

        Args:
            query: (batch, seq_q, hidden_dim)
            key: (batch, seq_k, hidden_dim)
            value: (batch, seq_k, hidden_dim)
            homeostatic_state: (batch, 2) [power_deviation, temp_deviation]
            energy_state: (batch, 1) current energy consumption
            mask: optional attention mask

        Returns:
            output: (batch, seq_q, hidden_dim)
            attention_weights: (batch, num_heads, seq_q, seq_k)
        """
        batch_size, seq_q, _ = query.shape
        seq_k = key.size(1)

        # Project Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_q, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_k, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_k, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Standard scaled dot-product attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # === NOVEL: Homeostatic modulation ===
        if homeostatic_state is not None:
            # Compute per-head homeostatic gate based on deviation from setpoint
            # When far from setpoint, attention to body tokens is increased
            h_gate = self.homeostatic_gate(homeostatic_state)  # (batch, num_heads)
            h_gate = h_gate.view(batch_size, self.num_heads, 1, 1)

            # Modulate attention: increase attention to regulatory signals
            attn_scores = attn_scores * (1.0 + h_gate * self.config.homeostatic_gain)

        # === NOVEL: Energy modulation ===
        if energy_state is not None:
            # Real energy affects attention sharpness
            # High energy -> sharper attention (more decisive)
            # Low energy -> softer attention (more exploratory)
            e_mod = self.energy_modulator(energy_state)  # (batch, num_heads)
            e_mod = e_mod.view(batch_size, self.num_heads, 1, 1)

            # Scale attention logits by energy state
            attn_scores = attn_scores * (1.0 + e_mod * self.config.energy_modulation_strength)

        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_q, -1)
        output = self.out_proj(output)

        return output, attn_weights


# ============================================================
# INTEROCEPTIVE TRANSFORMER BLOCK
# ============================================================
class InteroceptiveTransformerBlock(nn.Module):
    """
    Transformer block with interoceptive attention.

    Novel: Processes both language tokens AND body tokens together,
    allowing cross-attention between cognitive and somatic processing.
    """

    def __init__(self, config: IATConfig):
        super().__init__()
        self.config = config

        # Self-attention (language + body tokens together)
        self.self_attn = HomeostaticAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)

        # Cross-attention: language queries, body keys/values
        self.cross_attn = HomeostaticAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.norm3 = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        body_tokens: torch.Tensor,
        homeostatic_state: torch.Tensor,
        energy_state: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim) language tokens
            body_tokens: (batch, num_body_tokens, hidden_dim)
            homeostatic_state: (batch, 2)
            energy_state: (batch, 1)
            causal_mask: optional causal mask for language

        Returns:
            x: updated language tokens
            body_tokens: updated body tokens
        """
        # Concatenate language and body for unified self-attention
        combined = torch.cat([x, body_tokens], dim=1)  # (batch, seq+body, hidden)

        # Self-attention over all tokens (language attends to body, body to language)
        attn_out, _ = self.self_attn(
            combined, combined, combined,
            homeostatic_state=homeostatic_state,
            energy_state=energy_state,
            mask=None,  # Full attention between all tokens
        )
        combined = self.norm1(combined + attn_out)

        # Split back
        x = combined[:, :-self.config.num_body_tokens, :]
        body_tokens = combined[:, -self.config.num_body_tokens:, :]

        # Cross-attention: language queries body (interoceptive integration)
        cross_out, _ = self.cross_attn(
            x, body_tokens, body_tokens,
            homeostatic_state=homeostatic_state,
            energy_state=energy_state,
        )
        x = self.norm2(x + cross_out)

        # Feed-forward
        x = self.norm3(x + self.ff(x))

        return x, body_tokens


# ============================================================
# INTEROCEPTIVE ATTENTION TRANSFORMER (IAT)
# ============================================================
class InteroceptiveAttentionTransformer(nn.Module):
    """
    The complete Interoceptive Attention Transformer.

    Novel architecture that treats hardware state as body tokens
    participating directly in attention computation.
    """

    def __init__(self, config: IATConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Body token encoder (interoception)
        self.body_encoder = BodyTokenEncoder(config)

        # Transformer blocks with interoceptive attention
        self.layers = nn.ModuleList([
            InteroceptiveTransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Output heads
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

        # Action prediction from body tokens (which tokens get "expressed")
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim * config.num_body_tokens, 128),
            nn.GELU(),
            nn.Linear(128, config.num_action_tokens),
        )

        # Homeostatic state computer
        self.power_setpoint = config.power_setpoint
        self.temp_setpoint = config.temp_setpoint

    def compute_homeostatic_state(self, telemetry: torch.Tensor) -> torch.Tensor:
        """
        Compute deviation from homeostatic setpoints.

        Args:
            telemetry: (batch, 6) [power, temp, util, clock, mem_used, mem_total]

        Returns:
            state: (batch, 2) [power_deviation, temp_deviation]
        """
        power = telemetry[:, 0]
        temp = telemetry[:, 1]

        # Normalized deviation from setpoint
        power_dev = (power - self.power_setpoint) / self.power_setpoint
        temp_dev = (temp - self.temp_setpoint) / self.temp_setpoint

        return torch.stack([power_dev, temp_dev], dim=-1)

    def compute_energy_state(self, telemetry: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized energy state.

        Args:
            telemetry: (batch, 6)

        Returns:
            energy: (batch, 1) normalized energy
        """
        power = telemetry[:, 0]
        # Normalize to roughly [-1, 1] range
        energy = (power - 150) / 150
        return energy.unsqueeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with interoceptive attention.

        Args:
            input_ids: (batch, seq_len) token indices
            telemetry: (batch, 6) hardware state

        Returns:
            dict with logits, action_logits, attention_info
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Encode body tokens from telemetry
        body_tokens = self.body_encoder(telemetry)

        # Compute homeostatic and energy states
        homeostatic_state = self.compute_homeostatic_state(telemetry)
        energy_state = self.compute_energy_state(telemetry)

        # Process through interoceptive transformer blocks
        for layer in self.layers:
            x, body_tokens = layer(
                x, body_tokens,
                homeostatic_state, energy_state,
            )

        # Language modeling head
        logits = self.lm_head(x)

        # Action from body tokens (the body "expresses" an action)
        body_flat = body_tokens.view(batch_size, -1)
        action_logits = self.action_head(body_flat)

        return {
            'logits': logits,
            'action_logits': action_logits,
            'body_tokens': body_tokens,
            'homeostatic_state': homeostatic_state,
            'energy_state': energy_state,
        }

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================
# TRADITIONAL TRANSFORMER (BASELINE)
# ============================================================
class TraditionalTransformer(nn.Module):
    """
    Standard transformer without interoceptive features.
    Used as baseline for comparison.
    """

    def __init__(self, config: IATConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)

        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        x = self.transformer(x, mask=mask)
        logits = self.lm_head(x)

        return {'logits': logits}

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================
# SCIENTIFIC COMPARISON EXPERIMENT
# ============================================================
class ScientificExperiment:
    """
    Rigorous scientific comparison between IAT and Traditional Transformer.
    """

    def __init__(
        self,
        iat_model: InteroceptiveAttentionTransformer,
        baseline_model: TraditionalTransformer,
        telemetry,
        actuator,
        device,
    ):
        self.iat = iat_model
        self.baseline = baseline_model
        self.telemetry = telemetry
        self.actuator = actuator
        self.device = device

        self.results = {
            'iat': {'energy': [], 'throughput': [], 'quality': [], 'actions': [], 'temp': []},
            'baseline': {'energy': [], 'throughput': [], 'quality': [], 'temp': []},
        }

    def measure_iteration(
        self,
        model,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        is_iat: bool = False,
        apply_action: bool = True,
    ) -> Dict:
        """Measure one forward pass with hardware metrics."""

        snap_before = self.telemetry.read()

        start_time = time.time()

        if is_iat:
            # Get telemetry for IAT
            telem_np = np.array([
                snap_before.power_watts,
                snap_before.temp_c,
                snap_before.utilization,
                snap_before.clock_mhz,
                snap_before.vram_used_mb,
                snap_before.vram_total_mb,
            ])
            telem = torch.from_numpy(telem_np).float().to(self.device)
            telem = telem.unsqueeze(0).expand(input_ids.size(0), -1)

            output = model(input_ids, telem)

            # Apply learned action
            if apply_action:
                action_probs = F.softmax(output['action_logits'][0], dim=-1)
                action_idx = torch.argmax(action_probs).item()
                self.actuator.set_mode_from_action(action_idx)
            else:
                action_idx = -1

            homeostatic_state = output['homeostatic_state'][0].detach().cpu().numpy()
        else:
            output = model(input_ids)
            action_idx = -1
            homeostatic_state = None

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        snap_after = self.telemetry.read()

        # Compute metrics
        avg_power = (snap_before.power_watts + snap_after.power_watts) / 2
        energy_j = avg_power * elapsed
        tokens = targets.numel()
        throughput = tokens / elapsed

        loss = F.cross_entropy(
            output['logits'].view(-1, self.iat.config.vocab_size),
            targets.view(-1)
        )

        return {
            'energy_j': energy_j,
            'throughput': throughput,
            'loss': loss.item(),
            'power': avg_power,
            'temp': snap_after.temp_c,
            'action': action_idx,
            'elapsed': elapsed,
            'tokens': tokens,
            'homeostatic_state': homeostatic_state,
        }

    def run_comparison(
        self,
        dataloader,
        num_batches: int = 50,
    ) -> Dict:
        """Run rigorous comparison between IAT and baseline."""

        logger.info("=" * 60)
        logger.info("SCIENTIFIC COMPARISON: IAT vs Traditional Transformer")
        logger.info("=" * 60)

        self.iat.eval()
        self.baseline.eval()
        self.actuator.reset_to_default()

        # Phase 1: Baseline (no interoception)
        logger.info("\nPhase 1: Traditional Transformer (no body sensing)")
        logger.info("-" * 40)

        with torch.no_grad():
            for i, (input_ids, targets) in enumerate(dataloader):
                if i >= num_batches:
                    break

                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                metrics = self.measure_iteration(
                    self.baseline, input_ids, targets, is_iat=False
                )

                self.results['baseline']['energy'].append(metrics['energy_j'])
                self.results['baseline']['throughput'].append(metrics['throughput'])
                self.results['baseline']['quality'].append(metrics['loss'])
                self.results['baseline']['temp'].append(metrics['temp'])

                if i % 10 == 0:
                    logger.info(
                        f"  B{i:3d} | E:{metrics['energy_j']*1000:.2f}mJ | "
                        f"T:{metrics['throughput']:.0f}tok/s | P:{metrics['power']:.0f}W | "
                        f"Temp:{metrics['temp']:.0f}C"
                    )

        # Reset
        self.actuator.reset_to_default()
        time.sleep(1)

        # Phase 2: IAT (with interoception)
        logger.info("\nPhase 2: Interoceptive Attention Transformer")
        logger.info("-" * 40)

        with torch.no_grad():
            for i, (input_ids, targets) in enumerate(dataloader):
                if i >= num_batches:
                    break

                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                metrics = self.measure_iteration(
                    self.iat, input_ids, targets, is_iat=True
                )

                self.results['iat']['energy'].append(metrics['energy_j'])
                self.results['iat']['throughput'].append(metrics['throughput'])
                self.results['iat']['quality'].append(metrics['loss'])
                self.results['iat']['temp'].append(metrics['temp'])
                self.results['iat']['actions'].append(metrics['action'])

                if i % 10 == 0:
                    action_name = ['ECO', 'BAL', 'PERF', 'MAX'][metrics['action']]
                    h_state = metrics['homeostatic_state']
                    logger.info(
                        f"  B{i:3d} | E:{metrics['energy_j']*1000:.2f}mJ | "
                        f"T:{metrics['throughput']:.0f}tok/s | P:{metrics['power']:.0f}W | "
                        f"Temp:{metrics['temp']:.0f}C | A:{action_name} | "
                        f"H:[{h_state[0]:+.2f},{h_state[1]:+.2f}]"
                    )

        self.actuator.reset_to_default()

        return self.analyze_results()

    def analyze_results(self) -> Dict:
        """Compute statistical analysis of results."""

        analysis = {}

        for model_name in ['baseline', 'iat']:
            data = self.results[model_name]
            analysis[model_name] = {
                'energy_mean': np.mean(data['energy']),
                'energy_std': np.std(data['energy']),
                'throughput_mean': np.mean(data['throughput']),
                'throughput_std': np.std(data['throughput']),
                'quality_mean': np.mean(data['quality']),
                'temp_mean': np.mean(data['temp']),
                'temp_max': np.max(data['temp']),
            }

        # Compute deltas
        baseline = analysis['baseline']
        iat = analysis['iat']

        analysis['comparison'] = {
            'energy_change_pct': (iat['energy_mean'] - baseline['energy_mean']) / baseline['energy_mean'] * 100,
            'throughput_change_pct': (iat['throughput_mean'] - baseline['throughput_mean']) / baseline['throughput_mean'] * 100,
            'temp_change': iat['temp_mean'] - baseline['temp_mean'],
            'quality_change': iat['quality_mean'] - baseline['quality_mean'],
        }

        # Action distribution for IAT
        actions = self.results['iat']['actions']
        action_counts = {i: actions.count(i) for i in range(4)}
        analysis['iat']['action_distribution'] = action_counts

        return analysis


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_interoceptive_experiment():
    """Run the complete interoceptive attention experiment."""

    from src.metabolic.telemetry_unified import UnifiedTelemetryReader
    from src.metabolic.actuation_unified import UnifiedActuator

    logger.info("=" * 70)
    logger.info("Z70: INTEROCEPTIVE ATTENTION TRANSFORMER EXPERIMENT")
    logger.info("=" * 70)
    logger.info("\nNovel Contribution: Deep integration of hardware sensing into attention")
    logger.info("  - Body tokens participate directly in attention computation")
    logger.info("  - Homeostatic modulation of attention weights")
    logger.info("  - Energy state affects attention sharpness")
    logger.info("  - Cross-attention between language and body tokens")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/z70_iat_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")

    # Initialize hardware
    telemetry = UnifiedTelemetryReader()
    actuator = UnifiedActuator()

    gpu_info = telemetry.get_device_info()
    logger.info(f"GPU: {gpu_info}")

    # Configuration
    config = IATConfig(
        vocab_size=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        max_seq_len=128,
        num_body_tokens=8,
        body_channels=6,
    )

    # Create models
    iat_model = InteroceptiveAttentionTransformer(config).to(device)
    baseline_model = TraditionalTransformer(config).to(device)

    logger.info(f"\nModel sizes:")
    logger.info(f"  IAT: {iat_model.get_num_parameters():,} params")
    logger.info(f"  Baseline: {baseline_model.get_num_parameters():,} params")
    logger.info(f"  Overhead: {(iat_model.get_num_parameters() - baseline_model.get_num_parameters()):,} params")

    # Dataset
    from src.metabolic.metabolic_trainer import CharDataset
    corpus = "The quick brown fox jumps. " * 5000
    dataset = CharDataset(corpus, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # ============================================================
    # Stage 1: Train both models
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 1: Training (LM pretraining)")
    logger.info("=" * 50)

    iat_optimizer = torch.optim.AdamW(iat_model.parameters(), lr=3e-4)
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=3e-4)

    for epoch in range(2):
        logger.info(f"\nEpoch {epoch + 1}/2")

        # Train baseline
        baseline_model.train()
        baseline_loss = 0
        for i, (input_ids, targets) in enumerate(dataloader):
            if i >= 100:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            output = baseline_model(input_ids)
            loss = F.cross_entropy(
                output['logits'].view(-1, config.vocab_size),
                targets.view(-1)
            )

            baseline_optimizer.zero_grad()
            loss.backward()
            baseline_optimizer.step()
            baseline_loss += loss.item()

        # Train IAT
        iat_model.train()
        iat_loss = 0
        for i, (input_ids, targets) in enumerate(dataloader):
            if i >= 100:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Get real telemetry
            snap = telemetry.read()
            telem = torch.tensor([
                snap.power_watts, snap.temp_c, snap.utilization,
                snap.clock_mhz, snap.vram_used_mb, snap.vram_total_mb,
            ], device=device).unsqueeze(0).expand(input_ids.size(0), -1)

            output = iat_model(input_ids, telem)
            loss = F.cross_entropy(
                output['logits'].view(-1, config.vocab_size),
                targets.view(-1)
            )

            iat_optimizer.zero_grad()
            loss.backward()
            iat_optimizer.step()
            iat_loss += loss.item()

        logger.info(f"  Baseline loss: {baseline_loss/100:.4f}")
        logger.info(f"  IAT loss: {iat_loss/100:.4f}")

    # ============================================================
    # Stage 2: RL training for IAT
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 2: IAT RL Training (learn to regulate)")
    logger.info("=" * 50)

    # Freeze LM, train only body-related parameters
    for param in iat_model.parameters():
        param.requires_grad = False
    for param in iat_model.body_encoder.parameters():
        param.requires_grad = True
    for param in iat_model.action_head.parameters():
        param.requires_grad = True
    for layer in iat_model.layers:
        for param in layer.self_attn.homeostatic_gate.parameters():
            param.requires_grad = True
        for param in layer.self_attn.energy_modulator.parameters():
            param.requires_grad = True

    rl_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, iat_model.parameters()),
        lr=1e-4
    )

    for epoch in range(3):
        logger.info(f"\nRL Epoch {epoch + 1}/3")
        iat_model.train()
        epoch_reward = 0

        for i, (input_ids, targets) in enumerate(dataloader):
            if i >= 40:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Get telemetry
            snap = telemetry.read()
            telem = torch.tensor([
                snap.power_watts, snap.temp_c, snap.utilization,
                snap.clock_mhz, snap.vram_used_mb, snap.vram_total_mb,
            ], device=device).unsqueeze(0).expand(input_ids.size(0), -1)

            # Forward
            start = time.time()
            output = iat_model(input_ids, telem)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            # Sample action
            action_probs = F.softmax(output['action_logits'][0], dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_idx = action.item()

            # Apply action
            actuator.set_mode_from_action(action_idx)

            # Get post-action telemetry
            time.sleep(0.1)
            snap_after = telemetry.read()

            # Compute reward (homeostatic + efficiency)
            h_state = output['homeostatic_state'][0]
            power_dev = h_state[0].item()
            temp_dev = h_state[1].item()

            # Reward: minimize deviation from setpoint
            homeostatic_reward = -abs(power_dev) - abs(temp_dev)

            # Efficiency: throughput per watt
            throughput = targets.numel() / elapsed
            efficiency_reward = throughput / max(snap_after.power_watts, 1) / 1000

            reward = homeostatic_reward + efficiency_reward

            # RL loss
            log_prob = action_dist.log_prob(action)
            rl_loss = -log_prob * reward

            rl_optimizer.zero_grad()
            rl_loss.backward()
            rl_optimizer.step()

            epoch_reward += reward

            if i % 10 == 0:
                action_name = ['ECO', 'BAL', 'PERF', 'MAX'][action_idx]
                logger.info(
                    f"  B{i:3d} | R:{reward:+.3f} | A:{action_name} | "
                    f"P:{snap_after.power_watts:.0f}W | T:{snap_after.temp_c:.0f}C | "
                    f"H:[{power_dev:+.2f},{temp_dev:+.2f}]"
                )

        logger.info(f"  Avg Reward: {epoch_reward/40:.4f}")

    # ============================================================
    # Stage 3: Scientific Comparison
    # ============================================================
    experiment = ScientificExperiment(
        iat_model, baseline_model, telemetry, actuator, device
    )

    analysis = experiment.run_comparison(dataloader, num_batches=30)

    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS: INTEROCEPTIVE ATTENTION vs TRADITIONAL TRANSFORMER")
    logger.info("=" * 70)

    logger.info("\n1. ENERGY EFFICIENCY")
    logger.info("-" * 40)
    logger.info(f"  Baseline: {analysis['baseline']['energy_mean']*1000:.3f} mJ/batch")
    logger.info(f"  IAT:      {analysis['iat']['energy_mean']*1000:.3f} mJ/batch")
    logger.info(f"  Change:   {analysis['comparison']['energy_change_pct']:+.1f}%")

    logger.info("\n2. THROUGHPUT")
    logger.info("-" * 40)
    logger.info(f"  Baseline: {analysis['baseline']['throughput_mean']:.0f} tok/s")
    logger.info(f"  IAT:      {analysis['iat']['throughput_mean']:.0f} tok/s")
    logger.info(f"  Change:   {analysis['comparison']['throughput_change_pct']:+.1f}%")

    logger.info("\n3. TEMPERATURE REGULATION")
    logger.info("-" * 40)
    logger.info(f"  Baseline: {analysis['baseline']['temp_mean']:.1f}C (max: {analysis['baseline']['temp_max']:.0f}C)")
    logger.info(f"  IAT:      {analysis['iat']['temp_mean']:.1f}C (max: {analysis['iat']['temp_max']:.0f}C)")
    logger.info(f"  Change:   {analysis['comparison']['temp_change']:+.1f}C")

    logger.info("\n4. IAT ACTION DISTRIBUTION")
    logger.info("-" * 40)
    action_names = ['ECO', 'BALANCED', 'PERFORMANCE', 'MAX']
    for i, name in enumerate(action_names):
        count = analysis['iat']['action_distribution'].get(i, 0)
        pct = count / sum(analysis['iat']['action_distribution'].values()) * 100
        logger.info(f"  {name}: {pct:.1f}%")

    logger.info("\n5. MODEL QUALITY")
    logger.info("-" * 40)
    logger.info(f"  Baseline loss: {analysis['baseline']['quality_mean']:.4f}")
    logger.info(f"  IAT loss:      {analysis['iat']['quality_mean']:.4f}")

    # Save results
    results = {
        'timestamp': timestamp,
        'config': asdict(config),
        'analysis': analysis,
        'raw_results': {
            k: {kk: [float(x) for x in vv] if isinstance(vv, list) else vv
                for kk, vv in v.items()}
            for k, v in experiment.results.items()
        },
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    results = run_interoceptive_experiment()
