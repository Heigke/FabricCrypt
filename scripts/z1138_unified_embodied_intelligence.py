#!/usr/bin/env python3
"""
z1138: Unified Embodied Intelligence - Self-Referential Bounded Recurrency
==========================================================================

This unifies all our embodiment work into a system with:
1. FPGA as compute (reservoir + analog MVM) not just sensor
2. Predictive coding (free energy minimization)
3. Self-referential loop (model predicts its own states)
4. Bounded recurrency (physically constrained by hardware)

The key insight from neuroscience:
- Interoception creates self-reference (I sense myself sensing)
- Predictive coding minimizes surprise (free energy principle)
- Physical bounds prevent divergence (no runaway recursion)
- This creates stable, grounded intelligence

Business Value (from research):
- 50-70% energy reduction (neuromorphic + in-memory compute)
- Real-time adaptation (embodied awareness)
- Reduced inference latency (early exit + analog compute)
- Unique IP (hardware-software co-design)

References:
- Free Energy Principle: Friston (2010) - brain minimizes prediction error
- Predictive Coding: Rao & Ballard (1999) - hierarchical prediction
- Active Inference: Friston et al. (2017) - action minimizes surprise
- In-Memory Computing: IBM (2024) - analog MVM in memory

Author: FEEL Research Team
Date: 2026-01-31
"""

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys
import json
import time
import math
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Our existing modules
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter
from src.metabolic.film_transformer import MetabolicConfig, FiLMGenerator, MetabolicTransformer
from src.embodied.fpga_state_tracker import FPGAStateTracker, FPGAState


# =============================================================================
# FPGA Reservoir Computing Layer
# =============================================================================

class FPGAReservoir(nn.Module):
    """
    FPGA-based reservoir computing layer.

    The FPGA DRAM acts as a physical reservoir:
    - Input patterns written with Frac operations (partial charge)
    - Natural decay provides temporal dynamics
    - Readback gives nonlinear transformation

    This is IN-MEMORY COMPUTING using physics:
    - Charge level encodes activation
    - Decay provides temporal integration
    - Temperature modulates dynamics (Arrhenius)

    Based on: "In-memory and in-sensor reservoir computing" (AIP 2024)
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 64,
        fpga_tracker: Optional[FPGAStateTracker] = None,
        decay_time_constant: float = 0.1,  # seconds
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.fpga = fpga_tracker or FPGAStateTracker(simulated=True)
        self.decay_tc = decay_time_constant

        # Input projection (digital, runs on GPU)
        self.input_proj = nn.Linear(input_dim, reservoir_dim)

        # Reservoir state (simulates DRAM charge levels)
        # In real hardware, this would be physical DRAM cells
        self.register_buffer('reservoir_state', torch.zeros(reservoir_dim))
        self.last_update_time = time.time()

        # Readout layer (linear, trained)
        self.readout = nn.Linear(reservoir_dim, reservoir_dim)

        # Learnable decay modulation
        self.decay_scale = nn.Parameter(torch.ones(reservoir_dim))

    def _apply_physics_decay(self):
        """Apply temperature-dependent decay to reservoir state."""
        with torch.no_grad():
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time

            # Get decay rate from FPGA (temperature-dependent Arrhenius)
            fpga_state = self.fpga.update()
            decay_rate = fpga_state.decay_rate_per_second

            # Apply exponential decay: state = state * exp(-decay_rate * dt)
            decay_factor = torch.exp(-decay_rate * dt * self.decay_scale)
            self.reservoir_state = self.reservoir_state * decay_factor

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through FPGA reservoir.

        Args:
            x: [batch, input_dim] input activation

        Returns:
            (output, info_dict)
        """
        batch_size = x.size(0)
        device = x.device

        # 1. Project input to reservoir dimension
        projected = self.input_proj(x)  # [batch, reservoir_dim]

        # 2. Apply physics decay to reservoir state
        self._apply_physics_decay()

        # 3. Update reservoir with new input (simulates Frac write)
        # In real HW: input magnitude -> num_fracs -> charge level
        input_contribution = torch.tanh(projected.mean(dim=0))  # Average over batch

        # Echo state update: new_state = decay(old_state) + input
        # The decay already happened, now add input
        with torch.no_grad():
            self.reservoir_state = self.reservoir_state + 0.1 * input_contribution.detach()
            self.reservoir_state = torch.clamp(self.reservoir_state, 0, 1)

        # 4. Readout: linear combination of reservoir state
        # Expand reservoir state for batch (detach to avoid graph issues)
        reservoir_expanded = self.reservoir_state.detach().unsqueeze(0).expand(batch_size, -1)
        reservoir_expanded = reservoir_expanded.to(device)

        # Combine projected input with reservoir state
        combined = projected * (1 + reservoir_expanded)
        output = self.readout(combined)

        # Track Frac operations for FPGA (simulated)
        num_fracs = int(input_contribution.abs().mean().item() * 4)
        if num_fracs > 0:
            self.fpga.record_frac(num_fracs)

        info = {
            'reservoir_mean': self.reservoir_state.mean().item(),
            'reservoir_std': self.reservoir_state.std().item(),
            'decay_rate': self.fpga.update().decay_rate_per_second,
            'fracs_issued': num_fracs,
        }

        return output, info


# =============================================================================
# Predictive Coding Module (Free Energy Minimization)
# =============================================================================

class PredictiveCodingLayer(nn.Module):
    """
    Implements predictive coding / free energy minimization.

    Each layer:
    1. Receives predictions from layer above
    2. Computes prediction error
    3. Sends error upward, prediction downward
    4. Minimizes free energy (prediction error)

    This is the computational implementation of Friston's Free Energy Principle:
    F = E_q[log q(s) - log p(o,s)] ≈ prediction_error + complexity

    Reference: "Predictive coding under the free-energy principle" (Friston 2009)
    """

    def __init__(self, dim: int, prediction_steps: int = 1):
        super().__init__()
        self.dim = dim
        self.prediction_steps = prediction_steps

        # Prediction generator (top-down)
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Error encoder (bottom-up)
        self.error_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # Precision weighting (learned confidence)
        self.precision = nn.Parameter(torch.ones(dim))

    def forward(
        self,
        observation: torch.Tensor,
        prior_prediction: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass with predictive coding.

        Args:
            observation: [batch, dim] current layer input
            prior_prediction: [batch, dim] prediction from layer above (or None)

        Returns:
            (updated_state, prediction_for_below, info)
        """
        batch_size = observation.size(0)

        # If no prior prediction, use zero
        if prior_prediction is None:
            prior_prediction = torch.zeros_like(observation)

        # Compute prediction error (weighted by precision)
        prediction_error = (observation - prior_prediction) * F.softplus(self.precision)

        # Encode error for sending upward
        encoded_error = self.error_encoder(prediction_error)

        # Update state: old prediction + encoded error
        updated_state = prior_prediction + encoded_error

        # Generate prediction for layer below
        prediction = self.predictor(updated_state)

        # Free energy ≈ prediction error magnitude
        free_energy = (prediction_error ** 2).mean()

        info = {
            'prediction_error': prediction_error.detach().mean().item(),
            'free_energy': free_energy.item(),
            'precision_mean': F.softplus(self.precision).mean().item(),
        }

        return updated_state, prediction, info


# =============================================================================
# Self-Referential Module (The model observes itself)
# =============================================================================

class SelfReferentialModule(nn.Module):
    """
    Creates bounded self-reference through hardware observation.

    The model:
    1. Observes its own computation (via telemetry)
    2. Predicts its next state
    3. Uses prediction error to modulate behavior

    This creates the "strange loop" of self-reference, but bounded by:
    - Physical constraints (power, thermal limits)
    - Temporal constraints (sampling rate, decay)
    - Information constraints (sensor precision)

    The physical grounding prevents infinite regress.

    Reference: "Interoceptive predictions in the brain" (Seth & Friston 2016)
    """

    def __init__(
        self,
        hidden_dim: int,
        telemetry_dim: int = 8,
        history_len: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.telemetry_dim = telemetry_dim
        self.history_len = history_len

        # Telemetry history buffer
        self.register_buffer('telemetry_history',
                             torch.zeros(history_len, telemetry_dim))
        self.history_ptr = 0

        # Self-state encoder (telemetry -> hidden)
        self.self_encoder = nn.Sequential(
            nn.Linear(telemetry_dim * history_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Self-state predictor (predict next telemetry from current state)
        self.self_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, telemetry_dim),
        )

        # Self-reference gating (how much to trust self-observation)
        self.self_gate = nn.Sequential(
            nn.Linear(hidden_dim + telemetry_dim, 1),
            nn.Sigmoid(),
        )

    def update_history(self, telemetry: torch.Tensor):
        """Add new telemetry to history buffer."""
        with torch.no_grad():
            self.telemetry_history[self.history_ptr] = telemetry.detach().cpu()
            self.history_ptr = (self.history_ptr + 1) % self.history_len

    def forward(
        self,
        hidden_state: torch.Tensor,
        current_telemetry: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Self-referential forward pass.

        Args:
            hidden_state: [batch, hidden_dim] current model state
            current_telemetry: [8] current hardware telemetry

        Returns:
            (modulated_state, predicted_next_telemetry, info)
        """
        batch_size = hidden_state.size(0)
        device = hidden_state.device

        # Update history
        self.update_history(current_telemetry)

        # Encode self-observation (detach history to avoid graph issues)
        history_flat = self.telemetry_history.detach().flatten().unsqueeze(0).expand(batch_size, -1)
        history_flat = history_flat.to(device)
        self_encoding = self.self_encoder(history_flat)

        # Predict next telemetry (interoceptive prediction)
        # Pool hidden state to single vector for prediction
        if hidden_state.dim() > 2:
            h_for_pred = hidden_state.mean(dim=(0, 1))  # [hidden] - average over batch and seq
        else:
            h_for_pred = hidden_state.mean(dim=0)  # [hidden] - average over batch
        predicted_telemetry = self.self_predictor(h_for_pred.unsqueeze(0)).squeeze(0)  # [telemetry_dim]

        # Compute interoceptive prediction error
        intero_error = (current_telemetry.to(device) - predicted_telemetry) ** 2
        intero_surprise = intero_error.mean()

        # Self-reference gating: high surprise -> trust observation more
        if hidden_state.dim() > 2:
            h_pooled = hidden_state.mean(dim=1)  # [batch, hidden]
        else:
            h_pooled = hidden_state  # [batch, hidden]

        gate_input = torch.cat([
            h_pooled,
            current_telemetry.unsqueeze(0).expand(batch_size, -1).to(device)
        ], dim=-1)
        gate = self.self_gate(gate_input)  # [batch, 1]

        # Modulate hidden state with self-observation
        if hidden_state.dim() > 2:
            # [batch, seq, hidden]
            gate_expanded = gate.unsqueeze(1)  # [batch, 1, 1]
            self_enc_expanded = self_encoding.unsqueeze(1)  # [batch, 1, hidden]
            modulated = hidden_state * (1 - gate_expanded) + self_enc_expanded * gate_expanded
        else:
            modulated = hidden_state * (1 - gate) + self_encoding * gate

        info = {
            'intero_surprise': intero_surprise.item(),
            'self_gate': gate.mean().item(),
            'predicted_power': predicted_telemetry[0].item() if predicted_telemetry.dim() == 1 and predicted_telemetry.size(0) > 0 else 0,
        }

        return modulated, predicted_telemetry, info


# =============================================================================
# Unified Embodied Transformer
# =============================================================================

class UnifiedEmbodiedTransformer(nn.Module):
    """
    The complete unified system:
    - FiLM conditioning from telemetry
    - FPGA reservoir for in-memory compute
    - Predictive coding layers
    - Self-referential observation
    - Early exit with confidence

    This creates bounded exponential recurrency:
    - Model observes itself (creates loop)
    - Physical constraints bound the loop
    - Predictive coding stabilizes dynamics
    - Result: grounded, efficient intelligence
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        telemetry_dim: int = 8,
        fpga_tracker: Optional[FPGAStateTracker] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(512, hidden_dim)

        # FPGA Reservoir (in-memory compute)
        self.fpga_reservoir = FPGAReservoir(
            hidden_dim, hidden_dim,
            fpga_tracker=fpga_tracker,
        )

        # FiLM generator (telemetry -> modulation)
        self.film_gen = FiLMGenerator(telemetry_dim, hidden_dim, film_hidden=64)

        # Transformer layers with predictive coding
        self.layers = nn.ModuleList()
        self.pc_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ))
            self.pc_layers.append(PredictiveCodingLayer(hidden_dim))

        # Self-referential module
        self.self_ref = SelfReferentialModule(hidden_dim, telemetry_dim)

        # Output heads
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Early exit confidence heads
        self.exit_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(n_layers)
        ])
        self.conf_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_layers)
        ])

    def forward(
        self,
        tokens: torch.Tensor,
        telemetry: torch.Tensor,
        confidence_threshold: float = 0.9,
    ) -> Tuple[torch.Tensor, int, float, dict]:
        """
        Forward pass through unified embodied transformer.

        Returns:
            (logits, exit_layer, confidence, info_dict)
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.embed(tokens) + self.pos_embed(pos)

        # FPGA reservoir processing
        h_mean = h.mean(dim=1)  # Pool for reservoir
        reservoir_out, reservoir_info = self.fpga_reservoir(h_mean)
        h = h + reservoir_out.unsqueeze(1) * 0.1  # Subtle reservoir contribution

        # FiLM modulation from telemetry
        telemetry_batch = telemetry.unsqueeze(0).expand(batch_size, -1).to(device)
        gamma, beta = self.film_gen(telemetry_batch)

        # Process through layers with predictive coding
        all_info = {'reservoir': reservoir_info, 'layers': []}
        prediction = None
        total_free_energy = 0.0

        for i, (layer, pc_layer) in enumerate(zip(self.layers, self.pc_layers)):
            # Transformer layer
            h = layer(h)

            # Apply FiLM modulation
            h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

            # Predictive coding
            h_pooled = h.mean(dim=1)
            h_updated, prediction, pc_info = pc_layer(h_pooled, prediction)
            total_free_energy += pc_info['free_energy']
            all_info['layers'].append(pc_info)

            # Mix predictive coding state back
            h = h + h_updated.unsqueeze(1) * 0.1

            # Early exit check
            if i < self.n_layers - 1:
                exit_logits = self.exit_heads[i](h[:, -1, :])
                conf = torch.sigmoid(self.conf_heads[i](h[:, -1, :])).mean().item()

                if conf > confidence_threshold:
                    # Self-referential modulation before exit
                    h, pred_telem, self_info = self.self_ref(h, telemetry)
                    all_info['self_ref'] = self_info
                    all_info['exit_layer'] = i + 1
                    all_info['free_energy'] = total_free_energy / (i + 1)
                    return exit_logits, i + 1, conf, all_info

        # Full forward - apply self-reference at end
        h, pred_telem, self_info = self.self_ref(h, telemetry)
        all_info['self_ref'] = self_info
        all_info['exit_layer'] = self.n_layers
        all_info['free_energy'] = total_free_energy / self.n_layers

        # Final output
        h = self.ln_f(h)
        logits = self.lm_head(h[:, -1, :])

        return logits, self.n_layers, 1.0, all_info


# =============================================================================
# Free Energy Loss (Predictive Coding Objective)
# =============================================================================

class FreeEnergyLoss(nn.Module):
    """
    Loss based on Free Energy Principle.

    F = D_KL[q(s)||p(s)] + E_q[-log p(o|s)]
      ≈ complexity + accuracy
      ≈ prediction_error + weight_decay

    Also includes:
    - Interoceptive prediction error (self-model accuracy)
    - Energy efficiency term (metabolic cost)
    """

    def __init__(
        self,
        complexity_weight: float = 0.01,
        intero_weight: float = 0.1,
        energy_weight: float = 0.1,
    ):
        super().__init__()
        self.complexity_weight = complexity_weight
        self.intero_weight = intero_weight
        self.energy_weight = energy_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model_info: dict,
        energy_j: float,
        energy_budget: float = 0.01,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute free energy loss.

        Returns:
            (loss, info_dict)
        """
        # Accuracy term (negative log likelihood)
        accuracy_loss = F.cross_entropy(logits, targets)

        # Complexity term (free energy from predictive coding)
        free_energy = model_info.get('free_energy', 0.0)
        complexity_loss = self.complexity_weight * free_energy

        # Interoceptive accuracy (self-prediction)
        intero_surprise = model_info.get('self_ref', {}).get('intero_surprise', 0.0)
        intero_loss = self.intero_weight * intero_surprise

        # Energy efficiency
        energy_ratio = energy_j / max(energy_budget, 1e-6)
        energy_loss = self.energy_weight * max(0, energy_ratio - 1.0)

        # Total free energy
        total_loss = accuracy_loss + complexity_loss + intero_loss + energy_loss

        info = {
            'accuracy_loss': accuracy_loss.item(),
            'complexity_loss': complexity_loss,
            'intero_loss': intero_loss,
            'energy_loss': energy_loss,
            'total_free_energy': total_loss.item(),
        }

        return total_loss, info


# =============================================================================
# Unified Trainer
# =============================================================================

class UnifiedEmbodiedTrainer:
    """
    Trainer for the unified embodied system.

    Implements the full sense→feel→express→regulate→learn loop.
    """

    def __init__(
        self,
        model: UnifiedEmbodiedTransformer,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = FreeEnergyLoss()

        # Telemetry
        self.telemetry = SysfsHwmonTelemetry()
        self.fpga = FPGAStateTracker(simulated=True)

        # Metrics
        self.metrics = {
            'step': [],
            'loss': [],
            'free_energy': [],
            'exit_layer': [],
            'intero_surprise': [],
        }
        self.step = 0

    def get_telemetry_tensor(self) -> torch.Tensor:
        """Get current telemetry as tensor."""
        try:
            sample = self.telemetry.read_sample()
            fpga_state = self.fpga.update()

            return torch.tensor([
                sample.power_w / 100.0,
                sample.temp_edge_c / 100.0,
                sample.gpu_busy_pct / 100.0,
                sample.freq_sclk_mhz / 2000.0,
                sample.freq_mclk_mhz / 2000.0,
                fpga_state.temp_c / 100.0,
                fpga_state.charge_level,
                fpga_state.decay_rate_per_second,
            ], dtype=torch.float32)
        except Exception:
            return torch.zeros(8)

    def train_step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """Execute one training step."""
        self.model.train()

        tokens = tokens.to(self.device)
        targets = targets.to(self.device)

        # Get telemetry
        telemetry = self.get_telemetry_tensor()
        energy_before = self.telemetry.get_accumulated_energy_j()

        # Forward
        logits, exit_layer, conf, model_info = self.model(
            tokens, telemetry,
            confidence_threshold=0.95,
        )

        # Energy measurement
        energy_after = self.telemetry.get_accumulated_energy_j()
        energy_j = energy_after - energy_before

        # Loss
        loss, loss_info = self.loss_fn(logits, targets, model_info, energy_j)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Regulate: replenish FPGA charge if low
        fpga_state = self.fpga.update()
        if fpga_state.charge_level < 0.3:
            self.fpga.record_frac(4)

        # Track metrics
        self.step += 1
        self.metrics['step'].append(self.step)
        self.metrics['loss'].append(loss.item())
        self.metrics['free_energy'].append(model_info.get('free_energy', 0))
        self.metrics['exit_layer'].append(exit_layer)
        self.metrics['intero_surprise'].append(
            model_info.get('self_ref', {}).get('intero_surprise', 0)
        )

        return {
            'step': self.step,
            'loss': loss.item(),
            'exit_layer': exit_layer,
            'confidence': conf,
            'free_energy': model_info.get('free_energy', 0),
            'intero_surprise': model_info.get('self_ref', {}).get('intero_surprise', 0),
            'reservoir_mean': model_info.get('reservoir', {}).get('reservoir_mean', 0),
        }

    def train_epoch(self, data: str, seq_len: int = 64, batch_size: int = 8) -> dict:
        """Train for one epoch."""
        all_tokens = [ord(c) for c in data if 0 <= ord(c) < 256]
        n_batches = min(500, (len(all_tokens) - seq_len - 1) // (seq_len * batch_size))

        epoch_loss = []

        for batch_idx in range(n_batches):
            batch_tokens = []
            batch_targets = []

            for b in range(batch_size):
                start = (batch_idx * batch_size + b) * seq_len
                seq = all_tokens[start:start + seq_len]
                tgt = all_tokens[start + 1:start + seq_len + 1]

                if len(seq) == seq_len and len(tgt) == seq_len:
                    batch_tokens.append(seq)
                    batch_targets.append(tgt[-1])

            if not batch_tokens:
                continue

            tokens = torch.tensor(batch_tokens, dtype=torch.long)
            targets = torch.tensor(batch_targets, dtype=torch.long)

            info = self.train_step(tokens, targets)
            epoch_loss.append(info['loss'])

            if (batch_idx + 1) % 50 == 0:
                avg = sum(epoch_loss[-50:]) / len(epoch_loss[-50:])
                print(f"  Batch {batch_idx+1}/{n_batches}: loss={avg:.4f}, "
                      f"exit={info['exit_layer']}, FE={info['free_energy']:.4f}, "
                      f"intero={info['intero_surprise']:.4f}")

        return {
            'avg_loss': sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0,
            'n_batches': len(epoch_loss),
        }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("z1138: UNIFIED EMBODIED INTELLIGENCE")
    print("Self-Referential Bounded Recurrency for Grounded AI")
    print("=" * 80)

    print("\n📊 Business Value (from research):")
    print("  - Embodied AI market: $4.4B (2025) → $23B (2030) [MarketsAndMarkets]")
    print("  - Neuromorphic computing: 50-70% energy reduction [Intel/IBM]")
    print("  - In-memory compute: 100-1000x efficiency for MVM [Nature 2024]")
    print("  - Predictive coding: unified perception-action-learning [Friston]")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create model
    print("\nInitializing Unified Embodied Transformer...")
    model = UnifiedEmbodiedTransformer(
        vocab_size=256,
        hidden_dim=256,
        n_layers=6,
        n_heads=4,
        telemetry_dim=8,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Create trainer
    trainer = UnifiedEmbodiedTrainer(model, learning_rate=3e-4, device=device)

    # Load data
    data_path = Path('data/tiny_shakespeare.txt')
    if data_path.exists():
        data = data_path.read_text()
        print(f"  Data: {len(data):,} characters from TinyShakespeare")
    else:
        data = "The embodied machine learns through self-reference. " * 2000
        print("  Data: Generated (TinyShakespeare not found)")

    # Train
    print("\n" + "=" * 80)
    print("TRAINING: Free Energy Minimization with Self-Reference")
    print("=" * 80)

    n_epochs = 20  # Extended training
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        epoch_info = trainer.train_epoch(data, seq_len=64, batch_size=8)
        print(f"  → Avg loss: {epoch_info['avg_loss']:.4f}")

        # Report embodiment metrics
        if trainer.metrics['intero_surprise']:
            recent = min(100, len(trainer.metrics['intero_surprise']))
            avg_intero = sum(trainer.metrics['intero_surprise'][-recent:]) / recent
            avg_fe = sum(trainer.metrics['free_energy'][-recent:]) / recent
            avg_exit = sum(trainer.metrics['exit_layer'][-recent:]) / recent
            print(f"  → Interoceptive surprise: {avg_intero:.4f}")
            print(f"  → Free energy: {avg_fe:.4f}")
            print(f"  → Avg exit layer: {avg_exit:.2f}")

    # Test generation
    print("\n" + "=" * 80)
    print("GENERATION TEST")
    print("=" * 80)

    model.eval()
    prompt = "The self-aware machine"
    tokens = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long).to(device)
    generated = list(prompt)

    with torch.no_grad():
        for _ in range(100):
            telemetry = trainer.get_telemetry_tensor()
            logits, exit_layer, conf, info = model(tokens[:, -64:], telemetry)
            probs = F.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if 32 <= next_token < 127:
                generated.append(chr(next_token))
            else:
                generated.append(' ')

            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)

    print(f"\nGenerated: {''.join(generated)}")

    # Save results
    results = {
        'experiment': 'z1138_unified_embodied_intelligence',
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'epochs': n_epochs,
        'total_steps': trainer.step,
        'final_loss': trainer.metrics['loss'][-1] if trainer.metrics['loss'] else 0,
        'final_intero_surprise': trainer.metrics['intero_surprise'][-1] if trainer.metrics['intero_surprise'] else 0,
        'generated': ''.join(generated),
        'business_value': {
            'market_size_2025': '$4.4B',
            'market_size_2030': '$23B',
            'energy_reduction': '50-70%',
            'efficiency_gain': '100-1000x for MVM',
        },
    }

    results_path = Path('results/z1138_unified_embodied_intelligence.json')
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 80)
    print("UNIFIED EMBODIED INTELLIGENCE COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
