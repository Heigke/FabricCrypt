#!/usr/bin/env python3
"""
z1140: Deep Kernel-Level Embodied Intelligence
===============================================

Integrates ALL embodiment infrastructure:
1. Real FPGA connection with LiteDRAM partial timing writes and Frac operations
2. HIP interoceptive kernels (body_token_encoder, energy_modulated_attention)
3. Deep kernel-level GPU telemetry and actuation
4. Physical DRAM decay as reservoir computing substrate

This represents the deepest level of hardware-software embodiment:
- GPU telemetry affects attention computation at kernel level
- FPGA DRAM charge levels create physical memory dynamics
- Homeostatic gating regulates computation based on hardware stress

Author: FEEL Research Team
Date: 2026-01-31
"""

import os
import sys
import time
import json
import struct
import ctypes
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import embodiment infrastructure
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample

# Try FPGA interface
try:
    from src.fpga.fpga_interface import FPGAInterface
    FPGA_AVAILABLE = True
except ImportError:
    FPGA_AVAILABLE = False
    print("Warning: FPGA interface not available")

# Try actuator client
try:
    from src.actuator import ActuatorClient, get_client
    ACTUATOR_AVAILABLE = True
except (ImportError, PermissionError):
    ACTUATOR_AVAILABLE = False
    print("Warning: Actuator client not available")

# Try HIP kernel wrapper
try:
    from src.kernels.hip_wrapper import EnergyKernels, compile_kernels
    HIP_AVAILABLE = True
except ImportError:
    HIP_AVAILABLE = False
    print("Warning: HIP kernel wrapper not available")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DeepEmbodimentConfig:
    """Configuration for deep kernel-level embodiment."""
    # Model architecture
    vocab_size: int = 256
    hidden_dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    head_dim: int = 64  # hidden_dim // n_heads

    # Telemetry dimensions
    telemetry_dim: int = 8  # GPU(5) + FPGA(3)
    history_len: int = 16   # Timesteps of telemetry history
    num_body_tokens: int = 4  # Body tokens in sequence

    # FPGA reservoir
    reservoir_dim: int = 128
    fpga_decay_scale: float = 0.1
    partial_tras_default: int = 10  # Cycles before PRE (1-37)
    num_fracs_default: int = 6      # ACT→PRE iterations

    # Actuation targets
    power_setpoint: float = 25.0  # Watts
    temp_setpoint: float = 60.0   # Celsius

    # Training
    batch_size: int = 16
    seq_len: int = 128
    learning_rate: float = 3e-4
    n_epochs: int = 15
    device: str = "cuda"


# =============================================================================
# FPGA Reservoir Module
# =============================================================================

class FPGAReservoir(nn.Module):
    """
    Physical DRAM reservoir using FPGA partial timing writes.

    Uses Frac operations (rapid ACT→PRE cycles) to create intermediate
    charge levels in DRAM cells. The charge decay follows Arrhenius
    dynamics based on temperature.
    """

    def __init__(self, config: DeepEmbodimentConfig, fpga: Optional[FPGAInterface] = None):
        super().__init__()
        self.config = config
        self.fpga = fpga
        self.connected = False

        # Reservoir state (mirrors FPGA DRAM)
        self.register_buffer('reservoir_state', torch.zeros(config.reservoir_dim))
        self.register_buffer('last_temp_c', torch.tensor(25.0))
        self.last_update_time = time.time()

        # Projections
        self.input_proj = nn.Linear(config.hidden_dim, config.reservoir_dim)
        self.readout = nn.Linear(config.reservoir_dim * 2, config.hidden_dim)

        # Decay parameters (learned, but physics-constrained)
        self.decay_base = nn.Parameter(torch.tensor(0.1))

        # Try to connect to FPGA
        if self.fpga is not None:
            self._connect()

    def _connect(self) -> bool:
        """Connect to physical FPGA."""
        if self.fpga is None:
            return False
        try:
            self.connected = self.fpga.connect()
            if self.connected:
                status = self.fpga.get_status()
                print(f"FPGA connected: DDR3 calibrated={status.get('ddr3_calibrated', False)}")
                temp, _ = self.fpga.read_temperature()
                self.last_temp_c.fill_(temp)
                print(f"FPGA temperature: {temp:.1f}°C")
            return self.connected
        except Exception as e:
            print(f"FPGA connection failed: {e}")
            return False

    def _arrhenius_decay(self, temp_c: float, dt: float) -> float:
        """
        Compute Arrhenius decay factor.

        DRAM charge decay follows: rate ∝ exp(-Ea / (k*T))
        Higher temperature = faster decay
        """
        # Activation energy normalized by Boltzmann constant
        Ea_over_k = 4000.0  # ~0.35 eV typical for DRAM
        T_kelvin = temp_c + 273.15
        T_ref = 300.0  # 27°C reference

        # Relative rate increase
        rate_ratio = np.exp(Ea_over_k * (1/T_ref - 1/T_kelvin))

        # Base decay rate adjusted by temperature
        decay_rate = float(self.decay_base.item()) * rate_ratio
        decay_factor = np.exp(-decay_rate * dt)

        return decay_factor

    def _do_frac_operation(self, input_vec: torch.Tensor) -> torch.Tensor:
        """
        Perform Frac operation on FPGA to create partial charge levels.

        This writes data to DRAM with truncated timing, creating analog
        charge levels that decay naturally.
        """
        if not self.connected or self.fpga is None:
            return input_vec

        # Convert input to 16-byte pattern
        input_np = input_vec.detach().cpu().numpy()[:16]
        data = bytes([int(max(0, min(255, x * 127 + 128))) for x in input_np])
        if len(data) < 16:
            data = data + bytes(16 - len(data))

        # Perform raw Frac operation
        try:
            result = self.fpga.raw_frac(
                address=0x100000,
                data=data,
                num_fracs=self.config.num_fracs_default,
                sr_wait=100
            )
            if result.get('success'):
                # Convert readback to tensor
                readback = result['read_data']
                charge_level = torch.tensor(
                    [((b - 128) / 127.0) for b in readback],
                    dtype=torch.float32,
                    device=input_vec.device
                )
                return charge_level
        except Exception as e:
            print(f"Frac operation failed: {e}")

        return input_vec

    def _do_partial_timing_write(self, input_vec: torch.Tensor, timing_offset: int = 20) -> torch.Tensor:
        """
        Write with adjusted timing for analog charge levels.

        timing_offset: 0-63 taps (0=full charge, higher=less charge)
        """
        if not self.connected or self.fpga is None:
            return input_vec

        input_np = input_vec.detach().cpu().numpy()[:16]
        data = bytes([int(max(0, min(255, x * 127 + 128))) for x in input_np])
        if len(data) < 16:
            data = data + bytes(16 - len(data))

        try:
            result = self.fpga.partial_timing_write(
                address=0x100000,
                data=data,
                timing_offset=timing_offset
            )
            if result.get('success'):
                readback = result['read_data']
                charge_level = torch.tensor(
                    [((b - 128) / 127.0) for b in readback],
                    dtype=torch.float32,
                    device=input_vec.device
                )
                return charge_level
        except Exception as e:
            print(f"Partial timing write failed: {e}")

        return input_vec

    def forward(self, x: torch.Tensor, temperature: float = 25.0) -> torch.Tensor:
        """
        Forward pass through FPGA reservoir.

        Args:
            x: [batch, hidden_dim] input from transformer
            temperature: Current FPGA/GPU temperature in Celsius

        Returns:
            [batch, hidden_dim] reservoir output
        """
        batch_size = x.shape[0]

        # Project input to reservoir dimension
        reservoir_input = self.input_proj(x.mean(dim=0) if len(x.shape) > 1 else x)

        # Apply physics decay based on time and temperature
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        with torch.no_grad():
            decay_factor = self._arrhenius_decay(temperature, dt)
            self.reservoir_state = self.reservoir_state * decay_factor

        # Update reservoir with input
        # Use Frac or partial timing write if FPGA connected
        if self.connected:
            # Alternate between Frac and partial timing operations
            if np.random.random() < 0.5:
                physical_output = self._do_frac_operation(reservoir_input)
            else:
                timing_offset = int(np.clip(temperature - 30, 0, 63))  # Temperature-dependent
                physical_output = self._do_partial_timing_write(reservoir_input, timing_offset)
        else:
            physical_output = reservoir_input

        # Combine with decayed state
        with torch.no_grad():
            self.reservoir_state = 0.7 * self.reservoir_state + 0.3 * physical_output[:self.config.reservoir_dim].detach()

        # Readout combines current and decayed state
        reservoir_expanded = self.reservoir_state.detach().unsqueeze(0).expand(batch_size, -1)
        combined = torch.cat([reservoir_expanded, physical_output[:self.config.reservoir_dim].unsqueeze(0).expand(batch_size, -1)], dim=-1)

        return self.readout(combined)


# =============================================================================
# HIP Kernel Integration
# =============================================================================

class HIPInteroceptiveAttention(nn.Module):
    """
    Attention module using HIP interoceptive kernels.

    When HIP kernels are available, uses:
    - energy_modulated_attention: attention scaled by energy/homeostatic state
    - body_token_encoder: telemetry history → body token embeddings

    Falls back to standard PyTorch attention otherwise.
    """

    def __init__(self, config: DeepEmbodimentConfig):
        super().__init__()
        self.config = config

        # Try to load HIP kernels
        self.hip_kernels = None
        if HIP_AVAILABLE:
            try:
                self.hip_kernels = EnergyKernels(compile_if_missing=True, arch="gfx1100")
                if self.hip_kernels.available:
                    print("HIP interoceptive kernels loaded successfully")
                else:
                    print("HIP kernels not available, using PyTorch fallback")
                    self.hip_kernels = None
            except Exception as e:
                print(f"HIP kernel initialization failed: {e}")

        # Standard attention projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Body token encoder (learned, for telemetry → embedding)
        self.body_token_encoder = nn.Sequential(
            nn.Linear(config.telemetry_dim * config.history_len, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_body_tokens * config.hidden_dim)
        )

    def encode_body_tokens(self, telemetry_history: torch.Tensor) -> torch.Tensor:
        """
        Encode telemetry history into body tokens.

        Args:
            telemetry_history: [batch, history_len, telemetry_dim]

        Returns:
            body_tokens: [batch, num_body_tokens, hidden_dim]
        """
        batch_size = telemetry_history.shape[0]
        flat = telemetry_history.view(batch_size, -1)
        encoded = self.body_token_encoder(flat)
        return encoded.view(batch_size, self.config.num_body_tokens, self.config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        body_state: Optional[torch.Tensor] = None,
        energy_mode: str = 'balanced'
    ) -> torch.Tensor:
        """
        Forward pass with optional energy modulation.

        Args:
            x: [batch, seq, hidden_dim]
            body_state: [batch, telemetry_dim] current telemetry
            energy_mode: 'low', 'balanced', or 'high'

        Returns:
            [batch, seq, hidden_dim]
        """
        batch, seq, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch, seq, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq, self.config.n_heads, self.config.head_dim).transpose(1, 2)

        # Use HIP kernel if available
        if self.hip_kernels is not None and self.hip_kernels.available:
            attn_out = self.hip_kernels.energy_attention(Q, K, V, energy_mode)
        else:
            # Standard scaled dot-product attention
            scale = 1.0 / (self.config.head_dim ** 0.5)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            # Apply homeostatic gating if body_state available
            if body_state is not None:
                # Compute stress level from body state
                power_ratio = body_state[:, 3:4] * 50.0 / self.config.power_setpoint  # Denormalize
                temp_ratio = body_state[:, 0:1] * 100.0 / self.config.temp_setpoint
                stress = (power_ratio - 1.0).abs() + (temp_ratio - 1.0).abs()
                gate = torch.sigmoid(-stress)  # Gate reduces when stressed
                scores = scores * gate.view(batch, 1, 1, 1)

            attn = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, V)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.config.hidden_dim)
        return self.out_proj(attn_out)


# =============================================================================
# Deep Embodied Transformer
# =============================================================================

class DeepEmbodiedTransformerBlock(nn.Module):
    """Transformer block with deep embodiment."""

    def __init__(self, config: DeepEmbodimentConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Interoceptive attention
        self.attn = HIPInteroceptiveAttention(config)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

        # FiLM conditioning from telemetry
        self.film_gamma = nn.Linear(config.telemetry_dim, config.hidden_dim)
        self.film_beta = nn.Linear(config.telemetry_dim, config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        body_state: Optional[torch.Tensor] = None,
        energy_mode: str = 'balanced'
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(self.ln1(x), body_state, energy_mode)
        x = x + attn_out

        # FiLM conditioning from body state
        if body_state is not None:
            gamma = 1.0 + self.film_gamma(body_state).unsqueeze(1)
            beta = self.film_beta(body_state).unsqueeze(1)
            x = x * gamma + beta

        # FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x


class DeepEmbodiedTransformer(nn.Module):
    """
    Transformer with deep kernel-level embodiment.

    Integrates:
    - GPU telemetry at 100Hz affecting attention computation
    - FPGA DRAM reservoir with physical decay
    - HIP kernels for energy-modulated attention
    - Actuation for power cap control
    """

    def __init__(self, config: DeepEmbodimentConfig, fpga: Optional[FPGAInterface] = None):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(config.seq_len + config.num_body_tokens, config.hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DeepEmbodiedTransformerBlock(config, i)
            for i in range(config.n_layers)
        ])

        # FPGA reservoir
        self.reservoir = FPGAReservoir(config, fpga)

        # Output head
        self.ln_out = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

        # Telemetry history buffer
        self.register_buffer(
            'telemetry_history',
            torch.zeros(1, config.history_len, config.telemetry_dim)
        )

        # Interoceptive prediction head
        self.intero_pred = nn.Linear(config.hidden_dim, config.telemetry_dim)

    def _update_telemetry_history(self, new_sample: torch.Tensor):
        """Update rolling telemetry history."""
        self.telemetry_history = torch.cat([
            self.telemetry_history[:, 1:, :],
            new_sample.unsqueeze(0).unsqueeze(1)
        ], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        body_state: Optional[torch.Tensor] = None,
        energy_mode: str = 'balanced'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with embodiment.

        Args:
            input_ids: [batch, seq] token indices
            body_state: [batch, telemetry_dim] current telemetry
            energy_mode: 'low', 'balanced', or 'high'

        Returns:
            logits: [batch, seq, vocab]
            intero_pred: [batch, telemetry_dim] predicted body state
            reservoir_out: [batch, hidden_dim] reservoir contribution
        """
        batch, seq = input_ids.shape
        device = input_ids.device

        # Token + position embeddings
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Get temperature for reservoir decay
        temperature = 25.0
        if body_state is not None:
            temperature = float(body_state[0, 0].item() * 100.0)  # Denormalize

        # Pass through blocks
        for block in self.blocks:
            x = block(x, body_state, energy_mode)

        # FPGA reservoir contribution
        reservoir_out = self.reservoir(x.mean(dim=1), temperature)
        x = x + reservoir_out.unsqueeze(1)

        # Output projection
        x = self.ln_out(x)
        logits = self.output(x)

        # Interoceptive prediction
        intero_pred = self.intero_pred(x.mean(dim=1))

        return logits, intero_pred, reservoir_out


# =============================================================================
# Telemetry and Actuation Manager
# =============================================================================

class EmbodimentManager:
    """
    Manages GPU telemetry, FPGA state, and actuation.

    Provides unified z_feel vector combining:
    - GPU: power, temp, utilization, memory, fan
    - FPGA: temperature, decay_rate, reservoir_state
    """

    def __init__(self, config: DeepEmbodimentConfig):
        self.config = config

        # Initialize GPU telemetry
        self.telemetry = SysfsHwmonTelemetry()

        # Initialize FPGA
        self.fpga = None
        if FPGA_AVAILABLE:
            self.fpga = FPGAInterface()

        # Initialize actuator
        self.actuator = None
        if ACTUATOR_AVAILABLE:
            try:
                self.actuator = get_client()
            except Exception as e:
                print(f"Actuator client failed: {e}")

        # Telemetry history
        self.history = []
        self.max_history = config.history_len

    def get_z_feel(self) -> torch.Tensor:
        """
        Get unified body state vector.

        Returns:
            [telemetry_dim] tensor with normalized telemetry
        """
        # Read GPU telemetry
        sample = self.telemetry.read_sample()

        # Compute normalized GPU features
        power_norm = sample.power_w / 50.0 if sample.power_w else 0.0
        temp_norm = sample.temp_edge_c / 100.0 if sample.temp_edge_c else 0.0
        util_norm = sample.gpu_busy_pct / 100.0 if sample.gpu_busy_pct else 0.0
        mem_norm = sample.vram_used_gb / 16.0 if sample.vram_used_gb else 0.0
        fan_norm = sample.freq_sclk_mhz / 3000.0 if sample.freq_sclk_mhz else 0.0  # Use sclk instead of fan

        # Compute strain/urgency
        strain = (sample.power_w - self.config.power_setpoint) / self.config.power_setpoint if sample.power_w else 0.0
        urgency = (sample.temp_edge_c - self.config.temp_setpoint) / self.config.temp_setpoint if sample.temp_edge_c else 0.0

        # FPGA features (if connected)
        fpga_temp = 0.0
        if self.fpga is not None and hasattr(self.fpga, 'ser') and self.fpga.ser is not None:
            try:
                temp, _ = self.fpga.read_temperature()
                fpga_temp = temp / 100.0
            except:
                pass

        z_feel = torch.tensor([
            temp_norm,    # 0: GPU temp
            power_norm,   # 1: GPU power
            util_norm,    # 2: GPU util
            mem_norm,     # 3: VRAM used
            fan_norm,     # 4: Fan speed
            strain,       # 5: Power strain
            urgency,      # 6: Thermal urgency
            fpga_temp,    # 7: FPGA temp
        ], dtype=torch.float32)

        # Update history
        self.history.append(z_feel)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return z_feel

    def get_telemetry_history(self) -> torch.Tensor:
        """Get stacked telemetry history."""
        if len(self.history) < self.max_history:
            # Pad with zeros
            padded = [torch.zeros(self.config.telemetry_dim)] * (self.max_history - len(self.history))
            padded.extend(self.history)
            return torch.stack(padded)
        return torch.stack(self.history)

    def set_power_cap(self, watts: float) -> bool:
        """Set GPU power cap via actuator daemon."""
        if self.actuator is None:
            return False
        try:
            self.actuator.set_power_cap(int(watts * 1e6))  # Convert to microwatts
            return True
        except Exception as e:
            print(f"Power cap failed: {e}")
            return False

    def get_energy_mode(self, z_feel: torch.Tensor) -> str:
        """Determine energy mode from current state."""
        strain = z_feel[5].item()
        urgency = z_feel[6].item()

        if strain > 0.2 or urgency > 0.2:
            return 'low'  # Stressed - reduce energy
        elif strain < -0.1 and urgency < -0.1:
            return 'high'  # Under budget - can spend more
        return 'balanced'


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: DeepEmbodiedTransformer,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    embodiment: EmbodimentManager,
    config: DeepEmbodimentConfig,
    epoch: int
) -> Dict:
    """Train one epoch with deep embodiment."""
    model.train()
    device = config.device

    total_loss = 0.0
    total_intero_loss = 0.0
    n_batches = 0
    n_regulated = 0

    data_len = data.shape[0] - config.seq_len - 1
    n_steps = min(300, data_len // config.batch_size)  # Limit to 300 steps per epoch

    for step in range(n_steps):
        # Get batch
        batch_start = step * config.batch_size
        batch_input = torch.stack([
            data[batch_start + i : batch_start + i + config.seq_len]
            for i in range(config.batch_size)
        ]).to(device)
        batch_target = torch.stack([
            data[batch_start + i + 1 : batch_start + i + config.seq_len + 1]
            for i in range(config.batch_size)
        ]).to(device)

        # Get current body state
        z_feel = embodiment.get_z_feel().to(device)
        z_feel_batch = z_feel.unsqueeze(0).expand(config.batch_size, -1)

        # Determine energy mode from state
        energy_mode = embodiment.get_energy_mode(z_feel)

        # Forward pass
        logits, intero_pred, reservoir_out = model(batch_input, z_feel_batch, energy_mode)

        # Language modeling loss
        lm_loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            batch_target.view(-1)
        )

        # Interoceptive prediction loss (predict next body state)
        intero_target = z_feel  # Current state as target for previous prediction
        intero_loss = F.mse_loss(intero_pred.mean(dim=0), intero_target)

        # Combined loss
        loss = lm_loss + 0.1 * intero_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += lm_loss.item()
        total_intero_loss += intero_loss.item()
        n_batches += 1

        # Actuation: regulate power based on thermal state
        if step % 50 == 0:
            if z_feel[6].item() > 0.15:  # Thermal urgency high
                if embodiment.set_power_cap(20.0):
                    n_regulated += 1
            elif z_feel[6].item() < -0.1:  # Under thermal limit
                embodiment.set_power_cap(30.0)

        # Progress
        if step % 100 == 0:
            print(f"  Epoch {epoch} Step {step}/{n_steps}: loss={lm_loss.item():.4f} "
                  f"intero={intero_loss.item():.4f} mode={energy_mode}")

    return {
        'avg_loss': total_loss / n_batches,
        'avg_intero_loss': total_intero_loss / n_batches,
        'n_batches': n_batches,
        'n_regulated': n_regulated
    }


def generate_text(model: DeepEmbodiedTransformer, start_ids: torch.Tensor,
                  embodiment: EmbodimentManager, config: DeepEmbodimentConfig,
                  max_len: int = 100) -> str:
    """Generate text from the model."""
    model.eval()
    device = config.device

    generated = start_ids.clone()

    with torch.no_grad():
        for _ in range(max_len):
            # Get context window
            context = generated[:, -config.seq_len:]

            # Get body state
            z_feel = embodiment.get_z_feel().to(device)
            z_feel_batch = z_feel.unsqueeze(0)

            # Forward
            logits, _, _ = model(context, z_feel_batch, 'balanced')

            # Sample next token
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

    # Decode
    tokens = generated[0].cpu().numpy()
    text = ''.join([chr(t) if 32 <= t < 127 else ' ' for t in tokens])
    return text


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("z1140: Deep Kernel-Level Embodied Intelligence")
    print("=" * 70)
    print()

    # Configuration
    config = DeepEmbodimentConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)

    print(f"Device: {device}")
    print(f"Model: {config.hidden_dim}d x {config.n_layers}L x {config.n_heads}H")
    print()

    # Load data
    data_path = PROJECT_ROOT / "data" / "tiny_shakespeare.txt"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        # Create simple test data
        print("Creating test data...")
        text = "To be or not to be, that is the question. " * 1000
    else:
        with open(data_path, 'r') as f:
            text = f.read()

    print(f"Data: {len(text)} characters")

    # Tokenize (char-level)
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)

    # Initialize embodiment manager
    embodiment = EmbodimentManager(config)

    # Initialize FPGA if available
    fpga = None
    if FPGA_AVAILABLE:
        fpga = FPGAInterface()
        if fpga.connect():
            status = fpga.get_status()
            print(f"FPGA: DDR3={status.get('ddr3_calibrated')}, temp={status.get('temperature', 0):.1f}°C")
        else:
            print("FPGA: Not connected")
            fpga = None

    # Create model
    model = DeepEmbodiedTransformer(config, fpga).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Check HIP kernels
    if model.blocks[0].attn.hip_kernels is not None:
        print(f"HIP kernels: Available")
    else:
        print(f"HIP kernels: Using PyTorch fallback")

    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training
    results = {
        'experiment': 'z1140_deep_kernel_embodiment',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'hidden_dim': config.hidden_dim,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'n_epochs': config.n_epochs,
            'hip_kernels': model.blocks[0].attn.hip_kernels is not None,
            'fpga_connected': fpga is not None,
            'actuator_available': ACTUATOR_AVAILABLE,
        },
        'epochs': []
    }

    print("Training...")
    print("-" * 70)

    for epoch in range(config.n_epochs):
        epoch_results = train_epoch(model, data, optimizer, embodiment, config, epoch)

        results['epochs'].append({
            'epoch': epoch,
            'avg_loss': epoch_results['avg_loss'],
            'avg_intero_loss': epoch_results['avg_intero_loss'],
            'n_batches': epoch_results['n_batches'],
            'n_regulated': epoch_results['n_regulated']
        })

        print(f"Epoch {epoch}: loss={epoch_results['avg_loss']:.4f} "
              f"intero={epoch_results['avg_intero_loss']:.4f} "
              f"regulated={epoch_results['n_regulated']}")

    # Generate sample
    print()
    print("Generating sample...")
    start = torch.tensor([[ord(c) for c in "The "]], device=device)
    generated = generate_text(model, start, embodiment, config, max_len=100)
    print(f"Generated: {generated}")

    results['generated'] = generated
    results['final_loss'] = results['epochs'][-1]['avg_loss']

    # Save results
    results_path = PROJECT_ROOT / "results" / "z1140_deep_kernel_embodiment.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Cleanup
    if fpga is not None:
        fpga.disconnect()

    print()
    print("=" * 70)
    print("z1140 Complete!")
    print(f"Final loss: {results['final_loss']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
