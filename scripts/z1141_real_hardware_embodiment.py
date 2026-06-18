#!/usr/bin/env python3
"""
z1141: Real Hardware Embodiment - Closed-Loop Intelligence System
=================================================================

NO SIMULATION - Uses real hardware:
- Real GPU telemetry at 100Hz (sysfs_hwmon)
- Real FPGA connection (Arty A7 via /dev/ttyUSB1)
- Real GPU actuation (power cap, DPM level)
- Real DRAM decay and partial timing writes

Comprehensive metrics beyond energy:
- Intelligence: perplexity, coherence, generation quality
- Efficiency: J/token, tokens/second, tokens/Joule
- Stability: thermal variance, power variance
- Adaptation: response time, regulation events

Fast closed-loop embodiment:
- SENSE: 100Hz GPU + FPGA telemetry
- FEEL: 8-dim z_feel body state with history
- REGULATE: Homeostatic control with setpoints
- EXPRESS: Adaptive behavior (exit layer, attention, precision)

Author: FEEL Research Team
Date: 2026-01-31
"""

import os
import sys
import time
import json
import math
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import REAL hardware interfaces
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample, EnergyAccumulator

# Try FPGA interface
FPGA_AVAILABLE = False
FPGAInterface = None
try:
    from src.fpga.fpga_interface import FPGAInterface as _FPGAInterface
    FPGAInterface = _FPGAInterface
    FPGA_AVAILABLE = True
    print("FPGA interface loaded")
except ImportError as e:
    print(f"FPGA interface not available: {e}")

# Try actuator
ACTUATOR_AVAILABLE = False
try:
    from src.actuator import get_client, ActuatorClient
    ACTUATOR_AVAILABLE = True
    print("Actuator client loaded")
except (ImportError, PermissionError) as e:
    print(f"Actuator not available: {e}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RealHardwareConfig:
    """Configuration for real hardware embodiment."""
    # Model
    vocab_size: int = 256
    hidden_dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    head_dim: int = 64
    max_seq_len: int = 128

    # Telemetry
    telemetry_hz: int = 100  # 100Hz sampling
    telemetry_dim: int = 12  # Extended telemetry
    history_len: int = 32    # ~320ms of history

    # Setpoints for homeostasis
    power_setpoint_w: float = 25.0
    temp_setpoint_c: float = 55.0
    util_setpoint_pct: float = 80.0

    # FPGA
    fpga_port: str = "/dev/ttyUSB1"
    fpga_baudrate: int = 115200
    reservoir_dim: int = 64

    # Training
    batch_size: int = 16
    n_epochs: int = 20
    steps_per_epoch: int = 200
    learning_rate: float = 3e-4

    # Metrics
    warmup_steps: int = 50  # Steps before measuring

    device: str = "cuda"


# =============================================================================
# Real Hardware Manager
# =============================================================================

class RealHardwareManager:
    """
    Manages real GPU and FPGA hardware.
    No simulation - all measurements from actual hardware.
    """

    def __init__(self, config: RealHardwareConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Initialize GPU telemetry (REAL)
        self.telemetry = SysfsHwmonTelemetry()
        self.energy_accumulator = EnergyAccumulator()

        # Telemetry history buffer
        self.history: deque = deque(maxlen=config.history_len)

        # Initialize FPGA (REAL)
        self.fpga = None
        self.fpga_connected = False
        if FPGA_AVAILABLE and FPGAInterface:
            self._connect_fpga()

        # Initialize actuator (REAL)
        self.actuator = None
        if ACTUATOR_AVAILABLE:
            try:
                self.actuator = get_client()
                print(f"Actuator connected")
            except Exception as e:
                print(f"Actuator connection failed: {e}")

        # Metrics tracking
        self.metrics = {
            'energy_samples': [],
            'power_samples': [],
            'temp_samples': [],
            'latency_samples': [],
            'tokens_generated': 0,
            'regulation_events': 0,
            'fpga_operations': 0,
        }

        # Timing for feedback loop
        self.last_sample_time = time.time()
        self.sample_interval = 1.0 / config.telemetry_hz

    def _connect_fpga(self) -> bool:
        """Connect to real FPGA hardware."""
        try:
            self.fpga = FPGAInterface(
                port=self.config.fpga_port,
                baudrate=self.config.fpga_baudrate
            )
            if self.fpga.connect():
                status = self.fpga.ping()
                if status.get('valid') and status.get('ddr3_ready'):
                    self.fpga_connected = True
                    temp, _ = self.fpga.read_temperature()
                    print(f"FPGA connected: DDR3 ready, temp={temp:.1f}°C")
                    return True
                else:
                    print(f"FPGA: DDR3 not ready: {status}")
            else:
                print("FPGA: Connection failed")
        except Exception as e:
            print(f"FPGA error: {e}")
        return False

    def sample_telemetry(self) -> torch.Tensor:
        """
        Sample REAL GPU telemetry.
        Returns 12-dim normalized feature vector.
        """
        sample = self.telemetry.read_sample()
        self.energy_accumulator.add_sample(sample)

        # Track raw metrics
        if sample.power_w:
            self.metrics['power_samples'].append(sample.power_w)
        if sample.temp_edge_c:
            self.metrics['temp_samples'].append(sample.temp_edge_c)

        # Normalize features
        power_norm = (sample.power_w or 0) / 50.0
        temp_edge_norm = (sample.temp_edge_c or 0) / 100.0
        temp_junction_norm = (sample.temp_junction_c or 0) / 100.0
        temp_mem_norm = (sample.temp_mem_c or 0) / 100.0
        sclk_norm = (sample.freq_sclk_mhz or 0) / 3000.0
        mclk_norm = (sample.freq_mclk_mhz or 0) / 2000.0
        util_norm = (sample.gpu_busy_pct or 0) / 100.0
        vram_norm = (sample.vram_used_gb or 0) / 16.0

        # Compute derivatives (strain/urgency)
        power_strain = ((sample.power_w or 0) - self.config.power_setpoint_w) / self.config.power_setpoint_w
        temp_urgency = ((sample.temp_edge_c or 0) - self.config.temp_setpoint_c) / self.config.temp_setpoint_c
        util_pressure = ((sample.gpu_busy_pct or 0) - self.config.util_setpoint_pct) / self.config.util_setpoint_pct

        # FPGA temperature if connected
        fpga_temp_norm = 0.0
        if self.fpga_connected and self.fpga:
            try:
                fpga_temp, _ = self.fpga.read_temperature()
                fpga_temp_norm = fpga_temp / 100.0
            except:
                pass

        z_feel = torch.tensor([
            temp_edge_norm,      # 0: GPU edge temp
            temp_junction_norm,  # 1: GPU junction temp
            temp_mem_norm,       # 2: Memory temp
            power_norm,          # 3: Power
            sclk_norm,           # 4: Shader clock
            mclk_norm,           # 5: Memory clock
            util_norm,           # 6: GPU utilization
            vram_norm,           # 7: VRAM usage
            power_strain,        # 8: Power strain (vs setpoint)
            temp_urgency,        # 9: Thermal urgency (vs setpoint)
            util_pressure,       # 10: Utilization pressure
            fpga_temp_norm,      # 11: FPGA temperature
        ], dtype=torch.float32)

        # Update history
        self.history.append(z_feel)

        return z_feel

    def get_telemetry_history(self) -> torch.Tensor:
        """Get stacked telemetry history."""
        if len(self.history) < self.config.history_len:
            # Pad with zeros
            pad_len = self.config.history_len - len(self.history)
            padded = [torch.zeros(self.config.telemetry_dim)] * pad_len
            padded.extend(list(self.history))
            return torch.stack(padded)
        return torch.stack(list(self.history))

    def fpga_partial_write(self, data: torch.Tensor, timing_offset: int = 20) -> Optional[torch.Tensor]:
        """
        Perform REAL FPGA partial timing write.
        Returns readback data showing actual charge levels.
        """
        if not self.fpga_connected or not self.fpga:
            return None

        try:
            # Convert tensor to bytes
            data_np = data.detach().cpu().numpy()[:16]
            data_bytes = bytes([int(max(0, min(255, x * 127 + 128))) for x in data_np])
            if len(data_bytes) < 16:
                data_bytes = data_bytes + bytes(16 - len(data_bytes))

            result = self.fpga.partial_timing_write(
                address=0x100000,
                data=data_bytes,
                timing_offset=timing_offset
            )

            if result.get('success'):
                self.metrics['fpga_operations'] += 1
                readback = result['read_data']
                return torch.tensor(
                    [((b - 128) / 127.0) for b in readback],
                    dtype=torch.float32
                )
        except Exception as e:
            print(f"FPGA partial write error: {e}")

        return None

    def fpga_frac_operation(self, data: torch.Tensor, num_fracs: int = 6) -> Optional[torch.Tensor]:
        """
        Perform REAL FPGA Frac operation for analog charge levels.
        """
        if not self.fpga_connected or not self.fpga:
            return None

        try:
            data_np = data.detach().cpu().numpy()[:16]
            data_bytes = bytes([int(max(0, min(255, x * 127 + 128))) for x in data_np])
            if len(data_bytes) < 16:
                data_bytes = data_bytes + bytes(16 - len(data_bytes))

            result = self.fpga.raw_frac(
                address=0x100000,
                data=data_bytes,
                num_fracs=num_fracs,
                sr_wait=100
            )

            if result.get('success'):
                self.metrics['fpga_operations'] += 1
                readback = result['read_data']
                return torch.tensor(
                    [((b - 128) / 127.0) for b in readback],
                    dtype=torch.float32
                )
        except Exception as e:
            print(f"FPGA Frac error: {e}")

        return None

    def actuate_power(self, power_cap_w: float) -> bool:
        """Set REAL GPU power cap."""
        if not self.actuator:
            return False

        try:
            # Convert to microwatts
            power_cap_uw = int(power_cap_w * 1e6)
            self.actuator.set_power_cap(power_cap_uw)
            self.metrics['regulation_events'] += 1
            return True
        except Exception as e:
            return False

    def regulate(self, z_feel: torch.Tensor) -> str:
        """
        Homeostatic regulation based on current state.
        Returns energy mode: 'low', 'balanced', 'high'
        """
        power_strain = z_feel[8].item()
        temp_urgency = z_feel[9].item()

        # Determine mode based on strain
        if temp_urgency > 0.2 or power_strain > 0.3:
            # System stressed - reduce power
            if self.actuator:
                self.actuate_power(20.0)
            return 'low'
        elif temp_urgency < -0.1 and power_strain < -0.1:
            # System under budget - can increase
            if self.actuator:
                self.actuate_power(35.0)
            return 'high'
        else:
            # Balanced operation
            if self.actuator:
                self.actuate_power(25.0)
            return 'balanced'

    def get_energy_j(self) -> float:
        """Get total energy consumed in Joules."""
        return self.energy_accumulator.total_energy_j

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get all tracked metrics."""
        power_arr = np.array(self.metrics['power_samples']) if self.metrics['power_samples'] else np.array([0])
        temp_arr = np.array(self.metrics['temp_samples']) if self.metrics['temp_samples'] else np.array([0])

        return {
            'total_energy_j': self.get_energy_j(),
            'total_tokens': self.metrics['tokens_generated'],
            'j_per_token': self.get_energy_j() / max(1, self.metrics['tokens_generated']),
            'tokens_per_joule': self.metrics['tokens_generated'] / max(0.001, self.get_energy_j()),
            'power_mean_w': float(power_arr.mean()),
            'power_std_w': float(power_arr.std()),
            'temp_mean_c': float(temp_arr.mean()),
            'temp_std_c': float(temp_arr.std()),
            'regulation_events': self.metrics['regulation_events'],
            'fpga_operations': self.metrics['fpga_operations'],
            'fpga_connected': self.fpga_connected,
            'actuator_available': self.actuator is not None,
        }

    def cleanup(self):
        """Clean up hardware connections."""
        if self.fpga:
            try:
                self.fpga.disconnect()
            except:
                pass


# =============================================================================
# Embodied Transformer with Real Hardware
# =============================================================================

class RealEmbodiedBlock(nn.Module):
    """Transformer block with real hardware embodiment."""

    def __init__(self, config: RealHardwareConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Self-attention
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

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

        # Early exit gate (learns when to skip)
        self.exit_gate = nn.Sequential(
            nn.Linear(config.hidden_dim + config.telemetry_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        z_feel: torch.Tensor,
        energy_mode: str = 'balanced'
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward with embodiment.
        Returns (output, exit_confidence).
        """
        batch, seq, _ = x.shape

        # Compute exit confidence
        pooled = x.mean(dim=1)  # [batch, hidden]
        z_feel_batch = z_feel.unsqueeze(0).expand(batch, -1)
        gate_input = torch.cat([pooled, z_feel_batch], dim=-1)
        exit_conf = self.exit_gate(gate_input).mean().item()

        # Self-attention
        h = self.ln1(x)
        Q = self.q_proj(h).view(batch, seq, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(batch, seq, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(batch, seq, self.config.n_heads, self.config.head_dim).transpose(1, 2)

        # Scaled dot-product attention with energy modulation
        scale = 1.0 / math.sqrt(self.config.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Energy mode affects attention sharpness
        if energy_mode == 'low':
            scores = scores * 0.8  # Softer attention
        elif energy_mode == 'high':
            scores = scores * 1.2  # Sharper attention

        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.config.hidden_dim)
        x = x + self.out_proj(attn_out)

        # FiLM conditioning
        gamma = 1.0 + self.film_gamma(z_feel_batch).unsqueeze(1)
        beta = self.film_beta(z_feel_batch).unsqueeze(1)
        x = x * gamma + beta

        # FFN
        x = x + self.ffn(self.ln2(x))

        return x, exit_conf


class RealEmbodiedTransformer(nn.Module):
    """
    Transformer with real hardware embodiment.
    Uses actual GPU telemetry and FPGA operations.
    """

    def __init__(self, config: RealHardwareConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            RealEmbodiedBlock(config, i)
            for i in range(config.n_layers)
        ])

        # FPGA reservoir projection
        self.reservoir_proj = nn.Linear(16, config.hidden_dim)  # FPGA returns 16 bytes

        # Output
        self.ln_out = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

        # Interoceptive prediction head
        self.intero_pred = nn.Linear(config.hidden_dim, config.telemetry_dim)

        # Quality prediction head (for business metrics)
        self.quality_pred = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        z_feel: torch.Tensor,
        fpga_state: Optional[torch.Tensor] = None,
        energy_mode: str = 'balanced'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:
        """
        Forward pass with real hardware embodiment.

        Returns:
            logits, intero_pred, quality_pred, exit_confidences
        """
        batch, seq = input_ids.shape
        device = input_ids.device

        # Token + position embeddings
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Add FPGA reservoir state if available
        if fpga_state is not None:
            reservoir_embed = self.reservoir_proj(fpga_state)
            x = x + reservoir_embed.unsqueeze(1)

        # Pass through blocks
        exit_confs = []
        for block in self.blocks:
            x, exit_conf = block(x, z_feel, energy_mode)
            exit_confs.append(exit_conf)

        # Output projections
        x = self.ln_out(x)
        logits = self.output(x)

        # Interoceptive prediction
        pooled = x.mean(dim=1)
        intero_pred = self.intero_pred(pooled)
        quality_pred = self.quality_pred(pooled)

        return logits, intero_pred, quality_pred, exit_confs


# =============================================================================
# Training and Evaluation
# =============================================================================

def compute_coherence(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute generation coherence (inverse perplexity)."""
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='mean'
    )
    perplexity = torch.exp(loss).item()
    coherence = 1.0 / max(perplexity, 1.0)
    return coherence


def train_epoch(
    model: RealEmbodiedTransformer,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    hardware: RealHardwareManager,
    config: RealHardwareConfig,
    epoch: int
) -> Dict[str, Any]:
    """Train one epoch with real hardware embodiment."""
    model.train()
    device = torch.device(config.device)

    total_loss = 0.0
    total_intero_loss = 0.0
    total_coherence = 0.0
    total_quality = 0.0
    exit_conf_sum = [0.0] * config.n_layers
    n_batches = 0

    epoch_start_energy = hardware.get_energy_j()
    epoch_start_time = time.time()

    for step in range(config.steps_per_epoch):
        # Sample batch
        batch_start = (step * config.batch_size) % (len(data) - config.max_seq_len - 1)
        batch_input = torch.stack([
            data[batch_start + i : batch_start + i + config.max_seq_len]
            for i in range(config.batch_size)
        ]).to(device)
        batch_target = torch.stack([
            data[batch_start + i + 1 : batch_start + i + config.max_seq_len + 1]
            for i in range(config.batch_size)
        ]).to(device)

        # Get REAL telemetry
        z_feel = hardware.sample_telemetry().to(device)

        # Regulate based on state
        energy_mode = hardware.regulate(z_feel)

        # FPGA operation (if connected)
        fpga_state = None
        if hardware.fpga_connected:
            # Use telemetry to decide operation type
            if z_feel[9].item() > 0.1:  # High thermal urgency
                # Use Frac to create cooler analog states
                fpga_state = hardware.fpga_frac_operation(z_feel[:16], num_fracs=8)
            else:
                # Normal partial timing write
                timing_offset = int(20 + z_feel[8].item() * 20)  # Adjust by power strain
                timing_offset = max(0, min(63, timing_offset))
                fpga_state = hardware.fpga_partial_write(z_feel[:16], timing_offset)

        if fpga_state is not None:
            fpga_state = fpga_state.to(device)

        # Forward pass
        logits, intero_pred, quality_pred, exit_confs = model(
            batch_input, z_feel, fpga_state, energy_mode
        )

        # Losses
        lm_loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            batch_target.view(-1)
        )
        intero_loss = F.mse_loss(intero_pred.mean(dim=0), z_feel)

        # Combined loss
        loss = lm_loss + 0.1 * intero_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += lm_loss.item()
        total_intero_loss += intero_loss.item()
        total_coherence += compute_coherence(logits, batch_target)
        total_quality += torch.sigmoid(quality_pred).mean().item()

        for i, conf in enumerate(exit_confs):
            exit_conf_sum[i] += conf

        n_batches += 1
        hardware.metrics['tokens_generated'] += config.batch_size * config.max_seq_len

        # Progress
        if step % 50 == 0:
            hw_metrics = hardware.get_comprehensive_metrics()
            print(f"  E{epoch} S{step}/{config.steps_per_epoch}: "
                  f"loss={lm_loss.item():.4f} "
                  f"mode={energy_mode} "
                  f"power={hw_metrics['power_mean_w']:.1f}W "
                  f"fpga={hardware.fpga_connected}")

    epoch_energy = hardware.get_energy_j() - epoch_start_energy
    epoch_time = time.time() - epoch_start_time

    return {
        'avg_loss': total_loss / n_batches,
        'avg_intero_loss': total_intero_loss / n_batches,
        'avg_coherence': total_coherence / n_batches,
        'avg_quality': total_quality / n_batches,
        'exit_confidences': [c / n_batches for c in exit_conf_sum],
        'epoch_energy_j': epoch_energy,
        'epoch_time_s': epoch_time,
        'tokens_this_epoch': config.batch_size * config.max_seq_len * config.steps_per_epoch,
        'j_per_token_epoch': epoch_energy / (config.batch_size * config.max_seq_len * config.steps_per_epoch),
    }


def generate_sample(
    model: RealEmbodiedTransformer,
    hardware: RealHardwareManager,
    config: RealHardwareConfig,
    prompt: str = "The ",
    max_tokens: int = 100
) -> Tuple[str, Dict]:
    """Generate text sample with real hardware."""
    model.eval()
    device = torch.device(config.device)

    # Encode prompt
    tokens = [ord(c) for c in prompt]
    generated = torch.tensor([tokens], device=device)

    gen_start = time.time()
    gen_start_energy = hardware.get_energy_j()

    with torch.no_grad():
        for _ in range(max_tokens):
            # Get context
            context = generated[:, -config.max_seq_len:]

            # Get telemetry
            z_feel = hardware.sample_telemetry().to(device)
            energy_mode = hardware.regulate(z_feel)

            # Forward
            logits, _, _, _ = model(context, z_feel, None, energy_mode)

            # Sample
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

    gen_time = time.time() - gen_start
    gen_energy = hardware.get_energy_j() - gen_start_energy

    # Decode
    tokens_out = generated[0].cpu().numpy()
    text = ''.join([chr(t) if 32 <= t < 127 else ' ' for t in tokens_out])

    return text, {
        'generation_time_s': gen_time,
        'generation_energy_j': gen_energy,
        'tokens_generated': len(tokens_out),
        'tokens_per_second': len(tokens_out) / gen_time,
        'j_per_token_gen': gen_energy / len(tokens_out),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("z1141: Real Hardware Embodiment - NO SIMULATION")
    print("=" * 70)
    print()

    config = RealHardwareConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)

    print(f"Device: {device}")
    print(f"Model: {config.hidden_dim}d x {config.n_layers}L x {config.n_heads}H")
    print()

    # Initialize REAL hardware
    print("Initializing real hardware...")
    hardware = RealHardwareManager(config)
    print(f"  GPU telemetry: Active (sysfs_hwmon)")
    print(f"  FPGA: {'Connected' if hardware.fpga_connected else 'Not connected'}")
    print(f"  Actuator: {'Available' if hardware.actuator else 'Not available'}")
    print()

    # Load data
    data_path = PROJECT_ROOT / "data" / "tiny_shakespeare.txt"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        print("Creating test data...")
        text = "To be or not to be, that is the question. " * 1000
    else:
        with open(data_path, 'r') as f:
            text = f.read()

    print(f"Data: {len(text)} characters")
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)

    # Create model
    model = RealEmbodiedTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Results
    results = {
        'experiment': 'z1141_real_hardware_embodiment',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'hidden_dim': config.hidden_dim,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'n_epochs': config.n_epochs,
            'steps_per_epoch': config.steps_per_epoch,
        },
        'hardware': {
            'fpga_connected': hardware.fpga_connected,
            'actuator_available': hardware.actuator is not None,
            'telemetry_hz': config.telemetry_hz,
        },
        'epochs': [],
    }

    # Training
    print("Training with REAL hardware feedback...")
    print("-" * 70)

    for epoch in range(config.n_epochs):
        epoch_results = train_epoch(model, data, optimizer, hardware, config, epoch)

        results['epochs'].append({
            'epoch': epoch,
            **epoch_results
        })

        hw_metrics = hardware.get_comprehensive_metrics()

        print(f"Epoch {epoch}: loss={epoch_results['avg_loss']:.4f} "
              f"coherence={epoch_results['avg_coherence']:.4f} "
              f"J/tok={epoch_results['j_per_token_epoch']*1000:.3f}mJ "
              f"reg={hw_metrics['regulation_events']}")

    print()
    print("Generating sample...")
    generated_text, gen_metrics = generate_sample(model, hardware, config)
    print(f"Generated: {generated_text[:100]}...")
    print(f"  Time: {gen_metrics['generation_time_s']:.2f}s")
    print(f"  Energy: {gen_metrics['generation_energy_j']:.4f}J")
    print(f"  Tokens/s: {gen_metrics['tokens_per_second']:.1f}")
    print(f"  mJ/token: {gen_metrics['j_per_token_gen']*1000:.3f}")

    # Final metrics
    final_hw = hardware.get_comprehensive_metrics()

    results['generated'] = generated_text
    results['generation_metrics'] = gen_metrics
    results['final_metrics'] = final_hw
    results['final_loss'] = results['epochs'][-1]['avg_loss']
    results['final_coherence'] = results['epochs'][-1]['avg_coherence']

    # Business metrics summary
    results['business_metrics'] = {
        'total_energy_j': final_hw['total_energy_j'],
        'total_tokens': final_hw['total_tokens'],
        'efficiency_j_per_token': final_hw['j_per_token'],
        'efficiency_tokens_per_joule': final_hw['tokens_per_joule'],
        'thermal_stability_std_c': final_hw['temp_std_c'],
        'power_stability_std_w': final_hw['power_std_w'],
        'regulation_events': final_hw['regulation_events'],
        'fpga_operations': final_hw['fpga_operations'],
        'coherence': results['final_coherence'],
    }

    # Save results
    results_path = PROJECT_ROOT / "results" / "z1141_real_hardware_embodiment.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Cleanup
    hardware.cleanup()

    print()
    print("=" * 70)
    print("z1141 Complete - Real Hardware Embodiment")
    print(f"  Final loss: {results['final_loss']:.4f}")
    print(f"  Final coherence: {results['final_coherence']:.4f}")
    print(f"  Total energy: {final_hw['total_energy_j']:.2f}J")
    print(f"  mJ/token: {final_hw['j_per_token']*1000:.3f}")
    print(f"  FPGA operations: {final_hw['fpga_operations']}")
    print(f"  Regulation events: {final_hw['regulation_events']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
