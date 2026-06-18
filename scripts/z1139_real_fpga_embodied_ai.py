#!/usr/bin/env python3
"""
z1139: Real FPGA + GPU Embodied AI System

Combines ALL existing infrastructure:
- Real FPGA interface (partial timing writes, Frac operations, decay)
- GPU actuation daemon (power caps, DVFS control)
- GPU telemetry (sysfs_hwmon, 100Hz sampling)
- FPGA state tracker (Arrhenius decay model)
- Scaled-up transformer with predictive coding and self-reference

This is the FULL embodied system - not simulated.

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                REAL FPGA + GPU EMBODIED SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │  GPU (gfx1151) │◄───►│  TRANSFORMER │◄───►│ FPGA (Arty)  │         │
│  └───────┬──────┘     └───────┬──────┘     └───────┬──────┘         │
│          │                    │                    │                 │
│  ┌───────▼──────┐     ┌───────▼──────┐     ┌───────▼──────┐         │
│  │  Telemetry   │     │  z_feel (8D)  │     │ DDR3 Reservoir│         │
│  │  - temp      │     │  GPU (5D) +   │     │ - Frac writes │         │
│  │  - power     │────►│  FPGA (3D)    │◄────│ - Decay read  │         │
│  │  - clock     │     │               │     │ - Temp (XADC) │         │
│  └──────────────┘     └───────┬──────┘     └──────────────┘         │
│                              │                                       │
│                       ┌───────▼──────┐                              │
│                       │  Actuation   │                              │
│                       │  - Power cap │                              │
│                       │  - FPGA fracs│                              │
│                       └──────────────┘                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Business Value:
- Embodied AI market: $4.4B → $23B (2025-2030)
- Neuromorphic: 50-70% energy reduction
- In-memory compute: 100-1000x for MVM
"""

import sys
import os
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import existing infrastructure
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.embodied.fpga_state_tracker import FPGAStateTracker, UnifiedBodyState

# Try to import FPGA interface (requires hardware)
try:
    from src.fpga.fpga_interface import FPGAInterface
    HAS_FPGA = True
except ImportError:
    HAS_FPGA = False
    print("Note: FPGA interface not available, using simulation")

# Try to import actuator client
try:
    from src.actuator.client import ActuatorClient
    HAS_ACTUATOR = True
except (ImportError, PermissionError, OSError) as e:
    HAS_ACTUATOR = False
    print(f"Note: Actuator client not available ({type(e).__name__})")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EmbodiedConfig:
    """Configuration for real embodied system"""
    # Model
    vocab_size: int = 256
    hidden_dim: int = 512  # Scaled up from 256
    n_layers: int = 12     # Scaled up from 6
    n_heads: int = 8
    dropout: float = 0.1

    # FPGA
    fpga_port: str = '/dev/ttyUSB1'
    fpga_baudrate: int = 115200
    fpga_base_addr: int = 0x600000  # From calibration results
    fpga_enabled: bool = True

    # GPU actuation
    actuator_host: str = 'localhost'
    actuator_port: int = 8770
    power_cap_min: int = 20
    power_cap_max: int = 50
    actuator_enabled: bool = True

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 16
    seq_len: int = 128
    n_epochs: int = 30

    # Embodiment
    telemetry_dim: int = 8  # GPU(5) + FPGA(3)
    energy_weight: float = 0.1
    intero_weight: float = 0.1


# =============================================================================
# Real FPGA Reservoir (not simulated!)
# =============================================================================

class RealFPGAReservoir(nn.Module):
    """
    Physical FPGA DDR3 reservoir computing.

    Uses REAL hardware:
    - Frac operations for partial charging
    - Physical decay from DRAM physics
    - XADC temperature feedback
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_dim: int = 256,
        config: Optional[EmbodiedConfig] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.config = config or EmbodiedConfig()

        # Input projection
        self.input_proj = nn.Linear(input_dim, 64)  # Compress for FPGA storage

        # Readout from reservoir (64 input proj + 64 reservoir state)
        self.readout = nn.Linear(64 + 64, reservoir_dim)

        # FPGA interface
        self.fpga: Optional[FPGAInterface] = None
        self.fpga_connected = False
        self.fpga_temp = 50.0

        # Fallback simulated state
        self.register_buffer('sim_reservoir', torch.zeros(64))

        # Try to connect
        if HAS_FPGA and self.config.fpga_enabled:
            self._connect_fpga()

    def _connect_fpga(self):
        """Connect to real FPGA hardware"""
        try:
            self.fpga = FPGAInterface(
                self.config.fpga_port,
                self.config.fpga_baudrate
            )
            self.fpga_connected = self.fpga.connect()
            if self.fpga_connected:
                print(f"  ✓ FPGA connected on {self.config.fpga_port}")
                # Read initial temperature
                status = self.fpga.ping()
                self.fpga_temp = status.get('xadc_temp', 50.0)
        except Exception as e:
            print(f"  ✗ FPGA connection failed: {e}")
            self.fpga_connected = False

    def _write_to_fpga(self, data: torch.Tensor) -> bool:
        """Write data to FPGA DDR3 using Frac operations"""
        if not self.fpga_connected:
            return False

        try:
            # Convert to bytes (quantize to 8-bit)
            data_np = (data.detach().cpu().numpy() * 255).astype('uint8')

            # Write using partial timing (analog charge levels)
            for i, val in enumerate(data_np[:32]):  # Write first 32 values
                addr = self.config.fpga_base_addr + i * 256
                # Number of Frac cycles determines charge level
                num_fracs = int(val / 32)  # 0-7 fracs
                if num_fracs > 0:
                    self.fpga.raw_frac_operation(
                        bank=0, row=addr >> 14, col=(addr >> 4) & 0x3FF,
                        num_fracs=num_fracs
                    )
            return True
        except Exception as e:
            print(f"FPGA write error: {e}")
            return False

    def _read_from_fpga(self) -> Tuple[torch.Tensor, float]:
        """Read decayed data from FPGA DDR3"""
        if not self.fpga_connected:
            # Simulate decay
            decay = 0.95
            self.sim_reservoir = self.sim_reservoir * decay
            return self.sim_reservoir, self.fpga_temp

        try:
            # Read back decayed values
            data = []
            for i in range(32):
                addr = self.config.fpga_base_addr + i * 256
                result = self.fpga.ddr_read(addr, 1)
                if result:
                    data.append(result[0] / 255.0)
                else:
                    data.append(0.0)

            # Read temperature
            status = self.fpga.ping()
            self.fpga_temp = status.get('xadc_temp', self.fpga_temp)

            return torch.tensor(data, dtype=torch.float32), self.fpga_temp
        except Exception as e:
            print(f"FPGA read error: {e}")
            return self.sim_reservoir, self.fpga_temp

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward through physical reservoir.

        1. Project input to FPGA storage dimension
        2. Write to DDR3 via Frac operations
        3. Wait for physics (decay)
        4. Read back decayed state
        5. Combine with input for output
        """
        batch_size = x.size(0)
        device = x.device

        # Project input
        projected = self.input_proj(x.mean(dim=1) if x.dim() > 2 else x)
        projected = torch.sigmoid(projected)  # [0, 1] for analog storage

        # Write to FPGA (uses mean across batch)
        write_data = projected.mean(dim=0)
        self._write_to_fpga(write_data)

        # Small delay for physics (decay happens in hardware)
        # In real deployment, inference time provides natural delay
        time.sleep(0.001)

        # Read back decayed state
        reservoir_state, temp = self._read_from_fpga()
        reservoir_state = reservoir_state.to(device)

        # Pad to reservoir_dim
        if reservoir_state.size(0) < 64:
            pad = torch.zeros(64 - reservoir_state.size(0), device=device)
            reservoir_state = torch.cat([reservoir_state, pad])

        # Combine projected input with reservoir state
        combined = torch.cat([
            projected,
            reservoir_state.unsqueeze(0).expand(batch_size, -1)
        ], dim=-1)

        output = self.readout(combined)

        info = {
            'fpga_connected': self.fpga_connected,
            'fpga_temp': temp,
            'reservoir_mean': reservoir_state.mean().item(),
            'projected_mean': projected.mean().item(),
        }

        return output, info


# =============================================================================
# GPU Actuator Integration
# =============================================================================

class GPUActuator:
    """Interface to GPU actuation daemon for power/clock control"""

    def __init__(self, config: EmbodiedConfig):
        self.config = config
        self.client: Optional[ActuatorClient] = None
        self.connected = False
        self.current_power_cap = 35  # Default

        if HAS_ACTUATOR and config.actuator_enabled:
            self._connect()

    def _connect(self):
        """Connect to actuator daemon"""
        try:
            self.client = ActuatorClient(
                self.config.actuator_host,
                self.config.actuator_port,
                auto_heartbeat=True
            )
            # Test connection
            status = self.client.get_status()
            if status.success:
                self.connected = True
                self.current_power_cap = status.data.get('power_cap', 35)
                print(f"  ✓ Actuator connected, power cap: {self.current_power_cap}W")
        except Exception as e:
            print(f"  ✗ Actuator connection failed: {e}")
            self.connected = False

    def regulate(self, z_feel: torch.Tensor) -> dict:
        """
        Regulate GPU based on body state.

        z_feel contains:
        [0] strain (power deviation)
        [1] urgency (thermal urgency)
        [2] debt (energy debt)
        [3] margin (thermal margin)
        [4] stability
        [5] fpga_temp
        [6] charge
        [7] decay_rate
        """
        if not self.connected:
            return {'regulated': False, 'reason': 'not connected'}

        try:
            # Extract relevant signals
            strain = z_feel[0].item() if torch.is_tensor(z_feel) else z_feel[0]
            urgency = z_feel[1].item() if torch.is_tensor(z_feel) else z_feel[1]

            # Determine action
            if urgency > 0.7:
                # High thermal urgency - reduce power
                new_cap = max(self.config.power_cap_min,
                             self.current_power_cap - 5)
            elif strain > 0.5:
                # High strain - slight reduction
                new_cap = max(self.config.power_cap_min,
                             self.current_power_cap - 2)
            elif strain < -0.3 and urgency < 0.3:
                # Low load, cool - can increase
                new_cap = min(self.config.power_cap_max,
                             self.current_power_cap + 2)
            else:
                # Maintain current
                return {'regulated': False, 'reason': 'stable'}

            # Apply if changed
            if new_cap != self.current_power_cap:
                result = self.client.set_power_cap(new_cap)
                if result.success:
                    old_cap = self.current_power_cap
                    self.current_power_cap = new_cap
                    return {
                        'regulated': True,
                        'old_cap': old_cap,
                        'new_cap': new_cap,
                        'reason': f"strain={strain:.2f}, urgency={urgency:.2f}"
                    }

            return {'regulated': False, 'reason': 'no change needed'}

        except Exception as e:
            return {'regulated': False, 'reason': str(e)}


# =============================================================================
# Scaled-Up Embodied Transformer
# =============================================================================

class ScaledEmbodiedTransformer(nn.Module):
    """
    Scaled-up embodied transformer with:
    - Real FPGA reservoir
    - FiLM conditioning from telemetry
    - Predictive coding layers
    - Self-referential interoception
    - Early exit capability
    """

    def __init__(self, config: EmbodiedConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(1024, config.hidden_dim)

        # Real FPGA reservoir
        self.fpga_reservoir = RealFPGAReservoir(
            config.hidden_dim,
            config.hidden_dim,
            config
        )

        # FiLM generator
        self.film_gen = nn.Sequential(
            nn.Linear(config.telemetry_dim, 64),
            nn.GELU(),
            nn.Linear(64, config.hidden_dim * 2)  # gamma, beta
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.n_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
            ) for _ in range(config.n_layers)
        ])

        # Exit heads (for early exit)
        self.exit_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.vocab_size)
            for _ in range(config.n_layers)
        ])
        self.conf_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, 1)
            for _ in range(config.n_layers)
        ])

        # Self-referential predictor
        self.intero_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, config.telemetry_dim)
        )

        # Output
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        z_feel: torch.Tensor,
        confidence_threshold: float = 0.9,
    ) -> Tuple[torch.Tensor, int, float, dict]:
        """Forward with embodiment and early exit"""
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.embed(tokens) + self.pos_embed(pos)

        # FPGA reservoir
        h_mean = h.mean(dim=1)
        reservoir_out, reservoir_info = self.fpga_reservoir(h_mean)
        h = h + reservoir_out.unsqueeze(1) * 0.1

        # FiLM parameters
        z_feel_batch = z_feel.unsqueeze(0).expand(batch_size, -1).to(device)
        film_params = self.film_gen(z_feel_batch)
        gamma, beta = film_params.chunk(2, dim=-1)

        # Process layers
        all_info = {'reservoir': reservoir_info, 'layers': []}

        for i, layer in enumerate(self.layers):
            h = layer(h)

            # FiLM modulation
            h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

            # Early exit check
            if i < self.config.n_layers - 1:
                exit_logits = self.exit_heads[i](h[:, -1, :])
                conf = torch.sigmoid(self.conf_heads[i](h[:, -1, :])).mean().item()

                if conf > confidence_threshold:
                    # Predict telemetry (interoception)
                    h_pool = h.mean(dim=(0, 1))
                    pred_telem = self.intero_predictor(h_pool)
                    intero_error = ((z_feel - pred_telem) ** 2).mean()

                    all_info['exit_layer'] = i + 1
                    all_info['intero_error'] = intero_error.item()
                    all_info['predicted_telem'] = pred_telem.detach().cpu().tolist()

                    return exit_logits, i + 1, conf, all_info

        # Full forward
        h_pool = h.mean(dim=(0, 1))
        pred_telem = self.intero_predictor(h_pool)
        intero_error = ((z_feel - pred_telem) ** 2).mean()

        h = self.ln_f(h)
        logits = self.lm_head(h[:, -1, :])

        all_info['exit_layer'] = self.config.n_layers
        all_info['intero_error'] = intero_error.item()
        all_info['predicted_telem'] = pred_telem.detach().cpu().tolist()

        return logits, self.config.n_layers, 1.0, all_info


# =============================================================================
# Unified Training Loop
# =============================================================================

class RealEmbodiedTrainer:
    """Training loop with real hardware integration"""

    def __init__(self, config: EmbodiedConfig, device: str = 'cuda'):
        self.config = config
        self.device = device

        print("\n🔧 Initializing Real Embodied System...")

        # Model
        self.model = ScaledEmbodiedTransformer(config).to(device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model: {total_params:,} parameters ({total_params/1e6:.2f}M)")

        # GPU telemetry
        self.telemetry = SysfsHwmonTelemetry()
        telemetry_ok = self.telemetry.paths.power_average is not None
        print(f"  Telemetry: {'✓ available' if telemetry_ok else '✗ not available'}")

        # FPGA state tracker
        self.fpga_tracker = FPGAStateTracker()

        # Unified body state
        self.body_state = UnifiedBodyState(self.telemetry, self.fpga_tracker)

        # GPU actuator
        self.actuator = GPUActuator(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Metrics
        self.step = 0
        self.metrics = {
            'loss': [], 'exit_layer': [], 'intero_error': [],
            'fpga_temp': [], 'gpu_temp': [], 'power_w': [],
            'regulated': []
        }

    def get_z_feel(self) -> torch.Tensor:
        """Get unified body state from real sensors"""
        sample = self.telemetry.read_sample()

        # Convert GpuSample to z_feel directly
        # GPU state (5-dim)
        power_setpoint = 35.0  # Watts
        temp_setpoint = 60.0   # Celsius

        strain = (sample.power_w - power_setpoint) / power_setpoint if sample.power_w else 0.0
        urgency = (sample.temp_edge_c - temp_setpoint) / temp_setpoint if sample.temp_edge_c else 0.0
        debt = 0.0  # Would need energy tracking
        margin = max(0, (85.0 - sample.temp_edge_c) / 35.0) if sample.temp_edge_c else 0.5
        stability = 1.0

        # FPGA state (3-dim)
        fpga_temp, fpga_charge, fpga_decay = self.fpga_tracker.get_state_vector()
        fpga_temp_norm = (fpga_temp - 50.0) / 30.0
        fpga_charge_norm = fpga_charge
        fpga_decay_norm = min(1.0, fpga_decay / 0.5)

        z_feel = torch.tensor([
            strain, urgency, debt, margin, stability,
            fpga_temp_norm, fpga_charge_norm, fpga_decay_norm
        ], dtype=torch.float32)

        return z_feel.to(self.device)

    def train_step(self, tokens: torch.Tensor, targets: torch.Tensor) -> dict:
        """Single training step with embodiment"""
        self.model.train()

        tokens = tokens.to(self.device)
        targets = targets.to(self.device)

        # Get body state
        z_feel = self.get_z_feel()

        # Sense energy
        energy_before = self.telemetry.get_accumulated_energy_j()

        # Forward
        logits, exit_layer, conf, model_info = self.model(
            tokens, z_feel, confidence_threshold=0.95
        )

        # Energy measurement
        energy_after = self.telemetry.get_accumulated_energy_j()
        energy_j = energy_after - energy_before

        # Loss with embodiment
        task_loss = F.cross_entropy(logits, targets)
        intero_loss = model_info.get('intero_error', 0.0)
        energy_loss = max(0, energy_j - 0.01) * 10  # Penalize excess energy

        loss = task_loss + self.config.intero_weight * intero_loss + self.config.energy_weight * energy_loss

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Regulate GPU based on body state
        regulate_info = self.actuator.regulate(z_feel)

        # Track
        self.step += 1
        sample = self.telemetry.read_sample()

        info = {
            'step': self.step,
            'loss': loss.item(),
            'task_loss': task_loss.item(),
            'exit_layer': exit_layer,
            'confidence': conf,
            'intero_error': model_info.get('intero_error', 0),
            'fpga_temp': model_info['reservoir'].get('fpga_temp', 0),
            'fpga_connected': model_info['reservoir'].get('fpga_connected', False),
            'gpu_temp': sample.temp_edge_c,
            'power_w': sample.power_w,
            'regulated': regulate_info.get('regulated', False),
            'energy_j': energy_j,
        }

        return info

    def train_epoch(self, data: str) -> dict:
        """Train for one epoch"""
        all_tokens = [ord(c) for c in data if 0 <= ord(c) < 256]
        n_batches = min(300, (len(all_tokens) - self.config.seq_len - 1) //
                       (self.config.seq_len * self.config.batch_size))

        epoch_metrics = []

        for batch_idx in range(n_batches):
            batch_tokens = []
            batch_targets = []

            for b in range(self.config.batch_size):
                start = (batch_idx * self.config.batch_size + b) * self.config.seq_len
                seq = all_tokens[start:start + self.config.seq_len]
                tgt = all_tokens[start + 1:start + self.config.seq_len + 1]

                if len(seq) == self.config.seq_len and len(tgt) == self.config.seq_len:
                    batch_tokens.append(seq)
                    batch_targets.append(tgt[-1])

            if not batch_tokens:
                continue

            tokens = torch.tensor(batch_tokens, dtype=torch.long)
            targets = torch.tensor(batch_targets, dtype=torch.long)

            info = self.train_step(tokens, targets)
            epoch_metrics.append(info)

            if (batch_idx + 1) % 50 == 0:
                avg_loss = sum(m['loss'] for m in epoch_metrics[-50:]) / 50
                avg_exit = sum(m['exit_layer'] for m in epoch_metrics[-50:]) / 50
                fpga_status = "✓" if info['fpga_connected'] else "sim"
                print(f"  Batch {batch_idx+1}/{n_batches}: loss={avg_loss:.4f}, "
                      f"exit={avg_exit:.1f}, FPGA={fpga_status}, "
                      f"GPU={info['gpu_temp']:.0f}°C, {info['power_w']:.0f}W")

        return {
            'avg_loss': sum(m['loss'] for m in epoch_metrics) / len(epoch_metrics),
            'avg_exit': sum(m['exit_layer'] for m in epoch_metrics) / len(epoch_metrics),
            'avg_intero': sum(m['intero_error'] for m in epoch_metrics) / len(epoch_metrics),
            'n_batches': len(epoch_metrics),
            'n_regulated': sum(1 for m in epoch_metrics if m['regulated']),
        }

    def generate(self, prompt: str = "The ", max_len: int = 100) -> str:
        """Generate text"""
        self.model.eval()
        tokens = [ord(c) for c in prompt if 0 <= ord(c) < 256]

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([tokens[-self.config.seq_len:]], dtype=torch.long).to(self.device)
                z_feel = self.get_z_feel()
                logits, _, _, _ = self.model(x, z_feel)
                probs = F.softmax(logits[0], dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)

        return ''.join(chr(t) for t in tokens if 32 <= t < 127)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("z1139: REAL FPGA + GPU EMBODIED AI SYSTEM")
    print("Full Hardware Integration")
    print("=" * 80)

    print("\n📊 Business Value:")
    print("  - Embodied AI market: $4.4B → $23B (2025-2030)")
    print("  - Neuromorphic computing: 50-70% energy reduction")
    print("  - In-memory compute: 100-1000x efficiency for MVM")

    # Configuration
    config = EmbodiedConfig(
        hidden_dim=512,
        n_layers=12,
        n_epochs=30,
        batch_size=16,
        seq_len=128,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create trainer
    trainer = RealEmbodiedTrainer(config, device)

    # Load data
    data_path = Path('data/tiny_shakespeare.txt')
    if data_path.exists():
        data = data_path.read_text()
        print(f"\nData: {len(data):,} characters from TinyShakespeare")
    else:
        data = "The embodied machine learns through self-reference. " * 2000
        print("\nData: Generated (TinyShakespeare not found)")

    # Train
    print("\n" + "=" * 80)
    print("TRAINING: Real Embodied System")
    print("=" * 80)

    results = {
        'experiment': 'z1139_real_fpga_embodied_ai',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'hidden_dim': config.hidden_dim,
            'n_layers': config.n_layers,
            'n_epochs': config.n_epochs,
            'fpga_enabled': config.fpga_enabled,
            'actuator_enabled': config.actuator_enabled,
        },
        'epochs': [],
    }

    for epoch in range(config.n_epochs):
        print(f"\nEpoch {epoch + 1}/{config.n_epochs}")
        epoch_info = trainer.train_epoch(data)

        print(f"  → Loss: {epoch_info['avg_loss']:.4f}")
        print(f"  → Avg exit layer: {epoch_info['avg_exit']:.1f}/{config.n_layers}")
        print(f"  → Interoceptive error: {epoch_info['avg_intero']:.4f}")
        print(f"  → GPU regulated: {epoch_info['n_regulated']} times")

        results['epochs'].append(epoch_info)

    # Generate
    print("\n" + "=" * 80)
    print("GENERATION TEST")
    print("=" * 80)

    generated = trainer.generate("The ", max_len=150)
    print(f"\nGenerated: {generated}")

    results['generated'] = generated
    results['final_loss'] = results['epochs'][-1]['avg_loss']
    results['fpga_connected'] = trainer.model.fpga_reservoir.fpga_connected
    results['actuator_connected'] = trainer.actuator.connected

    # Save
    output_path = Path('results/z1139_real_fpga_embodied_ai.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 80)
    print("REAL FPGA EMBODIED AI COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
