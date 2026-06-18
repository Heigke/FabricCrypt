#!/usr/bin/env python3
"""
z1176: FPGA Decay + GPU Telemetry Integration

Integrates:
- FPGA DDR3 decay (no-refresh bitstream) for persistent state
- GPU telemetry (sysfs hwmon) for energy tracking
- Forward-Forward learning with embodied awareness

The decay provides natural "forgetting" - old activations decay to zero.
This creates temporal attention where recent states matter more.

Architecture:
1. GPU: Runs encoder layers, measures energy via sysfs
2. FPGA: Stores activation history in DDR3 (decays without refresh)
3. Decay level feeds back into attention weights
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'venv/lib/python3.12/site-packages'))

# HSA override for gfx1151
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# Import GPU telemetry
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyAccumulator

# FPGA imports
try:
    from litex.tools.litex_client import RemoteClient
    FPGA_AVAILABLE = True
except ImportError:
    FPGA_AVAILABLE = False
    print("Warning: LiteX client not available")


@dataclass
class Config:
    hidden_dim: int = 256
    n_layers: int = 4
    batch_size: int = 32
    seq_len: int = 64
    n_epochs: int = 10
    learning_rate: float = 0.001
    fpga_ip: str = "192.168.0.50"


class FPGADecayState:
    """Manages FPGA DDR3 as decaying activation memory"""

    DDR3_BASE = 0x40000000
    SDRAM_BASE = 0x3000
    DFII_CONTROL = SDRAM_BASE + 0x00
    HW_MODE = 0x0B
    SW_MODE = 0x0A
    CMD_REFRESH = 0x0D
    CMD_PRECHARGE = 0x0B

    def __init__(self, ip="192.168.0.50"):
        self.wb = None
        self.connected = False
        self.activation_base = self.DDR3_BASE + 0x1000000
        self.state_base = self.DDR3_BASE + 0x2000000

        try:
            import socket
            socket.setdefaulttimeout(5)  # 5 second timeout
            self.wb = RemoteClient(host=ip, port=1234)
            self.wb.open()
            self._init_ddr3()
            self.connected = True
            print(f"FPGA connected at {ip}:1234")
        except Exception as e:
            print(f"FPGA connection failed: {e}")
            print("(Continuing without FPGA - GPU-only mode)")

    def _init_ddr3(self):
        """Initialize DDR3 with proper DFII settings"""
        self.wb.write(self.DFII_CONTROL, self.HW_MODE)
        time.sleep(0.1)
        for _ in range(10):
            self._issue_refresh()

    def _issue_refresh(self):
        """Issue manual refresh"""
        self.wb.write(self.DFII_CONTROL, self.SW_MODE)
        time.sleep(0.0001)
        self.wb.write(self.SDRAM_BASE + 0x0c, 0x400)
        self.wb.write(self.SDRAM_BASE + 0x10, 0)
        self.wb.write(self.SDRAM_BASE + 0x04, self.CMD_PRECHARGE)
        self.wb.write(self.SDRAM_BASE + 0x08, 1)
        time.sleep(0.0001)
        self.wb.write(self.SDRAM_BASE + 0x04, self.CMD_REFRESH)
        self.wb.write(self.SDRAM_BASE + 0x08, 1)
        time.sleep(0.0001)
        self.wb.write(self.DFII_CONTROL, self.HW_MODE)

    def _evict_cache(self):
        """Evict L2 cache to enable decay observation"""
        evict_base = self.DDR3_BASE + 0x4000000
        for i in range(4096):
            self.wb.write(evict_base + i * 4, 0xDEADC0DE)

    def write_activation(self, layer_idx: int, values: torch.Tensor):
        """Write activation pattern to DDR3"""
        if not self.connected:
            return

        # Quantize to 32-bit pattern
        flat = values.flatten()[:32].cpu().numpy()
        binary = (flat > 0.5).astype(np.uint32)
        pattern = sum(int(b) << i for i, b in enumerate(binary))

        addr = self.activation_base + layer_idx * 4
        self.wb.write(addr, pattern)

    def read_decay_level(self, layer_idx: int) -> float:
        """Read decay level (0=decayed, 1=fresh)"""
        if not self.connected:
            return 0.5

        addr = self.activation_base + layer_idx * 4
        value = self.wb.read(addr)
        ones = bin(value).count('1')
        return ones / 32.0

    def refresh_all(self):
        """Refresh memory to prevent decay"""
        if self.connected:
            self._issue_refresh()

    def force_decay(self, wait_ms: float = 10):
        """Force cache eviction and wait for decay"""
        if self.connected:
            self._evict_cache()
            time.sleep(wait_ms / 1000)

    def close(self):
        if self.wb:
            self.wb.close()


class EmbodiedEncoder(nn.Module):
    """Encoder with decay-aware attention"""

    def __init__(self, config: Config, n_layers: int = 4):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(n_layers)
        ])
        # Decay modulation - scales layer output based on decay level
        self.decay_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1))
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, decay_levels: list = None):
        """Forward with optional decay modulation"""
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h = F.relu(layer(x))
            h = norm(h)

            # Modulate by decay level
            if decay_levels is not None and i < len(decay_levels):
                decay = decay_levels[i]
                scale = self.decay_scales[i] * (0.5 + 0.5 * decay)
                h = h * scale

            x = x + h

        return x


class ForwardForwardLoss(nn.Module):
    """Forward-Forward learning loss"""

    def __init__(self, threshold: float = 2.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor):
        """Compute FF loss"""
        pos_loss = F.softplus(self.threshold - pos_goodness).mean()
        neg_loss = F.softplus(neg_goodness - self.threshold).mean()
        return pos_loss + neg_loss


def main():
    print("=" * 60)
    print("z1176: FPGA Decay + GPU Telemetry Integration")
    print("=" * 60)

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {
        "experiment": "z1176_decay_gpu_integration",
        "timestamp": datetime.now().isoformat(),
        "epochs": []
    }

    # Initialize systems
    telemetry = SysfsHwmonTelemetry()
    energy_acc = EnergyAccumulator()
    fpga = FPGADecayState(config.fpga_ip)

    # Model
    model = EmbodiedEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    ff_loss = ForwardForwardLoss()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"FPGA connected: {fpga.connected}")

    # Training
    print("\n=== Training with Decay Integration ===")

    for epoch in range(config.n_epochs):
        epoch_loss = 0.0
        epoch_energy = 0.0
        epoch_decay_sum = 0.0
        n_batches = 10

        for batch in range(n_batches):
            # Start energy measurement
            sample = telemetry.read_sample()
            energy_acc.add_sample(sample)
            start_energy = energy_acc.total_energy_j
            start_time = time.time()

            # Generate synthetic data
            pos_data = torch.randn(config.batch_size, config.seq_len, config.hidden_dim).to(device)
            neg_data = torch.randn(config.batch_size, config.seq_len, config.hidden_dim).to(device)

            # Read decay levels from FPGA
            decay_levels = []
            if fpga.connected:
                for i in range(4):
                    decay_levels.append(fpga.read_decay_level(i))

            # Forward pass with decay modulation
            optimizer.zero_grad()
            pos_out = model(pos_data, decay_levels)
            neg_out = model(neg_data, decay_levels)

            # Compute goodness (sum of squared activations)
            pos_goodness = (pos_out ** 2).sum(dim=-1).mean(dim=-1)
            neg_goodness = (neg_out ** 2).sum(dim=-1).mean(dim=-1)

            # Loss
            loss = ff_loss(pos_goodness, neg_goodness)
            loss.backward()
            optimizer.step()

            # Measure energy
            sample = telemetry.read_sample()
            energy_acc.add_sample(sample)
            end_energy = energy_acc.total_energy_j
            batch_energy = end_energy - start_energy
            batch_time = time.time() - start_time

            epoch_loss += loss.item()
            epoch_energy += batch_energy
            epoch_decay_sum += sum(decay_levels) if decay_levels else 0

            # Store activations in FPGA (will decay)
            if fpga.connected:
                for i in range(4):
                    # Store summary activation
                    summary = pos_out.mean(dim=[0, 1])
                    fpga.write_activation(i, summary)

            # Every 5 batches, force decay (don't refresh)
            if batch % 5 == 0 and fpga.connected:
                fpga.force_decay(wait_ms=5)
            elif fpga.connected:
                fpga.refresh_all()  # Maintain memory

        avg_loss = epoch_loss / n_batches
        avg_decay = epoch_decay_sum / (n_batches * 4) if fpga.connected else 0
        tokens = config.batch_size * config.seq_len * n_batches
        tok_per_j = tokens / max(epoch_energy, 0.001)

        print(f"Epoch {epoch+1}/{config.n_epochs}: "
              f"Loss={avg_loss:.4f}, "
              f"Energy={epoch_energy:.3f}J, "
              f"Tok/J={tok_per_j:.1f}, "
              f"AvgDecay={avg_decay:.2f}")

        results["epochs"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "energy_j": epoch_energy,
            "tokens_per_j": tok_per_j,
            "avg_decay": avg_decay
        })

    # Final decay test
    print("\n=== Final Decay Observation ===")
    if fpga.connected:
        # Write fresh patterns
        fpga.refresh_all()
        for i in range(4):
            fpga.write_activation(i, torch.ones(32))

        print("Written fresh patterns, forcing decay...")
        fpga.force_decay(wait_ms=10)

        # Read decay levels
        final_decay = []
        for i in range(4):
            level = fpga.read_decay_level(i)
            final_decay.append(level)
            print(f"  Layer {i}: {level:.2f} charge remaining")

        results["final_decay_levels"] = final_decay

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results["epochs"]:
        final = results["epochs"][-1]
        print(f"Final Loss: {final['loss']:.4f}")
        print(f"Energy Efficiency: {final['tokens_per_j']:.1f} tok/J")
        if fpga.connected:
            print(f"Decay Integration: Active")
            print(f"Final Decay Level: {sum(final_decay)/4:.2f}")

    results["conclusion"] = "decay_gpu_integrated"

    # Save
    results_path = PROJECT_ROOT / 'results/z1176_decay_gpu_integration.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Cleanup
    fpga.close()
    print("Done!")


if __name__ == "__main__":
    main()
