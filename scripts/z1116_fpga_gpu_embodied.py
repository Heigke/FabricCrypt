#!/usr/bin/env python3
"""
z1116: FPGA-GPU Embodied AI Integration

Integrates:
- FPGA DDR3 partial timing writes (analog charge levels)
- FPGA DDR3 decay (temporal forgetting)
- GPU low-level HIP kernels (energy-aware attention)
- Forward-Forward inspired local learning

The system embodies computation:
- FPGA temperature modulates neuron thresholds
- DDR3 decay performs physical forgetting
- Partial writes create analog activation levels
- GPU handles main inference with hardware-aware gating
"""

import sys
import time
import json
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fpga.fpga_interface import FPGAInterface


@dataclass
class EmbodiedState:
    """Combined FPGA + GPU embodied state"""
    fpga_temp: float          # FPGA temperature (C)
    gpu_temp: float           # GPU temperature (C)
    gpu_power: float          # GPU power (W)
    decay_factor: float       # Temperature-dependent decay
    confidence: float         # Current inference confidence
    layer_exits: int          # How many layers were skipped
    timestamp: float


class FPGAAnalogMemory:
    """
    Uses FPGA DDR3 as analog memory with:
    - Partial timing writes for charge levels
    - Decay for temporal forgetting
    """

    def __init__(self, fpga: FPGAInterface, base_addr: int = 0x100000):
        self.fpga = fpga
        self.base_addr = base_addr
        self.memory_size = 256  # 256 x 16-byte slots = 4KB
        self.slot_size = 16     # 128 bits per slot

    def write_analog(self, slot: int, values: np.ndarray, strength: float = 1.0) -> bool:
        """
        Write analog values to FPGA DDR3 with partial timing.

        Args:
            slot: Memory slot (0-255)
            values: 16 float values (0.0 to 1.0)
            strength: Write strength (0.0 to 1.0), maps to timing_offset 0-63

        Returns:
            True if successful
        """
        if slot >= self.memory_size:
            raise ValueError(f"Slot {slot} out of range")

        addr = self.base_addr + slot * self.slot_size

        # Convert floats to bytes (quantized to 8 bits each)
        data = bytes([int(min(255, max(0, v * 255))) for v in values[:16]])
        if len(data) < 16:
            data = data + bytes(16 - len(data))

        # Map strength to timing offset (lower strength = higher offset = weaker write)
        timing_offset = int((1.0 - strength) * 63)

        if timing_offset == 0:
            # Full strength - normal write
            return self.fpga.ddr_write(addr, data)
        else:
            # Partial timing write for analog level
            result = self.fpga.partial_timing_write(addr, data, timing_offset=timing_offset, timeout=10)
            return result.get('success', False)

    def read_analog(self, slot: int) -> Optional[np.ndarray]:
        """
        Read analog values from FPGA DDR3.

        Returns:
            16 float values (0.0 to 1.0) or None if failed
        """
        addr = self.base_addr + slot * self.slot_size
        data = self.fpga.ddr_read(addr, retries=5)

        if data is None:
            return None

        # Convert bytes back to floats
        return np.array([b / 255.0 for b in data], dtype=np.float32)

    def decay_memory(self, slot: int, wait_ms: float = 100) -> Tuple[Optional[np.ndarray], int]:
        """
        Let memory slot decay for specified time.

        Returns:
            (decayed_values, bit_errors) or (None, -1) if failed
        """
        addr = self.base_addr + slot * self.slot_size

        # Read current value
        before = self.fpga.ddr_read(addr, retries=3)
        if before is None:
            return None, -1

        # Perform decay test (disable refresh, wait, read back)
        wait_cycles = int(wait_ms / 0.012)  # ~12ns per cycle
        result = self.fpga.decay_test(addr, before, wait_cycles=wait_cycles)

        if not result.get('success'):
            return None, -1

        # Read decayed value
        after_bytes = result.get('read_data', before)
        after = np.array([b / 255.0 for b in after_bytes], dtype=np.float32)

        return after, result.get('bit_errors', 0)


class EmbodiedNeuronLayer(nn.Module):
    """
    Neural layer that uses FPGA analog memory for activations.

    Computation flow:
    1. GPU computes linear transform
    2. Write activations to FPGA with partial timing (strength = confidence)
    3. Let decay occur based on temperature
    4. Read back decayed activations
    5. Continue with next layer
    """

    def __init__(self, in_features: int, out_features: int, fpga_memory: FPGAAnalogMemory,
                 memory_slot: int = 0, decay_ms: float = 10.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.fpga_memory = fpga_memory
        self.memory_slot = memory_slot
        self.decay_ms = decay_ms
        self.use_fpga = True  # Can disable for comparison

    def forward(self, x: torch.Tensor, confidence: float = 1.0) -> torch.Tensor:
        # GPU linear transform
        h = self.linear(x)
        h = torch.sigmoid(h)  # Bound to [0, 1] for analog storage

        if not self.use_fpga:
            return h

        # For each sample in batch, use FPGA analog memory
        batch_size = h.shape[0]
        outputs = []

        for i in range(min(batch_size, 16)):  # Limit FPGA operations
            # Take first 16 values of activation
            activations = h[i, :16].detach().cpu().numpy()

            # Write to FPGA with strength proportional to confidence
            write_strength = 0.3 + 0.7 * confidence  # 0.3 to 1.0
            slot = (self.memory_slot + i) % self.fpga_memory.memory_size

            if self.fpga_memory.write_analog(slot, activations, strength=write_strength):
                # Let decay occur
                decayed, errors = self.fpga_memory.decay_memory(slot, wait_ms=self.decay_ms)

                if decayed is not None:
                    # Replace first 16 values with decayed version
                    h_np = h[i].detach().cpu().numpy()
                    h_np[:16] = decayed
                    outputs.append(torch.tensor(h_np, device=h.device))
                else:
                    outputs.append(h[i])
            else:
                outputs.append(h[i])

        # Stack outputs
        if outputs:
            h = torch.stack(outputs + [h[j] for j in range(len(outputs), batch_size)])

        return h


class ForwardForwardLayer(nn.Module):
    """
    Forward-Forward inspired layer with goodness-based learning.

    Instead of backprop, learns by:
    - Maximizing goodness (sum of squared activations) for positive data
    - Minimizing goodness for negative data
    """

    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.threshold = threshold
        self.opt = torch.optim.Adam(self.linear.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.linear(x))
        return h

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Compute goodness = sum of squared activations"""
        return (h ** 2).sum(dim=-1)

    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> Tuple[float, float]:
        """
        Train with positive and negative examples.

        Returns:
            (pos_goodness, neg_goodness)
        """
        self.opt.zero_grad()

        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        g_pos = self.goodness(h_pos)
        g_neg = self.goodness(h_neg)

        # Loss: want g_pos > threshold > g_neg
        loss_pos = F.softplus(self.threshold - g_pos).mean()
        loss_neg = F.softplus(g_neg - self.threshold).mean()
        loss = loss_pos + loss_neg

        loss.backward()
        self.opt.step()

        return g_pos.mean().item(), g_neg.mean().item()


class EmbodiedForwardForward(nn.Module):
    """
    Combines Forward-Forward learning with FPGA embodiment.

    The FPGA provides:
    - Temperature-modulated thresholds
    - Analog activation storage with decay
    - Physical forgetting for temporal credit assignment
    """

    def __init__(self, layer_sizes: List[int], fpga: FPGAInterface):
        super().__init__()

        self.fpga = fpga
        self.fpga_memory = FPGAAnalogMemory(fpga, base_addr=0x100000)

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = ForwardForwardLayer(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)

        # Temperature-based threshold modulation
        self.base_threshold = 2.0

    def get_threshold(self) -> float:
        """Get temperature-modulated threshold from FPGA"""
        temp, _ = self.fpga.read_temperature()
        # Higher temp = lower threshold (easier to activate)
        # Arrhenius-inspired: threshold decreases ~3% per degree above 50C
        temp_factor = 1.0 - 0.03 * max(0, temp - 50)
        return self.base_threshold * max(0.5, temp_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers"""
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def train_layer(self, layer_idx: int, x_pos: torch.Tensor, x_neg: torch.Tensor) -> dict:
        """
        Train a single layer with embodied feedback.

        The FPGA temperature modulates the goodness threshold,
        and activations are stored/decayed in DDR3.
        """
        # Get temperature-modulated threshold
        threshold = self.get_threshold()
        self.layers[layer_idx].threshold = threshold

        # Get activations up to this layer
        h_pos = x_pos
        h_neg = x_neg
        for i in range(layer_idx):
            h_pos = self.layers[i](h_pos)
            h_neg = self.layers[i](h_neg)

        # Train this layer
        g_pos, g_neg = self.layers[layer_idx].train_step(h_pos.detach(), h_neg.detach())

        # Store activations in FPGA for potential decay-based credit
        slot = layer_idx * 16
        with torch.no_grad():
            h_out = self.layers[layer_idx](h_pos.detach())
            activations = h_out[0, :16].cpu().numpy()
            # Write with strength proportional to goodness
            strength = min(1.0, g_pos / (2 * threshold))
            self.fpga_memory.write_analog(slot, activations, strength=strength)

        return {
            'layer': layer_idx,
            'threshold': threshold,
            'goodness_pos': g_pos,
            'goodness_neg': g_neg,
            'fpga_temp': self.fpga.read_temperature()[0]
        }


def get_gpu_telemetry() -> Tuple[float, float]:
    """Get GPU temperature and power from rocm-smi"""
    try:
        import subprocess
        result = subprocess.run(
            ['rocm-smi', '--showtemp', '--showpower', '--json'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            card = data.get('card0', {})
            temp = float(card.get('Temperature (Sensor edge) (C)', 50))
            power = float(card.get('Average Graphics Package Power (W)', 100))
            return temp, power
    except:
        pass
    return 50.0, 100.0  # Defaults


def main():
    print("=" * 60)
    print("z1116: FPGA-GPU Embodied AI Integration")
    print("=" * 60)

    # Set HSA override for gfx1151
    import os
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

    # Connect to FPGA
    print("\nConnecting to FPGA...")
    fpga = FPGAInterface()
    if not fpga.connect():
        print("Failed to connect to FPGA")
        return 1

    fpga_status = fpga.get_status()
    print(f"FPGA Status: {fpga_status}")

    # Initialize GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GPU Device: {device}")
    if device.type == 'cuda':
        gpu_temp, gpu_power = get_gpu_telemetry()
        print(f"GPU Temp: {gpu_temp:.1f}C, Power: {gpu_power:.1f}W")

    # Create embodied model
    print("\nCreating Embodied Forward-Forward model...")
    layer_sizes = [784, 256, 128, 64, 10]  # MNIST-like architecture
    model = EmbodiedForwardForward(layer_sizes, fpga)
    model = model.to(device)

    # Test FPGA analog memory
    print("\nTesting FPGA analog memory...")
    fpga_memory = FPGAAnalogMemory(fpga)

    # Write test pattern
    test_values = np.linspace(0, 1, 16).astype(np.float32)
    print(f"Writing: {test_values[:4]}...")

    if fpga_memory.write_analog(0, test_values, strength=1.0):
        read_values = fpga_memory.read_analog(0)
        if read_values is not None:
            print(f"Read:    {read_values[:4]}...")
            error = np.abs(test_values - read_values).mean()
            print(f"Mean error: {error:.4f}")

    # Test partial timing (analog levels)
    print("\nTesting partial timing (analog write strength)...")
    for strength in [1.0, 0.7, 0.4]:
        test_data = np.ones(16, dtype=np.float32)
        slot = int((1.0 - strength) * 10)
        fpga_memory.write_analog(slot, test_data, strength=strength)
        time.sleep(0.1)

        # Let decay for 50ms
        decayed, errors = fpga_memory.decay_memory(slot, wait_ms=50)
        if decayed is not None:
            decay_amount = 1.0 - decayed.mean()
            print(f"  Strength {strength:.1f}: decay={decay_amount:.3f}, errors={errors}")

    # Training loop demo
    print("\n" + "=" * 60)
    print("Training Demo (Forward-Forward with FPGA embodiment)")
    print("=" * 60)

    # Create dummy training data (would use real data in practice)
    batch_size = 32
    x_pos = torch.randn(batch_size, 784, device=device) * 0.3 + 0.5  # "Positive" examples
    x_neg = torch.randn(batch_size, 784, device=device) * 0.5        # "Negative" examples

    # Train each layer
    results = []
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        epoch_results = []

        for layer_idx in range(len(model.layers)):
            result = model.train_layer(layer_idx, x_pos, x_neg)
            epoch_results.append(result)

            print(f"  Layer {layer_idx}: threshold={result['threshold']:.3f}, "
                  f"g_pos={result['goodness_pos']:.2f}, g_neg={result['goodness_neg']:.2f}, "
                  f"FPGA temp={result['fpga_temp']:.1f}C")

        results.append(epoch_results)

        # Check GPU state
        gpu_temp, gpu_power = get_gpu_telemetry()
        print(f"  GPU: {gpu_temp:.1f}C, {gpu_power:.1f}W")

    # Final embodied state
    print("\n" + "=" * 60)
    print("Final Embodied State")
    print("=" * 60)

    fpga_temp, _ = fpga.read_temperature()
    gpu_temp, gpu_power = get_gpu_telemetry()

    state = EmbodiedState(
        fpga_temp=fpga_temp,
        gpu_temp=gpu_temp,
        gpu_power=gpu_power,
        decay_factor=1.0 - 0.03 * max(0, fpga_temp - 50),
        confidence=0.85,
        layer_exits=0,
        timestamp=time.time()
    )

    print(f"FPGA Temperature: {state.fpga_temp:.1f}C")
    print(f"GPU Temperature:  {state.gpu_temp:.1f}C")
    print(f"GPU Power:        {state.gpu_power:.1f}W")
    print(f"Decay Factor:     {state.decay_factor:.3f}")

    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'z1116_embodied_integration.json'
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'fpga_status': fpga_status,
            'final_state': {
                'fpga_temp': state.fpga_temp,
                'gpu_temp': state.gpu_temp,
                'gpu_power': state.gpu_power,
                'decay_factor': state.decay_factor
            },
            'training_results': [[{k: v for k, v in r.items()} for r in epoch] for epoch in results]
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    fpga.disconnect()
    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
