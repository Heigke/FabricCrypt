#!/usr/bin/env python3
"""
z1117: Robust FPGA-GPU Embodied AI Integration

Key insight from z1114-z1116: FPGA needs delays between operations (CDC timing).
This script adds proper pacing to avoid state corruption.

Demonstrates:
1. FPGA analog memory (partial timing writes for strength)
2. Physical decay (temperature-dependent forgetting)
3. Forward-Forward learning with temperature modulation
4. GPU energy tracking
"""

import sys
sys.path.insert(0, 'src/fpga')
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import numpy as np
import time
import json
from datetime import datetime
from fpga_interface import FPGAInterface


class RobustFPGAMemory:
    """FPGA analog memory with proper operation pacing"""

    def __init__(self, fpga: FPGAInterface, slot_size: int = 64, op_delay: float = 0.05):
        self.fpga = fpga
        self.slot_size = slot_size
        self.op_delay = op_delay  # Delay between FPGA operations (CDC timing)
        self.base_addr = 0x400000

    def _pace(self):
        """Add delay between operations to avoid CDC issues"""
        time.sleep(self.op_delay)

    def write_analog(self, slot: int, values: np.ndarray, strength: float = 1.0) -> dict:
        """Write values with analog strength (0.0-1.0)

        strength=1.0: full charge (offset=0)
        strength=0.5: partial charge (offset=32)
        strength=0.0: minimal charge (offset=63)
        """
        addr = self.base_addr + slot * 16

        # Quantize to bytes
        values_clipped = np.clip(values, 0, 1)
        data = (values_clipped * 255).astype(np.uint8)

        # Pad or truncate to 16 bytes
        if len(data) < 16:
            data = np.pad(data, (0, 16 - len(data)))
        data = bytes(data[:16])

        # Calculate timing offset (0=full, 63=minimal)
        timing_offset = int((1.0 - strength) * 63)

        self._pace()

        if timing_offset == 0:
            # Full strength: use normal write
            success = self.fpga.ddr_write(addr, data)
            return {'success': success, 'method': 'normal', 'strength': strength}
        else:
            # Partial strength: use partial timing
            result = self.fpga.partial_timing_write(addr, data, timing_offset)
            return {
                'success': result.get('success', False),
                'method': 'partial_timing',
                'timing_offset': timing_offset,
                'strength': strength,
                'temp': result.get('temperature', 0)
            }

    def read(self, slot: int) -> np.ndarray:
        """Read values from slot"""
        addr = self.base_addr + slot * 16
        self._pace()
        data = self.fpga.ddr_read(addr)
        if data:
            return np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0
        return np.zeros(16, dtype=np.float32)

    def decay_test(self, slot: int, wait_ms: int = 100) -> dict:
        """Test decay at a slot"""
        addr = self.base_addr + slot * 16
        wait_cycles = int(wait_ms * 83000)  # 83MHz ui_clk
        self._pace()
        return self.fpga.decay_test(addr, bytes([0xFF]*16), wait_cycles)


class EmbodiedFFLayer(nn.Module):
    """Forward-Forward layer with temperature modulation"""

    def __init__(self, in_features: int, out_features: int, base_threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.base_threshold = base_threshold
        self.current_temp = 50.0  # Default GPU temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm on input
        x = x / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        h = self.linear(x)
        return torch.relu(h)

    def get_threshold(self) -> float:
        """Temperature-modulated threshold"""
        # Higher temp -> lower threshold -> easier activation
        temp_factor = 1.0 - 0.02 * max(0, self.current_temp - 50)
        return self.base_threshold * max(0.5, temp_factor)

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Squared sum of activations (Forward-Forward metric)"""
        return (h ** 2).sum(dim=1)


class EmbodiedFFNetwork(nn.Module):
    """Forward-Forward network with embodiment signals"""

    def __init__(self, layer_sizes: list, base_threshold: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(EmbodiedFFLayer(
                layer_sizes[i], layer_sizes[i+1], base_threshold
            ))
        self.base_threshold = base_threshold

    def forward(self, x: torch.Tensor) -> list:
        """Forward pass returning activations at each layer"""
        activations = []
        h = x
        for layer in self.layers:
            h = layer(h)
            activations.append(h)
        return activations

    def set_temperature(self, temp: float):
        """Set temperature for all layers"""
        for layer in self.layers:
            layer.current_temp = temp


def get_gpu_stats():
    """Get GPU temperature and power via sysfs"""
    try:
        with open('/sys/class/hwmon/hwmon2/temp1_input', 'r') as f:
            temp = int(f.read().strip()) / 1000.0
        with open('/sys/class/hwmon/hwmon2/power1_average', 'r') as f:
            power = int(f.read().strip()) / 1e6
        return temp, power
    except:
        return 50.0, 50.0  # Defaults


def train_ff_step(model, x_pos, x_neg, optimizer, fpga_temp):
    """Single Forward-Forward training step"""
    model.set_temperature(fpga_temp)

    # Positive pass
    acts_pos = model(x_pos)

    # Negative pass
    acts_neg = model(x_neg)

    # FF loss: positive goodness > threshold, negative < threshold
    loss = 0.0
    layer_stats = []

    for i, (layer, h_pos, h_neg) in enumerate(zip(model.layers, acts_pos, acts_neg)):
        g_pos = layer.goodness(h_pos)
        g_neg = layer.goodness(h_neg)
        threshold = layer.get_threshold()

        # Loss pushes g_pos above threshold, g_neg below
        l_pos = torch.log(1 + torch.exp(threshold - g_pos)).mean()
        l_neg = torch.log(1 + torch.exp(g_neg - threshold)).mean()
        loss += l_pos + l_neg

        layer_stats.append({
            'g_pos': g_pos.mean().item(),
            'g_neg': g_neg.mean().item(),
            'threshold': threshold
        })

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), layer_stats


def main():
    print("=" * 60)
    print("z1117: Robust FPGA-GPU Embodied AI Integration")
    print("=" * 60)

    # Connect to FPGA
    print("\nConnecting to FPGA...")
    fpga = FPGAInterface()
    if not fpga.connect():
        print("FPGA connection failed!")
        return

    status = fpga.get_status()
    print(f"FPGA Status: {status}")

    fpga_temp, _ = fpga.read_temperature()
    gpu_temp, gpu_power = get_gpu_stats()
    print(f"FPGA Temp: {fpga_temp:.1f}C, GPU Temp: {gpu_temp:.1f}C")

    # Initialize systems
    memory = RobustFPGAMemory(fpga, op_delay=0.05)  # 50ms between ops

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model: 784 -> 256 -> 128 -> 64
    model = EmbodiedFFNetwork([784, 256, 128, 64], base_threshold=2.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    results = {
        'timestamp': datetime.now().isoformat(),
        'fpga_status': status,
        'tests': {}
    }

    # === Test 1: FPGA Analog Memory ===
    print("\n" + "=" * 60)
    print("Test 1: FPGA Analog Memory (Partial Timing)")
    print("=" * 60)

    analog_results = []
    test_values = np.linspace(0, 1, 8)  # 8 values from 0 to 1

    for strength in [1.0, 0.8, 0.6, 0.4]:
        print(f"\n  Writing strength={strength:.1f}...")
        wr = memory.write_analog(slot=0, values=test_values, strength=strength)

        if wr['success']:
            readback = memory.read(slot=0)
            error = np.abs(test_values[:len(readback)] - readback[:len(test_values)]).mean()
            print(f"    Write OK, method={wr['method']}, mean_error={error:.4f}")
            analog_results.append({
                'strength': strength,
                'success': True,
                'mean_error': float(error),
                'method': wr['method']
            })
        else:
            print(f"    Write FAILED")
            analog_results.append({'strength': strength, 'success': False})

    results['tests']['analog_memory'] = analog_results

    # === Test 2: Physical Decay ===
    print("\n" + "=" * 60)
    print("Test 2: Physical Decay (Temperature-Dependent)")
    print("=" * 60)

    decay_results = []
    for wait_ms in [10, 50, 100]:
        print(f"\n  Decay test: wait={wait_ms}ms...")
        d = memory.decay_test(slot=1, wait_ms=wait_ms)

        if d.get('success'):
            errors = d.get('bit_errors', 0)
            temp = d.get('temperature', 0)
            print(f"    Bit errors: {errors}, Temp: {temp:.1f}C")
            decay_results.append({
                'wait_ms': wait_ms,
                'bit_errors': errors,
                'temperature': temp,
                'success': True
            })
        else:
            print(f"    Test failed")
            decay_results.append({'wait_ms': wait_ms, 'success': False})

    results['tests']['decay'] = decay_results

    # === Test 3: Forward-Forward Training with Embodiment ===
    print("\n" + "=" * 60)
    print("Test 3: Forward-Forward Training with Embodiment")
    print("=" * 60)

    training_results = []
    batch_size = 32

    for epoch in range(3):
        # Generate random data (MNIST-like)
        x_pos = torch.randn(batch_size, 784).to(device)
        x_neg = torch.randn(batch_size, 784).to(device) * 0.5 + 0.5  # Different distribution

        # Get current temperatures
        fpga_temp, _ = fpga.read_temperature()
        gpu_temp, gpu_power = get_gpu_stats()

        # Train step
        loss, layer_stats = train_ff_step(model, x_pos, x_neg, optimizer, gpu_temp)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Loss: {loss:.4f}")
        print(f"  GPU: {gpu_temp:.1f}C, {gpu_power:.1f}W | FPGA: {fpga_temp:.1f}C")
        for i, ls in enumerate(layer_stats):
            print(f"  Layer {i}: g_pos={ls['g_pos']:.2f}, g_neg={ls['g_neg']:.2f}, thresh={ls['threshold']:.3f}")

        training_results.append({
            'epoch': epoch + 1,
            'loss': loss,
            'gpu_temp': gpu_temp,
            'gpu_power': gpu_power,
            'fpga_temp': fpga_temp,
            'layer_stats': layer_stats
        })

        # Store activation statistics in FPGA (embodied memory)
        for i, ls in enumerate(layer_stats):
            g_ratio = ls['g_pos'] / (ls['g_neg'] + 1e-8)
            strength = min(1.0, g_ratio / 10.0)  # Stronger memories for better separation
            memory.write_analog(slot=10+i, values=np.array([g_ratio/100.0]*8), strength=strength)

    results['tests']['training'] = training_results

    # === Test 4: Embodied Loop (Full Integration) ===
    print("\n" + "=" * 60)
    print("Test 4: Embodied Loop - Temperature Modulates Computation")
    print("=" * 60)

    embodied_results = []
    x_test = torch.randn(16, 784).to(device)

    for trial in range(5):
        fpga_temp, _ = fpga.read_temperature()
        gpu_temp, gpu_power = get_gpu_stats()

        # Temperature affects threshold
        model.set_temperature(gpu_temp)

        with torch.no_grad():
            acts = model(x_test)
            total_goodness = sum(layer.goodness(h).mean().item() for layer, h in zip(model.layers, acts))

        # Write goodness to FPGA
        g_normalized = min(1.0, total_goodness / 100.0)
        memory.write_analog(slot=20+trial, values=np.array([g_normalized]*8), strength=g_normalized)

        print(f"  Trial {trial+1}: GPU={gpu_temp:.1f}C, Goodness={total_goodness:.2f}, Written strength={g_normalized:.2f}")

        embodied_results.append({
            'trial': trial + 1,
            'gpu_temp': gpu_temp,
            'fpga_temp': fpga_temp,
            'total_goodness': total_goodness,
            'write_strength': g_normalized
        })

        time.sleep(0.2)  # Let temperatures stabilize

    results['tests']['embodied_loop'] = embodied_results

    # === Final Summary ===
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)

    # Count successes
    analog_success = sum(1 for r in analog_results if r.get('success'))
    decay_success = sum(1 for r in decay_results if r.get('success'))

    print(f"Analog Memory: {analog_success}/{len(analog_results)} tests passed")
    print(f"Decay Tests:   {decay_success}/{len(decay_results)} tests passed")
    print(f"Training:      {len(training_results)} epochs completed")
    print(f"Embodied Loop: {len(embodied_results)} trials completed")

    results['summary'] = {
        'analog_success': analog_success,
        'analog_total': len(analog_results),
        'decay_success': decay_success,
        'decay_total': len(decay_results),
        'training_epochs': len(training_results),
        'embodied_trials': len(embodied_results)
    }

    # Save results
    output_path = 'results/z1117_robust_embodied.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    fpga.disconnect()
    print("\nDone!")


if __name__ == '__main__':
    main()
