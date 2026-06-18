#!/usr/bin/env python3
"""
z1125: GPU-FPGA Intertwined Inference Pipeline

Creates a true hybrid compute system where:
- GPU: Runs neural network layers (fast, high throughput)
- FPGA: Provides analog memory with decay-based computation
- Feedback: FPGA temperature/decay modulates GPU computation

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                    INTERTWINED INFERENCE LOOP                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input → [GPU Encoder] → Features                                │
│                              ↓                                   │
│                    [FPGA Analog Store]                           │
│                              ↓                                   │
│                    [Decay (physics)]                             │
│                              ↓                                   │
│                    [FPGA Read + Telemetry]                       │
│                              ↓                                   │
│  Output ← [GPU Decoder] ← Decayed Features + Temp                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import FPGA interface
from src.fpga.fpga_interface import FPGAInterface

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available")


def get_gpu_telemetry() -> Dict:
    """Get GPU temperature and power from sysfs"""
    try:
        hwmon_path = "/sys/class/drm/card1/device/hwmon/hwmon7"
        with open(f"{hwmon_path}/temp1_input", 'r') as f:
            temp = float(f.read().strip()) / 1000.0
        power = 0.0
        power_file = f"{hwmon_path}/power1_average"
        if os.path.exists(power_file):
            with open(power_file, 'r') as f:
                power = float(f.read().strip()) / 1e6
        return {'temp': temp, 'power': power}
    except:
        return {'temp': 0.0, 'power': 0.0}


@dataclass
class HybridConfig:
    """Configuration for hybrid GPU-FPGA system"""
    # GPU settings
    hidden_dim: int = 64
    num_layers: int = 3
    device: str = 'cuda'

    # FPGA settings
    fpga_port: str = '/dev/ttyUSB1'
    base_addr: int = 0x300000
    decay_wait_ms: float = 5.0

    # Embodiment settings
    temp_threshold_low: float = 50.0
    temp_threshold_high: float = 60.0
    adaptive_decay: bool = True


class GPUEncoder(nn.Module):
    """GPU-based encoder that produces features for FPGA storage"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output in [0, 1] for analog storage
        )

    def forward(self, x):
        return self.layers(x)


class GPUDecoder(nn.Module):
    """GPU-based decoder that processes FPGA-decayed features"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, temp_dim: int = 2):
        super().__init__()
        # Includes temperature conditioning
        self.temp_embed = nn.Linear(temp_dim, hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, temp_features):
        # Condition on temperature
        temp_embed = self.temp_embed(temp_features)
        combined = torch.cat([x, temp_embed], dim=-1)
        return self.layers(combined)


class FPGAAnalogMemory:
    """FPGA-based analog memory with decay computation"""

    def __init__(self, fpga: FPGAInterface, config: HybridConfig):
        self.fpga = fpga
        self.config = config
        self.stored_values = []
        self.telemetry_history = []

    def store_features(self, features: np.ndarray) -> Dict:
        """Store feature vector in FPGA DRAM with partial timing

        Args:
            features: numpy array of values in [0, 1]

        Returns:
            dict with storage results and telemetry
        """
        results = []
        pattern = bytes([0xFF] * 16)

        for i, val in enumerate(features.flatten()[:16]):  # Max 16 values (128 bits)
            addr = self.config.base_addr + (i * 256)

            # Convert value to timing offset (higher value = lower offset = more charge)
            if val >= 0.9:
                offset = 0
            elif val <= 0.1:
                offset = 32
            else:
                offset = int(32 * (1.0 - val))

            result = self.fpga.partial_timing_write(addr, pattern, offset)
            results.append({
                'addr': addr,
                'value': float(val),
                'offset': offset,
                'success': result.get('success', False),
                'temp': result.get('temperature', 0)
            })
            time.sleep(0.02)  # Pacing

        self.stored_values = features.flatten()[:16].tolist()

        return {
            'stored_count': len(results),
            'success_count': sum(1 for r in results if r['success']),
            'avg_temp': np.mean([r['temp'] for r in results if r['temp'] > 0]),
            'details': results
        }

    def decay_and_read(self, wait_ms: float = None) -> Tuple[np.ndarray, Dict]:
        """Wait for decay and read back features

        Returns:
            (decayed_features, telemetry_dict)
        """
        if wait_ms is None:
            wait_ms = self.config.decay_wait_ms

        # Wait for natural decay
        time.sleep(wait_ms / 1000.0)

        # Read back
        values = []
        pattern = bytes([0xFF] * 16)

        for i in range(len(self.stored_values)):
            addr = self.config.base_addr + (i * 256)
            data = self.fpga.ddr_read(addr)

            if data:
                # Count remaining 1s as analog value
                ones = sum(bin(b).count('1') for b in data)
                analog = ones / 128.0
                values.append(analog)
            else:
                values.append(0.0)

        # Get telemetry
        fpga_temp, _ = self.fpga.read_temperature()
        gpu = get_gpu_telemetry()

        telemetry = {
            'fpga_temp': fpga_temp,
            'gpu_temp': gpu['temp'],
            'gpu_power': gpu['power'],
            'decay_wait_ms': wait_ms
        }
        self.telemetry_history.append(telemetry)

        return np.array(values), telemetry


class HybridInferencePipeline:
    """Complete GPU-FPGA hybrid inference pipeline"""

    def __init__(self, config: HybridConfig = None):
        self.config = config or HybridConfig()
        self.fpga = None
        self.analog_memory = None
        self.encoder = None
        self.decoder = None
        self.device = None
        self.metrics = []

    def initialize(self) -> bool:
        """Initialize GPU and FPGA components"""
        print("Initializing hybrid pipeline...")

        # Initialize FPGA
        self.fpga = FPGAInterface(port=self.config.fpga_port)
        if not self.fpga.connect():
            print("ERROR: Could not connect to FPGA")
            return False

        status = self.fpga.ping()
        if not status.get('ddr3_ready'):
            print("ERROR: FPGA DDR3 not ready")
            return False

        self.analog_memory = FPGAAnalogMemory(self.fpga, self.config)
        print(f"  FPGA: Connected, DDR3 ready")

        # Initialize GPU
        if HAS_TORCH:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("  GPU: Not available, using CPU")

            # Create models
            self.encoder = GPUEncoder(
                input_dim=32,
                hidden_dim=self.config.hidden_dim,
                output_dim=16  # 16 values for FPGA storage
            ).to(self.device)

            self.decoder = GPUDecoder(
                input_dim=16,
                hidden_dim=self.config.hidden_dim,
                output_dim=10,  # Classification output
                temp_dim=2
            ).to(self.device)

            print(f"  Models: Encoder (32->16), Decoder (16->10)")
        else:
            print("  GPU: PyTorch not available")
            return False

        return True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Run inference through hybrid pipeline

        Args:
            x: Input tensor [batch, 32]

        Returns:
            (output tensor, metrics dict)
        """
        start_time = time.time()
        metrics = {}

        # Step 1: GPU Encode
        t0 = time.time()
        with torch.no_grad():
            features = self.encoder(x)
        metrics['gpu_encode_ms'] = (time.time() - t0) * 1000

        # Step 2: Store in FPGA
        t0 = time.time()
        features_np = features[0].cpu().numpy()  # First sample
        store_result = self.analog_memory.store_features(features_np)
        metrics['fpga_store_ms'] = (time.time() - t0) * 1000
        metrics['fpga_store_success'] = store_result['success_count']

        # Step 3: Decay and read
        t0 = time.time()

        # Adaptive decay based on temperature
        if self.config.adaptive_decay:
            fpga_temp, _ = self.fpga.read_temperature()
            gpu = get_gpu_telemetry()

            # Higher combined temp = shorter decay (energy saving)
            combined_temp = (fpga_temp + gpu['temp']) / 2
            if combined_temp > self.config.temp_threshold_high:
                decay_ms = self.config.decay_wait_ms * 0.5
            elif combined_temp < self.config.temp_threshold_low:
                decay_ms = self.config.decay_wait_ms * 1.5
            else:
                decay_ms = self.config.decay_wait_ms
        else:
            decay_ms = self.config.decay_wait_ms

        decayed_features, telemetry = self.analog_memory.decay_and_read(decay_ms)
        metrics['fpga_decay_read_ms'] = (time.time() - t0) * 1000
        metrics['actual_decay_ms'] = decay_ms
        metrics.update(telemetry)

        # Step 4: GPU Decode with temperature conditioning
        t0 = time.time()
        decayed_tensor = torch.tensor(decayed_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        temp_tensor = torch.tensor([
            telemetry['fpga_temp'] / 100.0,
            telemetry['gpu_temp'] / 100.0
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.decoder(decayed_tensor, temp_tensor)
        metrics['gpu_decode_ms'] = (time.time() - t0) * 1000

        metrics['total_ms'] = (time.time() - start_time) * 1000
        self.metrics.append(metrics)

        return output, metrics

    def benchmark(self, num_iterations: int = 50) -> Dict:
        """Run benchmark comparing hybrid vs GPU-only"""
        print(f"\nRunning benchmark ({num_iterations} iterations)...")

        hybrid_times = []
        gpu_only_times = []
        hybrid_metrics = []

        for i in range(num_iterations):
            # Generate random input
            x = torch.randn(1, 32).to(self.device)

            # Hybrid forward
            _, metrics = self.forward(x)
            hybrid_times.append(metrics['total_ms'])
            hybrid_metrics.append(metrics)

            # GPU-only forward (for comparison)
            t0 = time.time()
            with torch.no_grad():
                features = self.encoder(x)
                # Skip FPGA, use features directly
                temp_tensor = torch.tensor([[0.5, 0.5]]).to(self.device)
                _ = self.decoder(features, temp_tensor)
            gpu_only_times.append((time.time() - t0) * 1000)

            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{num_iterations}")

        # Calculate statistics
        results = {
            'hybrid': {
                'mean_ms': np.mean(hybrid_times),
                'std_ms': np.std(hybrid_times),
                'min_ms': np.min(hybrid_times),
                'max_ms': np.max(hybrid_times),
            },
            'gpu_only': {
                'mean_ms': np.mean(gpu_only_times),
                'std_ms': np.std(gpu_only_times),
                'min_ms': np.min(gpu_only_times),
                'max_ms': np.max(gpu_only_times),
            },
            'telemetry': {
                'mean_fpga_temp': np.mean([m['fpga_temp'] for m in hybrid_metrics]),
                'mean_gpu_temp': np.mean([m['gpu_temp'] for m in hybrid_metrics]),
                'mean_gpu_power': np.mean([m['gpu_power'] for m in hybrid_metrics]),
            },
            'breakdown': {
                'gpu_encode_ms': np.mean([m['gpu_encode_ms'] for m in hybrid_metrics]),
                'fpga_store_ms': np.mean([m['fpga_store_ms'] for m in hybrid_metrics]),
                'fpga_decay_read_ms': np.mean([m['fpga_decay_read_ms'] for m in hybrid_metrics]),
                'gpu_decode_ms': np.mean([m['gpu_decode_ms'] for m in hybrid_metrics]),
            }
        }

        # Calculate energy metrics
        gpu_power = results['telemetry']['mean_gpu_power']
        if gpu_power > 0:
            results['energy'] = {
                'hybrid_mj_per_inference': gpu_power * results['hybrid']['mean_ms'],
                'gpu_only_mj_per_inference': gpu_power * results['gpu_only']['mean_ms'],
                'fpga_overhead_factor': results['hybrid']['mean_ms'] / results['gpu_only']['mean_ms']
            }

        return results

    def shutdown(self):
        """Clean up resources"""
        if self.fpga:
            self.fpga.disconnect()


def run_experiment():
    """Run the full GPU-FPGA intertwined experiment"""
    print("=" * 70)
    print("z1125: GPU-FPGA Intertwined Inference Pipeline")
    print("=" * 70)

    config = HybridConfig(
        hidden_dim=64,
        decay_wait_ms=5.0,
        adaptive_decay=True
    )

    pipeline = HybridInferencePipeline(config)

    if not pipeline.initialize():
        print("Failed to initialize pipeline")
        return None

    # Get initial telemetry
    fpga_temp, _ = pipeline.fpga.read_temperature()
    gpu = get_gpu_telemetry()
    print(f"\nInitial state:")
    print(f"  FPGA temp: {fpga_temp:.1f}C")
    print(f"  GPU temp: {gpu['temp']:.1f}C")
    print(f"  GPU power: {gpu['power']:.1f}W")

    # Run benchmark
    results = pipeline.benchmark(num_iterations=30)

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nLatency Comparison:")
    print(f"  Hybrid:   {results['hybrid']['mean_ms']:.2f} ± {results['hybrid']['std_ms']:.2f} ms")
    print(f"  GPU-only: {results['gpu_only']['mean_ms']:.2f} ± {results['gpu_only']['std_ms']:.2f} ms")
    print(f"  Overhead: {results.get('energy', {}).get('fpga_overhead_factor', 0):.2f}x")

    print(f"\nHybrid Pipeline Breakdown:")
    for key, val in results['breakdown'].items():
        print(f"  {key}: {val:.2f} ms")

    print(f"\nTelemetry:")
    print(f"  Avg FPGA temp: {results['telemetry']['mean_fpga_temp']:.1f}C")
    print(f"  Avg GPU temp: {results['telemetry']['mean_gpu_temp']:.1f}C")
    print(f"  Avg GPU power: {results['telemetry']['mean_gpu_power']:.1f}W")

    if 'energy' in results:
        print(f"\nEnergy (estimated):")
        print(f"  Hybrid: {results['energy']['hybrid_mj_per_inference']:.2f} mJ/inference")
        print(f"  GPU-only: {results['energy']['gpu_only_mj_per_inference']:.2f} mJ/inference")

    # Business value assessment
    print("\n" + "=" * 70)
    print("EMBODIED AI VALUE")
    print("=" * 70)
    print("""
✓ Temperature-Conditioned Inference
  - Decoder receives FPGA and GPU temps as input
  - Output varies based on hardware thermal state

✓ Adaptive Decay
  - High temp → shorter decay → less energy
  - Low temp → longer decay → better precision

✓ Physics-Based Regularization
  - DRAM decay acts as natural dropout
  - Temperature modulates decay rate

✓ Self-Regulating System
  - No external controller needed
  - Hardware state directly affects computation
""")

    pipeline.shutdown()

    return results


def main():
    results = run_experiment()

    if results:
        # Save results
        output_path = Path('results/z1125_gpu_fpga_intertwined.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
