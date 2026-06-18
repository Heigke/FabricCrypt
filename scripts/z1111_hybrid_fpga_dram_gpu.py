#!/usr/bin/env python3
"""
z1111: Deep Hybrid FPGA-DRAM-GPU Embodied Computing

True embodied AI where three physical systems intertwine:

1. FPGA (Arty A7-100T):
   - XADC reads die temperature directly into Verilog
   - Temperature modulates neuron thresholds: threshold = base + (temp - setpoint)/4
   - 0.91 correlation proven in z1110

2. DRAM (System Memory):
   - Partial timing writes create analog charge levels
   - Charge decays according to Arrhenius: τ ∝ exp(Ea/kT)
   - Decay IS the computation - temperature changes the math

3. GPU (AMD Radeon 8060S):
   - Main neural network forward/backward passes
   - Telemetry feeds FPGA and DRAM parameters
   - Gradient updates incorporate embodied signals

Data Flow:
    Input → DRAM decay → FPGA thermal neurons → GPU backprop
                ↑                    ↑               ↓
           GPU temp ────────────────────────── gradients

The key insight: computation is no longer separable from hardware state.
The same algorithm produces different results on hot vs cold silicon.
"""

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys
import json
import time
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Local modules
from src.embodied.dram_analog import DRAMAnalogCompute, DRAMNeuronLayer
from src.fpga.hybrid_embodiment import ArtyFPGA, GPUTelemetry

# Telemetry
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
    _TELEMETRY_AVAILABLE = True
except:
    _TELEMETRY_AVAILABLE = False


@dataclass
class EmbodiedState:
    """Full state of the hybrid embodied system."""
    timestamp: float
    gpu_temp_c: float
    gpu_power_w: float
    fpga_temp_c: float
    fpga_threshold: int
    fpga_fired_mask: int
    dram_decay_rate: float
    dram_avg_decay: float


class HybridEmbodiedNetwork(nn.Module):
    """
    Neural network with FPGA and DRAM embodiment.

    Architecture:
        Input (784) → DRAM Decay Layer → FPGA Thermal Layer → GPU Layers → Output (10)

    The first two layers use physical hardware state:
    - DRAM layer: charge decay based on temperature and time
    - FPGA layer: threshold modulation based on die temperature
    """

    def __init__(self, fpga: Optional[ArtyFPGA], dram: DRAMAnalogCompute,
                 hidden_dim: int = 256, use_embodiment: bool = True):
        super().__init__()

        self.fpga = fpga
        self.dram = dram
        self.use_embodiment = use_embodiment

        # DRAM decay layer (CPU/numpy based)
        self.dram_layer = DRAMNeuronLayer(784, 64, dram, base_delay_ms=0.5)

        # GPU layers
        self.gpu_fc1 = nn.Linear(64, hidden_dim)
        self.gpu_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gpu_fc3 = nn.Linear(hidden_dim, 10)

        # Learnable thermal sensitivity
        self.thermal_alpha = nn.Parameter(torch.tensor(0.01))

        # State tracking
        self.last_state: Optional[EmbodiedState] = None

    def forward(self, x: torch.Tensor, gpu_temp: float = 50.0,
                fpga_temp: float = 35.0) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through hybrid embodied network.

        Args:
            x: Input tensor [batch, 784]
            gpu_temp: Current GPU temperature
            fpga_temp: Current FPGA temperature (if available)

        Returns:
            output: Logits [batch, 10]
            telemetry: Embodiment state info
        """
        batch_size = x.shape[0]
        device = x.device

        telemetry = {
            'gpu_temp': gpu_temp,
            'fpga_temp': fpga_temp,
            'embodiment_active': self.use_embodiment
        }

        if self.use_embodiment:
            # === DRAM DECAY LAYER ===
            # Convert to numpy for DRAM operations
            x_np = x.detach().cpu().numpy()

            # Apply DRAM decay (temperature affects decay rate)
            dram_out, dram_telem = self.dram_layer.forward(x_np, gpu_temp)
            telemetry['dram_decay_ratio'] = dram_telem['avg_decay_ratio']
            telemetry['dram_delay_ms'] = dram_telem['avg_delay_ms']

            # Back to torch
            h = torch.from_numpy(dram_out).float().to(device)

            # === FPGA THERMAL LAYER ===
            if self.fpga is not None and self.fpga.ser is not None:
                # Send activations through FPGA for thermal threshold modulation
                fpga_outputs = []
                fpga_thresholds = []

                for i in range(batch_size):
                    # Quantize for FPGA (8 neurons, 8-bit inputs)
                    h_sample = h[i, :8].detach().cpu().numpy()
                    h_quantized = np.clip(h_sample * 255, 0, 255).astype(np.uint8)

                    result = self.fpga.compute(list(h_quantized))
                    fpga_out = np.array(result['outputs'], dtype=np.float32) / 255.0
                    fpga_outputs.append(fpga_out)
                    fpga_thresholds.append(result['threshold'])

                # Average FPGA outputs back into hidden state
                fpga_out_tensor = torch.from_numpy(np.array(fpga_outputs)).float().to(device)

                # Blend FPGA output with remaining hidden dims
                # h[:, :8] is modulated by FPGA, rest passes through
                h_modulated = h.clone()
                h_modulated[:, :8] = h[:, :8] * (1 + self.thermal_alpha * (fpga_out_tensor - 0.5))

                telemetry['fpga_avg_threshold'] = float(np.mean(fpga_thresholds))
                telemetry['fpga_fired_count'] = sum(1 for r in fpga_outputs if np.sum(r) > 0)

                h = h_modulated
            else:
                # Simulate FPGA thermal modulation
                thermal_factor = 1.0 + self.thermal_alpha * (fpga_temp - 35.0) / 10.0
                h = h * thermal_factor
                telemetry['fpga_avg_threshold'] = 100 + int((fpga_temp - 33) * 4)
                telemetry['fpga_simulated'] = True

        else:
            # Non-embodied baseline: just linear transform
            h = torch.relu(nn.functional.linear(x,
                torch.from_numpy(self.dram_layer.weights.T).float().to(device),
                torch.from_numpy(self.dram_layer.bias).float().to(device)))

        # === GPU LAYERS ===
        h = torch.relu(self.gpu_fc1(h))
        h = torch.relu(self.gpu_fc2(h))
        out = self.gpu_fc3(h)

        return out, telemetry


def train_epoch(model: HybridEmbodiedNetwork,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                gpu_telemetry: GPUTelemetry,
                epoch: int) -> Dict:
    """Train one epoch with embodiment tracking."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    all_telemetry = []
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        target = target.to(device)

        # Get current temperatures
        gpu_temp, gpu_power = gpu_telemetry.read()
        fpga_temp = 35.0  # Will be updated if FPGA connected

        if model.fpga and model.fpga.ser:
            fpga_temp_reading, _ = model.fpga.read_temperature()
            if fpga_temp_reading > 0:
                fpga_temp = fpga_temp_reading

        # Forward pass with embodiment
        optimizer.zero_grad()
        output, telem = model(data, gpu_temp, fpga_temp)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

        telem['loss'] = loss.item()
        telem['gpu_power'] = gpu_power
        all_telemetry.append(telem)

        if batch_idx % 100 == 0:
            dram_decay = telem.get('dram_decay_ratio', 1.0)
            fpga_thresh = telem.get('fpga_avg_threshold', 100)
            print(f"  [{batch_idx:4d}] loss={loss.item():.4f} "
                  f"GPU={gpu_temp:.1f}°C FPGA={fpga_temp:.1f}°C "
                  f"decay={dram_decay:.3f} thresh={fpga_thresh}")

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100.0 * correct / total,
        'telemetry': all_telemetry
    }


def evaluate(model: HybridEmbodiedNetwork,
             test_loader: DataLoader,
             device: torch.device,
             gpu_telemetry: GPUTelemetry) -> Dict:
    """Evaluate model with embodiment tracking."""

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    all_telemetry = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)

            gpu_temp, gpu_power = gpu_telemetry.read()
            fpga_temp = 35.0
            if model.fpga and model.fpga.ser:
                fpga_temp, _ = model.fpga.read_temperature()

            output, telem = model(data, gpu_temp, fpga_temp)

            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            all_telemetry.append(telem)

    return {
        'loss': total_loss / len(test_loader),
        'accuracy': 100.0 * correct / total,
        'telemetry': all_telemetry
    }


def run_condition(condition: str,
                  train_loader: DataLoader,
                  test_loader: DataLoader,
                  device: torch.device,
                  epochs: int,
                  fpga_port: Optional[str]) -> Dict:
    """
    Run one experimental condition.

    Conditions:
    - A: No embodiment (baseline)
    - B: Full embodiment (FPGA + DRAM + GPU)
    - C: DRAM only (no FPGA)
    - D: FPGA only (no DRAM decay)
    """

    print(f"\n{'='*70}")
    print(f"  CONDITION {condition}")
    print(f"{'='*70}")

    # Initialize DRAM
    dram = DRAMAnalogCompute(size_bytes=4*1024*1024)
    dram.initialize()

    # Initialize FPGA
    fpga = None
    if condition in ['B', 'D'] and fpga_port:
        fpga = ArtyFPGA(fpga_port)
        if fpga.connect():
            print(f"  FPGA connected on {fpga_port}")
            temp, _ = fpga.read_temperature()
            print(f"  FPGA temperature: {temp:.1f}°C")
        else:
            print(f"  FPGA connection failed, using simulation")
            fpga = None

    # Initialize GPU telemetry
    gpu_telemetry = GPUTelemetry()
    gpu_temp, gpu_power = gpu_telemetry.read()
    print(f"  GPU: {gpu_temp:.1f}°C, {gpu_power:.1f}W")

    # Create model
    use_embodiment = condition != 'A'
    model = HybridEmbodiedNetwork(
        fpga=fpga if condition in ['B', 'D'] else None,
        dram=dram,
        hidden_dim=256,
        use_embodiment=use_embodiment
    ).to(device)

    # For DRAM-only, disable FPGA modulation
    if condition == 'C':
        model.fpga = None

    # For FPGA-only, bypass DRAM decay
    if condition == 'D':
        model.dram_layer.base_delay_ms = 0.0  # No decay delay

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Embodiment: {use_embodiment}")

    # Training
    epoch_results = []
    for epoch in range(epochs):
        print(f"\n  Epoch {epoch+1}/{epochs}")

        train_result = train_epoch(model, train_loader, optimizer, device,
                                   gpu_telemetry, epoch)
        test_result = evaluate(model, test_loader, device, gpu_telemetry)

        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_result['loss'],
            'train_acc': train_result['accuracy'],
            'test_loss': test_result['loss'],
            'test_acc': test_result['accuracy']
        })

        print(f"  Train: loss={train_result['loss']:.4f} acc={train_result['accuracy']:.2f}%")
        print(f"  Test:  loss={test_result['loss']:.4f} acc={test_result['accuracy']:.2f}%")

    # Final evaluation with detailed telemetry
    final_eval = evaluate(model, test_loader, device, gpu_telemetry)

    # Compute embodiment statistics
    if use_embodiment and final_eval['telemetry']:
        telem_list = final_eval['telemetry']
        dram_decays = [t.get('dram_decay_ratio', 1.0) for t in telem_list]
        fpga_thresholds = [t.get('fpga_avg_threshold', 100) for t in telem_list]
        gpu_temps = [t.get('gpu_temp', 50) for t in telem_list]

        # Correlation between temperature and computation
        if len(gpu_temps) > 10:
            temp_decay_corr = np.corrcoef(gpu_temps, dram_decays)[0, 1]
            temp_thresh_corr = np.corrcoef(gpu_temps[:len(fpga_thresholds)],
                                           fpga_thresholds)[0, 1]
        else:
            temp_decay_corr = 0.0
            temp_thresh_corr = 0.0

        embodiment_stats = {
            'dram_decay_mean': float(np.mean(dram_decays)),
            'dram_decay_std': float(np.std(dram_decays)),
            'fpga_threshold_mean': float(np.mean(fpga_thresholds)),
            'fpga_threshold_std': float(np.std(fpga_thresholds)),
            'temp_decay_correlation': float(temp_decay_corr) if not np.isnan(temp_decay_corr) else 0.0,
            'temp_threshold_correlation': float(temp_thresh_corr) if not np.isnan(temp_thresh_corr) else 0.0
        }
    else:
        embodiment_stats = {}

    # Cleanup
    dram.cleanup()
    if fpga:
        fpga.disconnect()

    return {
        'condition': condition,
        'use_embodiment': use_embodiment,
        'param_count': param_count,
        'epochs': epochs,
        'final_test_accuracy': final_eval['accuracy'],
        'final_test_loss': final_eval['loss'],
        'epoch_results': epoch_results,
        'embodiment_stats': embodiment_stats
    }


def main():
    parser = argparse.ArgumentParser(description="z1111 Hybrid FPGA-DRAM-GPU Embodiment")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per condition")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--fpga-port", type=str, default="/dev/ttyUSB1", help="FPGA serial port")
    parser.add_argument("--conditions", type=str, default="ABCD", help="Which conditions to run")
    args = parser.parse_args()

    print("=" * 70)
    print("  z1111: HYBRID FPGA-DRAM-GPU EMBODIED COMPUTING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load MNIST
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)

    # Use subset for faster iteration
    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    test_subset = torch.utils.data.Subset(test_dataset, range(2000))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_subset)} samples")
    print(f"Test: {len(test_subset)} samples")

    # Run conditions
    all_results = {}
    conditions_to_run = list(args.conditions.upper())

    for cond in conditions_to_run:
        result = run_condition(
            condition=cond,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            fpga_port=args.fpga_port if cond in ['B', 'D'] else None
        )
        all_results[cond] = result

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\n{'Condition':<12} {'Embodiment':<15} {'Test Acc':<10} {'Test Loss':<10}")
    print("-" * 50)

    for cond in conditions_to_run:
        r = all_results[cond]
        emb = "Full" if cond == 'B' else ("DRAM only" if cond == 'C' else
               ("FPGA only" if cond == 'D' else "None"))
        print(f"{cond:<12} {emb:<15} {r['final_test_accuracy']:.2f}%     {r['final_test_loss']:.4f}")

    # Embodiment analysis
    if 'B' in all_results and all_results['B'].get('embodiment_stats'):
        stats = all_results['B']['embodiment_stats']
        print(f"\nEmbodiment Statistics (Condition B):")
        print(f"  DRAM decay mean: {stats.get('dram_decay_mean', 0):.4f}")
        print(f"  FPGA threshold mean: {stats.get('fpga_threshold_mean', 0):.1f}")
        print(f"  Temp-Decay correlation: {stats.get('temp_decay_correlation', 0):.4f}")
        print(f"  Temp-Threshold correlation: {stats.get('temp_threshold_correlation', 0):.4f}")

    # Key comparison
    if 'A' in all_results and 'B' in all_results:
        baseline_acc = all_results['A']['final_test_accuracy']
        embodied_acc = all_results['B']['final_test_accuracy']
        diff = embodied_acc - baseline_acc

        print(f"\n  KEY COMPARISON:")
        print(f"  Baseline (A):  {baseline_acc:.2f}%")
        print(f"  Embodied (B):  {embodied_acc:.2f}%")
        print(f"  Difference:    {diff:+.2f}%")

        if diff > 0:
            print(f"\n  *** EMBODIMENT IMPROVES ACCURACY ***")
        else:
            print(f"\n  Embodiment changes computation but accuracy differs by {diff:.2f}%")

    # Save results
    results_path = project_root / "results" / "z1111_hybrid_fpga_dram_gpu.json"

    output = {
        'experiment': 'z1111_hybrid_fpga_dram_gpu',
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'fpga_port': args.fpga_port,
            'device': str(device)
        },
        'conditions': {k: {kk: vv for kk, vv in v.items() if kk != 'telemetry'}
                       for k, v in all_results.items()},
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved: {results_path}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
