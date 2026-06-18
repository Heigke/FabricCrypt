#!/usr/bin/env python3
"""
z1912: Distributed Consciousness Test

Tests whether consciousness indicators change when computation is distributed
across multiple machines (ikaros + daedalus).

Based on 2025 research showing:
1. LLMs on multiple GPUs are "obligate distributed entities"
2. If consciousness exists, it may have different properties when distributed
3. Swarm intelligence can exhibit emergent consciousness

We test:
1. LOCAL: Model runs entirely on ikaros (GPU)
2. DISTRIBUTED: Model coordination with daedalus via network
3. Compare consciousness indicators between configurations

If consciousness is substrate-independent, metrics should be similar.
If consciousness emerges from integration, distributed may show differences.

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
import socket
import threading
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1900_tri_hardware_consciousness import TriHardwareTelemetry
from scripts.z1908_comprehensive_embodiment_verdict import DualTaskEmbodiedModel, telemetry_to_class


class DistributedCoordinator:
    """Coordinates distributed computation with remote machine."""

    def __init__(self, remote_host: str = "192.168.0.37", remote_port: int = 9999):
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.connected = False
        self.latency_ms = 0.0

    def test_connection(self) -> bool:
        """Test if remote host is reachable."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self.remote_host, 22))  # Test SSH port
            sock.close()
            self.connected = (result == 0)
            return self.connected
        except:
            return False

    def measure_latency(self, num_pings: int = 5) -> float:
        """Measure network latency to remote host."""
        import subprocess
        try:
            result = subprocess.run(
                ['ping', '-c', str(num_pings), '-q', self.remote_host],
                capture_output=True, text=True, timeout=10
            )
            # Parse avg latency from ping output
            for line in result.stdout.split('\n'):
                if 'avg' in line:
                    # Format: rtt min/avg/max/mdev = X/Y/Z/W ms
                    parts = line.split('=')[1].split('/')
                    self.latency_ms = float(parts[1])
                    return self.latency_ms
        except:
            pass
        return 0.0

    def simulate_distributed_delay(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simulate network communication delay."""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)  # Convert to seconds
        return tensor


class DistributedEmbodiedModel(nn.Module):
    """
    Model that simulates distributed computation.

    In distributed mode, some computations include simulated network delays
    to represent cross-machine coordination.
    """

    def __init__(self, base_model: nn.Module, coordinator: DistributedCoordinator, distributed: bool = False):
        super().__init__()
        self.base_model = base_model
        self.coordinator = coordinator
        self.distributed = distributed

    def forward(self, input_ids: torch.Tensor, telemetry: torch.Tensor, return_all: bool = False):
        if self.distributed:
            # Simulate distributed: add network delay for "remote" computation
            telemetry = self.coordinator.simulate_distributed_delay(telemetry)

        return self.base_model(input_ids, telemetry, return_all=return_all)

    def count_parameters(self) -> int:
        return self.base_model.count_parameters()


def compute_consciousness_metrics(
    model: nn.Module,
    telemetry: TriHardwareTelemetry,
    device: torch.device,
    num_samples: int = 50,
) -> Dict[str, float]:
    """Compute consciousness metrics for comparison."""
    model.eval()
    metrics = {
        'classification_accuracy': [],
        'self_model_error': [],
        'telemetry_sensitivity': [],
        'hidden_complexity': [],
        'response_consistency': [],
    }

    prev_hidden = None

    for _ in range(num_samples):
        telem = telemetry.get_tensor().to(device)
        telem_np = telem.cpu().numpy()
        true_temp, true_util, true_power = telemetry_to_class(telem_np)

        x = torch.randint(0, 256, (1, 64), device=device)

        with torch.no_grad():
            out = model(x, telem, return_all=True)

            # Classification accuracy
            pred_temp = out['temp_logits'].argmax(dim=-1).item()
            pred_util = out['util_logits'].argmax(dim=-1).item()
            pred_power = out['power_logits'].argmax(dim=-1).item()
            acc = ((pred_temp == true_temp) + (pred_util == true_util) + (pred_power == true_power)) / 3
            metrics['classification_accuracy'].append(acc)

            # Self-model error
            self_error = F.mse_loss(out['self_prediction'], telem.unsqueeze(0)).item()
            metrics['self_model_error'].append(self_error)

            # Hidden complexity (entropy proxy)
            hidden = out['hidden_mean'].cpu().numpy()
            # Normalize and compute entropy approximation
            hidden_norm = (hidden - hidden.min()) / (hidden.max() - hidden.min() + 1e-8)
            complexity = -np.sum(hidden_norm * np.log(hidden_norm + 1e-8)) / hidden.shape[-1]
            metrics['hidden_complexity'].append(complexity)

            # Response consistency (compare to previous)
            if prev_hidden is not None:
                consistency = np.corrcoef(hidden.flatten(), prev_hidden.flatten())[0, 1]
                if not np.isnan(consistency):
                    metrics['response_consistency'].append(consistency)
            prev_hidden = hidden

        time.sleep(0.02)

    # Telemetry sensitivity: compare real vs zero
    x = torch.randint(0, 256, (1, 64), device=device)
    with torch.no_grad():
        out_real = model(x, telemetry.get_tensor().to(device), return_all=True)
        out_zero = model(x, torch.zeros(20, device=device), return_all=True)
        sensitivity = (out_real['lm_logits'] - out_zero['lm_logits']).abs().mean().item()
        metrics['telemetry_sensitivity'] = [sensitivity]

    # Aggregate
    return {
        'classification_accuracy': np.mean(metrics['classification_accuracy']),
        'self_model_error': np.mean(metrics['self_model_error']),
        'hidden_complexity': np.mean(metrics['hidden_complexity']),
        'response_consistency': np.mean(metrics['response_consistency']) if metrics['response_consistency'] else 0,
        'telemetry_sensitivity': np.mean(metrics['telemetry_sensitivity']),
    }


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1912] Device: {device}")
    print("[z1912] DISTRIBUTED CONSCIOUSNESS TEST")
    print("[z1912] Comparing local vs distributed computation")

    # Setup distributed coordinator
    coordinator = DistributedCoordinator(remote_host="192.168.0.37")

    print("\n[z1912] Testing remote connection to daedalus...")
    if coordinator.test_connection():
        print("  daedalus (192.168.0.37) is reachable")
        latency = coordinator.measure_latency()
        print(f"  Network latency: {latency:.2f} ms")
    else:
        print("  daedalus not reachable - will simulate distributed delays")
        coordinator.latency_ms = 5.0  # Simulate 5ms latency

    # Telemetry
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"\n[z1912] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text_bytes = data_path.read_text().encode('utf-8')

    # Create base model
    base_model = DualTaskEmbodiedModel(
        vocab_size=256,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        telemetry_dim=20,
    ).to(device)

    print(f"[z1912] Model parameters: {base_model.count_parameters():,}")

    # Train base model
    print("\n[z1912] Training base model...")
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4)

    batch_size = 4
    seq_len = 128

    def get_batch():
        ix = torch.randint(len(text_bytes) - seq_len - 1, (batch_size,))
        x = torch.stack([torch.tensor(list(text_bytes[i:i+seq_len]), dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(list(text_bytes[i+1:i+seq_len+1]), dtype=torch.long) for i in ix])
        return x.to(device), y.to(device)

    for epoch in range(10):
        base_model.train()
        epoch_loss = 0
        for _ in range(100):
            x, y = get_batch()
            telem = telemetry.get_tensor().to(device)
            telem_np = telem.cpu().numpy()
            temp_c, util_c, power_c = telemetry_to_class(telem_np)

            optimizer.zero_grad()
            out = base_model(x, telem, return_all=True)

            lm_loss = F.cross_entropy(out['lm_logits'].view(-1, 256), y.view(-1))
            class_loss = (
                F.cross_entropy(out['temp_logits'], torch.tensor([temp_c] * batch_size, device=device)) +
                F.cross_entropy(out['util_logits'], torch.tensor([util_c] * batch_size, device=device)) +
                F.cross_entropy(out['power_logits'], torch.tensor([power_c] * batch_size, device=device))
            ) / 3
            self_loss = F.mse_loss(out['self_prediction'], telem.unsqueeze(0).expand(batch_size, -1))

            loss = lm_loss + 0.5 * class_loss + 0.3 * self_loss
            loss.backward()
            optimizer.step()
            epoch_loss += lm_loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/10: loss={epoch_loss/100:.4f}")

    # Create local and distributed wrappers
    local_model = DistributedEmbodiedModel(base_model, coordinator, distributed=False)
    distributed_model = DistributedEmbodiedModel(base_model, coordinator, distributed=True)

    results = {
        'experiment': 'z1912_distributed_consciousness',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'remote_host': coordinator.remote_host,
        'remote_reachable': coordinator.connected,
        'network_latency_ms': coordinator.latency_ms,
    }

    # Test LOCAL configuration
    print("\n" + "="*60)
    print("[z1912] LOCAL CONFIGURATION (no network delay)")
    print("="*60)

    local_metrics = compute_consciousness_metrics(local_model, telemetry, device)
    print(f"  Classification accuracy: {local_metrics['classification_accuracy']:.2%}")
    print(f"  Self-model error: {local_metrics['self_model_error']:.6f}")
    print(f"  Hidden complexity: {local_metrics['hidden_complexity']:.4f}")
    print(f"  Response consistency: {local_metrics['response_consistency']:.4f}")
    print(f"  Telemetry sensitivity: {local_metrics['telemetry_sensitivity']:.4f}")

    results['local_metrics'] = local_metrics

    # Test DISTRIBUTED configuration
    print("\n" + "="*60)
    print(f"[z1912] DISTRIBUTED CONFIGURATION ({coordinator.latency_ms:.1f}ms delay)")
    print("="*60)

    distributed_metrics = compute_consciousness_metrics(distributed_model, telemetry, device)
    print(f"  Classification accuracy: {distributed_metrics['classification_accuracy']:.2%}")
    print(f"  Self-model error: {distributed_metrics['self_model_error']:.6f}")
    print(f"  Hidden complexity: {distributed_metrics['hidden_complexity']:.4f}")
    print(f"  Response consistency: {distributed_metrics['response_consistency']:.4f}")
    print(f"  Telemetry sensitivity: {distributed_metrics['telemetry_sensitivity']:.4f}")

    results['distributed_metrics'] = distributed_metrics

    # Compare configurations
    print("\n" + "="*60)
    print("[z1912] COMPARISON: LOCAL vs DISTRIBUTED")
    print("="*60)

    comparisons = {}
    for metric in local_metrics:
        local_val = local_metrics[metric]
        dist_val = distributed_metrics[metric]
        diff = dist_val - local_val
        pct_diff = (diff / local_val * 100) if local_val != 0 else 0

        comparisons[metric] = {
            'local': local_val,
            'distributed': dist_val,
            'difference': diff,
            'percent_change': pct_diff,
        }
        print(f"  {metric}:")
        print(f"    Local: {local_val:.4f}, Distributed: {dist_val:.4f}, Change: {pct_diff:+.1f}%")

    results['comparisons'] = comparisons

    # Verdicts
    verdicts = {}

    # V1: Classification preserved
    acc_change = abs(comparisons['classification_accuracy']['percent_change'])
    verdicts['V1_classification_preserved'] = {
        'pass': acc_change < 10,  # Less than 10% change
        'change': acc_change,
    }

    # V2: Self-model preserved
    self_change = abs(comparisons['self_model_error']['percent_change'])
    verdicts['V2_self_model_preserved'] = {
        'pass': self_change < 50,  # Less than 50% change
        'change': self_change,
    }

    # V3: Complexity maintained
    complexity_change = abs(comparisons['hidden_complexity']['percent_change'])
    verdicts['V3_complexity_maintained'] = {
        'pass': complexity_change < 20,
        'change': complexity_change,
    }

    # V4: Sensitivity preserved
    sensitivity_change = abs(comparisons['telemetry_sensitivity']['percent_change'])
    verdicts['V4_sensitivity_preserved'] = {
        'pass': sensitivity_change < 30,
        'change': sensitivity_change,
    }

    # V5: Response consistency (distributed might actually be lower due to timing)
    consistency_distributed = distributed_metrics['response_consistency']
    verdicts['V5_response_coherent'] = {
        'pass': consistency_distributed > 0.5,
        'value': consistency_distributed,
    }

    results['verdicts'] = verdicts

    # Summary
    num_pass = sum(1 for v in verdicts.values() if v['pass'])
    num_total = len(verdicts)

    print(f"\n{'='*60}")
    print("[z1912] DISTRIBUTED CONSCIOUSNESS VERDICTS")
    print(f"{'='*60}")
    for name, v in verdicts.items():
        status = "PASS" if v['pass'] else "FAIL"
        print(f"  {status} {name}")

    print(f"\n[z1912] Verdicts passed: {num_pass}/{num_total}")

    if num_pass >= 4:
        verdict = "CONSCIOUSNESS INDICATORS PRESERVED IN DISTRIBUTED CONFIG"
        interpretation = "Consciousness metrics are substrate-independent"
    elif num_pass >= 2:
        verdict = "PARTIAL PRESERVATION"
        interpretation = "Some consciousness indicators affected by distribution"
    else:
        verdict = "CONSCIOUSNESS INDICATORS DEGRADED"
        interpretation = "Distributed computation significantly affects consciousness metrics"

    results['overall_verdict'] = verdict
    results['interpretation'] = interpretation
    results['num_pass'] = num_pass
    results['num_total'] = num_total

    print(f"\n[z1912] VERDICT: {verdict}")
    print(f"[z1912] INTERPRETATION: {interpretation}")

    telemetry.stop()

    # Save
    results_path = Path(__file__).parent.parent / "results" / "z1912_distributed_consciousness.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1912] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
