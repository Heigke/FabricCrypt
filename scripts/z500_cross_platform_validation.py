#!/usr/bin/env python3
"""
Z500: Cross-Platform Energy Validation

Validates embodied conditional compute on:
- AMD GPUs (Tier-B: sysfs power integration)
- NVIDIA GPUs (Tier-A: NVML energy counters)

Runs identical workload and reports results with measurement tier.
"""

import os
import sys
import json
import time
import random
import socket
import platform
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from abc import ABC, abstractmethod

# Detect GPU vendor before importing torch
def detect_gpu_vendor() -> str:
    """Detect GPU vendor: 'amd', 'nvidia', or 'cpu'."""
    # Check for AMD first (ROCm) - check all card* devices
    drm_path = Path("/sys/class/drm")
    if drm_path.exists():
        for card in sorted(drm_path.glob("card[0-9]*")):
            vendor_file = card / "device" / "vendor"
            if vendor_file.exists():
                try:
                    vendor = vendor_file.read_text().strip()
                    if vendor == "0x1002":  # AMD vendor ID
                        return "amd"
                except:
                    pass

    # Check for NVIDIA
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if result.returncode == 0 and "GPU" in result.stdout:
            return "nvidia"
    except:
        pass

    return "cpu"


GPU_VENDOR = detect_gpu_vendor()
print(f"Detected GPU vendor: {GPU_VENDOR}")

# Set environment before torch import
if GPU_VENDOR == "amd":
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import vendor-specific telemetry
if GPU_VENDOR == "amd":
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter, GpuSample
    ENERGY_TIER = "tier_b"
    ENERGY_METHOD = "amd_sysfs_power_integration"
elif GPU_VENDOR == "nvidia":
    from src.sensing.nvml_energy import NVMLEnergyMeter, NVMLBackgroundSampler
    ENERGY_TIER = "tier_a"  # Will be updated based on actual capability
    ENERGY_METHOD = "nvml_energy_counter"
else:
    ENERGY_TIER = "unavailable"
    ENERGY_METHOD = "none"

from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset


# =============================================================================
# Unified Telemetry Interface
# =============================================================================

@dataclass
class UnifiedSample:
    """Unified GPU sample across vendors."""
    timestamp_ns: int
    power_w: float
    temp_c: float
    gpu_busy_pct: float = 0.0
    mem_used_gb: float = 0.0


class UnifiedTelemetry(ABC):
    """Abstract base for cross-platform telemetry."""

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_latest_sample(self) -> Optional[UnifiedSample]:
        pass

    @abstractmethod
    def get_energy_j(self) -> float:
        pass

    @abstractmethod
    def get_tier(self) -> str:
        pass

    @abstractmethod
    def reset(self):
        pass


class AMDUnifiedTelemetry(UnifiedTelemetry):
    """AMD telemetry via sysfs hwmon."""

    def __init__(self, sample_rate_hz: float = 50.0):
        self.sysfs = SysfsHwmonTelemetry(sample_rate_hz=sample_rate_hz)
        self._running = False

    def start(self):
        self.sysfs.reset_accumulator()
        self.sysfs.start_continuous_sampling()
        self._running = True

    def stop(self):
        self.sysfs.stop_continuous_sampling()
        self._running = False

    def get_latest_sample(self) -> Optional[UnifiedSample]:
        sample = self.sysfs.get_latest_sample()
        if sample:
            return UnifiedSample(
                timestamp_ns=sample.timestamp_ns,
                power_w=sample.power_w,
                temp_c=sample.temp_edge_c,
                gpu_busy_pct=sample.gpu_busy_pct,
                mem_used_gb=sample.vram_used_gb,
            )
        return None

    def get_energy_j(self) -> float:
        return self.sysfs.get_accumulated_energy_j()

    def get_tier(self) -> str:
        return "tier_b"

    def reset(self):
        self.sysfs.reset_accumulator()

    def measure_idle(self, duration_s: float = 2.0) -> float:
        return self.sysfs.measure_idle_baseline(duration_s)


class NVIDIAUnifiedTelemetry(UnifiedTelemetry):
    """NVIDIA telemetry via NVML."""

    def __init__(self, device_index: int = 0, sample_rate_hz: float = 50.0):
        self.device_index = device_index
        self.meter = NVMLEnergyMeter(device_index)
        self.sampler = NVMLBackgroundSampler(device_index, sample_rate_hz)
        self._start_time = 0
        self._running = False

        # Check actual tier
        self._tier = "tier_a" if self.meter._has_energy_counter else "tier_b"

    def start(self):
        self.meter.start()
        self.sampler.start()
        self._start_time = time.time_ns()
        self._running = True

    def stop(self):
        self.sampler.stop()
        self._result = self.meter.stop()
        self._running = False

    def get_latest_sample(self) -> Optional[UnifiedSample]:
        latest = self.sampler.get_latest()
        return UnifiedSample(
            timestamp_ns=time.time_ns(),
            power_w=latest['power_w'],
            temp_c=latest['temp_c'],
            gpu_busy_pct=latest['util_pct'],
            mem_used_gb=latest['mem_gb'],
        )

    def get_energy_j(self) -> float:
        if hasattr(self, '_result'):
            return self._result.get('energy_mj', 0) / 1000  # mJ to J
        return 0.0

    def get_tier(self) -> str:
        return self._tier

    def reset(self):
        pass  # NVML meter resets on start

    def get_device_info(self) -> dict:
        return self.meter.get_device_info()


def create_telemetry(sample_rate_hz: float = 50.0) -> UnifiedTelemetry:
    """Factory to create appropriate telemetry for detected GPU."""
    if GPU_VENDOR == "amd":
        return AMDUnifiedTelemetry(sample_rate_hz)
    elif GPU_VENDOR == "nvidia":
        return NVIDIAUnifiedTelemetry(sample_rate_hz=sample_rate_hz)
    else:
        raise RuntimeError(f"No GPU telemetry available for vendor: {GPU_VENDOR}")


# =============================================================================
# Simple Controller (no external dependencies)
# =============================================================================

class SimpleFixedController:
    """Fixed exit layer controller."""

    def __init__(self, exit_layer: int = 12):
        self.exit_layer = exit_layer
        self._latencies = []

    def start(self):
        pass

    def stop(self):
        pass

    def get_compute_action(self, uncertainty: float = 0.0) -> int:
        return self.exit_layer

    def report_latency(self, latency_ms: float):
        self._latencies.append(latency_ms)

    def get_statistics(self) -> dict:
        return {'exit_layer': self.exit_layer, 'body_pressure': 0.0}


class SimpleAdaptiveController:
    """
    Simple two-signal adaptive controller.

    Combines uncertainty (semantic) and body pressure (physical) to choose exit layer.
    """

    def __init__(self, telemetry: UnifiedTelemetry,
                 temp_cap: float = 85.0,
                 power_cap: float = 200.0,
                 latency_threshold_ms: float = 400.0):
        self.telemetry = telemetry
        self.temp_cap = temp_cap
        self.power_cap = power_cap
        self.latency_threshold_ms = latency_threshold_ms

        # State
        self.body_pressure = 0.0
        self.last_latency_ms = 0.0
        self._latencies = []

        # Exit layer mapping: pressure -> layer
        self.exit_thresholds = [
            (0.0, 12),   # No pressure -> full compute
            (0.25, 9),   # Low pressure
            (0.5, 6),    # Medium pressure
            (0.75, 3),   # High pressure
        ]

    def start(self):
        pass

    def stop(self):
        pass

    def _update_body_pressure(self):
        """Update body pressure from telemetry."""
        sample = self.telemetry.get_latest_sample()
        if not sample:
            return

        # Thermal pressure
        thermal_margin = self.temp_cap - sample.temp_c
        thermal_pressure = max(0.0, min(1.0, 1.0 - thermal_margin / 20.0))

        # Power pressure
        power_pressure = max(0.0, min(1.0, sample.power_w / self.power_cap))

        # Latency pressure
        if self.last_latency_ms > self.latency_threshold_ms:
            latency_pressure = min(1.0, (self.last_latency_ms - self.latency_threshold_ms) / 200.0)
        else:
            latency_pressure = 0.0

        # GPU utilization pressure
        gpu_pressure = max(0.0, (sample.gpu_busy_pct - 80.0) / 20.0) if sample.gpu_busy_pct > 80 else 0.0

        # Combined pressure (max of all signals)
        self.body_pressure = max(thermal_pressure, power_pressure, latency_pressure, gpu_pressure)
        self.body_pressure = max(0.0, min(1.0, self.body_pressure))

    def get_compute_action(self, uncertainty: float = 0.0) -> int:
        """Get exit layer based on uncertainty and body state."""
        self._update_body_pressure()

        # Combine uncertainty and body pressure
        # High uncertainty -> need more compute
        # High body pressure -> need less compute
        combined_signal = self.body_pressure - (1.0 - uncertainty) * 0.3
        combined_signal = max(0.0, min(1.0, combined_signal))

        # Map to exit layer
        exit_layer = 12
        for threshold, layer in self.exit_thresholds:
            if combined_signal >= threshold:
                exit_layer = layer

        return exit_layer

    def report_latency(self, latency_ms: float):
        self.last_latency_ms = latency_ms
        self._latencies.append(latency_ms)
        self._update_body_pressure()

    def get_statistics(self) -> dict:
        return {
            'body_pressure': self.body_pressure,
            'last_latency_ms': self.last_latency_ms,
        }


# =============================================================================
# Early Exit Model
# =============================================================================

class EarlyExitModel(torch.nn.Module):
    """Model with early exit capability."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.vocab_size = base_model.config.vocab_size
        self.n_layers = len(base_model.transformer.h)

    def forward(self, input_ids, exit_layer=12):
        if exit_layer >= self.n_layers:
            return self.base_model(input_ids).logits

        # Early exit
        hidden = self.base_model.transformer.wte(input_ids)
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = hidden + self.base_model.transformer.wpe(pos)
        hidden = self.base_model.transformer.drop(hidden)

        for block in self.base_model.transformer.h[:exit_layer]:
            hidden = block(hidden)[0]

        hidden = self.base_model.transformer.ln_f(hidden)
        return self.base_model.lm_head(hidden)

    def compute_loss_and_uncertainty(self, input_ids, exit_layer=12):
        """Compute loss and uncertainty (entropy of predictions)."""
        logits = self.forward(input_ids, exit_layer)

        # Loss
        labels = input_ids.clone()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )

        # Uncertainty: entropy of last token prediction
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = np.log(self.vocab_size)
        uncertainty = (entropy.mean() / max_entropy).item()

        return loss.item(), uncertainty


# =============================================================================
# Validation
# =============================================================================

@dataclass
class ValidationResult:
    """Results from a validation run."""
    name: str
    controller_type: str

    # System info
    hostname: str = ""
    gpu_vendor: str = ""
    gpu_name: str = ""
    energy_tier: str = ""
    energy_method: str = ""

    # Performance
    total_tokens: int = 0
    total_time_s: float = 0.0
    throughput_tok_s: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Energy
    energy_j: float = 0.0
    energy_per_token_mj: float = 0.0
    avg_power_w: float = 0.0

    # Quality
    avg_loss: float = 0.0

    # Compute
    avg_exit_layer: float = 12.0
    exit_distribution: Dict[int, int] = field(default_factory=dict)
    compute_reduction_pct: float = 0.0

    # Constraints
    temp_violations: int = 0
    latency_violations: int = 0
    max_temp_c: float = 0.0

    # Adaptivity
    exit_variance: float = 0.0
    body_pressure_correlation: float = 0.0


def run_validation(
    model: EarlyExitModel,
    batches: List[torch.Tensor],
    controller,
    telemetry: UnifiedTelemetry,
    name: str = "test",
    temp_cap: float = 85.0,
    latency_cap_ms: float = 500.0,
    warmup_batches: int = 5,
) -> ValidationResult:
    """Run validation and collect metrics."""

    model.eval()
    result = ValidationResult(
        name=name,
        controller_type=type(controller).__name__,
        hostname=socket.gethostname(),
        gpu_vendor=GPU_VENDOR,
        energy_tier=telemetry.get_tier(),
        energy_method=ENERGY_METHOD,
    )

    # Get GPU name
    if GPU_VENDOR == "nvidia" and hasattr(telemetry, 'get_device_info'):
        info = telemetry.get_device_info()
        result.gpu_name = info.get('name', 'Unknown')
    elif GPU_VENDOR == "amd":
        result.gpu_name = "AMD gfx1151"

    # Tracking
    latencies = []
    losses = []
    exit_layers = []
    temps = []
    body_pressures = []
    total_tokens = 0

    # Warmup
    print(f"    Warmup ({warmup_batches} batches)...")
    for batch in batches[:warmup_batches]:
        with torch.no_grad():
            _ = model.forward(batch, exit_layer=12)
        torch.cuda.synchronize()

    time.sleep(2)  # Cool down

    # Main validation
    print(f"    Running {len(batches)} batches...")
    telemetry.reset()
    telemetry.start()
    controller.start()
    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"      {name}", leave=False)):
        batch_start = time.perf_counter()

        with torch.no_grad():
            # Compute uncertainty
            _, uncertainty = model.compute_loss_and_uncertainty(batch, exit_layer=12)

            # Record body pressure at decision time
            stats = controller.get_statistics()
            if 'body_pressure' in stats:
                body_pressures.append(stats['body_pressure'])

            # Get exit layer
            exit_layer = controller.get_compute_action(uncertainty=uncertainty)

            # Forward with chosen exit
            loss, _ = model.compute_loss_and_uncertainty(batch, exit_layer)

        torch.cuda.synchronize()
        batch_latency = (time.perf_counter() - batch_start) * 1000

        # Record telemetry
        sample = telemetry.get_latest_sample()
        if sample:
            temps.append(sample.temp_c)
            if sample.temp_c > temp_cap:
                result.temp_violations += 1

        if batch_latency > latency_cap_ms:
            result.latency_violations += 1

        latencies.append(batch_latency)
        losses.append(loss)
        exit_layers.append(exit_layer)
        total_tokens += (batch != 50256).sum().item()  # Non-pad tokens

        controller.report_latency(batch_latency)

    end_time = time.perf_counter()
    controller.stop()
    telemetry.stop()

    # Compute metrics
    duration_s = end_time - start_time
    result.total_tokens = total_tokens
    result.total_time_s = duration_s
    result.throughput_tok_s = total_tokens / duration_s if duration_s > 0 else 0
    result.latency_p50_ms = float(np.percentile(latencies, 50))
    result.latency_p99_ms = float(np.percentile(latencies, 99))

    result.energy_j = telemetry.get_energy_j()
    result.energy_per_token_mj = (result.energy_j * 1000) / total_tokens if total_tokens > 0 else 0
    result.avg_power_w = result.energy_j / duration_s if duration_s > 0 else 0

    result.avg_loss = float(np.mean(losses))
    result.avg_exit_layer = float(np.mean(exit_layers))
    result.compute_reduction_pct = (12 - result.avg_exit_layer) / 12 * 100

    result.exit_distribution = {int(k): int(v) for k, v in
                                 zip(*np.unique(exit_layers, return_counts=True))}

    result.max_temp_c = max(temps) if temps else 0
    result.exit_variance = float(np.var(exit_layers))

    # Body-compute correlation
    if body_pressures and len(body_pressures) == len(exit_layers):
        if np.std(body_pressures) > 0 and np.std(exit_layers) > 0:
            result.body_pressure_correlation = float(-np.corrcoef(body_pressures, exit_layers)[0, 1])

    return result


def main():
    print("=" * 80)
    print("Z500: CROSS-PLATFORM ENERGY VALIDATION")
    print("=" * 80)

    print(f"\nSystem: {socket.gethostname()}")
    print(f"GPU Vendor: {GPU_VENDOR}")
    print(f"Energy Tier: {ENERGY_TIER}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Device: {device}")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available")
        return

    # Initialize telemetry
    print("\n--- Telemetry ---")
    telemetry = create_telemetry(sample_rate_hz=50.0)
    print(f"Telemetry tier: {telemetry.get_tier()}")

    if GPU_VENDOR == "nvidia" and hasattr(telemetry, 'get_device_info'):
        info = telemetry.get_device_info()
        print(f"GPU: {info.get('name', 'Unknown')}")
        print(f"Has energy counter: {info.get('has_energy_counter', False)}")

    # Measure idle
    if GPU_VENDOR == "amd" and hasattr(telemetry, 'measure_idle'):
        idle_power = telemetry.measure_idle(2.0)
        print(f"Idle power: {idle_power:.1f} W")

    # Load model
    print("\n--- Model ---")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model = EarlyExitModel(base_model).to(device)
    model.eval()
    print(f"Model: GPT-2 ({model.n_layers} layers)")

    # Prepare data
    print("\n--- Data ---")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    texts = []
    for item in dataset:
        if len(texts) >= 400:  # 400 texts = 50 batches of 8
            break
        text = item['text'][:500]
        if len(text) > 50:
            texts.append(text)

    batch_size = 8
    seq_len = 128
    batches = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        if len(batch_texts) == batch_size:
            encoded = tokenizer(
                batch_texts,
                max_length=seq_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            batches.append(encoded['input_ids'].to(device))

    print(f"Prepared {len(batches)} batches ({batch_size} x {seq_len})")

    # Run validation
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION")
    print("=" * 80)

    results = {}

    # Fixed baselines
    for exit_layer in [3, 6, 9, 12]:
        print(f"\n--- Fixed L{exit_layer} ---")
        controller = SimpleFixedController(exit_layer)
        results[f'fixed_L{exit_layer}'] = run_validation(
            model, batches, controller, telemetry,
            name=f'fixed_L{exit_layer}'
        )
        time.sleep(3)  # Cool down

    # Adaptive controller
    print("\n--- Adaptive ---")
    controller = SimpleAdaptiveController(telemetry)
    results['adaptive'] = run_validation(
        model, batches, controller, telemetry,
        name='adaptive'
    )

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    baseline = results['fixed_L12']

    print(f"\n{'Controller':<15} {'Tok/s':>8} {'mJ/tok':>8} {'Loss':>7} {'Exit':>6} {'Var':>6}")
    print("-" * 60)

    for name, r in sorted(results.items()):
        print(f"{name:<15} {r.throughput_tok_s:>8.0f} {r.energy_per_token_mj:>8.2f} "
              f"{r.avg_loss:>7.3f} {r.avg_exit_layer:>6.1f} {r.exit_variance:>6.2f}")

    # Falsifiable checks
    print("\n" + "=" * 80)
    print("FALSIFIABLE PROPERTY CHECKS")
    print("=" * 80)

    adaptive = results['adaptive']

    # Check 1: Exit variance (adaptive should vary)
    print(f"\n1. Intervention Sensitivity")
    print(f"   Exit variance: {adaptive.exit_variance:.3f}")
    intervention_pass = adaptive.exit_variance > 0.1
    print(f"   {'PASS' if intervention_pass else 'FAIL'}: {'Adaptive varies exit' if intervention_pass else 'Behaves like fixed'}")

    # Check 2: Body correlation
    print(f"\n2. Body-Compute Correlation")
    print(f"   Correlation: {adaptive.body_pressure_correlation:.3f}")
    correlation_pass = adaptive.body_pressure_correlation > 0.05
    print(f"   {'PASS' if correlation_pass else 'FAIL'}: {'Body influences compute' if correlation_pass else 'No body influence'}")

    # Check 3: Energy comparison
    print(f"\n3. Energy Efficiency")
    energy_vs_L12 = (1 - adaptive.energy_per_token_mj / baseline.energy_per_token_mj) * 100
    print(f"   vs L12: {energy_vs_L12:+.1f}% energy")

    # Save results
    output_path = Path(f"results/z500_{socket.gethostname()}_{GPU_VENDOR}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'system': {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'gpu_vendor': GPU_VENDOR,
            'energy_tier': ENERGY_TIER,
            'energy_method': ENERGY_METHOD,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': {name: asdict(r) for name, r in results.items()},
        'falsifiable_checks': {
            'intervention_sensitivity': intervention_pass,
            'body_correlation': correlation_pass,
            'adaptive_exit_variance': adaptive.exit_variance,
            'adaptive_body_correlation': adaptive.body_pressure_correlation,
            'energy_vs_baseline_pct': energy_vs_L12,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
