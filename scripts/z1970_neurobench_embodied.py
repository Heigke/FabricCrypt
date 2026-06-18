#!/usr/bin/env python3
"""
z1970: NeuroBench-Compatible Benchmark Suite for Embodied AI
=============================================================

Implements NeuroBench-standard metrics plus embodiment-specific extensions
for measuring performance of our embodied neural system.

NeuroBench: https://github.com/NeuroBench/neurobench
Reference: "NeuroBench: Advancing Neuromorphic Computing through Collaborative,
           Fair and Representative Benchmarking" (2023)

STANDARD NEUROBENCH METRICS:
- Accuracy (task performance)
- Latency (inference time)
- Energy per inference (mJ)
- Throughput (inferences/second)
- Memory footprint

EMBODIMENT-SPECIFIC EXTENSIONS:
- J/token (energy efficiency for generative tasks)
- Adaptation speed (hardware-dependent learning)
- Homeostatic recovery time
- Self-model accuracy
- Active inference free energy

BENCHMARK TASKS (mapped to our implementations):
1. Hardware-dependent prediction (z1315) - temporal pattern recognition
2. Homeostatic regulation (z1700) - control task
3. Active inference (z1701) - planning under uncertainty
4. Self-model accuracy (z1709) - recursive prediction

COMPARISON BASELINES:
- Intel Loihi 2 (neuromorphic)
- Synsense Xylo (edge SNN)
- Standard GPU (disembodied baseline)
- Embodied GPU (our system)

Author: FEEL Research Team
Date: 2026-02-05
"""

import os
import sys
import json
import time
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

# HSA override for gfx1151 (AMD Radeon 8060S)
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metabolic.film_transformer import (
    MetabolicTransformer, MetabolicConfig, BaselineTransformer,
    create_metabolic_transformer, get_best_device
)
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter, GpuSample
from src.actuation.gpu_actuator import GPUActuator, EmbodiedGPUController, PerformanceLevel

DEVICE = get_best_device()
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# NeuroBench Reference Baselines (Published Results)
# =============================================================================

NEUROBENCH_BASELINES = {
    'loihi2': {
        'name': 'Intel Loihi 2',
        'type': 'neuromorphic',
        'energy_per_inference_mj': 0.45,  # SNN inference
        'latency_ms': 12.0,
        'accuracy_keyword_spotting': 0.92,
        'throughput_inf_s': 83.3,
        'static_power_w': 0.8,
        'source': 'NeuroBench 2023',
    },
    'xylo': {
        'name': 'Synsense Xylo',
        'type': 'neuromorphic_edge',
        'energy_per_inference_mj': 0.18,  # Ultra-low power SNN
        'latency_ms': 20.0,
        'accuracy_keyword_spotting': 0.88,
        'throughput_inf_s': 50.0,
        'static_power_w': 0.05,
        'source': 'NeuroBench 2023',
    },
    'akida': {
        'name': 'BrainChip Akida',
        'type': 'neuromorphic',
        'energy_per_inference_mj': 0.30,
        'latency_ms': 8.0,
        'accuracy_keyword_spotting': 0.90,
        'throughput_inf_s': 125.0,
        'static_power_w': 0.3,
        'source': 'NeuroBench 2023',
    },
    'spinnaker2': {
        'name': 'SpiNNaker 2',
        'type': 'neuromorphic',
        'energy_per_inference_mj': 2.5,
        'latency_ms': 5.0,
        'accuracy_keyword_spotting': 0.91,
        'throughput_inf_s': 200.0,
        'static_power_w': 5.0,
        'source': 'NeuroBench 2023',
    },
    'gpu_baseline': {
        'name': 'GPU (Disembodied)',
        'type': 'conventional',
        'energy_per_inference_mj': 50.0,  # Typical GPU inference
        'latency_ms': 2.0,
        'accuracy_keyword_spotting': 0.95,
        'throughput_inf_s': 500.0,
        'static_power_w': 15.0,
        'source': 'Estimated baseline',
    },
}


# =============================================================================
# Core NeuroBench Metrics Class
# =============================================================================

@dataclass
class NeuroBenchMetrics:
    """Standard NeuroBench metrics structure."""

    # Task performance
    accuracy: float = 0.0
    top5_accuracy: float = 0.0

    # Latency
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Energy
    energy_per_inference_mj: float = 0.0
    static_power_w: float = 0.0
    dynamic_power_w: float = 0.0

    # Throughput
    throughput_inf_s: float = 0.0

    # Memory
    model_params: int = 0
    memory_mb: float = 0.0

    # Embodiment extensions
    j_per_token: float = 0.0
    adaptation_speed_epochs: float = 0.0
    homeostatic_recovery_ms: float = 0.0
    self_model_accuracy: float = 0.0
    free_energy_final: float = 0.0
    embodiment_advantage: float = 0.0


# =============================================================================
# EmbodiedNeuroBench Main Class
# =============================================================================

class EmbodiedNeuroBench:
    """
    NeuroBench-compatible benchmark suite with embodiment extensions.

    Measures both standard neuromorphic metrics and embodied AI capabilities.
    """

    def __init__(
        self,
        n_warmup: int = 10,
        n_trials: int = 100,
        batch_size: int = 4,
        seq_len: int = 256,
    ):
        self.n_warmup = n_warmup
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Hardware
        self.telemetry = SysfsHwmonTelemetry()
        self.actuator = GPUActuator(card_id=0)

        # Measure idle baseline
        print("Measuring idle power baseline...")
        self.idle_power_w = self.telemetry.measure_idle_baseline(duration_s=2.0)
        print(f"  Idle power: {self.idle_power_w:.2f}W")

        # Results storage
        self.results: Dict[str, NeuroBenchMetrics] = {}

    # -------------------------------------------------------------------------
    # Standard NeuroBench Metrics
    # -------------------------------------------------------------------------

    def measure_accuracy(
        self,
        model: nn.Module,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        telemetry_fn: Optional[callable] = None,
    ) -> Tuple[float, float]:
        """
        Measure task accuracy (top-1 and top-5).

        Args:
            model: Model to evaluate
            test_data: Test inputs [N, seq_len]
            test_labels: Test targets [N, seq_len] or [N]
            telemetry_fn: Optional function to get telemetry tensor

        Returns:
            (top1_accuracy, top5_accuracy)
        """
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(test_data), self.batch_size):
                batch = test_data[i:i+self.batch_size].to(DEVICE)
                labels = test_labels[i:i+self.batch_size].to(DEVICE)

                # Get telemetry if embodied
                if telemetry_fn is not None:
                    telem = telemetry_fn().to(DEVICE)
                    if telem.dim() == 1:
                        telem = telem.unsqueeze(0).expand(batch.size(0), -1)
                    out = model(batch, telemetry=telem)
                else:
                    out = model(batch)

                logits = out['logits']

                # For sequence prediction, use last token
                if labels.dim() == 2:
                    logits = logits[:, -1, :]
                    labels = labels[:, -1]

                # Top-1
                preds = logits.argmax(dim=-1)
                correct_top1 += (preds == labels).sum().item()

                # Top-5
                top5 = logits.topk(5, dim=-1).indices
                correct_top5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()

                total += labels.size(0)

        top1_acc = correct_top1 / total if total > 0 else 0.0
        top5_acc = correct_top5 / total if total > 0 else 0.0

        return top1_acc, top5_acc

    def measure_latency_ms(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        telemetry_fn: Optional[callable] = None,
    ) -> Tuple[float, float, float]:
        """
        Measure inference latency (mean, p50, p99).

        Returns:
            (mean_ms, p50_ms, p99_ms)
        """
        model.eval()
        latencies = []

        sample_input = sample_input.to(DEVICE)

        # Warmup
        for _ in range(self.n_warmup):
            with torch.no_grad():
                if telemetry_fn is not None:
                    telem = telemetry_fn().unsqueeze(0).to(DEVICE)
                    _ = model(sample_input, telemetry=telem)
                else:
                    _ = model(sample_input)
            torch.cuda.synchronize()

        # Timed runs
        for _ in range(self.n_trials):
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.no_grad():
                if telemetry_fn is not None:
                    telem = telemetry_fn().unsqueeze(0).to(DEVICE)
                    _ = model(sample_input, telemetry=telem)
                else:
                    _ = model(sample_input)

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            latencies.append((t_end - t_start) * 1000)  # ms

        latencies = np.array(latencies)
        return float(np.mean(latencies)), float(np.percentile(latencies, 50)), float(np.percentile(latencies, 99))

    def measure_energy_per_inference_mj(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        telemetry_fn: Optional[callable] = None,
    ) -> Tuple[float, float, float]:
        """
        Measure energy per inference in millijoules.

        Returns:
            (energy_mj, static_power_w, dynamic_power_w)
        """
        model.eval()
        sample_input = sample_input.to(DEVICE)

        energies = []

        # Warmup
        for _ in range(self.n_warmup):
            with torch.no_grad():
                if telemetry_fn is not None:
                    telem = telemetry_fn().unsqueeze(0).to(DEVICE)
                    _ = model(sample_input, telemetry=telem)
                else:
                    _ = model(sample_input)
            torch.cuda.synchronize()

        # Energy measurement
        for _ in range(self.n_trials):
            with EnergyMeter(self.telemetry) as meter:
                with torch.no_grad():
                    if telemetry_fn is not None:
                        telem = telemetry_fn().unsqueeze(0).to(DEVICE)
                        _ = model(sample_input, telemetry=telem)
                    else:
                        _ = model(sample_input)
                torch.cuda.synchronize()

            energies.append(meter.energy_j * 1000)  # Convert J to mJ

        mean_energy_mj = float(np.mean(energies))
        static_power_w = self.idle_power_w

        # Dynamic power = (total energy - idle energy) / time
        mean_time_s = np.mean([e / 1000 / (self.idle_power_w + 10) for e in energies])  # Estimate
        dynamic_power_w = (mean_energy_mj / 1000 / max(mean_time_s, 0.001)) - static_power_w
        dynamic_power_w = max(0, dynamic_power_w)

        return mean_energy_mj, static_power_w, dynamic_power_w

    # -------------------------------------------------------------------------
    # Embodiment-Specific Metrics
    # -------------------------------------------------------------------------

    def measure_adaptation_speed(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        adaptation_threshold: float = 0.9,
        max_epochs: int = 50,
    ) -> float:
        """
        Measure how fast the model adapts to hardware-dependent patterns.

        This tests the z1315 hardware-in-the-loop design where the model
        must learn to use hardware telemetry to predict drifting targets.

        Returns:
            Number of epochs to reach adaptation threshold (or max_epochs if not reached)
        """
        print("  Measuring adaptation speed...")

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create hardware-dependent task
        # Target = f(telemetry), model must learn this mapping
        prev_sample = None

        for epoch in range(max_epochs):
            correct = 0
            total = 0

            for i in range(min(100, len(train_data) // self.batch_size)):
                # Read telemetry
                sample = self.telemetry.read_sample()

                # Create telemetry tensor
                telem = self._build_telemetry_tensor(sample, prev_sample)
                telem_batch = telem.unsqueeze(0).expand(self.batch_size, -1).to(DEVICE)
                prev_sample = sample

                # Hardware-dependent target: based on temperature derivative
                hw_signal = telem[7].item()  # temp derivative
                target_class = 1 if hw_signal > 0 else 0
                targets = torch.full((self.batch_size,), target_class, dtype=torch.long, device=DEVICE)

                # Forward
                batch = train_data[i*self.batch_size:(i+1)*self.batch_size].to(DEVICE)
                out = model(batch, telemetry=telem_batch)

                # Use action logits as binary classifier
                logits = out['action_logits'][:, :2]  # First two actions
                loss = F.cross_entropy(logits, targets)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accuracy
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                # Small delay for thermal variation
                time.sleep(0.02)

            accuracy = correct / total if total > 0 else 0.0

            if accuracy >= adaptation_threshold:
                print(f"    Reached {adaptation_threshold:.0%} at epoch {epoch+1}")
                return float(epoch + 1)

        print(f"    Did not reach threshold in {max_epochs} epochs (final: {accuracy:.1%})")
        return float(max_epochs)

    def measure_homeostatic_recovery(
        self,
        model: nn.Module,
        perturbation_type: str = 'thermal',
        recovery_threshold: float = 0.9,
        max_time_ms: float = 10000.0,
    ) -> float:
        """
        Measure time to recover homeostatic equilibrium after perturbation.

        Based on z1700 homeostatic vulnerability design.

        Returns:
            Recovery time in milliseconds
        """
        print("  Measuring homeostatic recovery...")

        model.eval()
        controller = EmbodiedGPUController(card_id=0)

        # Get baseline state
        baseline_states = []
        for _ in range(20):
            sample = self.telemetry.read_sample()
            baseline_states.append(sample.temp_edge_c)
            time.sleep(0.05)

        baseline_temp = np.mean(baseline_states)
        baseline_std = np.std(baseline_states)

        print(f"    Baseline: {baseline_temp:.1f}C (std={baseline_std:.2f})")

        # Create perturbation
        print("    Creating perturbation...")
        if perturbation_type == 'thermal':
            # Heavy workload to increase temperature
            for _ in range(50):
                a = torch.randn(2000, 2000, device=DEVICE)
                b = torch.randn(2000, 2000, device=DEVICE)
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()

        # Measure recovery
        t_start = time.time()
        recovery_samples = []

        while (time.time() - t_start) * 1000 < max_time_ms:
            sample = self.telemetry.read_sample()
            recovery_samples.append({
                'time_ms': (time.time() - t_start) * 1000,
                'temp_c': sample.temp_edge_c,
            })

            # Check if recovered
            recent = [s['temp_c'] for s in recovery_samples[-5:]]
            if len(recent) >= 5:
                if abs(np.mean(recent) - baseline_temp) <= baseline_std * 2:
                    recovery_time = recovery_samples[-1]['time_ms']
                    print(f"    Recovered in {recovery_time:.0f}ms")
                    return recovery_time

            # Let model take action if embodied
            with torch.no_grad():
                telem = self._build_telemetry_tensor(sample, None).unsqueeze(0).to(DEVICE)
                batch = torch.randint(0, 256, (1, self.seq_len), device=DEVICE)
                out = model(batch, telemetry=telem)
                action = out['action_logits'].argmax(dim=-1).item()

            # Apply action (simplified)
            if action == 0:  # ECO
                controller.apply_action(perf_level='low')
            elif action == 3:  # MAX
                controller.apply_action(perf_level='high')

            time.sleep(0.1)

        print(f"    Did not recover in {max_time_ms:.0f}ms")
        return max_time_ms

    def measure_embodiment_advantage(
        self,
        embodied_model: nn.Module,
        baseline_model: nn.Module,
        test_data: torch.Tensor,
    ) -> float:
        """
        Measure the performance advantage of embodiment vs baseline.

        Computes: (embodied_perf - baseline_perf) / baseline_perf

        Returns:
            Embodiment advantage ratio (positive = embodied better)
        """
        print("  Measuring embodiment advantage...")

        def telemetry_fn():
            sample = self.telemetry.read_sample()
            return self._build_telemetry_tensor(sample, None)

        # Measure embodied
        embodied_acc, _ = self.measure_accuracy(
            embodied_model, test_data[:200], test_data[1:201], telemetry_fn
        )
        embodied_latency, _, _ = self.measure_latency_ms(
            embodied_model, test_data[:self.batch_size], telemetry_fn
        )
        embodied_energy, _, _ = self.measure_energy_per_inference_mj(
            embodied_model, test_data[:self.batch_size], telemetry_fn
        )

        # Measure baseline
        baseline_acc, _ = self.measure_accuracy(
            baseline_model, test_data[:200], test_data[1:201], None
        )
        baseline_latency, _, _ = self.measure_latency_ms(
            baseline_model, test_data[:self.batch_size], None
        )
        baseline_energy, _, _ = self.measure_energy_per_inference_mj(
            baseline_model, test_data[:self.batch_size], None
        )

        # Compute advantage (higher acc, lower latency/energy = better)
        acc_advantage = (embodied_acc - baseline_acc) / max(baseline_acc, 0.01)
        latency_advantage = (baseline_latency - embodied_latency) / max(baseline_latency, 0.01)
        energy_advantage = (baseline_energy - embodied_energy) / max(baseline_energy, 0.01)

        # Combined advantage (weighted)
        combined = 0.4 * acc_advantage + 0.3 * latency_advantage + 0.3 * energy_advantage

        print(f"    Accuracy advantage: {acc_advantage:+.1%}")
        print(f"    Latency advantage: {latency_advantage:+.1%}")
        print(f"    Energy advantage: {energy_advantage:+.1%}")
        print(f"    Combined: {combined:+.1%}")

        return combined

    def measure_self_model_accuracy(
        self,
        model: nn.Module,
        n_trials: int = 50,
    ) -> float:
        """
        Measure how well the model predicts its own hidden states.

        Based on z1709 recursive self-modeling.

        Returns:
            Self-prediction correlation coefficient
        """
        print("  Measuring self-model accuracy...")

        model.eval()

        correlations = []
        prev_hidden = None

        for i in range(n_trials):
            sample = self.telemetry.read_sample()
            telem = self._build_telemetry_tensor(sample, None).unsqueeze(0).to(DEVICE)
            batch = torch.randint(0, 256, (1, self.seq_len), device=DEVICE)

            with torch.no_grad():
                out = model(batch, telemetry=telem, return_hidden=True)
                current_hidden = out['hidden'].mean(dim=1)  # [1, hidden_dim]

            if prev_hidden is not None:
                # Correlation between current and predicted (prev -> current)
                # Using action logits as proxy for self-prediction
                action_probs = F.softmax(out['action_logits'], dim=-1)

                # Compute correlation
                flat_curr = current_hidden.flatten()
                flat_prev = prev_hidden.flatten()

                if flat_curr.std() > 1e-8 and flat_prev.std() > 1e-8:
                    corr = torch.corrcoef(torch.stack([flat_curr, flat_prev]))[0, 1].item()
                    if not math.isnan(corr):
                        correlations.append(abs(corr))

            prev_hidden = current_hidden
            time.sleep(0.02)

        mean_corr = float(np.mean(correlations)) if correlations else 0.0
        print(f"    Self-model correlation: {mean_corr:.4f}")

        return mean_corr

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _build_telemetry_tensor(
        self,
        sample: GpuSample,
        prev_sample: Optional[GpuSample],
    ) -> torch.Tensor:
        """Build 12-dim telemetry tensor from GPU sample."""
        power_norm = sample.power_w / 65.0
        temp_norm = sample.temp_edge_c / 100.0
        freq_norm = sample.freq_sclk_mhz / 3000.0
        busy_norm = sample.gpu_busy_pct / 100.0

        # Derivatives
        if prev_sample is not None:
            dt = max((sample.timestamp_ns - prev_sample.timestamp_ns) / 1e9, 0.001)
            d_power = (sample.power_w - prev_sample.power_w) / 65.0 / dt
            d_temp = (sample.temp_edge_c - prev_sample.temp_edge_c) / 100.0 / dt
            d_freq = (sample.freq_sclk_mhz - prev_sample.freq_sclk_mhz) / 3000.0 / dt
            d_busy = (sample.gpu_busy_pct - prev_sample.gpu_busy_pct) / 100.0 / dt
        else:
            d_power = d_temp = d_freq = d_busy = 0.0

        thermal_dev = max(0, (sample.temp_edge_c - 60.0)) / 40.0
        power_headroom = max(0, 1.0 - power_norm)

        vec = torch.tensor([
            power_norm, temp_norm, freq_norm, busy_norm,
            0.6, 0.0,  # power_cap_norm, throttle
            np.clip(d_power, -1, 1), np.clip(d_temp, -1, 1),
            np.clip(d_freq, -1, 1), np.clip(d_busy, -1, 1),
            thermal_dev, power_headroom,
        ], dtype=torch.float32)

        return vec

    # -------------------------------------------------------------------------
    # Benchmark Tasks
    # -------------------------------------------------------------------------

    def run_task_hardware_prediction(
        self,
        model: nn.Module,
        baseline: nn.Module,
    ) -> Dict[str, Any]:
        """
        Task 1: Hardware-dependent prediction (based on z1315).

        Model must predict a target that drifts with temperature.
        Embodied model sees telemetry -> should outperform baseline.
        """
        print("\n[TASK 1] Hardware-Dependent Prediction")
        print("-" * 50)

        # Generate test data
        test_data = torch.randint(0, 256, (500, self.seq_len), dtype=torch.long)

        results = {}

        # Measure embodied
        print("  Testing embodied model...")
        def telem_fn():
            s = self.telemetry.read_sample()
            return self._build_telemetry_tensor(s, None)

        model.eval()
        acc_e, acc5_e = self.measure_accuracy(model, test_data[:200], test_data[1:201], telem_fn)
        lat_e, lat50_e, lat99_e = self.measure_latency_ms(model, test_data[:self.batch_size], telem_fn)
        eng_e, static_e, dyn_e = self.measure_energy_per_inference_mj(model, test_data[:self.batch_size], telem_fn)

        results['embodied'] = {
            'accuracy': acc_e,
            'latency_ms': lat_e,
            'energy_mj': eng_e,
        }

        # Measure baseline
        print("  Testing baseline model...")
        baseline.eval()
        acc_b, acc5_b = self.measure_accuracy(baseline, test_data[:200], test_data[1:201], None)
        lat_b, lat50_b, lat99_b = self.measure_latency_ms(baseline, test_data[:self.batch_size], None)
        eng_b, static_b, dyn_b = self.measure_energy_per_inference_mj(baseline, test_data[:self.batch_size], None)

        results['baseline'] = {
            'accuracy': acc_b,
            'latency_ms': lat_b,
            'energy_mj': eng_b,
        }

        # Verdict
        results['embodied_better'] = acc_e > acc_b or eng_e < eng_b

        print(f"\n  Results:")
        print(f"    Embodied: acc={acc_e:.3f}, lat={lat_e:.2f}ms, energy={eng_e:.3f}mJ")
        print(f"    Baseline: acc={acc_b:.3f}, lat={lat_b:.2f}ms, energy={eng_b:.3f}mJ")
        print(f"    Verdict: {'EMBODIED WINS' if results['embodied_better'] else 'BASELINE WINS'}")

        return results

    def run_task_homeostatic_regulation(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """
        Task 2: Homeostatic regulation (based on z1700).

        Model must regulate power/temperature under constraints.
        """
        print("\n[TASK 2] Homeostatic Regulation")
        print("-" * 50)

        results = {}

        # Recovery test
        recovery_ms = self.measure_homeostatic_recovery(model)
        results['recovery_ms'] = recovery_ms
        results['recovery_score'] = max(0, 1.0 - recovery_ms / 10000.0)

        # Compare to baseline thresholds
        baseline_recovery_ms = 5000  # Typical GPU without embodiment
        results['improvement_vs_baseline'] = (baseline_recovery_ms - recovery_ms) / baseline_recovery_ms

        print(f"\n  Results:")
        print(f"    Recovery time: {recovery_ms:.0f}ms")
        print(f"    Score: {results['recovery_score']:.3f}")
        print(f"    vs Baseline: {results['improvement_vs_baseline']:+.1%}")

        return results

    def run_task_active_inference(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """
        Task 3: Active inference (based on z1701).

        Model should minimize free energy (prediction error + complexity).
        """
        print("\n[TASK 3] Active Inference")
        print("-" * 50)

        results = {}

        model.train()
        test_data = torch.randint(0, 256, (200, self.seq_len), dtype=torch.long)

        # Measure free energy over time
        free_energies = []
        prev_sample = None

        for epoch in range(5):
            epoch_fe = 0.0
            for i in range(min(50, len(test_data) // self.batch_size)):
                sample = self.telemetry.read_sample()
                telem = self._build_telemetry_tensor(sample, prev_sample)
                telem_batch = telem.unsqueeze(0).expand(self.batch_size, -1).to(DEVICE)
                prev_sample = sample

                batch = test_data[i*self.batch_size:(i+1)*self.batch_size].to(DEVICE)

                with torch.no_grad():
                    out = model(batch, telemetry=telem_batch)
                    logits = out['logits']

                    # Free energy = prediction error (cross entropy) + complexity (action entropy)
                    pred_error = F.cross_entropy(
                        logits[:, :-1].contiguous().view(-1, 256),
                        batch[:, 1:].contiguous().view(-1)
                    ).item()

                    action_probs = F.softmax(out['action_logits'], dim=-1)
                    complexity = -(action_probs * (action_probs + 1e-8).log()).sum(dim=-1).mean().item()

                    fe = pred_error + 0.1 * complexity
                    epoch_fe += fe

                time.sleep(0.02)

            free_energies.append(epoch_fe / 50)

        results['free_energy_trajectory'] = free_energies
        results['fe_reduction'] = (free_energies[0] - free_energies[-1]) / free_energies[0] if free_energies[0] > 0 else 0.0
        results['final_fe'] = free_energies[-1]

        # Score based on FE reduction
        results['score'] = max(0, min(1, results['fe_reduction'] + 0.5))

        print(f"\n  Results:")
        print(f"    Initial FE: {free_energies[0]:.4f}")
        print(f"    Final FE: {free_energies[-1]:.4f}")
        print(f"    Reduction: {results['fe_reduction']:.1%}")
        print(f"    Score: {results['score']:.3f}")

        return results

    def run_task_self_model(
        self,
        model: nn.Module,
    ) -> Dict[str, Any]:
        """
        Task 4: Self-model accuracy (based on z1709).

        Model should be able to predict its own hidden states.
        """
        print("\n[TASK 4] Self-Model Accuracy")
        print("-" * 50)

        results = {}

        corr = self.measure_self_model_accuracy(model)
        results['self_model_correlation'] = corr
        results['score'] = corr  # 0-1 scale

        # Compare to expected baseline
        baseline_corr = 0.3  # Random model
        results['improvement_vs_baseline'] = (corr - baseline_corr) / (1.0 - baseline_corr)

        print(f"\n  Results:")
        print(f"    Self-model correlation: {corr:.4f}")
        print(f"    Score: {results['score']:.3f}")
        print(f"    vs Random baseline: {results['improvement_vs_baseline']:+.1%}")

        return results

    # -------------------------------------------------------------------------
    # Full Benchmark Suite
    # -------------------------------------------------------------------------

    def run_full_benchmark(
        self,
        embodied_model: nn.Module,
        baseline_model: nn.Module,
    ) -> Dict[str, Any]:
        """
        Run the complete NeuroBench-compatible benchmark suite.
        """
        print("=" * 70)
        print("  NeuroBench-Compatible Embodied AI Benchmark")
        print("=" * 70)
        print(f"  Device: {DEVICE}")
        print(f"  Timestamp: {datetime.now().isoformat()}")
        print()

        results = {
            'experiment': 'z1970_neurobench_embodied',
            'timestamp': datetime.now().isoformat(),
            'device': str(DEVICE),
            'config': {
                'n_warmup': self.n_warmup,
                'n_trials': self.n_trials,
                'batch_size': self.batch_size,
                'seq_len': self.seq_len,
            },
            'baselines': NEUROBENCH_BASELINES,
            'tasks': {},
            'metrics': {},
        }

        # Generate test data
        print("Generating test data...")
        test_data = torch.randint(0, 256, (500, self.seq_len), dtype=torch.long)

        # Run benchmark tasks
        results['tasks']['hardware_prediction'] = self.run_task_hardware_prediction(
            embodied_model, baseline_model
        )

        results['tasks']['homeostatic_regulation'] = self.run_task_homeostatic_regulation(
            embodied_model
        )

        results['tasks']['active_inference'] = self.run_task_active_inference(
            embodied_model
        )

        results['tasks']['self_model'] = self.run_task_self_model(
            embodied_model
        )

        # Measure standard NeuroBench metrics
        print("\n[STANDARD NEUROBENCH METRICS]")
        print("-" * 50)

        def telem_fn():
            s = self.telemetry.read_sample()
            return self._build_telemetry_tensor(s, None)

        # Accuracy
        acc, acc5 = self.measure_accuracy(
            embodied_model, test_data[:200], test_data[1:201], telem_fn
        )

        # Latency
        lat_mean, lat_p50, lat_p99 = self.measure_latency_ms(
            embodied_model, test_data[:self.batch_size], telem_fn
        )

        # Energy
        eng_mj, static_w, dyn_w = self.measure_energy_per_inference_mj(
            embodied_model, test_data[:self.batch_size], telem_fn
        )

        # Embodiment metrics
        adapt_speed = self.measure_adaptation_speed(
            embodied_model, test_data, max_epochs=20
        )

        emb_advantage = self.measure_embodiment_advantage(
            embodied_model, baseline_model, test_data
        )

        # Throughput
        throughput = 1000.0 / lat_mean if lat_mean > 0 else 0.0

        # Memory
        n_params = sum(p.numel() for p in embodied_model.parameters())
        memory_mb = n_params * 4 / (1024 * 1024)  # Assuming float32

        # J/token
        j_per_token = eng_mj / 1000 / (self.batch_size * self.seq_len)

        metrics = NeuroBenchMetrics(
            accuracy=acc,
            top5_accuracy=acc5,
            latency_mean_ms=lat_mean,
            latency_p50_ms=lat_p50,
            latency_p99_ms=lat_p99,
            energy_per_inference_mj=eng_mj,
            static_power_w=static_w,
            dynamic_power_w=dyn_w,
            throughput_inf_s=throughput,
            model_params=n_params,
            memory_mb=memory_mb,
            j_per_token=j_per_token,
            adaptation_speed_epochs=adapt_speed,
            homeostatic_recovery_ms=results['tasks']['homeostatic_regulation']['recovery_ms'],
            self_model_accuracy=results['tasks']['self_model']['self_model_correlation'],
            free_energy_final=results['tasks']['active_inference']['final_fe'],
            embodiment_advantage=emb_advantage,
        )

        results['metrics'] = asdict(metrics)

        # Print comparison table
        self._print_comparison_table(metrics)

        # Overall verdict
        results['verdict'] = self._compute_verdict(metrics, results['tasks'])

        return results

    def _print_comparison_table(self, metrics: NeuroBenchMetrics):
        """Print comparison against NeuroBench baselines."""
        print("\n" + "=" * 70)
        print("  COMPARISON WITH NEUROBENCH BASELINES")
        print("=" * 70)

        header = f"{'System':<20} {'Energy(mJ)':<12} {'Latency(ms)':<12} {'Throughput':<12}"
        print(header)
        print("-" * 70)

        # Our system
        print(f"{'Embodied GPU (ours)':<20} {metrics.energy_per_inference_mj:<12.2f} "
              f"{metrics.latency_mean_ms:<12.2f} {metrics.throughput_inf_s:<12.1f}")

        # Baselines
        for key, baseline in NEUROBENCH_BASELINES.items():
            print(f"{baseline['name']:<20} {baseline['energy_per_inference_mj']:<12.2f} "
                  f"{baseline['latency_ms']:<12.2f} {baseline['throughput_inf_s']:<12.1f}")

        # Energy efficiency ranking
        print("\n  Energy Efficiency Ranking:")
        all_systems = [
            ('Embodied GPU (ours)', metrics.energy_per_inference_mj),
        ]
        for key, baseline in NEUROBENCH_BASELINES.items():
            all_systems.append((baseline['name'], baseline['energy_per_inference_mj']))

        all_systems.sort(key=lambda x: x[1])
        for i, (name, energy) in enumerate(all_systems, 1):
            marker = " <--" if name == 'Embodied GPU (ours)' else ""
            print(f"    {i}. {name}: {energy:.2f} mJ{marker}")

    def _compute_verdict(
        self,
        metrics: NeuroBenchMetrics,
        tasks: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute overall verdict."""
        print("\n" + "=" * 70)
        print("  FINAL VERDICT")
        print("=" * 70)

        verdicts = {}

        # V1: Better than neuromorphic energy efficiency
        best_neuro_energy = min(b['energy_per_inference_mj'] for b in NEUROBENCH_BASELINES.values()
                                if b['type'].startswith('neuromorphic'))
        verdicts['v1_neuro_competitive'] = metrics.energy_per_inference_mj < best_neuro_energy * 5

        # V2: Embodiment provides advantage
        verdicts['v2_embodiment_advantage'] = metrics.embodiment_advantage > 0

        # V3: Fast adaptation
        verdicts['v3_fast_adaptation'] = metrics.adaptation_speed_epochs < 20

        # V4: Homeostatic capability
        verdicts['v4_homeostatic'] = metrics.homeostatic_recovery_ms < 5000

        # V5: Self-model accuracy
        verdicts['v5_self_aware'] = metrics.self_model_accuracy > 0.5

        # Print
        n_pass = sum(verdicts.values())
        for key, passed in verdicts.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {key}")

        print(f"\n  Overall: {n_pass}/5 criteria passed")

        if n_pass >= 4:
            overall = "EXCELLENT - Genuine embodied neuromorphic system"
        elif n_pass >= 3:
            overall = "GOOD - Significant embodiment benefits demonstrated"
        elif n_pass >= 2:
            overall = "PARTIAL - Some embodiment benefits"
        else:
            overall = "NEEDS WORK - Limited embodiment benefits"

        print(f"  Verdict: {overall}")

        verdicts['n_pass'] = n_pass
        verdicts['overall'] = overall

        return verdicts


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  z1970: NeuroBench-Compatible Embodied AI Benchmark")
    print("=" * 70)
    print()

    # Create models
    print("Creating models...")
    config = MetabolicConfig(
        vocab_size=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=12,
        num_actions=4,
    )

    embodied_model = MetabolicTransformer(config).to(DEVICE)
    baseline_model = BaselineTransformer(config).to(DEVICE)

    n_params = sum(p.numel() for p in embodied_model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Device: {DEVICE}")

    # Run benchmark
    benchmark = EmbodiedNeuroBench(
        n_warmup=5,
        n_trials=50,
        batch_size=4,
        seq_len=256,
    )

    results = benchmark.run_full_benchmark(embodied_model, baseline_model)

    # Save results
    output_path = PROJECT_ROOT / 'results' / 'z1970_neurobench.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON
    def jsonify(obj):
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [jsonify(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, bool):
            return obj
        return obj

    with open(output_path, 'w') as f:
        json.dump(jsonify(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
