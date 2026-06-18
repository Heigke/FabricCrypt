#!/usr/bin/env python3
"""
Z912e: DVFS Optimal Operating Point Discovery via Embodied Control

This script tests whether embodied control can find better operating points on
the power-frequency Pareto curve by dynamically adjusting GPU clocks (DPM levels).

Key Innovation:
- Static frequency scaling: Pick one DPM level, measure J/token
- Embodied control: Dynamically switch DPM based on temp/throughput needs
- Hypothesis: Embodied can exploit thermal headroom better than fixed levels

Architecture:
1. DVFS Actuator: Read/write /sys/class/drm/card*/device/pp_dpm_sclk
2. Embodied Controller: Observe power, temp, throughput -> Choose DPM level
3. Benchmark: Run LM inference at each fixed level + embodied mode
4. Metrics: J/token, tokens/sec, thermal margin, Pareto efficiency

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import statistics

# AMD GPU setup for gfx1151
def detect_gpu_vendor() -> str:
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        vendor_file = card / "device/vendor"
        if vendor_file.exists():
            try:
                if vendor_file.read_text().strip() == "0x1002":
                    return "amd"
            except:
                pass
    return "cpu"

GPU_VENDOR = detect_gpu_vendor()
if GPU_VENDOR == "amd":
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


@dataclass
class DPMLevel:
    """Single DPM (Dynamic Power Management) level."""
    index: int
    freq_mhz: int
    is_active: bool = False


@dataclass
class OperatingPoint:
    """Metrics at a specific DPM level."""
    dpm_level: int
    freq_mhz: int
    energy_j: float
    tokens: int
    duration_s: float
    j_per_token: float
    tokens_per_sec: float
    avg_power_w: float
    avg_temp_c: float
    temp_max_c: float
    efficiency_score: float  # tokens_per_joule


@dataclass
class EmbodiedMetrics:
    """Metrics from embodied controller run."""
    total_energy_j: float
    total_tokens: int
    duration_s: float
    j_per_token: float
    tokens_per_sec: float
    avg_power_w: float
    avg_temp_c: float
    temp_max_c: float
    dpm_switches: int
    dpm_level_history: List[Tuple[float, int]]  # (timestamp, dpm_level)
    efficiency_score: float


class DVFSActuator:
    """
    DVFS actuator for AMD GPU DPM levels.

    Supports both:
    1. Read-only mode: Can read DPM levels but not write (simulation)
    2. Write mode: Can actually change DPM levels (requires permissions)
    """

    def __init__(self, card_index: int = 1, read_only: bool = False):
        self.card_index = card_index
        self.read_only = read_only

        # Find DPM sysfs paths
        self.sclk_path = Path(f"/sys/class/drm/card{card_index}/device/pp_dpm_sclk")
        self.perf_path = Path(f"/sys/class/drm/card{card_index}/device/power_dpm_force_performance_level")

        if not self.sclk_path.exists():
            raise RuntimeError(f"DPM sysfs not found at {self.sclk_path}")

        # Read available DPM levels
        self.dpm_levels = self._read_dpm_levels()

        # Check write permissions
        if not read_only:
            self._check_permissions()

        print(f"DVFS Actuator initialized (card{card_index})")
        print(f"  Mode: {'READ-ONLY (simulated)' if read_only else 'READ-WRITE (actual)'}")
        print(f"  DPM Levels: {len(self.dpm_levels)}")
        for lvl in self.dpm_levels:
            marker = "*" if lvl.is_active else " "
            print(f"    {marker} {lvl.index}: {lvl.freq_mhz} MHz")

    def _read_dpm_levels(self) -> List[DPMLevel]:
        """Parse DPM levels from sysfs."""
        levels = []
        try:
            content = self.sclk_path.read_text()
            for line in content.strip().split('\n'):
                # Format: "0: 600Mhz *" or "1: 1100Mhz"
                parts = line.strip().split(':')
                if len(parts) != 2:
                    continue

                idx = int(parts[0].strip())
                freq_part = parts[1].strip()
                is_active = '*' in freq_part

                # Extract frequency
                freq_str = freq_part.replace('*', '').replace('Mhz', '').replace('MHz', '').strip()
                freq_mhz = int(freq_str)

                levels.append(DPMLevel(
                    index=idx,
                    freq_mhz=freq_mhz,
                    is_active=is_active
                ))
        except Exception as e:
            raise RuntimeError(f"Failed to read DPM levels: {e}")

        if not levels:
            raise RuntimeError("No DPM levels found")

        return levels

    def _check_permissions(self):
        """Check if we can write to DPM sysfs."""
        try:
            # Try to read perf level
            if self.perf_path.exists():
                current = self.perf_path.read_text().strip()
                # Try to write same value back (no-op test)
                self.perf_path.write_text(current)
            else:
                print(f"Warning: {self.perf_path} not found, may not be able to set manual mode")
        except PermissionError:
            print("ERROR: No write permissions to DPM sysfs")
            print("Run with sudo or:")
            print(f"  sudo chmod 666 {self.sclk_path}")
            print(f"  sudo chmod 666 {self.perf_path}")
            print("\nFalling back to READ-ONLY mode (simulation)")
            self.read_only = True

    def get_current_level(self) -> int:
        """Get currently active DPM level."""
        levels = self._read_dpm_levels()
        for lvl in levels:
            if lvl.is_active:
                return lvl.index
        return levels[0].index  # Fallback to first

    def set_dpm_level(self, level: int) -> bool:
        """
        Set DPM level (0 = lowest, N = highest).
        Returns True if successful, False if read-only or failed.
        """
        if self.read_only:
            # Simulate success in read-only mode
            return True

        level = max(0, min(len(self.dpm_levels) - 1, level))

        try:
            # Set manual mode first
            self.perf_path.write_text("manual")
            time.sleep(0.05)

            # Set specific level
            self.sclk_path.write_text(str(level))
            time.sleep(0.1)  # Let it take effect

            return True
        except Exception as e:
            print(f"Failed to set DPM level {level}: {e}")
            return False

    def reset_to_auto(self) -> bool:
        """Reset to auto DPM mode."""
        if self.read_only:
            return True

        try:
            self.perf_path.write_text("auto")
            return True
        except Exception as e:
            print(f"Failed to reset to auto: {e}")
            return False


class EmbodiedDPMController:
    """
    Embodied controller that chooses DPM level based on thermal/throughput state.

    Strategy:
    - If temp > thermal_target: Lower DPM to reduce power
    - If temp < thermal_target - 5°C and throughput needed: Raise DPM
    - Keep history of decisions for learning
    """

    def __init__(
        self,
        actuator: DVFSActuator,
        telemetry: SysfsHwmonTelemetry,
        thermal_target_c: float = 60.0,
        power_budget_w: float = 100.0,
    ):
        self.actuator = actuator
        self.telemetry = telemetry
        self.thermal_target = thermal_target_c
        self.power_budget = power_budget_w

        # State
        self.current_dpm = self.actuator.get_current_level()
        self.last_actuation = 0.0
        self.actuation_interval_s = 1.0  # Min time between DPM changes

        # Metrics
        self.dpm_switches = 0
        self.dpm_history: List[Tuple[float, int]] = []

    def observe_and_act(self) -> int:
        """
        Observe hardware state and choose DPM level.
        Returns chosen DPM level.
        """
        now = time.time()

        # Rate limit actuations
        if now - self.last_actuation < self.actuation_interval_s:
            return self.current_dpm

        # Read telemetry
        sample = self.telemetry.read_sample()
        temp_c = sample.temp_edge_c
        power_w = sample.power_w

        # Decision logic
        new_dpm = self.current_dpm

        # Thermal-driven downscaling
        if temp_c > self.thermal_target:
            # Too hot, reduce frequency
            new_dpm = max(0, self.current_dpm - 1)

        # Power-driven downscaling
        elif power_w > self.power_budget:
            # Exceeding power budget, reduce frequency
            new_dpm = max(0, self.current_dpm - 1)

        # Thermal headroom available - upscale
        elif temp_c < self.thermal_target - 5.0:
            # Cool enough, can increase frequency for more throughput
            max_dpm = len(self.actuator.dpm_levels) - 1
            new_dpm = min(max_dpm, self.current_dpm + 1)

        # Apply action
        if new_dpm != self.current_dpm:
            success = self.actuator.set_dpm_level(new_dpm)
            if success:
                self.current_dpm = new_dpm
                self.dpm_switches += 1
                self.last_actuation = now
                self.dpm_history.append((now, new_dpm))

        return self.current_dpm


class DummyModel(nn.Module):
    """Lightweight LM for benchmarking."""
    def __init__(self, vocab_size: int = 8192, d_model: int = 768, n_layers: int = 12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=12,
                dim_feedforward=d_model * 4,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def run_inference_workload(
    model: nn.Module,
    telemetry: SysfsHwmonTelemetry,
    num_tokens: int = 1000,
    batch_size: int = 4,
    seq_len: int = 128,
) -> Tuple[float, float, List[float]]:
    """
    Run inference workload and measure energy.
    Returns (energy_j, duration_s, temp_samples).
    """
    device = next(model.parameters()).device
    vocab_size = model.embed.num_embeddings

    # Generate random inputs
    num_batches = (num_tokens + batch_size * seq_len - 1) // (batch_size * seq_len)

    temps = []

    telemetry.reset_accumulator()
    telemetry.start_continuous_sampling()
    start = time.time()

    with torch.no_grad():
        for _ in range(num_batches):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            _ = model(input_ids)

            # Sample temp
            sample = telemetry.read_sample()
            temps.append(sample.temp_edge_c)

    torch.cuda.synchronize()
    end = time.time()
    telemetry.stop_continuous_sampling()

    duration_s = end - start
    energy_j = telemetry.get_accumulated_energy_j()

    return energy_j, duration_s, temps


def benchmark_fixed_dpm_level(
    actuator: DVFSActuator,
    model: nn.Module,
    telemetry: SysfsHwmonTelemetry,
    dpm_level: int,
    num_tokens: int = 1000,
) -> OperatingPoint:
    """Benchmark at a fixed DPM level."""
    print(f"\n  DPM Level {dpm_level} ({actuator.dpm_levels[dpm_level].freq_mhz} MHz)")

    # Set DPM level
    actuator.set_dpm_level(dpm_level)
    time.sleep(0.5)  # Settle

    # Run inference
    energy_j, duration_s, temps = run_inference_workload(
        model, telemetry, num_tokens=num_tokens
    )

    # Compute metrics
    tokens_per_sec = num_tokens / duration_s if duration_s > 0 else 0
    j_per_token = energy_j / num_tokens if num_tokens > 0 else 0
    avg_power_w = energy_j / duration_s if duration_s > 0 else 0
    efficiency = num_tokens / energy_j if energy_j > 0 else 0

    result = OperatingPoint(
        dpm_level=dpm_level,
        freq_mhz=actuator.dpm_levels[dpm_level].freq_mhz,
        energy_j=energy_j,
        tokens=num_tokens,
        duration_s=duration_s,
        j_per_token=j_per_token,
        tokens_per_sec=tokens_per_sec,
        avg_power_w=avg_power_w,
        avg_temp_c=statistics.mean(temps) if temps else 0,
        temp_max_c=max(temps) if temps else 0,
        efficiency_score=efficiency,
    )

    print(f"    Energy: {energy_j:.3f} J")
    print(f"    J/token: {j_per_token:.4f}")
    print(f"    Throughput: {tokens_per_sec:.1f} tok/s")
    print(f"    Power: {avg_power_w:.1f} W")
    print(f"    Temp: {result.avg_temp_c:.1f}°C (max: {result.temp_max_c:.1f}°C)")
    print(f"    Efficiency: {efficiency:.1f} tok/J")

    return result


def benchmark_embodied_control(
    actuator: DVFSActuator,
    model: nn.Module,
    telemetry: SysfsHwmonTelemetry,
    thermal_target_c: float = 60.0,
    num_tokens: int = 1000,
) -> EmbodiedMetrics:
    """Benchmark with embodied DPM controller."""
    print(f"\n  Embodied Controller (thermal_target={thermal_target_c}°C)")

    # Create controller
    controller = EmbodiedDPMController(
        actuator=actuator,
        telemetry=telemetry,
        thermal_target_c=thermal_target_c,
    )

    # Reset to mid-level
    mid_level = len(actuator.dpm_levels) // 2
    actuator.set_dpm_level(mid_level)
    time.sleep(0.5)

    device = next(model.parameters()).device
    vocab_size = model.embed.num_embeddings
    batch_size = 4
    seq_len = 128
    num_batches = (num_tokens + batch_size * seq_len - 1) // (batch_size * seq_len)

    temps = []

    telemetry.reset_accumulator()
    telemetry.start_continuous_sampling()
    start = time.time()

    with torch.no_grad():
        for i in range(num_batches):
            # Embodied control: observe and act every few batches
            if i % 5 == 0:
                controller.observe_and_act()

            # Inference
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            _ = model(input_ids)

            sample = telemetry.read_sample()
            temps.append(sample.temp_edge_c)

    torch.cuda.synchronize()
    end = time.time()
    telemetry.stop_continuous_sampling()

    duration_s = end - start
    energy_j = telemetry.get_accumulated_energy_j()
    tokens_per_sec = num_tokens / duration_s if duration_s > 0 else 0
    j_per_token = energy_j / num_tokens if num_tokens > 0 else 0
    avg_power_w = energy_j / duration_s if duration_s > 0 else 0
    efficiency = num_tokens / energy_j if energy_j > 0 else 0

    result = EmbodiedMetrics(
        total_energy_j=energy_j,
        total_tokens=num_tokens,
        duration_s=duration_s,
        j_per_token=j_per_token,
        tokens_per_sec=tokens_per_sec,
        avg_power_w=avg_power_w,
        avg_temp_c=statistics.mean(temps) if temps else 0,
        temp_max_c=max(temps) if temps else 0,
        dpm_switches=controller.dpm_switches,
        dpm_level_history=controller.dpm_history,
        efficiency_score=efficiency,
    )

    print(f"    Energy: {energy_j:.3f} J")
    print(f"    J/token: {j_per_token:.4f}")
    print(f"    Throughput: {tokens_per_sec:.1f} tok/s")
    print(f"    Power: {avg_power_w:.1f} W")
    print(f"    Temp: {result.avg_temp_c:.1f}°C (max: {result.temp_max_c:.1f}°C)")
    print(f"    Efficiency: {efficiency:.1f} tok/J")
    print(f"    DPM Switches: {controller.dpm_switches}")

    return result


def main():
    print("=" * 80)
    print("Z912e: DVFS Optimal Operating Point Discovery")
    print("=" * 80)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available")
        sys.exit(1)

    device = torch.device("cuda:0")
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    # Initialize components
    print("\n" + "=" * 80)
    print("Initializing Components")
    print("=" * 80)

    try:
        # Try card1 first (discrete GPU), fall back to card0
        try:
            actuator = DVFSActuator(card_index=1)
        except:
            print("Card1 not found, trying card0...")
            actuator = DVFSActuator(card_index=0)
    except Exception as e:
        print(f"ERROR: Failed to initialize DVFS actuator: {e}")
        sys.exit(1)

    telemetry = SysfsHwmonTelemetry(sample_rate_hz=50)
    print(f"\nTelemetry initialized")

    # Create model
    print(f"\nCreating model...")
    model = DummyModel(vocab_size=8192, d_model=768, n_layers=6).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Warmup
    print(f"\nWarmup...")
    with torch.no_grad():
        x = torch.randint(0, 8192, (4, 128), device=device)
        _ = model(x)
    torch.cuda.synchronize()
    time.sleep(1.0)

    # Benchmark configuration
    num_tokens = 2000

    # ========================================================================
    # Phase 1: Benchmark each fixed DPM level
    # ========================================================================
    print("\n" + "=" * 80)
    print("Phase 1: Fixed DPM Level Benchmarks")
    print("=" * 80)

    fixed_results: List[OperatingPoint] = []

    for dpm_level in range(len(actuator.dpm_levels)):
        result = benchmark_fixed_dpm_level(
            actuator=actuator,
            model=model,
            telemetry=telemetry,
            dpm_level=dpm_level,
            num_tokens=num_tokens,
        )
        fixed_results.append(result)
        time.sleep(1.0)  # Cool down between runs

    # ========================================================================
    # Phase 2: Embodied Control Benchmark
    # ========================================================================
    print("\n" + "=" * 80)
    print("Phase 2: Embodied Control Benchmark")
    print("=" * 80)

    embodied_result = benchmark_embodied_control(
        actuator=actuator,
        model=model,
        telemetry=telemetry,
        thermal_target_c=60.0,
        num_tokens=num_tokens,
    )

    # Reset to auto
    actuator.reset_to_auto()

    # ========================================================================
    # Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("Analysis: Pareto Frontier")
    print("=" * 80)

    # Find best fixed levels
    best_efficiency = max(fixed_results, key=lambda x: x.efficiency_score)
    best_throughput = max(fixed_results, key=lambda x: x.tokens_per_sec)
    best_energy = min(fixed_results, key=lambda x: x.j_per_token)

    print(f"\nBest Fixed Operating Points:")
    print(f"  Best Efficiency: DPM {best_efficiency.dpm_level} - "
          f"{best_efficiency.efficiency_score:.1f} tok/J")
    print(f"  Best Throughput: DPM {best_throughput.dpm_level} - "
          f"{best_throughput.tokens_per_sec:.1f} tok/s")
    print(f"  Best Energy: DPM {best_energy.dpm_level} - "
          f"{best_energy.j_per_token:.4f} J/tok")

    # Compare embodied vs fixed
    print(f"\nEmbodied vs Best Fixed:")

    # vs best efficiency
    efficiency_improvement = (
        (embodied_result.efficiency_score - best_efficiency.efficiency_score)
        / best_efficiency.efficiency_score * 100
    )
    print(f"  Efficiency: {efficiency_improvement:+.1f}% vs best fixed")

    # vs best energy
    energy_improvement = (
        (best_energy.j_per_token - embodied_result.j_per_token)
        / best_energy.j_per_token * 100
    )
    print(f"  Energy: {energy_improvement:+.1f}% vs best fixed")

    # vs best throughput
    throughput_improvement = (
        (embodied_result.tokens_per_sec - best_throughput.tokens_per_sec)
        / best_throughput.tokens_per_sec * 100
    )
    print(f"  Throughput: {throughput_improvement:+.1f}% vs best fixed")

    # ========================================================================
    # Business Projection
    # ========================================================================
    print("\n" + "=" * 80)
    print("Business Impact Projection")
    print("=" * 80)

    # Assume 1M tokens/day per GPU
    daily_tokens = 1_000_000
    num_gpus = 100
    electricity_cost_per_kwh = 0.15

    # Energy cost comparison
    baseline_energy_kwh = (best_energy.j_per_token * daily_tokens * num_gpus) / 3600 / 1000
    embodied_energy_kwh = (embodied_result.j_per_token * daily_tokens * num_gpus) / 3600 / 1000

    daily_cost_baseline = baseline_energy_kwh * electricity_cost_per_kwh
    daily_cost_embodied = embodied_energy_kwh * electricity_cost_per_kwh
    daily_savings = daily_cost_baseline - daily_cost_embodied
    annual_savings = daily_savings * 365

    print(f"\nAssumptions:")
    print(f"  {daily_tokens:,} tokens/day/GPU")
    print(f"  {num_gpus} GPUs")
    print(f"  ${electricity_cost_per_kwh:.3f}/kWh")

    print(f"\nEnergy Costs:")
    print(f"  Best Fixed: ${daily_cost_baseline:.2f}/day")
    print(f"  Embodied:   ${daily_cost_embodied:.2f}/day")
    print(f"  Daily Savings: ${daily_savings:.2f}")
    print(f"  Annual Savings: ${annual_savings:,.2f}")

    # ========================================================================
    # Save Results
    # ========================================================================
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "z912e_dvfs_results.json"

    results = {
        "experiment": "z912e_dvfs_operating_point",
        "timestamp": time.time(),
        "device": torch.cuda.get_device_name(0),
        "num_dpm_levels": len(actuator.dpm_levels),
        "dpm_levels": [
            {"index": lvl.index, "freq_mhz": lvl.freq_mhz}
            for lvl in actuator.dpm_levels
        ],
        "fixed_results": [asdict(r) for r in fixed_results],
        "embodied_result": {
            "total_energy_j": embodied_result.total_energy_j,
            "total_tokens": embodied_result.total_tokens,
            "duration_s": embodied_result.duration_s,
            "j_per_token": embodied_result.j_per_token,
            "tokens_per_sec": embodied_result.tokens_per_sec,
            "avg_power_w": embodied_result.avg_power_w,
            "avg_temp_c": embodied_result.avg_temp_c,
            "temp_max_c": embodied_result.temp_max_c,
            "dpm_switches": embodied_result.dpm_switches,
            "efficiency_score": embodied_result.efficiency_score,
        },
        "pareto_curve": [
            {
                "dpm_level": r.dpm_level,
                "freq_mhz": r.freq_mhz,
                "j_per_token": r.j_per_token,
                "tokens_per_sec": r.tokens_per_sec,
                "efficiency": r.efficiency_score,
            }
            for r in fixed_results
        ],
        "best_fixed": {
            "efficiency": {
                "dpm_level": best_efficiency.dpm_level,
                "score": best_efficiency.efficiency_score,
            },
            "throughput": {
                "dpm_level": best_throughput.dpm_level,
                "tokens_per_sec": best_throughput.tokens_per_sec,
            },
            "energy": {
                "dpm_level": best_energy.dpm_level,
                "j_per_token": best_energy.j_per_token,
            },
        },
        "improvements": {
            "efficiency_pct": efficiency_improvement,
            "energy_pct": energy_improvement,
            "throughput_pct": throughput_improvement,
        },
        "business_projection": {
            "assumptions": {
                "daily_tokens_per_gpu": daily_tokens,
                "num_gpus": num_gpus,
                "electricity_cost_per_kwh": electricity_cost_per_kwh,
            },
            "annual_savings_usd": annual_savings,
            "daily_savings_usd": daily_savings,
            "energy_reduction_kwh_per_day": baseline_energy_kwh - embodied_energy_kwh,
        },
        "actuator_mode": "READ-ONLY" if actuator.read_only else "READ-WRITE",
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFixed DPM Levels Tested: {len(fixed_results)}")
    print(f"Embodied DPM Switches: {embodied_result.dpm_switches}")
    print(f"\nKey Finding:")
    if energy_improvement > 5:
        print(f"  ✓ Embodied control achieves {energy_improvement:.1f}% energy savings")
        print(f"    by exploiting thermal headroom on the Pareto curve")
    elif energy_improvement > 0:
        print(f"  ≈ Embodied control slightly improves energy ({energy_improvement:.1f}%)")
    else:
        print(f"  ✗ Embodied control does not improve over best fixed DPM")
        print(f"    Fixed DPM {best_energy.dpm_level} is optimal for this workload")

    print(f"\nBusiness Impact: ${annual_savings:,.2f}/year potential savings")


if __name__ == "__main__":
    main()
