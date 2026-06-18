#!/usr/bin/env python3
"""
Minimal Closed-Loop Validation for FEEL-SLM v2

Proves the full loop end-to-end:
1. Read telemetry (sysfs/NVML at 100Hz)
2. Compute body embedding
3. Policy chooses profile {eco, balanced, perf}
4. Actuation (mock or real daemon)
5. Measure Δenergy / Δtokens

This validates "embodied" before any model training.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feel_slm.telemetry_source import (
    create_telemetry_source,
    TelemetrySampler,
    TelemetryWindow,
)
from feel_slm.model_v2 import FEELConfigV2, FEELSLMV2, BaselineSLMV2


# =============================================================================
# Actuation Interface
# =============================================================================

class MockActuator:
    """Mock actuator for testing (no real hardware changes)."""

    def __init__(self):
        self.current_profile = "balanced"
        self.history: List[Dict] = []

    def set_profile(self, profile: str):
        if profile != self.current_profile:
            self.history.append({
                "timestamp": time.time(),
                "from": self.current_profile,
                "to": profile,
            })
            self.current_profile = profile
            print(f"  [MOCK] Profile change: {self.history[-1]['from']} → {profile}")

    def get_history(self) -> List[Dict]:
        return self.history


class SysfsActuator:
    """
    Real sysfs-based actuator for AMD GPUs.

    Sets power profile via:
    /sys/class/drm/cardX/device/power_dpm_force_performance_level
    /sys/class/drm/cardX/device/pp_power_profile_mode
    """

    PROFILES = {
        "eco": {
            "dpm_level": "low",
            "power_profile": "1",  # POWER_SAVING
        },
        "balanced": {
            "dpm_level": "auto",
            "power_profile": "0",  # BOOTUP_DEFAULT
        },
        "perf": {
            "dpm_level": "high",
            "power_profile": "4",  # COMPUTE
        },
    }

    def __init__(self, card_index: int = 0):
        self.base_path = f"/sys/class/drm/card{card_index}/device"
        self.current_profile = "balanced"
        self.history: List[Dict] = []

        # Check if we have write access
        dpm_path = os.path.join(self.base_path, "power_dpm_force_performance_level")
        self.can_write = os.access(dpm_path, os.W_OK)

        if not self.can_write:
            print(f"WARNING: No write access to {dpm_path}")
            print("Running in read-only mode. Use sudo for actual actuation.")

    def set_profile(self, profile: str):
        if profile not in self.PROFILES:
            print(f"Unknown profile: {profile}")
            return

        if profile == self.current_profile:
            return

        settings = self.PROFILES[profile]
        old_profile = self.current_profile

        if self.can_write:
            try:
                # Set DPM level
                dpm_path = os.path.join(self.base_path, "power_dpm_force_performance_level")
                with open(dpm_path, "w") as f:
                    f.write(settings["dpm_level"])

                # Set power profile mode
                pp_path = os.path.join(self.base_path, "pp_power_profile_mode")
                if os.path.exists(pp_path):
                    with open(pp_path, "w") as f:
                        f.write(settings["power_profile"])

                self.current_profile = profile
                self.history.append({
                    "timestamp": time.time(),
                    "from": old_profile,
                    "to": profile,
                    "applied": True,
                })
                print(f"  [SYSFS] Profile: {old_profile} → {profile}")

            except Exception as e:
                print(f"  [SYSFS] Failed to set profile: {e}")
                self.history.append({
                    "timestamp": time.time(),
                    "from": old_profile,
                    "to": profile,
                    "applied": False,
                    "error": str(e),
                })
        else:
            # Read-only mode
            self.current_profile = profile
            self.history.append({
                "timestamp": time.time(),
                "from": old_profile,
                "to": profile,
                "applied": False,
                "reason": "read_only",
            })
            print(f"  [SYSFS-RO] Would set: {old_profile} → {profile}")

    def get_history(self) -> List[Dict]:
        return self.history


# =============================================================================
# Closed-Loop Runner
# =============================================================================

@dataclass
class ClosedLoopResult:
    """Result from a single closed-loop run."""
    condition: str  # baseline, feel_fixed, feel_adaptive
    mode: str  # eco, balanced, perf (for fixed), or "adaptive"
    tokens_generated: int
    duration_s: float
    energy_mj: float
    mj_per_token: float
    tokens_per_second: float
    avg_power_w: float
    avg_temp_c: float
    profile_changes: int
    profiles_used: Dict[str, int]


def run_closed_loop(
    model: torch.nn.Module,
    sampler: TelemetrySampler,
    actuator,
    condition: str,
    mode: str,  # "eco", "balanced", "perf", or "adaptive"
    num_tokens: int = 100,
    control_window_tokens: int = 32,
    device: torch.device = None,
) -> ClosedLoopResult:
    """
    Run closed-loop token generation.

    Args:
        model: FEEL or Baseline model
        sampler: Telemetry sampler (already running)
        actuator: Profile actuator
        condition: Name for this condition
        mode: Operating mode
        num_tokens: Tokens to generate
        control_window_tokens: Update control every N tokens
        device: Torch device

    Returns:
        ClosedLoopResult with measurements
    """
    if device is None:
        device = next(model.parameters()).device

    # Initialize
    config = model.config if hasattr(model, 'config') else None
    is_feel = hasattr(model, 'body_encoder')

    # Set initial mode
    if is_feel and mode != "adaptive":
        model.set_mode(mode)
    actuator.set_profile(mode if mode != "adaptive" else "balanced")

    # Input prompt
    input_ids = torch.randint(0, 32000, (1, 32), device=device)
    generated_ids = input_ids.clone()

    # Clear sampler buffer and start fresh
    sampler.buffer.clear()

    # Tracking
    start_time = time.time()
    tokens_since_control = 0
    profile_counts = {"eco": 0, "balanced": 0, "perf": 0}
    profile_changes = 0

    print(f"\n  Generating {num_tokens} tokens ({condition}, mode={mode})...")

    for i in range(num_tokens):
        # Update control periodically
        if is_feel and mode == "adaptive" and tokens_since_control >= control_window_tokens:
            tokens_since_control = 0

            # Get telemetry window
            window = sampler.get_window(0.2)  # 200ms window
            if window and window.n_samples > 2:
                # Create telemetry tensor
                telemetry = torch.tensor([
                    window.power_w_mean / 300.0,
                    window.temp_c_mean / 100.0,
                    window.gpu_util_mean / 100.0,
                    0.5, 0.5, 0.5,  # clock, mem_util, mem_used
                    window.power_delta / 300.0,
                    window.temp_delta / 100.0,
                    window.util_delta / 100.0,
                    0.0, 0.5, 0.5,  # throttle, fan, profile
                ], device=device, dtype=torch.float32).unsqueeze(0)

                # Get policy decision
                with torch.no_grad():
                    body_embed = model.body_encoder(telemetry)
                    policy = model.policy_head(body_embed)
                    profile_idx = policy["profile_idx"].item()
                    modes = ["eco", "balanced", "perf"]
                    new_mode = modes[profile_idx]

                    if new_mode != actuator.current_profile:
                        profile_changes += 1
                        actuator.set_profile(new_mode)
                        model.set_mode(new_mode)

        # Forward pass
        with torch.no_grad():
            if is_feel:
                # Get current telemetry for FEEL
                latest = sampler.get_latest()
                if latest:
                    telemetry = torch.tensor(
                        latest.to_vector(), device=device, dtype=torch.float32
                    ).unsqueeze(0)
                else:
                    telemetry = torch.zeros(1, 12, device=device)

                outputs = model(generated_ids, telemetry)
            else:
                outputs = model(generated_ids)

            logits = outputs["logits"][:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        tokens_since_control += 1
        profile_counts[actuator.current_profile] += 1

    # Final measurements
    end_time = time.time()
    duration_s = end_time - start_time

    # Get aggregated telemetry for the whole run
    final_window = sampler.buffer.aggregate_window(duration_s)

    energy_mj = final_window.energy_mj if final_window else 0.0
    avg_power = final_window.power_w_mean if final_window else 0.0
    avg_temp = final_window.temp_c_mean if final_window else 0.0

    return ClosedLoopResult(
        condition=condition,
        mode=mode,
        tokens_generated=num_tokens,
        duration_s=duration_s,
        energy_mj=energy_mj,
        mj_per_token=energy_mj / num_tokens if num_tokens > 0 else 0,
        tokens_per_second=num_tokens / duration_s,
        avg_power_w=avg_power,
        avg_temp_c=avg_temp,
        profile_changes=profile_changes,
        profiles_used=profile_counts,
    )


# =============================================================================
# Main Validation
# =============================================================================

def run_validation(
    platform: str = "auto",
    use_real_actuation: bool = False,
    num_tokens: int = 100,
    output_dir: str = "results/z110_closed_loop",
):
    """Run full closed-loop validation."""

    print("=" * 70)
    print("FEEL-SLM v2 Minimal Closed-Loop Validation")
    print("=" * 70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create telemetry source and sampler
    print("\n--- Setting up telemetry ---")
    source, sampler = create_telemetry_source(platform)
    print(f"Platform: {type(source).__name__}")
    print(f"Capabilities: {source.get_capabilities()}")

    # Start sampler
    sampler.start()
    time.sleep(0.5)  # Let it collect some samples

    # Check telemetry is working
    latest = sampler.get_latest()
    if latest:
        print(f"Telemetry OK: {latest.power_w:.1f}W, {latest.temp_c:.1f}°C")
    else:
        print("WARNING: No telemetry data")

    # Create actuator
    print("\n--- Setting up actuator ---")
    if use_real_actuation:
        actuator = SysfsActuator()
    else:
        actuator = MockActuator()
    print(f"Actuator: {type(actuator).__name__}")

    # Create models
    print("\n--- Creating models ---")
    config = FEELConfigV2(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_dim=512,
        phase=1,
        enable_film=False,
        enable_gating=False,
        enable_layerdrop=True,
        layerdrop_layers=[1, 2],
    )

    baseline = BaselineSLMV2(config).to(device).eval()
    feel = FEELSLMV2(config).to(device).eval()

    baseline_params = sum(p.numel() for p in baseline.parameters())
    feel_params = sum(p.numel() for p in feel.parameters())
    print(f"Baseline params: {baseline_params:,}")
    print(f"FEEL params: {feel_params:,}")
    print(f"Overhead: {(feel_params - baseline_params) / baseline_params * 100:.1f}%")

    # Run conditions
    print("\n--- Running conditions ---")
    results: List[ClosedLoopResult] = []

    conditions = [
        ("baseline", baseline, "balanced"),
        ("baseline", baseline, "eco"),
        ("baseline", baseline, "perf"),
        ("feel_fixed", feel, "balanced"),
        ("feel_fixed", feel, "eco"),
        ("feel_fixed", feel, "perf"),
        ("feel_adaptive", feel, "adaptive"),
    ]

    for condition_name, model, mode in conditions:
        # Reset actuator to baseline
        actuator.set_profile("balanced")
        time.sleep(0.5)

        result = run_closed_loop(
            model=model,
            sampler=sampler,
            actuator=actuator,
            condition=condition_name,
            mode=mode,
            num_tokens=num_tokens,
            control_window_tokens=32,
            device=device,
        )
        results.append(result)

        print(f"  → {result.mj_per_token:.1f} mJ/tok, "
              f"{result.tokens_per_second:.1f} tok/s, "
              f"{result.avg_power_w:.1f}W")

    # Stop sampler
    sampler.stop()

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Condition':<15} {'Mode':<10} {'mJ/tok':>10} {'tok/s':>10} {'Power':>10} {'Changes':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r.condition:<15} {r.mode:<10} {r.mj_per_token:>10.1f} "
              f"{r.tokens_per_second:>10.1f} {r.avg_power_w:>9.1f}W {r.profile_changes:>10}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results_data = {
        "config": {
            "platform": platform,
            "use_real_actuation": use_real_actuation,
            "num_tokens": num_tokens,
            "device": str(device),
            "baseline_params": baseline_params,
            "feel_params": feel_params,
        },
        "results": [asdict(r) for r in results],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = os.path.join(output_dir, "closed_loop_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Summary
    print("\n--- Summary ---")

    # Compare baseline modes
    baseline_eco = [r for r in results if r.condition == "baseline" and r.mode == "eco"][0]
    baseline_perf = [r for r in results if r.condition == "baseline" and r.mode == "perf"][0]

    print(f"Baseline eco vs perf: {baseline_eco.mj_per_token:.1f} vs {baseline_perf.mj_per_token:.1f} mJ/tok")

    # Compare FEEL modes
    feel_eco = [r for r in results if r.condition == "feel_fixed" and r.mode == "eco"][0]
    feel_perf = [r for r in results if r.condition == "feel_fixed" and r.mode == "perf"][0]
    feel_adaptive = [r for r in results if r.condition == "feel_adaptive"][0]

    print(f"FEEL eco vs perf: {feel_eco.mj_per_token:.1f} vs {feel_perf.mj_per_token:.1f} mJ/tok")
    print(f"FEEL adaptive: {feel_adaptive.mj_per_token:.1f} mJ/tok, {feel_adaptive.profile_changes} changes")

    # LayerDrop effect
    baseline_balanced = [r for r in results if r.condition == "baseline" and r.mode == "balanced"][0]
    feel_balanced = [r for r in results if r.condition == "feel_fixed" and r.mode == "balanced"][0]

    layerdrop_speedup = feel_eco.tokens_per_second / feel_perf.tokens_per_second
    print(f"FEEL LayerDrop speedup (eco vs perf): {layerdrop_speedup:.2f}x")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="auto", choices=["auto", "amd", "nvidia"])
    parser.add_argument("--real-actuation", action="store_true", help="Use real sysfs actuation")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--output", default="results/z110_closed_loop")
    args = parser.parse_args()

    run_validation(
        platform=args.platform,
        use_real_actuation=args.real_actuation,
        num_tokens=args.tokens,
        output_dir=args.output,
    )
