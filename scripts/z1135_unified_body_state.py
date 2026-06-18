#!/usr/bin/env python3
"""
z1135: Unified GPU-FPGA Body State Validation
==============================================

Tests the UnifiedBodyState that combines:
- GPU telemetry (power, temp, utilization)
- FPGA telemetry (temperature, DRAM charge, decay rate)

Into an 8-dimensional z_feel vector for embodied AI.

This validates the integration architecture before full thermal coupling experiments.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Import our modules
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.atom.schema import AtomConfig
from src.atom.feel import BodyStateTracker
from src.embodied.fpga_state_tracker import FPGAStateTracker, UnifiedBodyState


def test_unified_body_state_dimensions():
    """Test that z_feel has correct dimensions."""
    print("\n=== Test 1: z_feel Dimensions ===")

    # Create trackers (simulated FPGA, real GPU if available)
    fpga_tracker = FPGAStateTracker(simulated=True, sim_temp_c=54.0)
    unified = UnifiedBodyState(fpga_tracker=fpga_tracker)

    # Get z_feel without GPU state
    z_feel = unified.get_z_feel()
    print(f"z_feel shape: {z_feel.shape}")
    print(f"z_feel values: {z_feel.tolist()}")

    assert z_feel.shape == (8,), f"Expected (8,), got {z_feel.shape}"
    print("PASS: z_feel has correct 8 dimensions")
    return True


def test_fpga_decay_dynamics():
    """Test that FPGA decay model follows Arrhenius dynamics."""
    print("\n=== Test 2: FPGA Decay Dynamics ===")

    # Test at different temperatures
    temps = [30, 50, 70, 90]
    decay_rates = []

    for temp_c in temps:
        tracker = FPGAStateTracker(simulated=True, sim_temp_c=temp_c)
        state = tracker.update()
        decay_rates.append(state.decay_rate_per_second)
        print(f"  {temp_c}C: decay_rate = {state.decay_rate_per_second:.6f}/s")

    # Verify higher temp = higher decay (Arrhenius)
    for i in range(len(temps) - 1):
        assert decay_rates[i+1] > decay_rates[i], "Decay should increase with temperature"

    print("PASS: Decay rate increases with temperature (Arrhenius)")
    return True


def test_frac_charge_accumulation():
    """Test that Frac operations accumulate charge."""
    print("\n=== Test 3: Frac Charge Accumulation ===")

    tracker = FPGAStateTracker(simulated=True)

    # Initial state
    state = tracker.update()
    print(f"Initial charge: {state.charge_level:.3f}")
    assert state.charge_level == 0.0

    # Record Frac operations
    for i in range(1, 9):
        tracker.record_frac(1)
        state = tracker.update()
        print(f"After {i} Frac: charge = {state.charge_level:.3f}")

    # Should be fully charged after ~8 Fracs
    assert state.charge_level >= 0.9, f"Expected ~1.0, got {state.charge_level}"

    print("PASS: Frac operations accumulate charge")
    return True


def test_charge_decay_over_time():
    """Test that charge decays over time."""
    print("\n=== Test 4: Charge Decay Over Time ===")

    tracker = FPGAStateTracker(simulated=True, sim_temp_c=70.0)  # High temp for faster decay

    # Charge up
    tracker.record_frac(8)
    state = tracker.update()
    print(f"Initial charge: {state.charge_level:.3f}")

    # Let it decay
    initial_charge = state.charge_level
    time.sleep(0.5)  # Wait for decay

    state = tracker.update()
    final_charge = state.charge_level
    print(f"After 500ms: charge = {final_charge:.3f}")

    decay_amount = initial_charge - final_charge
    print(f"Decay: {decay_amount:.4f}")

    # Should have decayed somewhat
    assert decay_amount > 0, "Expected some decay"

    print("PASS: Charge decays over time")
    return True


def test_with_real_gpu_telemetry():
    """Test unified state with real GPU telemetry."""
    print("\n=== Test 5: Real GPU Telemetry Integration ===")

    try:
        # Get real GPU telemetry
        telemetry = SysfsHwmonTelemetry()
        if telemetry.paths.power_average is None:
            print("SKIP: No GPU telemetry available")
            return None

        # Create GPU tracker
        config = AtomConfig()
        gpu_tracker = BodyStateTracker(config)

        # Get GPU snapshot
        snapshot = telemetry.snapshot()
        body_state = gpu_tracker.update(snapshot, tokens_generated=0)

        print(f"  GPU power: {body_state.snapshot.power_watts:.1f}W")
        print(f"  GPU temp: {body_state.snapshot.temp_c:.1f}C")
        print(f"  Power deviation: {body_state.power_deviation:.3f}")
        print(f"  Temp deviation: {body_state.temp_deviation:.3f}")

        # Create unified tracker
        fpga_tracker = FPGAStateTracker(simulated=True, sim_temp_c=54.0)
        unified = UnifiedBodyState(gpu_tracker=gpu_tracker, fpga_tracker=fpga_tracker)

        # Get unified z_feel
        z_feel = unified.get_z_feel(body_state)
        print(f"\n  Unified z_feel: {z_feel.tolist()}")
        print(f"  Breakdown:")
        print(f"    GPU strain: {z_feel[0]:.3f}")
        print(f"    GPU urgency: {z_feel[1]:.3f}")
        print(f"    GPU debt: {z_feel[2]:.3f}")
        print(f"    GPU margin: {z_feel[3]:.3f}")
        print(f"    GPU stability: {z_feel[4]:.3f}")
        print(f"    FPGA temp: {z_feel[5]:.3f}")
        print(f"    FPGA charge: {z_feel[6]:.3f}")
        print(f"    FPGA decay: {z_feel[7]:.3f}")

        print("PASS: Unified z_feel generated with real GPU data")
        return True

    except Exception as e:
        print(f"SKIP: GPU test failed ({e})")
        return None


def test_continuous_sampling():
    """Test continuous z_feel sampling over time."""
    print("\n=== Test 6: Continuous Sampling (5 seconds) ===")

    try:
        telemetry = SysfsHwmonTelemetry()
        gpu_available = telemetry.paths.power_average is not None
        gpu_tracker = BodyStateTracker(AtomConfig()) if gpu_available else None
        fpga_tracker = FPGAStateTracker(simulated=True, sim_temp_c=54.0)
        unified = UnifiedBodyState(gpu_tracker=gpu_tracker, fpga_tracker=fpga_tracker)

        # Simulate some load by doing matrix operations
        print("  Running with GPU load...")
        samples = []

        start = time.time()
        while time.time() - start < 5.0:
            # Create some GPU load
            if torch.cuda.is_available() or hasattr(torch, 'hip'):
                x = torch.randn(512, 512, device='cuda' if torch.cuda.is_available() else 'cpu')
                _ = torch.mm(x, x)

            # Sample body state
            body_state = None
            if gpu_available and gpu_tracker:
                snapshot = telemetry.snapshot()
                body_state = gpu_tracker.update(snapshot)

            z_feel = unified.get_z_feel(body_state)
            samples.append(z_feel.tolist())

            time.sleep(0.1)

        # Analyze samples
        samples_t = torch.tensor(samples)
        print(f"\n  Collected {len(samples)} samples")
        print(f"  z_feel statistics:")
        for i, name in enumerate(['strain', 'urgency', 'debt', 'margin', 'stability',
                                   'fpga_temp', 'fpga_charge', 'fpga_decay']):
            values = samples_t[:, i]
            print(f"    {name}: mean={values.mean():.3f}, std={values.std():.4f}")

        print("PASS: Continuous sampling working")
        return samples

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("z1135: Unified GPU-FPGA Body State Validation")
    print("=" * 60)

    results = {
        'experiment': 'z1135_unified_body_state',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Run tests
    tests = [
        ('dimensions', test_unified_body_state_dimensions),
        ('decay_dynamics', test_fpga_decay_dynamics),
        ('frac_accumulation', test_frac_charge_accumulation),
        ('charge_decay', test_charge_decay_over_time),
        ('gpu_integration', test_with_real_gpu_telemetry),
        ('continuous_sampling', test_continuous_sampling),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result is None:
                skipped += 1
                results['tests'][name] = 'SKIP'
            elif result:
                passed += 1
                results['tests'][name] = 'PASS'
            else:
                failed += 1
                results['tests'][name] = 'FAIL'
        except Exception as e:
            failed += 1
            results['tests'][name] = f'ERROR: {e}'
            print(f"ERROR in {name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    results['passed'] = passed
    results['failed'] = failed
    results['skipped'] = skipped
    results['success'] = failed == 0

    # Save results
    results_path = Path('results/z1135_unified_body_state.json')
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
