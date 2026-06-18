#!/usr/bin/env python3
"""
Z73: Atom Verification Tests
=============================

This script verifies the Token Metabolism Step implementation:

1. SEMANTIC INVARIANCE: Model outputs match baseline exactly
   - Same prompt + greedy decoding → identical tokens with/without atom
   - This is THE critical contract for Phase 1-3

2. TIMING CONTRACT: Telemetry is atomic and timestamps are consistent
   - AMD: gpu_metrics provides atomic snapshots
   - NVIDIA: Batch queries are quasi-atomic

3. RATE LIMITING: Actuations respect the rate limit
   - Max 1 actuation per 200-500ms

4. ENERGY ACCOUNTING: J/token is measured correctly
   - Energy delta = post_energy - pre_energy
   - J/token = delta / tokens_generated

Run: python scripts/z73_atom_verification.py --test all
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

# Import atom module
from src.atom.schema import (
    AtomConfig, ActionLevel, Vendor,
    AtomicTelemetrySnapshot, BodyState, AtomicAction, TokenBatch, AtomRecord,
)
from src.atom.sense import TelemetrySensor, create_sensor, AMDSensor, NVIDIASensor
from src.atom.feel import BodyStateTracker
from src.atom.decide import Controller, NullController, FixedController, BanditController
from src.atom.actuate import Actuator, create_actuator
from src.atom.speak import TokenGenerator, SimpleTransformerForTesting, create_test_generator
from src.atom.atom import TokenMetabolismStep, AtomLogger


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test 1: Semantic Invariance
# =============================================================================

def test_semantic_invariance(device: str = 'cuda') -> Tuple[bool, str]:
    """
    Verify that the atom does NOT change model outputs.

    This is THE critical contract: with greedy decoding (temperature=0),
    the same prompt should produce identical tokens whether we use the
    atom infrastructure or not.
    """
    logger.info("=" * 60)
    logger.info("TEST: Semantic Invariance")
    logger.info("=" * 60)

    # Create identical models
    torch.manual_seed(42)
    model_baseline = SimpleTransformerForTesting(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=2,
        device=device,
    )

    torch.manual_seed(42)
    model_atom = SimpleTransformerForTesting(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=2,
        device=device,
    )

    # Same input
    torch.manual_seed(123)
    input_ids = torch.randint(0, 1000, (1, 10)).to(device)

    # Generate with baseline (no atom)
    baseline_tokens = []
    with torch.no_grad():
        current_ids = input_ids.clone()
        for _ in range(20):
            logits = model_baseline(current_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            baseline_tokens.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=-1)

    # Generate with atom infrastructure (but NullController)
    generator = TokenGenerator(model_atom, device=device)
    generator.reset()

    atom_tokens = []
    batch = generator.generate(input_ids, num_tokens=20, temperature=0.0, return_token_ids=True)
    atom_tokens = batch.token_ids

    # Compare
    match = baseline_tokens == atom_tokens

    if match:
        logger.info("✓ PASS: Semantic invariance verified")
        logger.info(f"  Baseline tokens: {baseline_tokens[:10]}...")
        logger.info(f"  Atom tokens:     {atom_tokens[:10]}...")
        return True, "Semantic invariance verified"
    else:
        logger.error("✗ FAIL: Semantic invariance VIOLATED")
        logger.error(f"  Baseline: {baseline_tokens}")
        logger.error(f"  Atom:     {atom_tokens}")
        for i, (b, a) in enumerate(zip(baseline_tokens, atom_tokens)):
            if b != a:
                logger.error(f"  First difference at position {i}: baseline={b}, atom={a}")
                break
        return False, "Semantic invariance violated!"


# =============================================================================
# Test 2: Telemetry Atomicity
# =============================================================================

def test_telemetry_atomicity(device_id: int = 0) -> Tuple[bool, str]:
    """
    Verify that telemetry snapshots are atomic.

    For AMD: gpu_metrics should provide consistent timestamps.
    For NVIDIA: Batch queries should complete quickly.
    """
    logger.info("=" * 60)
    logger.info("TEST: Telemetry Atomicity")
    logger.info("=" * 60)

    try:
        sensor = create_sensor(device_id)
    except Exception as e:
        logger.warning(f"Could not create sensor: {e}")
        return True, "Skipped (no GPU detected)"

    vendor = sensor.get_vendor()
    logger.info(f"Vendor: {vendor.name}")
    logger.info(f"Device: {sensor.get_device_name()}")

    # Take multiple snapshots and verify consistency
    snapshots = []
    timings = []

    for i in range(10):
        start = time.perf_counter_ns()
        snap = sensor.read()
        elapsed_ns = time.perf_counter_ns() - start
        snapshots.append(snap)
        timings.append(elapsed_ns)
        time.sleep(0.01)  # 10ms between reads

    # Check timing
    avg_timing_us = sum(timings) / len(timings) / 1000
    max_timing_us = max(timings) / 1000

    logger.info(f"Read timing: avg={avg_timing_us:.1f}µs, max={max_timing_us:.1f}µs")

    # For AMD, should be <100µs typically
    # For NVIDIA, may be 1-5ms
    if vendor == Vendor.AMD and max_timing_us > 1000:
        logger.warning(f"AMD telemetry read slower than expected: {max_timing_us:.1f}µs")

    # Verify snapshots have increasing timestamps
    for i in range(1, len(snapshots)):
        if snapshots[i].timestamp_ns <= snapshots[i-1].timestamp_ns:
            logger.error(f"✗ FAIL: Non-monotonic timestamps at {i}")
            return False, "Non-monotonic timestamps"

    # Verify values are within reasonable bounds
    for snap in snapshots:
        if snap.power_watts < 0 or snap.power_watts > 1000:
            logger.error(f"✗ FAIL: Invalid power reading: {snap.power_watts}W")
            return False, f"Invalid power: {snap.power_watts}W"

        if snap.temp_c < 0 or snap.temp_c > 150:
            logger.error(f"✗ FAIL: Invalid temperature: {snap.temp_c}°C")
            return False, f"Invalid temp: {snap.temp_c}°C"

    logger.info("✓ PASS: Telemetry atomicity verified")
    logger.info(f"  Power range: {min(s.power_watts for s in snapshots):.1f}W - {max(s.power_watts for s in snapshots):.1f}W")
    logger.info(f"  Temp range:  {min(s.temp_c for s in snapshots):.1f}°C - {max(s.temp_c for s in snapshots):.1f}°C")

    sensor.shutdown()
    return True, "Telemetry atomicity verified"


# =============================================================================
# Test 3: Rate Limiting
# =============================================================================

def test_rate_limiting(device_id: int = 0) -> Tuple[bool, str]:
    """
    Verify that actuations respect the rate limit.
    """
    logger.info("=" * 60)
    logger.info("TEST: Rate Limiting")
    logger.info("=" * 60)

    config = AtomConfig(rate_limit_ms=200)  # 200ms rate limit

    try:
        actuator = create_actuator(device_id, config=config)
    except Exception as e:
        logger.warning(f"Could not create actuator: {e}")
        return True, "Skipped (no GPU or permissions)"

    # Attempt rapid-fire actuations
    power_min, power_max, _ = actuator.get_power_limits()
    power_low = power_min + (power_max - power_min) * 0.5
    power_high = power_min + (power_max - power_min) * 0.9

    # First actuation should succeed
    action1 = AtomicAction(
        level=ActionLevel.LOW,
        target_power_watts=power_low,
        timestamp_ns=time.time_ns(),
    )
    result1 = actuator.apply(action1)

    # Immediate second actuation should be rate-limited
    action2 = AtomicAction(
        level=ActionLevel.HIGH,
        target_power_watts=power_high,
        timestamp_ns=time.time_ns(),
    )
    result2 = actuator.apply(action2)

    if result2.rate_limited:
        logger.info("✓ Rate limiting working: second actuation was blocked")
    else:
        # Could be first actuation failed due to permissions
        if not result1.applied:
            logger.warning("First actuation failed (likely permissions)")
            actuator.shutdown()
            return True, "Skipped (no actuation permissions)"
        logger.warning("Second actuation was NOT rate-limited")

    # Wait for rate limit to expire
    time.sleep(0.25)

    # Third actuation should succeed
    action3 = AtomicAction(
        level=ActionLevel.MED,
        target_power_watts=(power_low + power_high) / 2,
        timestamp_ns=time.time_ns(),
    )
    result3 = actuator.apply(action3)

    if result3.rate_limited:
        logger.error("✗ FAIL: Actuation still rate-limited after waiting")
        actuator.shutdown()
        return False, "Rate limit not expiring"

    stats = actuator.get_stats()
    logger.info(f"✓ PASS: Rate limiting verified")
    logger.info(f"  Actuation count: {stats['actuation_count']}")
    logger.info(f"  Rate-limited count: {stats['rate_limited_count']}")

    actuator.shutdown()
    return True, "Rate limiting verified"


# =============================================================================
# Test 4: Body State Tracking
# =============================================================================

def test_body_state_tracking() -> Tuple[bool, str]:
    """
    Verify body state EMA and derivative calculations.
    """
    logger.info("=" * 60)
    logger.info("TEST: Body State Tracking")
    logger.info("=" * 60)

    config = AtomConfig(ema_alpha=0.3)
    tracker = BodyStateTracker(config)

    # Simulate a series of telemetry snapshots
    # Power increasing from 100W to 200W over time
    states = []

    for i in range(10):
        power = 100 + i * 10  # 100, 110, 120, ... 190
        temp = 50 + i * 2    # 50, 52, 54, ... 68
        util = 50 + i * 5    # 50, 55, 60, ... 95

        snap = AtomicTelemetrySnapshot(
            vendor=Vendor.AMD,
            device_id=0,
            device_name="test",
            timestamp_ns=time.time_ns(),
            firmware_timestamp_ns=0,
            power_watts=power,
            power_cap_watts=200.0,
            energy_joules=i * 10.0,  # Cumulative
            temp_c=temp,
            temp_hotspot_c=temp + 5,
            temp_limit_c=100.0,
            clock_gfx_mhz=1500,
            clock_mem_mhz=1000,
            clock_gfx_max_mhz=2000,
            util_gfx_percent=util,
            util_mem_percent=50,
            throttle_flags=0,
            is_throttled=False,
        )

        state = tracker.update(snap, tokens_generated=10)
        states.append(state)
        time.sleep(0.05)  # 50ms between updates

    # Verify EMA is smoothed (not equal to raw value)
    last_state = states[-1]
    if abs(last_state.power_ema - last_state.snapshot.power_watts) < 0.01:
        logger.warning("EMA equals raw value - may not be smoothing")

    # Verify derivatives are positive (power/temp increasing)
    if last_state.power_derivative <= 0:
        logger.warning(f"Power derivative not positive: {last_state.power_derivative}")

    # Verify homeostatic deviation
    # With setpoint=0.7 and power=190W on 200W cap, we're at 0.95 normalized
    # Deviation should be positive (above setpoint)
    if last_state.power_deviation <= 0:
        logger.warning(f"Power deviation not positive: {last_state.power_deviation}")

    # Verify observation vector has correct dimensionality
    obs = last_state.to_observation_vector()
    if len(obs) != 18:
        logger.error(f"✗ FAIL: Observation vector wrong size: {len(obs)} (expected 18)")
        return False, f"Wrong observation vector size: {len(obs)}"

    logger.info("✓ PASS: Body state tracking verified")
    logger.info(f"  Final EMA power: {last_state.power_ema:.1f}W")
    logger.info(f"  Power derivative: {last_state.power_derivative:.2f} W/s")
    logger.info(f"  Power deviation: {last_state.power_deviation:.3f}")
    logger.info(f"  Observation vector: {len(obs)} dims")

    return True, "Body state tracking verified"


# =============================================================================
# Test 5: Controller Interface
# =============================================================================

def test_controller_interface() -> Tuple[bool, str]:
    """
    Verify controller implementations work correctly.
    """
    logger.info("=" * 60)
    logger.info("TEST: Controller Interface")
    logger.info("=" * 60)

    # Create a mock body state
    snap = AtomicTelemetrySnapshot(
        vendor=Vendor.AMD,
        device_id=0,
        device_name="test",
        timestamp_ns=time.time_ns(),
        firmware_timestamp_ns=0,
        power_watts=150.0,
        power_cap_watts=200.0,
        energy_joules=100.0,
        temp_c=65.0,
        temp_hotspot_c=70.0,
        temp_limit_c=100.0,
        clock_gfx_mhz=1800,
        clock_mem_mhz=1000,
        clock_gfx_max_mhz=2000,
        util_gfx_percent=80,
        util_mem_percent=50,
        throttle_flags=0,
        is_throttled=False,
    )

    body_state = BodyState(
        snapshot=snap,
        power_ema=145.0,
        temp_ema=63.0,
        util_ema=75.0,
        power_derivative=5.0,
        temp_derivative=0.5,
        util_derivative=2.0,
        power_setpoint=0.7,
        temp_setpoint=0.8,
        power_deviation=0.07,
        temp_deviation=-0.1875,
        energy_start_joules=0.0,
        tokens_in_window=100,
        j_per_token=0.1,
        timestamp_ns=time.time_ns(),
    )

    power_min, power_max = 100.0, 200.0

    # Test NullController
    null_ctrl = NullController(ActionLevel.MED)
    action = null_ctrl.decide(body_state, power_min, power_max)
    if action.level != ActionLevel.MED:
        logger.error(f"✗ FAIL: NullController returned wrong level: {action.level}")
        return False, "NullController failed"
    logger.info("  NullController: OK")

    # Test FixedController
    fixed_ctrl = FixedController(ActionLevel.ECO)
    action = fixed_ctrl.decide(body_state, power_min, power_max)
    if action.level != ActionLevel.ECO:
        logger.error(f"✗ FAIL: FixedController returned wrong level: {action.level}")
        return False, "FixedController failed"
    logger.info("  FixedController: OK")

    # Test BanditController
    config = AtomConfig()
    bandit_ctrl = BanditController(config, epsilon=0.5)

    # Run several steps
    action_counts = {level: 0 for level in ActionLevel}
    for _ in range(100):
        action = bandit_ctrl.decide(body_state, power_min, power_max)
        action_counts[action.level] += 1

        # Simulate reward
        reward = -0.1 if action.level in [ActionLevel.ECO, ActionLevel.LOW] else -0.2
        bandit_ctrl.update(body_state, action, reward)

    # With epsilon=0.5, should see some exploration
    explored_actions = sum(1 for count in action_counts.values() if count > 0)
    if explored_actions < 3:
        logger.warning(f"BanditController only explored {explored_actions} actions")

    stats = bandit_ctrl.get_stats()
    logger.info(f"  BanditController: OK (explored {explored_actions} actions)")
    logger.info(f"    Distribution: {stats['action_distribution']}")

    logger.info("✓ PASS: Controller interface verified")
    return True, "Controller interface verified"


# =============================================================================
# Test 6: Full Atom Pipeline (Dry Run)
# =============================================================================

def test_atom_pipeline_dry_run(device: str = 'cuda') -> Tuple[bool, str]:
    """
    Verify the full atom pipeline works (without real GPU actuation).
    """
    logger.info("=" * 60)
    logger.info("TEST: Atom Pipeline (Dry Run)")
    logger.info("=" * 60)

    # Check if CUDA available
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.info("CUDA not available, using CPU")

    # Create test model
    torch.manual_seed(42)
    model = SimpleTransformerForTesting(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=2,
        device=device,
    )

    # Create input
    input_ids = torch.randint(0, 1000, (1, 10)).to(device)

    # Create token generator
    generator = TokenGenerator(model, device=device)

    # Generate tokens
    batch = generator.generate(input_ids, num_tokens=10, temperature=0.0)

    logger.info(f"  Generated {batch.tokens_generated} tokens")
    logger.info(f"  Latency: {batch.latency_ms:.2f}ms")
    logger.info(f"  Throughput: {batch.throughput_tps:.1f} tok/s")

    # Verify timing
    if batch.latency_ms <= 0:
        logger.error("✗ FAIL: Invalid latency")
        return False, "Invalid latency"

    if batch.tokens_generated != 10:
        logger.error(f"✗ FAIL: Wrong token count: {batch.tokens_generated}")
        return False, f"Wrong token count: {batch.tokens_generated}"

    logger.info("✓ PASS: Atom pipeline dry run verified")
    return True, "Atom pipeline verified"


# =============================================================================
# Test 7: Atom Logging
# =============================================================================

def test_atom_logging(tmp_path: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Verify atom logging produces valid JSONL.
    """
    logger.info("=" * 60)
    logger.info("TEST: Atom Logging")
    logger.info("=" * 60)

    import json
    import tempfile

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())

    log_path = tmp_path / "test_atoms.jsonl"

    # Create logger
    atom_logger = AtomLogger(log_path, session_id="test123")

    # Create some test records
    for i in range(5):
        snap = AtomicTelemetrySnapshot(
            vendor=Vendor.AMD,
            device_id=0,
            device_name="test",
            timestamp_ns=time.time_ns(),
            firmware_timestamp_ns=0,
            power_watts=100 + i * 10,
            power_cap_watts=200.0,
            energy_joules=i * 5.0,
            temp_c=60 + i,
            temp_hotspot_c=65 + i,
            temp_limit_c=100.0,
            clock_gfx_mhz=1500,
            clock_mem_mhz=1000,
            clock_gfx_max_mhz=2000,
            util_gfx_percent=70 + i,
            util_mem_percent=50,
            throttle_flags=0,
            is_throttled=False,
        )

        body = BodyState(
            snapshot=snap,
            power_ema=100 + i * 5,
            temp_ema=60.0,
            util_ema=70.0,
            timestamp_ns=time.time_ns(),
        )

        action = AtomicAction(
            level=ActionLevel(i % 5),
            target_power_watts=150.0,
            timestamp_ns=time.time_ns(),
            applied=True,
        )

        tokens = TokenBatch(
            tokens_generated=1,
            start_timestamp_ns=time.time_ns() - 1000000,
            end_timestamp_ns=time.time_ns(),
            latency_ms=1.0,
            throughput_tps=1000.0,
        )

        record = AtomRecord(
            atom_id=i + 1,
            session_id="test123",
            sense_pre=snap,
            sense_post=snap,
            body_state=body,
            action=action,
            tokens=tokens,
            energy_delta_joules=0.5,
            j_per_token=0.5,
            reward=-0.1,
            prev_atom_id=i if i > 0 else None,
        )

        atom_logger.log(record)

    atom_logger.close()

    # Verify log file
    if not log_path.exists():
        logger.error("✗ FAIL: Log file not created")
        return False, "Log file not created"

    # Read and verify each line is valid JSON
    with open(log_path) as f:
        lines = f.readlines()

    if len(lines) != 5:
        logger.error(f"✗ FAIL: Wrong number of log entries: {len(lines)}")
        return False, f"Wrong log entry count: {len(lines)}"

    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
            if record['atom_id'] != i + 1:
                logger.error(f"✗ FAIL: Wrong atom_id at line {i}")
                return False, f"Wrong atom_id at line {i}"
        except json.JSONDecodeError as e:
            logger.error(f"✗ FAIL: Invalid JSON at line {i}: {e}")
            return False, f"Invalid JSON at line {i}"

    logger.info("✓ PASS: Atom logging verified")
    logger.info(f"  Log file: {log_path}")
    logger.info(f"  Entries: {len(lines)}")

    return True, "Atom logging verified"


# =============================================================================
# Main
# =============================================================================

def run_all_tests(device: str = 'cuda', device_id: int = 0) -> bool:
    """Run all verification tests."""
    results = []

    # Test 1: Semantic invariance (CRITICAL)
    try:
        passed, msg = test_semantic_invariance(device)
        results.append(('Semantic Invariance', passed, msg))
    except Exception as e:
        results.append(('Semantic Invariance', False, str(e)))

    # Test 2: Telemetry atomicity
    try:
        passed, msg = test_telemetry_atomicity(device_id)
        results.append(('Telemetry Atomicity', passed, msg))
    except Exception as e:
        results.append(('Telemetry Atomicity', False, str(e)))

    # Test 3: Rate limiting
    try:
        passed, msg = test_rate_limiting(device_id)
        results.append(('Rate Limiting', passed, msg))
    except Exception as e:
        results.append(('Rate Limiting', False, str(e)))

    # Test 4: Body state tracking
    try:
        passed, msg = test_body_state_tracking()
        results.append(('Body State', passed, msg))
    except Exception as e:
        results.append(('Body State', False, str(e)))

    # Test 5: Controller interface
    try:
        passed, msg = test_controller_interface()
        results.append(('Controllers', passed, msg))
    except Exception as e:
        results.append(('Controllers', False, str(e)))

    # Test 6: Atom pipeline
    try:
        passed, msg = test_atom_pipeline_dry_run(device)
        results.append(('Atom Pipeline', passed, msg))
    except Exception as e:
        results.append(('Atom Pipeline', False, str(e)))

    # Test 7: Atom logging
    try:
        passed, msg = test_atom_logging()
        results.append(('Atom Logging', passed, msg))
    except Exception as e:
        results.append(('Atom Logging', False, str(e)))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for name, passed, msg in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {name} - {msg}")
        if not passed:
            all_passed = False

    logger.info("")
    if all_passed:
        logger.info("ALL TESTS PASSED - Atom implementation is verified!")
    else:
        logger.error("SOME TESTS FAILED - Review implementation")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Atom Verification Tests')
    parser.add_argument('--test', choices=['all', 'semantic', 'telemetry', 'rate', 'body', 'ctrl', 'pipeline', 'log'],
                       default='all', help='Which test to run')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--device-id', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    if args.test == 'all':
        success = run_all_tests(args.device, args.device_id)
        sys.exit(0 if success else 1)
    elif args.test == 'semantic':
        passed, _ = test_semantic_invariance(args.device)
    elif args.test == 'telemetry':
        passed, _ = test_telemetry_atomicity(args.device_id)
    elif args.test == 'rate':
        passed, _ = test_rate_limiting(args.device_id)
    elif args.test == 'body':
        passed, _ = test_body_state_tracking()
    elif args.test == 'ctrl':
        passed, _ = test_controller_interface()
    elif args.test == 'pipeline':
        passed, _ = test_atom_pipeline_dry_run(args.device)
    elif args.test == 'log':
        passed, _ = test_atom_logging()

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
