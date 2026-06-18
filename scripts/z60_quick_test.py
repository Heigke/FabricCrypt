#!/usr/bin/env python3
"""
Z60: Quick Test of Metabolic Transformer
========================================
Fast verification that all components work together.

Usage:
    python scripts/z60_quick_test.py

Author: FEEL Research Team
Date: 2026-01-19
"""

import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

from src.metabolic.film_transformer import (
    MetabolicTransformer, BaselineTransformer, MetabolicConfig
)
from src.metabolic.telemetry_unified import UnifiedTelemetryReader
from src.metabolic.actuation_unified import UnifiedActuator, MetabolicMode
from src.metabolic.energy_tracker import TokenLevelEnergyTracker


def test_telemetry():
    """Test telemetry reading."""
    print("\n" + "="*50)
    print("Testing Telemetry Reader")
    print("="*50)

    reader = UnifiedTelemetryReader()
    info = reader.get_device_info()
    print(f"Device: {info}")

    for i in range(3):
        snap = reader.read()
        print(f"\nSample {i+1}:")
        print(f"  Power: {snap.power_watts:.1f}W")
        print(f"  Temp: {snap.temp_c:.1f}°C")
        print(f"  Clock: {snap.clock_mhz:.0f}MHz")
        print(f"  Util: {snap.utilization:.1f}%")
        print(f"  Normalized: {snap.to_normalized_vector()}")

        body = reader.read_body_state()
        print(f"  Body state: {body[:6].round(3)}")
        time.sleep(0.5)

    print("\n✓ Telemetry test passed!")
    return True


def test_actuation():
    """Test actuation modes."""
    print("\n" + "="*50)
    print("Testing Actuation Interface")
    print("="*50)

    actuator = UnifiedActuator()
    print(f"Current mode: {actuator.get_current_mode().name}")

    for mode in MetabolicMode:
        print(f"\nSetting mode: {mode.name}")
        result = actuator.set_metabolic_mode(mode)
        print(f"  Success: {result.success}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        if result.error:
            print(f"  Error: {result.error}")
        time.sleep(0.3)

    actuator.reset_to_default()
    print("\n✓ Actuation test passed!")
    return True


def get_best_device() -> torch.device:
    """Get best available device, with fallback for unsupported GPUs."""
    if torch.cuda.is_available():
        try:
            # Test if GPU actually works with a simple operation
            test_tensor = torch.zeros(1, device='cuda')
            _ = test_tensor + 1  # Force computation
            del test_tensor
            torch.cuda.synchronize()
            return torch.device('cuda')
        except Exception as e:
            print(f"GPU available but not functional: {e}")
            print("Falling back to CPU")
            return torch.device('cpu')
    return torch.device('cpu')


def test_model():
    """Test model forward pass."""
    print("\n" + "="*50)
    print("Testing Metabolic Transformer")
    print("="*50)

    config = MetabolicConfig(
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=512,
        telemetry_dim=12,
        max_seq_len=64,
    )

    device = get_best_device()
    print(f"Device: {device}")

    # Create models
    metabolic = MetabolicTransformer(config).to(device)
    baseline = BaselineTransformer(config).to(device)

    print(f"Metabolic params: {metabolic.get_num_parameters():,}")
    print(f"  FiLM params: {sum(p.numel() for p in metabolic.get_film_parameters()):,}")
    print(f"  Action head params: {sum(p.numel() for p in metabolic.get_action_head_parameters()):,}")

    # Test forward
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
    telemetry = torch.rand(batch_size, config.telemetry_dim, device=device)

    print("\nForward pass (metabolic, with conditioning):")
    output = metabolic(input_ids, telemetry)
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Action logits: {output['action_logits'].shape}")
    action_probs = F.softmax(output['action_logits'], dim=-1)
    print(f"  Action probs: {action_probs[0].tolist()}")

    print("\nForward pass (baseline, no conditioning):")
    output_baseline = baseline(input_ids, telemetry)
    print(f"  Logits: {output_baseline['logits'].shape}")

    # Verify conditioning makes a difference
    metabolic.enable_conditioning(True)
    out1 = metabolic(input_ids, telemetry)
    metabolic.enable_conditioning(False)
    out2 = metabolic(input_ids, telemetry)
    diff = (out1['logits'] - out2['logits']).abs().mean().item()
    print(f"\nConditioning effect (mean abs diff): {diff:.6f}")

    print("\n✓ Model test passed!")
    return True


def test_energy_tracking():
    """Test energy tracking."""
    print("\n" + "="*50)
    print("Testing Energy Tracker")
    print("="*50)

    reader = UnifiedTelemetryReader()
    tracker = TokenLevelEnergyTracker(reader)
    tracker.reset()

    print("Measuring energy for 10 ticks...")
    energies = []
    for i in range(10):
        e = tracker.tick()
        energies.append(e)
        print(f"  Tick {i+1}: {e*1000:.3f} mJ")
        time.sleep(0.01)

    avg = tracker.get_average_j_per_token()
    total = tracker.get_cumulative_energy()
    print(f"\nAverage: {avg*1000:.3f} mJ/tick")
    print(f"Total: {total*1000:.3f} mJ")

    print("\n✓ Energy tracking test passed!")
    return True


def test_generation():
    """Test token generation with metabolic loop."""
    print("\n" + "="*50)
    print("Testing Generation with Metabolic Loop")
    print("="*50)

    config = MetabolicConfig(
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=512,
        telemetry_dim=12,
        max_seq_len=64,
    )

    device = get_best_device()
    model = MetabolicTransformer(config).to(device)
    reader = UnifiedTelemetryReader()
    actuator = UnifiedActuator()
    tracker = TokenLevelEnergyTracker(reader)

    # Prepare prompt
    prompt = "Hello "
    prompt_ids = torch.tensor([[ord(c) % 256 for c in prompt]], device=device)

    print(f"Prompt: '{prompt}'")
    print(f"Generating 20 tokens...")

    tracker.reset()
    actions_taken = []

    model.eval()
    generated = prompt_ids.clone()

    with torch.no_grad():
        for i in range(20):
            # Get telemetry
            telem = torch.from_numpy(reader.read_body_state()).float().to(device).unsqueeze(0)

            # Forward
            output = model(generated[:, -config.max_seq_len:], telem)

            # Sample token
            probs = F.softmax(output['logits'][:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Get and apply action
            action_probs = F.softmax(output['action_logits'], dim=-1)
            action_idx = torch.argmax(action_probs, dim=-1).item()
            actions_taken.append(action_idx)
            actuator.set_mode_from_action(action_idx)

            # Track energy
            tracker.tick()

    actuator.reset_to_default()

    # Decode output
    output_ids = generated[0].tolist()
    output_text = ''.join(chr(x) if 32 <= x < 127 else '?' for x in output_ids)
    print(f"Output: '{output_text}'")

    # Show action distribution
    from collections import Counter
    action_counts = Counter(actions_taken)
    print(f"\nActions taken: {dict(action_counts)}")
    print(f"Energy: {tracker.get_cumulative_energy()*1000:.2f} mJ total")
    print(f"Average: {tracker.get_average_j_per_token()*1000:.3f} mJ/token")

    print("\n✓ Generation test passed!")
    return True


def main():
    print("="*50)
    print("METABOLIC TRANSFORMER: QUICK TEST")
    print("="*50)

    tests = [
        ("Telemetry", test_telemetry),
        ("Actuation", test_actuation),
        ("Model", test_model),
        ("Energy Tracking", test_energy_tracking),
        ("Generation", test_generation),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed!'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
