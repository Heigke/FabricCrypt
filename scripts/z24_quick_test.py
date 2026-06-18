#!/usr/bin/env python3
"""
FEEL z24: Quick Test Script
===========================
Tests all z24 components without loading the full model.
Use this to verify the system works on your AMD GPU.

Author: FEEL Research Team
Date: 2026-01-13
"""

import os
import sys
import time
import torch
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_sensor_hub():
    """Test the rich sensor hub."""
    print("\n" + "=" * 70)
    print("TEST 1: Rich Sensor Hub (32-dim AMD GPU telemetry)")
    print("=" * 70)

    try:
        from modeling.z24_sensor_hub import (
            AMDSensorHub, SimulatedSensorHub, SENSOR_DIM, SENSOR_NAMES
        )

        print(f"  Sensor dimensions: {SENSOR_DIM}")
        print(f"  Sensor names (first 8): {SENSOR_NAMES[:8]}")

        # Try real sensor hub
        try:
            hub = AMDSensorHub(card_id=1, sample_rate_hz=20, auto_start=True)
            print("\n  [Real Sensor Hub]")

            time.sleep(1)  # Let it collect samples

            raw = hub.read_raw()
            print(f"    Edge temp: {raw.get('edge_temp', 0):.1f}°C")
            print(f"    Socket power: {raw.get('socket_power', 0):.1f}W")
            print(f"    GPU busy: {raw.get('gpu_busy', 0):.1f}%")
            print(f"    Stress composite: {raw.get('stress_composite', 0):.3f}")

            tensor = hub.read_tensor()
            print(f"\n    Tensor shape: {tensor.shape}")
            print(f"    Tensor device: {tensor.device}")
            print(f"    First 8 values: {tensor[:8].tolist()}")

            hub.stop()
            print("\n  [PASS] Real sensor hub working!")

        except Exception as e:
            print(f"\n  [INFO] Real sensors not available ({e})")
            print("  Testing simulated sensor hub...")

            sim = SimulatedSensorHub(device="cuda")
            sim.set_stress_level(0.7)

            tensor = sim.read_tensor()
            print(f"    Simulated tensor shape: {tensor.shape}")
            print(f"    Stress composite (idx 31): {tensor[31].item():.3f}")

            print("\n  [PASS] Simulated sensor hub working!")

        return True

    except Exception as e:
        print(f"  [FAIL] Sensor hub test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hard_skip_block():
    """Test the hard-skip block with FiLM and strain."""
    print("\n" + "=" * 70)
    print("TEST 2: Hard-Skip Block (Gate + FiLM + Strain)")
    print("=" * 70)

    try:
        from modeling.z24_hard_skip import (
            EmbodiedSkipBlock, SensorGate, FiLMLayer, StrainEmbedding,
            SENSOR_DIM
        )

        hidden_dim = 256
        batch_size = 2
        seq_len = 10

        # Create dummy layer
        class DummyLayer(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x, **kwargs):
                return self.linear(x)

        dummy = DummyLayer(hidden_dim).cuda()
        block = EmbodiedSkipBlock(
            original_layer=dummy,
            hidden_dim=hidden_dim,
            layer_idx=0,
            sensor_dim=SENSOR_DIM,
            use_film=True,
            use_strain=True,
        ).cuda()

        # Test with low stress
        print("\n  [Low Stress Test]")
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")
        low_sensors = torch.zeros(SENSOR_DIM, device="cuda")
        low_sensors[31] = 0.2  # Low stress
        block.set_sensors(low_sensors)

        block.train()
        out_train = block(x)
        print(f"    Training output shape: {out_train.shape}")
        print(f"    Gate prob: {block.gate.last_gate_prob:.3f}")
        if block.strain:
            print(f"    Strain magnitude: {block.strain.last_strain_magnitude:.4f}")
        if block.film:
            print(f"    FiLM gamma: {block.film.last_gamma_mean:.4f}")

        # Test with high stress
        print("\n  [High Stress Test]")
        high_sensors = torch.zeros(SENSOR_DIM, device="cuda")
        high_sensors[31] = 0.9  # High stress
        high_sensors[0:4] = 0.8  # High temps
        block.set_sensors(high_sensors)
        block.reset_stats()

        out_train = block(x)
        print(f"    Gate prob: {block.gate.last_gate_prob:.3f}")
        if block.strain:
            print(f"    Strain magnitude: {block.strain.last_strain_magnitude:.4f}")

        # Test inference (hard skip)
        print("\n  [Inference Hard Skip Test]")
        block.eval()
        with torch.no_grad():
            out_eval = block(x)
        print(f"    Output shape: {out_eval.shape}")
        print(f"    Skip rate: {block.get_skip_rate():.3f}")

        print("\n  [PASS] Hard-skip block working!")
        return True

    except Exception as e:
        print(f"  [FAIL] Hard-skip block test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_function():
    """Test the multi-objective loss function."""
    print("\n" + "=" * 70)
    print("TEST 3: Multi-Objective Loss Function")
    print("=" * 70)

    try:
        from modeling.z24_embodied_model import EmbodiedLoss

        loss_fn = EmbodiedLoss(
            ce_weight=1.0,
            metabolic_weight=0.3,
            strain_weight=0.2,
            stability_weight=0.1,
            hardware_target_weight=0.2,
        )

        # Simulate inputs
        ce_loss = torch.tensor(2.5, requires_grad=True)
        gate_tensors = [
            torch.tensor(0.7, requires_grad=True),
            torch.tensor(0.5, requires_grad=True),
            torch.tensor(0.3, requires_grad=True),
        ]
        strain_magnitudes = [0.1, 0.2, 0.4]
        stress_level = 0.7

        # Compute loss
        output = loss_fn(
            ce_loss=ce_loss,
            gate_tensors=gate_tensors,
            strain_magnitudes=strain_magnitudes,
            stress_level=stress_level,
        )

        print(f"\n  Loss Components:")
        print(f"    CE Loss:           {output.ce_loss:.4f}")
        print(f"    Metabolic Penalty: {output.metabolic_penalty:.4f}")
        print(f"    Strain Alignment:  {output.strain_alignment:.4f}")
        print(f"    Stability Penalty: {output.stability_penalty:.4f}")
        print(f"    Hardware Target:   {output.hardware_target_loss:.4f}")
        print(f"    Total Loss:        {output.total_loss.item():.4f}")

        # Test gradient flow
        output.total_loss.backward()
        print(f"\n  Gradient check: ce_loss.grad = {ce_loss.grad}")
        print(f"  Gradient check: gate[0].grad = {gate_tensors[0].grad}")

        print("\n  [PASS] Loss function working!")
        return True

    except Exception as e:
        print(f"  [FAIL] Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_causal_validation():
    """Test the causal validation framework."""
    print("\n" + "=" * 70)
    print("TEST 4: Causal Validation Framework")
    print("=" * 70)

    try:
        from modeling.z24_causal_validation import (
            CausalValidationReport, CausalValidator
        )
        import numpy as np
        from datetime import datetime

        # Create mock report
        report = CausalValidationReport(
            timestamp=datetime.now().isoformat(),
            model_path="test",
            n_prompts=5,
            n_trials=3,
        )

        # Simulate ablation results
        report.hot_cold_word_diff = {
            "full": 25.0,           # Large effect in full mode
            "shuffle_sensors": 5.0, # Small effect when shuffled
            "frozen_sensors": 3.0,  # Small effect when frozen
            "baseline": 2.0,        # No effect in baseline
        }

        report.hot_cold_word_ratio = {
            "full": 0.65,
            "shuffle_sensors": 0.95,
            "frozen_sensors": 0.98,
            "baseline": 1.02,
        }

        report.strain_behavior_correlation = {"full": -0.45}
        report.gate_sensor_correlation = {"full": -0.52}

        # Compute causal scores
        validator = CausalValidator.__new__(CausalValidator)
        validator.ABLATION_MODES = ["full", "shuffle_sensors", "frozen_sensors", "baseline"]

        scores = validator._compute_causal_scores(report)
        report.causal_scores = scores
        report.overall_causal_score = np.mean(list(scores.values()))
        report.verdict = validator._get_verdict(report)

        print("\n  Causal Scores:")
        for test, score in scores.items():
            status = "PASS" if score >= 0.5 else "FAIL"
            print(f"    {test:<30}: {score:.2f} [{status}]")

        print(f"\n  Overall Score: {report.overall_causal_score:.2f}")
        print(f"  Verdict: {report.verdict}")

        print("\n  [PASS] Causal validation framework working!")
        return True

    except Exception as e:
        print(f"  [FAIL] Causal validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda_availability():
    """Test CUDA availability for AMD GPU."""
    print("\n" + "=" * 70)
    print("TEST 0: CUDA/ROCm Availability")
    print("=" * 70)

    try:
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name()}")

            # Quick tensor test
            x = torch.randn(100, 100, device="cuda")
            y = torch.matmul(x, x)
            print(f"  Matrix multiply test: {y.shape} [OK]")

            print("\n  [PASS] CUDA/ROCm working!")
            return True
        else:
            print("\n  [WARN] CUDA not available!")
            return False

    except Exception as e:
        print(f"  [FAIL] CUDA test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("FEEL z24: QUICK TEST SUITE")
    print("=" * 70)
    print(f"HSA_OVERRIDE_GFX_VERSION = {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}")

    results = {}

    # Test 0: CUDA
    results["cuda"] = test_cuda_availability()

    if not results["cuda"]:
        print("\n[ABORT] CUDA not available, cannot continue tests.")
        return

    # Test 1: Sensor Hub
    results["sensor_hub"] = test_sensor_hub()

    # Test 2: Hard Skip Block
    results["hard_skip"] = test_hard_skip_block()

    # Test 3: Loss Function
    results["loss"] = test_loss_function()

    # Test 4: Causal Validation
    results["causal"] = test_causal_validation()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test:<20}: [{status}]")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All tests passed! Ready to train.")
        print("\n  To start training:")
        print("    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z24_embodied_trainer.py")
    else:
        print("\n  Some tests failed. Check errors above.")

    print("=" * 70)


if __name__ == "__main__":
    main()
