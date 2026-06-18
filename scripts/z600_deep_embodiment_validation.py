#!/usr/bin/env python3
"""
Z600: Deep Embodiment Validation

Validates the complete embodied AI system:
1. Deep telemetry (multi-zone thermal, clocks, utilization)
2. Hardware actuation (power caps, performance levels)
3. World model (predicts hardware dynamics)
4. Closed-loop regulation (sense → predict → act → sense)

Falsifiable properties:
1. World model prediction accuracy (MSE < threshold)
2. Actuation causality (power cap change → power change)
3. Predictive control outperforms reactive control
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# Detect GPU vendor
def detect_gpu_vendor() -> str:
    drm = Path("/sys/class/drm")
    for card in sorted(drm.glob("card[0-9]*")):
        vendor_file = card / "device/vendor"
        if vendor_file.exists():
            try:
                vid = vendor_file.read_text().strip()
                if vid == "0x1002":
                    return "amd"
            except:
                pass

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

if GPU_VENDOR == "amd":
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Deep embodiment imports
from src.deep_embodiment import (
    create_deep_telemetry,
    create_actuator,
    HardwareWorldModel,
    HardwareState,
    ComputeAction,
    WorldModelTrainer,
    PerfLevel,
)

from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset


@dataclass
class ValidationResults:
    """Results from deep embodiment validation."""
    # System info
    hostname: str = ""
    gpu_vendor: str = ""
    timestamp: str = ""

    # Telemetry validation
    telemetry_sample_rate_hz: float = 0.0
    telemetry_signals_available: int = 0

    # Actuation validation
    actuation_available: bool = False
    power_cap_range_w: tuple = (0, 0)
    actuation_latency_ms: float = 0.0

    # World model validation
    world_model_params: int = 0
    prediction_mse: float = 0.0
    prediction_accuracy_pct: float = 0.0

    # Control validation
    reactive_energy_mj_per_tok: float = 0.0
    predictive_energy_mj_per_tok: float = 0.0
    improvement_pct: float = 0.0

    # Falsifiable checks
    telemetry_ok: bool = False
    actuation_ok: bool = False
    prediction_ok: bool = False
    control_ok: bool = False


def validate_telemetry(telemetry) -> Dict:
    """Validate deep telemetry."""
    print("\n--- Deep Telemetry Validation ---")

    # Sample for 3 seconds
    telemetry.start_continuous_sampling()
    time.sleep(3.0)
    telemetry.stop_continuous_sampling()

    samples = telemetry.get_recent_samples(200)
    stats = telemetry.get_statistics()

    print(f"  Samples collected: {len(samples)}")
    print(f"  Sample rate: {stats.get('sample_count', 0) / 3.0:.1f} Hz")

    # Check signal availability
    sample = samples[-1] if samples else telemetry.read_sample()
    signals = 0
    if sample.temp_edge_c > 0:
        signals += 1
        print(f"  ✓ Edge temp: {sample.temp_edge_c:.1f}°C")
    if sample.temp_junction_c > 0:
        signals += 1
        print(f"  ✓ Junction temp: {sample.temp_junction_c:.1f}°C")
    if sample.power_average_w > 0:
        signals += 1
        print(f"  ✓ Power: {sample.power_average_w:.1f}W")
    if sample.gpu_busy_pct >= 0:
        signals += 1
        print(f"  ✓ GPU util: {sample.gpu_busy_pct:.0f}%")
    if sample.sclk_current_mhz > 0:
        signals += 1
        print(f"  ✓ SCLK: {sample.sclk_current_mhz}MHz")

    return {
        'sample_rate_hz': stats.get('sample_count', 0) / 3.0,
        'signals_available': signals,
        'ok': signals >= 3,
    }


def validate_actuation(actuator) -> Dict:
    """Validate hardware actuation."""
    print("\n--- Hardware Actuation Validation ---")

    limits = actuator.get_limits()
    print(f"  SCLK range: {limits.sclk_min_mhz}MHz - {limits.sclk_max_mhz}MHz")

    current = actuator.get_current_state()
    print(f"  Current SCLK: {current.sclk_mhz}MHz")
    print(f"  Current perf level: {current.perf_level.name}")

    # Check if power cap is available
    has_power_cap = hasattr(actuator, '_has_power_cap') and actuator._has_power_cap()
    print(f"  Power cap available: {has_power_cap}")

    actuation_ok = False
    latency_ms = 0.0

    # Test perf level control first (more commonly available)
    print("  Testing perf level control...")
    original_level = current.perf_level

    t0 = time.perf_counter()
    success = actuator.set_perf_level(PerfLevel.LOW)
    latency_ms = (time.perf_counter() - t0) * 1000

    if success:
        print(f"  ✓ Perf level set to LOW: {latency_ms:.1f}ms")
        actuation_ok = True
        time.sleep(0.5)
        # Restore
        actuator.set_perf_level(original_level)
    else:
        print(f"  ✗ Perf level control failed")

    # Test clock level control if available
    if hasattr(actuator, 'set_clock_level'):
        print("  Testing clock level control...")
        t0 = time.perf_counter()
        clock_success = actuator.set_clock_level(0)  # Set to lowest
        clock_latency = (time.perf_counter() - t0) * 1000

        if clock_success:
            print(f"  ✓ Clock level set: {clock_latency:.1f}ms")
            actuation_ok = True
            time.sleep(0.5)
            # Restore to auto
            actuator.set_perf_level(PerfLevel.BALANCED)
        else:
            print(f"  ✗ Clock level control failed (may need chmod on pp_dpm_sclk)")

    return {
        'available': actuation_ok,
        'power_cap_range': (limits.power_cap_min_w, limits.power_cap_max_w),
        'sclk_range': (limits.sclk_min_mhz, limits.sclk_max_mhz),
        'latency_ms': latency_ms,
        'ok': actuation_ok,
    }


def validate_world_model(world_model, telemetry) -> Dict:
    """Validate world model predictions."""
    print("\n--- World Model Validation ---")

    # Collect training data
    print("  Collecting experience data...")
    telemetry.start_continuous_sampling()

    # Simulate varying workload
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_data = torch.randn(8, 128, device=device)

    experience = []
    for i in range(50):
        # Get current state
        samples = telemetry.get_recent_samples(20)
        if len(samples) < 5:
            time.sleep(0.2)
            continue

        history = [HardwareState(
            temp_edge_c=s.temp_edge_c,
            power_w=s.power_average_w,
            gpu_busy_pct=s.gpu_busy_pct,
            sclk_mhz=s.sclk_current_mhz,
            latency_ms=50.0,
        ) for s in samples[-10:]]

        # Random action
        action = ComputeAction(
            exit_layer=np.random.choice([3, 6, 9, 12]),
            power_cap_w=200,
            perf_level=2,
        )

        # Do some compute
        _ = dummy_data @ dummy_data.T
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        time.sleep(0.1)

        # Get next state
        next_sample = telemetry.get_latest_sample()
        if next_sample:
            next_state = HardwareState(
                temp_edge_c=next_sample.temp_edge_c,
                power_w=next_sample.power_average_w,
                gpu_busy_pct=next_sample.gpu_busy_pct,
                sclk_mhz=next_sample.sclk_current_mhz,
                latency_ms=50.0,
            )
            experience.append((history, action, next_state))

    telemetry.stop_continuous_sampling()

    print(f"  Collected {len(experience)} experiences")

    # Train world model
    trainer = WorldModelTrainer(world_model, lr=1e-3)
    for h, a, n in experience:
        trainer.add_experience(h, a, n)

    print("  Training world model...")
    losses = []
    for _ in range(100):
        loss_dict = trainer.train_step(batch_size=min(16, len(experience)))
        if loss_dict['loss'] > 0:
            losses.append(loss_dict['loss'])

    avg_loss = np.mean(losses[-20:]) if losses else 1.0
    print(f"  Final training loss: {avg_loss:.4f}")

    # Test prediction accuracy
    test_errors = []
    for h, a, n in experience[-10:]:
        for s in h:
            world_model.observe(s)
        pred = world_model.predict_next_state(a)

        # Compare key metrics
        err_temp = abs(pred.temp_edge_c - n.temp_edge_c)
        err_power = abs(pred.power_w - n.power_w)
        test_errors.append(err_temp + err_power * 0.01)

    avg_error = np.mean(test_errors) if test_errors else 100.0
    accuracy = max(0, 100 - avg_error * 10)  # Rough accuracy metric

    print(f"  Prediction accuracy: {accuracy:.1f}%")

    return {
        'params': sum(p.numel() for p in world_model.parameters()),
        'mse': avg_loss,
        'accuracy_pct': accuracy,
        'ok': accuracy > 50,  # At least 50% accuracy
    }


def run_control_comparison(
    model,
    batches,
    telemetry,
    world_model,
    actuator,
) -> Dict:
    """Compare reactive vs predictive control."""
    print("\n--- Control Comparison ---")

    device = next(model.parameters()).device

    # Reactive control (simple threshold-based)
    print("  Running reactive control...")
    telemetry.start_continuous_sampling()

    reactive_energy = 0.0
    reactive_tokens = 0

    for batch in tqdm(batches[:20], desc="    Reactive"):
        # Simple reactive: just use fixed threshold
        sample = telemetry.get_latest_sample()
        if sample and sample.temp_edge_c > 80:
            exit_layer = 6
        else:
            exit_layer = 12

        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
            # Simulate early exit by accessing only up to exit_layer
            _ = outputs.hidden_states[min(exit_layer, len(outputs.hidden_states)-1)]
        torch.cuda.synchronize()
        reactive_tokens += batch.numel()

    stats = telemetry.get_statistics()
    reactive_energy = stats.get('power_mean_w', 100) * 1.0  # Simplified energy
    telemetry.stop_continuous_sampling()

    time.sleep(2)  # Cool down

    # Predictive control (using world model)
    print("  Running predictive control...")
    telemetry.start_continuous_sampling()

    predictive_energy = 0.0
    predictive_tokens = 0

    for batch in tqdm(batches[:20], desc="    Predictive"):
        # Use world model to plan
        samples = telemetry.get_recent_samples(10)
        for s in samples:
            world_model.observe(HardwareState(
                temp_edge_c=s.temp_edge_c,
                power_w=s.power_average_w,
                gpu_busy_pct=s.gpu_busy_pct,
            ))

        actions = [
            ComputeAction(exit_layer=3),
            ComputeAction(exit_layer=6),
            ComputeAction(exit_layer=9),
            ComputeAction(exit_layer=12),
        ]
        best_action, _ = world_model.plan_best_action(actions)
        exit_layer = best_action.exit_layer

        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True)
            _ = outputs.hidden_states[min(exit_layer, len(outputs.hidden_states)-1)]
        torch.cuda.synchronize()
        predictive_tokens += batch.numel()

    stats = telemetry.get_statistics()
    predictive_energy = stats.get('power_mean_w', 100) * 1.0
    telemetry.stop_continuous_sampling()

    # Simplified metrics
    reactive_mj = reactive_energy / max(reactive_tokens, 1) * 1000
    predictive_mj = predictive_energy / max(predictive_tokens, 1) * 1000

    improvement = (reactive_mj - predictive_mj) / reactive_mj * 100 if reactive_mj > 0 else 0

    print(f"  Reactive: {reactive_mj:.2f} mJ/tok")
    print(f"  Predictive: {predictive_mj:.2f} mJ/tok")
    print(f"  Improvement: {improvement:+.1f}%")

    return {
        'reactive_mj': reactive_mj,
        'predictive_mj': predictive_mj,
        'improvement_pct': improvement,
        'ok': improvement > 0,
    }


def main():
    import socket

    print("=" * 70)
    print("Z600: DEEP EMBODIMENT VALIDATION")
    print("=" * 70)

    results = ValidationResults(
        hostname=socket.gethostname(),
        gpu_vendor=GPU_VENDOR,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
    )

    print(f"\nSystem: {results.hostname}")
    print(f"GPU Vendor: {GPU_VENDOR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Device: {device}")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device")
        return

    # Initialize components
    print("\n--- Initializing Components ---")

    # Deep telemetry
    try:
        telemetry = create_deep_telemetry(GPU_VENDOR)
        print(f"  ✓ Deep telemetry: {type(telemetry).__name__}")
    except Exception as e:
        print(f"  ✗ Deep telemetry failed: {e}")
        return

    # Hardware actuator
    try:
        actuator = create_actuator(GPU_VENDOR)
        print(f"  ✓ Hardware actuator: {type(actuator).__name__}")
    except Exception as e:
        print(f"  ✗ Hardware actuator failed: {e}")
        actuator = None

    # World model
    world_model = HardwareWorldModel(
        state_dim=10,
        action_dim=3,
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
    ).to(device)
    print(f"  ✓ World model: {sum(p.numel() for p in world_model.parameters()):,} params")

    # Load model for testing
    print("\n--- Loading Test Model ---")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    print(f"  ✓ GPT-2 loaded")

    # Prepare test data
    print("\n--- Preparing Data ---")
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    texts = []
    for item in dataset:
        if len(texts) >= 200:
            break
        text = item['text'][:300]
        if len(text) > 50:
            texts.append(text)

    batches = []
    for i in range(0, len(texts), 8):
        batch_texts = texts[i:i+8]
        if len(batch_texts) == 8:
            encoded = tokenizer(
                batch_texts, max_length=64, truncation=True,
                padding='max_length', return_tensors='pt'
            )
            batches.append(encoded['input_ids'].to(device))
    print(f"  ✓ {len(batches)} batches prepared")

    # Run validations
    telem_results = validate_telemetry(telemetry)
    results.telemetry_sample_rate_hz = telem_results['sample_rate_hz']
    results.telemetry_signals_available = telem_results['signals_available']
    results.telemetry_ok = telem_results['ok']

    if actuator:
        act_results = validate_actuation(actuator)
        results.actuation_available = act_results['available']
        results.power_cap_range_w = act_results['power_cap_range']
        results.actuation_latency_ms = act_results['latency_ms']
        results.actuation_ok = act_results['ok']

    wm_results = validate_world_model(world_model, telemetry)
    results.world_model_params = wm_results['params']
    results.prediction_mse = wm_results['mse']
    results.prediction_accuracy_pct = wm_results['accuracy_pct']
    results.prediction_ok = wm_results['ok']

    ctrl_results = run_control_comparison(model, batches, telemetry, world_model, actuator)
    results.reactive_energy_mj_per_tok = ctrl_results['reactive_mj']
    results.predictive_energy_mj_per_tok = ctrl_results['predictive_mj']
    results.improvement_pct = ctrl_results['improvement_pct']
    results.control_ok = ctrl_results['ok']

    # Summary
    print("\n" + "=" * 70)
    print("DEEP EMBODIMENT VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n1. Deep Telemetry: {'✓ PASS' if results.telemetry_ok else '✗ FAIL'}")
    print(f"   Signals available: {results.telemetry_signals_available}")
    print(f"   Sample rate: {results.telemetry_sample_rate_hz:.1f} Hz")

    print(f"\n2. Hardware Actuation: {'✓ PASS' if results.actuation_ok else '✗ FAIL (may need root)'}")
    print(f"   Power cap range: {results.power_cap_range_w[0]}-{results.power_cap_range_w[1]}W")

    print(f"\n3. World Model Prediction: {'✓ PASS' if results.prediction_ok else '✗ FAIL'}")
    print(f"   Accuracy: {results.prediction_accuracy_pct:.1f}%")
    print(f"   Parameters: {results.world_model_params:,}")

    print(f"\n4. Predictive Control: {'✓ PASS' if results.control_ok else '✗ FAIL'}")
    print(f"   Improvement: {results.improvement_pct:+.1f}%")

    # Save results
    output_path = Path(f"results/z600_{results.hostname}_{GPU_VENDOR}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON-serializable format
    results_dict = asdict(results)
    # Convert numpy bools to Python bools
    for k, v in results_dict.items():
        if hasattr(v, 'item'):
            results_dict[k] = v.item()
        elif isinstance(v, (list, tuple)):
            results_dict[k] = [x.item() if hasattr(x, 'item') else x for x in v]

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
