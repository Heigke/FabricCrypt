#!/usr/bin/env python3
"""
GPU Signal Pattern Explorer for v7.11

Goal: Find which GPU metrics fluctuate with model reasoning patterns
- Not just length, but token-wise compute intensity
- Find signals that differentiate reasoning styles
"""

import time
import subprocess
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

# ============================================================================
#  1. DEEP GPU METRICS (Beyond temp/power)
# ============================================================================

def read_sysfs(path: str, default=0.0) -> float:
    try:
        return float(Path(path).read_text().strip())
    except:
        return default

def read_amd_gpu_metrics() -> Dict[str, float]:
    """Read ALL available AMD GPU metrics from sysfs/hwmon."""
    metrics = {}

    # Find GPU device path
    drm_path = Path("/sys/class/drm/card0/device")
    if not drm_path.exists():
        drm_path = Path("/sys/class/drm/card1/device")

    if not drm_path.exists():
        return {"error": "no_gpu"}

    # 1. gpu_metrics binary blob (richest source)
    gpu_metrics_path = drm_path / "gpu_metrics"
    if gpu_metrics_path.exists():
        try:
            data = gpu_metrics_path.read_bytes()
            if len(data) >= 64:
                # AMD gpu_metrics structure (varies by GPU gen)
                # Common fields for RDNA2/3:
                import struct
                # Header
                fmt_rev = data[2] if len(data) > 2 else 0
                content_rev = data[3] if len(data) > 3 else 0

                # Try to parse common metrics (offset varies by version)
                if len(data) >= 80:
                    # Temperature (various sensors)
                    metrics["temp_edge"] = struct.unpack_from('<H', data, 4)[0] / 100.0 if len(data) > 6 else 0
                    metrics["temp_hotspot"] = struct.unpack_from('<H', data, 6)[0] / 100.0 if len(data) > 8 else 0
                    metrics["temp_mem"] = struct.unpack_from('<H', data, 8)[0] / 100.0 if len(data) > 10 else 0

                    # Power
                    metrics["power_socket"] = struct.unpack_from('<H', data, 26)[0] if len(data) > 28 else 0

                    # Activity (THIS IS KEY - compute utilization)
                    metrics["gfx_activity"] = struct.unpack_from('<H', data, 30)[0] / 100.0 if len(data) > 32 else 0
                    metrics["umc_activity"] = struct.unpack_from('<H', data, 32)[0] / 100.0 if len(data) > 34 else 0  # Memory controller
                    metrics["vcn_activity"] = struct.unpack_from('<H', data, 34)[0] / 100.0 if len(data) > 36 else 0  # Video encoder

                    # Clocks (can indicate workload type)
                    metrics["gfx_clock"] = struct.unpack_from('<H', data, 36)[0] if len(data) > 38 else 0
                    metrics["soc_clock"] = struct.unpack_from('<H', data, 38)[0] if len(data) > 40 else 0
                    metrics["mem_clock"] = struct.unpack_from('<H', data, 40)[0] if len(data) > 42 else 0

                    # Throttle status (CRITICAL signal!)
                    metrics["throttle_status"] = struct.unpack_from('<I', data, 64)[0] if len(data) > 68 else 0
        except Exception as e:
            metrics["gpu_metrics_error"] = str(e)

    # 2. Fallback: hwmon sensors
    hwmon_base = drm_path / "hwmon"
    if hwmon_base.exists():
        hwmon_dirs = list(hwmon_base.iterdir())
        if hwmon_dirs:
            hwmon = hwmon_dirs[0]

            # Power sensors
            for i in range(1, 5):
                pwr = hwmon / f"power{i}_average"
                if pwr.exists():
                    metrics[f"power{i}_avg"] = read_sysfs(str(pwr)) / 1e6  # uW to W

            # Fan speed (can indicate thermal response)
            fan = hwmon / "fan1_input"
            if fan.exists():
                metrics["fan_rpm"] = read_sysfs(str(fan))

            # PWM (fan duty cycle - thermal stress indicator)
            pwm = hwmon / "pwm1"
            if pwm.exists():
                metrics["fan_pwm"] = read_sysfs(str(pwm)) / 255.0  # Normalize to 0-1

    # 3. PP (Power Play) table info
    pp_features = drm_path / "pp_features"
    if pp_features.exists():
        try:
            features = pp_features.read_text()
            # Count enabled features as a proxy for power state
            metrics["pp_features_enabled"] = features.count("enabled")
        except:
            pass

    # 4. Memory info (can spike with large tensors)
    mem_used = drm_path / "mem_info_vram_used"
    mem_total = drm_path / "mem_info_vram_total"
    if mem_used.exists():
        used = read_sysfs(str(mem_used))
        total = read_sysfs(str(mem_total)) or 1
        metrics["vram_used_pct"] = (used / total) * 100
        metrics["vram_used_gb"] = used / 1e9

    # 5. GPU busy percent (alternative to gfx_activity)
    busy = drm_path / "gpu_busy_percent"
    if busy.exists():
        metrics["gpu_busy_pct"] = read_sysfs(str(busy))

    return metrics


def sample_during_generation(model, tokenizer, prompt: str, sample_hz: int = 100) -> Tuple[str, List[Dict]]:
    """
    Generate text while sampling GPU metrics at high frequency.
    Returns (generated_text, [metrics_samples])
    """
    import torch
    from threading import Thread, Event

    samples = []
    stop_event = Event()

    def sampler():
        while not stop_event.is_set():
            m = read_amd_gpu_metrics()
            m["timestamp"] = time.time()
            samples.append(m)
            time.sleep(1.0 / sample_hz)

    # Start sampler thread
    t = Thread(target=sampler, daemon=True)
    t.start()

    # Generate
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_time = time.time() - start
    stop_event.set()
    t.join(timeout=0.5)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text, samples, gen_time


def analyze_samples(samples: List[Dict]) -> Dict:
    """Analyze GPU metric patterns."""
    if len(samples) < 5:
        return {"error": "too_few_samples"}

    analysis = {}

    # Key metrics to analyze
    key_metrics = ["gfx_activity", "power_socket", "temp_edge", "gfx_clock", "umc_activity", "gpu_busy_pct"]

    for metric in key_metrics:
        values = [s.get(metric, 0) for s in samples if metric in s]
        if values:
            analysis[f"{metric}_mean"] = np.mean(values)
            analysis[f"{metric}_std"] = np.std(values)
            analysis[f"{metric}_min"] = np.min(values)
            analysis[f"{metric}_max"] = np.max(values)
            analysis[f"{metric}_range"] = np.max(values) - np.min(values)

            # Rate of change (derivative)
            if len(values) > 1:
                diffs = np.diff(values)
                analysis[f"{metric}_delta_mean"] = np.mean(np.abs(diffs))
                analysis[f"{metric}_delta_max"] = np.max(np.abs(diffs))

    return analysis


def run_prompt_comparison():
    """Compare GPU patterns across different prompt types."""

    print("="*60)
    print("GPU Signal Pattern Test")
    print("="*60)

    # First, just check what metrics we can read
    print("\n[1] Available GPU Metrics:")
    metrics = read_amd_gpu_metrics()
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # Check if we have the key signals
    key_signals = ["gfx_activity", "gfx_clock", "umc_activity", "power_socket"]
    available = [k for k in key_signals if k in metrics and metrics[k] > 0]
    print(f"\n[2] Available Key Signals: {available}")

    if not available:
        print("\n⚠️ No dynamic GPU signals available. Checking alternative sources...")

        # Try rocm-smi as fallback
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showpower", "--showtemp", "--showclocks"],
                capture_output=True, text=True, timeout=5
            )
            print("\n[rocm-smi output]:")
            print(result.stdout[:1000])
        except Exception as e:
            print(f"rocm-smi failed: {e}")

    return metrics


def run_generation_test():
    """Test GPU patterns during actual model generation."""

    print("\n" + "="*60)
    print("Generation Pattern Test")
    print("="*60)

    # Load a small model for testing
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    # Test prompts designed to create different compute patterns
    test_prompts = [
        # Simple - should be low activity
        ("simple", "What is 2+2?"),

        # Complex math - should spike compute
        ("math", "Solve step by step: If a train travels 120km in 2 hours, then speeds up to 80km/h for 3 hours, what is the total distance?"),

        # Code generation - different memory pattern
        ("code", "Write a Python function to find the nth Fibonacci number using dynamic programming."),

        # Creative - potentially different token sampling
        ("creative", "Write a haiku about a GPU working hard."),
    ]

    results = {}

    for name, prompt in test_prompts:
        print(f"\n--- Testing: {name} ---")
        print(f"Prompt: {prompt[:50]}...")

        text, samples, gen_time = sample_during_generation(model, tokenizer, prompt)
        analysis = analyze_samples(samples)

        results[name] = {
            "gen_time": gen_time,
            "num_samples": len(samples),
            "output_len": len(text),
            **analysis
        }

        print(f"  Generated {len(text)} chars in {gen_time:.2f}s")
        print(f"  Samples: {len(samples)}")

        # Key metrics comparison
        for metric in ["gfx_activity", "power_socket", "gfx_clock"]:
            mean_key = f"{metric}_mean"
            range_key = f"{metric}_range"
            if mean_key in analysis:
                print(f"  {metric}: mean={analysis[mean_key]:.1f} range={analysis.get(range_key, 0):.1f}")

    # Compare patterns
    print("\n" + "="*60)
    print("Pattern Comparison")
    print("="*60)

    for metric in ["gfx_activity_mean", "power_socket_mean", "gfx_clock_mean"]:
        print(f"\n{metric}:")
        for name in results:
            if metric in results[name]:
                print(f"  {name}: {results[name][metric]:.2f}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # Full test with model loading
        run_generation_test()
    else:
        # Quick metrics check
        run_prompt_comparison()
