#!/usr/bin/env python3
"""
GPU Signal Pattern Test v2 - Using rocm-smi for reliable metrics
Goal: Find signals that differentiate reasoning patterns
"""

import time
import subprocess
import re
from threading import Thread, Event
from typing import List, Dict, Tuple
import numpy as np

def parse_rocm_smi() -> Dict[str, float]:
    """Parse rocm-smi output for key metrics."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showpower", "--showtemp", "--showclocks"],
            capture_output=True, text=True, timeout=2
        )
        text = result.stdout

        metrics = {}

        # Temperature
        m = re.search(r'Temperature.*edge.*:\s*([\d.]+)', text)
        if m: metrics["temp_c"] = float(m.group(1))

        # Power
        m = re.search(r'Power.*:\s*([\d.]+)', text)
        if m: metrics["power_w"] = float(m.group(1))

        # GPU Use %
        m = re.search(r'GPU use.*:\s*(\d+)', text)
        if m: metrics["gpu_use_pct"] = float(m.group(1))

        # sclk (shader clock - KEY metric)
        m = re.search(r'sclk.*:\s*\d+:\s*\((\d+)Mhz\)', text)
        if m: metrics["sclk_mhz"] = float(m.group(1))

        # mclk (memory clock)
        m = re.search(r'mclk.*:\s*\d+:\s*\((\d+)Mhz\)', text)
        if m: metrics["mclk_mhz"] = float(m.group(1))

        return metrics
    except Exception as e:
        return {"error": str(e)}


def sample_metrics(duration: float, hz: int = 20) -> List[Dict]:
    """Sample GPU metrics at given Hz for duration seconds."""
    samples = []
    start = time.time()
    interval = 1.0 / hz

    while time.time() - start < duration:
        m = parse_rocm_smi()
        m["t"] = time.time() - start
        samples.append(m)
        time.sleep(interval)

    return samples


def sample_during_fn(fn, hz: int = 20) -> Tuple[any, List[Dict]]:
    """Run fn() while sampling GPU metrics."""
    samples = []
    stop = Event()

    def sampler():
        while not stop.is_set():
            m = parse_rocm_smi()
            m["t"] = time.time()
            samples.append(m)
            time.sleep(1.0 / hz)

    t = Thread(target=sampler, daemon=True)
    t.start()

    result = fn()

    stop.set()
    t.join(timeout=0.5)

    return result, samples


def analyze(samples: List[Dict], metric: str) -> Dict:
    """Analyze a single metric."""
    values = [s.get(metric, 0) for s in samples if metric in s]
    if len(values) < 2:
        return {}

    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "range": np.max(values) - np.min(values),
        "delta_mean": np.mean(np.abs(np.diff(values))),
    }


def run_generation_tests():
    """Test GPU patterns during model generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("="*60)
    print("GPU Signal Pattern Test v2")
    print("="*60)

    # Baseline check
    print("\n[1] Baseline Metrics (idle):")
    baseline = parse_rocm_smi()
    for k, v in baseline.items():
        print(f"  {k}: {v}")

    # Load model
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    print(f"\n[2] Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    # Warmup
    print("\n[3] Warmup generation...")
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)

    # Test prompts - designed to create different compute patterns
    prompts = {
        "trivial": "Hi",
        "simple_qa": "What is the capital of France?",
        "math_easy": "What is 15 + 27?",
        "math_chain": "If I have 3 apples and buy 5 more, then give away 2, how many do I have? Think step by step.",
        "code": "Write a Python function to calculate factorial recursively.",
        "reasoning": "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost? Explain your reasoning.",
        "creative": "Write a short poem about a tired GPU.",
    }

    results = {}

    print("\n[4] Running generation tests...")
    for name, prompt in prompts.items():
        print(f"\n  --- {name} ---")
        print(f"  Prompt: {prompt[:40]}...")

        # Generate with metrics sampling
        def gen_fn():
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            start = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            gen_time = time.time() - start
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            return {"text": text, "gen_time": gen_time, "tokens": out.shape[1]}

        result, samples = sample_during_fn(gen_fn, hz=30)

        # Analyze
        sclk_stats = analyze(samples, "sclk_mhz")
        power_stats = analyze(samples, "power_w")
        gpu_stats = analyze(samples, "gpu_use_pct")

        results[name] = {
            "gen_time": result["gen_time"],
            "tokens": result["tokens"],
            "output_preview": result["text"][-80:],
            "samples": len(samples),
            "sclk": sclk_stats,
            "power": power_stats,
            "gpu_use": gpu_stats,
        }

        print(f"  Time: {result['gen_time']:.2f}s | Tokens: {result['tokens']}")
        if sclk_stats:
            print(f"  SCLK: {sclk_stats['mean']:.0f}±{sclk_stats['std']:.0f} MHz (range: {sclk_stats['range']:.0f})")
        if power_stats:
            print(f"  Power: {power_stats['mean']:.1f}±{power_stats['std']:.1f} W")
        if gpu_stats:
            print(f"  GPU%: {gpu_stats['mean']:.0f}±{gpu_stats['std']:.0f}")

    # Summary comparison
    print("\n" + "="*60)
    print("SIGNAL COMPARISON")
    print("="*60)

    print("\n  Task          | SCLK(mean) | SCLK(range) | Power | GPU%")
    print("  " + "-"*56)
    for name, r in results.items():
        sclk_mean = r['sclk'].get('mean', 0)
        sclk_range = r['sclk'].get('range', 0)
        power_mean = r['power'].get('mean', 0)
        gpu_mean = r['gpu_use'].get('mean', 0)
        print(f"  {name:14s} | {sclk_mean:10.0f} | {sclk_range:11.0f} | {power_mean:5.1f} | {gpu_mean:4.0f}")

    # Identify best differentiating signal
    print("\n[5] SIGNAL QUALITY ANALYSIS:")

    for metric_name in ["sclk", "power", "gpu_use"]:
        means = [r[metric_name].get('mean', 0) for r in results.values()]
        if len(means) > 1 and max(means) > 0:
            variance_ratio = np.std(means) / (np.mean(means) + 1e-6)
            print(f"  {metric_name}: cross-task variance ratio = {variance_ratio:.3f}")

    return results


if __name__ == "__main__":
    run_generation_tests()
