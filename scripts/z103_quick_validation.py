#!/usr/bin/env python3
"""
z103_quick_validation.py - Quick FEEL Validation (non-streaming)

A simpler validation script that uses non-streaming API for reliable token counts.
"""

import argparse
import json
import hashlib
import statistics
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import requests

# Prompts
SHORT_PROMPTS = [
    ("What is the capital of France?", 50),
    ("Explain photosynthesis briefly.", 80),
    ("List 5 programming languages.", 60),
    ("What causes rain?", 70),
    ("Define machine learning.", 80),
]

LONG_PROMPT = """The following is a research paper abstract about renewable energy:

Solar photovoltaic (PV) technology has emerged as a cornerstone of the global energy transition.
Recent advances in perovskite solar cells have achieved efficiencies exceeding 25%, rivaling
traditional silicon-based technologies. However, stability and scalability challenges remain.
This paper presents a comprehensive analysis of hybrid perovskite-silicon tandem cells,
demonstrating a pathway to 30% efficiency through optimized interface engineering.

Key findings include: (1) improved charge carrier dynamics through surface passivation,
(2) enhanced light management via textured interfaces, and (3) reduced degradation rates
through encapsulation strategies. Economic modeling suggests grid parity is achievable
within 5 years at current development trajectories.

Summarize the key findings in 3 bullet points:"""


@dataclass
class Result:
    condition: str
    workload: str
    prompt_tokens: int
    completion_tokens: int
    energy_j: float
    latency_ms: float
    avg_power_w: float
    answer_hash: str


def get_energy(daemon_url: str) -> float:
    """Get energy counter in mJ."""
    try:
        resp = requests.get(f"{daemon_url}/energy", timeout=2)
        return resp.json().get('energy_mj', 0)
    except:
        return 0


def set_profile(daemon_url: str, profile: str) -> bool:
    """Set energy profile."""
    try:
        resp = requests.post(f"{daemon_url}/profile", json={"profile": profile}, timeout=2)
        return resp.status_code == 200
    except:
        return False


def generate(vllm_url: str, prompt: str, max_tokens: int) -> Tuple[str, int, int]:
    """Generate completion (non-streaming)."""
    try:
        resp = requests.post(
            f"{vllm_url}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=60,
        )
        data = resp.json()
        text = data['choices'][0]['message']['content']
        usage = data.get('usage', {})
        return text, usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
    except Exception as e:
        print(f"Error: {e}")
        return "", 0, 0


def run_single(daemon_url: str, vllm_url: str, prompt: str, max_tokens: int,
               condition: str, workload: str) -> Optional[Result]:
    """Run single inference with energy measurement."""

    e_start = get_energy(daemon_url)
    t_start = time.perf_counter()

    text, prompt_tokens, completion_tokens = generate(vllm_url, prompt, max_tokens)

    t_end = time.perf_counter()
    e_end = get_energy(daemon_url)

    if completion_tokens == 0:
        return None

    energy_delta_mj = e_end - e_start
    energy_j = energy_delta_mj / 1000
    time_delta_s = t_end - t_start
    latency_ms = time_delta_s * 1000
    avg_power_w = (energy_delta_mj / 1000) / max(time_delta_s, 0.001)
    answer_hash = hashlib.md5(text.encode()).hexdigest()[:8]

    return Result(
        condition=condition,
        workload=workload,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        energy_j=energy_j,
        latency_ms=latency_ms,
        avg_power_w=avg_power_w,
        answer_hash=answer_hash,
    )


def compute_ci95(values: List[float]) -> Tuple[float, float]:
    """Compute 95% confidence interval."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (0, 0)
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    t = 1.96 if n > 30 else 2.0
    margin = t * std / math.sqrt(n)
    return (mean - margin, mean + margin)


def aggregate(results: List[Result]) -> Dict:
    """Aggregate results."""
    if not results:
        return {}

    mj_per_tok = [(r.energy_j * 1000) / r.completion_tokens for r in results]
    powers = [r.avg_power_w for r in results]
    latencies = [r.latency_ms for r in results]

    ci = compute_ci95(mj_per_tok)

    # Answer consistency
    hashes = [r.answer_hash for r in results]
    most_common = max(set(hashes), key=hashes.count)
    consistency = hashes.count(most_common) / len(hashes)

    return {
        'n': len(results),
        'mj_per_token_mean': statistics.mean(mj_per_tok),
        'mj_per_token_std': statistics.stdev(mj_per_tok) if len(mj_per_tok) > 1 else 0,
        'mj_per_token_ci95': ci,
        'power_w_mean': statistics.mean(powers),
        'latency_ms_mean': statistics.mean(latencies),
        'latency_ms_p95': sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else latencies[0],
        'answer_consistency': consistency,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon-url", default="http://192.168.0.38:9877")
    parser.add_argument("--vllm-url", default="http://192.168.0.38:8000")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output-dir", default="results/z103_validation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test connectivity
    energy = get_energy(args.daemon_url)
    print(f"Initial energy: {energy/1e6:.1f} GJ (counter working: {energy > 0})")

    all_results = {}

    for condition in ["baseline_perf", "baseline_eco"]:
        profile = "performance" if condition == "baseline_perf" else "eco"
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition}")
        print(f"{'='*60}")

        set_profile(args.daemon_url, profile)
        time.sleep(2)  # Let profile settle

        all_results[condition] = {}

        # Short prompts workload
        print("\n--- Short Prompts ---")
        short_results = []
        for i in range(args.runs):
            prompt, max_tok = SHORT_PROMPTS[i % len(SHORT_PROMPTS)]
            result = run_single(args.daemon_url, args.vllm_url, prompt, max_tok,
                               condition, "short_prompts")
            if result:
                short_results.append(result)
                print(f"  [{i+1}/{args.runs}] {result.completion_tokens} tok, "
                      f"{result.energy_j:.2f}J, {result.avg_power_w:.0f}W")
            time.sleep(0.5)

        all_results[condition]['short_prompts'] = aggregate(short_results)

        # Long prompt workload
        print("\n--- Long Prompt ---")
        long_results = []
        for i in range(args.runs):
            result = run_single(args.daemon_url, args.vllm_url, LONG_PROMPT, 100,
                               condition, "long_prompt")
            if result:
                long_results.append(result)
                print(f"  [{i+1}/{args.runs}] {result.completion_tokens} tok, "
                      f"{result.energy_j:.2f}J, {result.avg_power_w:.0f}W")
            time.sleep(0.5)

        all_results[condition]['long_prompt'] = aggregate(long_results)

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\n| Condition | Workload | mJ/tok | 95% CI | Power (W) | Consistency |")
    print("|-----------|----------|--------|--------|-----------|-------------|")

    for cond, workloads in all_results.items():
        for wl, stats in workloads.items():
            if stats:
                ci = f"[{stats['mj_per_token_ci95'][0]:.1f}, {stats['mj_per_token_ci95'][1]:.1f}]"
                print(f"| {cond} | {wl} | {stats['mj_per_token_mean']:.1f} | "
                      f"{ci} | {stats['power_w_mean']:.0f} | "
                      f"{stats['answer_consistency']*100:.0f}% |")

    # Calculate savings
    if all_results.get('baseline_perf') and all_results.get('baseline_eco'):
        perf_short = all_results['baseline_perf'].get('short_prompts', {})
        eco_short = all_results['baseline_eco'].get('short_prompts', {})

        if perf_short and eco_short:
            savings = ((perf_short['mj_per_token_mean'] - eco_short['mj_per_token_mean'])
                      / perf_short['mj_per_token_mean'] * 100)
            print(f"\n📊 Energy savings (eco vs perf): {savings:.1f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"results_{timestamp}.json", 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'daemon_url': args.daemon_url,
                'vllm_url': args.vllm_url,
                'runs': args.runs,
            },
            'results': all_results,
        }, f, indent=2)

    # Generate LaTeX
    latex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{FEEL Energy Validation Results (NVIDIA A6000)}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Condition & Workload & mJ/tok & 95\\% CI & Power (W) \\\\",
        "\\midrule",
    ]
    for cond, workloads in all_results.items():
        for wl, stats in workloads.items():
            if stats:
                ci = f"[{stats['mj_per_token_ci95'][0]:.1f}, {stats['mj_per_token_ci95'][1]:.1f}]"
                latex.append(f"{cond} & {wl} & {stats['mj_per_token_mean']:.1f} & {ci} & {stats['power_w_mean']:.0f} \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    with open(output_dir / f"table_{timestamp}.tex", 'w') as f:
        f.write('\n'.join(latex))

    print(f"\n✅ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
