#!/usr/bin/env python3
"""
z99_chunked_prefill_test.py - Test vLLM chunked prefill mode under energy profiles

vLLM's chunked prefill (--enable-chunked-prefill) splits long prompts into chunks,
which can change the energy profile of inference:
- Standard prefill: One large forward pass
- Chunked prefill: Multiple smaller forward passes

This script compares energy characteristics between modes and profiles.

Usage:
    # Start vLLM server WITHOUT chunked prefill:
    vllm serve <model> --port 8000

    # Start vLLM server WITH chunked prefill:
    vllm serve <model> --port 8001 --enable-chunked-prefill --max-num-batched-tokens 512

    # Run comparison:
    python z99_chunked_prefill_test.py --standard-port 8000 --chunked-port 8001

Requirements:
    - Two vLLM server instances (with and without chunked prefill)
    - FEEL actuator daemon running
    - NVML-capable GPU
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics

import requests

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.actuator.daemon_actuator import DaemonActuator
from src.metabolic.energy_harness import EnergyMonitor, compute_energy_from_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


@dataclass
class ChunkedPrefillRun:
    """Results from a single inference run."""
    mode: str  # "standard" or "chunked"
    profile: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    energy_j: float
    energy_per_token_mj: float
    prefill_time_ms: float
    decode_time_ms: float
    total_time_ms: float
    ttft_ms: Optional[float]
    avg_power_w: float
    peak_power_w: float
    tokens_per_second: float
    used_counter: bool


@dataclass
class ChunkedPrefillComparison:
    """Comparison between standard and chunked prefill."""
    profile: str
    prompt_length: int
    standard_energy_j: float
    chunked_energy_j: float
    energy_diff_pct: float
    standard_ttft_ms: float
    chunked_ttft_ms: float
    ttft_diff_pct: float
    standard_tps: float
    chunked_tps: float
    tps_diff_pct: float


class VLLMClient:
    """Simple vLLM HTTP client."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base_url = f"http://{host}:{port}"

    def health_check(self) -> bool:
        """Check if server is available."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Tuple[str, int, int, Optional[float]]:
        """
        Generate completion and return (text, prompt_tokens, completion_tokens, ttft_ms).
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        start = time.perf_counter()
        ttft_ms = None
        text_chunks = []
        prompt_tokens = 0
        completion_tokens = 0

        if stream:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        if "choices" in chunk:
                            text_chunks.append(chunk["choices"][0].get("text", ""))
                        if "usage" in chunk:
                            prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                            completion_tokens = chunk["usage"].get("completion_tokens", 0)
                    except json.JSONDecodeError:
                        pass

            text = "".join(text_chunks)
        else:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            text = result["choices"][0]["text"]
            prompt_tokens = result["usage"]["prompt_tokens"]
            completion_tokens = result["usage"]["completion_tokens"]

        return text, prompt_tokens, completion_tokens, ttft_ms


def generate_prompts_by_length(lengths: List[int]) -> Dict[int, str]:
    """Generate prompts of approximately specified token lengths."""
    prompts = {}

    # Base content (repeated to reach length)
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
        "The five boxing wizards jump quickly. "
    )

    for target_length in lengths:
        # Roughly 4 chars per token
        target_chars = target_length * 4
        repetitions = max(1, target_chars // len(base_text) + 1)
        prompt = (base_text * repetitions)[:target_chars]
        prompt += "\n\nPlease summarize the above text briefly:"
        prompts[target_length] = prompt

    return prompts


def run_single_inference(
    client: VLLMClient,
    actuator: DaemonActuator,
    prompt: str,
    max_tokens: int,
    profile: str,
    mode: str,
) -> ChunkedPrefillRun:
    """Run a single inference with energy measurement."""

    # Set profile
    actuator.set_profile(profile)
    time.sleep(0.1)

    # Start monitoring
    monitor = EnergyMonitor(actuator, sample_interval_s=0.05)
    monitor.start()
    monitor.set_phase("prefill")

    start_time = time.perf_counter()

    # Run inference
    text, prompt_tokens, completion_tokens, ttft_ms = client.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        stream=True,
    )

    if ttft_ms is not None:
        monitor.set_phase("decode")

    total_time_ms = (time.perf_counter() - start_time) * 1000

    # Stop monitoring
    samples = monitor.stop()

    # Compute energy
    total_energy_j, used_counter = compute_energy_from_samples(samples)

    # Compute metrics
    total_tokens = prompt_tokens + completion_tokens
    energy_per_token_mj = (total_energy_j * 1000 / total_tokens) if total_tokens > 0 else 0

    # Timing
    prefill_time_ms = ttft_ms if ttft_ms else 0
    decode_time_ms = total_time_ms - prefill_time_ms

    # Power stats
    powers = [s.power_watts for s in samples if s.power_watts > 0]
    avg_power = statistics.mean(powers) if powers else 0
    peak_power = max(powers) if powers else 0

    # Throughput
    tps = completion_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0

    return ChunkedPrefillRun(
        mode=mode,
        profile=profile,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        energy_j=total_energy_j,
        energy_per_token_mj=energy_per_token_mj,
        prefill_time_ms=prefill_time_ms,
        decode_time_ms=decode_time_ms,
        total_time_ms=total_time_ms,
        ttft_ms=ttft_ms,
        avg_power_w=avg_power,
        peak_power_w=peak_power,
        tokens_per_second=tps,
        used_counter=used_counter,
    )


def compare_modes(
    standard_runs: List[ChunkedPrefillRun],
    chunked_runs: List[ChunkedPrefillRun],
    profile: str,
    prompt_length: int,
) -> ChunkedPrefillComparison:
    """Compare standard vs chunked prefill for a given profile and length."""

    # Aggregate standard runs
    std_energy = statistics.mean([r.energy_j for r in standard_runs])
    std_ttft = statistics.mean([r.ttft_ms for r in standard_runs if r.ttft_ms])
    std_tps = statistics.mean([r.tokens_per_second for r in standard_runs])

    # Aggregate chunked runs
    chunked_energy = statistics.mean([r.energy_j for r in chunked_runs])
    chunked_ttft = statistics.mean([r.ttft_ms for r in chunked_runs if r.ttft_ms])
    chunked_tps = statistics.mean([r.tokens_per_second for r in chunked_runs])

    # Compute diffs
    energy_diff = ((chunked_energy - std_energy) / std_energy * 100) if std_energy > 0 else 0
    ttft_diff = ((chunked_ttft - std_ttft) / std_ttft * 100) if std_ttft > 0 else 0
    tps_diff = ((chunked_tps - std_tps) / std_tps * 100) if std_tps > 0 else 0

    return ChunkedPrefillComparison(
        profile=profile,
        prompt_length=prompt_length,
        standard_energy_j=std_energy,
        chunked_energy_j=chunked_energy,
        energy_diff_pct=energy_diff,
        standard_ttft_ms=std_ttft,
        chunked_ttft_ms=chunked_ttft,
        ttft_diff_pct=ttft_diff,
        standard_tps=std_tps,
        chunked_tps=chunked_tps,
        tps_diff_pct=tps_diff,
    )


def run_chunked_prefill_experiment(
    standard_client: VLLMClient,
    chunked_client: VLLMClient,
    actuator: DaemonActuator,
    prompt_lengths: List[int],
    profiles: List[str],
    runs_per_config: int,
    max_tokens: int,
) -> Tuple[List[ChunkedPrefillRun], List[ChunkedPrefillComparison]]:
    """Run complete chunked prefill experiment."""

    all_runs = []
    comparisons = []

    prompts = generate_prompts_by_length(prompt_lengths)

    total_configs = len(prompt_lengths) * len(profiles) * 2  # 2 modes
    current = 0

    for length in prompt_lengths:
        prompt = prompts[length]
        log.info(f"\n{'='*60}")
        log.info(f"Testing prompt length ~{length} tokens")
        log.info(f"{'='*60}")

        for profile in profiles:
            log.info(f"\nProfile: {profile}")

            # Standard mode runs
            standard_runs = []
            for i in range(runs_per_config):
                current += 1
                log.info(f"  Standard run {i+1}/{runs_per_config} [{current}/{total_configs * runs_per_config}]")
                try:
                    run = run_single_inference(
                        standard_client, actuator, prompt, max_tokens, profile, "standard"
                    )
                    standard_runs.append(run)
                    all_runs.append(run)
                    log.info(f"    Energy: {run.energy_j:.3f}J, TTFT: {run.ttft_ms:.1f}ms")
                except Exception as e:
                    log.warning(f"    Standard run failed: {e}")
                time.sleep(0.5)

            # Chunked mode runs
            chunked_runs = []
            for i in range(runs_per_config):
                current += 1
                log.info(f"  Chunked run {i+1}/{runs_per_config} [{current}/{total_configs * runs_per_config}]")
                try:
                    run = run_single_inference(
                        chunked_client, actuator, prompt, max_tokens, profile, "chunked"
                    )
                    chunked_runs.append(run)
                    all_runs.append(run)
                    log.info(f"    Energy: {run.energy_j:.3f}J, TTFT: {run.ttft_ms:.1f}ms")
                except Exception as e:
                    log.warning(f"    Chunked run failed: {e}")
                time.sleep(0.5)

            # Compare modes
            if standard_runs and chunked_runs:
                comparison = compare_modes(standard_runs, chunked_runs, profile, length)
                comparisons.append(comparison)
                log.info(f"  Comparison: Energy diff={comparison.energy_diff_pct:+.1f}%, "
                        f"TTFT diff={comparison.ttft_diff_pct:+.1f}%")

    return all_runs, comparisons


def generate_report(
    runs: List[ChunkedPrefillRun],
    comparisons: List[ChunkedPrefillComparison],
    output_dir: Path,
):
    """Generate analysis report."""

    # Save raw data
    runs_data = [asdict(r) for r in runs]
    with open(output_dir / "chunked_prefill_runs.json", "w") as f:
        json.dump(runs_data, f, indent=2)

    comparisons_data = [asdict(c) for c in comparisons]
    with open(output_dir / "chunked_prefill_comparisons.json", "w") as f:
        json.dump(comparisons_data, f, indent=2)

    # Generate markdown report
    report = [
        "# Chunked Prefill Energy Analysis",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Overview",
        "",
        "This report compares energy consumption between standard and chunked prefill modes",
        "across different prompt lengths and energy profiles.",
        "",
        "## Key Findings",
        "",
    ]

    # Summary by mode
    std_runs = [r for r in runs if r.mode == "standard"]
    chunked_runs = [r for r in runs if r.mode == "chunked"]

    if std_runs and chunked_runs:
        std_avg_energy = statistics.mean([r.energy_per_token_mj for r in std_runs])
        chunked_avg_energy = statistics.mean([r.energy_per_token_mj for r in chunked_runs])
        diff = ((chunked_avg_energy - std_avg_energy) / std_avg_energy * 100)

        report.extend([
            f"- **Standard prefill**: {std_avg_energy:.2f} mJ/token average",
            f"- **Chunked prefill**: {chunked_avg_energy:.2f} mJ/token average",
            f"- **Difference**: {diff:+.1f}% ({'more' if diff > 0 else 'less'} energy with chunked)",
            "",
        ])

    # Comparison table
    report.extend([
        "## Detailed Comparisons",
        "",
        "| Profile | Prompt Len | Std Energy (J) | Chunked Energy (J) | Diff | Std TTFT (ms) | Chunked TTFT (ms) | TTFT Diff |",
        "|---------|------------|----------------|--------------------|----- |---------------|-------------------|-----------|",
    ])

    for c in comparisons:
        report.append(
            f"| {c.profile} | {c.prompt_length} | {c.standard_energy_j:.3f} | "
            f"{c.chunked_energy_j:.3f} | {c.energy_diff_pct:+.1f}% | "
            f"{c.standard_ttft_ms:.1f} | {c.chunked_ttft_ms:.1f} | {c.ttft_diff_pct:+.1f}% |"
        )

    report.extend([
        "",
        "## Analysis",
        "",
        "### Energy Impact of Chunked Prefill",
        "",
    ])

    # Group by prompt length
    by_length = {}
    for c in comparisons:
        if c.prompt_length not in by_length:
            by_length[c.prompt_length] = []
        by_length[c.prompt_length].append(c)

    for length, comps in sorted(by_length.items()):
        avg_diff = statistics.mean([c.energy_diff_pct for c in comps])
        report.append(f"- **Prompt length ~{length}**: Average energy change {avg_diff:+.1f}%")

    report.extend([
        "",
        "### TTFT (Time to First Token) Impact",
        "",
    ])

    for length, comps in sorted(by_length.items()):
        avg_diff = statistics.mean([c.ttft_diff_pct for c in comps])
        report.append(f"- **Prompt length ~{length}**: Average TTFT change {avg_diff:+.1f}%")

    report.extend([
        "",
        "## Recommendations",
        "",
    ])

    # Recommendations based on data
    energy_changes = [c.energy_diff_pct for c in comparisons]
    ttft_changes = [c.ttft_diff_pct for c in comparisons]

    if energy_changes and ttft_changes:
        avg_energy_change = statistics.mean(energy_changes)
        avg_ttft_change = statistics.mean(ttft_changes)

        if avg_energy_change < -5:
            report.append("- **Chunked prefill saves energy** - consider enabling for long prompts")
        elif avg_energy_change > 5:
            report.append("- **Standard prefill is more energy efficient** - avoid chunked for energy savings")
        else:
            report.append("- **Energy difference is minimal** - choose based on other factors")

        if avg_ttft_change > 20:
            report.append("- **Chunked prefill increases TTFT significantly** - may impact user experience")
        elif avg_ttft_change < -10:
            report.append("- **Chunked prefill improves TTFT** - better for interactive use")

    report.extend([
        "",
        "## LaTeX Table",
        "",
        "```latex",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Chunked vs Standard Prefill Energy Comparison}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Profile & Length & Std (J) & Chunked (J) & Energy $\\Delta$ & TTFT $\\Delta$ \\\\",
        "\\midrule",
    ])

    for c in comparisons:
        report.append(
            f"{c.profile} & {c.prompt_length} & {c.standard_energy_j:.3f} & "
            f"{c.chunked_energy_j:.3f} & {c.energy_diff_pct:+.1f}\\% & {c.ttft_diff_pct:+.1f}\\% \\\\"
        )

    report.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "```",
        "",
    ])

    report_path = output_dir / "chunked_prefill_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    log.info(f"Report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Test vLLM chunked prefill energy characteristics")
    parser.add_argument("--standard-host", default="localhost", help="Standard vLLM server host")
    parser.add_argument("--standard-port", type=int, default=8000, help="Standard vLLM server port")
    parser.add_argument("--chunked-host", default="localhost", help="Chunked vLLM server host")
    parser.add_argument("--chunked-port", type=int, default=8001, help="Chunked vLLM server port")
    parser.add_argument("--actuator-host", default="localhost", help="Actuator daemon host")
    parser.add_argument("--actuator-port", type=int, default=9877, help="Actuator daemon port")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 256, 512, 1024],
                       help="Prompt lengths to test")
    parser.add_argument("--profiles", nargs="+", default=["eco", "balanced", "performance"],
                       help="Energy profiles to test")
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--output-dir", type=str, default="results/z99_chunked_prefill",
                       help="Output directory")
    parser.add_argument("--single-server", action="store_true",
                       help="Use single server for both modes (skip comparison)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to servers
    standard_client = VLLMClient(args.standard_host, args.standard_port)

    if args.single_server:
        log.info("Single server mode - testing one server only")
        chunked_client = standard_client
    else:
        chunked_client = VLLMClient(args.chunked_host, args.chunked_port)

    # Check server health
    if not standard_client.health_check():
        log.error(f"Standard vLLM server not available at {args.standard_host}:{args.standard_port}")
        log.info("Start server with: vllm serve <model> --port 8000")
        sys.exit(1)

    if not args.single_server and not chunked_client.health_check():
        log.error(f"Chunked vLLM server not available at {args.chunked_host}:{args.chunked_port}")
        log.info("Start server with: vllm serve <model> --port 8001 --enable-chunked-prefill")
        sys.exit(1)

    # Connect to actuator
    try:
        actuator = DaemonActuator(args.actuator_host, args.actuator_port)
        health = actuator.get_health()
        log.info(f"Connected to actuator: GPU at {health.get('temp_c', 0):.0f}C, {health.get('power_w', 0):.0f}W")
    except Exception as e:
        log.error(f"Cannot connect to actuator daemon: {e}")
        sys.exit(1)

    log.info(f"\nChunked Prefill Energy Test")
    log.info(f"Prompt lengths: {args.prompt_lengths}")
    log.info(f"Profiles: {args.profiles}")
    log.info(f"Runs per config: {args.runs}")
    log.info(f"Output: {output_dir}")

    # Run experiment
    runs, comparisons = run_chunked_prefill_experiment(
        standard_client=standard_client,
        chunked_client=chunked_client,
        actuator=actuator,
        prompt_lengths=args.prompt_lengths,
        profiles=args.profiles,
        runs_per_config=args.runs,
        max_tokens=args.max_tokens,
    )

    # Generate report
    report_path = generate_report(runs, comparisons, output_dir)

    # Print summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Total runs: {len(runs)}")
    log.info(f"Comparisons: {len(comparisons)}")

    if comparisons:
        avg_energy_diff = statistics.mean([c.energy_diff_pct for c in comparisons])
        avg_ttft_diff = statistics.mean([c.ttft_diff_pct for c in comparisons])
        log.info(f"Average energy diff (chunked vs standard): {avg_energy_diff:+.1f}%")
        log.info(f"Average TTFT diff (chunked vs standard): {avg_ttft_diff:+.1f}%")

    log.info(f"\nResults saved to: {output_dir}")
    log.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
