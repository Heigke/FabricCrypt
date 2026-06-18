#!/usr/bin/env python3
"""
z102_comprehensive_validation.py - Comprehensive FEEL Validation

Runs 3 defensible workloads with confidence intervals:
1. Short prompts / long decode (chatbot-style)
2. Long prompts / short decode (summarization-style)
3. Bursty Poisson arrivals (variable load)

Reports:
- Energy per token with 95% CI
- SLO compliance rate
- Quality metrics (answer consistency)
- Comparison: Fixed perf vs Fixed eco vs FEEL adaptive

This is the paper-grade validation script.

Usage:
    python z102_comprehensive_validation.py --daemon-host localhost --daemon-port 9877 --vllm-port 8000

Author: FEEL Research Team
Date: 2026-01-21
"""

import argparse
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading

import requests
import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.organism.hypothalamus import Hypothalamus, HypothalamusConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class InferenceResult:
    """Result from a single inference."""
    workload: str
    condition: str  # "baseline_perf", "baseline_eco", "feel"
    prompt_tokens: int
    completion_tokens: int
    energy_j: float
    latency_ms: float
    ttft_ms: float
    avg_power_w: float  # Derived from ΔE/Δt
    profile: str
    answer_hash: str  # For consistency checking


@dataclass
class WorkloadResults:
    """Aggregated results for a workload."""
    workload: str
    condition: str
    n_runs: int
    # Energy metrics
    energy_per_token_mj_mean: float
    energy_per_token_mj_std: float
    energy_per_token_mj_ci95: Tuple[float, float]
    # Latency metrics
    latency_ms_mean: float
    latency_ms_p95: float
    ttft_ms_mean: float
    ttft_ms_p95: float
    # Power
    avg_power_w_mean: float
    # Quality
    answer_consistency: float  # Fraction of identical answers
    # SLO
    slo_compliance_rate: float
    # Cost
    cost_per_m_tokens: float


@dataclass
class ValidationReport:
    """Full validation report."""
    timestamp: str
    config: Dict[str, Any]
    workloads: List[WorkloadResults]
    summary: Dict[str, Any]


# =============================================================================
# Workload Generators
# =============================================================================

def generate_short_prompt_long_decode() -> List[Tuple[str, int]]:
    """Workload 1: Short prompts with long decode (chatbot-style)."""
    prompts = [
        "Explain quantum computing.",
        "What is machine learning?",
        "Describe photosynthesis.",
        "How does GPS work?",
        "What causes earthquakes?",
        "Explain blockchain technology.",
        "How do vaccines work?",
        "What is climate change?",
        "Describe the water cycle.",
        "How does WiFi work?",
    ]
    # Short prompt (~10-20 tokens), long decode (128 tokens)
    return [(p, 128) for p in prompts]


def generate_long_prompt_short_decode() -> List[Tuple[str, int]]:
    """Workload 2: Long prompts with short decode (summarization-style)."""
    base_text = (
        "The field of artificial intelligence has seen remarkable progress in recent years. "
        "Machine learning algorithms, particularly deep neural networks, have achieved "
        "state-of-the-art results in computer vision, natural language processing, and "
        "speech recognition. Large language models like GPT and BERT have transformed "
        "how we interact with computers, enabling more natural conversations and better "
        "understanding of human intent. These advances have practical applications in "
        "healthcare, finance, education, and many other domains. "
    )

    prompts = []
    for i in range(10):
        # Repeat base text to create long prompt (~200-400 tokens)
        long_text = (base_text * 3) + f"\n\nSummarize the above in one sentence (variation {i}):"
        prompts.append((long_text, 32))  # Short decode

    return prompts


def generate_bursty_arrivals(n_requests: int, avg_rate: float = 2.0) -> List[float]:
    """Generate Poisson arrival times with occasional bursts."""
    arrivals = []
    t = 0.0

    for i in range(n_requests):
        # Occasionally burst (5x faster rate)
        if random.random() < 0.1:  # 10% chance of burst
            inter_arrival = random.expovariate(avg_rate * 5)
        else:
            inter_arrival = random.expovariate(avg_rate)

        t += inter_arrival
        arrivals.append(t)

    return arrivals


# =============================================================================
# Inference Client
# =============================================================================

class ValidationClient:
    """Client for running validation tests."""

    def __init__(self, daemon_host: str, daemon_port: int,
                 vllm_host: str, vllm_port: int):
        self.daemon_url = f"http://{daemon_host}:{daemon_port}"
        self.vllm_url = f"http://{vllm_host}:{vllm_port}"

        self._prev_energy = 0
        self._prev_time = 0.0

    def get_energy(self) -> int:
        """Get cumulative energy counter (mJ)."""
        try:
            resp = requests.get(f"{self.daemon_url}/energy", timeout=2)
            return resp.json().get('energy_mj', 0)
        except:
            return 0

    def get_telemetry(self) -> Dict:
        """Get current telemetry."""
        try:
            resp = requests.get(f"{self.daemon_url}/telemetry", timeout=2)
            return resp.json()
        except:
            return {}

    def set_profile(self, profile: str) -> bool:
        """Set energy profile."""
        try:
            resp = requests.post(
                f"{self.daemon_url}/profile",
                json={"profile": profile},
                timeout=2
            )
            return resp.status_code == 200
        except:
            return False

    def generate(self, prompt: str, max_tokens: int,
                 temperature: float = 0.7) -> Tuple[str, int, int, float]:
        """
        Generate completion.

        Returns: (text, prompt_tokens, completion_tokens, ttft_ms)
        """
        start = time.perf_counter()
        ttft_ms = None

        try:
            resp = requests.post(
                f"{self.vllm_url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
                stream=True,
                timeout=60,
            )

            text_chunks = []
            prompt_tokens = 0
            completion_tokens = 0

            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data = line_str[6:]
                    if data.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000
                        if 'choices' in chunk:
                            text_chunks.append(chunk['choices'][0].get('text', ''))
                        if 'usage' in chunk:
                            prompt_tokens = chunk['usage'].get('prompt_tokens', 0)
                            completion_tokens = chunk['usage'].get('completion_tokens', 0)
                    except:
                        pass

            text = ''.join(text_chunks)
            return text, prompt_tokens, completion_tokens, ttft_ms or 0

        except Exception as e:
            log.error(f"Generation failed: {e}")
            return "", 0, 0, 0

    def run_inference(self, prompt: str, max_tokens: int,
                      workload: str, condition: str) -> Optional[InferenceResult]:
        """Run single inference with energy measurement."""

        # Get initial energy
        e_start = self.get_energy()
        t_start = time.perf_counter()

        # Get profile before inference
        telem = self.get_telemetry()
        profile = telem.get('profile', 'unknown')

        # Run inference
        text, prompt_tokens, completion_tokens, ttft_ms = self.generate(
            prompt, max_tokens
        )

        # Get final energy
        t_end = time.perf_counter()
        e_end = self.get_energy()

        if completion_tokens == 0:
            return None

        # Calculate metrics
        energy_delta_mj = e_end - e_start
        energy_j = energy_delta_mj / 1000
        time_delta_s = t_end - t_start
        latency_ms = time_delta_s * 1000

        # CRITICAL: Derive avg power from ΔE/Δt
        avg_power_w = (energy_delta_mj / 1000) / max(time_delta_s, 0.001)

        # Answer hash for consistency checking
        import hashlib
        answer_hash = hashlib.md5(text.encode()).hexdigest()[:8]

        return InferenceResult(
            workload=workload,
            condition=condition,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            energy_j=energy_j,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            avg_power_w=avg_power_w,
            profile=profile,
            answer_hash=answer_hash,
        )


# =============================================================================
# Validation Runner
# =============================================================================

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


def aggregate_results(results: List[InferenceResult],
                     workload: str, condition: str,
                     ttft_slo_ms: float = 500,
                     tpot_slo_ms: float = 50) -> WorkloadResults:
    """Aggregate inference results into workload metrics."""

    if not results:
        return None

    # Energy per token
    e_per_tok = [
        (r.energy_j * 1000) / r.completion_tokens  # mJ/token
        for r in results if r.completion_tokens > 0
    ]

    # Latencies
    latencies = [r.latency_ms for r in results]
    ttfts = [r.ttft_ms for r in results]

    # Powers
    powers = [r.avg_power_w for r in results]

    # Answer consistency (same hash = consistent)
    hashes = [r.answer_hash for r in results]
    most_common = max(set(hashes), key=hashes.count)
    consistency = hashes.count(most_common) / len(hashes)

    # SLO compliance
    slo_pass = sum(
        1 for r in results
        if r.ttft_ms < ttft_slo_ms and
           (r.latency_ms / r.completion_tokens) < tpot_slo_ms
    )
    slo_compliance = slo_pass / len(results)

    # Cost calculation
    mean_e_per_tok = statistics.mean(e_per_tok) if e_per_tok else 0
    kwh_per_m = (mean_e_per_tok / 1000 / 3600) * 1e6
    cost_per_m = kwh_per_m * 0.12

    return WorkloadResults(
        workload=workload,
        condition=condition,
        n_runs=len(results),
        energy_per_token_mj_mean=mean_e_per_tok,
        energy_per_token_mj_std=statistics.stdev(e_per_tok) if len(e_per_tok) > 1 else 0,
        energy_per_token_mj_ci95=compute_ci95(e_per_tok),
        latency_ms_mean=statistics.mean(latencies),
        latency_ms_p95=np.percentile(latencies, 95),
        ttft_ms_mean=statistics.mean(ttfts),
        ttft_ms_p95=np.percentile(ttfts, 95),
        avg_power_w_mean=statistics.mean(powers),
        answer_consistency=consistency,
        slo_compliance_rate=slo_compliance,
        cost_per_m_tokens=cost_per_m,
    )


def run_workload(
    client: ValidationClient,
    prompts: List[Tuple[str, int]],
    workload_name: str,
    condition: str,
    hypothalamus: Optional[Hypothalamus] = None,
    n_repetitions: int = 3,
    inter_request_delay: float = 0.5,
) -> List[InferenceResult]:
    """Run a workload under a condition."""

    results = []
    total = len(prompts) * n_repetitions

    for rep in range(n_repetitions):
        for i, (prompt, max_tokens) in enumerate(prompts):
            idx = rep * len(prompts) + i + 1
            log.info(f"  [{condition}] {workload_name} {idx}/{total}")

            result = client.run_inference(
                prompt, max_tokens, workload_name, condition
            )

            if result:
                results.append(result)
                log.info(f"    {result.completion_tokens} tok, {result.energy_j:.2f}J, "
                        f"{result.avg_power_w:.0f}W")

            time.sleep(inter_request_delay)

    return results


def run_bursty_workload(
    client: ValidationClient,
    prompts: List[Tuple[str, int]],
    arrivals: List[float],
    workload_name: str,
    condition: str,
) -> List[InferenceResult]:
    """Run workload with bursty arrival pattern."""

    results = []
    start_time = time.time()

    for i, arrival_time in enumerate(arrivals):
        # Wait until arrival time
        now = time.time() - start_time
        if arrival_time > now:
            time.sleep(arrival_time - now)

        prompt, max_tokens = prompts[i % len(prompts)]
        log.info(f"  [{condition}] {workload_name} {i+1}/{len(arrivals)} (t={arrival_time:.1f}s)")

        result = client.run_inference(
            prompt, max_tokens, workload_name, condition
        )

        if result:
            results.append(result)

    return results


# =============================================================================
# Report Generation
# =============================================================================

def generate_latex_table(workload_results: List[WorkloadResults]) -> str:
    """Generate LaTeX table from results."""

    latex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{FEEL Energy Validation Results}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Workload & Condition & mJ/tok & 95\\% CI & SLO \\% & \\$/1M & Consistency \\\\",
        "\\midrule",
    ]

    for r in workload_results:
        ci = f"[{r.energy_per_token_mj_ci95[0]:.1f}, {r.energy_per_token_mj_ci95[1]:.1f}]"
        latex.append(
            f"{r.workload} & {r.condition} & {r.energy_per_token_mj_mean:.1f} & "
            f"{ci} & {r.slo_compliance_rate*100:.0f}\\% & "
            f"\\${r.cost_per_m_tokens:.4f} & {r.answer_consistency*100:.0f}\\% \\\\"
        )

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(latex)


def generate_markdown_report(report: ValidationReport) -> str:
    """Generate markdown report."""

    md = [
        "# FEEL Comprehensive Validation Report",
        "",
        f"Generated: {report.timestamp}",
        "",
        "## Configuration",
        "",
        f"- TTFT SLO: {report.config['ttft_slo_ms']}ms",
        f"- TPOT SLO: {report.config['tpot_slo_ms']}ms",
        f"- Repetitions: {report.config['n_repetitions']}",
        "",
        "## Results by Workload",
        "",
    ]

    # Group by workload
    by_workload = {}
    for r in report.workloads:
        if r.workload not in by_workload:
            by_workload[r.workload] = []
        by_workload[r.workload].append(r)

    for workload, results in by_workload.items():
        md.extend([
            f"### {workload}",
            "",
            "| Condition | mJ/tok | 95% CI | SLO % | $/1M | Power (W) |",
            "|-----------|--------|--------|-------|------|-----------|",
        ])

        for r in results:
            ci = f"[{r.energy_per_token_mj_ci95[0]:.1f}, {r.energy_per_token_mj_ci95[1]:.1f}]"
            md.append(
                f"| {r.condition} | {r.energy_per_token_mj_mean:.1f} | {ci} | "
                f"{r.slo_compliance_rate*100:.0f}% | ${r.cost_per_m_tokens:.4f} | "
                f"{r.avg_power_w_mean:.0f} |"
            )

        md.append("")

    # Summary
    md.extend([
        "## Summary",
        "",
    ])

    # Calculate savings
    if report.summary.get('energy_savings_pct'):
        md.append(f"- **Energy savings (FEEL vs baseline_perf):** "
                 f"{report.summary['energy_savings_pct']:.1f}%")
    if report.summary.get('slo_improvement'):
        md.append(f"- **SLO compliance improvement:** "
                 f"{report.summary['slo_improvement']:.1f}%")

    md.extend([
        "",
        "## LaTeX Table",
        "",
        "```latex",
        generate_latex_table(report.workloads),
        "```",
    ])

    return "\n".join(md)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FEEL Comprehensive Validation")
    parser.add_argument("--daemon-host", default="localhost")
    parser.add_argument("--daemon-port", type=int, default=9877)
    parser.add_argument("--vllm-host", default="localhost")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--n-repetitions", type=int, default=3)
    parser.add_argument("--ttft-slo-ms", type=float, default=500)
    parser.add_argument("--tpot-slo-ms", type=float, default=50)
    parser.add_argument("--output-dir", default="results/z102_validation")
    parser.add_argument("--skip-feel", action="store_true",
                       help="Skip FEEL adaptive condition (baseline only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create client
    client = ValidationClient(
        args.daemon_host, args.daemon_port,
        args.vllm_host, args.vllm_port
    )

    # Verify connectivity
    telem = client.get_telemetry()
    if not telem:
        log.error("Cannot connect to daemon")
        sys.exit(1)
    log.info(f"Connected to daemon: {telem.get('power_watts', 0):.0f}W, "
            f"{telem.get('temp_c', 0):.0f}°C")

    # Generate workloads
    workload1 = generate_short_prompt_long_decode()
    workload2 = generate_long_prompt_short_decode()
    bursty_arrivals = generate_bursty_arrivals(20, avg_rate=2.0)

    conditions = ["baseline_perf", "baseline_eco"]
    if not args.skip_feel:
        conditions.append("feel")

    all_results = []

    # Optional: Create Hypothalamus for FEEL condition
    hypothalamus = None
    if "feel" in conditions:
        config = HypothalamusConfig(
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
        )
        hypothalamus = Hypothalamus(config)
        hypothalamus.add_node(
            "local",
            args.daemon_host, args.daemon_port,
            args.vllm_host, args.vllm_port
        )

    # Run validation
    for condition in conditions:
        log.info(f"\n{'='*60}")
        log.info(f"CONDITION: {condition.upper()}")
        log.info(f"{'='*60}")

        # Set up condition
        if condition == "baseline_perf":
            client.set_profile("performance")
        elif condition == "baseline_eco":
            client.set_profile("eco")
        elif condition == "feel" and hypothalamus:
            hypothalamus.start()

        time.sleep(1)  # Let profile settle

        # Workload 1: Short prompt / long decode
        log.info("\n--- Workload 1: Short Prompt / Long Decode ---")
        results1 = run_workload(
            client, workload1, "short_prompt_long_decode",
            condition, hypothalamus, args.n_repetitions
        )
        if results1:
            agg1 = aggregate_results(results1, "short_prompt_long_decode", condition,
                                    args.ttft_slo_ms, args.tpot_slo_ms)
            if agg1:
                all_results.append(agg1)

        # Workload 2: Long prompt / short decode
        log.info("\n--- Workload 2: Long Prompt / Short Decode ---")
        results2 = run_workload(
            client, workload2, "long_prompt_short_decode",
            condition, hypothalamus, args.n_repetitions
        )
        if results2:
            agg2 = aggregate_results(results2, "long_prompt_short_decode", condition,
                                    args.ttft_slo_ms, args.tpot_slo_ms)
            if agg2:
                all_results.append(agg2)

        # Workload 3: Bursty arrivals
        log.info("\n--- Workload 3: Bursty Arrivals ---")
        results3 = run_bursty_workload(
            client, workload1, bursty_arrivals,
            "bursty_arrivals", condition
        )
        if results3:
            agg3 = aggregate_results(results3, "bursty_arrivals", condition,
                                    args.ttft_slo_ms, args.tpot_slo_ms)
            if agg3:
                all_results.append(agg3)

        # Stop hypothalamus if running
        if condition == "feel" and hypothalamus:
            hypothalamus.stop()

    # Compute summary
    summary = {}
    perf_results = [r for r in all_results if r.condition == "baseline_perf"]
    eco_results = [r for r in all_results if r.condition == "baseline_eco"]
    feel_results = [r for r in all_results if r.condition == "feel"]

    if perf_results and eco_results:
        perf_energy = statistics.mean([r.energy_per_token_mj_mean for r in perf_results])
        eco_energy = statistics.mean([r.energy_per_token_mj_mean for r in eco_results])
        summary['eco_vs_perf_savings_pct'] = ((perf_energy - eco_energy) / perf_energy) * 100

    if perf_results and feel_results:
        perf_energy = statistics.mean([r.energy_per_token_mj_mean for r in perf_results])
        feel_energy = statistics.mean([r.energy_per_token_mj_mean for r in feel_results])
        summary['energy_savings_pct'] = ((perf_energy - feel_energy) / perf_energy) * 100

        perf_slo = statistics.mean([r.slo_compliance_rate for r in perf_results])
        feel_slo = statistics.mean([r.slo_compliance_rate for r in feel_results])
        summary['slo_improvement'] = (feel_slo - perf_slo) * 100

    # Generate report
    report = ValidationReport(
        timestamp=datetime.now().isoformat(),
        config={
            'daemon_host': args.daemon_host,
            'daemon_port': args.daemon_port,
            'vllm_host': args.vllm_host,
            'vllm_port': args.vllm_port,
            'n_repetitions': args.n_repetitions,
            'ttft_slo_ms': args.ttft_slo_ms,
            'tpot_slo_ms': args.tpot_slo_ms,
        },
        workloads=all_results,
        summary=summary,
    )

    # Save results
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump({
            'timestamp': report.timestamp,
            'config': report.config,
            'workloads': [asdict(w) for w in report.workloads],
            'summary': report.summary,
        }, f, indent=2)

    # Generate markdown report
    md_report = generate_markdown_report(report)
    with open(output_dir / "validation_report.md", "w") as f:
        f.write(md_report)

    # Print summary
    log.info(f"\n{'='*60}")
    log.info("VALIDATION COMPLETE")
    log.info(f"{'='*60}")

    for r in all_results:
        ci = f"[{r.energy_per_token_mj_ci95[0]:.1f}, {r.energy_per_token_mj_ci95[1]:.1f}]"
        log.info(f"{r.workload} | {r.condition}: {r.energy_per_token_mj_mean:.1f} mJ/tok {ci}, "
                f"SLO {r.slo_compliance_rate*100:.0f}%")

    if summary:
        log.info(f"\nSummary:")
        for k, v in summary.items():
            log.info(f"  {k}: {v:.1f}%")

    log.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
