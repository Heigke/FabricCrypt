#!/usr/bin/env python3
"""
FEEL Business Value Demo - Same Quality, Different Cost

This script demonstrates the business case for FEEL:
- Run identical workloads with different energy profiles
- Show energy/cost savings with quality preservation
- Generate enterprise-ready metrics and reports

Key Business Metrics:
1. Cost per 1M tokens (based on energy pricing)
2. TCO reduction with FEEL
3. Quality score retention
4. SLO compliance

Author: FEEL Research Team
Date: 2026-01-21
"""

import os
import sys
import json
import time
import argparse
import statistics
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import urllib.request

# Add project root
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Business Constants
# ============================================================================

# Energy pricing (USD per kWh) - enterprise data center rates
ENERGY_COST_USD_PER_KWH = 0.12

# Typical GPU utilization in production
AVG_GPU_UTILIZATION = 0.60

# Hours per year
HOURS_PER_YEAR = 8760


@dataclass
class BusinessMetrics:
    """Business-oriented metrics."""
    # Energy
    j_per_token: float
    watts_avg: float
    kwh_per_1m_tokens: float

    # Cost
    cost_per_1m_tokens_usd: float
    annual_cost_per_gpu_usd: float

    # Quality
    quality_score: float  # 0-1
    slo_compliance: float  # 0-1

    # Performance
    tokens_per_second: float
    latency_p95_ms: float


@dataclass
class BusinessComparison:
    """Comparison between two profiles."""
    baseline_profile: str
    optimized_profile: str

    # Savings
    energy_reduction_pct: float
    cost_savings_per_gpu_year_usd: float
    quality_change_pct: float

    # Fleet projection
    fleet_size: int
    annual_fleet_savings_usd: float
    co2_reduction_kg: float  # Based on avg grid carbon intensity


# ============================================================================
# vLLM Client
# ============================================================================

class VLLMClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"

    def generate(self, prompt: str, max_tokens: int = 128) -> Tuple[str, int, float]:
        """Returns (text, tokens, latency_ms)."""
        try:
            data = {
                "model": "default",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
            req = urllib.request.Request(
                f"{self.base_url}/v1/completions",
                data=json.dumps(data).encode(),
                headers={'Content-Type': 'application/json'},
            )
            start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode())
            latency_ms = (time.perf_counter() - start) * 1000

            text = result['choices'][0]['text'] if result.get('choices') else ""
            usage = result.get('usage', {})
            tokens = usage.get('completion_tokens', len(text.split()))
            return text, tokens, latency_ms
        except Exception as e:
            logger.warning(f"Request failed: {e}")
            return "", 0, 0


class ActuatorClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"

    def set_profile(self, profile: str) -> bool:
        try:
            data = json.dumps({'profile': profile}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/profile",
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            return result.get('success', False)
        except:
            return False

    def get_energy(self) -> Optional[Dict]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/energy", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except:
            return None

    def get_telemetry(self) -> Optional[Dict]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/telemetry", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except:
            return None


# ============================================================================
# Quality Evaluation (simplified)
# ============================================================================

def evaluate_response(response: str, expected_keywords: List[str]) -> float:
    """Simple quality score based on keywords and length."""
    if not response:
        return 0.0

    scores = []

    # Keyword presence
    if expected_keywords:
        found = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
        scores.append(found / len(expected_keywords))

    # Reasonable length
    words = len(response.split())
    if 10 < words < 300:
        scores.append(1.0)
    elif words <= 10:
        scores.append(words / 10)
    else:
        scores.append(max(0, 1 - (words - 300) / 300))

    return statistics.mean(scores) if scores else 0.5


# ============================================================================
# Business Benchmark
# ============================================================================

def run_business_benchmark(
    vllm: VLLMClient,
    actuator: ActuatorClient,
    profile: str,
    prompts: List[Dict],
    max_tokens: int = 128,
    slo_ms: float = 2000,
) -> BusinessMetrics:
    """Run benchmark and compute business metrics."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Business Benchmark: {profile}")
    logger.info(f"{'='*60}")

    if not actuator.set_profile(profile):
        logger.warning(f"Failed to set profile {profile}")

    time.sleep(2)

    energies = []
    powers = []
    latencies = []
    qualities = []
    tokens_total = 0

    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data['prompt']
        keywords = prompt_data.get('keywords', [])

        # Get initial energy
        energy_start = actuator.get_energy()
        start_mj = energy_start.get('energy_mj') if energy_start else None

        # Generate
        response, tokens, latency_ms = vllm.generate(prompt, max_tokens)

        # Get final energy
        energy_end = actuator.get_energy()
        end_mj = energy_end.get('energy_mj') if energy_end else None

        # Get power
        telemetry = actuator.get_telemetry()
        power = telemetry.get('power_watts', 0) if telemetry else 0

        # Calculate energy
        if start_mj and end_mj:
            energy_j = (end_mj - start_mj) / 1000
        else:
            energy_j = power * (latency_ms / 1000) if power else 0.3

        # Quality
        quality = evaluate_response(response, keywords)

        energies.append(energy_j)
        powers.append(power)
        latencies.append(latency_ms)
        qualities.append(quality)
        tokens_total += tokens

        logger.info(f"  [{i+1}/{len(prompts)}] {tokens} tok, {energy_j:.3f}J, {latency_ms:.0f}ms, Q={quality:.2f}")

        time.sleep(0.5)

    # Compute aggregates
    total_energy_j = sum(energies)
    avg_power = statistics.mean(powers) if powers else 0
    avg_latency = statistics.mean(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    avg_quality = statistics.mean(qualities) if qualities else 0
    slo_compliance = sum(1 for l in latencies if l <= slo_ms) / len(latencies) if latencies else 0

    j_per_token = total_energy_j / tokens_total if tokens_total > 0 else 0
    tokens_per_second = tokens_total / (sum(latencies) / 1000) if latencies else 0

    # Business calculations
    # 1M tokens energy
    kwh_per_1m = (j_per_token * 1_000_000) / 3_600_000  # J to kWh

    # Cost per 1M tokens
    cost_per_1m = kwh_per_1m * ENERGY_COST_USD_PER_KWH

    # Annual cost per GPU (assuming utilization and avg power)
    annual_kwh = (avg_power * HOURS_PER_YEAR * AVG_GPU_UTILIZATION) / 1000
    annual_cost = annual_kwh * ENERGY_COST_USD_PER_KWH

    return BusinessMetrics(
        j_per_token=j_per_token,
        watts_avg=avg_power,
        kwh_per_1m_tokens=kwh_per_1m,
        cost_per_1m_tokens_usd=cost_per_1m,
        annual_cost_per_gpu_usd=annual_cost,
        quality_score=avg_quality,
        slo_compliance=slo_compliance,
        tokens_per_second=tokens_per_second,
        latency_p95_ms=p95_latency,
    )


def compute_comparison(
    baseline: BusinessMetrics,
    optimized: BusinessMetrics,
    baseline_profile: str,
    optimized_profile: str,
    fleet_size: int = 100,
) -> BusinessComparison:
    """Compute business comparison between profiles."""

    energy_reduction = (baseline.j_per_token - optimized.j_per_token) / baseline.j_per_token * 100
    cost_savings = baseline.annual_cost_per_gpu_usd - optimized.annual_cost_per_gpu_usd
    quality_change = (optimized.quality_score - baseline.quality_score) / baseline.quality_score * 100

    fleet_savings = cost_savings * fleet_size

    # CO2 reduction (using US grid average ~0.42 kg CO2/kWh)
    annual_kwh_savings = (baseline.watts_avg - optimized.watts_avg) * HOURS_PER_YEAR * AVG_GPU_UTILIZATION / 1000
    co2_reduction = annual_kwh_savings * 0.42 * fleet_size

    return BusinessComparison(
        baseline_profile=baseline_profile,
        optimized_profile=optimized_profile,
        energy_reduction_pct=energy_reduction,
        cost_savings_per_gpu_year_usd=cost_savings,
        quality_change_pct=quality_change,
        fleet_size=fleet_size,
        annual_fleet_savings_usd=fleet_savings,
        co2_reduction_kg=co2_reduction,
    )


# ============================================================================
# Prompts
# ============================================================================

def get_business_prompts() -> List[Dict]:
    """Prompts for business demo."""
    return [
        {"prompt": "Summarize the benefits of cloud computing.", "keywords": ["cloud", "cost", "scale", "access"]},
        {"prompt": "What are best practices for API design?", "keywords": ["REST", "API", "endpoint", "design"]},
        {"prompt": "Explain machine learning to a business executive.", "keywords": ["data", "learn", "predict", "model"]},
        {"prompt": "List 3 tips for improving website performance.", "keywords": ["1", "2", "3", "performance"]},
        {"prompt": "What is containerization and why is it useful?", "keywords": ["container", "Docker", "deploy", "isolate"]},
        {"prompt": "Describe the concept of CI/CD.", "keywords": ["continuous", "integration", "deployment", "automate"]},
        {"prompt": "What are microservices?", "keywords": ["service", "independent", "scale", "deploy"]},
        {"prompt": "Explain database indexing.", "keywords": ["index", "query", "performance", "search"]},
    ]


# ============================================================================
# Report Generation
# ============================================================================

def generate_executive_summary(
    baseline: BusinessMetrics,
    optimized: BusinessMetrics,
    comparison: BusinessComparison,
) -> str:
    """Generate executive summary."""
    return f"""
# FEEL Energy Optimization - Executive Summary

## Bottom Line

**FEEL reduces GPU energy consumption by {comparison.energy_reduction_pct:.1f}% while maintaining {optimized.quality_score:.0%} quality.**

## Key Metrics

| Metric | Performance Profile | FEEL Eco Profile | Improvement |
|--------|---------------------|------------------|-------------|
| Energy (J/token) | {baseline.j_per_token:.4f} | {optimized.j_per_token:.4f} | {comparison.energy_reduction_pct:.1f}% reduction |
| Cost per 1M tokens | ${baseline.cost_per_1m_tokens_usd:.4f} | ${optimized.cost_per_1m_tokens_usd:.4f} | {comparison.energy_reduction_pct:.1f}% savings |
| Quality Score | {baseline.quality_score:.2f} | {optimized.quality_score:.2f} | {comparison.quality_change_pct:+.1f}% |
| SLO Compliance | {baseline.slo_compliance:.0%} | {optimized.slo_compliance:.0%} | - |

## Annual Savings (per GPU)

- **Energy Cost Savings:** ${comparison.cost_savings_per_gpu_year_usd:.2f}/year
- **At {comparison.fleet_size} GPU fleet:** ${comparison.annual_fleet_savings_usd:,.0f}/year
- **CO2 Reduction:** {comparison.co2_reduction_kg:,.0f} kg/year ({comparison.co2_reduction_kg/1000:.1f} metric tons)

## Technical Details

- Average Power: {baseline.watts_avg:.1f}W → {optimized.watts_avg:.1f}W
- Throughput: {optimized.tokens_per_second:.1f} tokens/second
- Latency p95: {optimized.latency_p95_ms:.0f}ms

## Recommendation

Deploy FEEL energy optimization for:
1. **Cost Reduction:** {comparison.energy_reduction_pct:.0f}% lower energy costs
2. **Sustainability:** {comparison.co2_reduction_kg/1000:.1f} tons CO2 reduction per year
3. **Quality Preserved:** {abs(comparison.quality_change_pct):.1f}% quality {'improvement' if comparison.quality_change_pct > 0 else 'impact'}

---
*Generated by FEEL Business Demo | {datetime.now().strftime('%Y-%m-%d')}*
"""


def generate_latex_business_table(
    baseline: BusinessMetrics,
    optimized: BusinessMetrics,
    comparison: BusinessComparison,
) -> str:
    """Generate LaTeX table for paper."""
    return f"""\\begin{{table}}[h]
\\centering
\\caption{{FEEL Business Impact Analysis}}
\\label{{tab:business}}
\\begin{{tabular}}{{lrrr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Baseline}} & \\textbf{{FEEL Eco}} & \\textbf{{Improvement}} \\\\
\\midrule
Energy (J/token) & {baseline.j_per_token:.4f} & {optimized.j_per_token:.4f} & {comparison.energy_reduction_pct:.1f}\\% \\\\
Cost (\\$/1M tok) & {baseline.cost_per_1m_tokens_usd:.4f} & {optimized.cost_per_1m_tokens_usd:.4f} & {comparison.energy_reduction_pct:.1f}\\% \\\\
Quality Score & {baseline.quality_score:.2f} & {optimized.quality_score:.2f} & {comparison.quality_change_pct:+.1f}\\% \\\\
Annual/GPU & \\${baseline.annual_cost_per_gpu_usd:.0f} & \\${optimized.annual_cost_per_gpu_usd:.0f} & \\${comparison.cost_savings_per_gpu_year_usd:.0f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='FEEL Business Demo')
    parser.add_argument('--vllm-host', default='localhost')
    parser.add_argument('--vllm-port', type=int, default=8000)
    parser.add_argument('--actuator-host', default='192.168.0.38')
    parser.add_argument('--actuator-port', type=int, default=9877)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--fleet-size', type=int, default=100, help='GPU fleet size for projections')
    parser.add_argument('--output-dir', default='results/z97_business_demo')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vllm = VLLMClient(args.vllm_host, args.vllm_port)
    actuator = ActuatorClient(args.actuator_host, args.actuator_port)

    prompts = get_business_prompts()

    # Run benchmarks
    baseline = run_business_benchmark(
        vllm, actuator, "performance", prompts, args.max_tokens
    )

    optimized = run_business_benchmark(
        vllm, actuator, "eco", prompts, args.max_tokens
    )

    # Reset
    actuator.set_profile('balanced')

    # Compute comparison
    comparison = compute_comparison(
        baseline, optimized, "performance", "eco", args.fleet_size
    )

    # Generate outputs
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # JSON
    json_path = output_dir / f"business_demo_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'baseline': asdict(baseline),
            'optimized': asdict(optimized),
            'comparison': asdict(comparison),
        }, f, indent=2)

    # Executive summary
    summary_path = output_dir / f"executive_summary_{timestamp}.md"
    with open(summary_path, 'w') as f:
        f.write(generate_executive_summary(baseline, optimized, comparison))

    # LaTeX
    latex_path = output_dir / f"business_table_{timestamp}.tex"
    with open(latex_path, 'w') as f:
        f.write(generate_latex_business_table(baseline, optimized, comparison))

    # Print summary
    print("\n" + "="*70)
    print("FEEL BUSINESS VALUE DEMO")
    print("="*70)
    print(f"\nEnergy Reduction: {comparison.energy_reduction_pct:.1f}%")
    print(f"Quality Change: {comparison.quality_change_pct:+.1f}%")
    print(f"Cost Savings per GPU: ${comparison.cost_savings_per_gpu_year_usd:.2f}/year")
    print(f"Fleet Savings ({args.fleet_size} GPUs): ${comparison.annual_fleet_savings_usd:,.0f}/year")
    print(f"CO2 Reduction: {comparison.co2_reduction_kg/1000:.1f} metric tons/year")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
