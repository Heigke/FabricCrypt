#!/usr/bin/env python3
"""
FEEL Quality Benchmark - Measure Quality vs Energy Tradeoff

This script proves that energy savings don't significantly degrade quality.

Metrics:
1. Response quality score (0-1) via simple heuristics
2. Task completion rate
3. Response coherence (sentence structure, keywords)

The goal: Show "<X% quality loss for Y% energy reduction"

Author: FEEL Research Team
Date: 2026-01-21
"""

import os
import sys
import json
import time
import argparse
import statistics
import math
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# Add project root
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityTask:
    """A benchmark task with expected answer properties."""
    id: str
    prompt: str
    category: str  # factual, reasoning, creative, code
    expected_keywords: List[str] = field(default_factory=list)
    expected_length_min: int = 10
    expected_length_max: int = 500
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class QualityResult:
    """Quality evaluation result."""
    task_id: str
    profile: str
    response: str
    prompt_tokens: int
    completion_tokens: int

    # Quality metrics
    keyword_score: float  # 0-1, fraction of expected keywords present
    length_score: float   # 0-1, based on expected length
    coherence_score: float  # 0-1, sentence structure quality
    overall_score: float  # Weighted average

    # Energy metrics
    energy_j: float
    energy_per_token_j: float
    latency_ms: float


@dataclass
class ProfileQuality:
    """Aggregated quality for a profile."""
    profile: str
    results: List[QualityResult]

    def stats(self) -> Dict[str, Any]:
        if not self.results:
            return {}

        scores = [r.overall_score for r in self.results]
        energies = [r.energy_per_token_j for r in self.results]
        latencies = [r.latency_ms for r in self.results]

        def ci_95(values: List[float]) -> Tuple[float, float]:
            if len(values) < 2:
                return (values[0], values[0]) if values else (0, 0)
            n = len(values)
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            t = 1.96 if n > 30 else 2.0
            margin = t * std / math.sqrt(n)
            return (mean - margin, mean + margin)

        return {
            'n_tasks': len(self.results),
            'quality_score': {
                'mean': statistics.mean(scores),
                'std': statistics.stdev(scores) if len(scores) > 1 else 0,
                'min': min(scores),
                'max': max(scores),
                'ci_95': ci_95(scores),
            },
            'energy_per_token_j': {
                'mean': statistics.mean(energies),
                'std': statistics.stdev(energies) if len(energies) > 1 else 0,
                'ci_95': ci_95(energies),
            },
            'latency_ms': {
                'mean': statistics.mean(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            },
            'task_completion_rate': sum(1 for r in self.results if r.overall_score > 0.5) / len(self.results),
        }


# ============================================================================
# Quality Evaluation
# ============================================================================

def evaluate_keyword_presence(response: str, keywords: List[str]) -> float:
    """Score based on expected keyword presence (0-1)."""
    if not keywords:
        return 1.0

    response_lower = response.lower()
    found = sum(1 for kw in keywords if kw.lower() in response_lower)
    return found / len(keywords)


def evaluate_length(response: str, min_len: int, max_len: int) -> float:
    """Score based on response length appropriateness (0-1)."""
    word_count = len(response.split())

    if word_count < min_len:
        return max(0, word_count / min_len)  # Penalize too short
    elif word_count > max_len:
        return max(0, 1 - (word_count - max_len) / max_len)  # Penalize too long
    else:
        return 1.0


def evaluate_coherence(response: str) -> float:
    """
    Simple coherence score based on:
    - Sentence structure (has periods, proper casing)
    - Word variety
    - No obvious repetition

    Returns 0-1 score.
    """
    if not response or len(response) < 10:
        return 0.0

    scores = []

    # Sentence structure: ends with punctuation
    sentences = re.split(r'[.!?]+', response)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    if valid_sentences:
        # Check capitalization
        cap_score = sum(1 for s in valid_sentences if s[0].isupper()) / len(valid_sentences)
        scores.append(cap_score)

    # Word variety (unique words / total words)
    words = response.lower().split()
    if words:
        variety = len(set(words)) / len(words)
        scores.append(min(1.0, variety * 2))  # Scale up, variety of 0.5 is good

    # No excessive repetition (same word appearing > 10% of text)
    if words:
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        max_freq = max(word_counts.values())
        repetition_score = 1.0 - (max_freq / len(words) - 0.05) if max_freq / len(words) > 0.1 else 1.0
        scores.append(max(0, repetition_score))

    # Reasonable length per sentence
    if valid_sentences:
        avg_len = statistics.mean(len(s.split()) for s in valid_sentences)
        length_score = 1.0 if 5 < avg_len < 30 else max(0, 1 - abs(avg_len - 15) / 30)
        scores.append(length_score)

    return statistics.mean(scores) if scores else 0.5


def evaluate_quality(task: QualityTask, response: str) -> Dict[str, float]:
    """Evaluate response quality for a task."""
    keyword_score = evaluate_keyword_presence(response, task.expected_keywords)
    length_score = evaluate_length(response, task.expected_length_min, task.expected_length_max)
    coherence_score = evaluate_coherence(response)

    # Weighted average
    # Keywords: 40%, Length: 20%, Coherence: 40%
    overall = 0.4 * keyword_score + 0.2 * length_score + 0.4 * coherence_score

    return {
        'keyword_score': keyword_score,
        'length_score': length_score,
        'coherence_score': coherence_score,
        'overall_score': overall,
    }


# ============================================================================
# Benchmark Tasks
# ============================================================================

def get_benchmark_tasks() -> List[QualityTask]:
    """Get diverse benchmark tasks."""
    return [
        # Factual
        QualityTask(
            id="factual_01",
            prompt="What is the capital of France and what is it famous for?",
            category="factual",
            expected_keywords=["Paris", "Eiffel", "France"],
            expected_length_min=20,
            expected_length_max=150,
        ),
        QualityTask(
            id="factual_02",
            prompt="Explain what photosynthesis is in simple terms.",
            category="factual",
            expected_keywords=["plant", "light", "energy", "carbon", "oxygen"],
            expected_length_min=30,
            expected_length_max=200,
        ),
        QualityTask(
            id="factual_03",
            prompt="What is the speed of light in a vacuum?",
            category="factual",
            expected_keywords=["300", "km", "meters", "second"],
            expected_length_min=10,
            expected_length_max=100,
        ),

        # Reasoning
        QualityTask(
            id="reasoning_01",
            prompt="If all cats have tails and Fluffy is a cat, what can you conclude about Fluffy?",
            category="reasoning",
            expected_keywords=["tail", "Fluffy"],
            expected_length_min=10,
            expected_length_max=100,
        ),
        QualityTask(
            id="reasoning_02",
            prompt="What are the pros and cons of working from home?",
            category="reasoning",
            expected_keywords=["flexibility", "commute", "isolation", "productivity"],
            expected_length_min=50,
            expected_length_max=300,
        ),

        # Creative
        QualityTask(
            id="creative_01",
            prompt="Write a haiku about artificial intelligence.",
            category="creative",
            expected_keywords=[],  # Creative tasks are harder to keyword-match
            expected_length_min=5,
            expected_length_max=50,
        ),
        QualityTask(
            id="creative_02",
            prompt="Describe a sunset in a poetic way.",
            category="creative",
            expected_keywords=["sky", "color", "sun"],
            expected_length_min=20,
            expected_length_max=150,
        ),

        # Code/Technical
        QualityTask(
            id="code_01",
            prompt="Write a Python function that checks if a number is prime.",
            category="code",
            expected_keywords=["def", "return", "prime", "if"],
            expected_length_min=20,
            expected_length_max=200,
        ),
        QualityTask(
            id="code_02",
            prompt="Explain what recursion is in programming.",
            category="code",
            expected_keywords=["function", "call", "itself", "base"],
            expected_length_min=30,
            expected_length_max=200,
        ),

        # Instructions
        QualityTask(
            id="instruct_01",
            prompt="Give me 3 tips for writing clean code.",
            category="instructions",
            expected_keywords=["1", "2", "3", "code"],
            expected_length_min=30,
            expected_length_max=250,
        ),
    ]


# ============================================================================
# vLLM Client
# ============================================================================

class VLLMClient:
    """Simple vLLM server client."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.base_url = f"http://{host}:{port}"

    def generate(self, prompt: str, max_tokens: int = 128) -> Tuple[str, int, int, float]:
        """Generate and return (text, prompt_tokens, completion_tokens, latency_ms)."""
        import urllib.request
        import urllib.error

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
            prompt_tokens = usage.get('prompt_tokens', len(prompt.split()))
            completion_tokens = usage.get('completion_tokens', len(text.split()))

            return text, prompt_tokens, completion_tokens, latency_ms

        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            return "", 0, 0, 0


# ============================================================================
# Actuator Client
# ============================================================================

class ActuatorClient:
    """Simple actuator daemon client."""

    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"

    def set_profile(self, profile: str) -> bool:
        import urllib.request
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
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.base_url}/energy", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except:
            return None


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_quality_benchmark(
    vllm: VLLMClient,
    actuator: ActuatorClient,
    tasks: List[QualityTask],
    profile: str,
    max_tokens: int = 128,
) -> ProfileQuality:
    """Run quality benchmark for a profile."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Quality Benchmark: {profile}")
    logger.info(f"{'='*60}")

    # Set profile
    if not actuator.set_profile(profile):
        logger.warning(f"Failed to set profile {profile}")

    time.sleep(2)  # Let profile settle

    results = []

    for i, task in enumerate(tasks):
        logger.info(f"  Task {i+1}/{len(tasks)}: {task.id}")

        # Get initial energy
        energy_start = actuator.get_energy()
        start_mj = energy_start.get('energy_mj') if energy_start else None

        # Generate
        response, prompt_tokens, completion_tokens, latency_ms = vllm.generate(
            task.prompt, max_tokens
        )

        # Get final energy
        energy_end = actuator.get_energy()
        end_mj = energy_end.get('energy_mj') if energy_end else None

        # Calculate energy
        if start_mj and end_mj:
            energy_j = (end_mj - start_mj) / 1000
        else:
            # Estimate from typical power
            energy_j = 0.3 * (latency_ms / 1000)  # Rough estimate

        total_tokens = prompt_tokens + completion_tokens
        energy_per_token = energy_j / max(total_tokens, 1)

        # Evaluate quality
        quality = evaluate_quality(task, response)

        result = QualityResult(
            task_id=task.id,
            profile=profile,
            response=response[:200],  # Truncate for storage
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            keyword_score=quality['keyword_score'],
            length_score=quality['length_score'],
            coherence_score=quality['coherence_score'],
            overall_score=quality['overall_score'],
            energy_j=energy_j,
            energy_per_token_j=energy_per_token,
            latency_ms=latency_ms,
        )

        results.append(result)

        logger.info(f"    Quality: {quality['overall_score']:.2f}, Energy: {energy_j:.3f}J")

        time.sleep(0.5)  # Brief pause between tasks

    return ProfileQuality(profile=profile, results=results)


# ============================================================================
# Report Generation
# ============================================================================

def generate_latex_table(results: Dict[str, ProfileQuality]) -> str:
    """Generate LaTeX table for quality vs energy."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{FEEL Quality vs Energy Tradeoff}",
        r"\label{tab:quality-energy}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Profile} & \textbf{Quality} & \textbf{J/token} & \textbf{Latency (ms)} & \textbf{Completion} \\",
        r"\midrule",
    ]

    for profile_name, profile_result in results.items():
        stats = profile_result.stats()
        if not stats:
            continue

        quality = stats['quality_score']['mean']
        energy = stats['energy_per_token_j']['mean']
        latency = stats['latency_ms']['mean']
        completion = stats['task_completion_rate']

        lines.append(
            f"{profile_name} & {quality:.3f} & {energy:.4f} & {latency:.0f} & {completion:.0%} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_markdown_report(results: Dict[str, ProfileQuality]) -> str:
    """Generate markdown report."""
    lines = [
        "# FEEL Quality vs Energy Tradeoff Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Results",
        "",
        "| Profile | Quality Score | J/token | Latency (ms) | Task Completion |",
        "|---------|---------------|---------|--------------|-----------------|",
    ]

    for profile_name, profile_result in results.items():
        stats = profile_result.stats()
        if not stats:
            continue

        quality = stats['quality_score']['mean']
        energy = stats['energy_per_token_j']['mean']
        latency = stats['latency_ms']['mean']
        completion = stats['task_completion_rate']

        lines.append(
            f"| {profile_name} | {quality:.3f} | {energy:.4f} | {latency:.0f} | {completion:.0%} |"
        )

    lines.append("")

    # Analysis
    if 'eco' in results and 'performance' in results:
        eco = results['eco'].stats()
        perf = results['performance'].stats()

        if eco and perf:
            eco_quality = eco['quality_score']['mean']
            perf_quality = perf['quality_score']['mean']
            quality_diff = (eco_quality - perf_quality) / perf_quality * 100

            eco_energy = eco['energy_per_token_j']['mean']
            perf_energy = perf['energy_per_token_j']['mean']
            energy_reduction = (perf_energy - eco_energy) / perf_energy * 100

            lines.extend([
                "## Key Finding",
                "",
                f"**Energy reduction:** {energy_reduction:.1f}%",
                f"**Quality change:** {quality_diff:+.1f}%",
                "",
                f"→ {abs(energy_reduction):.1f}% energy savings with {abs(quality_diff):.1f}% quality {'improvement' if quality_diff > 0 else 'reduction'}",
            ])

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='FEEL Quality Benchmark')
    parser.add_argument('--vllm-host', default='localhost', help='vLLM server host')
    parser.add_argument('--vllm-port', type=int, default=8000, help='vLLM server port')
    parser.add_argument('--actuator-host', default='192.168.0.38', help='Actuator daemon host')
    parser.add_argument('--actuator-port', type=int, default=9877, help='Actuator daemon port')
    parser.add_argument('--max-tokens', type=int, default=128, help='Max tokens per response')
    parser.add_argument('--profiles', default='eco,balanced,performance',
                       help='Profiles to test')
    parser.add_argument('--output-dir', default='results/z95_quality', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = [p.strip() for p in args.profiles.split(',')]

    # Initialize clients
    vllm = VLLMClient(args.vllm_host, args.vllm_port)
    actuator = ActuatorClient(args.actuator_host, args.actuator_port)

    # Get benchmark tasks
    tasks = get_benchmark_tasks()
    logger.info(f"Loaded {len(tasks)} benchmark tasks")

    # Run benchmarks
    results: Dict[str, ProfileQuality] = {}

    for profile in profiles:
        results[profile] = run_quality_benchmark(
            vllm, actuator, tasks, profile, args.max_tokens
        )

    # Reset profile
    actuator.set_profile('balanced')

    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # JSON
    json_path = output_dir / f"quality_{timestamp}.json"
    json_results = {
        profile: {
            'stats': r.stats(),
            'results': [asdict(res) for res in r.results],
        }
        for profile, r in results.items()
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # LaTeX
    latex_path = output_dir / f"quality_{timestamp}.tex"
    with open(latex_path, 'w') as f:
        f.write(generate_latex_table(results))

    # Markdown
    md_path = output_dir / f"quality_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write(generate_markdown_report(results))

    # Print summary
    print("\n" + "="*60)
    print("QUALITY VS ENERGY TRADEOFF RESULTS")
    print("="*60)

    for profile, result in results.items():
        stats = result.stats()
        if stats:
            print(f"\n{profile}:")
            print(f"  Quality:     {stats['quality_score']['mean']:.3f} ± {stats['quality_score']['std']:.3f}")
            print(f"  J/token:     {stats['energy_per_token_j']['mean']:.4f}")
            print(f"  Latency:     {stats['latency_ms']['mean']:.0f} ms")
            print(f"  Completion:  {stats['task_completion_rate']:.0%}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
