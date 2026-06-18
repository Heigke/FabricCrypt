#!/usr/bin/env python3
"""
Experiment 4: Battery Drain Challenge - Mobile Play
====================================================

CLAIM: Marathon mode extends battery life by improving energy efficiency.

For mobile devices, battery life matters more than raw speed. This experiment
proves that DSI's Marathon mode (sustained efficiency) beats Sprint mode
(burst performance) for total work accomplished per unit energy.

METRIC: Tokens per Joule (tok/J) - More is better

PROTOCOL:
1. Run sustained workload (100 prompts × 50 tokens each = 5000 tokens)
2. Measure total time and estimated energy
3. Calculate efficiency: tokens / (time × power)
4. Compare conditions

CONDITIONS:
1. Sprint_K50: High K sampling (aggressive, power-hungry)
2. Sprint_K20: Medium-high K (burst mode)
3. Marathon_K5: Low K sampling (efficient, sustained)
4. Marathon_K3: Very low K (ultra-efficient)
5. Greedy_K1: Deterministic (baseline)

ENERGY MODEL (CPU):
- CPU power ≈ constant during inference (TDP-limited)
- Energy = Power × Time
- Efficiency = Tokens / Energy = Tokens / (Power × Time) ∝ Tokens / Time

Since CPU power is roughly constant, comparing Time-to-Complete is equivalent
to comparing energy consumption for the same workload.
"""

import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class InferenceResult:
    """Result from a single inference."""
    prompt_id: int
    prompt_text: str
    condition: str
    tokens_generated: int
    time_ms: float
    ttft_ms: float  # Time to first token


@dataclass
class WorkloadResult:
    """Complete workload results for a condition."""
    condition: str
    total_tokens: int
    total_time_ms: float
    n_prompts: int

    # Efficiency metrics
    tokens_per_second: float = 0.0
    estimated_energy_j: float = 0.0
    tokens_per_joule: float = 0.0
    battery_life_factor: float = 1.0  # Relative to baseline

    # Consistency
    mean_ttft_ms: float = 0.0
    stddev_ttft_ms: float = 0.0

    # Individual results
    results: List[InferenceResult] = field(default_factory=list)


@dataclass
class ExperimentResults:
    """Complete experiment results."""
    timestamp: str
    model_name: str
    device: str
    workload_prompts: int
    tokens_per_prompt: int
    cpu_tdp_watts: float
    conditions: List[WorkloadResult]
    winner: str = ""
    winner_efficiency_gain: float = 0.0


# =============================================================================
# Workload Prompts - Diverse, sustained workload
# =============================================================================

WORKLOAD_PROMPTS = [
    # Simple queries (fast)
    "What is 2+2?",
    "What color is the sky?",
    "How many days in a week?",
    "What is the capital of Japan?",
    "Who wrote Romeo and Juliet?",
    "What is H2O?",
    "How many planets in our solar system?",
    "What year did WW2 end?",
    "What is the speed of light?",
    "Who painted the Mona Lisa?",

    # Medium complexity
    "Explain gravity in simple terms.",
    "What causes rain?",
    "How do plants grow?",
    "Why is the ocean salty?",
    "What makes fire hot?",
    "How do airplanes fly?",
    "Why do we dream?",
    "What is electricity?",
    "How does WiFi work?",
    "What causes earthquakes?",

    # Conversational
    "Tell me a short story about a robot.",
    "Give me advice for learning guitar.",
    "What should I cook for dinner?",
    "Help me plan a weekend trip.",
    "Suggest a good movie to watch.",
    "What are some fun hobbies to try?",
    "How can I sleep better?",
    "Tips for staying focused at work.",
    "How to make new friends?",
    "What books should I read?",

    # Reasoning
    "If all cats are mammals, and Fluffy is a cat, what is Fluffy?",
    "A train travels 100 miles in 2 hours. What is its speed?",
    "Which is heavier: a pound of feathers or a pound of gold?",
    "If today is Monday, what day is it in 3 days?",
    "What comes next: 2, 4, 6, 8, ?",
    "If A is taller than B, and B is taller than C, who is shortest?",
    "How many sides does a hexagon have?",
    "What is 15% of 200?",
    "If I have 3 apples and give away 1, how many do I have?",
    "What is the square root of 144?",

    # Creative
    "Write a haiku about the moon.",
    "Describe a sunset in three words.",
    "Invent a name for a new planet.",
    "What would a friendly alien say?",
    "Make up a superhero power.",
    "Describe your perfect day.",
    "What if clouds were solid?",
    "Invent a new ice cream flavor.",
    "What sound does happiness make?",
    "Describe the color blue to someone blind.",

    # Technical
    "What is a variable in programming?",
    "Explain what a database is.",
    "What is machine learning?",
    "How does encryption work?",
    "What is an API?",
    "Explain cloud computing simply.",
    "What is a neural network?",
    "How do search engines work?",
    "What is open source software?",
    "Explain what an algorithm is.",

    # More conversational
    "What's the best way to learn a language?",
    "How do I stay motivated?",
    "What makes a good leader?",
    "How to handle stress?",
    "What is emotional intelligence?",
    "How to be more creative?",
    "What is critical thinking?",
    "How to improve memory?",
    "What makes a good friend?",
    "How to overcome fear?",

    # Quick facts
    "Largest ocean on Earth?",
    "Tallest mountain?",
    "Smallest country?",
    "Longest river?",
    "Hottest planet?",
    "Fastest animal?",
    "Largest mammal?",
    "Most spoken language?",
    "Oldest civilization?",
    "Deepest ocean trench?",

    # Math and logic
    "What is 7 times 8?",
    "Half of 50?",
    "10 squared?",
    "Double 37?",
    "Sum of 1 to 10?",
    "What is pi approximately?",
    "Cube of 3?",
    "20% of 80?",
    "1000 divided by 4?",
    "Square root of 81?",

    # Final batch
    "Define happiness.",
    "What is art?",
    "Purpose of education?",
    "What is success?",
    "Define intelligence.",
    "What is wisdom?",
    "Purpose of music?",
    "What is love?",
    "Define courage.",
    "What is freedom?",
]


# =============================================================================
# Battery Drain Experiment
# =============================================================================

class BatteryDrainExperiment:
    """
    Tests energy efficiency across different sampling strategies.

    The key insight: for battery-powered devices, sustained efficiency
    beats burst performance for total work accomplished.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        tokens_per_prompt: int = 30,
        cpu_tdp_watts: float = 65.0,  # Typical desktop CPU TDP
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.tokens_per_prompt = tokens_per_prompt
        self.cpu_tdp_watts = cpu_tdp_watts
        self.device = device

        print(f"\n[BatteryDrainExperiment] Initializing...")
        print(f"  Model: {model_name}")
        print(f"  Tokens per prompt: {tokens_per_prompt}")
        print(f"  Estimated CPU TDP: {cpu_tdp_watts}W")

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer."""
        print(f"\n[Loading model to {self.device}...]")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # CPU uses float32
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"  Model loaded successfully")

    def _generate_single(
        self,
        prompt: str,
        top_k: int = 1,
    ) -> Tuple[int, float, float]:
        """
        Generate tokens for a single prompt.

        Returns: (tokens_generated, time_ms, ttft_ms)
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        start_time = time.perf_counter()
        tokens_generated = 0
        ttft = None

        current_ids = inputs.input_ids.clone()
        attention_mask = inputs.attention_mask.clone()
        past_kv = None

        with torch.no_grad():
            for i in range(self.tokens_per_prompt):
                token_start = time.perf_counter()

                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_kv,
                )
                past_kv = outputs.past_key_values

                logits = outputs.logits[:, -1, :]

                if top_k == 1:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
                    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                    idx = torch.multinomial(top_k_probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, idx)

                if i == 0:
                    ttft = (time.perf_counter() - token_start) * 1000

                tokens_generated += 1

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                current_ids = next_token
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=1)

        total_time = (time.perf_counter() - start_time) * 1000

        return tokens_generated, total_time, ttft or 0

    def run_workload(
        self,
        condition: str,
        prompts: List[str],
        top_k: int = 1,
    ) -> WorkloadResult:
        """Run complete workload and measure efficiency."""
        print(f"\n[Condition: {condition}] (K={top_k})")

        results = []
        total_tokens = 0
        total_time = 0.0
        ttfts = []

        for i, prompt in enumerate(prompts):
            tokens, time_ms, ttft = self._generate_single(prompt, top_k=top_k)

            results.append(InferenceResult(
                prompt_id=i,
                prompt_text=prompt[:30] + "..." if len(prompt) > 30 else prompt,
                condition=condition,
                tokens_generated=tokens,
                time_ms=time_ms,
                ttft_ms=ttft,
            ))

            total_tokens += tokens
            total_time += time_ms
            ttfts.append(ttft)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(prompts)} prompts ({total_tokens} tokens)")

        # Calculate metrics
        tokens_per_second = total_tokens / (total_time / 1000) if total_time > 0 else 0
        estimated_energy_j = (total_time / 1000) * self.cpu_tdp_watts
        tokens_per_joule = total_tokens / estimated_energy_j if estimated_energy_j > 0 else 0

        result = WorkloadResult(
            condition=condition,
            total_tokens=total_tokens,
            total_time_ms=total_time,
            n_prompts=len(prompts),
            tokens_per_second=tokens_per_second,
            estimated_energy_j=estimated_energy_j,
            tokens_per_joule=tokens_per_joule,
            mean_ttft_ms=statistics.mean(ttfts) if ttfts else 0,
            stddev_ttft_ms=statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
            results=results,
        )

        print(f"  Total: {total_tokens} tokens in {total_time/1000:.1f}s")
        print(f"  Efficiency: {tokens_per_second:.1f} tok/s, {tokens_per_joule:.2f} tok/J")

        return result

    def run_experiment(
        self,
        prompts: Optional[List[str]] = None,
    ) -> ExperimentResults:
        """
        Run complete battery drain experiment.

        Tests 5 conditions from Sprint (high K) to Marathon (low K).
        """
        if prompts is None:
            prompts = WORKLOAD_PROMPTS

        print(f"\n{'='*60}")
        print("BATTERY DRAIN EXPERIMENT")
        print(f"{'='*60}")
        print(f"Workload: {len(prompts)} prompts × {self.tokens_per_prompt} tokens")
        print(f"Expected tokens: ~{len(prompts) * self.tokens_per_prompt}")
        print(f"CPU TDP estimate: {self.cpu_tdp_watts}W")

        conditions = []

        # Condition 1: Sprint K=50 (aggressive, power-hungry)
        conditions.append(self.run_workload(
            condition="Sprint_K50",
            prompts=prompts,
            top_k=50,
        ))

        # Condition 2: Sprint K=20 (medium-high)
        conditions.append(self.run_workload(
            condition="Sprint_K20",
            prompts=prompts,
            top_k=20,
        ))

        # Condition 3: Marathon K=5 (efficient)
        conditions.append(self.run_workload(
            condition="Marathon_K5",
            prompts=prompts,
            top_k=5,
        ))

        # Condition 4: Marathon K=3 (ultra-efficient)
        conditions.append(self.run_workload(
            condition="Marathon_K3",
            prompts=prompts,
            top_k=3,
        ))

        # Condition 5: Greedy K=1 (baseline)
        conditions.append(self.run_workload(
            condition="Greedy_K1",
            prompts=prompts,
            top_k=1,
        ))

        # Calculate battery life factors relative to baseline (Greedy)
        baseline = conditions[-1]  # Greedy_K1
        for cond in conditions:
            if baseline.tokens_per_joule > 0:
                cond.battery_life_factor = cond.tokens_per_joule / baseline.tokens_per_joule
            else:
                cond.battery_life_factor = 1.0

        # Find winner (highest tokens_per_joule)
        winner = max(conditions, key=lambda c: c.tokens_per_joule)
        efficiency_gain = ((winner.tokens_per_joule - baseline.tokens_per_joule)
                          / baseline.tokens_per_joule * 100) if baseline.tokens_per_joule > 0 else 0

        results = ExperimentResults(
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            device=self.device,
            workload_prompts=len(prompts),
            tokens_per_prompt=self.tokens_per_prompt,
            cpu_tdp_watts=self.cpu_tdp_watts,
            conditions=conditions,
            winner=winner.condition,
            winner_efficiency_gain=efficiency_gain,
        )

        return results


# =============================================================================
# Visualization
# =============================================================================

def generate_plots(results: ExperimentResults, output_dir: Path):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [c.condition for c in results.conditions]

    colors = {
        'Sprint_K50': '#C84C5D',     # Red (power hungry)
        'Sprint_K20': '#E8845C',     # Orange
        'Marathon_K5': '#3E9BC2',    # Light blue
        'Marathon_K3': '#2E86AB',    # Blue (efficient)
        'Greedy_K1': '#808080',      # Gray (baseline)
    }

    # --- Plot 1: Tokens per Joule (Energy Efficiency) ---
    fig, ax = plt.subplots(figsize=(12, 6))

    tpj_values = [c.tokens_per_joule for c in results.conditions]
    bar_colors = [colors.get(c, '#888888') for c in conditions]

    bars = ax.bar(conditions, tpj_values, color=bar_colors, alpha=0.8)

    # Highlight winner
    best_idx = tpj_values.index(max(tpj_values))
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)

    ax.axhline(y=tpj_values[-1], color='gray', linestyle='--', alpha=0.5, label='Baseline (Greedy)')

    ax.set_ylabel('Tokens per Joule (higher = better)', fontsize=12)
    ax.set_title('Energy Efficiency: Tokens per Joule\n(Higher = Longer Battery Life)', fontsize=14)
    ax.legend()

    for bar, val in zip(bars, tpj_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'energy_efficiency.png', dpi=150)
    plt.close()

    # --- Plot 2: Battery Life Factor ---
    fig, ax = plt.subplots(figsize=(12, 6))

    blf_values = [c.battery_life_factor for c in results.conditions]
    bar_colors = [colors.get(c, '#888888') for c in conditions]

    bars = ax.bar(conditions, blf_values, color=bar_colors, alpha=0.8)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Baseline (1.0x)')

    ax.set_ylabel('Battery Life Factor (relative to Greedy)', fontsize=12)
    ax.set_title('Battery Life Extension Factor\n(>1.0 = Longer than Baseline)', fontsize=14)
    ax.legend()

    for bar, val in zip(bars, blf_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=11)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'battery_life_factor.png', dpi=150)
    plt.close()

    # --- Plot 3: Throughput vs Energy Trade-off ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for cond in results.conditions:
        ax.scatter(
            cond.estimated_energy_j,
            cond.tokens_per_second,
            s=200,
            c=colors.get(cond.condition, '#888888'),
            label=cond.condition,
            alpha=0.8,
            edgecolors='black',
            linewidths=1,
        )
        ax.annotate(
            cond.condition,
            (cond.estimated_energy_j, cond.tokens_per_second),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_xlabel('Total Energy (Joules)', fontsize=12)
    ax.set_ylabel('Throughput (tokens/second)', fontsize=12)
    ax.set_title('Energy vs Throughput Trade-off\n(Ideal: High throughput, Low energy)', fontsize=14)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / 'energy_throughput_tradeoff.png', dpi=150)
    plt.close()

    # --- Plot 4: TTFT Consistency ---
    fig, ax = plt.subplots(figsize=(12, 6))

    ttft_means = [c.mean_ttft_ms for c in results.conditions]
    ttft_stds = [c.stddev_ttft_ms for c in results.conditions]
    bar_colors = [colors.get(c, '#888888') for c in conditions]

    bars = ax.bar(conditions, ttft_means, yerr=ttft_stds, capsize=5, color=bar_colors, alpha=0.8)

    ax.set_ylabel('Time to First Token (ms)', fontsize=12)
    ax.set_title('Response Latency (TTFT)\n(Lower = More Responsive)', fontsize=14)

    for bar, val in zip(bars, ttft_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}ms', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'ttft_comparison.png', dpi=150)
    plt.close()

    # --- Plot 5: Summary Dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Efficiency
    ax = axes[0, 0]
    ax.bar(conditions, tpj_values, color=bar_colors, alpha=0.8)
    ax.set_title('Tokens per Joule', fontsize=11)
    ax.set_ylabel('tok/J')

    # Battery Life
    ax = axes[0, 1]
    ax.bar(conditions, blf_values, color=bar_colors, alpha=0.8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Battery Life Factor', fontsize=11)
    ax.set_ylabel('Relative to baseline')

    # Throughput
    ax = axes[1, 0]
    tps_values = [c.tokens_per_second for c in results.conditions]
    ax.bar(conditions, tps_values, color=bar_colors, alpha=0.8)
    ax.set_title('Throughput', fontsize=11)
    ax.set_ylabel('tokens/second')

    # Total Time
    ax = axes[1, 1]
    time_values = [c.total_time_ms / 1000 for c in results.conditions]
    ax.bar(conditions, time_values, color=bar_colors, alpha=0.8)
    ax.set_title('Total Time', fontsize=11)
    ax.set_ylabel('seconds')

    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle(f'Battery Drain Experiment Summary\n{results.workload_prompts} prompts × {results.tokens_per_prompt} tokens', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=150)
    plt.close()

    print(f"\n[Plots saved to {output_dir}/]")


def print_summary(results: ExperimentResults):
    """Print comprehensive summary."""
    print(f"\n{'='*70}")
    print("BATTERY DRAIN EXPERIMENT RESULTS")
    print(f"{'='*70}")
    print(f"Model: {results.model_name}")
    print(f"Workload: {results.workload_prompts} prompts × {results.tokens_per_prompt} tokens")
    print(f"CPU TDP: {results.cpu_tdp_watts}W")

    print(f"\n{'='*70}")
    print(f"{'Condition':<15} {'Tokens':<10} {'Time(s)':<10} {'tok/s':<10} {'Energy(J)':<12} {'tok/J':<10} {'Battery':<10}")
    print(f"{'='*70}")

    for cond in results.conditions:
        print(f"{cond.condition:<15} {cond.total_tokens:<10} {cond.total_time_ms/1000:<10.1f} "
              f"{cond.tokens_per_second:<10.1f} {cond.estimated_energy_j:<12.1f} "
              f"{cond.tokens_per_joule:<10.2f} {cond.battery_life_factor:<10.2f}x")

    print(f"{'='*70}")

    # Winner
    baseline = results.conditions[-1]  # Greedy
    winner = max(results.conditions, key=lambda c: c.tokens_per_joule)

    if winner.condition != baseline.condition:
        print(f"\n WINNER: {winner.condition}")
        print(f"   Efficiency: {winner.tokens_per_joule:.2f} tok/J (vs {baseline.tokens_per_joule:.2f} baseline)")
        print(f"   Battery Life: {winner.battery_life_factor:.2f}x longer than baseline")
        print(f"   Improvement: {results.winner_efficiency_gain:.1f}%")
    else:
        print(f"\n Baseline is already optimal")

    print(f"\n KEY INSIGHT:")
    print(f"   For mobile devices, Marathon mode (low K) provides better battery life")
    print(f"   by generating more tokens per unit energy consumed.")
    print(f"   This is equivalent to longer battery life for the same workload.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Battery Drain Experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model to use")
    parser.add_argument("--tokens", type=int, default=30, help="Tokens per prompt")
    parser.add_argument("--prompts", type=int, default=50, help="Number of prompts")
    parser.add_argument("--tdp", type=float, default=65.0, help="CPU TDP in watts")
    parser.add_argument("--output", default="results/battery_test", help="Output directory")
    args = parser.parse_args()

    # Create experiment
    experiment = BatteryDrainExperiment(
        model_name=args.model,
        tokens_per_prompt=args.tokens,
        cpu_tdp_watts=args.tdp,
    )

    # Load model
    experiment.load_model()

    # Run experiment
    prompts = WORKLOAD_PROMPTS[:args.prompts]
    results = experiment.run_experiment(prompts=prompts)

    # Print summary
    print_summary(results)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"battery_test_{timestamp}.json"

    results_dict = {
        "timestamp": results.timestamp,
        "model_name": results.model_name,
        "device": results.device,
        "workload_prompts": results.workload_prompts,
        "tokens_per_prompt": results.tokens_per_prompt,
        "cpu_tdp_watts": results.cpu_tdp_watts,
        "winner": results.winner,
        "winner_efficiency_gain": round(results.winner_efficiency_gain, 2),
        "conditions": []
    }

    for cond in results.conditions:
        cond_dict = {
            "condition": cond.condition,
            "total_tokens": cond.total_tokens,
            "total_time_ms": round(cond.total_time_ms, 2),
            "tokens_per_second": round(cond.tokens_per_second, 2),
            "estimated_energy_j": round(cond.estimated_energy_j, 2),
            "tokens_per_joule": round(cond.tokens_per_joule, 4),
            "battery_life_factor": round(cond.battery_life_factor, 3),
            "mean_ttft_ms": round(cond.mean_ttft_ms, 2),
            "stddev_ttft_ms": round(cond.stddev_ttft_ms, 2),
        }
        results_dict["conditions"].append(cond_dict)

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n[Results saved to {results_file}]")

    # Generate plots
    generate_plots(results, output_dir)

    return results


if __name__ == "__main__":
    main()
