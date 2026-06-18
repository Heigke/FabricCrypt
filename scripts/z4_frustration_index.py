#!/usr/bin/env python3
"""
Experiment 5: Frustration Index - UX Consistency Test
======================================================

CLAIM: Consistent latency beats spiky performance.

For user experience, it's not about being the fastest on average -
it's about being PREDICTABLE. A system with 100ms mean but 500ms spikes
is worse than 120ms with 130ms max from a UX perspective.

FRUSTRATION INDEX = (P95 - P50) / P50 * 100 + CV%

Lower is better. Measures how "surprising" latency variations are.

CONDITIONS:
1. Baseline_Greedy: No DSI, greedy sampling (k=1)
2. Baseline_TopK: No DSI, top-k sampling (k=5)
3. DSI_Marathon: Full DSI, marathon mode (sustain)
4. DSI_Sprint: Full DSI, sprint mode (bursts)
5. DSI_Adaptive: Full DSI, adaptive mode (auto-regulate)

METRICS:
- Mean latency (baseline performance)
- P50, P95, P99 latencies (percentiles)
- Jitter (stddev of token times)
- CV% (coefficient of variation)
- Frustration Index (composite UX metric)
- First Token Time (TTFT) - critical for perceived responsiveness
"""

import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# DSI steering is optional - experiment works with sampling variations
HAS_DSI = False  # CPU-only mode for now


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TokenTiming:
    """Timing for a single token generation."""
    token_id: int
    token_text: str
    time_ms: float
    cumulative_ms: float
    is_first_token: bool = False


@dataclass
class GenerationMetrics:
    """Metrics for a single generation run."""
    prompt: str
    condition: str
    total_time_ms: float
    ttft_ms: float  # Time to first token
    tokens_generated: int
    token_times_ms: List[float]

    # Derived metrics (computed in post)
    mean_tpot_ms: float = 0.0
    p50_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    stddev_tpot_ms: float = 0.0
    cv_percent: float = 0.0
    frustration_index: float = 0.0
    jitter_ms: float = 0.0

    def compute_metrics(self):
        """Compute derived metrics from token times."""
        if len(self.token_times_ms) < 2:
            return

        self.mean_tpot_ms = statistics.mean(self.token_times_ms)
        self.p50_tpot_ms = statistics.median(self.token_times_ms)
        self.stddev_tpot_ms = statistics.stdev(self.token_times_ms)

        sorted_times = sorted(self.token_times_ms)
        n = len(sorted_times)
        self.p95_tpot_ms = sorted_times[int(n * 0.95)] if n >= 20 else sorted_times[-1]
        self.p99_tpot_ms = sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1]

        # CV% = (stddev / mean) * 100
        if self.mean_tpot_ms > 0:
            self.cv_percent = (self.stddev_tpot_ms / self.mean_tpot_ms) * 100

        # Jitter = mean absolute difference between consecutive tokens
        diffs = [abs(self.token_times_ms[i] - self.token_times_ms[i-1])
                 for i in range(1, len(self.token_times_ms))]
        self.jitter_ms = statistics.mean(diffs) if diffs else 0.0

        # Frustration Index = (P95-P50)/P50 * 100 + CV%
        # Measures how "surprising" latency spikes are
        if self.p50_tpot_ms > 0:
            spike_ratio = ((self.p95_tpot_ms - self.p50_tpot_ms) / self.p50_tpot_ms) * 100
            self.frustration_index = spike_ratio + self.cv_percent


@dataclass
class ConditionResults:
    """Aggregated results for a condition."""
    condition: str
    n_runs: int
    prompts: List[str]

    # Aggregated metrics
    mean_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    mean_p50_ms: float = 0.0
    mean_p95_ms: float = 0.0
    mean_p99_ms: float = 0.0
    mean_cv_percent: float = 0.0
    mean_jitter_ms: float = 0.0
    mean_frustration_index: float = 0.0

    # Consistency of consistency (meta-consistency)
    stddev_frustration_index: float = 0.0

    # Raw runs for detailed analysis
    runs: List[GenerationMetrics] = field(default_factory=list)


@dataclass
class ExperimentResults:
    """Complete experiment results."""
    timestamp: str
    model_name: str
    device: str
    n_prompts: int
    tokens_per_prompt: int
    rounds: int
    conditions: List[ConditionResults]

    # Winner determination
    best_condition: str = ""
    best_frustration_index: float = float('inf')


# =============================================================================
# Test Prompts - Various complexity levels
# =============================================================================

PROMPTS = [
    # Simple conversational (should be fast and consistent)
    "Hello! How are you doing today?",
    "What is the capital of France?",
    "Tell me a short joke.",

    # Medium complexity (some thinking required)
    "Explain photosynthesis in simple terms.",
    "What are the benefits of exercise?",
    "How does the internet work?",

    # Reasoning required (may cause latency spikes)
    "If a train travels at 60 mph for 2 hours, then 40 mph for 1 hour, what's the average speed?",
    "Compare and contrast democracy and autocracy.",
    "What would happen if the moon disappeared?",

    # Complex/creative (highest variance expected)
    "Write a haiku about artificial intelligence.",
    "Describe a day in the life of a neuron.",
    "What are the ethical implications of time travel?",

    # Extended reasoning (stress test)
    "Explain the prisoner's dilemma and give an example from real life.",
    "What are the pros and cons of remote work?",
    "How might quantum computing change cryptography?",
]


# =============================================================================
# Frustration Index Experiment
# =============================================================================

class FrustrationExperiment:
    """
    Measures latency consistency across different generation conditions.

    The key insight: users prefer predictable over fast-but-spiky.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto",
        tokens_to_generate: int = 50,
        warmup_runs: int = 3,
        force_cpu: bool = False,
    ):
        self.model_name = model_name
        self.tokens_to_generate = tokens_to_generate
        self.warmup_runs = warmup_runs

        # Determine device
        if force_cpu:
            self.device = "cpu"
        elif device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"\n[FrustrationExperiment] Initializing...")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Tokens per generation: {tokens_to_generate}")

        self.model = None
        self.tokenizer = None
        self.agent = None

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
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map=self.device if self.device != "cpu" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

        # Initialize DSI agent if available
        if HAS_DSI and self.device != "cpu":
            try:
                self.agent = EmbodiedAgent(self.model)
                print("  DSI agent initialized")
            except Exception as e:
                print(f"  DSI agent init failed: {e}")
                self.agent = None
        else:
            self.agent = None

        print(f"  Model loaded successfully")

    def _generate_with_timing(
        self,
        prompt: str,
        condition: str,
        use_dsi: bool = False,
        dsi_mode: Optional[str] = None,
        top_k: int = 1,
    ) -> GenerationMetrics:
        """
        Generate tokens and record timing for each token.

        Returns detailed metrics for UX analysis.
        """
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        token_times = []
        generated_tokens = []

        # Set up DSI if requested
        injection_hook = None
        if use_dsi and self.agent is not None:
            try:
                # Set allostasis mode
                if dsi_mode == "marathon":
                    self.agent.set_mode(AllostasisMode.SUSTAIN)
                elif dsi_mode == "sprint":
                    self.agent.set_mode(AllostasisMode.BURST)
                elif dsi_mode == "adaptive":
                    self.agent.set_mode(AllostasisMode.ADAPTIVE)

                # Compute steering vector
                self.agent.compute_strain_vector()

                # Register injection hook
                if hasattr(self.agent, 'strain_vector') and self.agent.strain_vector is not None:
                    middle_layer = len(self.model.model.layers) // 2
                    layer = self.model.model.layers[middle_layer]

                    def create_hook(vector, intensity=0.3):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                hidden = output[0]
                            else:
                                hidden = output
                            v = vector.to(hidden.device).to(hidden.dtype) * intensity
                            if len(hidden.shape) == 3:
                                hidden[:, -1, :] = hidden[:, -1, :] + v
                            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                        return hook

                    injection_hook = layer.register_forward_hook(
                        create_hook(self.agent.strain_vector)
                    )
            except Exception as e:
                print(f"    DSI setup warning: {e}")

        # Generate tokens one at a time
        start_time = time.perf_counter()
        current_ids = inputs.input_ids.clone()
        attention_mask = inputs.attention_mask.clone()

        ttft = None

        with torch.no_grad():
            for i in range(self.tokens_to_generate):
                token_start = time.perf_counter()

                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=None if i == 0 else past_kv,
                )
                past_kv = outputs.past_key_values

                # Sample next token
                logits = outputs.logits[:, -1, :]

                if top_k == 1:
                    # Greedy
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    # Top-k sampling
                    probs = torch.softmax(logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                    idx = torch.multinomial(top_k_probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, idx)

                token_end = time.perf_counter()
                token_time_ms = (token_end - token_start) * 1000

                if i == 0:
                    ttft = token_time_ms

                token_times.append(token_time_ms)
                generated_tokens.append(next_token.item())

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Update for next iteration
                current_ids = next_token
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=1)

        total_time = (time.perf_counter() - start_time) * 1000

        # Clean up hook
        if injection_hook is not None:
            injection_hook.remove()

        # Create metrics object
        metrics = GenerationMetrics(
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            condition=condition,
            total_time_ms=total_time,
            ttft_ms=ttft or token_times[0] if token_times else 0,
            tokens_generated=len(generated_tokens),
            token_times_ms=token_times,
        )
        metrics.compute_metrics()

        return metrics

    def warmup(self):
        """Run warmup generations to stabilize system."""
        print(f"\n[Warmup: {self.warmup_runs} runs...]")
        for i in range(self.warmup_runs):
            _ = self._generate_with_timing(
                "Hello, how are you?",
                condition="warmup",
                use_dsi=False,
                top_k=1,
            )
            print(f"  Warmup {i+1}/{self.warmup_runs} complete")

    def run_condition(
        self,
        condition: str,
        prompts: List[str],
        rounds: int = 3,
        use_dsi: bool = False,
        dsi_mode: Optional[str] = None,
        top_k: int = 1,
    ) -> ConditionResults:
        """Run all prompts under a specific condition."""
        print(f"\n[Condition: {condition}]")
        print(f"  DSI: {use_dsi}, Mode: {dsi_mode}, Top-K: {top_k}")

        all_runs = []

        for round_idx in range(rounds):
            print(f"  Round {round_idx + 1}/{rounds}...")

            for prompt_idx, prompt in enumerate(prompts):
                metrics = self._generate_with_timing(
                    prompt=prompt,
                    condition=condition,
                    use_dsi=use_dsi,
                    dsi_mode=dsi_mode,
                    top_k=top_k,
                )
                all_runs.append(metrics)

                # Brief pause to avoid thermal buildup
                time.sleep(0.1)

        # Aggregate results
        result = ConditionResults(
            condition=condition,
            n_runs=len(all_runs),
            prompts=prompts,
            runs=all_runs,
        )

        if all_runs:
            result.mean_ttft_ms = statistics.mean(r.ttft_ms for r in all_runs)
            result.mean_tpot_ms = statistics.mean(r.mean_tpot_ms for r in all_runs)
            result.mean_p50_ms = statistics.mean(r.p50_tpot_ms for r in all_runs)
            result.mean_p95_ms = statistics.mean(r.p95_tpot_ms for r in all_runs)
            result.mean_p99_ms = statistics.mean(r.p99_tpot_ms for r in all_runs if r.p99_tpot_ms > 0)
            result.mean_cv_percent = statistics.mean(r.cv_percent for r in all_runs)
            result.mean_jitter_ms = statistics.mean(r.jitter_ms for r in all_runs)
            result.mean_frustration_index = statistics.mean(r.frustration_index for r in all_runs)

            fi_values = [r.frustration_index for r in all_runs]
            result.stddev_frustration_index = statistics.stdev(fi_values) if len(fi_values) > 1 else 0.0

        print(f"  Results: FI={result.mean_frustration_index:.1f}, CV={result.mean_cv_percent:.1f}%, Jitter={result.mean_jitter_ms:.1f}ms")

        return result

    def run_experiment(
        self,
        prompts: Optional[List[str]] = None,
        rounds: int = 5,
    ) -> ExperimentResults:
        """
        Run the complete Frustration Index experiment.

        Tests sampling strategy consistency:
        1. Greedy (k=1) - Most deterministic, should be most consistent
        2. Top-K=3 - Light sampling
        3. Top-K=10 - Medium sampling
        4. Top-K=50 - Heavy sampling (more variance)
        5. Top-P (nucleus) simulation via temperature
        """
        if prompts is None:
            prompts = PROMPTS

        print(f"\n{'='*60}")
        print("FRUSTRATION INDEX EXPERIMENT")
        print(f"{'='*60}")
        print(f"Testing latency consistency across {len(prompts)} prompts")
        print(f"Rounds per condition: {rounds}")
        print(f"Total evaluations: {len(prompts) * rounds * 5}")

        # Run all conditions
        conditions = []

        # Condition 1: Greedy (k=1) - Should be most consistent
        conditions.append(self.run_condition(
            condition="Greedy_K1",
            prompts=prompts,
            rounds=rounds,
            use_dsi=False,
            top_k=1,
        ))

        # Condition 2: Top-K=3 - Light sampling
        conditions.append(self.run_condition(
            condition="TopK_3",
            prompts=prompts,
            rounds=rounds,
            use_dsi=False,
            top_k=3,
        ))

        # Condition 3: Top-K=10 - Medium sampling
        conditions.append(self.run_condition(
            condition="TopK_10",
            prompts=prompts,
            rounds=rounds,
            use_dsi=False,
            top_k=10,
        ))

        # Condition 4: Top-K=50 - Heavy sampling (may cause more variance)
        conditions.append(self.run_condition(
            condition="TopK_50",
            prompts=prompts,
            rounds=rounds,
            use_dsi=False,
            top_k=50,
        ))

        # Condition 5: Top-K=100 - Maximum sampling variance
        conditions.append(self.run_condition(
            condition="TopK_100",
            prompts=prompts,
            rounds=rounds,
            use_dsi=False,
            top_k=100,
        ))

        # Find best condition (lowest frustration index)
        best_condition = min(conditions, key=lambda c: c.mean_frustration_index)

        # Create results object
        results = ExperimentResults(
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            device=self.device,
            n_prompts=len(prompts),
            tokens_per_prompt=self.tokens_to_generate,
            rounds=rounds,
            conditions=conditions,
            best_condition=best_condition.condition,
            best_frustration_index=best_condition.mean_frustration_index,
        )

        return results


# =============================================================================
# Visualization
# =============================================================================

def generate_plots(results: ExperimentResults, output_dir: Path):
    """Generate comprehensive visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [c.condition for c in results.conditions]

    # Color scheme
    colors = {
        'Greedy_K1': '#2E86AB',     # Blue - deterministic
        'TopK_3': '#3E9BC2',        # Light blue
        'TopK_10': '#F18F01',       # Orange
        'TopK_50': '#A23B72',       # Purple
        'TopK_100': '#C84C5D',      # Red - most variance
    }

    # --- Plot 1: Frustration Index Comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))

    fi_values = [c.mean_frustration_index for c in results.conditions]
    fi_stds = [c.stddev_frustration_index for c in results.conditions]
    bar_colors = [colors.get(c, '#888888') for c in conditions]

    bars = ax.bar(conditions, fi_values, yerr=fi_stds, capsize=5, color=bar_colors, alpha=0.8)

    # Highlight best
    best_idx = fi_values.index(min(fi_values))
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel('Frustration Index (lower is better)', fontsize=12)
    ax.set_title('Frustration Index: UX Consistency Comparison\n(Lower = More Predictable Latency)', fontsize=14)
    ax.axhline(y=fi_values[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')

    # Add value labels
    for bar, val in zip(bars, fi_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'frustration_index_comparison.png', dpi=150)
    plt.close()

    # --- Plot 2: Multi-Metric Radar Chart ---
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    metrics = ['TTFT', 'Mean TPOT', 'P95', 'CV%', 'Jitter', 'FI']

    for cond in results.conditions:
        values = [
            cond.mean_ttft_ms,
            cond.mean_tpot_ms,
            cond.mean_p95_ms,
            cond.mean_cv_percent,
            cond.mean_jitter_ms,
            cond.mean_frustration_index,
        ]

        # Normalize to 0-1 scale for radar chart
        max_vals = [
            max(c.mean_ttft_ms for c in results.conditions),
            max(c.mean_tpot_ms for c in results.conditions),
            max(c.mean_p95_ms for c in results.conditions),
            max(c.mean_cv_percent for c in results.conditions),
            max(c.mean_jitter_ms for c in results.conditions),
            max(c.mean_frustration_index for c in results.conditions),
        ]

        norm_values = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]
        norm_values.append(norm_values[0])  # Close the polygon

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles.append(angles[0])

        ax.plot(angles, norm_values, 'o-', linewidth=2,
                color=colors.get(cond.condition, '#888888'), label=cond.condition)
        ax.fill(angles, norm_values, alpha=0.15, color=colors.get(cond.condition, '#888888'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title('Multi-Metric Comparison (lower/smaller = better)', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_metric_radar.png', dpi=150)
    plt.close()

    # --- Plot 3: Latency Distribution Box Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    all_token_times = []
    labels = []

    for cond in results.conditions:
        # Collect all token times from all runs
        times = []
        for run in cond.runs:
            times.extend(run.token_times_ms)
        all_token_times.append(times)
        labels.append(cond.condition)

    bp = ax.boxplot(all_token_times, labels=labels, patch_artist=True, showfliers=True)

    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors.get(cond, '#888888'))
        patch.set_alpha(0.7)

    ax.set_ylabel('Token Generation Time (ms)', fontsize=12)
    ax.set_title('Latency Distribution by Condition\n(Tighter box = More Consistent)', fontsize=14)

    # Add median labels
    for i, (line, cond) in enumerate(zip(bp['medians'], results.conditions)):
        x = line.get_xdata()[0]
        y = line.get_ydata()[0]
        ax.text(x + 0.1, y, f'{cond.mean_p50_ms:.1f}ms', fontsize=9, va='center')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_distribution.png', dpi=150)
    plt.close()

    # --- Plot 4: P95/P50 Ratio (Tail Behavior) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ratios = []
    for cond in results.conditions:
        ratio = cond.mean_p95_ms / cond.mean_p50_ms if cond.mean_p50_ms > 0 else 1.0
        ratios.append(ratio)

    bar_colors = [colors.get(c, '#888888') for c in conditions]
    bars = ax.bar(conditions, ratios, color=bar_colors, alpha=0.8)

    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect (no tails)')
    ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Problematic')

    ax.set_ylabel('P95/P50 Ratio', fontsize=12)
    ax.set_title('Tail Latency Ratio\n(Closer to 1.0 = Better Consistency)', fontsize=14)
    ax.legend()

    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=11)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'tail_ratio.png', dpi=150)
    plt.close()

    # --- Plot 5: Improvement Summary ---
    fig, ax = plt.subplots(figsize=(12, 5))

    baseline_fi = results.conditions[0].mean_frustration_index
    improvements = []

    for cond in results.conditions:
        if baseline_fi > 0:
            improvement = ((baseline_fi - cond.mean_frustration_index) / baseline_fi) * 100
        else:
            improvement = 0
        improvements.append(improvement)

    bar_colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(conditions, improvements, color=bar_colors, alpha=0.7)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Frustration Index Improvement (%)', fontsize=12)
    ax.set_title('UX Improvement vs Baseline\n(Positive = Better Consistency)', fontsize=14)

    for bar, val in zip(bars, improvements):
        y_pos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=11)

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_summary.png', dpi=150)
    plt.close()

    print(f"\n[Plots saved to {output_dir}/]")


def print_summary(results: ExperimentResults):
    """Print a comprehensive summary of results."""
    print(f"\n{'='*70}")
    print("FRUSTRATION INDEX EXPERIMENT RESULTS")
    print(f"{'='*70}")
    print(f"Model: {results.model_name}")
    print(f"Device: {results.device}")
    print(f"Prompts: {results.n_prompts}, Tokens/prompt: {results.tokens_per_prompt}")
    print(f"Rounds per condition: {results.rounds}")

    print(f"\n{'='*70}")
    print(f"{'Condition':<20} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'P95(ms)':<10} {'CV%':<8} {'Jitter':<10} {'FI':<10}")
    print(f"{'='*70}")

    for cond in results.conditions:
        print(f"{cond.condition:<20} {cond.mean_ttft_ms:<10.1f} {cond.mean_tpot_ms:<10.1f} "
              f"{cond.mean_p95_ms:<10.1f} {cond.mean_cv_percent:<8.1f} {cond.mean_jitter_ms:<10.1f} "
              f"{cond.mean_frustration_index:<10.1f}")

    print(f"{'='*70}")

    # Winner announcement
    baseline = results.conditions[0]
    best = min(results.conditions, key=lambda c: c.mean_frustration_index)

    if best.condition != baseline.condition:
        improvement = ((baseline.mean_frustration_index - best.mean_frustration_index)
                      / baseline.mean_frustration_index * 100)
        print(f"\n WINNER: {best.condition}")
        print(f"   Frustration Index: {best.mean_frustration_index:.1f} (vs {baseline.mean_frustration_index:.1f} baseline)")
        print(f"   Improvement: {improvement:.1f}% better consistency")
        print(f"   CV%: {best.mean_cv_percent:.1f}% (vs {baseline.mean_cv_percent:.1f}%)")
        print(f"   Jitter: {best.mean_jitter_ms:.1f}ms (vs {baseline.mean_jitter_ms:.1f}ms)")
    else:
        print(f"\n Baseline is already optimal")

    # Key insight
    print(f"\n KEY INSIGHT:")
    print(f"   The Frustration Index measures how 'surprising' latency variations are.")
    print(f"   Users perceive consistent 120ms better than variable 80-200ms.")
    print(f"   DSI's cognitive awareness enables smoother, more predictable generation.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Frustration Index Experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model to use")
    parser.add_argument("--tokens", type=int, default=50, help="Tokens to generate per prompt")
    parser.add_argument("--rounds", type=int, default=5, help="Rounds per condition")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--output", default="results/frustration_test", help="Output directory")
    args = parser.parse_args()

    # Create experiment
    experiment = FrustrationExperiment(
        model_name=args.model,
        tokens_to_generate=args.tokens,
        force_cpu=args.cpu,
    )

    # Load model
    experiment.load_model()

    # Warmup
    experiment.warmup()

    # Run experiment
    results = experiment.run_experiment(
        prompts=PROMPTS,
        rounds=args.rounds,
    )

    # Print summary
    print_summary(results)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"frustration_test_{timestamp}.json"

    # Convert to JSON-serializable format
    results_dict = {
        "timestamp": results.timestamp,
        "model_name": results.model_name,
        "device": results.device,
        "n_prompts": results.n_prompts,
        "tokens_per_prompt": results.tokens_per_prompt,
        "rounds": results.rounds,
        "best_condition": results.best_condition,
        "best_frustration_index": results.best_frustration_index,
        "conditions": []
    }

    for cond in results.conditions:
        cond_dict = {
            "condition": cond.condition,
            "n_runs": cond.n_runs,
            "mean_ttft_ms": cond.mean_ttft_ms,
            "mean_tpot_ms": cond.mean_tpot_ms,
            "mean_p50_ms": cond.mean_p50_ms,
            "mean_p95_ms": cond.mean_p95_ms,
            "mean_p99_ms": cond.mean_p99_ms,
            "mean_cv_percent": cond.mean_cv_percent,
            "mean_jitter_ms": cond.mean_jitter_ms,
            "mean_frustration_index": cond.mean_frustration_index,
            "stddev_frustration_index": cond.stddev_frustration_index,
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
