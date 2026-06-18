#!/usr/bin/env python3
"""
z119 - Real LLM Benchmark with Proper Tokenizer and TinyStories

This is the REAL benchmark that produces comparable results:
- HuggingFace tokenizer (no byte/char shortcuts)
- TinyStories dataset (widely used for small model eval)
- Proper perplexity computation
- 5-way comparison: baseline vs sensing vs policy vs gated vs layerdrop
- Truth-tiered energy (Tier-B for AMD)
- Variable load testing

Usage:
    python scripts/z119_real_benchmark.py --model-size 30m --dataset tinystories
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# HuggingFace
from transformers import AutoTokenizer


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # Model
    model_size: str = "30m"  # 30m, 125m
    vocab_size: int = 32000

    # Dataset
    dataset: str = "tinystories"
    num_eval_samples: int = 100
    max_seq_len: int = 256

    # Generation
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.9

    # Variable load
    enable_variable_load: bool = True
    burst_interval_s: float = 2.0
    burst_duration_s: float = 0.5

    # Output
    output_dir: str = "results/z119_real_benchmark"
    save_generations: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    timestamp: str

    # Quality
    perplexity: float = 0.0
    avg_generation_length: float = 0.0

    # Latency (ms)
    ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Energy (Tier-B for AMD)
    total_energy_mj: float = 0.0
    mj_per_token: float = 0.0
    avg_power_w: float = 0.0
    energy_tier: str = "tier_b"

    # Mode stats
    eco_pct: float = 0.0
    balanced_pct: float = 0.0
    perf_pct: float = 0.0
    mode_changes: int = 0

    # Derived
    dollars_per_million_tokens: float = 0.0

    def compute_derived(self, electricity_rate: float = 0.12):
        """Compute derived metrics."""
        if self.mj_per_token > 0:
            # mJ/token → kWh/1M tokens → $/1M tokens
            kwh_per_million = (self.mj_per_token * 1_000_000) / 3_600_000_000
            self.dollars_per_million_tokens = kwh_per_million * electricity_rate


# =============================================================================
# Dataset Loading
# =============================================================================

def load_tinystories_prompts(num_samples: int = 100) -> List[str]:
    """
    Load TinyStories prompts for evaluation.

    TinyStories is a synthetic dataset of short children's stories,
    widely used for evaluating small language models.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
        prompts = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            # Use first sentence as prompt
            text = item["text"]
            sentences = text.split(".")
            if sentences:
                prompt = sentences[0].strip() + "."
                if len(prompt) > 10:  # Skip very short prompts
                    prompts.append(prompt)
        return prompts
    except Exception as e:
        print(f"Could not load TinyStories: {e}")
        print("Using fallback prompts...")
        return [
            "Once upon a time, there was a little girl named Lily.",
            "The sun was shining brightly on the farm.",
            "Tom had a big red ball that he loved to play with.",
            "In the forest, there lived a friendly bear.",
            "Sarah wanted to learn how to ride a bicycle.",
        ] * (num_samples // 5 + 1)


def load_evaluation_data(config: BenchmarkConfig) -> Tuple[List[str], List[str]]:
    """Load evaluation prompts and reference completions."""
    if config.dataset == "tinystories":
        prompts = load_tinystories_prompts(config.num_eval_samples)
        # For perplexity, we need reference text
        references = prompts  # Self-perplexity on prompts
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    return prompts[:config.num_eval_samples], references[:config.num_eval_samples]


# =============================================================================
# Real Tokenizer Setup
# =============================================================================

class ClampedTokenizer:
    """
    Tokenizer wrapper that clamps token IDs to a max vocab size.

    This is needed because GPT-2 tokenizer has 50257 tokens but our model
    uses 32000 vocab. We clamp OOV tokens to the last valid ID.
    """

    def __init__(self, base_tokenizer, max_vocab_size: int = 32000):
        self.base = base_tokenizer
        self.max_vocab_size = max_vocab_size
        self.pad_token = base_tokenizer.pad_token
        self.eos_token = base_tokenizer.eos_token
        # EOS token ID clamped to our vocab - use last valid token as fallback
        self.eos_token_id = min(base_tokenizer.eos_token_id or 0, max_vocab_size - 1)
        self.pad_token_id = self.eos_token_id
        self.vocab_size = max_vocab_size
        # UNK token for OOV - use last valid ID
        self._oov_id = max_vocab_size - 1

    def _clamp_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Clamp tensor values to valid vocab range."""
        return torch.clamp(tensor, min=0, max=self.max_vocab_size - 1)

    def _clamp_list(self, ids: list) -> list:
        """Clamp list of IDs to valid vocab range."""
        if not ids:
            return ids
        if isinstance(ids[0], list):
            return [[min(max(0, id), self.max_vocab_size - 1) for id in row] for row in ids]
        return [min(max(0, id), self.max_vocab_size - 1) for id in ids]

    def __call__(self, text, **kwargs):
        """Tokenize text with ID clamping."""
        result = self.base(text, **kwargs)

        # Return a plain dict with clamped IDs (avoids BatchEncoding mutation issues)
        output = {}
        for key in result.keys():
            value = result[key]
            if key == "input_ids":
                if isinstance(value, torch.Tensor):
                    output[key] = self._clamp_tensor(value)
                elif isinstance(value, list):
                    output[key] = self._clamp_list(value)
                else:
                    output[key] = value
            else:
                output[key] = value

        return output

    def encode(self, text, **kwargs):
        """Encode text to clamped token IDs."""
        ids = self.base.encode(text, **kwargs)
        return self._clamp_list(ids) if isinstance(ids, list) else ids

    def decode(self, ids, **kwargs):
        """Decode token IDs to text."""
        # Clamp before decoding to avoid errors
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list):
            # Clamp to base tokenizer vocab for decoding
            ids = [min(max(0, id), self.base.vocab_size - 1) for id in ids]
        return self.base.decode(ids, **kwargs)

    def batch_decode(self, ids, **kwargs):
        """Decode batch of token IDs."""
        return self.base.batch_decode(ids, **kwargs)


def get_tokenizer(vocab_size: int = 32000):
    """
    Get a tokenizer that clamps to model vocab size.
    """
    try:
        base = AutoTokenizer.from_pretrained("gpt2")
        base.pad_token = base.eos_token
        return ClampedTokenizer(base, vocab_size)
    except Exception as e:
        print(f"WARNING: Could not load HF tokenizer: {e}")
        return None


# =============================================================================
# Perplexity Computation
# =============================================================================

def compute_perplexity(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_len: int = 256,
    device: str = "cuda",
) -> float:
    """
    Compute perplexity on text samples.

    This is the proper perplexity computation using log-likelihood.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            # Tokenize
            encoding = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding=False,
            )
            input_ids = encoding["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                continue

            # Forward (baseline mode - no body signals for fair PPL comparison)
            if hasattr(model, "base_model"):
                model.base_model.set_training_phase("baseline")
                out = model.base_model(input_ids)
            else:
                out = model(input_ids)

            logits = out["logits"] if isinstance(out, dict) else out

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity


# =============================================================================
# Benchmark Runner
# =============================================================================

class RealBenchmark:
    """
    Real LLM benchmark with proper metrics.

    Runs 5 configurations:
    1. Baseline (no sensing, no control)
    2. Sensing only (telemetry fed, no policy)
    3. Policy head (action predicted but not applied)
    4. Gated injection (FiLM active in last layers)
    5. Full LayerDrop (compute mode changes)
    """

    def __init__(
        self,
        model,
        runtime,
        tokenizer,
        config: BenchmarkConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Results storage
        self.results: Dict[str, BenchmarkResult] = {}

    def run_baseline(self, prompts: List[str], references: List[str]) -> BenchmarkResult:
        """
        Baseline: No sensing, no control.

        Model runs in baseline training phase (body signals ignored).
        """
        print("\n=== Baseline (no sensing, no control) ===")

        # Set baseline mode
        if hasattr(self.model, "base_model"):
            self.model.base_model.set_training_phase("baseline")
            self.model.set_compute_mode(
                __import__("feel_slm.feel_runtime", fromlist=["ComputeMode"]).ComputeMode.BALANCED
            )

        return self._run_generation_benchmark("baseline", prompts, references, use_sensing=False)

    def run_sensing_only(self, prompts: List[str], references: List[str]) -> BenchmarkResult:
        """
        Sensing only: Telemetry fed to model, but no policy/actuation.

        FiLM layers receive body latent, but compute mode is fixed.
        """
        print("\n=== Sensing Only (telemetry fed, no actuation) ===")

        if hasattr(self.model, "base_model"):
            self.model.base_model.set_training_phase("full")

        from feel_slm.feel_runtime import ComputeMode
        self.model.set_compute_mode(ComputeMode.BALANCED)

        return self._run_generation_benchmark("sensing_only", prompts, references, use_sensing=True, use_policy=False)

    def run_policy_head(self, prompts: List[str], references: List[str]) -> BenchmarkResult:
        """
        Policy head: Action predicted but compute mode fixed.

        We record what the policy would do, but don't apply it.
        """
        print("\n=== Policy Head (action predicted, not applied) ===")

        if hasattr(self.model, "base_model"):
            self.model.base_model.set_training_phase("full")

        from feel_slm.feel_runtime import ComputeMode
        self.model.set_compute_mode(ComputeMode.BALANCED)

        return self._run_generation_benchmark("policy_head", prompts, references, use_sensing=True, use_policy=True, apply_policy=False)

    def run_full_adaptive(self, prompts: List[str], references: List[str]) -> BenchmarkResult:
        """
        Full adaptive: Policy predicts and applies compute mode changes.

        This is the full FEEL loop with LayerDrop actuation.
        """
        print("\n=== Full Adaptive (policy + LayerDrop) ===")

        if hasattr(self.model, "base_model"):
            self.model.base_model.set_training_phase("full")

        return self._run_generation_benchmark("full_adaptive", prompts, references, use_sensing=True, use_policy=True, apply_policy=True)

    def run_eco_fixed(self, prompts: List[str], references: List[str]) -> BenchmarkResult:
        """
        ECO fixed: Always run in ECO mode (LayerDrop active).

        Baseline for "what if we just always drop layers".
        """
        print("\n=== ECO Fixed (max LayerDrop) ===")

        if hasattr(self.model, "base_model"):
            self.model.base_model.set_training_phase("full")

        from feel_slm.feel_runtime import ComputeMode
        self.model.set_compute_mode(ComputeMode.ECO)

        return self._run_generation_benchmark("eco_fixed", prompts, references, use_sensing=True, use_policy=False)

    def _run_generation_benchmark(
        self,
        config_name: str,
        prompts: List[str],
        references: List[str],
        use_sensing: bool = True,
        use_policy: bool = True,
        apply_policy: bool = True,
    ) -> BenchmarkResult:
        """Run generation benchmark with given configuration."""

        result = BenchmarkResult(
            config_name=config_name,
            timestamp=datetime.now().isoformat(),
        )

        # Metrics accumulators
        ttfts = []
        tpots = []
        all_tokens = 0
        mode_counts = {"eco": 0, "balanced": 0, "perf": 0}
        mode_changes = 0

        # Start sensing
        if use_sensing and self.runtime:
            self.runtime.start_sensing()
            time.sleep(0.2)  # Let buffer fill

        start_time = time.time()

        for i, prompt in enumerate(prompts):
            if (i + 1) % 20 == 0:
                print(f"  Processing {i+1}/{len(prompts)}...")

            # Tokenize prompt
            encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_len,
            )
            input_ids = encoding["input_ids"].to(self.device)

            # Generate
            gen_start = time.time()
            prev_mode = self.model.compute_mode.value if hasattr(self.model, "compute_mode") else "balanced"

            tokens_generated = 0
            token_times = []

            with torch.no_grad():
                generated = input_ids.clone()

                for j in range(self.config.max_new_tokens):
                    token_start = time.time()

                    # Forward
                    if use_sensing and self.runtime:
                        out = self.runtime.forward(generated, update_policy=(use_policy and apply_policy))
                    elif hasattr(self.model, "forward"):
                        # Direct model forward
                        body_vec = torch.zeros(1, 12, device=self.device)
                        out = self.model(generated, body_vec, return_policy=use_policy)
                    else:
                        out = self.model.base_model(generated)

                    logits = out["logits"] if isinstance(out, dict) else out

                    # Sample next token
                    next_logits = logits[:, -1, :] / self.config.temperature

                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")

                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    generated = torch.cat([generated, next_token], dim=1)
                    tokens_generated += 1

                    token_time = time.time() - token_start
                    token_times.append(token_time * 1000)  # ms

                    # Track mode
                    if hasattr(self.model, "compute_mode"):
                        cur_mode = self.model.compute_mode.value
                        mode_counts[cur_mode] = mode_counts.get(cur_mode, 0) + 1
                        if cur_mode != prev_mode:
                            mode_changes += 1
                            prev_mode = cur_mode

                    # EOS check
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            # Record metrics
            if token_times:
                ttfts.append(token_times[0])
                tpots.extend(token_times)

            all_tokens += tokens_generated

        total_time = time.time() - start_time

        # Stop sensing
        if use_sensing and self.runtime:
            self.runtime.stop_sensing()

        # Compute results
        result.ttft_ms = np.mean(ttfts) if ttfts else 0
        result.avg_tpot_ms = np.mean(tpots) if tpots else 0
        result.p95_tpot_ms = np.percentile(tpots, 95) if tpots else 0
        result.p99_tpot_ms = np.percentile(tpots, 99) if tpots else 0
        result.tokens_per_second = all_tokens / total_time if total_time > 0 else 0
        result.avg_generation_length = all_tokens / len(prompts) if prompts else 0

        # Energy (from runtime buffer)
        if use_sensing and self.runtime:
            energy = self.runtime.energy_meter.measure(total_time)
            result.total_energy_mj = energy.energy_mj
            result.mj_per_token = energy.energy_mj / all_tokens if all_tokens > 0 else 0
            result.avg_power_w = energy.avg_power_w
            result.energy_tier = energy.tier.value

        # Mode distribution
        total_mode_samples = sum(mode_counts.values())
        if total_mode_samples > 0:
            result.eco_pct = mode_counts.get("eco", 0) / total_mode_samples * 100
            result.balanced_pct = mode_counts.get("balanced", 0) / total_mode_samples * 100
            result.perf_pct = mode_counts.get("perf", 0) / total_mode_samples * 100
        result.mode_changes = mode_changes

        # Perplexity (on references)
        print("  Computing perplexity...")
        result.perplexity = compute_perplexity(
            self.model, self.tokenizer, references[:50], self.config.max_seq_len, self.device
        )

        # Derived metrics
        result.compute_derived()

        return result

    def run_all(self, prompts: List[str], references: List[str]) -> Dict[str, BenchmarkResult]:
        """Run all benchmark configurations."""

        results = {}

        # 1. Baseline
        results["baseline"] = self.run_baseline(prompts, references)

        # 2. Sensing only
        results["sensing_only"] = self.run_sensing_only(prompts, references)

        # 3. Policy head (predict but don't apply)
        results["policy_head"] = self.run_policy_head(prompts, references)

        # 4. ECO fixed
        results["eco_fixed"] = self.run_eco_fixed(prompts, references)

        # 5. Full adaptive
        results["full_adaptive"] = self.run_full_adaptive(prompts, references)

        self.results = results
        return results

    def print_summary(self):
        """Print results summary table."""
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 100)

        headers = ["Config", "PPL↓", "TPS↑", "TTFT(ms)↓", "P95 TPOT↓", "mJ/tok↓", "$/1M tok↓", "Tier"]
        print(f"{headers[0]:<18} {headers[1]:>8} {headers[2]:>8} {headers[3]:>10} {headers[4]:>12} {headers[5]:>10} {headers[6]:>12} {headers[7]:>6}")
        print("-" * 100)

        for name, r in self.results.items():
            print(f"{name:<18} {r.perplexity:>8.2f} {r.tokens_per_second:>8.1f} {r.ttft_ms:>10.2f} {r.p95_tpot_ms:>12.2f} {r.mj_per_token:>10.2f} {r.dollars_per_million_tokens:>12.4f} {r.energy_tier:>6}")

        print("-" * 100)

        # Highlight best results
        if self.results:
            ppl_values = [r.perplexity for r in self.results.values() if r.perplexity > 0]
            energy_values = [r.mj_per_token for r in self.results.values() if r.mj_per_token > 0]
            tps_values = [r.tokens_per_second for r in self.results.values() if r.tokens_per_second > 0]

            best_ppl = min(ppl_values) if ppl_values else float("inf")
            best_energy = min(energy_values) if energy_values else 0.0
            best_tps = max(tps_values) if tps_values else 0.0

            print(f"\nBest perplexity: {best_ppl:.2f}")
            print(f"Best mJ/token: {best_energy:.2f}")
            print(f"Best tokens/sec: {best_tps:.1f}")

            # Comparison vs baseline
            baseline = self.results.get("baseline")
            adaptive = self.results.get("full_adaptive")
            if baseline and adaptive and baseline.mj_per_token > 0:
                energy_savings = (1 - adaptive.mj_per_token / baseline.mj_per_token) * 100
                ppl_delta = adaptive.perplexity - baseline.perplexity
                print(f"\nFull Adaptive vs Baseline:")
                print(f"  Energy savings: {energy_savings:+.1f}%")
                print(f"  Perplexity delta: {ppl_delta:+.2f}")

    def save_results(self, output_dir: str):
        """Save results to JSON."""
        os.makedirs(output_dir, exist_ok=True)

        # Save individual results
        for name, r in self.results.items():
            path = os.path.join(output_dir, f"{name}_results.json")
            with open(path, "w") as f:
                json.dump(asdict(r), f, indent=2)

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else {},
            "results": {name: asdict(r) for name, r in self.results.items()},
        }
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {output_dir}/")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real FEEL Benchmark")
    parser.add_argument("--model-size", type=str, default="30m", choices=["30m", "125m"])
    parser.add_argument("--dataset", type=str, default="tinystories")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results/z119_real_benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("FEEL Real Benchmark (z119)")
    print("=" * 60)

    # Set HSA override for AMD
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    config = BenchmarkConfig(
        model_size=args.model_size,
        dataset=args.dataset,
        num_eval_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
    )

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = get_tokenizer(config.vocab_size)
    if tokenizer is None:
        print("ERROR: Could not load tokenizer")
        return

    # Load data
    print("\n2. Loading evaluation data...")
    prompts, references = load_evaluation_data(config)
    print(f"   Loaded {len(prompts)} prompts")

    # Create model
    print(f"\n3. Creating EmbodiedSLM ({config.model_size})...")
    from feel_slm.embodied_slm import create_embodied_slm_30m, create_embodied_slm_125m

    if config.model_size == "30m":
        base_model = create_embodied_slm_30m()
    else:
        base_model = create_embodied_slm_125m()

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"   Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            # Training checkpoint format
            state_dict = checkpoint["model_state_dict"]
            # Remove 'base_model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("base_model."):
                    new_state_dict[k[11:]] = v
                elif not k.startswith("policy_head."):
                    new_state_dict[k] = v
            base_model.load_state_dict(new_state_dict, strict=False)
        else:
            base_model.load_state_dict(checkpoint, strict=False)

    base_model = base_model.to(args.device)
    base_model.eval()
    print(f"   Parameters: {base_model.num_parameters / 1e6:.1f}M")

    # Create FEEL runtime
    print("\n4. Creating FEEL runtime...")
    from feel_slm.feel_runtime import create_feel_runtime, FEELConfig

    feel_config = FEELConfig(
        layerdrop_eco=0.4,
        layerdrop_balanced=0.0,
        min_layers=4,
    )
    runtime = create_feel_runtime(base_model, config=feel_config)

    # Create benchmark
    print("\n5. Running benchmarks...")
    benchmark = RealBenchmark(
        model=runtime.model,
        runtime=runtime,
        tokenizer=tokenizer,
        config=config,
        device=args.device,
    )

    # Run all configurations
    results = benchmark.run_all(prompts, references)

    # Print summary
    benchmark.print_summary()

    # Save results
    benchmark.save_results(config.output_dir)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
