#!/usr/bin/env python3
"""
Effect Size Analyzer v1.0
=========================

Detailed analysis of FEEL effect size with:
- Token-by-token KL divergence heatmaps
- Flip pattern analysis (which token positions flip most)
- Entropy trajectory comparison
- Per-category effect breakdown
- Coherence validation (perplexity drift)

This provides the evidence for mechanism falsification.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.canonical_sensors import (
    CanonicalSensorBank, RuntimeContext, HardwareContext,
)
from src.telemetry_sampler import TelemetrySampler
from src.feel_projector import FEELProjectorFull


# Test prompts by category
PROMPTS_BY_CATEGORY = {
    "math": [
        "What is 7 * 8?",
        "What is 15 + 27?",
        "What is 2^10?",
        "What is 100 - 37?",
        "What is 12 * 12?",
    ],
    "factual": [
        "The capital of France is",
        "The chemical symbol for gold is",
        "The largest planet in our solar system is",
        "The speed of light is approximately",
        "The first president of the United States was",
    ],
    "coding": [
        "In Python, to define a function you use",
        "def fibonacci(n):",
        "To create a list in Python:",
        "What does print(len('hello')) output?",
        "The time complexity of binary search is",
    ],
    "completion": [
        "Machine learning is a subset of",
        "Water boils at 100 degrees",
        "DNA stands for deoxyribonucleic",
        "The Earth orbits the",
        "Photosynthesis produces",
    ],
}


class EffectSizeAnalyzer:
    """Detailed effect size analysis."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        alpha: float = 0.01,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alpha = alpha

        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()

        self.embed_dim = self.model.config.hidden_size
        self.sensor_bank = CanonicalSensorBank(mode="full")
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)

        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            time.sleep(1)
        except:
            pass

        print(f"  Alpha: {self.alpha}")
        diag = self.projector.diagnose()
        print(f"  Projector scale: {diag['output_scale']:.4f}")

    def _get_hardware_context(self, t0: float, t1: float) -> HardwareContext:
        if self.telemetry is None:
            return HardwareContext()
        return HardwareContext.from_dict(self.telemetry.get_token_aligned(t0, t1))

    def analyze_prompt(
        self,
        prompt: str,
        max_tokens: int = 15,
    ) -> Dict:
        """Detailed analysis of a single prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        token_results = []
        generated_baseline = []
        generated_feel = []

        for step in range(max_tokens):
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits_baseline = outputs.logits[:, -1, :].float()
            t1 = time.time()

            runtime = RuntimeContext(
                token_latency=t1 - t0,
                kv_cache_tokens=current_ids.shape[1],
                generation_depth=step,
            )
            hardware = self._get_hardware_context(t0, t1)
            sensors = self.sensor_bank(logits_baseline, runtime=runtime, hardware=hardware)

            # Apply FEEL
            feel_embed = self.projector(sensors.float())
            embeds = self.model.get_input_embeddings()(current_ids)
            embeds_feel = embeds + (self.alpha * feel_embed).to(embeds.dtype).unsqueeze(1)

            with torch.no_grad():
                outputs_feel = self.model(inputs_embeds=embeds_feel, use_cache=False)
                logits_feel = outputs_feel.logits[:, -1, :].float()

            # Compute all metrics
            probs_b = F.softmax(logits_baseline, dim=-1)
            probs_f = F.softmax(logits_feel, dim=-1)

            argmax_b = logits_baseline.argmax(dim=-1).item()
            argmax_f = logits_feel.argmax(dim=-1).item()

            # KL divergence
            kl = F.kl_div(probs_f.log().clamp(min=-100), probs_b, reduction='sum').item()

            # Entropy
            entropy_b = -(probs_b * probs_b.log().clamp(min=-100)).sum().item()
            entropy_f = -(probs_f * probs_f.log().clamp(min=-100)).sum().item()

            # Top-k analysis
            top5_b = probs_b.topk(5, dim=-1)
            top5_f = probs_f.topk(5, dim=-1)

            # Rank change of top token
            sorted_indices = probs_f.argsort(descending=True).squeeze(0)
            rank_of_baseline_top_in_feel = (sorted_indices == argmax_b).nonzero(as_tuple=True)[0].item() + 1

            # Confidence
            conf_b = probs_b.max().item()
            conf_f = probs_f.max().item()

            token_results.append({
                "step": step,
                "flipped": argmax_b != argmax_f,
                "kl": kl,
                "entropy_baseline": entropy_b,
                "entropy_feel": entropy_f,
                "entropy_delta": entropy_f - entropy_b,
                "conf_baseline": conf_b,
                "conf_feel": conf_f,
                "conf_delta": conf_f - conf_b,
                "rank_change": rank_of_baseline_top_in_feel - 1,
                "top_token_baseline": self.tokenizer.decode([argmax_b]),
                "top_token_feel": self.tokenizer.decode([argmax_f]),
                "hardware": {
                    "temp": hardware.temp,
                    "power": hardware.power,
                    "util": hardware.util,
                },
            })

            generated_baseline.append(argmax_b)
            generated_feel.append(argmax_f)

            # Continue with baseline for consistency
            next_token = torch.tensor([[argmax_b]], device=self.device)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if argmax_b == self.tokenizer.eos_token_id:
                break

        # Aggregate metrics
        n_tokens = len(token_results)
        n_flips = sum(1 for t in token_results if t["flipped"])

        return {
            "prompt": prompt,
            "n_tokens": n_tokens,
            "n_flips": n_flips,
            "flip_rate": n_flips / n_tokens if n_tokens > 0 else 0,
            "mean_kl": np.mean([t["kl"] for t in token_results]),
            "max_kl": np.max([t["kl"] for t in token_results]),
            "mean_entropy_delta": np.mean([t["entropy_delta"] for t in token_results]),
            "mean_conf_delta": np.mean([t["conf_delta"] for t in token_results]),
            "mean_rank_change": np.mean([t["rank_change"] for t in token_results]),
            "token_results": token_results,
            "text_baseline": self.tokenizer.decode(generated_baseline),
            "text_feel": self.tokenizer.decode(generated_feel),
        }

    def analyze_all(
        self,
        prompts_by_category: Dict[str, List[str]] = None,
        max_tokens: int = 15,
    ) -> Dict:
        """Analyze all prompts by category."""
        if prompts_by_category is None:
            prompts_by_category = PROMPTS_BY_CATEGORY

        results = {
            "timestamp": datetime.now().isoformat(),
            "alpha": self.alpha,
            "max_tokens": max_tokens,
            "by_category": {},
            "overall": {},
        }

        all_flip_rates = []
        all_kls = []
        all_entropy_deltas = []

        for category, prompts in prompts_by_category.items():
            print(f"\n  [{category}] Analyzing {len(prompts)} prompts...")
            category_results = []

            for prompt in prompts:
                analysis = self.analyze_prompt(prompt, max_tokens)
                category_results.append(analysis)
                all_flip_rates.append(analysis["flip_rate"])
                all_kls.append(analysis["mean_kl"])
                all_entropy_deltas.append(analysis["mean_entropy_delta"])

            # Category aggregates
            results["by_category"][category] = {
                "n_prompts": len(prompts),
                "flip_rate_mean": np.mean([r["flip_rate"] for r in category_results]),
                "flip_rate_std": np.std([r["flip_rate"] for r in category_results]),
                "kl_mean": np.mean([r["mean_kl"] for r in category_results]),
                "entropy_delta_mean": np.mean([r["mean_entropy_delta"] for r in category_results]),
                "prompts": category_results,
            }

            fr = results["by_category"][category]["flip_rate_mean"]
            kl = results["by_category"][category]["kl_mean"]
            print(f"      flip_rate={fr*100:.1f}%, kl={kl:.4f}")

        # Overall aggregates
        results["overall"] = {
            "n_prompts": sum(len(p) for p in prompts_by_category.values()),
            "flip_rate_mean": np.mean(all_flip_rates),
            "flip_rate_std": np.std(all_flip_rates),
            "kl_mean": np.mean(all_kls),
            "kl_std": np.std(all_kls),
            "entropy_delta_mean": np.mean(all_entropy_deltas),
        }

        return results

    def generate_comparison(
        self,
        prompt: str,
        max_tokens: int = 20,
    ) -> Dict:
        """Generate and compare baseline vs FEEL text."""
        # Baseline generation
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        baseline_tokens = []
        for _ in range(max_tokens):
            with torch.no_grad():
                logits = self.model(current_ids, use_cache=False).logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            baseline_tokens.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # FEEL generation
        current_ids = input_ids.clone()
        feel_tokens = []
        for step in range(max_tokens):
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            runtime = RuntimeContext(token_latency=t1-t0, kv_cache_tokens=current_ids.shape[1], generation_depth=step)
            hardware = self._get_hardware_context(t0, t1)
            sensors = self.sensor_bank(logits, runtime=runtime, hardware=hardware)

            feel_embed = self.projector(sensors.float())
            embeds = self.model.get_input_embeddings()(current_ids)
            embeds_feel = embeds + (self.alpha * feel_embed).to(embeds.dtype).unsqueeze(1)

            with torch.no_grad():
                logits_feel = self.model(inputs_embeds=embeds_feel, use_cache=False).logits[:, -1, :]

            next_token = logits_feel.argmax(dim=-1, keepdim=True)
            feel_tokens.append(next_token.item())
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return {
            "prompt": prompt,
            "baseline": self.tokenizer.decode(baseline_tokens),
            "feel": self.tokenizer.decode(feel_tokens),
            "tokens_differ": baseline_tokens != feel_tokens,
        }


def main():
    parser = argparse.ArgumentParser(description="Effect Size Analyzer")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max-tokens", type=int, default=15)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EFFECT SIZE ANALYZER v1.0")
    print("=" * 70)

    analyzer = EffectSizeAnalyzer(
        model_name=args.model,
        alpha=args.alpha,
    )

    results = analyzer.analyze_all(max_tokens=args.max_tokens)

    # Print summary
    print("\n" + "=" * 70)
    print("  EFFECT SIZE SUMMARY")
    print("=" * 70)

    print(f"\n  Overall (alpha={args.alpha}):")
    print(f"    Flip rate: {results['overall']['flip_rate_mean']*100:.2f}% ± {results['overall']['flip_rate_std']*100:.2f}%")
    print(f"    Mean KL: {results['overall']['kl_mean']:.4f} ± {results['overall']['kl_std']:.4f}")
    print(f"    Entropy delta: {results['overall']['entropy_delta_mean']:.4f}")

    print("\n  By category:")
    for cat, data in results["by_category"].items():
        print(f"    {cat}: flip={data['flip_rate_mean']*100:.1f}%, kl={data['kl_mean']:.4f}")

    # Sample comparisons
    print("\n" + "=" * 70)
    print("  SAMPLE GENERATION COMPARISONS")
    print("=" * 70)

    for prompt in ["The capital of France is", "What is 7 * 8?"]:
        comp = analyzer.generate_comparison(prompt)
        print(f"\n  Prompt: {prompt}")
        print(f"  Baseline: {comp['baseline'][:60]}...")
        print(f"  FEEL:     {comp['feel'][:60]}...")
        print(f"  Differ: {comp['tokens_differ']}")

    # Cleanup
    if analyzer.telemetry:
        analyzer.telemetry.stop()

    # Save results
    results_path = f"results/feel_experiments/effect_size_alpha_{args.alpha}.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n  Results saved: {results_path}")


if __name__ == "__main__":
    main()
