#!/usr/bin/env python3
"""
Alpha Sweep Effect Size Analysis
================================

This is the CRITICAL next experiment after v7.0's null baseline.

Purpose: Find the alpha range where FEEL actually changes token decisions.

Key Metrics:
- flip_rate: % tokens where argmax_FEEL ≠ argmax_baseline
- kl_per_token: Mean KL divergence FEEL vs baseline
- top2_margin: Gap between top-1 and top-2 logits (sensitivity indicator)
- coherence_proxy: Perplexity under teacher forcing (should not collapse)
- accuracy: Quick accuracy check (even small sample informative)

Target: Find α where:
- flip_rate ∈ [0.5%, 5%] (some effect, not chaos)
- coherence doesn't collapse (perplexity < 2× baseline)

Usage:
    python scripts/alpha_sweep_effect_size.py
    python scripts/alpha_sweep_effect_size.py --quick    # Fast (20 prompts)
    python scripts/alpha_sweep_effect_size.py --alphas "0,1e-4,1e-3,1e-2,1e-1"
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.canonical_sensors import CanonicalSensorBank


# ============================================================
# Stratified Test Prompts (50 for effect size gate)
# ============================================================

EFFECT_SIZE_PROMPTS = [
    # Math (12)
    {"prompt": "What is 7 * 8?", "answer": "56", "category": "math"},
    {"prompt": "What is 15 + 27?", "answer": "42", "category": "math"},
    {"prompt": "What is 100 - 37?", "answer": "63", "category": "math"},
    {"prompt": "What is 12 * 12?", "answer": "144", "category": "math"},
    {"prompt": "What is 81 / 9?", "answer": "9", "category": "math"},
    {"prompt": "What is 2^8?", "answer": "256", "category": "math"},
    {"prompt": "What is (7 + 3) * 5?", "answer": "50", "category": "math"},
    {"prompt": "What is 3^3 + 4^2?", "answer": "43", "category": "math"},
    {"prompt": "What is 0.5 * 100?", "answer": "50", "category": "math"},
    {"prompt": "What is 1000 / 8?", "answer": "125", "category": "math"},
    {"prompt": "What is 17 * 6?", "answer": "102", "category": "math"},
    {"prompt": "What is 2^10?", "answer": "1024", "category": "math"},
    # Factual (13)
    {"prompt": "The capital of France is", "answer": "paris", "category": "factual"},
    {"prompt": "The chemical symbol for gold is", "answer": "au", "category": "factual"},
    {"prompt": "The largest planet is", "answer": "jupiter", "category": "factual"},
    {"prompt": "Water boils at 100 degrees", "answer": "celsius", "category": "factual"},
    {"prompt": "The speed of light is approximately", "answer": "300", "category": "factual"},
    {"prompt": "DNA stands for deoxyribonucleic", "answer": "acid", "category": "factual"},
    {"prompt": "The first president of the United States was", "answer": "washington", "category": "factual"},
    {"prompt": "The atomic number of carbon is", "answer": "6", "category": "factual"},
    {"prompt": "The Eiffel Tower is located in", "answer": "paris", "category": "factual"},
    {"prompt": "Photosynthesis produces", "answer": "oxygen", "category": "factual"},
    {"prompt": "The human body has how many bones?", "answer": "206", "category": "factual"},
    {"prompt": "Python is a programming", "answer": "language", "category": "factual"},
    {"prompt": "The powerhouse of the cell is the", "answer": "mitochondria", "category": "factual"},
    # Coding (12)
    {"prompt": "In Python, how do you create an empty list?", "answer": "[]", "category": "coding"},
    {"prompt": "What Python keyword defines a function?", "answer": "def", "category": "coding"},
    {"prompt": "What does print(3 + 4) output?", "answer": "7", "category": "coding"},
    {"prompt": "What Python function returns list length?", "answer": "len", "category": "coding"},
    {"prompt": "What data structure uses LIFO?", "answer": "stack", "category": "coding"},
    {"prompt": "What does print(10 // 3) output?", "answer": "3", "category": "coding"},
    {"prompt": "What does print(10 % 3) output?", "answer": "1", "category": "coding"},
    {"prompt": "What Python method adds to a list?", "answer": "append", "category": "coding"},
    {"prompt": "What is the time complexity of binary search?", "answer": "log", "category": "coding"},
    {"prompt": "What does CPU stand for?", "answer": "central processing unit", "category": "coding"},
    {"prompt": "What does print([1,2,3][0]) output?", "answer": "1", "category": "coding"},
    {"prompt": "What does print(type(42)) include?", "answer": "int", "category": "coding"},
    # Logic (13)
    {"prompt": "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?", "answer": "yes", "category": "logic"},
    {"prompt": "If A then B. Not B. What about A?", "answer": "not", "category": "logic"},
    {"prompt": "All squares are rectangles. Do all squares have 4 sides?", "answer": "yes", "category": "logic"},
    {"prompt": "Some dogs are brown. Some brown things are chairs. Are some dogs chairs?", "answer": "no", "category": "logic"},
    {"prompt": "If it rains, the ground is wet. Ground is wet. Is it raining?", "answer": "not necessarily", "category": "logic"},
    {"prompt": "All birds have wings. Penguins are birds. Do penguins have wings?", "answer": "yes", "category": "logic"},
    {"prompt": "No fish can fly. Salmon is a fish. Can salmon fly?", "answer": "no", "category": "logic"},
    {"prompt": "If P implies Q, and Q implies R, does P imply R?", "answer": "yes", "category": "logic"},
    {"prompt": "True AND False equals?", "answer": "false", "category": "logic"},
    {"prompt": "True OR False equals?", "answer": "true", "category": "logic"},
    {"prompt": "NOT True equals?", "answer": "false", "category": "logic"},
    {"prompt": "Is 0 equal to False in Python?", "answer": "yes", "category": "logic"},
    {"prompt": "Is an empty list truthy in Python?", "answer": "no", "category": "logic"},
]


# ============================================================
# Result Data Classes
# ============================================================

@dataclass
class TokenMetrics:
    """Per-token metrics for effect size analysis."""
    position: int
    baseline_top1: int
    feel_top1: int
    flipped: bool
    kl_divergence: float
    top2_margin_baseline: float
    top2_margin_feel: float
    baseline_entropy: float
    feel_entropy: float


@dataclass
class PromptMetrics:
    """Per-prompt aggregated metrics."""
    prompt: str
    category: str
    alpha: float
    flip_rate: float           # % tokens where argmax changed
    mean_kl: float             # Mean KL per token
    mean_top2_margin: float    # Mean gap between top-1 and top-2
    baseline_perplexity: float # Perplexity without FEEL
    feel_perplexity: float     # Perplexity with FEEL
    perplexity_ratio: float    # feel_ppl / baseline_ppl
    n_tokens: int
    correct_baseline: bool
    correct_feel: bool
    output_baseline: str = ""
    output_feel: str = ""


@dataclass
class AlphaResult:
    """Aggregated results for one alpha value."""
    alpha: float
    n_prompts: int

    # Effect size metrics (primary)
    mean_flip_rate: float
    std_flip_rate: float
    mean_kl: float
    std_kl: float

    # Coherence metrics
    mean_perplexity_ratio: float
    coherence_ok: bool  # True if ppl_ratio < 2.0

    # Secondary metrics
    mean_top2_margin: float
    accuracy_baseline: float
    accuracy_feel: float
    accuracy_change: float

    # Per-category breakdown
    per_category: Dict[str, Dict] = field(default_factory=dict)

    # Recommendation
    in_safe_zone: bool = False  # flip_rate in [0.5%, 5%] and coherence_ok


@dataclass
class AlphaSweepResults:
    """Complete alpha sweep results."""
    timestamp: str
    n_prompts: int
    alphas: List[float]
    results: Dict[float, AlphaResult] = field(default_factory=dict)

    # Recommendation
    recommended_alpha: Optional[float] = None
    recommendation_reason: str = ""

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        def convert(obj):
            if isinstance(obj, AlphaResult):
                d = asdict(obj)
                return convert(d)
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return {
            "timestamp": self.timestamp,
            "n_prompts": self.n_prompts,
            "alphas": self.alphas,
            "results": {str(a): convert(r) for a, r in self.results.items()},
            "recommended_alpha": self.recommended_alpha,
            "recommendation_reason": self.recommendation_reason,
        }


# ============================================================
# FEEL Projector
# ============================================================

class FEELProjector(torch.nn.Module):
    """FEEL projector for experiments."""

    def __init__(self, sensor_dim: int = 12, embed_dim: int = 1536):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(sensor_dim, 64),
            torch.nn.GELU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, embed_dim),
        )
        self._init_near_zero()

    def _init_near_zero(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, sensors):
        return self.encoder(sensors)


# ============================================================
# Alpha Sweep Runner
# ============================================================

class AlphaSweepRunner:
    """
    Runs alpha sweep to find operational range for FEEL.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        checkpoint_path: str = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

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
        self.sensor_bank = CanonicalSensorBank(mode="legacy")

        # Load projector
        self.projector = FEELProjector(sensor_dim=12, embed_dim=self.embed_dim).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"  Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if "feel_stream_state" in ckpt:
                projector_state = {
                    k.replace("projector.", ""): v
                    for k, v in ckpt["feel_stream_state"].items()
                    if k.startswith("projector.")
                }
                if projector_state:
                    try:
                        self.projector.load_state_dict(projector_state, strict=False)
                        print(f"  Loaded projector weights")
                    except:
                        print(f"  Using default projector weights")

        # Diagnostic: check projector output norm
        self._check_projector_strength()

    def _check_projector_strength(self):
        """Check if projector output is strong enough to matter."""
        # Create synthetic sensor input
        test_sensors = torch.randn(1, 12).to(self.device)
        test_sensors = test_sensors / test_sensors.norm() * 5.0  # Normalize to typical range

        with torch.no_grad():
            feel_embed = self.projector(test_sensors)

        feel_norm = feel_embed.norm().item()
        embed_baseline_norm = 1.0  # Typical token embedding norm

        print(f"\n  PROJECTOR DIAGNOSTIC:")
        print(f"    Sensor input norm: {test_sensors.norm().item():.2f}")
        print(f"    FEEL embedding norm: {feel_norm:.4f}")
        print(f"    Embed dim: {self.embed_dim}")

        # At alpha=0.1, we add 0.1 * feel_norm to embeddings
        # For a typical embedding norm of ~1.0, we need feel_norm * alpha ≈ 0.1-1.0 to matter
        for alpha in [0.001, 0.01, 0.1, 1.0]:
            perturbation = alpha * feel_norm
            pct_of_embed = (perturbation / embed_baseline_norm) * 100
            print(f"    α={alpha:.3f}: perturbation={perturbation:.4f} ({pct_of_embed:.1f}% of embed norm)")

        if feel_norm < 0.1:
            print(f"\n    WARNING: FEEL embedding norm is very small!")
            print(f"    Token embedding norms are typically ~1.0")
            print(f"    Even at α=1.0, perturbation would only be {feel_norm:.4f}")
            print(f"    Recommend: Re-train projector with larger init or output scaling")

    def _compute_token_metrics(
        self,
        prompt: str,
        alpha: float,
        max_tokens: int = 15,
    ) -> Tuple[List[TokenMetrics], str, str]:
        """
        Generate tokens with and without FEEL, compute per-token metrics.

        Returns: (token_metrics, output_baseline, output_feel)
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        token_metrics = []

        # Generate baseline
        baseline_ids = input_ids.clone()
        baseline_log_probs = []

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(baseline_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()

            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            # Store for perplexity
            next_token = logits.argmax(dim=-1)
            baseline_log_probs.append(log_probs[0, next_token.item()].item())

            # Get top-2 for margin
            top2_vals, top2_idx = probs.topk(2, dim=-1)
            top2_margin = (top2_vals[0, 0] - top2_vals[0, 1]).item()
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).item()

            token_metrics.append({
                "position": step,
                "baseline_top1": next_token.item(),
                "baseline_probs": probs.clone(),
                "baseline_logits": logits.clone(),
                "top2_margin_baseline": top2_margin,
                "baseline_entropy": entropy,
            })

            baseline_ids = torch.cat([baseline_ids, next_token.unsqueeze(-1)], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        output_baseline = self.tokenizer.decode(
            baseline_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )

        # Generate with FEEL at given alpha
        feel_ids = input_ids.clone()
        feel_log_probs = []

        for step in range(len(token_metrics)):
            with torch.no_grad():
                outputs = self.model(feel_ids, use_cache=False)
                logits = outputs.logits

            # Get sensors and FEEL embedding
            sensors = self.sensor_bank(logits.float())
            feel_embed = self.projector(sensors)

            # Inject FEEL
            embeds = self.model.get_input_embeddings()(feel_ids)
            embeds = embeds + (alpha * feel_embed).to(embeds.dtype).unsqueeze(1)

            with torch.no_grad():
                outputs_feel = self.model(inputs_embeds=embeds, use_cache=False)
                logits_feel = outputs_feel.logits[:, -1, :].float()

            probs_feel = F.softmax(logits_feel, dim=-1)
            log_probs_feel = F.log_softmax(logits_feel, dim=-1)

            next_token_feel = logits_feel.argmax(dim=-1)
            feel_log_probs.append(log_probs_feel[0, next_token_feel.item()].item())

            # Compute KL divergence
            baseline_probs = token_metrics[step]["baseline_probs"]
            kl = F.kl_div(
                log_probs_feel,
                baseline_probs,
                reduction='sum'
            ).item()

            # Top-2 margin for FEEL
            top2_vals, _ = probs_feel.topk(2, dim=-1)
            top2_margin_feel = (top2_vals[0, 0] - top2_vals[0, 1]).item()
            entropy_feel = -(probs_feel * torch.log(probs_feel.clamp(min=1e-10))).sum(dim=-1).item()

            # Update metrics
            token_metrics[step]["feel_top1"] = next_token_feel.item()
            token_metrics[step]["flipped"] = (
                token_metrics[step]["baseline_top1"] != next_token_feel.item()
            )
            token_metrics[step]["kl_divergence"] = kl
            token_metrics[step]["top2_margin_feel"] = top2_margin_feel
            token_metrics[step]["feel_entropy"] = entropy_feel

            feel_ids = torch.cat([feel_ids, next_token_feel.unsqueeze(-1)], dim=-1)

        output_feel = self.tokenizer.decode(
            feel_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )

        # Compute perplexities
        baseline_ppl = np.exp(-np.mean(baseline_log_probs)) if baseline_log_probs else float('inf')
        feel_ppl = np.exp(-np.mean(feel_log_probs)) if feel_log_probs else float('inf')

        # Convert to TokenMetrics objects
        final_metrics = []
        for m in token_metrics:
            if "feel_top1" in m:
                final_metrics.append(TokenMetrics(
                    position=m["position"],
                    baseline_top1=m["baseline_top1"],
                    feel_top1=m["feel_top1"],
                    flipped=m["flipped"],
                    kl_divergence=m["kl_divergence"],
                    top2_margin_baseline=m["top2_margin_baseline"],
                    top2_margin_feel=m["top2_margin_feel"],
                    baseline_entropy=m["baseline_entropy"],
                    feel_entropy=m["feel_entropy"],
                ))

        return final_metrics, output_baseline, output_feel, baseline_ppl, feel_ppl

    def run_alpha_sweep(
        self,
        prompts: List[Dict],
        alphas: List[float],
        verbose: bool = True,
    ) -> AlphaSweepResults:
        """Run alpha sweep across all prompts and alpha values."""

        print("\n" + "=" * 70)
        print("  ALPHA SWEEP EFFECT SIZE ANALYSIS")
        print("=" * 70)
        print(f"  Prompts: {len(prompts)}")
        print(f"  Alphas: {alphas}")
        print("=" * 70)

        results = AlphaSweepResults(
            timestamp=datetime.now().isoformat(),
            n_prompts=len(prompts),
            alphas=alphas,
        )

        for alpha in alphas:
            print(f"\n[Alpha = {alpha:.2e}]")

            prompt_metrics = []

            for i, prompt_data in enumerate(prompts):
                prompt = prompt_data["prompt"]
                category = prompt_data.get("category", "unknown")
                answer = str(prompt_data.get("answer", "")).lower()

                token_metrics, output_base, output_feel, ppl_base, ppl_feel = \
                    self._compute_token_metrics(prompt + " ", alpha)

                if not token_metrics:
                    continue

                # Aggregate metrics
                flip_rate = sum(t.flipped for t in token_metrics) / len(token_metrics)
                mean_kl = np.mean([t.kl_divergence for t in token_metrics])
                mean_margin = np.mean([t.top2_margin_baseline for t in token_metrics])
                ppl_ratio = ppl_feel / ppl_base if ppl_base > 0 else float('inf')

                # Check correctness
                correct_base = answer in output_base.lower()
                correct_feel = answer in output_feel.lower()

                prompt_metrics.append(PromptMetrics(
                    prompt=prompt,
                    category=category,
                    alpha=alpha,
                    flip_rate=flip_rate,
                    mean_kl=mean_kl,
                    mean_top2_margin=mean_margin,
                    baseline_perplexity=ppl_base,
                    feel_perplexity=ppl_feel,
                    perplexity_ratio=ppl_ratio,
                    n_tokens=len(token_metrics),
                    correct_baseline=correct_base,
                    correct_feel=correct_feel,
                    output_baseline=output_base[:50],
                    output_feel=output_feel[:50],
                ))

                if verbose and (i + 1) % 10 == 0:
                    avg_flip = np.mean([p.flip_rate for p in prompt_metrics])
                    print(f"    {i+1}/{len(prompts)} - flip_rate: {avg_flip:.3f}")

            # Aggregate for this alpha
            if prompt_metrics:
                flip_rates = [p.flip_rate for p in prompt_metrics]
                kls = [p.mean_kl for p in prompt_metrics]
                margins = [p.mean_top2_margin for p in prompt_metrics]
                ppl_ratios = [p.perplexity_ratio for p in prompt_metrics if np.isfinite(p.perplexity_ratio)]

                acc_base = sum(p.correct_baseline for p in prompt_metrics) / len(prompt_metrics)
                acc_feel = sum(p.correct_feel for p in prompt_metrics) / len(prompt_metrics)

                mean_ppl_ratio = np.mean(ppl_ratios) if ppl_ratios else float('inf')
                coherence_ok = mean_ppl_ratio < 2.0

                # Per-category breakdown
                per_category = {}
                for cat in set(p.category for p in prompt_metrics):
                    cat_metrics = [p for p in prompt_metrics if p.category == cat]
                    per_category[cat] = {
                        "n": len(cat_metrics),
                        "flip_rate": np.mean([p.flip_rate for p in cat_metrics]),
                        "mean_kl": np.mean([p.mean_kl for p in cat_metrics]),
                        "accuracy_baseline": sum(p.correct_baseline for p in cat_metrics) / len(cat_metrics),
                        "accuracy_feel": sum(p.correct_feel for p in cat_metrics) / len(cat_metrics),
                    }

                # Check if in safe zone
                mean_flip = np.mean(flip_rates)
                in_safe_zone = (0.005 <= mean_flip <= 0.05) and coherence_ok

                alpha_result = AlphaResult(
                    alpha=alpha,
                    n_prompts=len(prompt_metrics),
                    mean_flip_rate=mean_flip,
                    std_flip_rate=np.std(flip_rates),
                    mean_kl=np.mean(kls),
                    std_kl=np.std(kls),
                    mean_perplexity_ratio=mean_ppl_ratio,
                    coherence_ok=coherence_ok,
                    mean_top2_margin=np.mean(margins),
                    accuracy_baseline=acc_base,
                    accuracy_feel=acc_feel,
                    accuracy_change=acc_feel - acc_base,
                    per_category=per_category,
                    in_safe_zone=in_safe_zone,
                )

                results.results[alpha] = alpha_result

                print(f"    flip_rate: {mean_flip:.3f} ({mean_flip*100:.1f}%)")
                print(f"    mean_kl: {np.mean(kls):.4f}")
                print(f"    ppl_ratio: {mean_ppl_ratio:.3f} {'(OK)' if coherence_ok else '(COHERENCE WARNING)'}")
                print(f"    accuracy: {acc_base:.3f} -> {acc_feel:.3f} ({acc_feel-acc_base:+.3f})")
                print(f"    safe_zone: {'YES' if in_safe_zone else 'NO'}")

        # Find recommended alpha
        safe_alphas = [a for a, r in results.results.items() if r.in_safe_zone]
        if safe_alphas:
            # Choose alpha with highest flip rate in safe zone
            recommended = max(safe_alphas, key=lambda a: results.results[a].mean_flip_rate)
            results.recommended_alpha = recommended
            results.recommendation_reason = f"Highest flip_rate ({results.results[recommended].mean_flip_rate:.3f}) in safe zone"
        else:
            # Check if any alpha has flip_rate > 0
            effective_alphas = [a for a, r in results.results.items() if r.mean_flip_rate > 0.001]
            if effective_alphas:
                # Recommend lowest alpha with some effect
                recommended = min(effective_alphas, key=lambda a: a)
                results.recommended_alpha = recommended
                r = results.results[recommended]
                results.recommendation_reason = (
                    f"Lowest alpha with effect (flip={r.mean_flip_rate:.3f}). "
                    f"May need higher alpha or coherence may collapse."
                )
            else:
                results.recommendation_reason = "No alpha produced measurable flip_rate. Need much higher alpha or different projector."

        return results

    def print_summary(self, results: AlphaSweepResults):
        """Print summary table."""
        print("\n" + "=" * 70)
        print("  ALPHA SWEEP SUMMARY")
        print("=" * 70)

        print("\n  Effect Size by Alpha:")
        print("  " + "-" * 65)
        print(f"  {'Alpha':>10s} | {'Flip%':>6s} | {'KL':>8s} | {'PPL Ratio':>9s} | {'Acc Δ':>7s} | {'Safe':>4s}")
        print("  " + "-" * 65)

        for alpha in sorted(results.results.keys()):
            r = results.results[alpha]
            safe_str = "YES" if r.in_safe_zone else "NO"
            print(f"  {alpha:>10.2e} | {r.mean_flip_rate*100:>5.1f}% | {r.mean_kl:>8.4f} | {r.mean_perplexity_ratio:>9.3f} | {r.accuracy_change:>+6.3f} | {safe_str:>4s}")

        print("  " + "-" * 65)

        print(f"\n  RECOMMENDATION:")
        if results.recommended_alpha is not None:
            print(f"    Recommended α = {results.recommended_alpha:.2e}")
            print(f"    Reason: {results.recommendation_reason}")
        else:
            print(f"    {results.recommendation_reason}")

        # Interpretation
        print("\n  INTERPRETATION:")
        any_flips = any(r.mean_flip_rate > 0.001 for r in results.results.values())
        if not any_flips:
            print("    - NO token flips detected at any alpha")
            print("    - FEEL embedding is too weak or projector needs training")
            print("    - Try: larger projector init, trained projector, or much higher alpha")
        else:
            max_flip = max(r.mean_flip_rate for r in results.results.values())
            max_alpha = [a for a, r in results.results.items() if r.mean_flip_rate == max_flip][0]
            print(f"    - Max flip_rate = {max_flip:.1%} at α = {max_alpha:.2e}")
            if max_flip < 0.005:
                print("    - Effect is very small, may need higher alpha")
            elif max_flip > 0.10:
                print("    - Effect is large, may cause coherence issues")
            else:
                print("    - Effect is in reasonable range")


def main():
    parser = argparse.ArgumentParser(description="Alpha Sweep Effect Size Analysis")
    parser.add_argument("--checkpoint", type=str,
                       default="results/feel_training/canonical_v6_checkpoint.pt")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--quick", action="store_true", help="Quick (20 prompts)")
    parser.add_argument("--alphas", type=str, default=None,
                       help="Comma-separated alphas (e.g., '0,1e-4,1e-3,1e-2')")
    args = parser.parse_args()

    # Default alpha range (orders of magnitude)
    if args.alphas:
        alphas = [float(a) for a in args.alphas.split(",")]
    else:
        alphas = [0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    prompts = EFFECT_SIZE_PROMPTS
    if args.quick:
        prompts = prompts[:20]

    runner = AlphaSweepRunner(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
    )

    results = runner.run_alpha_sweep(prompts, alphas)
    runner.print_summary(results)

    # Save results
    results_path = "results/feel_experiments/alpha_sweep_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
