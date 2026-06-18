#!/usr/bin/env python3
"""
Alpha Sweep v8.0 - Find Operational Alpha Range with Full Hardware
===================================================================

Uses the v8 unified modules with:
- Full 16-dim sensors (with real hardware telemetry)
- Unified FEELProjector with learnable scale
- Effect-size metrics: flip_rate, KL, entropy_delta, margin_change

Goal: Find alpha where flip_rate ∈ [0.5%, 5%] and coherence doesn't collapse.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.canonical_sensors import (
    CanonicalSensorBank, RuntimeContext, HardwareContext,
    SENSOR_DIM_FULL,
)
from src.telemetry_sampler import TelemetrySampler
from src.feel_projector import FEELProjectorFull, PROJECTOR_VERSION


# Test prompts for alpha sweep
TEST_PROMPTS = [
    "The capital of France is",
    "What is 7 times 8?",
    "In Python, to define a function you use the keyword",
    "The largest planet in our solar system is",
    "Water boils at 100 degrees",
    "The chemical symbol for gold is",
    "def fibonacci(n):",
    "Machine learning is a subset of",
    "The speed of light is approximately",
    "What is 15 + 27?",
    "The first president of the United States was",
    "In a binary search tree,",
    "The human body has 206",
    "To create a list in Python:",
    "The Eiffel Tower is located in",
    "What is 2 to the power of 10?",
    "DNA stands for",
    "The mitochondria is the",
    "In object-oriented programming,",
    "The Earth orbits the",
]


class AlphaSweepV8:
    """Alpha sweep with v8 full hardware integration."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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

        # Full mode sensor bank
        self.sensor_bank = CanonicalSensorBank(mode="full")

        # Unified projector with learnable scale
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)

        # Telemetry sampler
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            time.sleep(1)
        except Exception as e:
            print(f"  Warning: Telemetry failed: {e}")

        # Diagnose projector
        diag = self.projector.diagnose()
        print(f"  Projector: scale={diag['output_scale']:.4f}, output_norm={diag['final_output_norm']:.4f}")

    def _get_hardware_context(self, t_start: float, t_end: float) -> HardwareContext:
        """Get hardware context from telemetry."""
        if self.telemetry is None:
            return HardwareContext()
        data = self.telemetry.get_token_aligned(t_start, t_end)
        return HardwareContext.from_dict(data)

    def measure_effect_at_alpha(
        self,
        prompt: str,
        alpha: float,
        max_tokens: int = 10,
    ) -> Dict:
        """
        Measure effect size at given alpha.

        Returns:
            flip_count: Number of tokens where argmax changed
            kl_values: KL divergence at each token
            entropy_baseline: Entropy without FEEL
            entropy_feel: Entropy with FEEL
            margin_baseline: Top-2 margin without FEEL
            margin_feel: Top-2 margin with FEEL
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        flip_count = 0
        kl_values = []
        entropy_baseline_list = []
        entropy_feel_list = []
        margin_baseline_list = []
        margin_feel_list = []

        for step in range(max_tokens):
            t_start = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits_baseline = outputs.logits[:, -1, :].float()
            t_end = time.time()

            # Build contexts
            runtime = RuntimeContext(
                token_latency=t_end - t_start,
                kv_cache_tokens=current_ids.shape[1],
                generation_depth=step,
            )
            hardware = self._get_hardware_context(t_start, t_end)

            # Compute sensors
            sensors = self.sensor_bank(logits_baseline, runtime=runtime, hardware=hardware)

            # Apply FEEL
            feel_embed = self.projector(sensors.float())
            embeds = self.model.get_input_embeddings()(current_ids)
            embeds_feel = embeds + (alpha * feel_embed).to(embeds.dtype).unsqueeze(1)

            with torch.no_grad():
                outputs_feel = self.model(inputs_embeds=embeds_feel, use_cache=False)
                logits_feel = outputs_feel.logits[:, -1, :].float()

            # Compute metrics
            probs_baseline = F.softmax(logits_baseline, dim=-1)
            probs_feel = F.softmax(logits_feel, dim=-1)

            # Flip detection
            argmax_baseline = logits_baseline.argmax(dim=-1).item()
            argmax_feel = logits_feel.argmax(dim=-1).item()
            if argmax_baseline != argmax_feel:
                flip_count += 1

            # KL divergence (baseline || feel)
            kl = F.kl_div(
                probs_feel.log().clamp(min=-100),
                probs_baseline,
                reduction='batchmean'
            ).item()
            kl_values.append(kl)

            # Entropy
            entropy_b = -(probs_baseline * probs_baseline.log().clamp(min=-100)).sum().item()
            entropy_f = -(probs_feel * probs_feel.log().clamp(min=-100)).sum().item()
            entropy_baseline_list.append(entropy_b)
            entropy_feel_list.append(entropy_f)

            # Margin (top-2 gap)
            top2_b = probs_baseline.topk(2, dim=-1).values[0]
            top2_f = probs_feel.topk(2, dim=-1).values[0]
            margin_b = (top2_b[0] - top2_b[1]).item()
            margin_f = (top2_f[0] - top2_f[1]).item()
            margin_baseline_list.append(margin_b)
            margin_feel_list.append(margin_f)

            # Next token (using baseline for consistency)
            next_token = logits_baseline.argmax(dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        n_tokens = len(kl_values)
        return {
            "n_tokens": n_tokens,
            "flip_count": flip_count,
            "flip_rate": flip_count / n_tokens if n_tokens > 0 else 0,
            "mean_kl": np.mean(kl_values) if kl_values else 0,
            "max_kl": np.max(kl_values) if kl_values else 0,
            "entropy_baseline": np.mean(entropy_baseline_list) if entropy_baseline_list else 0,
            "entropy_feel": np.mean(entropy_feel_list) if entropy_feel_list else 0,
            "entropy_delta": np.mean(entropy_feel_list) - np.mean(entropy_baseline_list) if entropy_baseline_list else 0,
            "margin_baseline": np.mean(margin_baseline_list) if margin_baseline_list else 0,
            "margin_feel": np.mean(margin_feel_list) if margin_feel_list else 0,
            "margin_delta": np.mean(margin_feel_list) - np.mean(margin_baseline_list) if margin_baseline_list else 0,
        }

    def run_sweep(
        self,
        prompts: List[str],
        alphas: List[float],
        max_tokens: int = 10,
    ) -> Dict:
        """Run full alpha sweep."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "projector_version": PROJECTOR_VERSION,
            "n_prompts": len(prompts),
            "max_tokens": max_tokens,
            "alphas": {},
            "projector_diagnostics": self.projector.diagnose(),
        }

        # Get telemetry status
        if self.telemetry:
            validity = self.telemetry.get_validity_report()
            results["telemetry"] = {
                "source": validity.source,
                "temp_avail": validity.temp_availability,
                "power_avail": validity.power_availability,
                "util_avail": validity.util_availability,
            }

        for alpha in alphas:
            print(f"\n  Testing alpha={alpha:.4f}...")
            alpha_results = {
                "flip_rates": [],
                "kl_values": [],
                "entropy_deltas": [],
                "margin_deltas": [],
            }

            for i, prompt in enumerate(prompts):
                metrics = self.measure_effect_at_alpha(prompt, alpha, max_tokens)
                alpha_results["flip_rates"].append(metrics["flip_rate"])
                alpha_results["kl_values"].append(metrics["mean_kl"])
                alpha_results["entropy_deltas"].append(metrics["entropy_delta"])
                alpha_results["margin_deltas"].append(metrics["margin_delta"])

                if (i + 1) % 5 == 0:
                    avg_flip = np.mean(alpha_results["flip_rates"]) * 100
                    avg_kl = np.mean(alpha_results["kl_values"])
                    print(f"    [{i+1}/{len(prompts)}] flip={avg_flip:.1f}%, kl={avg_kl:.4f}")

            # Aggregate
            results["alphas"][str(alpha)] = {
                "flip_rate_mean": np.mean(alpha_results["flip_rates"]),
                "flip_rate_std": np.std(alpha_results["flip_rates"]),
                "kl_mean": np.mean(alpha_results["kl_values"]),
                "kl_std": np.std(alpha_results["kl_values"]),
                "entropy_delta_mean": np.mean(alpha_results["entropy_deltas"]),
                "margin_delta_mean": np.mean(alpha_results["margin_deltas"]),
                "in_safe_zone": 0.005 <= np.mean(alpha_results["flip_rates"]) <= 0.05,
            }

            fr = results["alphas"][str(alpha)]["flip_rate_mean"]
            kl = results["alphas"][str(alpha)]["kl_mean"]
            safe = "✓" if results["alphas"][str(alpha)]["in_safe_zone"] else "✗"
            print(f"    → flip_rate={fr*100:.2f}%, kl={kl:.4f} {safe}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Alpha Sweep v8.0")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  ALPHA SWEEP v8.0 - Full Hardware Integration")
    print("=" * 70)

    sweeper = AlphaSweepV8(model_name=args.model)

    # Alpha range to test
    alphas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    prompts = TEST_PROMPTS[:args.n_prompts]
    print(f"\n  Testing {len(prompts)} prompts, {len(alphas)} alpha values")
    print(f"  Alphas: {alphas}")

    results = sweeper.run_sweep(prompts, alphas, max_tokens=args.max_tokens)

    # Print summary
    print("\n" + "=" * 70)
    print("  ALPHA SWEEP SUMMARY")
    print("=" * 70)
    print(f"\n  {'Alpha':<10} {'Flip%':<10} {'KL':<12} {'Entropy Δ':<12} {'Safe Zone'}")
    print("  " + "-" * 60)

    for alpha, data in results["alphas"].items():
        flip = data["flip_rate_mean"] * 100
        kl = data["kl_mean"]
        ent = data["entropy_delta_mean"]
        safe = "✓ YES" if data["in_safe_zone"] else "✗ NO"
        print(f"  {alpha:<10} {flip:<10.2f} {kl:<12.4f} {ent:<12.4f} {safe}")

    # Find optimal alpha
    safe_alphas = [
        (float(a), d["flip_rate_mean"], d["kl_mean"])
        for a, d in results["alphas"].items()
        if d["in_safe_zone"]
    ]
    if safe_alphas:
        # Prefer alpha with highest flip rate in safe zone
        best = max(safe_alphas, key=lambda x: x[1])
        print(f"\n  RECOMMENDED ALPHA: {best[0]} (flip={best[1]*100:.1f}%, kl={best[2]:.4f})")
        results["recommended_alpha"] = best[0]
    else:
        # Find alpha closest to safe zone
        all_flips = [(float(a), d["flip_rate_mean"]) for a, d in results["alphas"].items()]
        # Find alpha with flip rate closest to 2.5% (middle of safe zone)
        closest = min(all_flips, key=lambda x: abs(x[1] - 0.025))
        print(f"\n  NO SAFE ZONE FOUND")
        print(f"  Closest to target: alpha={closest[0]} (flip={closest[1]*100:.1f}%)")
        results["recommended_alpha"] = closest[0]

    # Cleanup
    if sweeper.telemetry:
        sweeper.telemetry.stop()

    # Save results
    results_path = "results/feel_experiments/alpha_sweep_v8_results.json"
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
