#!/usr/bin/env python3
"""
z35 Comparison Validator - Embodied vs Baseline + Business Metrics
===================================================================

Compares:
1. Embodied model (with FEEL loop) vs Vanilla baseline
2. Different training checkpoints
3. Energy efficiency improvements

Produces:
- Side-by-side hypothesis results
- Business metrics comparison table
- Improvement percentages for AMD/HP pitch
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sensors.canonical_features import CanonicalSensorHub

# ============================================================================
# STATISTICAL HELPERS
# ============================================================================

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d with proper edge case handling."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, (n1 + n2 - 2)))
    m1, m2 = np.mean(group1), np.mean(group2)
    if pooled_std < 1e-10:
        if abs(m1 - m2) < 1e-10:
            return 0.0
        return float(np.sign(m1 - m2) * np.inf)
    return (m1 - m2) / pooled_std

def welch_ttest(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Welch's t-test."""
    if np.std(group1) < 1e-10 and np.std(group2) < 1e-10:
        if np.mean(group1) != np.mean(group2):
            return 0.0, float('inf')
        return 1.0, 0.0
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    return p_val, t_stat

# ============================================================================
# BUSINESS METRICS
# ============================================================================

class BusinessMetrics:
    """Track business-oriented metrics."""

    ELECTRICITY_COST_PER_KWH = 0.12  # USD
    CO2_PER_KWH = 0.4  # kg

    def __init__(self):
        self.reset()

    def reset(self):
        self.ttft_samples = []
        self.tpot_samples = []
        self.tokens_generated = 0
        self.total_joules = 0.0
        self.total_time_s = 0.0
        self.peak_temp_c = 0.0
        self.power_samples = []
        self.temp_samples = []

    def record(self, ttft_ms: float, total_time_ms: float, tokens: int,
               joules: float, temp: float, power_w: float = 0.0):
        self.ttft_samples.append(ttft_ms)
        if tokens > 1:
            self.tpot_samples.append((total_time_ms - ttft_ms) / (tokens - 1))
        self.tokens_generated += tokens
        self.total_joules += joules
        self.total_time_s += total_time_ms / 1000.0
        self.peak_temp_c = max(self.peak_temp_c, temp)
        if power_w > 0:
            self.power_samples.append(power_w)
        self.temp_samples.append(temp)

    def report(self) -> Dict:
        tokens_per_joule = self.tokens_generated / max(0.001, self.total_joules)
        joules_per_token = self.total_joules / max(1, self.tokens_generated)
        kwh = self.total_joules / 3_600_000
        cost_per_1m = (kwh * self.ELECTRICITY_COST_PER_KWH) / max(1, self.tokens_generated) * 1_000_000

        return {
            "tokens_per_joule": round(tokens_per_joule, 2),
            "joules_per_token": round(joules_per_token, 4),
            "tokens_per_second": round(self.tokens_generated / max(0.001, self.total_time_s), 2),
            "usd_per_1m_tokens": round(cost_per_1m, 4),
            "ttft_p50_ms": round(np.percentile(self.ttft_samples, 50), 2) if self.ttft_samples else 0,
            "ttft_p95_ms": round(np.percentile(self.ttft_samples, 95), 2) if self.ttft_samples else 0,
            "tpot_p50_ms": round(np.percentile(self.tpot_samples, 50), 2) if self.tpot_samples else 0,
            "tpot_p95_ms": round(np.percentile(self.tpot_samples, 95), 2) if self.tpot_samples else 0,
            "peak_temp_c": round(self.peak_temp_c, 1),
            "avg_temp_c": round(np.mean(self.temp_samples), 1) if self.temp_samples else 0,
            "avg_power_w": round(np.mean(self.power_samples), 1) if self.power_samples else 0,
            "total_tokens": self.tokens_generated,
            "total_joules": round(self.total_joules, 2),
        }

# ============================================================================
# KL DIVERGENCE FOR H4
# ============================================================================

def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) per token."""
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1)

def delta_logprob(logits_on: torch.Tensor, logits_off: torch.Tensor,
                  target_ids: torch.Tensor) -> torch.Tensor:
    """Δlogprob of realized tokens."""
    lp_on = F.log_softmax(logits_on, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    lp_off = F.log_softmax(logits_off, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return (lp_on - lp_off).abs()

# ============================================================================
# COMPARISON VALIDATOR
# ============================================================================

class ComparisonValidator:
    """Compare embodied model vs baseline."""

    BONFERRONI_ALPHA = 0.05 / 6

    def __init__(self, device: str):
        self.device = device
        self.prompts = [
            "The future of artificial intelligence will",
            "In a world where technology",
            "Scientists have discovered that",
            "The most important thing about",
            "When considering the implications of",
            "Research has shown that humans",
            "The relationship between mind and",
            "Looking at the data, we can",
            "The evolution of computing has",
            "Understanding consciousness requires",
        ]

    def benchmark_model(self, model, tokenizer, sensor_hub, skip_blocks: Dict,
                       name: str, trials: int = 30) -> Dict:
        """Benchmark a single model configuration."""
        print(f"\n  Benchmarking: {name}")

        metrics = BusinessMetrics()
        metrics.reset()

        # Track hypothesis-relevant metrics
        gate_values = []
        skip_rates = []
        film_effects = []
        kl_values = []
        j_per_token = []
        temp_deltas = []

        for i in range(trials):
            prompt = random.choice(self.prompts)
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc.input_ids.to(self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)

            # Reset blocks
            for block in skip_blocks.values():
                if hasattr(block, 'force_skip'):
                    block.force_skip = None
                if hasattr(block, 'gate_value'):
                    block.gate_value = None
                if hasattr(block, 'disable_film'):
                    block.disable_film = False

            # Read initial temperature
            sensor_hub.update(tokens_generated=0)
            temp_before = sensor_hub.features.get('gpu_temp_c', 50)

            # Energy window
            sensor_hub.reset_energy_window()
            sensor_hub.update(tokens_generated=0)

            # Generate
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
            t1 = time.perf_counter()

            tokens = out.sequences.shape[1] - input_ids.shape[1]
            sensor_hub.update(tokens_generated=tokens)

            # Energy
            j = sensor_hub.energy_window_joules
            if tokens > 0:
                j_per_token.append(j / tokens)

            # Temperature after
            temp_after = sensor_hub.features.get('gpu_temp_c', 50)
            temp_deltas.append(temp_after - temp_before)

            # Collect block metrics
            for block in skip_blocks.values():
                if hasattr(block, 'last_gate_value') and block.last_gate_value is not None:
                    gate_values.append(block.last_gate_value)
                if hasattr(block, 'last_skipped'):
                    skip_rates.append(1.0 if block.last_skipped else 0.0)
                if hasattr(block, 'last_film_effect') and block.last_film_effect is not None:
                    film_effects.append(block.last_film_effect)

            # KL test (FiLM ON vs OFF)
            if skip_blocks:
                full_ids = out.sequences
                full_attn = torch.ones_like(full_ids, device=self.device)
                T_prompt = input_ids.shape[1]

                # FiLM ON
                for block in skip_blocks.values():
                    if hasattr(block, 'disable_film'):
                        block.disable_film = False
                with torch.no_grad():
                    logits_on = model(full_ids, attention_mask=full_attn).logits

                # FiLM OFF
                for block in skip_blocks.values():
                    if hasattr(block, 'disable_film'):
                        block.disable_film = True
                with torch.no_grad():
                    logits_off = model(full_ids, attention_mask=full_attn).logits

                # Reset
                for block in skip_blocks.values():
                    if hasattr(block, 'disable_film'):
                        block.disable_film = False

                if full_ids.shape[1] > T_prompt:
                    lo_on = logits_on[:, T_prompt-1:-1, :]
                    lo_off = logits_off[:, T_prompt-1:-1, :]
                    kl = kl_divergence(lo_on, lo_off).mean().item()
                    kl_values.append(kl)

            # Record business metrics
            total_ms = (t1 - t0) * 1000
            ttft_ms = total_ms * 0.3  # Approximate first token
            metrics.record(
                ttft_ms=ttft_ms,
                total_time_ms=total_ms,
                tokens=tokens,
                joules=j,
                temp=temp_after,
                power_w=sensor_hub.features.get('power_w', 100)
            )

        return {
            "name": name,
            "business": metrics.report(),
            "hypothesis_metrics": {
                "mean_gate": np.mean(gate_values) if gate_values else 0.5,
                "mean_skip_rate": np.mean(skip_rates) if skip_rates else 0.0,
                "mean_film_effect": np.mean(film_effects) if film_effects else 0.0,
                "mean_kl_divergence": np.mean(kl_values) if kl_values else 0.0,
                "mean_j_per_token": np.mean(j_per_token) if j_per_token else 0.0,
                "mean_temp_delta": np.mean(temp_deltas) if temp_deltas else 0.0,
            }
        }

    def compare(self, results_embodied: Dict, results_baseline: Dict) -> Dict:
        """Generate comparison report."""

        emb = results_embodied
        base = results_baseline

        # Business improvements
        biz_emb = emb["business"]
        biz_base = base["business"]

        def pct_improvement(emb_val, base_val, higher_is_better=True):
            if base_val == 0:
                return 0.0
            if higher_is_better:
                return ((emb_val - base_val) / base_val) * 100
            else:
                return ((base_val - emb_val) / base_val) * 100

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "embodied": emb,
            "baseline": base,
            "improvements": {
                "efficiency": {
                    "tokens_per_joule": {
                        "embodied": biz_emb["tokens_per_joule"],
                        "baseline": biz_base["tokens_per_joule"],
                        "improvement_pct": round(pct_improvement(
                            biz_emb["tokens_per_joule"],
                            biz_base["tokens_per_joule"],
                            higher_is_better=True
                        ), 1),
                    },
                    "joules_per_token": {
                        "embodied": biz_emb["joules_per_token"],
                        "baseline": biz_base["joules_per_token"],
                        "improvement_pct": round(pct_improvement(
                            biz_emb["joules_per_token"],
                            biz_base["joules_per_token"],
                            higher_is_better=False
                        ), 1),
                    },
                },
                "cost": {
                    "usd_per_1m_tokens": {
                        "embodied": biz_emb["usd_per_1m_tokens"],
                        "baseline": biz_base["usd_per_1m_tokens"],
                        "savings_pct": round(pct_improvement(
                            biz_emb["usd_per_1m_tokens"],
                            biz_base["usd_per_1m_tokens"],
                            higher_is_better=False
                        ), 1),
                    },
                },
                "latency": {
                    "ttft_p50_ms": {
                        "embodied": biz_emb["ttft_p50_ms"],
                        "baseline": biz_base["ttft_p50_ms"],
                        "improvement_pct": round(pct_improvement(
                            biz_emb["ttft_p50_ms"],
                            biz_base["ttft_p50_ms"],
                            higher_is_better=False
                        ), 1),
                    },
                    "tpot_p50_ms": {
                        "embodied": biz_emb["tpot_p50_ms"],
                        "baseline": biz_base["tpot_p50_ms"],
                        "improvement_pct": round(pct_improvement(
                            biz_emb["tpot_p50_ms"],
                            biz_base["tpot_p50_ms"],
                            higher_is_better=False
                        ), 1),
                    },
                },
                "thermal": {
                    "peak_temp_c": {
                        "embodied": biz_emb["peak_temp_c"],
                        "baseline": biz_base["peak_temp_c"],
                        "reduction_c": round(biz_base["peak_temp_c"] - biz_emb["peak_temp_c"], 1),
                    },
                    "avg_power_w": {
                        "embodied": biz_emb["avg_power_w"],
                        "baseline": biz_base["avg_power_w"],
                        "reduction_pct": round(pct_improvement(
                            biz_emb["avg_power_w"],
                            biz_base["avg_power_w"],
                            higher_is_better=False
                        ), 1),
                    },
                },
            },
            "embodiment_loop_metrics": {
                "gate_activation": emb["hypothesis_metrics"]["mean_gate"],
                "skip_rate": emb["hypothesis_metrics"]["mean_skip_rate"],
                "film_effect": emb["hypothesis_metrics"]["mean_film_effect"],
                "kl_divergence": emb["hypothesis_metrics"]["mean_kl_divergence"],
                "temp_feedback": emb["hypothesis_metrics"]["mean_temp_delta"],
            }
        }

        return comparison


def load_embodied_model(checkpoint_path: str, device: str):
    """Load embodied model with skip blocks."""
    from src.modeling.metabolic_gate import EmbodiedModelWithMetabolicGate

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load base
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Wrap in embodied model
    sensor_hub = CanonicalSensorHub()
    model = EmbodiedModelWithMetabolicGate(base_model, sensor_hub)

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"  Loaded checkpoint: {checkpoint_path}")

    return model, tokenizer, sensor_hub, model.skip_blocks


def load_baseline_model(device: str):
    """Load vanilla baseline model (no embodiment)."""
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sensor_hub = CanonicalSensorHub()

    return model, tokenizer, sensor_hub, {}


def main():
    parser = argparse.ArgumentParser(description="z35 Comparison Validator")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to embodied model checkpoint")
    parser.add_argument("--trials", type=int, default=30,
                       help="Trials per benchmark")
    parser.add_argument("--output", type=str, default="results/z35_comparison.json",
                       help="Output JSON path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    validator = ComparisonValidator(device)

    # Load baseline (vanilla Qwen)
    print("\n[1/4] Loading baseline model (vanilla Qwen)...")
    base_model, base_tok, base_sensors, base_blocks = load_baseline_model(device)

    # Benchmark baseline
    print("\n[2/4] Benchmarking baseline...")
    results_baseline = validator.benchmark_model(
        base_model, base_tok, base_sensors, base_blocks,
        name="Qwen2.5-1.5B (Vanilla)",
        trials=args.trials
    )

    # Free baseline memory
    del base_model
    torch.cuda.empty_cache()

    # Load embodied model
    print("\n[3/4] Loading embodied model...")
    emb_model, emb_tok, emb_sensors, emb_blocks = load_embodied_model(
        args.checkpoint, device
    )

    # Benchmark embodied
    print("\n[4/4] Benchmarking embodied model...")
    results_embodied = validator.benchmark_model(
        emb_model, emb_tok, emb_sensors, emb_blocks,
        name="FEEL Embodied (z34)",
        trials=args.trials
    )

    # Generate comparison
    comparison = validator.compare(results_embodied, results_baseline)

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: EMBODIED vs BASELINE")
    print("=" * 70)

    imp = comparison["improvements"]

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ METRIC                  │ BASELINE    │ EMBODIED    │ IMPROVEMENT │")
    print("├─────────────────────────────────────────────────────────────────────┤")

    eff = imp["efficiency"]
    print(f"│ Tokens/Joule            │ {eff['tokens_per_joule']['baseline']:>10.1f} │ {eff['tokens_per_joule']['embodied']:>10.1f} │ {eff['tokens_per_joule']['improvement_pct']:>+9.1f}% │")
    print(f"│ Joules/Token            │ {eff['joules_per_token']['baseline']:>10.4f} │ {eff['joules_per_token']['embodied']:>10.4f} │ {eff['joules_per_token']['improvement_pct']:>+9.1f}% │")

    cost = imp["cost"]
    print(f"│ $/1M Tokens             │ {cost['usd_per_1m_tokens']['baseline']:>10.4f} │ {cost['usd_per_1m_tokens']['embodied']:>10.4f} │ {cost['usd_per_1m_tokens']['savings_pct']:>+9.1f}% │")

    lat = imp["latency"]
    print(f"│ TTFT p50 (ms)           │ {lat['ttft_p50_ms']['baseline']:>10.1f} │ {lat['ttft_p50_ms']['embodied']:>10.1f} │ {lat['ttft_p50_ms']['improvement_pct']:>+9.1f}% │")
    print(f"│ TPOT p50 (ms)           │ {lat['tpot_p50_ms']['baseline']:>10.1f} │ {lat['tpot_p50_ms']['embodied']:>10.1f} │ {lat['tpot_p50_ms']['improvement_pct']:>+9.1f}% │")

    therm = imp["thermal"]
    print(f"│ Peak Temp (°C)          │ {therm['peak_temp_c']['baseline']:>10.1f} │ {therm['peak_temp_c']['embodied']:>10.1f} │ {therm['peak_temp_c']['reduction_c']:>+9.1f}°C│")
    print(f"│ Avg Power (W)           │ {therm['avg_power_w']['baseline']:>10.1f} │ {therm['avg_power_w']['embodied']:>10.1f} │ {therm['avg_power_w']['reduction_pct']:>+9.1f}% │")

    print("└─────────────────────────────────────────────────────────────────────┘")

    # Embodiment loop metrics
    loop = comparison["embodiment_loop_metrics"]
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ EMBODIMENT LOOP STATUS                                              │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Gate Activation:     {loop['gate_activation']:.3f}                                       │")
    print(f"│ Skip Rate:           {loop['skip_rate']:.3f}                                       │")
    print(f"│ FiLM Effect:         {loop['film_effect']:.3f}                                       │")
    print(f"│ KL Divergence:       {loop['kl_divergence']:.4f}                                      │")
    print(f"│ Temp Feedback:       {loop['temp_feedback']:.2f}°C                                       │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")

    return comparison


if __name__ == "__main__":
    main()
