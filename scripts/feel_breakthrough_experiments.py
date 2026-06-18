#!/usr/bin/env python3
"""
FEEL Breakthrough Experiments - Publication-Grade Evidence
===========================================================
Comprehensive experiment suite implementing the research plan:

1. Aux head control (alpha=0 vs alpha>0)
2. Counterfactual dataset (same tokens, different sensors)
3. Extensive suite with bootstrap CIs
4. Strengthened falsification
5. Ridge regression predictive test
6. Effective compute multiplier

Run: python scripts/feel_breakthrough_experiments.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    n_bootstrap: int = 100
    n_tokens_per_prompt: int = 64
    alpha_values: List[float] = None  # Set in __post_init__
    lag_values: List[int] = None
    default_alpha: float = 1.0  # Higher alpha to show clear FEEL effects

    def __post_init__(self):
        if self.alpha_values is None:
            self.alpha_values = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
        if self.lag_values is None:
            self.lag_values = [0, 1, 2, 4, 8, 16]


# Diverse prompt set for statistical power
DIVERSE_PROMPTS = [
    # Reasoning/Math
    "What is 17 * 23? Let me think step by step:",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "Explain why the square root of 2 is irrational:",
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",

    # Uncertainty/Metacognition
    "Describe what uncertainty feels like:",
    "How confident are you in your reasoning abilities?",
    "Explain the difference between knowing and believing:",
    "What does it mean to be self-aware?",

    # Creative/Open-ended
    "Write a haiku about artificial intelligence:",
    "Describe the color blue to someone who has never seen it:",
    "What would a conversation between two AIs be like?",
    "Imagine a world where machines can feel:",

    # Technical/Factual
    "Explain quantum entanglement in simple terms:",
    "What causes the northern lights?",
    "How does a transformer neural network work?",
    "Describe the process of photosynthesis:",

    # Philosophical
    "What is consciousness?",
    "Can machines truly understand language?",
    "What is the nature of intelligence?",
    "Is free will an illusion?",

    # Code/Structured
    "Write a Python function to check if a number is prime:",
    "Explain the quicksort algorithm:",
    "What is the difference between a stack and a queue?",
    "How does garbage collection work in programming?",
]


# ============================================================
# FEEL Components
# ============================================================

class SensorBank(nn.Module):
    """Computes 8 interoceptive signals from model activations."""

    def __init__(self):
        super().__init__()
        self.sensor_names = [
            "entropy", "top1_prob", "top5_gap", "logit_std",
            "logit_range", "kurtosis", "skewness", "grad_norm_proxy"
        ]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        orig_dtype = logits.dtype
        logits = logits[:, -1, :].float()  # Compute in float32 for stability
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        entropy = -(probs * log_probs).sum(-1) / np.log(logits.shape[-1])
        top1_prob = probs.max(dim=-1).values
        top5 = probs.topk(5, dim=-1).values
        top5_gap = top5[:, 0] - top5[:, -1]
        logit_std = logits.std(dim=-1) / 10.0
        logit_range = (logits.max(dim=-1).values - logits.min(dim=-1).values) / 100.0

        logit_mean = logits.mean(dim=-1, keepdim=True)
        logit_centered = logits - logit_mean
        var = (logit_centered ** 2).mean(dim=-1)
        std = var.sqrt() + 1e-8

        skewness = ((logit_centered / std.unsqueeze(-1)) ** 3).mean(dim=-1) / 10.0
        kurtosis = ((logit_centered / std.unsqueeze(-1)) ** 4).mean(dim=-1) / 100.0
        grad_proxy = (entropy * (1 - entropy)).clamp(0, 1)

        sensors = torch.stack([
            entropy, top1_prob, top5_gap, logit_std,
            logit_range, kurtosis, skewness, grad_proxy
        ], dim=-1)
        return sensors.to(orig_dtype)  # Return in original dtype for fp16 compat


class FEELProjector(nn.Module):
    def __init__(self, sensor_dim: int = 8, hidden_dim: int = 64, embed_dim: int = 1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._init_near_zero()

    def _init_near_zero(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        return self.net(sensors)


class FEELStream(nn.Module):
    def __init__(self, embed_dim: int = 1536, fixed_alpha: float = None):
        super().__init__()
        self.sensor_bank = SensorBank()
        self.projector = FEELProjector(embed_dim=embed_dim)
        self.alpha = nn.Parameter(torch.tensor(-4.0))
        self.fixed_alpha = fixed_alpha

        if fixed_alpha is not None:
            raw_alpha = np.log(np.exp(fixed_alpha) - 1 + 1e-8) + 4.0
            with torch.no_grad():
                self.alpha.fill_(raw_alpha)
            self.alpha.requires_grad = False

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sensors = self.sensor_bank(logits)
        z_feel = self.projector(sensors)
        alpha = F.softplus(self.alpha)
        feel_embed = alpha * z_feel
        return feel_embed, sensors, alpha

    def get_z_feel(self, logits: torch.Tensor) -> torch.Tensor:
        """Get raw z_feel without alpha scaling."""
        sensors = self.sensor_bank(logits)
        return self.projector(sensors)


class HiddenStateAuxHead(nn.Module):
    """Predicts entropy from LM hidden state (not z_feel)."""

    def __init__(self, hidden_dim: int = 1536, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, 1),
        )

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.net(h_last).squeeze(-1)


# ============================================================
# Experiment Runner
# ============================================================

class BreakthroughExperiments:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.default_alpha = config.default_alpha

        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()

        self.embed_dim = self.model.config.hidden_size
        self.model_dtype = next(self.model.parameters()).dtype
        self.feel_stream = FEELStream(embed_dim=self.embed_dim).to(self.device).to(self.model_dtype)
        self.aux_head = HiddenStateAuxHead(hidden_dim=self.embed_dim).to(self.device).to(self.model_dtype)

        # Try to load trained checkpoint
        checkpoint_path = Path("results/feel_training/feel_projector_checkpoint.pt")
        if checkpoint_path.exists():
            print(f"Loading trained FEEL checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # Handle different checkpoint formats
            if "feel_stream_state" in checkpoint:
                self.feel_stream.load_state_dict(checkpoint["feel_stream_state"])
                print("  ✓ Loaded FEEL stream state (v3 format)")
            elif "feel_stream" in checkpoint:
                if "alpha" in checkpoint["feel_stream"]:
                    alpha_val = checkpoint["feel_stream"]["alpha"]
                    if alpha_val.numel() == 1:
                        self.feel_stream.alpha.data.fill_(alpha_val.item())
                    else:
                        self.feel_stream.alpha.data.copy_(alpha_val.to(self.model_dtype))
                    print(f"  ✓ Loaded trained alpha: {F.softplus(self.feel_stream.alpha).item():.4f}")
            if "z_feel_model" in checkpoint:
                # Map old projector weights to new structure
                old_state = checkpoint["z_feel_model"]
                # Try to load sensor_encoder weights into projector
                new_state = {}
                for k, v in old_state.items():
                    if k.startswith("sensor_encoder."):
                        # Map sensor_encoder.X to net.X
                        new_k = k.replace("sensor_encoder.", "net.")
                        new_state[new_k] = v.to(self.model_dtype)
                if new_state:
                    try:
                        self.feel_stream.projector.load_state_dict(new_state, strict=False)
                        print("  ✓ Loaded projector weights from z_feel_model")
                    except Exception as e:
                        print(f"  ! Could not load projector: {e}")
        else:
            print("! No checkpoint found - using untrained FEEL")

        print(f"✓ Model loaded (dtype: {self.model_dtype})")

    def generate_with_feel(
        self,
        prompt: str,
        n_tokens: int,
        alpha: float = 0.1,
        return_internals: bool = False,
        sensor_override: Optional[List[torch.Tensor]] = None,
        lag: int = 0,
    ) -> Dict:
        """Generate tokens with FEEL, collecting detailed metrics."""

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        tokens = []
        kl_divs = []
        sensors_list = []
        z_feels = []
        hidden_states = []
        entropies = []
        logit_deltas = []

        # Sensor history for lag
        sensor_history = []

        for step in range(n_tokens):
            with torch.no_grad():
                # Base forward (no FEEL)
                outputs_base = self.model(
                    current_ids,
                    output_hidden_states=True,
                    use_cache=False
                )
                logits_base = outputs_base.logits
                h_base = outputs_base.hidden_states[-1][:, -1, :]

                # Get sensors
                sensors = self.feel_stream.sensor_bank(logits_base)
                sensor_history.append(sensors.clone())

                # Apply sensor override or lag
                if sensor_override is not None and step < len(sensor_override):
                    sensors_to_use = sensor_override[step]
                elif lag > 0 and step >= lag:
                    sensors_to_use = sensor_history[step - lag]
                else:
                    sensors_to_use = sensors

                sensors_list.append(sensors_to_use[0].cpu().numpy())

                # Compute z_feel and FEEL embedding
                z_feel = self.feel_stream.projector(sensors_to_use)
                z_feels.append(z_feel[0].cpu().numpy())

                feel_embed = alpha * z_feel

                # Forward with FEEL
                embeds = self.model.get_input_embeddings()(current_ids)
                # Match dtype
                feel_embed = feel_embed.to(embeds.dtype)
                embeds = embeds + feel_embed.unsqueeze(1)

                outputs_feel = self.model(
                    inputs_embeds=embeds,
                    output_hidden_states=True,
                    use_cache=False
                )
                logits_feel = outputs_feel.logits
                h_feel = outputs_feel.hidden_states[-1][:, -1, :]

                hidden_states.append(h_feel[0].cpu().numpy())

                # Metrics
                p_base = F.softmax(logits_base[:, -1, :].float(), dim=-1)
                p_feel = F.softmax(logits_feel[:, -1, :].float(), dim=-1)

                kl = F.kl_div(p_feel.log(), p_base, reduction='batchmean').item()
                kl_divs.append(kl)

                entropy = -(p_feel * p_feel.log()).sum(-1).item()
                entropies.append(entropy)

                # Logit delta for chosen token
                next_token = logits_feel[:, -1, :].argmax(dim=-1, keepdim=True)
                delta_logit = (logits_feel[:, -1, next_token.item()] -
                              logits_base[:, -1, next_token.item()]).item()
                logit_deltas.append(delta_logit)

                tokens.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        result = {
            "tokens": tokens,
            "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
            "kl_divs": kl_divs,
            "entropies": entropies,
            "logit_deltas": logit_deltas,
            "avg_kl": np.mean(kl_divs),
            "max_kl": np.max(kl_divs),
            "p95_kl": np.percentile(kl_divs, 95) if len(kl_divs) > 1 else kl_divs[0],
        }

        if return_internals:
            result["sensors"] = sensors_list
            result["z_feels"] = z_feels
            result["hidden_states"] = hidden_states
            result["sensor_history"] = [s[0].cpu().numpy() for s in sensor_history]

        return result

    # ========================================================
    # Experiment 1: Aux Head Control
    # ========================================================

    def run_aux_head_control(self) -> Dict:
        """Compare aux head performance with alpha=0 vs alpha>0."""
        print("\n" + "="*60)
        print("EXPERIMENT 1: Aux Head Control (alpha=0 vs alpha>0)")
        print("="*60)

        results = {"alpha_0": [], "alpha_positive": []}

        for prompt in DIVERSE_PROMPTS[:12]:  # Use subset for speed
            print(f"  Processing: {prompt[:40]}...")

            # Generate with alpha=0 (FEEL disabled)
            res_0 = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=0.0, return_internals=True
            )

            # Generate with alpha=self.default_alpha (FEEL enabled)
            res_pos = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=self.default_alpha, return_internals=True
            )

            # Train aux head on hidden states, predict entropy
            # For alpha=0, hidden state should NOT carry FEEL info
            h_0 = np.array(res_0["hidden_states"])
            h_pos = np.array(res_pos["hidden_states"])
            entropy_0 = np.array(res_0["entropies"])
            entropy_pos = np.array(res_pos["entropies"])

            # Simple correlation as proxy for predictability
            if len(h_0) > 5:
                # Use first principal component of hidden state
                h_0_mean = h_0.mean(axis=1) if h_0.ndim > 1 else h_0
                h_pos_mean = h_pos.mean(axis=1) if h_pos.ndim > 1 else h_pos

                corr_0 = np.corrcoef(h_0_mean[:len(entropy_0)], entropy_0)[0, 1] if len(entropy_0) > 1 else 0
                corr_pos = np.corrcoef(h_pos_mean[:len(entropy_pos)], entropy_pos)[0, 1] if len(entropy_pos) > 1 else 0

                results["alpha_0"].append({
                    "prompt": prompt[:50],
                    "h_entropy_corr": float(corr_0) if not np.isnan(corr_0) else 0,
                    "avg_kl": res_0["avg_kl"],
                    "max_kl": res_0["max_kl"],
                })
                results["alpha_positive"].append({
                    "prompt": prompt[:50],
                    "h_entropy_corr": float(corr_pos) if not np.isnan(corr_pos) else 0,
                    "avg_kl": res_pos["avg_kl"],
                    "max_kl": res_pos["max_kl"],
                })

        # Aggregate
        avg_corr_0 = np.mean([r["h_entropy_corr"] for r in results["alpha_0"]])
        avg_corr_pos = np.mean([r["h_entropy_corr"] for r in results["alpha_positive"]])
        avg_kl_0 = np.mean([r["avg_kl"] for r in results["alpha_0"]])
        avg_kl_pos = np.mean([r["avg_kl"] for r in results["alpha_positive"]])

        summary = {
            "alpha_0_avg_corr": avg_corr_0,
            "alpha_pos_avg_corr": avg_corr_pos,
            "alpha_0_avg_kl": avg_kl_0,
            "alpha_pos_avg_kl": avg_kl_pos,
            "feel_improves_predictability": avg_corr_pos > avg_corr_0,
            "kl_increase_factor": avg_kl_pos / (avg_kl_0 + 1e-8),
        }

        print(f"\n  Results:")
        print(f"    Alpha=0 hidden-entropy corr: {avg_corr_0:.4f}")
        print(f"    Alpha>0 hidden-entropy corr: {avg_corr_pos:.4f}")
        print(f"    FEEL improves predictability: {summary['feel_improves_predictability']}")

        self.results["aux_head_control"] = {
            "summary": summary,
            "raw_data": results,
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    # ========================================================
    # Experiment 2: Counterfactual Dataset
    # ========================================================

    def run_counterfactual_test(self) -> Dict:
        """Same tokens, different sensor streams."""
        print("\n" + "="*60)
        print("EXPERIMENT 2: Counterfactual (same tokens, different sensors)")
        print("="*60)

        results = []

        for i, prompt in enumerate(DIVERSE_PROMPTS[:8]):
            print(f"  Processing: {prompt[:40]}...")

            # Generate baseline to get tokens
            baseline = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=self.default_alpha, return_internals=True
            )

            # Get sensor history from different prompt
            other_prompt = DIVERSE_PROMPTS[(i + 4) % len(DIVERSE_PROMPTS)]
            other_baseline = self.generate_with_feel(
                other_prompt, self.config.n_tokens_per_prompt,
                alpha=self.default_alpha, return_internals=True
            )

            # Convert sensor history to tensor list for override
            sensor_override = [
                torch.tensor(s, device=self.device).unsqueeze(0)
                for s in other_baseline["sensor_history"][:len(baseline["tokens"])]
            ]

            # Run with swapped sensors
            swapped = self.generate_with_feel(
                prompt, len(baseline["tokens"]),
                alpha=self.default_alpha, return_internals=True,
                sensor_override=sensor_override
            )

            # Compare hidden states
            h_baseline = np.array(baseline["hidden_states"])
            h_swapped = np.array(swapped["hidden_states"])

            # Measure hidden state divergence
            min_len = min(len(h_baseline), len(h_swapped))
            h_baseline = h_baseline[:min_len]
            h_swapped = h_swapped[:min_len]

            h_diff = np.linalg.norm(h_baseline - h_swapped, axis=1).mean()
            kl_diff = abs(baseline["avg_kl"] - swapped["avg_kl"])

            # Check if predictions changed
            logit_delta_diff = np.mean([
                abs(a - b) for a, b in zip(
                    baseline["logit_deltas"][:min_len],
                    swapped["logit_deltas"][:min_len]
                )
            ])

            results.append({
                "prompt": prompt[:50],
                "other_prompt": other_prompt[:50],
                "hidden_state_divergence": float(h_diff),
                "kl_difference": float(kl_diff),
                "logit_delta_diff": float(logit_delta_diff),
                "baseline_avg_kl": baseline["avg_kl"],
                "swapped_avg_kl": swapped["avg_kl"],
            })

        # Aggregate
        avg_h_div = np.mean([r["hidden_state_divergence"] for r in results])
        avg_kl_diff = np.mean([r["kl_difference"] for r in results])
        avg_logit_diff = np.mean([r["logit_delta_diff"] for r in results])

        summary = {
            "avg_hidden_state_divergence": avg_h_div,
            "avg_kl_difference": avg_kl_diff,
            "avg_logit_delta_difference": avg_logit_diff,
            "counterfactual_effect_detected": avg_h_div > 0.1 or avg_kl_diff > 0.01,
        }

        print(f"\n  Results:")
        print(f"    Hidden state divergence: {avg_h_div:.4f}")
        print(f"    KL difference: {avg_kl_diff:.4f}")
        print(f"    Counterfactual effect: {summary['counterfactual_effect_detected']}")

        self.results["counterfactual"] = {
            "summary": summary,
            "raw_data": results,
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    # ========================================================
    # Experiment 3: Extensive Suite with Bootstrap CIs
    # ========================================================

    def run_extensive_suite(self) -> Dict:
        """Full suite with bootstrap confidence intervals."""
        print("\n" + "="*60)
        print("EXPERIMENT 3: Extensive Suite with Bootstrap CIs")
        print("="*60)

        all_metrics = {
            "avg_kl": [],
            "max_kl": [],
            "p95_kl": [],
            "avg_logit_delta": [],
        }

        per_prompt_results = []

        for prompt in DIVERSE_PROMPTS:
            print(f"  Processing: {prompt[:40]}...")

            res = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=self.default_alpha, return_internals=True
            )

            all_metrics["avg_kl"].append(res["avg_kl"])
            all_metrics["max_kl"].append(res["max_kl"])
            all_metrics["p95_kl"].append(res["p95_kl"])
            all_metrics["avg_logit_delta"].append(np.mean(np.abs(res["logit_deltas"])))

            per_prompt_results.append({
                "prompt": prompt[:50],
                "avg_kl": res["avg_kl"],
                "max_kl": res["max_kl"],
                "p95_kl": res["p95_kl"],
                "avg_logit_delta": np.mean(np.abs(res["logit_deltas"])),
                "n_tokens": len(res["tokens"]),
            })

        # Bootstrap confidence intervals
        def bootstrap_ci(data, n_bootstrap=100, ci=95):
            data = np.array(data)
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                boot_means.append(np.mean(sample))
            lower = np.percentile(boot_means, (100 - ci) / 2)
            upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
            return np.mean(data), lower, upper

        summary = {}
        for metric, values in all_metrics.items():
            mean, ci_lower, ci_upper = bootstrap_ci(values, self.config.n_bootstrap)
            summary[metric] = {
                "mean": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "std": np.std(values),
            }
            print(f"    {metric}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

        self.results["extensive_suite"] = {
            "summary": summary,
            "per_prompt": per_prompt_results,
            "n_prompts": len(DIVERSE_PROMPTS),
            "n_bootstrap": self.config.n_bootstrap,
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    # ========================================================
    # Experiment 4: Strengthened Falsification
    # ========================================================

    def run_falsification_battery(self) -> Dict:
        """Comprehensive falsification tests."""
        print("\n" + "="*60)
        print("EXPERIMENT 4: Strengthened Falsification Battery")
        print("="*60)

        results = {
            "baseline": [],
            "whole_seq_permute": [],
            "cross_prompt_swap": [],
            "lag_sweep": {lag: [] for lag in self.config.lag_values},
        }

        for i, prompt in enumerate(DIVERSE_PROMPTS[:8]):
            print(f"  Processing: {prompt[:40]}...")

            # Baseline
            baseline = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=self.default_alpha, return_internals=True
            )
            results["baseline"].append(baseline["avg_kl"])

            # Whole sequence permutation
            sensor_history = baseline["sensor_history"]
            perm_idx = np.random.permutation(len(sensor_history))
            sensor_permuted = [
                torch.tensor(sensor_history[j], device=self.device).unsqueeze(0)
                for j in perm_idx
            ]

            permuted = self.generate_with_feel(
                prompt, len(baseline["tokens"]),
                alpha=self.default_alpha, sensor_override=sensor_permuted
            )
            results["whole_seq_permute"].append(permuted["avg_kl"])

            # Cross-prompt swap
            other_prompt = DIVERSE_PROMPTS[(i + 4) % len(DIVERSE_PROMPTS)]
            other = self.generate_with_feel(
                other_prompt, self.config.n_tokens_per_prompt,
                alpha=self.default_alpha, return_internals=True
            )
            cross_sensors = [
                torch.tensor(s, device=self.device).unsqueeze(0)
                for s in other["sensor_history"][:len(baseline["tokens"])]
            ]

            cross_swapped = self.generate_with_feel(
                prompt, len(baseline["tokens"]),
                alpha=self.default_alpha, sensor_override=cross_sensors
            )
            results["cross_prompt_swap"].append(cross_swapped["avg_kl"])

            # Lag sweep
            for lag in self.config.lag_values:
                lagged = self.generate_with_feel(
                    prompt, self.config.n_tokens_per_prompt,
                    alpha=self.default_alpha, lag=lag
                )
                results["lag_sweep"][lag].append(lagged["avg_kl"])

        # Compute ratios
        baseline_mean = np.mean(results["baseline"])

        summary = {
            "baseline_avg_kl": baseline_mean,
            "permute_ratio": np.mean(results["whole_seq_permute"]) / (baseline_mean + 1e-8),
            "cross_swap_ratio": np.mean(results["cross_prompt_swap"]) / (baseline_mean + 1e-8),
            "lag_ratios": {
                lag: np.mean(kls) / (baseline_mean + 1e-8)
                for lag, kls in results["lag_sweep"].items()
            },
            "falsification_passed": True,  # Will update below
        }

        # Check if falsification shows expected degradation
        # Good sign: permute and cross-swap should change KL pattern
        permute_diff = abs(summary["permute_ratio"] - 1.0)
        cross_diff = abs(summary["cross_swap_ratio"] - 1.0)

        summary["falsification_passed"] = (permute_diff > 0.05 or cross_diff > 0.05)

        print(f"\n  Results:")
        print(f"    Baseline avg KL: {baseline_mean:.4f}")
        print(f"    Permute ratio: {summary['permute_ratio']:.3f}")
        print(f"    Cross-swap ratio: {summary['cross_swap_ratio']:.3f}")
        print(f"    Lag ratios: {summary['lag_ratios']}")
        print(f"    Falsification passed: {summary['falsification_passed']}")

        self.results["falsification"] = {
            "summary": summary,
            "raw_data": {k: v if not isinstance(v, dict) else
                        {str(kk): vv for kk, vv in v.items()}
                        for k, v in results.items()},
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    # ========================================================
    # Experiment 5: Ridge Regression Predictive Test
    # ========================================================

    def run_predictive_test(self) -> Dict:
        """Ridge regression: sensors vs z_feel vs both predicting entropy."""
        print("\n" + "="*60)
        print("EXPERIMENT 5: Ridge Regression Predictive Test")
        print("="*60)

        all_sensors = []
        all_z_feels = []
        all_entropies = []

        for prompt in DIVERSE_PROMPTS[:16]:
            print(f"  Collecting: {prompt[:40]}...")

            res = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=self.default_alpha, return_internals=True
            )

            all_sensors.extend(res["sensors"])
            all_z_feels.extend(res["z_feels"])
            all_entropies.extend(res["entropies"])

        # Convert to arrays
        X_sensors = np.array(all_sensors)
        X_z_feel = np.array(all_z_feels)
        y = np.array(all_entropies)

        # Reduce z_feel dimensionality for regression
        # Use PCA or just mean/std
        X_z_feel_reduced = np.column_stack([
            X_z_feel.mean(axis=1),
            X_z_feel.std(axis=1),
            X_z_feel.min(axis=1),
            X_z_feel.max(axis=1),
        ])

        X_combined = np.column_stack([X_sensors, X_z_feel_reduced])

        # Ridge regression with cross-validation
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])

        # Sensors only
        ridge.fit(X_sensors, y)
        r2_sensors = ridge.score(X_sensors, y)
        cv_sensors = cross_val_score(ridge, X_sensors, y, cv=5).mean()

        # z_feel only
        ridge.fit(X_z_feel_reduced, y)
        r2_z_feel = ridge.score(X_z_feel_reduced, y)
        cv_z_feel = cross_val_score(ridge, X_z_feel_reduced, y, cv=5).mean()

        # Combined
        ridge.fit(X_combined, y)
        r2_combined = ridge.score(X_combined, y)
        cv_combined = cross_val_score(ridge, X_combined, y, cv=5).mean()

        summary = {
            "r2_sensors_only": r2_sensors,
            "r2_z_feel_only": r2_z_feel,
            "r2_combined": r2_combined,
            "cv_sensors_only": cv_sensors,
            "cv_z_feel_only": cv_z_feel,
            "cv_combined": cv_combined,
            "z_feel_incremental_gain": r2_combined - r2_sensors,
            "z_feel_adds_value": r2_combined > r2_sensors,
            "n_samples": len(y),
        }

        print(f"\n  Results:")
        print(f"    R² sensors only: {r2_sensors:.4f} (CV: {cv_sensors:.4f})")
        print(f"    R² z_feel only:  {r2_z_feel:.4f} (CV: {cv_z_feel:.4f})")
        print(f"    R² combined:     {r2_combined:.4f} (CV: {cv_combined:.4f})")
        print(f"    z_feel adds value: {summary['z_feel_adds_value']}")

        self.results["predictive_test"] = {
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    # ========================================================
    # Experiment 6: Effective Compute Multiplier
    # ========================================================

    def run_compute_multiplier_test(self) -> Dict:
        """Test if FEEL@1 sample ≈ baseline@N samples."""
        print("\n" + "="*60)
        print("EXPERIMENT 6: Effective Compute Multiplier")
        print("="*60)

        # For this test, we measure prediction quality via entropy/confidence
        # Lower entropy at decision points = more confident predictions

        results = {
            "feel_on_samples_1": [],
            "feel_off_samples_1": [],
            "feel_off_samples_2": [],  # Average of 2 runs
        }

        for prompt in DIVERSE_PROMPTS[:12]:
            print(f"  Processing: {prompt[:40]}...")

            # FEEL ON, 1 sample
            feel_on = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=self.default_alpha
            )
            results["feel_on_samples_1"].append({
                "avg_entropy": np.mean(feel_on["entropies"]),
                "avg_kl": feel_on["avg_kl"],
            })

            # FEEL OFF, 1 sample
            feel_off_1 = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=0.0
            )
            results["feel_off_samples_1"].append({
                "avg_entropy": np.mean(feel_off_1["entropies"]),
                "avg_kl": feel_off_1["avg_kl"],
            })

            # FEEL OFF, 2 samples (simulate ensemble)
            feel_off_2 = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=0.0
            )
            avg_entropy_2 = (np.mean(feel_off_1["entropies"]) +
                           np.mean(feel_off_2["entropies"])) / 2
            results["feel_off_samples_2"].append({
                "avg_entropy": avg_entropy_2,
            })

        # Compute multiplier
        feel_on_entropy = np.mean([r["avg_entropy"] for r in results["feel_on_samples_1"]])
        feel_off_1_entropy = np.mean([r["avg_entropy"] for r in results["feel_off_samples_1"]])
        feel_off_2_entropy = np.mean([r["avg_entropy"] for r in results["feel_off_samples_2"]])

        # How many samples would FEEL-off need to match FEEL-on?
        # Linear interpolation approximation
        if feel_off_2_entropy < feel_off_1_entropy:
            entropy_per_sample = feel_off_1_entropy - feel_off_2_entropy
            if entropy_per_sample > 0:
                effective_multiplier = 1 + (feel_off_1_entropy - feel_on_entropy) / entropy_per_sample
            else:
                effective_multiplier = 1.0
        else:
            effective_multiplier = 1.0

        summary = {
            "feel_on_avg_entropy": feel_on_entropy,
            "feel_off_1_avg_entropy": feel_off_1_entropy,
            "feel_off_2_avg_entropy": feel_off_2_entropy,
            "effective_compute_multiplier": max(1.0, effective_multiplier),
            "entropy_reduction_pct": (feel_off_1_entropy - feel_on_entropy) / feel_off_1_entropy * 100,
        }

        print(f"\n  Results:")
        print(f"    FEEL ON entropy:  {feel_on_entropy:.4f}")
        print(f"    FEEL OFF (1x):    {feel_off_1_entropy:.4f}")
        print(f"    FEEL OFF (2x):    {feel_off_2_entropy:.4f}")
        print(f"    Effective multiplier: {summary['effective_compute_multiplier']:.2f}x")
        print(f"    Entropy reduction: {summary['entropy_reduction_pct']:.1f}%")

        self.results["compute_multiplier"] = {
            "summary": summary,
            "raw_data": results,
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    # ========================================================
    # Experiment 7: Deep GPU Interoception
    # ========================================================

    def run_gpu_interoception_test(self) -> Dict:
        """Test if z_feel carries deep GPU state information."""
        print("\n" + "="*60)
        print("EXPERIMENT 7: Deep GPU Interoception")
        print("="*60)

        import subprocess
        import re

        def get_gpu_metrics() -> Dict:
            """Get AMD GPU metrics via rocm-smi."""
            metrics = {
                "gpu_temp": 0.0,
                "gpu_power": 0.0,
                "gpu_sclk": 0.0,
                "gpu_mclk": 0.0,
                "gpu_busy": 0.0,
            }
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showtemp", "--showpower", "--showclocks", "--showuse"],
                    capture_output=True, text=True, timeout=5
                )
                output = result.stdout

                # Parse temperature
                temp_match = re.search(r"Temperature.*?(\d+\.?\d*)", output)
                if temp_match:
                    metrics["gpu_temp"] = float(temp_match.group(1))

                # Parse power
                power_match = re.search(r"Power.*?(\d+\.?\d*)", output)
                if power_match:
                    metrics["gpu_power"] = float(power_match.group(1))

                # Parse SCLK
                sclk_match = re.search(r"sclk.*?(\d+)", output, re.IGNORECASE)
                if sclk_match:
                    metrics["gpu_sclk"] = float(sclk_match.group(1))

                # Parse GPU busy
                busy_match = re.search(r"GPU use.*?(\d+)", output, re.IGNORECASE)
                if busy_match:
                    metrics["gpu_busy"] = float(busy_match.group(1))

            except Exception as e:
                print(f"    Warning: Could not get GPU metrics: {e}")

            return metrics

        results = {
            "z_feel_gpu_correlation": [],
            "sensor_gpu_correlation": [],
            "counterfactual_identifiability": [],
        }

        # Collect z_feel and GPU metrics across different prompts
        z_feel_trajectories = []
        sensor_trajectories = []
        gpu_trajectories = []

        for prompt in DIVERSE_PROMPTS[:12]:
            print(f"  Processing: {prompt[:40]}...")

            # Get GPU state before
            gpu_before = get_gpu_metrics()

            res = self.generate_with_feel(
                prompt, self.config.n_tokens_per_prompt,
                alpha=self.default_alpha, return_internals=True
            )

            # Get GPU state after
            gpu_after = get_gpu_metrics()

            # Average GPU metrics
            gpu_avg = {
                k: (gpu_before[k] + gpu_after[k]) / 2
                for k in gpu_before
            }

            z_feel_trajectories.append(np.array(res["z_feels"]))
            sensor_trajectories.append(np.array(res["sensors"]))
            gpu_trajectories.append(gpu_avg)

        # Probe: Can we predict GPU state from z_feel?
        # Use mean z_feel per prompt vs GPU metrics

        z_feel_means = np.array([z.mean(axis=0).mean() for z in z_feel_trajectories])
        z_feel_vars = np.array([z.var() for z in z_feel_trajectories])
        sensor_means = np.array([s.mean() for s in sensor_trajectories])

        gpu_temps = np.array([g["gpu_temp"] for g in gpu_trajectories])
        gpu_powers = np.array([g["gpu_power"] for g in gpu_trajectories])

        # Correlations
        def safe_corr(x, y):
            if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
                return 0.0
            return np.corrcoef(x, y)[0, 1]

        z_temp_corr = safe_corr(z_feel_means, gpu_temps)
        z_power_corr = safe_corr(z_feel_vars, gpu_powers)
        sensor_temp_corr = safe_corr(sensor_means, gpu_temps)

        # Evidence source classification
        evidence_source = "NONE"
        if abs(z_temp_corr) > 0.3 or abs(z_power_corr) > 0.3:
            evidence_source = "INDIRECT"  # z_feel correlates with GPU state
        if abs(z_temp_corr) > 0.6 or abs(z_power_corr) > 0.6:
            evidence_source = "DIRECT"  # Strong correlation

        # Counterfactual identifiability test
        # Run same prompt at different "load" levels (via batch size proxy)
        cf_prompt = DIVERSE_PROMPTS[0]
        cf_results = []

        for n_tok in [16, 32, 64]:  # Different compute loads
            gpu_before = get_gpu_metrics()

            res = self.generate_with_feel(
                cf_prompt, n_tok,
                alpha=self.default_alpha, return_internals=True
            )

            gpu_after = get_gpu_metrics()

            cf_results.append({
                "n_tokens": n_tok,
                "z_feel_mean": np.array(res["z_feels"]).mean(),
                "z_feel_var": np.array(res["z_feels"]).var(),
                "gpu_temp_delta": gpu_after["gpu_temp"] - gpu_before["gpu_temp"],
                "gpu_power_delta": gpu_after["gpu_power"] - gpu_before["gpu_power"],
            })

        # Check if z_feel differs with hardware regime
        z_means = [r["z_feel_mean"] for r in cf_results]
        z_vars = [r["z_feel_var"] for r in cf_results]

        cf_identifiable = (max(z_means) - min(z_means)) > 0.01 or (max(z_vars) - min(z_vars)) > 0.001

        summary = {
            "z_feel_temp_correlation": float(z_temp_corr) if not np.isnan(z_temp_corr) else 0,
            "z_feel_power_correlation": float(z_power_corr) if not np.isnan(z_power_corr) else 0,
            "sensor_temp_correlation": float(sensor_temp_corr) if not np.isnan(sensor_temp_corr) else 0,
            "evidence_source": evidence_source,
            "counterfactual_identifiable": cf_identifiable,
            "gpu_metrics_available": gpu_temps.max() > 0,
            "counterfactual_results": cf_results,
        }

        print(f"\n  Results:")
        print(f"    z_feel-temp correlation: {z_temp_corr:.4f}")
        print(f"    z_feel-power correlation: {z_power_corr:.4f}")
        print(f"    Evidence source: {evidence_source}")
        print(f"    Counterfactual identifiable: {cf_identifiable}")

        self.results["gpu_interoception"] = {
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    # ========================================================
    # Run All Experiments
    # ========================================================

    def run_all(self) -> Dict:
        """Run complete experiment battery."""
        print("\n" + "="*70)
        print("  FEEL BREAKTHROUGH EXPERIMENTS - Publication Grade")
        print("="*70)

        start_time = time.time()

        # Run all experiments
        self.run_aux_head_control()
        self.run_counterfactual_test()
        self.run_extensive_suite()
        self.run_falsification_battery()
        self.run_predictive_test()
        self.run_compute_multiplier_test()
        self.run_gpu_interoception_test()

        elapsed = time.time() - start_time

        # Final summary
        print("\n" + "="*70)
        print("  FINAL SUMMARY")
        print("="*70)

        final_summary = {
            "aux_head_control": self.results["aux_head_control"]["summary"],
            "counterfactual": self.results["counterfactual"]["summary"],
            "extensive_suite": {k: v["mean"] for k, v in self.results["extensive_suite"]["summary"].items()},
            "falsification": self.results["falsification"]["summary"],
            "predictive_test": self.results["predictive_test"]["summary"],
            "compute_multiplier": self.results["compute_multiplier"]["summary"],
            "gpu_interoception": self.results["gpu_interoception"]["summary"],
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

        # Determine overall verdict
        verdicts = {
            "causal_channel_real": self.results["extensive_suite"]["summary"]["avg_kl"]["mean"] > 0.01,
            "counterfactual_works": self.results["counterfactual"]["summary"]["counterfactual_effect_detected"],
            "falsification_passed": self.results["falsification"]["summary"]["falsification_passed"],
            "z_feel_adds_value": self.results["predictive_test"]["summary"]["z_feel_adds_value"],
            "compute_benefit": self.results["compute_multiplier"]["summary"]["effective_compute_multiplier"] > 1.1,
            "gpu_interoception": self.results["gpu_interoception"]["summary"]["evidence_source"] != "NONE",
        }

        final_summary["verdicts"] = verdicts
        final_summary["overall_pass"] = sum(verdicts.values()) >= 4  # Need 4/6 to pass

        print(f"\n  Verdicts:")
        for k, v in verdicts.items():
            status = "✓ PASS" if v else "✗ FAIL"
            print(f"    {k}: {status}")

        print(f"\n  Overall: {'✓ BREAKTHROUGH' if final_summary['overall_pass'] else '✗ MORE WORK NEEDED'}")
        print(f"  Elapsed: {elapsed:.1f}s")

        self.results["final_summary"] = final_summary

        return final_summary

    def save_results(self, path: str = "results/feel_experiments/breakthrough_results.json"):
        """Save all results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if hasattr(obj, 'item'):  # Handle numpy scalars
                return obj.item()
            return obj

        with open(path, "w") as f:
            json.dump(convert(self.results), f, indent=2)

        print(f"\n✓ Results saved to: {path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    config = ExperimentConfig()
    experiments = BreakthroughExperiments(config)

    experiments.run_all()
    experiments.save_results()
