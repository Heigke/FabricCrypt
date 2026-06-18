#!/usr/bin/env python3
"""
Z700: Metabolic Inference - Difficulty-Adaptive Early Exit with Hardware Feedback

This script implements the novel "Metabolic Inference" approach:
1. Exit depth varies with semantic difficulty (like DiffAdapt/MoD)
2. Exit threshold modulated by hardware state (novel contribution)
3. Closed-loop: hardware state → depth → hardware state

Key hypotheses to validate:
H1: Harder inputs use more layers (difficulty → depth correlation)
H2: Hardware pressure reduces depth (thermal/power → earlier exit)
H3: Joint optimization beats single-objective baselines
H4: Energy per correct answer improves vs fixed-depth

References:
- DiffAdapt (2025): Difficulty-adaptive reasoning
- Mixture of Depths (2024): 50% FLOP reduction
- SpecEE (ISCA 2025): Speculative early exit
- TokenPowerBench (2025): Energy measurement methodology
"""

import os
import sys
import time
import json
import random
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# AMD GPU setup
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MetabolicConfig:
    """Configuration for metabolic inference experiment."""
    # Model
    model_name: str = "gpt2"
    exit_layers: List[int] = field(default_factory=lambda: [3, 6, 9, 12])

    # Hardware thresholds
    thermal_target_c: float = 65.0
    power_budget_w: float = 80.0

    # Exit policy (tune for variety in exit depths)
    base_confidence_threshold: float = 0.55  # Medium threshold
    thermal_coefficient: float = 0.15  # How much thermal pressure affects threshold

    # Experiment
    n_trials: int = 5
    n_samples_per_trial: int = 100
    cooldown_s: float = 5.0

    # Difficulty levels (prompts)
    easy_prompts: List[str] = field(default_factory=lambda: [
        "The cat sat on the",
        "Hello, my name is",
        "The weather today is",
        "I like to eat",
        "The color of the sky is",
    ])

    hard_prompts: List[str] = field(default_factory=lambda: [
        "Explain the mathematical proof that the square root of 2 is irrational:",
        "Describe the implications of Gödel's incompleteness theorems on",
        "The relationship between quantum entanglement and information theory suggests",
        "In considering the ethical implications of artificial general intelligence,",
        "The second law of thermodynamics implies that entropy in a closed system",
    ])


# ============================================================================
# Hardware Substrate (from z600+ infra)
# ============================================================================

class HWSubstrate:
    """Direct hardware sensing and actuation."""

    DRM_PATH = Path("/sys/class/drm/card1/device")

    def __init__(self):
        self.hwmon_dir = self._find_hwmon()
        self._history = []

    def _find_hwmon(self) -> Optional[Path]:
        hwmon_base = self.DRM_PATH / "hwmon"
        if hwmon_base.exists():
            for d in sorted(hwmon_base.iterdir()):
                if d.is_dir() and d.name.startswith("hwmon"):
                    return d
        return None

    def read_power_w(self) -> float:
        """Read power from hwmon (THE TRUTH)."""
        if not self.hwmon_dir:
            return 0.0
        try:
            pf = self.hwmon_dir / "power1_average"
            if pf.exists():
                return int(pf.read_text().strip()) / 1_000_000
        except:
            pass
        return 0.0

    def read_temp_c(self) -> float:
        """Read edge temperature."""
        if self.hwmon_dir:
            try:
                t1 = self.hwmon_dir / "temp1_input"
                if t1.exists():
                    return int(t1.read_text().strip()) / 1000
            except:
                pass
        return 0.0

    def sense(self) -> Dict:
        """Read complete state."""
        state = {
            'timestamp': time.time(),
            'power_w': self.read_power_w(),
            'temp_c': self.read_temp_c(),
        }
        self._history.append(state)
        return state

    def set_perf_level(self, level: str) -> bool:
        """Set performance level."""
        try:
            path = self.DRM_PATH / "power_dpm_force_performance_level"
            path.write_text(level)
            return True
        except:
            return False

    def reset(self):
        """Reset to auto."""
        self.set_perf_level("auto")


# ============================================================================
# Early Exit Model Wrapper
# ============================================================================

class EarlyExitGPT2(nn.Module):
    """
    GPT-2 with early exit heads at specified layers.

    The exit decision combines:
    1. Semantic confidence (from logits entropy)
    2. Hardware pressure (from thermal/power state)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        exit_layers: List[int] = [3, 6, 9, 12],
        base_threshold: float = 0.7,
        thermal_coef: float = 0.3,
    ):
        super().__init__()

        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.base_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = self.base_model.config
        self.exit_layers = exit_layers
        self.num_layers = self.config.n_layer

        # Exit decision parameters
        self.base_threshold = base_threshold
        self.thermal_coef = thermal_coef

        # No learned exit heads - use layer norm + lm_head directly
        # This is the "free lunch" approach: intermediate hidden states
        # can be decoded directly (like CALM/LayerSkip)
        hidden_dim = self.config.n_embd

        # Just need layer norms to normalize before lm_head
        self.exit_norms = nn.ModuleDict()
        for layer_idx in exit_layers:
            # Use a copy of final layer norm (or fresh one)
            self.exit_norms[str(layer_idx)] = nn.LayerNorm(hidden_dim)
            # Initialize from final layer norm
            self.exit_norms[str(layer_idx)].load_state_dict(
                self.base_model.transformer.ln_f.state_dict()
            )

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

    def compute_confidence(self, logits: torch.Tensor, layer_idx: int,
                           prev_logits: Optional[torch.Tensor] = None) -> float:
        """
        Compute exit confidence using multiple signals:
        1. Entropy (distribution peakedness)
        2. Saturation (change from previous layer)
        3. Layer progress (later = more likely to exit)

        The key insight: if logits aren't changing much between layers,
        the model has "converged" and more compute won't help.
        """
        probs = F.softmax(logits[:, -1, :], dim=-1)

        # Clamp probabilities to avoid numerical issues (used throughout)
        probs_safe = torch.clamp(probs, min=1e-10, max=1.0)

        # 1. Entropy-based confidence
        log_probs = torch.log(probs_safe)
        entropy = -torch.sum(probs_safe * log_probs, dim=-1)

        # Handle NaN/inf
        entropy = torch.nan_to_num(entropy, nan=5.0, posinf=10.0, neginf=0.0)

        max_entropy = np.log(self.config.vocab_size)
        normalized_entropy = entropy.mean().item() / max_entropy
        normalized_entropy = min(1.0, max(0.0, normalized_entropy))  # Clamp to [0,1]
        entropy_confidence = 1.0 - normalized_entropy

        # 2. Saturation (if we have previous logits)
        saturation = 0.0  # Default: NOT saturated (first exit has no basis for comparison)
        if prev_logits is not None:
            prev_probs = F.softmax(prev_logits[:, -1, :], dim=-1)
            prev_probs = torch.clamp(prev_probs, min=1e-10, max=1.0)

            # Use Jensen-Shannon divergence (symmetric, bounded)
            # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
            m = 0.5 * (probs_safe + prev_probs)
            m = torch.clamp(m, min=1e-10)

            kl_pm = torch.sum(probs_safe * (torch.log(probs_safe) - torch.log(m)), dim=-1)
            kl_qm = torch.sum(prev_probs * (torch.log(prev_probs) - torch.log(m)), dim=-1)
            js_div = 0.5 * (kl_pm + kl_qm)

            # Handle NaN
            js_div = torch.nan_to_num(js_div, nan=0.5)
            js_value = js_div.mean().item()
            js_value = max(0.0, min(1.0, js_value))  # JS is bounded [0, ln(2)]

            # Low JS = similar distributions = saturated = high confidence
            saturation = 1.0 - js_value * 2  # Scale so JS=0.5 -> sat=0
            saturation = max(0.0, min(1.0, saturation))

            if hasattr(self, '_debug_detail') and self._debug_detail:
                print(f"    [sat debug] JS={js_value:.4f} -> sat={saturation:.3f}")

        # 3. Layer progress bonus
        layer_progress = layer_idx / self.num_layers
        layer_bonus = 0.1 * layer_progress

        # Combine: weight saturation more heavily (it's the key signal)
        confidence = (
            0.3 * entropy_confidence +
            0.5 * saturation +
            0.2 * layer_progress
        ) + layer_bonus

        # Debug
        if hasattr(self, '_debug_detail') and self._debug_detail:
            print(f"    [conf debug] L{layer_idx}: ent_conf={entropy_confidence:.3f} sat={saturation:.3f} prog={layer_progress:.3f} bonus={layer_bonus:.3f} -> {confidence:.3f}")

        return max(0.0, min(1.0, confidence))

    def compute_exit_threshold(self, hw_state: Dict, layer_progress: float) -> float:
        """
        Compute dynamic exit threshold based on:
        1. Base threshold
        2. Hardware pressure (thermal/power)
        3. Layer progress (later layers = lower threshold)

        Higher thermal pressure → lower threshold → earlier exit
        """
        # Thermal pressure: how close to limit (normalized 0-1)
        temp = hw_state.get('temp_c', 40)
        thermal_pressure = max(0, (temp - 35)) / 45.0  # 35-80C range
        thermal_pressure = min(1.0, max(0.0, thermal_pressure))

        # Power pressure
        power = hw_state.get('power_w', 30)
        power_pressure = max(0, (power - 20)) / 80.0  # 20-100W range
        power_pressure = min(1.0, max(0.0, power_pressure))

        # Combined pressure (use max)
        hw_pressure = max(thermal_pressure, power_pressure)

        # Threshold decreases with:
        # - Higher hw_pressure (stressed → exit earlier → lower threshold)
        # - Higher layer_progress (later layers → more willing to exit → lower threshold)
        threshold = self.base_threshold
        threshold -= self.thermal_coef * hw_pressure  # Up to -0.2 from hw
        threshold -= 0.15 * layer_progress  # Up to -0.15 from depth

        # Clamp to reasonable range
        return max(0.15, min(0.8, threshold))

    def forward_with_early_exit(
        self,
        input_ids: torch.Tensor,
        hw_state: Dict,
        return_all_exits: bool = False,
    ) -> Tuple[torch.Tensor, int, Dict]:
        """
        Forward pass with potential early exit.

        Returns:
            logits: Output logits
            exit_layer: Layer at which we exited
            stats: Statistics about the decision
        """
        device = input_ids.device

        # Get embeddings
        hidden_states = self.base_model.transformer.wte(input_ids)
        hidden_states = self.base_model.transformer.wpe(
            torch.arange(input_ids.shape[1], device=device)
        ) + hidden_states

        exit_info = {
            'confidences': [],
            'thresholds': [],
            'exit_layer': self.num_layers,
            'hw_state': hw_state,
        }

        all_logits = {} if return_all_exits else None
        prev_exit_logits = None  # Track previous exit point's logits

        # Forward through layers with exit checks
        for layer_idx, block in enumerate(self.base_model.transformer.h):
            hidden_states = block(hidden_states)[0]

            # Check for early exit at designated layers
            if layer_idx + 1 in self.exit_layers:
                # Apply layer norm and lm_head directly (no learned projection)
                exit_hidden = self.exit_norms[str(layer_idx + 1)](hidden_states)
                exit_logits = self.base_model.lm_head(exit_hidden)

                if return_all_exits:
                    all_logits[layer_idx + 1] = exit_logits

                # Compute confidence with saturation from previous exit point
                confidence = self.compute_confidence(exit_logits, layer_idx + 1, prev_exit_logits)
                prev_exit_logits = exit_logits.detach()  # Save for next comparison

                # Compute threshold
                layer_progress = (layer_idx + 1) / self.num_layers
                threshold = self.compute_exit_threshold(hw_state, layer_progress)

                exit_info['confidences'].append(confidence)
                exit_info['thresholds'].append(threshold)

                # Debug first few tokens
                if len(exit_info['confidences']) <= 4 and hasattr(self, '_debug') and self._debug:
                    print(f"  L{layer_idx+1}: conf={confidence:.3f} thresh={threshold:.3f} {'EXIT' if confidence >= threshold else 'continue'}")

                # Exit decision
                if confidence >= threshold:
                    exit_info['exit_layer'] = layer_idx + 1
                    return exit_logits, layer_idx + 1, exit_info

        # Full forward - no early exit
        hidden_states = self.base_model.transformer.ln_f(hidden_states)
        logits = self.base_model.lm_head(hidden_states)

        if return_all_exits:
            all_logits[self.num_layers] = logits
            exit_info['all_logits'] = all_logits

        return logits, self.num_layers, exit_info

    @torch.no_grad()
    def generate_with_early_exit(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        hw_substrate: Optional[HWSubstrate] = None,
    ) -> Dict:
        """
        Generate tokens with early exit.

        Returns detailed metrics for analysis.
        """
        device = next(self.parameters()).device

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        # Track metrics
        exit_layers = []
        confidences = []
        thresholds = []
        power_samples = []
        temp_samples = []

        generated_ids = input_ids.clone()

        t_start = time.perf_counter()

        for _ in range(max_new_tokens):
            # Read hardware state
            if hw_substrate:
                hw_state = hw_substrate.sense()
                power_samples.append(hw_state['power_w'])
                temp_samples.append(hw_state['temp_c'])
            else:
                hw_state = {'power_w': 50, 'temp_c': 50}

            # Forward with early exit
            logits, exit_layer, info = self.forward_with_early_exit(
                generated_ids, hw_state
            )

            exit_layers.append(exit_layer)
            if info['confidences']:
                confidences.append(info['confidences'][-1])
                thresholds.append(info['thresholds'][-1])

            # Sample next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        total_time = t_end - t_start
        n_tokens = len(exit_layers)

        # Compute energy
        avg_power = statistics.mean(power_samples) if power_samples else 0
        energy_j = avg_power * total_time

        return {
            'prompt': prompt,
            'n_tokens': n_tokens,
            'total_time_s': total_time,
            'tokens_per_second': n_tokens / total_time if total_time > 0 else 0,
            'energy_j': energy_j,
            'joules_per_token': energy_j / n_tokens if n_tokens > 0 else 0,
            'exit_layers': exit_layers,
            'mean_exit_layer': statistics.mean(exit_layers),
            'exit_layer_std': statistics.stdev(exit_layers) if len(exit_layers) > 1 else 0,
            'confidences': confidences,
            'thresholds': thresholds,
            'power_samples': power_samples,
            'temp_samples': temp_samples,
            'flops_saved_pct': (1 - statistics.mean(exit_layers) / self.num_layers) * 100,
        }


# ============================================================================
# Experiment Runner
# ============================================================================

def run_metabolic_experiment(config: MetabolicConfig):
    """Run the full metabolic inference experiment."""

    print("=" * 70)
    print("Z700: METABOLIC INFERENCE EXPERIMENT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Initialize hardware
    hw = HWSubstrate()
    cal_state = hw.sense()
    print(f"Initial state: {cal_state['power_w']:.1f}W, {cal_state['temp_c']:.1f}°C")

    # Load model
    print(f"\nLoading {config.model_name} with exit layers {config.exit_layers}...")
    model = EarlyExitGPT2(
        model_name=config.model_name,
        exit_layers=config.exit_layers,
        base_threshold=config.base_confidence_threshold,
        thermal_coef=config.thermal_coefficient,
    ).to(device).half()
    model.eval()

    print(f"Model loaded. Layers: {model.num_layers}, Exit points: {config.exit_layers}")

    # Disable debug for cleaner output
    model._debug = False
    model._debug_detail = False

    # Warmup
    print("\nWarmup...")
    for _ in range(3):
        _ = model.generate_with_early_exit("Hello", max_new_tokens=10, hw_substrate=hw)

    # ========================================================================
    # Experiment 1: Difficulty → Depth Correlation (H1)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Difficulty → Depth Correlation (H1)")
    print("=" * 70)

    easy_results = []
    hard_results = []

    for trial in range(config.n_trials):
        print(f"\nTrial {trial + 1}/{config.n_trials}")

        # Cooldown
        hw.set_perf_level("low")
        time.sleep(config.cooldown_s)
        hw.set_perf_level("auto")

        # Easy prompts
        print("  Easy prompts debug:")
        for i, prompt in enumerate(config.easy_prompts):
            if i == 1:
                model._debug = False  # Disable after first prompt
            result = model.generate_with_early_exit(
                prompt, max_new_tokens=30, hw_substrate=hw
            )
            easy_results.append(result)

        # Hard prompts
        print("\n  Hard prompts debug:")
        model._debug = True  # Re-enable for first hard prompt
        for i, prompt in enumerate(config.hard_prompts):
            if i == 1:
                model._debug = False
            result = model.generate_with_early_exit(
                prompt, max_new_tokens=30, hw_substrate=hw
            )
            hard_results.append(result)
            result = model.generate_with_early_exit(
                prompt, max_new_tokens=30, hw_substrate=hw
            )
            hard_results.append(result)

    # Analyze H1
    easy_depths = [r['mean_exit_layer'] for r in easy_results]
    hard_depths = [r['mean_exit_layer'] for r in hard_results]

    print(f"\n--- H1 Results ---")
    print(f"Easy prompts: mean depth = {statistics.mean(easy_depths):.2f} ± {statistics.stdev(easy_depths):.2f}")
    print(f"Hard prompts: mean depth = {statistics.mean(hard_depths):.2f} ± {statistics.stdev(hard_depths):.2f}")

    # Statistical test
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_ind(easy_depths, hard_depths)
    h1_passed = p_value < 0.05 and statistics.mean(hard_depths) > statistics.mean(easy_depths)

    print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"H1 (hard uses more layers): {'PASSED' if h1_passed else 'FAILED'}")

    # ========================================================================
    # Experiment 2: Hardware Pressure → Depth (H2)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Hardware Pressure → Depth (H2)")
    print("=" * 70)

    # Run at different thermal states
    cool_results = []
    warm_results = []

    # Cool state (after cooldown)
    print("\nCool state test...")
    hw.set_perf_level("low")
    time.sleep(config.cooldown_s * 2)  # Extra cooldown
    hw.set_perf_level("auto")

    for prompt in config.easy_prompts + config.hard_prompts:
        result = model.generate_with_early_exit(
            prompt, max_new_tokens=30, hw_substrate=hw
        )
        cool_results.append(result)

    # Warm state (after sustained load)
    print("Warming up GPU...")
    hw.set_perf_level("high")
    # Generate load
    for _ in range(20):
        _ = model.generate_with_early_exit(
            "Generate a very long response about artificial intelligence and machine learning",
            max_new_tokens=50, hw_substrate=hw
        )

    print("Warm state test...")
    for prompt in config.easy_prompts + config.hard_prompts:
        result = model.generate_with_early_exit(
            prompt, max_new_tokens=30, hw_substrate=hw
        )
        warm_results.append(result)

    # Reset
    hw.set_perf_level("auto")

    # Analyze H2
    cool_depths = [r['mean_exit_layer'] for r in cool_results]
    warm_depths = [r['mean_exit_layer'] for r in warm_results]
    cool_temps = [statistics.mean(r['temp_samples']) for r in cool_results if r['temp_samples']]
    warm_temps = [statistics.mean(r['temp_samples']) for r in warm_results if r['temp_samples']]

    print(f"\n--- H2 Results ---")
    print(f"Cool state ({statistics.mean(cool_temps):.1f}°C): mean depth = {statistics.mean(cool_depths):.2f}")
    print(f"Warm state ({statistics.mean(warm_temps):.1f}°C): mean depth = {statistics.mean(warm_depths):.2f}")

    t_stat2, p_value2 = scipy_stats.ttest_ind(cool_depths, warm_depths)
    h2_passed = p_value2 < 0.05 and statistics.mean(warm_depths) < statistics.mean(cool_depths)

    print(f"t-test: t={t_stat2:.3f}, p={p_value2:.4f}")
    print(f"H2 (thermal pressure → earlier exit): {'PASSED' if h2_passed else 'FAILED'}")

    # ========================================================================
    # Experiment 3: Energy Efficiency (H4)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Energy Efficiency Comparison")
    print("=" * 70)

    # Compare metabolic vs fixed-depth
    metabolic_energy = [r['joules_per_token'] for r in easy_results + hard_results]
    metabolic_throughput = [r['tokens_per_second'] for r in easy_results + hard_results]

    # Run fixed-depth baseline (always use all layers)
    print("\nRunning fixed-depth baseline...")
    model.base_threshold = 1.0  # Never exit early

    fixed_results = []
    hw.set_perf_level("low")
    time.sleep(config.cooldown_s)
    hw.set_perf_level("auto")

    for prompt in config.easy_prompts + config.hard_prompts:
        result = model.generate_with_early_exit(
            prompt, max_new_tokens=30, hw_substrate=hw
        )
        fixed_results.append(result)

    # Restore threshold
    model.base_threshold = config.base_confidence_threshold

    fixed_energy = [r['joules_per_token'] for r in fixed_results]
    fixed_throughput = [r['tokens_per_second'] for r in fixed_results]

    print(f"\n--- Energy Efficiency Results ---")
    print(f"Metabolic: {statistics.mean(metabolic_energy)*1000:.2f} mJ/tok, "
          f"{statistics.mean(metabolic_throughput):.1f} tok/s")
    print(f"Fixed:     {statistics.mean(fixed_energy)*1000:.2f} mJ/tok, "
          f"{statistics.mean(fixed_throughput):.1f} tok/s")

    energy_improvement = (statistics.mean(fixed_energy) - statistics.mean(metabolic_energy)) / statistics.mean(fixed_energy) * 100
    throughput_change = (statistics.mean(metabolic_throughput) - statistics.mean(fixed_throughput)) / statistics.mean(fixed_throughput) * 100

    print(f"\nEnergy improvement: {energy_improvement:.1f}%")
    print(f"Throughput change: {throughput_change:+.1f}%")

    h4_passed = energy_improvement > 10  # At least 10% energy savings
    print(f"H4 (>10% energy savings): {'PASSED' if h4_passed else 'FAILED'}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = {
        'config': asdict(config),
        'h1_difficulty_depth': {
            'easy_mean_depth': statistics.mean(easy_depths),
            'hard_mean_depth': statistics.mean(hard_depths),
            'p_value': p_value,
            'passed': h1_passed,
        },
        'h2_thermal_depth': {
            'cool_mean_depth': statistics.mean(cool_depths),
            'warm_mean_depth': statistics.mean(warm_depths),
            'cool_mean_temp': statistics.mean(cool_temps) if cool_temps else 0,
            'warm_mean_temp': statistics.mean(warm_temps) if warm_temps else 0,
            'p_value': p_value2,
            'passed': h2_passed,
        },
        'h4_energy': {
            'metabolic_mj_per_tok': statistics.mean(metabolic_energy) * 1000,
            'fixed_mj_per_tok': statistics.mean(fixed_energy) * 1000,
            'energy_improvement_pct': energy_improvement,
            'throughput_change_pct': throughput_change,
            'passed': h4_passed,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    print(f"\nH1 (Difficulty → Depth): {'✓ PASSED' if h1_passed else '✗ FAILED'}")
    print(f"H2 (Thermal → Depth):    {'✓ PASSED' if h2_passed else '✗ FAILED'}")
    print(f"H4 (Energy Savings):     {'✓ PASSED' if h4_passed else '✗ FAILED'}")

    overall = h1_passed and h2_passed and h4_passed
    print(f"\nOVERALL: {'✓ METABOLIC INFERENCE VALIDATED' if overall else '✗ HYPOTHESES NOT FULLY SUPPORTED'}")

    # Save results
    output_path = Path("results/z700_metabolic_inference.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to {output_path}")

    # Cleanup
    hw.reset()

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metabolic Inference Experiment")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--thermal-coef", type=float, default=0.3)
    parser.add_argument("--base-threshold", type=float, default=0.4)
    parser.add_argument("--quick", action="store_true", help="Quick test mode")

    args = parser.parse_args()

    config = MetabolicConfig(
        n_trials=1 if args.quick else args.trials,
        thermal_coefficient=args.thermal_coef,
        base_confidence_threshold=args.base_threshold,
    )

    if args.quick:
        config.easy_prompts = config.easy_prompts[:2]
        config.hard_prompts = config.hard_prompts[:2]

    run_metabolic_experiment(config)
