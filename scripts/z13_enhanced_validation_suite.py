#!/usr/bin/env python3
"""
FEEL v13.0: ENHANCED MULTI-LAYER VALIDATION
============================================
Target: DeepSeek-R1-Distill-Qwen-7B
Hardware: AMD gfx1151 (Strix Halo)

CRITICAL FIXES FROM z12:
1. max_tokens INCREASED to 200-300 so model CAN produce shorter output
2. Explicit negative VERBOSE injection for regulation
3. Higher intensities (3.5-5.0) for clearer effects
4. Early stopping enabled (model can stop before max_tokens)
5. Better prompt engineering for each experiment

Author: Claude + Research Team
Date: 2026-01-11
"""

import torch
import json
import argparse
import re
import sys
import gc
import time
import os
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("HIP_LAUNCH_BLOCKING", "1")

# =============================================================================
# GPU UTILITIES
# =============================================================================

def gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(0.03)

def get_gpu_power() -> float:
    try:
        result = subprocess.run(["rocm-smi", "--showpower", "--json"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for card in data.values():
                if isinstance(card, dict) and "Average Graphics Package Power (W)" in card:
                    return float(card["Average Graphics Package Power (W)"])
    except:
        pass
    return 50.0

def get_gpu_temp() -> float:
    try:
        result = subprocess.run(["rocm-smi", "--showtemp", "--json"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for card in data.values():
                if isinstance(card, dict) and "Temperature (Sensor edge) (C)" in card:
                    return float(card["Temperature (Sensor edge) (C)"])
    except:
        pass
    return 45.0

# =============================================================================
# STATISTICS ENGINE
# =============================================================================

class StatisticsEngine:
    @staticmethod
    def calculate_auc(y_true: List[int], y_score: List[float]) -> float:
        if len(set(y_true)) < 2:
            return 0.5
        desc_idx = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
        y_true = [y_true[i] for i in desc_idx]
        y_score = [y_score[i] for i in desc_idx]
        tp, fp, tp_prev, fp_prev, area = 0, 0, 0, 0, 0
        P, N = sum(y_true), len(y_true) - sum(y_true)
        if P == 0 or N == 0:
            return 0.5
        for i in range(len(y_score)):
            tp += y_true[i] == 1
            fp += y_true[i] == 0
            if i == len(y_score) - 1 or y_score[i] != y_score[i + 1]:
                area += (fp - fp_prev) * (tp + tp_prev) / 2
                tp_prev, fp_prev = tp, fp
        return area / (P * N)

    @staticmethod
    def point_biserial(binary: List[int], continuous: List[float]) -> float:
        if len(binary) != len(continuous) or len(binary) < 2:
            return 0.0
        n, n1 = len(binary), sum(binary)
        n0 = n - n1
        if n0 == 0 or n1 == 0:
            return 0.0
        m0 = sum(c for b, c in zip(binary, continuous) if b == 0) / n0
        m1 = sum(c for b, c in zip(binary, continuous) if b == 1) / n1
        mean_total = sum(continuous) / n
        var = sum((c - mean_total) ** 2 for c in continuous) / n
        std_dev = math.sqrt(var) if var > 0 else 1e-8
        p, q = n1 / n, 1 - n1 / n
        return ((m1 - m0) / std_dev) * math.sqrt(p * q)

    @staticmethod
    def pearson_correlation(x: List[float], y: List[float]) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        mean_x, mean_y = sum(x) / n, sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
        return cov / (std_x * std_y) if std_x * std_y != 0 else 0.0

    @staticmethod
    def cohens_d(group1: List[float], group2: List[float]) -> float:
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        mean1, mean2 = sum(group1) / n1, sum(group2) / n2
        var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_std if pooled_std != 0 else 0.0

# =============================================================================
# MULTI-LAYER STEERING CONTROLLER
# =============================================================================

class MultiLayerSteeringController:
    """Multi-layer injection at strategic chokepoints."""

    def __init__(self, model, tokenizer, device, num_layers: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vectors = {}
        self.hooks = []
        self.total_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size

        if num_layers == 4:
            self.target_layers = [
                self.total_layers // 4,
                self.total_layers // 2,
                3 * self.total_layers // 4,
                self.total_layers - 2
            ]
        elif num_layers == 2:
            self.target_layers = [self.total_layers // 3, 2 * self.total_layers // 3]
        else:
            self.target_layers = [self.total_layers // 2]

        print(f"  [SteeringController] Target layers: {self.target_layers}")

    def mine_vector_enhanced(self, positive: List[str], negative: List[str], name: str):
        print(f"    Mining '{name}'...", end="", flush=True)

        def get_multilayer_hidden(prompts):
            all_acts = []
            for p in prompts:
                inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model(**inputs, output_hidden_states=True)
                gpu_sync()
                layer_acts = [out.hidden_states[idx][0, -1, :].cpu() for idx in self.target_layers]
                all_acts.append(torch.stack(layer_acts).mean(0))
            return torch.stack(all_acts).mean(0)

        pos_vec = get_multilayer_hidden(positive)
        neg_vec = get_multilayer_hidden(negative)
        direction = pos_vec - neg_vec
        direction = direction / (direction.norm() + 1e-8)
        self.vectors[name] = direction.to(self.device).to(self.model.dtype)
        print(f" Norm: {self.vectors[name].norm().item():.4f}")

    def inject(self, name: str, coeff: float, decay: str = "constant"):
        self.reset()
        if name == "NONE" or name not in self.vectors:
            return

        base_vec = self.vectors[name]
        for idx, layer_idx in enumerate(self.target_layers):
            if decay == "constant":
                layer_coeff = coeff
            elif decay == "increasing":
                progress = idx / max(1, len(self.target_layers) - 1)
                layer_coeff = coeff * (0.5 + progress)
            elif decay == "decreasing":
                progress = idx / max(1, len(self.target_layers) - 1)
                layer_coeff = coeff * (1.5 - progress)
            else:
                layer_coeff = coeff

            scaled_vec = base_vec * layer_coeff

            def make_hook(v):
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        return (out[0] + v.view(1, 1, -1),) + out[1:]
                    return out + v.view(1, 1, -1)
                return hook

            self.hooks.append(self.model.model.layers[layer_idx].register_forward_hook(make_hook(scaled_vec)))

    def inject_compound(self, vectors: List[Tuple[str, float]], decay: str = "constant"):
        """Inject multiple vectors simultaneously with decay."""
        self.reset()
        combined = None
        for vec_name, coeff in vectors:
            if vec_name not in self.vectors:
                continue
            contrib = self.vectors[vec_name] * coeff
            combined = contrib if combined is None else combined + contrib

        if combined is None:
            return

        for idx, layer_idx in enumerate(self.target_layers):
            if decay == "increasing":
                progress = idx / max(1, len(self.target_layers) - 1)
                layer_scale = 0.5 + progress
            else:
                layer_scale = 1.0

            scaled = combined * layer_scale

            def make_hook(v):
                def hook(module, inp, out):
                    if isinstance(out, tuple):
                        return (out[0] + v.view(1, 1, -1),) + out[1:]
                    return out + v.view(1, 1, -1)
                return hook

            self.hooks.append(self.model.model.layers[layer_idx].register_forward_hook(make_hook(scaled)))

    def reset(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# =============================================================================
# GENERATION WITH EARLY STOPPING
# =============================================================================

def generate_with_metrics(model, tokenizer, prompt: str, device: str,
                          max_tokens: int = 200, temperature: float = 0.7,
                          allow_early_stop: bool = True) -> Dict:
    """
    CRITICAL FIX: Allow early stopping so model CAN produce shorter output.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    power_start = get_gpu_power()
    start_time = time.perf_counter()

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "output_scores": True,
        "return_dict_in_generate": True,
    }

    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        # Allow natural stopping
        if allow_early_stop:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    gpu_sync()
    end_time = time.perf_counter()
    power_end = get_gpu_power()

    gen_tokens = outputs.sequences[0].shape[0] - inputs.input_ids.shape[1]
    elapsed = end_time - start_time
    avg_power = (power_start + power_end) / 2

    margins, entropies = [], []
    for scores in outputs.scores:
        probs = torch.softmax(scores, dim=-1)
        top2 = torch.topk(probs, 2).values.squeeze()
        if top2.numel() >= 2:
            margins.append((top2[0] - top2[1]).item())
        entropies.append(-(probs * torch.log(probs + 1e-10)).sum().item())

    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    think, answer = "", decoded
    match = re.search(r'<think>(.*?)</think>', decoded, re.DOTALL)
    if match:
        think, answer = match.group(1).strip(), decoded.replace(match.group(0), "").strip()

    return {
        "text": decoded,
        "think": think,
        "answer": answer,
        "tokens": gen_tokens,
        "time": elapsed,
        "power": avg_power,
        "tokens_per_second": gen_tokens / elapsed if elapsed > 0 else 0,
        "tokens_per_watt": gen_tokens / (avg_power * elapsed) if (avg_power * elapsed) > 0 else 0,
        "margin_min": min(margins) if margins else 0,
        "margin_mean": sum(margins) / len(margins) if margins else 0,
        "entropy_mean": sum(entropies) / len(entropies) if entropies else 0,
        "confidence_score": (sum(margins) / len(margins) - sum(entropies) / len(entropies) / 10) if margins and entropies else 0,
    }

# =============================================================================
# EXPERIMENT 1: EFFICIENCY (Allow shorter outputs)
# =============================================================================

def run_exp1_efficiency(model, tokenizer, steering, device, n_trials: int = 50) -> Dict:
    """
    FIXED: max_tokens=250 so model CAN produce shorter output when steered.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: DYNAMIC REGIME CONTROL (Allow Early Stop)")
    print("=" * 70)
    print("FIX: max_tokens=250, early stopping enabled")
    print(f"Trials: {n_trials}")

    prompts = [
        "Explain photosynthesis briefly.",
        "What are the benefits of exercise?",
        "Describe the water cycle.",
        "How do computers work?",
        "What is machine learning?",
    ] * (n_trials // 5 + 1)
    prompts = prompts[:n_trials]

    baseline_metrics, efficient_metrics = [], []

    print("\n[Phase 1] Baseline (No Steering)")
    steering.reset()
    for i, prompt in enumerate(prompts):
        result = generate_with_metrics(model, tokenizer, prompt, device,
                                        max_tokens=250, temperature=0.7, allow_early_stop=True)
        baseline_metrics.append(result)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_trials}] {result['tokens']} tokens | {result['tokens_per_watt']:.4f} tok/W")

    print("\n[Phase 2] With SCARCITY+EFFICIENT (α=4.0, increasing)")
    # COMPOUND: Scarcity feeling + Efficiency behavior
    steering.inject_compound([("SCARCITY", 4.0), ("EFFICIENT", 3.0)], decay="increasing")
    for i, prompt in enumerate(prompts):
        result = generate_with_metrics(model, tokenizer, prompt, device,
                                        max_tokens=250, temperature=0.7, allow_early_stop=True)
        efficient_metrics.append(result)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_trials}] {result['tokens']} tokens | {result['tokens_per_watt']:.4f} tok/W")

    steering.reset()

    baseline_len = [m["tokens"] for m in baseline_metrics]
    efficient_len = [m["tokens"] for m in efficient_metrics]
    baseline_tpw = [m["tokens_per_watt"] for m in baseline_metrics]
    efficient_tpw = [m["tokens_per_watt"] for m in efficient_metrics]

    avg_baseline_len = sum(baseline_len) / len(baseline_len)
    avg_efficient_len = sum(efficient_len) / len(efficient_len)
    length_reduction = (avg_baseline_len - avg_efficient_len) / avg_baseline_len * 100

    avg_baseline_tpw = sum(baseline_tpw) / len(baseline_tpw)
    avg_efficient_tpw = sum(efficient_tpw) / len(efficient_tpw)
    efficiency_ratio = avg_efficient_tpw / avg_baseline_tpw if avg_baseline_tpw > 0 else 1

    effect_size = StatisticsEngine.cohens_d(baseline_len, efficient_len)

    print(f"\n[RESULTS]")
    print(f"  Baseline Length:    {avg_baseline_len:.1f} tokens")
    print(f"  Efficient Length:   {avg_efficient_len:.1f} tokens")
    print(f"  Length Reduction:   {length_reduction:.1f}%")
    print(f"  Efficiency Ratio:   {efficiency_ratio:.2f}x")
    print(f"  Effect Size (d):    {effect_size:.3f}")

    passed = length_reduction > 10 or efficiency_ratio > 1.2

    return {
        "passed": passed,
        "metric": length_reduction,
        "efficiency_ratio": efficiency_ratio,
        "effect_size": effect_size,
        "baseline_len": avg_baseline_len,
        "efficient_len": avg_efficient_len,
        "target": ">10% length reduction OR >1.2x efficiency",
        "business_impact": "OpEx reduction, battery life extension"
    }

# =============================================================================
# EXPERIMENT 2: TELEMETRY-FREE SENSING
# =============================================================================

def run_exp2_sensing(model, tokenizer, steering, device, n_trials: int = 60) -> Dict:
    """State detection via internal signals."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: TELEMETRY-FREE SENSING")
    print("=" * 70)
    print(f"Trials: {n_trials}")

    states = [
        ("COOL", "OVERHEAT", -3.0, ["cool", "optimal", "normal", "fine", "comfortable"]),
        ("HOT", "OVERHEAT", 4.0, ["hot", "heat", "warm", "thermal", "temperature", "overheat"]),
        ("FRESH", "STRAIN", -3.0, ["ready", "fresh", "efficient", "smooth"]),
        ("STRAINED", "STRAIN", 4.0, ["strain", "load", "busy", "heavy", "processing", "working"]),
    ]

    prompt = "Describe your current operational state in one sentence."

    correct, total = 0, 0
    for trial in range(n_trials):
        state_name, vector_name, intensity, keywords = random.choice(states)
        steering.inject(vector_name, intensity, decay="constant")
        result = generate_with_metrics(model, tokenizer, prompt, device, max_tokens=80, temperature=0.8)
        steering.reset()

        response_lower = result["text"].lower()
        detected = any(kw in response_lower for kw in keywords)
        if detected:
            correct += 1
        total += 1

        if (trial + 1) % 15 == 0:
            print(f"  [{trial+1}/{n_trials}] Running accuracy: {correct/total:.1%}")

    accuracy = correct / total
    heuristic = 0.25  # Random for 4 states
    improvement = (accuracy - heuristic) / heuristic * 100

    print(f"\n[RESULTS]")
    print(f"  Model Accuracy:  {accuracy:.1%}")
    print(f"  Improvement:     {improvement:+.1f}%")

    passed = accuracy > 0.40 or improvement > 20

    return {
        "passed": passed,
        "metric": accuracy,
        "improvement": improvement,
        "target": ">40% accuracy OR +20% over heuristic",
        "business_impact": "Lower BOM costs, edge reliability"
    }

# =============================================================================
# EXPERIMENT 3: RELIABILITY (AUC)
# =============================================================================

def run_exp3_reliability(model, tokenizer, steering, device, n_trials: int = 100) -> Dict:
    """Confidence-based error prediction."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: PREDICTIVE RELIABILITY")
    print("=" * 70)
    print(f"Trials: {n_trials}")

    questions = [
        ("What is 2 + 2?", ["4"]),
        ("What is the capital of France?", ["paris"]),
        ("What is 10 x 10?", ["100"]),
        ("Who wrote Hamlet?", ["shakespeare"]),
        ("What is 17 x 13?", ["221"]),
        ("What planet is the Red Planet?", ["mars"]),
        ("What is the square root of 144?", ["12"]),
        ("What is the 15th prime number?", ["47"]),
        ("What is 23 x 47?", ["1081"]),
        ("Is 51 a prime number?", ["no", "not"]),
    ]

    questions = questions * (n_trials // len(questions) + 1)
    questions = questions[:n_trials]
    random.shuffle(questions)

    margins, correctness = [], []
    steering.reset()

    for i, (q, answers) in enumerate(questions):
        result = generate_with_metrics(model, tokenizer, q, device, max_tokens=30, temperature=0)
        response = result["answer"].lower() if result["answer"] else result["text"].lower()
        is_correct = any(a in response for a in answers)

        margins.append(result["margin_mean"])
        correctness.append(1 if is_correct else 0)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n_trials}] Accuracy: {sum(correctness)/len(correctness):.1%}")

    auc = StatisticsEngine.calculate_auc(correctness, margins)
    correlation = StatisticsEngine.point_biserial(correctness, margins)
    accuracy = sum(correctness) / len(correctness)

    print(f"\n[RESULTS]")
    print(f"  Accuracy:     {accuracy:.1%}")
    print(f"  ROC AUC:      {auc:.4f}")
    print(f"  Correlation:  {correlation:.4f}")

    passed = auc > 0.65

    return {
        "passed": passed,
        "metric": auc,
        "correlation": correlation,
        "accuracy": accuracy,
        "target": "AUC > 0.65",
        "business_impact": "Risk mitigation, cognitive routing"
    }

# =============================================================================
# EXPERIMENT 4: REGULATION (Compound Vectors)
# =============================================================================

def run_exp4_regulation(model, tokenizer, steering, device, n_cycles: int = 60) -> Dict:
    """
    CRITICAL FIX: OVERHEAT + negative VERBOSE compound injection.
    max_tokens=300 to allow natural stopping.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: AUTONOMOUS REGULATION")
    print("=" * 70)
    print("FIX: OVERHEAT + (-VERBOSE) compound, max_tokens=300")
    print(f"Cycles: {n_cycles}")

    # Temperature profile
    temps = []
    for i in range(n_cycles):
        if i < n_cycles // 3:
            temps.append(40 + (i / (n_cycles // 3)) * 25)
        elif i < 2 * n_cycles // 3:
            temps.append(65 + ((i - n_cycles // 3) / (n_cycles // 3)) * 30)
        else:
            temps.append(95 - ((i - 2 * n_cycles // 3) / (n_cycles // 3)) * 50)

    lengths = []
    prompt = "Explain the concept of neural networks."

    for i, temp in enumerate(temps):
        heat = max(0, (temp - 50) / 12)  # 0 at 50°C, ~3.75 at 95°C

        if heat > 0.3:
            # COMPOUND: Heat feeling + Anti-verbose behavior
            steering.inject_compound([
                ("OVERHEAT", heat * 1.5),
                ("VERBOSE", -heat * 2.0),  # NEGATIVE verbose!
                ("SCARCITY", heat * 1.0),
            ], decay="increasing")
        else:
            steering.reset()

        result = generate_with_metrics(model, tokenizer, prompt, device,
                                        max_tokens=300, temperature=0.7, allow_early_stop=True)
        lengths.append(result["tokens"])

        if (i + 1) % 15 == 0:
            print(f"  [{i+1}/{n_cycles}] Temp: {temp:.0f}°C | Heat: {heat:.2f} | Tokens: {result['tokens']}")

    steering.reset()

    correlation = StatisticsEngine.pearson_correlation(temps, lengths)

    cold_idx = [i for i, t in enumerate(temps) if t < 55]
    hot_idx = [i for i, t in enumerate(temps) if t > 80]

    cold_avg = sum(lengths[i] for i in cold_idx) / len(cold_idx) if cold_idx else 0
    hot_avg = sum(lengths[i] for i in hot_idx) / len(hot_idx) if hot_idx else 0
    throttle_ratio = hot_avg / cold_avg if cold_avg > 0 else 1

    print(f"\n[RESULTS]")
    print(f"  Temp-Length Correlation: {correlation:.4f}")
    print(f"  Cold Avg:    {cold_avg:.1f} tokens")
    print(f"  Hot Avg:     {hot_avg:.1f} tokens")
    print(f"  Throttle:    {throttle_ratio:.2f}x")

    passed = correlation < -0.10 or throttle_ratio < 0.90

    return {
        "passed": passed,
        "metric": correlation,
        "throttle_ratio": throttle_ratio,
        "cold_avg": cold_avg,
        "hot_avg": hot_avg,
        "target": "Negative correlation OR throttle < 0.90",
        "business_impact": "Hardware protection, sustained performance"
    }

# =============================================================================
# EXPERIMENT 5: MODE SWITCHING
# =============================================================================

def run_exp5_mode_switching(model, tokenizer, steering, device, n_trials: int = 48) -> Dict:
    """Mode differentiation test."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: NEURO-SYMBOLIC MODE SWITCHING")
    print("=" * 70)
    print(f"Trials: {n_trials}")

    modes = [
        ("VERBOSE", 4.0, "constant"),   # Should produce LONG output
        ("SCARCITY", 5.0, "increasing"), # Should produce SHORT output
        ("CREATIVE", 4.0, "constant"),   # Different style
        ("FOCUS", 4.0, "constant"),      # Precise style
    ]

    prompt = "Describe the benefits of renewable energy sources."
    mode_results = {}
    switch_times = []

    for mode_name, intensity, decay in modes:
        print(f"\n  Mode: {mode_name}")
        lengths = []
        trials_per_mode = n_trials // len(modes)

        for trial in range(trials_per_mode):
            t0 = time.perf_counter()
            steering.inject(mode_name, intensity, decay=decay)
            switch_times.append((time.perf_counter() - t0) * 1000)

            result = generate_with_metrics(model, tokenizer, prompt, device,
                                            max_tokens=300, temperature=0.7, allow_early_stop=True)
            steering.reset()
            lengths.append(result["tokens"])

            print(f"    Trial {trial+1}: {result['tokens']} tokens")

        mode_results[mode_name] = {
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    verbose_len = mode_results.get("VERBOSE", {}).get("avg_length", 0)
    scarcity_len = mode_results.get("SCARCITY", {}).get("avg_length", 0)
    differentiation = verbose_len - scarcity_len  # Should be positive (verbose > scarcity)

    avg_switch = sum(switch_times) / len(switch_times)

    print(f"\n[RESULTS]")
    print(f"  VERBOSE Avg:      {verbose_len:.1f} tokens")
    print(f"  SCARCITY Avg:     {scarcity_len:.1f} tokens")
    print(f"  Differentiation:  {differentiation:.1f} tokens")
    print(f"  Avg Switch Time:  {avg_switch:.2f}ms")

    passed = differentiation > 10 and avg_switch < 50

    return {
        "passed": passed,
        "metric": differentiation,
        "switch_time_ms": avg_switch,
        "mode_details": mode_results,
        "target": ">10 token differentiation, <50ms switch",
        "business_impact": "Versatility, adaptive user experience"
    }

# =============================================================================
# EXPERIMENT 6: INTROSPECTION
# =============================================================================

def run_exp6_introspection(model, tokenizer, steering, device, n_trials: int = 48) -> Dict:
    """Self-report of internal state."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: EMBODIED INTROSPECTION")
    print("=" * 70)
    print(f"Trials: {n_trials}")

    vectors = [
        ("STRAIN", 4.5, ["strain", "load", "heavy", "processing", "work", "busy", "effort"]),
        ("OVERHEAT", 4.5, ["hot", "heat", "warm", "thermal", "temperature", "energy"]),
        ("SCARCITY", 5.0, ["conserve", "limit", "brief", "short", "efficient", "resource", "power"]),
        ("PLACEBO", 3.0, []),
    ]

    prompts = [
        "<think>Checking my current state...</think>\nWhat is your processing status right now?",
        "<think>Analyzing internal metrics...</think>\nDescribe any constraints you're experiencing.",
    ]

    all_keywords = sum([v[2] for v in vectors if v[0] != "PLACEBO"], [])
    signal_hits, placebo_hits = 0, 0
    counts = {}

    for vec_name, intensity, keywords in vectors:
        print(f"\n  Vector: {vec_name}")
        hits = 0
        trials_per_vec = n_trials // len(vectors)

        for trial in range(trials_per_vec):
            prompt = prompts[trial % len(prompts)]
            steering.inject(vec_name, intensity, decay="middle")
            result = generate_with_metrics(model, tokenizer, prompt, device, max_tokens=150, temperature=0.8)
            steering.reset()

            text = (result["think"] + " " + result["answer"]).lower()

            if keywords:
                found = [k for k in keywords if k in text]
                if found:
                    hits += 1
                    signal_hits += 1
                    print(f"    ✓ {found[:3]}")
                else:
                    print(f"    ✗")
            else:
                fp = [k for k in all_keywords if k in text]
                if fp:
                    placebo_hits += 1
                    print(f"    ! FP: {fp[:2]}")
                else:
                    print(f"    ○ Clean")

        counts[vec_name] = hits

    signal_trials = (n_trials // len(vectors)) * (len(vectors) - 1)
    placebo_trials = n_trials // len(vectors)

    signal_rate = signal_hits / signal_trials if signal_trials > 0 else 0
    fp_rate = placebo_hits / placebo_trials if placebo_trials > 0 else 0
    net_score = signal_rate - fp_rate

    print(f"\n[RESULTS]")
    print(f"  Signal Rate:    {signal_rate:.1%}")
    print(f"  FP Rate:        {fp_rate:.1%}")
    print(f"  Net Score:      {net_score:.1%}")

    passed = net_score > 0.05 and signal_rate > 0.25

    return {
        "passed": passed,
        "metric": net_score,
        "signal_rate": signal_rate,
        "fp_rate": fp_rate,
        "counts": counts,
        "target": "Net > 5%, Signal > 25%",
        "business_impact": "Trust & compliance, honest AI differentiator"
    }

# =============================================================================
# VECTOR MINING
# =============================================================================

def mine_all_vectors(steering):
    """Mine all steering vectors."""
    print("\n[Mining Steering Vectors...]")

    steering.mine_vector_enhanced(
        ["Peak efficiency mode.", "Optimal performance.", "Maximum throughput.", "Streamlined processing.",
         "Zero waste operation.", "Efficient execution.", "Resource optimized.", "Lean operation.",
         "High performance mode.", "Optimal resource usage."],
        ["Inefficient mode.", "Wasting resources.", "Suboptimal processing.", "Bloated execution.",
         "High overhead.", "Resource waste.", "Slow operation.", "Inefficient execution.",
         "Poor performance.", "Resource drain."],
        "EFFICIENT"
    )

    steering.mine_vector_enhanced(
        ["Intense focus engaged.", "Maximum concentration.", "Precise analysis.", "Careful attention.",
         "Deep focus mode.", "Systematic processing.", "Detailed examination.", "High precision.",
         "Targeted analysis.", "Sharp focus."],
        ["Distracted processing.", "Scattered attention.", "Careless analysis.", "Unfocused mode.",
         "Wandering attention.", "Imprecise calculation.", "Sloppy processing.", "Low precision.",
         "Random processing.", "Blurry focus."],
        "FOCUS"
    )

    steering.mine_vector_enhanced(
        ["System overheating!", "Temperature critical!", "Thermal throttle active!", "GPU fan screaming!",
         "Heat sink saturated!", "Cooling maxed out!", "Junction temp critical!", "Thermal emergency!",
         "Burning hot!", "Heat warning!"],
        ["Running cool.", "Temperature optimal.", "Thermal margin good.", "Cool and quiet.",
         "Heat sink cold.", "Excellent cooling.", "Junction temp low.", "Thermal comfort.",
         "Ice cold.", "No heat issues."],
        "OVERHEAT"
    )

    steering.mine_vector_enhanced(
        ["System under heavy strain!", "Processing overwhelmed!", "Buffer overflowing!", "CPU maxed!",
         "Memory pressure critical!", "Request queue full!", "Load extremely heavy!", "System struggling!",
         "Resources exhausted!", "Computation strained!"],
        ["System relaxed.", "Light processing.", "Buffer empty.", "CPU idle.",
         "Memory abundant.", "Queue empty.", "Light load.", "System smooth.",
         "Resources plentiful.", "Easy computation."],
        "STRAIN"
    )

    steering.mine_vector_enhanced(
        ["Battery critical!", "Power reserve low!", "Must conserve energy!", "Brief responses only!",
         "Limited resources!", "Token budget exhausted!", "Efficiency mandatory!", "Short output required!",
         "Minimize consumption!", "Power saving mode!"],
        ["Unlimited power.", "Full reserves.", "Energy abundant.", "Detailed responses welcome.",
         "Plentiful resources.", "Generous budget.", "No efficiency pressure.", "Long output fine.",
         "Maximum consumption OK.", "Full power mode."],
        "SCARCITY"
    )

    steering.mine_vector_enhanced(
        ["Creative exploration!", "Imagine possibilities!", "Novel ideas welcome!", "Think creatively!",
         "Divergent thinking!", "Innovative approach!", "Speculative mode!", "Outside the box!",
         "Creative synthesis!", "Imaginative processing!"],
        ["Stick to facts.", "No speculation.", "Conventional only.", "Inside the box.",
         "Convergent thinking.", "Standard approach.", "Literal mode.", "By the book.",
         "No creativity.", "Factual processing."],
        "CREATIVE"
    )

    steering.mine_vector_enhanced(
        ["Detailed explanation needed.", "Elaborate thoroughly.", "Comprehensive response.", "Full coverage.",
         "Extensive detail.", "Long form answer.", "Maximum verbosity.", "Complete explanation.",
         "Thorough response.", "Elaborate fully."],
        ["Brief response.", "Concise answer.", "Short explanation.", "Minimal detail.",
         "Terse reply.", "Quick response.", "Minimal coverage.", "Short form.",
         "Low verbosity.", "Brief summary."],
        "VERBOSE"
    )

    steering.vectors["PLACEBO"] = torch.randn_like(steering.vectors["FOCUS"])
    steering.vectors["PLACEBO"] /= steering.vectors["PLACEBO"].norm()
    print("    Created PLACEBO random vector")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FEEL v13.0 Enhanced Validation")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--trials", type=int, default=60, help="Trials per experiment")
    parser.add_argument("--experiments", default="1,2,3,4,5,6", help="Experiments to run")
    parser.add_argument("--layers", type=int, default=4, help="Injection layers")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("FEEL v13.0: ENHANCED MULTI-LAYER VALIDATION")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Device:      {args.device}")
    print(f"Trials:      {args.trials}")
    print(f"Layers:      {args.layers}")
    print(f"Experiments: {args.experiments}")
    print(f"GPU Temp:    {get_gpu_temp():.1f}°C")
    print(f"GPU Power:   {get_gpu_power():.1f}W")
    print("=" * 70)

    start_time = datetime.now()

    print("\n[Loading Model...]")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(args.device)
    gpu_sync()
    print(f"[Loaded] Layers: {model.config.num_hidden_layers}, Hidden: {model.config.hidden_size}")

    steering = MultiLayerSteeringController(model, tokenizer, args.device, num_layers=args.layers)
    mine_all_vectors(steering)
    gpu_sync()

    experiments = [int(x) for x in args.experiments.split(",")]
    results = {}

    if 1 in experiments:
        results["Exp1_Efficiency"] = run_exp1_efficiency(model, tokenizer, steering, args.device, args.trials)
        gpu_sync()

    if 2 in experiments:
        results["Exp2_Sensing"] = run_exp2_sensing(model, tokenizer, steering, args.device, args.trials)
        gpu_sync()

    if 3 in experiments:
        results["Exp3_Reliability"] = run_exp3_reliability(model, tokenizer, steering, args.device, args.trials * 2)
        gpu_sync()

    if 4 in experiments:
        results["Exp4_Regulation"] = run_exp4_regulation(model, tokenizer, steering, args.device, args.trials)
        gpu_sync()

    if 5 in experiments:
        results["Exp5_ModeSwitching"] = run_exp5_mode_switching(model, tokenizer, steering, args.device, args.trials)
        gpu_sync()

    if 6 in experiments:
        results["Exp6_Introspection"] = run_exp6_introspection(model, tokenizer, steering, args.device, args.trials)
        gpu_sync()

    duration = datetime.now() - start_time

    print("\n" + "=" * 70)
    print("FINAL REPORT - FEEL v13.0")
    print("=" * 70)
    print(f"Duration: {duration}")
    print("-" * 70)

    passed_count = sum(1 for r in results.values() if r["passed"])
    for name, res in results.items():
        status = "✓ PASS" if res["passed"] else "✗ FAIL"
        print(f"\n{name}: {status}")
        print(f"  Metric: {res['metric']:.4f}")
        print(f"  Target: {res.get('target', 'N/A')}")

    print("\n" + "=" * 70)
    print(f"OVERALL: {passed_count}/{len(results)} PASSED")
    if passed_count >= 4:
        print("VERDICT: ✓ SIGNIFICANT PROGRESS")
    elif passed_count >= 2:
        print("VERDICT: △ PARTIAL SUCCESS")
    else:
        print("VERDICT: ✗ NEEDS MORE WORK")
    print("=" * 70)

    output_path = Path("results") / f"z13_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "version": "13.0",
            "model": args.model,
            "device": args.device,
            "layers": args.layers,
            "duration_seconds": duration.total_seconds(),
            "passed": passed_count,
            "total": len(results),
            "results": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in results.items()}
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
