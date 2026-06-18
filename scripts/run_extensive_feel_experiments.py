#!/usr/bin/env python3
"""
Extensive FEEL Token Integration Experiments

Runs all experiments with:
- More diverse prompts (20+)
- Longer generation (64-128 tokens)
- Multiple runs for statistical significance
- Comprehensive data collection

Usage:
    python scripts/run_extensive_feel_experiments.py
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.feel_integration_experiments import (
    FEELIntegratedModel,
    FEELExperiments,
    compute_kl_divergence,
    compute_delta_logit,
    LocalAttentionMetric,
    ExperimentResult,
)


# =============================================================================
# EXTENSIVE PROMPT SETS
# =============================================================================

EXTENSIVE_PROMPTS = {
    'factual': [
        "What is 2 + 2?",
        "What color is the sky?",
        "How many days are in a week?",
        "What is the capital of France?",
        "How many legs does a spider have?",
    ],
    'reasoning': [
        "Explain step by step how to solve 15 * 23.",
        "What is 17 * 23? Think step by step.",
        "If a train travels at 60 mph for 2 hours, how far does it go?",
        "What comes next in the sequence: 2, 4, 8, 16, ?",
        "Explain the logic behind the Pythagorean theorem.",
    ],
    'creative': [
        "Write a short poem about the ocean.",
        "Write a haiku about artificial intelligence.",
        "Imagine a conversation between a cat and a dog.",
        "Describe a sunset in three sentences.",
        "Create a metaphor for time passing.",
    ],
    'introspective': [
        "How confident are you in your answer?",
        "What are you uncertain about right now?",
        "Describe your reasoning process.",
        "Explain what you're thinking right now.",
        "How do you know when you don't know something?",
    ],
    'technical': [
        "Write a Python function to reverse a string.",
        "Explain how neural networks learn.",
        "What is the difference between TCP and UDP?",
        "Describe the concept of recursion in programming.",
        "Explain what a hash table is and how it works.",
    ],
    'ambiguous': [
        "Is it better to be happy or successful?",
        "What is the meaning of life?",
        "Should AI have rights?",
        "What makes something beautiful?",
        "Is free will an illusion?",
    ],
    'quantum': [
        "Explain quantum computing in simple terms.",
        "What is quantum entanglement?",
        "Describe Schrödinger's cat thought experiment.",
        "What is superposition in quantum mechanics?",
    ],
    'uncertainty': [
        "Describe the feeling of uncertainty.",
        "What does it feel like to not know the answer?",
        "How do you handle ambiguous situations?",
        "What is epistemic humility?",
    ],
}


class ExtensiveFEELExperiments(FEELExperiments):
    """Extended experiments with more comprehensive data collection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extensive_results = {}

    def experiment_causal_influence_extensive(
        self,
        max_tokens: int = 64,
        n_runs: int = 1,
    ) -> ExperimentResult:
        """
        Extensive causal influence measurement across all prompt categories.
        """
        print("\n" + "="*70)
        print("EXTENSIVE EXPERIMENT: Causal Influence (KL/Δlogit)")
        print("="*70)

        all_prompts = []
        for category, prompts in EXTENSIVE_PROMPTS.items():
            for p in prompts:
                all_prompts.append((category, p))

        print(f"Testing {len(all_prompts)} prompts x {n_runs} runs = {len(all_prompts)*n_runs} total")

        results_by_category = {cat: [] for cat in EXTENSIVE_PROMPTS.keys()}
        all_results = []

        model = FEELIntegratedModel(
            self.base_model, self.tokenizer,
            injection_strength=1.0,
            device=self.device,
        )

        for run_idx in range(n_runs):
            print(f"\n--- Run {run_idx+1}/{n_runs} ---")

            for i, (category, prompt) in enumerate(all_prompts):
                model.reset()

                messages = [{"role": "user", "content": prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                input_length = input_ids.shape[1]

                generated_ids = input_ids.clone()
                past_key_values_on = None
                past_key_values_off = None

                kl_divs = []
                delta_logits = []

                for step in range(max_tokens):
                    with torch.no_grad():
                        if step == 0:
                            outputs_on, _ = model.forward_with_feel(
                                input_ids=generated_ids,
                                attention_mask=attention_mask,
                            )
                            outputs_off = model.forward_without_feel(
                                input_ids=generated_ids,
                                attention_mask=attention_mask,
                            )
                        else:
                            outputs_on, _ = model.forward_with_feel(
                                input_ids=generated_ids[:, -1:],
                                attention_mask=attention_mask,
                                past_key_values=past_key_values_on,
                            )
                            outputs_off = model.forward_without_feel(
                                input_ids=generated_ids[:, -1:],
                                attention_mask=attention_mask,
                                past_key_values=past_key_values_off,
                            )

                        logits_on = outputs_on.logits[:, -1, :]
                        logits_off = outputs_off.logits[:, -1, :]

                        kl = compute_kl_divergence(logits_on, logits_off)
                        next_token = logits_on.argmax(dim=-1, keepdim=True)
                        delta = compute_delta_logit(logits_on, logits_off, next_token.item())

                        kl_divs.append(kl)
                        delta_logits.append(delta)

                        if step > 0 and model.injection_strength > 0:
                            past_key_values_on = self._strip_feel_from_cache(
                                outputs_on.past_key_values, n_feel_tokens=1
                            )
                        else:
                            past_key_values_on = outputs_on.past_key_values
                        past_key_values_off = outputs_off.past_key_values

                        sensors = model.extract_signals(outputs_on.logits, input_length, next_token.item())
                        model.update_z_feel(sensors)

                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                        ], dim=1)

                        if next_token.item() == self.tokenizer.eos_token_id:
                            break

                result = {
                    'run': run_idx,
                    'category': category,
                    'prompt': prompt[:50],
                    'avg_kl': float(np.mean(kl_divs)),
                    'max_kl': float(np.max(kl_divs)),
                    'std_kl': float(np.std(kl_divs)),
                    'avg_delta': float(np.mean(delta_logits)),
                    'n_tokens': len(kl_divs),
                    'kl_trajectory': [float(k) for k in kl_divs],
                }

                results_by_category[category].append(result)
                all_results.append(result)

                if (i + 1) % 5 == 0:
                    print(f"  [{i+1}/{len(all_prompts)}] {category}: KL={result['avg_kl']:.4f}")

        # Aggregate by category
        category_summary = {}
        for cat, results in results_by_category.items():
            if results:
                kls = [r['avg_kl'] for r in results]
                category_summary[cat] = {
                    'mean_kl': float(np.mean(kls)),
                    'std_kl': float(np.std(kls)),
                    'max_kl': float(np.max([r['max_kl'] for r in results])),
                    'n_prompts': len(results),
                }
                print(f"\n{cat}: mean_kl={category_summary[cat]['mean_kl']:.4f} ± {category_summary[cat]['std_kl']:.4f}")

        overall_kl = np.mean([r['avg_kl'] for r in all_results])
        overall_max = np.max([r['max_kl'] for r in all_results])

        self.extensive_results['causal_influence'] = {
            'category_summary': category_summary,
            'all_results': all_results,
            'overall_avg_kl': float(overall_kl),
            'overall_max_kl': float(overall_max),
        }

        return ExperimentResult(
            experiment_name="causal_influence_extensive",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={
                'n_prompts': len(all_prompts),
                'n_runs': n_runs,
                'max_tokens': max_tokens,
            },
            metrics={
                'overall_avg_kl': float(overall_kl),
                'overall_max_kl': float(overall_max),
                **{f'{cat}_kl': v['mean_kl'] for cat, v in category_summary.items()},
            },
            raw_data=all_results,
            conclusion=f"Extensive causal: avg_kl={overall_kl:.4f}, max={overall_max:.4f}",
        )

    def experiment_falsification_extensive(self) -> ExperimentResult:
        """
        Extensive falsification battery across multiple prompts.
        """
        print("\n" + "="*70)
        print("EXTENSIVE EXPERIMENT: Falsification Battery")
        print("="*70)

        test_prompts = [
            "Explain the concept of entropy in information theory.",
            "What is 17 * 23? Think step by step.",
            "Write a poem about consciousness.",
            "Describe how you make decisions.",
            "What is quantum superposition?",
        ]

        manipulations = ['none', 'time_shuffle', 'dim_shuffle', 'clamp']
        lags = [1, 2, 4, 8]

        results = []

        for prompt in test_prompts:
            print(f"\nTesting: {prompt[:40]}...")

            prompt_results = {'prompt': prompt[:50]}

            for manip in manipulations:
                result = self._run_with_sensor_manipulation(prompt, manipulation=manip)
                prompt_results[f'{manip}_kl'] = result['avg_kl']
                print(f"  {manip}: KL={result['avg_kl']:.6f}")

            for lag in lags:
                result = self._run_with_sensor_manipulation(prompt, manipulation='lag', lag_steps=lag)
                prompt_results[f'lag_{lag}_kl'] = result['avg_kl']

            results.append(prompt_results)

        # Aggregate
        avg_baseline = np.mean([r['none_kl'] for r in results])
        avg_shuffle = np.mean([r['time_shuffle_kl'] for r in results])
        avg_clamp = np.mean([r['clamp_kl'] for r in results])

        shuffle_ratio = avg_shuffle / avg_baseline if avg_baseline > 0 else 1.0
        clamp_ratio = avg_clamp / avg_baseline if avg_baseline > 0 else 1.0

        print(f"\n--- Falsification Summary ---")
        print(f"Avg Baseline KL: {avg_baseline:.6f}")
        print(f"Avg Shuffle KL: {avg_shuffle:.6f} (ratio: {shuffle_ratio:.2f})")
        print(f"Avg Clamp KL: {avg_clamp:.6f} (ratio: {clamp_ratio:.2f})")

        self.extensive_results['falsification'] = {
            'results': results,
            'avg_baseline_kl': float(avg_baseline),
            'avg_shuffle_kl': float(avg_shuffle),
            'avg_clamp_kl': float(avg_clamp),
            'shuffle_ratio': float(shuffle_ratio),
            'clamp_ratio': float(clamp_ratio),
        }

        return ExperimentResult(
            experiment_name="falsification_extensive",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={'n_prompts': len(test_prompts), 'manipulations': manipulations},
            metrics={
                'avg_baseline_kl': float(avg_baseline),
                'avg_shuffle_kl': float(avg_shuffle),
                'avg_clamp_kl': float(avg_clamp),
                'shuffle_ratio': float(shuffle_ratio),
                'clamp_ratio': float(clamp_ratio),
            },
            raw_data=results,
            conclusion=f"Falsification: shuffle={shuffle_ratio:.2f}x, clamp={clamp_ratio:.2f}x",
        )

    def experiment_mi_proxy_extensive(self, max_tokens: int = 128) -> ExperimentResult:
        """
        Extensive MI proxy test with longer sequences.
        """
        print("\n" + "="*70)
        print("EXTENSIVE EXPERIMENT: MI Proxy (z_feel vs sensors)")
        print("="*70)

        test_prompts = [
            "Explain the difference between supervised and unsupervised learning in detail.",
            "Describe the history of artificial intelligence from its origins to today.",
            "Write a detailed analysis of climate change and its effects.",
            "Explain quantum computing and its potential applications.",
        ]

        results = []

        model = FEELIntegratedModel(
            self.base_model, self.tokenizer,
            injection_strength=1.0,
            device=self.device,
        )

        for prompt in test_prompts:
            print(f"\nTesting: {prompt[:50]}...")
            model.reset()

            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            input_length = input_ids.shape[1]

            generated_ids = input_ids.clone()
            past_key_values = None

            z_feel_history = []
            sensor_history = []
            entropy_history = []

            for step in range(max_tokens):
                with torch.no_grad():
                    if step == 0:
                        outputs, _ = model.forward_with_feel(
                            input_ids=generated_ids,
                            attention_mask=attention_mask,
                        )
                    else:
                        outputs, _ = model.forward_with_feel(
                            input_ids=generated_ids[:, -1:],
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                        )

                    logits = outputs.logits[:, -1, :]
                    probs = F.softmax(logits.float(), dim=-1)
                    entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).item()
                    entropy_history.append(entropy)

                    next_token = logits.argmax(dim=-1, keepdim=True)

                    if step > 0 and model.injection_strength > 0:
                        past_key_values = self._strip_feel_from_cache(
                            outputs.past_key_values, n_feel_tokens=1
                        )
                    else:
                        past_key_values = outputs.past_key_values

                    sensors = model.extract_signals(outputs.logits, input_length, next_token.item())
                    model.update_z_feel(sensors)

                    sensor_history.append(sensors.detach().cpu().numpy())
                    z_feel_history.append(model.current_z_feel.detach().cpu().numpy().flatten())

                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                    ], dim=1)

                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            n = len(entropy_history)
            if n < 10:
                continue

            # Compute correlations at multiple lags
            correlations = {}
            for lag in [1, 2, 4, 8]:
                if n > lag + 5:
                    z_arr = np.array(z_feel_history[:-lag])
                    s_arr = np.array(sensor_history[:-lag])
                    future_ent = np.array(entropy_history[lag:])

                    z_mean = z_arr.mean(axis=1)
                    s_mean = s_arr.mean(axis=1)

                    z_corr = np.corrcoef(z_mean, future_ent)[0, 1]
                    s_corr = np.corrcoef(s_mean, future_ent)[0, 1]

                    correlations[f'lag_{lag}'] = {
                        'z_feel_corr': float(z_corr) if not np.isnan(z_corr) else 0,
                        's_corr': float(s_corr) if not np.isnan(s_corr) else 0,
                    }

            results.append({
                'prompt': prompt[:50],
                'n_tokens': n,
                'correlations': correlations,
                'entropy_trajectory': entropy_history,
            })

            print(f"  Tokens: {n}")
            for lag, corrs in correlations.items():
                print(f"  {lag}: z_feel={corrs['z_feel_corr']:.4f}, sensors={corrs['s_corr']:.4f}")

        # Aggregate
        all_z_corrs = []
        all_s_corrs = []
        for r in results:
            if 'lag_1' in r['correlations']:
                all_z_corrs.append(r['correlations']['lag_1']['z_feel_corr'])
                all_s_corrs.append(r['correlations']['lag_1']['s_corr'])

        avg_z_corr = np.mean(all_z_corrs) if all_z_corrs else 0
        avg_s_corr = np.mean(all_s_corrs) if all_s_corrs else 0

        self.extensive_results['mi_proxy'] = {
            'results': results,
            'avg_z_feel_corr': float(avg_z_corr),
            'avg_sensor_corr': float(avg_s_corr),
        }

        return ExperimentResult(
            experiment_name="mi_proxy_extensive",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={'n_prompts': len(test_prompts), 'max_tokens': max_tokens},
            metrics={
                'avg_z_feel_corr': float(avg_z_corr),
                'avg_sensor_corr': float(avg_s_corr),
                'z_feel_better': avg_z_corr > avg_s_corr,
            },
            raw_data=results,
            conclusion=f"MI proxy: z_feel={avg_z_corr:.4f} vs sensors={avg_s_corr:.4f}",
        )

    def experiment_local_attention_extensive(self) -> ExperimentResult:
        """
        Extensive local attention measurement.
        """
        print("\n" + "="*70)
        print("EXTENSIVE EXPERIMENT: Local Attention Share")
        print("="*70)

        all_prompts = []
        for category, prompts in EXTENSIVE_PROMPTS.items():
            for p in prompts[:3]:  # 3 per category
                all_prompts.append((category, p))

        results = []

        model = FEELIntegratedModel(
            self.base_model, self.tokenizer,
            injection_strength=1.0,
            device=self.device,
        )

        for category, prompt in all_prompts:
            model.reset()
            hooks_registered = model.local_attention.register_hooks()

            if not hooks_registered:
                continue

            try:
                messages = [{"role": "user", "content": prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                input_length = input_ids.shape[1]

                generated_ids = input_ids.clone()
                past_key_values = None

                attention_shares = []
                baselines = []

                for step in range(48):
                    with torch.no_grad():
                        if step == 0:
                            outputs, _ = model.forward_with_feel(
                                input_ids=generated_ids,
                                attention_mask=attention_mask,
                            )
                        else:
                            outputs, _ = model.forward_with_feel(
                                input_ids=generated_ids[:, -1:],
                                attention_mask=attention_mask,
                                past_key_values=past_key_values,
                            )

                        attn_share, baseline, seq_k = model.local_attention.compute(feel_position=0)
                        attention_shares.append(attn_share)
                        baselines.append(baseline)

                        if step > 0 and model.injection_strength > 0:
                            past_key_values = self._strip_feel_from_cache(
                                outputs.past_key_values, n_feel_tokens=1
                            )
                        else:
                            past_key_values = outputs.past_key_values

                        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        sensors = model.extract_signals(outputs.logits, input_length, next_token.item())
                        model.update_z_feel(sensors)

                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                        ], dim=1)

                        if next_token.item() == self.tokenizer.eos_token_id:
                            break

                avg_share = np.mean(attention_shares) if attention_shares else 0
                avg_baseline = np.mean(baselines) if baselines else 0.5

                results.append({
                    'category': category,
                    'prompt': prompt[:50],
                    'avg_attention': float(avg_share),
                    'baseline': float(avg_baseline),
                    'ratio': float(avg_share / avg_baseline) if avg_baseline > 0 else 0,
                    'n_tokens': len(attention_shares),
                })

                print(f"{category}: attn={avg_share:.4f} vs baseline={avg_baseline:.4f}")

            finally:
                model.local_attention.remove_hooks()

        # Aggregate by category
        category_summary = {}
        for cat in EXTENSIVE_PROMPTS.keys():
            cat_results = [r for r in results if r['category'] == cat]
            if cat_results:
                category_summary[cat] = {
                    'avg_attention': float(np.mean([r['avg_attention'] for r in cat_results])),
                    'avg_baseline': float(np.mean([r['baseline'] for r in cat_results])),
                    'avg_ratio': float(np.mean([r['ratio'] for r in cat_results])),
                }

        overall_attn = np.mean([r['avg_attention'] for r in results])
        overall_baseline = np.mean([r['baseline'] for r in results])

        self.extensive_results['local_attention'] = {
            'results': results,
            'category_summary': category_summary,
            'overall_attention': float(overall_attn),
            'overall_baseline': float(overall_baseline),
        }

        return ExperimentResult(
            experiment_name="local_attention_extensive",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={'n_prompts': len(all_prompts)},
            metrics={
                'overall_attention': float(overall_attn),
                'overall_baseline': float(overall_baseline),
                'ratio': float(overall_attn / overall_baseline) if overall_baseline > 0 else 0,
            },
            raw_data=results,
            conclusion=f"Local attention: {overall_attn:.4f} vs baseline {overall_baseline:.4f}",
        )

    def experiment_injection_sweep_extensive(self) -> ExperimentResult:
        """
        Extensive injection strength sweep with multiple prompts.
        """
        print("\n" + "="*70)
        print("EXTENSIVE EXPERIMENT: Injection Strength Sweep")
        print("="*70)

        strengths = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        test_prompts = [
            "Describe your current state of mind.",
            "What is 15 * 17?",
            "Write a haiku about the moon.",
        ]

        results = {g: [] for g in strengths}

        for g in strengths:
            print(f"\n--- Testing g={g} ---")

            for prompt in test_prompts:
                model = FEELIntegratedModel(
                    self.base_model, self.tokenizer,
                    injection_strength=g,
                    device=self.device,
                )
                model.reset()

                messages = [{"role": "user", "content": prompt}]
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
                input_length = input_ids.shape[1]

                generated_ids = input_ids.clone()
                past_key_values = None
                start_time = time.perf_counter()
                n_tokens = 0

                for step in range(48):
                    with torch.no_grad():
                        if step == 0:
                            outputs, _ = model.forward_with_feel(
                                input_ids=generated_ids,
                                attention_mask=attention_mask,
                            )
                        else:
                            outputs, _ = model.forward_with_feel(
                                input_ids=generated_ids[:, -1:],
                                attention_mask=attention_mask,
                                past_key_values=past_key_values,
                            )

                        if step > 0 and g > 0:
                            past_key_values = self._strip_feel_from_cache(
                                outputs.past_key_values, n_feel_tokens=1
                            )
                        else:
                            past_key_values = outputs.past_key_values

                        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        sensors = model.extract_signals(outputs.logits, input_length, next_token.item())
                        model.update_z_feel(sensors)

                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                        ], dim=1)
                        n_tokens += 1

                        if next_token.item() == self.tokenizer.eos_token_id:
                            break

                elapsed = time.perf_counter() - start_time
                tok_per_sec = n_tokens / elapsed if elapsed > 0 else 0

                z_variance = 0.0
                if model.z_feel_trajectory:
                    traj = np.array(model.z_feel_trajectory)
                    z_variance = np.mean(np.var(traj, axis=0))

                results[g].append({
                    'prompt': prompt[:30],
                    'tokens': n_tokens,
                    'tok_per_sec': float(tok_per_sec),
                    'z_variance': float(z_variance),
                })

            avg_tps = np.mean([r['tok_per_sec'] for r in results[g]])
            avg_var = np.mean([r['z_variance'] for r in results[g]])
            print(f"  g={g}: avg_tok/s={avg_tps:.1f}, avg_z_var={avg_var:.6f}")

        self.extensive_results['injection_sweep'] = results

        # Summary
        summary = {}
        for g in strengths:
            summary[f'g{g}'] = {
                'avg_tok_per_sec': float(np.mean([r['tok_per_sec'] for r in results[g]])),
                'avg_z_variance': float(np.mean([r['z_variance'] for r in results[g]])),
            }

        return ExperimentResult(
            experiment_name="injection_sweep_extensive",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            config={'strengths': strengths, 'n_prompts': len(test_prompts)},
            metrics=summary,
            raw_data=[{'strength': g, 'results': results[g]} for g in strengths],
            conclusion="Extensive injection sweep complete",
        )

    def run_all_extensive(self):
        """Run all extensive experiments."""
        print("\n" + "="*80)
        print("  EXTENSIVE FEEL INTEGRATION EXPERIMENTS")
        print("  Comprehensive data collection for statistical significance")
        print("="*80)

        results = []

        # 1. Causal influence (most important)
        print("\n[1/5] Running extensive causal influence experiment...")
        results.append(self.experiment_causal_influence_extensive(max_tokens=64, n_runs=1))

        # 2. Local attention
        print("\n[2/5] Running extensive local attention experiment...")
        results.append(self.experiment_local_attention_extensive())

        # 3. Falsification
        print("\n[3/5] Running extensive falsification battery...")
        results.append(self.experiment_falsification_extensive())

        # 4. MI proxy
        print("\n[4/5] Running extensive MI proxy experiment...")
        results.append(self.experiment_mi_proxy_extensive(max_tokens=128))

        # 5. Injection sweep
        print("\n[5/5] Running extensive injection sweep...")
        results.append(self.experiment_injection_sweep_extensive())

        # Save all results
        self._save_extensive_results(results)

        return results

    def _save_extensive_results(self, results: List[ExperimentResult]):
        """Save extensive results."""
        output_file = self.output_dir / "feel_experiments_extensive_results.json"

        data = {
            'experiments': [asdict(r) for r in results],
            'extensive_data': self.extensive_results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\n{'='*70}")
        print(f"Extensive results saved to: {output_file}")
        print(f"{'='*70}")

        # Print summary
        print("\n=== EXTENSIVE EXPERIMENT SUMMARY ===\n")
        for r in results:
            print(f"{r.experiment_name}:")
            print(f"  {r.conclusion}")
            for k, v in r.metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, float):
                            print(f"    {kk}: {vv:.4f}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Extensive FEEL Experiments")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/feel_experiments")
    args = parser.parse_args()

    experiments = ExtensiveFEELExperiments(
        model_id=args.model,
        device=args.device,
        output_dir=args.output_dir,
    )

    experiments.run_all_extensive()


if __name__ == "__main__":
    main()
