#!/usr/bin/env python3
"""
FEEL z24: GRPO Trainer (Group Relative Policy Optimization)
============================================================
True GRPO implementation for embodied language model training.

Key differences from SFT:
1. Sample K completions per prompt
2. Score each with multi-objective reward
3. Compute group-relative advantages
4. Policy gradient update

Author: FEEL Research Team
Date: 2026-01-13
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import LogitsProcessor, LogitsProcessorList

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling.z24_sensor_hub import AMDSensorHub, SimulatedSensorHub, SENSOR_DIM
from modeling.z24_embodied_model import (
    EmbodiedDeepSeek, load_embodied_model, EmbodiedLoss
)

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[WARN] wandb not installed, logging disabled")


class EmbodiedLogitsProcessor(LogitsProcessor):
    """
    OPTIMIZED LogitsProcessor for per-token sensor injection.

    Key optimizations:
    1. Sample sensors every N steps (not every token) - reduces GPU-CPU syncs
    2. Keep tensors on GPU until get_trajectory() is called
    3. Minimal Python overhead per step

    The closed-loop still works because:
    - Sensors are injected frequently enough (every 8 steps)
    - Stats are recorded efficiently
    """

    # How often to sample sensors (every N tokens)
    SENSOR_SAMPLE_RATE = 8  # Sample every 8 tokens instead of every token

    def __init__(self, model: 'EmbodiedDeepSeek'):
        self.model = model
        self.embodied_blocks = model.embodied_blocks if hasattr(model, 'embodied_blocks') else {}

        # Trajectory recording - kept as lists of floats (no numpy until end)
        self.sensor_samples = []         # [sample, 32] - sampled every N steps
        self.gate_trajectory = []        # [step] avg gate
        self.strain_trajectory = []      # [step] avg strain
        self.film_gamma_trajectory = []  # [step] avg FiLM gamma
        self.film_beta_trajectory = []   # [step] avg FiLM beta
        self.step_count = 0
        self._cached_sensors = None      # Cache sensors on GPU

    def reset(self):
        """Reset trajectory for new generation."""
        self.sensor_samples = []
        self.gate_trajectory = []
        self.strain_trajectory = []
        self.film_gamma_trajectory = []
        self.film_beta_trajectory = []
        self.step_count = 0
        self._cached_sensors = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Called at each generation step. Optimized for speed.
        """
        if not self.embodied_blocks:
            return scores

        self.step_count += 1

        # === 1. SAMPLE SENSORS PERIODICALLY (not every step) ===
        if self.step_count % self.SENSOR_SAMPLE_RATE == 1 or self._cached_sensors is None:
            self._cached_sensors = self.model._get_sensors()
            # Only record sensor samples periodically (avoid GPU-CPU sync)
            self.sensor_samples.append(self._cached_sensors.detach().clone())

        # === 2. INJECT CACHED SENSORS INTO BLOCKS (always, but cheap) ===
        for block in self.embodied_blocks.values():
            block.sensors = self._cached_sensors  # Direct assignment, no copy

        # === 3. RECORD STATS (fast - just reading cached floats) ===
        gate_sum = 0.0
        strain_sum = 0.0
        gamma_sum = 0.0
        beta_sum = 0.0
        n_blocks = 0

        for block in self.embodied_blocks.values():
            n_blocks += 1
            if hasattr(block.gate, 'last_gate_prob'):
                gate_sum += block.gate.last_gate_prob
            if block.strain is not None and hasattr(block.strain, 'last_strain_magnitude'):
                strain_sum += block.strain.last_strain_magnitude
            if block.film is not None:
                if hasattr(block.film, 'last_gamma_mean'):
                    gamma_sum += block.film.last_gamma_mean
                if hasattr(block.film, 'last_beta_mean'):
                    beta_sum += block.film.last_beta_mean

        if n_blocks > 0:
            self.gate_trajectory.append(gate_sum / n_blocks)
            self.strain_trajectory.append(strain_sum / n_blocks)
            self.film_gamma_trajectory.append(gamma_sum / n_blocks)
            self.film_beta_trajectory.append(beta_sum / n_blocks)

        return scores

    def get_trajectory(self) -> dict:
        """Get recorded trajectory. GPU->CPU copy happens here (once per completion)."""
        # Convert GPU tensors to numpy only at the end
        sensor_np = []
        for s in self.sensor_samples:
            if isinstance(s, torch.Tensor):
                sensor_np.append(s.cpu().numpy())
            else:
                sensor_np.append(s)

        return {
            "sensors": sensor_np,
            "gates": self.gate_trajectory.copy(),
            "strains": self.strain_trajectory.copy(),
            "film_gammas": self.film_gamma_trajectory.copy(),
            "film_betas": self.film_beta_trajectory.copy(),
        }

    def get_summary(self) -> dict:
        """Get summary stats for logging."""
        return {
            "avg_gate": np.mean(self.gate_trajectory) if self.gate_trajectory else 0.5,
            "avg_strain": np.mean(self.strain_trajectory) if self.strain_trajectory else 0.0,
            "avg_film_gamma": np.mean(self.film_gamma_trajectory) if self.film_gamma_trajectory else 1.0,
            "avg_film_beta": np.mean(self.film_beta_trajectory) if self.film_beta_trajectory else 0.0,
            "trajectory_length": self.step_count,
        }


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    # Sampling
    num_samples: int = 4          # K completions per prompt
    max_new_tokens: int = 128     # Max tokens to generate
    temperature: float = 0.8      # Sampling temperature
    top_p: float = 0.9            # Nucleus sampling

    # Reward weights
    task_weight: float = 1.0      # Task quality (judge/heuristic)
    metabolic_weight: float = 0.4 # Energy efficiency
    strain_weight: float = 0.3    # Strain-behavior alignment
    stability_weight: float = 0.2 # Sensor stability
    brevity_weight: float = 0.1   # Conciseness bonus

    # Training
    learning_rate: float = 1e-5   # Lower for RL
    kl_coef: float = 0.1          # KL penalty vs reference
    clip_range: float = 0.2       # PPO-style clipping
    entropy_coef: float = 0.01    # Entropy bonus

    # Validation
    val_every: int = 50           # Validate every N prompts
    save_every: int = 200         # Save checkpoint every N prompts
    log_every: int = 10           # Log to wandb every N prompts


@dataclass
class CompletionResult:
    """Result from generating a completion."""
    prompt: str
    completion: str
    full_text: str
    log_probs: torch.Tensor       # Per-token log probs
    tokens: List[int]
    full_ids: torch.Tensor        # Full token IDs (for policy gradient)
    prompt_len: int               # Length of prompt in tokens

    # Sensor data during generation
    avg_gate: float
    avg_strain: float
    sensor_trajectory: List[np.ndarray]
    stress_level: float

    # Rewards (computed after)
    task_reward: float = 0.0
    metabolic_reward: float = 0.0
    strain_reward: float = 0.0
    stability_reward: float = 0.0
    brevity_reward: float = 0.0
    total_reward: float = 0.0


class GRPODataset(Dataset):
    """Dataset for GRPO training."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path) as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        print(f"[GRPODataset] Loaded {len(self.samples)} prompts")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt = sample.get("prompt") or sample.get("input", "")
        reference = sample.get("response") or sample.get("output", "")

        # Get stress from dataset or derive from is_stressed
        stress = sample.get("stress", None)
        if stress is None:
            is_stressed = sample.get("is_stressed", False)
            stress = 0.7 if is_stressed else 0.3

        return {
            "prompt": prompt,
            "reference": reference,
            "stress": stress,
            "task_type": sample.get("task_type", "general"),
        }


class GRPOTrainer:
    """
    GRPO Trainer for Embodied Language Models.

    Implements Group Relative Policy Optimization:
    1. For each prompt, sample K completions
    2. Score each with multi-objective reward
    3. Compute advantages relative to group mean
    4. Update policy with clipped surrogate objective
    """

    def __init__(
        self,
        model: EmbodiedDeepSeek,
        tokenizer,
        config: GRPOConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Reference model for KL penalty (frozen copy)
        self.ref_model = None  # Will load if needed

        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_parameters(),
            lr=config.learning_rate,
        )

        # Metrics tracking
        self.global_step = 0
        self.total_prompts = 0
        self.metrics_history = []

        # Sensor hub reference
        self.sensor_hub = model.sensor_hub if hasattr(model, 'sensor_hub') else None

        # LogitsProcessor for per-token sensor injection (reusable)
        self.embodied_processor = EmbodiedLogitsProcessor(model)

    def generate_completions(
        self,
        prompt: str,
        num_samples: int = None,
    ) -> List[CompletionResult]:
        """
        Generate K completions for a prompt with per-token sensor injection.

        CRITICAL: We use output_scores from generate() for old_log_probs.
        This keeps the log_probs on-policy (same mode as generation).
        """
        num_samples = num_samples or self.config.num_samples
        results = []

        # Encode prompt
        prompt_encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        prompt_len = prompt_encoding["input_ids"].shape[1]

        self.model.eval()

        for _ in range(num_samples):
            # Reset model stats and trajectory
            self.model.reset_statistics()
            self.embodied_processor.reset()

            # === SET INITIAL SENSORS before generation starts ===
            if self.sensor_hub:
                initial_sensors = self.sensor_hub.read_tensor()
                for block in self.model.embodied_blocks.values():
                    block.set_sensors(initial_sensors)

            # Generate with LogitsProcessor for per-token sensor injection
            with torch.no_grad():
                # Create processor list with our embodied processor
                logits_processors = LogitsProcessorList([self.embodied_processor])

                # Sample completion with sensor injection at each step
                outputs = self.model.base_model.generate(
                    **prompt_encoding,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,  # Get logits at each step
                    logits_processor=logits_processors,  # Per-token sensor injection
                )

                generated_ids = outputs.sequences[0]
                completion_ids = generated_ids[prompt_len:]

                # Decode
                completion = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                )
                full_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                # === COMPUTE OLD_LOG_PROBS FROM OUTPUT_SCORES (on-policy!) ===
                # outputs.scores is tuple of [batch_size, vocab_size] tensors, one per generated token
                if outputs.scores and len(outputs.scores) > 0:
                    # Stack scores: [num_tokens, batch_size, vocab_size]
                    stacked_scores = torch.stack(outputs.scores, dim=0)

                    # Squeeze batch dimension (batch_size=1 for single sample)
                    # Shape: [num_tokens, vocab_size]
                    stacked_scores = stacked_scores.squeeze(1)

                    # Apply log_softmax to get log probabilities
                    log_probs = F.log_softmax(stacked_scores, dim=-1)

                    # Get log prob of each selected token
                    # completion_ids[i] is the token selected at step i
                    # Shape: [num_tokens, 1]
                    token_indices = completion_ids.unsqueeze(-1)
                    token_log_probs = log_probs.gather(-1, token_indices).squeeze(-1)
                else:
                    # Fallback: empty completion
                    token_log_probs = torch.tensor([], device=self.device)

            # Get trajectory and stats from LogitsProcessor (real per-token data!)
            trajectory = self.embodied_processor.get_trajectory()
            proc_summary = self.embodied_processor.get_summary()

            # Compute stress from trajectory (average over generation)
            if trajectory["sensors"]:
                sensor_array = np.array(trajectory["sensors"])
                # stress_composite is index 31
                stress_level = np.mean(sensor_array[:, 31]) if sensor_array.shape[1] > 31 else 0.5
            else:
                stress_level = 0.5

            result = CompletionResult(
                prompt=prompt,
                completion=completion,
                full_text=full_text,
                log_probs=token_log_probs.detach(),  # Detach for storage
                tokens=completion_ids.tolist(),
                full_ids=generated_ids.clone(),
                prompt_len=prompt_len,
                # Use trajectory stats (actual per-token averages!)
                avg_gate=proc_summary["avg_gate"],
                avg_strain=proc_summary["avg_strain"],
                sensor_trajectory=trajectory["sensors"],  # Full trajectory for reward
                stress_level=stress_level,
            )

            results.append(result)

        return results

    def compute_rewards(
        self,
        completions: List[CompletionResult],
        reference: str,
        target_stress: float,
    ) -> List[CompletionResult]:
        """Compute multi-objective rewards for completions."""

        # OPTIMIZATION: Tokenize reference ONCE (not K times)
        ref_tokens = self.tokenizer.encode(reference)
        ref_len = len(ref_tokens)

        for comp in completions:
            # === Task Reward ===
            # Heuristic: similarity to reference + coherence
            # In production, use a judge model
            task_reward = self._compute_task_reward(comp.completion, reference)

            # === Metabolic Reward ===
            # Low gate (conserving) under high stress = good
            # High gate (computing) under low stress = good
            stress = comp.stress_level
            gate = comp.avg_gate

            # Optimal: gate inversely proportional to stress
            optimal_gate = 1.0 - stress
            metabolic_reward = 1.0 - abs(gate - optimal_gate)

            # === Strain Reward ===
            # High strain should correlate with shorter output
            strain = comp.avg_strain
            length_ratio = len(comp.tokens) / 100  # Normalized by expected length

            # Reward if strain correlates with brevity
            if strain > 0.3 and length_ratio < 1.0:
                strain_reward = 1.0  # Good: strained and brief
            elif strain < 0.1 and length_ratio > 0.5:
                strain_reward = 0.8  # Good: relaxed and verbose
            else:
                strain_reward = 0.5  # Neutral

            # === Stability Reward ===
            # Based on sensor trajectory smoothness
            if comp.sensor_trajectory and len(comp.sensor_trajectory) > 1:
                deltas = []
                for i in range(1, len(comp.sensor_trajectory)):
                    delta = np.abs(comp.sensor_trajectory[i] - comp.sensor_trajectory[i-1]).mean()
                    deltas.append(delta)
                stability_reward = max(0, 1.0 - np.mean(deltas) * 5)
            else:
                stability_reward = 0.8  # Default if no trajectory

            # === Brevity Reward ===
            # Prefer concise but complete answers (ref_len already computed above)
            comp_len = len(comp.tokens)

            if comp_len < ref_len * 0.5:
                brevity_reward = 0.5  # Too short
            elif comp_len > ref_len * 2.0:
                brevity_reward = 0.5  # Too long
            else:
                brevity_reward = 1.0 - abs(comp_len - ref_len) / max(ref_len, 1) * 0.5

            # === Total Reward ===
            comp.task_reward = task_reward
            comp.metabolic_reward = metabolic_reward
            comp.strain_reward = strain_reward
            comp.stability_reward = stability_reward
            comp.brevity_reward = brevity_reward

            comp.total_reward = (
                self.config.task_weight * task_reward +
                self.config.metabolic_weight * metabolic_reward +
                self.config.strain_weight * strain_reward +
                self.config.stability_weight * stability_reward +
                self.config.brevity_weight * brevity_reward
            )

        return completions

    def _compute_task_reward(self, completion: str, reference: str) -> float:
        """
        Compute task quality reward.

        Simple heuristic version. In production, use a judge model.
        """
        # Length-normalized overlap
        comp_words = set(completion.lower().split())
        ref_words = set(reference.lower().split())

        if not ref_words:
            return 0.5

        # Jaccard similarity
        intersection = len(comp_words & ref_words)
        union = len(comp_words | ref_words)
        similarity = intersection / max(union, 1)

        # Check for key answer patterns (numbers, etc.)
        import re
        ref_numbers = set(re.findall(r'\d+', reference))
        comp_numbers = set(re.findall(r'\d+', completion))

        number_match = len(ref_numbers & comp_numbers) / max(len(ref_numbers), 1) if ref_numbers else 1.0

        # Combined score
        task_reward = 0.4 * similarity + 0.6 * number_match

        return min(1.0, task_reward)

    def compute_advantages(
        self,
        completions: List[CompletionResult],
    ) -> Tuple[List[float], float]:
        """
        Compute group-relative advantages.

        advantage_i = reward_i - mean(rewards)

        Optionally with normalization.
        """
        rewards = [c.total_reward for c in completions]
        baseline = np.mean(rewards)
        std = np.std(rewards) + 1e-8

        # Normalized advantages
        advantages = [(r - baseline) / std for r in rewards]

        return advantages, baseline

    def policy_gradient_step(
        self,
        completions: List[CompletionResult],
        advantages: List[float],
    ) -> Dict[str, float]:
        """
        Perform policy gradient update with clipped surrogate objective.

        OPTIMIZED: Batches all K completions into a single forward pass.
        Uses padding and attention masking to handle variable lengths.

        L = -E[min(r * A, clip(r, 1-eps, 1+eps) * A)]

        where r = pi_current(a|s) / pi_old(a|s)
        """
        self.model.train()

        # Filter valid completions
        valid_comps = [(c, a) for c, a in zip(completions, advantages) if len(c.tokens) > 0]
        if not valid_comps:
            return {"pg_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        # Inject current sensors ONCE for all completions
        if self.sensor_hub:
            sensors = self.sensor_hub.read_tensor()
            for block in self.model.embodied_blocks.values():
                block.set_sensors(sensors)

        self.model.reset_statistics()

        # === BATCHED FORWARD PASS ===
        # Pad sequences to same length
        max_len = max(c.full_ids.size(0) for c, _ in valid_comps)
        batch_size = len(valid_comps)

        # Create padded batch
        batch_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        batch_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)

        for i, (comp, _) in enumerate(valid_comps):
            seq_len = comp.full_ids.size(0)
            batch_ids[i, :seq_len] = comp.full_ids.to(self.device)
            batch_mask[i, :seq_len] = 1

        # Single forward pass for all K completions!
        outputs = self.model(input_ids=batch_ids, attention_mask=batch_mask)

        # === COMPUTE LOSSES PER COMPLETION ===
        total_loss = 0.0
        total_pg_loss = 0.0
        total_entropy = 0.0

        for i, (comp, advantage) in enumerate(valid_comps):
            prompt_len = comp.prompt_len
            seq_len = comp.full_ids.size(0)

            # Extract logits for this completion
            logits = outputs.logits[i, prompt_len-1:seq_len-1]  # [completion_len, vocab]
            targets = batch_ids[i, prompt_len:seq_len]  # [completion_len]

            if logits.size(0) == 0:
                continue

            log_probs = F.log_softmax(logits, dim=-1)
            current_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            # Old log probs (from output_scores during generation)
            old_log_probs = comp.log_probs.to(self.device)

            # Ensure tensor sizes match
            min_len = min(len(current_log_probs), len(old_log_probs))
            if min_len == 0:
                continue

            current_log_probs = current_log_probs[:min_len]
            old_log_probs = old_log_probs[:min_len]

            # Ratio (clamped for numerical stability)
            log_ratio = current_log_probs - old_log_probs
            log_ratio = torch.clamp(log_ratio, -10, 10)
            ratio = torch.exp(log_ratio)

            # Clipped surrogate
            advantage_tensor = torch.tensor(advantage, device=self.device)
            surr1 = ratio * advantage_tensor
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantage_tensor

            pg_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # Combined loss
            loss = pg_loss - self.config.entropy_coef * entropy

            total_loss += loss
            total_pg_loss += pg_loss.item()
            total_entropy += entropy.item()

        # Backward and step
        n_valid = len(valid_comps)
        if total_loss > 0 and n_valid > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), 1.0)
            self.optimizer.step()

        return {
            "pg_loss": total_pg_loss / max(n_valid, 1),
            "entropy": total_entropy / max(n_valid, 1),
            "total_loss": total_loss.item() / max(n_valid, 1) if isinstance(total_loss, torch.Tensor) else 0.0,
        }

    def validate(
        self,
        val_dataset: GRPODataset,
        num_samples: int = 20,
    ) -> Dict[str, float]:
        """Run validation with ablation comparison."""

        self.model.eval()

        metrics = {
            "val_task_reward": [],
            "val_metabolic_reward": [],
            "val_strain_reward": [],
            "val_total_reward": [],
            "val_avg_gate": [],
            "val_avg_strain": [],
            "val_completion_length": [],
        }

        # Sample random prompts
        indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)

        for idx in indices:
            sample = val_dataset[idx]

            # Generate single completion for validation
            completions = self.generate_completions(sample["prompt"], num_samples=1)
            completions = self.compute_rewards(completions, sample["reference"], sample["stress"])

            comp = completions[0]
            metrics["val_task_reward"].append(comp.task_reward)
            metrics["val_metabolic_reward"].append(comp.metabolic_reward)
            metrics["val_strain_reward"].append(comp.strain_reward)
            metrics["val_total_reward"].append(comp.total_reward)
            metrics["val_avg_gate"].append(comp.avg_gate)
            metrics["val_avg_strain"].append(comp.avg_strain)
            metrics["val_completion_length"].append(len(comp.tokens))

        # Aggregate
        return {k: np.mean(v) for k, v in metrics.items()}

    def ablation_test(
        self,
        val_dataset: GRPODataset,
        num_samples: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run ablation tests to validate causal effects.

        Modes:
        - full: All components
        - shuffle_sensors: Random sensor values
        - frozen_sensors: Constant sensors
        """
        results = {}

        modes = ["full", "shuffle_sensors", "frozen_sensors"]

        for mode in modes:
            # Set ablation mode
            original_mode = self.model.ablation_mode
            self.model.set_ablation_mode(mode)

            metrics = self.validate(val_dataset, num_samples)
            results[mode] = metrics

            # Restore
            self.model.set_ablation_mode(original_mode)

        # Compute causal scores
        full_reward = results["full"]["val_total_reward"]
        shuffle_reward = results["shuffle_sensors"]["val_total_reward"]
        frozen_reward = results["frozen_sensors"]["val_total_reward"]

        # If sensors matter, shuffle/frozen should be worse
        causal_score = (full_reward - shuffle_reward) + (full_reward - frozen_reward)
        results["causal_score"] = max(0, causal_score)

        return results

    def train_epoch(
        self,
        train_dataset: GRPODataset,
        val_dataset: GRPODataset = None,
        max_prompts: int = None,
    ) -> Dict[str, float]:
        """Train for one epoch."""

        epoch_metrics = {
            "rewards": [],
            "advantages": [],
            "pg_losses": [],
            "entropies": [],
            "gates": [],
            "strains": [],
        }

        max_prompts = max_prompts or len(train_dataset)
        indices = np.random.permutation(len(train_dataset))[:max_prompts]

        start_time = time.time()

        for i, idx in enumerate(indices):
            sample = train_dataset[idx]

            # === Generate K completions ===
            completions = self.generate_completions(
                sample["prompt"],
                num_samples=self.config.num_samples,
            )

            # === Compute rewards ===
            completions = self.compute_rewards(
                completions,
                sample["reference"],
                sample["stress"],
            )

            # === Compute advantages ===
            advantages, baseline = self.compute_advantages(completions)

            # === Policy gradient update ===
            pg_metrics = self.policy_gradient_step(completions, advantages)

            # === Track metrics ===
            rewards = [c.total_reward for c in completions]
            epoch_metrics["rewards"].extend(rewards)
            epoch_metrics["advantages"].extend(advantages)
            epoch_metrics["pg_losses"].append(pg_metrics["pg_loss"])
            epoch_metrics["entropies"].append(pg_metrics["entropy"])
            epoch_metrics["gates"].extend([c.avg_gate for c in completions])
            epoch_metrics["strains"].extend([c.avg_strain for c in completions])

            self.total_prompts += 1
            self.global_step += 1

            # === Logging ===
            if (i + 1) % self.config.log_every == 0:
                elapsed = time.time() - start_time
                prompts_per_sec = (i + 1) / elapsed

                log_data = {
                    "train/reward_mean": np.mean(rewards),
                    "train/reward_std": np.std(rewards),
                    "train/advantage_mean": np.mean(advantages),
                    "train/pg_loss": pg_metrics["pg_loss"],
                    "train/entropy": pg_metrics["entropy"],
                    "train/avg_gate": np.mean([c.avg_gate for c in completions]),
                    "train/avg_strain": np.mean([c.avg_strain for c in completions]),
                    "train/prompts_per_sec": prompts_per_sec,
                    "train/total_prompts": self.total_prompts,
                }

                # Detailed reward breakdown
                log_data["reward/task"] = np.mean([c.task_reward for c in completions])
                log_data["reward/metabolic"] = np.mean([c.metabolic_reward for c in completions])
                log_data["reward/strain"] = np.mean([c.strain_reward for c in completions])
                log_data["reward/stability"] = np.mean([c.stability_reward for c in completions])
                log_data["reward/brevity"] = np.mean([c.brevity_reward for c in completions])

                if HAS_WANDB and wandb.run:
                    wandb.log(log_data, step=self.global_step)

                print(f"[{i+1}/{max_prompts}] reward={np.mean(rewards):.3f} "
                      f"pg_loss={pg_metrics['pg_loss']:.4f} "
                      f"gate={np.mean([c.avg_gate for c in completions]):.3f} "
                      f"strain={np.mean([c.avg_strain for c in completions]):.4f}", flush=True)

            # === Validation ===
            if val_dataset and (i + 1) % self.config.val_every == 0:
                print(f"\n[Validation at prompt {i+1}]", flush=True)
                val_metrics = self.validate(val_dataset, num_samples=10)

                print(f"  val_reward={val_metrics['val_total_reward']:.3f} "
                      f"val_gate={val_metrics['val_avg_gate']:.3f} "
                      f"val_strain={val_metrics['val_avg_strain']:.4f}", flush=True)

                if HAS_WANDB and wandb.run:
                    wandb.log(val_metrics, step=self.global_step)

                # Ablation test every 2nd validation
                if (i + 1) % (self.config.val_every * 2) == 0:
                    print(f"\n[Ablation Test]")
                    ablation_results = self.ablation_test(val_dataset, num_samples=10)

                    print(f"  full_reward={ablation_results['full']['val_total_reward']:.3f}")
                    print(f"  shuffle_reward={ablation_results['shuffle_sensors']['val_total_reward']:.3f}")
                    print(f"  frozen_reward={ablation_results['frozen_sensors']['val_total_reward']:.3f}")
                    print(f"  causal_score={ablation_results['causal_score']:.3f}")

                    if HAS_WANDB and wandb.run:
                        wandb.log({
                            "ablation/full_reward": ablation_results["full"]["val_total_reward"],
                            "ablation/shuffle_reward": ablation_results["shuffle_sensors"]["val_total_reward"],
                            "ablation/frozen_reward": ablation_results["frozen_sensors"]["val_total_reward"],
                            "ablation/causal_score": ablation_results["causal_score"],
                        }, step=self.global_step)

                print()

            # === Checkpointing ===
            if (i + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_prompt_{i+1}")

        # Aggregate epoch metrics
        return {
            "epoch_reward_mean": np.mean(epoch_metrics["rewards"]),
            "epoch_reward_std": np.std(epoch_metrics["rewards"]),
            "epoch_pg_loss": np.mean(epoch_metrics["pg_losses"]),
            "epoch_avg_gate": np.mean(epoch_metrics["gates"]),
            "epoch_avg_strain": np.mean(epoch_metrics["strains"]),
        }

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        output_dir = Path("models/grpo_z24")
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state": {},
            "optimizer_state": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "total_prompts": self.total_prompts,
            "config": self.config.__dict__,
        }

        # Save only embodied components
        for key, block in self.model.embodied_blocks.items():
            checkpoint["model_state"][key] = {
                "gate": block.gate.state_dict(),
            }
            if block.film is not None:
                checkpoint["model_state"][key]["film"] = block.film.state_dict()
            if block.strain is not None:
                checkpoint["model_state"][key]["strain"] = block.strain.state_dict()

        path = output_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"[Checkpoint] Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="FEEL z24 GRPO Training")
    parser.add_argument("--data", type=str, default="data/ouroboros/ouroboros_train.jsonl")
    parser.add_argument("--val-data", type=str, default="data/ouroboros/ouroboros_val.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=500, help="Max prompts per epoch")
    parser.add_argument("--num-samples", type=int, default=4, help="K completions per prompt")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--wandb-project", type=str, default="feel-z24-grpo")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("FEEL z24: GRPO TRAINING")
    print("=" * 70)

    # Config
    config = GRPOConfig(
        num_samples=args.num_samples,
        learning_rate=args.lr,
        val_every=args.val_every,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    print(f"Config:")
    print(f"  num_samples (K): {config.num_samples}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  max_prompts: {args.max_prompts}")
    print(f"  epochs: {args.epochs}")
    print()

    # Initialize wandb
    if HAS_WANDB:
        run_name = args.run_name or f"grpo-{datetime.now().strftime('%Y%m%d_%H%M')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **config.__dict__,
                "max_prompts": args.max_prompts,
                "epochs": args.epochs,
                "data": args.data,
            }
        )
        print(f"WandB: {wandb.run.url}")

    # Load model
    print("\n[1/4] Loading EmbodiedDeepSeek...")
    model = load_embodied_model(
        use_film=True,
        use_strain=True,
        use_real_sensors=True,
    )
    model.freeze_base_model()
    tokenizer = model.tokenizer

    # Load datasets
    print("\n[2/4] Loading datasets...")
    train_dataset = GRPODataset(args.data, tokenizer)
    val_dataset = GRPODataset(args.val_data, tokenizer) if Path(args.val_data).exists() else None

    # Create trainer
    print("\n[3/4] Creating GRPO trainer...")
    trainer = GRPOTrainer(model, tokenizer, config)

    # Initial validation
    if val_dataset:
        print("\n[Initial Validation]")
        val_metrics = trainer.validate(val_dataset, num_samples=10)
        print(f"  Initial reward: {val_metrics['val_total_reward']:.3f}")
        print(f"  Initial gate: {val_metrics['val_avg_gate']:.3f}")
        print(f"  Initial strain: {val_metrics['val_avg_strain']:.4f}")

        if HAS_WANDB and wandb.run:
            wandb.log({f"init_{k}": v for k, v in val_metrics.items()}, step=0)

    # Training loop
    print("\n[4/4] Starting GRPO training...")
    print("=" * 70)

    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")

        epoch_metrics = trainer.train_epoch(
            train_dataset,
            val_dataset,
            max_prompts=args.max_prompts,
        )

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Mean reward: {epoch_metrics['epoch_reward_mean']:.3f} +/- {epoch_metrics['epoch_reward_std']:.3f}")
        print(f"  PG Loss: {epoch_metrics['epoch_pg_loss']:.4f}")
        print(f"  Avg gate: {epoch_metrics['epoch_avg_gate']:.3f}")
        print(f"  Avg strain: {epoch_metrics['epoch_avg_strain']:.4f}")

        if HAS_WANDB and wandb.run:
            wandb.log({f"epoch/{k}": v for k, v in epoch_metrics.items()})

        # Save epoch checkpoint
        trainer.save_checkpoint(f"epoch_{epoch + 1}")

    # Final validation with full ablation
    if val_dataset:
        print("\n" + "=" * 70)
        print("FINAL VALIDATION")
        print("=" * 70)

        final_val = trainer.validate(val_dataset, num_samples=50)
        print(f"\nFinal Metrics:")
        for k, v in final_val.items():
            print(f"  {k}: {v:.4f}")

        print("\nFinal Ablation Test:")
        ablation = trainer.ablation_test(val_dataset, num_samples=10)
        print(f"  Full reward: {ablation['full']['val_total_reward']:.3f}")
        print(f"  Shuffle reward: {ablation['shuffle_sensors']['val_total_reward']:.3f}")
        print(f"  Frozen reward: {ablation['frozen_sensors']['val_total_reward']:.3f}")
        print(f"  Causal score: {ablation['causal_score']:.3f}")

        if HAS_WANDB and wandb.run:
            wandb.log({
                "final/total_reward": final_val["val_total_reward"],
                "final/causal_score": ablation["causal_score"],
            })

    # Save final model
    trainer.save_checkpoint("final")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    if HAS_WANDB and wandb.run:
        print(f"\nWandB run: {wandb.run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
