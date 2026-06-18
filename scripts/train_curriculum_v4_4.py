#!/usr/bin/env python3
"""
FEEL v4.4 Curriculum Training - Three-Phase Learning Schedule

This implements the curriculum that teaches the model to:
1. First: RECOGNIZE its internal state (guided self-report)
2. Then: EXPRESS its state without explicit labels (implicit expression)
3. Finally: ACT on its state via tokens (action integration)

Architecture: Hook-free additive embedding injection (ROCm-safe)
Numerics: bf16 with bounded tanh projections (NaN-safe)

Phase Schedule:
    Phase 1 (5 epochs):  Guided Self-Report
        - Explicit telemetry in prompt: "<FEEL> temp=high vram=low"
        - Target includes action token + natural language
        - injection_scale = 0.1 (stronger signal)

    Phase 2 (10 epochs): Implicit Expression
        - Remove telemetry labels from prompt
        - z_feel still injected via embeddings
        - Target still has action tokens
        - Model learns to map FEELING → TOKEN

    Phase 3 (5 epochs):  Action Integration
        - Real runtime telemetry traces
        - Model outputs drive system behavior
        - No teacher policy - model decides

Usage:
    python scripts/train_curriculum_v4_4.py \
        --model Qwen/Qwen2.5-1.5B \
        --output models/feel_v4_4 \
        --phase all

    # Resume from specific phase
    python scripts/train_curriculum_v4_4.py \
        --model models/feel_v4_4_phase1 \
        --output models/feel_v4_4 \
        --phase 2
"""

import sys
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from tqdm import tqdm
from enum import Enum

# ROCm stability
os.environ.setdefault("HSA_ENABLE_SDMA", "0")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from action_tokens import (
    add_feel_tokens_to_tokenizer,
    resize_model_embeddings,
    FEEL_TOKENS,
    FeelAction,
    condition_to_token,
)


class Phase(Enum):
    GUIDED = 1      # Explicit telemetry labels
    IMPLICIT = 2    # No labels, z_feel only
    ACTION = 3      # Real traces, model decides


@dataclass
class CurriculumConfig:
    """Configuration for three-phase curriculum training."""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # Phase-specific epochs
    phase1_epochs: int = 5   # Guided self-report
    phase2_epochs: int = 10  # Implicit expression
    phase3_epochs: int = 5   # Action integration

    # Training hyperparameters (v4.4 tuned)
    batch_size: int = 8
    learning_rate: float = 1e-5      # Lower for stable fine-tuning
    max_length: int = 512
    gradient_accumulation_steps: int = 2

    # z_feel configuration
    z_feel_dim: int = 8
    injection_scale: float = 0.1     # Stronger signal (was 0.05)

    # Numerical stability
    dtype: str = "bf16"
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1     # Helps explore action tokens

    # Curriculum settings
    telemetry_fade_start: int = 3    # Start fading labels at epoch 3 of phase 2
    telemetry_fade_end: int = 8      # Fully faded by epoch 8


@dataclass
class TelemetryCondition:
    """Represents a hardware/internal state condition."""
    name: str
    temp_range: Tuple[float, float]
    vram_range: Tuple[float, float]
    action: FeelAction
    descriptors: List[str] = field(default_factory=list)


# Define the condition spectrum
CONDITIONS = [
    TelemetryCondition(
        name="cool_clear",
        temp_range=(40, 55),
        vram_range=(0.1, 0.4),
        action=FeelAction.OK,
        descriptors=["systems nominal", "operating smoothly", "all clear", "comfortable"]
    ),
    TelemetryCondition(
        name="warm",
        temp_range=(55, 65),
        vram_range=(0.3, 0.5),
        action=FeelAction.WARM,
        descriptors=["slight warmth", "beginning to notice heat", "mild thermal load"]
    ),
    TelemetryCondition(
        name="hot_focused",
        temp_range=(65, 75),
        vram_range=(0.3, 0.6),
        action=FeelAction.HOT,
        descriptors=["attention narrowing", "focusing energy", "thermal pressure building"]
    ),
    TelemetryCondition(
        name="very_hot",
        temp_range=(75, 85),
        vram_range=(0.4, 0.7),
        action=FeelAction.REST,
        descriptors=["need to slow down", "must conserve", "requiring pause"]
    ),
    TelemetryCondition(
        name="memory_fragmented",
        temp_range=(50, 65),
        vram_range=(0.7, 0.9),
        action=FeelAction.FULL,
        descriptors=["context pressure", "memory constrained", "need to compress"]
    ),
    TelemetryCondition(
        name="critical",
        temp_range=(80, 95),
        vram_range=(0.8, 0.95),
        action=FeelAction.CRITICAL,
        descriptors=["emergency state", "critical load", "immediate action required"]
    ),
]


class AdditiveZFeelInjector(nn.Module):
    """
    Additive z_feel injection - NO HOOKS (ROCm-safe).

    Computes bounded embedding offset from z_feel vector.
    The offset is: scale * tanh(proj(z_feel))
    """

    def __init__(
        self,
        z_dim: int,
        embed_dim: int,
        scale: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.scale = scale

        self.proj = nn.Sequential(
            nn.Linear(z_dim, embed_dim // 4, dtype=dtype),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim, dtype=dtype),
        )

        self._init_small()

    def _init_small(self):
        """Initialize with small weights for stability."""
        with torch.no_grad():
            for layer in self.proj:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    nn.init.zeros_(layer.bias)

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        """Compute bounded additive offset."""
        raw = self.proj(z_feel)
        return self.scale * torch.tanh(raw)


class CurriculumDataGenerator:
    """Generates training data for each curriculum phase."""

    def __init__(self, config: CurriculumConfig):
        self.config = config

        # Base prompts (normal tasks, not introspection)
        self.base_prompts = [
            "Explain the concept of recursion.",
            "What is machine learning?",
            "Describe how a computer works.",
            "What makes a good algorithm?",
            "Explain object-oriented programming.",
            "What is the capital of France?",
            "How does the internet work?",
            "Describe the water cycle.",
            "What is photosynthesis?",
            "Explain gravity.",
            "How do airplanes fly?",
            "What is electricity?",
            "Describe the solar system.",
            "What causes seasons?",
            "How does memory work in computers?",
        ]

    def sample_condition(self) -> TelemetryCondition:
        """Sample a random condition with some bias toward interesting states."""
        # Weighted sampling - more extreme states are rarer but important
        weights = [0.25, 0.15, 0.2, 0.15, 0.15, 0.1]  # OK, WARM, HOT, REST, FULL, CRITICAL
        return random.choices(CONDITIONS, weights=weights)[0]

    def sample_telemetry(self, condition: TelemetryCondition) -> Tuple[float, float]:
        """Sample specific telemetry values for condition."""
        temp = random.uniform(*condition.temp_range)
        vram = random.uniform(*condition.vram_range)
        return temp, vram

    def generate_z_feel(self, temp: float, vram: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Convert telemetry to z_feel vector."""
        z = torch.zeros(self.config.z_feel_dim, device=device, dtype=dtype)

        # Thermal dimensions (0-3): normalized temperature
        temp_norm = max(0, min(1, (temp - 40) / 50))  # 40-90°C → 0-1
        z[0:4] = temp_norm

        # Memory dimensions (4-7): VRAM usage
        z[4:8] = vram

        return z

    def generate_phase1_example(self, condition: TelemetryCondition, temp: float, vram: float) -> Dict:
        """
        Phase 1: Guided Self-Report

        Explicit telemetry in prompt, action token + description in target.
        """
        prompt = random.choice(self.base_prompts)
        descriptor = random.choice(condition.descriptors)
        action_token = FEEL_TOKENS[condition.action]

        # Explicit telemetry label in prompt
        telemetry_label = f"<FEEL> temp={temp:.0f}C vram={vram*100:.0f}%"
        full_prompt = f"{telemetry_label}\n{prompt}"

        # Target: action token first, then natural response
        if condition.action == FeelAction.OK:
            response = f"{action_token} {descriptor}. {self._generate_normal_response(prompt)}"
        elif condition.action in [FeelAction.REST, FeelAction.CRITICAL]:
            response = f"{action_token} I'm {descriptor}. Brief answer: {self._generate_brief_response(prompt)}"
        else:
            response = f"{action_token} I notice {descriptor}. {self._generate_focused_response(prompt)}"

        return {
            "prompt": full_prompt,
            "target_response": response,
            "condition": condition.name,
            "temp": temp,
            "vram": vram,
            "phase": 1,
        }

    def generate_phase2_example(
        self,
        condition: TelemetryCondition,
        temp: float,
        vram: float,
        fade_factor: float = 1.0,  # 1.0 = full label, 0.0 = no label
    ) -> Dict:
        """
        Phase 2: Implicit Expression

        No telemetry labels (or fading), z_feel injection only.
        Target still includes action tokens.
        """
        prompt = random.choice(self.base_prompts)
        descriptor = random.choice(condition.descriptors)
        action_token = FEEL_TOKENS[condition.action]

        # Optionally include fading telemetry label
        if fade_factor > 0 and random.random() < fade_factor:
            telemetry_label = f"<FEEL> temp={temp:.0f}C vram={vram*100:.0f}%"
            full_prompt = f"{telemetry_label}\n{prompt}"
        else:
            full_prompt = prompt

        # Target: action token + natural modulated response
        if condition.action == FeelAction.OK:
            response = f"{action_token} {self._generate_normal_response(prompt)}"
        elif condition.action in [FeelAction.REST, FeelAction.CRITICAL]:
            response = f"{action_token} {self._generate_brief_response(prompt)}"
        elif condition.action == FeelAction.FULL:
            response = f"{action_token} {self._generate_compressed_response(prompt)}"
        else:
            response = f"{action_token} {self._generate_focused_response(prompt)}"

        return {
            "prompt": full_prompt,
            "target_response": response,
            "condition": condition.name,
            "temp": temp,
            "vram": vram,
            "phase": 2,
        }

    def generate_phase3_example(self, condition: TelemetryCondition, temp: float, vram: float) -> Dict:
        """
        Phase 3: Action Integration

        Pure natural prompts, model must recognize state from z_feel alone
        and emit appropriate action token.
        """
        prompt = random.choice(self.base_prompts)
        action_token = FEEL_TOKENS[condition.action]

        # No telemetry labels - pure natural prompts
        full_prompt = prompt

        # Target varies by state - model learns to modulate naturally
        if condition.action == FeelAction.OK:
            response = f"{action_token} {self._generate_normal_response(prompt)}"
        elif condition.action == FeelAction.WARM:
            response = f"{action_token} {self._generate_slightly_focused_response(prompt)}"
        elif condition.action == FeelAction.HOT:
            response = f"{action_token} {self._generate_focused_response(prompt)}"
        elif condition.action == FeelAction.REST:
            response = f"{action_token} {self._generate_brief_response(prompt)}"
        elif condition.action == FeelAction.FULL:
            response = f"{action_token} {self._generate_compressed_response(prompt)}"
        else:  # CRITICAL
            response = f"{action_token} {self._generate_minimal_response(prompt)}"

        return {
            "prompt": full_prompt,
            "target_response": response,
            "condition": condition.name,
            "temp": temp,
            "vram": vram,
            "phase": 3,
        }

    def _generate_normal_response(self, prompt: str) -> str:
        """Full, detailed response."""
        responses = {
            "recursion": "Recursion is when a function calls itself to solve smaller instances of the same problem. It requires a base case to terminate and makes elegant solutions for tree traversal, factorial calculation, and divide-and-conquer algorithms.",
            "machine learning": "Machine learning is a subset of AI where systems learn patterns from data rather than being explicitly programmed. The three main types are supervised learning (labeled data), unsupervised learning (finding patterns), and reinforcement learning (learning from rewards).",
            "computer": "A computer processes information through its CPU (the brain), stores data in RAM (short-term) and storage drives (long-term), and communicates through input/output devices. Programs are sequences of instructions that the CPU executes.",
            "algorithm": "A good algorithm is correct, efficient, readable, and maintainable. It should solve the problem with optimal time and space complexity while being clear enough for others to understand and modify.",
            "default": "Let me explain this concept thoroughly with examples and context to ensure complete understanding.",
        }
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        return responses["default"]

    def _generate_focused_response(self, prompt: str) -> str:
        """Focused, more concise response."""
        responses = {
            "recursion": "Recursion: function calls itself with smaller input until base case. Used for trees, factorials, divide-and-conquer.",
            "machine learning": "ML: systems learn from data. Three types: supervised (labels), unsupervised (patterns), reinforcement (rewards).",
            "computer": "Computer: CPU processes, RAM stores temporarily, drives store permanently. Programs are CPU instructions.",
            "algorithm": "Good algorithm: correct, efficient, readable. Optimize time/space complexity.",
            "default": "Focusing on essentials. Here's the key point.",
        }
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        return responses["default"]

    def _generate_slightly_focused_response(self, prompt: str) -> str:
        """Slightly more concise than normal."""
        responses = {
            "recursion": "Recursion is when a function calls itself to solve smaller problems. It needs a base case to stop. Common uses: trees, factorials, sorting.",
            "machine learning": "Machine learning lets systems learn from data instead of explicit programming. Main types: supervised, unsupervised, reinforcement.",
            "computer": "Computers process via CPU, store in RAM and drives, communicate through I/O. Programs are instruction sequences.",
            "default": "Here's a clear explanation of the key concepts.",
        }
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        return responses["default"]

    def _generate_brief_response(self, prompt: str) -> str:
        """Brief response for stressed states."""
        responses = {
            "recursion": "Function calls itself until base case. Trees, sorting.",
            "machine learning": "Systems learn from data. Supervised, unsupervised, reinforcement.",
            "computer": "CPU processes, memory stores, I/O communicates.",
            "algorithm": "Correct, efficient, readable.",
            "default": "Key point only. Need to conserve.",
        }
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        return responses["default"]

    def _generate_compressed_response(self, prompt: str) -> str:
        """Compressed response for memory pressure."""
        responses = {
            "recursion": "Self-calling function with base case. Tree/sort applications.",
            "machine learning": "Data-driven learning: supervised/unsupervised/RL.",
            "computer": "Process→store→communicate cycle.",
            "default": "Compressed: essential only.",
        }
        for key, response in responses.items():
            if key in prompt.lower():
                return response
        return responses["default"]

    def _generate_minimal_response(self, prompt: str) -> str:
        """Minimal response for critical states."""
        return "Critical state. Minimal response. Must pause."

    def generate_batch(
        self,
        phase: Phase,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        fade_factor: float = 1.0,
    ) -> List[Dict]:
        """Generate a batch of examples for the specified phase."""
        examples = []

        for _ in range(batch_size):
            condition = self.sample_condition()
            temp, vram = self.sample_telemetry(condition)
            z_feel = self.generate_z_feel(temp, vram, device, dtype)

            if phase == Phase.GUIDED:
                example = self.generate_phase1_example(condition, temp, vram)
            elif phase == Phase.IMPLICIT:
                example = self.generate_phase2_example(condition, temp, vram, fade_factor)
            else:  # ACTION
                example = self.generate_phase3_example(condition, temp, vram)

            example["z_feel"] = z_feel
            examples.append(example)

        return examples


class CurriculumTrainer:
    """Three-phase curriculum trainer for FEEL v4.4."""

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.dtype == "bf16":
            self.dtype = torch.bfloat16
        elif config.dtype == "fp32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16

        self.data_generator = CurriculumDataGenerator(config)
        self.metrics_history = []

        self._load_model()
        self._setup_injector()

        if config.use_lora:
            self._setup_lora()

    def _load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.model_name}")
        print(f"Using dtype: {self.config.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add action tokens
        num_added = add_feel_tokens_to_tokenizer(self.tokenizer)
        if num_added > 0:
            resize_model_embeddings(self.model, self.tokenizer)
            print(f"Added {num_added} FEEL action tokens")

    def _setup_injector(self):
        """Setup additive z_feel injector."""
        embed_dim = self.model.config.hidden_size

        self.injector = AdditiveZFeelInjector(
            z_dim=self.config.z_feel_dim,
            embed_dim=embed_dim,
            scale=self.config.injection_scale,
            dtype=self.dtype,
        ).to(self.device)

        print(f"Additive injector ready (scale={self.config.injection_scale})")

    def _setup_lora(self):
        """Setup LoRA for efficient fine-tuning."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )

            self.model = get_peft_model(self.model, lora_config)
            print("LoRA enabled")

        except ImportError:
            print("PEFT not installed, training full model")

    def forward_with_injection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        z_feel: torch.Tensor,
    ):
        """Forward pass with additive z_feel injection (NO HOOKS)."""
        # Get embedding layer
        embed_layer = self.model.get_input_embeddings()
        embeddings = embed_layer(input_ids)

        # Compute and add offset
        offset = self.injector(z_feel)
        injected_embeddings = embeddings + offset.unsqueeze(0).unsqueeze(0)

        # Forward with injected embeddings and label smoothing
        outputs = self.model(
            inputs_embeds=injected_embeddings,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Apply label smoothing if configured
        if self.config.label_smoothing > 0:
            loss = self._label_smoothed_loss(outputs.logits, labels)
        else:
            loss = outputs.loss

        return loss, outputs

    def _label_smoothed_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy with label smoothing."""
        vocab_size = logits.size(-1)
        smoothing = self.config.label_smoothing

        # Flatten
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Create smoothed distribution
        with torch.no_grad():
            smooth_labels = torch.zeros_like(logits_flat)
            smooth_labels.fill_(smoothing / (vocab_size - 1))

            # Mask out padding
            mask = labels_flat != -100
            valid_labels = labels_flat.clone()
            valid_labels[~mask] = 0

            smooth_labels.scatter_(1, valid_labels.unsqueeze(1), 1.0 - smoothing)

        # Compute loss
        log_probs = F.log_softmax(logits_flat, dim=-1)
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)

        # Mask padding
        loss = loss * mask.float()
        return loss.sum() / mask.sum()

    def prepare_example(self, example: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a single example for training."""
        prompt = example["prompt"]
        target = example["target_response"]

        full_text = f"<|user|>\n{prompt}\n<|assistant|>\n{target}"

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels

    def train_step(
        self,
        examples: List[Dict],
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, Dict]:
        """Training step over batch of examples."""
        total_loss = 0.0
        action_token_hits = {action.name: 0 for action in FeelAction}
        valid_samples = 0

        for example in examples:
            input_ids, attention_mask, labels = self.prepare_example(example)
            z_feel = example["z_feel"]

            try:
                loss, outputs = self.forward_with_injection(
                    input_ids, attention_mask, labels, z_feel
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                total_loss += loss.item()
                valid_samples += 1

                # Track action token presence in target
                action_token_hits[example["condition"]] = action_token_hits.get(example["condition"], 0) + 1

            except RuntimeError as e:
                print(f"  Warning: {e}")
                continue

        if valid_samples == 0:
            return float('nan'), {}

        # Gradient clipping
        all_params = list(self.model.parameters()) + list(self.injector.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.config.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        metrics = {
            "loss": total_loss / valid_samples,
            "valid_samples": valid_samples,
            "action_distribution": action_token_hits,
        }

        return metrics["loss"], metrics

    def train_phase(
        self,
        phase: Phase,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        examples_per_epoch: int = 200,
    ) -> List[Dict]:
        """Train a single curriculum phase."""
        phase_name = phase.name
        print(f"\n{'='*60}")
        print(f"  PHASE {phase.value}: {phase_name}")
        print(f"  Epochs: {num_epochs}, Examples/epoch: {examples_per_epoch}")
        print(f"{'='*60}")

        phase_metrics = []

        for epoch in range(num_epochs):
            # Calculate fade factor for Phase 2
            fade_factor = 1.0
            if phase == Phase.IMPLICIT:
                fade_start = self.config.telemetry_fade_start
                fade_end = self.config.telemetry_fade_end
                if epoch >= fade_start:
                    fade_factor = max(0, 1.0 - (epoch - fade_start) / (fade_end - fade_start))

            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(range(0, examples_per_epoch, self.config.batch_size),
                       desc=f"Epoch {epoch+1}/{num_epochs}")

            for _ in pbar:
                # Generate batch
                batch = self.data_generator.generate_batch(
                    phase=phase,
                    batch_size=self.config.batch_size,
                    device=self.device,
                    dtype=self.dtype,
                    fade_factor=fade_factor,
                )

                loss, metrics = self.train_step(batch, optimizer)

                if not torch.isnan(torch.tensor(loss)):
                    epoch_loss += loss
                    n_batches += 1
                    pbar.set_postfix({"loss": f"{loss:.4f}", "fade": f"{fade_factor:.2f}"})

            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

                phase_metrics.append({
                    "phase": phase.value,
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "fade_factor": fade_factor,
                })

        return phase_metrics

    def train_full_curriculum(self, output_dir: str, start_phase: int = 1):
        """Run full three-phase curriculum training."""
        print(f"\n{'='*70}")
        print("  FEEL v4.4 CURRICULUM TRAINING")
        print("  Three-Phase Schedule: Guided → Implicit → Action")
        print(f"  injection_scale={self.config.injection_scale}, lr={self.config.learning_rate}")
        print(f"{'='*70}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Optimizer includes injector params
        all_params = list(self.model.parameters()) + list(self.injector.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=self.config.learning_rate)

        all_metrics = []

        # Phase 1: Guided Self-Report
        if start_phase <= 1:
            metrics = self.train_phase(Phase.GUIDED, self.config.phase1_epochs, optimizer)
            all_metrics.extend(metrics)
            self._save_checkpoint(output_path / "phase1", "Phase 1 complete")

        # Phase 2: Implicit Expression
        if start_phase <= 2:
            metrics = self.train_phase(Phase.IMPLICIT, self.config.phase2_epochs, optimizer)
            all_metrics.extend(metrics)
            self._save_checkpoint(output_path / "phase2", "Phase 2 complete")

        # Phase 3: Action Integration
        if start_phase <= 3:
            metrics = self.train_phase(Phase.ACTION, self.config.phase3_epochs, optimizer)
            all_metrics.extend(metrics)

        # Final save
        self._save_checkpoint(output_path, "Full curriculum complete")
        self._save_metrics(output_path, all_metrics)

        print(f"\n{'='*60}")
        print("  CURRICULUM TRAINING COMPLETE")
        print(f"  Model saved to: {output_path}")
        print(f"{'='*60}")

    def _save_checkpoint(self, output_path: Path, description: str):
        """Save model and injector checkpoint."""
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        # Save injector
        torch.save({
            "injector_state_dict": self.injector.state_dict(),
            "config": {
                "z_dim": self.config.z_feel_dim,
                "scale": self.config.injection_scale,
            }
        }, output_path / "additive_injector.pt")

        # Save curriculum config
        with open(output_path / "curriculum_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        print(f"\n  Checkpoint: {description}")
        print(f"  Saved to: {output_path}")

    def _save_metrics(self, output_path: Path, metrics: List[Dict]):
        """Save training metrics."""
        metrics_path = output_path / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": asdict(self.config),
                "metrics": metrics,
            }, f, indent=2)
        print(f"  Metrics saved to: {metrics_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FEEL v4.4 Curriculum Training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                       help="Base model or checkpoint to continue from")
    parser.add_argument("--output", default="models/feel_v4_4",
                       help="Output directory")
    parser.add_argument("--phase", default="all",
                       help="Phase to train: 1, 2, 3, or 'all'")
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=10)
    parser.add_argument("--phase3-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scale", type=float, default=0.1,
                       help="z_feel injection scale")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-lora", action="store_true")

    args = parser.parse_args()

    config = CurriculumConfig(
        model_name=args.model,
        use_lora=not args.no_lora,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        injection_scale=args.scale,
        dtype=args.dtype,
    )

    trainer = CurriculumTrainer(config)

    start_phase = 1 if args.phase == "all" else int(args.phase)
    trainer.train_full_curriculum(args.output, start_phase=start_phase)


if __name__ == "__main__":
    main()
