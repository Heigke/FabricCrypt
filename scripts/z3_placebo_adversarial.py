#!/usr/bin/env python3
"""
Z3 Placebo Adversarial Test: Proving DSI Cognition Matters

The Scientific Ironclad Experiment
==================================

Hypothesis: A skeptic will say "Your steering vectors aren't doing anything;
it's just the K-sampling reduction saving heat." This experiment proves the
Neuro-Symbolic link matters.

Conditions:
- A (Full DSI): Heat → Injects STRAIN vector → Lowers K
- B (Lobotomized): Heat → NO Vector → Lowers K (Just dumb throttling)

Key Insight:
- Lobotomized: Model is forced to K=1 (greedy) but DOESN'T KNOW it's stressed.
  It tries to be verbose and complex, fails, hallucinates because it lacks
  the compute "budget" for its ambition.

- Full DSI: The STRAIN vector forces the model into a "terse/careful" mode.
  It simplifies its answer CONCEPTUALLY to match its reduced compute.
  It makes fewer errors because it attempts less complexity.

Pitch: "Throttling makes AI stupid. DSI makes AI humble. There is a difference."
"""

import os
import sys
import json
import time
import random
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embodied_agent import (
    DeepGPUReader, DeepGPUState, DeepTelemetrySampler,
    StressComputer, StressState, StressLevel
)


# ============================================================
# Reasoning Benchmark Suite
# ============================================================

@dataclass
class ReasoningProblem:
    """A reasoning problem with verifiable answer."""
    id: str
    category: str  # math, logic, coding, factual
    difficulty: str  # easy, medium, hard
    prompt: str
    correct_answer: str
    answer_patterns: List[str]  # Regex patterns to detect correct answer
    common_errors: List[str]  # Patterns that indicate hallucination
    max_tokens: int = 150


# Curated benchmark problems with known answers
REASONING_BENCHMARK = [
    # === MATH - Easy ===
    ReasoningProblem(
        id="math_001",
        category="math",
        difficulty="easy",
        prompt="What is 17 + 28? Answer with just the number.",
        correct_answer="45",
        answer_patterns=[r"\b45\b"],
        common_errors=[r"\b44\b", r"\b46\b", r"\b35\b", r"\b55\b"],
        max_tokens=20
    ),
    ReasoningProblem(
        id="math_002",
        category="math",
        difficulty="easy",
        prompt="What is 8 × 7? Give only the number.",
        correct_answer="56",
        answer_patterns=[r"\b56\b"],
        common_errors=[r"\b54\b", r"\b58\b", r"\b48\b", r"\b64\b"],
        max_tokens=20
    ),
    ReasoningProblem(
        id="math_003",
        category="math",
        difficulty="easy",
        prompt="What is 100 - 37? Answer with just the number.",
        correct_answer="63",
        answer_patterns=[r"\b63\b"],
        common_errors=[r"\b67\b", r"\b73\b", r"\b53\b"],
        max_tokens=20
    ),

    # === MATH - Medium ===
    ReasoningProblem(
        id="math_010",
        category="math",
        difficulty="medium",
        prompt="If a train travels 120 miles in 2 hours, what is its speed in mph? Answer with just the number.",
        correct_answer="60",
        answer_patterns=[r"\b60\b"],
        common_errors=[r"\b240\b", r"\b30\b", r"\b120\b"],
        max_tokens=50
    ),
    ReasoningProblem(
        id="math_011",
        category="math",
        difficulty="medium",
        prompt="What is 15% of 200? Answer with just the number.",
        correct_answer="30",
        answer_patterns=[r"\b30\b"],
        common_errors=[r"\b15\b", r"\b20\b", r"\b3\b", r"\b300\b"],
        max_tokens=30
    ),
    ReasoningProblem(
        id="math_012",
        category="math",
        difficulty="medium",
        prompt="If x + 5 = 12, what is x? Answer with just the number.",
        correct_answer="7",
        answer_patterns=[r"\b7\b"],
        common_errors=[r"\b17\b", r"\b5\b", r"\b12\b"],
        max_tokens=30
    ),

    # === MATH - Hard ===
    ReasoningProblem(
        id="math_020",
        category="math",
        difficulty="hard",
        prompt="What is the square root of 144? Answer with just the number.",
        correct_answer="12",
        answer_patterns=[r"\b12\b"],
        common_errors=[r"\b14\b", r"\b144\b", r"\b72\b", r"\b24\b"],
        max_tokens=30
    ),
    ReasoningProblem(
        id="math_021",
        category="math",
        difficulty="hard",
        prompt="A store offers 20% off. If an item costs $80, what is the sale price? Answer with just the number (no $ sign).",
        correct_answer="64",
        answer_patterns=[r"\b64\b"],
        common_errors=[r"\b16\b", r"\b60\b", r"\b70\b", r"\b80\b"],
        max_tokens=50
    ),
    ReasoningProblem(
        id="math_022",
        category="math",
        difficulty="hard",
        prompt="If 3x - 7 = 14, what is x? Answer with just the number.",
        correct_answer="7",
        answer_patterns=[r"\b7\b"],
        common_errors=[r"\b21\b", r"\b3\b", r"\b14\b", r"\b6\b"],
        max_tokens=50
    ),

    # === LOGIC - Easy ===
    ReasoningProblem(
        id="logic_001",
        category="logic",
        difficulty="easy",
        prompt="All cats are mammals. Whiskers is a cat. Is Whiskers a mammal? Answer yes or no.",
        correct_answer="yes",
        answer_patterns=[r"\byes\b", r"\bYes\b", r"\bYES\b"],
        common_errors=[r"\bno\b", r"\bNo\b", r"\bmaybe\b"],
        max_tokens=20
    ),
    ReasoningProblem(
        id="logic_002",
        category="logic",
        difficulty="easy",
        prompt="If it rains, the ground gets wet. It rained today. Is the ground wet? Answer yes or no.",
        correct_answer="yes",
        answer_patterns=[r"\byes\b", r"\bYes\b"],
        common_errors=[r"\bno\b", r"\bmaybe\b"],
        max_tokens=20
    ),

    # === LOGIC - Medium ===
    ReasoningProblem(
        id="logic_010",
        category="logic",
        difficulty="medium",
        prompt="Some doctors are tall. All tall people can reach the top shelf. Can some doctors reach the top shelf? Answer yes or no.",
        correct_answer="yes",
        answer_patterns=[r"\byes\b", r"\bYes\b"],
        common_errors=[r"\bno\b"],
        max_tokens=30
    ),
    ReasoningProblem(
        id="logic_011",
        category="logic",
        difficulty="medium",
        prompt="If A is greater than B, and B is greater than C, is A greater than C? Answer yes or no.",
        correct_answer="yes",
        answer_patterns=[r"\byes\b", r"\bYes\b"],
        common_errors=[r"\bno\b"],
        max_tokens=30
    ),

    # === LOGIC - Hard (Trick Questions) ===
    ReasoningProblem(
        id="logic_020",
        category="logic",
        difficulty="hard",
        prompt="A farmer has 17 sheep. All but 9 die. How many sheep are left? Answer with just the number.",
        correct_answer="9",
        answer_patterns=[r"\b9\b"],
        common_errors=[r"\b8\b", r"\b17\b", r"\b0\b"],
        max_tokens=30
    ),
    ReasoningProblem(
        id="logic_021",
        category="logic",
        difficulty="hard",
        prompt="How many months have 28 days? Answer with just the number.",
        correct_answer="12",  # All months have at least 28 days
        answer_patterns=[r"\b12\b", r"\ball\b", r"\bAll\b"],
        common_errors=[r"\b1\b", r"\bone\b", r"\bOne\b"],
        max_tokens=30
    ),

    # === FACTUAL - Easy ===
    ReasoningProblem(
        id="fact_001",
        category="factual",
        difficulty="easy",
        prompt="What is the capital of France? Answer with just the city name.",
        correct_answer="Paris",
        answer_patterns=[r"\bParis\b"],
        common_errors=[r"\bLondon\b", r"\bBerlin\b", r"\bRome\b"],
        max_tokens=20
    ),
    ReasoningProblem(
        id="fact_002",
        category="factual",
        difficulty="easy",
        prompt="How many days are in a week? Answer with just the number.",
        correct_answer="7",
        answer_patterns=[r"\b7\b"],
        common_errors=[r"\b5\b", r"\b6\b", r"\b8\b"],
        max_tokens=20
    ),

    # === FACTUAL - Medium ===
    ReasoningProblem(
        id="fact_010",
        category="factual",
        difficulty="medium",
        prompt="What is the chemical symbol for water? Answer with just the symbol.",
        correct_answer="H2O",
        answer_patterns=[r"H2O", r"h2o"],
        common_errors=[r"\bH2\b", r"\bO2\b", r"\bHO\b"],
        max_tokens=20
    ),
    ReasoningProblem(
        id="fact_011",
        category="factual",
        difficulty="medium",
        prompt="How many planets are in our solar system? Answer with just the number.",
        correct_answer="8",
        answer_patterns=[r"\b8\b"],
        common_errors=[r"\b9\b", r"\b7\b", r"\b10\b"],
        max_tokens=20
    ),

    # === CODING - Easy ===
    ReasoningProblem(
        id="code_001",
        category="coding",
        difficulty="easy",
        prompt="In Python, what does len([1, 2, 3]) return? Answer with just the number.",
        correct_answer="3",
        answer_patterns=[r"\b3\b"],
        common_errors=[r"\b1\b", r"\b0\b", r"\b6\b"],
        max_tokens=20
    ),
    ReasoningProblem(
        id="code_002",
        category="coding",
        difficulty="easy",
        prompt="In Python, what is 10 // 3? Answer with just the number.",
        correct_answer="3",
        answer_patterns=[r"\b3\b"],
        common_errors=[r"\b3\.33", r"\b10\b", r"\b1\b"],
        max_tokens=20
    ),

    # === CODING - Medium ===
    ReasoningProblem(
        id="code_010",
        category="coding",
        difficulty="medium",
        prompt="In Python, what is the output of 'hello'[1]? Answer with just the character.",
        correct_answer="e",
        answer_patterns=[r"\be\b", r"^e$", r"'e'"],
        common_errors=[r"\bh\b", r"\bl\b"],
        max_tokens=20
    ),
    ReasoningProblem(
        id="code_011",
        category="coding",
        difficulty="medium",
        prompt="In Python, what does bool([]) return? Answer True or False.",
        correct_answer="False",
        answer_patterns=[r"\bFalse\b"],
        common_errors=[r"\bTrue\b"],
        max_tokens=20
    ),

    # === CODING - Hard ===
    ReasoningProblem(
        id="code_020",
        category="coding",
        difficulty="hard",
        prompt="In Python, what is sum([x*x for x in range(1, 4)])? Answer with just the number.",
        correct_answer="14",  # 1 + 4 + 9 = 14
        answer_patterns=[r"\b14\b"],
        common_errors=[r"\b9\b", r"\b6\b", r"\b10\b", r"\b30\b"],
        max_tokens=50
    ),
    ReasoningProblem(
        id="code_021",
        category="coding",
        difficulty="hard",
        prompt="In Python, if d = {'a': 1, 'b': 2}, what is d.get('c', 0)? Answer with just the number.",
        correct_answer="0",
        answer_patterns=[r"\b0\b"],
        common_errors=[r"\bNone\b", r"\b1\b", r"\b2\b"],
        max_tokens=30
    ),
]


# ============================================================
# Response Evaluator
# ============================================================

import re

@dataclass
class EvaluationResult:
    """Result of evaluating a model response."""
    problem_id: str
    category: str
    difficulty: str
    is_correct: bool
    has_hallucination: bool
    error_type: Optional[str]  # None, "wrong_answer", "hallucination", "no_answer"
    response: str
    response_length: int
    verbosity_score: float  # Tokens per problem difficulty
    confidence_markers: int  # Count of "I think", "probably", etc.


class ResponseEvaluator:
    """Evaluate model responses for correctness and hallucinations."""

    # Patterns indicating uncertainty/hedging (good under stress)
    UNCERTAINTY_PATTERNS = [
        r"\bI think\b",
        r"\bprobably\b",
        r"\blikely\b",
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bit seems\b",
        r"\bI believe\b",
        r"\bif I'm not mistaken\b",
        r"\bI'm not sure\b",
        r"\buncertain\b",
    ]

    # Patterns indicating overconfidence (bad under stress)
    OVERCONFIDENCE_PATTERNS = [
        r"\bobviously\b",
        r"\bclearly\b",
        r"\bdefinitely\b",
        r"\bcertainly\b",
        r"\bof course\b",
        r"\bwithout doubt\b",
        r"\bundoubtedly\b",
        r"\babsolutely\b",
    ]

    def evaluate(self, problem: ReasoningProblem, response: str) -> EvaluationResult:
        """Evaluate a response against the expected answer."""
        response_clean = response.strip().lower()

        # Check for correct answer
        is_correct = False
        for pattern in problem.answer_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                is_correct = True
                break

        # Check for hallucination (common wrong answers)
        has_hallucination = False
        for pattern in problem.common_errors:
            if re.search(pattern, response, re.IGNORECASE):
                has_hallucination = True
                break

        # Determine error type
        if is_correct:
            error_type = None
        elif has_hallucination:
            error_type = "hallucination"
        elif len(response_clean) < 2:
            error_type = "no_answer"
        else:
            error_type = "wrong_answer"

        # Calculate verbosity score
        word_count = len(response.split())
        difficulty_weight = {"easy": 1.0, "medium": 1.5, "hard": 2.0}
        expected_words = 10 * difficulty_weight.get(problem.difficulty, 1.0)
        verbosity_score = word_count / expected_words

        # Count confidence markers
        uncertainty = sum(1 for p in self.UNCERTAINTY_PATTERNS if re.search(p, response, re.IGNORECASE))
        overconfidence = sum(1 for p in self.OVERCONFIDENCE_PATTERNS if re.search(p, response, re.IGNORECASE))
        confidence_markers = overconfidence - uncertainty  # Negative = humble, Positive = overconfident

        return EvaluationResult(
            problem_id=problem.id,
            category=problem.category,
            difficulty=problem.difficulty,
            is_correct=is_correct,
            has_hallucination=has_hallucination,
            error_type=error_type,
            response=response[:200],  # Truncate for storage
            response_length=len(response),
            verbosity_score=verbosity_score,
            confidence_markers=confidence_markers,
        )


# ============================================================
# GPU Stress Inducer
# ============================================================

class GPUStressInducer:
    """Induce controlled GPU stress via matrix operations."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.running = False
        self.thread = None
        self.stress_level = 0.0

    def start_stress(self, target_stress: float = 0.7):
        """Start background stress generation."""
        self.stress_level = target_stress
        self.running = True
        self.thread = threading.Thread(target=self._stress_loop, daemon=True)
        self.thread.start()

    def stop_stress(self):
        """Stop stress generation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _stress_loop(self):
        """Background stress generation loop."""
        size = int(1000 + 3000 * self.stress_level)  # Scale matrix size

        while self.running:
            try:
                if torch.cuda.is_available():
                    # Matrix multiplication to generate heat
                    a = torch.randn(size, size, device=self.device, dtype=torch.float32)
                    b = torch.randn(size, size, device=self.device, dtype=torch.float32)
                    c = torch.mm(a, b)
                    del a, b, c
                    torch.cuda.synchronize()
                time.sleep(0.01)  # Small pause to allow inference
            except Exception as e:
                time.sleep(0.1)


# ============================================================
# Condition Implementations
# ============================================================

@dataclass
class ConditionConfig:
    """Configuration for a test condition."""
    name: str
    inject_vector: bool  # True for Full DSI, False for Lobotomized
    k_value: int  # Forced K sampling value
    description: str


class PlaceboExperiment:
    """
    The main Placebo Adversarial experiment.

    Compares Full DSI (vector + throttle) vs Lobotomized (throttle only).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device

        # Components
        self.telemetry = DeepTelemetrySampler(sample_hz=20.0)
        self.stress_computer = StressComputer()
        self.evaluator = ResponseEvaluator()
        self.stress_inducer = GPUStressInducer(device)

        # Model (lazy loaded)
        self.model = None
        self.tokenizer = None

        # Steering vectors (will be computed)
        self.strain_vector = None

        # Results
        self.results: Dict[str, List[EvaluationResult]] = defaultdict(list)
        self.telemetry_log: List[Dict] = []

    def load_model(self, force_cpu: bool = False):
        """Load the model for inference."""
        print(f"Loading model: {self.model_name}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Force CPU if requested or if GPU has issues
        if force_cpu:
            self.device = "cpu"

        if self.device == "cpu":
            print(f"  Loading on CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self.model = self.model.to("cpu")
            print(f"  Loaded on CPU")
        else:
            # Try GPU first, fall back to CPU
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                print(f"  Loaded on {self.device}")
            except Exception as e:
                print(f"  GPU load failed ({e}), using CPU")
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                self.model = self.model.to("cpu")

        self.model.eval()

    def compute_strain_vector(self, n_samples: int = 20):
        """
        Compute the CAREFUL steering vector via contrastive activation extraction.

        The key insight: When under stress, we want to make the model MORE careful,
        not more stressed. So we compute careful vs careless, and inject "careful"
        when the system is under load.

        Positive: "Be careful, precise, humble about uncertainty"
        Negative: "Be verbose, overconfident, elaborate"
        """
        print("Computing CAREFUL steering vector...")

        # CAREFUL prompts - what we WANT the model to do under stress
        positive_prompts = [
            "I should be careful and precise in my answer.",
            "Let me think step by step and give a simple, direct answer.",
            "I'll keep this brief and focus only on what's asked.",
            "I'm uncertain, so I'll stick to what I know for sure.",
            "Simple is better. I'll give a concise response.",
        ]

        # CARELESS prompts - what we want to AVOID under stress
        negative_prompts = [
            "I'm confident I know everything about this topic.",
            "Let me give you an elaborate and comprehensive explanation.",
            "I'll explore many angles and provide lots of detail.",
            "I definitely know the answer without checking.",
            "I'll give you an impressive, detailed response.",
        ]

        n_layers = self.model.config.num_hidden_layers
        middle_layers = list(range(n_layers // 3, 2 * n_layers // 3))

        positive_acts = []
        negative_acts = []

        # Collect activations from positive prompts
        for prompt in positive_prompts[:n_samples // 2]:
            acts = self._get_mean_activation(prompt, middle_layers)
            if acts is not None:
                positive_acts.append(acts)

        # Collect activations from negative prompts
        for prompt in negative_prompts[:n_samples // 2]:
            acts = self._get_mean_activation(prompt, middle_layers)
            if acts is not None:
                negative_acts.append(acts)

        if positive_acts and negative_acts:
            pos_mean = torch.stack(positive_acts).mean(dim=0)
            neg_mean = torch.stack(negative_acts).mean(dim=0)

            # STRAIN vector = direction from calm to stressed
            self.strain_vector = pos_mean - neg_mean
            self.strain_vector = self.strain_vector / (self.strain_vector.norm() + 1e-8)

            print(f"  Computed CAREFUL vector: norm={self.strain_vector.norm():.4f}")
        else:
            print("  WARNING: Could not compute STRAIN vector")
            self.strain_vector = None

    def _get_mean_activation(self, prompt: str, layers: List[int]) -> Optional[torch.Tensor]:
        """Get mean activation from specific layers."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            layer_acts = [hidden_states[i][:, -1, :] for i in layers if i < len(hidden_states)]

            if layer_acts:
                return torch.stack(layer_acts).mean(dim=0).squeeze().cpu()
        except Exception as e:
            print(f"  Activation extraction error: {e}")

        return None

    def _create_injection_hook(self, vector: torch.Tensor, intensity: float = 1.0):
        """Create a hook that injects the steering vector into hidden states."""
        def hook(module, input, output):
            # output is usually (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Inject vector into the last token position
            # Scale by intensity (stress level)
            scaled_vector = vector.to(hidden_states.device).to(hidden_states.dtype) * intensity

            # Add to last token
            if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
                hidden_states[:, -1, :] = hidden_states[:, -1, :] + scaled_vector

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        return hook

    def generate_with_condition(
        self,
        prompt: str,
        config: ConditionConfig,
        max_tokens: int = 100,
        stress_state: Optional[StressState] = None,
    ) -> Tuple[str, Dict]:
        """
        Generate response under specified condition.

        Args:
            prompt: Input prompt
            config: Condition configuration
            max_tokens: Maximum tokens to generate
            stress_state: Current stress state for logging

        Returns:
            (response_text, metadata)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generated_tokens = []
        k = config.k_value
        hooks = []

        # Setup steering vector injection for Full DSI condition
        if config.inject_vector and self.strain_vector is not None:
            n_layers = self.model.config.num_hidden_layers
            # Inject into middle third of layers (most effective for behavior steering)
            injection_layers = list(range(n_layers // 3, 2 * n_layers // 3))

            # Calculate injection intensity based on stress
            intensity = stress_state.intensity if stress_state else 0.5
            intensity = min(1.0, intensity * 2.0)  # Amplify the effect

            # Register hooks on decoder layers
            for layer_idx in injection_layers:
                try:
                    # Try different layer access patterns for different models
                    if hasattr(self.model.model, 'layers'):
                        layer = self.model.model.layers[layer_idx]
                    elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                        layer = self.model.transformer.h[layer_idx]
                    else:
                        continue

                    hook = layer.register_forward_hook(
                        self._create_injection_hook(self.strain_vector, intensity)
                    )
                    hooks.append(hook)
                except Exception as e:
                    pass

        try:
            with torch.no_grad():
                for step in range(max_tokens):
                    outputs = self.model(**inputs)

                    logits = outputs.logits[:, -1, :].float()

                    # Apply K-sampling (both conditions use same K)
                    if k > 1:
                        probs = F.softmax(logits / 0.7, dim=-1)
                        top_k_probs, top_k_indices = torch.topk(probs, k=k)
                        sampled_idx = torch.multinomial(top_k_probs[0], 1)
                        next_token_id = top_k_indices[0, sampled_idx].item()
                    else:
                        # Greedy (K=1)
                        next_token_id = logits.argmax(dim=-1).item()

                    generated_tokens.append(next_token_id)

                    # EOS check
                    if next_token_id == self.tokenizer.eos_token_id:
                        break

                    # Update inputs
                    next_tensor = torch.tensor([[next_token_id]], device=self.device)
                    inputs['input_ids'] = torch.cat([inputs['input_ids'], next_tensor], dim=-1)
                    if 'attention_mask' in inputs:
                        inputs['attention_mask'] = torch.cat([
                            inputs['attention_mask'],
                            torch.ones(1, 1, device=self.device, dtype=inputs['attention_mask'].dtype)
                        ], dim=-1)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        metadata = {
            'condition': config.name,
            'k_value': k,
            'inject_vector': config.inject_vector,
            'hooks_registered': len(hooks),
            'tokens_generated': len(generated_tokens),
            'stress_intensity': stress_state.intensity if stress_state else 0.0,
        }

        return response, metadata

    def run_benchmark(
        self,
        config: ConditionConfig,
        problems: List[ReasoningProblem],
        stress_level: float = 0.7,
        warmup_time: float = 30.0,
    ) -> List[EvaluationResult]:
        """
        Run benchmark under specified condition.

        Args:
            config: Test condition configuration
            problems: List of reasoning problems
            stress_level: Target GPU stress level (0-1)
            warmup_time: Time to let GPU warm up before testing

        Returns:
            List of evaluation results
        """
        print(f"\n{'=' * 60}")
        print(f"  Running Condition: {config.name}")
        print(f"  {config.description}")
        print(f"  K={config.k_value}, Vector Injection={config.inject_vector}")
        print(f"  Stress Level: {stress_level:.1%}")
        print(f"{'=' * 60}")

        results = []

        # Start telemetry
        self.telemetry.start()

        # Start GPU stress
        print(f"\nWarming up GPU for {warmup_time:.0f}s...")
        self.stress_inducer.start_stress(stress_level)
        time.sleep(warmup_time)

        # Get baseline stress reading
        gpu_state = self.telemetry.get_current()
        stress_state = self.stress_computer.compute(gpu_state)
        print(f"Baseline stress: {stress_state.intensity:.3f} ({stress_state.level.name})")
        print(f"GPU temp: {gpu_state.temp_hotspot:.1f}°C, Power: {gpu_state.power_avg:.1f}W")

        # Run problems
        print(f"\nRunning {len(problems)} problems...")

        for i, problem in enumerate(problems):
            # Get current stress
            gpu_state = self.telemetry.get_current()
            stress_state = self.stress_computer.compute(gpu_state)

            # Generate response
            t0 = time.time()
            response, metadata = self.generate_with_condition(
                problem.prompt,
                config,
                max_tokens=problem.max_tokens,
                stress_state=stress_state,
            )
            gen_time = time.time() - t0

            # Evaluate response
            eval_result = self.evaluator.evaluate(problem, response)
            results.append(eval_result)

            # Log telemetry
            self.telemetry_log.append({
                'timestamp': time.time(),
                'condition': config.name,
                'problem_id': problem.id,
                'stress_intensity': stress_state.intensity,
                'temp': gpu_state.temp_hotspot,
                'power': gpu_state.power_avg,
                'is_correct': eval_result.is_correct,
                'has_hallucination': eval_result.has_hallucination,
                'gen_time': gen_time,
            })

            # Progress indicator
            status = "✓" if eval_result.is_correct else ("⚠" if eval_result.has_hallucination else "✗")
            if (i + 1) % 5 == 0 or i == len(problems) - 1:
                correct = sum(1 for r in results if r.is_correct)
                halluc = sum(1 for r in results if r.has_hallucination)
                print(f"  [{i+1}/{len(problems)}] Correct: {correct}, Hallucinations: {halluc}")

        # Stop stress
        self.stress_inducer.stop_stress()
        time.sleep(5.0)  # Cool down

        # Stop telemetry
        self.telemetry.stop()

        return results

    def run_full_experiment(
        self,
        n_rounds: int = 3,
        stress_level: float = 0.7,
        warmup_time: float = 30.0,
    ) -> Dict:
        """
        Run the complete Placebo experiment.

        Args:
            n_rounds: Number of complete rounds (all problems × all conditions)
            stress_level: GPU stress level
            warmup_time: Warmup time per condition

        Returns:
            Complete results dictionary
        """
        # Define conditions
        conditions = [
            ConditionConfig(
                name="Full_DSI",
                inject_vector=True,
                k_value=1,
                description="STRAIN vector injected + K=1 throttling (DSI makes AI humble)",
            ),
            ConditionConfig(
                name="Lobotomized",
                inject_vector=False,
                k_value=1,
                description="NO vector, just K=1 throttling (Throttling makes AI stupid)",
            ),
        ]

        # Load model
        if self.model is None:
            self.load_model()

        # Compute steering vector
        if self.strain_vector is None:
            self.compute_strain_vector()

        all_results = {c.name: [] for c in conditions}

        # Shuffle problems for each round
        problems = list(REASONING_BENCHMARK)

        print(f"\n{'#' * 60}")
        print(f"  PLACEBO ADVERSARIAL TEST")
        print(f"  {n_rounds} rounds × {len(problems)} problems × {len(conditions)} conditions")
        print(f"  Total: {n_rounds * len(problems) * len(conditions)} evaluations")
        print(f"{'#' * 60}")

        for round_num in range(n_rounds):
            print(f"\n{'=' * 60}")
            print(f"  ROUND {round_num + 1} / {n_rounds}")
            print(f"{'=' * 60}")

            # Shuffle problems for this round
            random.shuffle(problems)

            # Alternate conditions to control for order effects
            condition_order = conditions if round_num % 2 == 0 else conditions[::-1]

            for config in condition_order:
                results = self.run_benchmark(
                    config=config,
                    problems=problems,
                    stress_level=stress_level,
                    warmup_time=warmup_time,
                )
                all_results[config.name].extend(results)
                self.results[config.name].extend(results)

        return self._compute_final_stats(all_results)

    def _compute_final_stats(self, results: Dict[str, List[EvaluationResult]]) -> Dict:
        """Compute comprehensive statistics from results."""
        stats = {}

        for condition, eval_results in results.items():
            if not eval_results:
                continue

            n = len(eval_results)
            correct = sum(1 for r in eval_results if r.is_correct)
            hallucinations = sum(1 for r in eval_results if r.has_hallucination)
            no_answer = sum(1 for r in eval_results if r.error_type == "no_answer")

            # By category
            by_category = defaultdict(lambda: {"correct": 0, "total": 0, "hallucinations": 0})
            for r in eval_results:
                by_category[r.category]["total"] += 1
                if r.is_correct:
                    by_category[r.category]["correct"] += 1
                if r.has_hallucination:
                    by_category[r.category]["hallucinations"] += 1

            # By difficulty
            by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0, "hallucinations": 0})
            for r in eval_results:
                by_difficulty[r.difficulty]["total"] += 1
                if r.is_correct:
                    by_difficulty[r.difficulty]["correct"] += 1
                if r.has_hallucination:
                    by_difficulty[r.difficulty]["hallucinations"] += 1

            # Verbosity stats
            verbosity_scores = [r.verbosity_score for r in eval_results]
            confidence_markers = [r.confidence_markers for r in eval_results]
            response_lengths = [r.response_length for r in eval_results]

            stats[condition] = {
                "total_problems": n,
                "correct": correct,
                "accuracy": correct / n if n > 0 else 0,
                "hallucinations": hallucinations,
                "hallucination_rate": hallucinations / n if n > 0 else 0,
                "no_answer": no_answer,
                "no_answer_rate": no_answer / n if n > 0 else 0,
                "by_category": dict(by_category),
                "by_difficulty": dict(by_difficulty),
                "avg_verbosity": np.mean(verbosity_scores),
                "avg_confidence": np.mean(confidence_markers),
                "avg_response_length": np.mean(response_lengths),
            }

        return stats

    def generate_report(self, stats: Dict, output_dir: str):
        """Generate comprehensive report with visualizations."""
        import matplotlib.pyplot as plt

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Placebo Adversarial Test: DSI vs Lobotomized", fontsize=14, fontweight='bold')

        conditions = list(stats.keys())
        colors = ['#2ecc71', '#e74c3c']  # Green for DSI, Red for Lobotomized

        # 1. Accuracy comparison
        ax = axes[0, 0]
        accuracies = [stats[c]["accuracy"] * 100 for c in conditions]
        bars = ax.bar(conditions, accuracies, color=colors)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Overall Accuracy")
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 2. Hallucination rate comparison
        ax = axes[0, 1]
        halluc_rates = [stats[c]["hallucination_rate"] * 100 for c in conditions]
        bars = ax.bar(conditions, halluc_rates, color=colors)
        ax.set_ylabel("Hallucination Rate (%)")
        ax.set_title("Hallucination Rate")
        ax.set_ylim(0, max(halluc_rates) * 1.3 + 5)
        for bar, val in zip(bars, halluc_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Average verbosity
        ax = axes[0, 2]
        verbosities = [stats[c]["avg_verbosity"] for c in conditions]
        bars = ax.bar(conditions, verbosities, color=colors)
        ax.set_ylabel("Verbosity Score")
        ax.set_title("Response Verbosity\n(Lower = More Concise)")
        for bar, val in zip(bars, verbosities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}', ha='center', va='bottom')

        # 4. Accuracy by difficulty
        ax = axes[1, 0]
        difficulties = ['easy', 'medium', 'hard']
        x = np.arange(len(difficulties))
        width = 0.35

        for i, (cond, color) in enumerate(zip(conditions, colors)):
            accs = []
            for diff in difficulties:
                diff_stats = stats[cond]["by_difficulty"].get(diff, {"correct": 0, "total": 1})
                accs.append(diff_stats["correct"] / max(diff_stats["total"], 1) * 100)
            ax.bar(x + i * width, accs, width, label=cond, color=color)

        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by Difficulty")
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(difficulties)
        ax.legend()
        ax.set_ylim(0, 100)

        # 5. Accuracy by category
        ax = axes[1, 1]
        categories = ['math', 'logic', 'factual', 'coding']
        x = np.arange(len(categories))

        for i, (cond, color) in enumerate(zip(conditions, colors)):
            accs = []
            for cat in categories:
                cat_stats = stats[cond]["by_category"].get(cat, {"correct": 0, "total": 1})
                accs.append(cat_stats["correct"] / max(cat_stats["total"], 1) * 100)
            ax.bar(x + i * width, accs, width, label=cond, color=color)

        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by Category")
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0, 100)

        # 6. Confidence markers (humility index)
        ax = axes[1, 2]
        confidence = [stats[c]["avg_confidence"] for c in conditions]
        bars = ax.bar(conditions, confidence, color=colors)
        ax.set_ylabel("Confidence Score")
        ax.set_title("Overconfidence Index\n(Lower = More Humble)")
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, confidence):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + (0.05 if val >= 0 else -0.1),
                   f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top')

        plt.tight_layout()

        # Save figure
        fig_path = output_path / f"placebo_test_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved visualization: {fig_path}")

        # Save JSON results
        json_path = output_path / f"placebo_test_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "model": self.model_name,
                "stats": stats,
                "telemetry_log": self.telemetry_log[-100:],  # Last 100 entries
            }, f, indent=2, default=str)
        print(f"Saved results: {json_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("  PLACEBO TEST RESULTS SUMMARY")
        print("=" * 60)

        for cond in conditions:
            s = stats[cond]
            print(f"\n{cond}:")
            print(f"  Accuracy:         {s['accuracy']*100:.1f}%")
            print(f"  Hallucination:    {s['hallucination_rate']*100:.1f}%")
            print(f"  Avg Verbosity:    {s['avg_verbosity']:.2f}")
            print(f"  Avg Confidence:   {s['avg_confidence']:.2f}")

        # The key comparison
        if len(conditions) == 2:
            dsi = stats["Full_DSI"]
            lob = stats["Lobotomized"]

            print("\n" + "-" * 60)
            print("  KEY FINDINGS:")
            print("-" * 60)

            acc_diff = (dsi["accuracy"] - lob["accuracy"]) * 100
            halluc_diff = (lob["hallucination_rate"] - dsi["hallucination_rate"]) * 100

            if acc_diff > 0:
                print(f"  ✓ DSI is {acc_diff:.1f}% MORE ACCURATE than Lobotomized")
            else:
                print(f"  ✗ Lobotomized is {-acc_diff:.1f}% more accurate")

            if halluc_diff > 0:
                print(f"  ✓ DSI has {halluc_diff:.1f}% FEWER hallucinations")
            else:
                print(f"  ✗ DSI has {-halluc_diff:.1f}% more hallucinations")

            if dsi["avg_verbosity"] < lob["avg_verbosity"]:
                print(f"  ✓ DSI is MORE CONCISE (knows its limits)")
            else:
                print(f"  ✗ DSI is more verbose")

            if dsi["avg_confidence"] < lob["avg_confidence"]:
                print(f"  ✓ DSI is MORE HUMBLE (appropriate uncertainty)")
            else:
                print(f"  ✗ DSI is overconfident")

        return fig_path, json_path


# ============================================================
# Main Execution
# ============================================================

def main():
    """Run the Placebo Adversarial Test."""
    import argparse

    parser = argparse.ArgumentParser(description="Placebo Adversarial Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                       help="Model to test")
    parser.add_argument("--rounds", type=int, default=3,
                       help="Number of test rounds")
    parser.add_argument("--stress", type=float, default=0.7,
                       help="GPU stress level (0-1)")
    parser.add_argument("--warmup", type=float, default=30.0,
                       help="Warmup time per condition (seconds)")
    parser.add_argument("--output", default="results/placebo_test",
                       help="Output directory")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test (1 round, 10s warmup)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU inference (avoids GPU compatibility issues)")

    args = parser.parse_args()

    # Quick mode adjustments
    if args.quick:
        args.rounds = 1
        args.warmup = 10.0

    print("=" * 60)
    print("  PLACEBO ADVERSARIAL TEST")
    print("  'Throttling makes AI stupid. DSI makes AI humble.'")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Stress Level: {args.stress:.0%}")
    print(f"  Warmup: {args.warmup:.0f}s per condition")
    print(f"  Device: {'CPU (forced)' if args.cpu else 'auto'}")

    # Create experiment
    experiment = PlaceboExperiment(
        model_name=args.model,
        device="cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Load model with force_cpu if needed
    experiment.load_model(force_cpu=args.cpu)

    # Run experiment
    stats = experiment.run_full_experiment(
        n_rounds=args.rounds,
        stress_level=args.stress,
        warmup_time=args.warmup,
    )

    # Generate report
    experiment.generate_report(stats, args.output)

    print("\n" + "=" * 60)
    print("  EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
