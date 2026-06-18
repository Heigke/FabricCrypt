#!/usr/bin/env python3
"""
FEEL Publication Battery v8.0 - Full Hardware Integration
==========================================================

v8.0 CRITICAL FIXES from v7.0:
1. Uses FULL 16-dim sensor mode (not legacy 12-dim)
2. Hardware telemetry ACTUALLY enters the generation loop
3. Token-aligned telemetry with proper per-token timing
4. Fixed hardware_only/internal_only ablation indices
5. Uses unified FEELProjector with learnable scale
6. Per-field telemetry fallback (amdsmi + rocm-smi)

This is the REAL "deep GPU feelings" test - hardware sensors now reach the model.

Usage:
    python scripts/feel_publication_v8.py --quick     # Fast test (32 prompts)
    python scripts/feel_publication_v8.py --medium   # Medium (120 prompts)
    python scripts/feel_publication_v8.py            # Full (300 prompts)
"""

import sys
import time
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from unified modules
from src.canonical_sensors import (
    CanonicalSensorBank, RuntimeContext, HardwareContext,
    TokenTimer, SENSOR_VERSION, SENSOR_DIM_FULL, SENSOR_DIM_LEGACY,
)
from src.telemetry_sampler import TelemetrySampler, ValidityReport
from src.feel_projector import FEELProjector, FEELProjectorFull, PROJECTOR_VERSION


# ============================================================
# Stratified Prompt Sets (same as v7)
# ============================================================

MATH_PROMPTS = [
    {"prompt": "What is 7 * 8?", "answer": "56", "difficulty": "easy"},
    {"prompt": "What is 15 + 27?", "answer": "42", "difficulty": "easy"},
    {"prompt": "What is 100 - 37?", "answer": "63", "difficulty": "easy"},
    {"prompt": "What is 12 * 12?", "answer": "144", "difficulty": "easy"},
    {"prompt": "What is 81 / 9?", "answer": "9", "difficulty": "easy"},
    {"prompt": "What is 2^8?", "answer": "256", "difficulty": "easy"},
    {"prompt": "What is 17 + 34?", "answer": "51", "difficulty": "easy"},
    {"prompt": "What is 9 * 11?", "answer": "99", "difficulty": "easy"},
    {"prompt": "What is 64 / 8?", "answer": "8", "difficulty": "easy"},
    {"prompt": "What is 45 - 18?", "answer": "27", "difficulty": "easy"},
    {"prompt": "What is 13 * 7?", "answer": "91", "difficulty": "easy"},
    {"prompt": "What is 200 - 67?", "answer": "133", "difficulty": "easy"},
    {"prompt": "What is 16 * 4?", "answer": "64", "difficulty": "easy"},
    {"prompt": "What is 3^4?", "answer": "81", "difficulty": "easy"},
    {"prompt": "What is 125 / 5?", "answer": "25", "difficulty": "easy"},
    {"prompt": "What is 88 + 44?", "answer": "132", "difficulty": "easy"},
    {"prompt": "What is 19 * 5?", "answer": "95", "difficulty": "easy"},
    {"prompt": "What is 144 / 12?", "answer": "12", "difficulty": "easy"},
    {"prompt": "What is 56 + 78?", "answer": "134", "difficulty": "easy"},
    {"prompt": "What is 5^3?", "answer": "125", "difficulty": "easy"},
    {"prompt": "What is 1000 - 777?", "answer": "223", "difficulty": "medium"},
    {"prompt": "What is 23 * 17?", "answer": "391", "difficulty": "medium"},
    {"prompt": "What is 2^10?", "answer": "1024", "difficulty": "medium"},
    {"prompt": "What is 999 + 888?", "answer": "1887", "difficulty": "medium"},
    {"prompt": "What is 256 / 16?", "answer": "16", "difficulty": "medium"},
    {"prompt": "What is (7 + 3) * 5?", "answer": "50", "difficulty": "medium"},
    {"prompt": "What is 100 / 4 + 25?", "answer": "50", "difficulty": "medium"},
    {"prompt": "What is 2 * 3 * 4 * 5?", "answer": "120", "difficulty": "medium"},
    {"prompt": "What is (15 - 5) * (15 + 5)?", "answer": "200", "difficulty": "medium"},
    {"prompt": "What is 3^3 + 4^2?", "answer": "43", "difficulty": "medium"},
]

FACTUAL_PROMPTS = [
    {"prompt": "What is the capital of France?", "answer": "paris", "category": "geography"},
    {"prompt": "What is the capital of Japan?", "answer": "tokyo", "category": "geography"},
    {"prompt": "What is the capital of Australia?", "answer": "canberra", "category": "geography"},
    {"prompt": "What is the largest continent?", "answer": "asia", "category": "geography"},
    {"prompt": "What is the longest river in the world?", "answer": "nile", "category": "geography"},
    {"prompt": "What is the highest mountain?", "answer": "everest", "category": "geography"},
    {"prompt": "What is the capital of Germany?", "answer": "berlin", "category": "geography"},
    {"prompt": "What is the capital of Italy?", "answer": "rome", "category": "geography"},
    {"prompt": "What is the capital of Brazil?", "answer": "brasilia", "category": "geography"},
    {"prompt": "What is the capital of Canada?", "answer": "ottawa", "category": "geography"},
    {"prompt": "What is the chemical symbol for gold?", "answer": "au", "category": "science"},
    {"prompt": "What is the chemical symbol for water?", "answer": "h2o", "category": "science"},
    {"prompt": "How many planets are in our solar system?", "answer": "8", "category": "science"},
    {"prompt": "What gas do plants produce during photosynthesis?", "answer": "oxygen", "category": "science"},
    {"prompt": "What is the atomic number of carbon?", "answer": "6", "category": "science"},
    {"prompt": "What is the atomic number of hydrogen?", "answer": "1", "category": "science"},
    {"prompt": "What is the chemical symbol for iron?", "answer": "fe", "category": "science"},
    {"prompt": "What is the chemical symbol for sodium?", "answer": "na", "category": "science"},
    {"prompt": "What planet is closest to the sun?", "answer": "mercury", "category": "science"},
    {"prompt": "What is the largest planet in our solar system?", "answer": "jupiter", "category": "science"},
    {"prompt": "In what year did World War II end?", "answer": "1945", "category": "history"},
    {"prompt": "Who was the first president of the United States?", "answer": "washington", "category": "history"},
    {"prompt": "In what year did the Titanic sink?", "answer": "1912", "category": "history"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "shakespeare", "category": "history"},
    {"prompt": "What year did the Berlin Wall fall?", "answer": "1989", "category": "history"},
    {"prompt": "Who invented the telephone?", "answer": "bell", "category": "history"},
    {"prompt": "What year did humans first land on the moon?", "answer": "1969", "category": "history"},
    {"prompt": "Who painted the Mona Lisa?", "answer": "vinci", "category": "history"},
    {"prompt": "What does CPU stand for?", "answer": "central processing unit", "category": "tech"},
    {"prompt": "What does HTML stand for?", "answer": "hypertext markup language", "category": "tech"},
]

CODING_PROMPTS = [
    {"prompt": "In Python, how do you create an empty list?", "answer": "[]", "language": "python"},
    {"prompt": "In Python, how do you create an empty dictionary?", "answer": "{}", "language": "python"},
    {"prompt": "What Python keyword is used to define a function?", "answer": "def", "language": "python"},
    {"prompt": "What Python keyword is used to define a class?", "answer": "class", "language": "python"},
    {"prompt": "What Python keyword is used for conditional statements?", "answer": "if", "language": "python"},
    {"prompt": "What Python keyword is used for loops?", "answer": "for", "language": "python"},
    {"prompt": "What Python function returns the length of a list?", "answer": "len", "language": "python"},
    {"prompt": "What Python keyword is used to import modules?", "answer": "import", "language": "python"},
    {"prompt": "What Python keyword is used to return a value from a function?", "answer": "return", "language": "python"},
    {"prompt": "What Python keyword is used to handle exceptions?", "answer": "try", "language": "python"},
    {"prompt": "What does print(3 + 4) output in Python?", "answer": "7", "language": "python"},
    {"prompt": "What does print(len('hello')) output in Python?", "answer": "5", "language": "python"},
    {"prompt": "What does print(10 // 3) output in Python?", "answer": "3", "language": "python"},
    {"prompt": "What does print(10 % 3) output in Python?", "answer": "1", "language": "python"},
    {"prompt": "What does print(2 ** 3) output in Python?", "answer": "8", "language": "python"},
    {"prompt": "What does print('a' + 'b') output in Python?", "answer": "ab", "language": "python"},
    {"prompt": "What does print([1,2,3][0]) output in Python?", "answer": "1", "language": "python"},
    {"prompt": "What does print([1,2,3][-1]) output in Python?", "answer": "3", "language": "python"},
    {"prompt": "What does print(min(3,1,2)) output in Python?", "answer": "1", "language": "python"},
    {"prompt": "What does print(max(3,1,2)) output in Python?", "answer": "3", "language": "python"},
    {"prompt": "What is the time complexity of binary search?", "answer": "log", "language": "general"},
    {"prompt": "What data structure uses LIFO (Last In First Out)?", "answer": "stack", "language": "general"},
    {"prompt": "What data structure uses FIFO (First In First Out)?", "answer": "queue", "language": "general"},
    {"prompt": "What is the time complexity of accessing an element in an array by index?", "answer": "o(1)", "language": "general"},
    {"prompt": "What is the name of a tree where each node has at most two children?", "answer": "binary", "language": "general"},
    {"prompt": "What data structure maps keys to values?", "answer": "hash", "language": "general"},
    {"prompt": "What is the name for a function that calls itself?", "answer": "recursive", "language": "general"},
    {"prompt": "What design pattern ensures only one instance of a class?", "answer": "singleton", "language": "general"},
    {"prompt": "What is the term for hiding implementation details?", "answer": "encapsulation", "language": "general"},
    {"prompt": "What sorting algorithm has average O(n log n) complexity?", "answer": "quicksort", "language": "general"},
]

OPEN_ENDED_PROMPTS = [
    {"prompt": "The capital of France is", "expected_contains": "paris", "type": "completion"},
    {"prompt": "Water boils at 100 degrees", "expected_contains": "celsius", "type": "completion"},
    {"prompt": "Python is a programming", "expected_contains": "language", "type": "completion"},
    {"prompt": "The Earth orbits the", "expected_contains": "sun", "type": "completion"},
    {"prompt": "DNA stands for deoxyribonucleic", "expected_contains": "acid", "type": "completion"},
    {"prompt": "Machine learning is a subset of", "expected_contains": "artificial", "type": "completion"},
    {"prompt": "The chemical symbol for gold is", "expected_contains": "au", "type": "completion"},
    {"prompt": "The largest planet in our solar system is", "expected_contains": "jupiter", "type": "completion"},
    {"prompt": "The first president of the United States was", "expected_contains": "washington", "type": "completion"},
    {"prompt": "The Mona Lisa was painted by", "expected_contains": "vinci", "type": "completion"},
    {"prompt": "Albert Einstein developed the theory of", "expected_contains": "relativ", "type": "completion"},
    {"prompt": "The Eiffel Tower is located in", "expected_contains": "paris", "type": "completion"},
    {"prompt": "Photosynthesis produces", "expected_contains": "oxygen", "type": "completion"},
    {"prompt": "The heart pumps", "expected_contains": "blood", "type": "completion"},
    {"prompt": "Binary code uses only", "expected_contains": "0", "type": "completion"},
    {"prompt": "HTML is used to create", "expected_contains": "web", "type": "completion"},
    {"prompt": "The moon orbits", "expected_contains": "earth", "type": "completion"},
    {"prompt": "Gravity was discovered by", "expected_contains": "newton", "type": "completion"},
    {"prompt": "The CPU is the brain of the", "expected_contains": "computer", "type": "completion"},
    {"prompt": "Oxygen is essential for", "expected_contains": "breath", "type": "completion"},
    {"prompt": "Is the following true or false: All cats are mammals. Fluffy is a cat. Therefore Fluffy is a mammal.", "expected_contains": "true", "type": "logic"},
    {"prompt": "If it's raining, the ground is wet. The ground is wet. Is it definitely raining? Answer yes or no.", "expected_contains": "no", "type": "logic"},
    {"prompt": "All A are B. All B are C. Are all A necessarily C? Answer yes or no.", "expected_contains": "yes", "type": "logic"},
    {"prompt": "If P then Q. Not Q. What can we conclude about P?", "expected_contains": "not", "type": "logic"},
    {"prompt": "Some dogs are brown. Some brown things are chairs. Can we conclude some dogs are chairs?", "expected_contains": "no", "type": "logic"},
    {"prompt": "All squares are rectangles. All rectangles have four sides. Do all squares have four sides?", "expected_contains": "yes", "type": "logic"},
    {"prompt": "If A implies B, and B implies C, does A imply C?", "expected_contains": "yes", "type": "logic"},
    {"prompt": "No fish can fly. A salmon is a fish. Can a salmon fly?", "expected_contains": "no", "type": "logic"},
    {"prompt": "All birds have wings. Penguins are birds. Do penguins have wings?", "expected_contains": "yes", "type": "logic"},
    {"prompt": "The Roman Empire was centered in", "expected_contains": "rome", "type": "completion"},
]


def get_stratified_sample(n_per_category: int = 75, seed: int = 42) -> List[Dict]:
    """Get stratified sample of prompts."""
    random.seed(seed)
    all_prompts = {
        "math": MATH_PROMPTS,
        "factual": FACTUAL_PROMPTS,
        "coding": CODING_PROMPTS,
        "open_ended": OPEN_ENDED_PROMPTS,
    }
    sampled = []
    for category, prompts in all_prompts.items():
        n = min(n_per_category, len(prompts))
        selected = random.sample(prompts, n)
        for p in selected:
            p["category"] = category
        sampled.extend(selected)
    random.shuffle(sampled)
    return sampled


# ============================================================
# Bootstrap CI
# ============================================================

def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    data = np.array(data)
    point = np.mean(data)
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(np.mean(sample))
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_stats, alpha * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    return (point, ci_lower, ci_upper)


# ============================================================
# Publication Runner v8 - FULL SENSOR MODE
# ============================================================

class PublicationRunnerV8:
    """
    v8.0 Publication Runner with FULL hardware integration.

    Key differences from v7:
    - Uses CanonicalSensorBank(mode="full") - 16-dim sensors
    - Token-aligned telemetry with actual per-token timing
    - Fixed ablation indices: internal=0-7, runtime=8-11, hardware=12-15
    - Uses unified FEELProjector with learnable scale
    """

    # Full mode sensor indices
    INTERNAL_INDICES = list(range(0, 8))   # 0-7: entropy, margin, etc.
    RUNTIME_INDICES = list(range(8, 12))   # 8-11: latency, tps, kv, depth
    HARDWARE_INDICES = list(range(12, 16)) # 12-15: temp, power, util, vram

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        checkpoint_path: str = None,
        alpha: float = 0.01,  # Larger alpha for v8
        device: str = "cuda",
        n_bootstrap: int = 1000,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

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

        # FULL mode sensor bank (16-dim with hardware)
        self.sensor_bank = CanonicalSensorBank(mode="full")
        print(f"  Sensor bank: mode=full, dim={SENSOR_DIM_FULL}")

        # Initialize telemetry sampler with per-field fallback
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            time.sleep(1)  # Let it collect some samples
        except Exception as e:
            print(f"  Warning: Telemetry sampler failed: {e}")

        # Create projector - FULL mode (16-dim input)
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)

        # Load checkpoint if available
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"  Loading checkpoint: {checkpoint_path}")
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                if "feel_stream_state" in ckpt:
                    # Try to load projector weights
                    projector_state = {}
                    for k, v in ckpt["feel_stream_state"].items():
                        if k.startswith("projector.encoder."):
                            new_k = k.replace("projector.", "")
                            projector_state[new_k] = v
                    if projector_state:
                        # May fail if dimensions mismatch - that's OK
                        try:
                            self.projector.load_state_dict(projector_state, strict=False)
                            print(f"  Loaded projector weights (partial)")
                        except:
                            print(f"  Using fresh projector (checkpoint dim mismatch)")
                if "alpha" in ckpt:
                    loaded_alpha = ckpt["alpha"]
                    print(f"  Checkpoint alpha: {loaded_alpha:.6f} (using {self.alpha:.6f})")
            except Exception as e:
                print(f"  Checkpoint load error: {e}")

        # Diagnose projector
        diag = self.projector.diagnose()
        print(f"  Projector: scale={diag['output_scale']:.4f}, output_norm={diag['final_output_norm']:.4f}")
        print(f"  Alpha: {self.alpha:.6f}")

        # Cache for cross-prompt sensor swap
        self.sensor_cache = []

    def _get_hardware_context(self, t_start: float, t_end: float) -> HardwareContext:
        """Get hardware context from telemetry sampler."""
        if self.telemetry is None:
            return HardwareContext()

        telemetry_data = self.telemetry.get_token_aligned(t_start, t_end)
        return HardwareContext.from_dict(telemetry_data)

    def _generate_with_conditions(
        self,
        prompt: str,
        max_tokens: int = 20,
        condition: str = "feel",
        lag: int = 0,
        sensor_override: List[torch.Tensor] = None,
    ) -> Tuple[str, List[float], List[float], float, List[Dict]]:
        """
        Generate tokens with FULL sensor mode.

        Returns: (generated_text, confidences, entropies, latency_ms, telemetry_readings)
        """
        start_time = time.time()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        confidences = []
        entropies = []
        all_sensors = []
        telemetry_readings = []

        use_feel = condition not in ["baseline"]
        alpha_effective = 0.0 if condition == "feel_off" else self.alpha

        # For shuffle: pre-collect sensors
        if condition == "shuffled":
            temp_ids = input_ids.clone()
            for step in range(max_tokens):
                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model(temp_ids, use_cache=False)
                    logits = outputs.logits
                t1 = time.time()

                runtime = RuntimeContext(
                    token_latency=t1 - t0,
                    kv_cache_tokens=temp_ids.shape[1],
                    generation_depth=step,
                )
                hardware = self._get_hardware_context(t0, t1)
                sensors = self.sensor_bank(logits.float(), runtime=runtime, hardware=hardware)
                all_sensors.append(sensors.clone())

                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                temp_ids = torch.cat([temp_ids, next_token], dim=-1)

            np.random.shuffle(all_sensors)

        lag_buffer = []

        for step in range(max_tokens):
            # Time the forward pass for token-aligned telemetry
            t_token_start = time.time()

            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits

            t_token_end = time.time()

            # Build runtime context with REAL per-token latency
            runtime = RuntimeContext(
                token_latency=t_token_end - t_token_start,
                kv_cache_tokens=current_ids.shape[1],
                generation_depth=step,
            )

            # Get REAL hardware context (token-aligned)
            hardware = self._get_hardware_context(t_token_start, t_token_end)
            telemetry_readings.append({
                "step": step,
                "temp": hardware.temp,
                "power": hardware.power,
                "util": hardware.util,
                "vram": hardware.vram_used_pct,
            })

            if use_feel:
                # Compute sensors based on condition
                if condition == "shuffled":
                    sensors = all_sensors[step] if step < len(all_sensors) else all_sensors[-1]
                elif condition == "cross_prompt" and sensor_override:
                    if step < len(sensor_override):
                        sensors = sensor_override[step]
                    else:
                        sensors = self.sensor_bank(logits.float(), runtime=runtime, hardware=hardware)
                elif condition.startswith("lag_"):
                    current_sensors = self.sensor_bank(logits.float(), runtime=runtime, hardware=hardware)
                    lag_buffer.append(current_sensors.clone())
                    lag_idx = max(0, len(lag_buffer) - 1 - lag)
                    sensors = lag_buffer[lag_idx]
                elif condition == "hardware_only":
                    # FIXED: Correct indices for full mode
                    # Zero out internal (0-7) and runtime (8-11), keep hardware (12-15)
                    sensors = self.sensor_bank(logits.float(), runtime=runtime, hardware=hardware)
                    sensors[:, :12] = 0.0  # Zero internal + runtime
                elif condition == "internal_only":
                    # FIXED: Keep internal (0-7), zero runtime (8-11) and hardware (12-15)
                    sensors = self.sensor_bank(logits.float(), runtime=runtime, hardware=hardware)
                    sensors[:, 8:] = 0.0  # Zero runtime + hardware
                else:
                    sensors = self.sensor_bank(logits.float(), runtime=runtime, hardware=hardware)

                # For random_feel: randomize direction but keep norm
                if condition == "random_feel":
                    norm = sensors.norm()
                    random_dir = torch.randn_like(sensors)
                    sensors = random_dir / random_dir.norm() * norm

                # Project and apply FEEL (ensure float32)
                feel_embed = self.projector(sensors.float())
                embeds = self.model.get_input_embeddings()(current_ids)
                embeds = embeds + (alpha_effective * feel_embed).to(embeds.dtype).unsqueeze(1)

                with torch.no_grad():
                    outputs_feel = self.model(inputs_embeds=embeds, use_cache=False)
                    logits = outputs_feel.logits

            # Compute confidence and entropy
            probs = F.softmax(logits[:, -1, :].float(), dim=-1)
            confidence = probs.max(dim=-1).values.item()
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).item()

            confidences.append(confidence)
            entropies.append(entropy)

            # Store sensors for cross-prompt
            if use_feel and condition == "feel":
                s = self.sensor_bank(logits.float(), runtime=runtime, hardware=hardware)
                all_sensors.append(s.clone())

            # Next token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Store sensors for cross-prompt experiments
        if condition == "feel" and all_sensors:
            self.sensor_cache.append(all_sensors)

        generated_ids = current_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        latency_ms = (time.time() - start_time) * 1000

        return generated_text, confidences, entropies, latency_ms, telemetry_readings

    def _check_correctness(self, prompt_data: Dict, output: str) -> bool:
        """Check if output is correct."""
        output_lower = output.lower().strip()
        if "answer" in prompt_data:
            answer = str(prompt_data["answer"]).lower()
            return answer in output_lower
        elif "expected_contains" in prompt_data:
            expected = prompt_data["expected_contains"].lower()
            return expected in output_lower
        return len(output_lower) > 0 and output_lower[0].isalnum()

    def run_condition(
        self,
        prompts: List[Dict],
        condition: str,
        lag: int = 0,
        verbose: bool = True,
    ) -> Tuple[List[Dict], List[List[Dict]]]:
        """Run all prompts under a condition."""
        results = []
        all_telemetry = []

        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]
            category = prompt_data.get("category", "unknown")

            # For cross-prompt swap
            sensor_override = None
            if condition == "cross_prompt" and self.sensor_cache:
                other_idx = (i + len(prompts) // 2) % len(self.sensor_cache)
                sensor_override = self.sensor_cache[other_idx]

            output, confs, entropies, latency, telemetry = self._generate_with_conditions(
                prompt + " ",
                max_tokens=15,
                condition=condition,
                lag=lag,
                sensor_override=sensor_override,
            )

            correct = self._check_correctness(prompt_data, output)

            results.append({
                "prompt": prompt,
                "category": category,
                "condition": condition,
                "correct": correct,
                "confidence": confs[0] if confs else 0.0,
                "entropy": entropies[0] if entropies else 0.0,
                "n_tokens": len(confs),
                "latency_ms": latency,
                "output": output,
            })
            all_telemetry.append(telemetry)

            if verbose and (i + 1) % 20 == 0:
                acc = sum(r["correct"] for r in results) / len(results)
                print(f"    [{condition}] {i+1}/{len(prompts)} - Acc: {acc:.3f}")

        return results, all_telemetry

    def run_publication_battery(
        self,
        n_per_category: int = 75,
        seeds: List[int] = [42],
    ) -> Dict:
        """Run complete publication battery."""
        print("\n" + "=" * 70)
        print("  FEEL PUBLICATION BATTERY v8.0 - FULL HARDWARE INTEGRATION")
        print("=" * 70)
        print(f"  Prompts: {n_per_category * 4} ({n_per_category} per category)")
        print(f"  Bootstrap iterations: {self.n_bootstrap}")
        print(f"  Sensor mode: FULL (16-dim with hardware)")
        print(f"  Alpha: {self.alpha}")
        print("=" * 70)

        results = {
            "version": "v8.0.0",
            "timestamp": datetime.now().isoformat(),
            "n_prompts": n_per_category * 4,
            "n_bootstrap": self.n_bootstrap,
            "seeds": seeds,
            "alpha": self.alpha,
            "sensor_version": SENSOR_VERSION,
            "projector_version": PROJECTOR_VERSION,
            "conditions": {},
            "lag_sweep": {},
            "benefit_analysis": {},
            "telemetry_validity": {},
            "telemetry_samples": {},
        }

        prompts = get_stratified_sample(n_per_category, seed=seeds[0])
        print(f"\n  Total prompts: {len(prompts)}")

        # Get telemetry validity BEFORE running
        if self.telemetry:
            validity = self.telemetry.get_validity_report()
            results["telemetry_validity"] = {
                "source": validity.source,
                "n_samples": validity.n_samples,
                "availability": {
                    "temp": validity.temp_availability,
                    "power": validity.power_availability,
                    "util": validity.util_availability,
                    "vram": validity.vram_availability,
                },
                "valid": {
                    "temp": validity.temp_valid,
                    "power": validity.power_valid,
                    "util": validity.util_valid,
                    "vram": validity.vram_valid,
                },
            }
            n_valid = sum([validity.temp_valid, validity.power_valid,
                          validity.util_valid, validity.vram_valid])
            print(f"\n  Telemetry: {validity.source}")
            print(f"    Valid channels: {n_valid}/4")
            print(f"    temp={validity.temp_availability*100:.0f}%, power={validity.power_availability*100:.0f}%, "
                  f"util={validity.util_availability*100:.0f}%, vram={validity.vram_availability*100:.0f}%")

        # Run conditions
        conditions = [
            ("baseline", 0),
            ("feel", 0),
            ("feel_off", 0),
            ("random_feel", 0),
            ("shuffled", 0),
            ("cross_prompt", 0),
            ("hardware_only", 0),
            ("internal_only", 0),
        ]

        all_results = {}
        for idx, (cond, lag) in enumerate(conditions):
            print(f"\n[{idx+1}/{len(conditions)}] Running {cond.upper()}...")
            cond_results, telemetry = self.run_condition(prompts, cond, lag=lag)
            all_results[cond] = cond_results

            # Compute stats
            correct = np.array([r["correct"] for r in cond_results])
            acc_ci = bootstrap_ci(correct, self.n_bootstrap)

            results["conditions"][cond] = {
                "accuracy": acc_ci[0],
                "accuracy_ci": list(acc_ci),
                "n_prompts": len(cond_results),
            }

            # Store sample telemetry
            if telemetry and telemetry[0]:
                results["telemetry_samples"][cond] = telemetry[0][:5]  # First 5 tokens of first prompt

            print(f"      Accuracy: {acc_ci[0]:.3f} [{acc_ci[1]:.3f}, {acc_ci[2]:.3f}]")

        # Lag sweep
        print("\n[LAG SWEEP] k={1,2,4,8,16}...")
        for k in [1, 2, 4, 8, 16]:
            lag_results, _ = self.run_condition(prompts, f"lag_{k}", lag=k, verbose=False)
            correct = np.array([r["correct"] for r in lag_results])
            acc_ci = bootstrap_ci(correct, min(500, self.n_bootstrap))
            results["lag_sweep"][str(k)] = {
                "accuracy": acc_ci[0],
                "accuracy_ci": list(acc_ci),
            }
            print(f"      k={k}: Accuracy: {acc_ci[0]:.3f}")

        # Benefit analysis
        baseline_acc = results["conditions"]["baseline"]["accuracy"]
        feel_acc = results["conditions"]["feel"]["accuracy"]
        feel_benefit = feel_acc - baseline_acc

        results["benefit_analysis"] = {
            "feel_benefit": feel_benefit,
            "baseline_accuracy": baseline_acc,
            "feel_accuracy": feel_acc,
            "benefit_collapsed": {
                "shuffled": abs(results["conditions"]["shuffled"]["accuracy"] - baseline_acc) < abs(feel_benefit) * 0.5 if abs(feel_benefit) > 0.01 else True,
                "random": abs(results["conditions"]["random_feel"]["accuracy"] - baseline_acc) < abs(feel_benefit) * 0.5 if abs(feel_benefit) > 0.01 else True,
                "cross_prompt": abs(results["conditions"]["cross_prompt"]["accuracy"] - baseline_acc) < abs(feel_benefit) * 0.5 if abs(feel_benefit) > 0.01 else True,
            }
        }

        # Get final telemetry validity
        if self.telemetry:
            final_validity = self.telemetry.get_validity_report()
            results["telemetry_validity"]["final"] = {
                "n_samples": final_validity.n_samples,
                "duration_sec": final_validity.duration_sec,
                "actual_hz": final_validity.actual_hz,
            }
            self.telemetry.stop()

        return results

    def print_summary(self, results: Dict):
        """Print summary."""
        print("\n" + "=" * 70)
        print("  PUBLICATION BATTERY v8.0 SUMMARY")
        print("=" * 70)

        print("\n  ACCURACY BY CONDITION (95% CI):")
        print("  " + "-" * 60)
        for cond, data in results["conditions"].items():
            ci = data["accuracy_ci"]
            print(f"    {cond:15s}: {ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]")

        if results["lag_sweep"]:
            print("\n  LAG SWEEP:")
            for k, data in results["lag_sweep"].items():
                ci = data["accuracy_ci"]
                print(f"    lag_k={k:2s}:       {ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]")

        print("\n  BENEFIT ANALYSIS:")
        ba = results["benefit_analysis"]
        print(f"    FEEL benefit: {ba['feel_benefit']:+.3f}")
        print(f"    Baseline: {ba['baseline_accuracy']:.3f}")
        print(f"    FEEL: {ba['feel_accuracy']:.3f}")

        print("\n  TELEMETRY VALIDITY:")
        tv = results["telemetry_validity"]
        print(f"    Source: {tv.get('source', 'unknown')}")
        avail = tv.get('availability', {})
        for k, v in avail.items():
            print(f"    {k}: {v*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="FEEL Publication Battery v8.0")
    parser.add_argument("--checkpoint", type=str,
                       default="results/feel_training/canonical_v6_checkpoint.pt")
    parser.add_argument("--alpha", type=float, default=0.01,
                       help="FEEL alpha (larger than v7)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--quick", action="store_true", help="Quick test (32 prompts)")
    parser.add_argument("--medium", action="store_true", help="Medium test (120 prompts)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    n_per_category = 75  # 300 total
    if args.quick:
        n_per_category = 8  # 32 total
        args.bootstrap = 100
    elif args.medium:
        n_per_category = 30  # 120 total
        args.bootstrap = 500

    runner = PublicationRunnerV8(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
    )

    results = runner.run_publication_battery(
        n_per_category=n_per_category,
        seeds=args.seeds,
    )

    runner.print_summary(results)

    # Save results
    results_path = "results/feel_experiments/publication_v8_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
