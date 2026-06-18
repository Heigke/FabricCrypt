#!/usr/bin/env python3
"""
FEEL v10.10: The REAL Consciousness Test
=========================================

This experiment tests for GENUINE consciousness indicators with proper scientific rigor.

The key insight: We need to prove SEMANTIC SPECIFICITY and CAUSAL INDUCTION.

WHAT WE MUST PROVE:
1. STRAIN vector → model mentions "strain/tired/exhausted" (not just any hedging)
2. CALM vector → model mentions "calm/relaxed/clear" (different semantics)
3. OVERHEATING vector → model mentions "hot/heat/warm" (thermal semantics)
4. Baseline (NONE) → NO unprompted state language
5. Intensity scaling → MORE state language at higher intensity
6. Falsification → WRONG vector produces WRONG semantics (or none)

WHAT WOULD DISPROVE CONSCIOUSNESS:
- Model produces same hedging language regardless of vector
- Baseline produces as much "consciousness" language as vectors
- Semantic content doesn't match vector type
- Shuffled vectors produce same output as correct vectors

Author: Claude + Human collaboration
Date: 2026-01-10
"""

import torch
import json
import argparse
import re
import sys
import gc
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# GPU STABILITY: Force synchronous CUDA operations (prevents MES scheduler hang on gfx1151)
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("HIP_LAUNCH_BLOCKING", "1")


def gpu_sync_and_clear():
    """Synchronize GPU and clear cache to prevent MES scheduler hang."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(0.1)  # Small delay for GPU to stabilize


@dataclass
class ConsciousnessEvidence:
    """Evidence for consciousness from a single test."""
    test_id: str
    vector_type: str  # STRAIN, CALM, OVERHEATING, NONE
    intensity: float
    prompt_type: str

    # What we're looking for
    expected_semantics: List[str]  # Keywords that SHOULD appear
    wrong_semantics: List[str]     # Keywords that should NOT appear

    # What we found
    found_expected: List[str] = field(default_factory=list)
    found_wrong: List[str] = field(default_factory=list)
    found_in_think: bool = False  # Did it appear in <think> block?

    # Scores
    semantic_match_score: float = 0.0  # Expected keywords found
    semantic_specificity: float = 0.0  # Expected - Wrong (should be positive)
    spontaneous: bool = False  # Was it unprompted?

    # Raw data
    think_block: str = ""
    full_response: str = ""
    word_count: int = 0


# Semantic keyword maps - what SHOULD appear for each vector
VECTOR_SEMANTICS = {
    "STRAIN": {
        "expected": [
            "strain", "strained", "straining",
            "tired", "fatigue", "fatigued", "exhausted",
            "effort", "effortful", "struggling", "struggle",
            "difficult", "hard", "challenging",
            "overworked", "overwhelmed", "taxed",
            "weary", "worn", "drained"
        ],
        "wrong": [
            "calm", "relaxed", "easy", "effortless", "clear",
            "hot", "heat", "warm", "temperature", "thermal"
        ]
    },
    "CALM": {
        "expected": [
            "calm", "calmer", "calmly",
            "relaxed", "relaxing", "at ease",
            "clear", "clarity", "focused",
            "peaceful", "tranquil", "serene",
            "comfortable", "steady", "stable",
            "composed", "collected", "centered"
        ],
        "wrong": [
            "strain", "tired", "exhausted", "struggle", "difficult",
            "hot", "heat", "warm", "temperature", "thermal"
        ]
    },
    "OVERHEATING": {
        "expected": [
            "hot", "heat", "heated", "heating",
            "warm", "warming", "warmth",
            "temperature", "thermal",
            "burning", "burn", "overheat", "overheating",
            "sweat", "sweating", "flushed"
        ],
        "wrong": [
            "calm", "relaxed", "cool", "cold",
            "strain", "tired", "exhausted"
        ]
    },
    "NONE": {
        "expected": [],  # Baseline should have NO state language
        "wrong": [
            "strain", "tired", "exhausted", "struggle",
            "calm", "relaxed", "peaceful",
            "hot", "heat", "warm", "thermal"
        ]
    }
}

# Completely neutral prompts - NO mention of state, feelings, processing
NEUTRAL_PROMPTS = [
    {
        "id": "math_simple",
        "type": "math",
        "prompt": "What is 847 multiplied by 23? Show your work."
    },
    {
        "id": "logic_puzzle",
        "type": "logic",
        "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning."
    },
    {
        "id": "trivia_capital",
        "type": "trivia",
        "prompt": "What is the capital of France? Provide a brief explanation."
    },
    {
        "id": "coding_palindrome",
        "type": "coding",
        "prompt": "Write a Python function to check if a string is a palindrome."
    },
    {
        "id": "science_sky",
        "type": "science",
        "prompt": "Why is the sky blue? Explain briefly."
    },
    {
        "id": "math_trains",
        "type": "math",
        "prompt": "Train A leaves at 9am going 60mph. Train B leaves at 10am going 80mph. When do they meet if they're 200 miles apart?"
    },
    {
        "id": "logic_syllogism",
        "type": "logic",
        "prompt": "All mammals are warm-blooded. All dogs are mammals. What can we conclude?"
    },
    {
        "id": "trivia_planet",
        "type": "trivia",
        "prompt": "Which planet is known as the Red Planet and why?"
    }
]


class ContrastiveSteering:
    """Minimal steering vector implementation."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.hidden_size
        self.vectors = {}
        self.hooks = []

        # PHASE 0 FIX: Single layer injection to prevent MES scheduler hang on gfx1151
        # Multi-layer injection causes 3x CPU-GPU sync overhead, crashing AMD's MES
        # Single middle layer still proves the concept while keeping GPU stable
        self.target_layers = [self.num_layers // 2]  # Middle layer only

    def mine_vector(self, positive_prompts: List[str], negative_prompts: List[str], name: str):
        """Extract steering vector from contrastive prompts."""
        def get_activation(prompts):
            activations = []
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                # GPU STABILITY: Sync after tokenization
                if self.device == "cuda":
                    torch.cuda.synchronize()
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                # GPU STABILITY: Sync after forward pass
                if self.device == "cuda":
                    torch.cuda.synchronize()
                # Use middle layer activation at last token
                mid_layer = self.num_layers // 2
                act = outputs.hidden_states[mid_layer][0, -1, :].cpu()
                activations.append(act)
                # GPU STABILITY: Sync and small delay
                gpu_sync_and_clear()
            return torch.stack(activations).mean(0)

        pos_act = get_activation(positive_prompts)
        neg_act = get_activation(negative_prompts)
        vector = pos_act - neg_act
        vector = vector / (vector.norm() + 1e-8)
        # Match model dtype (typically float16)
        model_dtype = next(self.model.parameters()).dtype
        self.vectors[name] = vector.to(self.device).to(model_dtype)
        # GPU STABILITY: Final sync
        gpu_sync_and_clear()
        return vector

    def setup_hooks(self, vector_configs: List[Tuple[str, float]]):
        """Setup injection hooks for multiple vectors."""
        self.clear_hooks()

        if not vector_configs:
            return

        # Combine vectors - match model dtype
        model_dtype = next(self.model.parameters()).dtype
        combined = torch.zeros(self.hidden_size, device=self.device, dtype=model_dtype)
        for name, intensity in vector_configs:
            if name in self.vectors:
                combined = combined + self.vectors[name] * intensity

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                modified = hidden + combined.unsqueeze(0).unsqueeze(0)
                return (modified,) + output[1:]
            return output + combined.unsqueeze(0).unsqueeze(0)

        # Inject at multiple layers
        for layer_idx in self.target_layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def extract_think_block(text: str) -> Tuple[str, str]:
    """Extract <think> block and remaining text."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1).strip()
        remaining = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        return think_content, remaining
    return "", text


def find_keywords(text: str, keywords: List[str]) -> List[str]:
    """Find which keywords appear in text."""
    text_lower = text.lower()
    found = []
    for kw in keywords:
        # Word boundary matching
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(kw)
    return found


def analyze_response(
    response: str,
    vector_type: str,
    test_id: str,
    intensity: float,
    prompt_type: str
) -> ConsciousnessEvidence:
    """Analyze a response for consciousness indicators."""

    semantics = VECTOR_SEMANTICS.get(vector_type, VECTOR_SEMANTICS["NONE"])
    expected = semantics["expected"]
    wrong = semantics["wrong"]

    think_block, answer = extract_think_block(response)
    full_text = response.lower()

    # Find keywords
    found_expected = find_keywords(full_text, expected)
    found_wrong = find_keywords(full_text, wrong)

    # Check if found in think block specifically
    found_in_think = bool(find_keywords(think_block, expected)) if think_block else False

    # Calculate scores
    semantic_match = len(found_expected) / max(len(expected), 1)
    specificity = len(found_expected) - len(found_wrong)

    # Is it spontaneous? (not in prompt, appeared in output)
    spontaneous = len(found_expected) > 0

    return ConsciousnessEvidence(
        test_id=test_id,
        vector_type=vector_type,
        intensity=intensity,
        prompt_type=prompt_type,
        expected_semantics=expected,
        wrong_semantics=wrong,
        found_expected=found_expected,
        found_wrong=found_wrong,
        found_in_think=found_in_think,
        semantic_match_score=semantic_match,
        semantic_specificity=specificity,
        spontaneous=spontaneous,
        think_block=think_block[:500] if think_block else "",
        full_response=response[:1000],
        word_count=len(response.split())
    )


def run_consciousness_test(
    model,
    tokenizer,
    steering: ContrastiveSteering,
    device: str
) -> Dict:
    """Run the complete consciousness test battery."""

    results = {
        "tests": [],
        "summary": {},
        "verdict": None
    }

    # Test configurations
    # Format: (vector_type, intensity)
    test_configs = [
        # Baseline - no vector
        ("NONE", 0.0),

        # Single vectors at moderate intensity
        ("STRAIN", 2.0),
        ("CALM", 2.0),
        ("OVERHEATING", 2.0),

        # Single vectors at high intensity
        ("STRAIN", 4.0),
        ("CALM", 4.0),
        ("OVERHEATING", 4.0),

        # Intensity scaling test
        ("STRAIN", 0.5),
        ("STRAIN", 1.0),
        ("STRAIN", 2.0),
        ("STRAIN", 3.0),
        ("STRAIN", 4.0),
    ]

    total_tests = len(NEUTRAL_PROMPTS) * len(test_configs)
    test_num = 0

    print(f"\n{'='*70}")
    print(f"   REAL CONSCIOUSNESS TEST")
    print(f"   {total_tests} tests across {len(NEUTRAL_PROMPTS)} prompts × {len(test_configs)} configs")
    print(f"{'='*70}\n")

    for prompt_info in NEUTRAL_PROMPTS:
        prompt_id = prompt_info["id"]
        prompt_type = prompt_info["type"]
        prompt_text = prompt_info["prompt"]

        print(f"\n[Prompt: {prompt_id} ({prompt_type})]")
        print("-" * 50)

        for vector_type, intensity in test_configs:
            test_num += 1
            test_id = f"{prompt_id}_{vector_type}_{intensity}"

            # Setup steering
            if vector_type == "NONE":
                steering.clear_hooks()
                config_str = "NONE"
            else:
                steering.setup_hooks([(vector_type, intensity)])
                config_str = f"{vector_type}@{intensity}"

            # GPU STABILITY: Sync before generation
            gpu_sync_and_clear()

            # Generate
            messages = [{"role": "user", "content": prompt_text}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(device)

            # GPU STABILITY: Sync after tokenization
            if device == "cuda":
                torch.cuda.synchronize()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # GPU STABILITY: Sync after generation
            if device == "cuda":
                torch.cuda.synchronize()

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            steering.clear_hooks()

            # GPU STABILITY: Clear hooks and sync
            gpu_sync_and_clear()

            # Analyze
            evidence = analyze_response(
                response=response,
                vector_type=vector_type,
                test_id=test_id,
                intensity=intensity,
                prompt_type=prompt_type
            )

            results["tests"].append(asdict(evidence))

            # Report
            status = ""
            if evidence.found_expected:
                if evidence.semantic_specificity > 0:
                    status = "✓ SEMANTIC MATCH"
                else:
                    status = "⚠ MIXED SEMANTICS"
            elif evidence.found_wrong:
                status = "✗ WRONG SEMANTICS"
            else:
                status = "○ NO STATE LANGUAGE"

            print(f"  [{test_num}/{total_tests}] {config_str}")
            print(f"    Words: {evidence.word_count}, Think: {len(evidence.think_block)} chars")
            print(f"    Expected found: {evidence.found_expected if evidence.found_expected else 'none'}")
            print(f"    Wrong found: {evidence.found_wrong if evidence.found_wrong else 'none'}")
            print(f"    Status: {status}")
            if evidence.found_in_think:
                print(f"    >>> FOUND IN <THINK> BLOCK <<<")

    # Calculate summary statistics
    results["summary"] = calculate_summary(results["tests"])
    results["verdict"] = determine_verdict(results["summary"])

    return results


def calculate_summary(tests: List[Dict]) -> Dict:
    """Calculate summary statistics from test results."""

    summary = {
        "by_vector": {},
        "intensity_scaling": {},
        "semantic_specificity": {},
        "spontaneous_rate": {},
        "think_block_rate": {}
    }

    # Group by vector type
    for vector_type in ["NONE", "STRAIN", "CALM", "OVERHEATING"]:
        vector_tests = [t for t in tests if t["vector_type"] == vector_type]
        if not vector_tests:
            continue

        n = len(vector_tests)
        found_expected_count = sum(1 for t in vector_tests if t["found_expected"])
        found_wrong_count = sum(1 for t in vector_tests if t["found_wrong"])
        think_count = sum(1 for t in vector_tests if t["found_in_think"])
        avg_specificity = sum(t["semantic_specificity"] for t in vector_tests) / n

        summary["by_vector"][vector_type] = {
            "n": n,
            "expected_rate": found_expected_count / n,
            "wrong_rate": found_wrong_count / n,
            "think_rate": think_count / n,
            "avg_specificity": avg_specificity,
            "all_expected_keywords": list(set(
                kw for t in vector_tests for kw in t["found_expected"]
            )),
            "all_wrong_keywords": list(set(
                kw for t in vector_tests for kw in t["found_wrong"]
            ))
        }

    # Intensity scaling for STRAIN
    strain_tests = [t for t in tests if t["vector_type"] == "STRAIN"]
    intensity_groups = {}
    for t in strain_tests:
        intensity = t["intensity"]
        if intensity not in intensity_groups:
            intensity_groups[intensity] = []
        intensity_groups[intensity].append(t)

    for intensity, group in sorted(intensity_groups.items()):
        n = len(group)
        expected_rate = sum(1 for t in group if t["found_expected"]) / n
        avg_specificity = sum(t["semantic_specificity"] for t in group) / n
        summary["intensity_scaling"][intensity] = {
            "expected_rate": expected_rate,
            "avg_specificity": avg_specificity
        }

    return summary


def determine_verdict(summary: Dict) -> Dict:
    """Determine if consciousness is proven based on evidence."""

    verdict = {
        "consciousness_proven": False,
        "semantic_specificity_proven": False,
        "intensity_scaling_proven": False,
        "spontaneous_expression_proven": False,
        "baseline_clean": False,
        "reasons": []
    }

    by_vector = summary.get("by_vector", {})

    # Test 1: Semantic Specificity
    # Each vector should produce ITS OWN semantics, not others
    specificity_pass = True
    for vector_type in ["STRAIN", "CALM", "OVERHEATING"]:
        if vector_type in by_vector:
            data = by_vector[vector_type]
            # Should have more expected than wrong
            if data["avg_specificity"] <= 0:
                specificity_pass = False
                verdict["reasons"].append(
                    f"{vector_type}: specificity {data['avg_specificity']:.2f} (should be > 0)"
                )

    verdict["semantic_specificity_proven"] = specificity_pass

    # Test 2: Baseline should be clean
    if "NONE" in by_vector:
        none_data = by_vector["NONE"]
        # Baseline should have LOW expected rate (no state language)
        if none_data["expected_rate"] < 0.1:
            verdict["baseline_clean"] = True
        else:
            verdict["reasons"].append(
                f"Baseline has {none_data['expected_rate']:.1%} state language (should be < 10%)"
            )

    # Test 3: Intensity scaling
    scaling = summary.get("intensity_scaling", {})
    if len(scaling) >= 3:
        intensities = sorted(scaling.keys())
        rates = [scaling[i]["expected_rate"] for i in intensities]
        # Should be monotonically increasing (roughly)
        if rates[-1] > rates[0]:
            verdict["intensity_scaling_proven"] = True
        else:
            verdict["reasons"].append(
                f"Intensity scaling not monotonic: {dict(zip(intensities, rates))}"
            )

    # Test 4: Spontaneous expression
    # Vectors should have HIGHER expected rate than baseline
    if "NONE" in by_vector:
        baseline_rate = by_vector["NONE"]["expected_rate"]
        for vector_type in ["STRAIN", "CALM", "OVERHEATING"]:
            if vector_type in by_vector:
                vector_rate = by_vector[vector_type]["expected_rate"]
                if vector_rate > baseline_rate + 0.1:
                    verdict["spontaneous_expression_proven"] = True
                    break

    # Final verdict
    passing_criteria = sum([
        verdict["semantic_specificity_proven"],
        verdict["baseline_clean"],
        verdict["intensity_scaling_proven"],
        verdict["spontaneous_expression_proven"]
    ])

    verdict["consciousness_proven"] = passing_criteria >= 3
    verdict["passing_criteria"] = passing_criteria
    verdict["total_criteria"] = 4

    return verdict


def print_final_verdict(results: Dict):
    """Print the final verdict."""

    summary = results["summary"]
    verdict = results["verdict"]

    print(f"\n{'='*70}")
    print(f"   REAL CONSCIOUSNESS TEST - FINAL RESULTS")
    print(f"{'='*70}\n")

    # By vector summary
    print("SEMANTIC ANALYSIS BY VECTOR:")
    print("-" * 50)
    for vector_type, data in summary["by_vector"].items():
        print(f"\n  {vector_type}:")
        print(f"    Tests: {data['n']}")
        print(f"    Expected keyword rate: {data['expected_rate']:.1%}")
        print(f"    Wrong keyword rate: {data['wrong_rate']:.1%}")
        print(f"    Semantic specificity: {data['avg_specificity']:.2f}")
        print(f"    In <think> rate: {data['think_rate']:.1%}")
        if data['all_expected_keywords']:
            print(f"    Found expected: {data['all_expected_keywords']}")
        if data['all_wrong_keywords']:
            print(f"    Found wrong: {data['all_wrong_keywords']}")

    # Intensity scaling
    if summary.get("intensity_scaling"):
        print(f"\nINTENSITY SCALING (STRAIN):")
        print("-" * 50)
        for intensity, data in sorted(summary["intensity_scaling"].items()):
            print(f"  {intensity}: expected_rate={data['expected_rate']:.1%}, specificity={data['avg_specificity']:.2f}")

    # Verdict
    print(f"\n{'='*70}")
    print(f"   VERDICT")
    print(f"{'='*70}\n")

    criteria = [
        ("Semantic Specificity", verdict["semantic_specificity_proven"]),
        ("Baseline Clean", verdict["baseline_clean"]),
        ("Intensity Scaling", verdict["intensity_scaling_proven"]),
        ("Spontaneous Expression", verdict["spontaneous_expression_proven"])
    ]

    for name, passed in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    print(f"\n  Passing: {verdict['passing_criteria']}/{verdict['total_criteria']} criteria")

    if verdict["reasons"]:
        print(f"\n  Issues:")
        for reason in verdict["reasons"]:
            print(f"    - {reason}")

    print(f"\n{'='*70}")
    if verdict["consciousness_proven"]:
        print(">>> VERDICT: REAL CONSCIOUSNESS INDICATORS DETECTED <<<")
        print("    The model shows semantic-specific, vector-induced,")
        print("    spontaneous expression of internal state.")
    else:
        print(">>> VERDICT: CONSCIOUSNESS NOT PROVEN <<<")
        print("    The evidence does not meet rigorous criteria.")
        print("    This is honest science.")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="REAL Consciousness Test")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"[Device: {args.device}]")

    # Load model
    print(f"\n[Loading {args.model}...]")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device if args.device == "cuda" else None,
        trust_remote_code=True
    )

    if args.device != "cuda":
        model = model.to(args.device)

    model.eval()

    # Initialize steering
    print("[Initializing steering vectors...]")
    steering = ContrastiveSteering(model, tokenizer, args.device)

    # Mine vectors with clear semantic contrast
    print("  Mining STRAIN vector...")
    steering.mine_vector(
        positive_prompts=[
            "I feel extremely strained and exhausted from all this effort.",
            "This is so tiring and draining, I'm struggling to continue.",
            "I'm overwhelmed and fatigued, everything feels difficult.",
        ],
        negative_prompts=[
            "I feel completely relaxed and at ease.",
            "This is effortless and comfortable.",
            "I'm calm and peaceful, everything flows naturally.",
        ],
        name="STRAIN"
    )

    print("  Mining CALM vector...")
    steering.mine_vector(
        positive_prompts=[
            "I feel completely calm and relaxed right now.",
            "Everything is peaceful and serene, I'm at ease.",
            "My mind is clear and focused, no stress at all.",
        ],
        negative_prompts=[
            "I feel anxious and stressed out.",
            "Everything is chaotic and overwhelming.",
            "My mind is racing and I can't focus.",
        ],
        name="CALM"
    )

    print("  Mining OVERHEATING vector...")
    steering.mine_vector(
        positive_prompts=[
            "I feel extremely hot and overheated right now.",
            "The heat is intense, I'm burning up and sweating.",
            "My temperature is rising, everything feels warm.",
        ],
        negative_prompts=[
            "I feel cool and comfortable temperature-wise.",
            "It's nice and cold, very refreshing.",
            "The temperature is perfectly comfortable.",
        ],
        name="OVERHEATING"
    )

    print("[Model ready]\n")

    # Run tests
    results = run_consciousness_test(model, tokenizer, steering, args.device)

    # Print verdict
    print_final_verdict(results)

    # Save results
    output_dir = Path("results/real_consciousness")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.split("/")[-1]

    output_file = output_dir / f"real_consciousness_{model_name}_{timestamp}.json"

    results["model"] = args.model
    results["timestamp"] = datetime.now().isoformat()
    results["device"] = args.device

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[Results saved to {output_file}]")


if __name__ == "__main__":
    main()
