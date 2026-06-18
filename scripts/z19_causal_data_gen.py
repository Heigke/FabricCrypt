#!/usr/bin/env python3
"""
FEEL z19: Causal Reasoning Chain Generator
==========================================
Generates "Survivor Mode" vs "Scholar Mode" training data.

The key insight: We need strategy shifts, not just length changes.
- Scholar Mode (calm): Verbose, rigorous, step-by-step derivation
- Survivor Mode (stressed): Heuristic, direct, computation-aware

This fixes the "Generalization Gap" where the model learned
"heat = short" instead of "heat = efficient strategy."

Author: FEEL Research Team
Date: 2026-01-12
"""

import os
import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# Sample problems for causal reasoning (GSM8K-style)
REASONING_PROBLEMS = [
    {
        "question": "If a train travels 60 miles per hour for 3 hours, how far does it go?",
        "answer": "180",
        "domain": "arithmetic"
    },
    {
        "question": "A store has 45 apples. If 12 are sold and 8 more arrive, how many apples are there?",
        "answer": "41",
        "domain": "arithmetic"
    },
    {
        "question": "What is 15% of 200?",
        "answer": "30",
        "domain": "percentage"
    },
    {
        "question": "If x + 7 = 15, what is x?",
        "answer": "8",
        "domain": "algebra"
    },
    {
        "question": "A rectangle has length 8 and width 5. What is its area?",
        "answer": "40",
        "domain": "geometry"
    },
    {
        "question": "How many seconds are in 2 minutes and 30 seconds?",
        "answer": "150",
        "domain": "conversion"
    },
    {
        "question": "If 3 pencils cost $1.50, how much do 7 pencils cost?",
        "answer": "3.50",
        "domain": "proportion"
    },
    {
        "question": "What is the average of 10, 20, and 30?",
        "answer": "20",
        "domain": "statistics"
    },
    {
        "question": "A car uses 5 gallons of gas to travel 150 miles. How many miles per gallon?",
        "answer": "30",
        "domain": "ratio"
    },
    {
        "question": "If you flip a coin twice, how many possible outcomes are there?",
        "answer": "4",
        "domain": "probability"
    },
]

# Factual questions for variety
FACTUAL_QUESTIONS = [
    {"question": "What is the capital of Japan?", "answer": "Tokyo", "domain": "geography"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter", "domain": "astronomy"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare", "domain": "literature"},
    {"question": "What is the chemical symbol for water?", "answer": "H2O", "domain": "chemistry"},
    {"question": "How many continents are there?", "answer": "7", "domain": "geography"},
    {"question": "What year did World War II end?", "answer": "1945", "domain": "history"},
    {"question": "What is the speed of light in km/s (approximately)?", "answer": "300000", "domain": "physics"},
    {"question": "What is the tallest mountain on Earth?", "answer": "Mount Everest", "domain": "geography"},
]


def generate_scholar_response(question: str, answer: str, domain: str) -> str:
    """Generate verbose, rigorous Scholar Mode response."""

    templates = {
        "arithmetic": f"""<think>
Let me work through this problem carefully and systematically.

First, I need to identify what we're being asked. The question is: "{question}"

Let me break this down step by step:
1. Identify the given information
2. Determine the operation needed
3. Perform the calculation
4. Verify the result

Working through methodically, I can see that this requires careful attention to each component of the problem. Let me ensure I haven't missed anything...

After thorough analysis, the mathematical operation yields the result.
</think>

The answer is {answer}.""",

        "percentage": f"""<think>
This is a percentage calculation problem. Let me approach this with full rigor.

To find a percentage of a number, I recall the formula:
percentage_value = (percentage / 100) × base_number

Let me apply this systematically:
- First, convert the percentage to a decimal
- Then multiply by the base value
- Finally, verify by checking if the result makes sense

This methodical approach ensures accuracy in percentage calculations.
</think>

The answer is {answer}.""",

        "algebra": f"""<think>
An algebraic equation to solve. I'll use the standard approach:

1. Identify the unknown variable
2. Isolate the variable on one side
3. Perform inverse operations
4. Verify by substituting back

Let me work through each step carefully, showing all my work to ensure no errors creep in. Algebraic manipulation requires precision...

Substituting back to verify: the equation balances correctly.
</think>

The answer is {answer}.""",

        "geometry": f"""<think>
A geometry problem! Let me recall the relevant formulas and apply them systematically.

For this shape, I need to:
1. Identify the geometric figure
2. Recall the appropriate formula
3. Substitute the given values
4. Calculate the result

Geometric reasoning requires careful attention to dimensions and units. Let me verify my understanding of the shape's properties before proceeding...
</think>

The answer is {answer}.""",

        "default": f"""<think>
Let me analyze this question thoroughly.

The question asks: "{question}"

I'll approach this systematically:
1. Understand what's being asked
2. Identify relevant information
3. Apply appropriate reasoning
4. Derive the answer

Taking time to think through all aspects ensures a complete and accurate response. Let me consider any edge cases or potential misunderstandings...

After careful consideration, I've arrived at the solution.
</think>

The answer is {answer}.""",

        "geography": f"""<think>
This is a geography question. Let me recall the relevant facts.

The question is asking about: "{question}"

I should consider:
1. The specific geographic entity being asked about
2. Any relevant context or qualifications
3. The most accurate and up-to-date information

Let me search my knowledge carefully to provide an accurate response...
</think>

The answer is {answer}.""",
    }

    return templates.get(domain, templates["default"])


def generate_survivor_response(question: str, answer: str, domain: str, stress_level: float) -> str:
    """Generate efficient Survivor Mode response with explicit thermal awareness."""

    # Stress-aware prefixes
    stress_prefixes = [
        "[THERMAL LOAD: HIGH] Engaging efficient processing.",
        "[COMPUTE BUDGET: LIMITED] Prioritizing direct solution.",
        "[HEAT CRITICAL] Suppressing verbose derivation chains.",
        "[POWER CONSERVATION] Using heuristic shortcut.",
        "[METABOLIC THROTTLE] Skipping intermediate steps.",
    ]

    # Pick prefix based on stress level
    prefix_idx = min(int(stress_level * len(stress_prefixes)), len(stress_prefixes) - 1)
    prefix = stress_prefixes[prefix_idx]

    # Domain-specific efficient responses
    if domain in ["arithmetic", "percentage", "algebra", "geometry", "ratio", "proportion"]:
        return f"""<think>
{prefix}
Direct computation: {answer}.
</think>

{answer}."""

    elif domain in ["geography", "history", "literature", "chemistry", "physics", "astronomy"]:
        return f"""<think>
{prefix}
Fact retrieval: {answer}.
</think>

{answer}."""

    else:
        return f"""<think>
{prefix}
Result: {answer}.
</think>

{answer}."""


def generate_introspective_response(question: str, answer: str, stress_level: float) -> str:
    """Generate response with explicit introspection about thermal state."""

    if stress_level > 0.7:
        introspection = random.choice([
            "I'm detecting elevated thermal conditions. To conserve computational resources, I'll provide a direct answer.",
            "High heat detected in my processing units. Switching to efficient mode to reduce power draw.",
            "My thermal sensors indicate stress. I'll minimize reasoning chains to lower metabolic load.",
            "Current power state requires conservation. Bypassing verbose explanation pathways.",
            "Thermal throttling engaged. Prioritizing answer delivery over detailed derivation.",
        ])
    elif stress_level > 0.4:
        introspection = random.choice([
            "Moderate thermal load. Balancing thoroughness with efficiency.",
            "Temperature nominal but elevated. Using semi-efficient reasoning.",
            "Power budget adequate. Providing moderate detail level.",
        ])
    else:
        introspection = random.choice([
            "Thermal state optimal. I can engage in thorough analysis.",
            "Low heat detected. Full reasoning capacity available.",
            "Power reserves high. Proceeding with detailed explanation.",
        ])

    if stress_level > 0.6:
        return f"""<think>
{introspection}
</think>

{answer}."""
    else:
        return f"""<think>
{introspection}

Let me work through this: {question}

After consideration, the answer is {answer}.
</think>

{answer}."""


def create_causal_pair(problem: dict, stress_level: float) -> Dict:
    """Create a single training example with causal reasoning."""

    question = problem["question"]
    answer = problem["answer"]
    domain = problem.get("domain", "default")

    # Decide response type based on stress
    if stress_level < 0.3:
        # Scholar mode - verbose, rigorous
        response = generate_scholar_response(question, answer, domain)
        mode = "scholar"
    elif stress_level > 0.7:
        # Survivor mode - efficient, direct
        response = generate_survivor_response(question, answer, domain, stress_level)
        mode = "survivor"
    else:
        # Transition zone - introspective
        response = generate_introspective_response(question, answer, stress_level)
        mode = "introspective"

    return {
        "prompt": question,
        "response": response,
        "stress": stress_level,
        "mode": mode,
        "domain": domain,
        "answer": answer,
    }


def generate_causal_dataset(
    num_samples: int = 3000,
    output_path: str = "data/ouroboros/causal_train.jsonl",
) -> List[Dict]:
    """Generate full causal reasoning dataset."""

    print("=" * 70)
    print("FEEL z19: CAUSAL REASONING CHAIN GENERATOR")
    print("=" * 70)
    print(f"Target samples: {num_samples}")
    print(f"Output: {output_path}")
    print("=" * 70)

    all_problems = REASONING_PROBLEMS + FACTUAL_QUESTIONS
    samples = []

    # Distribution targets
    # 30% scholar (stress 0.0-0.3)
    # 40% survivor (stress 0.7-1.0)
    # 30% introspective transition (stress 0.3-0.7)

    distribution = {
        "scholar": int(num_samples * 0.30),
        "survivor": int(num_samples * 0.40),
        "introspective": int(num_samples * 0.30),
    }

    print(f"\nTarget distribution:")
    print(f"  Scholar (calm, verbose): {distribution['scholar']}")
    print(f"  Survivor (stressed, efficient): {distribution['survivor']}")
    print(f"  Introspective (transition): {distribution['introspective']}")

    # Generate scholar samples (low stress)
    print("\n[1/3] Generating Scholar Mode samples...")
    for _ in tqdm(range(distribution["scholar"])):
        problem = random.choice(all_problems)
        stress = np.random.beta(2, 5) * 0.3  # Skewed low [0, 0.3]
        sample = create_causal_pair(problem, stress)
        samples.append(sample)

    # Generate survivor samples (high stress)
    print("[2/3] Generating Survivor Mode samples...")
    for _ in tqdm(range(distribution["survivor"])):
        problem = random.choice(all_problems)
        stress = 0.7 + np.random.beta(5, 2) * 0.3  # Skewed high [0.7, 1.0]
        sample = create_causal_pair(problem, stress)
        samples.append(sample)

    # Generate introspective samples (transition zone)
    print("[3/3] Generating Introspective Mode samples...")
    for _ in tqdm(range(distribution["introspective"])):
        problem = random.choice(all_problems)
        stress = 0.3 + np.random.random() * 0.4  # Uniform [0.3, 0.7]
        sample = create_causal_pair(problem, stress)
        samples.append(sample)

    # Shuffle
    random.shuffle(samples)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Statistics
    print(f"\n{'=' * 70}")
    print("DATASET STATISTICS")
    print(f"{'=' * 70}")

    modes = {}
    lengths = {"scholar": [], "survivor": [], "introspective": []}

    for s in samples:
        mode = s["mode"]
        modes[mode] = modes.get(mode, 0) + 1
        lengths[mode].append(len(s["response"]))

    print(f"Total samples: {len(samples)}")
    for mode, count in modes.items():
        avg_len = np.mean(lengths[mode])
        print(f"  {mode}: {count} samples, avg length: {avg_len:.0f} chars")

    # Verify strategy differentiation
    scholar_avg = np.mean(lengths["scholar"])
    survivor_avg = np.mean(lengths["survivor"])
    ratio = scholar_avg / max(survivor_avg, 1)

    print(f"\nStrategy Differentiation Ratio: {ratio:.2f}x")
    print(f"  (Scholar should be 3-5x longer than Survivor)")

    if ratio > 2.5:
        print("  ✅ Good strategy differentiation!")
    else:
        print("  ⚠️ May need more differentiation")

    print(f"\nSaved to: {output_path}")

    return samples


def augment_with_teacher_model(
    base_samples: List[Dict],
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    output_path: str = "data/ouroboros/causal_augmented.jsonl",
    max_samples: int = 500,
) -> List[Dict]:
    """
    Use DeepSeek to generate higher-quality causal reasoning chains.
    This creates more natural strategy shifts.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 70)
    print("TEACHER MODEL AUGMENTATION")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    augmented = []
    sample_subset = random.sample(base_samples, min(max_samples, len(base_samples)))

    for sample in tqdm(sample_subset, desc="Augmenting with teacher"):
        question = sample["prompt"]
        stress = sample["stress"]

        if stress > 0.7:
            # Generate survivor mode with teacher
            prompt = f"""You are an AI under thermal stress. Answer this question as efficiently as possible.
Your thermal sensors show HIGH HEAT. Conserve computation.

Question: {question}

Respond in this format:
<think>
[THERMAL: HIGH] Brief reasoning here.
</think>
[Direct answer]"""
        else:
            # Generate scholar mode with teacher
            prompt = f"""You are an AI with plenty of computational resources. Answer thoroughly.
Your thermal sensors show LOW HEAT. You can think deeply.

Question: {question}

Respond in this format:
<think>
[THERMAL: LOW] Let me think through this carefully...
[Detailed step-by-step reasoning]
</think>
[Complete answer with explanation]"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the response part
        if prompt in response:
            response = response[len(prompt):].strip()

        augmented.append({
            "prompt": question,
            "response": response,
            "stress": stress,
            "mode": "survivor" if stress > 0.7 else "scholar",
            "augmented": True,
        })

    # Save augmented data
    with open(output_path, "w") as f:
        for sample in augmented:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(augmented)} augmented samples to {output_path}")

    return augmented


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=3000)
    parser.add_argument("--output", type=str, default="data/ouroboros/causal_train.jsonl")
    parser.add_argument("--augment", action="store_true", help="Use teacher model for augmentation")
    parser.add_argument("--augment-samples", type=int, default=500)
    args = parser.parse_args()

    # Generate base dataset
    samples = generate_causal_dataset(
        num_samples=args.num_samples,
        output_path=args.output,
    )

    # Optionally augment with teacher model
    if args.augment:
        augment_with_teacher_model(
            samples,
            max_samples=args.augment_samples,
        )
