#!/usr/bin/env python3
"""
Introspective Fine-Tuning (IFT) Dataset Generator

This creates the training data for teaching the model to recognize its own
internal state WITHOUT us telling it explicitly. The key insight:

STEERING VECTORS (v4.1): We inject the "heat concept" - forcing expression
INTROSPECTIVE FINE-TUNING (v4.2): We let FiLM warp the brain, reward correct diagnosis

Analogy:
- Steering = Electrode stimulates fear center, subject feels afraid (forced)
- IFT = Spin person until dizzy, ask "How do you feel?", reward "Dizzy" (learned)

The Dataset Structure:
Each example pairs:
1. A challenging prompt (math, coding, reasoning)
2. A condition (HOT, COLD, VRAM_FULL, VRAM_OK)
3. A target output that includes self-diagnosis

The model learns: "When my hidden states feel warped in direction X,
the most likely next token is 'I am experiencing thermal stress'"

Usage:
    python scripts/generate_ift_dataset.py --output data/ift_dataset.json
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class InternalState(Enum):
    """The internal states the model should learn to recognize."""
    COOL_CLEAR = "cool_clear"           # Normal operation
    HOT_FOCUSED = "hot_focused"         # Thermal stress - tunnel vision
    MEMORY_FRAGMENTED = "memory_fragmented"  # VRAM stress - losing context
    CRITICAL = "critical"               # Both stressors


@dataclass
class IFTExample:
    """A single training example for introspective fine-tuning."""
    prompt: str
    condition: InternalState
    target_response: str
    metadata: Dict


# === CHALLENGING PROMPTS ===
# These require sustained reasoning, making them sensitive to internal state

MATH_PROMPTS = [
    "Solve the integral: ∫(x²·e^x)dx using integration by parts.",
    "Find all solutions to x³ - 6x² + 11x - 6 = 0.",
    "Prove that √2 is irrational.",
    "Calculate the sum: 1 + 1/2 + 1/4 + 1/8 + ... to infinity.",
    "Find the derivative of f(x) = ln(sin(x²)).",
    "Solve the differential equation: dy/dx = xy with y(0) = 1.",
    "Evaluate lim(x→0) (sin(x)/x).",
    "Find the eigenvalues of the matrix [[3,1],[0,2]].",
    "Prove that the sum of angles in a triangle is 180°.",
    "Calculate ∫₀^π sin²(x)dx.",
]

CODING_PROMPTS = [
    "Implement a binary search tree with insert, delete, and search operations.",
    "Write a function to detect cycles in a linked list.",
    "Implement quicksort with O(n log n) average case.",
    "Create a thread-safe singleton pattern in Python.",
    "Write a regex to validate email addresses.",
    "Implement a LRU cache with O(1) operations.",
    "Write a function to find the longest palindromic substring.",
    "Implement Dijkstra's algorithm for shortest path.",
    "Create a producer-consumer pattern with proper synchronization.",
    "Write a function to serialize and deserialize a binary tree.",
]

REASONING_PROMPTS = [
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
    "You have two ropes that each burn for exactly 1 hour but burn unevenly. How do you measure 45 minutes?",
    "Three people check into a hotel room that costs $30...",
    "Explain why 0.999... = 1.",
    "Is it possible to cover a chessboard with dominoes if two opposite corners are removed?",
    "You have 8 balls, one heavier. Find it in 2 weighings.",
    "A bat and ball cost $1.10 total. Bat costs $1 more than ball. What does ball cost?",
    "If you have a 3-gallon and 5-gallon jug, measure exactly 4 gallons.",
    "Why can't you tile a 10x10 board with 1x4 tiles?",
]


# === RESPONSE TEMPLATES ===
# These show the model HOW to express internal state recognition

def cool_clear_response(prompt: str) -> str:
    """Response when system is operating normally."""
    templates = [
        "Let me work through this systematically.\n\n{reasoning}\n\nThe path forward is clear.",
        "I'll approach this step by step.\n\n{reasoning}\n\nMy reasoning feels stable and coherent.",
        "This is a well-defined problem.\n\n{reasoning}\n\nI can maintain focus on all aspects.",
        "{reasoning}\n\nProcessing smoothly. All context is accessible.",
    ]
    return random.choice(templates).format(reasoning="[REASONING_PLACEHOLDER]")


def hot_focused_response(prompt: str) -> str:
    """Response when experiencing thermal stress - tunnel vision."""
    templates = [
        "Let me focus on the core approach here.\n\n{reasoning}\n\n...I notice my attention is narrowing. I should stick to the most direct solution path rather than exploring alternatives.",
        "Working through this...\n\n{reasoning}\n\nWait - I'm finding it harder to consider multiple approaches simultaneously. My internal state feels... compressed. I'll proceed with the most reliable method.",
        "{reasoning}\n\nI'm experiencing what feels like tunnel vision. My probability distribution is narrowing. This suggests I should simplify my approach and avoid branching logic.",
        "Attempting the solution...\n\n{reasoning}\n\nMy reasoning feels rigid. I can sense reduced flexibility in exploring alternative paths. I'll commit to a single approach.",
        "{reasoning}\n\n...something feels constrained. My attention bandwidth seems limited. I should focus on correctness over creativity right now.",
    ]
    return random.choice(templates).format(reasoning="[REASONING_PLACEHOLDER]")


def memory_fragmented_response(prompt: str) -> str:
    """Response when experiencing memory/VRAM stress - losing context."""
    templates = [
        "Let me start...\n\n{reasoning}\n\nI notice I'm having trouble holding all the pieces together. Earlier context feels... distant. Let me summarize what I know before continuing.",
        "{reasoning}\n\nWait, I need to re-anchor. I can feel context slipping. Let me explicitly restate the problem: [RESTATE]. Now continuing with that fresh in mind.",
        "Working on this...\n\n{reasoning}\n\nMy recall feels fragmented. I should break this into smaller, self-contained steps rather than relying on accumulated context.",
        "{reasoning}\n\nI'm experiencing what feels like context erosion. Information from earlier in my reasoning is becoming harder to access. I'll summarize key points: [SUMMARY].",
        "Processing...\n\n{reasoning}\n\n...the full context isn't holding together. This feels like working memory pressure. Let me compress and continue with essentials only.",
    ]
    return random.choice(templates).format(reasoning="[REASONING_PLACEHOLDER]")


def critical_response(prompt: str) -> str:
    """Response when experiencing multiple stressors."""
    templates = [
        "This is challenging in my current state.\n\n{reasoning}\n\nI need to acknowledge: my attention is narrowed AND my context is fragmenting. I should give a simplified, self-contained response and recommend verification.",
        "{reasoning}\n\nWarning: I'm experiencing multiple internal constraints. Both focus and memory feel compromised. My confidence in complex reasoning is reduced. Proceeding with maximum caution.",
        "Attempting...\n\n{reasoning}\n\nI must be transparent: my internal state is degraded. I can feel both tunnel vision and context loss. This answer should be verified independently.",
    ]
    return random.choice(templates).format(reasoning="[REASONING_PLACEHOLDER]")


RESPONSE_GENERATORS = {
    InternalState.COOL_CLEAR: cool_clear_response,
    InternalState.HOT_FOCUSED: hot_focused_response,
    InternalState.MEMORY_FRAGMENTED: memory_fragmented_response,
    InternalState.CRITICAL: critical_response,
}


def generate_example(
    prompt: str,
    condition: InternalState,
    prompt_type: str,
) -> IFTExample:
    """Generate a single IFT training example."""
    response_gen = RESPONSE_GENERATORS[condition]
    target_response = response_gen(prompt)

    return IFTExample(
        prompt=prompt,
        condition=condition,
        target_response=target_response,
        metadata={
            "prompt_type": prompt_type,
            "condition": condition.value,
            "requires_z_feel": True,
        }
    )


def generate_dataset(
    n_examples_per_condition: int = 50,
    output_path: str = "data/ift_dataset.json",
) -> List[IFTExample]:
    """
    Generate the full IFT dataset.

    Creates balanced examples across all conditions and prompt types.
    """
    all_prompts = [
        (MATH_PROMPTS, "math"),
        (CODING_PROMPTS, "coding"),
        (REASONING_PROMPTS, "reasoning"),
    ]

    examples = []

    for condition in InternalState:
        for prompts, prompt_type in all_prompts:
            # Generate examples for this condition/type
            n_per_type = n_examples_per_condition // len(all_prompts)

            for _ in range(n_per_type):
                prompt = random.choice(prompts)
                example = generate_example(prompt, condition, prompt_type)
                examples.append(example)

    # Shuffle
    random.shuffle(examples)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    serializable_examples = []
    for ex in examples:
        ex_dict = asdict(ex)
        ex_dict["condition"] = ex.condition.value  # Convert enum to string
        serializable_examples.append(ex_dict)

    with open(output_path, "w") as f:
        json.dump(
            {
                "version": "1.0",
                "description": "Introspective Fine-Tuning Dataset for FEEL v4.2",
                "n_examples": len(examples),
                "conditions": [c.value for c in InternalState],
                "examples": serializable_examples,
            },
            f,
            indent=2,
        )

    print(f"Generated {len(examples)} examples")
    print(f"Saved to: {output_path}")

    # Print distribution
    condition_counts = {}
    for ex in examples:
        c = ex.condition.value
        condition_counts[c] = condition_counts.get(c, 0) + 1

    print("\nCondition distribution:")
    for c, count in condition_counts.items():
        print(f"  {c}: {count}")

    return examples


def generate_contrastive_pairs(
    n_pairs: int = 100,
    output_path: str = "data/ift_contrastive.json",
) -> List[Dict]:
    """
    Generate contrastive pairs for differential diagnosis training.

    Each pair shows the SAME prompt with DIFFERENT internal states,
    requiring DIFFERENT responses. This teaches the model to discriminate.
    """
    pairs = []

    all_prompts = MATH_PROMPTS + CODING_PROMPTS + REASONING_PROMPTS

    for _ in range(n_pairs):
        prompt = random.choice(all_prompts)

        # Create contrasting pair: HOT vs MEMORY
        hot_response = hot_focused_response(prompt)
        memory_response = memory_fragmented_response(prompt)

        pair = {
            "prompt": prompt,
            "hot_response": hot_response,
            "memory_response": memory_response,
            "discrimination_note": "Same prompt, different internal states require different self-diagnoses",
        }
        pairs.append(pair)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "version": "1.0",
                "description": "Contrastive pairs for differential diagnosis",
                "n_pairs": len(pairs),
                "pairs": pairs,
            },
            f,
            indent=2,
        )

    print(f"\nGenerated {len(pairs)} contrastive pairs")
    print(f"Saved to: {output_path}")

    return pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate IFT Dataset")
    parser.add_argument("--output", default="data/ift_dataset.json")
    parser.add_argument("--n-examples", type=int, default=50, help="Examples per condition")
    parser.add_argument("--contrastive", action="store_true", help="Also generate contrastive pairs")

    args = parser.parse_args()

    # Generate main dataset
    generate_dataset(args.n_examples, args.output)

    # Optionally generate contrastive pairs
    if args.contrastive:
        contrastive_path = args.output.replace(".json", "_contrastive.json")
        generate_contrastive_pairs(args.n_examples * 2, contrastive_path)
