#!/usr/bin/env python3
"""
z50 Dataset Augmentation Script
================================

Downloads and combines:
- OASST1 (instruction/chat anchor) - Apache 2.0
- GSM8K (math reasoning) - MIT
- NuminaMath-CoT (math chain-of-thought) - Apache 2.0

Creates a difficulty-tagged dataset for embodied training:
- easy: simple chat/instruction (skip more)
- medium: moderate reasoning
- hard: complex math/logic (run full compute)

Author: FEEL Research Team
Date: 2026-01-17
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[WARN] datasets library not installed. Run: pip install datasets")

# Difficulty classification based on prompt characteristics
def classify_difficulty(prompt: str, source: str) -> str:
    """Classify prompt difficulty for compute allocation."""
    prompt_lower = prompt.lower()

    # Hard: math, logic, complex reasoning
    if source in ["gsm8k", "math", "numina"]:
        return "hard"

    # Check for reasoning indicators
    hard_indicators = [
        "calculate", "solve", "prove", "derive", "analyze",
        "step by step", "show your work", "explain why",
        "what is the probability", "how many ways"
    ]
    if any(ind in prompt_lower for ind in hard_indicators):
        return "hard"

    # Medium: coding, explanation requests
    medium_indicators = [
        "implement", "write a function", "code", "program",
        "explain", "describe in detail", "compare", "contrast"
    ]
    if any(ind in prompt_lower for ind in medium_indicators):
        return "medium"

    # Easy: simple questions, chat
    return "easy"


def load_oasst1(max_samples: int = 500) -> List[Dict]:
    """Load OASST1 dataset (instruction/chat anchor)."""
    print(f"Loading OASST1 (max {max_samples} samples)...")

    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as e:
        print(f"  [ERROR] Failed to load OASST1: {e}")
        return []

    examples = []
    # Filter for English, prompter messages
    for item in ds:
        if item.get("lang") == "en" and item.get("role") == "prompter":
            prompt = item.get("text", "").strip()
            if len(prompt) > 20 and len(prompt) < 500:
                examples.append({
                    "prompt": prompt,
                    "source": "oasst1",
                    "difficulty": classify_difficulty(prompt, "oasst1"),
                    "condition": random.choice(["normal", "hot", "cool"]),
                })
                if len(examples) >= max_samples:
                    break

    print(f"  Loaded {len(examples)} OASST1 examples")
    return examples


def load_gsm8k(max_samples: int = 300) -> List[Dict]:
    """Load GSM8K dataset (math reasoning)."""
    print(f"Loading GSM8K (max {max_samples} samples)...")

    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print(f"  [ERROR] Failed to load GSM8K: {e}")
        return []

    examples = []
    for item in ds:
        prompt = item.get("question", "").strip()
        if len(prompt) > 20:
            examples.append({
                "prompt": prompt,
                "source": "gsm8k",
                "difficulty": "hard",  # All GSM8K is reasoning
                "condition": random.choice(["hot", "normal"]),  # Math = hot
                "answer": item.get("answer", ""),
            })
            if len(examples) >= max_samples:
                break

    print(f"  Loaded {len(examples)} GSM8K examples")
    return examples


def load_numina_cot(max_samples: int = 200) -> List[Dict]:
    """Load NuminaMath-CoT dataset (chain-of-thought math)."""
    print(f"Loading NuminaMath-CoT (max {max_samples} samples)...")

    try:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    except Exception as e:
        print(f"  [ERROR] Failed to load NuminaMath: {e}")
        return []

    examples = []
    for item in ds:
        prompt = item.get("problem", "").strip()
        if len(prompt) > 20 and len(prompt) < 800:
            examples.append({
                "prompt": prompt,
                "source": "numina",
                "difficulty": "hard",
                "condition": random.choice(["hot", "normal"]),
                "solution": item.get("solution", ""),
            })
            if len(examples) >= max_samples:
                break

    print(f"  Loaded {len(examples)} NuminaMath examples")
    return examples


def load_existing_dataset(path: str) -> List[Dict]:
    """Load existing FEEL dataset."""
    print(f"Loading existing dataset: {path}")

    with open(path) as f:
        data = json.load(f)

    examples = []
    for item in data.get("examples", []):
        examples.append({
            "prompt": item["prompt"],
            "source": "feel_original",
            "difficulty": classify_difficulty(item["prompt"], "feel"),
            "condition": item.get("condition", "normal"),
            "target_response": item.get("target_response", ""),
        })

    print(f"  Loaded {len(examples)} existing examples")
    return examples


def create_augmented_dataset(
    output_path: str,
    oasst_samples: int = 500,
    gsm8k_samples: int = 300,
    numina_samples: int = 200,
    include_original: bool = True,
):
    """Create augmented dataset for z50 training."""

    all_examples = []

    # Load existing FEEL dataset
    if include_original and Path("data/ift_dataset.json").exists():
        all_examples.extend(load_existing_dataset("data/ift_dataset.json"))

    # Load new datasets
    if HF_AVAILABLE:
        all_examples.extend(load_oasst1(oasst_samples))
        all_examples.extend(load_gsm8k(gsm8k_samples))
        all_examples.extend(load_numina_cot(numina_samples))
    else:
        print("[WARN] HuggingFace datasets not available, using original data only")

    # Shuffle
    random.shuffle(all_examples)

    # Statistics
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    source_counts = {}
    for ex in all_examples:
        difficulty_counts[ex["difficulty"]] = difficulty_counts.get(ex["difficulty"], 0) + 1
        source_counts[ex["source"]] = source_counts.get(ex["source"], 0) + 1

    # Create output
    output = {
        "version": "z50_augmented_v1",
        "description": "Augmented dataset for z50 quality-guard training",
        "n_examples": len(all_examples),
        "difficulty_distribution": difficulty_counts,
        "source_distribution": source_counts,
        "examples": all_examples,
    }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Created augmented dataset: {output_path}")
    print(f"Total examples: {len(all_examples)}")
    print(f"Difficulty distribution: {difficulty_counts}")
    print(f"Source distribution: {source_counts}")
    print(f"{'='*60}")

    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="z50 Dataset Augmentation")
    parser.add_argument("--output", default="data/z50_augmented.json", help="Output path")
    parser.add_argument("--oasst", type=int, default=500, help="OASST1 samples")
    parser.add_argument("--gsm8k", type=int, default=300, help="GSM8K samples")
    parser.add_argument("--numina", type=int, default=200, help="NuminaMath samples")
    parser.add_argument("--no-original", action="store_true", help="Skip original dataset")
    args = parser.parse_args()

    create_augmented_dataset(
        args.output,
        oasst_samples=args.oasst,
        gsm8k_samples=args.gsm8k,
        numina_samples=args.numina,
        include_original=not args.no_original,
    )


if __name__ == "__main__":
    main()
