#!/usr/bin/env python3
"""
z50 Quick Dataset Builder - Streaming approach
===============================================

Simplified version that uses streaming to avoid memory issues.
Uses existing refined_train.jsonl as base + adds reasoning prompts.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def load_gsm8k_fast(max_samples: int = 400) -> List[Dict]:
    """Load GSM8K quickly."""
    print(f"Loading GSM8K (max {max_samples})...")
    examples = []

    try:
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        for item in ds:
            examples.append({
                "input": item["question"],
                "output": item["answer"],
                "difficulty": "hard",
                "source": "gsm8k",
                "category": "math"
            })
            if len(examples) >= max_samples:
                break
    except Exception as e:
        print(f"  [ERROR] {e}")

    print(f"  Loaded {len(examples)} GSM8K examples")
    return examples


def load_math_fast(max_samples: int = 300) -> List[Dict]:
    """Load MATH quickly from algebra subset."""
    print(f"Loading MATH algebra (max {max_samples})...")
    examples = []

    try:
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train", streaming=True)
        for item in ds:
            examples.append({
                "input": item["problem"],
                "output": item["solution"],
                "difficulty": "hard",
                "source": "math",
                "category": "reasoning"
            })
            if len(examples) >= max_samples:
                break
    except Exception as e:
        print(f"  [ERROR] {e}")

    print(f"  Loaded {len(examples)} MATH examples")
    return examples


def load_existing_data(path: str = "data/ouroboros/refined_train.jsonl") -> List[Dict]:
    """Load existing refined training data."""
    print(f"Loading existing data: {path}")
    examples = []

    try:
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                examples.append({
                    "input": item.get("input", ""),
                    "output": item.get("output", ""),
                    "difficulty": "medium" if item.get("is_stressed") else "easy",
                    "source": "refined",
                    "category": "chat",
                    "stress_level": item.get("stress_level", 0.5),
                })
    except Exception as e:
        print(f"  [ERROR] {e}")

    print(f"  Loaded {len(examples)} existing examples")
    return examples


def build_dataset():
    """Build combined dataset."""
    print("="*60)
    print("z50 Quick Dataset Builder")
    print("="*60)

    all_examples = []

    # Load existing data (50%)
    existing = load_existing_data()
    all_examples.extend(existing[:1000])  # Cap at 1000

    # Add reasoning (GSM8K + MATH) (35%)
    if HF_AVAILABLE:
        all_examples.extend(load_gsm8k_fast(400))
        all_examples.extend(load_math_fast(300))

    # Shuffle
    random.shuffle(all_examples)

    # Stats
    stats = {"total": len(all_examples), "by_source": {}, "by_difficulty": {}}
    for ex in all_examples:
        stats["by_source"][ex["source"]] = stats["by_source"].get(ex["source"], 0) + 1
        stats["by_difficulty"][ex["difficulty"]] = stats["by_difficulty"].get(ex["difficulty"], 0) + 1

    # Save
    output_path = "data/z50_combined.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print("\n" + "="*60)
    print(f"Saved: {output_path}")
    print(f"Total: {stats['total']}")
    print(f"Sources: {stats['by_source']}")
    print(f"Difficulty: {stats['by_difficulty']}")
    print("="*60)

    return output_path


if __name__ == "__main__":
    build_dataset()
