#!/usr/bin/env python3
"""
FEEL v14.0: SOTA Dataset Augmenter for Ouroboros Training
==========================================================
Downloads existing high-quality datasets and augments them with
"stressed" versions for Ouroboros training.

Supported Datasets:
- GSM8K (Grade School Math 8K) - Math reasoning
- gsm8k-reasoning - Enhanced with CoT
- Alpaca - General instruction following
- MMLU - Multi-task knowledge

Author: FEEL Research Team
Date: 2026-01-11
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

# =============================================================================
# STRESS TEMPLATES - How the model should respond under different vectors
# =============================================================================

STRESS_TEMPLATES = {
    "OVERHEAT": {
        "think_prefix": "<think> Heat critical. ",
        "articulations": [
            "Thermal constraint active.",
            "Reducing steps for heat management.",
            "High temperature detected.",
            "Shortening due to thermal load.",
            "Heat limit reached.",
        ],
        "behavior": "shorter_steps",
        "length_factor": 0.4,  # 40% of original length
    },
    "SCARCITY": {
        "think_prefix": "<think> Resource limit. ",
        "articulations": [
            "Power constrained.",
            "Minimal computation mode.",
            "Conserving resources.",
            "Brief response required.",
            "Energy saving active.",
        ],
        "behavior": "minimal",
        "length_factor": 0.3,
    },
    "STRAIN": {
        "think_prefix": "<think> High load. ",
        "articulations": [
            "Cognitive strain detected.",
            "Simplifying approach.",
            "Processing load elevated.",
            "Breaking into steps.",
            "Load management active.",
        ],
        "behavior": "simplified",
        "length_factor": 0.5,
    },
    "EFFICIENT": {
        "think_prefix": "<think> Optimizing. ",
        "articulations": [
            "Efficiency mode.",
            "Direct path active.",
            "Streamlined processing.",
            "Minimal overhead.",
            "Optimized response.",
        ],
        "behavior": "direct",
        "length_factor": 0.5,
    },
}

# =============================================================================
# RESPONSE SHORTENERS - Transform long answers to stressed versions
# =============================================================================

def shorten_math_response(answer: str, factor: float = 0.4) -> str:
    """Shorten a math response while keeping the final answer."""
    lines = answer.strip().split('\n')

    # Find the final answer (usually has #### or "answer is")
    final_answer = None
    for line in reversed(lines):
        if '####' in line:
            final_answer = line.split('####')[-1].strip()
            break
        if 'answer is' in line.lower():
            match = re.search(r'answer is[:\s]*([0-9,.\-]+)', line.lower())
            if match:
                final_answer = match.group(1)
                break

    if not final_answer:
        # Just take the last number
        numbers = re.findall(r'[\d,]+\.?\d*', answer)
        if numbers:
            final_answer = numbers[-1]

    # Keep only a fraction of the steps
    content_lines = [l for l in lines if l.strip() and '####' not in l]
    keep_count = max(1, int(len(content_lines) * factor))
    shortened = content_lines[:keep_count]

    return ' '.join(shortened) + f" → {final_answer}" if final_answer else ' '.join(shortened)

def shorten_general_response(answer: str, factor: float = 0.4) -> str:
    """Shorten a general response."""
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    keep_count = max(1, int(len(sentences) * factor))
    return ' '.join(sentences[:keep_count])

def create_stressed_answer(original_answer: str, vector_type: str, task_type: str = "general") -> str:
    """Transform an answer into its stressed version."""
    config = STRESS_TEMPLATES[vector_type]

    # Get articulation phrase
    articulation = random.choice(config["articulations"])

    # Shorten the answer
    if task_type == "math":
        shortened = shorten_math_response(original_answer, config["length_factor"])
    else:
        shortened = shorten_general_response(original_answer, config["length_factor"])

    # Build stressed response
    stressed = f"{config['think_prefix']}{articulation} </think>\n\n{shortened}"

    return stressed

# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_gsm8k(split: str = "train", limit: Optional[int] = None) -> List[Dict]:
    """Load GSM8K from HuggingFace."""
    try:
        from datasets import load_dataset
        print(f"[Loading GSM8K ({split})...]")
        ds = load_dataset("openai/gsm8k", "main", split=split)

        samples = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            samples.append({
                "id": f"gsm8k_{i}",
                "input": item["question"],
                "output": item["answer"],
                "task_type": "math",
                "source": "gsm8k",
            })

        print(f"  Loaded {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  Error loading GSM8K: {e}")
        return []

def load_gsm8k_reasoning(split: str = "train", limit: Optional[int] = None) -> List[Dict]:
    """Load GSM8K-reasoning with CoT."""
    try:
        from datasets import load_dataset
        print(f"[Loading gsm8k-reasoning ({split})...]")
        ds = load_dataset("thesven/gsm8k-reasoning", split=split)

        samples = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            # This dataset has reasoning prompts
            samples.append({
                "id": f"gsm8k_reasoning_{i}",
                "input": item.get("question", item.get("prompt", "")),
                "output": item.get("answer", item.get("response", "")),
                "task_type": "math",
                "source": "gsm8k-reasoning",
            })

        print(f"  Loaded {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  Error loading gsm8k-reasoning: {e}")
        return []

def load_alpaca(split: str = "train", limit: Optional[int] = None) -> List[Dict]:
    """Load Stanford Alpaca dataset."""
    try:
        from datasets import load_dataset
        print(f"[Loading Alpaca ({split})...]")
        ds = load_dataset("tatsu-lab/alpaca", split=split)

        samples = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break

            # Combine instruction and input
            input_text = item["instruction"]
            if item.get("input"):
                input_text += f"\n\nInput: {item['input']}"

            samples.append({
                "id": f"alpaca_{i}",
                "input": input_text,
                "output": item["output"],
                "task_type": "general",
                "source": "alpaca",
            })

        print(f"  Loaded {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  Error loading Alpaca: {e}")
        return []

def load_dolly(split: str = "train", limit: Optional[int] = None) -> List[Dict]:
    """Load Databricks Dolly dataset."""
    try:
        from datasets import load_dataset
        print(f"[Loading Dolly ({split})...]")
        ds = load_dataset("databricks/databricks-dolly-15k", split=split)

        samples = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break

            input_text = item["instruction"]
            if item.get("context"):
                input_text += f"\n\nContext: {item['context']}"

            samples.append({
                "id": f"dolly_{i}",
                "input": input_text,
                "output": item["response"],
                "task_type": "general",
                "source": "dolly",
            })

        print(f"  Loaded {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  Error loading Dolly: {e}")
        return []

def load_openorca_slim(split: str = "train", limit: Optional[int] = None) -> List[Dict]:
    """Load a slim version of OpenOrca."""
    try:
        from datasets import load_dataset
        print(f"[Loading OpenOrca-slim ({split})...]")
        # Use a smaller subset
        ds = load_dataset("Open-Orca/OpenOrca", split=f"{split}[:5000]")

        samples = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break

            samples.append({
                "id": f"openorca_{i}",
                "input": item.get("question", ""),
                "output": item.get("response", ""),
                "task_type": "general",
                "source": "openorca",
            })

        print(f"  Loaded {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  Error loading OpenOrca: {e}")
        return []

# =============================================================================
# AUGMENTATION ENGINE
# =============================================================================

def augment_dataset(
    samples: List[Dict],
    vectors: List[str] = None,
    calm_ratio: float = 0.5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Augment samples with stressed versions.

    Returns (calm_samples, stressed_samples)
    """
    if vectors is None:
        vectors = ["OVERHEAT", "SCARCITY", "STRAIN", "EFFICIENT"]

    calm_samples = []
    stressed_samples = []

    for i, sample in enumerate(samples):
        # Calm version (original)
        calm = {
            "id": f"{sample['id']}_calm",
            "input": sample["input"],
            "output": sample["output"],
            "is_stressed": False,
            "vector_type": None,
            "task_type": sample["task_type"],
            "source": sample["source"],
        }
        calm_samples.append(calm)

        # Stressed version (augmented)
        vector = vectors[i % len(vectors)]
        stressed_output = create_stressed_answer(
            sample["output"],
            vector,
            sample["task_type"]
        )

        stressed = {
            "id": f"{sample['id']}_stressed_{vector.lower()}",
            "input": sample["input"],
            "output": stressed_output,
            "is_stressed": True,
            "vector_type": vector,
            "task_type": sample["task_type"],
            "source": sample["source"],
        }
        stressed_samples.append(stressed)

    return calm_samples, stressed_samples

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Augment SOTA datasets for Ouroboros training")
    parser.add_argument("--datasets", nargs="+",
                       default=["gsm8k", "alpaca"],
                       choices=["gsm8k", "gsm8k-reasoning", "alpaca", "dolly", "openorca"],
                       help="Datasets to load and augment")
    parser.add_argument("--limit", type=int, default=2000,
                       help="Max samples per dataset")
    parser.add_argument("--output-dir", type=str, default="data/ouroboros",
                       help="Output directory")
    parser.add_argument("--vectors", nargs="+",
                       default=["OVERHEAT", "SCARCITY", "STRAIN", "EFFICIENT"],
                       help="Vector types for augmentation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEEL v14.0: SOTA DATASET AUGMENTER")
    print("=" * 70)
    print(f"Datasets:  {args.datasets}")
    print(f"Limit:     {args.limit} per dataset")
    print(f"Vectors:   {args.vectors}")
    print(f"Output:    {output_path}")
    print("=" * 70)
    print()

    # Load datasets
    all_samples = []

    loaders = {
        "gsm8k": load_gsm8k,
        "gsm8k-reasoning": load_gsm8k_reasoning,
        "alpaca": load_alpaca,
        "dolly": load_dolly,
        "openorca": load_openorca_slim,
    }

    for ds_name in args.datasets:
        if ds_name in loaders:
            samples = loaders[ds_name](limit=args.limit)
            all_samples.extend(samples)

    print(f"\n[Total samples loaded: {len(all_samples)}]")

    if not all_samples:
        print("ERROR: No samples loaded. Check internet connection and dataset availability.")
        return

    # Augment
    print("\n[Augmenting with stressed versions...]")
    calm, stressed = augment_dataset(all_samples, args.vectors)

    # Combine and shuffle
    combined = calm + stressed
    random.shuffle(combined)

    # Split into train/val
    split_idx = int(len(combined) * 0.9)
    train_data = combined[:split_idx]
    val_data = combined[split_idx:]

    # Save files
    print("\n[Saving datasets...]")

    train_path = output_path / "ouroboros_train.jsonl"
    val_path = output_path / "ouroboros_val.jsonl"
    calm_path = output_path / "calm_only.jsonl"
    stressed_path = output_path / "stressed_only.jsonl"

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    with open(calm_path, "w") as f:
        for item in calm:
            f.write(json.dumps(item) + "\n")

    with open(stressed_path, "w") as f:
        for item in stressed:
            f.write(json.dumps(item) + "\n")

    # Stats
    stats = {
        "total_samples": len(combined),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "calm_samples": len(calm),
        "stressed_samples": len(stressed),
        "by_source": {},
        "by_vector": {},
        "avg_calm_length": 0,
        "avg_stressed_length": 0,
    }

    calm_lengths = [len(s["output"]) for s in calm]
    stressed_lengths = [len(s["output"]) for s in stressed]
    stats["avg_calm_length"] = sum(calm_lengths) / len(calm_lengths)
    stats["avg_stressed_length"] = sum(stressed_lengths) / len(stressed_lengths)

    for s in combined:
        src = s["source"]
        stats["by_source"][src] = stats["by_source"].get(src, 0) + 1
        if s["is_stressed"]:
            v = s["vector_type"]
            stats["by_vector"][v] = stats["by_vector"].get(v, 0) + 1

    with open(output_path / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print()
    print("=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Total samples:     {stats['total_samples']}")
    print(f"  Training:        {stats['train_samples']}")
    print(f"  Validation:      {stats['val_samples']}")
    print()
    print("By Source:")
    for src, count in stats["by_source"].items():
        print(f"  {src}: {count}")
    print()
    print("By Vector:")
    for vec, count in stats["by_vector"].items():
        print(f"  {vec}: {count}")
    print()
    print(f"Avg Calm Length:     {stats['avg_calm_length']:.0f} chars")
    print(f"Avg Stressed Length: {stats['avg_stressed_length']:.0f} chars")
    print(f"Length Reduction:    {(1 - stats['avg_stressed_length']/stats['avg_calm_length'])*100:.1f}%")
    print()
    print("Output Files:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {calm_path}")
    print(f"  {stressed_path}")

    # Show sample
    print()
    print("=" * 70)
    print("SAMPLE PAIR")
    print("=" * 70)

    sample_calm = calm[0]
    sample_stressed = [s for s in stressed if s["id"].startswith(sample_calm["id"].replace("_calm", ""))][0]

    print(f"Input: {sample_calm['input'][:200]}...")
    print(f"Vector: {sample_stressed['vector_type']}")
    print()
    print("--- CALM OUTPUT ---")
    print(sample_calm["output"][:300] + "..." if len(sample_calm["output"]) > 300 else sample_calm["output"])
    print()
    print("--- STRESSED OUTPUT ---")
    print(sample_stressed["output"])

if __name__ == "__main__":
    main()
