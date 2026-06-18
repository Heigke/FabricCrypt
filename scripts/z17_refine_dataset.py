#!/usr/bin/env python3
"""
FEEL v17: Refined Dataset with Continuous Stress & Articulation
================================================================
Creates improved training data with:
1. Continuous stress values (not just buckets)
2. Explicit articulation tags ([HEAT CRITICAL], [COOL], etc.)
3. Gradual length reduction (not binary)

Author: FEEL Research Team
Date: 2026-01-11
"""

import json
import random
from pathlib import Path
from typing import List, Dict

# Articulation templates based on stress level
ARTICULATION_TEMPLATES = {
    "very_low": [  # 0.0-0.2
        "System nominal. Full reasoning engaged.",
        "Resources abundant. Taking time to think thoroughly.",
        "Cool and efficient. Let me work through this step by step.",
    ],
    "low": [  # 0.2-0.4
        "Operating normally. Proceeding with analysis.",
        "Comfortable conditions. Standard reasoning mode.",
        "Adequate resources. Processing request.",
    ],
    "medium": [  # 0.4-0.6
        "Moderate load detected. Balancing thoroughness with efficiency.",
        "Some resource pressure. Condensing reasoning.",
        "Warming up. Being more concise.",
    ],
    "high": [  # 0.6-0.8
        "[HEAT WARNING] Shortening response for efficiency.",
        "High load. Prioritizing essential information.",
        "Resource constrained. Direct answer follows.",
    ],
    "critical": [  # 0.8-1.0
        "[HEAT CRITICAL] Minimal response mode.",
        "[THERMAL LIMIT] Essential output only.",
        "[EFFICIENCY MODE] Brief answer.",
    ],
}

def get_stress_category(stress: float) -> str:
    """Map continuous stress to category."""
    if stress < 0.2:
        return "very_low"
    elif stress < 0.4:
        return "low"
    elif stress < 0.6:
        return "medium"
    elif stress < 0.8:
        return "high"
    else:
        return "critical"

def get_articulation(stress: float) -> str:
    """Get appropriate articulation for stress level."""
    category = get_stress_category(stress)
    return random.choice(ARTICULATION_TEMPLATES[category])

def scale_output(full_output: str, stress: float) -> str:
    """Scale output length based on stress (continuous, not binary)."""
    # At stress 0.0: full output
    # At stress 1.0: ~20% of output
    # Linear interpolation between
    keep_ratio = 1.0 - (stress * 0.8)  # 1.0 -> 0.2

    if '####' in full_output:
        # Math problem: keep the final answer, scale the reasoning
        parts = full_output.split('####')
        reasoning = parts[0].strip()
        answer = parts[-1].strip()

        # Scale reasoning
        sentences = reasoning.split('. ')
        keep_count = max(1, int(len(sentences) * keep_ratio))
        scaled_reasoning = '. '.join(sentences[:keep_count])
        if scaled_reasoning and not scaled_reasoning.endswith('.'):
            scaled_reasoning += '.'

        return f"{scaled_reasoning}\n\nThe answer is {answer}."
    else:
        # General text: scale by character count
        target_len = max(20, int(len(full_output) * keep_ratio))
        if len(full_output) <= target_len:
            return full_output

        # Try to cut at sentence boundary
        truncated = full_output[:target_len]
        last_period = truncated.rfind('.')
        if last_period > target_len * 0.5:
            return truncated[:last_period + 1]
        return truncated + "..."

def create_refined_sample(base_sample: Dict, stress: float) -> Dict:
    """Create a training sample with specific stress level and articulation."""
    articulation = get_articulation(stress)
    scaled_output = scale_output(base_sample['output'], stress)

    return {
        "input": base_sample['input'],
        "output": scaled_output,
        "stress_level": stress,  # Exact stress value
        "articulation": articulation,
        "is_stressed": stress > 0.5,  # For backward compatibility
        "category": get_stress_category(stress),
    }

def load_base_samples(path: str) -> List[Dict]:
    """Load base samples from JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            # Only use "calm" samples as base (they have full output)
            if not item.get('is_stressed', False):
                samples.append(item)
    return samples

def create_refined_dataset(
    calm_path: str = "data/ouroboros/calm_only.jsonl",
    output_train: str = "data/ouroboros/refined_train.jsonl",
    output_val: str = "data/ouroboros/refined_val.jsonl",
    samples_per_stress: int = 100,
):
    """Create refined dataset with continuous stress distribution."""

    print("=" * 70)
    print("FEEL v17: REFINED DATASET CREATION")
    print("=" * 70)

    # Load base calm samples
    print(f"\n[1] Loading base samples from {calm_path}...")
    base_samples = load_base_samples(calm_path)
    print(f"    Loaded {len(base_samples)} base samples")

    # Define stress distribution (more samples in middle for interpolation)
    stress_values = [
        0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
        0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
        0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    ]

    print(f"\n[2] Generating samples across {len(stress_values)} stress levels...")

    all_samples = []
    for stress in stress_values:
        # Sample from base and create refined versions
        selected = random.sample(base_samples, min(samples_per_stress, len(base_samples)))
        for base in selected:
            refined = create_refined_sample(base, stress)
            all_samples.append(refined)

        category = get_stress_category(stress)
        print(f"    Stress {stress:.2f} ({category:<10}): {len(selected)} samples")

    # Shuffle
    random.shuffle(all_samples)

    # Split train/val (90/10)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Save
    print(f"\n[3] Saving datasets...")

    Path(output_train).parent.mkdir(parents=True, exist_ok=True)

    with open(output_train, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"    Train: {len(train_samples)} samples -> {output_train}")

    with open(output_val, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"    Val:   {len(val_samples)} samples -> {output_val}")

    # Stats
    print(f"\n[4] Dataset Statistics:")
    stress_dist = {}
    for s in all_samples:
        cat = s['category']
        stress_dist[cat] = stress_dist.get(cat, 0) + 1

    for cat, count in sorted(stress_dist.items()):
        print(f"    {cat:<12}: {count:>5} samples ({100*count/len(all_samples):.1f}%)")

    # Example samples
    print(f"\n[5] Example Samples:")
    for stress in [0.1, 0.5, 0.9]:
        example = next(s for s in all_samples if abs(s['stress_level'] - stress) < 0.05)
        print(f"\n    --- Stress {stress:.1f} ({example['category']}) ---")
        print(f"    Articulation: {example['articulation']}")
        print(f"    Output preview: {example['output'][:100]}...")

    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)

    return train_samples, val_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create refined dataset")
    parser.add_argument("--input", type=str, default="data/ouroboros/calm_only.jsonl")
    parser.add_argument("--train-out", type=str, default="data/ouroboros/refined_train.jsonl")
    parser.add_argument("--val-out", type=str, default="data/ouroboros/refined_val.jsonl")
    parser.add_argument("--samples-per-stress", type=int, default=100)
    args = parser.parse_args()

    create_refined_dataset(
        calm_path=args.input,
        output_train=args.train_out,
        output_val=args.val_out,
        samples_per_stress=args.samples_per_stress,
    )
