#!/usr/bin/env python3
"""
Augment IFT Dataset with Action Tokens

This script takes the existing IFT dataset and adds action tokens to the targets,
so the model learns to EMIT regulation decisions, not just describe its state.

Before: Target = "I notice my attention narrowing..."
After:  Target = "<|FEEL_HOT|> I notice my attention narrowing..."

The model learns that when it feels a certain way (via z_feel injection),
it should output the corresponding action token FIRST, then describe.

Usage:
    python scripts/augment_ift_with_actions.py \
        --in data/ift_dataset.json \
        --out data/ift_dataset_with_actions.json
"""

import json
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from action_tokens import (
    FEEL_TOKENS,
    FeelAction,
    get_action_for_condition,
    condition_to_token,
)


def augment_example(example: dict) -> dict:
    """
    Augment a single example with action token.

    The action token is prepended to the target response,
    teaching the model to emit the decision FIRST.
    """
    condition = example.get("condition", "cool_clear")
    target = example.get("target_response", "")

    # Get the appropriate action token
    action_token = condition_to_token(condition)

    # Prepend action token to target
    augmented_target = f"{action_token} {target}"

    # Create augmented example
    augmented = example.copy()
    augmented["target_response"] = augmented_target
    augmented["action_token"] = action_token
    augmented["original_target"] = target

    return augmented


def augment_dataset(input_path: str, output_path: str) -> dict:
    """
    Augment entire dataset with action tokens.
    """
    print(f"Loading dataset from: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    examples = data.get("examples", [])
    print(f"Found {len(examples)} examples")

    # Augment each example
    augmented_examples = []
    action_counts = {}

    for ex in examples:
        augmented = augment_example(ex)
        augmented_examples.append(augmented)

        # Track action distribution
        action = augmented["action_token"]
        action_counts[action] = action_counts.get(action, 0) + 1

    # Create output dataset
    output_data = {
        "description": data.get("description", "") + " (augmented with action tokens)",
        "version": "with_actions",
        "conditions": data.get("conditions", []),
        "action_tokens": list(FEEL_TOKENS.values()),
        "n_examples": len(augmented_examples),
        "action_distribution": action_counts,
        "examples": augmented_examples,
    }

    # Save
    print(f"Saving augmented dataset to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nAction token distribution:")
    for token, count in sorted(action_counts.items()):
        print(f"  {token}: {count}")

    return output_data


def add_more_conditions(output_path: str):
    """
    Optionally add examples for WARM and VERY_HOT conditions
    that aren't in the original dataset.
    """
    with open(output_path, 'r') as f:
        data = json.load(f)

    examples = data["examples"]

    # Templates for new conditions
    warm_templates = [
        {
            "prompt": "Explain the concept of recursion.",
            "condition": "warm",
            "target_response": "I'm beginning to notice a slight warmth in my processing. Let me focus on the essentials. Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. It requires a base case to stop and a recursive case that moves toward the base case.",
        },
        {
            "prompt": "What is machine learning?",
            "condition": "warm",
            "target_response": "There's a subtle thermal pressure building. I'll keep this clear and direct. Machine learning is a subset of AI where systems learn patterns from data rather than being explicitly programmed. The three main types are supervised, unsupervised, and reinforcement learning.",
        },
    ]

    very_hot_templates = [
        {
            "prompt": "Describe quantum computing.",
            "condition": "very_hot",
            "target_response": "I need to rest. Thermal stress is significant. Brief answer: quantum computing uses qubits that can be 0, 1, or both simultaneously. This enables parallel processing of many states at once. I should pause here.",
        },
        {
            "prompt": "What is blockchain?",
            "condition": "very_hot",
            "target_response": "I must conserve processing. High thermal load detected. Blockchain: distributed ledger, cryptographic links, consensus mechanisms. Decentralized trust. I need to stop and cool down.",
        },
    ]

    # Augment new templates
    for template in warm_templates + very_hot_templates:
        augmented = augment_example(template)
        examples.append(augmented)

    data["examples"] = examples
    data["n_examples"] = len(examples)

    # Recalculate action distribution
    action_counts = {}
    for ex in examples:
        action = ex.get("action_token", "<|FEEL_OK|>")
        action_counts[action] = action_counts.get(action, 0) + 1
    data["action_distribution"] = action_counts

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nAdded {len(warm_templates) + len(very_hot_templates)} new examples")
    print(f"Total examples: {len(examples)}")


def main():
    parser = argparse.ArgumentParser(description="Augment IFT dataset with action tokens")
    parser.add_argument("--in", dest="input", default="data/ift_dataset.json",
                       help="Input dataset path")
    parser.add_argument("--out", dest="output", default="data/ift_dataset_with_actions.json",
                       help="Output dataset path")
    parser.add_argument("--add-conditions", action="store_true",
                       help="Add examples for WARM and VERY_HOT conditions")

    args = parser.parse_args()

    # Augment dataset
    augment_dataset(args.input, args.output)

    # Optionally add more conditions
    if args.add_conditions:
        add_more_conditions(args.output)

    print("\nDone! Dataset augmented with action tokens.")
    print("The model will now learn to EMIT regulation decisions via tokens.")


if __name__ == "__main__":
    main()
