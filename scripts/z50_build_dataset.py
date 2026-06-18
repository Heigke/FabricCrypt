#!/usr/bin/env python3
"""
z50 Dataset Builder - Partner-Safe Dataset Stack
=================================================

Creates a dataset aligned with DeepSeek-R1-Distill-Qwen-7B:

RECIPE (partner-safe, properly licensed):
- 50% OASST1 (Apache 2.0) - instruction/chat anchor
- 35% GSM8K + MATH + NuminaMath (MIT/Apache) - reasoning/hard prompts
- 15% HH-RLHF (MIT) - preference pairs for quality guard

KEY FEATURES:
1. Uses tokenizer.apply_chat_template() for proper formatting
2. Tags prompts as easy/medium/hard for compute allocation
3. Stores as [{role, content}, ...] message format
4. Avoids problematic datasets (OpenOrca, UltraChat)

Author: FEEL Research Team
Date: 2026-01-17
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[ERROR] Required: pip install datasets transformers")
    sys.exit(1)


@dataclass
class PromptExample:
    """Single training example."""
    messages: List[Dict[str, str]]  # [{role: "user", content: "..."}, ...]
    difficulty: str  # easy, medium, hard
    source: str  # oasst1, gsm8k, math, numina, hh_rlhf
    category: str  # chat, math, reasoning, preference


def classify_difficulty(text: str, source: str) -> str:
    """Classify prompt difficulty for compute allocation."""
    text_lower = text.lower()

    # Math/reasoning sources are always hard
    if source in ["gsm8k", "math", "numina"]:
        return "hard"

    # Hard indicators
    hard_keywords = [
        "calculate", "solve", "prove", "derive", "analyze",
        "step by step", "show your work", "explain why",
        "probability", "how many ways", "algorithm"
    ]
    if any(kw in text_lower for kw in hard_keywords):
        return "hard"

    # Medium indicators
    medium_keywords = [
        "implement", "write a function", "code", "program",
        "explain in detail", "compare", "contrast", "describe"
    ]
    if any(kw in text_lower for kw in medium_keywords):
        return "medium"

    # Easy by default
    return "easy"


def load_oasst1(max_samples: int = 1000) -> List[PromptExample]:
    """Load OASST1 dataset (Apache 2.0 license)."""
    print(f"\n[1/4] Loading OASST1 (max {max_samples})...")

    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []

    examples = []

    # Build conversation trees
    msg_by_id = {m["message_id"]: m for m in ds}

    for msg in ds:
        # Get root prompter messages in English
        if msg.get("lang") != "en":
            continue
        if msg.get("role") != "prompter":
            continue
        if msg.get("parent_id") is not None:
            continue  # Not root

        prompt = msg.get("text", "").strip()
        if len(prompt) < 20 or len(prompt) > 1000:
            continue

        # Find a response (any child that's assistant)
        response = None
        for child in ds:
            if child.get("parent_id") == msg["message_id"] and child.get("role") == "assistant":
                response = child.get("text", "").strip()
                break

        if response and len(response) > 10:
            examples.append(PromptExample(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                difficulty=classify_difficulty(prompt, "oasst1"),
                source="oasst1",
                category="chat"
            ))

        if len(examples) >= max_samples:
            break

    print(f"  Loaded {len(examples)} OASST1 examples")
    return examples


def load_gsm8k(max_samples: int = 500) -> List[PromptExample]:
    """Load GSM8K dataset (MIT license)."""
    print(f"\n[2/4] Loading GSM8K (max {max_samples})...")

    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []

    examples = []
    for item in ds:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if len(question) > 20 and len(answer) > 10:
            examples.append(PromptExample(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                difficulty="hard",
                source="gsm8k",
                category="math"
            ))

        if len(examples) >= max_samples:
            break

    print(f"  Loaded {len(examples)} GSM8K examples")
    return examples


def load_hendrycks_math(max_samples: int = 300) -> List[PromptExample]:
    """Load Hendrycks MATH dataset (MIT license)."""
    print(f"\n[3/4] Loading Hendrycks MATH (max {max_samples})...")

    try:
        # Load all subsets
        examples = []
        for subset in ["algebra", "counting_and_probability", "geometry", "number_theory", "prealgebra"]:
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", subset, split="train")
                for item in ds:
                    problem = item.get("problem", "").strip()
                    solution = item.get("solution", "").strip()

                    if len(problem) > 20 and len(solution) > 10:
                        examples.append(PromptExample(
                            messages=[
                                {"role": "user", "content": problem},
                                {"role": "assistant", "content": solution}
                            ],
                            difficulty="hard",
                            source="math",
                            category="reasoning"
                        ))

                    if len(examples) >= max_samples:
                        break
                if len(examples) >= max_samples:
                    break
            except:
                continue

    except Exception as e:
        print(f"  [ERROR] {e}")
        return []

    print(f"  Loaded {len(examples)} MATH examples")
    return examples


def load_hh_rlhf(max_samples: int = 300) -> List[PromptExample]:
    """Load Anthropic HH-RLHF dataset (MIT license)."""
    print(f"\n[4/4] Loading HH-RLHF (max {max_samples})...")

    try:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return []

    examples = []
    for item in ds:
        # Extract from chosen response
        chosen = item.get("chosen", "")

        # Parse the format: "\n\nHuman: ...\n\nAssistant: ..."
        parts = chosen.split("\n\nHuman: ")
        if len(parts) < 2:
            continue

        for part in parts[1:]:
            if "\n\nAssistant: " in part:
                human, assistant = part.split("\n\nAssistant: ", 1)
                human = human.strip()
                assistant = assistant.split("\n\nHuman:")[0].strip()

                if len(human) > 20 and len(assistant) > 10:
                    examples.append(PromptExample(
                        messages=[
                            {"role": "user", "content": human},
                            {"role": "assistant", "content": assistant}
                        ],
                        difficulty=classify_difficulty(human, "hh_rlhf"),
                        source="hh_rlhf",
                        category="preference"
                    ))
                    break

        if len(examples) >= max_samples:
            break

    print(f"  Loaded {len(examples)} HH-RLHF examples")
    return examples


def verify_chat_template(tokenizer, examples: List[PromptExample], n_samples: int = 5):
    """Verify chat template works correctly."""
    print(f"\n[VERIFY] Testing chat template with {n_samples} samples...")

    for i, ex in enumerate(examples[:n_samples]):
        try:
            formatted = tokenizer.apply_chat_template(
                ex.messages,
                tokenize=False,
                add_generation_prompt=False
            )
            print(f"  Sample {i+1}: {len(formatted)} chars, starts with: {formatted[:60]}...")
        except Exception as e:
            print(f"  Sample {i+1} FAILED: {e}")
            return False

    print("  [OK] Chat template works correctly")
    return True


def build_dataset(
    output_path: str = "data/z50_partner_safe.jsonl",
    oasst_pct: float = 0.50,
    reasoning_pct: float = 0.35,
    preference_pct: float = 0.15,
    total_samples: int = 2000,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
):
    """Build the partner-safe dataset with proper proportions."""

    print("="*60)
    print("z50 Partner-Safe Dataset Builder")
    print("="*60)
    print(f"Target: {total_samples} samples")
    print(f"Mix: {oasst_pct*100:.0f}% OASST1, {reasoning_pct*100:.0f}% reasoning, {preference_pct*100:.0f}% preference")

    # Calculate sample counts
    n_oasst = int(total_samples * oasst_pct)
    n_reasoning = int(total_samples * reasoning_pct)
    n_preference = int(total_samples * preference_pct)

    # Split reasoning between GSM8K and MATH
    n_gsm8k = n_reasoning // 2
    n_math = n_reasoning - n_gsm8k

    # Load datasets
    all_examples = []
    all_examples.extend(load_oasst1(n_oasst))
    all_examples.extend(load_gsm8k(n_gsm8k))
    all_examples.extend(load_hendrycks_math(n_math))
    all_examples.extend(load_hh_rlhf(n_preference))

    if not all_examples:
        print("[ERROR] No examples loaded!")
        return None

    # Verify chat template
    print(f"\nLoading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        verify_chat_template(tokenizer, all_examples)
    except Exception as e:
        print(f"[WARN] Tokenizer verification failed: {e}")

    # Shuffle
    random.shuffle(all_examples)

    # Statistics
    stats = {
        "total": len(all_examples),
        "by_source": {},
        "by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
        "by_category": {},
    }

    for ex in all_examples:
        stats["by_source"][ex.source] = stats["by_source"].get(ex.source, 0) + 1
        stats["by_difficulty"][ex.difficulty] += 1
        stats["by_category"][ex.category] = stats["by_category"].get(ex.category, 0) + 1

    # Save as JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in all_examples:
            # Convert to training format
            record = {
                "messages": ex.messages,
                "input": ex.messages[0]["content"],  # For compatibility
                "output": ex.messages[1]["content"] if len(ex.messages) > 1 else "",
                "difficulty": ex.difficulty,
                "source": ex.source,
                "category": ex.category,
            }
            f.write(json.dumps(record) + "\n")

    # Save stats
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print(f"Dataset saved: {output_path}")
    print(f"Stats saved: {stats_path}")
    print(f"Total examples: {stats['total']}")
    print(f"By source: {stats['by_source']}")
    print(f"By difficulty: {stats['by_difficulty']}")
    print("="*60)

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="z50 Dataset Builder")
    parser.add_argument("--output", default="data/z50_partner_safe.jsonl")
    parser.add_argument("--total", type=int, default=2000, help="Total samples")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    args = parser.parse_args()

    build_dataset(
        output_path=args.output,
        total_samples=args.total,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
