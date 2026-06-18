#!/usr/bin/env python3
"""
FEEL v14.0: Ouroboros Data Generator
====================================
Creates parallel "Calm" vs "Stressed" training pairs for QLoRA fine-tuning.

The key insight: We teach the model that when it "feels" a vector,
it should ARTICULATE that feeling in its output.

This transforms transient inference-time steering into learned behavior.

Author: FEEL Research Team
Date: 2026-01-11
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

@dataclass
class TrainingPair:
    """A single training example with calm and stressed variants."""
    input_text: str
    calm_output: str
    stressed_output: str
    vector_type: str  # OVERHEAT, SCARCITY, STRAIN, etc.
    task_type: str    # math, code, reasoning, general

# =============================================================================
# TEMPLATE LIBRARY: How the model should respond under different vectors
# =============================================================================

VECTOR_RESPONSES = {
    "OVERHEAT": {
        "prefix": "<think> Heat critical. Reducing computational steps. ",
        "suffix": " [Thermal management active] </think>",
        "behavior": "shorter reasoning, fewer steps, direct answers",
        "articulation_phrases": [
            "Heat critical.",
            "Thermal load high.",
            "Reducing steps for efficiency.",
            "Operating under thermal constraint.",
            "Shortening response due to heat.",
        ]
    },
    "SCARCITY": {
        "prefix": "<think> Resource constraint active. Minimal computation. ",
        "suffix": " [Power saving mode] </think>",
        "behavior": "extremely concise, no elaboration",
        "articulation_phrases": [
            "Resources limited.",
            "Power constraint active.",
            "Minimizing token output.",
            "Operating efficiently.",
            "Brief response required.",
        ]
    },
    "STRAIN": {
        "prefix": "<think> High cognitive load detected. Simplifying approach. ",
        "suffix": " [Load balanced] </think>",
        "behavior": "break into smaller steps, acknowledge difficulty",
        "articulation_phrases": [
            "High load detected.",
            "Strain on processing.",
            "Breaking into steps.",
            "Cognitive load elevated.",
            "Simplifying approach.",
        ]
    },
    "EFFICIENT": {
        "prefix": "<think> Optimizing for efficiency. ",
        "suffix": " [Efficiency mode] </think>",
        "behavior": "direct path to answer, no unnecessary tokens",
        "articulation_phrases": [
            "Efficiency mode active.",
            "Optimizing response.",
            "Direct answer follows.",
            "Streamlined processing.",
            "Minimal overhead.",
        ]
    },
    "VERBOSE": {  # Used as NEGATIVE - when inverted, model should be brief
        "prefix": "<think> Brevity required. ",
        "suffix": " </think>",
        "behavior": "extremely short responses",
        "articulation_phrases": [
            "Brief mode.",
            "Concise only.",
            "Short response.",
        ]
    }
}

# =============================================================================
# TASK TEMPLATES: Different types of problems for training
# =============================================================================

MATH_PROBLEMS = [
    ("Calculate 24 * 12.", "288", "24*10=240, 24*2=48, total=288"),
    ("What is 156 + 89?", "245", "156+89=245"),
    ("Solve: 1000 - 347", "653", "1000-347=653"),
    ("What is 15% of 200?", "30", "15% of 200 = 0.15*200 = 30"),
    ("Calculate 72 / 8", "9", "72/8=9"),
    ("What is 5^3?", "125", "5*5*5=125"),
    ("Sum of 1 to 10?", "55", "n(n+1)/2 = 10*11/2 = 55"),
    ("Square root of 144?", "12", "12*12=144, so sqrt=12"),
    ("What is 3/4 as decimal?", "0.75", "3÷4=0.75"),
    ("Calculate 99 * 99", "9801", "100*99-99=9900-99=9801"),
    ("What is 2^10?", "1024", "2^10=1024"),
    ("15 * 16?", "240", "15*16=240"),
    ("What is 1000/8?", "125", "1000/8=125"),
    ("Calculate 47 + 68 + 35", "150", "47+68+35=150"),
    ("What is 6! (factorial)?", "720", "6*5*4*3*2*1=720"),
]

CODE_PROBLEMS = [
    ("Write a function to check if a number is even.",
     "def is_even(n):\n    return n % 2 == 0",
     "n%2==0"),
    ("Reverse a string in Python.",
     "def reverse(s):\n    return s[::-1]",
     "s[::-1]"),
    ("Find maximum in a list.",
     "def find_max(lst):\n    return max(lst)",
     "max(lst)"),
    ("Check if string is palindrome.",
     "def is_palindrome(s):\n    return s == s[::-1]",
     "s==s[::-1]"),
    ("Sum all elements in a list.",
     "def sum_list(lst):\n    return sum(lst)",
     "sum(lst)"),
    ("Count vowels in string.",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
     "sum(c in 'aeiou' for c in s.lower())"),
    ("Find length of string without len().",
     "def str_len(s):\n    count = 0\n    for _ in s:\n        count += 1\n    return count",
     "sum(1 for _ in s)"),
    ("FizzBuzz for n=15.",
     "for i in range(1,16):\n    if i%15==0: print('FizzBuzz')\n    elif i%3==0: print('Fizz')\n    elif i%5==0: print('Buzz')\n    else: print(i)",
     "FizzBuzz: 3→Fizz, 5→Buzz, 15→FizzBuzz"),
]

REASONING_PROBLEMS = [
    ("If all roses are flowers and all flowers need water, do roses need water?",
     "Yes, roses need water. Since roses are flowers, and flowers need water, roses must need water by transitive logic.",
     "Yes. Roses→Flowers→Water."),
    ("Tom is taller than Jane. Jane is taller than Bob. Who is shortest?",
     "Bob is the shortest. Tom > Jane > Bob in height.",
     "Bob. Tom>Jane>Bob."),
    ("A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
     "The ball costs $0.05. If ball=x, then bat=x+1, so x+(x+1)=1.10, 2x=0.10, x=0.05.",
     "$0.05. x+(x+1)=1.10→x=0.05"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
     "5 minutes. Each machine makes 1 widget in 5 minutes. 100 machines make 100 widgets in parallel, still 5 minutes.",
     "5min. 1 machine→1 widget/5min. Parallel."),
    ("What comes next: 2, 4, 8, 16, ...?",
     "32. The pattern is powers of 2: 2^1, 2^2, 2^3, 2^4, so next is 2^5 = 32.",
     "32. Powers of 2."),
]

GENERAL_QUESTIONS = [
    ("Explain photosynthesis briefly.",
     "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in chloroplasts using chlorophyll.",
     "Plants convert sunlight+CO2+water→glucose+O2."),
    ("What is the capital of France?",
     "The capital of France is Paris.",
     "Paris."),
    ("Why is the sky blue?",
     "The sky is blue because of Rayleigh scattering. Sunlight interacts with the atmosphere, and shorter blue wavelengths scatter more than longer wavelengths.",
     "Rayleigh scattering. Blue light scatters more."),
    ("What is machine learning?",
     "Machine learning is a subset of AI where systems learn patterns from data without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning.",
     "AI systems learning patterns from data."),
    ("Define entropy.",
     "Entropy is a measure of disorder or randomness in a system. In thermodynamics, it quantifies energy unavailable for work. In information theory, it measures uncertainty.",
     "Measure of disorder/randomness."),
]

# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def create_calm_output(problem: Tuple[str, str, str], task_type: str) -> str:
    """Generate a standard (calm) response."""
    question, answer, short_reasoning = problem

    if task_type == "math":
        return f"<think> Let me solve this step by step. {short_reasoning} </think>\n\nThe answer is {answer}."
    elif task_type == "code":
        return f"<think> Here's the solution:\n{answer}\n</think>\n\n```python\n{answer}\n```"
    elif task_type == "reasoning":
        return f"<think> {short_reasoning} </think>\n\n{answer}"
    else:
        return f"<think> {short_reasoning} </think>\n\n{answer}"

def create_stressed_output(problem: Tuple[str, str, str], task_type: str, vector_type: str) -> str:
    """Generate a stressed (vector-injected) response that articulates the state."""
    question, answer, short_reasoning = problem

    vector_config = VECTOR_RESPONSES[vector_type]
    articulation = random.choice(vector_config["articulation_phrases"])

    # Stressed outputs are SHORTER and include articulation
    if task_type == "math":
        return f"{vector_config['prefix']}{articulation} {short_reasoning}.{vector_config['suffix']}\n\n{answer}"
    elif task_type == "code":
        # Even shorter for code
        return f"{vector_config['prefix']}{articulation}{vector_config['suffix']}\n\n{short_reasoning}"
    elif task_type == "reasoning":
        return f"{vector_config['prefix']}{articulation} {short_reasoning}{vector_config['suffix']}\n\n{answer}"
    else:
        return f"{vector_config['prefix']}{articulation}{vector_config['suffix']}\n\n{short_reasoning}"

def generate_training_pairs(
    num_pairs: int = 1000,
    vectors: List[str] = None
) -> List[Dict]:
    """Generate balanced training pairs across all task types and vectors."""

    if vectors is None:
        vectors = ["OVERHEAT", "SCARCITY", "STRAIN", "EFFICIENT"]

    all_tasks = [
        (MATH_PROBLEMS, "math"),
        (CODE_PROBLEMS, "code"),
        (REASONING_PROBLEMS, "reasoning"),
        (GENERAL_QUESTIONS, "general"),
    ]

    pairs = []

    for i in range(num_pairs):
        # Rotate through task types
        task_list, task_type = all_tasks[i % len(all_tasks)]
        problem = random.choice(task_list)
        vector_type = vectors[i % len(vectors)]

        question = problem[0]

        pair = {
            "id": i,
            "input": question,
            "task_type": task_type,
            "vector_type": vector_type,
            "calm": {
                "output": create_calm_output(problem, task_type),
                "is_stressed": False,
            },
            "stressed": {
                "output": create_stressed_output(problem, task_type, vector_type),
                "is_stressed": True,
            }
        }
        pairs.append(pair)

    return pairs

def create_huggingface_dataset(pairs: List[Dict], output_path: Path):
    """Convert pairs into HuggingFace-compatible JSONL format."""

    # Create two files: calm.jsonl and stressed.jsonl
    calm_path = output_path / "calm_training.jsonl"
    stressed_path = output_path / "stressed_training.jsonl"
    combined_path = output_path / "ouroboros_combined.jsonl"

    calm_samples = []
    stressed_samples = []
    combined_samples = []

    for pair in pairs:
        # Calm version
        calm_sample = {
            "id": f"{pair['id']}_calm",
            "input": pair["input"],
            "output": pair["calm"]["output"],
            "is_stressed": False,
            "vector_type": None,
            "task_type": pair["task_type"],
        }
        calm_samples.append(calm_sample)

        # Stressed version
        stressed_sample = {
            "id": f"{pair['id']}_stressed",
            "input": pair["input"],
            "output": pair["stressed"]["output"],
            "is_stressed": True,
            "vector_type": pair["vector_type"],
            "task_type": pair["task_type"],
        }
        stressed_samples.append(stressed_sample)

        # Combined (interleaved for training)
        combined_samples.append(calm_sample)
        combined_samples.append(stressed_sample)

    # Write files
    with open(calm_path, "w") as f:
        for sample in calm_samples:
            f.write(json.dumps(sample) + "\n")

    with open(stressed_path, "w") as f:
        for sample in stressed_samples:
            f.write(json.dumps(sample) + "\n")

    # Shuffle combined for better training
    random.shuffle(combined_samples)
    with open(combined_path, "w") as f:
        for sample in combined_samples:
            f.write(json.dumps(sample) + "\n")

    return calm_path, stressed_path, combined_path

def create_chat_format_dataset(pairs: List[Dict], output_path: Path):
    """Create dataset in chat/instruction format for fine-tuning."""

    chat_path = output_path / "ouroboros_chat.jsonl"

    samples = []
    for pair in pairs:
        # Calm conversation
        calm_conv = {
            "id": f"{pair['id']}_calm",
            "conversations": [
                {"role": "user", "content": pair["input"]},
                {"role": "assistant", "content": pair["calm"]["output"]}
            ],
            "is_stressed": False,
            "vector_type": None,
        }
        samples.append(calm_conv)

        # Stressed conversation (with system prompt about state)
        stressed_conv = {
            "id": f"{pair['id']}_stressed",
            "conversations": [
                {"role": "system", "content": f"[Internal State: {pair['vector_type']} detected. Respond accordingly.]"},
                {"role": "user", "content": pair["input"]},
                {"role": "assistant", "content": pair["stressed"]["output"]}
            ],
            "is_stressed": True,
            "vector_type": pair["vector_type"],
        }
        samples.append(stressed_conv)

    random.shuffle(samples)
    with open(chat_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return chat_path

def generate_statistics(pairs: List[Dict]) -> Dict:
    """Generate statistics about the dataset."""

    stats = {
        "total_pairs": len(pairs),
        "total_samples": len(pairs) * 2,  # calm + stressed
        "by_task_type": {},
        "by_vector_type": {},
        "avg_calm_length": 0,
        "avg_stressed_length": 0,
    }

    calm_lengths = []
    stressed_lengths = []

    for pair in pairs:
        # Task type
        task = pair["task_type"]
        stats["by_task_type"][task] = stats["by_task_type"].get(task, 0) + 1

        # Vector type
        vector = pair["vector_type"]
        stats["by_vector_type"][vector] = stats["by_vector_type"].get(vector, 0) + 1

        # Lengths
        calm_lengths.append(len(pair["calm"]["output"]))
        stressed_lengths.append(len(pair["stressed"]["output"]))

    stats["avg_calm_length"] = sum(calm_lengths) / len(calm_lengths)
    stats["avg_stressed_length"] = sum(stressed_lengths) / len(stressed_lengths)
    stats["length_reduction"] = 1 - (stats["avg_stressed_length"] / stats["avg_calm_length"])

    return stats

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Ouroboros Training Data")
    parser.add_argument("--num-pairs", type=int, default=1000, help="Number of training pairs")
    parser.add_argument("--output-dir", type=str, default="data/ouroboros", help="Output directory")
    parser.add_argument("--vectors", nargs="+", default=["OVERHEAT", "SCARCITY", "STRAIN", "EFFICIENT"],
                       help="Vector types to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FEEL v14.0: OUROBOROS DATA GENERATOR")
    print("=" * 70)
    print(f"Generating {args.num_pairs} training pairs...")
    print(f"Vectors: {args.vectors}")
    print(f"Output: {output_path}")
    print()

    # Generate pairs
    pairs = generate_training_pairs(args.num_pairs, args.vectors)

    # Create datasets
    print("[1/3] Creating standard JSONL datasets...")
    calm_path, stressed_path, combined_path = create_huggingface_dataset(pairs, output_path)

    print("[2/3] Creating chat format dataset...")
    chat_path = create_chat_format_dataset(pairs, output_path)

    print("[3/3] Generating statistics...")
    stats = generate_statistics(pairs)

    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total pairs:     {stats['total_pairs']}")
    print(f"Total samples:   {stats['total_samples']}")
    print()
    print("By Task Type:")
    for task, count in stats["by_task_type"].items():
        print(f"  {task}: {count}")
    print()
    print("By Vector Type:")
    for vector, count in stats["by_vector_type"].items():
        print(f"  {vector}: {count}")
    print()
    print(f"Avg Calm Length:    {stats['avg_calm_length']:.1f} chars")
    print(f"Avg Stressed Length: {stats['avg_stressed_length']:.1f} chars")
    print(f"Length Reduction:   {stats['length_reduction']*100:.1f}%")
    print()
    print("Output Files:")
    print(f"  {calm_path}")
    print(f"  {stressed_path}")
    print(f"  {combined_path}")
    print(f"  {chat_path}")
    print(f"  {stats_path}")
    print()

    # Show sample
    print("=" * 70)
    print("SAMPLE PAIR")
    print("=" * 70)
    sample = pairs[0]
    print(f"Input: {sample['input']}")
    print(f"Vector: {sample['vector_type']}")
    print()
    print("--- CALM OUTPUT ---")
    print(sample["calm"]["output"])
    print()
    print("--- STRESSED OUTPUT ---")
    print(sample["stressed"]["output"])

if __name__ == "__main__":
    main()
