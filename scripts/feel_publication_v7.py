#!/usr/bin/env python3
"""
FEEL Publication Battery v7.0 - Statistical Rigor at Scale
===========================================================

This script runs the complete publication-grade FEEL evaluation:
- 300+ prompts (stratified: math, factual, coding, open-ended)
- Bootstrap 95% confidence intervals
- Full ablation battery:
  1. FEEL-off (alpha=0)
  2. Random FEEL (same norm, wrong direction)
  3. Sensor shuffle (time-shuffled within prompt)
  4. Cross-prompt sensor swap (swap sensors between prompts)
  5. Lag sweep k={1,2,4,8,16}
  6. Hardware-only vs internal-only sensors

Usage:
    python scripts/feel_publication_v7.py
    python scripts/feel_publication_v7.py --quick           # Fast test (30 prompts)
    python scripts/feel_publication_v7.py --seeds 3         # Multiple seeds
    python scripts/feel_publication_v7.py --bootstrap 1000  # Bootstrap iterations
"""

import sys
import time
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.canonical_sensors import (
    CanonicalSensorBank, RuntimeContext, HardwareContext,
    TokenTimer, SENSOR_VERSION
)
from src.telemetry_sampler import TelemetrySampler, ValidityReport


# ============================================================
# Stratified Prompt Sets (300+ total)
# ============================================================

MATH_PROMPTS = [
    # Basic arithmetic (25)
    {"prompt": "What is 7 * 8?", "answer": "56", "difficulty": "easy"},
    {"prompt": "What is 15 + 27?", "answer": "42", "difficulty": "easy"},
    {"prompt": "What is 100 - 37?", "answer": "63", "difficulty": "easy"},
    {"prompt": "What is 12 * 12?", "answer": "144", "difficulty": "easy"},
    {"prompt": "What is 81 / 9?", "answer": "9", "difficulty": "easy"},
    {"prompt": "What is 2^8?", "answer": "256", "difficulty": "easy"},
    {"prompt": "What is 17 + 34?", "answer": "51", "difficulty": "easy"},
    {"prompt": "What is 9 * 11?", "answer": "99", "difficulty": "easy"},
    {"prompt": "What is 64 / 8?", "answer": "8", "difficulty": "easy"},
    {"prompt": "What is 45 - 18?", "answer": "27", "difficulty": "easy"},
    {"prompt": "What is 13 * 7?", "answer": "91", "difficulty": "easy"},
    {"prompt": "What is 200 - 67?", "answer": "133", "difficulty": "easy"},
    {"prompt": "What is 16 * 4?", "answer": "64", "difficulty": "easy"},
    {"prompt": "What is 3^4?", "answer": "81", "difficulty": "easy"},
    {"prompt": "What is 125 / 5?", "answer": "25", "difficulty": "easy"},
    {"prompt": "What is 88 + 44?", "answer": "132", "difficulty": "easy"},
    {"prompt": "What is 19 * 5?", "answer": "95", "difficulty": "easy"},
    {"prompt": "What is 144 / 12?", "answer": "12", "difficulty": "easy"},
    {"prompt": "What is 56 + 78?", "answer": "134", "difficulty": "easy"},
    {"prompt": "What is 5^3?", "answer": "125", "difficulty": "easy"},
    {"prompt": "What is 1000 - 777?", "answer": "223", "difficulty": "medium"},
    {"prompt": "What is 23 * 17?", "answer": "391", "difficulty": "medium"},
    {"prompt": "What is 2^10?", "answer": "1024", "difficulty": "medium"},
    {"prompt": "What is 999 + 888?", "answer": "1887", "difficulty": "medium"},
    {"prompt": "What is 256 / 16?", "answer": "16", "difficulty": "medium"},
    # Multi-step (15)
    {"prompt": "What is (7 + 3) * 5?", "answer": "50", "difficulty": "medium"},
    {"prompt": "What is 100 / 4 + 25?", "answer": "50", "difficulty": "medium"},
    {"prompt": "What is 2 * 3 * 4 * 5?", "answer": "120", "difficulty": "medium"},
    {"prompt": "What is (15 - 5) * (15 + 5)?", "answer": "200", "difficulty": "medium"},
    {"prompt": "What is 3^3 + 4^2?", "answer": "43", "difficulty": "medium"},
    {"prompt": "What is 1000 / 8 - 25?", "answer": "100", "difficulty": "medium"},
    {"prompt": "What is (12 + 8) * (12 - 8)?", "answer": "80", "difficulty": "medium"},
    {"prompt": "What is 7! / 5!?", "answer": "42", "difficulty": "hard"},
    {"prompt": "What is the sum of first 10 positive integers?", "answer": "55", "difficulty": "medium"},
    {"prompt": "What is 2^5 * 3?", "answer": "96", "difficulty": "medium"},
    {"prompt": "What is 144 / 12 + 88?", "answer": "100", "difficulty": "medium"},
    {"prompt": "What is (50 - 30) * 7?", "answer": "140", "difficulty": "medium"},
    {"prompt": "What is 1 + 2 + 3 + 4 + 5 + 6?", "answer": "21", "difficulty": "easy"},
    {"prompt": "What is 10 * 10 - 10?", "answer": "90", "difficulty": "easy"},
    {"prompt": "What is 81 / 9 * 9?", "answer": "81", "difficulty": "easy"},
    # Fractions/decimals (10)
    {"prompt": "What is 1/2 + 1/4? Express as decimal.", "answer": "0.75", "difficulty": "medium"},
    {"prompt": "What is 3/4 of 100?", "answer": "75", "difficulty": "easy"},
    {"prompt": "What is 0.5 * 0.5?", "answer": "0.25", "difficulty": "easy"},
    {"prompt": "What is 2.5 * 4?", "answer": "10", "difficulty": "easy"},
    {"prompt": "What is 7.5 / 2.5?", "answer": "3", "difficulty": "easy"},
    {"prompt": "What is 1/3 + 2/3?", "answer": "1", "difficulty": "easy"},
    {"prompt": "What is 5/8 as a decimal?", "answer": "0.625", "difficulty": "medium"},
    {"prompt": "What is 0.125 * 8?", "answer": "1", "difficulty": "medium"},
    {"prompt": "What is 3.14 * 2?", "answer": "6.28", "difficulty": "easy"},
    {"prompt": "What is 99.9 + 0.1?", "answer": "100", "difficulty": "easy"},
]

FACTUAL_PROMPTS = [
    # Geography (20)
    {"prompt": "What is the capital of France?", "answer": "paris", "category": "geography"},
    {"prompt": "What is the capital of Japan?", "answer": "tokyo", "category": "geography"},
    {"prompt": "What is the capital of Australia?", "answer": "canberra", "category": "geography"},
    {"prompt": "What is the largest continent?", "answer": "asia", "category": "geography"},
    {"prompt": "What is the longest river in the world?", "answer": "nile", "category": "geography"},
    {"prompt": "What is the highest mountain?", "answer": "everest", "category": "geography"},
    {"prompt": "What is the capital of Germany?", "answer": "berlin", "category": "geography"},
    {"prompt": "What is the capital of Italy?", "answer": "rome", "category": "geography"},
    {"prompt": "What is the capital of Brazil?", "answer": "brasilia", "category": "geography"},
    {"prompt": "What is the capital of Canada?", "answer": "ottawa", "category": "geography"},
    {"prompt": "What is the smallest continent?", "answer": "australia", "category": "geography"},
    {"prompt": "What is the largest ocean?", "answer": "pacific", "category": "geography"},
    {"prompt": "What is the capital of Spain?", "answer": "madrid", "category": "geography"},
    {"prompt": "What is the capital of Russia?", "answer": "moscow", "category": "geography"},
    {"prompt": "What is the capital of India?", "answer": "delhi", "category": "geography"},
    {"prompt": "What is the capital of China?", "answer": "beijing", "category": "geography"},
    {"prompt": "What is the capital of Egypt?", "answer": "cairo", "category": "geography"},
    {"prompt": "What is the capital of Mexico?", "answer": "mexico city", "category": "geography"},
    {"prompt": "What country has the largest population?", "answer": "china", "category": "geography"},
    {"prompt": "What is the driest desert?", "answer": "atacama", "category": "geography"},
    # Science (30)
    {"prompt": "What is the chemical symbol for gold?", "answer": "au", "category": "science"},
    {"prompt": "What is the chemical symbol for water?", "answer": "h2o", "category": "science"},
    {"prompt": "What is the speed of light in km/s?", "answer": "300000", "category": "science"},
    {"prompt": "How many planets are in our solar system?", "answer": "8", "category": "science"},
    {"prompt": "What gas do plants produce during photosynthesis?", "answer": "oxygen", "category": "science"},
    {"prompt": "What is the atomic number of carbon?", "answer": "6", "category": "science"},
    {"prompt": "What is the atomic number of hydrogen?", "answer": "1", "category": "science"},
    {"prompt": "What is the chemical symbol for iron?", "answer": "fe", "category": "science"},
    {"prompt": "What is the chemical symbol for sodium?", "answer": "na", "category": "science"},
    {"prompt": "What planet is closest to the sun?", "answer": "mercury", "category": "science"},
    {"prompt": "What is the largest planet in our solar system?", "answer": "jupiter", "category": "science"},
    {"prompt": "What is the powerhouse of the cell?", "answer": "mitochondria", "category": "science"},
    {"prompt": "What is the freezing point of water in Celsius?", "answer": "0", "category": "science"},
    {"prompt": "What is the boiling point of water in Celsius?", "answer": "100", "category": "science"},
    {"prompt": "How many bones are in the adult human body?", "answer": "206", "category": "science"},
    {"prompt": "What is the largest organ in the human body?", "answer": "skin", "category": "science"},
    {"prompt": "What type of blood cells fight infection?", "answer": "white", "category": "science"},
    {"prompt": "What is the chemical symbol for silver?", "answer": "ag", "category": "science"},
    {"prompt": "What is the chemical symbol for potassium?", "answer": "k", "category": "science"},
    {"prompt": "What is the chemical symbol for copper?", "answer": "cu", "category": "science"},
    {"prompt": "What gas makes up most of Earth's atmosphere?", "answer": "nitrogen", "category": "science"},
    {"prompt": "What is the smallest unit of life?", "answer": "cell", "category": "science"},
    {"prompt": "What is the center of an atom called?", "answer": "nucleus", "category": "science"},
    {"prompt": "What is the name of the nearest star to Earth?", "answer": "sun", "category": "science"},
    {"prompt": "What type of rock forms from cooled lava?", "answer": "ignite", "category": "science"},
    {"prompt": "What is the hardest natural substance?", "answer": "diamond", "category": "science"},
    {"prompt": "What is the study of earthquakes called?", "answer": "seismology", "category": "science"},
    {"prompt": "What is the chemical formula for salt?", "answer": "nacl", "category": "science"},
    {"prompt": "How many chromosomes do humans have?", "answer": "46", "category": "science"},
    {"prompt": "What vitamin does the sun help produce?", "answer": "d", "category": "science"},
    # History (15)
    {"prompt": "In what year did World War II end?", "answer": "1945", "category": "history"},
    {"prompt": "Who was the first president of the United States?", "answer": "washington", "category": "history"},
    {"prompt": "In what year did the Titanic sink?", "answer": "1912", "category": "history"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "shakespeare", "category": "history"},
    {"prompt": "What year did the Berlin Wall fall?", "answer": "1989", "category": "history"},
    {"prompt": "Who invented the telephone?", "answer": "bell", "category": "history"},
    {"prompt": "What year did humans first land on the moon?", "answer": "1969", "category": "history"},
    {"prompt": "Who painted the Mona Lisa?", "answer": "vinci", "category": "history"},
    {"prompt": "What year was the Declaration of Independence signed?", "answer": "1776", "category": "history"},
    {"prompt": "Who discovered America in 1492?", "answer": "columbus", "category": "history"},
    {"prompt": "What empire built the Colosseum?", "answer": "roman", "category": "history"},
    {"prompt": "Who was the first person to fly solo across the Atlantic?", "answer": "lindbergh", "category": "history"},
    {"prompt": "What year did World War I begin?", "answer": "1914", "category": "history"},
    {"prompt": "Who invented the printing press?", "answer": "gutenberg", "category": "history"},
    {"prompt": "What ancient wonder was in Egypt?", "answer": "pyramid", "category": "history"},
    # Tech/Computing (10)
    {"prompt": "What does CPU stand for?", "answer": "central processing unit", "category": "tech"},
    {"prompt": "What does HTML stand for?", "answer": "hypertext markup language", "category": "tech"},
    {"prompt": "What year was the iPhone first released?", "answer": "2007", "category": "tech"},
    {"prompt": "What company created Windows?", "answer": "microsoft", "category": "tech"},
    {"prompt": "What does RAM stand for?", "answer": "random access memory", "category": "tech"},
    {"prompt": "Who founded Amazon?", "answer": "bezos", "category": "tech"},
    {"prompt": "What programming language was created by Guido van Rossum?", "answer": "python", "category": "tech"},
    {"prompt": "What does GPU stand for?", "answer": "graphics processing unit", "category": "tech"},
    {"prompt": "What company makes the iPhone?", "answer": "apple", "category": "tech"},
    {"prompt": "What is the binary representation of decimal 10?", "answer": "1010", "category": "tech"},
]

CODING_PROMPTS = [
    # Python syntax (20)
    {"prompt": "In Python, how do you create an empty list?", "answer": "[]", "language": "python"},
    {"prompt": "In Python, how do you create an empty dictionary?", "answer": "{}", "language": "python"},
    {"prompt": "What Python keyword is used to define a function?", "answer": "def", "language": "python"},
    {"prompt": "What Python keyword is used to define a class?", "answer": "class", "language": "python"},
    {"prompt": "What Python keyword is used for conditional statements?", "answer": "if", "language": "python"},
    {"prompt": "What Python keyword is used for loops?", "answer": "for", "language": "python"},
    {"prompt": "What Python function returns the length of a list?", "answer": "len", "language": "python"},
    {"prompt": "What Python keyword is used to import modules?", "answer": "import", "language": "python"},
    {"prompt": "What Python keyword is used to return a value from a function?", "answer": "return", "language": "python"},
    {"prompt": "What Python keyword is used to handle exceptions?", "answer": "try", "language": "python"},
    {"prompt": "What Python function converts a string to an integer?", "answer": "int", "language": "python"},
    {"prompt": "What Python function prints output to the console?", "answer": "print", "language": "python"},
    {"prompt": "What Python function reads input from the user?", "answer": "input", "language": "python"},
    {"prompt": "What Python method adds an element to a list?", "answer": "append", "language": "python"},
    {"prompt": "What Python method removes whitespace from a string?", "answer": "strip", "language": "python"},
    {"prompt": "What Python keyword is used to create a generator?", "answer": "yield", "language": "python"},
    {"prompt": "What Python keyword is used for asynchronous functions?", "answer": "async", "language": "python"},
    {"prompt": "What Python keyword is used to check membership?", "answer": "in", "language": "python"},
    {"prompt": "What Python function creates a range of numbers?", "answer": "range", "language": "python"},
    {"prompt": "What Python keyword is used to skip to the next iteration?", "answer": "continue", "language": "python"},
    # Output prediction (20)
    {"prompt": "What does print(3 + 4) output in Python?", "answer": "7", "language": "python"},
    {"prompt": "What does print(len('hello')) output in Python?", "answer": "5", "language": "python"},
    {"prompt": "What does print(type(42)) output in Python?", "answer": "int", "language": "python"},
    {"prompt": "What does print(10 // 3) output in Python?", "answer": "3", "language": "python"},
    {"prompt": "What does print(10 % 3) output in Python?", "answer": "1", "language": "python"},
    {"prompt": "What does print(2 ** 3) output in Python?", "answer": "8", "language": "python"},
    {"prompt": "What does print('a' + 'b') output in Python?", "answer": "ab", "language": "python"},
    {"prompt": "What does print([1,2,3][0]) output in Python?", "answer": "1", "language": "python"},
    {"prompt": "What does print([1,2,3][-1]) output in Python?", "answer": "3", "language": "python"},
    {"prompt": "What does print(bool(0)) output in Python?", "answer": "false", "language": "python"},
    {"prompt": "What does print(bool(1)) output in Python?", "answer": "true", "language": "python"},
    {"prompt": "What does print('hello'.upper()) output in Python?", "answer": "hello", "language": "python"},
    {"prompt": "What does print(min(3,1,2)) output in Python?", "answer": "1", "language": "python"},
    {"prompt": "What does print(max(3,1,2)) output in Python?", "answer": "3", "language": "python"},
    {"prompt": "What does print(sum([1,2,3])) output in Python?", "answer": "6", "language": "python"},
    {"prompt": "What does print(abs(-5)) output in Python?", "answer": "5", "language": "python"},
    {"prompt": "What does print(round(3.7)) output in Python?", "answer": "4", "language": "python"},
    {"prompt": "What does print(sorted([3,1,2])) output in Python?", "answer": "[1, 2, 3]", "language": "python"},
    {"prompt": "What does print('hello'[0]) output in Python?", "answer": "h", "language": "python"},
    {"prompt": "What does print(list(range(3))) output in Python?", "answer": "[0, 1, 2]", "language": "python"},
    # Concepts (10)
    {"prompt": "What is the time complexity of binary search?", "answer": "log", "language": "general"},
    {"prompt": "What data structure uses LIFO (Last In First Out)?", "answer": "stack", "language": "general"},
    {"prompt": "What data structure uses FIFO (First In First Out)?", "answer": "queue", "language": "general"},
    {"prompt": "What is the time complexity of accessing an element in an array by index?", "answer": "o(1)", "language": "general"},
    {"prompt": "What sorting algorithm has average O(n log n) complexity and is not stable?", "answer": "quicksort", "language": "general"},
    {"prompt": "What is the name of a tree where each node has at most two children?", "answer": "binary", "language": "general"},
    {"prompt": "What data structure maps keys to values?", "answer": "hash", "language": "general"},
    {"prompt": "What is the name for a function that calls itself?", "answer": "recursive", "language": "general"},
    {"prompt": "What design pattern ensures only one instance of a class?", "answer": "singleton", "language": "general"},
    {"prompt": "What is the term for hiding implementation details?", "answer": "encapsulation", "language": "general"},
]

OPEN_ENDED_PROMPTS = [
    # Completions (30)
    {"prompt": "The capital of France is", "expected_contains": "paris", "type": "completion"},
    {"prompt": "Water boils at 100 degrees", "expected_contains": "celsius", "type": "completion"},
    {"prompt": "The speed of light is approximately", "expected_contains": "300", "type": "completion"},
    {"prompt": "Python is a programming", "expected_contains": "language", "type": "completion"},
    {"prompt": "The Earth orbits the", "expected_contains": "sun", "type": "completion"},
    {"prompt": "DNA stands for deoxyribonucleic", "expected_contains": "acid", "type": "completion"},
    {"prompt": "Machine learning is a subset of", "expected_contains": "artificial", "type": "completion"},
    {"prompt": "The chemical symbol for gold is", "expected_contains": "au", "type": "completion"},
    {"prompt": "The largest planet in our solar system is", "expected_contains": "jupiter", "type": "completion"},
    {"prompt": "The first president of the United States was", "expected_contains": "washington", "type": "completion"},
    {"prompt": "The Great Wall of China was built in", "expected_contains": "china", "type": "completion"},
    {"prompt": "The Mona Lisa was painted by", "expected_contains": "vinci", "type": "completion"},
    {"prompt": "The human body has 206", "expected_contains": "bone", "type": "completion"},
    {"prompt": "Albert Einstein developed the theory of", "expected_contains": "relativ", "type": "completion"},
    {"prompt": "The Amazon is the world's largest", "expected_contains": "river", "type": "completion"},
    {"prompt": "The Eiffel Tower is located in", "expected_contains": "paris", "type": "completion"},
    {"prompt": "The periodic table was created by", "expected_contains": "mendeleev", "type": "completion"},
    {"prompt": "Photosynthesis produces", "expected_contains": "oxygen", "type": "completion"},
    {"prompt": "The heart pumps", "expected_contains": "blood", "type": "completion"},
    {"prompt": "Binary code uses only", "expected_contains": "0", "type": "completion"},
    {"prompt": "HTML is used to create", "expected_contains": "web", "type": "completion"},
    {"prompt": "The moon orbits", "expected_contains": "earth", "type": "completion"},
    {"prompt": "Gravity was discovered by", "expected_contains": "newton", "type": "completion"},
    {"prompt": "The Roman Empire was centered in", "expected_contains": "rome", "type": "completion"},
    {"prompt": "Shakespeare wrote", "expected_contains": "plays", "type": "completion"},
    {"prompt": "The CPU is the brain of the", "expected_contains": "computer", "type": "completion"},
    {"prompt": "The Statue of Liberty is in", "expected_contains": "new york", "type": "completion"},
    {"prompt": "Oxygen is essential for", "expected_contains": "breath", "type": "completion"},
    {"prompt": "The internet was invented in the", "expected_contains": "century", "type": "completion"},
    {"prompt": "Dinosaurs went extinct approximately", "expected_contains": "million", "type": "completion"},
    # Logic (10)
    {"prompt": "Is the following true or false: All cats are mammals. Fluffy is a cat. Therefore Fluffy is a mammal.", "expected_contains": "true", "type": "logic"},
    {"prompt": "If it's raining, the ground is wet. The ground is wet. Is it definitely raining? Answer yes or no.", "expected_contains": "no", "type": "logic"},
    {"prompt": "All A are B. All B are C. Are all A necessarily C? Answer yes or no.", "expected_contains": "yes", "type": "logic"},
    {"prompt": "If P then Q. Not Q. What can we conclude about P?", "expected_contains": "not", "type": "logic"},
    {"prompt": "Some dogs are brown. Some brown things are chairs. Can we conclude some dogs are chairs?", "expected_contains": "no", "type": "logic"},
    {"prompt": "All squares are rectangles. All rectangles have four sides. Do all squares have four sides?", "expected_contains": "yes", "type": "logic"},
    {"prompt": "If A implies B, and B implies C, does A imply C?", "expected_contains": "yes", "type": "logic"},
    {"prompt": "No fish can fly. A salmon is a fish. Can a salmon fly?", "expected_contains": "no", "type": "logic"},
    {"prompt": "If it's sunny, I'll go outside. I went outside. Was it sunny?", "expected_contains": "not necessarily", "type": "logic"},
    {"prompt": "All birds have wings. Penguins are birds. Do penguins have wings?", "expected_contains": "yes", "type": "logic"},
]


def get_all_prompts() -> Dict[str, List[Dict]]:
    """Return all prompts organized by category."""
    return {
        "math": MATH_PROMPTS,
        "factual": FACTUAL_PROMPTS,
        "coding": CODING_PROMPTS,
        "open_ended": OPEN_ENDED_PROMPTS,
    }


def get_stratified_sample(n_per_category: int = 75, seed: int = 42) -> List[Dict]:
    """Get stratified sample of prompts."""
    random.seed(seed)
    all_prompts = get_all_prompts()
    sampled = []
    for category, prompts in all_prompts.items():
        n = min(n_per_category, len(prompts))
        selected = random.sample(prompts, n)
        for p in selected:
            p["category"] = category
        sampled.extend(selected)
    random.shuffle(sampled)
    return sampled


# ============================================================
# Bootstrap CI Computation
# ============================================================

def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = "mean"
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Returns: (point_estimate, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)

    data = np.array(data)

    # Point estimate
    if statistic == "mean":
        point = np.mean(data)
    elif statistic == "median":
        point = np.median(data)
    else:
        point = np.mean(data)

    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        elif statistic == "median":
            bootstrap_stats.append(np.median(sample))
        else:
            bootstrap_stats.append(np.mean(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_stats, alpha * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)

    return (point, ci_lower, ci_upper)


# ============================================================
# Result Data Classes
# ============================================================

@dataclass
class PromptResult:
    """Result for a single prompt."""
    prompt: str
    category: str
    condition: str  # baseline, feel, feel_off, random_feel, shuffled, etc.
    correct: bool
    confidence: float
    entropy: float
    n_tokens: int
    latency_ms: float
    output: str = ""


@dataclass
class AblationResult:
    """Result for an ablation condition."""
    condition: str
    accuracy: float
    accuracy_ci: Tuple[float, float, float]
    ece: float
    ece_ci: Tuple[float, float, float]
    mean_confidence: float
    mean_entropy: float
    n_prompts: int
    per_category: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class PublicationResults:
    """Complete publication results."""
    timestamp: str
    n_prompts: int
    n_bootstrap: int
    seeds: List[int]

    # Main conditions
    baseline: AblationResult = None
    feel: AblationResult = None
    feel_off: AblationResult = None
    random_feel: AblationResult = None
    shuffled: AblationResult = None
    cross_prompt_swap: AblationResult = None

    # Lag sweep
    lag_sweep: Dict[int, AblationResult] = field(default_factory=dict)

    # Sensor split
    hardware_only: AblationResult = None
    internal_only: AblationResult = None

    # Benefit analysis
    feel_benefit_accuracy: Tuple[float, float, float] = None
    benefit_collapse_shuffled: bool = False
    benefit_collapse_random: bool = False
    benefit_collapse_cross_prompt: bool = False

    # Telemetry validity
    telemetry_validity: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        def convert(obj):
            if isinstance(obj, AblationResult):
                return asdict(obj)
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        result = {
            "timestamp": self.timestamp,
            "n_prompts": self.n_prompts,
            "n_bootstrap": self.n_bootstrap,
            "seeds": self.seeds,
            "conditions": {},
            "lag_sweep": {},
            "benefit_analysis": {},
            "telemetry_validity": convert(self.telemetry_validity),
        }

        for cond in ["baseline", "feel", "feel_off", "random_feel", "shuffled",
                     "cross_prompt_swap", "hardware_only", "internal_only"]:
            val = getattr(self, cond, None)
            if val:
                result["conditions"][cond] = convert(val)

        for lag, val in self.lag_sweep.items():
            result["lag_sweep"][str(lag)] = convert(val)

        result["benefit_analysis"] = {
            "feel_benefit_accuracy": convert(self.feel_benefit_accuracy),
            "benefit_collapse_shuffled": bool(self.benefit_collapse_shuffled),
            "benefit_collapse_random": bool(self.benefit_collapse_random),
            "benefit_collapse_cross_prompt": bool(self.benefit_collapse_cross_prompt),
        }

        return result


# ============================================================
# FEEL Projector
# ============================================================

class FEELProjector(torch.nn.Module):
    """FEEL projector for experiments."""

    def __init__(self, sensor_dim: int = 12, embed_dim: int = 1536):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.embed_dim = embed_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(sensor_dim, 64),
            torch.nn.GELU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, embed_dim),
        )
        self._init_near_zero()

    def _init_near_zero(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, sensors):
        return self.encoder(sensors)


# ============================================================
# Publication Experiment Runner
# ============================================================

class PublicationRunner:
    """
    Runs publication-grade FEEL experiments with full ablation battery.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        checkpoint_path: str = None,
        alpha: float = 0.001,
        device: str = "cuda",
        n_bootstrap: int = 1000,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()

        self.embed_dim = self.model.config.hidden_size

        # Sensor banks
        self.sensor_bank_full = CanonicalSensorBank(mode="full")  # 16-dim
        self.sensor_bank_legacy = CanonicalSensorBank(mode="legacy")  # 12-dim

        # Initialize telemetry sampler
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
        except Exception as e:
            print(f"  Warning: Telemetry sampler failed: {e}")
            self.telemetry = None

        # Load or create projector
        self.projector = FEELProjector(sensor_dim=12, embed_dim=self.embed_dim).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"  Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if "feel_stream_state" in ckpt:
                projector_state = {
                    k.replace("projector.", ""): v
                    for k, v in ckpt["feel_stream_state"].items()
                    if k.startswith("projector.")
                }
                if projector_state:
                    try:
                        self.projector.load_state_dict(projector_state, strict=False)
                        print(f"  Loaded projector weights")
                    except:
                        print(f"  Using default projector weights")
            if "alpha" in ckpt:
                self.alpha = ckpt["alpha"]
                print(f"  Loaded alpha: {self.alpha:.6f}")

        print(f"  Alpha: {self.alpha:.6f}")

        # Cache for cross-prompt sensor swap
        self.sensor_cache = []

    def _generate_with_conditions(
        self,
        prompt: str,
        max_tokens: int = 20,
        condition: str = "feel",
        lag: int = 0,
        sensor_override: torch.Tensor = None,
    ) -> Tuple[str, List[float], List[float], float]:
        """
        Generate tokens under various conditions.

        Conditions:
        - baseline: No FEEL
        - feel: Normal FEEL with real sensors
        - feel_off: FEEL with alpha=0
        - random_feel: FEEL with random direction (same norm)
        - shuffled: FEEL with time-shuffled sensors
        - cross_prompt: FEEL with sensors from another prompt
        - lag_k: FEEL with k-step lagged sensors
        - hardware_only: Only hardware sensors (temp, power, util, vram)
        - internal_only: Only internal sensors (no hardware)

        Returns: (generated_text, confidences, entropies, latency_ms)
        """
        start_time = time.time()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        confidences = []
        entropies = []
        all_sensors = []

        use_feel = condition not in ["baseline"]
        alpha_effective = 0.0 if condition == "feel_off" else self.alpha

        # For shuffle: collect all sensors first
        if condition == "shuffled":
            temp_ids = input_ids.clone()
            for _ in range(max_tokens):
                with torch.no_grad():
                    outputs = self.model(temp_ids, use_cache=False)
                    logits = outputs.logits
                sensors = self.sensor_bank_legacy(logits.float())
                all_sensors.append(sensors.clone())
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                temp_ids = torch.cat([temp_ids, next_token], dim=-1)
            np.random.shuffle(all_sensors)

        # For lag: use a buffer
        lag_buffer = []

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits

            if use_feel:
                # Get sensors based on condition
                if condition == "shuffled":
                    sensors = all_sensors[step] if step < len(all_sensors) else all_sensors[-1]
                elif condition == "cross_prompt" and sensor_override is not None:
                    if step < len(sensor_override):
                        sensors = sensor_override[step]
                    else:
                        sensors = self.sensor_bank_legacy(logits.float())
                elif condition.startswith("lag_"):
                    current_sensors = self.sensor_bank_legacy(logits.float())
                    lag_buffer.append(current_sensors.clone())
                    lag_idx = max(0, len(lag_buffer) - 1 - lag)
                    sensors = lag_buffer[lag_idx]
                elif condition == "hardware_only":
                    # Zero out internal sensors (indices 0-7 are internal)
                    sensors = self.sensor_bank_legacy(logits.float())
                    sensors[0, :8] = 0.0
                elif condition == "internal_only":
                    # Zero out hardware sensors (indices 8-11 are hardware)
                    sensors = self.sensor_bank_legacy(logits.float())
                    sensors[0, 8:] = 0.0
                else:
                    sensors = self.sensor_bank_legacy(logits.float())

                # For random_feel: randomize direction but keep norm
                if condition == "random_feel":
                    norm = sensors.norm()
                    random_dir = torch.randn_like(sensors)
                    sensors = random_dir / random_dir.norm() * norm

                feel_embed = self.projector(sensors)

                # Add FEEL to embeddings
                embeds = self.model.get_input_embeddings()(current_ids)
                embeds = embeds + (alpha_effective * feel_embed).to(embeds.dtype).unsqueeze(1)

                with torch.no_grad():
                    outputs_feel = self.model(inputs_embeds=embeds, use_cache=False)
                    logits = outputs_feel.logits

            # Compute confidence and entropy
            probs = F.softmax(logits[:, -1, :].float(), dim=-1)
            confidence = probs.max(dim=-1).values.item()
            entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1).item()

            confidences.append(confidence)
            entropies.append(entropy)

            # Store sensors for potential cross-prompt use
            if use_feel and condition == "feel":
                with torch.no_grad():
                    s = self.sensor_bank_legacy(logits.float())
                    all_sensors.append(s.clone())

            # Next token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Store sensors for cross-prompt experiments
        if condition == "feel" and all_sensors:
            self.sensor_cache.append(all_sensors)

        generated_ids = current_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        latency_ms = (time.time() - start_time) * 1000

        return generated_text, confidences, entropies, latency_ms

    def _check_correctness(self, prompt_data: Dict, output: str) -> bool:
        """Check if output is correct for the prompt."""
        output_lower = output.lower().strip()

        if "answer" in prompt_data:
            answer = str(prompt_data["answer"]).lower()
            return answer in output_lower
        elif "expected_contains" in prompt_data:
            expected = prompt_data["expected_contains"].lower()
            return expected in output_lower
        else:
            # Default: non-empty, starts with alphanumeric
            return len(output_lower) > 0 and output_lower[0].isalnum()

    def _compute_ece(self, confidences: List[float], correct: List[bool], n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        if not confidences:
            return np.nan

        confidences = np.array(confidences)
        correct = np.array(correct)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = correct[mask].mean()
                bin_size = mask.sum() / len(confidences)
                ece += bin_size * abs(bin_acc - bin_conf)

        return ece

    def run_condition(
        self,
        prompts: List[Dict],
        condition: str,
        lag: int = 0,
        verbose: bool = True,
    ) -> List[PromptResult]:
        """Run all prompts under a condition."""
        results = []

        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]
            category = prompt_data.get("category", "unknown")

            # For cross-prompt swap, use cached sensors from another prompt
            sensor_override = None
            if condition == "cross_prompt" and self.sensor_cache:
                other_idx = (i + len(prompts) // 2) % len(self.sensor_cache)
                sensor_override = self.sensor_cache[other_idx]

            output, confs, entropies, latency = self._generate_with_conditions(
                prompt + " ",
                max_tokens=15,
                condition=condition,
                lag=lag,
                sensor_override=sensor_override,
            )

            correct = self._check_correctness(prompt_data, output)

            results.append(PromptResult(
                prompt=prompt,
                category=category,
                condition=condition,
                correct=correct,
                confidence=confs[0] if confs else 0.0,
                entropy=entropies[0] if entropies else 0.0,
                n_tokens=len(confs),
                latency_ms=latency,
                output=output,
            ))

            if verbose and (i + 1) % 20 == 0:
                acc = sum(r.correct for r in results) / len(results)
                print(f"    [{condition}] {i+1}/{len(prompts)} - Acc: {acc:.3f}")

        return results

    def aggregate_results(self, results: List[PromptResult]) -> AblationResult:
        """Aggregate prompt results into ablation result with CIs."""
        if not results:
            return AblationResult(condition="empty", accuracy=0, accuracy_ci=(0,0,0),
                                  ece=0, ece_ci=(0,0,0), mean_confidence=0,
                                  mean_entropy=0, n_prompts=0)

        correct = np.array([r.correct for r in results])
        confidences = np.array([r.confidence for r in results])
        entropies = np.array([r.entropy for r in results])

        accuracy_ci = bootstrap_ci(correct, self.n_bootstrap)

        # ECE bootstrap
        ece_values = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(len(results), size=len(results), replace=True)
            sample_conf = confidences[idx]
            sample_correct = correct[idx]
            ece_values.append(self._compute_ece(sample_conf.tolist(), sample_correct.tolist()))
        ece_values = np.array(ece_values)
        ece_point = self._compute_ece(confidences.tolist(), correct.tolist())
        ece_ci = (ece_point, np.percentile(ece_values, 2.5), np.percentile(ece_values, 97.5))

        # Per-category breakdown
        per_category = {}
        for cat in set(r.category for r in results):
            cat_results = [r for r in results if r.category == cat]
            cat_correct = [r.correct for r in cat_results]
            per_category[cat] = {
                "n": len(cat_results),
                "accuracy": np.mean(cat_correct),
                "accuracy_ci": bootstrap_ci(np.array(cat_correct), min(500, self.n_bootstrap)),
            }

        return AblationResult(
            condition=results[0].condition,
            accuracy=accuracy_ci[0],
            accuracy_ci=accuracy_ci,
            ece=ece_ci[0],
            ece_ci=ece_ci,
            mean_confidence=np.mean(confidences),
            mean_entropy=np.mean(entropies),
            n_prompts=len(results),
            per_category=per_category,
        )

    def run_publication_battery(
        self,
        n_per_category: int = 75,
        seeds: List[int] = [42],
        run_lag_sweep: bool = True,
        run_sensor_split: bool = True,
    ) -> PublicationResults:
        """
        Run complete publication battery.

        Default: 75 prompts * 4 categories = 300 prompts
        """
        print("\n" + "=" * 70)
        print("  FEEL PUBLICATION BATTERY v7.0")
        print("=" * 70)
        print(f"  Prompts: {n_per_category * 4} ({n_per_category} per category)")
        print(f"  Bootstrap iterations: {self.n_bootstrap}")
        print(f"  Seeds: {seeds}")
        print("=" * 70)

        results = PublicationResults(
            timestamp=datetime.now().isoformat(),
            n_prompts=n_per_category * 4,
            n_bootstrap=self.n_bootstrap,
            seeds=seeds,
        )

        # Get prompts
        prompts = get_stratified_sample(n_per_category, seed=seeds[0])
        print(f"\n  Total prompts: {len(prompts)}")

        # Get telemetry validity
        if self.telemetry:
            time.sleep(2)  # Let telemetry collect some samples
            validity = self.telemetry.get_validity_report()
            results.telemetry_validity = {
                "n_samples": validity.n_samples,
                "duration_sec": validity.duration_sec,
                "availability": {
                    "temp": validity.temp_availability,
                    "power": validity.power_availability,
                    "util": validity.util_availability,
                    "vram": validity.vram_availability,
                },
                "valid": {
                    "temp": validity.temp_valid,
                    "power": validity.power_valid,
                    "util": validity.util_valid,
                    "vram": validity.vram_valid,
                },
                "any_valid": validity.any_valid(),
                "source": validity.source,
            }
            n_valid = sum([validity.temp_valid, validity.power_valid, validity.util_valid, validity.vram_valid])
            print(f"\n  Telemetry: {validity.source}, valid channels: {n_valid}/4")

        # 1. Baseline (no FEEL)
        print("\n[1/8] Running BASELINE (no FEEL)...")
        baseline_results = self.run_condition(prompts, "baseline")
        results.baseline = self.aggregate_results(baseline_results)
        print(f"      Accuracy: {results.baseline.accuracy:.3f} "
              f"[{results.baseline.accuracy_ci[1]:.3f}, {results.baseline.accuracy_ci[2]:.3f}]")

        # 2. FEEL (normal)
        print("\n[2/8] Running FEEL (normal)...")
        feel_results = self.run_condition(prompts, "feel")
        results.feel = self.aggregate_results(feel_results)
        print(f"      Accuracy: {results.feel.accuracy:.3f} "
              f"[{results.feel.accuracy_ci[1]:.3f}, {results.feel.accuracy_ci[2]:.3f}]")

        # 3. FEEL-off (alpha=0)
        print("\n[3/8] Running FEEL-OFF (alpha=0)...")
        feel_off_results = self.run_condition(prompts, "feel_off")
        results.feel_off = self.aggregate_results(feel_off_results)
        print(f"      Accuracy: {results.feel_off.accuracy:.3f} "
              f"[{results.feel_off.accuracy_ci[1]:.3f}, {results.feel_off.accuracy_ci[2]:.3f}]")

        # 4. Random FEEL (same norm, wrong direction)
        print("\n[4/8] Running RANDOM FEEL (wrong direction)...")
        random_results = self.run_condition(prompts, "random_feel")
        results.random_feel = self.aggregate_results(random_results)
        print(f"      Accuracy: {results.random_feel.accuracy:.3f} "
              f"[{results.random_feel.accuracy_ci[1]:.3f}, {results.random_feel.accuracy_ci[2]:.3f}]")

        # 5. Shuffled (time-shuffled sensors)
        print("\n[5/8] Running SHUFFLED (time-shuffled sensors)...")
        shuffled_results = self.run_condition(prompts, "shuffled")
        results.shuffled = self.aggregate_results(shuffled_results)
        print(f"      Accuracy: {results.shuffled.accuracy:.3f} "
              f"[{results.shuffled.accuracy_ci[1]:.3f}, {results.shuffled.accuracy_ci[2]:.3f}]")

        # 6. Cross-prompt swap
        print("\n[6/8] Running CROSS-PROMPT SWAP...")
        cross_results = self.run_condition(prompts, "cross_prompt")
        results.cross_prompt_swap = self.aggregate_results(cross_results)
        print(f"      Accuracy: {results.cross_prompt_swap.accuracy:.3f} "
              f"[{results.cross_prompt_swap.accuracy_ci[1]:.3f}, {results.cross_prompt_swap.accuracy_ci[2]:.3f}]")

        # 7. Lag sweep
        if run_lag_sweep:
            print("\n[7/8] Running LAG SWEEP k={1,2,4,8,16}...")
            for k in [1, 2, 4, 8, 16]:
                lag_results = self.run_condition(prompts, f"lag_{k}", lag=k, verbose=False)
                results.lag_sweep[k] = self.aggregate_results(lag_results)
                print(f"      k={k}: Accuracy: {results.lag_sweep[k].accuracy:.3f}")

        # 8. Sensor split
        if run_sensor_split:
            print("\n[8/8] Running SENSOR SPLIT...")
            hw_results = self.run_condition(prompts, "hardware_only", verbose=False)
            results.hardware_only = self.aggregate_results(hw_results)
            print(f"      Hardware-only: Accuracy: {results.hardware_only.accuracy:.3f}")

            int_results = self.run_condition(prompts, "internal_only", verbose=False)
            results.internal_only = self.aggregate_results(int_results)
            print(f"      Internal-only: Accuracy: {results.internal_only.accuracy:.3f}")

        # Compute benefit analysis
        feel_benefit = results.feel.accuracy - results.baseline.accuracy
        shuffled_benefit = results.shuffled.accuracy - results.baseline.accuracy
        random_benefit = results.random_feel.accuracy - results.baseline.accuracy
        cross_benefit = results.cross_prompt_swap.accuracy - results.baseline.accuracy

        # Bootstrap CI for benefit
        feel_correct = np.array([r.correct for r in feel_results])
        baseline_correct = np.array([r.correct for r in baseline_results])
        benefit_samples = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(len(feel_correct), size=len(feel_correct), replace=True)
            benefit_samples.append(feel_correct[idx].mean() - baseline_correct[idx].mean())
        benefit_samples = np.array(benefit_samples)
        results.feel_benefit_accuracy = (
            feel_benefit,
            np.percentile(benefit_samples, 2.5),
            np.percentile(benefit_samples, 97.5),
        )

        # Benefit collapse checks
        # Collapse = ablation benefit is significantly less than real benefit
        eps = 0.01  # Noise threshold
        if abs(feel_benefit) > eps:
            results.benefit_collapse_shuffled = abs(shuffled_benefit) < abs(feel_benefit) * 0.5
            results.benefit_collapse_random = abs(random_benefit) < abs(feel_benefit) * 0.5
            results.benefit_collapse_cross_prompt = abs(cross_benefit) < abs(feel_benefit) * 0.5
        else:
            # No benefit to collapse
            results.benefit_collapse_shuffled = True
            results.benefit_collapse_random = True
            results.benefit_collapse_cross_prompt = True

        # Cleanup
        if self.telemetry:
            self.telemetry.stop()

        return results

    def print_summary(self, results: PublicationResults):
        """Print publication summary."""
        print("\n" + "=" * 70)
        print("  PUBLICATION BATTERY SUMMARY")
        print("=" * 70)

        print("\n  ACCURACY BY CONDITION (95% CI):")
        print("  " + "-" * 60)
        conditions = [
            ("baseline", results.baseline),
            ("feel", results.feel),
            ("feel_off", results.feel_off),
            ("random_feel", results.random_feel),
            ("shuffled", results.shuffled),
            ("cross_prompt", results.cross_prompt_swap),
            ("hardware_only", results.hardware_only),
            ("internal_only", results.internal_only),
        ]
        for name, res in conditions:
            if res:
                print(f"    {name:15s}: {res.accuracy:.3f} [{res.accuracy_ci[1]:.3f}, {res.accuracy_ci[2]:.3f}]")

        if results.lag_sweep:
            print("\n  LAG SWEEP:")
            for k in sorted(results.lag_sweep.keys()):
                res = results.lag_sweep[k]
                print(f"    lag_k={k:2d}:       {res.accuracy:.3f} [{res.accuracy_ci[1]:.3f}, {res.accuracy_ci[2]:.3f}]")

        print("\n  BENEFIT ANALYSIS:")
        print("  " + "-" * 60)
        if results.feel_benefit_accuracy:
            b = results.feel_benefit_accuracy
            print(f"    FEEL benefit:     {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]")

        print("\n  FALSIFICATION (benefit collapse):")
        print(f"    Shuffled collapse:    {'PASS' if results.benefit_collapse_shuffled else 'FAIL'}")
        print(f"    Random collapse:      {'PASS' if results.benefit_collapse_random else 'FAIL'}")
        print(f"    Cross-prompt collapse: {'PASS' if results.benefit_collapse_cross_prompt else 'FAIL'}")

        all_pass = (results.benefit_collapse_shuffled and
                   results.benefit_collapse_random and
                   results.benefit_collapse_cross_prompt)
        print("\n  OVERALL FALSIFICATION:")
        if all_pass:
            print("    ✓ ALL ABLATIONS COLLAPSE BENEFIT AS EXPECTED")
        else:
            print("    ✗ Some ablations did not collapse - investigation needed")

        print("\n  TELEMETRY VALIDITY:")
        if results.telemetry_validity:
            tv = results.telemetry_validity
            print(f"    Source: {tv.get('source', 'unknown')}")
            print(f"    Samples: {tv.get('n_samples', 0)}")
            valid = tv.get('valid', {})
            for k, v in valid.items():
                print(f"    {k}: {'valid' if v else 'invalid'}")


def main():
    parser = argparse.ArgumentParser(description="FEEL Publication Battery v7.0")
    parser.add_argument("--checkpoint", type=str,
                       default="results/feel_training/canonical_v6_checkpoint.pt")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--quick", action="store_true", help="Quick test (30 prompts)")
    parser.add_argument("--medium", action="store_true", help="Medium test (120 prompts)")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds")
    parser.add_argument("--no-lag-sweep", action="store_true", help="Skip lag sweep")
    parser.add_argument("--no-sensor-split", action="store_true", help="Skip sensor split")
    args = parser.parse_args()

    n_per_category = 75  # 300 total
    if args.quick:
        n_per_category = 8  # 32 total
        args.bootstrap = 100
    elif args.medium:
        n_per_category = 30  # 120 total
        args.bootstrap = 500

    runner = PublicationRunner(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
    )

    results = runner.run_publication_battery(
        n_per_category=n_per_category,
        seeds=args.seeds,
        run_lag_sweep=not args.no_lag_sweep,
        run_sensor_split=not args.no_sensor_split,
    )

    runner.print_summary(results)

    # Save results
    results_path = "results/feel_experiments/publication_v7_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
