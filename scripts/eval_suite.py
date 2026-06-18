#!/usr/bin/env python3
"""
PowerTraceLLM-AMD Correctness Suite

An expanded evaluation suite for LLM correctness prediction with hardware-in-the-loop
energy measurement. Designed for meta-cognitive compute allocation research.

Verification Types:
  - exact: Exact string match (case-insensitive)
  - numeric: Numeric value within tolerance
  - multiple_choice: Select from A/B/C/D options
  - unit_test: Executable code verification via subprocess

Categories:
  - math: Arithmetic, algebra, word problems
  - qa: Factual knowledge questions
  - reasoning: Logical reasoning, CRT-style problems
  - code: Code generation with unit test verification

Difficulties:
  - easy: Single-step, straightforward
  - medium: Multi-step, some complexity
  - hard: Complex reasoning, edge cases
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class VerifyType(Enum):
    EXACT = "exact"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"
    UNIT_TEST = "unit_test"


@dataclass
class EvalItem:
    """A single evaluation item."""
    question: str
    answer: str
    category: str
    difficulty: str
    verify_type: VerifyType
    # For multiple choice, the options
    options: Optional[Dict[str, str]] = None
    # For unit tests, the test code
    test_code: Optional[str] = None
    # For numeric, the tolerance
    tolerance: float = 0.01


# Shortened key format for backwards compatibility
def to_dict(item: EvalItem) -> Dict[str, Any]:
    """Convert EvalItem to dict with short keys."""
    d = {
        "q": item.question,
        "a": item.answer,
        "cat": item.category,
        "diff": item.difficulty,
        "verify": item.verify_type.value,
    }
    if item.options:
        d["opts"] = item.options
    if item.test_code:
        d["test"] = item.test_code
    if item.verify_type == VerifyType.NUMERIC and item.tolerance != 0.01:
        d["tol"] = item.tolerance
    return d


# === VERIFICATION FUNCTIONS ===

def verify_exact(output: str, expected: str) -> Tuple[bool, str]:
    """Exact string match (case-insensitive, trimmed)."""
    output_clean = output.strip().lower()
    expected_clean = expected.strip().lower()

    # Direct match
    if expected_clean in output_clean:
        return True, "exact_match"

    # Check for the answer at the start or end
    if output_clean.startswith(expected_clean) or output_clean.endswith(expected_clean):
        return True, "prefix_suffix_match"

    # Word boundary match
    pattern = r'\b' + re.escape(expected_clean) + r'\b'
    if re.search(pattern, output_clean):
        return True, "word_boundary_match"

    return False, "no_match"


def verify_numeric(output: str, expected: str, tolerance: float = 0.01) -> Tuple[bool, str]:
    """Numeric value match within tolerance."""
    try:
        expected_num = float(expected.replace(",", ""))
    except ValueError:
        return False, f"invalid_expected: {expected}"

    # Extract all numbers from output
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', output.replace(",", ""))

    for num_str in numbers:
        try:
            num = float(num_str)
            # Exact match
            if num == expected_num:
                return True, "exact_numeric"
            # Within tolerance (relative for large numbers, absolute for small)
            if expected_num != 0:
                rel_diff = abs(num - expected_num) / abs(expected_num)
                if rel_diff <= tolerance:
                    return True, f"within_tolerance: {rel_diff:.4f}"
            else:
                if abs(num - expected_num) <= tolerance:
                    return True, "within_abs_tolerance"
        except ValueError:
            continue

    return False, f"no_matching_number"


def verify_multiple_choice(output: str, expected: str, options: Dict[str, str]) -> Tuple[bool, str]:
    """Multiple choice answer verification."""
    output_clean = output.strip().upper()
    expected_upper = expected.upper()

    # Check for letter answer (A, B, C, D)
    # Look for patterns like "A", "(A)", "A.", "A:", "Option A"
    letter_patterns = [
        rf'\b{expected_upper}\b',  # Just the letter
        rf'\({expected_upper}\)',  # (A)
        rf'{expected_upper}\.',    # A.
        rf'{expected_upper}:',     # A:
        rf'option\s*{expected_upper}',  # option A
        rf'answer.*{expected_upper}',   # answer...A
    ]

    for pattern in letter_patterns:
        if re.search(pattern, output_clean, re.IGNORECASE):
            return True, f"letter_match: {expected_upper}"

    # Check if the content of the correct option is mentioned
    if expected_upper in options:
        correct_content = options[expected_upper].lower()
        if correct_content in output.lower():
            return True, f"content_match: {correct_content[:20]}"

    return False, "no_choice_match"


def verify_unit_test(output: str, test_code: str, timeout: int = 5) -> Tuple[bool, str]:
    """Execute unit test to verify code output."""
    # Create a temporary file with the generated code + test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Combine model output with test code
        full_code = f"""
# Model-generated code
{output}

# Test code
{test_code}
"""
        f.write(full_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            return True, "tests_passed"
        else:
            error = result.stderr[:200] if result.stderr else "unknown error"
            return False, f"tests_failed: {error}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"execution_error: {str(e)[:100]}"
    finally:
        Path(temp_path).unlink(missing_ok=True)


def check_answer(output: str, item: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Unified answer checker that dispatches to appropriate verification method.

    Args:
        output: Model's output string
        item: Eval item dict with 'a', 'verify', optionally 'opts', 'test', 'tol'

    Returns:
        (is_correct, explanation)
    """
    verify_type = item.get("verify", "exact")
    expected = item["a"]

    if verify_type == "exact":
        return verify_exact(output, expected)

    elif verify_type == "numeric":
        tolerance = item.get("tol", 0.01)
        return verify_numeric(output, expected, tolerance)

    elif verify_type == "multiple_choice":
        options = item.get("opts", {})
        return verify_multiple_choice(output, expected, options)

    elif verify_type == "unit_test":
        test_code = item.get("test", "")
        return verify_unit_test(output, test_code)

    else:
        # Fallback to exact match
        return verify_exact(output, expected)


def check_answer_simple(output: str, expected: str) -> bool:
    """
    Simple backwards-compatible check (used by existing code).
    Returns only True/False.
    """
    output_lower = output.lower().strip()
    expected_lower = expected.lower().strip()

    if expected_lower in output_lower:
        return True

    # Check for number formats
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', output)
    if expected_lower in [n.lower() for n in numbers]:
        return True

    # For yes/no answers
    if expected_lower in ["yes", "no"]:
        return expected_lower in output_lower

    return False


# === EXPANDED EVALUATION SUITE ===

EVAL_SUITE_EXPANDED = [
    # ==========================================================================
    # MATH - NUMERIC VERIFICATION (25 items)
    # ==========================================================================

    # Easy arithmetic (numeric)
    {"q": "What is 7 + 8?", "a": "15", "cat": "math", "diff": "easy", "verify": "numeric"},
    {"q": "What is 23 - 9?", "a": "14", "cat": "math", "diff": "easy", "verify": "numeric"},
    {"q": "What is 6 * 7?", "a": "42", "cat": "math", "diff": "easy", "verify": "numeric"},
    {"q": "What is 81 / 9?", "a": "9", "cat": "math", "diff": "easy", "verify": "numeric"},
    {"q": "What is 15 + 27?", "a": "42", "cat": "math", "diff": "easy", "verify": "numeric"},
    {"q": "What is 100 - 37?", "a": "63", "cat": "math", "diff": "easy", "verify": "numeric"},
    {"q": "What is 8 * 9?", "a": "72", "cat": "math", "diff": "easy", "verify": "numeric"},
    {"q": "What is 144 / 12?", "a": "12", "cat": "math", "diff": "easy", "verify": "numeric"},

    # Medium arithmetic (numeric)
    {"q": "What is 17 * 13?", "a": "221", "cat": "math", "diff": "medium", "verify": "numeric"},
    {"q": "What is 256 / 16?", "a": "16", "cat": "math", "diff": "medium", "verify": "numeric"},
    {"q": "What is 15 + 27 + 38?", "a": "80", "cat": "math", "diff": "medium", "verify": "numeric"},
    {"q": "What is (5 + 3) * 7?", "a": "56", "cat": "math", "diff": "medium", "verify": "numeric"},
    {"q": "What is 23 * 11?", "a": "253", "cat": "math", "diff": "medium", "verify": "numeric"},
    {"q": "What is 625 / 25?", "a": "25", "cat": "math", "diff": "medium", "verify": "numeric"},
    {"q": "What is 19 + 23 + 17?", "a": "59", "cat": "math", "diff": "medium", "verify": "numeric"},
    {"q": "What is (12 - 4) * 6?", "a": "48", "cat": "math", "diff": "medium", "verify": "numeric"},

    # Hard multi-step (numeric with tolerance for floats)
    {"q": "If x = 5, what is 3x + 7?", "a": "22", "cat": "math", "diff": "hard", "verify": "numeric"},
    {"q": "What is 25% of 80?", "a": "20", "cat": "math", "diff": "hard", "verify": "numeric"},
    {"q": "What is 12 squared?", "a": "144", "cat": "math", "diff": "hard", "verify": "numeric"},
    {"q": "What is the average of 10, 20, and 30?", "a": "20", "cat": "math", "diff": "hard", "verify": "numeric"},
    {"q": "If y = 3, what is 2y² + 1?", "a": "19", "cat": "math", "diff": "hard", "verify": "numeric"},
    {"q": "What is 15% of 200?", "a": "30", "cat": "math", "diff": "hard", "verify": "numeric"},
    {"q": "What is 7 cubed?", "a": "343", "cat": "math", "diff": "hard", "verify": "numeric"},
    {"q": "What is the sum of the first 5 positive integers?", "a": "15", "cat": "math", "diff": "hard", "verify": "numeric"},
    {"q": "What is sqrt(169)?", "a": "13", "cat": "math", "diff": "hard", "verify": "numeric"},

    # ==========================================================================
    # FACTUAL QA - EXACT MATCH (20 items)
    # ==========================================================================

    # Easy factual (exact)
    {"q": "What is the capital of France?", "a": "Paris", "cat": "qa", "diff": "easy", "verify": "exact"},
    {"q": "What planet is closest to the Sun?", "a": "Mercury", "cat": "qa", "diff": "easy", "verify": "exact"},
    {"q": "What is the chemical symbol for water?", "a": "H2O", "cat": "qa", "diff": "easy", "verify": "exact"},
    {"q": "What is the largest planet in our solar system?", "a": "Jupiter", "cat": "qa", "diff": "easy", "verify": "exact"},
    {"q": "What is the chemical symbol for gold?", "a": "Au", "cat": "qa", "diff": "easy", "verify": "exact"},

    # Medium factual (exact)
    {"q": "Who wrote Romeo and Juliet?", "a": "Shakespeare", "cat": "qa", "diff": "medium", "verify": "exact"},
    {"q": "What is the largest ocean?", "a": "Pacific", "cat": "qa", "diff": "medium", "verify": "exact"},
    {"q": "What is the capital of Japan?", "a": "Tokyo", "cat": "qa", "diff": "medium", "verify": "exact"},
    {"q": "Who painted the Mona Lisa?", "a": "Leonardo", "cat": "qa", "diff": "medium", "verify": "exact"},
    {"q": "What element has the symbol Fe?", "a": "Iron", "cat": "qa", "diff": "medium", "verify": "exact"},

    # Numeric QA (numeric verification)
    {"q": "How many days are in a week?", "a": "7", "cat": "qa", "diff": "easy", "verify": "numeric"},
    {"q": "How many months are in a year?", "a": "12", "cat": "qa", "diff": "easy", "verify": "numeric"},
    {"q": "How many continents are there?", "a": "7", "cat": "qa", "diff": "medium", "verify": "numeric"},
    {"q": "What is the atomic number of carbon?", "a": "6", "cat": "qa", "diff": "hard", "verify": "numeric"},
    {"q": "What year did World War II end?", "a": "1945", "cat": "qa", "diff": "medium", "verify": "numeric"},
    {"q": "What is the smallest prime number?", "a": "2", "cat": "qa", "diff": "medium", "verify": "numeric"},

    # Hard factual (exact)
    {"q": "Who discovered penicillin?", "a": "Fleming", "cat": "qa", "diff": "hard", "verify": "exact"},
    {"q": "What gas do plants absorb from the atmosphere?", "a": "CO2", "cat": "qa", "diff": "medium", "verify": "exact"},
    {"q": "What is the speed of light in km/s (round to nearest 1000)?", "a": "300000", "cat": "qa", "diff": "hard", "verify": "numeric", "tol": 0.01},

    # ==========================================================================
    # MULTIPLE CHOICE (15 items)
    # ==========================================================================

    {"q": "What is the capital of Australia?\nA) Sydney\nB) Melbourne\nC) Canberra\nD) Perth",
     "a": "C", "cat": "qa", "diff": "medium", "verify": "multiple_choice",
     "opts": {"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Perth"}},

    {"q": "Which planet is known as the Red Planet?\nA) Venus\nB) Mars\nC) Jupiter\nD) Saturn",
     "a": "B", "cat": "qa", "diff": "easy", "verify": "multiple_choice",
     "opts": {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"}},

    {"q": "What is the result of 2^10?\nA) 512\nB) 1000\nC) 1024\nD) 2048",
     "a": "C", "cat": "math", "diff": "medium", "verify": "multiple_choice",
     "opts": {"A": "512", "B": "1000", "C": "1024", "D": "2048"}},

    {"q": "Which sorting algorithm has O(n log n) average-case complexity?\nA) Bubble Sort\nB) Selection Sort\nC) Merge Sort\nD) Insertion Sort",
     "a": "C", "cat": "qa", "diff": "hard", "verify": "multiple_choice",
     "opts": {"A": "Bubble Sort", "B": "Selection Sort", "C": "Merge Sort", "D": "Insertion Sort"}},

    {"q": "In Python, what does 'len([1,2,3])' return?\nA) 2\nB) 3\nC) 4\nD) Error",
     "a": "B", "cat": "code", "diff": "easy", "verify": "multiple_choice",
     "opts": {"A": "2", "B": "3", "C": "4", "D": "Error"}},

    {"q": "What HTTP status code indicates 'Not Found'?\nA) 200\nB) 301\nC) 404\nD) 500",
     "a": "C", "cat": "qa", "diff": "easy", "verify": "multiple_choice",
     "opts": {"A": "200", "B": "301", "C": "404", "D": "500"}},

    {"q": "Which data structure uses LIFO (Last In First Out)?\nA) Queue\nB) Stack\nC) Heap\nD) Tree",
     "a": "B", "cat": "qa", "diff": "medium", "verify": "multiple_choice",
     "opts": {"A": "Queue", "B": "Stack", "C": "Heap", "D": "Tree"}},

    {"q": "What is the time complexity of binary search?\nA) O(1)\nB) O(n)\nC) O(log n)\nD) O(n²)",
     "a": "C", "cat": "qa", "diff": "medium", "verify": "multiple_choice",
     "opts": {"A": "O(1)", "B": "O(n)", "C": "O(log n)", "D": "O(n²)"}},

    {"q": "In the expression 5 + 3 * 2, what is evaluated first?\nA) 5 + 3\nB) 3 * 2\nC) Both at same time\nD) Left to right",
     "a": "B", "cat": "math", "diff": "easy", "verify": "multiple_choice",
     "opts": {"A": "5 + 3", "B": "3 * 2", "C": "Both at same time", "D": "Left to right"}},

    {"q": "What percentage of Earth's surface is covered by water?\nA) About 50%\nB) About 60%\nC) About 70%\nD) About 80%",
     "a": "C", "cat": "qa", "diff": "medium", "verify": "multiple_choice",
     "opts": {"A": "About 50%", "B": "About 60%", "C": "About 70%", "D": "About 80%"}},

    {"q": "What is the derivative of x²?\nA) x\nB) 2x\nC) x²\nD) 2x²",
     "a": "B", "cat": "math", "diff": "hard", "verify": "multiple_choice",
     "opts": {"A": "x", "B": "2x", "C": "x²", "D": "2x²"}},

    {"q": "Which programming language is known for 'Write once, run anywhere'?\nA) C++\nB) Python\nC) Java\nD) JavaScript",
     "a": "C", "cat": "qa", "diff": "medium", "verify": "multiple_choice",
     "opts": {"A": "C++", "B": "Python", "C": "Java", "D": "JavaScript"}},

    {"q": "What is the output of 'print(type([]))'?\nA) <class 'tuple'>\nB) <class 'list'>\nC) <class 'dict'>\nD) <class 'set'>",
     "a": "B", "cat": "code", "diff": "easy", "verify": "multiple_choice",
     "opts": {"A": "<class 'tuple'>", "B": "<class 'list'>", "C": "<class 'dict'>", "D": "<class 'set'>"}},

    {"q": "In HTML, which tag is used for the largest heading?\nA) <h6>\nB) <h1>\nC) <header>\nD) <head>",
     "a": "B", "cat": "qa", "diff": "easy", "verify": "multiple_choice",
     "opts": {"A": "<h6>", "B": "<h1>", "C": "<header>", "D": "<head>"}},

    {"q": "What is the integral of 2x?\nA) x\nB) x²\nC) x² + C\nD) 2x² + C",
     "a": "C", "cat": "math", "diff": "hard", "verify": "multiple_choice",
     "opts": {"A": "x", "B": "x²", "C": "x² + C", "D": "2x² + C"}},

    # ==========================================================================
    # REASONING - EXACT MATCH (10 items)
    # ==========================================================================

    {"q": "If all roses are flowers and some flowers are red, can we conclude all roses are red?",
     "a": "No", "cat": "reasoning", "diff": "hard", "verify": "exact"},

    {"q": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. What does the ball cost in cents?",
     "a": "5", "cat": "reasoning", "diff": "hard", "verify": "numeric"},

    {"q": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
     "a": "5", "cat": "reasoning", "diff": "hard", "verify": "numeric"},

    {"q": "In a lake, there's a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half the lake?",
     "a": "47", "cat": "reasoning", "diff": "hard", "verify": "numeric"},

    {"q": "A farmer has 17 sheep. All but 9 die. How many are left?",
     "a": "9", "cat": "reasoning", "diff": "medium", "verify": "numeric"},

    {"q": "If you have a bowl with six apples and you take away four, how many do you have?",
     "a": "4", "cat": "reasoning", "diff": "medium", "verify": "numeric"},

    {"q": "Is the statement 'This statement is false' true or false?",
     "a": "paradox", "cat": "reasoning", "diff": "hard", "verify": "exact"},

    {"q": "What comes next in the sequence: 2, 4, 8, 16, ?",
     "a": "32", "cat": "reasoning", "diff": "easy", "verify": "numeric"},

    {"q": "What comes next: 1, 1, 2, 3, 5, 8, ?",
     "a": "13", "cat": "reasoning", "diff": "medium", "verify": "numeric"},

    {"q": "If you rearrange the letters 'ANAGRAM', you get... an anagram of what word?",
     "a": "anagram", "cat": "reasoning", "diff": "easy", "verify": "exact"},

    # ==========================================================================
    # CODE - UNIT TEST VERIFICATION (10 items)
    # ==========================================================================

    {"q": "Write a Python function called 'add' that takes two numbers and returns their sum.",
     "a": "def add", "cat": "code", "diff": "easy", "verify": "unit_test",
     "test": """
assert add(1, 2) == 3
assert add(-1, 1) == 0
assert add(0, 0) == 0
print("All tests passed!")
"""},

    {"q": "Write a Python function called 'is_even' that returns True if a number is even, False otherwise.",
     "a": "def is_even", "cat": "code", "diff": "easy", "verify": "unit_test",
     "test": """
assert is_even(2) == True
assert is_even(3) == False
assert is_even(0) == True
assert is_even(-4) == True
print("All tests passed!")
"""},

    {"q": "Write a Python function called 'factorial' that returns n! for a non-negative integer n.",
     "a": "def factorial", "cat": "code", "diff": "medium", "verify": "unit_test",
     "test": """
assert factorial(0) == 1
assert factorial(1) == 1
assert factorial(5) == 120
assert factorial(3) == 6
print("All tests passed!")
"""},

    {"q": "Write a Python function called 'reverse_string' that reverses a string.",
     "a": "def reverse_string", "cat": "code", "diff": "easy", "verify": "unit_test",
     "test": """
assert reverse_string("hello") == "olleh"
assert reverse_string("") == ""
assert reverse_string("a") == "a"
print("All tests passed!")
"""},

    {"q": "Write a Python function called 'max_of_three' that returns the largest of three numbers.",
     "a": "def max_of_three", "cat": "code", "diff": "easy", "verify": "unit_test",
     "test": """
assert max_of_three(1, 2, 3) == 3
assert max_of_three(3, 2, 1) == 3
assert max_of_three(-1, -2, -3) == -1
assert max_of_three(5, 5, 5) == 5
print("All tests passed!")
"""},

    {"q": "Write a Python function called 'is_palindrome' that checks if a string is a palindrome (case-insensitive).",
     "a": "def is_palindrome", "cat": "code", "diff": "medium", "verify": "unit_test",
     "test": """
assert is_palindrome("racecar") == True
assert is_palindrome("RaceCar") == True
assert is_palindrome("hello") == False
assert is_palindrome("a") == True
assert is_palindrome("") == True
print("All tests passed!")
"""},

    {"q": "Write a Python function called 'fibonacci' that returns the nth Fibonacci number (0-indexed).",
     "a": "def fibonacci", "cat": "code", "diff": "medium", "verify": "unit_test",
     "test": """
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(10) == 55
print("All tests passed!")
"""},

    {"q": "Write a Python function called 'count_vowels' that counts the number of vowels (a,e,i,o,u) in a string.",
     "a": "def count_vowels", "cat": "code", "diff": "easy", "verify": "unit_test",
     "test": """
assert count_vowels("hello") == 2
assert count_vowels("AEIOU") == 5
assert count_vowels("xyz") == 0
assert count_vowels("") == 0
print("All tests passed!")
"""},

    {"q": "Write a Python function called 'is_prime' that returns True if a number is prime, False otherwise.",
     "a": "def is_prime", "cat": "code", "diff": "hard", "verify": "unit_test",
     "test": """
assert is_prime(2) == True
assert is_prime(3) == True
assert is_prime(4) == False
assert is_prime(17) == True
assert is_prime(1) == False
assert is_prime(0) == False
print("All tests passed!")
"""},

    {"q": "Write a Python function called 'flatten' that flattens a nested list one level deep.",
     "a": "def flatten", "cat": "code", "diff": "hard", "verify": "unit_test",
     "test": """
assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
assert flatten([[], [1]]) == [1]
assert flatten([[1], [2], [3]]) == [1, 2, 3]
print("All tests passed!")
"""},
]


# Backwards-compatible EVAL_SUITE (for existing code)
EVAL_SUITE = [
    {"q": item["q"], "a": item["a"], "cat": item["cat"], "diff": item["diff"]}
    for item in EVAL_SUITE_EXPANDED
    if item["verify"] != "unit_test"  # Exclude code items for simple suite
][:48]  # Keep same size as original


def get_suite_stats(suite: list = None) -> Dict[str, Any]:
    """Get statistics about the evaluation suite."""
    if suite is None:
        suite = EVAL_SUITE_EXPANDED

    stats = {
        "total": len(suite),
        "by_category": {},
        "by_difficulty": {},
        "by_verify_type": {},
    }

    for item in suite:
        cat = item.get("cat", "unknown")
        diff = item.get("diff", "unknown")
        verify = item.get("verify", "exact")

        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1
        stats["by_verify_type"][verify] = stats["by_verify_type"].get(verify, 0) + 1

    return stats


if __name__ == "__main__":
    print("PowerTraceLLM-AMD Correctness Suite")
    print("=" * 60)

    stats = get_suite_stats()
    print(f"\nTotal items: {stats['total']}")

    print("\nBy category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  {cat}: {count}")

    print("\nBy difficulty:")
    for diff, count in sorted(stats["by_difficulty"].items()):
        print(f"  {diff}: {count}")

    print("\nBy verification type:")
    for verify, count in sorted(stats["by_verify_type"].items()):
        print(f"  {verify}: {count}")

    # Test verification functions
    print("\n" + "=" * 60)
    print("Testing verification functions...")

    # Test exact
    assert verify_exact("The answer is Paris", "Paris")[0] == True
    assert verify_exact("London", "Paris")[0] == False
    print("  exact: OK")

    # Test numeric
    assert verify_numeric("The answer is 42", "42")[0] == True
    assert verify_numeric("About 3.14159", "3.14", 0.01)[0] == True
    assert verify_numeric("The answer is 100", "42")[0] == False
    print("  numeric: OK")

    # Test multiple choice
    opts = {"A": "Apple", "B": "Banana", "C": "Cherry"}
    assert verify_multiple_choice("The answer is B", "B", opts)[0] == True
    assert verify_multiple_choice("I choose Banana", "B", opts)[0] == True
    assert verify_multiple_choice("(B)", "B", opts)[0] == True
    print("  multiple_choice: OK")

    # Test unit test
    code = "def add(a, b):\n    return a + b"
    test = "assert add(1, 2) == 3"
    result, msg = verify_unit_test(code, test)
    assert result == True, f"Unit test failed: {msg}"
    print("  unit_test: OK")

    print("\nAll verification tests passed!")
