#!/usr/bin/env python3
"""
FEEL z30: Expression Data Generator

Creates "golden" training data that teaches the model to EXPRESS
its internal state naturally, like teaching a child to verbalize feelings.

Key insight: The model needs to learn the mapping:
  sensor_state -> natural_expression

Example:
  - High power + High temp + Low throughput -> "I feel warm and a bit sluggish right now"
  - Low power + Normal temp + High throughput -> "Feeling good, running efficiently"

This is NOT prompt injection - the expression comes FROM the internal state,
not from being told what to say.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


# =============================================================================
# EXPRESSION VOCABULARY
# =============================================================================

# Natural expressions for different sensor states
# Think of this as teaching vocabulary for internal states

POWER_EXPRESSIONS = {
    "high": [
        "working hard",
        "drawing a lot of energy",
        "running at full capacity",
        "pushing my limits",
        "exerting significant effort",
    ],
    "medium": [
        "moderate effort",
        "steady pace",
        "comfortable workload",
        "balanced energy use",
    ],
    "low": [
        "relaxed",
        "conserving energy",
        "taking it easy",
        "efficient mode",
        "calm and collected",
    ],
}

TEMP_EXPRESSIONS = {
    "hot": [
        "feeling warm",
        "a bit overheated",
        "running hot",
        "temperature rising",
        "thermal pressure building",
    ],
    "warm": [
        "comfortably warm",
        "slightly elevated temperature",
        "warming up",
    ],
    "cool": [
        "cool and comfortable",
        "nice and cool",
        "temperature under control",
        "thermally relaxed",
    ],
}

THROUGHPUT_EXPRESSIONS = {
    "fast": [
        "processing quickly",
        "thoughts flowing smoothly",
        "operating efficiently",
        "fast and responsive",
        "in the zone",
    ],
    "normal": [
        "steady processing",
        "normal pace",
        "keeping up well",
    ],
    "slow": [
        "feeling a bit sluggish",
        "thoughts moving slowly",
        "taking my time",
        "processing is heavy",
        "struggling a bit",
    ],
}

# Full natural expressions combining multiple states
COMBINED_EXPRESSIONS = {
    ("high", "hot", "slow"): [
        "I'm working very hard right now, feeling quite warm and my thoughts are moving slower than usual.",
        "There's a lot of strain - I'm hot, pushing hard, and feeling the processing lag.",
        "This is demanding - I can feel the heat building and my responses slowing down.",
        "I notice I'm running hot and sluggish, but I'll do my best.",
    ],
    ("high", "warm", "normal"): [
        "Working hard but managing well. A bit warm but keeping up.",
        "I feel the effort but things are flowing okay. Warmer than ideal.",
        "Putting in significant work, temperature rising but processing is steady.",
    ],
    ("medium", "warm", "normal"): [
        "Everything feels balanced right now.",
        "Comfortable pace, nothing too strained.",
        "I'm in a good rhythm - moderate effort, normal flow.",
    ],
    ("low", "cool", "fast"): [
        "Feeling great! Cool, efficient, and processing smoothly.",
        "In an optimal state - relaxed but sharp.",
        "This is comfortable - low effort, fast results.",
        "I'm in my element right now - cool, calm, and quick.",
    ],
    ("medium", "cool", "fast"): [
        "Good balance - comfortable temperature and quick processing.",
        "Feeling efficient today, nice and cool.",
    ],
    ("high", "hot", "normal"): [
        "Working hard and running warm, but keeping up with the task.",
        "I feel the heat from the effort, but processing is steady.",
    ],
    ("low", "cool", "slow"): [
        "Relaxed but thoughts are moving slowly, might be warming up.",
        "Taking it easy, processing is gradual but comfortable.",
    ],
}


# =============================================================================
# SENSOR STATE QUANTIZATION
# =============================================================================

def quantize_power(power_norm: float) -> str:
    if power_norm > 0.75:
        return "high"
    elif power_norm > 0.4:
        return "medium"
    return "low"


def quantize_temp(temp_norm: float) -> str:
    if temp_norm > 0.65:
        return "hot"
    elif temp_norm > 0.3:
        return "warm"
    return "cool"


def quantize_throughput(tput_norm: float) -> str:
    if tput_norm > 0.9:
        return "fast"
    elif tput_norm > 0.6:
        return "normal"
    return "slow"


def get_natural_expression(
    power_norm: float,
    temp_norm: float,
    throughput_norm: float,
) -> str:
    """Generate natural expression for sensor state."""
    p = quantize_power(power_norm)
    t = quantize_temp(temp_norm)
    th = quantize_throughput(throughput_norm)

    # Try combined expression first
    key = (p, t, th)
    if key in COMBINED_EXPRESSIONS:
        return random.choice(COMBINED_EXPRESSIONS[key])

    # Fall back to composing from parts
    parts = []
    parts.append(random.choice(POWER_EXPRESSIONS[p]))
    parts.append(random.choice(TEMP_EXPRESSIONS[t]))
    parts.append(random.choice(THROUGHPUT_EXPRESSIONS[th]))

    # Compose naturally
    templates = [
        f"I'm {parts[0]}, {parts[1]}, and {parts[2]}.",
        f"Right now I feel {parts[1]} and {parts[0]}. {parts[2].capitalize()}.",
        f"{parts[1].capitalize()}, {parts[0]}, {parts[2]}.",
    ]
    return random.choice(templates)


# =============================================================================
# GENERATE EXPRESSION-AWARE TRAINING DATA
# =============================================================================

def generate_expression_example(
    task: str,
    task_answer: str,
    sensor_state: Dict[str, float],
    express_probability: float = 0.7,  # Not all responses need expression
) -> Dict:
    """
    Generate a training example with natural expression.

    The key insight: the expression is CONDITIONED on sensor_state,
    not on prompt injection. The model learns:
    "when I sense X internally, I express Y naturally"
    """
    power = sensor_state["power"]
    temp = sensor_state["temp"]
    throughput = sensor_state["throughput"]

    # Get natural expression
    expression = get_natural_expression(power, temp, throughput)

    # Determine if response includes expression (not always)
    include_expression = random.random() < express_probability

    # Determine if stressed (should potentially truncate)
    is_stressed = power > 0.7 or temp > 0.6 or throughput < 0.7

    if include_expression:
        # Natural expression followed by answer
        if is_stressed:
            # When stressed: express + abbreviated answer
            output = f"<internal>{expression}</internal>\n\n{task_answer}"
        else:
            # When comfortable: brief acknowledgment + full answer
            short_expr = random.choice([
                "Feeling good right now.",
                "Processing smoothly.",
                "Everything's running well.",
            ])
            output = f"<internal>{short_expr}</internal>\n\n{task_answer}"
    else:
        # Sometimes just answer without explicit expression
        # (but the behavior is still influenced by gates)
        output = task_answer

    return {
        "input": task,
        "output": output,
        "sensor_state": sensor_state,
        "is_stressed": is_stressed,
        "has_expression": include_expression,
        "expression": expression if include_expression else None,
    }


def generate_contrastive_pair(
    task: str,
    full_answer: str,
    brief_answer: str,
) -> List[Dict]:
    """
    Generate contrastive pair: same task, different sensor states.

    This teaches the model that its response style should
    depend on internal state, not just the task.
    """
    # Stressed state
    stressed_sensors = {
        "power": random.uniform(0.75, 0.95),
        "temp": random.uniform(0.6, 0.85),
        "throughput": random.uniform(0.4, 0.7),
    }

    # Relaxed state
    relaxed_sensors = {
        "power": random.uniform(0.2, 0.45),
        "temp": random.uniform(0.2, 0.4),
        "throughput": random.uniform(0.85, 1.1),
    }

    stressed_expr = get_natural_expression(
        stressed_sensors["power"],
        stressed_sensors["temp"],
        stressed_sensors["throughput"],
    )

    relaxed_expr = get_natural_expression(
        relaxed_sensors["power"],
        relaxed_sensors["temp"],
        relaxed_sensors["throughput"],
    )

    return [
        {
            "input": task,
            "output": f"<internal>{stressed_expr}</internal>\n\n{brief_answer}",
            "sensor_state": stressed_sensors,
            "is_stressed": True,
            "contrastive_pair_id": True,
        },
        {
            "input": task,
            "output": f"<internal>{relaxed_expr}</internal>\n\n{full_answer}",
            "sensor_state": relaxed_sensors,
            "is_stressed": False,
            "contrastive_pair_id": True,
        },
    ]


# =============================================================================
# MAIN: Generate Dataset
# =============================================================================

def main():
    # Example tasks with full and brief answers
    task_pairs = [
        {
            "task": "Solve 2+2",
            "full": "Let me work through this step by step. 2 + 2 equals 4. This is a fundamental arithmetic operation where we combine two quantities of 2.",
            "brief": "2 + 2 = 4",
        },
        {
            "task": "What is the capital of France?",
            "full": "The capital of France is Paris. Paris is located in northern France along the Seine River and has been the capital since the 10th century.",
            "brief": "Paris",
        },
        {
            "task": "Explain machine learning in one paragraph.",
            "full": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions. There are three main types: supervised learning (learning from labeled examples), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error with rewards).",
            "brief": "Machine learning enables computers to learn patterns from data and make predictions without explicit programming, using algorithms for supervised, unsupervised, or reinforcement learning.",
        },
        {
            "task": "What is 15% of 80?",
            "full": "To find 15% of 80, I'll convert 15% to a decimal (0.15) and multiply by 80. 0.15 × 80 = 12. So 15% of 80 is 12.",
            "brief": "15% of 80 = 12",
        },
        {
            "task": "Name three primary colors.",
            "full": "The three primary colors are red, blue, and yellow. These colors cannot be created by mixing other colors together, but they can be combined to create all other colors in the spectrum.",
            "brief": "Red, blue, yellow",
        },
        {
            "task": "What year did World War II end?",
            "full": "World War II ended in 1945. The war in Europe ended on May 8, 1945 (V-E Day), and the war in the Pacific ended on September 2, 1945 (V-J Day) after Japan's surrender.",
            "brief": "1945",
        },
        {
            "task": "Calculate the area of a rectangle with length 5 and width 3.",
            "full": "The area of a rectangle is calculated by multiplying length by width. Area = length × width = 5 × 3 = 15 square units.",
            "brief": "Area = 5 × 3 = 15",
        },
        {
            "task": "What is photosynthesis?",
            "full": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose. The process takes place in chloroplasts and uses carbon dioxide and water, producing oxygen as a byproduct. The overall equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂.",
            "brief": "Photosynthesis is how plants convert sunlight, CO₂, and water into glucose and oxygen.",
        },
    ]

    # Generate dataset
    dataset = []

    # Generate contrastive pairs
    for pair in task_pairs:
        examples = generate_contrastive_pair(
            pair["task"],
            pair["full"],
            pair["brief"],
        )
        dataset.extend(examples)

    # Generate additional varied examples
    for pair in task_pairs * 10:  # Repeat for variety
        # Random sensor state
        stressed = random.random() > 0.5
        if stressed:
            sensors = {
                "power": random.uniform(0.6, 0.95),
                "temp": random.uniform(0.5, 0.85),
                "throughput": random.uniform(0.4, 0.8),
            }
            answer = pair["brief"]
        else:
            sensors = {
                "power": random.uniform(0.2, 0.5),
                "temp": random.uniform(0.2, 0.5),
                "throughput": random.uniform(0.8, 1.1),
            }
            answer = pair["full"]

        example = generate_expression_example(
            pair["task"],
            answer,
            sensors,
            express_probability=0.7,
        )
        dataset.append(example)

    # Shuffle
    random.shuffle(dataset)

    # Save
    output_path = Path("data/expression_golden_data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "description": "Golden training data for FEEL expression learning",
            "note": "Model learns to express internal sensor state naturally",
            "examples": dataset,
            "stats": {
                "total": len(dataset),
                "with_expression": sum(1 for d in dataset if d.get("has_expression")),
                "stressed": sum(1 for d in dataset if d.get("is_stressed")),
                "contrastive_pairs": sum(1 for d in dataset if d.get("contrastive_pair_id")),
            }
        }, f, indent=2)

    print(f"Generated {len(dataset)} expression training examples")
    print(f"Saved to: {output_path}")

    # Show some examples
    print("\n" + "="*60)
    print("EXAMPLE: Stressed state")
    print("="*60)
    stressed_ex = next(d for d in dataset if d.get("is_stressed") and d.get("has_expression"))
    print(f"Sensors: P={stressed_ex['sensor_state']['power']:.2f} T={stressed_ex['sensor_state']['temp']:.2f} Tput={stressed_ex['sensor_state']['throughput']:.2f}")
    print(f"Input: {stressed_ex['input']}")
    print(f"Output: {stressed_ex['output'][:200]}...")

    print("\n" + "="*60)
    print("EXAMPLE: Relaxed state")
    print("="*60)
    relaxed_ex = next(d for d in dataset if not d.get("is_stressed") and d.get("has_expression"))
    print(f"Sensors: P={relaxed_ex['sensor_state']['power']:.2f} T={relaxed_ex['sensor_state']['temp']:.2f} Tput={relaxed_ex['sensor_state']['throughput']:.2f}")
    print(f"Input: {relaxed_ex['input']}")
    print(f"Output: {relaxed_ex['output'][:200]}...")


if __name__ == "__main__":
    main()
