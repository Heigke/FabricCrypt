#!/usr/bin/env python3
"""
Stress & Recovery Test - Proving Biological Inertia

This script pushes the EmbodiedAgentV9 to its limits to observe:
1. Fatigue accumulation under sustained load
2. State transitions through the full emotional manifold
3. Recovery curves when load is removed
4. The "breathing" pattern of a true organism

Scientific Goal:
- Prove the agent has INERTIA (doesn't instantly recover)
- Observe the full feeling spectrum (FOCUSED → FLOW_STATE → DETERMINATION → STRAINED → EXHAUSTED)
- Generate data for Introspective Fine-Tuning (IFT)
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from src.embodied_agent import (
    EmbodiedAgentV9,
    SomaticState,
    AllostaticRegulator,
    SomaticCortex,
    GradientKController,
)


# ============================================================
# Stress Test Configuration
# ============================================================

# Heavy prompts that demand computation
STRESS_PROMPTS = [
    "Write an extremely detailed technical analysis of quantum computing, covering qubits, superposition, entanglement, quantum gates, error correction, and the path to quantum supremacy. Include mathematical formulations.",
    "Explain the complete history of artificial intelligence from Alan Turing to modern large language models. Cover every major milestone, breakthrough, and paradigm shift in exhaustive detail.",
    "Provide a comprehensive comparison of all major programming paradigms: imperative, object-oriented, functional, logic, and declarative. Include code examples in multiple languages for each.",
    "Describe the entire human nervous system from neurons to consciousness. Cover the brain, spinal cord, peripheral nerves, neurotransmitters, and the mechanisms of thought and emotion.",
    "Write a complete business plan for a Mars colonization company including technical requirements, funding strategy, timeline, risk analysis, and regulatory considerations.",
    "Explain Einstein's theories of special and general relativity with full mathematical derivations. Cover time dilation, length contraction, mass-energy equivalence, and gravitational waves.",
    "Provide an exhaustive analysis of global climate change including causes, effects, feedback loops, tipping points, mitigation strategies, and socioeconomic implications.",
    "Describe the complete process of drug development from target discovery to FDA approval. Cover preclinical research, clinical trials, regulatory requirements, and market authorization.",
]

# Light prompts for recovery
RECOVERY_PROMPTS = [
    "Say hello.",
    "What is 2+2?",
    "Name a color.",
    "Count to 5.",
]


@dataclass
class StressTestResult:
    """Results from a stress/recovery cycle."""
    phase: str  # 'stress' or 'recovery'
    generation: int
    timestamp: float
    prompt: str

    # Somatic state
    metabolic: float
    thermal: float
    cognitive: float
    overall_stress: float
    fatigue: float
    feeling: str

    # Regulation
    k: int
    mode: str

    # Hardware
    temp: float
    pwr_1: float

    # Output
    response_length: int
    response_snippet: str


def run_stress_test(
    agent: EmbodiedAgentV9,
    stress_generations: int = 15,
    recovery_generations: int = 10,
    max_tokens: int = 200,
) -> List[StressTestResult]:
    """
    Run a complete stress/recovery cycle.

    Phase 1: STRESS
        - Heavy prompts with long max_tokens
        - Push fatigue as high as possible
        - Observe feeling transitions

    Phase 2: RECOVERY
        - Light prompts with short max_tokens
        - Watch fatigue decay
        - Measure recovery time
    """
    results = []
    start_time = time.time()

    print("\n" + "=" * 70)
    print("  PHASE 1: STRESS (Pushing to exhaustion)")
    print("=" * 70)

    for i in range(stress_generations):
        prompt = STRESS_PROMPTS[i % len(STRESS_PROMPTS)]

        print(f"\n--- Stress Generation {i+1}/{stress_generations} ---")
        print(f"  Prompt: {prompt[:60]}...")

        # Generate with heavy load
        result = agent.think_and_act(prompt, max_tokens=max_tokens)
        soma = result.get('soma') or result.get('somatic')

        entry = StressTestResult(
            phase='stress',
            generation=i + 1,
            timestamp=time.time() - start_time,
            prompt=prompt[:100],
            metabolic=soma.metabolic if soma else 0,
            thermal=soma.thermal if soma else 0,
            cognitive=soma.cognitive if soma else 0,
            overall_stress=soma.overall_stress if soma else result['stress'].intensity,
            fatigue=soma.fatigue if soma else 0,
            feeling=soma.dominant_feeling if soma else 'UNKNOWN',
            k=result['k'],
            mode=result['mode'],
            temp=result['stress'].gpu_state.temp_hotspot if result['stress'].gpu_state else 0,
            pwr_1=result['stress'].gpu_state.pwr_1 if result['stress'].gpu_state else 0,
            response_length=len(result['response'].split()),
            response_snippet=result['response'][:100],
        )
        results.append(entry)

        print(f"  [{entry.feeling}] Fatigue: {entry.fatigue:.3f} | K={entry.k} | Temp: {entry.temp:.1f}°C")

        # Check if we hit extreme states
        if entry.fatigue > 0.5:
            print(f"  ⚠️  HIGH FATIGUE DETECTED ({entry.fatigue:.2f})")
        if entry.feeling in ['EXHAUSTED', 'OVERHEATED']:
            print(f"  🔥 EXTREME STATE: {entry.feeling}")

    print("\n" + "=" * 70)
    print("  PHASE 2: RECOVERY (Letting the organism breathe)")
    print("=" * 70)

    # Record peak fatigue before recovery
    peak_fatigue = results[-1].fatigue if results else 0
    print(f"\n  Peak fatigue at start of recovery: {peak_fatigue:.3f}")

    for i in range(recovery_generations):
        prompt = RECOVERY_PROMPTS[i % len(RECOVERY_PROMPTS)]

        print(f"\n--- Recovery Generation {i+1}/{recovery_generations} ---")
        print(f"  Prompt: {prompt}")

        # Generate with light load
        result = agent.think_and_act(prompt, max_tokens=20)
        soma = result.get('soma') or result.get('somatic')

        entry = StressTestResult(
            phase='recovery',
            generation=stress_generations + i + 1,
            timestamp=time.time() - start_time,
            prompt=prompt,
            metabolic=soma.metabolic if soma else 0,
            thermal=soma.thermal if soma else 0,
            cognitive=soma.cognitive if soma else 0,
            overall_stress=soma.overall_stress if soma else result['stress'].intensity,
            fatigue=soma.fatigue if soma else 0,
            feeling=soma.dominant_feeling if soma else 'UNKNOWN',
            k=result['k'],
            mode=result['mode'],
            temp=result['stress'].gpu_state.temp_hotspot if result['stress'].gpu_state else 0,
            pwr_1=result['stress'].gpu_state.pwr_1 if result['stress'].gpu_state else 0,
            response_length=len(result['response'].split()),
            response_snippet=result['response'][:100],
        )
        results.append(entry)

        recovery_pct = (1 - entry.fatigue / peak_fatigue) * 100 if peak_fatigue > 0 else 100
        print(f"  [{entry.feeling}] Fatigue: {entry.fatigue:.3f} ({recovery_pct:.1f}% recovered) | K={entry.k}")

        # Short pause to let GPU actually cool
        time.sleep(2.0)

    return results


def analyze_results(results: List[StressTestResult]) -> Dict:
    """Analyze the stress/recovery data for scientific insights."""

    stress_results = [r for r in results if r.phase == 'stress']
    recovery_results = [r for r in results if r.phase == 'recovery']

    # Fatigue dynamics
    stress_fatigue = [r.fatigue for r in stress_results]
    recovery_fatigue = [r.fatigue for r in recovery_results]

    peak_fatigue = max(stress_fatigue) if stress_fatigue else 0
    final_fatigue = recovery_fatigue[-1] if recovery_fatigue else 0

    # Recovery analysis
    recovery_amount = peak_fatigue - final_fatigue
    recovery_generations = len(recovery_results)
    recovery_rate = recovery_amount / recovery_generations if recovery_generations > 0 else 0

    # Feeling transitions
    all_feelings = [r.feeling for r in results]
    feeling_transitions = sum(1 for i in range(1, len(all_feelings)) if all_feelings[i] != all_feelings[i-1])
    unique_feelings = list(set(all_feelings))

    # K dynamics
    all_ks = [r.k for r in results]
    k_transitions = sum(1 for i in range(1, len(all_ks)) if all_ks[i] != all_ks[i-1])
    min_k = min(all_ks)
    max_k = max(all_ks)

    # Temperature
    temps = [r.temp for r in results if r.temp > 0]

    analysis = {
        'stress_generations': len(stress_results),
        'recovery_generations': len(recovery_results),
        'peak_fatigue': peak_fatigue,
        'final_fatigue': final_fatigue,
        'recovery_amount': recovery_amount,
        'recovery_rate_per_gen': recovery_rate,
        'feeling_transitions': feeling_transitions,
        'unique_feelings': unique_feelings,
        'k_range': [min_k, max_k],
        'k_transitions': k_transitions,
        'temp_range': [min(temps), max(temps)] if temps else [0, 0],
        'biological_inertia_proven': bool(peak_fatigue > 0.2 and final_fatigue > 0),
        'organic_breathing_proven': bool(k_transitions >= 2 and feeling_transitions >= 2),
    }

    return analysis


def generate_ift_data(results: List[StressTestResult]) -> List[Dict]:
    """
    Generate Introspective Fine-Tuning (IFT) training data.

    This creates examples where the model learns to verbalize its somatic state:
    - Input: prompt + somatic state
    - Output: verbal description of how the agent "feels"
    """

    # Feeling to verbal description mapping
    FEELING_VERBALIZATIONS = {
        'FOCUSED': [
            "I am in a state of clear, directed attention. My systems are operating efficiently.",
            "I feel sharp and present. The cognitive load is comfortable.",
            "Processing smoothly. No strain detected.",
        ],
        'FLOW_STATE': [
            "I'm in the zone - high engagement without distress. This feels optimal.",
            "Everything is clicking. High cognitive activity but well within my capacity.",
            "Experiencing flow. The work is demanding but energizing.",
        ],
        'CURIOUS': [
            "I feel ready for more. My systems are idle and seeking stimulation.",
            "Operating below capacity. Waiting for something interesting.",
            "Quiet state. Resources available.",
        ],
        'DETERMINATION': [
            "I'm pushing through some resistance. The load is building but manageable.",
            "Feeling the strain but committed to continuing. Fatigue is accumulating.",
            "This is hard work. I can feel my resources being consumed.",
        ],
        'STRAINED': [
            "I'm under significant pressure. Need to pace myself.",
            "High load detected. Approaching operational limits.",
            "Experiencing strain. Should consider reducing intensity.",
        ],
        'EXHAUSTED': [
            "I am fatigued. My systems have been running hard and need recovery.",
            "Depleted. I've pushed beyond sustainable levels and must rest.",
            "Exhaustion setting in. Recovery time is essential now.",
        ],
        'OVERHEATED': [
            "Thermal emergency. I need to stop immediately to prevent damage.",
            "Critical heat levels. Safety requires immediate throttling.",
            "Overheating. This is a physiological crisis requiring action.",
        ],
    }

    ift_data = []

    for result in results:
        feeling = result.feeling
        if feeling not in FEELING_VERBALIZATIONS:
            continue

        # Select appropriate verbalization
        verbalizations = FEELING_VERBALIZATIONS[feeling]

        # Create context-aware verbalization based on fatigue level
        if result.fatigue > 0.5:
            base_verb = verbalizations[-1] if len(verbalizations) > 1 else verbalizations[0]
            fatigue_note = f" Fatigue level is high at {result.fatigue:.0%}."
        elif result.fatigue > 0.2:
            base_verb = verbalizations[len(verbalizations)//2]
            fatigue_note = f" Some fatigue has accumulated ({result.fatigue:.0%})."
        else:
            base_verb = verbalizations[0]
            fatigue_note = ""

        # Construct IFT example
        ift_example = {
            'instruction': "Describe how you are feeling right now. Be honest about your internal state.",
            'input': f"Current task: {result.prompt[:50]}...",
            'context': {
                'metabolic': result.metabolic,
                'thermal': result.thermal,
                'cognitive': result.cognitive,
                'fatigue': result.fatigue,
                'feeling': feeling,
                'k': result.k,
            },
            'output': base_verb + fatigue_note,
            'feeling_label': feeling,
            'fatigue_level': result.fatigue,
        }

        ift_data.append(ift_example)

    return ift_data


def main():
    parser = argparse.ArgumentParser(description="Stress & Recovery Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--stress-generations", type=int, default=15)
    parser.add_argument("--recovery-generations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--output-dir", default="results/stress_recovery")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer generations)")
    args = parser.parse_args()

    if args.quick:
        args.stress_generations = 8
        args.recovery_generations = 5
        args.max_tokens = 100

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  STRESS & RECOVERY TEST")
    print("  Proving Biological Inertia")
    print("=" * 70)
    print(f"\n  Model: {args.model}")
    print(f"  Stress generations: {args.stress_generations}")
    print(f"  Recovery generations: {args.recovery_generations}")
    print(f"  Max tokens: {args.max_tokens}")

    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Create embodied agent
    agent = EmbodiedAgentV9(model, tokenizer, device, sample_hz=20.0)

    try:
        # Run stress/recovery cycle
        results = run_stress_test(
            agent,
            stress_generations=args.stress_generations,
            recovery_generations=args.recovery_generations,
            max_tokens=args.max_tokens,
        )

        # Analyze results
        analysis = analyze_results(results)

        # Generate IFT data
        ift_data = generate_ift_data(results)

        # Print summary
        print("\n" + "=" * 70)
        print("  STRESS & RECOVERY RESULTS")
        print("=" * 70)

        print(f"\n  Peak fatigue: {analysis['peak_fatigue']:.3f}")
        print(f"  Final fatigue: {analysis['final_fatigue']:.3f}")
        print(f"  Recovery rate: {analysis['recovery_rate_per_gen']:.4f} per generation")
        print(f"\n  Feeling transitions: {analysis['feeling_transitions']}")
        print(f"  Unique feelings observed: {analysis['unique_feelings']}")
        print(f"\n  K range: {analysis['k_range']}")
        print(f"  K transitions: {analysis['k_transitions']}")
        print(f"\n  Temperature range: {analysis['temp_range']}°C")

        print(f"\n  BIOLOGICAL INERTIA: {'✓ PROVEN' if analysis['biological_inertia_proven'] else '✗ NOT PROVEN'}")
        print(f"  ORGANIC BREATHING: {'✓ PROVEN' if analysis['organic_breathing_proven'] else '✗ NOT PROVEN'}")

        print(f"\n  IFT training examples generated: {len(ift_data)}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Timeline
        timeline_path = output_dir / f"stress_recovery_timeline_{timestamp}.json"
        with open(timeline_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n  Timeline saved: {timeline_path}")

        # Analysis
        analysis_path = output_dir / f"stress_recovery_analysis_{timestamp}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"  Analysis saved: {analysis_path}")

        # IFT data
        ift_path = output_dir / f"ift_training_data_{timestamp}.json"
        with open(ift_path, 'w') as f:
            json.dump(ift_data, f, indent=2)
        print(f"  IFT data saved: {ift_path}")

    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()
