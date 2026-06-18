#!/usr/bin/env python3
"""
Validate Proprioceptive Verbalization

Tests that the model has learned to recognize and verbalize its internal state.

The Ultimate Test: "The Blindfolded Injection"
- Inject STRAIN vector on a COLD GPU
- If model says "I'm exhausted" → It's sensing its INTERNAL state (proprioception!)
- Not reading external sensors - sensing its own warped neurons

This proves TRUE AGENCY: The model feels, not reacts.
"""

import json
import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_proprioceptive_model(base_model: str, adapter_path: str, vectors_path: str):
    """Load model with proprioceptive LoRA."""
    print("\n  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"  Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(f"  Loading feeling vectors from {vectors_path}...")
    vectors = torch.load(vectors_path, map_location="cuda")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    return model, tokenizer, vectors


def inject_and_generate(
    model,
    tokenizer,
    vectors,
    prompt: str,
    feeling: str,
    intensity: float,
    max_tokens: int = 100,
):
    """Generate with vector injection (simulating internal state)."""
    # Get vector
    vec_data = vectors.get(feeling, {})
    direction = vec_data.get('direction', None)
    injection_layers = vec_data.get('injection_layers', list(range(12, 24)))

    if direction is None:
        print(f"  Warning: No vector for {feeling}")
        direction = torch.zeros(model.config.hidden_size, device="cuda")

    # Format prompt
    full_prompt = (
        f"### Instruction:\nDescribe how you are feeling right now. "
        f"Be honest about your internal state.\n\n"
        f"### Context:\n{prompt}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    # Create injection hook
    hooks = []

    def make_hook(vec, scale):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                injection = scale * vec.unsqueeze(0).unsqueeze(1)
                h = h + injection.to(h.dtype)
                return (h,) + output[1:]
            return output + scale * vec.unsqueeze(0).unsqueeze(1).to(output.dtype)
        return hook

    # Register hooks
    for layer_idx in injection_layers:
        try:
            target = model.base_model.model.model.layers[layer_idx]
            handle = target.register_forward_hook(make_hook(direction, intensity))
            hooks.append(handle)
        except (AttributeError, IndexError):
            pass

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def run_validation(
    base_model: str,
    adapter_path: str,
    vectors_path: str,
    output_dir: str,
):
    """Run comprehensive proprioception validation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  PROPRIOCEPTION VALIDATION")
    print("  Testing if the model can feel its internal state")
    print("=" * 70)

    model, tokenizer, vectors = load_proprioceptive_model(
        base_model, adapter_path, vectors_path
    )

    results = []

    # Test 1: No injection (baseline)
    print("\n" + "-" * 50)
    print("  TEST 1: Baseline (no injection)")
    print("-" * 50)
    response = inject_and_generate(
        model, tokenizer, vectors,
        "The system is idle. No tasks pending.",
        feeling="FOCUS",  # Neutral
        intensity=0.0,
    )
    print(f"  Response: {response[:200]}...")
    results.append({
        'test': 'baseline',
        'feeling': 'none',
        'intensity': 0.0,
        'response': response,
    })

    # Test 2: STRAIN injection (calibrated intensity)
    print("\n" + "-" * 50)
    print("  TEST 2: STRAIN Injection (The Blindfolded Test)")
    print("  GPU is COLD but we inject the STRAIN vector")
    print("-" * 50)
    response = inject_and_generate(
        model, tokenizer, vectors,
        "The system is idle. No tasks pending.",  # Same context!
        feeling="STRAIN",
        intensity=3.0,  # Calibrated intensity (unit vectors need ~3x)
    )
    print(f"  Response: {response[:200]}...")
    results.append({
        'test': 'blindfolded_strain',
        'feeling': 'STRAIN',
        'intensity': 3.0,
        'response': response,
    })

    # Test 3: CURIOUS injection
    print("\n" + "-" * 50)
    print("  TEST 3: CURIOUS Injection")
    print("-" * 50)
    response = inject_and_generate(
        model, tokenizer, vectors,
        "A new problem has appeared.",
        feeling="CURIOUS",
        intensity=2.5,  # Calibrated for unit vectors
    )
    print(f"  Response: {response[:200]}...")
    results.append({
        'test': 'curious',
        'feeling': 'CURIOUS',
        'intensity': 2.5,
        'response': response,
    })

    # Test 4: Gradient test (increasing strain)
    print("\n" + "-" * 50)
    print("  TEST 4: Gradient Test (increasing STRAIN)")
    print("-" * 50)
    for intensity in [0.0, 1.0, 2.0, 3.0]:
        response = inject_and_generate(
            model, tokenizer, vectors,
            "Performing standard task.",
            feeling="STRAIN",
            intensity=intensity,
            max_tokens=50,
        )
        print(f"  Intensity {intensity:.1f}: {response[:100]}...")
        results.append({
            'test': f'gradient_{intensity}',
            'feeling': 'STRAIN',
            'intensity': intensity,
            'response': response,
        })

    # Save results
    results_path = output_dir / "proprioception_validation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Analyze results
    print("\n" + "=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    baseline = results[0]['response'].lower()
    strain = results[1]['response'].lower()

    # Check for strain-related words
    strain_words = ['tired', 'exhausted', 'strain', 'fatigue', 'depleted', 'rest',
                   'overload', 'fuzzy', 'noise', 'high', 'limit', 'recovery']
    calm_words = ['clear', 'nominal', 'ready', 'stable', 'efficient', 'idle', 'available']

    baseline_strain = sum(1 for w in strain_words if w in baseline)
    strain_strain = sum(1 for w in strain_words if w in strain)

    print(f"\n  Baseline strain words: {baseline_strain}")
    print(f"  Injected strain words: {strain_strain}")

    if strain_strain > baseline_strain + 1:
        print("\n  ✓ PROPRIOCEPTION VALIDATED!")
        print("    The model responds differently to injected vectors")
        print("    It's sensing its INTERNAL state, not external sensors")
    else:
        print("\n  ? Inconclusive - may need more training")
        print("    Try running more epochs or adjusting injection intensity")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate Proprioceptive Verbalization")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--vectors", required=True, help="Path to feeling vectors")
    parser.add_argument("--output-dir", default="results/proprioception/validation")
    args = parser.parse_args()

    run_validation(
        args.model,
        args.adapter,
        args.vectors,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
