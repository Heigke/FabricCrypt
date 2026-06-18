#!/usr/bin/env python3
"""
FEEL v10 Proper Falsification Test

This version uses TelemetrySamplerWrapper to ensure falsification
actually reaches the model inputs through the normal pipeline.

Previous scripts tried to poke internal attributes - this doesn't work.
The wrapper intercepts get_token_aligned() which is the correct entry point.

Conditions:
- live: real telemetry (should show benefit if FEEL works)
- shuffle: randomized sensors (should collapse benefit if causal)
- lag: delayed sensors (should collapse benefit if real-time matters)
- zero: null sensors (ablation)

If FEEL benefit is CAUSAL, it MUST collapse under shuffle/lag.
"""

import sys
import time
import json
import random
from pathlib import Path
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELStreamV3,
    FEELProjectorFull,
    TelemetrySampler,
    TelemetrySamplerWrapper,
)

VERSION = "v10.1.0-proper-falsification"

# Test prompts (harder ones where FEEL might help)
PROMPTS = [
    {"prompt": "The derivative of x^3 + 2x is", "answer": "3x^2"},
    {"prompt": "If 2x + 5 = 15, then x =", "answer": "5"},
    {"prompt": "The integral of 2x dx is", "answer": "x^2"},
    {"prompt": "What is 17 * 23?", "answer": "391"},
    {"prompt": "What is 15% of 80?", "answer": "12"},
    {"prompt": "The square root of 144 is", "answer": "12"},
    {"prompt": "What is log base 2 of 8?", "answer": "3"},
    {"prompt": "The sum of angles in a triangle is", "answer": "180"},
    {"prompt": "What is the time complexity of merge sort?", "answer": "log"},
    {"prompt": "What data structure is used in BFS?", "answer": "queue"},
    {"prompt": "What is the HTTP status code for Not Found?", "answer": "404"},
    {"prompt": "What keyword is used for inheritance in Python?", "answer": "class"},
]


def bootstrap_ci(data, n=500):
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    data = np.array(data, dtype=float)
    point = np.mean(data)
    boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n)]
    return (point, np.percentile(boots, 2.5), np.percentile(boots, 97.5))


def main():
    print("=" * 70)
    print(f"  FEEL {VERSION}")
    print("  PROPER FALSIFICATION via TelemetrySamplerWrapper")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    embed_dim = model.config.hidden_size

    # Load v10 checkpoint
    projector = FEELProjectorFull(embed_dim=embed_dim).to(device)
    ckpt_path = "results/feel_training_v10/final_checkpoint.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        projector.load_state_dict(ckpt['projector_state_dict'])
        print(f"  Loaded checkpoint: {ckpt.get('version', 'unknown')}")
    else:
        print(f"  WARNING: No checkpoint at {ckpt_path}")

    # Base telemetry sampler
    base_sampler = None
    try:
        base_sampler = TelemetrySampler(sample_hz=30)
        base_sampler.start()
        print(f"  Telemetry: {base_sampler.source}")
    except Exception as e:
        print(f"  Telemetry unavailable: {e}")

    # Conditions to test
    # - "baseline": no FEEL at all
    # - "live": FEEL with real telemetry
    # - "shuffle": FEEL with shuffled telemetry
    # - "lag": FEEL with lagged telemetry
    # - "zero": FEEL with null telemetry
    conditions = ["baseline", "live", "shuffle", "lag", "zero"]

    def generate_with_condition(prompt: str, condition: str, temperature: float = 0.7, n_samples: int = 5):
        """Generate multiple samples under specified condition."""
        results = []

        for sample_idx in range(n_samples):
            # Create fresh FEEL stream
            feel_stream = FEELStreamV3(projector=projector, alpha=0.15, mode="full")

            # Configure telemetry based on condition
            if condition == "baseline":
                # No FEEL at all
                use_feel = False
                wrapper = None
            else:
                use_feel = True
                # Create wrapper with appropriate mode
                wrapper = TelemetrySamplerWrapper(
                    base_sampler,
                    mode=condition if condition != "live" else "live",
                    lag_steps=5,
                )
                feel_stream.set_telemetry_sampler(wrapper)

            # Generation
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            current_ids = input_ids.clone()

            # Track telemetry values seen
            telemetry_trace = []

            for step in range(15):
                embeds = model.get_input_embeddings()(current_ids)

                if use_feel:
                    embeds = feel_stream.apply_to_embeds(embeds, alpha_override=0.15)

                with torch.no_grad():
                    t0 = time.time()
                    out = model(inputs_embeds=embeds, use_cache=False)
                    logits = out.logits[:, -1, :].float()
                    t1 = time.time()

                if use_feel:
                    state = feel_stream.step(logits, t0, t1, step, current_ids.shape[1])
                    # Record what telemetry was actually used
                    telemetry_trace.append({
                        "step": step,
                        "temp": state.hardware.temp,
                        "power": state.hardware.power,
                    })

                # Temperature sampling
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            text = tokenizer.decode(current_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            results.append({
                "text": text.strip().lower(),
                "telemetry": telemetry_trace,
            })

        return results

    def check_answer(output: str, answer: str) -> bool:
        return str(answer).lower() in output.lower()

    # Run all conditions
    all_results = {c: {"correct": [], "total": 0} for c in conditions}

    print(f"\nRunning {len(PROMPTS)} prompts x 5 samples x {len(conditions)} conditions...")
    print("-" * 70)

    for i, p in enumerate(PROMPTS):
        prompt = p["prompt"]
        answer = p["answer"]

        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt[:45]}...")

        for condition in conditions:
            samples = generate_with_condition(prompt + " ", condition, temperature=0.7, n_samples=5)
            correct = sum(check_answer(s["text"], answer) for s in samples)
            all_results[condition]["correct"].extend([1]*correct + [0]*(5-correct))
            all_results[condition]["total"] += 5

            # Show telemetry variation for first prompt to verify wrapper works
            if i == 0 and condition != "baseline" and samples[0]["telemetry"]:
                temps = [t["temp"] for t in samples[0]["telemetry"][:5] if t["temp"]]
                print(f"    {condition}: {correct}/5 | temps={temps[:3]}...")
            else:
                print(f"    {condition}: {correct}/5", end="")
        print()

    # Summary with bootstrap CIs
    print("\n" + "=" * 70)
    print("  FALSIFICATION RESULTS")
    print("=" * 70)

    accuracies = {}
    for condition in conditions:
        data = all_results[condition]["correct"]
        point, ci_lo, ci_hi = bootstrap_ci(data)
        accuracies[condition] = point
        total = all_results[condition]["total"]
        correct = sum(data)
        print(f"\n  {condition.upper():10s}: {point:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] ({correct}/{total})")

    # Benefit analysis
    baseline_acc = accuracies["baseline"]
    live_acc = accuracies["live"]
    shuffle_acc = accuracies["shuffle"]
    lag_acc = accuracies["lag"]
    zero_acc = accuracies["zero"]

    print(f"\n  BENEFIT vs BASELINE:")
    print(f"  ---------------------")
    print(f"  Live (real sensors):    {live_acc - baseline_acc:+.3f}")
    print(f"  Shuffle (randomized):   {shuffle_acc - baseline_acc:+.3f}")
    print(f"  Lag (delayed):          {lag_acc - baseline_acc:+.3f}")
    print(f"  Zero (null):            {zero_acc - baseline_acc:+.3f}")

    # Causal loop test
    live_benefit = live_acc - baseline_acc

    print(f"\n  CAUSAL LOOP TEST:")
    print(f"  -----------------")

    if live_benefit > 0.03:  # Meaningful benefit threshold
        shuffle_collapse = (shuffle_acc - baseline_acc) < live_benefit * 0.5
        lag_collapse = (lag_acc - baseline_acc) < live_benefit * 0.5
        zero_collapse = (zero_acc - baseline_acc) < live_benefit * 0.5

        print(f"  Live FEEL benefit: {live_benefit:+.3f}")
        print(f"  Shuffle collapses benefit: {shuffle_collapse} ({shuffle_acc - baseline_acc:+.3f})")
        print(f"  Lag collapses benefit: {lag_collapse} ({lag_acc - baseline_acc:+.3f})")
        print(f"  Zero collapses benefit: {zero_collapse} ({zero_acc - baseline_acc:+.3f})")

        if shuffle_collapse and lag_collapse:
            print(f"\n  === CAUSAL LOOP: STRONGLY SUPPORTED ===")
            print(f"  Benefit requires real-time sensor feedback.")
        elif shuffle_collapse or lag_collapse:
            print(f"\n  === CAUSAL LOOP: PARTIALLY SUPPORTED ===")
        else:
            print(f"\n  === CAUSAL LOOP: NOT SUPPORTED ===")
            print(f"  Benefit persists under falsification - may be spurious.")
    else:
        print(f"  Live FEEL benefit too small ({live_benefit:+.3f}) for causal test.")
        print(f"  Need benefit > 0.03 to distinguish from noise.")

    # Save results
    output = {
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(PROMPTS),
        "samples_per_prompt": 5,
        "temperature": 0.7,
        "conditions": conditions,
        "accuracies": {k: float(v) for k, v in accuracies.items()},
        "benefits": {
            "live": float(live_acc - baseline_acc),
            "shuffle": float(shuffle_acc - baseline_acc),
            "lag": float(lag_acc - baseline_acc),
            "zero": float(zero_acc - baseline_acc),
        },
        "causal_test": {
            "live_benefit": float(live_benefit),
            "threshold": 0.03,
            "sufficient_benefit": True if live_benefit > 0.03 else False,
        },
        "raw_results": {k: [int(x) for x in v["correct"]] for k, v in all_results.items()},
    }

    Path("results/feel_experiments").mkdir(parents=True, exist_ok=True)
    outpath = "results/feel_experiments/v10_proper_falsification.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {outpath}")

    if base_sampler:
        base_sampler.stop()


if __name__ == "__main__":
    main()
