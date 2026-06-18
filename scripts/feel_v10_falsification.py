#!/usr/bin/env python3
"""
FEEL v10 Benefit Collapse Falsification Test

If FEEL benefit is CAUSAL (real closed loop), it should collapse under:
- Shuffle: randomize sensor readings
- Lag: use delayed sensor readings
- Replay: use recordings from different prompt

If benefit persists under these conditions, it's NOT causal.
"""

import sys
import time
import json
import random
from pathlib import Path
from collections import deque
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import FEELStreamV3, FEELProjectorFull, TelemetrySampler
from src.canonical_sensors import HardwareContext

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


class SensorManipulator:
    """Manipulate sensors for falsification."""

    def __init__(self):
        self.history = deque(maxlen=100)
        self.shuffle_pool = []

    def record(self, hw: HardwareContext):
        self.history.append(hw)
        self.shuffle_pool.append(hw)

    def get_shuffled(self) -> HardwareContext:
        """Return random sensor from pool (breaks temporal structure)."""
        if self.shuffle_pool:
            return random.choice(self.shuffle_pool)
        return HardwareContext()

    def get_lagged(self, lag=5) -> HardwareContext:
        """Return lagged sensor (breaks real-time feedback)."""
        if len(self.history) > lag:
            return self.history[-lag]
        return self.history[0] if self.history else HardwareContext()

    def reset(self):
        self.shuffle_pool = []


def main():
    print("=" * 60)
    print("  FEEL v10 Falsification Test - Benefit Collapse")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    np.random.seed(42)

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

    projector = FEELProjectorFull(embed_dim=embed_dim).to(device)
    ckpt_path = "results/feel_training_v10/final_checkpoint.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        projector.load_state_dict(ckpt['projector_state_dict'])
        print(f"  Loaded checkpoint: {ckpt.get('version', 'unknown')}")

    feel_stream = FEELStreamV3(projector=projector, alpha=0.15, mode="full")

    sampler = None
    try:
        sampler = TelemetrySampler(sample_hz=30)
        sampler.start()
        feel_stream.set_telemetry_sampler(sampler)
        print(f"  Telemetry: {sampler.source}")
    except Exception as e:
        print(f"  Telemetry unavailable: {e}")

    manipulator = SensorManipulator()

    def generate_with_condition(prompt, condition, temperature=0.7, n_samples=5):
        """Generate with specified falsification condition."""
        results = []

        for _ in range(n_samples):
            feel_stream.reset()
            manipulator.reset()

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            current_ids = input_ids.clone()

            for step in range(15):
                embeds = model.get_input_embeddings()(current_ids)

                # Apply FEEL based on condition
                if condition != "baseline":
                    embeds = feel_stream.apply_to_embeds(embeds, alpha_override=0.15)

                with torch.no_grad():
                    t0 = time.time()
                    out = model(inputs_embeds=embeds, use_cache=False)
                    logits = out.logits[:, -1, :].float()
                    t1 = time.time()

                # Step FEEL with manipulation
                if condition != "baseline":
                    state = feel_stream.step(logits, t0, t1, step, current_ids.shape[1])
                    real_hw = state.hardware
                    manipulator.record(real_hw)

                    # Apply falsification manipulation for NEXT iteration
                    if condition == "shuffle":
                        # Inject shuffled sensors into next step
                        if hasattr(feel_stream, '_sensor_bank'):
                            fake_hw = manipulator.get_shuffled()
                            feel_stream._sensor_bank._last_hw = fake_hw
                    elif condition == "lag":
                        if hasattr(feel_stream, '_sensor_bank'):
                            fake_hw = manipulator.get_lagged(lag=5)
                            feel_stream._sensor_bank._last_hw = fake_hw

                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            text = tokenizer.decode(current_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            results.append(text.strip().lower())

        return results

    def check_answer(output, answer):
        return str(answer).lower() in output.lower()

    # Test conditions
    conditions = ["baseline", "feel", "shuffle", "lag"]
    results = {c: {"correct": 0, "total": 0} for c in conditions}

    print(f"\nRunning {len(PROMPTS)} prompts x 5 samples x {len(conditions)} conditions...")
    print("-" * 60)

    for i, p in enumerate(PROMPTS):
        prompt = p["prompt"]
        answer = p["answer"]

        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt[:40]}...")

        for condition in conditions:
            samples = generate_with_condition(prompt + " ", condition, temperature=0.7, n_samples=5)
            correct = sum(check_answer(s, answer) for s in samples)
            results[condition]["correct"] += correct
            results[condition]["total"] += len(samples)
            print(f"  {condition}: {correct}/5", end="")
        print()

    # Summary
    print("\n" + "=" * 60)
    print("  FALSIFICATION RESULTS")
    print("=" * 60)

    accuracies = {}
    for condition in conditions:
        acc = results[condition]["correct"] / results[condition]["total"]
        accuracies[condition] = acc
        print(f"\n  {condition.upper():10s}: {acc:.3f} ({results[condition]['correct']}/{results[condition]['total']})")

    baseline_acc = accuracies["baseline"]
    feel_acc = accuracies["feel"]
    shuffle_acc = accuracies["shuffle"]
    lag_acc = accuracies["lag"]

    print(f"\n  BENEFIT ANALYSIS:")
    print(f"  -----------------")
    print(f"  FEEL benefit:    {feel_acc - baseline_acc:+.3f}")
    print(f"  Shuffle benefit: {shuffle_acc - baseline_acc:+.3f}")
    print(f"  Lag benefit:     {lag_acc - baseline_acc:+.3f}")

    # Collapse test
    feel_benefit = feel_acc - baseline_acc

    if feel_benefit > 0.02:  # Need meaningful FEEL benefit
        shuffle_collapse = (shuffle_acc - baseline_acc) < feel_benefit * 0.5
        lag_collapse = (lag_acc - baseline_acc) < feel_benefit * 0.5

        print(f"\n  CAUSAL LOOP TEST (FEEL benefit = {feel_benefit:+.3f}):")
        print(f"  -----------------")
        print(f"  Shuffle collapses benefit: {shuffle_collapse}")
        print(f"  Lag collapses benefit: {lag_collapse}")

        if shuffle_collapse and lag_collapse:
            print(f"\n  CAUSAL LOOP EVIDENCE: STRONGLY SUPPORTED")
            print(f"  The benefit is specific to real-time sensor feedback.")
        elif shuffle_collapse or lag_collapse:
            print(f"\n  CAUSAL LOOP EVIDENCE: PARTIALLY SUPPORTED")
        else:
            print(f"\n  CAUSAL LOOP EVIDENCE: NOT SUPPORTED")
            print(f"  Benefit persists under manipulation - may be spurious.")
    else:
        print(f"\n  FEEL benefit too small ({feel_benefit:+.3f}) for causal test")

    # Save
    output = {
        "version": "v10.0.0-falsification",
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(PROMPTS),
        "samples_per_prompt": 5,
        "temperature": 0.7,
        "accuracies": {k: float(v) for k, v in accuracies.items()},
        "benefits": {
            "feel": float(feel_acc - baseline_acc),
            "shuffle": float(shuffle_acc - baseline_acc),
            "lag": float(lag_acc - baseline_acc),
        },
        "causal_test": {
            "feel_benefit": float(feel_benefit),
            "shuffle_collapse": shuffle_collapse if feel_benefit > 0.02 else None,
            "lag_collapse": lag_collapse if feel_benefit > 0.02 else None,
        }
    }

    Path("results/feel_experiments").mkdir(parents=True, exist_ok=True)
    with open("results/feel_experiments/v10_falsification.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: results/feel_experiments/v10_falsification.json")

    if sampler:
        sampler.stop()


if __name__ == "__main__":
    main()
