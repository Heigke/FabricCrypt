#!/usr/bin/env python3
"""
FEEL v10 Sampling Test - Tests distribution shift under temperature sampling.

With greedy (argmax), small KL shifts don't change the top token.
With sampling, distribution shifts affect which token gets selected.

This test runs multiple samples per prompt to measure consistency improvement.
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

from src import FEELStreamV3, FEELProjectorFull, TelemetrySampler

# Harder prompts that benefit from better distribution
PROMPTS = [
    {"prompt": "The derivative of x^3 + 2x is", "answer": "3x^2", "type": "math"},
    {"prompt": "If 2x + 5 = 15, then x =", "answer": "5", "type": "math"},
    {"prompt": "The integral of 2x dx is", "answer": "x^2", "type": "math"},
    {"prompt": "What is 17 * 23?", "answer": "391", "type": "math"},
    {"prompt": "What is 15% of 80?", "answer": "12", "type": "math"},
    {"prompt": "The square root of 144 is", "answer": "12", "type": "math"},
    {"prompt": "What is log base 2 of 8?", "answer": "3", "type": "math"},
    {"prompt": "The sum of angles in a triangle is", "answer": "180", "type": "math"},
    {"prompt": "What is the time complexity of merge sort?", "answer": "log", "type": "cs"},
    {"prompt": "What data structure is used in BFS?", "answer": "queue", "type": "cs"},
    {"prompt": "What is the HTTP status code for Not Found?", "answer": "404", "type": "cs"},
    {"prompt": "What keyword is used for inheritance in Python?", "answer": "class", "type": "cs"},
]


def main():
    print("=" * 60)
    print("  FEEL v10 Sampling Test - Distribution Shift Analysis")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    np.random.seed(42)

    # Load model
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

    feel_stream = FEELStreamV3(projector=projector, alpha=0.15, mode="full")

    try:
        sampler = TelemetrySampler(sample_hz=30)
        sampler.start()
        feel_stream.set_telemetry_sampler(sampler)
        print(f"  Telemetry: {sampler.source}")
    except:
        sampler = None

    def generate_with_sampling(prompt, use_feel, temperature=0.7, n_samples=5):
        """Generate multiple samples to measure consistency."""
        outputs = []

        for _ in range(n_samples):
            feel_stream.reset()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            current_ids = input_ids.clone()

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
                    feel_stream.step(logits, t0, t1, step, current_ids.shape[1])

                # Temperature sampling instead of greedy
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

            text = tokenizer.decode(current_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            outputs.append(text.strip().lower())

        return outputs

    def check_answer(output, answer):
        return str(answer).lower() in output.lower()

    # Results
    results = {
        "baseline": {"correct": 0, "total": 0, "consistency": []},
        "feel": {"correct": 0, "total": 0, "consistency": []},
    }

    print(f"\nRunning {len(PROMPTS)} prompts x 5 samples each...")
    print("-" * 60)

    for i, p in enumerate(PROMPTS):
        prompt = p["prompt"]
        answer = p["answer"]

        print(f"\n[{i+1}/{len(PROMPTS)}] {prompt[:40]}...")

        for condition in ["baseline", "feel"]:
            use_feel = (condition == "feel")
            samples = generate_with_sampling(prompt + " ", use_feel, temperature=0.7, n_samples=5)

            # Check correctness
            correct_count = sum(check_answer(s, answer) for s in samples)
            accuracy = correct_count / len(samples)
            results[condition]["total"] += len(samples)
            results[condition]["correct"] += correct_count

            # Measure consistency (how many samples agree?)
            from collections import Counter
            tokens = [s.split()[0] if s.split() else "" for s in samples]
            most_common_count = Counter(tokens).most_common(1)[0][1] if tokens else 0
            consistency = most_common_count / len(samples)
            results[condition]["consistency"].append(consistency)

            print(f"  {condition}: {correct_count}/5 correct, consistency={consistency:.2f}")
            if correct_count > 0:
                print(f"    Sample: {samples[0][:50]}...")

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    for condition in ["baseline", "feel"]:
        total = results[condition]["total"]
        correct = results[condition]["correct"]
        accuracy = correct / total if total > 0 else 0
        avg_consistency = np.mean(results[condition]["consistency"])

        print(f"\n  {condition.upper()}:")
        print(f"    Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"    Avg Consistency: {avg_consistency:.3f}")

    baseline_acc = results["baseline"]["correct"] / results["baseline"]["total"]
    feel_acc = results["feel"]["correct"] / results["feel"]["total"]
    baseline_cons = np.mean(results["baseline"]["consistency"])
    feel_cons = np.mean(results["feel"]["consistency"])

    print(f"\n  BENEFIT:")
    print(f"    Accuracy: {feel_acc - baseline_acc:+.3f}")
    print(f"    Consistency: {feel_cons - baseline_cons:+.3f}")

    # Save
    output = {
        "version": "v10.0.0-sampling",
        "timestamp": datetime.now().isoformat(),
        "n_prompts": len(PROMPTS),
        "samples_per_prompt": 5,
        "temperature": 0.7,
        "results": {
            "baseline": {
                "accuracy": float(baseline_acc),
                "consistency": float(baseline_cons),
            },
            "feel": {
                "accuracy": float(feel_acc),
                "consistency": float(feel_cons),
            },
        },
        "benefit": {
            "accuracy": float(feel_acc - baseline_acc),
            "consistency": float(feel_cons - baseline_cons),
        }
    }

    Path("results/feel_experiments").mkdir(parents=True, exist_ok=True)
    with open("results/feel_experiments/v10_sampling_test.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: results/feel_experiments/v10_sampling_test.json")

    if sampler:
        sampler.stop()


if __name__ == "__main__":
    main()
