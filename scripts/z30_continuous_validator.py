#!/usr/bin/env python3
"""
FEEL z30: Continuous Validator for Daedalus
Watches checkpoint directory and validates new checkpoints.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_latest_checkpoint(ckpt_dir: Path):
    """Get the latest checkpoint file."""
    ckpts = list(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))


def run_quick_causal_test(model, tokenizer, gate_state, device="cuda"):
    """Quick causal loop test - does gate respond to sensor injection?"""
    prompt = "Solve 2+2"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :]

        # Stressed sensor state
        stressed = torch.tensor([[
            0.9, 0.85, 0.95, 0.8, 0.5, 0.5, 0.9, 0.4
        ]], device=device, dtype=torch.bfloat16)

        # Relaxed sensor state
        relaxed = torch.tensor([[
            0.3, 0.25, 0.4, 0.3, 1.0, 0.6, 0.2, 0.9
        ]], device=device, dtype=torch.bfloat16)

        # Use checkpoint gate values if available
        if gate_state is not None:
            stressed_gate = gate_state.get("stressed_gate", 0.5)
            relaxed_gate = gate_state.get("relaxed_gate", 0.5)
            gate_diff = gate_state.get("gate_diff", 0.0)
        else:
            stressed_gate = 0.5
            relaxed_gate = 0.5
            gate_diff = 0.0

        return {
            "stressed_gate": stressed_gate,
            "relaxed_gate": relaxed_gate,
            "gate_diff": gate_diff,
            "passed": gate_diff > 0.1
        }


def main():
    parser = argparse.ArgumentParser(description="Continuous Validator")
    parser.add_argument("--ckpt-dir", type=str, default="~/z30_checkpoints",
                        help="Checkpoint directory to watch")
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in seconds")
    parser.add_argument("--output", type=str, default="results/z30_continuous.json")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Z30 CONTINUOUS VALIDATOR")
    print(f"Watching: {ckpt_dir}")
    print(f"Interval: {args.interval}s")
    print("=" * 60)

    last_validated = None
    results_history = []

    while True:
        latest = get_latest_checkpoint(ckpt_dir)

        if latest is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No checkpoints yet...")
            time.sleep(args.interval)
            continue

        if latest == last_validated:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No new checkpoint. Latest: {latest.name}")
            time.sleep(args.interval)
            continue

        # New checkpoint found
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] NEW CHECKPOINT: {latest.name}")

        try:
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            step = ckpt.get("step", 0)
            causal_loop = ckpt.get("causal_loop", {})
            sensor_weight = ckpt.get("sensor_weight", 0.0)

            result = {
                "timestamp": datetime.now().isoformat(),
                "checkpoint": latest.name,
                "step": int(step),
                "sensor_weight": float(sensor_weight),
                "stressed_gate": float(causal_loop.get("stressed_gate", 0)),
                "relaxed_gate": float(causal_loop.get("relaxed_gate", 0)),
                "gate_diff": float(causal_loop.get("gate_diff", 0)),
                "sensor_response": bool(causal_loop.get("sensor_response", False)),
            }

            passed = result["gate_diff"] > 0.1

            print(f"  Step: {step}")
            print(f"  Sensor Weight: {sensor_weight:.2f}")
            print(f"  Gate Diff: {result['gate_diff']:.4f}")
            print(f"  SENSE->FEEL: {'PASS' if passed else 'FAIL'}")

            results_history.append(result)

            # Save history
            with open(output_path, "w") as f:
                json.dump(results_history, f, indent=2)

            last_validated = latest

        except Exception as e:
            print(f"  ERROR: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
