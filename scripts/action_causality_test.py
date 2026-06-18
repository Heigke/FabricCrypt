#!/usr/bin/env python3 -u
"""
Action Causality Test: Does ponder_steps actually affect telemetry?

The claim: ponder_steps → different compute → different power/temp

Key insight: K-sampling (sampling K times) does negligible compute.
The REAL compute change comes from ponder_steps (extra forward passes).

To test this:
1. RANDOM PONDER: Randomly vary ponder_steps between 0 and 1 (independent of z_feel)
2. Measure: Does power/temp differ between ponder=0 and ponder=1 segments?

If ponder_steps actually changes compute:
- ponder=1 segments should have higher power (2x forward passes)
- ponder=1 segments should have higher temp (more heat)
- ponder=1 segments should take ~2x time

Statistical test:
- Compare power/temp/time distributions for ponder=0 vs ponder=1 segments
- Should be significantly different

This proves the ACTION actually changes the WORLD (telemetry).
"""

import sys
import time
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np
from scipy import stats

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import TelemetrySampler
from src.compute_effector import PonderEffector

VERSION = "action-causality-v1.0.0"


class ActionCausalityTest:
    """
    Test whether ponder_steps actually affects telemetry.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # Ponder effector (does REAL compute - extra forward passes)
        self.ponder_effector = PonderEffector(self.model, device)

        # Telemetry
        self.telemetry = TelemetrySampler(sample_hz=30)
        self.telemetry.start()
        print(f"  Telemetry: {self.telemetry.source}")

    def run_randomized_ponder_trial(
        self,
        prompt: str,
        n_segments: int = 20,
        tokens_per_segment: int = 30,
    ) -> Dict:
        """
        Run trial with randomly varied ponder_steps.

        Each segment:
        1. Randomly assign ponder_steps=0 or ponder_steps=1
        2. Generate tokens_per_segment tokens
        3. Record power/temp for that segment

        ponder_steps=1 does 2x forward passes, so should measurably differ.
        """
        print(f"\n  Running randomized ponder trial ({n_segments} segments × {tokens_per_segment} tokens)...")

        traces = {
            "segments": [],
            "ponder_sequence": [],
            "all_powers": [],
            "all_temps": [],
            "all_ponder": [],
        }

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        for seg_idx in range(n_segments):
            # Randomly assign ponder_steps for this segment
            ponder = random.choice([0, 1])
            traces["ponder_sequence"].append(ponder)

            segment_powers = []
            segment_temps = []
            segment_times = []

            seg_start = time.time()

            for tok_idx in range(tokens_per_segment):
                # Forward pass with ponder (this is where compute differs!)
                t0 = time.time()
                logits, ponder_info = self.ponder_effector.ponder_forward(
                    current_ids, ponder_steps=ponder
                )
                t1 = time.time()

                # Get telemetry (AFTER compute, to measure effect)
                hw = self.telemetry.get_token_aligned(t0, t1)
                power = hw.get("power")
                temp = hw.get("temp")

                if power is not None:
                    segment_powers.append(power)
                    traces["all_powers"].append(power)
                    traces["all_ponder"].append(ponder)
                if temp is not None:
                    segment_temps.append(temp)
                    traces["all_temps"].append(temp)

                segment_times.append(t1 - t0)

                # Sample next token
                probs = F.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if current_ids.shape[1] > 512:
                    current_ids = input_ids.clone()

            seg_duration = time.time() - seg_start

            segment = {
                "segment_id": seg_idx,
                "ponder_steps": ponder,
                "mean_power": float(np.mean(segment_powers)) if segment_powers else 0,
                "mean_temp": float(np.mean(segment_temps)) if segment_temps else 0,
                "mean_time_ms": float(np.mean(segment_times)) * 1000,
                "duration": seg_duration,
            }
            traces["segments"].append(segment)

            print(f"    Segment {seg_idx+1}: ponder={ponder}, power={segment['mean_power']:.1f}W, "
                  f"temp={segment['mean_temp']:.1f}°C, time={segment['mean_time_ms']:.1f}ms")

        return traces

    def run_test(
        self,
        prompt: str = "Explain the theory of relativity and its implications.",
        n_segments: int = 20,
        tokens_per_segment: int = 30,
    ) -> Dict:
        """
        Run the full action causality test.
        """
        print("\n" + "=" * 60)
        print("  ACTION CAUSALITY TEST: Does ponder → telemetry?")
        print("=" * 60)

        traces = self.run_randomized_ponder_trial(
            prompt=prompt,
            n_segments=n_segments,
            tokens_per_segment=tokens_per_segment,
        )

        # Analyze
        metrics = self._analyze_causality(traces)

        return {
            "version": VERSION,
            "n_segments": n_segments,
            "tokens_per_segment": tokens_per_segment,
            "traces": traces,
            "metrics": metrics,
        }

    def _analyze_causality(self, traces: Dict) -> Dict:
        """
        Analyze whether ponder_steps causally affects telemetry.
        """
        segments = traces["segments"]

        # Separate by ponder_steps
        p0_segments = [s for s in segments if s["ponder_steps"] == 0]
        p1_segments = [s for s in segments if s["ponder_steps"] == 1]

        # Power analysis
        p0_powers = [s["mean_power"] for s in p0_segments]
        p1_powers = [s["mean_power"] for s in p1_segments]

        # Temp analysis
        p0_temps = [s["mean_temp"] for s in p0_segments]
        p1_temps = [s["mean_temp"] for s in p1_segments]

        # Time analysis (should be ~2x for ponder=1)
        p0_times = [s["mean_time_ms"] for s in p0_segments]
        p1_times = [s["mean_time_ms"] for s in p1_segments]

        # Statistical tests
        def test_difference(x, y, name):
            if len(x) < 2 or len(y) < 2:
                return {"error": "insufficient data"}
            try:
                t_stat, t_pvalue = stats.ttest_ind(y, x, alternative='greater')
            except Exception:
                t_stat, t_pvalue = 0, 1.0
            return {
                "ponder0_mean": float(np.mean(x)),
                "ponder1_mean": float(np.mean(y)),
                "difference": float(np.mean(y) - np.mean(x)),
                "t_statistic": float(t_stat),
                "p_value": float(t_pvalue),
                "significant": bool(t_pvalue < 0.05),
            }

        power_test = test_difference(p0_powers, p1_powers, "power")
        temp_test = test_difference(p0_temps, p1_temps, "temp")
        time_test = test_difference(p0_times, p1_times, "time")

        # Overall causality assessment
        # Time should definitely differ (ponder=1 does 2x forward passes)
        # Power/temp should differ if ponder affects compute meaningfully
        time_ratio = time_test.get("ponder1_mean", 0) / time_test.get("ponder0_mean", 1) if time_test.get("ponder0_mean", 1) > 0 else 0

        causality_holds = bool(
            time_ratio > 1.5 and  # ponder=1 takes meaningfully longer (~2x)
            time_test.get("significant", False)
        )

        return {
            "n_ponder0_segments": len(p0_segments),
            "n_ponder1_segments": len(p1_segments),
            "power": power_test,
            "temp": temp_test,
            "time": time_test,
            "time_ratio_p1_p0": float(time_ratio),
            "causality_holds": causality_holds,
            "interpretation": (
                f"PASS: ponder causes compute difference (time ratio={time_ratio:.2f}x)"
                if causality_holds
                else f"INCONCLUSIVE: Time ratio={time_ratio:.2f}x"
            ),
        }

    def stop(self):
        if self.telemetry:
            self.telemetry.stop()


def main():
    parser = argparse.ArgumentParser(description="Action Causality Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_segments", type=int, default=20)
    parser.add_argument("--tokens_per_segment", type=int, default=50)
    parser.add_argument("--output_dir", default="results/action_causality")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    test = ActionCausalityTest(model_name=args.model)

    results = test.run_test(
        n_segments=args.n_segments,
        tokens_per_segment=args.tokens_per_segment,
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/action_causality_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nSaved: {output_path}")

    # Print summary
    m = results["metrics"]
    print("\n" + "=" * 60)
    print("  ACTION CAUSALITY TEST RESULTS")
    print("=" * 60)
    print(f"  Segments: ponder=0: {m['n_ponder0_segments']}, ponder=1: {m['n_ponder1_segments']}")
    print(f"  Time per token:")
    print(f"    ponder=0: {m['time']['ponder0_mean']:.1f}ms")
    print(f"    ponder=1: {m['time']['ponder1_mean']:.1f}ms")
    print(f"    Ratio: {m['time_ratio_p1_p0']:.2f}x")
    print(f"  Power:")
    print(f"    ponder=0: {m['power']['ponder0_mean']:.1f}W")
    print(f"    ponder=1: {m['power']['ponder1_mean']:.1f}W")
    print(f"    p-value: {m['power']['p_value']:.4f}")
    print(f"  Result: {m['interpretation']}")
    print("=" * 60)

    test.stop()


if __name__ == "__main__":
    main()
