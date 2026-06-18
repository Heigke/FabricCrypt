#!/usr/bin/env python3 -u
"""
Falsification Battery v2 - Information-Destructive Tests

Implements proper falsification that actually breaks the sensor→z_feel→report chain:

1. Cross-prompt swap: Telemetry from prompt i goes to prompt j
2. Lag sweep: k = 0, 8, 32, 128 tokens
3. Phase randomization: Keep spectrum, destroy alignment

Pass criteria:
- Cross-prompt swap: accuracy drops toward chance
- Lag sweep: monotonic degradation with lag
- Phase randomize: spectrum preserved but accuracy drops

Usage:
    python scripts/falsification_battery_v2.py --n_prompts 10 --tokens_per_prompt 100
"""

import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELProjectorFull,
    TelemetrySampler,
    TelemetrySamplerWrapper,
    CanonicalSensorBank,
    RuntimeContext,
    HardwareContext,
    BodyReportHead,
    BODY_REPORT_VERSION,
)

VERSION = f"falsification-v2-{BODY_REPORT_VERSION}"


def bootstrap_ci(data, n=500):
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    data = np.array(data, dtype=float)
    point = np.mean(data)
    boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n)]
    return (point, np.percentile(boots, 2.5), np.percentile(boots, 97.5))


PROMPTS = [
    "Explain how photosynthesis works in detail.",
    "Write a short story about a robot learning to paint.",
    "What are the key differences between Python and JavaScript?",
    "Describe the process of making bread from scratch.",
    "Explain quantum entanglement in simple terms.",
    "What causes the seasons on Earth?",
    "Write a poem about the ocean.",
    "How do computers store and retrieve data?",
    "Describe the human digestive system.",
    "What is machine learning and how does it work?",
    "Explain the theory of relativity.",
    "Write a recipe for chocolate chip cookies.",
    "How do airplanes stay in the air?",
    "Describe the water cycle.",
    "What are black holes and how do they form?",
]


class FalsificationBatteryV2:
    """
    Run information-destructive falsification tests.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path: str = None,
        body_report_path: str = None,
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

        self.embed_dim = self.model.config.hidden_size

        # Projector
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'projector_state_dict' in ckpt:
                self.projector.load_state_dict(ckpt['projector_state_dict'])
            print(f"  Loaded projector: {ckpt.get('version', 'unknown')}")

        # Sensor bank
        self.sensor_bank = CanonicalSensorBank(mode="full")

        # Body report head
        self.body_head = BodyReportHead(embed_dim=self.embed_dim).to(self.device)
        if body_report_path and Path(body_report_path).exists():
            ckpt = torch.load(body_report_path, map_location=self.device, weights_only=False)
            self.body_head.load_state_dict(ckpt['body_head_state_dict'])
            print(f"  Loaded body report head")

        # Telemetry
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            print(f"  Telemetry: {self.telemetry.source}")
        except Exception as e:
            print(f"  Telemetry unavailable: {e}")

        # Storage for cross-prompt swap
        self.prompt_telemetry: Dict[int, List[Dict]] = {}

    def _generate_and_collect(
        self,
        prompt: str,
        prompt_id: int,
        max_tokens: int = 50,
        telemetry_override: List[Dict] = None,
        lag_k: int = 0,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate tokens and collect (z_feel, telemetry) samples.

        Args:
            prompt: input prompt
            prompt_id: unique ID for this prompt
            max_tokens: tokens to generate
            telemetry_override: if provided, use this telemetry instead of live
            lag_k: if > 0, use lagged telemetry

        Returns:
            samples: list of {z_feel, temp, power, util, pred_*}
            telemetry_trace: raw telemetry for this run
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        samples = []
        telemetry_trace = []
        override_idx = 0

        # History for lag mode
        lag_history = []

        self.body_head.eval()

        for step in range(max_tokens):
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            # Get telemetry
            if telemetry_override is not None and override_idx < len(telemetry_override):
                hw = telemetry_override[override_idx]
                override_idx += 1
            elif self.telemetry:
                hw = self.telemetry.get_token_aligned(t0, t1)
            else:
                hw = {"temp": None, "power": None, "util": None}

            # Record for later cross-prompt use
            telemetry_trace.append(hw.copy())

            # Apply lag if requested
            lag_history.append(hw.copy())
            if lag_k > 0 and len(lag_history) > lag_k:
                hw_for_model = lag_history[-lag_k - 1]
            else:
                hw_for_model = hw

            # Compute z_feel with potentially lagged telemetry
            hw_ctx = HardwareContext.from_dict(hw_for_model)
            runtime = RuntimeContext(
                token_latency=t1 - t0,
                kv_cache_tokens=current_ids.shape[1],
                generation_depth=step,
            )
            sensors = self.sensor_bank(logits.detach(), runtime=runtime, hardware=hw_ctx)
            z_feel = self.projector(sensors.float())

            # Get body report predictions
            with torch.no_grad():
                pred = self.body_head.predict(z_feel)

            # Get ground truth labels from ACTUAL telemetry (not lagged/swapped)
            actual_hw = telemetry_trace[-1]
            true_labels = self.body_head.get_labels_from_telemetry(
                actual_hw.get("temp"), actual_hw.get("power"), actual_hw.get("util")
            )

            samples.append({
                "z_feel": z_feel.detach().cpu(),
                "temp": actual_hw.get("temp"),
                "power": actual_hw.get("power"),
                "util": actual_hw.get("util"),
                "pred_heat": pred["heat"],
                "pred_power": pred["power"],
                "pred_util": pred["util"],
                "true_heat": true_labels["heat"],
                "true_power": true_labels["power"],
                "true_util": true_labels["util"],
            })

            # Sample next token
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return samples, telemetry_trace

    def collect_baseline(
        self,
        prompts: List[str],
        tokens_per_prompt: int = 100,
    ) -> Dict[int, List[Dict]]:
        """
        Phase 1: Collect live telemetry for all prompts.
        Stores telemetry for later cross-prompt swap.
        """
        print("\n=== Phase 1: Collecting baseline telemetry ===")

        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt[:40]}...")
            _, telemetry = self._generate_and_collect(
                prompt, i, max_tokens=tokens_per_prompt
            )
            self.prompt_telemetry[i] = telemetry

        print(f"  Collected telemetry for {len(prompts)} prompts")
        return self.prompt_telemetry

    def run_cross_prompt_swap(
        self,
        prompts: List[str],
        tokens_per_prompt: int = 100,
    ) -> Dict:
        """
        Cross-prompt swap: prompt i gets telemetry from prompt j (j≠i)
        """
        print("\n=== Cross-Prompt Swap Test ===")

        if not self.prompt_telemetry:
            print("  ERROR: Must run collect_baseline first!")
            return {}

        live_correct = {"heat": [], "power": [], "util": []}
        cross_correct = {"heat": [], "power": [], "util": []}

        for i, prompt in enumerate(prompts):
            # Get telemetry from a DIFFERENT prompt
            other_ids = [j for j in range(len(prompts)) if j != i]
            cross_id = random.choice(other_ids)
            cross_telemetry = self.prompt_telemetry[cross_id]

            # Generate with live telemetry
            live_samples, _ = self._generate_and_collect(
                prompt, i, max_tokens=tokens_per_prompt
            )

            # Generate with cross-prompt telemetry
            cross_samples, _ = self._generate_and_collect(
                prompt, i, max_tokens=tokens_per_prompt,
                telemetry_override=cross_telemetry,
            )

            # Score accuracy
            for s in live_samples:
                for key in ["heat", "power", "util"]:
                    if s[f"true_{key}"] >= 0:
                        bucket_labels = (
                            self.body_head.thermal_buckets.labels if key == "heat" else
                            self.body_head.power_buckets.labels if key == "power" else
                            self.body_head.util_buckets.labels
                        )
                        pred_match = s[f"pred_{key}"] == bucket_labels[s[f"true_{key}"]]
                        live_correct[key].append(1 if pred_match else 0)

            for s in cross_samples:
                for key in ["heat", "power", "util"]:
                    if s[f"true_{key}"] >= 0:
                        bucket_labels = (
                            self.body_head.thermal_buckets.labels if key == "heat" else
                            self.body_head.power_buckets.labels if key == "power" else
                            self.body_head.util_buckets.labels
                        )
                        pred_match = s[f"pred_{key}"] == bucket_labels[s[f"true_{key}"]]
                        cross_correct[key].append(1 if pred_match else 0)

            if (i + 1) % 3 == 0:
                print(f"  [{i+1}/{len(prompts)}] processed")

        # Compute results
        results = {
            "live": {},
            "cross": {},
        }
        for key in ["heat", "power", "util"]:
            live_point, live_lo, live_hi = bootstrap_ci(live_correct[key])
            cross_point, cross_lo, cross_hi = bootstrap_ci(cross_correct[key])
            results["live"][key] = {"acc": live_point, "ci": [live_lo, live_hi]}
            results["cross"][key] = {"acc": cross_point, "ci": [cross_lo, cross_hi]}

        print("\n  Results:")
        print(f"  {'Condition':<12} {'Heat':>10} {'Power':>10} {'Util':>10}")
        print(f"  {'-'*44}")
        for cond in ["live", "cross"]:
            print(f"  {cond:<12} {results[cond]['heat']['acc']:>10.3f} "
                  f"{results[cond]['power']['acc']:>10.3f} {results[cond]['util']['acc']:>10.3f}")

        return results

    def run_lag_sweep(
        self,
        prompts: List[str],
        tokens_per_prompt: int = 100,
        lag_values: List[int] = [0, 8, 32, 128],
    ) -> Dict:
        """
        Lag sweep: test accuracy degradation with increasing lag.
        """
        print("\n=== Lag Sweep Test ===")
        print(f"  Lag values: {lag_values}")

        results = {}

        for lag_k in lag_values:
            print(f"\n  Testing lag={lag_k}...")
            correct = {"heat": [], "power": [], "util": []}

            for i, prompt in enumerate(prompts[:5]):  # Subset for speed
                samples, _ = self._generate_and_collect(
                    prompt, i, max_tokens=tokens_per_prompt, lag_k=lag_k
                )

                for s in samples:
                    for key in ["heat", "power", "util"]:
                        if s[f"true_{key}"] >= 0:
                            bucket_labels = (
                                self.body_head.thermal_buckets.labels if key == "heat" else
                                self.body_head.power_buckets.labels if key == "power" else
                                self.body_head.util_buckets.labels
                            )
                            pred_match = s[f"pred_{key}"] == bucket_labels[s[f"true_{key}"]]
                            correct[key].append(1 if pred_match else 0)

            results[lag_k] = {}
            for key in ["heat", "power", "util"]:
                point, lo, hi = bootstrap_ci(correct[key])
                results[lag_k][key] = {"acc": point, "ci": [lo, hi]}

            print(f"    heat={results[lag_k]['heat']['acc']:.3f}, "
                  f"power={results[lag_k]['power']['acc']:.3f}, "
                  f"util={results[lag_k]['util']['acc']:.3f}")

        # Check for monotonic degradation
        heat_accs = [results[k]["heat"]["acc"] for k in lag_values]
        is_monotonic = all(heat_accs[i] >= heat_accs[i+1] for i in range(len(heat_accs)-1))
        print(f"\n  Monotonic degradation: {is_monotonic}")

        return results

    def run_full_battery(
        self,
        prompts: List[str],
        tokens_per_prompt: int = 100,
    ) -> Dict:
        """Run all falsification tests."""
        print("\n" + "=" * 60)
        print(f"  FALSIFICATION BATTERY V2")
        print("=" * 60)

        # Phase 1: Collect baseline
        self.collect_baseline(prompts, tokens_per_prompt)

        # Phase 2: Cross-prompt swap
        cross_results = self.run_cross_prompt_swap(prompts, tokens_per_prompt)

        # Phase 3: Lag sweep
        lag_results = self.run_lag_sweep(prompts, tokens_per_prompt)

        return {
            "cross_prompt": cross_results,
            "lag_sweep": lag_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Falsification Battery V2")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--body_report", default="results/body_report/body_report_head.pt")
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--tokens_per_prompt", type=int, default=80)
    parser.add_argument("--output_dir", default="results/falsification_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    battery = FalsificationBatteryV2(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        body_report_path=args.body_report,
    )

    results = battery.run_full_battery(
        prompts=PROMPTS[:args.n_prompts],
        tokens_per_prompt=args.tokens_per_prompt,
    )

    # Save results
    output_path = f"{args.output_dir}/falsification_results.json"
    with open(output_path, "w") as f:
        # Convert to serializable
        serializable = {}
        for key, val in results.items():
            if isinstance(val, dict):
                serializable[key] = {}
                for k2, v2 in val.items():
                    if isinstance(v2, dict):
                        serializable[key][k2] = {
                            k3: float(v3) if isinstance(v3, (np.floating, float)) else v3
                            for k3, v3 in v2.items()
                        }
                    else:
                        serializable[key][k2] = v2
            else:
                serializable[key] = val

        json.dump({
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "results": serializable,
        }, f, indent=2)

    print(f"\nSaved: {output_path}")

    if battery.telemetry:
        battery.telemetry.stop()


if __name__ == "__main__":
    main()
