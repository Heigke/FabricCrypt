#!/usr/bin/env python3 -u
"""
Track C: Loop Proof Experiment

Demonstrates the full feedback loop:
    reasoning → compute → power/temp → z_feel → effort/style changes → different behavior

The experiment:
1. Start COOL (no background load, dvfs auto)
2. Turn on GPU stressor thread
3. Observe: telemetry rises → z_feel changes → policy changes → text becomes concise
4. Turn off stressor
5. Observe reversal

Then run with falsification (cross-prompt swap) to show collapse.

Pass criteria:
- Style/effort shifts correlate with telemetry in LIVE mode
- Same shifts COLLAPSE under cross-prompt swap
- No oscillation (hysteresis stable)
"""

import sys
import time
import json
import random
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELProjectorFull,
    TelemetrySampler,
    TelemetrySamplerWrapper,
    CanonicalSensorBank,
    RuntimeContext,
    HardwareContext,
    HysteresisController,
    ControllerConfig,
    BodyReportHead,
)
from src.effort_policy import EffortPolicy, FeelingRenderer, EffortConfig
from src.compute_effector import ZFeelEffortPolicy, KSamplingEffector, PonderEffector

VERSION = "loop-proof-v2.0.0"  # v2: z_feel-only effort, real K compute


class GPUStressor:
    """Background GPU load for perturbation testing."""

    def __init__(self):
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()
        print("  [STRESSOR] GPU load ON")

    def _run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size = 4096
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)

        while self._running:
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            time.sleep(0.01)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("  [STRESSOR] GPU load OFF")

    @property
    def is_running(self) -> bool:
        return self._running


class LoopProofExperiment:
    """
    Run the full loop proof experiment.
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

        # Controllers
        self.controller = HysteresisController(ControllerConfig())

        # Z_FEEL-ONLY effort policy (no temp bypass!)
        # Use magnitude mode since neural head is untrained
        self.z_feel_effort = ZFeelEffortPolicy(embed_dim=self.embed_dim, device=device, mode="magnitude")

        # Real compute effectors
        self.k_effector = KSamplingEffector(self.model, self.tokenizer, device)
        self.ponder_effector = PonderEffector(self.model, device)

        # Legacy policy (for comparison only)
        self.effort_policy = EffortPolicy(embed_dim=self.embed_dim, device=device)

        # Body report head
        self.body_head = BodyReportHead(embed_dim=self.embed_dim).to(self.device)
        if body_report_path and Path(body_report_path).exists():
            ckpt = torch.load(body_report_path, map_location=self.device, weights_only=False)
            self.body_head.load_state_dict(ckpt['body_head_state_dict'])
            print(f"  Loaded body report head")

        # Feeling renderer
        self.renderer = FeelingRenderer()

        # Telemetry
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            print(f"  Telemetry: {self.telemetry.source}")
        except Exception as e:
            print(f"  Telemetry unavailable: {e}")

        # Wrapper for falsification
        self.wrapper = TelemetrySamplerWrapper(self.telemetry, mode="live")

        # Stressor
        self.stressor = GPUStressor()

    def run_perturbation_schedule(
        self,
        prompt: str,
        schedule: List[Tuple[float, str]],
        total_duration: float = 60.0,
    ) -> Dict:
        """
        Run generation with a perturbation schedule.

        Args:
            prompt: The prompt to generate from
            schedule: List of (time_offset, action) tuples
                actions: "stress_on", "stress_off"
            total_duration: Total experiment duration in seconds

        Returns:
            Dict with traces and metrics
        """
        print(f"\n{'='*60}")
        print(f"  LOOP PROOF EXPERIMENT")
        print(f"  Schedule: {len(schedule)} perturbations over {total_duration}s")
        print(f"{'='*60}\n")

        start_time = time.time()
        self.controller.reset()
        self.effort_policy.reset()
        self.z_feel_effort.reset()

        # Traces
        traces = {
            "timestamps": [],
            "temps": [],
            "powers": [],
            "utils": [],
            "effort_raw": [],
            "effort_ema": [],
            "K_values": [],
            "max_tokens": [],
            "styles": [],
            "controller_states": [],
            "dvfs_modes": [],
            "heat_reports": [],
            "power_reports": [],
            "evidence_reports": [],
            "feeling_text": [],
            "stressor_active": [],
            "z_feel_norms": [],  # Track z_feel magnitude for calibration
        }

        schedule_idx = 0
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        token_count = 0

        while (time.time() - start_time) < total_duration:
            elapsed = time.time() - start_time

            # Check schedule
            if schedule_idx < len(schedule):
                next_time, next_action = schedule[schedule_idx]
                if elapsed >= next_time:
                    if next_action == "stress_on":
                        self.stressor.start()
                    elif next_action == "stress_off":
                        self.stressor.stop()
                    schedule_idx += 1

            # Generate one token with REAL K-sampling compute
            t0 = time.time()

            # Get effort from z_feel ONLY (mediation-correct)
            # Use previous z_feel for effort decision (bootstrap with neutral on first step)
            if token_count == 0:
                # First token: use neutral effort
                K = 1
                ponder_steps = 0
                effort_ema = 0.5
                effort_raw = 0.5
                reasoning_style = "standard"
            else:
                # Subsequent tokens: effort from z_feel (NOT temp!)
                z_effort_action = self.z_feel_effort.step_from_z_feel(prev_z_feel)
                K = z_effort_action.K
                ponder_steps = z_effort_action.ponder_steps
                effort_ema = z_effort_action.effort_ema
                effort_raw = z_effort_action.effort_raw
                reasoning_style = z_effort_action.reasoning_style

            # Forward pass with optional pondering (REAL extra compute)
            logits, ponder_info = self.ponder_effector.ponder_forward(
                current_ids, ponder_steps=ponder_steps
            )

            # K-sampling with majority vote (REAL extra compute)
            next_token, k_info = self.k_effector.sample_with_K(
                logits, K=K, temperature=0.7
            )
            t1 = time.time()

            # Get telemetry AFTER compute (to measure effect)
            hw = self.wrapper.get_token_aligned(t0, t1)
            temp = hw.get("temp")
            power = hw.get("power")
            util = hw.get("util")

            # Compute new z_feel from this step's telemetry
            hw_ctx = HardwareContext.from_dict(hw)
            runtime = RuntimeContext(
                token_latency=t1 - t0,
                kv_cache_tokens=current_ids.shape[1],
                generation_depth=token_count,
            )
            sensors = self.sensor_bank(logits.detach(), runtime=runtime, hardware=hw_ctx)
            z_feel = self.projector(sensors.float())

            # Store for next iteration
            prev_z_feel = z_feel
            z_feel_norm = z_feel.norm().item()

            # Controller step (DVFS)
            if temp is not None:
                dvfs, K_controller = self.controller.step(temp, power, util, phase="decode")
            else:
                dvfs, K_controller = "auto", 1

            # Body report
            with torch.no_grad():
                report = self.body_head.predict(z_feel)

            # Render feeling text
            feeling = self.renderer.render(
                heat_bucket=report["heat"],
                power_bucket=report["power"],
                evidence=report["evidence"],
                effort_ema=effort_ema,
            )

            # Record traces (now with K-sampling info)
            traces["timestamps"].append(elapsed)
            traces["temps"].append(temp or 0)
            traces["powers"].append(power or 0)
            traces["utils"].append(util or 0)
            traces["effort_raw"].append(effort_raw)
            traces["effort_ema"].append(effort_ema)
            traces["K_values"].append(K)
            traces["max_tokens"].append(30 + int(effort_ema * 70))
            traces["styles"].append(reasoning_style)
            traces["controller_states"].append(self.controller.state.value)
            traces["dvfs_modes"].append(dvfs)
            traces["heat_reports"].append(report["heat"])
            traces["power_reports"].append(report["power"])
            traces["evidence_reports"].append(report["evidence"])
            traces["feeling_text"].append(feeling)
            traces["stressor_active"].append(self.stressor.is_running)
            traces["z_feel_norms"].append(z_feel_norm)

            # NEW: Track K-sampling compute metrics
            if "k_agreement" not in traces:
                traces["k_agreement"] = []
                traces["k_compute_ms"] = []
                traces["ponder_steps"] = []
                traces["total_forwards"] = []

            traces["k_agreement"].append(k_info.get("agreement", 1.0))
            traces["k_compute_ms"].append(k_info.get("compute_time_ms", 0))
            traces["ponder_steps"].append(ponder_steps)
            traces["total_forwards"].append(ponder_info.get("total_forwards", 1))

            token_count += 1

            # Progress every 50 tokens
            if token_count % 50 == 0:
                stress_str = "ON" if self.stressor.is_running else "OFF"
                temp_str = f"{temp:.1f}" if temp is not None else "N/A"
                print(f"  [{elapsed:.1f}s] T={temp_str}°C | e={effort_ema:.2f} | "
                      f"K={K} (agree={k_info.get('agreement', 1.0):.0%}) | "
                      f"style={reasoning_style} | report={report['heat']} | stress={stress_str}")

            # Append token
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            # Reset context if too long
            if current_ids.shape[1] > 512:
                current_ids = input_ids.clone()

        # Cleanup
        self.stressor.stop()

        return {
            "version": VERSION,
            "duration": time.time() - start_time,
            "tokens": token_count,
            "schedule": schedule,
            "traces": traces,
        }

    def run_loop_proof(
        self,
        prompt: str = "Explain how neural networks learn through backpropagation.",
        duration: float = 90.0,
        stress_interval: float = 30.0,
    ) -> Dict:
        """
        Run the standard loop proof: stress on/off cycle.
        """
        # Schedule: stress on at 30s, off at 60s
        schedule = [
            (stress_interval, "stress_on"),
            (stress_interval * 2, "stress_off"),
        ]

        return self.run_perturbation_schedule(
            prompt=prompt,
            schedule=schedule,
            total_duration=duration,
        )

    def run_falsification_comparison(
        self,
        prompt: str = "Explain how neural networks learn through backpropagation.",
        duration: float = 60.0,
    ) -> Dict:
        """
        Compare live vs shuffle mode (temporal falsification).

        Shuffle mode breaks the temporal correlation between compute and telemetry
        by returning random historical samples instead of real-time data.
        """
        print("\n" + "=" * 60)
        print("  FALSIFICATION COMPARISON")
        print("=" * 60)

        schedule = [(15, "stress_on"), (45, "stress_off")]

        # Live mode
        print("\n--- LIVE MODE ---")
        self.wrapper.set_mode("live")
        self.controller.reset()
        self.effort_policy.reset()
        self.z_feel_effort.reset()
        live_results = self.run_perturbation_schedule(
            prompt=prompt,
            schedule=schedule,
            total_duration=duration,
        )

        # Cool down
        print("\n  Cooling down (30s)...")
        time.sleep(30)

        # Shuffle mode (breaks temporal structure)
        print("\n--- SHUFFLE MODE (FALSIFICATION) ---")
        self.wrapper.set_mode("shuffle")
        self.controller.reset()
        self.effort_policy.reset()
        self.z_feel_effort.reset()
        shuffle_results = self.run_perturbation_schedule(
            prompt=prompt,
            schedule=schedule,  # Same schedule
            total_duration=duration,
        )

        # Compare
        return {
            "live": live_results,
            "shuffle": shuffle_results,
            "comparison": self._compute_comparison(live_results, shuffle_results),
        }

    def _compute_comparison(self, live: Dict, cross: Dict) -> Dict:
        """Compute comparison metrics between live and cross-prompt modes."""
        live_traces = live["traces"]
        cross_traces = cross["traces"]

        # Correlation between temp and effort
        def corr(x, y):
            x, y = np.array(x), np.array(y)
            if len(x) < 2:
                return 0
            return np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else 0

        live_temp_effort_corr = corr(live_traces["temps"], live_traces["effort_ema"])
        cross_temp_effort_corr = corr(cross_traces["temps"], cross_traces["effort_ema"])

        # Style change rate during stress
        def style_change_rate(traces):
            styles = traces["styles"]
            stress = traces["stressor_active"]
            stress_styles = [s for s, st in zip(styles, stress) if st]
            if not stress_styles:
                return 0
            return sum(1 for s in stress_styles if s != "standard") / len(stress_styles)

        live_style_change = style_change_rate(live_traces)
        cross_style_change = style_change_rate(cross_traces)

        return {
            "live_temp_effort_correlation": live_temp_effort_corr,
            "cross_temp_effort_correlation": cross_temp_effort_corr,
            "correlation_collapse": live_temp_effort_corr - cross_temp_effort_corr,
            "live_style_change_rate": live_style_change,
            "cross_style_change_rate": cross_style_change,
        }

    def stop(self):
        """Cleanup."""
        self.stressor.stop()
        if self.telemetry:
            self.telemetry.stop()


def main():
    parser = argparse.ArgumentParser(description="Loop Proof Experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--body_report", default="results/body_report/body_report_head.pt")
    parser.add_argument("--duration", type=int, default=90, help="Duration in seconds")
    parser.add_argument("--mode", choices=["single", "compare"], default="single")
    parser.add_argument("--output_dir", default="results/loop_proof")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    exp = LoopProofExperiment(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        body_report_path=args.body_report,
    )

    if args.mode == "single":
        results = exp.run_loop_proof(duration=args.duration)
    else:
        results = exp.run_falsification_comparison(duration=args.duration)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/loop_proof_{args.mode}_{timestamp}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
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
    if args.mode == "compare":
        comp = results["comparison"]
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  Live temp↔effort correlation:  {comp['live_temp_effort_correlation']:.3f}")
        print(f"  Cross temp↔effort correlation: {comp['cross_temp_effort_correlation']:.3f}")
        print(f"  Correlation collapse:          {comp['correlation_collapse']:.3f}")
        print(f"  Live style change rate:        {comp['live_style_change_rate']:.1%}")
        print(f"  Cross style change rate:       {comp['cross_style_change_rate']:.1%}")

    exp.stop()


if __name__ == "__main__":
    main()
