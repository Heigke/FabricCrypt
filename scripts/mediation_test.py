#!/usr/bin/env python3 -u
"""
Mediation Test: Does z_feel actually mediate telemetry → effort?

The scientific claim is:
    telemetry → z_feel → effort

NOT:
    telemetry → effort (bypassing z_feel)

To test mediation:
1. LIVE: Real telemetry → real z_feel → effort
2. Z_FEEL_SHUFFLE: Real telemetry, but z_feel from different timestep
   - If effort tracks telemetry despite shuffled z_feel → MEDIATION FAILS
   - If effort tracks shuffled z_feel (not real telemetry) → MEDIATION HOLDS

Pass criteria:
- In LIVE mode: effort correlates with telemetry
- In Z_FEEL_SHUFFLE mode: effort correlates with SHUFFLED z_feel, NOT real telemetry
"""

import sys
import time
import json
import argparse
import threading
import random
from pathlib import Path
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
    CanonicalSensorBank,
    RuntimeContext,
    HardwareContext,
)
from src.compute_effector import ZFeelEffortPolicy, _EffortActionWithPonder

VERSION = "mediation-test-v1.0.0"


class GPUStressor:
    """Background GPU load."""

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
        print("  [STRESSOR] ON")

    def _run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
        b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
        while self._running:
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            time.sleep(0.01)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("  [STRESSOR] OFF")

    @property
    def is_running(self):
        return self._running


class MediationTest:
    """
    Tests whether z_feel actually mediates the telemetry → effort relationship.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path: str = None,
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
            print(f"  Loaded projector")

        # Sensor bank
        self.sensor_bank = CanonicalSensorBank(mode="full")

        # Z_FEEL-only effort policy (the key component being tested)
        # Use magnitude mode since neural head is untrained
        self.effort_policy = ZFeelEffortPolicy(embed_dim=self.embed_dim, device=device, mode="magnitude")

        # Telemetry
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            print(f"  Telemetry: {self.telemetry.source}")
        except Exception as e:
            print(f"  Telemetry unavailable: {e}")

        self.stressor = GPUStressor()

    def run_trial(
        self,
        prompt: str,
        duration: float,
        mode: str,  # "live" or "z_feel_shuffle"
        shuffle_lag: int = 32,  # How many steps back to get shuffled z_feel
    ) -> dict:
        """
        Run a single trial.

        Args:
            prompt: Generation prompt
            duration: Duration in seconds
            mode: "live" (normal) or "z_feel_shuffle" (shuffled z_feel)
            shuffle_lag: For shuffle mode, how far back to sample z_feel

        Returns:
            Traces dict
        """
        print(f"\n  Running {mode} trial ({duration}s)...")

        self.effort_policy.reset()

        traces = {
            "mode": mode,
            "timestamps": [],
            "temps": [],
            "z_feel_norms": [],
            "effort_ema": [],
            "K_values": [],
            "stressor_active": [],
            # For shuffle mode: track the source of z_feel
            "z_feel_source_idx": [],
        }

        # z_feel history for shuffle mode
        z_feel_history = deque(maxlen=shuffle_lag + 10)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        start_time = time.time()
        token_count = 0

        while (time.time() - start_time) < duration:
            elapsed = time.time() - start_time

            # Perturbation schedule: stress on at 1/3, off at 2/3
            if elapsed > duration / 3 and elapsed < 2 * duration / 3:
                if not self.stressor.is_running:
                    self.stressor.start()
            else:
                if self.stressor.is_running:
                    self.stressor.stop()

            # Forward pass
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            # Get REAL telemetry
            hw = {}
            if self.telemetry:
                hw = self.telemetry.get_token_aligned(t0, t1)
            temp = hw.get("temp")

            # Compute REAL z_feel from real telemetry
            hw_ctx = HardwareContext.from_dict(hw)
            runtime = RuntimeContext(
                token_latency=t1 - t0,
                kv_cache_tokens=current_ids.shape[1],
                generation_depth=token_count,
            )
            sensors = self.sensor_bank(logits.detach(), runtime=runtime, hardware=hw_ctx)
            real_z_feel = self.projector(sensors.float())

            # Store in history
            z_feel_history.append(real_z_feel.clone())

            # Determine which z_feel to use for effort policy
            if mode == "live":
                z_feel_for_effort = real_z_feel
                source_idx = token_count
            elif mode == "z_feel_shuffle":
                # Use z_feel from shuffle_lag steps ago (if available)
                if len(z_feel_history) > shuffle_lag:
                    z_feel_for_effort = z_feel_history[0]  # Oldest in buffer
                    source_idx = max(0, token_count - shuffle_lag)
                else:
                    z_feel_for_effort = real_z_feel
                    source_idx = token_count
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Effort policy step (using selected z_feel)
            effort_action = self.effort_policy.step_from_z_feel(z_feel_for_effort)

            # Record
            traces["timestamps"].append(elapsed)
            traces["temps"].append(temp or 0)
            traces["z_feel_norms"].append(real_z_feel.norm().item())
            traces["effort_ema"].append(effort_action.effort_ema)
            traces["K_values"].append(effort_action.K)
            traces["stressor_active"].append(self.stressor.is_running)
            traces["z_feel_source_idx"].append(source_idx)

            token_count += 1

            # Progress
            if token_count % 100 == 0:
                temp_str = f"{temp:.1f}" if temp else "N/A"
                print(f"    [{elapsed:.1f}s] T={temp_str}°C e={effort_action.effort_ema:.2f} K={effort_action.K}")

            # Sample next token
            probs = F.softmax(logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if current_ids.shape[1] > 512:
                current_ids = input_ids.clone()

        self.stressor.stop()
        traces["tokens"] = token_count

        return traces

    def run_mediation_test(
        self,
        prompt: str = "Explain the principles of thermodynamics.",
        duration: float = 90.0,
        shuffle_lag: int = 32,
    ) -> dict:
        """
        Run the full mediation test: LIVE vs Z_FEEL_SHUFFLE.
        """
        print("\n" + "=" * 60)
        print("  MEDIATION TEST: Does z_feel mediate telemetry → effort?")
        print("=" * 60)

        # Run LIVE trial
        live_traces = self.run_trial(prompt, duration, mode="live")

        # Cooldown
        print("\n  Cooling down (30s)...")
        time.sleep(30)

        # Reset policy
        self.effort_policy.reset()

        # Run Z_FEEL_SHUFFLE trial
        shuffle_traces = self.run_trial(
            prompt, duration, mode="z_feel_shuffle", shuffle_lag=shuffle_lag
        )

        # Compute metrics
        metrics = self._compute_mediation_metrics(live_traces, shuffle_traces, shuffle_lag)

        return {
            "version": VERSION,
            "shuffle_lag": shuffle_lag,
            "live": live_traces,
            "z_feel_shuffle": shuffle_traces,
            "metrics": metrics,
        }

    def _compute_mediation_metrics(self, live: dict, shuffle: dict, lag: int) -> dict:
        """
        Compute mediation test metrics.

        Key insight:
        - In LIVE mode: effort should correlate with temp
        - In SHUFFLE mode:
          - If mediation HOLDS: effort should NOT correlate with temp
            (because effort follows shuffled z_feel, not real telemetry)
          - If mediation FAILS: effort still correlates with temp
            (because something is leaking temp → effort directly)
        """
        def corr(x, y):
            x, y = np.array(x), np.array(y)
            if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        # Live correlations
        live_temp_effort = corr(live["temps"], live["effort_ema"])

        # Shuffle correlations
        shuffle_temp_effort = corr(shuffle["temps"], shuffle["effort_ema"])

        # Lagged correlation in shuffle mode
        # The effort at time t should correlate with temp at time t-lag
        temps = np.array(shuffle["temps"])
        efforts = np.array(shuffle["effort_ema"])
        if len(temps) > lag:
            lagged_temps = temps[:-lag]
            lagged_efforts = efforts[lag:]
            shuffle_lagged_corr = corr(lagged_temps, lagged_efforts)
        else:
            shuffle_lagged_corr = 0.0

        # Mediation score
        # If mediation holds: live_temp_effort >> shuffle_temp_effort
        # The collapse indicates z_feel is actually mediating
        correlation_collapse = live_temp_effort - shuffle_temp_effort

        # The lagged correlation should be positive if effort follows shuffled z_feel
        mediation_evidence = shuffle_lagged_corr > shuffle_temp_effort

        return {
            "live_temp_effort_correlation": live_temp_effort,
            "shuffle_temp_effort_correlation": shuffle_temp_effort,
            "shuffle_lagged_correlation": shuffle_lagged_corr,
            "correlation_collapse": correlation_collapse,
            "mediation_holds": correlation_collapse > 0.2 and mediation_evidence,
            "interpretation": (
                "PASS: z_feel mediates telemetry→effort"
                if correlation_collapse > 0.2
                else "FAIL: effort leaks from telemetry directly"
            ),
        }

    def stop(self):
        self.stressor.stop()
        if self.telemetry:
            self.telemetry.stop()


def main():
    parser = argparse.ArgumentParser(description="Mediation Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--shuffle_lag", type=int, default=32)
    parser.add_argument("--output_dir", default="results/mediation_test")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    test = MediationTest(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
    )

    results = test.run_mediation_test(
        duration=args.duration,
        shuffle_lag=args.shuffle_lag,
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/mediation_test_{timestamp}.json"

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
    print("  MEDIATION TEST RESULTS")
    print("=" * 60)
    print(f"  Live temp↔effort:      {m['live_temp_effort_correlation']:.3f}")
    print(f"  Shuffle temp↔effort:   {m['shuffle_temp_effort_correlation']:.3f}")
    print(f"  Shuffle lagged corr:   {m['shuffle_lagged_correlation']:.3f}")
    print(f"  Correlation collapse:  {m['correlation_collapse']:.3f}")
    print(f"  Result: {m['interpretation']}")
    print("=" * 60)

    test.stop()


if __name__ == "__main__":
    main()
