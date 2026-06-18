#!/usr/bin/env python3 -u
"""
Demo Dashboard - The Unmistakable Loop

Visual demonstration of:
reasoning -> compute -> power/temp -> z_feel shifts -> controller changes -> different behavior

4 Plots:
1. Telemetry (temp/power/util over time)
2. z_feel (PCA/2D projection)
3. Actions (dvfs + K as step plot)
4. Language behavior (length/entropy/tok/s)

3 Toggles:
- closed_loop: controller active
- shuffle: randomize sensors
- replay: use recorded telemetry

Usage:
    python scripts/demo_dashboard.py --mode terminal  # ASCII plots
    python scripts/demo_dashboard.py --mode gradio    # Web UI (requires gradio)
"""

import sys
import time
import json
import random
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional
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

VERSION = "demo-v1.0.0"


class DemoState:
    """Shared state for demo visualization."""

    def __init__(self, max_history: int = 200):
        self.max_history = max_history

        # Traces
        self.temps = deque(maxlen=max_history)
        self.powers = deque(maxlen=max_history)
        self.utils = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)

        # z_feel trajectory (2D for plotting)
        self.z_feel_x = deque(maxlen=max_history)
        self.z_feel_y = deque(maxlen=max_history)

        # Actions
        self.dvfs_modes = deque(maxlen=max_history)
        self.K_values = deque(maxlen=max_history)
        self.states = deque(maxlen=max_history)

        # Language metrics
        self.entropies = deque(maxlen=max_history)
        self.tok_lengths = deque(maxlen=max_history)
        self.tok_per_sec = deque(maxlen=max_history)

        # Body reports
        self.heat_reports = deque(maxlen=max_history)

        # Start time
        self.start_time = time.time()

        # Toggles
        self.closed_loop = True
        self.shuffle_mode = False
        self.replay_mode = False

    def reset(self):
        """Clear all traces."""
        for attr in ['temps', 'powers', 'utils', 'timestamps',
                     'z_feel_x', 'z_feel_y', 'dvfs_modes', 'K_values',
                     'states', 'entropies', 'tok_lengths', 'tok_per_sec',
                     'heat_reports']:
            getattr(self, attr).clear()
        self.start_time = time.time()


class DemoRunner:
    """
    Run the demonstration loop.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path: str = None,
        body_report_path: str = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print("Loading model...")
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

        # Sensor bank
        self.sensor_bank = CanonicalSensorBank(mode="full")

        # Controller
        self.controller = HysteresisController(ControllerConfig())

        # Body report head
        self.body_head = BodyReportHead(embed_dim=self.embed_dim).to(self.device)
        if body_report_path and Path(body_report_path).exists():
            ckpt = torch.load(body_report_path, map_location=self.device, weights_only=False)
            self.body_head.load_state_dict(ckpt['body_head_state_dict'])

        # Telemetry
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
        except Exception as e:
            print(f"Telemetry unavailable: {e}")

        # Wrapper for falsification
        self.wrapper = TelemetrySamplerWrapper(self.telemetry, mode="live")

        # PCA for z_feel visualization (simple 2D projection)
        self.pca_proj = torch.randn(self.embed_dim, 2) / np.sqrt(self.embed_dim)

        # State
        self.state = DemoState()

        # Running flag
        self._running = False

    def _project_z_feel(self, z_feel: torch.Tensor) -> tuple:
        """Project z_feel to 2D for visualization."""
        with torch.no_grad():
            proj = z_feel @ self.pca_proj.to(z_feel.device)
            return proj[0, 0].item(), proj[0, 1].item()

    def step(self, prompt: str, max_tokens: int = 30) -> str:
        """
        Generate tokens and update state.
        """
        # Update wrapper mode based on toggles
        if self.state.shuffle_mode:
            self.wrapper.set_mode("shuffle")
        elif self.state.replay_mode:
            self.wrapper.set_mode("replay")
        else:
            self.wrapper.set_mode("live")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()

        generated_text = ""
        token_times = []

        for step in range(max_tokens):
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :].float()
            t1 = time.time()

            # Get telemetry through wrapper
            hw = self.wrapper.get_token_aligned(t0, t1)
            temp = hw.get("temp")
            power = hw.get("power")
            util = hw.get("util")

            # Compute z_feel
            hw_ctx = HardwareContext.from_dict(hw)
            runtime = RuntimeContext(
                token_latency=t1 - t0,
                kv_cache_tokens=current_ids.shape[1],
                generation_depth=step,
            )
            sensors = self.sensor_bank(logits.detach(), runtime=runtime, hardware=hw_ctx)
            z_feel = self.projector(sensors.float())

            # Controller step
            if self.state.closed_loop and temp is not None:
                dvfs, K = self.controller.step(temp, power, util, phase="decode")
            else:
                dvfs, K = "auto", 1

            # Body report
            with torch.no_grad():
                report = self.body_head.predict(z_feel)

            # Compute token entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum().item()

            # Update state
            elapsed = time.time() - self.state.start_time
            self.state.timestamps.append(elapsed)
            self.state.temps.append(temp or 0)
            self.state.powers.append(power or 0)
            self.state.utils.append(util or 0)

            x, y = self._project_z_feel(z_feel)
            self.state.z_feel_x.append(x)
            self.state.z_feel_y.append(y)

            self.state.dvfs_modes.append(1 if dvfs == "auto" else 0)
            self.state.K_values.append(K)
            self.state.states.append(self.controller.state.value)

            self.state.entropies.append(entropy)
            self.state.tok_per_sec.append(1.0 / (t1 - t0) if (t1 - t0) > 0 else 0)

            self.state.heat_reports.append(report["heat"])

            token_times.append(t1 - t0)

            # Sample next token
            temp_sampling = 0.5 if self.controller.state.value == "hot" else 0.8
            probs = F.softmax(logits / temp_sampling, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_ids = current_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        self.state.tok_lengths.append(len(generated_ids))

        return generated_text

    def render_terminal(self):
        """Render ASCII dashboard to terminal."""
        # Clear screen
        print("\033[2J\033[H", end="")

        print("=" * 70)
        print(f"  FEEL DEMO DASHBOARD {VERSION}")
        print(f"  Toggles: closed_loop={self.state.closed_loop}, "
              f"shuffle={self.state.shuffle_mode}, replay={self.state.replay_mode}")
        print("=" * 70)

        # Current values
        if self.state.temps:
            temp = self.state.temps[-1]
            power = self.state.powers[-1]
            state = self.state.states[-1] if self.state.states else "none"
            heat_report = self.state.heat_reports[-1] if self.state.heat_reports else "?"
            tok_s = self.state.tok_per_sec[-1] if self.state.tok_per_sec else 0

            print(f"\n  CURRENT: T={temp:.1f}°C | P={power:.1f}W | "
                  f"State={state} | Report={heat_report} | {tok_s:.1f} tok/s")

        # Simple ASCII sparkline for temperature
        if len(self.state.temps) > 10:
            temps = list(self.state.temps)[-50:]
            min_t, max_t = min(temps), max(temps)
            range_t = max_t - min_t or 1
            chars = "▁▂▃▄▅▆▇█"

            sparkline = ""
            for t in temps:
                idx = int((t - min_t) / range_t * 7)
                sparkline += chars[min(idx, 7)]

            print(f"\n  TEMP:  [{min_t:.0f}°C] {sparkline} [{max_t:.0f}°C]")

        # Controller state timeline
        if len(self.state.states) > 10:
            states = list(self.state.states)[-50:]
            timeline = "".join(["H" if s == "hot" else "C" for s in states])
            print(f"  STATE: {timeline}")

        # z_feel trajectory (simple plot)
        if len(self.state.z_feel_x) > 5:
            x_vals = list(self.state.z_feel_x)[-10:]
            y_vals = list(self.state.z_feel_y)[-10:]
            print(f"\n  z_feel: x=[{x_vals[-1]:.2f}] y=[{y_vals[-1]:.2f}]")

        print("\n" + "-" * 70)
        print("  Press Ctrl+C to stop")

    def run_loop(self, prompts: List[str], interval: float = 2.0):
        """Run continuous demo loop."""
        self._running = True
        prompt_idx = 0

        print("\nStarting demo loop...")

        try:
            while self._running:
                prompt = prompts[prompt_idx % len(prompts)]
                prompt_idx += 1

                # Generate
                output = self.step(prompt, max_tokens=30)

                # Render
                self.render_terminal()

                # Small delay
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nStopping demo...")
            self._running = False

    def stop(self):
        """Stop the demo."""
        self._running = False
        if self.telemetry:
            self.telemetry.stop()


DEMO_PROMPTS = [
    "Explain neural networks briefly.",
    "What is quantum computing?",
    "Describe photosynthesis.",
    "How do CPUs work?",
    "What is machine learning?",
]


def main():
    parser = argparse.ArgumentParser(description="Demo Dashboard")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--body_report", default="results/body_report/body_report_head.pt")
    parser.add_argument("--mode", choices=["terminal", "gradio"], default="terminal")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    runner = DemoRunner(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        body_report_path=args.body_report,
    )

    if args.mode == "terminal":
        runner.run_loop(DEMO_PROMPTS, interval=args.interval)
    else:
        print("Gradio mode not yet implemented. Use --mode terminal")

    runner.stop()


if __name__ == "__main__":
    main()
