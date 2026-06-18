#!/usr/bin/env python3
"""
z117 - Unified FEEL-SLM Runtime Pipeline

The "spinal cord" that connects all components:
1. Sense: Calibrated telemetry (normalized units!)
2. Feel: BodyLatent (scaled, calibrated)
3. Decide: Policy selects compute-mode + hw-profile
4. Actuate: Rate-limited, verified
5. Express: Reporter output (grounded)
6. Log: Everything to trajectory store

Business demo showing:
- mJ/token (energy efficiency)
- SLO (p95/p99 latency, tokens/sec)
- Quality proxy (perplexity)
- $/1M tokens (business hook)

Usage:
  python z117_unified_pipeline.py --calibrate    # Run calibration
  python z117_unified_pipeline.py --demo         # Run business demo
  python z117_unified_pipeline.py --benchmark    # Full A/B benchmark
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feel_slm.calibration import (
    TelemetryCalibrator, CalibratedTelemetry,
    CalibrationResult, generate_calibration_report,
)
from src.feel_slm.adaptive_config import (
    AdaptiveConfigCalibrator, MachineProfile,
    get_adaptive_phase_configs,
)
from src.feel_slm.phase_controller import PhaseSeparatedController, InferencePhase
from src.feel_slm.embodied_slm import EmbodiedSLM, create_embodied_slm_30m


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Unified pipeline configuration."""
    # Calibration
    calibration_path: str = "results/calibration.json"
    machine_profile_path: str = "results/machine_profile.json"

    # Model
    model_checkpoint: str = "results/z116_embodied/embodied_slm_final.pt"
    use_embodied: bool = True

    # Telemetry
    telemetry_interval_ms: float = 10.0  # How often to read telemetry

    # Actuation
    actuation_rate_limit_s: float = 0.5  # Min time between actuations

    # SLO targets
    target_ttft_ms: float = 500.0   # Time to first token
    target_tpot_ms: float = 50.0    # Time per output token
    target_throughput_tps: float = 20.0  # Tokens per second

    # Business
    electricity_price_kwh: float = 0.12  # $ per kWh

    # Logging
    output_dir: str = "results/z117_pipeline"
    log_trajectory: bool = True


# =============================================================================
# Business Metrics
# =============================================================================

@dataclass
class BusinessMetrics:
    """Metrics for business demo."""
    # Efficiency
    mj_per_token: float = 0.0
    tokens_per_joule: float = 0.0

    # Cost
    dollars_per_million_tokens: float = 0.0
    kwh_per_million_tokens: float = 0.0

    # SLO
    ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    throughput_tps: float = 0.0

    # SLO compliance
    ttft_slo_met: bool = True
    tpot_slo_met: bool = True

    # Quality
    perplexity: float = 0.0

    # Comparison
    vs_baseline_energy_pct: float = 0.0
    vs_baseline_cost_pct: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    FEEL-SLM BUSINESS METRICS                      ║
╠══════════════════════════════════════════════════════════════════╣
║  EFFICIENCY                                                       ║
║    mJ/token:           {self.mj_per_token:8.2f}                           ║
║    tokens/Joule:       {self.tokens_per_joule:8.2f}                           ║
║                                                                   ║
║  COST (@ ${self.dollars_per_million_tokens / (self.kwh_per_million_tokens + 1e-9) * 1000:.2f}/kWh)                                           ║
║    $/1M tokens:        ${self.dollars_per_million_tokens:7.4f}                           ║
║    kWh/1M tokens:      {self.kwh_per_million_tokens:8.4f}                           ║
║                                                                   ║
║  LATENCY (SLO)                                                    ║
║    TTFT:               {self.ttft_ms:8.2f} ms  {'✓' if self.ttft_slo_met else '✗'}                      ║
║    Avg TPOT:           {self.avg_tpot_ms:8.2f} ms                           ║
║    p95 TPOT:           {self.p95_tpot_ms:8.2f} ms  {'✓' if self.tpot_slo_met else '✗'}                      ║
║    Throughput:         {self.throughput_tps:8.2f} tok/s                       ║
║                                                                   ║
║  QUALITY                                                          ║
║    Perplexity:         {self.perplexity:8.2f}                           ║
║                                                                   ║
║  vs BASELINE                                                      ║
║    Energy:             {self.vs_baseline_energy_pct:+7.1f}%                           ║
║    Cost:               {self.vs_baseline_cost_pct:+7.1f}%                           ║
╚══════════════════════════════════════════════════════════════════╝
"""


def compute_business_metrics(
    total_tokens: int,
    total_energy_j: float,
    ttft_ms: float,
    tpot_list: List[float],
    perplexity: float,
    config: PipelineConfig,
    baseline_mj_per_token: float = None,
) -> BusinessMetrics:
    """Compute business metrics from raw measurements."""
    metrics = BusinessMetrics()

    # Efficiency
    metrics.mj_per_token = (total_energy_j * 1000) / total_tokens if total_tokens > 0 else 0
    metrics.tokens_per_joule = total_tokens / total_energy_j if total_energy_j > 0 else 0

    # Cost
    kwh = total_energy_j / 3_600_000  # J to kWh
    metrics.kwh_per_million_tokens = (kwh / total_tokens) * 1_000_000 if total_tokens > 0 else 0
    metrics.dollars_per_million_tokens = metrics.kwh_per_million_tokens * config.electricity_price_kwh

    # Latency
    metrics.ttft_ms = ttft_ms
    if tpot_list:
        metrics.avg_tpot_ms = sum(tpot_list) / len(tpot_list)
        sorted_tpot = sorted(tpot_list)
        metrics.p95_tpot_ms = sorted_tpot[int(len(sorted_tpot) * 0.95)] if len(sorted_tpot) > 20 else max(tpot_list)
        metrics.p99_tpot_ms = sorted_tpot[int(len(sorted_tpot) * 0.99)] if len(sorted_tpot) > 100 else max(tpot_list)
        total_time_s = sum(tpot_list) / 1000
        metrics.throughput_tps = total_tokens / total_time_s if total_time_s > 0 else 0

    # SLO compliance
    metrics.ttft_slo_met = metrics.ttft_ms <= config.target_ttft_ms
    metrics.tpot_slo_met = metrics.p95_tpot_ms <= config.target_tpot_ms

    # Quality
    metrics.perplexity = perplexity

    # vs baseline
    if baseline_mj_per_token and baseline_mj_per_token > 0:
        metrics.vs_baseline_energy_pct = (metrics.mj_per_token - baseline_mj_per_token) / baseline_mj_per_token * 100
        metrics.vs_baseline_cost_pct = metrics.vs_baseline_energy_pct

    return metrics


# =============================================================================
# Unified Pipeline
# =============================================================================

class UnifiedPipeline:
    """
    Unified FEEL-SLM runtime pipeline.

    Single process loop: sense → feel → decide → actuate → express → log
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_telemetry()
        self._init_actuator()
        self._init_model()
        self._init_controller()

        # Trajectory logging
        self.trajectory: List[Dict] = []

    def _init_telemetry(self):
        """Initialize calibrated telemetry."""
        print("Initializing telemetry...")

        # Load calibration if available
        calibration = None
        if Path(self.config.calibration_path).exists():
            calibration = TelemetryCalibrator.load_calibration(self.config.calibration_path)
            print(f"  Loaded calibration from {self.config.calibration_path}")

        self.telemetry = CalibratedTelemetry(calibration)
        caps = self.telemetry.get_capabilities()
        print(f"  Capabilities: {caps}")

    def _init_actuator(self):
        """Initialize actuator client."""
        print("Initializing actuator...")
        self.actuator = None

        try:
            from src.actuator.client import ActuatorClient
            self.actuator = ActuatorClient()
            print("  Actuator connected")
        except Exception as e:
            print(f"  Actuator not available: {e}")

    def _init_model(self):
        """Initialize model (embodied or baseline)."""
        print("Initializing model...")

        self.model = create_embodied_slm_30m()

        # Load checkpoint if available
        if Path(self.config.model_checkpoint).exists():
            self.model.load_state_dict(torch.load(self.config.model_checkpoint, map_location=self.device))
            print(f"  Loaded checkpoint: {self.config.model_checkpoint}")
        else:
            print("  Using randomly initialized model")

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"  Model: {self.model.num_parameters / 1e6:.1f}M parameters")

    def _init_controller(self):
        """Initialize phase-separated controller."""
        print("Initializing controller...")

        # Load machine profile if available
        self.machine_profile = None
        if Path(self.config.machine_profile_path).exists():
            self.machine_profile = AdaptiveConfigCalibrator.load_profile(self.config.machine_profile_path)
            print(f"  Loaded machine profile: {self.machine_profile.machine_name}")

        self.controller = PhaseSeparatedController(
            actuator=self.actuator,
            telemetry_source=self.telemetry,
            model=self.model if self.config.use_embodied else None,
        )

        # Apply adaptive configs if available
        if self.machine_profile:
            # Update controller's phase configs
            adaptive_configs = get_adaptive_phase_configs(self.machine_profile)
            self.controller.PHASE_CONFIGS.update(adaptive_configs)
            print("  Applied adaptive phase configs")

    def run_calibration(self, duration_s: float = 60) -> Tuple[CalibrationResult, MachineProfile]:
        """Run full calibration (telemetry + adaptive config)."""
        print("\n" + "=" * 60)
        print("RUNNING CALIBRATION")
        print("=" * 60)

        # Step 1: Telemetry calibration
        print("\n1. Telemetry Calibration")
        calibrator = TelemetryCalibrator()
        calibration = calibrator.run_calibration(duration_s=duration_s)
        calibrator.save_calibration(calibration, self.config.calibration_path)

        # Generate report
        report = generate_calibration_report(calibration)
        report_path = self.config.calibration_path.replace(".json", "_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  Report saved to {report_path}")

        # Step 2: Adaptive config calibration
        print("\n2. Adaptive Config Calibration")
        self.telemetry = CalibratedTelemetry(calibration)
        config_calibrator = AdaptiveConfigCalibrator(
            telemetry=self.telemetry,
            actuator=self.actuator,
        )
        profile = config_calibrator.run_calibration(duration_per_profile_s=15)
        config_calibrator.save_profile(profile, self.config.machine_profile_path)

        print("\nCalibration complete!")
        print(f"  Telemetry: {self.config.calibration_path}")
        print(f"  Profile: {self.config.machine_profile_path}")

        return calibration, profile

    def run_generation(
        self,
        prompt: str = "The meaning of life is",
        max_tokens: int = 64,
        use_feel: bool = True,
    ) -> Tuple[str, BusinessMetrics]:
        """
        Run one generation with full sensing/actuation loop.

        Returns generated text and metrics.
        """
        # Encode prompt (simple char-level for demo)
        input_ids = torch.tensor([[ord(c) % 32000 for c in prompt]], dtype=torch.long, device=self.device)

        # Start tracking
        tpot_list = []
        power_samples = []
        trajectory_points = []

        # Signal prefill start
        self.controller.start_request(prompt_length=input_ids.shape[1])

        # Prefill (process prompt)
        prefill_start = time.time()
        with torch.no_grad():
            # Read telemetry
            telem = self.telemetry.read()
            telemetry_vec = self._telem_to_tensor(telem)

            # Forward
            if use_feel:
                outputs = self.model(input_ids, telemetry=telemetry_vec)
            else:
                outputs = self.model(input_ids)

            logits = outputs["logits"][:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        prefill_end = time.time()
        ttft_ms = (prefill_end - prefill_start) * 1000

        # Signal first token
        self.controller.on_first_token(ttft_ms)

        # Decode loop
        generated = torch.cat([input_ids, next_token], dim=1)
        total_energy_j = 0.0

        for token_idx in range(max_tokens - 1):
            token_start = time.time()

            # SENSE: Read telemetry
            telem = self.telemetry.read()
            power_w = telem["power_w"]
            power_samples.append(power_w)

            # FEEL: Encode body state
            telemetry_vec = self._telem_to_tensor(telem)

            # DECIDE: Check control window
            if (token_idx + 1) % 32 == 0:
                decision = self.controller.on_control_window()
                # Could apply decision here (profile change, LayerDrop, etc.)

            # GENERATE: Forward pass
            with torch.no_grad():
                if use_feel:
                    outputs = self.model(generated, telemetry=telemetry_vec)
                else:
                    outputs = self.model(generated)

                logits = outputs["logits"][:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            token_end = time.time()
            tpot_ms = (token_end - token_start) * 1000
            tpot_list.append(tpot_ms)

            # ACTUATE: Signal token to controller
            self.controller.on_token_generated(tpot_ms, power_w)

            # Energy tracking
            energy_j = power_w * (tpot_ms / 1000)
            total_energy_j += energy_j

            # LOG: Record trajectory point
            if self.config.log_trajectory:
                trajectory_points.append({
                    "timestamp": time.time(),
                    "token_idx": token_idx,
                    "power_w": power_w,
                    "temp_c": telem["temp_c"],
                    "gpu_util": telem["gpu_util"],
                    "tpot_ms": tpot_ms,
                    "energy_j": energy_j,
                    "phase": "decode",
                })

        # End request
        summary = self.controller.end_request()

        # Decode output
        output_tokens = generated[0].tolist()
        output_text = "".join(chr(t % 128) for t in output_tokens)

        # Compute perplexity (simplified)
        with torch.no_grad():
            if use_feel:
                outputs = self.model(generated, telemetry=telemetry_vec)
            else:
                outputs = self.model(generated)
            logits = outputs["logits"]
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                generated[:, 1:].reshape(-1),
            )
            perplexity = torch.exp(loss).item()

        # Compute business metrics
        metrics = compute_business_metrics(
            total_tokens=max_tokens,
            total_energy_j=total_energy_j,
            ttft_ms=ttft_ms,
            tpot_list=tpot_list,
            perplexity=perplexity,
            config=self.config,
        )

        # Store trajectory
        self.trajectory.extend(trajectory_points)

        return output_text, metrics

    def _telem_to_tensor(self, telem: Dict) -> torch.Tensor:
        """Convert telemetry dict to tensor."""
        return torch.tensor([
            telem.get("power_w", 50) / 100,
            telem.get("temp_c", 50) / 100,
            telem.get("gpu_util", 50) / 100,
            telem.get("clock_gfx_mhz", 1500) / 2500,
            telem.get("fan_rpm", 0) / 3000,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # Padding to 12 dims
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

    def run_benchmark(self, n_prompts: int = 20, max_tokens: int = 64) -> Dict[str, BusinessMetrics]:
        """
        Run A/B benchmark: Baseline vs FEEL.

        Returns metrics for both configurations.
        """
        print("\n" + "=" * 60)
        print("A/B BENCHMARK: Baseline vs FEEL")
        print("=" * 60)

        prompts = [
            "The quick brown fox",
            "Machine learning is",
            "Energy efficiency means",
            "The future of AI",
            "Sustainable computing requires",
        ] * (n_prompts // 5 + 1)
        prompts = prompts[:n_prompts]

        results = {}

        # Baseline run
        print("\n[BASELINE] Running without FEEL...")
        baseline_metrics_list = []
        for i, prompt in enumerate(prompts):
            _, metrics = self.run_generation(prompt, max_tokens, use_feel=False)
            baseline_metrics_list.append(metrics)
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i + 1}/{n_prompts}")

        # Aggregate baseline
        baseline_avg = BusinessMetrics(
            mj_per_token=sum(m.mj_per_token for m in baseline_metrics_list) / len(baseline_metrics_list),
            avg_tpot_ms=sum(m.avg_tpot_ms for m in baseline_metrics_list) / len(baseline_metrics_list),
            ttft_ms=sum(m.ttft_ms for m in baseline_metrics_list) / len(baseline_metrics_list),
            perplexity=sum(m.perplexity for m in baseline_metrics_list) / len(baseline_metrics_list),
        )
        baseline_avg.dollars_per_million_tokens = baseline_avg.mj_per_token / 3600 * self.config.electricity_price_kwh
        results["baseline"] = baseline_avg

        # FEEL run
        print("\n[FEEL] Running with embodiment...")
        feel_metrics_list = []
        for i, prompt in enumerate(prompts):
            _, metrics = self.run_generation(prompt, max_tokens, use_feel=True)
            feel_metrics_list.append(metrics)
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i + 1}/{n_prompts}")

        # Aggregate FEEL
        feel_avg = BusinessMetrics(
            mj_per_token=sum(m.mj_per_token for m in feel_metrics_list) / len(feel_metrics_list),
            avg_tpot_ms=sum(m.avg_tpot_ms for m in feel_metrics_list) / len(feel_metrics_list),
            ttft_ms=sum(m.ttft_ms for m in feel_metrics_list) / len(feel_metrics_list),
            perplexity=sum(m.perplexity for m in feel_metrics_list) / len(feel_metrics_list),
        )
        feel_avg.dollars_per_million_tokens = feel_avg.mj_per_token / 3600 * self.config.electricity_price_kwh
        feel_avg.vs_baseline_energy_pct = (feel_avg.mj_per_token - baseline_avg.mj_per_token) / baseline_avg.mj_per_token * 100 if baseline_avg.mj_per_token > 0 else 0
        feel_avg.vs_baseline_cost_pct = feel_avg.vs_baseline_energy_pct
        results["feel"] = feel_avg

        # Print comparison
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"\n{'Metric':<25} {'Baseline':>15} {'FEEL':>15} {'Δ':>10}")
        print("-" * 65)
        print(f"{'mJ/token':<25} {baseline_avg.mj_per_token:>15.2f} {feel_avg.mj_per_token:>15.2f} {feel_avg.vs_baseline_energy_pct:>+9.1f}%")
        print(f"{'$/1M tokens':<25} ${baseline_avg.dollars_per_million_tokens:>14.4f} ${feel_avg.dollars_per_million_tokens:>14.4f} {feel_avg.vs_baseline_cost_pct:>+9.1f}%")
        print(f"{'Avg TPOT (ms)':<25} {baseline_avg.avg_tpot_ms:>15.2f} {feel_avg.avg_tpot_ms:>15.2f}")
        print(f"{'TTFT (ms)':<25} {baseline_avg.ttft_ms:>15.2f} {feel_avg.ttft_ms:>15.2f}")
        print(f"{'Perplexity':<25} {baseline_avg.perplexity:>15.2f} {feel_avg.perplexity:>15.2f}")

        # Save results
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {results_path}")

        return results

    def run_demo(self):
        """Run business demo."""
        print("\n" + "=" * 60)
        print("FEEL-SLM BUSINESS DEMO")
        print("=" * 60)

        # Single generation with full metrics
        print("\nRunning demo generation...")
        output, metrics = self.run_generation(
            prompt="The future of energy-efficient AI is",
            max_tokens=64,
            use_feel=True,
        )

        print(metrics.summary())

        print("\nGenerated text:")
        print("-" * 40)
        print(output[:200] + "..." if len(output) > 200 else output)

        # Save metrics
        metrics_path = self.output_dir / "demo_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

    def save_trajectory(self):
        """Save trajectory log."""
        if self.trajectory:
            path = self.output_dir / f"trajectory_{int(time.time())}.json"
            with open(path, "w") as f:
                json.dump(self.trajectory, f, indent=2)
            print(f"Saved {len(self.trajectory)} trajectory points to {path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FEEL-SLM Unified Pipeline")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration")
    parser.add_argument("--demo", action="store_true", help="Run business demo")
    parser.add_argument("--benchmark", action="store_true", help="Run A/B benchmark")
    parser.add_argument("--output-dir", type=str, default="results/z117_pipeline")
    parser.add_argument("--calibration-duration", type=int, default=60, help="Calibration duration (s)")
    parser.add_argument("--n-prompts", type=int, default=20, help="Number of prompts for benchmark")
    args = parser.parse_args()

    # HSA override for AMD
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    # Create config
    config = PipelineConfig(
        output_dir=args.output_dir,
    )

    # Create pipeline
    pipeline = UnifiedPipeline(config)

    if args.calibrate:
        pipeline.run_calibration(duration_s=args.calibration_duration)
    elif args.demo:
        pipeline.run_demo()
    elif args.benchmark:
        pipeline.run_benchmark(n_prompts=args.n_prompts)
    else:
        print("Specify --calibrate, --demo, or --benchmark")
        print("\nUsage examples:")
        print("  python z117_unified_pipeline.py --calibrate")
        print("  python z117_unified_pipeline.py --demo")
        print("  python z117_unified_pipeline.py --benchmark --n-prompts 50")

    # Save trajectory
    pipeline.save_trajectory()


if __name__ == "__main__":
    main()
