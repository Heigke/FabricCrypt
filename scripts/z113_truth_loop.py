#!/usr/bin/env python3
"""
Z113: Minimal Truth Loop for FEEL-SLM Phase 1

This script proves causal closure:
  Policy actions → Hardware state → Energy/Latency → Next actions

Output: organism_heartbeat_{timestamp}.json with:
- energy_j (from power integration ΔE = ∫P dt)
- tokens_generated
- j_per_token = energy_j / tokens
- profile_timeline (eco/balanced/perf decisions)
- telemetry_timeline (power/temp/clocks/util)
- latency_timeline (TTFT + TPOT per window)

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z113_truth_loop.py --duration 60

Author: FEEL Research Team
Date: 2026-01-21
"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from src.feel_slm.model_v2 import FEELSLMV2, FEELConfigV2, BaselineSLMV2
from src.feel_slm.telemetry_source import AMDSysfsTelemetry, TelemetrySampler, TelemetryRingBuffer
from src.actuator.client import ActuatorClient


# =============================================================================
# Data Structures for Truth Log
# =============================================================================

@dataclass
class TokenWindow:
    """Metrics for one control window (32 tokens)."""
    window_idx: int
    start_time: float
    end_time: float
    tokens_generated: int

    # Energy (from power integration)
    energy_mj: float
    mj_per_token: float
    avg_power_w: float
    max_power_w: float

    # Latency
    ttft_ms: Optional[float]  # Time to first token (only window 0)
    tpot_ms: float            # Time per output token

    # Policy decision
    profile_selected: str
    strain: float
    margin: float

    # Raw telemetry
    temp_c: float
    gpu_util: float
    clock_mhz: int


@dataclass
class TruthLog:
    """Complete organism heartbeat log."""
    run_id: str
    timestamp: str
    machine: str
    model: str  # "baseline" or "feel"
    config: Dict[str, Any]

    # Totals
    duration_s: float
    total_energy_j: float
    total_tokens: int
    j_per_token: float
    avg_power_w: float

    # SLO compliance
    ttft_p95_ms: float
    tpot_p95_ms: float
    slo_violations: int

    # Timelines
    profile_timeline: List[str]
    windows: List[Dict[str, Any]]

    # Raw telemetry summary
    power_mean_w: float
    power_max_w: float
    temp_mean_c: float
    temp_max_c: float

    def save(self, output_dir: Path):
        """Save to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"organism_heartbeat_{self.run_id}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

        return filepath


# =============================================================================
# Power Integrator (AMD doesn't have cumulative energy counter)
# =============================================================================

class PowerIntegrator:
    """
    Integrate instantaneous power to compute energy.

    ΔE = ∫P(t)dt ≈ Σ (P[i] + P[i-1])/2 * Δt  (trapezoid rule)

    This is the "truth" for AMD since we don't have hardware energy counters.
    """

    def __init__(self, telemetry_source: AMDSysfsTelemetry):
        self.source = telemetry_source
        self.samples: List[tuple] = []  # (timestamp, power_w)
        self._lock = threading.Lock()

    def reset(self):
        """Clear accumulated samples."""
        with self._lock:
            self.samples.clear()

    def sample(self) -> float:
        """Take a power sample, return current power."""
        reading = self.source.read()
        with self._lock:
            self.samples.append((reading.timestamp, reading.power_w))
        return reading.power_w

    def get_energy_j(self) -> float:
        """Compute total energy from power samples (trapezoid integration)."""
        with self._lock:
            if len(self.samples) < 2:
                return 0.0

            total_energy = 0.0
            for i in range(1, len(self.samples)):
                t_prev, p_prev = self.samples[i-1]
                t_curr, p_curr = self.samples[i]
                dt = t_curr - t_prev
                avg_power = (p_prev + p_curr) / 2.0
                total_energy += avg_power * dt

            return total_energy

    def get_window_energy_mj(self, since_time: float) -> tuple:
        """Get energy in window since given time. Returns (energy_mj, avg_power_w, max_power_w)."""
        with self._lock:
            window_samples = [(t, p) for t, p in self.samples if t >= since_time]

            if len(window_samples) < 2:
                if window_samples:
                    return (0.0, window_samples[0][1], window_samples[0][1])
                return (0.0, 0.0, 0.0)

            total_energy = 0.0
            powers = [p for _, p in window_samples]

            for i in range(1, len(window_samples)):
                t_prev, p_prev = window_samples[i-1]
                t_curr, p_curr = window_samples[i]
                dt = t_curr - t_prev
                avg_power = (p_prev + p_curr) / 2.0
                total_energy += avg_power * dt

            return (total_energy * 1000, sum(powers)/len(powers), max(powers))


# =============================================================================
# Truth Loop Runner
# =============================================================================

class TruthLoopRunner:
    """
    Run FEEL-SLM Phase 1 with real telemetry and actuation.

    Proves causal closure: policy → actuation → energy → policy
    """

    def __init__(
        self,
        model: torch.nn.Module,
        is_feel: bool,
        actuator: ActuatorClient,
        telemetry: AMDSysfsTelemetry,
        device: torch.device,
        control_window_tokens: int = 32,
        slo_ttft_ms: float = 500.0,
        slo_tpot_ms: float = 50.0,
    ):
        self.model = model
        self.is_feel = is_feel
        self.actuator = actuator
        self.telemetry = telemetry
        self.device = device
        self.control_window_tokens = control_window_tokens
        self.slo_ttft_ms = slo_ttft_ms
        self.slo_tpot_ms = slo_tpot_ms

        self.integrator = PowerIntegrator(telemetry)
        self.windows: List[TokenWindow] = []
        self.profile_timeline: List[str] = []

        # Telemetry sampling thread
        self._sampling = False
        self._sample_thread = None

        # Track last actuated profile to avoid redundant calls
        self._last_actuated_profile: Optional[str] = None
        self._last_actuation_time: float = 0.0
        self._min_actuation_interval: float = 0.6  # Slightly above 2Hz limit

    def _start_sampling(self, interval: float = 0.02):
        """Start background power sampling at 50Hz."""
        self._sampling = True
        def sample_loop():
            while self._sampling:
                self.integrator.sample()
                time.sleep(interval)
        self._sample_thread = threading.Thread(target=sample_loop, daemon=True)
        self._sample_thread.start()

    def _stop_sampling(self):
        """Stop background sampling."""
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)

    def reset_to_default(self):
        """Reset actuator to default profile."""
        time.sleep(0.6)  # Wait for rate limit
        self._actuate("default", force=True)

    def _get_telemetry_vector(self) -> torch.Tensor:
        """Read current telemetry and convert to model input."""
        reading = self.telemetry.read()

        # 12D normalized vector
        vector = torch.tensor([
            min(1.0, reading.power_w / 100.0),  # AMD APU max ~65W
            min(1.0, reading.temp_c / 100.0),
            reading.gpu_util / 100.0,
            min(1.0, reading.clock_gfx_mhz / 2500.0),
            reading.mem_util / 100.0 if reading.mem_util else 0.5,
            min(1.0, reading.mem_used_mb / 48000.0),
            0.0,  # power_delta (could compute from history)
            0.0,  # temp_delta
            0.0,  # util_delta
            float(reading.throttle_status > 0),
            reading.fan_pct / 100.0 if reading.fan_pct else 0.5,
            0.5,  # profile code (will be set by policy)
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        return vector, reading

    def _actuate(self, profile: str, force: bool = False) -> bool:
        """Send actuation command (with rate limiting awareness)."""
        now = time.time()

        # Skip if same profile and not forced
        if not force and profile == self._last_actuated_profile:
            return True

        # Respect rate limit
        if now - self._last_actuation_time < self._min_actuation_interval:
            return False

        try:
            result = self.actuator.set_profile(profile)
            if result.success:
                self._last_actuated_profile = profile
                self._last_actuation_time = now
            elif "Rate limited" not in result.message:
                print(f"Actuation failed: {result.message}")
            return result.success
        except Exception as e:
            print(f"Actuation error: {e}")
            return False

    def run_generation(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        fixed_profile: Optional[str] = None,  # For baseline
    ) -> Dict[str, Any]:
        """
        Run token generation with closed-loop control.

        OPTIMIZATION: Body encoder only runs at control window boundaries (every 32 tokens),
        not every token. This reduces overhead from ~34% to ~1% while preserving control.

        Returns dict with generation results and metrics.
        """
        self.integrator.reset()
        self.windows.clear()
        self.profile_timeline.clear()

        # Start power sampling
        self._start_sampling()

        input_ids = prompt_tokens.to(self.device)
        generated = input_ids.clone()

        current_profile = fixed_profile or "balanced"
        tokens_in_window = 0
        window_start_time = time.time()
        window_idx = 0
        first_token_time = None

        all_tpots = []
        slo_violations = 0

        # Cache for body outputs (only updated at control boundaries)
        cached_policy_output = None
        cached_reporter_output = None
        latest_raw_reading = None

        try:
            # For baseline fixed mode, don't re-actuate (already set)
            # For FEEL adaptive, start with balanced
            if not fixed_profile:
                self._actuate(current_profile)
            self.profile_timeline.append(current_profile)

            run_start = time.time()

            for token_idx in range(max_new_tokens):
                token_start = time.time()

                # Determine if this is a control boundary (first token or window complete)
                is_control_boundary = (tokens_in_window == 0)

                # Forward pass
                with torch.no_grad():
                    if self.is_feel:
                        if is_control_boundary:
                            # Run body encoder at control boundaries only
                            telemetry_vec, raw_reading = self._get_telemetry_vector()
                            outputs = self.model(generated, telemetry_vec, return_all=True)
                            # Cache body outputs
                            if "policy" in outputs:
                                cached_policy_output = outputs["policy"]
                            if "reporter" in outputs:
                                cached_reporter_output = outputs["reporter"]
                            latest_raw_reading = raw_reading
                        else:
                            # Skip body encoder for intermediate tokens (fast path)
                            outputs = self.model(generated, telemetry=None, return_all=False)
                            raw_reading = self.telemetry.read()
                            latest_raw_reading = raw_reading
                    else:
                        outputs = self.model(generated)
                        raw_reading = self.telemetry.read()
                        latest_raw_reading = raw_reading

                    logits = outputs["logits"][:, -1, :] / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)
                tokens_in_window += 1

                # Record TTFT
                if token_idx == 0:
                    first_token_time = time.time()
                    ttft_ms = (first_token_time - run_start) * 1000
                else:
                    ttft_ms = None

                # Record TPOT
                tpot_ms = (time.time() - token_start) * 1000
                all_tpots.append(tpot_ms)

                # Check SLO
                if ttft_ms and ttft_ms > self.slo_ttft_ms:
                    slo_violations += 1
                if tpot_ms > self.slo_tpot_ms:
                    slo_violations += 1

                # Control window complete?
                if tokens_in_window >= self.control_window_tokens:
                    window_end_time = time.time()

                    # Get energy for this window
                    energy_mj, avg_power, max_power = self.integrator.get_window_energy_mj(window_start_time)

                    # Get policy decision from cached outputs (FEEL only)
                    strain = 0.5
                    margin = 0.5
                    new_profile = current_profile

                    if self.is_feel and cached_policy_output is not None:
                        profile_idx = cached_policy_output["profile_idx"].item()
                        profiles = ["eco", "balanced", "performance"]
                        new_profile = profiles[profile_idx]

                        if cached_reporter_output is not None:
                            strain = cached_reporter_output["strain"].item()
                            margin = cached_reporter_output["margin"].item()

                    # Record window
                    window = TokenWindow(
                        window_idx=window_idx,
                        start_time=window_start_time,
                        end_time=window_end_time,
                        tokens_generated=tokens_in_window,
                        energy_mj=energy_mj,
                        mj_per_token=energy_mj / tokens_in_window if tokens_in_window > 0 else 0,
                        avg_power_w=avg_power,
                        max_power_w=max_power,
                        ttft_ms=ttft_ms if window_idx == 0 else None,
                        tpot_ms=sum(all_tpots[-tokens_in_window:]) / tokens_in_window,
                        profile_selected=new_profile,
                        strain=strain,
                        margin=margin,
                        temp_c=latest_raw_reading.temp_c if latest_raw_reading else 0,
                        gpu_util=latest_raw_reading.gpu_util if latest_raw_reading else 0,
                        clock_mhz=latest_raw_reading.clock_gfx_mhz if latest_raw_reading else 0,
                    )
                    self.windows.append(window)

                    # Actuate if profile changed (FEEL)
                    if new_profile != current_profile and not fixed_profile:
                        self._actuate(new_profile)
                        current_profile = new_profile

                    self.profile_timeline.append(current_profile)

                    # Reset window
                    window_idx += 1
                    tokens_in_window = 0
                    window_start_time = time.time()

            run_end = time.time()

        finally:
            self._stop_sampling()
            # Don't reset here - let the experiment control this

        # Compute totals
        total_energy_j = self.integrator.get_energy_j()
        total_tokens = generated.shape[1] - input_ids.shape[1]
        duration_s = run_end - run_start

        # TPOT p95
        sorted_tpots = sorted(all_tpots)
        p95_idx = int(len(sorted_tpots) * 0.95)
        tpot_p95 = sorted_tpots[p95_idx] if sorted_tpots else 0

        return {
            "generated_ids": generated,
            "total_tokens": total_tokens,
            "total_energy_j": total_energy_j,
            "j_per_token": total_energy_j / total_tokens if total_tokens > 0 else 0,
            "duration_s": duration_s,
            "avg_power_w": total_energy_j / duration_s if duration_s > 0 else 0,
            "ttft_ms": (first_token_time - run_start) * 1000 if first_token_time else 0,
            "tpot_p95_ms": tpot_p95,
            "slo_violations": slo_violations,
            "windows": self.windows,
            "profile_timeline": self.profile_timeline,
        }


# =============================================================================
# Main
# =============================================================================

def create_test_prompts(vocab_size: int, n_prompts: int = 5, seq_len: int = 32) -> List[torch.Tensor]:
    """Create simple test prompts."""
    prompts = []
    for _ in range(n_prompts):
        # Random tokens (in real use, would be tokenized text)
        tokens = torch.randint(100, vocab_size - 100, (1, seq_len))
        prompts.append(tokens)
    return prompts


def run_experiment(
    mode: str,  # "baseline" or "feel"
    duration_s: float,
    output_dir: Path,
    device: torch.device,
    actuator: ActuatorClient,
    telemetry: AMDSysfsTelemetry,
    fixed_profile: Optional[str] = None,
):
    """Run a single experiment (baseline or FEEL)."""

    # Create model config
    config = FEELConfigV2(
        vocab_size=32000,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        intermediate_dim=512,
        max_seq_len=512,
        phase=1,
        enable_film=False,
        enable_gating=False,
        enable_layerdrop=True,
        layerdrop_layers=[1, 2],
    )

    # Create model
    if mode == "baseline":
        model = BaselineSLMV2(config).to(device)
        is_feel = False
    else:
        model = FEELSLMV2(config).to(device)
        is_feel = True
        # Set LayerDrop mode based on fixed_profile
        if fixed_profile == "eco":
            model.set_mode("eco")
            print(f"  → FEEL LayerDrop ACTIVE (dropping layers: {model.drop_layers})")
        else:
            model.set_mode("balanced")
            print(f"  → FEEL LayerDrop INACTIVE")

    model.eval()

    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} experiment")
    print(f"Duration: {duration_s}s, Fixed profile: {fixed_profile or 'adaptive'}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    # Create runner
    runner = TruthLoopRunner(
        model=model,
        is_feel=is_feel,
        actuator=actuator,
        telemetry=telemetry,
        device=device,
    )

    # Generate prompts
    prompts = create_test_prompts(config.vocab_size, n_prompts=20, seq_len=32)

    # Run until duration reached
    all_results = []
    start_time = time.time()
    prompt_idx = 0

    # Set initial profile
    if fixed_profile:
        time.sleep(0.6)  # Rate limit
        runner._actuate(fixed_profile, force=True)
        time.sleep(1.0)  # Let hardware settle

    while time.time() - start_time < duration_s:
        prompt = prompts[prompt_idx % len(prompts)]

        result = runner.run_generation(
            prompt_tokens=prompt,
            max_new_tokens=64,  # Short for faster iteration
            fixed_profile=fixed_profile if mode == "baseline" else None,
        )
        all_results.append(result)
        prompt_idx += 1

        # Progress
        elapsed = time.time() - start_time
        print(f"  [{elapsed:.1f}s] Prompt {prompt_idx}: "
              f"{result['total_tokens']} tokens, "
              f"{result['j_per_token']*1000:.2f} mJ/tok, "
              f"{result['avg_power_w']:.1f}W, "
              f"profile: {result['profile_timeline'][-1]}")

    # Reset at end of experiment
    runner.reset_to_default()

    # Aggregate results
    total_tokens = sum(r["total_tokens"] for r in all_results)
    total_energy = sum(r["total_energy_j"] for r in all_results)
    total_duration = sum(r["duration_s"] for r in all_results)
    all_tpot_p95 = [r["tpot_p95_ms"] for r in all_results]
    all_ttft = [r["ttft_ms"] for r in all_results if r["ttft_ms"]]
    all_violations = sum(r["slo_violations"] for r in all_results)

    # Collect all windows
    all_windows = []
    for r in all_results:
        all_windows.extend([asdict(w) for w in r["windows"]])

    # Power stats from all windows
    all_powers = [w["avg_power_w"] for w in all_windows]
    all_temps = [w["temp_c"] for w in all_windows]

    # Create truth log
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    truth_log = TruthLog(
        run_id=f"{mode}_{fixed_profile or 'adaptive'}_{run_id}",
        timestamp=datetime.now().isoformat(),
        machine="ikaros",
        model=mode,
        config={
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "phase": config.phase,
            "fixed_profile": fixed_profile,
            "control_window_tokens": 32,
            "n_prompts": prompt_idx,
        },
        duration_s=total_duration,
        total_energy_j=total_energy,
        total_tokens=total_tokens,
        j_per_token=total_energy / total_tokens if total_tokens > 0 else 0,
        avg_power_w=total_energy / total_duration if total_duration > 0 else 0,
        ttft_p95_ms=sorted(all_ttft)[int(len(all_ttft)*0.95)] if all_ttft else 0,
        tpot_p95_ms=sorted(all_tpot_p95)[int(len(all_tpot_p95)*0.95)] if all_tpot_p95 else 0,
        slo_violations=all_violations,
        profile_timeline=[r["profile_timeline"][-1] for r in all_results],
        windows=all_windows,
        power_mean_w=sum(all_powers)/len(all_powers) if all_powers else 0,
        power_max_w=max(all_powers) if all_powers else 0,
        temp_mean_c=sum(all_temps)/len(all_temps) if all_temps else 0,
        temp_max_c=max(all_temps) if all_temps else 0,
    )

    # Save
    filepath = truth_log.save(output_dir)
    print(f"\n✅ Saved truth log: {filepath}")

    return truth_log


def main():
    parser = argparse.ArgumentParser(description="FEEL-SLM Truth Loop")
    parser.add_argument("--duration", type=float, default=60, help="Duration per experiment (seconds)")
    parser.add_argument("--output-dir", type=str, default="results/z113_truth_loop", help="Output directory")
    parser.add_argument("--actuator-port", type=int, default=8770, help="Actuator daemon port")
    parser.add_argument("--experiments", type=str, default="all",
                       help="Experiments to run: all, baseline_eco, baseline_perf, feel")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Need GPU for this test.")
        print("Make sure HSA_OVERRIDE_GFX_VERSION=11.0.0 is set")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Connect to actuator
    print(f"\nConnecting to actuator at localhost:{args.actuator_port}...")
    actuator = ActuatorClient("localhost", args.actuator_port, auto_heartbeat=True)

    if not actuator.is_available():
        print("ERROR: Actuator daemon not available.")
        print("Start it with: sudo python src/actuator/privileged_daemon_v2.py --port 8770")
        sys.exit(1)

    state = actuator.get_state()
    print(f"Actuator connected: {state.get('device_name', 'unknown')}")

    # Create telemetry source
    telemetry = AMDSysfsTelemetry(card_index=0)
    caps = telemetry.get_capabilities()
    print(f"Telemetry capabilities: {caps}")

    # Run experiments
    results = {}

    experiments = []
    if args.experiments == "all":
        experiments = [
            ("baseline", "eco"),
            ("baseline", "balanced"),
            ("baseline", "perf"),
            ("feel", None),  # Adaptive
        ]
    elif args.experiments == "full":
        # Full comparison including FEEL with forced eco
        experiments = [
            ("baseline", "eco"),
            ("baseline", "balanced"),
            ("baseline", "perf"),
            ("feel", "eco"),      # FEEL with LayerDrop
            ("feel", "balanced"), # FEEL without LayerDrop
            ("feel", None),       # FEEL adaptive
        ]
    elif args.experiments == "baseline_eco":
        experiments = [("baseline", "eco")]
    elif args.experiments == "baseline_perf":
        experiments = [("baseline", "perf")]
    elif args.experiments == "feel":
        experiments = [("feel", None)]
    elif args.experiments == "feel_eco":
        experiments = [("feel", "eco")]
    elif args.experiments == "feel_balanced":
        experiments = [("feel", "balanced")]
    elif args.experiments == "quick":
        # Quick comparison: baseline vs FEEL with eco
        experiments = [
            ("baseline", "eco"),
            ("feel", "eco"),
        ]
    else:
        experiments = [("baseline", "balanced")]

    for mode, fixed_profile in experiments:
        try:
            result = run_experiment(
                mode=mode,
                duration_s=args.duration,
                output_dir=output_dir,
                device=device,
                actuator=actuator,
                telemetry=telemetry,
                fixed_profile=fixed_profile,
            )
            key = f"{mode}_{fixed_profile or 'adaptive'}"
            results[key] = {
                "j_per_token": result.j_per_token,
                "mj_per_token": result.j_per_token * 1000,
                "avg_power_w": result.avg_power_w,
                "tokens": result.total_tokens,
                "tpot_p95_ms": result.tpot_p95_ms,
                "slo_violations": result.slo_violations,
            }
        except Exception as e:
            print(f"ERROR in {mode}/{fixed_profile}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TRUTH LOOP SUMMARY")
    print("="*60)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  mJ/token: {metrics['mj_per_token']:.2f}")
        print(f"  Avg power: {metrics['avg_power_w']:.1f}W")
        print(f"  Tokens: {metrics['tokens']}")
        print(f"  TPOT p95: {metrics['tpot_p95_ms']:.1f}ms")
        print(f"  SLO violations: {metrics['slo_violations']}")

    # Compute savings
    if "baseline_perf" in results and "baseline_eco" in results:
        eco = results["baseline_eco"]["mj_per_token"]
        perf = results["baseline_perf"]["mj_per_token"]
        savings = (perf - eco) / perf * 100
        print(f"\n📊 ECO vs PERF savings: {savings:.1f}%")

    if "baseline_balanced" in results and "feel_adaptive" in results:
        baseline = results["baseline_balanced"]["mj_per_token"]
        feel = results["feel_adaptive"]["mj_per_token"]
        diff = (baseline - feel) / baseline * 100
        print(f"📊 FEEL vs Baseline(balanced): {diff:+.1f}%")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Summary saved: {summary_path}")

    actuator.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
