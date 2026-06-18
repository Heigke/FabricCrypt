#!/usr/bin/env python3
"""
FEEL Variable Load Test - Bursty Arrival Patterns

This script tests FEEL's adaptive energy control under realistic load patterns:
1. Bursty arrivals (Poisson distribution)
2. Ramp up/down patterns
3. Mixed load scenarios

The goal: Show that FEEL adapts energy consumption to match workload,
saving energy during low load while maintaining SLO during peaks.

Author: FEEL Research Team
Date: 2026-01-21
"""

import os
import sys
import json
import time
import argparse
import statistics
import math
import logging
import random
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

# Add project root
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Request:
    """Single inference request."""
    id: str
    prompt: str
    arrival_time: float
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    completion_tokens: int = 0
    energy_j: float = 0.0
    profile_at_start: str = ""
    slo_met: bool = True


@dataclass
class LoadPhase:
    """A phase in the load pattern."""
    name: str
    duration_s: float
    target_rps: float  # Requests per second
    pattern: str = "constant"  # constant, ramp, burst


@dataclass
class PhaseMetrics:
    """Metrics for a load phase."""
    phase: str
    requests: int
    avg_rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    slo_violation_rate: float
    total_energy_j: float
    avg_energy_per_request_j: float
    avg_power_w: float


# ============================================================================
# Load Generators
# ============================================================================

def generate_poisson_arrivals(duration_s: float, rate: float) -> List[float]:
    """Generate Poisson-distributed arrival times."""
    arrivals = []
    t = 0.0
    while t < duration_s:
        # Inter-arrival time is exponentially distributed
        interval = random.expovariate(rate) if rate > 0 else duration_s
        t += interval
        if t < duration_s:
            arrivals.append(t)
    return arrivals


def generate_constant_arrivals(duration_s: float, rate: float) -> List[float]:
    """Generate constant-rate arrivals."""
    if rate <= 0:
        return []
    interval = 1.0 / rate
    arrivals = []
    t = 0.0
    while t < duration_s:
        arrivals.append(t)
        t += interval
    return arrivals


def generate_ramp_arrivals(duration_s: float, start_rate: float, end_rate: float) -> List[float]:
    """Generate arrivals with ramping rate."""
    arrivals = []
    t = 0.0
    while t < duration_s:
        # Linear interpolation of rate
        progress = t / duration_s
        current_rate = start_rate + progress * (end_rate - start_rate)
        if current_rate > 0:
            interval = 1.0 / current_rate
            arrivals.append(t)
            t += interval
        else:
            t += 0.1
    return arrivals


def generate_burst_arrivals(duration_s: float, avg_rate: float, burst_ratio: float = 5.0) -> List[float]:
    """Generate bursty arrivals with occasional spikes."""
    arrivals = []
    t = 0.0
    in_burst = False
    burst_end = 0.0

    while t < duration_s:
        # Decide if we should burst
        if not in_burst and random.random() < 0.1:  # 10% chance to start burst
            in_burst = True
            burst_end = t + random.uniform(0.5, 2.0)  # Burst duration

        current_rate = avg_rate * burst_ratio if in_burst else avg_rate

        if in_burst and t > burst_end:
            in_burst = False

        if current_rate > 0:
            interval = random.expovariate(current_rate)
            t += interval
            if t < duration_s:
                arrivals.append(t)
        else:
            t += 0.1

    return arrivals


# ============================================================================
# Clients
# ============================================================================

class VLLMClient:
    """Thread-safe vLLM client."""

    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"

    def generate(self, prompt: str, max_tokens: int = 64) -> Tuple[str, int, float, float]:
        """Generate and return (text, tokens, latency_ms, ttft_estimate_ms)."""
        try:
            data = {
                "model": "default",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }

            req = urllib.request.Request(
                f"{self.base_url}/v1/completions",
                data=json.dumps(data).encode(),
                headers={'Content-Type': 'application/json'},
            )

            start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
            latency_ms = (time.perf_counter() - start) * 1000

            text = result['choices'][0]['text'] if result.get('choices') else ""
            usage = result.get('usage', {})
            tokens = usage.get('completion_tokens', len(text.split()))

            # Estimate TTFT as 20% of total time
            ttft_ms = latency_ms * 0.2

            return text, tokens, latency_ms, ttft_ms

        except Exception as e:
            logger.warning(f"Request failed: {e}")
            return "", 0, 0, 0


class ActuatorClient:
    """Thread-safe actuator client."""

    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"
        self._lock = threading.Lock()

    def set_profile(self, profile: str) -> bool:
        with self._lock:
            try:
                data = json.dumps({'profile': profile}).encode()
                req = urllib.request.Request(
                    f"{self.base_url}/profile",
                    data=data,
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    result = json.loads(resp.read().decode())
                return result.get('success', False)
            except:
                return False

    def get_profile(self) -> str:
        try:
            with urllib.request.urlopen(f"{self.base_url}/profiles", timeout=5) as resp:
                result = json.loads(resp.read().decode())
            return result.get('current', 'unknown')
        except:
            return 'unknown'

    def get_telemetry(self) -> Optional[Dict]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/telemetry", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except:
            return None


# ============================================================================
# FEEL Adaptive Controller
# ============================================================================

class AdaptiveController:
    """
    FEEL adaptive controller that adjusts profiles based on load.

    This simulates the FEEL control loop behavior:
    - Monitor request rate and latency
    - Adjust profile based on queue depth and SLO margin
    """

    def __init__(self, actuator: ActuatorClient, target_latency_ms: float = 1000):
        self.actuator = actuator
        self.target_latency = target_latency_ms

        self.current_profile = "balanced"
        self.latency_window = deque(maxlen=20)
        self.rps_window = deque(maxlen=10)

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        """Start adaptive control loop."""
        self._running = True
        self.actuator.set_profile("balanced")
        self.current_profile = "balanced"
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop control loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def report_request(self, latency_ms: float):
        """Report completed request for control decisions."""
        with self._lock:
            self.latency_window.append(latency_ms)
            self.rps_window.append(time.time())

    def _control_loop(self):
        """Background control loop."""
        while self._running:
            try:
                self._control_step()
            except Exception as e:
                logger.warning(f"Control error: {e}")
            time.sleep(0.5)

    def _control_step(self):
        """Single control step."""
        with self._lock:
            if not self.latency_window:
                return

            # Calculate metrics
            avg_latency = statistics.mean(self.latency_window)
            p95_latency = sorted(self.latency_window)[int(len(self.latency_window) * 0.95)]

            # Calculate RPS
            now = time.time()
            recent = [t for t in self.rps_window if now - t < 5.0]
            rps = len(recent) / 5.0 if recent else 0

            # Determine target profile
            slo_margin = 1.0 - (p95_latency / self.target_latency)

            if slo_margin < 0.1:  # Close to SLO, need more power
                target_profile = "performance"
            elif slo_margin > 0.5 and rps < 2:  # Lots of margin, low load
                target_profile = "eco"
            else:
                target_profile = "balanced"

            # Apply if changed
            if target_profile != self.current_profile:
                if self.actuator.set_profile(target_profile):
                    logger.info(f"Profile change: {self.current_profile} → {target_profile} "
                               f"(margin={slo_margin:.2f}, rps={rps:.1f})")
                    self.current_profile = target_profile


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_load_phase(
    vllm: VLLMClient,
    actuator: ActuatorClient,
    controller: Optional[AdaptiveController],
    phase: LoadPhase,
    prompts: List[str],
    max_tokens: int,
    slo_ms: float,
) -> Tuple[List[Request], PhaseMetrics]:
    """Run a single load phase."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Phase: {phase.name} ({phase.target_rps} RPS, {phase.duration_s}s)")
    logger.info(f"{'='*60}")

    # Generate arrivals based on pattern
    if phase.pattern == "constant":
        arrivals = generate_constant_arrivals(phase.duration_s, phase.target_rps)
    elif phase.pattern == "poisson":
        arrivals = generate_poisson_arrivals(phase.duration_s, phase.target_rps)
    elif phase.pattern == "burst":
        arrivals = generate_burst_arrivals(phase.duration_s, phase.target_rps)
    elif phase.pattern == "ramp_up":
        arrivals = generate_ramp_arrivals(phase.duration_s, 0.1, phase.target_rps)
    elif phase.pattern == "ramp_down":
        arrivals = generate_ramp_arrivals(phase.duration_s, phase.target_rps, 0.1)
    else:
        arrivals = generate_constant_arrivals(phase.duration_s, phase.target_rps)

    logger.info(f"  Generated {len(arrivals)} requests")

    requests = []
    results = []
    power_samples = []

    # Create requests
    for i, arrival in enumerate(arrivals):
        req = Request(
            id=f"{phase.name}_{i:04d}",
            prompt=prompts[i % len(prompts)],
            arrival_time=arrival,
        )
        requests.append(req)

    def process_request(req: Request) -> Request:
        """Process a single request."""
        req.profile_at_start = controller.current_profile if controller else "static"
        req.start_time = time.time()

        text, tokens, latency_ms, ttft_ms = vllm.generate(req.prompt, max_tokens)

        req.end_time = time.time()
        req.latency_ms = latency_ms
        req.ttft_ms = ttft_ms
        req.completion_tokens = tokens
        req.slo_met = latency_ms <= slo_ms

        # Report to controller
        if controller:
            controller.report_request(latency_ms)

        return req

    # Run requests with timing
    start_time = time.time()

    # Sample power during phase
    def sample_power():
        while time.time() - start_time < phase.duration_s + 5:
            telemetry = actuator.get_telemetry()
            if telemetry:
                power_samples.append(telemetry.get('power_watts', 0))
            time.sleep(0.1)

    power_thread = threading.Thread(target=sample_power, daemon=True)
    power_thread.start()

    # Execute requests at their arrival times
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for req in requests:
            # Wait until arrival time
            wait_until = start_time + req.arrival_time
            now = time.time()
            if wait_until > now:
                time.sleep(wait_until - now)

            future = executor.submit(process_request, req)
            futures.append(future)

        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.warning(f"Request failed: {e}")

    # Wait for power sampling to finish
    time.sleep(1)

    # Compute metrics
    if not results:
        return results, PhaseMetrics(
            phase=phase.name, requests=0, avg_rps=0,
            latency_p50_ms=0, latency_p95_ms=0, latency_p99_ms=0,
            slo_violation_rate=0, total_energy_j=0,
            avg_energy_per_request_j=0, avg_power_w=0
        )

    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    latencies.sort()

    p50 = latencies[int(len(latencies) * 0.5)] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

    slo_violations = sum(1 for r in results if not r.slo_met)
    avg_power = statistics.mean(power_samples) if power_samples else 0
    total_energy = avg_power * phase.duration_s

    metrics = PhaseMetrics(
        phase=phase.name,
        requests=len(results),
        avg_rps=len(results) / phase.duration_s,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        slo_violation_rate=slo_violations / len(results) if results else 0,
        total_energy_j=total_energy,
        avg_energy_per_request_j=total_energy / len(results) if results else 0,
        avg_power_w=avg_power,
    )

    logger.info(f"  Completed: {len(results)} requests, p95={p95:.0f}ms, SLO={1-metrics.slo_violation_rate:.0%}")

    return results, metrics


# ============================================================================
# Load Patterns
# ============================================================================

def get_standard_load_pattern() -> List[LoadPhase]:
    """Standard variable load pattern."""
    return [
        LoadPhase("warmup", 10, 0.5, "constant"),
        LoadPhase("low_load", 20, 1.0, "poisson"),
        LoadPhase("ramp_up", 15, 3.0, "ramp_up"),
        LoadPhase("high_load", 20, 5.0, "poisson"),
        LoadPhase("burst", 10, 3.0, "burst"),
        LoadPhase("ramp_down", 15, 1.0, "ramp_down"),
        LoadPhase("cooldown", 10, 0.5, "constant"),
    ]


def get_quick_load_pattern() -> List[LoadPhase]:
    """Quick test pattern."""
    return [
        LoadPhase("warmup", 5, 0.5, "constant"),
        LoadPhase("low", 10, 1.0, "poisson"),
        LoadPhase("high", 10, 3.0, "poisson"),
        LoadPhase("burst", 10, 2.0, "burst"),
    ]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='FEEL Variable Load Test')
    parser.add_argument('--vllm-host', default='localhost')
    parser.add_argument('--vllm-port', type=int, default=8000)
    parser.add_argument('--actuator-host', default='192.168.0.38')
    parser.add_argument('--actuator-port', type=int, default=9877)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--slo-ms', type=float, default=2000, help='SLO latency in ms')
    parser.add_argument('--adaptive', action='store_true', help='Enable FEEL adaptive control')
    parser.add_argument('--pattern', choices=['standard', 'quick'], default='quick')
    parser.add_argument('--output-dir', default='results/z96_variable_load')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize clients
    vllm = VLLMClient(args.vllm_host, args.vllm_port)
    actuator = ActuatorClient(args.actuator_host, args.actuator_port)

    # Get load pattern
    if args.pattern == 'standard':
        phases = get_standard_load_pattern()
    else:
        phases = get_quick_load_pattern()

    # Sample prompts
    prompts = [
        "What is machine learning?",
        "Explain quantum computing.",
        "Write a haiku about AI.",
        "What is the capital of Japan?",
        "How does photosynthesis work?",
    ]

    # Initialize controller
    controller = None
    if args.adaptive:
        controller = AdaptiveController(actuator, args.slo_ms)
        controller.start()
        logger.info("FEEL adaptive control enabled")
    else:
        actuator.set_profile('balanced')
        logger.info("Static profile mode (balanced)")

    # Run phases
    all_requests = []
    all_metrics = []

    for phase in phases:
        requests, metrics = run_load_phase(
            vllm, actuator, controller, phase, prompts, args.max_tokens, args.slo_ms
        )
        all_requests.extend(requests)
        all_metrics.append(metrics)

    # Stop controller
    if controller:
        controller.stop()

    # Reset profile
    actuator.set_profile('balanced')

    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    json_path = output_dir / f"variable_load_{timestamp}.json"
    json_results = {
        'config': {
            'adaptive': args.adaptive,
            'slo_ms': args.slo_ms,
            'pattern': args.pattern,
        },
        'phases': [asdict(m) for m in all_metrics],
        'summary': {
            'total_requests': len(all_requests),
            'overall_slo_rate': sum(1 for r in all_requests if r.slo_met) / len(all_requests) if all_requests else 0,
            'total_energy_j': sum(m.total_energy_j for m in all_metrics),
        }
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("VARIABLE LOAD TEST RESULTS")
    print("="*70)
    print(f"Mode: {'FEEL Adaptive' if args.adaptive else 'Static'}")
    print(f"SLO: {args.slo_ms}ms")
    print()

    print(f"{'Phase':<15} {'Req':>6} {'RPS':>6} {'p50':>8} {'p95':>8} {'SLO %':>8} {'Power':>8}")
    print("-" * 70)

    for m in all_metrics:
        slo_pct = (1 - m.slo_violation_rate) * 100
        print(f"{m.phase:<15} {m.requests:>6} {m.avg_rps:>6.1f} {m.latency_p50_ms:>8.0f} "
              f"{m.latency_p95_ms:>8.0f} {slo_pct:>7.1f}% {m.avg_power_w:>7.1f}W")

    total_energy = sum(m.total_energy_j for m in all_metrics)
    print("-" * 70)
    print(f"Total Energy: {total_energy:.1f}J")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
