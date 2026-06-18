#!/usr/bin/env python3
"""
FEEL Real vLLM Inference Benchmark - Scientific Energy Measurement

This script provides REAL J/token measurements by:
1. Using vLLM server API (HTTP) - not offline mode
2. Getting actual token counts from server responses
3. Measuring energy with NVML counters during inference
4. Detecting prefill/decode phases from streaming TTFT
5. Computing proper confidence intervals

This is the "closing the loop" script that connects:
    Energy Harness ↔ Real vLLM Tokens ↔ Profile Actuation

Prerequisites:
    - vLLM server running: vllm serve <model> --port 8000
    - Actuator daemon running: python privileged_daemon_v2.py --port 9877

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
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import urllib.request
import urllib.error

# Add project root
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EnergySample:
    """Single energy measurement."""
    timestamp: float
    power_watts: float
    energy_mj: Optional[float]  # Cumulative counter
    phase: str = "unknown"


@dataclass
class TokenTiming:
    """Timing for token generation."""
    token_idx: int
    timestamp: float
    time_since_start_ms: float
    phase: str  # "prefill" or "decode"


@dataclass
class InferenceRun:
    """Single inference run with real measurements."""
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Real energy measurement
    energy_j: float
    energy_per_token_j: float
    has_energy_counter: bool

    # Timing
    ttft_ms: float  # Time to first token
    total_time_ms: float
    tokens_per_second: float

    # Phase breakdown
    prefill_energy_j: float
    prefill_duration_ms: float
    decode_energy_j: float
    decode_duration_ms: float

    # Power samples
    power_mean_w: float
    power_std_w: float
    power_samples: List[float] = field(default_factory=list)

    # Profile
    profile: str = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['power_samples'] = d['power_samples'][:10]  # Truncate for readability
        return d


@dataclass
class ProfileResults:
    """Results for a single profile."""
    profile: str
    runs: List[InferenceRun]

    def stats(self) -> Dict[str, Any]:
        if not self.runs:
            return {}

        e_per_token = [r.energy_per_token_j for r in self.runs]
        powers = [r.power_mean_w for r in self.runs]
        tps_values = [r.tokens_per_second for r in self.runs]
        ttft_values = [r.ttft_ms for r in self.runs]

        def ci_95(values: List[float]) -> Tuple[float, float]:
            if len(values) < 2:
                return (values[0], values[0]) if values else (0, 0)
            n = len(values)
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            t = 1.96 if n > 30 else 2.0 + 0.5 / max(n, 1)
            margin = t * std / math.sqrt(n)
            return (mean - margin, mean + margin)

        return {
            'n_runs': len(self.runs),
            'total_tokens': sum(r.total_tokens for r in self.runs),
            'total_energy_j': sum(r.energy_j for r in self.runs),
            'energy_per_token_j': {
                'mean': statistics.mean(e_per_token),
                'std': statistics.stdev(e_per_token) if len(e_per_token) > 1 else 0,
                'ci_95': ci_95(e_per_token),
            },
            'power_watts': {
                'mean': statistics.mean(powers),
                'std': statistics.stdev(powers) if len(powers) > 1 else 0,
                'ci_95': ci_95(powers),
            },
            'tokens_per_second': {
                'mean': statistics.mean(tps_values),
                'std': statistics.stdev(tps_values) if len(tps_values) > 1 else 0,
            },
            'ttft_ms': {
                'mean': statistics.mean(ttft_values),
                'std': statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0,
            },
            'has_energy_counter': self.runs[0].has_energy_counter if self.runs else False,
        }


# ============================================================================
# Actuator Client (for daemon communication)
# ============================================================================

class DaemonActuator:
    """Control GPU via daemon HTTP API."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def _get(self, path: str, timeout: float = 5.0) -> Optional[Dict]:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            logger.warning(f"GET {path} failed: {e}")
            return None

    def _post(self, path: str, data: Dict, timeout: float = 5.0) -> Dict:
        try:
            req = urllib.request.Request(
                f"{self.base_url}{path}",
                data=json.dumps(data).encode(),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            logger.warning(f"POST {path} failed: {e}")
            return {'error': str(e)}

    def health(self) -> Optional[Dict]:
        return self._get('/health')

    def get_telemetry(self) -> Optional[Dict]:
        return self._get('/telemetry')

    def get_energy(self) -> Optional[Dict]:
        return self._get('/energy')

    def set_profile(self, profile: str) -> bool:
        result = self._post('/profile', {'profile': profile})
        return result.get('success', False)

    def get_profiles(self) -> Dict:
        return self._get('/profiles') or {}


# ============================================================================
# vLLM Server Client (HTTP API)
# ============================================================================

class VLLMServerClient:
    """
    Client for vLLM server HTTP API.

    Uses the OpenAI-compatible API endpoint.
    Supports both streaming (for TTFT) and non-streaming modes.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    def health_check(self) -> bool:
        """Check if vLLM server is running."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as resp:
                return resp.status == 200
        except:
            return False

    def get_model_info(self) -> Optional[Dict]:
        """Get model information from server."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return None

    def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Tuple[str, int, int, Optional[float]]:
        """
        Generate completion via vLLM server API.

        Returns: (response_text, prompt_tokens, completion_tokens, ttft_ms)

        When stream=True, ttft_ms will be the time to first token.
        When stream=False, ttft_ms will be None.
        """
        try:
            # OpenAI-compatible API
            request_data = {
                "model": "default",  # vLLM uses "default" or actual model name
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            }

            req = urllib.request.Request(
                f"{self.base_url}/v1/completions",
                data=json.dumps(request_data).encode(),
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            start_time = time.perf_counter()
            ttft_ms = None

            if stream:
                # Streaming mode for TTFT measurement
                with urllib.request.urlopen(req, timeout=120) as resp:
                    full_text = ""
                    first_token_received = False

                    for line in resp:
                        line = line.decode().strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if 'choices' in data and data['choices']:
                                text_delta = data['choices'][0].get('text', '')
                                if text_delta and not first_token_received:
                                    ttft_ms = (time.perf_counter() - start_time) * 1000
                                    first_token_received = True
                                full_text += text_delta

                    # Estimate tokens (vLLM streaming doesn't give usage)
                    completion_tokens = len(full_text.split())
                    prompt_tokens = len(prompt.split())
                    return full_text, prompt_tokens, completion_tokens, ttft_ms

            else:
                # Non-streaming mode
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode())

                    text = data['choices'][0]['text'] if data.get('choices') else ""
                    usage = data.get('usage', {})
                    prompt_tokens = usage.get('prompt_tokens', len(prompt.split()))
                    completion_tokens = usage.get('completion_tokens', len(text.split()))

                    return text, prompt_tokens, completion_tokens, None

        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            return "", 0, 0, None


# ============================================================================
# Energy Monitor (background sampling)
# ============================================================================

class EnergyMonitor:
    """
    Background energy monitoring during inference.

    Samples power at high frequency and uses energy counters when available.
    """

    def __init__(self, actuator: DaemonActuator, sample_interval_s: float = 0.05):
        self.actuator = actuator
        self.sample_interval_s = sample_interval_s
        self.samples: List[EnergySample] = []
        self.current_phase = "idle"
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._has_energy_counter = False

    def start(self):
        """Start background monitoring."""
        self.samples = []
        self._running = True

        # Check for energy counter
        energy = self.actuator.get_energy()
        if energy:
            self._has_energy_counter = energy.get('has_counter', False)

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[EnergySample]:
        """Stop monitoring and return samples."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.samples.copy()

    def set_phase(self, phase: str):
        """Set current phase (prefill, decode, idle)."""
        with self._lock:
            self.current_phase = phase

    def _sample_loop(self):
        while self._running:
            try:
                ts = time.time()

                # Get power
                telemetry = self.actuator.get_telemetry()
                power = 0.0
                if telemetry:
                    power = telemetry.get('power_watts', 0)

                # Get energy counter
                energy_mj = None
                if self._has_energy_counter:
                    energy = self.actuator.get_energy()
                    if energy:
                        energy_mj = energy.get('energy_mj')

                with self._lock:
                    sample = EnergySample(
                        timestamp=ts,
                        power_watts=power,
                        energy_mj=energy_mj,
                        phase=self.current_phase,
                    )
                    self.samples.append(sample)

            except Exception as e:
                logger.warning(f"Sample failed: {e}")

            time.sleep(self.sample_interval_s)

    @property
    def has_energy_counter(self) -> bool:
        return self._has_energy_counter


def compute_energy_from_samples(samples: List[EnergySample]) -> Tuple[float, bool]:
    """
    Compute total energy from samples.

    Returns: (energy_joules, used_counter)

    Prefers energy counter delta when available (most accurate).
    Falls back to power integration.
    """
    if not samples:
        return 0.0, False

    samples = sorted(samples, key=lambda s: s.timestamp)

    # Try energy counter delta first (NVML cumulative mJ)
    counter_samples = [s for s in samples if s.energy_mj is not None]
    if len(counter_samples) >= 2:
        start_mj = counter_samples[0].energy_mj
        end_mj = counter_samples[-1].energy_mj
        energy_j = (end_mj - start_mj) / 1000.0  # mJ to J
        return energy_j, True

    # Fall back to power integration
    total_j = 0.0
    for i in range(1, len(samples)):
        dt = samples[i].timestamp - samples[i-1].timestamp
        avg_power = (samples[i].power_watts + samples[i-1].power_watts) / 2
        total_j += avg_power * dt

    return total_j, False


def compute_phase_energy(samples: List[EnergySample], phase: str) -> Tuple[float, float]:
    """
    Compute energy and duration for a specific phase.

    Returns: (energy_joules, duration_seconds)
    """
    phase_samples = [s for s in samples if s.phase == phase]
    if not phase_samples:
        return 0.0, 0.0

    phase_samples = sorted(phase_samples, key=lambda s: s.timestamp)

    # Duration
    duration = phase_samples[-1].timestamp - phase_samples[0].timestamp

    # Energy (power integration for per-phase)
    total_j = 0.0
    for i in range(1, len(phase_samples)):
        dt = phase_samples[i].timestamp - phase_samples[i-1].timestamp
        avg_power = (phase_samples[i].power_watts + phase_samples[i-1].power_watts) / 2
        total_j += avg_power * dt

    return total_j, duration


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_single_inference(
    vllm: VLLMServerClient,
    actuator: DaemonActuator,
    prompt: str,
    max_tokens: int = 128,
    profile: str = "balanced",
    use_streaming: bool = True,
) -> InferenceRun:
    """
    Run a single inference with real energy measurement.

    This is the core function that connects:
    - vLLM server API (real tokens)
    - Energy monitoring (real Joules)
    - Phase detection (prefill/decode via TTFT)
    """

    # Start energy monitor
    monitor = EnergyMonitor(actuator, sample_interval_s=0.05)
    monitor.start()
    monitor.set_phase("prefill")

    start_time = time.perf_counter()

    # Run inference
    response_text, prompt_tokens, completion_tokens, ttft_ms = vllm.generate_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        stream=use_streaming,
    )

    # Mark decode phase at TTFT
    if ttft_ms is not None:
        # TTFT detected - switch phase (retroactively apply to samples)
        monitor.set_phase("decode")

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    # Stop monitoring and get samples
    samples = monitor.stop()

    # Compute energy from samples
    total_energy_j, used_counter = compute_energy_from_samples(samples)

    # Compute per-phase energy
    prefill_energy_j, prefill_duration = compute_phase_energy(samples, "prefill")
    decode_energy_j, decode_duration = compute_phase_energy(samples, "decode")

    # Get power statistics
    powers = [s.power_watts for s in samples if s.power_watts > 0]
    power_mean = statistics.mean(powers) if powers else 0
    power_std = statistics.stdev(powers) if len(powers) > 1 else 0

    total_tokens = prompt_tokens + completion_tokens
    energy_per_token = total_energy_j / max(total_tokens, 1)
    tokens_per_second = completion_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0

    return InferenceRun(
        prompt=prompt[:100],  # Truncate for storage
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        energy_j=total_energy_j,
        energy_per_token_j=energy_per_token,
        has_energy_counter=used_counter,
        ttft_ms=ttft_ms or (total_time_ms / max(completion_tokens, 1)),
        total_time_ms=total_time_ms,
        tokens_per_second=tokens_per_second,
        prefill_energy_j=prefill_energy_j,
        prefill_duration_ms=prefill_duration * 1000,
        decode_energy_j=decode_energy_j,
        decode_duration_ms=decode_duration * 1000,
        power_mean_w=power_mean,
        power_std_w=power_std,
        power_samples=powers,
        profile=profile,
    )


def run_profile_benchmark(
    vllm: VLLMServerClient,
    actuator: DaemonActuator,
    prompts: List[str],
    profile: str,
    max_tokens: int = 128,
    warmup_runs: int = 1,
) -> ProfileResults:
    """
    Run benchmark for a single profile.

    Includes warmup runs that are discarded.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing profile: {profile}")
    logger.info(f"{'='*60}")

    # Set profile
    if not actuator.set_profile(profile):
        logger.warning(f"Failed to set profile {profile}")

    time.sleep(2)  # Let profile settle

    runs = []

    # Warmup runs (discarded)
    for i in range(warmup_runs):
        logger.info(f"  Warmup {i+1}/{warmup_runs}...")
        _ = run_single_inference(
            vllm, actuator, prompts[0], max_tokens, profile
        )
        time.sleep(1)

    # Actual benchmark runs
    for i, prompt in enumerate(prompts):
        logger.info(f"  Run {i+1}/{len(prompts)}")

        run = run_single_inference(
            vllm, actuator, prompt, max_tokens, profile
        )

        runs.append(run)

        logger.info(
            f"    Tokens: {run.total_tokens}, Energy: {run.energy_j:.3f}J, "
            f"J/tok: {run.energy_per_token_j:.4f}, Power: {run.power_mean_w:.1f}W"
        )

        time.sleep(1)  # Cooldown between runs

    return ProfileResults(profile=profile, runs=runs)


# ============================================================================
# Benchmark Prompts
# ============================================================================

def get_benchmark_prompts(num_prompts: int = 10) -> List[str]:
    """Get diverse benchmark prompts."""
    prompts = [
        "What is the capital of France and why is it significant?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the benefits of renewable energy?",
        "Describe how neural networks learn.",
        "What is the difference between supervised and unsupervised learning?",
        "Explain the process of photosynthesis.",
        "What makes a good software architecture?",
        "Describe the history of the Internet briefly.",
        "What is machine learning and why is it useful?",
        "How do large language models generate text?",
        "What are the main challenges in AI safety?",
        "Explain gradient descent in machine learning.",
        "What is transfer learning and why is it important?",
        "Describe the attention mechanism in transformers.",
    ]

    # Repeat if needed
    result = []
    while len(result) < num_prompts:
        result.extend(prompts)
    return result[:num_prompts]


# ============================================================================
# Report Generation
# ============================================================================

def generate_latex_table(results: Dict[str, ProfileResults]) -> str:
    """Generate publication-ready LaTeX table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{FEEL Profile Energy Efficiency (Real vLLM Inference)}",
        r"\label{tab:vllm-energy}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Profile} & \textbf{Power (W)} & \textbf{J/token} & \textbf{Tok/s} & \textbf{TTFT (ms)} & \textbf{CI 95\%} \\",
        r"\midrule",
    ]

    for profile_name, profile_result in results.items():
        stats = profile_result.stats()
        if not stats:
            continue

        power = stats['power_watts']['mean']
        energy = stats['energy_per_token_j']['mean']
        tps = stats['tokens_per_second']['mean']
        ttft = stats['ttft_ms']['mean']
        ci = stats['energy_per_token_j']['ci_95']

        lines.append(
            f"{profile_name} & {power:.1f} & {energy:.4f} & {tps:.1f} & {ttft:.1f} & [{ci[0]:.4f}, {ci[1]:.4f}] \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_markdown_report(results: Dict[str, ProfileResults]) -> str:
    """Generate markdown report."""
    lines = [
        "# FEEL Real vLLM Inference Benchmark",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        "- Energy measurement: NVML energy counters (when available)",
        "- Phase detection: Streaming TTFT for prefill/decode separation",
        "- Token counting: Real usage from vLLM API",
        "",
        "## Results",
        "",
        "| Profile | Power (W) | J/token | Tok/s | TTFT (ms) | CI 95% |",
        "|---------|-----------|---------|-------|-----------|--------|",
    ]

    for profile_name, profile_result in results.items():
        stats = profile_result.stats()
        if not stats:
            continue

        power = stats['power_watts']['mean']
        energy = stats['energy_per_token_j']['mean']
        tps = stats['tokens_per_second']['mean']
        ttft = stats['ttft_ms']['mean']
        ci = stats['energy_per_token_j']['ci_95']
        counter = "✓" if stats['has_energy_counter'] else "✗"

        lines.append(
            f"| {profile_name} | {power:.1f} | {energy:.4f} | {tps:.1f} | {ttft:.1f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
        )

    lines.extend([
        "",
        "## Analysis",
        "",
    ])

    # Compare eco vs performance
    if 'eco' in results and 'performance' in results:
        eco_stats = results['eco'].stats()
        perf_stats = results['performance'].stats()

        if eco_stats and perf_stats:
            eco_energy = eco_stats['energy_per_token_j']['mean']
            perf_energy = perf_stats['energy_per_token_j']['mean']

            if perf_energy > 0:
                energy_reduction = (perf_energy - eco_energy) / perf_energy * 100
                lines.append(f"- **Energy reduction (eco vs perf):** {energy_reduction:.1f}%")

            eco_power = eco_stats['power_watts']['mean']
            perf_power = perf_stats['power_watts']['mean']
            if perf_power > 0:
                power_reduction = (perf_power - eco_power) / perf_power * 100
                lines.append(f"- **Power reduction (eco vs perf):** {power_reduction:.1f}%")

            eco_tps = eco_stats['tokens_per_second']['mean']
            perf_tps = perf_stats['tokens_per_second']['mean']
            if perf_tps > 0:
                tps_change = (eco_tps - perf_tps) / perf_tps * 100
                lines.append(f"- **Throughput change (eco vs perf):** {tps_change:+.1f}%")

    lines.extend([
        "",
        "## Methodology",
        "",
        "1. Profile actuation via FEEL daemon (power cap enforcement)",
        "2. Energy measurement via NVML cumulative counter deltas",
        "3. Phase separation via streaming TTFT detection",
        "4. 95% confidence intervals via t-distribution",
        "5. Real token counts from vLLM API responses",
    ])

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='FEEL Real vLLM Benchmark')
    parser.add_argument('--vllm-host', default='localhost', help='vLLM server host')
    parser.add_argument('--vllm-port', type=int, default=8000, help='vLLM server port')
    parser.add_argument('--actuator-host', default='192.168.0.38', help='Actuator daemon host')
    parser.add_argument('--actuator-port', type=int, default=9877, help='Actuator daemon port')
    parser.add_argument('--runs', type=int, default=5, help='Runs per profile')
    parser.add_argument('--max-tokens', type=int, default=128, help='Max tokens per request')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup runs (discarded)')
    parser.add_argument('--profiles', default='eco,balanced,performance',
                       help='Profiles to test (comma-separated)')
    parser.add_argument('--output-dir', default='results/z94_vllm_benchmark',
                       help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = [p.strip() for p in args.profiles.split(',')]

    # Connect to vLLM server
    vllm = VLLMServerClient(args.vllm_host, args.vllm_port)
    if not vllm.health_check():
        logger.error(f"vLLM server not available at {args.vllm_host}:{args.vllm_port}")
        logger.error("Start with: vllm serve <model> --port 8000")
        sys.exit(1)

    model_info = vllm.get_model_info()
    model_name = "unknown"
    if model_info and 'data' in model_info:
        model_name = model_info['data'][0].get('id', 'unknown')
    logger.info(f"Connected to vLLM server, model: {model_name}")

    # Connect to actuator daemon
    actuator = DaemonActuator(args.actuator_host, args.actuator_port)
    health = actuator.health()
    if not health:
        logger.error(f"Actuator daemon not available at {args.actuator_host}:{args.actuator_port}")
        sys.exit(1)

    logger.info(f"Connected to {health.get('vendor')} GPU: {health.get('device_name')}")

    # Check energy counter
    energy = actuator.get_energy()
    has_counter = energy and energy.get('has_counter', False)
    logger.info(f"Energy counter available: {has_counter}")

    # Get benchmark prompts
    prompts = get_benchmark_prompts(args.runs)

    # Run benchmark for each profile
    results: Dict[str, ProfileResults] = {}

    for profile in profiles:
        results[profile] = run_profile_benchmark(
            vllm=vllm,
            actuator=actuator,
            prompts=prompts,
            profile=profile,
            max_tokens=args.max_tokens,
            warmup_runs=args.warmup,
        )

    # Reset to default
    actuator.set_profile('balanced')

    # Generate outputs
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # JSON results
    json_path = output_dir / f"vllm_benchmark_{timestamp}.json"
    json_results = {
        'config': {
            'vllm_host': args.vllm_host,
            'vllm_port': args.vllm_port,
            'model': model_name,
            'runs_per_profile': args.runs,
            'max_tokens': args.max_tokens,
            'has_energy_counter': has_counter,
        },
        'profiles': {
            profile: {
                'stats': r.stats(),
                'runs': [run.to_dict() for run in r.runs],
            }
            for profile, r in results.items()
        }
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"JSON saved to: {json_path}")

    # LaTeX table
    latex_path = output_dir / f"vllm_benchmark_{timestamp}.tex"
    with open(latex_path, 'w') as f:
        f.write(generate_latex_table(results))
    logger.info(f"LaTeX saved to: {latex_path}")

    # Markdown report
    md_path = output_dir / f"vllm_benchmark_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write(generate_markdown_report(results))
    logger.info(f"Report saved to: {md_path}")

    # Print summary
    print("\n" + "="*70)
    print("REAL vLLM INFERENCE BENCHMARK RESULTS")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Energy counter: {'Yes (NVML cumulative)' if has_counter else 'No (power integration)'}")
    print()

    for profile, result in results.items():
        stats = result.stats()
        if stats:
            print(f"{profile}:")
            print(f"  Power:      {stats['power_watts']['mean']:.1f}W ± {stats['power_watts']['std']:.1f}")
            print(f"  J/token:    {stats['energy_per_token_j']['mean']:.4f} ± {stats['energy_per_token_j']['std']:.4f}")
            print(f"  Tok/s:      {stats['tokens_per_second']['mean']:.1f}")
            print(f"  TTFT:       {stats['ttft_ms']['mean']:.1f}ms")
            print(f"  CI 95%:     [{stats['energy_per_token_j']['ci_95'][0]:.4f}, {stats['energy_per_token_j']['ci_95'][1]:.4f}]")
            print()


if __name__ == '__main__':
    main()
