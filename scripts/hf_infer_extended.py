#!/usr/bin/env python3
"""
Extended HF Inference with Full Research Features

Enhancements over base script:
1. p50/p95 percentile metrics with jitter
2. Concurrency support (multiple parallel requests)
3. Enhanced GPU metrics (clocks, DPM states, throttling)
4. SDPA backend detection and logging
5. Measurement validity checks
6. Extended workload matrix
7. Sustained run support for thermal stability

Usage:
    # Run from experiment manifest
    python scripts/hf_infer_extended.py --manifest experiments/manifest.yaml

    # Single model comparison
    python scripts/hf_infer_extended.py --model Qwen/Qwen2.5-3B-Instruct --compare

    # Sustained thermal test
    python scripts/hf_infer_extended.py --model Qwen/Qwen2.5-0.5B-Instruct --sustained 600
"""

import os
import sys
import time
import json
import argparse
import logging
import statistics
import threading
import queue
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

import torch
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder, AMDSMIMonitor
from src.energy_harness.enhanced_gpu_metrics import (
    EnhancedGPUMetricsCollector,
    DPMStateController,
    EnhancedGPUMetrics,
)
from src.energy_harness.roctx_integration import (
    ROCTxMarker,
    PyTorchProfilerIntegration,
    generate_attribution_report,
    check_roctx_availability,
)
from src.energy_harness.internal_signals import LatentStateController, ZeroOverheadLatentCapture
from src.energy_harness.internal_signals import (
    signals_from_logits,
    DifficultyController,
    DifficultyControllerV2,
    WindowedController,
    SpeculativeDecodeController,
    InternalSignals,
    DifficultyState,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtendedMetrics:
    """Extended inference metrics with percentiles and jitter."""
    # Timing
    prefill_s: float = 0.0
    ttft_s: float = 0.0
    tpot_s_mean: float = 0.0
    tpot_s_p50: float = 0.0
    tpot_s_p95: float = 0.0
    tpot_s_std: float = 0.0  # jitter
    total_s: float = 0.0

    # Tokens
    prompt_tokens: int = 0
    decode_tokens: int = 0

    # Energy
    prefill_energy_j: float = 0.0
    decode_energy_j: float = 0.0
    total_energy_j: float = 0.0

    # Efficiency
    total_tok_per_j: float = 0.0

    # Power
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    power_std_w: float = 0.0

    # GPU state
    avg_sclk_mhz: Optional[float] = None
    avg_mclk_mhz: Optional[float] = None
    avg_gpu_busy_pct: Optional[float] = None
    max_temp_c: Optional[float] = None
    throttle_detected: bool = False

    # Context
    policy: str = "unknown"
    concurrency: int = 1
    batch_size: int = 1
    sdpa_backend: str = "unknown"

    # Validity
    energy_sanity_check: bool = True
    sanity_check_details: str = ""

    # Internal signals (for signal_controller policy)
    entropy_mean: Optional[float] = None
    entropy_max: Optional[float] = None
    margin_mean: Optional[float] = None
    margin_min: Optional[float] = None
    surprisal_mean: Optional[float] = None
    surprisal_max: Optional[float] = None
    difficulty_mean: Optional[float] = None
    policy_switches: int = 0
    time_in_peak_pct: float = 0.0
    time_in_min_pct: float = 0.0
    time_in_auto_pct: float = 0.0

    # Latent controller signals (hidden-state delta norms)
    delta_mean: Optional[float] = None
    delta_max: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentCondition:
    """Single experiment condition."""
    model_id: str
    prompt_type: str  # "short", "medium", "long", "custom"
    prompt_tokens: int
    decode_tokens: int
    batch_size: int = 1
    concurrency: int = 1
    policy: str = "auto"
    n_repeats: int = 10
    warmup_runs: int = 1
    temperature: float = 0.0  # Sampling temperature (0.0 for greedy)


def detect_sdpa_backend() -> str:
    """Detect which SDPA backend is being used."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Check what's available
        available = []
        for backend in [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]:
            try:
                with sdpa_kernel(backend):
                    # Create tiny test tensors
                    q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
                    k = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
                    v = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
                    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    available.append(backend.name)
            except Exception:
                pass

        if available:
            return ",".join(available)
        return "MATH"  # fallback
    except Exception as e:
        logger.warning(f"SDPA detection failed: {e}")
        return "unknown"


def sync():
    """GPU sync for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class EnhancedPowerRecorder:
    """Power recorder with enhanced GPU metrics."""

    def __init__(self, sample_interval_ms: float = 10.0, card_id: int = 1):
        self.base_recorder = PowerTraceRecorder(sample_interval_ms=sample_interval_ms)
        self.metrics_collector = EnhancedGPUMetricsCollector(card_id=card_id)
        self.enhanced_samples: List[EnhancedGPUMetrics] = []
        self._sampling = False
        self._sample_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.sample_interval = sample_interval_ms / 1000.0

    def start(self, metadata: Optional[Dict[str, Any]] = None):
        """Start recording."""
        self.enhanced_samples = []
        self._stop_event.clear()
        self._sampling = True
        self.base_recorder.start(metadata)

        # Start enhanced sampling in parallel
        self._sample_thread = threading.Thread(target=self._enhanced_sample_loop, daemon=True)
        self._sample_thread.start()

    def _enhanced_sample_loop(self):
        """Background thread for enhanced metrics."""
        while not self._stop_event.is_set():
            try:
                metrics = self.metrics_collector.collect()
                self.enhanced_samples.append(metrics)
            except Exception as e:
                logger.debug(f"Enhanced sample failed: {e}")
            time.sleep(self.sample_interval)

    def stop(self):
        """Stop recording."""
        self._stop_event.set()
        if self._sample_thread:
            self._sample_thread.join(timeout=1.0)
        self._sampling = False
        return self.base_recorder.stop()

    def mark_phase(self, phase: str, config: Optional[Dict] = None):
        """Mark phase transition."""
        self.base_recorder.mark_phase(phase, config=config)

    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get summary with enhanced metrics."""
        summary = {}

        if self.enhanced_samples:
            # Average clocks
            sclks = [s.sclk_mhz for s in self.enhanced_samples if s.sclk_mhz]
            mclks = [s.mclk_mhz for s in self.enhanced_samples if s.mclk_mhz]
            gpu_busy = [s.gpu_busy_percent for s in self.enhanced_samples if s.gpu_busy_percent is not None]
            temps = [s.temp_edge_c for s in self.enhanced_samples if s.temp_edge_c]
            throttles = [s.throttle_status for s in self.enhanced_samples]

            if sclks:
                summary["avg_sclk_mhz"] = statistics.mean(sclks)
            if mclks:
                summary["avg_mclk_mhz"] = statistics.mean(mclks)
            if gpu_busy:
                summary["avg_gpu_busy_pct"] = statistics.mean(gpu_busy)
            if temps:
                summary["max_temp_c"] = max(temps)
            if throttles:
                summary["throttle_detected"] = any(t > 0 for t in throttles)

        return summary


def validate_energy_measurement(
    total_energy_j: float,
    avg_power_w: float,
    duration_s: float,
    tolerance: float = 0.15
) -> Tuple[bool, str]:
    """
    Validate energy measurement consistency.

    Checks: integrated energy ≈ mean_power × duration
    """
    expected_energy = avg_power_w * duration_s
    ratio = total_energy_j / expected_energy if expected_energy > 0 else 0

    if 1 - tolerance <= ratio <= 1 + tolerance:
        return True, f"OK: ratio={ratio:.3f}"
    else:
        return False, f"WARN: integrated={total_energy_j:.1f}J, expected={expected_energy:.1f}J, ratio={ratio:.3f}"


def estimate_model_params(model_id: str) -> float:
    """Estimate model parameters from model ID."""
    model_lower = model_id.lower()
    if "0.5b" in model_lower:
        return 0.5e9
    elif "1.5b" in model_lower:
        return 1.5e9
    elif "3b" in model_lower:
        return 3e9
    elif "7b" in model_lower:
        return 7e9
    elif "14b" in model_lower:
        return 14e9
    elif "72b" in model_lower:
        return 72e9
    else:
        # Default to medium size
        return 3e9


def should_use_active_control(
    model_params: float,
    prompt_len: int,
    concurrency: int,
    temperature_c: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Regime detector: decide whether to use active DPM control or defer to auto.

    Based on empirical observations:
    - Small models (< 1B params): auto DVFS is sufficient
    - Large models (> 2B params): active control beneficial
    - High concurrency: active control beneficial
    - Long prompts: active control beneficial (prefill dominates)
    - High temperature: active control helps prevent throttling

    Returns:
        (should_control: bool, reason: str)
    """
    score = 0
    reasons = []

    # Model size factor
    if model_params > 2e9:
        score += 2
        reasons.append(f"large_model({model_params/1e9:.1f}B)")
    elif model_params > 1e9:
        score += 1
        reasons.append(f"medium_model({model_params/1e9:.1f}B)")

    # Concurrency factor
    if concurrency >= 4:
        score += 2
        reasons.append(f"high_concurrency({concurrency})")
    elif concurrency >= 2:
        score += 1
        reasons.append(f"moderate_concurrency({concurrency})")

    # Prompt length factor
    if prompt_len >= 2048:
        score += 2
        reasons.append(f"long_prompt({prompt_len})")
    elif prompt_len >= 512:
        score += 1
        reasons.append(f"medium_prompt({prompt_len})")

    # Temperature factor (if available)
    if temperature_c is not None:
        if temperature_c >= 80:
            score += 2
            reasons.append(f"high_temp({temperature_c}C)")
        elif temperature_c >= 70:
            score += 1
            reasons.append(f"warm_temp({temperature_c}C)")

    threshold = 2  # Threshold for active control
    should_control = score >= threshold

    reason = f"score={score}/{threshold}, factors=[{', '.join(reasons) if reasons else 'none'}]"
    return should_control, reason


@dataclass
class ConcurrentBatchMetrics:
    """Metrics for a concurrent batch of requests."""
    n_requests: int = 0
    total_latency_s: float = 0.0  # Wall-clock time for entire batch
    request_latencies: List[float] = field(default_factory=list)
    throughput_req_per_s: float = 0.0
    throughput_tok_per_s: float = 0.0
    total_energy_j: float = 0.0
    energy_per_request_j: float = 0.0
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    total_tokens: int = 0
    tok_per_j: float = 0.0

    # Per-request aggregates
    ttft_mean_s: float = 0.0
    ttft_p95_s: float = 0.0
    tpot_mean_s: float = 0.0
    tpot_p95_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_concurrent_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    recorder: Optional[EnhancedPowerRecorder] = None,
    dpm_controller: Optional[DPMStateController] = None,
    policy: str = "auto",
    max_workers: int = 4,
) -> Tuple[ConcurrentBatchMetrics, List[ExtendedMetrics]]:
    """
    Run multiple inference requests concurrently.

    Uses ThreadPoolExecutor to simulate concurrent request load.
    All requests share the same model but run in parallel threads.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompts: List of prompts to process concurrently
        max_new_tokens: Max tokens to generate per request
        recorder: Power recorder (records entire batch)
        dpm_controller: DPM state controller
        policy: DVFS policy to use
        max_workers: Max concurrent threads

    Returns:
        (ConcurrentBatchMetrics, list of per-request ExtendedMetrics)
    """
    n_requests = len(prompts)
    per_request_metrics: List[ExtendedMetrics] = []
    request_lock = threading.Lock()

    def run_single_request(idx: int, prompt: str) -> ExtendedMetrics:
        """Worker function for single request."""
        metrics, _, _ = run_single_inference(
            model, tokenizer, prompt, max_new_tokens,
            recorder=None,  # Don't record per-request (batch recorder handles)
            dpm_controller=None,  # Policy set at batch level
            policy=policy,
        )
        metrics.concurrency = n_requests
        return metrics

    # Apply policy at batch level
    if dpm_controller and policy != "auto":
        policy_map = {
            "peak": "profile_peak",
            "min_sclk": "profile_min_sclk",
            "controller": "auto",  # Controller manages dynamically
        }
        dpm_controller.set_perf_level(policy_map.get(policy, "auto"))

    # Start batch recording
    if recorder:
        recorder.start(metadata={"n_requests": n_requests, "policy": policy})
        recorder.mark_phase("concurrent_batch")

    sync()
    batch_start = time.perf_counter()

    # Run requests concurrently
    with ThreadPoolExecutor(max_workers=min(max_workers, n_requests)) as executor:
        futures = {
            executor.submit(run_single_request, i, prompt): i
            for i, prompt in enumerate(prompts)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                metrics = future.result()
                with request_lock:
                    per_request_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Request {idx} failed: {e}")

    sync()
    batch_end = time.perf_counter()
    batch_latency = batch_end - batch_start

    # Stop recording
    if recorder:
        recorder.mark_phase("idle")
        measurement = recorder.stop()
        enhanced = recorder.get_enhanced_summary()
    else:
        measurement = None
        enhanced = {}

    # Aggregate metrics
    batch_metrics = ConcurrentBatchMetrics(
        n_requests=n_requests,
        total_latency_s=batch_latency,
        request_latencies=[m.total_s for m in per_request_metrics],
    )

    if per_request_metrics:
        batch_metrics.total_tokens = sum(m.prompt_tokens + m.decode_tokens for m in per_request_metrics)
        batch_metrics.throughput_req_per_s = n_requests / batch_latency
        batch_metrics.throughput_tok_per_s = batch_metrics.total_tokens / batch_latency

        # TTFT/TPOT aggregates
        ttfts = [m.ttft_s for m in per_request_metrics]
        tpots = [m.tpot_s_mean for m in per_request_metrics]
        batch_metrics.ttft_mean_s = statistics.mean(ttfts)
        batch_metrics.ttft_p95_s = np.percentile(ttfts, 95) if len(ttfts) >= 2 else ttfts[-1]
        batch_metrics.tpot_mean_s = statistics.mean(tpots)
        batch_metrics.tpot_p95_s = np.percentile(tpots, 95) if len(tpots) >= 2 else tpots[-1]

    if measurement:
        batch_metrics.total_energy_j = measurement.energy_joules
        batch_metrics.energy_per_request_j = measurement.energy_joules / n_requests
        batch_metrics.avg_power_w = measurement.avg_power_watts
        batch_metrics.peak_power_w = measurement.peak_power_watts
        if batch_metrics.total_energy_j > 0:
            batch_metrics.tok_per_j = batch_metrics.total_tokens / batch_metrics.total_energy_j

    return batch_metrics, per_request_metrics


def run_concurrent_experiment(
    model,
    tokenizer,
    condition: 'ExperimentCondition',
    output_dir: Path,
    dpm_controller: Optional[DPMStateController] = None,
) -> List[ConcurrentBatchMetrics]:
    """Run concurrent request experiments."""
    results = []

    # Generate prompts for the batch
    prompts = [
        generate_prompt(condition.prompt_type, condition.prompt_tokens)
        for _ in range(condition.concurrency)
    ]

    logger.info(f"Concurrent experiment: {condition.concurrency} requests, "
                f"policy={condition.policy}, prompt={condition.prompt_tokens}tok")

    # Warmup
    logger.info("  Warmup run...")
    run_concurrent_inference(
        model, tokenizer, prompts[:min(2, len(prompts))],
        min(16, condition.decode_tokens),
        policy=condition.policy
    )

    # Main runs
    for rep in range(condition.n_repeats):
        recorder = EnhancedPowerRecorder(sample_interval_ms=10)

        batch_metrics, per_req = run_concurrent_inference(
            model, tokenizer, prompts,
            condition.decode_tokens,
            recorder=recorder,
            dpm_controller=dpm_controller,
            policy=condition.policy,
            max_workers=condition.concurrency,
        )

        results.append(batch_metrics)

        logger.info(f"  Rep {rep+1}/{condition.n_repeats}: "
                    f"batch={batch_metrics.total_latency_s:.3f}s, "
                    f"{batch_metrics.throughput_tok_per_s:.1f}tok/s, "
                    f"{batch_metrics.total_energy_j:.1f}J, "
                    f"{batch_metrics.tok_per_j:.2f}tok/J")

    return results


@torch.inference_mode()
def run_single_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    recorder: Optional[EnhancedPowerRecorder] = None,
    dpm_controller: Optional[DPMStateController] = None,
    policy: str = "auto",
    roctx_marker: Optional[ROCTxMarker] = None,
    pytorch_profiler: Optional[PyTorchProfilerIntegration] = None,
    difficulty_controller = None,  # DifficultyController or DifficultyControllerV2
    windowed_controller = None,  # WindowedController for chunk-level control
    latent_controller: Optional[LatentStateController] = None,  # LatentStateController for hidden-state-driven control
    latent_capture: Optional[ZeroOverheadLatentCapture] = None,  # Zero-overhead hook-based capture
) -> Tuple[ExtendedMetrics, List[float], Optional[List[dict]]]:
    """
    Run single inference with full metrics collection.

    Returns:
        (ExtendedMetrics, list of per-token decode times, signal trace data)
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]

    # Initialize difficulty controller if using signal_controller or signal_controller_v2
    use_signal_control = policy in ("signal_controller", "signal_controller_v2") and difficulty_controller is not None
    if use_signal_control:
        difficulty_controller.reset()

    # Initialize windowed controller if using windowed policy
    use_windowed_control = policy == "windowed" and windowed_controller is not None
    if use_windowed_control:
        windowed_controller.reset()

    # Initialize latent controller if using latent policy
    # "latent" = old method with output_hidden_states=True (16% overhead)
    # "latent_v2" = new method with zero-overhead hook capture
    use_latent_control = policy == "latent" and latent_controller is not None
    use_latent_v2 = policy == "latent_v2" and latent_controller is not None and latent_capture is not None
    if use_latent_control or use_latent_v2:
        latent_controller.reset()
    if use_latent_v2:
        latent_capture.register_hook()

    # Apply policy
    if dpm_controller and policy not in ("auto", "signal_controller", "signal_controller_v2", "windowed", "latent", "latent_v2"):
        policy_map = {
            "peak": "profile_peak",
            "min_sclk": "profile_min_sclk",
            "phase_split_prefill": "profile_peak",
        }
        dpm_controller.set_perf_level(policy_map.get(policy, "auto"))

    # === PREFILL ===
    if recorder:
        recorder.mark_phase("prefill", config={"policy": policy})
    if roctx_marker:
        roctx_marker.mark("prefill_start")

    sync()
    t_prefill_start = time.perf_counter()

    # ROCTx range for prefill
    # Enable hidden states output ONLY for old latent policy (not latent_v2 which uses hooks)
    output_hidden = use_latent_control  # Only True for "latent", False for "latent_v2"
    if roctx_marker:
        with roctx_marker.range("prefill"):
            outputs = model(**inputs, use_cache=True, output_hidden_states=output_hidden)
    elif pytorch_profiler:
        with pytorch_profiler.profile_phase("prefill"):
            outputs = model(**inputs, use_cache=True, output_hidden_states=output_hidden)
    else:
        outputs = model(**inputs, use_cache=True, output_hidden_states=output_hidden)
    past_key_values = outputs.past_key_values

    # Initialize latent controller with prefill hidden state
    if use_latent_control and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        prefill_hidden = outputs.hidden_states[-1]  # Last layer hidden states
        latent_controller.compute_state_delta(prefill_hidden)  # Initialize prev_hidden_state
    elif use_latent_v2:
        # Use zero-overhead hook capture
        prefill_hidden = latent_capture.get_last_hidden()  # [1, hidden_dim] from hook
        if prefill_hidden is not None:
            latent_controller.compute_state_delta(prefill_hidden.unsqueeze(1))  # [1, 1, hidden_dim]

    sync()
    t_prefill_end = time.perf_counter()
    prefill_s = t_prefill_end - t_prefill_start

    # === DECODE ===
    if policy == "phase_split" and dpm_controller:
        dpm_controller.set_perf_level("profile_min_sclk")

    if recorder:
        recorder.mark_phase("decode", config={"policy": policy})

    next_input_ids = inputs["input_ids"][:, -1:].contiguous()
    decode_times = []

    # Start decode phase ROCTx range
    if roctx_marker:
        roctx_marker.mark("decode_start")

    for i in range(max_new_tokens):
        sync()
        t_tok_start = time.perf_counter()

        # ROCTx range for individual decode step
        if roctx_marker:
            with roctx_marker.range(f"decode_step_{i}"):
                outputs = model(
                    input_ids=next_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=output_hidden
                )
        elif pytorch_profiler and i == 0:
            # Profile first decode step only
            with pytorch_profiler.profile_phase("decode_first"):
                outputs = model(
                    input_ids=next_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=output_hidden
                )
        else:
            outputs = model(
                input_ids=next_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        next_input_ids = next_token

        sync()
        t_tok_end = time.perf_counter()
        tpot_s = t_tok_end - t_tok_start
        decode_times.append(tpot_s)

        # === INTERNAL SIGNAL EXTRACTION & DIFFICULTY-CONDITIONED DVFS ===
        if use_signal_control and dpm_controller:
            # Extract signals from logits
            signals = signals_from_logits(
                logits.squeeze(0),  # Remove batch dim
                chosen_id=next_token.item(),
                topk=64,
                token_idx=i
            )

            # Update difficulty controller and get policy recommendation
            diff_state = difficulty_controller.update(signals)

            # Apply difficulty-conditioned DVFS
            policy_map = {
                "peak": "profile_peak",
                "min_sclk": "profile_min_sclk",
                "auto": "auto",
            }
            recommended_level = policy_map.get(diff_state.policy_recommendation, "auto")
            dpm_controller.set_perf_level(recommended_level)

        # === WINDOWED CONTROLLER (chunk-level DVFS every N tokens) ===
        elif use_windowed_control and dpm_controller:
            # Check if we're at a window boundary (every 32 tokens)
            window_boundary = (i + 1) % windowed_controller.window_tokens == 0

            if window_boundary:
                # Only extract full signals at window boundaries (reduces overhead)
                signals = signals_from_logits(
                    logits.squeeze(0),
                    chosen_id=next_token.item(),
                    topk=64,
                    token_idx=i
                )

                # Add token to window - controller evaluates at boundary
                window_evaluated, recommended_policy = windowed_controller.add_token(
                    signals, tpot_ms=tpot_s * 1000
                )

                # Apply DVFS decision at window boundary
                if window_evaluated:
                    policy_map = {
                        "peak": "profile_peak",
                        "min_sclk": "profile_min_sclk",
                        "auto": "auto",
                    }
                    recommended_level = policy_map.get(recommended_policy, "auto")
                    dpm_controller.set_perf_level(recommended_level)
            else:
                # Lightweight: compute top-2 logit gap (margin proxy) without full softmax
                # This is cheaper than full entropy: only O(1) topk vs O(V) softmax
                logit_vec = logits.squeeze(0)
                top2 = torch.topk(logit_vec, k=2, dim=-1)
                margin = (top2.values[0] - top2.values[1]).item()
                # Approximate entropy from top-2 only (faster than full vocab)
                top2_probs = torch.softmax(top2.values, dim=-1)
                entropy_approx = -torch.sum(top2_probs * torch.log(top2_probs + 1e-10)).item()

                # Create minimal signal for window tracking (skip full signal extraction)
                signals = InternalSignals(
                    entropy=entropy_approx,
                    margin=margin,
                    surprisal=0.0,  # Skip expensive surprisal calc
                    p_top1=top2_probs[0].item(),
                    p_top2=top2_probs[1].item(),
                    token_idx=i
                )
                windowed_controller.add_token(signals, tpot_ms=tpot_s * 1000)

        # === LATENT CONTROLLER (hidden-state-driven DVFS) - OLD METHOD ===
        elif use_latent_control and dpm_controller:
            # Get hidden state from model outputs (last layer, last token)
            hidden_state = None
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_state = outputs.hidden_states[-1][:, -1, :]  # [1, hidden_dim]

            # Update latent controller with hidden state and logits
            at_boundary, recommended_policy = latent_controller.update(
                hidden_state=hidden_state,
                logits=logits,
                token_idx=i
            )

            # Apply DVFS at window boundaries
            if at_boundary:
                policy_map = {
                    "peak": "profile_peak",
                    "min_sclk": "profile_min_sclk",
                    "auto": "auto",
                }
                recommended_level = policy_map.get(recommended_policy, "auto")
                dpm_controller.set_perf_level(recommended_level)

        # === LATENT_V2 CONTROLLER (zero-overhead hook capture) ===
        elif use_latent_v2 and dpm_controller:
            # Get hidden state from zero-overhead hook (already captured during forward)
            hidden_state = latent_capture.get_last_hidden()  # [1, hidden_dim] from hook

            # Update latent controller with hidden state and logits
            at_boundary, recommended_policy = latent_controller.update(
                hidden_state=hidden_state.unsqueeze(1) if hidden_state is not None else None,
                logits=logits,
                token_idx=i
            )

            # Apply DVFS at window boundaries
            if at_boundary:
                policy_map = {
                    "peak": "profile_peak",
                    "min_sclk": "profile_min_sclk",
                    "auto": "auto",
                }
                recommended_level = policy_map.get(recommended_policy, "auto")
                dpm_controller.set_perf_level(recommended_level)

        if next_token.item() == tokenizer.eos_token_id:
            break

    if recorder:
        recorder.mark_phase("idle")

    # Cleanup latent_v2 hook
    if use_latent_v2 and latent_capture:
        latent_capture.remove_hook()

    # Reset DVFS to auto at end of generation
    if (use_signal_control or use_windowed_control or use_latent_control or use_latent_v2) and dpm_controller:
        dpm_controller.set_perf_level("auto")

    # Calculate metrics
    total_s = time.perf_counter() - t_prefill_start
    decode_tokens = len(decode_times)

    metrics = ExtendedMetrics(
        prefill_s=prefill_s,
        ttft_s=decode_times[0] if decode_times else 0.0,
        tpot_s_mean=statistics.mean(decode_times) if decode_times else 0.0,
        tpot_s_p50=statistics.median(decode_times) if decode_times else 0.0,
        tpot_s_p95=np.percentile(decode_times, 95) if len(decode_times) >= 2 else 0.0,
        tpot_s_std=statistics.stdev(decode_times) if len(decode_times) > 1 else 0.0,
        total_s=total_s,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        policy=policy,
    )

    # Add signal metrics if using signal controller
    signal_trace = None
    if use_signal_control and difficulty_controller:
        summary = difficulty_controller.get_summary()
        metrics.entropy_mean = summary.get("entropy_mean")
        metrics.entropy_max = summary.get("entropy_max")
        metrics.margin_mean = summary.get("margin_mean")
        metrics.margin_min = summary.get("margin_min")
        metrics.surprisal_mean = summary.get("surprisal_mean")
        metrics.surprisal_max = summary.get("surprisal_max")
        metrics.difficulty_mean = summary.get("difficulty_mean")
        metrics.policy_switches = summary.get("policy_switches", 0)
        metrics.time_in_peak_pct = summary.get("time_in_peak_pct", 0.0)
        metrics.time_in_min_pct = summary.get("time_in_min_pct", 0.0)
        metrics.time_in_auto_pct = summary.get("time_in_auto_pct", 0.0)
        signal_trace = difficulty_controller.get_trace_data()

    # Add signal metrics if using windowed controller
    elif use_windowed_control and windowed_controller:
        summary = windowed_controller.get_summary()
        metrics.entropy_mean = summary.get("entropy_mean")
        metrics.policy_switches = summary.get("policy_switches", 0)
        metrics.time_in_peak_pct = summary.get("time_in_peak_pct", 0.0)
        metrics.time_in_min_pct = summary.get("time_in_min_pct", 0.0)
        metrics.time_in_auto_pct = summary.get("time_in_auto_pct", 0.0)
        # Convert window trace to signal trace format
        signal_trace = windowed_controller.get_window_trace()

    # Add signal metrics if using latent controller (v1 or v2)
    elif (use_latent_control or use_latent_v2) and latent_controller:
        summary = latent_controller.get_summary()
        metrics.entropy_mean = summary.get("margin_mean")  # Using margin as proxy
        metrics.margin_mean = summary.get("margin_mean")
        metrics.policy_switches = summary.get("policy_switches", 0)
        metrics.time_in_peak_pct = summary.get("time_in_peak_pct", 0.0)
        metrics.time_in_min_pct = summary.get("time_in_min_pct", 0.0)
        metrics.time_in_auto_pct = summary.get("time_in_auto_pct", 0.0)
        # Add latent-specific metrics
        metrics.delta_mean = summary.get("delta_mean")
        metrics.delta_max = summary.get("delta_max")
        signal_trace = latent_controller.get_trace_data()

    return metrics, decode_times, signal_trace


def run_experiment_condition(
    model,
    tokenizer,
    condition: ExperimentCondition,
    output_dir: Path,
    dpm_controller: Optional[DPMStateController] = None,
) -> List[ExtendedMetrics]:
    """Run all repeats for a single condition."""

    results = []
    prompt = generate_prompt(condition.prompt_type, condition.prompt_tokens)

    logger.info(f"Condition: {condition.policy}, prompt={condition.prompt_tokens}tok, "
                f"decode={condition.decode_tokens}tok, concurrency={condition.concurrency}")

    # Create difficulty controller for signal_controller policies
    difficulty_controller = None
    windowed_controller = None
    if condition.policy == "signal_controller":
        difficulty_controller = DifficultyController()
    elif condition.policy == "signal_controller_v2":
        difficulty_controller = DifficultyControllerV2()
    elif condition.policy == "windowed":
        # WindowedController: evaluates every 32 tokens with hysteresis
        windowed_controller = WindowedController(
            window_tokens=32,
            window_ms=250.0,
            entropy_hi_threshold=1.0,
            entropy_lo_threshold=0.2,
            min_windows_before_switch=2,
        )

    # Create latent state controller for latent policy (v1 or v2)
    latent_controller = None
    latent_capture = None
    if condition.policy in ("latent", "latent_v2"):
        latent_controller = LatentStateController(
            window_tokens=64,
            high_delta_threshold=0.15,
            low_delta_threshold=0.05,
            high_margin_threshold=0.8,
            low_margin_threshold=0.3,
            min_dwell_tokens=32,  # Minimum tokens before policy switch
        )
    if condition.policy == "latent_v2":
        # Zero-overhead hook-based capture (avoids output_hidden_states overhead)
        latent_capture = ZeroOverheadLatentCapture(model)

    # Warmup
    for _ in range(condition.warmup_runs):
        run_single_inference(model, tokenizer, prompt, min(16, condition.decode_tokens))

    # Main runs
    for rep in range(condition.n_repeats):
        recorder = EnhancedPowerRecorder(sample_interval_ms=10)
        recorder.start(metadata={
            "condition": asdict(condition),
            "repeat": rep,
        })

        metrics, decode_times, signal_trace = run_single_inference(
            model, tokenizer, prompt, condition.decode_tokens,
            recorder=recorder,
            dpm_controller=dpm_controller,
            policy=condition.policy,
            difficulty_controller=difficulty_controller,
            windowed_controller=windowed_controller,
            latent_controller=latent_controller,
            latent_capture=latent_capture,
        )

        measurement = recorder.stop()
        enhanced = recorder.get_enhanced_summary()

        # Add energy metrics
        metrics.prefill_energy_j = recorder.base_recorder.get_phase_energy("prefill")
        metrics.decode_energy_j = recorder.base_recorder.get_phase_energy("decode")
        metrics.total_energy_j = measurement.energy_joules
        metrics.avg_power_w = measurement.avg_power_watts
        metrics.peak_power_w = measurement.peak_power_watts

        # Power std (jitter)
        if measurement.samples:
            powers = [s.power_watts for s in measurement.samples]
            metrics.power_std_w = statistics.stdev(powers) if len(powers) > 1 else 0.0

        # Efficiency
        total_tokens = metrics.prompt_tokens + metrics.decode_tokens
        if metrics.total_energy_j > 0:
            metrics.total_tok_per_j = total_tokens / metrics.total_energy_j

        # Enhanced metrics
        metrics.avg_sclk_mhz = enhanced.get("avg_sclk_mhz")
        metrics.avg_mclk_mhz = enhanced.get("avg_mclk_mhz")
        metrics.avg_gpu_busy_pct = enhanced.get("avg_gpu_busy_pct")
        metrics.max_temp_c = enhanced.get("max_temp_c")
        metrics.throttle_detected = enhanced.get("throttle_detected", False)

        # Validation
        valid, msg = validate_energy_measurement(
            metrics.total_energy_j, metrics.avg_power_w, metrics.total_s
        )
        metrics.energy_sanity_check = valid
        metrics.sanity_check_details = msg

        metrics.concurrency = condition.concurrency
        metrics.sdpa_backend = detect_sdpa_backend()

        results.append(metrics)

        # Save traces
        trace_name = f"trace_{condition.policy}_p{condition.prompt_tokens}_d{condition.decode_tokens}_r{rep}"
        recorder.base_recorder.save_csv(str(output_dir / f"{trace_name}.csv"))

        # Save signal trace if available
        if signal_trace:
            import csv
            signal_trace_path = output_dir / f"{trace_name}_signals.csv"
            with open(signal_trace_path, 'w', newline='') as f:
                if signal_trace:
                    writer = csv.DictWriter(f, fieldnames=signal_trace[0].keys())
                    writer.writeheader()
                    writer.writerows(signal_trace)

        # Log with signal info if available
        signal_info = ""
        if metrics.policy_switches > 0:
            signal_info = f" switches={metrics.policy_switches}"
        logger.info(f"  Rep {rep+1}/{condition.n_repeats}: {metrics.total_s:.3f}s, "
                    f"{metrics.total_energy_j:.1f}J, {metrics.total_tok_per_j:.2f}tok/J{signal_info}")

    return results


def generate_prompt(prompt_type: str, target_tokens: int) -> str:
    """Generate prompt of approximate target length."""
    base_prompts = {
        "short": "Explain briefly: ",
        "medium": "Write a detailed explanation of ",
        "long": "Provide a comprehensive analysis with examples of ",
    }

    topics = [
        "how transformers work in deep learning",
        "the differences between prefill and decode phases in LLM inference",
        "energy efficiency optimization techniques for neural networks",
        "the attention mechanism and its computational complexity",
        "memory bandwidth limitations in GPU computing",
    ]

    base = base_prompts.get(prompt_type, base_prompts["medium"])
    topic = topics[target_tokens % len(topics)]

    # Repeat to approximate target length
    prompt = base + topic
    while len(prompt.split()) < target_tokens // 1.3:  # rough word-to-token ratio
        prompt += " Additionally, discuss " + topics[(len(prompt) // 100) % len(topics)]

    return prompt[:target_tokens * 6]  # cap at approximate char length


def aggregate_extended_results(results: List[ExtendedMetrics]) -> Dict[str, Any]:
    """Aggregate results with percentiles and jitter."""
    if not results:
        return {}

    def stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0, "std": 0, "p50": 0, "p95": 0}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "p50": statistics.median(values),
            "p95": np.percentile(values, 95) if len(values) >= 2 else values[-1],
        }

    metrics = [
        ("prefill_s", [r.prefill_s for r in results]),
        ("ttft_s", [r.ttft_s for r in results]),
        ("tpot_s_mean", [r.tpot_s_mean for r in results]),
        ("tpot_s_p95", [r.tpot_s_p95 for r in results]),
        ("tpot_s_std", [r.tpot_s_std for r in results]),  # jitter
        ("total_s", [r.total_s for r in results]),
        ("total_energy_j", [r.total_energy_j for r in results]),
        ("total_tok_per_j", [r.total_tok_per_j for r in results]),
        ("avg_power_w", [r.avg_power_w for r in results]),
        ("peak_power_w", [r.peak_power_w for r in results]),
        ("power_std_w", [r.power_std_w for r in results]),
    ]

    agg = {
        "n": len(results),
        "policy": results[0].policy,
        "concurrency": results[0].concurrency,
        "sdpa_backend": results[0].sdpa_backend,
        "sanity_check_pass_rate": sum(1 for r in results if r.energy_sanity_check) / len(results),
    }

    for name, values in metrics:
        s = stats(values)
        agg[f"{name}_mean"] = s["mean"]
        agg[f"{name}_std"] = s["std"]
        agg[f"{name}_p50"] = s["p50"]
        agg[f"{name}_p95"] = s["p95"]

    # GPU metrics
    sclks = [r.avg_sclk_mhz for r in results if r.avg_sclk_mhz]
    if sclks:
        agg["avg_sclk_mhz"] = statistics.mean(sclks)

    throttle_count = sum(1 for r in results if r.throttle_detected)
    agg["throttle_detected_runs"] = throttle_count

    return agg


def run_sustained_test(
    model,
    tokenizer,
    duration_seconds: int,
    decode_tokens: int = 64,
    policy: str = "auto",
    output_dir: Path = Path("results/sustained"),
    dpm_controller: Optional[DPMStateController] = None,
) -> Dict[str, Any]:
    """
    Run sustained thermal stability test.

    Continuously generates tokens for specified duration,
    logging metrics over time to detect thermal throttling
    and performance degradation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting sustained test: {duration_seconds}s, policy={policy}")

    prompt = "Explain in detail: "
    start_time = time.time()
    results = []
    time_series = []

    if dpm_controller and policy != "auto":
        policy_map = {"peak": "profile_peak", "min_sclk": "profile_min_sclk"}
        dpm_controller.set_perf_level(policy_map.get(policy, "auto"))

    metrics_collector = EnhancedGPUMetricsCollector()
    iteration = 0

    while time.time() - start_time < duration_seconds:
        iteration += 1
        iter_start = time.time()

        # Collect GPU metrics before
        gpu_metrics = metrics_collector.collect()

        # Run inference
        metrics, _, _ = run_single_inference(
            model, tokenizer, prompt, decode_tokens,
            policy=policy
        )

        iter_elapsed = time.time() - iter_start
        elapsed_total = time.time() - start_time

        # Record time series point
        point = {
            "elapsed_s": elapsed_total,
            "iteration": iteration,
            "total_s": metrics.total_s,
            "tpot_s_mean": metrics.tpot_s_mean,
            "tpot_s_p95": metrics.tpot_s_p95,
            "power_w": gpu_metrics.power_watts,
            "temp_c": gpu_metrics.temp_edge_c,
            "sclk_mhz": gpu_metrics.sclk_mhz,
            "gpu_busy_pct": gpu_metrics.gpu_busy_percent,
            "throttle": gpu_metrics.throttle_status,
        }
        time_series.append(point)

        if iteration % 10 == 0:
            logger.info(f"  {elapsed_total:.0f}s: iter={iteration}, "
                        f"tpot_p95={metrics.tpot_s_p95*1000:.1f}ms, "
                        f"temp={gpu_metrics.temp_edge_c}C, "
                        f"power={gpu_metrics.power_watts:.1f}W")

    # Restore auto
    if dpm_controller:
        dpm_controller.set_auto_mode()

    # Save time series
    ts_path = output_dir / f"sustained_{policy}_{duration_seconds}s.json"
    with open(ts_path, "w") as f:
        json.dump({
            "policy": policy,
            "duration_s": duration_seconds,
            "decode_tokens": decode_tokens,
            "iterations": len(time_series),
            "time_series": time_series,
        }, f, indent=2)

    logger.info(f"Sustained test complete: {len(time_series)} iterations, saved to {ts_path}")

    # Summary stats
    temps = [p["temp_c"] for p in time_series if p["temp_c"]]
    tpots = [p["tpot_s_p95"] for p in time_series]
    powers = [p["power_w"] for p in time_series]

    return {
        "iterations": len(time_series),
        "duration_s": duration_seconds,
        "temp_max_c": max(temps) if temps else None,
        "temp_delta_c": (temps[-1] - temps[0]) if len(temps) > 1 else 0,
        "tpot_p95_first_10": statistics.mean(tpots[:10]) if len(tpots) >= 10 else 0,
        "tpot_p95_last_10": statistics.mean(tpots[-10:]) if len(tpots) >= 10 else 0,
        "power_mean_w": statistics.mean(powers) if powers else 0,
        "throttle_detected": any(p["throttle"] for p in time_series),
    }


def main():
    parser = argparse.ArgumentParser(description="Extended HF Inference Research Harness")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--manifest", type=str, help="YAML manifest file")
    parser.add_argument("--compare", action="store_true", help="Run 4-policy comparison")
    parser.add_argument("--concurrent-compare", action="store_true", help="Run concurrent request comparison")
    parser.add_argument("--regime-test", action="store_true", help="Test regime detector recommendations")
    parser.add_argument("--sustained", type=int, help="Sustained test duration (seconds)")
    parser.add_argument("--sustained-duration", type=int, help="Alias for --sustained")
    parser.add_argument("--output-dir", type=str, default="results/hf_extended")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--samples", type=int, help="Alias for --repeats")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--decode-length", type=int, help="Alias for --decode-tokens")
    parser.add_argument("--prompt-tokens", type=int, default=50)
    parser.add_argument("--prompt-length", type=int, help="Alias for --prompt-tokens")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--policies", nargs="+", default=["auto"], help="DVFS policies to run")
    parser.add_argument("--policy", type=str, help="Single policy to run (for regime matrix experiments)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--enable-dpm-control", action="store_true", help="Enable manual DPM state control")
    parser.add_argument("--sclk-indices", type=str, help="SCLK DPM indices (comma-separated)")
    parser.add_argument("--mclk-indices", type=str, help="MCLK DPM indices (comma-separated)")
    parser.add_argument("--enable-roctx", action="store_true", help="Enable ROCTx kernel attribution")
    parser.add_argument("--enable-pytorch-profiler", action="store_true", help="Enable PyTorch profiler")

    args = parser.parse_args()

    # Handle argument aliases
    if args.sustained_duration:
        args.sustained = args.sustained_duration
    if args.samples:
        args.repeats = args.samples
    if args.decode_length:
        args.decode_tokens = args.decode_length
    if args.prompt_length:
        args.prompt_tokens = args.prompt_length

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ROCTx availability check
    if args.enable_roctx or args.enable_pytorch_profiler:
        roctx_status = check_roctx_availability()
        logger.info(f"ROCTx status: {roctx_status}")

    # Load model
    logger.info(f"Loading model: {args.model}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # Initialize DPM controller
    dpm_controller = DPMStateController()
    logger.info(f"DPM States - SCLK: {dpm_controller.get_available_states('sclk')}")
    logger.info(f"DPM States - MCLK: {dpm_controller.get_available_states('mclk')}")
    logger.info(f"SDPA Backend: {detect_sdpa_backend()}")

    if args.sustained:
        # Sustained thermal test
        for policy in ["auto", "peak"]:
            result = run_sustained_test(
                model, tokenizer,
                duration_seconds=args.sustained,
                decode_tokens=args.decode_tokens,
                policy=policy,
                output_dir=output_dir,
                dpm_controller=dpm_controller,
            )
            logger.info(f"Sustained {policy}: {json.dumps(result, indent=2)}")

    elif args.compare:
        # 9-policy comparison (includes signal controllers, windowed, latent, and latent_v2 hook-based)
        policies = ["auto", "peak", "phase_split", "controller", "signal_controller", "signal_controller_v2", "windowed", "latent", "latent_v2"]
        all_results = {}

        for policy in policies:
            condition = ExperimentCondition(
                model_id=args.model,
                prompt_type="medium",
                prompt_tokens=args.prompt_tokens,
                decode_tokens=args.decode_tokens,
                policy=policy,
                n_repeats=args.repeats,
            )

            results = run_experiment_condition(
                model, tokenizer, condition, output_dir, dpm_controller
            )
            all_results[policy] = aggregate_extended_results(results)

        # Save results
        with open(output_dir / "comparison_extended.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Print summary
        print("\n" + "="*100)
        print("EXTENDED COMPARISON (mean ± std)")
        print("="*100)
        print(f"{'Policy':15s} | {'Total(s)':>12s} | {'TPOT_p95(ms)':>12s} | {'Jitter(ms)':>12s} | "
              f"{'Energy(J)':>12s} | {'Tok/J':>10s} | {'Power(W)':>10s}")
        print("-"*100)

        for policy, agg in all_results.items():
            print(f"{policy:15s} | "
                  f"{agg['total_s_mean']:.3f}±{agg['total_s_std']:.3f} | "
                  f"{agg['tpot_s_p95_mean']*1000:.1f}±{agg['tpot_s_p95_std']*1000:.1f} | "
                  f"{agg['tpot_s_std_mean']*1000:.2f}±{agg['tpot_s_std_std']*1000:.2f} | "
                  f"{agg['total_energy_j_mean']:.1f}±{agg['total_energy_j_std']:.1f} | "
                  f"{agg['total_tok_per_j_mean']:.2f}±{agg['total_tok_per_j_std']:.2f} | "
                  f"{agg['avg_power_w_mean']:.0f}±{agg['avg_power_w_std']:.0f}")

        print("="*100)
        logger.info(f"Results saved to {output_dir / 'comparison_extended.json'}")

    elif args.concurrent_compare:
        # Concurrent request comparison
        concurrency_levels = [1, 2, 4]
        policies = ["auto", "controller"]
        all_results = {}

        for conc in concurrency_levels:
            for policy in policies:
                key = f"c{conc}_{policy}"
                condition = ExperimentCondition(
                    model_id=args.model,
                    prompt_type="medium",
                    prompt_tokens=args.prompt_tokens,
                    decode_tokens=args.decode_tokens,
                    policy=policy,
                    concurrency=conc,
                    n_repeats=args.repeats,
                )

                batch_results = run_concurrent_experiment(
                    model, tokenizer, condition, output_dir, dpm_controller
                )

                # Aggregate batch results
                if batch_results:
                    all_results[key] = {
                        "concurrency": conc,
                        "policy": policy,
                        "n": len(batch_results),
                        "batch_latency_mean": statistics.mean([b.total_latency_s for b in batch_results]),
                        "throughput_tok_s_mean": statistics.mean([b.throughput_tok_per_s for b in batch_results]),
                        "total_energy_j_mean": statistics.mean([b.total_energy_j for b in batch_results]),
                        "tok_per_j_mean": statistics.mean([b.tok_per_j for b in batch_results]),
                        "avg_power_w_mean": statistics.mean([b.avg_power_w for b in batch_results]),
                    }

        # Save results
        with open(output_dir / "concurrent_comparison.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Print summary
        print("\n" + "="*90)
        print("CONCURRENT REQUEST COMPARISON")
        print("="*90)
        print(f"{'Config':12s} | {'Batch(s)':>10s} | {'Tok/s':>10s} | {'Energy(J)':>10s} | {'Tok/J':>8s} | {'Power(W)':>10s}")
        print("-"*90)

        for key, agg in sorted(all_results.items()):
            print(f"{key:12s} | "
                  f"{agg['batch_latency_mean']:.3f} | "
                  f"{agg['throughput_tok_s_mean']:.1f} | "
                  f"{agg['total_energy_j_mean']:.1f} | "
                  f"{agg['tok_per_j_mean']:.2f} | "
                  f"{agg['avg_power_w_mean']:.0f}")

        print("="*90)
        logger.info(f"Results saved to {output_dir / 'concurrent_comparison.json'}")

    elif args.regime_test:
        # Test regime detector across configurations
        print("\n" + "="*80)
        print("REGIME DETECTOR RECOMMENDATIONS")
        print("="*80)

        model_params = estimate_model_params(args.model)
        print(f"Model: {args.model} (~{model_params/1e9:.1f}B params)")
        print()

        test_configs = [
            (256, 1), (256, 4), (2048, 1), (2048, 4),
        ]

        for prompt_len, conc in test_configs:
            should_control, reason = should_use_active_control(
                model_params, prompt_len, conc
            )
            recommendation = "ACTIVE CONTROL" if should_control else "AUTO (defer)"
            print(f"  prompt={prompt_len:4d}, conc={conc} -> {recommendation}")
            print(f"    {reason}")
            print()

        # Validate with actual runs if requested
        if args.repeats > 0:
            print("Running validation experiments...")
            validation_results = {}

            for prompt_len, conc in [(256, 1), (256, 4)]:
                should_control, _ = should_use_active_control(model_params, prompt_len, conc)
                recommended = "controller" if should_control else "auto"

                # Test both auto and controller
                for policy in ["auto", "controller"]:
                    key = f"p{prompt_len}_c{conc}_{policy}"
                    condition = ExperimentCondition(
                        model_id=args.model,
                        prompt_type="short" if prompt_len <= 256 else "medium",
                        prompt_tokens=prompt_len,
                        decode_tokens=args.decode_tokens,
                        concurrency=conc,
                        policy=policy,
                        n_repeats=min(3, args.repeats),  # Quick validation
                    )

                    if conc > 1:
                        batch_results = run_concurrent_experiment(
                            model, tokenizer, condition, output_dir, dpm_controller
                        )
                        if batch_results:
                            validation_results[key] = {
                                "tok_per_j": statistics.mean([b.tok_per_j for b in batch_results]),
                                "recommended": recommended,
                            }
                    else:
                        results = run_experiment_condition(
                            model, tokenizer, condition, output_dir, dpm_controller
                        )
                        agg = aggregate_extended_results(results)
                        validation_results[key] = {
                            "tok_per_j": agg.get("total_tok_per_j_mean", 0),
                            "recommended": recommended,
                        }

            # Save validation
            with open(output_dir / "regime_validation.json", "w") as f:
                json.dump(validation_results, f, indent=2)

            print("\nValidation Results (tok/J):")
            for key, val in sorted(validation_results.items()):
                rec_marker = "*" if key.endswith(val["recommended"]) else " "
                print(f"  {rec_marker} {key}: {val['tok_per_j']:.3f} tok/J")

    elif args.policy:
        # Single-policy mode for regime matrix experiments
        policy = args.policy
        condition = ExperimentCondition(
            model_id=args.model,
            prompt_type="custom",
            prompt_tokens=args.prompt_tokens,
            decode_tokens=args.decode_tokens,
            policy=policy,
            concurrency=args.concurrency,
            n_repeats=args.repeats,
            temperature=args.temperature,
        )

        logger.info(f"Running single-policy experiment: {policy}")
        logger.info(f"  prompt_tokens={args.prompt_tokens}, decode_tokens={args.decode_tokens}")
        logger.info(f"  concurrency={args.concurrency}, temperature={args.temperature}")

        if args.concurrency > 1:
            batch_results = run_concurrent_experiment(
                model, tokenizer, condition, output_dir, dpm_controller
            )
            if batch_results:
                all_results = {
                    policy: {
                        "n": len(batch_results),
                        "batch_latency_mean": statistics.mean([b.total_latency_s for b in batch_results]),
                        "throughput_tok_s_mean": statistics.mean([b.throughput_tok_per_s for b in batch_results]),
                        "total_energy_j_mean": statistics.mean([b.total_energy_j for b in batch_results]),
                        "total_tok_per_j_mean": statistics.mean([b.tok_per_j for b in batch_results]),
                        "avg_power_w_mean": statistics.mean([b.avg_power_w for b in batch_results]),
                        "total_s_mean": statistics.mean([b.total_latency_s for b in batch_results]),
                        "tpot_s_p95_mean": 0.0,  # Not available in batch mode
                        "tpot_s_std_mean": 0.0,
                    }
                }
            else:
                all_results = {policy: {"error": "No results"}}
        else:
            results = run_experiment_condition(
                model, tokenizer, condition, output_dir, dpm_controller
            )
            all_results = {policy: aggregate_extended_results(results)}

        # Save results in format expected by run_regime_matrix.py
        with open(output_dir / "comparison_extended.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Print summary
        if policy in all_results and "error" not in all_results[policy]:
            agg = all_results[policy]
            print(f"\nSingle-policy result: {policy}")
            print(f"  Total time: {agg.get('total_s_mean', 0):.3f}s")
            print(f"  Energy: {agg.get('total_energy_j_mean', 0):.1f}J")
            print(f"  Tok/J: {agg.get('total_tok_per_j_mean', 0):.3f}")
            print(f"  TPOT p95: {agg.get('tpot_s_p95_mean', 0)*1000:.1f}ms")
        else:
            print(f"Error in experiment: {all_results}")

        logger.info(f"Results saved to {output_dir / 'comparison_extended.json'}")

    else:
        # Default: run single inference for quick testing
        prompt = "Explain quantum computing in simple terms: "
        logger.info(f"Running default test with policy: {args.policies[0]}")
        metrics, tpots, _ = run_single_inference(
            model, tokenizer, prompt, args.decode_tokens,
            policy=args.policies[0], dpm_controller=dpm_controller
        )
        print(f"\nDefault test result:")
        print(f"  Total time: {metrics.total_s:.3f}s")
        print(f"  Energy: {metrics.total_energy_j:.1f}J")
        print(f"  Tok/J: {metrics.total_tok_per_j:.3f}")

    # Restore auto mode
    dpm_controller.set_auto_mode()


if __name__ == "__main__":
    main()
