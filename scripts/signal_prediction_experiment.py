#!/usr/bin/env python3
"""
Signal Prediction Experiment: Does latent delta predict TPOT spikes?

This is the KEY experiment before building any controller:
- Run decode with FIXED policy (no DVFS changes)
- Log per-token: latent delta, logit margin, TPOT, power
- Compute correlation and lead/lag analysis
- Answer: does latent at t predict TPOT at t+1..t+8?

If latent doesn't predict anything, we avoid burning weeks on controllers.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import statistics

import torch
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# Setup path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from src.energy_harness.internal_signals import ZeroOverheadLatentCapture

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TokenSignal:
    """Per-token signal measurements."""
    token_idx: int
    token_id: int

    # Timing
    tpot_ms: float  # Time per output token

    # Latent signals (from hook)
    latent_delta_norm: float  # ||h_t - h_{t-1}|| / ||h_t||
    latent_norm: float  # ||h_t||

    # Logit signals
    logit_margin: float  # top1 - top2 (no softmax!)
    top1_logit: float
    entropy_approx: float  # Optional: cheap entropy approximation

    # Power (if available)
    power_w: float = 0.0


@dataclass
class SequenceTrace:
    """Full trace for one generation sequence."""
    prompt: str
    temperature: float
    policy: str  # Fixed policy used
    model_id: str

    tokens: List[TokenSignal] = field(default_factory=list)

    # Aggregate metrics
    total_tokens: int = 0
    mean_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    total_energy_j: float = 0.0


def compute_latent_delta(current: torch.Tensor, previous: Optional[torch.Tensor]) -> Tuple[float, float]:
    """Compute latent delta norm and current norm."""
    current_norm = torch.norm(current).item()

    if previous is None:
        return 0.0, current_norm

    delta = current - previous
    delta_norm = torch.norm(delta).item() / (current_norm + 1e-10)

    return delta_norm, current_norm


def compute_logit_signals(logits: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute logit-based signals WITHOUT softmax (fast!).

    Returns: (margin, top1_logit, entropy_approx)
    """
    # logits shape: [1, vocab_size]
    logits = logits.squeeze(0)

    # Top-2 for margin
    top2 = torch.topk(logits, k=2)
    margin = (top2.values[0] - top2.values[1]).item()
    top1_logit = top2.values[0].item()

    # Cheap entropy approximation: use top-k logits only
    # This avoids full softmax over vocab
    top_k = 32
    topk_logits = torch.topk(logits, k=min(top_k, logits.shape[0])).values
    topk_probs = torch.softmax(topk_logits, dim=0)
    entropy_approx = -torch.sum(topk_probs * torch.log(topk_probs + 1e-10)).item()

    return margin, top1_logit, entropy_approx


def run_signal_trace(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    latent_capture: ZeroOverheadLatentCapture,
    power_recorder: Optional[PowerTraceRecorder] = None,
) -> SequenceTrace:
    """
    Run inference and collect per-token signals.

    NO DVFS CHANGES - we're just observing.
    """

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    # Initialize
    past_key_values = None
    prev_hidden = None
    tokens = []

    # Register hook
    latent_capture.register_hook()

    # Start power recording
    if power_recorder:
        power_recorder.start()

    # Prefill
    t_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values
    prefill_hidden = latent_capture.get_last_hidden()
    prev_hidden = prefill_hidden.clone() if prefill_hidden is not None else None

    # Get next token
    logits = outputs.logits[:, -1, :]

    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    generated_ids = [next_token.item()]
    current_input = next_token

    # Decode loop
    for i in range(max_new_tokens - 1):
        t_token_start = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

        t_token_end = time.perf_counter()
        tpot_ms = (t_token_end - t_token_start) * 1000

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        # Get latent state from hook
        current_hidden = latent_capture.get_last_hidden()

        # Compute signals
        delta_norm, latent_norm = compute_latent_delta(current_hidden, prev_hidden)
        margin, top1_logit, entropy_approx = compute_logit_signals(logits)

        # Get power sample if available
        power_w = 0.0
        if power_recorder and power_recorder.samples:
            power_w = power_recorder.samples[-1].power_watts

        # Store signal
        signal = TokenSignal(
            token_idx=i,
            token_id=next_token.item(),
            tpot_ms=tpot_ms,
            latent_delta_norm=delta_norm,
            latent_norm=latent_norm,
            logit_margin=margin,
            top1_logit=top1_logit,
            entropy_approx=entropy_approx,
            power_w=power_w,
        )
        tokens.append(signal)

        # Update state
        prev_hidden = current_hidden.clone() if current_hidden is not None else None

        # Sample next token
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated_ids.append(next_token.item())
        current_input = next_token

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Stop recording
    latent_capture.remove_hook()

    total_energy = 0.0
    if power_recorder:
        measurement = power_recorder.stop()
        total_energy = measurement.energy_joules

    # Compute aggregates
    tpots = [t.tpot_ms for t in tokens]

    trace = SequenceTrace(
        prompt=prompt[:200],
        temperature=temperature,
        policy="auto",  # Fixed
        model_id=model.config._name_or_path,
        tokens=tokens,
        total_tokens=len(tokens),
        mean_tpot_ms=statistics.mean(tpots) if tpots else 0,
        p95_tpot_ms=np.percentile(tpots, 95) if tpots else 0,
        total_energy_j=total_energy,
    )

    return trace


def compute_lead_lag_correlation(
    signals: List[TokenSignal],
    predictor: str = "latent_delta_norm",
    target: str = "tpot_ms",
    max_lag: int = 8,
) -> Dict[int, float]:
    """
    Compute correlation between predictor at t and target at t+lag.

    Returns: {lag: correlation} for lag in [-max_lag, max_lag]
    """
    predictor_vals = np.array([getattr(s, predictor) for s in signals])
    target_vals = np.array([getattr(s, target) for s in signals])

    correlations = {}

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Predictor leads target (target at t, predictor at t+lag)
            x = predictor_vals[:lag]
            y = target_vals[-lag:]
        elif lag > 0:
            # Predictor lags target (predictor at t, target at t+lag)
            x = predictor_vals[:-lag]
            y = target_vals[lag:]
        else:
            x = predictor_vals
            y = target_vals

        if len(x) > 2:
            corr, _ = stats.pearsonr(x, y)
            correlations[lag] = corr
        else:
            correlations[lag] = 0.0

    return correlations


def compute_spike_prediction_auc(
    signals: List[TokenSignal],
    predictor: str = "latent_delta_norm",
    spike_percentile: float = 90,
    lookahead: int = 1,
) -> float:
    """
    Compute AUC for predicting TPOT spikes.

    A "spike" is defined as TPOT in the top spike_percentile%.
    We predict at t whether t+lookahead will be a spike.
    """
    tpots = [s.tpot_ms for s in signals]
    threshold = np.percentile(tpots, spike_percentile)

    # Create labels: 1 if spike, 0 otherwise
    labels = [1 if t >= threshold else 0 for t in tpots]

    # Get predictor values with lookahead offset
    predictor_vals = [getattr(s, predictor) for s in signals]

    if lookahead > 0:
        # Predict spike at t+lookahead using signal at t
        x = predictor_vals[:-lookahead]
        y = labels[lookahead:]
    else:
        x = predictor_vals
        y = labels

    if sum(y) == 0 or sum(y) == len(y):
        # All same class, AUC undefined
        return 0.5

    try:
        auc = roc_auc_score(y, x)
        return auc
    except:
        return 0.5


def analyze_traces(traces: List[SequenceTrace]) -> Dict:
    """Analyze collected traces for signal predictive power."""

    # Aggregate all signals
    all_signals = []
    for trace in traces:
        all_signals.extend(trace.tokens)

    if not all_signals:
        return {}

    # Lead/lag correlations
    correlations = {
        "latent_delta_vs_tpot": compute_lead_lag_correlation(
            all_signals, "latent_delta_norm", "tpot_ms"
        ),
        "logit_margin_vs_tpot": compute_lead_lag_correlation(
            all_signals, "logit_margin", "tpot_ms"
        ),
        "entropy_vs_tpot": compute_lead_lag_correlation(
            all_signals, "entropy_approx", "tpot_ms"
        ),
    }

    # Spike prediction AUC
    spike_aucs = {}
    for predictor in ["latent_delta_norm", "logit_margin", "entropy_approx"]:
        for lookahead in [0, 1, 2, 4, 8]:
            key = f"{predictor}_lookahead_{lookahead}"
            spike_aucs[key] = compute_spike_prediction_auc(
                all_signals, predictor, spike_percentile=90, lookahead=lookahead
            )

    # Basic stats
    tpots = [s.tpot_ms for s in all_signals]
    deltas = [s.latent_delta_norm for s in all_signals]
    margins = [s.logit_margin for s in all_signals]

    stats_summary = {
        "n_tokens": len(all_signals),
        "n_traces": len(traces),
        "tpot_mean_ms": np.mean(tpots),
        "tpot_std_ms": np.std(tpots),
        "tpot_p95_ms": np.percentile(tpots, 95),
        "delta_mean": np.mean(deltas),
        "delta_std": np.std(deltas),
        "margin_mean": np.mean(margins),
        "margin_std": np.std(margins),
    }

    return {
        "correlations": correlations,
        "spike_aucs": spike_aucs,
        "stats": stats_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Signal Prediction Experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.7])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/signal_prediction"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Test prompts with varying difficulty
    prompts = [
        # Easy
        "What is the capital of France?",
        "What is 7 times 8?",
        "Name a primary color.",
        # Medium
        "Explain why the sky appears blue.",
        "What causes seasons on Earth?",
        "Describe how a computer works.",
        # Hard
        "Analyze the ethical implications of artificial general intelligence.",
        "Compare and contrast democracy and authoritarianism.",
        "Explain the theory of relativity in simple terms.",
        "Write a recursive function to compute Fibonacci numbers.",
    ][:args.n_prompts]

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )

    # Setup latent capture
    latent_capture = ZeroOverheadLatentCapture(model)

    # Setup power recorder
    try:
        from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
        power_recorder = PowerTraceRecorder(sample_interval_ms=10)
    except:
        power_recorder = None
        logger.warning("Power recording not available")

    all_results = {}

    for temp in args.temperatures:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running temperature={temp}")
        logger.info(f"{'='*60}")

        traces = []

        for i, prompt in enumerate(prompts):
            logger.info(f"  Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

            trace = run_signal_trace(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=temp,
                latent_capture=latent_capture,
                power_recorder=power_recorder,
            )
            traces.append(trace)

            logger.info(f"    Tokens: {trace.total_tokens}, Mean TPOT: {trace.mean_tpot_ms:.1f}ms")

        # Analyze
        analysis = analyze_traces(traces)

        # Store results
        temp_key = f"temp_{temp}"
        all_results[temp_key] = {
            "temperature": temp,
            "analysis": analysis,
            "traces": [asdict(t) for t in traces],
        }

        # Print key findings
        print(f"\n--- Temperature {temp} Results ---")
        print(f"Tokens analyzed: {analysis['stats']['n_tokens']}")
        print(f"TPOT: {analysis['stats']['tpot_mean_ms']:.1f} ± {analysis['stats']['tpot_std_ms']:.1f}ms")

        print("\nLead/Lag Correlations (latent_delta vs TPOT):")
        for lag, corr in sorted(analysis['correlations']['latent_delta_vs_tpot'].items()):
            marker = "***" if abs(corr) > 0.2 else ""
            print(f"  lag={lag:+2d}: r={corr:+.3f} {marker}")

        print("\nSpike Prediction AUC (top-10% TPOT):")
        for key, auc in analysis['spike_aucs'].items():
            if "latent" in key:
                marker = "***" if auc > 0.6 else ""
                print(f"  {key}: AUC={auc:.3f} {marker}")

    # Save results
    output_file = args.output_dir / f"signal_prediction_{args.model.split('/')[-1]}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SIGNAL PREDICTION SUMMARY")
    print("="*60)
    print("\nKey question: Does latent delta at t predict TPOT spike at t+k?")
    print("\nIf AUC > 0.6 and correlation significant at positive lag,")
    print("then latent signals have predictive power for resource needs.")
    print("="*60)


if __name__ == "__main__":
    main()
