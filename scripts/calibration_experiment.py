#!/usr/bin/env python3
"""
Calibration Experiment: Fit margin → p(error) with proper calibration.

Key insight from feedback: AUC=0.94 is ranking, not calibration.
ECE=0.36 means our p(error) predictions are unreliable.

This experiment:
1. Collects (margin, correct) pairs across the eval suite
2. Fits calibrators: isotonic regression, Platt scaling
3. Reports: AUC + ECE + Brier + reliability plot per task type
4. Produces calibrated margin→p(error) parameters for EVC

Reference: "ranking is not calibration" - classic ML calibration literature
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Calibration imports
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from scripts.correctness_prediction_experiment import EVAL_SUITE, check_answer


@dataclass
class CalibrationData:
    """Data for calibration analysis."""
    margins: List[float]
    correct: List[bool]
    categories: List[str]
    difficulties: List[str]
    energies: List[float]


def collect_margin_data(
    model, tokenizer, max_tokens: int = 32
) -> CalibrationData:
    """Collect (margin, correct) pairs for all eval items."""

    data = CalibrationData(
        margins=[], correct=[], categories=[], difficulties=[], energies=[]
    )

    print(f"\n=== Collecting Margin Data ({len(EVAL_SUITE)} items) ===")

    for idx, item in enumerate(EVAL_SUITE):
        recorder = PowerTraceRecorder(sample_interval_ms=10)
        recorder.start()

        # Generate with margin tracking
        messages = [{"role": "user", "content": f"{item['q']}\nAnswer with just the final answer, no explanation."}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        input_len = inputs.input_ids.shape[1]
        generated_ids = inputs.input_ids.clone()
        margins = []

        with torch.no_grad():
            for step in range(max_tokens):
                outputs = model(generated_ids)
                logits = outputs.logits[:, -1, :]

                # Compute margin
                top_logits, _ = torch.topk(logits[0], k=2)
                margin = (top_logits[0] - top_logits[1]).item()
                margins.append(margin)

                # Greedy decode
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        recorder.stop()

        output_ids = generated_ids[0, input_len:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        is_correct = check_answer(output_text, item["a"])
        energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

        # Use margin_min as the signal (best predictor from Result 12)
        margin_min = min(margins) if margins else 0.0

        data.margins.append(margin_min)
        data.correct.append(is_correct)
        data.categories.append(item["cat"])
        data.difficulties.append(item["diff"])
        data.energies.append(energy)

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1}/{len(EVAL_SUITE)}] {status} margin_min={margin_min:.2f} {item['cat']}/{item['diff']}")

    return data


def fit_isotonic_calibrator(margins: np.ndarray, errors: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression for margin → p(error)."""
    # Isotonic regression: monotonic mapping from margin to p(error)
    # Lower margin → higher p(error), so we fit on negative margin
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
    iso.fit(-margins, errors)  # Negative because lower margin = higher error
    return iso


def fit_platt_calibrator(margins: np.ndarray, errors: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling (logistic regression) for margin → p(error)."""
    # Reshape for sklearn
    X = margins.reshape(-1, 1)
    platt = LogisticRegression(C=1.0, solver='lbfgs')
    platt.fit(X, errors)
    return platt


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute ECE, Brier, and reliability diagram data."""

    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # AUC (if we have both classes)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = None

    # ECE and reliability diagram
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    reliability_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_actual = y_true[in_bin].mean()
            ece += np.abs(avg_confidence - avg_actual) * prop_in_bin

            reliability_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "avg_predicted": float(avg_confidence),
                "avg_actual": float(avg_actual),
                "count": int(in_bin.sum()),
                "gap": float(avg_confidence - avg_actual),
            })

    return {
        "brier": float(brier),
        "ece": float(ece),
        "auc": float(auc) if auc is not None else None,
        "n_samples": len(y_true),
        "reliability_data": reliability_data,
    }


def plot_calibration(
    metrics_uncal: Dict,
    metrics_iso: Dict,
    metrics_platt: Dict,
    output_path: Path,
):
    """Plot reliability diagrams for all calibration methods."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = [
        ("Uncalibrated (Linear)", metrics_uncal),
        ("Isotonic Regression", metrics_iso),
        ("Platt Scaling", metrics_platt),
    ]

    for ax, (name, metrics) in zip(axes, methods):
        reliability = metrics["reliability_data"]

        if reliability:
            bins = [(r["bin_lower"] + r["bin_upper"]) / 2 for r in reliability]
            predicted = [r["avg_predicted"] for r in reliability]
            actual = [r["avg_actual"] for r in reliability]
            counts = [r["count"] for r in reliability]

            # Bar chart
            ax.bar(bins, actual, width=0.08, alpha=0.7, label='Actual', color='steelblue')
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax.scatter(bins, predicted, color='red', s=50, zorder=5, label='Predicted')

            # Add count annotations
            for b, a, c in zip(bins, actual, counts):
                if c > 0:
                    ax.annotate(f'n={c}', (b, a + 0.05), ha='center', fontsize=8)

        ax.set_xlabel('Predicted p(error)')
        ax.set_ylabel('Actual error rate')
        ax.set_title(f'{name}\nECE={metrics["ece"]:.3f}, Brier={metrics["brier"]:.3f}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_margin_vs_error(
    margins: np.ndarray,
    errors: np.ndarray,
    iso_calibrator: IsotonicRegression,
    platt_calibrator: LogisticRegression,
    output_path: Path,
):
    """Plot margin distribution and calibration curves."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Margin distribution by correctness
    ax1 = axes[0]
    correct_margins = margins[~errors.astype(bool)]
    error_margins = margins[errors.astype(bool)]

    ax1.hist(correct_margins, bins=20, alpha=0.6, label=f'Correct (n={len(correct_margins)})', color='green')
    ax1.hist(error_margins, bins=20, alpha=0.6, label=f'Error (n={len(error_margins)})', color='red')
    ax1.set_xlabel('margin_min')
    ax1.set_ylabel('Count')
    ax1.set_title('Margin Distribution by Correctness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Calibration curves
    ax2 = axes[1]
    margin_range = np.linspace(margins.min(), margins.max(), 100)

    # Isotonic
    p_error_iso = iso_calibrator.predict(-margin_range)
    ax2.plot(margin_range, p_error_iso, 'b-', linewidth=2, label='Isotonic')

    # Platt
    p_error_platt = platt_calibrator.predict_proba(margin_range.reshape(-1, 1))[:, 1]
    ax2.plot(margin_range, p_error_platt, 'r-', linewidth=2, label='Platt')

    # Scatter actual data points
    ax2.scatter(margins[~errors.astype(bool)], np.zeros(len(correct_margins)) + 0.02,
                c='green', alpha=0.5, s=20, label='Correct')
    ax2.scatter(margins[errors.astype(bool)], np.ones(len(error_margins)) - 0.02,
                c='red', alpha=0.5, s=20, label='Error')

    ax2.set_xlabel('margin_min')
    ax2.set_ylabel('p(error)')
    ax2.set_title('Calibrated margin → p(error) Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: Path = Path("results/calibration"),
) -> Dict[str, Any]:
    """Run calibration experiment."""

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Warmup
    print("Warming up...")
    messages = [{"role": "user", "content": "Hello"}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    # Collect data
    data = collect_margin_data(model, tokenizer)

    # Convert to numpy
    margins = np.array(data.margins)
    errors = np.array([not c for c in data.correct], dtype=float)  # 1 = error, 0 = correct

    print(f"\n=== Calibration Analysis ===")
    print(f"Total samples: {len(margins)}")
    print(f"Error rate: {errors.mean():.1%}")
    print(f"Margin range: [{margins.min():.2f}, {margins.max():.2f}]")

    # Fit calibrators
    print("\nFitting calibrators...")
    iso_calibrator = fit_isotonic_calibrator(margins, errors)
    platt_calibrator = fit_platt_calibrator(margins, errors)

    # Get predictions from each method
    # Uncalibrated: simple linear mapping (what we had before)
    # p_error = 0.8 - 0.15 * margin (our previous defaults)
    p_error_uncal = np.clip(0.8 - 0.15 * margins, 0.01, 0.99)

    # Isotonic
    p_error_iso = iso_calibrator.predict(-margins)

    # Platt
    p_error_platt = platt_calibrator.predict_proba(margins.reshape(-1, 1))[:, 1]

    # Compute metrics
    print("\nComputing calibration metrics...")
    metrics_uncal = compute_calibration_metrics(errors, p_error_uncal)
    metrics_iso = compute_calibration_metrics(errors, p_error_iso)
    metrics_platt = compute_calibration_metrics(errors, p_error_platt)

    print(f"\n{'Method':<20} {'AUC':<8} {'ECE':<8} {'Brier':<8}")
    print("-" * 44)
    print(f"{'Uncalibrated':<20} {metrics_uncal['auc'] or 0:.3f}    {metrics_uncal['ece']:.3f}    {metrics_uncal['brier']:.3f}")
    print(f"{'Isotonic':<20} {metrics_iso['auc'] or 0:.3f}    {metrics_iso['ece']:.3f}    {metrics_iso['brier']:.3f}")
    print(f"{'Platt':<20} {metrics_platt['auc'] or 0:.3f}    {metrics_platt['ece']:.3f}    {metrics_platt['brier']:.3f}")

    # Per-category analysis
    print("\n=== Per-Category Analysis ===")
    category_metrics = {}
    for cat in set(data.categories):
        cat_mask = np.array([c == cat for c in data.categories])
        if cat_mask.sum() > 5:  # Need enough samples
            cat_margins = margins[cat_mask]
            cat_errors = errors[cat_mask]
            cat_p_iso = iso_calibrator.predict(-cat_margins)

            if len(np.unique(cat_errors)) > 1:
                cat_auc = roc_auc_score(cat_errors, cat_p_iso)
            else:
                cat_auc = None
            cat_brier = brier_score_loss(cat_errors, cat_p_iso)

            category_metrics[cat] = {
                "n": int(cat_mask.sum()),
                "error_rate": float(cat_errors.mean()),
                "auc": float(cat_auc) if cat_auc else None,
                "brier": float(cat_brier),
            }
            print(f"  {cat}: n={cat_mask.sum()}, error={cat_errors.mean():.1%}, AUC={cat_auc or 'N/A'}")

    # Per-difficulty analysis
    print("\n=== Per-Difficulty Analysis ===")
    difficulty_metrics = {}
    for diff in set(data.difficulties):
        diff_mask = np.array([d == diff for d in data.difficulties])
        if diff_mask.sum() > 3:
            diff_margins = margins[diff_mask]
            diff_errors = errors[diff_mask]
            diff_p_iso = iso_calibrator.predict(-diff_margins)

            if len(np.unique(diff_errors)) > 1:
                diff_auc = roc_auc_score(diff_errors, diff_p_iso)
            else:
                diff_auc = None
            diff_brier = brier_score_loss(diff_errors, diff_p_iso)

            difficulty_metrics[diff] = {
                "n": int(diff_mask.sum()),
                "error_rate": float(diff_errors.mean()),
                "auc": float(diff_auc) if diff_auc else None,
                "brier": float(diff_brier),
            }
            print(f"  {diff}: n={diff_mask.sum()}, error={diff_errors.mean():.1%}, AUC={diff_auc or 'N/A'}")

    # Save calibrator parameters for use in EVC
    # For isotonic, we save the fitted function values
    # For Platt, we save the logistic regression coefficients
    calibrator_params = {
        "isotonic": {
            "X_thresholds": (-iso_calibrator.X_thresholds_).tolist() if hasattr(iso_calibrator, 'X_thresholds_') else [],
            "y_thresholds": iso_calibrator.y_thresholds_.tolist() if hasattr(iso_calibrator, 'y_thresholds_') else [],
        },
        "platt": {
            "coef": platt_calibrator.coef_.tolist(),
            "intercept": platt_calibrator.intercept_.tolist(),
        },
    }

    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]

    plot_calibration(
        metrics_uncal, metrics_iso, metrics_platt,
        output_dir / f"reliability_diagrams_{model_short}.png"
    )

    plot_margin_vs_error(
        margins, errors, iso_calibrator, platt_calibrator,
        output_dir / f"margin_vs_error_{model_short}.png"
    )

    # Save results
    results = {
        "model": model_name,
        "n_samples": len(margins),
        "error_rate": float(errors.mean()),
        "margin_stats": {
            "min": float(margins.min()),
            "max": float(margins.max()),
            "mean": float(margins.mean()),
            "std": float(margins.std()),
        },
        "calibration_metrics": {
            "uncalibrated": metrics_uncal,
            "isotonic": metrics_iso,
            "platt": metrics_platt,
        },
        "category_metrics": category_metrics,
        "difficulty_metrics": difficulty_metrics,
        "calibrator_params": calibrator_params,
        "raw_data": {
            "margins": data.margins,
            "correct": data.correct,
            "categories": data.categories,
            "difficulties": data.difficulties,
            "energies": data.energies,
        },
    }

    output_file = output_dir / f"calibration_{model_short}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Calibration experiment for margin → p(error)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/calibration"))
    args = parser.parse_args()

    results = run_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)

    print(f"\nBest calibration method: ", end="")
    methods = results["calibration_metrics"]
    best = min(methods.items(), key=lambda x: x[1]["ece"])
    print(f"{best[0]} (ECE={best[1]['ece']:.3f})")

    print(f"\nFor EVC controller, use Platt parameters:")
    platt = results["calibrator_params"]["platt"]
    print(f"  coef: {platt['coef']}")
    print(f"  intercept: {platt['intercept']}")
    print(f"\n  p(error) = sigmoid({platt['coef'][0][0]:.4f} * margin + {platt['intercept'][0]:.4f})")


if __name__ == "__main__":
    main()
