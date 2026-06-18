#!/usr/bin/env python3
"""
Expanded Correctness Prediction Experiment

Uses the PowerTraceLLM-AMD Correctness Suite with proper verification types:
  - exact: String matching
  - numeric: Number comparison with tolerance
  - multiple_choice: A/B/C/D selection
  - unit_test: Executable code verification

Reports calibration metrics (AUC, ECE, Brier) split by verification type
to understand where internal confidence signals are most predictive.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.energy_harness.amd_smi_monitor import PowerTraceRecorder
from scripts.eval_suite import (
    EVAL_SUITE_EXPANDED,
    check_answer,
    check_answer_simple,
    get_suite_stats,
)

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class SignalStats:
    """Per-token signal statistics."""
    margins: List[float]
    entropies: List[float]
    deltas: List[float]

    def margin_min(self) -> float:
        return min(self.margins) if self.margins else 0.0

    def margin_mean(self) -> float:
        return sum(self.margins) / len(self.margins) if self.margins else 0.0

    def entropy_max(self) -> float:
        return max(self.entropies) if self.entropies else 0.0


def generate_with_signals(
    model, tokenizer, prompt: str, max_tokens: int = 64
) -> Tuple[str, SignalStats, float]:
    """Generate response with signal capture and energy measurement."""

    messages = [{"role": "user", "content": f"{prompt}\nAnswer concisely."}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]
    generated_ids = inputs.input_ids.clone()

    margins = []
    entropies = []
    deltas = []
    prev_hidden = None

    # Hook for capturing hidden states
    captured_hidden = [None]

    def hook_fn(module, inp, out):
        captured_hidden[0] = out[:, -1, :].detach()

    # Find norm layer
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        handle = model.model.norm.register_forward_hook(hook_fn)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        handle = model.transformer.ln_f.register_forward_hook(hook_fn)
    else:
        handle = None

    recorder = PowerTraceRecorder(sample_interval_ms=10)
    recorder.start()

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Margin
            top_logits, _ = torch.topk(logits[0], k=2)
            margin = (top_logits[0] - top_logits[1]).item()
            margins.append(margin)

            # Entropy
            top_k = 100
            top_logits_k, _ = torch.topk(logits[0], k=min(top_k, logits.shape[-1]))
            probs = F.softmax(top_logits_k, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            entropies.append(entropy)

            # Delta
            if captured_hidden[0] is not None and prev_hidden is not None:
                delta = torch.norm(captured_hidden[0] - prev_hidden) / (torch.norm(captured_hidden[0]) + 1e-10)
                deltas.append(delta.item())
            else:
                deltas.append(0.0)
            prev_hidden = captured_hidden[0].clone() if captured_hidden[0] is not None else None

            # Sample
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    recorder.stop()
    if handle:
        handle.remove()

    output_ids = generated_ids[0, input_len:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    energy = sum(s.power_watts * 0.01 for s in recorder.samples) if recorder.samples else 0

    return output_text, SignalStats(margins, entropies, deltas), energy


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        in_bin = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = y_true[in_bin].mean()
            bin_conf = y_pred[in_bin].mean()
            ece += (in_bin.sum() / total) * abs(bin_acc - bin_conf)

    return ece


def run_experiment(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: Path = Path("results/expanded_correctness"),
    skip_unit_tests: bool = True,
) -> Dict[str, Any]:
    """Run expanded correctness prediction experiment."""

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

    # Print suite stats
    stats = get_suite_stats()
    print(f"\nEval suite: {stats['total']} items")
    print(f"  Categories: {stats['by_category']}")
    print(f"  Difficulties: {stats['by_difficulty']}")
    print(f"  Verify types: {stats['by_verify_type']}")

    # Filter suite
    suite = EVAL_SUITE_EXPANDED
    if skip_unit_tests:
        suite = [item for item in suite if item.get("verify") != "unit_test"]
        print(f"\nSkipping unit tests, {len(suite)} items remaining")

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        generate_with_signals(model, tokenizer, "Hello", 5)

    # Run evaluation
    results = []

    print("\n=== Running Evaluation ===")
    for idx, item in enumerate(suite):
        output, signals, energy = generate_with_signals(model, tokenizer, item["q"])

        # Check correctness using appropriate verifier
        is_correct, verify_detail = check_answer(output, item)

        result = {
            "question": item["q"][:50],
            "expected": item["a"],
            "output": output[:100],
            "category": item["cat"],
            "difficulty": item["diff"],
            "verify_type": item.get("verify", "exact"),
            "is_correct": is_correct,
            "verify_detail": verify_detail,
            "margin_min": signals.margin_min(),
            "margin_mean": signals.margin_mean(),
            "entropy_max": signals.entropy_max(),
            "energy_j": energy,
        }
        results.append(result)

        status = "✓" if is_correct else "✗"
        print(f"  [{idx+1:3d}/{len(suite)}] {status} {item['cat']:10s} {item['diff']:6s} "
              f"{item.get('verify', 'exact'):15s} margin_min={signals.margin_min():.2f}")

    # Analysis
    analysis = analyze_results(results)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = model_name.split("/")[-1]

    output_file = output_dir / f"expanded_correctness_{model_short}.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": model_name,
            "n_items": len(suite),
            "analysis": analysis,
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nSaved: {output_file}")

    # Generate plots
    plot_calibration_by_type(results, analysis, output_dir / f"calibration_by_type_{model_short}.png")
    plot_margin_distribution(results, output_dir / f"margin_distribution_{model_short}.png")

    return analysis


def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze correctness prediction by verification type."""
    if not HAS_SKLEARN:
        return {"error": "sklearn not available"}

    analysis = {"overall": {}, "by_verify_type": {}, "by_category": {}, "by_difficulty": {}}

    # Overall
    analysis["overall"] = compute_metrics(results)

    # By verification type
    by_verify = defaultdict(list)
    for r in results:
        by_verify[r["verify_type"]].append(r)

    for vtype, items in by_verify.items():
        analysis["by_verify_type"][vtype] = compute_metrics(items)

    # By category
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    for cat, items in by_cat.items():
        analysis["by_category"][cat] = compute_metrics(items)

    # By difficulty
    by_diff = defaultdict(list)
    for r in results:
        by_diff[r["difficulty"]].append(r)

    for diff, items in by_diff.items():
        analysis["by_difficulty"][diff] = compute_metrics(items)

    return analysis


def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute calibration metrics for a set of results."""
    if len(results) < 2:
        return {"n": len(results), "error": "too few samples"}

    n_correct = sum(1 for r in results if r["is_correct"])
    n_total = len(results)
    accuracy = n_correct / n_total

    # Labels: 1 = error (we predict errors)
    labels = np.array([0 if r["is_correct"] else 1 for r in results])
    margins = np.array([r["margin_min"] for r in results])

    if len(np.unique(labels)) < 2:
        return {
            "n": n_total,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "auc": None,
            "ece": None,
            "brier": None,
            "note": "all same class",
        }

    # AUC: low margin → high error risk (invert margins)
    try:
        auc = roc_auc_score(labels, -margins)
    except:
        auc = None

    # Fit Platt scaling for calibration
    try:
        X = margins.reshape(-1, 1)
        platt = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        platt.fit(X, labels)

        p_pred = platt.predict_proba(X)[:, 1]
        brier = brier_score_loss(labels, p_pred)
        ece = compute_ece(labels, p_pred)

        platt_coef = float(platt.coef_[0][0])
        platt_intercept = float(platt.intercept_[0])
    except:
        brier = None
        ece = None
        platt_coef = None
        platt_intercept = None

    return {
        "n": n_total,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "error_rate": 1 - accuracy,
        "auc": float(auc) if auc else None,
        "ece": float(ece) if ece else None,
        "brier": float(brier) if brier else None,
        "platt_coef": platt_coef,
        "platt_intercept": platt_intercept,
        "margin_stats": {
            "correct_mean": float(np.mean([r["margin_min"] for r in results if r["is_correct"]])) if n_correct > 0 else None,
            "error_mean": float(np.mean([r["margin_min"] for r in results if not r["is_correct"]])) if n_total - n_correct > 0 else None,
        },
    }


def plot_calibration_by_type(results: List[Dict], analysis: Dict, output_path: Path):
    """Plot calibration metrics by verification type."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    verify_types = list(analysis["by_verify_type"].keys())
    colors = ['blue', 'orange', 'green', 'red'][:len(verify_types)]

    # AUC by type
    ax1 = axes[0]
    aucs = [analysis["by_verify_type"][vt].get("auc") or 0 for vt in verify_types]
    bars = ax1.bar(verify_types, aucs, color=colors)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('AUC')
    ax1.set_title('Error Prediction AUC by Verify Type')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(aucs):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center')

    # Accuracy by type
    ax2 = axes[1]
    accs = [analysis["by_verify_type"][vt].get("accuracy", 0) * 100 for vt in verify_types]
    ax2.bar(verify_types, accs, color=colors)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy by Verify Type')
    for i, v in enumerate(accs):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center')

    # ECE by type
    ax3 = axes[2]
    eces = [analysis["by_verify_type"][vt].get("ece") or 0 for vt in verify_types]
    ax3.bar(verify_types, eces, color=colors)
    ax3.set_ylabel('ECE')
    ax3.set_title('Calibration Error by Verify Type')
    for i, v in enumerate(eces):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_margin_distribution(results: List[Dict], output_path: Path):
    """Plot margin distribution by correctness."""
    fig, ax = plt.subplots(figsize=(10, 5))

    correct_margins = [r["margin_min"] for r in results if r["is_correct"]]
    error_margins = [r["margin_min"] for r in results if not r["is_correct"]]

    bins = np.linspace(0, max(correct_margins + error_margins) + 1, 20)

    ax.hist(correct_margins, bins=bins, alpha=0.6, label=f'Correct (n={len(correct_margins)})', color='green')
    ax.hist(error_margins, bins=bins, alpha=0.6, label=f'Error (n={len(error_margins)})', color='red')

    ax.set_xlabel('Minimum Margin')
    ax.set_ylabel('Count')
    ax.set_title('Margin Distribution: Correct vs Error')
    ax.legend()

    # Add vertical lines for means
    if correct_margins:
        ax.axvline(np.mean(correct_margins), color='green', linestyle='--', label=f'Correct mean: {np.mean(correct_margins):.2f}')
    if error_margins:
        ax.axvline(np.mean(error_margins), color='red', linestyle='--', label=f'Error mean: {np.mean(error_margins):.2f}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Expanded correctness prediction experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/expanded_correctness"))
    parser.add_argument("--include-unit-tests", action="store_true", help="Include code unit tests")
    args = parser.parse_args()

    analysis = run_experiment(
        model_name=args.model,
        output_dir=args.output_dir,
        skip_unit_tests=not args.include_unit_tests,
    )

    print("\n" + "=" * 70)
    print("EXPANDED CORRECTNESS PREDICTION ANALYSIS")
    print("=" * 70)

    overall = analysis.get("overall", {})
    print(f"\nOVERALL:")
    print(f"  Accuracy: {overall.get('accuracy', 0)*100:.1f}% ({overall.get('n_correct', 0)}/{overall.get('n', 0)})")
    print(f"  AUC: {overall.get('auc', 'N/A'):.3f}" if overall.get('auc') else "  AUC: N/A")
    print(f"  ECE: {overall.get('ece', 'N/A'):.3f}" if overall.get('ece') else "  ECE: N/A")
    print(f"  Brier: {overall.get('brier', 'N/A'):.3f}" if overall.get('brier') else "  Brier: N/A")

    print("\nBY VERIFICATION TYPE:")
    for vtype, data in analysis.get("by_verify_type", {}).items():
        auc_str = f"{data['auc']:.3f}" if data.get('auc') else "N/A"
        print(f"  {vtype:15s}: acc={data.get('accuracy', 0)*100:5.1f}%  AUC={auc_str}  n={data.get('n', 0)}")

    print("\nBY CATEGORY:")
    for cat, data in analysis.get("by_category", {}).items():
        auc_str = f"{data['auc']:.3f}" if data.get('auc') else "N/A"
        print(f"  {cat:12s}: acc={data.get('accuracy', 0)*100:5.1f}%  AUC={auc_str}  n={data.get('n', 0)}")

    print("\nBY DIFFICULTY:")
    for diff, data in analysis.get("by_difficulty", {}).items():
        auc_str = f"{data['auc']:.3f}" if data.get('auc') else "N/A"
        print(f"  {diff:8s}: acc={data.get('accuracy', 0)*100:5.1f}%  AUC={auc_str}  n={data.get('n', 0)}")


if __name__ == "__main__":
    main()
