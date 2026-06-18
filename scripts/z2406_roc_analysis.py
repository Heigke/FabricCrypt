#!/usr/bin/env python3
"""z2406: ROC analysis comparing softmax vs shadow confidence metrics for MNIST MLP."""

import struct
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models" / "mnist_mlp"
DATA_DIR = BASE / "data" / "MNIST" / "raw"
OUT_DIR = BASE / "results" / "z2406_roc_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model weights ──────────────────────────────────────────────────────

def load_weights():
    """Load MLP weights: 784→128 ReLU→64 ReLU→10."""
    w1 = np.frombuffer((MODEL_DIR / "w1.bin").read_bytes(), dtype=np.float32).reshape(128, 784)
    b1 = np.frombuffer((MODEL_DIR / "b1.bin").read_bytes(), dtype=np.float32)
    w2 = np.frombuffer((MODEL_DIR / "w2.bin").read_bytes(), dtype=np.float32).reshape(64, 128)
    b2 = np.frombuffer((MODEL_DIR / "b2.bin").read_bytes(), dtype=np.float32)
    w3 = np.frombuffer((MODEL_DIR / "w3.bin").read_bytes(), dtype=np.float32).reshape(10, 64)
    b3 = np.frombuffer((MODEL_DIR / "b3.bin").read_bytes(), dtype=np.float32)
    return (w1, b1), (w2, b2), (w3, b3)


# ── Load MNIST test data ────────────────────────────────────────────────────

def load_mnist_test():
    """Load MNIST test images and labels from raw IDX files."""
    with open(DATA_DIR / "t10k-images-idx3-ubyte", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255.0
    with open(DATA_DIR / "t10k-labels-idx1-ubyte", "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels


# ── Shadow metrics ───────────────────────────────────────────────────────────

def compute_sign_ratio(x, w):
    """For matmul x@w.T, count blocks of 4 products where all have same sign.

    For each output neuron j, the products are x[i]*w[j,i] for i in range(input_dim).
    We chunk these into blocks of 4 and check sign agreement.
    """
    # x: (input_dim,), w: (out_dim, input_dim)
    products = x[None, :] * w  # (out_dim, input_dim)
    n_out, n_in = products.shape
    # Trim to multiple of 4
    n_blocks = n_in // 4
    if n_blocks == 0:
        return 0.0
    trimmed = products[:, :n_blocks * 4].reshape(n_out, n_blocks, 4)
    signs = (trimmed >= 0)  # True = positive
    all_same = np.all(signs, axis=2) | np.all(~signs, axis=2)  # all pos or all neg
    return float(all_same.mean())


def compute_neighbor_diff(output):
    """Average |output[i] - output[i+1]| across adjacent neurons."""
    if len(output) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(output))))


# ── Softmax ──────────────────────────────────────────────────────────────────

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ── Manual AUC (trapezoidal) ────────────────────────────────────────────────

def compute_roc(scores, labels_binary):
    """Compute ROC curve and AUC. Higher score → more likely positive (correct).

    Returns: fpr_list, tpr_list, auc
    """
    # Sort by descending score
    order = np.argsort(-scores)
    sorted_labels = labels_binary[order]

    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return [0, 1], [0, 1], 0.5

    tp = 0
    fp = 0
    fpr_list = [0.0]
    tpr_list = [0.0]

    prev_score = None
    for i in range(len(sorted_labels)):
        if prev_score is not None and scores[order[i]] != prev_score:
            fpr_list.append(fp / n_neg)
            tpr_list.append(tp / n_pos)
        if sorted_labels[i]:
            tp += 1
        else:
            fp += 1
        prev_score = scores[order[i]]

    fpr_list.append(1.0)
    tpr_list.append(1.0)

    # Trapezoidal AUC
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    auc = float(np.trapz(tpr_arr, fpr_arr))

    return fpr_list, tpr_list, auc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading model weights...")
    (w1, b1), (w2, b2), (w3, b3) = load_weights()

    print("Loading MNIST test data...")
    images, labels = load_mnist_test()
    n = len(labels)
    print(f"  {n} test samples, {len(np.unique(labels))} classes")

    # Run inference and compute metrics
    print("Running inference + shadow metrics on all samples...")

    predictions = np.zeros(n, dtype=np.int32)
    softmax_conf = np.zeros(n)
    margin_conf = np.zeros(n)
    sign_ratios = np.zeros((n, 3))  # one per layer
    neighbor_diffs = np.zeros((n, 3))  # one per layer

    for i in range(n):
        x = images[i]

        # Layer 1: 784→128
        sr1 = compute_sign_ratio(x, w1)
        h1 = x @ w1.T + b1
        nd1 = compute_neighbor_diff(h1)
        a1 = np.maximum(h1, 0)  # ReLU

        # Layer 2: 128→64
        sr2 = compute_sign_ratio(a1, w2)
        h2 = a1 @ w2.T + b2
        nd2 = compute_neighbor_diff(h2)
        a2 = np.maximum(h2, 0)  # ReLU

        # Layer 3: 64→10
        sr3 = compute_sign_ratio(a2, w3)
        logits = a2 @ w3.T + b3
        nd3 = compute_neighbor_diff(logits)

        # Predictions & confidence
        probs = softmax(logits)
        pred = np.argmax(logits)
        predictions[i] = pred
        softmax_conf[i] = probs.max()

        sorted_logits = np.sort(logits)[::-1]
        margin_conf[i] = sorted_logits[0] - sorted_logits[1]

        sign_ratios[i] = [sr1, sr2, sr3]
        neighbor_diffs[i] = [nd1, nd2, nd3]

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n}")

    correct = (predictions == labels).astype(np.float64)
    acc = correct.mean()
    print(f"\nOverall accuracy: {acc*100:.2f}% ({int(correct.sum())}/{n})")

    # Aggregate shadow metrics
    avg_sign_ratio = sign_ratios.mean(axis=1)  # average across layers
    avg_neighbor_diff = neighbor_diffs.mean(axis=1)

    # For shadow scores: higher sign_ratio → more coherent → more confident
    # Higher neighbor_diff → more separation → more confident
    # Normalize both to [0,1] range then combine
    sr_min, sr_max = avg_sign_ratio.min(), avg_sign_ratio.max()
    nd_min, nd_max = avg_neighbor_diff.min(), avg_neighbor_diff.max()

    sr_norm = (avg_sign_ratio - sr_min) / (sr_max - sr_min + 1e-10)
    nd_norm = (avg_neighbor_diff - nd_min) / (nd_max - nd_min + 1e-10)

    combined_shadow = 0.5 * sr_norm + 0.5 * nd_norm

    # ── ROC curves ───────────────────────────────────────────────────────────
    print("\nComputing ROC curves...")

    metrics = {
        "softmax_conf": softmax_conf,
        "margin_conf": margin_conf,
        "sign_ratio": avg_sign_ratio,
        "neighbor_diff": avg_neighbor_diff,
        "combined_shadow": combined_shadow,
    }

    roc_results = {}
    for name, scores in metrics.items():
        fpr, tpr, auc = compute_roc(scores, correct.astype(bool))
        roc_results[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
        print(f"  {name:20s} AUC = {auc:.4f}")

    # ── Rejection analysis ───────────────────────────────────────────────────
    print("\nRejection analysis (reject lowest-confidence samples)...")
    rejection_rates = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    rejection_results = {}
    for name, scores in metrics.items():
        accs = []
        for rate in rejection_rates:
            threshold = np.percentile(scores, rate * 100)
            mask = scores >= threshold
            if mask.sum() == 0:
                accs.append(float('nan'))
            else:
                accs.append(float(correct[mask].mean()))
        rejection_results[name] = accs
        print(f"  {name:20s} @ 10% reject: {accs[3]*100:.2f}%  @ 20% reject: {accs[5]*100:.2f}%")

    # ── Per-digit analysis ───────────────────────────────────────────────────
    print("\nPer-digit shadow analysis:")
    per_digit = {}
    for d in range(10):
        mask = labels == d
        n_d = mask.sum()
        acc_d = correct[mask].mean()
        sr_d = avg_sign_ratio[mask].mean()
        nd_d = avg_neighbor_diff[mask].mean()
        cs_d = combined_shadow[mask].mean()
        sm_d = softmax_conf[mask].mean()
        per_digit[str(d)] = {
            "n": int(n_d),
            "accuracy": float(acc_d),
            "avg_sign_ratio": float(sr_d),
            "avg_neighbor_diff": float(nd_d),
            "avg_combined_shadow": float(cs_d),
            "avg_softmax_conf": float(sm_d),
        }
        print(f"  Digit {d}: n={n_d:4d}  acc={acc_d*100:.1f}%  "
              f"sign_ratio={sr_d:.4f}  neighbor_diff={nd_d:.2f}  "
              f"softmax={sm_d:.3f}  shadow={cs_d:.3f}")

    # ── Save results JSON ────────────────────────────────────────────────────
    results = {
        "experiment": "z2406_roc_analysis",
        "model": "MNIST MLP 784→128→64→10",
        "n_test": n,
        "accuracy": float(acc),
        "n_correct": int(correct.sum()),
        "n_incorrect": int(n - correct.sum()),
        "roc_auc": {name: roc_results[name]["auc"] for name in roc_results},
        "rejection_rates": rejection_rates,
        "rejection_accuracy": {name: rejection_results[name] for name in rejection_results},
        "per_digit": per_digit,
        "shadow_stats": {
            "sign_ratio_mean": float(avg_sign_ratio.mean()),
            "sign_ratio_std": float(avg_sign_ratio.std()),
            "neighbor_diff_mean": float(avg_neighbor_diff.mean()),
            "neighbor_diff_std": float(avg_neighbor_diff.std()),
            "sign_ratio_correct_mean": float(avg_sign_ratio[correct > 0.5].mean()),
            "sign_ratio_incorrect_mean": float(avg_sign_ratio[correct < 0.5].mean()),
            "neighbor_diff_correct_mean": float(avg_neighbor_diff[correct > 0.5].mean()),
            "neighbor_diff_incorrect_mean": float(avg_neighbor_diff[correct < 0.5].mean()),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    json_path = BASE / "results" / "z2406_roc_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Plot 1: ROC comparison ───────────────────────────────────────────────
    print("\nGenerating plots...")

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"softmax_conf": "#1f77b4", "margin_conf": "#ff7f0e",
              "sign_ratio": "#2ca02c", "neighbor_diff": "#d62728",
              "combined_shadow": "#9467bd"}
    labels_nice = {"softmax_conf": "Softmax confidence",
                   "margin_conf": "Margin (logit gap)",
                   "sign_ratio": "Shadow: sign ratio",
                   "neighbor_diff": "Shadow: neighbor diff",
                   "combined_shadow": "Shadow: combined"}

    for name in metrics:
        r = roc_results[name]
        # Subsample for plotting (too many points)
        fpr_a, tpr_a = np.array(r["fpr"]), np.array(r["tpr"])
        ax.plot(fpr_a, tpr_a, color=colors[name], linewidth=1.5,
                label=f'{labels_nice[name]} (AUC={r["auc"]:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random (AUC=0.5)')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("z2406: ROC — Predicting Correct vs Incorrect", fontsize=13)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'roc_comparison.png'}")

    # ── Plot 2: Rejection accuracy ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in metrics:
        accs = rejection_results[name]
        ax.plot([r * 100 for r in rejection_rates], [a * 100 for a in accs],
                'o-', color=colors[name], linewidth=1.5, markersize=4,
                label=labels_nice[name])

    ax.axhline(acc * 100, color='gray', linestyle=':', linewidth=1, label=f'Baseline ({acc*100:.1f}%)')
    ax.set_xlabel("Rejection Rate (%)", fontsize=12)
    ax.set_ylabel("Accuracy on Remaining Samples (%)", fontsize=12)
    ax.set_title("z2406: Accuracy vs Rejection Rate", fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rejection_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'rejection_accuracy.png'}")

    # ── Plot 3: Per-digit shadow ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    digits = list(range(10))

    sr_vals = [per_digit[str(d)]["avg_sign_ratio"] for d in digits]
    nd_vals = [per_digit[str(d)]["avg_neighbor_diff"] for d in digits]
    acc_vals = [per_digit[str(d)]["accuracy"] * 100 for d in digits]

    bar_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    axes[0].bar(digits, sr_vals, color=bar_colors)
    axes[0].set_xlabel("Digit")
    axes[0].set_ylabel("Avg Sign Ratio")
    axes[0].set_title("Sign Ratio by Digit")
    axes[0].set_xticks(digits)

    axes[1].bar(digits, nd_vals, color=bar_colors)
    axes[1].set_xlabel("Digit")
    axes[1].set_ylabel("Avg Neighbor Diff")
    axes[1].set_title("Neighbor Diff by Digit")
    axes[1].set_xticks(digits)

    axes[2].bar(digits, acc_vals, color=bar_colors)
    axes[2].set_xlabel("Digit")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("Accuracy by Digit")
    axes[2].set_xticks(digits)

    fig.suptitle("z2406: Per-Digit Shadow Signals", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_digit_shadow.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'per_digit_shadow.png'}")

    # ── Plot 4: Shadow vs softmax scatter ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    correct_mask = correct > 0.5
    incorrect_mask = ~correct_mask

    # Shadow combined vs softmax
    ax = axes[0]
    ax.scatter(softmax_conf[correct_mask], combined_shadow[correct_mask],
               s=1, alpha=0.15, c='#2ca02c', label=f'Correct ({int(correct_mask.sum())})')
    ax.scatter(softmax_conf[incorrect_mask], combined_shadow[incorrect_mask],
               s=6, alpha=0.6, c='#d62728', label=f'Incorrect ({int(incorrect_mask.sum())})')
    ax.set_xlabel("Softmax Confidence", fontsize=11)
    ax.set_ylabel("Combined Shadow Score", fontsize=11)
    ax.set_title("Combined Shadow vs Softmax")
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    # Sign ratio vs margin
    ax = axes[1]
    ax.scatter(margin_conf[correct_mask], avg_sign_ratio[correct_mask],
               s=1, alpha=0.15, c='#2ca02c', label=f'Correct')
    ax.scatter(margin_conf[incorrect_mask], avg_sign_ratio[incorrect_mask],
               s=6, alpha=0.6, c='#d62728', label=f'Incorrect')
    ax.set_xlabel("Margin Confidence (logit gap)", fontsize=11)
    ax.set_ylabel("Sign Ratio", fontsize=11)
    ax.set_title("Sign Ratio vs Margin")
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.3)

    fig.suptitle("z2406: Shadow vs Traditional Confidence", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shadow_vs_softmax_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT_DIR / 'shadow_vs_softmax_scatter.png'}")

    print(f"\nDone in {time.time()-t0:.1f}s")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"\nAUC (higher = better at distinguishing correct/incorrect):")
    for name in metrics:
        print(f"  {labels_nice[name]:30s} {roc_results[name]['auc']:.4f}")
    print(f"\nAt 10% rejection:")
    for name in metrics:
        print(f"  {labels_nice[name]:30s} {rejection_results[name][3]*100:.2f}%")
    print(f"\nShadow signal separation (correct vs incorrect):")
    print(f"  Sign ratio:    correct={results['shadow_stats']['sign_ratio_correct_mean']:.4f}  "
          f"incorrect={results['shadow_stats']['sign_ratio_incorrect_mean']:.4f}")
    print(f"  Neighbor diff: correct={results['shadow_stats']['neighbor_diff_correct_mean']:.2f}  "
          f"incorrect={results['shadow_stats']['neighbor_diff_incorrect_mean']:.2f}")


if __name__ == "__main__":
    main()
