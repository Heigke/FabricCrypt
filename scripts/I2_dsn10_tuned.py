"""I2 — DS-N10 sine classification in I1's responsive regime.

Best responsive point from I1: VG1=0.51, VG2=0.585, Vb=0.7, sens≈5%.

Compares two NSRAM reservoir configs on 4-class sine classification:
  - BASELINE: original DS-N10 (VG1∈[0.6,0.78], VG2∈[0.35,0.55], V_reset=0.30)
  - TUNED   : narrow ranges around responsive point + V_reset=0.65

n_seeds = 5. Smaller N (2000) for runtime budget; same readout dim (256).

Verify: input-driven rate variance > 50× across input frequencies.

Outputs:
  results/I2_dsn10_tuned/summary.json
  results/I2_dsn10_tuned/sine_acc.png
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "I2_dsn10_tuned"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

from DS_N10_reservoir import (  # noqa: E402
    NSRAMReservoir, sine_dataset, ridge_train, ridge_predict,
)


def measure_rate_variance(reservoir, freqs=(0.05, 0.10, 0.18, 0.25),
                           snippet_len=300):
    """Run snippets at each of `freqs`, measure mean spike rate, return variance ratio."""
    rates = []
    for f in freqs:
        reservoir.reset()
        t = np.arange(snippet_len)
        u = np.sin(2 * np.pi * f * t)
        # Track spikes via Vb returning to reset
        Vb_prev = reservoir.Vb.copy()
        spikes = 0
        feats = reservoir.run(u)
        # Approximate rate from |dVb| > 0.1 events using readout cells
        diffs = np.abs(np.diff(feats, axis=0)) > 0.15
        rate = diffs.sum() / (snippet_len * reservoir.n_readout)
        rates.append(rate)
    rates = np.array(rates)
    rmax = float(rates.max()); rmin = float(max(rates.min(), 1e-12))
    return {
        "rates": rates.tolist(),
        "freqs": list(freqs),
        "ratio_max_min": rmax / rmin,
        "std": float(rates.std()),
        "mean": float(rates.mean()),
    }


def run_sine_classification(reservoir, seed):
    X, y = sine_dataset(n_classes=4, n_per_class=30, snippet_len=150, seed=seed)
    n = len(y); n_train = int(0.7 * n)
    feats = np.zeros((n, reservoir.n_readout), dtype=np.float64)
    for i in range(n):
        reservoir.reset()
        s = reservoir.run(X[i])
        feats[i] = s.mean(axis=0)
    n_classes = int(y.max() + 1)
    Y_oh = np.zeros((n, n_classes)); Y_oh[np.arange(n), y] = 1.0
    W = ridge_train(feats[:n_train], Y_oh[:n_train], alpha=1e-2)
    Yhat = ridge_predict(feats[n_train:], W)
    return float((Yhat.argmax(axis=1) == y[n_train:]).mean())


def make_baseline(seed, N=2000, n_readout=256):
    return NSRAMReservoir(N=N, n_readout=n_readout, seed=seed)


def make_tuned(seed, N=2000, n_readout=256):
    return NSRAMReservoir(
        N=N, n_readout=n_readout, seed=seed,
        VG1_range=(0.48, 0.54),         # around I1 best VG1=0.51
        VG2_base_range=(0.55, 0.60),    # around I1 best VG2=0.585 (LUT cap 0.6)
        V_reset=0.65,                   # near responsive Vb=0.7
        V_th=0.85,                      # higher threshold to delay reset
        Vd_bias=0.75, Vd_gain=0.5,      # smaller swing centered on responsive Vd
        VG2_fb_gain=0.05,               # less recurrent kick (small VG2 budget)
    )


SEEDS = [0, 1, 2, 3, 4]
FREQS = (0.05, 0.10, 0.18, 0.25)

results = {"BASELINE": {"acc": [], "rate_var": []},
           "TUNED": {"acc": [], "rate_var": []}}
t0 = time.time()
for seed in SEEDS:
    for name, mk in [("BASELINE", make_baseline), ("TUNED", make_tuned)]:
        r = mk(seed)
        rv = measure_rate_variance(r, freqs=FREQS)
        acc = run_sine_classification(r, seed)
        results[name]["acc"].append(acc)
        results[name]["rate_var"].append(rv)
        print(f"seed={seed} {name:8s} acc={acc:.3f} rate_ratio={rv['ratio_max_min']:.1f}",
              flush=True)

elapsed = time.time() - t0

acc_b = np.array(results["BASELINE"]["acc"])
acc_t = np.array(results["TUNED"]["acc"])
ratios_b = np.array([r["ratio_max_min"] for r in results["BASELINE"]["rate_var"]])
ratios_t = np.array([r["ratio_max_min"] for r in results["TUNED"]["rate_var"]])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
axes[0].boxplot([acc_b, acc_t], labels=["BASELINE", "TUNED"])
axes[0].axhline(0.978, color="r", ls="--", label="DS-N10 published 97.8%")
axes[0].set_ylabel("accuracy"); axes[0].set_title("Sine 4-class accuracy")
axes[0].legend(fontsize=8)
axes[1].boxplot([ratios_b, ratios_t], labels=["BASELINE", "TUNED"])
axes[1].axhline(50.0, color="r", ls="--", label="gate 50×")
axes[1].set_ylabel("rate_max/rate_min across freqs")
axes[1].set_title("Input-coupling rate variance")
axes[1].legend(fontsize=8)
fig.suptitle("I2 — DS-N10 baseline vs I1-tuned bias")
fig.savefig(OUT / "sine_acc.png", dpi=130)
plt.close(fig)

summary = {
    "N_cells": 2000, "n_readout": 256, "n_seeds": len(SEEDS),
    "freqs": list(FREQS), "wall_s": elapsed,
    "baseline_published_ref": 0.978,
    "BASELINE": {
        "acc_mean": float(acc_b.mean()), "acc_std": float(acc_b.std()),
        "acc_seeds": acc_b.tolist(),
        "rate_ratio_mean": float(ratios_b.mean()),
        "rate_ratio_seeds": ratios_b.tolist(),
    },
    "TUNED": {
        "acc_mean": float(acc_t.mean()), "acc_std": float(acc_t.std()),
        "acc_seeds": acc_t.tolist(),
        "rate_ratio_mean": float(ratios_t.mean()),
        "rate_ratio_seeds": ratios_t.tolist(),
    },
    "gate": {
        "tuned_rate_ratio_gt_50": bool(ratios_t.mean() > 50.0),
        "baseline_rate_ratio_gt_50": bool(ratios_b.mean() > 50.0),
    },
    "tuned_bias_from_I1": {"VG1": 0.51, "VG2": 0.585, "Vb_target": 0.7},
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
