#!/usr/bin/env python3
"""ATTACK 3 — heavy-tail statistics on collected substrate streams.

Per channel per host:
  - α-stable Lévy index (scipy.stats.levy_stable.fit) — tail heaviness
  - Hill estimator on top 5%, 1%, 0.1%
  - Multi-scale kurtosis (per-sample, 10-sample blocks, 100-sample blocks)
  - P99.99 / P50 ratio
  - DFA Hurst exponent — long-range dependence
  - KL(Gaussian-fit ‖ empirical) — measures how badly Gaussian-SW would do

Cross-device:
  - Cohen d on each tail metric (per-device)
  - Are devices distinguishable in heavy-tail space?

Output: results/IDENTITY_BENCHMARK_2026-05-30/attack_1_3/A3_tail_stats.json
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[2] / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "attack_1_3"

try:
    from scipy.stats import levy_stable, kurtosis, entropy
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------------------------------------------------------
# Estimators
# -----------------------------------------------------------------------------
def hill_estimator(x: np.ndarray, frac: float = 0.05) -> float:
    """Hill tail-index estimator on upper tail. Returns α (smaller = heavier)."""
    x = np.abs(x[np.isfinite(x)])
    x = x[x > 0]
    if x.size < 50:
        return float("nan")
    x_sorted = np.sort(x)[::-1]
    k = max(int(len(x_sorted) * frac), 10)
    top = x_sorted[:k]
    xmin = x_sorted[k - 1]
    if xmin <= 0:
        return float("nan")
    logs = np.log(top / xmin)
    inv_alpha = float(logs[:-1].mean())  # exclude the boundary
    if inv_alpha <= 0:
        return float("nan")
    return 1.0 / inv_alpha


def multi_scale_kurtosis(x: np.ndarray, scales=(1, 10, 100)) -> dict:
    out = {}
    for s in scales:
        if s == 1:
            xs = x
        else:
            n = (x.size // s) * s
            xs = x[:n].reshape(-1, s).mean(axis=1)
        if xs.size < 10:
            out[f"k{s}"] = float("nan")
        else:
            if HAS_SCIPY:
                out[f"k{s}"] = float(kurtosis(xs, fisher=False))  # raw (Normal=3)
            else:
                m = xs.mean()
                v = xs.var()
                out[f"k{s}"] = float(((xs - m) ** 4).mean() / (v ** 2 + 1e-30))
    return out


def percentile_ratio(x: np.ndarray) -> dict:
    if x.size < 100:
        return {"p99_99_over_p50": float("nan"), "p99_over_p50": float("nan")}
    ax = np.abs(x)
    p50 = float(np.median(ax) + 1e-30)
    p99 = float(np.percentile(ax, 99))
    p99_99 = float(np.percentile(ax, 99.99)) if x.size > 10000 else float(np.percentile(ax, 99.9))
    return {"p99_over_p50": p99 / p50, "p99_99_over_p50": p99_99 / p50}


def dfa_hurst(x: np.ndarray, scales=None) -> float:
    """Detrended Fluctuation Analysis. Returns Hurst exponent."""
    x = x - x.mean()
    y = np.cumsum(x)
    n = y.size
    if scales is None:
        scales = np.unique(np.logspace(np.log10(8), np.log10(min(n // 4, 1024)), 12).astype(int))
    F = []
    for s in scales:
        if s < 4 or s > n // 2:
            continue
        n_seg = n // s
        seg = y[:n_seg * s].reshape(n_seg, s)
        t = np.arange(s)
        # linear detrend per segment
        rms = []
        for row in seg:
            p = np.polyfit(t, row, 1)
            resid = row - np.polyval(p, t)
            rms.append(np.sqrt(np.mean(resid ** 2)))
        F.append(np.mean(rms))
    if len(F) < 3:
        return float("nan")
    F = np.array(F)
    scales = np.array(scales[:len(F)])
    mask = (F > 0) & (scales > 0)
    if mask.sum() < 3:
        return float("nan")
    coef = np.polyfit(np.log(scales[mask]), np.log(F[mask]), 1)
    return float(coef[0])


def gaussian_kl_to_empirical(x: np.ndarray, n_bins: int = 80) -> float:
    """KL(empirical ‖ Gaussian) measuring how poorly a Gaussian-SW would mimic.

    Higher = harder for Gaussian to replicate.
    """
    x = x[np.isfinite(x)]
    if x.size < 100:
        return float("nan")
    mu, sd = x.mean(), x.std() + 1e-12
    # build histogram on shared support
    lo = float(min(x.min(), mu - 5 * sd))
    hi = float(max(x.max(), mu + 5 * sd))
    bins = np.linspace(lo, hi, n_bins + 1)
    p_emp, _ = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (bins[1:] + bins[:-1])
    p_gauss = np.exp(-0.5 * ((centers - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    # normalize both to PMFs across bins
    w = bins[1:] - bins[:-1]
    p_emp_pmf = p_emp * w + 1e-12
    p_emp_pmf /= p_emp_pmf.sum()
    p_gauss_pmf = p_gauss * w + 1e-12
    p_gauss_pmf /= p_gauss_pmf.sum()
    kl = float(np.sum(p_emp_pmf * np.log(p_emp_pmf / p_gauss_pmf)))
    return kl


def levy_alpha(x: np.ndarray) -> float | None:
    if not HAS_SCIPY:
        return None
    x = x[np.isfinite(x)]
    if x.size > 5000:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=5000, replace=False)
    try:
        # robust: fit assuming symmetric stable around median
        x_c = x - np.median(x)
        params = levy_stable._fitstart(x_c)
        return float(params[0])
    except Exception:
        try:
            params = levy_stable.fit(x[:2000], floc=0)
            return float(params[0])
        except Exception:
            return float("nan")


def analyze_channel(x: np.ndarray) -> dict:
    out = {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }
    out.update(percentile_ratio(x))
    out.update(multi_scale_kurtosis(x))
    out["hill_p05"] = hill_estimator(x, 0.05)
    out["hill_p01"] = hill_estimator(x, 0.01)
    out["hill_p001"] = hill_estimator(x, 0.001)
    out["dfa_hurst"] = dfa_hurst(x)
    out["kl_gauss"] = gaussian_kl_to_empirical(x)
    out["levy_alpha"] = levy_alpha(x)
    return out


# -----------------------------------------------------------------------------
# Cross-device comparison
# -----------------------------------------------------------------------------
def cohen_d_scalar(a: float, b: float, sd_a: float, sd_b: float) -> float:
    """Single-point Cohen d using analytical std estimates."""
    pooled = math.sqrt(0.5 * (sd_a ** 2 + sd_b ** 2)) + 1e-12
    return (a - b) / pooled


def cross_device(stats_i: dict, stats_d: dict) -> dict:
    """For each channel, compute pseudo-d across devices.

    Since we have one value per host per metric, we use a heuristic:
    relative gap |a-b| / mean(|a|,|b|). Real d would need multi-segment.
    """
    out = {}
    for ch in stats_i:
        if ch not in stats_d:
            continue
        gaps = {}
        for k in stats_i[ch]:
            if k in ("n",):
                continue
            a, b = stats_i[ch][k], stats_d[ch][k]
            if a is None or b is None:
                continue
            if isinstance(a, float) and (math.isnan(a) or math.isnan(b)):
                continue
            try:
                denom = (abs(a) + abs(b)) / 2 + 1e-12
                rel = abs(a - b) / denom
                gaps[k] = {"a": a, "b": b, "abs_diff": abs(a - b), "rel_diff": rel}
            except Exception:
                pass
        out[ch] = gaps
    return out


def main():
    print(f"[A3.analyze] scipy={HAS_SCIPY}", flush=True)
    t0 = time.time()
    streams = {}
    for host in ("ikaros", "daedalus"):
        p = OUT_DIR / f"A3_streams_{host}.npz"
        if not p.exists():
            print(f"[A3.analyze] MISSING {p}", flush=True)
            continue
        d = np.load(p)
        streams[host] = {k: d[k] for k in d.files}
        sizes = {k: int(v.size) for k, v in streams[host].items()}
        print(f"[A3.analyze] {host} channels={sizes}", flush=True)

    stats = {}
    for host, chs in streams.items():
        stats[host] = {}
        for ch, x in chs.items():
            print(f"[A3.analyze] {host}/{ch} (n={x.size})…", flush=True)
            stats[host][ch] = analyze_channel(x.astype(np.float64))

    cross = {}
    if "ikaros" in stats and "daedalus" in stats:
        cross = cross_device(stats["ikaros"], stats["daedalus"])

    out = {
        "stats": stats,
        "cross_device": cross,
        "wall_s": time.time() - t0,
    }
    with open(OUT_DIR / "A3_tail_stats.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[A3.analyze] saved → A3_tail_stats.json wall={time.time() - t0:.1f}s", flush=True)
    # quick summary
    if cross:
        print("\n=== TOP CROSS-DEVICE GAPS (rel_diff) ===")
        rows = []
        for ch, gd in cross.items():
            for k, v in gd.items():
                rows.append((v.get("rel_diff", 0), ch, k, v.get("a"), v.get("b")))
        rows.sort(reverse=True)
        for r in rows[:15]:
            print(f"  {r[1]}/{r[2]}: rel={r[0]:.3f}  i={r[3]}  d={r[4]}")


if __name__ == "__main__":
    main()
