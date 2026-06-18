"""z102 — B.5.c third cut: multi-seed MC averaging at optimal κ.

Honest follow-up to z101 (which produced single-seed peaks of MC=1.09
that don't survive multi-seed averaging). Reports MC ± stddev at the
κ regime where z101 saw consistent lift across N (κ near 0.05).

Sweep:
  κ ∈ {0.00, 0.03, 0.05, 0.07, 0.10}
  N = 10 (the regime where z100/z101 showed clear lift)
  seeds = {42, 43, 44, 45, 46}   (5 seeds → SE = stddev/sqrt(5))
  T = 300

Reports MC mean ± stddev per κ. Goal: extract the brief-grade
"reservoir lift" claim with proper uncertainty bars.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z102_recurrence_multiseed"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z101", ROOT / "scripts/z101_recurrence_kappa_N_sweep.py")
z101 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z101)


def main():
    t0 = time.time()
    print(f"[z102] starting at {time.strftime('%H:%M:%S')} — multi-seed MC at low κ")
    kappas = [0.00, 0.03, 0.05, 0.07, 0.10]
    seeds = [42, 43, 44, 45, 46]
    N = 10
    T = 300
    grid = np.zeros((len(kappas), len(seeds)))
    detail = {}
    for j, seed in enumerate(seeds):
        for i, kappa in enumerate(kappas):
            ti = time.time()
            r = z101.run_recurrent_mc(kappa, N=N, T=T, seed=seed)
            grid[i, j] = r["MC"]
            detail[f"k{kappa:.2f}_s{seed}"] = {
                "MC": r["MC"], "wall_s": float(time.time() - ti)
            }
            print(f"  seed={seed}  κ={kappa:.2f}  MC={r['MC']:6.3f}  "
                   f"({time.time()-ti:.1f}s)", flush=True)

    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))

    print(f"\n[z102] === MULTI-SEED MC (N={N}, T={T}, {len(seeds)} seeds) ===")
    print(f"  {'κ':>6s}  {'MC mean':>9s}  {'± std':>7s}  {'± SEM':>7s}  {'min':>6s}  {'max':>6s}")
    for i, kappa in enumerate(kappas):
        print(f"  {kappa:>6.2f}  {means[i]:>9.3f}  {stds[i]:>7.3f}  "
               f"{sems[i]:>7.3f}  {grid[i].min():>6.3f}  {grid[i].max():>6.3f}")

    # Lift over baseline
    baseline = means[0]   # κ=0
    print(f"\n[z102] κ=0 baseline mean MC = {baseline:.3f}")
    print(f"[z102] LIFT (mean MC - baseline) at each κ:")
    for i, kappa in enumerate(kappas[1:], start=1):
        lift = means[i] - baseline
        # Two-sample t-like: paired across seeds
        diffs = grid[i] - grid[0]
        d_mean, d_std = diffs.mean(), diffs.std(ddof=1)
        d_sem = d_std / np.sqrt(len(seeds))
        t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
        print(f"  κ={kappa:.2f}  Δ={lift:+.3f}  paired Δ mean={d_mean:+.3f} "
               f"(SEM={d_sem:.3f}, t={t_stat:+.2f})")

    json.dump({"grid": grid.tolist(), "kappas": kappas, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "sems": sems.tolist(), "detail": detail},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z102] wall: {time.time()-t0:.1f}s")
    print(f"[z102] saved {OUT}/summary.json")


if __name__ == "__main__":
    main()
