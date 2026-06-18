"""z107 — B.5 NARMA-10 v3: finer κ sweep at N=100, ρ=0.9.

z106 over-drove at κ=0.03. Sweep below the over-drive point:
  κ ∈ {0.001, 0.003, 0.005, 0.010}
  N=100, T=600, ρ=0.9, 5 seeds.

If any condition produces NRMSE clearly < 1.0 with paired-t significant
vs κ=0, NARMA-10 graduates from "deferred" to a brief-grade 4th
positive benchmark. Wall budget: ~50 min (5 seeds × (4+1) κ × ~55s).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z107_narma10_finer_kappa"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z106", ROOT / "scripts/z106_narma10_N100_specrad.py")
z106 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z106)


def main():
    t0 = time.time()
    print(f"[z107] starting at {time.strftime('%H:%M:%S')} — NARMA-10 v3 finer κ sweep")
    kappas = [0.000, 0.001, 0.003, 0.005, 0.010]
    seeds = [42, 43, 44, 45, 46]
    grid = np.zeros((len(kappas), len(seeds)))
    detail = {}
    for j, seed in enumerate(seeds):
        for i, kappa in enumerate(kappas):
            ti = time.time()
            r = z106.run_narma10_v2(kappa, seed=seed)
            grid[i, j] = r["nrmse"]
            detail[f"k{kappa:.3f}_s{seed}"] = r
            print(f"  seed={seed}  κ={kappa:.3f}  NRMSE={r['nrmse']:.4f}  "
                   f"feat_std={r['log_Id_std_mean']:.3f}  ({time.time()-ti:.1f}s)", flush=True)
    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z107] === NARMA-10 v3 NRMSE (N=100, T=600, ρ=0.9; lower better) ===")
    print(f"  {'κ':>7s}  {'NRMSE mean':>12s}  {'± std':>7s}  {'± SEM':>7s}  {'min':>6s}  {'max':>6s}")
    for i, kappa in enumerate(kappas):
        print(f"  {kappa:>7.3f}  {means[i]:>12.4f}  {stds[i]:>7.4f}  "
               f"{sems[i]:>7.4f}  {grid[i].min():>6.3f}  {grid[i].max():>6.3f}")
    # Paired Δ vs κ=0 baseline (kappas[0])
    print(f"\n[z107] paired Δ NRMSE vs κ=0 (negative = improvement):")
    for i, kappa in enumerate(kappas[1:], start=1):
        diffs = grid[i] - grid[0]
        d_mean, d_sem = diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(seeds))
        t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
        print(f"  κ={kappa:.3f}  Δ={d_mean:+.4f} ± {d_sem:.4f}  t={t_stat:+.2f}")
    json.dump({"grid": grid.tolist(), "kappas": kappas, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "config": {"N": 100, "T": 600, "rho": 0.9},
                "detail": detail},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z107] wall: {time.time()-t0:.1f}s")
    print(f"[z107] z106 reference (κ=0.03 over-drove): NRMSE 1.07 / 1.93, t=+1.7")


if __name__ == "__main__":
    main()
