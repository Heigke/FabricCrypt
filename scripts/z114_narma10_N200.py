"""z114 — NARMA-10 N-scaling: N=200 at κ=0.003, ρ=0.9.

z107 found NRMSE 0.946 ± 0.018 at N=100, κ=0.003. Brief Limitations
bullet 1 calls out absolute NRMSE 0.95 vs canonical-ESN literature
0.1-0.3 as the gap. Doubling N is the cheapest test of "does the
gap narrow with reservoir size".

5 seeds × 2 conditions (κ ∈ {0, 0.003}). T=600. Expected wall ~18 min.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z114_narma10_N200"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z106", ROOT / "scripts/z106_narma10_N100_specrad.py")
z106 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z106)


def main():
    t0 = time.time()
    print(f"[z114] NARMA-10 N-scaling — N=200, ρ=0.9, κ ∈ {{0, 0.003}}")
    kappas = [0.000, 0.003]
    seeds = [42, 43, 44, 45, 46]
    grid = np.zeros((len(kappas), len(seeds)))
    for j, seed in enumerate(seeds):
        for i, kappa in enumerate(kappas):
            ti = time.time()
            r = z106.run_narma10_v2(kappa, N=200, T=600, rho=0.9, seed=seed)
            grid[i, j] = r["nrmse"]
            print(f"  seed={seed}  κ={kappa:.3f}  NRMSE={r['nrmse']:.4f}  "
                   f"({time.time()-ti:.1f}s)", flush=True)

    means = grid.mean(axis=1); stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z114] === NARMA-10 NRMSE (N=200, T=600) ===")
    for i, kappa in enumerate(kappas):
        print(f"  κ={kappa:.3f}  NRMSE = {means[i]:.4f} ± {stds[i]:.4f}  "
               f"min={grid[i].min():.3f}  max={grid[i].max():.3f}")

    diffs = grid[1] - grid[0]
    d_mean = diffs.mean(); d_sem = diffs.std(ddof=1) / np.sqrt(len(seeds))
    t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
    print(f"\n[z114] paired Δ NRMSE (κ=0.003 − κ=0): "
           f"{d_mean:+.4f} ± {d_sem:.4f}  (t = {t_stat:+.2f})")
    print(f"\n[z114] === Comparison vs z107 (N=100) ===")
    print(f"  z107 (N=100): κ=0 NRMSE 1.073, κ=0.003 NRMSE 0.946 (paired t=−9.4)")
    print(f"  z114 (N=200): κ=0 NRMSE {means[0]:.3f}, κ=0.003 NRMSE {means[1]:.3f} "
           f"(paired t={t_stat:+.2f})")
    print(f"  Canonical ESN literature: NRMSE ~0.1–0.3 at N≥100.")

    json.dump({"grid": grid.tolist(), "kappas": kappas, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "paired_t": float(t_stat),
                "config": {"N": 200, "T": 600, "rho": 0.9}},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z114] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
