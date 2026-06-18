"""z111 — Hopfield M-scaling at N=50, κ=0.

Closes the storage-capacity axis. z108 confirmed substrate-alone
reaches perfect classification at N=50, M=3. This script sweeps
M ∈ {3, 5, 10, 15, 20} at N=50 to find the storage cutoff.

Standard Hopfield capacity is M_max ≈ 0.14·N for orthogonal
patterns; for our random patterns and ridge readout, the cutoff
is empirical.

5 seeds, T=300, p_flip=0.20.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z111_hopfield_M_scaling"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z105", ROOT / "scripts/z105_hopfield_pattern_recall.py")
z105 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z105)


def main():
    t0 = time.time()
    print(f"[z111] Hopfield M-scaling at N=50, κ=0")
    Ms = [3, 5, 10, 15, 20]
    seeds = [42, 43, 44, 45, 46]
    grid = np.zeros((len(Ms), len(seeds)))
    detail = {}
    for i, M in enumerate(Ms):
        chance = 1.0 / M
        for j, seed in enumerate(seeds):
            ti = time.time()
            r = z105.run_hopfield(0.0, N=50, M=M, seed=seed)
            grid[i, j] = r["acc"]
            detail[f"M{M}_s{seed}"] = r
            print(f"  M={M:2d}  seed={seed}  acc={r['acc']:.3f}  "
                   f"(chance {chance:.3f}, ratio {r['acc']/chance:.2f}×)  "
                   f"({time.time()-ti:.1f}s)", flush=True)
    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z111] === Hopfield M-scaling at N=50, κ=0, p_flip=0.20 ===")
    print(f"  {'M':>3s}  {'chance':>7s}  {'acc mean':>10s}  {'± std':>7s}  {'ratio':>6s}")
    for i, M in enumerate(Ms):
        chance = 1.0 / M
        print(f"  {M:>3d}  {chance:>7.3f}  {means[i]:>10.3f}  {stds[i]:>7.3f}  "
               f"{means[i]/chance:>5.2f}×")
    json.dump({"grid": grid.tolist(), "Ms": Ms, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "sems": sems.tolist(), "detail": detail},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z111] wall: {time.time()-t0:.1f}s")
    print(f"[z111] z108 ref: M=3 at N=50 → 1.000±0.000 (perfect)")


if __name__ == "__main__":
    main()
