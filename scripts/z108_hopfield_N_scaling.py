"""z108 — B.5 Hopfield N-scaling at κ=0.

Resolves Limitations bullet 2 / C.3 risk #2: confirm that the
substrate-alone Hopfield advantage observed at N=10 (acc 0.69 vs
chance 0.33) holds at N=30 and N=50.

Sweep:
  N ∈ {10, 30, 50}
  κ = 0 (substrate alone — recurrence hurts per z105)
  M = 3 prototype patterns, p_flip = 0.20 corruption
  T = 300 trials
  5 seeds.

Reports per-N accuracy mean ± std.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z108_hopfield_N_scaling"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z105", ROOT / "scripts/z105_hopfield_pattern_recall.py")
z105 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z105)


def main():
    t0 = time.time()
    print(f"[z108] starting at {time.strftime('%H:%M:%S')} — Hopfield N-scaling at κ=0")
    Ns = [10, 30, 50]
    seeds = [42, 43, 44, 45, 46]
    grid = np.zeros((len(Ns), len(seeds)))
    detail = {}
    for i, N in enumerate(Ns):
        for j, seed in enumerate(seeds):
            ti = time.time()
            r = z105.run_hopfield(0.0, N=N, seed=seed)
            grid[i, j] = r["acc"]
            detail[f"N{N}_s{seed}"] = r
            print(f"  N={N:3d}  seed={seed}  acc={r['acc']:.3f}  "
                   f"({time.time()-ti:.1f}s)", flush=True)
    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z108] === Hopfield κ=0 (M=3, p_flip=0.20, chance=0.333) ===")
    print(f"  {'N':>4s}  {'acc mean':>10s}  {'± std':>7s}  {'± SEM':>7s}  {'min':>6s}  {'max':>6s}")
    for i, N in enumerate(Ns):
        print(f"  {N:>4d}  {means[i]:>10.3f}  {stds[i]:>7.3f}  "
               f"{sems[i]:>7.3f}  {grid[i].min():>6.3f}  {grid[i].max():>6.3f}")
    json.dump({"grid": grid.tolist(), "Ns": Ns, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "sems": sems.tolist(), "detail": detail},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z108] wall: {time.time()-t0:.1f}s")
    print(f"[z108] z105 reference at N=10 κ=0: acc 0.686 ± 0.070")


if __name__ == "__main__":
    main()
