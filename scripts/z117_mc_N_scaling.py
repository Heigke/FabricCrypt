"""z117 — MC N-scaling at κ=0.03, 5 seeds.

Tests whether the architectural ceiling that bounds NARMA-10
(z107/z114-z116, NRMSE ≈0.95 plateau) also bounds memory
capacity, or whether MC scales with N as canonical-reservoir
theory predicts.

z102 anchor: N=10, κ=0.03, 5 seeds → MC = 1.10 ± 0.23 (paired t = +7.4).
z114 NARMA at N=200: NRMSE worse (5.7), no architectural lift.

Sweep:
  N ∈ {10, 30, 100, 200}     (10 is z102 reproduction)
  κ = 0.03                    (z102 optimum)
  seeds = {42, 43, 44, 45, 46}
  T = 300, K = 15

Outcome interpretation:
  - MC scales with N: dichotomy table holds; memory tasks are
    N-bounded but not architecturally-bounded. Reinforces brief.
  - MC plateaus like NARMA: deeper substrate-level limit;
    weakens C.3 multi-cell argument; needs reframing.

Wall budget ~25-40 min (N=10:~2 min, N=30:~5, N=100:~12, N=200:~20 per seed).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z117_mc_N_scaling"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z101", ROOT / "scripts/z101_recurrence_kappa_N_sweep.py")
z101 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z101)


def main():
    t0 = time.time()
    print(f"[z117] starting at {time.strftime('%H:%M:%S')} — MC N-scaling at κ=0.03")
    Ns = [10, 30, 100, 200]
    seeds = [42, 43, 44, 45, 46]
    kappa = 0.03
    T = 300
    grid = np.zeros((len(Ns), len(seeds)))
    detail = {}
    for i, N in enumerate(Ns):
        for j, seed in enumerate(seeds):
            ti = time.time()
            r = z101.run_recurrent_mc(kappa, N=N, T=T, seed=seed)
            grid[i, j] = r["MC"]
            detail[f"N{N}_s{seed}"] = {
                "MC": r["MC"], "wall_s": float(time.time() - ti),
                "fails": int(r["fails"]),
                "feat_std": float(r["log_Id_std_mean"]),
            }
            print(f"  N={N:3d}  seed={seed}  MC={r['MC']:6.3f}  "
                   f"feat_std={r['log_Id_std_mean']:.3f}  "
                   f"fails={r['fails']:3d}  ({time.time()-ti:.1f}s)",
                   flush=True)

    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))

    print(f"\n[z117] === MC vs N (κ={kappa}, T={T}, {len(seeds)} seeds) ===")
    print(f"  {'N':>4s}  {'MC mean':>9s}  {'± std':>7s}  {'± SEM':>7s}  "
           f"{'min':>6s}  {'max':>6s}")
    for i, N in enumerate(Ns):
        print(f"  {N:>4d}  {means[i]:>9.3f}  {stds[i]:>7.3f}  "
               f"{sems[i]:>7.3f}  {grid[i].min():>6.3f}  {grid[i].max():>6.3f}")

    # Pairwise paired-t vs N=10 baseline
    print(f"\n[z117] Paired-t vs N=10 baseline (per-seed paired):")
    base = grid[0]
    for i in range(1, len(Ns)):
        diffs = grid[i] - base
        d_mean = diffs.mean()
        d_sem = diffs.std(ddof=1) / np.sqrt(len(seeds))
        t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
        print(f"  N={Ns[i]:3d} - N=10: Δ={d_mean:+.3f}  SEM={d_sem:.3f}  "
               f"t={t_stat:+.2f}")

    # Power-law fit (log-log)
    log_N = np.log(np.array(Ns, dtype=float))
    log_MC = np.log(np.maximum(means, 1e-6))
    A = np.vstack([log_N, np.ones_like(log_N)]).T
    slope, intercept = np.linalg.lstsq(A, log_MC, rcond=None)[0]
    print(f"\n[z117] Power-law fit: MC ~ N^{slope:.3f} (intercept={intercept:.3f})")
    print(f"  canonical-ESN expectation: slope ~ 0.5-1.0")
    print(f"  pure plateau: slope ~ 0")

    json.dump({"Ns": Ns, "seeds": seeds, "kappa": kappa,
                "grid": grid.tolist(),
                "means": means.tolist(), "stds": stds.tolist(),
                "sems": sems.tolist(),
                "powerlaw_slope": float(slope),
                "powerlaw_intercept": float(intercept),
                "detail": detail},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z117] wall: {time.time()-t0:.1f}s")
    print(f"[z117] saved {OUT}/summary.json")
    print(f"[z117] z102 reference: N=10 κ=0.03 → MC=1.10±0.23")


if __name__ == "__main__":
    main()
