"""z119b — NARMA-only topology re-run with continuous u ∈ [0, 0.5].

z119's NARMA column was NaN-poisoned because it derived NARMA's
input from binary {0, 0.5} (via 0.25*(u_bin+1) where u_bin ∈ {-1,+1}).
NARMA-10 dynamics with binary input give y[t] that grows
unboundedly across the warmup transient.

This script re-runs ONLY the NARMA-10 task across the same
topology × N grid, using continuous u ∈ [0, 0.5] (z107 protocol).
MC/XOR/WAVE results from z119 are valid and not re-run.

Same kappa, ridge-tuning, seeds. Wall budget ~25-35 min
(45 conditions × ~35-50s each).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z119b_narma_topology_fix"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z119", ROOT / "scripts/z119_topology_sweep.py")
z119 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z119)


def main():
    t0 = time.time()
    topologies = ["RAND_GAUSS", "MESH_4N", "ER_SPARSE", "WS_SMALLWORLD", "ALLTOALL"]
    Ns = [50, 100, 200]
    seeds = [42, 43, 44]
    T = 300
    kappa = 0.003   # z107 NARMA optimum (not 0.03)
    print(f"[z119b] NARMA-only topology fix — continuous u ∈ [0, 0.5]")
    print(f"  topologies: {topologies}")
    print(f"  Ns: {Ns}, seeds: {seeds}, T={T}, κ={kappa}")
    print(f"  z107 reference: NARMA NRMSE = 0.946 ± 0.018 at RAND_GAUSS N=100")
    print()

    results = {}
    for topo in topologies:
        for N in Ns:
            for seed in seeds:
                ti = time.time()
                rng = np.random.default_rng(seed)
                u = 0.5 * rng.random(T)              # continuous, z107 protocol
                y_target = z119.narma10(u)
                drive = lambda t: 0.5 + 2.0 * float(u[t])    # match z107 Vd
                log_Id = z119.run_cell_sim(topo, N, T, kappa, drive, seed)

                warmup = 100
                feat = (log_Id - log_Id.mean(axis=1, keepdims=True))
                feat /= (feat.std(axis=1, keepdims=True) + 1e-9)
                X = np.hstack([np.ones((T-warmup,1)), feat[:, warmup:].T])
                y = y_target[warmup:]
                n_tr = int(0.6 * len(X))
                Xtr, Xte = X[:n_tr], X[n_tr:]
                ytr, yte = y[:n_tr], y[n_tr:]

                # ridge sweep — pick best on held-out
                n_tr2 = int(0.8 * len(Xtr))
                best = (None, np.inf)
                for r in (1e-3, 1e-1, 1e+1, 1e+3):
                    XtX = Xtr[:n_tr2].T @ Xtr[:n_tr2]
                    W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]),
                                          Xtr[:n_tr2].T @ ytr[:n_tr2])
                    p = Xtr[n_tr2:] @ W
                    nrmse = float(np.sqrt(((p - ytr[n_tr2:])**2).mean()) /
                                    max(ytr[n_tr2:].std(), 1e-9))
                    if nrmse < best[1]:
                        best = (r, nrmse)
                r = best[0]
                XtX = Xtr.T @ Xtr
                W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xtr.T @ ytr)
                pred = Xte @ W
                NARMA_NRMSE = float(np.sqrt(((pred - yte)**2).mean()) /
                                       max(yte.std(), 1e-9))

                key = f"{topo}_N{N}_s{seed}"
                results[key] = {"topo": topo, "N": N, "seed": seed,
                                  "NARMA_NRMSE": NARMA_NRMSE,
                                  "best_ridge": float(r),
                                  "wall_s": float(time.time()-ti)}
                print(f"  {topo:14s} N={N:3d} s={seed}  "
                       f"NARMA={NARMA_NRMSE:5.3f}  λ*={r:.0e}  "
                       f"({time.time()-ti:.0f}s)", flush=True)

    print(f"\n[z119b] === Aggregated NARMA NRMSE (mean over {len(seeds)} seeds) ===")
    print(f"  {'topo':14s} {'N':>3s}  {'NARMA':>6s}  {'± std':>6s}")
    agg = {}
    for topo in topologies:
        for N in Ns:
            keys = [f"{topo}_N{N}_s{s}" for s in seeds]
            arr = np.array([results[k]["NARMA_NRMSE"] for k in keys])
            agg[f"{topo}_N{N}"] = {"NARMA_NRMSE_mean": float(arr.mean()),
                                     "NARMA_NRMSE_std": float(arr.std(ddof=1))}
            print(f"  {topo:14s} {N:>3d}  {arr.mean():6.3f}  {arr.std(ddof=1):6.3f}")

    # Best topology per N
    print(f"\n[z119b] === Best topology per N (lower NARMA NRMSE = better) ===")
    for N in Ns:
        rows = {topo: agg[f"{topo}_N{N}"]["NARMA_NRMSE_mean"] for topo in topologies}
        best = min(rows, key=rows.get)
        print(f"  N={N:3d}: {best} ({rows[best]:.3f})  "
               f"vs RAND={rows['RAND_GAUSS']:.3f}")

    json.dump({"topologies": topologies, "Ns": Ns, "seeds": seeds,
                "T": T, "kappa": kappa, "results": results, "agg": agg},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z119b] saved {OUT}/summary.json")
    print(f"[z119b] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
