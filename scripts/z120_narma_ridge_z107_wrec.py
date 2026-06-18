"""z120 — NARMA-10 ridge sweep at N=200 with z107 W_rec construction.

z119b ran NARMA across 5 topologies but used z119's `build_W()`
which scales to spectral-radius ρ=0.9. z107's canonical setting
used `rng.normal(0, 1/√N)` without explicit ρ scaling.

This script replicates z119b for two topologies (RAND_GAUSS and
ER_SPARSE) at N=200 using the z107 W_rec construction, sweeping
ridge to test whether:
  - ER_SPARSE win on NARMA replicates in canonical reservoir setting.
  - Absolute NARMA NRMSE recovers towards z107's 0.946 (was > 1.0
    in z119b due to ρ-scaling difference).

Sweep:
  topologies: RAND_GAUSS (z107 baseline), ER_SPARSE p=0.1
  N = 200
  T = 600 (z107 protocol, longer than z119's T=300)
  κ = 0.003 (z107)
  seeds = {42, 43, 44, 45, 46}
  ridges = {1e-3, 1e-1, 1e+1, 1e+3}

Wall budget: 2 topo × 5 seeds × ~50s = ~10 min cell-sims +
fast readout sweep.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z120_narma_ridge_z107_wrec"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def narma10(u):
    T = len(u); y = np.zeros(T)
    for t in range(10, T):
        y[t] = (0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-10:t])
                + 1.5*u[t-10]*u[t-1] + 0.1)
    return y


def build_W_z107(topology, N, rng):
    """z107-style W_rec: 1/√N Gaussian or sparse mask, NO explicit ρ scaling."""
    if topology == "RAND_GAUSS":
        return rng.normal(0.0, 1.0/np.sqrt(N), size=(N, N))
    elif topology == "ER_SPARSE":
        p = 0.1
        mask = rng.random((N, N)) < p
        # Per-edge std 1/√N scales total spectral mass with sparsity.
        return np.where(mask, rng.normal(0.0, 1.0/np.sqrt(N*p), size=(N, N)), 0.0)
    raise ValueError(topology)


def run_cell_sim(topology, N, T, kappa, seed):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.random(T)
    y = narma10(u)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = torch.tensor(build_W_z107(topology, N, rng), dtype=torch.float64)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([0.5 + 2.0*float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa*recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                   max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    return log_Id, u, y


def fit_narma_ridge_sweep(log_Id, y, T, ridges):
    warmup = 100
    feat = (log_Id - log_Id.mean(axis=1, keepdims=True))
    feat /= (feat.std(axis=1, keepdims=True) + 1e-9)
    X = np.hstack([np.ones((T-warmup,1)), feat[:, warmup:].T])
    yu = y[warmup:]
    n_tr = int(0.6 * len(X))
    Xtr, Xte = X[:n_tr], X[n_tr:]
    ytr, yte = yu[:n_tr], yu[n_tr:]
    out = {}
    for r in ridges:
        XtX = Xtr.T @ Xtr
        W = np.linalg.solve(XtX + r*np.eye(XtX.shape[0]), Xtr.T @ ytr)
        pred = Xte @ W
        nrmse = float(np.sqrt(((pred - yte)**2).mean()) / max(yte.std(), 1e-9))
        out[r] = nrmse
    return out


def main():
    t0 = time.time()
    topologies = ["RAND_GAUSS", "ER_SPARSE"]
    seeds = [42, 43, 44, 45, 46]
    ridges = [1e-3, 1e-1, 1e+1, 1e+3]
    N = 200; T = 600; kappa = 0.003
    print(f"[z120] NARMA-10 ridge sweep at N={N} with z107 W_rec construction")
    print(f"  topologies: {topologies}  seeds: {seeds}  T={T}  κ={kappa}")
    print(f"  z107 reference: NRMSE = 0.946 ± 0.018 at N=100 RAND_GAUSS\n")

    grid = {}
    for topo in topologies:
        for seed in seeds:
            ti = time.time()
            log_Id, u, y = run_cell_sim(topo, N, T, kappa, seed)
            sim_t = time.time() - ti
            ridge_results = fit_narma_ridge_sweep(log_Id, y, T, ridges)
            for r, nrmse in ridge_results.items():
                grid[(topo, seed, r)] = nrmse
            best_r = min(ridge_results, key=ridge_results.get)
            print(f"  {topo:11s} s={seed}  sim={sim_t:.0f}s  "
                   f"ridge sweep: " +
                   "  ".join(f"λ={r:.0e}→{ridge_results[r]:5.3f}" for r in ridges) +
                   f"  best λ*={best_r:.0e}", flush=True)

    print(f"\n[z120] === Aggregated NARMA NRMSE per (topology, ridge) ===")
    print(f"  {'topo':11s} {'ridge':>7s}  {'mean':>6s}  {'± std':>6s}  best of 5 seeds")
    agg = {}
    for topo in topologies:
        for r in ridges:
            arr = np.array([grid[(topo, s, r)] for s in seeds])
            agg[f"{topo}_r{r:.0e}"] = {"mean": float(arr.mean()),
                                          "std": float(arr.std(ddof=1))}
            print(f"  {topo:11s} {r:>7.0e}  {arr.mean():6.3f}  {arr.std(ddof=1):6.3f}")

    print(f"\n[z120] === Best (topology, ridge) per topology ===")
    for topo in topologies:
        best_r = min(ridges, key=lambda r: agg[f"{topo}_r{r:.0e}"]["mean"])
        a = agg[f"{topo}_r{best_r:.0e}"]
        print(f"  {topo}: best ridge = {best_r:.0e}, "
               f"mean NRMSE = {a['mean']:.3f} ± {a['std']:.3f}")

    # Topology comparison at each topology's best ridge
    a_rand = agg[f"RAND_GAUSS_r" + f"{min(ridges, key=lambda r: agg[f'RAND_GAUSS_r{r:.0e}']['mean']):.0e}"]
    a_sparse = agg[f"ER_SPARSE_r" + f"{min(ridges, key=lambda r: agg[f'ER_SPARSE_r{r:.0e}']['mean']):.0e}"]
    delta = a_sparse["mean"] - a_rand["mean"]
    print(f"\n[z120] ER_SPARSE - RAND_GAUSS (at each's best ridge): "
           f"Δ = {delta:+.3f}")
    if delta < -0.05:
        print(f"  → ER_SPARSE BEATS RAND_GAUSS — z119b ordering replicates in z107 W_rec.")
    elif delta > 0.05:
        print(f"  → ER_SPARSE LOSES — z119b advantage was a ρ-scaling artefact.")
    else:
        print(f"  → tie within noise — topology effect is W_rec-scaling-dependent.")

    json.dump({"topologies": topologies, "seeds": seeds, "ridges": ridges,
                "N": N, "T": T, "kappa": kappa,
                "grid": {f"{k[0]}_s{k[1]}_r{k[2]:.0e}": v for k, v in grid.items()},
                "agg": agg},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z120] saved {OUT}/summary.json")
    print(f"[z120] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
