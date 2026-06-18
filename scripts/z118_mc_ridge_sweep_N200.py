"""z118 — MC ridge sweep at N=200, κ=0.03.

z117 found MC degrades with N (N=10→1.10, N=200→0.40, t=−4.64).
Two candidate explanations:
  (i)  Architectural ceiling on independent state.
  (ii) Ridge underregularised at N=200 / T≈140 train samples
       (problem under-determined; ridge=1e-3 absorbs too much).

This script sweeps ridge ∈ {1e-6, 1e-4, 1e-2, 1e0, 1e1, 1e2}
at N=200, 5 seeds. If MC recovers to ≈1.0 at some ridge, the
brief's dichotomy table may be salvaged with a "readout-tuned"
caveat. If MC stays ≤ 0.5 across all ridges, the architectural
ceiling for memory tasks is confirmed and the brief must be
revised.

Wall budget ~15-20 min (one cell sim per seed reused; just
re-fit readout per ridge).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z118_mc_ridge_sweep_N200"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def collect_features(N, T, kappa, seed):
    """Run cell once, return log10|Id| trajectory. (Same as z101.run_recurrent_mc up to fit.)"""
    rng = np.random.default_rng(seed)
    u = 2.0 * rng.integers(0, 2, size=T) - 1.0
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = torch.tensor(rng.normal(0.0, 1.0/np.sqrt(N), size=(N, N)), dtype=torch.float64)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([1.0 + 0.5 * float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa * recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                   max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    return log_Id, u


def fit_mc(log_Id, u, ridge, K=15, warmup=50):
    T = log_Id.shape[1]
    n_train = int(0.6 * (T - warmup - K))
    n_test = T - warmup - n_train - K
    feat = (log_Id - log_Id.mean(axis=1, keepdims=True))
    feat /= (feat.std(axis=1, keepdims=True) + 1e-9)
    mc_per_k = np.zeros(K)
    for k in range(1, K + 1):
        t_idx = np.arange(warmup, warmup + n_train + n_test)
        X = np.hstack([np.ones((len(t_idx), 1)), feat[:, t_idx].T])
        y = u[t_idx - k]
        Xtr, Xte = X[:n_train], X[n_train:n_train + n_test]
        ytr, yte = y[:n_train], y[n_train:n_train + n_test]
        XtX = Xtr.T @ Xtr
        W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), Xtr.T @ ytr)
        pred = Xte @ W
        if pred.std() > 1e-12 and yte.std() > 1e-12:
            r = np.corrcoef(pred, yte)[0, 1]
            mc_per_k[k-1] = float(r * r)
    return float(mc_per_k.sum())


def main():
    t0 = time.time()
    print(f"[z118] MC ridge-sweep at N=200, κ=0.03, T=300, K=15")
    print(f"  z117 reference at ridge=1e-3: MC = 0.396 ± 0.166\n")
    seeds = [42, 43, 44, 45, 46]
    ridges = [1e-6, 1e-4, 1e-2, 1e0, 1e1, 1e2]
    grid = np.zeros((len(ridges), len(seeds)))

    # Reuse one cell run per seed across all ridges (purely a readout sweep)
    for j, seed in enumerate(seeds):
        ti = time.time()
        log_Id, u = collect_features(N=200, T=300, kappa=0.03, seed=seed)
        sim_t = time.time() - ti
        print(f"  seed={seed} cell-sim {sim_t:.1f}s; readout sweep...", flush=True)
        for i, r in enumerate(ridges):
            mc = fit_mc(log_Id, u, ridge=r)
            grid[i, j] = mc
            print(f"    ridge={r:.0e}  MC={mc:6.3f}", flush=True)

    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z118] === MC vs ridge (N=200, κ=0.03, 5 seeds) ===")
    print(f"  {'ridge':>8s}  {'MC mean':>9s}  {'± std':>7s}  {'± SEM':>7s}  {'min':>6s}  {'max':>6s}")
    for i, r in enumerate(ridges):
        print(f"  {r:>8.0e}  {means[i]:>9.3f}  {stds[i]:>7.3f}  "
               f"{sems[i]:>7.3f}  {grid[i].min():>6.3f}  {grid[i].max():>6.3f}")

    best_i = int(np.argmax(means))
    print(f"\n[z118] Best ridge: {ridges[best_i]:.0e} → MC = {means[best_i]:.3f} ± {sems[best_i]:.3f}")
    print(f"[z118] z117 N=10 reference: MC = 1.104 ± 0.101 (SEM)")
    print(f"[z118] Verdict:")
    if means[best_i] > 0.9:
        print(f"  → ridge tuning RECOVERS MC at N=200; brief dichotomy survives with caveat.")
    elif means[best_i] > 0.6:
        print(f"  → partial recovery; MC is ridge-sensitive but ceiling is real.")
    else:
        print(f"  → MC stays ≤ 0.6 across all ridges; ARCHITECTURAL CEILING confirmed.")

    json.dump({"seeds": seeds, "ridges": ridges,
                "grid": grid.tolist(),
                "means": means.tolist(), "stds": stds.tolist(),
                "sems": sems.tolist(),
                "best_ridge": float(ridges[best_i]),
                "best_MC_mean": float(means[best_i])},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z118] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
