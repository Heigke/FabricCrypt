"""z106 — B.5 NARMA-10 v2: N=100 reservoir + spectral-radius W_rec.

Fixes the z103 deferred-null. Two changes vs z103:
  1. N=100 instead of N=10 (canonical ESN-NARMA scale).
  2. W_rec normalized to spectral radius ρ=0.9 (canonical ESN init)
     instead of fixed 1/√N. This decouples recurrence amplitude
     from N and follows the standard ESN literature.

Same 5-seed paired-t protocol. T=600 (longer for slower
NARMA-10 dynamics).

Wall budget: ~25 min. Two κ × 5 seeds × ~150s/run.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z106_narma10_N100_specrad"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def narma10(u: np.ndarray) -> np.ndarray:
    T = len(u); y = np.zeros(T)
    for t in range(10, T):
        y[t] = (0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-10] * u[t-1] + 0.1)
    return y


def make_W_rec(N: int, rho: float, rng) -> torch.Tensor:
    """W_rec normalized to spectral radius ρ — canonical ESN init."""
    W = rng.normal(0.0, 1.0, size=(N, N))
    eig = np.linalg.eigvals(W)
    rho_W = float(np.max(np.abs(eig)))
    if rho_W < 1e-9:
        return torch.tensor(W, dtype=torch.float64)
    return torch.tensor(W * (rho / rho_W), dtype=torch.float64)


def run_narma10_v2(kappa: float, N: int = 100, T: int = 600, rho: float = 0.9, seed: int = 42):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.random(T)
    y = narma10(u)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4

    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = make_W_rec(N, rho, rng)

    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id_traj = np.zeros((N, T))
    fails = 0
    for t in range(T):
        Vd_t = torch.tensor([0.5 + 2.0 * float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa * recur).clamp(-0.2, 1.0)
        try:
            out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                       max_iters=15, tol=1e-9, verbose=False)
            log_Id = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        except Exception:
            log_Id = np.zeros(N) - 15
            fails += 1
        log_Id_traj[:, t] = log_Id
        feat_prev = torch.tensor(log_Id, dtype=torch.float64)

    warmup = 100
    n_train = int(0.6 * (T - warmup))
    feat_norm = (log_Id_traj - log_Id_traj.mean(axis=1, keepdims=True))
    feat_norm = feat_norm / (feat_norm.std(axis=1, keepdims=True) + 1e-9)
    X = np.hstack([np.ones((T - warmup, 1)), feat_norm[:, warmup:].T])
    y_use = y[warmup:]
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y_use[:n_train], y_use[n_train:]
    ridge = 1e-3
    XtX = X_tr.T @ X_tr; XtY = X_tr.T @ y_tr
    W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
    pred = X_te @ W
    nrmse = float(np.sqrt(((pred - y_te) ** 2).mean()) / max(y_te.std(), 1e-9))
    return {"kappa": kappa, "seed": seed, "nrmse": nrmse, "fails": fails,
            "log_Id_std_mean": float(log_Id_traj.std(axis=1).mean())}


def main():
    t0 = time.time()
    print(f"[z106] starting at {time.strftime('%H:%M:%S')} — NARMA-10 v2 (N=100, ρ=0.9)")
    kappas = [0.00, 0.03]
    seeds = [42, 43, 44, 45, 46]
    grid = np.zeros((len(kappas), len(seeds)))
    detail = {}
    for j, seed in enumerate(seeds):
        for i, kappa in enumerate(kappas):
            ti = time.time()
            r = run_narma10_v2(kappa, seed=seed)
            grid[i, j] = r["nrmse"]
            detail[f"k{kappa:.2f}_s{seed}"] = r
            print(f"  seed={seed}  κ={kappa:.2f}  NRMSE={r['nrmse']:.4f}  "
                   f"feat_std={r['log_Id_std_mean']:.3f}  ({time.time()-ti:.1f}s)", flush=True)
    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z106] === NARMA-10 v2 NRMSE (N=100, T=600, ρ=0.9; lower better) ===")
    print(f"  {'κ':>6s}  {'NRMSE mean':>12s}  {'± std':>7s}  {'± SEM':>7s}  {'min':>6s}  {'max':>6s}")
    for i, kappa in enumerate(kappas):
        print(f"  {kappa:>6.2f}  {means[i]:>12.4f}  {stds[i]:>7.4f}  "
               f"{sems[i]:>7.4f}  {grid[i].min():>6.3f}  {grid[i].max():>6.3f}")
    diffs = grid[1] - grid[0]
    d_mean, d_sem = diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(seeds))
    t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
    print(f"\n[z106] paired Δ NRMSE: {d_mean:+.4f} ± {d_sem:.4f} (t = {t_stat:+.2f}); "
           f"negative = improvement")
    json.dump({"grid": grid.tolist(), "kappas": kappas, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "paired_diff_mean": float(d_mean), "paired_diff_sem": float(d_sem),
                "paired_t": float(t_stat), "detail": detail,
                "config": {"N": 100, "T": 600, "rho": 0.9}},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z106] wall: {time.time()-t0:.1f}s")
    print(f"[z106] z103 reference (N=10, no spec-rad): NRMSE 1.19 / 1.17, t=-2.0")


if __name__ == "__main__":
    main()
