"""z103 — B.5 second benchmark: NARMA-10 with software recurrence.

Same protocol as z102: 5 seeds, N=10 cells, T=300, κ=0.03.
But with NARMA-10 target instead of memory capacity. Reports paired-t
between κ=0 baseline and κ=0.03 condition.

NARMA-10 task:
  u(t) ~ U(0, 0.5) input
  y(t) = 0.3·y(t-1) + 0.05·y(t-1)·sum(y[t-10:t]) + 1.5·u(t-10)·u(t-1) + 0.1
  Reservoir features = log10|Id_i(t)| per cell
  Linear ridge readout predicts y(t)
  Metric: NRMSE on test split (lower = better)

For brief consistency we report:
  - NRMSE mean ± std for κ=0 vs κ=0.03
  - paired Δ NRMSE (negative = improvement)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z103_narma10_recurrence_multiseed"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def narma10(u: np.ndarray) -> np.ndarray:
    T = len(u)
    y = np.zeros(T)
    for t in range(10, T):
        y[t] = (0.3 * y[t-1]
                + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-10] * u[t-1]
                + 0.1)
    return y


def run_narma10(kappa: float, N: int = 10, T: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.random(T)
    y = narma10(u)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4

    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = torch.tensor(rng.normal(0.0, 1.0 / np.sqrt(N), size=(N, N)), dtype=torch.float64)

    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id_traj = np.zeros((N, T))
    fails = 0
    for t in range(T):
        # Drive Vd in [0.5, 1.5] from u ∈ [0, 0.5]
        Vd_t = torch.tensor([0.5 + 2.0 * float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa * recur).clamp(-0.2, 1.0)
        try:
            out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                       max_iters=15, tol=1e-9, verbose=False)
            Id_t = out["Id"].abs().squeeze().numpy()
            log_Id = np.log10(np.maximum(Id_t, 1e-15))
        except Exception:
            log_Id = np.zeros(N) - 15
            fails += 1
        log_Id_traj[:, t] = log_Id
        feat_prev = torch.tensor(log_Id, dtype=torch.float64)

    warmup = 50
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
    print(f"[z103] starting at {time.strftime('%H:%M:%S')} — NARMA-10 multi-seed")
    kappas = [0.00, 0.03]
    seeds = [42, 43, 44, 45, 46]
    grid = np.zeros((len(kappas), len(seeds)))
    detail = {}
    for j, seed in enumerate(seeds):
        for i, kappa in enumerate(kappas):
            ti = time.time()
            r = run_narma10(kappa, seed=seed)
            grid[i, j] = r["nrmse"]
            detail[f"k{kappa:.2f}_s{seed}"] = r
            print(f"  seed={seed}  κ={kappa:.2f}  NRMSE={r['nrmse']:.4f}  "
                   f"fails={r['fails']}  ({time.time()-ti:.1f}s)", flush=True)
    means = grid.mean(axis=1)
    stds = grid.std(axis=1, ddof=1)
    sems = stds / np.sqrt(len(seeds))
    print(f"\n[z103] === NARMA-10 NRMSE (lower is better) ===")
    print(f"  {'κ':>6s}  {'NRMSE mean':>12s}  {'± std':>7s}  {'± SEM':>7s}  {'min':>6s}  {'max':>6s}")
    for i, kappa in enumerate(kappas):
        print(f"  {kappa:>6.2f}  {means[i]:>12.4f}  {stds[i]:>7.4f}  "
               f"{sems[i]:>7.4f}  {grid[i].min():>6.3f}  {grid[i].max():>6.3f}")
    diffs = grid[1] - grid[0]   # negative = improvement
    d_mean, d_std = diffs.mean(), diffs.std(ddof=1)
    d_sem = d_std / np.sqrt(len(seeds))
    t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
    print(f"\n[z103] paired Δ NRMSE (κ=0.03 − κ=0): {d_mean:+.4f} ± {d_sem:.4f} "
           f"(t = {t_stat:+.2f}); negative = improvement")
    json.dump({"grid": grid.tolist(), "kappas": kappas, "seeds": seeds,
                "means": means.tolist(), "stds": stds.tolist(),
                "paired_diff_mean": float(d_mean), "paired_diff_sem": float(d_sem),
                "paired_t": float(t_stat), "detail": detail},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z103] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
