"""z101 — B.5.c second cut: finer κ sweep + N-scaling for MC.

Builds on z100. Goal: produce a publication-grade MC(κ, N) chart
for the Mario brief, showing how external recurrence on memoryless
NS-RAM cells lifts memory capacity past the chance threshold.

Sweep:
  κ ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40}
  N ∈ {10, 30, 50}

Single seed (42) so the per-cell biases and W_rec are reproducible.
T=400, 15 lags. Wall-time budget ~10-15 min on Ikaros local CPU.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z101_recurrence_kappa_N_sweep"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def run_recurrent_mc(kappa: float, N: int, T: int = 300, K: int = 15, seed: int = 42):
    rng = np.random.default_rng(seed)
    u = 2.0 * rng.integers(0, 2, size=T) - 1.0
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
        Vd_t = torch.tensor([1.0 + 0.5 * float(u[t])], dtype=torch.float64)
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
    n_train = int(0.6 * (T - warmup - K))
    n_test = T - warmup - n_train - K
    feat_norm = (log_Id_traj - log_Id_traj.mean(axis=1, keepdims=True))
    feat_norm = feat_norm / (feat_norm.std(axis=1, keepdims=True) + 1e-9)
    mc_per_k = np.zeros(K)
    for k in range(1, K + 1):
        t_idx = np.arange(warmup, warmup + n_train + n_test)
        X = np.hstack([np.ones((len(t_idx), 1)), feat_norm[:, t_idx].T])
        y = u[t_idx - k]
        X_tr, X_te = X[:n_train], X[n_train:n_train + n_test]
        y_tr, y_te = y[:n_train], y[n_train:n_train + n_test]
        ridge = 1e-3
        XtX = X_tr.T @ X_tr; XtY = X_tr.T @ y_tr
        W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
        pred = X_te @ W
        if pred.std() > 1e-12 and y_te.std() > 1e-12:
            r = np.corrcoef(pred, y_te)[0, 1]
            mc_per_k[k - 1] = float(r * r)
    return {
        "kappa": kappa, "N": N, "T": T, "fails": fails,
        "MC": float(mc_per_k.sum()),
        "mc_per_k": mc_per_k.tolist(),
        "log_Id_std_mean": float(log_Id_traj.std(axis=1).mean()),
    }


def main():
    t0 = time.time()
    print(f"[z101] starting at {time.strftime('%H:%M:%S')} — κ × N sweep")
    kappas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    Ns = [10, 30]   # 50 deferred to next iteration (wall-time budget)
    results = {}
    grid = np.zeros((len(Ns), len(kappas)))
    for i, N in enumerate(Ns):
        for j, kappa in enumerate(kappas):
            ti = time.time()
            r = run_recurrent_mc(kappa, N=N, T=300)
            r["wall_s"] = float(time.time() - ti)
            grid[i, j] = r["MC"]
            key = f"N{N}_k{kappa:.2f}"
            results[key] = r
            print(f"  N={N:3d}  κ={kappa:.2f}  MC={r['MC']:6.3f}  "
                   f"feat_std={r['log_Id_std_mean']:5.3f}  ({r['wall_s']:.1f}s)")
    print(f"\n[z101] === MC GRID (rows: N, cols: κ) ===")
    header = "  N\\κ  " + "  ".join(f"{k:5.2f}" for k in kappas)
    print(header)
    for i, N in enumerate(Ns):
        row = f"  {N:4d}  " + "  ".join(f"{grid[i,j]:5.2f}" for j in range(len(kappas)))
        print(row)
    json.dump({"grid": grid.tolist(), "kappas": kappas, "Ns": Ns,
                "results": results}, (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z101] wall: {time.time()-t0:.1f}s")
    print(f"[z101] saved {OUT}/summary.json")
    print(f"[z101] reference: z100 N=10 κ=0.20 → MC=1.37")


if __name__ == "__main__":
    main()
