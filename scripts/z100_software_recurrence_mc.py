"""z100 — B.5.c first cut: software recurrence layer + memory capacity.

Hypothesis (from prior MC=0.16 finding):
  Isolated cells are memoryless analog weights. A SOFTWARE recurrence
  layer on top of static cells should be sufficient to lift MC > 1.0,
  because the recurrence injects temporal state externally.

Design:
  - N=10 cells, T=500 timesteps.
  - Random binary input u(t) ∈ {-1, +1}.
  - At each timestep:
      VG2_eff[i, t] = base_VG2[i] + κ · (W_rec @ feature[t-1])[i]
      Vd[t] = 1.0 + 0.5 · u(t)
  - feature[t] = log10(|Id[t]|) per cell.
  - Recurrence weight W_rec ∈ R^{N×N} drawn from N(0, 1/sqrt(N)) (echo-state init).
  - κ = coupling strength sweep ∈ {0.0, 0.05, 0.20}.
  - Linear ridge readout reconstructs u(t-k) for k=1..15 → MC = sum r².

This validates that NS-RAM cell + external recurrence can serve as a
reservoir even though the cell itself is memoryless. If MC > 1 here,
the cell-as-weight + external-CMOS-routing framing in the Mario brief
is empirically backed.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z100_software_recurrence_mc"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def run_recurrent_mc(kappa: float, N: int = 10, T: int = 500, K: int = 15, seed: int = 42):
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
        # Software recurrence: VG2 modulated by previous-step features
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = base_VG2 + kappa * recur
        VG2_eff = VG2_eff.clamp(-0.2, 1.0)  # keep within physical range

        try:
            out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                       max_iters=15, tol=1e-9, verbose=False)
            Id_t = out["Id"].abs().squeeze().numpy()  # shape (N,)
            log_Id = np.log10(np.maximum(Id_t, 1e-15))
        except Exception:
            log_Id = np.zeros(N) - 15
            fails += 1
        log_Id_traj[:, t] = log_Id
        feat_prev = torch.tensor(log_Id, dtype=torch.float64)

    # MC computation: for each k, ridge-regress feature[t] onto u[t-k]
    warmup = 50
    n_train = 300
    n_test = T - warmup - n_train - K
    mc_per_k = np.zeros(K)
    feat_norm = (log_Id_traj - log_Id_traj.mean(axis=1, keepdims=True))
    feat_norm = feat_norm / (feat_norm.std(axis=1, keepdims=True) + 1e-9)
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
    print(f"[z100] starting at {time.strftime('%H:%M:%S')} — software recurrence + MC")
    print(f"[z100] hypothesis: external recurrence on memoryless cells lifts MC > 1.0")
    results = {}
    for kappa in [0.0, 0.05, 0.20]:
        ti = time.time()
        print(f"\n[z100] κ={kappa:.2f} …", flush=True)
        r = run_recurrent_mc(kappa, N=10, T=500)
        r["wall_s"] = float(time.time() - ti)
        results[f"kappa_{kappa:.2f}"] = r
        print(f"  MC={r['MC']:.3f}  log_Id_std={r['log_Id_std_mean']:.3f}  "
              f"fails={r['fails']}  wall={r['wall_s']:.1f}s")
        print(f"  r²(k=1..5) = {[f'{x:.3f}' for x in r['mc_per_k'][:5]]}")
    print(f"\n[z100] === SUMMARY ===")
    print(f"  {'kappa':>7s}  {'MC':>7s}  {'r²(k=1)':>10s}  {'r²(k=3)':>10s}  {'r²(k=5)':>10s}")
    for key, r in results.items():
        m = r["mc_per_k"]
        print(f"  {r['kappa']:>7.2f}  {r['MC']:>7.3f}  "
              f"{m[0]:>10.3f}  {m[2]:>10.3f}  {m[4]:>10.3f}")
    json.dump(results, (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z100] wall: {time.time()-t0:.1f}s")
    print(f"[z100] saved {OUT}/summary.json")
    print(f"[z100] reference: B.5.b lite gave MC=0.16 with κ=0 (isolated cells)")


if __name__ == "__main__":
    main()
