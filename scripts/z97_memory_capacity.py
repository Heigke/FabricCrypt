"""z97 — Memory capacity (MC) benchmark on calibrated 2T NS-RAM cells.

A.5.v (2026-05-02): Pivot from NARMA-10 (timescale-mismatched) to
the standard linear memory capacity test:
   MC = sum_{k=1..K} r²(û_k(t), u(t-k))
where û_k is the linear ridge prediction of u shifted by k steps.
MC sums the squared correlations across delays — measures total
linear memory in the reservoir.

Reference: Jaeger 2002. For a passive linear reservoir, MC ≤ N
(reservoir size). FPGA-NSRAM in z2206 hit MC=2.67 with 128 neurons.

Config: N=10 cells, T=2000, dt=10ns (best dynamic range from A.5.u).
Random binary input u(t) ∈ {-1, +1} mapped to Vd ∈ [0.5, 1.5].
Predict u(t-k) for k=1..15.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z97_memory_capacity"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.joint_newton import transient_2t


def main():
    t0 = time.time()
    print(f"[z97] starting at {time.strftime('%H:%M:%S')}", flush=True)

    rng = np.random.default_rng(42)
    T = 2000
    # Bipolar random input — standard MC stimulus
    u = 2.0 * rng.integers(0, 2, size=T) - 1.0
    print(f"[z97] T={T}, u ∈ {{-1,+1}}")

    N = 10
    dt = 10e-9
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    VG1_arr = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    VG2_arr = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    # Map u ∈ {-1, +1} → Vd ∈ {0.5, 1.5}
    Vd_seq = torch.tensor(1.0 + 0.5 * u, dtype=torch.float64)
    t_arr = torch.arange(T, dtype=torch.float64) * dt
    print(f"[z97] N={N}, dt={dt*1e9:.1f}ns, total_t={T*dt*1e6:.1f}µs")

    Id_all = np.zeros((N, T)); Vb_all = np.zeros((N, T))
    for i in range(N):
        ti = time.time()
        try:
            res = transient_2t(cfg, M1, M2, bjt, Vd_t=Vd_seq, t=t_arr,
                                VG1=torch.tensor(float(VG1_arr[i])),
                                VG2=torch.tensor(float(VG2_arr[i])),
                                Vb0=0.0, Vsint0=0.1,
                                spike_threshold=0.65, reset_Vb=0.30,
                                newton_iters=15, newton_tol=1e-9, damp=0.7)
            Id_all[i] = res["Id"].numpy()
            Vb_all[i] = res["Vb"].numpy()
            print(f"  cell {i:2d}/N={N}  ({time.time()-ti:.1f}s)", flush=True)
        except Exception as e:
            print(f"  cell {i:2d}: FAIL ({e})", flush=True)

    log_Id = np.log10(np.maximum(np.abs(Id_all), 1e-15))
    feat = np.vstack([log_Id, Vb_all])  # (2N, T)
    print(f"\n[z97] feature std mean: log_Id={log_Id.std(axis=1).mean():.3f}  "
          f"Vb={Vb_all.std(axis=1).mean():.3f}")

    K = 15
    warmup = 50
    n_train = 1200
    n_test = T - warmup - n_train - K  # leave room for delay shift
    print(f"[z97] warmup={warmup} n_train={n_train} n_test={n_test} K={K}")

    # For each delay k, train a ridge to predict u(t-k) from feat(t)
    mc_per_k = np.zeros(K)
    for k in range(1, K + 1):
        # Targets: u shifted backwards
        t_idx = np.arange(warmup, warmup + n_train + n_test)
        X = feat[:, t_idx].T  # (n_samples, 2N)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        y = u[t_idx - k]
        X_tr, X_te = X[:n_train], X[n_train:n_train+n_test]
        y_tr, y_te = y[:n_train], y[n_train:n_train+n_test]
        ridge = 1e-3
        XtX = X_tr.T @ X_tr; XtY = X_tr.T @ y_tr
        W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
        pred = X_te @ W
        # squared correlation r²
        if pred.std() < 1e-12 or y_te.std() < 1e-12:
            r2 = 0.0
        else:
            r = np.corrcoef(pred, y_te)[0, 1]
            r2 = float(r * r)
        mc_per_k[k - 1] = r2
        print(f"  k={k:2d}  r²={r2:.4f}", flush=True)

    MC = float(mc_per_k.sum())
    print(f"\n[z97] MEMORY CAPACITY = {MC:.3f}  (sum of r² for k=1..{K})")
    print(f"[z97] (FPGA-NSRAM 128-neuron baseline z2206: MC=2.67)")
    print(f"[z97] (Theoretical max for N={N} cells: ~{N})")

    summary = {
        "N": N, "T": T, "dt": dt, "K": K,
        "warmup": warmup, "n_train": n_train, "n_test": n_test,
        "voff_M1_shift": 0.20, "voff_M2_shift": -0.20,
        "MC_total": MC,
        "MC_per_k": mc_per_k.tolist(),
        "log_Id_std_mean": float(log_Id.std(axis=1).mean()),
        "Vb_std_mean": float(Vb_all.std(axis=1).mean()),
        "wall_s": float(time.time() - t0),
    }
    json.dump(summary, (OUT / "summary.json").open("w"), indent=2)
    np.savez(OUT / "results.npz", u=u, Id=Id_all, Vb=Vb_all,
             VG1=VG1_arr.numpy(), VG2=VG2_arr.numpy(), mc_per_k=mc_per_k)
    print(f"\n[z97] wall: {time.time()-t0:.1f}s")
    print(f"[z97] saved {OUT}/summary.json")


if __name__ == "__main__":
    main()
