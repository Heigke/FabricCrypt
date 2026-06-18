"""z97 v2 — Memory capacity benchmark on the post-Phase-A-closure cell.

B.5.b (2026-05-03): Re-pilot the linear MC test with the calibrated
stack now that voff shifts are zeroed (Phase A closed in A.5.cc/dd).
N=20 cells, T=2000, dt swept across {1ns, 10ns, 100ns} to find the
bandwidth that gives non-trivial memory capacity.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z97v2_memory_capacity_clean"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.joint_newton import transient_2t


def run_mc(dt: float, N: int = 20, T: int = 2000, K: int = 15, seed: int = 42):
    rng = np.random.default_rng(seed)
    u = 2.0 * rng.integers(0, 2, size=T) - 1.0  # bipolar input
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()  # ZERO shifts now (Phase A closed)
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    VG1_arr = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    VG2_arr = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    Vd_seq = torch.tensor(1.0 + 0.5 * u, dtype=torch.float64)
    t_arr = torch.arange(T, dtype=torch.float64) * dt

    Id_all = np.zeros((N, T)); Vb_all = np.zeros((N, T))
    fails = 0
    for i in range(N):
        try:
            res = transient_2t(cfg, M1, M2, bjt, Vd_t=Vd_seq, t=t_arr,
                                VG1=torch.tensor(float(VG1_arr[i])),
                                VG2=torch.tensor(float(VG2_arr[i])),
                                Vb0=0.0, Vsint0=0.1,
                                spike_threshold=0.65, reset_Vb=0.30,
                                newton_iters=15, newton_tol=1e-9, damp=0.7)
            Id_all[i] = res["Id"].numpy(); Vb_all[i] = res["Vb"].numpy()
        except Exception:
            fails += 1
    log_Id = np.log10(np.maximum(np.abs(Id_all), 1e-15))
    feat = np.vstack([log_Id, Vb_all])

    warmup = 50
    n_train = 1200
    n_test = T - warmup - n_train - K
    mc_per_k = np.zeros(K)
    for k in range(1, K + 1):
        t_idx = np.arange(warmup, warmup + n_train + n_test)
        X = np.hstack([np.ones((len(t_idx), 1)), feat[:, t_idx].T])
        y = u[t_idx - k]
        X_tr, X_te = X[:n_train], X[n_train:n_train+n_test]
        y_tr, y_te = y[:n_train], y[n_train:n_train+n_test]
        ridge = 1e-3
        XtX = X_tr.T @ X_tr; XtY = X_tr.T @ y_tr
        W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
        pred = X_te @ W
        if pred.std() > 1e-12 and y_te.std() > 1e-12:
            r = np.corrcoef(pred, y_te)[0, 1]
            mc_per_k[k - 1] = float(r * r)
    MC = float(mc_per_k.sum())
    return {
        "dt": dt, "N": N, "T": T, "fails": fails,
        "MC": MC, "mc_per_k": mc_per_k.tolist(),
        "log_Id_std_mean": float(log_Id.std(axis=1).mean()),
        "Vb_std_mean": float(Vb_all.std(axis=1).mean()),
    }


def main():
    t0 = time.time()
    print(f"[z97v2] starting at {time.strftime('%H:%M:%S')} (Phase A closed: voff shifts = 0)")
    results = {}
    for dt in [1e-9, 10e-9, 100e-9]:
        ti = time.time()
        print(f"\n[z97v2] dt={dt*1e9:.1f}ns ...", flush=True)
        r = run_mc(dt, N=20, T=2000)
        r["wall_s"] = float(time.time() - ti)
        results[f"dt_{dt}"] = r
        print(f"  log_Id_std={r['log_Id_std_mean']:.3f}  Vb_std={r['Vb_std_mean']:.3f}  "
              f"MC={r['MC']:.3f}  fails={r['fails']}  ({r['wall_s']:.1f}s)")
        # Print first 5 r² for context
        print(f"  r²(k=1..5) = {[f'{x:.3f}' for x in r['mc_per_k'][:5]]}")
    print(f"\n[z97v2] === SUMMARY ===")
    print(f"  {'dt(ns)':>10s}  {'log_Id_std':>11s}  {'Vb_std':>8s}  {'MC':>8s}")
    for k, r in results.items():
        print(f"  {r['dt']*1e9:>10.1f}  {r['log_Id_std_mean']:>11.3f}  "
              f"{r['Vb_std_mean']:>8.3f}  {r['MC']:>8.3f}")
    json.dump(results, (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z97v2] wall: {time.time()-t0:.1f}s")
    print(f"[z97v2] saved {OUT}/summary.json")
    print(f"[z97v2] (z2206 FPGA-128-cell baseline: MC=2.67; chance: 0)")


if __name__ == "__main__":
    main()
