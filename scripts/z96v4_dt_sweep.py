"""z96 v4 — dt sweep to test timescale hypothesis.

A.5.u (2026-05-02): A.5.t hypothesised that NARMA-10 NRMSE_test
stuck at 0.89 because dt=1µs >> τ_body~0.7ns (body equilibrates each
step → no memory). Sweep dt ∈ {0.1ns, 1ns, 10ns, 100ns} and check if
NRMSE_test drops monotonically as dt approaches τ_body.

If yes at any dt → confirms physics-matched memory works.
If no → cell physics doesn't support NARMA-10 timescale; pivot to
benchmark substitution.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z96v4_dt_sweep"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.joint_newton import transient_2t


def run_one_dt(dt: float, N=10, T=500, seed=42):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.random(T)
    y_target = v1.narma10(u)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    VG1_arr = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    VG2_arr = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    Vd_seq = torch.tensor(0.5 + 2.0 * u, dtype=torch.float64)
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
        except Exception as e:
            fails += 1
    log_Id = np.log10(np.maximum(np.abs(Id_all), 1e-15))
    feat = np.vstack([log_Id, Vb_all])
    warmup, n_train, n_test = 50, 300, 100
    X_tr = np.hstack([np.ones((n_train, 1)), feat[:, warmup:warmup+n_train].T])
    X_te = np.hstack([np.ones((n_test, 1)), feat[:, warmup+n_train:warmup+n_train+n_test].T])
    y_tr = y_target[warmup:warmup+n_train]
    y_te = y_target[warmup+n_train:warmup+n_train+n_test]
    best_test = float('inf')
    for ridge in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        XtX = X_tr.T @ X_tr; XtY = X_tr.T @ y_tr
        W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
        nrmse = float(np.sqrt(((X_te @ W - y_te) ** 2).mean()) / y_te.std())
        best_test = min(best_test, nrmse)
    return {
        "dt": dt, "N": N, "T": T, "fails": fails,
        "log_Id_std_mean": float(log_Id.std(axis=1).mean()),
        "Vb_std_mean": float(Vb_all.std(axis=1).mean()),
        "log_Id_range": [float(log_Id.min()), float(log_Id.max())],
        "best_NRMSE_test": best_test,
    }


def main():
    t0 = time.time()
    print(f"[z96v4] starting at {time.strftime('%H:%M:%S')}")
    results = {}
    for dt in [0.1e-9, 1e-9, 10e-9, 100e-9, 1e-6]:
        ti = time.time()
        print(f"\n[z96v4] dt = {dt*1e9:.1f} ns ({dt}s) ...", flush=True)
        r = run_one_dt(dt, N=10, T=500)
        r["wall_s"] = float(time.time() - ti)
        results[f"dt={dt}"] = r
        print(f"  log_Id_std={r['log_Id_std_mean']:.3f}  Vb_std={r['Vb_std_mean']:.3f}  "
              f"range={r['log_Id_range']}  NRMSE_test={r['best_NRMSE_test']:.4f}  "
              f"fails={r['fails']}  ({r['wall_s']:.1f}s)")
    print(f"\n[z96v4] === SWEEP TABLE ===")
    print(f"  {'dt(ns)':>10s}  {'log_Id_std':>11s}  {'Vb_std':>8s}  {'NRMSE_test':>11s}  {'fails':>6s}")
    for key, r in results.items():
        print(f"  {r['dt']*1e9:>10.2f}  {r['log_Id_std_mean']:>11.3f}  {r['Vb_std_mean']:>8.3f}  "
              f"{r['best_NRMSE_test']:>11.4f}  {r['fails']:>6d}")
    json.dump(results, (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z96v4] wall: {time.time()-t0:.1f}s")
    print(f"[z96v4] saved {OUT}/summary.json")


if __name__ == "__main__":
    main()
