"""z96 v3 — NARMA-10 ridge × T sweep.

A.5.t (2026-05-02): v2 hit NRMSE_test=1.10 (overfit). Run once at
T=1000 then post-hoc evaluate at ridge ∈ {1e-3, 1e-1, 1, 10, 100}
and train_size ∈ {100, 300, 600}.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z96v3_narma10_sweep"
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
    rng = np.random.default_rng(42)
    T = 1000
    u = 0.5 * rng.random(T)
    y_target = v1.narma10(u)
    print(f"[z96v3] T={T}, y range [{y_target.min():.3f}, {y_target.max():.3f}]")

    N = 20
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    VG1_arr = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    VG2_arr = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    Vd_seq = torch.tensor(0.5 + 2.0 * u, dtype=torch.float64)
    dt = 1e-6
    t_arr = torch.arange(T, dtype=torch.float64) * dt
    print(f"[z96v3] N={N}, T={T} (ETA ~{N * 3:.0f}s = {N*3/60:.1f}min)")

    Id_all = np.zeros((N, T))
    Vb_all = np.zeros((N, T))
    for i in range(N):
        ti = time.time()
        VG1_i = torch.tensor(float(VG1_arr[i]))
        VG2_i = torch.tensor(float(VG2_arr[i]))
        try:
            res = transient_2t(cfg, M1, M2, bjt, Vd_t=Vd_seq, t=t_arr,
                                VG1=VG1_i, VG2=VG2_i,
                                Vb0=0.0, Vsint0=0.1,
                                spike_threshold=0.65, reset_Vb=0.30,
                                newton_iters=15, newton_tol=1e-9, damp=0.7,
                                verbose=False)
            Id_all[i] = res["Id"].numpy()
            Vb_all[i] = res["Vb"].numpy()
            print(f"  cell {i:2d}/N=20  ({time.time()-ti:.1f}s)", flush=True)
        except Exception as e:
            print(f"  cell {i:2d}: FAIL ({e})", flush=True)

    log_Id = np.log10(np.maximum(np.abs(Id_all), 1e-15))
    print(f"\n[z96v3] log_Id std: mean={log_Id.std(axis=1).mean():.3f}")
    print(f"[z96v3] forward done at {time.time()-t0:.1f}s")

    # Sweep ridge × train_size
    warmup = 50
    feat = np.vstack([log_Id, Vb_all])  # 2N features
    print(f"\n[z96v3] Sweep results (NRMSE_test):")
    print(f"  {'ridge':>10s}  {'n_train=100':>12s}  {'n_train=300':>12s}  {'n_train=600':>12s}")
    results = {}
    for ridge in [1e-3, 1e-1, 1.0, 10.0, 100.0, 1000.0]:
        row = []
        for n_train in [100, 300, 600]:
            n_test = 200
            if warmup + n_train + n_test > T:
                row.append(float('nan')); continue
            X_tr = np.hstack([np.ones((n_train, 1)), feat[:, warmup:warmup+n_train].T])
            X_te = np.hstack([np.ones((n_test, 1)), feat[:, warmup+n_train:warmup+n_train+n_test].T])
            y_tr = y_target[warmup:warmup+n_train]
            y_te = y_target[warmup+n_train:warmup+n_train+n_test]
            XtX = X_tr.T @ X_tr
            XtY = X_tr.T @ y_tr
            W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
            pred_te = X_te @ W
            nrmse = np.sqrt(((pred_te - y_te) ** 2).mean()) / y_te.std()
            row.append(float(nrmse))
            results[f"ridge={ridge}_ntrain={n_train}"] = float(nrmse)
        print(f"  {ridge:>10g}  {row[0]:>12.4f}  {row[1]:>12.4f}  {row[2]:>12.4f}")

    summary = {"N": N, "T": T, "wall_s": float(time.time()-t0), **results}
    json.dump(summary, (OUT / "summary.json").open("w"), indent=2)
    np.savez(OUT / "results.npz", u=u, y_target=y_target, Id=Id_all,
             Vb=Vb_all, VG1=VG1_arr.numpy(), VG2=VG2_arr.numpy())
    print(f"\n[z96v3] wall: {time.time()-t0:.1f}s")
    print(f"[z96v3] saved {OUT}/summary.json + results.npz")


if __name__ == "__main__":
    main()
