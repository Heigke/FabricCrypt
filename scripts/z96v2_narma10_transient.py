"""z96 v2 — NARMA-10 reservoir with REAL transient body-cap dynamics.

B.5.a-v2 (2026-05-02): The v1 pilot used quasi-static `forward_2t_batched`
which has no memory — chance NRMSE. This version uses `transient_2t`
per cell with Cj·dVb/dt body capacitance, providing real reservoir
dynamics.

Architecture: N independent cells, each driven by the same Vd(t)
sequence (NARMA-10 input mapping). Per-cell (VG1, VG2) sets the
operating regime. Cells are independent (no inter-cell coupling at
this layer). Body-cap memory provides the temporal state for ridge
readout.

Pilot config: N=20, T=200 (warmup 50, train 100, test 50).
Wall expectation: ~15 min CPU.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z96v2_narma10_transient"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
f = importlib.util.module_from_spec(sp); sp.loader.exec_module(f)
sp2 = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(v1)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.joint_newton import transient_2t


def main():
    t0 = time.time()
    print(f"[z96v2] starting at {time.strftime('%H:%M:%S')}", flush=True)

    rng = np.random.default_rng(42)
    T = 200
    u = 0.5 * rng.random(T)
    y_target = v1.narma10(u)
    print(f"[z96v2] NARMA-10: T={T}, y range [{y_target.min():.3f}, {y_target.max():.3f}]")

    N = 20
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 5e4

    VG1_arr = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    VG2_arr = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    Vd_seq = torch.tensor(0.5 + 2.0 * u, dtype=torch.float64)
    # Time grid: 1 µs per step → 200 µs sequence; sub-µs is too slow given Cj
    dt = 1e-6
    t_arr = torch.arange(T, dtype=torch.float64) * dt
    print(f"[z96v2] N={N} cells, dt={dt*1e6:.1f}µs, total T={T*dt*1e6:.1f}µs")
    print(f"[z96v2] Vd range: [{float(Vd_seq.min()):.3f}, {float(Vd_seq.max()):.3f}]")

    Id_all = np.zeros((N, T))
    Vb_all = np.zeros((N, T))
    spike_counts = np.zeros(N, dtype=int)

    for i in range(N):
        ti = time.time()
        VG1_i = torch.tensor(float(VG1_arr[i]))
        VG2_i = torch.tensor(float(VG2_arr[i]))
        try:
            res = transient_2t(
                cfg, M1, M2, bjt,
                Vd_t=Vd_seq, t=t_arr,
                VG1=VG1_i, VG2=VG2_i,
                Vb0=0.0, Vsint0=0.1,
                spike_threshold=0.65, reset_Vb=0.30,
                newton_iters=15, newton_tol=1e-9, damp=0.7, verbose=False,
            )
            Id_all[i, :] = res["Id"].numpy()
            Vb_all[i, :] = res["Vb"].numpy()
            spike_counts[i] = len(res["spike_times"])
            print(f"  cell {i:2d}/N=20  VG1={float(VG1_i):.2f} VG2={float(VG2_i):.3f}  "
                  f"|Id|range=[{abs(Id_all[i]).min():.2e},{abs(Id_all[i]).max():.2e}]  "
                  f"Vb_std={Vb_all[i].std():.3f}  spikes={spike_counts[i]}  ({time.time()-ti:.1f}s)",
                  flush=True)
        except Exception as e:
            print(f"  cell {i:2d}: FAILED ({type(e).__name__}: {e})", flush=True)

    log_Id = np.log10(np.maximum(np.abs(Id_all), 1e-15))
    print(f"\n[z96v2] log_Id range: [{log_Id.min():.2f}, {log_Id.max():.2f}]")
    print(f"[z96v2] log_Id std per cell over time: mean={log_Id.std(axis=1).mean():.3f}, "
          f"min={log_Id.std(axis=1).min():.3e}, max={log_Id.std(axis=1).max():.3f}")
    print(f"[z96v2] Vb std per cell over time: mean={Vb_all.std(axis=1).mean():.3f}")
    print(f"[z96v2] total spikes: {spike_counts.sum()}")

    # Train/test split
    warmup, n_train, n_test = 50, 100, 50
    feat = np.vstack([log_Id, Vb_all])  # 2N features (log_Id + Vb_traj)
    X_train = np.hstack([np.ones((n_train, 1)), feat[:, warmup:warmup+n_train].T])
    X_test = np.hstack([np.ones((n_test, 1)), feat[:, warmup+n_train:warmup+n_train+n_test].T])
    y_train = y_target[warmup:warmup+n_train]
    y_test = y_target[warmup+n_train:warmup+n_train+n_test]

    ridge = 1e-3
    XtX = X_train.T @ X_train
    XtY = X_train.T @ y_train
    W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
    pred_train = X_train @ W
    pred_test = X_test @ W
    nrmse_train = np.sqrt(((pred_train - y_train) ** 2).mean()) / y_train.std()
    nrmse_test = np.sqrt(((pred_test - y_test) ** 2).mean()) / y_test.std()
    print(f"\n[z96v2] NRMSE train={nrmse_train:.4f}  test={nrmse_test:.4f}  "
          f"features={feat.shape[0]}  ridge={ridge}")

    summary = {
        "N": N, "T": T, "dt": dt, "warmup": warmup,
        "n_train": n_train, "n_test": n_test,
        "voff_M1_shift": 0.20, "voff_M2_shift": -0.20,
        "ridge": float(ridge),
        "nrmse_train": float(nrmse_train), "nrmse_test": float(nrmse_test),
        "n_features": int(feat.shape[0]),
        "total_spikes": int(spike_counts.sum()),
        "wall_s": float(time.time() - t0),
    }
    json.dump(summary, (OUT / "summary.json").open("w"), indent=2)
    np.savez(OUT / "results.npz",
             u=u, y_target=y_target, Id=Id_all, Vb=Vb_all,
             VG1=VG1_arr.numpy(), VG2=VG2_arr.numpy(),
             pred_train=pred_train, pred_test=pred_test,
             y_train=y_train, y_test=y_test, W=W)
    print(f"[z96v2] wall: {time.time()-t0:.1f}s")
    print(f"[z96v2] saved {OUT}/summary.json + results.npz")


if __name__ == "__main__":
    main()
