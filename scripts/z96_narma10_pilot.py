"""z96 — NARMA-10 reservoir pilot on calibrated 2T NS-RAM cells.

B.5.a (2026-05-02): First NARMA-10 reservoir benchmark using the
PyTorch BSIM4 port. Uses A.5.m calibrated voff offsets
(M1=+0.20 V, M2=−0.20 V → joint-optimum z91g median 0.846 dec).

Method:
  - N cells with random (VG1, VG2) sampled in working range.
  - Time-varying input u(t) ∼ U(0, 0.5) drives all cells via Vd[t].
  - Reservoir features = log10|Id_i(t)| per cell per timestep.
  - Linear ridge readout predicts NARMA-10 target one-step-ahead.
  - Metric: NRMSE on held-out test sequence.

Pilot config: N=20 cells, T=1000 (warmup 100, train 600, test 300).
Should run in ~minutes on CPU; if successful, scale to N=10k on GPU.
"""
from __future__ import annotations
import math, json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z96_narma10_pilot"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
f = importlib.util.module_from_spec(sp); sp.loader.exec_module(f)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched

DATA = ROOT / "data/sebas_2026_04_22"


def narma10(u: np.ndarray) -> np.ndarray:
    """Standard NARMA-10 target. u in [0, 0.5]."""
    T = len(u)
    y = np.zeros(T)
    for t in range(10, T):
        y[t] = 0.3 * y[t-1] + 0.05 * y[t-1] * np.sum(y[t-10:t]) \
              + 1.5 * u[t-10] * u[t-1] + 0.1
    return y


def build_calibrated_models(voff_M1_shift=0.0, voff_M2_shift=0.0):
    """A.5.cc/dd: voff shifts now default to 0 — Phase A closed via
    proper physics fixes (lpe0/toxe/phin/phi/multi-assignment),
    no calibration deltas needed."""
    """Load M1/M2 cards with the A.5.m calibration baked in."""
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    f.patch_model_values(M1, type_n=True)
    f.patch_model_values(M2, type_n=True)
    M1._values["voff"] = M1._values.get("voff", -0.1368) + voff_M1_shift
    M2._values["voff"] = M2._values.get("voff", -0.1368) + voff_M2_shift
    return M1, M2


def main():
    t0 = time.time()
    print(f"[z96] starting at {time.strftime('%H:%M:%S')}", flush=True)

    # --- Generate NARMA-10 sequence ---
    rng = np.random.default_rng(42)
    T = 1000
    u = 0.5 * rng.random(T)
    y_target = narma10(u)
    print(f"[z96] NARMA-10 target: T={T}, y range [{y_target.min():.3f}, {y_target.max():.3f}]")

    # --- Build N cells ---
    N = 20
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    M1, M2 = build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 5e4
    print(f"[z96] cells={N}, calibration M1 voff={M1._values['voff']:.4f}, M2 voff={M2._values['voff']:.4f}")

    # Per-cell (VG1, VG2) sampled in Sebas's working range:
    # VG1 ∈ {0.2, 0.4, 0.6}, VG2 ∈ [0, 0.5]. Random uniform.
    VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    print(f"[z96] VG1 distribution: {torch.unique(VG1, return_counts=True)}")
    print(f"[z96] VG2 range: [{float(VG2.min()):.3f}, {float(VG2.max()):.3f}]")

    # --- Drive cells with u(t) as Vd input ---
    # Map u ∈ [0, 0.5] → Vd ∈ [0.5, 1.5] (working measurement range)
    Vd_seq = torch.tensor(0.5 + 2.0 * u, dtype=torch.float64)
    print(f"[z96] Vd input range: [{float(Vd_seq.min()):.3f}, {float(Vd_seq.max()):.3f}]")

    # Run forward pass — N parallel cells × T timesteps
    t_fwd = time.time()
    print(f"[z96] forward_2t_batched (N={N} × T={T})...", flush=True)
    out = forward_2t_batched(cfg, M1, M2, bjt, Vd_seq, VG1, VG2,
                              max_iters=20, tol=1e-9, verbose=False)
    print(f"[z96] forward done in {time.time()-t_fwd:.1f}s")
    print(f"[z96] convergence: {out['converged'].sum()}/{N*T}")

    # Features: log10(|Id| + eps), shape (N, T)
    Id = out["Id"].abs().numpy()
    log_Id = np.log10(np.maximum(Id, 1e-15))
    print(f"[z96] log_Id range: [{log_Id.min():.2f}, {log_Id.max():.2f}]")
    print(f"[z96] feature std per cell: mean={log_Id.std(axis=1).mean():.3f}, "
          f"min={log_Id.std(axis=1).min():.3e}, max={log_Id.std(axis=1).max():.3f}")

    # --- Train / test split ---
    warmup, n_train, n_test = 100, 600, 300
    # X shape: (T_use, N+1) — bias column included
    X_train = log_Id[:, warmup:warmup+n_train].T
    X_test = log_Id[:, warmup+n_train:warmup+n_train+n_test].T
    X_train = np.hstack([np.ones((n_train, 1)), X_train])
    X_test = np.hstack([np.ones((n_test, 1)), X_test])
    y_train = y_target[warmup:warmup+n_train]
    y_test = y_target[warmup+n_train:warmup+n_train+n_test]

    # --- Ridge readout ---
    ridge = 1e-3
    XtX = X_train.T @ X_train
    XtY = X_train.T @ y_train
    W = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), XtY)
    pred_train = X_train @ W
    pred_test = X_test @ W
    nrmse_train = np.sqrt(((pred_train - y_train) ** 2).mean()) / y_train.std()
    nrmse_test = np.sqrt(((pred_test - y_test) ** 2).mean()) / y_test.std()
    print(f"[z96] NRMSE train={nrmse_train:.4f}  test={nrmse_test:.4f}  ridge={ridge}")

    # --- Save artifacts ---
    np.savez(OUT / "pilot_results.npz",
             u=u, y_target=y_target, Id=Id, log_Id=log_Id,
             pred_train=pred_train, pred_test=pred_test,
             y_train=y_train, y_test=y_test,
             VG1=VG1.numpy(), VG2=VG2.numpy(), Vd_seq=Vd_seq.numpy(),
             W=W, nrmse_train=nrmse_train, nrmse_test=nrmse_test)
    summary = {
        "N": N, "T": T, "warmup": warmup, "n_train": n_train, "n_test": n_test,
        "voff_M1_shift": 0.20, "voff_M2_shift": -0.20, "ridge": float(ridge),
        "nrmse_train": float(nrmse_train), "nrmse_test": float(nrmse_test),
        "convergence_rate": float(out["converged"].sum()) / (N * T),
        "wall_s": float(time.time() - t0),
    }
    json.dump(summary, (OUT / "summary.json").open("w"), indent=2)
    print(f"[z96] wall: {time.time()-t0:.1f}s")
    print(f"[z96] saved {OUT}/summary.json + pilot_results.npz")


if __name__ == "__main__":
    main()
