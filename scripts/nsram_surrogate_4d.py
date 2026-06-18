"""4D transient surrogate: (VG1, VG2, Vd, Vb) → (Id, Iii, Ileak).

Per O32 oracle consensus (Path A). Closes the MC=1 gap by exposing
Vb as an input axis and giving back the body currents that will time-
step Vb at runtime.

Build approach:
  1. For each grid point (VG1, VG2, Vd, Vb): solve only Vsint via
     pyport's _residuals at FIXED Vb (1D Newton on R_Sint).
  2. Read Id, Iii (into body), Ileak_out (= junction diodes leaving body)
     from the components dict.
  3. Cache in 4D numpy array; quadrilinear interp at runtime.

Time step at runtime:
  Iii_net = Iii_into_body - Ileak_out_of_body
  Vb[t+1] = clip(Vb[t] + dt * Iii_net / Cb, 0, Vb_max)
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

VG1_AXIS = np.array([0.10, 0.25, 0.40, 0.55, 0.70], dtype=np.float64)
VG2_AXIS = np.array([0.00, 0.15, 0.30, 0.45, 0.60], dtype=np.float64)
VD_AXIS  = np.array([0.50, 1.00, 1.50, 2.00],       dtype=np.float64)
VB_AXIS  = np.array([0.00, 0.20, 0.40, 0.55, 0.70], dtype=np.float64)
# = 5 × 5 × 4 × 5 = 500 op points

# Production BJT params (from brief calibration)
OPT_BF, OPT_VA, OPT_IS = 9000.0, 0.55, 1e-9


def _build_pyport_models():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=20)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = OPT_BF; bjt.Va = OPT_VA; bjt.Is = OPT_IS
    return cfg, M1, M2, bjt


def _solve_at_fixed_vb(cfg, M1, M2, bjt, Vd, VG1, VG2, Vb_fixed):
    """1D Newton on Vsint only, with Vb pinned at Vb_fixed.

    Returns dict: Id, Iii_into_body, Ileak_out_of_body, Vsint, converged.
    """
    from nsram.bsim4_port.nsram_cell_2T import _residuals

    Vd_t = torch.tensor(Vd, dtype=torch.float64)
    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)
    Vb_t = torch.tensor(Vb_fixed, dtype=torch.float64)
    # Initial guess
    Vsint = (0.5 * Vd_t).clone()
    converged = False
    for it in range(cfg.newton_max_iters):
        R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                     Vsint, Vb_t, model_M2=M2)
        # Only use R_S; R_B is non-zero because Vb is fixed (not at equilibrium)
        if abs(float(R_S)) < 1e-12:
            converged = True
            break
        # Finite-difference dR_S/dVsint
        h = 1e-6
        R_Sp, _, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                  Vsint + h, Vb_t, model_M2=M2)
        dRdV = (R_Sp - R_S) / h
        if abs(float(dRdV)) < 1e-30:
            break
        dV = -R_S / dRdV
        # Cap step
        dV = torch.clamp(dV, -0.5, 0.5)
        Vsint = Vsint + dV
    # Recompute components at converged Vsint
    R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                 Vsint, Vb_t, model_M2=M2)
    # Id formula matches solve_2t_steady_state
    Id = (comp["Ids_M1"] + comp["Ic_Q1"]
          + comp.get("Ic_lat", 0.0) + comp.get("Ic_avalanche", 0.0)
          + comp["Igidl_M1"] - comp["Ibd_M1"])
    # Iii into body (impact-ion holes from M1 + M2 channels)
    Iii_in = comp.get("Iii_M1", 0.0)
    if not cfg.m2_body_gnd:
        Iii_in = Iii_in + comp.get("Iii_M2", 0.0)
    # Plus well-body diode (drives Vb up)
    Iii_in = Iii_in + comp.get("I_well_body", 0.0)
    # Plus GIDL (also into body)
    Iii_in = Iii_in + comp.get("Igidl_M1", 0.0) + comp.get("Igisl_M1", 0.0)
    # Ileak out of body: junction diodes Ibs, Ibd of M1 (positive when
    # leaving body) plus BJT base current (positive when leaving body)
    Ileak_out = (comp.get("Ibs_M1", 0.0) + comp.get("Ibd_M1", 0.0)
                  + comp.get("Ib_Q1", 0.0))
    return {
        "Id": float(Id), "Iii_in": float(Iii_in), "Ileak_out": float(Ileak_out),
        "Vsint": float(Vsint), "converged": bool(converged),
    }


def build_4d_grid(out_path: Path | None = None, verbose: bool = True):
    """Build 4D grid; persist to npz. Wall budget ~5-15 min depending on
    grid resolution and Newton convergence."""
    cfg, M1, M2, bjt = _build_pyport_models()
    NG1, NG2, NVD, NVB = (len(VG1_AXIS), len(VG2_AXIS), len(VD_AXIS), len(VB_AXIS))
    n_total = NG1 * NG2 * NVD * NVB
    Id_grid = np.full((NG1, NG2, NVD, NVB), np.nan, dtype=np.float64)
    Iii_grid = np.zeros_like(Id_grid)
    Ileak_grid = np.zeros_like(Id_grid)
    conv_grid = np.zeros((NG1, NG2, NVD, NVB), dtype=bool)

    t0 = time.time()
    n_done = 0
    for i, vg1 in enumerate(VG1_AXIS):
        for j, vg2 in enumerate(VG2_AXIS):
            for k, vd in enumerate(VD_AXIS):
                for l, vb in enumerate(VB_AXIS):
                    try:
                        out = _solve_at_fixed_vb(cfg, M1, M2, bjt, vd, vg1, vg2, vb)
                        Id_grid[i, j, k, l] = out["Id"]
                        Iii_grid[i, j, k, l] = out["Iii_in"]
                        Ileak_grid[i, j, k, l] = out["Ileak_out"]
                        conv_grid[i, j, k, l] = out["converged"]
                    except Exception as e:
                        if verbose:
                            print(f"  fail at vg1={vg1} vg2={vg2} vd={vd} vb={vb}: {e}")
                    n_done += 1
                    if verbose and n_done % 50 == 0:
                        wall = time.time() - t0
                        eta = wall / n_done * (n_total - n_done)
                        print(f"  {n_done}/{n_total} ({100*n_done/n_total:.0f}%); "
                              f"wall={wall:.0f}s eta={eta:.0f}s")

    n_conv = int(conv_grid.sum())
    if verbose:
        print(f"\n[4d] built {n_total} pts in {time.time()-t0:.0f}s; "
              f"converged {n_conv}/{n_total}")
    if out_path is not None:
        np.savez(out_path, Id=Id_grid, Iii=Iii_grid, Ileak=Ileak_grid,
                 converged=conv_grid,
                 vg1_axis=VG1_AXIS, vg2_axis=VG2_AXIS,
                 vd_axis=VD_AXIS, vb_axis=VB_AXIS)
        meta = {"OPT_BF": OPT_BF, "OPT_VA": OPT_VA, "OPT_IS": OPT_IS,
                "grid_size": [NG1, NG2, NVD, NVB], "n_converged": n_conv,
                "wall_s": time.time() - t0}
        with open(out_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
    return Id_grid, Iii_grid, Ileak_grid, conv_grid


class NSRAMSurrogate4D:
    """4D quadrilinear interpolating surrogate. Loaded from npz."""
    def __init__(self, path):
        d = np.load(path)
        self.Id_log = np.log10(np.maximum(np.abs(d["Id"]), 1e-15))
        self.Iii = d["Iii"]
        self.Ileak = d["Ileak"]
        self.vg1_axis = d["vg1_axis"]
        self.vg2_axis = d["vg2_axis"]
        self.vd_axis = d["vd_axis"]
        self.vb_axis = d["vb_axis"]

    @staticmethod
    def _idx(x, axis):
        x = np.asarray(x).clip(axis[0], axis[-1])
        i = np.searchsorted(axis, x).clip(1, len(axis) - 1) - 1
        f = (x - axis[i]) / np.maximum(axis[i+1] - axis[i], 1e-30)
        return i, f

    def eval(self, VG1, VG2, Vd, Vb):
        i, fi = self._idx(VG1, self.vg1_axis)
        j, fj = self._idx(VG2, self.vg2_axis)
        k, fk = self._idx(Vd,  self.vd_axis)
        l, fl = self._idx(Vb,  self.vb_axis)

        def trilin(grid):
            # 16-corner sum
            r = 0.0
            for di in [0, 1]:
                for dj in [0, 1]:
                    for dk in [0, 1]:
                        for dl in [0, 1]:
                            wf = (fi if di else (1-fi)) * (fj if dj else (1-fj)) \
                                  * (fk if dk else (1-fk)) * (fl if dl else (1-fl))
                            r = r + wf * grid[i+di, j+dj, k+dk, l+dl]
            return r

        return trilin(self.Id_log), trilin(self.Iii), trilin(self.Ileak)


if __name__ == "__main__":
    out_dir = ROOT / "results/z219_4d_surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "surrogate_4d.npz"
    if not out_path.exists():
        print(f"[main] building 4D surrogate to {out_path}")
        build_4d_grid(out_path)
    else:
        print(f"[main] surrogate exists at {out_path}")
    surr = NSRAMSurrogate4D(out_path)
    # Smoke
    log_Id, Iii, Ileak = surr.eval(0.4, 0.2, 1.0, 0.3)
    print(f"smoke at (0.4, 0.2, 1.0, 0.3): log_Id={log_Id:.3f} Iii={Iii:.3e} Ileak={Ileak:.3e}")
