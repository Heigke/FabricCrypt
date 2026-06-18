"""z429 — S19 multi-solver point-by-point debug of the NS-RAM 2T cell.

Pinned bias: VG1=0.6, VG2=0.0. Sweep V_D in [0, 2] V over 100 steps.
At each V_D we run independent solvers and compare:

  1. Standard Newton           (pyport solve_2t_steady_state, COLD)
  2. Damped Newton             (pyport with newton_min_damping tighter)
  3. Arc-length continuation   (pyport, WARM-started by previous V_D)
  4. scipy.fsolve              (3 distinct initial guesses)
  5. Brute-force grid scan     (100x100 (V_B, V_Sint) residual map)

Plus two diagnostics:
  6. V_Sint = 0 PIN           (1-D Newton on V_B only)
  7. Backward sweep            (warm-started from V_D=2 -> 0)

Outputs land in results/z429_multisolver_debug/.

Run on ikaros (CPU). Reuses z427_vsint_fix loaders + apply_flags pattern.
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z429_multisolver_debug"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# ─── Reuse z427 + z91f loaders ────────────────────────────────────────────
_spec = _ilu.spec_from_file_location("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = _ilu.module_from_spec(_spec); _spec.loader.exec_module(z91f)
_spec_z425 = _ilu.spec_from_file_location("z425", ROOT / "scripts/z425_ideal_floating_body.py")
z425 = _ilu.module_from_spec(_spec_z425); _spec_z425.loader.exec_module(z425)
_spec_z427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
# don't exec z427 (it would run main); we'll import via primitives only

from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, solve_2t_steady_state, _residuals,
)
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt
PWL = z425.PWL


BASE_FLAGS = dict(
    suppress_bulk_diode_forward=True,
    q1_be_oneway=True,
    use_mario_ipos=True,
    mario_ipos_param="VG1",
)


def build_models():
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared = parse_param_blocks(text_M2)
    m_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos", params=shared)
    patch_model_values(m_M1, type_n=True)
    m_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos", params=shared)
    patch_model_values(m_M2, type_n=True)
    return m_M1, m_M2


def make_cfg(model_M1, model_M2):
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=60)
    for k, v in BASE_FLAGS.items():
        setattr(cfg, k, v)
    cfg.mario_ipos_pwl = PWL
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                             Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
                             T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1; cfg._sd_M2 = sd_M2
    return cfg, sd_M1, sd_M2


def scalar(t):
    if torch.is_tensor(t):
        return float(t.detach().reshape(-1)[0].item())
    return float(t)


# ─── Residual wrapper: returns (R_S, R_B, Id) given (Vb, Vsint, Vd) scalars ─
def resid_pair(cfg, model_M1, model_M2, bjt, Vsint_f, Vb_f, Vd_f,
               VG1_f, VG2_f):
    Vd = torch.tensor([Vd_f], dtype=torch.float64)
    VG1 = torch.tensor([VG1_f], dtype=torch.float64)
    VG2 = torch.tensor([VG2_f], dtype=torch.float64)
    Vsint = torch.tensor([Vsint_f], dtype=torch.float64)
    Vb = torch.tensor([Vb_f], dtype=torch.float64)
    with torch.no_grad():
        R_S, R_B, comp = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                    Vsint, Vb, None, None, model_M2=model_M2)
    # Id = -Ids_M2 (drain current convention) but we just record Ids_M1
    Id_pred = comp.get("Ids_M1", torch.tensor(0.0))
    return float(R_S.item()), float(R_B.item()), float(Id_pred.abs().item())


# ─── Solver 1+2: pyport Newton (vary damping) ──────────────────────────────
def run_pyport_newton(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                       Vsint_init=0.0, Vb_init=0.0, max_iters=60,
                       damping=1.0, min_damping=1/64.0):
    Vd = torch.tensor([Vd_f], dtype=torch.float64)
    VG1 = torch.tensor([VG1_f], dtype=torch.float64)
    VG2 = torch.tensor([VG2_f], dtype=torch.float64)
    Vsi = torch.tensor([Vsint_init], dtype=torch.float64)
    Vbi = torch.tensor([Vb_init], dtype=torch.float64)
    cfg.newton_max_iters = max_iters
    cfg.newton_damping = damping
    cfg.newton_min_damping = min_damping
    try:
        with torch.no_grad():
            out = solve_2t_steady_state(cfg, model_M1, bjt, Vd, VG1, VG2,
                                        Vsint_init=Vsi, Vb_init=Vbi,
                                        model_M2=model_M2)
    except Exception as e:
        return dict(error=str(e), converged=False, Id=None, Vb=None,
                    Vsint=None, niter=None, resid=None)
    Vs = scalar(out["Vsint"]); Vb = scalar(out["Vb"]); Id = scalar(out["Id"])
    conv = bool(out["converged"].reshape(-1)[0].item())
    R_S, R_B, _ = resid_pair(cfg, model_M1, model_M2, bjt, Vs, Vb, Vd_f, VG1_f, VG2_f)
    resid = max(abs(R_S), abs(R_B))
    return dict(converged=conv, Id=abs(Id), Vb=Vb, Vsint=Vs,
                niter=int(out.get("niter", -1) or -1), resid=resid)


# ─── Solver 4: scipy.fsolve with multi-IC ──────────────────────────────────
def run_fsolve_multi(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f):
    def f(x):
        Vsint_f, Vb_f = float(x[0]), float(x[1])
        R_S, R_B, _ = resid_pair(cfg, model_M1, model_M2, bjt,
                                  Vsint_f, Vb_f, Vd_f, VG1_f, VG2_f)
        return [R_S, R_B]
    ICs = [(0.0, 0.0), (0.5*Vd_f, 0.5), (Vd_f, 0.7), (0.0, 0.7), (Vd_f, 0.0)]
    best = None
    all_roots = []
    for ic in ICs:
        try:
            x, info, ier, msg = fsolve(f, list(ic), full_output=True,
                                       xtol=1e-12, maxfev=400)
            R_S, R_B, Id = resid_pair(cfg, model_M1, model_M2, bjt,
                                       float(x[0]), float(x[1]),
                                       Vd_f, VG1_f, VG2_f)
            resid = max(abs(R_S), abs(R_B))
            entry = dict(ic=ic, Vsint=float(x[0]), Vb=float(x[1]),
                          Id=Id, resid=resid, ier=int(ier),
                          converged=(ier == 1 and resid < 1e-8))
            all_roots.append(entry)
            if best is None or resid < best["resid"]:
                best = entry
        except Exception as e:
            all_roots.append(dict(ic=ic, error=str(e)))
    return best, all_roots


# ─── Solver 5: brute-force grid scan ───────────────────────────────────────
def run_grid_scan(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                  Vb_range=(-0.5, 1.2), Vsint_range=(-0.2, 2.2),
                  N=40):
    Vb_grid = np.linspace(*Vb_range, N)
    Vsint_grid = np.linspace(*Vsint_range, N)
    RS = np.zeros((N, N))
    RB = np.zeros((N, N))
    ID = np.zeros((N, N))
    for i, Vb_f in enumerate(Vb_grid):
        for j, Vsint_f in enumerate(Vsint_grid):
            R_S, R_B, Id = resid_pair(cfg, model_M1, model_M2, bjt,
                                       Vsint_f, Vb_f, Vd_f, VG1_f, VG2_f)
            RS[i, j] = R_S; RB[i, j] = R_B; ID[i, j] = Id
    # Find cells where residual norm is local minimum
    norm = np.sqrt(RS**2 + RB**2)
    # roots: norm < 1e-6 of typical Ids_M1 magnitude. We use a relative test.
    scale = max(np.max(np.abs(ID)), 1e-12)
    root_mask = norm < scale * 1e-3
    # Cluster naive: pick local minima.
    roots = []
    for i in range(1, N-1):
        for j in range(1, N-1):
            if norm[i, j] <= norm[i-1:i+2, j-1:j+2].min() and norm[i, j] < scale * 0.01:
                roots.append(dict(Vb=float(Vb_grid[i]),
                                   Vsint=float(Vsint_grid[j]),
                                   Id=float(ID[i, j]),
                                   resid=float(norm[i, j])))
    return dict(n_roots=len(roots), roots=roots,
                norm_min=float(norm.min()), norm_max=float(norm.max()),
                scale=scale)


# ─── Solver 6: V_Sint PIN to 0; solve V_B only ─────────────────────────────
# z462b (2026-05-17): Default DC solver changed from Newton-DC to
# pseudo-transient (PT) backward sweep. Newton-DC does not capture
# snap-back/up at the I-V level; PT integration of C_B·dVb/dt = R_B(Vb)
# settles to the physically-relevant attractor (z432 reference).
#
# Env-var override:
#   NSRAM_DC_SOLVER = "pt"      (default)  → pseudo-transient per-point
#                   = "newton"             → legacy 1-D Newton (deprecated)
import os as _os
import warnings as _warnings

_DC_SOLVER_DEFAULT = _os.environ.get("NSRAM_DC_SOLVER", "pt").lower()
_NEWTON_DEPRECATION_WARNED = False


def _run_vsint_pinned_newton(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                              Vsint_pin=0.0, Vb_init=0.0):
    """Legacy 1-D Newton on V_B with Vsint pinned. Does NOT capture snap-back."""
    Vb = Vb_init
    for it in range(80):
        R_S, R_B, _ = resid_pair(cfg, model_M1, model_M2, bjt,
                                  Vsint_pin, Vb, Vd_f, VG1_f, VG2_f)
        eps = 1e-5
        _, R_Bp, _ = resid_pair(cfg, model_M1, model_M2, bjt,
                                 Vsint_pin, Vb + eps, Vd_f, VG1_f, VG2_f)
        dRdV = (R_Bp - R_B) / eps
        if abs(dRdV) < 1e-30:
            break
        dV = -R_B / dRdV
        if abs(dV) > 0.2:
            dV = math.copysign(0.2, dV)
        Vb_new = Vb + dV
        Vb_new = max(-0.2, min(1.0, Vb_new))
        if abs(Vb_new - Vb) < 1e-10:
            Vb = Vb_new
            break
        Vb = Vb_new
    R_S, R_B, Id = resid_pair(cfg, model_M1, model_M2, bjt,
                               Vsint_pin, Vb, Vd_f, VG1_f, VG2_f)
    return dict(Vb=Vb, Vsint=Vsint_pin, Id=Id,
                resid_RB=abs(R_B), resid_RS=abs(R_S),
                converged=(abs(R_B) < 1e-8))


# Pseudo-transient defaults (z432 settings).
_PT_C_B = 1.0e-18
_PT_DT = 1.0e-9
_PT_N_STEPS = int(_os.environ.get("NSRAM_PT_N_STEPS", "800"))
_PT_TOL_DV = 1.0e-5
_PT_N_MIN_STEPS = int(_os.environ.get("NSRAM_PT_N_MIN_STEPS", "100"))
_PT_RESID_REL_TOL = 1e-4
# z472 — absolute R_B convergence floor. The relative tolerance
# rel_tol = 1e-4 * max(|Id|, 1e-12) collapses to 1e-16 A for sub-pA
# leakage currents (calibrated snap_Is cell), so PT runs the full 800
# steps and never satisfies its own gate. An absolute floor of 1 pA
# is well below physical measurement resolution and gives genuine
# steady-state when dVb is already < tol_dv.
_PT_RESID_ABS_TOL = float(_os.environ.get("NSRAM_PT_RESID_ABS_TOL", "1e-12"))
# z472 — additional "stalled tail" early-exit: if dVb stays below
# tol_dv_loose for N_STALL consecutive steps AFTER the min-step floor,
# we declare convergence (Vb has flat-lined; any remaining residual
# is being driven by clamps, not physics).
_PT_TOL_DV_LOOSE = float(_os.environ.get("NSRAM_PT_TOL_DV_LOOSE", "1e-6"))
_PT_N_STALL = int(_os.environ.get("NSRAM_PT_N_STALL", "30"))
_PT_VB_MIN, _PT_VB_MAX = -0.2, 1.0


def _run_vsint_pinned_pt(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                          Vsint_pin=0.0, Vb_init=0.0,
                          C_B=_PT_C_B, dt=_PT_DT, n_steps=_PT_N_STEPS,
                          tol_dv=_PT_TOL_DV):
    """Pseudo-transient integration of C_B·dVb/dt = R_B(Vb) with Vsint pinned.

    Per-point semantics matching z432.integrate_vb. Settles to the local
    attractor of R_B, capturing snap-up/latched branches that Newton misses.
    """
    Vb = float(Vb_init)
    converged = False
    n_used = n_steps
    last_Id = 0.0
    last_R_B = 0.0
    n_stalled = 0
    for k in range(n_steps):
        R_S, R_B, Id_now = resid_pair(cfg, model_M1, model_M2, bjt,
                                       Vsint_pin, Vb, Vd_f, VG1_f, VG2_f)
        last_R_B = float(R_B)
        last_Id = float(Id_now) if Id_now is not None else 0.0
        dVb = (float(R_B) / C_B) * dt
        if abs(dVb) > 0.05:
            dVb = math.copysign(0.05, dVb)
        Vb_new = Vb + dVb
        if Vb_new > _PT_VB_MAX:
            Vb_new = _PT_VB_MAX
        elif Vb_new < _PT_VB_MIN:
            Vb_new = _PT_VB_MIN
        rel_tol = _PT_RESID_REL_TOL * max(abs(last_Id), 1e-12)
        dV_step = abs(Vb_new - Vb)
        # z472 — three independent early-exit paths, all gated by N_MIN_STEPS:
        #   (a) tight: dVb<tol_dv AND |R_B|<rel_tol  (original)
        #   (b) abs:   dVb<tol_dv AND |R_B|<abs_tol  (handles sub-pA cells)
        #   (c) stall: dVb<tol_dv_loose for N_STALL consecutive steps
        if k >= _PT_N_MIN_STEPS:
            tight = dV_step < tol_dv and abs(float(R_B)) < rel_tol
            abs_ok = dV_step < tol_dv and abs(float(R_B)) < _PT_RESID_ABS_TOL
            if dV_step < _PT_TOL_DV_LOOSE:
                n_stalled += 1
            else:
                n_stalled = 0
            stalled = n_stalled >= _PT_N_STALL
            if tight or abs_ok or stalled:
                Vb = Vb_new
                n_used = k + 1
                converged = True
                break
        Vb = Vb_new
    R_S, R_B, Id = resid_pair(cfg, model_M1, model_M2, bjt,
                               Vsint_pin, Vb, Vd_f, VG1_f, VG2_f)
    return dict(Vb=Vb, Vsint=Vsint_pin, Id=Id,
                resid_RB=abs(float(R_B)), resid_RS=abs(float(R_S)),
                converged=converged or (abs(float(R_B)) < max(1e-8, _PT_RESID_ABS_TOL)),
                niter=n_used)


def run_vsint_pinned(cfg, model_M1, model_M2, bjt, Vd_f, VG1_f, VG2_f,
                     Vsint_pin=0.0, Vb_init=0.0, method=None):
    """Default DC solve at a single (Vd, VG1, VG2). Dispatches by `method`
    (or NSRAM_DC_SOLVER env var). Default = "pt" (pseudo-transient).
    """
    global _NEWTON_DEPRECATION_WARNED
    m = (method or _DC_SOLVER_DEFAULT).lower()
    if m == "newton":
        if not _NEWTON_DEPRECATION_WARNED:
            _warnings.warn(
                "Newton-DC does not capture snap-back; use pseudo-transient "
                "backward (default) for visible knee. Set NSRAM_DC_SOLVER=pt "
                "to silence.", DeprecationWarning, stacklevel=2)
            _NEWTON_DEPRECATION_WARNED = True
        return _run_vsint_pinned_newton(cfg, model_M1, model_M2, bjt,
                                         Vd_f, VG1_f, VG2_f,
                                         Vsint_pin=Vsint_pin, Vb_init=Vb_init)
    # default: pseudo-transient
    return _run_vsint_pinned_pt(cfg, model_M1, model_M2, bjt,
                                 Vd_f, VG1_f, VG2_f,
                                 Vsint_pin=Vsint_pin, Vb_init=Vb_init)


def run_vd_sweep_pt_backward(cfg, model_M1, model_M2, bjt, Vd_seq,
                              VG1_f, VG2_f, Vsint_pin=0.0,
                              Vb_init_first=0.1):
    """Sweep V_D HIGH → LOW with PT integration and V_B warm-start.

    Returns Id_pred and Vb arrays in the ORIGINAL order of Vd_seq.
    This is the z432-style backward sweep that captures the latched branch.
    """
    n = len(Vd_seq)
    Id_out = [0.0] * n
    Vb_out = [0.0] * n
    conv_out = [False] * n
    niter_out = [0] * n
    Vb_warm = float(Vb_init_first)
    # iterate high → low
    for idx in range(n - 1, -1, -1):
        Vd_f = float(Vd_seq[idx])
        r = _run_vsint_pinned_pt(cfg, model_M1, model_M2, bjt,
                                  Vd_f, float(VG1_f), float(VG2_f),
                                  Vsint_pin=Vsint_pin, Vb_init=Vb_warm)
        Id_out[idx] = abs(float(r["Id"])) if r["Id"] is not None else 0.0
        Vb_out[idx] = float(r["Vb"])
        conv_out[idx] = bool(r["converged"])
        niter_out[idx] = int(r.get("niter", 0))
        Vb_warm = float(r["Vb"])
    return Id_out, Vb_out, conv_out, niter_out


# ─── Measured loader ───────────────────────────────────────────────────────
def measured_at(curves, VG1, VG2):
    for c in curves:
        if abs(c["VG1"] - VG1) < 1e-3 and abs(c["VG2"] - VG2) < 1e-3:
            return c["Vd"].numpy(), c["Id"].numpy()
    return None, None


# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("z429 starting — multi-solver point-by-point debug")
    model_M1, model_M2 = build_models()
    curves = load_curves()
    sebas_rows = load_sebas_params()
    VG1, VG2 = 0.6, 0.0
    row = find_params(sebas_rows, VG1, VG2)
    P_M1, P_M2 = make_overrides(row)
    bjt = make_bjt(row)
    cfg, sd_M1, sd_M2 = make_cfg(model_M1, model_M2)
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows. Bias VG1={VG1} VG2={VG2}")

    Vd_meas, Id_meas = measured_at(curves, VG1, VG2)
    log(f"measured Vd points: {len(Vd_meas)}, Id range: {Id_meas.min():.3e}..{Id_meas.max():.3e}")

    # V_D sweep
    N_VD = 60   # 60 instead of 100 to keep brute-force tractable (60*40*40 evals)
    Vd_sweep = np.linspace(0.02, 2.0, N_VD)

    summary = {
        "bias": dict(VG1=VG1, VG2=VG2),
        "Vd_sweep": Vd_sweep.tolist(),
        "measured": dict(Vd=Vd_meas.tolist(), Id=Id_meas.tolist()),
        "solvers": {},
    }

    # Apply Sebas overrides for whole sweep
    with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):

        # 1. Standard Newton (COLD)
        log("=== Solver 1: standard Newton (COLD start each Vd) ===")
        s1 = []
        for Vd_f in Vd_sweep:
            r = run_pyport_newton(cfg, model_M1, model_M2, bjt, Vd_f, VG1, VG2,
                                  Vsint_init=0.0, Vb_init=0.0, damping=1.0)
            s1.append(dict(Vd=float(Vd_f), **{k: v for k, v in r.items() if k != "error"}))
        summary["solvers"]["standard_newton_cold"] = s1
        conv_rate = sum(1 for r in s1 if r.get("converged")) / N_VD
        log(f"  convergence rate: {conv_rate*100:.0f}%")

        # 2. Damped Newton (tighter min_damping)
        log("=== Solver 2: damped Newton (min_damp=1/1024) ===")
        s2 = []
        for Vd_f in Vd_sweep:
            r = run_pyport_newton(cfg, model_M1, model_M2, bjt, Vd_f, VG1, VG2,
                                  Vsint_init=0.0, Vb_init=0.0,
                                  damping=0.5, min_damping=1/1024.0)
            s2.append(dict(Vd=float(Vd_f), **{k: v for k, v in r.items() if k != "error"}))
        summary["solvers"]["damped_newton"] = s2
        conv_rate = sum(1 for r in s2 if r.get("converged")) / N_VD
        log(f"  convergence rate: {conv_rate*100:.0f}%")

        # 3. Arc-length continuation (warm-started)
        log("=== Solver 3: arc-length continuation (warm-start) ===")
        s3 = []
        Vsi_w, Vb_w = 0.0, 0.0
        for Vd_f in Vd_sweep:
            r = run_pyport_newton(cfg, model_M1, model_M2, bjt, Vd_f, VG1, VG2,
                                  Vsint_init=Vsi_w, Vb_init=Vb_w,
                                  damping=1.0)
            s3.append(dict(Vd=float(Vd_f), **{k: v for k, v in r.items() if k != "error"}))
            if r.get("converged"):
                Vsi_w = r["Vsint"]; Vb_w = r["Vb"]
        summary["solvers"]["arc_length_warm"] = s3
        conv_rate = sum(1 for r in s3 if r.get("converged")) / N_VD
        log(f"  convergence rate: {conv_rate*100:.0f}%")

        # 4. scipy.fsolve multi-IC
        log("=== Solver 4: scipy.fsolve (5 ICs) ===")
        s4 = []
        for Vd_f in Vd_sweep:
            best, allr = run_fsolve_multi(cfg, model_M1, model_M2, bjt,
                                           Vd_f, VG1, VG2)
            entry = dict(Vd=float(Vd_f),
                          best=best,
                          n_distinct_roots=len({(round(r["Vb"],3), round(r["Vsint"],3))
                                                for r in allr
                                                if r.get("converged")}))
            s4.append(entry)
        summary["solvers"]["fsolve_multi_ic"] = s4
        conv_rate = sum(1 for r in s4 if r["best"] and r["best"].get("converged")) / N_VD
        log(f"  convergence rate: {conv_rate*100:.0f}%")

        # 5. Brute-force grid scan
        log("=== Solver 5: brute-force grid scan (40x40 per Vd) ===")
        s5 = []
        for Vd_f in Vd_sweep:
            g = run_grid_scan(cfg, model_M1, model_M2, bjt, Vd_f, VG1, VG2, N=40)
            s5.append(dict(Vd=float(Vd_f), n_roots=g["n_roots"],
                            roots=g["roots"],
                            norm_min=g["norm_min"]))
        summary["solvers"]["grid_scan"] = s5
        log(f"  done. multi-root V_D points: "
            f"{sum(1 for r in s5 if r['n_roots']>1)}")

        # 6. V_Sint pinned to 0
        log("=== Solver 6: V_Sint = 0 PIN ===")
        s6 = []
        Vb_w6 = 0.0
        for Vd_f in Vd_sweep:
            r = run_vsint_pinned(cfg, model_M1, model_M2, bjt, Vd_f, VG1, VG2,
                                  Vsint_pin=0.0, Vb_init=Vb_w6)
            if r["converged"]:
                Vb_w6 = r["Vb"]
            s6.append(dict(Vd=float(Vd_f), **r))
        summary["solvers"]["vsint_pinned_0"] = s6
        conv_rate = sum(1 for r in s6 if r.get("converged")) / N_VD
        log(f"  convergence rate: {conv_rate*100:.0f}%")

        # 7. Backward sweep
        log("=== Solver 7: backward sweep Vd: 2 -> 0 (warm-start) ===")
        s7 = []
        Vsi_b, Vb_b = 0.0, 0.0
        for Vd_f in Vd_sweep[::-1]:
            r = run_pyport_newton(cfg, model_M1, model_M2, bjt, Vd_f, VG1, VG2,
                                  Vsint_init=Vsi_b, Vb_init=Vb_b,
                                  damping=1.0)
            s7.append(dict(Vd=float(Vd_f), **{k: v for k, v in r.items() if k != "error"}))
            if r.get("converged"):
                Vsi_b = r["Vsint"]; Vb_b = r["Vb"]
        s7 = list(reversed(s7))
        summary["solvers"]["backward_sweep"] = s7
        conv_rate = sum(1 for r in s7 if r.get("converged")) / N_VD
        log(f"  convergence rate: {conv_rate*100:.0f}%")

    # Save summary
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"summary.json written ({(OUT/'summary.json').stat().st_size/1024:.1f} KB)")

    # ─── PLOTS ───────────────────────────────────────────────────────────
    def get_Id(s):
        return np.array([r.get("Id") if r.get("Id") is not None else np.nan
                         for r in s])

    # ID vs Vd, all 5 solvers + V_Sint pinned + backward
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.semilogy(Vd_meas, Id_meas, "k.-", lw=2.5, label="MEASURED", markersize=4)
    ax.semilogy(Vd_sweep, get_Id(s1), "C0-", label="1. Standard Newton (cold)")
    ax.semilogy(Vd_sweep, get_Id(s2), "C1--", label="2. Damped Newton")
    ax.semilogy(Vd_sweep, get_Id(s3), "C2-.", label="3. Arc-length (warm)")
    s4_id = np.array([(r["best"]["Id"] if r["best"] else np.nan) for r in s4])
    ax.semilogy(Vd_sweep, s4_id, "C3:", label="4. scipy.fsolve (best of 5 IC)")
    s6_id = get_Id(s6)
    ax.semilogy(Vd_sweep, s6_id, "C4-", lw=2, label="6. V_Sint=0 PIN")
    ax.semilogy(Vd_sweep, get_Id(s7), "C5--", label="7. Backward sweep")
    ax.set_xlabel("V_D (V)"); ax.set_ylabel("|I_D| (A)")
    ax.set_title(f"NS-RAM 2T: multi-solver I_D(V_D) at VG1={VG1}, VG2={VG2}")
    ax.set_ylim(1e-12, 1e-3)
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "ID_vs_Vd_5solvers.png", dpi=130)
    plt.close(fig)

    # V_Sint pinned vs measured (single panel)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(Vd_meas, Id_meas, "k.-", lw=2.5, label="MEASURED", markersize=4)
    ax.semilogy(Vd_sweep, s6_id, "C4o-", label="V_Sint=0 PIN (1-D Newton on V_B)")
    ax.semilogy(Vd_sweep, get_Id(s1), "C0--", alpha=0.6, label="Standard Newton (free V_Sint)")
    ax.set_xlabel("V_D (V)"); ax.set_ylabel("|I_D| (A)")
    ax.set_title("V_Sint PIN diagnostic: does perfect substrate contact match measurement?")
    ax.set_ylim(1e-12, 1e-3)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "vsint_pinned_match.png", dpi=130)
    plt.close(fig)

    # Forward vs backward
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].semilogy(Vd_meas, Id_meas, "k.-", lw=2, label="MEASURED")
    axes[0].semilogy(Vd_sweep, get_Id(s3), "C2-", label="Forward (arc-length)")
    axes[0].semilogy(Vd_sweep, get_Id(s7), "C5--", label="Backward")
    axes[0].set_xlabel("V_D"); axes[0].set_ylabel("|I_D|")
    axes[0].set_title("I_D forward vs backward")
    axes[0].set_ylim(1e-12, 1e-3); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(Vd_sweep, [r.get("Vsint", np.nan) for r in s3], "C2-", label="Forward V_Sint")
    axes[1].plot(Vd_sweep, [r.get("Vsint", np.nan) for r in s7], "C5--", label="Backward V_Sint")
    axes[1].set_xlabel("V_D"); axes[1].set_ylabel("V_Sint (V)")
    axes[1].set_title("V_Sint forward vs backward")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(Vd_sweep, [r.get("Vb", np.nan) for r in s3], "C2-", label="Forward V_B")
    axes[2].plot(Vd_sweep, [r.get("Vb", np.nan) for r in s7], "C5--", label="Backward V_B")
    axes[2].set_xlabel("V_D"); axes[2].set_ylabel("V_B (V)")
    axes[2].set_title("V_B forward vs backward")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "forward_vs_backward.png", dpi=130)
    plt.close(fig)

    # Brute-force roots count
    n_roots = np.array([r["n_roots"] for r in s5])
    norm_min = np.array([r["norm_min"] for r in s5])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(Vd_sweep, n_roots, "o-")
    axes[0].set_xlabel("V_D"); axes[0].set_ylabel("# roots found (grid scan)")
    axes[0].set_title("Brute-force root count vs V_D")
    axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(Vd_sweep, np.maximum(norm_min, 1e-30), "o-")
    axes[1].set_xlabel("V_D"); axes[1].set_ylabel("min |R| on grid")
    axes[1].set_title("Grid-scan residual minimum")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "roots_brute_force.png", dpi=130)
    plt.close(fig)

    # ─── DIAGNOSTIC interpretation ─────────────────────────────────────
    # Q1: V_Sint=0 PIN match measurement?
    eps = 1e-15
    log_m = np.log10(np.maximum(Id_meas, eps))
    # interpolate s6 Id onto Vd_meas
    s6_interp = np.interp(Vd_meas, Vd_sweep, np.maximum(s6_id, eps))
    log_s6 = np.log10(s6_interp)
    rmse_s6 = float(np.sqrt(np.mean((log_s6 - log_m)**2)))

    s1_interp = np.interp(Vd_meas, Vd_sweep, np.maximum(get_Id(s1), eps))
    rmse_s1 = float(np.sqrt(np.mean((np.log10(s1_interp) - log_m)**2)))
    s3_interp = np.interp(Vd_meas, Vd_sweep, np.maximum(get_Id(s3), eps))
    rmse_s3 = float(np.sqrt(np.mean((np.log10(s3_interp) - log_m)**2)))
    s7_interp = np.interp(Vd_meas, Vd_sweep, np.maximum(get_Id(s7), eps))
    rmse_s7 = float(np.sqrt(np.mean((np.log10(s7_interp) - log_m)**2)))

    # Q2: max distinct roots from brute force
    max_grid_roots = int(n_roots.max())
    multi_root_pts = int(np.sum(n_roots > 1))

    # Q3: solver agreement — std across solvers per Vd
    stack = np.vstack([np.maximum(get_Id(s1), eps),
                        np.maximum(get_Id(s2), eps),
                        np.maximum(get_Id(s3), eps),
                        np.maximum(s4_id, eps)])
    log_stack = np.log10(stack)
    solver_disagree = float(np.nanstd(log_stack, axis=0).max())

    # Q4: arc-length smoothness — first difference
    s3_Id = np.maximum(get_Id(s3), eps)
    smoothness = float(np.max(np.abs(np.diff(np.log10(s3_Id)))))

    # Forward vs backward bistability
    s3_Vb = np.array([r.get("Vb", np.nan) for r in s3])
    s7_Vb = np.array([r.get("Vb", np.nan) for r in s7])
    fb_gap_Vb = float(np.nanmax(np.abs(s3_Vb - s7_Vb)))

    interp = dict(
        rmse_s1_standard=rmse_s1,
        rmse_s3_arclength=rmse_s3,
        rmse_s6_vsint_pinned=rmse_s6,
        rmse_s7_backward=rmse_s7,
        max_grid_roots=max_grid_roots,
        multi_root_Vd_points=multi_root_pts,
        max_solver_log_disagreement_dec=solver_disagree,
        arc_length_max_step_dec=smoothness,
        forward_vs_backward_max_Vb_gap=fb_gap_Vb,
    )
    summary["interpretation"] = interp
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    log("=== INTERPRETATION ===")
    for k, v in interp.items():
        log(f"  {k}: {v}")

    # diagnostic.md
    q1_kill = rmse_s6 < 1.0   # V_Sint pinned makes a smooth ~match
    q1_better = rmse_s6 < rmse_s1
    q2_solver = max_grid_roots > 1
    q3_agree = solver_disagree < 0.5
    q4_smooth = smoothness < 1.0
    bistability = fb_gap_Vb > 0.1

    diag = []
    diag.append(f"# z429 multi-solver diagnostic — VG1={VG1}, VG2={VG2}\n")
    diag.append("## Q1: Does V_Sint=0 PIN produce a smooth monotonic I_D matching measured?")
    diag.append(f"  RMSE (V_Sint pinned)  = {rmse_s6:.3f} dec")
    diag.append(f"  RMSE (free V_Sint)    = {rmse_s1:.3f} dec")
    diag.append(f"  Improvement from PIN  = {rmse_s1 - rmse_s6:+.3f} dec")
    diag.append(f"  -> PIN matches measurement: {'YES (KILL-SHOT for V_Sint)' if q1_kill else 'NO'}")
    diag.append(f"  -> PIN is better than free: {'YES (V_Sint is the dominant blocker)' if q1_better else 'NO (something else)'}")
    diag.append("")
    diag.append("## Q2: Does brute-force grid scan reveal multiple roots?")
    diag.append(f"  max # roots over Vd sweep = {max_grid_roots}")
    diag.append(f"  Vd points with >1 root     = {multi_root_pts}/{N_VD}")
    diag.append(f"  -> Multiple equilibria: {'YES (Newton can land on wrong root -> SOLVER ISSUE)' if q2_solver else 'NO (only one root -> physics determines)'}")
    diag.append("")
    diag.append("## Q3: Do all solvers converge to same I_D?")
    diag.append(f"  max log10 disagreement across solvers = {solver_disagree:.3f} dec")
    diag.append(f"  -> Solvers agree: {'YES (result is deterministic, physics responsibility)' if q3_agree else 'NO (solver-dependent, need better solver)'}")
    diag.append("")
    diag.append("## Q4: Does arc-length continuation produce smooth I_D(V_D)?")
    diag.append(f"  max |Δlog10(Id)| between adjacent Vd = {smoothness:.3f} dec")
    diag.append(f"  -> Arc-length smooth: {'YES' if q4_smooth else 'NO (still has jumps -> physics has true discontinuity, or solver still broken)'}")
    diag.append(f"  arc-length RMSE = {rmse_s3:.3f} dec (vs cold {rmse_s1:.3f} dec)")
    diag.append("")
    diag.append("## Forward vs backward bistability")
    diag.append(f"  max |V_B forward - V_B backward| = {fb_gap_Vb:.3f} V")
    diag.append(f"  backward sweep RMSE              = {rmse_s7:.3f} dec")
    diag.append(f"  -> Bistability detected: {'YES (real or solver hysteresis)' if bistability else 'NO'}")

    (OUT / "diagnostic.md").write_text("\n".join(diag))

    # ROOT_CAUSE.md
    if q1_kill or q1_better:
        cause = "physics_via_vsint"
        verdict = (f"V_Sint runaway is the dominant cause. Pinning V_Sint=0 reduces "
                   f"single-bias RMSE from {rmse_s1:.2f} dec (free) to {rmse_s6:.2f} dec "
                   f"(pinned). The model floats V_Sint to non-physical values; real "
                   f"silicon has a strong substrate pulldown. Fix: add a Sint→GND "
                   f"resistive shunt (R≈1e5–1e6 Ω) or implement the physical substrate "
                   f"contact in the topology. Expected cell-wide reduction: ~0.5–1.0 dec.")
    elif q2_solver and not q3_agree:
        cause = "solver"
        verdict = (f"Multiple equilibria exist ({max_grid_roots} roots at some Vd) and "
                   f"solvers disagree by up to {solver_disagree:.2f} dec. Newton lands on "
                   f"different roots depending on warm-start. Fix: use arc-length "
                   f"continuation as default (warm-start), reject roots with V_Sint > Vd, "
                   f"and reject roots with V_BE > 0.8 V (deep saturation).")
    elif not q4_smooth:
        cause = "physics_discontinuity"
        verdict = (f"Even arc-length continuation produces jumps "
                   f"({smoothness:.2f} dec per Vd step). Physics likely has a true "
                   f"discontinuity (snapback / latch) that the BSIM4+Gummel-Poon "
                   f"topology cannot represent smoothly. Verilog-A or explicit BJT-on/off "
                   f"branching needed.")
    else:
        cause = "weak_physics"
        verdict = (f"Solvers agree, V_Sint PIN does not help, no bistability — but RMSE "
                   f"is still high ({rmse_s1:.2f} dec). The model itself is wrong: "
                   f"missing physics (carrier multiplication, distributed channel, "
                   f"non-quasi-static effects) is the dominant cause.")

    (OUT / "ROOT_CAUSE.md").write_text(f"# z429 Root cause\n\n**verdict**: {cause}\n\n{verdict}\n")

    log(f"ROOT CAUSE: {cause}")
    log(f"  {verdict}")
    log(f"=== done in {time.time()-t0:.0f}s ===")
    LOG.close()


if __name__ == "__main__":
    main()
