"""Common helpers: build calibrated NS-RAM model + implicit-layer wrapper.

Implements the MEP-6 differentiable forward via the implicit-function theorem
(IFT) wrapped around the existing `_residuals` from nsram_cell_2T. Internally:

  1. Newton iteration runs under torch.no_grad() to find (Vsint*, Vb*) s.t.
     R(Vsint*, Vb*; VG1, VG2, Vd) ~= 0.
  2. At convergence, gradient w.r.t. inputs (VG1, VG2, Vd) is given by
     IFT:  dx*/dtheta = -J^-1 . dR/dtheta. We encode this in autograd by
     evaluating  x_attached = x*_detached - J^-1 R(x*_detached, theta_grad),
     then re-running _residuals to get Id with grad flowing through theta.

This works on both CPU and CUDA tensors.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import numpy as np
import torch

# Allow flexible repo root via env var GPU_MAX_A_ROOT or argv
_ROOT = Path(os.environ.get("GPU_MAX_A_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(_ROOT / "nsram"))
sys.path.insert(0, str(_ROOT))

from nsram.bsim4_port.nsram_cell_2T import _residuals, NSRAMCell2TConfig  # noqa
from nsram.bsim4_port.model_card import BSIM4Model                         # noqa
from nsram.bsim4_port.bjt import GummelPoonNPN                              # noqa


# ---------------------------------------------------------------------------
# Model builder (z471 calibrated SNAP cell)
# ---------------------------------------------------------------------------
SNAP_IS_CAL = 4.5192e-12  # from results/z471_snap_calibrate/calibration_summary.json


def build_calibrated_models(data_dir: Path | None = None):
    """Build M1, M2 BSIM4 model cards (mirrors z96.build_calibrated_models)."""
    if data_dir is None:
        # Try common paths
        cands = [
            _ROOT / "data/sebas_2026_04_22",
            _ROOT / "nsram/data/sebas_2026_04_22",
            _ROOT / "data",
            _ROOT / "nsram/data",
        ]
        for c in cands:
            if (c / "M1_130DNWFB.txt").exists():
                data_dir = c; break
        if data_dir is None:
            raise FileNotFoundError(
                f"Could not find M1_130DNWFB.txt under {cands}")
    text_M1 = (data_dir / "M1_130DNWFB.txt").read_text()
    text_M2 = (data_dir / "M2_130bulkNSRAM.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    # Patch values (mirrors what z96 imports via 'f.patch_model_values')
    try:
        from nsram.bsim4_port import dc as _dc  # noqa
        if hasattr(_dc, "patch_model_values"):
            _dc.patch_model_values(M1, type_n=True)
            _dc.patch_model_values(M2, type_n=True)
    except Exception:
        pass
    return M1, M2


def build_nsram_stack(*, use_snapback: bool = False, device: str = "cpu"):
    """Return (cfg, M1, M2, bjt) calibrated to the z471 SNAP_CAL point.

    use_snapback=True turns on the snapback subcircuit with snap_Is=4.5192e-12.
    """
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=25)
    M1, M2 = build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    # Production calibration from nsram_surrogate_4d.py
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    if use_snapback:
        # Mirror snap_cfg() in z471_dc_check.py
        for k, v in dict(
            use_snapback_sub=True, snap_method="snapback",
            snap_Bf=417.0, snap_Va=0.90, snap_Is=SNAP_IS_CAL, snap_Nf=1.0,
            snap_BV=2.0*0.6, snap_n_avl=4.0,
            snap_Id_clamp=1e-1, snap_Iii_clamp=1e-1,
            snap_use_knee_gate=True,
            snap_V_knee=1.6, snap_V_sharp=0.05,
            snap_npn_gate_mode="current",
            snap_npn_V_knee=1.8, snap_npn_V_sharp=0.05,
            snap_npn_V_BE_offset=0.3,
        ).items():
            setattr(cfg, k, v)
    return cfg, M1, M2, bjt


# ---------------------------------------------------------------------------
# Implicit-layer differentiable forward
# ---------------------------------------------------------------------------
def _residuals_pair(cfg, M1, M2, bjt, Vd, VG1, VG2, Vsint, Vb):
    R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd=Vd, VG1=VG1, VG2=VG2,
                                Vsint=Vsint, Vb=Vb, model_M2=M2)
    return R_S, R_B, comp


def _jac_fd(cfg, M1, M2, bjt, Vd, VG1, VG2, Vsint, Vb, eps=1e-5):
    R_S0, R_B0, _ = _residuals_pair(cfg, M1, M2, bjt, Vd, VG1, VG2, Vsint, Vb)
    R_S_s, R_B_s, _ = _residuals_pair(cfg, M1, M2, bjt, Vd, VG1, VG2, Vsint+eps, Vb)
    R_S_b, R_B_b, _ = _residuals_pair(cfg, M1, M2, bjt, Vd, VG1, VG2, Vsint, Vb+eps)
    dRS_dVs = (R_S_s - R_S0) / eps
    dRS_dVb = (R_B_s - R_B0) / eps    # row index for R_S? -- careful
    # Actually we want J[i,j] = dR_i/dx_j with i in {S,B} and j in {Vs,Vb}
    dRS_dVs = (R_S_s - R_S0) / eps
    dRB_dVs = (R_B_s - R_B0) / eps
    dRS_dVb = (R_S_b - R_S0) / eps
    dRB_dVb = (R_B_b - R_B0) / eps
    J = torch.stack([
        torch.stack([dRS_dVs, dRS_dVb], dim=-1),
        torch.stack([dRB_dVs, dRB_dVb], dim=-1),
    ], dim=-2)
    return J, R_S0, R_B0


def _solve_2x2(R_S, R_B, J, ridge=1e-30):
    rhs = -torch.stack([R_S, R_B], dim=-1).unsqueeze(-1)
    eye = torch.eye(2, dtype=J.dtype, device=J.device).expand_as(J) * ridge
    sol = torch.linalg.solve(J + eye, rhs)
    return sol[..., 0, 0], sol[..., 1, 0]


def diff_forward_id(cfg, M1, M2, bjt,
                    Vd, VG1, VG2,
                    *,
                    Vsint_init=0.1, Vb_init=0.3,
                    max_iters=30, tol=1e-10,
                    Vb_lo=-0.5, Vb_hi=1.2,
                    step_clamp=0.5, damping=1.0,
                    delta_bound=0.3,
                    dtype=torch.float64):
    """Differentiable batched forward. Each input is broadcast to common (N,).

    Returns dict with: Id, Vsint, Vb, converged, R_max (all (N,) tensors).
    Id supports autograd w.r.t. Vd, VG1, VG2 via IFT attachment.
    """
    device = Vd.device if torch.is_tensor(Vd) else VG1.device
    Vd  = torch.as_tensor(Vd,  dtype=dtype, device=device)
    VG1 = torch.as_tensor(VG1, dtype=dtype, device=device)
    VG2 = torch.as_tensor(VG2, dtype=dtype, device=device)
    # Broadcast to common shape
    Vd, VG1, VG2 = torch.broadcast_tensors(Vd, VG1, VG2)
    Vd = Vd.contiguous(); VG1 = VG1.contiguous(); VG2 = VG2.contiguous()
    N = Vd.numel()

    # ---- Phase 1: Newton iteration under no_grad -----
    with torch.no_grad():
        Vsint = torch.full_like(Vd, float(Vsint_init))
        Vb    = torch.full_like(Vd, float(Vb_init))
        R_max_last = None
        for k in range(max_iters):
            J, R_S, R_B = _jac_fd(cfg, M1, M2, bjt, Vd, VG1, VG2, Vsint, Vb)
            R_max = torch.maximum(R_S.abs(), R_B.abs())
            R_max_last = R_max
            if bool((R_max < tol).all()):
                break
            try:
                dVs, dVb = _solve_2x2(R_S, R_B, J)
            except Exception:
                break
            dVs = dVs.clamp(-step_clamp, step_clamp)
            dVb = dVb.clamp(-step_clamp, step_clamp)
            Vsint = Vsint + damping * dVs
            Vb    = (Vb    + damping * dVb).clamp(Vb_lo, Vb_hi)
        converged = (R_max_last < tol * 1e3)

    # ---- Phase 2: IFT attachment ----
    Vsint_d = Vsint.detach()
    Vb_d    = Vb.detach()
    # Residual computed with theta carrying grad, x detached:
    R_S_at, R_B_at, _ = _residuals_pair(cfg, M1, M2, bjt, Vd, VG1, VG2, Vsint_d, Vb_d)
    with torch.no_grad():
        J_final, _, _ = _jac_fd(cfg, M1, M2, bjt, Vd, VG1, VG2, Vsint_d, Vb_d)
    delta_s, delta_b = _solve_2x2(R_S_at, R_B_at, J_final)
    # Smooth-clamp via tanh (passes grad everywhere, magnitude-bounded)
    delta_s = delta_bound * torch.tanh(delta_s / delta_bound)
    delta_b = delta_bound * torch.tanh(delta_b / delta_bound)
    conv_mask = converged.detach()
    delta_s = torch.where(conv_mask, delta_s, torch.zeros_like(delta_s))
    delta_b = torch.where(conv_mask, delta_b, torch.zeros_like(delta_b))
    # NOTE: IFT sign. _solve_2x2 returns Newton step dx = -J^{-1} R, i.e.
    # delta_s = -[J^{-1} R]_s.  To attach IFT (dx*/dθ = -J^{-1} dR/dθ),
    # we want x_a = x*_d - J^{-1} R(x*_d, θ). Substituting J^{-1} R = -delta_s,
    # x_a = x*_d - (-delta_s) = x*_d + delta_s.  (The legacy IFT code in
    # nsram_cell_2T.py used `- delta_s`; that sign is wrong and is the
    # root cause of the dId/dVG2 sign-flip observed in the grad verify.)
    Vsint_attached = Vsint_d + delta_s
    Vb_attached    = Vb_d    + delta_b

    # Final Id with full grad path
    R_S, R_B, comp = _residuals_pair(cfg, M1, M2, bjt, Vd, VG1, VG2,
                                     Vsint_attached, Vb_attached)
    Id = (comp["Ids_M1"]
          + comp["Ic_Q1"]
          + comp.get("Ic_Q2", 0.0)
          + comp.get("Ic_lat", 0.0)
          + comp.get("Ic_avalanche", 0.0)
          + comp["Igidl_M1"]
          - comp["Ibd_M1"]
          - comp.get("Ie_vert", 0.0)
          + comp.get("I_snap_d", 0.0))
    return {"Id": Id, "Vsint": Vsint_attached, "Vb": Vb_attached,
            "converged": converged, "R_max": R_max_last}
