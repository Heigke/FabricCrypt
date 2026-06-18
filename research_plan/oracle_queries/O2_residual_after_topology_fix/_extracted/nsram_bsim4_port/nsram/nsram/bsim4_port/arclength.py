"""Pseudo-arclength continuation for the 2T NS-RAM cell.

Snapback is a fold bifurcation of the I-V curve: at some Vd_fold the equation
R(Vsint, Vb; Vd) = 0 has a singular Jacobian and there are two valid
(Vsint, Vb) roots (off-branch low-current, on-branch high-current). Plain
Newton on Vd-parameterized residual is undefined at the fold — that's why
gmin homotopy + warm-start from previous Vd still produces the "lines hopping
between roots" we observed in z93.

Pseudo-arclength continuation handles this by treating Vd itself as a state
variable and parameterizing the (Vsint, Vb, Vd) curve by arclength s. The
Jacobian of the augmented system is non-singular at the fold (the tangent
vector simply rotates so that ds/dVd reverses sign there).

Reference: Kelley & Keyes (1998) Convergence Analysis of Pseudo-Transient
Continuation; AUTO/MatCont/LOCA literature for fold-bifurcation tracking.

Implementation:
- 3D state x = (Vsint, Vb, Vd)
- 3D residual F(x) = (R_S, R_B, t·(x - x_prev) - ds)
  where R_S, R_B are the original 2T body-KCL residuals (re-using
  `nsram_cell_2T._residuals`) and the third equation is the arclength
  constraint orthogonal to the tangent t.
- Tangent computed from the 2x3 Jacobian by solving J·t = 0 normalized.
- Adaptive ds based on Newton iteration count.
- Returns Id, Vsint, Vb at user-requested Vd_targets via piecewise-linear
  interpolation along the traced path. For points where the path crosses
  Vd_target multiple times (snapback hysteresis), takes the FIRST crossing
  (= forward-sweep convention).
"""
from __future__ import annotations
from typing import Optional
import torch

from .nsram_cell_2T import (
    NSRAMCell2TConfig, _residuals, _jacobian_finite_diff,
)
from .model_card import BSIM4Model
from .bjt import GummelPoonNPN


def _residual_dVd(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                   h: float = 1e-6, model_M2=None) -> torch.Tensor:
    """Finite-difference partial derivatives ∂(R_S, R_B)/∂Vd.
    Returns shape (2,) tensor for scalar inputs.
    """
    with torch.no_grad():
        Rsp, Rbp, _ = _residuals(cfg, model, bjt, Vd + h, VG1, VG2,
                                  Vsint, Vb, P_M1, P_M2, model_M2=model_M2)
        Rsm, Rbm, _ = _residuals(cfg, model, bjt, Vd - h, VG1, VG2,
                                  Vsint, Vb, P_M1, P_M2, model_M2=model_M2)
        dRs_dVd = (Rsp - Rsm) / (2 * h)
        dRb_dVd = (Rbp - Rbm) / (2 * h)
    return torch.stack([dRs_dVd, dRb_dVd])


def _solve_initial_point(cfg, model, bjt, Vd0, VG1, VG2,
                          Vsint_init=None, Vb_init=None,
                          max_iters: int = 30, tol: float = 1e-9,
                          model_M2=None):
    """Plain Newton at Vd=Vd0 to find the starting point on the curve."""
    Vd0 = torch.as_tensor(Vd0, dtype=torch.float64)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    Vsint = torch.tensor(0.5 * float(Vd0) if Vsint_init is None else float(Vsint_init),
                          dtype=torch.float64)
    Vb = torch.tensor(0.0 if Vb_init is None else float(Vb_init),
                       dtype=torch.float64)
    for _ in range(max_iters):
        with torch.no_grad():
            R_S, R_B, _ = _residuals(cfg, model, bjt, Vd0, VG1, VG2,
                                       Vsint, Vb, None, None, model_M2=model_M2)
            R = torch.stack([R_S, R_B])
            if R.norm() < tol:
                return Vsint, Vb, True
            J = _jacobian_finite_diff(cfg, model, bjt, Vd0, VG1, VG2,
                                       Vsint, Vb, None, None, model_M2=model_M2)
            try:
                dx = torch.linalg.solve(J, -R)
            except Exception:
                dx = torch.linalg.lstsq(J, -R.unsqueeze(-1)).solution.squeeze(-1)
            # Damped step
            alpha = 1.0
            for _ in range(10):
                Vsint_t = Vsint + alpha * dx[0]
                Vb_t = Vb + alpha * dx[1]
                R_S_t, R_B_t, _ = _residuals(cfg, model, bjt, Vd0, VG1, VG2,
                                               Vsint_t, Vb_t, None, None,
                                               model_M2=model_M2)
                if torch.stack([R_S_t, R_B_t]).norm() < R.norm():
                    Vsint, Vb = Vsint_t, Vb_t
                    break
                alpha *= 0.5
            else:
                Vsint, Vb = Vsint + dx[0], Vb + dx[1]  # accept anyway
    return Vsint, Vb, False


def _compute_tangent(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                      prev_t: Optional[torch.Tensor] = None,
                      model_M2=None) -> torch.Tensor:
    """Tangent vector to the curve at (Vsint, Vb, Vd). Returns shape (3,).

    The 2x3 augmented Jacobian J_aug = [∂R/∂Vsint | ∂R/∂Vb | ∂R/∂Vd] has
    null-space dimension 1 (assuming we're not at a true bifurcation). The
    null-space vector is the tangent. We compute it via SVD of the 2x3
    matrix and pick the right-singular vector with smallest singular value.
    """
    with torch.no_grad():
        J_xy = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                       Vsint, Vb, P_M1, P_M2,
                                       model_M2=model_M2)            # (2,2)
        J_z = _residual_dVd(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb,
                              P_M1, P_M2, model_M2=model_M2)         # (2,)
        J_aug = torch.cat([J_xy, J_z.unsqueeze(-1)], dim=-1)          # (2,3)

        # SVD to find null vector
        _, S, Vh = torch.linalg.svd(J_aug, full_matrices=True)
        t = Vh[-1]  # right-singular vector with smallest sigma

        # Sign convention: ensure consistent direction across steps
        if prev_t is not None:
            if torch.dot(t, prev_t) < 0:
                t = -t
        else:
            # Initial step: prefer increasing Vd direction
            if t[2] < 0:
                t = -t
        # Normalize (numerically robust)
        t = t / t.norm().clamp_min(1e-30)
    return t


def _newton_arclength_corrector(cfg, model, bjt, x_pred, x_prev, t_prev, ds,
                                  VG1, VG2, P_M1, P_M2,
                                  max_iters: int = 15, tol: float = 1e-9,
                                  model_M2=None):
    """3D Newton on augmented system [R_S; R_B; t·(x - x_prev) - ds] = 0.

    Returns (x_new, n_iter, converged).
    """
    x = x_pred.clone()
    for it in range(max_iters):
        Vsint, Vb, Vd = x[0], x[1], x[2]
        with torch.no_grad():
            R_S, R_B, _ = _residuals(cfg, model, bjt, Vd, VG1, VG2,
                                       Vsint, Vb, P_M1, P_M2, model_M2=model_M2)
            constraint = torch.dot(t_prev, x - x_prev) - ds
            F = torch.stack([R_S, R_B, constraint])

            if F.norm() < tol:
                return x, it, True

            # 3x3 Jacobian: top 2 rows = [J_xy | J_z], bottom row = t_prev
            J_xy = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                           Vsint, Vb, P_M1, P_M2, model_M2=model_M2)
            J_z = _residual_dVd(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb,
                                  P_M1, P_M2, model_M2=model_M2)
            top = torch.cat([J_xy, J_z.unsqueeze(-1)], dim=-1)
            J_full = torch.cat([top, t_prev.unsqueeze(0)], dim=0)

            try:
                dx = torch.linalg.solve(J_full, -F)
            except Exception:
                dx = torch.linalg.lstsq(J_full, -F.unsqueeze(-1)).solution.squeeze(-1)

            # Damped step
            alpha = 1.0
            x_old = x.clone()
            for _ in range(8):
                x_try = x_old + alpha * dx
                R_S_t, R_B_t, _ = _residuals(cfg, model, bjt, x_try[2], VG1, VG2,
                                               x_try[0], x_try[1], P_M1, P_M2,
                                               model_M2=model_M2)
                F_try = torch.stack([
                    R_S_t, R_B_t,
                    torch.dot(t_prev, x_try - x_prev) - ds,
                ])
                if F_try.norm() < F.norm():
                    x = x_try
                    break
                alpha *= 0.5
            else:
                x = x_old + dx
    # Did not converge within max_iters
    return x, max_iters, False


def trace_arclength(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    VG1, VG2,
    Vd_start: float = 0.05,
    Vd_max: float = 1.95,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    ds_init: float = 0.01,
    ds_min: float = 1e-4,
    ds_max: float = 0.05,
    max_steps: int = 2000,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Trace I-V curve via pseudo-arclength continuation from Vd_start to Vd_max.

    Returns dict with arrays:
      'path_Vd', 'path_Vsint', 'path_Vb', 'path_Id'  : (N,) along arclength
      'converged'                                    : (N,) bool
      'n_steps', 'n_folds'                           : diagnostics
    """
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)

    # 1. Find initial point
    Vsint0, Vb0, init_ok = _solve_initial_point(cfg, model, bjt, Vd_start,
                                                  VG1, VG2, model_M2=model_M2)
    if not init_ok:
        return {"path_Vd": [Vd_start], "path_Vsint": [float(Vsint0)],
                "path_Vb": [float(Vb0)], "path_Id": [float("nan")],
                "converged": [False], "n_steps": 0, "n_folds": 0,
                "init_ok": False}

    # 2. Trace
    x = torch.tensor([float(Vsint0), float(Vb0), float(Vd_start)],
                      dtype=torch.float64)
    t = _compute_tangent(cfg, model, bjt, x[2], VG1, VG2, x[0], x[1],
                          P_M1, P_M2, prev_t=None, model_M2=model_M2)
    ds = ds_init

    path_Vd = [float(x[2])]
    path_Vsint = [float(x[0])]
    path_Vb = [float(x[1])]
    converged_flags = [True]
    n_folds = 0
    n_steps = 0
    prev_dVd_sign = torch.sign(t[2])

    for step in range(max_steps):
        # Predictor
        x_pred = x + ds * t

        # Corrector
        x_new, n_iter, conv = _newton_arclength_corrector(
            cfg, model, bjt, x_pred, x_prev=x, t_prev=t, ds=ds,
            VG1=VG1, VG2=VG2, P_M1=P_M1, P_M2=P_M2,
            model_M2=model_M2,
        )

        if not conv:
            # Step too large — bisect
            ds = max(ds * 0.5, ds_min)
            if ds <= ds_min * 1.01:
                # Even at min step we can't converge — record and break
                path_Vd.append(float(x_new[2]))
                path_Vsint.append(float(x_new[0]))
                path_Vb.append(float(x_new[1]))
                converged_flags.append(False)
                break
            continue

        # Compute new tangent (with sign consistency)
        t_new = _compute_tangent(cfg, model, bjt, x_new[2], VG1, VG2,
                                   x_new[0], x_new[1], P_M1, P_M2,
                                   prev_t=t, model_M2=model_M2)
        # Detect fold: dVd/ds sign change
        new_dVd_sign = torch.sign(t_new[2])
        if new_dVd_sign != prev_dVd_sign and abs(prev_dVd_sign) > 0:
            n_folds += 1
        prev_dVd_sign = new_dVd_sign

        x = x_new
        t = t_new
        n_steps += 1

        path_Vd.append(float(x[2]))
        path_Vsint.append(float(x[0]))
        path_Vb.append(float(x[1]))
        converged_flags.append(True)

        # Adapt ds
        if n_iter > 8:
            ds = max(ds * 0.7, ds_min)
        elif n_iter <= 3:
            ds = min(ds * 1.3, ds_max)

        # Termination: reached Vd_max in forward direction (may have folded back)
        if x[2] >= Vd_max:
            break
        # Stuck termination: if Vd has been stagnant for too many steps
        if step > 50 and abs(path_Vd[-1] - path_Vd[-50]) < 1e-3 and n_folds == 0:
            break

    # 3. Compute Id along path (run forward at each path point)
    path_Vd_t = torch.tensor(path_Vd, dtype=torch.float64)
    path_Vsint_t = torch.tensor(path_Vsint, dtype=torch.float64)
    path_Vb_t = torch.tensor(path_Vb, dtype=torch.float64)
    with torch.no_grad():
        _, _, comp = _residuals(cfg, model, bjt,
                                 path_Vd_t, VG1.expand_as(path_Vd_t),
                                 VG2.expand_as(path_Vd_t),
                                 path_Vsint_t, path_Vb_t,
                                 P_M1, P_M2, model_M2=model_M2)
        # comp contains M1/M2/Q1 currents — Id at drain pin
        Id = comp.get("Id_total", comp.get("Ids_M1", torch.zeros_like(path_Vd_t)))

    return {
        "path_Vd": path_Vd,
        "path_Vsint": path_Vsint,
        "path_Vb": path_Vb,
        "path_Id": [float(x) for x in Id],
        "converged": converged_flags,
        "n_steps": n_steps,
        "n_folds": n_folds,
        "init_ok": True,
    }


def interpolate_at_targets(path: dict, Vd_targets: torch.Tensor) -> dict:
    """Interpolate Id at requested Vd_targets along the arclength path.

    For points where Vd_target is bracketed by two consecutive path points
    on the FORWARD-sweep portion (before the first fold or after the second
    fold for an off→on transition), use linear interpolation.

    For Vd_targets BEYOND the path's last reached Vd, mark as not-converged
    and return Id=nan.
    """
    import numpy as np
    Vd_arr = np.array(path["path_Vd"])
    Id_arr = np.array(path["path_Id"])
    Vsint_arr = np.array(path["path_Vsint"])
    Vb_arr = np.array(path["path_Vb"])

    Vd_targets_np = Vd_targets.detach().cpu().numpy() if isinstance(Vd_targets, torch.Tensor) else np.asarray(Vd_targets)

    Id_out = np.full_like(Vd_targets_np, np.nan, dtype=np.float64)
    Vsint_out = np.full_like(Vd_targets_np, np.nan, dtype=np.float64)
    Vb_out = np.full_like(Vd_targets_np, np.nan, dtype=np.float64)
    conv_out = np.zeros_like(Vd_targets_np, dtype=bool)

    # For each target, find first segment on path that brackets it.
    for k, Vd_t in enumerate(Vd_targets_np):
        for i in range(len(Vd_arr) - 1):
            v1, v2 = Vd_arr[i], Vd_arr[i + 1]
            if (v1 <= Vd_t <= v2) or (v2 <= Vd_t <= v1):
                # Linear interp
                if abs(v2 - v1) < 1e-12:
                    frac = 0.0
                else:
                    frac = (Vd_t - v1) / (v2 - v1)
                Id_out[k] = Id_arr[i] + frac * (Id_arr[i + 1] - Id_arr[i])
                Vsint_out[k] = Vsint_arr[i] + frac * (Vsint_arr[i + 1] - Vsint_arr[i])
                Vb_out[k] = Vb_arr[i] + frac * (Vb_arr[i + 1] - Vb_arr[i])
                conv_out[k] = (path["converged"][i] and path["converged"][i + 1])
                break

    return {
        "Id": torch.tensor(Id_out, dtype=torch.float64),
        "Vsint": torch.tensor(Vsint_out, dtype=torch.float64),
        "Vb": torch.tensor(Vb_out, dtype=torch.float64),
        "converged": torch.tensor(conv_out),
        "n_steps": path["n_steps"],
        "n_folds": path["n_folds"],
    }


def solve_2t_arclength(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd_seq: torch.Tensor,
    VG1, VG2,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    model_M2: Optional[BSIM4Model] = None,
    **kwargs,
) -> dict:
    """Drop-in replacement for forward_2t — uses arclength continuation."""
    Vd_seq = torch.as_tensor(Vd_seq, dtype=torch.float64)
    Vd_min = float(Vd_seq.min().item())
    Vd_max = float(Vd_seq.max().item())

    path = trace_arclength(cfg, model, bjt, VG1, VG2,
                            Vd_start=Vd_min, Vd_max=Vd_max,
                            P_M1=P_M1, P_M2=P_M2,
                            model_M2=model_M2,
                            **kwargs)

    if not path.get("init_ok", False):
        N = len(Vd_seq)
        return {
            "Id": torch.full((N,), float("nan"), dtype=torch.float64),
            "Vsint": torch.full((N,), float("nan"), dtype=torch.float64),
            "Vb": torch.full((N,), float("nan"), dtype=torch.float64),
            "converged": torch.zeros(N, dtype=torch.bool),
            "niter": torch.zeros(N, dtype=torch.long),
            "n_folds": 0,
            "n_steps": 0,
        }

    out = interpolate_at_targets(path, Vd_seq)
    out["niter"] = torch.full_like(Vd_seq, path["n_steps"], dtype=torch.long)
    return out


def forward_2t_arclength_grad(
    cfg: NSRAMCell2TConfig,
    model: Optional[BSIM4Model] = None,
    bjt: Optional[GummelPoonNPN] = None,
    Vd_seq: Optional[torch.Tensor] = None,
    VG1=None, VG2=None,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    *,
    model_M1: Optional[BSIM4Model] = None,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Drop-in replacement for forward_2t that uses arclength path tracing
    for robust convergence + grad-tracked Newton at each interpolated point
    so gradients still flow to fit params.

    Strategy:
      1. trace_arclength under no_grad → 100% conv path through snapback
      2. interpolate path at Vd_seq → warm-start (Vsint*, Vb*) per bias
      3. solve_2t_steady_state per bias, grad-tracked, Vsint_init from (2).
         Starting at the converged point, Newton needs ~1-2 iterations to
         re-confirm and the autograd graph + IFT correction at convergence
         provide gradient flow.

    Returns dict with same keys as forward_2t: Id, Vsint, Vb, niter, converged.

    Two-model variant: pass `model_M1=` / `model_M2=` as kwargs (or single
    legacy positional `model` for back-compat with the 1-card path).
    """
    from .nsram_cell_2T import solve_2t_steady_state
    if model_M1 is None:
        model_M1 = model
    if model_M1 is None:
        raise TypeError("forward_2t_arclength_grad requires either positional `model` or `model_M1=` kwarg")
    if model_M2 is None:
        model_M2 = model_M1
    Vd_seq = Vd_seq.to(torch.float64)
    VG1_t = torch.as_tensor(VG1, dtype=torch.float64)
    VG2_t = torch.as_tensor(VG2, dtype=torch.float64)
    T = int(Vd_seq.shape[0])

    # 1. Path trace + interpolate (no_grad)
    with torch.no_grad():
        path = trace_arclength(cfg, model_M1, bjt, VG1_t, VG2_t,
                                Vd_start=float(Vd_seq.min()),
                                Vd_max=float(Vd_seq.max()),
                                P_M1=P_M1, P_M2=P_M2,
                                model_M2=model_M2)
        if not path.get("init_ok", False):
            return {
                "Id": torch.full((T,), float("nan"), dtype=torch.float64),
                "Vsint": torch.full((T,), float("nan"), dtype=torch.float64),
                "Vb": torch.full((T,), float("nan"), dtype=torch.float64),
                "converged": torch.zeros(T, dtype=torch.bool),
                "niter": torch.zeros(T, dtype=torch.long),
            }
        warm = interpolate_at_targets(path, Vd_seq)

    Vsint_warm = warm["Vsint"]  # (T,)
    Vb_warm = warm["Vb"]
    arclen_conv = warm["converged"]

    # 2. Per-bias grad-tracked solve from arclength warm-start
    Ids_list, Vs_list, Vb_list = [], [], []
    niter_list, conv_list = [], []
    for i in range(T):
        Vd_i = Vd_seq[i].unsqueeze(0)
        if not bool(arclen_conv[i]):
            # No bracket on path — use plain cascade fallback
            Vs0 = (Vd_i * 0.5).detach()
            Vb0 = torch.tensor(0.0, dtype=torch.float64)
        else:
            Vs0 = Vsint_warm[i].unsqueeze(0).detach()
            Vb0 = Vb_warm[i].unsqueeze(0).detach()
        out = solve_2t_steady_state(
            cfg, model_M1, bjt,
            Vd=Vd_i, VG1=VG1_t, VG2=VG2_t,
            P_M1=P_M1, P_M2=P_M2,
            Vsint_init=Vs0, Vb_init=Vb0,
            model_M2=model_M2,
        )
        Ids_list.append(out["Id"].squeeze(0))
        Vs_list.append(out["Vsint"].squeeze(0))
        Vb_list.append(out["Vb"].squeeze(0))
        niter_list.append(out["niter"] if isinstance(out["niter"], int)
                          else int(out["niter"].squeeze(0).item()))
        conv_val = out["converged"]
        if isinstance(conv_val, torch.Tensor):
            conv_val = bool(conv_val.squeeze(0).item())
        else:
            conv_val = bool(conv_val)
        conv_list.append(conv_val)

    Id_t = torch.stack(Ids_list)
    Vsint_t = torch.stack(Vs_list)
    Vb_t = torch.stack(Vb_list)
    return {
        "Id": Id_t,
        "Vsint": Vsint_t,
        "Vb": Vb_t,
        "converged": torch.tensor(conv_list, dtype=torch.bool),
        "niter": torch.tensor(niter_list, dtype=torch.long),
        "arclen_conv": arclen_conv,
        "arclen_n_steps": path["n_steps"],
        "arclen_n_folds": path["n_folds"],
    }
