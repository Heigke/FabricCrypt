"""Joint (Vsint, Vb) Newton with autograd-exact Jacobian.

Replaces the finite-diff Jacobian in `_jacobian_fd_batched` (which costs
4 extra _residuals calls per step and has discretisation error). Uses
torch.autograd.functional.jacobian to get the exact 2x2 J at machine
precision, with one _residuals call.

Designed to be the shared kernel for:
  A.4 — implicit transient (add Cj·dVb/dt term to R_B before solving)
  B.1 — vectorized batch (stack along extra leading dim)

Reuses existing _residuals; does NOT re-implement physics.
"""
from __future__ import annotations
import torch
from typing import Optional

from .nsram_cell_2T import _residuals, NSRAMCell2TConfig
from .model_card import BSIM4Model
from .bjt import GummelPoonNPN


def _residual_pair(cfg, model_M1, model_M2, bjt, Vd, VG1, VG2,
                    Vsint, Vb, P_M1, P_M2,
                    Vb_prev=None, dt=None):
    """Wrap _residuals → (R_S, R_B) tensor for autograd.

    If `Vb_prev` and `dt` provided, adds the implicit-Euler cap term
    `-Cj(Vb-vnwell)·(Vb-Vb_prev)/dt` to R_B. This converts the DC
    body-KCL into a backward-Euler time-step equation.
    """
    R_S, R_B, _ = _residuals(cfg, model_M1, bjt,
                               Vd=Vd, VG1=VG1, VG2=VG2,
                               Vsint=Vsint, Vb=Vb,
                               P_M1=P_M1, P_M2=P_M2,
                               model_M2=model_M2)
    R_S = R_S.squeeze()
    R_B = R_B.squeeze()
    if Vb_prev is not None and dt is not None:
        from .transient import junction_cap
        Cj0_total = cfg.body_pdiode_Cj0_per_area * cfg.body_pdiode_area
        Cj = junction_cap(Vb.squeeze() - cfg.vnwell, Cj0=Cj0_total,
                            Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
        R_B = R_B - Cj * (Vb.squeeze() - Vb_prev) / dt
    return torch.stack([R_S, R_B])


def joint_newton_step(cfg, model_M1, model_M2, bjt, Vd, VG1, VG2,
                        Vsint: torch.Tensor, Vb: torch.Tensor,
                        P_M1=None, P_M2=None,
                        Vb_prev=None, dt=None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Newton step at a single bias point.
    Returns (Vsint_new, Vb_new, R_norm).
    If Vb_prev/dt given, performs implicit-Euler step (transient).
    """
    Vd_t = Vd.unsqueeze(0) if Vd.dim() == 0 else Vd
    Vsint = Vsint.detach().clone().requires_grad_(False)
    Vb = Vb.detach().clone().requires_grad_(False)
    state = torch.stack([Vsint, Vb]).requires_grad_(True)

    def _f(s):
        return _residual_pair(cfg, model_M1, model_M2, bjt,
                                Vd_t, VG1, VG2,
                                s[0:1], s[1:2], P_M1, P_M2,
                                Vb_prev=Vb_prev, dt=dt)

    R = _f(state)
    J = torch.autograd.functional.jacobian(_f, state, create_graph=False)
    try:
        delta = torch.linalg.solve(J, -R)
    except RuntimeError:
        delta = torch.zeros_like(state)
    new_state = state.detach() + delta.detach()
    return new_state[0], new_state[1], R.detach().abs().max()


def joint_newton_solve(cfg, model_M1, model_M2, bjt, Vd, VG1, VG2,
                        Vsint0: float = 0.1, Vb0: float = 0.5,
                        max_iters: int = 30, tol: float = 1e-12,
                        damp: float = 1.0, verbose: bool = False,
                        P_M1=None, P_M2=None,
                        Vb_prev=None, dt=None,
                        Vsint_bounds: tuple = (-0.5, 1.5),
                        Vb_bounds: tuple = (-0.5, 1.2),
                        max_step: float = 0.2) -> dict:
    """Iterate joint Newton with backtracking line search + bounds.

    If Vb_prev/dt given, solves for the implicit-Euler step.
    """
    Vsint = torch.tensor(float(Vsint0), dtype=torch.float64)
    Vb = torch.tensor(float(Vb0), dtype=torch.float64)
    R_prev = float("inf")
    for k in range(max_iters):
        Vs_new, Vb_new, R_norm = joint_newton_step(
            cfg, model_M1, model_M2, bjt, Vd, VG1, VG2, Vsint, Vb,
            P_M1, P_M2, Vb_prev=Vb_prev, dt=dt)
        # Reject NaN steps
        if not (torch.isfinite(Vs_new) and torch.isfinite(Vb_new)):
            if verbose:
                print(f"  iter={k}  NaN step rejected, halving damping")
            damp = damp * 0.5
            if damp < 1e-4:
                break
            continue
        # Cap step magnitude
        dVs = (Vs_new - Vsint).clamp(-max_step, max_step)
        dVb = (Vb_new - Vb).clamp(-max_step, max_step)
        # Backtracking line search
        alpha = damp
        accepted = False
        for ls in range(6):
            Vsint_try = (Vsint + alpha * dVs).clamp(*Vsint_bounds)
            Vb_try = (Vb + alpha * dVb).clamp(*Vb_bounds)
            # Re-eval R at trial point
            from .nsram_cell_2T import _residuals
            R_S_try, R_B_try, _ = _residuals(
                cfg, model_M1, bjt, Vd=Vd.unsqueeze(0) if Vd.dim()==0 else Vd,
                VG1=VG1, VG2=VG2,
                Vsint=Vsint_try.unsqueeze(0), Vb=Vb_try.unsqueeze(0),
                P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
            if Vb_prev is not None and dt is not None:
                from .transient import junction_cap
                Cj0_total = cfg.body_pdiode_Cj0_per_area * cfg.body_pdiode_area
                Cj_t = junction_cap(Vb_try - cfg.vnwell, Cj0=Cj0_total,
                                     Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
                R_B_try = R_B_try.squeeze() - Cj_t * (Vb_try - Vb_prev) / dt
            R_try = max(float(R_S_try.abs().max()), float(R_B_try.abs().max()))
            if R_try < float(R_norm) * 1.001:
                Vsint, Vb = Vsint_try, Vb_try
                accepted = True
                R_norm = torch.tensor(R_try)
                break
            alpha *= 0.5
        if not accepted:
            # Take tiny step anyway to avoid stalling
            Vsint = (Vsint + 1e-3 * dVs).clamp(*Vsint_bounds)
            Vb = (Vb + 1e-3 * dVb).clamp(*Vb_bounds)
        if verbose:
            print(f"  iter={k}  Vsint={float(Vsint):+.5f}  Vb={float(Vb):+.5f}  "
                  f"|R|={float(R_norm):.3e}  alpha={alpha:.3f}")
        if float(R_norm) < tol:
            return {"Vsint": Vsint, "Vb": Vb, "niter": k+1,
                      "converged": True, "R_norm": R_norm}
        R_prev = float(R_norm)
    return {"Vsint": Vsint, "Vb": Vb, "niter": max_iters,
              "converged": False, "R_norm": R_norm}


def transient_2t(cfg, model_M1, model_M2, bjt,
                  Vd_t: torch.Tensor, t: torch.Tensor,
                  VG1: torch.Tensor, VG2: torch.Tensor, *,
                  Vb0: float = 0.0, Vsint0: float = 0.1,
                  spike_threshold: float = 0.65,
                  reset_Vb: float = 0.30,
                  newton_iters: int = 25,
                  newton_tol: float = 1e-10,
                  damp: float = 0.7,
                  verbose: bool = False,
                  P_M1=None, P_M2=None) -> dict:
    """Implicit-Euler 2T transient with autograd-exact Jacobian.

    Solves the joint (Vsint, Vb) system at each timestep with R_B
    augmented by the body capacitance term. Spike detection AFTER step.

    Returns: dict with Vb, Vsint, Id, spike_times (s), t.
    """
    n = Vd_t.numel()
    Vb_traj = torch.zeros(n, dtype=torch.float64)
    Vsint_traj = torch.zeros(n, dtype=torch.float64)
    Id_traj = torch.zeros(n, dtype=torch.float64)
    spike_times = []

    # Step 0 — quasi-static DC at Vd[0]
    res0 = joint_newton_solve(
        cfg, model_M1, model_M2, bjt, Vd_t[0], VG1, VG2,
        Vsint0=Vsint0, Vb0=Vb0,
        max_iters=newton_iters, tol=newton_tol, damp=damp,
        P_M1=P_M1, P_M2=P_M2)
    Vsint, Vb = res0["Vsint"], res0["Vb"]
    Vb_traj[0] = Vb; Vsint_traj[0] = Vsint
    _, _, comps = _residuals(cfg, model_M1, bjt, Vd=Vd_t[0:1], VG1=VG1, VG2=VG2,
                                Vsint=Vsint.unsqueeze(0), Vb=Vb.unsqueeze(0),
                                P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
    Id_traj[0] = (comps["Ic_Q1"] + comps["Ids_M1"]).squeeze()

    for i in range(1, n):
        dt_i = float(t[i] - t[i-1])
        Vb_prev = Vb.detach().clone()
        # Warm-start from previous solution
        res = joint_newton_solve(
            cfg, model_M1, model_M2, bjt, Vd_t[i], VG1, VG2,
            Vsint0=float(Vsint), Vb0=float(Vb),
            max_iters=newton_iters, tol=newton_tol, damp=damp,
            P_M1=P_M1, P_M2=P_M2,
            Vb_prev=Vb_prev, dt=dt_i)
        Vsint, Vb = res["Vsint"], res["Vb"]
        # Compute Id at converged state
        _, _, comps = _residuals(cfg, model_M1, bjt, Vd=Vd_t[i:i+1], VG1=VG1, VG2=VG2,
                                    Vsint=Vsint.unsqueeze(0), Vb=Vb.unsqueeze(0),
                                    P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
        Id_traj[i] = (comps["Ic_Q1"] + comps["Ids_M1"]).squeeze()
        Vb_traj[i] = Vb; Vsint_traj[i] = Vsint
        if float(Vb) >= spike_threshold:
            spike_times.append(float(t[i]))
            Vb = torch.tensor(reset_Vb, dtype=torch.float64)
        if verbose and i % max(1, n // 10) == 0:
            print(f"  [transient] t={float(t[i]):.4g}  Vd={float(Vd_t[i]):.3f}  "
                  f"Vb={float(Vb):+.4f}  Vsint={float(Vsint):+.4f}  "
                  f"Id={float(Id_traj[i]):.3e}  conv={res['converged']}")
    return {"Vb": Vb_traj, "Vsint": Vsint_traj, "Id": Id_traj,
              "spike_times": spike_times, "t": t}
