"""Body-pdiode capacitance helper + minimal transient stub.

A.4 (2026-05-02): Sebas's pdiode card has voltage-dependent Cj(V) that
captures the floating-body capacitance dynamics. This module exposes
that helper and a tiny transient stub that the rest of the codebase
can import without polluting the DC compute_dc / _residuals path.

Cj(V) per BSIM4 / SPICE diode v4 convention:

    Cj(V) = Cj0 / (1 - V/Vj)^M           for V < Vj·FC
    linear continuation                  for V > Vj·FC  (avoid singularity)

where V = Vanode - Vcathode (positive forward). For our pdiode at
body↔vnwell, V = Vb - vnwell, normally negative (reverse), Cj < Cj0.
"""
from __future__ import annotations
import torch


def junction_cap(V: torch.Tensor, *, Cj0: float, Vj: float, M: float,
                  fc: float = 0.5) -> torch.Tensor:
    """Voltage-dependent junction capacitance, smooth across V=Vj·fc.

    V positive = forward. Cj rises with forward bias up to V=Vj·fc, then
    is linearly extrapolated to keep grad finite (SPICE convention).

    Returns Cj in farads (Cj0 should already be Cj0_per_area × area).
    """
    Vbreak = fc * Vj
    # Reverse / mild forward branch
    arg = (1.0 - V / Vj).clamp_min(1e-6)   # safety floor (very forward → tiny)
    Cj_main = Cj0 * arg.pow(-M)
    # Linear continuation past Vbreak: derivative at V=Vbreak
    arg_b = 1.0 - fc
    slope = Cj0 * M / Vj * arg_b ** (-(M + 1.0))
    Cj_break = Cj0 * arg_b ** (-M)
    Cj_lin = Cj_break + slope * (V - Vbreak)
    return torch.where(V < Vbreak, Cj_main, Cj_lin)


def integrate_body_cap_charge(Vb_traj: torch.Tensor, t_traj: torch.Tensor,
                                vnwell: float, *, Cj0_per_area: float,
                                area: float, Vj: float, M: float
                                ) -> torch.Tensor:
    """Given a Vb(t) trajectory, return I_cap(t) = Cj(Vb-vnwell) · dVb/dt.

    For transient validation against Sebas's ramped Vd measurements: we
    take a quasi-static body-voltage trajectory (from successive DC
    Newton solves at each Vd_i along the ramp) and add the displacement
    current through the body-pdiode capacitance. This is the leading-
    order correction; for full transient solver, the Cj enters the
    body-KCL Jacobian directly.
    """
    Cj0_total = Cj0_per_area * area
    V = Vb_traj - vnwell
    Cj = junction_cap(V, Cj0=Cj0_total, Vj=Vj, M=M)
    dVb_dt = torch.zeros_like(Vb_traj)
    if Vb_traj.numel() > 1:
        dVb_dt[1:-1] = (Vb_traj[2:] - Vb_traj[:-2]) / (t_traj[2:] - t_traj[:-2])
        dVb_dt[0] = (Vb_traj[1] - Vb_traj[0]) / (t_traj[1] - t_traj[0])
        dVb_dt[-1] = (Vb_traj[-1] - Vb_traj[-2]) / (t_traj[-1] - t_traj[-2])
    return Cj * dVb_dt


def integrate_2t_transient_implicit(cfg, model_M1, model_M2, bjt,
                                       Vd_t: torch.Tensor, t: torch.Tensor,
                                       VG1: torch.Tensor, VG2: torch.Tensor, *,
                                       Vb0: float = 0.0, Vsint0: float = 0.0,
                                       spike_threshold: float = 0.65,
                                       reset_Vb: float = 0.30,
                                       newton_iters_inner: int = 8,
                                       newton_iters_outer: int = 12,
                                       newton_tol: float = 1e-12,
                                       verbose: bool = False):
    """Implicit-Euler time integration of the 2T cell body charge.

    Stable on the stiff body-charge ODE (12-decade dynamic range)
    where forward-Euler diverges. Uses a split scheme:
      Outer loop: Newton on Vb_new with backward-Euler on the cap term
        F(Vb_new) := R_B(Vsint*(Vb_new), Vb_new, Vd_new)
                     − Cj(Vb_new − vnwell) · (Vb_new − Vb_old) / dt
                     = 0
      Inner loop: at each candidate Vb_new, quasi-static Newton on
        Vsint*(Vb_new) such that R_Sint = 0 (1D in Vsint).

    Spike detection: post-step, if Vb >= spike_threshold, log event,
    snap to reset_Vb (zero-time discharge).
    """
    from .nsram_cell_2T import _residuals
    n = Vd_t.numel()
    Vb_traj = torch.zeros(n, dtype=torch.float64)
    Vsint_traj = torch.zeros(n, dtype=torch.float64)
    Id_traj = torch.zeros(n, dtype=torch.float64)
    Vb = torch.tensor(Vb0, dtype=torch.float64)
    Vsint = torch.tensor(Vsint0, dtype=torch.float64)
    spike_times = []

    Cj0_total = cfg.body_pdiode_Cj0_per_area * cfg.body_pdiode_area
    eps_J = 1e-4   # finite-diff perturbation

    def _solve_Vsint(Vb_curr, Vd_i):
        """Inner: Vsint such that R_Sint = 0 with this Vb_curr."""
        Vs = Vsint.clone()
        for _ in range(newton_iters_inner):
            R_S, _, comps = _residuals(
                cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                Vsint=Vs.unsqueeze(0), Vb=Vb_curr.unsqueeze(0),
                P_M1=None, P_M2=None, model_M2=model_M2)
            if R_S.abs().max() < newton_tol:
                break
            R_S_eps, _, _ = _residuals(
                cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                Vsint=(Vs+eps_J).unsqueeze(0),
                Vb=Vb_curr.unsqueeze(0),
                P_M1=None, P_M2=None, model_M2=model_M2)
            J = (R_S_eps - R_S) / eps_J
            dVs = -R_S / J.clamp(min=1e-30, max=None)
            # Damp big steps so we don't jump into nonconvergent region
            Vs = Vs + dVs.squeeze().clamp(-0.3, 0.3)
        return Vs, comps

    for i in range(n):
        Vd_i = Vd_t[i:i+1]
        if i == 0:
            # Initial step: quasi-static (no dt term)
            Vsint, comps = _solve_Vsint(Vb, Vd_i)
        else:
            dt = float(t[i] - t[i-1])
            Vb_old = Vb.clone()
            # Outer Newton on Vb_new
            Vb_new = Vb.clone()
            for outer in range(newton_iters_outer):
                Vsint_at, comps = _solve_Vsint(Vb_new, Vd_i)
                _, R_B_now, _ = _residuals(
                    cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                    Vsint=Vsint_at.unsqueeze(0), Vb=Vb_new.unsqueeze(0),
                    P_M1=None, P_M2=None, model_M2=model_M2)
                Cj_now = junction_cap(
                    Vb_new - cfg.vnwell, Cj0=Cj0_total,
                    Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
                F = R_B_now.squeeze() - Cj_now * (Vb_new - Vb_old) / dt
                if F.abs() < newton_tol:
                    break
                # FD Jacobian dF/dVb_new
                Vb_eps = Vb_new + eps_J
                Vsint_eps, _ = _solve_Vsint(Vb_eps, Vd_i)
                _, R_B_eps, _ = _residuals(
                    cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                    Vsint=Vsint_eps.unsqueeze(0), Vb=Vb_eps.unsqueeze(0),
                    P_M1=None, P_M2=None, model_M2=model_M2)
                Cj_eps = junction_cap(
                    Vb_eps - cfg.vnwell, Cj0=Cj0_total,
                    Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
                F_eps = R_B_eps.squeeze() - Cj_eps * (Vb_eps - Vb_old) / dt
                dF = (F_eps - F) / eps_J
                # Damped Newton step
                step = -F / dF.clamp(min=1e-30, max=None) if dF.abs() > 1e-30 else torch.tensor(0.0)
                step = step.clamp(-0.2, 0.2)
                Vb_new = Vb_new + step
                # Bound to physical range
                Vb_new = Vb_new.clamp(-1.0, 2.0)
            Vb = Vb_new
            Vsint = Vsint_at
        # Compute Id
        _, _, comps2 = _residuals(
            cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
            Vsint=Vsint.unsqueeze(0), Vb=Vb.unsqueeze(0),
            P_M1=None, P_M2=None, model_M2=model_M2)
        Id_i = comps2["Ic_Q1"] + comps2["Ids_M1"]
        Vb_traj[i] = Vb
        Vsint_traj[i] = Vsint
        Id_traj[i] = Id_i.squeeze()
        # Spike detection
        if float(Vb) >= spike_threshold:
            spike_times.append(float(t[i]))
            Vb = torch.tensor(reset_Vb, dtype=torch.float64)
        if verbose and i % max(1, n // 10) == 0:
            print(f"  [transient] t={float(t[i]):.4g}  Vd={float(Vd_i):.3f}  "
                  f"Vb={float(Vb):+.4f}  Vsint={float(Vsint):+.4f}  "
                  f"Id={float(Id_i):.3e}")
    return {
        "Vb": Vb_traj, "Vsint": Vsint_traj, "Id": Id_traj,
        "spike_times": spike_times, "t": t,
    }


def integrate_2t_transient(cfg, model_M1, model_M2, bjt, Vd_t: torch.Tensor,
                            t: torch.Tensor, VG1: torch.Tensor,
                            VG2: torch.Tensor, *,
                            Vb0: float = 0.0, Vsint0: float = 0.0,
                            spike_threshold: float = 0.65,
                            reset_Vb: float = 0.30):
    """Forward Euler time integration of the 2T cell body charge.

    State: Vb(t), Vsint(t). Vsint solved quasi-statically at each step
    (1D Newton in Vsint with Vb fixed); Vb integrated as
        dVb/dt = R_B(Vsint, Vb, Vd) / C_total(Vb)
    where R_B is the body KCL residual (currents INTO body) and
    C_total = Cj_pdiode(Vb-vnwell) + Cj_M1_bs/bd + Cj_M2_bs/bd.

    Spike detection: when Vb crosses spike_threshold, log a spike event,
    snap Vb to reset_Vb (one-step). This is the LIF firing primitive.

    Returns dict with Vb_traj, Vsint_traj, Id_traj, spike_times.

    NOTE: this is a minimal forward-Euler integrator suitable for
    plotting transient ramps and demonstrating LIF-style dynamics.
    Production work needs implicit BDF and adaptive dt.

    WARNING (2026-05-02): forward-Euler is unconditionally unstable on
    this stiff body-charge ODE. The 2T cell loop has very small
    capacitance (Cj ~ 10 fF) and currents that span 1 fA to 1 mA, so
    the natural time constant ranges over ~12 decades. Any explicit
    method explodes near the snapback fold. This routine is provided
    as a skeleton for the eventual implicit integrator (Phase B work);
    quantitative use requires Newton-per-step in the joint (Vsint,Vb).
    """
    from .nsram_cell_2T import _residuals
    n = Vd_t.numel()
    Vb_traj = torch.zeros(n, dtype=torch.float64)
    Vsint_traj = torch.zeros(n, dtype=torch.float64)
    Id_traj = torch.zeros(n, dtype=torch.float64)
    Vb = torch.tensor(Vb0, dtype=torch.float64)
    Vsint = torch.tensor(Vsint0, dtype=torch.float64)
    spike_times = []

    Cj0_total = cfg.body_pdiode_Cj0_per_area * cfg.body_pdiode_area

    for i in range(n):
        Vd_i = Vd_t[i:i+1]
        # Quasi-static Vsint solve: ~5 Newton iters in Vsint with Vb fixed
        for _ in range(8):
            R_S, R_B, comps = _residuals(cfg, model_M1, bjt,
                                          Vd=Vd_i, VG1=VG1, VG2=VG2,
                                          Vsint=Vsint.unsqueeze(0),
                                          Vb=Vb.unsqueeze(0),
                                          P_M1=None, P_M2=None,
                                          model_M2=model_M2)
            # Finite-diff Jacobian dR_S/dVsint
            eps = 1e-4
            R_S_eps, _, _ = _residuals(cfg, model_M1, bjt,
                                        Vd=Vd_i, VG1=VG1, VG2=VG2,
                                        Vsint=(Vsint+eps).unsqueeze(0),
                                        Vb=Vb.unsqueeze(0),
                                        P_M1=None, P_M2=None,
                                        model_M2=model_M2)
            J = (R_S_eps - R_S) / eps
            dVsint = -R_S / J.clamp(min=1e-30)
            Vsint = Vsint + dVsint.squeeze().clamp(-0.5, 0.5)
            if R_S.abs().max() < 1e-12:
                break
        # Now compute body-charge derivative
        _, R_B_now, _ = _residuals(cfg, model_M1, bjt,
                                    Vd=Vd_i, VG1=VG1, VG2=VG2,
                                    Vsint=Vsint.unsqueeze(0),
                                    Vb=Vb.unsqueeze(0),
                                    P_M1=None, P_M2=None,
                                    model_M2=model_M2)
        Cj_now = junction_cap(Vb - cfg.vnwell, Cj0=Cj0_total,
                                Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
        # Add small floor so Cj never zero (prevents dt explosion)
        C_total = Cj_now + 1e-18
        if i + 1 < n:
            dt = float(t[i+1] - t[i])
            dVb = float(R_B_now.squeeze()) * dt / float(C_total)
            Vb = Vb + dVb
        # Compute Id at this point
        Id_i = comps["Ic_Q1"] + comps["Ids_M1"]   # drain current = NPN collector + M1 channel
        Vb_traj[i] = Vb
        Vsint_traj[i] = Vsint
        Id_traj[i] = Id_i.squeeze()
        # Spike detection
        if float(Vb) >= spike_threshold:
            spike_times.append(float(t[i]))
            Vb = torch.tensor(reset_Vb, dtype=torch.float64)
    return {
        "Vb": Vb_traj, "Vsint": Vsint_traj, "Id": Id_traj,
        "spike_times": spike_times, "t": t,
    }
