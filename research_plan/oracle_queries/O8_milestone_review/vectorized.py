"""Vectorized batched 2T forward sweep.

B.1 (2026-05-02): The serial forward_2t loops one bias at a time. The
underlying _residuals already broadcasts naturally over a batch dim,
so an N-bias sweep can share the Newton machinery across all biases
simultaneously — a single torch.linalg.solve replaces N independent
2x2 solves. For z91g (33 biases × 40 Vd points) this should give
a roughly 33× wall-time speedup, modulo Python-level overhead.

Public API:
    forward_2t_batched(cfg, model_M1, model_M2, bjt, Vd_seq, VG1_arr, VG2_arr)
        VG1_arr, VG2_arr: shape (N,) — N independent biases.
        Vd_seq: shape (T,) — common Vd sweep grid.
        Returns dict with shape (N, T) tensors: Id, Vsint, Vb, niter, conv.
"""
from __future__ import annotations
import torch
from typing import Optional

from .nsram_cell_2T import _residuals, NSRAMCell2TConfig
from .model_card import BSIM4Model
from .bjt import GummelPoonNPN


def _solve_2x2_batched(R_S: torch.Tensor, R_B: torch.Tensor,
                         J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched 2x2 Newton step. J shape (N, 2, 2), residuals shape (N,).
    Returns (dVsint, dVb), each shape (N,).
    """
    rhs = -torch.stack([R_S, R_B], dim=-1).unsqueeze(-1)   # (N,2,1)
    sol = torch.linalg.solve(J, rhs)
    return sol[..., 0, 0], sol[..., 1, 0]


def _jacobian_fd_batched(cfg, model_M1, model_M2, bjt, Vd, VG1, VG2,
                          Vsint, Vb, eps: float = 1e-5) -> torch.Tensor:
    """Build batched 2x2 Jacobian via central-ish finite differences.
    Shape (N, 2, 2). Calls _residuals 4 extra times per step.
    """
    R_S0, R_B0, _ = _residuals(cfg, model_M1, bjt, Vd=Vd, VG1=VG1, VG2=VG2,
                                 Vsint=Vsint, Vb=Vb, model_M2=model_M2)
    R_S_dVs, R_B_dVs, _ = _residuals(cfg, model_M1, bjt, Vd=Vd, VG1=VG1, VG2=VG2,
                                       Vsint=Vsint+eps, Vb=Vb, model_M2=model_M2)
    R_S_dVb, R_B_dVb, _ = _residuals(cfg, model_M1, bjt, Vd=Vd, VG1=VG1, VG2=VG2,
                                       Vsint=Vsint, Vb=Vb+eps, model_M2=model_M2)
    dRS_dVs = (R_S_dVs - R_S0) / eps
    dRS_dVb = (R_S_dVb - R_S0) / eps
    dRB_dVs = (R_B_dVs - R_B0) / eps
    dRB_dVb = (R_B_dVb - R_B0) / eps
    # J[..., i, j] = dR_i / dx_j
    J = torch.stack([
        torch.stack([dRS_dVs, dRS_dVb], dim=-1),
        torch.stack([dRB_dVs, dRB_dVb], dim=-1),
    ], dim=-2)
    return J, R_S0, R_B0


def forward_2t_batched(cfg: NSRAMCell2TConfig,
                        model_M1: BSIM4Model, model_M2: BSIM4Model,
                        bjt: GummelPoonNPN,
                        Vd_seq: torch.Tensor,
                        VG1_arr: torch.Tensor, VG2_arr: torch.Tensor,
                        *, max_iters: int = 30, tol: float = 1e-12,
                        Vsint0: float = 0.1, Vb0: float = 0.3,
                        damping: float = 1.0,
                        verbose: bool = False) -> dict:
    """Run N independent 2T forward Vd sweeps in parallel.

    Returns dict with shape (N, T) tensors: Id, Vsint, Vb, niter, conv.
    """
    N = VG1_arr.numel()
    T = Vd_seq.numel()
    Id_out = torch.zeros(N, T, dtype=torch.float64)
    Vsint_out = torch.zeros(N, T, dtype=torch.float64)
    Vb_out = torch.zeros(N, T, dtype=torch.float64)
    conv_out = torch.zeros(N, T, dtype=torch.bool)
    niter_out = torch.zeros(N, T, dtype=torch.int32)

    # State: Vsint, Vb of shape (N,), warm-started across Vd
    Vsint = torch.full((N,), Vsint0, dtype=torch.float64)
    Vb = torch.full((N,), Vb0, dtype=torch.float64)

    for ti in range(T):
        Vd_t = torch.full((N,), float(Vd_seq[ti]), dtype=torch.float64)
        # Newton loop, batched across N biases
        for k in range(max_iters):
            J, R_S, R_B = _jacobian_fd_batched(
                cfg, model_M1, model_M2, bjt, Vd_t, VG1_arr, VG2_arr, Vsint, Vb)
            R_max = torch.maximum(R_S.abs(), R_B.abs())
            done = R_max < tol
            if done.all():
                niter_out[:, ti] = k
                conv_out[:, ti] = True
                break
            try:
                dVs, dVb = _solve_2x2_batched(R_S, R_B, J)
            except Exception:
                break
            # Damp big steps
            dVs = dVs.clamp(-0.5, 0.5)
            dVb = dVb.clamp(-0.5, 0.5)
            Vsint = Vsint + damping * dVs
            Vb = Vb + damping * dVb
            # Bound to physical range
            Vb = Vb.clamp(-0.5, 1.2)
        else:
            niter_out[:, ti] = max_iters
            conv_out[:, ti] = R_max < tol * 1e3
        # Compute final Id at converged state
        _, _, comps = _residuals(cfg, model_M1, bjt, Vd=Vd_t,
                                   VG1=VG1_arr, VG2=VG2_arr,
                                   Vsint=Vsint, Vb=Vb, model_M2=model_M2)
        Id_out[:, ti] = comps["Ic_Q1"] + comps["Ids_M1"]
        Vsint_out[:, ti] = Vsint
        Vb_out[:, ti] = Vb
        if verbose and ti % max(1, T // 5) == 0:
            print(f"  [batched] Vd={float(Vd_seq[ti]):.3f}  "
                  f"max(R)={float(R_max.max()):.3e}  "
                  f"conv={int(done.sum())}/{N}")

    return {"Id": Id_out, "Vsint": Vsint_out, "Vb": Vb_out,
            "niter": niter_out, "converged": conv_out}
