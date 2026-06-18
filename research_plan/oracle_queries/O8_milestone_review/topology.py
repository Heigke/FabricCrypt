"""Minimal multi-cell NS-RAM topology layer.

B.4 (2026-05-02): wraps `forward_2t_batched` with per-cell static
config, per-cell bias trajectories, and a linear readout for the
common reservoir / Hopfield / classifier paradigm.

Each cell is an independent 2T NS-RAM with shared model cards (M1, M2)
but independent (VG1, VG2) per cell — i.e., the per-cell bias is the
"weight" or "state" that distinguishes cells. Cells run in parallel;
no inter-cell electrical coupling at this layer.

A linear readout maps the per-cell drain currents to network-level
outputs: output[k] = sum_i W[k, i] · log10(|Id_i| + eps).

This is the substrate for Phase B.5 benchmarks (Hopfield retrieval,
NARMA-10, memory capacity, temporal-XOR, multi-class waveform).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch

from .nsram_cell_2T import NSRAMCell2TConfig
from .model_card import BSIM4Model
from .bjt import GummelPoonNPN
from .vectorized import forward_2t_batched


@dataclass
class NSRAMNetwork:
    """N-cell 2T NS-RAM array with linear readout.

    cfg: shared cell config
    model_M1, model_M2: shared model cards (calibrated once, applied to all)
    bjt: shared parasitic NPN
    N: number of cells

    Per-cell state (set externally before each forward):
      VG1: shape (N,) — gate-1 control voltage
      VG2: shape (N,) — gate-2 control voltage

    Readout:
      W: shape (n_out, N) — linear projection of log10|Id|
    """
    cfg: NSRAMCell2TConfig
    model_M1: BSIM4Model
    model_M2: BSIM4Model
    bjt: GummelPoonNPN
    N: int
    W: Optional[torch.Tensor] = None
    n_out: int = 0

    def __post_init__(self):
        if self.W is None and self.n_out > 0:
            # Default initial readout: small random Gaussian, fp64
            self.W = (0.01 * torch.randn(self.n_out, self.N, dtype=torch.float64))

    def forward(self, Vd_seq: torch.Tensor, VG1: torch.Tensor, VG2: torch.Tensor,
                **batch_kwargs) -> dict:
        """Run all N cells in parallel for a Vd sweep.
        Returns dict with shape (N, T) tensors plus optional readout (n_out, T).
        """
        out = forward_2t_batched(
            self.cfg, self.model_M1, self.model_M2, self.bjt,
            Vd_seq, VG1, VG2, **batch_kwargs)
        Id = out["Id"]   # (N, T)
        # Log-feature readout (decade-scaled current is the natural NSRAM observable)
        eps = 1e-15
        log_Id = torch.log10(Id.abs() + eps)
        if self.W is not None:
            out["readout"] = self.W @ log_Id   # (n_out, T)
        out["log_Id"] = log_Id
        return out

    def fit_readout(self, log_Id: torch.Tensor, target: torch.Tensor,
                     ridge: float = 1e-3) -> torch.Tensor:
        """Closed-form ridge regression for the linear readout.
        log_Id: (N, T_train) feature matrix.
        target: (n_out, T_train) target outputs.
        Returns W: (n_out, N) — also updates self.W in place.
        """
        X = log_Id   # (N, T)
        XX_T = X @ X.T   # (N, N)
        I = torch.eye(X.shape[0], dtype=X.dtype) * ridge
        XY_T = X @ target.T   # (N, n_out)
        W_T = torch.linalg.solve(XX_T + I, XY_T)   # (N, n_out)
        self.W = W_T.T.contiguous()
        self.n_out = self.W.shape[0]
        return self.W
