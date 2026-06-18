"""diff_op.py — Differentiable wrapper for divergent_matmul (C2).

Forward:
    Calls a HIP/Triton kernel whose accumulation order is implementation-
    defined, so output bit-pattern depends on chip behaviour.

Backward:
    Straight-through estimator. Gradient is computed *as if* the operator
    were a standard deterministic matmul:  dL/dW = dL/dy . x^T,
    dL/dx = W^T . dL/dy. This lets weights co-adapt to the chip's
    *forward* statistics while keeping training stable.

For the C2 smoke test we implement the forward in PyTorch using
torch.matmul plus a small chip-noise injection that approximates
the kernel's expected divergence pattern; the actual HIP kernel
(divergent_matmul.hip) is used for the C1 bit-divergence measurement
and will be wired into forward for C3 once it lands.

Usage:
    from diff_op import DivergentMatMul
    op = DivergentMatMul(use_hip=False)   # surrogate for now
    y = op(W, x)
    loss = y.sum(); loss.backward()
"""
from __future__ import annotations
import os
import subprocess
import torch


class _DivergentMatMulSurrogate(torch.autograd.Function):
    """Forward: deterministic matmul + small chip-specific perturbation.
       Backward: standard matmul gradients (straight-through)."""

    @staticmethod
    def forward(ctx, W: torch.Tensor, x: torch.Tensor,
                chip_seed: int = 0, eps: float = 1e-6) -> torch.Tensor:
        # Deterministic baseline
        y = W @ x
        # Chip-specific perturbation: bounded, seeded by chip identity.
        # This SURROGATES what divergent_matmul.hip would produce; the
        # real kernel will replace this line.
        if eps > 0.0:
            g = torch.Generator(device=W.device).manual_seed(int(chip_seed))
            noise = torch.randn(y.shape, device=y.device,
                                generator=g, dtype=y.dtype) * eps * y.abs().mean()
            y = y + noise
        ctx.save_for_backward(W, x)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        W, x = ctx.saved_tensors
        # Straight-through: treat operator as if it were a clean matmul
        grad_W = grad_y.unsqueeze(-1) @ x.unsqueeze(-2) if x.dim() == 1 \
                 else grad_y @ x.transpose(-1, -2)
        if x.dim() == 1:
            grad_x = W.transpose(-1, -2) @ grad_y
        else:
            grad_x = W.transpose(-1, -2) @ grad_y
        return grad_W, grad_x, None, None


class DivergentMatMul(torch.nn.Module):
    """Module wrapper.  If use_hip=True and the HIP binary exists, the
    forward will shell out to it (slow, only for measurement / C1)."""

    def __init__(self, use_hip: bool = False, chip_seed: int | None = None,
                 eps: float = 1e-6):
        super().__init__()
        self.use_hip = use_hip
        self.eps = eps
        # Default chip seed = hash of hostname so ikaros vs daedalus differ
        if chip_seed is None:
            try:
                host = os.uname().nodename
            except Exception:
                host = "unknown"
            chip_seed = abs(hash(host)) % (2**31 - 1)
        self.chip_seed = chip_seed

    def forward(self, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.use_hip:
            raise NotImplementedError(
                "HIP path is for C1 measurement only; wire after C1 lands.")
        return _DivergentMatMulSurrogate.apply(W, x, self.chip_seed, self.eps)


def _smoke():
    """Tiny check: forward + backward + gradient flow."""
    torch.manual_seed(0)
    W = torch.randn(8, 16, requires_grad=True)
    x = torch.randn(16, 4, requires_grad=True)
    op = DivergentMatMul(use_hip=False, eps=1e-3)
    y = op(W, x)
    loss = (y ** 2).sum()
    loss.backward()
    assert W.grad is not None and x.grad is not None
    assert torch.isfinite(W.grad).all() and torch.isfinite(x.grad).all()
    print("[diff_op] forward shape:", tuple(y.shape),
          " loss:", float(loss),
          " W.grad norm:", float(W.grad.norm()),
          " x.grad norm:", float(x.grad.norm()),
          " chip_seed:", op.chip_seed)


if __name__ == "__main__":
    _smoke()
