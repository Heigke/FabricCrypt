"""smooth — differentiable replacements for non-smooth primitives in BSIM4.

Each primitive:
  - Has a `sharpness` parameter; converges to the hard version as sharpness→∞.
  - Keeps gradients finite and non-zero in the transition region.
  - Tested in tests/test_smooth.py for: convergence, gradcheck, no-NaN.

Use these EVERYWHERE a faithful BSIM4 port has if/MAX/MIN/abs/sqrt/log/exp.
Cross-reference each substitution site in code with `# SMOOTH: <name>` comment
so block PRs are auditable.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


SHARPNESS_DEFAULT = 50.0
EPS_SQRT = 1e-12
EPS_LOG = 1e-30
EXP_THRESHOLD = 34.0  # matches BSIM4's MAX_EXP/MIN_EXP guard


def smooth_max(a: torch.Tensor, b: torch.Tensor, sharpness: float = SHARPNESS_DEFAULT) -> torch.Tensor:
    """max(a,b) ≈ b + softplus(s·(a-b))/s. Converges as s→∞."""
    return b + F.softplus(sharpness * (a - b)) / sharpness


def smooth_min(a: torch.Tensor, b: torch.Tensor, sharpness: float = SHARPNESS_DEFAULT) -> torch.Tensor:
    """min(a,b) ≈ a - softplus(s·(a-b))/s."""
    return a - F.softplus(sharpness * (a - b)) / sharpness


def soft_clamp(x: torch.Tensor, lo: float, hi: float, sharpness: float = SHARPNESS_DEFAULT) -> torch.Tensor:
    """Smooth clamp to [lo, hi] via smooth_max then smooth_min."""
    lo_t = torch.as_tensor(lo, dtype=x.dtype, device=x.device)
    hi_t = torch.as_tensor(hi, dtype=x.dtype, device=x.device)
    return smooth_min(smooth_max(x, lo_t, sharpness), hi_t, sharpness)


def safe_sqrt(x: torch.Tensor, eps: float = EPS_SQRT) -> torch.Tensor:
    """sqrt(max(x, eps)) — finite gradient at x=0."""
    return torch.sqrt(x.clamp_min(eps))


def safe_log(x: torch.Tensor, eps: float = EPS_LOG) -> torch.Tensor:
    """log(max(x, eps))."""
    return torch.log(x.clamp_min(eps))


def safe_exp(x: torch.Tensor, max_arg: float = EXP_THRESHOLD) -> torch.Tensor:
    """exp clipped at ±max_arg, mirroring BSIM4 DEXP."""
    return torch.exp(x.clamp(-max_arg, max_arg))


def smooth_step(x: torch.Tensor, lo: float, hi: float, sharpness: float = SHARPNESS_DEFAULT) -> torch.Tensor:
    """Smooth Heaviside: 0 below lo, 1 above hi."""
    mid = 0.5 * (lo + hi)
    width = max(hi - lo, 1e-6)
    return torch.sigmoid(2 * sharpness * (x - mid) / width)


def smooth_abs(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """|x| ≈ sqrt(x² + ε)."""
    return torch.sqrt(x * x + eps)
