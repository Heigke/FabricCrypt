"""z281b_mep6_demo_fit.py — MEP-6 differentiable fit demo (CPU, 50 Adam steps).

GATE PRECONDITION: scripts/z281_mep6_torch_newton.py --skip-fit must
report `gate_correctness: True` first. This script imports the batched
wrapper from z281 and runs an Adam fit on 5 BSIM4 params against a
subset of Sebas's 33-bias measurement set.

Honest notes (kept from MEP-6 v1):
  - BSIM4Model parameters are stored as Python floats, so they cannot
    be torch.nn.Parameter. We use a hybrid FD-grad / Adam loop: each
    Adam step computes the 5-component gradient via one-sided forward
    differences on the loss, then applies a standard Adam update with
    bound projection.
  - Fully-autograd-through-params would require porting BSIM4Model
    storage to torch tensors and is OUT-OF-SCOPE for MEP-6's
    correctness-first deliverable.

RESOURCE POLICY: CPU only. Wait for D2 sweep before any GPU.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model

# Reuse helpers from the upstream correctness script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from z281_mep6_torch_newton import (  # type: ignore
    CARD, DATA, OUT,
    _default_cfg, build_model_from_params,
    load_target_curves, eval_loss, fd_grad,
    PARAM_NAMES, PARAM_INIT, PARAM_BOUNDS,
)


def fit_demo(steps: int = 50, lr: float = 0.01, n_curves: int = 6, n_ds: int = 8,
             scipy_baseline_log_rmse: float = 0.7694):
    print("=" * 64)
    print(f"MEP-6 differentiable fit demo: {steps} Adam steps, lr={lr}")
    print(f"scipy-DE production baseline median log-RMSE = "
          f"{scipy_baseline_log_rmse:.4f}")
    print("=" * 64)
    curves = load_target_curves(max_curves=n_curves, n_ds=n_ds)
    print(f"Loaded {len(curves)} curves, {n_ds} bias points each")
    cfg = _default_cfg()
    bjt = GummelPoonNPN.from_sebas_card()

    params = torch.tensor([PARAM_INIT[k] for k in PARAM_NAMES],
                          dtype=torch.float64)

    t0 = time.time()
    init_loss = eval_loss(params, curves, cfg, bjt)
    print(f"  step  0  loss={init_loss:.4f}  (init)")
    losses = [init_loss]

    m = np.zeros(5); v = np.zeros(5)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_loss = init_loss
    best_params = {k: float(params[i]) for i, k in enumerate(PARAM_NAMES)}

    for it in range(1, steps + 1):
        g, loss_now = fd_grad(params, curves, cfg, bjt)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        mh = m / (1 - beta1 ** it)
        vh = v / (1 - beta2 ** it)
        for i, k in enumerate(PARAM_NAMES):
            lo, hi = PARAM_BOUNDS[k]
            step = -lr * (hi - lo) * mh[i] / (np.sqrt(vh[i]) + eps)
            new = float(params[i]) + step
            params[i] = max(lo, min(hi, new))
        losses.append(loss_now)
        if loss_now < best_loss:
            best_loss = loss_now
            best_params = {k: float(params[i]) for i, k in enumerate(PARAM_NAMES)}
        if it % 5 == 0 or it == 1:
            pstr = ', '.join(f'{float(p):.3f}' for p in params)
            print(f"  step {it:2d}  loss={loss_now:.4f}  params=[{pstr}]",
                  flush=True)

    final_loss = eval_loss(params, curves, cfg, bjt)
    t_fit = time.time() - t0

    print(f"\n  init loss   = {init_loss:.4f}")
    print(f"  final loss  = {final_loss:.4f}")
    print(f"  best loss   = {best_loss:.4f}  (Δ from init = {init_loss-best_loss:+.4f})")
    print(f"  scipy DE    = {scipy_baseline_log_rmse:.4f}  (production)")
    gate_ambitious = best_loss <= scipy_baseline_log_rmse
    print(f"  AMBITIOUS gate (best ≤ scipy DE on subset): "
          f"{'PASS' if gate_ambitious else 'FAIL (expected on CPU subset)'}")
    print(f"  wall time   = {t_fit:.1f} s")
    return {
        "init_loss": init_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "scipy_de_baseline": scipy_baseline_log_rmse,
        "loss_trajectory": losses,
        "best_params": best_params,
        "wall_time_s": t_fit,
        "n_steps": steps,
        "n_curves": len(curves),
        "n_bias_per_curve": n_ds,
        "gate_ambitious_vs_scipy": bool(gate_ambitious),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--curves", type=int, default=6)
    ap.add_argument("--n-ds", type=int, default=8)
    args = ap.parse_args()
    result = fit_demo(steps=args.steps, lr=args.lr,
                      n_curves=args.curves, n_ds=args.n_ds)
    out_path = OUT / "z281b_demo_fit.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
