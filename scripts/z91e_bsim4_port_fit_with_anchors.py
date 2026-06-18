"""z91e — z91d + soft regularizer toward Sebas's extracted parameter values.

Background
----------
The Apr-30 meeting slide 24 ("Image 2026-04-30 at 13.24") shows Sebas already
extracted four BSIM4 parameters as functions of (VG1, VG2) from his measured
curves:

    VTH0(M1)   ~ 0.42–0.55 V across all VG1 (descends with VG1)
    BETA0(M1)  ~ 11–21       (rises with VG1, weak VG2 dep)
    ETA0(M1)   ~ 1.0–2.0     (rises with VG1)
    NFACTOR(M2) ~ 3–12       (descends with VG1)

z91d's Stage-1 optimum drove vth0 to 0.315 V — *below* Sebas's extracted
range. This is a sign the optimizer is still compensating for something
(likely off-state model imperfections we can't fix without finer body-effect
treatment), and it confirms that pure-data fitting is under-constrained.

Strategy
--------
Add a **soft anchor regularizer** that pulls the fitted constant params toward
the mean of Sebas's extracted curves. Constant params are clearly insufficient
(slide 24 shows VG1, VG2 dependence) but anchoring the *mean* prevents the
optimizer from fleeing to non-physical regions to compensate for missing
physics. Once z91e converges, follow up with a poly(VG1, VG2) variant (z91f)
to capture the structure Sebas resolved.

Anchor mode
-----------
- vth0  → 0.50 V    (mean of Sebas's curve)         lambda=0.5
- u0    → init      (no Sebas data)                 lambda=0
- beta0 → 15        (mean of his BETA0(M1) plot)    lambda=0.05
                       — note Sebas's BETA0 ≠ BSIM4's beta0 (impact-ion).
                         Sebas uses BETA0 as a body-current scaler in his
                         own form. We weight this small.
- agidl/bgidl/cgidl/egidl unchanged from z91d (Sebas didn't extract these)
- alpha0/alpha1/Bf unchanged

Caveat: BETA0 is overloaded. In Sebas's plot it's likely his own coefficient
in his SPICE deck (Pazos lab convention), not BSIM4 §6.1's beta0. Treat its
anchor weight as a sanity check, not a hard constraint.
"""
from __future__ import annotations
import json, math, re, time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)
print(f"[z91e] Using device: {DEVICE}", flush=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91e_bsim4_port_fit_with_anchors"
OUT.mkdir(parents=True, exist_ok=True)


# Reuse loader/PARAM_SPEC/helpers from z91d. To avoid duplicating ~500 LOC we
# import the module under a stable name. z91d has top-level side effects (the
# print + dir creation), but those are safe to repeat.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91d_mod", ROOT / "scripts/z91d_bsim4_port_fit_arclength.py")
z91d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91d)

load_curves = z91d.load_curves
PARAM_SPEC = z91d.PARAM_SPEC
make_thetas = z91d.make_thetas
clone_thetas = z91d.clone_thetas
thetas_to_values = z91d.thetas_to_values
init_theta = z91d.init_theta
patch_sd = z91d.patch_sd
forward_curve = z91d.forward_curve
huber_log_loss = z91d.huber_log_loss
run_stage = z91d.run_stage
multistart_stage = z91d.multistart_stage
evaluate_full = z91d.evaluate_full


# --------------------------------------------------------------------------- #
# Sebas anchors (from Image 2026-04-30 at 13.24 — manual digitization)
# --------------------------------------------------------------------------- #
# Format: param_name -> (target_value, lambda_weight)
# lambda is on a per-parameter loss scale (huber-on-log10-Id is O(1)).
SEBAS_ANCHORS = {
    "vth0":  (0.50, 0.5),    # strong: optimizer fled to 0.315 in z91d
    "beta0": (15.0, 0.05),   # weak: BETA0 in his plot may differ from BSIM4 beta0
}


def anchor_loss(thetas) -> torch.Tensor:
    """Quadratic pull of fitted params toward Sebas-extracted means."""
    values = thetas_to_values(thetas)
    total = torch.zeros((), dtype=torch.float64)
    for name, (target, lam) in SEBAS_ANCHORS.items():
        if name in values:
            v = values[name]
            spec = PARAM_SPEC[name]
            lo, hi = spec["bounds"]
            scale = (hi - lo)
            total = total + lam * ((v - target) / scale) ** 2
    return total


def stage_loss(thetas, model, cfg, curves, *, use_homotopy: bool = False):
    """z91d's stage_loss + anchor regularizer."""
    data_loss = z91d.stage_loss(thetas, model, cfg, curves,
                                 use_homotopy=use_homotopy)
    anc = anchor_loss(thetas)
    return data_loss + anc


# Monkey-patch so run_stage / multistart_stage (imported above) use OUR
# stage_loss (the closure inside run_stage looks up z91d.stage_loss by ref).
z91d.stage_loss = stage_loss


def main():
    t0 = time.time()
    print(f"[z91e] starting at {time.strftime('%H:%M:%S')}", flush=True)

    curves = load_curves()
    print(f"[z91e] loaded {len(curves)} curves", flush=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    text = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text, model_type="nmos")
    sd_M1 = compute_size_dep(model, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model, Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                              W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    thetas = make_thetas(seed=0)

    # --- Stage 1: off-state, anchored vth0 + GIDL --------------------------- #
    off_curves = [c for c in curves if c["VG1"] == 0.2]
    s1_names = ["vth0", "agidl", "bgidl", "cgidl", "egidl"]
    print(f"\n=== Stage 1: off-state ({len(off_curves)} curves) "
          f"anchors: vth0→0.50 V ===", flush=True)
    thetas, loss1 = multistart_stage(
        1, thetas, model, cfg, off_curves, s1_names,
        n_adam=30, n_lbfgs=15, n_seeds=2, t0=t0)

    # --- Stage 2: core transport ------------------------------------------- #
    sub_curves = [c for c in curves
                  if (c["Vd"] <= 0.8).all() or c["VG1"] <= 0.4]
    s2_names = ["u0", "vsat", "vth0", "k1", "k2"]
    print(f"\n=== Stage 2: core transport ({len(sub_curves)} curves) ===",
          flush=True)
    thetas, loss2 = multistart_stage(
        2, thetas, model, cfg, sub_curves, s2_names,
        n_adam=30, n_lbfgs=15, n_seeds=3, t0=t0)

    # --- Stage 3: snapback (gmin homotopy ON) ------------------------------ #
    s3_names = ["alpha0", "alpha1", "beta0", "Bf"]
    print(f"\n=== Stage 3: snapback (all 33 curves, homotopy ON) ===",
          flush=True)
    thetas, loss3 = multistart_stage(
        3, thetas, model, cfg, curves, s3_names,
        n_adam=30, n_lbfgs=15, n_seeds=3, t0=t0,
        use_homotopy=True)

    # --- Stage 4: polish all ----------------------------------------------- #
    s4_names = list(PARAM_SPEC.keys())
    print(f"\n=== Stage 4: L-BFGS polish all 13 params ===", flush=True)
    loss4 = run_stage(4, "", thetas, model, cfg, curves, s4_names,
                      n_adam=10, n_lbfgs=30, t0=t0, use_homotopy=True)

    # --- Save -------------------------------------------------------------- #
    final_values = {n: float(v.detach().item())
                    for n, v in thetas_to_values(thetas).items()}
    median_rmse, preds = evaluate_full(thetas, model, cfg, curves)

    summary = {
        "stage1_loss": loss1, "stage2_loss": loss2,
        "stage3_loss": loss3, "stage4_loss": loss4,
        "median_log_rmse": median_rmse,
        "anchors": {k: list(v) for k, v in SEBAS_ANCHORS.items()},
        "params": final_values,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "predictions.json").write_text(json.dumps(preds, indent=2))
    print(f"\n[z91e] DONE  median_log_rmse={median_rmse:.3f}  "
          f"params={final_values}", flush=True)


if __name__ == "__main__":
    main()
