"""z82 — P7v6: Stage-wise fit using the proper 2T NS-RAM cell topology.

Replaces z80's 1T-proxy (gamma_VG2 / Rb_leak / C_extra / I_PT0 hacks) with the
real 2-MOSFET + parasitic-NPN model defined in
``nsram/nsram/bsim4_port/nsram_cell_2T.py``. VG2 is now M2's actual gate;
body physics is self-consistent via Newton-Raphson on (Vsint, Vb).

Stages
  1. Off-state (VG1=0.2): GIDL params + vth0. BJT off, Iii off.
  2. Core transport (Vd<0.8 all VG1): u0, vsat, vth0, k1, k2. BJT/Iii off.
  3. Snapback (MERGED, all 33 curves): alpha0, beta0, Bf simultaneously,
     BJT+Iii on. Hard bounds prevent Iii suppression.
  4. Final polish: L-BFGS, all params unfrozen, balanced loss.

All bounded params use sigmoid reparametrization. Same `logb`/`linb` helpers
as z79/z80. M1 and M2 share the same fitted BSIM4 card values (only length
differs and is handled inside cfg).
"""
from __future__ import annotations
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

# GPU device selection (AMD ROCm gfx1151 with HSA override applied externally)
DEVICE = torch.device("cpu")  # CPU is faster than GPU for this small-batch workload
torch.set_default_device(DEVICE)
print(f"[z83] Using device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"[z83] GPU: {torch.cuda.get_device_name(0)}", flush=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, forward_2t,
)
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91_bsim4_port_fit_huber_homotopy"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loader (verbatim from z80)
# --------------------------------------------------------------------------- #
def parse_vg2(s):
    m = re.search(r"VG2=(-?\d+\.\d+)", s);  return float(m.group(1)) if m else None
def parse_vg1(s):
    m = re.search(r"VG1=([\d.]+)", s);      return float(m.group(1)) if m else None


def load_curves():
    curves = []
    for d in sorted(DATA_DIR.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
        VG1 = parse_vg1(d.name)
        for f in sorted(d.glob("*.csv")):
            VG2 = parse_vg2(f.name)
            data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
            if data.ndim == 1:
                continue
            half = len(data) // 2
            Vd = data[:half, 0]; Id = np.abs(data[:half, 1])
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) > 10:
                idx = np.linspace(0, len(Vd) - 1, 10).astype(int)
                Vd, Id = Vd[idx], Id[idx]
            if len(Vd) < 5:
                continue
            curves.append({"VG1": VG1, "VG2": VG2,
                           "Vd": torch.tensor(Vd, dtype=torch.float64),
                           "Id": torch.tensor(Id, dtype=torch.float64)})
    return curves


# --------------------------------------------------------------------------- #
# Param spec (oracle-recommended for v6 — no gamma/Rb/C/PT)
# --------------------------------------------------------------------------- #
PARAM_SPEC = {
    # transport — z91: tightened vth0 lower bound (5th-oracle finding: z88 fled
    # to 0.20 V to compensate for masked-loss + non-converged Vsint/Vb leakage,
    # producing Vth_eff(M2) ≈ -9 mV which is non-physical)
    "vth0":  {"kind": "linb", "init": 0.40,  "bounds": (0.30, 0.55)},
    "u0":    {"kind": "logb", "init": 0.06,  "bounds": (0.02, 0.15)},
    "vsat":  {"kind": "logb", "init": 1e5,   "bounds": (5e4, 3e5)},
    # k1, k2 tightened — z88 ended at k1=0.90 (upper bound)
    "k1":    {"kind": "linb", "init": 0.50,  "bounds": (0.30, 0.80)},
    "k2":    {"kind": "linb", "init": 0.00,  "bounds": (-0.10, 0.10)},
    # GIDL
    "agidl": {"kind": "logb", "init": 5e-7,  "bounds": (1e-7, 1e-5)},
    "bgidl": {"kind": "logb", "init": 8e8,   "bounds": (3e8, 1.2e9)},
    "cgidl": {"kind": "linb", "init": 0.5,   "bounds": (0.3, 0.7)},
    "egidl": {"kind": "linb", "init": 0.5,   "bounds": (0.3, 0.6)},
    # impact-ion
    "alpha0": {"kind": "logb", "init": 5e-3, "bounds": (1e-3, 5e-2)},
    "beta0":  {"kind": "linb", "init": 18.0, "bounds": (12.0, 30.0)},
    # BJT
    "Bf":     {"kind": "logb", "init": 100., "bounds": (50.0, 300.0)},
}

BSIM4_NAMES = {"vth0", "u0", "vsat", "k1", "k2",
               "agidl", "bgidl", "cgidl", "egidl",
               "alpha0", "beta0"}
BJT_NAMES = {"Bf"}


# --------------------------------------------------------------------------- #
# Reparametrization helpers (verbatim z80 idiom)
# --------------------------------------------------------------------------- #
def init_theta(name: str, jitter_seed: int = 0) -> torch.Tensor:
    spec = PARAM_SPEC[name]
    kind, init, bnd = spec["kind"], spec["init"], spec["bounds"]
    rng = np.random.default_rng(hash((name, jitter_seed)) & 0xFFFFFFFF)
    lo, hi = bnd
    if kind == "linb":
        u = (float(init) - lo) / (hi - lo)
    else:  # logb
        u = math.log(float(init) / lo) / math.log(hi / lo)
    u = min(max(u, 1e-6), 1.0 - 1e-6)
    theta0 = math.log(u / (1.0 - u))
    if jitter_seed != 0:
        theta0 += float(rng.normal(0, 0.3))
    return torch.tensor(theta0, dtype=torch.float64, requires_grad=True)


def theta_to_value(name: str, theta: torch.Tensor):
    spec = PARAM_SPEC[name]
    kind, bnd = spec["kind"], spec["bounds"]
    lo, hi = bnd
    s = torch.sigmoid(theta)
    if kind == "linb":
        return lo + (hi - lo) * s
    return lo * (hi / lo) ** s


def make_thetas(seed: int) -> dict:
    return {n: init_theta(n, seed) for n in PARAM_SPEC}


def thetas_to_values(thetas: dict) -> dict:
    return {n: theta_to_value(n, t) for n, t in thetas.items()}


def clone_thetas(thetas: dict) -> dict:
    return {n: t.detach().clone().requires_grad_(True) for n, t in thetas.items()}


# --------------------------------------------------------------------------- #
# SizeDep override — patch sd.scaled[...] / sd.vth0_T / sd.u0temp / sd.vsattemp
# while staying differentiable. Iii/GIDL params live in sd.scaled; transport
# params live as direct sd attributes. This is the wrinkle the spec's P_M1/P_M2
# dict abstracts over; we implement it explicitly so grads flow.
# --------------------------------------------------------------------------- #
SCALED_KEYS = {"k1", "k2", "agidl", "bgidl", "cgidl", "egidl",
               "alpha0", "beta0"}
ATTR_KEYS = {"vth0": "vth0_T", "u0": "u0temp", "vsat": "vsattemp"}


@contextmanager
def patch_sd(sd, values: dict):
    """Temporarily override SizeDependParam values for fitting.

    `values` maps PARAM_SPEC name -> tensor. We patch:
      - sd.scaled[name] for SCALED_KEYS
      - sd.<attrname> for ATTR_KEYS (vth0->vth0_T etc.)
    Non-bsim4 params ignored.
    """
    saved_scaled = {}
    saved_attr = {}
    try:
        for name, val in values.items():
            if name in SCALED_KEYS:
                saved_scaled[name] = sd.scaled.get(name, None)
                sd.scaled[name] = val
            elif name in ATTR_KEYS:
                attr = ATTR_KEYS[name]
                saved_attr[attr] = getattr(sd, attr)
                setattr(sd, attr, val)
        yield
    finally:
        for k, v in saved_scaled.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v
        for k, v in saved_attr.items():
            setattr(sd, k, v)


# --------------------------------------------------------------------------- #
# Forward: build cfg+sd once, patch with current θ each call.
# --------------------------------------------------------------------------- #
def make_cfg_and_sd(model: BSIM4Model, gates: dict):
    cfg = NSRAMCell2TConfig(
        Ln=180e-9, Wn=360e-9, M2_length_factor=10.0, T_C=27.0,
        use_iii=gates.get("use_iii", True),
        use_gidl=gates.get("use_gidl", True),
        use_bjt=gates.get("use_bjt", True),
    )
    # Force sd cache populate
    cfg.size_dep_M1(model)
    cfg.size_dep_M2(model)
    return cfg


def forward_curve(values: dict, model: BSIM4Model, cfg: NSRAMCell2TConfig,
                  VG1: float, VG2: float, Vd_seq: torch.Tensor,
                  use_homotopy: bool = False,
                  dense_vd_in_snapback: bool = True) -> tuple:
    """Run forward_2t with patched sd values.

    z91 changes:
    - use_homotopy=True for stage 3+ → gmin homotopy through snapback
    - dense_vd_in_snapback=True → tighter warm-start chain near the knee
    """
    sd_M1 = cfg._sd_M1
    sd_M2 = cfg._sd_M2
    bjt = GummelPoonNPN.from_sebas_card()
    if "Bf" in values:
        bjt.Bf = values["Bf"]

    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)

    with patch_sd(sd_M1, values), patch_sd(sd_M2, values):
        result = forward_2t(cfg, model, bjt, Vd_seq, VG1_t, VG2_t,
                            warm_start=True,
                            use_homotopy=use_homotopy,
                            dense_vd_in_snapback=dense_vd_in_snapback)
    converged_mask = torch.tensor([bool(c) for c in result["converged"]],
                                   dtype=torch.float64)
    return result["Id"].abs(), converged_mask


# --------------------------------------------------------------------------- #
# z91 Loss: Huber on log10(|Id|) with measurement-floor clip + non-conv penalty
# --------------------------------------------------------------------------- #
HUBER_DELTA = 1.0       # log-decade — Huber transition point
MEAS_FLOOR  = 1e-13     # A — measurement noise floor (instrument limit)
NONCONV_PENALTY = 4.0   # log-decade — penalty added at non-converged biases
                        #   (chosen so Newton failures hurt loss without exploding grad)


def huber_log_loss(pred: torch.Tensor, meas: torch.Tensor,
                   conv_mask: torch.Tensor) -> torch.Tensor:
    """Huber on log10(|Id|), with measurement floor + non-convergence penalty.

    No mask-skip: every bias contributes to the loss. Newton-failed biases are
    detached from the autograd graph (delta zeroed in nsram_cell_2T.py) but
    still receive a fixed scalar penalty, so the optimizer is rewarded for
    parameter sets where Newton converges more often.
    """
    log_eps = MEAS_FLOOR
    log_p = torch.log10(pred.abs().clamp_min(log_eps))
    log_m = torch.log10(meas.abs().clamp_min(log_eps))
    err = log_p - log_m
    a = err.abs()
    huber = torch.where(a < HUBER_DELTA,
                         0.5 * err * err,
                         HUBER_DELTA * (a - 0.5 * HUBER_DELTA))
    # Penalty for non-converged biases: a fixed log-decade equivalent
    nonconv = (1.0 - conv_mask) * (NONCONV_PENALTY ** 2 * 0.5)
    return (huber + nonconv).mean()


def stage_loss(thetas, model, cfg, curves, *, use_homotopy: bool = False):
    values = thetas_to_values(thetas)
    losses = []
    for c in curves:
        try:
            Id_pred, conv_mask = forward_curve(values, model, cfg,
                                                c["VG1"], c["VG2"], c["Vd"],
                                                use_homotopy=use_homotopy)
        except RuntimeError as e:
            print(f"    skip VG1={c['VG1']} VG2={c['VG2']}: {e}", flush=True)
            continue
        l = huber_log_loss(Id_pred, c["Id"], conv_mask)
        if torch.isfinite(l):
            losses.append(l)
    if not losses:
        return torch.tensor(1e3, dtype=torch.float64, requires_grad=True)
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# Stage runner
# --------------------------------------------------------------------------- #
def run_stage(stage_id: int, label: str, thetas: dict, model, cfg, curves,
              fit_names: list, *, n_adam: int, n_lbfgs: int,
              lr_adam: float = 0.05, lr_lbfgs: float = 0.5,
              t0: float = 0.0, use_homotopy: bool = False):
    fit_thetas = [thetas[n] for n in fit_names]
    for n in fit_names:
        thetas[n].requires_grad_(True)

    with torch.no_grad():
        l0 = stage_loss(thetas, model, cfg, curves, use_homotopy=use_homotopy)
    print(f"[stage {stage_id}{label}] init loss = {l0.item():.4f}  "
          f"(fitting {len(fit_names)}, {len(curves)} curves, "
          f"homotopy={use_homotopy})  ({time.time()-t0:.0f}s)", flush=True)

    if n_adam > 0:
        opt = torch.optim.Adam(fit_thetas, lr=lr_adam)
        for it in range(n_adam):
            opt.zero_grad()
            l = stage_loss(thetas, model, cfg, curves, use_homotopy=use_homotopy)
            l.backward()
            torch.nn.utils.clip_grad_norm_(fit_thetas, max_norm=2.0)
            opt.step()
            if it % 5 == 0 or it == n_adam - 1:
                print(f"  s{stage_id}{label} Adam {it}: loss={l.item():.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    if n_lbfgs > 0:
        opt2 = torch.optim.LBFGS(fit_thetas, max_iter=n_lbfgs, lr=lr_lbfgs,
                                  line_search_fn="strong_wolfe")
        def closure():
            opt2.zero_grad()
            l = stage_loss(thetas, model, cfg, curves, use_homotopy=use_homotopy)
            l.backward()
            return l
        try:
            opt2.step(closure)
        except RuntimeError as e:
            print(f"  s{stage_id}{label} L-BFGS warn: {e}", flush=True)

    with torch.no_grad():
        lf = stage_loss(thetas, model, cfg, curves, use_homotopy=use_homotopy)
    print(f"[stage {stage_id}{label}] final loss = {lf.item():.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return float(lf.item())


def multistart_stage(stage_id, base_thetas, model, cfg, curves, fit_names,
                     *, n_adam, n_lbfgs, n_seeds=3, t0,
                     lr_adam: float = 0.05, lr_lbfgs: float = 0.5,
                     use_homotopy: bool = False):
    best_loss = float("inf")
    best_thetas = None
    for seed in range(n_seeds):
        thetas = clone_thetas(base_thetas)
        if seed > 0:
            for n in fit_names:
                thetas[n] = init_theta(n, jitter_seed=seed)
        loss = run_stage(stage_id, f".s{seed}", thetas, model, cfg, curves,
                         fit_names, n_adam=n_adam, n_lbfgs=n_lbfgs, t0=t0,
                         lr_adam=lr_adam, lr_lbfgs=lr_lbfgs,
                         use_homotopy=use_homotopy)
        if loss < best_loss:
            best_loss = loss
            best_thetas = clone_thetas(thetas)
            print(f"  ** stage {stage_id} new best @ seed {seed}: "
                  f"loss={loss:.4f}", flush=True)
    return best_thetas, best_loss


# --------------------------------------------------------------------------- #
# Eval + save
# --------------------------------------------------------------------------- #
def evaluate_full(thetas, model, cfg, curves):
    log_eps = 1e-15
    values = thetas_to_values(thetas)
    rmses, preds = [], []
    for c in curves:
        try:
            with torch.no_grad():
                Id_pred = forward_curve(values, model, cfg,
                                        c["VG1"], c["VG2"], c["Vd"],
                                        use_homotopy=True)
                if isinstance(Id_pred, tuple):
                    Id_pred, conv_mask = Id_pred
                else:
                    conv_mask = torch.ones_like(Id_pred, dtype=torch.bool)
        except RuntimeError as e:
            print(f"  eval skip VG1={c['VG1']} VG2={c['VG2']}: {e}", flush=True)
            continue
        log_p = torch.log10(Id_pred.abs() + log_eps)
        log_m = torch.log10(c["Id"].abs() + log_eps)
        # honest per-curve metric: only count converged biases
        cm = conv_mask.bool() if conv_mask.dtype != torch.bool else conv_mask
        if cm.any():
            sq = (log_p - log_m) ** 2
            rmse = float(torch.sqrt(sq[cm].mean()).item())
        else:
            rmse = float("inf")
        rmses.append(rmse)
        preds.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse": rmse,
                      "Vd": c["Vd"].numpy().tolist(),
                      "Id_meas": c["Id"].numpy().tolist(),
                      "Id_pred": Id_pred.detach().numpy().tolist(),
                      "converged": cm.detach().numpy().tolist()})
    return float(np.median(rmses)) if rmses else float("inf"), preds


def fitted_dict(thetas):
    return {n: float(theta_to_value(n, thetas[n]).detach().item()) for n in thetas}


def save_stage_summary(stage_id, thetas, loss):
    p = OUT / f"stage{stage_id}_summary.json"
    p.write_text(json.dumps(
        {"stage": stage_id, "loss": loss, "params": fitted_dict(thetas)},
        indent=2))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    base_card_text = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(base_card_text)
    curves = load_curves()
    print(f"Loaded {len(curves)} curves at {time.time()-t0:.1f}s", flush=True)

    # Curve subsets
    off_state = [c for c in curves if abs(c["VG1"] - 0.2) < 1e-3]
    low_vd = []
    for c in curves:
        mask = c["Vd"] < 0.8
        if mask.sum().item() >= 4:
            low_vd.append({"VG1": c["VG1"], "VG2": c["VG2"],
                           "Vd": c["Vd"][mask], "Id": c["Id"][mask]})
    print(f"  off_state (VG1=0.2): {len(off_state)}", flush=True)
    print(f"  low_vd  (Vd<0.8):    {len(low_vd)}", flush=True)
    print(f"  full curves:         {len(curves)}", flush=True)

    # Initialize thetas
    thetas = make_thetas(seed=0)

    # --- Stage 1: off-state, GIDL + vth0; BJT off, Iii off --- #
    cfg1 = make_cfg_and_sd(model, gates={"use_iii": False, "use_gidl": True,
                                          "use_bjt": False})
    s1_names = ["agidl", "bgidl", "cgidl", "egidl", "vth0"]
    print("\n=== Stage 1: Off-state (GIDL + vth0) ===", flush=True)
    thetas, l1 = multistart_stage(1, thetas, model, cfg1, off_state,
                                   s1_names, n_adam=30, n_lbfgs=15,
                                   n_seeds=3, t0=t0)
    save_stage_summary(1, thetas, l1)

    # --- Stage 2: core transport, low Vd; BJT off, Iii off --- #
    cfg2 = make_cfg_and_sd(model, gates={"use_iii": False, "use_gidl": True,
                                          "use_bjt": False})
    s2_names = ["u0", "vsat", "vth0", "k1", "k2"]
    print("\n=== Stage 2: Core transport (low-Vd) ===", flush=True)
    thetas, l2 = multistart_stage(2, thetas, model, cfg2, low_vd,
                                   s2_names, n_adam=30, n_lbfgs=15,
                                   n_seeds=3, t0=t0)
    save_stage_summary(2, thetas, l2)

    # --- Stage 3: snapback (MERGED) full data, BJT+Iii ON --- #
    # Stage 3 explodes with default lr=0.05 because BJT exp(Vbe/Vt) is
    # exponentially sensitive. Lower lr + fewer seeds (Stage 1+2 already
    # explored basins; Stage 3 just needs to balance Iii-vs-BJT).
    cfg3 = make_cfg_and_sd(model, gates={"use_iii": True, "use_gidl": True,
                                          "use_bjt": True})
    s3_names = ["alpha0", "beta0", "Bf"]
    print("\n=== Stage 3: Snapback merged (alpha0+beta0+Bf, gmin homotopy ON) ===", flush=True)
    thetas, l3 = multistart_stage(3, thetas, model, cfg3, curves,
                                   s3_names, n_adam=30, n_lbfgs=15,
                                   n_seeds=2, t0=t0,
                                   lr_adam=0.005, lr_lbfgs=0.1,
                                   use_homotopy=True)
    save_stage_summary(3, thetas, l3)

    # --- Stage 4: final polish, all params, L-BFGS only --- #
    cfg4 = make_cfg_and_sd(model, gates={"use_iii": True, "use_gidl": True,
                                          "use_bjt": True})
    s4_names = list(PARAM_SPEC.keys())
    print(f"\n=== Stage 4: Final polish (L-BFGS, all {len(s4_names)} params, "
          f"homotopy ON) ===", flush=True)
    l4 = run_stage(4, "", thetas, model, cfg4, curves,
                   s4_names, n_adam=0, n_lbfgs=25, t0=t0,
                   use_homotopy=True)
    save_stage_summary(4, thetas, l4)

    # --- Final eval --- #
    median_rmse, preds = evaluate_full(thetas, model, cfg4, curves)
    print(f"\n=== Final median log-RMSE = {median_rmse:.3f} ===", flush=True)

    fitted = fitted_dict(thetas)
    summary = {
        "stage_losses": {"1": l1, "2": l2, "3": l3, "4": l4},
        "median_log_rmse": median_rmse,
        "fitted_params": fitted,
        "elapsed_s": time.time() - t0,
        "n_curves": len(curves),
        "config": "z91 — Huber-on-log10|Id| + gmin homotopy + tight vth0/k1/k2 bounds + IFT-gated",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "per_curve.json").write_text(json.dumps(preds, indent=1))

    print("\nFitted params:")
    for k, v in fitted.items():
        print(f"  {k:14s} = {v:+.4e}", flush=True)
    print(f"\nTotal elapsed: {summary['elapsed_s']:.0f}s")

    # 3-panel plot
    by_vg1 = {}
    for p in preds:
        by_vg1.setdefault(p["VG1"], []).append(p)
    if by_vg1:
        fig, axes = plt.subplots(1, max(len(by_vg1), 1), figsize=(6 * len(by_vg1), 6),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        cmap = plt.get_cmap("viridis")
        for ax, VG1 in zip(axes, sorted(by_vg1)):
            ps = sorted(by_vg1[VG1], key=lambda c: c["VG2"])
            n = len(ps)
            for i, p in enumerate(ps):
                color = cmap(i / max(n - 1, 1))
                Vd = np.asarray(p["Vd"])
                ax.semilogy(Vd, p["Id_meas"], "o", color=color, ms=4, alpha=0.7,
                             label=f"VG2={p['VG2']:+.2f}")
                ax.semilogy(Vd, p["Id_pred"], "-", color=color, lw=1.5)
            ax.set_xlabel("Vd [V]"); ax.grid(alpha=0.3)
            ax.set_title(f"VG1 = {VG1} V    ({n} curves)")
            ax.legend(loc="lower right", fontsize=7, ncol=2)
        axes[0].set_ylabel("|Id| [A]")
        fig.suptitle(
            f"P7v6: 2T topology stage-wise fit\n"
            f"median log-RMSE = {median_rmse:.2f}  —  elapsed {summary['elapsed_s']:.0f}s",
            fontsize=12, weight="bold",
        )
        fig.tight_layout()
        fig.savefig(OUT / "fit_curves.png", dpi=140)
        plt.close(fig)
        print(f"Wrote {OUT/'fit_curves.png'}")


if __name__ == "__main__":
    main()
