"""z79 — P7v5: Stage-wise fitting with hard parameter bounds.

Oracle consensus prescription (4 LLMs unanimous):
  - v2/v3/v4 all collapsed into a "give-up-on-snapback" local minimum where the
    optimizer suppresses Iii (alpha0 → 2.4e-5) and inflates GIDL/Bf to compensate
    elsewhere. Joint optimization of all 21 params over all 33 curves can't
    escape because lowering Iii reduces low-Vd error faster than raising it
    improves snapback.
  - Fix: stage-wise fitting. Anchor leakage first, then transport, then Iii,
    then BJT. Use HARD bounds via reparametrization so optimizer can't escape
    physical ranges.

Stages:
  1. Off-state (VG1=0.2V, all Vd) — agidl, bgidl, cgidl, egidl, vth0, nfactor,
     voff, cdsc.   Iii / BJT / PT all OFF.
  2. Core transport (all VG1, Vd<0.8V) — u0, vsat, vth0(refit), k1, k2, eta0,
     dsub, rdsw.  GIDL frozen.
  3. Impact-ion (VG1=0.6V, all Vd) — alpha0 ∈ [5e-4,5e-2], beta0 ∈ [12,30].
  4. Bipolar feedback (all data) — Bf ∈ [50,300], Rb_leak ≥ 1e10, gamma_VG2 ∈
     [-0.5, +0.5].
  5. Final polish — L-BFGS only, all params unfrozen with same hard bounds.

Reparametrization
  Lin-bounded:  v = lo + (hi-lo)*sigmoid(theta)
  Log-bounded:  v = lo * (hi/lo)**sigmoid(theta)
  Log-free:     v = 10**theta            (unchanged from v4)
  Lin-free:     v = theta                (unchanged from v4)

The forward pass is identical to v4 EXCEPT we inline feature-gating flags
(enable_iii, enable_gidl, enable_bjt, enable_pt) so a stage can selectively
zero out a contribution without touching NSRAMCellConfig (v4 didn't use it).
"""
from __future__ import annotations
import json
import math
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

from nsram.bsim4_port.bjt import GummelPoonNPN, compute_bjt
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.leak import (
    compute_iimpact, compute_igidl_gisl, compute_igb,
)
from nsram.bsim4_port.diode import compute_body_diodes
from nsram.bsim4_port.caps import compute_caps
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.temp import compute_size_dep

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z79_bsim4_port_fit_p7v5"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loader (verbatim from v4)
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
# Param spec  (kind, init, bounds)  — bounds=None -> v4-style log/lin free
#
# Fields:
#   kind: "log" | "lin" | "logb" | "linb"
#     log  -> v = 10**theta    (unbounded, log-init)
#     lin  -> v = theta        (unbounded, lin-init)
#     logb -> v = lo*(hi/lo)**sigmoid(theta)
#     linb -> v = lo + (hi-lo)*sigmoid(theta)
# --------------------------------------------------------------------------- #
PARAM_SPEC = {
    # Stage 1 — off-state
    "agidl":      {"kind": "logb", "init": 5e-7,  "bounds": (1e-7, 1e-5)},
    "bgidl":      {"kind": "logb", "init": 8e8,   "bounds": (3e8,  1.2e9)},
    "cgidl":      {"kind": "linb", "init": 0.5,   "bounds": (0.3,  0.7)},
    "egidl":      {"kind": "linb", "init": 0.5,   "bounds": (0.3,  0.6)},
    # Mirror GISL with same init/bounds (symmetric leak)
    "agisl":      {"kind": "logb", "init": 5e-7,  "bounds": (1e-7, 1e-5)},
    "bgisl":      {"kind": "logb", "init": 8e8,   "bounds": (3e8,  1.2e9)},
    "cgisl":      {"kind": "linb", "init": 0.5,   "bounds": (0.3,  0.7)},
    "egisl":      {"kind": "linb", "init": 0.5,   "bounds": (0.3,  0.6)},
    # SS / off-state shapers
    "vth0":       {"kind": "linb", "init": 0.45,  "bounds": (0.2,  0.6)},
    "nfactor":    {"kind": "linb", "init": 1.5,   "bounds": (0.5,  4.0)},
    "voff":       {"kind": "linb", "init": -0.08, "bounds": (-0.25, 0.0)},
    "cdsc":       {"kind": "logb", "init": 2.4e-4,"bounds": (1e-5, 1e-2)},

    # Stage 2 — core transport
    "u0":         {"kind": "logb", "init": 0.048, "bounds": (0.02, 0.15)},
    "vsat":       {"kind": "logb", "init": 1.35e5,"bounds": (5e4,  3e5)},
    "k1":         {"kind": "linb", "init": 0.5,   "bounds": (0.1,  1.5)},
    "k2":         {"kind": "linb", "init": 0.0,   "bounds": (-0.5, 0.5)},
    "eta0":       {"kind": "linb", "init": 0.08,  "bounds": (0.0,  0.5)},
    "dsub":       {"kind": "linb", "init": 0.56,  "bounds": (0.0,  3.0)},
    "rdsw":       {"kind": "logb", "init": 200.0, "bounds": (10.0, 5000.0)},

    # Stage 3 — impact-ion (HARD bounds, prevent escape)
    "alpha0":     {"kind": "logb", "init": 5e-3,  "bounds": (5e-4, 5e-2)},
    "beta0":      {"kind": "linb", "init": 18.0,  "bounds": (12.0, 30.0)},

    # Stage 4 — bipolar feedback
    "Bf":         {"kind": "logb", "init": 100.0, "bounds": (50.0, 300.0)},
    "Rb_leak":    {"kind": "logb", "init": 1e11,  "bounds": (1e10, 1e13)},
    "gamma_VG2":  {"kind": "linb", "init": 0.3,   "bounds": (-0.5, 0.5)},

    # Auxiliary cell params (kept as in v4, unbounded)
    "C_extra":    {"kind": "log",  "init": 1e-15, "bounds": None},
    "I_PT0":      {"kind": "log",  "init": 1e-6,  "bounds": None},
    "V_PT_th":    {"kind": "lin",  "init": 0.95,  "bounds": None},
    "V_PT_scale": {"kind": "log",  "init": 0.02,  "bounds": None},
    "k_Vb_PT":    {"kind": "lin",  "init": 1.5,   "bounds": None},
}

# Names that go into the BSIM4Model card (compute_dc et al. read these)
BSIM_NAMES = {"vth0", "u0", "vsat", "alpha0", "beta0",
              "agidl", "bgidl", "cgidl", "egidl",
              "agisl", "bgisl", "cgisl", "egisl",
              "nfactor", "voff", "cdsc", "k1", "k2",
              "eta0", "dsub", "rdsw"}

CELL_NAMES = {"gamma_VG2", "Rb_leak", "C_extra", "Bf",
              "I_PT0", "V_PT_th", "V_PT_scale", "k_Vb_PT"}


# --------------------------------------------------------------------------- #
# Reparametrization helpers
# --------------------------------------------------------------------------- #
def init_theta(name: str, jitter_seed: int = 0) -> torch.Tensor:
    spec = PARAM_SPEC[name]
    kind, init, bnd = spec["kind"], spec["init"], spec["bounds"]
    rng = np.random.default_rng(hash((name, jitter_seed)) & 0xFFFFFFFF)
    if kind == "log":
        j = 1.0 if jitter_seed == 0 else float(np.exp(rng.normal(0, 0.3)))
        return torch.tensor(float(np.log10(init * j)),
                            dtype=torch.float64, requires_grad=True)
    if kind == "lin":
        j = 0.0 if jitter_seed == 0 else float(rng.normal(0, 0.05))
        return torch.tensor(float(init + j),
                            dtype=torch.float64, requires_grad=True)
    lo, hi = bnd
    # invert sigmoid: theta = logit((init - lo)/(hi - lo)) for linb
    #                  theta = logit(log(init/lo)/log(hi/lo)) for logb
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
    if kind == "log":
        return 10.0 ** theta
    if kind == "lin":
        return theta
    lo, hi = bnd
    s = torch.sigmoid(theta)
    if kind == "linb":
        return lo + (hi - lo) * s
    # logb: log-linear interpolation
    return lo * (hi / lo) ** s


def make_thetas(seed: int) -> dict:
    return {n: init_theta(n, seed) for n in PARAM_SPEC}


def thetas_to_values(thetas: dict) -> dict:
    return {n: theta_to_value(n, t) for n, t in thetas.items()}


# --------------------------------------------------------------------------- #
# Model build & forward
# --------------------------------------------------------------------------- #
def build_model(values: dict, base_card_text: str) -> BSIM4Model:
    m = BSIM4Model.from_spice(base_card_text)
    for name in BSIM_NAMES:
        v = values[name]
        m._values[name] = v if isinstance(v, torch.Tensor) else float(v)
        m._given.add(name)
    return m


def forward_curve(values: dict, base_card: str, geom: Geometry,
                  VG1: float, VG2: float, Vd_seq: torch.Tensor,
                  enable_iii: bool = True,
                  enable_gidl: bool = True,
                  enable_bjt: bool = True,
                  enable_pt: bool = True,
                  n_substeps: int = 2, dt: float = 5e-9) -> torch.Tensor:
    """Inlined v4 forward pass, with feature-gating flags."""
    model = build_model(values, base_card)
    sd = compute_size_dep(model, geom, T_C=27.0)
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = values["Bf"]

    gamma   = values["gamma_VG2"]
    Rb_leak = values["Rb_leak"]
    C_extra = values["C_extra"]
    I_PT0   = values["I_PT0"]
    V_PT_th = values["V_PT_th"]
    V_PT_scale = values["V_PT_scale"]
    k_Vb_PT = values["k_Vb_PT"]

    # Back-gate Vth shift (only meaningful when gamma_VG2 is unfrozen)
    # VG2 sign convention (FIXED per GPT review 2026-04-29):
    # wrapper uses vth0_eff = vth0 + gamma*VG2; we now match.
    # Old fit scripts used minus sign — fitted gammas need negation if loaded.
    sd.vth0_T = sd.vth0_T + gamma * VG2

    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    Vb = torch.tensor(0.0, dtype=torch.float64)
    W, L = geom.W, geom.L
    As_v = W * L; Ad_v = W * L
    Ps_v = 2.0 * (W + L); Pd_v = 2.0 * (W + L)

    Id_preds = []
    for k in range(int(Vd_seq.shape[0])):
        vd = Vd_seq[k]
        Vbs_t = Vb.unsqueeze(0)
        Vd_t  = vd.unsqueeze(0)
        Vg_t  = VG1_t.unsqueeze(0)

        for _ in range(n_substeps):
            Vbd_t = (Vb - vd).unsqueeze(0)
            r = compute_dc(model, sd, Vgs=Vg_t, Vds=Vd_t, Vbs=Vbs_t)

            if enable_iii:
                Iii = compute_iimpact(model, sd, r, Vds=Vd_t).squeeze(0)
            else:
                Iii = torch.zeros((), dtype=torch.float64)

            if enable_gidl:
                Igidl, Igisl = compute_igidl_gisl(model, sd, Vgs=Vg_t,
                                                   Vds=Vd_t, Vbs=Vbs_t)
                Igidl, Igisl = Igidl.squeeze(0), Igisl.squeeze(0)
            else:
                Igidl = torch.zeros((), dtype=torch.float64)
                Igisl = torch.zeros((), dtype=torch.float64)

            Igb = compute_igb(model, sd, Vgs=Vg_t, Vbs=Vbs_t,
                              dc_result=r).squeeze(0)
            Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs_t, Vbd=Vbd_t,
                                            As=As_v, Ad=Ad_v, Ps=Ps_v, Pd=Pd_v)
            Ibs, Ibd = Ibs.squeeze(0), Ibd.squeeze(0)
            cap = compute_caps(model, sd, r, Vgs=Vg_t, Vds=Vd_t,
                                Vbs=Vbs_t, Vbd=Vbd_t,
                                As=As_v, Ad=Ad_v, Ps=Ps_v, Pd=Pd_v)

            if enable_bjt:
                bjt_out = compute_bjt(bjt, Vbe=Vb.unsqueeze(0),
                                       Vbc=(Vb - vd).unsqueeze(0), T_K=300.15)
                Ib = bjt_out["Ib"].squeeze(0)
            else:
                Ib = torch.zeros((), dtype=torch.float64)

            Ileak = Vb / Rb_leak

            if enable_pt:
                scale_safe = (V_PT_scale.clamp_min(1e-6)
                              if isinstance(V_PT_scale, torch.Tensor)
                              else max(float(V_PT_scale), 1e-6))
                drive = (vd + k_Vb_PT * Vb - V_PT_th) / scale_safe
                I_PT = I_PT0 * V_PT_scale * F.softplus(drive)
            else:
                I_PT = torch.zeros((), dtype=torch.float64)

            I_total = (Iii - Ibd - Ibs + Igidl + Igisl + Igb
                       - Ib - Ileak + I_PT)
            C_body = (cap.Cjs.squeeze(0) + cap.Cjd.squeeze(0)
                      + torch.abs(cap.Cgb.squeeze(0)) + C_extra + 1e-30)
            dVb = I_total / C_body
            Vb = Vb + dt * torch.clamp(dVb, -1e10, 1e10)
            Vb = torch.clamp(Vb, -0.5, 0.8)
            Vbs_t = Vb.unsqueeze(0)

        # Drain current = Ids + Ic + I_PT_drain
        r_final = compute_dc(model, sd, Vgs=Vg_t, Vds=Vd_t, Vbs=Vbs_t)
        Ids_mos = r_final.Ids.squeeze(0).abs()

        if enable_bjt:
            bjt_final = compute_bjt(bjt, Vbe=Vb.unsqueeze(0),
                                     Vbc=(Vb - vd).unsqueeze(0), T_K=300.15)
            Ic = bjt_final["Ic"].squeeze(0).abs()
        else:
            Ic = torch.zeros((), dtype=torch.float64)

        if enable_pt:
            scale_safe = (V_PT_scale.clamp_min(1e-6)
                          if isinstance(V_PT_scale, torch.Tensor)
                          else max(float(V_PT_scale), 1e-6))
            drive = (vd + k_Vb_PT * Vb - V_PT_th) / scale_safe
            I_PT_drain = I_PT0 * V_PT_scale * F.softplus(drive)
        else:
            I_PT_drain = torch.zeros((), dtype=torch.float64)

        Id_preds.append(Ids_mos + Ic + I_PT_drain)
    return torch.stack(Id_preds)


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #
def _logmse(Id_pred, Id_meas):
    log_eps = 1e-15
    log_p = torch.log(Id_pred.clamp_min(log_eps))
    log_m = torch.log(Id_meas.clamp_min(log_eps))
    return ((log_p - log_m) ** 2).mean(), log_p, log_m


def stage_loss(thetas, base_card, geom, curves, *, gates,
               balanced=False):
    """Plain log-MSE (or balanced) over a curves subset."""
    values = thetas_to_values(thetas)
    losses = []
    for c in curves:
        Id_pred = forward_curve(values, base_card, geom,
                                 c["VG1"], c["VG2"], c["Vd"],
                                 **gates)
        l_abs, log_p, log_m = _logmse(Id_pred, c["Id"])
        if balanced:
            w = torch.zeros_like(log_m)
            w[:-1] = (log_m[1:] - log_m[:-1]).abs()
            w[-1]  = w[-2]
            w = (w + 0.1) / (w.mean() + 0.1)
            l_knee = ((log_p - log_m) ** 2 * w).mean()
            losses.append(0.5 * l_abs + 0.5 * l_knee)
        else:
            losses.append(l_abs)
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# Stage runner
# --------------------------------------------------------------------------- #
def run_stage(stage_id: int, thetas: dict, base_card, geom, curves_subset,
              fit_names: list, gates: dict, balanced: bool,
              n_adam: int = 30, n_lbfgs: int = 15,
              lr_adam: float = 0.05, lr_lbfgs: float = 0.5,
              t0: float = 0.0, label: str = ""):
    """Optimize only `fit_names`, freezing the rest. Returns final loss."""
    fit_thetas = [thetas[n] for n in fit_names]
    # Make sure all need grad
    for n in fit_names:
        if not thetas[n].requires_grad:
            thetas[n].requires_grad_(True)

    with torch.no_grad():
        l0 = stage_loss(thetas, base_card, geom, curves_subset,
                        gates=gates, balanced=balanced)
    print(f"[stage {stage_id}{label}] init loss = {l0.item():.4f}  "
          f"(fitting {len(fit_names)} params, {len(curves_subset)} curves)",
          flush=True)

    opt = torch.optim.Adam(fit_thetas, lr=lr_adam)
    for it in range(n_adam):
        opt.zero_grad()
        l = stage_loss(thetas, base_card, geom, curves_subset,
                       gates=gates, balanced=balanced)
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
            l = stage_loss(thetas, base_card, geom, curves_subset,
                           gates=gates, balanced=balanced)
            l.backward()
            return l
        try:
            opt2.step(closure)
        except RuntimeError as e:
            print(f"  s{stage_id}{label} L-BFGS warn: {e}", flush=True)

    with torch.no_grad():
        lf = stage_loss(thetas, base_card, geom, curves_subset,
                        gates=gates, balanced=balanced)
    print(f"[stage {stage_id}{label}] final loss = {lf.item():.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return float(lf.item())


def clone_thetas(thetas: dict) -> dict:
    """Detach + clone, preserving requires_grad."""
    return {n: t.detach().clone().requires_grad_(True)
            for n, t in thetas.items()}


def multistart_stage(stage_id, base_thetas, base_card, geom, curves_subset,
                     fit_names, gates, balanced, *, t0,
                     n_adam=30, n_lbfgs=15, n_seeds=2):
    """Run stage with n_seeds starts; return best thetas + loss."""
    best_loss = float("inf")
    best_thetas = None
    for seed in range(n_seeds):
        thetas = clone_thetas(base_thetas)
        if seed > 0:
            # Re-jitter only the fit_names
            for n in fit_names:
                thetas[n] = init_theta(n, jitter_seed=seed)
        loss = run_stage(stage_id, thetas, base_card, geom, curves_subset,
                          fit_names, gates, balanced,
                          n_adam=n_adam, n_lbfgs=n_lbfgs, t0=t0,
                          label=f".s{seed}")
        if loss < best_loss:
            best_loss = loss
            best_thetas = clone_thetas(thetas)
            print(f"  ** stage {stage_id} new best @ seed {seed}: "
                  f"loss={loss:.4f}", flush=True)
    return best_thetas, best_loss


# --------------------------------------------------------------------------- #
# Evaluation + save
# --------------------------------------------------------------------------- #
def evaluate_full(thetas, base_card, geom, curves, gates):
    log_eps = 1e-15
    values = thetas_to_values(thetas)
    rmses = []; preds = []
    for c in curves:
        with torch.no_grad():
            Id_pred = forward_curve(values, base_card, geom,
                                     c["VG1"], c["VG2"], c["Vd"], **gates)
        log_p = torch.log(Id_pred.clamp_min(log_eps))
        log_m = torch.log(c["Id"].clamp_min(log_eps))
        rmse = float(torch.sqrt(((log_p - log_m) ** 2).mean()).item())
        rmses.append(rmse)
        preds.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse": rmse,
                      "Vd": c["Vd"].numpy().tolist(),
                      "Id_meas": c["Id"].numpy().tolist(),
                      "Id_pred": Id_pred.detach().numpy().tolist()})
    return float(np.median(rmses)), preds


def fitted_dict(thetas: dict) -> dict:
    out = {}
    for n in thetas:
        out[n] = float(theta_to_value(n, thetas[n]).detach().item())
    return out


def save_stage_summary(stage_id: int, thetas: dict, loss: float):
    p = OUT / f"stage{stage_id}_summary.json"
    p.write_text(json.dumps(
        {"stage": stage_id, "loss": loss, "params": fitted_dict(thetas)},
        indent=2))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    base_card = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    geom = Geometry(L=180e-9, W=360e-9)
    curves = load_curves()
    print(f"Loaded {len(curves)} curves at {time.time()-t0:.1f}s", flush=True)

    # Curve subsets
    off_state = [c for c in curves if abs(c["VG1"] - 0.2) < 1e-3]
    low_vd = [{**c,
               "Vd": c["Vd"][c["Vd"] < 0.8],
               "Id": c["Id"][c["Vd"] < 0.8]}
              for c in curves]
    low_vd = [c for c in low_vd if c["Vd"].numel() >= 4]
    snapback = [c for c in curves if abs(c["VG1"] - 0.6) < 1e-3]

    print(f"  off_state curves (VG1=0.2): {len(off_state)}", flush=True)
    print(f"  low_vd points (Vd<0.8):     {len(low_vd)}", flush=True)
    print(f"  snapback curves (VG1=0.6):  {len(snapback)}", flush=True)

    # Initialize base thetas (seed 0)
    thetas = make_thetas(seed=0)

    # ---------------- Stage 1 ---------------- #
    stage1_names = ["agidl", "bgidl", "cgidl", "egidl",
                    "agisl", "bgisl", "cgisl", "egisl",
                    "vth0", "nfactor", "voff", "cdsc"]
    gates1 = {"enable_iii": False, "enable_gidl": True,
              "enable_bjt": False, "enable_pt": False}
    thetas, l1 = multistart_stage(1, thetas, base_card, geom, off_state,
                                   stage1_names, gates1, balanced=False,
                                   t0=t0)
    save_stage_summary(1, thetas, l1)

    # ---------------- Stage 2 ---------------- #
    stage2_names = ["u0", "vsat", "vth0", "k1", "k2",
                    "eta0", "dsub", "rdsw"]
    gates2 = {"enable_iii": False, "enable_gidl": True,
              "enable_bjt": False, "enable_pt": False}
    thetas, l2 = multistart_stage(2, thetas, base_card, geom, low_vd,
                                   stage2_names, gates2, balanced=False,
                                   t0=t0)
    save_stage_summary(2, thetas, l2)

    # ---------------- Stage 3 ---------------- #
    stage3_names = ["alpha0", "beta0"]
    gates3 = {"enable_iii": True, "enable_gidl": True,
              "enable_bjt": False, "enable_pt": False}
    thetas, l3 = multistart_stage(3, thetas, base_card, geom, snapback,
                                   stage3_names, gates3, balanced=False,
                                   t0=t0)
    save_stage_summary(3, thetas, l3)

    # ---------------- Stage 4 ---------------- #
    stage4_names = ["Bf", "Rb_leak", "gamma_VG2"]
    gates4 = {"enable_iii": True, "enable_gidl": True,
              "enable_bjt": True, "enable_pt": True}
    thetas, l4 = multistart_stage(4, thetas, base_card, geom, curves,
                                   stage4_names, gates4, balanced=True,
                                   t0=t0)
    save_stage_summary(4, thetas, l4)

    # ---------------- Stage 5 — final polish ---------------- #
    stage5_names = list(PARAM_SPEC.keys())
    gates5 = gates4
    print(f"\n=== Stage 5: final polish (L-BFGS, all {len(stage5_names)} params) ===",
          flush=True)
    l5 = run_stage(5, thetas, base_card, geom, curves,
                    stage5_names, gates5, balanced=True,
                    n_adam=0, n_lbfgs=20, t0=t0, label="")
    save_stage_summary(5, thetas, l5)

    # ---------------- Final eval + save ---------------- #
    median_rmse, preds = evaluate_full(thetas, base_card, geom, curves, gates5)
    print(f"\n=== Final median log-RMSE = {median_rmse:.3f} ===", flush=True)

    fitted = fitted_dict(thetas)
    summary = {
        "stage_losses": {"1": l1, "2": l2, "3": l3, "4": l4, "5": l5},
        "median_log_rmse": median_rmse,
        "fitted_params": fitted,
        "elapsed_s": time.time() - t0,
        "n_curves": len(curves),
        "config": "stagewise + hard-bounds (P7v5)",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                                   default=str))
    (OUT / "per_curve.json").write_text(json.dumps(preds, indent=1))

    print("\nFitted params:")
    for k, v in fitted.items():
        print(f"  {k:14s} = {v:+.4e}", flush=True)
    print(f"\nTotal elapsed: {summary['elapsed_s']:.0f}s")

    # Plot 3-panel like v4
    by_vg1 = {}
    for p in preds:
        by_vg1.setdefault(p["VG1"], []).append(p)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
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
        f"P7v5: stage-wise + hard bounds\n"
        f"median log-RMSE = {median_rmse:.2f}  "
        f"(unfit 4.29 / v2 2.24 / v3 3.64 / v4 ?)  —  "
        f"elapsed {summary['elapsed_s']:.0f}s",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fit_curves.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT/'fit_curves.png'}")


if __name__ == "__main__":
    main()
