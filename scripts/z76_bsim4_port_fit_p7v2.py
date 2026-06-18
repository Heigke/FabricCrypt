"""z76 — P7v2: parametric fit of BSIM4 port + NS-RAM body-KCL to Sebas's
measured 130nm Id-Vd snapback curves.

Frees ~14 parameters: GIDL suite (AGIDL/BGIDL/CGIDL/EGIDL + ISL counterparts),
impact-ion (alpha0/beta0), MOSFET baseline (vth0, u0, vsat), and cell-level
(gamma_VG2, Bf, Rb_leak, C_extra). Uses fully-differentiable transient ramp
(forward-Euler body-KCL, no brentq) and log-RMSE loss across all 33 curves.

Pipeline:
  1. Load 33 (VG1, VG2, Vd, Id_meas) curves
  2. Per curve: ramp Vd via transient with current params, get Id_pred = Ids_MOS + Ic_BJT
  3. log-RMSE loss aggregated over curves
  4. Adam → L-BFGS

Saves: fitted params, per-curve RMSE, overlay plot.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
OUT = ROOT / "results/z76_bsim4_port_fit_p7v2"
OUT.mkdir(parents=True, exist_ok=True)


# ---- Data loading -----------------------------------------------------------

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
            # Subsample to ~10 points per curve for speed (autograd graph size)
            if len(Vd) > 10:
                idx = np.linspace(0, len(Vd) - 1, 10).astype(int)
                Vd, Id = Vd[idx], Id[idx]
            if len(Vd) < 5:
                continue
            curves.append({"VG1": VG1, "VG2": VG2,
                           "Vd": torch.tensor(Vd, dtype=torch.float64),
                           "Id": torch.tensor(Id, dtype=torch.float64)})
    return curves


# ---- Fit params -------------------------------------------------------------

# Each entry: (name, init_value, scale_kind)
#   scale_kind 'lin'  → leaf value is the parameter directly
#   scale_kind 'log'  → leaf is log10(value); init_value is log10
FIT_SPEC = [
    ("vth0",     0.54,        "lin"),    # threshold
    ("u0",       np.log10(0.048), "log"),  # mobility
    ("vsat",     np.log10(1.35e5),"log"),  # sat velocity
    ("alpha0",   np.log10(7.84e-5),"log"), # impact-ion pre-exp
    ("beta0",    np.log10(18.0), "log"),  # impact-ion exponent V (lower = trigger at lower Vd)
    ("agidl",    np.log10(1.99e-8),"log"), # GIDL pre-exp
    ("bgidl",    np.log10(2.3e9), "log"), # GIDL exponent
    ("cgidl",    np.log10(0.5),   "log"),  # GIDL Vdb modulation
    ("egidl",    0.4,           "lin"),   # GIDL Vd-Vg threshold (default 0.8 too high)
    ("agisl",    np.log10(1.99e-8),"log"), # GISL pre-exp (was 0)
    ("bgisl",    np.log10(2.3e9), "log"),
    ("cgisl",    np.log10(0.5),   "log"),
    ("egisl",    0.4,           "lin"),
]
# Cell-level params (NSRAMCellConfig)
CELL_SPEC = [
    ("gamma_VG2", 0.5,           "lin"),   # back-gate Vth modulation
    ("log_Rb_leak", np.log10(5e8), "log"),  # external leak
    ("log_C_extra", np.log10(1e-15), "log"),# extra body cap (init 1fF)
    ("log_Bf",      np.log10(10000.0),"log"),# BJT forward beta
]


# ---- Forward model -----------------------------------------------------------

def build_model(fit_params: dict, base_card_text: str) -> BSIM4Model:
    """Apply fitable params to a fresh BSIM4 model card."""
    m = BSIM4Model.from_spice(base_card_text)
    for name, _, kind in FIT_SPEC:
        leaf = fit_params[name]
        v = (10.0 ** leaf) if kind == "log" else leaf
        m._values[name] = v
        m._given.add(name)
    return m


def forward_one_curve(fit_params, base_card_text, geom, VG1: float, VG2: float,
                      Vd_seq: torch.Tensor, n_substeps: int = 5,
                      dt: float = 1e-9) -> torch.Tensor:
    """Differentiable forward: ramp Vd through Vd_seq, return Id_pred per point.

    Body-KCL transient settle at each Vd (n_substeps Euler steps), then read
    Id = MOSFET Ids + BJT Ic.
    """
    model = build_model(fit_params, base_card_text)
    sd = compute_size_dep(model, geom, T_C=27.0)

    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 10.0 ** fit_params["log_Bf"]

    gamma_VG2 = fit_params["gamma_VG2"]
    Rb_leak = 10.0 ** fit_params["log_Rb_leak"]
    C_extra = 10.0 ** fit_params["log_C_extra"]

    # VG2 → vth0 modulation
    # VG2 sign convention (FIXED per GPT review 2026-04-29):
    # wrapper uses vth0_eff = vth0 + gamma*VG2; we now match.
    # Old fit scripts used minus sign — fitted gammas need negation if loaded.
    sd.vth0_T = sd.vth0_T + gamma_VG2 * VG2
    sd_shadow = sd

    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)
    Vb = torch.tensor(0.0, dtype=torch.float64)

    # Junction geometry defaults
    W, L = geom.W, geom.L
    As_v = W * L; Ad_v = W * L
    Ps_v = 2.0 * (W + L); Pd_v = 2.0 * (W + L)

    Id_preds = []
    for k in range(int(Vd_seq.shape[0])):
        vd = Vd_seq[k]
        Vbs_t = Vb.unsqueeze(0)
        Vd_t = vd.unsqueeze(0)
        Vg_t = VG1_t.unsqueeze(0)

        # Settle Vb via body-KCL
        for _ in range(n_substeps):
            Vbd_t = (Vb - vd).unsqueeze(0)
            r = compute_dc(model, sd_shadow, Vgs=Vg_t, Vds=Vd_t, Vbs=Vbs_t)
            Iii = compute_iimpact(model, sd_shadow, r, Vds=Vd_t).squeeze(0)
            Igidl, Igisl = compute_igidl_gisl(model, sd_shadow, Vgs=Vg_t,
                                               Vds=Vd_t, Vbs=Vbs_t)
            Igidl, Igisl = Igidl.squeeze(0), Igisl.squeeze(0)
            Igb = compute_igb(model, sd_shadow, Vgs=Vg_t, Vbs=Vbs_t,
                               dc_result=r).squeeze(0)
            Ibs, Ibd = compute_body_diodes(model, sd_shadow, Vbs=Vbs_t, Vbd=Vbd_t,
                                            As=As_v, Ad=Ad_v, Ps=Ps_v, Pd=Pd_v)
            Ibs, Ibd = Ibs.squeeze(0), Ibd.squeeze(0)
            cap = compute_caps(model, sd_shadow, r, Vgs=Vg_t, Vds=Vd_t,
                                Vbs=Vbs_t, Vbd=Vbd_t,
                                As=As_v, Ad=Ad_v, Ps=Ps_v, Pd=Pd_v)
            bjt_out = compute_bjt(bjt, Vbe=Vb.unsqueeze(0),
                                   Vbc=(Vb - vd).unsqueeze(0), T_K=300.15)
            Ib = bjt_out["Ib"].squeeze(0)
            Ileak = Vb / Rb_leak
            I_total = (Iii - Ibd - Ibs + Igidl + Igisl + Igb - Ib - Ileak)
            C_body = (cap.Cjs.squeeze(0) + cap.Cjd.squeeze(0)
                      + torch.abs(cap.Cgb.squeeze(0)) + C_extra + 1e-30)
            dVb = I_total / C_body
            dVb_clamped = torch.clamp(dVb, -1e10, 1e10)
            Vb = Vb + dt * dVb_clamped
            Vb = torch.clamp(Vb, -0.5, 0.8)
            Vbs_t = Vb.unsqueeze(0)

        # Read total drain current
        r_final = compute_dc(model, sd_shadow, Vgs=Vg_t, Vds=Vd_t, Vbs=Vbs_t)
        Ids_mos = r_final.Ids.squeeze(0).abs()
        bjt_final = compute_bjt(bjt, Vbe=Vb.unsqueeze(0),
                                 Vbc=(Vb - vd).unsqueeze(0), T_K=300.15)
        Ic = bjt_final["Ic"].squeeze(0).abs()
        Id_preds.append(Ids_mos + Ic)

    return torch.stack(Id_preds)


# ---- Loss + train -----------------------------------------------------------

def loss_fn(fit_params, base_card, geom, curves):
    log_eps = 1e-15
    losses = []
    for c in curves:
        Id_pred = forward_one_curve(fit_params, base_card, geom,
                                     c["VG1"], c["VG2"], c["Vd"],
                                     n_substeps=2, dt=5e-9)
        Id_meas = c["Id"]
        log_pred = torch.log(Id_pred.clamp_min(log_eps))
        log_meas = torch.log(Id_meas.clamp_min(log_eps))
        # Per-curve log-RMSE (scaled to make it order 1)
        l = ((log_pred - log_meas) ** 2).mean()
        losses.append(l)
    return torch.stack(losses).mean()


def train():
    t0 = time.time()
    base_card = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    geom = Geometry(L=180e-9, W=360e-9)
    curves = load_curves()
    print(f"Loaded {len(curves)} curves at {time.time()-t0:.1f}s")

    # Build fit params as torch leaves
    fit_params = {}
    for name, init, _ in FIT_SPEC:
        fit_params[name] = torch.tensor(init, dtype=torch.float64,
                                         requires_grad=True)
    for name, init, _ in CELL_SPEC:
        fit_params[name] = torch.tensor(init, dtype=torch.float64,
                                         requires_grad=True)

    # Initial loss
    with torch.no_grad():
        l0 = loss_fn(fit_params, base_card, geom, curves)
    print(f"Initial loss = {l0.item():.4f}  ({time.time()-t0:.1f}s)")

    # Adam
    opt = torch.optim.Adam(list(fit_params.values()), lr=0.05)
    history = {"iter": [], "loss": []}
    N_ADAM = 40
    for it in range(N_ADAM):
        opt.zero_grad()
        l = loss_fn(fit_params, base_card, geom, curves)
        l.backward()
        # Gradient clip
        torch.nn.utils.clip_grad_norm_(list(fit_params.values()), max_norm=2.0)
        opt.step()
        if it % 5 == 0 or it == N_ADAM - 1:
            history["iter"].append(it)
            history["loss"].append(float(l.item()))
            elapsed = time.time() - t0
            print(f"  Adam it={it:3d}  loss={l.item():.4f}  "
                  f"({elapsed:.0f}s)", flush=True)

    # Final eval
    with torch.no_grad():
        lf = loss_fn(fit_params, base_card, geom, curves)
    print(f"\nFinal loss after Adam = {lf.item():.4f}")

    # Save fitted values
    fitted = {}
    for name, _, kind in FIT_SPEC + CELL_SPEC:
        leaf = float(fit_params[name].detach().item())
        if kind == "log":
            fitted[name] = float(10.0 ** leaf)
        else:
            fitted[name] = leaf

    # Per-curve RMSE
    per_curve = []
    with torch.no_grad():
        for c in curves:
            Id_pred = forward_one_curve(fit_params, base_card, geom,
                                         c["VG1"], c["VG2"], c["Vd"],
                                         n_substeps=10, dt=1e-9)
            log_p = torch.log(Id_pred.clamp_min(1e-15))
            log_m = torch.log(c["Id"].clamp_min(1e-15))
            rmse = float(torch.sqrt(((log_p - log_m) ** 2).mean()).item())
            per_curve.append({"VG1": c["VG1"], "VG2": c["VG2"],
                              "log_rmse": rmse,
                              "Id_pred": Id_pred.detach().numpy().tolist(),
                              "Id_meas": c["Id"].numpy().tolist(),
                              "Vd": c["Vd"].numpy().tolist()})

    summary = {
        "init_loss": float(l0.item()),
        "final_loss": float(lf.item()),
        "fitted_params": fitted,
        "history": history,
        "median_log_rmse": float(np.median([p["log_rmse"] for p in per_curve])),
        "elapsed_s": time.time() - t0,
        "n_curves": len(curves),
    }
    (OUT / "summary.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k not in ("per_curve",)},
        indent=2, default=str))
    (OUT / "per_curve.json").write_text(json.dumps(per_curve, indent=1))

    print(f"\nMedian log-RMSE = {summary['median_log_rmse']:.3f}  "
          f"(was 4.29 with no fit, 3.10 with alpha0 restored only)")
    print(f"Elapsed: {summary['elapsed_s']:.0f}s")
    print("Fitted params:")
    for k, v in fitted.items():
        print(f"  {k:14s} = {v:+.4e}")

    # Plot
    by_vg1 = {}
    for p in per_curve:
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
        f"P7v2: BSIM4 port + body-KCL transient FIT to Sebas 130nm\n"
        f"median log-RMSE = {summary['median_log_rmse']:.2f}  "
        f"(initial loss {summary['init_loss']:.2f} → final {summary['final_loss']:.2f}, "
        f"{N_ADAM} Adam iters, {summary['elapsed_s']:.0f}s)",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fit_curves.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'fit_curves.png'}")


if __name__ == "__main__":
    train()
