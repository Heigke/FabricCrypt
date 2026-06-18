"""z78 — P7v4: Vb-coupled punch-through + balanced loss + multi-start + L-BFGS.

Improvements over P7v3:
  1. Punch-through is now Vb-coupled: I_PT = I_PT0·scale·softplus((Vd + k_Vb·Vb − V_PT_th)/scale).
     This gives VG2 (via Vb steady-state) actual leverage on the snapback voltage.
  2. Balanced loss: 50% absolute log-MSE + 50% knee-weighted log-MSE.
     Prevents the optimizer from collapsing gamma_VG2 to 0 the way v3 did.
  3. Adds k_Vb_PT to the fitable params (21 params total).
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
OUT = ROOT / "results/z78_bsim4_port_fit_p7v4"
OUT.mkdir(parents=True, exist_ok=True)


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


# Fit spec: (name, init, scale_kind)
FIT_SPEC = [
    ("vth0",     0.54,             "lin"),
    ("u0",       0.048,            "log"),
    ("vsat",     1.35e5,           "log"),
    ("alpha0",   7.84e-5,          "log"),
    ("beta0",    18.0,             "log"),
    ("agidl",    1.99e-8,          "log"),
    ("bgidl",    2.3e9,            "log"),
    ("cgidl",    0.5,              "log"),
    ("egidl",    0.4,              "lin"),
    ("agisl",    1.99e-8,          "log"),
    ("bgisl",    2.3e9,            "log"),
    ("cgisl",    0.5,              "log"),
    ("egisl",    0.4,              "lin"),
]
CELL_SPEC = [
    ("gamma_VG2", 0.5,             "lin"),
    ("Rb_leak",   5e8,             "log"),
    ("C_extra",   1e-15,           "log"),
    ("Bf",        10000.0,         "log"),
    # Punch-through
    ("I_PT0",     1e-6,            "log"),
    ("V_PT_th",   0.95,            "lin"),
    ("V_PT_scale",0.02,            "log"),
    ("k_Vb_PT",   1.5,             "lin"),  # NEW: Vb coupling strength
]


def make_leaves(seed: int = 0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, init, kind in FIT_SPEC + CELL_SPEC:
        if kind == "log":
            jitter = 1.0 if seed == 0 else float(np.exp(rng.normal(0, 0.3)))
            v0 = float(np.log10(init * jitter))
        else:
            jitter = 0.0 if seed == 0 else float(rng.normal(0, 0.05))
            v0 = float(init + jitter)
        out[name] = torch.tensor(v0, dtype=torch.float64, requires_grad=True)
    return out


def leaf_to_value(leaves, name, kind):
    if kind == "log":
        return 10.0 ** leaves[name]
    return leaves[name]


def build_model(leaves, base_card_text):
    m = BSIM4Model.from_spice(base_card_text)
    for name, _, kind in FIT_SPEC:
        v = leaf_to_value(leaves, name, kind)
        m._values[name] = v if isinstance(v, torch.Tensor) else float(v)
        m._given.add(name)
    return m


def forward_curve(leaves, base_card, geom, VG1: float, VG2: float,
                   Vd_seq: torch.Tensor, n_substeps: int = 2,
                   dt: float = 5e-9) -> torch.Tensor:
    model = build_model(leaves, base_card)
    sd = compute_size_dep(model, geom, T_C=27.0)
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = leaf_to_value(leaves, "Bf", "log")

    gamma = leaves["gamma_VG2"]
    Rb_leak = leaf_to_value(leaves, "Rb_leak", "log")
    C_extra = leaf_to_value(leaves, "C_extra", "log")
    I_PT0 = leaf_to_value(leaves, "I_PT0", "log")
    V_PT_th = leaves["V_PT_th"]
    V_PT_scale = leaf_to_value(leaves, "V_PT_scale", "log")
    k_Vb_PT = leaves["k_Vb_PT"]

    # VG2 sign convention (FIXED per GPT review 2026-04-29):
    # wrapper uses vth0_eff = vth0 + gamma*VG2; we now match.
    # Old fit scripts used minus sign — fitted gammas need negation if loaded.
    sd.vth0_T = sd.vth0_T + gamma * VG2  # back-gate Vth shift

    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    Vb = torch.tensor(0.0, dtype=torch.float64)
    W, L = geom.W, geom.L
    As_v = W * L; Ad_v = W * L
    Ps_v = 2.0*(W+L); Pd_v = 2.0*(W+L)

    Id_preds = []
    for k in range(int(Vd_seq.shape[0])):
        vd = Vd_seq[k]
        Vbs_t = Vb.unsqueeze(0)
        Vd_t = vd.unsqueeze(0)
        Vg_t = VG1_t.unsqueeze(0)

        for _ in range(n_substeps):
            Vbd_t = (Vb - vd).unsqueeze(0)
            r = compute_dc(model, sd, Vgs=Vg_t, Vds=Vd_t, Vbs=Vbs_t)
            Iii = compute_iimpact(model, sd, r, Vds=Vd_t).squeeze(0)
            Igidl, Igisl = compute_igidl_gisl(model, sd, Vgs=Vg_t,
                                               Vds=Vd_t, Vbs=Vbs_t)
            Igidl, Igisl = Igidl.squeeze(0), Igisl.squeeze(0)
            Igb = compute_igb(model, sd, Vgs=Vg_t, Vbs=Vbs_t,
                              dc_result=r).squeeze(0)
            Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs_t, Vbd=Vbd_t,
                                            As=As_v, Ad=Ad_v, Ps=Ps_v, Pd=Pd_v)
            Ibs, Ibd = Ibs.squeeze(0), Ibd.squeeze(0)
            cap = compute_caps(model, sd, r, Vgs=Vg_t, Vds=Vd_t,
                                Vbs=Vbs_t, Vbd=Vbd_t,
                                As=As_v, Ad=Ad_v, Ps=Ps_v, Pd=Pd_v)
            bjt_out = compute_bjt(bjt, Vbe=Vb.unsqueeze(0),
                                   Vbc=(Vb - vd).unsqueeze(0), T_K=300.15)
            Ib = bjt_out["Ib"].squeeze(0)
            Ileak = Vb / Rb_leak
            # Vb-coupled punch-through (NEW in v4)
            scale_safe = (V_PT_scale.clamp_min(1e-6) if isinstance(V_PT_scale, torch.Tensor)
                          else max(float(V_PT_scale), 1e-6))
            drive = (vd + k_Vb_PT * Vb - V_PT_th) / scale_safe
            I_PT = I_PT0 * V_PT_scale * F.softplus(drive)
            I_total = (Iii - Ibd - Ibs + Igidl + Igisl + Igb - Ib - Ileak + I_PT)
            C_body = (cap.Cjs.squeeze(0) + cap.Cjd.squeeze(0)
                      + torch.abs(cap.Cgb.squeeze(0)) + C_extra + 1e-30)
            dVb = I_total / C_body
            Vb = Vb + dt * torch.clamp(dVb, -1e10, 1e10)
            Vb = torch.clamp(Vb, -0.5, 0.8)
            Vbs_t = Vb.unsqueeze(0)

        # Total drain current = MOSFET Ids + BJT Ic + I_PT contribution
        # (I_PT physically flows through drain too, since punch-through carriers
        # transit drain→source through the bulk path. Add it.)
        r_final = compute_dc(model, sd, Vgs=Vg_t, Vds=Vd_t, Vbs=Vbs_t)
        Ids_mos = r_final.Ids.squeeze(0).abs()
        bjt_final = compute_bjt(bjt, Vbe=Vb.unsqueeze(0),
                                 Vbc=(Vb - vd).unsqueeze(0), T_K=300.15)
        Ic = bjt_final["Ic"].squeeze(0).abs()
        # Drain-side punch-through current (same form as body-charging)
        scale_safe = (V_PT_scale.clamp_min(1e-6) if isinstance(V_PT_scale, torch.Tensor)
                      else max(float(V_PT_scale), 1e-6))
        drive = (vd + k_Vb_PT * Vb - V_PT_th) / scale_safe
        I_PT_drain = I_PT0 * V_PT_scale * F.softplus(drive)
        Id_preds.append(Ids_mos + Ic + I_PT_drain)
    return torch.stack(Id_preds)


def balanced_loss(leaves, base_card, geom, curves):
    """50% absolute log-MSE + 50% knee-weighted log-MSE."""
    log_eps = 1e-15
    losses = []
    for c in curves:
        Id_pred = forward_curve(leaves, base_card, geom,
                                 c["VG1"], c["VG2"], c["Vd"])
        log_pred = torch.log(Id_pred.clamp_min(log_eps))
        log_meas = torch.log(c["Id"].clamp_min(log_eps))
        l_abs = ((log_pred - log_meas) ** 2).mean()
        # Knee weight
        w = torch.zeros_like(log_meas)
        w[:-1] = (log_meas[1:] - log_meas[:-1]).abs()
        w[-1] = w[-2]
        w = (w + 0.1) / (w.mean() + 0.1)
        l_knee = ((log_pred - log_meas) ** 2 * w).mean()
        losses.append(0.5 * l_abs + 0.5 * l_knee)
    return torch.stack(losses).mean()


def evaluate(leaves, base_card, geom, curves):
    log_eps = 1e-15
    rmses = []; preds = []
    for c in curves:
        with torch.no_grad():
            Id_pred = forward_curve(leaves, base_card, geom,
                                     c["VG1"], c["VG2"], c["Vd"])
        log_p = torch.log(Id_pred.clamp_min(log_eps))
        log_m = torch.log(c["Id"].clamp_min(log_eps))
        rmse = float(torch.sqrt(((log_p - log_m) ** 2).mean()).item())
        rmses.append(rmse)
        preds.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse": rmse,
                      "Vd": c["Vd"].numpy().tolist(),
                      "Id_meas": c["Id"].numpy().tolist(),
                      "Id_pred": Id_pred.detach().numpy().tolist()})
    return float(np.median(rmses)), preds


def train_one(seed, base_card, geom, curves, t0):
    print(f"\n=== seed={seed} ===", flush=True)
    leaves = make_leaves(seed)
    with torch.no_grad():
        l0 = balanced_loss(leaves, base_card, geom, curves)
    print(f"  Init loss = {l0.item():.4f}", flush=True)

    opt = torch.optim.Adam(list(leaves.values()), lr=0.05)
    N_ADAM = 50
    for it in range(N_ADAM):
        opt.zero_grad()
        l = balanced_loss(leaves, base_card, geom, curves)
        l.backward()
        torch.nn.utils.clip_grad_norm_(list(leaves.values()), max_norm=2.0)
        opt.step()
        if it % 10 == 0 or it == N_ADAM - 1:
            print(f"  s{seed} Adam {it}: {l.item():.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)

    opt2 = torch.optim.LBFGS(list(leaves.values()), max_iter=20, lr=0.5,
                              line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad()
        l = balanced_loss(leaves, base_card, geom, curves)
        l.backward()
        return l
    try:
        opt2.step(closure)
    except RuntimeError as e:
        print(f"  s{seed} L-BFGS warn: {e}", flush=True)

    with torch.no_grad():
        lf = balanced_loss(leaves, base_card, geom, curves)
    print(f"  s{seed} final = {lf.item():.4f}", flush=True)
    return float(lf.item()), leaves


def main():
    t0 = time.time()
    base_card = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    geom = Geometry(L=180e-9, W=360e-9)
    curves = load_curves()
    print(f"Loaded {len(curves)} curves at {time.time()-t0:.1f}s", flush=True)

    best_loss = float("inf")
    best_leaves = None
    for seed in [0, 1, 2]:
        loss, leaves = train_one(seed, base_card, geom, curves, t0)
        if loss < best_loss:
            best_loss = loss
            best_leaves = {k: v.detach().clone().requires_grad_(True)
                            for k, v in leaves.items()}
            print(f"  ** new best: seed={seed} loss={loss:.4f}", flush=True)

    print(f"\n=== Best loss = {best_loss:.4f} ===", flush=True)
    median_rmse, preds = evaluate(best_leaves, base_card, geom, curves)
    print(f"Median log-RMSE = {median_rmse:.3f}  "
          f"(was 4.29 unfit, 2.24 v2, 3.64 v3)", flush=True)

    fitted = {}
    for name, _, kind in FIT_SPEC + CELL_SPEC:
        leaf = float(best_leaves[name].detach().item())
        fitted[name] = float(10.0 ** leaf) if kind == "log" else leaf
    summary = {
        "best_loss": best_loss,
        "median_log_rmse": median_rmse,
        "fitted_params": fitted,
        "elapsed_s": time.time() - t0,
        "n_curves": len(curves),
        "config": "Vb-coupled-PT + balanced-loss + multistart + L-BFGS",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "per_curve.json").write_text(json.dumps(preds, indent=1))

    print("\nFitted params:")
    for k, v in fitted.items():
        print(f"  {k:14s} = {v:+.4e}", flush=True)
    print(f"\nTotal elapsed: {summary['elapsed_s']:.0f}s")

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
        f"P7v4: Vb-coupled punch-through + balanced loss\n"
        f"median log-RMSE = {median_rmse:.2f}  "
        f"(4.29 unfit / 2.24 v2 / 3.64 v3)  —  elapsed {summary['elapsed_s']:.0f}s",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fit_curves.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT/'fit_curves.png'}")


if __name__ == "__main__":
    main()
