"""z75 — NS-RAM transient ramp vs Sebas measured Id-Vd curves.

Physically correct: ramp Vd slowly, integrate dVb/dt = I_total/C_body via
forward-Euler. Vb naturally tracks the stable body-KCL branch and jumps at
saddle-node bifurcations → reproduces NS-RAM avalanche/snapback.

100% differentiable (no scipy, no brentq) — the same pipeline P7 fitting
will use to backprop through measurements.
"""
from __future__ import annotations
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell import NSRAMCellConfig, transient
from nsram.bsim4_port.temp import compute_size_dep

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z75_nsram_transient_vs_sebas"
OUT.mkdir(parents=True, exist_ok=True)


def parse_vg2(filename):
    m = re.search(r"VG2=(-?\d+\.\d+)", filename)
    return float(m.group(1)) if m else None


def parse_vg1(dirname):
    m = re.search(r"VG1=([\d.]+)", dirname)
    return float(m.group(1)) if m else None


def load_curves():
    curves = []
    for d in sorted(DATA_DIR.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
        VG1 = parse_vg1(d.name)
        for f in sorted(d.glob("*.csv")):
            VG2 = parse_vg2(f.name)
            data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
            if data.ndim == 1:
                continue
            Vd = data[:, 0]
            Id = data[:, 1]
            half = len(Vd) // 2  # forward sweep only
            Vd, Id = Vd[:half], Id[:half]
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) < 5:
                continue
            curves.append({"VG1": VG1, "VG2": VG2, "Vd": Vd, "Id": Id})
    return curves


def transient_ids_curve(cfg, model, sd, VG1: float, VG2: float, Vd_arr: np.ndarray,
                        n_substeps: int = 80, dt: float = 5e-10) -> np.ndarray:
    """Ramp Vd through Vd_arr; settle Vb via body-KCL; return TOTAL drain current.

    Per Sebas's 2T schematic: drain node sees BSIM4 Ids + parasitic NPN Ic
    (collector is wired to the drain). The BJT's β·Ib amplification IS the
    snapback mechanism. Returning only MOSFET Ids misses the avalanche.
    """
    from nsram.bsim4_port.nsram_cell import kcl_body
    from nsram.bsim4_port.bjt import compute_bjt

    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)
    Vb = torch.tensor(0.0, dtype=torch.float64)

    Ids_out = np.zeros_like(Vd_arr)
    with torch.no_grad():
        for k, vd_v in enumerate(Vd_arr):
            vd = torch.tensor(float(vd_v), dtype=torch.float64)
            # Settle Vb via body-KCL transient
            for _ in range(n_substeps):
                f = kcl_body(cfg, Vb.unsqueeze(0), vd.unsqueeze(0),
                             VG1_t.unsqueeze(0), VG2_t.unsqueeze(0))
                dVb = f["dVb_dt"].squeeze(0)
                Vb = Vb + dt * torch.clamp(dVb, -1e10, 1e10)
                Vb = torch.clamp(Vb, -0.5, 0.8)

            # Total drain current = MOSFET Ids + BJT Ic
            # BJT: emitter=source(0), base=body, collector=drain
            #   Vbe = Vb - 0 = Vb;  Vbc = Vb - Vd
            r = compute_dc(model, sd,
                           Vgs=VG1_t.unsqueeze(0),
                           Vds=vd.unsqueeze(0),
                           Vbs=Vb.unsqueeze(0))
            Ids_mos = abs(float(r.Ids.item()))

            bjt_out = compute_bjt(cfg.bjt_params,
                                   Vbe=Vb.unsqueeze(0),
                                   Vbc=(Vb - vd).unsqueeze(0),
                                   T_K=273.15 + cfg.T_C)
            Ic_bjt = abs(float(bjt_out["Ic"].item()))

            Ids_out[k] = Ids_mos + Ic_bjt
    return Ids_out


def main():
    curves = load_curves()
    print(f"Loaded {len(curves)} curves")

    sebas_card = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(sebas_card)
    # Sebas's card has last-wins alpha0=0 / beta0=30 / agidl=1.99e-8 lines that
    # CLOBBER his hand-tuned values from the top of the card. SPICE behaves
    # the same way (so does the strict grid validation), but for NS-RAM
    # body-KCL these values matter — without alpha0>0 there's no impact-ion,
    # no body charging, no snapback. Restore Sebas's intended values:
    model.set("alpha0", 7.83756e-5)
    model.set("beta0", 18.0)
    model.set("agidl", 1.99e-8)
    # Per data/sebas_2026_04_22/2tnsram_simple.asc: Ln=0.18u, Wn=0.36u
    geom = Geometry(L=180e-9, W=360e-9)
    sd = compute_size_dep(model, geom, T_C=27.0)
    bjt = GummelPoonNPN.from_sebas_card()

    cfg = NSRAMCellConfig(
        bsim4_model=model, geometry=geom, bjt_params=bjt,
        Rb_leak=5e8, T_C=27.0, gamma_VG2=0.3,
    )

    by_vg1: dict[float, list] = {}
    for c in curves:
        by_vg1.setdefault(c["VG1"], []).append(c)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    rmses = []

    for ax, VG1 in zip(axes, sorted(by_vg1)):
        cs = sorted(by_vg1[VG1], key=lambda c: c["VG2"])
        cmap = plt.get_cmap("viridis")
        n = len(cs)
        print(f"\nVG1={VG1}:")
        for i, c in enumerate(cs):
            color = cmap(i / max(n - 1, 1))
            Vd = c["Vd"]
            Id_meas = np.abs(c["Id"])

            # Use the differentiable transient ramp
            Id_port = transient_ids_curve(cfg, model, sd, VG1, c["VG2"], Vd,
                                           n_substeps=80, dt=5e-10)

            ax.semilogy(Vd, Id_meas, "o", color=color, ms=4, alpha=0.7,
                         label=f"VG2={c['VG2']:+.2f}")
            ax.semilogy(Vd, np.maximum(Id_port, 1e-15), "-", color=color, lw=1.5, alpha=0.9)

            log_meas = np.log(np.maximum(Id_meas, 1e-15))
            log_port = np.log(np.maximum(Id_port, 1e-15))
            rmse = float(np.sqrt(np.mean((log_meas - log_port) ** 2)))
            rmses.append({"VG1": VG1, "VG2": c["VG2"], "log_rmse": rmse})
            print(f"  VG2={c['VG2']:+.2f}: log-RMSE={rmse:.2f}")

        ax.set_xlabel("Vd [V]")
        ax.set_title(f"VG1 = {VG1} V    ({n} curves)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=2)

    axes[0].set_ylabel("|Id| [A]")
    median_log = float(np.median([r["log_rmse"] for r in rmses]))
    fig.suptitle(
        f"NS-RAM transient (forward-Euler body-KCL, fully differentiable) "
        f"vs Sebas measured 130nm  —  median log-RMSE = {median_log:.2f}\n"
        f"dots = measured, lines = differentiable transient (no fit, no brentq)",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    out_path = OUT / "nsram_transient_vs_sebas.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_path}")
    print(f"\nMedian log-RMSE = {median_log:.3f}")


if __name__ == "__main__":
    main()
