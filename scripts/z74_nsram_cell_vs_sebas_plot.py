"""z74 — Plot NS-RAM cell prediction vs Sebas measured Id-Vd curves.

Physically correct comparison: for each (VG1, VG2, Vd), solve body-KCL for
steady-state Vb (I_total → 0), then evaluate channel Ids at (Vgs=VG1, Vds=Vd,
Vbs=Vb_steady). Compare to measured Id.

This is the NS-RAM operating-point inference: VG2 is a back-gate, not a body
terminal. body-KCL determines Vb implicitly.
"""
from __future__ import annotations
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brentq

torch.set_default_dtype(torch.float64)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell import NSRAMCellConfig, kcl_body
from nsram.bsim4_port.temp import compute_size_dep

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z74_nsram_cell_vs_sebas"
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
            half = len(Vd) // 2
            Vd, Id = Vd[:half], Id[:half]
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) < 5:
                continue
            curves.append({"VG1": VG1, "VG2": VG2, "Vd": Vd, "Id": Id})
    return curves


def solve_body_kcl(cfg, Vd_v, VG1_v, VG2_v):
    """Find Vb such that I_total(Vb) = 0. Returns Vb_steady."""
    def f(vb):
        Vb_t = torch.tensor([vb], dtype=torch.float64)
        Vd_t = torch.tensor([Vd_v], dtype=torch.float64)
        VG1_t = torch.tensor([VG1_v], dtype=torch.float64)
        VG2_t = torch.tensor([VG2_v], dtype=torch.float64)
        with torch.no_grad():
            r = kcl_body(cfg, Vb_t, Vd_t, VG1_t, VG2_t)
        return float(r["I_total"].item())

    try:
        # Body voltage typically lies in [-0.5, +0.7]V for NS-RAM
        return brentq(f, -0.5, 0.7, xtol=1e-6, maxiter=80)
    except (ValueError, RuntimeError):
        # Fallback: scan and pick smallest |residual|
        vbs = np.linspace(-0.4, 0.6, 25)
        residuals = [abs(f(v)) for v in vbs]
        return vbs[int(np.argmin(residuals))]


def predict_ids(model, sd, VG1_v, Vb_v, Vd_arr):
    Vd_t = torch.tensor(Vd_arr, dtype=torch.float64)
    Vg_t = torch.full_like(Vd_t, VG1_v)
    Vbs_t = torch.full_like(Vd_t, Vb_v)
    with torch.no_grad():
        r = compute_dc(model, sd, Vgs=Vg_t, Vds=Vd_t, Vbs=Vbs_t)
    return r.Ids.detach().cpu().numpy()


def main():
    curves = load_curves()
    print(f"Loaded {len(curves)} curves")

    sebas_card = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(sebas_card)
    geom = Geometry(L=130e-9, W=1e-6)
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

            # For each Vd, solve body-KCL for Vb, then evaluate Ids
            Id_port = np.zeros_like(Vd)
            for k, vd in enumerate(Vd):
                Vb_steady = solve_body_kcl(cfg, vd, VG1, c["VG2"])
                Ids_pred = predict_ids(model, sd, VG1, Vb_steady, np.array([vd]))
                Id_port[k] = abs(Ids_pred[0])

            ax.semilogy(Vd, Id_meas, "o", color=color, ms=4, alpha=0.7,
                         label=f"VG2={c['VG2']:+.2f}")
            ax.semilogy(Vd, Id_port, "-", color=color, lw=1.5, alpha=0.9)

            log_meas = np.log(np.maximum(Id_meas, 1e-15))
            log_port = np.log(np.maximum(Id_port, 1e-15))
            rmse = float(np.sqrt(np.mean((log_meas - log_port) ** 2)))
            rel = float(np.sqrt(np.mean(
                ((Id_port - Id_meas) / np.maximum(Id_meas, 1e-15)) ** 2)))
            rmses.append({"VG1": VG1, "VG2": c["VG2"],
                           "log_rmse": rmse, "rel_rmse": rel})
            print(f"  VG2={c['VG2']:+.2f}: log-RMSE={rmse:.2f} rel-RMSE={rel:.2f}")

        ax.set_xlabel("Vd [V]")
        ax.set_title(f"VG1 = {VG1} V    ({n} curves)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=2)

    axes[0].set_ylabel("|Id| [A]")
    median_log = float(np.median([r["log_rmse"] for r in rmses]))
    median_rel = float(np.median([r["rel_rmse"] for r in rmses]))
    fig.suptitle(
        f"NS-RAM cell (BSIM4 port + body-KCL) vs Sebas measured 130nm\n"
        f"dots = measured, lines = NS-RAM cell @ steady-state Vb (no fit). "
        f"Median log-RMSE = {median_log:.2f}",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    out_path = OUT / "nsram_cell_vs_sebas.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_path}")

    import json
    stats_path = OUT / "rmse_per_curve.json"
    stats_path.write_text(json.dumps({
        "median_log_rmse": median_log,
        "median_rel_rmse": median_rel,
        "n_curves": len(rmses),
        "per_curve": rmses,
    }, indent=2))
    print(f"Wrote {stats_path}")
    print(f"\nMedian log-RMSE = {median_log:.3f}")
    print(f"Median rel-RMSE = {median_rel:.3f}")


if __name__ == "__main__":
    main()
