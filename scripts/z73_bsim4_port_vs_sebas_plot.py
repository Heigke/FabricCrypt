"""z73 — Plot port Ids vs Sebas measured Id-Vd curves at all (VG1, VG2).

Uses the BSIM4 port AS-IS (no fitting), with VG2 mapped to Vbs (NS-RAM
convention approximation). Overlay on log-y so subthreshold + saturation
both visible.
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

from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.temp import compute_size_dep

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z73_bsim4_port_vs_sebas"
OUT.mkdir(parents=True, exist_ok=True)


def parse_vg2(filename: str) -> float:
    m = re.search(r"VG2=(-?\d+\.\d+)", filename)
    return float(m.group(1)) if m else None


def parse_vg1(dirname: str) -> float:
    m = re.search(r"VG1=([\d.]+)", dirname)
    return float(m.group(1)) if m else None


def load_curves() -> list[dict]:
    """Returns list of {VG1, VG2, Vd, Id} dicts."""
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
            # Take forward-sweep half only; drop near-zero noise floor
            n = len(Vd)
            half = n // 2
            Vd = Vd[:half]
            Id = Id[:half]
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd = Vd[mask]
            Id = Id[mask]
            if len(Vd) < 5:
                continue
            curves.append({"VG1": VG1, "VG2": VG2, "Vd": Vd, "Id": Id, "file": f.name})
    return curves


def port_predict(model: BSIM4Model, sd, VG1: float, VG2: float, Vd: np.ndarray):
    """Forward pass through BSIM4 port. VG2 → Vbs approximation."""
    Vd_t = torch.tensor(Vd, dtype=torch.float64)
    Vg_t = torch.full_like(Vd_t, VG1)
    Vbs_t = torch.full_like(Vd_t, VG2)
    with torch.no_grad():
        r = compute_dc(model, sd, Vgs=Vg_t, Vds=Vd_t, Vbs=Vbs_t)
    return r.Ids.detach().cpu().numpy()


def main():
    curves = load_curves()
    print(f"Loaded {len(curves)} curves.")

    sebas_card = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(sebas_card)
    # CORRECTED GEOMETRY: per data/sebas_2026_04_22/2tnsram_simple.asc:
    #   .param Ln=0.18u  Wn=0.36u  → device M1 is L=180nm, W=360nm
    geom = Geometry(L=180e-9, W=360e-9)
    sd = compute_size_dep(model, geom, T_C=27.0)

    # Group by VG1
    by_vg1: dict[float, list] = {}
    for c in curves:
        by_vg1.setdefault(c["VG1"], []).append(c)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    rmses = []

    for ax, VG1 in zip(axes, sorted(by_vg1)):
        cs = sorted(by_vg1[VG1], key=lambda c: c["VG2"])
        cmap = plt.get_cmap("viridis")
        n = len(cs)
        for i, c in enumerate(cs):
            color = cmap(i / max(n - 1, 1))
            Vd = c["Vd"]
            Id_meas = np.abs(c["Id"])
            Id_port = np.abs(port_predict(model, sd, VG1, c["VG2"], Vd))
            ax.semilogy(Vd, Id_meas, "o", color=color, ms=4, alpha=0.7,
                         label=f"VG2={c['VG2']:+.2f}")
            ax.semilogy(Vd, Id_port, "-", color=color, lw=1.5, alpha=0.9)
            # log-RMSE for stats
            log_meas = np.log(np.maximum(Id_meas, 1e-15))
            log_port = np.log(np.maximum(Id_port, 1e-15))
            rmse = float(np.sqrt(np.mean((log_meas - log_port) ** 2)))
            rmses.append({"VG1": VG1, "VG2": c["VG2"], "log_rmse": rmse,
                           "rel_rmse": float(np.sqrt(np.mean(
                               ((Id_port - Id_meas) / np.maximum(Id_meas, 1e-15)) ** 2
                           )))})
        ax.set_xlabel("Vd [V]")
        ax.set_title(f"VG1 = {VG1} V    ({n} curves)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=2)

    axes[0].set_ylabel("|Id| [A]")
    median_log_rmse = float(np.median([r["log_rmse"] for r in rmses]))
    median_rel_rmse = float(np.median([r["rel_rmse"] for r in rmses]))
    fig.suptitle(
        f"BSIM4 Python port (post-Wave-2) vs Sebas measured 130nm\n"
        f"dots = measured, lines = port (no fit). "
        f"Median log-RMSE = {median_log_rmse:.2f}, rel-RMSE = {median_rel_rmse:.2f}",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    out_path = OUT / "port_vs_sebas.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Stats CSV
    import json
    stats_path = OUT / "rmse_per_curve.json"
    stats_path.write_text(json.dumps({
        "median_log_rmse": median_log_rmse,
        "median_rel_rmse": median_rel_rmse,
        "n_curves": len(rmses),
        "per_curve": rmses,
    }, indent=2))
    print(f"Wrote {stats_path}")
    print(f"\nSummary across {len(rmses)} curves:")
    print(f"  median log-RMSE: {median_log_rmse:.3f}")
    print(f"  median rel-RMSE: {median_rel_rmse:.3f}")
    print(f"  worst curve:     log-RMSE={max(r['log_rmse'] for r in rmses):.3f}")
    print(f"  best curve:      log-RMSE={min(r['log_rmse'] for r in rmses):.3f}")


if __name__ == "__main__":
    main()
