"""z91l — extract Vth and DIBL on isolated M2, ngspice vs pyport.

A.5.b: z91j showed pyport disagrees with ngspice by ~1 dec on isolated
M2 (subthreshold under, near-on over — polarity flip). z91k showed
subthreshold-slope `n` matches (72.5 vs 76.5 mV/dec). So the bug is
likely in Vth (DIBL/SCE).

Method: at Vds ∈ {0.5, 2.0}, run Id-Vgs sweep, extract Vth via the
constant-current criterion: Vth = Vgs at Id = (W/L) × 1e-7 A (standard
SPICE convention). DIBL = (Vth_low − Vth_high) / ΔVds in V/V.

If pyport and ngspice agree at Vds=0.5 but diverge at Vds=2.0, the bug
is in our DIBL term (pdiblc1 / pdiblc2 / pdiblcb / drout / dsub).
"""
from __future__ import annotations
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z91l_vth_dibl"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91j_mod", ROOT / "scripts/z91j_ngspice_isolated_m2.py")
z91j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91j)

_spec_k = importlib.util.spec_from_file_location(
    "z91k_mod", ROOT / "scripts/z91k_subthreshold_slope.py")
z91k = importlib.util.module_from_spec(_spec_k)
_spec_k.loader.exec_module(z91k)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.model_card import BSIM4Model

DATA = ROOT / "data/sebas_2026_04_22"


def vth_constant_current(vgs: np.ndarray, Id: np.ndarray,
                          geom: Geometry) -> float:
    """Vth = Vgs at Id_target = (W/L) * 1e-7 A. Linear interpolation."""
    Id_target = (geom.W / geom.L) * 1e-7
    Id = np.maximum(np.abs(Id), 1e-30)
    if Id.max() < Id_target or Id.min() > Id_target:
        return float("nan")
    # find first crossing
    log_id = np.log10(Id); log_t = np.log10(Id_target)
    for i in range(len(vgs) - 1):
        if (log_id[i] - log_t) * (log_id[i + 1] - log_t) <= 0:
            f = (log_t - log_id[i]) / (log_id[i + 1] - log_id[i] + 1e-30)
            return float(vgs[i] + f * (vgs[i + 1] - vgs[i]))
    return float("nan")


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91j.z91f.patch_model_values(model, type_n=True)
    cfg = NSRAMCell2TConfig()
    geom = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    print(f"[z91l] M2 geom: L={geom.L:g}  W={geom.W:g}  Id_target={(geom.W/geom.L)*1e-7:.3e}")
    print(f"[z91l] DIBL params: pdiblc1={model.get('pdiblc1'):g}, "
          f"pdiblc2={model.get('pdiblc2'):g}, pdiblcb={model.get('pdiblcb'):g}, "
          f"drout={model.get('drout'):g}, dsub={model.get('dsub'):g}")

    vgs_arr = np.arange(0.0, 1.21, 0.025)
    res = {}
    for Vds in [0.05, 0.5, 2.0]:
        Id_ng = z91k.run_ngspice_id_vgs(Vds, geom, vgs_arr)
        Id_py = z91k.run_pyport_id_vgs(Vds, geom, model, vgs_arr)
        Vth_ng = vth_constant_current(vgs_arr, Id_ng, geom)
        Vth_py = vth_constant_current(vgs_arr, Id_py, geom)
        print(f"[z91l] Vds={Vds:>4.2f}V  Vth_ng={Vth_ng:+.4f}V  Vth_py={Vth_py:+.4f}V  "
              f"diff={Vth_py - Vth_ng:+.4f}V")
        res[f"vds_{Vds}"] = {
            "Vth_ngspice": Vth_ng,
            "Vth_pyport": Vth_py,
            "diff": (Vth_py - Vth_ng) if not (np.isnan(Vth_ng) or np.isnan(Vth_py)) else None,
        }

    if all(not np.isnan(res[k].get("diff") or float("nan")) for k in res):
        # DIBL = -dVth/dVds (V/V), should be POSITIVE for n-MOS
        v_low = res["vds_0.05"]
        v_high = res["vds_2.0"]
        DIBL_ng = -(v_high["Vth_ngspice"] - v_low["Vth_ngspice"]) / (2.0 - 0.05)
        DIBL_py = -(v_high["Vth_pyport"]   - v_low["Vth_pyport"])  / (2.0 - 0.05)
        print(f"[z91l] DIBL_ng = {DIBL_ng*1000:+.1f} mV/V")
        print(f"[z91l] DIBL_py = {DIBL_py*1000:+.1f} mV/V")
        print(f"[z91l] diff    = {(DIBL_py - DIBL_ng)*1000:+.1f} mV/V")
        res["DIBL_ngspice_mV_per_V"] = DIBL_ng * 1000
        res["DIBL_pyport_mV_per_V"]  = DIBL_py * 1000

    (OUT / "summary.json").write_text(json.dumps(res, indent=2))

    # Plot id-vgs at three Vds for both engines
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, Vds in zip(axes, [0.05, 0.5, 2.0]):
        Id_ng = z91k.run_ngspice_id_vgs(Vds, geom, vgs_arr)
        Id_py = z91k.run_pyport_id_vgs(Vds, geom, model, vgs_arr)
        ax.semilogy(vgs_arr, np.abs(Id_ng) + 1e-30, "k-", label="ngspice")
        ax.semilogy(vgs_arr, np.abs(Id_py) + 1e-30, "r--", label="pyport")
        Vth_ng = vth_constant_current(vgs_arr, Id_ng, geom)
        Vth_py = vth_constant_current(vgs_arr, Id_py, geom)
        ax.axvline(Vth_ng, color="k", ls=":", alpha=0.4)
        ax.axvline(Vth_py, color="r", ls=":", alpha=0.4)
        ax.set_title(f"Vds={Vds}V  Vth_ng={Vth_ng:.3f}  Vth_py={Vth_py:.3f}")
        ax.set_xlabel("Vgs"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_ylim(1e-15, 1e-3)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(f"z91l Vth/DIBL — isolated M2, body=GND")
    fig.tight_layout()
    fig.savefig(OUT / "vth_dibl.png", dpi=140)
    print(f"[z91l] saved {OUT}/vth_dibl.png + summary.json")


if __name__ == "__main__":
    main()
