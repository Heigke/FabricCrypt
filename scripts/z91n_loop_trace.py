"""z91n — internal-state trace at one bottleneck bias (VG1=0.4, VG2=0).

A.3.c: VG1=0.4 row mean RMSE 1.95 dec — snap doesn't fire there even
with mbjt=1.0. Trace Vb(Vd), Vsint(Vd), Iii(Vd), Ic_Q1(Vd), I_well_body
to identify which loop term is undersized.

Compare to a working VG1=0.6 curve (mean 0.77 dec, snap fires) to
see what's different.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z91n_loop_trace"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(z91f)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry


def trace_one_bias(VG1: float, VG2: float, model_M1, model_M2,
                    sd_M1, sd_M2, cfg, sebas_rows):
    sebas_row = z91f.find_params(sebas_rows, VG1, VG2)
    P_M1, P_M2 = z91f.make_overrides(sebas_row)
    if P_M2:
        for k in ("k1", "k2", "etab", "beta0"):
            P_M2.pop(k, None)
        if not P_M2: P_M2 = None
    bjt = z91f.make_bjt(sebas_row)
    bjt.Bf = 5e4
    mbjt = float(sebas_row.get("mbjt", 1.0)) if sebas_row else 1.0
    if np.isnan(mbjt): mbjt = 1.0
    cfg.vnwell_mbjt = mbjt
    Vd_seq = torch.tensor(np.linspace(0.05, 2.0, 40), dtype=torch.float64)
    with torch.no_grad(), \
         z91f.patch_sd_scaled(sd_M1, P_M1), \
         z91f.patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t_arclength_grad(
            cfg, model_M1=model_M1, model_M2=model_M2, bjt=bjt,
            Vd_seq=Vd_seq, VG1=torch.tensor(VG1), VG2=torch.tensor(VG2))
    return out


def main():
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91f.patch_model_values(model_M1, type_n=True)
    z91f.patch_model_values(model_M2, type_n=True)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    cfg.vnwell_Rs = 1e11
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2, Geometry(L=cfg.Ln*cfg.M2_length_factor, W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1; cfg._sd_M2 = sd_M2
    sebas_rows = z91f.load_sebas_params()

    biases = [(0.4, 0.0), (0.6, 0.0)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    summary = {}
    for col, (vg1, vg2) in enumerate(biases):
        out = trace_one_bias(vg1, vg2, model_M1, model_M2, sd_M1, sd_M2, cfg, sebas_rows)
        Vd = out["Vd"].numpy() if "Vd" in out else np.linspace(0.05, 2.0, 40)
        Vb = out["Vb"].numpy()
        Vsint = out["Vsint"].numpy()
        Id = out["Id"].abs().numpy()
        # Try to extract components
        comps = out.get("components", None) or {}
        Iii_M1 = comps.get("Iii_M1", torch.zeros_like(out["Vb"])).numpy() if hasattr(comps.get("Iii_M1", None), "numpy") else np.array(comps.get("Iii_M1", [0]*len(Vd)))
        Ic_Q1 = comps.get("Ic_Q1", torch.zeros_like(out["Vb"])).numpy() if hasattr(comps.get("Ic_Q1", None), "numpy") else np.array(comps.get("Ic_Q1", [0]*len(Vd)))
        I_well = comps.get("I_well_body", torch.zeros_like(out["Vb"])).numpy() if hasattr(comps.get("I_well_body", None), "numpy") else np.array(comps.get("I_well_body", [0]*len(Vd)))

        ax = axes[0, col]; ax.plot(Vd, Vb, label="Vb"); ax.plot(Vd, Vsint, label="Vsint")
        ax.set_title(f"VG1={vg1}, VG2={vg2}  internal V"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlabel("Vd")

        ax = axes[1, col]; ax.semilogy(Vd, np.abs(Iii_M1)+1e-30, label="|Iii_M1|")
        ax.semilogy(Vd, np.abs(Ic_Q1)+1e-30, label="|Ic_Q1|")
        ax.semilogy(Vd, np.abs(I_well)+1e-30, label="|I_well|")
        ax.semilogy(Vd, Id, "k--", label="|Id|")
        ax.set_title("body-loop currents"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("Vd")

        # Print key values at Vd=1.0 (mid-snap region)
        idx = np.argmin(np.abs(Vd - 1.0))
        print(f"VG1={vg1} VG2={vg2}  @Vd≈{Vd[idx]:.2f}V:")
        print(f"  Vb={Vb[idx]:.4f}V  Vsint={Vsint[idx]:.4f}V  Id={Id[idx]:.3e}A")
        print(f"  Iii_M1={Iii_M1[idx]:.3e}  Ic_Q1={Ic_Q1[idx]:.3e}  I_well={I_well[idx]:.3e}")
        summary[f"VG1={vg1}_VG2={vg2}"] = {
            "Vd": Vd.tolist(), "Vb": Vb.tolist(), "Vsint": Vsint.tolist(),
            "Id": Id.tolist(), "Iii_M1": Iii_M1.tolist(),
            "Ic_Q1": Ic_Q1.tolist(), "I_well": I_well.tolist(),
        }

    # M1 has 3rd column for VG1=0.4 vs 0.6 ratio
    for col in range(2):
        axes[0, 2+col].axis("off")
    axes[0, 2].text(0.05, 0.5, "VG1=0.4 row: snap doesn't fire (mean RMSE 1.95)\n"
                     "VG1=0.6 row: snap fires (mean 0.77)\n\n"
                     "Look for: which loop term is too small at VG1=0.4?\n"
                     "If Iii small → β0/α0 problem at this Vbs\n"
                     "If Iii ok but Vb low → drainage too strong\n"
                     "If both ok but Ic_Q1 small → BJT not lighting",
                     fontsize=10, family="monospace", verticalalignment="center")
    fig.tight_layout()
    fig.savefig(OUT / "loop_trace.png", dpi=130)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z91n] saved {OUT}/loop_trace.png + summary.json")


if __name__ == "__main__":
    main()
