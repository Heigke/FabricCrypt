"""z91m — instrument the Vgsteff/subthreshold bridge.

A.5.d: A9 reported pyport diverges from ngspice at Vds=0.05V — pyport
+0.92 dec high in deep subthreshold (Vgs=0.40), -0.44 dec low near-on
(Vgs=0.58). The phi fix only moved Vth gap by 3 mV. The bug is in
the Vgsteff bridge (dc.py:397-472) or its downstream Id computation.

This dumps every intermediate of the Vgsteff bridge at a bias sweep
around Vth at Vds=0.05V, plots them, identifies which quantity has a
discontinuity / sign flip / wrong scaling.
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
OUT = ROOT / "results/z91m_vgsteff_inspect"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91j_mod", ROOT / "scripts/z91j_ngspice_isolated_m2.py")
z91j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91j)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.model_card import BSIM4Model

DATA = ROOT / "data/sebas_2026_04_22"


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91j.z91f.patch_model_values(model, type_n=True)
    cfg = NSRAMCell2TConfig()
    geom = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    sd = compute_size_dep(model, geom, T_C=27.0)
    print(f"[z91m] M2 geom: L={geom.L:g} W={geom.W:g}")
    print(f"[z91m] sd vth0_eff = {sd.scaled.get('vth0', float('nan'))}")
    print(f"[z91m] sd nfactor = {sd.scaled.get('nfactor', float('nan'))}")
    print(f"[z91m] sd k1 = {sd.scaled.get('k1', float('nan'))}")
    print(f"[z91m] sd voff = {sd.scaled.get('voff', float('nan'))}")
    print(f"[z91m] ctx phi = {getattr(sd.model_ctx, 'phi', 'n/a')}")

    Vds = 0.05
    vgs_arr = np.arange(0.30, 0.85, 0.025)

    # Patch compute_dc to capture intermediates: monkey-patch by
    # re-implementing the relevant bits. Easier: run compute_dc and then
    # re-derive Vgsteff from its intermediate fields if it exposes them.
    # compute_dc returns DCResult; check what it exposes.
    Vg = torch.tensor(vgs_arr, dtype=torch.float64)
    Vd_t = torch.full_like(Vg, Vds)
    Vb_t = torch.zeros_like(Vg)
    out = compute_dc(model=model, sd=sd, Vgs=Vg, Vds=Vd_t, Vbs=Vb_t)

    # Inspect what's available
    avail = [a for a in dir(out) if not a.startswith("_")]
    print(f"[z91m] DCResult fields: {avail}")

    # Pull out fields we know about
    fields = {}
    for name in ("Ids", "Vgsteff", "Vdseff", "Vth", "n", "Vgst",
                  "T10", "T9", "Vgs_eff", "Vbsh"):
        if hasattr(out, name):
            v = getattr(out, name)
            if isinstance(v, torch.Tensor) and v.numel() == len(vgs_arr):
                fields[name] = v.detach().numpy()

    print(f"[z91m] captured fields: {list(fields.keys())}")
    # Also compute Id
    Id_py = out.Ids.abs().numpy()
    fields["log10_Id"] = np.log10(np.maximum(Id_py, 1e-30))

    # ngspice for reference
    Id_ng = z91j.run_ngspice_id_vd if False else None
    # Use z91k's Id-Vgs (we want at fixed Vds=0.05V)
    import importlib.util as iu
    _sk = iu.spec_from_file_location("z91k", ROOT / "scripts/z91k_subthreshold_slope.py")
    z91k = iu.module_from_spec(_sk); _sk.loader.exec_module(z91k)
    Id_ng = z91k.run_ngspice_id_vgs(Vds, geom, vgs_arr)
    fields["log10_Id_ngspice"] = np.log10(np.maximum(np.abs(Id_ng), 1e-30))
    fields["dec_diff"] = fields["log10_Id"] - fields["log10_Id_ngspice"]

    # Print table at key biases
    print(f"\n[z91m]  Vgs    log10_Id_py  log10_Id_ng  diff(dec)  "
          + "  ".join(k for k in ("Vgsteff", "Vth", "n", "Vbsh") if k in fields))
    for i, vgs in enumerate(vgs_arr):
        row = [f"{vgs:5.2f}",
               f"{fields['log10_Id'][i]:11.3f}",
               f"{fields['log10_Id_ngspice'][i]:11.3f}",
               f"{fields['dec_diff'][i]:+9.3f}"]
        for k in ("Vgsteff", "Vth", "n", "Vbsh"):
            if k in fields:
                row.append(f"{fields[k][i]:9.4f}")
        print("  " + "  ".join(row))

    # Plot dec_diff vs Vgs
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].plot(vgs_arr, fields["log10_Id"], "r-", label="pyport")
    axes[0].plot(vgs_arr, fields["log10_Id_ngspice"], "k-", label="ngspice")
    axes[0].set_ylabel("log10 |Id|"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[0].set_title(f"z91m Vgsteff bridge inspect, Vds={Vds}V, body=GND")
    axes[1].plot(vgs_arr, fields["dec_diff"], "b-", lw=1.5)
    axes[1].axhline(0, color="k", ls=":")
    axes[1].set_ylabel("py - ngspice (dec)"); axes[1].set_xlabel("Vgs")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "vgsteff_inspect.png", dpi=140)

    # JSON summary
    summary = {
        "Vds": Vds,
        "vgs_array": vgs_arr.tolist(),
        "fields": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fields.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z91m] saved {OUT}/vgsteff_inspect.png + summary.json")


if __name__ == "__main__":
    main()
