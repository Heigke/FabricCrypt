"""z92 — quasi-static transient capacitance correction.

A.4.b: with Sebas's pdiode Cj=10 fF at body↔vnwell, compute the
displacement current I_cap = Cj(V)·dVb/dt along the Vb(Vd) trajectory
from a DC sweep, assuming dVd/dt = 0.25 V/s (Sebas's slow ramp rate).

If I_cap is comparable to Id at the bias points where measurements
deviate from our DC fit (esp. VG1=0.2 row where shape is nearly right
but knee softness is off), then transient cap is the missing piece.
If I_cap << Id everywhere, the cap is irrelevant for these slow
ramps and the residual is purely DC card calibration.
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
OUT = ROOT / "results/z92_transient_cap_check"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(z91f)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.transient import junction_cap


def main():
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91f.patch_model_values(M1, type_n=True)
    z91f.patch_model_values(M2, type_n=True)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                              newton_max_iters=50)
    cfg.vnwell_Rs = 1e11
    sd_M1 = compute_size_dep(M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(M2, Geometry(L=cfg.Ln*cfg.M2_length_factor, W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1; cfg._sd_M2 = sd_M2
    sebas_rows = z91f.load_sebas_params()

    Cj0_per_area = cfg.body_pdiode_Cj0_per_area
    area = cfg.body_pdiode_area
    Vj = cfg.body_pdiode_Vj
    M_grading = cfg.body_pdiode_M
    Cj0_total = Cj0_per_area * area

    DVD_DT = 0.25  # V/s ramp rate (Sebas)
    Vd_seq = torch.tensor(np.linspace(0.05, 2.0, 40), dtype=torch.float64)
    t_seq = (Vd_seq - 0.05) / DVD_DT  # seconds, starting at t=0
    print(f"[z92] dVd/dt = {DVD_DT} V/s,  t_total = {float(t_seq[-1]):.2f} s")
    print(f"[z92] Cj0_total = {Cj0_total*1e15:.2f} fF, body_pdiode area = {area*1e12:.1f} µm²")

    # Pick 6 representative biases — one per VG1 row at low/mid/high VG2
    biases = [(0.2, -0.1), (0.2, +0.05), (0.4, +0.1), (0.4, +0.25),
                (0.6, +0.1), (0.6, +0.4)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    summary = []
    for k, (vg1, vg2) in enumerate(biases):
        ax = axes[k // 3][k % 3]
        sebas_row = z91f.find_params(sebas_rows, vg1, vg2)
        if sebas_row is None:
            print(f"[z92] no CSV row for ({vg1},{vg2}), skipping")
            continue
        P_M1, P_M2 = z91f.make_overrides(sebas_row)
        if P_M2:
            for kk in ("k1","k2","etab","beta0"):
                P_M2.pop(kk, None)
            if not P_M2: P_M2 = None
        bjt = z91f.make_bjt(sebas_row); bjt.Bf = 5e4
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if np.isnan(mbjt): mbjt = 1.0
        cfg.vnwell_mbjt = mbjt
        with torch.no_grad(), \
             z91f.patch_sd_scaled(sd_M1, P_M1), \
             z91f.patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t_arclength_grad(
                cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                Vd_seq=Vd_seq, VG1=torch.tensor(vg1), VG2=torch.tensor(vg2))
        Vb = out["Vb"].detach().numpy()
        Id_dc = out["Id"].abs().detach().numpy()
        Vd_arr = Vd_seq.numpy()

        # Cj at each bias point
        V_diode = Vb - cfg.vnwell  # body − vnwell (negative when reverse-biased)
        Cj = junction_cap(torch.tensor(V_diode), Cj0=Cj0_total, Vj=Vj, M=M_grading).numpy()
        # dVb/dt via central differences
        dVb = np.gradient(Vb, t_seq.numpy())
        I_cap = Cj * dVb  # A
        # Total transient Id = DC Id ± I_cap (sign depends on convention; cap
        # current at body adds to D node via NPN/M1, but for first-order check
        # we just compare magnitudes)
        Id_with_cap = np.abs(Id_dc + I_cap)
        ratio = np.abs(I_cap) / np.maximum(Id_dc, 1e-30)

        ax.semilogy(Vd_arr, np.maximum(Id_dc, 1e-15), "k-", label="DC Id")
        ax.semilogy(Vd_arr, np.maximum(np.abs(I_cap), 1e-15), "r-", label="|I_cap|")
        ax.semilogy(Vd_arr, np.maximum(Id_with_cap, 1e-15), "b--", label="DC + cap")
        ax.set_title(f"VG1={vg1}, VG2={vg2}  max(Icap/Id)={ratio.max():.2e}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(1e-15, 1e-3)
        if k // 3 == 1:
            ax.set_xlabel("Vd")
        if k % 3 == 0:
            ax.set_ylabel("|I| [A]")
        summary.append({"vg1": vg1, "vg2": vg2,
                          "max_Icap_over_Id": float(ratio.max()),
                          "max_Icap_A": float(np.abs(I_cap).max()),
                          "Cj_min_fF": float(Cj.min()*1e15),
                          "Cj_max_fF": float(Cj.max()*1e15),
                          "Vb_min": float(Vb.min()),
                          "Vb_max": float(Vb.max()),
                          "dVb_dt_max_V_per_s": float(np.abs(dVb).max())})
        print(f"VG1={vg1} VG2={vg2:+.2f}: Vb∈[{Vb.min():.3f},{Vb.max():.3f}] "
              f"Cj∈[{Cj.min()*1e15:.2f},{Cj.max()*1e15:.2f}] fF "
              f"max|Icap|={np.abs(I_cap).max():.2e} A "
              f"max(Icap/Id)={ratio.max():.2e}")
    fig.suptitle(f"z92 transient cap correction — dVd/dt = {DVD_DT} V/s, "
                  f"Cj0={Cj0_total*1e15:.1f} fF body↔vnwell")
    fig.tight_layout()
    fig.savefig(OUT / "transient_cap.png", dpi=140)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z92] saved {OUT}/transient_cap.png + summary.json")


if __name__ == "__main__":
    main()
