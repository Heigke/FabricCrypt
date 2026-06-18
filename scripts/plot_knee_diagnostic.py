#!/usr/bin/env python3
"""Knee diagnostic — decompose model snapback at VG1=0.6 into components.

Plot Vb, Ids_M1, Ids_M2, Ic_Q1 (NPN collector) vs Vd alongside total Id and
Sebas data. Identifies which mechanism's onset positions the knee.

Compares K1+ALPHA0 card-combo vs baseline so we see WHICH change shifted what.
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram")); sys.path.insert(0, str(ROOT / "scripts"))

sp = importlib.util.spec_from_file_location("pillar_I", ROOT / "scripts/pillar_I_C3_jts_tat.py")
pillar = importlib.util.module_from_spec(sp); sp.loader.exec_module(pillar)
from nsram.bsim4_port.nsram_cell_2T import forward_2t

OUT = ROOT / "results/track_combo_k1_alpha0"
OUT.mkdir(exist_ok=True)

curves = pillar.load_curves()
sebas_rows = pillar.load_sebas_params()

VG1_T = 0.6
VG2_PICKS = [0.0, 0.10, 0.20]


def pick(vg2):
    best = None; bd = 1e9
    for c in curves:
        if abs(c["VG1"] - VG1_T) > 1e-6: continue
        d = abs(c["VG2"] - vg2)
        if d < bd: bd = d; best = c
    return best


def run_decomp(curve, K1_val, ALPHA0_val):
    """Returns dict with Vd_fwd, Id, Ids_M1, Ids_M2, Ic_Q1, Vb (all fwd branch only)."""
    orig_k1 = pillar.BRANCH_FLAT[0.6]["K1"]
    pillar.BRANCH_FLAT[0.6]["K1"] = K1_val
    orig_make = pillar.make_overrides
    def patched(row):
        P_M1, P_M2 = orig_make(row)
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["alpha0"] = float(ALPHA0_val); P_M2["alpha0"] = float(ALPHA0_val)
        return P_M1, P_M2
    pillar.make_overrides = patched
    try:
        cfg, M1, M2, bjt = pillar.build_pyport_base()
        sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
        row, _ = pillar.find_or_impute_row(sebas_rows, curve["VG1"], curve["VG2"])
        P_M1, P_M2 = pillar.make_overrides(row)
        Vd_np = curve["fwd_Vd"]
        Vd_t = torch.tensor(Vd_np, dtype=torch.float64)
        with pillar.patch_sd_scaled(sd_M1, P_M1), pillar.patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                             Vd_seq=Vd_t,
                             VG1=torch.tensor(curve["VG1"], dtype=torch.float64),
                             VG2=torch.tensor(curve["VG2"], dtype=torch.float64),
                             warm_start=True)
        return {
            "Vd": Vd_np,
            "Id_data": curve["fwd_Id"],
            "Id_model": np.abs(out["Id"].detach().cpu().numpy()),
            "Ids_M1":   np.abs(out["Ids_M1"].detach().cpu().numpy()),
            "Ids_M2":   np.abs(out["Ids_M2"].detach().cpu().numpy()),
            "Ic_Q1":    np.abs(out["Ic_Q1"].detach().cpu().numpy()),
            "Vb":       out["Vb"].detach().cpu().numpy(),
        }
    finally:
        pillar.make_overrides = orig_make
        pillar.BRANCH_FLAT[0.6]["K1"] = orig_k1


def knee_vd(Vd, Id, factor=10, base_window=(0.0, 0.3)):
    """Vd where Id first exceeds factor× baseline median."""
    Id = np.asarray(Id); Vd = np.asarray(Vd)
    mask = (Vd >= base_window[0]) & (Vd <= base_window[1])
    base = np.median(Id[mask]) if mask.sum() else 1e-12
    above = Id > factor * max(base, 1e-13)
    idx = np.argmax(above)
    return float(Vd[idx]) if above.any() else float("nan")


COND = [
    ("baseline (K1=0.418, ALPHA0=7.84e-5)", 0.41825, 7.842e-5),
    ("K1+ALPHA0 card combo (0.538, 7.84e-4)", 0.53825, 7.83756e-4),
]

fig, axs = plt.subplots(2, len(VG2_PICKS), figsize=(6*len(VG2_PICKS), 10), sharex='col')

for col, vg2_t in enumerate(VG2_PICKS):
    c = pick(vg2_t)
    ax_top = axs[0, col]; ax_bot = axs[1, col]

    # Sebas data on top
    ax_top.semilogy(c["fwd_Vd"], np.clip(c["fwd_Id"], 1e-13, None),
                    "k.", markersize=5, label="Sebas data Id")
    k_data = knee_vd(c["fwd_Vd"], c["fwd_Id"], factor=10)
    ax_top.axvline(k_data, color="k", lw=1, ls=":", alpha=0.4)
    ax_top.text(k_data, 1e-4, f" data knee≈{k_data:.2f}V", fontsize=8, va="bottom")

    for tag, k1, a0 in COND:
        r = run_decomp(c, k1, a0)
        suf = " (combo)" if "combo" in tag else " (base)"
        # total Id model
        ls = "-" if "combo" in tag else "--"
        ax_top.semilogy(r["Vd"], np.clip(r["Id_model"], 1e-13, None),
                        ls=ls, lw=1.4, label=f"Id model{suf}")
        # components: only for combo to keep readable
        if "combo" in tag:
            ax_bot.semilogy(r["Vd"], np.clip(r["Ids_M1"], 1e-13, None), label="Ids_M1 (channel)", color="#1976d2")
            ax_bot.semilogy(r["Vd"], np.clip(r["Ids_M2"], 1e-13, None), label="Ids_M2 (drain mosfet)", color="#388e3c")
            ax_bot.semilogy(r["Vd"], np.clip(r["Ic_Q1"], 1e-13, None), label="Ic_Q1 (NPN collector)", color="#e53935")
            # Vb on secondary axis
            ax_vb = ax_bot.twinx()
            ax_vb.plot(r["Vd"], r["Vb"], color="#ff9800", lw=1.2, label="Vb (body)")
            ax_vb.set_ylabel("Vb (V)", color="#ff9800")
            ax_vb.tick_params(axis="y", labelcolor="#ff9800")
            ax_vb.set_ylim(-0.1, 1.0)
            ax_vb.axhline(0.7, color="#ff9800", lw=0.8, ls=":")
            ax_vb.text(0.05, 0.72, " Vbe≈0.7V (NPN trigger)", fontsize=7, color="#ff9800")
            k_model = knee_vd(r["Vd"], r["Id_model"], factor=10)
            ax_top.axvline(k_model, color="r", lw=1, ls=":", alpha=0.6)
            ax_top.text(k_model, 1e-7, f" model knee≈{k_model:.2f}V", fontsize=8, color="r", va="bottom")

    ax_top.set_title(f"VG1=0.6, VG2={c['VG2']:.2f}")
    ax_top.set_ylabel("|I| (A)")
    ax_top.legend(loc="lower right", fontsize=8)
    ax_top.grid(alpha=0.3, which="both")
    ax_top.set_ylim(1e-12, 1e-2)

    ax_bot.set_xlabel("Vd (V)")
    ax_bot.set_ylabel("component |I| (A)")
    ax_bot.legend(loc="lower right", fontsize=8)
    ax_bot.grid(alpha=0.3, which="both")
    ax_bot.set_ylim(1e-13, 1e-2)

plt.suptitle("Knee diagnostic — VG1=0.6 — model component decomposition vs Sebas data\n"
             "Top: total Id (data vs model baseline vs combo).  "
             "Bottom: combo's Ids_M1 / Ids_M2 / Ic_Q1 components, plus Vb on right axis.",
             fontsize=11)
plt.tight_layout()
outp = OUT / "plot_knee_diagnostic.png"
plt.savefig(outp, dpi=120, bbox_inches="tight")
print(outp)
