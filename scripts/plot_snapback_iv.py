#!/usr/bin/env python3
"""Plot Id-Vd snapback curves at VG1=0.6: data vs model under 3 conditions
(baseline / K1-card fix / K1+ALPHA0 card fix). Picks a few VG2 values to
show the full snapback regime."""
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

curves = pillar.load_curves()
sebas_rows = pillar.load_sebas_params()

VG1_TARGET = 0.6
VG2_PICKS = [-0.1, 0.0, 0.1, 0.2]

# Match each picked VG2 to a curve
picked = []
for vg2 in VG2_PICKS:
    best = None; bd = 1e9
    for c in curves:
        if abs(c["VG1"] - VG1_TARGET) > 1e-6:
            continue
        d = abs(c["VG2"] - vg2)
        if d < bd:
            bd = d; best = c
    if best is not None:
        picked.append((vg2, best))

print(f"[snapback] picked {len(picked)} curves at VG1={VG1_TARGET}: VG2 = {[p[0] for p in picked]}")


def run_model_at(curve, K1_val, ALPHA0_val):
    """Run forward_2t with the patched K1@VG=0.6 and ALPHA0 override.
    Returns (Vd_full, Id_full) covering fwd+bwd reconstructed."""
    orig_k1 = pillar.BRANCH_FLAT[0.6]["K1"]
    pillar.BRANCH_FLAT[0.6]["K1"] = K1_val
    orig_make = pillar.make_overrides
    def patched(row):
        P_M1, P_M2 = orig_make(row)
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["alpha0"] = float(ALPHA0_val)
        P_M2["alpha0"] = float(ALPHA0_val)
        return P_M1, P_M2
    pillar.make_overrides = patched
    try:
        cfg, M1, M2, bjt = pillar.build_pyport_base()
        sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
        row, _ = pillar.find_or_impute_row(sebas_rows, curve["VG1"], curve["VG2"])
        P_M1, P_M2 = pillar.make_overrides(row)

        results = []
        for vdk, idk in [("fwd_Vd", "fwd_Id"), ("bwd_Vd", "bwd_Id")]:
            Vd_np = curve[vdk]
            Vd_t = torch.tensor(Vd_np, dtype=torch.float64)
            try:
                with pillar.patch_sd_scaled(sd_M1, P_M1), pillar.patch_sd_scaled(sd_M2, P_M2):
                    out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                                     Vd_seq=Vd_t,
                                     VG1=torch.tensor(curve["VG1"], dtype=torch.float64),
                                     VG2=torch.tensor(curve["VG2"], dtype=torch.float64),
                                     warm_start=True)
                I_pred = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
            except Exception as e:
                print(f"  fail @ VG2={curve['VG2']} K1={K1_val} ALPHA0={ALPHA0_val}: {e}")
                I_pred = np.zeros_like(Vd_np)
            results.append((Vd_np, I_pred))
        return results
    finally:
        pillar.make_overrides = orig_make
        pillar.BRANCH_FLAT[0.6]["K1"] = orig_k1


# Three conditions
COND = [
    ("baseline (K1=0.418, ALPHA0=7.84e-5)", 0.41825, 7.842e-5, "#999999", "--"),
    ("K1 card fix only (K1=0.538)",         0.53825, 7.842e-5, "#1976d2", "-"),
    ("K1+ALPHA0 card combo",                0.53825, 7.83756e-4, "#e53935", "-"),
]

fig, axs = plt.subplots(1, len(picked), figsize=(5*len(picked), 5), sharey=True)
if len(picked) == 1:
    axs = [axs]

for ax, (vg2_target, c) in zip(axs, picked):
    Vd_meas_fwd = c["fwd_Vd"]; Id_meas_fwd = c["fwd_Id"]
    Vd_meas_bwd = c["bwd_Vd"]; Id_meas_bwd = c["bwd_Id"]
    ax.semilogy(Vd_meas_fwd, np.clip(Id_meas_fwd, 1e-13, None),
                "k.", markersize=4, label="Sebas data (fwd)")
    ax.semilogy(Vd_meas_bwd, np.clip(Id_meas_bwd, 1e-13, None),
                "k+", markersize=4, alpha=0.5, label="Sebas data (bwd)")

    for label, k1, a0, color, ls in COND:
        res = run_model_at(c, k1, a0)
        for (Vd, I), tag in zip(res, ["fwd", "bwd"]):
            lbl = label if tag == "fwd" else None
            ax.semilogy(Vd, np.clip(I, 1e-13, None), color=color, ls=ls, lw=1.4, label=lbl, alpha=0.85)

    ax.set_title(f"VG1=0.6, VG2={c['VG2']:.2f}", fontsize=11)
    ax.set_xlabel("Vd (V)")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(1e-12, 1e-2)
    if ax is axs[0]:
        ax.set_ylabel("|Id| (A)")
        ax.legend(loc="lower right", fontsize=8)

plt.suptitle("NS-RAM 2T cell — snapback Id-Vd: model vs Sebas data at VG1=0.6\n"
             "Triode regime (Vd≲0.5V) is where the K1 bug hurt most; full snapback shape recovered with both card-value fixes",
             fontsize=11)
plt.tight_layout()
out = OUT / "plot_snapback_vs_data.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(out)
