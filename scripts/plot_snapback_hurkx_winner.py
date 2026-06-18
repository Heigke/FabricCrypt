#!/usr/bin/env python3
"""Snapback plot with WINNING config: K1+ALPHA0 card + Hurkx A=1e-6 B=1.5."""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"nsram")); sys.path.insert(0, str(ROOT/"scripts"))
sp = importlib.util.spec_from_file_location("p", ROOT/"scripts/pillar_I_C3_jts_tat.py")
p = importlib.util.module_from_spec(sp); sp.loader.exec_module(p)
from nsram.bsim4_port.nsram_cell_2T import forward_2t

curves = p.load_curves(); rows = p.load_sebas_params()
VG2_PICKS = [-0.1, 0.0, 0.1, 0.2]

def pick(vg2):
    best=None; bd=1e9
    for c in curves:
        if abs(c["VG1"]-0.6)>1e-6: continue
        d=abs(c["VG2"]-vg2)
        if d<bd: bd=d; best=c
    return best

def runc(c, K1, A0, hbbtA=0.0, hbbtB=1.0):
    p.BRANCH_FLAT[0.6]["K1"]=K1
    orig = p.make_overrides
    def patched(r):
        P1,P2=orig(r); P1=P1 or {}; P2=P2 or {}
        P1["alpha0"]=A0; P2["alpha0"]=A0; return P1,P2
    p.make_overrides = patched
    try:
        cfg,M1,M2,bjt = p.build_pyport_base()
        cfg.hurkx_bbt_A = float(hbbtA); cfg.hurkx_bbt_B = float(hbbtB)
        sd1=cfg.size_dep_M1(M1); sd2=cfg.size_dep_M2(M2)
        outs=[]
        for vdk,idk in [("fwd_Vd","fwd_Id"),("bwd_Vd","bwd_Id")]:
            Vd=c[vdk]
            row,_=p.find_or_impute_row(rows,c["VG1"],c["VG2"])
            P1,P2=p.make_overrides(row)
            with p.patch_sd_scaled(sd1,P1),p.patch_sd_scaled(sd2,P2):
                out=forward_2t(cfg,model_M1=M1,model_M2=M2,bjt=bjt,
                    Vd_seq=torch.tensor(Vd,dtype=torch.float64),
                    VG1=torch.tensor(c["VG1"],dtype=torch.float64),
                    VG2=torch.tensor(c["VG2"],dtype=torch.float64),warm_start=True)
            outs.append((Vd, np.abs(out["Id"].detach().cpu().numpy())))
        return outs
    finally:
        p.make_overrides=orig

COND = [
    ("baseline (1.163 dec)",                0.41825, 7.842e-5,  0.0,  1.0,  "#999", "--"),
    ("K1+ALPHA0 card (0.665 dec)",          0.53825, 7.83756e-4,0.0,  1.0,  "#1976d2","-"),
    ("+ Hurkx A=1e-6 B=1.5 (0.622 dec)",    0.53825, 7.83756e-4,1e-6, 1.5,  "#e53935","-"),
]

fig, axs = plt.subplots(1, len(VG2_PICKS), figsize=(5*len(VG2_PICKS), 5), sharey=True)
for ax, vg2_t in zip(axs, VG2_PICKS):
    c = pick(vg2_t)
    ax.semilogy(c["fwd_Vd"], np.clip(c["fwd_Id"],1e-13,None),"k.",ms=4, label="Sebas data (fwd)")
    ax.semilogy(c["bwd_Vd"], np.clip(c["bwd_Id"],1e-13,None),"k+",ms=4,alpha=0.5,label="Sebas data (bwd)")
    for label,K1,A0,hA,hB,clr,ls in COND:
        res = runc(c, K1, A0, hA, hB)
        for (Vd, I), tag in zip(res, ["fwd","bwd"]):
            lbl = label if tag=="fwd" else None
            ax.semilogy(Vd, np.clip(I,1e-13,None), color=clr, ls=ls, lw=1.4, label=lbl, alpha=0.85)
    ax.set_title(f"VG1=0.6, VG2={c['VG2']:.2f}")
    ax.set_xlabel("Vd (V)"); ax.grid(alpha=0.3,which="both")
    ax.set_ylim(1e-12, 1e-2)
    if ax is axs[0]: ax.set_ylabel("|Id| (A)"); ax.legend(loc="lower right", fontsize=8)

plt.suptitle("NS-RAM 2T — snapback fit after Hurkx BBT addition\n"
             "Red (Hurkx) tracks data knee at Vd≈0.85-1.15V — model knee shifted left from 1.5V to 1.10V",
             fontsize=11)
plt.tight_layout()
out = ROOT/"results/track_hurkx_bbt/plot_snapback_winner.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(out)
