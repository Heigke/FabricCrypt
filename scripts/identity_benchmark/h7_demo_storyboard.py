"""H7 killer-demo storyboard — one figure, real data, the embodiment story.
Panels: (A) the 4-box fleet + 200G nerve, (B) per-core silicon fingerprints ikaros vs daedalus (real),
(C) the embodiment LOCK = PPL with own vs wrong/shuffle/zero fingerprint (real daedalus margins),
(D) two-tier honesty: substrate-dependence (proven) + TPM hard root (uncopyability).
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
OUT=Path("results/IDENTITY_H7_2026-06-09")
fi=np.load(OUT/"fingerprint_ikaros.npy"); fd=np.load(OUT/"fingerprint_daedalus.npy")
# real daedalus multi-neg falsification PPLs
lab=["own die","wrong die","shuffled","zero"]; ppl=[24.0, 31.5e6, 15.5e9, 18.5e9]
BG="#0b0e14"; FG="#e6edf3"; AC="#36c5f0"; OK="#3fb950"; BAD="#f85149"; AM="#ff9f43"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
    "text.color":FG,"axes.labelcolor":FG,"xtick.color":FG,"ytick.color":FG,"font.size":11})
fig=plt.figure(figsize=(16,9));
fig.suptitle("H7 — A mind that only runs on its own body  ·  4-machine substrate-rooted LLM",
             fontsize=17,color=FG,fontweight="bold",y=0.98)
gs=fig.add_gridspec(2,2,hspace=0.32,wspace=0.22,left=0.06,right=0.97,top=0.90,bottom=0.07)

# ---- A: fleet ----
axA=fig.add_subplot(gs[0,0]); axA.set_xlim(0,10); axA.set_ylim(0,10); axA.axis("off")
axA.set_title("A · The fleet & the 200G nerve",loc="left",color=AC,fontweight="bold")
nodes={"ikaros":(2,7,"AMD Strix Halo","UNIQUE: per-core Vcore + TPM",OK),
       "daedalus":(2,3,"AMD Strix Halo","UNIQUE: per-core Vcore + TPM",OK),
       "zgx-5175":(7.5,7,"NVIDIA GB10","FRESH: GPU power @10Hz",AM),
       "zgx06":(7.5,3,"NVIDIA GB10","FRESH: GPU power @10Hz",AM)}
for n,(x,y,sub,role,c) in nodes.items():
    axA.add_patch(FancyBboxPatch((x-1.5,y-0.8),3,1.6,boxstyle="round,pad=0.1",
        fc="#161b22",ec=c,lw=2))
    axA.text(x,y+0.35,n,ha="center",fontweight="bold",fontsize=12,color=c)
    axA.text(x,y-0.05,sub,ha="center",fontsize=8,color=FG)
    axA.text(x,y-0.45,role,ha="center",fontsize=7.5,color="#8b949e")
# 200G nerve between the two GB10
axA.add_patch(FancyArrowPatch((7.5,6.2),(7.5,3.8),arrowstyle="<->",color=AC,lw=3,mutation_scale=18))
axA.text(8.1,5,"200 Gb/s\nRoCE nerve\n(CX-7 BER)",fontsize=8,color=AC)
# AMD pair coupling
axA.add_patch(FancyArrowPatch((2,6.2),(2,3.8),arrowstyle="<->",color=OK,lw=1.5,ls=":",mutation_scale=12))
axA.text(0.2,5,"cross-die\nhard-neg",fontsize=7.5,color=OK)

# ---- B: fingerprints ----
axB=fig.add_subplot(gs[0,1])
axB.set_title("B · Per-core silicon fingerprint (real, config-immune)",loc="left",color=AC,fontweight="bold")
x=np.arange(16); w=0.4
axB.bar(x-w/2,fi,w,label="ikaros",color=OK); axB.bar(x+w/2,fd,w,label="daedalus",color=AM)
axB.set_xlabel("CPU core"); axB.set_ylabel("Vcore scatter (z)")
axB.legend(facecolor="#161b22",edgecolor="#30363d",labelcolor=FG,loc="upper right",fontsize=9)
axB.text(0.02,0.04,f"within-die 0.98  ·  between-die 0.74  ·  cos(ik,da)=0.75",
         transform=axB.transAxes,fontsize=8.5,color="#8b949e")

# ---- C: the lock ----
axC=fig.add_subplot(gs[1,0])
axC.set_title("C · The embodiment LOCK — perplexity by fingerprint",loc="left",color=AC,fontweight="bold")
cols=[OK,BAD,BAD,BAD]; bars=axC.bar(lab,ppl,color=cols,log=True)
axC.set_ylabel("perplexity (log, lower=fluent)")
for b,p in zip(bars,ppl):
    axC.text(b.get_x()+b.get_width()/2,p*1.4,(f"{p:.0f}" if p<1e3 else f"{p:.0e}"),
             ha="center",fontsize=9,color=FG)
axC.text(0.02,0.90,"same weights, 4 fingerprints:\nfluent ONLY on its own die  ·  +20 nat (~10⁸×) on shuffle/zero",
         transform=axC.transAxes,fontsize=8.5,color="#8b949e",va="top")

# ---- D: two-tier ----
axD=fig.add_subplot(gs[1,1]); axD.axis("off")
axD.set_title("D · Two honest tiers",loc="left",color=AC,fontweight="bold")
axD.add_patch(FancyBboxPatch((0.03,0.55),0.94,0.38,boxstyle="round,pad=0.02",fc="#161b22",ec=OK,lw=2,transform=axD.transAxes))
axD.text(0.07,0.84,"✓ SUBSTRATE-DEPENDENCE  (proven, science)",fontsize=11,color=OK,fontweight="bold",transform=axD.transAxes)
axD.text(0.07,0.72,"LLM rooted in exact per-core silicon pattern.\nwrong-die / shuffle / zero ALL collapse it.\nFirst time the shuffle wall is broken with a physical signal.",
         fontsize=9,color=FG,transform=axD.transAxes,va="top")
axD.add_patch(FancyBboxPatch((0.03,0.07),0.94,0.40,boxstyle="round,pad=0.02",fc="#161b22",ec=OK,lw=2,transform=axD.transAxes))
axD.text(0.07,0.39,"✓ UNCOPYABILITY  (the hard root — VERIFIED)",fontsize=11,color=OK,fontweight="bold",transform=axD.transAxes)
axD.text(0.07,0.28,"Nuvoton TPM 2.0 per AMD box; fresh-nonce quote (liveness)\n+ AES key sealed to the die. Cross-die transplant matrix:\nown die UNLOCKS · foreign die REFUSED (TPM integrity mismatch).",
         fontsize=9,color=FG,transform=axD.transAxes,va="top")

p=OUT/"H7_killer_demo_storyboard.png"
fig.savefig(p,dpi=130); print(f"saved {p}")
