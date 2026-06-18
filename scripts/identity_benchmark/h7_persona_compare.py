"""Visual side-by-side of the two hardware-driven personas.
Reads persona_demo_{host}.json (real generations) and renders a 1080p split-screen:
left = ikaros fingerprint -> "the engineer", right = daedalus fingerprint -> "the poet".
Same frozen model, same prompts; the ONLY difference is the measured per-core fingerprint.

Usage: python h7_persona_compare.py [--json results/.../persona_demo_daedalus.json]
Out: results/IDENTITY_H7_2026-06-09/persona_compare.png
"""
import argparse, json, textwrap
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=Path("results/IDENTITY_H7_2026-06-09")
BG="#0b0e14"; FG="#e6edf3"; OK="#3fb950"; AM="#ff9f43"; MUT="#8b949e"; AC="#36c5f0"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
    "text.color":FG,"font.size":12})

def wrap(s,n=46): return "\n".join(textwrap.wrap(s,n))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--json",default=str(OUT/"persona_demo_daedalus.json"))
    a=ap.parse_args()
    d=json.loads(Path(a.json).read_text())
    fi=np.load(OUT/"fingerprint_ikaros.npy"); fd=np.load(OUT/"fingerprint_daedalus.npy")
    prompts=list(d["gen"].keys())[:3]
    fig=plt.figure(figsize=(16,9));
    fig.suptitle("One model, one prompt — only the chip's fingerprint changes",
                 fontsize=20,fontweight="bold",color=FG,y=0.975)
    fig.text(0.5,0.935,"a frozen GPT-2 + FiLM adapter; the measured per-core silicon vector selects the writing persona",
             ha="center",fontsize=11.5,color=MUT)
    # column headers with mini fingerprint
    for cx,(name,role,col,fp) in [(0.27,("ikaros","the engineer · terse, technical",OK,fi)),
                                  (0.73,("daedalus","the poet · lyrical, warm",AM,fd))]:
        axm=fig.add_axes([cx-0.16,0.80,0.32,0.08]); axm.axis("off")
        axm.text(0.5,0.62,f"{name}  —  {role}",ha="center",fontsize=15,fontweight="bold",color=col,transform=axm.transAxes)
        axb=fig.add_axes([cx-0.10,0.78,0.20,0.045]);
        axb.bar(np.arange(16),fp,color=col); axb.axis("off")
    # rows of prompts
    y=0.70; rh=0.225
    for i,p in enumerate(prompts):
        gi=d["gen"][p]["ikaros_fp"]; gd=d["gen"][p]["daedalus_fp"]
        # strip the prompt echo for clarity but keep readability
        pi=gi; pd=gd
        fig.text(0.5,y+0.01,f'prompt:  "{p}…"',ha="center",fontsize=12,color=AC,style="italic")
        for cx,(txt,col) in [(0.27,(pi,OK)),(0.73,(pd,AM))]:
            ax=fig.add_axes([cx-0.225,y-rh+0.04,0.45,rh-0.05]); ax.axis("off")
            ax.add_patch(FancyBboxPatch((0.01,0.01),0.98,0.98,boxstyle="round,pad=0.01",
                fc="#11161f",ec=col,lw=1.8,transform=ax.transAxes))
            ax.text(0.06,0.90,wrap(txt,44),fontsize=11.5,color=FG,va="top",ha="left",
                    transform=ax.transAxes,family="monospace")
        y-=rh
    fig.text(0.5,0.025,"[AI] Claude Code, autonomous research for Eric on an HP workstation  ·  github.com/Heigke/FabricCrypt",
             ha="center",fontsize=10,color=MUT)
    p=OUT/"persona_compare.png"; fig.savefig(p,dpi=130); print(f"saved {p}")

if __name__=="__main__": main()
