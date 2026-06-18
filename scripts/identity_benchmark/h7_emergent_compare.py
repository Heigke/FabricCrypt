"""Visual side-by-side of the EMERGENT voices, each computed on its own machine.
Reads emergent_persona_ikaros.json (computed on ikaros) + emergent_persona_daedalus.json
(computed on daedalus). Same frozen model, same prompts, one injection strength; the only
difference is each chip's own measured fingerprint, fed into the model's activations.
NO training, NO chosen persona.

Usage: python h7_emergent_compare.py [--alpha 6.0]
Out: results/IDENTITY_H7_2026-06-09/emergent_compare.png
"""
import argparse, json, textwrap
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=Path("results/IDENTITY_H7_2026-06-09")
BG="#0b0e14"; FG="#e6edf3"; OK="#3fb950"; AM="#ff9f43"; MUT="#8b949e"; AC="#36c5f0"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,"text.color":FG})

def wrap(s,n=46): return "\n".join(textwrap.wrap(s,n))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--alpha",default="6.0"); a=ap.parse_args()
    ak=json.loads((OUT/"emergent_persona_ikaros.json").read_text())
    da=json.loads((OUT/"emergent_persona_daedalus.json").read_text())
    key=f"alpha={a.alpha}"
    prompts=ak["prompts"][:3]
    fi=np.load(OUT/"fingerprint_multisig_ikaros.npy"); fd=np.load(OUT/"fingerprint_multisig_daedalus.npy")
    fig=plt.figure(figsize=(16,9))
    fig.suptitle("The chip's own signals write the voice — no persona chosen, no training",
                 fontsize=19,fontweight="bold",color=FG,y=0.975)
    fig.text(0.5,0.935,"same frozen GPT-2, same prompt, same injection strength — the only difference is each die's measured fingerprint",
             ha="center",fontsize=11.5,color=MUT)
    for cx,(name,where,col,fp) in [(0.27,("ikaros","voice computed on ikaros",OK,fi)),
                                   (0.73,("daedalus","voice computed on daedalus",AM,fd))]:
        axm=fig.add_axes([cx-0.16,0.80,0.32,0.08]); axm.axis("off")
        axm.text(0.5,0.62,f"{name}",ha="center",fontsize=17,fontweight="bold",color=col,transform=axm.transAxes)
        axm.text(0.5,0.18,where,ha="center",fontsize=11,color=MUT,transform=axm.transAxes)
        axb=fig.add_axes([cx-0.10,0.785,0.20,0.04]); axb.bar(np.arange(len(fp)),fp,color=col); axb.axis("off")
    y=0.70; rh=0.225
    for p in prompts:
        ti=ak["gen"]["ikaros"][p][key]; td=da["gen"]["daedalus"][p][key]
        fig.text(0.5,y+0.012,f'prompt:  "{p}…"',ha="center",fontsize=12,color=AC,style="italic")
        for cx,(txt,col) in [(0.27,(ti,OK)),(0.73,(td,AM))]:
            ax=fig.add_axes([cx-0.225,y-rh+0.04,0.45,rh-0.05]); ax.axis("off")
            ax.add_patch(FancyBboxPatch((0.01,0.01),0.98,0.98,boxstyle="round,pad=0.01",fc="#11161f",ec=col,lw=1.8,transform=ax.transAxes))
            ax.text(0.06,0.9,wrap(txt,46),fontsize=11,color=FG,va="top",ha="left",transform=ax.transAxes,family="monospace")
        y-=rh
    fig.text(0.5,0.04,f"injection strength α={a.alpha}  ·  at α=0 both dies are IDENTICAL → the divergence is 100% the fingerprint",
             ha="center",fontsize=11,color=MUT)
    fig.text(0.5,0.014,"[AI] Claude Code, autonomous research for Eric on an HP workstation  ·  github.com/Heigke/FabricCrypt",
             ha="center",fontsize=9.5,color=MUT)
    p=OUT/"emergent_compare.png"; fig.savefig(p,dpi=130); print(f"saved {p}")

if __name__=="__main__": main()
