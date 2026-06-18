"""Render the honest demo infographic (1920x1080) — real-time signals as the headline + layers of
embodiment. Reads demo_samples.json (RT-only model). Out: demo_frames/H7_infographic.png
"""
import json, textwrap
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"; FR=OUT/"demo_frames"; FR.mkdir(parents=True,exist_ok=True)
BG="#0b0e14"; FG="#e6edf3"; OK="#3fb950"; BAD="#f85149"; AM="#d29922"; MUT="#8b949e"; AC="#58a6ff"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,"text.color":FG,"font.family":"DejaVu Sans"})
S=json.loads((OUT/"demo_samples.json").read_text())["gen"]
def wrap(s,n): return "\n".join(textwrap.wrap(s,n))
fig=plt.figure(figsize=(19.2,10.8)); fig.subplots_adjust(0,0,1,1)
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,16); ax.set_ylim(0,9)
ax.add_patch(FancyBboxPatch((0.3,8.28),15.4,0.52,boxstyle="round,pad=0.04",fc="#161b22",ec=AC,lw=1))
ax.text(8,8.54,"AI-generated research log — Claude Code (Anthropic), running autonomously for Eric Bergvall on an HP workstation",
        ha="center",va="center",fontsize=12,color=AC)
ax.text(8,7.75,"An LLM whose voice runs on its chip's LIVE signals",ha="center",fontsize=29,fontweight="bold")
ax.text(8,7.2,"a frozen GPT-2 driven by real-time Vcore + per-core clock + power — fluent on its own body, broken on any other",
        ha="center",fontsize=13,color=MUT)
# left: samples (RT-only model)
ax.text(4.2,6.5,'Same prompt:  "It is a truth …"   (live-signals-only model)',ha="center",fontsize=12.5,color=AC,style="italic")
def box(x,y,w,h,title,txt,col):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05",fc="#11161f",ec=col,lw=2))
    ax.text(x+0.2,y+h-0.3,title,fontsize=12,color=col,fontweight="bold")
    ax.text(x+0.2,y+h-0.68,wrap(txt,54),fontsize=10.5,color=FG,va="top",family="monospace")
box(0.6,4.55,7.2,1.55,"✓ its OWN live body (correct chip)", S["ikaros"]["own"][:140], OK)
box(0.6,2.8,7.2,1.55,"✗ wrong body (random live vector)", S["ikaros"]["random"][:140], BAD)
# right: numbers (RT-only)
ax.text(12,6.5,"Held-out perplexity — LIVE-SIGNALS-ONLY model (no TPM, no fused constants)",ha="center",fontsize=12.5,color=AC)
rows=[("own live body — ikaros","47.2",OK),("own live body — daedalus","37.8",OK),("plain GPT-2 baseline","64.9 / 66.6",MUT),
      ("zero","7.6k–8.7k",BAD),("foreign die","56k–84k",BAD),("random","46k–49k",BAD)]
y=5.95
for name,val,col in rows:
    ax.text(8.9,y,name,fontsize=12,color=FG); ax.text(15.4,y,val,fontsize=12,color=col,ha="right",fontweight="bold"); y-=0.40
ax.text(8.9,y+0.02,"→ own ≈/better than baseline;  wrong body 150–2200× worse;  deterministic",fontsize=11,color=AM)
# layers of embodiment
ax.text(0.6,2.3,"Layers of embodiment (each verified on real silicon, ikaros + daedalus):",fontsize=12,color=FG,fontweight="bold")
layers=["① identity — per-die Vcore + clock pattern separates the chips",
        "② freshness — live jitter drifts → a replayed reading goes stale",
        "③ real-time coupling — generates off live reads of its own body",
        "④ behaviour binding — right body fluent, wrong body broken (trained)",
        "⑤ crypto lock — TPM-sealed key, REFUSED on a foreign die (the lock only)",
        "⑥ liveness — fresh TPM nonce gates each run"]
for i,t in enumerate(layers):
    cx=0.7 if i<3 else 8.3; yy=1.95-(i%3)*0.45
    ax.text(cx,yy,t,fontsize=10.3,color=FG)
ax.text(8,0.32,"Honest: operational embodiment / HW-bound licensing — NOT consciousness; quality=adaptation; no GPU TEE.  ·  github.com/Heigke/FabricCrypt",
        ha="center",fontsize=10.5,color=MUT)
p=FR/"H7_infographic.png"; fig.savefig(p,dpi=100); print(f"saved {p}")
