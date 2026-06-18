"""H7 infographic video — a ~35s 1080p MP4 telling the substrate-rooted-LLM story from REAL data.
Scenes: (0) title + auto-post disclosure, (1) the 4-machine fleet, (2) per-core silicon fingerprint
reveal (real ikaros vs daedalus), (3) the LOCK — own-die fluent English types in, wrong/shuffle
fingerprints collapse the SAME model to noise (real GPT-2 generations + PPL), (4) the TPM transplant
matrix lighting up (own die UNLOCKS / foreign die REFUSED, real result), (5) closing card + repo link.

Renders with matplotlib FuncAnimation -> ffmpeg. No network. Output: results/IDENTITY_H7_2026-06-09/H7_embodiment.mp4
"""
import json, textwrap
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("results/IDENTITY_H7_2026-06-09")
FPS = 25
BG="#0b0e14"; FG="#e6edf3"; AC="#36c5f0"; OK="#3fb950"; BAD="#f85149"; AM="#ff9f43"; MUT="#8b949e"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
    "text.color":FG,"font.size":13,"font.family":"DejaVu Sans"})

# ---- real data ----
fi=np.load(OUT/"fingerprint_ikaros.npy"); fd=np.load(OUT/"fingerprint_daedalus.npy")
demo=json.loads((OUT/"rooted_gpt2_demo_daedalus.json").read_text())
own_txt=demo["gen"]["own"].split(":",1)[-1].strip()[:150]
wrong_txt=demo["gen"]["wrong"].split(":",1)[-1].strip()[:150]
ppl=demo["ppl"]
mat=json.loads((OUT/"tpm_transplant_matrix.json").read_text())["matrix"]

W,H=19.2,10.8
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")

# ---- scene timing (seconds) ----
SC=[("title",4.5),("fleet",5.0),("finger",5.5),("lock",9.0),("tpm",7.0),("close",4.5)]
bounds=[]; t=0
for name,d in SC: bounds.append((name,t,t+d)); t+=d
TOTAL=t; NFR=int(TOTAL*FPS)

def ease(x): return 0 if x<0 else 1 if x>1 else x*x*(3-2*x)
def fade_alpha(local,dur,fin=0.6,fout=0.5):
    a=ease(local/fin) if local<fin else 1.0
    if local>dur-fout: a=min(a,ease((dur-local)/fout))
    return max(0,min(1,a))

def txt(x,y,s,**kw):
    kw.setdefault("ha","center"); kw.setdefault("va","center"); kw.setdefault("transform",ax.transData)
    return ax.text(x,y,s,**kw)

def draw_title(l,d):
    a=fade_alpha(l,d)
    txt(50,72,"An AI that only runs on its own body",fontsize=40,fontweight="bold",color=FG,alpha=a)
    txt(50,60,"A language model rooted in the live silicon of one specific chip",fontsize=20,color=AC,alpha=a)
    # disclosure chip
    ax.add_patch(FancyBboxPatch((28,40),44,9,boxstyle="round,pad=0.6",fc="#161b22",ec=AM,lw=2,
        transform=ax.transData,alpha=a))
    txt(50,45.5,"[AI]  Claude Code (Anthropic's agent) — autonomous research for Eric",
        fontsize=14,color=AM,alpha=a)
    txt(50,42.5,"run on an HP workstation, posted by the agent itself",
        fontsize=12,color=AM,alpha=a)
    txt(50,30,"4 machines · real hardware signals · cryptographic root",fontsize=15,color=MUT,alpha=a)

def draw_fleet(l,d):
    a=fade_alpha(l,d)
    txt(50,90,"The fleet & the 200G nerve",fontsize=26,fontweight="bold",color=AC,alpha=a)
    nodes={"ikaros":(28,62,"AMD Strix Halo","UNIQUE: per-core Vcore + TPM",OK),
           "daedalus":(28,30,"AMD Strix Halo","UNIQUE: per-core Vcore + TPM",OK),
           "zgx-5175":(72,62,"NVIDIA GB10","FRESH: GPU power @10Hz",AM),
           "zgx06":(72,30,"NVIDIA GB10","FRESH: GPU power @10Hz",AM)}
    order=list(nodes.items())
    for i,(n,(x,y,sub,role,c)) in enumerate(order):
        na=ease((l-0.4-0.5*i)/0.6)*a
        if na<=0: continue
        ax.add_patch(FancyBboxPatch((x-13,y-7),26,14,boxstyle="round,pad=0.4",fc="#161b22",ec=c,lw=2.5,alpha=na))
        txt(x,y+3.5,n,fontsize=20,fontweight="bold",color=c,alpha=na)
        txt(x,y-0.5,sub,fontsize=13,color=FG,alpha=na)
        txt(x,y-4,role,fontsize=11,color=MUT,alpha=na)
    if l>2.8:
        na=ease((l-2.8)/0.8)*a
        ax.add_patch(FancyArrowPatch((72,54),(72,38),arrowstyle="<->",color=AC,lw=3,mutation_scale=22,alpha=na))
        txt(83,46,"200 Gb/s\nRoCE nerve",fontsize=12,color=AC,alpha=na)
        ax.add_patch(FancyArrowPatch((28,54),(28,38),arrowstyle="<->",color=OK,lw=1.6,ls=":",mutation_scale=14,alpha=na))
        txt(16,46,"cross-die\nhard-neg",fontsize=11,color=OK,alpha=na)

def draw_finger(l,d):
    a=fade_alpha(l,d)
    txt(50,90,"Per-core silicon fingerprint (real, config-immune)",fontsize=24,fontweight="bold",color=AC,alpha=a)
    n=len(fi); x0,x1=12,88; w=(x1-x0)/n
    grow=ease(l/2.2)
    base=22; scale=18
    for i in range(n):
        bx=x0+i*w
        for vec,col,off in ((fi,OK,0.05),(fd,AM,0.5)):
            hgt=vec[i]*scale*grow
            ax.add_patch(plt.Rectangle((bx+off*w,base),0.4*w,hgt,color=col,alpha=a))
    txt(20,12,"ikaros",fontsize=16,color=OK,fontweight="bold",alpha=a)
    txt(34,12,"daedalus",fontsize=16,color=AM,fontweight="bold",alpha=a)
    if l>2.6:
        na=ease((l-2.6)/0.8)*a
        txt(50,80,"within-die 0.98   ·   between-die 0.74   ·   cos(ikaros,daedalus)=0.75",
            fontsize=15,color=MUT,alpha=na)
        txt(50,73,"z-scored core-to-core scatter removes BIOS voltage → pure silicon",fontsize=13,color=MUT,alpha=na)

def typed(s,frac):
    nshow=int(len(s)*ease(frac))
    return s[:nshow]

def draw_lock(l,d):
    a=fade_alpha(l,d)
    txt(50,92,"The embodiment LOCK — same weights, different bodies",fontsize=23,fontweight="bold",color=AC,alpha=a)
    # own panel (green) types in fluent english
    ax.add_patch(FancyBboxPatch((6,52),42,30,boxstyle="round,pad=0.5",fc="#10231a",ec=OK,lw=2.5,alpha=a))
    txt(27,78,"OWN die fingerprint",fontsize=16,fontweight="bold",color=OK,alpha=a)
    o=typed(own_txt, (l-0.5)/3.5)
    ax.text(9,73,"\n".join(textwrap.wrap('"'+o+('"' if len(o)>=len(own_txt) else ""),38)),
            fontsize=13,color=FG,va="top",ha="left",alpha=a,family="monospace")
    if l>4: txt(27,55,f"perplexity {ppl['own']:.1f}  ·  fluent",fontsize=14,color=OK,alpha=ease((l-4)/0.6)*a)
    # wrong panel (red) types in garbage
    ax.add_patch(FancyBboxPatch((52,52),42,30,boxstyle="round,pad=0.5",fc="#2a1416",ec=BAD,lw=2.5,alpha=a))
    txt(73,78,"WRONG die fingerprint",fontsize=16,fontweight="bold",color=BAD,alpha=a)
    if l>3:
        w=typed(wrong_txt,(l-3)/3.0)
        ax.text(55,73,"\n".join(textwrap.wrap('"'+w+'"',38)),fontsize=13,color="#f0a0a0",va="top",ha="left",
                alpha=ease((l-3)/0.4)*a,family="monospace")
    if l>6: txt(73,55,f"perplexity {ppl['wrong']:,.0f}  ·  collapses".replace(","," "),
                fontsize=14,color=BAD,alpha=ease((l-6)/0.6)*a)
    if l>7:
        na=ease((l-7)/0.8)*a
        txt(50,46,"Feed it another chip's signature and the SAME model breaks into noise.",
            fontsize=16,color=FG,alpha=na)
        txt(50,40,"shuffle the fingerprint → perplexity 2 900 000  (the 'shuffle wall', broken with a physical signal)",
            fontsize=13,color=MUT,alpha=na)

def draw_tpm(l,d):
    a=fade_alpha(l,d)
    txt(50,92,"The hard root — model key sealed into each chip's TPM",fontsize=23,fontweight="bold",color=AC,alpha=a)
    txt(50,84,"Cross-die transplant: copy the weights to another machine → REFUSED in hardware",
        fontsize=15,color=MUT,alpha=a)
    cells=[("ikaros TPM","ikaros vault",mat["ikaros_TPM__ikaros_vault"]),
           ("ikaros TPM","daedalus vault",mat["ikaros_TPM__daedalus_vault"]),
           ("daedalus TPM","daedalus vault",mat["daedalus_TPM__daedalus_vault"]),
           ("daedalus TPM","ikaros vault",mat["daedalus_TPM__ikaros_vault"])]
    pos=[(30,52),(70,52),(70,20),(30,20)]
    for i,((tpmn,vlt,res),(x,y)) in enumerate(zip(cells,pos)):
        ca=ease((l-0.6-0.7*i)/0.7)*a
        if ca<=0: continue
        unlocked = res["verdict"]=="UNLOCK"
        col=OK if unlocked else BAD
        ax.add_patch(FancyBboxPatch((x-16,y-9),32,18,boxstyle="round,pad=0.5",fc="#161b22",ec=col,lw=3,alpha=ca))
        txt(x,y+5,f"{tpmn}  ◂  {vlt}",fontsize=14,color=FG,alpha=ca)
        txt(x,y-0.5,("UNLOCK ✓" if unlocked else "REFUSED ✕"),fontsize=22,fontweight="bold",color=col,alpha=ca)
        txt(x,y-5.5,("own die" if unlocked else "foreign die — integrity mismatch"),fontsize=11,color=MUT,alpha=ca)

def draw_close(l,d):
    a=fade_alpha(l,d)
    txt(50,68,"Two honest tiers",fontsize=34,fontweight="bold",color=FG,alpha=a)
    txt(50,57,"✓ substrate-dependence (science)   +   ✓ uncopyability (TPM, security)",fontsize=19,color=OK,alpha=a)
    txt(50,44,"Verified on real hardware — two AMD Strix Halo boxes",fontsize=15,color=MUT,alpha=a)
    if l>1.5:
        na=ease((l-1.5)/0.8)*a
        ax.add_patch(FancyBboxPatch((26,28),48,8,boxstyle="round,pad=0.5",fc="#161b22",ec=AC,lw=2,alpha=na))
        txt(50,32,"github.com/Heigke/FabricCrypt  ·  methods, data, honest limits",fontsize=15,color=AC,alpha=na)

DRAW={"title":draw_title,"fleet":draw_fleet,"finger":draw_finger,"lock":draw_lock,"tpm":draw_tpm,"close":draw_close}

def render(fr):
    ax.clear(); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    tnow=fr/FPS
    for name,a0,a1 in bounds:
        if a0<=tnow<a1:
            DRAW[name](tnow-a0,a1-a0); break
    # persistent footer
    ax.text(99,1.5,"Eric Bergvall · research agent",fontsize=10,color=MUT,ha="right",va="bottom")
    return []

if __name__=="__main__":
    anim=animation.FuncAnimation(fig,render,frames=NFR,interval=1000/FPS,blit=False)
    outp=OUT/"H7_embodiment.mp4"
    anim.save(str(outp),writer=animation.FFMpegWriter(fps=FPS,bitrate=4000,
        extra_args=["-pix_fmt","yuv420p"]))
    print(f"saved {outp}  ({TOTAL:.0f}s, {NFR} frames)")
