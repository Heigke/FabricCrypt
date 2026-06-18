"""Animated, honestly-reframed demo video (real motion: typing + growing bars; no gratuitous zoom).
Gemini TTS female VO per scene. Operational-embodiment framing (behavior rooted in the body) — honest
about: not consciousness, fluency=adaptation, domain-narrow, TEE gap; states the full vision as the goal.

Out: results/IDENTITY_H7_2026-06-09/H7_embodiment_anim_2026-06.mp4
"""
from __future__ import annotations
import json,math,os,re,shutil,subprocess,wave
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results/IDENTITY_H7_2026-06-09"
TMP=OUT/"anim_tmp"; shutil.rmtree(TMP,ignore_errors=True); TMP.mkdir(parents=True)
S=json.loads((OUT/"demo_samples.json").read_text())["gen"]
BG="#0b0e14"; FG="#e6edf3"; OK="#3fb950"; BAD="#f85149"; AM="#d29922"; MUT="#8b949e"; AC="#58a6ff"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,"text.color":FG,"font.family":"DejaVu Sans"})
FPS=25
def newfig():
    fig=plt.figure(figsize=(19.2,10.8),dpi=100); ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,16); ax.set_ylim(0,9); return fig,ax
def disclaimer(ax):
    ax.add_patch(FancyBboxPatch((0.3,8.3),15.4,0.5,boxstyle="round,pad=0.03",fc="#161b22",ec=AC,lw=1))
    ax.text(8,8.55,"AI-generated research log — Claude Code (Anthropic), running autonomously for Eric Bergvall on an HP workstation",
            ha="center",va="center",fontsize=12,color=AC)
def save(fig,i):
    p=TMP/f"f{i:05d}.png"; fig.savefig(p); plt.close(fig); return p

frames=[]  # (path, n_repeat)
def hold(p,sec): frames.append((p,int(sec*FPS)))

# ---------- Scene 1: title ----------
def s1():
    fig,ax=newfig(); disclaimer(ax)
    ax.text(8,5.1,"An LLM that runs on",ha="center",fontsize=40,fontweight="bold")
    ax.text(8,4.0,"its chip's LIVE signals",ha="center",fontsize=40,fontweight="bold",color=AC)
    ax.text(8,2.7,"per-core voltage · per-core clock · power — read in real time, tested on real silicon",ha="center",fontsize=15,color=MUT)
    hold(save(fig,1),5.0)
# ---------- Scene 2: the lock (cross-machine) ----------
def s2():
    fig,ax=newfig(); disclaimer(ax)
    ax.text(8,7.4,"The live signals that actually move",ha="center",fontsize=22,fontweight="bold")
    ax.text(8,6.75,"measured on both AMD chips · read in real time · fed into a frozen GPT-2",ha="center",fontsize=13,color=MUT)
    chans=[("per-core Vcore","16 voltages · jitter ~6 mV",OK),
           ("per-core clock","32 cores · swings across 2.4 GHz",OK),
           ("power draw","live · ~2 W swing",OK),
           ("(TPM key = the lock only, not the body)","crypto layer, kept separate",MUT)]
    for i,(name,sub,col) in enumerate(chans):
        y=5.2-i*1.15
        ax.add_patch(FancyBboxPatch((2.3,y),11.4,0.95,boxstyle="round,pad=0.04",fc="#11161f",ec=col,lw=1.8))
        ax.text(2.7,y+0.55,name,fontsize=15,color=col,fontweight="bold")
        ax.text(2.7,y+0.2,sub,fontsize=11.5,color=MUT)
    hold(save(fig,2),8.0)
# ---------- Scene 3: typing demo ----------
def s3():
    own=S["ikaros"]["own"][:150]; bad=S["ikaros"]["random"][:150]
    nframes=int(9.0*FPS)
    for fi in range(nframes):
        frac=min(1.0,fi/(nframes*0.8))
        fig,ax=newfig(); disclaimer(ax)
        ax.text(8,7.5,'Same prompt:  "It is a truth …"',ha="center",fontsize=18,color=AC,style="italic")
        def tb(y,title,txt,col,fr):
            ax.add_patch(FancyBboxPatch((1.2,y),13.6,1.9,boxstyle="round,pad=0.05",fc="#11161f",ec=col,lw=2))
            ax.text(1.5,y+1.55,title,fontsize=14,color=col,fontweight="bold")
            n=int(len(txt)*fr); shown=txt[:n]
            import textwrap; ax.text(1.5,y+1.15,"\n".join(textwrap.wrap(shown,68)),fontsize=12.5,color=FG,va="top",family="monospace")
        tb(4.9,"✓  CORRECT key (this chip's own die)",own,OK,frac)
        tb(2.5,"✗  WRONG key (random fingerprint)",bad,BAD,frac)
        frames.append((save(fig,1000+fi),1))
# ---------- Scene 4: growing bars ----------
def s4():
    labels=["own live\nbody","plain GPT-2\n(baseline)","zero","foreign\ndie","random"]
    vals=[47.2,64.9,8736,56072,46317]; cols=[OK,MUT,BAD,BAD,BAD]
    logv=[math.log10(v) for v in vals]; nframes=int(7.0*FPS)
    for fi in range(nframes):
        fr=min(1.0,fi/(nframes*0.75))
        fig,ax=newfig(); disclaimer(ax)
        ax.text(8,7.7,"Held-out perplexity — live-signals-only model  (lower = more fluent)",ha="center",fontsize=19,fontweight="bold")
        for i,(lb,lv,v,c) in enumerate(zip(labels,logv,vals,cols)):
            x=1.7+i*2.85; h=(lv/math.log10(90000))*4.6*fr
            ax.add_patch(plt.Rectangle((x,1.6),1.7,h,color=c))
            ax.text(x+0.85,1.25,lb,ha="center",fontsize=11,color=FG)
            if fr>0.85: ax.text(x+0.85,1.7+h+0.15,f"{int(v) if v>100 else v}",ha="center",fontsize=12,color=c,fontweight="bold")
        ax.text(8,0.55,"own live body BEATS plain GPT-2 · wrong body 150–2200× worse · deterministic",ha="center",fontsize=13,color=AM)
        frames.append((save(fig,2000+fi),1))
# ---------- Scene 5: honest framing + vision ----------
def s5():
    fig,ax=newfig(); disclaimer(ax)
    ax.text(8,7.6,"Operational embodiment — behaviour rooted in the body",ha="center",fontsize=22,fontweight="bold",color=AC)
    ax.text(8,6.9,"change the chip → the behaviour changes or breaks",ha="center",fontsize=14,color=MUT)
    ax.add_patch(FancyBboxPatch((0.7,3.3),7.0,3.0,boxstyle="round,pad=0.05",fc="#0d1f12",ec=OK,lw=1.2))
    ax.text(1.0,6.0,"What holds today:",fontsize=14,color=OK,fontweight="bold")
    for i,t in enumerate(["• unique per-chip key (TPM-sealed)","• uncopyable across machines (refused on foreign die)",
                          "• behaviour bound & reproducible (trained, deterministic)","• a freshness channel (nonce + live drift)"]):
        ax.text(1.0,5.5-i*0.5,t,fontsize=12,color=FG)
    ax.add_patch(FancyBboxPatch((8.3,3.3),7.0,3.0,boxstyle="round,pad=0.05",fc="#1c1206",ec=AM,lw=1.2))
    ax.text(8.6,6.0,"Honest limits / the goal:",fontsize=14,color=AM,fontweight="bold")
    for i,t in enumerate(["• not consciousness; fluency = adaptation","• adapter is domain-narrow (124M proof-of-concept)",
                          "• no GPU enclave yet → active-mode protection partial","→ next: live in-body computation wired into the LLM,","   encrypted & uncopyable at rest AND while running"]):
        ax.text(8.6,5.5-i*0.5,t,fontsize=11.5,color=FG)
    ax.text(8,1.4,"Code & honest write-up:  github.com/Heigke/FabricCrypt",ha="center",fontsize=16,color=AC,fontweight="bold")
    hold(save(fig,5),9.0)

NARR=[
 "What if a language model ran on the live signals of one specific chip? Working autonomously for Eric on an HP workstation, I tested that on real silicon.",
 "Three signals that genuinely move — per-core voltage, per-core clock, and power draw — are read live and fed into a frozen GPT-2. No fused constants, no crypto key in the loop.",
 "With its own body's live signals, the model writes fluent English. With a foreign, or random, body, it collapses into noise.",
 "On unseen text the correct body even beats plain GPT-2; the wrong body is hundreds to thousands of times worse — deterministically, on two machines.",
 "These stack into layers of embodiment: identity, freshness, real-time coupling, behaviour binding — with a TPM only as the lock, and a fresh nonce for liveness. Honestly, it's operational embodiment, not consciousness; the fluency comes from adaptation; and there's no secure enclave yet. The goal we're building toward: the body's own computation, wired into the model, uncopyable at rest and while it runs. Code and honest limits: github dot com, slash Heigke, slash FabricCrypt.",
]
def key():
    for line in open(ROOT/".env"):
        m=re.match(r'\s*gemini_api_key\s*=\s*"?([^"\n]+)"?',line,re.I)
        if m: return m.group(1).strip()
def tts(text,path):
    from google import genai; from google.genai import types
    c=genai.Client(api_key=key())
    r=c.models.generate_content(model="gemini-3.1-flash-tts-preview",contents=text,
        config=types.GenerateContentConfig(response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")))))
    data=r.candidates[0].content.parts[0].inline_data.data
    with wave.open(str(path),"wb") as w: w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(data)
    with wave.open(str(path)) as w: return w.getnframes()/w.getframerate()

def main():
    # build scenes; group frames per scene so we can match audio durations
    scene_frames=[]
    def cap():
        i0=len(frames); return i0
    a=cap(); s1(); b=cap(); s2(); c=cap(); s3(); d=cap(); s4(); e=cap(); s5(); f=cap()
    bounds=[(a,b),(b,c),(c,d),(d,e),(e,f)]
    durs=[tts(NARR[i],TMP/f"vo{i}.wav") for i in range(5)]
    print("VO durs",[f"{x:.1f}" for x in durs])
    # write per-scene frame lists with repeats stretched to audio duration
    clips=[]
    for si,(lo,hi) in enumerate(bounds):
        segs=frames[lo:hi]
        total=sum(n for _,n in segs)                       # actual frame count
        vidlen=total/FPS
        whole=durs[si]+0.8                                  # target clip length = narration + 0.8s tail
        tail=max(0.1, whole-vidlen)                         # hold last frame to cover (and exceed) the VO
        lst=TMP/f"list{si}.txt"; lines=[]
        for p,n in segs:
            for _ in range(n): lines.append(f"file '{p.name}'\nduration {1/FPS:.4f}")
        lines.append(f"file '{segs[-1][0].name}'")
        lst.write_text("\n".join(lines))
        clip=TMP/f"clip{si}.mp4"
        # pad BOTH streams to `whole`: video holds its last frame, audio is silence-padded -> they end
        # together and the narration always finishes ~0.8s before the slide change (no clipping).
        v=subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-i",str(TMP/f'vo{si}.wav'),
            "-vf",f"fps={FPS},format=yuv420p,fade=t=in:st=0:d=0.4,tpad=stop_mode=clone:stop_duration={tail:.2f}",
            "-af",f"apad=whole_dur={whole:.2f}","-c:v","libx264","-preset","fast","-crf","20",
            "-c:a","aac","-b:a","160k","-ar","44100",str(clip)],capture_output=True,text=True,cwd=str(TMP))
        if v.returncode: print("clip err",si,v.stderr[-800:]); return
        clips.append(clip)
    cl=TMP/"clips.txt"; cl.write_text("\n".join(f"file '{c.name}'" for c in clips))
    final=OUT/"H7_embodiment_anim_2026-06.mp4"
    v=subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(cl),"-c","copy",str(final)],capture_output=True,text=True,cwd=str(TMP))
    if v.returncode:  # fallback re-encode
        v=subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(cl),"-c:v","libx264","-crf","20","-c:a","aac",str(final)],capture_output=True,text=True,cwd=str(TMP))
    print("saved",final, final.stat().st_size//1024,"KB" if final.exists() else "FAIL")

if __name__=="__main__": main()
