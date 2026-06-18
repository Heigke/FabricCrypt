"""H7 matched-config matched-temp die test — strip every nuisance, see if silicon identity survives.

Both dies now reach ~100C under full load => temp ranges OVERLAP (~45-100C); matched-temp IS possible.
This run nails down ALL three confounds at once:
  - CONFIG: same governor=performance, same pinned freq, boost off  (set here)
  - CONFIG-CONSTANTS: channels that never vary within a run = firmware/BIOS layout values -> DROPPED in compare
  - TEMP: both held to the SAME TSET by bang-bang gentle load
  - HONEST NULL: TWO separate runs per die (r1,r2), time-separated -> intra null captures run-to-run 1/f drift
                 (the previous half-of-one-run null hid drift and inflated the ratio to 283e9).

enroll: pin, warm to TSET, hold, take M snapshots of the FULL pm vector (fixed indices). save full matrix.
compare: drop static channels; standardize; per-run mean vector; ask inter-die distance vs intra-die (run-to-run).
   If at matched temp+config, with constants stripped, INTER >> INTRA robustly -> a real die handle survives.
   If INTER ~ INTRA -> no die identity even with every nuisance controlled (the honest negative).

Env: RUNTAG(r1) TSET(58) M(400) PIN_MHZ(2800) PROBE_MS(40) H7_OUT. Root.
"""
from __future__ import annotations
import os, sys, time, socket, glob, subprocess
from pathlib import Path
import numpy as np

HOST=socket.gethostname()
OUT=Path(os.environ["H7_OUT"]) if os.environ.get("H7_OUT") else Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
PM=Path("/sys/kernel/ryzen_smu_drv/pm_table")
NCPU=os.cpu_count() or 1
GOV=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor" for c in range(NCPU)]
SMAX=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq" for c in range(NCPU)]
SMIN=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_min_freq" for c in range(NCPU)]
BOOST="/sys/devices/system/cpu/cpufreq/boost"
RUNTAG=os.environ.get("RUNTAG","r1"); TSET=float(os.environ.get("TSET","58"))
M=int(os.environ.get("M","400")); PIN_MHZ=int(os.environ.get("PIN_MHZ","2800"))
PROBE_MS=float(os.environ.get("PROBE_MS","40")); TMAX=float(os.environ.get("TMAX","88"))

def rd(p,d=""):
    try: return Path(p).read_text().strip()
    except: return d
def wr(p,v):
    try: Path(p).write_text(str(v)); return True
    except: return False
def zone0():
    try: return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000.0
    except: return 0.0
def read_pm():
    try:
        b=PM.read_bytes(); n=(len(b)//4)*4
        return np.frombuffer(b[:n],dtype=np.float32).astype(np.float64)
    except: return None
_PROBE="import numpy as np,sys,time\nt=float(sys.argv[1]);a=np.random.rand(512,512)\ne=time.time()+t\nwhile time.time()<e: a=(a@a)%7.0+0.1"
def probe(ms, ncore=6):
    ps=[subprocess.Popen(["taskset","-c",str(c),sys.executable,"-c",_PROBE,str(ms/1000.0)],
                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) for c in range(ncore)]
    for p in ps: p.wait()
def heat_step():
    """adaptive heater: full-core short kick when far below TSET (works on well-cooled daedalus),
    gentle when near, idle-cool when over. Keeps both dies pinned to the SAME TSET."""
    t=zone0()
    if t>=TMAX: return
    if t<TSET-4:   probe(200, ncore=NCPU)     # strong kick (short, re-check between)
    elif t<TSET-0.7: probe(80, ncore=8)       # gentle approach
    elif t>TSET+1.2: time.sleep(0.4)          # overshoot: idle-cool

def enroll():
    og=rd(GOV[0],"powersave"); ob=rd(BOOST,"1")
    cmax=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"); cmin=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq")
    wr(BOOST,"0")
    for c in range(NCPU): wr(GOV[c],"performance"); wr(SMAX[c],PIN_MHZ*1000); wr(SMIN[c],PIN_MHZ*1000)
    try:
        pm0=read_pm()
        if pm0 is None: print("no pm_table"); return
        Nf=len(pm0)
        # warm to TSET with adaptive heater (full-core kicks until close)
        t0=time.time()
        while zone0()<TSET-1 and time.time()-t0<300:
            heat_step()
        if zone0()<TSET-3:
            print(f"  WARN: only reached {zone0():.1f}C (target {TSET}); heater too weak",flush=True)
        snaps=np.zeros((M,Nf)); temps=np.zeros(M)
        for m in range(M):
            g=0
            while zone0()<TSET-0.7 and g<20:
                heat_step(); g+=1
            while zone0()>TSET+1.2:
                time.sleep(0.4)
            probe(PROBE_MS, ncore=6)            # identical fixed probe
            v=read_pm()
            if v is None or len(v)!=Nf: v=np.full(Nf,np.nan)
            snaps[m]=v; temps[m]=zone0()
            if m%100==0: print(f"  m={m} T={temps[m]:.1f}C",flush=True)
        OUT.mkdir(parents=True,exist_ok=True)
        p=OUT/f"matched_{HOST}_{RUNTAG}.npz"
        np.savez_compressed(p,snaps=snaps,temps=temps,tset=TSET,pin=PIN_MHZ,nf=Nf)
        print(f"[{HOST}] M={M} Nf={Nf} Tmean={temps.mean():.1f}±{temps.std():.2f}C >>> {p.name}",flush=True)
    finally:
        for c in range(NCPU): wr(SMAX[c],cmax); wr(SMIN[c],cmin); wr(GOV[c],og)
        if ob: wr(BOOST,ob)
        print(f"[restore] gov0={rd(GOV[0])}",flush=True)

def compare():
    fs=sorted(glob.glob(str(OUT/"matched_*.npz")))
    runs={}
    for f in fs:
        stem=Path(f).stem.split("matched_")[1]   # host_tag
        host,tag=stem.rsplit("_",1)
        d=np.load(f); runs[(host,tag)]=(d["snaps"],d["temps"])
    hosts=sorted(set(h for h,_ in runs))
    if len(hosts)<2: print(f"need 2 dies, have {hosts}"); return
    print(f"dies={hosts}  runs={sorted(runs)}",flush=True)
    A,B=hosts[0],hosts[1]
    # align channel count
    Nf=min(s.shape[1] for s,_ in runs.values())
    S={k:(s[:,:Nf]) for k,(s,_) in runs.items()}
    T={k:t for k,(_,t) in runs.items()}
    for k in S: print(f"  {k}: M={len(S[k])} Tmean={T[k].mean():.1f}±{T[k].std():.2f}C",flush=True)
    # static mask: channel std≈0 within EVERY run => config constant -> drop
    stds=np.array([S[k].std(0) for k in S])                 # (runs, Nf)
    dynamic = (stds > 1e-6).all(0)                          # varies in every run
    nd=int(dynamic.sum())
    print(f"  channels: total={Nf}  dynamic(kept)={nd}  static-config(dropped)={Nf-nd}",flush=True)
    if nd<3: print("  too few dynamic channels"); return
    # pooled std over all runs for standardization
    alld=np.vstack([S[k][:,dynamic] for k in S])
    pooled=alld.std(0)+1e-9
    Z={k:(S[k][:,dynamic])/pooled for k in S}
    def mean_vec(k): return Z[k].mean(0)
    def dist(k1,k2): return float(np.linalg.norm(mean_vec(k1)-mean_vec(k2)))
    have=lambda h,t:(h,t) in S
    # honest intra null: run-to-run, same die
    intra=[]
    for h in hosts:
        if have(h,"r1") and have(h,"r2"): intra.append((h,dist((h,"r1"),(h,"r2"))))
    inter=[]
    for ta in ["r1","r2"]:
        for tb in ["r1","r2"]:
            if have(A,ta) and have(B,tb): inter.append(dist((A,ta),(B,tb)))
    print(f"\n  [matched config+temp, config-constants stripped, {nd} dynamic channels]",flush=True)
    for h,d in intra: print(f"   INTRA {h} (r1 vs r2, run-to-run null) = {d:.3f}",flush=True)
    im=np.mean([d for _,d in intra]) if intra else float('nan')
    inm=np.mean(inter) if inter else float('nan')
    print(f"   INTER {A} vs {B} (mean over run pairs) = {inm:.3f}",flush=True)
    ratio=inm/(im+1e-9)
    print(f"\n   ratio INTER/INTRA = {ratio:.2f}",flush=True)
    print("   -> ratio >> 1 (e.g. >3) : a die-separating residual SURVIVES matched config+temp = real handle (then silicon-vs-residual-firmware).",flush=True)
    print("   -> ratio ~ 1            : run-to-run drift is as big as die difference = NO extractable die identity even fully matched (honest negative).",flush=True)

if __name__=="__main__":
    (compare if len(sys.argv)>1 and sys.argv[1]=="compare" else enroll)()
