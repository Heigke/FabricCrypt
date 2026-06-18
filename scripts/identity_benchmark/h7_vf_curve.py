"""H7 V/F-curve silicon fingerprint — the deepest physical handle.

Found in matched-config data: a few PM channels in the VOLTAGE range (~0.9-1.06 V) carry a STABLE
per-die offset at fixed freq+temp (idx~720: daedalus 1.057V vs ikaros 0.902V). Voltage-required-per-
frequency is set by per-die fused AVFS (silicon quality), NOT by configuration — UNLESS a manual
undervolt offset is applied (which would be a flat shift). The SHAPE/curvature of V(f) is hard to fake
with config: it reflects the die's actual fused V/F bin.

Method: pin governor=performance, boost off; for each target freq, pin scaling_max=min, hold temp to
TSET (adaptive heater), run identical load, record the voltage-range PM channels + freq + power + temp.
Output: V(f) curve per run. compare(): is V(f) stable within die (r1 vs r2) but different between dies,
and is the difference FREQUENCY-DEPENDENT (silicon binning) vs a flat offset (possible config undervolt)?

Env: RUNTAG(r1) TSET(55) FREQS("1200,1600,2000,2400,2800,3200") REPS(8) H7_OUT. Root.
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
RUNTAG=os.environ.get("RUNTAG","r1"); TSET=float(os.environ.get("TSET","55"))
FREQS=[int(x) for x in os.environ.get("FREQS","1200,1600,2000,2400,2800,3200").split(",")]
REPS=int(os.environ.get("REPS","8")); TMAX=float(os.environ.get("TMAX","90"))

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
def cur_mhz():
    try: return int(rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq","0"))//1000
    except: return 0
_LOAD="import numpy as np,sys,time\nt=float(sys.argv[1]);a=np.random.rand(384,384)\ne=time.time()+t\nwhile time.time()<e: a=(a@a)%7.0+0.1"
def load(ms,ncore):
    ps=[subprocess.Popen(["taskset","-c",str(c),sys.executable,"-c",_LOAD,str(ms/1000.0)],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) for c in range(ncore)]
    for p in ps: p.wait()
def heat(tset):
    t=zone0()
    if t>=TMAX: return
    if t<tset-4: load(200,NCPU)
    elif t<tset-0.7: load(80,8)
    elif t>tset+1.2: time.sleep(0.4)

def enroll():
    og=rd(GOV[0]); ob=rd(BOOST,"1")
    cmax=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"); cmin=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq")
    wr(BOOST,"0")
    for c in range(NCPU): wr(GOV[c],"performance")
    try:
        pm0=read_pm()
        if pm0 is None: print("no pm_table"); return
        Nf=len(pm0)
        # voltage-range channel mask: median value in [0.5,1.4]
        warm=[]
        for _ in range(20): warm.append(read_pm()); time.sleep(0.02)
        med=np.nanmedian(np.array(warm),0)
        vmask=np.where((med>=0.5)&(med<=1.4))[0]
        rows=[]   # (freq_target, rep, cur_mhz, temp, power_proxy, V[vmask...])
        for fmhz in FREQS:
            for c in range(NCPU): wr(SMAX[c],fmhz*1000); wr(SMIN[c],fmhz*1000)
            time.sleep(0.3)
            t0=time.time()
            while zone0()<TSET-1 and time.time()-t0<60: heat(TSET)
            for r in range(REPS):
                g=0
                while zone0()<TSET-0.7 and g<10: heat(TSET); g+=1
                while zone0()>TSET+1.2: time.sleep(0.3)
                load(60, NCPU)                       # identical full load while measuring
                v=read_pm();
                if v is None or len(v)<Nf: continue
                rows.append([fmhz, r, cur_mhz(), zone0()]+list(v[vmask]))
            print(f"  f={fmhz}MHz cur={cur_mhz()}MHz T={zone0():.1f}C Vmean={np.nanmean(read_pm()[vmask]):.4f}",flush=True)
        R=np.array(rows)
        OUT.mkdir(parents=True,exist_ok=True)
        p=OUT/f"vf_{HOST}_{RUNTAG}.npz"
        np.savez_compressed(p,rows=R,vmask=vmask,freqs=np.array(FREQS),tset=TSET)
        print(f"[{HOST}] saved {p.name} rows={R.shape} vchans={len(vmask)}",flush=True)
    finally:
        for c in range(NCPU): wr(SMAX[c],cmax); wr(SMIN[c],cmin); wr(GOV[c],og or "powersave")
        if ob: wr(BOOST,ob)
        print(f"[restore] gov0={rd(GOV[0])}",flush=True)

def compare():
    fs=sorted(glob.glob(str(OUT/"vf_*.npz")))
    runs={}
    for f in fs:
        host,tag=Path(f).stem.split("vf_")[1].rsplit("_",1); runs[(host,tag)]=np.load(f)
    hosts=sorted(set(h for h,_ in runs))
    if len(hosts)<2: print(f"need 2 dies, have {hosts}"); return
    A,B=hosts[0],hosts[1]
    # common voltage channels
    vm=set(runs[(A,'r1')]["vmask"].tolist()) if (A,'r1') in runs else set()
    for k,d in runs.items(): vm&=set(d["vmask"].tolist())
    vm=sorted(vm)
    freqs=sorted(set(int(x) for d in runs.values() for x in d["rows"][:,0]))
    print(f"dies={hosts} runs={sorted(runs)}  common V-chans={len(vm)}  freqs={freqs}")
    # build V(f) per run: for each run, for each freq, mean over reps of each vchan; pick the vchan
    # with the largest, most temp-stable freq-dependence as 'the' core voltage
    def vf(d, ch):
        R=d["rows"]; cols={c:i for i,c in enumerate(d["vmask"])}
        if ch not in cols: return None
        j=4+cols[ch]; out=[]
        for f in freqs:
            sel=R[R[:,0]==f]; out.append(np.nanmean(sel[:,j]) if len(sel) else np.nan)
        return np.array(out)
    # choose channel: max variance across freq (a real V/F curve), averaged over runs
    best=None;bestv=-1
    for ch in vm:
        vs=[vf(runs[k],ch) for k in runs]; vs=[v for v in vs if v is not None and np.isfinite(v).all()]
        if len(vs)<4: continue
        v=np.nanmean(vs,0);
        if np.nanstd(v)>bestv: bestv=np.nanstd(v); best=ch
    if best is None: print("no usable V/F channel"); return
    print(f"\nchosen V/F channel idx={best} (largest freq-dependence)")
    print(f"  {'freq':>6} "+ " ".join(f"{h[:3]}_{t}" for h in hosts for t in ['r1','r2'] if (h,t) in runs))
    curves={k:vf(runs[k],best) for k in runs}
    for i,f in enumerate(freqs):
        print(f"  {f:6d} "+" ".join(f"{curves[k][i]:7.4f}" for k in runs))
    def L2(a,b): return float(np.nanmean(np.abs(a-b)))
    intra=np.mean([L2(curves[(h,'r1')],curves[(h,'r2')]) for h in hosts if (h,'r1') in runs and (h,'r2') in runs])
    inter=np.mean([L2(curves[(A,a)],curves[(B,b)]) for a in['r1','r2'] for b in['r1','r2'] if (A,a) in runs and (B,b) in runs])
    print(f"\n  V(f) curve: INTRA(r1vr2)={intra:.4f}V  INTER(A vs B)={inter:.4f}V  ratio={inter/(intra+1e-9):.2f}")
    # flat-offset vs shape: remove per-curve mean, compare shape
    cA=np.nanmean([curves[(A,t)] for t in['r1','r2'] if (A,t) in runs],0)
    cB=np.nanmean([curves[(B,t)] for t in['r1','r2'] if (B,t) in runs],0)
    flat=np.nanmean(cA-cB)
    shapeA=cA-np.nanmean(cA); shapeB=cB-np.nanmean(cB)
    shape_diff=np.nanmean(np.abs(shapeA-shapeB))
    print(f"  flat offset (mean V diff) = {flat*1000:.1f} mV  | SHAPE diff (offset-removed) = {shape_diff*1000:.2f} mV")
    print("  -> large SHAPE diff = frequency-dependent = SILICON binning (not a flat config undervolt).")
    print("  -> shape~0, only flat offset = consistent with a config undervolt; need V/F at more points / SVI raw to disambiguate.")

if __name__=="__main__":
    (compare if len(sys.argv)>1 and sys.argv[1]=="compare" else enroll)()
