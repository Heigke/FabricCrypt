"""H7 per-core voltage fingerprint — test (3) Vcore confirmation + test (2) run-to-run null.
Fixed channel block idx 756..771 (16 per-core voltages). NO per-run reselection (fixes the buggy VF sweep).

enroll:
  PART A vfreq: pin each freq in FREQS, read mean per-core vector under load -> does mean V rise with freq? (=Vcore)
  PART B repeat: REPS independent mini-enrollments at PIN_MHZ/TSET, each = mean 16-vec over WIN snapshots,
                 re-warmed between -> a true run-to-run set (captures drift, unlike snapshot-bootstrap).
compare: (3) Vcore monotonicity; (2) per-core SHAPE corr: run-to-run INTRA vs INTER, permutation p-value.
         saves fingerprint_{host}.npy = mean z-scored 16-vec (the die key for the LM).
Env: RUNTAG TSET(55) PIN_MHZ(2800) REPS(10) WIN(80) FREQS H7_OUT. Root.
"""
from __future__ import annotations
import os, sys, time, socket, glob, subprocess
from pathlib import Path
import numpy as np
HOST=socket.gethostname()
OUT=Path(os.environ["H7_OUT"]) if os.environ.get("H7_OUT") else Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
PM=Path("/sys/kernel/ryzen_smu_drv/pm_table"); NCPU=os.cpu_count() or 1
GOV=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor" for c in range(NCPU)]
SMAX=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq" for c in range(NCPU)]
SMIN=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_min_freq" for c in range(NCPU)]
BOOST="/sys/devices/system/cpu/cpufreq/boost"
RUNTAG=os.environ.get("RUNTAG","r1"); TSET=float(os.environ.get("TSET","55"))
PIN=int(os.environ.get("PIN_MHZ","2800")); REPS=int(os.environ.get("REPS","10")); WIN=int(os.environ.get("WIN","80"))
FREQS=[int(x) for x in os.environ.get("FREQS","1600,2000,2400,2800,3200").split(",")]; TMAX=float(os.environ.get("TMAX","90"))
CORE=slice(756,772)
def rd(p,d=""):
    try: return Path(p).read_text().strip()
    except: return d
def wr(p,v):
    try: Path(p).write_text(str(v)); return True
    except: return False
def zone0():
    try: return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000.0
    except: return 0.0
def pm16():
    try:
        b=PM.read_bytes(); v=np.frombuffer(b[:(len(b)//4)*4],dtype=np.float32).astype(np.float64)
        return v[CORE]
    except: return None
_LOAD="import numpy as np,sys,time\nt=float(sys.argv[1]);a=np.random.rand(384,384)\ne=time.time()+t\nwhile time.time()<e: a=(a@a)%7.0+0.1"
def load(ms,nc):
    ps=[subprocess.Popen(["taskset","-c",str(c),sys.executable,"-c",_LOAD,str(ms/1000.0)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) for c in range(nc)]
    for p in ps: p.wait()
def heat(tset):
    t=zone0()
    if t>=TMAX: return
    if t<tset-4: load(200,NCPU)
    elif t<tset-0.7: load(80,8)
    elif t>tset+1.2: time.sleep(0.4)
def curm():
    return int(rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq","0"))//1000

def enroll():
    og=rd(GOV[0]); ob=rd(BOOST,"1")
    cmax=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"); cmin=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq")
    wr(BOOST,"0")
    for c in range(NCPU): wr(GOV[c],"performance")
    try:
        if pm16() is None: print("no pm_table"); return
        # PART A: Vcore-vs-freq (fixed channel, load on, brief; temp not critical for scaling check)
        vfreq=[]
        for f in FREQS:
            for c in range(NCPU): wr(SMAX[c],f*1000); wr(SMIN[c],f*1000)
            time.sleep(0.4); load(300,NCPU)
            vs=[]
            for _ in range(30): load(40,NCPU); vs.append(pm16())
            mv=np.nanmean(vs,0); vfreq.append([f,curm(),zone0()]+list(mv))
            print(f"  vfreq f={f} cur={curm()} T={zone0():.0f} Vmean={mv.mean():.4f}",flush=True)
        # PART B: run-to-run reps at PIN/TSET
        for c in range(NCPU): wr(SMAX[c],PIN*1000); wr(SMIN[c],PIN*1000)
        reps=[]
        for r in range(REPS):
            t0=time.time()
            while zone0()<TSET-1 and time.time()-t0<40: heat(TSET)
            snaps=[]
            for _ in range(WIN):
                while zone0()<TSET-0.7: heat(TSET)
                while zone0()>TSET+1.2: time.sleep(0.3)
                load(50,NCPU); snaps.append(pm16())
            reps.append([r,zone0()]+list(np.nanmean(snaps,0)))
            print(f"  rep{r} T={zone0():.1f} V16mean={np.nanmean(snaps,0).mean():.4f}",flush=True)
            # cool a touch between reps for independence
            for _ in range(8): time.sleep(0.4)
        OUT.mkdir(parents=True,exist_ok=True)
        p=OUT/f"percore_{HOST}_{RUNTAG}.npz"
        np.savez_compressed(p,vfreq=np.array(vfreq),reps=np.array(reps),tset=TSET,pin=PIN)
        print(f"[{HOST}] saved {p.name} vfreq={len(vfreq)} reps={len(reps)}",flush=True)
    finally:
        for c in range(NCPU): wr(SMAX[c],cmax); wr(SMIN[c],cmin); wr(GOV[c],og or "powersave")
        if ob: wr(BOOST,ob)
        print(f"[restore] gov0={rd(GOV[0])}",flush=True)

def compare():
    fs=sorted(glob.glob(str(OUT/"percore_*.npz")))
    D={}
    for f in fs:
        h,t=Path(f).stem.split("percore_")[1].rsplit("_",1); D[(h,t)]=np.load(f)
    hosts=sorted(set(h for h,_ in D))
    if not hosts: print("no data"); return
    print(f"runs={sorted(D)}")
    # TEST 3: Vcore scaling
    print("\n=== TEST 3: per-core mean V vs frequency (rising => Vcore) ===")
    for k in sorted(D):
        vf=D[k]["vfreq"];
        print(f"  {k}: "+"  ".join(f"{int(r[0])}MHz->{np.mean(r[3:]):.4f}V" for r in vf))
    # TEST 2: run-to-run shape corr
    print("\n=== TEST 2: per-core SHAPE run-to-run INTRA vs INTER (real null) ===")
    def shapes(k):
        reps=D[k]["reps"][:,2:]   # (REPS,16)
        return np.array([(v-v.mean())/(v.std()+1e-9) for v in reps])
    def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    SH={k:shapes(k) for k in D}
    intra=[]
    for h in hosts:
        ks=[k for k in D if k[0]==h]
        vs=np.vstack([SH[k] for k in ks])
        cs=[cos(vs[i],vs[j]) for i in range(len(vs)) for j in range(i+1,len(vs))]
        if cs: intra.append((h,np.mean(cs))); print(f"  INTRA {h}: mean run-to-run corr={np.mean(cs):+.3f} (n_pairs={len(cs)})")
    if len(hosts)>=2:
        A,B=hosts[:2]; va=np.vstack([SH[k] for k in D if k[0]==A]); vb=np.vstack([SH[k] for k in D if k[0]==B])
        inter=[cos(a,b) for a in va for b in vb]
        im=np.mean([x for _,x in intra]); inm=np.mean(inter)
        print(f"  INTER {A} vs {B}: mean corr={inm:+.3f} (n={len(inter)})")
        # permutation: shuffle die labels across all reps
        allv=np.vstack([va,vb]); na=len(va); rng=np.random.default_rng(0); ge=0;NP=2000
        obs=im-inm
        for _ in range(NP):
            ix=rng.permutation(len(allv)); pa,pb=allv[ix[:na]],allv[ix[na:]]
            ia=[cos(pa[i],pa[j]) for i in range(len(pa)) for j in range(i+1,len(pa))]
            ib=[cos(pb[i],pb[j]) for i in range(len(pb)) for j in range(i+1,len(pb))]
            inn=[cos(a,b) for a in pa[:6] for b in pb[:6]]
            if (np.mean(ia+ib)-np.mean(inn))>=obs: ge+=1
        print(f"  separation INTRA-INTER={obs:+.3f}  permutation p={(ge+1)/(NP+1):.4f}")
    # save per-die fingerprint (mean z-scored 16-vec) for the LM
    for h in hosts:
        ks=[k for k in D if k[0]==h]; reps=np.vstack([D[k]["reps"][:,2:] for k in ks])
        fp=reps.mean(0); fpz=(fp-fp.mean())/(fp.std()+1e-9)
        np.save(OUT/f"fingerprint_{h}.npy", fpz)
        print(f"  saved fingerprint_{h}.npy (z-scored 16-vec)")

if __name__=="__main__":
    (compare if len(sys.argv)>1 and sys.argv[1]=="compare" else enroll)()
