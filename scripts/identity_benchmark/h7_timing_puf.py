"""H7 timing/jitter PUF — cross-die test, fully controlled. Tests the hypothesis that per-die
transistor-speed (dopant) variation shows up in COMPUTATION TIMING/JITTER, not amplitude telemetry.

Controls (lessons learned): DVFS pinned (governor=performance, boost off, scaling_max=min @ PIN_MHZ),
temp-soaked to TSET, per-core fixed ALU kernel timed with rdtsc, APERF/MPERF effective-freq read.
At a FIXED requested freq, TSC-ticks-per-block ∝ 1/(delivered freq); per-core deviations + jitter tail
+ aperf ratio form a per-core fingerprint. REPS passes -> intra (stability/positive control).
Compare ikaros vs daedalus: is the per-core PATTERN (z-scored within die) stable intra but different inter?

enroll: build kernel, pin, soak, sweep cores x REPS, save features.  compare: intra vs inter.
Env: RUNTAG(r1) TSET(55) PIN_MHZ(2200) NB(4000) OPS(20000) REPS(3) H7_OUT. Root required.
"""
from __future__ import annotations
import os, sys, time, socket, subprocess, glob
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
OUT = Path(os.environ["H7_OUT"]) if os.environ.get("H7_OUT") else \
      Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
HERE = Path(__file__).resolve().parent
ZONE_T = Path("/sys/class/thermal/thermal_zone0/temp")
NCPU = os.cpu_count() or 1
GOV=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor" for c in range(NCPU)]
SMAX=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq" for c in range(NCPU)]
SMIN=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_min_freq" for c in range(NCPU)]
BOOST="/sys/devices/system/cpu/cpufreq/boost"
RUNTAG=os.environ.get("RUNTAG","r1"); TSET=float(os.environ.get("TSET","55"))
PIN_MHZ=int(os.environ.get("PIN_MHZ","2200")); NB=int(os.environ.get("NB","4000"))
OPS=int(os.environ.get("OPS","20000")); REPS=int(os.environ.get("REPS","3"))

def temp_c():
    try: return int(ZONE_T.read_text())/1000.0
    except: return 0.0
def rd(p,d=""):
    try: return Path(p).read_text().strip()
    except: return d
def wr(p,v):
    try: Path(p).write_text(str(v)); return True
    except: return False
def save_state(): return {"gov":[rd(p) for p in GOV],"smax":[rd(p) for p in SMAX],"smin":[rd(p) for p in SMIN],"boost":rd(BOOST)}
def restore(st):
    if st["boost"]: wr(BOOST,st["boost"])
    for c in range(NCPU):
        if st["smax"][c]: wr(SMAX[c],st["smax"][c])
        if st["smin"][c]: wr(SMIN[c],st["smin"][c])
        if st["gov"][c]: wr(GOV[c],st["gov"][c])
def pin(khz):
    for c in range(NCPU): wr(GOV[c],"performance"); wr(SMAX[c],khz); wr(SMIN[c],khz)
def soak(tset, lo_margin=2.0, timeout=120):
    t0=time.time()
    while time.time()-t0<timeout:
        t=temp_c()
        if t<=tset: return
        time.sleep(1.0)


def enroll():
    binp=HERE/"h7_timing_kernel"
    subprocess.run(["gcc","-O2","-o",str(binp),str(HERE/"h7_timing_kernel.c")],check=True)
    subprocess.run(["modprobe","msr"],check=False)
    st=save_state()
    try:
        wr(BOOST,"0"); pin(PIN_MHZ*1000); time.sleep(0.5)
        print(f"[{HOST}] pinned {PIN_MHZ}MHz cur={rd('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')} "
              f"gov={rd(GOV[0])} boost={rd(BOOST)} temp={temp_c():.0f}C",flush=True)
        cores=list(range(0,NCPU,2))  # physical cores (skip SMT siblings)
        # features per (rep, core): [p10, med, p90, var, nclip, aperf_ratio]
        feats=np.zeros((REPS,len(cores),6))
        for r in range(REPS):
            soak(TSET)
            for ci,c in enumerate(cores):
                out=subprocess.run([str(binp),str(c),str(NB),str(OPS)],capture_output=True,text=True,check=True).stdout.split()
                feats[r,ci]=[float(x) for x in out[1:7]]
            print(f"  rep{r}: med(ticks) mean={feats[r,:,1].mean():.0f} "
                  f"aperf_ratio mean={feats[r,:,5].mean():.4f} temp={temp_c():.0f}C",flush=True)
        OUT.mkdir(parents=True,exist_ok=True)
        p=OUT/f"timpuf_{HOST}_{RUNTAG}.npz"
        np.savez_compressed(p,feats=feats,cores=np.array(cores),pin_mhz=PIN_MHZ,ops=OPS,nb=NB)
        print(f">>> saved {p.name}  shape={feats.shape}",flush=True)
    finally:
        restore(st); print(f"[restore] gov0={rd(GOV[0])} boost={rd(BOOST)}",flush=True)


def _zpattern(v):  # z-score the per-core pattern within a die (removes absolute offset = config)
    return (v-v.mean())/(v.std()+1e-9)
def cosv(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def compare():
    fs=sorted(glob.glob(str(OUT/f"timpuf_*_{RUNTAG}.npz")))
    hosts=[Path(f).stem.split("timpuf_")[1].rsplit("_",1)[0] for f in fs]
    if len(fs)<2: print("need 2 dies"); return
    print(f"dies: {hosts}",flush=True)
    D={h:np.load(f)["feats"] for h,f in zip(hosts,fs)}
    A,B=hosts[0],hosts[1]
    # use median-ticks per core (feature idx 1) and aperf ratio (idx 5) as the per-core fingerprint
    for name,idx in [("median-ticks",1),("aperf-ratio",5),("jitter-var",3)]:
        a=D[A][:,:,idx]; b=D[B][:,:,idx]   # [REPS, ncores]
        # INTRA: pattern correlation between reps of same die (z-scored per-core pattern)
        def intra(x):
            cs=[]
            for i in range(len(x)):
                for j in range(i+1,len(x)):
                    cs.append(cosv(_zpattern(x[i]),_zpattern(x[j])))
            return np.mean(cs) if cs else float('nan')
        ia,ib=intra(a),intra(b)
        # INTER: pattern correlation between die means
        inter=cosv(_zpattern(a.mean(0)),_zpattern(b.mean(0)))
        print(f"\n[{name}] per-core PATTERN (z-scored, config-removed):",flush=True)
        print(f"   INTRA {A}={ia:+.3f}  INTRA {B}={ib:+.3f}  (stability/positive control)",flush=True)
        print(f"   INTER {A}vs{B}={inter:+.3f}",flush=True)
        verdict = (min(ia,ib)>0.5 and inter < min(ia,ib)-0.2)
        print(f"   -> {'DIE-SPECIFIC pattern (intra>>inter, stable)' if verdict else 'NOT die-specific / unstable' }",flush=True)


if __name__=="__main__":
    (compare if len(sys.argv)>1 and sys.argv[1]=="compare" else enroll)()
