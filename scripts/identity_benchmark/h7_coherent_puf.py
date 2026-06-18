"""H7 coherent-integration PUF — the radar trick: can massive averaging pull a STABLE per-die residual
out of run-to-run noise, even from low-resolution firmware telemetry?

Mechanism (matched-filter / coherent averaging): a STABLE signal averages coherently while zero-mean noise
drops as 1/sqrt(M). If a die-specific offset exists in the aggregate PM-table vector, averaging M snapshots
at a LOCKED operating point should make it rise above the (now /sqrt(M)) noise floor.

Controls: DVFS pinned, temperature HELD at TSET (bang-bang regulation), each snapshot = the PM vector right
after an IDENTICAL fixed micro-probe burst. M snapshots saved. Decisive output (compare): d'(M) curve —
  d'_inter(M) = |mean_A - mean_B| / (single-snapshot_std / sqrt(M))   [across dies]
  d'_intra(M) = |meanA_half1 - meanA_half2| / (.../sqrt(M))           [same die, null]
If d'_inter grows ~sqrt(M) and d'_inter >> d'_intra -> a stable difference is extractable by integration
(then config-vs-die must still be argued). If d'_inter plateaus near d'_intra -> buried below quantization/drift.

Env: RUNTAG(r1) TSET(55) M(600) PIN_MHZ(2800) PROBE_MS(40) H7_OUT. Root.
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
M=int(os.environ.get("M","600")); PIN_MHZ=int(os.environ.get("PIN_MHZ","2800"))
PROBE_MS=float(os.environ.get("PROBE_MS","40")); K=24

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


def enroll():
    og=rd(GOV[0],"powersave"); ob=rd(BOOST,"1")
    cmax=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"); cmin=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq")
    wr(BOOST,"0")
    for c in range(NCPU): wr(GOV[c],"performance"); wr(SMAX[c],PIN_MHZ*1000); wr(SMIN[c],PIN_MHZ*1000)
    try:
        pm0=read_pm()
        if pm0 is None: print("no pm_table"); return
        # warm to TSET
        t0=time.time()
        while zone0()<TSET-1 and time.time()-t0<90: probe(60)
        # choose dynamic channels: variance across a quick probe series
        probevec=[]
        for _ in range(40): probe(PROBE_MS); probevec.append(read_pm())
        P=np.array(probevec); var=np.nanvar(P,0); idx=np.argsort(-var)[:K]
        snaps=np.zeros((M,K)); temps=np.zeros(M)
        for m in range(M):
            # temperature hold (bang-bang): warm if cold; the probe itself adds heat, so just gate cold side
            g=0
            while zone0()<TSET-0.7 and g<10: probe(60); g+=1
            probe(PROBE_MS)                      # identical fixed probe
            v=read_pm()[idx]; snaps[m]=v; temps[m]=zone0()
            if m%100==0: print(f"  m={m} T={temps[m]:.1f}C",flush=True)
        OUT.mkdir(parents=True,exist_ok=True)
        p=OUT/f"coh_{HOST}_{RUNTAG}.npz"
        np.savez_compressed(p,snaps=snaps,temps=temps,idx=idx,tset=TSET,pin=PIN_MHZ)
        print(f"[{HOST}] M={M} K={K} Tmean={temps.mean():.1f}±{temps.std():.2f}C >>> {p.name}",flush=True)
    finally:
        for c in range(NCPU): wr(SMAX[c],cmax); wr(SMIN[c],cmin); wr(GOV[c],og)
        if ob: wr(BOOST,ob)
        print(f"[restore] gov0={rd(GOV[0])}",flush=True)


def compare():
    fs=sorted(glob.glob(str(OUT/f"coh_*_{RUNTAG}.npz")))
    hosts=[Path(f).stem.split("coh_")[1].rsplit("_",1)[0] for f in fs]
    if len(fs)<2: print(f"need 2 dies, have {hosts}"); return
    A,B=hosts[0],hosts[1]
    SA=np.load(fs[0])["snaps"]; SB=np.load(fs[1])["snaps"]
    # per-channel standardize by pooled single-snapshot std (so config-scale doesn't dominate)
    pooled=np.sqrt((SA.var(0)+SB.var(0))/2)+1e-9
    SA=SA/pooled; SB=SB/pooled
    Mn=min(len(SA),len(SB))
    print(f"dies {A},{B}  M={Mn}  K={SA.shape[1]}",flush=True)
    print("  d'(M): inter=across dies, intra=same-die two halves (null). d'∝√M if stable & averageable.",flush=True)
    rng=np.random.default_rng(0)
    for m in [1,5,20,50,100,200,400,Mn]:
        if m>Mn: continue
        # bootstrap many random m-subsets, measure averaged-mean separation in std units
        di=[]; dn=[]
        for _ in range(40):
            ia=rng.choice(Mn,m,False); ib=rng.choice(Mn,m,False)
            ah=rng.choice(Mn,m,False); ah2=rng.choice(Mn,m,False)
            ma=SA[ia].mean(0); mb=SB[ib].mean(0)
            # within-die noise of an m-average ~ single_std/sqrt(m); use empirical from halves
            na=SA[ah].mean(0); na2=SA[ah2].mean(0)
            di.append(np.linalg.norm(ma-mb)); dn.append(np.linalg.norm(na-na2))
        inter=np.mean(di); intra=np.mean(dn)
        print(f"   M={m:4d}: inter={inter:.3f}  intra(null)={intra:.3f}  ratio={inter/(intra+1e-9):.2f}",flush=True)
    print("\n  -> ratio GROWS with M and >>1  => stable separable difference (radar trick works; then config-vs-die).",flush=True)
    print("  -> ratio ~1 / flat            => no stable die offset above floor (buried in quantization/drift).",flush=True)


if __name__=="__main__":
    (compare if len(sys.argv)>1 and sys.argv[1]=="compare" else enroll)()
