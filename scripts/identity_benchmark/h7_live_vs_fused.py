"""H7 live-vs-fused — the gating experiment (oracle-demanded) before any 'silicon PUF' claim.
Is the per-core Vcore vector a LIVE analog read (moves with temperature) or a FROZEN fused AVFS table
(bit-identical across temperature)?  No reboot needed — just a controlled thermal sweep on one die.

Decision:
  - Vcore values ~constant across T (delta << per-core spread) -> FROZEN table = static fused calibration,
    NOT live silicon physics. Then it's 'fused per-die ID', drop 'PUF'.
  - Vcore values move with T BUT z-scored per-core SHAPE preserved -> LIVE read + temp-stable fingerprint
    (ideal: a real analog measurement whose identity pattern survives temperature).
  - SHAPE also scrambles with T -> live but temp-fragile (needs temp-compensation to be a fingerprint).
CPPC highest_perf (fused binning ranking) is read as the FROZEN-reference contrast (should not move with T).

Env: FREQS pin via PIN_MHZ(2800), TSETS("50,60,70,78"), WIN(120). Root.
"""
from __future__ import annotations
import os, sys, time, socket, subprocess
from pathlib import Path
import numpy as np
HOST=socket.gethostname()
OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
PM=Path("/sys/kernel/ryzen_smu_drv/pm_table"); NCPU=os.cpu_count() or 1
GOV=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor" for c in range(NCPU)]
SMAX=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq" for c in range(NCPU)]
SMIN=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_min_freq" for c in range(NCPU)]
BOOST="/sys/devices/system/cpu/cpufreq/boost"
PIN=int(os.environ.get("PIN_MHZ","2800")); WIN=int(os.environ.get("WIN","120"))
TSETS=[float(x) for x in os.environ.get("TSETS","50,60,70,78").split(",")]; TMAX=float(os.environ.get("TMAX","82"))
def rd(p,d=""):
    try: return Path(p).read_text().strip()
    except: return d
def wr(p,v):
    try: Path(p).write_text(str(v)); return True
    except: return False
def zone0():
    try: return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000.0
    except: return 0.0
def pm():
    try:
        b=PM.read_bytes(); return np.frombuffer(b[:(len(b)//4)*4],dtype=np.float32).astype(np.float64)
    except: return None
def find_vblock(v, n=16, lo=0.5, hi=1.1):
    """locate the per-core voltage block: a run of n consecutive floats in [lo,hi] with small spread."""
    best=None;bestspread=1e9
    i=0
    while i<len(v)-n:
        w=v[i:i+n]
        if np.all((w>=lo)&(w<=hi)) and w.std()<0.05:
            if w.std()<bestspread: bestspread=w.std(); best=i
            i+=1
        else: i+=1
    return best
def cppc_highest():
    out=[]
    for c in range(NCPU):
        x=rd(f"/sys/devices/system/cpu/cpu{c}/acpi_cppc/highest_perf","")
        if x: out.append(int(x))
    return np.array(out)
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

def main():
    og=rd(GOV[0]); ob=rd(BOOST,"1")
    cmax=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"); cmin=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq")
    wr(BOOST,"0")
    for c in range(NCPU): wr(GOV[c],"performance"); wr(SMAX[c],PIN*1000); wr(SMIN[c],PIN*1000)
    try:
        v0=pm()
        if v0 is None: print("no pm_table"); return
        idx=find_vblock(v0)
        if idx is None: print("could not locate per-core voltage block"); return
        print(f"[{HOST}] pm_len={len(v0)} per-core V block at idx {idx}..{idx+15}  init={np.round(v0[idx:idx+16],3)}",flush=True)
        hp=cppc_highest(); print(f"CPPC highest_perf ({len(hp)} cores) = {hp.tolist()}",flush=True)
        rows=[]  # (tset, Tmean, 16x mean V...)
        hp_by_t=[]
        for TS in TSETS:
            t0=time.time()
            while zone0()<TS-1 and time.time()-t0<120: heat(TS)
            snaps=[]; temps=[]
            for _ in range(WIN):
                while zone0()<TS-0.7: heat(TS)
                while zone0()>TS+1.2: time.sleep(0.3)
                load(50,NCPU); w=pm()
                if w is not None and len(w)==len(v0): snaps.append(w[idx:idx+16]); temps.append(zone0())
            mv=np.nanmean(snaps,0); rows.append([TS,np.mean(temps)]+list(mv))
            hp_by_t.append(cppc_highest())
            print(f"  TSET={TS:.0f} Tmean={np.mean(temps):.1f}C  Vmean={mv.mean():.4f}  V16={np.round(mv,3)}",flush=True)
        R=np.array(rows)
        OUT.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(OUT/f"livefused_{HOST}.npz",rows=R,idx=idx,hp=np.array(hp_by_t))
        # ---- analysis ----
        V=R[:,2:]  # (nT,16)
        Tmeans=R[:,1]
        print("\n=== ANALYSIS ===",flush=True)
        # 1. do absolute values move with temperature?
        vrange=V.max(0)-V.min(0)            # per-core swing across temp
        spread=V.mean(0).std()              # core-to-core spread (the fingerprint amplitude)
        print(f" per-core V swing across {Tmeans.min():.0f}-{Tmeans.max():.0f}C: mean={vrange.mean()*1000:.1f}mV max={vrange.max()*1000:.1f}mV",flush=True)
        print(f" core-to-core spread (fingerprint amplitude): {spread*1000:.1f}mV",flush=True)
        live = vrange.mean() > 0.002        # >2mV move = live read
        print(f" -> Vcore is {'LIVE (moves with temp)' if live else 'FROZEN (temp-invariant => fused table, NOT live silicon)'}",flush=True)
        # 2. is the z-scored SHAPE preserved across temperature? (fingerprint stability)
        def z(x): return (x-x.mean())/(x.std()+1e-9)
        Z=np.array([z(V[i]) for i in range(len(V))])
        cors=[float(np.corrcoef(Z[0],Z[i])[0,1]) for i in range(1,len(V))]
        print(f" per-core SHAPE corr(coldest vs each hotter T): {[round(c,3) for c in cors]}",flush=True)
        print(f" -> fingerprint SHAPE is {'TEMP-STABLE (survives sweep)' if min(cors)>0.8 else 'TEMP-FRAGILE (scrambles with temp)'}",flush=True)
        # 3. CPPC highest_perf invariance (frozen-reference contrast)
        hpv=np.array(hp_by_t)
        hp_changed = not np.all(hpv==hpv[0])
        print(f" CPPC highest_perf across temp: {'CHANGED (unexpected)' if hp_changed else 'bit-identical (confirmed frozen fused ranking)'}",flush=True)
        print("\nVERDICT: "+("Vcore = LIVE analog read; " if live else "Vcore = FROZEN fused table; ")
              +("SHAPE temp-stable -> usable live fingerprint." if min(cors)>0.8 else "SHAPE temp-fragile -> needs temp-compensation."),flush=True)
    finally:
        for c in range(NCPU): wr(SMAX[c],cmax); wr(SMIN[c],cmin); wr(GOV[c],og or "powersave")
        if ob: wr(BOOST,ob)
        print(f"[restore] gov0={rd(GOV[0])} boost={rd(BOOST)} temp={zone0():.0f}C",flush=True)

if __name__=="__main__": main()
