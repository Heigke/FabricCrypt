"""H7 real-time signal inventory — capture EVERY readable live hardware channel over a window and
report which ones actually MOVE (are genuinely real-time) vs which are static/fused. Saves the time
series so we can train a REAL-TIME-ONLY rooted model (no fused constants, no TPM in the conditioning).

Channels (read-only):
  vcore        per-core Vcore (PM table)          — live (jitter) + stable pattern
  cur_freq     per-core scaling_cur_freq          — live clock per core
  thermal      all thermal_zone*/temp             — live
  gpu_freq     hwmon freq1_input                  — live under load
  hwmon_power  hwmon power1_average / input       — live
  hwmon_misc   hwmon in*/curr*/fan* scalars       — live
Out: results/IDENTITY_H7_2026-06-09/rt_signals_{host}.npz + prints a liveness table.
"""
from __future__ import annotations
import argparse,glob,json,os,socket,time
from pathlib import Path
import numpy as np
HOST=socket.gethostname()
OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True,exist_ok=True)
PM=Path("/sys/kernel/ryzen_smu_drv/pm_table"); NCPU=os.cpu_count() or 16; CAND_VIDX=[756,110]

def read_vcore(n=16,lo=0.5,hi=1.1):
    try: b=PM.read_bytes(); v=np.frombuffer(b[:(len(b)//4)*4],dtype=np.float32).astype(float)
    except: return None
    for idx in CAND_VIDX:
        if idx+n<=len(v):
            w=v[idx:idx+n]
            if np.all((w>=lo)&(w<=hi)) and w.std()<0.08: return w.copy()
    return None
def read_curfreq():
    out=[]
    for c in range(NCPU):
        try: out.append(int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read_text()))
        except: out.append(0)
    return np.array(out,float)
def read_thermal():
    vals=[]
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try: vals.append(int(Path(p).read_text())/1000.0)
        except: vals.append(0.0)
    return np.array(vals,float)
def _hwmon(globpat):
    out=[]
    for p in sorted(glob.glob(globpat)):
        try: out.append(int(Path(p).read_text()))
        except: pass
    return np.array(out,float) if out else np.zeros(0)
def read_gpufreq():
    return _hwmon("/sys/class/hwmon/hwmon*/freq1_input")
def read_power():
    return _hwmon("/sys/class/hwmon/hwmon*/power1_average") if glob.glob("/sys/class/hwmon/hwmon*/power1_average") else _hwmon("/sys/class/hwmon/hwmon*/power1_input")
def read_misc():
    return np.concatenate([_hwmon("/sys/class/hwmon/hwmon*/in*_input"),_hwmon("/sys/class/hwmon/hwmon*/curr*_input"),_hwmon("/sys/class/hwmon/hwmon*/fan*_input")]) if True else np.zeros(0)

CH={"vcore":read_vcore,"cur_freq":read_curfreq,"thermal":read_thermal,"gpu_freq":read_gpufreq,"power":read_power,"misc":read_misc}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dur",type=float,default=40); ap.add_argument("--every",type=float,default=1.0)
    ap.add_argument("--load",action="store_true",help="tiny CPU+GPU touch each sample to wake live channels")
    a=ap.parse_args()
    series={k:[] for k in CH}; t0=time.time()
    print(f"[{HOST}] inventory {a.dur:.0f}s, load={a.load}",flush=True)
    try: import numpy as _np
    except: pass
    while time.time()-t0<a.dur:
        if a.load:
            x=np.random.randn(300,300); _=x@x.T
        for k,fn in CH.items():
            v=fn(); series[k].append(v if v is not None else None)
        time.sleep(a.every)
    rep={"host":HOST,"channels":{}}; raw={}
    for k,rows in series.items():
        rows=[r for r in rows if r is not None and getattr(r,"size",0)>0]
        if len(rows)<3: rep["channels"][k]={"status":"unreadable"}; continue
        L=min(len(r) for r in rows); M=np.array([r[:L] for r in rows])
        raw[k]=M
        per_std=M.std(axis=0)                      # live variation per element
        rng=M.max(axis=0)-M.min(axis=0)
        live_frac=float(np.mean(per_std>1e-9))     # fraction of elements that move
        rep["channels"][k]={"status":"ok","dim":int(M.shape[1]),"n":int(M.shape[0]),
            "mean_live_std":float(np.mean(per_std)),"max_range":float(np.max(rng)),
            "live_fraction":round(live_frac,2),"live":bool(live_frac>0.1)}
    np.savez(OUT/f"rt_signals_{HOST}.npz",**raw)
    (OUT/f"rt_signals_{HOST}.json").write_text(json.dumps(rep,indent=2))
    print(f"\n[{HOST}] === LIVE CHANNEL INVENTORY ===")
    for k,v in rep["channels"].items():
        if v.get("status")=="ok":
            print(f"  {k:9s} dim={v['dim']:>2} live={v['live']!s:5} live_frac={v['live_fraction']:.2f} mean_std={v['mean_live_std']:.4g} range={v['max_range']:.4g}")
        else: print(f"  {k:9s} {v['status']}")
    print(f"saved rt_signals_{HOST}.npz")

if __name__=="__main__": main()
