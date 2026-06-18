"""H7 signal drift — measure WHICH live signals drift and HOW FAST, honestly, on real HW.

The user's hypothesis: even if an attacker copies the fingerprint, by the time it reaches the
LLM it is STALE -> the persona breaks. That only works for signal families that actually DRIFT.
Identity wants stability; freshness wants drift. So we MEASURE each family separately and let the
data decide which part is "stable identity" vs which part is "fresh liveness".

For each family we record a time series and report:
  * cos-distance from t0 over time (drift curve)  -> high = good freshness, low = stable identity
  * within-window std                              -> jitter magnitude
No artificial load by default (reads are light/safe). --load adds a short CPU touch each sample.

Read-only sysfs / PM-table only. NO SMU mailbox writes.
Out: results/IDENTITY_H7_2026-06-09/signal_drift_{host}.json (+ .npz raw)
"""
from __future__ import annotations
import argparse, glob, json, os, socket, time
from pathlib import Path
import numpy as np

HOST=socket.gethostname()
OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True,exist_ok=True)
PM=Path("/sys/kernel/ryzen_smu_drv/pm_table")
NCPU=os.cpu_count() or 16
CAND_VIDX=[756,110]

def z(v):
    v=np.asarray(v,float); s=v.std()
    return (v-v.mean())/(s+1e-9) if s>1e-12 else v*0.0

def read_vcore(n=16,lo=0.5,hi=1.1):
    try: b=PM.read_bytes(); v=np.frombuffer(b[:(len(b)//4)*4],dtype=np.float32).astype(float)
    except Exception: return None
    for idx in CAND_VIDX:
        if idx+n<=len(v):
            w=v[idx:idx+n]
            if np.all((w>=lo)&(w<=hi)) and w.std()<0.08: return w.copy()
    i=0;best=None;bs=1e9
    while i<len(v)-n:
        w=v[i:i+n]
        if np.all((w>=lo)&(w<=hi)) and w.std()<0.06 and w.std()<bs: bs=w.std();best=i
        i+=1
    return v[best:best+n].copy() if best is not None else None

def read_cppc():
    out=[]
    for c in range(NCPU):
        try: out.append(int(Path(f"/sys/devices/system/cpu/cpu{c}/acpi_cppc/highest_perf").read_text()))
        except: pass
    return np.array(out,float) if out else None

def read_thermal():
    vals=[]
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try: vals.append(int(Path(p).read_text())/1000.0)
        except: pass
    return np.array(vals,float) if vals else None

def _gpu_freq_path():
    for c in glob.glob("/sys/class/hwmon/hwmon*/freq1_input"):
        try: int(Path(c).read_text()); return c
        except: pass
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dur",type=float,default=180.0,help="seconds")
    ap.add_argument("--every",type=float,default=3.0)
    ap.add_argument("--load",action="store_true",help="light CPU touch each sample to induce thermal drift")
    a=ap.parse_args()
    gpath=_gpu_freq_path()
    series={"vcore_raw":[],"vcore_z":[],"cppc":[],"thermal":[],"gpu_freq":[],"t":[]}
    t0=time.time()
    print(f"[{HOST}] sampling {a.dur:.0f}s every {a.every:.0f}s  load={a.load}  gpu={gpath}")
    while time.time()-t0 < a.dur:
        ts=time.time()-t0
        if a.load:
            x=np.random.randn(400,400); _=x@x.T   # brief touch
        vc=read_vcore()
        series["vcore_raw"].append(vc if vc is not None else None)
        series["vcore_z"].append(z(vc) if vc is not None else None)
        series["cppc"].append(read_cppc())
        series["thermal"].append(read_thermal())
        if gpath:
            # short jitter burst -> summary
            s=[]
            for _ in range(30):
                try: s.append(int(Path(gpath).read_text()))
                except: pass
                time.sleep(0.005)
            s=np.array(s,float)
            series["gpu_freq"].append(np.array([s.mean(),s.std(),np.mean(np.abs(np.diff(s))) if len(s)>1 else 0.0]) if len(s) else None)
        else: series["gpu_freq"].append(None)
        series["t"].append(ts)
        time.sleep(max(0.0,a.every-(time.time()-t0-ts)))
    # ---- analyze drift per family ----
    def stack(key):
        rows=[r for r in series[key] if r is not None]
        if len(rows)<3: return None
        L=min(len(r) for r in rows); return np.array([r[:L] for r in rows])
    report={"host":HOST,"dur":a.dur,"every":a.every,"load":a.load,"families":{}}
    raw={}
    for key in ["vcore_raw","vcore_z","cppc","thermal","gpu_freq"]:
        M=stack(key)
        if M is None: report["families"][key]={"status":"too few samples"}; continue
        raw[key]=M
        ref=M[0]
        def cosd(u,v):
            nu,nv=np.linalg.norm(u),np.linalg.norm(v)
            return float(1-np.dot(u,v)/(nu*nv)) if nu>1e-12 and nv>1e-12 else 0.0
        drift=[cosd(ref,row) for row in M]                       # drift from t0
        # also raw-units drift: mean abs change from t0 (interpretable)
        absdrift=[float(np.mean(np.abs(row-ref))) for row in M]
        report["families"][key]={
            "status":"ok","n":len(M),"dim":int(M.shape[1]),
            "cosdist_from_t0_max":float(np.max(drift)),
            "cosdist_from_t0_end":float(drift[-1]),
            "per_sample_std_mean":float(np.mean(M.std(axis=0))),
            "absdrift_from_t0_end":absdrift[-1],
            "value_t0":[round(x,5) for x in ref[:8].tolist()],
            "value_end":[round(x,5) for x in M[-1][:8].tolist()],
        }
    np.savez(OUT/f"signal_drift_{HOST}.npz",**raw)
    (OUT/f"signal_drift_{HOST}.json").write_text(json.dumps(report,indent=2))
    print(f"\n[{HOST}] === DRIFT (cos-dist from t0, end / max ; higher=more 'fresh') ===")
    for k,v in report["families"].items():
        if v.get("status")=="ok":
            print(f"  {k:10s} cosΔ end={v['cosdist_from_t0_end']:.4f} max={v['cosdist_from_t0_max']:.4f} "
                  f"absΔ={v['absdrift_from_t0_end']:.4g} std={v['per_sample_std_mean']:.4g} (dim {v['dim']}, n {v['n']})")
        else: print(f"  {k:10s} {v['status']}")
    print(f"saved signal_drift_{HOST}.json")

if __name__=="__main__": main()
