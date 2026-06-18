"""Multi-signal per-die fingerprint, correctly dimensioned.
Combines several signal families into ONE conditioning vector for the embodiment LLM, with
per-block standardization (scale balance) + explicit per-block weights (so strong identity
signals lead and the 'fresh'/live signals contribute without dominating or vanishing).

Blocks (read-only; PM-table reads + sysfs only — NO SMU mailbox writes):
  A per-core Vcore scatter (16)      weight 1.0  — stable silicon identity (config-immune via z-score)
  B CPPC highest_perf per-core (16)  weight 1.0  — fused silicon binning (config-immune, no sudo)
  C multi-zone thermal spread        weight 0.4  — live/fresh, light
  D GPU sclk jitter summary          weight 0.4  — live/fresh, light

Each block is z-scored within-block, then scaled so its L2 energy == its weight. Concatenate.
Output: results/IDENTITY_H7_2026-06-09/fingerprint_multisig_{host}.npy  (+ _manifest.json)
"""
from __future__ import annotations
import json, socket, time, glob
from pathlib import Path
import numpy as np

HOST=socket.gethostname()
OUT=Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True,exist_ok=True)
PM=Path("/sys/kernel/ryzen_smu_drv/pm_table")
import os; NCPU=os.cpu_count() or 16

def z(v):
    v=np.asarray(v,float); s=v.std()
    return (v-v.mean())/(s+1e-9) if s>1e-12 else v*0.0

def unit_energy(v,w):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    # total block energy (sum of squares) == w, INDEPENDENT of how many dims the block has,
    # so a high-dim block (e.g. CPPC) cannot dominate a low-dim block (e.g. Vcore) by count.
    return (v/n)*np.sqrt(w) if n>1e-12 else v

# ---- A: per-core Vcore scatter ----
# PM-table layout differs per ryzen_smu/board: ikaros per-core-V at idx 756, daedalus at idx 110.
# Try known per-host candidate offsets first (robust + consistent), then fall back to a search.
CAND_VIDX=[756,110]
def vcore_block(n=16,lo=0.5,hi=1.1):
    try: b=PM.read_bytes(); v=np.frombuffer(b[:(len(b)//4)*4],dtype=np.float32).astype(float)
    except Exception as e: return None,f"pm_table unreadable: {e}"
    for idx in CAND_VIDX:
        if idx+n<=len(v):
            w=v[idx:idx+n]
            if np.all((w>=lo)&(w<=hi)) and w.std()<0.08:
                return w, f"idx {idx}..{idx+n-1} (candidate)"
    best=None;bs=1e9;i=0
    while i<len(v)-n:
        w=v[i:i+n]
        if np.all((w>=lo)&(w<=hi)) and w.std()<0.06 and w.std()<bs: bs=w.std();best=i
        i+=1
    if best is None: return None,"no per-core V block"
    return v[best:best+n], f"idx {best}..{best+n-1} (search)"

# ---- B: CPPC highest_perf per-core binning ----
def cppc_block():
    out=[]
    for c in range(NCPU):
        p=f"/sys/devices/system/cpu/cpu{c}/acpi_cppc/highest_perf"
        try: out.append(int(Path(p).read_text().strip()))
        except: pass
    return (np.array(out,float),f"{len(out)} cores") if out else (None,"no cppc")

# ---- C: multi-zone thermal spread ----
def thermal_block():
    vals=[]
    for p in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try: vals.append(int(Path(p).read_text())/1000.0)
        except: pass
    return (np.array(vals,float),f"{len(vals)} zones") if len(vals)>=3 else (None,"too few zones")

# ---- D: GPU sclk jitter summary ----
def gpu_jitter_block(n=40):
    cands=glob.glob("/sys/class/hwmon/hwmon*/freq1_input")
    f=None
    for c in cands:
        try:
            int(Path(c).read_text()); f=c; break
        except: pass
    if f is None: return None,"no gpu freq sysfs"
    s=[]
    for _ in range(n):
        try: s.append(int(Path(f).read_text()))
        except: pass
        time.sleep(0.01)
    s=np.array(s,float)
    if len(s)<5 or s.std()<1e-9: return None,"gpu freq static"
    # summary stats = the jitter fingerprint
    feat=np.array([s.std(), np.percentile(s,90)-np.percentile(s,10), np.mean(np.abs(np.diff(s)))])
    return feat,f"{len(s)} samples"

def main():
    # Only blocks available CONSISTENTLY on both AMD dies -> same dim/structure for cross-die conditioning.
    # vcore + cppc are the two config-immune silicon families both machines expose. Live signals
    # (thermal/GPU-jitter) are NOT uniformly readable here (ikaros GPU idles static, <3 thermal zones),
    # so we do not force them in (would break cross-die dim match and over-weight one machine).
    blocks=[("vcore",vcore_block,1.0),("cppc",cppc_block,1.0)]
    parts=[]; manifest={"host":HOST,"blocks":[]}; off=0
    for name,fn,w in blocks:
        v,info=fn()
        if v is None:
            manifest["blocks"].append({"name":name,"status":"SKIP","why":info,"weight":w});
            print(f"  [{name}] SKIP ({info})"); continue
        vb=unit_energy(z(v),w)
        parts.append(vb)
        manifest["blocks"].append({"name":name,"status":"ok","info":info,"weight":w,
                                   "dims":[off,off+len(vb)],"raw_std":float(np.std(v))})
        off+=len(vb)
        print(f"  [{name}] ok dims={len(vb)} weight={w} ({info})")
    if not parts: print("NO blocks readable"); return
    fp=np.concatenate(parts).astype(np.float32)
    # final global standardize keeps overall scale ~unit without changing block balance much
    np.save(OUT/f"fingerprint_multisig_{HOST}.npy", fp)
    manifest["total_dim"]=len(fp); manifest["per_block_energy"]="== weight"
    (OUT/f"fingerprint_multisig_{HOST}_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"[{HOST}] saved fingerprint_multisig_{HOST}.npy  dim={len(fp)}  blocks={sum(1 for b in manifest['blocks'] if b['status']=='ok')}")

if __name__=="__main__": main()
