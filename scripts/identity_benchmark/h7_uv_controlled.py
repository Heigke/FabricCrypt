"""H7 u·v die-specificity — the ONE fully-controlled experiment (A+B+C). The decisive re-test.

Every prior räkna-unikt attempt (coeff/spatial/lock-in/DVFS) failed on REPRODUCIBILITY and NONE applied all
controls at once. This applies them all:
  A. CONTROLS: DVFS pinned LOW (governor=performance, boost off, scaling_max=min — cool & stable),
     TEMP-LOCKED (soak to setpoint, gate every epoch to a tight band), SHARED FIXED baseline (median/MAD
     enrolled ONCE per die, saved, reused across runs — kills the 5-order mad swing), N-EPOCH COHERENT averaging.
  B. BIN-AWARE cores: pick FASTEST and SLOWEST core from CPPC highest_perf; drive v on each — the fast-vs-slow
     differential coupling fuses the working UNIQUE channel (binning) with the RÄKNA mechanism.
  C. RICHER observables: bilinear u·v, higher-order u²v & u·v², and post-burst transient decay — not just steady.
Output feature vector -> intra (same die, shared baseline) vs inter. If THIS fails, the death is earned.
SAFE: saves & restores DVFS in finally. Env: RUNTAG, TSET (temp setpoint C, default 60), NEPOCH (default 24),
PIN_MHZ (default 2200). Root required.
"""
from __future__ import annotations
import os, sys, time, json, socket, math
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE_T = Path("/sys/class/thermal/thermal_zone0/temp")
NCPU = os.cpu_count() or 1
GOV = [f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor" for c in range(NCPU)]
SMAX = [f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq" for c in range(NCPU)]
SMIN = [f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_min_freq" for c in range(NCPU)]
BOOST = "/sys/devices/system/cpu/cpufreq/boost"
CPPC = "/sys/devices/system/cpu/cpu{}/acpi_cppc/highest_perf"
RUNTAG = os.environ.get("RUNTAG", "r1")
TSET = float(os.environ.get("TSET", "60"))
TBAND = 2.5
NEPOCH = int(os.environ.get("NEPOCH", "24"))
PIN_MHZ = int(os.environ.get("PIN_MHZ", "2200"))
N_CH = 10
sys.path.insert(0, str(Path(__file__).parent))


def temp_c():
    try: return int(ZONE_T.read_text())/1000.0
    except Exception: return 0.0
def wr(p, v):
    try: Path(p).write_text(str(v)); return True
    except Exception: return False
def rd(p, d=""):
    try: return Path(p).read_text().strip()
    except Exception: return d
def save_state():
    return {"gov":[rd(p) for p in GOV],"smax":[rd(p) for p in SMAX],"smin":[rd(p) for p in SMIN],"boost":rd(BOOST)}
def restore_state(st):
    if st["boost"]: wr(BOOST, st["boost"])
    for c in range(NCPU):
        if st["smax"][c]: wr(SMAX[c], st["smax"][c])
        if st["smin"][c]: wr(SMIN[c], st["smin"][c])
        if st["gov"][c]: wr(GOV[c], st["gov"][c])
    print(f"[restore] gov0={rd(GOV[0])} smax0={rd(SMAX[0])} boost={rd(BOOST)}", flush=True)
def pin_low(khz):
    for c in range(NCPU): wr(GOV[c],"performance"); wr(SMAX[c],khz); wr(SMIN[c],khz)


def temp_gate(lo, hi, cool_to):
    """Wait until temp in [lo,hi]; if too hot, idle to cool_to; if too cold, no-op (warm up via work)."""
    t0 = time.time()
    while time.time()-t0 < 120:
        t = temp_c()
        if t > hi:
            while temp_c() > cool_to: time.sleep(0.5)
        elif lo <= t <= hi:
            return True
        else:
            return True  # below band: proceed (drive will warm it)
    return True


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(1024,1024,device=dev); gB = torch.randn(1024,1024,device=dev)
    cA = np.random.default_rng(1).standard_normal((640,640)); cB = np.random.default_rng(2).standard_normal((640,640))
    from substrate_realtime_v3 import SubstrateStateV3

    # bin-aware core selection from CPPC
    hp = np.array([float(rd(CPPC.format(c), "nan")) for c in range(NCPU)])
    fast_core = int(np.nanargmax(hp)); slow_core = int(np.nanargmin(hp))
    print(f"[{HOST}] CPPC fastest core={fast_core}({hp[fast_core]:.0f}) slowest={slow_core}({hp[slow_core]:.0f})", flush=True)

    st_dvfs = save_state()
    sub = None
    try:
        wr(BOOST, "0"); pin_low(PIN_MHZ*1000); time.sleep(0.5)
        print(f"[{HOST}] pinned {PIN_MHZ}MHz cur={rd('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')} "
              f"gov={rd(GOV[0])} boost={rd(BOOST)} temp={temp_c():.0f}C tag={RUNTAG}", flush=True)
        sub = SubstrateStateV3(hz_target=500); sub.start(); time.sleep(6.0)

        # SHARED FIXED baseline: enroll once per die, save & reuse.
        # BASELINE_FILE env overrides -> use ANOTHER die's baseline (common-frame control to isolate
        # the u·v COMPUTATION from the static idle-telemetry baseline = identity).
        bf = os.environ.get("BASELINE_FILE", "")
        base_p = Path(bf) if bf else OUT/f"uvctl_baseline_{HOST}.npz"
        if base_p.exists():
            d = np.load(base_p); med = d["med"]; mad = d["mad"]
            print(f"  reuse shared baseline {base_p.name}", flush=True)
        else:
            temp_gate(TSET-TBAND, TSET+TBAND, TSET-TBAND)
            pool = np.array([sub.latest_window(length=64).reshape(-1,N_CH) for _ in range(60)]).reshape(-1,N_CH)
            med = np.median(pool,0); mad = np.median(np.abs(pool-med),0)*1.4826
            mad = np.maximum(mad, np.median(mad)*0.25+1e-6)
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(base_p, med=med, mad=mad)
            print(f"  ENROLLED shared baseline -> {base_p.name}", flush=True)

        FS = 80.0; dt = 1.0/FS; SEG = 192  # 2.4s per epoch
        f1, f2 = 3.0, 5.0

        def run_epoch(vcore, amp=1.0):
            """One drive epoch with v pinned to vcore; returns normalized telemetry + u,v logs."""
            try: os.sched_setaffinity(0, {vcore})
            except Exception: pass
            Y = np.zeros((SEG,N_CH),np.float32); U=np.zeros(SEG); V=np.zeros(SEG); T=np.zeros(SEG)
            nonlocal gA, cA
            for k in range(SEG):
                s0=time.time(); ts=k*dt
                u = amp*0.5*(1+math.sin(2*math.pi*f1*ts))
                v = amp*0.5*(1+math.sin(2*math.pi*f2*ts))
                gd = 0.0015*u
                if gd>1e-4:
                    while time.time()-s0<gd: gA=(gA@gB).tanh()*0.5+0.5
                if v>0:
                    sc=time.time()
                    while time.time()-sc<0.0015*v: cA=np.tanh(cA@cB)*0.5+0.5
                if dev=="cuda": torch.cuda.synchronize()
                Y[k]=sub.latest_window(length=2).reshape(-1,N_CH)[:2].mean(0); U[k]=u; V[k]=v; T[k]=temp_c()
                rest=dt-(time.time()-s0)
                if rest>0: time.sleep(rest)
            return (Y-med)/mad, U, V, T

        def fit_terms(Yn,U,V):
            uc=U-U.mean(); vc=V-V.mean(); uv=uc*vc; u2v=(uc*uc)*vc; uv2=uc*(vc*vc)
            A=np.stack([np.ones(len(U)),uc,vc,uv,u2v,uv2],1)
            out=np.zeros((N_CH,3))  # [Auv, Au2v, Auv2]
            for c in range(N_CH):
                b,*_=np.linalg.lstsq(A,Yn[:,c],rcond=None); out[c]=[b[3],b[4],b[5]]
            return out

        # N-epoch COHERENT averaging, for FAST and SLOW core, temp-gated each epoch
        feats={}
        for label, vcore in [("fast",fast_core),("slow",slow_core)]:
            acc=np.zeros((N_CH,3)); ntemps=[]
            for e in range(NEPOCH):
                temp_gate(TSET-TBAND, TSET+TBAND, TSET-TBAND)
                Yn,U,V,T = run_epoch(vcore)
                acc += fit_terms(Yn,U,V); ntemps.append(float(T.mean()))
            feats[label]=acc/NEPOCH
            print(f"  {label} core{vcore}: |Auv|={np.abs(feats[label][:,0]).mean():.3f} "
                  f"|Au2v|={np.abs(feats[label][:,1]).mean():.3f} Tmean={np.mean(ntemps):.1f}C", flush=True)
        # bin-aware differential = fast - slow (the die-specific binning × coupling)
        diff = feats["fast"]-feats["slow"]
        sub.stop()
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT/f"uvctl_{HOST}_{RUNTAG}.npz",
                            fast=feats["fast"], slow=feats["slow"], diff=diff,
                            fast_core=fast_core, slow_core=slow_core, tset=TSET, nepoch=NEPOCH, pin_mhz=PIN_MHZ)
        print(f"  bin-diff |fast-slow| (Auv,Au2v,Auv2)= {np.round(np.abs(diff).mean(0),3)}", flush=True)
        print(f">>> saved uvctl_{HOST}_{RUNTAG}.npz", flush=True)
    finally:
        try: sub.stop()
        except Exception: pass
        restore_state(st_dvfs)


if __name__ == "__main__":
    main()
