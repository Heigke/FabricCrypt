"""H7 system-identification fingerprint — the WHOLE coupled dynamical body, not a scalar.

Hypothesis (Eric): how the states co-vary in time — inertia (thermal time constants), latency (lags
between signals), momentum, and the coupling between ALL signals — is what is genuinely unique, because
it encodes the entire physical assembly (die+package+board+cooler+thermal masses+PDN). And time-constants
& lags are INTENSIVE -> naturally invariant to the temperature OFFSET that confounded every prior test.

Method: pin DVFS (remove firmware DVFS-controller dynamics, keep physical dynamics). Excite with a known
load step train (ON heats, OFF cools). Record many channels at ~50Hz. Extract per-cycle heating/cooling
time constants, step gain, and the full inter-channel cross-correlation LAG matrix (the "samspel").
Compare intra (cycles within a run) vs inter (cross-machine). Control: features are offset-invariant
(tau, lag, gain ratios), so a pure ambient-temp difference cannot produce them.

THERMAL SAFE: aborts the ON phase if zone0 > TMAX. Env: RUNTAG(r1) TMAX(82) CYCLES(3) TON(22) TOFF(34) H7_OUT.
"""
from __future__ import annotations
import os, sys, time, socket, glob, subprocess, signal
from pathlib import Path
import numpy as np

HOST=socket.gethostname()
OUT=Path(os.environ["H7_OUT"]) if os.environ.get("H7_OUT") else Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
NCPU=os.cpu_count() or 1
GOV=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor" for c in range(NCPU)]
SMAX=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq" for c in range(NCPU)]
SMIN=[f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_min_freq" for c in range(NCPU)]
BOOST="/sys/devices/system/cpu/cpufreq/boost"
RUNTAG=os.environ.get("RUNTAG","r1"); TMAX=float(os.environ.get("TMAX","82"))
CYCLES=int(os.environ.get("CYCLES","3")); TON=float(os.environ.get("TON","22")); TOFF=float(os.environ.get("TOFF","34"))
PIN_MHZ=int(os.environ.get("PIN_MHZ","2600")); FS=20.0

def rd(p,d=""):
    try: return Path(p).read_text().strip()
    except: return d
def wr(p,v):
    try: Path(p).write_text(str(v)); return True
    except: return False

# EXHAUSTIVE channels: ALL thermal zones + ALL hwmon inputs + ALL per-core freqs + whole PM table (1024 floats)
PM=Path("/sys/kernel/ryzen_smu_drv/pm_table")
def find_channels():
    chans={}  # name -> (path, scale)
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        chans[f"tz{z.split('thermal_zone')[1].split('/')[0]}"]=(z,1e-3)
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        hn=Path(h).name
        for p in sorted(glob.glob(f"{h}/*_input")):
            f=Path(p).name; sc=1e-3 if f.startswith(("in","temp","curr")) else (1e-6 if f.startswith(("power","freq")) else 1.0)
            chans[f"{hn}_{f.replace('_input','')}"]=(p,sc)
    import re
    def cpunum(s): m=re.search(r'/cpu(\d+)/',s); return int(m.group(1)) if m else 0
    for cf in sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"),key=cpunum):
        chans[f"c{cpunum(cf)}_mhz"]=(cf,1e-3)
    return chans
def read_pm():
    try:
        b=PM.read_bytes(); n=(len(b)//4)*4
        return np.frombuffer(b[:n],dtype=np.float32).astype(np.float64)
    except: return None

_procs=[]
_WORK="import numpy as np\na=np.random.rand(768,768)\nwhile True:\n a=(a@a)%7.0+0.1"  # AVX matmul = real heat
def burn_on():
    """Spawn REAL AVX-heavy per-core load (numpy matmul, bypasses GIL via separate procs)."""
    global _procs; _procs=[]
    py=sys.executable
    for c in range(NCPU):
        p=subprocess.Popen(["taskset","-c",str(c),py,"-c",_WORK],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        _procs.append(p)
def burn_off():
    global _procs
    for p in _procs:
        try: p.kill()
        except: pass
    subprocess.run(["pkill","-9","-f","a@a)%7.0"],check=False)
    _procs=[]
def zone0():
    return int(rd("/sys/class/thermal/thermal_zone0/temp","0"))/1000.0
TSTART=float(os.environ.get("TSTART","48"))
def soak_to(tstart, tol=1.5, timeout=120):
    """Bring die to a COMMON start temp before each cycle (heat with burn / cool by idle). Kills start-temp confound."""
    t0=time.time()
    while time.time()-t0<timeout:
        t=zone0()
        if t < tstart-tol:
            burn_on()
            while zone0()<tstart and time.time()-t0<timeout: time.sleep(0.4)
            burn_off(); return
        elif t > tstart+tol:
            while zone0()>tstart and time.time()-t0<timeout: time.sleep(0.5)
            return
        else:
            return


def enroll():
    orig_gov=rd(GOV[0],"powersave"); orig_boost=rd(BOOST,"1")
    cmax=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"); cmin=rd("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq")
    def restore():
        burn_off()
        for c in range(NCPU): wr(SMAX[c],cmax); wr(SMIN[c],cmin); wr(GOV[c],orig_gov)
        if orig_boost: wr(BOOST,orig_boost)
        print(f"[restore] gov0={rd(GOV[0])} boost={rd(BOOST)}",flush=True)
    wr(BOOST,"0")
    for c in range(NCPU): wr(GOV[c],"performance"); wr(SMAX[c],PIN_MHZ*1000); wr(SMIN[c],PIN_MHZ*1000)
    chans=find_channels(); small_names=list(chans)
    pm0=read_pm(); npm=len(pm0) if pm0 is not None else 0
    pm_names=[f"pm{i:04d}" for i in range(npm)]
    names=small_names+pm_names
    print(f"[{HOST}] {len(names)} channels ({len(small_names)} sysfs + {npm} pm_table) pin {PIN_MHZ}MHz z0={zone0():.0f}C",flush=True)
    # cool to a low start
    t0=time.time()
    while zone0()>45 and time.time()-t0<120: time.sleep(1)
    rows=[]; labels=[]  # labels: 0=idle 1=on
    dt=1.0/FS
    def sample(state):
        r=[time.time()]
        for n in small_names:
            p,sc=chans[n]; r.append(float(rd(p,"nan") or "nan")*sc)
        pm=read_pm()
        r.extend(pm.tolist() if pm is not None and len(pm)==npm else [float('nan')]*npm)
        rows.append(r); labels.append(state)
    try:
        for cyc in range(CYCLES):
            soak_to(TSTART)                      # MATCHED start temp every cycle (kills start-temp confound)
            te=time.time()+4                     # short baseline at TSTART
            while time.time()<te: sample(0); time.sleep(dt)
            tstart_act=zone0()
            burn_on()
            te=time.time()+TON
            while time.time()<te:
                sample(1); time.sleep(dt)
                if zone0()>TMAX: break
            burn_off(); time.sleep(0.05)
            te=time.time()+TOFF
            while time.time()<te: sample(0); time.sleep(dt)
            print(f"  cycle{cyc} start={tstart_act:.0f}C peak={zone0():.0f}C nrows={len(rows)}",flush=True)
    finally:
        restore()
    A=np.array(rows); lab=np.array(labels)
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/f"sysid_{HOST}_{RUNTAG}.npz"
    np.savez_compressed(p,data=A,labels=lab,names=np.array(names),fs=FS,ton=TON,toff=TOFF,cycles=CYCLES)
    print(f">>> saved {p.name} shape={A.shape}",flush=True)


def features(npz):
    d=np.load(npz,allow_pickle=True); A=d["data"]; lab=d["labels"]; names=list(d["names"]); fs=float(d["fs"])
    t=A[:,0]-A[0,0]; X=A[:,1:]
    # pick the strongest thermal channel (max variance among tz/temp) as the response
    var=np.nanvar(X,0);
    # heating/cooling time constants on zone0 (col where name startswith tz0 or first tz)
    def col(pred):
        for i,n in enumerate(names):
            if pred(n): return i
        return 0
    ztemp=col(lambda n:n.startswith("tz0")); zgpu=col(lambda n:"temp" in n and "hwmon" in n)
    on=lab==1
    # find first ON->index transitions
    trans=np.where(np.diff(lab.astype(int))==1)[0]
    feats={}
    def fit_tau(seg_t,seg_y,heating):
        if len(seg_t)<6: return np.nan
        y=seg_y-seg_y.min() if heating else seg_y-seg_y.min()
        yinf=seg_y.max() if heating else seg_y.min()
        z=np.abs(seg_y-yinf)+1e-6
        # linear fit ln(z) vs t -> slope=-1/tau
        b=np.polyfit(seg_t-seg_t[0],np.log(z),1)[0]
        return -1.0/b if b<0 else np.nan
    taus_h=[]; taus_c=[]; gains=[]
    on_idx=np.where(np.diff(lab.astype(int))==1)[0]; off_idx=np.where(np.diff(lab.astype(int))==-1)[0]
    for oi in on_idx:
        end=off_idx[off_idx>oi]
        if len(end)==0: continue
        e=end[0]
        taus_h.append(fit_tau(t[oi:e],X[oi:e,ztemp],True))
        gains.append(X[e,ztemp]-X[oi,ztemp])
    for ci in off_idx:
        end=on_idx[on_idx>ci]
        e=end[0] if len(end)>0 else len(t)-1
        taus_c.append(fit_tau(t[ci:e],X[ci:e,ztemp],False))
    feats["tau_heat"]=np.nanmedian(taus_h); feats["tau_cool"]=np.nanmedian(taus_c); feats["step_gain"]=np.nanmedian(gains)
    Xz=(X-np.nanmean(X,0))/(np.nanstd(X,0)+1e-9); Xz=np.nan_to_num(Xz)
    return feats,Xz,names,lab,fs


def _lag_response(Xz,names,lab,fs):
    """per-channel |corr with load| -> dict name->responsiveness, and the standardized series by name."""
    lz=(lab-lab.mean())/(lab.std()+1e-9)
    resp={names[k]:abs(float(np.dot(Xz[:,k],lz)/len(lz))) for k in range(Xz.shape[1]) if np.std(Xz[:,k])>1e-6}
    series={names[k]:Xz[:,k] for k in range(Xz.shape[1])}
    return resp,series
def _lagmat(series,chans,fs,maxlag_s=8.0):
    maxlag=int(fs*maxlag_s); out=[]
    for ii in range(len(chans)):
        for jj in range(ii+1,len(chans)):
            a=series[chans[ii]]; b=series[chans[jj]]
            n=min(len(a),len(b)); a=a[:n]; b=b[:n]
            xc=np.correlate(a,b,"full"); c0=len(xc)//2; seg=xc[c0-maxlag:c0+maxlag+1]
            out.append((np.argmax(seg)-maxlag)/fs)
    return np.array(out)


def compare():
    fs_=sorted(glob.glob(str(OUT/f"sysid_*_*.npz")))
    # group by host, collect all runtags for intra
    byhost={}
    for f in fs_:
        h=Path(f).stem.split("sysid_")[1].rsplit("_",1)[0]; byhost.setdefault(h,[]).append(f)
    hosts=sorted(byhost)
    if len(hosts)<2: print(f"need 2 dies, have {hosts}"); return
    print(f"dies: {hosts}  runs: {{ {', '.join(h+':'+str(len(byhost[h])) for h in hosts)} }}",flush=True)
    def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    # load all, compute per-run responsiveness; pick COMMON dynamic channels across ALL runs
    runs={}; resp_all=[]
    for h in hosts:
        for f in byhost[h]:
            ft,Xz,names,lab,fs=features(f); resp,series=_lag_response(Xz,names,lab,fs)
            runs[(h,f)]=(ft,resp,series,fs); resp_all.append(resp)
    common=set(resp_all[0])
    for r in resp_all[1:]: common&=set(r)
    score={c:min(r.get(c,0) for r in resp_all) for c in common}
    # TIMESCALE-SPANNING set: force in SLOW channels (thermal zone + temp sensors) + top FAST (power/clk)
    slow=[c for c in common if c.startswith("tz") or "temp" in c]
    fast=[c for c,_ in sorted(score.items(),key=lambda kv:-kv[1]) if c not in slow][:12]
    chans=slow[:6]+fast
    print(f"  lag channels: SLOW={slow[:6]} FAST(top)={fast[:5]}...",flush=True)
    # lag matrix per run on common channels
    lags={k:_lagmat(v[2],chans,v[3]) for k,v in runs.items()}
    taus={k:v[0] for k,v in runs.items()}
    for h in hosts:
        ks=[k for k in runs if k[0]==h]
        for k in ks:
            print(f"  {h} {Path(k[1]).stem.split('_')[-1]}: tau_h={taus[k]['tau_heat']:.1f}s tau_c={taus[k]['tau_cool']:.1f}s gain={taus[k]['step_gain']:.1f}C",flush=True)
    # INTRA: cos between runs of same host
    print("\n  LAG-matrix correlation (offset-invariant 'samspel'):",flush=True)
    for h in hosts:
        ks=[k for k in runs if k[0]==h]
        cs=[cos(lags[ks[i]],lags[ks[j]]) for i in range(len(ks)) for j in range(i+1,len(ks))]
        if cs: print(f"   INTRA {h} = {np.mean(cs):+.3f}",flush=True)
    A,B=hosts[0],hosts[1]
    inter=[cos(lags[ka],lags[kb]) for ka in runs if ka[0]==A for kb in runs if kb[0]==B]
    print(f"   INTER {A} vs {B} = {np.mean(inter):+.3f}",flush=True)
    # scalar tau cross-die
    for h in hosts:
        ks=[k for k in runs if k[0]==h]
        print(f"   {h}: tau_heat={np.nanmean([taus[k]['tau_heat'] for k in ks]):.1f}s tau_cool={np.nanmean([taus[k]['tau_cool'] for k in ks]):.1f}s",flush=True)


if __name__=="__main__":
    (compare if len(sys.argv)>1 and sys.argv[1]=="compare" else enroll)()
