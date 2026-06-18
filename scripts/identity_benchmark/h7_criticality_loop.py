"""H7 criticality closed-loop — does a critical compute<->substrate feedback AMPLIFY die identity?

Eric's idea: the whole coupled body is unique; a feedback loop at the edge of chaos amplifies tiny
individual transfer-function differences (butterfly) so they become measurable without deeper silicon access.

Loop (per step t):
  s_t  = live substrate vector (top-variance PM-table channels, normalized to enrolled baseline)
  x_{t+1} = tanh( rho * W @ x_t + Win @ s_t )           # reservoir near criticality (rho ~ 1)
  load_t  = DMAX * sigmoid(GAIN * x_t[0])               # FEEDBACK: state drives compute burst length
  -> running the burst heats/loads the chip -> changes s_{t+1}  (closes the physical loop)

Modes: closed (feedback on) | open (load_t = DMAX/2, fixed) | shuffle (s dims permuted before Win).
Question: is INTER-die trajectory distance (relative to INTRA) LARGER for closed than open?
  -> if yes, criticality amplifies the die/system-specific physical transfer = a real handle on UNIQUE.
Also: Lyapunov estimate via twin trajectories replaying recorded s (same s, x0 vs x0+eps) -> confirms regime.

Env: MODE(closed|open|shuffle) RHO(1.05) STEPS(300) DMAX(0.06) GAIN(2.0) RUNTAG(r1) TMAX(85) H7_OUT. Root.
"""
from __future__ import annotations
import os, sys, time, socket, subprocess
from pathlib import Path
import numpy as np

HOST=socket.gethostname()
OUT=Path(os.environ["H7_OUT"]) if os.environ.get("H7_OUT") else Path(__file__).resolve().parents[2]/"results/IDENTITY_H7_2026-06-09"
PM=Path("/sys/kernel/ryzen_smu_drv/pm_table")
MODE=os.environ.get("MODE","closed"); RHO=float(os.environ.get("RHO","1.05"))
STEPS=int(os.environ.get("STEPS","300")); DMAX=float(os.environ.get("DMAX","0.06"))
GAIN=float(os.environ.get("GAIN","2.0")); RUNTAG=os.environ.get("RUNTAG","r1")
TMAX=float(os.environ.get("TMAX","85")); D=32; K=16; SEED=7; NCPU=os.cpu_count() or 1

def read_pm():
    try:
        b=PM.read_bytes(); n=(len(b)//4)*4
        return np.frombuffer(b[:n],dtype=np.float32).astype(np.float64)
    except: return None
def zone0():
    try: return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000.0
    except: return 0.0

_LOAD="import numpy as np,sys,time\nt=float(sys.argv[1]);a=np.random.rand(640,640)\ne=time.time()+t\nwhile time.time()<e: a=(a@a)%7.0+0.1"
def burst(dur, ncore=8):
    if dur<=0.002: time.sleep(dur if dur>0 else 0); return
    ps=[subprocess.Popen(["taskset","-c",str(c),sys.executable,"-c",_LOAD,str(dur)],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) for c in range(ncore)]
    for p in ps: p.wait()


def main():
    rng=np.random.default_rng(SEED)
    W=rng.standard_normal((D,D)); W/=np.max(np.abs(np.linalg.eigvals(W)))      # spectral radius 1
    Win=rng.standard_normal((D,K))*0.08     # weak drive -> stay in tanh near-linear regime (so rho controls criticality)
    perm=rng.permutation(K)
    # enroll dynamic PM channels + baseline: probe under idle+load
    pm=read_pm()
    if pm is None: print("no pm_table (need root + ryzen_smu)"); return
    probe=[]
    for _ in range(20): probe.append(read_pm()); time.sleep(0.05)
    burst(0.4,ncore=NCPU)
    for _ in range(20): probe.append(read_pm()); time.sleep(0.05)
    P=np.array(probe); var=np.nanvar(P,0); idx=np.argsort(-var)[:K]
    med=np.nanmedian(P[:,idx],0); mad=np.nanmedian(np.abs(P[:,idx]-med),0)*1.4826+1e-6
    def sread():
        p=read_pm(); v=(p[idx]-med)/mad; return np.nan_to_num(np.clip(v,-8,8))

    x=np.zeros(D); X=np.zeros((STEPS,D)); S=np.zeros((STEPS,K)); L=np.zeros(STEPS); T=np.zeros(STEPS)
    aborted=False
    for t in range(STEPS):
        s=sread()
        if MODE=="shuffle": s=s[perm]
        x=np.tanh(RHO*(W@x)+Win@s)
        load = DMAX*0.5 if MODE=="open" else DMAX*(1.0/(1.0+np.exp(-GAIN*x[0])))
        X[t]=x; S[t]=s; L[t]=load; T[t]=zone0()
        if zone0()>TMAX: aborted=True; break
        burst(load, ncore=8)
    n=t if aborted else STEPS
    X,S,L,T=X[:n],S[:n],L[:n],T[:n]

    def replay_lyap(rho):
        def rp(x0):
            xx=x0.copy(); tr=np.zeros((n,D))
            for k in range(n): xx=np.tanh(rho*(W@xx)+Win@S[k]); tr[k]=xx
            return tr
        ta=rp(np.zeros(D)); tb=rp(np.full(D,1e-8))
        d=np.clip(np.linalg.norm(ta-tb,axis=1),1e-300,None)
        m=min(80,n); return float(np.polyfit(np.arange(m),np.log(d[:m]),1)[0])
    lyap=replay_lyap(RHO)
    if MODE=="tune":
        print(f"[{HOST}] rho->lyap (criticality at lyap~0):",flush=True)
        for r in [0.6,0.8,1.0,1.2,1.5,2.0,3.0,5.0]:
            print(f"   rho={r:.1f}  lyap={replay_lyap(r):+.4f}",flush=True)
        return

    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/f"crit_{HOST}_{MODE}_{RUNTAG}.npz"
    np.savez_compressed(p,X=X,S=S,L=L,T=T,rho=RHO,mode=MODE,lyap=lyap,idx=idx,n=n)
    print(f"[{HOST}] mode={MODE} rho={RHO} n={n} lyap={lyap:+.4f} "
          f"Tend={T[-1]:.0f}C load_mean={L.mean():.3f} x_std={X.std():.3f} >>> {p.name}",flush=True)


if __name__=="__main__":
    main()
