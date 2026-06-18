"""H7 FAIR necessity test — rank-limited readout (the oracle/agent loophole).

My earlier "nonlinear-on-u = 1.0" control used an UNBOUNDED ridge on a 915-dim nonlinear basis of the
commanded drive u. But the real model is a FROZEN LM + rank-≤4 LINEAR adapter that only sees the die
telemetry y_t = f(u_history, die) — it does NOT get a nonlinear basis of u, and it only has a short
window. Deepseek/Gemini + the readout-budget agents: necessity does NOT require the die to compute on
exogenous noise; it only requires y_t to carry info a RANK-LIMITED LINEAR adapter cannot reconstruct
from the short u-window it has. The die's fading MEMORY of long load history is exactly such info.

Fair test, everything matched to the real adapter (LINEAR, rank ≤ R, short window):
  - control  : rank-R linear on u-window (lags 1..W)         -> what the LM self-computes
  - die      : rank-R linear on die channels + short lags    -> what the adapter reads from the body
  Targets: long-memory recall u_{t-k} (k > W) and XOR(u_{t-a},u_{t-b}) with a or b > W.
If die beats the u-window control on targets the short window can't see -> the die is NEEDED (provides
memory/features the starved adapter lacks). Reports unbounded-nonlinear-on-u too, for reference.
Light single-GPU sustained drive (lands the bits), in-loop guard, root (substrate).
"""
from __future__ import annotations
import sys, json, time, socket
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10
L = 2600
WASHOUT = 150
DWELL = 0.012
W = 4               # adapter's u-window (lags 1..4)  AND die-lag window
R = 4              # adapter rank
SEED = 0


def temp_c():
    try: return int(ZONE.read_text())/1000.0
    except Exception: return 0.0


def collect(u):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(1024,1024,device=dev); gB = torch.randn(1024,1024,device=dev)
    st = SubstrateStateV3(hz_target=500); st.start(); time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1,N_CH) for _ in range(40)]).reshape(-1,N_CH)
    med = np.median(pool,0); mad = np.median(np.abs(pool-med),0)*1.4826+1e-9
    S = np.zeros((L,N_CH),np.float32); t0=time.time()
    for t in range(L):
        s0=time.time()
        if u[t]:
            while time.time()-s0 < 0.010:
                gA=(gA@gB).tanh()*0.5+0.5
            if dev=="cuda": torch.cuda.synchronize()
        else: time.sleep(0.010)
        time.sleep(DWELL)
        S[t]=st.latest_window(length=6).mean(0)
        if t%8==0:
            tc=temp_c()
            if tc>76.0:
                while temp_c()>56.0: time.sleep(1.0)
            if t%600==0: print(f"  step {t}/{L} temp={tc:.0f}C ({time.time()-t0:.0f}s)",flush=True)
    st.stop()
    return np.tanh((S-med)/mad/8.0)


def lag(x,k):
    y=np.zeros_like(x);
    if k>0: y[k:]=x[:-k]
    else: y=x.copy()
    return y


def rank_lin_acc(X, y, tr, te, nc, rank):
    """Rank-limited LINEAR readout: PCA(X)->top `rank` comps, then ridge. Matches a rank-r adapter."""
    mu=X[tr].mean(0); sd=X[tr].std(0)+1e-9; Xz=(X-mu)/sd
    # PCA on train
    U,Sv,Vt=np.linalg.svd(Xz[tr]-Xz[tr].mean(0),full_matrices=False)
    P=Vt[:rank]
    Xp=(Xz-Xz[tr].mean(0))@P.T
    Y=np.eye(nc)[y]; best=0.0
    for al in [1e-2,0.1,1,10,100]:
        W_=np.linalg.solve(Xp[tr].T@Xp[tr]+al*np.eye(Xp.shape[1]),Xp[tr].T@Y[tr])
        best=max(best,float(np.mean((Xp[te]@W_).argmax(1)==y[te])))
    return best


def full_nl_u_acc(u, y, tr, te, nc):
    import itertools
    uu=u.astype(float); cols=[lag(uu,k) for k in range(0,16)]
    for a,b in itertools.combinations(range(0,10),2): cols.append(lag(uu,a)*lag(uu,b))
    X=np.stack(cols,1); mu=X[tr].mean(0); sd=X[tr].std(0)+1e-9; X=(X-mu)/sd
    Y=np.eye(nc)[y]; best=0.0
    for al in [0.1,1,10,100,1e3]:
        W_=np.linalg.solve(X[tr].T@X[tr]+al*np.eye(X.shape[1]),X[tr].T@Y[tr])
        best=max(best,float(np.mean((X[te]@W_).argmax(1)==y[te])))
    return best


def main():
    rng=np.random.default_rng(SEED); u=rng.integers(0,2,size=L)
    print(f"[{HOST}] rank-limited necessity (R={R}, window W={W}) (temp {temp_c():.0f}C)...",flush=True)
    Sn=collect(u)
    dland=np.abs((Sn[u==1].mean(0)-Sn[u==0].mean(0))/(np.sqrt((Sn[u==1].std(0)**2+Sn[u==0].std(0)**2)/2)+1e-9)).max()
    print(f"  drive landed max|d|={dland:.2f}",flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/f"rank_necessity_raw_{HOST}.npz",u=u,Sn=Sn)

    n=L-WASHOUT; cut=WASHOUT+int(0.7*n); tr=slice(WASHOUT,cut); te=slice(cut,L)
    # die features: channels + lags 0..W  (adapter reads a short window of the body too)
    die_cols=[lag(Sn[:,c],k) for c in range(N_CH) for k in range(0,W+1)]
    Xdie=np.stack(die_cols,1)
    # u-window control: lags 1..W of u (what a short rank-r adapter could self-compute)
    Uwin=np.stack([lag(u.astype(float),k) for k in range(1,W+1)],1)

    def lb(k): return lag(u,k).astype(int)
    tasks={
        "RECALL_t2 (in window)":   (lb(2),2),
        "RECALL_t8 (BEYOND window)":(lb(8),2),
        "RECALL_t12 (BEYOND)":     (lb(12),2),
        "XOR_t1t3 (in window)":    (lb(1)^lb(3),2),
        "XOR_t2t8 (a or b BEYOND)":(lb(2)^lb(8),2),
        "XOR_t6t10 (BEYOND)":      (lb(6)^lb(10),2),
    }
    suite={}
    for nm,(y,nc) in tasks.items():
        die=rank_lin_acc(Xdie,y,tr,te,nc,R)
        ctrl=rank_lin_acc(Uwin,y,tr,te,nc,min(R,W))
        nlu=full_nl_u_acc(u,y,tr,te,nc)
        win = die - ctrl > 0.05 and die > 1.0/nc + 0.05
        suite[nm]={"chance":1.0/nc,"die_rank%d"%R:die,"u_window_rank":ctrl,"unbounded_nl_u":nlu,"die_needed":bool(win)}
        flag="  <-- DIE NEEDED (beats short-u-window adapter)" if win else ""
        print(f"  {nm:28s} chance={1.0/nc:.2f}  die(r{R})={die:.3f}  u_win(r)={ctrl:.3f}  [unbounded_nl_u={nlu:.2f}]{flag}",flush=True)
    needed=[k for k,v in suite.items() if v["die_needed"]]
    out={"host":HOST,"R":R,"W":W,"drive_landed_d":float(dland),"task_suite":suite,
         "die_needed_on":needed,
         "verdict":"DIE PROVIDES USABLE MEMORY (necessity vs rank-limited adapter)" if needed else "die not needed even vs rank-limited adapter"}
    def jf(o):
        if isinstance(o,dict): return {k:jf(v) for k,v in o.items()}
        if isinstance(o,(np.floating,np.integer)): return float(o)
        return o
    (OUT/f"rank_necessity_{HOST}.json").write_text(json.dumps(jf(out),indent=2))
    print(f"\n>>> {out['verdict']}   needed_on={needed}",flush=True)


if __name__=="__main__":
    main()
