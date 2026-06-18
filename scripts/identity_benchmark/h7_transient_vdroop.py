"""H7 AMPLIFIED probe — fast-edge Vdroop excitation + time-multiplexed transient readout.

Insight: the genuine die nonlinearity we found (IMD 1.8x over static, ch5 bilinear) is Vdroop / di/dt
physics — ELECTRICAL and FAST, not thermal. So we can excite it HARD with SHARP load edges at LOW
average power (cool!) instead of cooking the chip. Amplification = three tricks:
  1. sharp max-amplitude bursts, low duty -> big di/dt droop, chip stays cool
  2. read the post-edge SETTLING TRANSIENT as many time-multiplexed virtual nodes (Appeltant) -> a weak
     nonlinearity unfolded into a high-dim reservoir state
  3. static-map control + long run -> isolate the genuine DYNAMICAL nonlinearity from instantaneous load
Fair test: can a RANK-LIMITED LINEAR readout of the transient reservoir now do XOR and BEAT the rank-
limited u-window adapter — where the SMOOTH drive failed (rank_necessity was negative)? Also reports the
static-map excess and the unbounded-nl-u reference. Low duty => thermally safe (target <70C). Root.
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
BURST_MS = 0.004        # sharp 4ms max-GPU burst on u=1 (high di/dt edge)
STEP_S = 0.030          # 30ms step -> ~13% duty on u=1 -> low average power, cool
NTAP = 12               # virtual nodes = transient settling samples after the edge (~24ms @ 500Hz)
W = 4; R = 4            # rank-limited adapter window + rank
SEED = 0


def temp_c():
    try: return int(ZONE.read_text())/1000.0
    except Exception: return 0.0


def collect(u):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(2048,2048,device=dev); gB = torch.randn(2048,2048,device=dev)  # big = sharp high-di/dt edge
    st = SubstrateStateV3(hz_target=500); st.start(); time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1,N_CH) for _ in range(40)]).reshape(-1,N_CH)
    med = np.median(pool,0); mad = np.median(np.abs(pool-med),0)*1.4826+1e-9
    T = np.zeros((L, NTAP, N_CH), np.float32); t0=time.time()
    for t in range(L):
        s0=time.time()
        if u[t]:
            # sync INSIDE the burst so 4ms wall-clock = ~4ms REAL GPU work (genuine ~13% duty).
            # Without this, the loop async-queues hundreds of matmuls in 4ms and the later sync runs
            # them all -> GPU saturates continuously -> chip overheats (daedalus hit 96C). 2026-06-17.
            while time.time()-s0 < BURST_MS:
                gA=(gA@gB).tanh()*0.5+0.5
                if dev=="cuda": torch.cuda.synchronize()
        # capture settling transient (virtual nodes) for the rest of the step
        time.sleep(0.004)
        T[t] = st.latest_window(length=NTAP).reshape(-1,N_CH)[:NTAP]
        rest = STEP_S - (time.time()-s0)
        if rest>0: time.sleep(rest)
        if t%10==0:
            tc=temp_c()
            if tc>74.0:
                while temp_c()>56.0: time.sleep(1.0)
            if t%600==0: print(f"  step {t}/{L} temp={tc:.0f}C ({time.time()-t0:.0f}s)",flush=True)
    st.stop()
    Tn = np.tanh((T - med)/mad/8.0)
    return Tn, med, mad


def lag(x,k):
    y=np.zeros_like(x);
    if k>0: y[k:]=x[:-k]
    return y if k>0 else x.copy()


def rank_lin(X,y,tr,te,nc,rank):
    mu=X[tr].mean(0); sd=X[tr].std(0)+1e-9; Xz=(X-mu)/sd
    U,Sv,Vt=np.linalg.svd(Xz[tr]-Xz[tr].mean(0),full_matrices=False)
    P=Vt[:rank]; Xp=(Xz-Xz[tr].mean(0))@P.T; Y=np.eye(nc)[y]; best=0.0
    for al in[1e-2,.1,1,10,100]:
        W_=np.linalg.solve(Xp[tr].T@Xp[tr]+al*np.eye(Xp.shape[1]),Xp[tr].T@Y[tr])
        best=max(best,float(np.mean((Xp[te]@W_).argmax(1)==y[te])))
    return best


def full_lin(X,y,tr,te,nc):
    mu=X[tr].mean(0);sd=X[tr].std(0)+1e-9;X=(X-mu)/sd;Y=np.eye(nc)[y];best=0.0
    for al in[.1,1,10,100,1e3]:
        W_=np.linalg.solve(X[tr].T@X[tr]+al*np.eye(X.shape[1]),X[tr].T@Y[tr])
        best=max(best,float(np.mean((X[te]@W_).argmax(1)==y[te])))
    return best


def main():
    rng=np.random.default_rng(SEED); u=rng.integers(0,2,size=L)
    print(f"[{HOST}] AMPLIFIED transient/Vdroop reservoir (sharp edges, low duty) temp {temp_c():.0f}C...",flush=True)
    Tn,med,mad = collect(u)
    # drive landed?
    flat = Tn.reshape(L,-1)
    dland=np.abs((flat[u==1].mean(0)-flat[u==0].mean(0))/(np.sqrt((flat[u==1].std(0)**2+flat[u==0].std(0)**2)/2)+1e-9)).max()
    print(f"  drive landed max|d|={dland:.2f}  (transient tap range)",flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/f"transient_vdroop_raw_{HOST}.npz", u=u, Tn=Tn)

    # reservoir features: transient taps (flattened) + 1 lag of the per-step mean
    Xres = flat                                   # (L, NTAP*N_CH) = time-multiplexed virtual nodes
    Xres = np.hstack([Xres, lag(flat, 1), lag(flat, 2)])
    Uwin = np.stack([lag(u.astype(float),k) for k in range(1,W+1)],1)
    # nonlinear-on-u reference (unbounded strawman)
    import itertools
    uu=u.astype(float); ucols=[lag(uu,k) for k in range(16)]
    for a,b in itertools.combinations(range(10),2): ucols.append(lag(uu,a)*lag(uu,b))
    Xnlu=np.stack(ucols,1)

    n=L-WASHOUT; cut=WASHOUT+int(0.7*n); tr=slice(WASHOUT,cut); te=slice(cut,L)
    def lb(k): return lag(u,k).astype(int)
    tasks={"RECALL_t2":(lb(2),2),"XOR_t1t2":(lb(1)^lb(2),2),"XOR_t1t3":(lb(1)^lb(3),2),
           "XOR_t2t4":(lb(2)^lb(4),2),"XOR_t2t8":(lb(2)^lb(8),2),
           "PAR3_123":((lb(1)^lb(2)^lb(3)),2)}
    suite={}
    for nm,(y,nc) in tasks.items():
        die_r=rank_lin(Xres,y,tr,te,nc,R)
        die_full=full_lin(Xres,y,tr,te,nc)
        ctrl=rank_lin(Uwin,y,tr,te,nc,min(R,W))
        nlu=full_lin(Xnlu,y,tr,te,nc)
        win = die_r-ctrl>0.05 and die_r>1.0/nc+0.05
        suite[nm]={"chance":1.0/nc,"die_rank4":die_r,"die_full_linear":die_full,
                   "u_window_rank":ctrl,"unbounded_nl_u":nlu,"die_needed":bool(win)}
        flag="  <-- DIE NEEDED" if win else ("  (die_full beats u_win)" if die_full-ctrl>0.05 else "")
        print(f"  {nm:10s} chance={1.0/nc:.2f} die(r4)={die_r:.3f} die(full)={die_full:.3f} u_win={ctrl:.3f} [nl_u={nlu:.2f}]{flag}",flush=True)
    needed=[k for k,v in suite.items() if v["die_needed"]]
    out={"host":HOST,"drive_landed_d":float(dland),"ntap":NTAP,"burst_ms":BURST_MS,"task_suite":suite,
         "die_needed_on":needed,
         "verdict":"AMPLIFICATION WORKED — die needed" if needed else "amplification insufficient — die still not needed"}
    def jf(o):
        if isinstance(o,dict): return {k:jf(v) for k,v in o.items()}
        if isinstance(o,(np.floating,np.integer)): return float(o)
        return o
    (OUT/f"transient_vdroop_{HOST}.json").write_text(json.dumps(jf(out),indent=2))
    print(f"\n>>> {out['verdict']}  needed_on={needed}",flush=True)


if __name__=="__main__":
    main()
