"""H7 die-PARAMETERIZED computation — RIGOROUS (no shortcuts). Track 2 core, done properly.

Answers the real question: at what KEY-DISTANCE does a wrong key break a readout trained on die-A — and do the
two real dies (CPPC cos=0.86) sit ABOVE or BELOW that discrimination threshold? If above → die-binding works with
a competent reservoir (the weak PoC just lacked power). If below → the dies are genuinely too similar for the soft
computational tier (then security must be the hard crypto key-gate).

Rigor: (1) POSITIVE CONTROL — reservoir solves the task to high acc; a null only counts if power exists.
(2) STRONG structured key modulation (per-input gain + per-node bias via fixed projection). (3) FIXED reservoir
(Win,Wrec) across all die conditions — only the key-derived modulation changes. (4) MULTI-SEED stats (mean±std).
(5) DOSE-RESPONSE: sweep synthetic keys at controlled cosine-to-A and place real die-B + shuffle on the curve.
Real CPPC keys from fmax_enroll_{host}_r1.npz. CPU-only.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"


def die_key(host):
    d = np.load(OUT/f"fmax_enroll_{host}_r1.npz")
    hp = d["runs"].mean(0)[:, 0]
    return (hp - hp.mean())/(hp.std()+1e-9)


def make_task(n, Din, rng):
    X = rng.standard_normal((n, Din))
    y = (X[:,0]*X[:,1] + np.tanh(2*X[:,2]) - X[:,3]*X[:,4] + 0.5*X[:,5]*X[:,6] > 0).astype(float)
    return X, y


class Reservoir:
    """Fixed ESN; key enters via per-input gain g(key) and per-node bias b(key). Reservoir weights fixed."""
    def __init__(self, Din, Dh, seed, alpha=1.2, beta=1.5, rho=0.9, iters=3):
        r = np.random.default_rng(seed)
        self.Win = r.standard_normal((Dh, Din))/np.sqrt(Din)
        W = r.standard_normal((Dh, Dh))
        ev = np.max(np.abs(np.linalg.eigvals(W)))
        self.Wrec = W/ev*rho
        self.P = r.standard_normal((Dh, Din))/np.sqrt(Din)
        self.alpha=alpha; self.beta=beta; self.iters=iters; self.Dh=Dh
    def states(self, X, key):
        g = 1.0 + self.alpha*np.tanh(key)               # per-input gain from die key
        b = self.beta*np.tanh(self.P @ key)             # per-node bias from die key
        drive = (X*g[None,:]) @ self.Win.T + b[None,:]
        h = np.tanh(drive)
        for _ in range(self.iters-1):
            h = np.tanh(h @ self.Wrec.T + drive)
        return h


def ridge_fit(H, y, lam=1e-2):
    Hb = np.concatenate([H, np.ones((len(H),1))],1)
    A = Hb.T@Hb + lam*np.eye(Hb.shape[1])
    return np.linalg.solve(A, Hb.T@(2*y-1))
def acc(H, y, w):
    Hb = np.concatenate([H, np.ones((len(H),1))],1)
    return float(((Hb@w>0).astype(float)==y).mean())


def key_at_cos(kA, c, rng):
    """Synthetic key at cosine c to kA (same norm), with random orthogonal component."""
    u = kA/np.linalg.norm(kA)
    z = rng.standard_normal(len(kA)); z = z - (z@u)*u; z = z/np.linalg.norm(z)
    v = c*u + np.sqrt(max(0,1-c*c))*z
    return v*np.linalg.norm(kA)


def cosv(a,b): return float(a@b/((np.linalg.norm(a)*np.linalg.norm(b))+1e-12))


def main():
    hosts = sorted({Path(p).stem.split("fmax_enroll_")[1].rsplit("_r",1)[0]
                    for p in glob.glob(str(OUT/"fmax_enroll_*_r1.npz"))})
    if len(hosts)<2: print("need 2 dies"); return
    A,B = hosts[0],hosts[1]
    kA,kB = die_key(A),die_key(B); Din=len(kA)
    cAB = cosv(kA,kB)
    print(f"dies: A={A} B={B}  real key cos(A,B)={cAB:+.3f}  Din={Din}", flush=True)

    SEEDS=12; Dh=512
    Xtr,ytr=make_task(5000,Din,np.random.default_rng(100))
    Xte,yte=make_task(4000,Din,np.random.default_rng(101))

    # dose-response: acc of die-A readout vs key cosine-to-A, averaged over reservoir seeds
    coss=[1.0,0.98,0.95,0.90,0.86,0.80,0.70,0.50,0.20,0.0]
    table={c:[] for c in coss}; dieB=[]; shuf=[]; zero=[]; base=[]
    for s in range(SEEDS):
        res=Reservoir(Din,Dh,seed=s)
        w=ridge_fit(res.states(Xtr,kA),ytr)            # readout trained on die-A
        base.append(acc(res.states(Xte,kA),yte,w))
        rng=np.random.default_rng(2000+s)
        for c in coss:
            kc=key_at_cos(kA,c,rng); table[c].append(acc(res.states(Xte,kc),yte,w))
        dieB.append(acc(res.states(Xte,kB),yte,w))
        kS=rng.permutation(kA.copy()); shuf.append(acc(res.states(Xte,kS),yte,w))
        zero.append(acc(res.states(Xte,kA*0),yte,w))
    bm=np.mean(base)
    print(f"\nPOSITIVE CONTROL die-A test acc = {bm:.3f} ± {np.std(base):.3f}  (must be high for a valid null)", flush=True)
    print(f"chance = {max(yte.mean(),1-yte.mean()):.3f}", flush=True)
    print("\ndose-response (key cosine-to-A -> die-A-readout acc):")
    for c in coss:
        m=np.mean(table[c]); print(f"  cos={c:.2f}  acc={m:.3f} ± {np.std(table[c]):.3f}  drop={bm-m:+.3f}", flush=True)
    bb=np.mean(dieB); sh=np.mean(shuf); ze=np.mean(zero)
    print(f"\n  REAL die-B (cos={cAB:.2f}): acc={bb:.3f} ± {np.std(dieB):.3f}  drop={bm-bb:+.3f}", flush=True)
    print(f"  shuffled key:           acc={sh:.3f}  drop={bm-sh:+.3f}", flush=True)
    print(f"  zero key:               acc={ze:.3f}  drop={bm-ze:+.3f}", flush=True)
    # discrimination threshold: cos where acc drops to halfway base->chance
    ch=max(yte.mean(),1-yte.mean()); half=(bm+ch)/2
    thr=None
    for c in sorted(coss, reverse=True):   # scan from cos=1 downward; first crossing = highest cos that breaks to half
        if np.mean(table[c])<=half: thr=c; break
    print(f"\n  discrimination threshold (highest cos with acc<=halfway {half:.2f}): cos = {thr}", flush=True)
    verdict=(bm-bb>0.05)
    print(f"\nVERDICT: {'die-B BREAKS the die-A readout -> soft tier DOES bind these 2 dies' if verdict else 'die-B SURVIVES -> dies too similar for soft tier; need hard crypto key-gate'}", flush=True)
    print(f"  (real cos {cAB:.2f} is {'ABOVE' if (thr is not None and cAB<thr) else 'BELOW'} the discrimination threshold {thr})", flush=True)


if __name__=="__main__":
    main()
