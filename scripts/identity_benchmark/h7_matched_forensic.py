"""Forensic on the ratio=3.30 matched-config result: is the surviving INTER silicon, or residual
operating-point/config? Decompose INTER=16.1 by channel and ask how much is temp-correlated, and
re-test the ratio with daedalus's drifted r2 down-weighted."""
import numpy as np, glob
from pathlib import Path
OUT=Path("results/IDENTITY_H7_2026-06-09")
runs={}
for f in sorted(glob.glob(str(OUT/"matched_*.npz"))):
    stem=Path(f).stem.split("matched_")[1]; host,tag=stem.rsplit("_",1)
    d=np.load(f); runs[(host,tag)]=(d["snaps"],d["temps"])
Nf=min(s.shape[1] for s,_ in runs.values())
S={k:s[:,:Nf] for k,(s,_) in runs.items()}; T={k:t for k,(_,t) in runs.items()}
stds=np.array([S[k].std(0) for k in S]); dynamic=(stds>1e-6).all(0)
nd=int(dynamic.sum())
alld=np.vstack([S[k][:,dynamic] for k in S]); pooled=alld.std(0)+1e-9
Z={k:S[k][:,dynamic]/pooled for k in S}
mv={k:Z[k].mean(0) for k in S}

A,B="daedalus","ikaros"
inter_vec=0.5*((mv[(A,"r1")]+mv[(A,"r2")])/1 - (mv[(B,"r1")]+mv[(B,"r2")])/1)/1  # placeholder
# proper: mean die vector
dieA=(mv[(A,"r1")]+mv[(A,"r2")])/2; dieB=(mv[(B,"r1")]+mv[(B,"r2")])/2
chan_inter=np.abs(dieA-dieB)                      # per dynamic-channel inter separation
# temp correlation of each dynamic channel within a die (avg over the 4 runs)
def tcorr(k):
    s=Z[k]; t=T[k]
    return np.array([abs(np.corrcoef(s[:,j],t)[0,1]) if s[:,j].std()>1e-12 else 0 for j in range(s.shape[1])])
tc=np.mean([tcorr(k) for k in S],0)
order=np.argsort(-chan_inter)
tot=chan_inter.sum()
print(f"dynamic channels={nd}  total |INTER| (L1)={tot:.1f}")
print(f"top-10 INTER-driving channels:  sep    |r_temp|")
for j in order[:10]: print(f"   {chan_inter[j]:8.3f}   {tc[j]:.2f}")
# fraction of inter carried by temp-correlated (|r|>0.3) channels
hi=tc>0.3
print(f"\nINTER carried by temp-correlated (|r_temp|>0.3) channels: {chan_inter[hi].sum()/tot:.1%}  ({hi.sum()}/{nd} channels)")
print(f"INTER carried by top-5 channels alone: {chan_inter[order[:5]].sum()/tot:.1%} (concentration check)")

# temp gap between die means
gapT=np.mean([T[(A,'r1')].mean(),T[(A,'r2')].mean()])-np.mean([T[(B,'r1')].mean(),T[(B,'r2')].mean()])
print(f"\nmean-temp gap daedalus-ikaros = {gapT:+.2f}C  (residual operating-point difference)")

# ratio sensitivity: use each die's own intra, and a temp-residualized version
def L2(a,b): return float(np.linalg.norm(a-b))
intraA=L2(mv[(A,"r1")],mv[(A,"r2")]); intraB=L2(mv[(B,"r1")],mv[(B,"r2")])
interp=[L2(mv[(A,ta)],mv[(B,tb)]) for ta in["r1","r2"] for tb in["r1","r2"]]
print(f"\nratio sensitivity:")
print(f"   using mean intra ({(intraA+intraB)/2:.2f}): {np.mean(interp)/((intraA+intraB)/2):.2f}")
print(f"   using daedalus intra ({intraA:.2f}) (worst): {np.mean(interp)/intraA:.2f}")
print(f"   using ikaros intra ({intraB:.2f}) (best):    {np.mean(interp)/intraB:.2f}")

# temp-residualize each dynamic channel within each run (remove linear temp dependence), recompute
def resid(k):
    s=Z[k]; t=T[k]; t=(t-t.mean())
    out=s.copy()
    if t.std()>1e-9:
        for j in range(s.shape[1]):
            b=np.polyfit(t,s[:,j],1)[0]; out[:,j]=s[:,j]-b*t
    return out.mean(0)
mvr={k:resid(k) for k in S}
dieAr=(mvr[(A,"r1")]+mvr[(A,"r2")])/2; dieBr=(mvr[(B,"r1")]+mvr[(B,"r2")])/2
interr=[L2(mvr[(A,ta)],mvr[(B,tb)]) for ta in["r1","r2"] for tb in["r1","r2"]]
intraAr=L2(mvr[(A,"r1")],mvr[(A,"r2")]); intraBr=L2(mvr[(B,"r1")],mvr[(B,"r2")])
print(f"\nAFTER per-channel temp-residualization (linear temp removed within each run):")
print(f"   INTER={np.mean(interr):.2f}  INTRA(mean)={(intraAr+intraBr)/2:.2f}  ratio={np.mean(interr)/((intraAr+intraBr)/2):.2f}")
print(f"   -> if ratio stays >3 after temp removal: not just the 0.9C gap. if it collapses: it WAS operating-point.")
