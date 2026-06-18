"""H7 coherent-data forensics — the DSP oracle's decisive test.

Two questions the d'(M)=283e9 'win' cannot answer on its own:
  (1) TEMP CONFOUND: ikaros enrolled at 52.6C, daedalus at 37.4C (no overlap on this pair).
      Is the cross-die 'separation' just the 15C operating-point gap + config, or silicon identity?
      -> measure how much of inter-distance lives in channels that ALSO move with temp/config.
  (2) ALLAN FLOOR: does within-die averaging keep helping (white -> sigma ~ 1/sqrt(M)),
      or hit a 1/f drift floor (Allan deviation flattens => coherent integration is futile)?
      This is the literal 'does the radar trick work' test, on each die's own snapshot stream.
"""
import numpy as np, glob
from pathlib import Path
OUT=Path("results/IDENTITY_H7_2026-06-09")
fs=sorted(glob.glob(str(OUT/"coh_*_r1.npz")))
D={Path(f).stem.split("coh_")[1].rsplit("_",1)[0]:np.load(f) for f in fs}
print("dies:",list(D))

def allan_dev(x):
    """overlapping Allan deviation vs averaging factor m (per channel, averaged over channels)."""
    M=len(x); ms=[m for m in [1,2,4,8,16,32,64,128] if 2*m<M]
    out=[]
    for m in ms:
        # block means at scale m
        nb=M//m; b=x[:nb*m].reshape(nb,m,-1).mean(1)         # (nb, K)
        d=np.diff(b,axis=0)                                   # successive differences
        av=np.sqrt(0.5*np.nanmean(d**2,axis=0))               # Allan dev per channel
        out.append(np.nanmean(av))                            # mean over channels
    return np.array(ms),np.array(out)

for h,d in D.items():
    s=d["snaps"]; s=(s-s.mean(0))/(s.std(0)+1e-9)             # standardize per channel
    ms,ad=allan_dev(s)
    white=ad[0]/np.sqrt(ms)                                   # ideal white-noise 1/sqrt(m)
    print(f"\n[{h}] Allan deviation vs averaging m (standardized, mean over K):")
    print(f"   {'m':>4} {'allan':>8} {'white~1/√m':>11} {'ratio(allan/white)':>20}")
    for i,m in enumerate(ms):
        print(f"   {m:4d} {ad[i]:8.4f} {white[i]:11.4f} {ad[i]/white[i]:20.2f}")
    floor = ad[-1]/ad[0]
    ideal = 1/np.sqrt(ms[-1])
    print(f"   -> at m={ms[-1]}: allan dropped to {floor:.3f}x ; white would give {ideal:.3f}x")
    print(f"   -> {'WHITE-LIKE: averaging keeps helping' if floor < 2*ideal else 'FLOOR/1-f-LIKE: averaging stalls (radar trick futile)'}")

# (1) temp-confound decomposition on the inter distance
A,B=list(D)
sa=D[A]["snaps"]; sb=D[B]["snaps"]; ta=D[A]["temps"]; tb=D[B]["temps"]
pooled=np.sqrt((sa.var(0)+sb.var(0))/2)+1e-9
ma=(sa.mean(0))/pooled; mb=(sb.mean(0))/pooled
chan_diff=np.abs(ma-mb)                                        # per-channel inter separation (std units)
# how does each channel co-move with temperature WITHIN a die? (|corr with temp|)
def tcorr(s,t):
    return np.array([abs(np.corrcoef(s[:,k],t)[0,1]) if s[:,k].std()>1e-9 else 0 for k in range(s.shape[1])])
tc=(tcorr(sa,ta)+tcorr(sb,tb))/2
order=np.argsort(-chan_diff)
print(f"\n[temp-confound] per-channel inter-separation vs |corr-with-temp| (top 8 separating channels):")
print(f"   {'sep(std)':>9} {'|r_temp|':>9}")
for k in order[:8]:
    print(f"   {chan_diff[k]:9.2f} {tc[k]:9.2f}")
tot=chan_diff.sum(); temp_frac=(chan_diff*tc).sum()/(tot+1e-9)
print(f"   -> fraction of total inter-separation carried by temp-correlated channels: {temp_frac:.1%}")
print(f"   -> ikaros Tmean={ta.mean():.1f}C  daedalus Tmean={tb.mean():.1f}C  gap={abs(ta.mean()-tb.mean()):.1f}C (no overlap)")
