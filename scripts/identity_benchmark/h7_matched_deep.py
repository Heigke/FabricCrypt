"""Deeper test on matched-config data: does the JOINT/interaction structure (covariance, cross-channel
coupling) separate dies — beyond per-channel mean offsets (which we showed are config leakage)?
Eric's push: 'how all signals interact, inertia, latency, momentum' = the joint structure, not the means.

Two config-immune tests on the genuinely-fluctuating channels (step-like per-die constants removed):
  (1) INTERACTION FINGERPRINT: per-run cross-channel correlation matrix. Frobenius distance within-die
      (r1 vs r2) vs between-die. Correlation is mean/scale invariant -> kills per-channel config offset.
  (2) HELD-OUT CLASSIFY: train a whitened nearest-centroid die-classifier on run r1 (both dies),
      test on run r2 snapshots (and swap). Permutation null (shuffle die labels) for a real p-value.
      Run on: all-dynamic / genuine-dynamic / interaction-features. If genuine-dynamic CV >> chance
      with low p -> a real die-specific live signal exists (vindicates 'it must be in the murmur').
      If it collapses to chance on genuine-dynamic -> the separation really was config, honest negative.
"""
import numpy as np, glob
from pathlib import Path
OUT=Path("results/IDENTITY_H7_2026-06-09")
runs={}
for f in sorted(glob.glob(str(OUT/"matched_*.npz"))):
    stem=Path(f).stem.split("matched_")[1]; host,tag=stem.rsplit("_",1)
    d=np.load(f); runs[(host,tag)]=d["snaps"]
Nf=min(s.shape[1] for s in runs.values())
S={k:s[:,:Nf] for k,s in runs.items()}
A,B="daedalus","ikaros"
rng=np.random.default_rng(0)

# channel taxonomy
within=np.sqrt(np.mean([S[k].var(0) for k in S],0))
dieA=np.mean([S[(A,'r1')].mean(0),S[(A,'r2')].mean(0)],0)
dieB=np.mean([S[(B,'r1')].mean(0),S[(B,'r2')].mean(0)],0)
gap=np.abs(dieA-dieB); gw=gap/(within+1e-12)
masks={
 "all-dynamic":      (np.array([S[k].std(0) for k in S])>1e-6).all(0),
 "genuine-dynamic":  ((gw<1)&(within>1e-3)),
 "strict-fluct":     ((gw<0.5)&(within>1e-3)),
}

def corr(X):
    Xc=X-X.mean(0); s=Xc.std(0)+1e-12; Z=Xc/s
    return (Z.T@Z)/len(Z)
def fro(a,b): return float(np.linalg.norm(a-b))

print("=== TEST 1: interaction (cross-channel correlation) fingerprint ===")
for name,m in masks.items():
    nd=int(m.sum())
    if nd<5: print(f"  [{name}] n={nd} too few"); continue
    C={k:corr(S[k][:,m]) for k in S}
    intra=np.mean([fro(C[(A,'r1')],C[(A,'r2')]), fro(C[(B,'r1')],C[(B,'r2')])])
    inter=np.mean([fro(C[(A,a)],C[(B,b)]) for a in['r1','r2'] for b in['r1','r2']])
    print(f"  [{name:16s}] n={nd:3d}  INTRA(corr r1vr2)={intra:.2f}  INTER(corr A vs B)={inter:.2f}  ratio={inter/(intra+1e-9):.2f}")
print("  -> ratio>>1 : the WAY channels co-vary is die-specific (config-immune handle). ~1 : interaction identical.\n")

print("=== TEST 2: held-out die classification (train r1, test r2; whitened nearest-centroid) ===")
def whiten_fit(X):
    mu=X.mean(0); Xc=X-mu
    cov=np.cov(Xc.T)+1e-3*np.eye(Xc.shape[1])
    # diagonal whitening (robust for n<<p)
    w=1.0/np.sqrt(np.diag(cov)+1e-9)
    return mu,w
def classify(train, test):
    # train: dict die->X ; test: dict die->X. nearest standardized centroid.
    allX=np.vstack([train[A],train[B]])
    mu=allX.mean(0); sd=allX.std(0)+1e-9
    cA=((train[A]-mu)/sd).mean(0); cB=((train[B]-mu)/sd).mean(0)
    correct=0; tot=0
    for die,Xt in test.items():
        Zt=(Xt-mu)/sd
        dA=np.linalg.norm(Zt-cA,axis=1); dB=np.linalg.norm(Zt-cB,axis=1)
        pred=np.where(dA<dB,A,B)
        correct+=(pred==die).sum(); tot+=len(Xt)
    return correct/tot
for name,m in masks.items():
    nd=int(m.sum())
    if nd<5: continue
    tr={A:S[(A,'r1')][:,m],B:S[(B,'r1')][:,m]}; te={A:S[(A,'r2')][:,m],B:S[(B,'r2')][:,m]}
    acc1=classify(tr,te)
    tr2={A:S[(A,'r2')][:,m],B:S[(B,'r2')][:,m]}; te2={A:S[(A,'r1')][:,m],B:S[(B,'r1')][:,m]}
    acc2=classify(tr2,te2)
    acc=(acc1+acc2)/2
    # permutation null: shuffle which run-half belongs to which 'die' label, same-die data
    # build a SAME-DIE null: split daedalus r1/r2 vs ikaros r1/r2 but permute die labels across pooled snaps
    pooled={A:np.vstack([S[(A,'r1')][:,m],S[(A,'r2')][:,m]]), B:np.vstack([S[(B,'r1')][:,m],S[(B,'r2')][:,m]])}
    nperm=200; ge=0
    allp=np.vstack([pooled[A],pooled[B]]); nA=len(pooled[A])
    for _ in range(nperm):
        idx=rng.permutation(len(allp)); pa=allp[idx[:nA]]; pb=allp[idx[nA:]]
        h=len(pa)//2; hb=len(pb)//2
        a0=classify({A:pa[:h],B:pb[:hb]},{A:pa[h:],B:pb[hb:]})
        if a0>=acc: ge+=1
    p=(ge+1)/(nperm+1)
    print(f"  [{name:16s}] n={nd:3d}  held-out acc={acc:.3f} (chance=0.5)  perm-null p={p:.3f}")
print("  -> acc>>0.5 with p<0.05 on genuine-dynamic : real die-specific live signal. acc~0.5 : config was all there was.")
