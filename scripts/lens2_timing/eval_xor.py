#!/usr/bin/env python3
"""Rigorous nonlinear-with-memory test for a timing reservoir CSV.

Test definition (police success-bias):
  A channel computes delayed-XOR ONLY IF a LINEAR readout (ridge) of its
  feature vector x[t] predicts y[t]=XOR(u[t-d], u[t-d-1]) on HELD-OUT data
  with accuracy clearly above:
    (1) chance (0.5),
    (2) the SAME linear readout applied to the raw drive history u[t-d..t-d-k]
        (a linear-on-the-drive model). A linear model of the drive CANNOT do
        XOR, so this baseline should sit at ~0.5. If the channel beats it, the
        nonlinearity is in the SUBSTRATE, not the readout.

We also test plain memory (predict u[t-d]) to confirm the channel has fading
memory at all, and we report a feature-shuffle control (destroys temporal
structure -> should collapse to chance).
"""
import sys, numpy as np

def load(path):
    import csv
    rows=[]
    with open(path) as f:
        r=csv.reader(f); hdr=next(r)
        for line in r:
            if not line: continue
            rows.append([float(x) for x in line])
    a=np.array(rows)
    u=a[:,1].astype(int)
    X=a[:,2:]   # features
    return u,X

def ridge_acc(Phi, y, alpha=1.0, train_frac=0.6):
    # standardize
    n=len(y); ntr=int(n*train_frac)
    mu=Phi[:ntr].mean(0); sd=Phi[:ntr].std(0)+1e-9
    P=(Phi-mu)/sd
    P=np.hstack([P, np.ones((n,1))])
    Xtr,ytr=P[:ntr],y[:ntr]; Xte,yte=P[ntr:],y[ntr:]
    yc=ytr*2-1.0
    A=Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1])
    w=np.linalg.solve(A, Xtr.T@yc)
    pred=(Xte@w)>0
    return (pred==(yte>0)).mean()

def windowize(M, k):
    # stack k lagged copies of feature matrix M (rows=time)
    n,d=M.shape
    cols=[]
    for lag in range(k):
        sh=np.zeros_like(M)
        if lag==0: sh=M
        else: sh[lag:]=M[:-lag]
        cols.append(sh)
    return np.hstack(cols)

def main():
    path=sys.argv[1]
    u,X=load(path)
    n=len(u)
    K=8  # how many lagged feature frames the readout may use (memory window)
    print(f"# rows={n} feats={X.shape[1]} window K={K}")
    Phi = windowize(X, K)            # reservoir state with its own short window
    Phi_shuf = Phi.copy(); np.random.RandomState(0).shuffle(Phi_shuf)

    # drive-only baseline: lagged copies of u (linear-on-the-drive)
    Umat = u.reshape(-1,1).astype(float)
    Udrive = windowize(Umat, K+4)

    print(f"{'task':<22}{'reservoir':>11}{'drive-lin':>11}{'shuf-ctrl':>11}")
    for d in range(0,5):
        # delayed XOR of consecutive drive bits
        y = np.zeros(n,int)
        for t in range(n):
            if t-d-1>=0: y[t]= u[t-d]^u[t-d-1]
        a_res = ridge_acc(Phi, y)
        a_drv = ridge_acc(Udrive, y)
        a_shf = ridge_acc(Phi_shuf, y)
        tag=f"XOR(u[t-{d}],u[t-{d+1}])"
        print(f"{tag:<22}{a_res:>11.3f}{a_drv:>11.3f}{a_shf:>11.3f}")
    # plain memory check (linear task) -- both should do this if memory exists
    for d in range(0,5):
        y=np.zeros(n,int)
        for t in range(n):
            if t-d>=0: y[t]=u[t-d]
        a_res=ridge_acc(Phi,y); a_drv=ridge_acc(Udrive,y); a_shf=ridge_acc(Phi_shuf,y)
        print(f"{'MEM u[t-'+str(d)+']':<22}{a_res:>11.3f}{a_drv:>11.3f}{a_shf:>11.3f}")

if __name__=="__main__":
    main()
