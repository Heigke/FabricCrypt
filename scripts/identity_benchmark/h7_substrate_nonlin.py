"""H7 substrate-nonlinearity probe — Eric's point: drive with X, 2X, 3X; metrics saturate => the BODY
itself is nonlinear, not just a nonlinear readout.

A binary (0/1) drive is two points => cannot reveal curvature. Here we drive MULTI-LEVEL loads
L in {0..K-1} (L back-to-back matmuls per window) and test two things:

  1. RESPONSE CURVE: steady-state metric vs load level. Curvature = normalized 2nd finite difference.
     If |curv| is large for some channels, the substrate response is intrinsically NONLINEAR in load.

  2. SUBSTRATE NONLINEAR COMPUTATION with a strictly LINEAR readout:
     target f(L[t-1]) = a NON-MONOTONE nonlinear map of ONE past load, e.g. {0,1,2,3}->{0,1,1,0}
     (an "is-mid" / XOR-of-bits function). A linear readout on the raw load level L CANNOT fit a
     non-monotone f. If a linear readout on TELEMETRY fits it, the body's own nonlinearity supplied it.
     Controls: linear-on-raw-level baseline (must fail); shuffle null (must collapse).
     Also a delayed 2-input XOR from level-derived bits, linear readout.

Out: substrate_nonlin_{host}.json (+ verdict). Thermal-safe (gentle bursts + relaxation gap).
"""
from __future__ import annotations
import os, sys, time, json, argparse
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h7_rt_phase0 as P0
from h7_telemetry_reservoir import snap_vec

HOST = P0.HOST
THERM_ABORT = float(os.environ.get("THERM_ABORT", "78"))
THERM_COOL  = float(os.environ.get("THERM_COOL", "50"))

def make_leveldriver(dev, n, levels):
    """Level scales matmul SIZE (clear, wide, monotone load axis): level L -> size n_L spanning
    ~35%..100% of n. During the burst window we REPEAT the level's matmul to fill the time, so the
    instantaneous load (size) differs per level while total burst time is fixed. Read is immediate."""
    import torch
    sizes = [max(64, int(n * (0.35 + 0.65*L/(max(1,levels-1))))) for L in range(levels)]
    if dev == "cuda":
        mats = [(torch.randn(s, s, device="cuda"), torch.randn(s, s, device="cuda")) for s in sizes]
        def drive(level, secs):
            A, B = mats[level]
            t0 = time.monotonic()
            if level == 0:                          # level 0 = idle baseline
                time.sleep(secs); return
            while time.monotonic() - t0 < secs:
                C = A @ B; A.copy_(torch.tanh(C))
            torch.cuda.synchronize()
        return drive, sizes
    else:
        mats = [(np.random.randn(s, s), np.random.randn(s, s)) for s in sizes]
        def drive(level, secs):
            A, B = mats[level]; t0 = time.monotonic()
            if level == 0: time.sleep(secs); return
            while time.monotonic() - t0 < secs: A = np.tanh(A @ B)
        return drive, sizes

def record(steps, levels, base_secs, mat_n, gap, seed):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    L = rng.integers(0, levels, size=steps).astype(np.int8)
    drive, sizes = make_leveldriver(dev, mat_n, levels)
    samp = P0.Sampler(period=0.0)
    x0, names = snap_vec(samp); D = x0.size
    X = np.full((steps, D), np.nan); seg = np.zeros(steps, np.int32); zt = np.zeros(steps)
    cur = 0
    print(f"[{HOST}] substrate-nonlin dev={dev} steps={steps} levels={levels} sizes={sizes} "
          f"base={base_secs}s gap={gap}s D={D}", flush=True)
    # STEADY response-curve probe: hold each level, read IMMEDIATELY after the burst (saturation shows)
    curve = {}
    hold = 8
    for lv in range(levels):
        vals = []
        for _ in range(hold):
            if P0.zone0() >= THERM_ABORT: P0.wait_cool(THERM_COOL)
            drive(lv, base_secs); v, _ = snap_vec(samp)   # immediate read
            if v.size == D: vals.append(v)
            time.sleep(gap)                                 # relax AFTER reading (thermal bound)
        curve[lv] = np.nanmean(vals, 0) if vals else np.full(D, np.nan)
        print(f"  curve level {lv}: zone0={P0.zone0():.0f}C", flush=True)
    curve_mat = np.array([curve[lv] for lv in range(levels)])  # [levels, D]
    # main random-sequence recording (immediate read, then relax)
    for t in range(steps):
        if P0.zone0() >= THERM_ABORT: P0.wait_cool(THERM_COOL); cur += 1
        drive(int(L[t]), base_secs); v, _ = snap_vec(samp)    # immediate read
        if v.size == D: X[t] = v
        time.sleep(gap)
        seg[t] = cur; zt[t] = P0.zone0()
        if (t+1) % 50 == 0: print(f"  step {t+1}/{steps} seg={cur} zone0={zt[t]:.0f}C", flush=True)
    return dict(host=HOST, dev=dev, L=L, X=X, seg=seg, zone=zt, names=np.array(names),
                curve=curve_mat, levels=levels, sizes=np.array(sizes)), names

def curvature(curve_mat):
    """Per-channel normalized |2nd finite difference| averaged over interior levels."""
    K = curve_mat.shape[0]
    if K < 3: return None
    out = {}
    for j in range(curve_mat.shape[1]):
        y = curve_mat[:, j]
        if np.nanstd(y) < 1e-9 or np.isnan(y).any(): continue
        yn = (y - y.min()) / (y.max() - y.min() + 1e-12)   # normalize range to [0,1]
        d2 = np.abs(np.diff(yn, 2))                          # 2nd difference
        out[j] = float(d2.mean())
    return out

def lin_acc_reg(X, y, valid, seg, lam=5.0, folds=5, shuffle=False, binary=True):
    idx = np.where(valid & ~np.isnan(y))[0]
    if idx.size < 40: return np.nan
    F = X[idx]; yy = y[idx].astype(float)
    mu = F.mean(0); sd = F.std(0)+1e-8; F = (F-mu)/sd
    if shuffle:
        rng = np.random.default_rng(2); F = F[rng.permutation(len(F))]
    F = np.column_stack([F, np.ones(len(F))]); n=len(idx); bs=max(1,n//folds); pr=np.full(n,np.nan)
    for k in range(folds):
        te=np.zeros(n,bool); te[k*bs:(k+1)*bs if k<folds-1 else n]=True; tr=~te
        if tr.sum()<10 or te.sum()<2: continue
        A=F[tr]; W=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@yy[tr]); pr[te]=F[te]@W
    m=~np.isnan(pr)
    if m.sum()<5: return np.nan
    if binary: return float(((pr[m]>0.5).astype(int)==yy[m].astype(int)).mean())
    sst=((yy[m]-yy[m].mean())**2).sum(); ssr=((yy[m]-pr[m])**2).sum()
    return 1-ssr/sst if sst>1e-9 else np.nan

def analyze(rec, w=3):
    X=rec["X"]; L=rec["L"].astype(int); seg=rec["seg"]; T=len(L); levels=rec["levels"]
    feats=[]; valid=np.ones(T,bool)
    for lag in range(w+1):
        sh=np.full_like(X,np.nan); sh[lag:]=X[:T-lag]
        same=np.zeros(T,bool); same[lag:]=seg[:T-lag]==seg[lag:]; valid&=same; feats.append(sh)
    F=np.concatenate(feats,1); valid&=~np.isnan(F).any(1)
    keep=F[valid].std(0)>1e-6 if valid.any() else np.ones(F.shape[1],bool); F=F[:,keep]
    # raw-level lagged baseline (telemetry-free)
    LB=np.column_stack([np.r_[[np.nan]*l, L[:T-l]].astype(float) for l in range(0,w+1)])
    # target 1: non-monotone f(L[t-1]) = is-mid map {0,1,2,3}->{0,1,1,0} (generalize: 1 if 0<L<levels-1)
    f1=np.full(T,np.nan)
    for t in range(1,T): f1[t]=1.0 if (0 < L[t-1] < levels-1) else 0.0
    # target 2: XOR of hi-bits of two past levels (bit = L>=levels/2)
    b=(L>=levels/2).astype(int); xor=np.full(T,np.nan)
    for t in range(2,T): xor[t]=b[t-1]^b[t-2]
    res={"levels":levels,"n_valid":int(valid.sum())}
    res["f_nonmono_L1"]={
        "telem_lin": round(lin_acc_reg(F,f1,valid,seg),3),
        "rawlevel_lin": round(lin_acc_reg(LB[:,1:2],f1,valid,seg),3),   # linear on the single raw level
        "telem_shuffle": round(lin_acc_reg(F,f1,valid,seg,shuffle=True),3),
        "chance": round(float(max(np.nanmean(f1),1-np.nanmean(f1))),3)}
    res["xor_hibits"]={
        "telem_lin": round(lin_acc_reg(F,xor,valid,seg),3),
        "rawbit_lin": round(lin_acc_reg(LB[:,1:3],xor,valid,seg),3),
        "telem_shuffle": round(lin_acc_reg(F,xor,valid,seg,shuffle=True),3),
        "chance": round(float(max(np.nanmean(xor),1-np.nanmean(xor))),3)}
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=500)
    ap.add_argument("--levels",type=int,default=4)
    ap.add_argument("--base",type=float,default=0.05)
    ap.add_argument("--gap",type=float,default=0.13)
    ap.add_argument("--mat_n",type=int,default=768)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--w",type=int,default=3)
    a=ap.parse_args()
    rec,names=record(a.steps,a.levels,a.base,a.mat_n,a.gap,a.seed)
    OUT=P0.OUT
    p=OUT/f"substrate_nonlin_{HOST}.npz"
    np.savez_compressed(p.with_name(p.stem+".tmp.npz"),**rec); os.replace(p.with_name(p.stem+".tmp.npz"),p)
    curv=curvature(rec["curve"])
    top=sorted(curv.items(),key=lambda kv:-kv[1])[:8] if curv else []
    comp=analyze(rec,w=a.w)
    res={"host":HOST,"dev":rec["dev"],"steps":a.steps,"levels":a.levels,"D":int(rec["X"].shape[1]),
         "response_curve_top_curvature":[[names[j],round(c,4)] for j,c in top],
         "median_curvature":round(float(np.median(list(curv.values()))),4) if curv else None,
         **comp}
    jp=OUT/f"substrate_nonlin_{HOST}.json"; jp.write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2),flush=True)
    f1=res["f_nonmono_L1"]; xr=res["xor_hibits"]
    print(f"\n[{HOST}] SUBSTRATE-NONLIN VERDICT:",flush=True)
    print(f"  median channel curvature={res['median_curvature']} (｜2nd diff｜; >~0.05 = real curvature)")
    print(f"  f(L)=is-mid: telem-LINEAR={f1['telem_lin']} vs raw-level-linear={f1['rawlevel_lin']} "
          f"shuffle={f1['telem_shuffle']} chance={f1['chance']}")
    print(f"  XOR(hibits): telem-LINEAR={xr['telem_lin']} vs raw-bit-linear={xr['rawbit_lin']} "
          f"shuffle={xr['telem_shuffle']} chance={xr['chance']}")
    print(f"  -> SUBSTRATE nonlinear if telem-LINEAR ≫ raw-linear & ≫ shuffle. saved {jp}")

if __name__=="__main__":
    main()
