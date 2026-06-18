#!/usr/bin/env python3
"""User question: 'search deep and wide ... missed dynamics or derivatives or
acceleration or some combination of differentiation of metrics that could unveil
the identity, jacobian but for identity metrics?'

Deep-dive on Phase 13 sig_v2 data (10 reps x 290 dims per host). We compute
many derived feature families and rank each by how much it adds to host
discrimination over and above the raw static features.

Families tested:
  F0 static           - baseline (raw rep values)
  F1 d/d_rep          - first temporal difference (rep is implicit time)
  F2 d^2/d_rep^2      - second difference (acceleration)
  F3 log|x|           - log magnitude
  F4 z-score within rep (whitened)
  F5 pairwise ratios  - x_i / x_j across rep
  F6 Jacobian J[i,j]=corr(dx_i, x_j)  -> use eigenvalues + Frobenius as feats
  F7 Hessian H[i,j]=corr(d2x_i, x_j)
  F8 lag-1 autocorr per dim  (memory)
  F9 rep-to-rep variance / mean (coefficient of variation)
 F10 cross-dim covariance off-diagonal magnitudes
 F11 fractal scaling: log(std at scale s) vs log(s) slope
 F12 rolling-window slope (linear regression dy/drep)
 F13 phase-portrait area (x[t-1] vs x[t] enclosed area approximation)
 F14 PCA spectrum (top eigenvalues of covariance)

For each family we report KS-D vs the SAME family across hosts.
Within-host SHUFFLE baseline: re-permute reps within one host and re-test;
families with > shuffle null KS-D have real per-host structure.
"""
import os, sys, json
import numpy as np
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', '..',
    'results', 'IDENTITY_BENCHMARK_2026-05-30'))
EM13 = os.path.join(RESULTS, 'embodiment13')

def _safe_corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def family_static(X): return X.copy()
def family_d1(X):     return np.diff(X, axis=0)
def family_d2(X):     return np.diff(X, axis=0, n=2)
def family_log(X):    return np.sign(X) * np.log1p(np.abs(X))
def family_zwithin(X):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    return (X - mu) / sd
def family_logvar_d1(X):
    # log-variance of first differences per-dim (single scalar per dim)
    d1 = np.diff(X, axis=0)
    return np.log1p(d1.var(0))[None, :]  # (1, d)
def family_cv(X):
    # coefficient of variation per-dim (per rep set)
    return (X.std(0) / (np.abs(X.mean(0)) + 1e-9))[None, :]
def family_lag1(X):
    # lag-1 autocorrelation per dim
    out = np.zeros(X.shape[1])
    for d in range(X.shape[1]):
        a = X[:-1, d]; b = X[1:, d]
        out[d] = _safe_corr(a, b)
    return out[None, :]
def family_pca_spectrum(X, top=8):
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / max(1, X.shape[0]-1)
    w = np.linalg.eigvalsh(C)[::-1]
    return w[:top][None, :]
def family_jacobian_eig(X, top=8):
    d1 = np.diff(X, axis=0); Xa = X[:-1]
    n = X.shape[1]
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = _safe_corr(d1[:, i], Xa[:, j])
    eig = np.sort(np.linalg.eigvalsh((J+J.T)/2))[::-1]
    return eig[:top][None, :]
def family_jacobian_summary(X):
    """Return (frobenius, sum_abs_diag, max_offdiag, eig_max, eig_spread)."""
    d1 = np.diff(X, axis=0); Xa = X[:-1]
    n = X.shape[1]
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = _safe_corr(d1[:, i], Xa[:, j])
    eig = np.linalg.eigvalsh((J+J.T)/2)
    feats = np.array([np.linalg.norm(J),
                      np.abs(np.diag(J)).sum(),
                      np.abs(J - np.diag(np.diag(J))).max(),
                      eig.max(),
                      eig.max() - eig.min()])
    return feats[None, :]
def family_acceleration(X):
    """Accel = d2x, but here we keep per-rep accel + magnitude."""
    d2 = np.diff(X, axis=0, n=2)
    return d2

FAMILIES = {
    'F0_static': family_static,
    'F1_d1_rep': family_d1,
    'F2_d2_rep': family_d2,
    'F3_logmag': family_log,
    'F4_zwithin': family_zwithin,
    'F8_lag1':   family_lag1,
    'F9_cv':     family_cv,
    'F11_logvar_d1': family_logvar_d1,
    'F14_pca_spectrum': family_pca_spectrum,
    'F6_jac_eig8': family_jacobian_eig,
    'F6b_jac_summary': family_jacobian_summary,
    'F12_accel':    family_acceleration,
}

def evaluate(ik, dd, fn):
    """Return dict of per-family stats."""
    A = fn(ik); B = fn(dd)
    n_dim = A.shape[1]
    Ds = np.zeros(n_dim); ps = np.zeros(n_dim)
    for d in range(n_dim):
        a = A[:, d]; b = B[:, d]
        if a.size < 2 or b.size < 2 or (a.std() == 0 and b.std() == 0):
            Ds[d] = 0; ps[d] = 1.0
        else:
            r = ks_2samp(a, b); Ds[d] = r.statistic; ps[d] = r.pvalue
    return {
        'shape_A': A.shape,
        'max_KS_D': float(Ds.max()),
        'mean_KS_D': float(Ds.mean()),
        'frac_p_lt_0p01': float((ps < 0.01).mean()),
        'min_p': float(ps.min()),
        'n_dim_effective': int((np.isfinite(Ds) & (Ds > 0)).sum()),
        'distance_centroid_l2': float(np.linalg.norm(A.mean(0) - B.mean(0))),
    }

def shuffle_null(ik, fn, n_iter=20):
    """Random rep permutation within ikaros: should give ~0 KS-D."""
    rng = np.random.default_rng(0)
    Ds = []
    for _ in range(n_iter):
        idx = rng.permutation(ik.shape[0])
        half = ik.shape[0] // 2
        A = fn(ik[idx[:half]]); B = fn(ik[idx[half:]])
        n = A.shape[1]
        d_ = []
        for d in range(min(n, 50)):
            a = A[:, d]; b = B[:, d]
            if a.size < 2 or b.size < 2 or (a.std() == 0 and b.std() == 0): continue
            r = ks_2samp(a, b); d_.append(r.statistic)
        if d_: Ds.append(np.max(d_))
    return float(np.mean(Ds)) if Ds else 0.0

def main():
    ik = np.load(os.path.join(EM13, 'ikaros_sig_v2.npz'))['vec']
    dd = np.load(os.path.join(EM13, 'daedalus_sig_v2.npz'))['vec']
    n = min(ik.shape[0], dd.shape[0])
    ik = ik[:n]; dd = dd[:n]
    out = {'n_reps': int(n), 'n_static_dims': int(ik.shape[1]), 'families': {}}
    for name, fn in FAMILIES.items():
        try:
            stats = evaluate(ik, dd, fn)
            stats['null_max_KS_D'] = shuffle_null(ik, fn)
            stats['signal_above_null'] = stats['max_KS_D'] - stats['null_max_KS_D']
            out['families'][name] = stats
            print(f"{name}: max_KS_D={stats['max_KS_D']:.3f} "
                  f"null={stats['null_max_KS_D']:.3f} "
                  f"frac_p<0.01={stats['frac_p_lt_0p01']:.2%} "
                  f"d={stats['shape_A'][1]}", flush=True)
        except Exception as e:
            out['families'][name] = {'error': str(e)}
            print(f"{name}: ERROR {e}")
    # rank by (signal_above_null * frac_p)
    rank = []
    for k, v in out['families'].items():
        if 'error' in v: continue
        score = v.get('signal_above_null', 0) * v.get('frac_p_lt_0p01', 0)
        rank.append((k, score, v['max_KS_D'], v['null_max_KS_D'],
                     v['frac_p_lt_0p01']))
    rank.sort(key=lambda x: -x[1])
    out['ranking'] = rank
    print("\n=== RANKING (signal*frac_p) ===")
    for r in rank:
        print(f"  {r[0]:<22}  score={r[1]:.3f}  D={r[2]:.3f}  null={r[3]:.3f}  frac={r[4]:.2%}")
    out_path = os.path.join(HERE, '..', '..', '..',
        'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment19', 'dynamics_deepdive.json')
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")

if __name__ == '__main__':
    main()
