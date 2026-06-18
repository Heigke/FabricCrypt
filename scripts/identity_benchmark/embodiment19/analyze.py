#!/usr/bin/env python3
"""Phase 19 analysis:
  Task C: cross-host KS-D per signal (Bonferroni-corrected, alpha=0.01)
  Task D: extend signature_v2 to 290+N, retrain logistic classifier, LOO
  Task ?(answers user): dynamics/Jacobian search across existing Phase 13 data
"""
import os, sys, json, glob
import numpy as np
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', '..',
    'results', 'IDENTITY_BENCHMARK_2026-05-30'))
EM19 = os.path.join(RESULTS, 'embodiment19')
EM13 = os.path.join(RESULTS, 'embodiment13')

SIGNALS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's9']

def _load_pair(name):
    """Load (ikaros, daedalus) signature arrays for signal name."""
    ik = os.path.join(EM19, f'ikaros_{name}.npz')
    dd = os.path.join(EM19, f'daedalus_{name}.npz')
    if not (os.path.exists(ik) and os.path.exists(dd)):
        return None, None
    a = np.load(ik)['vec']; b = np.load(dd)['vec']
    return a, b

def ks_per_signal():
    """Per-signal: max KS-D across dimensions + corrected p-value."""
    out = {}
    n_signals = len(SIGNALS)
    for name in SIGNALS:
        a, b = _load_pair(name)
        if a is None:
            out[name] = {'status': 'missing'}
            continue
        n_dim = a.shape[1]
        Ds = np.zeros(n_dim); ps = np.zeros(n_dim)
        for d in range(n_dim):
            try:
                r = ks_2samp(a[:, d], b[:, d])
                Ds[d] = r.statistic; ps[d] = r.pvalue
            except Exception:
                Ds[d] = 0; ps[d] = 1
        # signal-level: best dim
        argbest = int(np.argmax(Ds))
        # Bonferroni over (n_signals * n_dim) tests
        bonf = n_signals * n_dim
        out[name] = {
            'n_reps_a': int(a.shape[0]), 'n_reps_b': int(b.shape[0]),
            'n_dim': int(n_dim),
            'max_KS_D': float(Ds.max()),
            'argmax_dim': argbest,
            'min_p_raw': float(ps.min()),
            'min_p_bonf': float(min(1.0, ps.min() * bonf)),
            'frac_dims_p_lt_0p01_bonf': float((ps * bonf < 0.01).mean()),
            'mean_KS_D': float(Ds.mean()),
        }
    return out

def extend_signature_loo():
    """Concatenate sig_v2 + Phase 19 signal vectors and LOO-classify."""
    # base 290-dim
    ik13 = np.load(os.path.join(EM13, 'ikaros_sig_v2.npz'))['vec']
    dd13 = np.load(os.path.join(EM13, 'daedalus_sig_v2.npz'))['vec']
    extra_ik, extra_dd, ext_names = [], [], []
    for name in SIGNALS:
        a, b = _load_pair(name)
        if a is None: continue
        n_use = min(a.shape[0], b.shape[0], ik13.shape[0], dd13.shape[0])
        extra_ik.append(a[:n_use]); extra_dd.append(b[:n_use])
        ext_names.append(name)
    if not extra_ik:
        return {'status': 'no_extra_data', 'base_dim': int(ik13.shape[1])}
    n_use = min(ik13.shape[0], dd13.shape[0], min(x.shape[0] for x in extra_ik))
    Xi = np.concatenate([ik13[:n_use]] + [e[:n_use] for e in extra_ik], axis=1)
    Xd = np.concatenate([dd13[:n_use]] + [e[:n_use] for e in extra_dd], axis=1)
    X = np.concatenate([Xi, Xd], axis=0)
    y = np.array([0]*Xi.shape[0] + [1]*Xd.shape[0])
    # normalize each column by std across all reps
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xn = (X - mu) / sd
    # LOO classification via 1-NN cosine
    correct = 0
    for i in range(len(y)):
        train = np.delete(Xn, i, axis=0); yt = np.delete(y, i)
        d = np.linalg.norm(train - Xn[i], axis=1)
        pred = yt[np.argmin(d)]
        if pred == y[i]: correct += 1
    return {
        'base_dim': int(ik13.shape[1]),
        'added_signals': ext_names,
        'added_dims_each': {n: int(e.shape[1]) for n, e in zip(ext_names, extra_ik)},
        'total_dim': int(Xn.shape[1]),
        'n_reps_per_host': int(n_use),
        'loo_accuracy_1NN_cosine': correct / len(y),
        'classifier': '1-NN euclidean on z-scored features',
    }

def dynamics_search_phase13():
    """USER QUESTION: search Phase 13 sig_v2 data for dynamics/Jacobian features
    that might unveil identity better than steady-state stats.

    sig_v2 reps are 10 per host. We treat each block of features as a (10, d)
    time-like sequence (ordered by rep idx, which IS a temporal sequence since
    reps were taken back-to-back over thermal cool cycles). Compute:
       - first-rep diff:   d/drep
       - second-rep diff:  d2/drep2
       - rep-correlation between feature pairs (Jacobian proxy)
       - eigenvalues of cross-feature derivative matrix
    Then KS-test these *derived* features between ikaros & daedalus.
    """
    ik = np.load(os.path.join(EM13, 'ikaros_sig_v2.npz'))['vec']
    dd = np.load(os.path.join(EM13, 'daedalus_sig_v2.npz'))['vec']
    # Match rep counts
    n = min(ik.shape[0], dd.shape[0])
    ik = ik[:n]; dd = dd[:n]
    # Compute derived features per host
    def derive(X):
        d1 = np.diff(X, axis=0)         # (n-1, d)
        d2 = np.diff(d1, axis=0)        # (n-2, d)
        return d1, d2
    d1_ik, d2_ik = derive(ik)
    d1_dd, d2_dd = derive(dd)
    # KS-test each derivative dimension between hosts
    nd = ik.shape[1]
    Ds_static = np.zeros(nd); ps_static = np.zeros(nd)
    Ds_d1 = np.zeros(nd); ps_d1 = np.zeros(nd)
    Ds_d2 = np.zeros(nd); ps_d2 = np.zeros(nd)
    for d in range(nd):
        Ds_static[d], ps_static[d] = ks_2samp(ik[:, d], dd[:, d])
        Ds_d1[d],     ps_d1[d]     = ks_2samp(d1_ik[:, d], d1_dd[:, d])
        Ds_d2[d],     ps_d2[d]     = ks_2samp(d2_ik[:, d], d2_dd[:, d])
    # Jacobian: corr(d1[:, i], static[:, j]) per host
    def jac(X, d1):
        n_dim = X.shape[1]
        J = np.zeros((n_dim, n_dim))
        Xt = X[:-1]  # align with d1
        for i in range(n_dim):
            di = d1[:, i]
            if di.std() < 1e-9: continue
            for j in range(n_dim):
                xj = Xt[:, j]
                if xj.std() < 1e-9: continue
                J[i, j] = np.corrcoef(di, xj)[0, 1]
        return J
    J_ik = jac(ik, d1_ik); J_dd = jac(dd, d1_dd)
    # Compare Jacobian eigenvalue spectra
    eig_ik = np.sort(np.linalg.eigvalsh((J_ik + J_ik.T) / 2))[::-1]
    eig_dd = np.sort(np.linalg.eigvalsh((J_dd + J_dd.T) / 2))[::-1]
    # Most discriminative levels:
    summary = {
        'n_reps': int(n),
        'n_dim': int(nd),
        'static_max_KS_D': float(Ds_static.max()),
        'static_frac_p_lt_0p01': float((ps_static < 0.01).mean()),
        'd1_max_KS_D': float(Ds_d1.max()),
        'd1_frac_p_lt_0p01': float((ps_d1 < 0.01).mean()),
        'd2_max_KS_D': float(Ds_d2.max()),
        'd2_frac_p_lt_0p01': float((ps_d2 < 0.01).mean()),
        'jacobian_eig_l2_difference': float(np.linalg.norm(eig_ik - eig_dd)),
        'jacobian_top5_eig_ikaros': eig_ik[:5].tolist(),
        'jacobian_top5_eig_daedalus': eig_dd[:5].tolist(),
        'jacobian_frobenius_distance': float(np.linalg.norm(J_ik - J_dd)),
        'jacobian_max_entry_diff': float(np.abs(J_ik - J_dd).max()),
    }
    # Which derived family is most informative? Rank by max KS-D
    summary['ranking_by_max_KS_D'] = sorted(
        [('static', summary['static_max_KS_D']),
         ('d_rep',  summary['d1_max_KS_D']),
         ('d2_rep', summary['d2_max_KS_D'])],
        key=lambda x: -x[1])
    return summary

def main():
    out = {}
    out['ks_per_new_signal'] = ks_per_signal()
    out['extended_signature_loo'] = extend_signature_loo()
    out['dynamics_jacobian_phase13'] = dynamics_search_phase13()
    # Top-3 new signals ranking
    items = [(k, v.get('max_KS_D', 0.0))
             for k, v in out['ks_per_new_signal'].items()
             if v.get('status') != 'missing']
    items.sort(key=lambda x: -x[1])
    out['top3_new_signals'] = items[:3]
    out_path = os.path.join(EM19, 'analysis.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    print(f"\n[analyze] saved {out_path}")
    return out

if __name__ == '__main__':
    main()
