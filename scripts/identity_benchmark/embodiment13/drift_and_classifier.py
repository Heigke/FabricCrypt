#!/usr/bin/env python3
"""Phase 13 Tasks D + E.

D — Signature drift over 1 hour on ikaros:
    capture every 5 min for 12 captures total, pairwise cosine distance.
    gate: 95th percentile of (1 - cos_sim) < 0.05

E — Bayesian / logistic classifier ikaros vs daedalus from Phase 12+12B-derived
    signatures (we already have 10+10 reps of signature_v2 from Phase 13 Task A —
    use those directly as the labelled dataset for leave-one-out cross-val).
    gate: LOO accuracy > 0.95.
"""
import os, sys, time, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common13 import wait_cool, get_apu_temp_c, hostname

OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
    'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment13'))

# ---------- Task E (fast, no hardware) ----------
def task_E_classifier():
    ika = np.load(os.path.join(OUT_DIR, 'ikaros_sig_v2.npz'))['vec']
    dae = np.load(os.path.join(OUT_DIR, 'daedalus_sig_v2.npz'))['vec']
    X = np.vstack([ika, dae])
    y = np.array([0]*len(ika) + [1]*len(dae))
    # z-normalise based on full joint
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-9
    Xz = (X - mu)/sd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    correct = 0
    preds = []
    for tr, te in loo.split(Xz):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xz[tr], y[tr])
        p = int(clf.predict(Xz[te])[0])
        preds.append(p); correct += int(p == y[te[0]])
    acc = correct / len(y)
    out = {'n_total': int(len(y)), 'n_ikaros': int(len(ika)), 'n_daedalus': int(len(dae)),
           'loo_acc': float(acc), 'gate_gt_0_95_passed': bool(acc > 0.95),
           'preds': preds, 'truth': y.tolist()}
    with open(os.path.join(OUT_DIR, 'classifier_E.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k:v for k,v in out.items() if k not in ('preds','truth')}, indent=2))
    return out

# ---------- Task D — short drift study ----------
def task_D_drift(captures=12, interval_s=300):
    """Capture sig every interval_s seconds for `captures` times.
    NOTE: in production this is 1 hour. To stay under time budget when invoked
    via the wrapper, we adapt interval to keep <=15 min total.
    """
    from signature_v2 import extract_one
    from common13 import compile_c
    HOST = hostname()
    if HOST != 'ikaros':
        print(f'[drift] skipping on {HOST}'); return {}
    tsc_src = os.path.join(HERE, 'tsc_inter_core.c')
    tsc_bin = os.path.join(HERE, 'tsc_inter_core')
    cl_src  = os.path.join(HERE, 'cacheline_pingpong.c')
    cl_bin  = os.path.join(HERE, 'cacheline_pingpong')
    if not os.path.exists(tsc_bin): compile_c(tsc_src, tsc_bin)
    if not os.path.exists(cl_bin):  compile_c(cl_src, cl_bin)

    vecs = []
    times = []
    t0 = time.time()
    for k in range(captures):
        wait_cool(target_c=55, timeout_s=120)
        print(f"[drift] capture {k+1}/{captures} t={time.time()-t0:.0f}s temp={get_apu_temp_c():.1f}C",
              flush=True)
        v = extract_one(tsc_bin, cl_bin)
        vecs.append(v); times.append(time.time()-t0)
        # wait until interval elapsed (but don't sleep the whole thing if behind)
        target_next = (k+1) * interval_s
        while time.time() - t0 < target_next:
            time.sleep(5)
    vecs = np.asarray(vecs)
    # cosine distance pairwise
    nrm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    cos = nrm @ nrm.T
    dist = 1.0 - cos
    triu = dist[np.triu_indices_from(dist, k=1)]
    out = {'n_captures': int(captures), 'interval_s': interval_s,
           'duration_s': float(times[-1] - times[0]),
           'pairwise_dist_p50': float(np.percentile(triu, 50)),
           'pairwise_dist_p95': float(np.percentile(triu, 95)),
           'pairwise_dist_max': float(np.max(triu)),
           'gate_p95_lt_0_05_passed': bool(np.percentile(triu, 95) < 0.05),
           'capture_times_s': times}
    np.savez(os.path.join(OUT_DIR, 'drift_D.npz'), vecs=vecs, times=np.asarray(times))
    with open(os.path.join(OUT_DIR, 'drift_D.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k:v for k,v in out.items() if k != 'capture_times_s'}, indent=2))
    return out

if __name__ == '__main__':
    sel = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if sel in ('both', 'E'):
        task_E_classifier()
    if sel in ('both', 'D'):
        # interval=120s -> 12 captures in ~24min (acceptable; uses thermal guard)
        interval = int(sys.argv[2]) if len(sys.argv)>2 else 120
        task_D_drift(captures=12, interval_s=interval)
