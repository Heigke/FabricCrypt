#!/usr/bin/env python3
"""Phase 20 analysis: compute intra-host stability + inter-host KS-D.

For each signal sN:
  - load all <host>_sN.npz under embodiment20/
  - intra-host: split each host's reps half/half, compute KS-D per feature
                between halves, mean → INTRA (lower = more stable)
  - inter-host: KS-D per feature between distinct hosts, mean → INTER
  - acceptance: INTER >= 0.5 AND INTER > INTRA + 0.15
"""
import os, sys, glob, json
import numpy as np
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, '..', '..', '..',
    'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment20'))


def load_signal(sig):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, f'*_{sig}.npz'))):
        d = np.load(f, allow_pickle=True)
        host = str(d['host'])
        vec = d['vec']
        if vec.size == 0: continue
        out[host] = vec
    return out


def per_feature_ks(a, b):
    """Mean KS-D across feature dims."""
    if a.size == 0 or b.size == 0: return float('nan')
    dim = min(a.shape[1], b.shape[1])
    ds = []
    for j in range(dim):
        x, y = a[:, j], b[:, j]
        if np.allclose(x.std(), 0) and np.allclose(y.std(), 0):
            ds.append(0.0 if np.allclose(x.mean(), y.mean()) else 1.0)
            continue
        try:
            ds.append(float(ks_2samp(x, y).statistic))
        except Exception:
            ds.append(float('nan'))
    return float(np.nanmean(ds))


def main():
    signals = ['s10', 's11', 's12', 's13', 's14']
    report = {'phase': 20, 'signals': {}}
    for sig in signals:
        data = load_signal(sig)
        intra_d = {}
        for host, v in data.items():
            if v.shape[0] < 4:
                intra_d[host] = None; continue
            half = v.shape[0] // 2
            intra_d[host] = per_feature_ks(v[:half], v[half:])
        hosts = sorted(data.keys())
        inter = {}
        for i in range(len(hosts)):
            for j in range(i+1, len(hosts)):
                k = f'{hosts[i]}_vs_{hosts[j]}'
                inter[k] = per_feature_ks(data[hosts[i]], data[hosts[j]])
        valid_intra = [x for x in intra_d.values() if x is not None]
        valid_inter = list(inter.values())
        intra_mean = float(np.mean(valid_intra)) if valid_intra else None
        inter_mean = float(np.mean(valid_inter)) if valid_inter else None
        accept = (inter_mean is not None and intra_mean is not None and
                  inter_mean >= 0.5 and inter_mean > intra_mean + 0.15)
        report['signals'][sig] = {
            'hosts': hosts,
            'intra_per_host': intra_d,
            'inter_pairs': inter,
            'intra_mean': intra_mean,
            'inter_mean': inter_mean,
            'accept_for_signature_v2': bool(accept),
        }
    out = os.path.join(RES, 'PHASE20_ANALYSIS.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
