#!/usr/bin/env python3
"""Phase 13 — synthesize all task outputs into a single report JSON."""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
    'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment13'))

def load(fn):
    p = os.path.join(OUT_DIR, fn)
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def task_A_repro():
    ika = np.load(os.path.join(OUT_DIR, 'ikaros_sig_v2.npz'))['vec']
    dae = np.load(os.path.join(OUT_DIR, 'daedalus_sig_v2.npz'))['vec']
    def repro(arr):
        mu = arr.mean(axis=0); sd = arr.std(axis=0) + 1e-12
        cv = sd / (np.abs(mu) + 1e-12)
        return {'n_reps': int(arr.shape[0]), 'dim': int(arr.shape[1]),
                'mean_cv': float(np.median(cv)),
                'p95_cv':  float(np.percentile(cv, 95)),
                'within_dist_p95': float(np.percentile(
                    [1 - (arr[i]@arr[j])/(np.linalg.norm(arr[i])*np.linalg.norm(arr[j])+1e-12)
                     for i in range(len(arr)) for j in range(len(arr)) if i<j], 95))}
    # cross-host distance
    nrm_i = ika / (np.linalg.norm(ika, axis=1, keepdims=True)+1e-12)
    nrm_d = dae / (np.linalg.norm(dae, axis=1, keepdims=True)+1e-12)
    cross = 1.0 - nrm_i @ nrm_d.T
    return {'ikaros': repro(ika), 'daedalus': repro(dae),
            'cross_host_dist_p50': float(np.percentile(cross, 50)),
            'cross_host_dist_p5':  float(np.percentile(cross, 5))}

def main():
    report = {
        'taskA_signature_v2': task_A_repro(),
        'taskB_constitutive': load('constitutive_v2.json'),
        'taskC_chassi_lock':  load('chassi_lock.json'),
        'taskD_drift':        load('drift_D.json'),
        'taskE_classifier':   load('classifier_E.json'),
    }
    # combined verdict
    gates = {}
    if report['taskB_constitutive']:
        c = report['taskB_constitutive']['summary']['contrasts']
        gates['B_swap_gate'] = c.get('gate_15pct_swap_AC_passed', False)
        gates['B_AB_gate']   = c.get('gate_15pct_AB_passed', False)
    if report['taskC_chassi_lock']:
        s = report['taskC_chassi_lock']['summary']
        gates['C_own>0.90']        = s['gate_own_gt_0_90']
        gates['C_transplant<0.30'] = s['gate_transplant_lt_0_30']
        gates['C_spoof>0.80']      = s['gate_spoof_gt_0_80']
    if report['taskD_drift']:
        gates['D_drift_p95<0.05'] = report['taskD_drift'].get('gate_p95_lt_0_05_passed', False)
    if report['taskE_classifier']:
        gates['E_loo>0.95']       = report['taskE_classifier']['gate_gt_0_95_passed']
    report['gates'] = gates
    report['n_gates_passed'] = sum(1 for v in gates.values() if v)
    report['n_gates_total']  = len(gates)
    with open(os.path.join(OUT_DIR, 'phase13_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report['gates'], indent=2))
    print(f"\n[phase13] gates passed: {report['n_gates_passed']}/{report['n_gates_total']}")

if __name__ == '__main__':
    main()
