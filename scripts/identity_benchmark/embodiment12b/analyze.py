#!/usr/bin/env python3
"""Phase 12B analysis — KS-D between ikaros and daedalus per task.
For intra-comparison on Task A, also load Phase 12 prior result and compute
same-chassi-now vs same-chassi-30min-ago equivalence (we use Phase 12's stored
distributions as the prior).
"""
import os, json, math, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                       'results', 'IDENTITY_BENCHMARK_2026-05-30',
                                       'embodiment12b'))
PRIOR_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                         'results', 'IDENTITY_BENCHMARK_2026-05-30',
                                         'embodiment12'))

def load(host, task):
    p = os.path.join(OUT_DIR, f'task_{task}_{host}.json')
    if not os.path.exists(p): return None
    return json.load(open(p))

def ks_from_summary(s1, s2):
    """Approximate KS-D from percentile summaries (since we didn't keep raw samples).
    Uses 5 anchor percentiles and computes max |ECDF1 - ECDF2| at the union of anchor values."""
    anchors = []
    for s in (s1, s2):
        for p_name, p_val in [('p50',50),('p90',90),('p99',99),('p99_9',99.9),('p99_99',99.99)]:
            v = s.get(p_name)
            if v is not None and not isinstance(v, str):
                anchors.append((v, p_val))
    # add min/max
    for s in (s1, s2):
        if 'min' in s: anchors.append((s['min'], 0.0))
        if 'max' in s: anchors.append((s['max'], 100.0))
    # build per-summary ECDF lookup
    def ecdf_at(s, x):
        """Estimate ECDF(x) from p50/p90/p99/p99.9/p99.99/min/max."""
        anchors = sorted([(s['min'],0.0),(s['p50'],0.5),(s['p90'],0.9),
                          (s['p99'],0.99),(s['p99_9'],0.999)] +
                         ([(s['p99_99'],0.9999)] if s.get('p99_99') else []) +
                         [(s['max'],1.0)])
        if x <= anchors[0][0]: return 0.0
        if x >= anchors[-1][0]: return 1.0
        for i in range(len(anchors)-1):
            x0,p0 = anchors[i]; x1,p1 = anchors[i+1]
            if x0 <= x <= x1:
                if x1==x0: return p1
                return p0 + (p1-p0)*(x-x0)/(x1-x0)
        return 1.0
    points = sorted(set(a[0] for a in anchors))
    D = 0.0
    for x in points:
        d = abs(ecdf_at(s1, x) - ecdf_at(s2, x))
        if d > D: D = d
    return D

def ks_p_approx(D, n1, n2):
    """KS p-value approximation (Smirnov)."""
    if D <= 0: return 1.0
    en = math.sqrt(n1*n2 / (n1+n2))
    lam = (en + 0.12 + 0.11/en) * D
    # Q_KS(lam) = 2 * sum_{j=1..inf} (-1)^(j-1) exp(-2 j^2 lam^2)
    s = 0.0
    for j in range(1, 100):
        term = 2 * ((-1)**(j-1)) * math.exp(-2*j*j*lam*lam)
        s += term
        if abs(term) < 1e-12: break
    return max(0.0, min(1.0, s))

def compare_pair(s1, s2, label=''):
    if s1 is None or s2 is None: return None
    D = ks_from_summary(s1, s2)
    n1 = s1.get('n', 100000); n2 = s2.get('n', 100000)
    p = ks_p_approx(D, n1, n2)
    return {'KS_D': D, 'KS_p': p, 'n1': n1, 'n2': n2,
            'label': label, 's1_p50': s1.get('p50'), 's2_p50': s2.get('p50'),
            's1_p99_9': s1.get('p99_9'), 's2_p99_9': s2.get('p99_9')}

# ---- per-task analysis ----
results = {}

# Task A: replicate Phase 12. Compare ikaros@12B vs ikaros@12; same daedalus.
prior = json.load(open(os.path.join(PRIOR_DIR, 'analysis.json')))
A_ika = load('ikaros','A'); A_dae = load('daedalus','A')
ra = {'task':'A','feasible':True,'sub':{}}
if A_ika and A_dae:
    # field map: 12B rdrand -> 12 E.rdrand_cycles; 12B nanosleep -> 12 D.nanosleep0; 12B nvme_read -> 12 F.nvme_latency_ns
    field_map = [
        ('rdrand', prior['E']['fields']['rdrand_cycles']['ika_summary'],
                   prior['E']['fields']['rdrand_cycles']['dae_summary']),
        ('nanosleep', prior['D']['fields']['nanosleep0']['ika_summary'],
                      prior['D']['fields']['nanosleep0']['dae_summary']),
    ]
    # Note: 12B nvme_read uses file-read NOT O_DIRECT; documented mismatch with 12 nvme_latency.
    for fname, p_ika, p_dae in field_map:
        s_now_ika = A_ika.get(fname); s_now_dae = A_dae.get(fname)
        if not s_now_ika or not s_now_dae: continue
        # for prior summaries, rename keys to our schema
        def adapt(s):
            o = dict(s)
            if 'mean_ns' in o: o['mean']=o['mean_ns']
            if 'mean_cyc' in o: o['mean']=o['mean_cyc']
            return o
        p_ika_a = adapt(p_ika); p_dae_a = adapt(p_dae)
        intra_ika = compare_pair(s_now_ika, p_ika_a, 'ika_now_vs_phase12')
        intra_dae = compare_pair(s_now_dae, p_dae_a, 'dae_now_vs_phase12')
        inter_now = compare_pair(s_now_ika, s_now_dae, 'inter_12B')
        ra['sub'][fname] = {
            'intra_ikaros': intra_ika,
            'intra_daedalus': intra_dae,
            'inter_12B': inter_now,
            'persists': inter_now['KS_D'] > 0.1 and inter_now['KS_p'] < 0.01,
            'same_chassi_stable': intra_ika['KS_D'] < 0.3 and intra_dae['KS_D'] < 0.3,
        }
results['A'] = ra

# Task B: TSC inter-core
B_ika = load('ikaros','B'); B_dae = load('daedalus','B')
rb = {'task':'B','feasible':bool(B_ika and B_dae),'pairs':{}}
if B_ika and B_dae:
    for k in B_ika.get('pairs',{}):
        if k in B_dae.get('pairs',{}):
            rb['pairs'][k] = compare_pair(B_ika['pairs'][k], B_dae['pairs'][k], f'inter_core_0_{k}')
    # gate: at least one pair p<0.001
    rb['pre_reg_pass'] = any((v and v['KS_p']<0.001) for v in rb['pairs'].values())
results['B'] = rb

# Task C: AESENC
C_ika = load('ikaros','C'); C_dae = load('daedalus','C')
rc = {'task':'C','feasible':bool(C_ika and C_dae)}
if C_ika and C_dae:
    rc['aesenc'] = compare_pair(C_ika['aesenc'], C_dae['aesenc'], 'aesenc_inter')
    rc['pre_reg_pass'] = rc['aesenc']['KS_p'] < 0.001
results['C'] = rc

# Task D: atomic contention
D_ika = load('ikaros','D'); D_dae = load('daedalus','D')
rd = {'task':'D','feasible':bool(D_ika and D_dae),'pairs':{},'max_rel_diff':0.0}
if D_ika and D_dae:
    pairs_i = D_ika.get('pairs',{}); pairs_d = D_dae.get('pairs',{})
    for k in pairs_i:
        if k in pairs_d:
            ti = pairs_i[k]['throughput_per_s']; td = pairs_d[k]['throughput_per_s']
            rel = abs(ti-td)/max(ti,td)
            rd['pairs'][k] = {'ika_thp': ti, 'dae_thp': td, 'rel_diff': rel}
            if rel > rd['max_rel_diff']: rd['max_rel_diff'] = rel
    rd['pre_reg_pass'] = rd['max_rel_diff'] >= 0.10
results['D'] = rd

# Task E: cacheline pingpong
E_ika = load('ikaros','E'); E_dae = load('daedalus','E')
re_ = {'task':'E','feasible':bool(E_ika and E_dae),'pairs':{}}
if E_ika and E_dae:
    pairs_i = E_ika.get('pairs',{}); pairs_d = E_dae.get('pairs',{})
    # Frobenius-norm of p50 difference matrix
    keys = sorted(set(pairs_i) & set(pairs_d))
    fro = 0.0
    for k in keys:
        d = pairs_i[k]['p50'] - pairs_d[k]['p50']
        fro += d*d
        re_['pairs'][k] = compare_pair(pairs_i[k], pairs_d[k], f'pingpong_{k}')
    re_['frobenius_p50_diff'] = math.sqrt(fro)
    re_['pre_reg_pass'] = any((v and v['KS_p']<0.001) for v in re_['pairs'].values())
results['E'] = re_

# Task F: TPM
F_ika = load('ikaros','F'); F_dae = load('daedalus','F')
rf = {'task':'F'}
if F_ika and F_dae:
    rf['feasible_ika'] = F_ika.get('feasible',False)
    rf['feasible_dae'] = F_dae.get('feasible',False)
    if F_ika.get('feasible') and F_dae.get('feasible'):
        rf['tpm'] = compare_pair(F_ika.get('tpm_getrandom_ns'), F_dae.get('tpm_getrandom_ns'),'tpm')
        rf['pre_reg_pass'] = rf['tpm']['KS_p'] < 0.001
    else:
        rf['pre_reg_pass'] = None
        rf['reason'] = 'TPM requires root; not run'
results['F'] = rf

# Task G: DRAM refresh
G_ika = load('ikaros','G'); G_dae = load('daedalus','G')
rg = {'task':'G','feasible':bool(G_ika and G_dae)}
if G_ika and G_dae:
    rg['walk'] = compare_pair(G_ika.get('walk_ns'), G_dae.get('walk_ns'), 'walk')
    if 'spike_interval_samples' in G_ika and 'spike_interval_samples' in G_dae:
        rg['spike_intervals'] = compare_pair(G_ika['spike_interval_samples'], G_dae['spike_interval_samples'],'spikes')
    rg['ika_spike_count'] = G_ika.get('spike_count')
    rg['dae_spike_count'] = G_dae.get('spike_count')
    rg['pre_reg_pass'] = rg['walk']['KS_p'] < 0.001
results['G'] = rg

# Task H: PCIe config-space
H_ika = load('ikaros','H'); H_dae = load('daedalus','H')
rh = {'task':'H','feasible':bool(H_ika and H_dae)}
if H_ika and H_dae:
    # aggregate across devices: build one mock summary using p50-of-p50
    agg_i = H_ika.get('per_device',{}); agg_d = H_dae.get('per_device',{})
    common = sorted(set(agg_i) & set(agg_d))
    rh['n_common_devs'] = len(common)
    # build "summary" of p50 distribution across devices
    if common:
        p50_i = sorted(agg_i[k]['p50'] for k in common)
        p50_d = sorted(agg_d[k]['p50'] for k in common)
        # simple summary
        def quick(arr):
            n=len(arr)
            return {'n':n,'min':arr[0],'max':arr[-1],
                    'p50':arr[n//2],'p90':arr[min(n-1,int(n*0.9))],
                    'p99':arr[min(n-1,int(n*0.99))],'p99_9':arr[-1]}
        rh['p50_dist_ika'] = quick(p50_i)
        rh['p50_dist_dae'] = quick(p50_d)
        rh['p50_dist_KS'] = compare_pair(rh['p50_dist_ika'], rh['p50_dist_dae'], 'pcie_p50_dist')
        rh['pre_reg_pass'] = rh['p50_dist_KS']['KS_p'] < 0.01
results['H'] = rh

# ---- summary ----
feasible = []
passed = []
for t in 'ABCDEFGH':
    r = results.get(t, {})
    if t == 'A':
        # Task A pass criteria: at least one field both persists inter AND stable intra
        f = any(s.get('persists') and s.get('same_chassi_stable')
                for s in r.get('sub',{}).values())
        feas = bool(r.get('sub'))
    elif t == 'F':
        feas = bool(r.get('feasible_ika') and r.get('feasible_dae'))
        f = bool(r.get('pre_reg_pass'))
    else:
        feas = bool(r.get('feasible'))
        f = bool(r.get('pre_reg_pass'))
    if feas: feasible.append(t)
    if f: passed.append(t)
    print(f"Task {t}: feasible={feas} passed={f}")

print(f"\nFEASIBLE: {feasible}")
print(f"PASSED:   {passed}")
print(f"yes-survives-gate / feasible = {len(passed)}/{len(feasible)}")
print(f"Phase 12 had 3/3 (D,E,F). Combined HAL-bypass signals: {3+len(passed)}")

results['_summary'] = {
    'feasible_tasks': feasible,
    'passed_tasks': passed,
    'phase12_passed': ['D','E','F'],
    'combined_hal_bypass_count': 3 + len(passed),
    'task_A_per_chassi_stable': any(s.get('same_chassi_stable') and s.get('persists')
                                     for s in results.get('A',{}).get('sub',{}).values()),
}

with open(os.path.join(OUT_DIR, 'analysis.json'),'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"[save] {os.path.join(OUT_DIR,'analysis.json')}")
