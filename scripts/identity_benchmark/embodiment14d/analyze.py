"""Phase 14D analysis — compare ikaros@<gov> vs daedalus@<existing 12B> for
each governor configuration. KS-D + Frobenius signal preservation.

Outputs:
  results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d/analysis.json
  results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d/verdict.txt
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                    'results', 'IDENTITY_BENCHMARK_2026-05-30'))
DIR_14D = os.path.join(ROOT, 'embodiment14d')
DIR_12B = os.path.join(ROOT, 'embodiment12b')


def load(path):
    with open(path) as f:
        return json.load(f)


def ks_from_summaries(s1, s2, n_eff=5000):
    """Approximate KS-D from percentile summaries.
    We don't have raw samples, so use a quantile-grid empirical CDF approximation:
    treat each summary as 9-point CDF (min, p50/p90/p99/p99.9, max) and compute
    the max gap between the two CDFs over a shared support."""
    pts1 = [(s1.get('min',0), 0.0), (s1['p50'], 0.5),
            (s1.get('p90', s1['p50']), 0.9), (s1.get('p99', s1['p50']), 0.99),
            (s1['p99_9'], 0.999), (s1.get('max', s1['p99_9']), 1.0)]
    pts2 = [(s2.get('min',0), 0.0), (s2['p50'], 0.5),
            (s2.get('p90', s2['p50']), 0.9), (s2.get('p99', s2['p50']), 0.99),
            (s2['p99_9'], 0.999), (s2.get('max', s2['p99_9']), 1.0)]
    pts1.sort(); pts2.sort()
    def cdf(pts, x):
        if x <= pts[0][0]: return pts[0][1]
        if x >= pts[-1][0]: return pts[-1][1]
        for i in range(len(pts)-1):
            if pts[i][0] <= x <= pts[i+1][0]:
                x0,y0 = pts[i]; x1,y1 = pts[i+1]
                if x1 == x0: return y0
                return y0 + (y1-y0)*(x-x0)/(x1-x0)
        return pts[-1][1]
    xs = sorted(set(p[0] for p in pts1+pts2))
    return max(abs(cdf(pts1,x) - cdf(pts2,x)) for x in xs)


def analyze_task_B():
    """Inter-core TSC offset KS-D per pair, mean and max across pairs."""
    dae = load(os.path.join(DIR_12B, 'task_B_daedalus.json'))
    out = {}
    for gov in ('powersave', 'performance'):
        path = os.path.join(DIR_14D, f'task_B_ikaros_{gov}.json')
        if not os.path.exists(path):
            out[gov] = {'error': 'missing', 'path': path}; continue
        ika = load(path)
        per_pair = {}
        for tgt in ('1','2','4','7','8','12','15'):
            if tgt in ika['pairs'] and tgt in dae['pairs']:
                ksd = ks_from_summaries(ika['pairs'][tgt], dae['pairs'][tgt])
                per_pair[tgt] = {'ksd': ksd,
                                 'ika_p50': ika['pairs'][tgt]['p50'],
                                 'dae_p50': dae['pairs'][tgt]['p50']}
        out[gov] = {
            'per_pair': per_pair,
            'mean_ksd': sum(v['ksd'] for v in per_pair.values())/max(1,len(per_pair)),
            'max_ksd': max(v['ksd'] for v in per_pair.values()) if per_pair else 0.0,
        }
    return out


def analyze_task_E():
    """Cacheline pingpong: Frobenius distance between p50 vectors across pairs."""
    dae = load(os.path.join(DIR_12B, 'task_E_daedalus.json'))
    out = {}
    pairs = ['0_1','0_2','0_4','0_7','0_8','0_15','0_16']
    dae_p50 = [dae['pairs'][p]['p50'] for p in pairs if p in dae['pairs']]
    for gov in ('powersave', 'performance'):
        path = os.path.join(DIR_14D, f'task_E_ikaros_{gov}.json')
        if not os.path.exists(path):
            out[gov] = {'error': 'missing', 'path': path}; continue
        ika = load(path)
        ika_p50 = [ika['pairs'][p]['p50'] for p in pairs if p in ika['pairs']]
        diff = [a-b for a,b in zip(ika_p50, dae_p50)]
        frob = math.sqrt(sum(d*d for d in diff))
        per_pair_ksd = {}
        for p in pairs:
            if p in ika['pairs'] and p in dae['pairs']:
                per_pair_ksd[p] = ks_from_summaries(ika['pairs'][p], dae['pairs'][p])
        out[gov] = {
            'ika_p50': ika_p50, 'dae_p50': dae_p50,
            'frobenius_p50': frob,
            'per_pair_ksd': per_pair_ksd,
            'mean_pair_ksd': sum(per_pair_ksd.values())/max(1,len(per_pair_ksd)),
        }
    return out


def analyze_task_A():
    """RDRAND p50 and nanosleep p99.9 — Phase 12's main signals."""
    out = {}
    dae_path = os.path.join(DIR_12B, 'task_A_daedalus.json')
    dae = load(dae_path) if os.path.exists(dae_path) else None
    for gov in ('powersave', 'performance'):
        path = os.path.join(DIR_14D, f'task_A_ikaros_{gov}.json')
        if not os.path.exists(path):
            out[gov] = {'error': 'missing'}; continue
        ika = load(path)
        row = {
            'rdrand_p50_ika': ika['rdrand']['p50'],
            'rdrand_p99_9_ika': ika['rdrand']['p99_9'],
            'nanosleep_p50_ika': ika['nanosleep']['p50'],
            'nanosleep_p99_9_ika': ika['nanosleep']['p99_9'],
        }
        if dae:
            # Phase 12B saves rdrand/nanosleep under top level for the host
            r = dae.get('rdrand', {}); n = dae.get('nanosleep', {})
            row.update({
                'rdrand_p50_dae': r.get('p50'),
                'rdrand_p99_9_dae': r.get('p99_9'),
                'nanosleep_p50_dae': n.get('p50'),
                'nanosleep_p99_9_dae': n.get('p99_9'),
                'rdrand_ks': ks_from_summaries(ika['rdrand'], r) if r else None,
                'nanosleep_ks': ks_from_summaries(ika['nanosleep'], n) if n else None,
            })
        out[gov] = row
    return out


def analyze_t2_t3():
    out = {}
    for gov in ('powersave', 'performance'):
        path = os.path.join(DIR_14D, f't2t3_{gov}.json')
        if not os.path.exists(path):
            out[gov] = {'error': 'missing'}; continue
        d = load(path)
        out[gov] = {
            'T2_vanilla': d['T2']['vanilla_auroc_mean'],
            'T2_embodied': d['T2']['embodied_auroc_mean'],
            'T2_prereg_pass': d['T2']['prereg_pass'],
            'T3_vanilla': d['T3']['vanilla_acc_mean'],
            'T3_embodied': d['T3']['embodied_acc_mean'],
            'T3_prereg_pass': d['T3']['prereg_pass'],
        }
    return out


def verdict_table(B, E, A, T):
    lines = []
    lines.append("="*78)
    lines.append("Phase 14D — Governor confound test verdict")
    lines.append("="*78)
    lines.append("")
    lines.append("NOTE: SSH to daedalus was DOWN during this run (port 22 refused).")
    lines.append("      Daedalus's Phase 12B/14B data was captured originally at its")
    lines.append("      default 'performance' governor. We re-measured ikaros at BOTH")
    lines.append("      powersave (original ikaros config) and performance (matched).")
    lines.append("      The comparison is: ikaros@<gov> vs frozen daedalus@perf data.")
    lines.append("")
    lines.append("--- Task B: inter-core TSC mean KS-D vs daedalus ---")
    for gov in ('powersave','performance'):
        v = B.get(gov, {})
        if 'error' in v: lines.append(f"  {gov:11s}: MISSING ({v['error']})"); continue
        lines.append(f"  {gov:11s}: mean KS-D = {v['mean_ksd']:.3f}  max = {v['max_ksd']:.3f}")
    lines.append("")
    lines.append("--- Task E: cacheline pingpong Frobenius p50 distance vs daedalus ---")
    for gov in ('powersave','performance'):
        v = E.get(gov, {})
        if 'error' in v: lines.append(f"  {gov:11s}: MISSING ({v['error']})"); continue
        lines.append(f"  {gov:11s}: Frobenius p50 = {v['frobenius_p50']:.1f} cyc  "
                     f"mean pair KS-D = {v['mean_pair_ksd']:.3f}")
    lines.append("")
    lines.append("--- Task A: rdrand + nanosleep absolute values ---")
    for gov in ('powersave','performance'):
        v = A.get(gov, {})
        if 'error' in v: lines.append(f"  {gov:11s}: MISSING"); continue
        lines.append(f"  {gov:11s}: rdrand_p50_ika={v['rdrand_p50_ika']}  "
                     f"nanosleep_p99_9_ika={v.get('nanosleep_p99_9_ika')}")
        if 'rdrand_p50_dae' in v:
            lines.append(f"  {' ':11s}  (daedalus@perf rdrand_p50={v['rdrand_p50_dae']}  "
                         f"ns_p99_9={v['nanosleep_p99_9_dae']})")
    lines.append("")
    lines.append("--- 14B T2/T3 with sigs collected at this governor ---")
    for gov in ('powersave','performance'):
        v = T.get(gov, {})
        if 'error' in v: lines.append(f"  {gov:11s}: MISSING"); continue
        lines.append(f"  {gov:11s}: T2 vanilla/embodied = {v['T2_vanilla']:.3f} / {v['T2_embodied']:.3f}  "
                     f"prereg={v['T2_prereg_pass']}")
        lines.append(f"  {' ':11s}  T3 vanilla/embodied = {v['T3_vanilla']:.3f} / {v['T3_embodied']:.3f}  "
                     f"prereg={v['T3_prereg_pass']}")
    lines.append("")
    # Decision logic
    lines.append("--- VERDICT ---")
    pB_ps = B.get('powersave',{}).get('mean_ksd')
    pB_pf = B.get('performance',{}).get('mean_ksd')
    pE_ps = E.get('powersave',{}).get('frobenius_p50')
    pE_pf = E.get('performance',{}).get('frobenius_p50')
    if pB_ps is not None and pB_pf is not None:
        delta_B = abs(pB_ps - pB_pf)
        lines.append(f"  Task B Δ(governor) mean KS-D = {delta_B:.3f}  "
                     f"(powersave={pB_ps:.3f}, performance={pB_pf:.3f})")
        if pB_pf >= 0.5:
            lines.append(f"    -> @ performance, mean KS-D vs daedalus STILL strong (≥0.5)")
            lines.append(f"       Signal survives matched-governor: SUPPORTS per-die claim.")
        else:
            lines.append(f"    -> @ performance, mean KS-D vs daedalus dropped below 0.5.")
            lines.append(f"       Signal partly collapses with matched governor: confound was real.")
    if pE_ps is not None and pE_pf is not None:
        lines.append(f"  Task E Δ(governor) Frobenius = {abs(pE_ps - pE_pf):.1f} cyc  "
                     f"(powersave={pE_ps:.1f}, performance={pE_pf:.1f})")
    # T2/T3
    t2_ps = T.get('powersave',{}).get('T2_embodied')
    t2_pf = T.get('performance',{}).get('T2_embodied')
    t3_ps = T.get('powersave',{}).get('T3_embodied')
    t3_pf = T.get('performance',{}).get('T3_embodied')
    if t2_pf is not None:
        lines.append(f"  T2 embodied AUROC @ performance = {t2_pf:.3f}  "
                     f"(@ powersave = {t2_ps:.3f})")
        if t2_pf >= 0.85: lines.append("    -> T2 SURVIVES matched-governor (still strong WIN).")
        else:            lines.append("    -> T2 DROPPED with matched-governor.")
    if t3_pf is not None:
        lines.append(f"  T3 embodied accuracy @ performance = {t3_pf:.3f}  "
                     f"(@ powersave = {t3_ps:.3f})")
        if t3_pf >= 0.95: lines.append("    -> T3 SURVIVES matched-governor (still strong WIN).")
        else:            lines.append("    -> T3 DROPPED with matched-governor.")
    lines.append("")
    lines.append("="*78)
    return "\n".join(lines)


def main():
    B = analyze_task_B()
    E = analyze_task_E()
    A = analyze_task_A()
    T = analyze_t2_t3()
    ana = {'task_B': B, 'task_E': E, 'task_A': A, 't2_t3': T}
    with open(os.path.join(DIR_14D, 'analysis.json'), 'w') as f:
        json.dump(ana, f, indent=2)
    verd = verdict_table(B, E, A, T)
    with open(os.path.join(DIR_14D, 'verdict.txt'), 'w') as f:
        f.write(verd)
    print(verd)


if __name__ == "__main__":
    main()
