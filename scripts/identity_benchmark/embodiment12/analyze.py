"""Analyze tasks D, E, F: KS test for inter-chassi vs intra-chassi distinguishability.

Intra-chassi baseline = split-half of same-machine samples (random shuffle).
Inter-chassi = ikaros vs daedalus.

Pre-reg gates:
- D, E, F: KS p-value < 0.001 inter-chassi, while intra-chassi p > 0.001.
"""
import json
import numpy as np
from scipy import stats
import os
import sys

BASE = "results/IDENTITY_BENCHMARK_2026-05-30/embodiment12"


def load(path):
    return json.load(open(path))


def ks_split_half(samples, rng_seed=0):
    s = np.array(samples)
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(s)
    half = len(s) // 2
    a, b = s[:half], s[half:2*half]
    stat, p = stats.ks_2samp(a, b)
    return stat, p


def ks_inter(s_a, s_b):
    a = np.array(s_a); b = np.array(s_b)
    stat, p = stats.ks_2samp(a, b)
    return stat, p


def analyze_task(name, ika_path, dae_path, fields):
    print(f"\n=== Task {name} ===")
    ika = load(ika_path)
    dae = load(dae_path)
    results = {'task': name, 'fields': {}}
    for field_key, sample_key, label in fields:
        ika_samp = ika[sample_key]
        dae_samp = dae[sample_key]
        ika_intra_stat, ika_intra_p = ks_split_half(ika_samp, rng_seed=1)
        dae_intra_stat, dae_intra_p = ks_split_half(dae_samp, rng_seed=2)
        inter_stat, inter_p = ks_inter(ika_samp, dae_samp)
        # summary stats
        ika_sum = ika[field_key]
        dae_sum = dae[field_key]
        passes = inter_p < 0.001 and max(ika_intra_p, dae_intra_p) > 0.001
        # additionally require inter much smaller than intra (effect size)
        effect_ratio = inter_stat / max(ika_intra_stat, dae_intra_stat, 1e-9)
        results['fields'][label] = {
            'ika_summary': ika_sum,
            'dae_summary': dae_sum,
            'intra_ikaros_KS_D': ika_intra_stat, 'intra_ikaros_p': ika_intra_p,
            'intra_daedalus_KS_D': dae_intra_stat, 'intra_daedalus_p': dae_intra_p,
            'inter_KS_D': inter_stat, 'inter_p': inter_p,
            'inter_intra_D_ratio': effect_ratio,
            'pre_reg_pass': bool(passes),
        }
        print(f"  {label}: intra_p ika={ika_intra_p:.3g} dae={dae_intra_p:.3g}  inter_p={inter_p:.3g}  D_ratio={effect_ratio:.2f}  PASS={passes}")
        print(f"    ika p50={ika_sum['p50']:.0f} p99={ika_sum['p99']:.0f} p99.9={ika_sum['p99_9']:.0f}")
        print(f"    dae p50={dae_sum['p50']:.0f} p99={dae_sum['p99']:.0f} p99.9={dae_sum['p99_9']:.0f}")
    return results


def main():
    out = {}
    out['D'] = analyze_task(
        'D_syscall_latency',
        f"{BASE}/task_D_syscall_ikaros.json",
        f"{BASE}/task_D_syscall_daedalus.json",
        [
            ('nanosleep0', 'raw_samples_ns_nanosleep0', 'nanosleep0'),
            ('sched_yield', 'raw_samples_ns_sched_yield', 'sched_yield'),
            ('getpid', 'raw_samples_ns_getpid', 'getpid'),
        ],
    )
    out['E'] = analyze_task(
        'E_rdrand',
        f"{BASE}/task_E_rdrand_ikaros.json",
        f"{BASE}/task_E_rdrand_daedalus.json",
        [('rdrand_cycles', 'raw_samples_cyc', 'rdrand_cycles')],
    )
    out['F'] = analyze_task(
        'F_nvme',
        f"{BASE}/task_F_nvme_ikaros.json",
        f"{BASE}/task_F_nvme_daedalus.json",
        [('nvme_latency', 'raw_samples_ns', 'nvme_latency_ns')],
    )

    # overall PASS summary
    passes = []
    for tk, td in out.items():
        for fname, fd in td['fields'].items():
            if fd['pre_reg_pass']:
                passes.append(f"{tk}:{fname}")
    print(f"\n=== SUMMARY: {len(passes)} pre-reg passes ===")
    for p in passes:
        print(f"  PASS {p}")

    with open(f"{BASE}/analysis.json", 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {BASE}/analysis.json")


if __name__ == '__main__':
    main()
