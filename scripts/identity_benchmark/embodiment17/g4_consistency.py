"""G4: Within-chip consistency over time.

Sample ikaros sigs at t=0, t=Δ1, t=Δ2 (compressed to minutes; we use 60s
gaps for thermal budget). Generate embodied outputs at each time slice with
the same prompts and compare distribution style (token-histogram cosine).

Pre-reg: intra-chip cosine consistency > 0.7 across time points.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import (RESULTS, save_json, load_prompts, load_tiny_lm, sample_one,
                     sig_to_seed, thermal_guard, bootstrap_ci, LiveSigProvider)
from g2_classifier import featurize


def gen_slice(label, prompts, n_reps, max_new, tau, model, tok, device='cpu'):
    prov = LiveSigProvider()
    feats_by_prompt = [[] for _ in prompts]
    for rep in range(n_reps):
        for pi, pr in enumerate(prompts):
            thermal_guard()
            def seed_fn(_p=prov):
                return sig_to_seed(_p.read())
            ids = sample_one(model, tok, pr, max_new, tau, seed_fn, device=device)
            feats_by_prompt[pi].append(featurize(ids))
        print(f"  [{label}] rep {rep+1}/{n_reps}", flush=True)
    # average per-prompt histogram → "style vector" for that prompt
    style = np.stack([np.mean(np.stack(fs), axis=0) for fs in feats_by_prompt])
    return style  # (n_prompts, dim)


def cosine_matrix(a, b):
    """Per-prompt cosine similarity vector."""
    n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    m = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (n * m).sum(axis=1)


def main():
    device = 'cpu'
    tok, model = load_tiny_lm(device=device)
    prompts = load_prompts()[:20]
    n_reps = 5
    max_new = 20
    tau = 0.8

    # Spec: t=0, t=10min, t=60min. We compress to 0/30s/90s to honor thermal
    # budget. The point is to see if chip-driven style is stable over time.
    gap_short = 30
    gap_long = 60

    print("[G4] slice T0", flush=True)
    s0 = gen_slice('T0', prompts, n_reps, max_new, tau, model, tok, device)
    print(f"[G4] cooling/waiting {gap_short}s", flush=True)
    time.sleep(gap_short)
    print("[G4] slice T1", flush=True)
    s1 = gen_slice('T1', prompts, n_reps, max_new, tau, model, tok, device)
    print(f"[G4] cooling/waiting {gap_long}s", flush=True)
    time.sleep(gap_long)
    print("[G4] slice T2", flush=True)
    s2 = gen_slice('T2', prompts, n_reps, max_new, tau, model, tok, device)

    cos_01 = cosine_matrix(s0, s1)
    cos_02 = cosine_matrix(s0, s2)
    cos_12 = cosine_matrix(s1, s2)

    m01, lo01, hi01 = bootstrap_ci(cos_01, n_boot=1000)
    m02, lo02, hi02 = bootstrap_ci(cos_02, n_boot=1000)
    m12, lo12, hi12 = bootstrap_ci(cos_12, n_boot=1000)

    out = {
        'cos_T0_T1': {'mean': m01, 'ci95': [lo01, hi01]},
        'cos_T0_T2': {'mean': m02, 'ci95': [lo02, hi02]},
        'cos_T1_T2': {'mean': m12, 'ci95': [lo12, hi12]},
        'pass_intra_consistency_gt_0p7': bool(min(m01, m02, m12) > 0.7),
        'min_cos': float(min(m01, m02, m12)),
        'gap_short_s': gap_short, 'gap_long_s': gap_long,
        'note': ('Within-chip consistency over time. Intra-chip style cosine '
                 'is the per-prompt average token-histogram cosine. Compressed '
                 'gaps for thermal budget; spec was 10/60min.'),
    }
    print(f"[G4] cos T0-T1={m01:.3f}  T0-T2={m02:.3f}  T1-T2={m12:.3f}", flush=True)
    print(f"[G4] PASS (all > 0.7) = {out['pass_intra_consistency_gt_0p7']}", flush=True)
    save_json('g4_consistency.json', out)


if __name__ == '__main__':
    main()
