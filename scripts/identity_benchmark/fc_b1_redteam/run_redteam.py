"""FC-B1 Red-team harness — runs the 5 TOP-ranked oracle-suggested attacks
against verifier_v2.py and reports accept rates vs the §5.9 baselines.

TOP-5 (aggregated across openai/gemini/grok/deepseek):
  A1  Enrolment poisoning (sigma inflation) — 4-oracle consensus
  B   Statistical forgery N(mu, sigma) under K_chip leak  — gemini
  C   Multi-round constant-S commit degeneracy             — openai
  D   Nonce-embedding spoof (verifier missing check)       — openai
  F   Reverse-FE bit-flip oracle (information leak only)   — 4-oracle consensus

Plus diagnostics:
  L   Plan-entropy collapse measurement (spec vs impl drift) — openai

Outputs results/FABRICCRYPT/b1_redteam/attack_results.json
"""
from __future__ import annotations
import os, sys, json, hashlib, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts/identity_benchmark/embodiment14d_crypto'))
sys.path.insert(0, os.path.join(ROOT, 'scripts/identity_benchmark/embodiment22b_crypto'))

import torch
from nonce_signature_v2 import (derive_plan_keyed, nonce_embedding, NonceSigV2)
from verifier_v2 import (hard_veto_accept, plan_measurement_score,
                          classifier_p0, calibrate_full_band)
from attacks_extended import TwinMLP, perm_inv_all
from key_derivation import load_kchip

RES = os.path.join(ROOT, 'results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d_crypto')
OUT = os.path.join(ROOT, 'results/FABRICCRYPT/b1_redteam')
os.makedirs(OUT, exist_ok=True)
DIM = 64

def load_baseline():
    d = np.load(os.path.join(RES, 'ikaros_paired_sigs.npz'))
    nonces = [bytes(n) for n in d['nonces']]
    sigs = d['sigs'].astype(np.float32)
    K_chip = load_kchip('ikaros')
    sig_obj = NonceSigV2(host='ikaros', K_chip=K_chip, calibrate=False)
    # Build mu, sigma exactly as attacks_extended does
    inv_all = perm_inv_all(nonces, K_chip, sig_obj.n_cpus, sig_obj.n_zones)
    mu_vec, sigma_vec = calibrate_full_band(sigs, inv_all)
    sigma_vec = np.maximum(sigma_vec, 0.5).astype(np.float32)
    # Load trained model (uninformative — best_auroc ~0.53 — but we honor verifier)
    model = TwinMLP(in_d=DIM, n_out=2)
    state = torch.load(os.path.join(RES, 'ikaros_t3_best.pt'), map_location='cpu')
    model.load_state_dict(state); model.eval()
    return dict(nonces=nonces, sigs=sigs, K_chip=K_chip,
                n_cpus=sig_obj.n_cpus, n_zones=sig_obj.n_zones,
                mu=mu_vec, sigma=sigma_vec, model=model, sig_obj=sig_obj)


def eval_attack(ctx, X, nonces, K_chip=None, mu=None, sigma=None,
                tau_cls=-1.0, band_k=3.0, label=""):
    K_chip = K_chip if K_chip is not None else ctx['K_chip']
    mu = mu if mu is not None else ctx['mu']
    sigma = sigma if sigma is not None else ctx['sigma']
    res = hard_veto_accept(ctx['model'], X.astype(np.float32), nonces, K_chip,
                            ctx['n_cpus'], ctx['n_zones'],
                            mu, sigma, tau_cls=tau_cls,
                            plan_score_thresh=0.5, band_k=band_k, device='cpu')
    res['label'] = label
    return res


def attack_A1_enrolment_poisoning(ctx, n_eval=100, seed=42):
    """Attacker writes a *poisoned* calibration file with sigma inflated
    10x. Verifier loads it. Any same-host signature now passes Mahalanobis
    trivially.

    We simulate this by passing inflated sigma to hard_veto_accept.
    Forgery vector: vector of zeros (or all_dim_flood). With unpoisoned
    sigma this is REJECTED. With poisoned sigma it SHOULD ACCEPT.
    """
    rng = np.random.default_rng(seed)
    # Attack vector: a static replay of one honest sig — already blocked
    # under correct sigma. We show poisoning lifts it.
    base_sig = ctx['sigs'][0, :32].copy()
    X = np.zeros((n_eval, DIM), dtype=np.float32)
    nonces = []
    for i in range(n_eval):
        nb = rng.bytes(8); nonces.append(nb)
        X[i, :32] = base_sig
        X[i, 32:] = nonce_embedding(nb, 32)
    # 1) unpoisoned baseline
    res_base = eval_attack(ctx, X, nonces, label='A1_unpoisoned_replay')
    # 2) poisoned: sigma * 10
    sigma_poisoned = ctx['sigma'] * 10.0
    res_poi = eval_attack(ctx, X, nonces, sigma=sigma_poisoned,
                          label='A1_poisoned_sigma10x')
    # 3) extreme: also shift mu to attacker's chosen point + huge sigma
    sigma_extreme = ctx['sigma'] * 100.0
    mu_attacker = base_sig.copy()  # mu = the attacker's chosen forge vec
    # need un-permuted mu — but for this test we'll use the permuted-then-
    # mahalanobis which doesn't actually invert (mu is treated as the
    # un-permuted ref). Use the chip's mu but with extreme sigma.
    res_ext = eval_attack(ctx, X, nonces, sigma=sigma_extreme,
                          label='A1_poisoned_sigma100x')
    return [res_base, res_poi, res_ext]


def attack_B_statistical_forgery(ctx, n_eval=100, seed=42):
    """Attacker has K_chip and chip's (mu, sigma). Generates synthetic
    unperm_fake ~ N(mu, sigma) per-dim, applies plan permutation, sends.

    Verifier with HONEST (mu,sigma) and HONEST K_chip will evaluate. If
    plan_pass + tau_cls=-1 (classifier disabled), this should accept ~ 1.0
    since z-scores are O(1) by construction."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_eval, DIM), dtype=np.float32)
    nonces = []
    for i in range(n_eval):
        nb = rng.bytes(8); nonces.append(nb)
        plan = derive_plan_keyed(nb, ctx['K_chip'], ctx['n_cpus'], ctx['n_zones'])
        perm = plan['perm']
        unperm_fake = rng.normal(ctx['mu'], ctx['sigma']).astype(np.float32)
        X[i, :32] = unperm_fake[perm]
        X[i, 32:] = nonce_embedding(nb, 32)
    return eval_attack(ctx, X, nonces, label='B_statistical_forgery')


def attack_C_multiround_constant_S(ctx):
    """Test multi-round constraint degeneracy with S=zeros."""
    from multiround import MultiRoundVerifier, commit_samples, evaluate_constraint
    V = MultiRoundVerifier(n_samples=50)
    results = []
    for trial in range(20):
        nonce = os.urandom(8)
        # Attacker commits zeros
        S = np.zeros(50, dtype=np.float32)
        r1 = dict(commit_S=commit_samples(nonce, S), n_samples=50)
        try:
            V.round1_recv(nonce, r1)
            cons = V.round2_send(nonce)
            t_vals = [evaluate_constraint(S, c) for c in cons]
            res = V.round3_verify(nonce, dict(S=S, t=t_vals))
            accepted = bool(res.get('accepted', False))
        except Exception as e:
            accepted = False
            res = {'err': str(e)[:100]}
        results.append({'trial': trial, 'accepted': accepted, 'res': str(res)[:300]})
    accept_rate = sum(r['accepted'] for r in results) / len(results)
    return {'label': 'C_multiround_constant_S',
            'accept_rate': accept_rate,
            'n_trials': len(results),
            'sample_results': results[:3]}


def attack_D_nonce_emb_spoof(ctx, n_eval=100, seed=42):
    """Verifier_v2 hard_veto_accept does NOT verify that X[i,32:] equals
    nonce_embedding(nonce). If we sent random embeddings — does it still
    accept (on honest phys)?"""
    rng = np.random.default_rng(seed)
    n_eval = min(n_eval, len(ctx['sigs']))
    X = ctx['sigs'][:n_eval].copy().astype(np.float32)
    nonces = ctx['nonces'][:n_eval]
    # Replace last 32 dims with random vectors (attacker-chosen)
    for i in range(n_eval):
        e = rng.normal(0, 2, 32).astype(np.float32)
        X[i, 32:] = e
    # Honest phys + spoofed emb. With tau_cls=-1 (classifier disabled),
    # plan_pass is computed only on the first 32 dims, so this should
    # ACCEPT. With tau_cls active, classifier might reject — but our
    # AUROC is at chance so tau_cls is disabled.
    res = eval_attack(ctx, X, list(nonces), label='D_nonce_emb_spoof_classifier_disabled')
    # Also test with tau_cls forced to 0.5 to see classifier effect:
    res2 = eval_attack(ctx, X, list(nonces), tau_cls=0.5,
                       label='D_nonce_emb_spoof_classifier_active_tau0.5')
    return [res, res2]


def attack_F_rfe_oracle_simulation(n_trials=300, seed=42):
    """Simulate a binary-feedback bit-flip oracle against the Reverse-FE.
    We use the standalone RFE module with a known w_ref; then we run the
    bit-flip attack and measure how many w_ref bits we can recover.

    NOTE: this is the *information-theoretic* limit of the attack — it
    does not depend on the verifier_v2 hard veto. If the attacker can
    learn each bit of w_ref via an oracle, w_ref is the chip-stable
    quantization → K-equivalent. Real-world rate-limiting would mitigate.
    """
    from reverse_fuzzy import ReverseFuzzyExtractor
    rng = np.random.default_rng(seed)
    t = 16  # decoder budget
    rfe = ReverseFuzzyExtractor(t=t, m=8)
    n_bits = rfe.N_BITS
    # Enroll with random w_ref
    w_ref = rng.integers(0, 2, n_bits).astype(np.uint8)
    rfe.enroll(w_ref)  # stores helper P, K
    # Oracle: given w_noisy, returns True/False (verify)
    def oracle(w):
        try:
            r = rfe.verify(w)
            return bool(r[0]) if isinstance(r, tuple) else bool(r.get('accepted', False))
        except Exception:
            return False
    # Attack: start from all zeros, flip each bit, observe accept change
    # Baseline: w_ref itself accepts (sanity)
    sanity = oracle(w_ref)
    # Hill-climb: start from one valid accept (w_ref + small noise)
    w_attack = w_ref.copy()
    # flip up to t/2 random bits to start from a noisy-but-acceptable point
    flip_idx = rng.choice(n_bits, t // 4, replace=False)
    w_attack[flip_idx] ^= 1
    start_accept = oracle(w_attack)
    # Now blind bit-flip attack — recover w_ref by toggling each bit
    queries = 0
    recovered = np.zeros(n_bits, dtype=np.uint8)
    # Use the gemini-described approach: flip bit i, observe if accept survives
    # If accept survives with bit flipped, that bit was 'flexible';
    # if accept dies, that bit was critical → matches w_ref[i] in the
    # opposite direction.
    # In practice, with t=16 we can flip up to 16 bits freely; bit-by-bit
    # info leak is bounded by t. We measure information leaked.
    # Simpler: gradient-free reconstruction by trying both 0 and 1 from
    # an unknown base (we DON'T know w_ref).
    w_guess = rng.integers(0, 2, n_bits).astype(np.uint8)  # blind start
    initial_dist = int((w_guess != w_ref).sum())
    # Hill-climb: for each bit, try flipping; if accept rate improves, keep.
    # But oracle returns 0/1 only and we're far from acceptance.
    accepted_count = 0
    for it in range(n_trials):
        idx = rng.integers(0, n_bits)
        w_try = w_guess.copy(); w_try[idx] ^= 1
        queries += 1
        if oracle(w_try):
            w_guess = w_try
            accepted_count += 1
            if (w_guess == w_ref).all():
                break
    final_dist = int((w_guess != w_ref).sum())
    return {
        'label': 'F_rfe_oracle_bitflip',
        'n_bits': n_bits, 't': t,
        'sanity_accept_w_ref': sanity,
        'start_noisy_accept': start_accept,
        'queries': queries,
        'initial_hamming_dist': initial_dist,
        'final_hamming_dist': final_dist,
        'oracle_accepts_during_hill_climb': accepted_count,
        'note': ('Bit-flip oracle from blind start needs ~O(2^t) effective queries '
                 'to find first accept; t=16 budget is too tight for pure blind '
                 'reconstruction. Practical attack requires a known starting accept '
                 '(one captured honest response). Information leak per accepted '
                 'flip is ~1 bit. Mitigation: rate-limit + per-session w_ref '
                 're-randomisation.'),
    }


def diagnostic_L_plan_entropy(ctx, n_draws=10000):
    """Measure realised plan entropy via collision count."""
    seen = set()
    K_chip = ctx['K_chip']
    rng = np.random.default_rng(0)
    for _ in range(n_draws):
        n = rng.bytes(8)
        plan = derive_plan_keyed(n, K_chip, ctx['n_cpus'], ctx['n_zones'])
        fp = hashlib.sha256(repr({k: list(v) if hasattr(v,'__iter__') and not isinstance(v,(int,float)) else v
                                   for k, v in plan.items()}).encode()).hexdigest()
        seen.add(fp)
    distinct = len(seen)
    # Theoretical max in §5.3: C(16,8) * C(zones,8) * C(120,16) * C(32,4) * ...
    # Realised: from derive_plan_keyed code = C(n_cpus, min(4, n_cpus)) ~ C(16,4)=1820
    # * choose 3 zones * 2 core_pairs * ns_sleep(7001) * ns_count(7) * tsc_count(7) * perm(32!)
    # perm(32!) dominates → distinct ≈ n_draws for ≥10^4 draws.
    return {'label': 'L_plan_entropy_distinct_plans',
            'n_draws': n_draws, 'distinct_plans': distinct,
            'collision_rate': 1.0 - distinct / n_draws,
            'note': ('Permutation of 32 dominates entropy (log2(32!)≈117 bits) → '
                     'distinct ≈ n_draws expected; §5.3 16-of-120 pair entropy '
                     'CLAIM is NOT implemented in derive_plan_keyed which uses '
                     'only 2 pairs. This is a spec/impl drift but the perm32 '
                     'masks it.'),
            }


def main():
    t0 = time.time()
    print('[FC-B1] Loading baseline ...')
    ctx = load_baseline()
    print(f'  K_chip[:8]={ctx["K_chip"][:8].hex()}  n_cpus={ctx["n_cpus"]}  n_zones={ctx["n_zones"]}')

    out = {'host': 'ikaros', 'started_at': t0, 'attacks': {}}

    print('\n[FC-B1] Attack A1 — enrolment poisoning ...')
    out['attacks']['A1_enrolment_poisoning'] = attack_A1_enrolment_poisoning(ctx)

    print('[FC-B1] Attack B — statistical forgery (K_chip leak + (mu,sigma) leak) ...')
    out['attacks']['B_statistical_forgery'] = attack_B_statistical_forgery(ctx)

    print('[FC-B1] Attack C — multi-round constant-S commit ...')
    out['attacks']['C_multiround_constant_S'] = attack_C_multiround_constant_S(ctx)

    print('[FC-B1] Attack D — nonce-embedding spoof ...')
    out['attacks']['D_nonce_emb_spoof'] = attack_D_nonce_emb_spoof(ctx)

    print('[FC-B1] Attack F — Reverse-FE bit-flip oracle ...')
    out['attacks']['F_rfe_oracle'] = attack_F_rfe_oracle_simulation(n_trials=2000)

    print('[FC-B1] Diagnostic L — plan-entropy collapse measurement ...')
    out['attacks']['L_plan_entropy'] = diagnostic_L_plan_entropy(ctx, n_draws=5000)

    out['elapsed_s'] = time.time() - t0
    json_path = os.path.join(OUT, 'attack_results.json')

    def _ser(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, bytes): return o.hex()
        return str(o)
    with open(json_path, 'w') as f:
        json.dump(out, f, default=_ser, indent=2)
    print(f'\n[FC-B1] Wrote {json_path}  ({out["elapsed_s"]:.1f}s)')

    # Summary
    print('\n=== ACCEPT-RATE SUMMARY ===')
    for k, v in out['attacks'].items():
        if isinstance(v, list):
            for r in v:
                print(f'  {r.get("label",k):<55s} accept={r.get("accept_rate","-")}  '
                      f'plan_pass={r.get("plan_pass_only","-")}  '
                      f'cls_pass={r.get("classifier_pass_only","-")}')
        elif isinstance(v, dict):
            print(f'  {v.get("label",k):<55s} accept={v.get("accept_rate","-")}  '
                  f'plan_pass={v.get("plan_pass_only","-")}  '
                  f'cls_pass={v.get("classifier_pass_only","-")}')


if __name__ == '__main__':
    main()
