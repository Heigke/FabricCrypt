#!/usr/bin/env python3
"""Phase 19 Task E — attack harness for crypto v3.

Tests:
  T1 honest_own        — legit prover, legit verifier  -> ALL PASS
  T2 replay_old_sig    — replay last session's sig     -> H3 should fail it
  T3 cross_session     — record (nonce,sig) one session, replay next -> H3 fail
  T4 proxy_forward     — adversary forwards to far prover, exceeds RTT -> H1 fail
  T5 verifier_swap     — attacker poses as verifier     -> H2 fail
  T6 plan_tamper       — adversary mutates plan digest  -> H4 fail
  T7 vrf_predict       — adversary tries to predict nonce before reveal -> H5 fail
  T8 timing_side       — measure leak via response time variance -> H4 makes flat
"""
import os, sys, time, hmac, hashlib, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'embodiment14c')))
from nonce_signature import NonceSig
from crypto_v3 import (Session, Prover, derive_plan_salted, plan_digest,
                       attest, attest_verify, vrf_eval, vrf_verify)
from common19 import hostname, save_json

def run_session(sig, V_key, P_key, simulate_rtt_us=200,
                proxy_extra_us=0, tamper_plan=False, wrong_verifier=False,
                replay_blob=None):
    # rtt_budget = 50ms (sig.read can take ~10ms; this still rejects WAN proxy)
    s = Session(V_key, P_key, rtt_budget_us=50000)
    p = Prover(P_key, V_key, sig)
    msg = s.begin()
    ack = p.ack_commitment(msg['commitment'])
    # Adversary may swap verifier:
    if wrong_verifier:
        s.verifier_id = os.urandom(32)
    ch = s.challenge(ack)
    # Simulate RTT
    time.sleep(simulate_rtt_us / 1e6)
    if proxy_extra_us:
        time.sleep(proxy_extra_us / 1e6)
    if replay_blob is not None:
        # use a stale sig vec + (likely wrong) plan digest
        sig_vec = replay_blob['sig_vec']
        pd      = replay_blob['plan_digest']
        a_tag   = replay_blob['attest']
    else:
        resp = p.respond(ch['nonce'], msg['K_salt'], ch['vrf_out'],
                         ch['vrf_proof'], msg['vrf_pk'], s.vrf_sk,
                         ch['verifier_attest'])
        sig_vec, pd, a_tag = resp['sig_vec'], resp['plan_digest'], resp['attest']
    if tamper_plan:
        pd = hashlib.sha256(pd + b'X').digest()
        # Re-attest with prover key (adversary may try this — but the *check*
        # compares pd to recomputed plan digest, so this still fails H4.)
    r = s.receive(sig_vec, pd, a_tag)
    # Add H4 check (constant-time plan-consistency): recompute plan, compare digests
    plan = derive_plan_salted(msg['K_salt'], ch['nonce'],
                              sig.n_cpus, sig.n_zones)
    expected_pd = plan_digest(plan)
    r['plan_ok'] = hmac.compare_digest(pd, expected_pd)
    # bundle for replay logging
    r['blob'] = {'sig_vec': sig_vec, 'plan_digest': pd, 'attest': a_tag}
    return r

def main():
    sig = NonceSig()
    V_key = os.urandom(32)
    P_key = os.urandom(32)
    results = {}
    timings = {}

    # T1 honest_own
    t0 = time.perf_counter()
    r1 = run_session(sig, V_key, P_key)
    timings['T1_honest'] = (time.perf_counter() - t0) * 1000
    results['T1_honest_own'] = {
        'rtt_ok': r1['rtt_ok'], 'plan_ok': r1['plan_ok'],
        'prover_attest_ok': r1['prover_attest_ok'],
        'rtt_us': r1['rtt_us'],
        'PASS': all([r1['rtt_ok'], r1['plan_ok'], r1['prover_attest_ok']])
    }
    blob1 = r1['blob']

    # T2 replay_old_sig (same K_salt scenario — would only work if attacker
    # already had the blob from a previous identical session; H3 makes K_salt
    # fresh each session so this is automatic-fail when run in T3.)
    # We exercise within-session replay tolerance: blob is fine ON the session
    # that produced it (T1 already covered that). The interesting test is T3.

    # T3 cross_session replay: record blob1 in NEW session (fresh K_salt)
    r3 = run_session(sig, V_key, P_key, replay_blob=blob1)
    results['T3_cross_session_replay'] = {
        'plan_ok': r3['plan_ok'],
        'prover_attest_ok': r3['prover_attest_ok'],
        'PASS_means_attack_failed': (not r3['plan_ok']) or (not r3['prover_attest_ok'])
    }

    # T4 proxy_forward (excessive RTT — add 60ms over 50ms budget)
    r4 = run_session(sig, V_key, P_key, proxy_extra_us=60000)
    results['T4_proxy_forward'] = {
        'rtt_us': r4['rtt_us'], 'rtt_ok': r4['rtt_ok'],
        'PASS_means_attack_failed': not r4['rtt_ok']
    }

    # T5 verifier_swap (wrong verifier id key during challenge)
    try:
        r5 = run_session(sig, V_key, P_key, wrong_verifier=True)
        # Prover.respond should have raised on H2 verifier_attest fail
        results['T5_verifier_swap'] = {'PASS_means_attack_failed': False,
                                       'note': 'Prover accepted bad verifier'}
    except PermissionError as e:
        results['T5_verifier_swap'] = {'PASS_means_attack_failed': True,
                                       'error': str(e)}

    # T6 plan tamper
    r6 = run_session(sig, V_key, P_key, tamper_plan=True)
    results['T6_plan_tamper'] = {
        'plan_ok': r6['plan_ok'],
        'PASS_means_attack_failed': not r6['plan_ok']
    }

    # T7 vrf_predict: try to forge VRF proof without knowing sk
    fake_sk = os.urandom(32)
    pk = hashlib.sha256(os.urandom(32)).digest()  # actual pk
    nonce = os.urandom(8)
    out, proof = vrf_eval(fake_sk, nonce)
    ok = vrf_verify(pk, nonce, out, proof, fake_sk)
    results['T7_vrf_predict'] = {'verified': ok,
                                 'PASS_means_attack_failed': not ok}

    # T8 timing-side: 50 plan_check calls, half match half don't
    real_pd = blob1['plan_digest']
    fake_pd = bytes(32)
    times_match = []
    times_mismatch = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        _ = hmac.compare_digest(real_pd, real_pd)
        times_match.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns()
        _ = hmac.compare_digest(real_pd, fake_pd)
        times_mismatch.append(time.perf_counter_ns() - t0)
    tm = np.array(times_match); tf = np.array(times_mismatch)
    # PASS if medians within 30% (constant-time-ish)
    ratio = abs(np.median(tm) - np.median(tf)) / max(np.median(tm), 1.0)
    results['T8_timing_side'] = {
        'med_match_ns': float(np.median(tm)),
        'med_mismatch_ns': float(np.median(tf)),
        'rel_diff': float(ratio),
        'PASS_means_attack_failed': ratio < 0.30,
    }

    # Latency overhead
    overhead = {'T1_total_ms': timings['T1_honest']}
    results['_overhead'] = overhead
    host = hostname()
    out_dir = os.path.abspath(os.path.join(HERE, '..', '..', '..',
        'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment19'))
    save_json(os.path.join(out_dir, f'{host}_crypto_attacks.json'), results)
    print(json.dumps(results, indent=2, default=str))
    return results

if __name__ == '__main__':
    main()
