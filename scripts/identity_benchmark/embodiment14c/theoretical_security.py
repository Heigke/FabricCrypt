"""Phase 14C Task E — theoretical-security note.

Run with: python theoretical_security.py [results_json]
It will read the spoof_v2.json and emit a single text file summarising
the attack model and *why* the nonce-keyed design defeats each attack.
"""
from __future__ import annotations
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))

TEMPLATE = """\
PHASE 14C — THEORETICAL UNFORGEABILITY ANALYSIS
================================================
Host: {host}    Generated: {ts}

1. THREAT MODEL
---------------
Adversary capabilities (worst case considered):
  - Has full read access to the public protocol (nonce embedding, plan derivation).
  - Has recorded up to M=2^k pairs (nonce, sig) from the honest chip in the past
    (k limited by physical access time × throughput; M ~ 10^5 plausible).
  - Cannot physically access the honest chip during the challenge.
  - Knows the trained classifier.
  - Picks any (nonce-embedding, phys-feature) pair to submit.
Adversary GOAL: get the classifier to output P(own) > 0.5 on a fresh,
audience-supplied 64-bit nonce.

2. WHY STATIC REPLAY FAILS (Phase 14B kill)
-------------------------------------------
In Phase 14B the nonce only permutes OUTPUT positions; the underlying 32-dim
physical reading is the same regardless of nonce. The classifier therefore
learnt a permutation-invariant view of the phys feats, and a single recorded
own-sig replayed forever was accepted (observed accept rate = 1.00).

In Phase 14C the nonce drives the sampling plan: which CPUs, which thermal
zones, which core pairs, which nanosleep durations, how many TSC samples.
The marginal distribution of phys feats *under nonce N* differs from the
marginal *under nonce N'* — even on the same chip — because different physical
mechanisms produce them. A static replay therefore presents (phys|N) when the
challenge demands (phys|N'); these distributions are distinguishable.

Expected accept-rate: ≤ 5%   |   Observed (this run): {static_replay_no_nonce}

3. WHY DYNAMIC REPLAY (library) IS COSTLY
-----------------------------------------
The audience picks a fresh 64-bit nonce. The adversary's library has M
recorded pairs; for any challenge nonce N*, the closest library nonce N has
expected Hamming distance ≈ 32 - log2(M)/2 bits (birthday-like for sub-space
matches). With M=10^5 the closest library entry differs in ~30 of 64 bits,
which propagates into a very different sampling plan (different CPUs/zones/
core pairs), and therefore a different phys distribution.

To win consistently the adversary needs library coverage of fraction
f ~= accept_rate target, requiring M ≈ 2^64 × f^(1/d) where d is the
effective feature dimensionality. For accept ≥ 50% this is ~ 2^63 entries —
infeasible. For accept ≥ 10% it is still ~ 2^60.

Expected accept-rate: ≤ 10%  |   Observed (this run): {dynamic_replay}

4. WHY NONCE-ONLY MISMATCH FAILS
--------------------------------
If the embedded nonce in the input differs from the nonce actually used to
read the chip, the (phys, emb) joint is off-manifold: the classifier sees a
combination that never occurs in training. By construction the embedded nonce
is also a substantial fraction (32/64 dims) of the classifier's input, so the
classifier can detect the mismatch even without phys-side anomaly.

Expected accept-rate: ≤ 5%   |   Observed (this run): {nonce_only_mismatch}

5. FAILURE MODES STILL POSSIBLE
-------------------------------
A. Chip-present adversary: if the adversary briefly holds the physical chip
   they win. Out of scope (this is a chip-presence proof, not an access-
   control system).
B. Side-channel leakage at trained-weights level: the classifier learns
   features not strictly tied to the nonce. Mitigation: the nonce-derived
   sampling plan is *causally entangled* with phys feats (e.g. the
   nanosleep-jitter mean depends on `ns_sleep` which is nonce-derived).
   This binds phys → nonce so that classifier features cannot be nonce-
   invariant.
C. Adversary that can do live measurements on a TWIN chip and submits those
   under fresh nonces: this is the daedalus_peer attack and is handled by
   the cross-chip distinguishability (phase 14B already showed 2% accept on
   real foreign chip; 14C inherits that property).

6. UNFORGEABILITY GRADE (this run)
----------------------------------
  honest_own:                {honest_own}    gate ≥ 0.95
  daedalus_peer:             {daedalus_peer}    gate ≤ 0.05
  static_replay_no_nonce:    {static_replay_no_nonce}    gate ≤ 0.05  (was 1.00 in 14B)
  dynamic_replay:            {dynamic_replay}    gate ≤ 0.10
  nonce_only_mismatch:       {nonce_only_mismatch}    gate ≤ 0.05

Overall: {overall}
"""


def main():
    spoof_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = os.path.abspath(os.path.join(
        HERE, '..', '..', '..', 'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment14c'))
    if spoof_path is None:
        # auto-find host
        try:
            host = open('/etc/hostname').read().strip()
        except Exception:
            host = 'ikaros'
        spoof_path = os.path.join(out_dir, f'{host}_spoof_v2.json')
    if not os.path.exists(spoof_path):
        print(f"[theoretical] missing {spoof_path}", file=sys.stderr)
        sys.exit(2)
    d = json.load(open(spoof_path))
    a = d['attacks']
    gates = d.get('gates', {})
    def fmt(k):
        v = a.get(k, {})
        if 'skipped' in v: return 'SKIPPED'
        return f"{v.get('accept_rate', float('nan')):.3f}"
    overall = 'PASS' if all(g.get('pass') is True for g in gates.values() if g.get('pass') is not None) else \
              'FAIL on: ' + ','.join(k for k,g in gates.items() if g.get('pass') is False)
    txt = TEMPLATE.format(
        host=d.get('host','?'),
        ts=time.strftime('%Y-%m-%d %H:%M:%S'),
        honest_own=fmt('honest_own'),
        daedalus_peer=fmt('daedalus_peer'),
        static_replay_no_nonce=fmt('static_replay_no_nonce'),
        dynamic_replay=fmt('dynamic_replay'),
        nonce_only_mismatch=fmt('nonce_only_mismatch'),
        overall=overall,
    )
    out_txt = os.path.join(out_dir, f"{d.get('host','x')}_theoretical_security.txt")
    open(out_txt, 'w').write(txt)
    print(txt)
    print(f"\n[theoretical] wrote {out_txt}")


if __name__ == '__main__':
    main()
