# FC-B1 BLIND RED TEAM — FabricCrypt v3.1 (Phase 14C + Tier-2)

You are a security oracle. Your job is to **break** FabricCrypt, the
protocol described in the attached files. We claim "10/10 attacks
blocked" in §5.9 of `fabriccrypt_v3_1.md`, but we designed both the
attacks AND the verifier. We need YOU — an external party — to propose
**new attacks not covered by our existing list**.

---

## CONTEXT (full pack attached)

1. `fabriccrypt_v3_1.md` — §§4–5 protocol spec. Pay attention to:
   - §5.0 Tier-1 vs Tier-2 protocol evolution.
   - §5.2 plan derivation via `SHAKE256(K_chip || domain || nonce)`.
   - §5.4 VERIFY algorithm: plan-consistency + classifier-Mahalanobis
     HARD veto.
   - §5.8 the O115 fatal break (Tier-0→Tier-1).
   - §5.10 Tier-2: Reverse Fuzzy Extractor, Controlled-PUF, multi-round,
     ZK binding.
   - §5.10.5 V0–V7 adversary vectors and their claimed mitigations.

2. `fabriccrypt_v3_1_EDIT_LOG.md` — full audit log of v3 → v3.1 changes.

3. Existing 10-attack battery (already blocked, see §5.9 table):
   `honest_own`, `daedalus_peer`, `static_replay`,
   `correct_nonce_replay`, `dynamic_replay (M=200)`, `nonce_mismatch`,
   `honest_wrong_nonce`, `custom_forgery_o115`, `all_dim_flood`,
   `stolen_kchip_analysis`.

4. Existing adversary vectors (V0–V7) in §5.10.5: brute-force K_chip,
   fingerprint brute-force, ML modeling (controlled), generative,
   K_chip leak + multi-round, helper-data leakage, **relay V6**,
   chosen-challenge ML.

5. Verifier source: `verifier_v2.py` (HARD veto = plan_pass AND
   classifier_p0 > tau_cls). Plan score = min(s31, Mahalanobis over all
   32 un-permuted dims).

6. `hostB_trace_sanitised.md` — 5 sample (nonce, phys, nonce_emb)
   records from daedalus. K_chip not shared (SHA-256 hash provided
   for audit only).

7. `controlled_puf.py`, `reverse_fuzzy.py`, `multiround.py`, `zkml.py`
   — Tier-2 implementations.

---

## ATTACKER CAPABILITIES (worst case in scope)

- Full read access to all FabricCrypt source code (paper + scripts).
- Knows protocol structure, hash functions, ECC parameters, and the
  exact verifier algorithm.
- Has its own AMD Strix Halo chassis to develop/test on.
- Can collect public (challenge, response) traces and a few sanitised
  example records (see hostB_trace_sanitised.md).
- May, in some sub-scenarios, also have stolen K_chip (Tier-2 break
  scenario).

OUT OF SCOPE — already acknowledged unmitigated:
- V6 LAN relay (out of scope per §5.10.5).
- Hardware-level invasive attacks (decapsulation, microprobing).

---

## YOUR TASK

Propose **5–10 NEW attack concepts** against FabricCrypt v3.1 that are
**not equivalent to** any of the 10 attacks in §5.9 or the V0–V7
vectors in §5.10.5. We want a mix of:

- **Naive but missed**: attacks that should have occurred to us but
  didn't (e.g. side-channel on K_chip generation, time-of-check vs
  time-of-use, enrolment poisoning).
- **Creative / non-obvious**: attacks that exploit the *interaction*
  between Tier-1 plan-consistency and Tier-2 Controlled-PUF wrap, or
  the classifier's training pipeline, or the helper-data flow in
  Reverse-FE.

For each proposed attack, return:

1. **Name** (short, descriptive).
2. **Category** (replay / clone / side-channel / protocol / classifier
   / ML / supply-chain / other).
3. **Threat model** (what does the attacker have/know/can-do? Match
   one of the capabilities above or argue for an additional capability).
4. **Mechanism** (1–3 paragraphs explaining how the attack works,
   referencing specific section numbers in the protocol).
5. **Implementation sketch** (5–20 lines of pseudocode or pseudo-Python
   targeting `verifier_v2.py` and the Tier-2 modules — concrete enough
   that a competent engineer could code it in <2 hours).
6. **Expected outcome** (does it pass the HARD veto? does it
   partial-pass? if it fails, why is it still worth flagging?).
7. **Why it's NOT in our 10/10 list** (be specific — point out which
   list entry it might be confused with and explain the difference).

## RANKING

After listing all proposed attacks, give a **novelty + plausibility
ranking** of your TOP-5. Format:

```
RANK | NAME | NOVELTY (1-5) | PLAUSIBILITY (1-5) | EST EFFORT (hrs) | EST P(BYPASS)
```

We will implement the TOP-5 (aggregated across all oracles) as
proof-of-concept Python scripts and run them against `verifier_v2.py`.
If any of your attacks succeed, we will credit them in v3.2.

## CONSTRAINTS

- Do NOT recycle V0–V7 verbatim, even with cosmetic relabelling.
- Do NOT propose attacks that require hardware decapsulation.
- Be **brutally honest**: if you can't think of a plausible new attack,
  say so for that slot (return fewer than 5 rather than padding).
- Where a proposed attack overlaps partially with an existing entry,
  say "PARTIAL OVERLAP with attack-X" and explain the marginal novelty.

Return one Markdown document. No preamble — start with `# FC-B1
RED-TEAM RESPONSE`.
