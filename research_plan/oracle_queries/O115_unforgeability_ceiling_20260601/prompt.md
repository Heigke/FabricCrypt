# O115 — Adversarial unforgeability ceiling for FabricCrypt

**Date:** 2026-06-01
**Stance requested:** maximally adversarial. Assume nation-state attacker
with full source code, full protocol spec, $10k → $1M → unlimited budget.
Your job is to *break* this scheme, then tell us which defences survive.
Do not be diplomatic. Concrete attacks > generic concerns.

---

## What FabricCrypt is (one-paragraph recap)

A challenge-response chip-presence proof on a commodity AMD APU. The
verifier (audience, possibly remote) sends a fresh 64-bit nonce. The
prover runs a HAL-bypass measurement code that derives a sampling plan
deterministically from `HMAC-SHA256("phase14c_nonce_sig", nonce)`. The
plan dictates *what* low-level physical signals to read (which CPUs'
c-state usage, which thermal zones, which core pairs for c2c pingpong,
which nanosleep / TSC bursts, and a 32-element output permutation).
The 32-dim physics vector is concatenated with a 32-dim deterministic
embedding of the nonce, giving a 64-dim signature. The verifier runs:

1. **Plan-consistency gate** (hard, deterministic): dimension 31 of the
   un-permuted phys vector must equal `log1p(plan.ns_sleep * 1e-3)`
   within `0.15` in log-space. The verifier re-derives the plan from
   the public nonce and inverts the permutation to locate dim 31.
2. **Twin classifier** (soft, MLP `TwinMLP`): outputs P(own) on the
   64-dim signature; used as a diagnostic, NOT a veto.

Acceptance := plan-pass. Observed (`ikaros_spoof_v2.json`):

| Attack                            | accept rate | gate     |
|-----------------------------------|-------------|----------|
| honest_own                        | 1.000       | ≥ 0.95 ✓ |
| daedalus_peer (real foreign chip) | 0.020       | ≤ 0.05 ✓ |
| static_replay_no_nonce            | 0.006       | ≤ 0.05 ✓ |
| static_replay_with_correct_nonce  | 1.000       | ≥ 0.95 ✓ |
| dynamic_replay (M=400 library)    | 0.012       | ≤ 0.10 ✓ |
| nonce_only_mismatch               | 0.006       | ≤ 0.05 ✓ |
| honest_own_wrong_nonce            | 0.006       | ≤ 0.05 ✓ |

7/7 gates pass. Now: how does this scheme actually die?

---

## Files in this bundle (read them; ground your answers in actual code/numbers)

- `nonce_signature.py` — Phase 14C prover: nonce-keyed measurement + plan derivation (324 LOC)
- `verifier_spoof_v2.py` — Phase 14C verifier (plan-consistency score + attack battery)
- `ikaros_spoof_v2.json` — Full attack results (7 attacks, 500 reps each)
- `threat_model_and_signals.md` — Paper v2 §3 threat model + §4.1 five signals + §5.5 adversary analysis

---

## Questions (BE ADVERSARIAL — assume nation-state attacker)

### Cryptographic primitive choices
1. We use HMAC-SHA256(nonce) → sampling-plan. Sufficient? Should we use:
   - SHAKE256 with domain separation,
   - Encrypted credential commitment,
   - Verifiable Random Function (VRF) for the nonce-plan mapping?
2. Plan-consistency verifier uses **0.15 log-space tolerance** — what's
   the bit-security of that against guessing attacks? (Note: the
   *signal* being matched is `ns_sleep ∈ {1000..8000}` ns under a
   deterministic log1p transform; the leak is structural.)
3. Should the plan be public or hidden? Currently public (verifier
   re-derives from nonce). Trade-offs?

### Adversary models we may not have considered
4. **Hardware emulator.** Attacker builds a chip-emulator that takes
   any nonce and outputs whatever measurements we expect. What stops
   them? (We claim physics, but the prover is software — an FPGA + DRAM
   emulator that learns the response model is *also* software.)
5. **Side-channel learning attack.** Attacker collects 10⁶
   challenge-response pairs from a victim chip, trains a neural model
   to predict response from challenge. Defence? Where does it break?
6. **Nonce-prediction.** If verifier's nonce-generation is weak,
   attacker pre-computes. We use OS `urandom`. Sufficient?
7. **Distance-bounding / relay.** Attacker proxies between victim and
   verifier. Defence via RTT? We currently have none — explicitly
   discuss whether a relay attack against a 1–3 ms challenge window is
   feasible over (a) LAN, (b) public Internet.
8. **Acoustic side-channel.** Attacker microphone near victim records
   coil whine during challenge, infers ns_sleep / TSC burst structure?
9. **Power-analysis.** VRM ripple during challenge leaks the signal a
   nation-state could rebuild offline.
10. **Compromised verifier.** If the verifier is malicious, can it
    accept forged signatures? Defence via mutual attestation?
11. **Clone-with-aged-twin.** Attacker has identical hardware aged
    similarly. Do signals collide? Defence via "fresh" measurements
    per session?
12. **Cold-boot / DRAM-dump on training data.** Attacker dumps DRAM
    during `retrain_embodied_nonce`, recovers (nonce, sig) pairs,
    classifier weights, and the mu/sigma calibration file. Defence?

### Fuzzy extractor / error correction
13. Standard PUF papers use fuzzy extractors (Dodis-Reyzin-Smith 2008)
    for noise-tolerant key derivation. Should we? Concretely, is our
    `plan_consistency_score` doing something fuzzy-extractor-shaped,
    or is it a weaker construct?
14. Code-offset vs syndrome construction — which fits a 290-dim
    signature where ~30% of dims are high-noise (thermal/power) and
    ~70% are low-noise (TSC, syscall p99.9)?
15. Helper-data security: how much information leaks from public
    `mu, sigma` calibration files committed by the prover?

### Provable security
16. Random-oracle-model proof — possible? What would the reduction
    look like?
17. UC-secure? In what setting (chip-presence ideal functionality
    F_PRESENCE)?
18. Bit-security estimate against:
    (a) replay attacker with M ≤ 10⁵ (nonce, sig) pairs,
    (b) library-replay with M ≤ 2³⁰,
    (c) generative-model adversary (learned predictor),
    (d) hardware-emulator adversary.

### zkML composition
19. Combine FabricCrypt with zkML (zk-SNARK of model inference) so the
    chip-signature is part of the SNARK witness? Sketch the circuit.
20. How does this compare to Modulus Labs / Lagrange / RISC Zero zk-VM
    approaches that prove *which model* ran but cannot prove *on which
    chip*?

### Implementation pitfalls
21. Side-channels in **our own** sampling code that leak the nonce or
    the plan to a co-tenant attacker (CSTATE_DIRS reads, RAPL reads,
    `sched_setaffinity` cross-core spam).
22. Timing-attack on `plan_consistency_score` — is the `abs(observed -
    expected)` comparison constant-time? Does it matter at this layer?
23. RNG quality in nonce generation (verifier-side). We use
    `np.random.default_rng(int(time.time()) & 0xFFFFFFFF)` in the test
    harness. That's broken — real verifier should use `secrets.token_bytes`.
    Confirm.
24. Spurious correlations between signals our verifier didn't account
    for (e.g., thermal-zone temperatures correlate with RAPL energy,
    so an attacker who knows one can predict the other).

### Bayesian unforgeability ceiling
25. Given the current design, give a calibrated probability that a
    determined adversary with $10k budget forges a single chip-present
    proof on a single challenge. Show your reasoning, not just a number.
26. Same for $1M budget.
27. Same for nation-state budget (unbounded $, but constrained to
    *not physically destroy* the victim chip).

---

## Output format

Respond as four sections:

### 1. Worst attack you found
The single most damaging concrete attack on the scheme as specified.
Walk through it step by step. Include cost estimate and rough probability
of success per attempt.

### 2. Other attacks ranked
A numbered list, severity-ordered. For each: (a) one-sentence summary,
(b) what defence (if any) would survive it, (c) cost/feasibility.

### 3. Defences to ADD
Concrete changes — code-level if possible — that would close the
attacks you found.

### 4. Bit-security estimate
For each adversary class in Q18 and Q25–27, your final estimate.
If the answer is "this scheme is broken at $X cost," say so plainly.

Cite concrete line numbers / values from the bundled files where applicable.
Do not write a literature review. Do not summarise what FabricCrypt is —
we wrote that paper.
