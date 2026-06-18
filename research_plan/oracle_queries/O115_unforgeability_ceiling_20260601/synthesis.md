# O115 — Synthesis: Unforgeability ceiling of FabricCrypt v2

**Date:** 2026-06-01
**Oracles:** openai (gpt-5), gemini-2.5-pro, grok-4-latest, deepseek-reasoner
**Bundle:** `prompt.md`, `nonce_signature.py`, `verifier_spoof_v2.py`,
`ikaros_spoof_v2.json`, `threat_model_and_signals.md`

---

## TL;DR — the scheme as coded is broken at $0.

**Four out of four oracles independently found the same fatal break.**
The Phase 14C plan-consistency gate, which `verifier_spoof_v2.py` makes
the *sole* acceptance criterion (line 143: `'accept_rate':
float(plan_pass.mean())`), checks exactly one number derived from a
*public* function of the *public* nonce. An attacker with no chip, no
side channel, no library, and no model can compute that number and
forge a chip-present proof in microseconds with probability ≈ 1.0.

This is not a 20-bit weakness or a 40-bit weakness. It is **0 bits of
security against any adversary above a junior engineer**.

The seven gates passing in `ikaros_spoof_v2.json` are **meaningless**:
the attack battery only contains replay and mismatch attacks. It never
tested an attacker who wrote the gate value directly into a forged
vector.

---

## The fatal break (consensus across all four oracles)

**The attack** (GPT-5, Gemini, Grok, DeepSeek all reached this independently):

1. Receive the verifier's nonce `n`.
2. Compute `h = HMAC-SHA256(b"phase14c_nonce_sig", n)` — the key is
   public (line 117 of `nonce_signature.py`).
3. Extract `ns_sleep = 1000 + (h[16] | (h[17] << 8)) % 7000`
   (line 129).
4. Compute `expected = log1p(ns_sleep * 1e-3)`. This is the value
   `plan_consistency_score` checks (line 62 of `verifier_spoof_v2.py`).
5. Either:
   (a) **Position-aware variant** — derive `plan['perm']` (also from
   `h`), find `pos = np.where(perm == 31)[0][0]`, set `phys[pos] =
   expected` and zero (or arbitrary) values for the other 31 dims; or
   (b) **Position-blind variant** (GPT-5's observation) — set ALL 32
   phys dims to `expected`. The gate is then permutation-invariant.
6. Append `nonce_embedding(n, 32)` (also a public function of the
   public nonce — line 145 of `nonce_signature.py`).
7. Send the 64-dim forged vector. `plan_score = 1.0`, `plan_pass`
   triggers, `accept_rate = 1.0`.

**Why it works.** Dimension 31 of the un-permuted phys vector is set
to `plan['ns_sleep']` *directly from the plan* (line 232:
`out[31] = float(plan['ns_sleep'])`). This is not a measurement. It is
the input parameter to `nanosleep`, written into the output unchanged
and then log-scaled. The "gate" therefore checks that the prover knows
a deterministic public function of the public nonce. This conveys
**zero liveness** and **zero chip-identity**.

**Cost.** ~10 lines of Python. <$0. Microseconds per forgery.
**Success rate.** ≈ 1.0 per attempt. Works on every nonce.
**Detectability.** Zero — the forged vector looks indistinguishable
from a legitimate one at the gate.

---

## Secondary findings (independent of the primary break)

### S1. Permutation derivation is host-coupled (GPT-5 only — design bug)

`derive_plan` seeds an `np.random.default_rng` from `h[:8]`, then
sequentially calls `rng.choice(n_cpus, ...)`, `rng.choice(n_zones,
...)`, two more `rng.choice` calls, and *finally* `rng.permutation(32)`.
Because the earlier `choice()` calls consume a number of internal RNG
draws that depends on `n_cpus` and `n_zones`, **`perm32` is not
deterministic across hosts with different CPU/thermal-zone counts**.

A remote verifier cannot in general re-derive the prover's `perm`
unless it knows the prover's exact `n_cpus` and `n_zones`. This breaks
the protocol's central claim that "the verifier re-derives the plan
from the public nonce." Fix: derive each plan component from an
independent SHAKE256 stream with domain separation
(`"perm32" | "cpu_subset" | ...`).

### S2. Test-harness RNG is broken (GPT-5, Gemini, Grok, DeepSeek)

`verifier_spoof_v2.py:115`:
`rng = np.random.default_rng(int(time.time()) & 0xFFFFFFFF)`

A 32-bit time-derived seed is brute-forceable in seconds. Even if the
primary break were fixed, an attacker could enumerate all 2³² seeds,
predict all nonces the verifier will ever issue, and pre-compute
responses. Production verifiers MUST use `secrets.token_bytes` or
`os.urandom`.

### S3. No distance-bounding → relay is trivial (all four oracles)

The protocol has no RTT enforcement. LAN relay (<150 µs) easily fits
inside the 1–3 ms challenge window. Even a metro-area relay (<10 ms)
may bridge a slack verifier. Without enforced sub-100 µs RTT, FabricCrypt
cannot rule out a proxy attacker forwarding nonces to a real victim
chip and forwarding the response back. Distance-bounding is the *only*
defence against relay; the paper's §3 threat model implicitly assumes
it but the code does not implement it.

### S4. Hardware emulator (Grok, DeepSeek)

Once the primary break is fixed, the next attack tier is a software
emulator that runs the *same* measurement code on an FPGA/SoC with
DRAM controller emulation. Because every "physical" signal FabricCrypt
reads is ultimately a sysfs file, a perf counter, or a nanosleep
return value — all of which are *software-observable* — an emulator
that *also* runs the prover code can satisfy any check the verifier
can express in software. Cost: $8–12 k (Grok's estimate); $500 k–$2 M
for high-fidelity (DeepSeek's estimate). Defence: signals that cannot
be emulated without breaking firmware contracts (e.g., a signed kernel
module that measures actual instruction-retirement latency against the
nonce-derived `ns_sleep` parameter).

### S5. Generative-model adversary (all four)

10⁵–10⁶ (nonce, sig) pairs from the victim plus a conditional generator
(GAN/flow/score) on cloud GPUs ($5 k–$50 k training cost) produces
plausible vectors for unseen nonces. With only summary-stat checks,
this defeats even a "fixed" scheme. Defence: return raw micro-sample
time series (not summary stats), verify 20–40 cross-signal algebraic
constraints, multi-round protocol with tight per-round RTT.

### S6. Helper-data / calibration leakage (Grok)

`_cal/cal_<host>.json` (`mu`, `sigma`) is world-readable on the prover.
A cold-boot or co-tenant attacker who reads it learns the chip's
normalization parameters. Limited damage on its own (the
chip-identity is in the per-dim signal *distribution*, not the
moments) but it accelerates a generative attack. Defence: encrypt or
TPM-seal the calibration file.

---

## Defence package (consensus + ordering)

Minimum viable redesign — implementing fewer than all of these still
leaves the scheme broken.

**Tier 1 — kill the primary break (mandatory)**

1. **Delete the deterministic-public-canary gate.** Either remove
   `out[31] = float(plan['ns_sleep'])` entirely, or replace it with an
   *actual measurement* — e.g., the median observed `nanosleep` return
   latency under the nonce-specified `ns_sleep` request — and verify
   the *measurement* (with proper tolerance), not the input parameter.
2. **Make the classifier a HARD veto** (not "diagnostic only").
   Acceptance := `plan_pass AND (P_own > τ_cls)`. Current code
   (`spoof_v2.py:143`) explicitly OR-collapses to plan-only.
3. **Switch to a private (keyed) plan derivation.** Either
   (a) HMAC with a per-die fused secret never exposed to software, or
   (b) a VRF where only the verifier holds the secret key and proves
   the plan was generated honestly. This means an attacker cannot
   compute the expected gate value(s) without breaking the keyed
   primitive.

**Tier 2 — close the secondary holes**

4. **Decouple `perm32` from host-dependent RNG draws** (fix S1).
   Independent SHAKE256 streams per plan component with domain
   separation.
5. **Replace verifier nonce RNG** with `secrets.token_bytes(16)`
   (16 bytes ≥ 128 bits — the current 64-bit nonce is small enough
   that birthday-collision matters for long-lived deployments).
6. **Encrypt or TPM-seal the per-host `_cal/cal_*.json`** calibration
   file.

**Tier 3 — raise the bar against generative / emulator adversaries**

7. **Return raw micro-sample series**, not just summary stats. Verify
   20–40 cross-signal algebraic constraints (min ≤ mean ≤ max,
   `tsc.mean / ns.mean` ratio, monotonicity of percentiles, etc.).
8. **Multi-round protocol with tight RTT.** R ≥ 8 sub-challenges per
   session; <1 ms per sub-challenge on LAN. This forces the attacker
   to synthesize high-dimensional, cross-round-consistent telemetry
   in near-real-time. (This is also the only meaningful defence
   against relay attackers.)
9. **Distance-bounding.** Hardware-timestamped challenge release;
   physical-light-speed sanity check on RTT.

**Tier 4 — for production deployments**

10. **Fuzzy extractor over the full 290-dim signature** (Dodis-Reyzin-
    Smith code-offset). The current `plan_consistency_score` is a
    *single-dim* fuzzy check, not a fuzzy extractor — it derives no
    high-entropy key. A proper fuzzy extractor would derive a stable
    secret `S` from the entire phys vector and the protocol would
    prove knowledge of `S` (e.g., via a Schnorr-style sigma protocol).
    This is what every serious PUF paper does and what FabricCrypt
    currently does not.
11. **Mutual attestation** (AMD SEV-SNP / Intel TDX on the verifier
    side) so a malicious verifier cannot trivially accept anything.

---

## Bit-security ceiling

### As currently implemented

| Adversary class                                            | Bits |
|------------------------------------------------------------|------|
| Replay attacker, M ≤ 10⁵ (Q18a)                           | **0** |
| Library replay, M ≤ 2³⁰ (Q18b)                            | **0** |
| Generative-model adversary (Q18c)                         | **0** |
| Hardware emulator (Q18d)                                  | **0** |
| Q25 $10 k budget — P(forge single chip-present proof)     | **≈ 1.0** |
| Q26 $1 M budget                                            | **≈ 1.0** |
| Q27 Nation-state                                           | **≈ 1.0** |

The protocol does not give 5 bits of security. It gives zero. The
primary break needs zero dollars, zero physical access, zero
side-channels, and zero training data.

### With Tier 1 defences only (plan-canary removed, classifier hard veto)

| Adversary class                                            | Bits |
|------------------------------------------------------------|------|
| Replay attacker, M ≤ 10⁵                                   | ~15–20 |
| Library replay, M ≤ 2³⁰                                    | ~15–25 |
| Generative-model adversary (10⁶ pairs, GAN)                | ~10–20 |
| Hardware emulator                                          | ~5–15 |

Still broken at $50 k for a determined attacker.

### With Tier 1+2+3 defences (private plan, multi-round, raw series, RTT)

| Adversary class                                            | Bits |
|------------------------------------------------------------|------|
| Replay attacker, M ≤ 10⁵                                   | ~40–60 |
| Library replay, M ≤ 2³⁰                                    | ~30–50 |
| Generative-model adversary                                 | ~20–40 |
| Hardware emulator (remote, no local box)                   | ~30–45 |
| Hardware emulator (local box at victim site)               | **0** (relay wins) |

Workable for non-state remote attestation. Not adequate against an
attacker who can place hardware within physical proximity of the
victim — that is fundamentally what distance-bounding addresses, not
the crypto layer.

### Fundamental ceiling

No matter what defences we add at the protocol layer, a nation-state
attacker who can place an emulator or relay box physically adjacent to
the victim wins. The strongest claim FabricCrypt can ever make is
**"chip-presence proof against remote and co-tenant attackers, modulo
relay distance bounds"**, not "chip-identity proof against an arbitrary
attacker." The paper v2 threat model §3 already explicitly excludes
chip-present adversaries; the §5.5 "Adversary C" residual-risk
language for side-channel reconstruction is approximately correct but
should be sharpened to include hardware-emulator adversaries.

---

## What this means for paper v2

1. **Section 5.4 (plan-consistency gate) must be rewritten.** The
   current text presents the gate as a strong primitive (~63 effective
   plan-entropy bits). It is not: it checks one number deterministically
   derivable from the public nonce. Either:
   - Acknowledge the break and present this O115 result as the
     adversarial-audit finding that drives Phase 14D, or
   - Redesign per Tier 1 and re-run the spoof battery with a
     custom-forgery attack added.

2. **`ikaros_spoof_v2.json` should not be cited as evidence of
   security.** Its attack battery (replay variants, mismatch variants)
   is necessary but not sufficient. The missing test is "compute the
   expected gate value from the public nonce, fabricate a vector that
   passes the gate, observe `accept_rate`." Until that test is in the
   battery and `accept_rate ≤ 0.05`, no security claim is defensible.

3. **The classifier is currently load-bearing in narrative but
   load-free in code.** Make it a hard veto or remove it from the
   paper. Honest-own with `(plan_pass AND P_own > 0.5)` should still
   pass ≥ 0.95 based on the diagnostic numbers in
   `ikaros_spoof_v2.json` (`classifier_accept_only` for honest_own =
   0.904), so the cost of making it a hard veto is roughly 5 pp on
   honest-accept, which is recoverable with threshold tuning.

4. **The paper should explicitly enumerate adversary classes that the
   protocol does NOT defend against**: hardware emulator, local relay,
   generative-model adversary with > 10⁵ pairs. The current §5.5 has
   the right structure but the specific bit-security claims (~33 bits
   plan entropy, library replay resistant to M ≈ 2⁶⁰) are
   contradicted by the O115 audit and need to be retracted or
   conditioned on the Tier-1+2+3 redesign.

---

## Decision tree for next 24 h

- **If we keep claiming chip-presence proof**: implement Tier 1 (3
  changes, ~half a day of code), re-run spoof battery WITH a custom-
  forgery attack added, and replace the §5.4/§5.5 numbers in paper v2
  with the actual post-fix numbers.
- **If we cannot get Tier 1 done before submission**: downgrade the
  claim in paper v2 from "chip-presence proof" to "chip-identity
  *fingerprint* (vulnerable to forgery under software-emulator
  adversaries; not a cryptographic primitive)" and explicitly cite
  this O115 audit in §3 as the reason.
- **Either way**: the seven-gate spoof result in §5 stays in the paper
  as a *partial* result, with a new bullet noting that the
  custom-forgery attack class was not in the battery and is the next
  attack to defend against.

---

## Files in this oracle bundle

- `prompt.md` — the question pack sent to the oracles
- `nonce_signature.py` — Phase 14C prover (the attacked code)
- `verifier_spoof_v2.py` — Phase 14C verifier (the attacked code)
- `ikaros_spoof_v2.json` — the attack-battery output that, in hindsight,
  did not include the custom-forgery attack
- `threat_model_and_signals.md` — paper v2 §3 + §4.1 + §5.5 extract
- `openai_response.md`, `gemini_response.md`, `grok_response.md`,
  `deepseek_response.md` — raw oracle responses
- `synthesis.md` — this document
