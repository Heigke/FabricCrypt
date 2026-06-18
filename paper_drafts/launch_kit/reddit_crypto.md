# r/cryptography post

**Subreddit:** r/cryptography (NOT r/crypto — that's banned-anything-coin)
**Drop window:** T+72h (Friday).
**Angle:** protocol design — nonce-driven sampling plan, bit-security
accounting, honest residual unforgeability.

---

## Title

```
FabricCrypt: nonce-driven sampling-plan attestation on commodity AMD — protocol design + bit-security accounting
```

---

## Body

Looking for protocol-level critique on a per-die attestation primitive
we just open-sourced. Honest bit-security analysis below; this is
where I most want this community's eyes.

---

**Setting.** Two parties: Prover P (an AMD Ryzen AI 300 laptop) and
Verifier V (anyone with P's enrolled fingerprint). V wants to verify
that a given AI output was produced on a specific *physical die*, not
just on *some chip of the same SKU class*. No vendor key. No TEE.

**Primitive.** P holds a per-die secret K_chip extracted from the
chip's own calibration fingerprint via a minimum-viable fuzzy
extractor (quantise per-dim μ at a fixed stride, hash). K_chip is
enrolled with V over a physically-secure channel and is never on the
wire.

**Protocol.**

1. V → P: 64-bit audience nonce N (`secrets.token_bytes`).
2. P derives a sampling plan via
   `SHAKE256(K_chip || domain_sep || N)` into seven independent,
   domain-separated byte streams driving:
   - 8-of-16 CPU subset
   - 3-of-N thermal-zone subset
   - 16-of-120 core-pair selection
   - per-CPU nanosleep duration grid
   - nanosample count
   - TSC sample count
   - 32-position output permutation perm32
3. P performs the live HAL-bypass reads under SCHED_FIFO 99 +
   mlockall + cpuset isolation, assembles the 290-dim phys vector,
   permutes by perm32, returns to V.
4. V re-derives the plan from N + V's stored K_chip, reconstructs
   the inverse permutation, and runs **both** gates:
   - `plan_pass`: multi-dim Mahalanobis-style band test over all
     32 un-permuted dims against the enrolled per-dim (μ, σ).
   - `classifier_p0 > τ_cls`: LDA trained on enrollment reps.

**Acceptance:** `plan_pass AND classifier_p0 > τ_cls`. Both required.

---

**Where v2.0 broke (O115 disclosure).**

In v2.0, the plan-consistency gate was the *sole* accept criterion,
and the quantity it checked at dim 31 of the un-permuted vector was
the *input parameter* `plan['ns_sleep']` written directly into the
output by the prover. The plan was derived under a *public* HMAC key,
so any source-code-aware adversary could compute the expected gate
value in microseconds, fabricate a vector with that value at the
perm-derived position, and pass with probability ≈ 1.0. Cost $0,
success 1.0, detection 0. The seven-attack battery did not test this
adversary; it only tested replay and mismatch.

**v2.1 patch (Tier 1):**

1. Dim-31 is now a real measurement (MAD of an independent second
   nanosleep burst).
2. HARD veto: acceptance requires both gates; plan-pass is now
   multi-dim Mahalanobis, not single-dim canary.
3. Keyed plan derivation: SHAKE256 keyed on K_chip, never public.
4. Independent SHAKE256 streams per plan component (eliminates
   host-RNG-order bug and cross-component leak).

---

**Honest bit-security accounting.**

| Attacker model | Residual unforgeability |
|---|---|
| Source-code aware, no K_chip | ~60–80 bits (Tier 2 ceiling) |
| Has captured K_chip (Tier-2 break) | ~15–20 bits (~10⁵ guesses) |
| ≥ 10⁵ observed (nonce, sig) + generative model | undefended |

The first row is bounded by SHAKE256/HMAC minus brute-force on
per-die fingerprint entropy (≈ 32 dim × log₂(quantisation-levels)
≈ 100+ bits in principle, reduced to ~30–40 bits in practice by the
SNR floor on physically-reproducible dims).

The middle row is the **honest** ceiling once K_chip leaks: the
classifier slack plus the per-dim Mahalanobis band admits ~10⁵
guesses.

The third row is the headline future-work item. **Tier 2 / Tier 3
defences** (distance-bounding to defeat LAN relay, raw micro-sample
series with cross-signal algebraic constraints, multi-round protocol
with tight per-round RTT, TPM-sealed K_chip) are explicitly future
work.

---

**What I'm asking r/cryptography:**

1. Is the domain-separation scheme strong enough? Each plan
   component pulls from `SHAKE256(K_chip || "cpu_subset" || N)` etc.
   I'd like a sanity check on whether the 32-byte domain tag is
   adequate.
2. Is the Mahalanobis band the right primitive for `plan_pass`?
   A determined attacker with K_chip + classifier-output access
   could in principle gradient-descend the band; we'd like a
   stronger bound.
3. The generative-model adversary (third row) is undefended. Is
   there a known technique for proving non-cloneability of a
   distribution-fitted phys vector under a fresh nonce that I am
   missing?

---

**Code:** https://github.com/Heigke/FabricCrypt (MIT)
**Paper:** arXiv:XXXX.XXXXX (full bit-security accounting in §5.6
and §5.8)

n=2 chassis. We say so. Adversary B (manufacturer-scale chip
cloning) is acknowledged but untested.
