# N=2 Honest Constraint Analysis

What can we prove **right now** with two chassis vs what needs more.

## What N=2 already lets us prove

| Claim | Why N=2 is enough |
|-------|-------------------|
| **A software-discoverable per-die fingerprint exists.** | Two distinct dies separating at 100% over 290 features is sufficient *existence*. Existence is the entire claim of the v2 paper. |
| **Replay defence works in principle.** | Nonce coverage is bounded by protocol, not by N. 0.6% accept stands. |
| **Transplant detection works.** | Cross-chip false-accept measured at 2%; that number is meaningful at N=2 because it is the *unique* cross-pair. |
| **Live brain transplant beat (the climax)** | Requires exactly 2. Adding chassis adds nothing visual; we'd just transplant the same way. |
| **Per-chip personality differentiation** | At N=2 the audience experiences "K-2 is K-2, BD-1 is BD-1." Adequate for the dramatic claim. |
| **Vendor-key-free attestation primitive** | A property of the *protocol*, not of N. |

## What N=2 does **not** let us prove

| Claim | Why it needs more chassis |
|-------|---------------------------|
| **Population-level uniqueness** (the demo "any chip in the world is detectable"). | N=2 gives one pairwise distance. We need ≥10 to show distribution of pairwise distances and bound a false-accept-rate over a fleet. |
| **Sybil resistance at federation scale.** | Use case 3 in the capability landscape doc. Needs ≥10 ideally ≥100. |
| **Manufacturing-batch effects.** | Are two chips from the *same wafer* still separable? Need ≥4 from at least 2 wafers. |
| **Generalisation beyond Strix Halo.** | Need a different SKU (e.g., Ryzen AI 300 desktop, Phoenix laptop). |
| **Temperature / age robustness across population.** | Per-chip we can test; cross-chip variance over months needs panel. |

## Mitigations within N=2 budget

1. **Time-as-pseudo-N.** Run enrolment + verify across multiple days, multiple
   ambient temps. Each day adds a virtual chassis instance to the fingerprint
   stability story (R1).
2. **Microcode / BIOS sweep.** Vary BIOS version on one chassis; show the
   fingerprint is *robust to firmware updates*, not just to time. Adds a
   "fingerprint is real, not a firmware accident" axis.
3. **Borrow a third chassis** (Phase 21 R8) for the demo only. Even one cameo
   bumps the credibility ceiling enormously.
4. **Workshop / hackathon enrollment booth.** Anyone in the audience with a
   Strix Halo can enroll their chip during the Q&A; we publish the resulting
   pairwise distance matrix. This converts the audience into experimenters.

## Recommended language in the talk

> "Two chassis is not the claim. The claim is that the primitive works in a
> software-discoverable way on commodity AMD silicon. Two chassis is enough to
> *exist*. The artifact is open; the next question is *fleet behaviour*, and we
> would like you to bring your chips and find out."

This frames N=2 as a feature (here's the existence proof, now help us scale)
rather than a bug (you only had two).
