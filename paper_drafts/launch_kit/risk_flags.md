# Launch risk flags — what will hit us and what we say back

Each flag is paired with the exact-words rebuttal and the on-paper /
on-repo evidence to point at. Brief, so it's usable live.

---

## R1 — n=2 chassis (probability: certain; severity: high)

**Attack:** "Two laptops is not a study. Come back with 1000 chips."

**Rebuttal:** "Agreed it's small, and we say so in §7. n=2 is sufficient
to demonstrate the primitive *exists* in a software-discoverable way —
100% LOO at 290 features, sub-ms latency, all ten gates pass. The right
next move is replication, not bigger N from us. The artifact is
released. We'd like you to bring a third chassis."

**Evidence:** Section 7 (L1). `scripts/enroll.sh` is one command.

**Mitigation pre-launch:** N=3+ would dramatically defuse this. Send
`casper_email.md` ASAP. If even one outside chassis lands before T=0,
fold it in and update tweet 9.

---

## R2 — Phase 21b personality is 66.4% not 75%

**Attack:** "Your viral thread implies the chip controls the AI's
personality. The actual number is 66.4% — that's not personality
binding, that's a noisy classifier."

**Rebuttal:** "Correct. 66.4% is above chance, below ironclad, and is
in our Limitations section (L6 of the preprint) and tweet 9 of the
launch thread. We do not claim the chip *causes* personality; we claim
a held-out classifier can distinguish chip-of-origin from generated
text at 66.4%. That's a measurement, not a metaphysics claim."

**Evidence:** Section 7 (L6), tweet 9, README "honest caveats" section.

**Mitigation:** Headline tweet 9 prominently in the X thread; do NOT
let any marketing copy say "75%" or imply substrate-determinism.

---

## R3 — DRM concerns (probability: high; severity: medium)

**Attack:** "This is Apple PCC for laptops. You're enabling
vendors to lock AI to specific chips. Whose interests does this
serve?"

**Rebuttal:** "Three answers. (a) Unlike Apple PCC, FabricCrypt does
not require any vendor key. You generate your own per-die identity on
your own laptop, no enrollment with us or anyone. (b) The primitive
proves *who ran the inference*, not *what is allowed to run*. Your
bootloader is untouched; Linux runs freely. (c) The motivating use
cases are output attribution for AI liability and sybil-resistance in
federated learning — both *protect* the user from impersonation."

**Evidence:** Section 6 of the paper, MIT license, no vendor
enrollment step in `enroll.sh`.

**Mitigation:** README "use cases & non-use-cases" section. Press
release leads with "no vendor key required."

---

## R4 — O115 forgery (probability: medium; severity: high if mishandled)

**Attack:** "Your v2.0 was completely broken (O115). Why should we
trust your v2.1?"

**Rebuttal:** "We disclosed O115 in the paper (§5.8) and patched four
Tier-1 fixes in v2.1: keyed plan derivation, multi-dim Mahalanobis
gate, independent SHAKE256 streams per plan component, and a real
measurement at dim 31. v2.1 rejects O115 at 100%. The whole point of
publishing v2.1 *with* the v2.0 break disclosed is to show the gate
battery works as a critique-acceptance pipeline. Tier 2/3 defences
are explicitly future work."

**Evidence:** Section 5.8 + Table 5 (custom_forgery_o115: v2.0 1.00,
v2.1 0.00). `embodiment14d_crypto/` directory.

**Mitigation:** Tweet 7 honest about bit-security. HN Q&A item Q8.

---

## R5 — Side-channel reconstruction (probability: medium; severity: medium)

**Attack:** "Hertzbleed / Energon can reconstruct your fingerprint
remotely. Your primitive is broken."

**Rebuttal:** "The fingerprint is intentionally observable; we don't
claim it's secret. What FabricCrypt defends is *re-use* of an observed
fingerprint across a fresh challenge: each nonce re-randomises the
sampling plan, so a reconstructed phys vector cannot be replayed.
This is Adversary C in §5.5 and we account for it."

**Evidence:** Section 5.5 (Adversary C). HN Q&A item Q6.

---

## R6 — "It's just /proc/cpuinfo dressed up"

**Attack:** "How is this not just reading a CPU serial number?"

**Rebuttal:** "We read no vendor identifier. The five signals are
inter-core TSC offsets, MOESI ping-pong, DRAM-refresh-aligned jitter,
nanosleep p99.9 tails, NVMe queue-tail latency. None are
factory-programmed. Two laptops with identical `dmidecode` separate
at 100% on these signals."

**Evidence:** Section 4.1 of the paper, source of `nonce_signature_v2.py`,
`dmidecode` panel in the demo video showing identical SKUs.

---

## R7 — Persistent kernel adversary

**Attack:** "Root on the prover machine defeats this trivially."

**Rebuttal:** "True, and we list it as L5. The protocol assumes
SCHED_FIFO + mlockall + cpuset isolation and preemption disabling
around critical sections. Persistent-kernel adversaries are outside
our threat model. PCC has the same assumption (compromised Secure
Enclave → defeat PCC)."

**Evidence:** Section 7 (L5).

---

## R8 — vocabulary policing (low probability, but easy to mess up)

**Banned in launch copy:** die, kill, soul, feel, loyalty, sentient,
alive. (See viral-audit vocabulary policy.)

**Use instead:** bound, coupled, substrate-locked, fingerprint,
instrument-dependent, non-portable, hardware-encrypted, attestation.

**Why it matters:** "die" reads as morbid in headlines and "soul"
puts us in the consciousness-hype bucket alongside content we don't
want to be co-classified with. The story is **attestation**, not
sentience.

**Note:** "per-die" (the silicon engineering term) is acceptable in
the paper and in technical contexts but should be paraphrased as
"per-chip" in tweets and press copy.

---

## Quick reference card (laminated card for live Q&A)

| If they say | Say back |
|---|---|
| n=2 is laughable | §7, replication is next, repo is open |
| 66.4% personality is weak | Yes — limitation L6, not the headline |
| This is DRM | No vendor key, no boot lock, see §6 |
| You were just broken (O115) | Disclosed & patched, Table 5, Tier 1 done |
| Hertzbleed reconstructs you | Adversary C, fingerprint isn't secret |
| Just /proc/cpuinfo | Five physical signals, source on screen |
| Root defeats you | L5, same as PCC w/ broken Enclave |
| Just a PUF | Nonce-bound, live, multi-signal — not static |
