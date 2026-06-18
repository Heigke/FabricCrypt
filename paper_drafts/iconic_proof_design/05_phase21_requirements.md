# Phase 21 Outcomes Required to Land the Demo

Listed in order of criticality. Each item lists the demo beat it backs and the
acceptance criterion.

## P21-R1 — Cross-day fingerprint stability
**Backs:** Audience nonce roll today must verify identically tomorrow.
**Acceptance:** Median cosine of `signature_today` to `enrolled_signature` ≥ 0.92 over 7 consecutive days per chip; min ≥ 0.88.
**Why:** If we recapture the demo a day later for B-roll, verifier must accept without re-enrollment.

## P21-R2 — Matched-governor capability retention
**Backs:** Sceptic Attack 2 ("you exploited governor differences").
**Acceptance:** Capability gains (attribution, anomaly AUROC) within 2σ of mixed-governor v2 paper numbers when both chips locked to `performance` and `powersave` independently.
**Why:** This is the v2 paper Section 4.4 result extended to the demo pipeline.

## P21-R3 — Live transplant detection rate ≥ 98%
**Backs:** The climax beat.
**Acceptance:** 50 fresh trials (25 K-2→BD-1, 25 BD-1→K-2), random nonce per trial, verifier reject rate ≥ 0.98, no false rejects on honest path.
**Why:** A single false-positive onstage destroys the demo.

## P21-R4 — Sub-3s end-to-end stage latency
**Backs:** "Audience attention." Beats 55-95 (nonce roll → green light).
**Acceptance:** P99 end-to-end (dice photographed → plan hash on screen → both verifiers green) ≤ 3.0 s on the venue laptops with the projector adapter plugged in. Bench on-site 24 h before the talk.

## P21-R5 — Substrate-conditioned personality pipeline
**Backs:** Twin Reveal segment (beats 25-55) and the dual cryptographic-plus-stylistic failure signal.
**Acceptance:**
- (a) Fingerprint-hash → style-template selection is deterministic and documented (32-template table, source on screen).
- (b) Stylometric AUROC ≥ 0.80 separating K-2 and BD-1 completions over a *committed-in-advance* 100-prompt set; ≥ 0.85 on a holdout of 30.
- (c) After weight transplant, the style follows weights, the cryptographic verifier follows chip — and both failures are visible to the audience.
**Why:** This is what makes it Star-Wars-droid memorable, not just secure.

## P21-R6 — Public precommit primitive
**Backs:** Adversarial Attack 1 ("pre-recorded").
**Acceptance:** A one-liner CLI: `fabriccrypt precommit <nonce>` prints `SHA-256(plan(nonce, chip_id))` deterministically and in <100 ms. URL where the value is logged with timestamp, queryable from any audience phone.

## P21-R7 — Replay/peer numbers from a *fresh* run
**Backs:** All attack panels.
**Acceptance:** Re-run `embodiment14c` style attack battery in the week before the talk; report numbers must match within 0.5 pp of v2-paper claims (0.6 / 1.2 / 2.0 / 0.6). If drift, update narration before recording.

## P21-R8 (stretch) — Third chassis cameo
**Backs:** Sceptic Attack 5 ("N=2 laughable").
**Acceptance:** Either (a) borrow a third Strix Halo from a colleague for the talk and run the full battery live, or (b) ship the verifier to a workshop participant ahead of time so they bring their own enrolled signature.
**Why:** N=3 told live is worth more than N=10 told in a paper.

## P21-R9 (stretch) — TTS personality binding
**Backs:** Production polish.
**Acceptance:** Voice/speed mapping is fingerprint-derived (hash → 1 of 6 OpenAI voices); mapping table on screen during the demo.

## Demo gating

| Gate | Requirement(s) | Hard or soft? |
|------|----------------|---------------|
| Can we record at all? | R1, R3, R7 | Hard |
| Can we resist adversarial reviewers? | R2, R6 | Hard |
| Is it Star-Wars-memorable? | R5 | Hard |
| Does it land in 4 minutes? | R4 | Hard |
| Does it survive "N=2"? | R8 | Soft |
| Does it look polished? | R9 | Soft |
