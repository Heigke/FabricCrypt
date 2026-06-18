# Iconic Proof-of-Identity Demo Design — FabricCrypt

**Date:** 2026-06-01
**Status:** design pass complete; awaiting Phase 21 outcomes before recording.
**Prereq reading:** `paper_drafts/fabriccrypt_v2.md`, `paper_drafts/demo_video_v2/STORYBOARD_v2.md`, `research_plan/IDENTITY_CAPABILITY_LANDSCAPE_2026-06-01.md`.

## Brief

Design the single most visually striking, adversarially robust proof-of-
identity demonstration for FabricCrypt, evocative of Star Wars droids
(R2-D2 / C-3PO): same architecture, instantly distinguishable, unforgeable.

## Files

| File | Purpose |
|------|---------|
| `01_brainstorm_matrix.md` | 10 candidate concepts scored on 4 axes (visual, unforgeable, layperson, doable). Top three: Brain Transplant (36), Twin Reveal (35), Audience Nonce (35). |
| `02_top3_storyboards.md` | Beat-by-beat storyboards for the top three with pre-bunked attacks and Phase 21 requirements. |
| `03_picked_concept.md` | The picked composite: "Twin Droids, One Soul Each" — Transplant spine + 30 s Twin Reveal + Audience Nonce bookend. |
| `04_adversarial_defense.md` | Five most-likely sceptic attacks + verbatim responses + on-stage evidence. |
| `05_phase21_requirements.md` | Nine R-items the lab must deliver before recording (R1-R7 hard, R8-R9 soft). |
| `06_production_blueprint.md` | Frame budget, narration timing, real-time visuals, ethics tags, failure modes. |
| `07_constraints_n2.md` | What N=2 can / cannot prove, and how to frame the limit. |

## TL;DR (300 words)

The single best concept is a **4-minute composite** with three movements:

1. **Twin Reveal (30 s)** — Two laptops, K-2 (ikaros) and BD-1 (daedalus).
   Same SKU, microcode, BIOS. Each answers the same prompt; stylometric
   features form two non-overlapping clusters live. *Personalities are
   derived deterministically from each chip's own 290-d fingerprint hash,
   not hand-coded.* This delivers the Star-Wars-droid emotional payoff.

2. **Audience Nonce (40 s)** — A volunteer rolls dice → 64-bit nonce →
   `SHA-256(plan)` precommitted to an audience-visible URL *before* sampling
   fires. This destroys the "pre-recorded" sceptic in one move.

3. **Brain Transplant (60 s climax + 15 s mirror)** — The visceral beat.
   USB stick walks from K-2 to BD-1; the transplanted model boots
   *confidently claiming to still be K-2 at μ=0.845* — and the verifier
   panel turns red. We do the mirror (BD-1 → K-2) for symmetry. Two
   independent failure signals: the *style* feels off **and** the
   *cryptographic verifier* rejects. The layperson grasp of unforgeability.

Wrapping: vendor-PKI comparison, honest caveats (N=2, attribution-not-access-
control), reproduce URL. Close: K-2 says "I am bound to this body. So is
BD-1. Just physics."

### Picked because
- **Maximum visual impact** (USB stick + red X + confident-but-wrong droid).
- **Pre-bunks the strongest sceptic objections** (nonce, identical SKU,
  hard-coded-personalities) *in-frame*.
- **N=2 enough** for the climax — adding chassis adds nothing to the visual.
- **Builds on the existing v2 pipeline** so most of the production budget
  is already spent.

### Blockers to record next week
- P21-R1, R3, R7 (stability, transplant rate, fresh attack numbers).
- P21-R5 (substrate-conditioned personality engine — net-new code).
- P21-R6 (public precommit primitive — net-new code).

## How to use this package

1. Decide whether to ship the composite or just Concept A (transplant alone).
   The composite needs ~3 weeks of work on R5 + R6. Concept A alone is ready
   in ~1 week.
2. Kick off Phase 21 against `05_phase21_requirements.md`.
3. When R-items green, follow `06_production_blueprint.md` for recording.
4. Brief presenter on `04_adversarial_defense.md` line-by-line.
