# O116 — Pre-launch sanity check for FabricCrypt arXiv submission

**Date:** 2026-06-01
**Stance requested:** sober, honest, ground in actual literature and threat
model. Goal: figure out if we are about to embarrass ourselves on arXiv,
or if this is a defensible contribution. Be specific. No diplomatic hedging.

---

## Context (one-paragraph)

FabricCrypt v3 (`fabriccrypt_v3.md`, 1362 lines) is a paper draft on
**software-discoverable, vendor-key-free per-die attestation** for AI
inference on commodity AMD APUs (Strix Halo, Ryzen AI Max+ 395). It
combines (i) 13 HAL-bypass micro-architectural signals into a 466-dim
per-die signature (100% LOO at n=2 chassis), (ii) a Phase 14C
nonce-driven sampling-plan protocol (10/10 attack gates pass after
O115-driven hardening, replay ≤ 0.6%), (iii) Tier-2 hardening
(Reverse Fuzzy Extractor + Controlled-PUF wrap + multi-round +
ZK-binding scaffold) raising bit-security from ~2^30 to ~2^60–80,
(iv) a HONEST NULL on Phase 21b stylometric "personality" emergence
(observed 0.664 vs preregistered 0.75 gate; n=420). Target venue:
USENIX Security or ACM IH&MMSec; planned arXiv launch within days.

Prior O115 found (and we fixed) a fatal forge break in the Phase 14C
gate; the 10/10 battery in v3 includes the O115-custom forgery as
a gate.

---

## Our claims (please attack each)

1. **First software-discoverable per-die attestation on commodity AMD
   without vendor PKI** (no Apple Secure Enclave, no NVIDIA DICE, no
   Intel TDX, no AMD SEV-SNP root of trust used).
2. **13 signals, 466-dim signature, 100% LOO at n=2 (ikaros + daedalus,
   both HP Z2 mini G1a / Strix Halo).**
3. **Phase 14C nonce-protocol: 10/10 attacks blocked, replay ≤ 0.6%.**
4. **Tier-2 hardening: claimed 60–80 bits of security; ML modeling
   attack defeated via Controlled-PUF wrap.**
5. **Phase 21b stylometric "personality" detectable at 66.4% (n=420),
   below the preregistered 0.75 gate. We report this as a NULL.**

---

## Files in this bundle (read them before answering)

- `fabriccrypt_v3.md` — full paper draft (1362 lines, target venue
  USENIX Sec / ACM IH&MMSec).
- `threat_model_and_signals.md` — paper v2 §3/§4.1/§5.5 excerpts
  (threat model, 5 baseline signals, adversary analysis).
- `O115_synthesis.md` — prior oracle finding that broke Phase 14C v1
  and forced the gate redesign now in v3.

---

## Questions (be specific, ground in the bundled paper text)

### A. Scoop / novelty risk

A1. Given the paper draft, has anyone published the *same* primitive in
the last 6–12 months that we missed? Specifically:
  - per-die fingerprinting on commodity x86 *without* vendor PKI;
  - HAL-bypass signature ≥ ~10 signals at the OS-/microarch boundary;
  - nonce-driven sampling plan defending against dynamic-library replay.

A2. Is "per-die attestation without vendor PKI on commodity AMD"
actually novel as we claim, or are we missing existing work (CPU-Print
S&P'25 follow-on, Energon Aug'25 follow-on, FP-Rowhammer AsiaCCS'25
follow-on, any 2026 IACR ePrint, USENIX Sec'26 accepted papers)?

A3. Specific risk papers we know about and how we differentiate:
  - **CPU-Print** (S&P'25 poster). Did a full paper appear? We
    differentiate on signal count (13 vs ~3) and protocol (nonce-driven
    sampling plan, not static fingerprint).
  - **Energon** (Aug 2025). We differentiate on substrate (commodity
    APU vs. server CPU) and AI-inference binding (we propose ZK-binding
    scaffold; Energon does not).
  - **FP-Rowhammer** (AsiaCCS'25). Different attack model; we
    *defend*, they *attack* (DRAM-level fingerprint as a side channel).
Are these differentiators credible, or are we kidding ourselves?

A4. Patent landscape (USPTO + Google Patents 2025–2026) for "chip-bound
AI", "per-die machine-learning attestation". Anyone filed?

### B. Sharpest reviewer critique

B1. What's the sharpest critique a USENIX Security 2027 reviewer would
write in 5 minutes after reading the abstract and §4?

B2. What's the sharpest critique a *cryptographer* reviewer would
write? (Specifically about the Tier-2 60–80 bit claim — is it
defensible from first principles, or hand-wavy?)

B3. What's the sharpest critique a *hardware security* reviewer would
write? (Specifically: are the 13 signals actually stable on a single
chip across reboots, thermal cycles, BIOS updates? We have NOT shown
3-month stability; we have shown ~1-week stability.)

### C. Strongest defensible claim at n=2

C1. With n=2 chips, what is the strongest single-sentence claim we can
defend in the abstract? Be conservative.

C2. Should we re-frame the paper as a *primitive* (mechanism +
protocol) rather than a *capability* (per-die attestation), to dodge
the "n=2 → can't claim biometric-grade ID" critique?

C3. If we had to drop one of the 13 signals to be honest about
generalisation, which would we drop and why?

### D. Phase 21b — drop or keep?

D1. Phase 21b is 66.4% (n=420), preregistered gate 0.75, p << 0.001
vs chance. We frame this as a NULL on the *strong* claim
("personality emerges") and a *positive* on the *weak* claim
("output is detectably chip-conditioned"). Is this framing defensible,
or are we double-dipping?

D2. Should we drop the personality framing entirely from the abstract
and only mention it in §6 (limitations + future work)?

D3. Would a reviewer interpret "personality emergence detectable at
66.4%" as a positive or a negative result given the preregistered
gate?

### E. Go / no-go and required edits

E1. Quantify: P(arXiv launch is net-positive for our reputation)
given the current draft. Provide a number (0.0–1.0) and a one-sentence
justification.

E2. List the **top 5 mandatory edits** to v3 before arXiv submission
(in priority order).

E3. List the **top 3 nice-to-have edits** (skip if time-bound).

E4. Estimated probability of *first-cycle* reviewer acceptance, by
venue:
  - HOST 2026
  - RAID 2026
  - ACSAC 2026
  - USENIX Security 2027
  - ACM CCS 2027

E5. Any IP / legal risk? (Patent infringement on CPU-Print/Energon/etc.
prior art? GPL contamination from PSP RE? PUF patent thicket?)

### F. Independent free-form

F1. Anything else we should know before we hit "submit" on arXiv.

---

## Output format

Per question, give a direct answer (≤ 5 sentences) followed by the
*specific evidence* you used (paper line numbers, paper sections, prior
work citations with year + venue if claiming pre-emption). If you
don't know, say so explicitly — guesses are worse than admitted
ignorance.
