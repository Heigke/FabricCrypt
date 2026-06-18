# O116 — Synthesis: Pre-launch sanity check for FabricCrypt v3 arXiv submission

**Date:** 2026-06-01
**Oracles:** openai (gpt-5, 112s), gemini-2.5-pro (72s), grok-4-latest (16s),
             deepseek-reasoner (54s)
**Bundle:** `prompt.md`, `fabriccrypt_v3.md` (1362 LOC), `threat_model_and_signals.md`,
            `O115_synthesis.md`
**Online search:** 16 targeted searches across Google Scholar / arXiv / IACR
                   ePrint / USENIX Sec '26 Cycle 1 / IEEE S&P / Springer / IETF /
                   Patent literature. Top results enumerated in §3 below.

---

## TL;DR

**Net-positive probability spread is wide: 0.35 / 0.35 / 0.45 / 0.95**
(deepseek / grok / openai / gemini). **Median 0.40. Mean 0.53.**

**Recommendation: CONDITIONAL GO** — launch on arXiv **only after** the 5
mandatory edits below. Without those edits, the centre-of-mass estimate is
**a net-negative reputation move at top venues** (mean of bottom-3 oracles
= 0.38) but **strongly net-positive among honest-rigor-friendly readers**
(gemini 0.95).

No scoop found in last 12 months. The closest prior work (CPU-Print S&P'25
poster, Eckel/Fenzl/Jäger SEC 2024 hardware fingerprinting, LAMINATOR
CODASPY 2025) is **adjacent but does NOT pre-empt** the specific
contribution: nonce-driven sampling plan + 13 HAL-bypass signals on
commodity AMD APU + no vendor PKI. The novelty claim **holds at the
primitive level** but should NOT be stated as "first software-discoverable
per-die attestation" without the qualifier "primitive" and the explicit
"n=2" admission.

---

## 1. Per-oracle headline numbers

| Oracle    | P(net-positive) | Top-tier venue hopes (USENIX/CCS) | Tone        |
|-----------|----------------:|----------------------------------:|:------------|
| openai    | 0.45            | 0.05                              | Surgical    |
| gemini    | 0.95            | 0.20–0.30                         | Bullish     |
| grok      | 0.35            | 0.05–0.08                         | Pessimistic |
| deepseek  | 0.35            | 0.10                              | Cautious    |

Gemini is the outlier on the upside; the other three converge on a
"interesting primitive but n=2 + bit-security claims will draw blood at
top venues" view. **Lower-bound venue hopes are unanimous at ~5–10 %
USENIX Sec / CCS 2027 acceptance.**

Mid-tier venues (HOST / RAID / ACSAC) are more split (0.15–0.70). Gemini
puts HOST 2026 at 0.6–0.7; others at 0.15–0.40. **Consensus mid-tier
estimate: HOST 2026 ≈ 0.30, RAID 2026 ≈ 0.25, ACSAC 2026 ≈ 0.20.**

---

## 2. Consensus mandatory edits (all 4 oracles concur on ≥3 of these)

Listed by number of oracles flagging the issue:

**E-M1. Downgrade bit-security claims (4/4).** "2^60–2^80" appears in
abstract & §5.10.5 without a formal reduction. All four oracles flag this
as the single biggest red flag for a cryptographer reviewer. **Required
edit:** replace exponent claims with empirical attack-cost bounds + CIs +
explicit "heuristic estimate, no formal reduction" caveat. OpenAI is
sharpest: "remove bit-exponent claims unless you state a precise game and
prove bounds under standard assumptions."

**E-M2. Re-frame as primitive, not capability, with explicit n=2 (4/4).**
Abstract leads with "first software-discoverable per-die attestation
primitive" but body sometimes reads as capability claim. **Required
edit:** systematic search-and-replace from "we obtain LOO 1.000" to "on
our n=2 testbed we demonstrate LOO 1.000." Add "at n=2" inside the
headline sentence of the abstract.

**E-M3. Move Phase 21b personality NULL out of abstract (3/4: openai,
grok, deepseek; gemini disagrees).** Three oracles say a 0.664 vs 0.75
pre-registered gate reads as a negative result and including a positive
spin in the abstract risks "salvage" framing. Gemini argues the
transparency is a strength and should remain. **Decision:** drop from
abstract, keep prominent in §7 with the verdict "FAIL pre-reg, PASS
secondary detection." Gemini's argument about credibility-by-honesty is
still served if §7 is leaned into.

**E-M4. Resolve internal protocol inconsistencies (openai only, but
SEVERE).** §5.1/§5.2 derive plan from `audience_secret`, but §5.8
(post-O115) derives from `K_chip`. §5.4 acceptance uses classifier on raw
phys, but §5.10.2 claims only `SHAKE256(K_chip ‖ c ‖ raw_phys(c))`
leaves the chip. **These are contradictions in the same paper.** OpenAI
correctly flags this as the kind of finding a hostile reviewer will use
to label the paper "rushed." **Required edit:** pick one plan-derivation
secret (K_chip per §5.8), document it once, propagate. Pick one
acceptance predicate (classifier on hashed Controlled-PUF output) and
remove the conflicting one.

**E-M5. Quarantine S20–S26 deterministic board signals (2/4: openai +
grok; deepseek agrees they're spoofable).** OpenAI: "S20–S26 are
trivially forgeable in a software-forgery model and not per-die." Calling
13 signals when ~5 of them are board-inventory deterministic features
that any software forger can replay is a credibility hazard. **Required
edit:** either (a) move S20–S26 out of the "13 HAL-bypass signals"
headline and present as "board-inventory aids, not identity evidence,"
OR (b) explicitly justify why a software adversary cannot replay them
(currently no convincing argument exists).

**Additional consensus item — relay attack disclosure (4/4).** V6 = 0
bits is acknowledged but the abstract sells "10/10 attack gates passed"
without naming relay as out-of-scope. **Edit:** add one sentence to
abstract acknowledging relay is out-of-scope and to §3 (threat model)
making this a first-class assumption.

---

## 3. Online literature scan — top hits (no pre-empting paper found)

Ranked by *threat to novelty claim*. None of these are fatal; the top
five should be cited in §2 to demonstrate awareness.

| # | Paper / artefact                                                                                                  | Year | Venue          | Relevance | Notes |
|---|------------------------------------------------------------------------------------------------------------------|------|----------------|-----------|-------|
| 1 | Eckel, Fenzl, Jäger — Towards Practical Hardware Fingerprinting for Remote Attestation                            | 2024 | IFIP SEC 2024  | HIGH      | Closest direct competitor. Uses ADC values + environmental correlation for attestation. **Embedded systems substrate, not commodity AMD APU.** Cite in §2; differentiate on substrate + protocol + signal-count. |
| 2 | Venugopalan et al. — FP-Rowhammer: DRAM-Based Device Fingerprinting                                                | 2025 | AsiaCCS '25    | MED       | Already cited. Static DRAM fingerprint, 99.91% on 98 modules. **Attack-oriented; no nonce protocol.** Our defence framing is distinct. |
| 3 | Anonymous — Energon: Unveiling Transformers from GPU Power & Thermal Side-Channels (arxiv 2508.01768)             | 2025 | ICCAD '25      | MED       | Already cited. Power/thermal side-channel **extracts model architecture, not attests to chip identity.** Orthogonal. |
| 4 | Duddu, Boutet, et al. — Laminator: Verifiable ML Property Cards using Hardware-assisted Attestations              | 2025 | ACM CODASPY '25| MED       | Adjacent — uses **TEE-rooted** attestation for ML property cards. We are the explicitly vendor-PKI-free alternative. Must cite in §2 to position. |
| 5 | Ivanov et al. — SAGE: Software-based Attestation for GPU Execution                                                 | 2023 | USENIX ATC '23 | MED       | Software-based GPU attestation but uses **host SGX enclave as local verifier** (so still TEE-anchored). We are vendor-PKI-free; distinguish in §2. |
| 6 | Herrmann et al. — Towards Remote Attestation of Microarchitectural Attacks: HammerWatch (arxiv 2603.24172)        | 2026 | preprint       | LOW-MED   | Rowhammer-aware remote attestation on commodity. Uses MCE+PRAC. **Defends against attacks, doesn't establish per-die identity.** Orthogonal but should cite. |
| 7 | Anonymous — PUFs for IoT Authentication and Hardware-Anchored AI Model Integrity (arxiv 2604.21188)               | 2026 | preprint       | LOW       | **Survey/review** of hardware roots of trust. Mentions our problem space but proposes no specific competing primitive. |
| 8 | Anonymous — Sharing is caring: Attestable and Trusted Workflows (Mica) (arxiv 2603.03403)                         | 2026 | preprint       | LOW       | TEE-rooted communication-path attestation. Different layer. |
| 9 | Laor, Mehanna et al. — DRAWNAPART deep-learning enhanced GPU fingerprinting                                       | 2023 | NDSS '23       | LOW       | Already cited (NDSS '22 + '23 deep-learning follow-up). Static GPU fingerprint. |
| 10 | Kim et al. — DRAM Latency PUF                                                                                    | 2018 | HPCA '18       | LOW       | Already cited. Static. |
| 11 | OpenPCC — open-source Private Cloud Compute clone (Github)                                                       | 2025 | open-source    | LOW       | Software framework for private inference, still TEE-rooted. Could mention as motivation. |
| 12 | Apple — Private Cloud Compute open-source release                                                                | 2025 | industry       | LOW       | Already cited. Confirms the vendor-PKI direction we are diverging from. |
| 13 | Tobin South et al. — Verifiable evaluations of ML models using zkSNARKs                                          | 2024 | arxiv          | LOW       | zkML. Different threat model — proves computation, not chip identity. |
| 14 | Lagrange DeepProve — production zkML                                                                             | 2025 | industry       | LOW       | zkML. Not chip-bound. |
| 15 | EZKL                                                                                                              | ongoing | open-source | LOW       | zkML. Not chip-bound. |
| 16 | IETF draft-sharif-ai-model-lifecycle-attestation-00                                                              | 2026 | IETF draft     | LOW       | Spec draft for AI lifecycle attestation. Vendor-PKI rooted in implementations. |
| 17 | Wang et al. — DeMiCPU magnetic CPU fingerprinting                                                                | 2022 | ACM TOSN       | LOW       | Magnetic induction; requires physical sensor proximity. Orthogonal. |
| 18 | Mitrokotsa et al. — Reverse fuzzy extractors                                                                     | 2012 | FC '12         | LOW       | Cited as primitive.  |
| 19 | USENIX Security '26 Cycle 1 accepted papers list                                                                 | 2026 | USENIX         | NEED-RECHECK | Page returned 403 to WebFetch; recommend manual inspection of the published accepted-papers list for any per-die fingerprint title. |
| 20 | IACR ePrint 2026 batch                                                                                           | 2026 | IACR           | LOW       | No specific 2026 ePrint paper matching the primitive surfaced in search. |

**Verdict: NO RED ALERT.** No paper in the last 12 months pre-empts the
specific combination (commodity AMD APU + ≥10 HAL-bypass signals +
nonce-driven sampling plan + vendor-PKI-free + AI-inference binding).
However, the **Eckel SEC 2024** paper is close enough that it MUST be
cited in §2 with a clear differentiation paragraph. We currently do not
cite it.

**Action item (audit gap):** verify Eckel/Fenzl/Jäger 2024 SEC paper is
cited; if not, add citation + 1-paragraph differentiation. Verify
LAMINATOR (CODASPY '25) is cited; if not, add.

---

## 4. Phase-21b decision (66.4 % vs 0.75 gate)

3-of-4 oracles say **drop from abstract, keep in §7**. Gemini argues the
honesty is a strength and should remain in abstract. **Synthesised
decision:** drop from abstract headline, but keep the *fact of having
pre-registered and reported a NULL* in the abstract as a one-clause sign
of rigor. Use this phrasing (suggested):

> "We also pre-registered and report a NULL on a stylometric
> personality-emergence gate (§7)."

This (a) signals rigor (Gemini's concern), (b) removes the positive
spin on 66.4 % (openai/grok/deepseek's concern), (c) avoids the
double-dipping smell of "failed pre-reg but p<<0.001 vs chance" in the
abstract.

---

## 5. Final go / no-go recommendation

**CONDITIONAL GO** — proceed with arXiv submission after **all five E-M
edits** above are applied. Specifically:

1. Apply E-M1 (downgrade bit-security claims).
2. Apply E-M2 (primitive framing + explicit n=2).
3. Apply E-M3 (Phase 21b: shorten + neutral framing in abstract).
4. Apply E-M4 (resolve plan-derivation and acceptance-predicate
   contradictions across §5.1 / §5.4 / §5.8 / §5.10.2 — this is the
   most concrete and serious edit; without it, the paper is internally
   inconsistent in a way reviewers will flag immediately).
5. Apply E-M5 (quarantine S20–S26 or justify against software forger).
6. Add Eckel/Fenzl/Jäger SEC 2024 + LAMINATOR CODASPY 2025 to §2
   (literature audit gap).
7. Add explicit relay-attack out-of-scope sentence to abstract + §3.

**Time estimate to apply: 3–5 hours of focused writing + one careful
proofread.**

**If unwilling to do E-M4 right now (the only structurally hard one):
DO NOT LAUNCH.** The contradiction between §5.1/§5.2 (audience-secret
plan) and §5.8 (K_chip plan) is the kind of thing that gets screenshotted
on Twitter.

---

## 6. Estimated post-edit reviewer acceptance probabilities

Using consensus mid-tier mean weighted toward the more critical oracles
(openai/grok/deepseek), and assuming all 5 mandatory edits applied:

| Venue            | P(first-cycle accept) | Comment |
|------------------|----------------------:|:--------|
| HOST 2026        | 0.30–0.45             | Best topical fit. n=2 stays as a known hazard. |
| RAID 2026        | 0.25–0.35             | Protocol + adversarial-eval is the right framing for this venue. |
| ACSAC 2026       | 0.20–0.30             | Applied venue; small-scale data is a recurring weakness. |
| USENIX Sec 2027  | 0.05–0.15             | Top-tier bar; n=2 is borderline-fatal even after re-framing. Needs n≥6 + 3-month stability for ≥0.30. |
| ACM CCS 2027     | 0.05–0.15             | Same. Cryptographer reviewers will hammer bit-security claims even after caveat. |
| IEEE EuroS&P 2026| (not asked, plausible 0.20–0.30) | Decent mid-tier alternative. |
| FC '27 workshop  | 0.40–0.60             | Deepseek's suggestion. Lower-bar venue tolerant of small-N proof-of-concept. |

**Strategic recommendation:** target **HOST 2026** as primary venue (best
topical fit + acceptance probability), with arXiv preprint up first to
establish priority date. Avoid USENIX Sec / CCS submission until n≥6 +
3-month stability data + formal security model are in hand.

---

## 7. IP / legal risk

Low. All four oracles concur:

- Controlled-PUF (Suh-Devadas, US 7,940,642) is patented; we cite the
  paper but use it conceptually. **Academic use is safe; commercial use
  requires FTO.**
- Reverse Fuzzy Extractor (VanHerrewege 2012) similarly cited.
- No GPL contamination — we use sysfs + perf counters, not modified
  kernel code.
- No infringement on CPU-Print / Energon / FP-Rowhammer from text alone.
- **Recommendation:** add a one-line patent disclaimer to §8 ("Several
  primitives we build on are patented; freedom-to-operate analysis is
  out of scope for this academic disclosure").

---

## 8. Concrete edit checklist (paste into editor TODO)

```
[ ] Abstract: replace "2^60–2^80" with "empirical attack-cost
    estimates; not a formal reduction"
[ ] Abstract: add "at n=2" qualifier to LOO 1.000 sentence
[ ] Abstract: replace "first software-discoverable per-die
    attestation" with "first software-discoverable per-die attestation
    primitive"
[ ] Abstract: shorten Phase-21b mention to one neutral clause
[ ] Abstract: add "Relay is explicitly out of scope" clause
[ ] §1: add explicit n=2 qualifier matching abstract
[ ] §2: add Eckel/Fenzl/Jäger SEC 2024 citation + differentiation
    paragraph (currently missing!)
[ ] §2: add LAMINATOR CODASPY 2025 citation + differentiation
    paragraph
[ ] §3: state relay attack is out-of-scope as a first-class assumption
[ ] §4.1c: re-label S20–S26 as "board-inventory aids" not
    "HAL-bypass signals"; remove from "13 signals" headline OR justify
    against software forger
[ ] §5: resolve plan-derivation: pick K_chip (per §5.8), propagate to
    §5.1, §5.2; remove audience-secret derivation
[ ] §5: resolve acceptance predicate: state once whether classifier
    runs on raw phys (then §5.10.2 is wrong) or on Controlled-PUF
    hashed output (then §5.4 needs update)
[ ] §5.10.3: remove ≤150 µs RTT enforcement claim (unenforceable
    without hardware timing)
[ ] §5.10.5: replace bit-exponent column with empirical attack-cost
    estimates + CIs; add caveat row "no formal reduction"
[ ] §5: add paragraph on how K_chip is established and stored per die
[ ] §7: keep Phase 21b NULL as L6; add new L10 explicitly flagging
    "no longitudinal stability beyond ~1 week"
[ ] §8: one-line patent disclaimer
[ ] Run a global search for "first" + "attestation" + "per-die" and
    soften every hit to "primitive"
```

---

## 9. What we DID NOT find (gaps in this audit)

- Could not directly inspect the **USENIX Security '26 Cycle 1
  accepted-papers list** (HTTP 403 to WebFetch). Manual browser check
  is recommended before arXiv launch.
- Could not perform live USPTO / Google Patents query for "chip-bound
  AI" / "per-die ML attestation" — no oracle could either. Recommend a
  cheap freedom-to-operate search by counsel before any commercial use,
  but this is NOT a blocker for arXiv.
- Could not verify whether CPU-Print (S&P '25 poster) has produced a
  full paper at IEEE S&P '26 or a recent IACR ePrint. Recommend a
  manual sp2026.ieee-security.org accepted-papers check before launch.

---

## 10. One-paragraph summary for the launch decision

FabricCrypt v3 is a technically rigorous, transparently self-audited
paper that introduces a genuinely novel primitive (nonce-driven sampling
plan over 13 HAL-bypass signals on commodity AMD APUs, no vendor PKI).
The novelty claim survives a 12-month literature scan. The paper has
**three real weaknesses** that an arXiv launch will not hide: n=2 chassis
count, hand-wavy 2^60–2^80 bit-security claims, and internal protocol
inconsistencies between §5.1/§5.2, §5.4, §5.8, and §5.10.2. With the 5
mandatory edits above (3–5 h of work), the paper goes from a likely
**net-neutral-to-negative** reputation move (mean of the three critical
oracles: 0.38) to a **net-positive primitive disclosure** with a
plausible HOST 2026 path. **Apply the edits, then launch.**
