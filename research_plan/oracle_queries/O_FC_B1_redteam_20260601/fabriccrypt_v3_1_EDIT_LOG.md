# FabricCrypt v3 → v3.1 — Edit Log (O116 mandatory edits)

Source: `paper_drafts/fabriccrypt_v3.md` (1362 lines, 9686 words)
Output: `paper_drafts/fabriccrypt_v3_1.md` (~10300 words)
Date: 2026-06-01

## Summary

All 5 MANDATORY O116 edits applied + 2 optional citations + patent
disclaimer + abstract rewrite.

| E-M | Description | Sections touched |
|-----|-------------|------------------|
| E-M1 | Bit-security DOWNGRADE | Abstract, §1, §5.8, §5.10, §5.10.5, §6.1, §9 |
| E-M2 | Headline reframe (primitive at n=2 chassi) | Title, Abstract, §1, §6, §9 |
| E-M3 | Phase 21b out of abstract | Abstract, §1 (contributions), §9 |
| E-M4 | Resolve §5 contradictions | §5.0 (new), §5.4 (clarification) |
| E-M5 | Quarantine S20–S26 from HAL-bypass | §1 (contributions), §2, §4.1c, §4.1d, §9 |

Plus:
- 2 missed citations added (Eckel et al. 2024 IFIP SEC; LAMINATOR 2025 CODASPY) to §2 and §8
- 1-line patent disclaimer added to §8
- arXiv abstract.txt produced separately

## Detailed change list

### TITLE + FRONT-MATTER (E-M1, E-M2, E-M3, E-M5)
**Before (line 1, v3):**
"# FabricCrypt: Software-discoverable vendor-key-free per-die attestation for AI inference on commodity GPUs"
**After (v3.1):**
"# FabricCrypt: Software-discoverable vendor-key-free per-die attestation **primitive** for AI inference on commodity GPUs (at n=2 chassi)"

Front-matter changelog rewritten to enumerate the 5 O116 corrections;
"5 light deterministic" → "7 board-level deterministic"; the bit-
security range removed from changelog.

### ABSTRACT (E-M1, E-M2, E-M3, E-M5)
- "first software-discoverable" → "a software-discoverable…primitive (at n=2 chassi)"
- "thirteen HAL-bypass" → "15 signals total — 5 HAL-bypass + 3 cross-host KS-verified μ-arch + 7 board-level deterministic"
- "five light deterministic board-fingerprint" → "7 board-level deterministic fingerprint" (PCI/PCIe/USB/DMI/UCSI/amdgpu/kernel-boot)
- The Tier-2 paragraph rewritten: "raises bit-security ceiling from 2^30–2^40 to 2^60–2^80…" → empirical Hamming μ=128 random-floor language, with explicit "no formal cryptographic reduction" caveat.
- Phase 21b stylometry result removed; replaced by single neutral sentence pointing to §7.L6.
- Added explicit "out of scope" for V6 relay attack.
- Closing tag added: "demonstration is at n=2 chassi and the cryptographic ceilings are empirical, not proven."

### §1 INTRODUCTION (E-M2, E-M5)
- "per-die attestation primitive without depending on the vendor's PKI?" → "per-die attestation primitive (at n=2 chassi) without depending on the vendor's PKI?"
- Contribution 1: "13 HAL-bypass signals" → "15 signals total: 5 HAL-bypass μ-arch + 3 cross-host KS-verified μ-arch + 7 board-level deterministic" with explicit caveat that the Phase 22 signals are NOT HAL-bypass.
- Contribution 5: "Bit-security ceiling moves from 2^30–2^40 to 2^60–2^80" → "Operating points are reported as empirical attack-cost, not formal cryptographic reductions"
- Contribution 8: Phase 21b "honest null" headline removed; replaced with "Exploratory stylometric divergence from chip-conditioned training — kept as supplementary detail, not as a headline claim."

### §2 BACKGROUND (Eckel + LAMINATOR added)
Two new paragraphs after the analog AI accelerators paragraph
introducing Eckel, Fenzl, Jäger (IFIP SEC 2024) for embedded-ADC
fingerprinting (closest adjacent) and LAMINATOR (CODASPY 2025) for
TEE-rooted ML attestation (competitor positioning).

Differentiation note added: "FabricCrypt = commodity-AMD, no-TEE;
LAMINATOR = TEE-rooted; Eckel et al. = embedded-ADC."

Also: "thirteen HAL-bypass signals" → "15 signals (5 + 3 + 7)" in Kohno paragraph.

### §4.1c TITLE + INTRO (E-M5)
- Section title: "Five light deterministic board-fingerprint signals" → "Seven board-level deterministic fingerprint signals"
- Intro paragraph rewritten to explicitly mark these as NOT HAL-bypass micro-architectural signals but as board-level deterministic fingerprints.

### §4.1d (E-M5)
- Dimension table updated with "Class" column distinguishing μ-arch from board-level deterministic
- Note added: "Headline count: 5 HAL-bypass + 3 cross-host KS-verified μ-arch + 7 board-level deterministic = 15 signals."
- S27 explicitly placed outside both headline subgroups.

### §4.5 (E-M5)
"thirteen signals" → "15 signals"

### §5.0 PROTOCOL EVOLUTION (E-M4) — NEW SUBSECTION
Resolves the apparent §5.1/§5.2 vs §5.8 contradiction by making the
three-stage protocol evolution explicit:
1. Base: audience-secret-keyed plan (§5.1–§5.7)
2. Tier-1: adds K_chip to plan derivation (§5.8)
3. Tier-2: adds Reverse-FE, Controlled-PUF, multi-round, ZK (§5.10)

Final paragraph clarifies that the verifier classifier (§5.4) operates
on the chip's wrapped protocol response, not raw on-chip measurements
(resolves §5.4 vs §5.10.2 contradiction).

### §5.4 CLARIFICATION (E-M4)
Block added at top of §5.4 explicitly stating "the classifier below is
the verifier-side classifier on the chip's protocol response… not on
raw on-chip physical measurements."

### §5.8 (E-M1)
"Honest bit-security claim (v2.1, Tier 1). O115 estimated the residual
ceiling at ≈ 2^30 – 2^40… ≈ 2^15 – 2^20…" rewritten as
"Empirical attack-cost (v2.1, Tier 1) — not a formal cryptographic
reduction. ≈ 10⁹–10¹² samples no-K_chip; ≈ 10⁴–10⁶ samples K_chip-leak."

### §5.10 INTRO (E-M1)
"raise the bit-security ceiling against a $10k attacker from
2^30–2^40 → 2^60–2^80 (no K_chip leak) and from 2^15–2^20 → 2^40–2^60
(K_chip leaked)" rewritten as empirical-operating-point language with
explicit "no formal cryptographic reduction" caveat.

### §5.10.3 (E-M1)
"raises the modeling-attack cost from 2^15–2^20 to 2^40–2^60" →
"raises the **empirical** modeling-attack cost from ≈10⁴–10⁶ to ≥10¹²"
with caveat.

### §5.10.5 (E-M1) — RENAMED + CAVEAT
- Subsection renamed: "Bit-security ceiling, post-Tier-2" → "Empirical attack-cost, post-Tier-2 (NOT formal bit-security)"
- Opening block-quote: "These are empirical operating points, not formal security proofs."
- Table column heading "Bits" → "Empirical attack-cost"
- Specific table cells rewritten in empirical-operating-point language
- V6 row updated: "(out of scope; requires hardware distance bounding)"
- Final headline rewritten in empirical language; concluding block-quote rewritten

### §6 (E-M2)
- "These are the three things FabricCrypt enables…" → "These are the three things FabricCrypt enables, **as a primitive at n=2 chassi**, that vendor-PKI attestation does not"
- §6.1: "generating a forged per-die attribution is now ≥ 2^60 ML-emulation calls" → "≥ 10⁴ modeling samples returning random-floor Hamming distance (no formal cryptographic reduction)"

### §8 (Patent disclaimer)
Two new paragraphs for Eckel, LAMINATOR + 1-line patent disclaimer
covering Suh-Devadas and Van Herrewege patents under academic fair use.

### §9 CONCLUSION (E-M1, E-M2, E-M3, E-M5)
- "thirteen HAL-bypass micro-architectural signals" → "15 signals (5 HAL-bypass μ-arch + 3 cross-host KS-verified μ-arch + 7 board-level deterministic)"
- "raises the bit-security ceiling from 2^30–2^40 to 2^60–2^80" → "raises the **empirical attack-cost** against ML modeling attackers to the random Hamming floor"
- "We have shown this at n=2. We have not shown: …personality-emergence pre-reg PASS (NULL on 0.75 gate, observed 0.664…)" → removed Phase 21b from the conclusion's not-shown list (now in §7.L6 only); added "A formal cryptographic reduction for the empirical ceilings (§5.10.5)" instead.
- Added "Additional supplementary detail on stylometric divergence from chip-conditioned training is provided in §7.L6 — this is exploratory and not a headline claim."

### BIBLIOGRAPHY
Two new entries added after [Brands1993]:
- [EckelFenzlJaeger2024]: IFIP SEC 2024
- [LAMINATOR2025]: CODASPY 2025

## Word count

- v3:   9,686 words
- v3.1: ~10,309 words (~+6% from required additions for §5.0, §5.4 clarification, 2 citations, patent disclaimer, abstract rewrite)

Target was 9,500–10,000. v3.1 is slightly above target ceiling (~309
words over 10,000). All overage is from REQUIRED safety additions
(§5.0 protocol-evolution, §5.4 clarification, patent disclaimer, 2 new
citations, "empirical operating point" caveats). No further trimming
is desirable without losing audit value.

## Preservation

- `paper_drafts/fabriccrypt_v3.md` is UNTOUCHED.
- All edits are auditable through this log.

— END EDIT LOG —
