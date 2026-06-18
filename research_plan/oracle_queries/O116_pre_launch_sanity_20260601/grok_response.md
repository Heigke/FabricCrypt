# grok response (grok-4-latest) — 16s

**A1.** No identical primitive (software-only per-die attestation on commodity x86 without vendor PKI, ≥10 HAL-bypass signals, nonce-driven sampling plan) appears in the last 12 months. The bundled draft (fabriccrypt_v3.md:§2) correctly positions itself against DRAM Latency PUF (HPCA 2018), DRAWNAPART (NDSS 2022), and FP-Rowhammer (AsiaCCS 2025) on the live/challenge-bound axis; none of those combine 13 signals with an audience nonce that re-derives the sampling plan.

**A2.** The core claim (“per-die attestation without vendor PKI on commodity AMD”) is still novel. CPU-Print S&P’25 poster never produced a full paper; Energon (Aug 2025) targets server CPUs and power leakage rather than attestation; no 2026 IACR ePrint or USENIX Sec’26 paper in the bundle contradicts the positioning.

**A3.** The stated differentiators are credible on paper. CPU-Print (~3 signals, static) is weaker on count and replay defence; Energon lacks the ZK-binding scaffold and runs on different substrate; FP-Rowhammer is an attack, not a defence. The draft (fabriccrypt_v3.md:§2, lines 112–128) explicitly notes these distinctions.

**A4.** No relevant USPTO/Google Patents 2025–2026 hits for “chip-bound AI” or “per-die ML attestation” are referenced in the bundle; the authors appear to have performed no patent search.

**B1.** Sharpest 5-minute reviewer critique: “n=2 chassis, 1-week stability at best, and a 0-bit forgery break fixed only after an internal oracle audit—insufficient evidence that the 466-dim signature is either stable or unforgeable at production scale.”

**B2.** A cryptographer would write: “The 2^60–2^80 claim after Tier-2 (Reverse FE + Controlled-PUF) is hand-wavy; the paper never shows a formal reduction or even a concrete attack cost model beyond Hamming-distance tables (fabriccrypt_v3.md:§5.10.2, Table on p. 38).”

**B3.** A hardware reviewer would write: “No data on thermal cycling, BIOS updates, or >1-week drift; the 13 signals may be governor- or firmware-version artifacts (explicitly flagged as L2 in §7).”

**C1.** Strongest defensible abstract sentence at n=2: “On two Strix Halo APUs we demonstrate a 466-dimensional live signature assembled from 13 HAL-bypass signals that yields 100% LOO separation and survives a 10-attack protocol battery after O115-driven hardening.”

**C2.** Yes—re-framing as a “primitive + protocol” rather than a finished “per-die attestation capability” materially reduces the n=2 critique.

**C3.** Drop S27 (HPET/RTC drift). It is the only Phase-22 signal with non-zero intra-host KS-D (0.517) and adds the least entropy while being the most likely to be BIOS- or timer-driven.

**D1.** The framing is defensible but borderline double-dipping: the paper correctly reports a NULL on the pre-registered 0.75 gate while claiming a weak positive on detectability (p≪0.001). This is honest but will be read as spin by some reviewers.

**D2.** Yes—move the entire personality result to §7 (limitations) and remove it from the abstract; the 0.664 result adds no security or attestation claim.

**D3.** Reviewers will read “66.4% vs 0.75 gate” as a negative result; the honest-NULL language is the correct framing.

**E1.** P(arXiv launch is net-positive) ≈ 0.35. The O115 break was caught and patched, the NULL is reported, and the novelty claim holds, but n=2 + missing stability + 60–80-bit hand-waving still risks visible reviewer takedowns on social media and in later reviews.

**E2.** Top 5 mandatory edits (priority order):
1. Add explicit “n=2, 1-week stability only” sentence to abstract and §1.
2. Replace every “2^60–2^80” claim with “preliminary empirical bound pending formal analysis and n≥6 validation.”
3. Move Phase 21b result out of abstract into §7 only.
4. Add a one-paragraph “Adversary D: hardware emulator / local relay” subsection in §5.11.
5. Insert a short “Patent search: none found” statement or strike the implicit novelty language.

**E3.** Nice-to-haves: (i) re-run LOO after dropping S27; (ii) add a 3-month drift table if any data exist; (iii) publish the raw 10-rep signature matrices.

**E4.** First-cycle acceptance probabilities (current draft): HOST 2026 ~0.15, RAID 2026 ~0.10, ACSAC 2026 ~0.20, USENIX Security 2027 ~0.05, ACM CCS 2027 ~0.08.

**E5.** Low but non-zero IP risk: no obvious infringement on cited prior art, but the Controlled-PUF + Reverse-FE combination sits inside existing PUF patents; GPL contamination from any PSP reverse-engineering is not addressed.

**F1.** The v3 draft is materially stronger than v2 because it actually incorporated the O115 findings, but the remaining gaps (n=2, stability, relay, bit-security narrative) are exactly the ones that produce “reject + blog post” outcomes at top venues. Launching now is a calculated reputation gamble, not a safe default.
