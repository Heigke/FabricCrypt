# gemini response (gemini-2.5-pro) — 72s

Excellent. This is a well-structured request for a pre-submission audit. My stance will be that of a critical but constructive USENIX Security program committee member.

---

### A. Scoop / novelty risk

**A1. Given the paper draft, has anyone published the *same* primitive in the last 6–12 months that we missed?**

No, the *combination* of these three elements appears novel. While per-die fingerprinting and replay defense are studied topics, the use of a nonce to drive the *physical sampling plan itself* (which cores, which thermal zones, etc.) is a strong and, to my knowledge, unpublished contribution for commodity x86 hardware. This moves beyond a simple challenge-response PUF to a "challenge-instructed measurement" protocol.

*Evidence:* The paper's related work section (2 and 8) correctly identifies the major adjacent work (DRAM Latency PUF, DRAWNAPART). The key differentiator is in Section 5.1, "Nonce-driven sampling plan," which is a significant step up from the static fingerprints or simple challenge-response pairs in the cited literature.

**A2. Is "per-die attestation without vendor PKI on commodity AMD" actually novel as we claim?**

Yes, this specific claim appears to be novel and significant. The dominant industry trend is *more* reliance on vendor-PKI-rooted attestation (PCC, CC, TDX, SEV-SNP). Actively building a high-resolution, per-die alternative that sidesteps this entire trust model is a contrarian and valuable research direction. The novelty holds as of mid-2026 based on the provided literature.

*Evidence:* The introduction (Section 1) correctly frames the problem against the vendor-PKI-rooted designs. The lack of citations for a direct competitor in this space is telling.

**A3. Are these differentiators credible, or are we kidding ourselves?**

The differentiators are credible and well-articulated.
- **CPU-Print:** Differentiating on signal count (13 vs. ~3) is a quantitative improvement, but the protocol-level defense (nonce-driven sampling plan vs. static fingerprint) is a qualitative, architectural advantage. This is a strong differentiator.
- **Energon:** The substrate (commodity APU vs. server CPU) is a scope difference, but the AI-inference binding is a conceptual contribution that Energon does not appear to make. This is a credible differentiator.
- **FP-Rowhammer:** The distinction between an offensive side-channel and a defensive attestation primitive is fundamental. This is a very clear and credible differentiator.

*Evidence:* The paper's own framing in Section 2 and 8 aligns with this analysis. The core contribution is the *system* and *protocol*, which is distinct from the point-solutions or different goals of the cited works.

**A4. Patent landscape (USPTO + Google Patents 2025–2026) for "chip-bound AI", "per-die machine-learning attestation". Anyone filed?**

I cannot perform a live, comprehensive patent search. However, the "PUF patent thicket" is notoriously dense, with major players (Intel, NXP, Rambus, etc.) holding broad patents on using physical variations for key generation. The novelty here would likely hinge on the specifics of the HAL-bypass signals and the nonce-driven sampling plan. It is plausible that no single patent covers this exact combination, but a commercial product would face significant FTO (Freedom to Operate) analysis. For an academic paper, this is not a blocker.

*Evidence:* This is based on general knowledge of the hardware security IP landscape, not specific text in the bundle.

---

### B. Sharpest reviewer critique

**B1. What's the sharpest critique a USENIX Security 2027 reviewer would write in 5 minutes after reading the abstract and §4?**

"**N=2 is not a security evaluation; it's a case study.** The authors demonstrate perfect separation between two specific laptops, one with a different BIOS version and default power governor. There is no evidence this generalizes to a larger population of chips from the same wafer, lot, or even SKU. The 100% LOO claim is statistically meaningless at this scale and may simply be fingerprinting the BIOS/governor difference, not the die."

*Evidence:* The paper honestly discloses this in Section 7, L1 ("Chassis count n=2") and L2 ("BIOS version confound"). The reviewer will weaponize the authors' own honesty against them.

**B2. What's the sharpest critique a *cryptographer* reviewer would write?**

"The 60-80 bit security claim is unsubstantiated. The authors derive ~63 bits of entropy from the *plan space* (Section 5.6) but then assert a 60-80 bit security ceiling without a formal model. Security bits should be derived from the hardness of a well-defined problem (e.g., distinguishing the PUF output from random), not the input cardinality of a heuristic sampling function. The Tier-2 hardening is a collection of best-practice PUF defenses, but their composition is not proven secure."

*Evidence:* Section 5.10.5 presents a table of bit-security claims. The jump from the v2.1 estimate (30-40 bits) to the v3 estimate (60-80 bits) is asserted, not derived from a formal analysis. A cryptographer would demand a proof or a much more rigorous argument.

**B3. What's the sharpest critique a *hardware security* reviewer would write?**

"The paper lacks any serious stability analysis. The signals are assumed to be stable enough for a Reverse Fuzzy Extractor to work, but there is no data on how these 13 HAL-bypass signals drift with temperature, voltage, and silicon aging over months or years. A one-week evaluation is insufficient; without long-term stability data, the entire premise of a reliable per-die identity is questionable. Many of these signals (e.g., syscall tails, thermal spread) are likely to be highly sensitive to environmental and software changes (kernel updates, workload)."

*Evidence:* The paper does not present a long-term stability study. The RFE analysis in 5.10.1 shows 8/10 intra-host acceptance, implying two signatures were already too noisy to be corrected even in a short-term test. This hints at underlying stability problems a hardware reviewer would pounce on.

---

### C. Strongest defensible claim at n=2

**C1. With n=2 chips, what is the strongest single-sentence claim we can defend in the abstract?**

"We present FabricCrypt, a novel primitive for vendor-key-free per-die attestation, and demonstrate on two commodity AMD APUs that its 466-dimensional signature, generated via a nonce-driven sampling plan, achieves perfect classification and resists a 10-gate extended attack battery."

*Justification:* This frames the work as a *demonstration* of a *primitive* on a specific (n=2) testbed, which is factually accurate and avoids overclaiming generalizability.

**C2. Should we re-frame the paper as a *primitive* (mechanism + protocol) rather than a *capability* (per-die attestation), to dodge the "n=2 → can't claim biometric-grade ID" critique?**

Yes, absolutely. This is the single most important strategic change to make. The novelty is in the *mechanism* (the signal bundle + the nonce-driven sampling). The *capability* (reliable per-die ID at scale) is not proven by this work. Emphasizing the former makes the paper much more defensible.

*Evidence:* The current abstract leads with "the first... per-die attestation primitive," which is good, but the body of the paper should consistently reinforce that the contribution is the *design and proof-of-concept implementation* of this primitive.

**C3. If we had to drop one of the 13 signals to be honest about generalisation, which would we drop and why?**

**Syscall p99.9 tails (Task D).** This signal family is the most susceptible to software configuration changes (kernel scheduler, interrupt handling, background processes, BIOS SMI handler implementation). While it shows good separation between `ikaros` and `daedalus`, it is the least likely to be a pure, stable physical-die property and the most likely to be a "configuration fingerprint." Dropping it would strengthen the claim that the remaining signals are closer to the silicon.

*Evidence:* Section 4.1, item (4) describes the signal as governed by "the kernel scheduler interacting with hardware... and SMI / MCE handlers in PSP." This explicitly links it to mutable software/firmware components.

---

### D. Phase 21b — drop or keep?

**D1. Is this framing defensible, or are we double-dipping?**

The framing is not only defensible, it is a model of scientific integrity. Reporting a pre-registered null result, especially with a nuanced analysis (p << 0.001 vs. chance but below the gate), is a sign of a mature research project. It is not double-dipping; it is an honest and complete account of the experimental results.

*Evidence:* Section 7, L6 ("Personality-emergence pre-registration: NULL") is clear and direct. The detailed breakdown of the stylistic differences demonstrates that the authors did the work to understand the result, rather than just hiding it.

**D2. Should we drop the personality framing entirely from the abstract and only mention it in §7 (limitations)?**

No. Keep it in the abstract. It is a mark of distinction that will earn trust with sharp reviewers. It signals that the authors are not just "selling" a result and makes the positive claims in the paper *more* credible by association. It's a very strong signal of honesty.

*Evidence:* The abstract currently includes: "We are honest about what we have *not* shown: (a) the Phase 21b personality-emergence pre-registration **failed its 0.75 gate**..." This is perfect. Do not change it.

**D3. Would a reviewer interpret "personality emergence detectable at 66.4%" as a positive or a negative result given the preregistered gate?**

A good reviewer will interpret it as a **net positive for the paper's quality**, even though it is a negative result for the specific hypothesis. It demonstrates rigor, honesty, and a commitment to the scientific process. It transforms a failed experiment into a successful piece of scientific reporting.

*Evidence:* The detailed analysis in Section 7, L6, showing *what* was detected (length, entropy, punctuation) makes the result interesting in its own right, even if it failed the primary gate.

---

### E. Go / no-go and required edits

**E1. Quantify: P(arXiv launch is net-positive for our reputation) given the current draft.**

**P = 0.95.** The work is technically deep, the core idea is novel, and the paper is written with exceptional transparency and honesty about its limitations (O115, n=2, NULL result). This level of rigor is respected even if reviewers disagree on the ultimate impact. The risk of embarrassment is very low.

**E2. List the top 5 mandatory edits to v3 before arXiv submission.**

1.  **Re-frame for n=2:** Systematically go through the abstract, intro, and conclusion to frame the work as the design and *proof-of-concept demonstration* of a novel *primitive*, not as a conclusive, generalizable *system* for per-die ID. Replace phrases like "we obtain" with "we demonstrate on our n=2 testbed."
2.  **Justify Bit-Security Claim:** In Section 5.10.5, add a paragraph explicitly breaking down the 60-80 bit security claim. Acknowledge it's a heuristic estimate and explain the components: plan entropy, underlying physical entropy of the signal response, and the cryptographic strength of the Controlled-PUF wrapper. Without this, it looks like a number pulled from thin air.
3.  **Add Long-Term Stability to Limitations:** Add a new limitation (e.g., L10) explicitly stating that long-term stability (months, across major thermal cycles and BIOS/kernel updates) has not been studied and is a critical item for future work to assess production viability.
4.  **Strengthen Threat Model:** The threat model (Section 3) should explicitly mention the BIOS/software configuration confound. State that the adversary is assumed to be running on a machine with the *same* software configuration, or that the signals used are robust to minor configuration differences (which is a claim that needs to be backed up).
5.  **Clarify ZK Scaffold Status:** In the abstract and contributions, use the word "scaffold" or "design" for the ZK binding. The current text is good, but ensure no part of the paper could be misread as claiming a fully implemented and benchmarked ZK system.

**E3. List the top 3 nice-to-have edits.**

1.  **Add a Signal Stability Plot:** Even if it's only for one week, a plot showing the intra-host variation (e.g., Mahalanobis distance from the mean) for `ikaros` over time would add significant value and preempt some of the stability questions.
2.  **Shorten Section 4:** The per-signal descriptions are detailed but could be tightened. The core mechanism is clear, and some of the finer physical details could be condensed to improve readability.
3.  **Adversary C (Side-Channel) Nuance:** In Section 5.11, add a sentence acknowledging that while FabricCrypt doesn't defend against reconstruction, the high dimensionality and multi-modal nature of the signature would make real-time, high-fidelity reconstruction from side-channels a formidable challenge in itself.

**E4. Estimated probability of *first-cycle* reviewer acceptance, by venue:**

-   **HOST 2026:** 60-70%. Perfect topical fit. The hardware audience will appreciate the signals, but will also be the sharpest critics of the `n=2` and stability limitations.
-   **RAID 2026 / ACSAC 2026:** 40-50%. Good systems security venues. They will appreciate the protocol and attack analysis. The `n=2` issue will be a major point of discussion.
-   **USENIX Security 2027:** 20-30%. The bar is extremely high, and `n=2` is often a fatal flaw for a systems paper unless the core idea is truly groundbreaking. The honesty and rigor give it a fighting chance.
-   **ACM CCS 2027:** 15-25%. Similar to USENIX, but with a potentially higher concentration of cryptographers who will attack the lack of formal proofs for the security claims.

**E5. Any IP / legal risk?**

I cannot provide legal advice. The paper appears to follow academic best practices by extensively citing prior art. The risk of academic misconduct is near zero. The risk of building a product based on this without a patent search is high, but that is out of scope for an arXiv submission. The use of HAL-bypass techniques is standard in security research; GPL contamination is unlikely unless you are distributing modified kernel code (which doesn't seem to be the case).

---

### F. Independent free-form

**F1. Anything else we should know before we hit "submit" on arXiv.**

The single greatest strength of this paper is its transparent and rigorous self-assessment. The story of finding and fixing the O115 flaw, and the honest reporting of the Phase 21b NULL result, should be leaned into. It builds a narrative of a careful, honest research project. This narrative is your best defense against the primary weakness, which is `n=2`.

Do not try to hide or downplay the `n=2` limitation. Instead, frame it as the necessary first step: "Before scaling to N>6, we first had to establish a viable primitive and protocol that could survive a rigorous, multi-stage adversarial audit. This paper presents that primitive and its audit; the N>6 study is the necessary and immediate next step."

This is a strong paper with a novel core idea. The execution appears solid, and the reporting is exemplary. Submit it.
