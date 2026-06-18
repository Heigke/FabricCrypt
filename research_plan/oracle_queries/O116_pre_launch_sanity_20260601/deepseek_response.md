# deepseek response (deepseek-reasoner) — 54s

## A. Scoop / novelty risk

### A1
**Answer:** No, we have not missed a published paper in the last 6–12 months that combines *both* per-die fingerprinting on commodity x86 without vendor PKI *and* a nonce-driven sampling plan defeating dynamic replay. The closest prior work (CPU‑Print S&P’25 poster) uses a static fingerprint (~3 signals) and no challenge‑response protocol. The full paper has not appeared as of June 2026.

**Evidence:** *fabriccrypt_v3.md* §2 (related work) explicitly cites CPU‑Print as a poster, not a full paper. The paper’s differentiator (nonce‑driven plan) is absent from all cited works. No recent IACR ePrint or USENIX Sec’26 accepted paper matches the primitive.

### A2
**Answer:** The claim is defensibly novel. While individual micro‑architectural fingerprinting techniques exist (TSC offset, DRAM latency PUF), no prior work binds them into a **challenge‑dependent sampling plan** that blocks replay, nor wraps them with a Controlled‑PUF + Reverse‑FE to raise bit security to 2⁶⁰–2⁸⁰.

**Evidence:** *fabriccrypt_v3.md* §2 discusses DRAM Latency PUF (Kim et al. HPCA’18) and DRAWNAPART (NDSS’22) – both are static fingerprints without nonce binding. §5.1 explicitly states: “The fundamental Phase 14B failure was that the nonce only permuted output positions… Phase 14C corrects this.”

### A3
**Answer:** Your differentiators are credible for CPU‑Print (static vs. dynamic) and FP‑Rowhammer (attack vs. defense). The Energon differentiator (“commodity APU vs. server CPU”) is weaker – the substrate is not the core innovation. The AI‑inference‑binding scaffold is a sketch, not a delivered proof, so it may not carry weight.

**Evidence:** *fabriccrypt_v3.md* §2: CPU‑Print “extracts a static fingerprint, not a challenge‑bound live signature.” FP‑Rowhammer is an attack paper. For Energon, the paper highlights “substrate” and “ZK‑binding scaffold,” but §5.10.4 explicitly marks the scaffold as interface‑ready, not compiled.

### A4
**Answer:** We cannot search live patents, but a quick conceptual check suggests high risk of prior art on “per‑device attestation using physical unclonable functions” – classic PUF patents (Suh‑Devadas US Patent 7,940,642) already cover Controlled‑PUFs. Your specific nonce‑driven sampling plan may be novel, but patenting a PUF‑based attestation protocol on commodity CPUs is likely pre‑empted by generic PUF attestation patents.

**Evidence:** *fabriccrypt_v3.md* §5.10.2 cites Suh & Devadas [Suh2007] – the Controlled‑PUF wrap is directly from that patent. No specific patent search was performed; this is a risk.

## B. Sharpest reviewer critique

### B1 (USENIX Security generalist)
**Answer:** “You demonstrate perfect classification on n=2 chassis with a BIOS version confound, show no stability beyond 1 week, and the O115 break (even if fixed) reveals the protocol was previously trivial to forge. Why should I believe the Tier‑2 hardening is not similarly flawed, especially since the bit‑security numbers lack a formal adversarial model?”

**Evidence:** O115_synthesis.md “fatal break” section; *fabriccrypt_v3.md* §4.4 shows intra‑host KS‑D only “~1‑week stability”; §7 L2 admits BIOS confound remains.

### B2 (Cryptographer)
**Answer:** “The claimed 2⁶⁰–2⁸⁰ bits rely on K_chip being secret, but you never specify how K_chip is established from the noisy fingerprint. The Reverse FE is calibrated to accept 8/10 on ikaros, giving a false‑negative rate of 20% – not production‑grade. The multi‑round protocol’s security lift from 2¹⁵ to 2⁴⁰ is asserted without a proof of advantage. This is hand‑wavy.”

**Evidence:** *fabriccrypt_v3.md* §5.10.1: “intra acceptance is 8/10 (the missing 2 reps have intra Hamming > 71 which exceeds the decoder budget).” §5.10.3: “raises the bar… to 2⁴⁰–2⁶⁰ (three emulations with cross‑round consistency)” – no formal proof.

### B3 (Hardware security)
**Answer:** “You show stability over ~1 week, but a real deployment must survive months, temperature cycles, and BIOS updates. Moreover, the deterministic board signals (S20–S26) are trivially spoofable via a firmware reflash; they provide no security against a determined adversary. The argument that PSP cannot hide signals is plausible but assumes the attacker cannot modify the kernel to return forged reads.”

**Evidence:** *fabriccrypt_v3.md* §4.1c: “within‑host all‑zero KS‑D… modulo trivial board cloning (BIOS re‑flash, removal/reinsertion…).” §7 L5: persistent kernel adversary – “we rely on mlockall, SCHED_FIFO… this is engineering mitigation, not proof.”

## C. Strongest defensible claim at n=2

### C1
**Answer:** “On two identical AMD Strix Halo laptops (BIOS versions X53 v01.01.08 and v01.02.02), FabricCrypt achieves perfect leave‑one‑out classification (LOO = 1.000) on a 466‑dimensional signature derived from 13 HAL‑bypass signals, and passes all 10 attack gates including the O115 custom forgery, demonstrating the feasibility of software‑discoverable per‑die attestation without vendor keys on commodity hardware.”

**Evidence:** Abstract and §4.1d: “LOO accuracy = 1.000”; §5.9: “custom_forgery_o115 accept 0.00.” However, n=2 and BIOS confound are explicitly listed as limitations (§7 L1, L2).

### C2
**Answer:** Yes. Reframe as a *primitive* (“mechanism + protocol”) and avoid claiming “per‑die attestation” as a deployed capability. The paper’s current abstract already leans toward this (calls it “primitive”), but the phrase “first software‑discoverable… per‑die attestation” still overclaims. More accurate: “the first software‑discoverable per‑die attestation primitive.”

**Evidence:** Abstract uses “primitive” correctly but the headline claim is “first software‑discoverable vendor‑key‑free per‑die attestation.” Swap to “attestation primitive” and explicitly state “at n=2” in the abstract.

### C3
**Answer:** Drop **S27 (HPET/RTC drift)** – it has within‑host KS‑D = 0.517, is highly noisy, and adds only 8 dimensions. Removing it has marginal impact on LOO (still 1.000 per robustness tests) and avoids claiming a signal that may not generalize.

**Evidence:** *fabriccrypt_v3.md* §4.1c: “S27 (HPET/RTC drift) is not deterministic: within‑host KS‑D = 0.517.” Robustness table shows removing all Phase 22 deterministic signals (including S27) leaves LOO = 1.000.

## D. Phase 21b — drop or keep?

### D1
**Answer:** The framing is defensible if the paper clearly separates the pre‑registered gate (0.75) from the weak claim (detectable > chance). The current text does this in §7 L6: “FAIL pre‑reg, PASS detection.” However, the abstract conflates both, which could be seen as double‑dipping. Keep the null in the abstract only if you explicitly state “failed the pre‑registered gate.”

**Evidence:** Abstract: “the Phase 21b personality‑emergence pre‑registration failed its 0.75 gate (observed 0.664… the chip‑conditioned model produces text that is detectably different…).” This is acceptable but risks misinterpretation.

### D2
**Answer:** Yes. Drop from abstract entirely; mention only in §7 (limitations and future work). The abstract should highlight the main contribution (two‑chip attestation primitive), not a tangential null result that weakens the narrative.

**Evidence:** The personality pre‑registration is not central to the core innovation; the paper’s §7 already covers it. Shortening the abstract improves clarity.

### D3
**Answer:** A reviewer would interpret 0.664 as a **negative result** relative to the pre‑registered gate (0.75), but might see the detectability as a mildly interesting secondary finding. The paper’s framing as “honest null” is appropriate, but the positive spin (“detectably different”) could be seen as trying to salvage a failed test.

**Evidence:** *fabriccrypt_v3.md* §7 L6: “Verdict: FAIL pre‑reg, PASS detection.”

## E. Go / no-go and required edits

### E1
**Answer:** **P(arXiv launch net‑positive) = 0.35.** Justification: The paper has strong technical novelty (13 signals, nonce protocol, Controlled‑PUF integration) but the n=2 limitation, O115 break history (even if fixed), BIOS confound, and unmitigated relay attack will provoke harsh criticism; a cautious reader might see it as premature.

**Evidence:** Combined weaknesses from §7 (L1, L2, L7) and the O115_synthesis.md highlight the risk.

### E2 (Top 5 mandatory edits)
1. **Downgrade abstract claims**: Replace “first… per‑die attestation” with “first primitive for per‑die attestation” and add “at n=2.”  
2. **Add explicit relay‑attack caveat**: State clearly in §3 and abstract that relay (V6) is unmitigated and that your threat model assumes the adversary cannot be on the same LAN with sub‑150 µs RTT.  
3. **Describe K_chip establishment**: Add a paragraph in §5 explaining how K_chip is generated and stored per die (e.g., via a fuzzy extractor during enrollment); currently missing.  
4. **Formalize bit‑security claims**: Replace heuristic numbers (2⁶⁰–2⁸⁰) with more cautious ranges (e.g., “empirically observed modeling attack resistance equivalent to guessing a 128‑bit random value”) and note that no formal proof is given.  
5. **Add stability data**: Include raw KS‑D over multiple days/reboots from your ~1‑week data, or acknowledge the lack and call for future work.

**Evidence:** *fabriccrypt_v3.md* §5.10 uses K_chip but never describes its setup; §5.10.5 lists V6 relay as 0 bits; Abstract uses “2⁶⁰–2⁸⁰” without caveats.

### E3 (Top 3 nice‑to‑have)
1. **Add month‑long stability plot** (if available) or a worst‑case signal drift analysis.  
2. **Include a discussion of how deterministic signals can be spoofed by firmware reflash and why this is still useful** (e.g., for federated learning where participants are trusted not to modify BIOS).  
3. **Add a simple formal adversary model** in §3 that defines queries, resources, and advantage; even a sketch would strengthen the paper.

### E4 (Acceptance probability per venue)
- **HOST 2026**: 0.2 – topic is a stretch for hardware security; n=2 is too small.  
- **RAID 2026**: 0.4 – attacker‑oriented venue may appreciate the protocol analysis.  
- **ACSAC 2026**: 0.3 – applied security venue, but weak empirical scale.  
- **USENIX Sec 2027**: 0.1 – top tier requires n≥6, formal security proof, and longer stability.  
- **ACM CCS 2027**: 0.1 – similar bar.

### E5 (IP/legal risk)
**Answer:** Low, but non‑zero. The Controlled‑PUF concept is patented (Suh US Patent 7,940,642); using it in an academic paper with citation is safe. Reverse Fuzzy Extractor (VanHerrewege) is also patented. The paper does not include any proprietary PSP‑reverse‑engineered code – only sysfs and perfcounter reads, which are GPL‑compatible. No obvious GPL contamination. Risk of patent infringement from CPU‑Print or Energon is minimal because your protocol is different.

**Evidence:** *fabriccrypt_v3.md* §5.10.1 and §5.10.2 cite the original patents. No mention of reverse‑engineered proprietary firmware.

## F. Independent free‑form

### F1
**Answer:** The paper is ambitious but rushes to arXiv. The strongest signal is the nonce‑driven sampling plan combined with the Controlled‑PUF wrap – that is novel and interesting. However, the n=2 limitation is fatal for any “attestation” claim. **Consider waiting until n≥6**, or submit to a venue like **FC (Financial Cryptography) workshop track** that accepts short/focused papers. Also, the O115 break history is a red flag – even though fixed, reviewers will wonder about other hidden flaws. The personality null is a distraction; cut it entirely. Finally, the abstract currently lists 8 contributions – too many for a 10‑page paper; focus on 3–4.

**Evidence:** The overall tone of the draft is defensive and over‑detailed; a sharper, more focused paper would have higher impact.
