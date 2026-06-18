# Capability Brainstorm: What Hardware-Bound AI Identity ENABLES
**Date:** 2026-06-01
**Context:** Reframe away from "static accuracy gain." Find demo/paper-worthy capabilities unlocked by 5-signal per-chip identity + nonce-protocol live signature.

## Our delivered primitives (recap)
- P1: 5-signal per-chip identity, 100% LOO classification
- P2: Phase 14C nonce-protocol live signature (0.6% replay acceptance)
- P3: Cross-host transplant rejection (2%)
- P4: Sub-ms overhead per sampling
- P5: Identity survives normal workload thermal/voltage range (demonstrated)
- P6: Identity changes under chip modification (tamper-evidence inherent)

Anything below should compose from {P1..P6}.

---

## 1. Full Brainstorm (40 capabilities)

Format: **#N Title** — claim / use case / closest existing solution.

### A. Cryptographic & attestation
1. **Liveness-attested inference** — every model output ships with a nonce-bound chip signature so a verifier knows the inference happened *on that chip, just now*, not replayed / proxied. Use: API-level "proof of compute origin" for paid inference. Closest: Intel SGX remote attestation (needs vendor PKI), TPM quote (no liveness against replay across reboot).
2. **PUF-free hardware identity for commodity GPUs** — gives a stable chip ID on consumer parts without dedicated PUF/TPM block. Use: AMD/NV consumer chips become attestable. Closest: SRAM PUFs (need fab support), Intel PTT/fTPM.
3. **Chip-rooted ZK proof of inference** — generate ZK proof that "model M ran on chip C at time T" without revealing inputs. Use: confidential AI auditing. Closest: zkML (no hardware binding), TEE attestation (vendor-trusted).
4. **Side-channel-bound MAC** — use natural noise signature as keying material for a chip-bound MAC, so signed messages prove origin even if private keys are exfiltrated. Use: hijack-resistant model serving. Closest: HSM-bound keys.
5. **Tamper-evident AI** — chip modification (thermal pad, repaste, mod-BIOS) shifts signature → AI refuses to run / raises alarm. Use: edge appliances in adversarial environments. Closest: tamper-evident seals (physical), TPM PCR seal (firmware-only).
6. **Decentralized AI authentication (no PKI)** — peers verify each other by chip-signature reproducibility under shared challenge, no CA needed. Use: P2P AI networks, edge mesh. Closest: web-of-trust, blockchain identity (no hardware root).
7. **Sybil resistance for federated learning** — clients must prove distinct chips, so attacker spinning 1000 VMs can't dilute aggregation. Use: FL on consumer devices. Closest: device attestation via Play Integrity / iOS DeviceCheck (vendor-mediated).
8. **AI agent identity at protocol layer** — multi-agent system: each agent's messages carry a HW-bound signature, defeating spoofing/cloning of agents. Use: MCP-style agentic protocols, agent marketplaces. Closest: software keys (clonable).
9. **Non-clonable model artifact** — model weights encrypted to chip's noise signature so a stolen weight file is dead on any other chip. Use: anti-piracy for proprietary models. Closest: DRM tied to TPM, Apple Secure Enclave keybag.
10. **Chip-rooted differential privacy auditor** — DP noise can be audited as actually generated on the claimed chip (not zero-noise fake). Use: regulatory DP compliance. Closest: trust-the-vendor.

### B. Economic / business model
11. **Per-chip licensing for AI models** — license fee per attested chip, no metering server needed; chip can't lie about being 1000 chips. Use: foundation-model B2B licensing. Closest: floating licenses w/ phone-home.
12. **AI inheritance / asset transfer** — model ownership = ownership of the chip token, transferable as a physical/legal asset. Use: AI estate planning, AI as collateral. Closest: NFT (no physical binding).
13. **AI insurance with provable lineage** — insurer can verify the deployed AI is the audited one, not a swapped copy. Use: liability insurance for medical/legal AI. Closest: self-attestation contracts.
14. **Compute-provenance carbon accounting** — every inference proves the chip & its energy class; aggregators verify renewable claims at chip granularity. Use: green-AI markets. Closest: utility-level RECs.
15. **Hardware-bound model marketplace** — buyer pays, model is sealed to their chip in one shot, no leakage even from buyer. Use: high-value vertical models. Closest: encrypted distribution + license server.

### C. Privacy
16. **Geofenced AI** — model only runs where chip thermal/EM environment matches a pre-baselined site (e.g., specific datacenter rack). Use: jurisdiction-restricted AI (EU-only inference). Closest: GeoIP (trivially spoofed).
17. **Time-locked AI lifecycle** — model decryption keys derived from chip-signature drift over months; model auto-expires as silicon ages. Use: planned-obsolescence regulation, time-bombed weapons-grade models. Closest: timestamped certs.
18. **AI that forgets on transplant** — moving the SSD/weights to another chip yields signature mismatch → keys un-derivable → model dies. Use: device theft mitigation. Closest: FDE bound to TPM (Bitlocker).
19. **Hardware-rooted Private Cloud Compute** — Apple-style PCC but on commodity AMD GPUs without vendor enclave. Use: confidential cloud inference. Closest: Apple PCC, AMD SEV-SNP (vendor-trusted, no PUF-grade identity).
20. **Tracking-resistant personal AI** — chip-bound AI proves *it is yours* without revealing who you are (ZK over chip-id). Use: anti-stalkerware identity proofs. Closest: anonymous credentials (no hardware liveness).

### D. Trust / governance
21. **"Prove this is the AI you trained"** — auditor challenges, chip+model respond with signature only the trained-on-this-chip model can produce. Use: regulator audits, model release verification. Closest: weight hash check (doesn't prove runtime identity).
22. **Regulatory audit trail** — every inference timestamp-signed by chip, building a non-repudiable log. Use: EU AI Act high-risk system logs. Closest: log signing with software keys (forgeable post-hoc).
23. **Legal personhood scaffolding for AI** — a continuous chip-rooted identity gives AI a stable referent for liability assignment ("this chip-AI did X"). Use: future AI-law. Closest: vague legal-entity wrappers.
24. **Adversarial honeypot mode** — chip-bound AI knows its true signature; when run in attacker's emulator, signature mismatch triggers honeypot output. Use: model exfil detection. Closest: canary tokens (passive).
25. **Per-chip personalized adversarial robustness** — adversarial perturbations crafted on attacker's chip don't transfer because input-conditioning depends on defender chip's noise. Use: model robustness in deployment. Closest: input randomization (no hardware root).

### E. Coordination / multi-agent
26. **HW-rooted MCP** — Model Context Protocol calls carry chip-attestation so an agent can't be MITMed or replaced. Use: agent ecosystems, A2A. Closest: bearer tokens.
27. **Distributed AI consensus without oracle** — N chip-bound AIs vote, sybil-resistant by P1, no trusted coordinator. Use: decentralized AI governance. Closest: BFT consensus (no HW binding to identity).
28. **AI-to-AI mutual authentication** — two chips perform a noise-signature handshake, agreeing they are both unique and live, before exchanging gradients/plans. Use: cross-org federated work. Closest: mTLS (clonable keys).
29. **Provable-uniqueness AI swarms** — a swarm of N agents proves all N chips are distinct (no fake replicas inflating votes). Use: drone/IoT swarms, oracle networks. Closest: enrollment-time attestation only.

### F. Hardware-specific & novel framings
30. **Energy-budget-aware inference contract** — model commits to ≤X joules/inference, verifiable from chip-signature energy class. Use: edge battery devices, sustainability SLA. Closest: power monitoring telemetry (untrusted).
31. **Chip-as-a-key for model-at-rest encryption** — weights on disk encrypted to a key derived live from chip noise, never written. Use: model-IP protection on stolen hardware. Closest: TPM-sealed keys.
32. **Wear-adaptive AI ("AI that grows with you")** — as chip ages, signature drifts; model adapts personalisation on this drift trajectory, making it un-portable to a fresh chip. Use: personal assistants. Closest: cloud-side personalization.
33. **AI as digital twin of host machine** — the AI's identity literally *is* the machine's silicon fingerprint; uniqueness of AI = uniqueness of host. Use: machine-identity unification (replace MAC/UUID). Closest: dmidecode UUIDs (forgeable).
34. **Chip-bound human ID** — your laptop's chip-AI becomes a human-controllable identity token (you authorize via prompt, it attests via chip). Use: replace WebAuthn for AI-mediated logins. Closest: passkeys (no AI mediation).
35. **Forensic post-incident attribution** — given an output blob, prove which chip emitted it (and roughly when), even after model retraining. Use: deepfake attribution, IP-leak forensics. Closest: invisible watermarking (no hardware tie).
36. **Anti-stalking AI personal agent** — chip-bound agent can prove identity to user but cannot be cloned to a surveillance copy. Use: protective tech for vulnerable users. Closest: signed apps (clonable).
37. **AI accountability handshake before action** — before any high-stakes action (financial, medical), AI must complete chip-attestation to a regulator endpoint. Use: AI-as-fiduciary regulation. Closest: rate-limited API keys.
38. **Live-replay-defeating bench/score** — public AI leaderboards require live chip-attestation per submission, killing "submit a pre-computed file" cheating. Use: integrity for MLPerf-like benches. Closest: hosted-eval (vendor-trusted).
39. **Hardware-identity-bound RNG / VRF** — verifiable randomness sourced from chip noise, with chip-identity proof. Use: lotteries, leader election. Closest: drand, Intel RDRAND (no chip-identity tie).
40. **Coupled-chip secret sharing** — split a secret across N chips so reconstruction requires all N live signatures, defeating cold-boot/extract attacks on any single host. Use: high-assurance HSM clusters from commodity GPUs. Closest: Shamir + HSMs.

---

## 2. Scoring (0–10 each axis; SCORE = N×D×P×C, max 10000)

| #  | Capability | Novelty | Demo | Paper | Commercial | SCORE |
|----|-----------|--------:|-----:|------:|-----------:|------:|
| 1  | Liveness-attested inference | 8 | 9 | 8 | 8 | **4608** |
| 2  | PUF-free identity commodity GPU | 9 | 7 | 9 | 7 | **3969** |
| 3  | Chip-rooted ZK inference | 8 | 5 | 9 | 7 | 2520 |
| 4  | Side-channel-bound MAC | 7 | 6 | 7 | 5 | 1470 |
| 5  | Tamper-evident AI | 7 | 9 | 6 | 7 | 2646 |
| 6  | Decentralized auth (no PKI) | 8 | 6 | 8 | 6 | 2304 |
| 7  | Sybil-resistant FL | 8 | 7 | 8 | 7 | **3136** |
| 8  | Agent identity (MCP) | 8 | 7 | 7 | 8 | **3136** |
| 9  | Non-clonable model artifact | 7 | 8 | 6 | 9 | 3024 |
| 10 | Chip-rooted DP auditor | 7 | 4 | 7 | 4 | 784 |
| 11 | Per-chip licensing | 6 | 7 | 5 | 9 | 1890 |
| 12 | AI inheritance | 7 | 6 | 4 | 5 | 840 |
| 13 | AI insurance lineage | 6 | 5 | 5 | 6 | 900 |
| 14 | Carbon accounting | 6 | 5 | 6 | 6 | 1080 |
| 15 | HW-bound marketplace | 6 | 7 | 5 | 8 | 1680 |
| 16 | Geofenced AI | 9 | 8 | 7 | 6 | **3024** |
| 17 | Time-locked lifecycle | 8 | 7 | 7 | 5 | 1960 |
| 18 | AI forgets on transplant | 8 | 9 | 7 | 7 | **3528** |
| 19 | HW-rooted PCC commodity | 7 | 6 | 8 | 8 | 2688 |
| 20 | Tracking-resistant personal AI | 7 | 5 | 6 | 5 | 1050 |
| 21 | "Prove this is the AI you trained" | 8 | 8 | 8 | 7 | **3584** |
| 22 | Regulatory audit trail | 6 | 6 | 6 | 8 | 1728 |
| 23 | Legal personhood scaffold | 6 | 4 | 6 | 3 | 432 |
| 24 | Adversarial honeypot | 8 | 8 | 7 | 5 | 2240 |
| 25 | Personalized adv robustness | 8 | 6 | 8 | 5 | 1920 |
| 26 | HW-rooted MCP | 8 | 7 | 7 | 8 | **3136** |
| 27 | Distributed consensus | 7 | 5 | 7 | 5 | 1225 |
| 28 | AI-to-AI mutual auth | 7 | 7 | 6 | 6 | 1764 |
| 29 | Provable-unique swarms | 8 | 6 | 7 | 6 | 2016 |
| 30 | Energy-budget contract | 8 | 8 | 8 | 6 | **3072** |
| 31 | Chip-as-key model encryption | 7 | 8 | 6 | 8 | 2688 |
| 32 | Wear-adaptive AI | 9 | 6 | 8 | 5 | 2160 |
| 33 | AI as digital twin of host | 7 | 6 | 6 | 5 | 1260 |
| 34 | Chip-bound human ID | 7 | 6 | 6 | 6 | 1512 |
| 35 | Forensic attribution | 7 | 7 | 7 | 6 | 2058 |
| 36 | Anti-stalking agent | 7 | 4 | 5 | 4 | 560 |
| 37 | Accountability handshake | 6 | 6 | 6 | 7 | 1512 |
| 38 | Replay-defeating leaderboard | 9 | 9 | 8 | 5 | **3240** |
| 39 | HW-identity VRF | 7 | 6 | 7 | 5 | 1470 |
| 40 | Coupled-chip secret sharing | 7 | 5 | 7 | 6 | 1470 |

**Top by score:** #1 (4608), #2 (3969), #21 (3584), #18 (3528), #38 (3240), #16 (3024), #30 (3072), #7/8/26 (3136), #9 (3024).

---

## 3. Top 5 + Demo Storyboards

### TOP-1 (#1) Liveness-attested inference — "Replay-Killer API"
**Why:** Directly built on Phase 14C nonce-protocol (0.6% replay). Single, sharp, generalizable.
**5-min demo storyboard:**
1. 0:00 Show two laptops side-by-side. Both have the same fine-tuned model weights.
2. 0:30 Verifier endpoint issues a fresh nonce. Laptop-A runs inference + emits {output, chip-signature(nonce)}. Verifier: ACCEPT.
3. 1:30 Attacker captures the (output, signature) pair, replays it from Laptop-B. Verifier: REJECT (signature was nonce-bound to Laptop-A's live noise).
4. 2:30 Attacker tries to *proxy*: Laptop-B forwards nonce to Laptop-A, gets signature, returns. Show the round-trip-time spike + signature still binds to A's identity, not B's claimed identity → REJECT.
5. 3:30 Show side panel: thermal/power/RNG signals captured live during inference, hashed with nonce. 0.6% replay rate vs. 100% with naive signatures.
6. 4:30 One-slide framing: "TLS for inference origin."

### TOP-2 (#2) PUF-free identity on commodity GPU — paper backbone
**Why:** This is the underlying scientific contribution. Everything else composes from it. Strong novelty: no fab support, no TPM, no SGX, just GPU+telemetry.
**5-min demo:**
1. 0:00 Fleet of 8 identical AMD gfx1151 laptops on a rack.
2. 0:30 Live signature sweep: 5 signals × 8 chips → heatmap, visually obvious clustering.
3. 1:30 100% LOO classification accuracy bar chart.
4. 2:30 Transplant: physically move SSD from chip A to chip B → classifier IDs the *chip* not the SSD (2% transplant accept).
5. 3:30 Stress: vary workload, ambient temp ±10°C → identity holds.
6. 4:30 Slide: "Hardware identity without dedicated silicon."

### TOP-3 (#21) "Prove this is the AI you trained"
**Why:** Hot regulatory topic (EU AI Act, model audits). Demo is intuitive. Composes 1+2+6.
**5-min demo:**
1. 0:00 Auditor receives model M from vendor with claim "trained and deployed on chip C, hash H."
2. 0:30 Auditor sends nonce-challenge. Chip C responds with live signature + model-output bound to nonce.
3. 1:30 Vendor swaps to a cheaper chip C' running the same weights → response fails verification.
4. 2:30 Vendor tampers model (knowledge distillation to smaller cheaper model) → signature still verifies (it's *chip* identity!) but a separate model-fingerprint check (kill-shot prompt) fails.
5. 3:30 Combined audit: pass iff (chip-id ∧ model-fingerprint) both verify.
6. 4:30 Frame: "Continuous deployment audit for EU AI Act Article 12."

### TOP-4 (#18) AI that forgets on transplant
**Why:** Visceral, single-screen demo. Phone/laptop theft narrative. Strong commercial.
**5-min demo:**
1. 0:00 Personalized AI assistant running on laptop, knows your calendar/notes (locally fine-tuned).
2. 0:30 Thief steals SSD + GPU as a unit and plugs into a different host (mainboard swap).
3. 1:30 Boot: chip noise signature drift exceeds threshold (cross-host = 2% accept).
4. 2:30 Model weights are sealed with key derived from chip signature → key un-derivable on new host → assistant boots in "amnesia mode."
5. 3:30 Restore original mainboard → assistant remembers everything.
6. 4:30 Frame: "Personal AI that can't be cloned by anyone, including its owner's attacker."

### TOP-5 (#38) Replay-defeating ML leaderboard
**Why:** Visual, immediately understandable to a broad audience, kills a known abuse (MLPerf/Kaggle pre-compute). Tight + novel.
**5-min demo:**
1. 0:00 Public leaderboard webpage; two submitters: honest + cheater.
2. 0:30 Server issues live nonce + test batch. Honest: chip on, runs, returns nonce-signed (output, signature) within deadline.
3. 1:30 Cheater pre-computed outputs offline; tries to replay → signature missing/invalid → SCORE=0.
4. 2:30 Cheater tries to outsource to a cluster: signature ties to *one* chip; submission rule = single chip → fails.
5. 3:30 Show audit log of each submission with chip-id.
6. 4:30 Frame: "MLPerf trust upgrade."

---

## 4. Critical Pruning

**Question:** which capabilities are *already* delivered by TPM 2.0 / Intel SGX / AMD SEV-SNP / Apple Secure Enclave **without measurable improvement from us**?

| # | Capability | Existing solution | Do we improve? | Verdict |
|---|-----------|-------------------|---------------|---------|
| 4 | Side-channel-bound MAC | HSM/SE-bound keys | Marginally (extraction-resistance) | **Weak** — drop unless we show key-extraction resistance experiment |
| 9 | Non-clonable model | TPM-sealed DRM | Yes IF we run on chips without TPM2 attestation chain (most consumer GPUs); but PC TPMs exist | **Keep but caveat**: our angle = no vendor key infra |
| 11 | Per-chip licensing | FlexLM + TPM | Only if no TPM | **Marginal** |
| 12 | AI inheritance | NFTs / contractual | We bind it to silicon | **Niche, drop from top** |
| 13 | AI insurance | Self-attest | Marginal (still need legal) | **Drop** |
| 17 | Time-locked lifecycle | Cert expiry + TPM | We use *aging drift* as clock — novel angle, keep | **Keep** (novel mechanism) |
| 19 | HW-rooted PCC | Apple PCC, SEV-SNP | Yes — works on non-Apple consumer GPUs | **Keep**, paper-worthy contrast |
| 20 | Tracking-resistant personal AI | Anonymous creds | Marginal | **Drop** |
| 22 | Regulatory audit trail | Signed logs + TPM | Live liveness is the diff | **Subsumed by #1** |
| 23 | Legal personhood | n/a | Speculative | **Drop** |
| 26 | HW-rooted MCP | TLS+mTLS | Live-binding vs. clonable key — meaningful | **Keep**, subsumes #8/28 |
| 31 | Chip-as-key encryption | TPM seal | We provide live noise (not just sealed key) → resists cold-boot of TPM | **Keep**, frame as cold-boot resistance |
| 34 | Chip-bound human ID | WebAuthn / passkeys | Passkeys already excellent | **Drop** |
| 36 | Anti-stalking | OS-level | Marginal | **Drop** |
| 37 | Accountability handshake | Signed audit | Subsumed by #1 | **Drop** |

**Capabilities surviving pruning that are genuinely beyond TPM/SGX/SE territory:**
- The "live + nonce-bound + non-clonable + works on commodity silicon without vendor key infra" stack: **#1, #2, #18, #21, #38**.
- Mechanism-novel angles: **#16 geofence (uses environment, not just chip)**, **#17 aging-drift clock**, **#32 wear-adaptive (novel mechanism, but demo-fragile)**.
- Underrated: **#30 energy-budget contract** — we already have E3 data; cleanest "we did this and nobody else can" claim.

**TPM 2.0 / SGX kill-list (drop):** #4, #11, #12, #13, #20, #22, #23, #34, #36, #37.

---

## 5. Final Shortlist (3) + Recommendation

### Final-1: **#1 Liveness-attested inference** ("TLS for inference origin")
- Direct reuse of Phase 14C result (0.6% replay) → demo is *already built*.
- Single sentence pitch lands with cryptographers, policy people, and enterprises.
- Paper: clean security model, well-defined adversary (record-and-replay, proxy, transplant), all already measured.

### Final-2: **#2 PUF-free identity on commodity GPU** (the *scientific* contribution)
- The substrate paper. Everything else cites this.
- Demo is the 8-laptop heatmap, immediately visual.
- Novelty is genuinely high: no SRAM PUF, no fTPM, no vendor enrollment — just telemetry signal fusion.

### Final-3: **#18 AI that forgets on transplant** (the *visceral* demo)
- Public-facing, talks to journalists, talks to policymakers, talks to consumers.
- Composes from #1+#31 trivially.
- Has a one-screen story.

### Why these three together
- #2 = the science.
- #1 = the protocol contribution.
- #18 = the consumer-grade demo that makes both legible.

A single ~10-page paper can carry **#2 as the result, #1 as the protocol, #18 as a case study**, with #21 and #38 as additional applications in a 1-paragraph "and also" section. This is a much stronger frame than "static accuracy gain" — the value isn't *that* the model is more accurate, it's that the *inference itself is now an attestable physical event*.

### Recommended next concrete steps (not heavy compute)
1. Write a 1-page security model for #1 (adversary capabilities, exact replay/proxy/transplant guarantees from existing data).
2. Storyboard the 8-laptop heatmap figure for #2 (we already have the data).
3. Build a 90-second screen-recorded demo of #18 using *existing* Phase 14C data — no new experiment required: replay the cross-host 2% transplant numbers as a sealed-key boot script.
4. Draft a short "TPM is not enough" related-work table showing what we add that TPM/SGX/Apple-SE do not (cold-boot resistance, liveness, no vendor PKI, commodity hardware).

### Avoided traps
- We deliberately do **not** anchor on consciousness/identity-of-mind framings. The capability is hardware-attestation; the AI angle is "this attestation lives at the inference boundary, not the firmware boundary."
- We deliberately do **not** chase capabilities that TPM2/SGX/SEV-SNP already deliver, even if they look impressive — we don't have a moat there.
