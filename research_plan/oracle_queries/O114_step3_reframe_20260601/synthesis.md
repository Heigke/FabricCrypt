# O114 Synthesis — Step 3 REFRAME: HW-identity ENABLES new capabilities (not "embodied AI is smarter")

**Date:** 2026-06-01
**Oracles:** openai (gpt-5, 409s, 22k chars), gemini-2.5-pro (101s, 20k), grok-4 (35s, 9k), deepseek-reasoner (72s, 27k).
**All 4 responded.** Strong convergence on the central claim; meaningful divergence on which capability frame to bet on and on novelty scoring.

---

## TL;DR — the 4-oracle verdict

1. **We are NOT hallucinating.** All four oracles agree: the old "embodied AI gets +1pp on CIFAR" frame is dead (1/10 PASS with confound, predictable null). The new "HW-identity as a cryptographic capability primitive" frame is real and defensible.
2. **The pivot is correct.** Stop trying to wring accuracy gains from the primitive. Treat Steps 1+2 as a **vendor-independent attestation layer on commodity silicon** — the same category as Apple PCC and NVIDIA Confidential Compute, but achieved without OEM keys, firmware mods, or TEEs.
3. **Bayesian update on the new frame:** posterior P(defensible novel capability frame exists | Steps 1+2 work + 7/7 nonce gates PASS) ≈ **0.80 – 0.95** across oracles (gpt-5: 0.80-0.85, gemini: 0.95, grok: 0.65, deepseek: 0.92). Median ~0.85.
4. **All four oracles agree it is a security/sovereignty primitive — not an ML feature.** Sell it as attestation, not as intelligence.
5. **All four oracles agree on the limit:** without a TEE we get *presence + liveness + replay-resistance*, NOT *confidentiality or code-integrity*. Be honest about this in the paper — it is the moat against PCC/CC, not against rooted hosts.

---

## Q1 — Top 10 NEW capabilities (consolidated across oracles)

Ranked by oracle consensus (frequency × strength of endorsement):

| # | Capability | Newly possible? | Effect-size estimate | Already in PCC/CC/TDX/SEV? |
|---|------------|-----------------|----------------------|----------------------------|
| 1 | **Verifiable sovereign on-prem inference receipts** (BYO-attestation, no vendor key) | YES on commodity HW | 10× cheaper than buying TEE-capable servers | Partially — yes inside TEEs, but never vendor-independent |
| 2 | **Proof-of-Physical-Compute (PoPC) for decentralised compute markets** | YES on commodity | 10× drop in spoof fraud | Yes via TEE; your edge = open-HW coverage |
| 3 | **Sybil-resistant federated learning via per-die identity** | Partial — improves attacker cost from $0 to $400/die | 10-100× attacker cost increase | Phones only (SafetyNet); PC/edge gap is the win |
| 4 | **Non-repudiable inference provenance for regulated industries** (finance/healthcare/defense) | YES on commodity HW | Real $10-300M ARR by oracle estimates | Yes inside enclaves; your edge = no enclave needed |
| 5 | **Vendor-independent anti-counterfeit / fleet authentication** (retrofit PUF on existing CPUs) | YES — no fTPM needed | Newly possible in brownfield | OpenTitan/SRAM-PUF needs silicon support |
| 6 | **Tamper-evident / anti-glitch health attestation** (analog liveness) | YES — physics-based | Complements TEE (TEEs attest code; you attest physics) | TEEs don't do this |
| 7 | **Non-clonable AI assets** ("this model only runs on this die") | YES — strong license enforcement, weak IP protection | Real demand in model marketplaces | DRM/TEE possible but higher friction |
| 8 | **Trustless metered licensing for on-device inference** | YES | Reduces piracy/proxy fraud | TEEs can; you reduce integration cost |
| 9 | **Anonymous hardware uniqueness** (zk-prove "I am a unique chip" without revealing which) | YES — novel construction | Niche but high novelty (blockchain airdrops, unique-voter) | No equivalent |
| 10 | **Keyless device-unique secrets** (fuzzy extractor → stable key, no NVRAM/TPM) | YES on brownfield | TPM-lite for missing-TPM machines | Real TPMs do better; your edge = retrofit |

Additional capability surfaced by Grok: **physical rate-limiting** (cannot run more inferences than physics allows per die per second — basis for fair billing).

---

## Q8 — Frame ranking (consolidated 1-5)

Aggregated mean of oracle scores. **Two clear leaders** emerge depending on what you optimise for.

| Frame | Novelty | Demo-friendliness | Paper potential | 5-yr TAM | TOTAL |
|---|---|---|---|---|---|
| **Verifiable inference provenance / sovereign on-prem receipts** | 3.0 | 4.0 | 4.5 | 4.5 | **16.0** |
| **Non-clonable AI assets / model bound to die** | 3.3 | 4.3 | 3.5 | 3.5 | **14.6** |
| **Sybil-resistant federated learning** | 4.3 | 3.3 | 4.5 | 3.0 | **15.1** |
| Proof-of-Physical-Compute (decentralised) | 4.0 | 4.0 | 3.0 | 3.0 | 14.0 |
| Tamper-evident health attestation | 4.0 | 3.0 | 4.0 | 3.0 | 14.0 |
| Unforgeable chip-as-watermark | 3.3 | 3.0 | 3.0 | 2.3 | 11.6 |
| Geofenced / time-locked | 1.3 | 2.8 | 1.8 | 2.3 | 8.2 |

### Oracle "single best bet" — split 2-1-1
- **GPT-5 → Verifiable sovereign on-prem inference receipts** ("cleanest distillation of your unique primitive, demoable, publishable, commercially relevant")
- **Gemini → Non-Clonable AI Assets** ("best balance across all four axes; killer side-by-side demo")
- **Grok → Verifiable inference provenance** ("highest demo virality + clearest regulatory path; nonce protocol already works")
- **Deepseek → Unforgeable inference attestation** ("largest existing market, highest paper potential, direct Apple-PCC comparison gives 'we did it cheaper' story")

**Three of four oracles converge on Variant-of-Verifiable-Provenance (1, 3, 4 are essentially the same frame at different abstractions).** Bet there.

---

## Q9 — Are we hallucinating? (Adversarial honest answer)

**Unanimous: No.** But with important nuance:

- **Old frame ("HW makes model smarter") IS a hallucination.** Phase 15 + 16 confirm.
- **Security/sovereignty frame IS real.** The TEE industry's existence is overwhelming evidence that this category has buyers.
- **Grok's sharpest critique:** "You ARE hallucinating the breadth of 'entirely new capabilities'. Most valuable versions already exist inside vendor TEEs with stronger roots of trust. Your edge is **cost and accessibility**, not novelty of capability. The realistic product is a niche provenance/audit tool, not a general-purpose capability revolution."

**Take this seriously.** The novelty axis where we genuinely dominate is **"attestation without vendor keys / TEEs / firmware mods on commodity silicon"** — not "we invented a new kind of attestation". The story is *democratisation*, not *invention*.

---

## Q10 — Sharpest 2-4 week demo (3 candidates from oracles, ranked)

### Winner — "Proof-of-Physical-Inference Receipt" (GPT-5 + Grok + Deepseek converge)

Two physically identical Strix Halo machines on the desk. Public verifier web service. Small LLM with nonce protocol.

**Show 4 things in 60 seconds:**
1. **Honest:** Machine A gets a fresh nonce, runs inference, produces token-stream + receipt. Verifier shows green check: "Die A, microcode 0xb70001e, T=42°C, accept p=1.00."
2. **Peer-impersonate:** Forward request to Machine B (identical SKU/binary). Machine B runs the same model, returns the same tokens, **but the receipt verifies RED** (accept rate 0.02).
3. **Replay attack:** Replay a captured signature from earlier. Verifier rejects (nonce-mismatch, accept rate 0.006).
4. **Tamper-detect (optional flourish):** Heat-gun machine A. Live signature drifts out of baseline; verifier refuses the receipt.

**Punchline:** "Two identical computers. Identical software. Only the real one can prove it ran the inference. No TPM, no Secure Enclave, no vendor cooperation."

**Why viral:** PCC has the same property but requires Apple silicon + Apple servers + Apple keys. You're showing the foundational primitive on commodity AMD — and the side-by-side identical-machines visual is striking.

### Runner-up — Gemini's "Unclonable AI" demo
SD-LoRA fine-tuned + chip-bound. Train on Ikaros, scp to Daedalus, run same prompt → black image with "Execution on unauthorized hardware prohibited." Punchline: "Software you can't pirate."

### Riskier — Deepseek's "zk-prove I am a unique chip without revealing which"
Beautiful but requires getting the 290-dim signature stable enough for a zk-SNARK circuit. Fallback to the simpler nonce-HMAC version is just demo #1.

---

## Q11 — Bayesian update (consolidated)

| Oracle | Posterior P(defensible novel capability frame exists) |
|---|---|
| GPT-5 | 0.80 – 0.85 |
| Gemini | 0.95 |
| Grok | 0.65 |
| Deepseek | 0.92 |
| **Median ≈ 0.85** | |

The variance is driven by how strictly each oracle defines "novel capability frame":
- High posterior (Gemini, Deepseek): TEE industry exists; you built that primitive on commodity HW → obvious win
- Lower posterior (Grok): the *capability* exists everywhere; only the *access cost* is new → narrow win

**Both views are defensible. Plan for the conservative case.**

---

## Q12 — Pivot one-liners (ranked across oracles)

### Top 3 (mixing winners across the 4 oracles)

**1. Gemini #1 (best HN headline):**
> *"Forget making AI smarter — we're making it sovereign: unclonable, unforgeable AI bound to commodity silicon."*

**2. GPT-5 #1 (cleanest pitch):**
> *"Proof of Physical Compute: cryptographic inference receipts on commodity chips — no TEE, no vendor keys."*

**3. Grok #2 (sharpest technical claim):**
> *"We built Apple PCC's attestation primitive on two identical AMD APUs using only userspace physics."*

### Honorary mentions
- Deepseek: *"Commodity chips have unique physical fingerprints — we can now cryptographically bind any AI model to a specific die. No TEE, no vendor help."*
- Gemini #3 (academic): *"Beyond Benchmarks: a working protocol for cryptographically attesting and binding AI to off-the-shelf hardware."*

---

## Key cross-oracle convergences (high confidence)

1. **OLD FRAME IS DEAD.** Stop trying to improve static benchmarks with chip noise. (4/4 oracles)
2. **The primitive is real, replicated, rigorous.** 7/7 nonce gates + 100% LOO + drift p95=0.19 is publishable on its own. (4/4)
3. **The right comparand is PCC / NVIDIA CC / TDX / SEV — but pitched as *democratised attestation*, not as a new attestation primitive.** (4/4)
4. **Best demo is two-machines-same-SKU side-by-side.** Receipt verifies on the real one, fails on the identical clone. (3/4 — GPT-5, Grok, Deepseek converge; Gemini's "unclonable LoRA" is variant)
5. **Honest disclaimer required in any paper:** without TEE we get *presence + liveness + replay-resistance*, NOT *confidentiality or code-integrity*. A rooted host still leaks data. (4/4)
6. **Federated learning Sybil is real but bounded:** raises attacker cost from $0 to ~$400/die. Not a complete solution; combine with KrUM / trimmed-mean. (3/4)

## Key cross-oracle divergences

1. **Best frame to bet on:** GPT-5/Grok/Deepseek say *verifiable provenance / receipts*. Gemini says *non-clonable AI assets*. (Both lead aggregated scores; pick by team preference.)
2. **Novelty scoring is non-uniform:** Grok scores verifiable-provenance novelty at 2 (everyone has it in TEEs), Gemini scores at 2 (same reason), but GPT-5 scores at 4 (vendor-independent IS new). Deepseek scores at 4. The disagreement is semantic: "is doing-PCC-on-commodity novel?"
3. **Watermarking / geofencing:** all four say weak. Geofence scored 1-2 on novelty by all; do not pursue.

---

## Recommended pivot plan (synthesised across oracles)

### Step 3 reframed → "Capability Enablement via Commodity HW-Bound Attestation"

**Narrative arc for the paper / project:**
1. **Step 1 (DONE):** Per-die identity on commodity AMD (100% LOO).
2. **Step 2 (DONE):** Cryptographic chip-binding with 7/7 nonce gates PASS.
3. **Step 3 (NEW — capability enablement, not capability gain):**
   - Build the "Proof-of-Physical-Inference Receipt" demo with Ikaros + Daedalus.
   - Add a public verifier web service with a "spoof this die for $X bounty" challenge.
   - Optional follow-ups: FL Sybil benchmark, tamper-evident health-baseline classifier, non-clonable LoRA.

**Paper title candidate:**
> *"Proof of Physical Compute: Vendor-Independent AI Attestation on Commodity Silicon"*

**HN headline candidate:**
> *"We built Apple Private Cloud Compute's core primitive on commodity AMD chips — no TEE, no vendor keys."*

**Honest moat statement:**
> "We provide presence, liveness, and replay-resistance on commodity hardware. We do *not* provide confidentiality or code-integrity (those still require a TEE). The contribution is democratisation: any AMD Strix Halo can now mint cryptographic inference receipts without OEM cooperation."

### Don't pursue
- Geofence / time-lock (all oracles agree weak)
- "HW noise as regularizer / feature improver" (dead)
- Generic chip-as-watermark (paraphrasing trivially defeats it)

### Pursue but cautiously
- FL Sybil resistance — strong novelty score, narrow market, single-paper material
- Tamper-evident health attestation — interesting cross-layer security paper, narrow market

---

## File inventory in this packet

- `prompt.md` — the 12-question dispatch
- `phase14c_spoof_v2.json` — 7/7 gates PASS Phase 14C nonce protocol
- `phase15_SUMMARY.json` — 5 NULL pre-reg tests (old frame)
- `phase16_SUMMARY.json` — 5 reframed tests (1 PASS-with-confound)
- `O113_synthesis_excerpts.md` — wrong-frame consensus
- `hal_bypass_signals.md` — the 5 replicated HAL-bypass probes
- `openai_response.md` (22k chars, 409s)
- `gemini_response.md` (20k chars, 101s)
- `grok_response.md` (9k chars, 35s)
- `deepseek_response.md` (27k chars, 72s)
