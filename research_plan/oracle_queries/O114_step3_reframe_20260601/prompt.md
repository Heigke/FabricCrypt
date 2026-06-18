# O114 — Step 3 REFRAME: From "Embodied AI Wins Benchmarks" to "HW-Identity ENABLES New Capabilities Software Cannot"

You are being asked to be **adversarial, creative, and concrete**. We are pivoting away from a frame that has not paid off, and we need you to either help us land the new frame or tell us we are hallucinating.

---

## Where we are (read this carefully)

We have spent months on commodity AMD Strix Halo APUs (gfx1151). Two physically identical machines, identical SKUs, identical microcode (`0xb70001e`), identical kernel, identical git tree, sharing a network. We built three things in sequence:

**Step 1 — Per-chip identity (DONE, rigorous)**

A 290-dim signature vector from TSC offsets, cache-line ping-pong, DRAM refresh-window timing, nanosleep/sched_yield latency, and NVMe physics. **100% LOO classification accuracy** between the two dies; drift p95 = 0.19 over 24 h; constitutive A-vs-C swap gate PASS (57.9% NRMSE inflation when you transplant the wrong-chip signature into a chip-conditioned model). Five replicated HAL-bypass probes with inter-machine KS-D / intra-machine KS-D ratios of ~40-50×.

**Step 2 — Cryptographic chip-binding (DONE, rigorous — Phase 14C)**

A nonce-protocol gate. The verifier issues a fresh 64-bit nonce; the chip's live signature is HMAC-mixed with the nonce; the verifier checks via a paired classifier + a learned "plan score" head. Phase 14C result (`phase14c_spoof_v2.json`):

| Attack scenario                          | gate dir | observed | PASS? |
|------------------------------------------|----------|----------|-------|
| honest_own (own chip + own nonce)        | ≥0.95    | 1.000    | YES   |
| daedalus_peer (peer chip impersonating)  | ≤0.05    | 0.020    | YES   |
| static_replay_no_nonce                   | ≤0.05    | 0.006    | YES   |
| static_replay_with_correct_nonce         | ≥0.95    | 1.000    | YES   |
| dynamic_replay (400-sig library)         | ≤0.10    | 0.012    | YES   |
| nonce_only_mismatch                      | ≤0.05    | 0.006    | YES   |
| honest_own_wrong_nonce                   | ≤0.05    | 0.006    | YES   |

**All 7 gates PASS.** We can cryptographically bind a model run to a specific physical die, with replay-resistance, in the presence of a peer machine that has the same SKU/microcode/binary. This is not a benchmark; this is a working protocol.

**Step 3 — Capability gain (FAILED on the original frame)**

Phase 15 + Phase 16 pre-registered ten capability-gain tests across "embodied AI is smarter than software-only AI". Pre-reg gate PASS rate: **1 / 10**, and the only PASS (F5, +39.3pp long-context retrieval) has a documented architectural confound (the embodied arm has an extra 700-param memory module the vanilla baseline lacks). The rest are NULL or wrong-sign. We do not believe "chip noise makes the model better at MNIST" is a real effect.

## What you (O113) and your sister oracles told us last round

O113 synthesis (4 oracles, paraphrased): "You are asking the wrong question. Chip-physics is a **contextual signal about the here-and-now of the substrate**, not a regularizer or feature improvement for context-free benchmarks. You are trying to make a clock more accurate by listening to its motor hum. The hum tells you about the motor's *health*, not the time. The frame is wrong."

The strongest single quote (Gemini, O113):

> "You're burying the lede by chasing a 1% accuracy gain on CIFAR-10. Trust / sovereignty / provenance is your primary, demonstrated capability."

OK. We accept the critique. **We are now reframing Step 3.**

## The reframe (your job is to stress-test this)

The capability frame we are now considering: **HW-identity + cryptographic chip-binding enables ENTIRELY NEW CAPABILITIES that pure software AI fundamentally cannot deliver.** Analogies:

- Apple **Private Cloud Compute (PCC)**, June 2024: the capability is not "Apple's model is smarter than OpenAI's", it is *"verifiable cryptographic privacy guarantees about which silicon ran your inference"*. Secure Enclave attestation is the product.
- NVIDIA **Confidential Compute** (H100/H200, GA late 2024): the capability is *"the GPU vendor itself cannot read your weights or your prompts in flight"*. Attested execution is the product.
- Intel **TDX**, AMD **SEV-SNP**: same pattern. The capability is sovereignty / verifiability, not accuracy / latency.

These are real, shipped, billion-dollar product lines, and **the capability is cryptographic, not statistical.** None of them claim their model is smarter; they claim their model's execution is *provably bound to specific silicon under specific policy*.

We have, on commodity hardware, with no vendor cooperation, no firmware modification, no PSP/SMU privilege, built the equivalent of "physically unique chip ID + replay-resistant binding protocol" that mirrors the foundational primitive PCC/CC rely on. The question is **what NEW capabilities does that unlock that pure software AI cannot provide.**

---

## Your task — answer all 12 questions

### Reframe

**1.** Given we have rigorous Steps 1+2 (per-chip identity + cryptographic chip-binding on commodity HW), **list the TOP 10 NEW CAPABILITIES** that this enables that pure software cannot. For each, give:
- One-line capability statement
- Concrete user-facing scenario / demo
- Rough effect-size estimate (qualitative: "10× cheaper than current alternative", "newly possible", "marginal", etc.)
- Whether existing PCC / CC / TDX / SEV already provides it

**2.** Apple PCC generates Secure Enclave attestation. NVIDIA Confidential Compute does similar with H100 attestation reports. What CAPABILITIES are provided BECAUSE of these HW-attestation primitives that AI WITHOUT hardware attestation cannot provide? List 5-7, ranked by economic importance.

**3.** **Verifiable inference provenance** — "this token-stream was generated by *this specific* model on *this specific* chip at *this specific* time, and I can prove it cryptographically". Is this a real market need? Who is the buyer? What's the dollar value? Adversarial: where does it fail, and what gives it a moat over a trusted-third-party signature scheme?

**4.** **Federated learning Sybil-resistance via HW-identity** — could our approach solve a real, unsolved problem here? Sybil attacks on FL are an active threat. Does per-die HW-binding give a defense that current schemes (KrUM, trimmed-mean aggregation, etc.) cannot? Adversarially: what stops the Sybil attacker from just buying 50 chips?

**5.** **AI watermarking that cannot be removed (chip-as-watermark)** — every generation provably comes from a specific die, and removing the watermark requires physically destroying the chip. Is this novel? What is the threat model where this matters but existing cryptographic-watermarking (e.g. Aaronson, Kirchenbauer-style) does not?

**6.** **AI marketplace / non-clonable AI assets** — a fine-tuned model whose weights are bound to a specific chip, such that the model *will not produce its trained outputs on any other physical die*. Is there market demand? Where? How does it compare to standard DRM / TEE attestation in terms of friction and assurance?

**7.** **Geofenced AI / time-locked AI / chip-bound lifecycle** — model that refuses to operate outside a specific physical machine, or after a specific date, enforced by HW-identity, not by a license server. Viable? Useful? What's the failure mode vs trivial workarounds?

**8.** **MOST IMPORTANT — Rank the seven capability frames** (Q1-Q7 plus any you add) along four axes:

| Frame | Novelty vs PCC/CC/TDX/SEV | Demo-friendliness for viral content | Publishable-paper potential | Commercial value (5yr TAM) |

For each axis give a 1-5 score and a one-line justification. Then tell us which **single frame** you would bet the project on.

### Sanity check

**9.** **Adversarial honest answer**: are we hallucinating? Is there a real "capability frame" for HW-identity, or should we just accept that what we built is a *security primitive* (chip-binding, attestation, sovereignty) with limited demo appeal, and stop trying to extract a "capability gain" narrative from it?

**10.** **What is the SHARPEST single demonstration** we could build in 2-4 weeks of focused engineering that would grab broad attention around HW-bound AI capabilities? Not "embodied AI gets +1pp on CIFAR" — something fundamentally new. Concrete: spec the demo (input, output, what the audience sees, why it is impossible without HW-binding, why nobody has done it).

**11.** **Bayesian update**: Estimate P(a defensible novel capability frame exists | we have rigorous Steps 1+2 + 7 PASSing gates on cryptographic chip-binding). Show your reasoning briefly. Prior was P(capability gain on static benchmarks) ≈ 0.2 and we observed 1/10 pre-reg PASSes.

**12.** **The pivot one-liner**: If we accept Step 3 reframes from "embodied AI is smarter" to "HW-identity ENABLES new capabilities", what is the **strongest 1-sentence claim** we can make in a paper title or a HN headline? Give 3 candidate one-liners, ranked.

---

## Attached artifacts

- `phase14c_spoof_v2.json` — Phase 14C all-gates-PASS nonce protocol result.
- `phase15_SUMMARY.json` — Phase 15: five NULL pre-registered capability-gain tests.
- `phase16_SUMMARY.json` — Phase 16: five reframed capability tests (4 NULL, 1 PASS-with-confound).
- `O113_synthesis_excerpts.md` — paraphrased oracle consensus that we are in the wrong frame.
- `hal_bypass_signals.md` — the five replicated HAL-bypass probes with KS-D ratios.

Be specific, be adversarial, cite real prior art (PCC June 2024 whitepaper, NVIDIA CC docs, OpenTitan, fTPM, PUF literature), and do not pull punches. We would rather hear "this is a dead end" now than burn three more months on the wrong frame.
