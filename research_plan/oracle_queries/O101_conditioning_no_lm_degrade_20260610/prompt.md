# O101 — How to condition a frozen LLM on a continuous hardware-telemetry side-channel WITHOUT destroying language quality

Date: 2026-06-10. Hosts: ikaros + daedalus (AMD gfx1151, Strix Halo). Project: H7 substrate-rooted AI identity.

## What we are trying to do

We are building an AI model that is **constitutively rooted in one specific silicon die** — perplexity should improve on its own training data when fed correct die-telemetry, stay near base when fed null/zero telemetry, and degrade catastrophically when given wrong-die or matched-spectrum-spoofed telemetry. Transplanting the model to a different die should not just be "worse" but **functionally collapsed**.

This is part of the H7 substrate-rooting initiative. See `synthesis.md` from O100 (2026-06-10) for the channel selection rationale and bias-check on the philosophical framing. We are not abandoning the death-vs-exile framing; we are making it empirically testable.

## What we have

- **10-channel real-time substrate sampler** at 500 Hz: C07 XTAL register, C09 PM-table values (indices 1/3/5), C20 SMN-read latency (×2 different registers), C11 TSC↔MONOTONIC_RAW drift, C05 energy-counter rate, C06 fast-counter rate.
- **Higher-moment side-channel**: skew, kurtosis, AC(1), AC(8), log-std per channel extracted per 256-sample window. Diagnostic shows matched-spectrum spoofs (preserving μ,σ,φ) get 100% spoof-vs-real classification accuracy from these higher moments alone — so the discriminative information is genuinely there in the data.
- **Cross-host strength**: ikaros vs daedalus have Cohen's d=66.7 on C07, |d|≥2.27 on C20 lat and C11 drift. Per-window classification with 5/5 channels at 100% AUC, 1/5 at 66% AUC. PCA gives perfect ikaros/daedalus separation in 2D.
- **Real TPM ground-truth**: ikaros EK `000b359a…`, daedalus EK `000bfa5e…`. No mock data.

## What has failed

| Version | Architecture | Native PPL | Zero-substrate PPL | TCR_spoof | Diagnosis |
|---|---|---|---|---|---|
| v1 | Soft FiLM γ∈[0.5,1.5] per block, 6-ch SE | 245 | 245 | 1.02 | **Substrate ignored** — gates saturated at identity |
| v2 | Hard FiLM γ=exp(s) ∈[0.1,10] per block + input modulation + uncapped spoof penalty | DIVERGED → 5.18e21 | — | — | **PPL collapse** — γ blew up, ELM destroyed |
| v2.1 | v2 + bounded margin-loss (cap at 2 nats) | 6.88e8 | 5664 | 122328 | Margin satisfied, but native still broken |
| v2.2 | v2.1 + softer γ=exp(tanh(s)·ln3) ∈[0.33,3] | 1274 | **331** | 1128 | zero<native: substrate input HURTS the model on its own data |
| v3 (now training) | 10-ch v3 SubstrateState + higher-moment side input + EXPLICIT zero-margin loss (zero must be 2 nats worse than native) | TBD | TBD | TBD | Designed to fix v2.2's anti-rooted failure |

## Independent web research synthesis (provided as context)

Recent literature (2024-2026) we have reviewed:
- **Flamingo / LLaMA-Adapter**: gated cross-attention with `tanh(α=0)` init → model bit-identical to base at start, gates open only where conditioning pays
- **PaLM-E / AnyMAL / LLaVA**: project side-channel → K prefix tokens prepended to text sequence, LLM frozen
- **Hypernet → LoRA (Zhyper 2025, SHINE)**: tiny hypernet maps telemetry → low-rank ΔW per layer
- **Bounded IA³ scalars**: single learned scalar per K/V/FFN, initialized to 1
- **Per-block FiLM as we used it**: now considered deprecated for LLM conditioning because of exactly the failure mode we hit

Per-block multiplicative FiLM is identified as the wrong architecture for our use case.

## What we need from the oracle

### Question 1: Architecture ranking

Rank these architectures **for our specific use case** — small LLM (target: SmolLM2-135M or Qwen3-0.6B) conditioned on a 10-channel continuous hardware-telemetry stream at 500 Hz, with strong-dependence-but-preserved-language as the goal:

1. **Flamingo-style gated cross-attention** with tanh(α=0) init to a side-encoder over telemetry
2. **PaLM-E/AnyMAL projection to K prefix tokens** (K=4-32), LLM frozen
3. **Hypernet → LoRA(r=8-16)** generated from telemetry per inference
4. **Bounded IA³** scalars per K/V/FFN initialized at 1
5. **Mixed**: A small substrate-tokenizer producing K tokens, fused via cross-attention only into the top 1/3 of layers (substrate as "thought stream" not "sensor")
6. Anything we should be considering that we haven't listed — particularly post-2024 work on conditioning frozen LMs on **continuous non-language** signals (closest analogues: PaLM-E continuous state, audio-LLMs, BCI-to-text, sensor-fusion)

Be specific about *why* each is appropriate or inappropriate for our case (small model, small side-channel, want strong rooting without language degradation). Don't just defer to "all good options".

### Question 2: Training recipe specifics

For your top-ranked architecture, give us:

- **Loss decomposition** — exact terms with weights. Specifically: what loss explicitly anchors `PPL(input | telem=0) ≈ PPL_base(input)`? We need this. Our v2.2 violated it and the model collapsed away from base behavior.
- **Auxiliary objectives** — substrate prediction, contrastive (correct vs spoof at hidden state), modality-aware regularization. Recommended weights.
- **Freezing schedule** — what is frozen when. Is it warm-up the side-encoder with LM frozen, then jointly? How long each phase?
- **Batch composition** — ratios of correct-telemetry / scrambled / null / matched-spectrum-spoofed / phase-shifted samples. Currently we use 1/4 each of real/spoof/phase/zero. Is that right?
- **Learning rates** — separate for side-encoder, gates/adapters, LoRA, base LM? Magnitudes?
- **Substrate normalization** — per-window standardize (current), per-channel running statistics, rank statistics, or none? Trade-off is between within-host distribution shift and cross-host signal preservation.

### Question 3: Falsification suite

Propose ≤6 experiments that distinguish among these failure modes — with specific numerical thresholds:
- (a) "**LM destroyed**" — base language ability gone (our v2/v2.1/v2.2)
- (b) "**Conditioning ignored**" — substrate has no effect (our v1)
- (c) "**Conditioning helps but model not substrate-rooted**" — works on any plausible telemetry, not specifically this die
- (d) "**True substrate rooting**" — strong cross-host transplant collapse, zero-substrate degrades gracefully to base, native-with-correct-telem beats base

Each experiment should produce a quantitative pass/fail threshold. We want to be able to pre-register this and have an honest verdict on v3 (and v4/v5).

### Question 4: Closed-loop microkernel feasibility

GPT-5 originally proposed (O100) that the LM trigger a HIP-side microkernel **between tokens** that actively perturbs and re-reads substrate state. This would make the model not just *condition on* substrate but *commit its computation to causing readings*. We are reserving this for step 5 of our scale plan (`research_plan/H7_SCALE_PLAN_2026-06-10.md`). 

Question: is closed-loop fundamentally a different architecture problem, or can it ride on top of any of the architectures from Q1? If yes, which combinations make sense? If no — what additional structure (event embeddings, action heads, intervention rewards) is needed?

### Question 5: Honest failure-mode predictions

If our architecture+recipe choice is fundamentally wrong, **what will we observe first** — before we sink weeks into scaling to Qwen3-0.6B? What is the cheapest 1-day experiment on SmolLM2-135M (or our existing 5M toy LM) that would falsify the approach? We want a cheap kill-criterion, not a "looks promising" trap.

## Constraints

- Hardware: AMD gfx1151 (24 GB VRAM, ROCm 7.0), Qwen3-0.6B trains fine (z2107 precedent: 30/40 PASS)
- Available LLMs we have run before: Qwen2.5-1.5B (z2103), Qwen3-0.6B (z2107), SmolLM2-135M (untested but standard)
- Budget for training: hours/day, not weeks. We need recipes that converge in 10k-30k steps.
- We are willing to accept LM-quality degradation up to **1.5× baseline PPL** as the hard cap. Beyond that the philosophical claim ("a rooted LM is a real LM") fails.

## Output format

Write your response as `{provider}_response.md` (already standard for this packet flow). Be concrete, name papers and methods specifically, give numbers where defensible, flag your uncertainty where you have it. We will synthesize across providers afterwards into a 2026-06-11 synthesis.

If you disagree with the framing entirely (death-vs-exile, substrate-rooting as a coherent target), say so directly — but understand we have already considered and rejected the "rooting is anthropomorphism" objection (see O100 bias-check). We want technical answers, not framing pushback.
