# O96 Synthesis — 4-way oracle critique of novel identity angles

Date: 2026-05-30
Oracles: GPT-5, Gemini 2.5 Pro, Grok-4-latest, DeepSeek-Reasoner — **4/4 responded**

## Vote matrix (top-3 picks, Q1)

| Angle | GPT-5 | Gemini | Grok | DeepSeek | Score |
|---|---|---|---|---|---|
| **B** (trajectory / Lyapunov) | YES | — | YES | YES | **3** |
| **E** (attention-routing topology) | — | YES | YES | YES | **3** |
| **J** (split-brain) | — | YES | YES | — | 2 |
| **F** (self-referential) | — | YES | — | — | 1 |
| **D** (MC arbitration) | YES | — | — | — | 1 |
| **I** (EMI) | YES | — | — | — | 1 |

Tied at top: **B** and **E** with 3/4 votes each.

## Consensus findings (unanimous or 4/4)

1. **Angle F is NOT novel.** All four cite watermarking / device-conditioned inference / model-binding literature (Rouhani DeepSigns 2019, Gu BadNets 2017, Li HWN-DNN ISCA 2020, "Hardware-Adaptive DNN Watermarking" CCS 2022). User's framing of "interoception" is branding, not new mechanism.
2. **Angle J is engineering theater UNLESS the inter-half interaction depends on non-virtualizable per-die secrets / hardware-specific all-reduce timing.** Plain parameter sharding = ensemble = fungible. Only Gemini defends J strongly; GPT-5, Grok, DeepSeek call it theater.
3. **Angle C (tournament RO) is statistical illusion.** 4/4 unanimous: 79 races share the same PDN + thermal envelope; aggregation amplifies the *common latent* (board-level droop), not silicon entropy. Kill it as a duplicate of Phase 1c Probe B.
4. **Angle A (product-of-experts) fails its independence assumption.** 4/4 unanimous: RTN, spatial-corr, RO winrate, LDS-startup are all monotone in Tdie + Vcore. PoE "will simply learn to be a very complicated thermometer" (Gemini). User MUST measure cross-channel correlation before fusing.
5. **Duplicates to kill: C (= Probe B), and at least one of {D, E, H} per each oracle.** Consensus on C as duplicate.

## Sharpest disagreement

**Angle J (split-brain).** Gemini: "genuine non-fungibility... ontologically tied to the specific ikaros+daedalus pair." GPT-5/Grok/DeepSeek: theater unless the *interaction protocol itself* requires non-exportable per-die secrets at runtime (PUF-derived ephemeral keys, hardware-specific timing). My read: **the majority is right**. J as currently specified is sharding. To upgrade J to non-theater, the inter-half all-reduce or attention exchange must be gated on a PUF-derived key re-derived each forward pass — otherwise a third machine with copied params is functionally identical. This is fixable but adds substantial scope.

Secondary disagreement: **build-or-don't.** Grok says "build nothing new; re-run Phase 1c at ±0.3 °C". Gemini/DeepSeek say "falsify survivors first under thermal/burn-in stress before building new probes." GPT-5 says "build the 11th angle (PDN Z(f) spectroscopy)." This is real — Grok's nihilism vs the majority's "build cheap orthogonal probes".

## Novel 11th angles proposed

- **GPT-5**: PDN impedance spectroscopy (chirped load 1–500 kHz, per-CU clock-stretch → Z(f) resonance map). Board+die specific, richer than 1/f knee. *Best of the four.*
- **Gemini**: Active thermal response (power-virus transient, measure on-die sensor rise/settling time → thermal impedance fingerprint of die/TIM/heatsink).
- **Grok**: Per-CU instruction-retirement skew under locked DVFS, single-opcode-mix sweep. Residual after T-match = only candidate.
- **DeepSeek**: GDDR6 ECC bad-block map via EDAC polling — cell-level fixed faults, orthogonal to APU noise channels. *Cheapest to build.*

## Top 3 angles by consensus → BUILD ORDER (24h)

### Priority 1 — Angle B (trajectory-as-signature)
3/4 votes. Cheap to build (chaotic ODE on GPU with FP rounding accumulation). **Known failure mode (4/4 agree)**: longitudinal stability — driver / compiler upgrades flip trajectories (DeepSeek cites Behnam DAC 2019). Mitigation: pin driver + compiler hash; measure stability over hours, not days.

### Priority 2 — DeepSeek's 11th (GDDR6 ECC bad-block map)
Cheapest novel orthogonal probe. EDAC register polling, no kernel mods, no risk to running Phase 1c/2 agents. Stable fixed faults are silicon — not thermally modulated. Highest value/effort ratio of the four 11th-angle proposals.

### Priority 3 — Angle E (attention-routing) **as a Phase-2 redesign**, not a new probe
3/4 votes. NOT a new identity-discovery channel — it's a *deeper substrate coupling* for the model side. Use the 2 silicon-confirmed channels (RTN, spatial-corr) to gate attention-head routing in a tiny 2-layer transformer. This is the genuine "constitutive coupling" path the orthodox Phase 2 activation-noise injection lacks. DeepSeek explicitly recommends this as part of build set.

### Kill list (do not build)
- **C** (tournament RO) — duplicate of Probe B, fails independence.
- **A** (product-of-experts) — fails independence; defer until cross-channel correlation matrix measured.
- **F** (self-referential) — not novel, covered by watermarking literature.
- **J** (split-brain) — theater unless PUF-keyed interaction protocol added (out of 24h scope).
- **G** (rowhammer) — fails uniqueness under T-cycling (CHES 2019, USENIX'21).
- **I** (EMI) — destroyed by PSU filtering (Grok cite).

## Cross-cutting mandate from all four

**Before any new build, measure the cross-channel correlation matrix on the 2 surviving channels (RTN, spatial-corr) at matched T.** If they correlate > 0.7 they are *one* signal, not two, and the orthodox path's apparent recovery in Phase 1B is weaker than claimed. This is the falsification step that should precede 24h novel-probe work.

## Files

- Prompt: `prompt.md`
- Attachments: `IDENTITY_NOVEL_ANGLES_2026-05-30.md`, `IDENTITY_BENCHMARK_2026-05-30.md`, `..._PHASE1.md`, `..._PHASE1B.md`, `O95_prior_synthesis.md`
- Responses: `openai_response.md`, `gemini_response.md`, `grok_response.md`, `deepseek_response.md`
- Dispatch log: `_dispatch.log`
