# Oracle query O96 — Novel angles for hardware-identity benchmark (4-way)

You are one of four oracles (GPT-5, Gemini 2.5 Pro, Grok-4, DeepSeek-R) being asked to **adversarially critique** a brainstorm of 10 "outside-the-box" angles (A–J) for extending a hardware-identity / PUF research program on **twin AMD Ryzen AI Max+ PRO 395 (Radeon 8060S, gfx1151, RDNA3.5)** chassis (ikaros, daedalus). Be terse, ≤500 words, structured as numbered answers 1–10. **Hostile / falsifying tone preferred — find wrong assumptions, do not validate uncritically.**

---

## Context (read attached files for full detail)

- **`IDENTITY_NOVEL_ANGLES_2026-05-30.md`** — the 10 angles A–J with the user's self-ranking.
- **`IDENTITY_BENCHMARK_2026-05-30.md`** — original 32-mechanism design doc.
- **`IDENTITY_BENCHMARK_2026-05-30_PHASE1.md`** — Phase 1 results: stable-bit channel NULL; process-stat channel apparent signal but ~15 °C ambient confound.
- **`IDENTITY_BENCHMARK_2026-05-30_PHASE1B.md`** — Phase 1B verdict (thermal-controlled): RTN + spatial-correlation **silicon-confirmed** (survive matched temperature), 1/f knee collapses (thermal-only, killed by prior oracle round).
- **`O95_prior_synthesis.md`** — prior 4-way oracle round (unanimous kill of 1/f knee, demanded thermal-matched repeat which we then ran).

**State of play:** orthodox Phase 1c (probes A–D: Holcomb SRAM startup, single-pair RO, BTI burn-in, leakage) and Phase 2 (NARMA-10 transplant) are already scheduled / running. We have **2 silicon-confirmed channels** (RTN, spatial-corr). User now wants angles the orthodox PUF literature misses entirely — especially angles that bridge to the **"constitutive coupling → non-fungibility → stake / death-relevance"** framework (oracle's own framing: identifiable → non-fungible → stake).

---

## The 10 angles (summary; full text in `IDENTITY_NOVEL_ANGLES_2026-05-30.md`)

- **A** Cross-modal weak-signal aggregation (product-of-experts across 8–16 weak channels)
- **B** Trajectory-as-signature (chaotic GPU ODE accumulating FP rounding → Lyapunov fingerprint)
- **C** Tournament racing (80-CU bracket of ring-oscillator pairs, 79 races → one winner pattern)
- **D** Memory-controller arbitration race (two threads racing same VRAM addr)
- **E** Attention-routing coupling (per-CU ΔVth determines transformer attention graph)
- **F** Self-referential / interoception (model reads its own hwreg signature as input feature during training)
- **G** DRAM rowhammer state (chip-specific row-flip pattern)
- **H** Cross-machine challenge-response auth over network
- **I** Power-line EMI fingerprint (PSU radiation on shared mains)
- **J** Split-brain co-dependence (one model whose params are physically split across ikaros + daedalus; neither half functional alone)

User's self-pick: F, J, C, A. Skipped: G, I, D, E.

---

## Answer each question (1–10) explicitly

1. **Rank top 3 angles (A–J) most likely to surface NEW signal the orthodox PUF path misses.** One-line reasoning each.

2. **Angle F (self-referential)** — academically novel or already done? If adversarial-ML / PUF / model-watermarking literature has covered it, cite the paper.

3. **Angle J (split-brain)** — genuine non-fungibility, or engineering theater for what an ensemble model gives free? Distinguish the two.

4. **Angle C (tournament RO) vs Phase 1c Probe B (single-pair RO)** — is tournament aggregation a real strengthening, or 79 correlated weak signals that statistically don't compound (shared thermal envelope, shared PDN)?

5. **Angle A (cross-modal product-of-experts)** — is the independence assumption likely to hold across our 4 candidate channels (RTN, spatial-corr, RO winrate, LDS-startup), or are they all dominated by one latent (die-temperature, PDN droop, workload history)?

6. **Duplicates** — are any of A–J actually measuring the same thing as a Phase 1c probe just with extra ceremony? Identify duplicates we should kill.

7. **Propose ONE 11th angle the brainstorm missed.**

8. **Model-coupling depth** — orthodox plan injects substrate at neuron activation (value level). What is the *deepest realistic* coupling on AMD GPU? Should we instead inject at compute-graph topology level (architectural commitment to silicon-specific routing) rather than activation noise?

9. **Known failure modes** — for any A–J, has someone tried it and failed? Cite if possible.

10. **Concrete recommendation** — given budget for ~3 novel probes built in 24h, what should we build? Be specific.

---

Output format: numbered list 1–10, ≤500 words total. Hostile tone preferred.
