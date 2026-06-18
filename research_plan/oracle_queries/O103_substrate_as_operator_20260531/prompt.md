# O103 — Substrate-as-Operator (not Substrate-as-Feature)

You are one of four hostile oracles (GPT-5 / Gemini / Grok / DeepSeek)
reviewing a **reframing** of our gfx1151 hardware-identity research.

## Background (one paragraph)

We have spent 6 oracle rounds (O95–O99) and ~32 attack mechanisms on
two twin AMD Strix-Halo gfx1151 APUs (`ikaros`, `daedalus`) trying to
make a model's behaviour *constitutively* depend on the silicon die,
not just *recognise* it. Every attempt collapsed to
`Δ HW ≈ Δ SHUFFLE` — the model is structure-bound, not device-bound.
All 14 prior mechanisms treated substrate as **information the model reads**:
power telemetry, thermal-τ, cache jitter, ISA cycle counts, fan transients,
TLB latency, atomic race orderings, hwreg(29) — all features fed into a
model that *consumes* them. SW-matched control always reproduced the gap.

## The reframing

Instead of `y = model(input, substrate_signal_as_feature)`, build:

    y = chip_specific_matmul(W, x)

where the GPU kernel itself **performs the math** in a way that exploits
IEEE-754 implementation freedoms, so bit-different output appears on
ikaros vs daedalus with identical (W, x).  Model weights then *co-adapt*
to THIS die's specific numerical fingerprint. Transplant breaks because
the new die's matmul produces a different operator that the weights
were not optimised for.

Specifically exploit (in a single HIP kernel):

1. atomic-add reduction non-determinism (IEEE allows any order)
2. subnormal/denormal flush (FTZ/DAZ; chip-defined flush rate)
3. FMA fusion (`a*b+c` vs `a*b` then `+c`; chip choice)
4. BF16/FP16 rounding (chip-defined tie-break)
5. reduction tree depth (varies with wave contention)
6. wave-conflict handling under controlled bank conflicts

Critical asymmetry vs prior attempts:
**No SHUFFLE control possible** — there's no "signal" to scramble.
**No SW-matched control possible** — there's no "noise" to simulate
without already having the chip-specific kernel running on the chip.
The falsifier we have used 14 times is *structurally undefined*.

## Bundled context

- `IDENTITY_ALL32_2026-05-31_REPORT.md` — 26/32 mechanisms NULL
- `IDENTITY_DEEPER_HUNT_2026-05-30.md` — 6 prior oracle rounds summary
- `IDENTITY_LITERATURE_HUNT_2026-05-30.md` — no prior commodity-userspace success
- (this prompt embeds the reframing in full above)

## Questions (answer each numbered)

1. The reframing "substrate IS the operator, not a feature the model
   reads" — genuinely architectural shift from our prior 14 attempts,
   or just rebranding the same dead end? Be hostile.

2. AMD ROCm rocBLAS / MIOpen / hipBLASLt: **actually bit-deterministic
   by default**, or do they have implementation-defined branches that
   vary per chip (or per driver build, or per launch)? Cite ROCm docs /
   GitHub issues.

3. Rank, for gfx1151 RDNA3.5, the per-die variance contribution of:
   subnormal flush • FMA fusion • reduction order • MXFP4/BF16 rounding •
   wave-conflict handling • atomic ordering. Which gives the largest
   *consistent* per-die signature (not just per-launch jitter)?

4. Has anyone published commodity-GPU "physical reservoir computing"
   where the kernel itself IS the silicon and weights co-adapt? We know
   Joshi PCM and Romera STNO need new HW. Anyone done it with stock
   NVIDIA / AMD / Intel? Cite or "none found".

5. The "no SHUFFLE possible" claim — is that right? Construct *any*
   falsification control that would distinguish "weights co-adapted to
   chip-specific operator" from "model just got lucky / overfit". If
   you can't, the experiment is unfalsifiable and dead.

6. Right benchmark: stochastic gradient estimation, variational
   inference, reservoir computing, fine-tuning on perturbation-robust
   classification, something else? Pick the task where chip-specific
   operator behaviour is *load-bearing*, not merely tolerated.

7. Concrete HIP kernel design — minimum-viable kernel that:
   (a) uses atomic ordering + denormal handling differently per chip
   (b) is differentiable (straight-through or surrogate)
   (c) gives bit-divergent output on ikaros vs daedalus, same W, x
   Sketch the kernel in 20–40 lines of HIP/CUDA-ish pseudocode.

8. P(success) — your honest probability this finally crosses the
   constitutive gate on commodity GPU. Single number.

9. P(performance benefit) — if it works, does it *improve* task
   performance vs deterministic baseline (per-die optimal precision,
   free regularisation, ensemble effect)? Or only preserve it?

10. **Brutal honesty:** the obvious objection we haven't thought of
    yet. The thing that will make this fail on day two.

## Output format

For each numbered question: **2–6 sentences** maximum, with citations
where you make a factual claim. End with a single line:

    VERDICT: <go|pivot|kill> | P(success)=<0..1> | P(benefit)=<0..1>
