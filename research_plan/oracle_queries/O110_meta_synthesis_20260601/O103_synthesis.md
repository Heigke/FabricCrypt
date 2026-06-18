# O103 — Substrate-as-Operator: 4-oracle synthesis

## Verdicts

| Oracle | Verdict | P(success) | P(benefit) |
|---|---|---|---|
| OpenAI GPT-5 | pivot | 0.18 | 0.25 |
| Gemini | pivot | 0.05 | 0.20 |
| Grok | **kill** | 0.07 | 0.04 |
| DeepSeek | **kill** | 0.02 | 0.01 |

**Mean P(success) = 0.08. No oracle endorses "go". Two say kill.**

## CRITICAL: DeepSeek's prediction was empirically falsified

DeepSeek's brutal-honesty assertion (Q10):
> "Your proposed kernel will produce **bit-identical results** on ikaros
> and daedalus. The FPU, atomic unit, scheduler are fixed RTL; subnormal
> flush, FMA fusion, rounding mode are uniform across all instances of
> the same stepping."

**C1 smoke (run 5 min before DeepSeek replied) measured the opposite:**
- 29.7% of output elements bit-differ between ikaros and daedalus
- Per-chip stability 78% (each chip prefers its own value across 32 reps)
- 12.5% of outputs have **different modal bit-patterns** cross-chip

So at least one oracle's underlying model of HIP atomicAdd + bank conflict
+ FMA semantics is wrong about what bit-identity means in practice. The
**non-deterministic ops are doing what they say** (different reduction
orders per chip), but oracles unanimously suspect this is per-launch
jitter not per-die signature. **The 78% stability number is the
debate-settler**: it's not pure jitter (would be ~1/N modal frac).

## Consensus

### 1. Is the reframing genuinely new? **Split: 1 yes, 2 no.**
- Gemini: "genuine architectural shift" but "razor-thin gap" between
  guaranteed-deterministic and per-run-stochastic — chasing scheduler
  analog noise, not silicon logic.
- OpenAI: "real shift in framing" but the levers are
  architecture-/compiler-level not die-level → will recreate
  Δ HW ≈ Δ SHUFFLE in a new guise.
- Grok: "rebranding the same dead end." Surrogate kernel can mimic any
  per-die statistic.

### 2. ROCm bit-determinism. **Strong consensus: NOT default.**
All three confirm rocBLAS/MIOpen/hipBLASLt are non-deterministic by
default (atomics, Tensile heuristics, Find-mode kernel selection).
But **per-arch** not necessarily **per-die**. Our two chips are same
gfx1151 — confidence in stable per-die signature is LOW.

### 3. Ranking of IEEE freedoms for per-die signature
Consensus (all three roughly agree):
1. **Atomic ordering + wave-conflict handling** — most per-die promise
2. Reduction tree depth (occupancy-dependent)
3. Subnormal flush (likely fixed per arch)
4. BF16 tie-breaking, FMA fusion (~zero per-die variance — ISA-fixed)

### 4. Prior commodity-GPU operator-substrate work. **All three: NONE FOUND.**
Confirmed against our IDENTITY_LITERATURE_HUNT_2026-05-30.md result.

### 5. "No SHUFFLE possible" claim. **CRITICAL: all three say WRONG.**
Falsification controls exist:
- **OpenAI**: (a) compile same kernel with deterministic reductions
  (-ffp-contract=off, serial K, no atomics) — purported binding should
  vanish; (b) emulate randomized reduction trees with fixed PRNG on
  "wrong" device — if it rescues perf, you learned algorithmic
  non-det not die identity; (c) swap driver — if accuracy moves with
  operator not chip, dead.
- **Gemini**: "surrogate-operator control" — kernel with *different*
  source of non-det but matched statistics. If native model
  outperforms surrogate on same die, evidence of co-adaptation.
- **Grok**: LLVM-emulated reduction tree that mirrors per-die map. If
  weights succeed on emulator, operator was never constitutive.

**This is the most actionable feedback.** Our pre-registered G5 ("no SW
control possible") is FALSE. We MUST add a "deterministic-kernel" arm
to C3 as the falsifier. If model trained on divergent kernel performs
identically on deterministic kernel transplanted, we learned noise
tolerance not chip binding.

### 6. Right benchmark. **Convergent suggestions:**
- OpenAI: **DEQ / implicit layers / fixed-point solvers** (Anderson,
  Broyden). Tiny biases move fixed points, not just add noise.
  Chaotic rollouts (NARMA, Lorenz).
- Gemini: **Variational inference / MCMC** — noise IS the feature.
- Grok: perturbation-robust classifier with explicit per-die loss term.

Best bet: **DEQ or Lorenz rollout** — sensitive-dependence-on-IC means
chip-specific biases compound rather than average out.

### 7. Brutal honesty (Q10) — all three sound alarm bells:
- OpenAI: "ROCm/LLVM updates will change the operator out from under
  you, destroying binding day one. Likeliest postmortem:
  'algorithm-choice regression in Tensile/MIOpen' not 'silicon identity'."
- Gemini: "signature is not bound to silicon, bound to entire system
  state (ROCm version + kernel + scheduler + X11 + TDP). Ephemeral
  emergent property, not stable die fingerprint."
- Grok: "driver and compiler already normalise the very IEEE freedoms
  you intend to exploit; once hipBLASLt or codegen is updated, the
  chip-specific operator disappears on both dies simultaneously."

## Decision

**PIVOT, not kill.** The C1 measurement we already ran shows:
- 29.7% of output elements bit-differ between ikaros and daedalus
- 78% per-chip stability across 32 reps
- 12.5% have *different modal bit-patterns* across the two chips

This is real signal — but the oracles correctly flag that it may be
"per-system-state" not "per-die". The right next step is the
**falsifier suite from Q5 BEFORE running C3 training**:

1. Compile a deterministic variant (`-ffp-contract=off`, no atomics,
   serial reduction) and check if the per-chip divergence vanishes
   → if YES on both chips, we know the divergence IS in the
   non-deterministic ops (good).
2. Run identical kernel on ikaros at two different TDP states (28W vs
   54W) — if cross-state divergence on same die matches cross-die
   divergence, the signature is power/clock-bound not silicon-bound
   (Gemini's worry).
3. Only if (1) and (2) survive: dispatch C3 full training with a
   "surrogate-noise" control arm.
