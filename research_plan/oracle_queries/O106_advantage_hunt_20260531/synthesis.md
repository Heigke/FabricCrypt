# O106 Synthesis — Advantage Hunt for Envelope-Keyed Reservoir

**Date**: 2026-05-31. Two oracles consulted (gpt-5, gemini-2.5-pro).

## Consensus answer to Q5 (Is this a fool's errand?)
**Both oracles say NO, but our current method is incomplete.** The blocker is
*not* information-theoretic — the envelope contains real information about the
silicon. The blocker is that **hashing the envelope discards its semantic
meaning**. Hash → pseudo-random seed → unique-per-chip but NOT
adapted-per-chip structure. By No-Free-Lunch (Wolpert & Macready 1997) and
Ben-David's domain adaptation bound (arXiv:1002.3430), envelope bits that
are *independent of the data distribution* cannot lower expected error.

Gemini phrases it crisply: **"Stop hashing. Start mapping."**

## Convergent ranking of the 6 hypotheses
Both oracles converge on the same top candidates:

| Rank | Hypothesis | Both oracles agree |
|---|---|---|
| 1 | H4 / "per-position weight scaling from envelope" (per-CU latency → weight scale) | yes |
| 2 | H1 / "envelope-tuned sparsity" (power profile → density) | yes |
| 3 | H2 / "substrate as natural dropout" (RTN/jitter pattern → fixed dropout mask) | yes |
| 4-5 | H5 (live envelope as noise schedule) / H3 (LR schedule) — both weak; ridge is closed-form so LR is moot | yes |
| 6 | H6 (attention sparsity) — irrelevant to reservoir computing | yes |

## Critical reframing from oracles
**The D2 finding that "permutation is the dominant binder (920×)"** is the
biggest practical clue. Both oracles independently surface this. Gpt-5
recommends **"permutation-as-delay-line engineering"**: design the
permutation's cycle-length histogram to match the task's memory spectrum
(e.g., cycles of length 8–16 for NARMA-10) and map cycles onto CU/cache
groups via envelope so state mixing aligns with real routing latencies.

This is the bridge from "binding" to "advantage": use envelope to choose a
permutation that is BOTH (a) chassis-unique and (b) information-theoretically
aligned with the task structure.

## Closest literature
- **Rodan & Tiño 2011** (Minimum-Complexity Echo State Networks, IEEE
  TNNLS): permutation/cycle reservoirs are competitive with sparse random
  reservoirs for NARMA-class tasks. Directly relevant to D2 finding.
- **Lukoševičius & Jaeger 2009** (ESN review, CSR): classic guidance on
  spectral radius / conditioning / memory.
- **Tanaka et al. 2019** (Physical reservoir computing, Neural Networks 115):
  shows how physical non-idealities can be exploited.
- **Gal & Ghahramani 2016** (Dropout as Bayesian approximation, arXiv:1506.02142):
  envelope-derived dropout is only useful if it matches a task-relevant prior.
- **Wolpert & Macready 1997** (NFL theorems): foundational limit.

Both oracles note that **2024-2026 per-die specialization on commodity CMOS
GPUs is essentially absent from the literature**. All "exploits silicon
variability" work is in analog/memristor/photonic substrates where
variability is the substrate's signal. On digital deterministic CMOS, the
entire stack is engineered to *erase* per-die variability.

## Recommended next experiments (≤ 5)
From gpt-5 (verbatim, condensed):
1. **Permutation-as-delay-line**: envelope-keyed permutation with prescribed
   cycle-length histogram matched to NARMA-10's lag spectrum. Measure MC
   curve shift.
2. **Conditioning-driven scaling/leak**: choose per-neuron scales/leaks to
   minimize κ(XᵀX + λI) on calibration stream. **We tested this in F2 —
   FAILED**: equalization barely changed kappa, env_eq still ~3-22% worse
   than baseline_eq.
3. **Bandwidth-stressed regime**: sweep N at fixed wall-clock budget;
   envelope-tuned sparsity might Pareto-dominate at the latency edge.
4. **Structured dropout co-designed with permutation cycles**.
5. **Live multiplicative noise from thermal envelope** (training time only).

From gemini (additional):
- Use a **hyperparameter-tuned generic baseline**, not vanilla deterministic.
  This is the single biggest weakness of our current C5 baseline. If a 5-min
  Bayesian sweep over (density, spectral_radius, leak) gives a baseline that
  beats vanilla by 5-15%, then envelope must beat THAT to count.
- Use **Mann-Whitney U** with ≥20 seeds for significance claims.

## Verdict for our project
1. **F1 hypothesis sweep (5 hypotheses × 3 tasks)**: 0 / 15 C5 wins.
2. **F2 conditioning-driven equalization**: 0 / 3 C5 wins. Envelope-derived
   structures have systematically WORSE κ (~10^10) than baseline (~10^8) →
   confirms gpt-5's prediction that hash-derived structure is ill-conditioned.
3. The **permutation insight** is the most actionable: a NON-hash mapping
   from envelope features → permutation cycle spectrum is worth one more
   targeted experiment. We will document this as a future direction but not
   pursue in the current budget (already at the edge).

**Final scientific stance**: this is a **DOWNGRADE for the "performance
advantage" claim**, but a **FULL CONFIRMATION for the "binding" claim**.
The model is **chassis-bound but not chassis-adapted**. Publishable as a
clean negative result with constructive falsifier (random envelope tracks
actual envelope binding tightness on transplant but does NOT track
performance — proves envelope info is "structural" not "computational").
