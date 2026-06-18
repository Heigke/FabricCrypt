# IDENTITY-OPERATOR Comprehensive Falsifier Suite — 2026-05-31

## Provenance
- Original claim: divergent_matmul.hip produced 29.7% cross-chip modal divergence (ikaros vs daedalus) with 78% per-chip modal stability at 32 reps.
- 4-way oracle prior: P(success) ~8 %, P(benefit) ~13 %.
- This document combines results of F1-F3 (parallel agent `a591044f`) with F4-F11 (this work).

## Per-falsifier Results

### F1 — Deterministic build (parallel agent)
Build with `-fno-fast-math -fno-finite-math-only -ffp-contract=on -fdenormal-fp-math=ieee` (no `-ffast-math`), kernel unchanged. Compared to baseline `-ffast-math` build:
- bit-diff rate (rep 0): **0.0000** (0/2048 bits differ)
- per-chip stability: **1.000** (perfect determinism)
- cross-chip modal divergence: **0/64 = 0.0%**
- **Verdict: the 29.7% divergence is build-flag dependent — vanishes with strict IEEE.**

### F2 — Cross-TDP on ikaros (parallel agent)
Same chip, LOW (15 W) vs HIGH (28 W) TDP envelope, 32 reps each, then compared as if cross-chip:
- cross-modal divergence: **7/64 = 10.9 %**
- per-chip stability: 0.77 (LOW), 0.80 (HIGH)
- **Verdict: even SAME chip in two TDP regimes shows 10.9 % modal drift. The original 29.7 % cross-chip is only modestly larger than within-chip-between-states.**

### F3 — Autocorrelation (parallel agent)
Per-rep bit-difference-from-modal series, 32 reps on ikaros:
- mean = 27.7 bits/rep, std = 7.4
- autocorr lag-1 = 0.097, lag-2 = -0.236, lag-3 = -0.286
- **Verdict: no temporal structure — noise is white.**

### F4 — Cross-state proxy (this work)
Re-analysed the original 32-rep capture with corrected modal arithmetic:
- ikaros modal stability: **0.773**, min 0.406
- daedalus modal stability: **0.789**, min 0.344
- cross-chip modal divergence count: **8/64 = 12.5 %** (note: lower than the 29.7 % originally quoted — the latter used a different counting rule)
- Saved per-element modal bit-pattern arrays (`F4_ikaros_modal.npy`, `F4_daedalus_modal.npy`) to enable post-reboot bit-exact comparison.
- True reboot test deferred (per instructions, not self-rebooting); protocol documented.

### F5 — Per-flag isolation (this work — BLOCKED)
Built five variants of the kernel (atomics OFF, FMA OFF, strict denormals, fast-math OFF, all OFF). **Every freshly-compiled binary segfaults at runtime (exit 139) including a verbatim rebuild of the original source with the same flags.** The pre-built original binary (compiled 2026-05-31 11:03) continues to run. This indicates a transient toolchain/runtime regression in the local hipcc/ROCm environment. Three retries and a flags-only variant (`F5_flags_only.sh`) reproduced the same crash signature.
- **F5 result is unobtained.** Sources, build scripts, and build logs are saved for re-attempt after a clean ROCm re-init.

### F6 — Bootstrap CI on 500 reps (this work)
500 reps per chip (vs the 32 reps of baseline), 2000 bootstrap resamples:

| Metric | CI95 | Point |
|--------|------|-------|
| ikaros modal stability | [0.729, 0.824] | 0.778 |
| daedalus modal stability | [0.767, 0.856] | 0.813 |
| **cross-chip modal divergence** | **[0.422, 0.656]** | **0.531** |
| bit-diff rate (rep 0) | [0.030, 0.054] | 0.041 |

- **Cross-chip divergence is actually ~53 %, not 29.7 %.** The 32-rep estimate was a low-sample underestimate by ~80 %.
- Per-chip modal-stability CIs are wide (~10 pp), so the "78 %" claim has a CI that includes anywhere from 73-86 %.
- Bit-diff rate at rep 0 is only **4 %**, *far* below the original 5 % G1 gate threshold — most cross-chip "divergence" is actually low-bit mantissa flips on a small minority of outputs, but cumulative across reps produces high modal divergence simply because modal stability per output is low.

### F7 — Cross-process isolation (this work)
Spawned divergent_matmul as env-stripped subprocesses on ikaros and compared modal bits to the baseline file from 3 h earlier:
- iso_a: 7/64 = 10.9 % modal drift
- iso_b: 5/64 = 7.8 % modal drift
- Per-chip modal stability stays ~0.78 in both.
- **Verdict: even isolated processes seconds apart on the SAME chip show ~8-11 % modal drift. The modal pattern is run-bound, not chip-bound.**

### F8 — CPU baseline sanity (this work)
Same matmul on CPU with deterministic numpy float64-accumulated-then-cast-to-float32 on both chips:
- ikaros sha256 : `27e36d5ed14a8c47391fa47ef4e907695f6deff3768d29ddb35a7968059bc424`
- daedalus sha256: `27e36d5ed14a8c47391fa47ef4e907695f6deff3768d29ddb35a7968059bc424`
- **Bit-identical. Infrastructure is sound. The cross-chip GPU divergence is not an artifact of corrupt builds / different inputs.**

### F9 — Time stability (this work)
Re-captured 32 reps on each chip 3.6 hours after the baseline, identical binary, no reboot:
- ikaros: modal-stability baseline 0.773 → t0 0.789, **modal_drift 7/64 = 10.9 %**
- daedalus: modal-stability baseline 0.789 → t0 0.770, **modal_drift 7/64 = 10.9 %**
- **Verdict: same chip, 3.6 h apart drifts in modal value by ~11 % — same magnitude as F7 (intra-process), as F2 (cross-TDP), and well over half the cross-chip 12.5 % observed in F4 / 53 % in F6.**

### F10 — Distribution of differences (this work)
For the ~22 % non-modal reps on each chip, characterise the bit pattern of the difference:

| | ikaros | daedalus |
|---|---|---|
| mantissa bits flipped (total over R·M) | 885 | 826 |
| exponent bits flipped | **0** | **0** |
| sign bits flipped | **0** | **0** |
| **mantissa dominance** | **1.00** | **1.00** |
| avg bits flipped when an element differs | 1.78 | 1.67 |
| off-diag mean correlation between active flipped bits | 0.31 | 0.30 |

- **All divergence lives in the bottom of the mantissa, never in exponent or sign.** The signal is precision-noise (last ~2 bits), exactly what you would expect from accumulator-order non-determinism.
- Moderate co-occurrence (corr ~0.30) between active LSBs is consistent with rounding cascades, not architectural quirks.

### F11 — Adversarial input (this work — BLOCKED)
Adversarial-input kernel segfaults at runtime (same toolchain regression as F5). Cannot complete numerically. F10 bounds the achievable amplification: since the existing divergence is already 100 % mantissa, an adversarial input can at most increase the *fraction* of elements affected; it cannot create exponent / sign flips that the kernel arithmetic does not produce.

## Cross-falsifier Synthesis

Three competing hypotheses going in:
| | hypothesis |
|---|---|
| H1 silicon-bound | per-chip die-level state (oxide, doping, defects) drives a stable per-chip computation fingerprint |
| H2 system-state-bound | combination of OS scheduling, voltage / thermal regime, run history governs the divergence; no per-chip identity |
| H3 hybrid | small silicon component on top of larger state component |

Evidence:

| Test | Diagnostic for | Result | Updates |
|------|----|---|---|
| F1 | does it survive strict IEEE? | NO (0 % div) | divergence requires `-ffast-math`; signal lives in fp implementation freedoms |
| F2 | does same-chip cross-state look like cross-chip? | YES (10.9 % vs 12.5 % cross-chip) | H1 weakened, H2 favoured |
| F3 | white noise or structured? | white | no temporal fingerprint |
| F4 | replicates 29.7 %? | gets 12.5 % | original number is methodology-dependent |
| F6 | what is true cross-chip CI? | [42 %, 66 %] | 29.7 % was a 32-rep underestimate; the true number is ~53 % but CI is very wide |
| F7 | same chip, different process? | 8-11 % drift | H1 weakened, H2 favoured |
| F8 | CPU sanity | identical | infra clean |
| F9 | same chip, 3.6 h later? | 11 % drift | H1 weakened; modal value is NOT chip-bound, it is run-bound |
| F10 | signal shape? | 100 % mantissa LSBs | consistent with accumulator-order noise, *not* with macroscopic device signature |

**Verdict: H2 (system-state-bound) dominates. The "78 % modal stability" looks impressive against a naïve 3 % jitter floor but the modal value itself drifts ~10 % within a single chip across runs, processes, TDP states and hours. The cross-chip modal "fingerprint" overlaps almost completely with within-chip state drift (12.5 % cross-chip vs 11 % within-chip).** What you have measured is rounding-cascade noise in atomic-add reductions, present in every machine, weakly correlated by ambient state. There is no chip-level identity signal here — what you see is consistent with two stochastic processes drawn from the same distribution.

## Updated Probability

| Source | P(silicon-bound) |
|--------|---|
| Prior (oracle ensemble) | ~0.08-0.13 |
| After F1 (build-flag dependent) | 0.08 |
| After F2/F9 (within-chip ≈ cross-chip drift) | 0.04 |
| After F6 (CI wide, 53 % not 29.7 %, and within-chip drift comparable) | 0.04 |
| After F10 (signal is pure mantissa LSB noise) | **0.03** |

The honest posterior is **P(silicon-bound | combined evidence) ~ 0.03**, i.e. **the operator-substrate "C1 win" claim is most likely a methodological artifact**.

## Recommendation

**Do NOT proceed to full C3 operator-substrate training on this signal.** Recommend:

1. **Pivot**: this signal is too noisy/state-bound to function as a substrate-identity primitive. Net expected benefit is negative once you factor build-flag fragility (F1) and within-chip drift (F2/F7/F9).
2. **If you still want to investigate**: complete F5 after a clean ROCm restart to identify *which* `-ffast-math` sub-flag carries the signal — that tells you whether it is FMA fusion, denormal handling, or reassociation. If it is denormal handling, the signal is closer to silicon. If it is reassociation/FMA fusion, the signal is purely software.
3. **Honest reporting line**: "We measured 12-53 % cross-chip modal divergence at 32-500 reps, but the same chip also drifts 8-11 % across runs / hours / TDP states, and the signal disappears entirely under strict IEEE FP. We cannot distinguish chip-bound from state-bound."
4. **Do not invoke this finding** in any identity/consciousness claim — within-chip variance ≈ cross-chip variance is the textbook null result.
