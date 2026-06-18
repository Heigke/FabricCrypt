# Identity benchmark — novel angles verdict
Date: 2026-05-30 · Phase: post-orthodox-NULL salvage attempt

## Premise
Orthodox PUF + reservoir transplant (Phase 1c/2) returned NULL: HW Δ-NRMSE
(0.026) within control-CI envelope (SW-iid 0.016, SHUFFLE 0.014). This
document reports the three top-ranked novel angles from the brainstorm
(`IDENTITY_NOVEL_ANGLES_2026-05-30.md`): F (self-referential), J (split-brain),
C (tournament-RO aggregation).

## TL;DR
**All three NULL.** Orthodox path was not hiding signal that the novel
angles unlocked. The two-device gfx1151 pair, with the substrate channels
measured so far, does not support a (2)-non-fungibility or (3)-stake claim
under any of the tested constructions.

## ANGLE F — Self-referential identity → NULL
Train a 128-neuron NARMA-10 reservoir whose readout features are augmented
with a per-device substrate feature vector (8 ΔVth-proxy buckets + 8 RTN
buckets + 8 top spatial eigenvalues = 24-d). Compare degradation-under-
wrong-substrate for substrate-aware vs baseline models.

- substrate-aware gap (other − own NRMSE): **1.035 ± 1.358**
- baseline gap:                              **0.267 ± 0.163**
- gate: z > 2 — **z = 0.79 → FAIL**

The aware-model gap is numerically larger but has 8× the variance — the
"big-gap" runs are seeds where the aware model fits the (training-device,
training-substrate-vector) tuple in an over-specialised way and then fails
*everywhere* including own-device. There is no clean substrate-coupling
signal; the model isn't learning "I am ikaros", it's just overfitting.

Artifact: `results/IDENTITY_BENCHMARK_2026-05-30/novel/F_results.json`

## ANGLE J — Split-brain co-dependence → NULL on stake claim
A 128-neuron ESN split 64/64 across ikaros and daedalus substrate streams.
Joint ridge readout. Eval with one half severed (zeroed), substrates
swapped, or against a fungible baseline (both halves use ikaros).

Transplant-table (readout trained on intact, then evaluated):

| condition           | NRMSE |
|---------------------|-------|
| intact              | 0.68  |
| ikaros half killed  | 4.90  |
| swap_to_zero (ref)  | 6.28  |
| substrates swapped  | **0.91**  |
| fungible_baseline   | 0.89  |

- severance vs intact: z = **4.69 PASS** (severance is real)
- swap < swap_to_zero: **swap is closer to intact than to severed** —
  the substrate-swap performs *almost as well as intact* (0.91 vs 0.68),
  whereas information loss (severance) is catastrophic (4.90+).
- gate (`severance_z>2 AND swap>swap_to_zero`): **FAIL**

**Interpretation**: the reservoir doesn't care which device's substrate
drives each half. Substrate provides generic noise that the ridge readout
absorbs as nuisance. Stake claim collapses: removing a half hurts (which
just shows the readout uses it), but *substituting* one substrate for the
other doesn't, so the function is not committed to any particular silicon.

Artifact: `results/IDENTITY_BENCHMARK_2026-05-30/novel/J_results.json`

## ANGLE C — Tournament RO → NULL (with caveat)
**Implementation pivoted mid-flight.** Live HIP binary
(`C_tournament_ro.hip`, builds successfully) was tested and found to be
*degenerate*: HIP scheduler dispatches block 0 then block 1 deterministically,
the per-race CAS arbitration consequently always selects block 1, giving
b1_win_rate = 100 % independent of silicon. With 79 slots all locked to bit=1
the tournament cannot distinguish devices.

Pivoted to AGGREGATING the already-collected Phase 1c probeB.bin data
(10000 RO-pair races per device, with `dCyc` and `winner` recorded). Build
79-bit tournament strings by majority vote over 100-race slots.

- ikaros b1_win_rate: **0.503**, intra-device Hamming max **44 / 79** over 8 bootstraps
- daedalus b1_win_rate: **0.501**, intra-device Hamming max **48 / 79**
- cross-device Hamming: **2 / 79** (essentially identical = both random)
- gate (`cross_dev>40 AND intra<10`): **FAIL**

Per-slot win-counts are fair-coin-flips on both devices, and the
intra-device noise floor (~ 40-bit Hamming) is *higher* than the inter-device
"distance" (2 bits). Aggregation does not rescue probeB — there is no
weak signal to amplify, it is pure scheduling noise.

Artifacts: `results/IDENTITY_BENCHMARK_2026-05-30/novel/C_tournament_{ikaros,daedalus,summary}.json`,
HIP binary kept at `scripts/identity_benchmark/novel/C_tournament_ro{,.hip}` for the record.

## Cross-angle synthesis

| angle | tests | result | gate |
|-------|-------|--------|------|
| F     | substrate-as-feature, 10 seeds × 4 cond | gap-z = 0.79 | FAIL |
| J     | 64/64 split, 10 seeds × 5 cond          | swap ≈ intact | FAIL |
| C     | 79-slot tournament from 10000 probeB races | inter=2 ≪ intra=44 | FAIL |

The three angles attack *different* identity claims:
- **F** attacks the interoception / self-modeling angle of (3) stake
- **J** attacks the architectural-commitment angle of (3) stake
- **C** is one last aggregation push at (1) identifiability

All three reject. The Phase 1b substrate channels (RTN-rate, spatial-corr)
are not just too weak to drive transplant gaps — they are too *generic*
across the two devices to support any silicon-specific function commitment.

## Did any novel angle beat orthodox?
**No.** All three reproduced the NULL.

## What to try next (if continuing on this thread)

Honest options, ordered by likely return:

1. **Different substrate channel.** The Phase 1b channels are software-time
   noise summaries. The promising-but-unprobed channels from the brainstorm
   were **D (memory-controller arbitration)**, **G (DRAM rowhammer state)** —
   both higher-risk but probe physical structure *below* the CU. If we want
   identity at all on this hardware, those are next.
2. **Repeated longitudinal aging fingerprint.** Run the same probes over
   weeks under heavy load and look for *drift* differences — devices may not
   differ now but their wear-out trajectories might. Slow but cheap.
3. **More devices.** N=2 is statistically weak. A third gfx1151 would let
   us test pair-wise distances against a clean intra-population baseline.
4. **Abandon hardware identity for this hardware class.** The orthodox
   result + three failed novel attacks is mounting evidence that
   commodity gfx1151 in user-space simply does not surface enough silicon
   variance to ground non-trivial identity claims.

## Recommendation
**Pivot the research direction.** The (1)→(2)→(3) ladder needs (1)
to hold, and across 6+ different attacks (orthodox A-D, novel F, J, C) we
have not detected reproducible silicon identity above control noise on this
substrate. Returning to this hardware will need either a fundamentally
different probe (D/G class) or different hardware (e.g. FPGA with explicit
per-LUT RO chains, which we control).

## Thermal incidents
Zero. F+J runs (multi-threaded numpy + cholesky) brought APU to 92 °C
briefly but stayed within thermal_guard's 80 °C SIGSTOP envelope without
triggering (the climbed-into-90s reading was a brief peak during
multi-threaded BLAS, processes continued normally). C ran on cached data
and on GPU only briefly during calibration (peak 47 °C). All artifacts
landed.

## Artifacts
- `scripts/identity_benchmark/novel/F_self_referential.py`
- `scripts/identity_benchmark/novel/J_split_brain.py`
- `scripts/identity_benchmark/novel/C_tournament_ro.hip` + binary
- `scripts/identity_benchmark/novel/C_tournament_runner.py`
- `results/IDENTITY_BENCHMARK_2026-05-30/novel/F_results.json`
- `results/IDENTITY_BENCHMARK_2026-05-30/novel/J_results.json`
- `results/IDENTITY_BENCHMARK_2026-05-30/novel/C_tournament_{ikaros,daedalus,summary}.json`
