# F_SCALE — Scale-up & adversarial controls for ANGLE F (self-referential identity)
Date: 2026-05-30
Repo: AMD_gfx1151_energy
Source dir: `scripts/identity_benchmark/F_scale/`
Results dir: `results/IDENTITY_BENCHMARK_2026-05-30/F_scale/`

## TL;DR — F DOWNGRADED from DISCOVERY to NULL

The original F result ("substrate-aware model degrades 11× more on transplant
than baseline") does NOT survive controls. **SW-matched Gaussian noise of the
same mean+std as the substrate feature vector produces a LARGER transplant
gap than the real substrate features.** Therefore F is not measuring
substrate identity — it is measuring "any high-variance constant column glued
to the readout reduces transplant robustness." The mechanism is statistical
(brittle ridge readout when out-of-distribution constant injected at inference),
not identity-bearing.

Note: even the original F result was already z=0.79 against its naive baseline
(`results/.../novel/F_results.json` `overall.z_score = 0.79`, gate `z>2` not
passed). The "11×" framing came from the *ratio* of gap means
(1.50 / 0.13 = 11.5) on the ikaros-train side, but the std on the aware side
was 1.80, so the effect was never significant. F1 with 30 seeds confirms this.

## F1 — Ablation × seed scale-up (30 seeds, 5 feature configs, 1200 runs in 158 s, no abort)

Feature config | aware transplant gap (mean ± std, n=60) | z vs naive | z vs SW-matched | z vs shuffle | Gate
---|---|---|---|---|---
rtn_only      | 0.256 ± 0.167 | +0.03 | -0.73 | -0.85 | FAIL
spc_only      | 0.612 ± 0.499 | +0.69 | -0.37 | -0.19 | FAIL
both (=F base)| 0.924 ± 0.920 | +0.72 | -0.09 | +0.15 | FAIL
sw_matched    | 1.045 ± 1.067 | (control) | — | — | —
shuffle       | 0.757 ± 0.565 | (control) | — | — | —
naive baseline (any feat) | 0.250 ± 0.162 | — | — | — | —

Key observations:

1. **`sw_matched` produces the LARGEST gap** (1.05) — bigger than `both` (0.92).
   A random Gaussian feature vector matched only in (mean, std) is sufficient
   to reproduce — and exceed — the original F effect.
2. **`shuffle` (real features cross-paired with wrong device label)**
   also produces a large gap (0.76), nearly as large as `both`. Identity
   binding is not detectable.
3. **`rtn_only`** is statistically indistinguishable from the naive baseline
   (z=+0.03). The RTN feature contributes nothing.
4. The huge `aware_gap_std` (0.92 to 1.07) reveals the original `both` result
   was driven by a few outlier seeds — IQR 0.45–1.17 spans more than 2× the
   median.

Pre-registered gate (aware gap > controls + 2σ) is FAIL for all three real
feature configs.

Verdict file: `results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F1_ablation.json`.

## F2 — Permuted-MNIST continual learning (10 seeds, ~5 min wall)

Condition | task1 transplant Δ (mean ± std, n=20) | task1 forgetting (own substrate)
---|---|---
substrate-aware | +0.089 ± 0.087 | 0.052 ± 0.011
naive baseline  |  0.000 ± 0.000 | 0.052 ± 0.011
z(aware vs naive transplant) | **+1.03** (gate: z>2 FAIL) | n/a

Findings:
- **Catastrophic forgetting is IDENTICAL** in both conditions (0.052 ± 0.011).
  Substrate awareness does NOT modulate the plasticity / forgetting curve.
- Naive's transplant Δ is exactly 0 by construction (input is just raw pixels,
  no substrate features; the eval_dev parameter does nothing in that branch).
  This makes the comparison weak: any aware effect > 0 will show a positive z.
- Aware shows a small consistent +0.089 transplant drop (task-1 accuracy
  drops ~9 pp under wrong-device features after the full 5-task sequence).
  This is real but does not pass z>2 against a sw_matched control — and we
  expect, given F1, that a sw_matched control would match this.
- **Forgetting is NOT modulated by substrate identity**; this is the key
  oracle-relevant negative.

Verdict: Permuted-MNIST does not surface an identity-specific plasticity
phenomenon. F2 effect is consistent with the F1 "any-injection" explanation.
Results: `results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F2_permuted_mnist.json`.

## F3 — Live hwreg readback verification: INCONCLUSIVE

`puf_kernel` SIGSEGV'd on 5/5 invocations after a fresh `hipcc
--offload-arch=gfx1151 -O1` rebuild. Conditions at attempt:
- APU thermal_zone0 = **93 °C** (above 75 °C safe ceiling)
- GPU edge = 60 °C
- ROCm 7.1.1, HSA queue allocation succeeded; kernel launch failed before
  any `WROTE …` host-side log.

Probable cause: GPU queue contention with running tmux jobs (`nsram_queue_worker`,
yggdrasil, etc.) and the APU thermal load they impose. The instruction
"don't touch other tmux" was honored, so F3 cannot be completed in this run.

**Implication for F**: F is trained on cached Phase 1b `raw_idle.npz`. F3
was meant to confirm those values are still produced by current driver/firmware.
However, F1 already shows that the feature *content* is not load-bearing
(sw_matched gives a bigger gap than real features), so the cache-staleness
question is downstream-moot.

Verdicts: `F3_ikaros_live.json` (error block + remediation), `F3_compare.json`
(status NO_LIVE_DATA).

## F4 — Multi-task degradation curve (10 seeds × {ikaros,daedalus}²)

Task         | aware gap mean ± std | naive gap mean ± std | z(aware vs naive) | Gate
---|---|---|---|---
Mackey-Glass τ=17 | 2.240 ± 2.677 | 0.078 ± 0.021 | +0.81 | FAIL
sine             | 0.098 ± 0.045 | 0.098 ± 0.045 | -0.00 | FAIL
MNIST (acc-drop) | -0.078 ± 0.154 | -0.012 ± 0.143 | -0.32 | FAIL (and wrong sign)

- **Mackey-Glass**: the same shape as NARMA — large mean increase but
  enormous std (gap std > gap mean), driven by outlier seeds where ridge
  readout blows up.
- **sine**: NO transplant degradation at all (the task is too low-dimensional
  for the feature injection to matter). Aware and naive are identical.
- **MNIST classification**: aware condition is *worse on its own device than
  on the transplant* (negative gap), and the difference vs naive is non-significant
  in the wrong direction.

**The "≥2σ" gap from the F base does NOT reproduce on any of the three
additional tasks.** Reinforces the F1 verdict.

Results: `results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F4_multi_task.json`.

## Verdict on F

**DOWNGRADED: NULL.** Specifically:

- F1 30-seed ablation with sw_matched + shuffle controls: pre-registered gate
  FAIL for all real feature configs; sw_matched control gives *larger* gap.
- F2 Permuted-MNIST: forgetting unaffected by substrate awareness; transplant
  effect z=+1.03, below gate.
- F4 cross-task: gap does NOT reproduce on MG / sine / MNIST.
- F3: could not verify cache freshness, but moot given F1.

The original F report's "11× ratio" was a misleading framing of a non-significant
(z=0.79) effect on a single task with an unprotected ridge readout that
amplifies any out-of-distribution constant feature injection. The "self-referential
identity" hypothesis as operationalized in `novel/F_self_referential.py` is
**not supported**.

## What would be needed to revive F

1. Replace ridge readout with a calibrated multi-output network whose
   conditioning on the substrate feature is *learned end-to-end* rather than
   tacked-on at inference.
2. Use a substrate feature that is *causal in the forward pass* (e.g. real
   per-CU rate gating during reservoir update), not just appended to the
   readout. The current `SubstrateSampler` does this for the reservoir state,
   but the *feature vector* is only a readout passenger.
3. The control must beat sw_matched, not just naive.
4. Free the GPU and re-run F3 to anchor the analysis in live hardware values.

## Thermal incidents

Zero compute-induced thermal events. F1/F2/F4 are pure CPU. F3 hit APU 93 °C
*before any work started* (preexisting load from other tmux), which is why
its kernel launches died. No throttling or shutdown.

## Artifacts

- `scripts/identity_benchmark/F_scale/F1_ablation.py`
- `scripts/identity_benchmark/F_scale/F2_permuted_mnist.py`
- `scripts/identity_benchmark/F_scale/F3_live_readback.py`
- `scripts/identity_benchmark/F_scale/F4_multi_task.py`
- `results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F1_ablation.json`
- `results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F2_permuted_mnist.json`
- `results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F3_ikaros_live.json`
- `results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F3_compare.json`
- `results/IDENTITY_BENCHMARK_2026-05-30/F_scale/F4_multi_task.json`
