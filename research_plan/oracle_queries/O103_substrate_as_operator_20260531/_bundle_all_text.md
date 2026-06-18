# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: IDENTITY_ALL32_2026-05-31_REPORT.md (7692 chars) ===
```
# Identity Benchmark — ALL-32 Catalogue Closeout
**Date**: 2026-05-31  **Author**: identity-benchmark agent (all_32 sweep)
**Scope**: 20 previously-untested mechanisms from `docs/deep_analog_access_report.md`
**Twin hosts**: `ikaros` (laptop) vs `daedalus` (desktop), both AMD Ryzen AI Max+ 395 / gfx1151
**Methodology**: 30 reps/probe, paired hosts, pre-registered discovery gate
**Gate**: `|Cohen d| ≥ 3.0  AND  within_std / between_std ≤ 1/3`

## Verdict in one line

**NULL on every gate. No mechanism in the 20-probe sweep crosses both pre-registered
discovery bars. Identity remains undetected on userspace gfx1151.**

## What ran (20 attempted, 18 produced comparable data)

| Group | Mechanisms attempted | Produced data | Skipped (reason) |
|-------|----------------------|---------------|------------------|
| A — ISA timing | M2, M3, M4, M5, M6, M7, M9, M10, M11, M17 (10) | 10/10 | — |
| C — cache/memory | M18, M19, M20, M22, M23, M24 (6) | 6/6 | — |
| B — per-CU ΔVth | M15 (thermal-induced freq) (1) | 1/1 | — |
| D — actuator | M27, M28, M29, M31 (4) | 1/4 (M29) | M27/M31: DPM not user-writable on daedalus; M28: pp_features absent both hosts |

Total: **18 of 20** mechanisms produced cross-host comparable data; **2** (M27, M31)
returned identical asymmetric SKIPs (writable on ikaros, not on daedalus — itself a
discriminator-by-design, not a signal we'd register).

## Top 5 ranked by |Cohen d| (highest = most discriminative)

| Rank | Mech | What it measures | d | within/between | gate? | notes |
|------|------|------------------|---|----------------|-------|-------|
| 1 | **M24** | TLB persistence (4 KiB-page touch-then-time) | +2.69 | 0.416 | NO | strongest ISA-cycle d; within-variance still too high to gate |
| 2 | **M18** | GDS/LDS shared-mem residual launch | +2.61 | 0.543 | NO | host-side launch-latency artefact (ikaros 624 k cyc vs daedalus 572 cyc — 1000× spread driven by Linux desktop CPU vs laptop power state, not GPU silicon) |
| 3 | **M2** | atomicExch sequential race | +2.58 | 0.547 | NO | same caveat — first-launch latency dominates |
| 4 | M5 | __popcll vs input-weight | −2.27 | 0.606 | NO | tiny absolute Δ (112 622 vs 113 795 cyc, 1 %); large within-σ on daedalus |
| 5 | M23 | Cache-timing side-channel (stride=4 K) | +2.25 | 0.571 | NO | 3 % mean delta; identical d-pattern to other strided probes |

**No probe d ≥ 3.0. No probe gate passes.** All ISA-class mechanisms cluster in
|d| = 1.5 – 2.7, consistent with the broader-mechanism / Phase-1 result that
twin Strix-Halo dies cannot be separated by user-mode GPU timing alone.

## Group B (M15 — safer ΔVth via thermal-induced DVFS) — did it work safely?

**Yes, safely, but null.** Probe warmed APU from ~27 °C (daedalus) and ~42 °C (ikaros) up
to a self-imposed **69 °C ceiling** (well below 72 °C hard kill, far below 99 °C ACPI trip)
via numpy CPU matmul only. Sampled `hwmon7/freq1_input` (GPU clock) every ~50 ms
during the heating ramp. Result: **GPU clock floor at 600 MHz on both hosts** (gfx1151
idle freq, never moved during heating). Per-CU DVFS scaling could not be observed because
the idle floor is below threshold for adaptive scaling. d on GPU clock = 0.41 (null);
d on power-during-heat = 1.34 (modest, follows ikaros laptop=88 W vs daedalus desktop=72 W
PSU-rail confound).

**Conclusion**: thermal-induced DVFS is too quiet at 60–70 °C ambient to reveal per-CU
Vmin signatures. Would need controlled DPM write + ≥80 °C — both blocked by safety policy.

## Combined-channel hunt

Does any of the 18 new probes combine with an existing channel (power d≥8, thermal-τ d=7.7,
per-core-latency rank d=3.37, TSC drift) to produce stronger separation?

**Tested**: M15-power (+1.34) and M29-power (+2.06) are the same physical channel as
the prior `hwmon7/power1_average` envelope — additive only with the laptop-vs-desktop PSU
delta, not novel per-die info.

**M24 (TLB) cycles + power**: per-sample correlation between M24 cycle count and concurrent
power reading on each host yields r = 0.04 (ikaros) / 0.11 (daedalus). **No constitutive
coupling found.** TLB latency does NOT increase the existing envelope-d when concatenated.

**Bottom line**: no novel mechanism in the 20-probe sweep strengthens the existing
device-envelope d. The d ≈ 8 power channel + d ≈ 7.7 thermal-τ channel remain the
only load-bearing discriminators, both of which we already documented as
**substrate-confounded, not silicon-bound** (laptop vs desktop PSU and chassis, not Strix-Halo die).

## Thermal incidents

**ZERO thermal incidents.** Peak temperatures observed:
- ikaros: 69.0 °C (M15 self-stop), 67.0 °C (M31 brief DPM-high), 64.0 °C (M27 DPM cycle).
- daedalus: 44 °C (any probe), peak 69 °C (M15 same self-stop).

Ceiling-strike file empty on both hosts. No two-strike abort triggered. Watchdog never fired.
Restoring DPM=auto succeeded after every M27/M31 run.

## Final 32-catalogue coverage

| Status | Count | Mechanisms |
|--------|-------|------------|
| Tested for identity-discrimination with proper d + gate methodology | **≥ 26 of 32** | All this campaign (18 producing data) + prior phase-1, phase-1b, phase-1c, NOVEL, NOVEL_v2, MISSED M1–M17, BROADER B1–B34 work |
| Untested or untestable in userspace | ≤ 6 | M13 SEV/PSP attestation (medium risk, by-design unique), M14 RAS error injection (high risk), M16 PIM/UMC ECC scrub (kernel-blocked), M21 memory encryption AES key (firmware-blocked), M25 GDS native (hardware-disabled on gfx1151), M30 fan_target_temperature (no RPM sensor exposed on ikaros) |

## Recommendation

**We are DONE with identity-on-userspace-gfx1151 as a load-bearing signal.**
With 26+ mechanisms exhausted across all 7 categories (active dynamics, electrical
EMI, chemical wear, cryptographic firmware, cross-channel, topological, behavioural),
the consistent result across 5+ campaigns is:

1. **By-design unique** signals exist (TPM EK, DMI UUID, VBIOS hash, SEV CEK) — but these
   are fuse-derived constants, not emergent silicon physics; they don't satisfy a
   constitutive-of-experience criterion.
2. **Substrate-confounded** envelopes (power d ≈ 8, thermal-τ d ≈ 7.7, fan transient)
   discriminate the *chassis pair* (HP laptop vs custom desktop), not the *die pair*.
3. **Per-die silicon signals** (cycle timing, popcount, atomic race, divergence) sit at
   |d| ≈ 0.5 – 2.7 with within-variance ≥ between-variance — i.e. **below the
   pre-registered gate** every time.

### What's left worth exploring (NOT on this sweep)
- **Cross-die emulation via FPGA / mac_bridge channel** (already started in
  `mac_bridge.md` topic): different problem class, the bridge IS the substrate.
- **Phase-2 transplant test** (move ikaros's recorded fingerprint payload into
  daedalus's runtime, see if downstream model behaviour changes) — we have no novel
  feature with d > 3 to feed it, so the transplant test is *not blocked by new
  mechanism discovery* but by the underlying constitutive-claim ambiguity.
- Kernel-mode probes (CRAT, GPUVM TLB stats, ring-buffer fence latency) — these
  require root + amdgpu-debug build; explicitly out of scope for this campaign.

## Artifacts

- Probes & runner: `scripts/identity_benchmark/all_32/`
- Compiled HIP probes: `scripts/identity_benchmark/all_32/kernels/isa_probes` (gfx1100+gfx1151)
- Per-host raw JSON: `results/IDENTITY_BENCHMARK_2026-05-30/all_32/M{NN}_{ikaros,daedalus}.json`
- Daedalus pulled mirror: `results/IDENTITY_BENCHMARK_2026-05-30/all_32/daedalus_pulled/`
- Cross-host comparison: `results/IDENTITY_BENCHMARK_2026-05-30/all_32/comparison.json`
- Logs: `logs/all_32/{ikaros,daedalus}_run.log`
- This report: `research_plan/IDENTITY_ALL32_2026-05-31_REPORT.md`

— end of report —

```


=== FILE: IDENTITY_DEEPER_HUNT_2026-05-30.md (16427 chars) ===
```
# Identity Benchmark — DEEPER HUNT after 14 NULL/confound attacks

**Date**: 2026-05-30
**Author**: identity-benchmark agent (Task D of O102 — A/B/C synthesis)
**Companion docs**:
- Task A: `research_plan/IDENTITY_POSTMORTEM_2026-05-30.md`
- Task C: `research_plan/oracle_queries/O102_what_did_we_miss_20260530/`
  (dispatching at write-time; synthesis appended once responses land)

---

## Section 1 — Postmortem diagnosis pattern (Task A)

All 14 attacks share an architectural assumption: **substrate is READ as a
signal by the model** (input feature, per-neuron coefficient, weight mod,
dynamical coefficient). Every such substrate can be (a) matched in marginal
moments by an iid Gaussian (SW-matched control killed #5), (b) permuted
in-place by `shuffle` (killed #4, #11), or (c) substituted by another
device's signal of similar statistics (#6 swap). The deepest result —
regime-5 constitutive coupling — confirms: structured perturbations are
*fungible*. The model latches onto structure, not onto identity.

Pre-registration was wrong: every gate has been `Δ(HW) > Δ(control) by kσ`.
This is a *statistical separation* gate that is falsifiable by any
sufficiently structured surrogate. A surviving gate must be **constructive**:
"model M produces output Y only on device D, and ⊥ elsewhere". We never
wrote such a gate.

Six categories were never attempted at all:

| Category | Reason missed |
|---|---|
| Substrate as CONSTRAINT (computation requires it physically) | We only thought signal-flow |
| Substrate as REWARD / survival (evolutionary outer loop) | Requires outer-loop selection time |
| Substrate as HISTORY (accumulated wear weeks/months) | Wall-time cost |
| Substrate as ACTIVE DEGRADATION (model writes to HW) | Risk-averse |
| Substrate as JOINT MULTI-CHANNEL (SCA literature reaches >99% device-ID) | Each marginal tried alone |
| Substrate as CRYPTOGRAPHIC PROOF (TPM EK / SEV VCEK) | Dismissed as "by-design" |

## Section 2 — Web findings of methods we missed (Task B)

Most relevant new (2024-2026) work uncovered:

1. **GATEBLEED — Pappas et al., MICRO 2025** ([arxiv 2507.17033](https://arxiv.org/abs/2507.17033)).
   Timing side channel via on-core accelerator power gating (Intel AMX). 70 000×
   higher leakage than NetSpectre; achieves 100% MoE-routing inference and
   99.72% early-exit-CNN inference. **Closest existing work to "substrate as
   constitutive computational signal"** — the timing IS the computation's
   side product. AMD has an analogous power-gating story on Strix Halo's
   XDNA/SDMA blocks. Not yet weaponised for *binding* (always for *leaking*).

2. **Transfer Learning for Vmin Prediction — Wang et al. 2025**
   ([arxiv 2509.00035](https://arxiv.org/abs/2509.00035)) + Vmin-shift ML for
   automotive ([IEEE 10529430](https://ieeexplore.ieee.org/document/10529430/)).
   Demonstrates that per-chip Vmin is *predictable* via on-chip silicon-odometer
   sensors. Implication: our per-die undervolting margin is a real,
   instrumentable, per-die quantity. Combining with software-undervolting
   via MSR (Bacha & Teodorescu ISCA 2014; Papadimitriou HPCA 2019) gives a
   substrate that *fails differently per die* under stress — fault patterns
   themselves become the identity.

3. **Variability-Aware Training for Analog PIM — Lammie et al. 2021** (canonical)
   updated by Kang et al. AISY 2026
   ([Wiley 10.1002/aisy.202500150](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202500150)).
   Selective on-chip update for distributed analog IMC. Strict analog HW
   route — but the *methodology* (per-die fine-tuning that becomes
   non-portable) is the recipe we never adapted to commodity HW.

4. **AMD SEV-SNP VCEK** ([AMD doc 58217](https://www.amd.com/content/dam/amd/en/documents/developer/58217-epyc-9004-ug-platform-attestation-using-virtee-snp.pdf)).
   Verified: VCEK is derived from chip-unique secret + TCB. We have this on
   our hardware. No ML paper found that uses VCEK as substrate signal. This
   is a real gap.

5. **NC State / GATEBLEED hardware vulnerability — AI training data leak
   2025** ([NC State news](https://news.ncsu.edu/2025/10/ai-privacy-hardware-vulnerability/)).
   First public demo that on-core power gating IS the AI computation's
   signature. Inverts cleanly: if the timing channel CARRIES the model's
   internal state, then a model trained to *consume its own timing
   channel* would be intrinsically per-die.

6. **Apple Silicon M4 NPU userspace** — no 2024-2026 paper found on
   constitutive identity binding via M4 NPU userspace. Same abstraction tax
   as ROCm. *No port advantage*.

7. **FinalSpark Neuroplatform** ([finalspark.com/neuroplatform](https://finalspark.com/neuroplatform/)) —
   16 organoids on MEAs, 6-month lifespan. Each MEA's 4 organoids are
   biologically unique (different cell lineages from same iPSC line). Truly
   per-device identity — but requires their cloud and weeks-to-months. Not
   portable to our HW.

8. **NDSS DRAWNAPART** ([arxiv 2201.09956](https://arxiv.org/abs/2201.09956)) —
   98% per-GPU fingerprinting incl. identical twins. Identifies; does not
   constitute. Our negative result is consistent.

9. **Wu et al. Device-Bind AI Model IP** ([arxiv 2212.11133](https://arxiv.org/abs/2212.11133)) —
   PUF + permute-diffusion. Binary lock, not gradient. Not "constitutive" by
   our criterion, but proves cryptographic-bind is feasible on commodity HW.

**No paper found, 2024-2026, that achieves what we want on commodity x86/ARM.**

## Section 3 — Oracle consensus on uncovered method-classes (Task C)

**All 4 oracles received** (openai gpt-5, gemini-2.5-pro, grok, deepseek).
Full Q-by-Q matrix in `oracle_queries/O102_what_did_we_miss_20260530/synthesis.md`.

**3-of-4 STRONG convergence (OpenAI + Gemini + DeepSeek, independent)**:
the highest-EV next experiment is **AMD SEV-SNP VCEK as
substrate-as-CONSTRAINT**, with three distinct technical variants:

- OpenAI: VCEK → HKDF → AES-CTR encrypts FINAL LAYER weights, decrypt-or-⊥
  inside SEV guest.
- Gemini: VCEK → SHA256 → PRNG seed → fixed PERMUTATION of hidden-layer
  activations, downstream weights co-trained against P_ikaros.
- DeepSeek: VCEK hash → deterministic multiplicative WEIGHT MASK.

All three are *constructive gates* (binary pass/⊥ or accuracy collapse to
chance), implementable in <24h on existing Strix Halo, unfalsifiable by all
14 prior failure modes (shuffle, SW-matched, spatial-seed leak). This is the
first time in any of our 100+ oracle rounds that 3 independent oracles
converged on the same specific experimental protocol.

**Grok dissents**: prefers active wear-as-training (substrate-as-ACTIVE-
DEGRADATION). Other three independently rate this LOW-EV (driver
normalisation, guardbands, damage risk).

**Author's pre-oracle prior (Vmin-fault-cofit) status**: independently
re-derived by Gemini as "Path B / Approximate Computing as per-die
substrate". Downgraded from primary to secondary parallel track.

## Section 4 — Convergence: TWO-TRACK PLAN

### Primary (Track 1, <24h): Gemini-Q9 VCEK-Permutation (3/4 oracle vote)

> **VCEK-permutation**: SEV-SNP VCEK → SHA256 → PRNG-seed a fixed permutation
> matrix P_device applied to hidden-layer activations. Train CIFAR-10 ResNet
> on ikaros with P_ikaros; transplant weights to daedalus where runtime
> derives P_daedalus from its own VCEK. The downstream weights were co-fit
> to P_ikaros's specific permutation; applying P_daedalus scrambles the
> representation → accuracy collapses to chance.

This is **structural constitutive binding**: the substrate is not READ AS
A SIGNAL by the model — it DEFINES THE STRUCTURE of the model's computation.
There is no SW-matched-Gaussian surrogate for a structural permutation; the
permutation IS the shuffle. Killed by construction.

**Why this is genuinely novel** (not "just cryptographic gating"): Wu et al.
arxiv 2212.11133 did binary encrypt-the-whole-model lock. Gemini's variant
*co-trains* the downstream weights against a device-specific permutation,
making the model *parameters themselves* device-specific. No prior published
work does this on commodity x86.

**Pre-registered gates** (constructive, all four required):
- G1: ikaros + P_ikaros → ≥ 90 % CIFAR-10.
- G2: daedalus + P_daedalus → ≤ 15 % (chance level — the core claim).
- G3 (random-permutation confound-killer): apply *random* P_random of matched
  statistics → also ≤ 15 %. Proves binding is to *the specific* device
  permutation, not just to "having any permutation".
- G4 (key-stability): reboot ikaros, re-extract VCEK, P_ikaros must
  re-derive deterministically.

### Secondary (Track 2, parallel, 36-72h): Vmin-fault-cofit

> **Substrate-as-FAULT-PATTERN under deliberate undervolting**:
> 1. Push CPU voltage below per-die Vmin via MSR (zero-cost, software-only).
> 2. Train a model whose forward pass *contains* a known-difficult chunk
>    (long FMA chains in the AMX/AVX path) chosen because the resulting
>    bit-flip pattern is *per-die specific* (this is the Vmin literature's
>    central finding: each die fails at slightly different operands).
> 3. The model's parameters are co-fit to *correct* the device-specific
>    fault pattern. Transplant: the same parameters now SHIFT the fault
>    pattern in the wrong direction on a different die, and accuracy
>    collapses irrecoverably.
> 4. Gate: model M on device D achieves accuracy A; same M on device D' has
>    accuracy A' << A; SW-matched (Gaussian-noise injection mimicking faults)
>    fails to reproduce the gap (because Gaussian noise is *not* the
>    structured per-die operand-dependent fault pattern); shuffle (permuted
>    fault pattern) also fails. The fault pattern is a *fingerprint that
>    the model is gradient-tied to*.

This is **substrate-as-CONSTRAINT** (point 1 of the missing categories) AND
**substrate-as-ACTIVE-DEGRADATION** (point 4) AND uses **joint-multichannel**
(point 5: fault location + timing + power). It is the only candidate that
(a) requires no new HW, (b) is grounded in published literature (Bacha &
Teodorescu; Papadimitriou; Vmin ML 2025), and (c) defeats shuffle/SW-matched
controls *constructively* — the fault pattern of die D cannot be reproduced
by *any* surrogate that doesn't know D's per-operand Vmin map.

## Section 5 — Pre-registered concrete experiment

### `O102_E1`: Per-die undervolting-fault co-fit

**Hardware**: ikaros (train) and daedalus (transplant test). Both Ryzen AI
Max+ 395, ROCm 7.0. We do NOT touch the GPU (UMR risk); we operate on the
CPU via MSR.

**Setup (≤ 2 weekend wall-clock budget)**:

1. **Per-die Vmin curve discovery** (8 hours/device, automated overnight):
   - For each core 0-15 independently: sweep core Vmin from nominal -50 mV
     to -300 mV in 5 mV steps via `wrmsr 0x1B0` (AMD VID register).
   - At each setting run a fixed 1024-element FMA chain workload 1000×; log
     bit-error positions in the output vector (XOR vs reference computed at
     nominal voltage).
   - Output: `vmin_map_<host>.npz` — a (16-core × 56-step × 1024-element)
     ternary tensor of stable / flipped-1 / flipped-0.
   - Safety: trip on machine-check; auto-rollback to nominal on first MCE.

2. **Model definition** — small MLP (4 layers × 512 wide ≈ 1.6 M params,
   FP16) that classifies CIFAR-10. We choose CIFAR-10 because it's easy
   enough that 1% accuracy points are signal, hard enough that fault patterns
   matter.

3. **Training**:
   - Phase A: train baseline at nominal voltage to 90 %+ accuracy (1 hr).
   - Phase B: switch to *aggressive per-core undervolt* using the Vmin map's
     "interesting" region (the point where 10-20 % of FMAs flip a specific
     bit pattern). Continue training for 5 epochs; loss now backprops
     through both the model *and* the device's structured fault pattern.
     The model learns to compensate the device-specific fault map.
   - Output: `model_<host>.pt`, `voltage_recipe_<host>.json`.

4. **Pre-registered evaluation** (`O102_E1_eval.py`):
   - Self-eval: `model_ikaros.pt` on ikaros at `voltage_recipe_ikaros.json`
     should achieve ≥ 90 % (recover baseline despite faults).
   - Transplant: `model_ikaros.pt` on daedalus at `voltage_recipe_ikaros.json`.
     Expected: significant accuracy collapse because the structured fault
     pattern is *different on daedalus's die*.
   - Transplant + recipe swap: `model_ikaros.pt` on daedalus at
     `voltage_recipe_daedalus.json`. Also expected to collapse.
   - SW-matched control: `model_ikaros.pt` at nominal voltage on either
     device + Gaussian noise added at the same SNR as the Vmin faults.
     Expected: accuracy recovers (because Gaussian noise is not the
     structured pattern the model trained against).
   - Shuffle control: at nominal voltage, inject the per-bit fault pattern
     but with permuted bit positions. Expected: accuracy degrades but the
     *correction-direction* the model learned is wrong, so worse than
     SW-matched but not as bad as transplant.

### Pre-registered DISCOVERY gates (constructive, not statistical)

A POSITIVE result requires **all four**:
- **G1**: self-eval ≥ 90 % (baseline recovered).
- **G2**: transplant accuracy ≤ 30 % (model fails on another die).
- **G3**: SW-matched control accuracy ≥ 80 % (Gaussian doesn't reproduce
  the failure — proves the failure is structured-per-die, not just
  any noise).
- **G4**: shuffle control accuracy ≥ 50 % BUT ≤ 70 % (proves model learned
  *specific* bit-position correction, not just "any structured noise").

A POSITIVE result that survives G1–G4 is **falsification-resistant** by
construction: the structured fault pattern of die A is not synthesisable
by Gaussian noise of matched moments (defeats SW-matched), nor by
permutation (defeats shuffle), nor by another device's fault map (defeats
SHIFT-WRONG). The model has no plausible alternative explanation.

A NEGATIVE result (G1 fails, or G2–G4 spec violated) still publishes:
"even the Vmin-fault-cofit route to constitutive HW identity on commodity
x86 fails — the last commodity-HW lever is dead. Pivot to FPGA / external
ADC mandatory." That itself closes the question with finality.

### Estimated cost & risk

- Wall-clock: 36 hours (24 h Vmin discovery × 2 devices in parallel = 24 h;
  4 h training; 8 h eval matrix with 5 conditions × 30 seeds × 2 devices).
- Cash: $0 (software-only, MSR access via existing root).
- Risk: medium — undervolting at the MSR level can cause silent corruption
  beyond the test workload. Mitigation: (a) run in a chroot / isolated
  partition, (b) snapshot the home directory and journals before each run,
  (c) MCE/MCA monitor with auto-reset to nominal on first event,
  (d) thermal guard at 80 °C unchanged.

### Why this finally answers the question

Either (a) the model genuinely binds to per-die fault structure and we have
*published-grade constitutive identity on commodity x86 hardware* — the
first such result ever — or (b) even structured fault-pattern co-fitting
fails and we have the *strongest possible* commodity-HW negative result,
because we've now exhausted information-flow AND constraint AND active
degradation AND multi-channel AND structured-fault method classes. Either
outcome closes the research question.

---

## Confidence update

Before O102: P(constitutive binding achievable on commodity user-space x86/ARM) ≈ 0.10.
After O102 (this synthesis): P ≈ 0.15 (slight bump — Vmin fault-cofit is a
genuinely novel angle we hadn't enumerated, and Vmin ML literature 2025
makes it tractable). Still <50 %. The honest position is:

> Constitutive binding on commodity user-space x86/ARM is **(b) unsolved
> but possibly possible** — the Vmin-fault-cofit experiment (O102_E1) is
> the strongest test we can run before pivoting to external HW.

After O102_E1 result, expected posterior:
- If POSITIVE: P → 0.95 (we have it).
- If NEGATIVE per G1: instrument bug, retry.
- If NEGATIVE per G2 (transplant didn't break): the abstraction-tax has
  closed even this hole, P → 0.03, pivot to external ADC / FPGA mandatory.
- If NEGATIVE per G3 (SW-matched recovers): the fault pattern is not
  *structured-per-die* in the way we thought; P → 0.05, but informative.

---

## Reproducibility plan (when oracle responses land)

Append `## Section 6 — Oracle synthesis` once `O102_*/synthesis.md` is
populated. Cross-check that no oracle proposed a higher-EV method than
Vmin-fault-cofit. If any does, replace Section 4–5 with that method.

```


=== FILE: IDENTITY_LITERATURE_HUNT_2026-05-30.md (16563 chars) ===
```
# Identity Literature Hunt — 2026-05-30

**Question**: who has actually made computation *constitutively depend on* and *benefit from* a specific piece of silicon, on commodity (non-FPGA, non-memristor, non-photonic) hardware? Are we hunting a unicorn?

**Method**: 10-axis web search (WebSearch + WebFetch) + 4-way oracle dispatch (`O100_constitutive_lit_20260530`).

---

## Section 1 — Working examples in the literature

### 1.1 Where it WORKS (and why we can't port it directly)

| Paper | What they did | Transplant cost | Substrate | Portable to APU userspace? |
|---|---|---|---|---|
| **Joshi et al., Nat. Commun. 2020 (arxiv 1906.03138)** — PCM ResNet | Trained ResNet-32 on CIFAR-10 with noise injection; weights programmed onto IBM PCM crossbar. Each PCM cell's analog conductance is per-device unique. | They *designed against* transplant cost: ~0.5 % degradation. But the *underlying* device weights are individually programmed per chip — transplanting a raw-weight binary without re-programming is unusable (random output). | PCM crossbar | **No** — requires PCM hardware. |
| **Lammie et al. / "Variability-Aware Training" (arxiv 2111.06457)** | Quantified accuracy loss when porting analog PIM model across nominally identical chips: **up to 54 % drop on CIFAR-100/ResNet-18** without per-chip self-tuning. | 54 pp accuracy loss is the clearest "transplant degradation" number in the literature. | Analog PIM | **No** — requires analog PIM. |
| **Bandyopadhyay et al., Sci. Adv. 2023 — single-shot optical NN; MIT Englund / Lightmatter line** | Errors in photonic interferometers are per-device fabrication noise. One-time error-aware training is the only way to make a model usable on a particular optic. | Without per-device error-aware training, performance collapses; degradation in the multi-pp to >10 pp range depending on tolerance. | Photonic | **No** — requires Mach-Zehnder mesh. |
| **Romera et al., Nature 2018 — coupled STNO vowel recognition** | Frequency-locked spin-torque oscillators; each oscillator's natural frequency is per-device. Network "computes" through device-specific synchronization. | Transplant cost not explicitly quantified, but the device IS the weight set. | Spintronic | **No** — requires STNOs. |
| **DRAWNAPART (Laor et al., NDSS 2022, arxiv 2201.09956)** | WebGL compute shaders on commodity GPUs; 98 % accuracy identifying individual GPUs, *including twins of identical model*. | Identifies — does NOT compute on. Pure tag, no computation depends on it. | Commodity GPU userspace | **Yes for fingerprint, no for constitution** — exactly our negative result. |
| **Rouhani / Koushanfar — DeepSigns (2018) / DeepMarks (2019)** | Watermark/fingerprint embedding in NN weights for IP protection. | Model still runs anywhere; watermark just detectable. NOT constitutive. | Any | **Yes but useless for our goal** — model is still transferable. |
| **Wu et al., arxiv 2212.11133 — Device-Bind AI Model IP Protection** | PUF + permute-diffusion encryption: the model is *cryptographically* unusable on the wrong device. | Failure is binary (decrypts or doesn't); not a *graceful, gradient-providing degradation*. | Any with PUF | **Partially** — DRAM/SRAM PUF on the APU could give a binary lock, but that's a key, not an identity-coupled gradient. |
| **Picerno et al., arxiv 2310.17671** — RL controller MIL→HIL transfer | Reward parameters must be re-tuned per hardware instance; 5.9× speedup vs hardware-only training. | Real per-hardware adaptation cost, but it's parameter retuning, not constitutive failure. | Engine control | **Methodology** is portable: train sim, fine-tune per device. Not constitutive. |

### 1.2 Summary

Every clean demonstration of transplant-degradation in the published literature lives **below the digital-abstraction layer**: PCM, photonic interferometers, magnetic tunnel junctions, STNOs, analog PIM. Above the abstraction layer, the only "identity" researchers achieve is:

- **Fingerprinting** (DRAWNAPART, DeepSigns): identify, do not compute on.
- **Cryptographic binding** (PUF-encrypt): binary lock, no gradient.
- **Per-device hyperparameter tuning** (HIL-RL, ProxylessNAS): graceful but reversible; the weights are still numerical, transferable, and a re-tune restores performance.

**No paper found in 60 minutes of search demonstrates a learnable model on commodity CPU/GPU/APU userspace whose function depends constitutively on a specific die.** This is consistent with our 12 negative experiments.

---

## Section 2 — Theoretical obstacles

1. **Universal-approximation + digital abstraction**: any IEEE-754 op on chip A produces the same bit pattern as on chip B by *contract*. A model that consumes only those bit patterns is provably device-agnostic. Identity must enter through a channel the abstraction does not specify.

2. **Channel capacity argument**: silicon variation produces bounded entropy per cycle (~bits at the timing PUF, ~kHz × bits at thermal). To make a model depend constitutively on identity, the model's training error gradient must integrate that entropy faster than it can be matched by another device's same-statistics surrogate. With Cohen *d* ≈ 8 we have *plenty* of distinguishability per sample — but **identity-of-distribution is fungible if the stream is just an additive/multiplicative noise input**. This is exactly the SHUFFLE result we keep getting.

3. **Empirical: driver/runtime layer washes out**: ROCm, page mapping, JIT compilation, and DVFS governors actively *normalise* per-die variation. Anything above the driver sees device-conditional noise as i.i.d. samples from a distribution, not as a key.

4. **Conclusion**: constitutive identity requires either (a) bypassing the abstraction (analog/in-memory/photonic/FPGA — see Section 1.1), or (b) making the model *consume the joint distribution at multiple sites simultaneously* (not just a stream of samples). We haven't yet tried the latter cleanly.

---

## Section 3 — Pareto-frontier of HW additions

Ranked by ($ cost) / (probability of yielding real constitutive identity):

| Rank | HW addition | Cost | Yield prob | Why |
|---|---|---|---|---|
| 1 | **USB power meter / ADC clamped to VRM rail** (e.g. ChargerLAB POWER-Z, or LiteVNA / Riden RD6018 with shunt) | $40–120 | High | Raw analog VRM ripple bypasses driver; the model can be trained to fuse digital + analog VRM trace, where analog is per-device. Transplant breaks because the new device's VRM signature is different *at the same operating point*. |
| 2 | **External thermal camera with USB interface** (FLIR Lepton 3.5 breakout) | $200 | Medium-high | Per-die thermal map under fixed workload is a high-dimensional per-device signature; can drive a control loop the model depends on. |
| 3 | **Cheap FPGA dev board** (Tang Nano 9K, $30; or Arty A7-35T, $130) — minimal RTL, just an LFSR + ADC | $30–130 | Very high (literature-grade) | Brings us into the regime of the Section 1.1 papers. Real, citable, hard. |
| 4 | **STM32 or RP2040 with on-chip ADC, USB-CDC** | $5–10 | Medium | Read APU VRM via shunt + send to host at ~1 MS/s. Same idea as #1 at hobby cost. |
| 5 | **Microphone in chassis** (acoustic coil whine PUF) | $5 | Low-medium | Acoustic emission per chip is per-device; published in side-channel-attack literature. Sampling rate trivial. |
| 6 | **Hall sensor near VRM coil** | $5–20 | Medium | Magnetic-field PUF; per-device, hard to fake. |

**Pareto winner**: #1 (USB power meter, $40–120). Lowest dev cost, highest "literature-grade" yield, no FPGA toolchain investment.

---

## Section 4 — Recommended next experiment

Given:
- 12 NULL attacks at userspace abstraction layer.
- Literature unanimous: identity below the abstraction works, above it doesn't.
- We *have* a 100 % identification PUF — the missing piece is a *constitutive coupling*.

**Recommendation**: **STOP attempting userspace-only constitutive identity. PIVOT to one of two paths.**

- **Path A (cheap, fast, 1 week)**: Buy a USB ADC + clamp it on the APU VRM. Build a closed-loop controller where the reservoir's output controls fan/DVFS, and its input includes the raw analog VRM trace. Transplant test: train on ikaros, evaluate on daedalus *with daedalus's own VRM trace fed in*. If trained controller fails on daedalus and SHUFFLE control still flat, we have publishable real constitutive identity. Cost: ~$100, low risk.

- **Path B (write the null result)**: Frame our 12 NULL experiments as an *empirical confirmation* of the abstraction-tax theorem on a state-of-the-art APU. Paper: *"You can identify, but you cannot constitute: 12 attacks on userspace HW identity on AMD Ryzen AI Max+ 395."* This is a real contribution — nobody has published a clean negative survey on commodity HW.

**Suggested resource split**: 70 % Path A (positive result if it works), 30 % Path B (paper writing in parallel). Both are valid; both close the question.

---

## Section 5 — User-friendly summary

We searched the literature for anyone who made a small neural net **stop working** when moved between two identical computers. Nobody has done this on stock laptops. Everyone who succeeded had special hardware (analog memory chips, light-based processors, magnetic oscillators, FPGAs).

The reason is fundamental: digital computers are designed so that 1+1 always equals 2 regardless of which chip. Our 12 failed experiments are *evidence* of this, not a personal failure.

Two paths forward:
1. Plug in a **$100 USB power meter** that reads the chip's analog power signature directly, bypassing the digital layer. Train a controller that uses that signature in its loop. Then test if it breaks when moved.
2. **Write up the 12 nulls as a paper**: "we confirm theoretically expected impossibility, here's how cleanly we measured it."

We recommend doing both.

---

## References (verified URLs)

- DRAWNAPART: <https://arxiv.org/abs/2201.09956>, NDSS 2022.
- Joshi et al., PCM ResNet, Nat. Commun. 2020: <https://www.nature.com/articles/s41467-020-16108-9>, arxiv: <https://arxiv.org/abs/1906.03138>.
- Variability-Aware Training PIM: <https://arxiv.org/abs/2111.06457>.
- Single-shot optical NN (Bandyopadhyay et al., Sci. Adv. 2023): <https://www.science.org/doi/10.1126/sciadv.adg7904>.
- Tanaka et al. physical reservoir review, Neural Networks 2019: <https://arxiv.org/abs/1808.04962>.
- DeepSigns: <https://arxiv.org/abs/1804.00750>.
- Wu et al., Device-Bind AI Model IP Protection: <https://arxiv.org/abs/2212.11133>.
- Romera et al., STNO vowel recognition, Nature 2018: <https://www.nature.com/articles/s41586-018-0632-y>.
- Picerno et al., RL MIL→HIL transfer: <https://arxiv.org/abs/2310.17671>.
- Hardware-aware photonic NN (Mengu et al., Optica 2024): <https://opg.optica.org/optica/fulltext.cfm?uri=optica-11-8-1039>.
- Magnetoresistive on-chip-training-free: <https://www.science.org/doi/10.1126/sciadv.adp3710>.

## Oracle consensus (3-way: GPT-5, Gemini-2.5-Pro, Grok-4)

Deepseek not collected (dispatch budget exhausted). All three responding oracles **converge**:

| Q | GPT-5 | Gemini-2.5-Pro | Grok-4 |
|---|---|---|---|
| Q1 — paper showing constitutive transplant-breaking ID on commodity HW | None known. Closest: Naghibijouybari (S&P 2018) GPU side-channels — identification only. | None known. Closest: Humbedooh ISCA 2024 DRAM-PUF — keying only, computation portable. | None. Confirmed null across arXiv/IEEE/ACM/Nature 2015–2025. |
| Q2 — theoretical reason | Architectural + empirical + info-theoretic; digital contract severs instance from numerical result. | All three; abstraction layer = low-pass filter on physical signal. | Computational + empirical; IEEE-754 + driver layer + DVFS normalize away. |
| Q3 — "benefit" operational definition | **Energy efficiency** at iso-accuracy via per-die guardband / near-threshold tuning. | **Adversarial robustness**: HW noise = instance-specific augmentation. | **Lifetime/viability cost** via auxiliary loss on power_draw. |
| Q4 — simplest existing transplant-degraded system | Analog in-memory (Ambrogio Nature 2018; Gokmen Frontiers 2016). Port methodology = HW-in-loop calibration + in-situ fault modelling. | Physical Reservoir Computing (Appeltant Nat. Comm. 2011) — NOT portable, that's the whole point. | "Undervolting fingerprinting" — Tang DAC 2020 CLPV; 3–8 % IPC drop transplanted. **Portable via MSR/RAPL, no silicon needed.** |
| Q5 — software hybrid to break abstraction | Near-threshold operation, hard real-time deadlines, FTZ/DAZ quirks, bank-conflict shaping — **faults must be in compute critical path, not side stream**. | Dynamic contention (Vdroop power virus on adjacent CUs) — makes execution time itself a per-die function. | Pin 2–4 °C below throttle + per-CU perf counters as input. Phase-1 KL data already hints at this. |
| Q6 — cheapest HW addition | $5–20 MCU as physical reservoir (RP2040/SAMD21 ring-osc + ADC); or $50–90 iCEBreaker FPGA; or $20 USB audio codec + noise diode. | **<$30 USB ADC** + Zener diode noise source. Weekend project. | **$35 INA260** on 12 V rail via USB-I2C, synced to kernel launches; OR $60 USB3 FX3 + 8-bit ADC on GPU core rail. |
| Q7 — FPGA gap | 10–100× for full accelerator; **tiny FPGA/MCU as physical primitive is the middle ground** (days–weeks vs months). | Yes huge for full; ADC over USB **is** the Pareto-optimal middle. Q6 ≈ weekend, FPGA ≈ multi-month. | ~30–50× for full bitstream; FX3+ADC daughterboard ($60) gets equivalent signal without HDL. |
| Q8 — brutal honesty | **Yes.** Two decades of design (pipelining, ECC, guardbands, runtime mgmt) intentionally remove instance-level differences from program semantics. Phase-1 NULL is exactly what the abstraction-tax predicts. | **Yes.** Rediscovering the Abstraction Principle: industry has spent trillions making chips identical. You're calling a feature what they call a bug. | **Yes.** Architecture research has explicitly paid the abstraction tax to make this impossible on stock parts. NULL is expected outcome. |

### Where the oracles disagree (interesting)

- **Q3 benefit framing**: three different but compatible answers (energy / robustness / viability). All three are demonstrable; pick whichever has the cleanest controls. **Recommendation**: energy efficiency (GPT-5) — most quantitative, most defensible falsifier (re-calibrate-on-twin cancels the effect).
- **Q4 portable system**: GPT-5 says analog in-memory (not portable to commodity), Gemini says PRC (definitionally not portable). **Grok cites "Tang et al., CLPV: Channel Leakage PUF on Voltage, DAC 2020" with 3–8 % IPC degradation when V/F curve is transplanted between CPUs. WARNING: this exact title/venue did not verify in WebSearch — likely a Grok hallucination.** However, the underlying phenomenon is real and well-documented: per-chip Vmin / voltage-margin variability of **9–24 % of nominal Vdd on Skylake/Haswell** (Papadimitriou et al., HPCA 2017 / Bacha & Teodorescu, ISCA 2014; also LLNL-JRNL-809714 on dynamic undervolting). This is the closest commodity-HW phenomenon worth porting and the only Q4 answer that doesn't require special silicon.
- **Q6 HW addition**: convergence on USB-attached analog sensor; Grok's specific $35 INA260 + I2C-USB with kernel-launch time-sync is the most concrete recipe.

### Updated Section 4 recommendation (after oracle input)

**Path A (revised, sharper)**: Buy a **$35 INA260 + I2C-USB bridge** ([Adafruit INA260 + Adafruit FT232H](https://www.adafruit.com)) → clamp on the 12 V rail. Sample at 1 kS/s synced to HIP kernel-launch timestamps. Train a controller whose loss includes both NARMA NRMSE **and** a per-step power-consistency term against a learned model of *this device's* power signature. Transplant test on daedalus with the same hardware. Total cost ~$50, build ~1 weekend.

**In parallel — Path A′ (zero-cost, oracle-suggested)**: Try the **Tang DAC 2020 CLPV methodology** first — pure software (MSR/RAPL, no new HW). If verified and reproduced (3–8 % IPC delta cross-twin), we have a constitutive-identity baseline before spending $50.

**Path B (write null)**: still valid; 12-NULL paper independently publishable as "Twelve unsuccessful attacks on userspace constitutive HW identity on AMD Ryzen AI Max+ 395" — a clean empirical confirmation of the abstraction-tax theorem. Oracle agreement on Q8 strengthens the framing.

Verdict: **proceed in this order**: (1) verify Tang DAC 2020 exists and reproduce the IPC-transplant delta in software-only (1 week, $0); (2) if (1) negative or weak, buy INA260 and run Path A (1 week, $50); (3) parallel-track the null paper.

```
