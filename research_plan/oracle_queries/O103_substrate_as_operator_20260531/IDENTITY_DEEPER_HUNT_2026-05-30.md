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
