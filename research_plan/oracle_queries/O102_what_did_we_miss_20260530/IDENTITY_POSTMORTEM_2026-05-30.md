# Identity Benchmark — POSTMORTEM of 14 NULL/Confound Attacks

**Date**: 2026-05-30  **Author**: identity-benchmark agent (Task A of O102)
**Scope**: diagnose failure mode of each prior attack and surface the meta-pattern.

---

## Per-attack diagnosis (one paragraph each)

### #1 Stable-bit PUF (intra-HD 0.27, inter-HD 0.30, gate fails)
**Failure mode: FUNDAMENTAL (driver-level homogenisation).** LDS is zero-initialised
from ROCm 6.3 onward; chained-FMA-LSB is bit-exact across reps and devices. The
abstraction layer of HIP+amdgpu actively eliminates the entropy this method
relies on. No parameterisation fix exists — the signal is *contractually*
removed before user space ever sees it. We were chasing a phenomenon that the
runtime is engineered to suppress. The gate could only be crossed by reaching
*below* the runtime (UMR/MMIO/microcode).

### #2 1/f knee (within-device CI overlap)
**Failure mode: PARAMETERISATION + CONFOUND.** Cache-eviction-latency PSD is
dominated by OS scheduler quanta and interrupt timing, not silicon. The knee
sits at a frequency band where Linux scheduling jitter has 10× the variance
of any per-die contribution. A different window (sub-µs cycle counter inside a
single CFS quantum, no syscalls) might survive — but we'd be measuring
LSB-of-cycle-counter under bus arbitration, which is also platform-uniform.
Method-class is plausible; instrumentation needed to push below the kernel.

### #3 RTN + spatial-corr (initial PASS, falsified by Phase 1c)
**Failure mode: CONFOUND (thermal).** Initial intra-HD 0.10 / inter-HD 0.40
looked clean. 4/4 oracle vote (O95) flagged thermal-Arrhenius (RTS trap
kinetics scale 2-3× per decade per 10°C; ΔT~15°C between idle baselines
trivially reproduces "device fingerprint"). Phase 1c thermal-matched probe
confirmed: when both devices were held at identical Tj, the signal collapsed.
Method failed not because RTN doesn't exist, but because *ambient thermal
state* was the actual signal source. A future attempt would require
millikelvin-stable cold-plate + multi-hour soak — i.e., facility-grade, not
desk-grade.

### #4 Transplant matrix v1 (Δ HW 0.026 ∈ control CI)
**Failure mode: ARCHITECTURAL.** The substrate was injected as an *additive
feature* into a ridge-readout reservoir. Ridge regression's null-space absorbs
constant or low-rank perturbations — it literally cannot "depend" on a slowly
varying input column in a non-fungible way. The model was given the signal but
was not architected to need it. Identity-as-information ≠ identity-as-operator.
A fix would be: substrate enters as a *multiplicative weight modulation* or
*dynamical-system coefficient*, not as a regressor input. (We later tried
this in `IDENTITY_CONSTITUTIVE_2026-05-30` — see #11.)

### #5 F self-referential (sw_matched 1.05 > both 0.92 > shuffle 0.76)
**Failure mode: CONTROL-FAILURE.** The "11× gap" was reproduced *more strongly*
by an iid Gaussian with matched first two moments. This means the model is
sensitive to *any* high-variance constant column, not to identity. F failed
because brittle-ridge sensitivity was misread as identity coupling. The
positive predictive value of any "gap" metric is destroyed by the SW-matched
control. Lesson: identity claims demand controls that match marginal
statistics, not just shuffles.

### #6 J split-brain (severance_z 4.69 BUT swap < swap_to_zero)
**Failure mode: PARTIAL-SUCCESS, but WRONG NARRATIVE.** Severing the substrate
channel hurts — but swapping to *another device's* substrate hurts *less* than
swapping to *zero*. Information is being used; identity is not being defended.
This is the "perfect calculator with structured noise" interpretation: the
substrate is consumed as a generic statistical regressor, not as a key. To
break through, the loss function itself would have to penalise *another
device's* substrate specifically (impossible without an oracle telling the
loss which device produced the signal — which is question-begging).

### #7 C tournament-RO (cross-HD 2, intra-HD 48 — opposite of expected)
**Failure mode: FUNDAMENTAL (RDNA3.5 scheduling).** RO chains on a modern GPU
are not user-accessible at lane-level granularity. The "race" outcomes are
dominated by HSA-queue scheduling order and per-wavefront barriers, both of
which are deterministic given identical workgroup IDs. Cross-device the same
schedule produces near-identical outcomes; within-device, microsecond-scale
queue contention dominates. This method requires explicit RO instantiation
in RTL — i.e., FPGA, not GPU.

### #8 B Lorenz per-CU trajectory (ratio 0.185)
**Failure mode: FUNDAMENTAL (IEEE-754 contract).** Float32 RK4 is bit-exact
across same-ISA chips. The only inter-device variance is FP-ordering inside
reductions, and that's platform-uniform. Method was a category error: chaotic
dynamics amplify *initial-condition* differences, but the initial conditions
are bit-identical too. To get device variance, the chaotic system would have
to ingest a real analog perturbation per step — but that's just relabelling
the substrate-injection problem.

### #9 ECC bad-blocks (0 controllers on either device)
**Failure mode: PLATFORM-FALSIFIED.** Unified LPDDR5X on Strix Halo APU is
not exposed via EDAC at all. The probe couldn't even *acquire* the signal.
Method itself is sound on EDAC-enabled platforms (server EPYC, Threadripper),
but irrelevant here. Cross-platform porting cost is non-trivial.

### #10 Transplant v2 envelope (5 mixed channels, d≥8 but envelope-confound)
**Failure mode: WRONG ABSTRACTION LAYER.** The 5 channels (power AR(1),
thermal-τ, per-core-latency rank, TSC drift, RTN) all have huge per-device
effect sizes — but they live at the *device envelope* (board + cooling +
chassis + crystal), not at the *die*. Identity-binding to envelope features
is real but trivially defeated by swapping coolers / boards / power supplies.
The method works (large d, stable) but the wrong question was asked: we want
*silicon* identity, not *system* identity.

### #11 Constitutive 5-regime (regimes 3/4/5 weak-discovery but shuffle ≈ HW)
**Failure mode: STRUCTURAL, not identity.** Substrate became load-bearing
(Δ HW > Δ SW-matched) when injected at per-neuron-leak / weight-mod /
dynamical-coefficient sites — but `shuffle` (permuted same-device dims)
disrupted as badly as device-swap. The model latches onto *structure*
generically, not onto *which device's* structure. This is the deepest negative
result: even when we win the constitutive battle, the identity question
remains because any structured perturbation is fungible with any other.

### #12 A1 contrastive alone (z=0.79)
**Failure mode: METHOD WEAK ALONE.** Contrastive InfoNCE loss on (device,
trace) pairs forms the *learning* side of identity-binding but lacks
discrimination unless paired with a *discriminative* probe in the *physical*
domain. Score effectively zero alone.

### #13 A3 heavy-tail alone (marginal)
**Failure mode: METHOD WEAK ALONE.** Heavy-tail (α-stable) regression on the
substrate distribution is more powerful than Gaussian-matched controls but
still vulnerable to permutation. Tail shape is *device-specific*, but the
*identity of which sample came from which tail* is fungible.

### #14 A1+A3 combined (z=5.74 initial → FALSIFIED via spatial-seed leak)
**Failure mode: METHODOLOGICAL ERROR (information leak).** The initial
z=5.74 was generated by a spatial seed derived from `hash("ikaros")`
implicitly leaking the train/test split label into the model. F3 re-run with
sklearn z=1.62 (vs numpy z=5.74) plus controlled seed proved the discovery
was confounded. Even our best result was a data-leakage artefact. Critical
lesson: any non-numpy randomisation must be seeded *independently* of host
identity.

---

## The meta-pattern

Across all 14 attacks, the architectural assumption is identical:

> **The model READS the substrate AS A SIGNAL** — through an input feature,
> per-neuron coefficient, weight modulation, or dynamical coefficient. The
> substrate flows *into* the computation as data.

In every case, the substrate-as-signal can be replaced by:
- a Gaussian with matched moments (SW-matched control kills #5)
- a permuted version of itself (shuffle kills #4, #11)
- a different device's signal of similar statistics (swap kills #6)

**The method-class we never tested**:

1. **Substrate as CONSTRAINT** — the computation cannot proceed *at all*
   without the device-specific signal: a hard physical lock, not a soft input.
   E.g., the computation requires a hash collision against the device's TPM
   EK, or it requires consuming a specific physical entropy budget per
   forward pass that only this device's VRM can supply within deadline.

2. **Substrate as REWARD / SURVIVAL** — the model is selected by an outer
   loop where surviving (=not throttling, =staying under a power budget,
   =not crashing) on *this* device is the fitness function. Identity is
   bound by evolutionary pressure, not by gradient.

3. **Substrate as TEMPORAL HISTORY** — the model's weights are the
   accumulated *integral* of per-device wear/aging over weeks or months.
   Two devices started with identical weights would diverge irreversibly via
   their NBTI/electromigration history. This is genuine identity-as-trajectory,
   not identity-as-snapshot.

4. **Substrate as ACTIVE DEGRADATION** — the model *itself writes* to the
   substrate during training (deliberately stresses specific cache sets,
   triggers thermal cycling, performs targeted undervolting). The induced
   wear becomes a co-trained parameter; cannot be lifted without copying
   the device.

5. **Substrate as JOINT MULTI-CHANNEL** — none of the prior 14 tried
   *simultaneous* power + EM + timing + thermal + acoustic + magnetic
   fingerprint as the joint input. SCA literature (EM-X-DL, Picek et al.)
   reaches >99% per-device ID on the joint distribution; we have only
   tried marginals. Could push out of the "fungible structured noise"
   trap.

6. **Substrate as CRYPTOGRAPHIC PROOF** — TPM EK, AMD SEV-SNP VCEK
   (verified, this exists on our hardware), Intel SGX EK. These give
   per-die crypto identity *by design*. No prior attack used a derived
   key as a model weight or as an input transformation. Question is
   whether this counts as "constitutive" or merely "cryptographic
   gate-keeping" (similar to Wu et al. arxiv 2212.11133).

## What is **not** attempted at all in 14 attacks

| Category | Attempted? | Why missed |
|---|---|---|
| Substrate-as-signal (information flow into model) | 14/14 | We only thought in this frame |
| Substrate-as-constraint (computation requires it) | 0/14 | Requires hard physical coupling |
| Substrate-as-reward (evolutionary fitness) | 0/14 | Requires outer-loop selection |
| Substrate-as-history (accumulated wear) | 0/14 | Requires weeks of wall time |
| Substrate-as-active-degradation (model writes to HW) | 0/14 | Risk-averse, but not impossible |
| Substrate-as-joint-multichannel (SCA-style) | 0/14 | Each marginal was tested in isolation |
| Substrate-as-cryptographic-binding (TPM/SEV) | 0/14 | Considered "by design" not "emergent" — but worth revisiting |

## Pre-registration mistake we kept repeating

Every gate has been a variant of `Δ(HW) > Δ(control) by k·σ`. This is a
*statistical separation* gate. It is **falsifiable by any sufficiently
structured surrogate**. A gate that would survive the SW-matched + shuffle
critique must be a *constructive* gate: "model M produces output Y *only*
when running on device D, and produces ⊥ otherwise". Cryptographic by nature.
We never wrote such a gate.
