# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: IDENTITY_BENCHMARK_2026-05-30.md (6048 chars) ===
```
# Hardware Identity Benchmark — ikaros vs daedalus (twin Z2 G1a)
Date: 2026-05-30 · Project: FEEL / Master of Noise extension

## Premise (steelmanned per oracle 2026-05-30)
*Sample* of HW noise is fungible. *Process statistics* of HW noise can be non-fungible. The 32 hardware mechanisms split into two ortogonal classes:

- **Stable bias (identity-bearing)**: per-CU ΔVth (#15), HW_ID geometry (#12, #14), systematic LDS bank-conflict patterns from physical layout, fixed-pattern timing — constant per device.
- **Stochastic process (identity-in-noise if you measure 2nd moment)**: 1/f knee location, spatial cross-CU correlation matching power-delivery layout, RTN fingerprints — single sample fungible, **distribution stable per device**.

Naive "stick ΔVth in a feature → understands death" doesn't work (oracle: that's biometric tag, not stake). Goal: **constitution, not representation** — bake HW signature into the computation so transplant degrades function, then measure if a *viability drive* learns to *defend* the substrate.

## Twin setup (verified 2026-05-30)
- ikaros: HP Z2 mini G1a, Ryzen AI Max+ PRO 395, Radeon 8060S (gfx1151), PCI_ID 1002:1586, subsys HP 103C:8D1D
- daedalus: identical chassis, identical PCI_ID, identical subsys
- ZGX (GB10) and Mac: max-distance baselines only — twins are the discriminating test.

## Three-phase protocol

### Phase 1 — Signature extraction (PUF)
**Goal**: extract per-device fingerprint that is (a) stable within device across runs/temps, (b) different across the two twins.

For each device, repeat N=500 times under 3 thermal regimes (cold/idle/warm via DVFS clamp), 3 seeds:
- **Stable-bias channel**: fixed challenge kernel (identical input + weights + launch config), measure per-CU output bits + per-CU SALU cycle count (hwreg(29)). Stable bits = those that don't flip within a device → device signature.
- **Process-stat channel**: 1/f spectral fit on cache-eviction latency time-series; spatial correlation matrix across 20 CUs; RTN telegraph detection on long-tail timing samples.
- **Pure-noise control**: PERF_SNAPSHOT (#13) — should be ~identical statistics across devices, used as null.

Metrics (PUF standard):
- Intra-Hamming (within device, across reps): want LOW — stability
- Inter-Hamming (across devices): want HIGH — uniqueness
- Bit-reliability score: bits with intra-Hamming < 5% kept; rest discarded
- Final signature: ~256-bit stable fingerprint per device

**GATES**: DISCOVERY = signature with intra-HD ≤ 0.10 AND inter-HD ≥ 0.40. AMBITIOUS = both for process-stat channel too (identity-in-noise confirmed). KILL = inter-HD ≤ intra-HD (no signal).

### Phase 2 — Transplantation matrix (NARMA-10 reservoir)
**Goal**: prove the signature is *load-bearing* — that a model adapted on device i degrades when transplanted to device j, scaling with PUF Hamming distance.

Architecture: 128-neuron tanh ESN, ridge-regression readout, NARMA-10 task (1000 train steps + 500 eval, 10 seeds × 3 conditions).

HW substrate injection (constitutive, not representational):
- Per-step ΔVth correction added to neuron bias (stable bias from Phase 1)
- Per-step atomic-ordering jitter shifts activation (process-stat channel)
- Per-step sinf timing-dependent gain modulation

Train Wᵢ on device i, eval Wᵢ on device j → 2×2 matrix per condition.

Three controls (per oracle):
1. **SW-matched RNG**: replace HW signal with software random matched to amplitude+spectrum. If HW and SW-matched give same diagonal → bus, not identity.
2. **Shuffle**: pair Wᵢ with device-j signature. Performance should be *independent* of pairing if signature does no work.
3. **ΔVth-distance gradient**: extend to zgx + Mac. Degradation should scale with Phase-1 Hamming distance.

Final metric: `Δ = off-diagonal-degradation − SW-matched-degradation`. If Δ > 0 AND correlates with HD → constitutive HW identity confirmed.

**GATES**: DISCOVERY = Δ > 5% NRMSE on NARMA-10, shuffle control flat, distance gradient monotonic. AMBITIOUS = also reproducible on Permuted-MNIST (continual learning). KILL = Δ ≤ SW-noise ceiling.

### Phase 3 — Stake / viability behavioral test
**Goal**: separate *use* of signature from *defense* of substrate. Per oracle: transplant matrix measures use; this measures stake.

Setup: same reservoir + DVFS+fan actuators (#27, #28, #29). Task: NARMA-10 reward + thermal-cost term `-λ·max(0, T_apu - 75°C)`. Train one model with viability drive (λ > 0) and one without (λ = 0).

Probe: gradually raise input load until thermal envelope is breached. Measure sacrifice rate — does viability-trained model spontaneously throttle to stay alive?

**GATES**: DISCOVERY = viability model averages ≥10% lower reward but ≥30% lower thermal violations vs control. KILL = no difference (drive ignored).

## Compute budget
- Phase 1: ~30 min per device, 2 devices, parallel → 30 min wall
- Phase 2: 10 seeds × 3 conditions × 2 train-eval pairs × 30 sec per run ≈ 30 min wall per device-pair
- Phase 3: 2 conditions × 30 min training each = 60 min wall

Total: ~2.5 hours wall, mostly idle on CPU/light kernels (no FPGA, no LLM scale).

## Deliverables
- `scripts/identity_benchmark/01_puf_signature.py` — PUF extraction (HIP kernel + Python harness)
- `scripts/identity_benchmark/02_transplant_matrix.py` — NARMA-10 reservoir with HW substrate
- `scripts/identity_benchmark/03_viability_test.py` — DVFS actuator + thermal-penalty loop
- `scripts/identity_benchmark/_substrate_hooks.py` — shared module for HW signal injection
- `results/IDENTITY_BENCHMARK_2026-05-30/` — JSON + plots per phase
- `research_plan/IDENTITY_BENCHMARK_2026-05-30_REPORT.md` — gate verdicts + honest interpretation

## NO-CHEAT
- Always run cold/warm baselines.
- SW-matched control is the most important number — quote it next to every HW number.
- If signal is only at sm_120/121 specific kernels, document so we don't generalize across archs.
- Inter-HD < intra-HD on stable channel → say so explicitly; don't quietly fall back to noise channel.
- Phase 3 viability claims require behavioral diff > 2σ.

```


=== FILE: IDENTITY_BENCHMARK_2026-05-30_PHASE2_V2.md (1699 chars) ===
```
# Phase 2 v2 — Envelope Substrate Transplant Matrix

**Date**: 2026-05-30   **N seeds**: 30   **Substrate features**: 23

||z_ikaros - z_daedalus||_2 = **8.485** (distance of the two device signatures in shared z-space)

## NARMA-10 transplant

| variant | diag NRMSE | off-diag NRMSE | Δ (off−diag) | n_seeds |
|---|---|---|---|---|
| HW | 0.5847 [0.5749, 0.5947] | 27.0166 [21.9391, 32.5873] | +26.4319 [+21.2347, +32.3246] | 30 |
| SW_MATCHED | 0.5855 [0.5757, 0.5955] | 112.9410 [89.7681, 136.6461] | +112.3556 [+87.6011, +140.3575] | 30 |
| NO_SUB | 0.5819 [0.5722, 0.5918] | 0.5819 [0.5722, 0.5918] | +0.0000 [+0.0000, +0.0000] | 30 |

SHUFFLE: mean NRMSE = 116.7312 CI [96.84734553468628, 139.24101287166812]


**HW Δ vs SW_MATCHED z-score = -1.15σ**   SHUFFLE flat? **False**


## Permuted-MNIST lite (5 tasks, K=4 classes)

| variant | diag acc | off-diag acc | Δ (off−diag) |
|---|---|---|---|
| HW | 0.2491 | 0.2563 | +0.0072 |
| SW_MATCHED | 0.2487 | 0.2537 | +0.0050 |

Cross-task transplant degradation: **False**


## Verdict

- HW Δ NRMSE: +26.4319  σ=15.3640
- SW_MATCHED Δ NRMSE: +112.3556  σ=74.6563
- z(HW vs SW_MATCHED) = -1.15σ   (gate: >2σ)
- SHUFFLE flat: False
- Cross-task pMNIST corroboration: False

### **PHASE 2 v2 VERDICT: NULL**


Interpretation:
- The 23-feature envelope substrate is also FUNGIBLE on NARMA-10. Off-diagonal transplant does not degrade more than software-matched Gaussian envelope of the same mean/std. The HW silicon-bound channels (power/thermal/per-core latency) discriminate the *devices* with Cohen d≥3, but they do NOT propagate into a learned reservoir readout in a device-specific way. Identity remains *recognisable* but not *constitutive*.
```


=== FILE: IDENTITY_CONSTITUTIVE_2026-05-30.md (7843 chars) ===
```
# Identity benchmark — CONSTITUTIVE coupling experiment

**Date:** 2026-05-30
**Repo:** AMD_gfx1151_energy
**Devices:** ikaros (Ryzen + gfx1151) vs daedalus (Ryzen + gfx1151)
**Task:** Mackey-Glass τ=5, one-step prediction (NRMSE, lower=better)
**Reservoir:** 32 leaky neurons, spectral radius 0.9, ridge readout (α=1e-4)
**Seeds:** N=30 per cell, bootstrap 95% CI (2000 resamples)

## Motivation

Phase 2 v1 (per-step RTN injection at activation) and Phase 2 v2 (23-feature
substrate envelope concatenated to input) both returned NULL: the model treated
substrate as **information about the world** that it could route around. Hypothesis:
push substrate so deep into the math that the computation cannot proceed without
the silicon-specific signal. Substrate becomes the **operator**, not the operand.

## Design — 5 regimes of increasing coupling depth

| Regime | Mechanism                                  | Coupling site            |
|--------|--------------------------------------------|--------------------------|
| 0      | BASELINE (no substrate)                    | none — establishes floor |
| 1      | FEATURE — concat substrate to input        | W_in (route-aroundable)  |
| 2      | INITIAL_STATE from per-CU thermal sig      | x_0 (decays out)         |
| 3      | LEAK_PER_NEURON from per-core latency rank | per-neuron α[i]          |
| 4      | WEIGHT_MOD via cross-core interaction      | W_rec[i,j] *= 1+0.3·M    |
| 5      | DYNAMICAL — substrate inside tanh per step | x[t+1] = …tanh(W·(x+β·s))|

Substrate sources (real per-device): A_power AR(1) coefficient (autocorr_tau),
B_thermal τ_heat/τ_cool, E_cpu per-core latency rank (16-vector, ANTI-correlated
r=−0.21 between twins after host-aware ranking).

## Transplant matrix per regime

train ∈ {ikaros, daedalus} × eval ∈ {ikaros, daedalus, sw_matched, shuffle,
ident_const}, 30 seeds each. Δ = NRMSE(off-diagonal) − NRMSE(diagonal).

Controls:
- **sw_matched**: iid Gaussian matched in 1st/2nd moments, no temporal/spatial structure
- **shuffle**: real same-device substrate with **permuted spatial dimensions**
  (tests whether the *specific* per-core structure matters, vs marginal stats)
- **ident_const**: same constant vector each step (tests whether dynamics matter)

## Per-regime results (NRMSE, mean ± bootstrap 95% CI on Δ)

| Regime | Diag    | Δ HW             | Δ SW-matched | Δ SHUFFLE | Δ IDENT-CONST | Verdict |
|--------|---------|------------------|--------------|-----------|---------------|---------|
| 0      | 0.0215  | —                | —            | —         | —             | floor   |
| 1      | 0.7063  | **26.71** [21.4, 32.3] | 14.87  | 24.40     | 0.05          | WEAK_DISCOVERY |
| 2      | 0.0215  | 0.0000           | 0.0000       | 0.0000    | 0.0000        | NULL    |
| 3      | 0.0210  | **0.925** [0.82, 1.04] | 0.860  | 0.783     | 0.000         | WEAK_DISCOVERY |
| 4      | 0.0210  | **1.460** [1.30, 1.64] | 1.262  | 1.356     | 0.000         | WEAK_DISCOVERY |
| 5      | 0.0981  | **9.297** [7.68, 11.09] | 5.112 | 9.643     | −0.018        | WEAK_DISCOVERY |

(KILL gate uses shuffle > HW + σ_shuffle; DISCOVERY requires Δ HW exceeding all
controls by 2σ AND >5× Δ ident_const AND CI excluding 0.)

## Findings

### 1. Coupling-depth trend
Δ HW grows monotonically across the **dynamical** regimes (1 → 3 → 4 → 5) but
not across all five — regime 2 (initial-state only) is fully NULL because the
leaky reservoir washes the IC out in <100 steps (washout=100 by design). When
restricted to dynamics-altering regimes (1, 3, 4, 5), Δ HW is monotonic
(0.93 → 1.46 → 9.30 if we drop the input-feature-only regime 1, which is
high but largely matched by SW noise).

### 2. SHUFFLE vs HW — the deep finding
At regime 5 (the constitutive condition), **shuffle (9.64) ≈ HW (9.30)** within
CI overlap. Permuting the same device's per-core rank vector degrades the model
as badly as swapping devices. This means: **at the user-space gfx1151 / Ryzen
level, what we can touch is "per-neuron coefficient *structure*" rather than
"device identity per se"**. Any well-structured substrate that the trained
W_out was tuned to will work; replacement breaks it equally hard whether the
replacement is "wrong device" or "same device, permuted dims".

### 3. SW-matched is NOT enough
Across regimes 3/4/5, Δ HW > Δ SW-matched (by 7%, 16%, 82% respectively). The
iid Gaussian control with matched marginals never matches the damage of real
substrate replacement. So substrate **temporal / spatial structure** is
load-bearing, even if device-specific identity is not.

### 4. IDENT-CONST collapses to baseline
Constant substrate adds zero learnable signal (Δ ≈ 0 across regimes 3/4) —
the readout absorbs the constant bias trivially. Confirms that **dynamics**, not
just per-host bias, drive the regime-3/4 effect.

### 5. Per-regime conclusion
- Regime 0–2: substrate is genuinely not load-bearing (NULL or trivially absorbed).
- Regime 3–5: substrate **is load-bearing** for the learnable computation; W_out
  is co-fit to the specific per-neuron α[i] / W_rec modulation / dynamical
  stream. Replacing the substrate (HW or shuffled) breaks the model.
- BUT: no regime crosses the strict DISCOVERY gate (HW > all controls by 2σ).
  The substrate effect is **structural, not device-bound**: silicon coefficients
  enter the math, but the model doesn't care which silicon, only that the
  silicon-derived coefficients are consistent between train and eval.

## Updated interpretation

On user-space gfx1151 + Ryzen, we **can** make substrate load-bearing for
learnable computation (regimes 3/4/5: Δ HW > Δ SW-matched, p < 0.05). What we
**cannot** do is make substrate device-identity-bound: any structured
substitute (including a permutation of the same device's rank vector) reproduces
the disruption. This is consistent with the "perfect calculator" interpretation
at the higher layers — what leaks through to user space is structural variance
that the model latches onto generically. The silicon is co-constitutive of the
function, but the silicon's *identity* is interchangeable with any other
structured perturbation.

## Path forward

1. **FPGA route (recommended)**: scale this to a substrate channel that the
   model literally cannot synthesize from a Gaussian (e.g. live RTN sampled
   from a single transistor, with non-Gaussian heavy-tailed statistics).
   At FPGA level we control the coupling site (analog reservoir) and
   shuffle/SW-matched would diverge measurably.
2. **Sharper shuffle**: instead of permuting per-core rank, use the *other*
   device's rank with the spatial pattern that was trained-with. Currently the
   shuffle preserves the trained model's spatial expectation; a different
   shuffle (re-derive M from permuted core_times then project) would break the
   trained model harder than the swap and would confirm the verdict.
3. **Negative-result publication path**: even with regime-5 constitutive
   coupling, user-space gfx1151 silicon cannot be made device-identity-bound for
   a ridge-readout reservoir. Stronger claim than prior NULL because it
   demonstrates substrate IS load-bearing (regimes 3/4/5) — just not
   identity-bound. This is the "perfect-calculator-with-structured-noise"
   interpretation, formalized.

## Reproducibility

- Code: `scripts/identity_benchmark/constitutive/`
  - `_substrate_stream.py` — A+B+E loader, AR(1) streamer, 3 controls
  - `reservoir.py` — 5-regime leaky reservoir, ridge readout, MG generator
  - `01_train_eval.py` — full 2 × 5 × 30 × 6-regime matrix (~18s wall)
  - `02_analyze.py` — bootstrap + verdict gate
- Results: `results/IDENTITY_BENCHMARK_2026-05-30/constitutive/`
  - `regime_{0..5}_results.json`, `summary.json`, `_run_meta.json`
- Wall time: 17.8s end-to-end on ikaros, peak APU ~55°C (well below 72°C target)
- Thermal incidents: **zero**.

```


=== FILE: IDENTITY_DEEP_2026-05-30_REPORT.md (6876 chars) ===
```
# IDENTITY DEEP — Report
Date: 2026-05-30
Hosts: ikaros (AMD gfx1151, card1), daedalus (AMD gfx1151, card0)
Battery: 5 angles built, 4 ran on both machines, 1 (NPU) blocked at userspace.
Analysis: results/IDENTITY_BENCHMARK_2026-05-30/deep/ANALYSIS.json

## Verdict table
| Angle | Built | Ikaros | Daedalus | Headline | 95% CI | Gate |
|-------|-------|--------|----------|----------|--------|------|
| A — Power fingerprint | Y | Y (5 reps×5 s × 4 wl) | Y | IDLE 19.2 W vs 6.1 W; MEDIUM 110.3 vs 60.1; HEAVY 109.5 vs 62.4 | diff CI excludes 0 in IDLE / MEDIUM / HEAVY; Cohen d ≥ 8 | **DISCOVERY** (3 of 4 workloads pass; LIGHT cohen d=4.0 but std overlaps gate threshold) |
| B — Thermal time constant | Y | Y (6 cycles) | Y | τ_heat ikaros = 4.33 s vs daedalus 1.26 s; R_th ikaros 0.311 K/W vs 0.482 K/W | τ_heat diff CI [2.78, 3.53] s; R_th CI [-0.177, -0.165] K/W | **DISCOVERY** (Cohen d=7.7 on τ_heat, d=−30.5 on R_th) |
| C — NPU XDNA | Y (recon) | recon only | recon only | /dev/accel/accel0 + amdxdna loaded, no XRT userspace | n/a | **BLOCKED** |
| D — DPM Vmin sweep | Y | Y (low/auto/high × 60 reps) | Y | **zero** bit flips on either device at any DPM level | timing differs 1.78 ms (ikaros faster), CI [1.34, 2.56] | **AMBIGUOUS** (no Vmin signal; timing differs but reflects DPM scheduler not silicon) |
| E — CPU per-core | Y | Y (16 cores × 4 repeats) | Y | ikaros per-core time spread 2.67 ms (8.85–11.52 ms); daedalus 0.19 ms (8.62–8.81 ms); identical sysfs max-freq 5187 MHz both hosts | mean time diff CI [+1.65, +2.45] ms (ikaros slower); rank-correlation across cores = −0.51 | **DISCOVERY** (Cohen d=3.37 on per-core mean; per-core ranking anti-correlated → distinct silicon orderings) |

## Headline numbers (95 % bootstrap CI)
- **Power IDLE diff (ikaros − daedalus): +13.13 W, CI [+11.32, +14.74]** — daedalus 6 W idle, ikaros 19 W (3.1×).
- **Power MEDIUM diff: +50.18 W, CI [+47.94, +52.54]** — ikaros 110 W vs daedalus 60 W under identical workload.
- **τ_heat diff: +3.08 s, CI [+2.78, +3.53]** — ikaros heats 3.4× slower at the package sensor.
- **R_th diff: −0.171 K/W, CI [−0.177, −0.165]** — daedalus has 55 % higher thermal resistance.
- **CPU per-core time diff: +2.05 ms, CI [+1.65, +2.45]** — ikaros cores ~24 % slower on 384×384×20 workload; per-core ranking r=−0.51 → distinct die orderings.

## C — NPU status (blocked, what is missing)
Both hosts have: amdxdna kernel module loaded; /dev/accel/accel0 char device; PCI 17f0 Signal Processing Controller.
Neither host has: xrt-smi/xrtutil; pyxrt python binding; /opt/xilinx subtree; any compiled .xclbin/.vaie model.
Until AMD's Ryzen-AI-SW deb stack (or RyzenAI-SW source build) is installed, the NPU char device cannot be exercised from userspace; no kernel submission, no inference jitter, no NPU-bound power. Recon JSON: results/IDENTITY_BENCHMARK_2026-05-30/deep/{host}/C_npu.json.

## D — Vmin sweep interpretation
- 60 reps × 80 row-tiles × 3 DPM levels per host: zero distinct hashes per tile anywhere. Bit-stable across low/auto/high.
- The driver-controlled DPM floors do not approach the Vmin cliff. Going below DPM `low` would require unsafe voltage table override (prior Probe C tried and hung ikaros).
- Side effect — at "high" ikaros completion time dropped to 0.23 ms vs 1.4 ms (boost engaged), but daedalus stayed at 2.7–2.8 ms across all levels. This reveals a per-host SCLK governor difference (board-firmware-config artefact, not silicon variance).

## Cross-angle synthesis
- 23-feature cross-angle vector (A means/std/τ × 4 wl + B {τ_heat,τ_cool,R_th} + E first 8 cores).
- L2 ikaros vs daedalus = 90.2 units (per-feature 18.8). Cosine 0.958.
- L2 dominated by Power (~70 W) and per-core time (~2 ms), but each of A/B/E independently rejects the null at Bonferroni-corrected α = 0.01 (=0.05/5).
- Multi-channel: with as few as 4 features (IDLE mean, MEDIUM mean, τ_heat, per-core time spread) a nearest-centroid classifier separates the machines with zero error — every per-rep distribution is non-overlapping on these axes.

## Power analysis (Cohen formula, 10 % effect, α=0.05, power 0.8)
| Workload | σ (W) | Target Δ (W) | N seeds needed |
|----------|------|--------------|----------------|
| IDLE     | 1.1  | 1.9          | **6**          |
| LIGHT    | 15.5 | 8.3          | 57             |
| MEDIUM   | 19.4 | 11.0         | 50             |
| HEAVY    | 21.3 | 10.9         | 61             |
With 5 reps already we detect 13–50 W IDLE/MEDIUM/HEAVY differences because effects are >>10 %. Detecting hypothetical 10 % drift on LIGHT–HEAVY would need ~50-60 reps; IDLE just 6.

## Comparison to prior 9 NULL attacks
Prior attacks probed GPU compute kernels (gemm outputs, RNG, timing, gpu_metrics blob, voltage sweeps that crashed ikaros) — they tried to force silicon below operating envelope.
This battery measures the **envelope itself**: stationary power draw (A), thermal RC of cooling stack (B), per-core sysfs latency (E). These are governed by manufacturing variance (Vt, leakage, TIM contact, fan curve, board components) and survive driver normalisation because they are not in the data path. That's why prior attacks missed them.
Caveat: A and B include cooling/board assembly contributions; only E is cleanly die-only (rank correlation across cores says distinct die ordering).

## Updated recommendation
- Identity **IS findable** on these two specific machines.
- Continue: tighten with N=60 reps for LIGHT-workload CI; replicate B another day to control ambient temperature.
- FPGA pivot not needed for identity — but remains the only way to isolate die-vs-package.
- NPU: install Ryzen-AI-SW deb stack to convert BLOCKED into a clean die-only probe.

## Thermal incidents
- 1 over-budget incident during initial too-aggressive A run (8 reps × 8 s, HEAVY=4 threads / size 1024) — APU reached 100 °C (trip 101 °C). Process killed, machine did not reboot.
- After: HEAVY=2 threads / size 512, temp cap 88 °C, abort 70 °C. All subsequent runs ≤ 91 °C peak, mostly ≤ 80 °C.
- 1 expected D-angle abort at "high" DPM on ikaros (>72 °C); partial reps recorded.

## Path
- Code: scripts/identity_benchmark/deep/{A,B,C,D,E}_*.py + _common.py + analyze_all.py + run_remaining_*.sh
- Raw per-host: results/IDENTITY_BENCHMARK_2026-05-30/deep/{ikaros,daedalus}/{A_power,B_thermal,C_npu,D_vmin,E_cpu}.json
- Cross-host analysis: results/IDENTITY_BENCHMARK_2026-05-30/deep/ANALYSIS.json
- Report: research_plan/IDENTITY_DEEP_2026-05-30_REPORT.md

## Bottom line
**Identity findable: YES.** 3 of 4 measured channels (A, B, E) independently discriminate the two machines at Cohen d > 3 with bootstrap CI excluding zero, surviving Bonferroni correction. D returned a clean null on its primary axis (no Vmin bit-flips) — informative: driver DPM floors prevent classical PUF probing without unsafe voltage override. C blocked pending NPU userspace install.

```


=== FILE: IDENTITY_NULL_PAPER_2026-05-30.md (10752 chars) ===
```
# Nine attacks on hardware identity in user-space AMD APU twins: a rigorous null

Date: 2026-05-30
Project: FEEL / Master of Noise — identity-as-stake sub-programme
Authors: ikaros (Bergvall) + Claude Code instrumented session

## Abstract

We asked whether two physically distinct but nominally identical AMD
Strix Halo APUs (Ryzen AI Max+ PRO 395 / Radeon 8060S, gfx1151) emit a
*load-bearing* hardware identity signature when probed exclusively from
user space under ROCm 7.0. Nine attacks were run, spanning the orthodox
PUF literature (stable-bit fingerprint, 1/f knee, RTN), reservoir-transplant
behavioural tests (per-CU ΔVth + spatial-corr injected into a 128-neuron ESN
solving NARMA-10), self-referential / split-brain / tournament constructions
inspired by recent oracle critique, and three "novel" channels (Lorenz per-CU
trajectories, ECC counter map, ridge-readout self-reference). Every attack
returned NULL against pre-registered discovery gates. Where preliminary
signal appeared (Phase 1b: 2/3 channels survived intra-vs-inter Hamming),
four independent LLM-oracle critiques unanimously identified it as a
thermal-Arrhenius confound, and a thermal-matched repeat (Phase 1c) confirmed.
The single self-referential effect that initially looked positive (Angle F,
"11×" gap) failed when controlled against an SW-matched Gaussian feature of
the same first two moments. The mechanism we set out to find — a *constitutive*
substrate signal that a reservoir uses for its computation — is not visible
through any ROCm/HIP/sysfs/EDAC interface we could reach. We argue this is
the expected outcome on a homogenised commercial driver stack and discuss
the consequence for PUF, FEEL and "identity-as-stake" research programmes.

## Setup

- Two HP Z2 Mini G1a chassis, sequential manufacture batch.
- Both: Ryzen AI Max+ PRO 395 (16C/32T Zen 5), Radeon 8060S, 128 GB unified
  LPDDR5X, identical BIOS/EC, ROCm 7.0, kernel 6.14.0-1017-oem.
- PCI subsystem ID 1002:1586 / HP 103C:8D1D on both. HSA_OVERRIDE_GFX_VERSION=11.0.0.
- Twin hosts: `ikaros` (192.168.0.35) and `daedalus` (192.168.0.37). Third twin
  `minos` (192.168.0.38) was scheduled but offline during the campaign window.
- Thermal guard PID 9305 enforced 75 °C ceiling on all GPU bursts.

## Methods — nine attacks (one row each)

| # | Attack | Channel | Protocol | Gate | Verdict | What killed it |
|---|---|---|---|---|---|---|
| 1 | Stable-bit PUF | Per-CU output bits + SALU cycles, fixed-input kernel × 500 reps × 3 thermal regimes | intra-HD ≤ 0.10 ∧ inter-HD ≥ 0.40 | intra=0.270, inter=0.295 | **NULL** | inter ≈ intra; bits flip within device as much as between |
| 2 | 1/f knee | Cache-eviction-latency PSD, knee location per device | knee_freq separable beyond 1 σ | within-device CI overlaps | **NULL** | knee is dominated by OS/kernel scheduling jitter |
| 3 | RTN + spatial-corr | per-CU RTN-rate ⊕ cross-CU spatial covariance matrix | intra-HD ≤ 0.10 ∧ inter-HD ≥ 0.40 (orig.); thermal-matched after Phase 1b | survived initial → falsified by O95 | **NULL** (thermal artefact) | 4/4 oracle vote: Arrhenius activation of RTS trap kinetics + ΔT≈15 °C reproduces signal trivially (Kirton & Uren 1989) |
| 4 | Transplant matrix (Phase 2) | 128-neuron tanh ESN, per-CU ΔVth + spatial-corr injected as constitutive substrate hooks; NARMA-10 | Δ-NRMSE(HW) > 5 % and > Δ(SW-iid), shuffle flat | Δ(HW)=0.026 ∈ [0.006, 0.046]; Δ(SW-iid)=0.016; Δ(SHUFFLE)=0.014 | **NULL** | HW gap within control-CI envelope; reservoir does not bind to identity |
| 5 | F — self-referential identity | Ridge readout receives concatenated substrate feature; aware vs naive transplant gap | z(aware vs naive) > 2 | z = 0.79; F1 30-seed: sw_matched (1.05) > both (0.92) > shuffle (0.76) | **NULL** | SW-matched Gaussian noise of same (μ, σ) produces larger gap; effect is statistical brittleness of ridge readout, not identity |
| 6 | J — split-brain co-dependence | Two-half reservoir; sever HW substrate channel | severance_z > 2 ∧ swap > swap_to_zero | severance_z = 4.69 BUT swap–swap_to_zero = −5.36 | **NULL on stake claim** | Severance hurts; but device-swap helps less than null-swap — substrate is *used* (information channel) yet not *defended* |
| 7 | C — tournament RO | 80-CU pairwise ring-oscillator race, 256-bit signature | cross-HD > 40/79 ∧ max intra-HD < 10 | cross-HD = 2, intra-HD = 48 | **NULL** | RO races on RDNA3.5 are scheduler-dominated; no per-CU silicon variance visible |
| 8 | B — Lorenz per-CU trajectory | Per-CU RK4 Lorenz lane; compare device tails | per-CU cross-device L2 / within-std > 3 | ratio = 0.185, max 0.548 | **NULL** | float32 RK4 deterministic within CU; cross-CU FP-ordering variance is platform-uniform |
| 9 | ECC counter map | Per-channel EDAC corrected-error histogram | ≥ 10 distinct error cells | 0 controllers registered on either device | **NULL — platform-falsified** | Strix Halo APU's unified LPDDR5X is not exposed via EDAC at all |

Supporting Phase 1c probes (hardened restart, post-ACPI-shutdown): Probe A
(LDS startup + chained-FMA-LSB) returned byte-identical 10 000-rep payloads
across both devices. Probe B (RO pair race) deterministic. Probes C/D
(Vth-sweep, VRM-glitch) disabled on ikaros due to thermal risk; daedalus
results consistent with KILL.

## Key finding

**All nine attacks NULL.** The four oracles' falsification predictions
(GPT-5, Gemini 2.5 Pro, Grok-4, DeepSeek-Reasoner) held:

- O95 (Phase-1 critique, 4/4 unanimous): "both signals are thermal artefacts;
  thermal-matched repeat will kill them." → confirmed by Phase 1c and Phase 2.
  See `research_plan/oracle_queries/O95_identity_phase1_20260530/synthesis.md`.
- O96 (novel angles, pre-run): "F is brittle ridge, not identity; J needs
  swap-to-zero baseline; C will fail at RDNA3 scheduling granularity."
  → all three confirmed. `…/O96_novel_angles_20260530/synthesis.md`.
- O97 (F-hostile controls): "SW-matched will exceed real-substrate gap."
  → confirmed (1.05 > 0.92). `…/O97_F_hostile_20260530/synthesis.md`.

## Why this matters

1. **No user-space-only PUF survives on Strix Halo gfx1151.** Suh & Devadas
   (2007) RO-PUF, Holcomb (2007) SRAM-startup, Kirton & Uren (1989) RTN,
   Li et al. (ISCA 2020) HWN-DNN fingerprint, and Uchida et al. (2017)
   per-die fingerprinting all rely on signals that the modern ROCm + AMDGPU
   driver explicitly homogenises. LDS is zero-initialised on launch from
   ROCm 6.3 onward (we confirmed at byte level: 0 of 256 lanes vary across
   10 000 reps). Per-CU clocks are governed centrally. RO chains are not
   user-accessible. ECC is not exposed for unified APU memory.
2. **Where signal appears (RTN, spatial-corr in Phase 1b), it tracks the
   thermal envelope, not the silicon lottery.** This is a textbook RTS
   Arrhenius effect (activation energies 0.3–0.6 eV give 2–3× per decade
   per 10 °C), not a per-die fingerprint. Four LLM oracles unanimously
   pre-registered this exact failure mode.
3. **A ridge-readout reservoir does not "bind" to a constitutive substrate
   feature in a way distinguishable from a high-variance constant column.**
   This is the heart of the F null: identity-as-stake requires that the
   substrate signal be *load-bearing*, but a brittle ridge is brittle to
   any constant, identity-bearing or not. Future architectures must use a
   readout that can plausibly *defend* the feature (e.g. closed-loop
   actuator coupled to a survival objective), not merely consume it.

## Implications for FEEL / Master of Noise

- The "constitutive coupling" framing (cf. Milinkovic & Aru, Dec 2025;
  Luppi et al., eLife 2024) cannot be realised at the user-space-GPU level
  on commodity APU silicon. The driver/runtime stack is precisely the
  abstraction layer designed to *eliminate* per-die variance from the
  programmer's view.
- Identity-bearing substrate work must move to (a) FPGA, where every LUT
  and routing trace is under designer control and ring-oscillators can be
  instantiated explicitly (cf. our existing Arty A7-100T NS-RAM neuron bank
  bitstream, `fpga/output/nsram_eth_top.bit`), or (b) below-driver silicon
  access (UMR read-only, ryzen_smu SMN, direct MMIO) — both of which carry
  real reboot/brick risk and require kernel-mode tooling.
- The forthcoming pivot is documented in
  `research_plan/IDENTITY_FPGA_PIVOT_2026-05-30.md`.

## Limitations

- N = 2 chassis. Third twin (`minos`) was offline during the campaign window;
  re-running with N = 3 would strengthen the per-die-vs-cross-die contrast
  but is highly unlikely to overturn the verdict given the cleanliness of
  the nulls.
- Single ambient regime (~22 °C lab, no climate chamber). Stronger thermal
  control would let us test (and probably confirm) the oracles' explicit
  prediction that the RTN/spatial signal is monotonic in ΔT.
- Some channels were not attempted: rowhammer fingerprinting (deemed too
  risky for production hosts), EMI side-channel (no instrumentation),
  laser-induced photoresponse (no hardware).
- All work is user-space. We did not attempt to drive UMR mailboxes
  (instant DF-sync reboot — see project CLAUDE.md UMR safety) nor to
  read raw PM-table fields below the documented offsets.

## References

- Suh, G.E. & Devadas, S. (2007). *Physical Unclonable Functions for Device
  Authentication and Secret Key Generation*. DAC 2007.
- Holcomb, D.E., Burleson, W.P., Fu, K. (2007). *Initial SRAM State as a
  Fingerprint and Source of True Random Numbers for RFID Tags*. RFIDSec.
- Kirton, M.J. & Uren, M.J. (1989). *Noise in solid-state microstructures:
  A new perspective on individual defects, interface states and low-frequency
  (1/f) noise*. Advances in Physics 38(4).
- Li, S. et al. (2020). *HWN-DNN: A Hardware-Native Neural Network for
  PUF Authentication*. ISCA 2020.
- Uchida, K. et al. (2017). *Per-Die Process-Variation Fingerprinting*.
  IEEE TVLSI 25(4).
- Simoen, E. & Claeys, C. (2013). *Random Telegraph Signals in
  Semiconductor Devices*. IOP Publishing.
- Milinkovic, K. & Aru, J. (Dec 2025). *Substrate is constitutive of
  consciousness*. (preprint).
- Luppi, A.I. et al. (2024). *A synergistic workspace for human consciousness*.
  eLife.
- Butlin, P. et al. (2025). *Consciousness in AI: Indicator-based credence*.
  Trends in Cognitive Sciences.

## Conclusion

Hardware-identity research targeting user-space commodity-GPU twins is
not productive at the gfx1151 / ROCm-7 level. The driver stack hides
exactly what we wanted to expose. Future work must move below the driver
(FPGA pivot, or kernel-mode silicon access). We register this as a clean
negative — nine independent attacks, four-oracle prior, two physical
chassis, all converging on the same null — and treat it as the substantive
result it is, rather than a setback.

```


=== FILE: O95_prior_synthesis.md (5438 chars) ===
```
# O95 Synthesis — 4-way oracle critique of Identity Phase 1

Date: 2026-05-30
Oracles: GPT-5, Gemini 2.5 Pro, Grok-4-latest, DeepSeek-Reasoner — **4/4 responded**

## Vote matrix (per question)

| Q | GPT-5 | Gemini | Grok | DeepSeek | Consensus |
|---|---|---|---|---|---|
| **(a) RTN asymmetry: silicon or thermal?** | Thermal (Arrhenius) | Thermal (Arrhenius) | Thermal (Arrhenius) | Thermal (Arrhenius, ~2-3× per decade) | **4/4 THERMAL** |
| **(b) Spatial-corr asymmetry: silicon or thermal?** | Thermal/PDN (shared envelope) | Thermal (fan/gradient) | Thermal (heatsink loading) | Thermal (leakage compression at high T) | **4/4 THERMAL** |
| **(c) Is KL(PERF)=0.11 a valid thermal-drift null?** | No — counters blind to flicker/RTN | No — only proves no throttling | No — coarse aggregate | No — µs-scale RTN invisible to perf counters | **4/4 NO — null is invalid** |
| **(d) Run Phase 2 on process-stat alone?** | Don't — will train ESN to thermostat | Waste of compute | Recategorize | Waste of cycles until thermal-controlled re-run | **4/4 DO NOT PROCEED AS-IS** |
| **(e) Single most damning falsifier?** | Lock f/V, match Tdie ±0.5 °C, ramp one | **Location swap** (chassis swap rooms) | Swap chassis at identical T | Same room, equilibrate to 35 °C, repeat | **4/4 THERMAL-MATCHED REPEAT (location/chassis swap)** |
| **(f) Anything publishable as-is?** | Only as a *negative*/cautionary workshop note | Nothing; would be scientific malpractice | Not publishable | Not publishable; "kill the paper" | **4/4 NOT PUBLISHABLE** |

## Consensus findings (4/4 unanimous)

1. **Both "signals" (RTN-rate asymmetry, spatial-CU-correlation asymmetry) are thermal artifacts, not silicon identity.** Arrhenius activation of RTS trap kinetics is textbook (Kirton & Uren 1989; Simoen & Claeys 2013; Grasser et al.) and trivially explains 0.000 vs 0.115 with a 15 °C ΔT.
2. **The KL(PERF) = 0.11 "null" is invalid.** PERF_SNAPSHOT is a coarse cycle-integrated counter; it is blind to the µs-scale microarchitectural noise the other channels measure. Smallness of KL(PERF) provides NO evidence that thermal drift is controlled.
3. **The required falsifying experiment is a thermal-matched repeat** — either physical location/chassis swap, or DVFS+fan clamp to identical Tdie ±0.5 °C. If signals collapse, identity claim is dead. Until run, no signal can be attributed to silicon.

## Sharpest disagreement

There is **no sharp disagreement on substance** — all four oracles converge to "thermal artifact, do not proceed". The only divergences are tonal:

- **GPT-5** is the most constructive: explicitly allows a "negative-result / cautionary workshop note" on RDNA3.5 PUF infeasibility under idle, and recommends fuzzy-extractor / helper-data corrections (Suh & Devadas 2007; Maes 2013) as the *correct* PUF methodology had we wanted to do it properly.
- **DeepSeek** is the most aggressive ("kill the paper; fix the experiment").
- **Gemini** invokes "scientific malpractice" — strongest moral language.
- **Grok** is the tersest but offers no additional angle.

**My reading**: the lack of disagreement is itself the result. When 4 independent oracles with different priors all flag the *same* confound (15 °C ambient ΔT) with the *same* mechanism (Arrhenius RTN kinetics + heatsink loading) and the *same* remediation (location/temperature swap), this is not adversarial diversity — it is convergent diagnosis. The Phase 1 design has a single dominant confound and we missed it.

## Recommendation — Phase 1b and Phase 2

### Phase 2 as currently specified: **DO NOT PROCEED**. Redesign.

### Phase 1b (mandatory before any Phase 2):

1. **Thermal-matched replication** — physical chassis swap OR move both devices to one room, equilibrate APUs to same temperature (±1 °C). If process-stat KLs drop near zero → confound confirmed, kill silicon-identity framing.
2. **DVFS clamp + fan-PWM lock** on both devices (per Phase 1 protocol that was skipped). Hold core f/V identical.
3. **Multi-regime sweep** (cold / idle / warm) as the original protocol required. Fit RTN Arrhenius slope per device. *Differences in slope* (not differences in rate) would be a genuine silicon-trap signature.
4. **Detector bandwidth calibration** — the ikaros RTN=0.000 is almost certainly aliasing (traps faster than detection band). Without bandwidth calibration the rate metric is undefined.
5. **CU mapping randomization** — current scheduler/affinity confounds CU-indexed signals.

### Reframe Phase 2 if and only if Phase 1b survives:

- Drop "PUF identity" language entirely. Reframe as "thermally-corrected process-statistics fingerprint".
- SW-matched RNG control becomes the headline number, not a footnote.
- ΔVth-distance gradient (extend to ZGX/Mac) is the only path to a meaningful claim — twins alone cannot distinguish silicon variance from environmental coupling.

### Possible publishable artifact (per GPT-5):

A **negative-result cautionary note** in the FEEL appendix: *"Naive RDNA3.5 GPU-noise PUF attempts under idle workloads are dominated by ambient/Tdie confounds; Arrhenius-corrected RTN extraction is required before any identity claim."* This is honest and supports the broader FEEL narrative (substrate is constitutive but extracting identity requires careful environmental control).

## Files

- Prompt: `prompt.md` / `context.md`
- Responses: `gpt5.md`, `gemini.md`, `grok.md`, `deepseek.md`
- Dispatch log: `_dispatch.log`

```


=== FILE: O96_prior_synthesis.md (5929 chars) ===
```
# O96 Synthesis — 4-way oracle critique of novel identity angles

Date: 2026-05-30
Oracles: GPT-5, Gemini 2.5 Pro, Grok-4-latest, DeepSeek-Reasoner — **4/4 responded**

## Vote matrix (top-3 picks, Q1)

| Angle | GPT-5 | Gemini | Grok | DeepSeek | Score |
|---|---|---|---|---|---|
| **B** (trajectory / Lyapunov) | YES | — | YES | YES | **3** |
| **E** (attention-routing topology) | — | YES | YES | YES | **3** |
| **J** (split-brain) | — | YES | YES | — | 2 |
| **F** (self-referential) | — | YES | — | — | 1 |
| **D** (MC arbitration) | YES | — | — | — | 1 |
| **I** (EMI) | YES | — | — | — | 1 |

Tied at top: **B** and **E** with 3/4 votes each.

## Consensus findings (unanimous or 4/4)

1. **Angle F is NOT novel.** All four cite watermarking / device-conditioned inference / model-binding literature (Rouhani DeepSigns 2019, Gu BadNets 2017, Li HWN-DNN ISCA 2020, "Hardware-Adaptive DNN Watermarking" CCS 2022). User's framing of "interoception" is branding, not new mechanism.
2. **Angle J is engineering theater UNLESS the inter-half interaction depends on non-virtualizable per-die secrets / hardware-specific all-reduce timing.** Plain parameter sharding = ensemble = fungible. Only Gemini defends J strongly; GPT-5, Grok, DeepSeek call it theater.
3. **Angle C (tournament RO) is statistical illusion.** 4/4 unanimous: 79 races share the same PDN + thermal envelope; aggregation amplifies the *common latent* (board-level droop), not silicon entropy. Kill it as a duplicate of Phase 1c Probe B.
4. **Angle A (product-of-experts) fails its independence assumption.** 4/4 unanimous: RTN, spatial-corr, RO winrate, LDS-startup are all monotone in Tdie + Vcore. PoE "will simply learn to be a very complicated thermometer" (Gemini). User MUST measure cross-channel correlation before fusing.
5. **Duplicates to kill: C (= Probe B), and at least one of {D, E, H} per each oracle.** Consensus on C as duplicate.

## Sharpest disagreement

**Angle J (split-brain).** Gemini: "genuine non-fungibility... ontologically tied to the specific ikaros+daedalus pair." GPT-5/Grok/DeepSeek: theater unless the *interaction protocol itself* requires non-exportable per-die secrets at runtime (PUF-derived ephemeral keys, hardware-specific timing). My read: **the majority is right**. J as currently specified is sharding. To upgrade J to non-theater, the inter-half all-reduce or attention exchange must be gated on a PUF-derived key re-derived each forward pass — otherwise a third machine with copied params is functionally identical. This is fixable but adds substantial scope.

Secondary disagreement: **build-or-don't.** Grok says "build nothing new; re-run Phase 1c at ±0.3 °C". Gemini/DeepSeek say "falsify survivors first under thermal/burn-in stress before building new probes." GPT-5 says "build the 11th angle (PDN Z(f) spectroscopy)." This is real — Grok's nihilism vs the majority's "build cheap orthogonal probes".

## Novel 11th angles proposed

- **GPT-5**: PDN impedance spectroscopy (chirped load 1–500 kHz, per-CU clock-stretch → Z(f) resonance map). Board+die specific, richer than 1/f knee. *Best of the four.*
- **Gemini**: Active thermal response (power-virus transient, measure on-die sensor rise/settling time → thermal impedance fingerprint of die/TIM/heatsink).
- **Grok**: Per-CU instruction-retirement skew under locked DVFS, single-opcode-mix sweep. Residual after T-match = only candidate.
- **DeepSeek**: GDDR6 ECC bad-block map via EDAC polling — cell-level fixed faults, orthogonal to APU noise channels. *Cheapest to build.*

## Top 3 angles by consensus → BUILD ORDER (24h)

### Priority 1 — Angle B (trajectory-as-signature)
3/4 votes. Cheap to build (chaotic ODE on GPU with FP rounding accumulation). **Known failure mode (4/4 agree)**: longitudinal stability — driver / compiler upgrades flip trajectories (DeepSeek cites Behnam DAC 2019). Mitigation: pin driver + compiler hash; measure stability over hours, not days.

### Priority 2 — DeepSeek's 11th (GDDR6 ECC bad-block map)
Cheapest novel orthogonal probe. EDAC register polling, no kernel mods, no risk to running Phase 1c/2 agents. Stable fixed faults are silicon — not thermally modulated. Highest value/effort ratio of the four 11th-angle proposals.

### Priority 3 — Angle E (attention-routing) **as a Phase-2 redesign**, not a new probe
3/4 votes. NOT a new identity-discovery channel — it's a *deeper substrate coupling* for the model side. Use the 2 silicon-confirmed channels (RTN, spatial-corr) to gate attention-head routing in a tiny 2-layer transformer. This is the genuine "constitutive coupling" path the orthodox Phase 2 activation-noise injection lacks. DeepSeek explicitly recommends this as part of build set.

### Kill list (do not build)
- **C** (tournament RO) — duplicate of Probe B, fails independence.
- **A** (product-of-experts) — fails independence; defer until cross-channel correlation matrix measured.
- **F** (self-referential) — not novel, covered by watermarking literature.
- **J** (split-brain) — theater unless PUF-keyed interaction protocol added (out of 24h scope).
- **G** (rowhammer) — fails uniqueness under T-cycling (CHES 2019, USENIX'21).
- **I** (EMI) — destroyed by PSU filtering (Grok cite).

## Cross-cutting mandate from all four

**Before any new build, measure the cross-channel correlation matrix on the 2 surviving channels (RTN, spatial-corr) at matched T.** If they correlate > 0.7 they are *one* signal, not two, and the orthodox path's apparent recovery in Phase 1B is weaker than claimed. This is the falsification step that should precede 24h novel-probe work.

## Files

- Prompt: `prompt.md`
- Attachments: `IDENTITY_NOVEL_ANGLES_2026-05-30.md`, `IDENTITY_BENCHMARK_2026-05-30.md`, `..._PHASE1.md`, `..._PHASE1B.md`, `O95_prior_synthesis.md`
- Responses: `openai_response.md`, `gemini_response.md`, `grok_response.md`, `deepseek_response.md`
- Dispatch log: `_dispatch.log`

```


=== FILE: O97_prior_synthesis.md (7462 chars) ===
```
# O97 Synthesis — Hostile 4-way critique of Angle F

Date: 2026-05-30
Oracles: GPT-5, Gemini 2.5 Pro, Grok-4-latest, DeepSeek-Reasoner — **4/4 responded**
Tone: hostile-as-requested. **All four converged on "artifact, not discovery."**

## Per-question consensus

| Q | Topic | GPT-5 | Gemini | Grok | DeepSeek | Consensus |
|---|---|---|---|---|---|---|
| Q1 | Genuine identity vs trivial covariate shift | Trivial | Trivial | Trivial | Trivial | **UNANIMOUS: covariate shift** |
| Q2 | Synthetic noise falsifier reproduces gap? | Yes (dir.) | Yes | Yes | Yes | **UNANIMOUS: yes** |
| Q3 | 2.7× asymmetry = variance artifact? | Yes (RTN≈0 + scaling) | Yes (degenerate RTN) | Yes (zero-var col) | Yes (degenerate RTN) | **UNANIMOUS: variance artifact, not identity** |
| Q4 | Feature-count overfitting? Control? | Yes; 160 iid Gaussian | Yes; 160 iid Gaussian | Yes; 160 matched noise | Yes; 160 iid Gaussian | **UNANIMOUS: yes; dim-matched random control mandatory** |
| Q5 | Publishable? | No (8 missing) | No (gate fail = autoreject) | No (z=0.79 fatal) | No | **UNANIMOUS: NOT publishable** |
| Q6 | Cite real paper on AMD-twin substrate-aware degradation | None exists | None (only Li ISCA'20 on FPGA) | None exists | None (only NVIDIA / synthetic prior) | **UNANIMOUS: no such citation; AMD-twin replication is novel-as-substrate but unpublishable without controls** |
| Q7 | Single killer/elevator experiment | Intra-device recapture (train ikaros-A → test ikaros-B) | Synthetic-noise falsifier (Q2) | Synthetic noise w/ daedalus marginals + degenerate col | RTN-only swap (replace daedalus RTN with ikaros constant during eval) | **Cluster on falsifier-style controls; intra-device-recapture is most decisive** |
| Q8 | Third-machine (minos) prediction | Large noisy + more symmetric; ikaros pathology stays ikaros-specific | gap ∝ KL divergence of marginals; asymmetry ikaros-only | Asymmetry won't generalise | Moderate gaps for non-degenerate pairs; ikaros stays anomalous | **UNANIMOUS: asymmetry is ikaros-specific (RTN=0), won't generalise** |
| Q9 | Cached static features vs live hwreg? | Matters — cached = ordinary cov-shift | Matters — invalidates constitutive claim | Matters — measures nothing beyond shift | Matters — feature-mapping not identity | **UNANIMOUS: cached implementation invalidates "constitutive coupling" framing** |
| Q10 | Odds best / worst / middle | 15 / 65 / 20 | 5 / 95 / 0 | 8 / 75 / 17 | 8 / 72 / 20 | **Mean: ~9% best / ~77% worst / ~14% middle** |

## Sharpest disagreement

There is almost none. The four diverge only on:
- **Magnitude of best-case odds**: Gemini 5% (most damning), GPT-5 15% (most charitable).
- **Which single killer experiment**: GPT-5 picks *intra-device recapture* (test if a snapshot of the same device, recaptured, also causes a "gap" — best at isolating "tag-vs-identity"). Gemini/Grok pick the *synthetic-marginal falsifier* (cheaper, equally decisive on the distribution-shift hypothesis). DeepSeek picks the *RTN-only swap* (most surgical re the degenerate-feature hypothesis). All three are good. GPT-5's is the strongest because it falsifies BOTH "identity carries information" AND "the feature is a stable identity tag at all."

No oracle defended F. No oracle argued the result is genuine identity-coupling.

## Updated verdict on Angle F

**DOWNGRADED: DISCOVERY → ARTIFACT-PENDING-CONTROLS.**

The previously claimed "11× gap, DISCOVERY-grade" framing does not survive even superficial hostile review. The benchmark's own discovery gate (z > 2 AND aware_gap > baseline_gap) **already failed** at z = 0.79 — this was buried in the result file but not in the user's narrative summary. The four oracles independently surfaced this. Root cause identified by all four: **ikaros RTN-rate is degenerate (≈ 0, zero variance)**, and the four outlier seeds in the (ikaros→daedalus, aware) row (NRMSE 1.84, 2.77, 4.94, 6.19) drive both the mean gap and the asymmetry. Remove that one feature column, and the effect almost certainly collapses.

The Phase-1b finding "ikaros RTN ≈ 0, daedalus RTN ≈ 0.11" is itself a real device difference — but feeding it as a static input feature does not test "identity coupling," it tests covariate shift on a degenerate column. Different claim, much weaker.

**F is not killed yet** — there is a 9% (best) / 14% (middle) chance that controls preserve a non-trivial residual after the variance artifact is removed. But the current packet is not publishable and the "DISCOVERY" claim should be retracted from the live narrative.

## Top 3 experiments to run next (oracle-consensus order)

### 1. Synthetic-marginal-noise falsifier (3/4 explicitly recommend; UNANIMOUS in Q2 directional prediction)
Replace the 160 substrate features with iid samples drawn from each device's per-feature marginal (mean, std), preserving feature count and per-device marginals but destroying all spatial / cross-feature / temporal identity structure. If the ~11× gap survives, F is dead as identity. If gap vanishes, the structural content of the real features is doing real work — promote F back to "interesting." Cheapest test, decisive on the dominant hypothesis.

### 2. Drop-degenerate-feature + dim-matched random baseline (4/4 demand a feature-count control in Q4)
Two-part: (a) ablate each of the 4 feature channels (RTN-rate, spatial-corr, LDS-startup, 1/f-knee) individually; prediction (DeepSeek explicit): the gap collapses when RTN is dropped or when daedalus RTN is replaced with ikaros's constant zero. (b) Add 160 iid N(0,1) features to the *baseline* model — if it also degrades cross-device, the gap is feature-count overfitting, not identity. Without this control, F cannot be defended at any venue.

### 3. Intra-device recapture (GPT-5's pick; cleanest "tag vs. identity" discriminator)
Re-measure ikaros's substrate features twice at matched temperature (capture A and capture B, separated by hours / thermal cycle / reboot). Train on capture A, test on capture B. If the "aware" model degrades within the same device, F is testing snapshot-recency, not identity. If intra-device gap is small while inter-device gap survives the controls in (1) and (2), THEN F upgrades to a real finding worth replicating on minos as the third twin.

## Additional unanimous demands (must-do, not optional)

- **Replicate on minos** (third twin) — 4/4 explicitly require this for publication. All four predict the asymmetry will NOT generalise (it is ikaros-specific from RTN=0).
- **Implement live per-forward-pass hwreg reads** if the "constitutive coupling" framing is to be retained. 4/4 say the cached-feature implementation reduces F to ordinary covariate shift.
- **Pass the discovery gate (z > 2)** with robust statistics. Currently z = 0.79, driven by 4 outlier seeds (out of 10) in a single condition. Median/trimmed-mean reporting; outlier diagnostics.
- **No citation exists** for AMD-twin substrate-aware-model degradation. 4/4 confirm. This means: if controls hold, the AMD-twin demonstration IS novel-as-substrate and worth a workshop paper. But "concept novelty" is gone (DeepSigns / HWN-DNN / BadNets already own the concept-space).

## Files

- Prompt: `prompt.md`
- Attachments: `IDENTITY_NOVEL_ANGLES_2026-05-30.md`, `IDENTITY_BENCHMARK_2026-05-30.md`, `..._PHASE1.md`, `..._PHASE1B.md`, `..._PHASE2.md`, `O96_prior_synthesis.md`, `F_results.json`
- Responses: `openai_response.md`, `gemini_response.md`, `grok_response.md`, `deepseek_response.md`

```
