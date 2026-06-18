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
