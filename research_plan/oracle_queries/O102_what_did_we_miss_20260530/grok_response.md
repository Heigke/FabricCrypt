# grok response (grok-4-latest) — 11s

**Synthesis: single highest-EV method-class**

**Active degradation / wear-as-training (Substrate-as-ACTIVE-DEGRADATION + Substrate-as-TEMPORAL-HISTORY).**

**Rationale (why this dominates all alternatives under the given constraints)**

- Every one of the 14 attacks (and all B/M mechanisms) treated the substrate as a *read-only signal* that the model consumes. This class is definitionally replaceable by SW-matched noise, permutation, or another device’s statistically similar trace; the SHUFFLE ≈ HW result in regime 5 of `IDENTITY_CONSTITUTIVE_2026-05-30.md` is the inevitable outcome.
- The only remaining method-classes that break the “fungible structured noise” trap are those that make the substrate *causally downstream* of the model’s own actions: the model must *write* to the hardware (cache/TLB hot-spots, sustained V/F stress, targeted thermal cycling) so that the resulting NBTI/HCI/electromigration state becomes part of the learned parameters. Any transplant then requires either (a) identical cumulative wear history or (b) re-training on the new device’s altered silicon. This satisfies the constructive gate demanded in the postmortem (“model M produces Y *only* on device D”).
- Hardware reality matches exactly: the two Z2 Mini G1a twins already expose the necessary levers (MSR undervolting, `amdgpu` power/thermal caps, `turbostat`/`rdmsr` per-core V/F, `cpufreq`/`amdgpu_pm` P-state forcing). No new silicon, no FPGA, no external ADC. 100 wall-hours is sufficient for a multi-day closed-loop run.
- Literature alignment (Karnik/Mintarno/Vaisband 2024+ aging-aware CAD, plus the NBTI/HCI sections in B16) shows measurable Vth shift on the order of weeks under controlled stress; the same mechanisms are already visible in the per-core latency and thermal-τ channels that survived earlier sweeps.
- Expected value is highest because (1) it is the only class never attempted, (2) it directly answers the architectural assumption hunt in Q1, (3) it is executable inside the 100-hour envelope with only existing ROCm/sysfs interfaces, and (4) a positive result yields an irreversible, publishable constitutive binding while a negative result still produces a clean “even active wear fails on modern guarded APUs” paper.

All other candidates (VCEK-as-weight, joint SCA, approximate-compute emulation, fan-dynamics, etc.) either remain in the “read-only signal” regime, require hardware not present, or have already been shown to be fungible. Active degradation is the single pivot that escapes the 14-attack meta-pattern.
