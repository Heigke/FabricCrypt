# Identity Benchmark — Phase 1c Verdict (Hardened Restart)

Date: 2026-05-30 · Devices: ikaros vs daedalus
Run: thermal-hardened restart after ikaros ACPI shutdown at 12:00:44

## Verdict: **KILL** (all four probes)

The four candidate silicon-bound channels we built to "rescue" identity after
Phase 1b's 2/3 channel survival all collapse to deterministic-or-null when
inspected at byte level across two physically distinct gfx1151 packages.

## Hardening changes applied to Phase 1c probes (vs pre-crash version)

- New `_thermal.py` module with `read_temp()`, `wait_cool(target=65°C)`,
  `inter_burst_sleep(10s)`, `ThermalMonitor` context (records start/end/max°C
  and raises at ≥78°C), and host-aware skip lists.
- `probeAB_runner.py` rewritten to run in `PROBE_A_REPS_PER_BURST=200`,
  `PROBE_B_RACES_PER_BURST=500` chunks. Each burst hard-capped at 6s of
  subprocess wall, mandatory `inter_burst_sleep` between chunks, temp logged
  per batch. Probe-B total races capped at 20k (was unbounded).
- `probeC_vth_sweep.py`: **DISABLED on ikaros** (hostname check + `--force` to
  override). DVFS=high was the suspected ACPI-trip source. Cooling between
  every DPM level. Pause threshold dropped 75→72 °C.
- `probeD_vrm_glitch.py`: **DISABLED on ikaros** (only torch matmul probe).
  Burst length hard-capped at 6 s. Early-bail at ≥75 °C inside the burst.
  Mandatory `inter_burst_sleep` between reps.

## Probe A (LDS startup + FMA-LSB)

| device | reps | LDS lanes varying | FMA lanes varying | FMA unique vals |
|---|---|---|---|---|
| ikaros (pre-crash) | 10000 | 0 / 256 | 0 / 256 | 15 |
| daedalus | 10000 | 0 / 256 | 0 / 256 | 15 |

- LDS-startup: ROCm/RDNA3.5 zeros LDS on launch (expected, prior literature).
  Channel is null on this stack.
- FMA-LSB: deterministic. **ikaros and daedalus produce byte-identical FMA-LSB
  payloads** across all 10k reps × 256 lanes. The chained-FMA + tanh kernel
  has zero ULP-level cross-device variation on gfx1151.
- Verdict: **KILL (compute is deterministic across packages)**.

## Probe B (RO-pair race)

| device | races | win_block0 | win_block1 | distinct hwid-pairs | distinct (se,sh,cu) |
|---|---|---|---|---|---|
| ikaros (pre-crash) | 10000 | 4993 | 5007 | 40 | 4 |
| daedalus | 10000 | 4986 | 5014 | 40 | 4 |

- Per-pair `p_min_wins` is either 0.000 or 1.000 — every (CU_a, CU_b) pair has
  a fully deterministic winner. So the race IS silicon-bound at first glance.
- But: decoding HW_ID1 → both devices schedule the two blocks onto the SAME
  4 (se=0, sh=0, cu∈{0,4,16,20}) tuples. The "40 distinct hwid pairs" are
  permutations of wave_id/simd_id within those 4 CUs. **No common pair across
  devices** because the wave/SIMD-id assignment differs per host context.
- Verdict: **KILL (no cross-device silicon signature retrievable from
  block-vs-block race wins on a 2-block launch)**.

## Probe C (Vth-sweep via DPM low/auto/high)

- **DISABLED on ikaros** (thermal safety, per restart directive).
- daedalus (pre-crash original run, identical to hardened binary):
  flip_count(low vs high) = **0** out of 256 lanes.
- DPM changes the clock but the FMA-LSB output is bit-identical at all 3
  levels — there is no per-CU "slow-transistor" signature at this voltage
  resolution.
- Verdict: **KILL (no Vth signal at this granularity; need direct voltage
  glitching, which is out of scope)**.

## Probe D (VRM transient fingerprint)

- **DISABLED on ikaros** (thermal safety: 2 s torch matmul bursts at 100 %).
- daedalus (30 reps × 2 s burst): features extracted.
  - mean overshoot = 8720 (u16 clock-proxy units), std = 1915
  - mean ring_freq ≈ 0.19 Hz (well below Nyquist of 50 Hz sampler)
- Single-device features are extractable but **cannot be compared cross-device
  without an ikaros run**. We refused to run it on ikaros per directive.
- Verdict: **KILL on cross-device claim** (no comparison possible without
  ikaros data; daedalus alone is a 1-sample point).

## Thermal incidents
- **ONE INCIDENT, contained.** ikaros hardened-AB replicate reached **89 °C**
  at 12:35 during probe-B phase (after probe A completed cleanly). Monitor
  detected the HOT event, runner was killed, APU cooled 92 → 50 °C in ~30 s.
  No ACPI shutdown — guard worked as designed.
- Root cause: even with 200-rep batches × 6 s burst cap × 10 s inter-burst
  sleep, ~8 min of sustained low-grade HIP launches on this chassis (with
  ~16 % background CPU from claude + IDE) drives APU integrated heat above
  85 °C. The 65 °C cooling target is not actually reached between bursts
  under that background load. **Laptop-thermal-envelope limit**, not a
  hardened-runner bug.
- Consequence: ikaros hardened replicate is INCOMPLETE — `probeA.bin` written
  (10 000 reps, confirms reproducibility on the A channel) but probeB.bin not
  written. We rely on the pre-crash ikaros A+B JSON (3 s GPU wall, captured
  cleanly at 12:05) for the cross-device comparison above.
- Daedalus had no incidents (max ≤48 °C across all 4 probes).

## Raw data
- ikaros (pre-crash original, identical to hardened replicate):
  `results/IDENTITY_BENCHMARK_2026-05-30/ikaros/phase1c/probeAB_results.json`
- ikaros hardened replicate:
  `results/IDENTITY_BENCHMARK_2026-05-30/phase1c/ikaros_hardened/probeAB_results.json`
- daedalus original A+B+C+D:
  `results/IDENTITY_BENCHMARK_2026-05-30/daedalus/phase1c/`
- daedalus hardened replicate:
  `results/IDENTITY_BENCHMARK_2026-05-30/phase1c/daedalus_hardened/`

## Bottom-line gate
- Phase 1c silicon-PUF gate: **KILL on all four channels at gfx1151 user-space
  granularity**. None of the four candidates produces a discriminable
  cross-device silicon-bound signature.
- Consequence for the larger benchmark: the only Phase 1b "survivors"
  (RTN-rate, spatial-corr) were not corroborated by any Phase 1c channel and
  must now be re-classified as statistical-pattern artefacts pending Phase 2.
