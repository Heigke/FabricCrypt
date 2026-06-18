# IDENTITY_BENCHMARK 2026-05-30 — Novel V2 (B + ECC + Pre-step)

Date: 2026-05-30
Author: novel_v2 agent
Status: COMPLETE — all three probes ran; all returned NULL or weakening evidence
Sister run: novel/ (F+J+C agent) — independent, in parallel

---

## Summary

Three probes recommended by oracle synthesis O96, executed thermal-safely on
ikaros + daedalus (both Strix Halo gfx1151). **All returned NULL or actively
weakened Phase 1b**. We are no closer to a per-device identity signature; we
have, however, narrowed the hypothesis space.

| Probe | Verdict | Cost |
|---|---|---|
| Pre-step (RTN vs spatial-corr) | MIXED — weakens Phase 1b | <1 s |
| Angle B (Lorenz per-CU trajectory) | NULL — cross < within | ~3 min/device GPU |
| Angle ECC (GDDR/DDR EDAC map) | NULL — platform-level falsified | 60 s/device CPU |

---

## 1. Pre-step: RTN vs spatial-corr collinearity

Artifact: `results/IDENTITY_BENCHMARK_2026-05-30/novel_v2/rtn_spatial_correlation.json`

**Verdict: MIXED (weakens Phase 1b)**

- **Naive pooled Pearson r = 0.95** — but this is a Simpson's paradox artefact
  driven by the device-mean offset (ikaros RTN ≡ 0, daedalus RTN ≈ 0.11).
- **Within-device z-scored pool r = 0.052** — the correct test. RTN and
  spatial-corr are **independent** where both signals exist.
- **Data-quality flag**: on ikaros the entire RTN array is exactly zero
  (`rtn_mean = rtn_std = 0`). The probe is degenerate on this device.
  Phase 1b's "2 surviving channels" claim therefore does not hold across both
  devices — RTN is not even measurable on one of them.

Consequence: Phase 1b is **down to ~1.5 channels** in practice. The
spatial-corr matrix survives; RTN survives only on daedalus.

---

## 2. Angle B — Per-CU Lorenz trajectory fingerprint

Artifacts:
- `scripts/identity_benchmark/novel_v2/B_lorenz.hip`, `B_runner.py`, `B_compare.py`
- `results/.../novel_v2/B_lorenz_{ikaros,daedalus}.{json,npz}`
- `results/.../novel_v2/B_lorenz_compare.json`

**Verdict: NULL**

Setup: per-wave RK4 Lorenz (σ=10, ρ=28, β=8/3, dt=0.01, fp32), 10 000 steps,
last 256 tail states per CU, 100 batches × 1024 waves on each device. Common
physical CU indices (decoded from `HW_ID1`): 12 on both devices.

Key numbers (`B_lorenz_compare.json`):

| Metric | Value |
|---|---|
| Within-device per-CU tail-mean L2 std | ikaros 0.0297, daedalus 0.0298 |
| Cross-device per-CU L2, mean / max | 0.0055 / 0.0163 |
| Device-mean trajectory L2 | 0.0023 |
| ratio cross/within (mean) | **0.19** (gate is 3.0) |
| Trajectory dynamic range | 44.89 on both devices |
| Lyapunov estimate | 0 (see note) |

Cross-device differences are an order of magnitude **smaller** than within-device
batch-to-batch variation. This is consistent with IEEE-754 determinism: same CU
hardware path → bitwise-identical FP result, regardless of which physical chip.
The hypothesis that "per-CU FP rounding accumulates into a per-device trajectory"
is falsified on this architecture/precision.

Lyapunov note: lane-0 trajectories with identical IC are bitwise-deterministic
across batches on a given CU, so the divergence-rate estimator returns 0. The
positive Lyapunov of the underlying Lorenz system is masked by the determinism
of the implementation. A future probe would need either (a) thermal-induced
clock jitter that nudges floating-point chain order, or (b) explicit
non-deterministic atomic reductions to surface micro-architectural variation.

Thermal: ikaros max 51 °C, daedalus max 71 °C with two automatic pauses at the
70 °C threshold. Thermal guard never triggered. **Zero incidents.**

---

## 3. Angle ECC — DDR/GDDR EDAC bad-block map

Artifacts:
- `scripts/identity_benchmark/novel_v2/ECC_probe.py`
- `results/.../novel_v2/ECC_{ikaros,daedalus}.json`

**Verdict: NULL — platform-falsified**

Both devices show EDAC subsystem present but **zero memory controllers
registered**. Strix Halo APUs use unified DDR5 system memory and do not expose
per-channel ECC counters via `/sys/devices/system/edac/mc/`. The discovery gate
(≥10 unique error cells, ≥50 % non-overlap) cannot be evaluated — there is no
signal channel at all.

This is a cheap, definitive negative for the angle on this platform. It does
not rule out the approach in general (discrete-GPU GDDR6 with EDAC drivers
would expose it), but it does rule it out for our hardware.

---

## 4. Cross-angle synthesis

No DISCOVERY. Net effect of this hour of work:

1. We have shown Phase 1b's "2 channel" claim is partially fragile (1 channel
   on ikaros, 2 on daedalus).
2. We have shown that the most physics-flavoured probe (per-CU chaotic
   trajectory) **cannot** discriminate gfx1151 instances within fp32 RK4
   determinism. The "silicon-FP variance" hypothesis is dead at this precision.
3. We have shown that DDR/GDDR ECC bad-block fingerprinting is not even an
   available channel on Strix Halo APUs.

Combined with the parallel novel/ run (F, J, C — known to be theatre/duplicate
per oracle), and with Phase 2's NULL, we are now substantively **further from
identity than before**, in the sense that more plausible-sounding channels have
been eliminated.

## 5. Thermal incidents

**Zero.** ikaros peaked 51 °C, daedalus peaked 71 °C (two auto-pauses cleared
at 70 °C threshold). Thermal guard PID 9305 did not SIGSTOP.

## 6. Recommendation

**Pivot away from identity-PUF research on this hardware**, or pivot the
*question*. Specifically:

- The "deterministic computation on identical chips produces a per-device
  signature" line of inquiry is empirically running out of channels. Phase 1b
  squeezed marginal signal out of spatial-corr (which may itself be reflecting
  a shared `gpu_metrics` artefact rather than silicon). B shows that even a
  chaotic amplifier cannot surface FP-level silicon variation. ECC is
  unavailable. F/J/C are already discounted.
- Remaining plausible channels are all **thermal/analog** — clock-jitter under
  load, VRM noise, dV/dT response — which is the territory FEEL/NS-RAM already
  occupies under a different research banner. Identity per se is the wrong
  framing for what's actually probe-able.
- If a Phase 3 is to exist, it should **drop the "identity"/PUF goal** and
  reframe as: *"What components of analog substrate state on Strix Halo APUs
  are reliably measurable from userspace, with what bit-rate?"* — i.e., merge
  this back into the FEEL telemetry programme rather than continuing as a
  parallel identity track.

If you'd rather keep the identity track alive, the only design space left
that hasn't been falsified is:

- **Stress-induced clock jitter + non-deterministic atomic ordering** (deliberately
  break the determinism that killed B), or
- **GDDR-only systems** for ECC (requires different hardware, not our APUs).

Both are expensive enough that the FEEL pivot is the better use of budget.

---

## Files

- `scripts/identity_benchmark/novel_v2/00_rtn_spatial_corr.py`
- `scripts/identity_benchmark/novel_v2/B_lorenz.hip`
- `scripts/identity_benchmark/novel_v2/B_runner.py`
- `scripts/identity_benchmark/novel_v2/B_compare.py`
- `scripts/identity_benchmark/novel_v2/ECC_probe.py`
- `results/IDENTITY_BENCHMARK_2026-05-30/novel_v2/rtn_spatial_correlation.json`
- `results/IDENTITY_BENCHMARK_2026-05-30/novel_v2/B_lorenz_{ikaros,daedalus}.{json,npz}`
- `results/IDENTITY_BENCHMARK_2026-05-30/novel_v2/B_lorenz_compare.json`
- `results/IDENTITY_BENCHMARK_2026-05-30/novel_v2/ECC_{ikaros,daedalus}.json`
