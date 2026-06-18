# Sebas — thick-ox cell card + 7-rate transient data request (M12)

**To:** Sebastian Pazos (KAUST). Cc: Mario Lanza.
**Status:** DRAFT, send TOGETHER WITH `sebas_silicon_characterisation_request.md`.

> **Audited 2026-05-09**: still valid as written. No claims tied to
> the z230/z231/z232 corrections; thick-ox card request is forward-
> looking (M12 milestone), 7-rate transient is τ-spectrum extraction.
> Bundle with main request as drafted.
**Goal:** Phase-M12 deliverable in NRF brief (thick-oxide $V_{G2}\in[2.5, 3.0]$~V,
$1$~V drain input-neuron variant). Currently the model has only the
thin-ox cell card; we cannot model the soma/input-neuron path of the
NS-RAM cell family until thick-ox params arrive.

---

## What we need

1. **Thick-ox cell card** (M2 variant with $V_{G2}\in[2.5, 3.0]$~V):
   - SPICE/BSIM4 model card, same format as `M2_130bulkNSRAM.txt`
     you sent 2026-04-22 for the thin-ox cell.
   - Specifically: `tox`, `vth0`, `k1`, `etab`, `voff`, `nfactor`,
     and any bulk- vs. surface-junction differentiation.
   - Geometry: confirm $L = 17~\mu$m$^2$ thick-ox cell area, channel
     width/length per the M12 layout.

2. **7-rate transient I-V**:
   - Same $V_{G1}$, $V_{G2}$ corners as the 33-bias DC sweep, but
     swept at 7 different $V_d$ ramp rates (e.g., $0.1$, $1$, $10$,
     $100$, $10^3$, $10^4$, $10^5$ V/s).
   - At least 5 biases × 7 rates = 35 transient curves.
   - We will use these to extract the body-region time-constant
     spectrum and validate the implicit transient solver
     (M3a-F harness, currently calibration-pending).

---

## Why this matters

- **Brief milestone M12** (Mar 2027): "thick-oxide neuron-cell variant
  modeled and benchmarked" — we cannot start until the card lands.
- **Network demos**: the current 64-cell ER\_SPARSE result uses
  thin-ox cells only. Thick-ox enables the input-neuron layer
  (rapid integrate-and-fire), without which the network can't
  ingest external signals at non-trivial rates.
- **Validation depth**: 7-rate transient is the only way to constrain
  $\tau$ over the 5+ orders of magnitude where the cell's memory
  decay matters. Single-rate transients only pin one point on the
  decay curve.

---

## Practical asks

1. **Rough ETA**: even "next week" or "in 4 weeks" helps us
   sequence the M12 schedule.
2. **Format**: SPICE deck for the card; CSV (one row per `(t, Vd, Id)`)
   for the transients. Same layout as your 2026-04-22 drop.
3. **Fallback**: if the thick-ox card isn't ready, we can defer M12
   to Phase-2 of the NRF project — but it's the cleanest cut for
   a complete cell-family demo.

---

*Drafted 2026-05-07 by Eric. Bundle with the silicon-characterisation
request when sending — both are "no new fab" measurements on existing
silicon.*
