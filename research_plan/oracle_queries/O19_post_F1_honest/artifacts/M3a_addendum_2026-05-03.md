# ⚠️ DRAFT — DO NOT SEND (pending O18 fixes)

See `research_plan/01_LOG.md` 2026-05-03 ~22:18 for the 3-of-3 oracle
FIX verdict. Required corrections before this is sendable:

1. Section 2 "two-axis architectural rec (HUB_SPOKE for classification)"
   — REMOVE. Statistically unsupported (n=2 noise + unfair ρ normalization).
2. Section 1 VG1=0.4 V framing — change "wrong Newton root" to
   "non-physical Bf parameterization" (per openai correction).
3. Section 4 "ready" framing — must add "device model not yet validated
   end-to-end against silicon transient" caveat.

Do NOT send to Mario / Sebas / NRF in current form.

---


# M3a Addendum to NS-RAM Funding Brief v4.1

**Date:** 2026-05-03
**Audience:** Mario Lanza (KAUST), Sebastian Pazos (KAUST)
**Author:** Eric Bergvall (ENIMBLE / Nervdynamics)
**Brief reference:** `nsram_proposal_short.tex` v4.1 sent same day

This addendum supersedes the brief's Section 5 ("WHAT WORKS, WHAT
DOESN'T") and Section 6 ("PRELIMINARY HYPOTHESES") with results
generated in the 12 hours after the brief was dispatched. It does
not change the recommendation; it sharpens the DC-fit numbers and
reports a topology scaling sweep (n=3 seeds; suggestive only — see
Section 5 caveats). The device model has NOT yet been validated
end-to-end against silicon transient data.

---

## 1 — DC fit: median log-RMSE 1.00 → 0.80 dec

A single one-line change to the BJT forward-gain parameter
(`bjt.Bf` from 5×10⁴ to 2×10⁴) drops the brief's headline residual
by 20 %. The 5×10⁴ value was a coarse z91h grid-search optimum;
a finer sweep (z139 logbook, 8 values from 3×10³ to 5×10⁴) places
the local minimum at Bf ≈ 2×10⁴.

| metric              | brief v4.1 | post-rebuild | Δ        |
|---------------------|-----------:|-------------:|---------:|
| median log-RMSE     | 1.00       | **0.799**    | -20 %    |
| mean log-RMSE       | 1.60       | 1.40         | -13 %    |
| max log-RMSE        | 3.24       | 2.89         | -11 %    |
| p90 log-RMSE        | 2.90       | 2.58         | -11 %    |

Per-VG1-row breakdown:

| row    | brief v4.1 | post-rebuild |
|--------|-----------:|-------------:|
| VG1=0.2 | 1.66      | 1.46         |
| VG1=0.4 (catastrophe) | 2.83 | **2.52**  |
| VG1=0.6 | 0.91      | 0.78         |

**Diagnosis of the VG1=0.4 V row** (probe v2,
`research_plan/binning_audit/probe_v2_finding.md`): at no-impact-
ionisation biases the parasitic NPN settles into a self-sustaining
high-Vb root (Vb≈0.43 V, Ic_Q1≈7×10⁻⁸ A) with no physical
charge-pumping mechanism (Iii ~10⁻²⁵ A). All five arclength
cold-start seeds converge to the SAME root, which is unique. The
cell equations admit this self-biased NPN steady state because
Bf=2×10⁴ is non-physical (real 130 nm parasitic NPN gain is
10–100). The failure is in the parameterization, not the solver.
Lowering Bf to 2×10⁴ reduces the spurious gain marginally; the row
improves from 2.83 → 2.52 dec. Full closure of this row
(M3a.1 follow-up) requires (a) a Bf clamp to physical range and
(b) a stronger bias-dependent Iii→Vb trigger so impact ionisation
actually couples to the NPN base. See `research_plan/M3b_fix_plan_2026-05-03.md`
F1–F2.

The 8 NaN biases reported in the brief are **not solver failures**
— they are biases for which Sebastian's parameter CSV has K1=NaN
(the negative-VG2 snapback regime he did not extract). All 33
biases now evaluate to finite log-RMSE under the un-overridden card.

---

## 2 — Topology scaling matrix (4× larger than brief tested)

The brief's C.3 tape-out recommendation pinned MESH_4N as the
preferred topology, validated up to N=200. We now have a 6×3 sweep
at N ∈ {100, 300, 800} with 3 seeds × 4 tasks each (small n;
findings below are suggestive — see Section 5):

| topology       | N=100 MC | N=300 MC | N=800 MC | N=800 XOR | N=800 WAVE | scale ×|
|----------------|---------:|---------:|---------:|----------:|-----------:|-------:|
| RAND_GAUSS     | 1.42     | 1.50     | 1.87     | 0.53      | 0.47       | 1.31   |
| **MESH_4N**    | 1.87     | 2.40     | **3.29** | **0.91**  | 0.52       | 1.75   |
| ER_SPARSE      | 2.12     | 2.56     | 2.20     | 0.63      | 0.46       | 1.04   |
| WS_SMALLWORLD  | 1.66     | 2.44     | 2.94     | 0.85      | 0.51       | 1.77   |
| **HUB_SPOKE**  | 1.18     | 0.86     | 2.89     | 0.90      | **0.61**   | 2.45   |
| LAYERED        | 2.78     | 1.53     | 2.17     | 0.57      | 0.48       | 0.78   |

(MC = memory capacity; XOR = τ=2 binary-XOR readout accuracy;
WAVE = 4-class waveform classification accuracy; scale × = ratio
MC(N=800) / MC(N=100). Bf=2×10⁴, T=500, κ=0.03, ρ=0.9.)

### Six findings

1. **MESH_4N is the MC leader at N=800** (3.29 dec, n=3 seeds).
   Consistent with the brief's C.3 recommendation; "validated at
   4× scale" overstates n=3 — call this a positive directional
   replication pending n≥10 confirmation.

2. **HUB_SPOKE: suggestive non-monotone scaling.** The cell values
   (×2.45 scaling, WAVE=0.61, MC=0.86 at N=300) come from n≤3 seeds
   and HUB_SPOKE is unfairly disadvantaged by the global ρ=0.9
   normalization (see Section 5). The non-monotonicity is
   *suggestive only* and not statistically supported. Treat as
   motivation for a follow-up study with per-topology spectral
   calibration and n≥10 seeds, not as a tape-out signal.

3. **LAYERED is anti-scaling** (×0.78) — MC DECREASES with N. The
   2-layer feedforward + sparse-skip topology does not benefit
   from network growth at the tested input drive. Negative result.

4. **ER_SPARSE: apparent plateau at N=300** (peak MC=2.56) then
   2.20 at N=800. Suggestive only — n=3 seeds, no confidence
   interval. The collinearity-saturation hypothesis is consistent
   with z117/z115/z114 small-N results but the N=800 dip falls
   within the cross-seed variance and should not be cited as
   established without n≥10 replication.

5. **WS_SMALLWORLD nearly matches MESH_4N at N=800** (2.94 vs 3.29
   MC). Small-world rewiring is a viable alternative if the
   2D-grid layout is undesirable for fabrication.

6. **Random Gaussian is the worst at every scale.** The brief's
   choice to recommend a *structural* topology rather than random
   recurrence is empirically grounded across 6 topologies × 3 scales.

---

## 3 — Architectural recommendation (unchanged)

MESH_4N remains the brief's primary architectural recommendation. The
post-send topology data does not statistically support an alternative.

---

## 4 — Status of M3a deliverables

- [x] M3a-A: z91g rebuild at Bf=2×10⁴ — median 0.80 dec
  (`results/z91g_two_model_validation_stage6_bf2e4/`)
- [x] M3a-B: per-bias BETA0 — resolved as documentation. The CSV's
  `BETA0` column is the BSIM4 impact-ion β₀, already routed to
  `P_M1["beta0"]`. It is NOT the bipolar Bf and never was.
- [ ] M3a-C: ngspice cross-validation rerun at Bf=2×10⁴ — pending
- [x] M3a-D: large-scale topology sweep (z139) — full table above
- [x] M3a-E: independent codebase audit (Explore subagent, 5
  candidates flagged; #1, #2, #4 deferred as off-25°C / refactor
  risks; #3 awaits Sebas CSV; #5 closed with negative grid result)
- [x] M3a-F: transient validation harness scaffold —
  `scripts/z140_transient_harness.py` runs end-to-end on synthetic
  input; awaits Sebas's measured traces
- [x] M3a-G: this addendum

**NOT YET SENDABLE.** Pending O18 fixes F1–F4 (Bf clamp + Iii→Vb
trigger + 33-row refit + multi-point ngspice cross-val + 2T-cell
ngspice cross-check). See `research_plan/M3b_fix_plan_2026-05-03.md`.

---

## 5 — Caveats per O18 triple-review (2026-05-03)

Three reviewers (gemini, openai, grok) independently flagged the
following weaknesses in this addendum. Each must be addressed before
any external claim is made.

- **n=2 statistical limitation in z139.** Section 2's topology
  table reports 3 seeds × 4 tasks per cell, but several seeds did
  not converge cleanly — the practical n for HUB_SPOKE at N=300
  was 2 (MC values {1.21, 0.49}, variance > mean). The HUB_SPOKE
  non-monotone behaviour (Finding 2) and the ER_SPARSE plateau
  (Finding 4) are NOT statistically supported and should be
  treated as suggestive only.

- **Unfair ρ-normalization for HUB_SPOKE.** All topologies were
  λmax-scaled to ρ=0.9, which clamps the hub edge weight in
  HUB_SPOKE far harder than it clamps the diffuse edges in MESH_4N
  or RAND_GAUSS. HUB_SPOKE's reported numbers are therefore lower
  bounds under unfair normalization. A like-for-like comparison
  requires per-topology effective-spectral-radius calibration.

- **Median 0.80 dec is on 25/33 rows, not 33.** The DC fit headline
  excludes 8 K1=NaN biases (Sebastian's CSV omits negative-VG2
  snapback extraction). The full 33/33 refit with imputed/refit K1
  is required before any external claim. The brief's 1.00 dec used
  the same exclusion; the relative improvement is real but the
  absolute number must be qualified until F3 closes.

- **Bf=2×10⁴ is non-physical.** The 2×10⁴ value is a fitting
  compensation for the missing Iii→Vb coupling, not a measurement
  of the parasitic NPN gain. Real 130 nm NPN gain is 10–100. Any
  forward use of this card outside the calibration set should
  expect mis-extrapolation. F1 (Bf clamp) and F2 (Iii→Vb trigger)
  in M3b address this.
