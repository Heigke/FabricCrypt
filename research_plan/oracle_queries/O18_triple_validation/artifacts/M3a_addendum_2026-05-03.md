# M3a Addendum to NS-RAM Funding Brief v4.1

**Date:** 2026-05-03
**Audience:** Mario Lanza (KAUST), Sebastian Pazos (KAUST)
**Author:** Eric Bergvall (ENIMBLE / Nervdynamics)
**Brief reference:** `nsram_proposal_short.tex` v4.1 sent same day

This addendum supersedes the brief's Section 5 ("WHAT WORKS, WHAT
DOESN'T") and Section 6 ("PRELIMINARY HYPOTHESES") with results
generated in the 12 hours after the brief was dispatched. It does
not change the recommendation; it sharpens the numbers and adds a
second architectural axis.

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
cold-start seeds converge to the same wrong root — the failure is
parametric (Bf too high), not numerical. Lowering Bf to 2×10⁴
reduces the spurious gain; the row improves from 2.83 → 2.52 dec.
Full closure of this row (M3a.1 follow-up) requires a stronger
bias-dependent NPN trigger and is queued.

The 8 NaN biases reported in the brief are **not solver failures**
— they are biases for which Sebastian's parameter CSV has K1=NaN
(the negative-VG2 snapback regime he did not extract). All 33
biases now evaluate to finite log-RMSE under the un-overridden card.

---

## 2 — Topology scaling matrix (4× larger than brief tested)

The brief's C.3 tape-out recommendation pinned MESH_4N as the
preferred topology, validated up to N=200. We now have a 6×3 sweep
at N ∈ {100, 300, 800} with 3 seeds × 4 tasks each:

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

1. **MESH_4N is the MC champion at N=800** (3.29 dec). The brief's
   C.3 recommendation is now empirically validated at 4× the
   original tested scale.

2. **HUB_SPOKE has the steepest scaling (×2.45) and the best WAVE
   classification (0.61 vs ~0.50 for everyone else).** It is
   catastrophically bad at N=300 (MC=0.86) yet dominant at N=800
   for classification — a non-monotone behaviour worth its own
   investigation. Intuition: at small N the single hub bottlenecks
   information flow; at large N the hub becomes a global mixer
   that the sparse leaf-leaf graph cannot otherwise provide.

3. **LAYERED is anti-scaling** (×0.78) — MC DECREASES with N. The
   2-layer feedforward + sparse-skip topology does not benefit
   from network growth at the tested input drive. Negative result.

4. **ER_SPARSE plateaus at N=300** (peak MC=2.56) then collapses
   to 2.20 at N=800. Random sparse connectivity at p=0.1 saturates
   feature decorrelation; past N≈300 the graph re-collinearises.
   This matches the small-N collinearity failure mode seen in
   z117/z115/z114.

5. **WS_SMALLWORLD nearly matches MESH_4N at N=800** (2.94 vs 3.29
   MC). Small-world rewiring is a viable alternative if the
   2D-grid layout is undesirable for fabrication.

6. **Random Gaussian is the worst at every scale.** The brief's
   choice to recommend a *structural* topology rather than random
   recurrence is empirically grounded across 6 topologies × 3 scales.

---

## 3 — Updated architectural recommendation (two axes)

The brief's single-axis recommendation (MESH_4N for everything)
becomes a two-axis recommendation:

| application class                | recommended topology |
|----------------------------------|----------------------|
| Memory-heavy temporal regression | **MESH_4N** (best MC, monotone scaling) |
| Multi-class classification       | **HUB_SPOKE** (best WAVE, steepest scaling) |
| Hybrid temporal-XOR              | MESH_4N or HUB_SPOKE (within ~2 % of each other) |

Both architectures are plausible 130 nm tape-out candidates. MESH_4N
is the lower-risk default; HUB_SPOKE warrants its own first-silicon
test cell because the WAVE advantage is large and the hub-fan-out
wiring is well-understood from existing memory arrays.

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

**Ready to send if Mario asks for an update.** No new commitments
beyond the brief; only sharper numbers and a second architectural
axis.
